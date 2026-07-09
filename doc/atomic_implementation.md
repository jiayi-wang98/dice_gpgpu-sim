# Atomic operations in dice_gpgpu-sim

This document describes how the DICE simulator implements two distinct atomic
flavors:

1. **`atom.global.*`** — global-memory atomics, modeled as an *"atom-as-load"*
   transaction that bypasses L1D, traverses the NoC, performs the RMW at L2
   pop, and returns through the response FIFO to the writeback path.
2. **`atom.shared.*`** — shared-memory atomics, modeled as the **DICE
   accumulator-PE intrinsic**: the RMW fires *synchronously* at
   functional-execution time on the local SM, charged at PIPE_A energy, and
   never touches the LDST queue, NoC, or L2.

Both share the same PTX-level decode (`atom_impl`) and the same callback
machinery (`atom_callback`); they diverge at the point where the dispatcher
decides whether to defer the callback (global) or fire it immediately
(shared).

The benchmark `cuda/reduce_large_acc/reduce_large_acc.cu` exercises both in a
single kernel — `dice_cta_acc_add` (atom.shared) for the per-CTA reduction and
`atomicAdd(out, ...)` (atom.global) for the cross-CTA publish. We use it as
the running example below.

---

## 1. PTX decode — shared by both paths

The shared front-end is in `src/cuda-sim/instructions.cc`.

### 1.1 `atom_impl` — captures the address, defers the RMW

```cpp
// src/cuda-sim/instructions.cc:1472
void atom_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
    // ...
    thread->m_last_effective_address = effective_address_final;
    thread->m_last_memory_space      = space;
    thread->m_last_dram_callback.function    = atom_callback;
    thread->m_last_dram_callback.instruction = pI;
}
```

`atom_impl` does **not** perform the read-modify-write. It only:

- resolves the effective address (handling `generic→global` or
  `generic→shared` conversion if needed),
- decides which `memory_space_t` the atomic targets, and
- stashes a `dram_callback_t` on the thread that points to `atom_callback`
  plus the PTX instruction.

The actual RMW happens later — either synchronously a few lines below in
`dice_exec_inst_light` (shared path), or at L2-pop time when `mem_fetch::
do_atomic()` runs the callback (global path).

### 1.2 `atom_callback` — the RMW kernel

```cpp
// src/cuda-sim/instructions.cc:1146
void atom_callback(const inst_t *inst, ptx_thread_info *thread) {
    // 1) decode the operand types and the atomic operation kind
    //    (ADD / MAX / MIN / CAS / EXCH / AND / OR / XOR / INC / DEC)
    // 2) pick the memory_space_t (global or shared) — generic addresses are
    //    routed via whichspace(eaddr).
    // 3) read the old value from memory:
    //        mem->read(effective_address, size/8, &data.s64);
    // 4) write the old value to the destination register `d`:
    //        thread->set_operand_value(dst, data, ...);
    // 5) compute the new value via the switch(m_atomic_spec):
    //        op_result = f(data, src2_data[, src3_data for CAS])
    // 6) write back to memory:
    //        mem->write(effective_address, size/8, &op_result, ...);
}
```

`atom_callback` is type-aware: each ATOMIC_* op has a `switch(to_type)` block
that dispatches on `B32/S32/U32/F32/U64/...`. For example `ATOMIC_ADD`
supports `U32/S32/U64/F32`; `ATOMIC_CAS` supports `B32/B64/S32` only.

When `atom_callback` fires, it uses `thread->m_shared_mem` (for shared space)
or `thread->get_global_memory()` (for global space). Both are
`memory_space_impl<>` instances; the `read`/`write` calls are byte-granular,
synchronous in functional simulation.

### 1.3 Dispatch-time split — `dice_exec_inst_light`

After `atom_impl` runs via `OP_DEF` in the opcode switch, control returns to
`dice_exec_inst_light`, which inspects the captured space and dispatches:

```cpp
// src/cuda-sim/cuda-sim.cc:2319
if (pI->get_opcode() == ATOM_OP && !skip) {
    assert(m_last_dram_callback.function != NULL);

    if (insn_space.get_type() == shared_space) {
        // ── Shared path: fire synchronously, charge PIPE_A ─────────────────
        m_last_dram_callback.function(m_last_dram_callback.instruction, this);
        m_last_dram_callback.function = NULL;
        extern unsigned long long g_dice_acc_op_count;
        g_dice_acc_op_count++;
    } else {
        // ── Global path: defer; register callback on the cfg_block ────────
        CFGBlock->add_callback(tid,
                               m_last_dram_callback.function,
                               m_last_dram_callback.instruction,
                               this,
                               true /*atomic*/);
        m_last_dram_callback.function = NULL;
    }
}
```

This is the **single fork in the simulator** that distinguishes the two
flavors. Everything downstream is determined by which branch runs.

---

## 2. `atom.global.*` — Design A: atom-as-load

The global path treats an atomic as a load-shaped transaction that bypasses
L1D entirely, performs the RMW at L2 pop, and writes the destination register
through the normal load-writeback path. This is the "Design A" decision
captured in tasks #29–#33 and #38; the alternative "atom-at-LDST" model is no
longer used.

### 2.1 Pipeline overview

```
dispatch (atom_impl + add_callback)         ⟵ functional decode + register callback on cfg_block
            │
            ▼
DICE-ILP IR — dest reg appears in LD_DEST_REGS (atom counts as a load)
            │
            ▼
mem_access generated with access.set_atomic(true)
            │
            ▼
temporal coalescer — atomic flag carried in dice_transaction_info::is_atomic
   (cross-tid same-line atomics merge into one mf, mirroring SIMT's
    memory_coalescing_arch_atomic per-warp granularity)
            │
            ▼
ldst_unit::process_memory_access_queue_l1cache_cgra:
   if (access.is_atomic()) {            ◀── L1D bypass
      mf = alloc_cgra(access, ...);
      icnt->push(mf);                   ◀── direct to NoC
      m_dice_mem_request_queue->pop_request(port);
   }
            │
            ▼
NoC + L2 (memory_sub_partition)
            │
            ▼
memory_sub_partition::pop():
   if (mf && mf->isatomic()) mf->do_atomic();   ◀── RMW happens here
            │
            ▼
response_fifo at the originating cgra_core
            │
            ▼
cgra_core::cycle (response stage):
   if (mf->isatomic()) {
     m_next_global_cgra = mf;          ◀── route to writeback like a regular load miss
     m_response_fifo.pop_front();
   }
            │
            ▼
writeback_cgra:
   - writes dst register from mf
   - releases scoreboard reservation
   - increments number_of_loads_done (atom counts uniformly as a load)
```

### 2.2 Why bypass L1D? (the "Design A" choice)

In real hardware, the L1D-write-allocate / write-back behavior of a per-SM
cache is inconsistent with strict atomic ordering across CTAs. NVIDIA routes
SM-level atomics directly to the L2 atomic unit. DICE mirrors this: any
`mem_access_t` with `is_atomic == true` skips L1D allocation and is pushed
straight to the NoC.

The relevant L1D-bypass logic lives in
`src/gpgpu-sim/cgra_core.cc:2767` inside
`ldst_unit::process_memory_access_queue_l1cache_cgra`:

```cpp
// src/gpgpu-sim/cgra_core.cc:2767
if (access.is_atomic()) {
    unsigned ctrl_size = access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
    unsigned size      = access.get_size() + ctrl_size;
    if (m_icnt->full(size, true)) return ICNT_RC_FAIL;

    mem_fetch *mf = m_mf_allocator->alloc_cgra(
        cgra_block, access,
        m_cgra_core->get_gpu()->gpu_sim_cycle +
            m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
    m_icnt->push(mf);
    m_dice_mem_request_queue->pop_request(mf->get_ldst_port_num());
    return NO_RC_FAIL;
}
// non-atomic accesses fall through to the L1D latency queue …
```

### 2.3 Atomic flag through the temporal coalescer

A subtle point: DICE's temporal coalescer merges multiple same-line accesses
across dispatched threads into a single `mem_fetch`. Without an atomic-aware
merge predicate, an atomic and a regular load (or two atomics with different
ops) could be merged incorrectly. The coalescer therefore carries
`is_atomic` as part of its merge key:

```cpp
// src/gpgpu-sim/cgra_core.cc:3348
new_info.is_atomic = access.is_atomic();
// ...
if ((m_coalescing_transaction_info_buffer[port].block_addr  == new_info.block_addr) &&
    (m_coalescing_transaction_info_buffer[port].access_type == new_info.access_type) &&
    (m_coalescing_transaction_info_buffer[port].space       == new_info.space) &&
    (m_coalescing_transaction_info_buffer[port].ld_dest_regs == new_info.ld_dest_regs) &&
    (m_coalescing_transaction_info_buffer[port].is_atomic    == new_info.is_atomic) &&   // ◀── must match
    (m_coalescing_transaction_info_buffer[port].block        == cgra_block)) {
    // merge
}
```

When the coalescer emits the merged transaction it sets
`access.set_atomic(info.is_atomic)` so the L1D-bypass predicate above sees the
flag.

### 2.4 The deferred RMW at L2 pop

When the atomic `mem_fetch` reaches the L2, `memory_sub_partition::pop()`
fires `mf->do_atomic()`:

```cpp
// src/gpgpu-sim/l2cache.cc:816
mem_fetch *memory_sub_partition::pop() {
    mem_fetch *mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if (mf && mf->isatomic()) mf->do_atomic();    // ◀── RMW fires here
    // ...
    return mf;
}
```

`mem_fetch::do_atomic()` (line 150 of `mem_fetch.cc`) routes through the
DICE-specific deferred callback path on the cfg_block:

```cpp
// src/gpgpu-sim/mem_fetch.cc:150
void mem_fetch::do_atomic() {
    if (!m_inst.empty()) {                     // SIMT path (warp_inst_t carries callbacks)
        m_inst.do_atomic(m_access.get_warp_mask());
        return;
    }
    // DICE: per-tid callbacks live on the cfg_block, keyed by the tid set
    // attached to the (post-coalesce) mem_access_t.
    if (m_cgra_block) m_cgra_block->do_atomic_dice(m_access.get_tids());
}
```

`cgra_block_state_t::do_atomic_dice` walks the per-tid callback array stored
on the `dice_cfg_block_t` (set up by `add_callback` at dispatch time) and
invokes `atom_callback(inst, thread)` for each tid in the set:

```cpp
// src/cuda-sim/dice_metadata.h:287
void do_atomic(const std::set<unsigned> &tids) {
    if (!m_should_do_atomic) return;
    for (auto tid : tids) {
        dram_callback_t &cb = m_per_scalar_thread[tid].callback;
        if (cb.function && cb.thread) {
            cb.function(cb.instruction, cb.thread);   // ◀── atom_callback fires
            cb.function = NULL;
            // ...
        }
    }
}
```

Each `atom_callback` invocation does the byte-granular RMW on the shared
global `memory_space_impl` instance. Because the simulator runs L2 pops
serially per cycle, the per-tid callbacks within a single coalesced mf fire
in deterministic order; cross-tid same-line atomics produce the same answer
they would on real hardware where the L2 atomic unit serializes RMWs.

### 2.5 Return through the response FIFO

After L2 pop, the atomic mf reaches the originating CGRA-core's
`m_response_fifo`. The cycle stage at line 2596 of `cgra_core.cc` checks the
atomic flag and routes the mf to `m_next_global_cgra` so the regular
load-writeback path can finish the job:

```cpp
// src/gpgpu-sim/cgra_core.cc:2596
if (mf->isatomic()) {
    if (m_next_global_cgra == NULL) {
        mf->set_status(IN_SHADER_FETCHED, /*now*/);
        m_response_fifo.pop_front();
        m_next_global_cgra = mf;            // ◀── into the regular global writeback slot
    }
    return;
}
```

From this point the writeback machinery is **the same code path used for
ordinary global-load misses**:

- `writeback_cgra_ldst` writes the OLD value (captured by `atom_callback` at
  L2 pop) into the dst register listed in the metadata's `LD_DEST_REGS`.
- The scoreboard reservation for that register is released.
- `cgra_block_state_t::inc_number_of_loads_done()` advances; the cfg_block's
  load-completion counter expects atoms uniformly with loads, so the
  DICE-ILP back end emits the atom's dst reg into `LD_DEST_REGS` for this to
  match (task #33).

### 2.6 Energy accounting (atom.global)

A global atomic is accounted exactly like the load-shaped transaction it
models: L1D access is **not** charged (it bypasses), L2/NoC/MC/DRAM are
charged via the normal `NOC_A`, `L2CP`, `MCP`, `DRAMP` counters. There is no
dedicated atom-energy term for the global path — the cost lives in the
transaction's hops.

---

## 3. `atom.shared.*` — the DICE accumulator-PE intrinsic

The shared path is the *behavioral model of the in-fabric-feedback PE*
(formerly "state-PE"). It does not enter the LDST queue, does not generate a
`mem_fetch`, does not touch the NoC/L2/DRAM, and incurs **no** SHRD bank
energy. Instead, the RMW fires synchronously during functional execution and
is charged at PIPE_A — one pipeline-add's worth of energy — modeling a single
PE-cycle stateful accumulator running in CTA-dispatch order.

### 3.1 The synchronous fire

Quoting the shared-space branch from `dice_exec_inst_light` again:

```cpp
// src/cuda-sim/cuda-sim.cc:2332
if (insn_space.get_type() == shared_space) {
    // DICE accumulator-PE model: atom.shared executes on the local SM
    // (no NoC, no L2 atomic unit). Fire the RMW + dst RF write
    // synchronously — this is the behavioral analog of one PE-cycle
    // stateful accumulator running in CTA-dispatch order. The atom
    // does NOT enter the LDST queue (no global mem traffic).
    m_last_dram_callback.function(m_last_dram_callback.instruction, this);
    m_last_dram_callback.function = NULL;

    // DICEwattch: charge as accumulator-PE op (PIPE_A energy), not as
    // a SMEM bank atomic. The counter is summed across SMs at report
    // time. Atom.shared in DICE is, by design, the acc-PE intrinsic.
    extern unsigned long long g_dice_acc_op_count;
    g_dice_acc_op_count++;
}
```

Two observations:

1. **Synchronous execution.** `atom_callback` runs in the same call as the
   PTX-level decode. The slot is read, modified, and written before the
   simulator returns from this thread's instruction step. Subsequent
   threads dispatched in CTA-order observe the updated slot value
   immediately. This is exactly what an in-fabric-feedback PE provides in
   real hardware: a single-PE-cycle accumulator with a feedback wire that
   threads through dispatched threads in CTA order.

2. **No LDST traffic.** The mem-access generator never emits an
   `atom.shared` into the per-port queues. There is no coalescing, no
   `mem_fetch`, no NoC packet, no L2 visit, no response FIFO. The
   "memory" the callback touches is `thread->m_shared_mem`, which is a
   local per-CTA `memory_space_impl<16K>` initialized at CTA launch.

### 3.2 No LDST-queue counting

Because `atom.shared` is dispatch-time synchronous, the LDST-queue depth and
the cfg_block's `get_num_loads()` should NOT account for it. This is enforced
in the "skip" branch of `dice_exec_inst_light`:

```cpp
// src/cuda-sim/cuda-sim.cc:2197
else if (pI->has_memory_read() || pI->get_opcode() == ATOM_OP) {
    // atom.shared (DICE accumulator-PE model) is also in-fabric on
    // the local SM and not enqueued in LDST. Treat both the same way:
    // do NOT decrement loads when predicated off.
    // Global atom under Design A: dst is in LD_DEST_REGS and counts
    // toward get_num_loads(), so skipped atoms must dec_loads (mirror
    // of regular ld).
    const bool is_smem_atom =
        (pI->get_opcode() == ATOM_OP) &&
        (pI->get_space().get_type() == shared_space);
    if (pI->get_space().get_type() != srf_space && !is_smem_atom) {
        CFGBlock->dec_loads();
    }
}
```

The DICE-ILP front end mirrors this: `atom.shared` ops do **not** appear in
the cfg_block's `LD_DEST_REGS` (since they aren't dispatched through the
LDST unit), while `atom.global` ops **do** (since they go through the
load-writeback path described in §2.5).

### 3.3 Energy: `g_dice_acc_op_count` → PIPE_A

A single global counter `g_dice_acc_op_count` (defined in
`src/cuda-sim/cuda-sim.cc:61`) is incremented on every successful
`atom.shared` fire. DICEwattch reads it at kernel-report time:

```cpp
// src/gpgpu-sim/power_stat.h:230
extern unsigned long long g_dice_acc_op_count;
return g_dice_acc_op_count;
```

The per-op energy is charged at the **PIPE_A coefficient** (the per-cycle
pipeline-add energy), exposed in the DICEwattch params as
`m_dice_params.dice_acc_op_e`:

```cpp
// src/gpuwattch/gpgpu_sim_wrapper.cc:215
m_dice_params.s.PIPE_A      = 0.0257;        // nJ
m_dice_params.dice_acc_op_e = 0.0257;        // defaults to PIPE_A
// ...
energy_per_kernel +=
    counters.acc_ops * dice_params.dice_acc_op_e;  // (line 461)
```

DICEwattch reports both the raw op count and the resulting energy:

```
dice_acc_ops = 3.14573e+06, e_acc_ops_nJ = 80845.2
```

(from a typical softmax kernel run — 3.15 M acc ops × 25.7 pJ/op ≈ 80.8 µJ.)

### 3.4 Why not just SHRD_ACC?

A natural alternative would be to count `atom.shared` as a normal SMEM
access and charge SHRD_ACC (~1.3 nJ in the RTX 2060S AccelWattch
calibration). We deliberately do **not** do this because the architectural
claim of the IFF mechanism is precisely that it **replaces** the SMEM bank
access with a single in-fabric PE op. Charging SHRD_ACC would be charging
for hardware the IFF design eliminates. Per-op energy at PIPE_A is ≈50×
cheaper than SHRD_ACC; that ratio is the dominant per-op term in the IFF
energy reduction reported in Table V of the paper.

The DICEwattch report explicitly separates the IFF op cost from any
residual SHRD_ACC term so the two contributions are auditable per-kernel.

---

## 4. Worked example: `reduce_large_acc.cu`

The benchmark `cuda/reduce_large_acc/reduce_large_acc.cu` is a clean
single-kernel demonstration that exercises **both** atomic flavors:

```cuda
#include "../dice_test/dice_atomics.h"

__global__ void reduce_large_acc_kernel(const int *in, int *out, int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (tid == 0) __dice_acc_slots[0] = 0;
    __syncthreads();

    int v = (gid < N) ? in[gid] : 0;

    // ── atom.shared (DICE acc op) — per-CTA reduce via IFF ─────────────
    int prefix = 0;
    dice_cta_acc_add(prefix, v);                      // (A) below

    __syncthreads();
    if (tid == 0)
        atomicAdd(out, (int)__dice_acc_slots[0]);    // (B) below
}
```

Two atomic call sites; two completely different paths through the simulator.

### (A) `dice_cta_acc_add(prefix, v)` — per-CTA reduction

This macro expands to:

```cuda
dice_cta_acc_add_slot<__COUNTER__ % 32>(prefix, v);
```

which lowers to:

```cuda
atomicAdd((int *)&__dice_acc_slots[SLOT], v);
```

In PTX this is `atom.shared.add.s32 [&__dice_acc_slots[SLOT]], v`. The DICE
flow through the simulator:

1. `atom_impl` resolves the SMEM address and stashes `atom_callback` on
   the thread.
2. `dice_exec_inst_light` sees `shared_space` and **fires
   `atom_callback` synchronously** — the slot is read, added to `v`, and
   written back in one functional step.
3. `g_dice_acc_op_count++` (charged at PIPE_A by DICEwattch).
4. No `mem_fetch` is generated. The LDST queue is untouched. No NoC
   packet, no L2 visit, no response FIFO.
5. The next thread in CTA-dispatch order sees the updated slot value
   when *its* `atom_impl` runs.

After `__syncthreads`, slot 0 holds the CTA-wide sum (because each of the
256 threads accumulated its value into it in dispatch order). The
write-back of `prefix` happens because `atom_callback` writes the OLD
slot value to the dst register; this gives each thread its
exclusive-prefix-sum-so-far for free.

### (B) `atomicAdd(out, ...)` — cross-CTA publish

This is a textbook global atomic. The flow:

1. `atom_impl` resolves the global address and stashes the callback.
2. `dice_exec_inst_light` sees `global_space` and **defers**: it calls
   `CFGBlock->add_callback(tid, atom_callback, ...)` to register the
   callback on the per-tid array of the cfg_block.
3. The mem-access generator emits a `mem_access_t` for this atom with
   `access.set_atomic(true)`.
4. The temporal coalescer attempts merges — but since this branch is
   guarded by `if (tid == 0)`, only one tid emits the access per CTA, so
   the coalescer just passes it through.
5. `process_memory_access_queue_l1cache_cgra` sees `access.is_atomic()`,
   **bypasses L1D**, allocates a `mem_fetch` with `is_atomic = true`,
   pushes to the NoC, and pops the request from the LDST queue.
6. NoC → L2. `memory_sub_partition::pop()` calls `mf->do_atomic()`,
   which routes through `m_cgra_block->do_atomic_dice(tids)` →
   `dice_cfg_block_t::do_atomic(tids)` → the registered `atom_callback`
   fires. The RMW happens here, modifying `*out` in the global
   `memory_space_impl`.
7. The mf returns through the response FIFO. The cycle stage at line
   2596 of `cgra_core.cc` routes it into `m_next_global_cgra`.
8. `writeback_cgra_ldst` writes the OLD value of `*out` (captured by
   `atom_callback` at step 6) into the dst register slot listed in
   `LD_DEST_REGS`, releases the scoreboard, and increments
   `number_of_loads_done`.

If 256 CTAs participate, this happens 256 times — one per CTA. The 256
atoms from different CTAs are serialized at the L2 atomic unit, exactly as
on real hardware. The OLD value returned to each `tid==0` writer is unused
in this kernel (no destination register), so the writeback is a no-op for
the user but still increments the cfg_block's load-completion counter.

### Side-by-side flow

```
                    atom.shared (acc-PE)        atom.global (Design A)
                    ─────────────────────       ──────────────────────
Decode              atom_impl                   atom_impl
Defer / immediate   IMMEDIATE                   defer (add_callback)
LDST queue          (skipped)                   enqueue
Coalescer           (skipped)                   merge by (line, atomic, ...)
L1D                 (skipped)                   BYPASS
NoC                 (skipped)                   traverse
L2 RMW              (skipped)                   mem_sub_partition::pop()
                                                 → mf->do_atomic()
                                                 → cfg_block->do_atomic_dice
                                                 → atom_callback per tid
Response FIFO       (n/a)                       returns
Writeback           (n/a — slot in SMEM)        writeback_cgra_ldst
                                                 (dst reg + scoreboard)
Energy term         g_dice_acc_op_count          NOC_A + L2CP + (MCP/DRAMP)
                    × PIPE_A                     (no L1D charge — bypassed)
```

---

## 5. Quick file/function index

| Concern | File | Symbol |
|---|---|---|
| PTX decode of `atom.*` | `src/cuda-sim/instructions.cc` | `atom_impl` (line 1472) |
| RMW kernel | `src/cuda-sim/instructions.cc` | `atom_callback` (line 1146) |
| Dispatch-time fork (shared vs global) | `src/cuda-sim/cuda-sim.cc` | `dice_exec_inst_light` (line 2319) |
| acc-op counter | `src/cuda-sim/cuda-sim.cc` | `g_dice_acc_op_count` (line 61) |
| Defer callback to cfg_block | `src/cuda-sim/dice_metadata.h` | `add_callback` (line 265) |
| Per-tid callback fire | `src/cuda-sim/dice_metadata.h` | `do_atomic(tids)` (line 287) |
| Atomic flag through coalescer | `src/gpgpu-sim/cgra_core.cc` | `dice_transaction_info::is_atomic` (line 3348) |
| L1D bypass (atom-as-load) | `src/gpgpu-sim/cgra_core.cc` | `process_memory_access_queue_l1cache_cgra` (line 2767) |
| Response-FIFO routing for atomic mf | `src/gpgpu-sim/cgra_core.cc` | line 2596 (`if (mf->isatomic())`) |
| L2 RMW fire | `src/gpgpu-sim/l2cache.cc` | `memory_sub_partition::pop` (line 819) |
| mem_fetch dispatch to per-tid callbacks | `src/gpgpu-sim/mem_fetch.cc` | `mem_fetch::do_atomic` (line 150) |
| DICEwattch acc-op energy | `src/gpuwattch/gpgpu_sim_wrapper.cc` | `dice_acc_op_e` (line 218), `acc_ops × dice_acc_op_e` (line 461) |
| DICEwattch report line | `src/gpuwattch/gpgpu_sim_wrapper.cc` | `"dice_acc_ops = ..."` (line 589) |
