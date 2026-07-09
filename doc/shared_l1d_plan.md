# Sharing the L1D across the 4 CPs of a CGRA cluster

## Background — why this matters

DICE decomposes one "SM-equivalent" into a **CGRA cluster** containing
`n_cores_per_cluster` (default = 4) **CPs / sub-cores**, each running an
independent 4×5 CGRA fabric. In gpgpu-sim terms, each CP is a separate
`shader_core_ctx` (subclassed as `cgra_core_ctx`), and each one owns its
own `ldst_unit`, which in turn owns its own `m_L1D`.

In contrast, a real NVIDIA SM (Turing, Ampere, Hopper) has **four warp
schedulers / sub-cores sharing a single L1D + SMEM** (unified storage,
typically 128–256 KB, banked into `gpgpu_l1_banks` banks). The four
schedulers issue LD/ST requests concurrently, the L1D arbitrates them on
its banks, and *cache lines fetched by one sub-core are immediately
visible to the other three*.

The architectural mismatch this creates in DICE today:

| Aspect | Real A100 SM | DICE cluster (current) |
|---|---|---|
| L1D instances per SM/cluster | **1** | **4** (one per CP) |
| L1D capacity per SM/cluster | 192 KB unified | `4 × per-CP L1` (logically same total, but **partitioned**) |
| Cross-sub-core line sharing | yes, implicit | **no** — a line fetched by CP-0 is invisible to CP-1/2/3 |
| Concurrent LD/ST throughput | 4 sub-cores × per-bank ports through one L1 | 4 sub-cores × per-bank ports through 4 L1s |
| Bank conflict model | one shared `l1_banks` array | per-CP `l1_banks` array (decoupled) |

For workloads where the four CPs of a cluster touch the **same** global
addresses (CTAs from the same kernel sharing producer-consumer data
through L1D, or texture-like reuse), the per-CP-private model under-counts
hit-rate vs the real GPU. For the IFF reduction/scan benchmarks the
absolute effect is small (most traffic is `atom.shared` which doesn't
touch L1D, or coalesced GMEM loads), but for the **paper's apples-to-apples
A100 comparison this is a real systematic difference** between DICE and
the GPU baseline.

## What this branch will change

Move L1D ownership from `ldst_unit` (per-CP) to a new per-cluster owner,
preserve per-CP `ldst_unit` request paths (each CP still has its own
queue, coalescer, mshr fan-out), but route every L1D hit/miss request
through the **one shared L1D** with `gpgpu_l1_banks` ports of concurrency
across all four CPs of the cluster.

The CGRA fabric, RF, SMEM, dispatcher, accumulator-PE infrastructure,
etc., are unchanged. The only change is the address of `m_L1D` in
memory and the arbitration of L1 ports.

## Code locations affected

Everything is in `src/gpgpu-sim/`. Concretely:

| File / line | What it does today | What it needs to do |
|---|---|---|
| `shader.h:1416` (`l1_cache *m_L1D;` inside `ldst_unit`) | Each `ldst_unit` owns one L1D | Replace with `l1_cache *m_L1D` *reference* set by the cluster at construction time |
| `shader.cc:2548` (`ldst_unit::ldst_unit(... cgra_core_ctx*, ...)` constructor) | Calls `new l1_cache(...)` with a per-CP name | Don't allocate; receive the shared L1D as a constructor arg |
| `cgra_core.cc:554` (`cgra_simt_core_cluster::create_cgra_core_ctx()`) | Loops `m_cgra_core[i] = new exec_cgra_core_ctx(...)` for `n_cores_per_cluster` CPs, each constructing its own L1D internally | Allocate **one** `l1_cache` keyed by cluster id BEFORE the loop, then pass that pointer into each `cgra_core_ctx` (which forwards it into the per-CP `ldst_unit`) |
| `shader.h:1454` (`std::vector<std::vector<mem_fetch*>> l1_latency_queue;`) | Per-CP `[bank][stage]` queue feeding L1D | Stays per-CP for the latency stages, but the **bank dispatch arbiter** needs to coordinate across the 4 CPs because they target one banked L1 |
| `cgra_core.cc:2548` (`const unsigned n_l1_banks = m_config->m_L1D_config.l1_banks;`) | Used in `L1_latency_queue_cycle_cgra` per ldst_unit | The cluster needs a per-cycle bank-arbiter that picks ≤ `l1_banks` requests across the 4 CPs and serves them on the one L1D |
| `cgra_core.cc:2767` (`process_memory_access_queue_l1cache_cgra`) | Per-CP path that allocates a `mem_fetch` and pushes to `m_icnt` (atomic bypass) OR enters its own `l1_latency_queue` | Atomic bypass stays per-CP (icnt is a flat node space). Non-atomic L1 path routes into the shared L1's bank arbiter. |
| `cgra_core.cc:2828` (`L1_latency_queue_cycle_cgra`) | Per-CP cycle stepper for the per-bank latency queue | Move the cycling of the **shared** banks to the cluster cycle; per-CP only maintains its launch queue |
| `shader.cc:2531` (regular SIMT `ldst_unit::ldst_unit(... shader_core_ctx*, ...)` constructor that also calls `new l1_cache`) | The SIMT-mode constructor — used by GPU baseline runs | **Do not touch.** This branch must not break the existing SIMT mode (used for the GPU SIMT baselines in Table V). |
| `cgra_core.h:69` (`delete m_ldst_unit;` in `~cgra_core_ctx()`) | OK — `ldst_unit` no longer owns L1, so destructor stays correct as long as we null `ldst_unit::m_L1D` instead of `delete`-ing it | Add a `bool m_owns_L1D` flag (or just don't `delete` in the ldst_unit dtor; the cluster owns and frees it) |

## Configuration knob

Add one knob to gate the new behaviour so we can A/B test on the same
branch:

```text
# Shared-L1D mode for DICE clusters. When 1, the 4 CPs in each CGRA
# cluster share a single L1D of size `gpgpu_cache:dl1`. When 0 (default,
# legacy DICE behaviour), each CP has its own L1D.
-dice_shared_l1d 1
```

`shader.h` config parser registers it; `cgra_simt_core_cluster::create_cgra_core_ctx` branches on it.

When `dice_shared_l1d=1`, the **per-CP** L1D size in the config string
(`gpgpu_cache:dl1`) is interpreted as the **per-cluster** L1D size — i.e.
the user is now setting "the SM's L1D", not "the CP's L1D". For the A100
config we already wrote, `S:3:128:128 = 48 KB` per CP would need to become
`S:12:128:128 = 192 KB` for the cluster (matching A100 per-SM L1).

We'll document this clearly in `gpgpusim_dice_a100.config` and probably
ship two variants:
- `gpgpusim_dice_a100_private_l1.config` — 48 KB per CP × 4 = 192 KB
  partitioned (the current "DICE legacy" behaviour)
- `gpgpusim_dice_a100_shared_l1.config` — 192 KB unified shared (the new
  mode, A100-faithful)

## Bank-port arbitration semantics

A100's L1D is 4-banked (`gpgpu_l1_banks 4`). With the shared model, the
arbiter at the cluster cycle picks up to 4 requests **per cycle, across
all 4 CPs**, with each request mapped to a bank by the existing
`m_L1D_config.set_bank(addr)` function. Conflicts:

- **Same bank, different CPs**: serialise — one wins per cycle, others
  stall in their per-CP queue.
- **Same bank, same CP**: behaves as today (the per-CP latency queue
  drains one bank slot per cycle).
- **Different banks across CPs**: all proceed in parallel.

A simple **round-robin per cycle** arbiter is the first implementation;
LRR or oldest-first refinements are follow-ups.

The arbiter lives in `cgra_simt_core_cluster::cycle()` (or a new
`cluster_l1d_cycle()` called there) — it iterates the 4 CPs in
round-robin starting from a stride, draining each CP's
`l1_latency_queue` heads into the shared L1D.

## Power-model implications

DICEwattch currently counts L1D accesses via per-shader counters; with
one shared L1D, those counters must aggregate at the cluster level. The
energy-per-access calibration doesn't change (it's an L1 access, banked
4-way, same as A100's). The only book-keeping fix is making sure we
don't double-count when 4 CPs hit the same line — at that point the
"4 hits on the same line" really is 4 hit accesses on real hardware
(each scheduler still has to read its banked sub-block), so it's fine to
count them all.

For per-line **MSHR coalescing** (multiple in-flight misses to the same
line collapse into one), the existing L1 mshr logic already does this
correctly inside one `l1_cache`. Moving from 4 L1s to 1 means MSHR
coalescing now spans the 4 CPs, which is closer to real GPU behaviour
and should reduce miss traffic — that's the win.

## Implementation phases

1. **Phase A — wiring (no behaviour change with shared_l1d=0).**
   - Add `dice_shared_l1d` config knob, default 0.
   - Add a `bool m_owns_l1d` flag to `ldst_unit`; constructor takes an
     optional `l1_cache *shared_l1d`. If non-null, store it as
     `m_L1D = shared_l1d` and set `m_owns_l1d = false`.
   - Cluster (in `cgra_core.cc:554` loop) allocates one shared L1D when
     `dice_shared_l1d=1`, passes the pointer into every CP's
     `ldst_unit`. When `=0`, behaves as today.
   - **Build, run the existing test suite at `dice_shared_l1d=0`** —
     numbers must be identical to the current branch (no regression).

2. **Phase B — bank arbitration.**
   - Move `L1_latency_queue_cycle_cgra` from per-`ldst_unit` to a
     cluster-level helper.
   - Implement round-robin arbiter across the 4 CPs' head requests,
     respecting `l1_banks` parallelism.
   - **Validation**: turn `dice_shared_l1d=1` on and re-run the same
     tests; numbers should differ but kernels should still produce
     correct results. Sanity-check: hit rate of a producer-consumer
     micro-benchmark (e.g., two CPs reading the same line) should
     improve under shared mode.

3. **Phase C — bookkeeping cleanup.**
   - Aggregate per-CP L1 stats into one per-cluster stat when shared.
   - Update `accelwattch_dice_sim.xml` access-counter aggregation if
     needed.

4. **Phase D — A100 config update.**
   - Ship `gpgpusim_dice_a100_shared_l1.config` with `dice_shared_l1d=1`
     and a per-cluster `dl1` size of 192 KB (4× the per-CP variant).
   - Re-run the Table V state-PE benchmarks under the shared-L1 model
     so the paper's A100 numbers are apples-to-apples with the GPU
     A100 baseline.

## Things that explicitly do NOT change

- The CGRA fabric (PE count, switch-box, p-graph dispatch).
- The RF, SMEM, accumulator-PE, in-fabric feedback path.
- `atom.global` and `atom.shared` paths (`atom.global` bypasses L1D,
  `atom.shared` is in-CP synchronous — neither touches the L1 we're
  restructuring).
- Per-CP CTA scheduling, dispatcher, and metadata cache.
- The GPU SIMT (`shader_core_ctx`) path in `shader.cc` — we only touch
  the cgra_core (DICE) constructor of `ldst_unit`.

## Validation plan

After each phase, run the **DICEWattch existing benchmark suite** on
both 2060S and A100 configs, with the new knob off and on, and diff the
cycle counts:

- `dice_shared_l1d=0` → numbers must be **bit-identical** to the
  `DICEWattch` baseline.
- `dice_shared_l1d=1` → numbers will move (more L1 hits per line, fewer
  NoC packets); expected directions are higher hit rate, lower L2/DRAM
  traffic, slightly more L1 bank conflicts.

Targeted micro-benchmarks for the shared-mode validation:

1. **Cross-CP line sharing**: a kernel where 4 CTAs (one per CP) read
   the same input array. Shared-L1 should hit ~3× more than the
   private model (one cold-miss-per-line vs four).
2. **Bank-conflict storm**: a kernel where all 4 CPs hammer
   `same_line_diff_offset` addresses. Shared-L1 should serialise at
   the bank port; private-L1 wouldn't (and would over-count throughput).
3. **No-sharing baseline**: a kernel where each CP touches private
   data. Shared and private should perform identically (modulo arbiter
   overhead).

## Open questions

1. **SMEM ownership**: keep per-CP (SMEM is per-CTA) or also share?
   I think *keep per-CP* — SMEM is per-CTA and a CTA lives on one CP,
   so there's no architectural reason to share. The "shared-L1D" change
   is only about L1, not SMEM.
2. **Atomics**: `atom.shared` is in-CP only (already correct).
   `atom.global` bypasses L1D entirely (Design A) — also unaffected. No
   coordination needed between CPs for atomics in this branch.
3. **Compiler / PPTX impact**: none. The compiler and PPTX format are
   blind to the L1 organisation; this is purely a simulator memory-
   subsystem change.

## Acceptance criteria

- `dice_shared_l1d=0` regression test: identical cycles/energy to the
  `DICEWattch` HEAD on all 13 state-PE benchmarks at the original
  2060S config sizes.
- `dice_shared_l1d=1` on the A100 config: kernels still produce correct
  results; cycle counts shift by ≤ ±20% (hit rate gains roughly offset
  bank-conflict serialisation for our benchmark mix); L1 hit-rate stat
  improves measurably on at least one cross-sharing micro-benchmark.
- A100 IFF benchmarks reported in the paper are re-run under
  `dice_shared_l1d=1` and the table caption is updated to say "DICE
  configured with shared L1 across the 4 CPs of each cluster, matching
  the A100 SM L1 organisation."
