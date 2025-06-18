#include "../abstract_hardware_model.h"
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/memory.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../option_parser.h"
#include "cgra_core.h"
#include "stat-tool.h"
#include "scoreboard.h"
#include "shader_trace.h"

extern int g_debug_execution;

//map from int to string
const char *mem_stage_stall_type_str(enum mem_stage_stall_type type) {
  switch (type) {
    case NO_RC_FAIL:
      return "NO_RC_FAIL";
    case BK_CONF:
      return "BK_CONF";
    case MSHR_RC_FAIL:
      return "MSHR_RC_FAIL";
    case ICNT_RC_FAIL:
      return "ICNT_RC_FAIL";
    case COAL_STALL:
      return "COAL_STALL";
    case TLB_STALL:
      return "TLB_STALL";
    case DATA_PORT_STALL:
      return "DATA_PORT_STALL";
    case WB_ICNT_RC_FAIL:
      return "WB_ICNT_RC_FAIL";
    case WB_CACHE_RSRV_FAIL:
      return "WB_CACHE_RSRV_FAIL";
    default:
      assert(0);
      break;
  }
  return "UNKOWN_MEM_STAGE_STALL_TYPE";
}


cgra_core_ctx::cgra_core_ctx(gpgpu_sim *gpu,simt_core_cluster *cluster,
  unsigned cgra_core_id, unsigned tpc_id,const shader_core_config *config,
  const memory_config *mem_config,shader_core_stats *stats){
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: create_cgra_core_ctx() id=%d\n", cgra_core_id);
    fflush(stdout);
  }
  m_cluster = cluster;
  m_config = config;
  m_memory_config = mem_config;
  m_stats = stats;
  m_max_block_size = config->dice_cgra_core_max_threads;
  m_kernel_block_size = m_max_block_size;
  m_liveThreadCount = 0;
  m_not_completed = 0;
  m_exec_stalled_by_writeback_buffer_full = false;
  m_exec_stalled_by_ldst_unit_queue_full = false;
  m_active_blocks = 0;
  m_gpu = gpu;
  m_kernel = NULL;
  m_cgra_core_id = cgra_core_id;
  m_tpc = tpc_id;

  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
  m_thread = (ptx_thread_info **)calloc(m_max_block_size,sizeof(ptx_thread_info *));
  m_simt_stack.resize(MAX_CTA_PER_SHADER);
  for (unsigned i = 0; i < m_gpu->max_cta_per_core(); i++) {
    initializeSIMTStack(i,m_max_block_size);
    //initializeSIMTStack(i,256);
  }
}


void cgra_core_ctx::initializeSIMTStack(unsigned hw_cta_id, unsigned num_threads){
  m_simt_stack[hw_cta_id] = new simt_stack(m_cgra_core_id, num_threads, m_gpu);
}

void cgra_core_ctx::get_pdom_stack_top_info(unsigned hw_cta_id, unsigned *pc,
  unsigned *rpc) const {
  m_simt_stack[hw_cta_id]->get_pdom_stack_top_info(pc, rpc);
}

void cgra_core_ctx::deleteSIMTStack() {
  for (unsigned i = 0; i < m_simt_stack.size(); i++) {
    if (m_simt_stack[i]) {
      delete m_simt_stack[i];
      m_simt_stack[i] = NULL;
    }
  }
}

void cgra_core_ctx::updateSIMTStack(unsigned hw_cta_id, dice_cfg_block_t *cfg_block) {
  unsigned kernel_cta_size = get_cta_size(hw_cta_id);
  simt_mask_t thread_done(kernel_cta_size);
  addr_vector_t next_pc;
  for (unsigned i = 0; i < kernel_cta_size; i++) {
    unsigned thread_offset = get_cta_start_tid(hw_cta_id);
    if (ptx_thread_done(thread_offset+i)) {
      thread_done.set(i);
      next_pc.push_back((address_type)-1);
    } else {
      if (cfg_block->reconvergence_pc == RECONVERGE_RETURN_PC) {
        cfg_block->reconvergence_pc = get_return_meta_pc(m_thread[thread_offset+i]);
      }
      next_pc.push_back(m_thread[thread_offset+i]->get_meta_pc());
    }
  }
  if(m_cgra_core_id == get_dice_trace_sampling_core()){
    m_simt_stack[hw_cta_id]->update_sid(thread_done, next_pc, cfg_block->reconvergence_pc,
    cfg_block->op, cfg_block->get_metadata()->m_size, cfg_block->get_metadata()->m_PC);
  } else {
    m_simt_stack[hw_cta_id]->update(thread_done, next_pc, cfg_block->reconvergence_pc,
    cfg_block->op, cfg_block->get_metadata()->m_size, cfg_block->get_metadata()->m_PC);
  }
}

bool cgra_core_ctx::ptx_thread_done(unsigned hw_thread_id) const {
  return ((m_thread[hw_thread_id] == NULL) ||
          m_thread[hw_thread_id]->is_done());
}


void cgra_core_ctx::execute_CFGBlock(cgra_block_state_t* cgra_block) {
  dice_cfg_block_t* cfg_block = cgra_block->get_current_cfg_block();
  for (unsigned t = 0; t < m_kernel_block_size; t++) {
    if (cfg_block->active(t)) {
      m_thread[t]->dice_exec_block(cfg_block,t);
      checkExecutionStatusAndUpdate(cgra_block,t);
    }
  }
}

bool cgra_core_ctx::ldst_unit_response_buffer_full() const{
  return m_ldst_unit->response_buffer_full();
}

void cgra_core_ctx::execute_1thread_CFGBlock(cgra_block_state_t* cgra_block, unsigned tid, unsigned lane_id) {
  dice_cfg_block_t* cfg_block = cgra_block->get_current_cfg_block();
  unsigned unrolling_factor = cgra_block->get_unrolling_factor();
  if(cgra_block->is_parameter_load()){
    for (unsigned t = 0; t < m_kernel_block_size; t++) {
      unsigned core_tid=t+get_cta_start_tid(cgra_block->get_cta_id());
      if (cfg_block->active(t)) {
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          printf("DICE-Sim Functional: cycle %d, cgra_core %u executed thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_core_id, t,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
        }
        m_thread[core_tid]->dice_exec_block(cfg_block,t);
        //only generate one memory accesses 
        if(core_tid==tid) {
          std::list<unsigned> masked_ops_reg;
          cfg_block->generate_mem_accesses(t,masked_ops_reg, unrolling_factor, lane_id);
          //push mem_access to ldst_unit's queue
          m_ldst_unit->dice_push_accesses(cfg_block,cgra_block);
        }
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          //cfg_block->print_mem_ops_tid(t);
        }
        //check status and update
        checkExecutionStatusAndUpdate(cgra_block,core_tid);
      }
    }
  } else {
    unsigned cta_tid_offset = get_cta_start_tid(cgra_block->get_cta_id());
    unsigned local_tid = tid - cta_tid_offset;
    if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
      printf("DICE-Sim DEBUG: cycle %d, cgra_core %u executing thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_core_id, local_tid,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
    }
    if (cfg_block->active(local_tid)) {
      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE-Sim Functional: cycle %d, cgra_core %u executed thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_core_id, local_tid,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
      }
      m_thread[tid]->dice_exec_block(cfg_block,local_tid);
      //generate memory accesses to ldst unit
      std::list<unsigned> masked_ops_reg;
      cfg_block->generate_mem_accesses(local_tid,masked_ops_reg, unrolling_factor, lane_id);
      //push mem_access to ldst_unit's queue
      if(g_debug_execution==4 &m_cgra_core_id == get_dice_trace_sampling_core()){
        cfg_block->print_m_accessq();
      }
      m_ldst_unit->dice_push_accesses(cfg_block,cgra_block);
      //release masked scoreboard record
      for (std::list<unsigned>::iterator it = masked_ops_reg.begin(); it != masked_ops_reg.end(); ++it) {
        m_scoreboard->releaseRegisterFromLoad(tid,(*it));
      }
      if(g_debug_execution==4 &m_cgra_core_id == get_dice_trace_sampling_core()){
        cfg_block->print_mem_ops_tid(local_tid);
      }
      //check status and update
      checkExecutionStatusAndUpdate(cgra_block,tid);
    }
  }
}

void cgra_core_ctx::checkExecutionStatusAndUpdate(cgra_block_state_t* cgra_block, unsigned tid) {
  dice_cfg_block_t* cfg_block = cgra_block->get_current_cfg_block();
  //get instructions from the cfg_block
  for(unsigned i=0; i<cfg_block->get_diceblock()->ptx_instructions.size(); i++){
    ptx_instruction *pI = cfg_block->get_diceblock()->ptx_instructions[i];
    if (pI->isatomic()) cgra_block->inc_n_atomic();
    if (pI->space.is_local() && (pI->is_load() || pI->is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = translate_local_memaddr(
        pI->get_addr(tid), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
      pI->data_size, (new_addr_type *)localaddrs);
      pI->set_addr(tid, (new_addr_type *)localaddrs, num_addrs);
    }
  }
  if (ptx_thread_done(tid)) {
    cgra_block->set_completed(tid-get_cta_start_tid(cgra_block->get_cta_id()));
  }
  if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
    m_liveThreadCount--;
  }
}

unsigned cgra_core_ctx::translate_local_memaddr(
  address_type localaddr, unsigned tid, unsigned num_shader,
  unsigned datasize, new_addr_type *translated_addrs) {
// During functional execution, each thread sees its own memory space for
// local memory, but these need to be mapped to a shared address space for
// timing simulation.  We do that mapping here.

address_type thread_base = 0;
unsigned max_concurrent_threads = 0;
//if (m_config->gpgpu_local_mem_map) {
//  // Dnew = D*N + T%nTpC + nTpC*C
//  // N = nTpC*nCpS*nS (max concurent threads)
//  // C = nS*K + S (hw cta number per gpu)
//  // K = T/nTpC   (hw cta number per core)
//  // D = data index
//  // T = thread
//  // nTpC = number of threads per CTA
//  // nCpS = number of CTA per shader
//  //
//  // for a given local memory address threads in a CTA map to contiguous
//  // addresses, then distribute across memory space by CTAs from successive
//  // shader cores first, then by successive CTA in same shader core
//  thread_base =
//      4 * (kernel_padded_threads_per_cta *
//               (m_sid + num_shader * (tid / kernel_padded_threads_per_cta)) +
//           tid % kernel_padded_threads_per_cta);
//  max_concurrent_threads =
//      kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
//} else {
  // legacy mapping that maps the same address in the local memory space of
  // all threads to a single contiguous address region
  thread_base = 4 * (m_config->n_thread_per_shader * m_cgra_core_id + tid);
  max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
//}
assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

// If requested datasize > 4B, split into multiple 4B accesses
// otherwise do one sub-4 byte memory access
unsigned num_accesses = 0;

if (datasize >= 4) {
  // >4B access, split into 4B chunks
  assert(datasize % 4 == 0);  // Must be a multiple of 4B
  num_accesses = datasize / 4;
  assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
  assert(
      localaddr % 4 ==
      0);  // Address must be 4B aligned - required if accessing 4B per
           // request, otherwise access will overflow into next thread's space
  for (unsigned i = 0; i < num_accesses; i++) {
    address_type local_word = localaddr / 4 + i;
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  thread_base + LOCAL_GENERIC_START;
    translated_addrs[i] = linear_address;
  }
} else {
  // Sub-4B access, do only one access
  assert(datasize > 0);
  num_accesses = 1;
  address_type local_word = localaddr / 4;
  address_type local_word_offset = localaddr % 4;
  assert((localaddr + datasize - 1) / 4 ==
         local_word);  // Make sure access doesn't overflow into next 4B chunk
  address_type linear_address = local_word * max_concurrent_threads * 4 +
                                local_word_offset + thread_base +
                                LOCAL_GENERIC_START;
  translated_addrs[0] = linear_address;
}
return num_accesses;
}


void cgra_core_ctx::set_max_cta(const kernel_info_t &kernel){
  // calculate the max cta count and cta size for local memory address mapping
  //DICE-TODO in the future. For now, we assume that each core can only run 1 CTA at a time.
  // this is on going ...
  //In the next gen, a cgra_core can run multiple CTAs at a time and there's a similar "CTA scheduler" as warp scheduler to handle concurrent CTA execution.
  //kernel_max_cta_per_shader = 1;
  //kernel_padded_threads_per_cta = kernel.threads_per_cta();

  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  //align to multiples of 256
  kernel_padded_threads_per_cta = (gpu_cta_size % 256) ? 256 * ((gpu_cta_size / 256) + 1): gpu_cta_size;
}

void cgra_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
  long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_cgra_core_id];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_cgra_core_id];
}

unsigned cgra_core_ctx::get_dice_trace_sampling_core(){
  return m_gpu->get_dice_trace_sampling_core();
}

void cgra_core_ctx::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I->get_stats();          // Get L1I stats
  cs += m_L1B->get_stats();         // Get L1B stats
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

float cgra_core_ctx::get_current_occupancy(unsigned long long &active,
  unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_blocks > 0) {
    total += m_max_block_size;
    active += m_kernel_block_size;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

void cgra_core_ctx::issue_block2core(kernel_info_t &kernel){
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: issue_block2core id=%d\n", m_cgra_core_id);
    fflush(stdout);
  }
  //choose a new CTA to run, reinit core to run the new CTA,
  //init thread infos and simt stacks for the new CTA
  set_max_cta(kernel); //1 for now
  kernel.inc_running();
  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  max_cta_per_core = kernel_max_cta_per_shader; //1 CTA per core for now
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status_table->is_free(i)) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);
  // determine hardware threads that will be used for this CTA
  int cta_size = kernel.threads_per_cta();
  int padded_cta_size = cta_size;
  if (cta_size % 256)
    padded_cta_size = ((cta_size / 256) + 1) * (256);

  unsigned int start_thread, end_thread;
  start_thread = free_cta_hw_id * padded_cta_size; 
  end_thread = start_thread + cta_size; //according to kernel's block_size

  // reset the microarchitecture state of the selected hardware thread contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_cgra_core_id, i, cta_size - (i - start_thread),
        m_config->dice_cgra_core_max_threads, this, free_cta_hw_id, get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
  }
  assert(nthreads_in_block > 0 && nthreads_in_block <=m_config->dice_cgra_core_max_threads);  
  // should be at least one, but less than max

  m_cta_status_table->init_cta(free_cta_hw_id);
  m_cta_status_table->set_num_live_threads(free_cta_hw_id, start_thread, end_thread);

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }

  // initialize the SIMT stacks and fetch hardware
  init_CTA(free_cta_hw_id, start_thread, end_thread, ctaid, padded_cta_size, kernel);
  //DICE-TODO
  //m_active_blocks++;
  printf("DICE Sim uArch: core id: %d, cta:%2u, start_tid:%4u, end_tid:%4u,initialized @(%lld,%lld), m_kernel_block_size=%d\n",
                m_cgra_core_id, ctaid, start_thread, end_thread, m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, m_kernel_block_size);
  fflush(stdout);
}

// return the next pc of a thread
address_type cgra_core_ctx::next_meta_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread->get_meta_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
}

bool cgra_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  return (m_active_blocks < m_config->max_cta(kernel));
  //return (m_active_blocks < 1);//TODO
}

void cgra_core_ctx::init_CTA(unsigned cta_id, unsigned start_thread,
  unsigned end_thread, unsigned ctaid,int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_meta_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned n_active = 0;
    simt_mask_t active_threads(cta_size);
    //set core-level threads state
    for (unsigned tid = start_thread; tid < end_thread; tid++) {
      n_active++;
      assert(!m_active_threads.test(tid));
      m_active_threads.set(tid);
      active_threads.set(tid-start_thread);
    }
    //init simt stack
    m_simt_stack[cta_id]->reset();
    m_simt_stack[cta_id]->resize_warp(cta_size);
    m_simt_stack[cta_id]->launch(start_pc, active_threads);
    if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_0_%d_simt.txt",ctaid);
        unsigned pc, rpc;
        m_simt_stack[cta_id]->resume(fname);
        m_simt_stack[cta_id]->get_pdom_stack_top_info(&pc, &rpc);
        for (unsigned t = 0; t < m_config->dice_cgra_core_max_threads; t++) {
          if (m_thread != NULL) {
            m_thread[t]->set_next_meta_pc(pc);
            m_thread[t]->update_metadata_pc();
          }
        }
      start_pc = pc;
    }
    //m_cgra_block_state[MF_DE]->init(start_pc, cta_id, active_threads);
    m_not_completed += n_active;
    ++m_active_blocks;
  }
}

void exec_simt_core_cluster::create_cgra_core_ctx() {
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: create_cgra_core_ctx() cluster id=%d\n", m_cluster_id);
    fflush(stdout);
  }
  m_core = NULL;
  m_cgra_core = new cgra_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_cgra_core[i] = new exec_cgra_core_ctx(m_gpu, this, sid, m_cluster_id,
                                         m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}


unsigned cgra_core_ctx::sim_init_thread(
  kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
  unsigned threads_left, unsigned num_threads, cgra_core_ctx *cgra,
  unsigned hw_cta_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread_dice(kernel, thread_info, sid, tid, threads_left,
                           num_threads, cgra, hw_cta_id, 0, gpu);
}

void cgra_core_ctx::reinit(unsigned start_thread, unsigned end_thread,bool reset_not_completed) {
  if (reset_not_completed) {
    m_not_completed = 0;
    m_active_threads.reset();
  }
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  //do this later
  //for (unsigned i = start_thread / 256;
  //     i < end_thread / 256; ++i) {
  //  m_simt_stack[i]->reset(); //reset all simt stacks in the range, although just use the first one.
  //}
  m_scoreboard->clear_tid(start_thread, end_thread);
  //reset block state block size
  //delete existing block state
  //for (unsigned i = 0; i < NUM_DICE_STAGE; i++) {
  //  delete m_cgra_block_state[i];
  //}
  //m_cgra_block_state.clear();
  ////create new block state
  //unsigned total_pipeline_stages = NUM_DICE_STAGE;
  //m_cgra_block_state.reserve(total_pipeline_stages);
  //for (unsigned i = 0; i < total_pipeline_stages; i++) {
  //  m_cgra_block_state.push_back(new cgra_block_state_t(this, m_kernel_block_size));
  //}

  //reset cgra_unit state
  //m_predict_pc_set=0;
  //m_cgra_unit->reinit();
  //reinit fetch buffer
  //m_metadata_fetch_buffer = ifetch_buffer_t();
  //m_bitstream_fetch_buffer = ifetch_buffer_t();
}


void cgra_core_ctx::create_front_pipeline(){
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: create_front_pipeline() id=%d, kernel_block_size = %d\n", m_cgra_core_id, m_kernel_block_size);
    fflush(stdout);
  }
  unsigned total_pipeline_stages = NUM_DICE_STAGE;
  m_cgra_block_state.reserve(total_pipeline_stages);
  for (unsigned i = 0; i < total_pipeline_stages; i++) {
    m_cgra_block_state.push_back(new cgra_block_state_t(this, m_kernel_block_size));
  }

  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),m_config->dice_cgra_core_max_threads);
  m_not_completed = 0;
  m_active_threads.reset();
  m_active_blocks = 0;
  //for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_num_cta_live_threads[i] = 0;

  //create fetch_scheduler hardware
  m_cta_status_table = new cta_status_table(m_gpu, this, MAX_CTA_PER_SHADER);
  m_fetch_scheduler = new fetch_scheduler(m_gpu,this,m_cta_status_table);

  for (unsigned i = 0; i < m_config->dice_cgra_core_max_threads; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }

  // m_icnt = new shader_memory_interface(this,cluster);
  if (m_config->gpgpu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator =
      new shader_core_mem_fetch_allocator(m_cgra_core_id, m_tpc, m_memory_config);

#define STRSIZE 1024
  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_cgra_core_id);
  m_L1I = new read_only_cache(name, m_config->m_L1I_config, m_cgra_core_id,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE);

  snprintf(name, STRSIZE, "L1B_%03d", m_cgra_core_id);
  m_L1B = new read_only_cache(name, m_config->m_L1I_config, m_cgra_core_id,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE);
  
}

void cgra_core_ctx::create_dispatcher(){
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: create_dispatcher() id=%d\n", m_cgra_core_id);
    fflush(stdout);
  }
  m_scoreboard = new Scoreboard(m_cgra_core_id, m_kernel_block_size, m_gpu);
  m_dispatcher_rfu = new dispatcher_rfu_t(m_config, this, &(m_cgra_block_state[DP_CGRA]), m_scoreboard);
}

void cgra_core_ctx::create_execution_unit(){
  if(g_debug_execution >= 3){
    printf("DICE Sim uArch: create_execution_unit() id=%d\n", m_cgra_core_id);
    fflush(stdout);
  }
  m_cgra_unit = new cgra_unit(m_config, this, &(m_cgra_block_state[DP_CGRA]));
  m_ldst_unit = new ldst_unit(m_icnt, m_mem_fetch_allocator, this, m_dispatcher_rfu, &(m_cgra_block_state[DP_CGRA]), m_scoreboard ,m_config, m_memory_config, m_stats, m_cgra_core_id, m_tpc);
  m_block_commit_table = new block_commit_table(m_gpu, this);
}

//hardware simulation
void cgra_core_ctx::cycle(){
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_cgra_core_id]++;
  execute();
  decode();
  fetch_bitstream();
  fetch_metadata();
  cta_schedule();
}

void cgra_core_ctx::accept_metadata_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: accept_metadata_fetch_response() pc=0x%08x\n", mf->get_addr());
    fflush(stdout);
  }
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle+ m_gpu->gpu_tot_sim_cycle);
}

void cgra_core_ctx::accept_bitstream_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
     printf("DICE Sim uArch: accept_bitstream_fetch_response() addr=0x%08x\n", mf->get_addr());
     fflush(stdout);
  }
  m_L1B->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void cgra_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: accept_ldst_unit_response() addr=0x%08x\n", mf->get_addr());
    fflush(stdout);
  }
  m_ldst_unit->fill(mf);
}

#define PROGRAM_MEM_START                                      \
  0xF0000000 /* should be distinct from other memory spaces... \
                check ptx_ir.h to verify this does not overlap \
                other memory spaces */
//outer execution pipeline
void cgra_core_ctx::fetch_metadata(){
  if (!m_metadata_fetch_buffer.m_valid) { //if there is no metadata in the buffer
    if (m_L1I->access_ready()) { //if the instruction cache is ready to be accessed (i.e. there are data responses)
      mem_fetch *mf = m_L1I->next_access(); //get response from the instruction cache
      if(m_cgra_block_state[MF_DE]->dummy() || m_cgra_block_state[MF_DE]->get_metadata_pc() !=(mf->get_addr()-PROGRAM_MEM_START)){
        //Note: it's possible that when addresses dont match but the current one is still a prefetch block from other CTAs.
        //assert(!m_cgra_block_state[MF_DE]->is_prefetch_block());  // Verify that we got the instruction we were expecting.
        //printf("DICE Sim uArch [PREFETCH_META_DISCARD]: Cycle %d, hw_cta=%d ,Block=%d, pc=0x%04x, fetched_pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id() ,m_cgra_block_state[MF_DE]->get_metadata_pc(),mf->get_addr());
        //fflush(stdout);
      } else {
        m_cgra_block_state[MF_DE]->clear_imiss_pending(); //clear the metadata miss flag
        m_metadata_fetch_buffer = ifetch_buffer_t(m_cgra_block_state[MF_DE]->get_metadata_pc(), mf->get_access_size(), mf->get_wid()); //set the metadata fetch buffer
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(m_cgra_block_state[MF_DE]->get_metadata_pc());
          printf("DICE Sim uArch [FETCH_META_END]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle,m_cgra_block_state[MF_DE]->get_cta_id(),meta->meta_id,mf->get_addr());
          fflush(stdout);
        }
        assert(m_cgra_block_state[MF_DE]->get_metadata_pc() ==(mf->get_addr()-PROGRAM_MEM_START));  // Verify that we got the instruction we were expecting.
        //m_metadata_fetch_buffer.m_valid = true; //set valid in the fetch buffer
        m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle); //set status;
      }
      delete mf;
    } else {
      if(m_cgra_block_state[MF_DE]->dummy()) return;
      // check if it's waiting on cache miss or can fetch new instruction
      if (m_cgra_block_state[MF_DE]->done_exit()){
        //do nothing
      }
      else if (!m_cta_status_table->fetch_stalled_by_simt_stack(m_cgra_block_state[MF_DE]->get_cta_id()) && !m_cgra_block_state[MF_DE]->imiss_pending() &&
          m_cgra_block_state[MF_DE]->metadata_buffer_empty()) {//if not stalled by simt stack
        address_type pc;
        //set next pc according to stack top
        address_type next_pc, rpc;
        unsigned cta_id = m_cgra_block_state[MF_DE]->get_cta_id();
        m_simt_stack[cta_id]->get_pdom_stack_top_info(&next_pc, &rpc);
        m_cgra_block_state[MF_DE]->set_next_pc(next_pc);
        pc = m_cgra_block_state[MF_DE]->get_metadata_pc(); //just next_pc
        address_type ppc = pc + PROGRAM_MEM_START;
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(pc);
          printf("DICE Sim uArch [FETCH_META_START]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), meta->meta_id ,pc);
          fflush(stdout);
        }
        unsigned nbytes = 32; //32 bytes for each metadata
        unsigned offset_in_block =
            pc & (m_config->m_L1I_config.get_line_sz() - 1);
        if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
          nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);
        // TODO: replace with use of allocator
        // mem_fetch *mf = m_mem_fetch_allocator->alloc()
        mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
        mem_fetch *mf = new mem_fetch(
            acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
            0, m_cgra_core_id, m_tpc, m_memory_config,
            m_gpu->gpu_tot_sim_cycle +m_gpu->gpu_tot_sim_cycle);
        std::list<cache_event> events;
        enum cache_request_status status;
        if (m_config->perfect_inst_const_cache)
          status = HIT;
        else
          status = m_L1I->access(
              (new_addr_type)ppc, mf,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
        if (status == MISS) {
          m_cgra_block_state[MF_DE]->set_imiss_pending();
          m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
        } else if (status == HIT) {
          if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
            dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(pc);
            printf("DICE Sim uArch [FETCH_META_END]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), meta->meta_id ,pc);
            fflush(stdout);
          }
          m_metadata_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
          m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
          delete mf;
        } else {
          assert(status == RESERVATION_FAIL);
          delete mf;
        }
      } else {
        //if stalled by simt stack and have predict pc, prefetch predicted pc
        if (m_cta_status_table->fetch_stalled_by_simt_stack(m_cgra_block_state[MF_DE]->get_cta_id()) && !m_cgra_block_state[MF_DE]->imiss_pending() && m_cgra_block_state[MF_DE]->metadata_buffer_empty()) {
          address_type pc;
          address_type predict_pc = m_cta_status_table->get_prefetch_pc(m_cgra_block_state[MF_DE]->get_cta_id()); 
          m_cgra_block_state[MF_DE]->set_next_pc(predict_pc);
          m_cgra_block_state[MF_DE]->set_prefetch();//set as a prefetch block
          pc = m_cgra_block_state[MF_DE]->get_metadata_pc(); //just next_pc
          address_type ppc = pc + PROGRAM_MEM_START;
          if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
            dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(pc);
            printf("DICE Sim uArch [FETCH_META_START]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), meta->meta_id ,pc);
            fflush(stdout);
          }
          unsigned nbytes = 32; //32 bytes for each metadata
          unsigned offset_in_block =
              pc & (m_config->m_L1I_config.get_line_sz() - 1);
          if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
            nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);
          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
          mem_fetch *mf = new mem_fetch(
              acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
              0, m_cgra_core_id, m_tpc, m_memory_config,
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
          std::list<cache_event> events;
          enum cache_request_status status;
          if (m_config->perfect_inst_const_cache)
            status = HIT;
          else
            status = m_L1I->access(
                (new_addr_type)ppc, mf,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
          if (status == MISS) {
            m_cgra_block_state[MF_DE]->set_imiss_pending();
            m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
          } else if (status == HIT) {
            if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
              dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(pc);
              printf("DICE Sim uArch [FETCH_META_END]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), meta->meta_id ,pc);
              fflush(stdout);
            }
            m_metadata_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
            m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
            delete mf;
          } else {
            assert(status == RESERVATION_FAIL);
            delete mf;
          }
        }
      }
    }
  }
  //handle branch prediction flush or continue
  if(m_cgra_block_state[MF_DE]->is_prefetch_block()){
    //if prefetch stall clear, which means actual stack is updated
    if(!m_cta_status_table->fetch_stalled_by_simt_stack(m_cgra_block_state[MF_DE]->get_cta_id())){
      m_cgra_block_state[MF_DE]->clear_prefetch();
      //check if the block metadata align with stack top
      address_type next_pc, rpc;
      unsigned cta_id = m_cgra_block_state[MF_DE]->get_cta_id();
      m_simt_stack[cta_id]->get_pdom_stack_top_info(&next_pc, &rpc);
      //branch hit
      if (m_cgra_block_state[MF_DE]->get_metadata_pc() == next_pc){
        //need to reset active mask 
        if(m_cgra_block_state[MF_DE]->decode_done()){
          m_cgra_block_state[MF_DE]->clear_decode_done();
        }
      } else {
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [PREDICT_BRANCH_MISS]: Cycle %d, hw_cta=%d, predict pc=0x%04x, pc=0x%04x\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), m_cgra_block_state[MF_DE]->get_metadata_pc(), next_pc);
          fflush(stdout);
        }
        //branch miss, reset the block
        m_cgra_block_state[MF_DE]->reset();
        m_metadata_fetch_buffer = ifetch_buffer_t();
        m_bitstream_fetch_buffer = ifetch_buffer_t();
      }
    }
  }
  m_L1I->cycle();
}

void cgra_core_ctx::register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel) {
  assert(m_cta_status_table->is_free(cta_num) == false);
  m_cta_status_table->decrease_num_live_threads(cta_num);
  if (m_cta_status_table->is_free(cta_num)) {
    // Increment the completed CTAs
    m_stats->ctas_completed++;
    m_gpu->inc_completed_cta();
    shader_CTA_count_unlog(m_cgra_core_id, 1);
    m_active_blocks--;

    printf("DICE-Sim uArch: core %d Finished CTA #%u (%lld,%lld), %u CTAs running\n",
      m_cgra_core_id,cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
      m_active_blocks);

    if (m_active_blocks == 0) {
      printf("DICE-Sim uArch: core %d Empty (last released kernel %u \'%s\').\n",m_cgra_core_id,kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
    m_kernel = NULL;
    }
    kernel->dec_running();
    if (!m_gpu->kernel_more_cta_left(kernel)) {
      if (!kernel->running()) {
        printf("DICE-Sim uArch: DICE detected kernel %u \'%s\' finished on cgra core %u.\n", kernel->get_uid(), kernel->name().c_str(), m_cgra_core_id);
        if (m_kernel == kernel) m_kernel = NULL;
        m_gpu->set_kernel_done(kernel);
      }
    }
  } 
}

void cgra_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1I) m_L1I->get_sub_stats(css);
}

void cgra_core_ctx::get_L1B_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1B) m_L1B->get_sub_stats(css);
}

void cgra_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css);
}
void cgra_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1C_sub_stats(css);
}
void cgra_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1T_sub_stats(css);
}

dice_cfg_block_t* cgra_block_state_t::get_current_cfg_block() {
  assert(m_metadata_buffer.m_cfg_block!=NULL); 
  return m_metadata_buffer.m_cfg_block; 
}

unsigned cgra_block_state_t::active_count() const{
  return m_active_threads.count(); 
}

bool cgra_block_state_t::is_parameter_load(){
  return get_current_metadata()->is_parameter_load;
}

unsigned cgra_block_state_t::get_unrolling_factor(){
  unsigned unrolling_factor = get_current_metadata()->unrolling_factor;
  if(m_cgra_core->get_config()->dice_enable_unrolling!=1) {
    //if unrolling disabled, return 1
    unrolling_factor = 1;
  }
  return unrolling_factor;
}

void cgra_block_state_t::inc_store_req() { 
  m_stores_outstanding++;
  if(g_debug_execution==3 &m_cgra_core->get_id() == m_cgra_core->get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: inc_store_req() stores_outstanding=%d\n", m_stores_outstanding);
    fflush(stdout);
  }
}

void cgra_block_state_t::dec_store_req() {
  assert(m_stores_outstanding > 0);
  m_stores_outstanding--;
  if(g_debug_execution==3 &m_cgra_core->get_id() == m_cgra_core->get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: dec_store_req() stores_outstanding=%d\n", m_stores_outstanding);
    fflush(stdout);
  }
}

dice_metadata* cgra_block_state_t::get_current_metadata(){
  return get_current_cfg_block()->get_metadata(); 
}

dice_block_t *cgra_block_state_t::get_dice_block(){ 
  return get_current_metadata()->dice_block; 
}

address_type cgra_block_state_t::get_bitstream_pc(){ 
  return get_dice_block()->ptx_instructions.front()->get_PC();
}

unsigned cgra_block_state_t::get_bitstream_size(){ 
  return get_current_metadata()->bitstream_length;
}

unsigned cgra_block_state_t::get_block_latency(){
  return get_current_metadata()->latency;
}

void cgra_block_state_t::metadata_buffer_fill(dice_cfg_block_t *cfg_block) {
  assert(cfg_block!=NULL);
  if(m_metadata_buffer.m_cfg_block)
    delete m_metadata_buffer.m_cfg_block; //delete previous cfg block
  m_metadata_buffer.m_cfg_block = cfg_block;
  m_active_threads = cfg_block->get_active_mask();
  m_metadata_buffer.m_valid = true;
}

bool cgra_block_state_t::barrier_reached(){
  if (get_current_metadata()->barrier) {
    //check block commit table
    unsigned cta_id = get_cta_id();
    if(m_cgra_core->get_block_commit_table()->check_block_exist(cta_id)) return false;
    else return true;
  } else { //no barrier
    return true;
  }
}

void cgra_core_ctx::fetch_bitstream(){
  if(m_cgra_block_state[MF_DE]->dummy()) return;
  if (!m_bitstream_fetch_buffer.m_valid) { //if there is no bitstream in the buffer
    if (m_L1B->access_ready()) { //if the bitstream cache is ready to be accessed (i.e. there are data responses)
      mem_fetch *mf = m_L1B->next_access(); //get response from the instruction cache
      //when previous kernel mis predict and still have outstanding bitstream fetch request
      if(m_cgra_block_state[MF_DE]->decode_done()==false || m_cgra_block_state[MF_DE]->get_bitstream_pc() !=(mf->get_addr()-PROGRAM_MEM_START)){
        printf("DICE Sim uArch [PREFETCH_BITS_DISCARD]: Core %d, Cycle %d, hw_cta=%d\n",m_cgra_core_id, m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id());
        printf("m_cgra_block_state[MF_DE]->decode_done()=%d, (mf->get_addr()-PROGRAM_MEM_START)=%d",m_cgra_block_state[MF_DE]->decode_done(),(mf->get_addr()-PROGRAM_MEM_START));
        fflush(stdout);
        //note: it's possible that after misprediction, the next block is still a prefetch block from another CTA. This assert no longer needed
        //assert(!m_cgra_block_state[MF_DE]->is_prefetch_block());  // Verify that we got the instruction we were expecting.
      } else {
        m_cgra_block_state[MF_DE]->clear_bmiss_pending(); //clear the metadata miss flag
        m_bitstream_fetch_buffer = ifetch_buffer_t(m_cgra_block_state[MF_DE]->get_bitstream_pc(), mf->get_access_size(), mf->get_wid()); //set the metadata fetch buffer
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(m_cgra_block_state[MF_DE]->get_metadata_pc());
          printf("DICE Sim uArch [FETCH_BITS_END]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id(), meta->meta_id,mf->get_addr());
          fflush(stdout);
        }
        assert(m_cgra_block_state[MF_DE]->get_bitstream_pc() ==(mf->get_addr()-PROGRAM_MEM_START));  // Verify that we got the instruction we were expecting.
        //m_bitstream_fetch_buffer.m_valid = true; //set valid in the fetch buffer
        m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle); //set status
        m_cgra_block_state[MF_DE]->bitstream_buffer_fill(); //fill the bitstream buffer
      }
      delete mf;
    } else {
      // this code fetches metadata from the bitstream-cache or generates memory
      if (!m_cgra_block_state[MF_DE]->functional_done() &&
          !m_cgra_block_state[MF_DE]->bmiss_pending() &&
          m_cgra_block_state[MF_DE]->bitstream_buffer_waiting()) { //metadata ready but no bitstream is not fetched
        if(m_cgra_block_state[MF_DE]->done_exit()){
          return;
        }
        address_type pc;
        pc = m_cgra_block_state[MF_DE]->get_bitstream_pc();
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(m_cgra_block_state[MF_DE]->get_metadata_pc());
          printf("DICE Sim uArch [FETCH_BITS_START]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle,m_cgra_block_state[MF_DE]->get_cta_id(),meta->meta_id,pc);
          fflush(stdout);
        }
        address_type ppc = pc + PROGRAM_MEM_START;
        unsigned nbytes = m_cgra_block_state[MF_DE]->get_bitstream_size()*8; //bitstream length = 8* # of ptx instructions in the block
        unsigned offset_in_block =
            pc & (m_config->m_L1I_config.get_line_sz() - 1); //reuse now, DICE-TODO: add seperate L1B config
        if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
          nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);
        // TODO: replace with use of allocator
        // mem_fetch *mf = m_mem_fetch_allocator->alloc()
        mem_access_t acc(BITSTREAM_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
        mem_fetch *mf = new mem_fetch(
            acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
            0, m_cgra_core_id, m_tpc, m_memory_config,
            m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
        std::list<cache_event> events;
        enum cache_request_status status;
        if (m_config->perfect_inst_const_cache)
          status = HIT;
        else
          status = m_L1B->access(
              (new_addr_type)ppc, mf,
              m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, events);
        if (status == MISS) {
          m_cgra_block_state[MF_DE]->set_bmiss_pending();
          m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
        } else if (status == HIT) {
          if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
            dice_metadata* meta = m_gpu->gpgpu_ctx->pc_to_metadata(m_cgra_block_state[MF_DE]->get_metadata_pc());
            printf("DICE Sim uArch [FETCH_BITS_END]: Cycle %d, hw_cta=%d, Block=%d, pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle,m_cgra_block_state[MF_DE]->get_cta_id(),meta->meta_id,pc);
            fflush(stdout);
          }
          m_bitstream_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
          m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
          m_cgra_block_state[MF_DE]->bitstream_buffer_fill(); //fill the bitstream buffer
          delete mf;
        } else {
          assert(status == RESERVATION_FAIL);
          delete mf;
        }
      } else {
        //if(m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle>553 && m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle<558){
        //  printf("DICE-Sim uArch: cycle %d, fetch_bitstream() stalled because of\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
        //  printf("function done = %d\n", m_cgra_block_state[MF_DE]->functional_done());
        //  printf("bmiss_pending = %d\n", m_cgra_block_state[MF_DE]->bmiss_pending());
        //  printf("bitstream_buffer_waiting = %d\n", m_cgra_block_state[MF_DE]->bitstream_buffer_waiting());
        //}
      }
    }
  }
  m_L1B->cycle();
}

dice_cfg_block_t* cgra_core_ctx::get_dice_cfg_block(address_type pc, unsigned cta_id) {
  // read the metadata from the functional model
  dice_metadata* metadata = m_gpu->gpgpu_ctx->get_meta_from_pc(pc);
  dice_cfg_block_t* cfg_block = new dice_cfg_block_t(m_cgra_core_id, m_kernel_block_size, metadata, m_gpu->gpgpu_ctx);
  cfg_block->set_active(m_simt_stack[cta_id]->get_active_mask());
  return cfg_block;
}

void cgra_core_ctx::cache_flush() { m_ldst_unit->flush(); }

void cgra_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }

void cgra_core_ctx::decode(){
  if(m_cgra_block_state[MF_DE]->dummy()) return;
  unsigned cta_id = m_cgra_block_state[MF_DE]->get_cta_id();
  unsigned start_thread = m_cta_status_table->get_start_thread(cta_id);
  unsigned end_thread = m_cta_status_table->get_end_thread(cta_id);
  if (m_metadata_fetch_buffer.m_valid && m_cgra_block_state[MF_DE]->decode_done() == false) {
    // decode and place them into ibuffer
    address_type pc = m_metadata_fetch_buffer.m_pc;
    dice_cfg_block_t *cfg_block;
    if(m_cgra_block_state[MF_DE]->metadata_buffer_empty()){
      cfg_block = get_dice_cfg_block(pc,m_cgra_block_state[MF_DE]->get_cta_id());
      m_cgra_block_state[MF_DE]->metadata_buffer_fill(cfg_block);
    } else {
      cfg_block = m_cgra_block_state[MF_DE]->get_current_cfg_block();
      cfg_block->set_active(m_simt_stack[cta_id]->get_active_mask());
      m_cgra_block_state[MF_DE]->set_active_threads(m_simt_stack[cta_id]->get_active_mask());
    }
    if (cfg_block) {
      m_stats->m_num_decoded_insn[m_cgra_core_id]++;
    }
    //update simt stack or call a stack stall
    //check if branch involved in the block
    dice_metadata* metadata = m_cgra_block_state[MF_DE]->get_current_metadata();
    //check if the block is exit block
    if (metadata->is_ret) {
      //wait for previous block to finish
      m_cgra_block_state[MF_DE]->set_done_exit();
      if(!m_cgra_block_state[DP_CGRA]->dummy() && (m_cgra_block_state[DP_CGRA]->get_cta_id() == cta_id)){
        //which means for this cta, DP-CGRA block is not finished yet
        //printf("DICE-Sim uArch: cycle %d, decode block is exit block but DP-CGRA block is not finished\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
        return; 
      }

      //switch this to block_commit_table 
      //if (!m_cgra_block_state[MEM_WRITEBACK]->dummy() && !m_cgra_block_state[MEM_WRITEBACK]->block_done()) {
      //  //printf("DICE-Sim uArch: cycle %d, decode block is exit block but MEM_WRITEBACK block is not finished\n", m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle);
      //  return;
      //}
      if(m_block_commit_table->check_block_exist(cta_id)){
        //still have pending writebacks in current cta
        m_block_commit_table->mark_return_block(cta_id);
        m_cta_status_table->set_ret(cta_id);
        //initialize fetch hardware
        delete m_cgra_block_state[MF_DE];
        m_cgra_block_state[MF_DE] = new cgra_block_state_t(this, m_kernel_block_size);
        //m_cgra_block_state[MF_DE]->init(unsigned(-1), m_cgra_block_state[DP_CGRA]->get_cta_id(), m_cgra_block_state[DP_CGRA]->get_active_threads());
        //m_cgra_block_state[MF_DE] = m_fetch_scheduler->next_fetch_block();
        m_metadata_fetch_buffer = ifetch_buffer_t();
        m_bitstream_fetch_buffer = ifetch_buffer_t();
        if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DECODE-PUSH-RETURN]: cycle %d, hw_cta=%d, decode metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id, pc);
          fflush(stdout);
        }
        return;
      }
      //direct exit here instead of run ret
      //iterate through all active threads
      //register exit
      complete_cta(cta_id);

      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [DECODE-RETURN]: cycle %d, hw_cta=%d, decode metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id, pc);
        fflush(stdout);
      }

      //initialize fetch hardware
      delete m_cgra_block_state[MF_DE];
      m_cgra_block_state[MF_DE] = new cgra_block_state_t(this, m_kernel_block_size);
      //m_cgra_block_state[MF_DE]->init(unsigned(-1), m_cgra_block_state[DP_CGRA]->get_cta_id(), m_cgra_block_state[DP_CGRA]->get_active_threads());
      //m_cgra_block_state[MF_DE] = m_fetch_scheduler->next_fetch_block();
      m_metadata_fetch_buffer = ifetch_buffer_t();
      m_bitstream_fetch_buffer = ifetch_buffer_t();

      return;
    }
    if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
      printf("DICE Sim uArch [DECODE]: cycle %d, hw_cta=%d, decode metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id ,pc);
      fflush(stdout);
    }

    if(m_cgra_block_state[MF_DE]->is_prefetch_block()){
      //do nothing with the stack
    } else if (metadata->branch) {
      if(metadata->uni_bra){
        //update simt stack top with branch target pc
        m_simt_stack[m_cgra_block_state[MF_DE]->get_cta_id()]->update_no_divergence(metadata->branch_target_meta_pc);
      }
      else {
        //predicate branch, wait for active mask to be done to update
        //create fake stack first and mark waiting for active mask
        address_type predict_pc = branch_predictor(metadata);
        m_cta_status_table->set_fetch_stalled_by_simt_stack(m_cgra_block_state[MF_DE]->get_cta_id(),m_cgra_block_state[MF_DE]->get_current_metadata()->meta_id,predict_pc);
      }
    } else {
      //update simt stack top with next pc
      m_simt_stack[m_cgra_block_state[MF_DE]->get_cta_id()]->update_no_divergence(metadata->get_PC()+metadata->metadata_size());
    }
    //set state and fill the cfg block
    m_cgra_block_state[MF_DE]->set_decode_done();
  }
}

void cgra_core_ctx::complete_cta(unsigned hw_cta_id){
  unsigned start_thread = m_cta_status_table->get_start_thread(hw_cta_id);
  unsigned end_thread = m_cta_status_table->get_end_thread(hw_cta_id);
  for (unsigned tid = start_thread; tid < end_thread; tid++) {
    if (m_threadState[tid].m_active == true) {
      bool empty = m_thread[tid]->callstack_pop();
      if (empty) {
        m_thread[tid]->set_done();
        m_thread[tid]->registerExit();
      }
      m_threadState[tid].m_active = false;
      if (m_thread[tid] == NULL) {
        register_cta_thread_exit(hw_cta_id, m_kernel);
      } else {
        register_cta_thread_exit(hw_cta_id, &(m_thread[tid]->get_kernel()));
      }
      m_not_completed -= 1;
      m_active_threads.reset(tid);
    }
  }
}

//simple predict rule: if target pc > current pc, then predict not taken
//if target pc < current pc, then predict taken
address_type cgra_core_ctx::branch_predictor(dice_metadata* metadata){
  address_type pc = metadata->get_PC();
  address_type target_pc = metadata->branch_target_meta_pc;
  if(target_pc > pc){
    return pc + metadata->metadata_size();
  }
  else{
    return target_pc;
  }
}

bool cgra_block_state_t::loads_done(){
  assert(m_num_loads_done <= get_current_cfg_block()->get_num_loads());
  if(m_num_loads_done == get_current_cfg_block()->get_num_loads()){
    return true;
  }
  return false;
}
bool cgra_block_state_t::stores_out(){
  if(m_num_stores_done > get_current_cfg_block()->get_num_stores()){
    printf("DICE Sim uArch: stores_out() stores_outstanding=%d, num_stores_done=%d\n", m_num_stores_done, get_current_cfg_block()->get_num_stores());
    fflush(stdout);
  }
  assert(m_num_stores_done <= get_current_cfg_block()->get_num_stores());
  if(m_num_stores_done == get_current_cfg_block()->get_num_stores()){
    return true;
  }
  return false;
}

void cgra_block_state_t::clear_prefetch() { 
  is_prefetch = false; 
  if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: Cycle %d, clear_prefetch() for cta_id = %d\n", m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cta_id);
    fflush(stdout);
  }
}

bool cgra_block_state_t::mem_access_queue_empty(){ return get_current_cfg_block()->accessq_empty(); }

void cgra_core_ctx::execute(){
  writeback();
  cgra_execute_block();
  dispatch();
}
//inner pipeline in execute();
void cgra_core_ctx::dispatch(){
  m_dispatcher_rfu->dispatch();
  //if(m_cgra_core_id==0 && m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle > 1891 && m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle < 1895){
  //  printf("DICE Sim uArch [DISPATCHER]:\n");
  //  if(m_cgra_block_state[DP_CGRA]==NULL){
  //    printf("m_cgra_block_state[DP_CGRA] is NULL\n");
  //  }
  //  else{
  //    printf("m_cgra_block_state[DP_CGRA]->dummy()=%d\n",m_cgra_block_state[DP_CGRA]->dummy());
  //    printf("m_cgra_block_state[MF_DE]->ready_to_dispatch() = %d\n",m_cgra_block_state[MF_DE]->ready_to_dispatch());
  //    printf("m_cgra_block_state[MF_DE]->decode_done()=%d\n",m_cgra_block_state[MF_DE]->decode_done());
  //    printf("m_cgra_block_state[MF_DE]->metadata_buffer_empty()=%d\n",m_cgra_block_state[MF_DE]->metadata_buffer_empty());
  //    printf("m_cgra_block_state[MF_DE]->bitstream_buffer_waiting()=%d\n",m_cgra_block_state[MF_DE]->bitstream_buffer_waiting());
  //  }
  //  fflush(stdout);
  //}

  //check if current block is finished
  if(m_cgra_block_state[DP_CGRA]==NULL || m_cgra_block_state[DP_CGRA]->dummy()){ //in write-back stage already
    //check if there is a block in MF_DE stage and ready to dispatch
    if(m_cgra_block_state[MF_DE]->ready_to_dispatch() && m_cgra_block_state[MF_DE]->barrier_reached()){
      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [DISPATCH_START]: Cycle %d, hw_cta=%d, Block=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id() ,m_cgra_block_state[MF_DE]->get_current_metadata()->meta_id,m_cgra_block_state[MF_DE]->get_current_metadata()->get_PC());
        fflush(stdout);
      }
      if(m_cgra_block_state[DP_CGRA] != NULL){
        delete m_cgra_block_state[DP_CGRA]; //delete previous DP_CGRA block
      }
      m_cgra_block_state[DP_CGRA] = m_cgra_block_state[MF_DE];
      m_cgra_unit->set_latency(m_cgra_block_state[DP_CGRA]->get_block_latency());
      m_cgra_block_state[MF_DE] = new cgra_block_state_t(this, m_kernel_block_size);
      //m_cgra_block_state[MF_DE]->init(unsigned(-1), m_cgra_block_state[DP_CGRA]->get_cta_id(), m_cgra_block_state[DP_CGRA]->get_active_threads());
      //m_cgra_block_state[MF_DE] = m_fetch_scheduler->next_fetch_block();
      m_metadata_fetch_buffer = ifetch_buffer_t();
      m_bitstream_fetch_buffer = ifetch_buffer_t();
    }
  }
}

void cgra_core_ctx::cta_schedule(){
  if(m_cgra_block_state[MF_DE]->dummy()){ //no content
    unsigned next_cta = m_fetch_scheduler->next_fetch_block();
    if(next_cta!=unsigned(-1)){
      const simt_mask_t &temp = m_simt_stack[next_cta]->get_active_mask();
      delete m_cgra_block_state[MF_DE]; //delete previous MF_DE block
      m_cgra_block_state[MF_DE] = new cgra_block_state_t(this, m_kernel_block_size);
      m_cgra_block_state[MF_DE]->init(unsigned(-1), next_cta, temp); //set as valid now
      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [CTA_SCHEDULE]: Cycle %d, hw_cta=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MF_DE]->get_cta_id());
        fflush(stdout);
      }
    }
  }
}

void cgra_core_ctx::cgra_execute_block(){
  if(m_cgra_block_state[DP_CGRA]->dispatch_done()) {
    //set total number of execute
    unsigned cta_id = m_cgra_block_state[DP_CGRA]->get_cta_id();
    unsigned total_need_exec = m_cgra_block_state[DP_CGRA]->is_parameter_load()? 1 : m_cgra_block_state[DP_CGRA]->active_count();
    //printf("DICE Sim uArch [CGRA_EXECU_START]: Cycle %d, hw_cta=%d, Block=%d, total need exec=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id,m_cgra_block_state[DP_CGRA]->get_current_metadata()->meta_id, total_need_exec);
    //printf("m_cgra_unit->get_num_executed_thread() = %d\n", m_cgra_unit->get_num_executed_thread());
    if((m_cgra_unit->get_num_executed_thread()==total_need_exec) && !m_cgra_block_state[DP_CGRA]->cgra_fabric_done()){
      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [CGRA_EXECU_END]: Cycle %d, hw_cta=%d, Block=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id, m_cgra_block_state[DP_CGRA]->get_current_metadata()->meta_id);
        fflush(stdout);
      }
      m_cgra_block_state[DP_CGRA]->set_cgra_fabric_done();
    }
  }
  m_cgra_unit->cycle();
}

void cgra_core_ctx::writeback(){
  //check if cgra fabric is finished
  //if all load writeback are done, then set writeback done
  /*
  if(!m_cgra_block_state[MEM_WRITEBACK]->dummy()){
    if(m_cgra_block_state[MEM_WRITEBACK]->loads_done() && m_cgra_block_state[MEM_WRITEBACK]->stores_done() && !m_cgra_block_state[MEM_WRITEBACK]->writeback_done()){
      m_cgra_block_state[MEM_WRITEBACK]->set_writeback_done();
      m_gpu->gpu_sim_block += m_cgra_block_state[MEM_WRITEBACK]->active_count();
      if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [WRITEBACK_END]: Cycle %d, Block=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->meta_id);
        fflush(stdout);
      }
      //TODO, can prefetch next block here or better SIMT stack operation
      if (m_fetch_stalled_by_simt_stack && (!m_cgra_block_state[MEM_WRITEBACK]->dummy())){
        if(m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->meta_id == m_fetch_waiting_block_id) {
          assert(m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->branch);
          dice_cfg_block_t *cfg_block = m_cgra_block_state[MEM_WRITEBACK]->get_current_cfg_block();
          assert(cfg_block != NULL);
          //check if predicate registers are all written back
          updateSIMTStack(cfg_block);
          clear_stalled_by_simt_stack();
        }
      }
    } 
    //else {
    //  //if(g_debug_execution >= 3 && m_cgra_core_id==0 && m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle > 4290 && m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle < 4295){
    //  //  printf("DICE Sim uArch [WRITEBACK-DEBUG]: cycle %d, writeback state for dice block id=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->meta_id);
    //  //  printf("loads_done = %d\n", m_cgra_block_state[MEM_WRITEBACK]->loads_done());
    //  //  printf("finished number of loads = %d, need number of loads = %d, \n", m_cgra_block_state[MEM_WRITEBACK]->get_number_of_loads_done(),m_cgra_block_state[MEM_WRITEBACK]->get_current_cfg_block()->get_num_loads());
    //  //  printf("stores_done = %d\n", m_cgra_block_state[MEM_WRITEBACK]->stores_done());
    //  //  printf("finished number of stores = %d, need number of stores = %d, \n", m_cgra_block_state[MEM_WRITEBACK]->get_number_of_stores_done(),m_cgra_block_state[MEM_WRITEBACK]->get_current_cfg_block()->get_num_stores());
    //  //  fflush(stdout);
    //  //}
    //}
  }
  */
  
  m_block_commit_table->check_and_release();

  //if done, move to next stage
  //if(m_cgra_block_state[DP_CGRA]->cgra_fabric_done() && m_cgra_block_state[DP_CGRA]->mem_access_queue_empty() && (m_cgra_block_state[MEM_WRITEBACK]->block_done() || m_cgra_block_state[MEM_WRITEBACK]->dummy())){
  if(m_cgra_block_state[DP_CGRA]->cgra_fabric_done() && m_cgra_block_state[DP_CGRA]->mem_access_queue_empty() && (m_block_commit_table->available())){
    //move to writeback stage
    unsigned index=m_block_commit_table->get_available_index();
    m_block_commit_table->reserve(index, m_cgra_block_state[DP_CGRA]);
    if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
      printf("DICE Sim uArch [WRITEBACK_START]: Cycle %d, hw_cta=%d, Block=%d, table_index=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_block_state[DP_CGRA]->get_cta_id(), m_cgra_block_state[DP_CGRA]->get_current_metadata()->meta_id,index);
      fflush(stdout);
    }
    unsigned cta_id = m_cgra_block_state[DP_CGRA]->get_cta_id();
    simt_mask_t active_mask = m_cgra_block_state[DP_CGRA]->get_active_threads();
    //m_cgra_block_state[MEM_WRITEBACK] = m_cgra_block_state[DP_CGRA];
    m_cgra_block_state[DP_CGRA] = new cgra_block_state_t(this, m_kernel_block_size);
    //m_cgra_block_state[DP_CGRA]->init(unsigned(-1), cta_id, active_mask);
    m_cgra_unit->flush_pipeline();
  }

  //register writeback
  for(unsigned lane_id=0; lane_id < 4; lane_id++){
    if(m_cgra_unit->out_valid(lane_id)){//not stalled
      m_cgra_unit->inc_num_executed_thread();
      unsigned tid=m_cgra_unit->out_tid(lane_id);
      execute_1thread_CFGBlock(m_cgra_block_state[DP_CGRA], tid, lane_id);
      //check if bank conflict with writeback request from ldst unit, if yes, then writeback to writeback_buffer
      m_dispatcher_rfu->writeback_cgra(m_cgra_block_state[DP_CGRA],tid);//TODO
      //move to dispatcher to check if bankconflict
      //m_scoreboard->releaseRegisters(m_cgra_block_state[DP_CGRA]->get_current_metadata(), tid);
      //move to m_dispatcher_rfu->writeback_ldst
      //m_scoreboard->releaseRegistersFromLoad(m_cgra_block_state[DP_CGRA]->get_current_metadata(), tid);
    }
  }
  //cycle every RF bank
  m_dispatcher_rfu->rf_cycle();

  //ldst unit writeback
  m_ldst_unit->cycle_cgra();
}

unsigned cgra_core_ctx::get_cta_start_tid(unsigned hw_cta_id) const {
  assert(hw_cta_id < m_cta_status_table->get_size());
  return m_cta_status_table->get_start_thread(hw_cta_id);
}

unsigned cgra_core_ctx::get_cta_end_tid(unsigned hw_cta_id) const {
  assert(hw_cta_id < m_cta_status_table->get_size());
  return m_cta_status_table->get_end_thread(hw_cta_id);
}

void cgra_core_ctx::clear_fetch_stalled_by_simt_stack(unsigned hw_cta_id, unsigned fetch_waiting_block_id){
  m_cta_status_table->clear_fetch_stalled_by_simt_stack(hw_cta_id, fetch_waiting_block_id);
}

bool cgra_core_ctx::fetch_stalled_by_simt_stack(unsigned hw_cta_id){
  assert(!m_cta_status_table->is_free(hw_cta_id));
  return m_cta_status_table->fetch_stalled_by_simt_stack(hw_cta_id);
}
unsigned cgra_core_ctx::get_fetch_waiting_block_id(unsigned hw_cta_id) {
  assert(!m_cta_status_table->is_free(hw_cta_id));
  assert(fetch_stalled_by_simt_stack(hw_cta_id));
  return m_cta_status_table->get_fetch_waiting_block_id(hw_cta_id);
}

bool block_commit_table::check_block_exist(unsigned hw_cta_id) {
  for(unsigned i = 0; i < m_max_block_size; i++) {
    if (m_commit_table[i] != NULL && m_commit_table[i]->get_cta_id() == hw_cta_id) return true;
  }
  return false;
}

unsigned block_commit_table::find_block_index(unsigned hw_cta_id) {
  for(unsigned i = 0; i < m_max_block_size; i++) {
    if (m_commit_table[i] != NULL && m_commit_table[i]->get_cta_id() == hw_cta_id) return i;
  }
  return unsigned(-1);
}

void block_commit_table::mark_return_block(unsigned hw_cta_id) {
  assert(check_block_exist(hw_cta_id));
  //unsigned index = find_block_index(hw_cta_id);
  //need to find all block index
  for(unsigned i = 0; i < m_max_block_size; i++) {
    if (m_commit_table[i] != NULL && m_commit_table[i]->get_cta_id() == hw_cta_id) {
      m_ret[i] = true;
    }
  }
}

void dispatcher_rfu_t::rf_cycle(){
  for(unsigned i = 0; i < m_rf_bank_controller.size(); i++){
    m_rf_bank_controller[i]->cycle();
  }
}

void shader_memory_interface::push_cgra(mem_fetch *mf) {
  m_cgra_core->inc_simt_to_mem(mf->get_num_flits(true));
  m_cluster->icnt_inject_request_packet(mf);
}

void perfect_memory_interface::push_cgra(mem_fetch *mf) {
  if (mf && mf->isatomic())
    mf->do_atomic();  // execute atomic inside the "memory subsystem"
  m_cgra_core->inc_simt_to_mem(mf->get_num_flits(true));
  m_cluster->push_response_fifo(mf);
}


void cgra_unit::print() const {
  printf("%s CGRA pipeline (current latency = %d, is_busy = %d):  ", m_name.c_str(),m_latency,is_busy);
  for(unsigned lane_id = 0; lane_id < 4; lane_id++){
    for(unsigned i = 0; i < MAX_CGRA_FABRIC_LATENCY; i++){
      printf("%d ", shift_registers[lane_id][i]);
    }
  }
  printf("\n");
  fflush(stdout);
}

void cgra_unit::cycle(){
  //shift the shift registers to the right
  if(stalled()){
    is_busy = true;
    return;
  }
  for(unsigned lane_id = 0; lane_id < 4; lane_id++){
    for (unsigned i = MAX_CGRA_FABRIC_LATENCY-1; i > 0; i--) {
      shift_registers[lane_id][i] = shift_registers[lane_id][i - 1];
    }
  }
  //shift_registers[0] = unsigned(-1); //will be modified later
  is_busy = false;
  for(unsigned lane_id = 0; lane_id < 4; lane_id++){
    for(unsigned i = 0; i < m_latency+1; i++){
      if(shift_registers[lane_id][i] != unsigned(-1)){
        is_busy = true;
        break;
      }
      if(is_busy) break;
    }
  }
  if(g_debug_execution==3 &m_cgra_core->get_id() == m_cgra_core->get_dice_trace_sampling_core()){
    //print();
  }
}

cgra_unit::cgra_unit(const shader_core_config *config, cgra_core_ctx *cgra_core, cgra_block_state_t **block){
  //reserve shift registers
  //m_max_cgra_latency = config->dice_cgra_core_max_latency;
  m_name = "CGRA_default";
  m_executing_block = block;
  m_cgra_core = cgra_core;
  for (unsigned lane_id = 0; lane_id <4; lane_id++){
    for (unsigned i = 0; i < MAX_CGRA_FABRIC_LATENCY; i++) {
      shift_registers[lane_id][i] = unsigned(-1); //unsigned(-1) means empty/invalid
    }
  }
  m_latency = MAX_CGRA_FABRIC_LATENCY-1;
  stalled_by_ldst_unit_queue_full = false;
  stalled_by_wb_buffer_full = false;
  is_busy = false;
  m_num_executed_thread = 0;
  reinit();
}

void cgra_core_ctx::exec(unsigned tid, unsigned lane_id){
  m_cgra_unit->exec(tid, lane_id);
  //run function simulation here
  //move to when cgra out valid.
  if(tid != unsigned(-1)){
    //execute_1thread_CFGBlock(m_cgra_block_state[DP_CGRA]->get_current_cfg_block(), tid);
    //if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
    //  printf("DICE-Sim Functional: cycle %d, cgra_core %u executed thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, m_cgra_core_id, tid,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
    //}
  }
}

void cgra_core_ctx::set_exec_stalled_by_writeback_buffer_full(){
  m_exec_stalled_by_writeback_buffer_full = true;
  m_cgra_unit->set_stalled_by_wb_buffer_full();
}
void cgra_core_ctx::clear_exec_stalled_by_writeback_buffer_full(){
  m_exec_stalled_by_writeback_buffer_full = false;
  m_cgra_unit->clear_stalled_by_wb_buffer_full();
}

void cgra_core_ctx::set_exec_stalled_by_ldst_unit_queue_full() { 
  m_exec_stalled_by_ldst_unit_queue_full = true; 
  m_cgra_unit->set_stalled_by_ldst_unit_queue_full(); 
}

void cgra_core_ctx::clear_exec_stalled_by_ldst_unit_queue_full() { 
  m_exec_stalled_by_ldst_unit_queue_full = false; 
  m_cgra_unit->clear_stalled_by_ldst_unit_queue_full(); 
}

bool cgra_core_ctx::check_ldst_unit_stall(){
  return m_ldst_unit->mem_access_queue_full();
}

unsigned cgra_core_ctx::get_cta_size(unsigned hw_cta_id) {
  return m_cta_status_table->get_cta_size(hw_cta_id);
}

rf_bank_controller::rf_bank_controller(unsigned bank_id, dispatcher_rfu_t *rfu) {
  m_rfu = rfu;
  m_bank_id = bank_id;
  m_cgra_buffer_size = m_rfu->get_config()->dice_cgra_core_rf_cgra_wb_buffer_size;
  m_ldst_buffer_size = m_rfu->get_config()->dice_cgra_core_rf_ldst_wb_buffer_size;
}

bool rf_bank_controller::wb_buffer_full(){
  return (m_cgra_writeback_buffer.size() >= m_cgra_buffer_size) || 
         (m_ldst_writeback_buffer.size() >= m_ldst_buffer_size);
}

void rf_bank_controller::cycle(){
  if(m_ldst_writeback_buffer.size() > 0){
    //std::pair<unsigned,cgra_block_state_t*> wb = m_ldst_writeback_buffer.front();
    m_ldst_writeback_buffer.pop_front();
    return;
  }
  if(m_cgra_writeback_buffer.size() > 0){
    //std::pair<unsigned,cgra_block_state_t*> wb = m_cgra_writeback_buffer.front();
    m_cgra_writeback_buffer.pop_front();
  }
}

bool dispatcher_rfu_t::writeback_buffer_full(dice_metadata* metadata) const {
  //check if the writeback buffer is full for each direct register writeback
  //get reg_num from metadata
  //for(std::list<operand_info>::iterator it = metadata->out_regs.begin(); it != metadata->out_regs.end(); ++it){
  //  unsigned reg_num = (*it).reg_num();
  //  if(reg_num >= m_rf_bank_controller.size()){
  //    printf("DICE Sim uArch [ERROR]: cycle %d, hw_cta=%d, reg_num %d larger than max_number of RF banks = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id(), reg_num,m_rf_bank_controller.size());
  //    fflush(stdout);
  //    assert(reg_num < m_rf_bank_controller.size());
  //  }
  //  if(m_rf_bank_controller[reg_num]->wb_buffer_full()){
  //    return true;
  //  }
  //}
  for(unsigned bank_id = 0; bank_id < m_rf_bank_controller.size(); bank_id++){
    if(m_rf_bank_controller[bank_id]->wb_buffer_full()){
      return true;
    }
  }
  return false;
}

void dispatcher_rfu_t::dispatch(){
  //send to execute in ready buffer
  for(unsigned lane_id = 0; lane_id < 4; lane_id++){
    if(m_ready_threads[lane_id].size() > 0){
      if(!exec_stalled()){
        unsigned tid = m_ready_threads[lane_id].front();
        if(tid!=unsigned(-1) && g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, lane_id= %d, operands ready of thread %d for dice block id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id(), lane_id, tid ,(*m_dispatching_block)->get_current_metadata()->meta_id);
          fflush(stdout);
        }
        m_ready_threads[lane_id].pop_front();
        m_cgra_core->exec(tid,lane_id);//issue thread to cgra unit
        m_dispatched_thread++;
      } 
    } else {
      if(!exec_stalled()) {
        m_cgra_core->exec(unsigned(-1),lane_id);//issue nop to cgra unit
      }
    }
  }

  if((*m_dispatching_block)->ready_to_dispatch()) {
    if(!exec_stalled()) {
      if(writeback_buffer_full((*m_dispatching_block)->get_current_metadata())){
        //stall the exec
        m_cgra_core->set_exec_stalled_by_writeback_buffer_full();
        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, exec stalled because of writeback buffer full\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id());
          fflush(stdout);
        }
      }
      if(m_cgra_core->check_ldst_unit_stall()){
        //stall the exec
        m_cgra_core->set_exec_stalled_by_ldst_unit_queue_full();
        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, exec stalled because of ldst unit queue full\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id());
          fflush(stdout);
        }
      }
    } else {
      if(m_cgra_core->is_exec_stalled_by_writeback_buffer_full() && !writeback_buffer_full((*m_dispatching_block)->get_current_metadata())){
        //restart the exec
        m_cgra_core->clear_exec_stalled_by_writeback_buffer_full();
        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, clear stall caused by writeback buffer full\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,(*m_dispatching_block)->get_cta_id());
          fflush(stdout);
        }
      }
      if(m_cgra_core->is_exec_stalled_by_ldst_unit_queue_full() && !m_cgra_core->check_ldst_unit_stall()){
        //restart the exec
        m_cgra_core->clear_exec_stalled_by_ldst_unit_queue_full();
        
        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, clear stall caused by ldst unit queue full\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,(*m_dispatching_block)->get_cta_id());
          fflush(stdout);
        }
      }
    }
  }
  //check if all thread in current block is dispatched. if not, keep dispatching
  if ((*m_dispatching_block)->ready_to_dispatch() && !(*m_dispatching_block)->dispatch_done()) {
    //unsigned max_coalesce = m_cgra_core->get_config()->dice_ldst_unit_temporal_coalescing_interval;
    unsigned max_coalesce = 32;
    //number of dispatch, if parameter load, just dispatch 1 thread.
    unsigned total_need_dispatch = (*m_dispatching_block)->is_parameter_load()? 1:(*m_dispatching_block)->active_count();
    unsigned unrolling_factor = (*m_dispatching_block)->is_parameter_load()? 1:(*m_dispatching_block)->get_unrolling_factor();
    //calculating current dispatched thread count
    unsigned real_dispatched_count = get_actual_dispatched_count();
    if(real_dispatched_count < total_need_dispatch){
      std::vector<unsigned> next_ready_threads;
      std::vector<unsigned> actual_dispatch_threads;
      next_ready_threads.resize(unrolling_factor);
      actual_dispatch_threads.resize(4);
      actual_dispatch_threads[0] = unsigned(-1);
      actual_dispatch_threads[1] = unsigned(-1);
      actual_dispatch_threads[2] = unsigned(-1);
      actual_dispatch_threads[3] = unsigned(-1);
      for(unsigned unrolling_index=0;unrolling_index<unrolling_factor;unrolling_index++){
        next_ready_threads[unrolling_index] = next_active_thread(unrolling_factor, unrolling_index,max_coalesce);
      }
      //check no active threads error
      bool no_active_thread = true;
      for(unsigned unrolling_index=0;unrolling_index<unrolling_factor;unrolling_index++){
        if(next_ready_threads[unrolling_index] != unsigned(-1)){
          no_active_thread = false;
          break;
        }
      }
      if(no_active_thread && real_dispatched_count < total_need_dispatch){
        printf("DICE-Sim uArch: [Error] cycle %d, core %d, no more active thread found in the block\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cgra_core->get_id());
        printf("DICE-Sim uArch: current dispatched number = %d, need total = %d \n",get_actual_dispatched_count(), (*m_dispatching_block)->active_count());
        fflush(stdout);
        assert(0 && "no more active thread found in the block");
      }
      
      //need to check next step with selective dispatch
      switch(unrolling_factor){
        case 1: 
          actual_dispatch_threads[0] = next_ready_threads[0];
          break;
        case 2: {
          unsigned pi_0 = next_ready_threads[0];
          unsigned pi_1 = next_ready_threads[1]-8; 
          //dispatch smallest indexes
          //get smallest index first
          unsigned min_index = 0;
          min_index = pi_0 < pi_1 ? pi_0 : pi_1;
          //dispatch the smallest index
          if(pi_0 == min_index){
            actual_dispatch_threads[0] = next_ready_threads[0];
          } 
          if(pi_1 == min_index){
            actual_dispatch_threads[1] = next_ready_threads[1];
          } 
          break;
        }
        case 4: {
          unsigned pi_0 = next_ready_threads[0];
          unsigned pi_1 = next_ready_threads[1]-8;
          unsigned pi_2 = next_ready_threads[2]-16;
          unsigned pi_3 = next_ready_threads[3]-24;
          //dispatch smallest indexes
          //get smallest index first
          unsigned min_index = 0;
          min_index = pi_0 < pi_1 ? pi_0 : pi_1;
          min_index = min_index < pi_2 ? min_index : pi_2;
          min_index = min_index < pi_3 ? min_index : pi_3;
          //dispatch the smallest index
          if(pi_0 == min_index){
            actual_dispatch_threads[0] = next_ready_threads[0];
          } 
          if(pi_1 == min_index){
            actual_dispatch_threads[1] = next_ready_threads[1];
          } 
          if(pi_2 == min_index){
            actual_dispatch_threads[2] = next_ready_threads[2];
          } 
          if(pi_3 == min_index){
            actual_dispatch_threads[3] = next_ready_threads[3];
          }
          break;
        }
        default:
          assert(0 && "unrolling factor not supported");
          break;
      }
      //or use SIMT-stack active count m_cgra_core->m_simt_stack->get_active_mask()->count();
      //find next active thread id
      //unsigned tid = next_active_thread(unrolling_factor, unrolling_index);
      //DICE-TODO: simulate read operands for this tid
      //if(g_debug_execution==3 &m_cgra_core_id == get_dice_trace_sampling_core()){
      //  printf("DICE Sim uArch [DISPATCHER]: cycle %d, dispatch and get operands of thread %d for dice block id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , tid ,(*m_dispatching_block)->get_current_metadata()->meta_id);
      //  fflush(stdout);
      //}
      //check scoreboard collision
      bool collision = false;
      for(unsigned unrolling_index=0;unrolling_index<unrolling_factor;unrolling_index++){
        unsigned tid = actual_dispatch_threads[unrolling_index];
        if(tid == unsigned(-1)){
          continue;
        }
        unsigned core_tid = tid+m_cgra_core->get_cta_start_tid((*m_dispatching_block)->get_cta_id());
        if(m_scoreboard->checkCollision(core_tid, (*m_dispatching_block)->get_current_metadata())) {
          collision = true;
          break;
        }
      }
      if(!exec_stalled() && !collision){ //not dispatch if writeback buffer is full f9r current dispatching metadata or cgra_is_stalled
        for(unsigned lane_id=0;lane_id<4;lane_id++){
          unsigned tid = actual_dispatch_threads[lane_id];
          unsigned core_tid = unsigned(-1);
          if(tid != unsigned(-1)){
            core_tid = tid+m_cgra_core->get_cta_start_tid((*m_dispatching_block)->get_cta_id());
            read_operands((*m_dispatching_block)->get_current_metadata(), core_tid);
            if((*m_dispatching_block)->is_parameter_load()){
              //check active mask
              unsigned cta_id = (*m_dispatching_block)->get_cta_id();
              for(unsigned t=0;t<m_cgra_core->get_cta_size(cta_id);t++){
                if((*m_dispatching_block)->active(t) == true){
                  m_scoreboard->reserveRegisters((*m_dispatching_block)->get_current_metadata(), t+m_cgra_core->get_cta_start_tid((*m_dispatching_block)->get_cta_id()));
                }
              }
            } else {
              m_scoreboard->reserveRegisters((*m_dispatching_block)->get_current_metadata(), core_tid);
            }
            if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
              printf("DICE Sim uArch [DISPATCHER]: cycle %d, hw_cta=%d, dispatch and get operands of thread %d for dice block id = %d, lane_id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id() ,core_tid ,(*m_dispatching_block)->get_current_metadata()->meta_id,lane_id);
              fflush(stdout);
            }
            m_last_dispatched_tid[lane_id] = tid;
          } else {
            m_dispatched_bubble_count++;
          }
          m_ready_threads[lane_id].push_back(core_tid);
        }
      }
    }
    else{
      //all threads in the block are dispatched
      //DICE-TODO: simulate writeback
      if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
        printf("DICE Sim uArch [DISPATCH_END]: Cycle %d, hw_cta=%d, Block=%d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , (*m_dispatching_block)->get_cta_id(), (*m_dispatching_block)->get_current_metadata()->meta_id);
        fflush(stdout);
      }
      (*m_dispatching_block)->set_dispatch_done();
      m_dispatched_thread = 0;
      m_dispatched_bubble_count = 0;
      for(unsigned i = 0; i < m_last_dispatched_tid.size(); i++){
        m_last_dispatched_tid[i] = unsigned(-1);
      }
    }
  } else {
    //if(m_cgra_core->get_id()==0 && m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle >1785 && m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle <1790){
    //  printf("DICE-Sim uArch: cycle %d, dispatch() block id = %d, stalled because of\n", (*m_dispatching_block)->get_current_metadata()->meta_id ,m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle );
    //  printf("ready_to_dispatch = %d\n", (*m_dispatching_block)->ready_to_dispatch());
    //  printf("dispatch_done = %d\n", (*m_dispatching_block)->dispatch_done());
    //  printf("decoded = %d\n", (*m_dispatching_block)->decode_done());
    //  printf("metadata_buffer_empty = %d\n", (*m_dispatching_block)->metadata_buffer_empty());
    //}
  }
}

void dispatcher_rfu_t::writeback_cgra(cgra_block_state_t* block, unsigned tid){
  //writeback register from cgra
  //check each output register in the metadata
  dice_metadata* metadata = block->get_current_metadata();
  unsigned num_writeback = 0;
  bool all_valid = true;
  std::map<unsigned, std::set<unsigned>> invalid_regs_map = block->get_current_cfg_block()->map_tid_invalid_writeback_regs;
  //check if tid is in invalid_regs_map
  std::map<unsigned, std::set<unsigned>>::iterator it_map = invalid_regs_map.find(tid);
  std::set<unsigned> invalid_regs;
  if(it_map != invalid_regs_map.end()){
    invalid_regs = invalid_regs_map[tid];
    all_valid = false;
  }

  for(std::list<operand_info>::iterator it = metadata->out_regs.begin(); it != metadata->out_regs.end(); ++it){
    unsigned reg_num = (*it).reg_num();
    unsigned bank_id = reg_number_to_bank_mapping(reg_num,tid,32);
    //check if the writeback register is valid
    if(all_valid) {
      if(bank_id<32){
        if(m_rf_bank_controller[bank_id]->wb_buffer_full() == true){
          printf("**ERROR: cycle %d, core %d, reg %d, RF bank %d is full for tid=%d, please check and make sure no wrieback bank conflict in cgra_block %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,m_cgra_core->get_id(),reg_num, bank_id ,tid, metadata->meta_id);
          fflush(stdout);
        }
        assert(m_rf_bank_controller[bank_id]->wb_buffer_full() == false);
        //push to writeback buffer
        m_rf_bank_controller[bank_id]->push_to_cgra_wb_buffer(tid,block);
      }
      num_writeback++;
      m_cgra_core->incregfile_writes(1);
    } else {
      //check if the register is in the invalid register set
      std::set<unsigned>::iterator it_invalid = invalid_regs.find(reg_num);
      if(it_invalid == invalid_regs.end()){
        if(bank_id<32){
          assert(m_rf_bank_controller[bank_id]->wb_buffer_full() == false);
          //push to writeback buffer
          m_rf_bank_controller[bank_id]->push_to_cgra_wb_buffer(tid,block);
        }
        num_writeback++;
        m_cgra_core->incregfile_writes(1);
      } else {
        //do not push to writeback buffer
        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [WRITEBACK_INVALID]: cycle %d, core %d, reg %d, RF bank %d is invalid for tid=%d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,m_cgra_core->get_id(),reg_num, bank_id ,tid);
          fflush(stdout);
        }
      }
    }
    //clear scoreboard
    m_scoreboard->releaseRegister(tid, reg_num);
  }
  ////cycle every RF bank
  //for(unsigned i = 0; i < m_config->dice_cgra_core_max_rf_banks;i++){
  //  m_rf_bank_controller[i]->cycle();
  //}
  m_num_write_access += num_writeback;
}

bool dispatcher_rfu_t::writeback_ldst(cgra_block_state_t* block, unsigned reg_num, std::set<unsigned> tids){
  for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it){
    unsigned tid = *it;
    unsigned bank_id = reg_number_to_bank_mapping(reg_num,tid,32);
    if(m_rf_bank_controller[bank_id]->ldst_buffer_full() == false){
      if(block->is_parameter_load()){
        assert(bank_id >=32);//constant memory, not gpr memory
        unsigned cta_id = block->get_cta_id();
        for(unsigned t=0;t<m_cgra_core->get_cta_size(cta_id);t++){
          if(block->active(t) == true) 
          {
            unsigned cta_start_tid = m_cgra_core->get_cta_start_tid(cta_id);
            m_scoreboard->releaseRegisterFromLoad(t+cta_start_tid, reg_num);
            m_cgra_core->incregfile_writes(1);
          }
        }
      } else {
        unsigned hw_tid_offset = m_cgra_core->get_cta_start_tid(block->get_cta_id());
        m_scoreboard->releaseRegisterFromLoad(hw_tid_offset+tid, reg_num);
        m_cgra_core->incregfile_writes(1);
      }
      if(bank_id < 32){
        //push to writeback buffer
        m_rf_bank_controller[bank_id]->push_to_ldst_wb_buffer(tid,block);
      }
      //increase load writeback counter
      block->inc_number_of_loads_done();
      if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
        printf("DICE Sim: [WRITEBACK]: cycle %d, load writeback done for thread %d, reg %d, current load done %d, need totoal %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , tid, reg_num, block->get_number_of_loads_done(), block->get_current_cfg_block()->get_num_loads());
        fflush(stdout);
      }
    }
    else if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
      printf("DICE-Sim: [WRITEBACK]: cycle %d, core %d, reg_num %d, RF bank %d conflict from ldst_unit for tid=%d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,m_cgra_core->get_id(),reg_num, bank_id ,tid);
      fflush(stdout);
      return false;
    }
  }
  return true;
}

bool dispatcher_rfu_t::can_writeback_ldst_reg(unsigned bank_id, unsigned count){
  if(m_rf_bank_controller[bank_id]->ldst_buffer_credit() >= count){
    return true;
  }
  return false;
}

bool dispatcher_rfu_t::can_writeback_ldst_regs(std::set<unsigned> regs, std::set<unsigned> tids){
  std::map<unsigned, unsigned> bank_id_count_map;
  //initialize bank_id_count_map
  for(unsigned i = 0; i < m_rf_bank_controller.size(); i++){
    bank_id_count_map[i] = 0;
  }
  //count the number of registers writes in each bank
  for(std::set<unsigned>::iterator it = regs.begin(); it != regs.end(); ++it){
    unsigned reg_num = *it;
    for (std::set<unsigned>::iterator tid = tids.begin(); tid != tids.end(); ++tid){
      unsigned bank_id = reg_number_to_bank_mapping(reg_num,*tid,32);
      if(bank_id < m_rf_bank_controller.size()){
        bank_id_count_map[bank_id]++;
      }
    }
  }
  for(std::map<unsigned, unsigned>::iterator it = bank_id_count_map.begin(); it != bank_id_count_map.end(); ++it){
    unsigned bank_id = it->first;
    unsigned count = it->second;
    if(count > 0 && can_writeback_ldst_reg(bank_id,count) == false){
      return false;
    }
  }
  return true;
}

unsigned dispatcher_rfu_t::next_active_thread(unsigned unrolling_factor, unsigned unrolling_index, unsigned max_coalesce){
  unsigned next_tid=m_last_dispatched_tid[unrolling_index]+1;
  switch(unrolling_factor){
    case 1:
      break;
    case 2:
      if(next_tid == 0) next_tid += 8*unrolling_index; // {0,8}
      else if (next_tid % 8 == 0) next_tid += 8;  //{0,8}, {1,9}, {2,10}, {3,11},....{7,15}, {16,24}, ...
      break;
    case 4:
      if(next_tid == 0) next_tid += 8*unrolling_index; // {0,32,64,96}
      else if (next_tid % 8 == 0) next_tid += 24;  // {0, 8, 16, 24}, {1, 9, 17, 25}, {2, 10, 18, 26}, {3, 11, 19, 27},....{7,15,23,31}, {32,40,48,56}, ...
      break;
    default:
      printf("DICE-Sim uArch: [Error] cycle %d, core %d, unrolling factor %d not supported\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cgra_core->get_id(), unrolling_factor);
      fflush(stdout);
      assert(0 && "unrolling factor not supported, pls use 1,2,4");
      break;
  }
  unsigned hw_cta_id = (*m_dispatching_block)->get_cta_id();
  if(next_tid >= m_cgra_core->get_cta_size(hw_cta_id)){
    printf("DICE-Sim uArch: [Warning] cycle %d, core %d, no more active thread found in the block\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cgra_core->get_id());
    printf("DICE-Sim uArch: current dispatched number = %d, need total = %d \n",get_actual_dispatched_count(), (*m_dispatching_block)->active_count());
    fflush(stdout);
    //return unsigned(-1);
    assert(0);
  }
  while((*m_dispatching_block)->active(next_tid) == false){
    if(next_tid >= m_cgra_core->get_cta_size(hw_cta_id)){
      printf("DICE-Sim uArch: [Warning] cycle %d, core %d, no more active thread found in the block in lane = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cgra_core->get_id(), unrolling_index);
      printf("DICE-Sim uArch: current dispatched number = %d, need total = %d \n",get_actual_dispatched_count(), (*m_dispatching_block)->active_count());
      fflush(stdout);
    }
    next_tid++;
    switch(unrolling_factor){
      case 1:
        break;
      case 2:
        if(next_tid == 0) next_tid += 8*unrolling_index; // {0,8}
        else if (next_tid % 8 == 0) next_tid += 8;  //{0,8}, {1,9}, {2,10}, {3,11},....{7,15}, {16,24}, ...
        break;
      case 4:
        if(next_tid == 0) next_tid += 8*unrolling_index; // {0,32,64,96}
        else if (next_tid % 8 == 0) next_tid += 24;  // {0, 8, 16, 24}, {1, 9, 17, 25}, {2, 10, 18, 26}, {3, 11, 19, 27},....{7,15,23,31}, {32,40,48,56}, ...
        break;
      default:
        printf("DICE-Sim uArch: [Error] cycle %d, core %d, unrolling factor %d not supported\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , m_cgra_core->get_id(), unrolling_factor);
        fflush(stdout);
        assert(0 && "unrolling factor not supported, pls use 1,2,4");
        break;
    }
    if(unrolling_factor == 1){
      assert(next_tid < m_cgra_core->get_cta_size(hw_cta_id));
    }
    if(next_tid >= m_cgra_core->get_cta_size(hw_cta_id)){
      return unsigned(-1);
    }
  }
  return next_tid;
}


void dispatcher_rfu_t::read_operands(dice_metadata *metadata, unsigned tid){
  unsigned num_input_regs = 0;
  //get number of input operands from metadata
  for(std::list<operand_info>::iterator it = metadata->in_regs.begin(); it != metadata->in_regs.end(); ++it){
    if((*it).is_builtin()) continue;
    num_input_regs++;
  }
  m_num_read_access += num_input_regs;
  m_cgra_core->incregfile_reads(num_input_regs);
}

//Scoreboard
void Scoreboard::reserveRegisters(const dice_metadata *metadata, unsigned tid){
  //use iterator
  for(std::list<operand_info>::const_iterator it = metadata->out_regs.begin(); it != metadata->out_regs.end(); ++it){
    if (it->reg_num() > 0) {
      reserveRegister(tid, it->reg_num());
    }
  }

  // Keep track of long operations
  if (metadata->load_destination_regs.size()!=0) {
    for (std::list<operand_info>::const_iterator it = metadata->load_destination_regs.begin(); it != metadata->load_destination_regs.end(); ++it) {
      if (it->reg_num() > 0) {
        longopregs[tid].insert(it->reg_num());
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - tid:%d, reg: %d\n",tid, it->reg_num());
      }
    }
  }
}

void Scoreboard::releaseRegisters(const dice_metadata *metadata, unsigned tid){
  for(std::list<operand_info>::const_iterator it = metadata->out_regs.begin(); it != metadata->out_regs.end(); ++it){
    if (it->reg_num() > 0) {
      releaseRegister(tid, it->reg_num());
    }
  }
}

void Scoreboard::releaseRegistersFromLoad(const dice_metadata *metadata, unsigned tid){
  for (std::list<operand_info>::const_iterator it = metadata->load_destination_regs.begin(); it != metadata->load_destination_regs.end(); ++it) {
    if (it->reg_num() > 0) {
      longopregs[tid].erase(it->reg_num());
      SHADER_DPRINTF(SCOREBOARD, "Release New longopreg - tid:%d, reg: %d\n",tid, it->reg_num());
    }
  }
}

void Scoreboard::releaseRegisterFromLoad(unsigned tid, unsigned reg_num){
  if (reg_num > 0) {
    assert(longopregs[tid].find(reg_num) != longopregs[tid].end());
    longopregs[tid].erase(reg_num);
    SHADER_DPRINTF(SCOREBOARD, "Release New longopreg - tid:%d, reg: %d\n",tid, reg_num);
  }
}

bool Scoreboard::checkCollision(unsigned tid, const dice_metadata *metadata) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;
  for (std::list<operand_info>::const_iterator it = metadata->out_regs.begin(); it != metadata->out_regs.end(); ++it) {
    inst_regs.insert(it->reg_num());
  }

  for(std::list<operand_info>::const_iterator it = metadata->load_destination_regs.begin(); it != metadata->load_destination_regs.end(); ++it){
    inst_regs.insert(it->reg_num());
  }

  for (std::list<operand_info>::const_iterator it = metadata->in_regs.begin(); it != metadata->in_regs.end(); ++it) {
    if(it->is_builtin()){
      continue;
    }
    inst_regs.insert(it->reg_num());
  }

  if (metadata->branch) {
    if (!metadata->uni_bra) {
      inst_regs.insert(metadata->branch_pred->reg_num());
    }
  }
  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table[tid].find(*it2) != reg_table[tid].end()) {
      //printf("DICE-Sim uArch: cycle %d, collision detected for thread %d, reg %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, tid, *it2);
      return true;
    }
  }

  std::set<unsigned int>::const_iterator it3;
  for(it3 = longopregs[tid].begin(); it3 != longopregs[tid].end(); it3++){
    if(inst_regs.find(*it3) != inst_regs.end()){
      //printf("DICE-Sim uArch: cycle %d, collision detected for thread %d, reg %d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, tid, *it3);
      return true;
    }
  }
  return false;
}


void ldst_unit::writeback_cgra(){
  // process next instruction that is going to writeback
  if (m_next_cgra_writeback!=NULL) {
    bool can_writeback = m_dispatcher_rfu->can_writeback_ldst_regs(m_next_cgra_writeback->get_regs_num(),m_next_cgra_writeback->get_tids());
    std::set<unsigned> writeback_regs = m_next_cgra_writeback->get_regs_num();
    //assert(can_writeback);//this should always has the highest priority//no longer the case
    //check if it's write allocate
    if(can_writeback==false){
      std::set<unsigned> tids = m_next_cgra_writeback->get_tids();
      std::set<unsigned> writeback_regs = m_next_cgra_writeback->get_regs_num();
      if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core() && (m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle)>45000){
        printf("DICE Sim uArch: [LDST_UNIT_WRITEBACK_FAIL]: cycle %d, tid =", m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
        for (std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
          printf(" %d",*it);
        }
        printf(", reg %d, addr 0x%08x\n",writeback_regs, m_next_cgra_writeback->get_addr());
        fflush(stdout);
      }
    }
    if(m_next_cgra_writeback->get_access_type() == L1_WR_ALLOC_R){
      can_writeback = false;
    }
    if (can_writeback) {
      for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
        if(m_dispatcher_rfu->writeback_ldst(m_next_cgra_writeback->get_cgra_block_state(), *it, m_next_cgra_writeback->get_tids())){
        } else {
          assert(0 && "ERROR detect no RF conflict but actually RF conflict! ");
        }
      }
      //DICE-TODO handle load to shared space
      //bool block_completed = false;
      //if (m_next_cgra_writeback->get_space() != shared_space) {
      //  assert(m_pending_writes[m_next_cgra_writeback->get_tid()][m_next_cgra_writeback->get_reg_num()] > 0);
      //  unsigned still_pending = --m_pending_writes[m_next_cgra_writeback->get_tid()][m_next_cgra_writeback->get_reg_num()];
      //  if (!still_pending) {
      //    m_pending_writes[m_next_cgra_writeback->get_tid()].erase(m_next_cgra_writeback->get_reg_num());
      //    block_completed = true;
      //  }
      //} 
      //else {  // shared
      //  m_scoreboard->releaseRegister(m_next_wb.warp_id(),
      //                                m_next_wb.out[r]);
      //  block_completed = true;
      //}
      //if (block_completed) {
      //  m_next_cgra_writeback->get_cgra_block_state()->set_writeback_done();
      //}
      delete m_next_cgra_writeback;
      m_next_cgra_writeback=NULL;
      m_last_inst_gpu_sim_cycle = m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ;
      m_last_inst_gpu_tot_sim_cycle = m_cgra_core->get_gpu()->gpu_tot_sim_cycle;
    }
  }

  unsigned serviced_client = -1;
  for (unsigned c = 0; (m_next_cgra_writeback==NULL) && (c < m_num_writeback_clients);
       c++) {
    unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;
    switch (next_client) {
      case 0:  // shared memory DICE-TODO
        //if (!m_pipeline_reg[0]->empty()) {
        //  m_next_wb = *m_pipeline_reg[0];
        //  if (m_next_wb.isatomic()) {
        //    m_next_wb.do_atomic();
        //    m_core->decrement_atomic_count(m_next_wb.warp_id(),
        //                                   m_next_wb.active_count());
        //  }
        //  m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
        //  m_pipeline_reg[0]->clear();
        //  serviced_client = next_client;
        //}
        break;
      case 1:  // texture response
        if (m_L1T->access_ready()) {
          mem_fetch *mf = m_L1T->next_access();
          m_next_cgra_writeback = mf;
          //delete mf;
          serviced_client = next_client;
        }
        break;
      case 2:  // const cache response
        if (m_L1C->access_ready()) {
          mem_fetch *mf = m_L1C->next_access();
          m_next_cgra_writeback = mf;
          //delete mf;
          serviced_client = next_client;
        }
        break;
      case 3:  // global/local
        if (m_next_global_cgra!=NULL) {
          m_next_cgra_writeback = m_next_global_cgra;
          //if (m_next_global->isatomic()) {
          //  m_core->decrement_atomic_count(
          //      m_next_global->get_wid(),
          //      m_next_global->get_access_warp_mask().count());
          //}
          //delete m_next_global_cgra;
          m_next_global_cgra = NULL;
          serviced_client = next_client;
        }
        break;
      case 4:
        if (m_L1D && m_L1D->access_ready()) {
          mem_fetch *mf = m_L1D->next_access();
          m_next_cgra_writeback = mf;
          //delete mf;
          serviced_client = next_client;
        }
        break;
      default:
        assert(0);abort();
    }
  }
  // update arbitration priority only if:
  // 1. the writeback buffer was available
  // 2. a client was serviced
  if (serviced_client != (unsigned)-1) {
    m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients;
  }
}

void ldst_unit::process_request(){
}

void ldst_unit::cycle_cgra(){
  //writeback incoming missing cache data first
  writeback_cgra();

  //run each cache cycle
  m_L1T->cycle();
  m_L1C->cycle();
  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle_cgra();
  }

  m_dice_mem_request_queue->update_coaleasing_counter();

  //process and writeback new memory access
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
  mem_stage_access_type type;
  //contant cache cycle, round robin arbiter from all load ports to check all load request to contant cache
  unsigned contant_port = m_dice_mem_request_queue->get_next_process_port_constant();
  if(contant_port != unsigned(-1)){
    mem_access_t access = m_dice_mem_request_queue->get_request(contant_port);
    cgra_block_state_t* cgra_block = access.get_cgra_block_state();
    constant_cycle_cgra(cgra_block, access, rc_fail, type);
    m_dice_mem_request_queue->set_last_processed_port_constant(contant_port);
  }

  rc_fail = NO_RC_FAIL;
  //texture cache cycle, round robin arbiter from all load ports to check all load request to contant cache
  unsigned texture_port = m_dice_mem_request_queue->get_next_process_port_texture();
  if(texture_port != unsigned(-1)){
    mem_access_t access = m_dice_mem_request_queue->get_request(texture_port);
    cgra_block_state_t* cgra_block = access.get_cgra_block_state();
    texture_cycle_cgra(cgra_block, access, rc_fail, type);
    m_dice_mem_request_queue->set_last_processed_port_texture(texture_port);
  }

  rc_fail = NO_RC_FAIL;
  //memory cycle, round robin arbiter from all load and store ports to check all load request to contant cache
  //can process L1_bank numbers request per cycle
  std::list<unsigned> issued_ports; //store ports that have been issued
  issued_ports.push_back(unsigned(-1)); //initialize with -1 to avoid empty list
  for (int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {  
    unsigned memory_port = m_dice_mem_request_queue->get_next_process_port_memory();
    if(std::find(issued_ports.begin(), issued_ports.end(), memory_port) != issued_ports.end()) {
      break; // port has already been issued
    }
    if(memory_port != unsigned(-1)){
      //printf("DICE-Sim uArch: [MEMORY]: cycle %d, memory port %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , memory_port);
      mem_access_t access = m_dice_mem_request_queue->get_request(memory_port);
      cgra_block_state_t* cgra_block = access.get_cgra_block_state();
      memory_cycle_cgra(cgra_block, access, rc_fail, type);
      m_dice_mem_request_queue->set_last_processed_port_memory(memory_port);
      issued_ports.push_back(memory_port); //add to issued ports
    }
  }
  //shared memory cycle is modeled as latency for each thread directly
  //
  std::map<unsigned, std::map<new_addr_type, unsigned>> bank_accs;  // bank -> word address -> access count
  bool accept_new_access = true;
  bool clear_current_access = false;
  if(shared_cycle_count_set()){
    accept_new_access = false;
    clear_current_access = decrease_shared_cycle_count();
  }
  if(accept_new_access){
    unsigned total_accesses = 0;
    for (int k = 0; k < m_config->num_shmem_bank; k++) {  
      unsigned shared_port = m_dice_mem_request_queue->get_next_process_port_shared();
      if(shared_port != unsigned(-1)){
        mem_access_t access = m_dice_mem_request_queue->get_request(shared_port);
        cgra_block_state_t* cgra_block = access.get_cgra_block_state();
        new_addr_type addr = access.get_addr();
        unsigned bank = m_config->shmem_bank_func(addr);
        new_addr_type word = line_size_based_tag_func_cgra(addr, m_config->WORD_SIZE);
        bank_accs[bank][word]++;
      }
    }
    if (m_config->shmem_limited_broadcast) {
      // step 2: look for and select a broadcast bank/word if one occurs
      bool broadcast_detected = false;
      new_addr_type broadcast_word = (new_addr_type)-1;
      unsigned broadcast_bank = (unsigned)-1;
      std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
      for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
        unsigned bank = b->first;
        std::map<new_addr_type, unsigned> &access_set = b->second;
        std::map<new_addr_type, unsigned>::iterator w;
        for (w = access_set.begin(); w != access_set.end(); ++w) {
          if (w->second > 1) {
            // found a broadcast
            broadcast_detected = true;
            broadcast_bank = bank;
            broadcast_word = w->first;
            break;
          }
        }
        if (broadcast_detected) break;
      }
    
      // step 3: figure out max bank accesses performed, taking account of
      // broadcast case
      unsigned max_bank_accesses = 0;
      for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
        unsigned bank_accesses = 0;
        std::map<new_addr_type, unsigned> &access_set = b->second;
        std::map<new_addr_type, unsigned>::iterator w;
        for (w = access_set.begin(); w != access_set.end(); ++w)
          bank_accesses += w->second;
        if (broadcast_detected && broadcast_bank == b->first) {
          for (w = access_set.begin(); w != access_set.end(); ++w) {
            if (w->first == broadcast_word) {
              unsigned n = w->second;
              assert(n > 1);  // or this wasn't a broadcast
              assert(bank_accesses >= (n - 1));
              bank_accesses -= (n - 1);
              break;
            }
          }
        }
        if (bank_accesses > max_bank_accesses)
          max_bank_accesses = bank_accesses;
      }
    
      // step 4: accumulate
      total_accesses += max_bank_accesses;
    } else {
      // step 2: look for the bank with the maximum number of access to
      // different words
      unsigned max_bank_accesses = 0;
      std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
      for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
        max_bank_accesses =
            std::max(max_bank_accesses, (unsigned)b->second.size());
      }
    
      // step 3: accumulate
      total_accesses += max_bank_accesses;
    }
    //execute shared memory access for all ports
    if (total_accesses == 1) { // no bank conflict
      for (int k = 0; k < m_config->num_shmem_bank; k++) {  
        unsigned shared_port = m_dice_mem_request_queue->get_next_process_port_shared();
        if(shared_port != unsigned(-1)){
          mem_access_t access = m_dice_mem_request_queue->get_request(shared_port);
          cgra_block_state_t* cgra_block = access.get_cgra_block_state();
          shared_cycle_cgra(cgra_block, access, rc_fail, type);
          m_dice_mem_request_queue->set_last_processed_port_shared(shared_port);
        }
      }
    } else {
      set_shared_cycle_count(total_accesses);
    }
  }

  if(clear_current_access){
    //clear current access
    for (int k = 0; k < m_config->num_shmem_bank; k++) {  
      unsigned shared_port = m_dice_mem_request_queue->get_next_process_port_shared();
      if(shared_port != unsigned(-1)){
        mem_access_t access = m_dice_mem_request_queue->get_request(shared_port);
        cgra_block_state_t* cgra_block = access.get_cgra_block_state();
        shared_cycle_cgra(cgra_block, access, rc_fail, type);
        m_dice_mem_request_queue->set_last_processed_port_shared(shared_port);
      }
    }
  }

  rc_fail = NO_RC_FAIL;


  //coaleascing
  m_dice_mem_request_queue->coalesce_cycle();


  //process interconnect data
  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      if (m_L1T->fill_port_free()) {
        m_L1T->fill(mf, m_cgra_core->get_gpu()->gpu_sim_cycle+ m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
          m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
        std::set<unsigned> tids = mf->get_tids();
        std::set<unsigned> writeback_regs = mf->get_regs_num();
        m_L1C->fill(mf, m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
        if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, const cache fill for tid =", m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
          for (std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
            printf(" %d",*it);
          }
          printf(", reg ");
          for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
            printf(" %d ",*it);
          }
          printf(", addr 0x%08x\n", mf->get_addr());
          fflush(stdout);
        }
        m_response_fifo.pop_front();
      }
    } else {
      if (mf->get_type() == WRITE_ACK ||(m_config->gpgpu_perfect_mem && mf->get_is_write())) {
        m_cgra_core->store_ack(mf);
        m_response_fifo.pop_front();
        delete mf;
      } else {
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only

        bool bypassL1D = false;
        //DICE-TODO
        //if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
        //  bypassL1D = true;
        //} else if (mf->get_access_type() == GLOBAL_ACC_R ||
        //           mf->get_access_type() ==
        //               GLOBAL_ACC_W) {  // global memory access
        //  if (m_cgra_core->get_config()->gmem_skip_L1D) bypassL1D = true;
        //}
        if (bypassL1D) {
          if (m_next_global_cgra == NULL) {
            mf->set_status(IN_SHADER_FETCHED,
              m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
            m_next_global_cgra = mf;
          }
        } else {
          if (m_L1D->fill_port_free()) {
            //mf->print(stdout);
            //fflush(stdout);
            m_L1D->fill(mf, m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }
}


mem_stage_stall_type ldst_unit::process_memory_access_queue_cgra(cache_t *cache, cgra_block_state_t* cgra_block, mem_access_t access) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (!cache->data_port_free()) return DATA_PORT_STALL;
  mem_fetch *mf = m_mf_allocator->alloc_cgra(cgra_block, access ,m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
  std::list<cache_event> events;
  enum cache_request_status status = cache->access(mf->get_addr(), mf, m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle, events);
  return process_cache_access_cgra(cache, cgra_block, events, mf, status);
}

mem_stage_stall_type ldst_unit::process_cache_access_cgra(
    cache_t *cache, cgra_block_state_t* cgra_block,
    std::list<cache_event> &events, mem_fetch *mf,
    enum cache_request_status status) {
  mem_stage_stall_type result = NO_RC_FAIL;
  bool write_sent = was_write_sent(events);
  bool read_sent = was_read_sent(events);
  if (write_sent) {
    unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                           ? (mf->get_data_size() / SECTOR_SIZE)
                           : 1;

    for (unsigned i = 0; i < inc_ack; ++i) cgra_block->inc_store_req();
  }
  if (status == HIT) {
    std::set<unsigned> tids = mf->get_tids();
    bool can_writeback = m_dispatcher_rfu->can_writeback_ldst_regs(mf->get_regs_num(),tids);
    if(can_writeback){
      std::set<unsigned> writeback_regs = mf->get_regs_num();
      for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
        if(m_dispatcher_rfu->writeback_ldst(cgra_block, *it, tids)){
        } else {
          assert(0 && "ERROR detect no RF conflict but actually RF conflict! ");
        }
      }
      assert(!read_sent);
      if (mf->is_write()) {
        //inc write ack
        unsigned stores_done_inc = tids.size();
        for(int i=0; i<stores_done_inc; i++){
          cgra_block->inc_number_of_stores_done();
        }
        if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, writeback done for thread ", m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
          for (std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
            printf(" %d",*it);
          }
          printf(", reg ");
          for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
            printf(" %d",*it);
          }
          printf(", current store done %d, need total %d\n",cgra_block->get_number_of_stores_done(), cgra_block->get_current_cfg_block()->get_num_stores());
          fflush(stdout);
        } 
      }
      //direct writeback
      //cgra_block->get_current_cfg_block()->pop_mem_access(mf->get_ldst_port_num());
      m_dice_mem_request_queue->pop_request(mf->get_ldst_port_num());
      if (!write_sent) delete mf;
    }
  } else if (status == RESERVATION_FAIL) {
    result = BK_CONF;
    assert(!read_sent);
    assert(!write_sent);
    delete mf;
  } else {
    assert(status == MISS || status == HIT_RESERVED);
    //cgra_block->get_current_cfg_block()->pop_mem_access(mf->get_ldst_port_num());
    m_dice_mem_request_queue->pop_request(mf->get_ldst_port_num());
  }
  //if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL;
  return result;
}

void cgra_block_state_t::inc_number_of_stores_done() {
  m_num_stores_done++;
}


mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache_cgra(l1_cache *cache, cgra_block_state_t* cgra_block, mem_access_t access) {
  mem_stage_stall_type result = NO_RC_FAIL;
  //bool no_bk_conf = 0;
  if (m_config->m_L1D_config.l1_latency > 0) {
    //for (int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {  // We can handle at max l1_banks reqs per cycle // move to top level
      mem_fetch *mf = m_mf_allocator->alloc_cgra(cgra_block, access, m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
      std::set<unsigned> tids = mf->get_tids();
      std::set<unsigned> writeback_regs = mf->get_regs_num();
      unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr());
      assert(bank_id < m_config->m_L1D_config.l1_banks);

      //debug info
      for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
        unsigned tid = *it;
        //if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
        //  printf("DICE Sim uArch: [LDST_UNIT_L1D_LATENCY_QUEUE_PUSH]: Cycle %d, Push access(tid=%d,block=%d,addr=0x%08x) to latency queue \n regs =",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , tid, cgra_block->get_current_metadata()->meta_id, mf->get_addr());
        //  
        //  for(std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
        //    printf(" %d",*it);
        //  }
        //  printf("\n");
        //  fflush(stdout);
        //}
      }

      if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
          NULL) {
        l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;

        if (mf->is_write()) {
          unsigned inc_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf->get_data_size() / SECTOR_SIZE)
                  : 1;

          for (unsigned i = 0; i < inc_ack; ++i) cgra_block->inc_store_req();
        }
        //direct writeback
        //cgra_block->get_current_cfg_block()->pop_mem_access(mf->get_ldst_port_num());
        m_dice_mem_request_queue->pop_request(mf->get_ldst_port_num());
      } else {
        result = BK_CONF;
        for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
          unsigned tid = *it;
          if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
            printf("DICE Sim uArch: [LDST_UNIT_L1D_LATENCY_QUEUE_STALL]: Cycle %d, Bank Conflict for access(tid=%d,block=%d,addr=0x%08x)\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , tid, cgra_block->get_current_metadata()->meta_id, mf->get_addr());
            fflush(stdout);
          }
        }
        delete mf;
      }
    //}
    //if (no_bk_conf==0 && result != BK_CONF) result = COAL_STALL;
    return result;
  } else {
    mem_fetch *mf = m_mf_allocator->alloc_cgra(cgra_block, access, m_cgra_core->get_gpu()->gpu_sim_cycle + m_cgra_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(
        mf->get_addr(), mf,
        m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle ,
        events);
    return process_cache_access_cgra(cache,cgra_block, events, mf, status);
  }
}

void ldst_unit::L1_latency_queue_cycle_cgra() {
  for (int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    if ((l1_latency_queue[j][0]) != NULL) {
      mem_fetch *mf_next = l1_latency_queue[j][0];
      std::set<unsigned> tids = mf_next->get_tids();
      std::list<cache_event> events;
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle,
                        events);

      bool write_sent = was_write_sent(events);
      bool read_sent = was_read_sent(events);

      if (status == HIT) {        
        //check if writeback can be processed
        //writeback
        if(!mf_next->is_write()){
          bool can_writeback = m_dispatcher_rfu->can_writeback_ldst_regs(mf_next->get_regs_num(),tids);
          if(can_writeback){
            std::set<unsigned> writeback_regs = mf_next->get_regs_num();
            for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
              if(m_dispatcher_rfu->writeback_ldst(mf_next->get_cgra_block_state(), *it, tids)){
              } else {
                assert(0 && "ERROR detect no RF conflict but actually RF conflict! ");
              }
            }
          } else {
            return;
          }
        }
        assert(!read_sent);

        l1_latency_queue[j][0] = NULL;
        if (mf_next->is_write()) {
          std::set<unsigned> writeback_regs = mf_next->get_regs_num();
          //inc write ack
          unsigned stores_done_inc = tids.size();
          for (int i=0; i<stores_done_inc; i++){
            mf_next->get_cgra_block_state()->inc_number_of_stores_done();
          }
          if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
            for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
              printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, writeback done for thread %d, regs =",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , *it);
              for(std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it){
                printf(" %d",*it);
              }
              printf(", current store done %d, need total %d\n",mf_next->get_cgra_block_state()->get_number_of_stores_done(), mf_next->get_cgra_block_state()->get_current_cfg_block()->get_num_stores());
              fflush(stdout);
            }
          } 
        } 
        // For write hit in WB policy
        if (mf_next->is_write() && !write_sent) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;

          mf_next->set_reply();

          for (unsigned i = 0; i < dec_ack; ++i) m_cgra_core->store_ack(mf_next);
        }
        if (!write_sent) delete mf_next;

      } else if (status == RESERVATION_FAIL) {
        assert(!read_sent);
        assert(!write_sent);
        if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
          for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
            printf("DICE Sim uArch: [LDST_UNIT_L1D_ACCESS_STALL]: Cycle %d, Bank Conflict for access(tid=%d,block=%d,addr=0x%08x)\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , *it, mf_next->get_cgra_block_state()->get_current_metadata()->meta_id, mf_next->get_addr());
          }
          fflush(stdout);
        }
      } else {
        assert(status == MISS || status == HIT_RESERVED);
        l1_latency_queue[j][0] = NULL;
        if(mf_next->is_write()){
          unsigned stores_done_inc = tids.size();
          for(int i=0; i<stores_done_inc; i++){
            mf_next->get_cgra_block_state()->inc_number_of_stores_done();
          }
          if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
            for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
              printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, Store ack for thread %d, addr 0x%08x, number_of_stores_done = %d, need total = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , *it, mf_next->get_addr(), mf_next->get_cgra_block_state()->get_number_of_stores_done(), mf_next->get_cgra_block_state()->get_current_cfg_block()->get_num_stores());
            }
            fflush(stdout);
          }
        }
      }
    }

    for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
         ++stage)
      if (l1_latency_queue[j][stage] == NULL) {
        l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
        l1_latency_queue[j][stage + 1] = NULL;
      }
  }
}


void cgra_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  mf->get_cgra_block_state()->dec_store_req();
}

bool ldst_unit::constant_cycle_cgra(cgra_block_state_t *cgra_block, mem_access_t access, mem_stage_stall_type &rc_fail,
                               mem_stage_access_type &fail_type) {
  if (!cgra_block->ready_to_dispatch() || ((access.get_type() != CONST_ACC_R) ))
    return true;
  mem_stage_stall_type fail;
  std::set<unsigned> tids = access.get_tids();
  fail = process_memory_access_queue_cgra(m_L1C, cgra_block, access);

  if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
    unsigned count = tids.size();
    for (std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
      printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, const cache access for thread %d, addr = 0x%04x, block = %d, status = %s\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , *it, access.get_addr(), cgra_block->get_current_metadata()->meta_id, mem_stage_stall_type_str(fail));
    }
    fflush(stdout);
  }

  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = C_MEM;
    if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
      m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
                                             // conflict, but this maintains
                                             // previous behavior.
    }
  } else {
    return true;
  }
  return false;  // DICE-TODO: what does this indicate?
}

bool ldst_unit::texture_cycle_cgra(cgra_block_state_t *cgra_block, mem_access_t access, mem_stage_stall_type &rc_fail,
  mem_stage_access_type &fail_type) {
  if (!cgra_block->ready_to_dispatch() || ((access.get_type() != TEXTURE_ACC_R) ))
    return true;
  mem_stage_stall_type fail = process_memory_access_queue_cgra(m_L1T, cgra_block, access);
  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = T_MEM;
  } else{
    return true;
  }
  return false;  // DICE-TODO: what does this indicate?
}

bool ldst_unit::shared_cycle_cgra(cgra_block_state_t *cgra_block, mem_access_t access, mem_stage_stall_type &rc_fail,
                             mem_stage_access_type &fail_type) {
  if (access.get_space() != shared_space) return true;

  //cgra_block->get_current_cfg_block()->pop_mem_access(access->get_ldst_port_num());
  std::set<unsigned> writeback_regs = access.get_ldst_regs();
  std::set<unsigned> tids = access.get_tids();
  if(!access.is_write()){
    if(m_dispatcher_rfu->can_writeback_ldst_regs(writeback_regs,tids)){
      for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
        if(m_dispatcher_rfu->writeback_ldst(cgra_block, *it, tids)){
        } else {
          assert(0 && "ERROR detect no RF conflict but actually RF conflict! ");
        }
      }
      m_dice_mem_request_queue->pop_request(access.get_ldst_port_num());
    } 
  } else {
    m_dice_mem_request_queue->pop_request(access.get_ldst_port_num());
  }
  //increment writeback or store num
  if(access.is_write()){
    unsigned stores_done_inc = tids.size();
    for(int i=0; i<stores_done_inc; i++){
      cgra_block->inc_number_of_stores_done();
    }
    if(g_debug_execution==3 &m_cgra_core_id == m_cgra_core->get_dice_trace_sampling_core()){
      for (std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it) {
        printf("DICE Sim uArch: [LDST_UNIT]: cycle %d, writeback done for thread %d, reg:", m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , *it);
        for (std::set<unsigned>::iterator it = writeback_regs.begin(); it != writeback_regs.end(); ++it) {
          printf(" %d",*it);
        }
        printf(", current store done %d, need total %d\n",cgra_block->get_number_of_stores_done(), cgra_block->get_current_cfg_block()->get_num_stores());
      }
      fflush(stdout);
    }
  }
  m_stats->gpgpu_n_shmem_bank_access[m_cgra_core_id]++;

  bool stall = false;
  if (stall) {
    fail_type = S_MEM;
    rc_fail = BK_CONF;
  } else
    rc_fail = NO_RC_FAIL;
  return !stall;
}


bool ldst_unit::memory_cycle_cgra(cgra_block_state_t *cgra_block, mem_access_t access,
                             mem_stage_stall_type &stall_reason,
                             mem_stage_access_type &access_type) {
  if (!cgra_block->ready_to_dispatch()  || ((access.get_space() != global_space) &&
                       (access.get_space() != local_space) &&
                       (access.get_space() != param_space_local)))
    return true;

  mem_stage_stall_type stall_cond = NO_RC_FAIL;
  bool bypassL1D = false;
  //DICE-TODO: not bypass L1D for now
  //if (CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL)) {
  //  bypassL1D = true;
  //} else 
  //{
  //  if (access->get_space().is_global()) {  // global memory access
  //    // skip L1 cache if the option is enabled
  //    if (m_cgra_core->get_config()->gmem_skip_L1D )//&& (CACHE_L1 != inst.cache_op))
  //      bypassL1D = true;
  //  }
  //}
  //if (bypassL1D) {
  //  // bypass L1 cache
  //  unsigned control_size =
  //      inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
  //  unsigned size = access.get_size() + control_size;
  //  // printf("Interconnect:Addr: %x, size=%d\n",access.get_addr(),size);
  //  if (m_icnt->full(size, inst.is_store() || inst.isatomic())) {
  //    stall_cond = ICNT_RC_FAIL;
  //  } else {
  //    mem_fetch *mf =
  //        m_mf_allocator->alloc(inst, access,
  //                              m_core->get_gpu()->gpu_sim_cycle +
  //                                  m_core->get_gpu()->gpu_tot_sim_cycle);
  //    m_icnt->push(mf);
  //    inst.accessq_pop_back();
  //    // inst.clear_active( access.get_warp_mask() );
  //    if (inst.is_load()) {
  //      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
  //        if (inst.out[r] > 0)
  //          assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0);
  //    } else if (inst.is_store())
  //      m_cgra_core->inc_store_req(inst.warp_id());
  //  }
  //} else 
  {
    //assert(CACHE_UNDEFINED != inst.cache_op);
    stall_cond = process_memory_access_queue_l1cache_cgra(m_L1D, cgra_block,access);
  }
  //if (!cgra_block->get_current_cfg_block()->accessq_empty() && stall_cond == NO_RC_FAIL)
  //  stall_cond = COAL_STALL;
  if (stall_cond != NO_RC_FAIL) {
    stall_reason = stall_cond;
    bool iswrite = access.is_write();
    if (access.get_space().is_local())
      access_type = (iswrite) ? L_MEM_ST : L_MEM_LD;
    else
      access_type = (iswrite) ? G_MEM_ST : G_MEM_LD;
  } else {
    return true;
  }
  return false;
}


void ldst_unit::dice_push_accesses(dice_cfg_block_t *cfg_block,cgra_block_state_t* cgra_block){ 
  std::vector<std::list<mem_access_t>> accessq = cfg_block->get_accessq();
  for(unsigned i = 0; i < accessq.size(); i++){
    unsigned num_access = 1;
    if(!cfg_block->get_metadata()->is_parameter_load) {
      assert(accessq[i].size() <=1 ); //otherwise overflow
    } else {
      num_access = accessq[i].size();
    }
    if(accessq[i].empty()) continue;
    mem_access_t access = accessq[i].front();
    cfg_block->pop_mem_access(i);
    //accessq[i].pop_front();
    access.assign_cgra_block_state(cgra_block);
    assert(access.get_cgra_block_state() != NULL);
    //check global memory access
    bool to_coalesce = false;
    if(m_config->dice_ldst_unit_enable_temporal_coalescing){
      if((access.get_space() == global_space) || (access.get_space() == local_space) || (access.get_space() == param_space_local)){
        to_coalesce = true;
      }
    }

    if(to_coalesce){
      if(i< (accessq.size()/2)){
        m_dice_mem_request_queue->push_ld_request_pre_coalesce(access,i);
      } else {
        m_dice_mem_request_queue->push_st_request_pre_coalesce(access,i-accessq.size()/2);
      }
    } else {
      if(i< (accessq.size()/2)){
        m_dice_mem_request_queue->push_ld_request(access,i);
      } else {
        m_dice_mem_request_queue->push_st_request(access,i-accessq.size()/2);
      }
    }
    //test
    //access.print(stdout);
    num_access--;
    if(cfg_block->get_metadata()->is_parameter_load){
      while(num_access){
        accessq = cfg_block->get_accessq();
        mem_access_t access = accessq[i].front();
        cfg_block->pop_mem_access(i);
        //accessq[i].pop_front();
        access.assign_cgra_block_state(cgra_block);
        assert(access.get_cgra_block_state() != NULL);
        if(i< (accessq.size()/2)){
          m_dice_mem_request_queue->push_ld_request(access,i);
        } else {
          m_dice_mem_request_queue->push_st_request(access,i-accessq.size()/2);
        }
        num_access--;
        //test
        //access.print(stdout);
      }
    }
  }
}

bool ldst_unit::one_mem_access_queue_full(unsigned port_id){
  return m_dice_mem_request_queue->is_full(port_id);
}

bool ldst_unit::mem_access_queue_full(){
  for(unsigned i = 0; i < (m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports); i++){
    if(one_mem_access_queue_full(i)){
      return true;
    }
  }
  return false;
}

dice_mem_request_queue::dice_mem_request_queue(const shader_core_config *config, ldst_unit* ldst_unit){
  m_config = config;
  m_ldst_unit = ldst_unit;
  m_ld_req_queue_pre_coalesce.resize(m_config->dice_cgra_core_num_ld_ports);
  m_st_req_queue_pre_coalesce.resize(m_config->dice_cgra_core_num_st_ports);
  m_ld_req_queue.resize(m_config->dice_cgra_core_num_ld_ports);
  m_st_req_queue.resize(m_config->dice_cgra_core_num_st_ports);
  m_ld_port_credit.resize(m_config->dice_cgra_core_num_ld_ports);
  m_st_port_credit.resize(m_config->dice_cgra_core_num_st_ports);
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    m_ld_port_credit[i] = m_config->dice_cgra_core_num_ld_ports_queue_size;
    m_st_port_credit[i] = m_config->dice_cgra_core_num_st_ports_queue_size;
  }
  m_last_processed_port_contant = m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports-1;
  m_last_processed_port_texture = m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports-1;
  m_last_processed_port_memory = m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports-1;
  m_last_processed_port_shared = m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports-1;
  enable_temporal_coaleascing = m_config->dice_ldst_unit_enable_temporal_coalescing;
  temporal_coalescing_interval = m_config->dice_ldst_unit_temporal_coalescing_interval;
  temporal_coalescing_max_cmd = m_config->dice_ldst_unit_temporal_coalescing_max_cmd;
  m_ld_coalescing_counter.resize(m_config->dice_cgra_core_num_ld_ports);
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    m_ld_coalescing_counter[i] = 0;
  }
  m_st_coalescing_counter.resize(m_config->dice_cgra_core_num_st_ports);
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_st_ports; i++){
    m_st_coalescing_counter[i] = 0;
  }
}

unsigned dice_mem_request_queue::get_next_process_port_constant() {
  //check which port has contant memory request
  std::vector<unsigned> port_has_contant_request;
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if(!m_ld_req_queue[i].empty()){
      mem_access_t access = m_ld_req_queue[i].front();
      if(access.get_type() == CONST_ACC_R){
        port_has_contant_request.push_back(i);
      }
    }
  }
  if(port_has_contant_request.empty()){
    return unsigned(-1);
  }
  if(port_has_contant_request.size() == 1){
    return port_has_contant_request[0];
  }
  //start from last processed port
  unsigned next_port = (m_last_processed_port_contant+1) % m_config->dice_cgra_core_num_ld_ports;
  while(true){
    if(std::find(port_has_contant_request.begin(), port_has_contant_request.end(), next_port) != port_has_contant_request.end()){
      return next_port;
    }
    next_port = (next_port+1) % m_config->dice_cgra_core_num_ld_ports;
    if(next_port == m_last_processed_port_contant){
      return unsigned(-1);
    }
  }
}

unsigned dice_mem_request_queue::get_next_process_port_texture() {
  //check which port has texture memory request
  std::vector<unsigned> port_has_texture_request;
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if(!m_ld_req_queue[i].empty()){
      mem_access_t access = m_ld_req_queue[i].front();
      if(access.get_type() == TEXTURE_ACC_R){
        port_has_texture_request.push_back(i);
      }
    }
  }
  if(port_has_texture_request.empty()){
    return unsigned(-1);
  }
  if(port_has_texture_request.size() == 1){
    return port_has_texture_request[0];
  }
  //start from last processed port
  unsigned next_port = (m_last_processed_port_texture+1) % m_config->dice_cgra_core_num_ld_ports;
  while(true){
    if(std::find(port_has_texture_request.begin(), port_has_texture_request.end(), next_port) != port_has_texture_request.end()){
      return next_port;
    }
    next_port = (next_port+1) % m_config->dice_cgra_core_num_ld_ports;
    if(next_port == m_last_processed_port_texture){
      return unsigned(-1);
    }
  }
}

unsigned dice_mem_request_queue::get_next_process_port_memory()
{
  //check which port has global/local memory request
  std::vector<unsigned> port_has_memory_request;
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if(!m_ld_req_queue[i].empty()){
      mem_access_t access = m_ld_req_queue[i].front();
      if ((access.get_space() != global_space) && (access.get_space() != local_space) && (access.get_space() != param_space_local)){
        continue;
      } else {
        port_has_memory_request.push_back(i);
      }
    }
  }
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_st_ports; i++){
    if(!m_st_req_queue[i].empty()){
      mem_access_t access = m_st_req_queue[i].front();
      if ((access.get_space() != global_space) && (access.get_space() != local_space) && (access.get_space() != param_space_local)){
        continue;
      } else {
        port_has_memory_request.push_back(i+m_config->dice_cgra_core_num_ld_ports);
      }
    }
  }
  if(port_has_memory_request.empty()){
    return unsigned(-1);
  }

  //only one port has memory request
  if(port_has_memory_request.size() == 1){
    //check if the port is still waiting for coalescing
    m_last_processed_port_memory = port_has_memory_request[0];
    return port_has_memory_request[0];
  }
  //start from last processed port
  unsigned next_port = (m_last_processed_port_memory+1) % (m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports);
  while(true){
    if(std::find(port_has_memory_request.begin(), port_has_memory_request.end(), next_port) != port_has_memory_request.end()){
      m_last_processed_port_memory = next_port;
      return next_port;
    }
    next_port = (next_port+1) % (m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports);
    if(next_port == m_last_processed_port_memory){
      return unsigned(-1);
    }
  }
}

void dice_mem_request_queue::update_coaleasing_counter(){
  //update coaleasing_counter based on the first request in each queue
  //if the first request space is global/local, then increase the counter, otherwise reset the counter
  //if the counter is larger than the interval, then back to zero
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if(!m_ld_req_queue_pre_coalesce[i].empty()){
      mem_access_t access = m_ld_req_queue_pre_coalesce[i].front();
      if ((access.get_space() == global_space) || (access.get_space() == local_space) || (access.get_space() == param_space_local)){
        if(m_ld_coalescing_counter[i] < temporal_coalescing_interval){
          m_ld_coalescing_counter[i]++;
        }
      } else {
        m_ld_coalescing_counter[i] = 0;
      }
    }
  }
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_st_ports; i++){
    if(!m_st_req_queue_pre_coalesce[i].empty()){
      mem_access_t access = m_st_req_queue_pre_coalesce[i].front();
      if ((access.get_space() == global_space) || (access.get_space() == local_space) || (access.get_space() == param_space_local)){
        if(m_st_coalescing_counter[i] < temporal_coalescing_interval){
          m_st_coalescing_counter[i]++;
        }
      } else {
        m_st_coalescing_counter[i] = 0;
      }
    }
  }
}

void dice_mem_request_queue::coalesce_cycle(){
  //check if the coalescing interval is done
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if((m_ld_coalescing_counter[i] >= temporal_coalescing_interval) || (m_ld_req_queue_pre_coalesce[i].size() >= temporal_coalescing_max_cmd)){
      do_ld_coalescing(i);
      m_ld_coalescing_counter[i] = 0;
    }
  }
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_st_ports; i++){
    if((m_st_coalescing_counter[i] >= temporal_coalescing_interval) || (m_st_req_queue_pre_coalesce[i].size() >= temporal_coalescing_max_cmd)){
      do_st_coalescing(i);
      m_st_coalescing_counter[i] = 0;
    }
  }
}


void dice_mem_request_queue::do_ld_coalescing(unsigned port){
  //iterate through the first coalescing interval elements in the queue
  assert(m_ld_req_queue_pre_coalesce[port].size() > 0);
  unsigned coalescing_counter = 0;
  bool sector_segment_size = false; 
  if (m_config->gpgpu_coalesce_arch >= 20 &&
    m_config->gpgpu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    sector_segment_size = false;
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }
  unsigned segment_size = sector_segment_size? 32: 128;
  std::map<new_addr_type, dice_transaction_info> rd_transactions;  // each block addr maps to a list of transactions
  cgra_block_state_t* cgra_block = m_ld_req_queue_pre_coalesce[port].front().get_cgra_block_state();
  //step 1, gather all transactions
  while((!m_ld_req_queue_pre_coalesce[port].empty()) && (coalescing_counter < temporal_coalescing_max_cmd) && (cgra_block == m_ld_req_queue_pre_coalesce[port].front().get_cgra_block_state())){
    mem_access_t access = m_ld_req_queue_pre_coalesce[port].front();
    pop_ld_request_pre_coalesce(port);
    if ((access.get_space() == global_space) || (access.get_space() == local_space) || (access.get_space() == param_space_local)){
      //coalescing
      unsigned addr = access.get_addr();
      if(g_debug_execution==3 &m_ldst_unit->get_cgra_core_id() == m_ldst_unit->get_cgra_core()->get_dice_trace_sampling_core()){
        printf("DICE Sim uArch: [LD_COALESCING]: Cycle %d, Port %d, cmd addr = %p.\n",m_ldst_unit->get_cgra_core()->get_gpu()->gpu_sim_cycle +  m_ldst_unit->get_cgra_core()->get_gpu()->gpu_tot_sim_cycle , port, addr);
        fflush(stdout);
      }
      unsigned size = access.get_size();
      mem_access_type access_type = access.get_type();
      memory_space_t space = access.get_space();
      std::set<unsigned> tids = access.get_tids();
      unsigned block_address = line_size_based_tag_func_cgra(addr, segment_size);
      dice_transaction_info &info = rd_transactions[block_address];
      unsigned chunk = (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte
                                          // chunk does this thread access?
      info.chunks.set(chunk);
      info.access_type = access_type;
      info.space = space;
      for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it){
        unsigned tid = *it;
        info.active_threads.insert(tid);
      }
      //mem_access_sector_mask_t sector_mask;
      //sector_mask.set(chunk);
      unsigned idx = (addr & 127);
      //std::bitset<128> byte_mask;
      //active_mask_t active_mask = *m_block_active_mask;
      for (unsigned i = 0; i < size; i++){
        if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) {
          //byte_mask.set(idx + i);
          info.bytes.set(idx + i);
        }
      }
      //check port index, if ld, find ld_dest reg index in metadata, if store, just circularly use 4-7
      //use iterator to find the ld_dest_reg
      std::set<unsigned> ld_dest_regs = access.get_ldst_regs();
      info.ld_dest_regs = ld_dest_regs;
      info.port_idx.insert(port);
      
      //check if second access needed
      if (block_address != line_size_based_tag_func_cgra(addr + size - 1, segment_size)) {
        addr = addr + size - 1;
        unsigned block_address = line_size_based_tag_func_cgra(addr, segment_size);
        dice_transaction_info &info =rd_transactions[block_address];
        unsigned chunk = (addr & 127) / 32;
        info.chunks.set(chunk);
        for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it){
          unsigned tid = *it;
          info.active_threads.insert(tid);
        }
        info.access_type = access_type;
        info.space = space;
        //sector_mask.set(chunk);
        unsigned idx = (addr & 127);
        for (unsigned i = 0; i < size; i++){
          if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) {
            //byte_mask.set(idx + i);
            info.bytes.set(idx + i);
          }
        }
        info.ld_dest_regs = ld_dest_regs;
        info.port_idx.insert(port);
      }
      coalescing_counter++;
    } else {
      break;
    }
  }
  if(g_debug_execution==3 &m_ldst_unit->get_cgra_core_id() == m_ldst_unit->get_cgra_core()->get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: [LD_COALESCING]: Cycle %d, Port %d, Coalesced %d cmd into %d cmd.\n",m_ldst_unit->get_cgra_core()->get_gpu()->gpu_sim_cycle +  m_ldst_unit->get_cgra_core()->get_gpu()->gpu_tot_sim_cycle , port, coalescing_counter, rd_transactions.size());
    fflush(stdout);
  }
  //step 2, reduce size of each transaction
  for (std::map<new_addr_type, dice_transaction_info>::iterator t = rd_transactions.begin(); t != rd_transactions.end(); t++) {
    new_addr_type addr = t->first;
    const dice_transaction_info &info = t->second;
    memory_coalescing_arch_reduce(false, info, addr, segment_size, port, cgra_block);
  }
}

void dice_mem_request_queue::do_st_coalescing(unsigned port){
  //iterate through the first coalescing interval elements in the queue
  if(g_debug_execution==3 &m_ldst_unit->get_cgra_core_id() == m_ldst_unit->get_cgra_core()->get_dice_trace_sampling_core()){
    printf("DICE Sim uArch: [ST_COALESCING]: Cycle %d, Port %d\n",m_ldst_unit->get_cgra_core()->get_gpu()->gpu_sim_cycle + m_ldst_unit->get_cgra_core()->get_gpu()->gpu_tot_sim_cycle , port);
    fflush(stdout);
  }
  assert(m_st_req_queue_pre_coalesce[port].size() > 0);
  unsigned coalescing_counter = 0;
  bool sector_segment_size = false;
  if (m_config->gpgpu_coalesce_arch >= 20 &&
    m_config->gpgpu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    sector_segment_size = false;
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }
  unsigned segment_size = sector_segment_size? 32: 128;
  std::map<new_addr_type, dice_transaction_info> st_transactions;  // each block addr maps to a list of transactions
  cgra_block_state_t* cgra_block = m_st_req_queue_pre_coalesce[port].front().get_cgra_block_state();
  //step 1, gather all transactions
  while(!m_st_req_queue_pre_coalesce[port].empty() && coalescing_counter < temporal_coalescing_interval && cgra_block == m_st_req_queue_pre_coalesce[port].front().get_cgra_block_state()){
    mem_access_t access = m_st_req_queue_pre_coalesce[port].front();
    pop_st_request_pre_coalesce(port);
    if ((access.get_space() == global_space) || (access.get_space() == local_space) || (access.get_space() == param_space_local)){
      //coalescing
      unsigned addr = access.get_addr();
      unsigned size = access.get_size();
      mem_access_type access_type = access.get_type();
      memory_space_t space = access.get_space();
      std::set<unsigned> tids = access.get_tids();
      unsigned block_address = line_size_based_tag_func_cgra(addr, segment_size);
      dice_transaction_info &info = st_transactions[block_address];
      unsigned chunk = (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte
                                          // chunk does this thread access?
      info.chunks.set(chunk);
      info.access_type = access_type;
      info.space = space;
      for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it){
        unsigned tid = *it;
        info.active_threads.insert(tid);
      }
      //mem_access_sector_mask_t sector_mask;
      //sector_mask.set(chunk);
      unsigned idx = (addr & 127);
      //std::bitset<128> byte_mask;
      //active_mask_t active_mask = *m_block_active_mask;
      for (unsigned i = 0; i < size; i++){
        if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) {
          //byte_mask.set(idx + i);
          info.bytes.set(idx + i);
        }
      }
      //check port index, if ld, find ld_dest reg index in metadata, if store, just circularly use 4-7
      //use iterator to find the ld_dest_reg
      std::set<unsigned> ld_dest_regs = access.get_ldst_regs();
      info.ld_dest_regs = ld_dest_regs;
      info.port_idx.insert(port+m_config->dice_cgra_core_num_ld_ports);
      
      //check if second access needed
      if (block_address != line_size_based_tag_func_cgra(addr + size - 1, segment_size)) {
        addr = addr + size - 1;
        unsigned block_address = line_size_based_tag_func_cgra(addr, segment_size);
        dice_transaction_info &info =st_transactions[block_address];
        unsigned chunk = (addr & 127) / 32;
        info.chunks.set(chunk);
        for(std::set<unsigned>::iterator it = tids.begin(); it != tids.end(); ++it){
          unsigned tid = *it;
          info.active_threads.insert(tid);
        }
        info.access_type = access_type;
        info.space = space;
        //sector_mask.set(chunk);
        unsigned idx = (addr & 127);
        for (unsigned i = 0; i < size; i++){
          if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) {
            //byte_mask.set(idx + i);
            info.bytes.set(idx + i);
          }
        }
        info.ld_dest_regs = ld_dest_regs;
        info.port_idx.insert(port+m_config->dice_cgra_core_num_ld_ports);
      }
      coalescing_counter++;
    } else {
      break;
    }
  }
  //step 2, reduce size of each transaction
  for (std::map<new_addr_type, dice_transaction_info>::iterator t = st_transactions.begin(); t != st_transactions.end(); t++) {
    new_addr_type addr = t->first;
    const dice_transaction_info &info = t->second;
    memory_coalescing_arch_reduce(true, info, addr, segment_size, port+m_config->dice_cgra_core_num_ld_ports, cgra_block);
  }
}

void dice_mem_request_queue::memory_coalescing_arch_reduce(bool is_write, const dice_transaction_info &info,new_addr_type addr, unsigned segment_size, unsigned port, cgra_block_state_t* cgra_block) {
  mem_access_type access_type = info.access_type;
  assert((addr & (segment_size - 1)) == 0);

  const std::bitset<4> &q = info.chunks;
  assert(q.count() >= 1);
  std::bitset<2> h;  // halves (used to check if 64 byte segment can be
                     // compressed into a single 32 byte segment)

  unsigned size = segment_size;
  if (segment_size == 128) {
    bool lower_half_used = q[0] || q[1];
    bool upper_half_used = q[2] || q[3];
    if (lower_half_used && !upper_half_used) {
      // only lower 64 bytes used
      size = 64;
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else if ((!lower_half_used) && upper_half_used) {
      // only upper 64 bytes used
      addr = addr + 64;
      size = 64;
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    } else {
      assert(lower_half_used && upper_half_used);
    }
  } else if (segment_size == 64) {
    // need to set halves
    if ((addr % 128) == 0) {
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else {
      assert((addr % 128) == 64);
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    }
  }
  if (size == 64) {
    bool lower_half_used = h[0];
    bool upper_half_used = h[1];
    if (lower_half_used && !upper_half_used) {
      size = 32;
    } else if ((!lower_half_used) && upper_half_used) {
      addr = addr + 32;
      size = 32;
    } else {
      assert(lower_half_used && upper_half_used);
    }
  }
  if(is_write){
    mem_access_t access(access_type, addr, info.space, size, is_write, info.active_threads, info.ld_dest_regs, port ,info.active, info.bytes, info.chunks,m_config->gpgpu_ctx);
    access.assign_cgra_block_state(cgra_block);
    m_st_req_queue[port-m_config->dice_cgra_core_num_ld_ports].push_back(access);
  } else {
    mem_access_t access(access_type, addr, info.space, size, is_write, info.active_threads, info.ld_dest_regs, port ,info.active, info.bytes, info.chunks,m_config->gpgpu_ctx);
    access.assign_cgra_block_state(cgra_block);
    m_ld_req_queue[port].push_back(access);
  }
}

unsigned dice_mem_request_queue::get_next_process_port_shared(){
  //check which port has shared memory request
  std::vector<unsigned> port_has_shared_request;
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_ld_ports; i++){
    if(!m_ld_req_queue[i].empty()){
      mem_access_t access = m_ld_req_queue[i].front();
      if(access.get_space() == shared_space){
        port_has_shared_request.push_back(i);
      }
    }
  }
  for(unsigned i = 0; i < m_config->dice_cgra_core_num_st_ports; i++){
    if(!m_st_req_queue[i].empty()){
      mem_access_t access = m_st_req_queue[i].front();
      if(access.get_space() == shared_space){
        port_has_shared_request.push_back(i+m_config->dice_cgra_core_num_ld_ports);
      }
    }
  }
  if(port_has_shared_request.empty()){
    return unsigned(-1);
  }
  if(port_has_shared_request.size() == 1){
    return port_has_shared_request[0];
  }
  //start from last processed port
  unsigned next_port = (m_last_processed_port_shared+1) % (m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports);
  while(true){
    if(std::find(port_has_shared_request.begin(), port_has_shared_request.end(), next_port) != port_has_shared_request.end()){
      return next_port;
    }
    next_port = (next_port+1) % (m_config->dice_cgra_core_num_ld_ports+m_config->dice_cgra_core_num_st_ports);
    if(next_port == m_last_processed_port_shared){
      return unsigned(-1);
    }
  }
}


block_commit_table::block_commit_table(class gpgpu_sim* gpu, class cgra_core_ctx* cgra_core) {
  m_cgra_core = cgra_core;
  m_gpu = gpu;
  m_max_block_size = m_gpu->max_cta_per_core();
  m_commit_table.resize(m_max_block_size);
  m_ret.resize(m_max_block_size);
  for (unsigned i = 0; i < m_max_block_size; i++) {
    m_commit_table[i] = NULL;
    m_ret[i]=false;
  }
}

void block_commit_table::check_and_release() {
  for (unsigned i = 0; i < m_max_block_size; i++) {
    if (m_commit_table[i] != NULL) {
      if (m_commit_table[i]->block_done()) {
        m_gpu->gpu_sim_block += m_commit_table[i]->active_count();
        unsigned cta_id = m_commit_table[i]->get_cta_id();
        //TODO, can prefetch next block here or better SIMT stack operation
        if (m_cgra_core->fetch_stalled_by_simt_stack(cta_id)){
          if(m_commit_table[i]->get_current_metadata()->meta_id == m_cgra_core->get_fetch_waiting_block_id(cta_id)){
            assert(m_commit_table[i]->get_current_metadata()->branch);
            dice_cfg_block_t *cfg_block = m_commit_table[i]->get_current_cfg_block();
            assert(cfg_block != NULL);
            //check if predicate registers are all written back
            m_cgra_core->updateSIMTStack(m_commit_table[i]->get_cta_id(),cfg_block);
            m_cgra_core->clear_fetch_stalled_by_simt_stack(m_commit_table[i]->get_cta_id(),m_commit_table[i]->get_current_metadata()->meta_id);
            //m_cgra_core->clear_stalled_by_simt_stack();
          }
        }

        if(g_debug_execution==3 &m_cgra_core->get_id()== m_cgra_core->get_dice_trace_sampling_core()){
          printf("DICE Sim uArch [WRITEBACK_END]: Cycle %d, hw_cta=%d, Block=%d, table_index=%d\n",m_gpu->gpu_sim_cycle+m_gpu->gpu_tot_sim_cycle, cta_id, m_commit_table[i]->get_current_metadata()->meta_id,i);
          fflush(stdout);
        }

        delete m_commit_table[i];
        m_commit_table[i] = NULL;

        //return 
        if(m_ret[i]){
          //need to check if same cta still exists 
          if(!check_block_exist(cta_id)) m_cgra_core->complete_cta(cta_id);
        }

        m_ret[i] = false;
      }
    }
  }
}

unsigned fetch_scheduler::next_fetch_block(){
  //find all ctas that is not blocked by simt stack
  std::vector<unsigned> ready_cta_index;
  std::vector<address_type> ready_cta_pc;
  std::vector<unsigned> valid_cta_index;
  std::vector<address_type> valid_cta_pc;
  for(unsigned i = 0; i < cta_status_table_size; i++){
    unsigned start_index = (i+m_last_fetch_cta_id+1)%cta_status_table_size;
    if(!m_cta_status_table->is_free(start_index) && !m_cta_status_table->is_ret(start_index)){//has cta and not get ret block
      if(!m_cta_status_table->fetch_stalled_by_simt_stack(start_index)){
        //get stack top info
        //among ready ctas, find if any ctas next metadata pc is the same as current metadata pc
        address_type next_pc, rpc;
        m_cgra_core->get_simt_stack(start_index)->get_pdom_stack_top_info(&next_pc, &rpc);
        if(next_pc==m_previous_fetch_pc) {
          //printf("DICE Sim uArch [FETCH_SCHEDULER_META_MATCH]: Cycle %d, Block %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , start_index);
          m_last_fetch_cta_id = start_index;
          m_previous_fetch_pc = next_pc;
          return start_index;
        } else {
          ready_cta_pc.push_back(next_pc);
          ready_cta_index.push_back(start_index);
        }
      } else {
        //address_type next_pc, rpc;
        //m_cgra_core->get_simt_stack(start_index)->get_pdom_stack_top_info(&next_pc, &rpc);
        address_type next_pc = m_cta_status_table->get_prefetch_pc(start_index);
        if(next_pc==m_previous_fetch_pc) {
          //printf("DICE Sim uArch [FETCH_SCHEDULER_META_MATCH_PREFETCH]: Cycle %d, cta %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , start_index);
          m_last_fetch_cta_id = start_index;
          m_previous_fetch_pc = next_pc;
          return start_index;
        } else {
          valid_cta_pc.push_back(next_pc);
          valid_cta_index.push_back(start_index);
        }
      }
    }
  }
  //find one in ready state
  if(ready_cta_index.size() > 0){
    //printf("DICE Sim uArch [FETCH_SCHEDULER_READY]: Cycle %d, cta %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , ready_cta_index[0]);
    m_last_fetch_cta_id = ready_cta_index[0];
    m_previous_fetch_pc = ready_cta_pc[0];
    return ready_cta_index[0];
  } 

  //find one in valid state
  if(valid_cta_index.size() > 0){
    //printf("DICE Sim uArch [FETCH_SCHEDULER_VALID]: Cycle %d, cta %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle +  m_cgra_core->get_gpu()->gpu_tot_sim_cycle , valid_cta_index[0]);
    m_last_fetch_cta_id = valid_cta_index[0];
    m_previous_fetch_pc = valid_cta_pc[0];
    return valid_cta_index[0];
  }
  
  return unsigned(-1);
}

//unsigned reg_number_to_bank_mapping(unsigned reg_number, unsigned tid){
//  bool gpr;
//  if(reg_number<=32) { //constant registers
//    gpr = false;
//  }
//  else if (reg_number>64) { //predicate registers
//    gpr = false;
//  }
//  else { //general purpose registers
//    gpr = true;
//  }
//  if(gpr) return ((reg_number-1)+tid)%32; //swizzling arrangement
//  else return 32;
//}

unsigned reg_number_to_bank_mapping(unsigned reg_number, unsigned tid, unsigned max_coalesce){
  bool gpr;
  if(reg_number<=32) { //constant registers
    gpr = false;
  }
  else if (reg_number>64) { //predicate registers
    gpr = false;
  }
  else { //general purpose registers
    gpr = true;
  }
  if(gpr) {
    if(tid%32<8){
      return (reg_number-1+tid)%32;
    } else if (tid%32<16){
      return (reg_number-1+16+tid)%32;
    } else if (tid%128<24){
      return (reg_number-1+8+tid)%32;
    } else {
      return (reg_number-1+24+tid)%32;
    }
  }
  else return 32;
}


cgra_block_state_t::~cgra_block_state_t() {
  if (m_metadata_buffer.m_cfg_block) {
      dice_cfg_block_t *cfg_block = m_metadata_buffer.m_cfg_block;
      //printf("CGRA core %d: Deleting metadata buffer cfg_block at %p\n", m_cgra_core->get_id(), cfg_block);
      delete cfg_block;
  }
}