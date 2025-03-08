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

extern int g_debug_execution;

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
  m_fetch_stalled_by_simt_stack = false;
  m_active_blocks = 0;
  m_gpu = gpu;
  m_kernel = NULL;
  m_cgra_core_id = cgra_core_id;
  m_tpc = tpc_id;

  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
  m_thread = (ptx_thread_info **)calloc(m_max_block_size,sizeof(ptx_thread_info *));
  initializeSIMTStack(m_max_block_size);
}


void cgra_core_ctx::initializeSIMTStack(unsigned num_threads){
  m_simt_stack = new simt_stack(m_cgra_core_id, m_max_block_size, m_gpu);
}

void cgra_core_ctx::resizeSIMTStack(unsigned num_threads){
  m_simt_stack->resize_warp(num_threads);
}

void cgra_core_ctx::get_pdom_stack_top_info(unsigned *pc,
  unsigned *rpc) const {
  m_simt_stack->get_pdom_stack_top_info(pc, rpc);
}

void cgra_core_ctx::deleteSIMTStack() {
  if (m_simt_stack) {
    delete m_simt_stack;
    m_simt_stack = NULL;
  }
}

void cgra_core_ctx::updateSIMTStack(dice_cfg_block_t *cfg_block) {
  simt_mask_t thread_done(m_kernel_block_size);
  addr_vector_t next_pc;
  for (unsigned i = 0; i < m_kernel_block_size; i++) {
    if (ptx_thread_done(i)) {
      thread_done.set(i);
      next_pc.push_back((address_type)-1);
    } else {
      if (cfg_block->reconvergence_pc == RECONVERGE_RETURN_PC) {
        cfg_block->reconvergence_pc = get_return_meta_pc(m_thread[i]);
      }
      next_pc.push_back(m_thread[i]->get_meta_pc());
    }
  }
  if(m_cgra_core_id == 0){
    m_simt_stack->update_sid(thread_done, next_pc, cfg_block->reconvergence_pc,
    cfg_block->op, cfg_block->get_metadata()->m_size, cfg_block->get_metadata()->m_PC);
  } else {
    m_simt_stack->update(thread_done, next_pc, cfg_block->reconvergence_pc,
    cfg_block->op, cfg_block->get_metadata()->m_size, cfg_block->get_metadata()->m_PC);
  }
}

bool cgra_core_ctx::ptx_thread_done(unsigned hw_thread_id) const {
  return ((m_thread[hw_thread_id] == NULL) ||
          m_thread[hw_thread_id]->is_done());
}


void cgra_core_ctx::execute_CFGBlock(dice_cfg_block_t* cfg_block) {
  for (unsigned t = 0; t < m_kernel_block_size; t++) {
    if (cfg_block->active(t)) {
      m_thread[t]->dice_exec_block(cfg_block,t);
      checkExecutionStatusAndUpdate(t);
    }
  }
}

bool cgra_core_ctx::ldst_unit_response_buffer_full() const{
  return m_ldst_unit->response_buffer_full();
}

void cgra_core_ctx::execute_1thread_CFGBlock(dice_cfg_block_t* cfg_block, unsigned tid) {
  if (cfg_block->active(tid)) {
    if(g_debug_execution >= 3 && m_cgra_core_id==0){
      printf("DICE-Sim Functional: cycle %d, cgra_core %u executed thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle, m_cgra_core_id, tid,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
    }
    m_thread[tid]->dice_exec_block(cfg_block,tid);
    checkExecutionStatusAndUpdate(tid);
  }
}

void cgra_core_ctx::checkExecutionStatusAndUpdate(unsigned tid) {
  if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
     m_liveThreadCount--;
   }
}

void cgra_core_ctx::set_max_cta(const kernel_info_t &kernel){
  // calculate the max cta count and cta size for local memory address mapping
  //DICE-TODO in the future. For now, we assume that each core can only run 1 CTA at a time.
  //In the next gen, a cgra_core can run multiple CTAs at a time and there's a similar "CTA scheduler" as warp scheduler to handle concurrent CTA execution.
  kernel_max_cta_per_shader = 1;
  kernel_padded_threads_per_cta = kernel.threads_per_cta();
}

void cgra_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
  long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_cgra_core_id];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_cgra_core_id];
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
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);
  // determine hardware threads that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  unsigned int start_thread, end_thread;
  start_thread = free_cta_hw_id * cta_size; //0 for now
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

  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }

  // initialize the SIMT stacks and fetch hardware
  init_CTA(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  //DICE-TODO
  //m_active_blocks++;
  printf("DICE Sim uArch: core id: %d, cta:%2u, start_tid:%4u, end_tid:%4u,initialized @(%lld,%lld), m_kernel_block_size=%d\n",
                m_cgra_core_id, ctaid, start_thread, end_thread, m_gpu->gpu_sim_cycle,
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
  //return (m_active_blocks < m_config->max_cta(kernel));
  return (m_active_blocks < 1);//TODO
}

void cgra_core_ctx::init_CTA(unsigned cta_id, unsigned start_thread,
  unsigned end_thread, unsigned ctaid,int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_meta_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned n_active = 0;
    simt_mask_t active_threads(end_thread - start_thread);
    for (unsigned tid = 0; tid < m_config->dice_cgra_core_max_threads; tid++) {
      if (tid < end_thread) {
        n_active++;
        assert(!m_active_threads.test(tid));
        m_active_threads.set(tid);
        active_threads.set(tid);
      }
    }
    //init simt stack
    m_simt_stack->launch(start_pc, active_threads);
    if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_0_%d_simt.txt",ctaid);
        unsigned pc, rpc;
        m_simt_stack->resume(fname);
        m_simt_stack->get_pdom_stack_top_info(&pc, &rpc);
        for (unsigned t = 0; t < m_config->dice_cgra_core_max_threads; t++) {
          if (m_thread != NULL) {
            m_thread[t]->set_next_meta_pc(pc);
            m_thread[t]->update_metadata_pc();
          }
        }
      start_pc = pc;
    }
    m_cgra_block_state[MF_DE]->init(start_pc, cta_id, active_threads);
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
  //set actual kernel block size
  m_kernel_block_size = end_thread - start_thread;
  resizeSIMTStack(m_kernel_block_size);
  m_simt_stack->reset();

  m_scoreboard->clear();
  m_scoreboard->resize(m_kernel_block_size);
  
  //reset block state block size
  //delete existing block state
  for (unsigned i = 0; i < NUM_DICE_STAGE; i++) {
    delete m_cgra_block_state[i];
  }
  m_cgra_block_state.clear();
  //create new block state
  unsigned total_pipeline_stages = NUM_DICE_STAGE;
  m_cgra_block_state.reserve(total_pipeline_stages);
  for (unsigned i = 0; i < total_pipeline_stages; i++) {
    m_cgra_block_state.push_back(new cgra_block_state_t(this, m_kernel_block_size));
  }

  //reset cgra_unit state
  m_cgra_unit->reinit();
  //reinit fetch buffer
  m_metadata_fetch_buffer = ifetch_buffer_t();
  m_bitstream_fetch_buffer = ifetch_buffer_t();
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
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
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
  m_ldst_unit = new ldst_unit(m_icnt, m_mem_fetch_allocator, this, m_dispatcher_rfu, m_scoreboard ,m_config, m_memory_config, m_stats, m_cgra_core_id, m_tpc);
}

//hardware simulation
void cgra_core_ctx::cycle(){
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_cgra_core_id]++;
  execute();
  decode();
  fetch_bitstream();
  fetch_metadata();
}

void cgra_core_ctx::accept_metadata_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if(g_debug_execution >= 3 && m_cgra_core_id==0){
    printf("DICE Sim uArch: accept_metadata_fetch_response() pc=0x%03x\n", mf->get_addr());
    fflush(stdout);
  }
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void cgra_core_ctx::accept_bitstream_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if(g_debug_execution >= 3 && m_cgra_core_id==0){
     printf("DICE Sim uArch: accept_bitstream_fetch_response() addr=0x%03x\n", mf->get_addr());
     fflush(stdout);
  }
  m_L1B->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void cgra_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  if(g_debug_execution >= 3 && m_cgra_core_id==0){
    printf("DICE Sim uArch: accept_ladst_unit_response() addr=0x%03x\n", mf->get_addr());
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
      m_cgra_block_state[MF_DE]->clear_imiss_pending(); //clear the metadata miss flag
      m_metadata_fetch_buffer = ifetch_buffer_t(m_cgra_block_state[MF_DE]->get_metadata_pc(), mf->get_access_size(), mf->get_wid()); //set the metadata fetch buffer
      if(g_debug_execution >= 3 && m_cgra_core_id==0){
        printf("DICE Sim uArch [FETCH_META]: cycle %d, fetched metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle,mf->get_addr());
        fflush(stdout);
      }
      assert(m_cgra_block_state[MF_DE]->get_metadata_pc() ==(mf->get_addr()-PROGRAM_MEM_START));  // Verify that we got the instruction we were expecting.
      //m_metadata_fetch_buffer.m_valid = true; //set valid in the fetch buffer
      m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle); //set status;
      delete mf;
    } else {
      // check if it's waiting on cache miss or can fetch new instruction
      //TODO, add pending writes clear check
      if (m_cgra_block_state[MF_DE]->done_exit()){
        return;
      }
      // this code fetches metadata from the i-cache or generates memory
      if (!m_fetch_stalled_by_simt_stack && !m_cgra_block_state[MF_DE]->imiss_pending() &&
          m_cgra_block_state[MF_DE]->metadata_buffer_empty()) {
        address_type pc;
        //set next pc according to srack top
        address_type next_pc, rpc;
        m_simt_stack->get_pdom_stack_top_info(&next_pc, &rpc);
        m_cgra_block_state[MF_DE]->set_next_pc(next_pc);
        pc = m_cgra_block_state[MF_DE]->get_metadata_pc();
        address_type ppc = pc + PROGRAM_MEM_START;
        if(g_debug_execution >= 3 && m_cgra_core_id==0){
          printf("DICE Sim uArch [FETCH_META]: cycle %d, fetch_metadata() pc=0x%03x\n", m_gpu->gpu_sim_cycle, pc);
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
            m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
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
          m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle);
        } else if (status == HIT) {
          if(g_debug_execution >= 3 && m_cgra_core_id==0){
            printf("DICE Sim uArch [FETCH_META]: cycle %d, fetched metadata (cache-hit)\n",m_gpu->gpu_sim_cycle);
            fflush(stdout);
          }
          m_metadata_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
          m_cgra_block_state[MF_DE]->set_last_fetch(m_gpu->gpu_sim_cycle);
          delete mf;
        } else {
          assert(status == RESERVATION_FAIL);
          delete mf;
        }
      }
    }
  }
  m_L1I->cycle();
}

void cgra_core_ctx::register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  m_cta_status[cta_num]--;
  if (!m_cta_status[cta_num]) {
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
  m_metadata_buffer.m_cfg_block = cfg_block;
  m_active_threads = cfg_block->get_active_mask();
  m_metadata_buffer.m_valid = true;
 }

void cgra_core_ctx::fetch_bitstream(){
  if (!m_bitstream_fetch_buffer.m_valid) { //if there is no bitstream in the buffer
    if (m_L1B->access_ready()) { //if the bitstream cache is ready to be accessed (i.e. there are data responses)
      mem_fetch *mf = m_L1B->next_access(); //get response from the instruction cache
      m_cgra_block_state[MF_DE]->clear_bmiss_pending(); //clear the metadata miss flag
      m_bitstream_fetch_buffer = ifetch_buffer_t(m_cgra_block_state[MF_DE]->get_bitstream_pc(), mf->get_access_size(), mf->get_wid()); //set the metadata fetch buffer
      if(g_debug_execution >= 3 && m_cgra_core_id==0){
        printf("DICE Sim uArch [FETCH_BITS]: cycle %d, fetched bitstream pc=0x%04x\n",m_gpu->gpu_sim_cycle, mf->get_addr());
        fflush(stdout);
      }
      assert(m_cgra_block_state[MF_DE]->get_bitstream_pc() ==(mf->get_addr()-PROGRAM_MEM_START));  // Verify that we got the instruction we were expecting.
      //m_bitstream_fetch_buffer.m_valid = true; //set valid in the fetch buffer
      m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle); //set status
      m_cgra_block_state[MF_DE]->bitstream_buffer_fill(); //fill the bitstream buffer
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
        if(g_debug_execution >= 3 && m_cgra_core_id==0){
          printf("DICE Sim uArch [FETCH_BITS]: cycle %d, fetch_bitstream() pc=0x%03x\n", m_gpu->gpu_sim_cycle, pc);
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
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
        if (status == MISS) {
          m_cgra_block_state[MF_DE]->set_bmiss_pending();
          m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle);
        } else if (status == HIT) {
          if(g_debug_execution >= 3 && m_cgra_core_id==0){
            printf("DICE Sim uArch [FETCH_BITS]: cycle %d, fetched bitstream (cache-hit)\n",m_gpu->gpu_sim_cycle);
            fflush(stdout);
          }
          m_bitstream_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
          m_cgra_block_state[MF_DE]->set_last_bitstream_fetch(m_gpu->gpu_sim_cycle);
          m_cgra_block_state[MF_DE]->bitstream_buffer_fill(); //fill the bitstream buffer
          delete mf;
        } else {
          assert(status == RESERVATION_FAIL);
          delete mf;
        }
      } else {
        //if(m_gpu->gpu_sim_cycle>553 && m_gpu->gpu_sim_cycle<558){
        //  printf("DICE-Sim uArch: cycle %d, fetch_bitstream() stalled because of\n", m_gpu->gpu_sim_cycle);
        //  printf("function done = %d\n", m_cgra_block_state[MF_DE]->functional_done());
        //  printf("bmiss_pending = %d\n", m_cgra_block_state[MF_DE]->bmiss_pending());
        //  printf("bitstream_buffer_waiting = %d\n", m_cgra_block_state[MF_DE]->bitstream_buffer_waiting());
        //}
      }
    }
  }
  m_L1B->cycle();
}

dice_cfg_block_t* cgra_core_ctx::get_dice_cfg_block(address_type pc){
  // read the metadata from the functional model
  dice_metadata* metadata = m_gpu->gpgpu_ctx->get_meta_from_pc(pc);
  dice_cfg_block_t* cfg_block = new dice_cfg_block_t(m_cgra_core_id, m_kernel_block_size, metadata);
  cfg_block->set_active(m_simt_stack->get_active_mask());
  return cfg_block;
}

void cgra_core_ctx::cache_flush() { m_ldst_unit->flush(); }

void cgra_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }

void cgra_core_ctx::decode(){
  if (m_metadata_fetch_buffer.m_valid && m_cgra_block_state[MF_DE]->decode_done() == false) {
    // decode and place them into ibuffer
    address_type pc = m_metadata_fetch_buffer.m_pc;
    dice_cfg_block_t *cfg_block = get_dice_cfg_block(pc);
    m_cgra_block_state[MF_DE]->metadata_buffer_fill(cfg_block);
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
      if(!m_cgra_block_state[DP_CGRA]->dummy() && !m_cgra_block_state[DP_CGRA]->cgra_fabric_done()){
        //printf("DICE-Sim uArch: cycle %d, decode block is exit block but DP-CGRA block is not finished\n", m_gpu->gpu_sim_cycle);
        return;
      }
      if (!m_cgra_block_state[MEM_WRITEBACK]->dummy() && !m_cgra_block_state[MEM_WRITEBACK]->writeback_done()) {
        //printf("DICE-Sim uArch: cycle %d, decode block is exit block but MEM_WRITEBACK block is not finished\n", m_gpu->gpu_sim_cycle);
        return;
      }
      //direct exit here instead of run ret
      //iterate through all active threads
      //register exit
      if(g_debug_execution >= 3 && m_cgra_core_id==0){
        printf("DICE Sim uArch [DECODE-RETURN]: cycle %d, decode metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle, pc);
        fflush(stdout);
      }
      for (unsigned tid = 0; tid < m_config->dice_cgra_core_max_threads; tid++) {
        if (m_threadState[tid].m_active == true) {
          bool empty = m_thread[tid]->callstack_pop();
          if (empty) {
            m_thread[tid]->set_done();
            m_thread[tid]->registerExit();
          }
          m_threadState[tid].m_active = false;
          unsigned cta_id = m_cgra_block_state[MF_DE]->get_cta_id();
          if (m_thread[tid] == NULL) {
            register_cta_thread_exit(cta_id, m_kernel);
          } else {
            register_cta_thread_exit(cta_id,
                                     &(m_thread[tid]->get_kernel()));
          }
          m_not_completed -= 1;
          m_active_threads.reset(tid);
        }
      }
      return;
    }
    if(g_debug_execution >= 3 && m_cgra_core_id==0){
      printf("DICE Sim uArch [DECODE]: cycle %d, decode metadata pc=0x%04x\n",m_gpu->gpu_sim_cycle, pc);
      fflush(stdout);
    }
    if (metadata->branch) {
      if(metadata->uni_bra){
        //update simt stack top with branch target pc
        m_simt_stack->update_no_divergence(metadata->branch_target_meta_pc);
      }
      else {
        //predicate branch, wait for active mask to be done to update
        set_stalled_by_simt_stack(m_cgra_block_state[MF_DE]->get_current_metadata()->meta_id);
      }
    } else {
      //update simt stack top with next pc
      m_simt_stack->update_no_divergence(metadata->get_PC()+metadata->metadata_size());
    }
    //set state and fill the cfg block
    m_cgra_block_state[MF_DE]->set_decode_done();
    //set_can_fetch_metadata();
  }
}

void cgra_core_ctx::set_can_fetch_metadata() {m_metadata_fetch_buffer.m_valid = false;}

void cgra_core_ctx::execute(){
  writeback();
  cgra_execute_block();
  dispatch();
}
//inner pipeline in execute();
void cgra_core_ctx::dispatch(){
  m_dispatcher_rfu->cycle();
  //if(m_cgra_core_id==0 && m_gpu->gpu_sim_cycle > 1891 && m_gpu->gpu_sim_cycle < 1895){
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
    if(m_cgra_block_state[MF_DE]->ready_to_dispatch()){
      if(g_debug_execution >= 3 && m_cgra_core_id==0){
        printf("DICE Sim uArch [DISPATCHER]: cycle %d, dispatch metadata id=%d, pc=0x%04x\n",m_gpu->gpu_sim_cycle, m_cgra_block_state[MF_DE]->get_current_metadata()->meta_id,m_cgra_block_state[MF_DE]->get_current_metadata()->get_PC());
        fflush(stdout);
      }
      m_cgra_block_state[DP_CGRA] = m_cgra_block_state[MF_DE];
      m_cgra_unit->set_latency(m_cgra_block_state[DP_CGRA]->get_block_latency());
      m_cgra_block_state[MF_DE] = new cgra_block_state_t(this, m_kernel_block_size);
      m_cgra_block_state[MF_DE]->init(unsigned(-1), m_cgra_block_state[DP_CGRA]->get_cta_id(), m_cgra_block_state[DP_CGRA]->get_active_threads());
      m_metadata_fetch_buffer = ifetch_buffer_t();
      m_bitstream_fetch_buffer = ifetch_buffer_t();
    }
  }
}

void cgra_core_ctx::cgra_execute_block(){
  if(m_cgra_block_state[DP_CGRA]->dispatch_done() && m_cgra_unit->can_exec() && !m_cgra_block_state[DP_CGRA]->cgra_fabric_done()){
    if(g_debug_execution >= 3 && m_cgra_core_id==0){
      printf("DICE Sim uArch [CGRA_EXECU]: cycle %d, cgra_execute finished for dice block id=%d\n",m_gpu->gpu_sim_cycle, m_cgra_block_state[DP_CGRA]->get_current_metadata()->meta_id);
      fflush(stdout);
    }
    m_cgra_block_state[DP_CGRA]->set_cgra_fabric_done();
  }
  m_cgra_unit->cycle();
}

void cgra_core_ctx::writeback(){
  //register writeback
  if(m_cgra_unit->out_valid()){
    unsigned tid=m_cgra_unit->out_tid();
    execute_1thread_CFGBlock(m_cgra_block_state[DP_CGRA]->get_current_cfg_block(), tid);

    m_dispatcher_rfu->writeback(m_cgra_block_state[DP_CGRA],0,tid);//TODO
    // TODO: need to move to dispatcher to check if bankconflict
    m_scoreboard->releaseRegisters(m_cgra_block_state[DP_CGRA]->get_current_metadata(), tid);
    //TODO move to m_dispatcher_rfu->writeback_ldst
    m_scoreboard->releaseRegistersFromLoad(m_cgra_block_state[DP_CGRA]->get_current_metadata(), tid);
  }

  //check if cgra fabric is finished
  if(m_cgra_block_state[DP_CGRA]->cgra_fabric_done() && (m_cgra_block_state[MEM_WRITEBACK]->writeback_done() || m_cgra_block_state[MEM_WRITEBACK]->dummy())){
    //move to writeback stage
    if(g_debug_execution >= 3 && m_cgra_core_id==0){
      printf("DICE Sim uArch [WRITEBACK]: cycle %d, writeback on going for dice block id=%d\n",m_gpu->gpu_sim_cycle, m_cgra_block_state[DP_CGRA]->get_current_metadata()->meta_id);
      fflush(stdout);
    }
    m_cgra_block_state[MEM_WRITEBACK] = m_cgra_block_state[DP_CGRA];
    m_cgra_block_state[DP_CGRA] = new cgra_block_state_t(this, m_kernel_block_size);
    m_cgra_block_state[DP_CGRA]->init(unsigned(-1), m_cgra_block_state[MEM_WRITEBACK]->get_cta_id(), m_cgra_block_state[MEM_WRITEBACK]->get_active_threads());
    //FAKE writeback now
    //TODO
    m_cgra_block_state[MEM_WRITEBACK]->set_writeback_done();
    //if metadata fetch stalled by simt stack,update simt stack and clear it
    if (m_fetch_stalled_by_simt_stack && (!m_cgra_block_state[MEM_WRITEBACK]->dummy())){
      if(m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->meta_id == m_fetch_waiting_block_id) {
        assert(m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->branch);
        dice_cfg_block_t *cfg_block = m_cgra_block_state[MEM_WRITEBACK]->get_current_cfg_block();
        assert(cfg_block != NULL);
        updateSIMTStack(cfg_block);
        clear_stalled_by_simt_stack();
      }
    }
    if(g_debug_execution >= 3 && m_cgra_core_id==0){
      printf("DICE Sim uArch [WRITEBACK]: cycle %d, writeback finished for dice block id=%d\n",m_gpu->gpu_sim_cycle, m_cgra_block_state[MEM_WRITEBACK]->get_current_metadata()->meta_id);
      fflush(stdout);
    }
  }
  //ldst unit writeback
  //TODO
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
  printf("%s CGRA running = ", m_name.c_str());
  (*m_executing_block)->get_current_metadata()->dump();
}

void cgra_unit::cycle(){
  //shift the shift registers to the right
  for (unsigned i = m_latency; i > 0; i--) {
    shift_registers[i] = shift_registers[i - 1];
  }
  //shift_registers[0] = unsigned(-1); //will be modified later
  is_busy = false;
  for(unsigned i = 0; i < m_latency; i++){
    if(shift_registers[i] != unsigned(-1)){
      is_busy = true;
      break;
    }
  }
  //if((*m_executing_block)->ready_to_dispatch()){
  //  if((*m_executing_block)->dispatch_done() && is_busy==false){
  //    (*m_executing_block)->set_cgra_fabric_done();
  //  }
  //}
}

cgra_unit::cgra_unit(const shader_core_config *config, cgra_core_ctx *cgra_core, cgra_block_state_t **block){
  //reserve shift registers
  //m_max_cgra_latency = config->dice_cgra_core_max_latency;
  m_name = "CGRA_default";
  m_executing_block = block;
  m_cgra_core = cgra_core;
  for (unsigned i = 0; i < MAX_CGRA_FABRIC_LATENCY; i++) {
    shift_registers[i] = unsigned(-1); //unsigned(-1) means empty/invalid
  }
  m_latency = MAX_CGRA_FABRIC_LATENCY-1;
}

void cgra_core_ctx::exec(unsigned tid){
  m_cgra_unit->exec(tid);
  //run function simulation here
  //move to when cgra out valid.
  if(tid != unsigned(-1)){
    //execute_1thread_CFGBlock(m_cgra_block_state[DP_CGRA]->get_current_cfg_block(), tid);
    //if(g_debug_execution >= 3 && m_cgra_core_id==0){
    //  printf("DICE-Sim Functional: cycle %d, cgra_core %u executed thread %u run dice-block %d\n",m_gpu->gpu_sim_cycle, m_cgra_core_id, tid,m_cgra_block_state[DP_CGRA]->get_current_cfg_block()->get_metadata()->meta_id);
    //}
  }
}

void dispatcher_rfu_t::cycle(){
  //send to execute in ready buffer
  if(m_ready_threads.size() > 0){
    unsigned tid = m_ready_threads.front();
    //if(g_debug_execution >= 3 && m_cgra_core->get_id()==0){
    //  printf("DICE Sim uArch [DISPATCHER]: cycle %d, operands ready of thread %d for dice block id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle, tid ,(*m_dispatching_block)->get_current_metadata()->meta_id);
    //  fflush(stdout);
    //}
    m_scoreboard->reserveRegisters((*m_dispatching_block)->get_current_metadata(), tid);
    m_ready_threads.pop_front();
    m_cgra_core->exec(tid);//issue thread to cgra unit
    m_dispatched_thread++;
  } else {
    m_cgra_core->exec(unsigned(-1));//issue nop to cgra unit
  }
  //check if all thread in current block is dispatched. if not, keep dispatching
  if ((*m_dispatching_block)->ready_to_dispatch() && !(*m_dispatching_block)->dispatch_done()) {
    if((m_dispatched_thread+m_ready_threads.size()) < (*m_dispatching_block)->active_count()){//or use SIMT-stack active count m_cgra_core->m_simt_stack->get_active_mask()->count();
      //find next active thread id
      unsigned tid = next_active_thread();
      //DICE-TODO: simulate read operands for this tid
      //if(g_debug_execution >= 3 && m_cgra_core->get_id()==0){
      //  printf("DICE Sim uArch [DISPATCHER]: cycle %d, dispatch and get operands of thread %d for dice block id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle, tid ,(*m_dispatching_block)->get_current_metadata()->meta_id);
      //  fflush(stdout);
      //}
      if(!m_scoreboard->checkCollision(tid, (*m_dispatching_block)->get_current_metadata())){
        m_ready_threads.push_back(tid);
        m_last_dispatched_tid = tid;
      }
    }
    else{
      //all threads in the block are dispatched
      //DICE-TODO: simulate writeback
      if(g_debug_execution >= 3 && m_cgra_core->get_id()==0){
        printf("DICE Sim uArch [DISPATCHER]: cycle %d, all threads dispatched for dice block id = %d\n",m_cgra_core->get_gpu()->gpu_sim_cycle, (*m_dispatching_block)->get_current_metadata()->meta_id);
        fflush(stdout);
      }
      (*m_dispatching_block)->set_dispatch_done();
      m_dispatched_thread = 0;
      m_last_dispatched_tid = unsigned(-1);
    }
  } else {
    //if(m_cgra_core->get_id()==0 && m_cgra_core->get_gpu()->gpu_sim_cycle>553 && m_cgra_core->get_gpu()->gpu_sim_cycle<555){
    //  printf("DICE-Sim uArch: cycle %d, dispatch() block id = %d, stalled because of\n", (*m_dispatching_block)->get_current_metadata()->meta_id ,m_cgra_core->get_gpu()->gpu_sim_cycle);
    //  printf("ready_to_dispatch = %d\n", (*m_dispatching_block)->ready_to_dispatch());
    //  printf("dispatch_done = %d\n", (*m_dispatching_block)->dispatch_done());
    //  printf("decoded = %d\n", (*m_dispatching_block)->decode_done());
    //  printf("metadata_buffer_empty = %d\n", (*m_dispatching_block)->metadata_buffer_empty());
    //}
  }
}

void dispatcher_rfu_t::writeback(cgra_block_state_t* block, unsigned reg_num, unsigned tid){
  //writeback
  //TODO
}

unsigned dispatcher_rfu_t::next_active_thread(){
  unsigned next_tid=m_last_dispatched_tid+1;
  assert(next_tid < m_config->dice_cgra_core_max_threads);
  if(next_tid >= (*m_dispatching_block)->get_block_size()){
    printf("DICE-Sim uArch: [Error] cycle %d, core %d, no more active thread found in the block\n",m_cgra_core->get_gpu()->gpu_sim_cycle, m_cgra_core->get_id());
    printf("DICE-Sim uArch: current dispatched number = %d, need total = %d \n",m_dispatched_thread+m_ready_threads.size(), (*m_dispatching_block)->active_count());
    fflush(stdout);
    assert(next_tid < (*m_dispatching_block)->get_block_size());
  }
  while((*m_dispatching_block)->active(next_tid) == false){
    if(next_tid >= (*m_dispatching_block)->get_block_size()-1){
      printf("DICE-Sim uArch: [Error] cycle %d, core %d, no more active thread found in the block\n",m_cgra_core->get_gpu()->gpu_sim_cycle, m_cgra_core->get_id());
      printf("DICE-Sim uArch: current dispatched number = %d, need total = %d \n",m_dispatched_thread+m_ready_threads.size(), (*m_dispatching_block)->active_count());
      fflush(stdout);
    }
    next_tid++;
    assert(next_tid < (*m_dispatching_block)->get_block_size());
  }
  return next_tid;
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
    }
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
      //printf("DICE-Sim uArch: cycle %d, collision detected for thread %d, reg %d\n",m_gpu->gpu_sim_cycle, tid, *it2);
      return true;
    }
  }
  return false;
}


void ldst_unit::cycle_cgra(){
  writeback_cgra();
  //TODO
}

void ldst_unit::writeback_cgra(){
}