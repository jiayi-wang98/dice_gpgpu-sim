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

cgra_core_ctx::cgra_core_ctx(gpgpu_sim *gpu,simt_core_cluster *cluster,
  unsigned cgra_core_id, unsigned tpc_id,const shader_core_config *config,
  const memory_config *mem_config,shader_core_stats *stats){
  m_cluster = cluster;
  m_config = config;
  m_memory_config = mem_config;
  m_stats = stats;
  m_block_size = config->dice_cgra_core_max_threads;

  m_cgra_core_id = cgra_core_id;
  m_tpc = tpc_id;

  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
  m_thread = (ptx_thread_info **)calloc(m_block_size,sizeof(ptx_thread_info *));
  initializeSIMTStack(m_block_size);
}


void cgra_core_ctx::initializeSIMTStack(unsigned num_threads){
  m_simt_stack = new simt_stack(0, m_block_size, m_gpu);
  m_block_size=num_threads;
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
  simt_mask_t thread_done(m_block_size);
  addr_vector_t next_cfg_pc;
  for (unsigned i = 0; i < m_block_size; i++) {
    if (ptx_thread_done(i)) {
      thread_done.set(i);
      next_cfg_pc.push_back((address_type)-1);
    } else {
      if (cfg_block->reconvergence_pc == RECONVERGE_RETURN_PC)
      cfg_block->reconvergence_pc = get_return_pc(m_thread[i]);
      next_cfg_pc.push_back(m_thread[i]->get_pc());
    }
  }
  m_simt_stack->update(thread_done, next_cfg_pc, cfg_block->reconvergence_pc,
    cfg_block->op, cfg_block->metadata_size, cfg_block->metadata_pc);
}

bool cgra_core_ctx::ptx_thread_done(unsigned hw_thread_id) const {
  return ((m_thread[hw_thread_id] == NULL) ||
          m_thread[hw_thread_id]->is_done());
}


void cgra_core_ctx::execute_CFGBlock(dice_cfg_block_t* cfg_block) {
  for (unsigned t = 0; t < m_block_size; t++) {
    if (cfg_block->active(t)) {
      m_thread[t]->dice_exec_block(cfg_block,t);
      checkExecutionStatusAndUpdate(t);
    }
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

void cgra_core_ctx::issue_block2core(kernel_info_t &kernel){
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
  m_n_active_cta++;
  printf("DICE Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u,initialized @(%lld,%lld)\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle);
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

void cgra_core_ctx::init_CTA(unsigned cta_id, unsigned start_thread,
  unsigned end_thread, unsigned ctaid,int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_meta_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned n_active = 0;
    simt_mask_t active_threads(m_config->dice_cgra_core_max_threads);
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
    //DICE-TODO
    //m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id);
    m_not_completed += n_active;
    ++m_active_blocks;
  }
}

void exec_simt_core_cluster::create_cgra_core_ctx() {
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
  m_simt_stack->reset();
}


void cgra_core_ctx::create_front_pipeline(){
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),m_config->dice_cgra_core_max_threads);
  m_not_completed = 0;
  m_active_threads.reset();
  m_n_active_cta = 0;
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
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
  
}
//hardware simulation
void cgra_core_ctx::cycle(){
  //TODO
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
      m_cgra_block_state->clear_imiss_pending(); //clear the metadata miss flag
      m_metadata_fetch_buffer = ifetch_buffer_t(m_cgra_block_state->get_metadata_pc(), mf->get_access_size(), mf->get_wid()); //set the metadata fetch buffer
      assert(m_cgra_block_state->get_metadata_pc() ==(mf->get_addr()-PROGRAM_MEM_START));  // Verify that we got the instruction we were expecting.
      m_metadata_fetch_buffer.m_valid = true; //set valid in the fetch buffer
      m_cgra_block_state->set_last_fetch(m_gpu->gpu_sim_cycle); //set status
      delete mf;
    } else {
      // check if it's waiting on cache miss or can fetch new instruction
      //TODO, add pending writes clear check
      if (m_cgra_block_state->hardware_done() && 
        m_cgra_block_state->done_exit()) {
        bool did_exit = false;
        for (unsigned tid = 0; tid < m_config->dice_cgra_core_max_threads; tid++) {
          if (m_threadState[tid].m_active == true) {
            m_threadState[tid].m_active = false;
            unsigned cta_id = m_cgra_block_state->get_cta_id();
            if (m_thread[tid] == NULL) {
              register_cta_thread_exit(cta_id, m_kernel);
            } else {
              register_cta_thread_exit(cta_id,
                                       &(m_thread[tid]->get_kernel()));
            }
            m_not_completed -= 1;
            m_active_threads.reset(tid);
            did_exit = true;
          }
        }
        if (did_exit) m_cgra_block_state->set_done_exit();
        --m_active_blocks;
        assert(m_active_blocks >= 0);
      }
      // this code fetches metadata from the i-cache or generates memory
      if (!m_cgra_block_state->functional_done() &&
          !m_cgra_block_state->imiss_pending() &&
          m_cgra_block_state->metadata_buffer_empty()) {
        address_type pc;
        pc = m_cgra_block_state->get_metadata_pc();
        address_type ppc = pc + PROGRAM_MEM_START;
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
          m_cgra_block_state->set_imiss_pending();
          m_cgra_block_state->set_last_fetch(m_gpu->gpu_sim_cycle);
        } else if (status == HIT) {
          m_metadata_fetch_buffer = ifetch_buffer_t(pc, nbytes, 0);
          m_cgra_block_state->set_last_fetch(m_gpu->gpu_sim_cycle);
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
    m_n_active_cta--;
    shader_CTA_count_unlog(m_cgra_core_id, 1);

    printf("DICE-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
      cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
      m_n_active_cta);

    if (m_n_active_cta == 0) {
      printf("DICE-Sim uArch: Empty (last released kernel %u \'%s\').\n",kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
    m_kernel = NULL;
    }
  } 
}

dice_cfg_block_t* cgra_block_state_t::get_current_cfg_block() {
  assert(m_metadata_buffer.m_cfg_block!=NULL); 
  return m_metadata_buffer.m_cfg_block; 
}

dice_metadata* cgra_block_state_t::get_current_metadata(){
  return get_current_cfg_block()->get_metadata(); 
}

dice_block_t *cgra_block_state_t::get_dice_block(){ 
  return get_current_metadata()->dice_block; 
}

void cgra_core_ctx::fetch_bitstream(){
  //TODO
}
void cgra_core_ctx::execute(){
  //TODO
}
//inner pipeline in execute();
void cgra_core_ctx::read_operands(){
  //TODO
}
void cgra_core_ctx::cgra_execute_block(){
  //TODO
}
void cgra_core_ctx::writeback(){
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