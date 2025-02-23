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


void cgra_core_ctx::execute_cfg_block_t(dice_cfg_block_t &cfg_block) {
  //for (unsigned t = 0; t < m_block_size; t++) {
  //  if (cfg_block.active(t)) {
  //    m_thread[t]->dice_exec_block(cfg_block);
  //    // virtual function
  //    checkExecutionStatusAndUpdate(t);
  //  }
  //}
}

void cgra_core_ctx::checkExecutionStatusAndUpdate(unsigned tid) {
  if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
     m_liveThreadCount--;
   }
}