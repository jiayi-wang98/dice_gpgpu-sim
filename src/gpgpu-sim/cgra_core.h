#ifndef CGRA_CORE_H
#define CGRA_CORE_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx_sim.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "traffic_breakdown.h"
#include "shader.h"

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;
class dice_cfg_block_t;

class cgra_core_ctx {
  public:
    cgra_core_ctx(gpgpu_sim *gpu, kernel_info_t *kernel,unsigned threads_per_cgra)
    : m_gpu(gpu),
      m_kernel(kernel),
      m_simt_stack(NULL),
      m_thread(NULL),
      m_block_size(threads_per_cgra) {
      m_thread = (ptx_thread_info **)calloc(threads_per_cgra,sizeof(ptx_thread_info *));
      initializeSIMTStack(threads_per_cgra);
    }
    
    virtual ~cgra_core_ctx() { free(m_thread); }
    //get meta info
    class gpgpu_sim *get_gpu() {
      return m_gpu;
    }
    kernel_info_t *get_kernel_info() { return m_kernel; }
    class ptx_thread_info **get_thread_info() {
      return m_thread;
    }
    unsigned get_block_size() const { return m_block_size;}

    //stack operation
    void initializeSIMTStack(unsigned num_threads);
    void deleteSIMTStack();
    void get_pdom_stack_top_info(unsigned *pc,unsigned *rpc) const;
    void updateSIMTStack(dice_cfg_block_t *cfg_block);

    //thread operation
    bool ptx_thread_done(unsigned hw_thread_id) const;
    void execute_cfg_block_t(dice_cfg_block_t &cfg_block);
    void checkExecutionStatusAndUpdate(unsigned tid);
  protected:
    class gpgpu_sim *m_gpu;
    kernel_info_t *m_kernel;
    simt_stack *m_simt_stack;  // pdom based reconvergence context for each cta/block
    class ptx_thread_info **m_thread;
    unsigned m_block_size; //in DICE, there's no warp, programs are executed block by block
    unsigned m_liveThreadCount;
};
    
#endif