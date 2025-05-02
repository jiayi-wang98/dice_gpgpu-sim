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
#include "scoreboard.h"

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;
class dice_cfg_block_t;
class cgra_block_state_t;
class dice_metadata;
class dice_block_t;
class cgra_unit;
class dispatcher_rfu_t;
class block_commit_table;

#define MAX_CTA_PER_CGRA_CORE 8
#define MAX_THREAD_PER_CGRA_CORE 2048

enum cgra_block_stage {
  MF_DE,
  DP_CGRA,
  NUM_DICE_STAGE
};

class cgra_core_ctx {
  public:
    cgra_core_ctx(class gpgpu_sim *gpu,class simt_core_cluster *cluster,
      unsigned cgra_core_id, unsigned tpc_id,const shader_core_config *config,
      const memory_config *mem_config,shader_core_stats *stats);
    
    virtual ~cgra_core_ctx() { free(m_thread); }
    //get meta info
    class gpgpu_sim *get_gpu() {
      return m_gpu;
    }

    unsigned get_dice_trace_sampling_core();
    kernel_info_t *get_kernel_info() { return m_kernel; }
    class ptx_thread_info **get_thread_info() {
      return m_thread;
    }
    unsigned get_max_block_size() const { return m_max_block_size;}
    unsigned get_n_active_cta() const { return m_active_blocks; }
    
    unsigned translate_local_memaddr(
      address_type localaddr, unsigned tid, unsigned num_shader,
      unsigned datasize, new_addr_type *translated_addrs);
    void reinit(unsigned start_thread, unsigned end_thread,bool reset_not_completed);
    unsigned sim_init_thread(
      kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
      unsigned threads_left, unsigned num_threads, cgra_core_ctx *cgra,
      unsigned hw_cta_id, gpgpu_t *gpu);
    void init_CTA(unsigned cta_id, unsigned start_thread,
      unsigned end_thread, unsigned ctaid,int cta_size, kernel_info_t &kernel);
    address_type next_meta_pc(int tid) const;
    //dice_cfg_block_t* get_next_dice_block(address_type pc);
    void get_icnt_power_stats(long &n_simt_to_mem,long &n_mem_to_simt) const;
    void get_cache_stats(cache_stats &cs);
    float get_current_occupancy(unsigned long long &active,unsigned long long &total) const;
    //stack operation
    void initializeSIMTStack(unsigned hw_cta_id, unsigned num_threads);
    void deleteSIMTStack();
    void get_pdom_stack_top_info(unsigned hw_cta_id, unsigned *pc,unsigned *rpc) const;
    void updateSIMTStack(unsigned hw_cta_id, dice_cfg_block_t *cfg_block);
    void set_predict_pc(address_type pc);
    void clear_predict_pc();
    address_type branch_predictor(class dice_metadata *metadata);
    //thread operation
    bool ptx_thread_done(unsigned hw_thread_id) const;
    void execute_CFGBlock(cgra_block_state_t* cfg_block);
    void execute_1thread_CFGBlock(cgra_block_state_t* cgra_block,unsigned tid);
    void checkExecutionStatusAndUpdate(cgra_block_state_t* cfg_block, unsigned tid);

    //hardware simulation
    void create_front_pipeline();
    void create_dispatcher();
    void create_execution_unit();
    void cycle();
    //outer execution pipeline 
    void clear_fetch_stalled_by_simt_stack(unsigned hw_cta_id, unsigned fetch_waiting_block_id);
    bool fetch_stalled_by_simt_stack(unsigned hw_cta_id);
    unsigned get_fetch_waiting_block_id(unsigned hw_cta_id);

    void set_exec_stalled_by_writeback_buffer_full();
    void clear_exec_stalled_by_writeback_buffer_full();
    bool is_exec_stalled_by_writeback_buffer_full() const { return m_exec_stalled_by_writeback_buffer_full; }
    bool is_exec_stalled_by_ldst_unit_queue_full() const { return m_exec_stalled_by_ldst_unit_queue_full; }
    bool check_ldst_unit_stall();
    bool is_exec_stalled() const { return m_exec_stalled_by_writeback_buffer_full || m_exec_stalled_by_ldst_unit_queue_full; }
    void set_exec_stalled_by_ldst_unit_queue_full();
    void clear_exec_stalled_by_ldst_unit_queue_full();

    void cta_schedule();
    void fetch_metadata();
    void fetch_bitstream();
    void decode();
    void execute();
    void exec(unsigned tid);
    //inner pipeline in execute();
    void dispatch();
    void cgra_execute_block();
    void writeback();
    void complete_cta(unsigned hw_cta_id);

    //issue block to core
    unsigned isactive() const {
      if (m_active_blocks > 0)
        return 1;
      else
        return 0;
    }

    void cache_flush();
    void cache_invalidate();
    dice_cfg_block_t* get_dice_cfg_block(address_type pc, unsigned cta_id);
    void set_max_cta(const kernel_info_t &kernel);
    unsigned get_kernel_block_size() const { return m_kernel_block_size; }
    void set_kernel_block_size(unsigned block_size) { m_kernel_block_size = block_size; }
    void set_kernel(kernel_info_t *k) {
      assert(k);
      m_kernel = k;
      printf("DICE-Sim uArch: CGRA_core %d bind to kernel %u \'%s\'\n", m_cgra_core_id,
             m_kernel->get_uid(), m_kernel->name().c_str());
      set_kernel_block_size(m_kernel->threads_per_cta());
    }
    kernel_info_t *get_kernel() { return m_kernel; }
    bool can_issue_1block(kernel_info_t &kernel);
    void issue_block2core(kernel_info_t &kernel);
    // used for local address mapping with single kernel launch
    unsigned kernel_max_cta_per_shader;
    unsigned kernel_padded_threads_per_cta;
    const shader_core_config *get_config() const { return m_config; }
    unsigned get_not_completed() const { return m_not_completed; }

    void inc_simt_to_mem(unsigned n_flits) {
      m_stats->n_simt_to_mem[m_cgra_core_id] += n_flits;
    }
    void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);
    bool fetch_unit_response_buffer_full() const { return false; }
    bool bitstream_unit_response_buffer_full() const { return false; }
    bool ldst_unit_response_buffer_full() const;
    void accept_metadata_fetch_response(mem_fetch *mf);
    void accept_bitstream_fetch_response(mem_fetch *mf);
    void accept_ldst_unit_response(mem_fetch *mf);
    unsigned get_id() const { return m_cgra_core_id; }
    unsigned get_cta_size(unsigned hw_cta_id);
    void store_ack(class mem_fetch *mf);

    //get stack
    simt_stack* get_simt_stack(unsigned hw_cta_id) const {
      return m_simt_stack[hw_cta_id];
    }

    unsigned get_cta_start_tid(unsigned hw_cta_id) const;

    unsigned get_cta_end_tid(unsigned hw_cta_id) const;

    //status
    void get_L1I_sub_stats(struct cache_sub_stats &css) const ;
    void get_L1B_sub_stats(struct cache_sub_stats &css) const ;
    void get_L1D_sub_stats(struct cache_sub_stats &css) const ;
    void get_L1C_sub_stats(struct cache_sub_stats &css) const ;
    void get_L1T_sub_stats(struct cache_sub_stats &css) const ;
    void incregfile_reads(unsigned active_count) {
      m_stats->m_read_regfile_acesses[m_cgra_core_id] =
          m_stats->m_read_regfile_acesses[m_cgra_core_id] + active_count;
    }
    void incregfile_writes(unsigned active_count) {
      m_stats->m_write_regfile_acesses[m_cgra_core_id] =
          m_stats->m_write_regfile_acesses[m_cgra_core_id] + active_count;
    }
    class block_commit_table *get_block_commit_table() {
      return m_block_commit_table;
    }
  protected:
    unsigned m_cgra_core_id;
    unsigned m_tpc;  // texture processor cluster id (aka, node id when using
    // interconnect concentration)
    class simt_core_cluster *m_cluster;
    class gpgpu_sim *m_gpu;
    const shader_core_config *m_config; //DICE-TODO: need to change to cgra_core_config or dice_config
    const memory_config *m_memory_config;
    kernel_info_t *m_kernel;
    
    address_type m_predict_pc;
    bool m_predict_pc_set;//if m_predict_pc is valid or not
    class ptx_thread_info **m_thread;
    unsigned m_max_block_size; //hardware support maximum block size
    unsigned m_kernel_block_size; //in DICE, there's no warp, programs are executed block by block
    unsigned m_liveThreadCount;
    //unsigned m_num_cta_live_threads[MAX_CTA_PER_CGRA_CORE];  // CTAs status
    unsigned m_not_completed;  // number of threads to be completed (==0 when all thread on this core completed)
    std::bitset<MAX_THREAD_PER_CGRA_CORE> m_active_threads;
    unsigned m_active_blocks;

    // statistics
    shader_core_stats *m_stats;
    unsigned long long m_last_inst_gpu_sim_cycle;
    unsigned long long m_last_inst_gpu_tot_sim_cycle;

  
    // thread contexts
    thread_ctx_t *m_threadState;

    // interconnect interface
    mem_fetch_interface *m_icnt;
    shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

    std::vector<cgra_block_state_t*> m_cgra_block_state;//current decoded block state
    // fetch
    bool m_exec_stalled_by_writeback_buffer_full;
    bool m_exec_stalled_by_ldst_unit_queue_full;
    unsigned m_fetch_waiting_block_id;
  
    class cta_status_table *m_cta_status_table;//holding status for multiple CTAs
    class fetch_scheduler *m_fetch_scheduler;
    read_only_cache *m_L1I;  // instruction cache
    ifetch_buffer_t m_metadata_fetch_buffer;

    std::vector <simt_stack *> m_simt_stack;  // pdom based reconvergence context for each cta/block
    //decode & bitstream fetch and load
    read_only_cache *m_L1B;  // bitstreamcache
    ifetch_buffer_t m_bitstream_fetch_buffer;

    //read operands and dispatch
    dispatcher_rfu_t *m_dispatcher_rfu;

    //execute
    cgra_unit *m_cgra_unit;
    Scoreboard *m_scoreboard;
    
    //ldst_unit
    ldst_unit *m_ldst_unit;

    //writeback block commit table
    block_commit_table *m_block_commit_table;
  };


class block_commit_table{
  public:
    block_commit_table(class gpgpu_sim* gpu, class cgra_core_ctx* cgra_core);
    ~block_commit_table() {
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i]) delete m_commit_table[i];
      }
    }
    cgra_block_state_t *get(unsigned index) { return m_commit_table[index]; }
    void reserve(unsigned index, cgra_block_state_t *block) { assert(m_commit_table[index]==NULL); m_commit_table[index] = block; }
    void release(unsigned index) {
      assert(m_commit_table[index] != NULL);
      delete m_commit_table[index];
      m_commit_table[index] = NULL;
      m_ret[index] = false;
    }

    bool available() {
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] == NULL) return true;
      }
      return false;
    }

    unsigned get_available_index() {
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] == NULL) return i;
      }
      assert(0);
      return 0;
    }

    bool is_full() {
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] == NULL) return false;
      }
      return true;
    }

    bool is_empty() {
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] != NULL) return false;
      }
      return true;
    }

    unsigned number_of_occupied() {
      unsigned count = 0;
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] != NULL) count++;
      }
      return count;
    }

    unsigned number_of_empty() {
      unsigned count = 0;
      for (unsigned i = 0; i < m_max_block_size; i++) {
        if (m_commit_table[i] == NULL) count++;
      }
      return count;
    }

    void check_and_release();

    void mark_return_block(unsigned hw_cta_id);

    unsigned find_block_index(unsigned hw_cta_id);

    bool check_block_exist(unsigned hw_cta_id);

  private:
    class gpgpu_sim *m_gpu;
    class cgra_core_ctx *m_cgra_core;
    unsigned m_max_block_size;
    std::vector<cgra_block_state_t *> m_commit_table;
    std::vector<bool> m_ret;
};

class exec_cgra_core_ctx : public cgra_core_ctx {
  public:
   exec_cgra_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                        unsigned cgra_core_id, unsigned tpc_id,
                        const shader_core_config *config,
                        const memory_config *mem_config,
                        shader_core_stats *stats)
       : cgra_core_ctx(gpu, cluster, cgra_core_id, tpc_id, config, mem_config,
                         stats) {
     create_front_pipeline();
     create_dispatcher();
     create_execution_unit();
   }
 };


class cta_status_table{
  public:
    cta_status_table(class gpgpu_sim* gpu, cgra_core_ctx* m_cgra_core, unsigned max_cta_per_core) {
      m_gpu = gpu;
      m_cgra_core = m_cgra_core;
      m_max_cta_per_core = max_cta_per_core;
      m_cta_status.resize(max_cta_per_core);
    }
    ~cta_status_table() {
    }

    unsigned get_size() const {
      return m_max_cta_per_core;
    }

    void reset() {
      for (unsigned i = 0; i < m_max_cta_per_core; i++) {
        m_cta_status[i].m_valid = false;
        m_cta_status[i].m_fetch_stalled_by_simt_stack = false;
        m_cta_status[i].m_ret = false;
      }
    }

    bool full() const {
      for (unsigned i = 0; i < m_max_cta_per_core; i++) {
        if (!m_cta_status[i].m_valid) return false;
      }
      return true;
    }

    unsigned get_free_index() {
      for (unsigned i = 0; i < m_max_cta_per_core; i++) {
        if (!m_cta_status[i].m_valid) return i;
      }
      assert(0);
      return 0;
    }


    void init_cta(unsigned hw_cta_id) {
      assert(hw_cta_id < m_max_cta_per_core);
      m_cta_status[hw_cta_id].m_valid = true;
      m_cta_status[hw_cta_id].m_fetch_stalled_by_simt_stack = false;
      m_cta_status[hw_cta_id].m_fetch_waiting_block_id = 0;
      m_cta_status[hw_cta_id].m_num_live_threads = 0;
    }

    void set_num_live_threads(unsigned hw_cta_id, unsigned start_thread, unsigned end_thread) {
      assert(hw_cta_id < m_max_cta_per_core);
      unsigned num_live_threads = end_thread - start_thread;
      m_cta_status[hw_cta_id].m_num_live_threads = num_live_threads;
      m_cta_status[hw_cta_id].m_start_thread = start_thread;
      m_cta_status[hw_cta_id].m_end_thread = end_thread;
      m_cta_status[hw_cta_id].m_cta_size = (num_live_threads % 256) ? 256*(num_live_threads/256+1) : num_live_threads;
    }

    void deactive_cta(unsigned hw_cta_id) {
      assert(hw_cta_id < m_max_cta_per_core);
      m_cta_status[hw_cta_id].m_valid = false;
      m_cta_status[hw_cta_id].m_fetch_stalled_by_simt_stack = false;
      m_cta_status[hw_cta_id].m_ret = false;
    }

    bool is_free(unsigned hw_cta_id) const {
      assert(hw_cta_id < m_max_cta_per_core);
      return !m_cta_status[hw_cta_id].m_valid;
    }

    void decrease_num_live_threads(unsigned hw_cta_id) {
      assert(hw_cta_id < m_max_cta_per_core);
      assert(m_cta_status[hw_cta_id].m_num_live_threads > 0);
      m_cta_status[hw_cta_id].m_num_live_threads--;
      if(m_cta_status[hw_cta_id].m_num_live_threads == 0) {
        deactive_cta(hw_cta_id);
      }
    }

    unsigned get_cta_size(unsigned hw_cta_id) const {
      assert(hw_cta_id < m_max_cta_per_core);
      return m_cta_status[hw_cta_id].m_cta_size;
    }
  

    void set_fetch_stalled_by_simt_stack(unsigned hw_cta_id, unsigned fetch_waiting_block_id, address_type prefetch_pc){
      assert(m_cta_status[hw_cta_id].m_valid);
      m_cta_status[hw_cta_id].m_fetch_stalled_by_simt_stack = true;
      m_cta_status[hw_cta_id].m_prefetch_pc = prefetch_pc;
      m_cta_status[hw_cta_id].m_fetch_waiting_block_id = fetch_waiting_block_id;
    }

    bool fetch_stalled_by_simt_stack(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_fetch_stalled_by_simt_stack;
    }

    void clear_fetch_stalled_by_simt_stack(unsigned hw_cta_id, unsigned fetch_waiting_block_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      assert(fetch_waiting_block_id == m_cta_status[hw_cta_id].m_fetch_waiting_block_id);
      m_cta_status[hw_cta_id].m_fetch_stalled_by_simt_stack = false;
    }

    unsigned get_start_thread(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_start_thread;
    }

    unsigned get_end_thread(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_end_thread;
    }

    unsigned get_prefetch_pc(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_prefetch_pc;
    }

    unsigned get_fetch_waiting_block_id(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_fetch_waiting_block_id;
    }

    bool is_ret(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      return m_cta_status[hw_cta_id].m_ret;
    }

    void set_ret(unsigned hw_cta_id){
      assert(m_cta_status[hw_cta_id].m_valid);
      m_cta_status[hw_cta_id].m_ret = true;
    }

  private:
    class gpgpu_sim *m_gpu;
    class cgra_core_ctx *m_cgra_core;
    unsigned m_max_cta_per_core;
    struct cta_status_entry {
      cta_status_entry() {
        m_valid = false;
        m_fetch_stalled_by_simt_stack = false;

        m_fetch_waiting_block_id = 0;
        m_num_live_threads = 0;
        m_cta_size = 256;
        m_start_thread = 0;
        m_end_thread = 0;
        m_ret = false;
      }
      bool m_valid;
      bool m_fetch_stalled_by_simt_stack;
      unsigned m_prefetch_pc;
      unsigned m_fetch_waiting_block_id;
      unsigned m_num_live_threads;
      unsigned m_cta_size;
      unsigned m_start_thread;
      unsigned m_end_thread;
      bool m_ret;
    };
    std::vector<cta_status_entry> m_cta_status;
};

class fetch_scheduler{
  public:
    fetch_scheduler(class gpgpu_sim *gpu, class cgra_core_ctx *cgra_core, cta_status_table *cta_status_table) {
      m_gpu = gpu;
      m_cgra_core = cgra_core;
      m_cta_status_table = cta_status_table;
      m_previous_fetch_pc = unsigned(-1);
      cta_status_table_size = cta_status_table->get_size();
      m_last_fetch_cta_id = cta_status_table_size-1;
    }
    ~fetch_scheduler() {
    }
    unsigned next_fetch_block();
  private:
    class gpgpu_sim *m_gpu;
    class cgra_core_ctx *m_cgra_core;
    class cta_status_table *m_cta_status_table;
    address_type m_previous_fetch_pc;
    unsigned cta_status_table_size;
    unsigned m_last_fetch_cta_id;
};

//hardware runtime status
 class cgra_block_state_t{
    public:
    cgra_block_state_t(class cgra_core_ctx *cgra_core, unsigned block_size)
         : m_cgra_core(cgra_core), m_block_size(block_size) {
       m_stores_outstanding = 0;
       m_thread_in_pipeline = 0;
       reset();
     }
     void reset() {
       assert(m_stores_outstanding == 0);
       assert(m_thread_in_pipeline == 0);
       m_imiss_pending = false;
       m_bmiss_pending = false;
       m_decoded = false;
       n_completed = m_block_size;
       m_n_atomic = 0;
       m_membar = false;
       m_done_exit = false;
       m_last_fetch = 0;
       m_last_bitstream_fetch = 0;
       m_next = 0;
       dispatch_completed = false;
       cgra_fabric_completed = false;
       m_num_loads_done = 0;
       m_num_stores_done = 0;
       writeback_completed = false;
       is_prefetch = false;
       m_metadata_buffer.m_valid = false;
       m_metadata_buffer.m_bitstream_valid = false;
       m_valid = false;
     }
     void init(address_type start_metadata_pc, unsigned cta_id,
               const simt_mask_t &active) {
       m_cta_id = cta_id;
       m_next_metadata_pc = start_metadata_pc;
       assert(n_completed >= active.count());
       assert(n_completed <= m_block_size);
       n_completed -= active.count();  // active threads are not yet completed
       m_active_threads = active;
       m_done_exit = false;
       m_metadata_buffer.m_valid = false;
       m_metadata_buffer.m_bitstream_valid = false;
       m_valid = true;
     }

     void set_completed(unsigned lane) {
      assert(m_active_threads.test(lane));
      m_active_threads.reset(lane);
      n_completed++;
    }

    bool dummy() const { 
      if(!m_valid) return true; //empty block
      //if(hardware_done()) return true;
      return false;
    }

     unsigned active_count () const;
     
     void set_dispatch_done() {
       dispatch_completed = true;
     }
     bool dispatch_done() const { return dispatch_completed; }

     void set_decode_done() { m_decoded = true; }

     void clear_decode_done() { m_decoded = false; }//this will happen when prefetching

     bool decode_done() const { return m_decoded; }

     bool is_parameter_load();

     bool barrier_reached();

     bool ready_to_dispatch(){
       if (m_metadata_buffer.m_valid && m_metadata_buffer.m_bitstream_valid) {
         if(is_prefetch) return false;
         if(m_decoded) {
          return true;
         }
       }
       return false;
     }

     void set_last_fetch(unsigned long long sim_cycle) {
      m_last_fetch = sim_cycle;
     }

     void set_last_bitstream_fetch(unsigned long long sim_cycle) {
      m_last_bitstream_fetch = sim_cycle;
     }

     bool hardware_done() {
      return functional_done() && stores_done() && !thread_in_pipeline();
     }

     bool done_exit() const { return m_done_exit; }
     void set_done_exit() { m_done_exit = true; }

     bool thread_in_pipeline() const { return m_thread_in_pipeline > 0; }
     void inc_thread_in_pipeline() { m_thread_in_pipeline++; }
     void dec_thread_in_pipeline() {
       assert(m_thread_in_pipeline > 0);
       m_thread_in_pipeline--;
     }

     void metadata_buffer_fill(dice_cfg_block_t *cfg_block);

     void bitstream_buffer_fill() {
      m_metadata_buffer.m_bitstream_valid = true;
     }

     bool functional_done() const {
      return n_completed==m_block_size;
     }
     bool cgra_fabric_done() const {
      return cgra_fabric_completed;
     }
     void set_cgra_fabric_done() {
      cgra_fabric_completed = true;
     }

     bool block_done(){
      return loads_done() && stores_done(); 
     }

     bool loads_done();
     bool stores_out();
     bool mem_access_queue_empty();

     bool stores_done() { return stores_out() && (m_stores_outstanding == 0); }
     void inc_store_req();
     void dec_store_req();
     unsigned get_stores_outstanding() const { return m_stores_outstanding; }

     bool metadata_buffer_empty() const {
      if (m_metadata_buffer.m_valid) return false;
      return true;
     }

     bool bitstream_buffer_waiting() const {
      return (m_metadata_buffer.m_valid && m_metadata_buffer.m_bitstream_valid==false);
     }

     unsigned get_unrolling_factor();

     dice_cfg_block_t* get_current_cfg_block();
     dice_metadata* get_current_metadata();
     dice_block_t *get_dice_block();
     virtual address_type get_metadata_pc() const { return m_next_metadata_pc; }
     void set_next_pc(address_type pc) { m_next_metadata_pc = pc; }
     virtual address_type get_bitstream_pc(); 
     unsigned get_bitstream_size();
     unsigned get_block_latency();
     
     bool imiss_pending() const { return m_imiss_pending; }
     void set_imiss_pending() { m_imiss_pending = true; }
     void clear_imiss_pending() { m_imiss_pending = false; }

     bool bmiss_pending() const { return m_bmiss_pending; }
     void set_bmiss_pending() { m_bmiss_pending = true; }
     void clear_bmiss_pending() { m_bmiss_pending = false; }
     unsigned get_cta_id() const { return m_cta_id; }
     unsigned get_block_size() const { return m_block_size; }
   
     class cgra_core_ctx * get_cgra_core() { return m_cgra_core; }
     bool active(unsigned tid) const { return m_active_threads.test(tid); }
     simt_mask_t get_active_threads() const { return m_active_threads; }

     unsigned get_n_atomic() const { return m_n_atomic; }
     void inc_n_atomic() { m_n_atomic++; }
     void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

     void inc_number_of_loads_done() { m_num_loads_done++; }
     void dec_number_of_loads_done() { m_num_loads_done--; }
     unsigned get_number_of_loads_done() { return m_num_loads_done; }


     void inc_number_of_stores_done();
     void dec_number_of_stores_done() { m_num_stores_done--; }
     unsigned get_number_of_stores_done() { return m_num_stores_done; }

    void set_prefetch() { is_prefetch = true; }
    bool is_prefetch_block() { return is_prefetch; }
    void clear_prefetch();

    private:
     class cgra_core_ctx *m_cgra_core;
     bool m_valid;
     unsigned m_cta_id;
     unsigned m_block_size;
   
     address_type m_next_metadata_pc;
     bool dispatch_completed;  
     bool cgra_fabric_completed;
     bool writeback_completed;
     unsigned n_completed;  // number of threads in block completed
     simt_mask_t m_active_threads;
   
     bool m_decoded;
     bool m_imiss_pending; //metadata miss
     bool m_bmiss_pending; //bitstream miss pending
   
     struct ibuffer_entry {
       ibuffer_entry() {
         m_valid = false;
         m_bitstream_valid = false;
         m_cfg_block = NULL;
       }
       dice_cfg_block_t *m_cfg_block;
       bool m_valid;
       bool m_bitstream_valid;
     };
   
     ibuffer_entry m_metadata_buffer;
     unsigned m_next;
   
     unsigned m_n_atomic;  // number of outstanding atomic operations
     bool m_membar;        // if true, block is waiting at memory barrier
   
     bool m_done_exit;  // true once thread exit has been registered for threads in this block
   
     unsigned long long m_last_fetch;
     unsigned long long m_last_bitstream_fetch;
   
     unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                     // acknowledged
     unsigned m_num_loads_done;  // number of load that have been done       
     unsigned m_num_stores_done;  // number of stores that have been acknowledged                   
     unsigned m_thread_in_pipeline;

     bool is_prefetch;
};

#define MAX_CGRA_FABRIC_LATENCY 32
class cgra_unit {
  public:
   cgra_unit(const shader_core_config *config, cgra_core_ctx *cgra_core, cgra_block_state_t **block);
   ~cgra_unit() {}
  
   // modifiers
   void reinit(){
      is_busy = false;
      stalled_by_ldst_unit_queue_full = false;
      stalled_by_wb_buffer_full = false;
      for(unsigned i=0; i<MAX_CGRA_FABRIC_LATENCY; i++){
        shift_registers[i] = unsigned(-1);
      }
      m_num_executed_thread = 0;
   }
   void flush_pipeline() {
      for(unsigned i=0; i<MAX_CGRA_FABRIC_LATENCY; i++){
        shift_registers[i] = unsigned(-1);
      }
      m_num_executed_thread = 0;
   }
   void exec(unsigned tid) {
    shift_registers[0]=tid;
   }
   void cycle();
   void set_latency(unsigned l) { assert(l<MAX_CGRA_FABRIC_LATENCY); m_latency = l; }
   unsigned out_tid() {
      return shift_registers[m_latency];
   }
   unsigned out_valid() {
      if(stalled()) return false;
      return out_tid() != unsigned(-1);
   }
 
   // accessors
   bool is_idle() const {
     return !is_busy;
   }
   bool stallable() const {};
   void print() const;
   const char *get_name() { return m_name.c_str(); }
   bool is_stalled() { return (stalled_by_wb_buffer_full || stalled_by_ldst_unit_queue_full); }
   void set_stalled_by_wb_buffer_full() { stalled_by_wb_buffer_full = true; }
   void clear_stalled_by_wb_buffer_full() { stalled_by_wb_buffer_full = false; }
   void set_stalled_by_ldst_unit_queue_full() { stalled_by_ldst_unit_queue_full = true; }
   void clear_stalled_by_ldst_unit_queue_full() { stalled_by_ldst_unit_queue_full = false; }
   bool stalled() const { return stalled_by_ldst_unit_queue_full || stalled_by_wb_buffer_full; }
   void inc_num_executed_thread() { m_num_executed_thread++; }
   unsigned get_num_executed_thread() { return m_num_executed_thread; }

  protected:
   unsigned m_latency;
   std::string m_name;
   cgra_core_ctx *m_cgra_core;
   const shader_core_config *m_config; //DICE-TODO: need to change to cgra_core_config or dice_config;
   unsigned shift_registers[MAX_CGRA_FABRIC_LATENCY]; //storing thread id
   bool is_busy;
   cgra_block_state_t **m_executing_block;
   bool stalled_by_wb_buffer_full;
   bool stalled_by_ldst_unit_queue_full;
   unsigned m_num_executed_thread;
 };

 class rf_bank_controller {
  public:
    rf_bank_controller(unsigned bank_id, class dispatcher_rfu_t *rfu);
    bool wb_buffer_full();
    void push_to_cgra_wb_buffer(unsigned tid, cgra_block_state_t *block) {
      m_cgra_writeback_buffer.push_back(std::make_pair(tid, block));
    }

    void push_to_ldst_wb_buffer(unsigned tid, cgra_block_state_t *block) {
      m_ldst_writeback_buffer.push_back(std::make_pair(tid, block));
    }

    void cycle();
    
    //bool occupied_by_ldst_unit;
    bool occupied_by_ldst_unit() const {
      return m_ldst_writeback_buffer.size() > 0;
    }

    bool ldst_buffer_full() const {
      return m_ldst_writeback_buffer.size() >= m_ldst_buffer_size;
    }

    unsigned ldst_buffer_credit() const {
      return m_ldst_buffer_size - m_ldst_writeback_buffer.size();
    }
  private:
    unsigned m_cgra_buffer_size;
    unsigned m_ldst_buffer_size;
    unsigned m_bank_id;
    std::list<std::pair<unsigned,cgra_block_state_t*>> m_cgra_writeback_buffer; //(tid,block)
    std::list<std::pair<unsigned,cgra_block_state_t*>> m_ldst_writeback_buffer; //(tid,block)
    
    //backward pointer
    dispatcher_rfu_t *m_rfu;
};

 class dispatcher_rfu_t{
  public:
    dispatcher_rfu_t(const shader_core_config *config, cgra_core_ctx *cgra_core, cgra_block_state_t **block, Scoreboard *scoreboard){
      m_config = config;
      m_cgra_core = cgra_core;
      m_dispatching_block = block;
      m_dispatched_thread = 0;
      m_last_dispatched_tid.resize(32);
      m_scoreboard = scoreboard;
      m_num_read_access = 0;
      m_num_write_access = 0;
      init_rf_bank_controller();
    }
    void init_rf_bank_controller(){
      for(unsigned i=0; i<m_config->dice_cgra_core_max_rf_banks; i++){
        m_rf_bank_controller.push_back(new rf_bank_controller(i, this));
      }
    }
    ~dispatcher_rfu_t() {
      for(unsigned i=0; i<m_config->dice_cgra_core_max_rf_banks; i++){
        delete m_rf_bank_controller[i];
      }
    }
    void rf_cycle();
    void dispatch();
    void writeback_cgra(cgra_block_state_t* block,unsigned tid);
    bool writeback_ldst(cgra_block_state_t* block,unsigned reg_num, std::set<unsigned> tids);
    unsigned next_active_thread(unsigned unrolling_factor, unsigned unrolling_index);
    bool idle() { return m_dispatching_block == NULL; }
    cgra_block_state_t *get_dispatching_block() { return (*m_dispatching_block); }
    bool current_finished() { return (*m_dispatching_block)->dispatch_done(); }
    bool writeback_buffer_full(dice_metadata *metadata) const;
    const shader_core_config *get_config() { return m_config; }
    bool exec_stalled() const { return m_cgra_core->is_exec_stalled(); }
    void read_operands(dice_metadata *metadata, unsigned tid);
    bool can_writeback_ldst_reg(unsigned reg_num, unsigned count);
    bool can_writeback_ldst_regs(std::set<unsigned> regs, std::set<unsigned> tids);

  private:
    const shader_core_config *m_config;
    cgra_core_ctx *m_cgra_core;
    cgra_block_state_t **m_dispatching_block;
    unsigned m_dispatched_thread;
    std::vector<unsigned> m_last_dispatched_tid;
    std::list<unsigned> m_ready_threads;
    Scoreboard *m_scoreboard;
    std::vector<rf_bank_controller*> m_rf_bank_controller;

    //statistics
    unsigned long long m_num_read_access;
    unsigned long long m_num_write_access;
 };


 class dice_mem_request_queue{
  //TODO coallesce memory request 
  //Add mshr for memory request
  public:
    dice_mem_request_queue(const shader_core_config *config, class ldst_unit* ldst_unit);

    void push_ld_request(mem_access_t access, unsigned port) {
      m_ld_req_queue[port].push_back(access);
      m_ld_port_credit[port]--;
      //printf("DICE-Sim uArch:  push_ld_request, port = %d, credit = %d\n",  port, m_ld_port_credit[port]);
      //fflush(stdout);
      assert(access.get_cgra_block_state() != NULL);
    }

    void push_ld_request_pre_coalesce(mem_access_t access, unsigned port) {
      m_ld_req_queue_pre_coalesce[port].push_back(access);
      m_ld_port_credit[port]--;
    }

    void push_st_request_pre_coalesce(mem_access_t access, unsigned port) {
      m_st_req_queue_pre_coalesce[port].push_back(access);
      m_st_port_credit[port]--;
    }

    void push_st_request(mem_access_t access, unsigned port) {
      m_st_req_queue[port].push_back(access);
      m_st_port_credit[port]--;
    }

    void pop_ld_request(unsigned port) {
      m_ld_req_queue[port].pop_front();
      m_ld_port_credit[port]++;
    }

    void pop_st_request(unsigned port) {
      m_st_req_queue[port].pop_front();
      m_st_port_credit[port]++;
    }

    void pop_ld_request_pre_coalesce(unsigned port) {
      m_ld_req_queue_pre_coalesce[port].pop_front();
      m_ld_port_credit[port]++;
    }

    void pop_st_request_pre_coalesce(unsigned port) {
      m_st_req_queue_pre_coalesce[port].pop_front();
      m_st_port_credit[port]++;
    }
    
    void pop_request(unsigned port) {
      if(port<m_ld_req_queue.size()) pop_ld_request(port);
      else {
        if(port-m_ld_req_queue.size() >= m_st_req_queue.size()) {
          printf("DICE-Sim uArch: pop_request, port = %d, m_ld_req_queue.size() = %d, m_st_req_queue.size() = %d\n", port, m_ld_req_queue.size(), m_st_req_queue.size());
          fflush(stdout);
        }
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        pop_st_request(port-m_ld_req_queue.size());
      }
    }

    bool ld_port_full(unsigned port) const {
      return m_ld_port_credit[port] == 0;
    }

    bool st_port_full(unsigned port) const {
      return m_st_port_credit[port] == 0;
    }

    bool is_full(unsigned port) const {
      if(port<m_ld_req_queue.size()) return ld_port_full(port);
      else {
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        return st_port_full(port-m_ld_req_queue.size());
      }
    }

    mem_access_t get_ld_request(unsigned port) {
      assert(!m_ld_req_queue[port].empty());
      return m_ld_req_queue[port].front();
    }

    mem_access_t get_st_request(unsigned port) {
      assert(!m_st_req_queue[port].empty());
      return m_st_req_queue[port].front();
    }

    mem_access_t get_request(unsigned port) {
      if(port<m_ld_req_queue.size()) return get_ld_request(port);
      else {
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        return get_st_request(port-m_ld_req_queue.size());
      } 
    }

    bool queue_empty(unsigned port) {
      if(port<m_ld_req_queue.size()) return m_ld_req_queue[port].empty();
      else {
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        return m_st_req_queue[port-m_ld_req_queue.size()].empty();
      }
    }

    unsigned port_num() const { return m_ld_req_queue.size() + m_st_req_queue.size(); }

    unsigned get_next_process_port_constant();
    unsigned get_next_process_port_texture();
    unsigned get_next_process_port_memory();
    unsigned get_next_process_port_shared();
    void set_last_processed_port_constant(unsigned port) { m_last_processed_port_contant = port; }
    void set_last_processed_port_texture(unsigned port) { m_last_processed_port_texture = port; }
    void set_last_processed_port_memory(unsigned port) { m_last_processed_port_memory = port; }
    void set_last_processed_port_shared(unsigned port) { m_last_processed_port_shared = port; }

    void update_coaleasing_counter();
    bool coalesce_interval_done(unsigned port){
      printf("coalesce_interval_done start, port = %d\n", port);
      if(port>= m_ld_req_queue.size()) {
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        //check store
        if(m_st_coalescing_counter[port-m_ld_req_queue.size()] = temporal_coalescing_interval) {
          m_st_coalescing_counter[port-m_ld_req_queue.size()] = 0;
          printf("coalesce_interval_done end, t\n"); 
          return true;
        } else {
          printf("coalesce_interval_done end, f\n"); 
          return false;
        }
      } else {
        //check load
        if(m_ld_coalescing_counter[port] == temporal_coalescing_interval) {
          m_ld_coalescing_counter[port] = 0;
          printf("coalesce_interval_done end, t\n"); 
          return true;
        } else {
          printf("coalesce_interval_done end,f \n"); 
          return false;
        }
      }
    }

    void do_coalescing(unsigned port) {
      if(port>= m_ld_req_queue.size()) {
        assert(port-m_ld_req_queue.size() < m_st_req_queue.size());
        //check store
        do_st_coalescing(port-m_ld_req_queue.size());
      } else {
        //check load
        do_ld_coalescing(port);
      }
    }

    void coalesce_cycle();

    void do_ld_coalescing(unsigned port);
    void do_st_coalescing(unsigned port);

    const shader_core_config *m_config;
    ldst_unit *m_ldst_unit;
    std::vector<unsigned> m_ld_port_credit;
    std::vector<unsigned> m_st_port_credit;
    std::vector<std::list<mem_access_t>> m_ld_req_queue_pre_coalesce;
    std::vector<std::list<mem_access_t>> m_st_req_queue_pre_coalesce;
    std::vector<std::list<mem_access_t>> m_ld_req_queue;
    std::vector<std::list<mem_access_t>> m_st_req_queue;
    struct dice_transaction_info {
      std::bitset<4> chunks;  // bitmask: 32-byte chunks accessed
      mem_access_byte_mask_t bytes;
      active_mask_t active;  // threads in this transaction
  
      bool test_bytes(unsigned start_bit, unsigned end_bit) {
        for (unsigned i = start_bit; i <= end_bit; i++)
          if (bytes.test(i)) return true;
        return false;
      }
      std::set<unsigned> ld_dest_regs;
      std::set<unsigned> port_idx;
      mem_access_type access_type;
      memory_space_t space;
      std::set<unsigned> active_threads;
    };

    void memory_coalescing_arch_reduce(bool is_write, const dice_transaction_info &info,new_addr_type addr, unsigned segment_size, unsigned port, cgra_block_state_t *block);

  private:
    unsigned m_last_processed_port_contant;
    unsigned m_last_processed_port_texture;
    unsigned m_last_processed_port_memory;
    unsigned m_last_processed_port_shared;
    std::vector<unsigned> m_ld_coalescing_counter;
    std::vector<unsigned> m_st_coalescing_counter;
    unsigned enable_temporal_coaleascing;
    unsigned temporal_coalescing_interval;
 };
#endif