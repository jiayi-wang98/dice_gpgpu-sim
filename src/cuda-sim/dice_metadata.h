#ifndef DICE_METADATA_H_INCLUDED
#define DICE_METADATA_H_INCLUDED
#include <string>
#include "../abstract_hardware_model.h"
#define DICE_METADATA_LINEBUF_SIZE 4096
class gpgpu_context;
typedef void* yyscan_t;


class operand_info;
class ptx_instruction;
struct dice_block_t;

const unsigned MAX_ACCESSES_PER_BLOCK_PER_THREAD = 8;

class dice_metadata {
  public:
    dice_metadata(class gpgpu_context* ctx) {
      gpgpu_ctx = ctx;
      m_source_file = "";
      meta_id = 0;
      unrolling_factor = 1;
      unrolling_strategy = 0;
      latency = 0;
      bitstream_label = "";
      bitstream_length = 0;
      num_store = 0;
      branch = false;
      uni_bra = false;
      branch_pred = NULL;
      branch_pred_pole = true;
      branch_target_meta_id = 0;
      reconvergence_meta_id = 0;
      is_exit = false;
      is_ret = false;
      is_entry = false;
      is_parameter_load = false;
      m_dicemeta_mem_index = 0;
      m_PC = 0;
      m_size = 32; //bytes
    }

    ~dice_metadata() {
    }
    void set_m_metadata_mem_index(unsigned index) { m_dicemeta_mem_index = index; }
    void set_PC(addr_t PC) { m_PC = PC; }
    addr_t get_PC() const { return m_PC; }
  
    unsigned get_m_metadata_mem_index() { return m_dicemeta_mem_index; }
    unsigned metadata_size() { return m_size; }
    dice_block_t *get_diceblock() { return dice_block; } 
    bool has_loads() { return load_destination_regs.size() > 0; }
    bool has_stores() { return num_store > 0; }

    class gpgpu_context* gpgpu_ctx;
    std::string m_source_file;
    unsigned m_source_line;

    int meta_id;
    int unrolling_factor;
    int unrolling_strategy;
    int latency;
    std::string bitstream_label;
    int bitstream_length;
    std::list<operand_info> in_regs;
    std::list<operand_info> out_regs;
    std::list<operand_info> load_destination_regs;
    int num_store;
    bool branch;
    bool uni_bra;
    operand_info* branch_pred;
    bool branch_pred_pole;
    int branch_target_meta_id;
    int reconvergence_meta_id;
    unsigned branch_target_meta_pc;
    unsigned reconvergence_meta_pc;

    bool is_exit;
    bool is_ret;
    bool is_entry;
    bool is_parameter_load;
    dice_block_t *dice_block;

    unsigned m_dicemeta_mem_index;
    addr_t m_PC;
    unsigned m_size;  // bytes
  
    
    void dump();
};

class dice_metadata_parser {
 private:
  bool g_debug_dicemeta_generation;
  dice_metadata* g_current_dbb;
  std::list<operand_info> g_operands;
  std::list<bool> g_operand_poles;
  gpgpu_context* gpgpu_ctx;
  std::string g_current_function_name;
  function_info* g_current_function_info;

 public:
  dice_metadata_parser(gpgpu_context* ctx)
      : g_operands()
  {
    assert(ctx != NULL);
    gpgpu_ctx = ctx;
    g_current_dbb = NULL;
    g_error_detected = 0;
    g_debug_dicemeta_generation = false;
  }
  yyscan_t scanner;
  char linebuf[DICE_METADATA_LINEBUF_SIZE];
  unsigned col;
  const char* g_dice_metadata_filename;
  int g_error_detected;

  void add_operand(const char *identifier);
  void add_operand_pole(bool positive);
  void add_builtin_operand(int builtin, int dim_modifier);
  void read_parser_environment_variables();
  void create_new_dbb(int meta_id);
  void set_unrolling_factor(int factor){
    g_current_dbb->unrolling_factor = factor;
  }
  void set_unrolling_strategy(int strategy){
    g_current_dbb->unrolling_strategy = strategy;
  }
  void set_latency(int lat){
    g_current_dbb->latency = lat;
  }
  //void add_operand(...)
  void commit_dbb();
  void set_num_store(int num){
    g_current_dbb->num_store = num;
  }
  void set_branch(bool branch){
    g_current_dbb->branch = branch;
  }
  void set_branch_uni(bool uni_bra){
    g_current_dbb->uni_bra = uni_bra;
  }
  //void set_branch_pred(const char* pred);
  void set_branch_target_dbb(int dbb){
    g_current_dbb->branch_target_meta_id = dbb;
  }
  void set_reconvergence_dbb(int dbb){
    g_current_dbb->reconvergence_meta_id = dbb;
  }
  void set_is_ret(){
    g_current_dbb->is_ret = true;
  }
  void set_is_parameter_load(){
    g_current_dbb->is_parameter_load = true;
  }
  void set_bitstream_label(const char* label){
    g_current_dbb->bitstream_label = std::string(label);
  }
  void set_bitstream_length(int length){
    g_current_dbb->bitstream_length = length;
  }

  void set_function_name(const char* name);

  void set_in_regs();
  void set_out_regs();
  void set_ld_dest_regs();
  void set_branch_pred();
  void commit_function();
};

struct dice_block_t {
  dice_block_t(const std::string &block_label, unsigned ID, ptx_instruction *begin, ptx_instruction *end, 
                bool entry, bool ex)
    : label(block_label), dbb_id(ID), ptx_begin(begin), ptx_end(end), is_entry(entry), is_exit(ex) {}

  ptx_instruction *ptx_begin;
  ptx_instruction *ptx_end;
  std::vector<ptx_instruction *> ptx_instructions;
  bool is_entry;
  bool is_exit;
  unsigned dbb_id;
  std::string label;
  unsigned get_start_pc();
  unsigned get_end_pc();
  unsigned get_block_size(){
    return get_end_pc() - get_start_pc();
  }
};

void dice_metadata_assemble(std::string kname, void *kinfo);


#define MAX_LDST_UNIT_PORTS 8
//for performance simulation
class dice_cfg_block_t{
  public:
    dice_cfg_block_t(dice_metadata *metadata);
    dice_cfg_block_t(unsigned uid, unsigned block_size, dice_metadata *metadata, gpgpu_context* ctx);
    ~dice_cfg_block_t(){
      if(m_block_active_mask) delete m_block_active_mask;
    }
    address_type metadata_pc;  // program counter address of metadata
    unsigned metadata_size;   // size of metadata in bytes
    address_type reconvergence_pc;  // program counter address of reconvergence
    op_type op;  // operation type
    addr_t branch_target_meta_pc;  // program counter address of branch target
    memory_space_t space;
    gpgpu_context* gpgpu_ctx;

    std::map<unsigned, std::set<unsigned>> map_tid_invalid_writeback_regs; //map from tid to invalid reg_num set

    bool active(unsigned tid) const { return m_block_active_mask->test(tid); }
    unsigned active_count() const { return m_block_active_mask->count(); }
    unsigned get_num_stores();
    unsigned get_num_loads();
    simt_mask_t get_active_mask() const { return *m_block_active_mask; }
    void set_active(const active_mask_t &active);
    void set_not_active(unsigned tid);
    void set_addr(unsigned n, new_addr_type addr) {
      if (!m_per_scalar_thread_valid) {
        m_per_scalar_thread.resize(m_block_size);
        m_per_scalar_thread_valid = true;
      }
      assert(n < m_per_scalar_thread.size() && "Index n out of bounds");
      m_per_scalar_thread[n].memreqaddr[0] = addr;
      m_per_scalar_thread[n].count++;
    }
    void add_mem_op(unsigned n, new_addr_type addr, memory_space_t space, _memory_op_t insn_memory_op, unsigned size, unsigned ld_dest_reg = 0, unsigned enable = 1) {
      if (!m_per_scalar_thread_valid) {
        m_per_scalar_thread.resize(m_block_size);
        m_per_scalar_thread_valid = true;
      }
      assert(n < m_per_scalar_thread.size() && "Index n out of bounds");
      unsigned index=m_per_scalar_thread[n].count;
      assert(index < MAX_ACCESSES_PER_BLOCK_PER_THREAD);
      m_per_scalar_thread[n].memreqaddr[index] = addr;
      m_per_scalar_thread[n].space[index] = space;
      m_per_scalar_thread[n].mem_op[index] = insn_memory_op;
      m_per_scalar_thread[n].size[index] = size;
      m_per_scalar_thread[n].ld_dest_reg[index] = ld_dest_reg;
      m_per_scalar_thread[n].count++;
      m_per_scalar_thread[n].enable = enable;
      //printf("DICE Sim uArch [MEM_ACCESS]: thread %u, addr 0x%04x, space %d, mem_op %d, size %d ,num_of_mem_access = %d\n", n, addr, space, insn_memory_op , size,index);
      //fflush(stdout);
    }
    void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
      if (!m_per_scalar_thread_valid) {
        m_per_scalar_thread.resize(m_block_size); 
        m_per_scalar_thread_valid = true;
      }
      assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
      for (unsigned i = 0; i < num_addrs; i++)
        m_per_scalar_thread[n].memreqaddr[i] = addr[i];
      m_per_scalar_thread[n].count+=num_addrs;
    }
    dice_metadata *get_metadata() { return m_metadata; }
    void generate_mem_accesses(unsigned tid, std::list<unsigned> &masked_ops_reg);

    bool accessq_empty() const {
      for(unsigned i=0; i<MAX_LDST_UNIT_PORTS; i++){
        if(!m_accessq[i].empty()) return false;
      }
      return true;
    }

    std::vector<std::list<mem_access_t>>& get_accessq() { return m_accessq; }

    void pop_mem_access(unsigned port) {
      assert(!m_accessq[port].empty());
      m_accessq[port].pop_front();
    }

    void print_m_accessq() {
      if (accessq_empty())
        return;
      else {
        printf("Printing mem access generated\n");
        for(int i=0;i<MAX_LDST_UNIT_PORTS;i++){
          if(m_accessq[i].empty()) continue;
          printf("LDST Unit: Port %d\n",i);
          std::list<mem_access_t>::iterator it;
          for (it = m_accessq[i].begin(); it != m_accessq[i].end(); ++it) {
            printf("MEM_TXN_GEN:%s:%llx, Size:%d, LDST Unit Port:%d \n",
                   mem_access_type_str(it->get_type()), it->get_addr(),
                   it->get_size(), it->get_ldst_port_num());
          }
        }
      }
    }
    dice_block_t *get_diceblock() { return m_diceblock; }
    void print_mem_ops(unsigned core_id = 0){
      for (unsigned i = 0; i < m_per_scalar_thread.size(); i++){
        for (unsigned j = 0; j < m_per_scalar_thread[i].count; j++){
          printf("DICE Sim: [MEM_ACCESS]: core %d, thread %u, addr 0x%04x, space %d, mem_op %d, size %d, ld_dest_reg %d\n",core_id, i, m_per_scalar_thread[i].memreqaddr[j], m_per_scalar_thread[i].space[j], m_per_scalar_thread[i].mem_op[j], m_per_scalar_thread[i].size[j], m_per_scalar_thread[i].ld_dest_reg[j]);
          fflush(stdout);
        }
      }
    }
    void print_mem_ops_tid(unsigned tid, unsigned core_id = 0){
      assert(tid < m_per_scalar_thread.size());
      for (unsigned j = 0; j < m_per_scalar_thread[tid].count; j++){
        printf("DICE Sim: [MEM_ACCESS]: core %d, thread %u, addr 0x%04x, space %d, mem_op %d, size %d, ld_dest_reg %d\n",core_id, tid, m_per_scalar_thread[tid].memreqaddr[j], m_per_scalar_thread[tid].space[j], m_per_scalar_thread[tid].mem_op[j], m_per_scalar_thread[tid].size[j], m_per_scalar_thread[tid].ld_dest_reg[j]);
        fflush(stdout);
      }
      
    }
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
    void memory_coalescing_arch_reduce_and_send(bool is_write, const dice_transaction_info &info, new_addr_type addr, unsigned segment_size);
  protected:
    unsigned m_uid;
    unsigned m_block_size;
    class shader_core_config *m_config;
    simt_mask_t* m_block_active_mask;
    unsigned m_active_count;
    dice_metadata *m_metadata;
    dice_block_t *m_diceblock;
    bool m_per_scalar_thread_valid;
    struct per_thread_info {
      per_thread_info() {
        for (unsigned i = 0; i < MAX_ACCESSES_PER_BLOCK_PER_THREAD; i++){
          memreqaddr[i] = 0;
          space[i] = undefined_space;
          mem_op[i] = no_memory_op;
          size[i] = 0;
        }
        count = 0;
        enable = 0;
      }
      dram_callback_t callback;
      new_addr_type
          memreqaddr[MAX_ACCESSES_PER_BLOCK_PER_THREAD];  // effective address,
                                                         // upto 8 different
                                                         // requests (to support
                                                         // 32B access in 8 chunks
                                                         // of 4B each)
      memory_space_t space[MAX_ACCESSES_PER_BLOCK_PER_THREAD];
      _memory_op_t mem_op[MAX_ACCESSES_PER_BLOCK_PER_THREAD];
      unsigned size[MAX_ACCESSES_PER_BLOCK_PER_THREAD];
      unsigned ld_dest_reg[MAX_ACCESSES_PER_BLOCK_PER_THREAD];
      unsigned count;
      unsigned enable;
    };
    std::vector<per_thread_info> m_per_scalar_thread;
    std::vector<std::list<mem_access_t>> m_accessq; //ldst_port->access per port
};

#endif
