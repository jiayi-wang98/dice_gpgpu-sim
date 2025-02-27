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
      unrolling_factor = 0;
      unrolling_strategy = 0;
      latency = 0;
      bitstream_label = "";
      bitstream_length = 0;
      num_store = 0;
      branch = false;
      uni_bra = false;
      branch_pred = NULL;
      branch_target_meta_id = 0;
      reconvergence_meta_id = 0;
      is_exit = false;
      is_ret = false;
      is_entry = false;
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
    int branch_target_meta_id;
    int reconvergence_meta_id;
    unsigned branch_target_meta_pc;
    unsigned reconvergence_meta_pc;

    bool is_exit;
    bool is_ret;
    bool is_entry;
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

//for performance simulation
class dice_cfg_block_t{
  public:
    dice_cfg_block_t(dice_metadata *metadata);
    dice_cfg_block_t(unsigned uid, unsigned block_size, dice_metadata *metadata);
    ~dice_cfg_block_t(){
      if(m_block_active_mask) delete m_block_active_mask;
    }
    address_type metadata_pc;  // program counter address of metadata
    unsigned metadata_size;   // size of metadata in bytes
    address_type reconvergence_pc;  // program counter address of reconvergence
    op_type op;  // operation type
    addr_t branch_target_meta_pc;  // program counter address of branch target
    memory_space_t space;

    bool active(unsigned tid) const { return m_block_active_mask->test(tid); }
    unsigned active_count() const { return m_block_active_mask->count(); }
    void set_active(const active_mask_t &active);
    void set_not_active(unsigned tid);
    void set_addr(unsigned n, new_addr_type addr) {
      if (!m_per_scalar_thread_valid) {
        m_per_scalar_thread.resize(1536);//TODO: use config
        m_per_scalar_thread_valid = true;
      }
      assert(n < m_per_scalar_thread.size() && "Index n out of bounds");
      m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
      if (!m_per_scalar_thread_valid) {
        m_per_scalar_thread.resize(1536); //TODO: use config
        m_per_scalar_thread_valid = true;
      }
      assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
      for (unsigned i = 0; i < num_addrs; i++)
        m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }
    dice_metadata *get_metadata() { return m_metadata; }

  protected:
    unsigned m_uid;
    simt_mask_t* m_block_active_mask;
    dice_metadata *m_metadata;
    dice_block_t *m_diceblock;
    bool m_per_scalar_thread_valid;
    struct per_thread_info {
      per_thread_info() {
        for (unsigned i = 0; i < MAX_ACCESSES_PER_BLOCK_PER_THREAD; i++)
          memreqaddr[i] = 0;
      }
      dram_callback_t callback;
      new_addr_type
          memreqaddr[MAX_ACCESSES_PER_BLOCK_PER_THREAD];  // effective address,
                                                         // upto 8 different
                                                         // requests (to support
                                                         // 32B access in 8 chunks
                                                         // of 4B each)
    };
    std::vector<per_thread_info> m_per_scalar_thread;
};

#endif
