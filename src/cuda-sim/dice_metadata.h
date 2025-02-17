#ifndef DICE_METADATA_H_INCLUDED
#define DICE_METADATA_H_INCLUDED
#include <string>
#include "ptx_ir.h"

#define DICE_METADATA_LINEBUF_SIZE 1024
class gpgpu_context;
typedef void* yyscan_t;


class operand_info;
class ptx_instruction;
struct dice_block_t;


class dice_metadata {
  public:
    dice_metadata(class gpgpu_context* ctx) {
      gpgpu_ctx = ctx;
      dbb_id = 0;
      unrolling_factor = 0;
      unrolling_strategy = 0;
      latency = 0;
      num_store = 0;
      branch = false;
      uni_bra = false;
      branch_pred = NULL;
      branch_target_dbb = 0;
      reconvergence_dbb = 0;
      is_exit = false;
      is_ret = false;
      is_entry = false;
    }

    class gpgpu_context* gpgpu_ctx;
    int dbb_id;
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
    int branch_target_dbb;
    int reconvergence_dbb;

    bool is_exit;
    bool is_ret;
    bool is_entry;
    dice_block_t *dice_block;

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
  void create_new_dbb(int dbb_id);
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
    g_current_dbb->branch_target_dbb = dbb;
  }
  void set_reconvergence_dbb(int dbb){
    g_current_dbb->reconvergence_dbb = dbb;
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
  bool is_entry;
  bool is_exit;
  unsigned dbb_id;
  std::string label;
};
#endif
