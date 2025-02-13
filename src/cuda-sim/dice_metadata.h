#ifndef DICE_METADATA_H_INCLUDED
#define DICE_METADATA_H_INCLUDED
#include <string>
#include "ptx_ir.h"

#define DICE_METADATA_LINEBUF_SIZE 1024
class gpgpu_context;
typedef void* yyscan_t;


class operand_info;

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

    void dump();
};

class dice_metadata_parser {
 private:
  bool g_debug_dicemeta_generation;
  dice_metadata* g_current_dbb;
  std::list<dice_metadata*> g_dice_metadata_list;
  std::list<operand_info> g_operands;
 public:
  dice_metadata_parser(gpgpu_context* ctx)
      : g_dice_metadata_list(),  
        g_operands()
  {
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
  class gpgpu_context* gpgpu_ctx;


  void add_scalar_operand(const char *identifier);
  void dump();
  void read_parser_environment_variables();
  void create_new_dbb(int dbb_id);
  int metadata_list_size();
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
};

#endif
