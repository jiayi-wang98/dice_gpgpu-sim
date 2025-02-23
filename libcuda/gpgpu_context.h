#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/cuda_device_runtime.h"
#include "../src/cuda-sim/ptx-stats.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "cuda_api_object.h"
#include "../src/cuda-sim/dice_metadata.h"

class gpgpu_context {
 public:
  gpgpu_context() {
    g_global_allfiles_symbol_table = NULL;
    sm_next_access_uid = 0;
    warp_inst_sm_next_uid = 0;
    operand_info_sm_next_uid = 1;
    kernel_info_m_next_uid = 1;
    g_num_ptx_inst_uid = 0;
    g_ptx_cta_info_uid = 1;
    symbol_sm_next_uid = 1;
    function_info_sm_next_uid = 1;
    debug_tensorcore = 0;
    metadata_start_pc = 0;
    api = new cuda_runtime_api(this);
    ptxinfo = new ptxinfo_data(this);
    dicemeta_parser = new dice_metadata_parser(this);
    ptx_parser = new ptx_recognizer(this);
    the_gpgpusim = new GPGPUsim_ctx(this);
    func_sim = new cuda_sim(this);
    device_runtime = new cuda_device_runtime(this);
    stats = new ptx_stats(this);
  }
  // global list
  symbol_table *g_global_allfiles_symbol_table;
  const char *g_filename;
  unsigned sm_next_access_uid;
  unsigned warp_inst_sm_next_uid;
  unsigned operand_info_sm_next_uid;  // uid for operand_info
  unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
  unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
  unsigned long long g_ptx_cta_info_uid;
  unsigned symbol_sm_next_uid;  // uid for symbol
  unsigned function_info_sm_next_uid;
  std::vector<ptx_instruction *> s_g_pc_to_insn;  // a direct mapping from PC to instruction
  bool debug_tensorcore;

  // objects pointers for each file
  cuda_runtime_api *api;
  ptxinfo_data *ptxinfo;
  ptx_recognizer *ptx_parser;

  //DICE-support
  dice_metadata_parser *dicemeta_parser;
  std::vector<dice_metadata* > s_g_pc_to_meta;  // a direct mapping from PC to dice_metadata
  unsigned metadata_start_pc;

  GPGPUsim_ctx *the_gpgpusim;
  cuda_sim *func_sim;
  cuda_device_runtime *device_runtime;
  ptx_stats *stats;
  // member function list
  dice_metadata* get_meta_from_pc(unsigned pc){return s_g_pc_to_meta[pc-metadata_start_pc];}
  void synchronize();
  void exit_simulation();
  void print_simulation_time();
  int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid);
  void cuobjdumpParseBinary(unsigned int handle);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                         unsigned source_num);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename(
      const char *filename);

  void gpgpu_ptx_info_load_from_filename(const char *filename,
                                         unsigned sm_version);
  void gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                      unsigned source_num,
                                      unsigned sm_version = 20,
                                      int no_of_ptx = 0);
  //DICE-support
  int g_dice_enabled;
  class symbol_table *dice_pptx_load_from_filename(const char *filename);
  void dice_metadata_load_from_filename(const char *filename);

  void print_ptx_file(const char *p, unsigned source_num, const char *filename);
  class symbol_table *init_parser(const char *, ptx_recognizer *parser);
  class gpgpu_sim *gpgpu_ptx_sim_init_perf();
  void start_sim_thread(int api);
  struct _cuda_device_id *GPGPUSim_Init();
  void ptx_reg_options(option_parser_t opp);
  const ptx_instruction *pc_to_instruction(unsigned pc);
  const warp_inst_t *ptx_fetch_inst(address_type pc);
  //DICE support
  dice_metadata *dice_fetch_metadata(addr_t pc);
  dice_metadata *pc_to_metadata(unsigned pc);
  
  unsigned translate_pc_to_ptxlineno(unsigned pc);
};
gpgpu_context *GPGPU_Context();

#endif /* __gpgpu_context_h__ */
