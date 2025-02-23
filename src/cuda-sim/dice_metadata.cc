#include "dice_metadata.h"
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include "../../libcuda/gpgpu_context.h"
#include "cuda-sim.h"
#include "ptx_ir.h"
#include "ptx_parser.h"
#include "dicemeta.tab.h"

typedef void *yyscan_t;
extern int dicemeta_get_lineno(yyscan_t yyscanner);
extern int dicemeta_lex_init(yyscan_t *scanner);
extern void dicemeta_set_in(FILE *_in_str, yyscan_t yyscanner);
extern int dicemeta_parse(yyscan_t scanner, dice_metadata *dicemeta);
extern int dicemeta_lex_destroy(yyscan_t scanner);

#define DICE_PARSE_DPRINTF(...)                                                 \
  if (g_debug_dicemeta_generation) {                                            \
    printf(" %s:%u => ", g_dice_metadata_filename, dicemeta_get_lineno(scanner)); \
    printf("   (%s:%u) ", __FILE__, __LINE__);                                  \
    printf(__VA_ARGS__);                                                        \
    printf("\n");                                                               \
    fflush(stdout);                                                             \
  } //has bugs here causing segfault when trying to print out the debug message

void function_info::link_block_in_dicemeta() {
  printf("Linking DICE blocks to metadata of function\'%s\':\n", m_name.c_str());
  //iterate over m_dice_metadata
  std::vector<dice_metadata *>::iterator dice_itr;
  for (dice_itr = m_dice_metadata.begin(); dice_itr != m_dice_metadata.end();
       dice_itr++) {
    //get their bitstream_label
    std::string bitstream_label = std::string((*dice_itr)->bitstream_label);
    printf("DICE METADATA Linking: %s\n",bitstream_label.c_str());
    //find dice_basic_block with the same label
    std::vector<dice_block_t *>::iterator dice_block_itr;
    bool found = false;
    for (dice_block_itr = m_dice_blocks.begin();
         dice_block_itr != m_dice_blocks.end(); dice_block_itr++) {
      if ((*dice_block_itr)->label == bitstream_label) {
        //link the dice_basic_block to the dice_metadata
        (*dice_itr)->dice_block = *dice_block_itr;
        found = true;
        break;
      }
    }
    if (!found) {
      printf("DICE METADATA Error: Could not find dice code block with label %s\n",
        ((*dice_itr)->bitstream_label).c_str());
      //print available dice block labels
      printf("Available dice block labels: ");
      for (dice_block_itr = m_dice_blocks.begin();
             dice_block_itr != m_dice_blocks.end(); dice_block_itr++) {
        printf(" %s", (*dice_block_itr)->label.c_str());
      }
      printf("\n");
      fflush(stdout);
      assert(0);
    }
  }
}

void function_info::dump_dice_metadata() {
  printf("Dumping DICE metadata for function \'%s\':\n", m_name.c_str());
  std::vector<dice_metadata *>::iterator dice_itr;
  for (dice_itr = m_dice_metadata.begin(); dice_itr != m_dice_metadata.end();
       dice_itr++) {
    (*dice_itr)->dump();
  }
}

void gpgpu_context::dice_metadata_load_from_filename(const char *filename) {
  std::string metadata_filename(filename);
  char buff[1024], extra_flags[1024];
  extra_flags[0] = 0;
  g_filename = strdup(filename); 
  
  //TODO: replace this with customized metadata generator from Darren
  //snprintf(
  //    buff, 1024,
  //    "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  /dev/null 2> %s",
  //    extra_flags, filename, metadata_filename.c_str());
  //int result = system(buff);
  //if (result != 0) {
  //  printf("GPGPU-Sim DICE_METADATA: ERROR ** while loading Metadata %d\n", result);
  //  printf("               Ensure .meta is in your path.\n");
  //  exit(1);
  //}

  FILE *dicemeta_in;
  dicemeta_parser->g_dice_metadata_filename = strdup(metadata_filename.c_str());
  dicemeta_in = fopen(dicemeta_parser->g_dice_metadata_filename, "r");
  dicemeta_lex_init(&(dicemeta_parser->scanner));
  dicemeta_set_in(dicemeta_in, dicemeta_parser->scanner);
  dicemeta_parse(dicemeta_parser->scanner, dicemeta_parser);
  dicemeta_lex_destroy(dicemeta_parser->scanner);
  fclose(dicemeta_in);
}

void dice_metadata::dump(){
  printf("DICE Metadata:\n");
  printf("Metadata ID: %d\n", meta_id);
  printf("PC: %p\n", m_PC);
  printf("Metadata Mem Index: %d\n", m_dicemeta_mem_index);
  printf("Size: %d\n", m_size);
  printf("BITSTREAM_ADDR: %s : %p\n", bitstream_label.c_str(),dice_block->get_start_pc());
  printf("BITSTREAM_LENGTH: %d\n", bitstream_length);
  printf("Unrolling Factor: %d\n", unrolling_factor);
  printf("Unrolling Strategy: %d\n", unrolling_strategy);
  printf("Latency: %d\n", latency);fflush(stdout);
  printf("In Registers:"); fflush(stdout);
  if(in_regs.size()>0){
    for (std::list<operand_info>::iterator it = in_regs.begin(); it != in_regs.end(); ++it){
      std::cout<<(it->name()).c_str();
      printf(" ");
    }
    printf("\n");
  } else {
    printf("None\n");
  }
  printf("Out Registers:");
  if(out_regs.size()>0){
    for (std::list<operand_info>::iterator it = out_regs.begin(); it != out_regs.end(); ++it){
      std::cout<<(it->name()).c_str();
      printf(" ");
    }
    printf("\n");
  } else {
    printf("None\n");
  }
  printf("Load Destination Registers:");
  if(load_destination_regs.size()>0){
    for (std::list<operand_info>::iterator it = load_destination_regs.begin(); it != load_destination_regs.end(); ++it){
      std::cout<<(it->name()).c_str();
      printf(" ");
    }
    printf("\n");
  } else {
    printf("None\n");
  }
  printf("Number of Store: %d\n", num_store);
  printf("Branch: %d\n", branch);
  if (branch) {
    printf("Uni Branch: %d\n", uni_bra);
    printf("Branch Prediction:"); 
    if (branch_pred!=NULL) std::cout<<(branch_pred->name());
    printf("\n");
    printf("Branch Target Metadata ID: %d, PC = %p\n", branch_target_meta_id, branch_target_meta_pc);
    printf("Reconvergence Metadata ID: %d, PC = %p\n", reconvergence_meta_id, reconvergence_meta_pc);
  }
  printf("Is Exit: %d\n", is_exit);
  printf("Is Ret: %d\n", is_ret);
  printf("Is Entry: %d\n", is_entry);
  printf("\n\n");
  fflush(stdout);
}

void dice_metadata_parser::add_operand(const char *identifier) {
  //DICE_PARSE_DPRINTF("add_operand");
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: add operand %s\n", identifier); fflush(stdout);
  assert(gpgpu_ctx != NULL);
  fflush(stdout);
  //function_info *func_info = gpgpu_ctx->ptx_parser->g_global_symbol_table->lookup_function(g_current_function_name);
  symbol_table* symtab = g_current_function_info->get_symtab();
  const symbol *s = symtab->lookup(identifier);
  if (s == NULL) {
    std::string msg = std::string("operand \"") + identifier + "\" has no declaration.";
    printf("DICE Metadata Parser: Error %s\n", msg.c_str()); fflush(stdout);
    abort();
  }
  if(g_debug_dicemeta_generation) s->print_info(stdout);
  g_operands.push_back(operand_info(s, gpgpu_ctx));
}

void dice_metadata_parser::read_parser_environment_variables() {
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    int debug_execution = 0;
    sscanf(dbg_level, "%d", &debug_execution);
    if (debug_execution >= 3) g_debug_dicemeta_generation = true;
    printf("DICE Metadata Parser: Debugging level = %d, %d\n", debug_execution, g_debug_dicemeta_generation); fflush(stdout);
  }
}

void dice_metadata_parser::commit_dbb(){
  //printf("g_debug_dicemeta_generation = %d\n", g_debug_dicemeta_generation); fflush(stdout); 
  //DICE_PARSE_DPRINTF("commit_dbb");
  // check if the current dbb is empty
  if (g_current_dbb == NULL){
    printf("DICE Metadata Parser: Empty DBB, abort\n"); fflush(stdout);
    abort();
  }
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: Commit Meta %d\n", g_current_dbb->meta_id); 
  g_current_function_info->add_dice_metadata(g_current_dbb);
  //g_current_dbb = new dice_metadata(gpgpu_ctx);
}

void dice_metadata_parser::create_new_dbb(int meta_id){
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: Create New Meta %d\n", meta_id); 
  g_current_dbb = new dice_metadata(gpgpu_ctx);
  g_current_dbb->meta_id = meta_id;
}

void dice_metadata_parser::set_in_regs(){
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: Set In Registers\n"); 
  g_current_dbb->in_regs = g_operands;
  g_operands = std::list<operand_info>();
}

void dice_metadata_parser::set_out_regs(){
  if(g_debug_dicemeta_generation)  printf("DICE Metadata Parser: Set out Registers\n");  
  g_current_dbb->out_regs = g_operands;
  g_operands = std::list<operand_info>();
}

void dice_metadata_parser::set_ld_dest_regs(){
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: Set LD destination Registers\n");  
  g_current_dbb->load_destination_regs = g_operands;
  g_operands = std::list<operand_info>();
}

void dice_metadata_parser::set_branch_pred(){
  printf("DICE Metadata Parser: Set Branch Predication\n"); fflush(stdout); 
  int num_operands = 0;
  for (std::list<operand_info>::iterator it = g_operands.begin(); it != g_operands.end(); ++it){
    num_operands++;
    if(num_operands>1){
      printf("DICE Metadata Parser: ERROR: Branch Predicate Regs more than 1\n"); fflush(stdout); 
      abort();
    }
  }
  if (num_operands == 0){
    printf("DICE Metadata Parser: ERROR: No Branch Predicate Regs\n"); fflush(stdout); 
    abort();
  }
  g_current_dbb->branch_pred = new operand_info(std::move(g_operands.front()));
  g_operands.clear();
}

void dice_metadata_parser::add_builtin_operand(int builtin, int dim_modifier) {
  if(g_debug_dicemeta_generation) printf("DICE Metadata Parser: Add builtin operand\n");
  g_operands.push_back(operand_info(builtin, dim_modifier, gpgpu_ctx));
}

void dice_metadata_parser::set_function_name(const char *name){
  if(g_debug_dicemeta_generation)  printf("DICE Metadata Parser: Set Function Name\n"); 
  g_current_function_name = name;
  g_current_function_info = gpgpu_ctx->ptx_parser->g_global_symbol_table->lookup_function(g_current_function_name);
}

void dice_metadata_parser::commit_function(){
  g_current_function_info->link_block_in_dicemeta();
  dice_metadata_assemble(g_current_function_name, g_current_function_info);
  if (g_debug_dicemeta_generation) {
    g_current_function_info->dump_dice_metadata();
    fflush(stdout); 
  }
  g_current_function_info = NULL;
  g_current_dbb = NULL;
  g_current_function_name = "";
  printf("DICE Metadata Parser: Commited Function\n"); fflush(stdout); 
}

void dice_metadata_assemble(std::string kname, void *kinfo){
  printf("DICE Metadata Assemble\n"); fflush(stdout); 
  function_info *func_info = (function_info *)kinfo;
  if ((function_info *)kinfo == NULL) {
    printf("GPGPU-Sim PTX: Warning - missing function definition \'%s\'\n",
           kname.c_str());
    return;
  }
  if (func_info->is_extern()) {
    printf(
        "GPGPU-Sim PTX: skipping assembly for extern declared function "
        "\'%s\'\n",
        func_info->get_name().c_str());
    return;
  }
  func_info->metadata_assemble();
}

unsigned dice_block_t::get_start_pc(){
  return ptx_begin->get_PC(); 
}

unsigned dice_block_t::get_end_pc(){
  return ptx_end->get_PC(); 
}


//DICE-support
dice_metadata *gpgpu_context::dice_fetch_metadata(addr_t pc) {
  printf("DICE Metadata Fetch: %p\n", pc); fflush(stdout);
  return pc_to_metadata(pc);
}

//DICE-support
dice_metadata *gpgpu_context::pc_to_metadata(unsigned pc) {
  if ((pc-metadata_start_pc) < 0) {
    printf("DICE Metadata Fetch: ERROR: PC to Metadata\n"); fflush(stdout);
    abort();
  }
  if ((pc-metadata_start_pc) < s_g_pc_to_meta.size())
    return s_g_pc_to_meta[pc-metadata_start_pc];
  else{
    printf("DICE Metadata Fetch: Warning: PC to Metadata pointing to NULL address!\n"); fflush(stdout);
    return NULL;
  }
}


dice_metadata* DICEfunctionalCoreSim::getExecuteMetadata(){
  unsigned pc, rpc;
  m_simt_stack[0]->get_pdom_stack_top_info(&pc, &rpc);
  printf("DICE pdom_stack_top_info: %p, %p\n", pc, rpc); fflush(stdout);
  dice_metadata* metadata = m_gpu->gpgpu_ctx->dice_fetch_metadata(pc);
  assert(metadata != NULL);
  metadata->set_active(m_simt_stack[0]->get_active_mask());
  return metadata;
}

void dice_metadata::set_active(const active_mask_t &active) {
  assert(active.size() > 0);
  //active.dump();
  simt_mask_t *active_mask = new simt_mask_t(active);
  m_block_active_mask = active_mask;
  //Atomic TODO
  //if (m_isatomic) {
  //  for (unsigned i = 0; i < m_config->get_warp_size(); i++) {
  //    if (!m_warp_active_mask.test(i)) {
  //      m_per_scalar_thread[i].callback.function = NULL;
  //      m_per_scalar_thread[i].callback.instruction = NULL;
  //      m_per_scalar_thread[i].callback.thread = NULL;
  //    }
  //  }
  //}
}

void dice_metadata::set_not_active(unsigned tid) {
  m_block_active_mask->reset(tid);
}