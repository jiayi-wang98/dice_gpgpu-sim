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

/// extern prototypes

extern std::map<unsigned, const char *> get_duplicate();

typedef void *yyscan_t;
extern int dicemeta_get_lineno(yyscan_t yyscanner);
extern int dicemeta_lex_init(yyscan_t *scanner);
extern void dicemeta_set_in(FILE *_in_str, yyscan_t yyscanner);
extern int dicemeta_parse(yyscan_t scanner, dice_metadata *dicemeta);
extern int dicemeta_lex_destroy(yyscan_t scanner);

static bool g_save_embedded_ptx;
static int g_occupancy_sm_number;

#define DICE_PARSE_DPRINTF(...)                                                 \
  if (g_debug_dicemeta_generation) {                                            \
    printf(" %s:%u => ", g_dice_metadata_filename, dicemeta_get_lineno(scanner)); \
    printf("   (%s:%u) ", __FILE__, __LINE__);                                  \
    printf(__VA_ARGS__);                                                        \
    printf("\n");                                                               \
    fflush(stdout);                                                             \
  } //has bugs here causing segfault when trying to print out the debug message

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


void dice_metadata_parser::dump(){
  printf("DICE Metadata:\n"); fflush(stdout);
  printf("Total Number of DICE Blocks: %d\n", g_dice_metadata_list.size()); fflush(stdout);
  //loop over std::list<dice_metadata*> g_dice_metadata_list;
  assert(g_dice_metadata_list.size()!=0);
  for (std::list<dice_metadata*>::iterator it = g_dice_metadata_list.begin(); it != g_dice_metadata_list.end(); ++it){
    printf("DICE Metadata: DUMPING !!!\n"); fflush(stdout);
    assert((*it)!=NULL);
    (*it)->dump();
  }
}

void dice_metadata::dump(){
  printf("DICE Metadata:\n");
  printf("DBB ID: %d\n", dbb_id);
  printf("Unrolling Factor: %d\n", unrolling_factor);
  printf("Unrolling Strategy: %d\n", unrolling_strategy);
  printf("Latency: %d\n", latency);fflush(stdout);
  printf("In Registers:\n"); fflush(stdout);
  if(in_regs.size()>0){
    for (std::list<operand_info>::iterator it = in_regs.begin(); it != in_regs.end(); ++it){
      printf((it->name()).c_str());
      printf(" ");
    }
  }
  printf("Out Registers:\n");
  if(out_regs.size()>0){
    for (std::list<operand_info>::iterator it = out_regs.begin(); it != out_regs.end(); ++it){
      printf((it->name()).c_str());
      printf(" ");
    }
  }
  printf("Load Destination Registers:\n");
  if(load_destination_regs.size()>0){
    for (std::list<operand_info>::iterator it = load_destination_regs.begin(); it != load_destination_regs.end(); ++it){
      printf((it->name()).c_str());
      printf(" ");
    }
  }
  printf("Number of Store: %d\n", num_store);
  printf("Branch: %d\n", branch);
  printf("Uni Branch: %d\n", uni_bra);
  printf("Branch Prediction:"); fflush(stdout);
  if (branch_pred!=NULL) printf((branch_pred->name()).c_str());
  printf("\n");
  printf("Branch Target DBB: %d\n", branch_target_dbb);
  printf("Reconvergence DBB: %d\n", reconvergence_dbb);
  printf("Is Exit: %d\n", is_exit);
  printf("Is Ret: %d\n", is_ret);
  printf("Is Entry: %d\n", is_entry);
  printf("In Registers:\n"); fflush(stdout);
}

void dice_metadata_parser::add_scalar_operand(const char *identifier) {
  DICE_PARSE_DPRINTF("add_scalar_operand");
  const symbol *s = gpgpu_ctx->pptx_parser->g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    std::string msg = std::string("operand \"") + identifier + "\" has no declaration.";
    printf("DICE Metadata Parser: Error %s\n", msg.c_str());
    abort();
  }
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
  printf("DICE Metadata Parser: Commit DBB %d\n", g_current_dbb->dbb_id); fflush(stdout); 
  g_dice_metadata_list.push_back(g_current_dbb);
  //g_current_dbb = new dice_metadata(gpgpu_ctx);
}

void dice_metadata_parser::create_new_dbb(int dbb_id){
  printf("DICE Metadata Parser: Create New DBB %d\n", dbb_id); fflush(stdout); 
  g_current_dbb = new dice_metadata(gpgpu_ctx);
  g_current_dbb->dbb_id = dbb_id;
}

int dice_metadata_parser::metadata_list_size(){
  int num=0;
  for (std::list<dice_metadata*>::iterator it = g_dice_metadata_list.begin(); it != g_dice_metadata_list.end(); ++it){
    num++;
    if (num>1000){
      printf("DICE Metadata Parser: ERROR size > %d\n", num); fflush(stdout); 
      break;
    }
  }
  return num;
}