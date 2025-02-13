#ifndef DICE_HARDWARE_MODEL_INCLUDED
#define DICE_HARDWARE_MODEL_INCLUDED

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

#ifdef __cplusplus

#include "abstract_hardware_model.h"
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

class dice_metadata;
class ptx_instruction;
class ptx_thread_info;

class dice_inst_block_t {
    dice_inst_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                      bool entry, bool ex) {
      dbb_id = ID;
      ptx_begin = begin;
      ptx_end = end;
      is_entry = entry;
      is_exit = ex;
      immediatepostdominator_id = -1;
    }
    ptx_instruction *ptx_begin;
    ptx_instruction *ptx_end;
    int immediatepostdominator_id;
    bool is_entry;
    bool is_exit;
    unsigned dbb_id;
    
    dice_metadata *m_dice_meta;

    void set_dice_meta(dice_metadata *dm) { m_dice_meta = dm; }
};

#endif