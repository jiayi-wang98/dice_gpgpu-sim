//GP-DICE NEW
#ifndef pdfg_ir_INCLUDED
#define pdfg_ir_INCLUDED

#include "../abstract_hardware_model.h"

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

//#include "ptx.tab.h"
#include "ptx_sim.h"

#include "memory.h"

class gpgpu_context;

struct pdfg_block_t {
  pdfg_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                bool entry, bool ex) {
    bb_id = ID;
    ptx_begin = begin;
    ptx_end = end;
    is_entry = entry;
    is_exit = ex;
    immediatepostdominator_id = -1;
    immediatedominator_id = -1;
  }

  ptx_instruction *ptx_begin;
  ptx_instruction *ptx_end;
  std::set<int>
      predecessor_ids;  // indices of other basic blocks in m_basic_blocks array
  std::set<int> successor_ids;
  std::set<int> postdominator_ids;
  std::set<int> dominator_ids;
  std::set<int> Tmp_ids;
  int immediatepostdominator_id;
  int immediatedominator_id;
  bool is_entry;
  bool is_exit;
  unsigned bb_id;

  // if this dfg block dom B
  bool dom(const pdfg_block_t *B) {
    return (B->dominator_ids.find(this->bb_id) != B->dominator_ids.end());
  }

  // if this dfg block pdom B
  bool pdom(const pdfg_block_t *B) {
    return (B->postdominator_ids.find(this->bb_id) !=
            B->postdominator_ids.end());
  }
};