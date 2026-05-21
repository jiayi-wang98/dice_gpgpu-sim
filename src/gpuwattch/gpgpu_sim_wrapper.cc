// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington, Ahmed ElTantawy,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpu_sim_wrapper.h"
#include <sys/stat.h>
#include "xmlParser.h"
#define SP_BASE_POWER 0
#define SFU_BASE_POWER 0

static const char* pwr_cmp_label[] = {
    "IBP,", "ICP,",  "DCP,",   "TCP,",   "CCP,",        "SHRDP,",
    "RFP,", "SPP,",  "SFUP,",  "FPUP,",  "SCHEDP,",     "L2CP,",
    "MCP,", "NOCP,", "DRAMP,", "PIPEP,", "IDLE_COREP,", "CONST_DYNAMICP,",
    "BCP,",
    // AccelWattch-style ALU subdivisions (appended at tail so existing
    // GPUWattch trace columns stay backward-compatible).
    "DPUP,",     "INT_MUL24P,", "INT_MUL32P,", "INT_MULP,", "INT_DIVP,",
    "FP_MULP,",  "FP_DIVP,",    "FP_SQRTP,",   "FP_LGP,",   "FP_SINP,",
    "FP_EXP,",   "DP_MULP,",    "DP_DIVP,",    "TENSORP,",  "TEXP,",
    "STATICP"};

enum pwr_cmp_t {
  IBP = 0,
  ICP,
  DCP,
  TCP,
  CCP,
  SHRDP,
  RFP,
  SPP,           // INTP (base integer ALU)
  SFUP,          // SFUP (base SFU lump)
  FPUP,          // FPUP (base FP32)
  SCHEDP,
  L2CP,
  MCP,
  NOCP,
  DRAMP,
  PIPEP,
  IDLE_COREP,
  CONST_DYNAMICP,
  BCP,
  // ---- AccelWattch ALU subdivisions ----
  DPUP,
  INT_MUL24P,
  INT_MUL32P,
  INT_MULP,
  INT_DIVP,
  FP_MULP,
  FP_DIVP,
  FP_SQRTP,
  FP_LGP,
  FP_SINP,
  FP_EXP,
  DP_MULP,
  DP_DIVP,
  TENSORP,
  TEXP,
  STATICP,
  NUM_COMPONENTS_MODELLED
};

gpgpu_sim_wrapper::gpgpu_sim_wrapper(bool power_simulation_enabled,
                                     char* xmlfile) {
  kernel_sample_count = 0;
  total_sample_count = 0;

  kernel_tot_power = 0;

  num_pwr_cmps = NUM_COMPONENTS_MODELLED;
  num_perf_counters = NUM_PERFORMANCE_COUNTERS;

  // Initialize per-component counter/power vectors
  avg_max_min_counters<double> init;
  kernel_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, init);
  kernel_cmp_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, init);

  kernel_power = init;   // Per-kernel powers
  gpu_tot_power = init;  // Global powers

  sample_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, 0);

  sample_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, 0);
  initpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);
  effpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);

  const_dynamic_power = 0;
  proc_power = 0;

  g_power_filename = NULL;
  g_power_trace_filename = NULL;
  g_metric_trace_filename = NULL;
  g_steady_state_tracking_filename = NULL;
  xml_filename = xmlfile;
  g_power_simulation_enabled = power_simulation_enabled;
  g_power_trace_enabled = false;
  g_steady_power_levels_enabled = false;
  g_power_trace_zlevel = 0;
  g_power_per_cycle_dump = false;
  gpu_steady_power_deviation = 0;
  gpu_steady_min_period = 0;

  gpu_stat_sample_freq = 0;
  p = new ParseXML();
  if (g_power_simulation_enabled) {
    p->parse(xml_filename);
  }
  proc = new Processor(p);
  power_trace_file = NULL;
  metric_trace_file = NULL;
  steady_state_tacking_file = NULL;
  has_written_avg = false;
  init_inst_val = false;

  // DICE overlay defaults off; enable_dice_power_model() must be called
  // before mcpat_cycle if the user requested -gpgpu_dice_power_model 1.
  m_dice_power_enabled = false;
  m_dice_params = {};
}

// Load DICE per-access energies from a small standalone overlay XML.
// Schema: <component id="root"><component id="dice_power">
//             <param name="DICE_*" value="..."/> ... </component></component>
void gpgpu_sim_wrapper::enable_dice_power_model(const char* dice_xml_path) {
  if (dice_xml_path == NULL || dice_xml_path[0] == '\0') {
    fprintf(stderr,
            "[DICE-Power] -gpgpu_dice_power_model 1 set but no XML path; "
            "disabling DICE overlay.\n");
    m_dice_power_enabled = false;
    return;
  }
  XMLNode root = XMLNode::openFileHelper(dice_xml_path, "component");
  XMLNode node = root.getChildNode("component");
  const unsigned n = node.nChildNode("param");
  // Defaults (12 nm-scaled from the 4subcore workbook; see XML header).
  // ---- DICE-specific structure defaults (12 nm-scaled CACTI from the
  //      4subcore workbook component sheets) ----
  m_dice_params.bcache_read_e   = 0.019637;
  m_dice_params.bcache_write_e  = 0.019300;
  m_dice_params.simt_stack_rd_e = 0.006707;
  m_dice_params.simt_stack_wr_e = 0.005826;
  m_dice_params.scoreboard_rd_e = 0.000438;
  m_dice_params.scoreboard_wr_e = 0.000449;
  m_dice_params.dispatcher_th_e = 0.000097;
  m_dice_params.bct_pushpop_e   = 0.0001452;
  m_dice_params.bct_dec_e       = 0.0000726;
  m_dice_params.active_cta_e    = 0.0004840;
  m_dice_params.sched_eblock_e  = 0.0000968;
  // ---- AccelWattch scaling-coefficient defaults (copy from
  //      gpu-rodinia/.../cfg/accelwattch_ptx_sim.xml RTX2060S target) ----
  m_dice_params.s = {};  // zero-initialize all
  m_dice_params.s.TOT_INST       = 2.0;
  m_dice_params.s.FP_INT         = 4.57;
  m_dice_params.s.IC_H           = 11.44089762;
  m_dice_params.s.IC_M           = 21.76302498;
  m_dice_params.s.DC_RH          = 7.737353491;
  m_dice_params.s.DC_RM          = 8.618027871;
  m_dice_params.s.DC_WH          = 0.53469516;
  m_dice_params.s.DC_WM          = 13.9055689;
  m_dice_params.s.CC_H           = 0.65916315;
  m_dice_params.s.CC_M           = 0.73418985;
  m_dice_params.s.TC_H           = 0.01021;
  m_dice_params.s.TC_M           = 0.02466;
  m_dice_params.s.SHRD_ACC       = 1.313660815;
  m_dice_params.s.REG_RD         = 0.053279375;
  m_dice_params.s.REG_WR         = 0.079919063;
  m_dice_params.s.INT_ACC        = 3.429666768;
  m_dice_params.s.FP_ACC         = 0.711591276;
  m_dice_params.s.DP_ACC         = 0.742812382;
  m_dice_params.s.INT_MUL24_ACC  = 0.290770573;
  m_dice_params.s.INT_MUL32_ACC  = 0.252598514;
  m_dice_params.s.INT_MUL_ACC    = 0.148636575;
  m_dice_params.s.INT_DIV_ACC    = 5.121706665;
  m_dice_params.s.FP_MUL_ACC     = 0.212559571;
  m_dice_params.s.FP_DIV_ACC     = 4.599926258;
  m_dice_params.s.FP_SQRT_ACC    = 1.241271438;
  m_dice_params.s.FP_LG_ACC      = 0.59034036;
  m_dice_params.s.FP_SIN_ACC     = 0.212555149;
  m_dice_params.s.FP_EXP_ACC     = 0.702043615;
  m_dice_params.s.DP_MUL_ACC     = 0.282564496;
  m_dice_params.s.TENSOR_ACC     = 2.485;
  m_dice_params.s.TEX_ACC        = 0.212559047;
  m_dice_params.s.MEM_RD         = 0.02772;
  m_dice_params.s.MEM_WR         = 0.0336;
  m_dice_params.s.MEM_PRE        = 0.00924;
  m_dice_params.s.L2_RH          = 1.046834662;
  m_dice_params.s.L2_RM          = 2.670605032;
  m_dice_params.s.L2_WH          = 3.269555394;
  m_dice_params.s.L2_WM          = 2.18020968;
  m_dice_params.s.NOC_A          = 83.18977901;
  m_dice_params.s.PIPE_A         = 0.0257;
  // Accumulator-PE op energy: one PE-cycle pipeline add when atom.shared
  // fires. Defaults to the PIPE_A scaling coefficient.
  m_dice_params.dice_acc_op_e    = 0.0257;
  m_dice_params.s.IDLE_CORE_N    = 1.0;
  m_dice_params.s.constant_power = 32.32522272;
  // ---- DICEwattch per-pipe energy constants (one per pipeline,
  //      per GPU technology). Defaults below = back-derived from a
  //      RTX2060S AccelWattch nn_cuda run; tune per target node. ----
  m_dice_params.e = {};
  m_dice_params.e.E_int_pipe  = 0.000462;     // base INT, naked
  m_dice_params.e.E_fpu_pipe  = 0.018854;     // base FP,  naked
  m_dice_params.e.E_dpu_pipe  = 0.018854;     // base DP,  naked
  m_dice_params.e.E_sfu_pipe  = 0.038674;     // SFU pipe, naked
  m_dice_params.e.E_icache    = 0.018286;     // per-scaled-IC access
  m_dice_params.e.E_ccache    = 0.033562;     // per-scaled-CC access
  m_dice_params.e.E_tcache    = 0.0;
  m_dice_params.e.E_dcache    = 0.029329;     // per-scaled-DC access
  m_dice_params.e.E_shmem     = 0.030448;     // per-scaled-SHRD access
  m_dice_params.e.E_l2        = 0.034778;     // per-scaled-L2 access
  m_dice_params.e.E_dram      = 70.346;       // per-scaled-DRAM access
  m_dice_params.e.E_mcp       = 8.388;        // per-scaled-MEM access (MC datapath)
  m_dice_params.e.E_noc       = 0.002683;     // per-scaled-NOC flit
  m_dice_params.e.E_rf_read   = 0.056525;     // per-scaled-REG access
  m_dice_params.e.E_rf_write  = 0.056525;
  m_dice_params.e.E_ibp       = 0.000588;     // per-scaled-TOT_INST
  // ---- Fixed-power slots ----
  m_dice_params.idle_core_p_mW  = 0.012138;
  m_dice_params.const_p_mW      = 32.325;
  m_dice_params.static_p_mW     = 37.240;
  m_dice_params.clock_freq_ghz  = 1.47;
  for (unsigned i = 0; i < n; i++) {
    XMLNode p = node.getChildNode("param", i);
    const char* k = p.getAttribute("name");
    const double v = atof(p.getAttribute("value"));
    // ---- DICE-specific structure energies ----
    if      (strcmp(k, "DICE_BCACHE_READ_E")    == 0) m_dice_params.bcache_read_e   = v;
    else if (strcmp(k, "DICE_BCACHE_WRITE_E")   == 0) m_dice_params.bcache_write_e  = v;
    else if (strcmp(k, "DICE_SIMT_STACK_RD_E")  == 0) m_dice_params.simt_stack_rd_e = v;
    else if (strcmp(k, "DICE_SIMT_STACK_WR_E")  == 0) m_dice_params.simt_stack_wr_e = v;
    else if (strcmp(k, "DICE_SCOREBOARD_RD_E")  == 0) m_dice_params.scoreboard_rd_e = v;
    else if (strcmp(k, "DICE_SCOREBOARD_WR_E")  == 0) m_dice_params.scoreboard_wr_e = v;
    else if (strcmp(k, "DICE_DISPATCHER_TH_E")  == 0) m_dice_params.dispatcher_th_e = v;
    else if (strcmp(k, "DICE_BCT_PUSHPOP_E")    == 0) m_dice_params.bct_pushpop_e   = v;
    else if (strcmp(k, "DICE_BCT_DEC_E")        == 0) m_dice_params.bct_dec_e       = v;
    else if (strcmp(k, "DICE_ACTIVE_CTA_E")     == 0) m_dice_params.active_cta_e    = v;
    else if (strcmp(k, "DICE_SCHED_EBLOCK_E")   == 0) m_dice_params.sched_eblock_e  = v;
    else if (strcmp(k, "DICE_ACC_OP_E")         == 0) m_dice_params.dice_acc_op_e   = v;
    // ---- AccelWattch-format scaling coefficients (drop-in copy from any
    //      AccelWattch GPU XML; names must match accelwattch_ptx_sim.xml) ----
    else if (strcmp(k, "TOT_INST")       == 0) m_dice_params.s.TOT_INST       = v;
    else if (strcmp(k, "FP_INT")         == 0) m_dice_params.s.FP_INT         = v;
    else if (strcmp(k, "IC_H")           == 0) m_dice_params.s.IC_H           = v;
    else if (strcmp(k, "IC_M")           == 0) m_dice_params.s.IC_M           = v;
    else if (strcmp(k, "DC_RH")          == 0) m_dice_params.s.DC_RH          = v;
    else if (strcmp(k, "DC_RM")          == 0) m_dice_params.s.DC_RM          = v;
    else if (strcmp(k, "DC_WH")          == 0) m_dice_params.s.DC_WH          = v;
    else if (strcmp(k, "DC_WM")          == 0) m_dice_params.s.DC_WM          = v;
    else if (strcmp(k, "TC_H")           == 0) m_dice_params.s.TC_H           = v;
    else if (strcmp(k, "TC_M")           == 0) m_dice_params.s.TC_M           = v;
    else if (strcmp(k, "CC_H")           == 0) m_dice_params.s.CC_H           = v;
    else if (strcmp(k, "CC_M")           == 0) m_dice_params.s.CC_M           = v;
    else if (strcmp(k, "SHRD_ACC")       == 0) m_dice_params.s.SHRD_ACC       = v;
    else if (strcmp(k, "REG_RD")         == 0) m_dice_params.s.REG_RD         = v;
    else if (strcmp(k, "REG_WR")         == 0) m_dice_params.s.REG_WR         = v;
    else if (strcmp(k, "NON_REG_OPs")    == 0) m_dice_params.s.NON_REG_OPs    = v;
    else if (strcmp(k, "INT_ACC")        == 0) m_dice_params.s.INT_ACC        = v;
    else if (strcmp(k, "FP_ACC")         == 0) m_dice_params.s.FP_ACC         = v;
    else if (strcmp(k, "DP_ACC")         == 0) m_dice_params.s.DP_ACC         = v;
    else if (strcmp(k, "INT_MUL24_ACC")  == 0) m_dice_params.s.INT_MUL24_ACC  = v;
    else if (strcmp(k, "INT_MUL32_ACC")  == 0) m_dice_params.s.INT_MUL32_ACC  = v;
    else if (strcmp(k, "INT_MUL_ACC")    == 0) m_dice_params.s.INT_MUL_ACC    = v;
    else if (strcmp(k, "INT_DIV_ACC")    == 0) m_dice_params.s.INT_DIV_ACC    = v;
    else if (strcmp(k, "FP_MUL_ACC")     == 0) m_dice_params.s.FP_MUL_ACC     = v;
    else if (strcmp(k, "FP_DIV_ACC")     == 0) m_dice_params.s.FP_DIV_ACC     = v;
    else if (strcmp(k, "FP_SQRT_ACC")    == 0) m_dice_params.s.FP_SQRT_ACC    = v;
    else if (strcmp(k, "FP_LG_ACC")      == 0) m_dice_params.s.FP_LG_ACC      = v;
    else if (strcmp(k, "FP_SIN_ACC")     == 0) m_dice_params.s.FP_SIN_ACC     = v;
    else if (strcmp(k, "FP_EXP_ACC")     == 0) m_dice_params.s.FP_EXP_ACC     = v;
    else if (strcmp(k, "DP_MUL_ACC")     == 0) m_dice_params.s.DP_MUL_ACC     = v;
    else if (strcmp(k, "DP_DIV_ACC")     == 0) m_dice_params.s.DP_DIV_ACC     = v;
    else if (strcmp(k, "TENSOR_ACC")     == 0) m_dice_params.s.TENSOR_ACC     = v;
    else if (strcmp(k, "TEX_ACC")        == 0) m_dice_params.s.TEX_ACC        = v;
    else if (strcmp(k, "MEM_RD")         == 0) m_dice_params.s.MEM_RD         = v;
    else if (strcmp(k, "MEM_WR")         == 0) m_dice_params.s.MEM_WR         = v;
    else if (strcmp(k, "MEM_PRE")        == 0) m_dice_params.s.MEM_PRE        = v;
    else if (strcmp(k, "L2_RH")          == 0) m_dice_params.s.L2_RH          = v;
    else if (strcmp(k, "L2_RM")          == 0) m_dice_params.s.L2_RM          = v;
    else if (strcmp(k, "L2_WH")          == 0) m_dice_params.s.L2_WH          = v;
    else if (strcmp(k, "L2_WM")          == 0) m_dice_params.s.L2_WM          = v;
    else if (strcmp(k, "NOC_A")          == 0) m_dice_params.s.NOC_A          = v;
    else if (strcmp(k, "PIPE_A")         == 0) m_dice_params.s.PIPE_A         = v;
    else if (strcmp(k, "IDLE_CORE_N")    == 0) m_dice_params.s.IDLE_CORE_N    = v;
    else if (strcmp(k, "constant_power") == 0) m_dice_params.s.constant_power = v;
    else if (strcmp(k, "idle_core_power")== 0) m_dice_params.idle_core_p_mW   = v;
    // ---- DICEwattch per-pipe energy constants (one per pipe per GPU) ----
    else if (strcmp(k, "DICE_E_INT_PIPE") == 0) m_dice_params.e.E_int_pipe = v;
    else if (strcmp(k, "DICE_E_FPU_PIPE") == 0) m_dice_params.e.E_fpu_pipe = v;
    else if (strcmp(k, "DICE_E_DPU_PIPE") == 0) m_dice_params.e.E_dpu_pipe = v;
    else if (strcmp(k, "DICE_E_SFU_PIPE") == 0) m_dice_params.e.E_sfu_pipe = v;
    else if (strcmp(k, "DICE_E_ICACHE")   == 0) m_dice_params.e.E_icache   = v;
    else if (strcmp(k, "DICE_E_CCACHE")   == 0) m_dice_params.e.E_ccache   = v;
    else if (strcmp(k, "DICE_E_TCACHE")   == 0) m_dice_params.e.E_tcache   = v;
    else if (strcmp(k, "DICE_E_DCACHE")   == 0) m_dice_params.e.E_dcache   = v;
    else if (strcmp(k, "DICE_E_SHMEM")    == 0) m_dice_params.e.E_shmem    = v;
    else if (strcmp(k, "DICE_E_L2")       == 0) m_dice_params.e.E_l2       = v;
    else if (strcmp(k, "DICE_E_DRAM")     == 0) m_dice_params.e.E_dram     = v;
    else if (strcmp(k, "DICE_E_MCP")      == 0) m_dice_params.e.E_mcp      = v;
    else if (strcmp(k, "DICE_E_NOC")      == 0) m_dice_params.e.E_noc      = v;
    else if (strcmp(k, "DICE_E_RF_READ")  == 0) m_dice_params.e.E_rf_read  = v;
    else if (strcmp(k, "DICE_E_RF_WRITE") == 0) m_dice_params.e.E_rf_write = v;
    else if (strcmp(k, "DICE_E_IBP")      == 0) m_dice_params.e.E_ibp      = v;
    // ---- Static / clock ----
    else if (strcmp(k, "DICE_STATIC_P_MW")   == 0) m_dice_params.static_p_mW   = v;
    else if (strcmp(k, "DICE_CLOCK_FREQ_GHZ")== 0) m_dice_params.clock_freq_ghz= v;
    else {
      fprintf(stderr, "[DICE-Power] WARN: unknown param %s in %s\n",
              k, dice_xml_path);
    }
  }
  m_dice_params.loaded = true;
  m_dice_power_enabled = true;
  fprintf(stdout, "[DICE-Power] loaded overlay XML: %s\n", dice_xml_path);
}

// Emit a DICEwattch power report in the same on-disk format as
// gpgpu_sim_wrapper::print_power_kernel_stats() (GPUWattch). Per-kernel
// avg/max/min are identical here because DICEwattch operates on cumulative
// kernel totals rather than per-sample data, but the format mirrors
// GPUWattch so the same downstream parsers/diff tools can consume both.
void gpgpu_sim_wrapper::print_dice_power_kernel_report(
    const std::string& kernel_info_string, unsigned long long gpu_sim_cycle,
    const dice_report_counters_t& c_cum) {
  if (!m_dice_power_enabled || !m_dice_params.loaded) return;
  const dice_power_params_t& d = m_dice_params;

  // Caller passes cumulative-since-sim-start counter values. AccelWattch's
  // per-kernel power needs per-kernel deltas, so subtract the previous
  // kernel's snapshot. Without this, multi-invocation benchmarks (BFS, GE,
  // gaussian) inflate every later kernel's reported power.
  static dice_report_counters_t prev_c = {};
  dice_report_counters_t c;
  c.l1b_acc              = c_cum.l1b_acc              - prev_c.l1b_acc;
  c.simt_stack_rd        = c_cum.simt_stack_rd        - prev_c.simt_stack_rd;
  c.simt_stack_wr        = c_cum.simt_stack_wr        - prev_c.simt_stack_wr;
  c.dispatched_threads   = c_cum.dispatched_threads   - prev_c.dispatched_threads;
  c.scoreboard_ld_reserve= c_cum.scoreboard_ld_reserve- prev_c.scoreboard_ld_reserve;
  c.e_blocks             = c_cum.e_blocks             - prev_c.e_blocks;
  c.cta                  = c_cum.cta                  - prev_c.cta;
  c.reg_reads            = c_cum.reg_reads            - prev_c.reg_reads;
  c.reg_writes           = c_cum.reg_writes           - prev_c.reg_writes;
  c.l1d_acc              = c_cum.l1d_acc              - prev_c.l1d_acc;
  c.icache_acc           = c_cum.icache_acc           - prev_c.icache_acc;
  c.ccache_acc           = c_cum.ccache_acc           - prev_c.ccache_acc;
  c.bcache_acc           = c_cum.bcache_acc           - prev_c.bcache_acc;
  c.tcache_acc           = c_cum.tcache_acc           - prev_c.tcache_acc;
  c.shmem_acc            = c_cum.shmem_acc            - prev_c.shmem_acc;
  c.acc_ops              = c_cum.acc_ops              - prev_c.acc_ops;
  c.int_ops              = c_cum.int_ops              - prev_c.int_ops;
  c.fpu_ops              = c_cum.fpu_ops              - prev_c.fpu_ops;
  c.sfu_ops              = c_cum.sfu_ops              - prev_c.sfu_ops;
  c.dp_ops               = c_cum.dp_ops               - prev_c.dp_ops;
  c.int_mul24_ops        = c_cum.int_mul24_ops        - prev_c.int_mul24_ops;
  c.int_mul32_ops        = c_cum.int_mul32_ops        - prev_c.int_mul32_ops;
  c.int_mul_ops          = c_cum.int_mul_ops          - prev_c.int_mul_ops;
  c.int_div_ops          = c_cum.int_div_ops          - prev_c.int_div_ops;
  c.fp_mul_ops           = c_cum.fp_mul_ops           - prev_c.fp_mul_ops;
  c.fp_div_ops           = c_cum.fp_div_ops           - prev_c.fp_div_ops;
  c.fp_sqrt_ops          = c_cum.fp_sqrt_ops          - prev_c.fp_sqrt_ops;
  c.fp_lg_ops            = c_cum.fp_lg_ops            - prev_c.fp_lg_ops;
  c.fp_sin_ops           = c_cum.fp_sin_ops           - prev_c.fp_sin_ops;
  c.fp_exp_ops           = c_cum.fp_exp_ops           - prev_c.fp_exp_ops;
  c.dp_mul_ops           = c_cum.dp_mul_ops           - prev_c.dp_mul_ops;
  c.dp_div_ops           = c_cum.dp_div_ops           - prev_c.dp_div_ops;
  c.tensor_ops           = c_cum.tensor_ops           - prev_c.tensor_ops;
  c.tex_ops              = c_cum.tex_ops              - prev_c.tex_ops;
  c.l2_read_acc          = c_cum.l2_read_acc          - prev_c.l2_read_acc;
  c.l2_write_acc         = c_cum.l2_write_acc         - prev_c.l2_write_acc;
  c.dram_rd              = c_cum.dram_rd              - prev_c.dram_rd;
  c.dram_wr              = c_cum.dram_wr              - prev_c.dram_wr;
  c.dram_pre             = c_cum.dram_pre             - prev_c.dram_pre;
  c.noc_flits            = c_cum.noc_flits            - prev_c.noc_flits;
  c.idle_core_cycles     = c_cum.idle_core_cycles     - prev_c.idle_core_cycles;
  prev_c = c_cum;

  // ---- Per-component dynamic energy (nJ).
  // For each shared-with-baseline counter the energy follows AccelWattch:
  //     E = raw_count * scaling_coeff * pipe_per_count_energy
  // (mirroring McPAT's counter accumulation + per-pipe energy model).
  // DICE-specific structures (IBP/BCP/SCHEDP/PIPEP-overlay) still use
  // their own per-access energies. ----
  const auto& s = d.s;
  const auto& e = d.e;
  // DICE has no instruction buffer (i-cache feeds the dispatcher directly,
  // and the SIMT stack drives p-graph metadata). IBP slot kept in the
  // report for format compatibility but always zero.
  const double e_IBP        = 0.0;
  const double e_ICP        = (c.icache_acc * s.IC_H) * e.E_icache;
  const double e_DCP        = (c.l1d_acc    * s.DC_RH) * e.E_dcache;
  const double e_TCP        = (c.tcache_acc * s.TC_H) * e.E_tcache;
  const double e_CCP        = (c.ccache_acc * s.CC_H) * e.E_ccache;
  const double e_SHRDP      = (c.shmem_acc  * s.SHRD_ACC) * e.E_shmem;
  const double e_RFP        = c.reg_reads  * s.REG_RD * e.E_rf_read
                            + c.reg_writes * s.REG_WR * e.E_rf_write;
  // ALU subdivisions: each routes to its pipe (INT / FPU / DPU / SFU).
  const double e_SPP        = c.int_ops        * s.INT_ACC       * e.E_int_pipe;
  const double e_FPUP       = c.fpu_ops        * s.FP_ACC        * e.E_fpu_pipe;
  const double e_DPUP       = c.dp_ops         * s.DP_ACC        * e.E_dpu_pipe;
  const double e_SFUP       = c.sfu_ops        * s.FP_SQRT_ACC   * e.E_sfu_pipe;
  const double e_INT_MUL24P = c.int_mul24_ops  * s.INT_MUL24_ACC * e.E_sfu_pipe;
  const double e_INT_MUL32P = c.int_mul32_ops  * s.INT_MUL32_ACC * e.E_sfu_pipe;
  const double e_INT_MULP   = c.int_mul_ops    * s.INT_MUL_ACC   * e.E_sfu_pipe;
  const double e_INT_DIVP   = c.int_div_ops    * s.INT_DIV_ACC   * e.E_sfu_pipe;
  const double e_FP_MULP    = c.fp_mul_ops     * s.FP_MUL_ACC    * e.E_sfu_pipe;
  const double e_FP_DIVP    = c.fp_div_ops     * s.FP_DIV_ACC    * e.E_sfu_pipe;
  const double e_FP_SQRTP   = c.fp_sqrt_ops    * s.FP_SQRT_ACC   * e.E_sfu_pipe;
  const double e_FP_LGP     = c.fp_lg_ops      * s.FP_LG_ACC     * e.E_sfu_pipe;
  const double e_FP_SINP    = c.fp_sin_ops     * s.FP_SIN_ACC    * e.E_sfu_pipe;
  const double e_FP_EXP     = c.fp_exp_ops     * s.FP_EXP_ACC    * e.E_sfu_pipe;
  const double e_DP_MULP    = c.dp_mul_ops     * s.DP_MUL_ACC    * e.E_sfu_pipe;
  const double e_DP_DIVP    = c.dp_div_ops     * s.DP_DIV_ACC    * e.E_sfu_pipe;
  const double e_TENSORP    = c.tensor_ops     * s.TENSOR_ACC    * e.E_sfu_pipe;
  const double e_TEXP       = c.tex_ops        * s.TEX_ACC       * e.E_sfu_pipe;
  const double e_L2CP       = c.l2_read_acc  * s.L2_RH * e.E_l2
                            + c.l2_write_acc * s.L2_WH * e.E_l2;
  const double e_NOCP       = c.noc_flits * s.NOC_A * e.E_noc;
  const double e_DRAMP      = c.dram_rd  * s.MEM_RD  * e.E_dram
                            + c.dram_wr  * s.MEM_WR  * e.E_dram
                            + c.dram_pre * s.MEM_PRE * e.E_dram;
  // MCP: same counters as DRAMP but charged at the MC datapath energy.
  const double e_MCP        = c.dram_rd  * s.MEM_RD  * e.E_mcp
                            + c.dram_wr  * s.MEM_WR  * e.E_mcp
                            + c.dram_pre * s.MEM_PRE * e.E_mcp;
  // DICE-specific blocks (use their own per-access energies).
  const double e_SCHEDP     = c.e_blocks * d.sched_eblock_e;
  const double e_PIPEP      = c.simt_stack_rd      * d.simt_stack_rd_e
                            + c.simt_stack_wr      * d.simt_stack_wr_e
                            + c.dispatched_threads * d.dispatcher_th_e
                            + c.reg_reads          * d.scoreboard_rd_e
                            + 2.0 * c.scoreboard_ld_reserve * d.scoreboard_wr_e
                            + c.cta                * d.active_cta_e
                            + c.e_blocks           * d.bct_pushpop_e
                            + c.l1d_acc            * d.bct_dec_e
                            // Accumulator-PE op: one PE-cycle pipeline add
                            // per atom.shared firing. The acc-PE is a
                            // stateful self-feedback PE in DICE, so the
                            // op cost is PIPE_A, not SHRD_ACC.
                            + c.acc_ops            * d.dice_acc_op_e;
  const double e_BCP        = c.bcache_acc * d.bcache_read_e;

  // ---- Convert per-kernel energy to average power (mW).
  //      P[mW] = (E[nJ] * f[GHz]) / cycles. Fixed-power slots stay in mW.
  const double freq_ghz = d.clock_freq_ghz > 0 ? d.clock_freq_ghz : 1.47;
  auto mW = [&](double e_nJ) {
    return (gpu_sim_cycle > 0) ? (e_nJ * freq_ghz / (double)gpu_sim_cycle)
                               : 0.0;
  };

  // Indexed by pwr_cmp_t (so labels line up with pwr_cmp_label[]).
  double cmp_p[NUM_COMPONENTS_MODELLED] = {0};
  cmp_p[IBP]            = mW(e_IBP);
  cmp_p[ICP]            = mW(e_ICP);
  cmp_p[DCP]            = mW(e_DCP);
  cmp_p[TCP]            = mW(e_TCP);
  cmp_p[CCP]            = mW(e_CCP);
  cmp_p[SHRDP]          = mW(e_SHRDP);
  cmp_p[RFP]            = mW(e_RFP);
  cmp_p[SPP]            = mW(e_SPP);
  cmp_p[SFUP]           = mW(e_SFUP);
  cmp_p[FPUP]           = mW(e_FPUP);
  cmp_p[SCHEDP]         = mW(e_SCHEDP);
  cmp_p[L2CP]           = mW(e_L2CP);
  cmp_p[MCP]            = mW(e_MCP);
  cmp_p[NOCP]           = mW(e_NOCP);
  cmp_p[DRAMP]          = mW(e_DRAMP);
  cmp_p[PIPEP]          = mW(e_PIPEP);
  cmp_p[IDLE_COREP]     = d.idle_core_p_mW;
  cmp_p[CONST_DYNAMICP] = d.const_p_mW;
  cmp_p[BCP]            = mW(e_BCP);
  cmp_p[DPUP]           = mW(e_DPUP);
  cmp_p[INT_MUL24P]     = mW(e_INT_MUL24P);
  cmp_p[INT_MUL32P]     = mW(e_INT_MUL32P);
  cmp_p[INT_MULP]       = mW(e_INT_MULP);
  cmp_p[INT_DIVP]       = mW(e_INT_DIVP);
  cmp_p[FP_MULP]        = mW(e_FP_MULP);
  cmp_p[FP_DIVP]        = mW(e_FP_DIVP);
  cmp_p[FP_SQRTP]       = mW(e_FP_SQRTP);
  cmp_p[FP_LGP]         = mW(e_FP_LGP);
  cmp_p[FP_SINP]        = mW(e_FP_SINP);
  cmp_p[FP_EXP]         = mW(e_FP_EXP);
  cmp_p[DP_MULP]        = mW(e_DP_MULP);
  cmp_p[DP_DIVP]        = mW(e_DP_DIVP);
  cmp_p[TENSORP]        = mW(e_TENSORP);
  cmp_p[TEXP]           = mW(e_TEXP);
  cmp_p[STATICP]        = d.static_p_mW;

  double kernel_avg_power = 0;
  for (unsigned i = 0; i < NUM_COMPONENTS_MODELLED; ++i)
    kernel_avg_power += cmp_p[i];

  // Indexed by perf_count_t.
  double cnt[NUM_PERFORMANCE_COUNTERS] = {0};
  cnt[TOT_INST]      = c.dispatched_threads;
  cnt[FP_INT]        = c.int_ops + c.fpu_ops;
  cnt[IC_H]          = c.icache_acc;   // standalone DICEwattch treats all as hits
  cnt[IC_M]          = 0;
  cnt[DC_RH]         = c.l1d_acc;
  cnt[DC_RM]         = 0;
  cnt[DC_WH]         = 0;
  cnt[DC_WM]         = 0;
  cnt[TC_H]          = c.tcache_acc;
  cnt[TC_M]          = 0;
  cnt[CC_H]          = c.ccache_acc;
  cnt[CC_M]          = 0;
  cnt[SHRD_ACC]      = c.shmem_acc;
  cnt[REG_RD]        = c.reg_reads;
  cnt[REG_WR]        = c.reg_writes;
  cnt[NON_REG_OPs]   = 0;
  cnt[SP_ACC]        = c.int_ops + c.int_mul24_ops + c.int_mul32_ops
                     + c.int_mul_ops + c.int_div_ops;
  cnt[SFU_ACC]       = c.sfu_ops + c.fp_sqrt_ops + c.fp_lg_ops
                     + c.fp_sin_ops + c.fp_exp_ops + c.tensor_ops
                     + c.tex_ops;
  cnt[FPU_ACC]       = c.fpu_ops + c.fp_mul_ops + c.fp_div_ops
                     + c.dp_ops + c.dp_mul_ops + c.dp_div_ops;
  cnt[MEM_RD]        = c.dram_rd;
  cnt[MEM_WR]        = c.dram_wr;
  cnt[MEM_PRE]       = c.dram_pre;
  cnt[L2_RH]         = c.l2_read_acc;
  cnt[L2_RM]         = 0;
  cnt[L2_WH]         = c.l2_write_acc;
  cnt[L2_WM]         = 0;
  cnt[NOC_A]         = c.noc_flits;
  cnt[PIPE_A]        = 0;
  cnt[IDLE_CORE_N]   = c.idle_core_cycles;
  cnt[CONST_DYNAMICN]= 0;
  cnt[DICE_L1B_ACC]         = c.l1b_acc;
  cnt[DICE_SIMT_STACK_RD_N] = c.simt_stack_rd;
  cnt[DICE_SIMT_STACK_WR_N] = c.simt_stack_wr;
  cnt[DICE_DISPATCH_TH_N]   = c.dispatched_threads;
  cnt[DICE_SCB_LD_RSV_N]    = c.scoreboard_ld_reserve;
  cnt[DICE_E_BLOCKS_N]      = c.e_blocks;
  cnt[DICE_CTA_N]           = c.cta;
  // Note: acc_ops isn't allocated a perf_count_t slot, but we emit it
  // alongside the kernel header so a single-line `grep acc_ops` works.

  // ---- Cross-kernel accumulators (mirror GPUWattch's gpu_tot_*). ----
  static int    g_dice_kernel_count   = 0;
  static double g_dice_gpu_tot_avg    = 0.0;
  static double g_dice_gpu_tot_max    = 0.0;
  static double g_dice_gpu_tot_min    = 0.0;
  g_dice_kernel_count++;
  g_dice_gpu_tot_avg += kernel_avg_power;
  if (kernel_avg_power > g_dice_gpu_tot_max) g_dice_gpu_tot_max = kernel_avg_power;
  if (g_dice_gpu_tot_min == 0 || kernel_avg_power < g_dice_gpu_tot_min)
    g_dice_gpu_tot_min = kernel_avg_power;

  // ---- Write report (GPUWattch-style) ----
  static bool wrote_header = false;
  const char* report_path = "gpgpusim_dice_power_report.log";
  std::ofstream rf(report_path, std::ios::app);
  if (!rf.is_open()) {
    fprintf(stderr, "[DICEwattch] could not open %s for writing\n",
            report_path);
    return;
  }
  if (!wrote_header) {
    rf << "# DICEwattch Power Report (XML-driven, GPUWattch-format)\n";
    wrote_header = true;
  }

  rf << kernel_info_string << "\n";
  rf << "Kernel Average Power Data:\n";
  rf << "kernel_avg_power = " << kernel_avg_power << "\n";
  rf << "dice_acc_ops = " << c.acc_ops
     << ", e_acc_ops_nJ = " << (c.acc_ops * d.dice_acc_op_e) << "\n";
  for (unsigned i = 0; i < NUM_COMPONENTS_MODELLED; ++i)
    rf << "gpu_avg_" << pwr_cmp_label[i] << " = " << cmp_p[i] << "\n";
  for (unsigned i = 0; i < NUM_PERFORMANCE_COUNTERS; ++i)
    rf << "gpu_avg_" << perf_count_label[i] << " = " << cnt[i] << "\n";

  rf << "\nKernel Maximum Power Data:\n";
  rf << "kernel_max_power = " << kernel_avg_power << "\n";
  for (unsigned i = 0; i < NUM_COMPONENTS_MODELLED; ++i)
    rf << "gpu_max_" << pwr_cmp_label[i] << " = " << cmp_p[i] << "\n";
  for (unsigned i = 0; i < NUM_PERFORMANCE_COUNTERS; ++i)
    rf << "gpu_max_" << perf_count_label[i] << " = " << cnt[i] << "\n";

  rf << "\nKernel Minimum Power Data:\n";
  rf << "kernel_min_power = " << kernel_avg_power << "\n";
  for (unsigned i = 0; i < NUM_COMPONENTS_MODELLED; ++i)
    rf << "gpu_min_" << pwr_cmp_label[i] << " = " << cmp_p[i] << "\n";
  for (unsigned i = 0; i < NUM_PERFORMANCE_COUNTERS; ++i)
    rf << "gpu_min_" << perf_count_label[i] << " = " << cnt[i] << "\n";

  rf << "\nAccumulative Power Statistics Over Previous Kernels:\n";
  rf << "gpu_tot_avg_power = "
     << (g_dice_gpu_tot_avg / g_dice_kernel_count) << "\n";
  rf << "gpu_tot_max_power = " << g_dice_gpu_tot_max << "\n";
  rf << "gpu_tot_min_power = " << g_dice_gpu_tot_min << "\n";
  rf << "\n\n";
  rf.flush();
  rf.close();
  fprintf(stdout, "[DICEwattch] kernel report appended to %s\n", report_path);
}

// Record per-sample DICE-specific perf-counter deltas. Called from
// power_interface::mcpat_cycle() each sample.
void gpgpu_sim_wrapper::set_dice_power(double l1b_acc, double simt_stack_rd,
                                      double simt_stack_wr,
                                      double dispatched_threads,
                                      double scoreboard_ld_reserve,
                                      double e_blocks, double cta) {
  sample_perf_counters[DICE_L1B_ACC]         = l1b_acc;
  sample_perf_counters[DICE_SIMT_STACK_RD_N] = simt_stack_rd;
  sample_perf_counters[DICE_SIMT_STACK_WR_N] = simt_stack_wr;
  sample_perf_counters[DICE_DISPATCH_TH_N]   = dispatched_threads;
  sample_perf_counters[DICE_SCB_LD_RSV_N]    = scoreboard_ld_reserve;
  sample_perf_counters[DICE_E_BLOCKS_N]      = e_blocks;
  sample_perf_counters[DICE_CTA_N]           = cta;
}

gpgpu_sim_wrapper::~gpgpu_sim_wrapper() {}

bool gpgpu_sim_wrapper::sanity_check(double a, double b) {
  if (b == 0)
    return (abs(a - b) < 0.00001);
  else
    return (abs(a - b) / abs(b) < 0.00001);

  return false;
}
void gpgpu_sim_wrapper::init_mcpat(
    char* xmlfile, char* powerfilename, char* power_trace_filename,
    char* metric_trace_filename, char* steady_state_filename,
    bool power_sim_enabled, bool trace_enabled, bool steady_state_enabled,
    bool power_per_cycle_dump, double steady_power_deviation,
    double steady_min_period, int zlevel, double init_val,
    int stat_sample_freq) {
  // Write File Headers for (-metrics trace, -power trace)

  reset_counters();
  static bool mcpat_init = true;

  // initialize file name if it is not set
  time_t curr_time;
  time(&curr_time);
  char* date = ctime(&curr_time);
  char* s = date;
  while (*s) {
    if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
    if (*s == '\n' || *s == '\r') *s = 0;
    s++;
  }

  if (mcpat_init) {
    g_power_filename = powerfilename;
    g_power_trace_filename = power_trace_filename;
    g_metric_trace_filename = metric_trace_filename;
    g_steady_state_tracking_filename = steady_state_filename;
    xml_filename = xmlfile;
    g_power_simulation_enabled = power_sim_enabled;
    g_power_trace_enabled = trace_enabled;
    g_steady_power_levels_enabled = steady_state_enabled;
    g_power_trace_zlevel = zlevel;
    g_power_per_cycle_dump = power_per_cycle_dump;
    gpu_steady_power_deviation = steady_power_deviation;
    gpu_steady_min_period = steady_min_period;

    gpu_stat_sample_freq = stat_sample_freq;

    // p->sys.total_cycles=gpu_stat_sample_freq*4;
    p->sys.total_cycles = gpu_stat_sample_freq;
    power_trace_file = NULL;
    metric_trace_file = NULL;
    steady_state_tacking_file = NULL;

    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "w");
      metric_trace_file = gzopen(g_metric_trace_filename, "w");
      if ((power_trace_file == NULL) || (metric_trace_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(power_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);

      gzprintf(power_trace_file, "power,");
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        gzprintf(power_trace_file, pwr_cmp_label[i]);
      }
      gzprintf(power_trace_file, "\n");

      gzsetparams(metric_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(metric_trace_file, perf_count_label[i]);
      }
      gzprintf(metric_trace_file, "\n");

      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
    if (g_steady_power_levels_enabled) {
      steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "w");
      if ((steady_state_tacking_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(steady_state_tacking_file, g_power_trace_zlevel,
                  Z_DEFAULT_STRATEGY);
      gzprintf(steady_state_tacking_file, "start,end,power,IPC,");
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(steady_state_tacking_file, perf_count_label[i]);
      }
      gzprintf(steady_state_tacking_file, "\n");

      gzclose(steady_state_tacking_file);
    }

    mcpat_init = false;
    has_written_avg = false;
    powerfile.open(g_power_filename);
    int flg = chmod(g_power_filename, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(flg == 0);
  }
  sample_val = 0;
  init_inst_val = init_val;  // gpu_tot_sim_insn+gpu_sim_insn;
}

void gpgpu_sim_wrapper::reset_counters() {
  avg_max_min_counters<double> init;
  for (unsigned i = 0; i < num_perf_counters; ++i) {
    sample_perf_counters[i] = 0;
    kernel_cmp_perf_counters[i] = init;
  }
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    sample_cmp_pwr[i] = 0;
    kernel_cmp_pwr[i] = init;
  }

  // Reset per-kernel counters
  kernel_sample_count = 0;
  kernel_tot_power = 0;
  kernel_power = init;

  return;
}

void gpgpu_sim_wrapper::set_inst_power(bool clk_gated_lanes, double tot_cycles,
                                       double busy_cycles, double tot_inst,
                                       double int_inst, double fp_inst,
                                       double load_inst, double store_inst,
                                       double committed_inst) {
  p->sys.core[0].gpgpu_clock_gated_lanes = clk_gated_lanes;
  p->sys.core[0].total_cycles = tot_cycles;
  p->sys.core[0].busy_cycles = busy_cycles;
  p->sys.core[0].total_instructions =
      tot_inst * p->sys.scaling_coefficients[TOT_INST];
  p->sys.core[0].int_instructions =
      int_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].fp_instructions =
      fp_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].load_instructions = load_inst;
  p->sys.core[0].store_instructions = store_inst;
  p->sys.core[0].committed_instructions = committed_inst;
  sample_perf_counters[FP_INT] = int_inst + fp_inst;
  sample_perf_counters[TOT_INST] = tot_inst;
}

void gpgpu_sim_wrapper::set_regfile_power(double reads, double writes,
                                          double ops) {
  p->sys.core[0].int_regfile_reads =
      reads * p->sys.scaling_coefficients[REG_RD];
  p->sys.core[0].int_regfile_writes =
      writes * p->sys.scaling_coefficients[REG_WR];
  p->sys.core[0].non_rf_operands =
      ops * p->sys.scaling_coefficients[NON_REG_OPs];
  sample_perf_counters[REG_RD] = reads;
  sample_perf_counters[REG_WR] = writes;
  sample_perf_counters[NON_REG_OPs] = ops;
}

void gpgpu_sim_wrapper::set_icache_power(double hits, double misses) {
  p->sys.core[0].icache.read_accesses =
      hits * p->sys.scaling_coefficients[IC_H] +
      misses * p->sys.scaling_coefficients[IC_M];
  p->sys.core[0].icache.read_misses =
      misses * p->sys.scaling_coefficients[IC_M];
  sample_perf_counters[IC_H] = hits;
  sample_perf_counters[IC_M] = misses;
}

void gpgpu_sim_wrapper::set_ccache_power(double hits, double misses) {
  p->sys.core[0].ccache.read_accesses =
      hits * p->sys.scaling_coefficients[CC_H] +
      misses * p->sys.scaling_coefficients[CC_M];
  p->sys.core[0].ccache.read_misses =
      misses * p->sys.scaling_coefficients[CC_M];
  sample_perf_counters[CC_H] = hits;
  sample_perf_counters[CC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}

void gpgpu_sim_wrapper::set_tcache_power(double hits, double misses) {
  p->sys.core[0].tcache.read_accesses =
      hits * p->sys.scaling_coefficients[TC_H] +
      misses * p->sys.scaling_coefficients[TC_M];
  p->sys.core[0].tcache.read_misses =
      misses * p->sys.scaling_coefficients[TC_M];
  sample_perf_counters[TC_H] = hits;
  sample_perf_counters[TC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}

void gpgpu_sim_wrapper::set_shrd_mem_power(double accesses) {
  p->sys.core[0].sharedmemory.read_accesses =
      accesses * p->sys.scaling_coefficients[SHRD_ACC];
  sample_perf_counters[SHRD_ACC] = accesses;
}

void gpgpu_sim_wrapper::set_l1cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {
  p->sys.core[0].dcache.read_accesses =
      read_hits * p->sys.scaling_coefficients[DC_RH] +
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.read_misses =
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.write_accesses =
      write_hits * p->sys.scaling_coefficients[DC_WH] +
      write_misses * p->sys.scaling_coefficients[DC_WM];
  p->sys.core[0].dcache.write_misses =
      write_misses * p->sys.scaling_coefficients[DC_WM];
  sample_perf_counters[DC_RH] = read_hits;
  sample_perf_counters[DC_RM] = read_misses;
  sample_perf_counters[DC_WH] = write_hits;
  sample_perf_counters[DC_WM] = write_misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}

void gpgpu_sim_wrapper::set_l2cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {
  p->sys.l2.total_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                             read_misses * p->sys.scaling_coefficients[L2_RM] +
                             write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                            read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_accesses = write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_hits = read_hits * p->sys.scaling_coefficients[L2_RH];
  p->sys.l2.read_misses = read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_hits = write_hits * p->sys.scaling_coefficients[L2_WH];
  p->sys.l2.write_misses = write_misses * p->sys.scaling_coefficients[L2_WM];
  sample_perf_counters[L2_RH] = read_hits;
  sample_perf_counters[L2_RM] = read_misses;
  sample_perf_counters[L2_WH] = write_hits;
  sample_perf_counters[L2_WM] = write_misses;
}

void gpgpu_sim_wrapper::set_idle_core_power(double num_idle_core) {
  p->sys.num_idle_cores = num_idle_core;
  sample_perf_counters[IDLE_CORE_N] = num_idle_core;
}

void gpgpu_sim_wrapper::set_duty_cycle_power(double duty_cycle) {
  p->sys.core[0].pipeline_duty_cycle =
      duty_cycle * p->sys.scaling_coefficients[PIPE_A];
  sample_perf_counters[PIPE_A] = duty_cycle;
}

void gpgpu_sim_wrapper::set_mem_ctrl_power(double reads, double writes,
                                           double dram_precharge) {
  p->sys.mc.memory_accesses = reads * p->sys.scaling_coefficients[MEM_RD] +
                              writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.memory_reads = reads * p->sys.scaling_coefficients[MEM_RD];
  p->sys.mc.memory_writes = writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.dram_pre = dram_precharge * p->sys.scaling_coefficients[MEM_PRE];
  sample_perf_counters[MEM_RD] = reads;
  sample_perf_counters[MEM_WR] = writes;
  sample_perf_counters[MEM_PRE] = dram_precharge;
}

void gpgpu_sim_wrapper::set_exec_unit_power(double fpu_accesses,
                                            double ialu_accesses,
                                            double sfu_accesses) {
  p->sys.core[0].fpu_accesses =
      fpu_accesses * p->sys.scaling_coefficients[FPU_ACC];
  // Integer ALU (not present in Tesla)
  p->sys.core[0].ialu_accesses =
      ialu_accesses * p->sys.scaling_coefficients[SP_ACC];
  // Sfu accesses
  p->sys.core[0].mul_accesses =
      sfu_accesses * p->sys.scaling_coefficients[SFU_ACC];

  sample_perf_counters[SP_ACC] = ialu_accesses;
  sample_perf_counters[SFU_ACC] = sfu_accesses;
  sample_perf_counters[FPU_ACC] = fpu_accesses;
}

void gpgpu_sim_wrapper::set_active_lanes_power(double sp_avg_active_lane,
                                               double sfu_avg_active_lane) {
  p->sys.core[0].sp_average_active_lanes = sp_avg_active_lane;
  p->sys.core[0].sfu_average_active_lanes = sfu_avg_active_lane;
}

void gpgpu_sim_wrapper::set_NoC_power(double noc_tot_reads,
                                      double noc_tot_writes) {
  p->sys.NoC[0].total_accesses =
      noc_tot_reads * p->sys.scaling_coefficients[NOC_A] +
      noc_tot_writes * p->sys.scaling_coefficients[NOC_A];
  sample_perf_counters[NOC_A] = noc_tot_reads + noc_tot_writes;
}

void gpgpu_sim_wrapper::power_metrics_calculations() {
  total_sample_count++;
  kernel_sample_count++;

  // Current sample power
  double sample_power =
      proc->rt_power.readOp.dynamic + sample_cmp_pwr[CONST_DYNAMICP];

  // Average power
  // Previous + new + constant dynamic power (e.g., dynamic clocking power)
  kernel_tot_power += sample_power;
  kernel_power.avg = kernel_tot_power / kernel_sample_count;
  for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
    kernel_cmp_pwr[ind].avg += (double)sample_cmp_pwr[ind];
  }

  for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
    kernel_cmp_perf_counters[ind].avg += (double)sample_perf_counters[ind];
  }

  // Max Power
  if (sample_power > kernel_power.max) {
    kernel_power.max = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].max = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].max = sample_perf_counters[ind];
    }
  }

  // Min Power
  if (sample_power < kernel_power.min || (kernel_power.min == 0)) {
    kernel_power.min = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].min = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].min = sample_perf_counters[ind];
    }
  }

  gpu_tot_power.avg = (gpu_tot_power.avg + sample_power);
  gpu_tot_power.max =
      (sample_power > gpu_tot_power.max) ? sample_power : gpu_tot_power.max;
  gpu_tot_power.min =
      ((sample_power < gpu_tot_power.min) || (gpu_tot_power.min == 0))
          ? sample_power
          : gpu_tot_power.min;
}

void gpgpu_sim_wrapper::print_trace_files() {
  open_files();

  for (unsigned i = 0; i < num_perf_counters; ++i) {
    gzprintf(metric_trace_file, "%f,", sample_perf_counters[i]);
  }
  gzprintf(metric_trace_file, "\n");

  gzprintf(power_trace_file, "%f,", proc_power);
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    gzprintf(power_trace_file, "%f,", sample_cmp_pwr[i]);
  }
  gzprintf(power_trace_file, "\n");

  close_files();
}

void gpgpu_sim_wrapper::update_coefficients() {
  initpower_coeff[FP_INT] = proc->cores[0]->get_coefficient_fpint_insts();
  effpower_coeff[FP_INT] =
      initpower_coeff[FP_INT] * p->sys.scaling_coefficients[FP_INT];

  initpower_coeff[TOT_INST] = proc->cores[0]->get_coefficient_tot_insts();
  effpower_coeff[TOT_INST] =
      initpower_coeff[TOT_INST] * p->sys.scaling_coefficients[TOT_INST];

  initpower_coeff[REG_RD] =
      proc->cores[0]->get_coefficient_regreads_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  initpower_coeff[REG_WR] =
      proc->cores[0]->get_coefficient_regwrites_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  initpower_coeff[NON_REG_OPs] =
      proc->cores[0]->get_coefficient_noregfileops_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  effpower_coeff[REG_RD] =
      initpower_coeff[REG_RD] * p->sys.scaling_coefficients[REG_RD];
  effpower_coeff[REG_WR] =
      initpower_coeff[REG_WR] * p->sys.scaling_coefficients[REG_WR];
  effpower_coeff[NON_REG_OPs] =
      initpower_coeff[NON_REG_OPs] * p->sys.scaling_coefficients[NON_REG_OPs];

  initpower_coeff[IC_H] = proc->cores[0]->get_coefficient_icache_hits();
  initpower_coeff[IC_M] = proc->cores[0]->get_coefficient_icache_misses();
  effpower_coeff[IC_H] =
      initpower_coeff[IC_H] * p->sys.scaling_coefficients[IC_H];
  effpower_coeff[IC_M] =
      initpower_coeff[IC_M] * p->sys.scaling_coefficients[IC_M];

  initpower_coeff[CC_H] = (proc->cores[0]->get_coefficient_ccache_readhits() +
                           proc->get_coefficient_readcoalescing());
  initpower_coeff[CC_M] = (proc->cores[0]->get_coefficient_ccache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  effpower_coeff[CC_H] =
      initpower_coeff[CC_H] * p->sys.scaling_coefficients[CC_H];
  effpower_coeff[CC_M] =
      initpower_coeff[CC_M] * p->sys.scaling_coefficients[CC_M];

  initpower_coeff[TC_H] = (proc->cores[0]->get_coefficient_tcache_readhits() +
                           proc->get_coefficient_readcoalescing());
  initpower_coeff[TC_M] = (proc->cores[0]->get_coefficient_tcache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  effpower_coeff[TC_H] =
      initpower_coeff[TC_H] * p->sys.scaling_coefficients[TC_H];
  effpower_coeff[TC_M] =
      initpower_coeff[TC_M] * p->sys.scaling_coefficients[TC_M];

  initpower_coeff[SHRD_ACC] =
      proc->cores[0]->get_coefficient_sharedmemory_readhits();
  effpower_coeff[SHRD_ACC] =
      initpower_coeff[SHRD_ACC] * p->sys.scaling_coefficients[SHRD_ACC];

  initpower_coeff[DC_RH] = (proc->cores[0]->get_coefficient_dcache_readhits() +
                            proc->get_coefficient_readcoalescing());
  initpower_coeff[DC_RM] =
      (proc->cores[0]->get_coefficient_dcache_readmisses() +
       proc->get_coefficient_readcoalescing());
  initpower_coeff[DC_WH] = (proc->cores[0]->get_coefficient_dcache_writehits() +
                            proc->get_coefficient_writecoalescing());
  initpower_coeff[DC_WM] =
      (proc->cores[0]->get_coefficient_dcache_writemisses() +
       proc->get_coefficient_writecoalescing());
  effpower_coeff[DC_RH] =
      initpower_coeff[DC_RH] * p->sys.scaling_coefficients[DC_RH];
  effpower_coeff[DC_RM] =
      initpower_coeff[DC_RM] * p->sys.scaling_coefficients[DC_RM];
  effpower_coeff[DC_WH] =
      initpower_coeff[DC_WH] * p->sys.scaling_coefficients[DC_WH];
  effpower_coeff[DC_WM] =
      initpower_coeff[DC_WM] * p->sys.scaling_coefficients[DC_WM];

  initpower_coeff[L2_RH] = proc->get_coefficient_l2_read_hits();
  initpower_coeff[L2_RM] = proc->get_coefficient_l2_read_misses();
  initpower_coeff[L2_WH] = proc->get_coefficient_l2_write_hits();
  initpower_coeff[L2_WM] = proc->get_coefficient_l2_write_misses();
  effpower_coeff[L2_RH] =
      initpower_coeff[L2_RH] * p->sys.scaling_coefficients[L2_RH];
  effpower_coeff[L2_RM] =
      initpower_coeff[L2_RM] * p->sys.scaling_coefficients[L2_RM];
  effpower_coeff[L2_WH] =
      initpower_coeff[L2_WH] * p->sys.scaling_coefficients[L2_WH];
  effpower_coeff[L2_WM] =
      initpower_coeff[L2_WM] * p->sys.scaling_coefficients[L2_WM];

  initpower_coeff[IDLE_CORE_N] =
      p->sys.idle_core_power * proc->cores[0]->executionTime;
  effpower_coeff[IDLE_CORE_N] =
      initpower_coeff[IDLE_CORE_N] * p->sys.scaling_coefficients[IDLE_CORE_N];

  initpower_coeff[PIPE_A] = proc->cores[0]->get_coefficient_duty_cycle();
  effpower_coeff[PIPE_A] =
      initpower_coeff[PIPE_A] * p->sys.scaling_coefficients[PIPE_A];

  initpower_coeff[MEM_RD] = proc->get_coefficient_mem_reads();
  initpower_coeff[MEM_WR] = proc->get_coefficient_mem_writes();
  initpower_coeff[MEM_PRE] = proc->get_coefficient_mem_pre();
  effpower_coeff[MEM_RD] =
      initpower_coeff[MEM_RD] * p->sys.scaling_coefficients[MEM_RD];
  effpower_coeff[MEM_WR] =
      initpower_coeff[MEM_WR] * p->sys.scaling_coefficients[MEM_WR];
  effpower_coeff[MEM_PRE] =
      initpower_coeff[MEM_PRE] * p->sys.scaling_coefficients[MEM_PRE];

  initpower_coeff[SP_ACC] =
      proc->cores[0]->get_coefficient_ialu_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  ;
  initpower_coeff[SFU_ACC] = proc->cores[0]->get_coefficient_sfu_accesses();
  initpower_coeff[FPU_ACC] = proc->cores[0]->get_coefficient_fpu_accesses();

  effpower_coeff[SP_ACC] =
      initpower_coeff[SP_ACC] * p->sys.scaling_coefficients[SP_ACC];
  effpower_coeff[SFU_ACC] =
      initpower_coeff[SFU_ACC] * p->sys.scaling_coefficients[SFU_ACC];
  effpower_coeff[FPU_ACC] =
      initpower_coeff[FPU_ACC] * p->sys.scaling_coefficients[FPU_ACC];

  initpower_coeff[NOC_A] = proc->get_coefficient_noc_accesses();
  effpower_coeff[NOC_A] =
      initpower_coeff[NOC_A] * p->sys.scaling_coefficients[NOC_A];

  const_dynamic_power =
      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);

  for (unsigned i = 0; i < num_perf_counters; i++) {
    initpower_coeff[i] /= (proc->cores[0]->executionTime);
    effpower_coeff[i] /= (proc->cores[0]->executionTime);
  }
}

void gpgpu_sim_wrapper::update_components_power() {
  update_coefficients();

  proc_power = proc->rt_power.readOp.dynamic;

  sample_cmp_pwr[IBP] =
      (proc->cores[0]->ifu->IB->rt_power.readOp.dynamic +
       proc->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
       proc->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
       proc->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
       proc->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic) /
      (proc->cores[0]->executionTime);

  sample_cmp_pwr[ICP] = proc->cores[0]->ifu->icache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[DCP] = proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[TCP] = proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[CCP] = proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[SHRDP] =
      proc->cores[0]->lsu->sharedmemory.rt_power.readOp.dynamic /
      (proc->cores[0]->executionTime);

  sample_cmp_pwr[RFP] =
      (proc->cores[0]->exu->rfu->rt_power.readOp.dynamic /
       (proc->cores[0]->executionTime)) *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);

  sample_cmp_pwr[SPP] =
      (proc->cores[0]->exu->exeu->rt_power.readOp.dynamic /
       (proc->cores[0]->executionTime)) *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);

  sample_cmp_pwr[SFUP] = (proc->cores[0]->exu->mul->rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime));

  sample_cmp_pwr[FPUP] = (proc->cores[0]->exu->fp_u->rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime));

  sample_cmp_pwr[SCHEDP] = proc->cores[0]->exu->scheu->rt_power.readOp.dynamic /
                           (proc->cores[0]->executionTime);

  sample_cmp_pwr[L2CP] = (proc->XML->sys.number_of_L2s > 0)
                             ? proc->l2array[0]->rt_power.readOp.dynamic /
                                   (proc->cores[0]->executionTime)
                             : 0;

  sample_cmp_pwr[MCP] = (proc->mc->rt_power.readOp.dynamic -
                         proc->mc->dram->rt_power.readOp.dynamic) /
                        (proc->cores[0]->executionTime);

  sample_cmp_pwr[NOCP] =
      proc->nocs[0]->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

  sample_cmp_pwr[DRAMP] =
      proc->mc->dram->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

  sample_cmp_pwr[PIPEP] =
      proc->cores[0]->Pipeline_energy / (proc->cores[0]->executionTime);

  sample_cmp_pwr[IDLE_COREP] =
      proc->cores[0]->IdleCoreEnergy / (proc->cores[0]->executionTime);

  // This constant dynamic power (e.g., clock power) part is estimated via
  // regression model.
  sample_cmp_pwr[CONST_DYNAMICP] = 0;
  double cnst_dyn =
      proc->get_const_dynamic_power() / (proc->cores[0]->executionTime);
  // If the regression scaling term is greater than the recorded constant
  // dynamic power then use the difference (other portion already added to
  // dynamic power). Else, all the constant dynamic power is accounted for, add
  // nothing.
  if (p->sys.scaling_coefficients[CONST_DYNAMICN] > cnst_dyn)
    sample_cmp_pwr[CONST_DYNAMICP] =
        (p->sys.scaling_coefficients[CONST_DYNAMICN] - cnst_dyn);

  proc_power += sample_cmp_pwr[CONST_DYNAMICP];

  // BCP default = 0 unless DICE overlay enables it below.
  sample_cmp_pwr[BCP] = 0;

  // ---- DICE overlay: replace SCHEDP/PIPEP/RFP and populate BCP -----------
  // Formulas follow the '4subcore power calculator.xlsx' DICE Energy block:
  //   SCHEDP = N_e_blocks * sched_eblock_E
  //   PIPEP  = simt_stack_rd*E_rd + simt_stack_wr*E_wr
  //          + dispatched_th*E_disp + reg_rd*E_scb_rd
  //          + 2*scb_ld_rsv*E_scb_wr + cta*E_cta
  //          + e_blocks*E_bct_pp   + L1D_acc*E_bct_dec
  //   RFP    = reg_rd*E_rf_rd + reg_wr*E_rf_wr
  //   BCP    = L1B_acc*E_bc_rd
  // The XML defines per-access energies in nJ; per-sample power [W] is
  // therefore (sum_terms_nJ * 1e-9) / executionTime_seconds.
  if (m_dice_power_enabled && m_dice_params.loaded) {
    const double exec_t = proc->cores[0]->executionTime;
    const double nJ_to_J = 1e-9;
    const double n_eblocks   = sample_perf_counters[DICE_E_BLOCKS_N];
    const double n_stack_rd  = sample_perf_counters[DICE_SIMT_STACK_RD_N];
    const double n_stack_wr  = sample_perf_counters[DICE_SIMT_STACK_WR_N];
    const double n_disp_th   = sample_perf_counters[DICE_DISPATCH_TH_N];
    const double n_scb_ld    = sample_perf_counters[DICE_SCB_LD_RSV_N];
    const double n_cta       = sample_perf_counters[DICE_CTA_N];
    const double n_reg_rd    = sample_perf_counters[REG_RD];
    const double n_reg_wr    = sample_perf_counters[REG_WR];
    const double n_l1b       = sample_perf_counters[DICE_L1B_ACC];
    // Total L1D accesses (read + write), matching the workbook's B42.
    const double n_l1d       = sample_perf_counters[DC_RH] +
                               sample_perf_counters[DC_RM] +
                               sample_perf_counters[DC_WH] +
                               sample_perf_counters[DC_WM];

    double schedp_e_nJ = n_eblocks * m_dice_params.sched_eblock_e;
    double pipep_e_nJ  = n_stack_rd  * m_dice_params.simt_stack_rd_e
                       + n_stack_wr  * m_dice_params.simt_stack_wr_e
                       + n_disp_th   * m_dice_params.dispatcher_th_e
                       + n_reg_rd    * m_dice_params.scoreboard_rd_e
                       + 2.0 * n_scb_ld * m_dice_params.scoreboard_wr_e
                       + n_cta       * m_dice_params.active_cta_e
                       + n_eblocks   * m_dice_params.bct_pushpop_e
                       + n_l1d       * m_dice_params.bct_dec_e;
    double rfp_e_nJ    = n_reg_rd * m_dice_params.s.REG_RD * m_dice_params.e.E_rf_read
                       + n_reg_wr * m_dice_params.s.REG_WR * m_dice_params.e.E_rf_write;
    double bcp_e_nJ    = n_l1b * m_dice_params.bcache_read_e;

    sample_cmp_pwr[SCHEDP] = schedp_e_nJ * nJ_to_J / exec_t;
    sample_cmp_pwr[PIPEP]  = pipep_e_nJ  * nJ_to_J / exec_t;
    sample_cmp_pwr[RFP]    = rfp_e_nJ    * nJ_to_J / exec_t;
    sample_cmp_pwr[BCP]    = bcp_e_nJ    * nJ_to_J / exec_t;

    // Recompute proc_power as the strict sum of components so downstream
    // averaging and the kernel-stats report stay coherent (we just replaced
    // four of McPAT's slots).
    double sum = 0;
    for (unsigned i = 0; i < num_pwr_cmps; i++) sum += sample_cmp_pwr[i];
    proc_power = sum;
    // Skip the McPAT-internal sanity check in DICE mode by construction.
    return;
  }

  double sum_pwr_cmp = 0;
  for (unsigned i = 0; i < num_pwr_cmps; i++) {
    sum_pwr_cmp += sample_cmp_pwr[i];
  }
  bool check = false;
  check = sanity_check(sum_pwr_cmp, proc_power);
  assert("Total Power does not equal the sum of the components\n" && (check));
}

void gpgpu_sim_wrapper::compute() { proc->compute(); }
void gpgpu_sim_wrapper::print_power_kernel_stats(
    double gpu_sim_cycle, double gpu_tot_sim_cycle, double init_value,
    const std::string& kernel_info_string, bool print_trace) {
  detect_print_steady_state(1, init_value);
  if (g_power_simulation_enabled) {
    powerfile << kernel_info_string << std::endl;

    sanity_check((kernel_power.avg * kernel_sample_count), kernel_tot_power);
    powerfile << "Kernel Average Power Data:" << std::endl;
    powerfile << "kernel_avg_power = " << kernel_power.avg << std::endl;

    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_avg_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].avg / kernel_sample_count << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_avg_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].avg / kernel_sample_count
                << std::endl;
    }

    powerfile << std::endl << "Kernel Maximum Power Data:" << std::endl;
    powerfile << "kernel_max_power = " << kernel_power.max << std::endl;
    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_max_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].max << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_max_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].max << std::endl;
    }

    powerfile << std::endl << "Kernel Minimum Power Data:" << std::endl;
    powerfile << "kernel_min_power = " << kernel_power.min << std::endl;
    for (unsigned i = 0; i < num_pwr_cmps; ++i) {
      powerfile << "gpu_min_" << pwr_cmp_label[i] << " = "
                << kernel_cmp_pwr[i].min << std::endl;
    }
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      powerfile << "gpu_min_" << perf_count_label[i] << " = "
                << kernel_cmp_perf_counters[i].min << std::endl;
    }

    powerfile << std::endl
              << "Accumulative Power Statistics Over Previous Kernels:"
              << std::endl;
    powerfile << "gpu_tot_avg_power = "
              << gpu_tot_power.avg / total_sample_count << std::endl;
    powerfile << "gpu_tot_max_power = " << gpu_tot_power.max << std::endl;
    powerfile << "gpu_tot_min_power = " << gpu_tot_power.min << std::endl;
    powerfile << std::endl << std::endl;
    powerfile.flush();

    if (print_trace) {
      print_trace_files();
    }
  }
}
void gpgpu_sim_wrapper::dump() {
  if (g_power_per_cycle_dump) proc->displayEnergy(2, 5);
}

void gpgpu_sim_wrapper::print_steady_state(int position, double init_val) {
  double temp_avg = sample_val / (double)samples.size();
  double temp_ipc = (init_val - init_inst_val) /
                    (double)(samples.size() * gpu_stat_sample_freq);

  if ((samples.size() >
       gpu_steady_min_period)) {  // If steady state occurred for some time,
                                  // print to file
    has_written_avg = true;
    gzprintf(steady_state_tacking_file, "%u,%d,%f,%f,", sample_start,
             total_sample_count, temp_avg, temp_ipc);
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      gzprintf(steady_state_tacking_file, "%f,",
               samples_counter.at(i) / ((double)samples.size()));
    }
    gzprintf(steady_state_tacking_file, "\n");
  } else {
    if (!has_written_avg && position)
      gzprintf(steady_state_tacking_file,
               "ERROR! Not enough steady state points to generate average\n");
  }

  sample_start = 0;
  sample_val = 0;
  init_inst_val = init_val;
  samples.clear();
  samples_counter.clear();
  pwr_counter.clear();
  assert(samples.size() == 0);
}

void gpgpu_sim_wrapper::detect_print_steady_state(int position,
                                                  double init_val) {
  // Calculating Average
  if (g_power_simulation_enabled && g_steady_power_levels_enabled) {
    steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "a");
    if (position == 0) {
      if (samples.size() == 0) {
        // First sample
        sample_start = total_sample_count;
        sample_val = proc->rt_power.readOp.dynamic;
        init_inst_val = init_val;
        samples.push_back(proc->rt_power.readOp.dynamic);
        assert(samples_counter.size() == 0);
        assert(pwr_counter.size() == 0);

        for (unsigned i = 0; i < (num_perf_counters); ++i) {
          samples_counter.push_back(sample_perf_counters[i]);
        }

        for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
          pwr_counter.push_back(sample_cmp_pwr[i]);
        }
        assert(pwr_counter.size() == (double)num_pwr_cmps);
        assert(samples_counter.size() == (double)num_perf_counters);
      } else {
        // Get current average
        double temp_avg = sample_val / (double)samples.size();

        if (abs(proc->rt_power.readOp.dynamic - temp_avg) <
            gpu_steady_power_deviation) {  // Value is within threshold
          sample_val += proc->rt_power.readOp.dynamic;
          samples.push_back(proc->rt_power.readOp.dynamic);
          for (unsigned i = 0; i < (num_perf_counters); ++i) {
            samples_counter.at(i) += sample_perf_counters[i];
          }

          for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
            pwr_counter.at(i) += sample_cmp_pwr[i];
          }

        } else {  // Value exceeds threshold, not considered steady state
          print_steady_state(position, init_val);
        }
      }
    } else {
      print_steady_state(position, init_val);
    }
    gzclose(steady_state_tacking_file);
  }
}

void gpgpu_sim_wrapper::open_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "a");
      metric_trace_file = gzopen(g_metric_trace_filename, "a");
    }
  }
}
void gpgpu_sim_wrapper::close_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
  }
}
