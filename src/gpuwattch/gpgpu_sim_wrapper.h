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

#ifndef GPGPU_SIM_WRAPPER_H_
#define GPGPU_SIM_WRAPPER_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include "processor.h"

using namespace std;

template <typename T>
struct avg_max_min_counters {
  T avg;
  T max;
  T min;

  avg_max_min_counters() {
    avg = 0;
    max = 0;
    min = 0;
  }
};

// ---- DICE power-model overlay parameters (per-access dynamic energies
//      in nJ, all already scaled to the simulator's target node).
//      Loaded from -gpgpu_dice_power_xml. -----------------------------------
// AccelWattch-style scaling coefficients (names match the param names
// in accelwattch_ptx_sim.xml so a user can copy any AccelWattch GPU XML
// section verbatim into the DICE overlay).
struct accelwattch_scaling_t {
  double TOT_INST; double FP_INT;
  double IC_H; double IC_M;
  double DC_RH; double DC_RM; double DC_WH; double DC_WM;
  double TC_H; double TC_M;
  double CC_H; double CC_M;
  double SHRD_ACC;
  double REG_RD; double REG_WR; double NON_REG_OPs;
  double INT_ACC; double FP_ACC; double DP_ACC;
  double INT_MUL24_ACC; double INT_MUL32_ACC; double INT_MUL_ACC; double INT_DIV_ACC;
  double FP_MUL_ACC; double FP_DIV_ACC;
  double FP_SQRT_ACC; double FP_LG_ACC; double FP_SIN_ACC; double FP_EXP_ACC;
  double DP_MUL_ACC; double DP_DIV_ACC;
  double TENSOR_ACC; double TEX_ACC;
  double MEM_RD; double MEM_WR; double MEM_PRE;
  double L2_RH; double L2_RM; double L2_WH; double L2_WM;
  double NOC_A;
  double PIPE_A; double IDLE_CORE_N; double constant_power;
};

// Per-pipeline McPAT-derived energy constants (one value per pipeline,
// per GPU model). These are the per-(scaled-counter-unit) energies that
// McPAT/CACTI would compute internally for a given technology + structure
// size; we expose them here so DICEwattch can produce an analytical
// AccelWattch-equivalent report without invoking McPAT at runtime.
struct dice_pipe_energy_t {
  double E_int_pipe;       // INTP base
  double E_fpu_pipe;       // FPUP base
  double E_dpu_pipe;       // DPUP base
  double E_sfu_pipe;       // all *_MUL* / *_DIV / FP_SQRT / LG / SIN / EXP / TENSOR / TEX
  double E_icache;         // ICP (per IC_H/IC_M scaled count)
  double E_ccache;         // CCP
  double E_tcache;         // TCP
  double E_dcache;         // DCP
  double E_shmem;          // SHRDP
  double E_l2;             // L2CP
  double E_dram;           // DRAMP
  double E_mcp;            // MCP
  double E_noc;            // NOCP
  double E_rf_read;        // RF read access
  double E_rf_write;       // RF write access
  double E_ibp;            // IBP (per TOT_INST)
};

struct dice_power_params_t {
  // ---- DICE-specific structures (unchanged) ----
  double bcache_read_e;
  double bcache_write_e;
  double simt_stack_rd_e;
  double simt_stack_wr_e;
  double scoreboard_rd_e;
  double scoreboard_wr_e;
  double dispatcher_th_e;
  double bct_pushpop_e;
  double bct_dec_e;
  double active_cta_e;
  double sched_eblock_e;
  // Per-op energy for the DICE accumulator-PE (one PE-cycle pipeline
  // add when atom.shared fires). Defaults to PIPE_A (~0.0257 nJ).
  double dice_acc_op_e;
  // ---- AccelWattch-format coefficients + per-pipe energies ----
  accelwattch_scaling_t s;
  dice_pipe_energy_t    e;
  // ---- Fixed-power slots ----
  double idle_core_p_mW;
  double const_p_mW;
  double static_p_mW;
  double clock_freq_ghz;
  bool loaded;
};

class gpgpu_sim_wrapper {
 public:
  gpgpu_sim_wrapper(bool power_simulation_enabled, char* xmlfile);
  ~gpgpu_sim_wrapper();

  // DICE power overlay control
  void enable_dice_power_model(const char* dice_xml_path);
  bool dice_power_model_enabled() const { return m_dice_power_enabled; }
  // Per-sample DICE perf counts (deltas since previous mcpat_cycle).
  void set_dice_power(double l1b_acc, double simt_stack_rd,
                      double simt_stack_wr, double dispatched_threads,
                      double scoreboard_ld_reserve, double e_blocks,
                      double cta);
  // Emit a standalone DICE power report (uses cumulative-counter totals
  // and the analytical workbook formulas). Independent of McPAT, so it
  // works whether or not -power_simulation_enabled is set.
  // Mirrors AccelWattch's per-kernel report format (per-component energy
  // and power, plus a total).
  struct dice_report_counters_t {
    // DICE-specific counters
    double l1b_acc;
    double simt_stack_rd;
    double simt_stack_wr;
    double dispatched_threads;
    double scoreboard_ld_reserve;
    double e_blocks;
    double cta;
    // RF / cache accesses (shared with baseline)
    double reg_reads;
    double reg_writes;
    double l1d_acc;
    double icache_acc;
    double ccache_acc;
    double bcache_acc;
    double tcache_acc;
    double shmem_acc;
    // DICE accumulator-PE op count (atom.shared firings). Charged at
    // PIPE_A energy, not at SHRD_ACC (SMEM bank).
    double acc_ops;
    // Arithmetic unit accesses (AccelWattch-aligned subdivision).
    double int_ops;        // base int (ALU_OP / INTP_OP, no special_op)
    double fpu_ops;        // base fp  (SP_OP,  no special_op)
    double sfu_ops;        // lumped fallback SFU
    double dp_ops;         // DP_OP without special_op
    double int_mul24_ops;
    double int_mul32_ops;
    double int_mul_ops;
    double int_div_ops;
    double fp_mul_ops;
    double fp_div_ops;
    double fp_sqrt_ops;
    double fp_lg_ops;
    double fp_sin_ops;
    double fp_exp_ops;
    double dp_mul_ops;
    double dp_div_ops;
    double tensor_ops;
    double tex_ops;
    // Memory subsystem
    double l2_read_acc;
    double l2_write_acc;
    double dram_rd;
    double dram_wr;
    double dram_pre;
    double noc_flits;
    // Idle core cycles (cumulative across all SMs)
    double idle_core_cycles;
  };
  void print_dice_power_kernel_report(
      const std::string& kernel_info_string,
      unsigned long long gpu_sim_cycle,
      const dice_report_counters_t& c);
  const dice_power_params_t& get_dice_params() const { return m_dice_params; }

  void init_mcpat(char* xmlfile, char* powerfile, char* power_trace_file,
                  char* metric_trace_file, char* steady_state_file,
                  bool power_sim_enabled, bool trace_enabled,
                  bool steady_state_enabled, bool power_per_cycle_dump,
                  double steady_power_deviation, double steady_min_period,
                  int zlevel, double init_val, int stat_sample_freq);
  void detect_print_steady_state(int position, double init_val);
  void close_files();
  void open_files();
  void compute();
  void dump();
  void print_trace_files();
  void update_components_power();
  void update_coefficients();
  void reset_counters();
  void print_power_kernel_stats(double gpu_sim_cycle, double gpu_tot_sim_cycle,
                                double init_value,
                                const std::string& kernel_info_string,
                                bool print_trace);
  void power_metrics_calculations();
  void set_inst_power(bool clk_gated_lanes, double tot_cycles,
                      double busy_cycles, double tot_inst, double int_inst,
                      double fp_inst, double load_inst, double store_inst,
                      double committed_inst);
  void set_regfile_power(double reads, double writes, double ops);
  void set_icache_power(double accesses, double misses);
  void set_ccache_power(double accesses, double misses);
  void set_tcache_power(double accesses, double misses);
  void set_shrd_mem_power(double accesses);
  void set_l1cache_power(double read_accesses, double read_misses,
                         double write_accesses, double write_misses);
  void set_l2cache_power(double read_accesses, double read_misses,
                         double write_accesses, double write_misses);
  void set_idle_core_power(double num_idle_core);
  void set_duty_cycle_power(double duty_cycle);
  void set_mem_ctrl_power(double reads, double writes, double dram_precharge);
  void set_exec_unit_power(double fpu_accesses, double ialu_accesses,
                           double sfu_accesses);
  void set_active_lanes_power(double sp_avg_active_lane,
                              double sfu_avg_active_lane);
  void set_NoC_power(double noc_tot_reads, double noc_tot_write);
  bool sanity_check(double a, double b);

 private:
  void print_steady_state(int position, double init_val);

  Processor* proc;
  ParseXML* p;
  // power parameters
  double const_dynamic_power;
  double proc_power;

  unsigned num_perf_counters;  // # of performance counters
  unsigned num_pwr_cmps;       // # of components modelled
  int kernel_sample_count;     // # of samples per kernel
  int total_sample_count;      // # of samples per benchmark

  std::vector<avg_max_min_counters<double> >
      kernel_cmp_pwr;  // Per-kernel component power avg/max/min values
  std::vector<avg_max_min_counters<double> >
      kernel_cmp_perf_counters;  // Per-kernel component avg/max/min performance
                                 // counters

  double kernel_tot_power;  // Total per-kernel power
  avg_max_min_counters<double>
      kernel_power;  // Per-kernel power avg/max/min values
  avg_max_min_counters<double>
      gpu_tot_power;  // Global GPU power avg/max/min values (across kernels)

  bool has_written_avg;

  std::vector<double> sample_cmp_pwr;  // Current sample component powers
  std::vector<double>
      sample_perf_counters;  // Current sample component perf. counts
  std::vector<double> initpower_coeff;
  std::vector<double> effpower_coeff;

  // For calculating steady-state average
  unsigned sample_start;
  double sample_val;
  double init_inst_val;
  std::vector<double> samples;
  std::vector<double> samples_counter;
  std::vector<double> pwr_counter;

  char* xml_filename;
  char* g_power_filename;
  char* g_power_trace_filename;
  char* g_metric_trace_filename;
  char* g_steady_state_tracking_filename;
  bool g_power_simulation_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_trace_enabled;
  bool g_power_per_cycle_dump;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;
  int g_power_trace_zlevel;
  double gpu_stat_sample_frequency;
  int gpu_stat_sample_freq;

  std::ofstream powerfile;
  gzFile power_trace_file;
  gzFile metric_trace_file;
  gzFile steady_state_tacking_file;

  // ---- DICE power overlay state ----
  bool m_dice_power_enabled;
  dice_power_params_t m_dice_params;
};

#endif /* GPGPU_SIM_WRAPPER_H_ */
