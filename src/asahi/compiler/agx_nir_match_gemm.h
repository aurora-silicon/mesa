/* SPDX-License-Identifier: MIT */
#pragma once

#include "compiler/nir/nir.h"

enum {
   AGX_GEMM_WIDTH = 32,
   AGX_GEMM_SUBGROUPS = 4,
   AGX_GEMM_ROWS_PER_LANE = 8,
   AGX_GEMM_TILE_WORDS = AGX_GEMM_WIDTH * AGX_GEMM_WIDTH,
};

enum agx_gemm_bias_kind {
   AGX_GEMM_BIAS_NONE = 0,
   AGX_GEMM_BIAS_ROW = 1,
   AGX_GEMM_BIAS_COL = 2,
};

#define AGX_GEMM_MAX_C_GROUPS 4

struct agx_gemm_c_group {
   nir_block *insert_block;
   nir_instr *insert_before;
   bool guarded;
   enum agx_gemm_bias_kind bias_kind;
   enum gl_access_qualifier store_access;
   enum gl_access_qualifier bias_access;
   bool bias_constant;
   nir_def *c_base;
   nir_scalar c_stride;
   nir_def *bias_base;
   nir_def *row_base;
   nir_def *col_orig;
   nir_scalar m_limit;
   nir_scalar n_limit;
   bool has_clamp;
   bool clamp_agx;
   uint32_t clamp_lo_bits;
   uint32_t clamp_hi_bits;
   unsigned clamp_max_fp_ctrl;
   unsigned clamp_min_fp_ctrl;
   nir_intrinsic_instr *stores[AGX_GEMM_ROWS_PER_LANE];
};

struct agx_gemm_match {
   nir_loop *loop;
   nir_instr *ready;
   nir_instr *consumed;
   uint32_t a_bytes;
   uint32_t b_bytes;
   nir_phi_instr *initial[AGX_GEMM_ROWS_PER_LANE];
   nir_alu_instr *fma[AGX_GEMM_WIDTH][AGX_GEMM_ROWS_PER_LANE];
   nir_phi_instr *exit[AGX_GEMM_ROWS_PER_LANE];
   /* Optional direct device-memory B reads (skipping B shared-tile staging):
    * when all 8 B producer stores in the ready block copy an unguarded
    * row-major 32x32 slice from a read-only buffer, lower B fragments straight
    * to vec2 load_agx with immediate byte offsets (+0/+32/+64/+96B) and drop
    * B's 4 KiB shared tile.
    */
   bool b_direct;
   bool b_constant;
   enum gl_access_qualifier b_access;
   nir_intrinsic_instr *b_stores[AGX_GEMM_ROWS_PER_LANE];
   nir_def *b_base;
   nir_scalar b_stride;
   nir_def *b_k_base;
   nir_def *b_col_orig;
   /* Optional direct device-memory C writeback (skipping shared-memory C
    * staging and its workgroup barrier): when all uses of exit[0..7] write
    * directly to a row-major 32x32 C tile (with optional row/col bias and
    * optional row < M && col < N bounds guards), store vec2 fragments straight
    * from the cmat accumulator lanes. */
   bool c_direct;
   unsigned c_group_count;
   struct agx_gemm_c_group c_groups[AGX_GEMM_MAX_C_GROUPS];
};

/*
 * Match a 32x32 tile of FP32 dot products in explicitly lowered AGX3 shared
 * memory. The caller supplies a key with fixed 32-lane AGX3C subgroups.
 * Success certifies the shared tile, CFG and recurrence. The IR is unchanged
 * apart from analysis metadata.
 */
bool agx_nir_match_gemm(nir_shader *shader, struct agx_gemm_match *match);

/* Apply the conservative shared-memory epilogue to a successful match. */
bool agx_nir_lower_gemm(nir_shader *shader, const struct agx_gemm_match *match);
