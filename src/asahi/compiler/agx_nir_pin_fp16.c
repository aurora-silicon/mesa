/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "nir_builder.h"

/* AGX3B composes a 16-bit register onto the f32add/f32mul/f32fma source
 * slots (f43/f26/f60) and narrows an f32 result into a half destination in
 * the same instruction, so an explicit float(f16(x)) addend or multiplicand
 * costs nothing when the rounded half survives to the backend. Upstream
 * nir_opt_algebraic collapses the inexact f2f32(f2f16(x)) round-trip long
 * before the backend runs -- "float? -> float? -> floatS" -- and DCE takes
 * the f16 def with it. Pin the widening exact wherever a direct f32 ALU use
 * can take the half in place, leaving every other round-trip on the cheaper
 * collapsed form. */

static bool
use_admits_half(nir_src *use)
{
   nir_instr *parent = nir_src_use_instr(use);
   if (parent->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu(parent);
   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
      return alu->def.bit_size == 32;
   default:
      return false;
   }
}

static bool
pin_fp16_widen(UNUSED nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_f2f32 || nir_alu_instr_is_exact(alu))
      return false;

   nir_def *src = alu->src[0].src.ssa;
   nir_instr *src_instr = nir_def_instr(src);
   if (!src_instr || src_instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *src_alu = nir_instr_as_alu(src_instr);
   switch (src_alu->op) {
   case nir_op_f2f16:
   case nir_op_f2f16_rtne:
   case nir_op_f2f16_rtz:
      break;
   default:
      return false;
   }

   bool feeds_f32_alu = false;
   nir_foreach_use(use, &alu->def)
      feeds_f32_alu |= use_admits_half(use);

   if (!feeds_f32_alu)
      return false;

   alu->fp_math_ctrl |= nir_fp_exact;
   return true;
}

bool
agx_nir_pin_fp16_widen(nir_shader *s)
{
   return nir_shader_alu_pass(s, pin_fp16_widen, nir_metadata_all, NULL);
}
