/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

/*
 * AGX generally flushes FP32 denorms. However, the min/max instructions do not
 * as they are implemented with cmpsel. We need to flush the results of fp32
 * min/max for correctness. Doing so naively will generate redundant flushes, so
 * this pass tries to be clever and elide flushes when possible.
 *
 */
/*
 * A consumer in the min/max algebra absorbs a possibly-denormal or NaN edge
 * only if it will inspect that edge itself. That means it must still be an
 * unlowered fmin/fmax: when it is lowered later in this same pass (or a later
 * invocation), could_be_denorm() sees through our raw fmin_agx/fmax_agx def
 * and it canonicalizes its own output where the edge can actually escape.
 *
 * An already-lowered fmin_agx/fmax_agx made its canonicalize decision while
 * our def was still a plain NIR fmin/fmax, which could_be_denorm() treats as
 * flushed. Eliding our flush would break that assumption, so it does not
 * count as absorbing.
 */
static bool
absorbs_edge(const nir_alu_instr *user)
{
   if (user->op != nir_op_fmin && user->op != nir_op_fmax)
      return false;

   /* A signed-zero-preserving min/max expands into compares and selects
    * that observe the raw edge, and a NaN-preserving one forwards a NaN
    * payload our fadd would otherwise canonicalize, so neither absorbs.
    */
   return !nir_alu_instr_is_signed_zero_preserve(user) &&
          !nir_alu_instr_is_nan_preserve(user);
}

/*
 * A denormal or NaN result edge only needs flushing where it leaves the
 * min/max algebra. fmin_agx/fmax_agx select between their two operand bit
 * patterns (IEEE minNum/maxNum): a denormal edge either wins and is flushed
 * by the consumer's own output canonicalize, or loses and never reaches the
 * output, and the consumer's could_be_denorm() recursion already sees through
 * a raw fmin_agx/fmax_agx def to decide. NaN edges are discarded by the same
 * selection, or forwarded to the same output canonicalize. Either way an
 * intermediate fadd(x, -0.0) is redundant when every use feeds another
 * fp32 min/max through value-preserving plumbing (copies, vector packs,
 * selects and phi nodes), which cannot observe the edge's bit pattern.
 *
 * Returns true when every use of def is absorbed by the min/max algebra.
 */
static bool
all_uses_in_minmax(nir_def *def, unsigned *budget)
{
   nir_foreach_use_including_if(src, def) {
      if (!*budget)
         return false;
      --*budget;

      /* An if condition tests the raw bits, it does not plumb them. */
      if (nir_src_is_if(src))
         return false;

      nir_instr *parent = nir_src_use_instr(src);
      if (parent->type == nir_instr_type_phi) {
         if (!all_uses_in_minmax(&nir_instr_as_phi(parent)->def, budget))
            return false;
         continue;
      }
      if (parent->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *user = nir_instr_as_alu(parent);
      if (absorbs_edge(user))
         continue;

      switch (user->op) {
      /* fcanonicalize flushes the edge itself, like our fadd would. */
      case nir_op_fcanonicalize:
         continue;

      /* The rest forward the edge's bit pattern (denormality and NaN-hood
       * included) to their own uses.
       */
      case nir_op_mov:
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
      case nir_op_vec5:
      case nir_op_vec8:
      case nir_op_vec16:
         if (!all_uses_in_minmax(&user->def, budget))
            return false;
         continue;
      case nir_op_bcsel:
         /* The condition observes the raw bits; only the two value edges
          * forward them.
          */
         if (src == &user->src[0].src)
            return false;
         if (!all_uses_in_minmax(&user->def, budget))
            return false;
         continue;
      default:
         return false;
      }
   }
   return true;
}

static bool
normal_bound(nir_scalar value, bool is_max)
{
   value = nir_scalar_chase_movs(value);
   if (!nir_scalar_is_const(value))
      return false;
   const uint32_t bits = nir_scalar_as_uint(value);
   const unsigned exponent = (bits >> 23) & 255;
   return exponent > 0 && exponent < 255 && (bits >> 31) == !is_max;
}

static bool
could_be_denorm(nir_scalar s, unsigned depth, unsigned *budget)
{
   if (!*budget)
      return true;
   --*budget;
   s = nir_scalar_chase_movs(s);
   /* Constants can be denorms only if they are denorms. */
   if (nir_scalar_is_const(s)) {
      /* Classify FP32 bits before promotion to double normalizes them. */
      uint32_t bits = nir_scalar_as_uint(s);
      return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
   }

   if (!depth)
      return true;
   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(s.def));
      nir_foreach_phi_src(src, phi) {
         if (could_be_denorm(nir_get_scalar(src->src.ssa, s.comp), depth - 1,
                             budget))
            return true;
      }
      return false;
   }
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_bcsel)
      return could_be_denorm(nir_scalar_chase_alu_src(s, 1), depth - 1,
                             budget) ||
             could_be_denorm(nir_scalar_chase_alu_src(s, 2), depth - 1, budget);

   /* Floating-point instructions flush denormals, so ALU results can only be
    * denormal if they are not from a float instruction. Crucially fmin/fmax
    * flushes in NIR, so this pass handles chains of fmin/fmax properly.
    */
   if (nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_fmin_agx || op == nir_op_fmax_agx) {
         nir_scalar a = nir_scalar_chase_alu_src(s, 0);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1);
         const bool is_max = op == nir_op_fmax_agx;
         if (normal_bound(a, is_max) || normal_bound(b, is_max))
            return false;
         return could_be_denorm(a, depth - 1, budget) ||
                could_be_denorm(b, depth - 1, budget);
      }
      nir_alu_type T = nir_op_infos[op].output_type;
      return nir_alu_type_get_base_type(T) != nir_type_float;
   }

   /* Otherwise, assume it could be denormal (say, loading from a buffer). */
   return true;
}

static bool
lower(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       (alu->def.bit_size != 32))
      return false;

   /* Lower the op, we'll fix up the denorms right after. */
   if (alu->op == nir_op_fmax)
      alu->op = nir_op_fmax_agx;
   else
      alu->op = nir_op_fmin_agx;

   bool canonicalize = false;
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      nir_scalar scalar = nir_get_scalar(&alu->def, c);
      nir_scalar src0 = nir_scalar_chase_alu_src(scalar, 0);
      nir_scalar src1 = nir_scalar_chase_alu_src(scalar, 1);
      const bool is_max = alu->op == nir_op_fmax_agx;
      if (normal_bound(src0, is_max) || normal_bound(src1, is_max))
         continue;
      unsigned budget = 64;
      canonicalize |=
         could_be_denorm(src0, 8, &budget) || could_be_denorm(src1, 8, &budget);
   }

   if (canonicalize) {
      /* If every use of the result stays inside the min/max algebra, each
       * denormal or NaN edge is absorbed downstream (see all_uses_in_minmax)
       * and this flush is redundant. Consumers inspect our raw fmin_agx/
       * fmax_agx def through could_be_denorm() and canonicalize their own
       * output only where it actually leaves the chain, so a chain of
       * min/max collapses to a single flush at the last escaping use.
       */
      unsigned budget = 64;
      if (!all_uses_in_minmax(&alu->def, &budget)) {
         b->cursor = nir_after_instr(&alu->instr);
         b->fp_math_ctrl = nir_fp_no_fast_math;

         nir_def *canonicalized = nir_fadd_imm(b, &alu->def, -0.0);
         nir_def_rewrite_uses_after(&alu->def, canonicalized);
      }
   }

   return true;
}

bool
agx_nir_lower_fminmax(nir_shader *s)
{
   return nir_shader_alu_pass(s, lower, nir_metadata_control_flow, NULL);
}
