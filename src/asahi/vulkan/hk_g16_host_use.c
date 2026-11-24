/*
 * Copyright 2026 Dylan Ironside
 * SPDX-License-Identifier: MIT
 */

#include "hk_g16_host_use.h"

/* Depth to which hk_g16_host_float_use() follows readers that may still fold
 * into a copy of a host result. Deeper chains are refused.
 *
 * The check runs before agx_compile_shader_nir(), whose nir_opt_algebraic
 * passes can still remove a reader and hand its readers the host result
 * itself. It assumes these rules of nir_opt_algebraic.py are the ones that do
 * so, each through fcanonicalize(a), which lines 161-163 drop when fp32
 * denormals need not be flushed:
 *
 *   fmul a, 1.0                                  (292)
 *   fadd a, -0.0 and fadd(nsz) a, 0.0            (236-237)
 *   ffma(nsz,nnan) +-0.0, a, b                   (305-306)
 *   flrp(nnan,nsz) a, b, +-0.0 and a, b, 1.0     (317-319)
 *   ~flrp a, a, b                                (320)
 *   ~fadd(fneg(a), fadd(a, b))                   (271-272)
 *   fexp2(flog2 a) and flog2(fexp2 a)            (1878-1879)
 *   fpow a, 1.0 and fpow(fpow(a, 2.2), 0.454545) (1893, 1898; AGX lowers fpow
 *                                                to fexp2 and flog2 first)
 *   frcp(frcp a)                                 (1917)
 *   f2f32(f2f64 a)                               (1293)
 *   fmul(bcsel(c, -1.0, 1.0), b)                 (2493, hands b to a bcsel)
 *
 * and, for readers the check refuses outright, fneg(fneg a) (230), fmin and
 * fmax a, a and the range forms (991-992, 1019-1022). fsat(nsz) a (1130) and
 * ffloor, fceil, ftrunc and ffract (2017-2028) need a range fact about the
 * operand that a value loaded from the uniforms never has.
 *
 * A reader becomes one of these identities only when certain other sources
 * fold to constants. nir_opt_algebraic also folds arithmetic on values that
 * vary across the draw to constants (fmul(nsz,nnan) a, 0.0 at 277,
 * fadd(fneg a, a) at 266-267), so only a source that varies across the draw
 * and is read from outside the arithmetic (an input, a texture or a load), or
 * is fmul, fadd or ffma of such sources alone, is known never to become one.
 *
 * tests/test-hk-g16-host-use.c builds chains from these rules, runs the real
 * passes with the AGX options and fails if a host result accepted here still
 * reaches a reader that sees its bits, so a new rule that does so shows up
 * there. */
#define HK_G16_HOST_FOLD_DEPTH 4

static bool hk_g16_host_float_use_depth(const nir_src *use, unsigned depth);

/* Whether `s` varies across the draw in a way nir_opt_algebraic never turns
 * into a constant: read, through moves only, from an intrinsic or a texture,
 * or computed by fmul, fadd or ffma from such values alone. No rule folds
 * those to a constant; the ones that fold arithmetic on varying values need a
 * constant operand or an fneg of the other one. */
static bool
hk_g16_host_scalar_varies(nir_scalar s, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (!s.def->divergent)
      return false;
   const nir_instr *instr = nir_def_instr(s.def);
   if (instr->type == nir_instr_type_intrinsic ||
       instr->type == nir_instr_type_tex)
      return true;
   if (instr->type != nir_instr_type_alu || depth >= HK_G16_HOST_FOLD_DEPTH)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_fmul && alu->op != nir_op_fadd &&
       alu->op != nir_op_ffma)
      return false;
   for (unsigned j = 0; j < nir_op_infos[alu->op].num_inputs; ++j) {
      if (!hk_g16_host_scalar_varies(nir_scalar_chase_alu_src(s, j),
                                     depth + 1))
         return false;
   }
   return true;
}

/* Whether every component of source `i` of `alu` passes
 * hk_g16_host_scalar_varies(). */
static bool
hk_g16_host_src_varies(const nir_alu_instr *alu, unsigned i)
{
   for (unsigned c = 0; c < nir_ssa_alu_instr_src_components(alu, i); ++c) {
      if (!hk_g16_host_scalar_varies(
             nir_get_scalar(alu->src[i].src.ssa, alu->src[i].swizzle[c]), 0))
         return false;
   }
   return true;
}

/* nir_opt_algebraic also turns fadd(fneg(a), fadd(a, b)) into b (inexact), so
 * `sum` = fadd(a, b), where b is the host result read through `use`, hands b
 * on unchanged to every reader of an fadd that adds fneg(a) to it. Those
 * readers must pass hk_g16_host_float_use() as well. */
static bool
hk_g16_host_cancel_ok(nir_alu_instr *sum, const nir_src *use, unsigned depth)
{
   const nir_def *a = sum->src[&sum->src[0].src == use ? 1 : 0].src.ssa;
   nir_foreach_use(next, &sum->def) {
      nir_instr *instr = nir_src_use_instr(next);
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *outer = nir_instr_as_alu(instr);
      if (outer->op != nir_op_fadd)
         continue;
      const nir_def *x =
         outer->src[&outer->src[0].src == next ? 1 : 0].src.ssa;
      nir_alu_instr *neg = nir_def_as_alu_or_null(x);
      if (!neg || neg->op != nir_op_fneg || neg->src[0].src.ssa != a)
         continue;
      if (depth >= HK_G16_HOST_FOLD_DEPTH)
         return false;
      nir_foreach_use_including_if(reader, &outer->def) {
         if (!hk_g16_host_float_use_depth(reader, depth + 1))
            return false;
      }
   }
   return true;
}

/* Whether `alu`, reading the host result as source `u`, can never be folded
 * into a copy of it: some source that every identity above needs to be a
 * constant varies across the draw (hk_g16_host_src_varies()), or the op has
 * no such identity. */
static bool
hk_g16_host_reader_kept(const nir_alu_instr *alu, unsigned u)
{
   switch (alu->op) {
   case nir_op_fmul:
   case nir_op_fmulz:
   case nir_op_fadd:
      return hk_g16_host_src_varies(alu, 1 - u);
   case nir_op_ffma:
   case nir_op_ffmaz:
      /* Multiplicand: needs the other one to be 1.0 and the addend -0.0.
       * Addend: needs either multiplicand to be 0.0. */
      if (u < 2)
         return hk_g16_host_src_varies(alu, 1 - u) ||
                hk_g16_host_src_varies(alu, 2);
      return hk_g16_host_src_varies(alu, 0) && hk_g16_host_src_varies(alu, 1);
   case nir_op_flrp:
      /* a or b: needs t to be 0.0 or 1.0, or a and b to be the same value. */
      return u == 2 || hk_g16_host_src_varies(alu, 2);
   case nir_op_fpow:
      /* Base: needs the exponent to be a constant. */
      return u == 1 || hk_g16_host_src_varies(alu, 1);
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_frcp:
   case nir_op_f2f64:
      /* Its own reader may undo it. */
      return false;
   default:
      return true;
   }
}

static bool
hk_g16_host_float_use_depth(const nir_src *use, unsigned depth)
{
   if (nir_src_is_if(use))
      return false;
   nir_instr *user = nir_src_use_instr(use);
   if (user->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(user);
   switch (alu->op) {
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fcsel:
   case nir_op_fcsel_gt:
   case nir_op_fcsel_ge:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fcanonicalize:
      return false;
   default:
      break;
   }
   const nir_op_info *info = &nir_op_infos[alu->op];
   unsigned u = info->num_inputs;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      if (&alu->src[i].src == use)
         u = i;
   }
   if (u == info->num_inputs ||
       nir_alu_type_get_base_type(info->input_types[u]) != nir_type_float)
      return false;
   if (alu->op == nir_op_fadd && !hk_g16_host_cancel_ok(alu, use, depth))
      return false;
   if (hk_g16_host_reader_kept(alu, u))
      return true;
   if (depth >= HK_G16_HOST_FOLD_DEPTH)
      return false;
   nir_foreach_use_including_if(next, &alu->def) {
      if (!hk_g16_host_float_use_depth(next, depth + 1))
         return false;
   }
   return true;
}

bool
hk_g16_host_float_use(const nir_src *use)
{
   return hk_g16_host_float_use_depth(use, 0);
}
