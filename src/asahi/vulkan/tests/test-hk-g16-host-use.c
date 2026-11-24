/*
 * Copyright 2026 Dylan Ironside
 * SPDX-License-Identifier: MIT
 */

/* hk_g16_host_float_use() refuses host results that nir_opt_algebraic could
 * hand on unchanged to a reader that sees their bits as they are, and its
 * rule list mirrors nir_opt_algebraic.py. This test builds chains from a host
 * result h out of the folds it knows about and out of ordinary arithmetic,
 * ends each chain in one reader, and asks hk_g16_host_float_use() about every
 * use of h. It then runs the real algebraic passes to a fixed point with the
 * AGX options. Whenever every use of h was accepted, the bits of h must not
 * reach anything but a float ALU source that computes a new value, through
 * ops that pass an operand on unchanged, so a new identity rule that reopens
 * the hole fails here. */

#include "hk_g16_host_use.h"
#include "nir_builder.h"
#include "compiler/agx_compile.h"
#include <stdio.h>

enum step {
   S_MUL1,   /* fmul(x, 1.0) */
   S_ADDM0,  /* fadd(x, -0.0) */
   S_ADD0,   /* fadd(x, 0.0), folds under nsz */
   S_FMA1M0, /* ffma(x, 1.0, -0.0) */
   S_CANCU,  /* fadd(fneg(u), fadd(u, x)) */
   S_CANCV,  /* fadd(fneg(v), fadd(v, x)) */
   S_LRPU0,  /* flrp(x, u, 0.0), folds under nnan and nsz */
   S_LRPV0,  /* flrp(x, v, 0.0), folds under nnan and nsz */
   S_EXPLOG, /* fexp2(flog2(x)) */
   S_RCPRCP, /* frcp(frcp(x)) */
   S_POW1,   /* fpow(x, 1.0) */
   S_MULU,   /* fmul(x, u) */
   S_ADDU,   /* fadd(x, u) */
   S_FMAU,   /* ffma(x, u, u2) */
   S_MULV,   /* fmul(x, v) */
   S_ADDV,   /* fadd(x, v) */
   S_ADDV0,  /* fadd(x, fmul(v, 0.0)), folds under nsz and nnan */
   S_FMAV0,  /* ffma(fmul(v, 0.0), v2, x), folds under nsz and nnan */
   S_ADDVV,  /* fadd(x, fadd(fneg(v), v)), folds under nnan and nsz */
   S_MULB,   /* fmul(bcsel(v < 0, -1.0, 1.0), x) */
   S_ADDW,   /* fadd(x, fmul(v, v2)) */
   S_COUNT,
   S_IDENT = S_LRPU0, /* the long chains use the steps before this one */
};

enum reader {
   R_FNEG,
   R_FABS,
   R_FCSEL,
   R_BCSEL,
   R_FMIN,
   R_FMAX,
   R_FCANON,
   R_STORE,
   R_IADD,
   R_FSIN,
   R_FMULV,
   R_COUNT,
};

struct vals {
   nir_def *u, *u2, *v, *v2;
};

static nir_def *
apply(nir_builder *b, enum step s, nir_def *x, const struct vals *in)
{
   nir_def *u = in->u, *u2 = in->u2, *v = in->v, *v2 = in->v2;
   switch (s) {
   case S_MUL1:   return nir_fmul_imm(b, x, 1.0);
   case S_ADDM0:  return nir_fadd(b, x, nir_imm_float(b, -0.0f));
   case S_ADD0:   return nir_fadd(b, x, nir_imm_float(b, 0.0f));
   case S_FMA1M0: return nir_ffma(b, x, nir_imm_float(b, 1.0f),
                                  nir_imm_float(b, -0.0f));
   case S_CANCU:  return nir_fadd(b, nir_fneg(b, u), nir_fadd(b, u, x));
   case S_CANCV:  return nir_fadd(b, nir_fneg(b, v), nir_fadd(b, v, x));
   case S_LRPU0:  return nir_flrp(b, x, u, nir_imm_float(b, 0.0f));
   case S_LRPV0:  return nir_flrp(b, x, v, nir_imm_float(b, 0.0f));
   case S_EXPLOG: return nir_fexp2(b, nir_flog2(b, x));
   case S_RCPRCP: return nir_frcp(b, nir_frcp(b, x));
   case S_POW1:   return nir_fpow(b, x, nir_imm_float(b, 1.0f));
   case S_MULU:   return nir_fmul(b, x, u);
   case S_ADDU:   return nir_fadd(b, x, u);
   case S_FMAU:   return nir_ffma(b, x, u, u2);
   case S_MULV:   return nir_fmul(b, x, v);
   case S_ADDV:   return nir_fadd(b, x, v);
   case S_ADDV0:  return nir_fadd(b, x, nir_fmul_imm(b, v, 0.0));
   case S_FMAV0:  return nir_ffma(b, nir_fmul_imm(b, v, 0.0), v2, x);
   case S_ADDVV:  return nir_fadd(b, x, nir_fadd(b, nir_fneg(b, v), v));
   case S_MULB:
      return nir_fmul(b,
                      nir_bcsel(b, nir_flt_imm(b, v, 0.0),
                                nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f)),
                      x);
   case S_ADDW:   return nir_fadd(b, x, nir_fmul(b, v, v2));
   default:       return x;
   }
}

static nir_def *
emit_reader(nir_builder *b, enum reader r, nir_def *x, nir_def *v, nir_def *v2)
{
   switch (r) {
   case R_FNEG:   return nir_fneg(b, x);
   case R_FABS:   return nir_fabs(b, x);
   case R_FCSEL:  return nir_fcsel(b, v, x, v2);
   case R_BCSEL:  return nir_bcsel(b, nir_flt_imm(b, v, 0.0), x, v2);
   case R_FMIN:   return nir_fmin(b, x, v);
   case R_FMAX:   return nir_fmax(b, x, v);
   case R_FCANON: return nir_fcanonicalize(b, x);
   case R_IADD:   return nir_iadd_imm(b, x, 1);
   case R_FSIN:   return nir_fsin(b, x);
   case R_FMULV:  return nir_fmul(b, x, v);
   default:       return x;
   }
}

/* Whether the bits of `def` reach anything other than a float source of an
 * ALU instruction that computes a new value, following the ops that pass an
 * operand on unchanged or with only its sign changed. */
static bool
bits_escape(nir_def *def, unsigned depth)
{
   if (depth > 8)
      return true;
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return true;
      nir_instr *instr = nir_src_use_instr(use);
      if (instr->type != nir_instr_type_alu)
         return true;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      const nir_op_info *info = &nir_op_infos[alu->op];
      unsigned i = 0;
      while (i < info->num_inputs && &alu->src[i].src != use)
         ++i;
      if (i == info->num_inputs)
         return true;
      bool passes;
      switch (alu->op) {
      case nir_op_fneg:
      case nir_op_fabs:
      case nir_op_fmin:
      case nir_op_fmax:
      case nir_op_fcanonicalize:
         passes = true;
         break;
      case nir_op_bcsel:
      case nir_op_fcsel:
      case nir_op_fcsel_gt:
      case nir_op_fcsel_ge:
         passes = i != 0;
         break;
      default:
         passes = nir_op_is_vec_or_mov(alu->op);
         break;
      }
      if (passes) {
         if (bits_escape(&alu->def, depth + 1))
            return true;
      } else if (nir_alu_type_get_base_type(info->input_types[i]) !=
                 nir_type_float) {
         return true;
      }
   }
   return false;
}

static void
optimize(nir_shader *s)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_algebraic_late);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
}

struct tally {
   unsigned cases, accepted, holes, refused_would_leak;
};

static void
check(const enum step *steps, unsigned n, enum reader r, uint32_t fp_ctrl,
      struct tally *t)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "g16-host-use");
   b.fp_math_ctrl = fp_ctrl;
   nir_def *h = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *coord = nir_load_frag_coord(&b);
   const struct vals in = {
      .u = nir_load_preamble(&b, 1, 32, .base = 8),
      .u2 = nir_load_preamble(&b, 1, 32, .base = 10),
      .v = nir_channel(&b, coord, 0),
      .v2 = nir_channel(&b, coord, 1),
   };
   nir_def *x = h;
   for (unsigned i = 0; i < n; ++i)
      x = apply(&b, steps[i], x, &in);
   x = emit_reader(&b, r, x, in.v, in.v2);
   nir_store_global(&b, x, nir_imm_int64(&b, 0), .align_mul = 4);

   nir_divergence_analysis(b.shader);
   bool accepted = true;
   nir_foreach_use_including_if(use, h) {
      if (!hk_g16_host_float_use(use))
         accepted = false;
   }

   optimize(b.shader);
   bool leaks = bits_escape(h, 0);

   t->cases++;
   t->accepted += accepted;
   t->refused_would_leak += !accepted && leaks;
   if (accepted && leaks) {
      t->holes++;
      if (t->holes <= 4) {
         fprintf(stderr, "hole: fp_math_ctrl=0x%x reader=%u steps=", fp_ctrl,
                 r);
         for (unsigned i = 0; i < n; ++i)
            fprintf(stderr, "%s%u", i ? "," : "", steps[i]);
         fprintf(stderr, "\n");
         nir_print_shader(b.shader, stderr);
      }
   }
   ralloc_free(b.shader);
}

/* Every chain of `len` steps drawn from the first `alphabet` steps. */
static void
enumerate(unsigned len, unsigned alphabet, uint32_t fp_ctrl, struct tally *t)
{
   enum step steps[8];
   unsigned total = 1;
   for (unsigned i = 0; i < len; ++i)
      total *= alphabet;
   for (unsigned code = 0; code < total; ++code) {
      unsigned c = code;
      for (unsigned i = 0; i < len; ++i) {
         steps[i] = c % alphabet;
         c /= alphabet;
      }
      for (unsigned r = 0; r < R_COUNT; ++r)
         check(steps, len, r, fp_ctrl, t);
   }
}

int
main(void)
{
   static const uint32_t ctrls[] = {nir_fp_fast_math,
                                    nir_fp_preserve_sz_inf_nan};
   struct tally t = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(ctrls); ++i) {
      struct tally m = {0};
      for (unsigned len = 0; len <= 3; ++len)
         enumerate(len, S_COUNT, ctrls[i], &m);
      /* Identity chains past HK_G16_HOST_FOLD_DEPTH. */
      for (unsigned len = 4; len <= 6; ++len)
         enumerate(len, S_IDENT, ctrls[i], &m);
      printf("fp_math_ctrl=0x%x cases=%u accepted=%u holes=%u "
             "refused_would_leak=%u\n",
             ctrls[i], m.cases, m.accepted, m.holes, m.refused_would_leak);
      t.cases += m.cases;
      t.accepted += m.accepted;
      t.holes += m.holes;
      t.refused_would_leak += m.refused_would_leak;
   }

   printf("cases=%u accepted=%u holes=%u refused_would_leak=%u\n", t.cases,
          t.accepted, t.holes, t.refused_would_leak);
   /* Some refused chain must have leaked, or the test checks nothing. */
   return t.holes || !t.refused_would_leak || !t.accepted;
}
