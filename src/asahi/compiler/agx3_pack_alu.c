/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 scalar/float/integer ALU, compares/selects, bitops, conversions,
 * transcendentals/derivatives, uniform ALU, and generic selector fallback.
 */

#include "agx3_pack_internal.h"
#include "agx3_copy_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_funop_core.h"
#include "agx3_iadd_core.h"
#include "agx3_isel_core.h"
#include "agx3_stage_core.h"
#include "agx3_uniform_alu.h"
#include "agx3_uniform_integer_instr.h"
#include "util/half_float.h"

#include <limits.h>
#include <math.h>

static enum agx3_sel_op
agx3_map_opcode(unsigned op)
{
   switch (op) {
   case AGX_OPCODE_STOP:
      return AGX3_SEL_OP_END;
   case AGX_OPCODE_THREADGROUP_BARRIER:
      return AGX3_SEL_OP_BARRIER;
   case AGX_OPCODE_MOV_IMM:
      return AGX3_SEL_OP_MOV_IMM;
   case AGX_OPCODE_FADD:
      return AGX3_SEL_OP_FADD;
   case AGX_OPCODE_FMUL:
      return AGX3_SEL_OP_FMUL;
   case AGX_OPCODE_FFMA:
      return AGX3_SEL_OP_FFMA;
   case AGX_OPCODE_IADD:
      return AGX3_SEL_OP_IADD;
   case AGX_OPCODE_IMAD:
      return AGX3_SEL_OP_IMAD;
   case AGX_OPCODE_ASR:
      return AGX3_SEL_OP_ASR;
   case AGX_OPCODE_BITOP:
      return AGX3_SEL_OP_BITOP;
   case AGX_OPCODE_BFI:
      return AGX3_SEL_OP_SLL;
   case AGX_OPCODE_BFEIL:
      return AGX3_SEL_OP_LSR;
   case AGX_OPCODE_RCP:
      return AGX3_SEL_OP_F_RCP;
   case AGX_OPCODE_RSQRT:
      return AGX3_SEL_OP_F_RSQRT;
   case AGX_OPCODE_ROUNDEVEN:
      return AGX3_SEL_OP_F_RINT;
   default:
      return AGX3_SEL_OP_UNSUPPORTED;
   }
}

/* Fill the destination and a single value source into the normalized form,
 * used by the ops the generic path cannot describe (convert). */
static void
agx3_set_dest_and_value(const agx_instr *I, agx_index value,
                        struct agx3_sel_instr *S)
{
   if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
      S->has_dest = true;
      S->dest = I->dest[0].value;
      S->dest_size = I->dest[0].size;
   }
   S->nr_srcs = 1;
   S->src[0].reg = value.value;
   S->src[0].size = value.size;
   S->src[0].is_reg = agx3_src_is_register(value);
   S->src[0].live = !value.kill && !value.discard;
}

static bool
agx3_translate_instr(const agx_instr *I, struct agx3_sel_instr *S)
{
   memset(S, 0, sizeof(*S));

   /* convert carries its direction as an immediate mode in the first source
    * and its rounding in `round`; the actual value is the second source. Only
    * the four 32-bit language casts NIR emits (f2i/f2u round-to-zero, i2f/u2f
    * round-to-nearest-even) are grounded. */
   if (I->op == AGX_OPCODE_CONVERT) {
      if (I->nr_srcs != 2 || I->src[0].type != AGX_INDEX_IMMEDIATE)
         return false;

      const unsigned mode = I->src[0].value;
      const enum agx_round round = I->round;

      if (mode == AGX_CONVERT_F_TO_S32 && round == AGX_ROUND_RTZ)
         S->op = AGX3_SEL_OP_F_TO_I32;
      else if (mode == AGX_CONVERT_F_TO_U32 && round == AGX_ROUND_RTZ)
         S->op = AGX3_SEL_OP_F_TO_U32;
      else if (mode == AGX_CONVERT_S32_TO_F && round == AGX_ROUND_RTE)
         S->op = AGX3_SEL_OP_I32_TO_F;
      else if (mode == AGX_CONVERT_U32_TO_F && round == AGX_ROUND_RTE)
         S->op = AGX3_SEL_OP_U32_TO_F;
      else
         return false;

      agx3_set_dest_and_value(I, I->src[1], S);
      return true;
   }

   /* fmin/fmax lower to fcmpsel(a, b, a, b, cond) -- a compare-select whose two
    * selected values are the two compared values, tied. cond LTN is a minimum,
    * GTN a maximum. Any other fcmpsel (a general ?: select) has no grounded
    * AGX3 mapping and is refused. */
   if (I->op == AGX_OPCODE_FCMPSEL) {
      if (I->nr_srcs != 4 || I->invert_cond ||
          !agx3_src_is_register(I->src[0]) ||
          !agx3_src_is_register(I->src[1]) || I->src[0].abs || I->src[0].neg ||
          I->src[1].abs || I->src[1].neg ||
          I->src[0].value != I->src[2].value ||
          I->src[1].value != I->src[3].value)
         return false;

      if (I->fcond == AGX_FCOND_LTN)
         S->op = AGX3_SEL_OP_FMIN;
      else if (I->fcond == AGX_FCOND_GTN)
         S->op = AGX3_SEL_OP_FMAX;
      else
         return false;

      if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
         S->has_dest = true;
         S->dest = I->dest[0].value;
         S->dest_size = I->dest[0].size;
      }
      S->nr_srcs = 2;
      S->src[0].reg = I->src[0].value;
      S->src[0].size = I->src[0].size;
      S->src[0].is_reg = true;
      S->src[0].live = !I->src[0].kill && !I->src[0].discard;
      S->src[1].reg = I->src[1].value;
      S->src[1].size = I->src[1].size;
      S->src[1].is_reg = true;
      S->src[1].live = !I->src[1].kill && !I->src[1].discard;
      return true;
   }

   S->op = agx3_map_opcode(I->op);
   if (S->op == AGX3_SEL_OP_UNSUPPORTED)
      return false;

   if (I->nr_srcs > 3)
      return false;

   S->saturate = I->saturate;
   S->shift = I->shift;
   S->imm = (uint32_t)I->imm;
   S->truth_table = I->truth_table;

   if (I->nr_dests >= 1 && I->dest[0].type != AGX_INDEX_NULL) {
      S->has_dest = true;
      S->dest = I->dest[0].value;
      S->dest_size = I->dest[0].size;
   }

   S->nr_srcs = I->nr_srcs;
   for (unsigned i = 0; i < I->nr_srcs; ++i) {
      S->src[i].reg = I->src[i].value;
      S->src[i].size = I->src[i].size;
      S->src[i].abs = I->src[i].abs;
      S->src[i].neg = I->src[i].neg;
      S->src[i].is_reg = agx3_src_is_register(I->src[i]);
      S->src[i].live = !I->src[i].kill && !I->src[i].discard;
   }

   /* NIR ishl/ushr arrive as a bitfield insert/extract with a zero base and a
    * zero insert mask: bfi(zero, value, amount) and bfeil(zero, value,
    * amount). That exact shape is a shift; anything else with these opcodes is
    * a genuine bitfield op with no grounded AGX3 mapping, so refuse it. Rewrite
    * the shift into the core's two-source (value, amount) form. */
   if (S->op == AGX3_SEL_OP_SLL || S->op == AGX3_SEL_OP_LSR) {
      if (I->nr_srcs != 3 || !agx3_src_is_zero_imm(I->src[0]) || I->imm != 0)
         return false;

      S->src[0] = S->src[1]; /* value  */
      S->src[1] = S->src[2]; /* amount */
      S->nr_srcs = 2;
      S->src[2] = (struct agx3_sel_src){0};
   }

   /* NIR isub is iadd(a, -b): an iadd whose second source is negated and whose
    * first is not. Map it to the grounded isub and drop the neg. A negate
    * (iadd(0, -a)) has an immediate-zero first source and is left unmapped. */
   if (S->op == AGX3_SEL_OP_IADD && I->nr_srcs == 2 && I->src[1].neg &&
       !I->src[0].neg && agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) && I->shift == 0) {
      S->op = AGX3_SEL_OP_ISUB;
      S->src[1].neg = false;
   }

   /* NIR imul is imad(a, b, 0): a multiply-add with a zero addend. Map it to
    * the grounded imul. */
   if (S->op == AGX3_SEL_OP_IMAD && I->nr_srcs == 3 &&
       agx3_src_is_zero_imm(I->src[2]) && I->shift == 0) {
      S->op = AGX3_SEL_OP_IMUL;
      S->nr_srcs = 2;
      S->src[2] = (struct agx3_sel_src){0};
   }

   /* Unary bitops (NOT = ~a, MOV = a) depend only on src[0]. Drop the unread
    * second source in the selector representation (matching SLL/LSR and IMUL
    * above) so agx3_emit_alu() does not inspect src[1] when checking source
    * registers, baseline lifetime bits, s1, or synchronous-prelude destination
    * overlap on waited MOV/NOT instructions. */
   if (S->op == AGX3_SEL_OP_BITOP && I->nr_srcs == 2 &&
       ((I->truth_table & 0xfu) == AGX3_SEL_TT_NOT ||
        (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV)) {
      S->nr_srcs = 1;
      S->src[1] = (struct agx3_sel_src){0};
   }

   return true;
}

bool
agx3_emit(struct util_dynarray *emission, enum agx_isa_encoding enc,
          const struct agx3_sel_instr *S)
{
   uint8_t bytes[AGX3_SEL_MAX_LENGTH];
   uint8_t len = 0;

   if (!agx3_select_encode(enc, S, bytes, &len))
      return false;

   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return true;
}

/* Native G17G fmin/fmax (compiled without fast math) retain condition 3/7
 * and tie each selected operand to its comparand. They are minNum/maxNum,
 * not general ordered comparisons: GE/LE plus a selection swap returns a
 * NaN when just one input is NaN. Keep the existing unordered-comparison
 * mapping for untied selects. */
static bool
agx3_fcmpsel_tied_minmax(const agx_instr *I)
{
   return !I->invert_cond && !I->saturate &&
          (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
          agx_is_equiv(I->src[0], I->src[2]) &&
          agx_is_equiv(I->src[1], I->src[3]);
}

static bool
agx3_fcmpsel_all_half_regs(const agx_instr *I)
{
   for (unsigned i = 0; i < 4; ++i) {
      const agx_index x = I->src[i];
      if (!agx3_src_is_register(x) || x.memory || x.size != AGX_SIZE_16 ||
          agx_channels(x) != 1 || x.abs || x.neg || x.kill ||
          (x.cache && x.discard))
         return false;
   }
   return true;
}

/*
 * The five comparison codes witnessed in Apple's own selects. f110 is
 * literally `enum agx_fcond`; eq, lt, gt, ge and le were each swept from a
 * kernel differing only in the operator. The NaN-ignoring ltn/gtn codes carry
 * the tied minimum/maximum forms instead and are not witnessed here.
 */
static bool
agx3_fcond_witnessed_select(enum agx_fcond cond)
{
   return cond == AGX_FCOND_EQ || cond == AGX_FCOND_LT ||
          cond == AGX_FCOND_GT || cond == AGX_FCOND_GE ||
          cond == AGX_FCOND_LE;
}

/* Reverse a comparison so the operands can be exchanged. */
static enum agx_fcond
agx3_fcond_reverse(enum agx_fcond cond)
{
   switch (cond) {
   case AGX_FCOND_LT: return AGX_FCOND_GT;
   case AGX_FCOND_GT: return AGX_FCOND_LT;
   case AGX_FCOND_LE: return AGX_FCOND_GE;
   case AGX_FCOND_GE: return AGX_FCOND_LE;
   default:           return cond;
   }
}

/*
 * Route native uniform arithmetic before the existing register-only paths.
 * Return 1 on emission, 0 for other instructions, and -1 on an invalid operand.
 */
static int
agx3_try_uniform_integer(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         unsigned wait)
{
   struct agx3_uniform_integer_desc d;
   if (!agx3_uniform_integer_for_encoding(enc, I, true, &d))
      return 0;
   if (wait >= (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 64))
      return -1;
   d.wait_mask = wait;
   uint8_t bytes[16], length;
   if (!agx3_uniform_integer_encode(enc, &d, bytes, &length))
      return -1;
   memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
   return 1;
}

static int
agx3_try_uniform_float(struct util_dynarray *emission,
                       enum agx_isa_encoding enc, const agx_instr *I, unsigned wait)
{
   const bool half = I->op == AGX_OPCODE_HADD || I->op == AGX_OPCODE_HMUL ||
                     I->op == AGX_OPCODE_HFMA;
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || (m5 && half) ||
       (!half && I->op != AGX_OPCODE_FADD && I->op != AGX_OPCODE_FMUL &&
        I->op != AGX_OPCODE_FFMA))
      return 0;
   bool uniform = false;
   agx_foreach_src(I, s)
      uniform |= I->src[s].type == AGX_INDEX_UNIFORM;
   if (!uniform)
      return 0;

   const bool fma = I->op == AGX_OPCODE_FFMA || I->op == AGX_OPCODE_HFMA;
   const bool add = I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_HADD;
   const unsigned count = fma ? 3 : 2;
   const enum agx_size size = half ? AGX_SIZE_16 : AGX_SIZE_32;
   if (I->nr_dests != 1 || I->nr_srcs != count ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       I->dest[0].size != size || agx_channels(I->dest[0]) != 1)
      return -1;

   struct agx3_uniform_float_src src[3] = {0};
   for (unsigned s = 0; s < count; ++s) {
      const agx_index v = I->src[s];
      if (v.memory || v.kill || agx_channels(v) != 1 ||
          (v.type != AGX_INDEX_IMMEDIATE && v.size != size &&
           !(!half && v.type == AGX_INDEX_UNIFORM && v.size == AGX_SIZE_16)))
         return -1;
      src[s] = (struct agx3_uniform_float_src){
         .kind = v.type == AGX_INDEX_UNIFORM     ? AGX3_FLOAT_UNIFORM
                 : v.type == AGX_INDEX_IMMEDIATE ? AGX3_FLOAT_IMMEDIATE
                 : agx3_src_is_register(v)       ? AGX3_FLOAT_REGISTER
                                                 : 0,
         .value = v.value,
         .abs = v.abs,
         .neg = v.neg,
         .cache = v.cache,
         .discard = v.discard,
         .half = v.size == AGX_SIZE_16 && v.type == AGX_INDEX_UNIFORM,
      };
   }
   const enum agx3_uniform_float_op op = fma ? AGX3_UNIFORM_FMA :
                                           add ? AGX3_UNIFORM_FADD : AGX3_UNIFORM_FMUL;
   uint8_t bytes[16], length;
   bool ok = half
      ? agx3_uniform_half_encode(enc, op, I->dest[0].value, src,
                                 I->saturate, I->dest[0].cache, wait, bytes, &length)
      : agx3_uniform_float_encode(enc, op, I->dest[0].value, src,
                                  I->saturate, I->dest[0].cache, wait, bytes, &length);
   if (!ok)
      return -1;
   memcpy(util_dynarray_grow_bytes(emission, 1, length), bytes, length);
   return 1;
}

/* Same return convention for modified register-source arithmetic. */
static int
agx3_try_float_modifier(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I, unsigned wait,
                        bool g18g)
{
   const bool is_add = I->op == AGX_OPCODE_FADD;
   const bool is_mul = I->op == AGX_OPCODE_FMUL;
   const bool is_fma = I->op == AGX_OPCODE_FFMA;
   if (!is_add && !is_mul && !is_fma)
      return 0;

   bool has_mod = I->saturate;
   bool scalar_regs = I->nr_dests == 1 && agx3_mod_src_ok(I->dest[0]) &&
                      agx_channels(I->dest[0]) == 1;
   for (unsigned s = 0; s < I->nr_srcs; ++s)
      has_mod |= I->src[s].abs || I->src[s].neg;
   /* Native AGX3B/AGX3C FMA (and AGX3C MUL) consumes full and half registers in
    * independently typed operand slots, including unmodified inputs. Route
    * mixed operands through the same lifetime-aware modifier encoder. Plain F32
    * stays compact. */
   if ((enc == AGX_ISA_ENCODING_AGX3C && (is_fma || is_mul)) ||
       (enc == AGX_ISA_ENCODING_AGX3B && is_fma))
      for (unsigned s = 0; s < I->nr_srcs; ++s)
         has_mod |= agx3_src_is_register(I->src[s]) &&
                    I->src[s].size == AGX_SIZE_16;
   const bool narrow = enc == AGX_ISA_ENCODING_AGX3C &&
                       I->dest[0].size == AGX_SIZE_16;
   has_mod |= narrow;
   if (!has_mod)
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      scalar_regs &= agx3_mod_src_ok(I->src[s]) && agx_channels(I->src[s]) == 1;
   }
   /* All-zero runtime hints are meaningful too: the legacy specimen encoder
    * contains discard flags that would kill live loop invariants. Preserve
    * the actual source lifetimes and waits for every plain scalar register
    * operation on G18G, including operations with no modifiers. */
   if (!has_mod && !(g18g && scalar_regs))
      return 0; /* the compact no-modifier packer handles it */

   const unsigned want = is_fma ? 3 : 2;
   /* Check the destination itself: a narrow source is admitted, so a vector
    * or spilled destination is no longer excluded by the source checks. */
   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       agx_channels(I->dest[0]) != 1 ||
       (!narrow && (I->dest[0].size != AGX_SIZE_32 ||
                        (I->dest[0].value & 1) != 0)) ||
       I->dest[0].value >= 256 || I->nr_srcs != want)
      return -1;
   /* G17G: a modified 32-bit register plus a minifloat constant is Apple's
    * immediate-operand f32add; Mesa's fabs arrives as `fadd abs(x), -0.0`.
    * Carried for G18G. */
   if (enc == AGX_ISA_ENCODING_AGX3C && g18g && is_add &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
       !I->src[1].neg && I->src[1].value <= 0xff &&
       agx3_mod_src_any_width(I->src[0]) && I->src[0].size == AGX_SIZE_32) {
      if (I->src[0].kill || wait > 0xff)
         return -1;
      const struct agx3_falu_mods m = {
         .src0_abs = I->src[0].abs,
         .src0_neg = I->src[0].neg,
         .saturate = I->saturate,
      };
      const struct agx3_falu_runtime runtime = {
         .cache = {I->src[0].cache},
         .discard = {I->src[0].discard},
         .dest_cache = I->dest[0].cache,
         .wait_mask = wait,
      };
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_falu_add_imm_runtime(I->dest[0].value, I->src[0].value,
                                     (uint8_t)I->src[1].value, &m, &runtime,
                                     bytes, &len))
         return -1;
      memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
      return 1;
   }

   for (unsigned s = 0; s < want; ++s)
      if (!agx3_mod_src_any_width(I->src[s]))
         return -1;

   struct agx3_falu_mods m = {
      .src0_abs = I->src[0].abs,
      .src0_neg = I->src[0].neg,
      .src0_16 = I->src[0].size == AGX_SIZE_16,
      .src1_abs = I->src[1].abs,
      .src1_neg = I->src[1].neg,
      .src1_16 = I->src[1].size == AGX_SIZE_16,
      .saturate = I->saturate,
   };

   uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
   uint8_t len = 0;
   bool ok;
   const uint16_t d = I->dest[0].value, s0 = I->src[0].value,
                  s1 = I->src[1].value;
   /* G18G: the runtime form's wait field is eight bits. */
   if (g18g && wait > 0xff)
      return -1;
   if ((is_add || is_mul) &&
       (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C)) {
      struct agx3_falu_runtime runtime = {
         .dest_cache = I->dest[0].cache,
         .wait_mask = wait,
      };
      for (unsigned i = 0; i < 2; ++i) {
         if (I->src[i].kill)
            return -1;
         runtime.cache[i] = I->src[i].cache;
         runtime.discard[i] = I->src[i].discard;
      }
      ok = narrow
         ? (is_add ? agx3_falu_add_narrow_m5(d, s0, s1, &m, &runtime, bytes, &len)
                   : agx3_falu_mul_narrow_m5(d, s0, s1, &m, &runtime, bytes, &len))
         : is_add
         ? agx3_falu_add_mod_runtime(enc, d, s0, s1, &m, &runtime, bytes, &len)
         : agx3_falu_mul_mod_runtime(enc, d, s0, s1, &m, &runtime, bytes, &len);
   } else if (is_add)
      ok = agx3_falu_add_mod(enc, d, s0, s1, &m, bytes, &len);
   else if (is_mul)
      ok = agx3_falu_mul_mod(enc, d, s0, s1, &m, bytes, &len);
   else {
      m.addend_abs = I->src[2].abs;
      m.addend_neg = I->src[2].neg;
      m.addend_16 = I->src[2].size == AGX_SIZE_16;
      if (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) {
         struct agx3_falu_runtime runtime = {
            .dest_cache = I->dest[0].cache,
            .wait_mask = wait,
         };
         for (unsigned i = 0; i < 3; ++i) {
            if (I->src[i].kill)
               return -1;
            runtime.cache[i] = I->src[i].cache;
            runtime.discard[i] = I->src[i].discard;
         }
         ok = narrow
            ? agx3_falu_fma_narrow_m5(d, s0, s1, I->src[2].value,
                                      &m, &runtime, bytes, &len)
            : agx3_falu_fma_mod_runtime(enc, d, s0, s1, I->src[2].value,
                                        &m, &runtime, bytes, &len);
      } else {
         ok = agx3_falu_fma_mod(enc, d, s0, s1, I->src[2].value,
                               &m, bytes, &len);
      }
   }
   if (!ok || (wait && enc != AGX_ISA_ENCODING_AGX3B && !is_add &&
               !((is_fma || is_mul) && enc == AGX_ISA_ENCODING_AGX3C)))
      return -1;

   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return 1;
}

/*
 * The transcendental / rounding / gradient funops with a corpus-grounded
 * AGX3 form. roundeven keeps its existing select-core route. The native sinc
 * has a dedicated IR opcode, fed by G17P's split-constant range reduction
 * and masked quadrant selects. AGX2 sin_pt_1/2 remain unmapped here.
 */
static int
agx3_map_funop(unsigned op)
{
   switch (op) {
   case AGX_OPCODE_FLOOR:
      return AGX3_FUNOP_FLOOR;
   case AGX_OPCODE_CEIL:
      return AGX3_FUNOP_CEIL;
   case AGX_OPCODE_TRUNC:
      return AGX3_FUNOP_TRUNC;
   case AGX_OPCODE_LOG2:
      return AGX3_FUNOP_LOG2;
   case AGX_OPCODE_EXP2:
      return AGX3_FUNOP_EXP2;
   case AGX_OPCODE_SINC_AGX3:
      return AGX3_FUNOP_SINC;
   case AGX_OPCODE_DFDX:
      return AGX3_FUNOP_DFDX;
   case AGX_OPCODE_DFDY:
      return AGX3_FUNOP_DFDY;
   default:
      return -1;
   }
}

/* G18G (agx3d) also takes the G17G-family reciprocal square root and square
 * root step through the funop encoder (agx3_emit_funop). */
static int
agx3_map_funop_g18g(unsigned op)
{
   switch (op) {
   case AGX_OPCODE_RSQRT:
      return AGX3_FUNOP_RSQRT;
   case AGX_OPCODE_SRSQRT:
      return AGX3_FUNOP_SQRT_STEP;
   default:
      return agx3_map_funop(op);
   }
}

static bool
agx3_emit_funop(struct util_dynarray *emission, enum agx_isa_encoding enc,
                const agx_instr *I, enum agx3_funop op, uint16_t wait_mask,
                bool g18g)
{
   /* G18G (agx3d) keeps j873's G17G-family unary encoder for its rounding,
    * log/exp, rsqrt and sqrt-step forms. */
   if (g18g && enc == AGX_ISA_ENCODING_AGX3C && op != AGX3_FUNOP_RINT &&
       (op <= AGX3_FUNOP_EXP2 || op == AGX3_FUNOP_RSQRT ||
        op == AGX3_FUNOP_SQRT_STEP)) {
      if (I->nr_dests != 1 || I->nr_srcs != 1 ||
          I->dest[0].type == AGX_INDEX_NULL ||
          !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          !agx3_src_is_register(I->src[0]) || I->src[0].memory ||
          I->src[0].kill || agx_channels(I->dest[0]) != 1 ||
          agx_channels(I->src[0]) != 1 ||
          (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32) ||
          (I->src[0].size != AGX_SIZE_16 && I->src[0].size != AGX_SIZE_32))
         return false;
      const struct agx3_funop_runtime runtime = {
         .dest_cache = I->dest[0].cache,
         .src_cache = I->src[0].cache,
         .src_discard = I->src[0].discard,
         .dest_16 = I->dest[0].size == AGX_SIZE_16,
         .src_16 = I->src[0].size == AGX_SIZE_16,
      };
      uint8_t bytes[AGX3_FUNOP_MAX_LENGTH];
      uint8_t len = 0;
      return agx3_funop_encode_g17g(op, I->dest[0].value, I->src[0].value,
                                    I->src[0].abs, I->src[0].neg, I->saturate,
                                    wait_mask, &runtime, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && op <= AGX3_FUNOP_EXP2) {
      if (I->nr_dests != 1 || I->nr_srcs != 1)
         return false;

      const agx_index dest = I->dest[0], src = I->src[0];
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 || !agx3_src_is_register(src) || src.memory ||
          (src.size != AGX_SIZE_16 && src.size != AGX_SIZE_32) ||
          agx_channels(src) != 1 || src.kill || (src.cache && src.discard))
         return false;

      const enum agx3_alu_hint hint = src.cache ? AGX3_ALU_HINT_CACHE :
                                    src.discard ? AGX3_ALU_HINT_DISCARD :
                                                  AGX3_ALU_HINT_NONE;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      return agx3_alu_unary_m5(op, dest.value, src.value,
                               dest.size == AGX_SIZE_16,
                               src.size == AGX_SIZE_16, src.abs, src.neg,
                               I->saturate, hint, dest.cache, wait_mask,
                               bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (enc == AGX_ISA_ENCODING_AGX3C &&
       (op == AGX3_FUNOP_DFDX || op == AGX3_FUNOP_DFDY)) {
      if (I->nr_dests != 1 || I->nr_srcs != 1)
         return false;

      const agx_index dest = I->dest[0], src = I->src[0];
      if (!agx3_src_is_register(dest) || dest.memory ||
          !agx3_src_is_register(src) || src.memory ||
          agx_channels(dest) != 1 || agx_channels(src) != 1 ||
          dest.size != src.size ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32))
         return false;

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      return agx3_alu_derivative_m5(op == AGX3_FUNOP_DFDY, dest.value,
                                    src.value, dest.size == AGX_SIZE_16,
                                    src.abs, src.neg, I->saturate, wait_mask,
                                    bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (I->nr_dests != 1 || I->nr_srcs != 1 ||
       I->dest[0].type == AGX_INDEX_NULL ||
       !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
       I->dest[0].size != AGX_SIZE_32 || !agx3_src_is_register(I->src[0]) ||
       I->src[0].memory || I->src[0].size != AGX_SIZE_32)
      return false;

   uint8_t bytes[AGX3_FUNOP_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_funop_encode_source_lifetime(
             enc, op, I->dest[0].value, I->src[0].value, I->src[0].abs,
             I->src[0].neg, I->saturate, wait_mask, !I->src[0].discard,
             bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* Map a resolved selection op to the composer ALU op that can carry a wait.
 * Returns AGX3_ALU_NUM for ops with no wait-carrying form. */
static enum agx3_alu_op
agx3_sel_to_alu(const struct agx3_sel_instr *S)
{
   switch (S->op) {
   case AGX3_SEL_OP_MOV_IMM:
      return AGX3_ALU_LDIMM;
   case AGX3_SEL_OP_IADD:
      return AGX3_ALU_IADD;
   case AGX3_SEL_OP_ISUB:
      return AGX3_ALU_ISUB;
   case AGX3_SEL_OP_IMAD:
      return AGX3_ALU_IMADD;
   case AGX3_SEL_OP_IMUL:
      return AGX3_ALU_IMUL;
   case AGX3_SEL_OP_FADD:
      return AGX3_ALU_FADD;
   case AGX3_SEL_OP_FMUL:
      return AGX3_ALU_FMUL;
   case AGX3_SEL_OP_FFMA:
      return AGX3_ALU_FMA;
   case AGX3_SEL_OP_ASR:
      return AGX3_ALU_ASR;
   case AGX3_SEL_OP_SLL:
      return AGX3_ALU_SLL;
   case AGX3_SEL_OP_LSR:
      return AGX3_ALU_LSR;
   case AGX3_SEL_OP_F_RCP:
      return AGX3_ALU_RCP;
   case AGX3_SEL_OP_F_RSQRT:
      return AGX3_ALU_RSQRT;
   case AGX3_SEL_OP_F_RINT:
      return AGX3_ALU_RINT;
   case AGX3_SEL_OP_I32_TO_F:
      return AGX3_ALU_I2F;
   case AGX3_SEL_OP_U32_TO_F:
      return AGX3_ALU_U2F;
   case AGX3_SEL_OP_F_TO_I32:
      return AGX3_ALU_F2I;
   case AGX3_SEL_OP_BITOP:
      switch (S->truth_table & 0xfu) {
      case AGX3_SEL_TT_AND:
         return AGX3_ALU_BAND;
      case AGX3_SEL_TT_OR:
         return AGX3_ALU_BOR;
      case AGX3_SEL_TT_XOR:
         return AGX3_ALU_BXOR;
      case AGX3_SEL_TT_NOT:
         return AGX3_ALU_BNOT;
      default:
         return AGX3_ALU_NUM;
      }
   default:
      return AGX3_ALU_NUM; /* F_TO_U32, FMIN/FMAX: no wait-carrying form yet */
   }
}

/* Emit one selected ALU instruction, carrying a wait mask when needed. With no
 * wait it uses the compact hand packer; with a wait it uses the composer form,
 * which is byte-identical for wait 0. */
static bool
agx3_emit_alu(struct util_dynarray *emission, enum agx_isa_encoding enc,
              const struct agx3_sel_instr *S, uint16_t wait_mask, bool g18g)
{
   if (wait_mask == 0)
      return agx3_emit(emission, enc, S);

   const enum agx3_alu_op op = agx3_sel_to_alu(S);

   /*
    * Every wait-carrying form deposits its sources into register fields
    * (agx3_alu_table.h gives f_src0/f_src1/f_src2; only ldimm has an f_imm,
    * and it has no sources). agx3_sel_instr keeps an immediate's value in the
    * same `reg` slot a register index uses, so handing one to the composer
    * would encode the constant AS a register number: `iadd d, r0, #1` became
    * `iadd d, r0, r1`, reading whatever half 1 happened to hold. That was
    * silently wrong code, not a refusal.
    *
    * An instruction with an immediate source therefore skips the composer
    * entirely and takes the prelude path below, which retires the wait and
    * then emits the ordinary no-wait form through the select core -- where
    * the real immediate encodings live.
    */
   /* The composer takes register numbers only: a saturate, shift or source
    * modifier would be dropped. Before this check a 32-bit iadd_sat that
    * waited on a load composed as a plain iadd. */
   bool all_sources_registers = !S->saturate && !S->shift;
   /* NOT's second bitop source is an ignored immediate; the unary `inv` form
    * reads only the first (llama.cpp q3_K mat-vec: in-place NOT, wait 0xc). */
   const unsigned nr_read = op == AGX3_ALU_BNOT ? 1 : S->nr_srcs;
   for (unsigned i = 0; i < nr_read; ++i)
      all_sources_registers &= S->src[i].is_reg &&
                               S->src[i].size == AGX3_SEL_SIZE_32 &&
                               !S->src[i].abs && !S->src[i].neg;

   /* The composer's baseline is one Apple specimen, lifetime bits included.
    * Never let it end the lifetime of a source the shader still reads. */
   bool baseline_ends_live_source = false;
   for (unsigned i = 0; op != AGX3_ALU_NUM && i < S->nr_srcs; ++i)
      baseline_ends_live_source |=
         S->src[i].is_reg && S->src[i].live &&
         (g18g
           ? agx3_alu_form_ends_source_g18g
           : agx3_alu_form_ends_source)(enc, op, i);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   const uint16_t s0 = S->nr_srcs > 0 ? S->src[0].reg : 0;
   const uint16_t s1 = S->nr_srcs > 1 ? S->src[1].reg : 0;
   const uint16_t s2 = S->nr_srcs > 2 ? S->src[2].reg : 0;

   if (op != AGX3_ALU_NUM && all_sources_registers &&
       !baseline_ends_live_source &&
       S->has_dest && S->dest_size == AGX3_SEL_SIZE_32 &&
       agx3_alu_compose(enc, op, S->dest, s0, s1, s2, S->imm, wait_mask,
                        bytes, &len)) {
      return agx3_append(emission, bytes, len);
   }

   /* Some compact ALU forms cannot carry the needed scoreboard wait. Retire
    * it with a synchronous prelude, then emit the ordinary no-wait form.
    * A fresh destination can take an ldimm; an in-place accumulator instead
    * uses bit_and(x, x), which preserves every bit of its live source.
    */
   if (!S->has_dest || S->dest_size != AGX_SIZE_32)
      return false;

   bool overlaps_source = false;
   bool exact_in_place = false;
   for (unsigned i = 0; i < S->nr_srcs; ++i) {
      if (!S->src[i].is_reg)
         continue;
      const unsigned dest_halves = agx_size_align_16(S->dest_size);
      const unsigned halves = agx_size_align_16(S->src[i].size);
      if (S->dest < S->src[i].reg + halves &&
          S->src[i].reg < S->dest + dest_halves) {
         overlaps_source = true;
         exact_in_place |=
            S->src[i].reg == S->dest && S->src[i].size == AGX_SIZE_32;
      }
   }

   if (overlaps_source &&
       (!exact_in_place || (wait_mask & (wait_mask - 1)) != 0))
      return false;

   while (wait_mask != 0) {
      const uint16_t prewait = wait_mask & -wait_mask;
      uint8_t prelude[AGX3_ALU_MAX_LENGTH];
      uint8_t prelude_len = 0;
      const bool ok =
         exact_in_place
            ? agx3_band_carrier(enc, S->dest, prewait, prelude,
                                &prelude_len)
            : agx3_alu_compose(enc, AGX3_ALU_LDIMM, S->dest, S->dest, S->dest,
                               0, 0, prewait, prelude, &prelude_len);
      if (!ok || !agx3_append(emission, prelude, prelude_len))
         return false;
      wait_mask &= ~prewait;
   }

   /* The wait is retired; emit the plain form. Go through the select core
    * rather than the composer so that immediate operands use their own
    * encodings instead of being mistaken for register numbers. */
   return agx3_emit(emission, enc, S);
}

/* The compact register+minifloat ALU forms (agx3_alu_fadd_minifloat(),
 * agx3_alu_fadd_minifloat_mod() at ep1, agx3_alu_fmul_minifloat_narrow())
 * take only the wait masks {0x00, 0x01, 0x02, 0x03, 0x04, 0x08, 0x10, 0x20};
 * the long forms take all 64. A wait on two unrelated producers outside that
 * set is retired first through an identity on the instruction's destination
 * (a partial wait on r0 could race an unrelated producer of r0, one on the
 * destination could miss its WAW dependency). Encodable masks still take one
 * instruction. */
static bool
agx3_fuse_compact_minifloat_wait(struct util_dynarray *emission,
                                 enum agx_isa_encoding enc, const agx_instr *I,
                                 uint16_t *wait)
{
   const uint16_t mask = *wait;

   /* Already encodable: empty, a single slot, or the native pair. */
   if (mask == 3 || !(mask & (mask - 1)))
      return true;

   /* The destination can itself carry one of these pending generations.
    * Complete the full instruction wait before the identity touches it. */
   if (!agx3_retire_result_wait(emission, enc, I, mask))
      return false;

   *wait = 0;
   return true;
}

/* An operand wait covers the instruction's inputs and overwritten destination,
 * not an arbitrary r0. An r0 identity can race an unrelated load still
 * writing r0. Carry the exact dependency mask on one scalar operand,
 * preserving its bits and width. No scratch register or unrelated wait is
 * needed. */
static bool
agx3_retire_select_operand_wait(struct util_dynarray *emission,
                                enum agx_isa_encoding enc,
                                agx_index operand, uint16_t mask)
{
   const uint16_t src[] = {operand.value, operand.value};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const bool ok = operand.size == AGX_SIZE_16
      ? agx3_alu_bit_reg16(enc, AGX3_ALU_BAND, operand.value, src, hints,
                           mask, bytes, &len)
      : agx3_alu_bit_reg(enc, AGX3_ALU_BAND, operand.value, src, hints,
                         mask, bytes, &len);
   return ok && agx3_append(emission, bytes, len);
}

static bool
agx3_retire_select_waits(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t mask)
{
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      const agx_index operand = I->src[s];
      if (!agx3_src_is_register(operand) || operand.memory ||
          agx_channels(operand) != 1 ||
          (operand.size != AGX_SIZE_16 && operand.size != AGX_SIZE_32))
         continue;

      return agx3_retire_select_operand_wait(emission, enc, operand, mask);
   }

   return false;
}

/* Find the exact scalar operand carrying a live scoreboard slot, including
 * the previous destination value for a write-after-write dependency. Search
 * backward from each half of the operand to its latest physical-register
 * writer (the halves of a 32-bit operand may be written separately, as
 * constant-program input imports are); a synchronous overwrite terminates the
 * search for that half. */
static bool
agx3_find_select_wait_operand(const struct agx3_slot_instr *sl,
                              const struct agx3_slot_result *res,
                              unsigned consumer, const agx_instr *I,
                              uint8_t slot, agx_index *out)
{
   for (unsigned s = 0; s < I->nr_srcs + I->nr_dests; ++s) {
      const agx_index operand = s < I->nr_srcs ? I->src[s]
                                              : I->dest[s - I->nr_srcs];
      if (!agx3_src_is_register(operand) || operand.memory ||
          agx_channels(operand) != 1 ||
          (operand.size != AGX_SIZE_16 && operand.size != AGX_SIZE_32))
         continue;

      const unsigned halves = agx_index_size_16(operand);
      for (unsigned h = operand.value; h < operand.value + halves; ++h) {
         for (unsigned j = consumer; j-- > 0;) {
            bool writes = false;
            for (unsigned d = 0; d < sl[j].nr_dsts; ++d)
               writes |= sl[j].dst[d] == h;
            if (!writes)
               continue;

            if (res[j].slot == slot) {
               *out = operand;
               return true;
            }
            break;
         }
      }
   }

   return false;
}

/*
 * applegpu_g17p never gives one instruction more than one scoreboard wait.
 * Across 36 kernels whose consumers each read two of four or six loads in
 * crossing orders (short sub/addsat/asr, int*int, int+(short<<2),
 * int-short), Apple reuses a slot for loads consumed together and otherwise
 * orders the loads so each consumer names at most one newly pending slot.
 * A multi-slot ISEL wait was also unreliable on hardware. Retire every slot
 * but the highest through an identity on the exact operand that slot writes,
 * leaving the instruction the single wait Apple's forms carry.
 */
static bool
agx3_retire_extra_waits(struct util_dynarray *emission,
                        enum agx_isa_encoding enc,
                        const struct agx3_slot_instr *sl,
                        const struct agx3_slot_result *res,
                        unsigned consumer, const agx_instr *I,
                        uint16_t *wait)
{
   while (util_bitcount(*wait) > 1) {
      const uint16_t bit = *wait & (uint16_t)-(int16_t)*wait;
      agx_index operand;
      if (!agx3_find_select_wait_operand(sl, res, consumer, I,
                                         (uint8_t)ffs(bit), &operand) ||
          !agx3_retire_select_operand_wait(emission, enc, operand, bit))
         return false;
      *wait &= (uint16_t)~bit;
   }
   return true;
}

enum agx3_pack_result
agx3_pack_uniform_float_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
       I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
       I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
      const enum agx3_uniform_float_op op =
         I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ? AGX3_UNIFORM_FADD :
         I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ? AGX3_UNIFORM_FMUL :
                                               AGX3_UNIFORM_FMA;
      const unsigned count = op == AGX3_UNIFORM_FMA ? 3 : 2;
      if (!ctx->is_preamble || I->nr_dests || I->nr_srcs != count ||
          I->saturate || I->imm >= 128 || (I->imm & 1))
         AGX3_REFUSE_RET(I);
      struct agx3_uniform_float_src src[3] = {0};
      for (unsigned s = 0; s < count; ++s) {
         const agx_index index = I->src[s];
         const bool reg = index.type == AGX_INDEX_REGISTER;
         if ((!reg && index.type != AGX_INDEX_IMMEDIATE) ||
             index.value >= 256 ||
             index.memory || index.abs || index.neg ||
             (reg && (index.size != AGX_SIZE_32 ||
                      agx_channels(index) != 1)))
            AGX3_REFUSE_RET(I);
         src[s] = (struct agx3_uniform_float_src){
            .kind = reg ? AGX3_FLOAT_REGISTER : AGX3_FLOAT_IMMEDIATE,
            .value = index.value,
            .cache = index.cache,
            .discard = index.discard,
         };
      }
      uint8_t bytes[16], len;
      if (!agx3_uniform_float_store_encode(enc, op, I->imm, src,
                                            st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * `export` is a zero-byte pseudo-op, exactly as on AGX2
    * (agx_pack.c:1242).  Register allocation has already turned it into
    * a parallel copy into its ABI register and rewritten the source to
    * that register (agx_register_allocate.c:1296-1310), so there is
    * nothing left to encode here. The slot pass must not treat this
    * pseudo-op as a hardware read; a no-stop part drains outstanding
    * results before handing its ABI registers to a separate epilog.
    */
   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_unary_and_imad(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_SRSQRT) {
      if (enc == AGX_ISA_ENCODING_AGX3C && st->wait) {
         if (!agx3_retire_result_wait(emission, enc, I, st->wait))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~st->wait;
         st->wait = 0;
      }
      const agx_index dest=I->dest[0], src=I->src[0];
      const bool half = src.size == AGX_SIZE_16;
      /* f30 carries the source's absolute value, so `sqrt(abs(x))` --
       * the standard safe-sqrt idiom -- encodes without a separate
       * move. Only the 32-bit form is witnessed; the half step keeps
       * its narrower grounded domain. */
      if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
          !agx3_vary_reg32(dest) ||
          !agx3_src_is_register(src) || src.memory ||
          agx_channels(dest) != 1 || agx_channels(src) != 1 ||
          (src.neg && half) || src.kill || (src.abs && half) ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      /* The half-input leaf requires a retained source. Cache hints
       * are advisory, so retain even a last-use half after legalization
       * has widened its other reads for the following multiply. */
      const enum agx3_alu_hint half_hint =
         ctx->key->dev.chip == AGX_CHIP_G17P
            ? AGX3_ALU_HINT_CACHE : agx3_post_ra_hint(src);
      const bool ok =
         half ? agx3_alu_sqrt_step_half(enc, dest.value, src.value,
                                        half_hint,
                                        enc == AGX_ISA_ENCODING_AGX3C ? dest.cache : true,
                                        st->wait, bytes, &len)
              : (src.size == AGX_SIZE_32 && !(src.value & 1) &&
                 src.value < 256 &&
                 agx3_alu_sqrt_step_source_modifiers(enc, dest.value, src.value,
                                        agx3_post_ra_hint(src),
                                        dest.cache, st->wait, src.abs, src.neg,
                                        bytes, &len));
      if (!ok || !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Reciprocal widths are independent. In particular, mediump texture
    * data commonly feeds a full-precision divide in GTK's colour
    * conversion shaders. f10/f26 select the result/source widths, while
    * f30/f31 carry source ABS/NEG without a separate move. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_RCP &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest=I->dest[0], src=I->src[0];
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          (src.size != AGX_SIZE_16 && src.size != AGX_SIZE_32) ||
          agx_channels(src) != 1 || src.kill ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      if (!agx3_alu_rcp(
             enc, dest.value, src.value, dest.size == AGX_SIZE_16,
             src.size == AGX_SIZE_16, src.abs, src.neg,
             agx3_post_ra_hint(src), dest.cache, st->wait, bytes, &len) ||
          !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Reciprocal square root, at either width. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_RSQRT &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      /* f30/f31 carry the source's ABS/NEG. inversesqrt(abs(x)) is the
       * safe form of the idiom and had no encoding at all. The half
       * form has no modifier witness, so it stays width-matched and
       * unmodified. */
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != dest.size || agx_channels(src) != 1 ||
          src.kill || (half && (src.abs || src.neg)) ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool ok =
         half ? agx3_alu_hrsqrt(enc, dest.value, src.value,
                                agx3_post_ra_hint(src), dest.cache,
                                st->wait, bytes, &len)
              : agx3_alu_f32rsqrt_mod(enc, dest.value, src.value,
                                      agx3_post_ra_hint(src),
                                      dest.cache, st->wait, src.neg,
                                      src.abs, bytes, &len);
      if (!ok || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* C rounding/log/exp share native scalar width, lifetime and SAT
    * fields. Keep B's existing per-operation admission below. */
   const int c_unary = I->op == AGX_OPCODE_ROUNDEVEN ? AGX3_FUNOP_RINT :
                                                    agx3_map_funop(I->op);
   if (enc == AGX_ISA_ENCODING_AGX3C &&
       c_unary >= AGX3_FUNOP_FLOOR && c_unary <= AGX3_FUNOP_EXP2) {
      if (!agx3_emit_funop(emission, enc, I, c_unary, st->wait,
                           agx_is_g18g(&ctx->key->dev)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* f30 and f31 are the source's absolute-value and negate modifiers
    * (NIR folds a negation into the source, e.g. exp(-x)). The half
    * form has a negate bit too. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_EXP2 &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != dest.size || agx_channels(src) != 1 ||
          (half && src.abs) || src.kill ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!(half ? agx3_alu_hexp2_neg(enc, dest.value, src.value,
                                      agx3_post_ra_hint(src), dest.cache,
                                      st->wait, src.neg, bytes, &len)
                 : agx3_alu_f32exp2_mod(enc, dest.value, src.value,
                                        agx3_post_ra_hint(src),
                                        dest.cache, st->wait, src.neg,
                                        src.abs, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* f10 and f26 independently select scalar result/source width. This
    * matters for mediump texture results promoted before logarithmic
    * transfer functions, as in GTK's component-transfer shader. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_LOG2 &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest = I->dest[0], src = I->src[0];
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          (src.size != AGX_SIZE_16 && src.size != AGX_SIZE_32) ||
          agx_channels(src) != 1 || src.kill ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_log2(
             enc, dest.value, src.value, dest.size == AGX_SIZE_16,
             src.size == AGX_SIZE_16, src.abs, src.neg,
             agx3_post_ra_hint(src), dest.cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* floor() and ceil() at either width: the 32-bit form is the half
    * one with the width selectors raised, with the same source
    * modifiers. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_FLOOR &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != dest.size || agx_channels(src) != 1 ||
          (half && (src.abs || src.neg)) || src.kill ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!(half ? agx3_alu_hfloor(enc, dest.value, src.value,
                                   agx3_post_ra_hint(src),
                                   dest.cache, st->wait, bytes, &len)
                 : agx3_alu_f32floor(enc, dest.value, src.value,
                                     agx3_post_ra_hint(src), dest.cache,
                                     st->wait, src.neg,
                                     src.abs, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_CEIL &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != dest.size || agx_channels(src) != 1 ||
          (half && (src.abs || src.neg)) || src.kill ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!(half ? agx3_alu_hceil(enc, dest.value, src.value,
                                  agx3_post_ra_hint(src), dest.cache,
                                  st->wait, bytes, &len)
                 : agx3_alu_f32ceil(enc, dest.value, src.value,
                                    agx3_post_ra_hint(src), dest.cache,
                                    st->wait, src.neg, src.abs, bytes,
                                    &len)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B &&
       (I->op == AGX_OPCODE_TRUNC ||
        I->op == AGX_OPCODE_ROUNDEVEN) &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_16 &&
       agx_channels(I->src[0]) == 1 && !I->src[0].abs &&
       !I->src[0].neg && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard)) {
      const agx_index dest = I->dest[0], src = I->src[0];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool ok = I->op == AGX_OPCODE_TRUNC
         ? agx3_alu_htrunc(enc, dest.value, src.value,
                           agx3_post_ra_hint(src), dest.cache, st->wait,
                           bytes, &len)
         : agx3_alu_hroundeven(enc, dest.value, src.value,
                               agx3_post_ra_hint(src), dest.cache, st->wait,
                               bytes, &len);
      if (!ok || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A 32-bit roundeven had no handler here and fell through to the
    * select core, whose f32rint encoding point holds its source in a
    * six-bit field and therefore refuses any source above r63. That is
    * not a rare shape: hk lowers every float texture result through a
    * scaled-view recovery whose integer fetch ends in fround_even, and
    * one unrolled loop of texelFetch() puts those sources well past
    * r63. A refusal is fatal -- the shader does not compile and
    * vkCreateComputePipelines fails -- so it cost 106 robustness2
    * cases. Use the f32rint form family that floor and ceil already
    * use, which carries the family's full register fields, its f30/f31
    * source modifiers and its wait slots. Shapes this does not claim
    * keep their existing route. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_ROUNDEVEN &&
       I->nr_dests == 1 && I->nr_srcs == 1 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == 1 && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard)) {
      const agx_index dest = I->dest[0], src = I->src[0];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_f32roundeven(enc, dest.value, src.value,
                                 agx3_post_ra_hint(src), dest.cache,
                                 st->wait, src.neg, src.abs, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_BITTEST_SELECT_AGX3) {
      struct agx3_bittest_desc d = {
         .dest = I->dest[0].value,
         .mask = I->imm,
         .wait_mask = st->wait,
      };
      if (I->nr_dests != 1 || I->nr_srcs != 3 ||
          !agx3_src_is_register(I->dest[0]) ||
          I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
          I->imm > UINT8_MAX || I->src[0].abs || I->src[0].neg)
         AGX3_REFUSE_RET(I);
      for (unsigned s = 0; s < 3; ++s) {
         if (!agx3_src_is_register(I->src[s]) ||
             I->src[s].size != AGX_SIZE_32 || I->src[s].memory ||
             I->src[s].abs)
            AGX3_REFUSE_RET(I);
         d.src[s] = I->src[s].value;
         d.cache[s] = I->src[s].cache;
         d.discard[s] = I->src[s].discard;
         if (s)
            d.negate[s - 1] = I->src[s].neg;
      }
      uint8_t bytes[AGX3_ISEL_MAX_LENGTH], len = 0;
      if (!agx3_bittest_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Transcendental / rounding / gradient funops. The gradients need
    * the 2x2 quad, and every witness is a fragment shader. */
   const int fop = agx_is_g18g(&ctx->key->dev)
                      ? agx3_map_funop_g18g(I->op)
                      : agx3_map_funop(I->op);
   if (fop >= 0) {
      uint16_t funop_wait = st->wait;
      /* Apple gradient witnesses expose no wait-field map. A real
       * desktop shader may nevertheless reach DFDX/DFDY with an
       * outstanding interpolation/coefficient dependency. Retire it
       * through the already-grounded scoreboard no-op, then preserve
       * the byte-exact zero-wait gradient encoding. The same holds for
       * any funop whose witnessed form has no wait fields (f32sinc, for
       * instance, scheduled right behind an async producer). */
      if ((fop == AGX3_FUNOP_DFDX || fop == AGX3_FUNOP_DFDY ||
           !agx3_funop_carries_wait(enc, (enum agx3_funop)fop)) &&
          funop_wait) {
         if (!agx3_retire_result_wait(emission, enc, I, funop_wait))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~funop_wait;
         funop_wait = 0;
      }
      if (((fop == AGX3_FUNOP_DFDX || fop == AGX3_FUNOP_DFDY) &&
           ctx->stage != MESA_SHADER_FRAGMENT) ||
          !agx3_emit_funop(emission, enc, I, (enum agx3_funop)fop,
                           funop_wait, agx_is_g18g(&ctx->key->dev)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * A zero immediate has its own instruction, `clr`, at both widths.
    * Apple emits it wherever a register is set to zero, and it is the
    * compact grounded way to materialize a 16-bit zero.
    */
   if (I->op == AGX_OPCODE_MOV_IMM && I->imm == 0 && st->wait == 0 &&
       I->nr_dests == 1 && I->nr_srcs == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32)) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
      uint8_t len = 0;
      if (agx3_clr(enc, I->dest[0].value,
                   I->dest[0].size == AGX_SIZE_16, false, bytes, &len)) {
         if (!agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      /* Fall through to ldimm when clr cannot express it. */
   }

   /*
    * Uniform -> GPR.  AGX3 ALU operands are register-or-immediate, so
    * agx_lower_uniform_sources() materializes every generic-ALU uniform
    * source as a move; agx_lower_pseudo() then turns that move into a
    * bitop whose source is still a uniform.  Apple moves a uniform into
    * a register with `copy` -- 49 of the 100 G16G corpus copies do
    * exactly this -- so encode that rather than refusing the bitop.
    * Without it no shader that reads a pushed uniform in an ALU op can
    * be compiled at all.
    */
   /*
    * `imad dst32, srcA, #imm, srcC` -- a constant multiplier, as in
    * every index-derived address (`gid * stride + base`); the
    * multiplier goes in the immediate field, not a register field.
    */
   const int uniform_integer = agx3_try_uniform_integer(emission, enc, I, st->wait);
   if (uniform_integer < 0)
      AGX3_REFUSE_RET(I);
   if (uniform_integer > 0)
      return AGX3_PACK_EMITTED;

   /*
    * A multiply-add whose multiplier and addend are both the constant
    * zero is zero, whatever the other factor: x * 0 + 0. The backend
    * produces it when immediate inlining lands a zero in a
    * umul_2x32_64 NIR could not fold -- soft-fp64's mantissa products
    * (gpu_shader_fp64.builtin) are `imad d64, x.abs, #0, #0`. No Apple
    * form multiplies by an immediate into 64 bits (G17G's imadd64 is
    * witnessed with register multipliers only), and Apple would never
    * multiply by zero: it zeroes a register with `clr`, and a 64-bit
    * pair with two 32-bit ones -- applegpu_g17g initializes a ulong
    * accumulator as `clr r2; clr r3` (g18ref-out/probe/m6-alu-misc
    * zero64_loop, m6-alu-sel2 zero64_loop2), and applegpu_g18g's
    * 32-bit clr is 23 05 00 00 00 00 (test/native/g18g
    * m5_read_array_uniform), with the f11 cache form in
    * m5-control-flow-reference. clr carries no wait, so a pending
    * producer of this destination retires first.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_IMAD &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->shift == 0 &&
       !I->saturate && agx3_src_is_register(I->dest[0]) &&
       !I->dest[0].memory && agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_32 ||
        I->dest[0].size == AGX_SIZE_64) &&
       !(I->dest[0].value & 1) &&
       I->dest[0].value +
             (I->dest[0].size == AGX_SIZE_64 ? 4u : 2u) <= 256 &&
       ((agx3_plain_immediate(I->src[0]) && I->src[0].value == 0) ||
        (agx3_plain_immediate(I->src[1]) && I->src[1].value == 0)) &&
       agx3_plain_immediate(I->src[2]) && I->src[2].value == 0 &&
       !I->src[2].neg) {
      uint8_t lo[AGX3_STAGE_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
      uint8_t lo_len = 0, hi_len = 0;
      const bool wide = I->dest[0].size == AGX_SIZE_64;
      if ((st->wait && !agx3_retire_slots_noop(emission, enc, st->wait)) ||
          !agx3_clr(enc, I->dest[0].value, false, I->dest[0].cache, lo,
                    &lo_len) ||
          (wide && !agx3_clr(enc, I->dest[0].value + 2, false,
                             I->dest[0].cache, hi, &hi_len)) ||
          !agx3_append(emission, lo, lo_len) ||
          (wide && !agx3_append(emission, hi, hi_len)))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~st->wait;
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs >= 3 &&
       I->shift == 0 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) &&
       !I->dest[0].memory &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32 ||
        I->dest[0].size == AGX_SIZE_64) &&
       agx_channels(I->dest[0]) == 1) {
      /* Multiplication commutes: NIR may place an immediate first. */
      const bool constant_first = I->src[0].type == AGX_INDEX_IMMEDIATE &&
                                  I->src[1].type == AGX_INDEX_REGISTER;
      agx_index a = I->src[constant_first ? 1 : 0];
      agx_index m = I->src[constant_first ? 0 : 1];
      const agx_index c = I->src[2];

      /* Multiplication commutes. Apple keeps the 32-bit operand of a
       * mixed-width 32-bit product in slot A and the half in slot B. */
      if ((I->dest[0].size == AGX_SIZE_32 ||
           (enc == AGX_ISA_ENCODING_AGX3C && I->dest[0].size == AGX_SIZE_64)) &&
          a.size == AGX_SIZE_16 &&
          agx3_src_is_register(m) && m.size == AGX_SIZE_32) {
         const agx_index half = a;
         a = m;
         m = half;
      }

      /*
       * 32x32 -> 64. This is how `umul_high`, `imul_high` and NIR's
       * `*mul_2x32_64` all reach the backend, and therefore how every
       * integer division by a non-constant does. The source-size
       * convention is the one agx_pack.c already uses for AGX2: an
       * operand narrower than the result is zero-extended when it
       * carries `abs` and sign-extended otherwise, which maps onto the
       * per-slot sign-extend bits f28/f43/f58.
       */
      const bool narrow = I->dest[0].size == AGX_SIZE_16;
      const bool wide = I->dest[0].size == AGX_SIZE_64;
      const bool widening = !narrow && !wide;

      /* The multiplier is either an 8-bit immediate or a register of
       * the result width, plus the paired 16x16->32 widening form.
       * Both 16-bit and 32-bit forms are witnessed byte-exactly. */
      const bool mult_imm = m.type == AGX_INDEX_IMMEDIATE && !m.abs &&
                            !m.neg && m.value <= 0xff;
      const bool widening_half_multiplier =
         (widening || (wide && enc == AGX_ISA_ENCODING_AGX3C)) &&
         m.size == AGX_SIZE_16 && !m.neg;
      const bool mult_reg =
         agx3_src_is_register(m) && !m.memory &&
         ((m.size == (narrow ? AGX_SIZE_16 : AGX_SIZE_32) &&
           (wide || !m.abs) && !m.neg) ||
          widening_half_multiplier) &&
         agx_channels(m) == 1;

      /* An immediate-zero addend is the pure-multiply form, whose
       * addend operand is absent rather than zero. Native constant
       * and register multiplier variants have distinct baselines. */
      const bool no_addend = c.type == AGX_INDEX_IMMEDIATE &&
                             c.value == 0 && !c.abs && !c.neg;
      /*
       * `.neg` on an IMAD addend is an arithmetic negate, not a source
       * modifier: agx_pack.c gives it a dedicated bit and permits it on
       * src[2] alone.  AGX3 spells the negated form as its own
       * instruction, `imsub`, so admit the modifier here and let
       * agx3_imad_encode() select the opcode, so `a*b - imm` encodes.
       */
      const bool addend_imm =
         c.type == AGX_INDEX_IMMEDIATE && c.value != 0 &&
         c.value <= 0xff && !c.abs && !c.memory;
      /*
       * A bare narrow addend is sign-extended (agx_optimizer_signext()
       * fuses SIGNEXT into the multiply-add by dropping it): f54 is
       * slot C's width (1 = 16-bit, 2 = 32-bit), f58 its sign extension
       * (set for short, clear for ushort/ABS).
       */
      const bool addend_ok =
         no_addend || addend_imm ||
         (agx3_src_is_register(c) && !c.memory &&
          agx_channels(c) == 1 && !(c.abs && c.neg) &&
          ((!narrow && c.size == AGX_SIZE_32) ||
           (wide && c.size == AGX_SIZE_64) ||
           (c.size == AGX_SIZE_16 &&
             (narrow ? (!c.abs && !c.neg) : (c.abs || c.neg)))));
      const bool addend_reg = !no_addend && !addend_imm;

      const bool widening_half_a =
         widening && a.size == AGX_SIZE_16 && !a.neg;
      const bool a_ok = agx3_src_is_register(a) && !a.memory &&
         agx_channels(a) == 1 &&
         (narrow ? (a.size == AGX_SIZE_16 && !a.abs && !a.neg)
                 : (((wide || !a.abs) && !a.neg &&
                     a.size == AGX_SIZE_32) || widening_half_a));
      /* Each half operand of a 32-bit product widens by its own sign.
       * Apple pairs a half register multiplier only with no addend or
       * a 32-bit register addend, and a half multiplicand times a
       * constant with any register addend. */
      const bool half_a = widening && a.size == AGX_SIZE_16;
      const bool half_m = widening && mult_reg && m.size == AGX_SIZE_16;
      const bool half_operands_ok =
         (!half_a && !half_m) ||
         (!addend_imm &&
          (no_addend || c.size == AGX_SIZE_32 ||
           (!half_m && c.size == AGX_SIZE_16) ||
           (enc == AGX_ISA_ENCODING_AGX3C && c.size == AGX_SIZE_16 && c.abs)));

      if (a_ok && (mult_imm || mult_reg) && addend_ok &&
          half_operands_ok) {
         const struct agx3_imad_desc d = {
            .dest = I->dest[0].value,
            .dest_16 = narrow,
            .dest_64 = wide,
            .dest_cache = I->dest[0].cache,
            .src_a = a.value,
            .src_a_16 = a.size == AGX_SIZE_16,
            .src_a_sign_extend = (wide || half_a) && !a.abs,
            .multiplier = mult_imm ? (uint8_t)m.value : 0,
            .multiplier_is_register = mult_reg,
            .multiplier_reg = mult_reg ? m.value : 0,
            .multiplier_16 = mult_reg && m.size == AGX_SIZE_16,
            .multiplier_kill = mult_reg && m.discard,
            .multiplier_cache = mult_reg && m.cache,
            .multiplier_sign_extend =
               (wide || half_m) && mult_reg && !m.abs,
            .src_a_kill = a.discard,
            .src_a_cache = a.cache,
            .src_c_kill = addend_reg && c.discard,
            .src_c_cache = addend_reg && c.cache,
            .no_addend = no_addend && !(wide && enc == AGX_ISA_ENCODING_AGX3C),
            .addend_is_immediate = addend_imm ||
               (wide && enc == AGX_ISA_ENCODING_AGX3C && no_addend),
            .addend_immediate = addend_imm ? c.value : 0,
            .src_c = addend_reg ? c.value : 0,
            .src_c_16 = addend_reg && c.size == AGX_SIZE_16,
            .src_c_64 = addend_reg && c.size == AGX_SIZE_64,
            /* Bare narrow addend: sign-extend unless ABS marks it
             * unsigned. Only for a widening result: a 16-bit result
             * widens nothing and the encoder refuses the bit there. */
            .src_c_sign_extend =
               addend_reg && !narrow &&
               (wide ? (c.size != AGX_SIZE_64 && !c.abs)
                     : (c.size == AGX_SIZE_16 && !c.abs)),
            /* A negated addend (register or immediate) selects Apple's
             * `imsub`. */
            .negate_addend = !no_addend && c.neg,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if ((agx_is_g18g(&ctx->key->dev)
               ? agx3_imad_encode_g18g
               : agx3_imad_encode)(enc, &d, bytes, &len) &&
             agx3_append(emission, bytes, len))
            return AGX3_PACK_EMITTED;
      }
   }

   /* A uniform first comparand (agx3_m5_fcmpsel_uniform_cmp0). */
   unsigned m5_fsel_cond;
   bool m5_fsel_swap;
   if (enc == AGX_ISA_ENCODING_AGX3C &&
       agx3_m5_fcmpsel_uniform_cmp0(I, &m5_fsel_cond, &m5_fsel_swap)) {
      const agx_index cmp1 = I->src[1];
      const agx_index sel1 = I->src[m5_fsel_swap ? 2 : 3];
      if (!agx3_vary_reg32(I->dest[0]) || !agx3_vary_reg32(cmp1) ||
          !agx3_vary_reg32(sel1) || cmp1.kill || sel1.kill ||
          (cmp1.cache && cmp1.discard) || (sel1.cache && sel1.discard))
         AGX3_REFUSE_RET(I);
      const struct agx3_fsel32_gpu1_desc d = {
         .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
         .cmp0 = I->src[0].value, .cmp0_uniform = true,
         .cmp1 = cmp1.value, .cmp1_hint = agx3_post_ra_hint(cmp1),
         .sel_immediate = {true, false},
         .sel = {0, sel1.value},
         .sel_hint = {AGX3_ALU_HINT_NONE, agx3_post_ra_hint(sel1)},
         .condition = m5_fsel_cond, .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_select32_gpu1(&d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_fcmpsel(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   /* Keep the uniform half of a native tied min/max in the uniform file.
    * Canonicalize just as the register/minifloat path below does.
    * G18G keeps its own compare-select forms below. */
   if (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       !I->fcmpsel_minifloat_select &&
       (agx3_vary_reg32(I->dest[0]) || agx3_half_reg(I->dest[0])) &&
       agx_channels(I->dest[0]) == 1 &&
       (((agx3_vary_reg32(I->src[0]) || agx3_half_reg(I->src[0])) &&
         (agx3_vary_reg32(I->src[1]) || agx3_half_reg(I->src[1]) ||
          (I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value <= 255))) ||
        (I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value <= 255 &&
         (agx3_vary_reg32(I->src[1]) || agx3_half_reg(I->src[1])))) &&
       ((I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= UINT8_MAX) ||
        (agx3_half_reg(I->dest[0]) && agx3_half_reg(I->src[2]))) &&
       ((I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= UINT8_MAX) ||
        (agx3_half_reg(I->dest[0]) && agx3_half_reg(I->src[3])))) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index cmp[2] = {I->src[reverse ? 1 : 0],
                                I->src[reverse ? 0 : 1]};
      unsigned condition = I->fcond;
      bool swap = I->invert_cond;
      if (!agx3_fcmpsel_tied_minmax(I)) {
         if (condition == AGX_FCOND_LTN) {
            condition = AGX_FCOND_GE;
            swap = !swap;
         } else if (condition == AGX_FCOND_GTN) {
            condition = AGX_FCOND_LE;
            swap = !swap;
         }
      }
      if (reverse)
         condition = agx3_fcond_reverse(condition);
      for (unsigned j = 0; j < 4; ++j) {
         if (I->src[j].abs || I->src[j].neg || I->src[j].memory ||
             agx_channels(I->src[j]) != 1 ||
             (I->src[j].cache && I->src[j].discard))
            AGX3_REFUSE_RET(I);
      }
      const struct agx3_fsel32_gpu1_desc d = {
         .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
         .cmp0 = cmp[0].value, .cmp1 = cmp[1].value,
         .cmp0_16 = cmp[0].size == AGX_SIZE_16,
         .cmp1_16 = cmp[1].type != AGX_INDEX_IMMEDIATE &&
                    cmp[1].size == AGX_SIZE_16,
         .cmp1_immediate = cmp[1].type == AGX_INDEX_IMMEDIATE,
         .cmp0_hint = agx3_post_ra_hint(cmp[0]),
         .cmp1_hint = agx3_post_ra_hint(cmp[1]),
         .sel_immediate = {
            I->src[swap ? 3 : 2].type == AGX_INDEX_IMMEDIATE,
            I->src[swap ? 2 : 3].type == AGX_INDEX_IMMEDIATE,
         },
         .sel_integer = true,
         .sel = {I->src[swap ? 3 : 2].value, I->src[swap ? 2 : 3].value},
         .sel_hint = {agx3_post_ra_hint(I->src[swap ? 3 : 2]),
                      agx3_post_ra_hint(I->src[swap ? 2 : 3])},
         .condition = condition, .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool encoded = I->dest[0].size == AGX_SIZE_32
         ? agx3_alu_fcmp_select32_gpu1(&d, bytes, &len)
         : agx3_alu_fcmp_select16_gpu1(&d, bytes, &len);
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       ((agx3_vary_reg32(I->src[0]) || agx3_half_reg(I->src[0])) ||
        agx3_src_is_zero_imm(I->src[0])) &&
       ((agx3_vary_reg32(I->src[1]) || agx3_half_reg(I->src[1])) ||
        agx3_src_is_zero_imm(I->src[1])) &&
       (agx3_vary_reg32(I->src[2]) ||
        (I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= UINT8_MAX)) &&
       (agx3_vary_reg32(I->src[3]) ||
        (I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= UINT8_MAX)) &&
       !(I->src[2].type == AGX_INDEX_IMMEDIATE &&
         I->src[3].type == AGX_INDEX_IMMEDIATE)) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index cmp[2] = {I->src[reverse ? 1 : 0], I->src[reverse ? 0 : 1]};
      unsigned condition = I->fcond;
      bool swap = I->invert_cond;
      if (!agx3_fcmpsel_tied_minmax(I)) {
         if (condition == AGX_FCOND_LTN) {
            condition = AGX_FCOND_GE;
            swap = !swap;
         } else if (condition == AGX_FCOND_GTN) {
            condition = AGX_FCOND_LE;
            swap = !swap;
         }
      }
      if (reverse)
         condition = agx3_fcond_reverse(condition);
      const agx_index sel[2] = {I->src[swap ? 3 : 2], I->src[swap ? 2 : 3]};
      /* Generic select immediates are integer bit patterns. Only a
       * selected operand tied to a compared float carries the
       * comparison's minifloat encoding (for example a depth clamp).
       * A fused boolean bit-mask select must retain its raw bits. */
      const bool tied_float =
         (agx_is_equiv(I->src[0], I->src[2]) &&
          agx_is_equiv(I->src[1], I->src[3])) ||
         (agx_is_equiv(I->src[0], I->src[3]) &&
          agx_is_equiv(I->src[1], I->src[2]));
      for (unsigned i = 0; i < 4; ++i) {
         if (I->src[i].kill || agx_channels(I->src[i]) != 1 ||
             (I->src[i].cache && I->src[i].discard))
            AGX3_REFUSE_RET(I);
      }
      const struct agx3_fsel32_gpu1_desc d = {
         .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
         .cmp0 = cmp[0].value, .cmp1 = cmp[1].value,
         .cmp0_16 = cmp[0].size == AGX_SIZE_16,
         .cmp1_16 = cmp[1].type != AGX_INDEX_IMMEDIATE && cmp[1].size == AGX_SIZE_16,
         .cmp1_immediate = cmp[1].type == AGX_INDEX_IMMEDIATE,
         .cmp0_hint = agx3_post_ra_hint(cmp[0]),
         .cmp1_hint = agx3_post_ra_hint(cmp[1]),
         .sel_immediate = {sel[0].type == AGX_INDEX_IMMEDIATE,
                           sel[1].type == AGX_INDEX_IMMEDIATE},
         .sel_integer = !tied_float && !I->fcmpsel_minifloat_select,
         .sel = {sel[0].value, sel[1].value},
         .sel_hint = {agx3_post_ra_hint(sel[0]), agx3_post_ra_hint(sel[1])},
         .condition = condition, .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_select32_gpu1(&d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
       !I->saturate &&
       (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
       (I->src[0].type == AGX_INDEX_UNIFORM || I->src[1].type == AGX_INDEX_UNIFORM)) {
      const unsigned u = I->src[0].type == AGX_INDEX_UNIFORM ? 0 : 1;
      const agx_index uniform = I->src[u], reg = I->src[u ^ 1];
      const bool half = I->dest[0].size == AGX_SIZE_16;
      if (!agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          agx_channels(I->dest[0]) != 1 ||
          (!half && I->dest[0].size != AGX_SIZE_32) ||
          !agx3_src_is_register(reg) ||
          !agx_is_equiv(I->src[0], I->src[2]) ||
          !agx_is_equiv(I->src[1], I->src[3]))
         AGX3_REFUSE_RET(I);
      for (unsigned i = 0; i < 4; ++i) {
         if (I->src[i].size != I->dest[0].size ||
             agx_channels(I->src[i]) != 1 || I->src[i].memory ||
             I->src[i].abs || I->src[i].neg || I->src[i].kill ||
             I->src[i].cache != I->src[i ^ 2].cache ||
             I->src[i].discard != I->src[i ^ 2].discard)
            AGX3_REFUSE_RET(I);
      }
      if (uniform.cache || uniform.discard)
         AGX3_REFUSE_RET(I);
      uint16_t fsel_wait = st->wait;
      if (fsel_wait & (fsel_wait - 1)) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      uint8_t bytes[16], length;
      if (!agx3_uniform_minmax_encode(enc, I->dest[0].value, reg.value,
              uniform.value, half, I->fcond == AGX_FCOND_GTN,
              reg.cache, reg.discard, I->dest[0].cache, fsel_wait, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native terrain fmax(a, fabs(b)) carries ABS on both comparison
    * and selected-value slots. This is not a general select modifier.
    */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       ctx->stage == MESA_SHADER_VERTEX && !ctx->is_preamble &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
       !I->saturate && I->fcond == AGX_FCOND_GTN &&
       I->src[1].abs && I->src[3].abs) {
      bool valid = agx3_src_is_register(I->dest[0]) &&
                   I->dest[0].size == AGX_SIZE_32 &&
                   !I->dest[0].memory && agx_channels(I->dest[0]) == 1;
      for (unsigned i = 0; i < 4; ++i) {
         const agx_index x = I->src[i];
         valid &= agx3_src_is_register(x) && x.size == AGX_SIZE_32 &&
                  agx_channels(x) == 1 && !x.memory && !x.kill &&
                  !x.neg && x.abs == (i % 2 == 1) &&
                  !(x.cache && x.discard) &&
                  agx_is_equiv(x, I->src[i ^ 2]) &&
                  agx3_post_ra_hint(x) == agx3_post_ra_hint(I->src[i ^ 2]);
      }
      if (valid) {
         if (st->wait && !agx3_retire_select_waits(emission, enc, I, st->wait))
            AGX3_REFUSE_RET(I);
         const uint16_t src[] = {I->src[0].value, I->src[1].value};
         const enum agx3_alu_hint hints[] = {
            agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_fmax_abs_f32(enc, I->dest[0].value, src, hints,
                                  I->dest[0].cache, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * Native tied 32-bit comparison/selection is fmin/fmax.  This is the
    * shape NIR hands us for max(x, 0.0) -- normalize(), clamp(),
    * saturate() and the discard compare -- and Apple encodes it as a
    * compact six-byte fsel, taking the register form when the second
    * operand is a register and the minifloat form when it is a
    * constant.  Both were witnessed byte-exactly; see agx3_alu_core.h.
    */
   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d) &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
       (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
       !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
       I->dest[0].value < 256) {
      /*
       * A minimum and a maximum are commutative -- NIR declares
       * fmin_agx and fmax_agx 2src_commutative -- and a tied GTN/LTN
       * fcmpsel can only have come from one of them, since every other
       * condition a comparison produces is ordered. So the constant may
       * be written on either side, and max(0.0, x) normalizes to the
       * max(x, 0.0) Apple encodes.
       */
      const unsigned reg_pair =
         (agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
          I->src[0].size == AGX_SIZE_32) ? 0 : 1;
      const agx_index reg_a = I->src[reg_pair];
      const agx_index reg_b = I->src[reg_pair + 2];
      const agx_index other_a = I->src[reg_pair ^ 1];
      const agx_index other_b = I->src[(reg_pair ^ 1) + 2];

      const bool tied_reg =
         agx3_src_is_register(reg_a) && agx3_src_is_register(reg_b) &&
         !reg_a.memory && !reg_b.memory &&
         reg_a.size == AGX_SIZE_32 && reg_b.size == AGX_SIZE_32 &&
         agx_channels(reg_a) == 1 && agx_channels(reg_b) == 1 &&
         reg_a.value == reg_b.value && reg_a.value < 256 &&
         agx3_post_ra_hint(reg_a) == agx3_post_ra_hint(reg_b);
      const bool second_imm =
         other_a.type == AGX_INDEX_IMMEDIATE &&
         other_b.type == AGX_INDEX_IMMEDIATE &&
         other_a.value == other_b.value && other_a.value <= 255;
      const bool second_reg =
         !second_imm && agx3_src_is_register(other_a) &&
         agx3_src_is_register(other_b) && !other_a.memory &&
         !other_b.memory && other_a.size == AGX_SIZE_32 &&
         other_b.size == AGX_SIZE_32 &&
         agx_channels(other_a) == 1 && agx_channels(other_b) == 1 &&
         other_a.value == other_b.value && other_a.value < 256 &&
         agx3_post_ra_hint(other_a) == agx3_post_ra_hint(other_b);
      bool clean = tied_reg;
      for (unsigned i = 0; i < 4; ++i) {
         clean &= !I->src[i].abs && !I->src[i].neg && !I->src[i].kill &&
                  !(I->src[i].cache && I->src[i].discard);
      }

      if ((second_imm || second_reg) && clean) {
         /* This form carries one scoreboard slot; retire anything wider
          * first, exactly as the other fsel paths do. The G17G-family
          * (agx3c) form G18G takes has all eight independent wait bits. */
         uint16_t fsel_wait = st->wait;
         if (enc == AGX_ISA_ENCODING_AGX3B && (fsel_wait & (fsel_wait - 1))) {
            if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
               AGX3_REFUSE_RET(I);
            fsel_wait = 0;
         }

         const bool maximum = I->fcond == AGX_FCOND_GTN;
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         bool composed;
         if (second_imm) {
            composed =
               maximum
                  ? agx3_alu_fmax_f32_imm(
                       enc, I->dest[0].value, reg_a.value,
                       (uint8_t)other_a.value, agx3_post_ra_hint(reg_a),
                       I->dest[0].cache, fsel_wait, bytes, &len)
                  : agx3_alu_fmin_f32_imm(
                       enc, I->dest[0].value, reg_a.value,
                       (uint8_t)other_a.value, agx3_post_ra_hint(reg_a),
                       I->dest[0].cache, fsel_wait, bytes, &len);
         } else {
            const uint16_t src[4] = {
               reg_a.value, other_a.value, reg_b.value, other_b.value};
            const enum agx3_alu_hint hints[4] = {
               agx3_post_ra_hint(reg_a), agx3_post_ra_hint(other_a),
               agx3_post_ra_hint(reg_b), agx3_post_ra_hint(other_b)};
            composed =
               maximum
                  ? agx3_alu_fmax_f32(enc, I->dest[0].value, src, hints,
                                      I->dest[0].cache, fsel_wait, bytes,
                                      &len)
                  : agx3_alu_fmin_f32(enc, I->dest[0].value, src, hints,
                                      I->dest[0].cache, fsel_wait, bytes,
                                      &len);
         }
         if (!composed || !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Native tied half comparison/selection is scalar fmin/fmax. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d) &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
       (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
       !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       I->src[0].value == I->src[2].value &&
       I->src[1].value == I->src[3].value &&
       agx3_fcmpsel_all_half_regs(I)) {
      uint16_t src[4];
      enum agx3_alu_hint hints[4];
      for (unsigned i = 0; i < 4; ++i) {
         src[i] = I->src[i].value;
         hints[i] = agx3_post_ra_hint(I->src[i]);
      }
      uint16_t fsel_wait = st->wait;
      if (enc == AGX_ISA_ENCODING_AGX3B &&
          (fsel_wait & (fsel_wait - 1))) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool composed = I->fcond == AGX_FCOND_GTN
         ? agx3_alu_fmax_half(enc, I->dest[0].value, src, hints,
                              I->dest[0].cache, fsel_wait, bytes, &len)
         : agx3_alu_fmin_half(enc, I->dest[0].value, src, hints,
                              I->dest[0].cache, fsel_wait, bytes, &len);
      if (!composed ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* f32 compared against a minifloat, selecting one full register plus
    * zero or two half-width byte constants: both are native FSEL
    * forms. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_fcond_witnessed_select(I->fcond) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32) &&
       agx_channels(I->dest[0]) == 1 &&
       ((agx3_mod_src_ok(I->src[0]) &&
         I->src[1].type == AGX_INDEX_IMMEDIATE) ||
        (I->src[0].type == AGX_INDEX_IMMEDIATE &&
         agx3_mod_src_ok(I->src[1]))) &&
       ((I->dest[0].size == AGX_SIZE_16 &&
         I->src[2].type == AGX_INDEX_IMMEDIATE &&
         I->src[3].type == AGX_INDEX_IMMEDIATE) ||
        (I->dest[0].size == AGX_SIZE_32 &&
         ((I->src[2].type == AGX_INDEX_IMMEDIATE) !=
          (I->src[3].type == AGX_INDEX_IMMEDIATE))))) {
      const bool dest_16 = I->dest[0].size == AGX_SIZE_16;
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index compare = I->src[immediate_first ? 1 : 0];
      const agx_index threshold = I->src[immediate_first ? 0 : 1];
      const unsigned selected_index[2] = {
         I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
      bool selected_immediate[2];
      uint16_t selected[2];
      enum agx3_alu_hint hints[3] = {
         agx3_post_ra_hint(compare),
         AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_NONE,
      };
      const bool compare_neg = compare.neg;
      bool invalid = threshold.value > UINT8_MAX || threshold.abs ||
                     threshold.neg || compare.abs || compare.kill ||
                     threshold.cache || threshold.discard;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= I->src[i].memory || I->src[i].kill ||
                    (I->src[i].cache && I->src[i].discard) ||
                    agx_channels(I->src[i]) != 1;
      for (unsigned i = 2; i < 4; ++i)
         invalid |= I->src[i].abs || I->src[i].neg;
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index value = I->src[selected_index[i]];
         selected_immediate[i] = value.type == AGX_INDEX_IMMEDIATE;
         invalid |= selected_immediate[i]
                       ? (value.value > UINT8_MAX || value.cache ||
                          value.discard)
                       : (!agx3_src_is_register(value) ||
                          value.size != I->dest[0].size);
         selected[i] = value.value;
         if (!selected_immediate[i])
            hints[i + 1] = agx3_post_ra_hint(value);
      }

      unsigned condition = immediate_first
                              ? agx3_fcond_reverse(I->fcond)
                              : I->fcond;
      uint8_t minifloat = threshold.value;
      if (compare_neg) {
         /* Ordered IEEE comparisons preserve their NaN result when
          * both operands are negated, while their direction reverses.
          * Apple's -x < -0 and -x < 1 kernels use exactly this
          * canonical x > +0 / x > -1 form. */
         condition = agx3_fcond_reverse(condition);
         minifloat ^= 0x80;
      }

      uint16_t fsel_wait = st->wait;
      const bool retire_wait =
         dest_16 ? (fsel_wait > 32 ||
                    (fsel_wait && (fsel_wait & (fsel_wait - 1))))
                 : (fsel_wait != 0 && fsel_wait != 32);
      if (retire_wait)
         fsel_wait = 0;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!invalid &&
          agx3_alu_fcmp_minifloat_select_mixed(
             enc, I->dest[0].value, compare.value,
             minifloat, selected, selected_immediate, hints,
             condition, dest_16, I->dest[0].cache, fsel_wait, bytes,
             &len)) {
         if (retire_wait &&
             !agx3_retire_result_wait(emission, enc, I, st->wait))
            AGX3_REFUSE_RET(I);
         if (!agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Native scalar half comparison/select.  This is the general
    * shaderFloat16 ternary form: compare two half registers and select
    * either of two half registers without widening or materializing a
    * boolean.  Keep every operand's physical half and lifetime hint.
    */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_fcmpsel_all_half_regs(I)) {
      uint16_t src[4];
      enum agx3_alu_hint hints[4];
      for (unsigned i = 0; i < 4; ++i) {
         src[i] = I->src[i].value;
         hints[i] = agx3_post_ra_hint(I->src[i]);
      }

      unsigned condition = I->fcond;
      bool swap = I->invert_cond;
      if (condition == AGX_FCOND_LTN) {
         condition = AGX_FCOND_GE;
         swap = !swap;
      } else if (condition == AGX_FCOND_GTN) {
         condition = AGX_FCOND_LE;
         swap = !swap;
      }
      if (swap) {
         const uint16_t value = src[2];
         const enum agx3_alu_hint hint = hints[2];
         src[2] = src[3];
         src[3] = value;
         hints[2] = hints[3];
         hints[3] = hint;
      }

      uint16_t fsel_wait = st->wait;
      if (fsel_wait != 0) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_select_half(
             enc, I->dest[0].value, src, hints, condition,
             I->dest[0].cache, fsel_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native mixed-width comparison against floating zero with full
    * float alternatives. Keep the half comparator in its assigned half
    * register; neither widening nor a separate comparison is needed. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 &&
       agx3_mod_src_ok(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_mod_src_ok(I->src[2]) && agx3_mod_src_ok(I->src[3]) &&
       agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1) {
      const bool zero_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index zero = I->src[zero_first ? 0 : 1];
      const agx_index compare = I->src[zero_first ? 1 : 0];
      if (zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0 &&
          compare.type == AGX_INDEX_REGISTER && !compare.memory &&
          /* A negated f32 compare uses the signed-immediate form below. */
          !(compare.neg && compare.size == AGX_SIZE_32 &&
            agx3_fcond_witnessed_select(I->fcond)) &&
          agx_channels(compare) == 1 &&
          (compare.size == AGX_SIZE_16 || compare.size == AGX_SIZE_32)) {
         unsigned condition = I->fcond;
         bool swap = I->invert_cond;
         /* LTN/GTN are minNum/maxNum conditions. Rewriting them to
          * GE/LE with swapped integer selections returns the NaN when
          * src0 is NaN. Untied condition-3/7 selects are not grounded. */
         if (ctx->key->dev.chip != AGX_CHIP_G17P &&
             (condition == AGX_FCOND_LTN || condition == AGX_FCOND_GTN))
            AGX3_REFUSE_RET(I);
         if (zero_first) {
            switch (condition) {
            case AGX_FCOND_LT: condition = AGX_FCOND_GT; break;
            case AGX_FCOND_GT: condition = AGX_FCOND_LT; break;
            case AGX_FCOND_LE: condition = AGX_FCOND_GE; break;
            case AGX_FCOND_GE: condition = AGX_FCOND_LE; break;
            default: break;
            }
         }
         const agx_index selected[] = {
            I->src[swap ? 3 : 2], I->src[swap ? 2 : 3]};
         const uint16_t sources[] = {
            compare.value, selected[0].value, selected[1].value};
         const enum agx3_alu_hint hints[] = {
            agx3_post_ra_hint(compare), agx3_post_ra_hint(selected[0]),
            agx3_post_ra_hint(selected[1])};
         bool invalid = I->saturate;
         for (unsigned i = 0; i < 4; ++i)
            invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                       (I->src[i].cache && I->src[i].discard);
         if (invalid)
            AGX3_REFUSE_RET(I);
         uint16_t fsel_wait = st->wait;
         if (fsel_wait != 0 && fsel_wait != 32) {
            if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
               AGX3_REFUSE_RET(I);
            fsel_wait = 0;
         }
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_fcmp_select_zero(enc, I->dest[0].value, sources,
                 compare.size == AGX_SIZE_16, hints, condition,
                 I->dest[0].cache, fsel_wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* A scalar half compared against a minifloat, selecting two full
    * floats: the half stays in its physical half; a leading immediate
    * reverses the comparison, an inverted condition exchanges the
    * results. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_fcond_witnessed_select(I->fcond) &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       (agx3_vary_reg32(I->src[2]) ||
        I->src[2].type == AGX_INDEX_IMMEDIATE) &&
       (agx3_vary_reg32(I->src[3]) ||
        I->src[3].type == AGX_INDEX_IMMEDIATE) &&
       !(I->src[2].type == AGX_INDEX_IMMEDIATE &&
         I->src[3].type == AGX_INDEX_IMMEDIATE) &&
       agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
       ((agx3_half_reg(I->src[0]) &&
         I->src[1].type == AGX_INDEX_IMMEDIATE) ||
        (I->src[0].type == AGX_INDEX_IMMEDIATE &&
         agx3_half_reg(I->src[1])))) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index immediate = I->src[immediate_first ? 0 : 1];
      const agx_index compare = I->src[immediate_first ? 1 : 0];
      const unsigned true_src = I->invert_cond ? 3 : 2;
      const unsigned false_src = I->invert_cond ? 2 : 3;
      bool invalid = immediate.value > UINT8_MAX;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                    I->src[i].memory || agx_channels(I->src[i]) != 1 ||
                    (I->src[i].cache && I->src[i].discard);
      if (invalid)
         AGX3_REFUSE_RET(I);

      const uint16_t selected[] = {
         I->src[true_src].value, I->src[false_src].value};
      const bool selected_immediate[] = {
         I->src[true_src].type == AGX_INDEX_IMMEDIATE,
         I->src[false_src].type == AGX_INDEX_IMMEDIATE};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(compare),
         selected_immediate[0]
            ? AGX3_ALU_HINT_NONE
            : agx3_post_ra_hint(I->src[true_src]),
         selected_immediate[1]
            ? AGX3_ALU_HINT_NONE
            : agx3_post_ra_hint(I->src[false_src])};
      const unsigned condition =
         immediate_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
      uint16_t fsel_wait = st->wait;
      if (fsel_wait != 0 && fsel_wait != 32) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_half_minifloat_select_f32(
             enc, I->dest[0].value, compare.value,
             (uint8_t)immediate.value, selected, selected_immediate,
             hints, condition, I->dest[0].cache, fsel_wait, bytes,
             &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * A 32-bit float compared against a minifloat immediate, selecting
    * either of two 32-bit registers (`x < k ? a : b`). The immediate
    * may be on either side; a leading immediate reverses the
    * comparison.
    */
   /*
    * G17G float comparison to a 16-bit boolean: agx_lower_pseudo()
    * turns FCMP into fcmpsel(a, b, #1, #0) (or #0, #1 when inverted),
    * which is Apple's `fseli16` with its two small selected constants
    * (gpu/m5-xfb-alu-reference.metal, -alu3-, -alu4-). The comparand
    * is a minifloat immediate or a 32-bit register; eq, lt, gt, ge and
    * le are witnessed. Only the slot-8 wait is witnessed, so any other
    * producer retires through the band no-op first.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       !I->invert_cond &&
       (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
        I->fcond == AGX_FCOND_GT || I->fcond == AGX_FCOND_GE ||
        I->fcond == AGX_FCOND_LE) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       !I->src[0].abs && !I->src[0].neg &&
       !(I->src[0].cache && I->src[0].discard) &&
       ((I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
         !I->src[1].neg && I->src[1].value <= 255) ||
        (agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
         I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
         !I->src[1].abs && !I->src[1].neg && !I->src[1].cache)) &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 7 &&
       I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 7) {
      const bool reg_comparand = I->src[1].type != AGX_INDEX_IMMEDIATE;
      /* A partial r0 identity can race an unrelated load into r0.
       * Carry the wait on a select operand and include all of this
       * select's dependencies, including a possible slot-8 producer
       * of that operand. A lone native slot-8 wait stays on FSEL. */
      const uint16_t retire = (st->wait & ~128u) ? st->wait : 0;
      if (retire &&
          !agx3_retire_select_waits(emission, enc, I, retire))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_bool16_gpu1(
             I->dest[0].value, I->src[0].value,
             agx3_post_ra_hint(I->src[0]), reg_comparand,
             I->src[1].value, reg_comparand && I->src[1].discard,
             (uint8_t)I->src[2].value, (uint8_t)I->src[3].value,
             I->fcond, retire ? 0 : st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~retire;
      return AGX3_PACK_EMITTED;
   }

   /*
    * G17G/G18G float comparisons the forms around this one leave out:
    * a 32-bit result selecting integer constants, or a register and a
    * nonzero integer constant, and a comparison with a 16-bit operand.
    * Mesa carries integers in fcmpsel's selected slots, which is what
    * `uint(a < b)`, `b2i32(flt(a, b))` and every GLSL `a < b ? 1 : 0`
    * into an int become -- constant_expressions' tess-eval tests are
    * nothing else. Apple, applegpu_g18g and applegpu_g17g alike
    * (g18ref-out/probe/m6-alu-sel, -sel2, -hints):
    *
    *   uint(a < b)              fseli16 h2 = ^r1 < ^r2 ? #1 : #0
    *                            clr h3
    *   uint(a < 0.5f)           the same, comparand minifloat 0.5
    *   uint(float(h) >= b)      fseli16 h2 = ^r1 <= h4 ? #1 : #0 (f41 = 1)
    *                            clr h3
    *   ushort(a != float(h))    fseli16 h2 = ^r1 == h4 ? #0 : #1
    *   (a < b) ? 2u : 1u        fseli32 r0 = ^r1 < ^r2 ? #2 : #1
    *   (a == 0.0f) ? 2u : 1u    fseli32 r0 = ^r0 == 0.0 ? #2 : #1
    *   (a < b) ? c : 1u         fseli32 r0 = ^r1 < ^r2 ? r4 : #1
    *
    * A 0/1 pair is a boolean, which Apple writes to the low half and
    * zero-extends by clearing the high one, at every comparand kind;
    * any other constants take `fseli32`'s integer immediates. A 16-bit
    * comparand is only ever the second one: a leading half is exchanged
    * with the float and the relation reversed, as Apple orders it.
    * Shapes other than these fall through to the handlers below.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32)) {
      const bool dest_16 = I->dest[0].size == AGX_SIZE_16;
      agx_index c0 = I->src[0], c1 = I->src[1];
      unsigned condition = I->fcond;
      bool swap = I->invert_cond;
      if (condition == AGX_FCOND_LTN) {
         condition = AGX_FCOND_GE;
         swap = !swap;
      } else if (condition == AGX_FCOND_GTN) {
         condition = AGX_FCOND_LE;
         swap = !swap;
      }
      /* The float comparand goes first: exchange a leading immediate or
       * half with a 32-bit register, reversing the relation. */
      if (agx3_vary_reg32(c1) &&
          (c0.type == AGX_INDEX_IMMEDIATE || agx3_half_reg(c0))) {
         const agx_index t = c0;
         c0 = c1;
         c1 = t;
         condition = agx3_fcond_reverse(condition);
      }
      const bool cmp1_imm = agx3_plain_immediate(c1) && !c1.abs &&
                            !c1.neg && c1.value <= 255;
      const bool cmp1_16 = agx3_half_reg(c1) && !c1.abs && !c1.neg;
      const agx_index t = I->src[swap ? 3 : 2];
      const agx_index f = I->src[swap ? 2 : 3];
      const bool t_imm = agx3_plain_immediate(t) && !t.abs && !t.neg &&
                         t.value <= 255;
      const bool f_imm = agx3_plain_immediate(f) && !f.abs && !f.neg &&
                         f.value <= 255;
      const bool t_reg = dest_16 ? agx3_half_reg(t) && !t.abs && !t.neg
                                 : agx3_vary_reg32(t);
      const bool f_reg = dest_16 ? agx3_half_reg(f) && !f.abs && !f.neg
                                 : agx3_vary_reg32(f);
      const bool boolean =
         t_imm && f_imm && t.value + f.value == 1 && !dest_16;
      bool plain = true;
      for (unsigned i = 0; i < 4; ++i)
         plain &= !I->src[i].kill && !I->src[i].memory &&
                  !(I->src[i].cache && I->src[i].discard);
      /* The native half-result form can select a half register and
       * an integer literal, in either order, while comparing float
       * with half. The comparison's half may be odd and cached. */
      const bool mine =
         plain && agx3_vary_reg32(c0) &&
         (dest_16 || !(I->dest[0].value & 1)) &&
         (agx3_vary_reg32(c1) || cmp1_imm || cmp1_16) &&
         (t_imm || t_reg) && (f_imm || f_reg) &&
         agx3_fcond_witnessed_select(condition) &&
         (dest_16 ? cmp1_16 && (t_imm || f_imm)
                  : (boolean ||
                     (!cmp1_16 &&
                      ((t_imm && f_imm) || (t_imm && t.value) ||
                       (f_imm && f.value)))));
      if (mine) {
         const struct agx3_fsel32_gpu1_desc d = {
            .dest = I->dest[0].value,
            .dest_cache = I->dest[0].cache,
            .cmp0 = c0.value,
            .cmp0_hint = agx3_post_ra_hint(c0),
            .cmp1_immediate = cmp1_imm,
            .cmp1 = c1.value,
            .cmp1_hint =
               cmp1_imm ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(c1),
            .cmp1_16 = cmp1_16,
            .sel_immediate = {t_imm, f_imm},
            .sel = {t.value, f.value},
            .sel_hint = {t_imm ? AGX3_ALU_HINT_NONE
                               : agx3_post_ra_hint(t),
                         f_imm ? AGX3_ALU_HINT_NONE
                               : agx3_post_ra_hint(f)},
            .sel_integer = true,
            .condition = condition,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (dest_16 || boolean) {
            /* A boolean into a 32-bit result: the half select into the
             * low half, then the clear of the high half. The select
             * carries the wait; the clear writes nothing it waits on. */
            uint8_t hi[AGX3_STAGE_MAX_LENGTH], hi_len = 0;
            if (!agx3_alu_fcmp_select16_gpu1(&d, bytes, &len) ||
                (!dest_16 &&
                 !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                           &hi_len)) ||
                !agx3_append(emission, bytes, len) ||
                (!dest_16 && !agx3_append(emission, hi, hi_len)))
               AGX3_REFUSE_RET(I);
         } else if (!agx3_alu_fcmp_select32_gpu1(&d, bytes, &len) ||
                    !agx3_append(emission, bytes, len)) {
            AGX3_REFUSE_RET(I);
         }
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_fcond_witnessed_select(I->fcond) &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
       agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
       ((agx3_mod_src_ok(I->src[0]) &&
         I->src[1].type == AGX_INDEX_IMMEDIATE) ||
        (agx3_mod_src_ok(I->src[1]) &&
         I->src[0].type == AGX_INDEX_IMMEDIATE))) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      agx_index immediate = I->src[immediate_first ? 0 : 1];
      const agx_index compare = I->src[immediate_first ? 1 : 0];
      const unsigned true_src = I->invert_cond ? 3 : 2;
      const unsigned false_src = I->invert_cond ? 2 : 3;

      bool invalid = immediate.value > 255 ||
                     agx_channels(compare) != 1;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= (I->src[i].abs && i != (immediate_first ? 1 : 0)) ||
                    (I->src[i].neg && i != (immediate_first ? 1 : 0)) ||
                    I->src[i].kill || I->src[i].memory ||
                    (I->src[i].cache && I->src[i].discard);
      if (invalid)
         AGX3_REFUSE_RET(I);

      unsigned condition =
         immediate_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
      /* -r cmp k is r reverse(cmp) -k for every ordered relation,
       * including signed zeros and unordered NaN inputs. Preserve ABS
       * and the chosen values; only the comparison is reversed. */
      if (compare.neg) {
         immediate.value ^= 0x80;
         condition = agx3_fcond_reverse(condition);
      }

      const enum agx3_alu_hint hints[4] = {
         agx3_post_ra_hint(compare), AGX3_ALU_HINT_NONE,
         agx3_post_ra_hint(I->src[true_src]),
         agx3_post_ra_hint(I->src[false_src])};

      uint16_t fsel_wait = st->wait;
      if ((compare.abs && fsel_wait != 0 && fsel_wait != 32) ||
          (fsel_wait & (fsel_wait - 1))) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_select_f32_imm(
             enc, I->dest[0].value, compare.value,
             (uint8_t)immediate.value, compare.abs,
             I->src[true_src].value,
             I->src[false_src].value, hints, condition,
             I->dest[0].cache, fsel_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Full/half comparisons select between full-register alternatives.
    * WBOIT uses EQ against half +inf; update_lightmap uses ordered LT
    * against a half threshold. The condition selector is orthogonal to
    * the native mixed-width routing and is byte-grounded for both. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 &&
       agx3_fcond_witnessed_select(I->fcond) && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
       agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1 &&
       ((agx3_vary_reg32(I->src[0]) && agx3_half_reg(I->src[1])) ||
        (agx3_vary_reg32(I->src[1]) && agx3_half_reg(I->src[0])))) {
      /* The half may be on either side. Exchanging the compared
       * operands reverses the comparison and nothing else; an inverted
       * condition exchanges the two selected operands, which is how
       * Apple spells `!=`. */
      const bool half_first = agx3_half_reg(I->src[0]);
      const unsigned order[4] = {
         half_first ? 1u : 0u, half_first ? 0u : 1u,
         I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
      uint16_t src[4];
      enum agx3_alu_hint hints[4];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      bool invalid = agx_channels(I->src[0]) != 1 ||
                     agx_channels(I->src[1]) != 1;
      for (unsigned i = 0; i < 4; ++i) {
         invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                    (I->src[i].cache && I->src[i].discard);
         src[i] = I->src[order[i]].value;
         hints[i] = agx3_post_ra_hint(I->src[order[i]]);
      }
      const unsigned condition =
         half_first ? agx3_fcond_reverse(I->fcond) : I->fcond;
      uint16_t fsel_wait = st->wait;
      if (fsel_wait != 0 && fsel_wait != 32) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      if (invalid ||
          !agx3_alu_fcmp_select_f32_f16(
             enc, I->dest[0].value, src, hints, condition,
             I->dest[0].cache, fsel_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * G17G's 32-bit float select: a register compared against a
    * register or a minifloat, choosing two registers (`fseli32`) or a
    * register and a constant (`fself32`). This is what the transform-
    * feedback programs' `x < k ? 0.0 : y` becomes, and until now the
    * minifloat comparand fell through to the general handler below,
    * which admits only registers and the exact zero. Apple's own
    * kernels (gpu/m5-xfb-alu7/8/10/11-reference.metal) pin every
    * field; see agx3_alu_core.h.
    *
    * Mesa's fcmpsel carries integer immediates in its selected slots,
    * and `fself32` reads its constant as a minifloat, so zero is the
    * only selected constant admitted here. A leading immediate
    * reverses the comparison; an inverted condition exchanges the
    * results; LTN/GTN are the negations of GE/LE with the results
    * exchanged, exactly as the general handler spells them.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       ((agx3_vary_reg32(I->src[0]) &&
         (agx3_vary_reg32(I->src[1]) ||
          I->src[1].type == AGX_INDEX_IMMEDIATE)) ||
        (I->src[0].type == AGX_INDEX_IMMEDIATE &&
         agx3_vary_reg32(I->src[1]))) &&
       (agx3_vary_reg32(I->src[2]) || agx3_src_is_zero_imm(I->src[2])) &&
       (agx3_vary_reg32(I->src[3]) || agx3_src_is_zero_imm(I->src[3])) &&
       !(I->src[2].type == AGX_INDEX_IMMEDIATE &&
         I->src[3].type == AGX_INDEX_IMMEDIATE)) {
      const bool immediate_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index cmp0 = I->src[immediate_first ? 1 : 0];
      const agx_index cmp1 = I->src[immediate_first ? 0 : 1];
      unsigned condition = I->fcond;
      bool swap = I->invert_cond;
      if (condition == AGX_FCOND_LTN) {
         condition = AGX_FCOND_GE;
         swap = !swap;
      } else if (condition == AGX_FCOND_GTN) {
         condition = AGX_FCOND_LE;
         swap = !swap;
      }
      if (immediate_first)
         condition = agx3_fcond_reverse(condition);
      const agx_index sel[2] = {I->src[swap ? 3 : 2], I->src[swap ? 2 : 3]};
      bool invalid = !agx3_fcond_witnessed_select(condition) ||
                     cmp1.value > UINT8_MAX;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= I->src[i].kill || I->src[i].memory ||
                    agx_channels(I->src[i]) != 1 ||
                    (I->src[i].cache && I->src[i].discard);
      if (invalid)
         AGX3_REFUSE_RET(I);

      const struct agx3_fsel32_gpu1_desc d = {
         .dest = I->dest[0].value,
         .dest_cache = I->dest[0].cache,
         .cmp0 = cmp0.value,
         .cmp0_hint = agx3_post_ra_hint(cmp0),
         .cmp1_immediate = cmp1.type == AGX_INDEX_IMMEDIATE,
         .cmp1 = cmp1.value,
         .cmp1_hint = cmp1.type == AGX_INDEX_IMMEDIATE
                         ? AGX3_ALU_HINT_NONE
                         : agx3_post_ra_hint(cmp1),
         .sel_immediate = {sel[0].type == AGX_INDEX_IMMEDIATE,
                           sel[1].type == AGX_INDEX_IMMEDIATE},
         .sel = {sel[0].value, sel[1].value},
         .sel_hint = {sel[0].type == AGX_INDEX_IMMEDIATE
                         ? AGX3_ALU_HINT_NONE
                         : agx3_post_ra_hint(sel[0]),
                      sel[1].type == AGX_INDEX_IMMEDIATE
                         ? AGX3_ALU_HINT_NONE
                         : agx3_post_ra_hint(sel[1])},
         .condition = condition,
         /* Eight wait bits, like every GPU1 form; only slot 8 is
          * witnessed, which is also the only slot a G17G load
          * publishes on. */
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_fcmp_select32_gpu1(&d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Scalar-half comparisons selecting full values are a native FSEL
    * form of their own.  Vertex-format normalization produces this for
    * `fabs(half) < half ? float : float`; widening the comparison would
    * change mediump rounding and is unnecessary. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_fcond_witnessed_select(I->fcond) &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_half_reg(I->src[0]) && agx3_half_reg(I->src[1]) &&
       agx3_vary_reg32(I->src[2]) && agx3_vary_reg32(I->src[3]) &&
       agx_channels(I->src[2]) == 1 && agx_channels(I->src[3]) == 1) {
      const unsigned selected[] = {
         I->invert_cond ? 3u : 2u, I->invert_cond ? 2u : 3u};
      const uint16_t src[] = {
         I->src[0].value, I->src[1].value,
         I->src[selected[0]].value, I->src[selected[1]].value};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(I->src[0]),
         agx3_post_ra_hint(I->src[1]),
         agx3_post_ra_hint(I->src[selected[0]]),
         agx3_post_ra_hint(I->src[selected[1]])};
      const bool src_abs[] = {I->src[0].abs, I->src[1].abs};
      bool invalid = I->src[0].neg || I->src[1].neg;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= I->src[i].memory || I->src[i].kill ||
                    (I->src[i].cache && I->src[i].discard) ||
                    (i >= 2 && (I->src[i].abs || I->src[i].neg));

      uint16_t fsel_wait = st->wait;
      if (fsel_wait != 0 && fsel_wait != 32) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (invalid ||
          !agx3_alu_fcmp_select_f16_f16_f32(
             enc, I->dest[0].value, src, hints, src_abs, I->fcond,
             I->dest[0].cache, fsel_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* update_lightmap forms a half boolean mask by comparing f32 against
    * a minifloat constant.  Equality is commutative, so normalize the
    * immediate to Apple's second compare slot without changing the
    * selected values. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->invert_cond &&
       (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
        I->fcond == AGX_FCOND_GT) && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       ((I->src[0].type == AGX_INDEX_IMMEDIATE &&
         agx3_vary_reg32(I->src[1])) ||
        (agx3_vary_reg32(I->src[0]) &&
         I->src[1].type == AGX_INDEX_IMMEDIATE)) &&
       agx3_src_is_register(I->src[2]) && !I->src[2].memory &&
       I->src[2].size == AGX_SIZE_16 && agx_channels(I->src[2]) == 1 &&
       I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 255) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index immediate = I->src[imm_first ? 0 : 1];
      const agx_index compare = I->src[imm_first ? 1 : 0];
      unsigned condition = I->fcond;
      if (imm_first) {
         if (condition == AGX_FCOND_LT)
            condition = AGX_FCOND_GT;
         else if (condition == AGX_FCOND_GT)
            condition = AGX_FCOND_LT;
      }
      bool invalid = immediate.value > 255;
      for (unsigned i = 0; i < 4; ++i)
         invalid |= I->src[i].abs || I->src[i].neg || I->src[i].kill ||
                    I->src[i].memory ||
                    (I->src[i].cache && I->src[i].discard);
      uint16_t fsel_wait = st->wait;
      if (fsel_wait != 0 && fsel_wait != 32) {
         if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
            AGX3_REFUSE_RET(I);
         fsel_wait = 0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (invalid ||
          !agx3_alu_fcmp_minifloat_select_half(
             enc, I->dest[0].value, compare.value, immediate.value,
             I->src[2].value, I->src[3].value,
             agx3_post_ra_hint(compare),
             agx3_post_ra_hint(I->src[2]), condition,
             I->dest[0].cache,
             fsel_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests==1 && I->nr_srcs==4 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory && agx_channels(I->dest[0])==1 &&
       (I->dest[0].size==AGX_SIZE_16 || I->dest[0].size==AGX_SIZE_32) &&
       I->src[2].type==AGX_INDEX_IMMEDIATE && I->src[3].type==AGX_INDEX_IMMEDIATE &&
       I->src[2].value<=255 && I->src[3].value<=255 &&
       ((I->src[0].type==AGX_INDEX_IMMEDIATE && I->src[0].value==0 && agx3_vary_reg32(I->src[1])) ||
        (I->src[1].type==AGX_INDEX_IMMEDIATE && I->src[1].value==0 && agx3_vary_reg32(I->src[0])))) {
      const bool zero_first=I->src[0].type==AGX_INDEX_IMMEDIATE;
      const agx_index compare=I->src[zero_first?1:0];
      unsigned condition=I->fcond;
      bool swap=I->invert_cond;
      if(condition==AGX_FCOND_LTN) {condition=AGX_FCOND_GE;swap=!swap;}
      else if(condition==AGX_FCOND_GTN) {condition=AGX_FCOND_LE;swap=!swap;}
      if(zero_first) {
         switch(condition) {
         case AGX_FCOND_LT:condition=AGX_FCOND_GT;break;
         case AGX_FCOND_GT:condition=AGX_FCOND_LT;break;
         case AGX_FCOND_LE:condition=AGX_FCOND_GE;break;
         case AGX_FCOND_GE:condition=AGX_FCOND_LE;break;
         default:break;
         }
      }
      const uint16_t src[]={compare.value,0,I->src[swap?3:2].value,I->src[swap?2:3].value};
      const bool compare_abs[]={false,false};
      const bool compare_neg[]={I->src[zero_first?1:0].neg,
                                I->src[zero_first?0:1].neg};
      const bool immediate[]={true,true};
      const enum agx3_alu_hint hints[]={agx3_post_ra_hint(compare),AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
      bool invalid=I->saturate || agx_channels(compare)!=1;
      for(unsigned i=0;i<4;i++)invalid|=I->src[i].abs || (i>=2 && I->src[i].neg) ||
         I->src[i].kill || (I->src[i].cache && I->src[i].discard);
      if(invalid)AGX3_REFUSE_RET(I);
      uint16_t fsel_wait=st->wait;
      if(fsel_wait!=0 && fsel_wait!=32) {
         if(!agx3_retire_select_waits(emission,enc,I,fsel_wait))AGX3_REFUSE_RET(I);
         fsel_wait=0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,true,
           compare_abs,compare_neg,immediate,hints,
           condition,I->dest[0].size==AGX_SIZE_16,I->dest[0].cache,fsel_wait,bytes,&len) ||
         !agx3_append(emission,bytes,len))AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native FSEL selects retained 16/32-bit register values or a register
    * and an immediate directly. Preserve each operand's lifetime hints
    * without materializing a separate comparison result.
    *
    * A negated comparator rides through to the encoder rather than
    * failing the shader closed: f65 names the second comparator, and
    * every other placement is an operand exchange or a relation flip
    * that agx3_alu_fcmp_select_reg_imm() performs. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d) &&
       I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests==1 && I->nr_srcs==4 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory && agx_channels(I->dest[0])==1 &&
       (I->dest[0].size==AGX_SIZE_16 || I->dest[0].size==AGX_SIZE_32) &&
       ((agx3_mod_src_ok(I->src[0]) && agx3_mod_src_ok(I->src[1])) ||
        (agx3_vary_reg32(I->src[0]) && agx3_src_is_zero_imm(I->src[1])) ||
        (agx3_src_is_zero_imm(I->src[0]) && agx3_vary_reg32(I->src[1]))) &&
       (agx3_src_is_register(I->src[2]) ||
        agx3_src_is_register(I->src[3]) ||
        /* G17G's fseli16 also selects two integer bytes, e.g. the
         * 255/0 mask a conditional discard lowers to. */
        (ctx->key->dev.emit_agx3d &&
         I->dest[0].size == AGX_SIZE_16)) &&
       /* A uniform alternative takes the general descriptor below. */
       I->src[2].type != AGX_INDEX_UNIFORM &&
       I->src[3].type != AGX_INDEX_UNIFORM) {
      /* G16G's condition-3/7 full-register form is grounded only when
       * the alternatives are tied min/max operands, handled above.
       * Rewriting an untied all-register form to ordered GE/LE returns
       * the NaN operand. G17P retains its separately qualified legacy
       * form, and immediate-alternative M4 witnesses remain below. */
      if (ctx->key->dev.chip != AGX_CHIP_G17P &&
          I->dest[0].size == AGX_SIZE_32 &&
          (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
          agx3_src_is_register(I->src[0]) &&
          agx3_src_is_register(I->src[1]) &&
          agx3_src_is_register(I->src[2]) &&
          agx3_src_is_register(I->src[3]))
         AGX3_REFUSE_RET(I);
      const bool zero_first=agx3_src_is_zero_imm(I->src[0]);
      const bool compare_zero=zero_first || agx3_src_is_zero_imm(I->src[1]);
      unsigned condition=I->fcond;
      bool swap=I->invert_cond;
      if(condition==AGX_FCOND_LTN) {condition=AGX_FCOND_GE;swap=!swap;}
      else if(condition==AGX_FCOND_GTN) {condition=AGX_FCOND_LE;swap=!swap;}
      if(zero_first) {
         switch(condition) {
         case AGX_FCOND_LT:condition=AGX_FCOND_GT;break;
         case AGX_FCOND_GT:condition=AGX_FCOND_LT;break;
         case AGX_FCOND_LE:condition=AGX_FCOND_GE;break;
         case AGX_FCOND_GE:condition=AGX_FCOND_LE;break;
         default:break;
         }
      }
      const agx_index operands[]={I->src[zero_first?1:0],I->src[zero_first?0:1],
                                 I->src[swap?3:2],I->src[swap?2:3]};
      uint16_t src[4]; enum agx3_alu_hint hints[4]; bool immediate[2];
      bool compare_abs[2]; bool compare_neg[2];
      bool invalid=I->saturate;
      for(unsigned i=0;i<4;i++) {
         const agx_index x=operands[i];
         const bool imm=x.type==AGX_INDEX_IMMEDIATE;
         /* Abs and neg are admitted on either comparator (f65 negates
          * the second; agx3_alu_fcmp_select_reg_imm() swaps the operands
          * and flips the relation for the first). Refused: a modifier on
          * a selected value (no field; a select may carry an integer), a
          * killed or spilled operand, a vector operand, conflicting
          * lifetime hints, an out-of-range immediate, and a selected
          * register whose width does not match the destination. */
         invalid|=(i>=2 && (x.abs || x.neg)) || x.kill || x.memory ||
            agx_channels(x)!=1 ||
            (x.cache && x.discard) ||
            (imm ? x.value>255 : (!agx3_src_is_register(x) ||
              x.size!=(i<2?AGX_SIZE_32:I->dest[0].size)));
         src[i]=x.value; hints[i]=imm?AGX3_ALU_HINT_NONE:agx3_post_ra_hint(x);
         if(i<2){compare_abs[i]=x.abs;compare_neg[i]=x.neg;}
         if(i>=2)immediate[i-2]=imm;
      }
      if(invalid)AGX3_REFUSE_RET(I);
      /* G17G's select carries its wait on slot 8, G17P's on slot 6;
       * anything else retires before the select issues. */
      const uint16_t fsel_native_wait =
         enc == AGX_ISA_ENCODING_AGX3C ? 128u : 32u;
      uint16_t fsel_wait=st->wait;
      if(fsel_wait!=0 && fsel_wait!=fsel_native_wait) {
         if(!agx3_retire_select_waits(emission,enc,I,fsel_wait))AGX3_REFUSE_RET(I);
         fsel_wait=0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      if(!agx3_alu_fcmp_select_reg_imm(enc,I->dest[0].value,src,compare_zero,
           compare_abs,compare_neg,immediate,hints,condition,I->dest[0].size==AGX_SIZE_16,
           I->dest[0].cache,fsel_wait,bytes,&len) ||
         !agx3_append(emission,bytes,len))AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Float comparison selecting integer byte constants, preserving the
    * native 16/32-bit destination width assigned by register allocation. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 &&
       I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_16 || I->dest[0].size == AGX_SIZE_32) &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       (I->src[0].size == AGX_SIZE_16 || I->src[0].size == AGX_SIZE_32) &&
       (I->src[1].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_32) &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
       I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value <= 255 &&
       /* This specialization's abs encoder
        * (agx3_alu_fcmp_select_imm_width_abs) has no negate argument,
        * so a comparison with both an abs and a negate falls through to
        * the general compare-select, which deposits f47, f64 and f65
        * independently. */
       !((I->src[0].abs || I->src[1].abs) &&
         (I->src[0].neg || I->src[1].neg))) {
      /* A negated comparison source. f65 names the SECOND comparator
       * and only that one; `-a > b` is Apple's own `b < -a`, so the
       * encoder swaps the operands and reverses the relation instead of
       * refusing. Every relation, both result widths and any pair of
       * selected immediates are witnessed -- see the byte ladder above
       * agx3_alu_fcmp_select_imm_width_internal(). `dot(view - world,
       * normal) > 0.0` is the shape that reaches this. */
      /* An absolute comparison source moves the instruction to the
       * longer ep28..35 endpoint. It is a different form, not a bit on
       * the short one, so it gets its own encoder. Mixed absolute and
       * negate modifiers also use that endpoint, but are handled by the
       * general descriptor below. */
      const bool any_abs = I->src[0].abs || I->src[1].abs;
      const bool any_neg = I->src[0].neg || I->src[1].neg;
      const bool negated = !any_abs && any_neg;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
      const bool src_16[] = {I->src[0].size == AGX_SIZE_16,
                             I->src[1].size == AGX_SIZE_16};
      const bool src_neg[] = {I->src[0].neg, I->src[1].neg};
      if (!(any_abs && any_neg)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         unsigned condition = I->fcond;
         bool swap = I->invert_cond;
         /* The tied min/max handler above owns condition 3/7. There is
          * no grounded untied encoding with those NaN semantics. */
         if (ctx->key->dev.chip != AGX_CHIP_G17P &&
             (condition == AGX_FCOND_LTN || condition == AGX_FCOND_GTN))
            AGX3_REFUSE_RET(I);
         uint16_t fsel_wait = st->wait;
         if (fsel_wait != 0 && fsel_wait != 32) {
            if (!agx3_retire_select_waits(emission, enc, I, fsel_wait))
               AGX3_REFUSE_RET(I);
            fsel_wait = 0;
         }
         const bool src_abs[] = {I->src[0].abs, I->src[1].abs};
         if (I->saturate || I->src[0].kill || I->src[1].kill ||
             (I->src[0].cache && I->src[0].discard) ||
             (I->src[1].cache && I->src[1].discard) ||
             I->src[2].abs || I->src[2].neg ||
             I->src[3].abs || I->src[3].neg ||
             !(any_abs
                  ? agx3_alu_fcmp_select_imm_width_abs(
                       enc, I->dest[0].value, src, src_16, hints,
                       condition, I->src[swap ? 3 : 2].value,
                       I->src[swap ? 2 : 3].value,
                       I->dest[0].size == AGX_SIZE_16,
                       I->dest[0].cache, fsel_wait, src_abs, bytes, &len)
               : negated
                  ? agx3_alu_fcmp_select_imm_width_neg(
                       enc, I->dest[0].value, src, src_16, hints, src_neg,
                       condition, I->src[swap ? 3 : 2].value,
                       I->src[swap ? 2 : 3].value,
                       I->dest[0].size == AGX_SIZE_16,
                       I->dest[0].cache, fsel_wait, bytes, &len)
                  : agx3_alu_fcmp_select_imm_width(
                       enc, I->dest[0].value, src, src_16, hints,
                       condition, I->src[swap ? 3 : 2].value,
                       I->src[swap ? 2 : 3].value,
                       I->dest[0].size == AGX_SIZE_16,
                       I->dest[0].cache, fsel_wait, bytes, &len)) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * The general compare-select: a register in a select slot, or an
    * immediate in a compare slot (the block above handles immediate
    * selects with register comparators), and a register/register
    * comparison mixing absolute and negate modifiers, which needs this
    * endpoint's independent modifier fields.
    */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FCMPSEL &&
       I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32)) {
      const agx_index c0 = I->src[0], c1 = I->src[1];
      const bool c0_imm = c0.type == AGX_INDEX_IMMEDIATE;
      const bool c1_imm = c1.type == AGX_INDEX_IMMEDIATE;

      /* At most one compare source may be an immediate (any minifloat
       * byte). As Apple does, `k < b` reverses the relation so the
       * immediate is in slot 1, the slot whose kind field takes an
       * immediate; a modifier on it is folded into the constant. */
      /* A uniform may be compared against an immediate zero, which
       * is how Apple selects on a constant-program flag. */
      const bool c0_uni = c0.type == AGX_INDEX_UNIFORM;
      const bool c1_uni = c1.type == AGX_INDEX_UNIFORM;
      /* A slot-1 uniform is compared against a register without a
       * negate (Apple's r < u shape, agx3_alu_fcmp_select); the
       * encoder refuses the exchange a negated slot 0 would need. */
      const bool uniform_ok =
         (!c0_uni && !c1_uni) ||
         (c0_uni != c1_uni && (c0_uni ? c1_imm : c0_imm) &&
          (c0_uni ? c1 : c0).value == 0 &&
          agx3_plain_immediate(c0_uni ? c1 : c0) &&
          agx3_plain_uniform(c0_uni ? c0 : c1)) ||
         (c1_uni && !c0_uni && !c0_imm && agx3_plain_uniform_wide(c1) &&
          agx3_src_is_register(c0) && !c0.neg);
      if (!(c0_imm && c1_imm) && uniform_ok &&
          (!c0_imm || (c0.value <= 0xff && agx3_plain_immediate(c0))) &&
          (!c1_imm || (c1.value <= 0xff && agx3_plain_immediate(c1))) &&
          (c0_imm || c0_uni || agx3_src_is_register(c0)) &&
          (c1_imm || c1_uni || agx3_src_is_register(c1))) {
         const agx_index sel_t = I->src[2], sel_f = I->src[3];
         const bool t_imm = sel_t.type == AGX_INDEX_IMMEDIATE;
         const bool f_imm = sel_f.type == AGX_INDEX_IMMEDIATE;
         /* A uniform alternative (f70 / f87, kind 3) of the result's
          * width: Apple's `c ? u : r` and `c ? r : u` (native corpus
          * form 121, 32073e090002, beside a slot-1 uniform comparand). */
         const bool t_uni = sel_t.type == AGX_INDEX_UNIFORM;
         const bool f_uni = sel_f.type == AGX_INDEX_UNIFORM;

         unsigned condition = I->fcond;
         bool swap = I->invert_cond;
         if (condition == AGX_FCOND_LTN) {
            condition = AGX_FCOND_GE;
            swap = !swap;
         } else if (condition == AGX_FCOND_GTN) {
            condition = AGX_FCOND_LE;
            swap = !swap;
         }
         /* Put the immediate in compare slot 1. */
         const bool reverse = c0_imm;
         const agx_index cmp0 = reverse ? c1 : c0;
         const agx_index cmp1 = reverse ? c0 : c1;
         if (reverse)
            condition = agx3_fcond_reverse(condition);

         const agx_index tv = swap ? sel_f : sel_t;
         const agx_index fv = swap ? sel_t : sel_f;
         const bool tv_imm = swap ? f_imm : t_imm;
         const bool fv_imm = swap ? t_imm : f_imm;
         const bool tv_uni = swap ? f_uni : t_uni;
         const bool fv_uni = swap ? t_uni : f_uni;

         const struct agx3_fsel_desc d = {
            .dest = I->dest[0].value,
            .dest_16 = I->dest[0].size == AGX_SIZE_16,
            .dest_cache = I->dest[0].cache,
            .cmp0 = cmp0.value,
            .cmp0_uniform = cmp0.type == AGX_INDEX_UNIFORM,
            .cmp0_hint = cmp0.type == AGX_INDEX_UNIFORM
                            ? AGX3_ALU_HINT_NONE
                            : agx3_post_ra_hint(cmp0),
            .cmp0_abs = cmp0.abs,
            .cmp0_neg = cmp0.neg,
            .cmp0_16 = cmp0.size == AGX_SIZE_16,
            .cmp1_is_immediate = reverse ? c0_imm : c1_imm,
            .cmp1_uniform = cmp1.type == AGX_INDEX_UNIFORM,
            .cmp1 = cmp1.value,
            .cmp1_hint = cmp1.type == AGX_INDEX_UNIFORM
                            ? AGX3_ALU_HINT_NONE
                            : agx3_post_ra_hint(cmp1),
            .cmp1_abs = cmp1.abs,
            .cmp1_neg = cmp1.neg,
            .cmp1_16 = cmp1.size == AGX_SIZE_16,
            .sel0_kind = tv_imm   ? AGX3_FSEL_SEL_IMMEDIATE
                         : tv_uni ? AGX3_FSEL_SEL_UNIFORM
                                  : AGX3_FSEL_SEL_REGISTER,
            .sel1_kind = fv_imm   ? AGX3_FSEL_SEL_IMMEDIATE
                         : fv_uni ? AGX3_FSEL_SEL_UNIFORM
                                  : AGX3_FSEL_SEL_REGISTER,
            .sel0 = tv.value,
            .sel1 = fv.value,
            .sel0_hint = (tv_imm || tv_uni) ? AGX3_ALU_HINT_NONE
                                            : agx3_post_ra_hint(tv),
            .sel1_hint = (fv_imm || fv_uni) ? AGX3_ALU_HINT_NONE
                                            : agx3_post_ra_hint(fv),
            .sel_16 = I->dest[0].size == AGX_SIZE_16,
            .condition = condition,
            .wait_mask = 0,
         };

         const bool operands_ok =
            !tv.neg && !fv.neg && !tv.abs &&
            !fv.abs && !cmp0.kill && !cmp1.kill &&
            !tv.kill && !fv.kill && !tv.memory && !fv.memory &&
            (!tv_imm || agx3_plain_immediate(tv)) &&
            (!fv_imm || agx3_plain_immediate(fv)) &&
            (!tv_uni || (agx3_plain_uniform(tv) &&
                         tv.size == I->dest[0].size)) &&
            (!fv_uni || (agx3_plain_uniform(fv) &&
                         fv.size == I->dest[0].size)) &&
            !(cmp0.cache && cmp0.discard) &&
            !(cmp1.cache && cmp1.discard) &&
            !(tv.cache && tv.discard) && !(fv.cache && fv.discard) &&
            (tv_imm || tv_uni || agx3_src_is_register(tv)) &&
            (fv_imm || fv_uni || agx3_src_is_register(fv)) &&
            agx_channels(cmp0) == 1 && agx_channels(cmp1) == 1;

         uint8_t bytes[AGX3_ALU_MAX_LENGTH];
         uint8_t len = 0;
         if (operands_ok) {
            /* No witness of this endpoint carries a scoreboard wait, so
             * retire outstanding slots first rather than dropping the
             * dependency. */
            if (st->wait != 0 &&
                !agx3_retire_select_waits(emission, enc, I, st->wait))
               AGX3_REFUSE_RET(I);
            if (agx3_alu_fcmp_select(enc, &d, bytes, &len) &&
                agx3_append(emission, bytes, len))
               return AGX3_PACK_EMITTED;
         }
         AGX3_REFUSE_RET(I);
      }
   }

   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_int_and_moves(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   /* The native half add takes an immediate in slot A. Unshifted
    * addition commutes, so register + immediate uses that same form.
    * Keep the source's liveness and wait on the register after swapping;
    * a shifted second operand cannot be commuted this way. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
       !I->src[0].abs && !I->src[0].neg && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard) &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value <= 15 &&
       !I->src[1].memory && !I->src[1].abs && !I->src[1].neg) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_iadd16_gpu1(
             I->dest[0].value, I->dest[0].cache, true,
             I->src[1].value, AGX3_ALU_HINT_NONE,
             I->src[0].value, agx3_post_ra_hint(I->src[0]),
             0, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* G17G half add with a shifted second operand, register or 0..15
    * immediate first operand (gpu/m5-shift-add16-reference.metal and
    * the coverage folds of Apple's depth/stencil writers).
    *
    * The second operand may also be a 32-bit register: a half result
    * depends only on the low half of each operand -- an add and a left
    * shift carry upwards only -- so the instruction reads that half,
    * which is the register's own index. Apple does exactly this for
    * ushort(1u + (x << 2)) of a uint x that stays live:
    * applegpu_g18g/g17g `iadd32 h4 = #1 + (h0 << 2)` with f39 = 1 and
    * f40 while r0 is later read whole (g18ref-out/probe/m6-alu-hints
    * h_iadd16_shl_live; m6-alu-misc iadd16_shl2). That is the dynamic
    * varying index `1 + (i << 2)` Mesa computes in 16 bits. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift <= 4 && I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       (I->src[1].size == AGX_SIZE_16 ||
        (I->src[1].size == AGX_SIZE_32 && !(I->src[1].value & 1))) &&
       agx_channels(I->src[1]) == 1 &&
       agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       !(I->src[1].cache && I->src[1].discard) &&
       ((I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value <= 15 &&
         !I->src[0].memory && !I->src[0].abs && !I->src[0].neg) ||
        (agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
         I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
         !I->src[0].abs && !I->src[0].neg && !I->src[0].kill &&
         !(I->src[0].cache && I->src[0].discard)))) {
      const bool immediate_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_iadd16_gpu1(
             I->dest[0].value, I->dest[0].cache, immediate_first,
             I->src[0].value,
             immediate_first ? AGX3_ALU_HINT_NONE
                             : agx3_post_ra_hint(I->src[0]),
             I->src[1].value, agx3_post_ra_hint(I->src[1]),
             I->shift, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift >= 1 && I->shift <= 4 && I->dest[0].size == AGX_SIZE_16 &&
       I->src[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       agx3_src_is_register(I->dest[0]) &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       I->src[0].value <= 15 && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       !(I->src[1].cache && I->src[1].discard)) {
      const struct agx3_shift_add_desc d = {
         .dest = I->dest[0].value,
         .dest_hint = I->dest[0].cache,
         .src0_immediate = true,
         .src0 = I->src[0].value,
         .src1 = I->src[1].value,
         .src1_kill = I->src[1].discard,
         .shift = I->shift,
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_WRITEOUT_MAX_LENGTH], len;
      if (!agx3_shift_add16_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->dest[0].size == AGX_SIZE_16 && I->src[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       agx3_src_is_register(I->dest[0]) && agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       !I->src[0].abs && !I->src[0].neg && !I->src[1].abs && !I->src[1].neg) {
      const uint16_t src[2] = {I->src[0].value, I->src[1].value};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (I->src[0].kill || I->src[1].kill ||
          !agx3_alu_shift_add16(enc, I->dest[0].value, src, hints, I->shift,
                                I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Narrow iadd_sat and uadd_sat. ABS marks both halves unsigned;
    * Apple's saturating iadd instead marks signed halves, so the
    * signedness is a property of the whole instruction. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
       I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_16 && !I->src[0].neg &&
       I->src[1].size == AGX_SIZE_16 && !I->src[1].neg) {
      const agx_index dest = I->dest[0];
      const agx_index a = I->src[0], b = I->src[1];
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || a.memory || b.memory ||
          a.abs != b.abs || a.kill || b.kill ||
          agx_channels(a) != 1 || agx_channels(b) != 1 ||
          (a.cache && a.discard) || (b.cache && b.discard))
         AGX3_REFUSE_RET(I);

      const uint16_t src[2] = {a.value, b.value};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(a), agx3_post_ra_hint(b)};
      uint16_t w = st->wait;
      if (!agx3_retire_extra_waits(emission, enc, st->sl, st->res, st->instr_index,
                                   I, &w))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_add_sat16(enc, dest.value, src, hints, !a.abs,
                              dest.cache, w, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* AGX3C narrow iadd_sat/uadd_sat and isub_sat/usub_sat (subtrahend
    * negated): C's iadd32/isub32 point with half widths. ABS marks both
    * halves unsigned, as on G17P. */
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
       I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_16 && !I->src[0].neg &&
       I->src[1].size == AGX_SIZE_16) {
      const agx_index dest = I->dest[0];
      const agx_index a = I->src[0], b = I->src[1];
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || a.memory || b.memory ||
          a.abs != b.abs || a.kill || b.kill ||
          agx_channels(a) != 1 || agx_channels(b) != 1 ||
          (a.cache && a.discard) || (b.cache && b.discard))
         AGX3_REFUSE_RET(I);

      const uint16_t src[2] = {a.value, b.value};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(a), agx3_post_ra_hint(b)};
      uint16_t w = st->wait;
      if (!agx3_retire_extra_waits(emission, enc, st->sl, st->res, st->instr_index,
                                   I, &w))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_addsub_sat16_m5(dest.value, src, hints, !a.abs,
                                    b.neg, dest.cache, w, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* NIR's narrow isub is iadd(a, -b), and ineg is iadd(#0, -a). Apple
    * emits both as the scalar-half isub, the negation with an
    * immediate-zero minuend, so neither needs a widened temporary. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && I->dest[0].size == AGX_SIZE_16 &&
       agx3_src_is_register(I->src[1]) &&
       I->src[1].size == AGX_SIZE_16 && I->src[1].neg &&
       (agx3_src_is_zero_imm(I->src[0]) ||
        (agx3_src_is_register(I->src[0]) &&
         I->src[0].size == AGX_SIZE_16))) {
      const agx_index dest = I->dest[0];
      const agx_index minuend = I->src[0], subtrahend = I->src[1];
      const bool negate = minuend.type == AGX_INDEX_IMMEDIATE;
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || minuend.memory || minuend.abs ||
          minuend.neg || minuend.kill ||
          (minuend.cache && minuend.discard) ||
          (!negate && agx_channels(minuend) != 1) ||
          subtrahend.memory || subtrahend.abs || subtrahend.kill ||
          agx_channels(subtrahend) != 1 ||
          (subtrahend.cache && subtrahend.discard))
         AGX3_REFUSE_RET(I);

      uint16_t w = st->wait;
      if (enc == AGX_ISA_ENCODING_AGX3B &&
          !agx3_retire_extra_waits(emission, enc, st->sl, st->res, st->instr_index,
                                   I, &w))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      bool encoded;
      if (negate) {
         encoded = agx3_alu_neg16(enc, dest.value, subtrahend.value,
                                  agx3_post_ra_hint(subtrahend),
                                  dest.cache, w, bytes, &len);
      } else {
         const uint16_t src[2] = {minuend.value, subtrahend.value};
         const enum agx3_alu_hint hints[2] = {
            agx3_post_ra_hint(minuend), agx3_post_ra_hint(subtrahend)};
         encoded = agx3_alu_sub16(enc, dest.value, src, hints,
                                  dest.cache, w, bytes, &len);
      }
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Register arithmetic shares the original canonical form, but its
    * native source hints come from post-RA lifetime.
    *
    * The SUBTRACT additionally carries a shift on slot B (f54). NIR
    * strength-reduces a multiply by a small negative constant into
    * `a - (b << N)` -- `x * -7` becomes `x - (x << 3)` -- so a shifted
    * subtract is an ordinary shape, and refusing it failed every such
    * multiply closed. The ADD keeps its `shift == 0` restriction: its
    * shifted form is modelled separately by agx3_iadd_encode, and
    * re-routing a shape that already encodes is what cost two hardware
    * probes before.
    *
    * G17G takes this path too, at shift zero. Reaching the generic
    * select template instead pinned f27 -- a first-operand discard --
    * as if it were a form constant, so a register read a second time
    * after the add read dead. `vec4(texture(s, uv).r)` is enough to
    * show it: the broadcast becomes two moves off one sampled
    * register, and only the first sees the value. Apple's own G17S
    * corpus carries iadd32 with every live/dead combination of
    * f27/f42, so the post-RA hints go in the instruction. */
   if ((enc == AGX_ISA_ENCODING_AGX3B ||
        (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         I->shift == 0)) &&
       I->op == AGX_OPCODE_IADD &&
       (I->shift == 0 || I->src[1].neg) &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       !I->saturate && agx3_mod_src_ok(I->dest[0]) &&
       agx3_mod_src_ok(I->src[0]) && agx3_mod_src_ok(I->src[1]) &&
       !I->src[0].abs && !I->src[0].neg && !I->src[1].abs) {
      const enum agx3_alu_op op = I->src[1].neg ? AGX3_ALU_ISUB
                                              : AGX3_ALU_IADD;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (I->src[0].kill || I->src[1].kill ||
          (I->src[0].cache && I->src[0].discard) ||
          (I->src[1].cache && I->src[1].discard) ||
          !agx3_alu_integer_reg_shift_cache(
             enc, op, I->dest[0].value, src, hints, (uint8_t)I->shift,
             enc == AGX_ISA_ENCODING_AGX3B,
             st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A local-memory byte offset widens the 16-bit local invocation
    * index with `iadd dst32, #0, src16.abs`. The native integer iadd
    * immediate form does not admit zero in its witnessed endpoint, but
    * zero-extension needs no new arithmetic encoding: copy the low half
    * and clear the high half, exactly like the existing widened BITOP
    * move below. The copy carries the source's real wait/lifetime. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && agx3_src_is_register(I->dest[0]) &&
       !I->dest[0].memory && I->dest[0].size == AGX_SIZE_32 &&
       agx_channels(I->dest[0]) == 1 && !(I->dest[0].value & 1)) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index zero = I->src[imm_first ? 0 : 1];
      const agx_index half = I->src[imm_first ? 1 : 0];
      /* Post-RA discard is the physical lifetime bit consumed by COPY;
       * SSA kill is allocator bookkeeping and must not reject the same
       * physical source shape. */
      if (zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0 &&
          !zero.memory && !zero.abs && !zero.neg &&
          agx3_src_is_register(half) && !half.memory && half.abs &&
          !half.neg && half.size == AGX_SIZE_16 &&
          agx_channels(half) == 1 && !(half.cache && half.discard)) {
         uint8_t lo[AGX3_COPY_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
         uint8_t lo_len = 0, hi_len = 0;
         if (!agx3_copy_encode_reg16(enc, I->dest[0].value, half.value,
                                     I->dest[0].cache, half.cache,
                                     half.discard, st->wait, lo, &lo_len) ||
             !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                       &hi_len) ||
             !agx3_append(emission, lo, lo_len) ||
             !agx3_append(emission, hi, hi_len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       !I->shift && agx3_half_reg(I->dest[0]) &&
       agx3_half_reg(I->src[0]) && !I->src[0].abs && !I->src[0].neg &&
       agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].neg &&
       !I->src[1].abs && !I->src[1].memory) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_isub_imm16_m5(I->dest[0].value, I->src[0].value,
              I->src[1].value, agx3_post_ra_hint(I->src[0]),
              I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_half_reg(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
      const bool swap = I->src[1].type == AGX_INDEX_IMMEDIATE;
      const agx_index a = I->src[swap ? 1 : 0], b = I->src[swap ? 0 : 1];
      if (swap && I->shift)
         AGX3_REFUSE_RET(I);
      if ((a.type == AGX_INDEX_IMMEDIATE || agx3_half_reg(a)) &&
          agx3_half_reg(b) && !a.abs && !a.neg && !b.abs && !b.neg &&
          !a.memory && !b.memory && !a.kill && !b.kill &&
          agx_channels(a) == 1 && agx_channels(b) == 1) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_iadd16_gpu1(I->dest[0].value, I->dest[0].cache,
                 a.type == AGX_INDEX_IMMEDIATE, a.value,
                 a.type == AGX_INDEX_IMMEDIATE ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(a),
                 b.value, agx3_post_ra_hint(b), I->shift, st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift <= 4 && agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].neg && !I->src[1].neg &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       (I->src[0].size == AGX_SIZE_16 || I->src[0].size == AGX_SIZE_32) &&
       (I->src[1].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_32) &&
       (I->src[0].size == AGX_SIZE_16 || !I->src[0].abs) &&
       (I->src[1].size == AGX_SIZE_16 || !I->src[1].abs)) {
      const struct agx3_iadd_desc d = {
         .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
         .src_a_is_register = true, .src_a_reg = I->src[0].value,
         .src_a_16 = I->src[0].size == AGX_SIZE_16,
         .src_a_sign_extend = I->src[0].size == AGX_SIZE_16 && !I->src[0].abs,
         .src_a_cache = I->src[0].cache, .src_a_kill = I->src[0].discard,
         .src = I->src[1].value, .src_16 = I->src[1].size == AGX_SIZE_16,
         .sign_extend = I->src[1].size == AGX_SIZE_16 && !I->src[1].abs,
         .src_cache = I->src[1].cache, .src_kill = I->src[1].discard,
         .shift = I->shift, .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], len;
      if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && agx3_vary_reg32(I->dest[0]) &&
       agx3_vary_reg32(I->src[0]) && agx3_mod_src_ok(I->src[1]) &&
       agx_channels(I->dest[0]) == 1 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].abs && !I->src[0].neg &&
       !I->src[1].abs) {
      const struct agx3_iadd_desc d = {
         .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
         .src_a_is_register = true, .src_a_reg = I->src[0].value,
         .src_a_cache = I->src[0].cache,
         .src_a_kill = I->src[0].discard,
         .src = I->src[1].value, .src_cache = I->src[1].cache,
         .src_kill = I->src[1].discard, .wait_mask = st->wait,
         .subtract = I->src[1].neg,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
      if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * `iadd dst32, #imm, src << shift` -- constant address arithmetic,
    * which is what every shader with more than one memory access is
    * built out of. The source may be a 16-bit half, zero- or
    * sign-extended (abs and neg respectively, as on AGX2).
    *
    * This carries its own scoreboard wait, so it does not need the
    * wait-retiring prelude and does not disturb the operands.
    */
   /*
    * G17G/G18G register-plus-immediate adds with the register in slot A,
    * the two shapes Apple emits that way (agx3_alu_iadd32_reg_imm_gpu1):
    * NIR's uadd_sat against a constant -- IADD.sat with ABS marking
    * the unsigned register, as agx_compile.c spells it -- and
    * agx_lower_pseudo()'s sign extension of a half, `iadd d, h, #0`.
    * Addition commutes, so a leading immediate is taken too.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && I->shift == 0 &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index reg = I->src[imm_first ? 1 : 0];
      const agx_index imm = I->src[imm_first ? 0 : 1];
      const bool reg_ok =
         agx3_src_is_register(reg) && !reg.memory && !reg.neg &&
         !reg.kill && !(reg.cache && reg.discard) &&
         agx_channels(reg) == 1 &&
         (reg.size == AGX_SIZE_16 ||
          (reg.size == AGX_SIZE_32 && !(reg.value & 1)));
      const bool imm_ok = agx3_plain_immediate(imm) && !imm.neg &&
                          imm.value <= 255;
      /* The immediate carries no sign, so ABS on it is a no-op. */
      const bool unsigned_sat = I->saturate &&
                                reg.size == AGX_SIZE_32 && reg.abs;
      const bool sign_extend = !I->saturate &&
                               reg.size == AGX_SIZE_16 && !reg.abs &&
                               imm_ok && imm.value == 0;
      if (reg_ok && imm_ok && (unsigned_sat || sign_extend)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_iadd32_reg_imm_gpu1(
                I->dest[0].value, I->dest[0].cache, reg.value,
                reg.size == AGX_SIZE_16, sign_extend,
                agx3_post_ra_hint(reg), (uint8_t)imm.value,
                unsigned_sat, st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * G17G saturating add and subtract over two 32-bit registers.
    * agx_compile.c spells NIR's four saturating ops as IADD.sat with
    * ABS marking an unsigned operand and NEG the subtrahend, so the
    * operand modifiers map onto Apple's isub32/iadd32 signedness bits
    * and the saturate field (gpu/m5-xfb-alu-reference.metal, -alu2-,
    * -alu3-). XFB's bounds-safe counter arithmetic is usub_sat.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
       I->shift == 0 && agx3_src_is_register(I->dest[0]) &&
       !I->dest[0].memory && I->dest[0].size == AGX_SIZE_32 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       !I->src[0].neg && !(I->src[0].cache && I->src[0].discard) &&
       agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
       I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
       !(I->src[1].cache && I->src[1].discard)) {
      const struct agx3_iadd_desc d = {
         .dest = I->dest[0].value,
         .dest_cache = I->dest[0].cache,
         .src_a_is_register = true,
         .src_a_reg = I->src[0].value,
         .src_a_kill = I->src[0].discard,
         .src_a_sign_extend_32 = !I->src[0].abs,
         .src = I->src[1].value,
         .src_kill = I->src[1].discard,
         .sign_extend_32 = !I->src[1].abs,
         .subtract = I->src[1].neg,
         .saturate = true,
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH];
      uint8_t len = 0;
      if (!(agx_is_g18g(&ctx->key->dev)
             ? agx3_iadd_encode_g18g
             : agx3_iadd_encode)(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_IADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 &&
       agx_channels(I->dest[0]) == 1) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index constant = imm_first ? I->src[0] : I->src[1];
      const agx_index reg = imm_first ? I->src[1] : I->src[0];

      /* Native G17G sign extension keeps the half in slot A and
       * adds immediate zero in slot B, in one instruction. */
      if (enc == AGX_ISA_ENCODING_AGX3C && !I->saturate && !I->shift &&
          agx3_plain_immediate(constant) && constant.value == 0 &&
          agx3_half_reg(reg) && agx_channels(reg) == 1 && !reg.neg &&
          !reg.kill && !(reg.cache && reg.discard)) {
         const struct agx3_isub_imm_desc d = {
            .dest = I->dest[0].value, .dest_cache = I->dest[0].cache,
            .src = reg.value, .src_16 = true, .sign_extend = !reg.abs,
            .src_cache = reg.cache, .src_kill = reg.discard,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH], len;
         if (!agx3_iadd_immediate_encode(enc, &d, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }

      /* iadd_sat, uadd_sat, isub_sat and usub_sat over 32-bit
       * registers: the register iadd/isub point with f53. ABS on both
       * operands selects unsigned saturation; otherwise both sign
       * fields are set. */
      if ((enc == AGX_ISA_ENCODING_AGX3B ||
           enc == AGX_ISA_ENCODING_AGX3C) && I->saturate) {
         agx_index a = I->src[0], b = I->src[1];
         if (a.neg && !b.neg) {
            const agx_index t = a;
            a = b;
            b = t;
         }
         if (I->shift != 0 || !agx3_src_is_register(a) ||
             !agx3_src_is_register(b) || a.memory || b.memory ||
             a.size != AGX_SIZE_32 || b.size != AGX_SIZE_32 ||
             agx_channels(a) != 1 || agx_channels(b) != 1 ||
             a.neg || a.kill || b.kill || a.abs != b.abs ||
             (a.cache && a.discard) || (b.cache && b.discard))
            AGX3_REFUSE_RET(I);

         uint16_t w = st->wait;
         if (!agx3_retire_extra_waits(emission, enc, st->sl, st->res,
                                      st->instr_index, I, &w))
            AGX3_REFUSE_RET(I);
         const struct agx3_iadd_desc d = {
            .dest = I->dest[0].value,
            .dest_cache = enc == AGX_ISA_ENCODING_AGX3B || I->dest[0].cache,
            .src_a_is_register = true,
            .src_a_reg = a.value,
            .src_a_kill = a.discard,
            .src_a_cache = a.cache,
            .src_a_sign_extend = !a.abs,
            .src = b.value,
            .src_kill = b.discard,
            .src_cache = b.cache,
            .sign_extend = !b.abs,
            .subtract = b.neg,
            .saturate = true,
            .wait_mask = w,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if (!(agx_is_g18g(&ctx->key->dev)
                ? agx3_iadd_encode_g18g
                : agx3_iadd_encode)(enc, &d, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }

      /* reg - #imm. A narrow register is sign-extended when bare and
       * zero-extended under ABS; NEG on it would negate the register
       * too, which this form does not express. */
      if (!imm_first && I->shift == 0 && !I->saturate &&
          constant.type == AGX_INDEX_IMMEDIATE && constant.neg &&
          !constant.abs && !constant.memory && constant.value <= 0xff &&
          agx3_src_is_register(reg) && !reg.memory && !reg.kill &&
          agx_channels(reg) == 1 && !(reg.cache && reg.discard) &&
          !reg.neg &&
          ((reg.size == AGX_SIZE_32 && !reg.abs) ||
           reg.size == AGX_SIZE_16)) {
         const struct agx3_isub_imm_desc d = {
            .dest=I->dest[0].value, .src=reg.value,
            .immediate=constant.value, .src_16=reg.size==AGX_SIZE_16,
            .sign_extend=reg.size==AGX_SIZE_16 && !reg.abs,
            .src_cache=reg.cache,
            .src_kill=reg.discard, .wait_mask=st->wait,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH],len;
         if (agx3_isub_immediate_encode(enc,&d,bytes,&len) &&
             agx3_append(emission,bytes,len))
            return AGX3_PACK_EMITTED;
      }

      /* A bare narrow iadd source is sign-extended (ABS denotes
       * unsigned): SIGNEXT's `iadd dst32, src16, #0`, the `src16 << 4`
       * form and a non-zero immediate addend alike. Other bare
       * half/shift combinations are refused. */
      const bool signed_half_widen =
         (enc == AGX_ISA_ENCODING_AGX3B ||
          (enc == AGX_ISA_ENCODING_AGX3C && I->shift == 4)) &&
         !I->saturate &&
         constant.type == AGX_INDEX_IMMEDIATE &&
         reg.size == AGX_SIZE_16 && !reg.abs && !reg.neg &&
         (I->shift == 0 || I->shift == 4);
      /* #imm - reg is AGX IR's iadd(#imm, -reg). NEG is a negation, not
       * a sign extension (on AGX2 too): it selects Apple's isub with
       * the immediate in slot A, `16 - int(short b)` being
       * 1f 01 56 00 02 20 00 08 15 0c. Before, a negated register was
       * added, and a negated half was added sign-extended. Only the
       * unshifted subtraction is witnessed. */
      const bool imm_minus_reg =
         (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
         !I->saturate &&
         reg.neg && I->shift == 0 &&
         (reg.size == AGX_SIZE_16 ||
          (reg.size == AGX_SIZE_32 && !reg.abs));
      if (constant.type == AGX_INDEX_IMMEDIATE && !constant.abs &&
          !constant.neg && constant.value <= 0xff && !I->saturate &&
          agx3_src_is_register(reg) && !reg.memory &&
          agx_channels(reg) == 1 && (!reg.neg || imm_minus_reg) &&
          ((reg.size == AGX_SIZE_32 && !reg.abs) ||
           (reg.size == AGX_SIZE_16 &&
            (reg.abs || reg.neg || signed_half_widen)))) {
         const struct agx3_iadd_desc d = {
            .dest = I->dest[0].value,
            .dest_cache = true,
            .immediate = (uint8_t)constant.value,
            .src = reg.value,
            .src_kill = reg.discard,
            /* G18G's isub32 alone carries the source's cache hint. */
            .src_cache = enc == AGX_ISA_ENCODING_AGX3C && reg.cache &&
                         (!agx_is_g18g(&ctx->key->dev) || reg.neg),
            .src_16 = reg.size == AGX_SIZE_16,
            .sign_extend = reg.size == AGX_SIZE_16 && !reg.abs,
            .shift = I->shift,
            .subtract = reg.neg,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if ((agx_is_g18g(&ctx->key->dev)
               ? agx3_iadd_encode_g18g
               : agx3_iadd_encode)(enc, &d, bytes, &len) &&
             agx3_append(emission, bytes, len))
            return AGX3_PACK_EMITTED;
         /* Outside the witnessed domain; fall through to refuse. */
      }

      /*
       * Register + (register << shift). This is the second half of the
       * shifted-imad split in agx_emit_imadshl_agx(): AGX3 has no
       * shifted imad, so the product is added here with the shift on
       * the addend. Apple emits exactly this, and f54 carries the
       * shift (witnessed at 1..4).
       */
      const agx_index original_a = I->src[0];
      const agx_index original_b = I->src[1];
      /* Integer add commutes: put a reversed signed-half accumulator in
       * the slot-A32 + slot-B16 form. */
      const bool commute_signed_half =
         enc == AGX_ISA_ENCODING_AGX3B &&
         original_a.size == AGX_SIZE_16 &&
         !original_a.abs && !original_a.neg &&
         original_b.size == AGX_SIZE_32 &&
         !original_b.abs && !original_b.neg;
      const agx_index a = commute_signed_half ? original_b : original_a;
      const agx_index bsrc = commute_signed_half ? original_a : original_b;

      /*
       * Only take this path for a SHIFTED add. An unshifted register
       * add already had a working encoding through the select core, and
       * re-routing it here changed its bytes -- which cost two compute
       * probes on hardware. New encodings belong on shapes that were
       * previously refused, not on shapes that already worked.
       */
      /* A 16-bit struct index scales into an address the same way:
       * the half addend is sign-extended when bare and zero-extended
       * under ABS, through slot B's own width and sign fields. */
      const bool shifted_half =
         enc == AGX_ISA_ENCODING_AGX3B && bsrc.size == AGX_SIZE_16 &&
         !bsrc.neg;
      /* a - (b << n): AGX3c's isub32 takes the shifted operand the
       * same way (agx3_iadd_encode, m6-isub-shift). */
      const bool shifted_sub = enc == AGX_ISA_ENCODING_AGX3C &&
                               ctx->key->dev.emit_agx3d &&
                               bsrc.neg && !bsrc.abs &&
                               bsrc.size == AGX_SIZE_32;
      if (I->shift != 0 &&
          agx3_src_is_register(a) && !a.memory && !a.abs && !a.neg &&
          a.size == AGX_SIZE_32 && agx_channels(a) == 1 &&
          agx3_src_is_register(bsrc) && !bsrc.memory &&
          (shifted_half || shifted_sub ||
           (!bsrc.abs && !bsrc.neg && bsrc.size == AGX_SIZE_32)) &&
          agx_channels(bsrc) == 1) {
         uint16_t w = st->wait;
         if (shifted_half &&
             !agx3_retire_extra_waits(emission, enc, st->sl, st->res,
                                      st->instr_index, I, &w))
            AGX3_REFUSE_RET(I);
         const struct agx3_iadd_desc d = {
            .dest_cache = true,
            .dest = I->dest[0].value,
            .src_a_is_register = true,
            .src_a_reg = a.value,
            .src_a_kill = a.discard,
            .src = bsrc.value,
            .src_kill = bsrc.discard,
            .src_16 = shifted_half,
            .sign_extend = shifted_half && !bsrc.abs,
            .shift = I->shift,
            .subtract = shifted_sub,
            .wait_mask = w,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if ((agx_is_g18g(&ctx->key->dev)
               ? agx3_iadd_encode_g18g
               : agx3_iadd_encode)(enc, &d, bytes, &len) &&
             agx3_append(emission, bytes, len))
            return AGX3_PACK_EMITTED;
      }

      /*
       * Register +/- register with at least one 16-bit operand. As on
       * AGX2, a narrow operand is zero-extended under ABS and
       * sign-extended otherwise, and NEG negates it: iadd(a, -b) is
       * Apple's isub. The all-32-bit unshifted add keeps the
       * select-core encoding.
       */
      agx_index na = original_a, nb = original_b;
      /* Addition commutes: keep a negated operand in slot B, and a 32-bit
       * operand in slot A of a mixed-width add. */
      if ((na.neg && !nb.neg) ||
          (!na.neg && !nb.neg && !na.abs && !nb.abs &&
           na.size == AGX_SIZE_16 && nb.size == AGX_SIZE_32)) {
         const agx_index t = na;
         na = nb;
         nb = t;
      }
      const bool na_ok =
         agx3_src_is_register(na) && !na.memory &&
         agx_channels(na) == 1 && !na.neg &&
         !(na.cache && na.discard) &&
         (na.size == AGX_SIZE_16 ||
          (na.size == AGX_SIZE_32 && !na.abs));
      const bool nb_ok =
         agx3_src_is_register(nb) && !nb.memory &&
         agx_channels(nb) == 1 &&
         !(nb.cache && nb.discard) &&
         (nb.size == AGX_SIZE_16 ||
          (nb.size == AGX_SIZE_32 && !nb.abs));

      if (I->shift == 0 && !I->saturate && na_ok && nb_ok &&
          (na.size == AGX_SIZE_16 || nb.size == AGX_SIZE_16)) {
         const bool b3 = enc == AGX_ISA_ENCODING_AGX3B;
         uint16_t w = st->wait;
         if (b3 && !agx3_retire_extra_waits(emission, enc, st->sl, st->res,
                                            st->instr_index, I, &w))
            AGX3_REFUSE_RET(I);
         const struct agx3_iadd_desc d = {
            .dest = I->dest[0].value,
            .dest_cache = true,
            .src_a_is_register = true,
            .src_a_reg = na.value,
            .src_a_kill = na.discard,
            /* Apple's operand cache hints, on the subtraction this
             * path newly encodes. The additions it already emitted
             * keep their hardware-proven bytes without them. */
            .src_a_cache = nb.neg && na.cache,
            .src_a_16 = na.size == AGX_SIZE_16,
            .src_a_sign_extend = na.size == AGX_SIZE_16 && !na.abs,
            .src = nb.value,
            .src_kill = nb.discard,
            .src_cache = nb.neg && nb.cache,
            .src_16 = nb.size == AGX_SIZE_16,
            .sign_extend = nb.size == AGX_SIZE_16 && !nb.abs,
            .subtract = nb.neg,
            .wait_mask = w,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if (!(agx_is_g18g(&ctx->key->dev)
                ? agx3_iadd_encode_g18g
                : agx3_iadd_encode)(enc, &d, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * A 32-bit register move. agx_lower_pseudo() spells MOV as this
    * BITOP, and G17P routes it through the select core's
    * `iadd d, s, #0`. G17G has the same grounded immediate add
    * (agx3_iadd_core.c's applegpu_g17s `iadd32` ladder), so use it
    * with a zero addend rather than an unwitnessed 32-bit COPY.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       !(I->src[0].cache && I->src[0].discard) &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1) {
      const struct agx3_iadd_desc d = {
         .dest = I->dest[0].value,
         .dest_cache = I->dest[0].cache,
         .immediate = 0,
         .src = I->src[0].value,
         .src_kill = I->src[0].discard,
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH];
      uint8_t len = 0;
      if (!(agx_is_g18g(&ctx->key->dev)
             ? agx3_iadd_encode_g18g
             : agx3_iadd_encode)(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * Widening move: a 32-bit destination fed by a 16-bit half. Apple
    * does not have a widening `copy`; its compiler moves the half and
    * clears the upper one, which is what a zero-extend is. (Its ALU can
    * also just consume a 16-bit operand in a 32-bit op -- the width is
    * per-operand -- but that folds into the consumer and there is no
    * consumer here.) Both halves of this pair are witnessed encodings.
    * The half-copy corpus does not map every combination of scoreboard
    * waits. Preserve its directly encodable waits. For any other mask,
    * retire the producer through the grounded band no-op first, as for
    * uniform copies and derivatives, then emit the exact zero-wait
    * copy-and-clear pair.
    */
   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].value & 1) == 0) {
      /* The half COPY names a seven-bit destination. Above it, add the
       * zero-extended half to an immediate 0: agx3_iadd_encode()'s
       * narrow slot-B source form, as Apple emits for
       * `uint(ushort a[i]) + 7`, names an eight-bit destination. */
      if (I->dest[0].value >= AGX3_COPY_REG16_DEST_HALVES) {
         const struct agx3_iadd_desc d = {
            .dest = I->dest[0].value,
            .dest_cache = true,
            .immediate = 0,
            .src = I->src[0].value,
            .src_kill = I->src[0].discard,
            .src_16 = true,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_IADD_MAX_LENGTH];
         uint8_t len = 0;
         if (!(agx_is_g18g(&ctx->key->dev)
                ? agx3_iadd_encode_g18g
                : agx3_iadd_encode)(enc, &d, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      uint8_t lo[AGX3_COPY_MAX_LENGTH], hi[AGX3_STAGE_MAX_LENGTH];
      uint8_t lo_len = 0, hi_len = 0;
      bool copied = agx3_copy_encode_reg16(
         enc, I->dest[0].value, I->src[0].value, I->dest[0].cache,
         I->src[0].cache, I->src[0].discard, st->wait, lo, &lo_len);
      bool retired = false;
      if (!copied && st->wait) {
         retired = agx3_retire_result_wait(emission, enc, I, st->wait);
         copied = retired && agx3_copy_encode_reg16(
                               enc, I->dest[0].value, I->src[0].value,
                               I->dest[0].cache, I->src[0].cache,
                               I->src[0].discard, 0, lo, &lo_len);
      }
      if (!copied ||
          !agx3_clr(enc, I->dest[0].value + 1, true, false, hi,
                    &hi_len) ||
          !agx3_append(emission, lo, lo_len) ||
          !agx3_append(emission, hi, hi_len))
         AGX3_REFUSE_RET(I);
      if (retired)
         st->outstanding &= (uint16_t)~st->wait;
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx3_vary_reg32(I->src[0]) &&
       agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1 &&
       agx3_src_is_zero_imm(I->src[1]) && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard)) {
      const struct agx3_iadd_desc d = {
         .dest=I->dest[0].value, .dest_cache=I->dest[0].cache,
         .src=I->src[0].value, .src_cache=I->src[0].cache,
         .src_kill=I->src[0].discard, .wait_mask=st->wait,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
      if (!agx3_iadd_encode(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A narrowing integer move copies the low half of its 32-bit
    * source. The same native 16-bit COPY used above names that low half
    * directly, so no arithmetic conversion or new encoding is needed. */
   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_COPY_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_copy_encode_reg16(
             enc, I->dest[0].value, I->src[0].value,
             I->dest[0].cache, I->src[0].cache,
             I->src[0].discard, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV && I->nr_dests == 1 &&
       I->nr_srcs == 2 && agx3_src_is_register(I->src[0]) &&
       !I->src[0].memory && !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_COPY_MAX_LENGTH];
      uint8_t len = 0;
      if (agx3_copy_encode_reg32(enc, I->dest[0].value, I->src[0].value,
                                 I->dest[0].cache, I->src[0].cache,
                                 I->src[0].discard, st->wait, bytes, &len)) {
         if (!agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * The same lowering at 16 bits, register to register: Mesa keeps
    * booleans in halves and the select core's 32-bit `iadd d, s, #0`
    * move cannot express them. Apple copies a half with `copy`.
    */
   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      bool encoded = agx3_copy_encode_reg16(
         enc, I->dest[0].value, I->src[0].value, I->dest[0].cache,
         I->src[0].cache, I->src[0].discard, st->wait, bytes, &len);
      if (!encoded)
         encoded = agx3_alu_bit_imm(
            enc, AGX3_ALU_BOR, true, I->dest[0].value,
            I->src[0].value, 0, agx3_post_ra_hint(I->src[0]), st->wait,
            bytes, &len);
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_32 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_COPY_MAX_LENGTH];
      uint8_t len = 0;
      /* Apple's uniform->GPR copy witness has no wait field. Retire a
       * pending producer through the already-grounded band no-op, then
       * emit that exact copy instead of guessing scoreboard bits on a
       * form that has none in the corpus. Meta image copies reach this
       * when global IDs and pushed extents share a scoreboard slot. */
      const bool retired =
         st->wait == 0 || agx3_retire_result_wait(emission, enc, I, st->wait);
      if (!retired ||
          !agx3_copy_encode_uniform_to_reg32(enc, I->dest[0].value,
                                             I->src[0].value, bytes,
                                             &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~st->wait;
      return AGX3_PACK_EMITTED;
   }

   /* The same move at 16 bits. Mesa keeps booleans and halves in
    * 16-bit registers, so agx_lower_uniform_sources() emits this shape
    * for every narrow generic-ALU uniform source; without an encoder
    * it stayed a bitop reading the uniform file and the packer refused
    * the whole shader. Apple's own corpus carries the form, and it
    * likewise has no wait field, so retire through the grounded band
    * no-op first. */
   if (I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg &&
       I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_COPY_MAX_LENGTH];
      uint8_t len = 0;
      const bool retired =
         st->wait == 0 || agx3_retire_result_wait(emission, enc, I, st->wait);
      if (!retired ||
          !agx3_copy_encode_uniform_to_reg16(enc, I->dest[0].value,
                                             I->src[0].value,
                                             I->dest[0].cache, bytes,
                                             &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~st->wait;
      return AGX3_PACK_EMITTED;
   }

   /*
    * Fused integer compare-select. agx_lower_pseudo() turns every
    * AGX_OPCODE_ICMP into one of these, so without it no shader with an
    * integer comparison compiles at all. Apple's `isel` is the same
    * instruction with the same operand order and the same condition
    * numbering (probe-witnessed; see agx3_isel_core.h).
    */
   if (I->op == AGX_OPCODE_ICMPSEL && I->nr_dests == 1 &&
       I->nr_srcs == 4) {
      struct agx3_isel_desc d;
      memset(&d, 0, sizeof(d));

      const agx_index dest = I->dest[0];
      agx_index a = I->src[0], b = I->src[1];

      /* A 16-bit register compared against a 32-bit one is Mesa's
       * folded u2u32 (agx_optimizer copy-propagates the widening
       * mov, so the narrow value zero-extends as on every AGX). GPU1
       * encodes that inside the select with the wide comparand first,
       * reversing the condition when the source had the narrow one
       * first, exactly as Apple's compiler orders it
       * (gpu/m5-mixed-compare-reference.metal). */
      const bool narrow_first =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         agx3_src_is_register(a) &&
         agx3_src_is_register(b) && a.size == AGX_SIZE_16 &&
         b.size == AGX_SIZE_32;
      const bool swap_compare =
         a.type == AGX_INDEX_IMMEDIATE || narrow_first;
      if (swap_compare) {
         const agx_index tmp = a;
         a = b;
         b = tmp;
      }
      const agx_index t = I->src[2], f = I->src[3];
      const bool narrow_second =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
         agx3_src_is_register(a) &&
         agx3_src_is_register(b) && a.size == AGX_SIZE_32 &&
         b.size == AGX_SIZE_16;

      const bool dest_ok =
         agx3_src_is_register(dest) && !dest.memory &&
         agx_channels(dest) == 1 &&
         (dest.size == AGX_SIZE_32 || dest.size == AGX_SIZE_16);
      /* One aligned 32-bit uniform comparand is native beside a
       * 32-bit register (agx3_isel_core.c); agx_lower_uniform_sources
       * keeps no other uniform compare shape. */
      const bool a_uniform = a.type == AGX_INDEX_UNIFORM;
      const bool b_uniform = b.type == AGX_INDEX_UNIFORM;
      const bool a_reg32 = agx3_src_is_register(a) &&
                           a.size == AGX_SIZE_32 &&
                           agx_channels(a) == 1;
      const bool b_reg32 = agx3_src_is_register(b) &&
                           b.size == AGX_SIZE_32 &&
                           agx_channels(b) == 1;
      const bool cmp_ok = !a.memory && !a.abs && !a.neg &&
         !b.memory && !b.abs && !b.neg &&
         ((a_uniform && a.size == AGX_SIZE_32 && b_reg32) ||
          (b_uniform && b.size == AGX_SIZE_32 && a_reg32) ||
          (agx3_src_is_register(a) &&
           (a.size == AGX_SIZE_32 || a.size == AGX_SIZE_16) &&
           agx_channels(a) == 1 &&
           ((b.type == AGX_INDEX_IMMEDIATE && b.value <= 255) ||
            (agx3_src_is_register(b) &&
             (b.size == a.size ||
              (ctx->key->dev.emit_agx3d ? narrow_second
               : enc == AGX_ISA_ENCODING_AGX3C &&
                    (b.size == AGX_SIZE_16 || b.size == AGX_SIZE_32))) &&
             agx_channels(b) == 1))));

      if (!dest_ok || !cmp_ok)
         AGX3_REFUSE_RET(I);

      d.dest = dest.value;
      d.dest_16 = dest.size == AGX_SIZE_16;
      d.cmp0 = a.value;
      d.cmp1 = b.value;
      d.cmp1_immediate = b.type == AGX_INDEX_IMMEDIATE;
      d.cmp0_uniform = a_uniform;
      d.cmp1_uniform = b_uniform;
      d.cmp_16 = a.size == AGX_SIZE_16;
      if (ctx->key->dev.emit_agx3d)
         d.cmp1_16 = narrow_second;
      else
         d.cmp_mixed_width = b.type != AGX_INDEX_IMMEDIATE && b.size != a.size;
      d.explicit_hints = true;
      d.dest_cache = dest.cache;
      const agx_index hinted[4] = {a, b, t, f};
      for (unsigned n = 0; n < 4; ++n) {
         if (hinted[n].type == AGX_INDEX_REGISTER) {
            d.cache[n] = hinted[n].cache;
            d.discard[n] = hinted[n].discard;
         }
      }

      /* AGX3 has one equality condition, 4. Mesa spells the same
       * comparison two ways (UEQ = 0 and SEQ = 4) because AGX2 did;
       * the probes compiled `uint ==` and `int ==` and Apple emitted 4
       * for both, equality being sign-agnostic. Every other condition
       * keeps its number. */
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         d.cond = AGX3_ISEL_ICOND_EQ;
         break;
      case AGX_ICOND_ULT:
         d.cond = AGX3_ISEL_ICOND_ULT;
         break;
      case AGX_ICOND_UGT:
         d.cond = AGX3_ISEL_ICOND_UGT;
         break;
      case AGX_ICOND_SLT:
         d.cond = AGX3_ISEL_ICOND_SLT;
         break;
      case AGX_ICOND_SGT:
         d.cond = AGX3_ISEL_ICOND_SGT;
         break;
      default:
         AGX3_REFUSE_RET(I);
      }

      /* Canonicalize an immediate LHS by reversing the comparison,
       * retaining the selected operands and their actual lifetimes. */
      if (swap_compare) {
         switch (d.cond) {
         case AGX3_ISEL_ICOND_ULT: d.cond = AGX3_ISEL_ICOND_UGT; break;
         case AGX3_ISEL_ICOND_UGT: d.cond = AGX3_ISEL_ICOND_ULT; break;
         case AGX3_ISEL_ICOND_SLT: d.cond = AGX3_ISEL_ICOND_SGT; break;
         case AGX3_ISEL_ICOND_SGT: d.cond = AGX3_ISEL_ICOND_SLT; break;
         case AGX3_ISEL_ICOND_EQ: break;
         }
      }

      /* Each selected operand is independently a register or an
       * 8-bit immediate; the encoder knows all four combinations. */
      bool selects_ok = true;
      const agx_index sel[2] = {t, f};
      enum agx3_isel_operand kind[2];
      uint16_t value[2];

      for (unsigned n = 0; n < 2; ++n) {
         const agx_index s = sel[n];
         if (s.type == AGX_INDEX_IMMEDIATE && !s.memory && !s.abs &&
             !s.neg) {
            kind[n] = AGX3_ISEL_IMMEDIATE;
            value[n] = s.value;
         } else if (agx3_src_is_register(s) && !s.memory && !s.abs &&
                    !s.neg && s.size == dest.size &&
                    agx_channels(s) == 1) {
            kind[n] = AGX3_ISEL_REGISTER;
            value[n] = s.value;
         } else if (s.type == AGX_INDEX_UNIFORM && !s.memory &&
                    !s.abs && !s.neg && s.size == AGX_SIZE_32 &&
                    dest.size == AGX_SIZE_32) {
            kind[n] = AGX3_ISEL_UNIFORM;
            value[n] = s.value;
         } else {
            selects_ok = false;
            break;
         }
      }

      if (!selects_ok)
         AGX3_REFUSE_RET(I);

      d.true_kind = kind[0];
      d.false_kind = kind[1];
      d.sel_true = value[0];
      d.sel_false = value[1];
      /* A multi-slot ISEL wait is not reliable on G17P. After all
       * producers have issued, retire each slot through an identity of
       * the exact operand written by that slot. Each carrier has one
       * grounded wait bit; ISEL itself then needs none. */
      d.wait_mask = st->wait;
      if (enc == AGX_ISA_ENCODING_AGX3B && util_bitcount(st->wait) > 1) {
         uint16_t remaining = st->wait;
         while (remaining) {
            const uint16_t bit = remaining & (uint16_t)-(int16_t)remaining;
            const uint8_t producer_slot = (uint8_t)ffs(bit);
            agx_index operand;
            if (!agx3_find_select_wait_operand(
                   st->sl, st->res, st->instr_index, I, producer_slot, &operand) ||
                !agx3_retire_select_operand_wait(
                   emission, enc, operand, bit))
               AGX3_REFUSE_RET(I);
            remaining &= (uint16_t)~bit;
         }
         d.wait_mask = 0;
      }

      uint8_t bytes[AGX3_ISEL_MAX_LENGTH];
      uint8_t len = 0;
      if (!(agx_is_g18g(&ctx->key->dev)
               ? agx3_isel_encode_g18g(enc, &d, bytes, &len)
               : agx3_isel_encode(enc, &d, bytes, &len)) ||
          !agx3_append(emission, bytes, len)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr,
                    "isel refused: dest=%u/%s cmp=%u,%u cond=%u "
                    "t=%s:%u f=%s:%u wait=%x\n",
                    d.dest, d.dest_16 ? "16" : "32", d.cmp0, d.cmp1,
                    d.cond, d.true_kind ? "imm" : "reg", d.sel_true,
                    d.false_kind ? "imm" : "reg", d.sel_false,
                    d.wait_mask);
         AGX3_REFUSE_RET(I);
      }
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_MOV_IMM &&
       I->nr_dests == 1 && !I->nr_srcs && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 && I->imm <= UINT32_MAX &&
       (I->dest[0].size == AGX_SIZE_16 || I->dest[0].size == AGX_SIZE_32)) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_m5_ldimm(I->dest[0].value,
              I->dest[0].size == AGX_SIZE_16, I->dest[0].cache,
              I->imm, st->wait, bytes, &len)) {
         if (!st->wait || !agx3_retire_result_wait(emission, enc, I, st->wait) ||
             !agx3_alu_m5_ldimm(I->dest[0].value,
                 I->dest[0].size == AGX_SIZE_16, I->dest[0].cache,
                 I->imm, 0, bytes, &len))
            AGX3_REFUSE_RET(I);
      }
      if (!agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_MOV_IMM && I->nr_dests == 1 &&
       I->nr_srcs == 0 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d) {
         /* G17G's half ldimm carries the eight wait slots itself. */
         if (I->imm > 0xffff ||
             !agx3_alu_mov16_gpu1(I->dest[0].value, I->dest[0].cache,
                                  I->imm, st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      /* The exact G17P LDIMM16 form has no witnessed scoreboard field.
       * A materialized half constant can nevertheless be scheduled
       * directly after an async producer, so retire its dependency
       * through the established native no-op before loading it. */
      if (I->imm > 0xffff ||
          (st->wait && !agx3_retire_result_wait(emission, enc, I, st->wait)) ||
          !agx3_alu_mov16(enc, I->dest[0].value, I->imm, 0, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_float_and_convert(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   /* Native uniform arithmetic avoids per-use COPY temporaries while
    * preserving the existing GPR dependency mask. Uniform reads do not
    * acquire register cache/discard hints. */
   const int uniform_float = agx3_try_uniform_float(emission, enc, I, st->wait);
   if (uniform_float > 0)
      return AGX3_PACK_EMITTED;
   if (uniform_float < 0)
      AGX3_REFUSE_RET(I);

   /* Constant propagation can expose a float add whose two operands
    * are exact minifloats after scheduling. Fold it to the ordinary
    * native LDIMM instead of pretending either immediate is a GPR. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate && !st->wait &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs &&
       !I->src[0].neg && !I->src[1].neg) {
      const float value = agx_minifloat_decode(I->src[0].value) +
                          agx_minifloat_decode(I->src[1].value);
      uint32_t bits;
      memcpy(&bits, &value, sizeof(bits));
      struct agx3_encoded_ldimm encoded;
      if (!agx3_pack_ldimm_u32(enc, I->dest[0].value, bits, &encoded) ||
          !agx3_append(emission, encoded.bytes, encoded.length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A constant f32-to-f16 conversion is represented by the same ADD
    * with -0 used for the register form. Fold its finite minifloat
    * inputs to the exact half literal; the saturate bit used by depth
    * output clamps before the single conversion, just like Apple's
    * dynamic narrowing instruction. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !st->wait &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs &&
       !I->src[0].neg && !I->src[1].neg) {
      float value = agx_minifloat_decode(I->src[0].value) +
                    agx_minifloat_decode(I->src[1].value);
      if (!isfinite(value))
         AGX3_REFUSE_RET(I);
      if (I->saturate)
         value = CLAMP(value, 0.0f, 1.0f);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_mov16(enc, I->dest[0].value,
                          _mesa_float_to_half(value), 0, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* AGX models a narrowing float conversion as add with exact -0.
    * Keep one native operation, with actual source lifetime and waits. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
      const bool swap = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src = I->src[swap ? 1 : 0];
      const agx_index zero = I->src[swap ? 0 : 1];
      if (agx3_src_is_register(src) && !src.memory && !src.abs &&
          (!src.neg || !I->saturate) &&
          (!src.kill || src.discard) && src.size == AGX_SIZE_32 &&
          agx_channels(src) == 1 &&
          zero.type == AGX_INDEX_IMMEDIATE && zero.value == 0x80 &&
          !zero.memory && !zero.abs && !zero.neg) {
         const struct agx3_falu_runtime runtime = {
            .cache = {src.cache}, .discard = {src.discard},
            .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], len;
         /* -x is exactly -0.0 - x. */
         const bool encoded =
            src.neg ? agx3_falu_narrow_imm_sub_f32(
                         enc, I->dest[0].value, 0x80, src.value,
                         &runtime, bytes, &len)
                    : agx3_falu_narrow_f32(enc, I->dest[0].value,
                                           src.value, I->saturate,
                                           &runtime, bytes, &len);
         if (!encoded || !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FMUL &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[0]) && agx3_vary_reg32(I->src[1])) {
      uint16_t src[2]; enum agx3_alu_hint hints[2];
      for (unsigned i=0;i<2;i++) {
         agx_index x=I->src[i];
         if (agx_channels(x) != 1 || x.abs || x.neg || x.kill ||
             (x.cache && x.discard))
            AGX3_REFUSE_RET(I);
         src[i]=x.value; hints[i]=agx3_post_ra_hint(x);
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      if (!agx3_alu_fmul_narrow(enc,I->dest[0].value,src,hints,
                                I->dest[0].cache,st->wait,bytes,&len) ||
          !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Both operands must be registers for the mixed-half endpoint. A
    * 32-bit register times a minifloat IMMEDIATE narrowed to a half is
    * a different, independently witnessed form (f10 selects the
    * destination width on the ordinary f32mul); let it fall through to
    * the minifloat path below instead of being claimed and refused
    * here. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FMUL &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       ((I->src[0].size == AGX_SIZE_32 &&
         I->src[1].size == AGX_SIZE_16) ||
        (I->src[0].size == AGX_SIZE_16 &&
         I->src[1].size == AGX_SIZE_32))) {
      const bool half_first = I->src[0].size == AGX_SIZE_16;
      const agx_index f32 = I->src[half_first ? 1 : 0];
      const agx_index f16 = I->src[half_first ? 0 : 1];
      /* G17G's mixed-half witnesses retain live half inputs in their
       * GPR without the optional cache hint. G17P also has native
       * retained-source cache witnesses. */
      const enum agx3_alu_hint f16_hint =
         enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d
            ? (f16.discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_NONE)
            : agx3_post_ra_hint(f16);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_src_is_register(f32) || !agx3_src_is_register(f16) ||
          f32.memory || f16.memory || (f32.value & 1) ||
          agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
          f32.abs || f32.neg || f32.kill || f16.abs || f16.neg ||
          f16.kill || (f32.cache && f32.discard) ||
          (f16.cache && f16.discard) ||
          !agx3_alu_fmul_mixed_half(
             enc, I->dest[0].value, f32.value, f16.value,
             agx3_post_ra_hint(f32), f16_hint,
             I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_HMUL &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       ((agx3_src_is_register(I->src[0]) &&
         I->src[1].type == AGX_INDEX_IMMEDIATE) ||
        (I->src[0].type == AGX_INDEX_IMMEDIATE &&
         agx3_src_is_register(I->src[1])))) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      agx_index dst = I->dest[0];
      agx_index src = I->src[immediate_first ? 1 : 0];
      agx_index imm = I->src[immediate_first ? 0 : 1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
          src.memory || src.size != AGX_SIZE_16 ||
          agx_channels(src) != 1 || src.abs || src.neg || src.kill ||
          (src.cache && src.discard) || imm.memory || imm.abs ||
          imm.neg ||
          !agx3_alu_hmul_minifloat(
             enc, dst.value, src.value, imm.value,
             agx3_post_ra_hint(src), dst.cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native f32fma also writes a half destination while consuming a
    * full register and two minifloats. The result width is independent
    * of arithmetic/input widths; this is not a mixed f16fma. */
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_FFMA &&
       I->nr_dests == 1 && I->nr_srcs == 3 &&
       agx3_half_reg(I->dest[0])) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src = I->src[reverse ? 1 : 0];
      const agx_index mul = I->src[reverse ? 0 : 1], add = I->src[2];
      if (agx3_mod_src_ok(src) && agx_channels(src) == 1 &&
          agx3_plain_immediate(mul) && agx3_plain_immediate(add) &&
          mul.value <= UINT8_MAX && add.value <= UINT8_MAX) {
         uint8_t multiplier = mul.value;
         uint8_t addend = add.value;
         if (mul.abs)
            multiplier &= 0x7f;
         if (src.neg ^ mul.neg)
            multiplier ^= 0x80;
         if (add.abs)
            addend &= 0x7f;
         if (add.neg)
            addend ^= 0x80;
         const struct agx3_falu_runtime runtime = {
            .cache = {src.cache}, .discard = {src.discard},
            .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], len;
         if (src.kill ||
             !agx3_falu_fma_two_imm_narrow_m5(
                I->dest[0].value, src.value, multiplier, addend,
                src.abs, I->saturate, &runtime, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_FFMA &&
       I->nr_dests == 1 && I->nr_srcs == 3 && !I->saturate &&
       agx3_half_reg(I->dest[0])) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src = I->src[reverse ? 1 : 0];
      const agx_index mul = I->src[reverse ? 0 : 1], add = I->src[2];
      if (agx3_half_reg(src) && agx3_plain_immediate(mul) &&
          agx3_plain_immediate(add) && mul.value <= UINT8_MAX &&
          add.value <= UINT8_MAX && !src.abs && !src.kill &&
          !(src.cache && src.discard) && !mul.abs && !add.abs) {
         const uint8_t multiplier = mul.value ^ ((src.neg ^ mul.neg) ? 0x80 : 0);
         const uint8_t addend = add.value ^ (add.neg ? 0x80 : 0);
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_hfma_immediates_m5(I->dest[0].value, src.value,
                multiplier, addend, agx3_post_ra_hint(src),
                I->dest[0].cache, st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* A saturating half multiply is the same instruction with G17P's
    * f14 set, which moves it to the two-byte-longer ep2 point, e.g.
    * clamp(texture(tex, uv) * vcolor, 0.0, 1.0) -> fmul.sat on halves. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_HMUL && I->nr_dests == 1 && I->nr_srcs == 2) {
      uint16_t src[2]; enum agx3_alu_hint hints[2];
      agx_index dst=I->dest[0];
      if (!agx3_src_is_register(dst) || dst.memory || dst.size != AGX_SIZE_16 ||
          agx_channels(dst) != 1)
         AGX3_REFUSE_RET(I);
      for (unsigned i=0;i<2;i++) {
         agx_index x=I->src[i];
         if (!agx3_src_is_register(x) || x.memory || x.size != AGX_SIZE_16 ||
             agx_channels(x) != 1 || x.neg || x.kill ||
             (x.cache && x.discard))
            AGX3_REFUSE_RET(I);
         src[i]=x.value; hints[i]=agx3_post_ra_hint(x);
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      const bool src_abs[2] = {I->src[0].abs, I->src[1].abs};
      bool encoded = I->saturate
         ? !src_abs[0] && !src_abs[1] &&
              agx3_alu_hmul_saturate(enc,dst.value,src,hints,dst.cache,
                                     st->wait,bytes,&len)
         : (src_abs[0] || src_abs[1])
              ? agx3_alu_hmul_abs(enc, dst.value, src, hints, src_abs,
                                  dst.cache, st->wait, bytes, &len)
              : agx3_alu_hmul(enc,dst.value,src,hints,dst.cache,st->wait,
                              bytes,&len);

      /* The compact multiply does not cover every pair of odd half
       * registers.  The grounded three-source FMA does, and an exact
       * multiply is the fused product plus negative zero: it preserves
       * both the product rounding and signed zero.  Use this only when
       * the ordinary non-saturating instruction cannot encode the
       * assigned registers. */
      if (!encoded && !I->saturate && !src_abs[0] && !src_abs[1]) {
         encoded = agx3_alu_hfma_addend_minifloat(
            enc, dst.value, src, UINT8_C(0x80), hints, dst.cache,
            st->wait, bytes, &len);
      }

      if (!encoded || !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_HADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       !(I->saturate && ctx->key->dev.emit_agx3d) &&
       agx3_half_reg(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
      const bool swap = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src = I->src[swap ? 1 : 0], imm = I->src[swap ? 0 : 1];
      /* The only witnessed saturating form is the fsat lowering, an
       * add of minifloat negative zero to a plain source. G18G uses
       * the native short form below. */
      if (agx3_half_reg(src) && agx_channels(src) == 1 && !src.abs &&
          !src.kill && !(src.cache && src.discard) &&
          imm.type == AGX_INDEX_IMMEDIATE && imm.value <= UINT8_MAX &&
          !imm.abs && !imm.neg && !imm.memory &&
          (!I->saturate || (!src.neg && imm.value == UINT8_C(0x80)))) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_hadd_minifloat(enc, I->dest[0].value, src.value,
                imm.value, src.neg, I->saturate,
                agx3_post_ra_hint(src), I->dest[0].cache, st->wait, bytes,
                &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d) &&
       I->op == AGX_OPCODE_HADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && I->saturate &&
       I->src[1].type == AGX_INDEX_IMMEDIATE) {
      const agx_index dst = I->dest[0];
      const agx_index src = I->src[0];
      const agx_index imm = I->src[1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != AGX_SIZE_16 || agx_channels(src) != 1 || src.abs ||
          src.neg || src.kill || (src.cache && src.discard) ||
          imm.memory || imm.abs || imm.neg || imm.kill || imm.cache ||
          imm.discard || imm.size != AGX_SIZE_16 ||
          agx_channels(imm) != 1 || imm.value != UINT8_C(0x80) ||
          !agx3_alu_hadd_saturate_negzero(
             enc, dst.value, src.value, agx3_post_ra_hint(src),
             dst.cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* G18G: a half register plus a minifloat in either operand. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_HADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       (I->src[0].type == AGX_INDEX_IMMEDIATE ||
        I->src[1].type == AGX_INDEX_IMMEDIATE)) {
      const unsigned immediate = I->src[0].type == AGX_INDEX_IMMEDIATE ? 0 : 1;
      const agx_index dst = I->dest[0];
      const agx_index src = I->src[1 - immediate];
      const agx_index imm = I->src[immediate];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != AGX_SIZE_16 || agx_channels(src) != 1 || src.abs ||
          src.kill || (src.cache && src.discard) ||
          imm.memory || imm.abs || imm.neg || imm.kill || imm.cache ||
          imm.discard || imm.size != AGX_SIZE_16 ||
          agx_channels(imm) != 1 ||
          !agx3_alu_hadd_minifloat(enc, dst.value, src.value, imm.value, src.neg,
                                 false, agx3_post_ra_hint(src), dst.cache, st->wait,
                                 bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_HADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->src[1].type == AGX_INDEX_IMMEDIATE) {
      const agx_index dst = I->dest[0];
      const agx_index src = I->src[0];
      const agx_index imm = I->src[1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          src.size != AGX_SIZE_16 || agx_channels(src) != 1 || src.abs ||
          !src.neg || src.kill || (src.cache && src.discard) ||
          imm.memory || imm.abs || imm.neg || imm.kill || imm.cache ||
          imm.discard || imm.size != AGX_SIZE_16 ||
          agx_channels(imm) != 1 || imm.value != UINT8_C(0x30) ||
          !agx3_alu_hadd_one_neg(enc, dst.value, src.value,
                                 agx3_post_ra_hint(src), dst.cache, st->wait,
                                 bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Same for the saturating half add. The saturating half add against
    * an immediate is handled above; this is the register/register one. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_HADD &&
       I->nr_dests == 1 && I->nr_srcs == 2) {
      uint16_t src[2];
      enum agx3_alu_hint hints[2];
      unsigned neg_mask = 0;
      agx_index dst = I->dest[0];
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1)
         AGX3_REFUSE_RET(I);
      for (unsigned i = 0; i < 2; ++i) {
         agx_index x = I->src[i];
         if (!agx3_src_is_register(x) || x.memory ||
             x.size != AGX_SIZE_16 || agx_channels(x) != 1 ||
             (x.abs && (enc != AGX_ISA_ENCODING_AGX3C ||
                        ctx->key->dev.emit_agx3d)) ||
             (x.neg && (enc != AGX_ISA_ENCODING_AGX3C ||
                        (I->saturate && !ctx->key->dev.emit_agx3d))) ||
             x.kill || (x.cache && x.discard))
            AGX3_REFUSE_RET(I);
         src[i] = x.value;
         hints[i] = agx3_post_ra_hint(x);
         neg_mask |= x.neg << i;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool src_neg[] = {I->src[0].neg, I->src[1].neg};
      const bool src_abs[] = {I->src[0].abs, I->src[1].abs};
      const bool encoded =
         ctx->key->dev.emit_agx3d && neg_mask
         ? agx3_alu_hadd_neg(enc, dst.value, src, hints,
                             neg_mask, I->saturate, dst.cache,
                             st->wait, bytes, &len)
         : enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d
         ? agx3_alu_hadd_mod_m5(dst.value, src, hints, src_abs, src_neg,
                                 dst.cache, I->saturate, st->wait, bytes, &len)
         : !I->saturate && (src_neg[0] || src_neg[1])
         ? agx3_alu_hadd_signed(enc, dst.value, src, hints, src_neg,
                                 dst.cache, st->wait, bytes, &len)
         : I->saturate ? agx3_alu_hadd_saturate(enc, dst.value, src, hints,
                                              dst.cache, st->wait, bytes, &len)
                     : agx3_alu_hadd(enc, dst.value, src, hints,
                                     dst.cache, st->wait, bytes, &len);
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_HFMA &&
       I->nr_dests == 1 && I->nr_srcs == 3 &&
       (!I->saturate || enc == AGX_ISA_ENCODING_AGX3C)) {
      const bool immediate_addend =
         I->src[2].type == AGX_INDEX_IMMEDIATE;
      if (immediate_addend) {
         if (I->saturate)
            AGX3_REFUSE_RET(I);
         const agx_index dst = I->dest[0];
         const agx_index addend = I->src[2];
         uint16_t src[2];
         enum agx3_alu_hint hints[2];

         if (!agx3_src_is_register(dst) || dst.memory ||
             dst.size != AGX_SIZE_16 || agx_channels(dst) != 1 ||
             addend.memory || addend.abs || addend.neg || addend.kill ||
             addend.cache || addend.discard ||
             addend.size != AGX_SIZE_16 || agx_channels(addend) != 1 ||
             addend.value > UINT8_MAX)
            AGX3_REFUSE_RET(I);

         for (unsigned i = 0; i < 2; ++i) {
            const agx_index x = I->src[i];
            if (!agx3_src_is_register(x) || x.memory ||
                x.size != AGX_SIZE_16 || agx_channels(x) != 1 || x.abs ||
                x.neg || x.kill || (x.cache && x.discard))
               AGX3_REFUSE_RET(I);
            src[i] = x.value;
            hints[i] = agx3_post_ra_hint(x);
         }

         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_hfma_addend_minifloat(
                enc, dst.value, src, addend.value, hints, dst.cache,
                st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }

      uint16_t src[3];
      enum agx3_alu_hint hints[3];
      agx_index dst = I->dest[0];
      if (!agx3_src_is_register(dst) || dst.memory ||
          dst.size != AGX_SIZE_16 || agx_channels(dst) != 1)
         AGX3_REFUSE_RET(I);
      for (unsigned i = 0; i < 3; ++i) {
         agx_index x = I->src[i];
         if (!agx3_src_is_register(x) || x.memory ||
             x.size != AGX_SIZE_16 || agx_channels(x) != 1 || x.abs ||
             x.kill ||
             (x.cache && x.discard))
            AGX3_REFUSE_RET(I);
         src[i] = x.value;
         hints[i] = agx3_post_ra_hint(x);
      }
      const bool product_neg = I->src[0].neg != I->src[1].neg;
      if (product_neg && I->src[0].neg) {
         const uint16_t swap_src = src[0];
         src[0] = src[1];
         src[1] = swap_src;
         const enum agx3_alu_hint swap_hint = hints[0];
         hints[0] = hints[1];
         hints[1] = swap_hint;
      }

      /* Carry one producer in ep3's selector or several producers in
       * the extended endpoint's independent wait mask. */

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool encoded = I->saturate
         ? agx3_alu_hfma_saturate_m5(dst.value, src, hints, product_neg,
                                    I->src[2].neg, dst.cache, st->wait, bytes, &len)
         : agx3_alu_hfma(enc, dst.value, src, hints, product_neg,
                         I->src[2].neg, dst.cache, st->wait, bytes, &len);
      if (!encoded ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native scalar f32 x f16 -> f32. Keep the half register distinct
    * from its neighbour, preserve post-RA hints and producer waits. */
   if (enc == AGX_ISA_ENCODING_AGX3B &&
       I->op == AGX_OPCODE_FMUL && I->nr_dests == 1 &&
       I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) &&
       !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
       (I->src[0].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_16) &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1])) {
      const bool half_first = I->src[0].size == AGX_SIZE_16;
      const agx_index f32 = I->src[half_first ? 1 : 0];
      const agx_index f16 = I->src[half_first ? 0 : 1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (f32.size != AGX_SIZE_32 || f16.size != AGX_SIZE_16 ||
          agx_channels(f32) != 1 || agx_channels(f16) != 1 ||
          f32.memory || f16.memory ||
          f32.neg || f16.abs || f16.neg ||
          f32.kill || f16.kill || (f32.cache && f32.discard) ||
          (f16.cache && f16.discard) ||
          !(f32.abs
               ? agx3_alu_fmul_abs_mixed(
                    enc, I->dest[0].value, f32.value, f16.value,
                    agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                    I->dest[0].cache, st->wait, bytes, &len)
               : agx3_alu_fmul_mixed(
                    enc, I->dest[0].value, f32.value, f16.value,
                    agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
                    I->dest[0].cache, st->wait, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_IADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate && I->shift == 0 &&
       I->dest[0].type == AGX_INDEX_REGISTER && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[imm_first ? 0 : 1];
      const agx_index reg = I->src[imm_first ? 1 : 0];
      if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
          !imm.abs && !imm.memory &&
          reg.type == AGX_INDEX_REGISTER && !reg.memory &&
          reg.size == AGX_SIZE_16 && agx_channels(reg) == 1 &&
          !reg.abs && !reg.neg && !reg.kill && !(reg.cache && reg.discard)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         uint16_t immediate_wait = st->wait;
         if (imm.neg && immediate_wait != 0 && immediate_wait != 32) {
            if (!agx3_retire_result_wait(emission, enc, I, immediate_wait))
               AGX3_REFUSE_RET(I);
            immediate_wait = 0;
         }
         const bool encoded = imm.neg
            ? agx3_alu_sub_imm16(enc, I->dest[0].value, reg.value,
                                 imm.value, agx3_post_ra_hint(reg),
                                 I->dest[0].cache, immediate_wait, bytes,
                                 &len)
            : agx3_alu_add_imm16(enc, I->dest[0].value, reg.value,
                                 imm.value, agx3_post_ra_hint(reg),
                                 I->dest[0].cache, st->wait, bytes, &len);
         if (!encoded ||
             !agx3_append(emission,bytes,len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Float to a 16-bit integer. The source width is its own field, so
    * a half source is the same instruction. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->round == AGX_ROUND_RTZ &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       (I->src[0].value == AGX_CONVERT_F_TO_S16 ||
        I->src[0].value == AGX_CONVERT_F_TO_U16) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       (agx3_vary_reg32(I->src[1]) || agx3_half_reg(I->src[1])) &&
       agx_channels(I->src[1]) == 1 &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       !(I->src[1].cache && I->src[1].discard)) {
      uint16_t convert_wait = st->wait;
      if (enc == AGX_ISA_ENCODING_AGX3B && convert_wait != 0 && convert_wait != 32) {
         if (!agx3_retire_result_wait(emission, enc, I, convert_wait))
            AGX3_REFUSE_RET(I);
         convert_wait = 0;
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_float_to_int16(
             enc, I->dest[0].value, I->src[1].value,
             I->src[1].size == AGX_SIZE_16,
             I->src[0].value == AGX_CONVERT_F_TO_S16,
             agx3_post_ra_hint(I->src[1]), I->dest[0].cache,
             convert_wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* shaderFloat16 commonly converts invocation/index values directly
    * from signed or unsigned32 into a scalar half. This has a native
    * result-width form distinct from the established i32-to-f32 form. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->round == AGX_ROUND_RTE &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       (I->src[0].value == AGX_CONVERT_U32_TO_F ||
        I->src[0].value == AGX_CONVERT_S32_TO_F) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1 &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       !(I->src[1].cache && I->src[1].discard)) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_i32_to_f16(
             enc, I->dest[0].value, I->src[1].value,
             I->src[0].value == AGX_CONVERT_S32_TO_F,
             agx3_post_ra_hint(I->src[1]), I->dest[0].cache, st->wait,
             bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* int8/int16 conversions keep their input in a half register. The
    * int2flt byte selector reads only the low eight bits, so a byte
    * converts without an extension. Apple has no unsigned byte to half
    * form: `half(uchar)` masks the byte in place and converts the
    * resulting ushort, so that sequence is emitted here too. */
    if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->round == AGX_ROUND_RTE &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].size == AGX_SIZE_16 &&
       (((I->src[0].value == AGX_CONVERT_S8_TO_F ||
          I->src[0].value == AGX_CONVERT_U8_TO_F) &&
         (I->dest[0].size == AGX_SIZE_16 ||
          I->dest[0].size == AGX_SIZE_32)) ||
        ((I->src[0].value == AGX_CONVERT_S16_TO_F ||
          I->src[0].value == AGX_CONVERT_U16_TO_F) &&
         (I->dest[0].size == AGX_SIZE_16 ||
          (enc == AGX_ISA_ENCODING_AGX3C && I->dest[0].size == AGX_SIZE_32))))) {
      const agx_index dest = I->dest[0], src = I->src[1];
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || !agx3_src_is_register(src) ||
          src.memory || src.abs || src.neg || src.kill ||
          agx_channels(src) != 1 || (src.cache && src.discard))
         AGX3_REFUSE_RET(I);

      const enum agx_convert mode = I->src[0].value;
      const bool byte =
         mode == AGX_CONVERT_S8_TO_F || mode == AGX_CONVERT_U8_TO_F;
      const bool is_signed =
         mode == AGX_CONVERT_S8_TO_F || mode == AGX_CONVERT_S16_TO_F;
      const bool dest_16 = dest.size == AGX_SIZE_16;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (byte && !is_signed && dest_16) {
         if (!agx3_alu_bit_imm(enc, AGX3_ALU_BAND, true, dest.value,
                               src.value, 0xff, agx3_post_ra_hint(src),
                               st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len) ||
             !agx3_alu_narrow_int_to_float(
                enc, dest.value, true, dest.value, false, false,
                AGX3_ALU_HINT_DISCARD, dest.cache, 0, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }

      if (!agx3_alu_narrow_int_to_float(
             enc, dest.value, dest_16, src.value, byte, is_signed,
             agx3_post_ra_hint(src), dest.cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* GTK's turbulence fragment keeps its octave arithmetic in scalar
    * halves, then indexes the permutation table with a direct f16-to-
   * i32 cast. flt2int has a native half-source selector; widening the
   * value would change both register pressure and rounding. Preserve
   * the signed/unsigned mode, post-RA lifetime/cache state and all six
   * producer dependencies in the native conversion itself. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT && I->nr_srcs == 2 &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
        I->src[0].value == AGX_CONVERT_F_TO_U32) &&
       I->src[1].size == AGX_SIZE_16) {
      if (I->nr_dests != 1 || I->saturate ||
          I->round != AGX_ROUND_RTZ ||
          I->src[0].abs || I->src[0].neg || I->src[0].memory ||
          I->src[0].kill || I->src[0].cache || I->src[0].discard ||
          !agx3_vary_reg32(I->dest[0]) ||
          agx_channels(I->dest[0]) != 1 ||
          !agx3_src_is_register(I->src[1]) || I->src[1].memory ||
          agx_channels(I->src[1]) != 1 || I->src[1].abs ||
          I->src[1].neg || I->src[1].kill ||
          (I->src[1].cache && I->src[1].discard))
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_f16_to_int32(
             enc, I->dest[0].value, I->src[1].value,
             I->src[0].value == AGX_CONVERT_F_TO_S32,
             agx3_post_ra_hint(I->src[1]), I->dest[0].cache, st->wait,
             bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Coordinates and loop counters use16-bit halves. Apple has a direct
    * signed/unsigned half int2flt form, distinct from the32-bit form.
    * Preserve liveness and async producer waits; do not reinterpret the
    * neighbouring half as part of a32-bit integer or widen in place.
    */
   if (I->op == AGX_OPCODE_CONVERT && I->nr_dests == 1 &&
       I->nr_srcs == 2 && !I->saturate &&
       I->round == AGX_ROUND_RTE &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       (I->src[0].value == AGX_CONVERT_U16_TO_F ||
        I->src[0].value == AGX_CONVERT_S16_TO_F) &&
       agx3_src_is_register(I->dest[0]) &&
       I->dest[0].size == AGX_SIZE_32 && agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[1]) && !I->src[1].abs &&
       !I->src[1].neg && !I->src[1].kill &&
       !(I->src[1].cache && I->src[1].discard) &&
       I->src[1].size == AGX_SIZE_16 &&
       agx_channels(I->src[1]) == 1) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      if (!(I->src[0].value == AGX_CONVERT_S16_TO_F ? agx3_alu_s16_to_f32 :
             agx3_alu_u16_to_f32)(enc, I->dest[0].value, I->src[1].value,
                               agx3_post_ra_hint(I->src[1]), st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
      const bool swap=I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src=I->src[swap?1:0], imm=I->src[swap?0:1];
      if (agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_16 &&
          agx_channels(src) == 1 && !src.kill &&
          (ctx->key->dev.emit_agx3d || !(src.abs && src.neg)) &&
          imm.type == AGX_INDEX_IMMEDIATE && imm.value == 0x80 &&
          !imm.abs && !imm.neg && !imm.memory) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
         if (!agx3_alu_widen_half(enc,I->dest[0].value,src.value,
                                   agx3_post_ra_hint(src),src.abs,src.neg,
                                   I->dest[0].cache,
                                   st->wait,bytes,&len) ||
             !agx3_append(emission,bytes,len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FMUL &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       (I->src[0].type == AGX_INDEX_IMMEDIATE ||
        I->src[1].type == AGX_INDEX_IMMEDIATE)) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[immediate_first ? 0 : 1];
      const agx_index half = I->src[immediate_first ? 1 : 0];
      if (imm.value <= UINT8_MAX && !imm.memory && !imm.abs &&
          !imm.neg && agx3_half_reg(half) &&
          agx_channels(half) == 1 && !half.abs && !half.neg &&
          !half.kill && !(half.cache && half.discard)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_fmul_half_minifloat_to_f32(
                enc, I->dest[0].value, half.value, imm.value,
                agx3_post_ra_hint(half), st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* G17G multiplies a scalar half by a minifloat straight into a
    * float (port/hmulwit); a negated half folds into the minifloat's
    * sign, exactly as for the float-source multiply below. */
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_FMUL &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       (I->src[0].type == AGX_INDEX_IMMEDIATE) !=
          (I->src[1].type == AGX_INDEX_IMMEDIATE)) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      agx_index imm = I->src[reverse ? 0 : 1];
      agx_index src = I->src[reverse ? 1 : 0];
      if (agx3_half_reg(src) && (!src.kill || src.discard) &&
          !(src.cache && src.discard) && agx3_plain_immediate(imm) &&
          imm.value <= UINT8_MAX && !imm.abs && !imm.neg) {
         if (src.neg) {
            imm.value ^= 0x80;
            src.neg = false;
         }
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_fmul_half_minifloat_m5(
                I->dest[0].value, src.value, imm.value, I->dest[0].cache,
                src.abs, I->saturate, agx3_post_ra_hint(src), st->wait, bytes,
                &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0])) {
      const bool reverse = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[reverse ? 0 : 1];
      const agx_index src = I->src[reverse ? 1 : 0];
      if (agx3_half_reg(src) && !src.kill && !(src.cache && src.discard) &&
          agx_channels(src) == 1 && agx3_plain_immediate(imm) &&
          imm.value <= UINT8_MAX && !imm.abs && !imm.neg) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_half_add_imm_to_f32_m5(I->dest[0].value, src.value,
                imm.value, src.abs, src.neg, agx3_post_ra_hint(src),
                I->dest[0].cache, st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* G18G keeps its minifloat add forms below. */
   if (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       (agx3_vary_reg32(I->dest[0]) || agx3_half_reg(I->dest[0])) && agx_channels(I->dest[0]) == 1) {
      const bool swap = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index src = I->src[swap ? 1 : 0];
      const agx_index imm = I->src[swap ? 0 : 1];
      if (agx3_mod_src_ok(src) && agx_channels(src) == 1 &&
          imm.type == AGX_INDEX_IMMEDIATE && imm.value <= UINT8_MAX &&
          !imm.memory && !imm.abs && !imm.neg && !imm.cache &&
          !imm.discard && !imm.kill && agx_channels(imm) == 1) {
         const struct agx3_falu_runtime runtime = {
            .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
            .cache = {src.cache}, .discard = {src.discard},
         };
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
         if (!(I->dest[0].size == AGX_SIZE_16
               ? agx3_falu_add_imm_narrow_m5(I->dest[0].value, src.value, imm.value,
                     src.abs, src.neg, I->saturate, &runtime, bytes, &length)
               : agx3_falu_add_imm_m5(I->dest[0].value, src.value, imm.value,
                     src.abs, src.neg, I->saturate, &runtime, bytes, &length)) ||
             !agx3_append(emission, bytes, length))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Pixel-centre +0.5 and coordinate scaling use minifloat sources,
    * not GPRs. Native f32add/f32mul accept this pair directly, with the
    * register's last-use hint and the compact form's supported waits.
    * Commuting plain operands changes no operation or rounding.
    */
   /* The multiply's destination width is independent of its source
    * width (f10 vs f26), so a 32-bit value scaled by a minifloat
    * straight into a half is one instruction (agx_optimizer_fmov_rev()
    * folds an f2f16 into the multiply). Only FMUL narrows. */
   if ((I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL) &&
       I->nr_dests == 1 &&
       I->nr_srcs == 2 &&
       (I->op == AGX_OPCODE_FADD || !I->saturate ||
        enc == AGX_ISA_ENCODING_AGX3C) &&
       agx3_src_is_register(I->dest[0]) &&
       (I->dest[0].size == AGX_SIZE_32 ||
        (I->op == AGX_OPCODE_FMUL &&
         I->dest[0].size == AGX_SIZE_16)) &&
       agx_channels(I->dest[0]) == 1 &&
       (I->src[0].type == AGX_INDEX_IMMEDIATE ||
        I->src[1].type == AGX_INDEX_IMMEDIATE)) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      agx_index imm = I->src[immediate_first ? 0 : 1];
      agx_index reg = I->src[immediate_first ? 1 : 0];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      const bool multiply = I->op == AGX_OPCODE_FMUL;
      /* G17G: a modified register plus a minifloat is one witnessed
       * long f32add (gpu/m5-xfb-alu4-reference.metal, -alu5-); this is
       * how fabs and fneg arrive, as `fadd mod(x), -0.0`. */
      if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
          !I->saturate &&
          (reg.abs || reg.neg) && !reg.kill && !imm.memory &&
          !imm.abs && !imm.neg && imm.value <= 0xff &&
          agx3_src_is_register(reg) && !reg.memory &&
          !(reg.cache && reg.discard) && reg.size == AGX_SIZE_32 &&
          agx_channels(reg) == 1 && I->dest[0].size == AGX_SIZE_32 &&
          st->wait <= 0xff) {
         /* Apple spells a negated multiplicand by negating the
          * constant instead (-fabs(a) * 2.0 is fabs(a) * -2.0), which
          * is exact: the minifloat's sign bit is bit 7. */
         const struct agx3_falu_mods m = {
            .src0_abs = reg.abs,
            .src0_neg = !multiply && reg.neg,
         };
         const struct agx3_falu_runtime runtime = {
            .cache = {reg.cache},
            .discard = {reg.discard},
            .dest_cache = I->dest[0].cache,
            .wait_mask = st->wait,
         };
         const uint8_t minifloat =
            (uint8_t)imm.value ^ (multiply && reg.neg ? 0x80 : 0);
         uint8_t mod_bytes[AGX3_FALU_MOD_MAX_LENGTH];
         uint8_t mod_len = 0;
         const bool ok = multiply
            ? agx3_falu_mul_imm_runtime(I->dest[0].value, reg.value,
                                        minifloat, &m, &runtime,
                                        mod_bytes, &mod_len)
            : agx3_falu_add_imm_runtime(I->dest[0].value, reg.value,
                                        minifloat, &m, &runtime,
                                        mod_bytes, &mod_len);
         if (!ok || !agx3_append(emission, mod_bytes, mod_len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      /* Use the already-supported signed minifloat encoding for -r * k.
       * This changes neither the multiplication nor its rounding, and
       * leaves ABS, widths, cache/discard hints and waits intact. */
      if ((enc == AGX_ISA_ENCODING_AGX3B ||
           enc == AGX_ISA_ENCODING_AGX3C) && multiply && reg.neg &&
          imm.value <= UINT8_MAX && !imm.memory && !imm.abs && !imm.neg) {
         imm.value ^= 0x80;
         reg.neg = false;
      }

      if (!multiply && !I->saturate && reg.abs && !reg.neg && !reg.kill &&
          imm.value == 0x80 && !imm.memory && !imm.abs && !imm.neg &&
          agx3_src_is_register(reg) && !reg.memory &&
          !(reg.cache && reg.discard) && reg.size == AGX_SIZE_32 &&
          agx_channels(reg) == 1) {
         if (!agx3_alu_fadd_abs_negzero(
                enc, I->dest[0].value, reg.value, I->dest[0].cache,
                agx3_post_ra_hint(reg), st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      if (!multiply && !I->saturate && reg.abs && reg.neg && !reg.kill &&
          imm.value == 0x30 && !imm.memory && !imm.abs && !imm.neg &&
          agx3_src_is_register(reg) && !reg.memory &&
          !(reg.cache && reg.discard) && reg.size == AGX_SIZE_32 &&
          agx_channels(reg) == 1) {
         if (!agx3_alu_fadd_one_absneg(
                enc, I->dest[0].value, reg.value, I->dest[0].cache,
                agx3_post_ra_hint(reg), st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
      if (imm.memory || imm.abs || imm.neg ||
          !agx3_src_is_register(reg) ||
          (reg.kill && !reg.discard) ||
          (multiply && reg.neg) ||
          (reg.cache && reg.discard) ||
          reg.size != AGX_SIZE_32 || agx_channels(reg) != 1)
         AGX3_REFUSE_RET(I);

      bool encoded;
      if (!multiply && !reg.abs && (I->saturate || reg.neg)) {
         /* Saturation selects the long ep2, which carries all six wait
          * bits; a bare negation stays on the compact ep1. */
         uint16_t encoded_wait = st->wait;
         if (!I->saturate &&
             !agx3_fuse_compact_minifloat_wait(emission, enc, I,
                                               &encoded_wait))
            AGX3_REFUSE_RET(I);
         /* Neither form has a field for slot 7 or 8, which AGX3C hands
          * out once more results are in flight: hk's image-to-buffer
          * copy loads four texels a thread, and its UNORM repack
          * sat(texel + -0.0) can wait on a texel load that got slot 8.
          * Retire such a wait through the band carrier on the
          * destination, as the uniform-to-register copy does, and add
          * with no wait. */
         if (encoded_wait & ~UINT16_C(0x3f)) {
            if (!agx3_retire_result_wait(emission, enc, I,
                                         encoded_wait))
               AGX3_REFUSE_RET(I);
            encoded_wait = 0;
         }
         encoded = (agx_is_g18g(&ctx->key->dev)
                     ? agx3_alu_fadd_minifloat_mod_g18g
                     : agx3_alu_fadd_minifloat_mod)(
            enc, I->dest[0].value, reg.value, imm.value, reg.neg,
            I->saturate, agx3_post_ra_hint(reg), encoded_wait, bytes,
            &len);
      } else if (multiply) {
         /* The recovered six-byte minifloat endpoint fixes bits 32:33.
          * Treating bit 33 as an ABS modifier produced wrong values on
          * G17P: abs(0.027...) * 2.0 saturated above 1.0 while the
          * identical plain multiply was exact. Use two independently
          * Apple-grounded instructions instead. FABS writes the final
          * destination as a retained intermediate; the plain FMUL then
          * consumes it in place. This adds no register, spill, wait, or
          * serialization and preserves the source's real lifetime. */
         if (reg.abs && enc != AGX_ISA_ENCODING_AGX3C) {
            uint8_t abs_bytes[AGX3_ALU_MAX_LENGTH], abs_len = 0;
            uint8_t mul_bytes[AGX3_ALU_MAX_LENGTH], mul_len = 0;
            if (I->dest[0].size != AGX_SIZE_32 ||
                !agx3_alu_fadd_abs_negzero(
                   enc, I->dest[0].value, reg.value, true,
                   agx3_post_ra_hint(reg), st->wait, abs_bytes, &abs_len) ||
                !agx3_alu_fmul_minifloat_mod(
                   enc, I->dest[0].value, I->dest[0].value, imm.value,
                   I->dest[0].cache, false, AGX3_ALU_HINT_DISCARD, 0,
                   mul_bytes, &mul_len) ||
                !agx3_append(emission, abs_bytes, abs_len) ||
                !agx3_append(emission, mul_bytes, mul_len))
               AGX3_REFUSE_RET(I);
            return AGX3_PACK_EMITTED;
         }
         /* Every grounded minifloat multiply is a compact form, at both
          * destination widths, so a multi-slot wait needs the identity
          * prelude here exactly as it does on the compact ADD below. */
         uint16_t encoded_wait = st->wait;
         if (enc == AGX_ISA_ENCODING_AGX3B &&
             !agx3_fuse_compact_minifloat_wait(emission, enc, I,
                                               &encoded_wait))
            AGX3_REFUSE_RET(I);
         encoded = agx3_alu_fmul_minifloat_narrow(
            enc, I->dest[0].value, reg.value, imm.value,
            I->dest[0].cache, reg.abs,
            I->dest[0].size == AGX_SIZE_16, I->saturate,
            agx3_post_ra_hint(reg),
            encoded_wait, bytes, &len);
      } else if (reg.abs) {
         encoded = agx3_alu_fadd_abs_minifloat(
            enc, I->dest[0].value, reg.value, imm.value,
            I->dest[0].cache, reg.neg, I->saturate,
            agx3_post_ra_hint(reg), st->wait, bytes, &len);
      } else if (enc == AGX_ISA_ENCODING_AGX3C &&
                 ctx->key->dev.emit_agx3d && (st->wait & ~0x3fu) &&
                 !I->saturate && !reg.kill &&
                 I->dest[0].size == AGX_SIZE_32 &&
                 !(I->dest[0].value & 1) && I->dest[0].value < 256) {
         /* The compact form has no wait field past slot 6. Apple's
          * G17G compiler spells x + minifloat waiting on slot 8 in the
          * long f32add form (metal-nt applegpu_g17g, m6-late-slot-add:
          * 2900044a2220ac1203081900, f17 = 96, f59 = 1), the field
          * vector agx3_falu_add_imm_runtime composes with no
          * modifier. */
         const struct agx3_falu_mods m = {0};
         const struct agx3_falu_runtime runtime = {
            .cache = {reg.cache},
            .discard = {reg.discard},
            .dest_cache = I->dest[0].cache,
            .wait_mask = st->wait,
         };
         uint8_t long_bytes[AGX3_FALU_MOD_MAX_LENGTH];
         uint8_t long_len = 0;
         if (!agx3_falu_add_imm_runtime(I->dest[0].value, reg.value,
                                        (uint8_t)imm.value, &m,
                                        &runtime, long_bytes,
                                        &long_len) ||
             !agx3_append(emission, long_bytes, long_len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      } else {
         /* A real sky shader reaches this operation with one source
          * dependency and an independent write-after-write dependency
          * on its retained destination. */
         uint16_t encoded_wait = st->wait;
         if (!agx3_fuse_compact_minifloat_wait(emission, enc, I,
                                               &encoded_wait))
            AGX3_REFUSE_RET(I);
         encoded = (agx_is_g18g(&ctx->key->dev)
                     ? agx3_alu_fadd_minifloat_g18g
                     : agx3_alu_fadd_minifloat)(
            enc, I->dest[0].value, reg.value, imm.value,
            I->dest[0].cache, agx3_post_ra_hint(reg), encoded_wait,
            bytes, &len);
      }
      if (!encoded || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_bitop_and_shift(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_half_reg(I->dest[0])) {
      const unsigned tt = I->truth_table & 15;
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[immediate_first ? 0 : 1];
      const agx_index reg = I->src[immediate_first ? 1 : 0];
      const bool complement = tt == 2 || tt == 4;
      if ((tt == AGX3_SEL_TT_AND || complement) &&
          imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
          !imm.memory && !imm.abs && !imm.neg &&
          agx3_half_reg(reg) && !reg.abs && !reg.neg &&
          !(reg.cache && reg.discard)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_and_imm16_m5(
                I->dest[0].value, reg.value, imm.value, complement,
                complement && ((tt == 4) != immediate_first),
                agx3_post_ra_hint(reg), I->dest[0].cache, st->wait,
                bytes, &len) || !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* A narrow NOT is Apple's unary 16-bit `inv`; the bitop's second
    * source is ignored by the truth table. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_NOT &&
       I->nr_dests == 1 && I->nr_srcs >= 1 && !I->saturate &&
       I->dest[0].size == AGX_SIZE_16 && I->src[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1 &&
       agx3_src_is_register(I->dest[0]) &&
       agx3_src_is_register(I->src[0]) && !I->src[0].abs &&
       !I->src[0].neg) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (I->src[0].kill ||
          !agx3_alu_bit_not_reg16(enc, I->dest[0].value,
                                  I->src[0].value,
                                  agx3_post_ra_hint(I->src[0]),
                                  I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->dest[0].size == AGX_SIZE_16 &&
       (I->src[0].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[0])) &&
       (I->src[1].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[1])) &&
       agx_channels(I->dest[0]) == 1 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       agx3_src_is_register(I->dest[0]) && agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       !I->src[0].abs && !I->src[0].neg && !I->src[1].abs && !I->src[1].neg) {
      const unsigned tt = I->truth_table & 15;
      if (tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR || tt == AGX3_SEL_TT_XOR) {
         const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
            tt == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR;
         /* A narrow bitwise result depends only on the low half of
          * each input, including wider IR registers. Native register
          * indices already address halves: retain those indices and
          * the actual lifetime hints without a conversion or copy. */
         const uint16_t src[2] = {I->src[0].value, I->src[1].value};
         const enum agx3_alu_hint hints[2] = {
            agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (I->src[0].kill || I->src[1].kill ||
             !agx3_alu_bit_reg16(enc, op, I->dest[0].value, src, hints, st->wait,
                                 bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[0]) && agx3_vary_reg32(I->src[1]) &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
      const unsigned tt = I->truth_table & 0xfu;
      const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
         tt == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR;
      if (tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR ||
          tt == AGX3_SEL_TT_XOR) {
         if (I->src[0].kill || I->src[1].kill ||
             (I->src[0].cache && I->src[0].discard) ||
             (I->src[1].cache && I->src[1].discard))
            AGX3_REFUSE_RET(I);
         const uint16_t src[2]={I->src[0].value,I->src[1].value};
         const enum agx3_alu_hint hints[2]={
            agx3_post_ra_hint(I->src[0]),agx3_post_ra_hint(I->src[1])};
         uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0;
         /* The compact form is grounded for G17P only. G18G also
          * reaches this dispatch, with the agx3c (G17G) encoding,
          * before its agx3d re-encode. */
         const bool encoded = enc == AGX_ISA_ENCODING_AGX3C
            ? agx3_alu_bit_reg(enc, op, I->dest[0].value, src,
                                hints, st->wait, bytes, &len)
            : agx3_alu_bit_reg_native(enc, op, I->dest[0].value, src,
                                       hints, I->dest[0].cache, st->wait,
                                       bytes, &len);
         if (!encoded || !agx3_append(emission,bytes,len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* G17G's 32-bit and-not, `x & ~y` (truth table 0x2) or with the
    * sources exchanged (0x4): bit_andc over aligned pairs, with the
    * actual post-RA lifetimes (agx3_alu_bit_andc32_gpu1). */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       ((I->truth_table & 0xfu) == 0x2u ||
        (I->truth_table & 0xfu) == 0x4u) &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[0]) && agx_channels(I->src[0]) == 1 &&
       agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1) {
      const bool swap = (I->truth_table & 0xfu) == 0x4u;
      const agx_index keep = I->src[swap ? 1 : 0];
      const agx_index clear = I->src[swap ? 0 : 1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (keep.kill || clear.kill || (keep.cache && keep.discard) ||
          (clear.cache && clear.discard) ||
          !agx3_alu_bit_andc32_gpu1(I->dest[0].value, I->dest[0].cache,
                                    keep.value, agx3_post_ra_hint(keep),
                                    clear.value,
                                    agx3_post_ra_hint(clear), st->wait,
                                    bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && agx3_vary_reg32(I->dest[0]) &&
       agx3_half_reg(I->src[0]) && agx3_half_reg(I->src[1]) &&
       !I->src[0].abs && !I->src[0].neg &&
       !I->src[1].abs && !I->src[1].neg &&
       !I->src[0].kill && !I->src[1].kill &&
       !(I->src[0].cache && I->src[0].discard) &&
       !(I->src[1].cache && I->src[1].discard)) {
      const unsigned tt = I->truth_table & 15;
      if (tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR ||
          tt == AGX3_SEL_TT_XOR) {
         const uint16_t src[2] = {I->src[0].value, I->src[1].value};
         const enum agx3_alu_hint hints[2] = {
            agx3_post_ra_hint(I->src[0]), agx3_post_ra_hint(I->src[1])};
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
         if (!agx3_alu_bit_half_to_word_m5(
                tt == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
                tt == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR,
                I->dest[0].value, src, hints, I->dest[0].cache, st->wait,
                bytes, &len) || !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       ((I->truth_table & 0xfu) == AGX3_SEL_TT_AND ||
        (I->truth_table & 0xfu) == AGX3_SEL_TT_OR ||
        (I->truth_table & 0xfu) == AGX3_SEL_TT_XOR) &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       ((agx3_vary_reg32(I->src[0]) && agx3_half_reg(I->src[1])) ||
        (agx3_half_reg(I->src[0]) && agx3_vary_reg32(I->src[1]))) &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
      const bool word_first = I->src[0].size == AGX_SIZE_32;
      const agx_index word = I->src[word_first ? 0 : 1];
      const agx_index half = I->src[word_first ? 1 : 0];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_bit_mixed_m5(
              (I->truth_table & 0xfu) == AGX3_SEL_TT_AND ? AGX3_ALU_BAND :
              (I->truth_table & 0xfu) == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR,
              I->dest[0].value, word.value, half.value,
              agx3_post_ra_hint(word), agx3_post_ra_hint(half),
              I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* AGX3b's compact 32-bit move cannot address full registers above
    * r95; the 16-bit bitwise-immediate form addresses every half, so
    * such a move is two ordered `x | 0` halves. The producer wait goes
    * on the first. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_MOV &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->shift == 0 && agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[0]) &&
       agx_channels(I->src[0]) == 1 && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard) &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].value == 0 && !I->src[1].memory &&
       !I->src[1].abs && !I->src[1].neg &&
       (I->dest[0].value >= 192 || I->src[0].value >= 192)) {
      uint8_t low[AGX3_ALU_MAX_LENGTH], high[AGX3_ALU_MAX_LENGTH];
      uint8_t low_len, high_len;
      if (!agx3_alu_bit_imm(
             enc, AGX3_ALU_BOR, true, I->dest[0].value,
             I->src[0].value, 0, agx3_post_ra_hint(I->src[0]),
             st->wait, low, &low_len) ||
          !agx3_alu_bit_imm(
             enc, AGX3_ALU_BOR, true, I->dest[0].value + 1,
             I->src[0].value + 1, 0,
             agx3_post_ra_hint(I->src[0]), 0, high, &high_len) ||
          !agx3_append(emission, low, low_len) ||
          !agx3_append(emission, high, high_len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* G17G half AND against an 8-bit immediate, the stencil mask of
    * Apple's varying-stencil fragment (gpu/m5-fragment-depth-
    * reference.metal). Other operations and widths stay closed there. */
   /* G17G 16-bit register bit operations: AND/OR/XOR over halves
    * (gpu/m5-xfb-alu9-reference.metal) and the ctz kernel's
    * `bit_andc h3, h3, h2` (gpu/m5-xfb-alu3-reference.metal). The
    * truth table 0x2 is a & ~b; 0x4 is the same with the sources
    * exchanged.
    *
    * A narrow result depends only on the low half of a 32-bit source,
    * and a register index already names a half, so such a source is
    * its even half. Apple does exactly this for `ushort(x) & k` with
    * x still live as a word: `bit_and h2, h2, h0`, x in r0
    * (test/native/g17g/m5-libagx-reference.metal and16_of32_live). */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       ((I->truth_table & 0xfu) == AGX3_SEL_TT_AND ||
        (I->truth_table & 0xfu) == AGX3_SEL_TT_OR ||
        (I->truth_table & 0xfu) == AGX3_SEL_TT_XOR ||
        (I->truth_table & 0xfu) == 0x2u ||
        (I->truth_table & 0xfu) == 0x4u) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 && I->dest[0].size == AGX_SIZE_16 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       !I->src[0].abs && !I->src[0].neg && !I->src[0].kill &&
       (I->src[0].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[0])) &&
       agx_channels(I->src[0]) == 1 &&
       !(I->src[0].cache && I->src[0].discard) &&
       agx3_src_is_register(I->src[1]) && !I->src[1].memory &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       (I->src[1].size == AGX_SIZE_16 || agx3_vary_reg32(I->src[1])) &&
       agx_channels(I->src[1]) == 1 &&
       !(I->src[1].cache && I->src[1].discard)) {
      const unsigned table = I->truth_table & 0xfu;
      const bool andc = table == 0x2u || table == 0x4u;
      const bool swap = table == 0x4u;
      const agx_index keep = I->src[swap ? 1 : 0];
      const agx_index clear = I->src[swap ? 0 : 1];
      const enum agx3_alu_op op =
         table == AGX3_SEL_TT_AND ? AGX3_ALU_BAND
         : table == AGX3_SEL_TT_OR ? AGX3_ALU_BOR : AGX3_ALU_BXOR;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      const bool ok = andc
         ? agx3_alu_bit_andc16_gpu1(I->dest[0].value, I->dest[0].cache,
                                    keep.value, agx3_post_ra_hint(keep),
                                    clear.value,
                                    agx3_post_ra_hint(clear), st->wait,
                                    bytes, &len)
         : agx3_alu_bit_reg16_gpu1(op, I->dest[0].value,
                                   I->dest[0].cache, I->src[0].value,
                                   agx3_post_ra_hint(I->src[0]),
                                   I->src[1].value,
                                   agx3_post_ra_hint(I->src[1]), st->wait,
                                   bytes, &len);
      if (!ok || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_AND &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       I->dest[0].size == AGX_SIZE_16) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[immediate_first ? 0 : 1];
      const agx_index reg = I->src[immediate_first ? 1 : 0];
      if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 0xff &&
          !imm.abs && !imm.neg && !imm.memory &&
          agx3_src_is_register(reg) && !reg.memory && !reg.abs &&
          !reg.neg && !reg.kill && agx_channels(reg) == 1 &&
          !(reg.cache && reg.discard) && reg.size == AGX_SIZE_16) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_bit_imm_gpu1(AGX3_ALU_BAND, I->dest[0].value,
                                    I->dest[0].cache, reg.value,
                                    (uint8_t)imm.value,
                                    agx3_post_ra_hint(reg), st->wait,
                                    bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /*
    * The same bitwise operation against an IMMEDIATE. The operand kind
    * is a field of the encoding point the register form already uses
    * (f37, with the constant in f29), so `x & 0xff`, `x | mask` and
    * `x ^ mask` are that instruction with one field changed -- not a
    * separate form, and not something to fail a shader closed over.
    * AND/OR/XOR all commute, so an immediate in either source slot
    * lands in the one slot whose kind field takes it. AGX3C has the
    * same point (native G17G witnesses in agx3_alu_bit_imm()).
    */
   if ((enc == AGX_ISA_ENCODING_AGX3B ||
        enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_BITOP &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       agx_channels(I->dest[0]) == 1 &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32)) {
      const unsigned tt = I->truth_table & 0xfu;
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[immediate_first ? 0 : 1];
      const agx_index reg = I->src[immediate_first ? 1 : 0];
      const bool narrow = I->dest[0].size == AGX_SIZE_16;
      if ((tt == AGX3_SEL_TT_AND || tt == AGX3_SEL_TT_OR ||
           tt == AGX3_SEL_TT_XOR) &&
          imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 0xff &&
          !imm.abs && !imm.neg && !imm.memory &&
          agx3_src_is_register(reg) && !reg.memory && !reg.abs &&
          !reg.neg && !reg.kill && agx_channels(reg) == 1 &&
          !(reg.cache && reg.discard) &&
          reg.size == I->dest[0].size) {
         const enum agx3_alu_op op = tt == AGX3_SEL_TT_AND
                                        ? AGX3_ALU_BAND
                                        : tt == AGX3_SEL_TT_OR
                                             ? AGX3_ALU_BOR
                                             : AGX3_ALU_BXOR;
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_bit_imm(enc, op, narrow, I->dest[0].value,
                               reg.value, (uint8_t)imm.value,
                               agx3_post_ra_hint(reg), st->wait, bytes,
                               &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* The generic selector's original f32-to-i32 vector had a fixed
    * source-discard bit and no cache fields. That destroys a value that
    * remains live for a following compare. Encode the actual post-RA
    * lifetime state through the native flt2int fields instead. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT &&
       I->nr_srcs == 2 &&
       I->src[0].type == AGX_INDEX_IMMEDIATE &&
       (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
        I->src[0].value == AGX_CONVERT_F_TO_U32) &&
       I->src[1].size == AGX_SIZE_32) {
      if (I->nr_dests != 1 || I->saturate ||
          I->round != AGX_ROUND_RTZ ||
          I->src[0].abs || I->src[0].neg || I->src[0].memory ||
          I->src[0].kill || I->src[0].cache || I->src[0].discard ||
          !agx3_vary_reg32(I->dest[0]) ||
          agx_channels(I->dest[0]) != 1 ||
          !agx3_vary_reg32(I->src[1]) ||
          agx_channels(I->src[1]) != 1 || I->src[1].abs ||
          I->src[1].neg || I->src[1].kill ||
          (I->src[1].cache && I->src[1].discard))
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_float_to_int32(
             enc, I->dest[0].value, I->src[1].value,
             I->src[0].value == AGX_CONVERT_F_TO_S32,
             agx3_post_ra_hint(I->src[1]), I->dest[0].cache, st->wait,
             bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_CONVERT &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_vary_reg32(I->src[1]) && agx_channels(I->src[1]) == 1) {
      struct agx3_sel_instr S;
      if (agx3_translate_instr(I, &S) &&
          (S.op == AGX3_SEL_OP_I32_TO_F || S.op == AGX3_SEL_OP_U32_TO_F)) {
         if (I->src[1].kill || (I->src[1].cache && I->src[1].discard))
            AGX3_REFUSE_RET(I);
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_int_to_float(enc, I->dest[0].value, I->src[1].value,
                 S.op == AGX3_SEL_OP_I32_TO_F,
                 agx3_post_ra_hint(I->src[1]), I->dest[0].cache,
                 st->wait, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_FFMA &&
       I->nr_dests == 1 && I->nr_srcs == 3 &&
       (agx3_vary_reg32(I->dest[0]) || agx3_half_reg(I->dest[0])) && agx_channels(I->dest[0]) == 1 &&
       agx3_mod_src_any_width(I->src[0]) &&
       agx3_mod_src_any_width(I->src[1]) &&
       (I->dest[0].size == AGX_SIZE_16 || I->saturate || I->src[0].size == AGX_SIZE_16 ||
        I->src[1].size == AGX_SIZE_16) &&
       agx3_plain_immediate(I->src[2]) && I->src[2].value <= UINT8_MAX) {
      const struct agx3_falu_mods m = {
         .src0_abs = I->src[0].abs, .src0_neg = I->src[0].neg,
         .src0_16 = I->src[0].size == AGX_SIZE_16,
         .src1_abs = I->src[1].abs, .src1_neg = I->src[1].neg,
         .src1_16 = I->src[1].size == AGX_SIZE_16,
         .saturate = I->saturate,
      };
      const struct agx3_falu_runtime runtime = {
         .cache = {I->src[0].cache, I->src[1].cache},
         .discard = {I->src[0].discard, I->src[1].discard},
         .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
      };
      uint8_t addend = I->src[2].value;
      if (I->src[2].abs)
         addend &= 0x7f;
      if (I->src[2].neg)
         addend ^= 0x80;
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
      if (I->src[0].kill || I->src[1].kill ||
          !(I->dest[0].size == AGX_SIZE_16
             ? agx3_falu_fma_addend_imm_narrow_m5(
                  I->dest[0].value, I->src[0].value, I->src[1].value,
                  addend, &m, &runtime, bytes, &length)
             : agx3_falu_fma_addend_imm_width_mod(
                  enc, I->dest[0].value, I->src[0].value, I->src[1].value,
                  addend, &m, &runtime, bytes, &length)) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FFMA &&
       !I->saturate && I->nr_dests == 1 && I->nr_srcs == 3 &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       agx3_mod_src_ok(I->src[0]) &&
       agx_channels(I->src[0]) == 1 &&
       agx3_mod_src_ok(I->src[1]) &&
       agx_channels(I->src[1]) == 1 &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
       !I->src[2].abs && !I->src[2].neg && !I->src[2].memory) {
      /* Multiplication is commutative. Apple's G17P form puts a lone
       * absolute multiplier in the f35 source slot; match that
       * canonical order while moving the post-RA lifetime hints with
       * their operands. */
      const bool swap = !I->src[0].abs && I->src[1].abs;
      const agx_index src0 = I->src[swap ? 1 : 0];
      const agx_index src1 = I->src[swap ? 0 : 1];
      struct agx3_falu_runtime runtime = {
         .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
      };
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index src = I->src[swap ? 1 - i : i];
         if (src.kill)
            AGX3_REFUSE_RET(I);
         runtime.cache[i] = src.cache;
         runtime.discard[i] = src.discard;
      }
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
      if (!agx3_falu_fma_addend_imm_mod(enc, I->dest[0].value,
              src0.value, src1.value, I->src[2].value,
              src0.abs, src0.neg, src1.abs, src1.neg,
              &runtime, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FFMA &&
       !I->saturate && I->nr_dests == 1 && I->nr_srcs == 3 &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1 &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 255 &&
       !I->src[2].abs && !I->src[2].neg && !I->src[2].memory) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[imm_first ? 0 : 1];
      const agx_index reg = I->src[imm_first ? 1 : 0];
      /* A negated multiplicand folds into the minifloat multiplier,
       * k * -a == -k * a exactly, which is how Apple spells it:
       * applegpu_g18g/g17g fma(-a, 0.03125f, 1.0f) is this form with
       * f34 = 0x82 (-0.03125), f51 = 48 (1.0) and no negate bit
       * (g18ref-out/probe/m6-alu-misc ffma_neg_kk). */
      const bool fold_neg = enc == AGX_ISA_ENCODING_AGX3C &&
                            ctx->key->dev.emit_agx3d && reg.neg;
      agx_index reg_plain = reg;
      reg_plain.neg = false;
      if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
          !imm.abs && !imm.neg && !imm.memory &&
          (!reg.neg || fold_neg) &&
          agx3_vary_reg32(reg_plain) && agx_channels(reg) == 1) {
         if (reg.kill)
            AGX3_REFUSE_RET(I);
         const struct agx3_falu_runtime runtime = {
            .dest_cache = I->dest[0].cache, .wait_mask = st->wait,
            .cache = {reg.cache}, .discard = {reg.discard},
         };
         const uint8_t multiplier =
            (uint8_t)(fold_neg ? imm.value ^ 0x80 : imm.value);
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
         if (!agx3_falu_fma_two_imm(enc, I->dest[0].value,
                 reg.value, multiplier, I->src[2].value,
                 &runtime, bytes, &length) ||
             !agx3_append(emission, bytes, length))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Native FMA with a minifloat multiplier. Preserve both async
    * dependencies in the instruction, including the extended two-wait
    * form; do not serialize independent texture operations. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || ctx->key->dev.emit_agx3d) &&
       I->op == AGX_OPCODE_FFMA &&
       (enc != AGX_ISA_ENCODING_AGX3C || !I->saturate) && I->nr_dests == 1 && I->nr_srcs == 3 &&
       agx3_vary_reg32(I->dest[0]) && agx_channels(I->dest[0]) == 1) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[imm_first ? 0 : 1];
      const agx_index reg = I->src[imm_first ? 1 : 0];
      const agx_index add = I->src[2];
      agx_index add_plain = add;
      add_plain.neg = false;
      if (imm.type == AGX_INDEX_IMMEDIATE && imm.value <= 255 &&
          !imm.memory && !imm.abs && !imm.neg &&
          agx3_vary_reg32(reg) && agx_channels(reg) == 1 &&
          agx3_vary_reg32(add_plain) && agx_channels(add) == 1 &&
          !add.abs) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
         if (reg.kill || add.kill || (reg.cache && reg.discard) ||
             (add.cache && add.discard) ||
             !agx3_alu_fma_minifloat_mod(enc,I->dest[0].value,reg.value,
                 imm.value,add.value,add.neg,I->saturate,agx3_post_ra_hint(reg),
                 agx3_post_ra_hint(add),I->dest[0].cache,st->wait,bytes,&len) ||
             !agx3_append(emission,bytes,len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* The same native FMA, with the modifiers the compact minifloat form
    * above cannot carry: operand kind (f42: f34 immediate or f35
    * register) is a field of the same encoding point as the f30/f31
    * and f64/f65 modifiers and the f26/f60 widths, so
    * `k * fabs(a) + fabs(b)` is one instruction. A modifier on the
    * immediate is folded into the minifloat (fma(0.5, -b, c) has
    * f34 = -0.5). */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FFMA &&
       I->nr_dests == 1 && I->nr_srcs == 3 &&
       (agx3_vary_reg32(I->dest[0]) ||
        (enc == AGX_ISA_ENCODING_AGX3C && agx3_half_reg(I->dest[0]))) &&
       agx_channels(I->dest[0]) == 1 &&
       (I->src[0].type == AGX_INDEX_IMMEDIATE) !=
          (I->src[1].type == AGX_INDEX_IMMEDIATE)) {
      const bool imm_first = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[imm_first ? 0 : 1];
      const agx_index reg = I->src[imm_first ? 1 : 0];
      const agx_index add = I->src[2];

      if (agx3_plain_immediate(imm) && imm.value <= 255 &&
          agx3_mod_src_any_width(reg) && agx3_mod_src_any_width(add)) {
         if (reg.kill || add.kill || (reg.cache && reg.discard) ||
             (add.cache && add.discard))
            AGX3_REFUSE_RET(I);

         uint8_t multiplier = imm.value;
         if (imm.abs)
            multiplier &= 0x7f;
         if (imm.neg)
            multiplier ^= 0x80;

         /* The multiplicand's negate folds into the multiplier too:
          * k * -b == -k * b, and Apple encodes it that way rather than
          * setting f31, which it never emits beside an immediate. */
         if (reg.neg)
            multiplier ^= 0x80;

         const struct agx3_falu_mods m = {
            .src1_abs = reg.abs,
            .src1_16 = reg.size == AGX_SIZE_16,
            .addend_abs = add.abs,
            .addend_neg = add.neg,
            .addend_16 = add.size == AGX_SIZE_16,
            .saturate = I->saturate,
         };
         const struct agx3_falu_runtime runtime = {
            .cache = {false, reg.cache, add.cache},
            .discard = {false, reg.discard, add.discard},
            .dest_cache = I->dest[0].cache,
            .wait_mask = st->wait,
         };
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
         const bool narrow = I->dest[0].size == AGX_SIZE_16;
         if (!(narrow ? agx3_falu_fma_mul_imm_narrow_m5(
                           I->dest[0].value, multiplier, reg.value,
                           add.value, &m, &runtime, bytes, &length)
                      : agx3_falu_fma_mul_imm_mod(enc, I->dest[0].value,
                           multiplier, reg.value, add.value, &m,
                           &runtime, bytes, &length)) ||
             !agx3_append(emission, bytes, length))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* The legacy generic packers embed specimen discard flags. Keep
    * G17P's plain register ALU on the post-RA-aware path so an alpha or
    * blend factor reused by later channels is not discarded early. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       !I->saturate && (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
        I->op == AGX_OPCODE_FFMA)) {
      bool plain = I->nr_dests == 1 &&
                   agx3_vary_reg32(I->dest[0]) &&
                   agx_channels(I->dest[0]) == 1 &&
                   I->nr_srcs == (I->op == AGX_OPCODE_FFMA ? 3 : 2);
      for (unsigned i = 0; i < I->nr_srcs; ++i)
         plain &= agx3_vary_reg32(I->src[i]) &&
                  agx_channels(I->src[i]) == 1;
      if (plain) {
         uint16_t src[3] = {0};
         enum agx3_alu_hint hints[3] = {0};
         for (unsigned i = 0; i < I->nr_srcs; ++i) {
            if (I->src[i].kill ||
                (I->src[i].cache && I->src[i].discard))
               AGX3_REFUSE_RET(I);
            src[i] = I->src[i].value;
            hints[i] = agx3_post_ra_hint(I->src[i]);
         }
         const enum agx3_alu_op op = I->op == AGX_OPCODE_FADD
            ? AGX3_ALU_FADD : I->op == AGX_OPCODE_FMUL
            ? AGX3_ALU_FMUL : AGX3_ALU_FMA;
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
         if (!agx3_alu_float_reg(enc, op, I->dest[0].value, src, hints,
                                  st->wait, bytes, &len)) {
            /* Carry exactly this operation's dependencies on its
             * destination, then retry the unchanged operands/hints.
             * Independent outstanding work remains in flight. */
            if (st->wait == 0 ||
                !agx3_retire_result_wait(emission, enc, I, st->wait) ||
                !agx3_alu_float_reg(enc, op, I->dest[0].value, src,
                                     hints, 0, bytes, &len))
               AGX3_REFUSE_RET(I);
         }
         if (!agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_ASR &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].value >= 1 &&
       !I->src[1].abs && !I->src[1].neg) {
      const agx_index dest=I->dest[0], src=I->src[0];
      const bool half = dest.size == AGX_SIZE_16 &&
                        src.size == AGX_SIZE_16;
      const bool word = agx3_vary_reg32(dest) && agx3_vary_reg32(src);
      /* G18G keeps j873's widening form. */
      const bool widen = ctx->key->dev.emit_agx3d &&
                         agx3_vary_reg32(dest) && src.size == AGX_SIZE_16;
      const bool mixed = enc == AGX_ISA_ENCODING_AGX3C &&
                         !ctx->key->dev.emit_agx3d &&
                         dest.size == AGX_SIZE_32 && src.size == AGX_SIZE_16;
      if ((!half && !word && !widen && !mixed) ||
          I->src[1].value > (src.size == AGX_SIZE_16 ? 15 : 31) ||
          !agx3_src_is_register(dest) || dest.memory ||
          !agx3_src_is_register(src) || src.memory ||
          src.abs || src.neg || agx_channels(dest)!=1 || agx_channels(src)!=1)
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      const bool encoded =
         enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d
         ? agx3_alu_asr_imm_m5(dest.value,src.value,
                               dest.size == AGX_SIZE_16,
                               src.size == AGX_SIZE_16,I->src[1].value,
                               agx3_post_ra_hint(src),dest.cache,st->wait,
                               bytes,&length)
         : half ? agx3_alu_asr_imm16(enc,dest.value,src.value,I->src[1].value,
                               agx3_post_ra_hint(src),dest.cache,st->wait,
                               bytes,&length)
         : widen
         ? agx3_alu_asr_imm16_to32(enc,dest.value,src.value,I->src[1].value,
                                   agx3_post_ra_hint(src),dest.cache,st->wait,
                                   bytes,&length)
         : agx3_alu_asr_imm32(enc,dest.value,src.value,I->src[1].value,
                               agx3_post_ra_hint(src),dest.cache,st->wait,
                               bytes,&length);
      if (!encoded ||
          !agx3_append(emission,bytes,length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native905 half-count logical shifts. Instruction selection has
    * explicitly subdivided the count before RA, preserving NIR's
    * modulo-16 semantics without borrowing or retyping a register. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       (I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       I->dest[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 &&
       I->src[2].size == AGX_SIZE_16 &&
       agx3_src_is_register(I->src[2])) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index shift = I->src[2], base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || !agx3_src_is_register(value) ||
          value.memory || agx_channels(value) != 1 ||
          agx_channels(shift) != 1 || shift.memory ||
          value.abs || value.neg || value.kill ||
          shift.abs || shift.neg || shift.kill ||
          (value.cache && value.discard) || (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t src[2] = {value.value, shift.value};
      const bool src_16[2] = {true, true};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(value), agx3_post_ra_hint(shift),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      const bool encoded = I->op == AGX_OPCODE_BFI
         ? (agx_is_g18g(&ctx->key->dev)
             ? agx3_alu_sll_reg_g18g
             : agx3_alu_sll_reg)(
               enc,dest.value,true,dest.cache,src,src_16,
                            hints,st->wait,bytes,&len)
         : (agx_is_g18g(&ctx->key->dev)
             ? agx3_alu_srl_reg_g18g
             : agx3_alu_srl_reg)(
               enc,dest.value,true,dest.cache,src,src_16,
                            hints,st->wait,bytes,&len);
      if (!encoded || !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_ASR &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       I->dest[0].size >= I->src[0].size &&
       I->dest[0].size <= AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32) {
      const agx_index dest = I->dest[0], value = I->src[0], shift = I->src[1];
      if (dest.memory || value.memory || shift.memory ||
          value.abs || value.neg || shift.abs || shift.neg ||
          agx_channels(dest) != 1 || agx_channels(value) != 1 ||
          agx_channels(shift) != 1 || value.kill || shift.kill ||
          (value.cache && value.discard) || (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(value), agx3_post_ra_hint(shift),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_asr_reg_m5(dest.value, value.value, shift.value,
              dest.size == AGX_SIZE_16, value.size == AGX_SIZE_16,
              hints, dest.cache, st->wait, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A dynamic signed-short shift stays narrow on G17P. Apple emits a
    * single mixed-width SRA: half value/result and a full register
    * count. Preserve the two source lifetimes and producer waits. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_ASR &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       I->dest[0].size == AGX_SIZE_16 &&
       I->src[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_32) {
      const agx_index dest = I->dest[0];
      const agx_index value = I->src[0];
      const agx_index shift = I->src[1];
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(value) || value.memory || value.abs ||
          value.neg || value.kill || agx_channels(value) != 1 ||
          !agx3_src_is_register(shift) || shift.memory || shift.abs ||
          shift.neg || shift.kill || agx_channels(shift) != 1 ||
          (value.cache && value.discard) ||
          (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);

      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(value), agx3_post_ra_hint(shift),
      };
      uint16_t w = st->wait;
      if (!agx3_retire_extra_waits(emission, enc, st->sl, st->res, st->instr_index,
                                   I, &w))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
      if (!agx3_alu_asr_reg16(enc, dest.value, value.value, shift.value,
                              hints, dest.cache, w, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native immediate-count logical shifts do not require a GPR
    * holding the count. Keep the existing masked/register forms below. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       (I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       I->dest[0].size == AGX_SIZE_32 &&
       I->src[1].size == AGX_SIZE_32 &&
       I->src[2].type == AGX_INDEX_IMMEDIATE &&
       I->src[2].value >= 1 && I->src[2].value <= 31) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index base = I->src[0], shift = I->src[2];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 || !agx3_src_is_register(value) ||
          value.memory || agx_channels(value) != 1 || value.abs ||
          value.neg || value.kill || (value.cache && value.discard) ||
          shift.memory || shift.abs || shift.neg || shift.kill ||
          shift.cache || shift.discard)
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_shift_imm32(enc, I->op == AGX_OPCODE_BFI,
                                dest.value, value.value, shift.value,
                                agx3_post_ra_hint(value), dest.cache, st->wait,
                                bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A scalar-half `1 << count` needs no register holding the one: the
    * native `sll` takes an immediate value. Apple reads a half count
    * here; NIR's uint count register supplies its low half. A cached
    * count keeps its register through the clear last-use bit. */
   if ((enc == AGX_ISA_ENCODING_AGX3B ||
        enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_BFI &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].value == 1 && agx3_src_is_register(I->src[2]) &&
       I->dest[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 &&
       I->src[2].size == AGX_SIZE_32) {
      const agx_index dest = I->dest[0];
      const agx_index one = I->src[1];
      const agx_index shift = I->src[2];
      const agx_index base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard || one.memory ||
          one.abs || one.neg || one.kill || one.cache || one.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(shift) || shift.memory ||
          agx_channels(shift) != 1 || shift.abs || shift.neg ||
          shift.kill || (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);

      const enum agx3_alu_hint shift_hint =
         shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
      if (!(enc == AGX_ISA_ENCODING_AGX3C
               ? agx3_alu_sll_one_m5(dest.value, true, shift.value,
                                     shift_hint, dest.cache, st->wait,
                                     bytes, &length)
               : agx3_alu_sll_one_reg32(enc, dest.value, shift.value,
                                        shift_hint, dest.cache, st->wait,
                                        bytes, &length)) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BFI &&
       I->nr_dests == 1 && I->nr_srcs == 3 && !I->bfi_mask &&
       !I->saturate && agx3_src_is_zero_imm(I->src[0]) &&
       agx3_src_is_register(I->dest[0]) &&
       agx3_src_is_register(I->src[1]) &&
       agx3_src_is_register(I->src[2]) &&
       I->dest[0].size <= AGX_SIZE_32 &&
       I->src[1].size <= AGX_SIZE_32 && I->src[2].size <= AGX_SIZE_32 &&
       agx_channels(I->dest[0]) == 1 &&
       agx_channels(I->src[1]) == 1 && agx_channels(I->src[2]) == 1 &&
       !I->src[1].abs && !I->src[1].neg &&
       !I->src[2].abs && !I->src[2].neg) {
      const uint16_t sources[] = {I->src[1].value, I->src[2].value};
      const bool half[] = {I->src[1].size == AGX_SIZE_16,
                           I->src[2].size == AGX_SIZE_16};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(I->src[1]),
         I->src[2].cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(I->src[2]),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_sll_reg(enc, I->dest[0].value,
            I->dest[0].size == AGX_SIZE_16, I->dest[0].cache,
            sources, half, hints, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* A register-count ushort shift is one native `sll`. The shift
    * operand remains uint in NIR; Apple's mixed-width witness pins
    * source A at width 1 and source B at width 2. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFI &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       agx3_src_is_register(I->src[2]) &&
       I->dest[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 &&
       I->src[2].size == AGX_SIZE_32) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index shift = I->src[2];
      const agx_index base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
          !agx3_src_is_register(value) || value.memory ||
          value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
          !agx3_src_is_register(shift) || shift.memory ||
          shift.size != AGX_SIZE_32 || agx_channels(shift) != 1 ||
          value.abs || value.neg || value.kill ||
          shift.abs || shift.neg || shift.kill ||
          (value.cache && value.discard) ||
          (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t src[2] = {value.value, shift.value};
      const bool src_16[2] = {true, false};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(value),
         /* This endpoint has a discard bit but no measured cache bit
          * for the count. NONE retains the physical register. */
         shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!(agx_is_g18g(&ctx->key->dev)
             ? agx3_alu_sll_reg_g18g
             : agx3_alu_sll_reg)(enc, dest.value, true, dest.cache, src,
                            src_16, hints, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native G17G/G18G half shifts through eight bits use the same
    * one/two shift-add lowering as G17P, with the GPU1 iadd32 field
    * layout. Keep the temporary in the destination's own half and
    * put the producer wait on the first instruction only. */
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BFI &&
       I->nr_dests == 1 && I->nr_srcs == 3 && !I->bfi_mask && !I->saturate &&
       agx3_src_is_zero_imm(I->src[0]) && agx3_half_reg(I->dest[0]) &&
       agx3_half_reg(I->src[1]) && !I->src[1].abs && !I->src[1].neg &&
       I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value <= 8) {
      const unsigned amount = I->src[2].value;
      const bool split = amount > 4;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_iadd16_gpu1(I->dest[0].value, split || I->dest[0].cache,
              true, 0, AGX3_ALU_HINT_NONE, I->src[1].value,
              agx3_post_ra_hint(I->src[1]), split ? amount - 4 : amount,
              st->wait, bytes, &len) || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      if (split &&
          (!agx3_alu_iadd16_gpu1(I->dest[0].value, I->dest[0].cache,
              true, 0, AGX3_ALU_HINT_NONE, I->dest[0].value,
              AGX3_ALU_HINT_DISCARD, 4, 0, bytes, &len) ||
           !agx3_append(emission, bytes, len)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * G17P has no single 16-bit shift-left form beyond four bits. Apple
    * lowers ushort << {5,6,7,8} to two witnessed shift-adds: first
    * 0 + (value << (amount - 4)), then 0 + (temporary << 4). The
    * destination itself is the temporary, so this remains post-RA and
    * needs no invented register. Alpha-to-coverage uses shift 7 here.
    */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFI &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       I->src[2].type == AGX_INDEX_IMMEDIATE &&
       I->src[2].value <= 8 && !I->src[2].memory &&
       !I->src[2].abs && !I->src[2].neg && !I->src[2].kill &&
       !I->src[2].cache && !I->src[2].discard) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
          !agx3_src_is_register(value) || value.memory ||
          value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
          value.abs || value.neg || value.kill ||
          (value.cache && value.discard))
         AGX3_REFUSE_RET(I);

      /* The shift-add descriptor has no source cache bit. A cached
       * value is still live, which the clear last-use bit preserves. */
      const bool split = I->src[2].value > 4;
      const struct agx3_shift_add_desc first = {
         .dest = dest.value,
         .dest_hint = split || dest.cache,
         .src0_immediate = true,
         .src0 = 0,
         .src1 = value.value,
         .src1_kill = value.discard,
         .shift = split ? I->src[2].value - 4 : I->src[2].value,
         .wait_mask = st->wait,
      };
      uint8_t bytes[AGX3_WRITEOUT_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_shift_add16_encode(enc, &first, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);

      if (split) {
         const struct agx3_shift_add_desc second = {
            .dest = dest.value,
            .dest_hint = dest.cache,
            .src0_immediate = true,
            .src0 = 0,
            .src1 = dest.value,
            .src1_kill = true,
            .shift = 4,
         };
         if (!agx3_shift_add16_encode(enc, &second, bytes, &len) ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
      }
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_EXTR &&
       I->nr_dests==1 && I->nr_srcs==3 && I->bfi_mask==0 &&
       I->src[2].type==AGX_INDEX_IMMEDIATE &&
       I->src[2].value>=1 && I->src[2].value<=31 &&
       !I->src[2].abs && !I->src[2].neg) {
      const agx_index dst=I->dest[0], lo=I->src[0], hi=I->src[1];
      if (!agx3_vary_reg32(dst) || !agx3_vary_reg32(lo) ||
          !agx3_vary_reg32(hi) || lo.abs || lo.neg || hi.abs || hi.neg ||
          agx_channels(dst)!=1 || agx_channels(lo)!=1 || agx_channels(hi)!=1)
         AGX3_REFUSE_RET(I);
      const uint16_t src[2]={lo.value,hi.value};
      const enum agx3_alu_hint hints[2]={agx3_post_ra_hint(lo),agx3_post_ra_hint(hi)};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
      if (!agx3_alu_extract_imm32(enc,dst.value,src,I->src[2].value,hints,
                                   dst.cache,st->wait,bytes,&length) ||
          !agx3_append(emission,bytes,length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Apple exposes [[sample_mask]] in bits 4..7 of a native ushort
    * special-register read.  Preserve that narrow value through the
    * exact immediate SRL instead of materializing a full-width shift
    * count, which selects a different G17P instruction domain. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_BFEIL && I->nr_dests == 1 &&
       I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       I->src[2].type == AGX_INDEX_IMMEDIATE &&
       I->src[2].value >= 1 && I->src[2].value <= 31 &&
       !I->src[2].memory && !I->src[2].abs && !I->src[2].neg &&
       !I->src[2].kill && !I->src[2].cache &&
       !I->src[2].discard) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
          !agx3_src_is_register(value) || value.memory ||
          value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
          value.abs || value.neg || value.kill ||
          (value.cache && value.discard))
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_srl_imm16(
             enc, dest.value, value.value, I->src[2].value,
             agx3_post_ra_hint(value), dest.cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BFEIL &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       I->imm == 0 && agx3_src_is_zero_imm(I->src[0]) &&
       agx3_src_is_register(I->src[2])) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index shift = I->src[2];
      if (!agx3_src_is_register(dest) || dest.memory ||
          (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
          agx_channels(dest) != 1 || !agx3_src_is_register(value) ||
          value.memory || value.abs || value.neg || value.kill ||
          (value.size != AGX_SIZE_16 && value.size != AGX_SIZE_32) ||
          agx_channels(value) != 1 || shift.memory ||
          shift.size != AGX_SIZE_32 || agx_channels(shift) != 1 ||
          shift.abs || shift.neg || shift.kill ||
          (value.cache && value.discard) || (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t src[2] = {value.value, shift.value};
      const bool narrow[2] = {value.size == AGX_SIZE_16, false};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(value),
         shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!agx3_alu_srl_reg(enc, dest.value, dest.size == AGX_SIZE_16,
                            dest.cache, src, narrow, hints, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Dynamic ushort right-shifts use the same mixed-width register
    * domain as SLL. agx_emit_alu has already changed the ordinary NIR
    * count into G17P's signed bitfield boundary. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_BFEIL &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->bfi_mask == 0 &&
       agx3_src_is_zero_imm(I->src[0]) &&
       agx3_src_is_register(I->src[2]) &&
       I->dest[0].size == AGX_SIZE_16 &&
       I->src[1].size == AGX_SIZE_16 &&
       I->src[2].size == AGX_SIZE_32) {
      const agx_index dest = I->dest[0], value = I->src[1];
      const agx_index shift = I->src[2];
      const agx_index base = I->src[0];
      if (I->saturate || base.memory || base.abs || base.neg ||
          base.kill || base.cache || base.discard ||
          !agx3_src_is_register(dest) || dest.memory ||
          dest.size != AGX_SIZE_16 || agx_channels(dest) != 1 ||
          !agx3_src_is_register(value) || value.memory ||
          value.size != AGX_SIZE_16 || agx_channels(value) != 1 ||
          shift.memory || shift.size != AGX_SIZE_32 ||
          agx_channels(shift) != 1 || value.abs || value.neg ||
          value.kill || shift.abs || shift.neg || shift.kill ||
          (value.cache && value.discard) ||
          (shift.cache && shift.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t src[2] = {value.value, shift.value};
      const bool src_16[2] = {true, false};
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(value),
         /* This endpoint has a discard bit but no grounded cache bit
          * for the shift count. NONE also retains the register, so a
          * cached post-RA source loses no lifetime information. */
         shift.cache ? AGX3_ALU_HINT_NONE : agx3_post_ra_hint(shift),
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!(agx_is_g18g(&ctx->key->dev)
             ? agx3_alu_srl_reg_g18g
             : agx3_alu_srl_reg)(enc, dest.value, true, dest.cache, src,
                            src_16, hints, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_BFEIL &&
       I->nr_dests == 1 && I->nr_srcs == 3 && I->imm >= 1 && I->imm <= 31 &&
       agx3_src_is_zero_imm(I->src[0])) {
      /* Read packed fields in place: GCC can otherwise read channel
       * bits from a partially populated local operand copy. */
      const agx_index *dest = &I->dest[0];
      const agx_index *value = &I->src[1];
      const agx_index *shift = &I->src[2];
      const bool reg_shift=agx3_src_is_register(*shift);
      if (!agx3_src_is_register(*dest) || dest->memory || dest->size != AGX_SIZE_32 ||
          agx_channels(*dest) != 1 || !agx3_src_is_register(*value) || value->memory ||
          (value->size != AGX_SIZE_32 &&
           !(enc == AGX_ISA_ENCODING_AGX3C && value->size == AGX_SIZE_16)) || agx_channels(*value) != 1 || value->abs || value->neg ||
          shift->memory || shift->abs || shift->neg || agx_channels(*shift) != 1 ||
          (reg_shift ? (shift->size != AGX_SIZE_32 &&
           !(enc == AGX_ISA_ENCODING_AGX3C && shift->size == AGX_SIZE_16))
           : shift->type != AGX_INDEX_IMMEDIATE))
         AGX3_REFUSE_RET(I);
      const enum agx3_alu_hint hints[2] = {
         agx3_post_ra_hint(*value), reg_shift ? agx3_post_ra_hint(*shift) : AGX3_ALU_HINT_NONE,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      const bool encoded = enc == AGX_ISA_ENCODING_AGX3C
         ? agx3_alu_bfeil_m5(dest->value,value->value,value->size == AGX_SIZE_16,
              shift->value,reg_shift,reg_shift && shift->size == AGX_SIZE_16,
              I->imm,hints,st->wait,bytes,&length)
         : agx3_alu_bfeil(enc,dest->value,value->value,shift->value,reg_shift,
                          I->imm,hints,st->wait,bytes,&length);
      if (!encoded ||
          !agx3_append(emission,bytes,length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /*
    * G17G find-top-one. Apple's clz kernels keep the source cached and
    * the destination cached, and only wait for the load in slot 8
    * (gpu/m5-xfb-alu3-reference.metal); retire any other producer
    * through the grounded band no-op first. A 16-bit source into a
    * 32-bit result is the witnessed half form into the low half plus
    * the same clear of the upper half the widening move uses.
    */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FFS &&
       I->nr_dests == 1 && I->nr_srcs == 1) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = src.size == AGX_SIZE_16;
      if (!agx3_vary_reg32(dest) || agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory || src.abs ||
          src.neg || agx_channels(src) != 1 ||
          (!half && (src.size != AGX_SIZE_32 || (src.value & 1))) ||
          (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t retire = st->wait & ~128u;
      if (retire && !agx3_retire_slots_noop(emission, enc, retire))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_ft1_gpu1(dest.value, src.value, half,
                             agx3_post_ra_hint(src), true, st->wait & 128u,
                             bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      if (half) {
         uint8_t hi[AGX3_STAGE_MAX_LENGTH], hi_len = 0;
         if (!agx3_clr(enc, dest.value + 1, true, false, hi, &hi_len) ||
             !agx3_append(emission, hi, hi_len))
            AGX3_REFUSE_RET(I);
      }
      st->outstanding &= (uint16_t)~retire;
      return AGX3_PACK_EMITTED;
   }

   /* G17G bit reverse of a word (agx3_alu_rev_gpu1), retiring any
    * producer but the witnessed slot-8 load first, like find-top-one
    * above. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITREV &&
       I->nr_dests == 1 && I->nr_srcs == 1) {
      const agx_index dest = I->dest[0], src = I->src[0];
      if (!agx3_vary_reg32(dest) || agx_channels(dest) != 1 ||
          !agx3_vary_reg32(src) || agx_channels(src) != 1 ||
          src.kill || (src.cache && src.discard))
         AGX3_REFUSE_RET(I);
      const uint16_t retire = st->wait & ~128u;
      if (retire && !agx3_retire_slots_noop(emission, enc, retire))
         AGX3_REFUSE_RET(I);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_rev_gpu1(dest.value, src.value,
                             agx3_post_ra_hint(src), dest.cache,
                             st->wait & 128u, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~retire;
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FFS &&
       I->nr_dests == 1 && I->nr_srcs == 1) {
      const agx_index dest = I->dest[0], src = I->src[0];
      if (!agx3_vary_reg32(dest) || !agx3_vary_reg32(src) ||
          agx_channels(dest) != 1 || agx_channels(src) != 1 ||
          (enc == AGX_ISA_ENCODING_AGX3B && dest.value == src.value) || src.abs || src.neg)
         AGX3_REFUSE_RET(I);
      /* Apple's G17P clz FT1 carries wait0 and the cached destination
       * form (op-enable 3). The uncached/op-enable-2 combination with
       * a discarded source exposes the internal logarithmic stage
       * instead of the completed bit index. */
      const uint16_t ft1_wait = enc == AGX_ISA_ENCODING_AGX3B ? st->wait | 1u : st->wait;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_ft1_u32(enc, dest.value, src.value,
                            agx3_post_ra_hint(src),
                            true, ft1_wait, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITREV &&
       I->nr_dests == 1 && I->nr_srcs == 1 &&
       agx3_vary_reg32(I->dest[0]) &&
       (agx3_vary_reg32(I->src[0]) || agx3_half_reg(I->src[0])) &&
       !I->src[0].abs && !I->src[0].neg) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_bitrev_m5(I->dest[0].value, I->src[0].value,
              I->src[0].size == AGX_SIZE_16,
              agx3_post_ra_hint(I->src[0]), I->dest[0].cache, st->wait,
              bytes, &length) || !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_POPCOUNT && I->nr_dests == 1 &&
       I->nr_srcs == 1) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16 &&
                        src.size == AGX_SIZE_16;
      const bool word = dest.size == AGX_SIZE_32 &&
                        src.size == AGX_SIZE_32;
      /* NIR's bit_count of a 16-bit value returns 32 bits. Apple counts
       * into the low half and clears the high half, as for
       * `uint(popcount(ushort))`. */
      const bool widened = dest.size == AGX_SIZE_32 &&
                           src.size == AGX_SIZE_16 &&
                           !(dest.value & 1);
      if ((!half && !word && !widened) ||
          !agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 ||
          !agx3_src_is_register(src) || src.memory ||
          agx_channels(src) != 1 || src.abs || src.neg)
         AGX3_REFUSE_RET(I);

      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      enum agx3_alu_hint hint = agx3_post_ra_hint(src);
      /* An in-place popcount consumes the old register value and then
       * overwrites it. Post-RA has no separate lifetime hint on this
       * tied operand, but that is exactly the witnessed discard form
       * (including Apple's U32 dest=src=0 program). */
      if (hint == AGX3_ALU_HINT_NONE && dest.value == src.value)
         hint = AGX3_ALU_HINT_DISCARD;
      bool encoded = (agx_is_g18g(&ctx->key->dev)
                       ? agx3_alu_popcount_g18g
                       : agx3_alu_popcount)(enc, dest.value, src.value,
                                       !word, hint, dest.cache, st->wait,
                                       bytes, &length);
      /* G17G's cbs has witnessed waits for load slots 1 and 8 only.
       * Retire any other producer through the grounded band no-op
       * first, as the uniform copies do, and count with no wait. */
      if (!encoded && enc == AGX_ISA_ENCODING_AGX3C &&
          ctx->key->dev.emit_agx3d && st->wait &&
          agx3_retire_slots_noop(emission, enc, st->wait)) {
         st->outstanding &= (uint16_t)~st->wait;
         encoded = (agx_is_g18g(&ctx->key->dev)
                     ? agx3_alu_popcount_g18g
                     : agx3_alu_popcount)(
                       enc, dest.value, src.value, !word,
                                     hint, dest.cache, 0, bytes,
                                     &length);
      }
      if (!encoded || !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      if (widened &&
          (!agx3_clr(enc, dest.value + 1, true, false, bytes,
                     &length) ||
           !agx3_append(emission, bytes, length)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* ~(a | b) (truth table 0x1), a & ~b (0x2), ~a & b (0x4),
    * ~(a & b) (0x7), ~(a ^ b) (0x9), a | ~b (0xB) and ~a | b (0xD) are
    * their own two-register instructions; the one-sided forms take
    * their operands in order. */
   const unsigned bit_tt = I->truth_table & 0xfu;
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       (bit_tt == 2 || bit_tt == 4 || bit_tt == 11 || bit_tt == 13) &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_half_reg(I->dest[0]) && agx3_half_reg(I->src[0]) &&
       agx3_half_reg(I->src[1]) && !I->src[0].abs && !I->src[0].neg &&
       !I->src[1].abs && !I->src[1].neg) {
      const bool swap = bit_tt == 4 || bit_tt == 13;
      const agx_index a = I->src[swap ? 1 : 0];
      const agx_index b = I->src[swap ? 0 : 1];
      const uint16_t src[] = {a.value, b.value};
      const enum agx3_alu_hint hints[] = {
         agx3_post_ra_hint(a), agx3_post_ra_hint(b)};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (!agx3_alu_complement_reg16_m5(
             I->dest[0].value, src, hints, bit_tt == 11 || bit_tt == 13,
             I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_BITOP &&
       (bit_tt == 0x1u || bit_tt == 0x2u || bit_tt == 0x4u ||
        bit_tt == 0x7u || bit_tt == 0x9u || bit_tt == 0xbu ||
        bit_tt == 0xdu) &&
       I->nr_dests == 1 && I->nr_srcs == 2) {
      const agx_index dest = I->dest[0];
      if (!agx3_vary_reg32(dest) || agx_channels(dest) != 1)
         AGX3_REFUSE_RET(I);
      const bool swap = bit_tt == 0x4u || bit_tt == 0xdu;
      uint16_t src[2];
      bool half[2];
      enum agx3_alu_hint hints[2];
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index x = I->src[swap ? 1 - i : i];
         half[i] = x.size == AGX_SIZE_16;
         if (!(agx3_vary_reg32(x) ||
               (enc == AGX_ISA_ENCODING_AGX3C &&
                (bit_tt == 2 || bit_tt == 4) && agx3_half_reg(x))) ||
             agx_channels(x) != 1 || x.abs || x.neg)
            AGX3_REFUSE_RET(I);
         src[i] = x.value;
         hints[i] = agx3_post_ra_hint(x);
      }
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      bool encoded;
      switch (bit_tt) {
      case 0x1u:
         encoded = agx3_alu_bit_nor(enc, dest.value, src, hints,
                                    dest.cache, st->wait, bytes, &length);
         break;
      case 0x2u:
      case 0x4u:
         encoded = enc == AGX_ISA_ENCODING_AGX3C
            ? agx3_alu_bit_andc_width_m5(dest.value, src, half, hints,
                                        dest.cache, st->wait, bytes, &length)
            : agx3_alu_bit_andc(enc, dest.value, src, hints,
                                dest.cache, st->wait, bytes, &length);
         break;
      case 0x7u:
         encoded = agx3_alu_bit_nand(enc, dest.value, src, hints,
                                     dest.cache, st->wait, bytes, &length);
         break;
      case 0x9u:
         encoded = agx3_alu_bit_xnor(enc, dest.value, src, hints,
                                     dest.cache, st->wait, bytes, &length);
         break;
      default:
         encoded = agx3_alu_bit_orc(enc, dest.value, src, hints,
                                    dest.cache, st->wait, bytes, &length);
         break;
      }
      if (!encoded || !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* The same seven tables on AGX3c, at either width: all halves or all
    * aligned 32-bit pairs (agx3_alu_bit_named_gpu1, m6-bitops). */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_BITOP &&
       (bit_tt == 0x1u || bit_tt == 0x2u || bit_tt == 0x4u ||
        bit_tt == 0x7u || bit_tt == 0x9u || bit_tt == 0xbu ||
        bit_tt == 0xdu) &&
       I->nr_dests == 1 && I->nr_srcs == 2) {
      const agx_index dest = I->dest[0];
      const bool word = dest.size == AGX_SIZE_32;
      if (!agx3_src_is_register(dest) || dest.memory ||
          agx_channels(dest) != 1 ||
          (dest.size != AGX_SIZE_16 && !word))
         AGX3_REFUSE_RET(I);
      const bool swap = bit_tt == 0x4u || bit_tt == 0xdu;
      uint16_t src[2];
      enum agx3_alu_hint hints[2];
      for (unsigned i = 0; i < 2; ++i) {
         const agx_index x = I->src[swap ? 1 - i : i];
         if (!agx3_src_is_register(x) || x.memory ||
             agx_channels(x) != 1 || x.abs || x.neg ||
             x.size != dest.size || (x.cache && x.discard))
            AGX3_REFUSE_RET(I);
         src[i] = x.value;
         hints[i] = agx3_post_ra_hint(x);
      }
      const unsigned table = bit_tt == 0x4u ? 0x2u
                             : bit_tt == 0xdu ? 0xbu
                                              : bit_tt;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_bit_named_gpu1(table, word, dest.value, dest.cache,
                                   src[0], hints[0], src[1], hints[1],
                                   st->wait, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

static enum agx3_pack_result
agx3_pack_alu_fallback(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   /* Half negation is encoded as -src + -0.0h. This exact f16add.ep2
    * form keeps the source lifetime and producer wait in one native
    * instruction, including the matrix-inverse and faceforward paths.
    * On AGX3C saturate(half) is the same form with its saturate bit. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 &&
       (!I->saturate || enc == AGX_ISA_ENCODING_AGX3C) &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 &&
       agx_channels(I->dest[0]) == 1) {
      const bool immediate_first =
         I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index imm = I->src[immediate_first ? 0 : 1];
      const agx_index reg = I->src[immediate_first ? 1 : 0];
      if (imm.type == AGX_INDEX_IMMEDIATE && imm.value == 0x80 &&
          !imm.memory && !imm.abs && !imm.neg &&
          agx3_src_is_register(reg) && !reg.memory &&
          reg.size == AGX_SIZE_16 && agx_channels(reg) == 1 &&
          ((enc == AGX_ISA_ENCODING_AGX3C) || (!reg.abs && reg.neg)) &&
          !reg.kill &&
          !(reg.cache && reg.discard)) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         bool encoded = enc == AGX_ISA_ENCODING_AGX3C
            ? agx3_alu_half_add_negzero_m5(
                 I->dest[0].value, reg.value, reg.abs, reg.neg,
                 I->saturate, agx3_post_ra_hint(reg), I->dest[0].cache, st->wait, bytes, &len)
            : agx3_alu_hadd_neg_negzero(
                enc, I->dest[0].value, reg.value,
                agx3_post_ra_hint(reg), I->dest[0].cache, st->wait,
                bytes, &len);
         if (!encoded ||
             !agx3_append(emission, bytes, len))
            AGX3_REFUSE_RET(I);
         return AGX3_PACK_EMITTED;
      }
   }

   /* Apple keeps exact half constants in scalar halves through an f32
    * add (widening first would cost a full register). Addition
    * commutes, so either IR order maps to the native f32/f16 order. */
   if ((enc == AGX_ISA_ENCODING_AGX3B ||
        enc == AGX_ISA_ENCODING_AGX3C) && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       !I->src[0].abs && !I->src[0].neg &&
       !I->src[1].abs && !I->src[1].neg &&
       ((agx3_mod_src_ok(I->src[0]) && agx3_half_reg(I->src[1])) ||
        (agx3_half_reg(I->src[0]) && agx3_mod_src_ok(I->src[1])))) {
      const bool half_first = agx3_half_reg(I->src[0]);
      const agx_index f32 = I->src[half_first ? 1 : 0];
      const agx_index f16 = I->src[half_first ? 0 : 1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (f32.kill || f16.kill || (f32.cache && f32.discard) ||
          (f16.cache && f16.discard) ||
          !agx3_alu_fadd_mixed_half(
             enc, I->dest[0].value, f32.value, f16.value,
             agx3_post_ra_hint(f32), agx3_post_ra_hint(f16),
             I->dest[0].cache, st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* G17P adds an absolute f32 source directly to a scalar-half
    * register. Keep the half in its assigned register rather than
    * widening it through a temporary. */
   if (enc == AGX_ISA_ENCODING_AGX3B && I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_vary_reg32(I->dest[0]) &&
       agx_channels(I->dest[0]) == 1 &&
       agx3_mod_src_ok(I->src[0]) &&
       agx_channels(I->src[0]) == 1 && I->src[0].abs &&
       !I->src[0].neg && agx3_half_reg(I->src[1]) &&
       agx_channels(I->src[1]) == 1 && !I->src[1].abs &&
       !I->src[1].neg) {
      const agx_index dst = I->dest[0], f32 = I->src[0],
                      f16 = I->src[1];
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (f32.kill || f16.kill || (f32.cache && f32.discard) ||
          (f16.cache && f16.discard) ||
          !agx3_alu_fadd_abs_mixed_half(
             enc, dst.value, f32.value, f16.value,
             agx3_post_ra_hint(f32), agx3_post_ra_hint(f16), dst.cache,
             st->wait, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Native910: preserve the half-precision fabs/negfabs operation,
    * including -0, rather than widening it or clearing bits in software. */
   if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
       I->op == AGX_OPCODE_FADD &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
       I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
       agx3_src_is_register(I->src[0]) && !I->src[0].memory &&
       I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
       (I->src[0].abs || I->src[0].neg) && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard) &&
       I->src[1].type == AGX_INDEX_IMMEDIATE &&
       I->src[1].value == 0x80 && !I->src[1].memory &&
       !I->src[1].abs && !I->src[1].neg && !I->src[1].kill &&
       !I->src[1].cache && !I->src[1].discard) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      if (!I->src[0].abs) {
          if (!agx3_alu_hadd_neg_negzero(enc,I->dest[0].value,I->src[0].value,
                                        agx3_post_ra_hint(I->src[0]),
                                        I->dest[0].cache,st->wait,bytes,&len) ||
              !agx3_append(emission,bytes,len))
             AGX3_REFUSE_RET(I);
          return AGX3_PACK_EMITTED;
       }
       if (!agx3_alu_hadd_abs_negzero(enc,I->dest[0].value,I->src[0].value,
                                    agx3_post_ra_hint(I->src[0]),
                                    I->src[0].neg,I->dest[0].cache,
                                    st->wait,bytes,&len) ||
          !agx3_append(emission,bytes,len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

    /* Native973: all-half comparison and arbitrary half-register
     * alternatives. Preserve arm bits and invert by swapping arms. */
    if (enc == AGX_ISA_ENCODING_AGX3C && ctx->key->dev.emit_agx3d &&
        I->op == AGX_OPCODE_FCMPSEL &&
        I->nr_dests == 1 && I->nr_srcs == 4 && !I->saturate &&
        agx3_src_is_register(I->dest[0]) && !I->dest[0].memory &&
        I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1) {
       bool half_registers = true;
       uint16_t src[4];
       enum agx3_alu_hint hints[4];
       for (unsigned s=0;s<4;++s) {
          const agx_index v=I->src[s];
          half_registers &= agx3_src_is_register(v) && !v.memory &&
             v.size == AGX_SIZE_16 && agx_channels(v) == 1 &&
             !v.abs && !v.neg && !v.kill && !(v.cache && v.discard);
          src[s]=v.value;hints[s]=agx3_post_ra_hint(v);
       }
       uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0;
       if (half_registers && agx3_alu_fcmp_select_half_gpu1(
              I->dest[0].value,src,hints,I->fcond,I->invert_cond,
              I->dest[0].cache,st->wait,bytes,&len)) {
          if (!agx3_append(emission,bytes,len)) AGX3_REFUSE_RET(I);
          return AGX3_PACK_EMITTED;
       }
    }

   /* Modified G17P ADD/FMA carry their native dependency waits and the
    * actual post-RA hints, without draining unrelated texture slots. */
   const int mod = agx3_try_float_modifier(emission, enc, I, st->wait,
                                           agx_is_g18g(&ctx->key->dev));
   if (mod == 1)
      return AGX3_PACK_EMITTED;
   if (mod < 0)
      AGX3_REFUSE_RET(I);

   /* AGX3C NOT of a half or a 32-bit register that the selector cannot
    * encode is C's unary `inv` (the bitop's second source is ignored by
    * the truth table). The selector's own forms are kept wherever they
    * exist, so only the former refusals change. */
   if (enc == AGX_ISA_ENCODING_AGX3C && I->op == AGX_OPCODE_BITOP &&
       (I->truth_table & 0xfu) == AGX3_SEL_TT_NOT &&
       I->nr_dests == 1 && I->nr_srcs >= 1 && !I->saturate &&
       I->dest[0].size == I->src[0].size &&
       (I->dest[0].size == AGX_SIZE_16 ||
        I->dest[0].size == AGX_SIZE_32) &&
       agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1 &&
       agx3_src_is_register(I->dest[0]) &&
       agx3_src_is_register(I->src[0]) && !I->src[0].abs &&
       !I->src[0].neg && !I->src[0].kill &&
       !(I->src[0].cache && I->src[0].discard)) {
      struct agx3_sel_instr S;
      struct util_dynarray trial;
      util_dynarray_init(&trial, NULL);
      const bool selected = agx3_translate_instr(I, &S) &&
                            agx3_emit_alu(&trial, enc, &S, st->wait,
                                          agx_is_g18g(&ctx->key->dev));
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      const bool ok = selected
         ? trial.size <= UINT8_MAX &&
              agx3_append(emission, (const uint8_t *)trial.data,
                          (uint8_t)trial.size)
         : (agx3_alu_bit_not_m5(I->dest[0].value, I->src[0].value,
                                I->dest[0].size == AGX_SIZE_32,
                                agx3_post_ra_hint(I->src[0]),
                                I->dest[0].cache, st->wait, bytes, &len) &&
            agx3_append(emission, bytes, len));
      util_dynarray_fini(&trial);
      if (!ok)
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* An immediate has no source dependency but can still overwrite a
    * pending result. G18G immediate moves have no embedded wait form; the
    * M5 keeps its embedded wait. */
   const bool retire_immediate_wait =
      enc == AGX_ISA_ENCODING_AGX3C && agx_is_g18g(&ctx->key->dev) &&
      I->op == AGX_OPCODE_MOV_IMM &&
      I->nr_dests == 1 &&
      agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) && st->wait;
   const uint16_t alu_wait = retire_immediate_wait ? 0 : st->wait;
   if (retire_immediate_wait) {
      uint8_t retire_bytes[AGX3_ALU_MAX_LENGTH], retire_length = 0;
      if (!agx3_band_carrier(enc, I->dest[0].value, st->wait,
                             retire_bytes, &retire_length) ||
          !agx3_append(emission, retire_bytes, retire_length))
         AGX3_REFUSE_RET(I);
   }

   struct agx3_sel_instr S;
   if (!agx3_translate_instr(I, &S) ||
       !agx3_emit_alu(emission, enc, &S, alu_wait,
                      agx_is_g18g(&ctx->key->dev))) {
      if (agx_compiler_debug & AGX_DBG_SHADERS)
         fprintf(stderr,
                 "ALUX %s wait=%x shift=%u d{v=%u sz=%u} s0{t=%u v=%u "
                 "sz=%u abs=%u} s1{t=%u v=%u sz=%u abs=%u}\n",
                 agx_opcodes_info[I->op].name, st->wait, I->shift,
                 I->dest[0].value, I->dest[0].size, I->src[0].type,
                 I->src[0].value, I->src[0].size, I->src[0].abs,
                 I->src[1].type, I->src[1].value, I->src[1].size,
                 I->src[1].abs);
      AGX3_REFUSE_RET(I);
   }
   return AGX3_PACK_EMITTED;
}

enum agx3_pack_result
agx3_pack_alu_instr(struct agx3_pack_state *st)
{
   enum agx3_pack_result r;

   if ((r = agx3_pack_alu_unary_and_imad(st)) != AGX3_PACK_UNHANDLED)
      return r;
   if ((r = agx3_pack_alu_fcmpsel(st)) != AGX3_PACK_UNHANDLED)
      return r;
   if ((r = agx3_pack_alu_int_and_moves(st)) != AGX3_PACK_UNHANDLED)
      return r;
   if ((r = agx3_pack_alu_float_and_convert(st)) != AGX3_PACK_UNHANDLED)
      return r;
   if ((r = agx3_pack_alu_bitop_and_shift(st)) != AGX3_PACK_UNHANDLED)
      return r;
   return agx3_pack_alu_fallback(st);
}
