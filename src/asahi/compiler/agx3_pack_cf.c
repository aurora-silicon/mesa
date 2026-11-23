/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 control flow, scoreboard wait retirement, branch fixups, early-compare
 * rewriting, and loop-head hints.
 */

#include "agx3_pack_internal.h"
#include "agx3_cf_core.h"
#include "agx3_compose/agx3_decode.h"
#include "agx3_pixel_core.h"
#include "agx3_vary_core.h"

static __thread bool agx3_carrier1140;
static __thread uint16_t agx3_r0_inflight;

void
agx3_carrier_init(bool carrier1140)
{
   agx3_carrier1140 = carrier1140;
   agx3_r0_inflight = 0;
}

bool
agx3_carrier_is_1140(void)
{
   return agx3_carrier1140;
}

void
agx3_carrier_update_r0(uint16_t issued, uint16_t wait)
{
   agx3_r0_inflight |= issued;
   agx3_r0_inflight &= (uint16_t)~wait;
}

bool
agx3_band_carrier(enum agx_isa_encoding enc, uint16_t reg, uint16_t mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_carrier1140 && enc == AGX_ISA_ENCODING_AGX3C
             ? agx3_alu_band_identity_live_gpu1(reg, mask, out, len)
             : agx3_alu_band_identity_wait(enc, reg, mask, out, len);
}

/* The AGX3 compare forms are 32-bit and address an eight-bit half-register
 * index, so all 128 architectural 32-bit registers are reachable. Source
 * modifiers and memory/uniform operands are refused here. */
static bool
agx3_cf_reg32(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_32 &&
          (src.value & 1) == 0 && src.value < 256 && !src.abs && !src.neg;
}

/* Native 16-bit register comparison addresses all 256 individual halves. */
static bool
agx3_cf_reg16(agx_index src)
{
   return agx3_src_is_register(src) && !src.memory && src.size == AGX_SIZE_16 &&
          src.value < 256 && !src.abs && !src.neg;
}

/* A float register that can be canonicalized against a minifloat. Negation
 * is admitted here because agx3_emit_cf_compare() moves it across the
 * relation and constant before calling the unmodified native encoder. */
static bool
agx3_cf_immediate_freg(agx_index src)
{
   if (!agx3_src_is_register(src) || src.memory || src.value >= 256 || src.abs)
      return false;
   if (src.size == AGX_SIZE_16)
      return true;
   return src.size == AGX_SIZE_32 && !(src.value & 1);
}

static bool
agx3_cf_map_icond(enum agx_icond in, enum agx3_cf_icond *out)
{
   switch (in) {
   case AGX_ICOND_UEQ:
   case AGX_ICOND_SEQ:
      *out = AGX3_CF_ICOND_EQ;
      return true;
   case AGX_ICOND_ULT:
      *out = AGX3_CF_ICOND_ULT;
      return true;
   case AGX_ICOND_UGT:
      *out = AGX3_CF_ICOND_UGT;
      return true;
   case AGX_ICOND_SLT:
      *out = AGX3_CF_ICOND_SLT;
      return true;
   case AGX_ICOND_SGT:
      *out = AGX3_CF_ICOND_SGT;
      return true;
   }

   return false;
}

/* Canonicalize `immediate relation register` to the already-grounded
 * `register swapped-relation immediate` encoding.  Equality is symmetric;
 * the four ordered relations exchange LT and GT without changing signedness. */
static bool
agx3_cf_swap_icond(enum agx3_cf_icond *cond)
{
   switch (*cond) {
   case AGX3_CF_ICOND_EQ:
      return true;
   case AGX3_CF_ICOND_ULT:
      *cond = AGX3_CF_ICOND_UGT;
      return true;
   case AGX3_CF_ICOND_UGT:
      *cond = AGX3_CF_ICOND_ULT;
      return true;
   case AGX3_CF_ICOND_SLT:
      *cond = AGX3_CF_ICOND_SGT;
      return true;
   case AGX3_CF_ICOND_SGT:
      *cond = AGX3_CF_ICOND_SLT;
      return true;
   }

   return false;
}

/* Normalize Mesa's seven float relations to AGX3's three base conditions,
 * or-NaN selector, and the consuming cnd instruction's invert bit. */
static bool
agx3_cf_map_fcond(enum agx_fcond in, enum agx3_cf_fcond *out, bool *or_nan,
                  bool *invert)
{
   *or_nan = false;

   switch (in) {
   case AGX_FCOND_EQ:
      *out = AGX3_CF_FCOND_EQ;
      return true;
   case AGX_FCOND_LT:
      *out = AGX3_CF_FCOND_LT;
      return true;
   case AGX_FCOND_GT:
      *out = AGX3_CF_FCOND_GT;
      return true;
   case AGX_FCOND_LTN:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      return true;
   case AGX_FCOND_GTN:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      return true;
   case AGX_FCOND_GE:
      *out = AGX3_CF_FCOND_LT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   case AGX_FCOND_LE:
      *out = AGX3_CF_FCOND_GT;
      *or_nan = true;
      *invert = !*invert;
      return true;
   }

   return false;
}

static bool
agx3_cf_is_zero(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value == 0 && !src.abs &&
          !src.neg;
}

/* A float constant the control-flow compare can carry in f18: the ordinary
 * minifloat this ISA uses for immediate float operands everywhere else.
 * Zero is included, and keeps producing the bytes it always did. */
static bool
agx3_cf_fminifloat(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value <= 0xff && !src.abs &&
          !src.neg;
}

/* A constant the witnessed 16-bit compare can carry. Above 255 Apple changes
 * the instruction, so anything wider stays refused. */
static bool
agx3_cf_imm8(agx_index src)
{
   return src.type == AGX_INDEX_IMMEDIATE && src.value <= UINT8_MAX &&
          !src.abs && !src.neg;
}

/* An aligned 32-bit uniform the icmp first-operand uniform class addresses. */
static bool
agx3_cf_uniform32(agx_index src)
{
   return src.type == AGX_INDEX_UNIFORM && src.size == AGX_SIZE_32 &&
          agx_channels(src) == 1 && !(src.value & 1) && src.value < 256 &&
          !src.abs && !src.neg;
}

/* Emit the compare half of a fused AGX IR control-flow instruction. The flag
 * is consumed by the immediately following cnd instruction. Loop predicates
 * use distinct persistent slots by nesting depth; ordinary ifs reuse slot 0. */
static bool
agx3_emit_cf_compare(struct util_dynarray *emission, enum agx_isa_encoding enc,
                     const agx_instr *I, bool is_float, unsigned flag_slot,
                     bool *invert, bool g18g)
{
   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;

   if (I->nr_srcs != 2)
      return false;

   if (is_float && enc == AGX_ISA_ENCODING_AGX3C &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].neg && !I->src[1].neg &&
       !I->src[0].kill && !I->src[1].kill &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       (I->src[0].size == AGX_SIZE_16 || I->src[0].size == AGX_SIZE_32) &&
       (I->src[1].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_32)) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool half[] = {I->src[0].size == AGX_SIZE_16,
                           I->src[1].size == AGX_SIZE_16};
      const bool absolute[] = {I->src[0].abs, I->src[1].abs};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard},
      };
      return agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) &&
             agx3_cf_fcmp_width_m5(cond, or_nan, flag_slot, src, half,
                                    absolute, hints, 0, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (!is_float && enc == AGX_ISA_ENCODING_AGX3C &&
       (I->src[0].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_16) &&
       (agx3_cf_reg16(I->src[0]) || agx3_cf_reg32(I->src[0])) &&
       (agx3_cf_reg16(I->src[1]) || agx3_cf_reg32(I->src[1]))) {
      enum agx3_cf_icond cond;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool half[] = {I->src[0].size == AGX_SIZE_16,
                           I->src[1].size == AGX_SIZE_16};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard},
      };
      return agx3_cf_map_icond(I->icond, &cond) &&
             agx3_cf_icmp_width_m5(cond, flag_slot, src, half, hints, bytes, &len) &&
             agx3_append(emission, bytes, len);
   }

   if (!is_float && agx3_cf_reg16(I->src[0]) && agx3_cf_reg16(I->src[1]) &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1) {
      enum agx3_cf_icond condition;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard},
      };

      /* agx_lower_pseudo spells an unconditional G17P loop break as the
       * integer identity r0l == r0l.  Register allocation may mark both
       * occurrences cached and killed because the fixed architectural half
       * has no SSA definition.  Native half comparisons ground the live/live
       * register form, and cache/discard are only operand lifetime hints, so
       * normalize this compiler-generated identity back to that form.  Keep
       * every non-identity comparison's independently allocated hints. */
      const bool unconditional_break =
         I->op == AGX_OPCODE_BREAK_IF_ICMP && !I->invert_cond &&
         (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ) &&
         I->src[0].type == I->src[1].type &&
         I->src[0].value == I->src[1].value &&
         I->src[0].size == I->src[1].size;
      if (unconditional_break) {
         hints[0] = (struct agx3_cf_source_hints){false, false};
         hints[1] = (struct agx3_cf_source_hints){false, false};
      }
      if ((!unconditional_break && (I->src[0].kill || I->src[1].kill)) ||
          !agx3_cf_map_icond(I->icond, &condition) ||
          !agx3_cf_icmp_reg16(enc, condition, flag_slot, src, hints, 0, bytes,
                              &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* Branching on a 16-bit integer against a constant (Mesa keeps booleans
    * and small ints in halves: `if (b)` is `r0h != 0`). The constant may be
    * on either side; it is moved to the right with the relation swapped, as
    * on the 32-bit path. agx3_cf_icmp_imm16_runtime() refuses immediates and
    * relations it cannot encode. */
   if (!is_float &&
       ((agx3_cf_reg16(I->src[0]) && agx3_cf_imm8(I->src[1])) ||
        (agx3_cf_imm8(I->src[0]) && agx3_cf_reg16(I->src[1])))) {
      const bool imm_first = !agx3_cf_reg16(I->src[0]);
      const agx_index source = I->src[imm_first ? 1 : 0];
      const agx_index constant = I->src[imm_first ? 0 : 1];
      enum agx3_cf_icond cond16;
      if (source.kill || !agx3_cf_map_icond(I->icond, &cond16) ||
          (imm_first && !agx3_cf_swap_icond(&cond16)) ||
          !(g18g
             ? agx3_cf_icmp_imm16_runtime_g18g
             : agx3_cf_icmp_imm16_runtime)(enc, cond16, flag_slot, source.value,
             (uint8_t)constant.value,
             (struct agx3_cf_source_hints){source.cache, source.discard},
             0, bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* Branching on a float against a constant: f18, the compare's second
    * operand, holds the minifloat. */
   if (is_float && (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       ((agx3_cf_fminifloat(I->src[1]) &&
         agx3_cf_immediate_freg(I->src[0])) ||
        (agx3_cf_fminifloat(I->src[0]) &&
         agx3_cf_immediate_freg(I->src[1])))) {
      const bool zero_first=agx3_cf_fminifloat(I->src[0]);
      const agx_index source=I->src[zero_first?1:0];
      const agx_index constant=I->src[zero_first?0:1];
      enum agx_fcond relation=I->fcond;
      if(zero_first) {
         switch(relation) {
         case AGX_FCOND_LT:relation=AGX_FCOND_GT;break;
         case AGX_FCOND_GT:relation=AGX_FCOND_LT;break;
         case AGX_FCOND_LE:relation=AGX_FCOND_GE;break;
         case AGX_FCOND_GE:relation=AGX_FCOND_LE;break;
         case AGX_FCOND_LTN:relation=AGX_FCOND_GTN;break;
         case AGX_FCOND_GTN:relation=AGX_FCOND_LTN;break;
         default:break;
         }
      }
      uint8_t minifloat = (uint8_t)constant.value;
      /* Move a register negate across the comparison: -x REL c is
       * x REVERSE(REL) -c. The minifloat sign is its high bit, so this keeps
       * the existing native compare-immediate instruction and covers half
       * inverse control flow without a materialized negate. */
      if (source.neg) {
         minifloat ^= 0x80;
         switch (relation) {
         case AGX_FCOND_LT: relation=AGX_FCOND_GT; break;
         case AGX_FCOND_GT: relation=AGX_FCOND_LT; break;
         case AGX_FCOND_LE: relation=AGX_FCOND_GE; break;
         case AGX_FCOND_GE: relation=AGX_FCOND_LE; break;
         case AGX_FCOND_LTN: relation=AGX_FCOND_GTN; break;
         case AGX_FCOND_GTN: relation=AGX_FCOND_LTN; break;
         default: break;
         }
      }
      /* Both signed zeros compare identically. Apple's native -half == 0
       * branch removes the negate and keeps the immediate at positive zero,
       * whose compact field form is the one this endpoint accepts. */
      if ((minifloat & 0x7f) == 0)
         minifloat = 0;
      enum agx3_cf_fcond condition;bool or_nan;
      const bool mapped =
         agx3_cf_map_fcond(relation, &condition, &or_nan, invert);
      const bool encoded =
         mapped && !source.kill && !source.abs && agx_channels(source) == 1 &&
         agx3_cf_fcmp_zero_imm(
            enc, condition, or_nan, flag_slot, source.value,
            source.size == AGX_SIZE_16, minifloat,
            (struct agx3_cf_source_hints){source.cache, source.discard}, 0,
            bytes, &len);
      if (!encoded) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr,
                    "fcmp immediate refused: src=%u size=%u channels=%u "
                    "kill=%u abs=%u neg=%u cache=%u discard=%u imm=%u "
                    "relation=%u mapped=%u condition=%u or_nan=%u slot=%u\n",
                    source.value, source.size, agx_channels(source),
                    source.kill, source.abs, source.neg, source.cache,
                    source.discard, minifloat, relation, mapped,
                    mapped ? condition : 0, mapped ? or_nan : 0, flag_slot);
         return false;
      }
      return agx3_append(emission,bytes,len);
   }

   if (is_float && (enc == AGX_ISA_ENCODING_AGX3B ||
                    enc == AGX_ISA_ENCODING_AGX3C) &&
       agx3_src_is_register(I->src[0]) &&
       agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_32 &&
       I->src[1].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs &&
       (I->src[0].neg != I->src[1].neg) &&
       !(I->src[0].cache && I->src[0].discard) && !I->src[0].kill &&
       !(I->src[1].cache && I->src[1].discard) && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool neg[] = {I->src[0].neg, I->src[1].neg};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg_neg(enc, cond, or_nan, flag_slot, src, neg, hints,
                                bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* An absolute comparison source: like `fsel`, the control-flow compare
    * puts ABS on a longer encoding point, so it has its own encoder (the
    * plain one below would drop the modifier). ABS with NEG is refused: NEG
    * exists only on the short form. */
   if (is_float && enc == AGX_ISA_ENCODING_AGX3B &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       (I->src[0].abs || I->src[1].abs) &&
       !I->src[0].neg && !I->src[1].neg &&
       !(I->src[0].cache && I->src[0].discard) && !I->src[0].kill &&
       !(I->src[1].cache && I->src[1].discard) && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const bool abs[] = {I->src[0].abs, I->src[1].abs};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg_abs(enc, cond, or_nan, flag_slot, src, abs, hints,
                                0, bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* Two half-float registers compare natively, as Apple's G17P compiler
    * does for `if (h0 < h1)`, with both sources' actual lifetimes. */
   if (is_float && (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       agx3_src_is_register(I->src[0]) && agx3_src_is_register(I->src[1]) &&
       I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
       agx_channels(I->src[0]) == 1 && agx_channels(I->src[1]) == 1 &&
       !I->src[0].memory && !I->src[1].memory &&
       !I->src[0].abs && !I->src[1].abs && !I->src[0].neg &&
       !I->src[1].neg && !I->src[0].kill && !I->src[1].kill) {
      enum agx3_cf_fcond cond;
      bool or_nan;
      const uint16_t src[] = {I->src[0].value, I->src[1].value};
      const struct agx3_cf_source_hints hints[] = {
         {I->src[0].cache, I->src[0].discard},
         {I->src[1].cache, I->src[1].discard}};
      if (!agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert) ||
          !agx3_cf_fcmp_reg16(enc, cond, or_nan, flag_slot, src, hints, 0,
                              bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   /* An aligned 32-bit uniform against an
    * 8-bit immediate, in either operand order, kept in place by
    * agx_lower_uniform_sources(). Canonicalize the uniform to the first
    * operand (swapping the relation, as the register/immediate path below
    * does) and emit the uniform/immediate icmp loop-guard form. */
   if (!is_float && (enc == AGX_ISA_ENCODING_AGX3B ||
                     enc == AGX_ISA_ENCODING_AGX3C)) {
      const bool uniform_first = agx3_cf_uniform32(I->src[0]) &&
                                 agx3_cf_imm8(I->src[1]);
      const bool uniform_second = agx3_cf_imm8(I->src[0]) &&
                                  agx3_cf_uniform32(I->src[1]);
      if (uniform_first || uniform_second) {
         const agx_index uniform = I->src[uniform_first ? 0 : 1];
         const agx_index constant = I->src[uniform_first ? 1 : 0];
         enum agx3_cf_icond cond;
         if (!agx3_cf_map_icond(I->icond, &cond) ||
             (uniform_second && !agx3_cf_swap_icond(&cond)) ||
             !agx3_cf_icmp_uniform_imm(enc, cond, flag_slot, uniform.value,
                                       (uint8_t)constant.value, bytes, &len))
            return false;
         return agx3_append(emission, bytes, len);
      }
   }

   /* A register against an aligned 32-bit uniform, admitted by
    * agx_instr_accepts_uniform() as operand 1. */
   if (!is_float && (enc == AGX_ISA_ENCODING_AGX3B ||
                     enc == AGX_ISA_ENCODING_AGX3C) &&
       agx3_cf_reg32(I->src[0]) && agx_channels(I->src[0]) == 1 &&
       !I->src[0].kill && I->src[1].type == AGX_INDEX_UNIFORM &&
       I->src[1].size == AGX_SIZE_32 && agx_channels(I->src[1]) == 1 &&
       !I->src[1].abs && !I->src[1].neg) {
      enum agx3_cf_icond cond;
      if (!agx3_cf_map_icond(I->icond, &cond) ||
          !agx3_cf_icmp_reg_uniform32(
             enc, cond, flag_slot, I->src[0].value, I->src[1].value,
             (struct agx3_cf_source_hints){I->src[0].cache,
                                           I->src[0].discard},
             bytes, &len))
         return false;
      return agx3_append(emission, bytes, len);
   }

   bool ok;
   if (is_float) {
      if (!agx3_cf_reg32(I->src[0]))
         return false;
      enum agx3_cf_fcond cond;
      bool or_nan;
      if (!agx3_cf_reg32(I->src[1]) ||
          !agx3_cf_map_fcond(I->fcond, &cond, &or_nan, invert))
         return false;

      ok = agx3_cf_fcmp_reg(enc, cond, or_nan, flag_slot, I->src[0].value,
                            I->src[1].value, bytes, &len);
   } else {
      enum agx3_cf_icond cond;
      if (!agx3_cf_map_icond(I->icond, &cond))
         return false;

      agx_index lhs = I->src[0];
      agx_index rhs = I->src[1];
      if (lhs.type == AGX_INDEX_IMMEDIATE && !lhs.abs && !lhs.neg &&
          lhs.value <= UINT8_MAX && agx3_cf_reg32(rhs)) {
         agx_index temporary = lhs;
         lhs = rhs;
         rhs = temporary;
         if (!agx3_cf_swap_icond(&cond))
            return false;
      }

      if (!agx3_cf_reg32(lhs))
         return false;
      if (rhs.type == AGX_INDEX_IMMEDIATE && !rhs.abs && !rhs.neg &&
          rhs.value <= UINT8_MAX) {
         ok = agx3_cf_icmp_imm(enc, cond, flag_slot, lhs.value, rhs.value,
                               bytes, &len);
      } else if (agx3_cf_reg32(rhs)) {
         ok = agx3_cf_icmp_reg(enc, cond, flag_slot, lhs.value, rhs.value,
                               bytes, &len);
      } else {
         return false;
      }
   }

   return ok && agx3_append(emission, bytes, len);
}

static bool
agx3_emit_cnd(struct util_dynarray *emission, enum agx_isa_encoding enc,
              unsigned op, unsigned flag_slot, bool invert, unsigned count)
{
   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   bool ok;

   switch (op) {
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
      ok = agx3_cf_cndst(enc, flag_slot, invert, count, bytes, &len);
      break;
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
      /* cndlt IS the two-level cndcf, in a compact four-byte form. A loop
       * owns two architectural levels, so a break with nothing else open --
       * and the loop's own continue test -- lands on it; a break that also
       * has enclosing ifs open needs the wide form's larger count. */
      ok = count <= 2
              ? agx3_cf_cndlt(enc, flag_slot, invert, bytes, &len)
              : agx3_cf_cndcf(enc, flag_slot, invert, count, bytes, &len);
      break;
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
      ok = agx3_cf_cndef(enc, bytes, &len);
      break;
   case AGX_OPCODE_POP_EXEC:
      ok = agx3_cf_cndend(enc, count, bytes, &len);
      break;
   default:
      return false;
   }

   return ok && agx3_append(emission, bytes, len);
}

/*
 * Retire scoreboard slots ahead of a control-flow boundary.
 *
 * agx3_slot_assign() is linear: "waited once = satisfied globally" holds only
 * if every path into a block sees the same scoreboard, so the scoreboard must
 * be empty there. Waiting early only costs latency, so retire everything with
 * `band(r0, r0, r0)` (an identity carrying a six-bit wait mask, as in
 * agx3_emit_alu()'s prelude). r0 must be ready or its producer in the mask; a
 * partial wait for an unrelated destination needs a carrier covering it.
 */
bool
agx3_retire_slots_noop(struct util_dynarray *emission, enum agx_isa_encoding enc,
                       uint16_t mask)
{
   if (mask == 0)
      return true;

   /* Enforce the precondition above instead of trusting every caller: a
    * result still in flight to r0 joins the mask, so it lands before the
    * carrier reads and rewrites r0 (see agx3_r0_inflight). */
   if (agx3_carrier1140) {
      mask |= agx3_r0_inflight;
      agx3_r0_inflight &= (uint16_t)~mask;
   }

   /* Wait on every slot in the one instruction that reads r0. Retiring the
    * slots one instruction at a time reads and writes r0 while a later slot
    * in the mask may still be producing it, which loses that result. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   return agx3_band_carrier(enc, 0, mask, bytes, &len) &&
          agx3_append(emission, bytes, len);
}

/* Carry `mask` on an identity of `reg`, which must be ready once `mask` has
 * retired: every pending producer of it is in the mask. Only touch one half
 * for a half register: its adjacent half may still belong to an independent
 * asynchronous load. Keep the existing slot-6 carrier contract; changing that
 * protocol requires separate hardware qualification. */
bool
agx3_retire_wait_on(struct util_dynarray *emission, enum agx_isa_encoding enc,
                    agx_index reg, uint16_t mask)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   /* The immediate halfword form also reaches high registers unavailable to
    * the word bitop form. The full register wait covers this subset too. */
   const bool half = reg.size == AGX_SIZE_16 || reg.value >= 192;
   const bool ok = half
      ? agx3_alu_bit_imm(enc, AGX3_ALU_BOR, true, reg.value, reg.value, 0,
                          AGX3_ALU_HINT_NONE,
                          mask | (1u << (enc == AGX_ISA_ENCODING_AGX3C ? 7 : 5)),
                          bytes, &len)
      : agx3_alu_band_identity_wait(enc, reg.value, mask, bytes, &len);
   return ok && agx3_append(emission, bytes, len);
}

/* Carry a result-producing instruction's complete logical wait on its own
 * destination. The slot pass already includes WAW producers and outstanding
 * payload readers of that footprint, in addition to the instruction's RAW
 * dependencies. An identity preserves an in-place source without touching an
 * unrelated r0 generation. Before any part of I writes its destination, this
 * must receive the full wait, or the caller must have already retired that
 * full wait without issuing any new readers/writers of the destination. The
 * latter permits resource-only waits during operand staging, when the result
 * footprint is already ready. A partial mask alone is not sufficient.
 */
bool
agx3_retire_result_wait(struct util_dynarray *emission,
                        enum agx_isa_encoding enc, const agx_instr *I,
                        uint16_t mask)
{
   if (!mask)
      return true;
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return agx3_retire_slots_noop(emission, enc, mask);
   if (I->nr_dests != 1 || !agx3_src_is_register(I->dest[0]) ||
       I->dest[0].memory)
      return false;

   return agx3_retire_wait_on(emission, enc, I->dest[0], mask);
}

/* Allocate the actual operand footprint from the sampler-private register
 * file. Lifetimes follow the consuming operations' scoreboard slots, shared
 * with texel fetches and image writes, rather than three fixed sample groups.
 * The caller has already retired I's complete operand/result wait and has not
 * started its staging or result write, so I's result is a ready carrier.
 */
bool
agx3_allocate_staging(struct util_dynarray *emission,
                      enum agx_isa_encoding enc, const agx_instr *I,
                      struct agx3_staging_banks *banks,
                      uint16_t *outstanding, uint8_t slot,
                      unsigned span, unsigned avoid_end, uint16_t *base)
{
   uint16_t waits;
   /* Eight-half sample banks (2D-array layer + comparison, packed per-lane
    * LOD, 3D) align to eight halves so consecutive banks stay disjoint and
    * satisfy PACK_SMPIR's bank alignment on both AGX3B and AGX3C. Ordinary
    * four-half coordinates keep their native four-half alignment. */
   const unsigned alignment = span == 8 ? 8 : 4;
   if (!agx3_staging_plan_above(banks, span, alignment, AGX3_STAGING_HALFS,
                                avoid_end, base, &waits))
      return false;

   if (waits) {
      if (!agx3_retire_result_wait(emission, enc, I, waits))
         return false;
      *outstanding &= (uint16_t)~waits;
      *outstanding |= (uint16_t)(1u << (slot - 1));
      agx3_staging_forget(banks, waits);
   }
   return true;
}

/* The instructions agx3_try_control_flow() claims. */
bool
agx3_is_cf_op(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_BEGIN_CF:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
   case AGX_OPCODE_POP_EXEC:
   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      return true;
   default:
      return false;
   }
}

static bool
agx3_is_loop_push(const agx_instr *I, unsigned nest)
{
   return I->op == AGX_OPCODE_IF_FCMP && I->target == NULL && I->nest == nest &&
          I->fcond == AGX_FCOND_EQ && !I->invert_cond && I->nr_srcs == 2 &&
          agx3_cf_is_zero(I->src[0]) && agx3_cf_is_zero(I->src[1]);
}

static bool
agx3_is_loop_entry_push(const agx_instr *I)
{
   return agx3_is_loop_push(I, 1);
}

static bool
agx3_is_loop_guard_push(const agx_instr *I)
{
   return agx3_is_loop_push(I, 0);
}

static int
agx3_try_control_flow(struct util_dynarray *emission,
                      struct util_dynarray *fixups, enum agx_isa_encoding enc,
                      enum agx_chip chip,
                      const agx_instr *I, uint16_t wait_mask,
                      uint16_t outstanding, bool after_loop_backedge,
                      bool loop_exit_jump, unsigned loop_depth,
                      off_t *compare_end, bool g18g)
{
   const bool is_if =
      I->op == AGX_OPCODE_IF_ICMP || I->op == AGX_OPCODE_IF_FCMP;
   const bool is_while =
      I->op == AGX_OPCODE_WHILE_ICMP || I->op == AGX_OPCODE_WHILE_FCMP;
   const bool is_break_if =
      I->op == AGX_OPCODE_BREAK_IF_ICMP || I->op == AGX_OPCODE_BREAK_IF_FCMP;
   const bool is_else =
      I->op == AGX_OPCODE_ELSE_ICMP || I->op == AGX_OPCODE_ELSE_FCMP;
   const bool is_jump = I->op == AGX_OPCODE_JMP_EXEC_ANY ||
                        I->op == AGX_OPCODE_JMP_EXEC_NONE ||
                        I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER;

   /*
    * Slot assignment is linear, so a control-flow boundary is safe only when
    * nothing is in flight: then every block is entered with an empty
    * scoreboard whichever predecessor ran. A result still outstanding here
    * would carry its slot across the branch (or a backedge), which needs a
    * predecessor-aware analysis this backend does not have, so it is refused.
    */
   const bool live_begin = enc == AGX_ISA_ENCODING_AGX3C &&
      I->op == AGX_OPCODE_BEGIN_CF;
   if (outstanding != 0 &&
       ((!live_begin && I->op == AGX_OPCODE_BEGIN_CF) || is_if || is_while || is_break_if ||
        is_else || I->op == AGX_OPCODE_POP_EXEC || is_jump))
      return -1;

   if (I->op == AGX_OPCODE_BEGIN_CF)
      return wait_mask == 0 ? 1 : -1; /* architectural state: zero bytes */

   if (agx3_is_loop_guard_push(I)) {
      /* A top-level loop first opens the ordinary outer control-flow scope.
       * Slot 6 is Apple's always-true predicate, so no compare is needed. */
      return wait_mask == 0 && agx3_emit_cnd(emission, enc, I->op, 6, false, 1)
                ? 1
                : -1;
   }

   if (agx3_is_loop_entry_push(I)) {
      /* agx_push_exec(1), used before every AGX3 loop, becomes Apple's
       * unconditional TRUE-slot loop entry and opens two architectural
       * exec-mask levels. No compare is emitted. */
      return wait_mask == 0 && agx3_emit_cnd(emission, enc, I->op, 6, false, 2)
                ? 1
                : -1;
   }

   if (is_if || is_while || is_break_if) {
      /* The all-clear compare/cnd forms are witnessed only for register-ready
       * inputs.  A local dependency wait also keeps memory-dependent control
       * flow fail-closed. */
      if (wait_mask != 0)
         return -1;
      /* A loop owns two exec-mask levels (Mesa's nest=1 break and the
       * loop's continue test, CNDLT; the backedge footer pops them with
       * CNDEND(2)), and every enclosing `if` the break also escapes adds
       * one. Apple's CNDCF count is the number of levels open inside the
       * innermost loop, i.e. nest + 1. */
      if (is_if && I->nest != 1)
         return -1;
      if (is_break_if && I->nest == 0)
         return -1;

      bool invert = I->invert_cond;
      const bool is_float = I->op == AGX_OPCODE_IF_FCMP ||
                            I->op == AGX_OPCODE_WHILE_FCMP ||
                            I->op == AGX_OPCODE_BREAK_IF_FCMP;
      /* G16G reuses each loop's slot for its breaks and wraps deep loop nests;
       * G17P assigns a deep break the next slot so it cannot overwrite the
       * live predicate of an enclosing conditional. NEST counts exec-mask
       * levels exited, not loop depth. A flag slot is an ordinary resource
       * (Apple reuses slot 1 for several loops' exit predicates), and every
       * compare here is immediately followed by the CND consuming it, so the
       * six slots wrap modulo six for nests deeper than six. */
      if ((is_break_if || is_while) && loop_depth == 0)
         return -1;
      const unsigned loop_slot = loop_depth - 1;
      const unsigned flag_slot = chip == AGX_CHIP_G16G
         ? (is_break_if || is_while ? loop_slot % 6 : 0)
         : (is_break_if ? (loop_slot + (I->nest > 1)) % 6
                        : (is_while ? loop_slot % 6 : 0));
      if (flag_slot >= 6)
         return -1;
      if (!agx3_emit_cf_compare(emission, enc, I, is_float, flag_slot,
                                &invert, g18g)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr, "AGX3 control-flow compare refused\n");
         return -1;
      }
      if (compare_end)
         *compare_end = emission->size;
      /* CNDLT retains lanes for which its predicate is true, so a shallow
       * BREAK_IF needs the complementary continue predicate. CNDCF instead
       * consumes the break predicate directly. Native break_continue uses
       * EQ(i, 9) followed by CNDCF(invert=0); deep_break uses EQ(guard, 0)
       * followed by CNDCF(invert=1). Flipping a deep unconditional break's
       * true predicate makes it false and leaves the loop running forever. */
      if (is_break_if && I->nest == 1)
         invert = !invert;
      const unsigned levels = is_break_if ? I->nest + 1 : (is_if ? 1 : 2);
      if (!agx3_emit_cnd(emission, enc, I->op, flag_slot, invert, levels)) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr, "AGX3 control-flow mask transition refused\n");
         return -1;
      }
      return 1;
   }

   if (is_else) {
      /* The only witnessed else is the canonical unconditional
       * else_fcmp(0, 0, eq, nest=1) emitted by Mesa's structurizer. */
      if (wait_mask != 0 || I->op != AGX_OPCODE_ELSE_FCMP || I->nest != 1 ||
          I->fcond != AGX_FCOND_EQ || I->invert_cond || I->nr_srcs != 2 ||
          !agx3_cf_is_zero(I->src[0]) || !agx3_cf_is_zero(I->src[1]))
         return -1;
      return agx3_emit_cnd(emission, enc, I->op, 0, false, 1) ? 1 : -1;
   }

   if (I->op == AGX_OPCODE_POP_EXEC) {
      /* Mesa's loop footer is jmp_exec_any + pop_exec(1), while AGX3's loop
       * entry/test opens two architectural levels and therefore pops two. */
      const unsigned count = after_loop_backedge ? 2 : I->nest;
      return wait_mask == 0 &&
                   agx3_emit_cnd(emission, enc, I->op, 0, false, count)
                ? 1
                : -1;
   }

   if (is_jump) {
      if (wait_mask != 0 || I->target == NULL)
         return -1;

      const bool is_bn = I->op != AGX_OPCODE_JMP_EXEC_ANY;
      struct agx3_branch_fixup fixup = {
         .block = I->target,
         .offset = emission->size,
         .skip_to_end = I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER,
         .loop_exit = loop_exit_jump,
         .is_bn = is_bn,
      };
      util_dynarray_append(fixups, fixup);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      return agx3_cf_branch(enc, is_bn, bytes, &len) &&
                   agx3_append(emission, bytes, len)
                ? 1
                : -1;
   }

   /* agx_lower_pseudo rewrites every AGX3 unconditional break into the
    * always-true BREAK_IF above, at any nesting depth, so one reaching here
    * means that pass did not run. Keep the guard rather than emitting an
    * exec-mask transition with no compare in front of it. */
   if (I->op == AGX_OPCODE_BREAK)
      return -1;

   return 0;
}

bool
agx3_fixup_branch(struct util_dynarray *emission, enum agx_isa_encoding enc,
                  struct agx3_branch_fixup fixup)
{
   /* JMP_EXEC_NONE_AFTER normally targets the last instruction in a block.
    * A shallow loop break is different: its footer may contain CNDEND(2)
    * followed by CNDEND(1) for an enclosing scope.  The branch has to execute
    * both.  Skipping the loop pop leaves every lane inactive at END and the
    * GPU walks off the executable allocation. */
   if (fixup.loop_exit && fixup.block->loop_exit_offset < 0)
      return false;

   const off_t target =
      fixup.loop_exit
         ? fixup.block->loop_exit_offset
         : fixup.skip_to_end ? fixup.block->last_offset
                             : fixup.block->offset;
   const int64_t displacement = (int64_t)target - (int64_t)fixup.offset;

   /* Only Apple's witnessed directions are accepted: bn skips forward and b
    * is a backwards loop edge. This also rejects zero-displacement branches. */
   if ((fixup.is_bn && displacement <= 0) ||
       (!fixup.is_bn && displacement >= 0))
      return false;
   if (fixup.offset < 0 ||
       (uint64_t)fixup.offset + AGX3_CF_MAX_LENGTH > emission->size)
      return false;

   uint8_t *branch = (uint8_t *)emission->data + fixup.offset;
   return agx3_cf_patch_branch(enc, fixup.is_bn, branch, displacement);
}

/*
 * Apple's complete G17P float2-position shader uses a protocol that is not
 * selected by the generic round-robin scoreboard pass: its sole attribute
 * LDDEV carries f79=2 and signals slot 6. Keep its index last-use exception
 * fenced to the witnessed post-RA shape. All UVSWs, including this shader's,
 * require explicit staging producers; copying their one-hot wait masks alone
 * does not produce the data they consume.
 */
bool
agx3_is_g17p_float2_position_vs(agx_context *ctx,
                                enum agx_isa_encoding enc)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || ctx->stage != MESA_SHADER_VERTEX)
      return false;

   unsigned instructions = 0, vertex_ids = 0, loads = 0, stores = 0;
   uint8_t uvs_mask = 0;

   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         instructions++;

         switch (I->op) {
         case AGX_OPCODE_GET_SR:
            if (I->sr != AGX_SR_VERTEX_ID)
               return false;
            vertex_ids++;
            break;
         case AGX_OPCODE_DEVICE_LOAD:
            if (I->nr_dests != 1 || I->nr_srcs != 2 || I->mask != 3 ||
                I->format != AGX_FORMAT_I32 ||
                I->dest[0].type != AGX_INDEX_REGISTER ||
                I->dest[0].size != AGX_SIZE_32 ||
                agx_channels(I->dest[0]) != 2 ||
                I->src[0].type != AGX_INDEX_UNIFORM ||
                I->src[0].size != AGX_SIZE_64 || I->src[0].value != 0 ||
                I->src[1].type != AGX_INDEX_REGISTER ||
                I->src[1].size != AGX_SIZE_32)
               return false;
            loads++;
            break;
         case AGX_OPCODE_ST_VARY:
            if (I->nr_dests != 0 || I->nr_srcs != 2 ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value > 3 ||
                (uvs_mask & (1u << I->src[0].value)))
               return false;
            uvs_mask |= 1u << I->src[0].value;
            stores++;
            break;
         case AGX_OPCODE_MOV_IMM:
            if (I->imm != 0 && I->imm != 0x3f800000)
               return false;
            break;
         default:
            return false;
         }
      }
   }

   return instructions == 8 && vertex_ids == 1 && loads == 1 && stores == 4 &&
          uvs_mask == 0xf;
}

/* Loop-head hint.
 *
 * Every integer, conversion and control form of AGX3b carries three 2-bit
 * fields at instruction bits 18..23, each passed through the encoder LUT
 * [1, 2, 0] (isa/AGX3b.xml, e.g. sll f68/f69/f70 at 22:23/20:21/18:19, iadd
 * f55/f56/f57 at the same bits), so a field value of 0 encodes as 0b01.
 * Apple's G17P compiler sets the first two of them to 2 (bits 20..23 = 0)
 * only on the instruction at a loop's backward-branch target, a 12-byte
 * `sll` in both the fragment and the compute loop; no other witnessed Apple
 * instruction has a non-zero value there. Mesa always emitted 0. Semantics
 * are unknown; this reproduces only Apple's exact use: an allowlisted
 * integer/conversion ALU form (opcode bits 0..10 and length from AGX3b.xml)
 * that begins a loop header, whose bits 18..23 still hold the default
 * 0b010101. */
static const struct {
   uint16_t opcode;
   uint8_t length;
} agx3_loop_head_hint_forms[] = {
   {0x01f, 12}, /* imsub */   {0x027, 12}, /* sll */
   {0x09f, 12}, /* imadd */   {0x0a7, 12}, /* srl */
   {0x11f, 10}, /* isub */    {0x127, 12}, /* extract */
   {0x19f, 10}, /* iadd */    {0x1a7, 10}, /* sra */
   {0x227, 12}, /* sllu */    {0x2a7, 12}, /* srlu */
   {0x3a7, 10}, /* srau */    {0x417, 8},  /* normflt */
   {0x497, 10}, /* fltnorm */ {0x727, 10}, /* flt2int */
   {0x7a7, 8},  /* int2flt */
};

unsigned
agx3_apply_loop_head_hints(agx_context *ctx, struct util_dynarray *emission)
{
   unsigned applied = 0;
   agx_foreach_block(ctx, block) {
      if (!block->loop_header || block->offset < 0 ||
          (size_t)block->offset + 4 > emission->size)
         continue;
      uint8_t *bytes = (uint8_t *)emission->data + block->offset;
      const uint16_t opcode = (uint16_t)(bytes[0] | ((bytes[1] & 0x7) << 8));
      /* Only Apple's witnessed form, the 12-byte `sll` at a loop head. */
      if (opcode != 0x027)
         continue;
      bool ok = false;
      for (unsigned i = 0; i < ARRAY_SIZE(agx3_loop_head_hint_forms); ++i) {
         if (agx3_loop_head_hint_forms[i].opcode == opcode &&
             (size_t)block->offset + agx3_loop_head_hint_forms[i].length <=
                emission->size) {
            ok = true;
            break;
         }
      }
      /* Bits 18..23 live in byte 2 bits 2..7: the default is 0b010101. */
      if (!ok || (bytes[2] & 0xfc) != 0x54)
         continue;
      bytes[2] &= (uint8_t)~0x50; /* bits 20 and 22: f(n), f(n+1) = 2 */
      applied++;
   }
   return applied;
}

/* Early compares.
 *
 * Apple's G17P loops test the counter as soon as it is final: in a
 * fragment loop the `icmp` sits right after `i++` and its `cndlt` comes
 * six instructions later at the bottom of the body. Our emission
 * always writes a control-flow compare immediately before the CND that
 * consumes it. This moves an integer loop/if compare back to just after the
 * last definition of its register sources, leaving the CND in place.
 *
 * The compare may only cross a straight run of emitted bytes: every crossed
 * instruction is a plain synchronous ALU instruction (no scoreboard slot or
 * wait, no control flow, so the exec mask and every flag slot are unchanged
 * across the run), no branch lands inside the run, and no recorded branch
 * instruction lies in it. The run may span blocks whose own bytes are empty
 * (the structurizer's empty then/else blocks between a rotated loop body and
 * its latch). A compare source marked for discard must not be read by a
 * crossed instruction. The compare bytes are rotated, not re-encoded. */
static bool
agx3_early_compare_crossable(const agx_instr *I)
{
   if (I->agx3_tex_coord_stage)
      return false;
   switch (I->op) {
   case AGX_OPCODE_MOV_IMM:
   case AGX_OPCODE_IADD:
   case AGX_OPCODE_IMAD:
   case AGX_OPCODE_BFI:
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_EXTR:
   case AGX_OPCODE_ASR:
   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_BITOP:
   case AGX_OPCODE_INTL:
   case AGX_OPCODE_FADD:
   case AGX_OPCODE_FMUL:
   case AGX_OPCODE_FFMA:
      return true;
   default:
      return false;
   }
}

static bool
agx3_instr_touches_regs(const agx_instr *I, bool dests, unsigned reg,
                        unsigned count)
{
   if (dests) {
      agx_foreach_reg_dest(I, d) {
         const unsigned r = I->dest[d].value;
         const unsigned n = agx_index_size_16(I->dest[d]);
         if (r < reg + count && reg < r + n)
            return true;
      }
   } else {
      agx_foreach_reg_src(I, s) {
         const unsigned r = I->src[s].value;
         const unsigned n = agx_index_size_16(I->src[s]);
         if (r < reg + count && reg < r + n)
            return true;
      }
   }
   return false;
}

off_t
agx3_fixup_target(const struct agx3_branch_fixup *f)
{
   return f->loop_exit ? f->block->loop_exit_offset
          : f->skip_to_end ? f->block->last_offset
                           : f->block->offset;
}

unsigned
agx3_apply_early_compares(agx_context *ctx, struct util_dynarray *emission,
                          size_t initial_size,
                          const struct util_dynarray *spans,
                          const struct util_dynarray *compares,
                          const struct util_dynarray *fixups)
{
   const struct agx3_instr_span *sp = spans->data;
   unsigned moved = 0;

   /* AGX_G17P_ENTRY_COMPARE only lets a register-free compare cross waiting
    * instructions when nothing in the shader writes the uniform file, so no
    * outstanding uniform write can be what a crossed wait orders. */
   bool writes_uniforms = false;
   agx_foreach_instr_global(ctx, I) {
      writes_uniforms |= I->op == AGX_OPCODE_UNIFORM_STORE ||
                         I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                         I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                         I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3;
   }

   util_dynarray_foreach(compares, struct agx3_early_compare, c) {
      const agx_instr *T = c->I;
      if ((T->op != AGX_OPCODE_IF_ICMP && T->op != AGX_OPCODE_WHILE_ICMP &&
           T->op != AGX_OPCODE_BREAK_IF_ICMP) ||
          c->compare_start != c->instr_start || c->span_count == 0 ||
          c->compare_end - c->compare_start > AGX3_CF_MAX_LENGTH)
         continue;
      const unsigned t = c->span_count - 1;
      if (sp[t].I != T || sp[t].start != c->instr_start)
         continue;

      /* Walk back over crossable instructions to the last source writer. */
      unsigned from = t;
      while (from > 0) {
         const struct agx3_instr_span *x = &sp[from - 1];
         /* BEGIN_CF emits no bytes on
          * AGX3 (agx3_try_control_flow) and changes neither the exec mask nor
          * any flag slot, so an empty BEGIN_CF span is crossable. Without
          * this the first compare of a shader (e.g. a loop guard) is pinned
          * against its CNDST; Apple hoists it (the guard icmp sits four
          * instructions ahead of its cndst). */
         if (x->I->op == AGX_OPCODE_BEGIN_CF &&
             !x->wait && !x->slot && x->start == sp[from].start) {
            from--;
            continue;
         }
         bool writes = false;
         agx_foreach_reg_src(T, s) {
            writes |= agx3_instr_touches_regs(x->I, true, T->src[s].value,
                                              agx_index_size_16(T->src[s]));
         }
         /* A compare with no register source (uniform against immediate)
          * cannot depend on a crossed instruction's scoreboard wait: that
          * wait orders the crossed instruction's own register reads only.
          * Such a compare may cross a waiting
          * synchronous ALU instruction; it still never crosses an
          * instruction that starts an asynchronous (slotted) operation. */
         bool reg_free = true;
         agx_foreach_reg_src(T, s)
            reg_free = false;
         const bool wait_ok = !x->wait || (reg_free && !writes_uniforms);
         if (writes || !wait_ok || x->slot ||
             !agx3_early_compare_crossable(x->I))
            break;
         bool read_discarded = false;
         agx_foreach_reg_src(T, s) {
            read_discarded |= T->src[s].discard &&
               agx3_instr_touches_regs(x->I, false, T->src[s].value,
                                       agx_index_size_16(T->src[s]));
         }
         if (read_discarded)
            break;
         from--;
      }
      if (from == t)
         continue;

      /* Contiguity: spans [from, t) own exactly [lo, compare_start). */
      const off_t lo = sp[from].start;
      bool ok = lo < c->compare_start;
      util_dynarray_foreach(fixups, struct agx3_branch_fixup, f) {
         const off_t target = agx3_fixup_target(f);
         if ((f->offset >= lo && f->offset < c->compare_end) ||
             (target > lo && target <= c->compare_start))
            ok = false;
      }
      /* Offsets the fragment linker records into this part must not move. */
      const off_t recorded[] = {
         ctx->agx3_colour_link.entry_emitted
            ? (off_t)(initial_size + ctx->agx3_colour_link.entry_offset) : -1,
      };
      for (unsigned r = 0; r < ARRAY_SIZE(recorded); ++r)
         ok &= !(recorded[r] >= lo && recorded[r] < c->compare_end);
      /* So must the UVSWs it relocates (agx3_note_uvs_reloc()). */
      for (unsigned r = 0; r < ctx->agx3_uvs_reloc_count; ++r) {
         uint32_t at;
         memcpy(&at,
                (const uint8_t *)ctx->agx3_uvs_relocs.data +
                   r * AGX3_UVS_RELOC_SIZE +
                   offsetof(struct agx3_uvs_reloc, offset),
                sizeof(at));
         ok &= !((off_t)at >= lo && (off_t)at < c->compare_end);
      }
      if (!ok)
         continue;

      uint8_t *base = emission->data;
      const size_t clen = (size_t)(c->compare_end - c->compare_start);
      const size_t blen = (size_t)(c->compare_start - lo);
      uint8_t compare[AGX3_CF_MAX_LENGTH];
      memcpy(compare, base + c->compare_start, clen);
      memmove(base + lo + clen, base + lo, blen);
      memcpy(base + lo, compare, clen);

      /* Blocks that began inside the moved run now begin at the CND. */
      agx_foreach_block(ctx, block) {
         if (block->offset > lo && block->offset <= c->compare_start)
            block->offset = c->compare_end;
         if (block->last_offset > lo && block->last_offset <= c->compare_start)
            block->last_offset = c->compare_end;
      }
      moved++;
   }
   return moved;
}

enum agx3_pack_result
agx3_pack_cf_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   uint16_t cf_wait = st->wait;
   uint16_t cf_outstanding = st->outstanding_here;
   const bool live_begin = enc == AGX_ISA_ENCODING_AGX3C &&
      I->op == AGX_OPCODE_BEGIN_CF;
   if (agx3_is_cf_op(I) && !live_begin) {
      const uint16_t retire = (uint16_t)(cf_wait | cf_outstanding);
      if (retire != 0) {
         if (!agx3_retire_slots_noop(emission, enc, retire))
            AGX3_REFUSE_RET(I);
         cf_wait = 0;
         cf_outstanding = 0;
         st->outstanding = 0;
      }
   }

   if (st->m5_sidefx_entry && !st->g17p_colour_open &&
       (I->op == AGX_OPCODE_IF_ICMP || I->op == AGX_OPCODE_IF_FCMP ||
        I->op == AGX_OPCODE_ATOMIC || I->op == AGX_OPCODE_DEVICE_STORE ||
        I->op == AGX_OPCODE_IMAGE_WRITE)) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         AGX3_REFUSE_RET(I);
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - st->initial_size;
      st->g17p_colour_open = true;
   }
   if (I == st->m5_coverage_entry_before && !st->g17p_colour_open) {
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH], entry_len;
      if (!agx3_pixel_colour_entry(enc, entry, &entry_len) ||
          !agx3_append(emission, entry, entry_len))
         AGX3_REFUSE_RET(I);
      ctx->agx3_colour_link.entry_emitted = true;
      ctx->agx3_colour_link.entry_offset = emission->size - entry_len - st->initial_size;
      st->g17p_colour_open = true;
   }

   const bool loop_entry_push = agx3_is_loop_entry_push(I);
   const bool loop_pop =
      I->op == AGX_OPCODE_POP_EXEC && st->after_loop_backedge;
   if (loop_pop)
      st->block->loop_exit_offset = emission->size;
   if (st->fragment_ray_query && !st->ray_query_acquired && agx3_is_cf_op(I)) {
      uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len;
      const bool ok = st->g17p_colourless_fragment
         ? agx3_pixel_atomic_acquire(enc, pixel, &pixel_len)
         : agx3_pixel_wait(enc, 4, 0, pixel, &pixel_len);
      if (!ok || !agx3_append(emission, pixel, pixel_len)) AGX3_REFUSE_RET(I);
      st->ray_query_acquired = true;
      ctx->agx3_colour_link.rt_query_write_acquired = !st->g17p_colourless_fragment;
   }
   const off_t cf_start = emission->size;
   off_t cf_compare_end = -1;
   const int cf = agx3_try_control_flow(
      emission, st->fixups, enc, ctx->key->dev.chip, I, cf_wait,
      cf_outstanding,
      st->after_loop_backedge,
      I->op == AGX_OPCODE_JMP_EXEC_NONE_AFTER && st->after_loop_exit_test,
      st->loop_depth, &cf_compare_end, agx_is_g18g(&ctx->key->dev));
   if (st->early_compare && cf > 0 && cf_compare_end > cf_start) {
      struct agx3_early_compare c = {
         .I = I, .block = st->block, .instr_start = st->span_start,
         .compare_start = cf_start, .compare_end = cf_compare_end,
         .span_count = util_dynarray_num_elements(
            st->spans, struct agx3_instr_span),
      };
      util_dynarray_append(st->compares, c);
   }
   if (cf < 0)
      AGX3_REFUSE_RET(I);
   if (cf > 0) {
      if (loop_entry_push)
         st->loop_depth++;
      else if (loop_pop && st->loop_depth > 0) {
         st->loop_depth--;
      }
      st->after_loop_backedge = I->op == AGX_OPCODE_JMP_EXEC_ANY;
      st->after_loop_exit_test =
         (I->op == AGX_OPCODE_BREAK_IF_ICMP ||
          I->op == AGX_OPCODE_BREAK_IF_FCMP) &&
         I->nest == 1;
      return AGX3_PACK_EMITTED;
   }
   st->after_loop_backedge = false;
   st->after_loop_exit_test = false;

   return AGX3_PACK_UNHANDLED;
}
