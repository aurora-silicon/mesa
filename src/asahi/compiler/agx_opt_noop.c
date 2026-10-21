/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/lut.h"
#include "asahi/isa/agx_minifloat.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/*
 * Post-RA no-op ALU cleanup.
 *
 * Instruction selection emits copy idioms for operand form constraints, e.g.
 * the G17P iadd(A=0, B=reg) form that materialises a shift boundary register,
 * or a mov that a later pass rewrites around it. When the register allocator
 * coalesces such a copy's destination onto its source the instruction is
 * left computing x -> x: a pure no-op that still costs an ALU slot.
 *
 * DCE cannot see this (the destination is live out through itself), so run a
 * dedicated sweep after pseudo lowering and before register-cache annotation
 * (self-copies neither extend nor shorten the destination's liveness, so the
 * hint pass sees the same program with or without them).
 */

/* Whether two indices name the same physical register. Liveness hint bits
 * (kill/cache/discard) do not affect the location, but float modifiers do
 * change the value read. */
static bool
same_register(agx_index a, agx_index b)
{
   return a.type == AGX_INDEX_REGISTER && b.type == AGX_INDEX_REGISTER &&
          a.value == b.value && a.size == b.size && a.memory == b.memory &&
          !a.abs && !a.neg && !b.abs && !b.neg;
}

static bool
is_zero_imm(agx_index idx)
{
   return idx.type == AGX_INDEX_IMMEDIATE && idx.value == 0 && !idx.abs &&
          !idx.neg;
}

static bool
is_imm_value(agx_index idx, uint32_t v)
{
   return idx.type == AGX_INDEX_IMMEDIATE && idx.value == v && !idx.abs &&
          !idx.neg;
}

/* For a BITOP whose second source is a constant, whether the truth table
 * reduces to a copy of the first source for that constant. LUT2 bit i is the
 * output for (a, b) = (i & 1, i >> 1): copying a needs lut[0]=0, lut[1]=1 when
 * b == 0 and lut[2]=0, lut[3]=1 when b == 1. */
static bool
lut_is_copy_a(uint32_t lut, bool b_is_one)
{
   const unsigned pair = b_is_one ? (lut >> 2) & 3 : lut & 3;
   return pair == 2;
}

static bool
is_const_src(agx_index idx, uint32_t *bits)
{
   if (idx.type != AGX_INDEX_IMMEDIATE || idx.abs || idx.neg)
      return false;

   /* The immediates file stores the raw value; all-ones is per size. */
   const uint32_t mask = idx.size == AGX_SIZE_32 ? 0xffffffff : 0xffff;
   if (idx.value == 0)
      *bits = 0;
   else if (idx.value == mask)
      *bits = 1;
   else
      return false;
   return true;
}

static bool
agx_instr_is_noop(const agx_instr *I)
{
   if (I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_REGISTER)
      return false;

   const agx_index dst = I->dest[0];

   switch (I->op) {
   case AGX_OPCODE_IADD:
      /* dst = a + (b << shift): the copy idioms iadd(0, x) and iadd(x, 0). */
      if (I->shift != 0 || I->saturate)
         return false;
      if (is_zero_imm(I->src[0]))
         return same_register(dst, I->src[1]);
      if (is_zero_imm(I->src[1]))
         return same_register(dst, I->src[0]);
      return false;

   case AGX_OPCODE_IMAD:
      /* dst = a * b + (c << shift). */
      if (I->saturate)
         return false;
      if (is_zero_imm(I->src[2])) {
         if (is_imm_value(I->src[1], 1))
            return same_register(dst, I->src[0]);
         if (is_imm_value(I->src[0], 1))
            return same_register(dst, I->src[1]);
      }
      if (I->shift == 0 &&
          (is_zero_imm(I->src[0]) || is_zero_imm(I->src[1])))
         return same_register(dst, I->src[2]);
      return false;

   case AGX_OPCODE_MOV:
      /* Pre-lower_pseudo form of the bitwise copy. */
      return same_register(dst, I->src[0]);

   case AGX_OPCODE_BITOP: {
      if (I->nr_srcs != 2)
         return false;

      if (I->truth_table == UTIL_LUT2(a))
         return same_register(dst, I->src[0]);
      if (I->truth_table == UTIL_LUT2(b))
         return same_register(dst, I->src[1]);

      uint32_t bits;
      if (is_const_src(I->src[1], &bits) && lut_is_copy_a(I->truth_table, bits))
         return same_register(dst, I->src[0]);
      return false;
   }

   case AGX_OPCODE_FADD:
      /* fmov is fadd(x, -0.0): minifloat 0x80. */
      if (I->saturate)
         return false;
      return I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0x80 &&
             !I->src[1].abs && !I->src[1].neg &&
             same_register(dst, I->src[0]);

   default:
      return false;
   }
}

void
agx_opt_noop_postra(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      if (agx_opcodes_info[I->op].can_eliminate && agx_instr_is_noop(I))
         agx_remove_instruction(I);
   }
}
