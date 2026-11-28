/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/isa/disasm.h"
#include "compiler/nir/nir_builder.h"
#include "util/lut.h"
#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
}

/* agx_nir_shift_amount_zero_or_16 recognizes dynamic 32-bit shift amounts
 * proven to be 0 or 16, and agx_opt_select_shifted_half folds a tagged 32-bit
 * right shift plus low-half register copy(s) into a single 16-bit icmpsel once
 * register assignment is known. Because the rewrite only fires post-RA when it
 * eliminates both the shift and a copy, no shader grows on any encoding.
 */

namespace {

enum amount {
   AND_16,       /* l & 16 */
   AND_16_31,    /* (l & 16) & 31, as the late shift-count mask makes it */
   BIT_SHL,      /* ((l >> 3) & 1) << 4 */
   BFE_SHL,      /* ubfe(l, 3, 1) << 4 */
   B2I_SHL,      /* b2i32(l < 0) << 4 */
   MADSHL_ZERO,  /* 0 * l + (((l >> 3) & 1) << 4) */
   SHL_SHL,      /* ((l & 1) << 2) << 2 */
   SWIZZLE,      /* (vec2(l, l >> 1) & 16).y */
   FIRST_REJECT, /* everything below is not provably 0 or 16 */
   AND_0X30 = FIRST_REJECT, /* l & 0x30: 0, 16, 32 or 48 */
   MADSHL_PRODUCT,          /* l * l + (((l >> 3) & 1) << 4) */
   BFE2_SHL,                /* ubfe(l, 3, 2) << 4: 0, 16, 32 or 48 */
   SHL_TOO_FAR,             /* ((l >> 3) & 1) << 5 */
   VARIABLE,                /* l */
   CONSTANT,                /* 16, left to constant folding */
   DEEP,                    /* l & 16 under eight masks of 0xffff */
   NUM_AMOUNTS,
};

static nir_def *
build_amount(nir_builder *b, enum amount a, nir_def *l)
{
   nir_def *bit3 = nir_iand_imm(b, nir_ushr_imm(b, l, 3), 1);
   switch (a) {
   case AND_16:
      return nir_iand_imm(b, l, 16);
   case AND_16_31:
      return nir_iand_imm(b, nir_iand_imm(b, l, 16), 31);
   case BIT_SHL:
      return nir_ishl_imm(b, bit3, 4);
   case BFE_SHL:
      return nir_ishl_imm(
         b, nir_ubitfield_extract(b, l, nir_imm_int(b, 3), nir_imm_int(b, 1)),
         4);
   case B2I_SHL:
      return nir_ishl_imm(b, nir_b2i32(b, nir_ilt_imm(b, l, 0)), 4);
   case MADSHL_ZERO:
      return nir_imadshl_agx(b, nir_imm_int(b, 0), l, bit3, nir_imm_int(b, 4));
   case SHL_SHL:
      return nir_ishl_imm(b, nir_ishl_imm(b, nir_iand_imm(b, l, 1), 2), 2);
   case SWIZZLE: {
      nir_def *v = nir_iand_imm(b, nir_vec2(b, l, nir_ushr_imm(b, l, 1)), 16);
      return nir_channel(b, v, 1);
   }
   case AND_0X30:
      return nir_iand_imm(b, l, 0x30);
   case MADSHL_PRODUCT:
      return nir_imadshl_agx(b, l, l, bit3, nir_imm_int(b, 4));
   case BFE2_SHL:
      return nir_ishl_imm(
         b, nir_ubitfield_extract(b, l, nir_imm_int(b, 3), nir_imm_int(b, 2)),
         4);
   case SHL_TOO_FAR:
      return nir_ishl_imm(b, bit3, 5);
   case VARIABLE:
      return l;
   case CONSTANT:
      return nir_imm_int(b, 16);
   case DEEP: {
      nir_def *v = nir_iand_imm(b, l, 16);
      for (unsigned i = 0; i < 8; ++i)
         v = nir_iand_imm(b, v, 0xffff);
      return v;
   }
   default:
      return NULL;
   }
}

static uint32_t
eval_amount(enum amount a, uint32_t l)
{
   const uint32_t bit3 = (l >> 3) & 1;
   switch (a) {
   case AND_16:
   case AND_16_31:
   case DEEP:
      return l & 16;
   case BIT_SHL:
   case BFE_SHL:
   case MADSHL_ZERO:
      return bit3 << 4;
   case B2I_SHL:
      return ((int32_t)l < 0) << 4;
   case SHL_SHL:
      return (l & 1) << 4;
   case SWIZZLE:
      return (l >> 1) & 16;
   case AND_0X30:
      return l & 0x30;
   case MADSHL_PRODUCT:
      return l * l + (bit3 << 4);
   case BFE2_SHL:
      return ((l >> 3) & 3) << 4;
   case SHL_TOO_FAR:
      return bit3 << 5;
   case VARIABLE:
      return l;
   case CONSTANT:
      return 16;
   default:
      return 0;
   }
}

TEST(SelectShiftedHalf, RecognizesOnlyAmountsOfZeroOrSixteen)
{
   const uint32_t ws[] = {0x00000000u, 0x0000ffffu, 0xffff0000u,
                          0x80007fffu, 0x7fff8000u, 0xdeadbeefu};
   const uint32_t ls[] = {0u,  1u,  8u,          9u,          16u,
                          24u, 48u, 0x80000018u, 0xffffffffu, 0x7ffffff7u};
   for (unsigned a = 0; a < NUM_AMOUNTS; ++a) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "amt");
      nir_def *l = agx_test_load_preamble(&b, 1, 32, 2);
      nir_def *s = build_amount(&b, (enum amount)a, l);
      const bool recognized = agx_nir_shift_amount_zero_or_16(s, 0);
      EXPECT_EQ(recognized, a < FIRST_REJECT) << "amount " << a;
      ralloc_free(b.shader);

      for (bool arithmetic : {false, true}) {
         for (uint32_t w : ws) {
            for (uint32_t lval : ls) {
               const uint32_t sval = eval_amount((enum amount)a, lval);
               if (recognized) {
                  EXPECT_TRUE(sval == 0 || sval == 16)
                     << "amount " << a << " l 0x" << std::hex << lval;
                  const uint16_t sel =
                     sval == 0 ? (uint16_t)(w & 0xffffu) : (uint16_t)(w >> 16);
                  const uint32_t shifted =
                     arithmetic ? (uint32_t)((int32_t)w >> (sval & 31))
                                : (w >> (sval & 31));
                  EXPECT_EQ(sel, (uint16_t)shifted)
                     << "amount " << a << " arithmetic " << arithmetic
                     << std::hex << " w 0x" << w << " l 0x" << lval;
               }
            }
         }
      }
   }
}

static unsigned
count_opcode(const agx_context *ctx, enum agx_opcode op)
{
   unsigned n = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == op)
         ++n;
   }
   return n;
}

static agx_builder *
make_postra_builder(void *memctx, agx_isa_encoding enc)
{
   agx_builder *b = agx_test_builder(memctx);
   agx_shader_key *key = rzalloc(memctx, agx_shader_key);
   key->dev.encoding = enc;
   b->shader->key = key;
   return b;
}

static agx_instr *
emit_copy16(agx_builder *b, unsigned dst_h, unsigned src_h)
{
   return agx_bitop_to(b, agx_register(dst_h, AGX_SIZE_16),
                       agx_register(src_h, AGX_SIZE_16), agx_zero(),
                       UTIL_LUT2(a));
}

TEST(SelectShiftedHalf, PostRaPeepholeCases)
{
   void *memctx = ralloc_context(NULL);

   /* 1. Basic BFEIL + low-half copy folds to 1 icmpsel (Placement A). */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BITOP), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 1u);
   }

   /* 2. ASR + two copies (broadcast) folds shift + first copy; second reads dst. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX2);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_asr_to(b, agx_register(4, AGX_SIZE_32),
                                 agx_register(8, AGX_SIZE_32),
                                 agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      emit_copy16(b, 13, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_32));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ASR), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BITOP), 1u);
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_BITOP) {
            EXPECT_EQ(I->src[0].value, 12u);
         }
      }
   }

   /* 3. Placement B when copy dst is read/written between shift and copy. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX2);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(4, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      agx_mov_imm_to(b, agx_register(12, AGX_SIZE_32), 0x1000);
      agx_unit_test(b, agx_register(12, AGX_SIZE_32));
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BITOP), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 1u);
   }

   /* 4. Rejected when high half of shift result is read. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(4, AGX_SIZE_32));
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 5. Rejected when shift low half is still live after the copy. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 5, 4);
      agx_unit_test(b, agx_register(4, AGX_SIZE_32));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 6. Rejected when not tagged shift_amount_zero_or_16. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_ushr_to(b, agx_register(4, AGX_SIZE_32), agx_register(8, AGX_SIZE_32),
                  agx_register(2, AGX_SIZE_32));
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 7. Accepted when high half of shift dst is overwritten before being read. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      agx_mov_imm_to(b, agx_register(5, AGX_SIZE_16), 0x1234);
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(5, AGX_SIZE_16));
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BITOP), 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 1u);
   }

   /* 8. Rejected when first copy dst is overwritten before second copy. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_mov_imm_to(b, agx_register(12, AGX_SIZE_16), 0);
      emit_copy16(b, 13, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_32));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 9. Rejected when word and shift amount overlap. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 10. AGX3B rejects multi-slot async wait on replacement icmpsel; AGX3C accepts. */
   for (agx_isa_encoding enc : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      agx_builder *b = make_postra_builder(memctx, enc);
      agx_context *ctx = b->shader;

      agx_device_load_to(b, agx_register(8, AGX_SIZE_16),
                         agx_uniform(0, AGX_SIZE_64),
                         agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16, 1, 0, 0);
      agx_device_load_to(b, agx_register(9, AGX_SIZE_16),
                         agx_uniform(0, AGX_SIZE_64),
                         agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16, 1, 0, 0);
      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      agx_opt_select_shifted_half(ctx);
      const bool expect_fold = enc == AGX_ISA_ENCODING_AGX3C;
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), expect_fold ? 1u : 0u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), expect_fold ? 0u : 1u);
   }

   /* 11. Rejected when a fused break after the copy keeps the shift result live. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;
      agx_block *blk = agx_start_block(ctx);
      agx_block *brk = agx_test_block(ctx);

      agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                  agx_register(8, AGX_SIZE_32),
                                  agx_register(2, AGX_SIZE_32));
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 12, 4);
      agx_break_if_icmp(b, agx_register(0, AGX_SIZE_32), agx_zero(), 1, false,
                        AGX_ICOND_UEQ, brk);
      agx_unit_test(b, agx_register(12, AGX_SIZE_16));

      b->cursor = agx_after_block(brk);
      agx_unit_test(b, agx_register(4, AGX_SIZE_16));
      (void)blk;

      agx_opt_select_shifted_half(ctx);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_ICMPSEL), 0u);
   }

   /* 12. CSE merges identical shifts differing only in shift_amount_zero_or_16
    * and retains the proof bit on the surviving shift. */
   {
      agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3C);
      agx_context *ctx = b->shader;
      agx_index w = agx_temp(ctx, AGX_SIZE_32);
      agx_index amt = agx_temp(ctx, AGX_SIZE_32);
      agx_index d0 = agx_temp(ctx, AGX_SIZE_32);
      agx_index d1 = agx_temp(ctx, AGX_SIZE_32);
      agx_preload_to(b, w, agx_register(0, AGX_SIZE_32));
      agx_preload_to(b, amt, agx_register(2, AGX_SIZE_32));
      agx_instr *s0 = agx_ushr_to(b, d0, w, amt);
      agx_instr *s1 = agx_ushr_to(b, d1, w, amt);
      s1->shift_amount_zero_or_16 = true;
      agx_unit_test(b, d0);
      agx_unit_test(b, d1);

      agx_opt_cse(ctx);
      agx_dce(ctx, false);
      EXPECT_EQ(count_opcode(ctx, AGX_OPCODE_BFEIL), 1u);
      EXPECT_TRUE(s0->shift_amount_zero_or_16);
   }

   /* 13. AGX3B packed regression: two independent async loads A and B, a
    * tagged shift of A, an intervening icmpsel reading A, B, and the copy
    * destination (forcing placement at the copy), then the low-half copy.
    * Placement at the copy would transfer A's wait onto the intervening
    * icmpsel, making it wait on both A and B and emit two AGX3B carriers. */
   {
      auto pack_agx3b = [&](bool run_fold, unsigned *bytes_out,
                            unsigned *instrs_out) {
         agx_builder *b = make_postra_builder(memctx, AGX_ISA_ENCODING_AGX3B);
         agx_context *ctx = b->shader;
         ctx->key->dev.chip = AGX_CHIP_G17P;
         ctx->stage = MESA_SHADER_COMPUTE;

         /* Load A -> r8, Load B -> r10. */
         agx_device_load_to(b, agx_register(8, AGX_SIZE_32),
                            agx_uniform(0, AGX_SIZE_64),
                            agx_register(0, AGX_SIZE_32), AGX_FORMAT_I32, 1, 0,
                            0);
         agx_device_load_to(b, agx_register(10, AGX_SIZE_32),
                            agx_uniform(0, AGX_SIZE_64),
                            agx_register(2, AGX_SIZE_32), AGX_FORMAT_I32, 1, 0,
                            0);
         /* Tagged shift of A (r8) by r2 -> r4. */
         agx_instr *sh = agx_ushr_to(b, agx_register(4, AGX_SIZE_32),
                                     agx_register(8, AGX_SIZE_32),
                                     agx_register(2, AGX_SIZE_32));
         sh->shift_amount_zero_or_16 = true;
         /* Intervening icmpsel reads A (r8), B (r10), and copy dst r12l (in r12). */
         agx_icmpsel_to(b, agx_register(14, AGX_SIZE_32),
                        agx_register(12, AGX_SIZE_32), agx_zero(),
                        agx_register(8, AGX_SIZE_32),
                        agx_register(10, AGX_SIZE_32), AGX_ICOND_UEQ);
         /* Copy low half of shift result (r4l) into r12l. */
         emit_copy16(b, 12, 4);
         agx_device_store(b, agx_register(12, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16, 1, 0,
                          0);
         agx_device_store(b, agx_register(14, AGX_SIZE_32),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(2, AGX_SIZE_32), AGX_FORMAT_I32, 1, 0,
                          0);
         agx_stop(b);

         agx_opt_register_cache(ctx);
         if (run_fold && agx_opt_select_shifted_half(ctx))
            agx_opt_register_cache(ctx);

         struct util_dynarray bin = UTIL_DYNARRAY_INIT;
         ASSERT_TRUE(agx3_pack_binary(ctx, &bin));
         *bytes_out = bin.size;
         unsigned n_instrs = 0;
         size_t off = 0;
         const uint8_t *code = (const uint8_t *)bin.data;
         while (off < bin.size) {
            signed len = agx3b_instr_length(code + off, bin.size - off, NULL);
            ASSERT_GT(len, 0);
            off += (size_t)len;
            ++n_instrs;
         }
         *instrs_out = n_instrs;
         util_dynarray_fini(&bin);
      };

      unsigned with_bytes = 0, with_instrs = 0;
      unsigned without_bytes = 0, without_instrs = 0;
      pack_agx3b(true, &with_bytes, &with_instrs);
      pack_agx3b(false, &without_bytes, &without_instrs);
      EXPECT_LE(with_instrs, without_instrs);
      EXPECT_LE(with_bytes, without_bytes);
   }

   /* 14. AGX3B packed regression: six outstanding device loads fill all six
    * scoreboard slots, a tagged shift consumes the second load, a store reads
    * the copy destination (forcing placement at the copy), two more loads
    * arrive before the copy, and a later ICMPSEL consumes the third and fourth
    * loads. Moving the shift perturbs physical slot retirement so the later
    * ICMPSEL expands into two wait carriers and grows the binary; the
    * packer-backed AGX3B guard must reject the fold. */
   {
      enum { MODE_UNFOLDED, MODE_OPT, MODE_FORCED_FOLD };
      auto pack_slot_pressure = [&](int mode, unsigned *bytes_out,
                                    unsigned *instrs_out) {
         agx_builder *b = agx_test_builder(memctx);
         agx_context *ctx = b->shader;
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         ctx->key = &key;
         ctx->stage = MESA_SHADER_COMPUTE;
         ctx->max_reg = 32;

         auto load32 = [&](unsigned dst_reg, unsigned idx_reg) {
            agx_device_load_to(b, agx_register(dst_reg, AGX_SIZE_32),
                               agx_uniform(0, AGX_SIZE_64),
                               agx_register(idx_reg, AGX_SIZE_32),
                               AGX_FORMAT_I32, BITFIELD_MASK(1), 0, 0);
         };
         auto store32 = [&](unsigned src_reg, unsigned idx_reg) {
            agx_device_store(b, agx_register(src_reg, AGX_SIZE_32),
                             agx_uniform(0, AGX_SIZE_64),
                             agx_register(idx_reg, AGX_SIZE_32), AGX_FORMAT_I32,
                             BITFIELD_MASK(1), 0, 0);
         };

         /* Six outstanding device loads filling all 6 AGX3B slots. */
         load32(4, 0);   /* load 1 -> r2 */
         load32(6, 0);   /* load 2 -> r3 (consumed by tagged shift) */
         load32(8, 0);   /* load 3 -> r4 (consumed by later ICMPSEL) */
         load32(10, 0);  /* load 4 -> r5 (consumed by later ICMPSEL) */
         load32(12, 0);  /* load 5 -> r6 */
         load32(14, 0);  /* load 6 -> r7 */

         if (mode != MODE_FORCED_FOLD) {
            agx_instr *sh =
               agx_bfeil_to(b, agx_register(16, AGX_SIZE_32), agx_zero(),
                            agx_register(6, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), 0);
            sh->shift_amount_zero_or_16 = true;
         }

         /* Store reads r9l (18), forcing Placement B at the copy. */
         agx_device_store(b, agx_register(18, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16,
                          BITFIELD_MASK(1), 0, 0);

         /* Two more loads arrive while all 6 slots are under pressure. */
         load32(20, 0);  /* load 7 -> r10 */
         load32(22, 0);  /* load 8 -> r11 */

         /* Copy low half of shift result (r8l = 16) into r9l (18), or
          * manually folded ICMPSEL at the copy position. */
         if (mode == MODE_FORCED_FOLD) {
            agx_icmpsel_to(b, agx_register(18, AGX_SIZE_16),
                           agx_register(2, AGX_SIZE_32), agx_zero(),
                           agx_register(6, AGX_SIZE_16),
                           agx_register(7, AGX_SIZE_16), AGX_ICOND_UEQ);
         } else {
            emit_copy16(b, 18, 16);
         }

         /* Later ICMPSEL consuming loads 3 and 4 (r4 and r5). */
         agx_icmpsel_to(b, agx_register(24, AGX_SIZE_32),
                        agx_register(2, AGX_SIZE_32), agx_zero(),
                        agx_register(8, AGX_SIZE_32),
                        agx_register(10, AGX_SIZE_32), AGX_ICOND_UEQ);

         /* Keep remaining loads and results live. */
         agx_device_store(b, agx_register(18, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(2, AGX_SIZE_32), AGX_FORMAT_I16,
                          BITFIELD_MASK(1), 0, 0);
         store32(24, 0);
         store32(4, 0);
         store32(12, 0);
         store32(14, 0);
         store32(20, 0);
         store32(22, 0);
         agx_stop(b);

         agx_opt_register_cache(ctx);
         /* Perturb a preserved incoming hint (as texture publication lowering
          * can) so we also verify that rejecting every fold restores the exact
          * incoming hints rather than overwriting them. */
         agx_instr *first = agx_first_instr(agx_start_block(ctx));
         const bool orig_cache = first->src[1].cache;
         first->src[1].cache = !orig_cache;
         if (mode == MODE_OPT) {
            EXPECT_FALSE(agx_opt_select_shifted_half(ctx));
            EXPECT_EQ(first->src[1].cache, !orig_cache);
         }
         first->src[1].cache = orig_cache;

         struct util_dynarray bin = UTIL_DYNARRAY_INIT;
         ASSERT_TRUE(agx3_pack_binary(ctx, &bin));
         *bytes_out = bin.size;
         unsigned n_instrs = 0;
         size_t off = 0;
         const uint8_t *code = (const uint8_t *)bin.data;
         while (off < bin.size) {
            signed len = agx3b_instr_length(code + off, bin.size - off, NULL);
            ASSERT_GT(len, 0);
            off += (size_t)len;
            ++n_instrs;
         }
         *instrs_out = n_instrs;
         util_dynarray_fini(&bin);
      };

      unsigned without_bytes = 0, without_instrs = 0;
      unsigned with_bytes = 0, with_instrs = 0;
      unsigned forced_bytes = 0, forced_instrs = 0;
      pack_slot_pressure(MODE_UNFOLDED, &without_bytes, &without_instrs);
      pack_slot_pressure(MODE_OPT, &with_bytes, &with_instrs);
      pack_slot_pressure(MODE_FORCED_FOLD, &forced_bytes, &forced_instrs);
      EXPECT_GT(forced_instrs, without_instrs);
      EXPECT_GT(forced_bytes, without_bytes);
      EXPECT_LE(with_instrs, without_instrs);
      EXPECT_LE(with_bytes, without_bytes);
   }

   /* 15. AGX3B relocatable vertex shader: speculative packing during
    * agx_opt_select_shifted_half must not leak UVS relocation records onto
    * the live context. */
   {
      agx_builder *b = agx_test_builder(memctx);
      agx_context *ctx = b->shader;
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.g17p_uvs_reloc_base = 4;
      key.g17p_uvs_reloc_size = 4;
      ctx->key = &key;
      ctx->stage = MESA_SHADER_VERTEX;
      ctx->max_reg = 16;

      agx_device_load_to(b, agx_register(4, AGX_SIZE_32),
                         agx_uniform(0, AGX_SIZE_64),
                         agx_register(0, AGX_SIZE_32), AGX_FORMAT_I32,
                         BITFIELD_MASK(1), 0, 0);
      agx_instr *sh =
         agx_bfeil_to(b, agx_register(6, AGX_SIZE_32), agx_zero(),
                      agx_register(4, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), 0);
      sh->shift_amount_zero_or_16 = true;
      emit_copy16(b, 8, 6);
      agx_instr *st =
         agx_st_vary(b, agx_immediate(4), agx_register(8, AGX_SIZE_32));
      st->last = true;
      agx_stop(b);

      agx_opt_register_cache(ctx);
      ASSERT_TRUE(agx_opt_select_shifted_half(ctx));
      EXPECT_EQ(ctx->agx3_uvs_reloc_count, 0u);
      EXPECT_EQ(ctx->agx3_uvs_relocs.size, 0u);
      agx_opt_register_cache(ctx);

      struct util_dynarray bin = UTIL_DYNARRAY_INIT;
      ASSERT_TRUE(agx3_pack_binary(ctx, &bin));
      EXPECT_EQ(ctx->agx3_uvs_reloc_count, 1u);
      util_dynarray_fini(&bin);
   }

   /* 16. AGX3B batch scalability: 128 independent candidates fold in a single
    * batch verification (O(1) packs) and strictly shrink the packed binary. */
   {
      auto pack_many = [&](bool run_fold, unsigned *bytes_out,
                           unsigned *instrs_out) {
         agx_builder *b = agx_test_builder(memctx);
         agx_context *ctx = b->shader;
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         ctx->key = &key;
         ctx->stage = MESA_SHADER_COMPUTE;
         ctx->max_reg = 16;

         for (unsigned i = 0; i < 128; ++i) {
            agx_instr *sh =
               agx_bfeil_to(b, agx_register(6, AGX_SIZE_32), agx_zero(),
                            agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), 0);
            sh->shift_amount_zero_or_16 = true;
            emit_copy16(b, 8, 6);
            agx_device_store(b, agx_register(8, AGX_SIZE_16),
                             agx_uniform(0, AGX_SIZE_64),
                             agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16,
                             BITFIELD_MASK(1), 0, 0);
         }
         agx_stop(b);

         agx_opt_register_cache(ctx);
         if (run_fold) {
            ASSERT_TRUE(agx_opt_select_shifted_half(ctx));
            agx_opt_register_cache(ctx);
         }

         struct util_dynarray bin = UTIL_DYNARRAY_INIT;
         ASSERT_TRUE(agx3_pack_binary(ctx, &bin));
         *bytes_out = bin.size;
         unsigned n_instrs = 0;
         size_t off = 0;
         const uint8_t *code = (const uint8_t *)bin.data;
         while (off < bin.size) {
            signed len = agx3b_instr_length(code + off, bin.size - off, NULL);
            ASSERT_GT(len, 0);
            off += (size_t)len;
            ++n_instrs;
         }
         *instrs_out = n_instrs;
         util_dynarray_fini(&bin);
      };

      unsigned with_bytes = 0, with_instrs = 0;
      unsigned without_bytes = 0, without_instrs = 0;
      pack_many(true, &with_bytes, &with_instrs);
      pack_many(false, &without_bytes, &without_instrs);
      EXPECT_EQ(with_instrs + 128, without_instrs);
      EXPECT_LT(with_bytes, without_bytes);
   }

   /* 17. AGX3B batch rollback unblocks earlier candidate: a later shift S2
    * (only legal at S2) writes S1's source word ahead of S1's copy during the
    * reverse batch scan, temporarily blocking S1 (only legal at its copy).
    * When the packer rejects S2 (two-load ISEL carrier growth) and rolls back
    * the batch, the fallback scan must still visit and commit S1. */
   {
      auto pack_unblocked = [&](bool run_fold, unsigned *bytes_out,
                                unsigned *instrs_out) {
         agx_builder *b = agx_test_builder(memctx);
         agx_context *ctx = b->shader;
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         ctx->key = &key;
         ctx->stage = MESA_SHADER_COMPUTE;
         ctx->max_reg = 24;

         /* Two independent loads consumed by S2 so folding S2 into an ICMPSEL
          * triggers AGX3B multi-slot carrier expansion and gets rejected. */
         agx_device_load_to(b, agx_register(8, AGX_SIZE_32),
                            agx_uniform(0, AGX_SIZE_64),
                            agx_register(0, AGX_SIZE_32), AGX_FORMAT_I32,
                            BITFIELD_MASK(1), 0, 0);
         agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
                            agx_uniform(0, AGX_SIZE_64),
                            agx_register(0, AGX_SIZE_32), AGX_FORMAT_I32,
                            BITFIELD_MASK(1), 0, 0);

         /* S1: shift r6_r7 by r12_r13 into r14_r15. */
         agx_instr *s1 =
            agx_bfeil_to(b, agx_register(14, AGX_SIZE_32), agx_zero(),
                         agx_register(6, AGX_SIZE_32),
                         agx_register(12, AGX_SIZE_32), 0);
         s1->shift_amount_zero_or_16 = true;

         /* Store reads r16l, forcing S1 to be legal only at its copy c1_1. */
         agx_device_store(b, agx_register(16, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16,
                          BITFIELD_MASK(1), 0, 0);

         /* S2: shift r8_r9 by r2_r3 into r18_r19; its copy c1_2 writes r6l. */
         agx_instr *s2 =
            agx_bfeil_to(b, agx_register(18, AGX_SIZE_32), agx_zero(),
                         agx_register(8, AGX_SIZE_32),
                         agx_register(2, AGX_SIZE_32), 0);
         s2->shift_amount_zero_or_16 = true;

         /* c1_1: low-half copy of S1 into r16l. */
         emit_copy16(b, 16, 14);

         /* Clobber r8_r9 before c1_2, forcing S2 to be legal only at S2. */
         agx_mov_imm_to(b, agx_register(8, AGX_SIZE_32), 1);

         /* c1_2: low-half copy of S2 into r6l. */
         emit_copy16(b, 6, 18);

         agx_device_store(b, agx_register(16, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16,
                          BITFIELD_MASK(1), 0, 0);
         agx_device_store(b, agx_register(6, AGX_SIZE_16),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I16,
                          BITFIELD_MASK(1), 0, 0);
         agx_device_store(b, agx_register(8, AGX_SIZE_32),
                          agx_uniform(0, AGX_SIZE_64),
                          agx_register(0, AGX_SIZE_32), AGX_FORMAT_I32,
                          BITFIELD_MASK(1), 0, 0);
         agx_stop(b);

         agx_opt_register_cache(ctx);
         if (run_fold) {
            ASSERT_TRUE(agx_opt_select_shifted_half(ctx));
            agx_opt_register_cache(ctx);
         }

         struct util_dynarray bin = UTIL_DYNARRAY_INIT;
         ASSERT_TRUE(agx3_pack_binary(ctx, &bin));
         *bytes_out = bin.size;
         unsigned n_instrs = 0;
         size_t off = 0;
         const uint8_t *code = (const uint8_t *)bin.data;
         while (off < bin.size) {
            signed len = agx3b_instr_length(code + off, bin.size - off, NULL);
            ASSERT_GT(len, 0);
            off += (size_t)len;
            ++n_instrs;
         }
         *instrs_out = n_instrs;
         util_dynarray_fini(&bin);
      };

      unsigned with_bytes = 0, with_instrs = 0;
      unsigned without_bytes = 0, without_instrs = 0;
      pack_unblocked(true, &with_bytes, &with_instrs);
      pack_unblocked(false, &without_bytes, &without_instrs);
      EXPECT_LT(with_instrs, without_instrs);
      EXPECT_LT(with_bytes, without_bytes);
   }

   ralloc_free(memctx);
}

/* Whole compute shaders, compiled from identical NIR with the post-RA fold
 * enabled and disabled. */
struct shape_builder {
   nir_builder *b;
   nir_def *base, *lane, *s;
   bool arithmetic = false;

   nir_def *addr(unsigned offset, unsigned stride)
   {
      return nir_iadd(b, nir_iadd_imm(b, base, offset),
                      nir_u2u64(b, nir_imul_imm(b, lane, stride)));
   }
   nir_def *load(unsigned offset)
   {
      return nir_load_global(b, 1, 32, addr(offset, 4), .align_mul = 4);
   }
   nir_def *load16(unsigned offset)
   {
      return nir_load_global(b, 1, 16, addr(offset, 2), .align_mul = 2);
   }
   nir_def *load4(unsigned offset)
   {
      return nir_load_global(b, 4, 32, addr(offset, 16), .align_mul = 16);
   }
   void store(nir_def *v, unsigned offset, unsigned stride)
   {
      nir_store_global(b, v, addr(offset, stride), .align_mul = stride);
   }
   nir_def *shift(nir_def *w)
   {
      return arithmetic ? nir_ishr(b, w, s) : nir_ushr(b, w, s);
   }
   nir_def *half(nir_def *w) { return nir_u2u16(b, shift(w)); }
   nir_def *pack4(nir_def *w)
   {
      return nir_vec4(b, half(nir_channel(b, w, 0)), half(nir_channel(b, w, 1)),
                      half(nir_channel(b, w, 2)), half(nir_channel(b, w, 3)));
   }
   void push_loop() { nir_push_loop(b); }
   void pop_loop()
   {
      nir_def *poll = nir_load_global(b, 1, 32, addr(16384, 4),
                                      .access = ACCESS_VOLATILE, .align_mul = 4);
      nir_break_if(b, nir_ieq_imm(b, poll, 0));
      nir_pop_loop(b, NULL);
   }
   void push_if(unsigned bit)
   {
      nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, lane, bit), 0));
   }
};

enum shape {
   /* Four loaded words, their halves gathered into one vector: smaller. */
   FOUR_WORD_PACK,
   FOUR_WORD_PACK_IN_LOOP,
   /* No larger: */
   FOUR_WORDS_ALSO_STORED,    /* the shifted words are stored too */
   FOUR_WORD_VECTOR_STORED,   /* the loaded vector is stored too */
   FOUR_WORDS_BEFORE_LOOP,    /* loaded before a loop, packed in it */
   WORD_AND_HALF,             /* the shift stored whole and as a half */
   TWO_IFS,                   /* a half stored in each of two ifs */
   IF_ELSE,                   /* a half stored in each arm */
   SHIFT_BEFORE_LOOP,         /* two halves stored in a loop */
   SHIFT_IN_LOOP,             /* shift and two half stores in a loop */
   ONE_REMOTE_IF,             /* two half stores in one if */
   LITERAL_WORD,              /* a half of a literal word */
   DUPLICATE_IN_LOOP,         /* vec2(h, h) in a loop, word loaded before */
   DUPLICATE_IN_LOOP_STORED,  /* the same, with the word stored in the loop */
   TWO_IN_LOOP_STORED,        /* two words' halves, both words stored */
   DUPLICATE_WORD_STORED,     /* vec2(h, h) and the word stored, no loop */
   BOTH_HALVES_OF_WORD,       /* vec2 of a word's selected and other half */
   /* The four regression shapes from the earlier NIR pass: */
   UNIFORM_WORD_VEC2_DUP,     /* uniform word, half used twice in vec2 */
   HALF_BESIDE_CONST16,       /* loaded word's half packed beside 16-bit const */
   HALF_BROADCAST_VEC4_IN_IF, /* loaded word's half broadcast to vec4 in if */
   TWO_HALF_LOADS_VEC2_DUP,   /* word from two 16-bit loads, half used twice */
   NUM_SHAPES,
};

static void
build_shape(shape_builder &p, enum shape shape)
{
   nir_builder *b = p.b;
   switch (shape) {
   case FOUR_WORD_PACK:
      p.store(p.pack4(p.load4(0)), 4096, 8);
      break;
   case FOUR_WORD_PACK_IN_LOOP:
      p.push_loop();
      p.store(p.pack4(p.load4(0)), 4096, 8);
      p.pop_loop();
      break;
   case FOUR_WORDS_ALSO_STORED: {
      nir_def *w = p.load4(0);
      p.store(p.pack4(w), 4096, 8);
      p.store(nir_ushr(b, w, p.s), 8192, 16);
      break;
   }
   case FOUR_WORD_VECTOR_STORED: {
      nir_def *w = p.load4(0);
      p.store(p.pack4(w), 4096, 8);
      p.store(w, 8192, 16);
      break;
   }
   case FOUR_WORDS_BEFORE_LOOP: {
      nir_def *w = p.load4(0);
      p.push_loop();
      p.store(p.pack4(w), 4096, 8);
      p.pop_loop();
      break;
   }
   case WORD_AND_HALF: {
      nir_def *sh = nir_ushr(b, p.load(0), p.s);
      p.store(sh, 4096, 4);
      p.store(nir_u2u16(b, sh), 8192, 2);
      break;
   }
   case TWO_IFS: {
      nir_def *sh = nir_ushr(b, p.load(0), p.s);
      p.push_if(4);
      p.store(nir_u2u16(b, sh), 4096, 2);
      nir_pop_if(b, NULL);
      p.push_if(8);
      p.store(nir_u2u16(b, sh), 8192, 2);
      nir_pop_if(b, NULL);
      break;
   }
   case IF_ELSE: {
      nir_def *sh = nir_ushr(b, p.load(0), p.s);
      p.push_if(4);
      p.store(nir_u2u16(b, sh), 4096, 2);
      nir_push_else(b, NULL);
      p.store(nir_u2u16(b, sh), 8192, 2);
      nir_pop_if(b, NULL);
      break;
   }
   case SHIFT_BEFORE_LOOP: {
      nir_def *sh = nir_ushr(b, p.load(0), p.s);
      p.push_loop();
      p.store(nir_u2u16(b, sh), 8192, 2);
      p.store(nir_u2u16(b, sh), 12288, 2);
      p.pop_loop();
      break;
   }
   case SHIFT_IN_LOOP: {
      nir_def *w = p.load(0);
      p.push_loop();
      nir_def *sh = nir_ushr(b, w, p.s);
      p.store(nir_u2u16(b, sh), 8192, 2);
      p.store(nir_u2u16(b, sh), 12288, 2);
      p.pop_loop();
      break;
   }
   case ONE_REMOTE_IF: {
      nir_def *sh = nir_ushr(b, p.load(0), p.s);
      p.push_if(4);
      p.store(nir_u2u16(b, sh), 4096, 2);
      p.store(nir_u2u16(b, sh), 8192, 2);
      nir_pop_if(b, NULL);
      break;
   }
   case LITERAL_WORD:
      p.store(nir_vec2(b, p.half(nir_imm_int(b, 0x12345678)), p.half(p.load(0))),
              8192, 4);
      break;
   case DUPLICATE_IN_LOOP:
   case DUPLICATE_IN_LOOP_STORED: {
      nir_def *w = p.load(0);
      p.push_loop();
      nir_def *h = p.half(w);
      p.store(nir_vec2(b, h, h), 8192, 4);
      if (shape == DUPLICATE_IN_LOOP_STORED)
         p.store(w, 12288, 4);
      p.pop_loop();
      break;
   }
   case TWO_IN_LOOP_STORED: {
      nir_def *w = p.load(0), *w2 = p.load(4096);
      p.push_loop();
      p.store(nir_vec2(b, p.half(w), p.half(w2)), 8192, 4);
      p.store(w, 12288, 4);
      p.store(w2, 20480, 4);
      p.pop_loop();
      break;
   }
   case DUPLICATE_WORD_STORED: {
      nir_def *w = p.load(0);
      nir_def *h = p.half(w);
      p.store(nir_vec2(b, h, h), 8192, 4);
      p.store(w, 12288, 4);
      break;
   }
   case BOTH_HALVES_OF_WORD: {
      nir_def *w = p.load(0);
      nir_def *other = nir_u2u16(b, nir_ushr(b, w, nir_isub_imm(b, 16, p.s)));
      p.store(nir_vec2(b, p.half(w), other), 8192, 4);
      break;
   }
   case UNIFORM_WORD_VEC2_DUP: {
      nir_def *w = agx_test_load_preamble(b, 1, 32, 4);
      nir_def *h = p.half(w);
      p.store(nir_vec2(b, h, h), 8192, 4);
      break;
   }
   case HALF_BESIDE_CONST16: {
      nir_def *h = p.half(p.load(0));
      p.store(nir_vec2(b, h, nir_imm_intN_t(b, 0x3c00, 16)), 8192, 4);
      break;
   }
   case HALF_BROADCAST_VEC4_IN_IF: {
      nir_def *w = p.load(0);
      p.push_if(4);
      nir_def *h = p.half(w);
      p.store(nir_vec4(b, h, h, h, h), 8192, 8);
      nir_pop_if(b, NULL);
      break;
   }
   case TWO_HALF_LOADS_VEC2_DUP: {
      nir_def *lo = p.load16(0);
      nir_def *hi = p.load16(2048);
      nir_def *w = nir_pack_32_2x16_split(b, lo, hi);
      nir_def *h = p.half(w);
      p.store(nir_vec2(b, h, h), 8192, 4);
      break;
   }
   default:
      UNREACHABLE("shape");
   }
}

struct size {
   unsigned instrs, bytes;
};

struct target {
   agx_isa_encoding encoding;
   agx_chip chip;
   bool emit_agx3d;
};

static const target kTargets[] = {
   {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
   {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, false},
   /* G18G compiles as G17G and re-encodes as agx3d */
   {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, true},
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
   {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
};

static unsigned
count_packed_instrs(const target &t, const uint8_t *code, size_t size)
{
   unsigned count = 0;
   size_t off = 0;
   while (off < size) {
      signed len =
         t.encoding == AGX_ISA_ENCODING_AGX2
            ? agx2_instr_length(code + off, size - off, NULL)
         : t.encoding == AGX_ISA_ENCODING_AGX3B
            ? agx3b_instr_length(code + off, size - off, NULL)
         : t.emit_agx3d
            ? agx3d_instr_length(code + off, size - off, NULL)
            : agx3c_instr_length(code + off, size - off, NULL);
      EXPECT_GT(len, 0);
      if (len <= 0)
         break;
      /* Stop at trailing zero alignment padding after the terminal stop. */
      if (t.encoding == AGX_ISA_ENCODING_AGX2 && len == 2 && code[off] == 0 &&
          code[off + 1] == 0)
         break;
      off += (size_t)len;
      ++count;
   }
   return count;
}

template <typename BuildFn>
static void
compile_pair(const target &t, BuildFn &&build_fn, size *with_out,
             size *without_out)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "shape");
   shape_builder p;
   p.b = &b;
   p.base = agx_test_load_preamble(&b, 1, 64, 0);
   p.lane = nir_load_local_invocation_index(&b);
   p.s = nir_iand_imm(&b, p.lane, 16);
   build_fn(p);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_preprocess_nir(b.shader);

   nir_shader *baseline = nir_shader_clone(NULL, b.shader);

   auto finish = [&](nir_shader *s, bool run_fold) -> size {
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_shader_part part = {};
      EXPECT_TRUE(run_fold
                     ? agx_compile_shader_nir(s, &key, &part)
                     : agx_compile_shader_nir_without_select_shifted_half(
                          s, &key, &part));
      const uint8_t *code =
         (const uint8_t *)part.binary + part.info.main_offset;
      const size result = {
         count_packed_instrs(t, code, part.info.main_size),
         part.info.binary_size,
      };
      free(part.binary);
      ralloc_free(s);
      return result;
   };

   *with_out = finish(b.shader, true);
   *without_out = finish(baseline, false);
}

TEST(SelectShiftedHalf, CompilesSmallerOnEveryEncoding)
{
   for (unsigned t = 0; t < ARRAY_SIZE(kTargets); ++t) {
      for (unsigned s = 0; s < NUM_SHAPES; ++s) {
         size with = {}, without = {};
         compile_pair(
            kTargets[t], [&](shape_builder &p) { build_shape(p, (enum shape)s); },
            &with, &without);
         /* The four-word packs select each half into place instead of
          * shifting the word and copying its low half. */
         if (s <= FOUR_WORD_PACK_IN_LOOP)
            EXPECT_LT(with.instrs, without.instrs)
               << "target " << t << " shape " << s;
         else
            EXPECT_LE(with.instrs, without.instrs)
               << "target " << t << " shape " << s;
         EXPECT_LE(with.bytes, without.bytes)
            << "target " << t << " shape " << s;
      }
   }
}

/* 936-shape synthetic sweep (6 word sources x 13 half uses x 3 extra uses x
 * 4 control-flow/shift-op contexts) across all five targets (4,680 pairs):
 * verify zero regressions in instruction count or binary size. */
static void
build_sweep_shape(shape_builder &p, unsigned word_src, unsigned half_use,
                  unsigned extra_use, unsigned cf_mode)
{
   nir_builder *b = p.b;
   p.arithmetic = (cf_mode == 1);

   auto make_word = [&](unsigned off) -> nir_def * {
      switch (word_src) {
      case 0:
         return p.load(off);
      case 1:
         return p.load4(off);
      case 2:
         return agx_test_load_preamble(b, 1, 32, 4 + (off ? 2 : 0));
      case 3:
         return nir_pack_32_2x16_split(b, p.load16(off), p.load16(off + 2048));
      case 4:
         return nir_ixor(b, p.load(off), p.lane);
      case 5:
         return nir_imm_int(b, off ? 0x9abcdef0u : 0x12345678u);
      default:
         UNREACHABLE("word_src");
      }
   };

   const bool in_if = (cf_mode == 2);
   const bool in_loop = (cf_mode == 3);
   nir_def *w = NULL, *w2 = NULL;

   if (in_if) {
      w = make_word(0);
      if (half_use == 4)
         w2 = make_word(512);
      p.push_if(4);
   } else if (in_loop) {
      /* Load word before loop when extra_use == 1, inside loop otherwise. */
      if (extra_use == 1) {
         w = make_word(0);
         if (half_use == 4)
            w2 = make_word(512);
      }
      p.push_loop();
      if (extra_use != 1) {
         w = make_word(0);
         if (half_use == 4)
            w2 = make_word(512);
      }
   } else {
      w = make_word(0);
      if (half_use == 4)
         w2 = make_word(512);
   }

   nir_def *w0 = w->num_components > 1 ? nir_channel(b, w, 0) : w;
   nir_def *w2_0 = w2 ? (w2->num_components > 1 ? nir_channel(b, w2, 0) : w2) : NULL;
   nir_def *sh0 = p.shift(w0);
   nir_def *h0 = nir_u2u16(b, sh0);

   switch (half_use) {
   case 0:
      p.store(h0, 4096, 2);
      break;
   case 1:
      p.store(nir_vec2(b, h0, h0), 4096, 4);
      break;
   case 2:
      p.store(nir_vec2(b, h0, nir_imm_intN_t(b, 0x3c00, 16)), 4096, 4);
      break;
   case 3:
      p.store(nir_vec2(b, h0, p.load16(1024)), 4096, 4);
      break;
   case 4:
      p.store(nir_vec2(b, h0, p.half(w2_0)), 4096, 4);
      break;
   case 5:
      p.store(nir_vec4(b, h0, h0, h0, h0), 4096, 8);
      break;
   case 6:
      if (w->num_components == 4)
         p.store(p.pack4(w), 4096, 8);
      else
         p.store(nir_vec4(b, h0, p.half(p.load(512)), p.half(p.load(1024)),
                          p.half(p.load(1536))),
                 4096, 8);
      break;
   case 7: {
      nir_def *other =
         nir_u2u16(b, nir_ushr(b, w0, nir_isub_imm(b, 16, p.s)));
      p.store(nir_vec2(b, h0, other), 4096, 4);
      break;
   }
   case 8:
      p.store(nir_iadd_imm(b, h0, 1), 4096, 2);
      break;
   case 9:
      p.store(nir_vec2(b, nir_iadd_imm(b, h0, 1), h0), 4096, 4);
      break;
   case 10:
      p.store(h0, 4096, 2);
      p.store(h0, 6144, 2);
      break;
   case 11:
      p.store(nir_vec2(b, h0, h0), 4096, 4);
      p.store(nir_vec2(b, h0, nir_imm_intN_t(b, 0x1234, 16)), 6144, 4);
      break;
   case 12:
      p.store(nir_vec4(b, h0, nir_imm_intN_t(b, 0x3c00, 16), h0, p.load16(1024)),
              4096, 8);
      break;
   default:
      UNREACHABLE("half_use");
   }

   if (extra_use == 1)
      p.store(w, 8192, 4 * w->num_components);
   else if (extra_use == 2)
      p.store(sh0, 8192, 4);

   if (in_if)
      nir_pop_if(b, NULL);
   else if (in_loop)
      p.pop_loop();
}

TEST(SelectShiftedHalf, Sweep936ShapesNeverGrows)
{
   unsigned smaller = 0;
   for (unsigned t = 0; t < ARRAY_SIZE(kTargets); ++t) {
      for (unsigned ws = 0; ws < 6; ++ws) {
         for (unsigned hu = 0; hu < 13; ++hu) {
            for (unsigned eu = 0; eu < 3; ++eu) {
               for (unsigned cf = 0; cf < 4; ++cf) {
                  size with = {}, without = {};
                  compile_pair(
                     kTargets[t],
                     [&](shape_builder &p) {
                        build_sweep_shape(p, ws, hu, eu, cf);
                     },
                     &with, &without);
                  EXPECT_LE(with.instrs, without.instrs)
                     << "t=" << t << " ws=" << ws << " hu=" << hu
                     << " eu=" << eu << " cf=" << cf;
                  EXPECT_LE(with.bytes, without.bytes)
                     << "t=" << t << " ws=" << ws << " hu=" << hu
                     << " eu=" << eu << " cf=" << cf;
                  if (with.instrs < without.instrs)
                     ++smaller;
               }
            }
         }
      }
   }
   EXPECT_GT(smaller, 0u);
}

} // namespace
