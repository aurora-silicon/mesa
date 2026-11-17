/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* agx_nir_select_shifted_half rewrites u2u16(w >> s) into a select of one of
 * w's halves when s is provably 0 or 16. Each shape of s below is either
 * recognized or rejected; recognized or not, the shader must compute the low
 * half of w >> (s & 31) (checked by substituting w and l and folding). A
 * whole compute shader must also get smaller on M5 and stay the same on the
 * targets that do not run the pass.
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

/* Builds u2u16(shift(w, amount)) (the word optionally a swizzled vector
 * component), runs the pass, substitutes w and l and folds. Returns the half
 * the shader computes; *progress tells whether the pass rewrote it. */
static uint16_t
evaluate(enum amount a, bool arithmetic, bool vector_word, uint32_t wv,
         uint32_t lv, bool *progress)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "shift");
   nir_def *w = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *l = agx_test_load_preamble(&b, 1, 32, 2);
   nir_def *word = w;
   if (vector_word)
      word = nir_channel(&b, nir_vec2(&b, nir_imm_int(&b, 0x12345678), w), 1);
   nir_def *s = build_amount(&b, a, l);
   nir_def *shifted =
      arithmetic ? nir_ishr(&b, word, s) : nir_ushr(&b, word, s);
   agx_test_store_preamble(&b, nir_u2u16(&b, shifted), 4);

   *progress = false;
   NIR_PASS(*progress, b.shader, agx_nir_select_shifted_half);

   b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
   nir_def_rewrite_uses(w, nir_imm_int(&b, wv));
   nir_def_rewrite_uses(l, nir_imm_int(&b, lv));
   nir_opt_constant_folding(b.shader);

   uint16_t result = 0xdead;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic !=
                nir_intrinsic_store_preamble)
            continue;
         nir_src *src = &nir_instr_as_intrinsic(instr)->src[0];
         EXPECT_TRUE(nir_src_is_const(*src));
         result = (uint16_t)nir_src_as_uint(*src);
      }
   }
   ralloc_free(b.shader);
   return result;
}

TEST(M5SelectShiftedHalf, RecognizesOnlyAmountsOfZeroOrSixteen)
{
   const uint32_t ws[] = {0x00000000u, 0x0000ffffu, 0xffff0000u,
                          0x80007fffu, 0x7fff8000u, 0xdeadbeefu};
   const uint32_t ls[] = {0u,  1u,  8u,          9u,          16u,
                          24u, 48u, 0x80000018u, 0xffffffffu, 0x7ffffff7u};
   for (unsigned a = 0; a < NUM_AMOUNTS; ++a) {
      for (bool arithmetic : {false, true}) {
         for (bool vector_word : {false, true}) {
            for (uint32_t w : ws) {
               for (uint32_t l : ls) {
                  bool progress;
                  const uint16_t got = evaluate((enum amount)a, arithmetic,
                                                vector_word, w, l, &progress);
                  const unsigned k = eval_amount((enum amount)a, l) & 31;
                  const uint32_t want =
                     arithmetic ? (uint32_t)((int32_t)w >> k) : w >> k;
                  EXPECT_EQ(got, (uint16_t)want)
                     << "amount " << a << " arithmetic " << arithmetic
                     << std::hex << " w 0x" << w << " l 0x" << l;
                  EXPECT_EQ(progress, a < FIRST_REJECT) << "amount " << a;
               }
            }
         }
      }
   }
}

/* A compute shader that loads four words and stores the low half of each
 * shifted right by lane & 16 (or by lane & 0x30, which the pass must leave
 * alone and which costs the same otherwise), through the whole compiler. */
static unsigned
compile_halves(agx_isa_encoding encoding, agx_chip chip, bool emit_agx3d,
               bool provable, bool *ok)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "halves");
   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *s = nir_iand_imm(&b, lane, provable ? 16 : 0x30);
   nir_def *offs = nir_u2u64(&b, nir_imul_imm(&b, lane, 16));
   nir_def *w =
      nir_load_global(&b, 4, 32, nir_iadd(&b, addr, offs), .align_mul = 16);
   /* Stored as one vector of halves, which needs them in consecutive 16-bit
    * registers (as a matrix operand does): a shifted word's low half is
    * copied there unless the half is selected. */
   nir_def *h[4];
   for (unsigned i = 0; i < 4; ++i)
      h[i] = nir_u2u16(&b, nir_ushr(&b, nir_channel(&b, w, i), s));
   nir_store_global(&b, nir_vec4(&b, h[0], h[1], h[2], h[3]),
                    nir_iadd(&b, nir_iadd_imm(&b, addr, 4096), offs),
                    .align_mul = 8);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.dev.chip = chip;
   key.dev.emit_agx3d = emit_agx3d;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   *ok = agx_compile_shader_nir(b.shader, &key, &part);
   const unsigned instrs = part.info.stats.instrs;
   free(part.binary);
   ralloc_free(b.shader);
   return instrs;
}

TEST(M5SelectShiftedHalf, CompilesSmallerOnlyOnM5)
{
   struct target {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
   };
   unsigned with[4], without[4];
   for (unsigned t = 0; t < 4; ++t) {
      bool ok;
      with[t] = compile_halves(targets[t].encoding, targets[t].chip,
                               targets[t].emit_agx3d, true, &ok);
      EXPECT_TRUE(ok) << "target " << t;
      without[t] = compile_halves(targets[t].encoding, targets[t].chip,
                                  targets[t].emit_agx3d, false, &ok);
      EXPECT_TRUE(ok) << "target " << t;
   }
   /* M5 selects each of the four halves into place instead of shifting the
    * word and copying its low half; the other targets compile both alike. */
   EXPECT_LT(with[0], without[0]) << "M5";
   for (unsigned t = 1; t < 4; ++t)
      EXPECT_EQ(with[t], without[t]) << "target " << t;
}

} // namespace
