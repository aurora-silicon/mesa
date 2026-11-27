/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* agx_nir_fold_pixel_coord turns the pixel-centre round trip
 * f2[ui]32[_sat](u2f32(a@16) + 0.5) back into u2u32(a). Every 16-bit input
 * lands in [0.5, 65535.5], so truncation, and saturation to the 32-bit
 * range, both give a back; the folded shader must keep that value for every
 * input and leave no float work behind.
 */

namespace {

enum class conv { f2u32, f2i32, f2u32_sat, f2i32_sat };

static nir_def *
build_conv(nir_builder *b, conv kind, nir_def *x)
{
   switch (kind) {
   case conv::f2u32:
      return nir_f2u32(b, x);
   case conv::f2i32:
      return nir_f2i32(b, x);
   case conv::f2u32_sat:
      return nir_f2u32_sat(b, x);
   case conv::f2i32_sat:
      return nir_f2i32_sat(b, x);
   }
   return NULL;
}

static unsigned
count_float_alu(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         const nir_op op = nir_instr_as_alu(instr)->op;
         if (op == nir_op_fadd || op == nir_op_u2f32 || op == nir_op_f2u32 ||
             op == nir_op_f2i32 || op == nir_op_f2u32_sat ||
             op == nir_op_f2i32_sat)
            n++;
      }
   }
   return n;
}

static void
check_folds(conv kind)
{
   const uint16_t values[] = {0, 1, 2, 0x7ffe, 0x7fff, 0x8000,
                              0x8001, 0xfffe, 0xffff, 1919};
   for (uint16_t v : values) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "pixel coord");
      nir_def *in = agx_test_load_preamble(&b, 1, 16, 0);
      nir_def *f = nir_fadd_imm(&b, nir_u2f32(&b, in), 0.5);
      agx_test_store_preamble(&b, build_conv(&b, kind, f), 4);

      bool progress = false;
      NIR_PASS(progress, b.shader, agx_nir_fold_pixel_coord);
      EXPECT_TRUE(progress);
      nir_opt_dce(b.shader);
      EXPECT_EQ(count_float_alu(b.shader), 0u);

      b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
      nir_def_rewrite_uses(in, nir_imm_intN_t(&b, v, 16));
      nir_opt_constant_folding(b.shader);
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_store_preamble)
               continue;
            nir_src *src = &nir_instr_as_intrinsic(instr)->src[0];
            ASSERT_TRUE(nir_src_is_const(*src));
            EXPECT_EQ(nir_src_as_uint(*src), (uint64_t)v)
               << "value " << v << " kind " << (int)kind;
         }
      }
      ralloc_free(b.shader);
   }
}

/* A 32-bit source can exceed the exactly representable range, and an addend
 * other than 0.5 changes the rounding: neither may fold. */
static void
check_no_fold(conv kind, bool wide, float addend)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "pixel coord");
   nir_def *in = agx_test_load_preamble(&b, 1, wide ? 32 : 16, 0);
   nir_def *f = nir_fadd_imm(&b, nir_u2f32(&b, in), addend);
   agx_test_store_preamble(&b, build_conv(&b, kind, f), 4);

   bool progress = false;
   NIR_PASS(progress, b.shader, agx_nir_fold_pixel_coord);
   EXPECT_FALSE(progress) << "kind " << (int)kind << " wide " << wide;
   ralloc_free(b.shader);
}

TEST(FoldPixelCoord, TruncatingConversionsFold)
{
   check_folds(conv::f2u32);
   check_folds(conv::f2i32);
}

TEST(FoldPixelCoord, SaturatingConversionsFold)
{
   check_folds(conv::f2u32_sat);
   check_folds(conv::f2i32_sat);
}

TEST(FoldPixelCoord, WideSourceOrOtherAddendStays)
{
   for (conv kind : {conv::f2u32, conv::f2i32, conv::f2u32_sat,
                     conv::f2i32_sat}) {
      check_no_fold(kind, true, 0.5f);
      check_no_fold(kind, false, 0.25f);
   }
}

} // namespace
