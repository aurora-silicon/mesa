/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

namespace {

const uint32_t values[] = {
   0x00000000, 0x80000000, /* +0, -0 */
   0x00000001, 0x80000001, /* smallest denormals */
   0x00000002, 0x80000002, /* distinct denormals of the same signs */
   0x3f800000, 0xbf800000, /* +1, -1 */
   0x7f800000, 0xff800000, /* infinities */
   0x7fc00000,             /* NaN */
};

bool
is_nan(uint32_t x)
{
   return (x & 0x7f800000u) == 0x7f800000u && (x & 0x007fffffu);
}

float
as_float(uint32_t x)
{
   float f;
   memcpy(&f, &x, 4);
   return f;
}

/* IEEE 754-2019 minimumNumber/maximumNumber: -0 < +0, a NaN loses. */
uint32_t
reference(uint32_t x, uint32_t y, bool is_max, bool ftz)
{
   if (ftz) {
      x = (x & 0x7f800000u) ? x : (x & 0x80000000u);
      y = (y & 0x7f800000u) ? y : (y & 0x80000000u);
   }
   if (is_nan(x))
      return y;
   if (is_nan(y))
      return x;
   if (as_float(x) == as_float(y)) {
      bool neg = is_max ? (x & y & 0x80000000u) : ((x | y) & 0x80000000u);
      return neg ? (x | 0x80000000u) : (x & 0x7fffffffu);
   }
   return (as_float(x) < as_float(y)) != is_max ? x : y;
}

unsigned
count(nir_shader *s, nir_op op)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         n += instr->type == nir_instr_type_alu &&
              nir_instr_as_alu(instr)->op == op;
      }
   }
   return n;
}

/* Lower fmin/fmax(x, y), or (y, x) if swap, with signed zeros preserved, then
 * fold it for x, y. */
uint32_t
lowered(uint32_t x, uint32_t y, bool is_max, bool ftz, bool constant_y,
        bool swap)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "signed zero minmax");
   if (ftz) {
      b.shader->info.float_controls_execution_mode =
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   }
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *c = constant_y ? nir_imm_int(&b, y)
                           : agx_test_load_preamble(&b, 1, 32, 2);
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   nir_def *s0 = swap ? c : a, *s1 = swap ? a : c;
   nir_def *out = is_max ? nir_fmax(&b, s0, s1) : nir_fmin(&b, s0, s1);
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);

   EXPECT_TRUE(agx_nir_lower_fminmax_signed_zero(b.shader, true));
   EXPECT_EQ(count(b.shader, nir_op_fmin) + count(b.shader, nir_op_fmax), 0u);
   if (!constant_y) {
      EXPECT_EQ(count(b.shader, nir_op_bcsel), 1u);
   } else if ((y & 0x7fffffffu) == 0) {
      EXPECT_EQ(count(b.shader, nir_op_bcsel) +
                   count(b.shader, nir_op_fmin_agx) +
                   count(b.shader, nir_op_fmax_agx),
                1u);
   }
   nir_validate_shader(b.shader, "signed zero minmax");

   b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
   nir_def_rewrite_uses(a, nir_imm_int(&b, x));
   if (!constant_y)
      nir_def_rewrite_uses(c, nir_imm_int(&b, y));
   nir_opt_constant_folding(b.shader);

   uint32_t result = ~0u;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_store_output) {
            nir_src *src = &nir_instr_as_intrinsic(instr)->src[0];
            EXPECT_TRUE(nir_src_is_const(*src));
            result = nir_src_as_uint(*src);
         }
      }
   }
   ralloc_free(b.shader);
   return result;
}

} // namespace

TEST(FminmaxSignedZero, MatchesIEEE)
{
   for (bool swap : {false, true}) {
   for (bool constant_y : {false, true}) {
   for (bool ftz : {false, true}) {
      for (bool is_max : {false, true}) {
         for (uint32_t x : values) {
            for (uint32_t y : values) {
               SCOPED_TRACE(testing::Message()
                            << std::hex << x << " " << y << " max " << is_max
                            << " ftz " << ftz << " const " << constant_y
                            << " swap " << swap);
               uint32_t expected = reference(x, y, is_max, ftz);
               uint32_t got = lowered(x, y, is_max, ftz, constant_y, swap);
               if (is_nan(expected))
                  EXPECT_TRUE(is_nan(got));
               else
                  EXPECT_EQ(got, expected);
            }
         }
      }
   }
   }
   }
}

/* A tie with a nonzero constant is between identical bits: no tie select. */
TEST(FminmaxSignedZero, NonzeroConstantIsBare)
{
   for (bool is_max : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "constant minmax");
      nir_def *a = nir_fadd_imm(&b, agx_test_load_preamble(&b, 1, 32, 0), 1.0);
      b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
      nir_def *k = nir_imm_float(&b, is_max ? -1.0 : 1.0);
      nir_def *out = is_max ? nir_fmax(&b, a, k) : nir_fmin(&b, a, k);
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax_signed_zero(b.shader, true));
      EXPECT_EQ(count(b.shader, is_max ? nir_op_fmax_agx : nir_op_fmin_agx), 1u);
      EXPECT_EQ(count(b.shader, nir_op_bcsel), 0u);
      nir_validate_shader(b.shader, "constant minmax");
      ralloc_free(b.shader);
   }
}

/* Without signed-zero preservation the generic path is kept. */
TEST(FminmaxSignedZero, IgnoresNoSignedZero)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "nsz minmax");
   b.fp_math_ctrl = 0;
   nir_def *out = nir_fmin(&b, agx_test_load_preamble(&b, 1, 32, 0),
                           agx_test_load_preamble(&b, 1, 32, 2));
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   EXPECT_FALSE(agx_nir_lower_fminmax_signed_zero(b.shader, true));
   ralloc_free(b.shader);
}
