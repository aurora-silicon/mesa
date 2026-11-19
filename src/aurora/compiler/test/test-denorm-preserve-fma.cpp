/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static float
as_float(uint32_t x)
{
   float f;
   memcpy(&f, &x, sizeof(f));
   return f;
}

static uint32_t
as_bits(float f)
{
   uint32_t x;
   memcpy(&x, &f, sizeof(x));
   return x;
}

/* Lower one constant fp32 ffma for DenormPreserve and fold it. The native ffma
 * folds to the host's fused result, so the exact path is what is checked
 * wherever an operand, the product or the result is tiny. */
static uint32_t
fold_fma(uint32_t x, uint32_t y, uint32_t z)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "denorm preserve fma");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_DENORM_PRESERVE_FP32 |
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   nir_def *r = nir_ffma(&b, nir_imm_int(&b, x), nir_imm_int(&b, y),
                         nir_imm_int(&b, z));
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, r, nir_imm_int(&b, 0), nir_type_float32, FRAG_RESULT_DATA0);
   EXPECT_TRUE(agx_nir_lower_denorm_preserve(b.shader));

   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
      progress |= nir_opt_remove_phis(b.shader);
   } while (progress);
   nir_validate_shader(b.shader, "denorm preserve fma");

   uint32_t result = 0;
   EXPECT_TRUE(nir_src_is_const(store->src[0]));
   if (nir_src_is_const(store->src[0]))
      result = nir_src_as_uint(store->src[0]);
   ralloc_free(b.shader);
   return result;
}

static void
check(uint32_t x, uint32_t y, uint32_t z)
{
   const uint32_t expected =
      as_bits(std::fma(as_float(x), as_float(y), as_float(z)));
   const uint32_t got = fold_fma(x, y, z);
   if (std::isnan(as_float(expected)))
      EXPECT_TRUE(std::isnan(as_float(got)))
         << std::hex << x << " " << y << " " << z << " -> " << got;
   else
      EXPECT_EQ(got, expected) << std::hex << x << " " << y << " " << z;
}

TEST(DenormPreserveFMA, SpecialValues)
{
   static const uint32_t values[] = {
      0x00000000, /* 0 */
      0x00000001, /* smallest denormal */
      0x000ffe00, /* denormal */
      0x007fffff, /* largest denormal */
      0x00800000, /* smallest normal */
      0x00800001, /* smallest normal + 1 ulp */
      0x0c000000, /* 2^-103: products below 2^-126 */
      0x34000000, /* 2^-23 */
      0x3f7fffff, /* 1 - 1 ulp */
      0x3f800000, /* 1 */
      0x3f800001, /* 1 + 1 ulp */
      0x7f7fffff, /* largest normal */
      0x7f800000, /* infinity */
   };
   for (uint32_t x : values) {
      for (uint32_t y : values) {
         for (uint32_t z : values) {
            /* The sign of y only reaches the product through x's. */
            for (uint32_t signs = 0; signs < 4; signs++) {
               check(x ^ (signs & 1 ? 0x80000000u : 0), y,
                     z ^ (signs & 2 ? 0x80000000u : 0));
            }
         }
      }
   }
}

/* x * y - round(x * y) is the rounding error of the product, which only a
 * single rounding returns; separate multiply and add rounding returns zero.
 * These keep the native fma, so they check that ffma is not split. */
TEST(DenormPreserveFMA, StaysFused)
{
   static const float pairs[][2] = {
      {0x1.000002p0f, 0x1.000002p0f},
      {1.25f, 0x1.000004p0f},
      {3.0f, 0x1.555556p-2f},
   };
   for (const auto &p : pairs) {
      const float product = p[0] * p[1];
      check(as_bits(p[0]), as_bits(p[1]), as_bits(-product));
   }
   /* Correctly rounded with one rounding, but not with two. */
   check(as_bits(1.25f), as_bits(0x1.000004p0f), as_bits(0x1.000002p-126f));
}

/* A denormal operand takes the exact path. Each result differs from a
 * separately rounded product plus z. */
TEST(DenormPreserveFMA, ExactPathStaysFused)
{
   static const uint32_t cases[][3] = {
      {0x0075fbdeu, 0x4357ae2cu, 0x8446cd98u},
      {0x00704277u, 0x46e9c561u, 0x87cd061bu},
      {0x001c82cau, 0x4c357993u, 0x8c21b01bu},
      {0x000a022bu, 0x4e8fffbau, 0x8db426acu},
      {0x001364d9u, 0x48dbdf8bu, 0x888541a8u},
   };
   for (const auto &c : cases) {
      /* volatile: no contraction into an fma */
      volatile float product = as_float(c[0]) * as_float(c[1]);
      const float split = product + as_float(c[2]);
      EXPECT_NE(as_bits(split), as_bits(std::fma(as_float(c[0]), as_float(c[1]),
                                                 as_float(c[2]))));
      check(c[0], c[1], c[2]);
   }
}
