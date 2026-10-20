/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_nir.h"
}

static unsigned
fadds(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_alu &&
                 nir_instr_as_alu(instr)->op == nir_op_fadd;
         }
      }
   }
   return n;
}

static void
constants(bool is_max)
{
   const struct { uint32_t bits; bool flush; } cases[] = {
      {0, false}, {0x80000000u, false},
      {1, true}, {0x80000001u, true},
      {0x007fffffu, true}, {0x807fffffu, true},
      {0x00800000u, false}, {0x80800000u, false},
      {0x3f800000u, false}, {0xbf800000u, false},
      {0x7f800000u, false}, {0xff800000u, false},
      {0x7fc00000u, false}, {0x7f800001u, false},
   };
   for (const auto &test : cases) {
      const uint32_t bits = test.bits;
      SCOPED_TRACE(bits);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                      &agx_nir_options, "constant-ftz");
      nir_def *a = nir_imm_int(&b, bits);
      nir_def *other = nir_imm_float(&b, 1.0f);
      nir_def *out = is_max ? nir_fmax(&b, a, other) : nir_fmin(&b, a, other);
      nir_intrinsic_instr *store = nir_store_output(&b, out, nir_imm_int(&b, 0));
      nir_io_semantics semantics = {};
      semantics.location = FRAG_RESULT_DATA0;
      nir_intrinsic_set_io_semantics(store, semantics);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
      EXPECT_EQ(fadds(b.shader), unsigned(test.flush && !is_max));
      nir_validate_shader(b.shader, "constant-ftz");
      ralloc_free(b.shader);
   }
}
TEST(FminmaxConstantDenorm, DirectConstantMin) { constants(false); }
TEST(FminmaxConstantDenorm, DirectConstantMax) { constants(true); }

TEST(FminmaxConstantDenorm, NormalBoundsExcludeSubnormalResults)
{
   for (bool is_max : {false, true}) {
      for (unsigned side : {0u, 1u}) {
         for (uint32_t bits :
              {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
               0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u, 0x7f800000u,
               0xff800000u, 0x7fc00000u}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(is_max);
            SCOPED_TRACE(side);
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "normal bound ftz");
            nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
            nir_def *bound = nir_imm_int(&b, bits);
            nir_def *x = side == 0 ? bound : input;
            nir_def *y = side == 0 ? input : bound;
            nir_def *out = is_max ? nir_fmax(&b, x, y) : nir_fmin(&b, x, y);
            agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                                  FRAG_RESULT_DATA0);
            ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
            const unsigned exponent = (bits >> 23) & 255;
            const bool normal_bound =
               exponent && exponent != 255 && (bits >> 31) == !is_max;
            EXPECT_EQ(fadds(b.shader), unsigned(!normal_bound));
            nir_validate_shader(b.shader, "normal bound ftz");
            ralloc_free(b.shader);
         }
      }
   }
}

TEST(FminmaxConstantDenorm, SelectsPreservePossibleSubnormals)
{
   for (uint32_t x : {0u, 1u, 0x007fffffu, 0x00800000u, 0x3f800000u}) {
      for (uint32_t y : {0u, 0x80000001u, 0x80800000u, 0xbf800000u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "select ftz");
         nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
         nir_def *selected =
            nir_bcsel(&b, condition, nir_imm_int(&b, x), nir_imm_int(&b, y));
         nir_def *out = nir_fmin(&b, selected, nir_imm_float(&b, 1.0));
         agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                               FRAG_RESULT_DATA0);
         ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
         const bool denorm_x = !(x & 0x7f800000u) && (x & 0x007fffffu);
         const bool denorm_y = !(y & 0x7f800000u) && (y & 0x007fffffu);
         EXPECT_EQ(fadds(b.shader), unsigned(denorm_x || denorm_y));
         nir_validate_shader(b.shader, "select ftz");
         ralloc_free(b.shader);
      }
   }
}

TEST(FminmaxConstantDenorm, VectorLaneRequiresCanonicalization)
{
   for (bool is_max : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "vector ftz");
      nir_def *x = nir_imm_ivec2(&b, 0x3f800000, 1);
      nir_def *y = nir_imm_ivec2(&b, 0x3f800000, 2);
      nir_def *out = is_max ? nir_fmax(&b, x, y) : nir_fmin(&b, x, y);
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
      EXPECT_EQ(fadds(b.shader), 1u);
      nir_validate_shader(b.shader, "vector ftz");
      ralloc_free(b.shader);
   }
}

TEST(FminmaxConstantDenorm, PhiSourcesPreservePossibleSubnormals)
{
   for (uint32_t bits : {0u, 1u, 0x00800000u, 0xbf800000u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "phi ftz");
      nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
      nir_push_if(&b, condition);
      nir_def *x = nir_imm_int(&b, bits);
      nir_push_else(&b, NULL);
      nir_def *y = nir_imm_float(&b, 1.0);
      nir_pop_if(&b, NULL);
      nir_def *value = nir_if_phi(&b, x, y);
      nir_def *out = nir_fmin(&b, value, nir_imm_float(&b, 1.0));
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
      EXPECT_EQ(fadds(b.shader), unsigned(bits == 1));
      nir_validate_shader(b.shader, "phi ftz");
      ralloc_free(b.shader);
   }
}

TEST(FminmaxConstantDenorm, DeepSelectorsKeepConservativeFlush)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "bounded select ftz");
   nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
   nir_def *value = nir_imm_float(&b, 1.0);
   for (unsigned i = 0; i < 20; ++i)
      value = nir_bcsel(&b, condition, value, nir_imm_float(&b, 2.0));
   nir_def *out = nir_fmin(&b, value, nir_imm_float(&b, 4.0));
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
   EXPECT_EQ(fadds(b.shader), 1u);
   nir_validate_shader(b.shader, "bounded select ftz");
   ralloc_free(b.shader);
}

TEST(FminmaxConstantDenorm, MinmaxChainFlushesOnlyAtExit)
{
   /* Each intermediate result stays inside the min/max algebra, so only the
    * final result that leaves it gets the canonicalizing fadd.
    */
   for (bool is_max : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "chain ftz");
      nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
      nir_def *c = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *d = agx_test_load_preamble(&b, 1, 32, 8);
      nir_def *inner = is_max ? nir_fmax(&b, a, c) : nir_fmin(&b, a, c);
      nir_def *mid = is_max ? nir_fmax(&b, inner, d) : nir_fmin(&b, inner, d);
      nir_def *out = is_max ? nir_fmin(&b, mid, a) : nir_fmax(&b, mid, a);
      agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                            FRAG_RESULT_DATA0);
      ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
      EXPECT_EQ(fadds(b.shader), 1u);
      nir_validate_shader(b.shader, "chain ftz");
      ralloc_free(b.shader);
   }
}

TEST(FminmaxConstantDenorm, MinmaxChainThroughPhiFlushesOnce)
{
   /* Copies, selects and phis plumb the edge through to the next min/max
    * without observing it, so they do not defeat the elision.
    */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "chain phi ftz");
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *c = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *d = agx_test_load_preamble(&b, 1, 32, 8);
   nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
   nir_push_if(&b, condition);
   nir_def *then_min = nir_fmin(&b, a, c);
   nir_push_else(&b, NULL);
   nir_def *else_min = nir_fmax(&b, d, a);
   nir_pop_if(&b, NULL);
   nir_def *value = nir_if_phi(&b, then_min, else_min);
   nir_def *out = nir_fmin(&b, value, d);
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
   EXPECT_EQ(fadds(b.shader), 1u);
   nir_validate_shader(b.shader, "chain phi ftz");
   ralloc_free(b.shader);
}

TEST(FminmaxConstantDenorm, LeavingUseKeepsFlush)
{
   /* A result that leaves the min/max algebra still flushes; the downstream
    * min/max then sees the canonicalized edge.
    */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fanout ftz");
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *c = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *d = agx_test_load_preamble(&b, 1, 32, 8);
   nir_def *inner = nir_fmin(&b, a, c);
   agx_test_store_output(&b, inner, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   nir_def *out = nir_fmin(&b, inner, d);
   agx_test_store_output(&b, out, nir_imm_int(&b, 1), nir_type_float32,
                         FRAG_RESULT_DATA0);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
   EXPECT_EQ(fadds(b.shader), 2u);
   nir_validate_shader(b.shader, "fanout ftz");
   ralloc_free(b.shader);
}

TEST(FminmaxConstantDenorm, LoweredMinmaxConsumerKeepsFlush)
{
   /* An already-lowered fmin_agx decided canonicalization while our result
    * still claimed to flush; it cannot absorb the edge, so the flush stays.
    */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "lowered consumer ftz");
   nir_def *a = agx_test_load_preamble(&b, 1, 32, 0);
   nir_def *c = agx_test_load_preamble(&b, 1, 32, 4);
   nir_def *d = agx_test_load_preamble(&b, 1, 32, 8);
   nir_def *inner = nir_fmin(&b, a, c);
   nir_def *out = nir_fmin_agx(&b, inner, d);
   agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
   EXPECT_EQ(fadds(b.shader), 1u);
   nir_validate_shader(b.shader, "lowered consumer ftz");
   ralloc_free(b.shader);
}

TEST(FminmaxConstantDenorm, RawMinmaxDoesNotProveCanonicalization)
{
   for (bool phi : {false, true}) {
      for (bool inner_max : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "raw minmax ftz");
         nir_def *input = agx_test_load_preamble(&b, 1, 32, 0);
         nir_def *bound = nir_imm_float(&b, inner_max ? -1.0 : 1.0);
         nir_def *raw = inner_max ? nir_fmax_agx(&b, input, bound)
                                  : nir_fmin_agx(&b, input, bound);
         nir_def *condition = nir_ieq_imm(&b, nir_load_sample_id(&b), 0);
         nir_def *value;
         if (phi) {
            nir_push_if(&b, condition);
            nir_def *x = nir_mov(&b, raw);
            nir_push_else(&b, NULL);
            nir_def *y = nir_imm_float(&b, 0.0);
            nir_pop_if(&b, NULL);
            value = nir_if_phi(&b, x, y);
         } else {
            value = nir_bcsel(&b, condition, raw, nir_imm_float(&b, 0.0));
         }
         nir_def *out =
            inner_max ? nir_fmax(&b, value, bound) : nir_fmin(&b, value, bound);
         agx_test_store_output(&b, out, nir_imm_int(&b, 0), nir_type_float32,
                               FRAG_RESULT_DATA0);
         ASSERT_TRUE(agx_nir_lower_fminmax(b.shader));
         EXPECT_EQ(fadds(b.shader), 1u);
         nir_validate_shader(b.shader, "raw minmax ftz");
         ralloc_free(b.shader);
      }
   }
}
