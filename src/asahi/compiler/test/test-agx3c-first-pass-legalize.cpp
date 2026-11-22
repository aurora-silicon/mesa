/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "util/u_dynarray.h"
}

/*
 * Verify that AGX3C (G17G / G18G) legalizes all previously-retried operand
 * shapes directly in the first compilation pass (without blind whole-function
 * recompile retries), while leaving natively encodable minifloat immediates
 * and tied min/max intact.
 */

namespace {

struct agx3c_target {
   const char *name;
   agx_chip chip;
   bool emit_agx3d;
};

static const agx3c_target kTargets[] = {
   {"M5", AGX_CHIP_G17C, false},
   {"G18G", AGX_CHIP_G18G, true},
};

static bool
legalize_ra_and_pack(agx_context *ctx)
{
   struct agx_shader_info info = {};
   ctx->out = &info;
   agx_lower_agx3c_unsupported_operands(ctx);
   agx_lower_uniform_sources(ctx);
   agx_dce(ctx, true);
   agx_ra(ctx);
   if (ctx->ra_failed)
      return false;
   agx_lower_pseudo(ctx);
   agx_opt_register_cache(ctx);

   /* UNIT_TEST pins liveness for DCE/RA; strip it before binary packing. */
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op == AGX_OPCODE_UNIT_TEST)
         agx_remove_instruction(I);
   }

   struct util_dynarray bin = UTIL_DYNARRAY_INIT;
   bool ok = agx3_pack_binary(ctx, &bin);
   if (!ok) {
      ADD_FAILURE() << "agx3_pack_binary refused: " << agx3_last_pack_refusal();
   }
   util_dynarray_fini(&bin);
   return ok;
}

static agx_builder *
make_agx3c_builder(void *mem, agx_shader_key *key, const agx3c_target &t)
{
   memset(key, 0, sizeof(*key));
   key->dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key->dev.chip = t.chip;
   key->dev.emit_agx3d = t.emit_agx3d;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = nir_shader_create(mem, MESA_SHADER_COMPUTE,
                                      &agx_nir_options);
   return b;
}

TEST(AGX3CFirstPassLegalize, UntiedLtnGtnFcmpselAndFcmp)
{
   for (const auto &t : kTargets) {
      for (enum agx_fcond cond : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key;
         agx_builder *b = make_agx3c_builder(mem, &key, t);

         agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_32));
         agx_index c = agx_preload(b, agx_register(2, AGX_SIZE_32));
         agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_32));
         agx_index y = agx_preload(b, agx_register(6, AGX_SIZE_32));
         agx_index dst = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *sel = agx_fcmpsel_to(b, dst, a, c, x, y, cond);

         agx_index cmp_dst = agx_temp(b->shader, AGX_SIZE_16);
         agx_instr *cmp = agx_fcmp_to(b, cmp_dst, a, c, cond, false);

         agx_unit_test(b, dst);
         agx_unit_test(b, cmp_dst);
         agx_stop(b);

         EXPECT_TRUE(agx_lower_agx3c_unsupported_operands(b->shader)) << t.name;
         EXPECT_EQ(sel->fcond,
                   cond == AGX_FCOND_LTN ? AGX_FCOND_GE : AGX_FCOND_LE)
            << t.name;
         EXPECT_TRUE(agx_is_equiv(sel->src[2], y)) << t.name;
         EXPECT_TRUE(agx_is_equiv(sel->src[3], x)) << t.name;
         EXPECT_EQ(cmp->fcond,
                   cond == AGX_FCOND_LTN ? AGX_FCOND_GE : AGX_FCOND_LE)
            << t.name;
         EXPECT_TRUE(cmp->invert_cond) << t.name;

         EXPECT_TRUE(legalize_ra_and_pack(b->shader)) << t.name;
         ralloc_free(mem);
      }
   }
}

TEST(AGX3CFirstPassLegalize, TiedMinMaxLtnGtnPreserved)
{
   for (const auto &t : kTargets) {
      for (enum agx_fcond cond : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key;
         agx_builder *b = make_agx3c_builder(mem, &key, t);

         agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_32));
         agx_index c = agx_preload(b, agx_register(2, AGX_SIZE_32));
         agx_index dst = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *mm = agx_fcmpsel_to(b, dst, a, c, a, c, cond);
         agx_unit_test(b, dst);
         agx_stop(b);

         EXPECT_FALSE(agx_lower_agx3c_unsupported_operands(b->shader))
            << t.name;
         EXPECT_EQ(mm->fcond, cond) << t.name;
         EXPECT_TRUE(legalize_ra_and_pack(b->shader)) << t.name;
         ralloc_free(mem);
      }
   }
}

TEST(AGX3CFirstPassLegalize, HalfAndSignedIcmpIcmpsel)
{
   for (const auto &t : kTargets) {
      for (enum agx_icond cond :
           {AGX_ICOND_UEQ, AGX_ICOND_SEQ, AGX_ICOND_ULT, AGX_ICOND_UGT,
            AGX_ICOND_SLT, AGX_ICOND_SGT}) {
         for (enum agx_size dst_sz : {AGX_SIZE_16, AGX_SIZE_32}) {
            void *mem = ralloc_context(NULL);
            agx_shader_key key;
            agx_builder *b = make_agx3c_builder(mem, &key, t);

            agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_16));
            agx_index c = agx_preload(b, agx_register(1, AGX_SIZE_16));
            agx_index d_icmp = agx_temp(b->shader, dst_sz);
            agx_icmp_to(b, d_icmp, a, c, cond, false);

            agx_index d_imm = agx_temp(b->shader, dst_sz);
            agx_icmp_to(b, d_imm, a, agx_immediate(5), cond, false);

            agx_index d_big = agx_temp(b->shader, dst_sz);
            agx_icmpsel_to(b, d_big, a, agx_abs(agx_immediate(300)),
                           agx_immediate(1), agx_abs(agx_immediate(400)), cond);

            agx_unit_test(b, d_icmp);
            agx_unit_test(b, d_imm);
            agx_unit_test(b, d_big);
            agx_stop(b);

            EXPECT_TRUE(legalize_ra_and_pack(b->shader))
               << t.name << " cond=" << cond << " dst=" << dst_sz;
            ralloc_free(mem);
         }
      }
   }
}

TEST(AGX3CFirstPassLegalize, UnsupportedFloatImmediatesAndNativeKept)
{
   for (const auto &t : kTargets) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key;
      agx_builder *b = make_agx3c_builder(mem, &key, t);

      agx_index r32 = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index h16 = agx_preload(b, agx_register(2, AGX_SIZE_16));

      /* Native minifloat add must stay inline (not materialized). */
      agx_index native_add_dst = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *native_add =
         agx_fadd_to(b, native_add_dst, r32, agx_immediate_f(0.5f));

      /* Modified minifloat immediate: folded into minifloat byte, stays imm. */
      agx_index neg_imm_dst = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *neg_imm =
         agx_fadd_to(b, neg_imm_dst, r32, agx_neg(agx_immediate_f(1.0f)));

      /* Unsupported: narrowing FMUL d16 = h16 * imm. */
      agx_index nar_mul_dst = agx_temp(b->shader, AGX_SIZE_16);
      agx_fmul_to(b, nar_mul_dst, h16, agx_immediate_f(0.5f));

      /* Unsupported: FFMA d32 = h16 * imm + imm. */
      agx_index fma_dst = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, fma_dst, h16, agx_immediate_f(0.5f),
                  agx_immediate_f(1.0f));

      /* Unsupported: two-immediate half FADD. */
      agx_index two_imm_h = agx_temp(b->shader, AGX_SIZE_16);
      agx_fadd_to(b, two_imm_h, agx_immediate_f(0.5f), agx_immediate_f(1.0f));

      /* Unsupported: negated f32 -> f16 conversion. */
      agx_index neg_cvt = agx_temp(b->shader, AGX_SIZE_16);
      agx_fadd_to(b, neg_cvt, agx_neg(r32), agx_negzero());

      /* Unsupported: narrow FADD of r32 + nonzero imm (G18G) and h16 + imm. */
      agx_index nar_r32_imm = agx_temp(b->shader, AGX_SIZE_16);
      agx_fadd_to(b, nar_r32_imm, r32, agx_immediate_f(0.5f));
      agx_index nar_h16_imm = agx_temp(b->shader, AGX_SIZE_16);
      agx_fadd_to(b, nar_h16_imm, h16, agx_immediate_f(0.5f));

      /* Unsupported: HMUL with neg, sat+imm, and HADD with sat+imm. */
      agx_index hmul_neg = agx_temp(b->shader, AGX_SIZE_16);
      agx_hmul_to(b, hmul_neg, agx_neg(h16), h16);
      agx_index hmul_sat_imm = agx_temp(b->shader, AGX_SIZE_16);
      agx_hmul_to(b, hmul_sat_imm, h16, agx_immediate_f(0.5f))->saturate = true;
      agx_index hadd_sat_imm = agx_temp(b->shader, AGX_SIZE_16);
      agx_hadd_to(b, hadd_sat_imm, h16, agx_immediate_f(0.5f))->saturate = true;

      /* Unsupported: IADD imm16.abs - r32. */
      agx_index iadd_sub = agx_temp(b->shader, AGX_SIZE_32);
      agx_iadd_to(b, iadd_sub, agx_abs(agx_immediate(1000)), agx_neg(r32), 0);

      agx_unit_test(b, native_add_dst);
      agx_unit_test(b, neg_imm_dst);
      agx_unit_test(b, nar_mul_dst);
      agx_unit_test(b, fma_dst);
      agx_unit_test(b, two_imm_h);
      agx_unit_test(b, neg_cvt);
      agx_unit_test(b, nar_r32_imm);
      agx_unit_test(b, nar_h16_imm);
      agx_unit_test(b, hmul_neg);
      agx_unit_test(b, hmul_sat_imm);
      agx_unit_test(b, hadd_sat_imm);
      agx_unit_test(b, iadd_sub);
      agx_stop(b);

      agx_lower_agx3c_unsupported_operands(b->shader);
      EXPECT_EQ(native_add->src[1].type, AGX_INDEX_IMMEDIATE) << t.name;
      EXPECT_EQ(neg_imm->src[1].type, AGX_INDEX_IMMEDIATE) << t.name;
      EXPECT_FALSE(neg_imm->src[1].neg) << t.name;
      EXPECT_EQ(neg_imm->src[1].value, agx_minifloat_encode(-1.0f)) << t.name;

      EXPECT_TRUE(legalize_ra_and_pack(b->shader)) << t.name;
      ralloc_free(mem);
   }
}

TEST(AGX3CFirstPassLegalize, MixedF16F32ComparesAndSelects)
{
   for (const auto &t : kTargets) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key;
      agx_builder *b = make_agx3c_builder(mem, &key, t);

      agx_index f32 = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index h16 = agx_preload(b, agx_register(2, AGX_SIZE_16));
      agx_index x32 = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index y32 = agx_preload(b, agx_register(6, AGX_SIZE_32));
      agx_index x16 = agx_preload(b, agx_register(8, AGX_SIZE_16));

      agx_index d_fcmp = agx_temp(b->shader, AGX_SIZE_16);
      agx_fcmp_to(b, d_fcmp, h16, f32, AGX_FCOND_LT, false);

      agx_index d_sel32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_fcmpsel_to(b, d_sel32, h16, f32, x32, y32, AGX_FCOND_GE);

      /* Mismatched selected reg width (half selected into 32-bit dest). */
      agx_index d_mixsel = agx_temp(b->shader, AGX_SIZE_32);
      agx_fcmpsel_to(b, d_mixsel, f32, agx_immediate_f(0.0f), x16, y32,
                     AGX_FCOND_GT);

      agx_unit_test(b, d_fcmp);
      agx_unit_test(b, d_sel32);
      agx_unit_test(b, d_mixsel);
      agx_stop(b);

      EXPECT_TRUE(legalize_ra_and_pack(b->shader)) << t.name;
      ralloc_free(mem);
   }
}

TEST(AGX3CFirstPassLegalize, FormerlyRetriedShapesNativeAndLegalized)
{
   for (const auto &t : kTargets) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key;
      agx_builder *b = make_agx3c_builder(mem, &key, t);
      /* Fragment stage so half DFDX/DFDY are valid. */
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

      agx_index r0 = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index r1 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index h0 = agx_preload(b, agx_register(4, AGX_SIZE_16));
      agx_index h1 = agx_preload(b, agx_register(5, AGX_SIZE_16));

      /* 1. 32-bit a & ~b (tt 0x2) and ~a & b (tt 0x4) pack natively. */
      agx_index andc_ab = agx_temp(b->shader, AGX_SIZE_32);
      agx_bitop_to(b, andc_ab, r0, r1, 0x2);
      agx_index andc_ba = agx_temp(b->shader, AGX_SIZE_32);
      agx_bitop_to(b, andc_ba, r0, r1, 0x4);

      /* 2. Half |a| * |b| packs natively via HMUL (and FMUL narrow). */
      agx_index hmul_abs = agx_temp(b->shader, AGX_SIZE_16);
      agx_hmul_to(b, hmul_abs, agx_abs(h0), agx_abs(h1));
      agx_index fmul_abs_h = agx_temp(b->shader, AGX_SIZE_16);
      agx_fmul_to(b, fmul_abs_h, agx_abs(h0), agx_abs(h1));

      /* 3. Half DFDX / DFDY pack natively via agx3_alu_derivative_m5. */
      agx_index d_dfdx = agx_temp(b->shader, AGX_SIZE_16);
      agx_dfdx_to(b, d_dfdx, h0);
      agx_index d_dfdy = agx_temp(b->shader, AGX_SIZE_16);
      agx_dfdy_to(b, d_dfdy, h1);

      /* 4. Unfused FCMP (WebRender box-shadow shape: plain, abs, neg, both
       * 16-bit and 32-bit destinations). */
      agx_index fcmp_16 = agx_temp(b->shader, AGX_SIZE_16);
      agx_fcmp_to(b, fcmp_16, r0, r1, AGX_FCOND_LT, false);
      agx_index fcmp_32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_fcmp_to(b, fcmp_32, r0, r1, AGX_FCOND_GE, true);
      agx_index fcmp_abs = agx_temp(b->shader, AGX_SIZE_16);
      agx_fcmp_to(b, fcmp_abs, agx_abs(r0), agx_neg(r1), AGX_FCOND_LE, false);

      /* 5. Mixed half/f32 ALU & FCMPSEL with 16-bit and 32-bit destinations. */
      agx_index mix_fadd_16 = agx_temp(b->shader, AGX_SIZE_16);
      agx_fadd_to(b, mix_fadd_16, r0, h0);
      agx_index mix_fadd_32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_fadd_to(b, mix_fadd_32, r0, h0);
      agx_index mix_fmul_16 = agx_temp(b->shader, AGX_SIZE_16);
      agx_fmul_to(b, mix_fmul_16, agx_abs(r0), h0);
      agx_index mix_ffma_16 = agx_temp(b->shader, AGX_SIZE_16);
      agx_ffma_to(b, mix_ffma_16, h0, r0, h1);
      agx_index mix_sel_16 = agx_temp(b->shader, AGX_SIZE_16);
      agx_fcmpsel_to(b, mix_sel_16, h0, h1, h0, h1, AGX_FCOND_LE);
      agx_index mod_sel_32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_fcmpsel_to(b, mod_sel_32, agx_abs(r0), r1, r0, r1, AGX_FCOND_LT);

      agx_unit_test(b, andc_ab);
      agx_unit_test(b, andc_ba);
      agx_unit_test(b, hmul_abs);
      agx_unit_test(b, fmul_abs_h);
      agx_unit_test(b, d_dfdx);
      agx_unit_test(b, d_dfdy);
      agx_unit_test(b, fcmp_16);
      agx_unit_test(b, fcmp_32);
      agx_unit_test(b, fcmp_abs);
      agx_unit_test(b, mix_fadd_16);
      agx_unit_test(b, mix_fadd_32);
      agx_unit_test(b, mix_fmul_16);
      agx_unit_test(b, mix_ffma_16);
      agx_unit_test(b, mix_sel_16);
      agx_unit_test(b, mod_sel_32);
      agx_stop(b);

      EXPECT_TRUE(legalize_ra_and_pack(b->shader)) << t.name;
      ralloc_free(mem);
   }
}

TEST(AGX3CFirstPassLegalize, RefusalIncludesSourceFilenameAndLine)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key;
   agx_builder *b = make_agx3c_builder(mem, &key, kTargets[0]);

   /* Emit a DFDX in a compute shader, which agx3_pack_alu.c refuses. */
   agx_index r0 = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index dst = agx_temp(b->shader, AGX_SIZE_32);
   agx_dfdx_to(b, dst, r0);
   agx_unit_test(b, dst);
   agx_stop(b);

   struct agx_shader_info info = {};
   b->shader->out = &info;
   agx_ra(b->shader);
   agx_lower_pseudo(b->shader);
   agx_foreach_instr_global_safe(b->shader, I) {
      if (I->op == AGX_OPCODE_UNIT_TEST)
         agx_remove_instruction(I);
   }
   struct util_dynarray bin = UTIL_DYNARRAY_INIT;
   EXPECT_FALSE(agx3_pack_binary(b->shader, &bin));
   const std::string refusal = agx3_last_pack_refusal();
   EXPECT_NE(refusal.find("agx3_pack_alu.c:"), std::string::npos)
      << "refusal was: " << refusal;
   util_dynarray_fini(&bin);
   ralloc_free(mem);
}

TEST(AGX3CFirstPassLegalize, EndToEndNirCompile)
{
   for (const auto &t : kTargets) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "agx3c_pass1");
      nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *f0 = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *f1 =
         nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4), .align_mul = 4);
      nir_def *h0 =
         nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 8), .align_mul = 2);
      nir_def *h1 =
         nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 10), .align_mul = 2);
      nir_def *s0 =
         nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 12), .align_mul = 2);
      nir_def *s1 =
         nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 14), .align_mul = 2);

      /* Unordered float compare (LTN/GTN), mixed f16/f32 compare, signed
       * 16-bit integer compare, and minifloat-select constant. */
      nir_def *unord = nir_inot(&b, nir_fge(&b, f0, f1));
      nir_def *r0 = nir_bcsel(&b, unord, nir_imm_float(&b, 1.0f), f0);

      nir_def *mixed_cmp = nir_flt(&b, nir_f2f32(&b, h0), f1);
      nir_def *r1 =
         nir_bcsel(&b, mixed_cmp, nir_fadd_imm(&b, f0, 0.5),
                   nir_fmul_imm(&b, f1, 2.0));

      nir_def *s16_lt = nir_ilt(&b, s0, s1);
      nir_def *s16_imm = nir_ilt_imm(&b, s0, -17);
      nir_def *r2 =
         nir_bcsel(&b, nir_ior(&b, s16_lt, s16_imm), nir_u2u32(&b, h1),
                   nir_imm_int(&b, 7));

      nir_store_global(&b, nir_vec3(&b, r0, r1, r2),
                       nir_iadd_imm(&b, addr, 32), .align_mul = 16);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;

      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << t.name;
      EXPECT_GT(part.info.main_size, 0u) << t.name;

      free(part.binary);
      ralloc_free(b.shader);
   }
}

} // namespace
