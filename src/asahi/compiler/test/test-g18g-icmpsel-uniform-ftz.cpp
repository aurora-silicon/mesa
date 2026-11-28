/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>
#include "agx_test.h"

extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

static agx_instr *
def_of(agx_context *s, agx_index value)
{
   agx_foreach_instr_global(s, I) {
      agx_foreach_dest(I, d) {
         if (agx_is_equiv(I->dest[d], value))
            return I;
      }
   }
   return NULL;
}

TEST(G18GIcmpsel, SignedHalfUniformComparandSignExtends)
{
   for (bool is_signed : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.emit_agx3d = true;
      b->shader->key = &key;

      /* r60 (32-bit) SLT/ULT u90 (16-bit) ? r62 : r63 */
      agx_icmpsel_to(b, agx_register(41, AGX_SIZE_32),
                     agx_register(60, AGX_SIZE_32),
                     agx_uniform(90, AGX_SIZE_16),
                     agx_register(62, AGX_SIZE_32),
                     agx_register(63, AGX_SIZE_32),
                     is_signed ? AGX_ICOND_SLT : AGX_ICOND_ULT);
      agx_stop(b);

      ASSERT_TRUE(agx_lower_agx3_g18g_half_icmpsel(b->shader));

      agx_instr *sel = NULL;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_ICMPSEL)
            sel = I;
      }
      ASSERT_NE(sel, nullptr);

      agx_instr *def = def_of(b->shader, sel->src[1]);
      ASSERT_NE(def, nullptr);
      if (is_signed) {
         ASSERT_EQ(def->op, AGX_OPCODE_SIGNEXT)
            << "signed select must sign-extend the uniform comparand";
         agx_instr *inner = def_of(b->shader, def->src[0]);
         ASSERT_NE(inner, nullptr);
         EXPECT_EQ(inner->op, AGX_OPCODE_MOV);
         EXPECT_TRUE(agx_is_equiv(inner->src[0],
                                  agx_uniform(90, AGX_SIZE_16)));
      } else {
         ASSERT_EQ(def->op, AGX_OPCODE_MOV)
            << "unsigned select must zero-extend the uniform comparand";
         EXPECT_TRUE(agx_is_equiv(def->src[0],
                                  agx_uniform(90, AGX_SIZE_16)));
      }
      ralloc_free(mem);
   }
}

TEST(G18GFtz, LoadsKeepBitsAndFloatOperationsFlush)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   &agx_nir_options,
                                                   "g18-ftz-scope");
   nir_def *offset = nir_imm_int(&b, 0);
   nir_def *input = nir_load_input(&b, 1, 32, offset);
   nir_io_semantics semantics = {};
   semantics.location = VARYING_SLOT_VAR0;
   nir_intrinsic_set_io_semantics(nir_def_as_intrinsic(input), semantics);
   nir_intrinsic_set_dest_type(nir_def_as_intrinsic(input), nir_type_float32);

   nir_def *raw = nir_load_ubo(&b, 1, 32, offset, offset);
   nir_intrinsic_set_range(nir_def_as_intrinsic(raw), 4);
   nir_intrinsic_set_align(nir_def_as_intrinsic(raw), 4, 0);
   nir_def *negative = nir_imm_int(&b, 0x80000001u);
   nir_def *positive = nir_imm_int(&b, 1);
   nir_def *values = nir_vec4(&b, input, nir_fneg(&b, raw), nir_fabs(&b, raw),
                              nir_fadd(&b, raw, raw));
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, values, offset, nir_type_float32, FRAG_RESULT_DATA0);

   agx_nir_lower_g18_ftz(b.shader);
   nir_validate_shader(b.shader, "g18-ftz-scope");

   nir_def_rewrite_uses(input, positive);
   nir_def_rewrite_uses(raw, negative);
   nir_opt_constant_folding(b.shader);

   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   const nir_const_value *out = nir_src_as_const_value(store->src[0]);
   EXPECT_EQ(out[0].u32, 1u);
   EXPECT_EQ(out[1].u32, 0u);
   EXPECT_EQ(out[2].u32, 0u);
   EXPECT_EQ(out[3].u32, 0x80000000u);
   ralloc_free(b.shader);
}

static bool
is_nir_flush(nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fcanonicalize ||
          (alu->op == nir_op_fadd && nir_src_is_const(alu->src[1].src) &&
           nir_src_comp_as_uint(alu->src[1].src, alu->src[1].swizzle[0]) ==
              0x80000000u);
}

static void
check_g18g_subgroup_fp32_ftz(nir_intrinsic_op intr_op, nir_op red_op,
                             unsigned mode, bool expect_lowered)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options,
                                                  "g18-ftz-subgroup");
   b.shader->info.float_controls_execution_mode = mode;
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *p = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *lane = nir_u2u64(&b, nir_ishl_imm(&b, nir_load_subgroup_invocation(&b), 2));
   nir_def *x = nir_load_global(&b, 1, 32, nir_iadd(&b, p, lane), .align_mul = 4);
   nir_def *r =
      intr_op == nir_intrinsic_reduce
         ? nir_reduce(&b, x, .reduction_op = red_op, .cluster_size = 0)
      : intr_op == nir_intrinsic_inclusive_scan
         ? nir_inclusive_scan(&b, x, .reduction_op = red_op)
         : nir_exclusive_scan(&b, x, .reduction_op = red_op);
   nir_store_global(&b, r, nir_iadd(&b, nir_iadd_imm(&b, p, 256), lane),
                    .align_mul = 4);

   agx_preprocess_nir_g18g(b.shader);
   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G18G);

   bool saw_native = false;
   bool saw_alu = false;
   bool saw_load_use = false;
   bool all_load_uses_flushed = true;

   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_reduce ||
                intr->intrinsic == nir_intrinsic_inclusive_scan ||
                intr->intrinsic == nir_intrinsic_exclusive_scan)
               saw_native = true;
         } else if (instr->type == nir_instr_type_alu) {
            if (nir_instr_as_alu(instr)->op == red_op)
               saw_alu = true;
         }
      }
   }

   nir_foreach_use_including_if(src, x) {
      saw_load_use = true;
      if (nir_src_is_if(src) || !is_nir_flush(nir_src_use_instr(src)))
         all_load_uses_flushed = false;
   }

   if (expect_lowered) {
      EXPECT_FALSE(saw_native)
         << "FTZ(32) must lower native fp32 subgroup op " << intr_op;
      EXPECT_TRUE(saw_alu)
         << "lowered fp32 subgroup op must emit ALU op " << red_op;
      EXPECT_TRUE(saw_load_use && all_load_uses_flushed)
         << "loaded input must be flushed before subgroup lowering";

      EXPECT_TRUE(agx_nir_lower_g18_ftz(b.shader));
      nir_validate_shader(b.shader, "g18-ftz-subgroup");
   } else {
      EXPECT_TRUE(saw_native)
         << "without FTZ(32), whole-wave fp32 subgroup op stays native";
   }

   ralloc_free(b.shader);
}

TEST(G18GFtz, SubgroupFloatReduceLowersUnderFtz32)
{
   for (nir_op op : {nir_op_fadd, nir_op_fmul}) {
      check_g18g_subgroup_fp32_ftz(nir_intrinsic_reduce, op,
                                   FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                                   true);
      check_g18g_subgroup_fp32_ftz(nir_intrinsic_reduce, op, 0, false);
   }
}

TEST(G18GFtz, SubgroupFloatScanLowersUnderFtz32)
{
   for (nir_intrinsic_op intr :
        {nir_intrinsic_inclusive_scan, nir_intrinsic_exclusive_scan}) {
      for (nir_op op : {nir_op_fadd, nir_op_fmul}) {
         check_g18g_subgroup_fp32_ftz(intr, op,
                                      FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32,
                                      true);
         check_g18g_subgroup_fp32_ftz(intr, op, 0, false);
      }
   }
}
