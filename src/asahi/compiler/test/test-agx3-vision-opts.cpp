/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <cmath>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
#include <functional>
#include <unordered_map>
#include <vector>
#include "util/format/u_format.h"
extern "C" {
#include "agx_nir.h"
#include "agx_compiler.h"
}

static nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static nir_def *
make_buffer_addr(nir_builder *b, nir_def *base_u64, nir_def *byte_offset)
{
   return nir_iadd(b, base_u64, nir_u2u64(b, byte_offset));
}

static nir_def *
make_desc_base(nir_builder *b, int preamble_base)
{
   nir_def *set = nir_load_preamble(b, 1, 64, .base = preamble_base);
   nir_def *desc = nir_load_global_constant(
      b, 4, 32, set,
      .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                      ACCESS_BUFFER_BINDING_AGX),
      .align_mul = 16);
   return nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                 nir_channel(b, desc, 1));
}

static unsigned
count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_alu_op(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(block->cf_node.parent);
            if (nir_if_first_then_block(nif) == block)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_loops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)) == block)
            n++;
      }
   }
   return n;
}

/* --- Commit 1: agx_nir_merge_global_words --- */

TEST(AGX3VisionOpts, MergeGlobalWordsNarrowU8AndSignedDuplicates)
{
   nir_builder b = make_compute_shader("merge-u8-signed");
   nir_def *base = make_desc_base(&b, 4);
   nir_def *idx = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *out = make_desc_base(&b, 8);

   /* Signed offsets {-4, 0, +4, +4} of 32-bit loads -> single vec3 load + CSE. */
   nir_def *l0 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, -4)), .align_mul = 4);
   nir_def *l1 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, idx), .align_mul = 4);
   nir_def *l2 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, 4)), .align_mul = 4);
   nir_def *l3 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, 4)), .align_mul = 4);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, l0, l1), nir_iadd(&b, l2, l3));
   nir_store_global(&b, sum, out, .align_mul = 4);

   ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 4u);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge signed duplicates");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, MergeGlobalWordsU8LoopFreeUnpackGate)
{
   for (bool free_unpack : {true, false}) {
      nir_builder b = make_compute_shader(free_unpack ? "merge-u8-free" : "merge-u8-nonfree");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc = nir_local_variable_create(b.impl, free_unpack ? glsl_float_type() : glsl_uint16_t_type(), "acc");
      nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc, free_unpack ? nir_imm_float(&b, 0.0f) : nir_imm_intN_t(&b, 0, 16), 1);

      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, iv);
         nir_break_if(&b, nir_ige(&b, i, n));
         nir_def *cur = nir_load_var(&b, acc);
         for (unsigned k = 0; k < 4; ++k) {
            nir_def *v = nir_load_global(
               &b, 1, 8, make_buffer_addr(&b, base, nir_iadd_imm(&b, i, k)),
               .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
               .align_mul = 1);
            if (free_unpack)
               cur = nir_fadd(&b, cur, nir_u2f32(&b, v));
            else
               cur = nir_iadd(&b, cur, nir_ishl_imm(&b, nir_u2u16(&b, v), 1));
         }
         nir_store_var(&b, acc, cur, 1);
         nir_store_var(&b, iv, nir_iadd_imm(&b, i, 4), 1);
      }
      nir_pop_loop(&b, loop);
      nir_def *res = nir_load_var(&b, acc);
      nir_store_global(&b, free_unpack ? res : nir_u2u32(&b, res), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 4u);

      bool changed = agx_nir_merge_global_words(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "merge u8 loop gate");
      EXPECT_EQ(changed, free_unpack);
      EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global),
                free_unpack ? 1u : 4u);
      ralloc_free(b.shader);
   }
}

/* --- Commit 2: agx_nir_reassoc_const_offsets, u8_absdiff_fp, unroll_and_jam --- */

TEST(AGX3VisionOpts, ReassocConstOffsetsRespectsSingleUse)
{
   for (bool multi_use : {false, true}) {
      nir_builder b = make_compute_shader(multi_use ? "reassoc-multi" : "reassoc-single");
      nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *c = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *out = make_desc_base(&b, 4);

      nir_def *ap4 = nir_iadd_imm(&b, a, 4);
      nir_def *cp8 = nir_iadd_imm(&b, c, 8);
      nir_def *sum = nir_iadd(&b, ap4, cp8);
      nir_store_global(&b, sum, out, .align_mul = 4);
      if (multi_use) {
         nir_store_global(&b, ap4, make_buffer_addr(&b, out, nir_imm_int(&b, 4)), .align_mul = 4);
         nir_store_global(&b, cp8, make_buffer_addr(&b, out, nir_imm_int(&b, 8)), .align_mul = 4);
      }

      bool changed = agx_nir_reassoc_const_offsets(b.shader);
      nir_validate_shader(b.shader, "reassoc");
      EXPECT_EQ(changed, !multi_use);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, U8AbsdiffFpTransformsReusedAndIsBitExact)
{
   for (bool reused : {true, false}) {
      nir_builder b = make_compute_shader(reused ? "absdiff-reused" : "absdiff-unique");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);

      nir_def *a[4], *c[4];
      for (unsigned i = 0; i < 4; ++i) {
         a[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)), .align_mul = 1);
         c[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, 16 + i)), .align_mul = 1);
      }

      /* Chain 1: sum_i |a[i] - c[i]|. Chain 2 (when reused): sum_i |a[i] - c[(i+1)&3]|. */
      nir_def *acc1 = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < 4; ++i) {
         nir_def *d = nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[i]));
         acc1 = nir_iadd(&b, acc1, nir_iabs(&b, d));
      }
      nir_store_global(&b, acc1, out, .align_mul = 4);

      if (reused) {
         nir_def *acc2 = nir_imm_int(&b, 0);
         for (unsigned i = 0; i < 4; ++i) {
            nir_def *d = nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[(i + 1) & 3]));
            acc2 = nir_iadd(&b, acc2, nir_iabs(&b, d));
         }
         nir_store_global(&b, acc2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)), .align_mul = 4);
      }

      bool changed = agx_nir_opt_u8_absdiff_fp(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8 absdiff fp");
      EXPECT_EQ(changed, reused);
      if (reused) {
         EXPECT_GT(count_alu_op(b.shader, nir_op_u2f32), 0u);
         EXPECT_GT(count_alu_op(b.shader, nir_op_f2u32), 0u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_iabs), 0u);
      }
      ralloc_free(b.shader);
   }

   /* Run the actual pass on known u8 values, then replace loads with constants
    * and fold with nir_opt_constant_folding to verify the rewritten NIR produces
    * the exact integer sum. */
   {
      nir_builder b = make_compute_shader("absdiff-eval");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      const uint8_t av[4] = {0, 255, 13, 200};
      const uint8_t cv[4] = {255, 1, 200, 13};
      nir_def *a[4], *c[4];
      for (unsigned i = 0; i < 4; ++i) {
         a[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)), .align_mul = 1);
         c[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, 16 + i)), .align_mul = 1);
      }
      nir_def *acc1 = nir_imm_int(&b, 0);
      nir_def *acc2 = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < 4; ++i) {
         acc1 = nir_iadd(&b, acc1, nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[i]))));
         acc2 = nir_iadd(&b, acc2, nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[(i + 1) & 3]))));
      }
      nir_intrinsic_instr *st1 = nir_store_global(&b, acc1, out, .align_mul = 4);
      nir_intrinsic_instr *st2 = nir_store_global(&b, acc2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)), .align_mul = 4);

      ASSERT_TRUE(agx_nir_opt_u8_absdiff_fp(b.shader));
      nir_validate_shader(b.shader, "u8 absdiff fp eval");
      /* Substitute constant u8 values into the rewritten NIR and constant-fold. */
      b.cursor = nir_before_impl(b.impl);
      for (unsigned i = 0; i < 4; ++i) {
         nir_def_replace(a[i], nir_imm_intN_t(&b, av[i], 8));
         nir_def_replace(c[i], nir_imm_intN_t(&b, cv[i], 8));
      }
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      uint32_t exp1 = 0, exp2 = 0;
      for (unsigned i = 0; i < 4; ++i) {
         exp1 += (uint32_t)std::abs((int)av[i] - (int)cv[i]);
         exp2 += (uint32_t)std::abs((int)av[i] - (int)cv[(i + 1) & 3]);
      }
      ASSERT_TRUE(nir_src_is_const(st1->src[0]));
      ASSERT_TRUE(nir_src_is_const(st2->src[0]));
      EXPECT_EQ(nir_src_as_uint(st1->src[0]), exp1);
      EXPECT_EQ(nir_src_as_uint(st2->src[0]), exp2);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, UnrollAndJam2DNestedLoopAndBarrierRejection)
{
   for (bool with_barrier : {false, true}) {
      nir_builder b = make_compute_shader(with_barrier ? "jam-barrier" : "jam-2d");
      nir_def *left_base = make_desc_base(&b, 4);
      nir_def *right_base = make_desc_base(&b, 8);
      nir_def *out_base = make_desc_base(&b, 12);
      nir_def *max_d = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *width = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
      nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

      nir_variable *dv = nir_local_variable_create(b.impl, glsl_int_type(), "d");
      nir_variable *best_cost = nir_local_variable_create(b.impl, glsl_uint_type(), "best_cost");
      nir_variable *best_d = nir_local_variable_create(b.impl, glsl_int_type(), "best_d");
      nir_store_var(&b, dv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, best_cost, nir_imm_int(&b, 0xffffffffu), 1);
      nir_store_var(&b, best_d, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *d = nir_load_var(&b, dv);

         if (with_barrier)
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP);

         nir_variable *kv = nir_local_variable_create(b.impl, glsl_int_type(), "k");
         nir_variable *sad = nir_local_variable_create(b.impl, glsl_uint_type(), "sad");
         nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, sad, nir_imm_int(&b, 0), 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *k = nir_load_var(&b, kv);
            nir_break_if(&b, nir_ige(&b, k, width));
            nir_def *l = nir_load_global(
               &b, 1, 8, make_buffer_addr(&b, left_base, nir_iadd(&b, x, k)),
               .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
               .align_mul = 1);
            nir_def *r = nir_load_global(
               &b, 1, 8, make_buffer_addr(&b, right_base, nir_isub(&b, nir_iadd(&b, x, k), d)),
               .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
               .align_mul = 1);
            nir_def *diff = nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, l), nir_u2u32(&b, r)));
            nir_store_var(&b, sad, nir_iadd(&b, nir_load_var(&b, sad), diff), 1);
            nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
         }
         nir_pop_loop(&b, inner);

         nir_def *cost = nir_load_var(&b, sad);
         nir_def *cur_best = nir_load_var(&b, best_cost);
         nir_def *better = nir_ult(&b, cost, cur_best);
         nir_store_var(&b, best_cost, nir_bcsel(&b, better, cost, cur_best), 1);
         nir_store_var(&b, best_d, nir_bcsel(&b, better, d, nir_load_var(&b, best_d)), 1);
         nir_def *next_d = nir_iadd_imm(&b, d, 1);
         nir_store_var(&b, dv, next_d, 1);
         nir_break_if(&b, nir_ige(&b, next_d, max_d));
      }
      nir_pop_loop(&b, outer);

      nir_store_global(&b, nir_load_var(&b, best_d), out_base, .align_mul = 4);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_cse);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_EQ(count_loops(b.shader), 2u);

      unsigned widest = 0;
      bool jammed = agx_nir_unroll_and_jam(b.shader, 8, &widest);
      nir_validate_shader(b.shader, "unroll and jam");
      EXPECT_EQ(jammed, !with_barrier);
      if (!with_barrier) {
         EXPECT_GT(count_loops(b.shader), 2u);
         EXPECT_GE(widest, 2u);
      }
      ralloc_free(b.shader);
   }
}

/* --- Commit 3: guard_selected_loads, forward_merge_loads, fuse_selects --- */

TEST(AGX3VisionOpts, GuardSelectedLoadsWrapsReadonlyAndRefusesWritable)
{
   for (bool readonly : {true, false}) {
      nir_builder b = make_compute_shader(readonly ? "guard-ro" : "guard-rw");
      nir_def *base_ptr = make_desc_base(&b, 4);
      nir_def *out_ptr = make_desc_base(&b, 8);
      nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *center = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);

      enum gl_access_qualifier acc =
         readonly ? (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER)
                  : (enum gl_access_qualifier)0;
      nir_def *n0 = nir_load_agx(&b, 1, 32, base_ptr, nir_iadd_imm(&b, idx, -1),
                                 .format = PIPE_FORMAT_R32_UINT, .access = acc);
      nir_def *n1 = nir_load_agx(&b, 1, 32, base_ptr, nir_iadd_imm(&b, idx, 1),
                                 .format = PIPE_FORMAT_R32_UINT, .access = acc);
      nir_def *c0 = nir_uge(&b, center, n0);
      nir_def *c1 = nir_uge(&b, center, n1);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *res = nir_bcsel(&b, c1, nir_bcsel(&b, c0, center, zero), zero);
      nir_store_agx(&b, res, out_ptr, idx, .format = PIPE_FORMAT_R32_UINT);

      ASSERT_EQ(count_ifs(b.shader), 0u);
      bool changed = agx_nir_opt_guard_selected_loads(b.shader);
      nir_validate_shader(b.shader, "guard selected loads");
      EXPECT_EQ(changed, readonly);
      EXPECT_EQ(count_ifs(b.shader), readonly ? 1u : 0u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, ForwardMergeLoadsEliminatesRoundTrip)
{
   nir_builder b = make_compute_shader("forward-merge");
   nir_def *ptr = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *cond = nir_ine_imm(&b, idx, 0);
   nir_def *va = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *vb = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

   nir_push_if(&b, cond);
   nir_store_agx(&b, va, ptr, idx, .format = PIPE_FORMAT_R32_UINT);
   nir_push_else(&b, NULL);
   nir_store_agx(&b, vb, ptr, idx, .format = PIPE_FORMAT_R32_UINT);
   nir_pop_if(&b, NULL);

   nir_def *reread = nir_load_agx(&b, 1, 32, ptr, idx, .format = PIPE_FORMAT_R32_UINT);
   nir_store_agx(&b, reread, out, idx, .format = PIPE_FORMAT_R32_UINT);

   ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
   EXPECT_TRUE(agx_nir_opt_forward_merge_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "forward merge loads");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, FuseSelectsCollapsesSharedPredicateMultiUse)
{
   nir_builder b = make_compute_shader("fuse-selects-shared");
   nir_def *out = make_desc_base(&b, 4);
   nir_def *p0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *p1 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_def *v0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
   nir_def *v1 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 32);
   nir_def *v2 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 16), .range = 32);

   nir_def *a = nir_ine_imm(&b, p0, 0);
   nir_def *c = nir_ine_imm(&b, p1, 0);
   nir_def *both = nir_ior(&b, a, c);
   /* Two selects both collapse `both` so `ior` becomes completely dead. */
   nir_def *r0 = nir_bcsel(&b, both, nir_bcsel(&b, a, v0, v1), v2);
   nir_def *r1 = nir_bcsel(&b, both, nir_bcsel(&b, a, v1, v2), v0);
   nir_store_global(&b, nir_iadd(&b, r0, r1), out, .align_mul = 4);

   ASSERT_EQ(count_alu_op(b.shader, nir_op_ior), 1u);
   EXPECT_TRUE(agx_nir_fuse_selects(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "fuse selects shared");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_ior), 0u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, MergeGlobalWordsSkipsRobustBoundedNegativeOffset)
{
   /* Soft-fault robust 32-bit storage loads with negative lowest offset
    * {X-4, X, X+4}: at X = 0, X-4 wraps to 0xFFFFFFFC so merging into one
    * vec3 load at base + u2u64(X-4) would read channel 1 from base + 2^32. */
   for (bool is_u8 : {false, true}) {
      nir_builder b = make_compute_shader(is_u8 ? "robust-neg-u8" : "robust-neg-u32");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *idx = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *bound = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      const unsigned bits = is_u8 ? 8 : 32;
      const unsigned stride = bits / 8;

      nir_def *vals[3];
      for (int k = -1; k <= 1; ++k) {
         nir_def *off = k == 0 ? idx : nir_iadd_imm(&b, idx, (uint32_t)(k * (int)stride));
         nir_def *ld = nir_load_global(&b, 1, bits, make_buffer_addr(&b, base, off),
                                       .align_mul = stride);
         nir_def *end = nir_uadd_sat(&b, off, nir_imm_int(&b, stride));
         if (is_u8) {
            vals[k + 1] = nir_u2u32(&b, nir_bcsel(&b, nir_uge(&b, bound, end), ld,
                                                  nir_imm_intN_t(&b, 0, 8)));
         } else {
            vals[k + 1] = nir_bounds_agx(&b, ld, end, bound);
         }
      }
      nir_def *sum = nir_iadd(&b, nir_iadd(&b, vals[0], vals[1]), vals[2]);
      nir_store_global(&b, sum, out, .align_mul = 4);

      ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 3u);
      EXPECT_FALSE(agx_nir_merge_global_words(b.shader));
      nir_validate_shader(b.shader, "merge robust negative offset");
      EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 3u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, UnrollAndJamRejectsOuterReductionUsedInInnerLoop)
{
   nir_builder b = make_compute_shader("jam-red-in-inner");
   nir_def *left_base = make_desc_base(&b, 4);
   nir_def *right_base = make_desc_base(&b, 8);
   nir_def *out_base = make_desc_base(&b, 12);
   nir_def *max_d = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *width = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_variable *dv = nir_local_variable_create(b.impl, glsl_int_type(), "d");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, dv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 7), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *d = nir_load_var(&b, dv);
      nir_def *cur_acc = nir_load_var(&b, acc);

      nir_variable *kv = nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_variable *sad = nir_local_variable_create(b.impl, glsl_uint_type(), "sad");
      nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sad, nir_imm_int(&b, 0), 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, kv);
         nir_break_if(&b, nir_ige(&b, k, width));
         nir_def *l = nir_load_global(
            &b, 1, 8, make_buffer_addr(&b, left_base, nir_iadd(&b, x, k)),
            .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
            .align_mul = 1);
         nir_def *r = nir_load_global(
            &b, 1, 8, make_buffer_addr(&b, right_base, nir_isub(&b, nir_iadd(&b, x, k), d)),
            .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
            .align_mul = 1);
         /* Cross-lane dependency: inner loop reads outer reduction phi cur_acc. */
         nir_def *term = nir_iadd(&b, nir_u2u32(&b, l), nir_iadd(&b, nir_u2u32(&b, r), cur_acc));
         nir_store_var(&b, sad, nir_iadd(&b, nir_load_var(&b, sad), term), 1);
         nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
      }
      nir_pop_loop(&b, inner);

      nir_store_var(&b, acc, nir_iadd(&b, cur_acc, nir_load_var(&b, sad)), 1);
      nir_def *next_d = nir_iadd_imm(&b, d, 1);
      nir_store_var(&b, dv, next_d, 1);
      nir_break_if(&b, nir_ige(&b, next_d, max_d));
   }
   nir_pop_loop(&b, outer);

   nir_store_global(&b, nir_load_var(&b, acc), out_base, .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);

   unsigned widest = 0;
   EXPECT_FALSE(agx_nir_unroll_and_jam(b.shader, 8, &widest));
   nir_validate_shader(b.shader, "unroll and jam reject inner red");
   EXPECT_EQ(count_loops(b.shader), 2u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, ForwardMergeLoadsChecksSignExtend)
{
   nir_builder b = make_compute_shader("forward-merge-sext");
   nir_def *ptr = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *cond = nir_ine_imm(&b, idx, 0);
   nir_def *va = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *vb = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

   nir_push_if(&b, cond);
   nir_store_agx(&b, va, ptr, idx, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
   nir_push_else(&b, NULL);
   nir_store_agx(&b, vb, ptr, idx, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
   nir_pop_if(&b, NULL);

   /* Load uses sign_extend=true while stores use sign_extend=false -> must not forward. */
   nir_def *reread = nir_load_agx(&b, 1, 32, ptr, idx, .format = PIPE_FORMAT_R32_UINT,
                                  .sign_extend = true);
   nir_store_agx(&b, reread, out, idx, .format = PIPE_FORMAT_R32_UINT);

   EXPECT_FALSE(agx_nir_opt_forward_merge_loads(b.shader));
   nir_validate_shader(b.shader, "forward merge sign_extend");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, NarrowU8FaddTreeTransformsAndEvaluatesExact)
{
   nir_builder b = make_compute_shader("narrow-u8-fadd");
   nir_def *base = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   const uint8_t vals[4] = {10, 200, 55, 255};
   nir_def *lds[4];
   nir_def *fsum = nir_imm_float(&b, 0.0f);
   for (unsigned i = 0; i < 4; ++i) {
      lds[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)),
                               .align_mul = 1);
      nir_def *f = nir_u2f32(&b, lds[i]);
      fsum = (i == 0) ? f : nir_fadd(&b, fsum, f);
   }
   nir_intrinsic_instr *st = nir_store_global(&b, fsum, out, .align_mul = 4);

   ASSERT_EQ(count_alu_op(b.shader, nir_op_u2f32), 4u);
   ASSERT_EQ(count_alu_op(b.shader, nir_op_fadd), 3u);
   EXPECT_TRUE(agx_nir_opt_narrow_u8_fadd_tree(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "narrow u8 fadd tree");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_u2f32), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_i2f32), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_fadd), 0u);

   b.cursor = nir_before_impl(b.impl);
   for (unsigned i = 0; i < 4; ++i)
      nir_def_replace(lds[i], nir_imm_intN_t(&b, vals[i], 8));
   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   NIR_PASS(_, b.shader, nir_opt_dce);
   ASSERT_TRUE(nir_src_is_const(st->src[0]));
   EXPECT_FLOAT_EQ(nir_src_as_float(st->src[0]), 10.0f + 200.0f + 55.0f + 255.0f);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, FoldConstSelectAluEvaluatesExact)
{
   for (int cond_val : {0, 1}) {
      nir_builder b = make_compute_shader("fold-const-sel");
      nir_def *out = make_desc_base(&b, 4);
      nir_def *p = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *cond = nir_ine_imm(&b, p, 0);
      nir_def *sel = nir_bcsel(&b, cond, nir_imm_int(&b, 10), nir_imm_int(&b, 20));
      nir_def *res = nir_iadd(&b, sel, nir_imm_int(&b, 7));
      nir_intrinsic_instr *st = nir_store_global(&b, res, out, .align_mul = 4);

      ASSERT_EQ(count_alu_op(b.shader, nir_op_iadd), 1u);
      EXPECT_TRUE(agx_nir_fold_const_select_alu(b.shader));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "fold const select alu");
      EXPECT_EQ(count_alu_op(b.shader, nir_op_iadd), 0u);

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(p, nir_imm_int(&b, cond_val));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_TRUE(nir_src_is_const(st->src[0]));
      EXPECT_EQ(nir_src_as_uint(st->src[0]), cond_val ? 17u : 27u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, FuseSelectsEvaluatesExactOnAllBooleanInputs)
{
   for (unsigned mask = 0; mask < 4; ++mask) {
      const bool av = mask & 1;
      const bool cv = (mask >> 1) & 1;
      nir_builder b = make_compute_shader("fuse-selects-eval");
      nir_def *out = make_desc_base(&b, 4);
      nir_def *pa = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *pc = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *a = nir_ine_imm(&b, pa, 0);
      nir_def *c = nir_ine_imm(&b, pc, 0);
      nir_def *r1 = nir_bcsel(&b, nir_ior(&b, a, c),
                              nir_bcsel(&b, a, nir_imm_int(&b, 11), nir_imm_int(&b, 22)),
                              nir_imm_int(&b, 33));
      nir_def *r2 = nir_bcsel(&b, a,
                              nir_bcsel(&b, a, nir_imm_int(&b, 44), nir_imm_int(&b, 55)),
                              nir_imm_int(&b, 66));
      nir_intrinsic_instr *st1 = nir_store_global(&b, r1, out, .align_mul = 4);
      nir_intrinsic_instr *st2 = nir_store_global(&b, r2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)), .align_mul = 4);

      EXPECT_TRUE(agx_nir_fuse_selects(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "fuse selects eval");

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(pa, nir_imm_int(&b, av ? 1 : 0));
      nir_def_replace(pc, nir_imm_int(&b, cv ? 1 : 0));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_TRUE(nir_src_is_const(st1->src[0]));
      ASSERT_TRUE(nir_src_is_const(st2->src[0]));
      EXPECT_EQ(nir_src_as_uint(st1->src[0]), av ? 11u : (cv ? 22u : 33u));
      EXPECT_EQ(nir_src_as_uint(st2->src[0]), av ? 44u : 66u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchExactThreeAndFour)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      /* mode 0: rad=2 (trip == 3, straight-line 3x, 0 inner loops)
       * mode 1: rad=4 (trip == 5, straight-line 4x + 1-iter remainder)
       * mode 2: rad=0 (trip == 1, scalar fallback)
       * mode 3: x0=INT32_MAX-2, bound=INT32_MAX (REL_LE wrap guard rejects exact-3) */
      nir_builder b = make_compute_shader("small-window-unswitch");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xhi = nir_iadd(&b, x0, rad);
      nir_def *yhi = nir_iadd(&b, y0, rad);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      for (unsigned r = 0; r < 4; ++r) {
         NIR_PASS(_, b.shader, nir_opt_copy_prop);
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_algebraic);
         NIR_PASS(_, b.shader, nir_opt_dead_cf);
         NIR_PASS(_, b.shader, nir_opt_dce);
      }
      nir_validate_shader(b.shader, "small window unswitch");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(rad, nir_imm_int(&b, 2));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(rad, nir_imm_int(&b, 4));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(rad, nir_imm_int(&b, 0));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(rad, nir_imm_int(&b, 2));
         nir_def_replace(x0, nir_imm_int(&b, INT32_MAX - 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      for (unsigned r = 0; r < 6; ++r) {
         NIR_PASS(_, b.shader, nir_opt_copy_prop);
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_algebraic);
         NIR_PASS(_, b.shader, nir_opt_dead_cf);
         NIR_PASS(_, b.shader, nir_opt_dce);
      }
      nir_validate_shader(b.shader, "small window unswitch specialized");

      /* Compute maximum loop nesting depth and loop count after specialization. */
      unsigned loops = 0, max_depth = 0;
      nir_foreach_block(block, b.impl) {
         unsigned depth = 0;
         for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
            if (n->type == nir_cf_node_loop)
               depth++;
         if (depth > max_depth)
            max_depth = depth;
         if (block->cf_node.parent->type == nir_cf_node_loop &&
             block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
            loops++;
      }

      if (mode == 0) {
         /* Exact 3x3 window: both inner and outer loops are completely peeled
          * into 9 straight-line loads with zero remaining loops. */
         EXPECT_EQ(max_depth, 0u);
         EXPECT_EQ(loops, 0u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 9u);
      } else if (mode == 1) {
         /* 5-tap: 4 straight-line loads + 1 in compact remainder loop. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
      } else if (mode == 2) {
         /* 1-tap: scalar fallback outer + inner loop. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      } else {
         /* bound == INT32_MAX: exact-3 no-wrap guard rejects straight-line removal. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchSkipsWide32BitLoop)
{
   nir_builder b = make_compute_shader("small-window-skip-32bit");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
   nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
   nir_def *xhi = nir_iadd(&b, x0, rad);
   nir_def *yhi = nir_iadd(&b, y0, rad);

   nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, y_var, y0, 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, y_var);
      nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, yex);

      nir_def *row_off = nir_imul(&b, y, stride);
      nir_store_var(&b, x_var, x0, 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xex);

         nir_def *idx = nir_iadd(&b, row_off, x);
         nir_def *v = nir_load_agx(&b, 1, 32, base, idx,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .format = PIPE_FORMAT_R32_UINT);
         nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), v), 1);
         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, inner);
      nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
   }
   nir_pop_loop(&b, outer);
   nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
   nir_validate_shader(b.shader, "small window skip 32bit");

   /* Standard 2-arm unswitch only (main 8x + scalar tail = 5 loops total). */
   unsigned loops = 0;
   nir_foreach_block(block, b.impl) {
      if (block->cf_node.parent->type == nir_cf_node_loop &&
          block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
         loops++;
   }
   EXPECT_EQ(loops, 5u);
   ralloc_free(b.shader);
}


static void
count_loop_shape(nir_function_impl *impl, unsigned *out_loops,
                 unsigned *out_max_depth)
{
   unsigned loops = 0, max_depth = 0;
   nir_foreach_block(block, impl) {
      unsigned depth = 0;
      for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
         if (n->type == nir_cf_node_loop)
            depth++;
      if (depth > max_depth)
         max_depth = depth;
      if (block->cf_node.parent->type == nir_cf_node_loop &&
          block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
         loops++;
   }
   *out_loops = loops;
   *out_max_depth = max_depth;
}

static void
fold_after_specialize(nir_shader *shader, const char *when)
{
   for (unsigned r = 0; r < 6; ++r) {
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_constant_folding);
      NIR_PASS(_, shader, nir_opt_algebraic);
      NIR_PASS(_, shader, nir_opt_dead_cf);
      NIR_PASS(_, shader, nir_opt_dce);
   }
   nir_validate_shader(shader, when);
}

TEST(AGX3VisionOpts, SmallWindowUnswitchDecreasing)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      /* mode 0: strict > (REL_GT), x0=10, xlo=7 (trip == 3 -> exact-3)
       * mode 1: strict > (REL_GT), x0=10, xlo=5 (trip == 5 -> 4+1)
       * mode 2: strict > (REL_GT), x0=10, xlo=9 (trip == 1 -> scalar)
       * mode 3: inclusive >= (REL_GE), x0=10, xlo=8 (trip == 3 -> exact-3)
       * mode 4: inclusive >= (REL_GE), x0=INT32_MIN+2, xlo=INT32_MIN (wrap guard) */
      const bool inclusive = (mode >= 3);
      nir_builder b = make_compute_shader("small-window-decreasing");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *span = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xlo = nir_isub(&b, x0, span);
      nir_def *yhi = nir_iadd(&b, y0, span);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_def *exit_cond = inclusive ? nir_ilt(&b, x, xlo)
                                           : nir_ige(&b, xlo, x);
            nir_if *xex = nir_push_if(&b, exit_cond);
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, -1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window decreasing");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 3));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 5));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 1));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 3) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(x0, nir_imm_int(&b, INT32_MIN + 2));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      fold_after_specialize(b.shader, "small window decreasing specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 0) {
         /* Inner trip == 3 (exclusive span=3), outer trip == 4 (inclusive span=3):
          * inner peeled straight-line, outer remains a depth-1 loop. */
         EXPECT_EQ(max_depth, 1u);
         EXPECT_GE(count_intrinsics(b.shader, nir_intrinsic_load_agx), 3u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx) % 3u, 0u);
      } else if (mode == 3) {
         /* Exact 3x3 (inclusive span=2 on both axes): both loops peeled. */
         EXPECT_EQ(max_depth, 0u);
         EXPECT_EQ(loops, 0u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 9u);
      } else if (mode == 1) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
      } else {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchUnsigned)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      /* mode 0: unsigned <= (REL_LE), x0=0x80000000u, span=2 (trip == 3 -> exact-3)
       * mode 1: unsigned <= (REL_LE), x0=0x80000000u, span=4 (trip == 5 -> 4+1)
       * mode 2: unsigned <= (REL_LE), x0=UINT32_MAX-2, span=2 (wrap guard)
       * mode 3: unsigned >= (REL_GE), x0=0x80000005u, span=2 (trip == 3 -> exact-3)
       * mode 4: unsigned >= (REL_GE), x0=2u, span=2 (bound==0 wrap guard)
       * mode 5: unsigned <= (REL_LE), x0=0x80000000u, span=0 (trip == 1 -> scalar) */
      const bool dec = (mode == 3 || mode == 4);
      nir_builder b = make_compute_shader("small-window-unsigned");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *span = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xbound = dec ? nir_isub(&b, x0, span) : nir_iadd(&b, x0, span);
      nir_def *yhi = nir_iadd(&b, y0, span);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_uint_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_uint_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ult(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_def *exit_cond = dec ? nir_ult(&b, x, xbound)
                                     : nir_ult(&b, xbound, x);
            nir_if *xex = nir_push_if(&b, exit_cond);
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, dec ? -1 : 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window unsigned");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 4));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(x0, nir_imm_int(&b, (int)(UINT32_MAX - 2u)));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 3) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000005u));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 4) {
         nir_def_replace(x0, nir_imm_int(&b, 2));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 0));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      fold_after_specialize(b.shader, "small window unsigned specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 0 || mode == 3) {
         EXPECT_EQ(max_depth, 0u);
         EXPECT_EQ(loops, 0u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 9u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 1) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 5) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 2) {
         /* bound == UINT32_MAX: exact-3 unsigned no-wrap guard rejects peeling. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      } else {
         /* bound == 0u (REL_GE): exact-3 unsigned no-wrap guard sends to the
          * unchanged copy, where nir_opt_algebraic folds ult(x, 0) -> false. */
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchExactThreeLiveOutIV)
{
   nir_builder b = make_compute_shader("small-window-live-out-iv");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *out_iv = make_desc_base(&b, 16);
   nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
   nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
   nir_def *xhi = nir_iadd(&b, x0, rad);
   nir_def *yhi = nir_iadd(&b, y0, rad);

   nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, y_var, y0, 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, y_var);
      nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, yex);

      nir_def *row_off = nir_imul(&b, y, stride);
      nir_store_var(&b, x_var, x0, 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xex);

         nir_def *idx = nir_iadd(&b, row_off, x);
         nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .format = PIPE_FORMAT_R8_UINT);
         nir_store_var(&b, acc_var,
                       nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, inner);

      /* Read the inner induction variable after the loop exits (should equal
       * x0 + 3 on the exact-3 path because of the final exiting header run). */
      nir_store_agx(&b, nir_load_var(&b, x_var), out_iv, y,
                    .access = ACCESS_NON_READABLE,
                    .format = PIPE_FORMAT_R32_UINT);
      nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
   }
   nir_pop_loop(&b, outer);
   nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
   fold_after_specialize(b.shader, "small window live-out iv");

   b.cursor = nir_before_impl(b.impl);
   nir_def_replace(x0, nir_imm_int(&b, 10));
   nir_def_replace(rad, nir_imm_int(&b, 2));
   nir_def_replace(y0, nir_imm_int(&b, 0));
   fold_after_specialize(b.shader, "small window live-out iv specialized");

   unsigned loops = 0, max_depth = 0;
   count_loop_shape(b.impl, &loops, &max_depth);
   EXPECT_EQ(max_depth, 0u);
   EXPECT_EQ(loops, 0u);

   unsigned iv_stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_agx)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[0]));
         EXPECT_EQ(nir_src_as_uint(intr->src[0]), 13u);
         iv_stores++;
      }
   }
   EXPECT_GE(iv_stores, 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, SmallWindowUnswitchInnerStore)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      /* mode 0: rad=2 (trip == 3 -> exact 3 stores, exiting header runs no 4th store)
       * mode 1: rad=4 (trip == 5 -> 4 peeled stores + 1 in remainder loop) */
      nir_builder b = make_compute_shader("small-window-inner-store");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *dst = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xhi = nir_iadd(&b, x0, rad);
      nir_def *yhi = nir_iadd(&b, y0, rad);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_store_var(&b, y_var, y0, 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_agx(&b, v, dst, idx,
                          .access = ACCESS_NON_READABLE,
                          .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window inner store");

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(x0, nir_imm_int(&b, 10));
      nir_def_replace(rad, nir_imm_int(&b, mode == 0 ? 2 : 4));
      nir_def_replace(y0, nir_imm_int(&b, 0));
      fold_after_specialize(b.shader, "small window inner store specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      const unsigned n_loads = count_intrinsics(b.shader, nir_intrinsic_load_agx);
      const unsigned n_stores = count_intrinsics(b.shader, nir_intrinsic_store_agx);
      if (mode == 0) {
         EXPECT_EQ(max_depth, 0u);
         EXPECT_EQ(loops, 0u);
         EXPECT_EQ(n_stores, 9u);
         EXPECT_EQ(n_loads, 9u);
      } else {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(n_loads, 5u);
         EXPECT_EQ(n_stores, 5u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, FoldW8WordUnpackThreeComponentsAGX3B)
{
   nir_builder b = make_compute_shader("fold-w8-word-unpack-3comp");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *v = nir_load_agx(&b, 3, 8, base, gid,
                             .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                             .format = PIPE_FORMAT_R8_UINT);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, nir_u2u32(&b, nir_channel(&b, v, 0)),
                                        nir_u2u32(&b, nir_channel(&b, v, 1))),
                           nir_u2u32(&b, nir_channel(&b, v, 2)));
   nir_store_agx(&b, sum, out, gid, .access = ACCESS_NON_READABLE,
                 .format = PIPE_FORMAT_R32_UINT);

   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   EXPECT_EQ(part.info.scratch_size, 0u);
   EXPECT_EQ(part.info.stats.spills, 0u);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, SameBlockStoreLoadForward)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      /* mode 0: same-block store_agx -> matching load_agx forwards value
       * mode 1: intervening store_agx to another index blocks forwarding
       * mode 2: ACCESS_COHERENT blocks forwarding
       * mode 3: intervening barrier blocks forwarding
       * mode 4: intervening global atomic blocks forwarding
       * mode 5: mismatched SIGN_EXTEND blocks forwarding
       * mode 6: mismatched BASE shift blocks forwarding
       * mode 7: converting format (R16_UNORM with 16-bit value) blocks forwarding */
      nir_builder b = make_compute_shader("same-block-store-load-fwd");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *idx = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *val32 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

      const bool unorm = mode == 7;
      enum pipe_format fmt = unorm ? PIPE_FORMAT_R16_UNORM : PIPE_FORMAT_R32_UINT;
      unsigned bits = unorm ? 16u : 32u;
      nir_def *val = unorm ? nir_u2u16(&b, val32) : val32;
      gl_access_qualifier acc = mode == 2 ? ACCESS_COHERENT : (gl_access_qualifier)0;

      nir_store_agx(&b, val, base, idx, .access = acc, .base = 0,
                    .format = fmt, .sign_extend = false);
      if (mode == 1) {
         nir_store_agx(&b, nir_imm_int(&b, 7), base, nir_iadd_imm(&b, idx, 1),
                       .access = (gl_access_qualifier)0, .format = PIPE_FORMAT_R32_UINT);
      } else if (mode == 3) {
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_DEVICE,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_ssbo);
      } else if (mode == 4) {
         nir_global_atomic(&b, 32, base, nir_imm_int(&b, 1),
                           .atomic_op = nir_atomic_op_iadd);
      }
      nir_def *ld = nir_load_agx(&b, 1, bits, base, idx, .access = acc,
                                 .base = mode == 6 ? 2 : 0,
                                 .format = fmt,
                                 .sign_extend = mode == 5);
      nir_def *res = unorm ? nir_u2u32(&b, ld) : nir_iadd_imm(&b, ld, 1);
      nir_store_global(&b, res, out, .align_mul = 4);

      bool progress = agx_nir_opt_forward_merge_loads(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "same block store load fwd");

      if (mode == 0) {
         EXPECT_TRUE(progress);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
      } else {
         EXPECT_FALSE(progress);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Exact3x3OuterPeelAndGuards)
{
   for (unsigned mode = 0; mode < 7; ++mode) {
      /* mode 0: 3x6 window (inner == 3, outer == 6) -> inner peeled, outer stays loop
       * mode 1: 3x3 with two inner loops -> outer stays loop (count_nested_loops guard)
       * mode 2: 3x3 with barrier in outer body -> outer stays loop (outer_peelable guard)
       * mode 3: 3x2 window (inner == 3, outer == 2) -> inner peeled, outer stays loop
       * mode 4: 3x4 window (inner == 3, outer == 4) -> inner peeled, outer stays loop
       * mode 5: 3x3 with outer REL_LE bound at INT32_MAX -> no-wrap guard keeps outer loop
       * mode 6: exact 3x3 window -> both outer and inner peel completely (9 loads, 0 loops) */
      nir_builder b = make_compute_shader("exact-3x3-outer-guards");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 20);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 20);
      nir_def *xrad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 20);
      nir_def *yrad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 20);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 16), .range = 20);
      nir_def *xhi = nir_iadd(&b, x0, xrad);
      nir_def *yhi = nir_iadd(&b, y0, yrad);

      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, y_var, y0, 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);
         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *v = nir_load_agx(&b, 1, 8, base, nir_iadd(&b, row_off, x),
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, inner);

         if (mode == 1) {
            nir_store_var(&b, x_var, x0, 1);
            nir_loop *inner2 = nir_push_loop(&b);
            {
               nir_def *x = nir_load_var(&b, x_var);
               nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, xex);

               nir_def *v = nir_load_agx(&b, 1, 8, base, nir_iadd(&b, row_off, x),
                                         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                         .format = PIPE_FORMAT_R8_UINT);
               nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
               nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
            }
            nir_pop_loop(&b, inner2);
         } else if (mode == 2) {
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_shared);
         }
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "exact 3x3 outer guards");

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(x0, nir_imm_int(&b, 10));
      nir_def_replace(xrad, nir_imm_int(&b, 2));
      const int y0_val = mode == 5 ? (INT32_MAX - 2) : 0;
      int yrad_val = 2;
      if (mode == 0)
         yrad_val = 5;
      else if (mode == 3)
         yrad_val = 1;
      else if (mode == 4)
         yrad_val = 3;
      nir_def_replace(y0, nir_imm_int(&b, y0_val));
      nir_def_replace(yrad, nir_imm_int(&b, yrad_val));
      fold_after_specialize(b.shader, "exact 3x3 outer guards specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 6) {
         EXPECT_EQ(max_depth, 0u);
         EXPECT_EQ(loops, 0u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 9u);
      } else {
         EXPECT_GE(max_depth, 1u);
         EXPECT_GE(count_intrinsics(b.shader, nir_intrinsic_load_agx), 3u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave4InvariantIfUnswitch)
{
   for (int mode = 0; mode < 4; ++mode) {
      /* mode 0: positive (invariant if with >= 12 work instrs & >= 2 invariant ALU)
       * mode 1: negative (loop-variant condition)
       * mode 2: negative (arm contains break)
       * mode 3: negative (loop contains barrier)
       */
      nir_builder b = make_compute_shader("wave4_test");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *y = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_def *stride = nir_load_preamble(&b, 1, 32, .base = 12);
      nir_def *flag = nir_ine_imm(&b, nir_load_preamble(&b, 1, 32, .base = 14), 0);

      nir_variable *x_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_store_var(&b, x_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *ex = nir_push_if(&b, nir_ige(&b, x, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ex);

         if (mode == 3) {
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_shared);
         }

         nir_def *cond = (mode == 1) ? nir_ilt(&b, x, y) : flag;
         nir_if *nif = nir_push_if(&b, cond);
         {
            if (mode == 2) {
               nir_jump(&b, nir_jump_break);
            } else {
               nir_def *yf = nir_i2f32(&b, y);
               nir_def *ys = nir_fmul_imm(&b, yf, 0.5f);
               nir_def *y0 = nir_f2i32(&b, ys);
               nir_def *y1 = nir_iadd_imm(&b, y0, 1);
               nir_def *r0 = nir_imul(&b, y0, stride);
               nir_def *r1 = nir_imul(&b, y1, stride);
               nir_def *xf = nir_i2f32(&b, x);
               nir_def *xs = nir_fmul_imm(&b, xf, 0.5f);
               nir_def *x0 = nir_f2i32(&b, xs);
               nir_def *v0 = nir_load_agx(&b, 1, 8, base, nir_iadd(&b, r0, x0),
                                          .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                          .format = PIPE_FORMAT_R8_UINT);
               nir_def *v1 = nir_load_agx(&b, 1, 8, base, nir_iadd(&b, r1, x0),
                                          .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                          .format = PIPE_FORMAT_R8_UINT);
               nir_def *sum = nir_iadd(&b, nir_u2u32(&b, v0), nir_u2u32(&b, v1));
               nir_def *avg = nir_ushr_imm(&b, sum, 1);
               nir_store_agx(&b, nir_u2u8(&b, avg), out, nir_iadd(&b, nir_imul(&b, y, stride), x),
                             .format = PIPE_FORMAT_R8_UINT);
            }
         }
         nir_push_else(&b, nif);
         {
            nir_def *idx = nir_iadd(&b, nir_imul(&b, y, stride), x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_agx(&b, v, out, idx, .format = PIPE_FORMAT_R8_UINT);
         }
         nir_pop_if(&b, nif);

         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, loop);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      bool changed = agx_nir_opt_unswitch_invariant_if(b.shader);
      nir_validate_shader(b.shader, "unswitch invariant if");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 0) {
         EXPECT_TRUE(changed);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(max_depth, 1u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(loops, 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave4Pack64StoreLoadForward)
{
   for (int match = 0; match < 2; ++match) {
      nir_builder b = make_compute_shader("wave4_test");
      nir_def *lo0 = nir_load_preamble(&b, 1, 32, .base = 0);
      nir_def *hi0 = nir_load_preamble(&b, 1, 32, .base = 2);
      nir_def *lo1 = nir_load_preamble(&b, 1, 32, .base = 0);
      nir_def *hi1 = nir_load_preamble(&b, 1, 32, .base = match ? 2 : 4);
      nir_def *base0 = nir_pack_64_2x32_split(&b, lo0, hi0);
      nir_def *base1 = nir_pack_64_2x32_split(&b, lo1, hi1);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *val = nir_u2u8(&b, nir_load_preamble(&b, 1, 32, .base = 10));
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 12);

      nir_store_agx(&b, val, base0, idx, .format = PIPE_FORMAT_R8_UINT);
      nir_def *reloaded = nir_load_agx(&b, 1, 8, base1, idx,
                                       .format = PIPE_FORMAT_R8_UINT);
      nir_store_agx(&b, reloaded, out, idx, .format = PIPE_FORMAT_R8_UINT);

      bool changed = agx_nir_opt_forward_merge_loads(b.shader);
      nir_opt_dce(b.shader);
      nir_validate_shader(b.shader, "pack64 store load forward");

      if (match) {
         EXPECT_TRUE(changed);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave4SpeculativeLicm)
{
   for (int spec = 0; spec < 2; ++spec) {
      nir_builder b = make_compute_shader("wave4_test");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *y = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_def *stride = nir_load_preamble(&b, 1, 32, .base = 12);

      nir_variable *x_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_store_var(&b, x_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *ex = nir_push_if(&b, nir_ige(&b, x, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ex);

         /* Top-level invariant ALU: should hoist only when spec == 1. */
         nir_def *row = nir_imul(&b, y, stride);
         /* Top-level invariant load: must NEVER speculatively hoist past break. */
         nir_def *inv_ld = nir_load_agx(&b, 1, 32, base, row,
                                        .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                        .format = PIPE_FORMAT_R32_UINT);
         /* Nested-if invariant ALU: must NEVER speculatively hoist from branch. */
         nir_variable *acc =
            nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
         nir_store_var(&b, acc, inv_ld, 1);
         nir_if *cond = nir_push_if(&b, nir_ieq_imm(&b, x, 0));
         {
            nir_def *cold = nir_iadd_imm(&b, nir_imul(&b, y, y), 7);
            nir_store_var(&b, acc, nir_iadd(&b, inv_ld, cold), 1);
         }
         nir_pop_if(&b, cond);

         nir_store_agx(&b, nir_load_var(&b, acc), out, nir_iadd(&b, row, x),
                       .format = PIPE_FORMAT_R32_UINT);
         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, loop);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      agx_nir_opt_licm(b.shader, 64, spec != 0);
      nir_validate_shader(b.shader, "speculative licm");

      unsigned imul_in_loop = 0, loads_in_loop = 0;
      nir_foreach_block(block, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
            if (n->type == nir_cf_node_loop)
               in_loop = true;
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_imul)
               imul_in_loop++;
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
               loads_in_loop++;
         }
      }

      /* Invariant load must always stay inside the loop. */
      EXPECT_EQ(loads_in_loop, 1u);
      /* Top-level imul hoists only when speculative; nested-if imul stays. */
      EXPECT_EQ(imul_in_loop, spec ? 1u : 2u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave4InvariantIfUnswitchLiveOutAndTwoBreaks)
{
   /* Case A: ALU live-out value used after the loop -> unswitches cleanly and
    * preserves valid NIR. */
   {
      nir_builder b = make_compute_shader("unswitch_live_out");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *flag = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *k = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 12);
      nir_def *cond = nir_ine_imm(&b, flag, 0);

      nir_variable *iv_v = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_variable *acc_v = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, iv_v, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc_v, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_v);
      nir_if *brk = nir_push_if(&b, nir_ige(&b, iv, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk);

      nir_if *inv_if = nir_push_if(&b, cond);
      nir_def *t = nir_iadd_imm(&b, k, 3);
      nir_def *t2 = nir_imul_imm(&b, t, 5);
      nir_def *v = nir_load_agx(&b, 1, 32, base, iv,
                                .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                .format = PIPE_FORMAT_R32_UINT);
      for (int i = 0; i < 12; ++i)
         v = nir_iadd(&b, v, t2);
      nir_store_var(&b, acc_v, nir_iadd(&b, nir_load_var(&b, acc_v), v), 1);
      nir_push_else(&b, inv_if);
      nir_store_var(&b, acc_v, nir_iadd(&b, nir_load_var(&b, acc_v), k), 1);
      nir_pop_if(&b, inv_if);

      nir_store_var(&b, iv_v, nir_iadd_imm(&b, iv, 1), 1);
      nir_pop_loop(&b, loop);

      /* Live-out ALU accumulator used after the loop */
      nir_def *final_acc = nir_load_var(&b, acc_v);
      nir_store_agx(&b, final_acc, out_base, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);

      nir_lower_vars_to_ssa(b.shader);
      nir_opt_dce(b.shader);
      nir_validate_shader(b.shader, "before unswitch live_out");
      bool progress = agx_nir_opt_unswitch_invariant_if(b.shader);
      nir_validate_shader(b.shader, "after unswitch live_out");
      EXPECT_TRUE(progress);
      ralloc_free(b.shader);
   }

   /* Case B: Loop with two breaks feeding a successor-block phi -> must be
    * refused safely so the exit phi never references stale break blocks. */
   {
      nir_builder b = make_compute_shader("unswitch_two_breaks");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *flag = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *k = nir_load_preamble(&b, 1, 32, .base = 10);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 12);
      nir_def *cond = nir_ine_imm(&b, flag, 0);

      nir_variable *iv_v = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_variable *res_v = nir_local_variable_create(b.impl, glsl_uint_type(), "res");
      nir_store_var(&b, iv_v, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, res_v, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_v);
      nir_if *brk1 = nir_push_if(&b, nir_ige(&b, iv, n));
      nir_store_var(&b, res_v, nir_imm_int(&b, 1), 1);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk1);

      nir_if *inv_if = nir_push_if(&b, cond);
      nir_def *t = nir_iadd_imm(&b, k, 3);
      nir_def *t2 = nir_imul_imm(&b, t, 5);
      nir_def *v = nir_load_agx(&b, 1, 32, base, iv,
                                .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                .format = PIPE_FORMAT_R32_UINT);
      for (int i = 0; i < 12; ++i)
         v = nir_iadd(&b, v, t2);
      nir_store_var(&b, res_v, v, 1);
      nir_push_else(&b, inv_if);
      nir_store_var(&b, res_v, k, 1);
      nir_pop_if(&b, inv_if);

      nir_if *brk2 = nir_push_if(&b, nir_ieq_imm(&b, nir_load_var(&b, res_v), 42));
      nir_store_var(&b, res_v, nir_imm_int(&b, 2), 1);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk2);

      nir_store_var(&b, iv_v, nir_iadd_imm(&b, iv, 1), 1);
      nir_pop_loop(&b, loop);

      nir_store_agx(&b, nir_load_var(&b, res_v), out_base, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);

      nir_lower_vars_to_ssa(b.shader);
      nir_opt_dce(b.shader);
      nir_validate_shader(b.shader, "before unswitch two_breaks");
      bool progress = agx_nir_opt_unswitch_invariant_if(b.shader);
      nir_validate_shader(b.shader, "after unswitch two_breaks");
      EXPECT_FALSE(progress);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave4Agx3bCodegenW8ConvertAndUniformBase)
{
   unsigned instrs_by_mode[4] = {};

   for (int mode = 0; mode < 4; ++mode) {
      /* mode 0: aligned low uniform bases (u0:u1, u4:u5), R8_UINT vec4 unpack + u2f32 + u2u8 store
       * mode 1: aligned low uniform bases, R8_SINT -> i2f16 & i2f32
       * mode 2: high/unaligned uniform bases (u2:u3, u256:u257)
       * mode 3: coherent store with aligned low uniform base
       */
      nir_builder b = make_compute_shader("wave4_test");
      b.shader->info.internal = false;
      int in_base_u = (mode == 2) ? 2 : 0;
      int out_base_u = (mode == 2) ? 256 : 4;
      nir_def *in_base = nir_pack_64_2x32_split(
         &b, nir_load_preamble(&b, 1, 32, .base = in_base_u),
         nir_load_preamble(&b, 1, 32, .base = in_base_u + 2));
      nir_def *out_base = nir_pack_64_2x32_split(
         &b, nir_load_preamble(&b, 1, 32, .base = out_base_u),
         nir_load_preamble(&b, 1, 32, .base = out_base_u + 2));
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 12);

      if (mode == 1) {
         nir_def *sv = nir_load_agx(&b, 2, 8, in_base, idx,
                                    .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                    .format = PIPE_FORMAT_R8_UINT);
         nir_def *f16 = nir_i2f16(&b, nir_channel(&b, sv, 0));
         nir_def *f32 = nir_i2f32(&b, nir_channel(&b, sv, 1));
         nir_def *sum = nir_fadd(&b, nir_f2f32(&b, f16), f32);
         nir_store_agx(&b, sum, out_base, idx, .format = PIPE_FORMAT_R32_UINT);
      } else {
         nir_def *uv = nir_load_agx(&b, 4, 8, in_base, idx,
                                    .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                    .format = PIPE_FORMAT_R8_UINT);
         nir_def *acc = nir_u2f32(&b, nir_channel(&b, uv, 0));
         acc = nir_fadd(&b, acc, nir_u2f32(&b, nir_channel(&b, uv, 1)));
         acc = nir_fadd(&b, acc, nir_u2f32(&b, nir_channel(&b, uv, 2)));
         acc = nir_fadd(&b, acc, nir_u2f32(&b, nir_channel(&b, uv, 3)));
         nir_def *byte_out = nir_u2u8(&b, nir_f2u32(&b, acc));
         gl_access_qualifier st_acc =
            (mode == 3) ? ACCESS_COHERENT : (gl_access_qualifier)0;
         nir_store_agx(&b, byte_out, out_base, idx,
                       .access = st_acc, .format = PIPE_FORMAT_R8_UINT);
      }

      struct agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17P;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.has_scratch = true;
      key.no_preamble = true;

      agx_preprocess_nir(b.shader);
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      EXPECT_EQ(part.info.scratch_size, 0u);
      instrs_by_mode[mode] = part.info.stats.instrs;
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* Mode 0 (folded aligned low uniforms) must emit fewer instructions than
    * mode 2 (unaligned/high uniform bases) and mode 3 (coherent store). */
   EXPECT_LT(instrs_by_mode[0], instrs_by_mode[2]);
   EXPECT_LT(instrs_by_mode[0], instrs_by_mode[3]);

   /* Direct IR verification of AGX3B uniform-base folding:
    * aligned low uniforms fold on non-coherent load/store, while unaligned,
    * high (>= 256), and coherent stores must not fold. */
   for (int tc = 0; tc < 4; ++tc) {
      nir_builder nb = make_compute_shader("uniform_base_ir");
      nb.shader->info.internal = false;
      void *mem_ctx = ralloc_context(NULL);
      struct agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17P;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;

      agx_context *ctx = rzalloc(mem_ctx, agx_context);
      ctx->nir = nb.shader;
      ctx->stage = MESA_SHADER_COMPUTE;
      ctx->key = &key;
      list_inithead(&ctx->blocks);

      agx_block *blk = rzalloc(ctx, agx_block);
      list_inithead(&blk->instructions);
      list_addtail(&blk->link, &ctx->blocks);

      agx_builder ab = agx_init_builder(ctx, agx_after_block(blk));
      unsigned lo_u = (tc == 1) ? 2 : (tc == 2) ? 256 : 4;
      agx_index base = agx_temp(ctx, AGX_SIZE_64);
      agx_instr *col = agx_collect_to(&ab, base, 2);
      col->src[0] = agx_uniform(lo_u, AGX_SIZE_32);
      col->src[1] = agx_uniform(lo_u + 2, AGX_SIZE_32);

      agx_index idx = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_imm_to(&ab, idx, 0);
      agx_index val = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_imm_to(&ab, val, 1);

      agx_instr *ld = agx_device_load_to(&ab, agx_temp(ctx, AGX_SIZE_32), base,
                                         idx, AGX_FORMAT_I32, 0x1, 0, false);
      agx_instr *st = agx_device_store(&ab, val, base, idx, AGX_FORMAT_I32,
                                       0x1, 0, tc == 3 /* coherent */);
      agx_optimizer_forward(ctx);

      if (tc == 0) {
         /* Aligned low uniform, non-coherent: both load and store fold. */
         EXPECT_EQ(ld->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_EQ(ld->src[0].value, 4u);
         EXPECT_EQ(ld->src[0].size, AGX_SIZE_64);
         EXPECT_EQ(st->src[1].type, AGX_INDEX_UNIFORM);
         EXPECT_EQ(st->src[1].value, 4u);
         EXPECT_EQ(st->src[1].size, AGX_SIZE_64);
      } else if (tc == 1 || tc == 2) {
         /* Unaligned (u2:u3) or high (u256:u257): neither folds. */
         EXPECT_NE(ld->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_NE(st->src[1].type, AGX_INDEX_UNIFORM);
      } else {
         /* Coherent store: load still folds, coherent store does not. */
         EXPECT_EQ(ld->src[0].type, AGX_INDEX_UNIFORM);
         EXPECT_NE(st->src[1].type, AGX_INDEX_UNIFORM);
      }

      ralloc_free(mem_ctx);
      ralloc_free(nb.shader);
   }

   /* Verify scalar W8 store peels the u2u8 0xff mask on AGX3B:
    * mode 0: store_agx(R8_UINT, u2u8(x)) peels the 0xff mask (only the two
    *         uniform->GPR movs remain: alu == 2).
    * mode 1: store_agx(R8_UINT, u2u8(iand(x, 0x7f))) keeps the explicit
    *         0x7f mask (alu == 3).
    * mode 2: store_agx(R16_UINT, u2u16(u2u8(x))) cannot peel the 0xff mask
    *         because a W16 store writes all 16 bits (alu == 4). */
   unsigned alu_cnt[3] = {0, 0, 0};
   for (int mode = 0; mode < 3; ++mode) {
      nir_builder b = make_compute_shader("w8_store_peel");
      b.shader->info.internal = false;
      nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 4);
      nir_def *val32 = nir_load_preamble(&b, 1, 32, .base = 6);
      if (mode == 1)
         val32 = nir_iand_imm(&b, val32, 0x7f);
      nir_def *val8 = nir_u2u8(&b, val32);
      if (mode == 2)
         nir_store_agx(&b, nir_u2u16(&b, val8), out_base, idx, .format = PIPE_FORMAT_R16_UINT);
      else
         nir_store_agx(&b, val8, out_base, idx, .format = PIPE_FORMAT_R8_UINT);

      struct agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17P;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.no_preamble = true;
      agx_preprocess_nir(b.shader);
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      alu_cnt[mode] = part.info.stats.alu;
      free(part.binary);
      ralloc_free(b.shader);
   }
   EXPECT_EQ(alu_cnt[0], 2u);
   EXPECT_EQ(alu_cnt[1], 3u);
   EXPECT_EQ(alu_cnt[2], 4u);
}

TEST(AGX3VisionOpts, PackNarrowMuxTreeAndFoldIsub)
{
   for (int mode = 0; mode < 3; ++mode) {
      /* mode 0: contiguous layout (all 16 leaves loaded in preheader)
       * mode 1: strided layout (leaves 0,4,8,12 loaded in dominating block)
       * mode 2: unbounded index (no mask) -> must reject safely
       */
      nir_builder b = make_compute_shader("pack_mux_tree");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *center = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *cond = nir_ine_imm(&b, center, 0);

      nir_def *leaves[16] = {};
      if (mode == 1) {
         for (unsigned i = 0; i < 16; i += 4) {
            leaves[i] = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, i),
                                     .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                     .format = PIPE_FORMAT_R8_UINT);
         }
      }

      nir_if *guard_if = nir_push_if(&b, cond);
      for (unsigned i = 0; i < 16; ++i) {
         if (!leaves[i]) {
            leaves[i] = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, i),
                                     .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                     .format = PIPE_FORMAT_R8_UINT);
         }
      }

      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *iv_var = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, iv_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_var);
      nir_if *brk = nir_push_if(&b, nir_ige_imm(&b, iv, 24));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk);

      nir_def *idx = (mode == 2) ? iv : nir_iand_imm(&b, iv, 15);
      /* Build 15-bcsel balanced tree over leaves[0..15] */
      auto build_tree = [&](auto &&self, unsigned lo, unsigned hi) -> nir_def * {
         if (hi - lo == 1)
            return leaves[lo];
         unsigned mid = (lo + hi) / 2;
         nir_def *l = self(self, lo, mid);
         nir_def *r = self(self, mid, hi);
         return nir_bcsel(&b, nir_ilt_imm(&b, idx, (int32_t)mid), l, r);
      };
      nir_def *picked = build_tree(build_tree, 0, 16);
      nir_def *diff = nir_iadd(&b, center, nir_ineg(&b, nir_u2u32(&b, picked)));
      nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), diff), 1);
      nir_store_var(&b, iv_var, nir_iadd_imm(&b, iv, 1), 1);
      nir_pop_loop(&b, loop);

      nir_store_agx(&b, nir_load_var(&b, acc_var), out_base, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);
      nir_pop_if(&b, guard_if);

      nir_lower_vars_to_ssa(b.shader);
      nir_validate_shader(b.shader, "before pack_mux");

      bool progress = agx_nir_opt_pack_narrow_mux_tree(b.shader);
      nir_validate_shader(b.shader, "after pack_mux");
      nir_opt_dce(b.shader);

      unsigned ubfe_cnt = 0, isub_cnt = 0, bcsel8_cnt = 0;
      nir_foreach_block(blk, b.impl) {
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ubitfield_extract)
               ubfe_cnt++;
            else if (alu->op == nir_op_isub)
               isub_cnt++;
            else if (alu->op == nir_op_bcsel && alu->def.bit_size == 8)
               bcsel8_cnt++;
         }
      }

      EXPECT_TRUE(progress);
      EXPECT_EQ(isub_cnt, 1u);
      if (mode == 2) {
         EXPECT_EQ(ubfe_cnt, 0u);
         EXPECT_EQ(bcsel8_cnt, 15u);
      } else {
         EXPECT_EQ(ubfe_cnt, 1u);
         EXPECT_EQ(bcsel8_cnt, 0u);
      }

      ralloc_free(b.shader);
   }
}


TEST(AGX3VisionOpts, PureReadonlyLoopLicmHoist)
{
   /* mode 0: pure read-only loop with header load_agx (count) and loop-invariant
    *         body load_agx (desc_a) -> Phase 1 hoists count before loop,
    *         Phase 2 wraps loop in zero-speculation guard and hoists desc_a.
    * mode 1: loop contains store_agx -> neither load hoisted.
    * mode 2: loop contains barrier -> neither load hoisted.
    * mode 3: load_agx is ACCESS_COHERENT -> neither load hoisted.
    * mode 4: budget too tight (budget = 4 halves) -> body 32x4 load not hoisted. */
   for (int mode = 0; mode < 5; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_licm");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_cnt = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_cnt = nir_pack_64_2x32_split(&b, nir_channel(&b, p_cnt, 0),
                                              nir_channel(&b, p_cnt, 1));
      nir_def *p_a = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_a = nir_pack_64_2x32_split(&b, nir_channel(&b, p_a, 0),
                                            nir_channel(&b, p_a, 1));
      nir_def *p_b = nir_load_preamble(&b, 2, 32, .base = 8);
      nir_def *a_b = nir_pack_64_2x32_split(&b, nir_channel(&b, p_b, 0),
                                            nir_channel(&b, p_b, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 12);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));

      gl_access_qualifier acc =
         mode == 3 ? (gl_access_qualifier)(ACCESS_COHERENT | ACCESS_CAN_SPECULATE)
                   : (gl_access_qualifier)ACCESS_CAN_SPECULATE;

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_int_type(), "i");
      nir_variable *sum_var = nir_local_variable_create(b.impl, glsl_int_type(), "sum");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum_var, nir_imm_int(&b, 0), 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_def *cnt = nir_load_agx(&b, 1, 32, a_cnt, nir_imm_int(&b, 0),
                                     .access = acc, .base = 0,
                                     .format = PIPE_FORMAT_R32_UINT,
                                     .sign_extend = false);
         nir_push_if(&b, nir_ige(&b, i, cnt));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         if (mode == 2) {
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                        .memory_scope = SCOPE_WORKGROUP,
                        .memory_semantics = NIR_MEMORY_ACQ_REL,
                        .memory_modes = nir_var_mem_ssbo);
         }

         nir_def *va = nir_load_agx(&b, 4, 32, a_a, gid,
                                    .access = acc, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_def *vb = nir_load_agx(&b, 4, 32, a_b, i,
                                    .access = ACCESS_CAN_SPECULATE, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_def *x = nir_ixor(&b, nir_channel(&b, va, 0), nir_channel(&b, vb, 0));
         nir_def *s = nir_iadd(&b, nir_load_var(&b, sum_var), x);
         if (mode == 1) {
            nir_store_agx(&b, s, a_o, i,
                          .access = ACCESS_NON_READABLE, .base = 0,
                          .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
         }
         nir_store_var(&b, sum_var, s, 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, sum_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);

      unsigned budget = (mode == 4) ? 4 : 64;
      NIR_PASS(_, b.shader, agx_nir_opt_licm, budget, true);
      nir_validate_shader(b.shader, "after pure readonly loop licm");

      unsigned loads_in_loop = 0;
      unsigned loads_outside_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent) {
            if (n->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx) {
               if (in_loop)
                  loads_in_loop++;
               else
                  loads_outside_loop++;
            }
         }
      }

      if (mode == 0) {
         /* Both cnt (Phase 1) and va (Phase 2) hoisted; only vb stays in loop. */
         EXPECT_EQ(loads_in_loop, 1u);
         EXPECT_EQ(loads_outside_loop, 2u);

         /* Verify the Phase 2 zero-trip guard polarity and merge-phi else
          * source so flipped-guard and undef-init regressions are caught. */
         nir_if *guard = NULL;
         foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
            if (node->type == nir_cf_node_if) {
               guard = nir_cf_node_as_if(node);
               break;
            }
         }
         ASSERT_NE(guard, nullptr);

         /* Guard condition must be inot(0 >= cnt), entering when cnt > 0. */
         nir_scalar gcond = nir_get_scalar(guard->condition.ssa, 0);
         ASSERT_TRUE(nir_scalar_is_alu(gcond));
         ASSERT_EQ(nir_scalar_alu_op(gcond), nir_op_inot);
         nir_scalar brk = nir_scalar_chase_alu_src(gcond, 0);
         ASSERT_TRUE(nir_scalar_is_alu(brk));
         ASSERT_EQ(nir_scalar_alu_op(brk), nir_op_ige);
         nir_scalar lhs = nir_scalar_chase_alu_src(brk, 0);
         nir_scalar rhs = nir_scalar_chase_alu_src(brk, 1);
         EXPECT_TRUE(nir_scalar_is_const(lhs));
         EXPECT_EQ(nir_scalar_as_uint(lhs), 0u);
         EXPECT_EQ(nir_def_instr(rhs.def)->type, nir_instr_type_intrinsic);
         EXPECT_EQ(nir_instr_as_intrinsic(nir_def_instr(rhs.def))->intrinsic,
                   nir_intrinsic_load_agx);

         /* Merge phi after guard must take the preheader init value (0) on
          * the zero-trip else path and the loop header phi on the then path. */
         nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&guard->cf_node));
         nir_block *then_end = nir_if_last_then_block(guard);
         nir_block *else_end = nir_if_last_else_block(guard);
         nir_phi_instr *merge_phi = NULL;
         nir_foreach_phi(phi, after) {
            merge_phi = phi;
            break;
         }
         ASSERT_NE(merge_phi, nullptr);
         nir_phi_src *else_src = nir_phi_get_src_from_block(merge_phi, else_end);
         nir_phi_src *then_src = nir_phi_get_src_from_block(merge_phi, then_end);
         ASSERT_NE(else_src, nullptr);
         ASSERT_NE(then_src, nullptr);
         EXPECT_TRUE(nir_src_is_const(else_src->src));
         EXPECT_EQ(nir_src_as_uint(else_src->src), 0u);
         EXPECT_EQ(nir_def_instr(then_src->src.ssa)->type, nir_instr_type_phi);
      } else if (mode == 4) {
         /* Budget 4 halves forbids hoisting va (8 halves). */
         EXPECT_GE(loads_in_loop, 2u);
      } else {
         /* Stores, barriers, and coherent loads block hoisting. */
         EXPECT_EQ(loads_in_loop, 3u);
         EXPECT_EQ(loads_outside_loop, 0u);
      }

      ralloc_free(b.shader);
   }

   /* Mode 5: early-break linear search with an exit phi
    *   int f = -1;
    *   for (uint i = 0; i < n; i++) {
    *      uint k = keys[0];
    *      if (data[i] == k) { f = int(i); break; }
    *   }
    *   out[g] = f;
    * Phase 2 must bail when the loop successor starts with a phi so SSA
    * dominance is preserved and lanes with n == 0 still observe -1. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_early_break_phi");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_keys = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_keys = nir_pack_64_2x32_split(&b, nir_channel(&b, p_keys, 0),
                                               nir_channel(&b, p_keys, 1));
      nir_def *p_data = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_data = nir_pack_64_2x32_split(&b, nir_channel(&b, p_data, 0),
                                               nir_channel(&b, p_data, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 8);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 12);

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *f_var = nir_local_variable_create(b.impl, glsl_int_type(), "f");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, f_var, nir_imm_int(&b, -1), 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_uge(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *k = nir_load_agx(&b, 1, 32, a_keys, nir_imm_int(&b, 0),
                                   .access = ACCESS_CAN_SPECULATE, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);
         nir_def *d = nir_load_agx(&b, 1, 32, a_data, i,
                                   .access = ACCESS_CAN_SPECULATE, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);
         nir_push_if(&b, nir_ieq(&b, d, k));
         {
            nir_store_var(&b, f_var, i, 1);
            nir_jump(&b, nir_jump_break);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, f_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);

      NIR_PASS(_, b.shader, agx_nir_opt_licm, 64, true);
      nir_validate_shader(b.shader, "after early-break exit-phi licm");

      unsigned loads_in_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *cf = blk->cf_node.parent; cf; cf = cf->parent) {
            if (cf->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
               loads_in_loop++;
         }
      }
      EXPECT_EQ(loads_in_loop, 2u);
      ralloc_free(b.shader);
   }

   /* Mode 6: speculative == false (e.g. AGX3C / graphics stages) must NOT
    * hoist pure read-only loop loads. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "ro_loop_no_spec");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *p_a = nir_load_preamble(&b, 2, 32, .base = 0);
      nir_def *a_a = nir_pack_64_2x32_split(&b, nir_channel(&b, p_a, 0),
                                            nir_channel(&b, p_a, 1));
      nir_def *p_o = nir_load_preamble(&b, 2, 32, .base = 4);
      nir_def *a_o = nir_pack_64_2x32_split(&b, nir_channel(&b, p_o, 0),
                                            nir_channel(&b, p_o, 1));
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_int_type(), "i");
      nir_variable *s_var = nir_local_variable_create(b.impl, glsl_int_type(), "s");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, s_var, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_ige(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);
         nir_def *va = nir_load_agx(&b, 1, 32, a_a, gid,
                                    .access = ACCESS_CAN_SPECULATE, .base = 0,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false);
         nir_store_var(&b, s_var, nir_iadd(&b, nir_load_var(&b, s_var), va), 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      nir_store_agx(&b, nir_load_var(&b, s_var), a_o, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, agx_nir_opt_licm, 64, false);
      nir_validate_shader(b.shader, "after non-speculative licm");

      unsigned loads_in_loop = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *cf = blk->cf_node.parent; cf; cf = cf->parent) {
            if (cf->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
               loads_in_loop++;
         }
      }
      EXPECT_EQ(loads_in_loop, 1u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, PackNarrowMuxTreePreNegateAndMaskPeel)
{
   /* Verify the exact 8-bit complement identity across all byte pairs. */
   for (int c = 0; c < 256; ++c) {
      for (int bval = 0; bval < 256; ++bval) {
         int inv_b = (~bval) & 0xff;
         int inv_c = 255 - c;
         ASSERT_EQ(inv_b - inv_c, c - bval);
      }
   }

   for (int neg_is_true = 0; neg_is_true < 2; ++neg_is_true) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "pack_mux_prenegate");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 2);
      nir_def *inv_cond = nir_ilt_imm(&b, gid, 4);

      nir_def *c8 = nir_load_agx(&b, 1, 8, base, nir_imm_int(&b, 99),
                                 .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                                 .base = 0, .format = PIPE_FORMAT_R8_UINT,
                                 .sign_extend = false);
      nir_def *c32 = nir_u2u32(&b, c8);

      nir_def *v[8];
      for (unsigned i = 0; i < 8; i += 2) {
         v[i] = nir_load_agx(&b, 1, 8, base, nir_iadd_imm(&b, gid, i),
                             .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                             .base = 0, .format = PIPE_FORMAT_R8_UINT,
                             .sign_extend = false);
      }

      nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_int_type(), "acc");
      nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_push_if(&b, nir_ilt_imm(&b, gid, 1000));
      for (unsigned i = 1; i < 8; i += 2) {
         v[i] = nir_load_agx(&b, 1, 8, base, nir_iadd_imm(&b, gid, i),
                             .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
                             .base = 0, .format = PIPE_FORMAT_R8_UINT,
                             .sign_extend = false);
      }

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_var);
         nir_push_if(&b, nir_ige_imm(&b, i, 24));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *idx = nir_iand_imm(&b, i, 7);
         nir_def *s01 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 1), v[0], v[1]);
         nir_def *s23 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 3), v[2], v[3]);
         nir_def *s45 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 5), v[4], v[5]);
         nir_def *s67 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 7), v[6], v[7]);
         nir_def *s03 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 2), s01, s23);
         nir_def *s47 = nir_bcsel(&b, nir_ilt_imm(&b, idx, 6), s45, s67);
         nir_def *mux = nir_bcsel(&b, nir_ilt_imm(&b, idx, 4), s03, s47);

         nir_def *b32 = nir_u2u32(&b, mux);
         nir_def *diff = nir_iadd(&b, c32, nir_ineg(&b, b32));
         nir_def *ndiff = nir_ineg(&b, diff);
         nir_def *signed_diff = neg_is_true ? nir_bcsel(&b, inv_cond, ndiff, diff)
                                            : nir_bcsel(&b, inv_cond, diff, ndiff);

         nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), signed_diff), 1);
         nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      nir_pop_if(&b, NULL);

      nir_store_agx(&b, nir_load_var(&b, acc_var), out, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, agx_nir_opt_pack_narrow_mux_tree);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "after pack mux prenegate");

      unsigned loop_ineg = 0, loop_iand7 = 0, loop_isub = 0, loop_ubfe = 0;
      nir_foreach_block(blk, b.impl) {
         bool in_loop = false;
         for (nir_cf_node *n = blk->cf_node.parent; n; n = n->parent)
            if (n->type == nir_cf_node_loop)
               in_loop = true;
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ineg)
               loop_ineg++;
            else if (alu->op == nir_op_isub)
               loop_isub++;
            else if (alu->op == nir_op_ubitfield_extract)
               loop_ubfe++;
            else if (alu->op == nir_op_iand) {
               for (unsigned s = 0; s < 2; ++s) {
                  nir_scalar sc = nir_scalar_resolved(alu->src[s].src.ssa, 0);
                  if (nir_scalar_is_const(sc) && nir_scalar_as_uint(sc) == 7)
                     loop_iand7++;
               }
            }
         }
      }
      EXPECT_EQ(loop_ubfe, 1u);
      EXPECT_EQ(loop_isub, 1u);
      EXPECT_EQ(loop_ineg, 0u);
      EXPECT_EQ(loop_iand7, 0u);
      ralloc_free(b.shader);
   }
}


TEST(AGX3VisionOpts, Agx3bDirectByteFromWordU2f32)
{
   auto compile_unpack = [](enum agx_isa_encoding enc, enum agx_chip chip,
                            bool emit_agx3d) {
      nir_builder b = make_compute_shader("u2f32-rgba8-unpack");
      nir_def *in = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *w = nir_load_agx(
         &b, 1, 32, in, gid,
         .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                              ACCESS_CAN_REORDER),
         .base = 0, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
      nir_def *b0 = nir_u2f32(&b, nir_iand_imm(&b, w, 0xff));
      nir_def *b1 = nir_u2f32(&b, nir_ubitfield_extract_imm(&b, w, 8, 8));
      nir_def *b2 = nir_u2f32(&b, nir_ubitfield_extract_imm(&b, w, 16, 8));
      nir_def *b3 = nir_u2f32(&b, nir_ushr_imm(&b, w, 24));
      nir_def *sum = nir_fadd(&b, nir_fadd(&b, b0, b1), nir_fadd(&b, b2, b3));
      nir_store_agx(&b, sum, out, gid,
                    .access = ACCESS_NON_READABLE, .base = 0,
                    .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {};
      key.dev.encoding = enc;
      key.dev.chip = chip;
      key.dev.emit_agx3d = emit_agx3d;
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_EQ(part.info.stats.spills, 0u);
      unsigned instrs = part.info.stats.instrs;
      free(part.binary);
      ralloc_free(b.shader);
      return instrs;
   };

   /* AGX3B (G15/G16/G17P) matches AGX3C (G17S) direct byte-from-word unpack,
    * avoiding extra shift/mask ALU instructions needed on AGX2 and G18G. */
   unsigned agx3b_instrs =
      compile_unpack(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false);
   unsigned agx3c_instrs =
      compile_unpack(AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false);
   unsigned agx2_instrs =
      compile_unpack(AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false);
   EXPECT_EQ(agx3b_instrs, agx3c_instrs);
   EXPECT_LT(agx3b_instrs, agx2_instrs);
}

TEST(AGX3VisionOpts, HoistRuntimeUnrollGuardBound)
{
   auto count_entry_bcsels = [](nir_shader *shader) {
      unsigned n = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_block *entry = nir_start_block(impl);
      nir_foreach_instr(instr, entry) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_bcsel)
            n++;
      }
      return n;
   };

   /* Positive control: inner counted loop nested inside two outer loops with
    * an outer-invariant bound hoists guard_bound() into entry before both
    * outer loops. */
   {
      nir_builder b = make_compute_shader("hoist-guard-bound-2d");
      nir_def *in = make_desc_base(&b, 4);
      nir_def *wbuf = make_desc_base(&b, 8);
      nir_def *out = make_desc_base(&b, 12);
      nir_def *h = nir_load_preamble(&b, 1, 32, .base = 16);
      nir_def *w = nir_load_preamble(&b, 1, 32, .base = 20);
      nir_def *klen = nir_load_preamble(&b, 1, 32, .base = 24);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, y_var, nir_imm_int(&b, 0), 1);

      nir_loop *y_loop = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_break_if(&b, nir_ige(&b, y, h));
         nir_store_var(&b, x_var, nir_imm_int(&b, 0), 1);
         nir_loop *x_loop = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_break_if(&b, nir_ige(&b, x, w));
            nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);
            nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);
            nir_loop *k_loop = nir_push_loop(&b);
            {
               nir_def *k = nir_load_var(&b, k_var);
               nir_break_if(&b, nir_ige(&b, k, klen));
               nir_def *idx = nir_iadd(&b, nir_iadd(&b, y, x), k);
               nir_def *p = nir_load_agx(
                  &b, 1, 32, in, idx,
                  .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                  .base = 0, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
               nir_def *wt = nir_load_agx(
                  &b, 1, 32, wbuf, k,
                  .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                  .base = 0, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
               nir_store_var(&b, acc_var,
                             nir_ffma(&b, nir_u2f32(&b, p), wt, nir_load_var(&b, acc_var)), 1);
               nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
            }
            nir_pop_loop(&b, k_loop);
            nir_store_agx(&b, nir_load_var(&b, acc_var), out, nir_iadd(&b, y, x),
                          .access = ACCESS_NON_READABLE, .base = 0,
                          .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, x_loop);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, y_loop);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);

      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, &widest));
      nir_validate_shader(b.shader, "hoist guard bound 2d");
      EXPECT_EQ(widest, 8u);
      EXPECT_GE(count_entry_bcsels(b.shader), 1u);
      ralloc_free(b.shader);
   }

   /* Negative control: inner bound depends on the outer loop IV, so
    * guard_bound() stays inside the outer loop and does not hoist to entry. */
   {
      nir_builder b = make_compute_shader("hoist-guard-bound-variant");
      nir_def *in = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *h = nir_load_preamble(&b, 1, 32, .base = 16);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *k_var = nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_int_type(), "acc");
      nir_store_var(&b, y_var, nir_imm_int(&b, 0), 1);

      nir_loop *y_loop = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_break_if(&b, nir_ige(&b, y, h));
         nir_def *bound = nir_iadd_imm(&b, y, 16);
         nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);
         nir_loop *k_loop = nir_push_loop(&b);
         {
            nir_def *k = nir_load_var(&b, k_var);
            nir_break_if(&b, nir_ige(&b, k, bound));
            nir_def *v = nir_load_agx(
               &b, 1, 32, in, k,
               .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
               .base = 0, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
            nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), v), 1);
            nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
         }
         nir_pop_loop(&b, k_loop);
         nir_store_agx(&b, nir_load_var(&b, acc_var), out, y,
                       .access = ACCESS_NON_READABLE, .base = 0,
                       .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, y_loop);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);

      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, NULL));
      nir_validate_shader(b.shader, "hoist guard bound variant");
      EXPECT_EQ(count_entry_bcsels(b.shader), 0u);
      ralloc_free(b.shader);
   }
}

namespace {

struct NirExecResult {
   uint32_t out_sum = 0;
   uint32_t iters = 0;
   bool hit_step_limit = false;
   bool faulted = false;
};

/* Minimal structured-CF NIR interpreter for testing runtime loop unrolling
 * across arbitrary runtime inputs (push_constant[0] = start, [4] = bound). */
static NirExecResult
exec_loop_nir(nir_shader *shader, uint32_t start_val, uint32_t bound_val,
              unsigned max_iters = 512)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   std::unordered_map<nir_def *, uint64_t> vals;
   std::unordered_map<nir_def *, uint64_t> regs;
   NirExecResult res;
   nir_block *prev_block = NULL;

   auto get_val = [&](nir_def *d) -> uint64_t {
      auto it = vals.find(d);
      EXPECT_NE(it, vals.end()) << "missing SSA value";
      return it != vals.end() ? it->second : 0;
   };

   std::function<bool(struct exec_list *)> run_list =
      [&](struct exec_list *list) -> bool {
      foreach_list_typed(nir_cf_node, node, node, list) {
         if (res.hit_step_limit || res.faulted)
            return true;
         switch (node->type) {
         case nir_cf_node_block: {
            nir_block *blk = nir_cf_node_as_block(node);
            std::vector<std::pair<nir_def *, uint64_t>> phi_updates;
            nir_foreach_instr(instr, blk) {
               if (instr->type == nir_instr_type_phi) {
                  nir_phi_instr *phi = nir_instr_as_phi(instr);
                  nir_phi_src *ps = nir_phi_get_src_from_block(phi, prev_block);
                  EXPECT_NE(ps, nullptr);
                  if (ps)
                     phi_updates.push_back({&phi->def, get_val(ps->src.ssa)});
                  continue;
               }
               if (!phi_updates.empty()) {
                  for (auto &kv : phi_updates)
                     vals[kv.first] = kv.second;
                  phi_updates.clear();
               }
               switch (instr->type) {
               case nir_instr_type_load_const: {
                  nir_load_const_instr *lc = nir_instr_as_load_const(instr);
                  vals[&lc->def] = nir_const_value_as_uint(lc->value[0], lc->def.bit_size);
                  break;
               }
               case nir_instr_type_undef: {
                  nir_undef_instr *u = nir_instr_as_undef(instr);
                  vals[&u->def] = 0;
                  break;
               }
               case nir_instr_type_alu: {
                  nir_alu_instr *alu = nir_instr_as_alu(instr);
                  const nir_op_info *info = &nir_op_infos[alu->op];
                  nir_const_value src_arr[4][NIR_MAX_VEC_COMPONENTS] = {};
                  nir_const_value *srcs[4];
                  for (unsigned i = 0; i < info->num_inputs; ++i) {
                     unsigned bs = alu->src[i].src.ssa->bit_size;
                     src_arr[i][0] = nir_const_value_for_uint(get_val(alu->src[i].src.ssa), bs);
                     srcs[i] = src_arr[i];
                  }
                  unsigned eval_bs = 0;
                  if (!nir_alu_type_get_type_size(info->output_type)) {
                     eval_bs = alu->def.bit_size;
                  } else {
                     for (unsigned i = 0; i < info->num_inputs; ++i) {
                        if (!nir_alu_type_get_type_size(info->input_types[i])) {
                           eval_bs = alu->src[i].src.ssa->bit_size;
                           break;
                        }
                     }
                     if (!eval_bs)
                        eval_bs = alu->def.bit_size;
                  }
                  nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
                  uint16_t poison = 0;
                  nir_eval_const_opcode(alu->op, dest, &poison, 1, eval_bs, srcs, 0);
                  vals[&alu->def] = nir_const_value_as_uint(dest[0], alu->def.bit_size);
                  break;
               }
               case nir_instr_type_intrinsic: {
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  switch (intr->intrinsic) {
                  case nir_intrinsic_load_push_constant: {
                     uint32_t off = (uint32_t)get_val(intr->src[0].ssa) + nir_intrinsic_base(intr);
                     vals[&intr->def] = (off == 0) ? start_val : bound_val;
                     break;
                  }
                  case nir_intrinsic_load_preamble: {
                     vals[&intr->def] = 0;
                     break;
                  }
                  case nir_intrinsic_decl_reg: {
                     regs[&intr->def] = 0;
                     break;
                  }
                  case nir_intrinsic_load_reg: {
                     vals[&intr->def] = regs[intr->src[0].ssa];
                     break;
                  }
                  case nir_intrinsic_store_reg: {
                     regs[intr->src[1].ssa] = get_val(intr->src[0].ssa);
                     break;
                  }
                  case nir_intrinsic_load_agx: {
                     /* Deterministic per-index payload: (idx * 2654435761u) ^ 0x5a5a5a5au. */
                     uint32_t idx = (uint32_t)get_val(intr->src[1].ssa);
                     res.iters++;
                     if (res.iters > max_iters) {
                        res.hit_step_limit = true;
                        return true;
                     }
                     uint32_t v = (idx * 2654435761u) ^ 0x5a5a5a5au;
                     vals[&intr->def] = intr->def.bit_size == 8 ? (v & 0xffu) : v;
                     break;
                  }
                  case nir_intrinsic_store_agx: {
                     res.out_sum = (uint32_t)get_val(intr->src[0].ssa);
                     break;
                  }
                  default:
                     ADD_FAILURE() << "unexpected intrinsic " << nir_intrinsic_infos[intr->intrinsic].name;
                     break;
                  }
                  break;
               }
               case nir_instr_type_jump: {
                  nir_jump_instr *j = nir_instr_as_jump(instr);
                  EXPECT_EQ(j->type, nir_jump_break);
                  prev_block = blk;
                  return true;
               }
               default:
                  ADD_FAILURE() << "unexpected instr type";
                  break;
               }
            }
            for (auto &kv : phi_updates)
               vals[kv.first] = kv.second;
            prev_block = blk;
            break;
         }
         case nir_cf_node_if: {
            nir_if *nif = nir_cf_node_as_if(node);
            bool cond = (get_val(nif->condition.ssa) & 1u) != 0;
            bool broke = run_list(cond ? &nif->then_list : &nif->else_list);
            if (broke)
               return true;
            break;
         }
         case nir_cf_node_loop: {
            nir_loop *lp = nir_cf_node_as_loop(node);
            while (!res.hit_step_limit) {
               if (run_list(&lp->body))
                  break;
            }
            break;
         }
         default:
            break;
         }
      }
      return false;
   };

   run_list(&impl->body);
   return res;
}

static nir_shader *
build_counted_sum_shader(bool is_unsigned, bool inclusive, int step,
                         bool bottom_tested, bool const_bound = false,
                         uint32_t bound_imm = 0)
{
   nir_builder b = make_compute_shader("bin-tail-exec");
   nir_def *in = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *out = nir_load_preamble(&b, 1, 64, .base = 8);
   nir_def *start = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *bound = const_bound
                       ? nir_imm_int(&b, bound_imm)
                       : nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);

   nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, i_var, start, 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   auto make_break_cond = [&](nir_def *v) -> nir_def * {
      if (step > 0) {
         if (inclusive)
            return is_unsigned ? nir_ult(&b, bound, v) : nir_ilt(&b, bound, v);
         return is_unsigned ? nir_uge(&b, v, bound) : nir_ige(&b, v, bound);
      } else {
         if (inclusive)
            return is_unsigned ? nir_ult(&b, v, bound) : nir_ilt(&b, v, bound);
         return is_unsigned ? nir_uge(&b, bound, v) : nir_ige(&b, bound, v);
      }
   };

   nir_loop *lp = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, i_var);
      if (!bottom_tested)
         nir_break_if(&b, make_break_cond(i));
      nir_def *v = nir_load_agx(
         &b, 1, 32, in, i,
         .access = (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .base = 0, .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);
      nir_def *next_acc = nir_iadd(&b, nir_imul_imm(&b, nir_load_var(&b, acc_var), 3), v);
      nir_def *next_i = nir_iadd_imm(&b, i, step);
      nir_store_var(&b, acc_var, next_acc, 1);
      nir_store_var(&b, i_var, next_i, 1);
      if (bottom_tested)
         nir_break_if(&b, make_break_cond(next_i));
   }
   nir_pop_loop(&b, lp);

   nir_store_agx(&b, nir_load_var(&b, acc_var), out, nir_imm_int(&b, 0),
                 .access = ACCESS_NON_READABLE, .base = 0,
                 .format = PIPE_FORMAT_R32_UINT, .sign_extend = false);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   return b.shader;
}

} /* namespace */

TEST(AGX3VisionOpts, BinaryTailPeelShapeAndGuards)
{
   /* Top-tested REL_LT: wrap is impossible, so binary-peeling eliminates the
    * scalar remainder loop completely (1 loop total). */
   {
      nir_shader *s = build_counted_sum_shader(false, false, 1, false);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 8, 320, NULL));
      nir_validate_shader(s, "binary tail top lt");
      EXPECT_EQ(count_loops(s), 1u);
      ralloc_free(s);
   }

   /* Top-tested REL_LE with non-max constant bound: wrap is impossible, so
    * the scalar remainder loop is eliminated (1 loop total). */
   {
      nir_shader *s = build_counted_sum_shader(true, true, 1, false, true, 100u);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 8, 320, NULL));
      nir_validate_shader(s, "binary tail const non-max le");
      EXPECT_EQ(count_loops(s), 1u);
      ralloc_free(s);
   }

   /* Top-tested REL_LE with runtime bound keeps a guarded wrap fallback loop
    * after the 4/2/1 tail so bound == max still wraps like the original. */
   {
      nir_shader *s = build_counted_sum_shader(true, true, 1, false);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 8, 320, NULL));
      nir_validate_shader(s, "binary tail runtime le wrap guard");
      EXPECT_EQ(count_loops(s), 2u);
      ralloc_free(s);
   }

   /* Negative control: factor=4 does not binary-peel the tail. */
   {
      nir_shader *s = build_counted_sum_shader(false, false, 1, false);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 320, NULL));
      nir_validate_shader(s, "factor4 keeps remainder");
      EXPECT_EQ(count_loops(s), 2u);
      ralloc_free(s);
   }
}

TEST(AGX3VisionOpts, BinaryTailPeelExecution)
{
   const uint32_t trips[] = {0, 1, 3, 7, 8, 9, 15, 16, 100};

   for (bool is_unsigned : {false, true}) {
      for (bool inclusive : {false, true}) {
         for (int step : {1, -1}) {
            for (bool bottom : {false, true}) {
               nir_shader *ref = build_counted_sum_shader(is_unsigned, inclusive, step, bottom);
               nir_shader *opt = nir_shader_clone(NULL, ref);
               ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL))
                  << "u=" << is_unsigned << " inc=" << inclusive
                  << " step=" << step << " bot=" << bottom;
               nir_validate_shader(opt, "binary tail exec");

               for (uint32_t n : trips) {
                  uint32_t start, bound;
                  if (step > 0) {
                     start = is_unsigned ? 10u : (uint32_t)(-5);
                     if (n == 0) {
                        if (bottom)
                           continue; /* bottom-tested loops always run >= 1 iter */
                        bound = inclusive ? (start - 1u) : start;
                     } else {
                        bound = inclusive ? (start + n - 1u) : (start + n);
                     }
                  } else {
                     start = is_unsigned ? 200u : 50u;
                     if (n == 0) {
                        if (bottom)
                           continue;
                        bound = inclusive ? (start + 1u) : start;
                     } else {
                        bound = inclusive ? (start - n + 1u) : (start - n);
                     }
                  }

                  NirExecResult r_ref = exec_loop_nir(ref, start, bound);
                  NirExecResult r_opt = exec_loop_nir(opt, start, bound);
                  EXPECT_FALSE(r_ref.hit_step_limit);
                  EXPECT_FALSE(r_opt.hit_step_limit);
                  EXPECT_EQ(r_opt.iters, r_ref.iters)
                     << "u=" << is_unsigned << " inc=" << inclusive
                     << " step=" << step << " bot=" << bottom << " n=" << n;
                  EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
                     << "u=" << is_unsigned << " inc=" << inclusive
                     << " step=" << step << " bot=" << bottom << " n=" << n;
               }
               ralloc_free(ref);
               ralloc_free(opt);
            }
         }
      }
   }

   /* Wrap boundary 1: top-tested unsigned REL_LE at bound == UINT32_MAX
    * starting at 0xfffffff9 must wrap past 0xffffffff to 0 and keep running
    * (not terminate after 7 tail iterations). */
   {
      nir_shader *ref = build_counted_sum_shader(true, true, 1, false);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "wrap u32 le max");
      NirExecResult r_ref = exec_loop_nir(ref, 0xfffffff9u, 0xffffffffu, 20);
      NirExecResult r_opt = exec_loop_nir(opt, 0xfffffff9u, 0xffffffffu, 20);
      EXPECT_TRUE(r_ref.hit_step_limit);
      EXPECT_TRUE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
      ralloc_free(ref);
      ralloc_free(opt);
   }

   /* Wrap boundary 2: top-tested signed REL_GE at bound == INT32_MIN starting
    * at INT32_MIN + 6 must wrap past INT32_MIN to INT32_MAX and keep running. */
   {
      nir_shader *ref = build_counted_sum_shader(false, true, -1, false);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "wrap s32 ge min");
      uint32_t start = (uint32_t)(INT32_MIN + 6);
      uint32_t bound = (uint32_t)INT32_MIN;
      NirExecResult r_ref = exec_loop_nir(ref, start, bound, 20);
      NirExecResult r_opt = exec_loop_nir(opt, start, bound, 20);
      EXPECT_TRUE(r_ref.hit_step_limit);
      EXPECT_TRUE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
      ralloc_free(ref);
      ralloc_free(opt);
   }

   /* Constant non-extreme bounds (scalar remainder loop eliminated, 1 loop
    * total): execute top-tested REL_LE/REL_GE including small bounds near 0
    * and starts near 0 / INT32_MIN / INT32_MAX that hit the guard `never` arm. */
   struct ConstCase {
      bool is_unsigned;
      int step;
      uint32_t bound;
   };
   const ConstCase const_cases[] = {
      {true, 1, 100u},                   /* unsigned LE const bound 100 */
      {true, 1, 2u},                     /* unsigned LE small bound (hits 4-chunk never) */
      {false, -1, (uint32_t)(-5)},       /* signed GE const bound -5 */
      {false, -1, (uint32_t)(INT32_MAX - 2)}, /* signed GE near INT32_MAX (hits 4-chunk never) */
      {false, 1, 100u},                  /* signed LE const bound 100 */
      {true, -1, 1u},                    /* unsigned GE near 0 */
   };
   const uint32_t const_trips[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16};
   for (const auto &cc : const_cases) {
      nir_shader *ref = build_counted_sum_shader(cc.is_unsigned, true, cc.step,
                                                 false, true, cc.bound);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "binary tail const bound exec");
      EXPECT_EQ(count_loops(opt), 1u);
      for (uint32_t n : const_trips) {
         if (cc.is_unsigned && cc.step > 0 && n > cc.bound + 1u)
            continue;
         if (!cc.is_unsigned && cc.step < 0 &&
             (int64_t)(int32_t)cc.bound + (int64_t)n - 1 > INT32_MAX)
            continue;
         uint32_t start;
         if (n == 0) {
            if ((cc.is_unsigned && cc.step > 0 && cc.bound == UINT32_MAX) ||
                (cc.is_unsigned && cc.step < 0 && cc.bound == 0u))
               continue;
            start = cc.step > 0 ? (cc.bound + 1u) : (cc.bound - 1u);
         } else {
            start = cc.step > 0 ? (cc.bound - n + 1u) : (cc.bound + n - 1u);
         }
         NirExecResult r_ref = exec_loop_nir(ref, start, cc.bound);
         NirExecResult r_opt = exec_loop_nir(opt, start, cc.bound);
         EXPECT_FALSE(r_ref.hit_step_limit);
         EXPECT_FALSE(r_opt.hit_step_limit);
         EXPECT_EQ(r_opt.iters, r_ref.iters)
            << "const u=" << cc.is_unsigned << " step=" << cc.step
            << " bound=" << cc.bound << " n=" << n;
         EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
            << "const u=" << cc.is_unsigned << " step=" << cc.step
            << " bound=" << cc.bound << " n=" << n;
      }
      ralloc_free(ref);
      ralloc_free(opt);
   }

   /* Starts near 0 / INT32_MIN / INT32_MAX with trip counts 1..7, step +1/-1
    * (exercises the guard_bound `never` arm on 4- and 2-chunk guards). */
   for (bool is_unsigned : {false, true}) {
      for (bool inclusive : {false, true}) {
         for (int step : {1, -1}) {
            nir_shader *ref = build_counted_sum_shader(is_unsigned, inclusive,
                                                       step, false);
            nir_shader *opt = nir_shader_clone(NULL, ref);
            ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
            nir_validate_shader(opt, "binary tail near-limit exec");
            const uint32_t starts[] = {
               0u,
               step > 0 ? (is_unsigned ? 0u : (uint32_t)INT32_MIN)
                        : (is_unsigned ? 6u : (uint32_t)INT32_MAX),
            };
            for (uint32_t base_start : starts) {
               for (uint32_t n = 1; n <= 7; ++n) {
                  if (is_unsigned && step < 0 && base_start < n)
                     continue;
                  uint32_t bound =
                     step > 0 ? (inclusive ? (base_start + n - 1u) : (base_start + n))
                              : (inclusive ? (base_start - n + 1u) : (base_start - n));
                  NirExecResult r_ref = exec_loop_nir(ref, base_start, bound);
                  NirExecResult r_opt = exec_loop_nir(opt, base_start, bound);
                  EXPECT_FALSE(r_ref.hit_step_limit);
                  EXPECT_FALSE(r_opt.hit_step_limit);
                  EXPECT_EQ(r_opt.iters, r_ref.iters)
                     << "near u=" << is_unsigned << " inc=" << inclusive
                     << " step=" << step << " start=" << base_start << " n=" << n;
                  EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
                     << "near u=" << is_unsigned << " inc=" << inclusive
                     << " step=" << step << " start=" << base_start << " n=" << n;
               }
            }
            ralloc_free(ref);
            ralloc_free(opt);
         }
      }
   }

   /* Wrap boundary 3: bottom-tested exact_tail starting at UINT32_MAX with
    * bound == 5 (REL_LT) wraps to 0 on iteration 0 and must execute 0..4. */
   {
      nir_shader *ref = build_counted_sum_shader(true, false, 1, true);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "wrap exact_tail u32");
      NirExecResult r_ref = exec_loop_nir(ref, 0xffffffffu, 5u, 64);
      NirExecResult r_opt = exec_loop_nir(opt, 0xffffffffu, 5u, 64);
      EXPECT_FALSE(r_ref.hit_step_limit);
      EXPECT_FALSE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.iters, 6u);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum);
      ralloc_free(ref);
      ralloc_free(opt);
   }
}

TEST(AGX3VisionOpts, Agx3bComputeDeviceLoadIlpSchedule)
{
   auto run_sched = [](enum agx_isa_encoding enc, enum agx_chip chip,
                       unsigned extra_live_32) -> unsigned {
      void *mem = ralloc_context(NULL);
      struct agx_shader_key key = {};
      key.dev.encoding = enc;
      key.dev.chip = chip;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *body = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      body->loop_header = true;
      agx_block_add_successor(entry, body);
      agx_block_add_successor(body, body);
      agx_block_add_successor(body, exit);

      agx_index base = agx_uniform(4, AGX_SIZE_64);
      agx_index off = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index held[100];
      assert(extra_live_32 <= 100);
      for (unsigned i = 0; i < extra_live_32; ++i)
         held[i] = agx_mov_imm(b, 32, 100 + i);

      /* Emit 4 independent device_loads in pressure order (load, immediate use). */
      b->cursor = agx_after_block(body);
      for (unsigned i = 0; i < 4; ++i) {
         agx_index addr_off = agx_iadd(b, off, agx_immediate(i * 4), 0);
         agx_index val = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *ld = agx_device_load_to(
            b, val, base, addr_off, AGX_FORMAT_I32, 0x1, 0, 0);
         ld->memory_can_reorder = true;
         acc = agx_fadd(b, acc, val);
      }
      for (unsigned i = 0; i < extra_live_32; ++i)
         acc = agx_iadd(b, acc, held[i], 0);
      agx_device_store(b, acc, agx_uniform(8, AGX_SIZE_64), off, AGX_FORMAT_I32,
                       0x1, 0, 0);

      agx_ilp_schedule(b->shader);

      unsigned loads_before_first_fadd = 0;
      agx_foreach_instr_in_block(body, I) {
         if (I->op == AGX_OPCODE_FADD)
            break;
         if (I->op == AGX_OPCODE_DEVICE_LOAD)
            loads_before_first_fadd++;
      }
      ralloc_free(mem);
      return loads_before_first_fadd;
   };

   /* Positive control: AGX3B compute with headroom batches loads before fadd. */
   EXPECT_GE(run_sched(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, 0), 2u);
   /* Negative control 1: High register pressure (near 104-half occupancy cap)
    * keeps pressure-minimizing order (1 load before first fadd). */
   EXPECT_EQ(run_sched(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, 45), 1u);
   /* Negative control 2: Near AGX3B 192-half general file limit (program_peak
    * ~190 halves, in 208-half step >= AGX3B_NUM_REGS) refuses load batching. */
   EXPECT_EQ(run_sched(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, 93), 1u);
   /* Negative control 3: AGX2 does not take the AGX3B compute load path. */
   EXPECT_EQ(run_sched(AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, 0), 1u);
}

static nir_def *
push_u32(nir_builder *b, unsigned offset)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset), .range = 16);
}

TEST(AGX3VisionOpts, ExactTailConstInitOmitsDeadFallback)
{
   auto build_bottom_const_init = [](bool is_unsigned, int32_t init_val) -> nir_shader * {
      nir_builder b = make_compute_shader("exact_tail_const_init");
      nir_def *bound = push_u32(&b, 4);
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);

      nir_variable *iv_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, iv_var, nir_imm_int(&b, (uint32_t)init_val), 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *i = nir_load_var(&b, iv_var);
      nir_def *acc = nir_load_var(&b, acc_var);
      nir_def *v = nir_load_agx(
         &b, 1, 32, base, i,
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .format = PIPE_FORMAT_R32_UINT);
      nir_def *next_acc = nir_iadd(&b, acc, v);
      nir_def *next_i = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, acc_var, next_acc, 1);
      nir_store_var(&b, iv_var, next_i, 1);
      nir_def *exit_cond =
         is_unsigned ? nir_uge(&b, next_i, bound) : nir_ige(&b, next_i, bound);
      nir_if *ex = nir_push_if(&b, exit_cond);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, ex);
      nir_pop_loop(&b, loop);

      nir_store_agx(&b, nir_load_var(&b, acc_var), out, nir_imm_int(&b, 0),
                    .access = ACCESS_NON_READABLE, .format = PIPE_FORMAT_R32_UINT);
      nir_lower_vars_to_ssa(b.shader);
      nir_opt_copy_prop(b.shader);
      nir_opt_dce(b.shader);
      return b.shader;
   };

   /* Positive control: constant non-extreme init (0) with REL_LT omits the dead
    * wrap fallback loop (leaving only the 8x main loop) and matches reference
    * execution across trip counts 0 (1 do-while iter) through 17 plus INT32_MIN. */
   for (bool is_unsigned : {false, true}) {
      nir_shader *ref = build_bottom_const_init(is_unsigned, 0);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "exact_tail const init 0");
      EXPECT_EQ(count_loops(opt), 1u) << "u=" << is_unsigned;

      const uint32_t bounds[] = {0u, 1u, 2u, 3u, 4u, 7u, 8u, 9u, 15u, 16u, 17u,
                                 0x80000000u};
      for (uint32_t bound : bounds) {
         if (is_unsigned && bound == 0x80000000u)
            continue;
         NirExecResult r_ref = exec_loop_nir(ref, 0u, bound, 128);
         NirExecResult r_opt = exec_loop_nir(opt, 0u, bound, 128);
         EXPECT_FALSE(r_ref.hit_step_limit);
         EXPECT_FALSE(r_opt.hit_step_limit);
         EXPECT_EQ(r_opt.iters, r_ref.iters) << "u=" << is_unsigned << " b=" << bound;
         EXPECT_EQ(r_opt.out_sum, r_ref.out_sum) << "u=" << is_unsigned << " b=" << bound;
      }
      ralloc_free(ref);
      ralloc_free(opt);
   }

   /* Negative control: constant extreme init (INT32_MAX / UINT32_MAX) MUST keep
    * the wrap fallback loop (2 loops) so iteration 0 wrapping is preserved. */
   for (bool is_unsigned : {false, true}) {
      int32_t extreme = is_unsigned ? (int32_t)UINT32_MAX : INT32_MAX;
      nir_shader *ref = build_bottom_const_init(is_unsigned, extreme);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, NULL));
      nir_validate_shader(opt, "exact_tail extreme init");
      EXPECT_EQ(count_loops(opt), 2u) << "u=" << is_unsigned;

      uint32_t wrap_bound = is_unsigned ? 5u : (uint32_t)(INT32_MIN + 5);
      NirExecResult r_ref = exec_loop_nir(ref, (uint32_t)extreme, wrap_bound, 64);
      NirExecResult r_opt = exec_loop_nir(opt, (uint32_t)extreme, wrap_bound, 64);
      EXPECT_EQ(r_opt.iters, 6u);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum);
      ralloc_free(ref);
      ralloc_free(opt);
   }
}

TEST(AGX3VisionOpts, UniformDmaNoPreambleFoldsShortCircuitIf)
{
   /* When !form_preamble (AGX3B compute) lowers a uniform-DMA load inside a
    * short-circuit branch into load_preamble, post-DMA peephole select must
    * fold the now-trivial branch into a select rather than leaving control
    * flow around load_preamble. */
   auto compile_once = [](enum agx_chip chip, bool dont_flatten) {
      nir_builder b = make_compute_shader("uniform_dma_peephole");
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);

      nir_if *nif = nir_push_if(&b, nir_ige_imm(&b, gid, 0));
      if (dont_flatten)
         nif->control = nir_selection_control_dont_flatten;
      nir_def *limit = nir_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, root, 8), .align_mul = 4);
      nir_def *oob = nir_ige(&b, gid, limit);
      nir_push_else(&b, nif);
      nir_def *t = nir_imm_true(&b);
      nir_pop_if(&b, nif);

      nir_def *res = nir_b2i32(&b, nir_if_phi(&b, oob, t));
      nir_store_global(&b, res,
                       nir_iadd(&b, out, nir_imul_imm(&b, nir_u2u64(&b, gid), 4)),
                       .align_mul = 4);
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      key.reserved_preamble = 8;
      key.no_preamble = true;
      key.g17p_uniform_dma = true;
      key.g17p_dma_root = 0;
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.agx3_uniform_dma_count, 0u);
      std::vector<uint8_t> bin;
      if (part.binary)
         bin.assign((const uint8_t *)part.binary,
                    (const uint8_t *)part.binary + part.info.binary_size);
      free(part.binary);
      ralloc_free(b.shader);
      return bin;
   };

   for (enum agx_chip chip : {AGX_CHIP_G15G, AGX_CHIP_G17P}) {
      const std::vector<uint8_t> folded = compile_once(chip, false);
      const std::vector<uint8_t> kept = compile_once(chip, true);
      EXPECT_FALSE(folded.empty());
      EXPECT_FALSE(kept.empty());
      EXPECT_NE(folded, kept);
      EXPECT_LT(folded.size(), kept.size());
   }
}

TEST(AGX3VisionOpts, UniformDmaNoPreamblePreservesScalarizedVecStore)
{
   /* Post-uniform-DMA peephole select and constant folding can introduce vector
    * bcsel or re-fold vec4 load_const values after agx_optimize_nir's final
    * scalarization pass. Ensure the !form_preamble uniform-DMA path re-runs
    * scalarization so vector SSBO/global stores never hit missing allocated_vec
    * entries in agx_recollect_vector. */
   nir_builder b = make_compute_shader("uniform_dma_vec4_store");
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_if *nif = nir_push_if(&b, nir_ige_imm(&b, gid, 0));
   nir_def *limit = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, root, 8), .align_mul = 4);
   nir_def *v_then = nir_vec4(&b, limit, nir_imm_int(&b, 2),
                              nir_imm_int(&b, 3), nir_imm_int(&b, 4));
   nir_push_else(&b, nif);
   nir_def *v_else = nir_imm_ivec4(&b, 10, 20, 30, 40);
   nir_pop_if(&b, nif);

   nir_def *sel = nir_if_phi(&b, v_then, v_else);
   nir_def *cvec = nir_imm_ivec4(&b, 1, 2, 3, 4);
   nir_def *addr0 = nir_iadd(&b, out, nir_imul_imm(&b, nir_u2u64(&b, gid), 32));
   nir_def *addr1 = nir_iadd_imm(&b, addr0, 16);
   nir_store_global(&b, sel, addr0, .align_mul = 16);
   nir_store_global(&b, cvec, addr1, .align_mul = 16);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.reserved_preamble = 8;
   key.no_preamble = true;
   key.g17p_uniform_dma = true;
   key.g17p_dma_root = 0;
   struct agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.agx3_uniform_dma_count, 0u);
   EXPECT_GT(part.info.binary_size, 0u);
   free(part.binary);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, BinaryTailChunksSkippedLocallyWithoutDontFlatten)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "binary_tail_local_skip");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *n = push_u32(&b, 0);
   nir_def *src_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 4), push_u32(&b, 8));
   nir_def *dst_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 12), push_u32(&b, 16));

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_int_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_if *brk = nir_push_if(&b, nir_ige(&b, i, n));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, brk);

      nir_def *off = nir_u2u64(&b, nir_imul_imm(&b, nir_iadd(&b, gid, i), 4));
      nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, src_addr, off), .align_mul = 4);
      nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);

   nir_def *out_off = nir_u2u64(&b, nir_imul_imm(&b, gid, 4));
   nir_store_global(&b, nir_load_var(&b, acc), nir_iadd(&b, dst_addr, out_off), .align_mul = 4);

   nir_lower_vars_to_ssa(b.shader);
   nir_opt_constant_folding(b.shader);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);

   ASSERT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, NULL));
   nir_validate_shader(b.shader, "after unroll");

   unsigned tail_ifs = 0;
   unsigned dont_flatten_ifs = 0;
   foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         tail_ifs++;
         if (nif->control == nir_selection_control_dont_flatten)
            dont_flatten_ifs++;
      }
   }
   EXPECT_GE(tail_ifs, 3u);
   EXPECT_EQ(dont_flatten_ifs, 0u);
   EXPECT_FALSE(agx_nir_opt_flatten_loads(b.shader, 48, 4));

   ralloc_free(b.shader);
}

/* Generalized NIR interpreter for interior bounds unswitching tests:
 * reads 32-bit inputs from push_constant[0, 4, 8, 12, ...]. */
static NirExecResult
exec_stencil_nir(nir_shader *shader, const std::vector<uint32_t> &pc,
                 unsigned max_iters = 512,
                 const std::unordered_map<uint64_t, uint32_t> *const_mem = nullptr)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   std::unordered_map<nir_def *, uint64_t> vals;
   std::unordered_map<nir_def *, uint64_t> regs;
   NirExecResult res;
   nir_block *prev_block = NULL;

   auto get_val = [&](nir_def *d) -> uint64_t {
      auto it = vals.find(d);
      EXPECT_NE(it, vals.end()) << "missing SSA value";
      return it != vals.end() ? it->second : 0;
   };

   std::function<bool(struct exec_list *)> run_list =
      [&](struct exec_list *list) -> bool {
      foreach_list_typed(nir_cf_node, node, node, list) {
         if (res.hit_step_limit)
            return true;
         switch (node->type) {
         case nir_cf_node_block: {
            nir_block *blk = nir_cf_node_as_block(node);
            std::vector<std::pair<nir_def *, uint64_t>> phi_updates;
            nir_foreach_instr(instr, blk) {
               if (instr->type == nir_instr_type_phi) {
                  nir_phi_instr *phi = nir_instr_as_phi(instr);
                  nir_phi_src *ps = nir_phi_get_src_from_block(phi, prev_block);
                  EXPECT_NE(ps, nullptr);
                  if (ps)
                     phi_updates.push_back({&phi->def, get_val(ps->src.ssa)});
                  continue;
               }
               if (!phi_updates.empty()) {
                  for (auto &kv : phi_updates)
                     vals[kv.first] = kv.second;
                  phi_updates.clear();
               }
               switch (instr->type) {
               case nir_instr_type_load_const: {
                  nir_load_const_instr *lc = nir_instr_as_load_const(instr);
                  vals[&lc->def] = nir_const_value_as_uint(lc->value[0], lc->def.bit_size);
                  break;
               }
               case nir_instr_type_undef: {
                  nir_undef_instr *u = nir_instr_as_undef(instr);
                  vals[&u->def] = 0;
                  break;
               }
               case nir_instr_type_alu: {
                  nir_alu_instr *alu = nir_instr_as_alu(instr);
                  if (alu->op == nir_op_imadshl_agx) {
                     uint32_t a = (uint32_t)get_val(alu->src[0].src.ssa);
                     uint32_t b = (uint32_t)get_val(alu->src[1].src.ssa);
                     uint32_t c = (uint32_t)get_val(alu->src[2].src.ssa);
                     uint32_t sh = (uint32_t)get_val(alu->src[3].src.ssa);
                     vals[&alu->def] = (uint32_t)((uint32_t)(a * b) + (c << sh));
                     break;
                  }
                  const nir_op_info *info = &nir_op_infos[alu->op];
                  nir_const_value src_arr[4][NIR_MAX_VEC_COMPONENTS] = {};
                  nir_const_value *srcs[4];
                  for (unsigned i = 0; i < info->num_inputs; ++i) {
                     unsigned bs = alu->src[i].src.ssa->bit_size;
                     src_arr[i][0] = nir_const_value_for_uint(get_val(alu->src[i].src.ssa), bs);
                     srcs[i] = src_arr[i];
                  }
                  unsigned eval_bs = 0;
                  if (!nir_alu_type_get_type_size(info->output_type)) {
                     eval_bs = alu->def.bit_size;
                  } else {
                     for (unsigned i = 0; i < info->num_inputs; ++i) {
                        if (!nir_alu_type_get_type_size(info->input_types[i])) {
                           eval_bs = alu->src[i].src.ssa->bit_size;
                           break;
                        }
                     }
                     if (!eval_bs)
                        eval_bs = alu->def.bit_size;
                  }
                  nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
                  uint16_t poison = 0;
                  nir_eval_const_opcode(alu->op, dest, &poison, 1, eval_bs, srcs, 0);
                  vals[&alu->def] = nir_const_value_as_uint(dest[0], alu->def.bit_size);
                  break;
               }
               case nir_instr_type_intrinsic: {
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  switch (intr->intrinsic) {
                  case nir_intrinsic_load_push_constant: {
                     uint32_t off = (uint32_t)get_val(intr->src[0].ssa) + nir_intrinsic_base(intr);
                     uint32_t idx = off / 4;
                     vals[&intr->def] = idx < pc.size() ? pc[idx] : 0;
                     break;
                  }
                  case nir_intrinsic_load_preamble:
                     vals[&intr->def] = 0;
                     break;
                  case nir_intrinsic_load_global_constant: {
                     uint64_t addr = get_val(intr->src[0].ssa);
                     if (!const_mem || const_mem->find(addr) == const_mem->end()) {
                        res.faulted = true;
                        return true;
                     }
                     vals[&intr->def] = const_mem->at(addr);
                     break;
                  }
                  case nir_intrinsic_decl_reg:
                     regs[&intr->def] = 0;
                     break;
                  case nir_intrinsic_load_reg:
                     vals[&intr->def] = regs[intr->src[0].ssa];
                     break;
                  case nir_intrinsic_store_reg:
                     regs[intr->src[1].ssa] = get_val(intr->src[0].ssa);
                     break;
                  case nir_intrinsic_load_agx: {
                     uint32_t idx = (uint32_t)get_val(intr->src[1].ssa);
                     res.iters++;
                     if (res.iters > max_iters) {
                        res.hit_step_limit = true;
                        return true;
                     }
                     uint32_t v = (idx * 2654435761u) ^ 0x5a5a5a5au;
                     vals[&intr->def] = intr->def.bit_size == 8 ? (v & 0xffu) : v;
                     break;
                  }
                  case nir_intrinsic_store_agx:
                     res.out_sum = (uint32_t)get_val(intr->src[0].ssa);
                     break;
                  default:
                     ADD_FAILURE() << "unexpected intrinsic " << nir_intrinsic_infos[intr->intrinsic].name;
                     break;
                  }
                  break;
               }
               case nir_instr_type_jump: {
                  nir_jump_instr *j = nir_instr_as_jump(instr);
                  EXPECT_EQ(j->type, nir_jump_break);
                  prev_block = blk;
                  return true;
               }
               default:
                  ADD_FAILURE() << "unexpected instr type";
                  break;
               }
            }
            for (auto &kv : phi_updates)
               vals[kv.first] = kv.second;
            prev_block = blk;
            break;
         }
         case nir_cf_node_if: {
            nir_if *nif = nir_cf_node_as_if(node);
            bool c = (get_val(nif->condition.ssa) & 1) != 0;
            if (run_list(c ? &nif->then_list : &nif->else_list))
               return true;
            break;
         }
         case nir_cf_node_loop: {
            nir_loop *loop = nir_cf_node_as_loop(node);
            unsigned guard = 0;
            while (!run_list(&loop->body)) {
               if (++guard > max_iters) {
                  res.hit_step_limit = true;
                  return true;
               }
            }
            if (res.hit_step_limit)
               return true;
            break;
         }
         default:
            break;
         }
      }
      return false;
   };

   run_list(&impl->body);
   return res;
}

static nir_shader *
make_1d_interior_bounds_shader(int form)
{
   nir_builder b = make_compute_shader("interior_1d");
   nir_def *bound = push_u32(&b, 0);
   nir_def *base = push_u32(&b, 4);
   nir_def *stride = push_u32(&b, 8);
   nir_def *limit = push_u32(&b, 12);
   nir_def *src = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *dst = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_variable *i_var = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, i_var, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_loop *loop = nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, i_var);
   nir_if *ex = nir_push_if(&b, nir_ige(&b, i, bound));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, ex);

   nir_def *coord = nir_iadd(&b, base, nir_imul(&b, i, stride));
   nir_def *in_bounds = NULL;
   if (form == 1) {
      nir_if *sc = nir_push_if(&b, nir_ige_imm(&b, coord, 0));
      nir_def *t = nir_ilt(&b, coord, limit);
      nir_push_else(&b, sc);
      nir_def *f = nir_imm_false(&b);
      nir_pop_if(&b, sc);
      in_bounds = nir_if_phi(&b, t, f);
   } else if (form == 2) {
      in_bounds = nir_bcsel(&b, nir_ige_imm(&b, coord, 0),
                            nir_ilt(&b, coord, limit), nir_imm_false(&b));
   } else if (form == 3) {
      in_bounds = nir_iand(&b, nir_ige_imm(&b, coord, 0),
                           nir_ilt(&b, coord, limit));
   } else {
      in_bounds = nir_ult(&b, coord, limit);
   }

   nir_if *guard = nir_push_if(&b, in_bounds);
   {
      nir_def *v = nir_load_agx(
         &b, 1, 32, src, coord,
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .format = PIPE_FORMAT_R32_UINT);
      nir_def *w = nir_iadd(&b, nir_imul_imm(&b, v, 3), i);
      nir_def *s = nir_iadd(&b, nir_load_var(&b, acc_var), nir_ixor(&b, w, coord));
      nir_store_var(&b, acc_var, s, 1);
   }
   nir_pop_if(&b, guard);

   nir_store_var(&b, i_var, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);

   nir_store_agx(&b, nir_load_var(&b, acc_var), dst, nir_imm_int(&b, 0),
                 .access = ACCESS_NON_READABLE, .format = PIPE_FORMAT_R32_UINT);

   nir_lower_vars_to_ssa(b.shader);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   return b.shader;
}

static nir_shader *
make_2d_interior_bounds_shader(bool inner_entry_guard = false)
{
   nir_builder b = make_compute_shader("interior_2d");
   nir_def *kh = push_u32(&b, 0);
   nir_def *kw = push_u32(&b, 4);
   nir_def *oy = push_u32(&b, 8);
   nir_def *sy = push_u32(&b, 12);
   nir_def *py = push_u32(&b, 16);
   nir_def *dy = push_u32(&b, 20);
   nir_def *ih = push_u32(&b, 24);
   nir_def *bx = push_u32(&b, 28);
   nir_def *dx = push_u32(&b, 32);
   nir_def *iw = push_u32(&b, 36);
   nir_def *src = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *dst = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_def *entry_cond = inner_entry_guard
                            ? nir_igt_imm(&b, kh, 0)
                            : nir_iand(&b, nir_igt_imm(&b, kh, 0),
                                       nir_igt_imm(&b, kw, 0));
   nir_if *pos = nir_push_if(&b, entry_cond);
   {
      nir_variable *ky_var = nir_local_variable_create(b.impl, glsl_uint_type(), "ky");
      nir_store_var(&b, ky_var, nir_imm_int(&b, 0), 1);

      nir_loop *yloop = nir_push_loop(&b);
      nir_def *ky = nir_load_var(&b, ky_var);
      nir_def *iy_part = nir_imadshl_agx(&b, dy, ky, py, nir_imm_int(&b, 0));
      nir_def *iy = nir_imadshl_agx(&b, oy, sy, iy_part, nir_imm_int(&b, 0));

      nir_if *ysc = nir_push_if(&b, nir_ige_imm(&b, iy, 0));
      nir_def *yoob = nir_ige(&b, iy, ih);
      nir_push_else(&b, ysc);
      nir_def *yt = nir_imm_true(&b);
      nir_pop_if(&b, ysc);
      nir_def *yin = nir_inot(&b, nir_if_phi(&b, yoob, yt));

      nir_if *yguard = nir_push_if(&b, yin);
      {
         nir_if *kw_guard = inner_entry_guard
                               ? nir_push_if(&b, nir_igt_imm(&b, kw, 0))
                               : NULL;
         nir_variable *kx_var = nir_local_variable_create(b.impl, glsl_uint_type(), "kx");
         nir_store_var(&b, kx_var, nir_imm_int(&b, 0), 1);

         nir_loop *xloop = nir_push_loop(&b);
         nir_def *kx = nir_load_var(&b, kx_var);
         nir_def *ix = nir_imadshl_agx(&b, dx, kx, bx, nir_imm_int(&b, 0));

         nir_if *xsc = nir_push_if(&b, nir_ige_imm(&b, ix, 0));
         nir_def *xoob = nir_ige(&b, ix, iw);
         nir_push_else(&b, xsc);
         nir_def *xt = nir_imm_true(&b);
         nir_pop_if(&b, xsc);
         nir_def *xin = nir_inot(&b, nir_if_phi(&b, xoob, xt));

         nir_if *xguard = nir_push_if(&b, xin);
         {
            nir_def *idx = nir_iadd(&b, nir_imul(&b, iy, iw), ix);
            nir_def *v = nir_load_agx(
               &b, 1, 32, src, idx,
               .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
               .format = PIPE_FORMAT_R32_UINT);
            nir_def *mix = nir_ixor(&b, nir_imul_imm(&b, v, 5), nir_iadd(&b, ky, kx));
            nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), mix), 1);
         }
         nir_pop_if(&b, xguard);

         nir_def *kx_next = nir_iadd_imm(&b, kx, 1);
         nir_store_var(&b, kx_var, kx_next, 1);
         nir_if *xex = nir_push_if(&b, nir_ige(&b, kx_next, kw));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xex);
         nir_pop_loop(&b, xloop);
         if (kw_guard)
            nir_pop_if(&b, kw_guard);
      }
      nir_pop_if(&b, yguard);

      nir_def *ky_next = nir_iadd_imm(&b, ky, 1);
      nir_store_var(&b, ky_var, ky_next, 1);
      nir_if *yex = nir_push_if(&b, nir_ige(&b, ky_next, kh));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, yex);
      nir_pop_loop(&b, yloop);
   }
   nir_pop_if(&b, pos);

   nir_store_agx(&b, nir_load_var(&b, acc_var), dst, nir_imm_int(&b, 0),
                 .access = ACCESS_NON_READABLE, .format = PIPE_FORMAT_R32_UINT);

   nir_lower_vars_to_ssa(b.shader);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   return b.shader;
}

TEST(AGX3VisionOpts, InteriorBoundsUnswitch1DAnd2DForms)
{
   for (int form = 1; form <= 4; ++form) {
      nir_shader *s = make_1d_interior_bounds_shader(form);
      EXPECT_TRUE(agx_nir_opt_unswitch_interior_bounds(s)) << "form=" << form;
      nir_validate_shader(s, "after interior 1d");
      EXPECT_EQ(count_loops(s), 2u) << "form=" << form;

      unsigned branchless = 0, marked_fallback = 0;
      nir_function_impl *impl = nir_shader_get_entrypoint(s);
      std::function<void(struct exec_list *)> walk = [&](struct exec_list *list) {
         foreach_list_typed(nir_cf_node, n, node, list) {
            if (n->type == nir_cf_node_loop) {
               nir_loop *lp = nir_cf_node_as_loop(n);
               unsigned nifs = 0;
               foreach_list_typed(nir_cf_node, cn, node, &lp->body) {
                  if (cn->type == nir_cf_node_if)
                     nifs++;
               }
               if (nifs == 1 && !lp->partially_unrolled)
                  branchless++;
               if (lp->partially_unrolled)
                  marked_fallback++;
            } else if (n->type == nir_cf_node_if) {
               walk(&nir_cf_node_as_if(n)->then_list);
               walk(&nir_cf_node_as_if(n)->else_list);
            }
         }
      };
      walk(&impl->body);
      EXPECT_EQ(branchless, 1u) << "form=" << form;
      EXPECT_EQ(marked_fallback, 1u) << "form=" << form;
      ralloc_free(s);
   }

   nir_shader *s2 = make_2d_interior_bounds_shader();
   EXPECT_TRUE(agx_nir_opt_unswitch_interior_bounds(s2));
   nir_validate_shader(s2, "after interior 2d");
   EXPECT_EQ(count_loops(s2), 4u);

   unsigned fast_2d_loops = 0, fallback_loops = 0;
   nir_function_impl *impl2 = nir_shader_get_entrypoint(s2);
   std::function<void(struct exec_list *)> walk2 = [&](struct exec_list *list) {
      foreach_list_typed(nir_cf_node, n, node, list) {
         if (n->type == nir_cf_node_loop) {
            nir_loop *lp = nir_cf_node_as_loop(n);
            unsigned nifs = 0;
            foreach_list_typed(nir_cf_node, cn, node, &lp->body) {
               if (cn->type == nir_cf_node_if)
                  nifs++;
            }
            if (lp->partially_unrolled)
               fallback_loops++;
            else if (nifs == 1)
               fast_2d_loops++;
            walk2(&lp->body);
         } else if (n->type == nir_cf_node_if) {
            walk2(&nir_cf_node_as_if(n)->then_list);
            walk2(&nir_cf_node_as_if(n)->else_list);
         }
      }
   };
   walk2(&impl2->body);
   EXPECT_EQ(fast_2d_loops, 2u);
   EXPECT_EQ(fallback_loops, 2u);
   ralloc_free(s2);
}

TEST(AGX3VisionOpts, InteriorBoundsUnswitchExecution)
{
   struct Case1D {
      int32_t bound, base, stride, limit;
   };
   const Case1D cases1d[] = {
      {3, 2, 1, 10},
      {5, 1, 2, 10},
      {5, 0, 2, 9},
      {5, 0, 2, 8},
      {3, -1, 1, 10},
      {3, 8, 1, 10},
      {0, 2, 1, 10},
      {-2, 2, 1, 10},
      {4, 2, 0, 10},
      {4, 5, -1, 10},
      {4, 2, 1, 0},
      {4, 2, 1, -5},
      {5, 0, 0x40000000, 10},
      {2, 0x7fffffff, 1, 0x7fffffff},
      {1, 0, 1, 1},
   };

   for (int form = 1; form <= 4; ++form) {
      nir_shader *ref = make_1d_interior_bounds_shader(form);
      nir_shader *opt = nir_shader_clone(NULL, ref);
      ASSERT_TRUE(agx_nir_opt_unswitch_interior_bounds(opt));
      nir_validate_shader(opt, "1d exec opt");

      for (const auto &c : cases1d) {
         std::vector<uint32_t> pc = {(uint32_t)c.bound, (uint32_t)c.base,
                                     (uint32_t)c.stride, (uint32_t)c.limit};
         NirExecResult r_ref = exec_stencil_nir(ref, pc);
         NirExecResult r_opt = exec_stencil_nir(opt, pc);
         EXPECT_FALSE(r_ref.hit_step_limit);
         EXPECT_FALSE(r_opt.hit_step_limit);
         EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
            << "form=" << form << " N=" << c.bound << " base=" << c.base
            << " stride=" << c.stride << " limit=" << c.limit;
         EXPECT_EQ(r_opt.iters, r_ref.iters)
            << "form=" << form << " N=" << c.bound << " base=" << c.base
            << " stride=" << c.stride << " limit=" << c.limit;
      }
      ralloc_free(ref);
      ralloc_free(opt);
   }

   for (bool inner_guard : {false, true}) {
   nir_shader *ref2 = make_2d_interior_bounds_shader(inner_guard);
   nir_shader *opt2 = nir_shader_clone(NULL, ref2);
   ASSERT_TRUE(agx_nir_opt_unswitch_interior_bounds(opt2));
   nir_validate_shader(opt2, "2d exec opt");

   struct Case2D {
      int32_t kh, kw, oy, sy, py, dy, ih, bx, dx, iw;
   };
   const Case2D cases2d[] = {
      {3, 3, 2, 1, -1, 1, 8, 2, 1, 8},
      {5, 5, 3, 1, -2, 1, 10, 2, 1, 10},
      {3, 3, 0, 1, -1, 1, 8, 2, 1, 8},
      {3, 3, 2, 1, -1, 1, 8, -1, 1, 8},
      {3, 3, 7, 1, -1, 1, 8, 6, 1, 8},
      {5, 5, 1, 2, -1, 2, 12, 1, 2, 12},
      {0, 3, 2, 1, -1, 1, 8, 2, 1, 8},
      {3, 0, 2, 1, -1, 1, 8, 2, 1, 8},
      {5, 3, 1, 1, 0, 0x40000000, 10, 1, 1, 10},
      {3, 5, 1, 1, 0, 1, 10, 0, 0x40000000, 10},
   };
   for (const auto &c : cases2d) {
      std::vector<uint32_t> pc = {
         (uint32_t)c.kh, (uint32_t)c.kw, (uint32_t)c.oy, (uint32_t)c.sy,
         (uint32_t)c.py, (uint32_t)c.dy, (uint32_t)c.ih, (uint32_t)c.bx,
         (uint32_t)c.dx, (uint32_t)c.iw,
      };
      NirExecResult r_ref = exec_stencil_nir(ref2, pc);
      NirExecResult r_opt = exec_stencil_nir(opt2, pc);
      EXPECT_FALSE(r_ref.hit_step_limit);
      EXPECT_FALSE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
         << "kh=" << c.kh << " kw=" << c.kw << " oy=" << c.oy << " bx=" << c.bx;
      EXPECT_EQ(r_opt.iters, r_ref.iters)
         << "kh=" << c.kh << " kw=" << c.kw << " oy=" << c.oy << " bx=" << c.bx;
   }
   ralloc_free(ref2);
   ralloc_free(opt2);
   }
}


/* Bug 1 regression: 2D interior-bounds check where both outer (y) and inner (x)
 * limits come from the SAME in-loop load_global_constant (shared def inside the
 * outer loop). Clones emitted inside y's chk_if must not leak into x's remap. */
TEST(AGX3VisionOpts, InteriorBounds2DSharedInLoopConstLimit)
{
   auto build_2d_shared_limit = [&]() -> nir_shader * {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options,
                                                     "stencil_2d_shared_lim");
      auto pc = [&](unsigned idx) {
         return nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                       .base = (int)(idx * 4), .range = 4);
      };
      /* pc[0]=kh, pc[1]=kw, pc[2]=base_y, pc[3]=base_x, pc[4]=addr_lo, pc[5]=addr_hi */
      nir_def *kh = pc(0);
      nir_def *kw = pc(1);
      nir_def *base_y = pc(2);
      nir_def *base_x = pc(3);
      nir_def *addr = nir_pack_64_2x32_split(&b, pc(4), pc(5));
      nir_def *ptr = nir_imm_int64(&b, 0x1000);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *ky = nir_local_variable_create(b.impl, glsl_int_type(), "ky");
      nir_variable *kx = nir_local_variable_create(b.impl, glsl_int_type(), "kx");
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, ky, nir_imm_int(&b, 0), 1);

      nir_loop *yloop = nir_push_loop(&b);
      {
         nir_def *yv = nir_load_var(&b, ky);
         nir_push_if(&b, nir_ige(&b, yv, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         /* Shared in-loop constant load used as BOTH ih and iw! */
         nir_def *shared_dim =
            nir_load_global_constant(&b, 1, 32, addr, .align_mul = 4);
         nir_def *iy = nir_iadd(&b, base_y, yv);
         nir_def *y_in = nir_iand(&b, nir_ige_imm(&b, iy, 0),
                                  nir_ilt(&b, iy, shared_dim));
         nir_push_if(&b, y_in);
         {
            nir_store_var(&b, kx, nir_imm_int(&b, 0), 1);
            nir_loop *xloop = nir_push_loop(&b);
            {
               nir_def *xv = nir_load_var(&b, kx);
               nir_push_if(&b, nir_ige(&b, xv, kw));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *ix = nir_iadd(&b, base_x, xv);
               nir_def *x_in = nir_iand(&b, nir_ige_imm(&b, ix, 0),
                                        nir_ilt(&b, ix, shared_dim));
               nir_push_if(&b, x_in);
               {
                  nir_def *idx = nir_iadd(&b, nir_imul(&b, iy, shared_dim), ix);
                  nir_def *ld = nir_load_agx(&b, 1, 32, ptr, idx,
                                             .format = PIPE_FORMAT_R32_UINT);
                  nir_def *sum = nir_iadd(&b, nir_load_var(&b, acc), ld);
                  nir_def *mix = nir_ixor(&b, sum, nir_iadd_imm(&b, idx, 7));
                  nir_store_var(&b, acc, mix, 1);
               }
               nir_pop_if(&b, NULL);
               nir_store_var(&b, kx, nir_iadd_imm(&b, xv, 1), 1);
            }
            nir_pop_loop(&b, xloop);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, ky, nir_iadd_imm(&b, yv, 1), 1);
      }
      nir_pop_loop(&b, yloop);

      nir_store_agx(&b, nir_load_var(&b, acc), ptr, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "build_2d_shared_limit");
      return b.shader;
   };

   nir_shader *ref = build_2d_shared_limit();
   nir_shader *opt = build_2d_shared_limit();
   ASSERT_TRUE(agx_nir_opt_unswitch_interior_bounds(opt));
   nir_validate_shader(opt, "after 2d shared limit interior unswitch");

   const uint64_t valid_addr = 0x2000;
   std::unordered_map<uint64_t, uint32_t> mem = {{valid_addr, 16u}};
   struct { int32_t kh, kw, by, bx; uint64_t addr; } cases[] = {
      {3, 3, 2, 2, valid_addr},   /* interior */
      {3, 3, 0, 14, valid_addr},  /* x border */
      {3, 3, -1, 2, valid_addr},  /* y < 0 */
      {0, 3, 2, 2, 0},            /* kh == 0: poison addr, must not fault */
      {3, 0, 2, 2, valid_addr},   /* kw == 0 */
   };
   for (auto &tc : cases) {
      std::vector<uint32_t> pc = {
         (uint32_t)tc.kh, (uint32_t)tc.kw,
         (uint32_t)tc.by, (uint32_t)tc.bx,
         (uint32_t)(tc.addr & 0xffffffffu), (uint32_t)(tc.addr >> 32),
      };
      auto r_ref = exec_stencil_nir(ref, pc, 512, &mem);
      auto r_opt = exec_stencil_nir(opt, pc, 512, &mem);
      EXPECT_FALSE(r_ref.faulted);
      EXPECT_FALSE(r_opt.faulted);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
   }
   ralloc_free(ref);
   ralloc_free(opt);
}

/* Bug 2 regression: 2D top-tested inner loop whose base and limit come from
 * non-speculatable load_global_constant inside the inner loop body. When kw == 0
 * (inner trip 0) while kh > 0 and y is interior, the original loop never runs
 * those inner-body loads, so the guard must NOT fault on a poison inner addr. */
TEST(AGX3VisionOpts, InteriorBounds2DInLoopConstLoadsTripZero)
{
   auto build_2d_inner_const = [&]() -> nir_shader * {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options,
                                                     "stencil_2d_inner_const");
      auto pc = [&](unsigned idx) {
         return nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                       .base = (int)(idx * 4), .range = 4);
      };
      /* pc[0]=kh, pc[1]=kw, pc[2]=base_y, pc[3]=ih, pc[4..5]=inner_base_addr,
       * pc[6..7]=inner_lim_addr */
      nir_def *kh = pc(0);
      nir_def *kw = pc(1);
      nir_def *base_y = pc(2);
      nir_def *ih = pc(3);
      nir_def *inner_base_addr = nir_pack_64_2x32_split(&b, pc(4), pc(5));
      nir_def *inner_lim_addr = nir_pack_64_2x32_split(&b, pc(6), pc(7));
      nir_def *ptr = nir_imm_int64(&b, 0x1000);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *ky = nir_local_variable_create(b.impl, glsl_int_type(), "ky");
      nir_variable *kx = nir_local_variable_create(b.impl, glsl_int_type(), "kx");
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, ky, nir_imm_int(&b, 0), 1);

      nir_loop *yloop = nir_push_loop(&b);
      {
         nir_def *yv = nir_load_var(&b, ky);
         nir_push_if(&b, nir_ige(&b, yv, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         nir_def *iy = nir_iadd(&b, base_y, yv);
         nir_def *y_in = nir_iand(&b, nir_ige_imm(&b, iy, 0),
                                  nir_ilt(&b, iy, ih));
         nir_push_if(&b, y_in);
         {
            nir_store_var(&b, kx, nir_imm_int(&b, 0), 1);
            nir_loop *xloop = nir_push_loop(&b);
            {
               nir_def *xv = nir_load_var(&b, kx);
               nir_push_if(&b, nir_ige(&b, xv, kw));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               /* Non-speculatable loads INSIDE top-tested inner body! */
               nir_def *bx = nir_load_global_constant(&b, 1, 32,
                                                      inner_base_addr,
                                                      .align_mul = 4);
               nir_def *ix = nir_iadd(&b, bx, xv);
               nir_def *x_ge0 = nir_ige_imm(&b, ix, 0);
               nir_variable *x_in_v =
                  nir_local_variable_create(b.impl, glsl_bool_type(), "xin");
               nir_store_var(&b, x_in_v, nir_imm_false(&b), 1);
               nir_push_if(&b, x_ge0);
               {
                  nir_def *iw = nir_load_global_constant(&b, 1, 32,
                                                         inner_lim_addr,
                                                         .align_mul = 4);
                  nir_store_var(&b, x_in_v, nir_ilt(&b, ix, iw), 1);
               }
               nir_pop_if(&b, NULL);

               nir_push_if(&b, nir_load_var(&b, x_in_v));
               {
                  nir_def *idx = nir_iadd(&b, nir_imul_imm(&b, iy, 32), ix);
                  nir_def *ld = nir_load_agx(&b, 1, 32, ptr, idx,
                                             .format = PIPE_FORMAT_R32_UINT);
                  nir_def *sum = nir_iadd(&b, nir_load_var(&b, acc), ld);
                  nir_def *mix = nir_ixor(&b, sum, nir_iadd_imm(&b, idx, 11));
                  nir_store_var(&b, acc, mix, 1);
               }
               nir_pop_if(&b, NULL);
               nir_store_var(&b, kx, nir_iadd_imm(&b, xv, 1), 1);
            }
            nir_pop_loop(&b, xloop);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, ky, nir_iadd_imm(&b, yv, 1), 1);
      }
      nir_pop_loop(&b, yloop);

      nir_store_agx(&b, nir_load_var(&b, acc), ptr, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "build_2d_inner_const");
      return b.shader;
   };

   nir_shader *ref = build_2d_inner_const();
   nir_shader *opt = build_2d_inner_const();
   ASSERT_TRUE(agx_nir_opt_unswitch_interior_bounds(opt));
   nir_validate_shader(opt, "after 2d inner const interior unswitch");

   const uint64_t addr_bx_pos = 0x3000;
   const uint64_t addr_bx_neg = 0x3004;
   const uint64_t addr_iw = 0x3008;
   std::unordered_map<uint64_t, uint32_t> mem = {
      {addr_bx_pos, 4u},
      {addr_bx_neg, (uint32_t)-10},
      {addr_iw, 16u},
   };
   struct {
      int32_t kh, kw, by, ih;
      uint64_t bx_addr, iw_addr;
   } cases[] = {
      /* Critical Bug 2 case: kh=3, y interior, kw=0 with POISON addresses 0! */
      {3, 0, 2, 16, 0, 0},
      /* Negative kw with poison addresses: must not fault! */
      {3, -5, 2, 16, 0, 0},
      /* bx < 0 with poison iw_addr: short-circuit must prevent reading iw! */
      {3, 3, 2, 16, addr_bx_neg, 0},
      /* Valid interior */
      {3, 3, 2, 16, addr_bx_pos, addr_iw},
      /* Valid border */
      {3, 15, 2, 16, addr_bx_pos, addr_iw},
      /* Outer trip 0 with poison addresses */
      {0, 3, 2, 16, 0, 0},
   };
   for (auto &tc : cases) {
      std::vector<uint32_t> pc = {
         (uint32_t)tc.kh, (uint32_t)tc.kw,
         (uint32_t)tc.by, (uint32_t)tc.ih,
         (uint32_t)(tc.bx_addr & 0xffffffffu), (uint32_t)(tc.bx_addr >> 32),
         (uint32_t)(tc.iw_addr & 0xffffffffu), (uint32_t)(tc.iw_addr >> 32),
      };
      auto r_ref = exec_stencil_nir(ref, pc, 512, &mem);
      auto r_opt = exec_stencil_nir(opt, pc, 512, &mem);
      EXPECT_FALSE(r_ref.faulted) << "ref faulted for kw=" << tc.kw;
      EXPECT_FALSE(r_opt.faulted) << "opt faulted for kw=" << tc.kw;
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum);
      EXPECT_EQ(r_opt.iters, r_ref.iters);
   }
   ralloc_free(ref);
   ralloc_free(opt);
}


TEST(AGX3VisionOpts, InteriorBounds2DEarlyOuterBreakTripZero)
{
   /* Fault-safety regression: a 2D outer loop with an early break before the
    * outer bounds guard must NOT be unswitched, because non-speculatable outer
    * limit and inner bound/base/limit load_global_constant instructions would
    * otherwise be hoisted before the loop and fault when iteration 0 breaks
    * early. */
   auto build_early_break = [&]() {
      nir_builder b = make_compute_shader("interior_2d_early_break");
      nir_def *kh = push_u32(&b, 0);
      nir_def *early_brk = push_u32(&b, 4);
      nir_def *by = push_u32(&b, 8);
      nir_def *ih_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 12), push_u32(&b, 16));
      nir_def *kw_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 20), push_u32(&b, 24));
      nir_def *bx_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 28), push_u32(&b, 32));
      nir_def *iw_addr = nir_pack_64_2x32_split(&b, push_u32(&b, 36), push_u32(&b, 40));
      nir_def *ptr = nir_load_preamble(&b, 1, 64, .base = 0);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *ky = nir_local_variable_create(b.impl, glsl_uint_type(), "ky");
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, ky, nir_imm_int(&b, 0), 1);

      nir_loop *yloop = nir_push_loop(&b);
      {
         nir_def *yv = nir_load_var(&b, ky);
         nir_if *yex = nir_push_if(&b, nir_ige(&b, yv, kh));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         /* Early outer break before outer guard and inner loop. */
         nir_if *ebrk = nir_push_if(&b, nir_ine_imm(&b, early_brk, 0));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, ebrk);

         nir_def *ih = nir_load_global_constant(&b, 1, 32, ih_addr, .align_mul = 4);
         nir_def *iy = nir_iadd(&b, by, yv);
         nir_if *ysc = nir_push_if(&b, nir_ige_imm(&b, iy, 0));
         nir_def *yoob = nir_ige(&b, iy, ih);
         nir_push_else(&b, ysc);
         nir_def *yt = nir_imm_true(&b);
         nir_pop_if(&b, ysc);
         nir_def *yin = nir_inot(&b, nir_if_phi(&b, yoob, yt));

         nir_if *yg = nir_push_if(&b, yin);
         {
            nir_def *kw = nir_load_global_constant(&b, 1, 32, kw_addr, .align_mul = 4);
            nir_variable *kx = nir_local_variable_create(b.impl, glsl_uint_type(), "kx");
            nir_store_var(&b, kx, nir_imm_int(&b, 0), 1);
            nir_loop *xloop = nir_push_loop(&b);
            {
               nir_def *xv = nir_load_var(&b, kx);
               nir_if *xex = nir_push_if(&b, nir_ige(&b, xv, kw));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, xex);

               nir_def *bx = nir_load_global_constant(&b, 1, 32, bx_addr, .align_mul = 4);
               nir_def *iw = nir_load_global_constant(&b, 1, 32, iw_addr, .align_mul = 4);
               nir_def *ix = nir_iadd(&b, bx, xv);
               nir_if *xsc = nir_push_if(&b, nir_ige_imm(&b, ix, 0));
               nir_def *xoob = nir_ige(&b, ix, iw);
               nir_push_else(&b, xsc);
               nir_def *xt = nir_imm_true(&b);
               nir_pop_if(&b, xsc);
               nir_def *xin = nir_inot(&b, nir_if_phi(&b, xoob, xt));

               nir_if *xg = nir_push_if(&b, xin);
               {
                  nir_def *idx = nir_iadd(&b, nir_imul(&b, iy, iw), ix);
                  nir_def *v = nir_load_agx(
                     &b, 1, 32, ptr, idx,
                     .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                                     ACCESS_CAN_REORDER),
                     .format = PIPE_FORMAT_R32_UINT);
                  nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
               }
               nir_pop_if(&b, xg);
               nir_store_var(&b, kx, nir_iadd_imm(&b, xv, 1), 1);
            }
            nir_pop_loop(&b, xloop);
         }
         nir_pop_if(&b, yg);
         nir_store_var(&b, ky, nir_iadd_imm(&b, yv, 1), 1);
      }
      nir_pop_loop(&b, yloop);

      nir_store_agx(&b, nir_load_var(&b, acc), ptr, nir_imm_int(&b, 0),
                    .format = PIPE_FORMAT_R32_UINT);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "build_early_break");
      return b.shader;
   };

   nir_shader *ref = build_early_break();
   nir_shader *opt = build_early_break();
   EXPECT_FALSE(agx_nir_opt_unswitch_interior_bounds(opt));
   nir_validate_shader(opt, "after early outer break rejection");

   const uint64_t addr_ih = 0x4000, addr_kw = 0x4004, addr_bx = 0x4008, addr_iw = 0x400c;
   std::unordered_map<uint64_t, uint32_t> mem = {
      {addr_ih, 16u}, {addr_kw, 3u}, {addr_bx, 2u}, {addr_iw, 16u},
   };
   /* Early outer break with poison addresses 0 must not fault. */
   std::vector<uint32_t> pc_poison = {3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0};
   auto r_ref_p = exec_stencil_nir(ref, pc_poison, 512, &mem);
   auto r_opt_p = exec_stencil_nir(opt, pc_poison, 512, &mem);
   EXPECT_FALSE(r_ref_p.faulted);
   EXPECT_FALSE(r_opt_p.faulted);
   EXPECT_EQ(r_opt_p.out_sum, r_ref_p.out_sum);
   EXPECT_EQ(r_opt_p.iters, 0u);

   /* Normal execution without early break matches reference. */
   std::vector<uint32_t> pc_valid = {
      3, 0, 2,
      (uint32_t)addr_ih, 0, (uint32_t)addr_kw, 0,
      (uint32_t)addr_bx, 0, (uint32_t)addr_iw, 0,
   };
   auto r_ref_v = exec_stencil_nir(ref, pc_valid, 512, &mem);
   auto r_opt_v = exec_stencil_nir(opt, pc_valid, 512, &mem);
   EXPECT_FALSE(r_ref_v.faulted);
   EXPECT_FALSE(r_opt_v.faulted);
   EXPECT_EQ(r_opt_v.out_sum, r_ref_v.out_sum);
   EXPECT_EQ(r_opt_v.iters, 9u);

   ralloc_free(ref);
   ralloc_free(opt);
}

TEST(AGX3VisionOpts, RotatedSmallWindow3x3And4xPeelExecution)
{
   /* Run interior bounds unswitch + runtime loop unrolling on a rotated 2D
    * stencil (32-bit loads), verifying exact execution across 3x3 (fully
    * straight-line), 4x4 (exact 4-peel skipping tail), 5x5 (4-peel + 1 tail),
    * 6..7, 8..11 (8x main), 1..2 (scalar), rectangular, border, and wrap cases. */
   for (unsigned factor : {8u, 4u}) {
   nir_shader *ref = make_2d_interior_bounds_shader(factor == 4u);
   nir_shader *opt = nir_shader_clone(NULL, ref);
   ASSERT_TRUE(agx_nir_opt_unswitch_interior_bounds(opt));
   unsigned widest = 0;
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, factor, 320, &widest));
   nir_opt_dead_cf(opt);
   nir_opt_copy_prop(opt);
   nir_opt_constant_folding(opt);
   nir_opt_dce(opt);
   nir_validate_shader(opt, "after rotated small-window peel");
   EXPECT_EQ(widest, factor);

   /* Verify a loop-free 3x3 straight-line branch exists (a then_list containing
    * >= 9 load_agx instructions and 0 nested loops). */
   bool found_3x3_straight = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(opt);
   std::function<unsigned(struct exec_list *, bool *)> scan =
      [&](struct exec_list *list, bool *has_loop) -> unsigned {
      unsigned loads = 0;
      foreach_list_typed(nir_cf_node, n, node, list) {
         if (n->type == nir_cf_node_block) {
            nir_foreach_instr(instr, nir_cf_node_as_block(n)) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
                  loads++;
            }
         } else if (n->type == nir_cf_node_loop) {
            *has_loop = true;
         } else if (n->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(n);
            bool tl = false, el = false;
            unsigned tloads = scan(&nif->then_list, &tl);
            unsigned eloads = scan(&nif->else_list, &el);
            if (!tl && tloads >= 9)
               found_3x3_straight = true;
            if (tl || el)
               *has_loop = true;
            loads += tloads + eloads;
         }
      }
      return loads;
   };
   bool top_loop = false;
   scan(&impl->body, &top_loop);
   EXPECT_TRUE(found_3x3_straight);

   struct Case2D {
      int32_t kh, kw, oy, sy, py, dy, ih, bx, dx, iw;
   };
   const Case2D cases[] = {
      {1, 1, 2, 1, 0, 1, 16, 2, 1, 16},
      {2, 2, 2, 1, 0, 1, 16, 2, 1, 16},
      {3, 3, 2, 1, -1, 1, 16, 2, 1, 16},        /* 3x3 straight-line */
      {3, 4, 2, 1, -1, 1, 16, 2, 1, 16},        /* 3x4 exact 4-peel */
      {4, 3, 2, 1, -1, 1, 16, 2, 1, 16},        /* 4x3 inner 3-peel */
      {4, 4, 2, 1, -1, 1, 16, 2, 1, 16},        /* 4x4 exact 4-peel */
      {5, 5, 3, 1, -2, 1, 16, 2, 1, 16},        /* 5x5 4-peel + 1 tail */
      {3, 5, 2, 1, -1, 1, 16, 2, 1, 16},        /* 3x5 */
      {5, 3, 3, 1, -2, 1, 16, 2, 1, 16},        /* 5x3 */
      {6, 6, 3, 1, -2, 1, 20, 2, 1, 20},        /* 6x6 */
      {7, 7, 3, 1, -2, 1, 20, 2, 1, 20},        /* 7x7 */
      {8, 8, 3, 1, -2, 1, 20, 2, 1, 20},        /* 8x8 */
      {9, 9, 3, 1, -2, 1, 20, 2, 1, 20},        /* 9x9 */
      {3, 11, 2, 1, -1, 1, 24, 2, 1, 24},       /* 3x11 */
      {3, 3, 0, 1, -1, 1, 16, 2, 1, 16},        /* border y */
      {5, 5, 3, 1, -2, 1, 16, -1, 1, 16},       /* border x */
      {0, 3, 2, 1, -1, 1, 16, 2, 1, 16},        /* kh = 0 */
      {3, 0, 2, 1, -1, 1, 16, 2, 1, 16},        /* kw = 0 */
      {5, 5, 1, 2, -1, 2, 24, 1, 2, 24},        /* strided + dilated 5x5 */
   };
   for (const auto &c : cases) {
      std::vector<uint32_t> pc = {
         (uint32_t)c.kh, (uint32_t)c.kw, (uint32_t)c.oy, (uint32_t)c.sy,
         (uint32_t)c.py, (uint32_t)c.dy, (uint32_t)c.ih, (uint32_t)c.bx,
         (uint32_t)c.dx, (uint32_t)c.iw,
      };
      NirExecResult r_ref = exec_stencil_nir(ref, pc, 2048);
      NirExecResult r_opt = exec_stencil_nir(opt, pc, 2048);
      EXPECT_FALSE(r_ref.hit_step_limit);
      EXPECT_FALSE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
         << "kh=" << c.kh << " kw=" << c.kw << " oy=" << c.oy << " bx=" << c.bx;
      EXPECT_EQ(r_opt.iters, r_ref.iters)
         << "kh=" << c.kh << " kw=" << c.kw << " oy=" << c.oy << " bx=" << c.bx;
   }
   ralloc_free(ref);
   ralloc_free(opt);
   }
}


TEST(AGX3VisionOpts, Rotated8BitInnerNonZeroStart)
{
   /* Exercise the f >= 8 && loop_has_8bit_load(inner) small-window split on a
    * rotated 8-bit inner loop with a runtime non-zero start iv0 (both positive
    * and negative starts, iv0 >= bound single-iteration rotated edge cases, and
    * near-INT32_MIN/MAX boundaries) across trip counts 1..10. */
   auto build_rot8_nonzero = [&]() {
      nir_builder b = make_compute_shader("rot8_nonzero_start");
      nir_def *kh = push_u32(&b, 0);
      nir_def *kx0 = push_u32(&b, 4);
      nir_def *kx_end = push_u32(&b, 8);
      nir_def *src = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *dst = nir_load_preamble(&b, 1, 64, .base = 4);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);

      nir_if *pos = nir_push_if(&b, nir_igt_imm(&b, kh, 0));
      {
         nir_variable *ky = nir_local_variable_create(b.impl, glsl_uint_type(), "ky");
         nir_store_var(&b, ky, nir_imm_int(&b, 0), 1);
         nir_loop *yloop = nir_push_loop(&b);
         {
            nir_def *yv = nir_load_var(&b, ky);
            nir_variable *kx = nir_local_variable_create(b.impl, glsl_uint_type(), "kx");
            nir_store_var(&b, kx, kx0, 1);
            nir_loop *xloop = nir_push_loop(&b);
            {
               nir_def *xv = nir_load_var(&b, kx);
               nir_def *idx = nir_iadd(&b, nir_imul_imm(&b, yv, 32), xv);
               nir_def *b8 = nir_load_agx(
                  &b, 1, 8, src, idx,
                  .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                                  ACCESS_CAN_REORDER),
                  .format = PIPE_FORMAT_R8_UINT);
               nir_def *v = nir_u2u32(&b, b8);
               nir_def *mix = nir_ixor(&b, nir_imul_imm(&b, v, 7), nir_iadd(&b, yv, xv));
               nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), mix), 1);

               nir_def *xnext = nir_iadd_imm(&b, xv, 1);
               nir_store_var(&b, kx, xnext, 1);
               nir_if *xex = nir_push_if(&b, nir_ige(&b, xnext, kx_end));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, xex);
            }
            nir_pop_loop(&b, xloop);

            nir_def *ynext = nir_iadd_imm(&b, yv, 1);
            nir_store_var(&b, ky, ynext, 1);
            nir_if *yex = nir_push_if(&b, nir_ige(&b, ynext, kh));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, yex);
         }
         nir_pop_loop(&b, yloop);
      }
      nir_pop_if(&b, pos);

      nir_store_agx(&b, nir_load_var(&b, acc), dst, nir_imm_int(&b, 0),
                    .access = ACCESS_NON_READABLE, .format = PIPE_FORMAT_R32_UINT);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "build_rot8_nonzero");
      return b.shader;
   };

   nir_shader *ref = build_rot8_nonzero();
   nir_shader *opt = nir_shader_clone(NULL, ref);
   unsigned widest = 0;
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(opt, 8, 320, &widest));
   nir_opt_dead_cf(opt);
   nir_opt_copy_prop(opt);
   nir_opt_constant_folding(opt);
   nir_opt_dce(opt);
   nir_validate_shader(opt, "after rotated 8-bit non-zero start peel");
   EXPECT_EQ(widest, 8u);

   /* Verify a straight-line 3x3 8-bit branch was emitted (9+ 8-bit loads, no loop). */
   bool found_3x3 = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(opt);
   std::function<unsigned(struct exec_list *, bool *)> scan =
      [&](struct exec_list *list, bool *has_loop) -> unsigned {
      unsigned loads = 0;
      foreach_list_typed(nir_cf_node, n, node, list) {
         if (n->type == nir_cf_node_block) {
            nir_foreach_instr(instr, nir_cf_node_as_block(n)) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx &&
                   nir_instr_as_intrinsic(instr)->def.bit_size == 8)
                  loads++;
            }
         } else if (n->type == nir_cf_node_loop) {
            *has_loop = true;
         } else if (n->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(n);
            bool tl = false, el = false;
            unsigned tloads = scan(&nif->then_list, &tl);
            unsigned eloads = scan(&nif->else_list, &el);
            if (!tl && tloads >= 9)
               found_3x3 = true;
            if (tl || el)
               *has_loop = true;
            loads += tloads + eloads;
         }
      }
      return loads;
   };
   bool top_loop = false;
   scan(&impl->body, &top_loop);
   EXPECT_TRUE(found_3x3);

   struct Case {
      int32_t kh, kx0, kx_end;
   };
   std::vector<Case> cases;
   for (int32_t kh : {1, 3, 4}) {
      for (int32_t trip = 1; trip <= 10; ++trip) {
         cases.push_back({kh, 2, 2 + trip});
         cases.push_back({kh, -3, -3 + trip});
      }
   }
   cases.push_back({3, 5, 5});
   cases.push_back({3, 5, 2});
   cases.push_back({3, -1, -4});
   cases.push_back({3, INT32_MIN, INT32_MIN + 3});
   cases.push_back({3, INT32_MIN, INT32_MIN + 5});
   cases.push_back({3, INT32_MAX - 3, INT32_MAX});
   cases.push_back({3, INT32_MAX - 5, INT32_MAX});
   cases.push_back({3, INT32_MAX - 1, INT32_MAX});

   for (const auto &c : cases) {
      std::vector<uint32_t> pc = {
         (uint32_t)c.kh, (uint32_t)c.kx0, (uint32_t)c.kx_end,
      };
      auto r_ref = exec_stencil_nir(ref, pc, 2048);
      auto r_opt = exec_stencil_nir(opt, pc, 2048);
      EXPECT_FALSE(r_ref.hit_step_limit);
      EXPECT_FALSE(r_opt.hit_step_limit);
      EXPECT_EQ(r_opt.out_sum, r_ref.out_sum)
         << "kh=" << c.kh << " kx0=" << c.kx0 << " kx_end=" << c.kx_end;
      EXPECT_EQ(r_opt.iters, r_ref.iters)
         << "kh=" << c.kh << " kx0=" << c.kx0 << " kx_end=" << c.kx_end;
   }
   ralloc_free(ref);
   ralloc_free(opt);
}

/* --- Wave 8: AGX3B subgroup ballot/lt-mask & ufind_msb, signed pow2 idiv,
 * store_global vectorization, dominating bound guard & single-arm loop unswitch --- */

TEST(AGX3VisionOpts, Wave8Agx3bSubgroupBallotLtMaskAndFirstInv)
{
   /* Compute shader: bit_count(ballot(true) & ~(0xffffffff << (lane & 31)))
    * and bit_count(ballot(true) & ((1 << (lane & 31)) - 1)) must lower to
    * load_active_subgroup_invocation_agx with zero remaining ballots. */
   for (unsigned form = 0; form < 2; form++) {
      SCOPED_TRACE(form);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "wave8_ballot_lt");
      nir_def *lane = nir_load_subgroup_invocation(&b);
      nir_def *lane_m = nir_iand_imm(&b, lane, 31);
      nir_def *lt_mask = form == 0
         ? nir_inot(&b, nir_ishl(&b, nir_imm_int(&b, -1), lane_m))
         : nir_iadd_imm(&b, nir_ishl(&b, nir_imm_int(&b, 1), lane_m), -1);
      nir_def *bal = nir_ballot(&b, 1, 32, nir_imm_true(&b));
      nir_def *cnt = nir_bit_count(&b, nir_iand(&b, bal, lt_mask));
      nir_store_global(&b, cnt, nir_imm_int64(&b, 0x1000), .align_mul = 4);

      agx_nir_lower_subgroups_g17p(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "wave8 ballot lt compute");

      unsigned ballots = 0, active_inv = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            ballots += (op == nir_intrinsic_ballot);
            active_inv += (op == nir_intrinsic_load_active_subgroup_invocation_agx);
         }
      }
      EXPECT_EQ(ballots, 0u);
      EXPECT_EQ(active_inv, 1u);
      ralloc_free(b.shader);
   }

   /* Negative: fragment shader (helper lanes may exist) MUST NOT trigger. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "wave8_ballot_lt_fs");
      nir_def *lane = nir_load_subgroup_invocation(&b);
      nir_def *lt_mask = nir_inot(&b, nir_ishl(&b, nir_imm_int(&b, -1), lane));
      nir_def *bal = nir_ballot(&b, 1, 32, nir_imm_true(&b));
      nir_def *cnt = nir_bit_count(&b, nir_iand(&b, bal, lt_mask));
      nir_store_global(&b, cnt, nir_imm_int64(&b, 0x1000), .align_mul = 4);

      agx_nir_lower_subgroups_g17p(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "wave8 ballot lt fragment");

      unsigned ballots = 0, active_inv = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            ballots += (op == nir_intrinsic_ballot);
            active_inv += (op == nir_intrinsic_load_active_subgroup_invocation_agx);
         }
      }
      EXPECT_EQ(ballots, 1u);
      EXPECT_EQ(active_inv, 0u);
      ralloc_free(b.shader);
   }

   /* AGX3B emulates_ufind_msb=true: first_invocation and vote_ieq lower via
    * bit_count(first_bit - 1) with zero ufind_msb/find_lsb ops. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "wave8_first_inv_agx3b");
      nir_def *v = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
      nir_def *first = nir_read_first_invocation(&b, v);
      nir_def *eq = nir_b2i32(&b, nir_vote_ieq(&b, 1, v));
      nir_store_global(&b, nir_iadd(&b, first, eq), nir_imm_int64(&b, 0x2000), .align_mul = 4);

      agx_nir_lower_subgroups_g17p(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "wave8 first_inv agx3b");

      unsigned ufind = 0, bit_counts = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_op op = nir_instr_as_alu(instr)->op;
            ufind += (op == nir_op_ufind_msb || op == nir_op_find_lsb);
            bit_counts += (op == nir_op_bit_count);
         }
      }
      EXPECT_EQ(ufind, 0u);
      EXPECT_GE(bit_counts, 2u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, Wave8FoldSignedPow2Idiv)
{
   /* Test folding imul(imin(imax(a, -1), 1), ushr(iabs(a), k)) for k in [1, 30]
    * and verify exactness across boundary values including INT32_MIN. */
   const int32_t samples[] = {
      INT32_MIN, INT32_MIN + 1, -1025, -1024, -257, -256, -255,
      -3, -2, -1, 0, 1, 2, 3, 255, 256, 257, 1024, INT32_MAX - 1, INT32_MAX
   };

   for (unsigned k : {1u, 4u, 8u, 16u, 30u}) {
      SCOPED_TRACE(k);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "wave8_signed_pow2_idiv");
      nir_def *a = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
      nir_def *sgn = nir_imin(&b, nir_imax(&b, a, nir_imm_int(&b, -1)), nir_imm_int(&b, 1));
      nir_def *mag = nir_ushr_imm(&b, nir_iabs(&b, a), k);
      nir_def *res = nir_imul(&b, sgn, mag);
      nir_store_global(&b, res, nir_imm_int64(&b, 0x2000), .align_mul = 4);

      EXPECT_TRUE(agx_nir_fuse_algebraic_late(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "wave8 signed pow2 idiv");

      unsigned imuls = 0, iabses = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_op op = nir_instr_as_alu(instr)->op;
            imuls += (op == nir_op_imul);
            iabses += (op == nir_op_iabs);
         }
      }
      EXPECT_EQ(imuls, 0u);
      EXPECT_EQ(iabses, 0u);

      /* Algebraic oracle: (a + ((a >> 31) >>> (32 - k))) >> k == sign(a) * (|a| >> k) */
      for (int32_t v : samples) {
         int32_t sgn_v = v < 0 ? -1 : (v > 0 ? 1 : 0);
         uint32_t abs_v = v == INT32_MIN ? 0x80000000u : (uint32_t)(v < 0 ? -v : v);
         int32_t ref_v = (int32_t)((uint32_t)sgn_v * (abs_v >> k));
         uint32_t bias = ((uint32_t)(v >> 31)) >> (32u - k);
         int32_t folded_v = ((int32_t)((uint32_t)v + bias)) >> k;
         EXPECT_EQ(folded_v, ref_v) << "v=" << v << " k=" << k;
      }
      ralloc_free(b.shader);
   }

   /* Negative: k = 31 must NOT fold (32 - 31 = 1, INT32_MIN + 0x7fffffff overflows). */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "wave8_signed_pow2_idiv_k31");
      nir_def *a = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000), .align_mul = 4);
      nir_def *sgn = nir_imin(&b, nir_imax(&b, a, nir_imm_int(&b, -1)), nir_imm_int(&b, 1));
      nir_def *mag = nir_ushr_imm(&b, nir_iabs(&b, a), 31);
      nir_def *res = nir_imul(&b, sgn, mag);
      nir_store_global(&b, res, nir_imm_int64(&b, 0x2000), .align_mul = 4);

      agx_nir_fuse_algebraic_late(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      unsigned imuls = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_imul)
               imuls++;
         }
      }
      EXPECT_EQ(imuls, 1u);
      ralloc_free(b.shader);
   }
}

