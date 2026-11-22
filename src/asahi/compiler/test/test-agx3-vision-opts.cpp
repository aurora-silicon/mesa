/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <cmath>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
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

