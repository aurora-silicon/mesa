/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

static struct agx_shader_key
agx3c_key(bool emit_agx3d = false)
{
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.emit_agx3d = emit_agx3d;
   key.reserved_preamble = 16;
   return key;
}

static unsigned
count_intr(const nir_shader *nir, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
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

static void
emit_shared_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

/* Build a tree reduction over wg invocations (power of 2: 16, 32, or 64) for a
 * vec2 fadd array (at byte 0) and a scalar iadd array (at byte wg * 8). */
static nir_builder
build_tree_reduction(unsigned wg, bool read_intermediate, bool dynamic_reader,
                     bool all_lanes_read_slot0, bool divergent_guard,
                     bool exact_fadd = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "tree-reduce");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = wg * 12;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *in_addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out_addr = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *dyn_idx = nir_load_preamble(&b, 1, 32, .base = 8);

   nir_def *v = nir_load_global(&b, 3, 32,
                                nir_iadd(&b, in_addr, nir_u2u64(&b, nir_imul_imm(&b, tid, 12))),
                                .align_mul = 4);
   nir_def *vf = nir_vec2(&b, nir_channel(&b, v, 0), nir_channel(&b, v, 1));
   nir_def *vi = nir_channel(&b, v, 2);

   if (divergent_guard)
      nir_push_if(&b, nir_ine_imm(&b, tid, 31));

   nir_def *f_off = nir_ishl_imm(&b, tid, 3);
   nir_def *i_off = nir_iadd_imm(&b, nir_ishl_imm(&b, tid, 2), wg * 8);
   nir_store_shared(&b, vf, f_off, .base = 0, .write_mask = 0x3, .align_mul = 8);
   nir_store_shared(&b, vi, i_off, .base = 0, .write_mask = 0x1, .align_mul = 4);
   emit_shared_barrier(&b);

   for (unsigned stride = wg / 2; stride >= 1; stride >>= 1) {
      nir_def *cond = (stride == 1) ? nir_ieq_imm(&b, tid, 0)
                                    : nir_ult_imm(&b, tid, stride);
      nir_push_if(&b, cond);
      nir_def *f_lo_off = (stride == 1) ? nir_imm_int(&b, 0) : f_off;
      nir_def *f_hi_off = (stride == 1) ? nir_imm_int(&b, 8)
                                        : nir_iadd_imm(&b, f_off, stride * 8);
      nir_def *flo = nir_load_shared(&b, 2, 32, f_lo_off, .base = 0, .align_mul = 8);
      nir_def *fhi = nir_load_shared(&b, 2, 32, f_hi_off, .base = 0, .align_mul = 8);
      b.fp_math_ctrl = exact_fadd ? nir_fp_exact : 0;
      nir_def *fx = nir_fadd(&b, nir_channel(&b, flo, 0), nir_channel(&b, fhi, 0));
      nir_def *fy = nir_fadd(&b, nir_channel(&b, flo, 1), nir_channel(&b, fhi, 1));
      b.fp_math_ctrl = 0;
      nir_store_shared(&b, nir_vec2(&b, fx, fy), f_lo_off, .base = 0,
                       .write_mask = 0x3, .align_mul = 8);

      nir_def *i_lo_off = (stride == 1) ? nir_imm_int(&b, wg * 8) : i_off;
      nir_def *i_hi_off = (stride == 1) ? nir_imm_int(&b, wg * 8 + 4)
                                        : nir_iadd_imm(&b, i_off, stride * 4);
      nir_def *ilo = nir_load_shared(&b, 1, 32, i_lo_off, .base = 0, .align_mul = 4);
      nir_def *ihi = nir_load_shared(&b, 1, 32, i_hi_off, .base = 0, .align_mul = 4);
      nir_store_shared(&b, nir_iadd(&b, ilo, ihi), i_lo_off, .base = 0,
                       .write_mask = 0x1, .align_mul = 4);
      nir_pop_if(&b, NULL);
      emit_shared_barrier(&b);
   }

   if (all_lanes_read_slot0) {
      nir_def *rf = nir_load_shared(&b, 2, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 8);
      nir_def *ri = nir_load_shared(&b, 1, 32, nir_imm_int(&b, wg * 8), .base = 0, .align_mul = 4);
      nir_def *res = nir_vec3(&b, nir_channel(&b, rf, 0), nir_channel(&b, rf, 1), ri);
      nir_store_global(&b, res, nir_iadd(&b, out_addr, nir_u2u64(&b, nir_imul_imm(&b, tid, 12))),
                       .align_mul = 4);
   } else {
      nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
      nir_def *rf = nir_load_shared(&b, 2, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 8);
      nir_def *ri = nir_load_shared(&b, 1, 32, nir_imm_int(&b, wg * 8), .base = 0, .align_mul = 4);
      if (read_intermediate) {
         nir_def *mid = nir_load_shared(&b, 1, 32, nir_imm_int(&b, wg * 8 + 4),
                                        .base = 0, .align_mul = 4);
         ri = nir_iadd(&b, ri, mid);
      }
      if (dynamic_reader) {
         nir_def *dyn = nir_load_shared(&b, 1, 32, nir_ishl_imm(&b, dyn_idx, 2),
                                        .base = 0, .align_mul = 4);
         ri = nir_iadd(&b, ri, dyn);
      }
      nir_def *res = nir_vec3(&b, nir_channel(&b, rf, 0), nir_channel(&b, rf, 1), ri);
      nir_store_global(&b, res, out_addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
   }

   if (divergent_guard)
      nir_pop_if(&b, NULL);

   return b;
}

TEST(Agx3cSubgroupReduce, Wg32FullEliminationAndExactFloatButterfly)
{
   nir_builder b = build_tree_reduction(32, false, false, false, false, true);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "wg32 tree reduce");

   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 0u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 0u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_barrier), 0u);
   EXPECT_EQ(b.shader->info.shared_size, 0u);
   /* 2 float channels * 5 butterfly steps = 10 shuffle_xor, 1 integer reduce. */
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);

   /* Verify exact fp_math_ctrl was preserved on every fadd. */
   unsigned fadd_exact = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_fadd) {
               EXPECT_EQ(nir_instr_as_alu(instr)->fp_math_ctrl, nir_fp_exact);
               fadd_exact++;
            }
         }
      }
   }
   EXPECT_EQ(fadd_exact, 10u);
   ralloc_free(b.shader);

   /* End-to-end AGX3C compile verification. */
   nir_builder e2e = build_tree_reduction(32, false, false, false, false, false);
   agx_preprocess_nir_m5(e2e.shader);
   struct agx_shader_key key = agx3c_key(false);
   struct agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(e2e.shader, &key, &part));
   EXPECT_EQ(part.info.local_size, 0u);
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
   ralloc_free(e2e.shader);
}

TEST(Agx3cSubgroupReduce, Wg64KeepsCrossSubgroupStepLowersSubgroup0Tail)
{
   nir_builder b = build_tree_reduction(64, false, false, false, false, false);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "wg64 tree reduce");

   /* Initial stores (2) + stride=32 loads (4) remain; all stride=16..1 shared
    * ops and the 6 subgroup barriers vanish. */
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 2u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 4u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_barrier), 1u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   ralloc_free(b.shader);

   /* End-to-end AGX3C compile verification. */
   nir_builder e2e = build_tree_reduction(64, false, false, false, false, false);
   agx_preprocess_nir_m5(e2e.shader);
   struct agx_shader_key key = agx3c_key(false);
   struct agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(e2e.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
   ralloc_free(e2e.shader);
}

TEST(Agx3cSubgroupReduce, Wg16PartialSubgroup)
{
   nir_builder b = build_tree_reduction(16, false, false, false, false, false);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "wg16 tree reduce");

   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 0u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 0u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 8u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   ralloc_free(b.shader);
}

TEST(Agx3cSubgroupReduce, AllLanesReadSlot0PreservesSlot0StoreAndBarrier)
{
   nir_builder b = build_tree_reduction(32, false, false, true, false, false);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "all lanes read slot 0");

   /* Intermediate steps removed; slot-0 stores (2) + trailing barrier (1) +
    * post-reduction reads (2) remain. */
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 2u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 2u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_barrier), 1u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   ralloc_free(b.shader);
}

TEST(Agx3cSubgroupReduce, NegativeNonDeadIntermediateSlotRefused)
{
   nir_builder b = build_tree_reduction(32, true, false, false, false, false);
   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(b.shader));
   EXPECT_GT(count_intr(b.shader, nir_intrinsic_load_shared), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 0u);
   ralloc_free(b.shader);
}

TEST(Agx3cSubgroupReduce, NegativeDynamicSharedReaderRefused)
{
   nir_builder b = build_tree_reduction(32, false, true, false, false, false);
   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(b.shader));
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 0u);
   ralloc_free(b.shader);
}

TEST(Agx3cSubgroupReduce, NegativeDivergentEnclosingIfRefused)
{
   nir_builder b = build_tree_reduction(32, false, false, false, true, false);
   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(b.shader));
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 0u);
   ralloc_free(b.shader);
}

TEST(Agx3cSubgroupReduce, NegativeCrossSubgroupOnlyStepRefused)
{
   /* Only a stride=32 step (cross-subgroup) with no intra-subgroup tail must
    * never be lowered to intra-subgroup shuffle. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "cross-sg-only");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 256;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *off = nir_ishl_imm(&b, tid, 2);
   nir_store_shared(&b, tid, off, .base = 0, .write_mask = 0x1, .align_mul = 4);
   emit_shared_barrier(&b);
   nir_push_if(&b, nir_ult_imm(&b, tid, 32));
   nir_def *lo = nir_load_shared(&b, 1, 32, off, .base = 0, .align_mul = 4);
   nir_def *hi = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, off, 128), .base = 0, .align_mul = 4);
   nir_store_shared(&b, nir_iadd(&b, lo, hi), off, .base = 0, .write_mask = 0x1, .align_mul = 4);
   nir_pop_if(&b, NULL);
   emit_shared_barrier(&b);
   nir_def *out = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 4);
   nir_store_global(&b, out, nir_load_preamble(&b, 1, 64, .base = 0), .align_mul = 4);

   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(b.shader));
   ralloc_free(b.shader);
}

static nir_builder
build_single_lane_prologue(unsigned wg, unsigned alu_chain_len, bool leak_mid)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "single-lane-prologue");
   b.shader->info.workgroup_size[0] = wg;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *y = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *out_addr = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_def *acc = x;
   nir_def *leaked = NULL;
   for (unsigned i = 0; i < alu_chain_len; i++) {
      acc = nir_ffma(&b, acc, y, nir_imm_float(&b, (float)(i + 1) * 0.25f));
      if (i == alu_chain_len / 2)
         leaked = acc;
   }

   nir_def *is0 = nir_ieq_imm(&b, tid, 0);
   nir_def *sel = nir_bcsel(&b, is0, acc, nir_imm_float(&b, 0.0f));
   if (leak_mid)
      sel = nir_fadd(&b, sel, leaked);

   nir_store_global(&b, sel, nir_iadd(&b, out_addr, nir_u2u64(&b, nir_ishl_imm(&b, tid, 2))),
                    .align_mul = 4);
   return b;
}

TEST(Agx3cSubgroupReduce, SinkSingleLanePrologueWg64AndNotWg32)
{
   /* wg=64 with 16 private ALU ops -> sunk into if (tid == 0). */
   nir_builder b64 = build_single_lane_prologue(64, 16, false);
   EXPECT_TRUE(agx_nir_opt_sink_single_lane_select(b64.shader));
   nir_validate_shader(b64.shader, "sink wg64");
   bool found_if = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(b64.shader);
   foreach_list_typed(nir_cf_node, n, node, &impl->body) {
      if (n->type == nir_cf_node_if)
         found_if = true;
   }
   EXPECT_TRUE(found_if);
   ralloc_free(b64.shader);

   /* wg=32 (single subgroup): must NOT wrap in if. */
   nir_builder b32 = build_single_lane_prologue(32, 16, false);
   EXPECT_FALSE(agx_nir_opt_sink_single_lane_select(b32.shader));
   ralloc_free(b32.shader);

   /* Short cone (< threshold): must NOT wrap in if. */
   nir_builder bshort = build_single_lane_prologue(64, 4, false);
   EXPECT_FALSE(agx_nir_opt_sink_single_lane_select(bshort.shader));
   ralloc_free(bshort.shader);

   /* Leaked intermediate value reduces private cone below threshold: must NOT sink. */
   nir_builder bleaked = build_single_lane_prologue(64, 16, true);
   EXPECT_FALSE(agx_nir_opt_sink_single_lane_select(bleaked.shader));
   ralloc_free(bleaked.shader);
}

TEST(Agx3cSubgroupReduce, Wg16ProdBound32RefusesStride16Step)
{
   /* wg=16 with guarded if (tid < 32) producer and stride=16..1: slots 16..31
    * are not backed by active lanes, so matching prod_bound=32 with stride=16
    * would either shuffle from nonexistent lanes 16..31 or drop slots 16..31
    * via nir_reduce. The pass must refuse prod_bound=32 (keeping the stride=16
    * shared loads of s[tid] and s[tid+16] and pre-fill/producer stores and
    * barriers) and may only lower the valid 16-lane stride=8..1 tail. */
   for (bool is_float : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         is_float ? "wg16-guard32-stride16-f32" : "wg16-guard32-stride16-i32");
      b.shader->info.internal = false;
      b.shader->info.workgroup_size[0] = 16;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.shared_size = 32 * 4;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *out_addr = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *off = nir_ishl_imm(&b, tid, 2);

      /* Pre-fill upper half slots 16..31 from lanes 0..15. */
      nir_def *upper_val = is_float ? nir_imm_float(&b, 1.0f) : nir_imm_int(&b, 1);
      nir_store_shared(&b, upper_val, nir_iadd_imm(&b, off, 64),
                       .base = 0, .write_mask = 0x1, .align_mul = 4);
      emit_shared_barrier(&b);

      nir_push_if(&b, nir_ult_imm(&b, tid, 32));
      nir_def *lower_val = is_float ? nir_u2f32(&b, tid) : tid;
      nir_store_shared(&b, lower_val, off, .base = 0, .write_mask = 0x1, .align_mul = 4);
      nir_pop_if(&b, NULL);
      emit_shared_barrier(&b);

      for (unsigned stride = 16; stride >= 1; stride >>= 1) {
         nir_def *cond = (stride == 1) ? nir_ieq_imm(&b, tid, 0)
                                       : nir_ult_imm(&b, tid, stride);
         nir_push_if(&b, cond);
         nir_def *lo_off = (stride == 1) ? nir_imm_int(&b, 0) : off;
         nir_def *hi_off = (stride == 1) ? nir_imm_int(&b, 4)
                                         : nir_iadd_imm(&b, off, stride * 4);
         nir_def *lo = nir_load_shared(&b, 1, 32, lo_off, .base = 0, .align_mul = 4);
         nir_def *hi = nir_load_shared(&b, 1, 32, hi_off, .base = 0, .align_mul = 4);
         nir_def *sum = is_float ? nir_fadd(&b, lo, hi) : nir_iadd(&b, lo, hi);
         nir_store_shared(&b, sum, lo_off, .base = 0,
                          .write_mask = 0x1, .align_mul = 4);
         nir_pop_if(&b, NULL);
         emit_shared_barrier(&b);
      }

      nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
      nir_def *res = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 4);
      nir_store_global(&b, res, out_addr, .align_mul = 4);
      nir_pop_if(&b, NULL);

      EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "wg16 guard32 keeps stride16");

      /* Stride=16 loads (s[tid], s[tid+16]), pre-fill + tid<32 stores, and
       * both barriers before stride=16 must all remain intact. */
      EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 2u);
      EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 2u);
      EXPECT_EQ(count_intr(b.shader, nir_intrinsic_barrier), 2u);

      if (is_float) {
         /* Only 4 butterfly steps (masks 8, 4, 2, 1), never mask 16. */
         EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 4u);
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_shuffle_xor) {
                     ASSERT_TRUE(nir_src_is_const(intr->src[1]));
                     EXPECT_LT(nir_src_as_uint(intr->src[1]), 16u);
                  }
               }
            }
         }
      } else {
         /* Integer tail reduces the stride=16 sum (load_shared + load_shared),
          * not raw tid, so slots 16..31 are preserved. */
         EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_reduce) {
                     nir_instr *src_instr = nir_def_instr(intr->src[0].ssa);
                     ASSERT_EQ(src_instr->type, nir_instr_type_alu);
                     EXPECT_EQ(nir_instr_as_alu(src_instr)->op, nir_op_iadd);
                  }
               }
            }
         }
      }
      ralloc_free(b.shader);
   }
}

static nir_builder
build_uniform_if_wrapped_reduction(bool reader_in_else, bool with_trailing_barrier)
{
   const unsigned wg = 32;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      reader_in_else ? "uniform-if-else-reader" : "uniform-if-merge-reader");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = wg * 12;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *in_addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *out_addr = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *uflag = nir_load_preamble(&b, 1, 32, .base = 8);

   nir_def *v = nir_load_global(&b, 3, 32,
                                nir_iadd(&b, in_addr, nir_u2u64(&b, nir_imul_imm(&b, tid, 12))),
                                .align_mul = 4);
   nir_def *vf = nir_vec2(&b, nir_channel(&b, v, 0), nir_channel(&b, v, 1));
   nir_def *vi = nir_channel(&b, v, 2);

   /* Initialize slot 0 before the uniform branch so the else/skip path reads
    * a well-defined value. */
   nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
   nir_store_shared(&b, vf, nir_imm_int(&b, 0), .base = 0, .write_mask = 0x3, .align_mul = 8);
   nir_store_shared(&b, vi, nir_imm_int(&b, wg * 8), .base = 0, .write_mask = 0x1, .align_mul = 4);
   nir_pop_if(&b, NULL);
   emit_shared_barrier(&b);

   nir_push_if(&b, nir_ine_imm(&b, uflag, 0));
   {
      nir_def *f_off = nir_ishl_imm(&b, tid, 3);
      nir_def *i_off = nir_iadd_imm(&b, nir_ishl_imm(&b, tid, 2), wg * 8);
      nir_store_shared(&b, vf, f_off, .base = 0, .write_mask = 0x3, .align_mul = 8);
      nir_store_shared(&b, vi, i_off, .base = 0, .write_mask = 0x1, .align_mul = 4);
      emit_shared_barrier(&b);

      for (unsigned stride = wg / 2; stride >= 1; stride >>= 1) {
         nir_def *cond = (stride == 1) ? nir_ieq_imm(&b, tid, 0)
                                       : nir_ult_imm(&b, tid, stride);
         nir_push_if(&b, cond);
         nir_def *f_lo_off = (stride == 1) ? nir_imm_int(&b, 0) : f_off;
         nir_def *f_hi_off = (stride == 1) ? nir_imm_int(&b, 8)
                                           : nir_iadd_imm(&b, f_off, stride * 8);
         nir_def *flo = nir_load_shared(&b, 2, 32, f_lo_off, .base = 0, .align_mul = 8);
         nir_def *fhi = nir_load_shared(&b, 2, 32, f_hi_off, .base = 0, .align_mul = 8);
         nir_def *fx = nir_fadd(&b, nir_channel(&b, flo, 0), nir_channel(&b, fhi, 0));
         nir_def *fy = nir_fadd(&b, nir_channel(&b, flo, 1), nir_channel(&b, fhi, 1));
         nir_store_shared(&b, nir_vec2(&b, fx, fy), f_lo_off, .base = 0,
                          .write_mask = 0x3, .align_mul = 8);

         nir_def *i_lo_off = (stride == 1) ? nir_imm_int(&b, wg * 8) : i_off;
         nir_def *i_hi_off = (stride == 1) ? nir_imm_int(&b, wg * 8 + 4)
                                           : nir_iadd_imm(&b, i_off, stride * 4);
         nir_def *ilo = nir_load_shared(&b, 1, 32, i_lo_off, .base = 0, .align_mul = 4);
         nir_def *ihi = nir_load_shared(&b, 1, 32, i_hi_off, .base = 0, .align_mul = 4);
         nir_store_shared(&b, nir_iadd(&b, ilo, ihi), i_lo_off, .base = 0,
                          .write_mask = 0x1, .align_mul = 4);
         nir_pop_if(&b, NULL);
         if (stride > 1 || with_trailing_barrier)
            emit_shared_barrier(&b);
      }
   }
   if (reader_in_else) {
      nir_push_else(&b, NULL);
      nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
      nir_def *rf = nir_load_shared(&b, 2, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 8);
      nir_def *ri = nir_load_shared(&b, 1, 32, nir_imm_int(&b, wg * 8), .base = 0, .align_mul = 4);
      nir_def *res = nir_vec3(&b, nir_channel(&b, rf, 0), nir_channel(&b, rf, 1), ri);
      nir_store_global(&b, res, out_addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   if (!reader_in_else) {
      nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
      nir_def *rf = nir_load_shared(&b, 2, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 8);
      nir_def *ri = nir_load_shared(&b, 1, 32, nir_imm_int(&b, wg * 8), .base = 0, .align_mul = 4);
      nir_def *res = nir_vec3(&b, nir_channel(&b, rf, 0), nir_channel(&b, rf, 1), ri);
      nir_store_global(&b, res, out_addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
   }

   return b;
}

TEST(Agx3cSubgroupReduce, UniformIfWrappedReductionElseBranchReader)
{
   /* With a trailing barrier: lowers the then-branch reduction via slot-0
    * stores without rewriting the non-dominated else-branch loads to SSA. */
   nir_builder b = build_uniform_if_wrapped_reduction(true, true);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "uniform if else reader");

   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 2u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   ralloc_free(b.shader);

   /* Without a trailing barrier: slot-0 store cannot be safely emitted before
    * a trailing barrier, so the reduction must be refused with valid SSA. */
   nir_builder nb = build_uniform_if_wrapped_reduction(true, false);
   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(nb.shader));
   nir_validate_shader(nb.shader, "uniform if else reader no barrier");
   ralloc_free(nb.shader);
}

TEST(Agx3cSubgroupReduce, UniformIfWrappedReductionPostMergeReader)
{
   /* With a trailing barrier: lowers the then-branch reduction and stores
    * slot 0 before the trailing barrier; post-merge tid==0 reader stays a
    * shared load (not rewritten to non-dominating then-branch SSA). */
   nir_builder b = build_uniform_if_wrapped_reduction(false, true);
   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "uniform if post-merge reader");

   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 2u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_shuffle_xor), 10u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   ralloc_free(b.shader);

   /* Without a trailing barrier: must be refused with valid SSA. */
   nir_builder nb = build_uniform_if_wrapped_reduction(false, false);
   EXPECT_FALSE(agx_nir_opt_shared_tree_reduce(nb.shader));
   nir_validate_shader(nb.shader, "uniform if post-merge reader no barrier");
   ralloc_free(nb.shader);
}

TEST(Agx3cSubgroupReduce, Wg64DisjointSharedNoTrailingBarrierPreservesWgBarrier)
{
   /* wg=64, no trailing barrier after stride=1. Subgroup 1 writes disjoint
    * shared array X before the producer and lane 0 reads X[32] after the final
    * step. Deleting all workgroup step barriers would introduce a race; the
    * pass must keep a workgroup-scope shared barrier (or refuse). */
   const unsigned wg = 64;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "wg64-disjoint-shared-no-trailing-bar");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = wg;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   /* Array A (32 * 4 = 128 bytes at offset 0), disjoint array X (64 * 4 = 256 bytes at offset 256). */
   b.shader->info.shared_size = 512;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *out_addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *off = nir_ishl_imm(&b, tid, 2);

   /* Disjoint shared write by all 64 lanes (including subgroup 1) before producer. */
   nir_store_shared(&b, nir_iadd_imm(&b, tid, 100), nir_iadd_imm(&b, off, 256),
                    .base = 0, .write_mask = 0x1, .align_mul = 4);

   /* Subgroup-0 reduction on A[0..31], with WG barriers between steps and NO trailing barrier. */
   nir_push_if(&b, nir_ult_imm(&b, tid, 32));
   nir_store_shared(&b, tid, off, .base = 0, .write_mask = 0x1, .align_mul = 4);
   nir_pop_if(&b, NULL);
   emit_shared_barrier(&b);

   for (unsigned stride = 16; stride >= 1; stride >>= 1) {
      nir_def *cond = (stride == 1) ? nir_ieq_imm(&b, tid, 0)
                                    : nir_ult_imm(&b, tid, stride);
      nir_push_if(&b, cond);
      nir_def *lo_off = (stride == 1) ? nir_imm_int(&b, 0) : off;
      nir_def *hi_off = (stride == 1) ? nir_imm_int(&b, 4)
                                      : nir_iadd_imm(&b, off, stride * 4);
      nir_def *lo = nir_load_shared(&b, 1, 32, lo_off, .base = 0, .align_mul = 4);
      nir_def *hi = nir_load_shared(&b, 1, 32, hi_off, .base = 0, .align_mul = 4);
      nir_store_shared(&b, nir_iadd(&b, lo, hi), lo_off, .base = 0,
                       .write_mask = 0x1, .align_mul = 4);
      nir_pop_if(&b, NULL);
      if (stride > 1)
         emit_shared_barrier(&b);
   }

   /* Lane 0 reads A[0] and subgroup 1's disjoint X[32] after the final step. */
   nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
   nir_def *ra = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 4);
   nir_def *rx = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 256 + 32 * 4),
                                 .base = 0, .align_mul = 4);
   nir_store_global(&b, nir_iadd(&b, ra, rx), out_addr, .align_mul = 4);
   nir_pop_if(&b, NULL);

   EXPECT_TRUE(agx_nir_opt_shared_tree_reduce(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "wg64 disjoint shared barrier preservation");

   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_reduce), 1u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_store_shared), 1u);
   EXPECT_EQ(count_intr(b.shader, nir_intrinsic_load_shared), 1u);

   /* Must preserve at least one workgroup-scope shared barrier. */
   unsigned wg_shared_bars = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_barrier &&
                nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP &&
                nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP &&
                (nir_intrinsic_memory_modes(intr) & nir_var_mem_shared))
               wg_shared_bars++;
         }
      }
   }
   EXPECT_EQ(wg_shared_bars, 1u);
   ralloc_free(b.shader);
}

} /* namespace */
