/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

#include "agx3_slot_core.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_range_analysis.h"
#include "util/bitset.h"
#include "util/sparse_bitset.h"

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __func__, __LINE__, #cond);       \
         return false;                                                         \
      }                                                                        \
   } while (0)

static bool
test_nir_search_comm_pruning(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "search_comm_pruning");

   nir_def *p0 = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *p1 = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *p2 = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *p3 = nir_load_preamble(&b, 1, 32, .base = 6);

   /* Nested commutative fadd/fmul tree that exercises multi-comm pruning and
    * matches a commutative algebraic rule (a * 1.0 + 0.0 -> a). */
   nir_def *c1 = nir_imm_float(&b, 1.0f);
   nir_def *c0 = nir_imm_float(&b, 0.0f);
   nir_def *c2 = nir_imm_float(&b, 2.0f);

   nir_def *m0 = nir_fmul(&b, c1, p0);
   nir_def *s0 = nir_fadd(&b, c0, m0);

   /* Deep commutative expression with no matching constant collapse, exercising
    * commutative failure pruning across multiple commutative ops. */
   nir_def *t0 = nir_fmul(&b, nir_fadd(&b, p0, p1), nir_fadd(&b, p2, p3));
   nir_def *t1 = nir_fmul(&b, nir_fadd(&b, p1, p2), nir_fadd(&b, p0, p3));
   nir_def *sum = nir_fadd(&b, nir_fadd(&b, t0, t1), nir_fmul(&b, s0, c2));

   /* Integer + bool constant matches in nir_search. */
   nir_def *i0 = nir_iadd(&b, nir_ior(&b, p0, nir_imm_int(&b, 0)),
                          nir_imm_int(&b, 0));
   nir_def *res = nir_fadd(&b, sum, nir_i2f32(&b, i0));

   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 8);
   nir_store_global(&b, res, addr);

   bool progress = nir_opt_algebraic(b.shader);
   CHECK(progress);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "after search_comm_pruning");

   /* Verify that (c0 + (c1 * p0)) collapsed so neither imm 1.0 nor imm 0.0
    * remains in a multiply/add with p0. */
   unsigned fmul_count = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_fmul)
            fmul_count++;
      }
   }
   CHECK(fmul_count == 3);

   ralloc_free(b.shader);
   return true;
}

static bool
test_nir_opt_dce_fastpaths(void)
{
   /* 1. Loop-free shader with if/else and dead chains. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "dce_loop_free");

      nir_def *cond = nir_ine_imm(&b, nir_load_preamble(&b, 1, 32, .base = 0), 0);
      nir_def *a = nir_load_preamble(&b, 1, 32, .base = 2);
      nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 4);

      /* Dead chain before if. */
      nir_def *d0 = nir_iadd_imm(&b, a, 1);
      nir_def *d1 = nir_imul_imm(&b, d0, 3);
      (void)nir_ixor(&b, d1, d0);

      nir_def *live_then = NULL;
      nir_def *live_else = NULL;
      nir_push_if(&b, cond);
      {
         nir_def *dead_arm = nir_iadd_imm(&b, a, 7);
         (void)nir_imul_imm(&b, dead_arm, 5);
         live_then = nir_iadd_imm(&b, a, 11);
      }
      nir_push_else(&b, NULL);
      {
         nir_def *dead_else = nir_iadd_imm(&b, a, 13);
         (void)nir_imul_imm(&b, dead_else, 9);
         live_else = nir_iadd_imm(&b, a, 17);
      }
      nir_pop_if(&b, NULL);

      nir_def *phi = nir_if_phi(&b, live_then, live_else);
      nir_store_global(&b, phi, addr);

      CHECK(nir_opt_dce(b.shader));
      nir_validate_shader(b.shader, "after dce_loop_free");
      /* Second call should find nothing dead. */
      CHECK(!nir_opt_dce(b.shader));
      ralloc_free(b.shader);
   }

   /* 2. Loop with back-edge phi + dead loop-internal chain + no-dead re-run. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "dce_with_loop");

      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 0);
      nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 2);
      nir_def *zero = nir_imm_int(&b, 0);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_variable *idx = nir_local_variable_create(b.impl, glsl_uint_type(), "idx");
      nir_store_var(&b, acc, zero, 1);
      nir_store_var(&b, idx, zero, 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, idx);
         nir_push_if(&b, nir_uge(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         /* Dead chain inside back-edge loop. */
         nir_def *dead0 = nir_imul_imm(&b, i, 7);
         (void)nir_iadd_imm(&b, dead0, 42);

         nir_def *cur = nir_load_var(&b, acc);
         nir_store_var(&b, acc, nir_iadd(&b, cur, i), 1);
         nir_store_var(&b, idx, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_global(&b, nir_load_var(&b, acc), addr);

      nir_lower_vars_to_ssa(b.shader);
      CHECK(nir_opt_dce(b.shader));
      nir_validate_shader(b.shader, "after dce_with_loop");
      /* Exercises the loop !any_dead skip path. */
      CHECK(!nir_opt_dce(b.shader));
      ralloc_free(b.shader);
   }

   return true;
}

static bool
test_nir_cse_single_block_and_hash(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "cse_single_block");

   nir_def *base_a = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *base_b = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 4);

   /* Build >160 expressions with duplicates (including commutative duplicates and
    * duplicate load_preamble intrinsics) so the 256-slot inline CSE set is
    * exercised without resizing. */
   nir_def *acc = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < 90; ++i) {
      nir_def *p0 = nir_load_preamble(&b, 1, 32, .base = 0);
      nir_def *c = nir_imm_int(&b, i + 1);
      nir_def *x1 = nir_iadd(&b, p0, c);
      nir_def *x2 = nir_iadd(&b, c, base_a);
      nir_def *m = nir_imul(&b, x1, x2);
      acc = nir_iadd(&b, acc, nir_ixor(&b, m, base_b));
   }
   nir_store_global(&b, acc, addr);

   CHECK(nir_opt_cse(b.shader));
   nir_validate_shader(b.shader, "after cse_single_block");
   CHECK(!nir_opt_cse(b.shader));

   ralloc_free(b.shader);
   return true;
}

static bool
test_nir_range_fp_class_tables_and_cache(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "range_fp_class");

   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *y = nir_load_preamble(&b, 1, 32, .base = 2);
   nir_def *pos2 = nir_imm_float(&b, 2.0f);
   nir_def *neg_half = nir_imm_float(&b, -0.5f);
   nir_def *pzero = nir_imm_float(&b, 0.0f);
   nir_def *nzero = nir_imm_float(&b, -0.0f);

   /* x * x is non-negative (or NaN); with preserve_signed_zero, never -0.0 either. */
   nir_def *sq = nir_fmul(&b, x, x);
   nir_def_as_alu(sq)->fp_math_ctrl = nir_fp_preserve_signed_zero;
   /* (-0.5) * 2.0 = -1.0 (or LT_ZERO_GT_NEG_ONE / LT_NEG_ONE / NEG_ONE). */
   nir_def *prod_c = nir_fmul(&b, neg_half, pos2);
   /* (+0.0) + (-0.0) = +0.0. */
   nir_def *zsum = nir_fadd(&b, pzero, nzero);
   /* ffma(abs(x), 2.0, 2.0) >= 2.0 (or NaN/+Inf). */
   nir_def *ax = nir_fabs(&b, x);
   nir_def *fma = nir_ffma(&b, ax, pos2, pos2);
   /* mulz(y, 0.0) = +0.0. */
   nir_def *mz = nir_fmulz(&b, y, pzero);

   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_store_global(&b, nir_fadd(&b, nir_fadd(&b, sq, prod_c),
                                 nir_fadd(&b, zsum, nir_fadd(&b, fma, mz))),
                    addr);

   nir_index_ssa_defs(b.impl);
   nir_fp_analysis_state fp_state = nir_create_fp_analysis_state(b.impl);

   /* First queries populate cache; second queries hit cache fast path. */
   fp_class_mask m_sq_1 = nir_analyze_fp_class(&fp_state, sq);
   fp_class_mask m_sq_2 = nir_analyze_fp_class(&fp_state, sq);
   CHECK(m_sq_1 == m_sq_2);
   CHECK((m_sq_1 & (FP_CLASS_ANY_NEG | FP_CLASS_NEG_ZERO)) == 0);

   fp_class_mask m_prod = nir_analyze_fp_class(&fp_state, prod_c);
   CHECK(m_prod == nir_analyze_fp_class(&fp_state, prod_c));
   CHECK((m_prod & FP_CLASS_ANY_POS) == 0);
   CHECK((m_prod & FP_CLASS_NEG_ONE) != 0);

   fp_class_mask m_zsum = nir_analyze_fp_class(&fp_state, zsum);
   CHECK((m_zsum & FP_CLASS_POS_ZERO) != 0);
   CHECK((m_zsum & (FP_CLASS_ANY_POS | FP_CLASS_ANY_NEG | FP_CLASS_NAN)) == 0);

   fp_class_mask m_fma = nir_analyze_fp_class(&fp_state, fma);
   CHECK((m_fma & (FP_CLASS_ANY_NEG | FP_CLASS_ANY_ZERO |
                   FP_CLASS_GT_ZERO_LT_POS_ONE | FP_CLASS_POS_ONE)) == 0);
   CHECK((m_fma & FP_CLASS_GT_POS_ONE) != 0);

   fp_class_mask m_mz = nir_analyze_fp_class(&fp_state, mz);
   CHECK((m_mz & FP_CLASS_POS_ZERO) != 0);
   CHECK((m_mz & (FP_CLASS_ANY_POS | FP_CLASS_ANY_NEG | FP_CLASS_NAN)) == 0);

   /* Invalidate and re-query to verify clean cache reset. */
   nir_invalidate_fp_analysis_state(&fp_state);
   CHECK(nir_analyze_fp_class(&fp_state, sq) == m_sq_1);
   CHECK(nir_analyze_fp_class(&fp_state, fma) == m_fma);

   nir_free_fp_analysis_state(&fp_state);
   ralloc_free(b.shader);
   return true;
}

static bool
test_nir_divergence_fastpaths(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "divergence_fastpaths");

   nir_def *u = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *c = nir_imm_int(&b, 42);
   nir_def *und = nir_undef(&b, 1, 32);

   nir_def *u_add = nir_iadd(&b, u, c);
   nir_def *d_add = nir_iadd(&b, gid, u_add);

   nir_def *inv_in_loop = NULL;
   nir_def *var_in_loop = NULL;
   nir_variable *ctr = nir_local_variable_create(b.impl, glsl_uint_type(), "ctr");
   nir_store_var(&b, ctr, gid, 1);
   nir_push_loop(&b);
   {
      nir_def *lc = nir_imm_int(&b, 7);
      inv_in_loop = nir_iadd(&b, u_add, lc);
      nir_def *cur = nir_load_var(&b, ctr);
      var_in_loop = nir_iadd(&b, cur, inv_in_loop);
      nir_store_var(&b, ctr, var_in_loop, 1);
      nir_push_if(&b, nir_ieq_imm(&b, var_in_loop, 0));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
   }
   nir_pop_loop(&b, NULL);

   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 2);
   nir_store_global(&b, nir_iadd(&b, nir_iadd(&b, d_add, nir_load_var(&b, ctr)), und), addr);

   nir_lower_vars_to_ssa(b.shader);
   nir_divergence_analysis(b.shader);

   CHECK(!c->divergent && !c->loop_invariant);
   CHECK(!und->divergent && !und->loop_invariant);
   CHECK(!u_add->divergent);
   CHECK(d_add->divergent);
   CHECK(!inv_in_loop->divergent && inv_in_loop->loop_invariant);
   CHECK(var_in_loop->divergent && !var_in_loop->loop_invariant);

   ralloc_free(b.shader);
   return true;
}

static bool
test_sparse_bitset_from_dense(void)
{
   void *mem_ctx = ralloc_context(NULL);
   const unsigned nbits = 3100;
   BITSET_DECLARE(dense, 3100);
   memset(dense, 0, sizeof(dense));

   /* Empty bitset produces empty sparse bitset. */
   struct u_sparse_bitset sb;
   u_sparse_bitset_init(&sb, nbits, mem_ctx);
   u_sparse_bitset_from_dense(&sb, dense, nbits);
   for (unsigned i = 0; i < nbits; ++i)
      CHECK(!u_sparse_bitset_test(&sb, i));
   u_sparse_bitset_free(&sb);

   /* Sparse bits across multiple 1024-bit chunks with an empty middle chunk. */
   const unsigned set_bits[] = {0, 1, 31, 32, 63, 64, 511, 1023, 2048, 2049, 3071, 3099};
   for (unsigned i = 0; i < ARRAY_SIZE(set_bits); ++i)
      BITSET_SET(dense, set_bits[i]);

   u_sparse_bitset_init(&sb, nbits, mem_ctx);
   u_sparse_bitset_from_dense(&sb, dense, nbits);
   for (unsigned i = 0; i < nbits; ++i)
      CHECK(u_sparse_bitset_test(&sb, i) == BITSET_TEST(dense, i));

   u_sparse_bitset_free(&sb);
   ralloc_free(mem_ctx);
   return true;
}

static bool
test_agx3_slot_core_compact_tokens_and_overflow(void)
{
   /* Special tokens must sit below the huge static TPS range. */
   CHECK(AGX3_SLOT_TEX_COORD_TOKEN < AGX3_SLOT_TPS_TOKEN_BASE);
   CHECK(AGX3_SLOT_DYNAMIC_TPS_TOKEN < AGX3_SLOT_TPS_TOKEN_BASE);
   CHECK(AGX3_SLOT_TEX_COORD_TOKEN != AGX3_SLOT_DYNAMIC_TPS_TOKEN);

   /* 1. Special token dependency + WAW + WAR tracking. */
   {
      struct agx3_slot_instr instrs[4] = {
         {
            .is_async = true,
            .nr_dsts = 2,
            .dst = {AGX3_SLOT_TEX_COORD_TOKEN, AGX3_SLOT_DYNAMIC_TPS_TOKEN},
            .nr_srcs = 1,
            .src = {10},
            .latches_srcs = true,
         },
         {
            /* Synchronous write to latched src 10 triggers WAR wait. */
            .is_async = false,
            .nr_dsts = 1,
            .dst = {10},
         },
         {
            /* Consumer of TEX_COORD_TOKEN already waited above -> no wait. */
            .is_async = false,
            .nr_srcs = 1,
            .src = {AGX3_SLOT_TEX_COORD_TOKEN},
         },
         {
            /* Async op writing DYNAMIC_TPS_TOKEN reuses retired slot cleanly. */
            .is_async = true,
            .nr_dsts = 1,
            .dst = {AGX3_SLOT_DYNAMIC_TPS_TOKEN},
         },
      };
      struct agx3_slot_result out[4] = {0};
      CHECK(agx3_slot_assign(instrs, 4, 6, out));
      CHECK(out[0].slot == 1 && out[0].wait_mask == 0);
      CHECK(out[1].wait_mask == (1u << 0));
      CHECK(out[2].wait_mask == 0);
      CHECK(out[3].slot >= 1 && out[3].slot <= 6);
   }

   /* 2. >32 tracked tokens per slot (exercises UINT8_MAX full-scan fallback)
    * plus if/else save/restore/join and slot retirement. */
   {
      struct agx3_slot_instr instrs[12] = {0};
      /* 6 async ops on preferred slot 1 with 8 distinct dsts each = 48 tokens > 32. */
      for (unsigned k = 0; k < 6; ++k) {
         instrs[k].is_async = true;
         instrs[k].preferred_slot = 1;
         instrs[k].reissues_live_slot = true;
         instrs[k].nr_dsts = 8;
         for (unsigned d = 0; d < 8; ++d)
            instrs[k].dst[d] = (uint16_t)(k * 8 + d);
         instrs[k].nr_srcs = 8;
         for (unsigned s = 0; s < 8; ++s)
            instrs[k].src[s] = (uint16_t)(100 + k * 8 + s);
         instrs[k].latches_srcs = true;
      }
      /* Push an if arm, wait slot 1 in then, skip in else, join at pop. */
      instrs[6].cf_push = 1;
      instrs[6].cf_arm = true;
      instrs[7].nr_srcs = 1;
      instrs[7].src[0] = 47; /* reads last dst -> waits slot 1 in then arm */
      instrs[7].cf_else = true;
      /* Else arm writes nothing, leaves slot 1 pending. */
      instrs[8].cf_pop = 1;
      /* After join, slot 1 is still pending from else arm; reading dst 0 must wait. */
      instrs[9].nr_srcs = 1;
      instrs[9].src[0] = 0;
      /* Writing latched src 147 after slot 1 was waited needs no wait. */
      instrs[10].nr_dsts = 1;
      instrs[10].dst[0] = 147;
      /* Draining CF barrier retires everything cleanly. */
      instrs[11].cf = AGX3_SLOT_CF_DRAIN;

      struct agx3_slot_result out[12] = {0};
      CHECK(agx3_slot_assign(instrs, 12, 6, out));
      CHECK(out[7].wait_mask == (1u << 0));
      CHECK(out[9].wait_mask == (1u << 0));
      CHECK(out[10].wait_mask == 0);

      bool drained[12] = {false};
      agx3_slot_retire_at_cf(instrs, 12, 0, out, drained);
      CHECK(!drained[11]);
   }

   return true;
}

static bool
test_agx_backend_compile_wave2_g17p_and_g17s(void)
{
   const enum agx_chip chips[2] = {AGX_CHIP_G17P, AGX_CHIP_G17S};
   for (unsigned c = 0; c < 2; ++c) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "wave2_backend_loop");

      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 0);
      nir_def *scale = nir_load_preamble(&b, 1, 32, .base = 2);
      nir_def *bias = nir_load_preamble(&b, 1, 32, .base = 4);
      nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 6);

      nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_variable *idx = nir_local_variable_create(b.impl, glsl_uint_type(), "idx");
      nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);
      nir_store_var(&b, idx, nir_imm_int(&b, 0), 1);

      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, idx);
         nir_push_if(&b, nir_uge(&b, i, n));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         /* Loop-invariant expression exercises agx_nir_opt_licm dense liveness. */
         nir_def *inv = nir_fadd(&b, nir_fmul(&b, scale, scale), bias);
         nir_def *cur = nir_load_var(&b, acc);
         nir_def *term = nir_ffma(&b, nir_u2f32(&b, i), inv, cur);
         nir_store_var(&b, acc, term, 1);
         nir_store_var(&b, idx, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_store_global(&b, nir_load_var(&b, acc), addr);

      struct agx_shader_key key = {
         .dev.chip = chips[c],
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .dev.soft_fault = true,
         .reserved_preamble = 8,
         .g17p_native_uniforms = true,
         .promote_constants = true,
      };

      agx_preprocess_nir_native(b.shader, chips[c]);

      struct agx_shader_part compiled = {0};
      CHECK(agx_compile_shader_nir(b.shader, &key, &compiled));
      CHECK(compiled.info.binary_size > 0);
      free(compiled.binary);
      ralloc_free(b.shader);
   }

   return true;
}

static bool
test_licm_sparse_liveness(void)
{
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "licm_sparse_liveness");
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);

   /* Make the dense slab hundreds of MiB while keeping live sets small. */
   for (unsigned i = 1; i <= 12000; ++i) {
      nir_push_if(&b, nir_ult_imm(&b, id, i));
      nir_store_global(&b, nir_ior_imm(&b, id, i), addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
   }

   nir_def *limit = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *angle = nir_load_preamble(&b, 1, 32, .base = 6);
   nir_variable *index =
      nir_local_variable_create(b.impl, glsl_uint_type(), "index");
   nir_store_var(&b, index, nir_imm_int(&b, 0), 1);

   nir_loop *loop = nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, index);
   nir_break_if(&b, nir_uge(&b, i, limit));
   nir_def *invariant = nir_fsinc_agx3(&b, angle);
   nir_def *varying = nir_fmul(&b, invariant, nir_u2f32(&b, i));
   nir_store_global(&b, varying, addr, .align_mul = 4);
   nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);

   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);
   nir_validate_shader(b.shader, "before sparse LICM");
   nir_metadata_require(b.impl, nir_metadata_block_index);

   nir_block *preheader = nir_loop_predecessor_block(loop);
   nir_block *body = nir_def_block(varying);
   CHECK((uint64_t)b.impl->num_blocks * BITSET_WORDS(b.impl->ssa_alloc) >
         262144);
   CHECK(nir_def_block(invariant) == body);

#ifdef __linux__
   struct rusage before, after;
   CHECK(getrusage(RUSAGE_SELF, &before) == 0);
#endif
   CHECK(agx_nir_opt_licm(b.shader, 64, false));
#ifdef __linux__
   /* The dense slab alone would grow peak RSS by more than 256 MiB. */
   CHECK(getrusage(RUSAGE_SELF, &after) == 0);
   CHECK(after.ru_maxrss - before.ru_maxrss < 256 * 1024);
#endif

   CHECK(nir_def_block(invariant) == preheader);
   CHECK(nir_def_block(varying) == body);
   nir_validate_shader(b.shader, "after sparse LICM");

   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   return true;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && strcmp(argv[1], "--licm-sparse-liveness") == 0)
      return test_licm_sparse_liveness() ? 0 : 1;

   bool ok = true;
   ok &= test_nir_search_comm_pruning();
   ok &= test_nir_opt_dce_fastpaths();
   ok &= test_nir_cse_single_block_and_hash();
   ok &= test_nir_range_fp_class_tables_and_cache();
   ok &= test_nir_divergence_fastpaths();
   ok &= test_sparse_bitset_from_dense();
   ok &= test_agx3_slot_core_compact_tokens_and_overflow();
   ok &= test_agx_backend_compile_wave2_g17p_and_g17s();
   if (!ok)
      return 1;
   printf("PASS: test-agx3-pso-compile-wave2\n");
   return 0;
}
