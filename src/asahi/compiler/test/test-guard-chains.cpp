/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

static struct agx_shader_key
g17p_key(void)
{
   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 16;
   return key;
}

static struct agx_shader_key
agx3c_key(void)
{
   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.reserved_preamble = 16;
   return key;
}

static nir_def *
make_load_global(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
                 unsigned align_mul, gl_access_qualifier access)
{
   nir_def *def = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return def;
}

static void
make_store_global(nir_builder *b, nir_def *val, nir_def *addr,
                  unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_write_mask(intr, nir_component_mask(val->num_components));
}

static unsigned
count_ifs(const nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_cf_node *next = nir_cf_node_next(&block->cf_node);
         if (next && next->type == nir_cf_node_if)
            n++;
      }
   }
   return n;
}

static unsigned
count_if_depth_in_list(const struct exec_list *list, unsigned depth,
                       unsigned *max_depth, unsigned *top_level_ifs)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         const nir_if *nif = nir_cf_node_as_if(node);
         n++;
         if (depth == 0)
            (*top_level_ifs)++;
         if (depth + 1 > *max_depth)
            *max_depth = depth + 1;
         n += count_if_depth_in_list(&nif->then_list, depth + 1, max_depth,
                                     top_level_ifs);
         n += count_if_depth_in_list(&nif->else_list, depth + 1, max_depth,
                                     top_level_ifs);
      } else if (node->type == nir_cf_node_loop) {
         const nir_loop *loop = nir_cf_node_as_loop(node);
         n += count_if_depth_in_list(&loop->body, depth, max_depth,
                                     top_level_ifs);
      }
   }
   return n;
}

static bool
uses_zero_page_const(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_load_const)
               continue;
            const nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            if (lc->def.bit_size == 64 &&
                lc->value[0].u64 == AGX_ZERO_PAGE_ADDRESS)
               return true;
         }
      }
   }
   return false;
}

TEST(GuardChains, FuseNestedGuardWithDominatingLoadNoRedirect)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-dom");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);

   /* Dominating unconditional load of pos[wid]. */
   nir_def *self_addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3));
   nir_def *self_pos =
      make_load_global(&b, 2, 32, self_addr, 8, (gl_access_qualifier)0);

   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *izero = nir_imm_int(&b, 0);
   nir_def *acc_in = nir_imm_int(&b, 1);

   nir_push_if(&b, nir_ine(&b, tid, wid));
   nir_def *j_addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3));
   nir_def *j_pos =
      make_load_global(&b, 2, 32, j_addr, 8, (gl_access_qualifier)0);
   nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                          nir_channel(&b, j_pos, 0));
   nir_def *cond = nir_flt(&b, nir_fmul(&b, dx, dx), nir_imm_float(&b, 4.0f));
   nir_push_if(&b, cond);
   nir_def *v_addr =
      nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3));
   nir_def *vel =
      make_load_global(&b, 2, 32, v_addr, 8, (gl_access_qualifier)0);
   nir_def *fx = nir_fmul(&b, nir_frsq(&b, nir_fmul(&b, dx, dx)),
                          nir_channel(&b, vel, 0));
   nir_def *flag = nir_imm_int(&b, 8);
   nir_pop_if(&b, NULL);
   nir_def *inner_fx = nir_if_phi(&b, fx, zero);
   nir_def *inner_flag = nir_if_phi(&b, flag, izero);
   nir_def *ored = nir_ior(&b, acc_in, inner_flag);
   nir_pop_if(&b, NULL);

   nir_def *out_fx = nir_if_phi(&b, inner_fx, zero);
   nir_def *out_flag = nir_if_phi(&b, ored, acc_in);
   make_store_global(&b, nir_vec2(&b, out_fx, out_flag),
                     nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3)),
                     8);

   EXPECT_EQ(count_ifs(b.shader), 2u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after fuse-dom");
   EXPECT_EQ(count_ifs(b.shader), 1u);
   /* Dominating load proved pos[tid] safe when tid == wid: no zero-page redirect. */
   EXPECT_FALSE(uses_zero_page_const(b.shader));
   ralloc_free(b.shader);
}

TEST(GuardChains, FuseNestedGuardWithoutDominatingLoadRedirectsToZeroPage)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-redir");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *zero = nir_imm_float(&b, 0.0f);

   nir_push_if(&b, nir_ult(&b, tid, nir_imm_int(&b, 16)));
   nir_def *addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
   nir_def *val =
      make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
   nir_push_if(&b, nir_flt(&b, val, nir_imm_float(&b, 1.0f)));
   nir_def *r = nir_frsq(&b, val);
   nir_pop_if(&b, NULL);
   nir_def *inner_r = nir_if_phi(&b, r, zero);
   nir_pop_if(&b, NULL);

   nir_def *res = nir_if_phi(&b, inner_r, zero);
   make_store_global(&b, res,
                     nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)),
                     4);

   EXPECT_EQ(count_ifs(b.shader), 2u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after fuse-redir");
   EXPECT_EQ(count_ifs(b.shader), 1u);
   EXPECT_TRUE(uses_zero_page_const(b.shader));
   ralloc_free(b.shader);
}

TEST(GuardChains, NegativeControlRejectsSideEffectsAndVolatile)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-fuse");
      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *zero = nir_imm_float(&b, 0.0f);

      nir_push_if(&b, nir_ult(&b, tid, nir_imm_int(&b, 16)));
      nir_def *val = make_load_global(
         &b, 1, 32, addr, 4,
         mode == 0 ? ACCESS_VOLATILE : (gl_access_qualifier)0);
      if (mode == 1)
         make_store_global(&b, val, addr, 4);
      nir_push_if(&b, nir_flt(&b, val, nir_imm_float(&b, 1.0f)));
      nir_def *r = nir_frsq(&b, val);
      nir_pop_if(&b, NULL);
      nir_def *ir = nir_if_phi(&b, r, zero);
      nir_pop_if(&b, NULL);

      make_store_global(&b, nir_if_phi(&b, ir, zero), addr, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-fuse");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, HoistAndCoalesceFourIterNBodyChain)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nbody-chain");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
   nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

   nir_def *self_pos = make_load_global(
      &b, 2, 32,
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
      (gl_access_qualifier)0);
   nir_def *self_vel = make_load_global(
      &b, 2, 32,
      nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
      (gl_access_qualifier)0);

   nir_def *acc_x = nir_imm_float(&b, 0.0f);
   nir_def *acc_y = nir_imm_float(&b, 0.0f);
   nir_def *acc_f = nir_imm_int(&b, 0);
   nir_def *zero_i = nir_imm_int(&b, 0);

   for (unsigned k = 0; k < 4; ++k) {
      nir_def *j = k ? nir_iadd_imm(&b, tid, 32 * k) : tid;
      nir_push_if(&b, nir_ine(&b, j, wid));
      nir_def *j_pos = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                             nir_channel(&b, j_pos, 0));
      nir_def *dy = nir_fsub(&b, nir_channel(&b, self_pos, 1),
                             nir_channel(&b, j_pos, 1));
      nir_def *d2 =
         nir_ffma(&b, dx, dx, nir_fmul(&b, dy, dy));
      nir_push_if(&b, nir_flt(&b, d2, r2));
      nir_def *j_vel = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *inv = nir_frsq(&b, d2);
      nir_def *nx = nir_fmul(&b, dx, inv);
      nir_def *ny = nir_fmul(&b, dy, inv);
      nir_def *dvx = nir_fsub(&b, nir_channel(&b, self_vel, 0),
                              nir_channel(&b, j_vel, 0));
      nir_def *dvy = nir_fsub(&b, nir_channel(&b, self_vel, 1),
                              nir_channel(&b, j_vel, 1));
      nir_def *dot = nir_ffma(&b, dvx, nx, nir_fmul(&b, dvy, ny));
      nir_def *imp = nir_fmul(&b, dot, nir_imm_float(&b, -0.312f));
      nir_def *nx_acc = nir_ffma(&b, imp, nx, acc_x);
      nir_def *ny_acc = nir_ffma(&b, imp, ny, acc_y);
      nir_def *approaching = nir_flt(&b, dot, nir_imm_float(&b, 0.0f));
      nir_def *sx = nir_bcsel(&b, approaching, nx_acc, acc_x);
      nir_def *sy = nir_bcsel(&b, approaching, ny_acc, acc_y);
      nir_def *sf =
         nir_bcsel(&b, approaching, nir_imm_int(&b, 8), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
      nir_def *ix = nir_if_phi(&b, sx, acc_x);
      nir_def *iy = nir_if_phi(&b, sy, acc_y);
      nir_def *iflg = nir_if_phi(&b, sf, zero_i);
      nir_def *of = nir_ior(&b, acc_f, iflg);
      nir_pop_if(&b, NULL);

      acc_x = nir_if_phi(&b, ix, acc_x);
      acc_y = nir_if_phi(&b, iy, acc_y);
      acc_f = nir_if_phi(&b, of, acc_f);
   }

   make_store_global(
      &b, nir_vec3(&b, acc_x, acc_y, acc_f),
      nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 4)), 16);

   /* Before: 8 ifs (4 top-level nested pairs). */
   EXPECT_EQ(count_ifs(b.shader), 8u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after nbody-chain");

   /* After: 1 top-level coalesced fast-path guard containing 4 inner ifs = 5 ifs total. */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   unsigned max_depth = 0, top_ifs = 0;
   unsigned total_ifs = count_if_depth_in_list(&impl->body, 0, &max_depth, &top_ifs);
   EXPECT_EQ(top_ifs, 1u);
   EXPECT_EQ(total_ifs, 5u);
   EXPECT_EQ(max_depth, 2u);
   EXPECT_FALSE(uses_zero_page_const(b.shader));

   /* End-to-end compile on both G17P (AGX3B) and G17S (AGX3C) with zero spills. */
   nir_shader *copy = nir_shader_clone(NULL, b.shader);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_shader_key k17p = g17p_key();
   struct agx_shader_part out17p = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &k17p, &out17p));
   EXPECT_EQ(out17p.info.stats.spills, 0u);
   free(out17p.binary);

   agx_preprocess_nir_m5(copy);
   struct agx_shader_key k3c = agx3c_key();
   struct agx_shader_part out3c = {};
   ASSERT_TRUE(agx_compile_shader_nir(copy, &k3c, &out3c));
   EXPECT_EQ(out3c.info.stats.spills, 0u);
   free(out3c.binary);

   ralloc_free(copy);
   ralloc_free(b.shader);
}

TEST(GuardChains, NegativeControlDoesNotHoistAcrossStoreOrSubgroup)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-hoist");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *a1 =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 32)), 2));

      nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v0, nir_imm_float(&b, 1.0f)));
      nir_def *r0;
      if (mode == 0) {
         make_store_global(&b, v0, a0, 4);
         r0 = nir_frsq(&b, v0);
      } else {
         r0 = nir_shuffle_xor(&b, v0, nir_imm_int(&b, 1));
      }
      nir_pop_if(&b, NULL);
      nir_def *p0 = nir_if_phi(&b, r0, v0);

      /* Must NOT hoist v1 above the first if (mode 0 has store, mode 1 has subgroup). */
      nir_def *v1 = make_load_global(&b, 1, 32, a1, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v1, nir_imm_float(&b, 1.0f)));
      nir_def *r1 = nir_frsq(&b, v1);
      nir_pop_if(&b, NULL);
      nir_def *p1 = nir_if_phi(&b, r1, v1);

      make_store_global(&b, nir_fadd(&b, p0, p1), a0, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-hoist");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, NegativeControlDoesNotHoistAcrossStoreOrBarrierBetweenIfs)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-mid-fence");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *a1 =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 32)), 2));

      nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v0, nir_imm_float(&b, 1.0f)));
      nir_def *r0 = nir_frsq(&b, v0);
      nir_pop_if(&b, NULL);
      nir_def *p0 = nir_if_phi(&b, r0, v0);

      /* Side-effecting / non-reorderable instruction in `mid` between two pure
       * ifs: mode 0 has a global store to the same address loaded below; mode 1
       * has a workgroup memory+execution barrier before a shared load. Neither
       * subsequent load may hoist above the fence or the preceding if. */
      nir_def *v1;
      if (mode == 0) {
         make_store_global(&b, p0, a1, 4);
         v1 = make_load_global(&b, 1, 32, a1, 4, (gl_access_qualifier)0);
      } else {
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_shared);
         v1 = nir_load_shared(&b, 1, 32, nir_ishl_imm(&b, tid, 2), .align_mul = 4);
      }

      nir_push_if(&b, nir_flt(&b, v1, nir_imm_float(&b, 1.0f)));
      nir_def *r1 = nir_frsq(&b, v1);
      nir_pop_if(&b, NULL);
      nir_def *p1 = nir_if_phi(&b, r1, v1);

      make_store_global(&b, nir_fadd(&b, p0, p1), a0, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-mid-fence");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, HoistPreservesDominanceAcrossUnhoistedMidBlock)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "hoist-unhoisted-mid");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
   nir_def *a2 =
      nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 64)), 2));

   nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
   nir_def *c0 = nir_flt(&b, v0, nir_imm_float(&b, 1.0f));
   nir_def *c1 = nir_flt(&b, v0, nir_imm_float(&b, 2.0f));

   nir_push_if(&b, c0);
   nir_def *r0 = nir_frsq(&b, v0);
   nir_pop_if(&b, NULL);
   nir_def *p0 = nir_if_phi(&b, r0, v0);

   /* mid_1 has only a pure constant/ALU def (no load, not if_1's condition),
    * so mid_1 is not hoisted. Its def must still block dependent instructions
    * in mid_2 from hoisting above if_0. */
   nir_def *scale = nir_fmul(&b, v0, nir_imm_float(&b, 0.9f));

   nir_push_if(&b, c1);
   nir_def *r1 = nir_frsq(&b, p0);
   nir_pop_if(&b, NULL);
   nir_def *p1 = nir_if_phi(&b, r1, p0);

   /* mid_2 loads v2 and computes a scaled value + condition depending on
    * `scale` (defined in mid_1). v2 itself is independent and may hoist, but
    * `scaled` and `c2` must remain dominated by `scale`. */
   nir_def *v2 = make_load_global(&b, 1, 32, a2, 4, (gl_access_qualifier)0);
   nir_def *scaled = nir_fmul(&b, v2, scale);
   nir_def *c2 = nir_flt(&b, scaled, nir_imm_float(&b, 1.0f));

   nir_push_if(&b, c2);
   nir_def *r2 = nir_frsq(&b, scaled);
   nir_pop_if(&b, NULL);
   nir_def *p2 = nir_if_phi(&b, r2, p1);

   make_store_global(&b, p2, a0, 4);

   agx_nir_opt_guard_chains(b.shader);
   nir_validate_shader(b.shader, "after hoist-unhoisted-mid");
   ralloc_free(b.shader);
}

TEST(GuardChains, WeakenOuterGuardAndSinkIneExclusion)
{
   for (int extra_use = 0; extra_use < 2; ++extra_use) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "weaken-outer-guard");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
      nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

      nir_def *self_pos = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *self_vel = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
         (gl_access_qualifier)0);

      nir_def *acc_x = nir_imm_float(&b, 0.0f);
      nir_def *acc_y = nir_imm_float(&b, 0.0f);
      nir_def *acc_f = nir_imm_int(&b, 0);
      nir_def *ine_escape = NULL;

      for (unsigned k = 0; k < 2; ++k) {
         nir_def *j = k ? nir_iadd_imm(&b, tid, 32 * k) : tid;
         nir_def *ne = nir_ine(&b, j, wid);
         if (k == 1)
            ine_escape = ne;
         nir_def *j_pos = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
            (gl_access_qualifier)ACCESS_CAN_SPECULATE);
         nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                                nir_channel(&b, j_pos, 0));
         nir_def *dy = nir_fsub(&b, nir_channel(&b, self_pos, 1),
                                nir_channel(&b, j_pos, 1));
         nir_def *d2 = nir_ffma(&b, dx, dx, nir_fmul(&b, dy, dy));
         nir_def *hit = nir_iand(&b, nir_flt(&b, d2, r2), ne);

         nir_push_if(&b, hit);
         nir_def *j_vel = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
            (gl_access_qualifier)0);
         nir_def *inv = nir_frsq(&b, d2);
         nir_def *nx = nir_fmul(&b, dx, inv);
         nir_def *ny = nir_fmul(&b, dy, inv);
         nir_def *dvx = nir_fsub(&b, nir_channel(&b, self_vel, 0),
                                 nir_channel(&b, j_vel, 0));
         nir_def *dvy = nir_fsub(&b, nir_channel(&b, self_vel, 1),
                                 nir_channel(&b, j_vel, 1));
         nir_def *dot = nir_ffma(&b, dvx, nx, nir_fmul(&b, dvy, ny));
         nir_def *imp = nir_fmul(&b, dot, nir_imm_float(&b, -0.312f));
         nir_def *nx_acc = nir_ffma(&b, imp, nx, acc_x);
         nir_def *ny_acc = nir_ffma(&b, imp, ny, acc_y);
         nir_def *approaching = nir_flt(&b, dot, nir_imm_float(&b, 0.0f));
         nir_def *sx = nir_bcsel(&b, approaching, nx_acc, acc_x);
         nir_def *sy = nir_bcsel(&b, approaching, ny_acc, acc_y);
         nir_def *sf = nir_ior(&b, acc_f, nir_bcsel(&b, approaching, nir_imm_int(&b, 8), nir_imm_int(&b, 0)));
         nir_pop_if(&b, NULL);
         acc_x = nir_if_phi(&b, sx, acc_x);
         acc_y = nir_if_phi(&b, sy, acc_y);
         acc_f = nir_if_phi(&b, sf, acc_f);
      }

      if (extra_use)
         acc_f = nir_iadd(&b, acc_f, nir_b2i32(&b, ine_escape));

      make_store_global(&b, nir_vec3(&b, acc_x, acc_y, acc_f), out_base, 4);
      nir_validate_shader(b.shader, "before weaken outer guard");
      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after weaken outer guard");

      /* Count nir_op_ine in the entry block (before the coalesced outer if). */
      nir_block *entry = nir_start_block(b.impl);
      unsigned ine_in_entry = 0;
      nir_foreach_instr(instr, entry) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_ine)
            ine_in_entry++;
      }
      if (!extra_use) {
         /* Both ine exclusions must be sunk inside outer->then_list. */
         EXPECT_EQ(ine_in_entry, 0u);
      } else {
         /* Extra-use ine must stay in entry block; the single-use one sinks. */
         EXPECT_EQ(ine_in_entry, 1u);
      }

      struct agx_shader_key key = agx3c_key();
      key.no_preamble = true;
      agx_preprocess_nir(b.shader);
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_EQ(part.info.scratch_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, WeakenOuterGuardConjShapesAndNegativeSink)
{
   /* Verify all match_conj_with_ine() shapes and negative sinking bounds:
    * mode 0: bcsel(fge, false, ine)        -- inverted bcsel path -> inot(fge)
    * mode 1: bcsel(flt, ine, false)        -- non-inverted bcsel path -> flt
    * mode 2: iand(inot(fge), ine)          -- inot(cmp) as p -> inot(fge)
    * mode 3: bcsel(inot(flt), false, ine)  -- inot(cmp) + inverted bcsel -> flt
    * mode 4: iand(flt, inot(ieq))          -- inot(ieq) as q -> flt */
   for (int mode = 0; mode < 5; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "weaken-shapes");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
      nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

      nir_def *self_pos = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *self_vel = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
         (gl_access_qualifier)0);

      nir_def *acc_x = nir_imm_float(&b, 0.0f);
      nir_def *acc_y = nir_imm_float(&b, 0.0f);
      nir_def *acc_f = nir_imm_int(&b, 0);

      /* Shared operand of ine also used inside inner if[0] body: must stay in
       * entry block rather than sinking into outer->then_list. */
      nir_def *shared_body = nir_iadd_imm(&b, tid, 77);
      nir_def *excl_op = NULL;

      for (unsigned k = 0; k < 2; ++k) {
         nir_def *j = k ? nir_iadd_imm(&b, tid, 32 * k) : tid;
         nir_def *lhs = k == 0 ? shared_body
                               : nir_ixor(&b, j, nir_imm_int(&b, 5));
         if (k == 1)
            excl_op = lhs;
         nir_def *q = mode == 4 ? nir_inot(&b, nir_ieq(&b, lhs, wid))
                                : nir_ine(&b, lhs, wid);

         nir_def *j_pos = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
            (gl_access_qualifier)ACCESS_CAN_SPECULATE);
         nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                                nir_channel(&b, j_pos, 0));
         nir_def *dy = nir_fsub(&b, nir_channel(&b, self_pos, 1),
                                nir_channel(&b, j_pos, 1));
         nir_def *d2 = nir_ffma(&b, dx, dx, nir_fmul(&b, dy, dy));

         nir_def *hit = NULL;
         switch (mode) {
         case 0:
            hit = nir_bcsel(&b, nir_fge(&b, d2, r2), nir_imm_false(&b), q);
            break;
         case 1:
            hit = nir_bcsel(&b, nir_flt(&b, d2, r2), q, nir_imm_false(&b));
            break;
         case 2:
            hit = nir_iand(&b, nir_inot(&b, nir_fge(&b, d2, r2)), q);
            break;
         case 3:
            hit = nir_bcsel(&b, nir_inot(&b, nir_flt(&b, d2, r2)),
                            nir_imm_false(&b), q);
            break;
         case 4:
            hit = nir_iand(&b, nir_flt(&b, d2, r2), q);
            break;
         }

         nir_push_if(&b, hit);
         nir_def *j_vel = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
            (gl_access_qualifier)0);
         nir_def *inv = nir_frsq(&b, d2);
         nir_def *nx = nir_fmul(&b, dx, inv);
         nir_def *ny = nir_fmul(&b, dy, inv);
         nir_def *dvx = nir_fsub(&b, nir_channel(&b, self_vel, 0),
                                 nir_channel(&b, j_vel, 0));
         nir_def *dvy = nir_fsub(&b, nir_channel(&b, self_vel, 1),
                                 nir_channel(&b, j_vel, 1));
         nir_def *dot = nir_ffma(&b, dvx, nx, nir_fmul(&b, dvy, ny));
         nir_def *imp = nir_fmul(&b, dot, nir_imm_float(&b, -0.312f));
         nir_def *nx_acc = nir_ffma(&b, imp, nx, acc_x);
         nir_def *ny_acc = nir_ffma(&b, imp, ny, acc_y);
         nir_def *sf = k == 0 ? nir_iadd(&b, acc_f, shared_body)
                              : nir_ior(&b, acc_f, nir_imm_int(&b, 8));
         nir_pop_if(&b, NULL);
         acc_x = nir_if_phi(&b, nx_acc, acc_x);
         acc_y = nir_if_phi(&b, ny_acc, acc_y);
         acc_f = nir_if_phi(&b, sf, acc_f);
      }

      make_store_global(&b, nir_vec3(&b, acc_x, acc_y, acc_f), out_base, 4);
      nir_validate_shader(b.shader, "before weaken shapes");
      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after weaken shapes");

      nir_block *entry = nir_start_block(b.impl);
      unsigned ine_or_ieq_in_entry = 0;
      unsigned conj_in_entry = 0;
      nir_foreach_instr(instr, entry) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_op op = nir_instr_as_alu(instr)->op;
         if (op == nir_op_ine || op == nir_op_ieq)
            ine_or_ieq_in_entry++;
         if (op == nir_op_iand || op == nir_op_bcsel)
            conj_in_entry++;
      }
      EXPECT_EQ(ine_or_ieq_in_entry, 0u) << "mode=" << mode;
      EXPECT_EQ(conj_in_entry, 0u) << "mode=" << mode;
      /* shared_body is also used inside the inner if body -> must stay in entry. */
      EXPECT_EQ(nir_def_instr(shared_body)->block, entry) << "mode=" << mode;
      /* excl_op is used only by ine/ieq -> must sink out of entry. */
      EXPECT_NE(nir_def_instr(excl_op)->block, entry) << "mode=" << mode;

      /* Verify outer coalesced guard polarity: ior of inot(fge) for modes 0
       * and 2, and ior of plain flt for modes 1, 3, and 4. */
      nir_cf_node *next_cf = nir_cf_node_next(&entry->cf_node);
      ASSERT_NE(next_cf, nullptr) << "mode=" << mode;
      ASSERT_EQ(next_cf->type, nir_cf_node_if) << "mode=" << mode;
      nir_if *outer_if = nir_cf_node_as_if(next_cf);
      nir_instr *cond_instr = nir_def_instr(outer_if->condition.ssa);
      ASSERT_EQ(cond_instr->type, nir_instr_type_alu) << "mode=" << mode;
      nir_alu_instr *cond_ior = nir_instr_as_alu(cond_instr);
      ASSERT_EQ(cond_ior->op, nir_op_ior) << "mode=" << mode;
      const bool expect_inverted_fge = (mode == 0 || mode == 2);
      for (unsigned src_i = 0; src_i < 2; ++src_i) {
         nir_instr *term_i = nir_def_instr(cond_ior->src[src_i].src.ssa);
         ASSERT_EQ(term_i->type, nir_instr_type_alu)
            << "mode=" << mode << " src=" << src_i;
         nir_alu_instr *term = nir_instr_as_alu(term_i);
         if (expect_inverted_fge) {
            ASSERT_EQ(term->op, nir_op_inot)
               << "mode=" << mode << " src=" << src_i;
            nir_instr *inner_i = nir_def_instr(term->src[0].src.ssa);
            ASSERT_EQ(inner_i->type, nir_instr_type_alu)
               << "mode=" << mode << " src=" << src_i;
            EXPECT_EQ(nir_instr_as_alu(inner_i)->op, nir_op_fge)
               << "mode=" << mode << " src=" << src_i;
         } else {
            EXPECT_EQ(term->op, nir_op_flt)
               << "mode=" << mode << " src=" << src_i;
         }
      }

      struct agx_shader_key key = agx3c_key();
      key.no_preamble = true;
      agx_preprocess_nir(b.shader);
      struct agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << "mode=" << mode;
      EXPECT_EQ(part.info.scratch_size, 0u) << "mode=" << mode;
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* Negative sinking check: an operand of ine also consumed by a loop-header
    * phi back-edge must stay in the loop header (prev_block), not sink into
    * outer->then_list. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "weaken-loop-phi-neg-sink");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
      nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

      nir_def *self_pos = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *init_idx = tid;
      nir_block *preheader = nir_cursor_current_block(b.cursor);

      nir_loop *loop = nir_push_loop(&b);
      nir_block *header = nir_loop_first_block(loop);
      nir_phi_instr *idx_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&idx_phi->instr, &idx_phi->def, 1, 32);
      nir_phi_instr_add_src(idx_phi, preheader, init_idx);

      b.cursor = nir_after_block(header);
      nir_def *next_idx = nir_iadd_imm(&b, &idx_phi->def, 1);
      nir_def *acc = nir_imm_float(&b, 0.0f);
      nir_def *excl_loop_op = NULL;

      for (unsigned k = 0; k < 2; ++k) {
         /* Keep load addresses and break on idx_phi so next_idx's only uses
          * are k=0's ine and the loop-header phi back-edge. */
         nir_def *load_idx =
            k == 0 ? &idx_phi->def : nir_iadd_imm(&b, &idx_phi->def, 32);
         nir_def *lhs =
            k == 0 ? next_idx : nir_ixor(&b, &idx_phi->def, nir_imm_int(&b, 5));
         if (k == 1)
            excl_loop_op = lhs;
         nir_def *ne = nir_ine(&b, lhs, wid);
         nir_def *j_pos = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, load_idx), 3)),
            8, (gl_access_qualifier)ACCESS_CAN_SPECULATE);
         nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                                nir_channel(&b, j_pos, 0));
         nir_def *dy = nir_fsub(&b, nir_channel(&b, self_pos, 1),
                                nir_channel(&b, j_pos, 1));
         nir_def *d2 = nir_ffma(&b, dx, dx, nir_fmul(&b, dy, dy));
         nir_def *hit = nir_iand(&b, nir_flt(&b, d2, r2), ne);

         nir_push_if(&b, hit);
         nir_def *j_vel = make_load_global(
            &b, 2, 32,
            nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
            (gl_access_qualifier)0);
         nir_def *inv = nir_frsq(&b, d2);
         nir_def *add = nir_ffma(&b, inv, nir_channel(&b, j_vel, 0), acc);
         nir_def *add2 = nir_ffma(&b, inv, nir_channel(&b, j_vel, 1), add);
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, add2, acc);
      }

      make_store_global(&b, acc, out_base, 4);
      nir_push_if(&b, nir_ige(&b, &idx_phi->def, nir_imm_int(&b, 4)));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_block *latch = nir_cursor_current_block(b.cursor);
      nir_phi_instr_add_src(idx_phi, latch, next_idx);
      nir_instr_insert(nir_before_block(header), &idx_phi->instr);
      nir_pop_loop(&b, loop);

      /* Confirm next_idx has only two uses: k=0's ine and the back-edge phi. */
      unsigned next_idx_uses = 0, next_idx_phi_uses = 0, next_idx_ine_uses = 0;
      nir_foreach_use_including_if(u, next_idx) {
         next_idx_uses++;
         ASSERT_FALSE(nir_src_is_if(u));
         nir_instr *ui = nir_src_use_instr(u);
         if (ui->type == nir_instr_type_phi)
            next_idx_phi_uses++;
         else if (ui->type == nir_instr_type_alu &&
                  nir_instr_as_alu(ui)->op == nir_op_ine)
            next_idx_ine_uses++;
      }
      ASSERT_EQ(next_idx_uses, 2u);
      ASSERT_EQ(next_idx_phi_uses, 1u);
      ASSERT_EQ(next_idx_ine_uses, 1u);

      nir_validate_shader(b.shader, "before weaken loop phi");
      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after weaken loop phi");

      /* next_idx feeds idx_phi back-edge -> must stay in header, while
       * excl_loop_op (used only by k=1's ine) sinks out of header. */
      EXPECT_EQ(nir_def_instr(next_idx)->block, header);
      EXPECT_NE(nir_def_instr(excl_loop_op)->block, header);
      unsigned ine_in_header = 0;
      nir_foreach_instr(instr, header) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_ine)
            ine_in_header++;
      }
      EXPECT_EQ(ine_in_header, 0u);
      ralloc_free(b.shader);
   }
}

} // namespace
