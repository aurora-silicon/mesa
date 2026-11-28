/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <vector>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compiler.h"
#include "agx_nir.h"
}

namespace {

nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

nir_def *
load_preamble_ex(nir_builder *b, unsigned comps, unsigned bits, unsigned base)
{
   nir_def *d = nir_load_preamble(b, comps, bits);
   nir_intrinsic_set_base(nir_def_as_intrinsic(d), base);
   return d;
}

nir_def *
load_push_u32(nir_builder *b, unsigned offset, unsigned range = 32)
{
   nir_def *d = nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset));
   nir_intrinsic_set_range(nir_def_as_intrinsic(d), range);
   return d;
}

nir_def *
load_global_ex(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
               unsigned align_mul,
               gl_access_qualifier access = (gl_access_qualifier)0)
{
   nir_def *d = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return d;
}

nir_def *
load_global_const_ex(nir_builder *b, unsigned comps, unsigned bits,
                     nir_def *addr, unsigned align_mul,
                     gl_access_qualifier access)
{
   nir_def *d = nir_load_global_constant(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return d;
}

void
store_global_ex(nir_builder *b, nir_def *val, nir_def *addr, unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
}

void
emit_workgroup_barrier(nir_builder *b)
{
   nir_intrinsic_instr *intr = nir_barrier(b);
   nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
}

nir_def *
make_buffer_addr(nir_builder *b, unsigned preamble_base, nir_def *byte_offset)
{
   nir_def *set = load_preamble_ex(b, 1, 64, preamble_base);
   nir_def *desc = load_global_const_ex(
      b, 4, 32, set, 16,
      (gl_access_qualifier)(ACCESS_CAN_SPECULATE | ACCESS_BUFFER_BINDING_AGX));
   nir_def *base = nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                          nir_channel(b, desc, 1));
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

unsigned
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

unsigned
count_loops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(parent)) == block)
            n++;
      }
   }
   return n;
}

unsigned
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

/* Extra reads of the bit-pack accumulator `d` outside its select chain. */
enum bitpack_extra_use {
   BITPACK_CHAIN_ONLY,
   /* `extra += d` inside the loop. */
   BITPACK_D_IN_LOOP,
   /* `with_bit = d | (1u << i); extra += with_bit; d = cond ? with_bit : d`. */
   BITPACK_IOR_IN_LOOP,
   /* `extra = d` before the update, read after the loop. */
   BITPACK_D_AFTER_LOOP,
};

/* Helper building a counted [0, trips) loop with >= 20 body instructions,
 * global loads, and an optional monotonic 32-bit bit-pack accumulator. */
nir_shader *
build_const_bitpack_loop(unsigned trips, bool bottom_tested, bool with_load,
                         bool with_barrier,
                         bitpack_extra_use use = BITPACK_CHAIN_ONLY)
{
   nir_builder b = make_compute_shader("const-bitpack-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
   nir_variable *acc =
      nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_variable *extra =
      nir_local_variable_create(b.impl, glsl_uint_type(), "extra");

   nir_def *c0 = load_push_u32(&b, 0, 32);
   nir_def *c1 = load_push_u32(&b, 4, 32);
   nir_def *stride = load_push_u32(&b, 8, 32);

   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, extra, nir_imm_int(&b, 0), 1);

   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      if (!bottom_tested)
         nir_break_if(&b, nir_ige(&b, i, nir_imm_int(&b, trips)));

      nir_def *a = NULL;
      nir_def *bv = NULL;
      if (with_load) {
         nir_def *pat = load_global_ex(
            &b, 4, 32, make_buffer_addr(&b, 4, nir_ishl_imm(&b, i, 4)), 16, ro);
         nir_def *x0 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 0), c0),
                                nir_imul(&b, nir_channel(&b, pat, 1), c1));
         nir_def *y0 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 0), c1),
                                nir_imul(&b, nir_channel(&b, pat, 1), c0));
         nir_def *x1 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 2), c0),
                                nir_imul(&b, nir_channel(&b, pat, 3), c1));
         nir_def *y1 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 2), c1),
                                nir_imul(&b, nir_channel(&b, pat, 3), c0));
         nir_def *o0 = nir_iadd(&b, x0, nir_imul(&b, y0, stride));
         nir_def *o1 = nir_iadd(&b, x1, nir_imul(&b, y1, stride));
         a = load_global_ex(
            &b, 1, 32, make_buffer_addr(&b, 8, nir_ishl_imm(&b, o0, 2)), 4, ro);
         bv = load_global_ex(
            &b, 1, 32, make_buffer_addr(&b, 8, nir_ishl_imm(&b, o1, 2)), 4, ro);
      } else {
         nir_def *v = nir_iadd(&b, i, c0);
         for (unsigned k = 0; k < 20; ++k)
            v = nir_ixor(&b, nir_iadd(&b, v, c1), nir_imul(&b, v, stride));
         a = v;
         bv = c1;
      }

      if (with_barrier)
         emit_workgroup_barrier(&b);

      nir_def *d = nir_load_var(&b, acc);
      if (use == BITPACK_D_IN_LOOP)
         nir_store_var(&b, extra, nir_iadd(&b, nir_load_var(&b, extra), d), 1);
      else if (use == BITPACK_D_AFTER_LOOP)
         nir_store_var(&b, extra, d, 1);

      nir_def *mask =
         nir_ishl(&b, nir_imm_int(&b, 1), nir_iand_imm(&b, i, 31));
      nir_def *with_bit = nir_ior(&b, d, mask);
      if (use == BITPACK_IOR_IN_LOOP)
         nir_store_var(&b, extra,
                       nir_iadd(&b, nir_load_var(&b, extra), with_bit), 1);
      nir_def *d_next = nir_bcsel(&b, nir_ilt(&b, a, bv), with_bit, d);
      nir_store_var(&b, acc, d_next, 1);

      nir_def *next = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, iv, next, 1);
      if (bottom_tested)
         nir_break_if(&b, nir_ige(&b, next, nir_imm_int(&b, trips)));
   }
   nir_pop_loop(&b, loop);

   nir_def *res = nir_iadd(&b, nir_load_var(&b, acc), nir_load_var(&b, extra));
   store_global_ex(&b, res, make_buffer_addr(&b, 12, nir_imm_int(&b, 0)), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "const-bitpack setup");
   return b.shader;
}

} /* namespace */

/* Bottom-tested rotated loop (do { ... ++iv; } while (iv < n)) must lower
 * idiv/imod and udiv/umod into a running (q, r) recurrence just like a
 * top-tested loop, while refusing step != 1. */
TEST(FmWave2IvDivMod, BottomTestedLoop)
{
   for (int mode = 0; mode < 4; ++mode) {
      /* mode 0: signed guarded D > 0, step 1
       * mode 1: signed unknown-sign D, step 1
       * mode 2: unsigned udiv/umod, step 1
       * mode 3: signed step 2 (must not match) */
      nir_builder b = make_compute_shader("iv-divmod-bottom");
      nir_def *out = make_buffer_addr(&b, 8, nir_imm_int(&b, 0));
      nir_def *n = load_push_u32(&b, 0, 16);
      nir_def *d = load_push_u32(&b, 4, 16);

      nir_if *guard = NULL;
      if (mode == 0)
         guard = nir_push_if(&b, nir_ilt(&b, nir_imm_int(&b, 0), d));

      nir_def *iv_init = nir_imm_int(&b, 0);
      nir_def *acc_init = nir_imm_int(&b, 0);

      nir_loop *loop = nir_push_loop(&b);
      nir_block *pre = nir_loop_predecessor_block(loop);
      nir_block *header = nir_loop_first_block(loop);

      nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
      nir_phi_instr_add_src(iv_phi, pre, iv_init);

      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(acc_phi, pre, acc_init);

      nir_def *q = mode == 2 ? nir_udiv(&b, &iv_phi->def, d)
                             : nir_idiv(&b, &iv_phi->def, d);
      nir_def *r = mode == 2 ? nir_umod(&b, &iv_phi->def, d)
                             : nir_imod(&b, &iv_phi->def, d);
      nir_def *sum = nir_iadd(&b, &acc_phi->def, nir_iadd(&b, q, r));
      nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, mode == 3 ? 2 : 1);

      nir_def *done =
         mode == 2 ? nir_uge(&b, iv_next, n) : nir_ige(&b, iv_next, n);
      nir_push_if(&b, done);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_block *latch = nir_loop_last_block(loop);
      nir_phi_instr_add_src(iv_phi, latch, iv_next);
      nir_phi_instr_add_src(acc_phi, latch, sum);
      nir_instr_insert(nir_before_block(header), &iv_phi->instr);
      nir_instr_insert(nir_before_block(header), &acc_phi->instr);
      nir_pop_loop(&b, loop);

      store_global_ex(&b, sum, out, 4);
      if (guard)
         nir_pop_if(&b, guard);
      nir_validate_shader(b.shader, "iv-divmod-bottom setup");

      bool changed = agx_nir_opt_iv_divmod(b.shader);
      nir_validate_shader(b.shader, "iv-divmod-bottom lowered");
      if (mode < 3) {
         EXPECT_TRUE(changed) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 0u) << "mode " << mode;
         if (mode == 0)
            EXPECT_EQ(count_alu_op(b.shader, nir_op_ineg), 0u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 1u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 1u);
      }
      ralloc_free(b.shader);
   }
}

/* Push-constant divisor and bound (`bs * bs`) loaded inside the loop (before
 * LICM) must still match, hoist the divisor load to the preheader, and share
 * a single (q, r) recurrence across idiv and imod. */
TEST(FmWave2IvDivMod, InLoopPushConstantDivisorAndBound)
{
   nir_builder b = make_compute_shader("iv-divmod-inloop-pc");
   nir_def *out = make_buffer_addr(&b, 8, nir_imm_int(&b, 0));
   nir_def *zero = nir_imm_int(&b, 0);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *pre = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);

   nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
   nir_phi_instr_add_src(iv_phi, pre, zero);

   nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
   nir_phi_instr_add_src(acc_phi, pre, zero);

   nir_def *bs0 = load_push_u32(&b, 20, 32);
   nir_def *bs1 = load_push_u32(&b, 20, 32);
   nir_def *bound = nir_imul(&b, bs0, bs1);
   nir_push_if(&b, nir_ige(&b, &iv_phi->def, bound));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *d_q = load_push_u32(&b, 20, 32);
   nir_def *d_r = load_push_u32(&b, 20, 32);
   nir_def *q = nir_idiv(&b, &iv_phi->def, d_q);
   nir_def *r = nir_imod(&b, &iv_phi->def, d_r);
   nir_def *sum = nir_iadd(&b, &acc_phi->def, nir_iadd(&b, q, r));
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 1);

   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_instr_add_src(iv_phi, latch, iv_next);
   nir_phi_instr_add_src(acc_phi, latch, sum);
   nir_instr_insert(nir_before_block(header), &iv_phi->instr);
   nir_instr_insert(nir_before_block(header), &acc_phi->instr);
   nir_pop_loop(&b, loop);

   store_global_ex(&b, &acc_phi->def, out, 4);
   nir_validate_shader(b.shader, "iv-divmod-inloop-pc setup");

   EXPECT_TRUE(agx_nir_opt_iv_divmod(b.shader));
   nir_validate_shader(b.shader, "iv-divmod-inloop-pc lowered");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
   ralloc_free(b.shader);
}

TEST(FmWave2ConstLoopUnroll, TopTested32TripBitpack8x)
{
   nir_shader *s = build_const_bitpack_loop(32, false, true, false);
   ASSERT_EQ(count_loops(s), 1u);
   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "const-bitpack 8x");

   /* Exact divisor: still one loop (no remainder loop) and widest == 8. */
   EXPECT_EQ(count_loops(s), 1u);
   EXPECT_EQ(widest, 8u);
   /* Bit-pack funnel-shift strength reduction: one ushr by 8 and one ishl by
    * 24 for the chunk merge; no variable 1u << (iv + k) shifts remain. */
   EXPECT_EQ(count_alu_op(s, nir_op_ushr), 1u);
   /* Running again is a no-op (partially_unrolled is set). */
   EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
   ralloc_free(s);
}

TEST(FmWave2ConstLoopUnroll, BottomTested32TripBitpack4x)
{
   nir_shader *s = build_const_bitpack_loop(32, true, true, false);
   ASSERT_EQ(count_loops(s), 1u);
   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 4, &widest));
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "const-bitpack bottom 4x");

   EXPECT_EQ(count_loops(s), 1u);
   EXPECT_EQ(widest, 4u);
   EXPECT_EQ(count_alu_op(s, nir_op_ushr), 1u);
   ralloc_free(s);
}

TEST(FmWave2ConstLoopUnroll, RefusalsAndEscapingAccumulator)
{
   /* Non-divisible trip count (30 is divisible by neither 8 nor 4). */
   {
      nir_shader *s = build_const_bitpack_loop(30, false, true, false);
      unsigned widest = 0;
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
      EXPECT_EQ(widest, 0u);
      ralloc_free(s);
   }

   /* Pure-ALU loop without memory loads is left alone. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, false, false);
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, NULL));
      ralloc_free(s);
   }

   /* Loop with a workgroup barrier is refused. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, true, true);
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, NULL));
      ralloc_free(s);
   }

   /* When `d` is read inside the loop outside the bit-pack chain, unrolling
    * still succeeds, but bit-pack funnel-shift strength reduction is skipped
    * so intermediate values of `d` stay exact. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, true, false,
                                               BITPACK_D_IN_LOOP);
      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "const-bitpack escaping d");
      EXPECT_EQ(widest, 8u);
      EXPECT_EQ(count_alu_op(s, nir_op_ushr), 0u);
      ralloc_free(s);
   }
}

/* `with_bit = d | (1u << i)` read outside the select: after the funnel-shift
 * rewrite `d` carries the shifted partial word on every outer iteration but
 * the first, so that reader would see wrong bits. Unrolling still happens;
 * the bit-pack rewrite must not. Same shapes as the 8x and bottom 4x tests
 * above apart from the extra reader. */
TEST(FmWave2ConstLoopUnroll, EscapingOrKeepsExactBitpack)
{
   for (unsigned bottom = 0; bottom < 2; ++bottom) {
      const unsigned factor = bottom ? 4 : 8;
      nir_shader *s = build_const_bitpack_loop(32, bottom, true, false,
                                               BITPACK_IOR_IN_LOOP);
      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, factor, &widest));
      NIR_PASS(_, s, nir_opt_copy_prop);
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "const-bitpack escaping ior");
      EXPECT_EQ(count_loops(s), 1u);
      EXPECT_EQ(widest, factor);
      EXPECT_EQ(count_alu_op(s, nir_op_ushr), 0u) << "bottom=" << bottom;
      ralloc_free(s);
   }
}

/* Bottom-tested loop whose `d` from before the last update is read after the
 * loop: that exit value is the previous word, which the funnel-shift form
 * holds shifted left by the factor, so the bit-pack rewrite must not run. */
TEST(FmWave2ConstLoopUnroll, BottomTestedDAfterLoopKeepsExactBitpack)
{
   for (unsigned factor = 4; factor <= 8; factor *= 2) {
      nir_shader *s = build_const_bitpack_loop(32, true, true, false,
                                               BITPACK_D_AFTER_LOOP);
      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, factor, &widest));
      NIR_PASS(_, s, nir_opt_copy_prop);
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "const-bitpack bottom d after loop");
      EXPECT_EQ(count_loops(s), 1u);
      EXPECT_EQ(widest, factor);
      EXPECT_EQ(count_alu_op(s, nir_op_ushr), 0u) << "factor=" << factor;
      ralloc_free(s);
   }
}

/* Five additive terms in a byte-load address (as in a 2D neighborhood loop
 * with border + row*width + col + k) must parse in agx_nir_linear and merge
 * into one 4-byte vector load. */
TEST(FmWave2MergeBytes, FiveTermAffineAddress)
{
   nir_builder b = make_compute_shader("merge-5term-u8");
   nir_def *t0 = load_push_u32(&b, 0, 32);
   nir_def *t1 = load_push_u32(&b, 4, 32);
   nir_def *t2 = load_push_u32(&b, 8, 32);
   nir_def *t3 = load_push_u32(&b, 12, 32);
   nir_def *t4 = load_push_u32(&b, 16, 32);

   nir_def *base_off =
      nir_iadd(&b, nir_iadd(&b, nir_iadd(&b, t0, t1), nir_iadd(&b, t2, t3)),
               t4);
   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 4; ++k) {
      nir_def *v = load_global_ex(
         &b, 1, 8, make_buffer_addr(&b, 4, nir_iadd_imm(&b, base_off, k)), 1,
         ro);
      sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
   }
   store_global_ex(&b, sum, make_buffer_addr(&b, 8, nir_imm_int(&b, 0)), 4);

   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge 5-term u8");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 1u);
   ralloc_free(b.shader);
}

/* In a two-level indirect load chain (p0 -> addr0 -> img0; p1 -> addr1 -> img1)
 * inside a compute block, the ILP scheduler must issue independent outer load
 * p1 ahead of addr0 (which stalls on p0) instead of serializing by source
 * order. */
TEST(FmWave2IlpSchedule, TwoLevelIndirectLoadInterleave)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *blk = agx_start_block(b->shader);
   b->cursor = agx_after_block(blk);

   agx_index base0 = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index base1 = agx_preload(b, agx_register(6, AGX_SIZE_64));

   agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p0 = agx_device_load_to(
      b, p0, base0, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *alu_a0 = agx_iadd_to(b, a0, p0, agx_immediate(1), 0);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base1, a0, AGX_FORMAT_I32, 0x1, 0, false);

   agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p1 = agx_device_load_to(
      b, p1, base0, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a1, p1, agx_immediate(1), 0);
   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v1, base1, a1, AGX_FORMAT_I32, 0x1, 0, false);

   agx_unit_test(b, agx_iadd(b, v0, v1, 0));

   agx_ilp_schedule(b->shader);

   int pos_p0 = -1, pos_p1 = -1, pos_a0 = -1, idx = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I == load_p0)
         pos_p0 = idx;
      else if (I == load_p1)
         pos_p1 = idx;
      else if (I == alu_a0)
         pos_a0 = idx;
      idx++;
   }
   ASSERT_GE(pos_p0, 0);
   ASSERT_GE(pos_p1, 0);
   ASSERT_GE(pos_a0, 0);
   EXPECT_LT(pos_p1, pos_a0);
   ralloc_free(mem);
}

/* A device store between two indirect load chains must still serialize later
 * loads after the store (preserving RAW/WAR ordering). */
TEST(FmWave2IlpSchedule, StoreStillSerializesLaterLoads)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *blk = agx_start_block(b->shader);
   b->cursor = agx_after_block(blk);

   agx_index base0 = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index base1 = agx_preload(b, agx_register(6, AGX_SIZE_64));

   agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, p0, base0, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0,
                      false);
   agx_index a0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a0, p0, agx_immediate(1), 0);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base1, a0, AGX_FORMAT_I32, 0x1, 0, false);

   agx_instr *st = agx_device_store(b, v0, base0, agx_immediate(4),
                                    AGX_FORMAT_I32, 0x1, 0, false);

   agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p1 = agx_device_load_to(
      b, p1, base0, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a1, p1, agx_immediate(1), 0);
   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v1, base1, a1, AGX_FORMAT_I32, 0x1, 0, false);

   agx_unit_test(b, agx_iadd(b, v0, v1, 0));

   agx_ilp_schedule(b->shader);

   int pos_st = -1, pos_p1 = -1, idx = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I == st)
         pos_st = idx;
      else if (I == load_p1)
         pos_p1 = idx;
      idx++;
   }
   ASSERT_GE(pos_st, 0);
   ASSERT_GE(pos_p1, 0);
   EXPECT_GT(pos_p1, pos_st);
   ralloc_free(mem);
}

/* A compute loop block with two device loads (such as an 8x-unrolled byte loop
 * vectorized into two u8x4 loads) must hoist the second load ahead of the
 * first load's ALU consumers. */
TEST(FmWave2IlpSchedule, TwoLoadComputeLoopHoists)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
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

   b->cursor = agx_after_block(entry);
   agx_index base = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index acc = agx_preload(b, agx_register(6, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0,
                      false);
   agx_index s0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *add0 = agx_iadd_to(b, s0, acc, v0, 0);

   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load1 = agx_device_load_to(
      b, v1, base, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index s1 = agx_iadd(b, s0, v1, 0);
   agx_unit_test(b, s1);

   agx_ilp_schedule(b->shader);

   int pos_add0 = -1, pos_load1 = -1, idx = 0;
   agx_foreach_instr_in_block(body, I) {
      if (I == add0)
         pos_add0 = idx;
      else if (I == load1)
         pos_load1 = idx;
      idx++;
   }
   ASSERT_GE(pos_add0, 0);
   ASSERT_GE(pos_load1, 0);
   EXPECT_LT(pos_load1, pos_add0);
   ralloc_free(mem);
}

/* An 8-step unrolled Abel prefix-sum chain (`w += s; s += v_k`) collapses into
 * a balanced tree with `s_phi << 3` and loop-carried depth 1-2. */
TEST(FmWave2AbelChain, BalancedTree8x)
{
   nir_builder b = make_compute_shader("abel-tree-8x");
   nir_def *n = load_push_u32(&b, 0, 16);
   nir_def *zero = nir_imm_int(&b, 0);
   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *pre = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);

   nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
   nir_phi_instr_add_src(iv_phi, pre, zero);

   nir_phi_instr *w_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&w_phi->instr, &w_phi->def, 1, 32);
   nir_phi_instr_add_src(w_phi, pre, zero);

   nir_phi_instr *s_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&s_phi->instr, &s_phi->def, 1, 32);
   nir_phi_instr_add_src(s_phi, pre, zero);

   nir_push_if(&b, nir_ige(&b, &iv_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *w = &w_phi->def;
   nir_def *s = &s_phi->def;
   for (unsigned k = 0; k < 8; ++k) {
      nir_def *v = nir_u2u32(
         &b, load_global_ex(
                &b, 1, 8, make_buffer_addr(&b, 4, nir_iadd_imm(&b, &iv_phi->def, k)),
                1, ro));
      w = nir_iadd(&b, w, s);
      s = nir_iadd(&b, s, v);
   }
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 8);

   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_instr_add_src(iv_phi, latch, iv_next);
   nir_phi_instr_add_src(w_phi, latch, w);
   nir_phi_instr_add_src(s_phi, latch, s);
   nir_instr_insert(nir_before_block(header), &iv_phi->instr);
   nir_instr_insert(nir_before_block(header), &w_phi->instr);
   nir_instr_insert(nir_before_block(header), &s_phi->instr);
   nir_pop_loop(&b, loop);

   store_global_ex(&b, nir_iadd(&b, &w_phi->def, &s_phi->def),
                   make_buffer_addr(&b, 8, zero), 4);
   nir_validate_shader(b.shader, "abel-tree-8x setup");

   EXPECT_TRUE(agx_nir_opt_factor_loop_mac(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "abel-tree-8x lowered");
   /* Emits 4 shifts: a01<<1, b45<<1, sa<<2, s_phi<<3. */
   EXPECT_GE(count_alu_op(b.shader, nir_op_ishl), 4u);
   ralloc_free(b.shader);
}

static unsigned
count_b2b_raw_and_max_ic(agx_block *blk, unsigned *max_ic_out)
{
   unsigned raw = 0;
   unsigned cur_ic = 0, max_ic = 0;
   uint32_t prev_dst = ~0u;
   agx_foreach_instr_in_block(blk, I) {
      if (I->op == AGX_OPCODE_PHI)
         continue;
      bool hit_raw = false;
      if (prev_dst != ~0u) {
         agx_foreach_ssa_src(I, s) {
            if (I->src[s].value == prev_dst)
               hit_raw = true;
         }
      }
      if (hit_raw)
         raw++;
      bool is_ic = I->op == AGX_OPCODE_BFEIL || I->op == AGX_OPCODE_BFI ||
                   I->op == AGX_OPCODE_IMAD || I->op == AGX_OPCODE_ASR ||
                   I->op == AGX_OPCODE_EXTR;
      if (is_ic) {
         cur_ic++;
         if (cur_ic > max_ic)
            max_ic = cur_ic;
      } else {
         cur_ic = 0;
      }
      prev_dst = (I->nr_dests > 0 && I->dest[0].type == AGX_INDEX_NORMAL)
                    ? I->dest[0].value
                    : ~0u;
   }
   *max_ic_out = max_ic;
   return raw;
}

static void
build_xorshift_loop_body(agx_builder *b, agx_block *body,
                         const agx_index init_st[16], agx_index out_cur[16])
{
   b->cursor = agx_after_block(body);
   agx_instr *phis[16];
   agx_index cur[16];
   for (unsigned i = 0; i < 16; ++i) {
      cur[i] = agx_temp(b->shader, AGX_SIZE_32);
      phis[i] = agx_phi_to(b, cur[i], 2);
      phis[i]->src[0] = init_st[i];
   }
   for (unsigned it = 0; it < 2; ++it) {
      for (unsigned g = 0; g < 4; ++g) {
         unsigned prev_g = (g + 3) & 3;
         for (unsigned lane = 0; lane < 4; ++lane) {
            unsigned idx = g * 4 + lane;
            unsigned pidx = prev_g * 4 + lane;
            agx_index sh = agx_temp(b->shader, AGX_SIZE_32);
            agx_bfeil_to(b, sh, agx_zero(), cur[idx], agx_immediate(9), 0);
            agx_index xo = agx_temp(b->shader, AGX_SIZE_32);
            agx_xor_to(b, xo, cur[pidx], sh);
            agx_index ad = agx_temp(b->shader, AGX_SIZE_32);
            agx_iadd_to(b, ad, cur[idx], xo, 0);
            cur[idx] = ad;
         }
      }
   }
   for (unsigned i = 0; i < 16; ++i) {
      phis[i]->src[1] = cur[i];
      out_cur[i] = cur[i];
   }
}

/* On AGX3B compute pure-ALU loops, bounded lookahead inside the current
 * occupancy tier plus IC/SCIB pipe spacing eliminates bfeil->xor RAW stalls
 * without clumping >2 IC instructions back to back, while AGX3C keeps its
 * existing schedule. */
TEST(FmWave2IlpSchedule, Agx3bComputeLoopInterleavesIcAndScib)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = enc;
      key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G16G : AGX_CHIP_G17S;
      key.has_scratch = true;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, agx_shader_info);
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

      /* Wider prologue in entry (program_peak >= 120 halves) that dies before
       * entering the narrower compute loop (peak ~36 halves). */
      b->cursor = agx_after_block(entry);
      agx_index st[16];
      for (unsigned i = 0; i < 16; ++i)
         st[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
      agx_index wide[48];
      for (unsigned i = 0; i < 48; ++i)
         wide[i] = agx_iadd(b, st[i & 15], agx_immediate(i + 1), 0);
      agx_index wsum = wide[0];
      for (unsigned i = 1; i < 48; ++i)
         wsum = agx_iadd(b, wsum, wide[i], 0);
      agx_unit_test(b, wsum);

      agx_index cur[16];
      build_xorshift_loop_body(b, body, st, cur);

      b->cursor = agx_after_block(exit);
      agx_index esum = cur[0];
      for (unsigned i = 1; i < 16; ++i)
         esum = agx_iadd(b, esum, cur[i], 0);
      agx_unit_test(b, esum);

      agx_validate(b->shader, "agx3b compute ic/scib before");
      agx_ilp_schedule(b->shader);
      agx_validate(b->shader, "agx3b compute ic/scib after");

      unsigned max_ic = 0;
      unsigned raw = count_b2b_raw_and_max_ic(body, &max_ic);
      if (enc == AGX_ISA_ENCODING_AGX3B) {
         EXPECT_LE(max_ic, 2u);
         EXPECT_LE(raw, 4u);
      } else {
         /* AGX3C does not enable AGX3B's bounded ALU lookahead or IC pacing. */
         EXPECT_GT(raw, 16u);
      }
      ralloc_free(mem);
   }
}

/* On AGX3B compute pure-ALU loops:
 * - When peak sits well below the 104-half occupancy step (nkeep=20, ~46h),
 *   bounded ALU lookahead (+8 halves) interleaves 4 independent 3-stage chains
 *   so RAW back-to-back stalls disappear.
 * - When peak is already near the 104-half occupancy step (nkeep=46, ~98h >
 *   104 - 8), the tier cap keeps lookahead at AGX_ILP_SLACK=2 so scheduling
 *   does not widen pressure across the 104-half step.
 * - When pre-RA pressure exceeds 256 halves (nkeep=130), scheduling must not
 *   abort inside agx_occupancy_for_register_count(). */
TEST(FmWave2IlpSchedule, Agx3bComputeLoopRespectsOccupancyTierCapAndHighPressure)
{
   for (unsigned nkeep : {12u, 20u, 32u, 44u, 46u, 130u}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G16G;
      key.has_scratch = true;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, agx_shader_info);
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

      b->cursor = agx_after_block(entry);
      agx_index base = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index keep[140];
      for (unsigned i = 0; i < nkeep; ++i)
         keep[i] = agx_iadd(b, base, agx_immediate(i + 1), 0);

      /* Wider prologue in entry so program_peak exceeds the loop's tier. */
      agx_index wide[40];
      for (unsigned i = 0; i < 40; ++i)
         wide[i] = agx_iadd(b, base, agx_immediate(200 + i), 0);
      agx_index wsum = wide[0];
      for (unsigned i = 1; i < 40; ++i)
         wsum = agx_iadd(b, wsum, wide[i], 0);
      agx_unit_test(b, wsum);

      /* Single loop-carried accumulator fed by 4 independent shifted-iadd
       * chains: interleaving all 4 chains needs +6..+8 halves of live temps,
       * exceeding AGX_ILP_SLACK=2 unless AGX3B's bounded lookahead is active. */
      b->cursor = agx_after_block(body);
      agx_index acc = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *acc_phi = agx_phi_to(b, acc, 2);
      acc_phi->src[0] = base;

      /* Accumulate each lane immediately in source order so baseline peak is
       * only +2..+4 halves; interleaving all 4 independent chains needs +6..+8
       * halves, which AGX3B allows well below 104h but refuses near 104h. */
      agx_index next_acc = acc;
      for (unsigned lane = 0; lane < 4; ++lane) {
         agx_index s1 = agx_temp(b->shader, AGX_SIZE_32);
         agx_iadd_to(b, s1, agx_zero(), keep[lane], 3);
         agx_index s2 = agx_temp(b->shader, AGX_SIZE_32);
         agx_iadd_to(b, s2, agx_zero(), s1, 4);
         agx_index xo = agx_temp(b->shader, AGX_SIZE_32);
         agx_xor_to(b, xo, keep[lane + 4], s2);
         agx_index ad = agx_temp(b->shader, AGX_SIZE_32);
         agx_iadd_to(b, ad, keep[lane + 8], xo, 0);
         next_acc = agx_iadd(b, next_acc, ad, 0);
      }
      acc_phi->src[1] = next_acc;

      b->cursor = agx_after_block(exit);
      agx_index esum = next_acc;
      for (unsigned i = 0; i < nkeep; ++i)
         esum = agx_iadd(b, esum, keep[i], 0);
      agx_unit_test(b, esum);

      agx_validate(b->shader, "agx3b tier cap before");
      agx_ilp_schedule(b->shader);
      agx_validate(b->shader, "agx3b tier cap after");

      unsigned max_ic = 0;
      unsigned raw = count_b2b_raw_and_max_ic(body, &max_ic);
      if (nkeep == 12u || nkeep == 32u) {
         /* Peak ~30h (inside 48h granule) and ~70h (inside 96h granule):
          * bounded ALU lookahead (+8h) interleaves all 4 chains. */
         EXPECT_LE(raw, 4u) << "nkeep=" << nkeep;
      } else if (nkeep == 20u || nkeep == 44u || nkeep == 46u) {
         /* Peak ~46h (near 48h), ~94h (near 96h), and ~98h (near 104h):
          * granule/occupancy cap refuses the +6-half interleaved schedule. */
         EXPECT_GE(raw, 12u) << "nkeep=" << nkeep;
      }
      ralloc_free(mem);
   }
}

static int
fm_block_peak_halves(agx_context *ctx, agx_block *block)
{
   agx_compute_liveness(ctx);
   std::vector<uint8_t> size16(MAX2(ctx->alloc, 1u), 0);
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         size16[I->dest[d].value] = agx_index_size_16(I->dest[d]);
   }
   std::vector<bool> live(MAX2(ctx->alloc, 1u), false);
   int out = 0;
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v) {
      live[v] = true;
      out += size16[v];
   }
   int pressure = 0, peak = 0;
   agx_foreach_instr_in_block_rev(block, I) {
      int delta = 0;
      agx_foreach_ssa_dest(I, d) {
         if (live[I->dest[d].value])
            delta -= agx_index_size_16(I->dest[d]);
      }
      agx_foreach_ssa_src(I, s) {
         bool dupe = false;
         for (unsigned i = 0; i < s; ++i) {
            if (agx_is_equiv(I->src[i], I->src[s])) {
               dupe = true;
               break;
            }
         }
         if (!dupe && !live[I->src[s].value])
            delta += agx_index_size_16(I->src[s]);
      }
      pressure += delta;
      peak = MAX2(peak, pressure);
      agx_foreach_ssa_dest(I, d)
         live[I->dest[d].value] = false;
      agx_foreach_ssa_src(I, s)
         live[I->src[s].value] = true;
   }
   return out + peak;
}

/* AGX3B integer-ALU loop lookahead must stay inside the loop's current
 * register granule / occupancy tier (48, 96, 104, 112, 128, 136, 144, 160,
 * 184) under allocator-compatible widths, reserved registers (any_cf),
 * preloads, and late kills across G15S, G16G, and G17P. */
TEST(FmWave2IlpSchedule, Agx3bComputeIntLoopPreserves48And96HalfGranules)
{
   auto run_one = [&](enum agx_chip chip, unsigned nr_vec3, unsigned live16,
                      unsigned alt_extra16, bool any_cf, unsigned preload_reg,
                      bool use_bfeil, bool run_ilp, int &before_out,
                      int &after_out, unsigned &alloc_out) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      key.has_scratch = true;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, agx_shader_info);
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = any_cf;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

      agx_block *pre = agx_start_block(b->shader);
      agx_block *loop_pre = agx_test_block(b->shader);
      agx_block *body = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *loop_exit = agx_test_block(b->shader);
      agx_block *alt1 = agx_test_block(b->shader);
      agx_block *alt2 = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      body->loop_header = true;
      agx_block_add_successor(pre, loop_pre);
      agx_block_add_successor(pre, alt1);
      agx_block_add_successor(loop_pre, body);
      agx_block_add_successor(body, latch);
      agx_block_add_successor(body, loop_exit);
      agx_block_add_successor(latch, body);
      agx_block_add_successor(loop_exit, exit);
      agx_block_add_successor(alt1, alt2);
      agx_block_add_successor(alt2, exit);

      b->cursor = agx_after_block(pre);
      agx_index base = agx_preload(b, agx_register(preload_reg, AGX_SIZE_32));
      agx_index srcs32[8];
      for (unsigned i = 0; i < 8; ++i)
         srcs32[i] = agx_iadd(b, base, agx_immediate(i + 1), 0);

      std::vector<agx_index> held;
      for (unsigned i = 0; i < nr_vec3; ++i) {
         agx_index vec = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
         agx_index cs[3];
         for (unsigned j = 0; j < 3; ++j)
            cs[j] = agx_mov_imm(b, 32, 100 + 4 * i + j);
         agx_instr *col = agx_collect_to(b, vec, 3);
         for (unsigned j = 0; j < 3; ++j)
            col->src[j] = cs[j];
         held.push_back(vec);
      }
      for (unsigned i = 0; i < live16; ++i)
         held.push_back(agx_mov_imm(b, 16, 1 + i));

      b->cursor = agx_after_block(alt1);
      std::vector<agx_index> alt_vals;
      for (unsigned i = 0; i < alt_extra16; ++i)
         alt_vals.push_back(agx_mov_imm(b, 16, 0x300 + i));
      b->cursor = agx_after_block(alt2);
      for (agx_index v : alt_vals)
         agx_unit_test(b, v);

      b->cursor = agx_after_block(body);
      agx_index acc = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *acc_phi = agx_phi_to(b, acc, 2);
      acc_phi->src[0] = base;

      agx_index next_acc = acc;
      for (unsigned lane = 0; lane < 4; ++lane) {
         agx_index s1 = agx_temp(b->shader, AGX_SIZE_32);
         if (use_bfeil)
            agx_bfeil_to(b, s1, agx_zero(), srcs32[lane], agx_immediate(9), 0);
         else
            agx_iadd_to(b, s1, agx_zero(), srcs32[lane], 3);
         agx_index s2 = agx_temp(b->shader, AGX_SIZE_32);
         agx_iadd_to(b, s2, agx_zero(), s1, 4);
         agx_index xo = agx_temp(b->shader, AGX_SIZE_32);
         agx_xor_to(b, xo, srcs32[lane + 4], s2);
         agx_index ad = agx_temp(b->shader, AGX_SIZE_32);
         agx_iadd_to(b, ad, srcs32[lane], xo, 0);
         next_acc = agx_iadd(b, next_acc, ad, 0);
      }
      acc_phi->src[1] = next_acc;

      b->cursor = agx_after_block(exit);
      agx_index res = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *res_phi = agx_phi_to(b, res, 2);
      res_phi->src[0] = next_acc;
      res_phi->src[1] = base;
      agx_unit_test(b, res);
      for (agx_index v : held) {
         unsigned ch = agx_channels(v);
         if (ch == 1) {
            agx_unit_test(b, v);
         } else {
            agx_instr *sp = agx_split(b, ch, v);
            for (unsigned j = 0; j < ch; ++j)
               sp->dest[j] = agx_temp(b->shader, AGX_SIZE_32);
            for (unsigned j = 0; j < ch; ++j)
               agx_unit_test(b, sp->dest[j]);
         }
      }

      agx_validate(b->shader, "before pressure");
      agx_pressure_schedule(b->shader);
      agx_validate(b->shader, "after pressure");
      before_out = fm_block_peak_halves(b->shader, body);

      if (run_ilp) {
         agx_ilp_schedule(b->shader);
         agx_validate(b->shader, "after ilp");
      }
      after_out = fm_block_peak_halves(b->shader, body);

      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      alloc_out = b->shader->max_reg + 1;
      ralloc_free(mem);
   };

   static const unsigned boundaries[] = {48, 96, 104, 112, 128, 136, 144, 160, 184};

   /* Targeted regression cases: any_cf=true, preload at half 2, BFEIL chains,
    * 32-bit vec3 rounding near the 104-half (7 vec3 + 20 scalars), 136-half,
    * 160-half (3 vec3 + 108 scalars), and 184-half boundaries across all
    * AGX3B chips. */
   for (enum agx_chip chip : {AGX_CHIP_G15S, AGX_CHIP_G16G, AGX_CHIP_G17P}) {
      struct {
         unsigned nr_vec3;
         unsigned live16;
         unsigned alt_extra16;
         unsigned bound;
      } repros[] = {
         { 3u, 108u, 24u, 160u },
         { 7u, 20u,  24u, 104u },
         { 3u, 84u,  24u, 136u },
         { 3u, 132u, 24u, 184u },
      };
      for (const auto &r : repros) {
         int b0 = 0, a0 = 0, b1 = 0, a1 = 0;
         unsigned alloc0 = 0, alloc1 = 0;
         run_one(chip, r.nr_vec3, r.live16, r.alt_extra16, true, 2, true,
                 false, b0, a0, alloc0);
         run_one(chip, r.nr_vec3, r.live16, r.alt_extra16, true, 2, true,
                 true, b1, a1, alloc1);
         if (alloc0 <= r.bound) {
            EXPECT_LE(alloc1, r.bound)
               << "chip=" << chip << " nr_vec3=" << r.nr_vec3
               << " live16=" << r.live16 << " bound=" << r.bound
               << " alloc0=" << alloc0 << " alloc1=" << alloc1;
         }
      }
   }

   /* Sweep all 9 AGX3B boundaries (48..184) with scalars and 1/3/7 live vec3s,
    * with and without any_cf/preload-offset/BFEIL: ILP must never cross any
    * hardware register boundary compared to pressure-only RA. */
   for (bool any_cf : {false, true}) {
      for (bool use_bfeil : {false, true}) {
         const unsigned preload_reg = any_cf ? 2u : 0u;
         for (unsigned nr_vec3 : {0u, 1u, 3u, 7u}) {
            for (unsigned live16 = 4; live16 <= 148; live16 += 2) {
               if (nr_vec3 * 8 + live16 + 26 > 188)
                  continue;
               int b0 = 0, a0 = 0, b1 = 0, a1 = 0;
               unsigned alloc0 = 0, alloc1 = 0;
               run_one(AGX_CHIP_G16G, nr_vec3, live16, 24, any_cf, preload_reg,
                       use_bfeil, false, b0, a0, alloc0);
               run_one(AGX_CHIP_G16G, nr_vec3, live16, 24, any_cf, preload_reg,
                       use_bfeil, true, b1, a1, alloc1);
               for (unsigned bound : boundaries) {
                  if (alloc0 <= bound) {
                     EXPECT_LE(alloc1, bound)
                        << "any_cf=" << any_cf << " bfeil=" << use_bfeil
                        << " nr_vec3=" << nr_vec3 << " live16=" << live16
                        << " bound=" << bound << " before=" << b1
                        << " after=" << a1 << " alloc0=" << alloc0
                        << " alloc1=" << alloc1;
                  }
               }
            }
         }
      }
   }
}

/* Verify the <= peak + 8 growth bound independently of granule boundaries:
 * when body pressure is well inside the 96-half tier (~56 halves vs cap 92),
 * six 64-bit lanes reading base with a -> b0 -> c(a, b0) (where b0 does not kill a) have
 * every ready instruction growing by +4 once live reaches 8, so ILP exceeds
 * fwd_cap=8 and grows by +12 halves unless the <= ra_peak + 8 bound rejects
 * it. Reverting the <= ra_peak + 8 bound fails this test. */
TEST(FmWave2IlpSchedule, Agx3bComputeIntLoopEnforces8HalfGrowthBound)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G16G;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *pre = agx_start_block(b->shader);
   agx_block *loop_pre = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *loop_exit = agx_test_block(b->shader);
   agx_block *alt1 = agx_test_block(b->shader);
   agx_block *alt2 = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   body->loop_header = true;
   agx_block_add_successor(pre, loop_pre);
   agx_block_add_successor(pre, alt1);
   agx_block_add_successor(loop_pre, body);
   agx_block_add_successor(body, latch);
   agx_block_add_successor(body, loop_exit);
   agx_block_add_successor(latch, body);
   agx_block_add_successor(loop_exit, exit);
   agx_block_add_successor(alt1, alt2);
   agx_block_add_successor(alt2, exit);

   auto iadd64 = [&](agx_index x, agx_index y, unsigned shift = 0) {
      agx_index dst = agx_temp(b->shader, AGX_SIZE_64);
      agx_iadd_to(b, dst, x, y, shift);
      return dst;
   };

   b->cursor = agx_after_block(pre);
   agx_index base = agx_preload(b, agx_register(0, AGX_SIZE_64));
   std::vector<agx_index> keep;
   for (unsigned i = 0; i < 20; ++i)
      keep.push_back(agx_mov_imm(b, 32, 1 + i));

   /* Raise program_ra_peak above the 96-half tier so max_cap = tier_cap = 92
    * leaves > 30 halves above body's pressure-scheduled peak (~56). */
   b->cursor = agx_after_block(alt1);
   agx_index wide[48];
   for (unsigned i = 0; i < 48; ++i)
      wide[i] = agx_mov_imm(b, 32, 100 + i);
   b->cursor = agx_after_block(alt2);
   for (unsigned i = 0; i < 48; ++i)
      agx_unit_test(b, wide[i]);

   b->cursor = agx_after_block(body);
   agx_index acc = agx_temp(b->shader, AGX_SIZE_64);
   agx_instr *acc_phi = agx_phi_to(b, acc, 2);
   acc_phi->src[0] = base;

   agx_index cur = acc;
   for (unsigned lane = 0; lane < 6; ++lane) {
      agx_index a = iadd64(base, agx_immediate(lane + 1), 1);
      agx_index b0 = iadd64(a, agx_immediate(lane + 17), 2);
      agx_index c = iadd64(a, b0);
      cur = iadd64(cur, c);
   }
   acc_phi->src[1] = cur;

   b->cursor = agx_after_block(exit);
   agx_index res = agx_temp(b->shader, AGX_SIZE_64);
   agx_instr *res_phi = agx_phi_to(b, res, 2);
   res_phi->src[0] = cur;
   res_phi->src[1] = base;
   agx_unit_test(b, res);
   for (agx_index k : keep)
      agx_unit_test(b, k);

   agx_validate(b->shader, "before pressure");
   agx_pressure_schedule(b->shader);
   agx_validate(b->shader, "after pressure");
   const int before = fm_block_peak_halves(b->shader, body);

   agx_ilp_schedule(b->shader);
   agx_validate(b->shader, "after ilp");
   const int after = fm_block_peak_halves(b->shader, body);

   EXPECT_LE(after, before + 8) << "before=" << before << " after=" << after;
   ralloc_free(mem);
}

/* The IC/SCIB pipe split and 2-cycle IC pacing must stay strictly gated to
 * AGX3B compute integer-ALU loops and must not alter vertex/fragment loops or
 * mixed FP/int compute loops. */
TEST(FmWave2IlpSchedule, Agx3bPipeSplitAndIcPacingDoNotLeakOutsideComputeIntLoops)
{
   /* Case 1 & 2: G16G vertex and fragment loops with BFEIL, BFEIL, IADD must
    * keep BFEIL, BFEIL, IADD (legacy single integer pipe, no IC pacing). */
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G16G;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, agx_shader_info);
      b->shader->stage = stage;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = stage;

      agx_block *pre = agx_start_block(b->shader);
      agx_block *body = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      body->loop_header = true;
      agx_block_add_successor(pre, body);
      agx_block_add_successor(body, latch);
      agx_block_add_successor(latch, body);
      agx_block_add_successor(body, exit);

      b->cursor = agx_after_block(pre);
      agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index c = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index d = agx_preload(b, agx_register(4, AGX_SIZE_32));

      b->cursor = agx_after_block(body);
      agx_index b0 = agx_temp(b->shader, AGX_SIZE_32);
      agx_bfeil_to(b, b0, agx_zero(), a, agx_immediate(9), 0);
      agx_index b1 = agx_temp(b->shader, AGX_SIZE_32);
      agx_bfeil_to(b, b1, agx_zero(), c, agx_immediate(9), 0);
      agx_index ad = agx_iadd(b, d, agx_immediate(1), 0);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, b0);
      agx_unit_test(b, b1);
      agx_unit_test(b, ad);

      agx_ilp_schedule(b->shader);

      std::vector<enum agx_opcode> ops;
      agx_foreach_instr_in_block(body, I)
         ops.push_back(I->op);
      ASSERT_EQ(ops.size(), 3u) << "stage=" << stage;
      EXPECT_EQ(ops[0], AGX_OPCODE_BFEIL) << "stage=" << stage;
      EXPECT_EQ(ops[1], AGX_OPCODE_BFEIL) << "stage=" << stage;
      EXPECT_EQ(ops[2], AGX_OPCODE_IADD) << "stage=" << stage;
      ralloc_free(mem);
   }

   /* Case 3, 4, 5: Non-vacuous mixed FP/SFU + integer G16G compute loops
    * with >= 8 integer ops (so int_ops >= 8 is satisfied and ONLY the pure-int
    * filter keeps split_int_pipes disabled):
    * - Case 3: FADD + 7 BFEIL + 2 IADD must keep FADD, BFEIL, BFEIL (not
    *   promote IADD to ops[2] via IC pacing).
    * - Case 4 & 5: RCP / FCMPSEL + 7 BFEIL + 2 IADD must keep BFEIL, BFEIL
    *   at ops[0..1] (not promote IADD to ops[1] via IC pacing). */
   enum mixed_kind { MIXED_FADD, MIXED_RCP, MIXED_FCMPSEL };
   for (mixed_kind kind : {MIXED_FADD, MIXED_RCP, MIXED_FCMPSEL}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G16G;
      key.has_scratch = true;

      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, agx_shader_info);
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

      agx_block *pre = agx_start_block(b->shader);
      agx_block *body = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      body->loop_header = true;
      agx_block_add_successor(pre, body);
      agx_block_add_successor(body, latch);
      agx_block_add_successor(latch, body);
      agx_block_add_successor(body, exit);

      b->cursor = agx_after_block(pre);
      agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index c = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index d = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index f = agx_preload(b, agx_register(6, AGX_SIZE_32));

      b->cursor = agx_after_block(body);
      std::vector<agx_index> outs;
      if (kind == MIXED_FADD)
         outs.push_back(agx_fadd(b, f, f));

      for (unsigned i = 0; i < 7; ++i) {
         agx_index bi = agx_temp(b->shader, AGX_SIZE_32);
         agx_bfeil_to(b, bi, agx_zero(), (i & 1) ? c : a,
                      agx_immediate(5 + i), 0);
         outs.push_back(bi);
      }
      outs.push_back(agx_iadd(b, d, agx_immediate(1), 0));
      outs.push_back(agx_iadd(b, d, agx_immediate(2), 0));

      if (kind == MIXED_RCP) {
         outs.push_back(agx_rcp(b, f));
      } else if (kind == MIXED_FCMPSEL) {
         outs.push_back(agx_fcmpsel(b, f, agx_zero(), f, agx_zero(),
                                    AGX_FCOND_GT));
      }

      b->cursor = agx_after_block(exit);
      for (agx_index v : outs)
         agx_unit_test(b, v);

      agx_ilp_schedule(b->shader);

      std::vector<enum agx_opcode> ops;
      agx_foreach_instr_in_block(body, I)
         ops.push_back(I->op);
      ASSERT_EQ(ops.size(), 10u) << "kind=" << kind;
      if (kind == MIXED_FADD) {
         EXPECT_EQ(ops[0], AGX_OPCODE_FADD) << "kind=" << kind;
         EXPECT_EQ(ops[1], AGX_OPCODE_BFEIL) << "kind=" << kind;
         EXPECT_EQ(ops[2], AGX_OPCODE_BFEIL) << "kind=" << kind;
      } else {
         EXPECT_EQ(ops[0], AGX_OPCODE_BFEIL) << "kind=" << kind;
         EXPECT_EQ(ops[1], AGX_OPCODE_BFEIL) << "kind=" << kind;
      }
      ralloc_free(mem);
   }
}
