/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "agx_test.h"

#include <gtest/gtest.h>

/*
 * A constant program's ILP schedule may raise a block's register pressure up
 * to a cap. Values live across the block hold their registers through all of
 * it, so they count toward that cap too.
 *
 * The middle block of a three-block G17P constant program holds independent
 * ALU chains (immediate, add, observe). The ILP order issues every immediate
 * first, holding all of them at once; the source order holds one at a time.
 * The first block defines values the last block reads.
 */
class IlpCprogCap : public testing::Test {
 protected:
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};

   ~IlpCprogCap()
   {
      ralloc_free(mem);
   }

   /* Whether the ILP schedule reordered the middle block. */
   bool reorders(unsigned live_across, unsigned chains)
   {
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->is_preamble = true;

      agx_block *last = list_first_entry(&b->shader->blocks, agx_block, link);
      auto next_block = [&]() {
         agx_block *next = agx_test_block(b->shader);
         last->successors[0] = next;
         util_dynarray_append(&next->predecessors, last);
         last = next;
         b->cursor = agx_after_block(last);
      };

      std::vector<agx_index> across;
      for (unsigned i = 0; i < live_across; ++i)
         across.push_back(agx_mov_imm(b, 32, i));

      next_block();
      agx_block *middle = last;
      for (unsigned i = 0; i < chains; ++i) {
         agx_index v = agx_mov_imm(b, 32, 1000 + i);
         agx_unit_test(b, agx_iadd(b, v, v, 0));
      }

      next_block();
      for (agx_index v : across)
         agx_unit_test(b, v);

      std::vector<agx_instr *> original;
      agx_foreach_instr_in_block(middle, I)
         original.push_back(I);

      agx_ilp_schedule(b->shader);

      unsigned i = 0;
      bool changed = false;
      agx_foreach_instr_in_block(middle, I) {
         EXPECT_LT(i, original.size());
         changed |= i >= original.size() || I != original[i];
         ++i;
      }
      EXPECT_EQ(i, original.size());
      return changed;
   }
};

/* 16 halves live across plus 80 for the immediates: inside the cap. */
TEST_F(IlpCprogCap, SchedulesBlockThatFits)
{
   EXPECT_TRUE(reorders(8, 40));
}

/* The same block with 64 halves live across would need 144: it keeps its
 * order, although what the block itself adds still fits. */
TEST_F(IlpCprogCap, CountsValuesLiveAcrossTheBlock)
{
   EXPECT_FALSE(reorders(32, 40));
}

/* Straight-line AGX3B fragment blocks with 2+ independent SFU chains interleave
 * those chains within agx_ilp_agx3b_fragment_cap() (including blocks ending in
 * SAMPLE_MASK_AGX3), while AGX3C/AGX2, over-cap (>192h/256h) blocks, and blocks
 * with WAIT_PIX or ZS_EMIT_AGX3 keep their original pressure order. */
TEST_F(IlpCprogCap, Agx3bFragmentSfuInterleavesUnderCap)
{
   auto run_sfu = [&](enum agx_isa_encoding enc, enum agx_chip chip,
                      unsigned live_across, bool with_wait_pix,
                      bool with_zs_emit, bool with_sample_mask,
                      bool &both_sinc_before_fmul, bool &mask_still_last) {
      void *local = ralloc_context(NULL);
      agx_shader_key k = {};
      k.dev.encoding = enc;
      k.dev.chip = chip;
      agx_builder *b = agx_test_builder(local);
      b->shader->key = &k;
      b->shader->stage = MESA_SHADER_FRAGMENT;

      agx_block *blk0 = list_first_entry(&b->shader->blocks, agx_block, link);
      agx_index a = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index c = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_block *blk1 = blk0;
      std::vector<agx_index> across;
      if (live_across) {
         for (unsigned i = 0; i < live_across; ++i)
            across.push_back(agx_mov_imm(b, 32, 0x3f800000 + i));
         blk1 = agx_test_block(b->shader);
         blk0->successors[0] = blk1;
         util_dynarray_append(&blk1->predecessors, blk0);
         b->cursor = agx_after_block(blk1);
      }

      if (with_wait_pix)
         agx_wait_pix(b, 0x0001);
      agx_index k1 = agx_immediate_f(0.5f);
      agx_index k2 = agx_immediate_f(1.0f);

      /* Chain 1 */
      agx_index t1 = agx_ffma(b, a, k1, k2);
      agx_index t1b = agx_fadd(b, t1, k2);
      agx_instr *sinc1 = agx_sinc_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32), t1b);
      agx_instr *mul1 = agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32), sinc1->dest[0], t1b);

      /* Chain 2 */
      agx_index t2 = agx_ffma(b, c, k1, k2);
      agx_index t2b = agx_fadd(b, t2, k2);
      agx_instr *sinc2 = agx_sinc_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32), t2b);
      agx_instr *mul2 = agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32), sinc2->dest[0], t2b);

      agx_index sum = agx_fadd(b, mul1->dest[0], mul2->dest[0]);
      agx_instr *mask_instr = NULL;
      if (with_zs_emit) {
         agx_instr *col = agx_alloc_instr(b, AGX_OPCODE_COLLECT, 1, 2);
         col->dest[0] = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
         col->src[0] = sum;
         col->src[1] = sum;
         agx_builder_insert(&b->cursor, col);
         agx_zs_emit_agx3(b, col->dest[0], 1);
      }
      if (with_sample_mask) {
         agx_index m16 = agx_fcmpsel(b, sum, agx_immediate(0), agx_immediate(0),
                                     agx_immediate(1), AGX_FCOND_LT);
         mask_instr = agx_sample_mask_agx3(b, agx_immediate(1), m16, 2);
      } else {
         agx_unit_test(b, sum);
      }

      if (live_across) {
         agx_block *blk2 = agx_test_block(b->shader);
         blk1->successors[0] = blk2;
         util_dynarray_append(&blk2->predecessors, blk1);
         b->cursor = agx_after_block(blk2);
         for (agx_index v : across)
            agx_unit_test(b, v);
      }

      agx_validate(b->shader, "before agx3b fragment SFU schedule");

      std::vector<agx_instr *> before;
      agx_foreach_instr_in_block(blk1, I)
         before.push_back(I);

      agx_ilp_schedule(b->shader);
      agx_validate(b->shader, "after agx3b fragment SFU schedule");

      unsigned pos_sinc1 = 0, pos_sinc2 = 0, pos_mul1 = 0, pos_mul2 = 0;
      unsigned pos_mask = 0, idx = 0;
      bool changed = false;
      agx_foreach_instr_in_block(blk1, I) {
         changed |= (I != before[idx]);
         if (I == sinc1) pos_sinc1 = idx;
         if (I == sinc2) pos_sinc2 = idx;
         if (I == mul1) pos_mul1 = idx;
         if (I == mul2) pos_mul2 = idx;
         if (I == mask_instr) pos_mask = idx;
         ++idx;
      }
      both_sinc_before_fmul =
         pos_sinc1 < pos_mul1 && pos_sinc2 < pos_mul1 &&
         pos_sinc1 < pos_mul2 && pos_sinc2 < pos_mul2;
      mask_still_last = !mask_instr || pos_mask + 1 == idx;
      ralloc_free(local);
      return changed;
   };

   bool overlap = false, mask_last = false;
   EXPECT_TRUE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 0, false, false, false, overlap, mask_last));
   EXPECT_TRUE(overlap);

   /* Alpha-test discard (SAMPLE_MASK_AGX3) interleaves cleanly and keeps SAMPLE_MASK_AGX3 last. */
   EXPECT_TRUE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 0, false, false, true, overlap, mask_last));
   EXPECT_TRUE(overlap);
   EXPECT_TRUE(mask_last);

   /* AGX3C and AGX2 are untouched. */
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, 0, false, false, false, overlap, mask_last));
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, 0, false, false, false, overlap, mask_last));

   /* WAIT_PIX and ZS_EMIT_AGX3 blocks are untouched. */
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 0, true, false, false, overlap, mask_last));
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 0, false, true, false, overlap, mask_last));

   /* Live-across pressure exceeding 192h and >256h pre-RA leaves the block untouched. */
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 100, false, false, false, overlap, mask_last));
   EXPECT_FALSE(run_sfu(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, 140, false, false, false, overlap, mask_last));
}

static int
block_peak_halves(agx_context *ctx, agx_block *block)
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

/* Straight-line AGX3B fragment SFU scheduling must preserve the 48-half
 * register granule for pre-RA peaks <= 48 (specifically 45..48), keeping both
 * the scheduled pre-RA peak and post-RA allocation within 48 halves. */
TEST_F(IlpCprogCap, Agx3bFragmentSfuPreserves48HalfGranule)
{
   unsigned seen_45_48 = 0;
   for (unsigned live = 26; live <= 42; ++live) {
      void *local = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(local);
      agx_context *ctx = b->shader;
      agx_shader_key k = {};
      k.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      k.dev.chip = AGX_CHIP_G16G;
      ctx->key = &k;
      ctx->out = rzalloc(local, agx_shader_info);
      ctx->stage = MESA_SHADER_FRAGMENT;
      ctx->nir = rzalloc(local, nir_shader);
      ctx->nir->info.stage = MESA_SHADER_FRAGMENT;

      agx_block *entry = list_first_entry(&ctx->blocks, agx_block, link);
      agx_block *body = agx_test_block(ctx);
      agx_block *exit = agx_test_block(ctx);
      agx_block_add_successor(entry, body);
      agx_block_add_successor(body, exit);

      b->cursor = agx_after_block(entry);
      agx_index t0 = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index t1 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      std::vector<agx_index> held;
      for (unsigned i = 0; i < live; ++i)
         held.push_back(agx_mov_imm(b, 16, 0x3c00 + i));

      b->cursor = agx_after_block(body);
      agx_index acc = agx_mov_imm(b, 32, 0);
      for (unsigned i = 0; i < 8; ++i) {
         agx_index x = agx_fadd(b, t0, agx_uniform(2 * i, AGX_SIZE_32));
         agx_index lg = agx_log2(b, x);
         agx_index scaled = agx_fmul(b, lg, t1);
         agx_index ex = agx_exp2(b, scaled);
         acc = agx_fadd(b, acc, ex);
      }
      agx_unit_test(b, acc);

      b->cursor = agx_after_block(exit);
      agx_index sum = held[0];
      for (unsigned i = 1; i < live; ++i)
         sum = agx_fadd(b, sum, held[i]);
      agx_unit_test(b, sum);

      agx_validate(ctx, "before pressure scheduling");
      agx_pressure_schedule(ctx);
      agx_validate(ctx, "after pressure scheduling");
      const int before = block_peak_halves(ctx, body);

      agx_ilp_schedule(ctx);
      agx_validate(ctx, "after ILP scheduling");
      const int after = block_peak_halves(ctx, body);

      agx_ra(ctx);
      ASSERT_FALSE(ctx->ra_failed);

      if (before >= 45 && before <= 48)
         seen_45_48 |= 1u << (before - 45);

      if (before <= 48) {
         EXPECT_LE(after, 48) << "live=" << live << " before=" << before;
         EXPECT_LE(ctx->max_reg, 47u) << "live=" << live << " before=" << before
                                      << " after=" << after;
      }

      ralloc_free(local);
   }
   EXPECT_EQ(seen_45_48, 0xfu);
}

/* Straight-line AGX3B fragment SFU scheduling must enforce the <= peak + 16
 * growth bound even when the granule/occupancy cap leaves > 16 halves of room. */
TEST_F(IlpCprogCap, Agx3bFragmentSfuEnforces16HalfGrowthBound)
{
   for (unsigned live : {2u, 44u}) {
      void *local = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(local);
      agx_context *ctx = b->shader;
      agx_shader_key k = {};
      k.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      k.dev.chip = AGX_CHIP_G16G;
      ctx->key = &k;
      ctx->out = rzalloc(local, agx_shader_info);
      ctx->stage = MESA_SHADER_FRAGMENT;
      ctx->nir = rzalloc(local, nir_shader);
      ctx->nir->info.stage = MESA_SHADER_FRAGMENT;

      agx_block *entry = list_first_entry(&ctx->blocks, agx_block, link);
      agx_block *body = agx_test_block(ctx);
      agx_block *exit = agx_test_block(ctx);
      agx_block_add_successor(entry, body);
      agx_block_add_successor(body, exit);

      b->cursor = agx_after_block(entry);
      agx_index t0 = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index t1 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      std::vector<agx_index> held;
      for (unsigned i = 0; i < live; ++i)
         held.push_back(agx_mov_imm(b, 16, 0x3c00 + i));

      b->cursor = agx_after_block(body);
      for (unsigned i = 0; i < 14; ++i) {
         agx_index x = agx_fadd(b, t0, agx_uniform(2 * i, AGX_SIZE_32));
         agx_index lg = agx_log2(b, x);
         agx_index scaled = agx_fmul(b, lg, t1);
         agx_index ex = agx_exp2(b, scaled);
         agx_unit_test(b, ex);
      }

      b->cursor = agx_after_block(exit);
      agx_index sum = held[0];
      for (unsigned i = 1; i < live; ++i)
         sum = agx_fadd(b, sum, held[i]);
      agx_unit_test(b, sum);

      agx_validate(ctx, "before pressure scheduling");
      agx_pressure_schedule(ctx);
      agx_validate(ctx, "after pressure scheduling");
      const int before = block_peak_halves(ctx, body);

      agx_ilp_schedule(ctx);
      agx_validate(ctx, "after ILP scheduling");
      const int after = block_peak_halves(ctx, body);

      EXPECT_LE(after, before + 16) << "live=" << live << " before=" << before;
      ralloc_free(local);
   }
}

/* Straight-line AGX3B fragment SFU scheduling must also preserve the 96-half
 * register allocation granule (including with live vec3 alignment pressure)
 * by reserving the 8-half RA margin below 96 halves. */
TEST_F(IlpCprogCap, Agx3bFragmentSfuPreserves96HalfGranule)
{
   struct {
      unsigned count;
      unsigned live_lo;
      unsigned live_hi;
   } cases[] = {
      {2u, 63u, 66u},
      {8u, 53u, 58u},
      {14u, 51u, 56u},
   };
   unsigned checked = 0;
   for (const auto &c : cases) {
      for (unsigned live = c.live_lo; live <= c.live_hi; ++live) {
         void *local = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(local);
         agx_context *ctx = b->shader;
         agx_shader_key k = {};
         k.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         k.dev.chip = AGX_CHIP_G16G;
         ctx->key = &k;
         ctx->out = rzalloc(local, agx_shader_info);
         ctx->stage = MESA_SHADER_FRAGMENT;
         ctx->nir = rzalloc(local, nir_shader);
         ctx->nir->info.stage = MESA_SHADER_FRAGMENT;

         agx_block *entry = list_first_entry(&ctx->blocks, agx_block, link);
         agx_block *body = agx_test_block(ctx);
         agx_block *exit = agx_test_block(ctx);
         agx_block_add_successor(entry, body);
         agx_block_add_successor(body, exit);

         b->cursor = agx_after_block(entry);
         agx_index t0 = agx_preload(b, agx_register(0, AGX_SIZE_32));
         agx_index t1 = agx_preload(b, agx_register(2, AGX_SIZE_32));
         std::vector<agx_index> held;
         for (unsigned i = 0; i < 3; ++i) {
            agx_index vec = agx_vec_temp(ctx, AGX_SIZE_32, 3);
            agx_index srcs[3];
            for (unsigned j = 0; j < 3; ++j)
               srcs[j] = agx_mov_imm(b, 32, 0x3f800000 + 4 * i + j);
            agx_instr *col = agx_collect_to(b, vec, 3);
            for (unsigned j = 0; j < 3; ++j)
               col->src[j] = srcs[j];
            held.push_back(vec);
         }
         for (unsigned i = 0; i < live; ++i)
            held.push_back(agx_mov_imm(b, 16, 0x3c00 + i));

         b->cursor = agx_after_block(body);
         agx_index acc = agx_mov_imm(b, 32, 0);
         for (unsigned i = 0; i < c.count; ++i) {
            agx_index x = agx_fadd(b, t0, agx_uniform(2 * i, AGX_SIZE_32));
            agx_index lg = agx_log2(b, x);
            agx_index scaled = agx_fmul(b, lg, t1);
            agx_index ex = agx_exp2(b, scaled);
            acc = agx_fadd(b, acc, ex);
         }
         agx_unit_test(b, acc);

         b->cursor = agx_after_block(exit);
         for (agx_index v : held) {
            unsigned ch = agx_channels(v);
            if (ch == 1) {
               agx_unit_test(b, v);
            } else {
               agx_instr *sp = agx_split(b, ch, v);
               for (unsigned j = 0; j < ch; ++j)
                  sp->dest[j] = agx_temp(ctx, AGX_SIZE_32);
               for (unsigned j = 0; j < ch; ++j)
                  agx_unit_test(b, sp->dest[j]);
            }
         }

         agx_validate(ctx, "before pressure scheduling");
         agx_pressure_schedule(ctx);
         agx_validate(ctx, "after pressure scheduling");
         const int before = block_peak_halves(ctx, body);

         agx_ilp_schedule(ctx);
         agx_validate(ctx, "after ILP scheduling");
         const int after = block_peak_halves(ctx, body);

         agx_ra(ctx);
         ASSERT_FALSE(ctx->ra_failed);

         if (before <= 96) {
            checked++;
            EXPECT_LE(after, 96)
               << "count=" << c.count << " live=" << live << " before=" << before;
            EXPECT_LE(ctx->max_reg, 95u)
               << "count=" << c.count << " live=" << live << " before=" << before
               << " after=" << after;
         }

         ralloc_free(local);
      }
   }
   EXPECT_GE(checked, 15u);
}
