/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_debug.h"
#include "agx_nir.h"
}

static unsigned
cap_for(enum agx_isa_encoding encoding, bool emit_agx3d = false)
{
   agx_device_key dev = {};
   dev.encoding = encoding;
   dev.emit_agx3d = emit_agx3d;
   return agx_sampled_compute_register_cap(&dev);
}

/* The cap follows from static TPS support, the register file, and the
 * occupancy steps, not from the chip it was tuned on. */
TEST(SampledComputeRegisterCap, DerivedFromFileAndOccupancy)
{
   /* G17C (AGX3C): static TPS + 252-half file -> 160 halves (384 -> 640). */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3C), 160u);
   EXPECT_EQ(agx_occupancy_for_register_count(160).max_threads, 640u);
   agx_device_key g17c = {};
   g17c.encoding = AGX_ISA_ENCODING_AGX3C;
   g17c.chip = AGX_CHIP_G17C;
   EXPECT_EQ(agx_register_file_halves(&g17c), 252u);
   EXPECT_EQ(agx_occupancy_for_register_count(252).max_threads, 384u);

   /* AGX2 uses helper-backed device-memory stack spills with explicit 2-slot
    * scoreboard waits rather than static TPS: no optional spill cap. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX2), 0u);
   /* AGX3B reaches 512 threads uncapped: too small a step for the spills. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3B), 0u);

   /* AGX3D also has static TPS and a 252-half file; launches without
    * optional_spills keep the cap off. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3C, true), 160u);
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3D), 160u);

   /* No register file, no cap. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_UNKNOWN), 0u);
}

/* A compute program with configurable live halves and texture reads:
 * with >= 4 texture reads, the first allocation spills to fit the cap; the
 * retry after a failed allocation (ra_budget_cut / disable_sampled_cap) or a
 * launch without optional_spills works from the whole file. */
static void
build_sampled_compute_air(void *mem, agx_builder **b_out, agx_shader_key *key,
                          enum agx_isa_encoding encoding, enum agx_chip chip,
                          bool emit_agx3d, unsigned ra_budget_cut,
                          bool disable_sampled_cap, bool optional_spills,
                          unsigned tex_loads, unsigned live_vals)
{
   agx_builder *b = agx_test_builder(mem);
   *key = {};
   key->dev.encoding = encoding;
   key->dev.chip = chip;
   key->dev.emit_agx3d = emit_agx3d;
   key->has_scratch = true;
   key->optional_spills = optional_spills;
   b->shader->key = key;
   b->shader->out = rzalloc(mem, struct agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->nir->info.workgroup_size[0] = 32;
   b->shader->nir->info.workgroup_size[1] = 1;
   b->shader->nir->info.workgroup_size[2] = 1;
   b->shader->ra_budget_cut = ra_budget_cut;
   b->shader->disable_sampled_cap = disable_sampled_cap;

   agx_index texels[4] = {};
   for (unsigned i = 0; i < tex_loads; ++i) {
      texels[i] = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_index x = agx_mov(b, agx_uniform(8 + 4 * i, AGX_SIZE_32));
      agx_index y = agx_mov(b, agx_uniform(10 + 4 * i, AGX_SIZE_32));
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coords, 2);
      collect->src[0] = x;
      collect->src[1] = y;
      agx_texture_load_to(b, texels[i], coords, agx_immediate(0),
                          agx_immediate(0), agx_immediate(0), agx_immediate(0),
                          agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0xf,
                          false, false);
   }

   agx_index prev = agx_uniform(160, AGX_SIZE_32);
   agx_index *values = ralloc_array(mem, agx_index, live_vals);
   /* Each value depends on the previous one, so none rematerializes. */
   for (unsigned i = 0; i < live_vals; ++i) {
      values[i] = agx_fadd(b, prev,
                           agx_uniform(32 + 2 * (i % 64), AGX_SIZE_32));
      prev = values[i];
   }
   for (unsigned i = 0; i < live_vals; ++i)
      agx_unit_test(b, values[i]);
   for (unsigned i = 0; i < tex_loads; ++i) {
      agx_instr *split = agx_split(b, 4, texels[i]);
      agx_foreach_dest(split, d) {
         split->dest[d] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[d]);
      }
   }

   *b_out = b;
}

static void
allocate_sampled_compute(enum agx_isa_encoding encoding, enum agx_chip chip,
                         unsigned ra_budget_cut, unsigned *max_reg,
                         unsigned *scratch_B, bool optional_spills = true,
                         unsigned tex_loads = 4, bool emit_agx3d = false)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = NULL;
   agx_shader_key key = {};
   build_sampled_compute_air(mem, &b, &key, encoding, chip, emit_agx3d,
                             ra_budget_cut, false, optional_spills, tex_loads,
                             84);

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   *max_reg = b->shader->max_reg;
   *scratch_B = b->shader->scratch_size_B;
   ralloc_free(mem);
}

TEST(SampledComputeRegisterCap, RetryAfterFailedAllocationUsesWholeFile)
{
   unsigned max_reg, scratch_B;

   ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(
      AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, 0, &max_reg, &scratch_B));
   EXPECT_LT(max_reg, 160u);
   EXPECT_GT(scratch_B, 0u);

   ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(
      AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, 16, &max_reg, &scratch_B));
   EXPECT_GE(max_reg, 160u);
   EXPECT_EQ(scratch_B, 0u);
}

/* Encodings without launch-allocated static TPS (AGX2) stay uncapped even when
 * optional_spills is true and >= 4 texture loads are present. */
TEST(SampledComputeRegisterCap, NoCapOnAGX2WithoutStaticTPS)
{
   const enum agx_chip chips[] = {AGX_CHIP_G13G, AGX_CHIP_G14G};
   for (enum agx_chip chip : chips) {
      SCOPED_TRACE(chip);
      void *mem = ralloc_context(NULL);
      agx_builder *b = NULL;
      agx_shader_key key = {};
      build_sampled_compute_air(mem, &b, &key, AGX_ISA_ENCODING_AGX2, chip,
                                false, 0, false, true, 4, 84);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_FALSE(b->shader->sampled_cap_applied);
      EXPECT_GE(b->shader->max_reg, 160u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
   }
}

/* A launch that cannot take the cap's spills keeps the whole file. */
TEST(SampledComputeRegisterCap, NoCapWithoutOptionalSpills)
{
   struct {
      enum agx_isa_encoding encoding;
      enum agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };

   for (const auto &t : targets) {
      SCOPED_TRACE(t.chip);
      unsigned max_reg, scratch_B;

      ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(
         t.encoding, t.chip, 0, &max_reg, &scratch_B, false, 4, t.emit_agx3d));
      EXPECT_GE(max_reg, 160u);
      EXPECT_EQ(scratch_B, 0u);
   }
}

/* Compute programs below four texture reads (0 and 3) keep the whole file. */
TEST(SampledComputeRegisterCap, NoCapBelowTextureThreshold)
{
   for (unsigned tex_loads : {0u, 3u}) {
      SCOPED_TRACE(tex_loads);
      unsigned max_reg, scratch_B;

      ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(
         AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, 0, &max_reg, &scratch_B, true,
         tex_loads));
      EXPECT_GE(max_reg, 160u);
      EXPECT_EQ(scratch_B, 0u);
   }
}

static bool
lower_test_descriptor_set(nir_builder *b, nir_intrinsic_instr *intr, void *)
{
   if (intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(
      &intr->def,
      nir_load_preamble(b, 1, 64, .base = (int)(4 * nir_intrinsic_desc_set(intr))));
   return true;
}

static nir_shader *
build_agx3c_sampled_compute_nir(unsigned n_tex, unsigned n_live)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "sampled cap test");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid =
      nir_u2u64(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0));
   nir_def *addr =
      nir_iadd(&b, nir_imm_int64(&b, 0x10000), nir_imul_imm(&b, gid, 8));
   nir_def *base = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
   nir_def *delta =
      nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4), .align_mul = 4);

   nir_def *tex_sum = nir_imm_float(&b, 0.0f);
   for (unsigned i = 0; i < n_tex; ++i) {
      nir_def *texture =
         nir_bindless_image_agx(&b, nir_imm_int(&b, (i % 4) * 64), .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_txf;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->coord_components = 2;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                        nir_imm_ivec2(&b, i, 0));
      tex->src[1] =
         nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_intN_t(&b, 0, 16));
      tex->src[2] =
         nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      tex_sum = nir_fadd(&b, tex_sum, nir_channel(&b, &tex->def, 0));
   }

   std::vector<nir_def *> vals(n_live);
   nir_def *prev = base;
   for (unsigned i = 0; i < n_live; ++i) {
      prev = nir_fadd(&b, prev, delta);
      vals[i] = prev;
   }
   /* Consume in reverse order so all n_live values stay live simultaneously. */
   nir_def *sum = tex_sum;
   for (int i = (int)n_live - 1; i >= 0; --i)
      sum = nir_fadd(&b, nir_fmul_imm(&b, sum, 1.5f), vals[i]);
   nir_store_global(&b, sum, addr, .align_mul = 4);

   if (!agx_nir_lower_g17p_indexed_samplers(b.shader, true))
      return NULL;
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_test_descriptor_set,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* When the 160-half cap pushes spill demand past the RA_MEM bound (4 KiB),
 * compilation drops the cap and retries from the whole register file instead
 * of refusing a program that fits there. */
TEST(SampledComputeRegisterCap, RetryOnSpillExhaustionUsesWholeFile)
{
   /* Direct agx_ra() check on AGX3C with a 4 KiB spill bound: 1115 live 32-bit
    * values (~2230 halves) exceed the 2048-half spill bound under the 160-half
    * cap, but fit cleanly in the 252-half file. */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = NULL;
      agx_shader_key key = {};
      build_sampled_compute_air(mem, &b, &key, AGX_ISA_ENCODING_AGX3C,
                                AGX_CHIP_G17C, false, 0, false, true, 4, 1115);
      key.max_scratch_B = 4096;
      agx_ra(b->shader);
      EXPECT_TRUE(b->shader->ra_failed);
      EXPECT_TRUE(b->shader->ra_mem_exhausted);
      EXPECT_TRUE(b->shader->sampled_cap_applied);
      ralloc_free(mem);

      mem = ralloc_context(NULL);
      build_sampled_compute_air(mem, &b, &key, AGX_ISA_ENCODING_AGX3C,
                                AGX_CHIP_G17C, false, 0, true, true, 4, 1115);
      key.max_scratch_B = 4096;
      agx_ra(b->shader);
      EXPECT_FALSE(b->shader->ra_failed);
      EXPECT_FALSE(b->shader->ra_mem_exhausted);
      EXPECT_FALSE(b->shader->sampled_cap_applied);
      EXPECT_GE(b->shader->max_reg, 160u);
      EXPECT_GT(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
   }

   /* Full agx_compile_shader_nir() retry path on G17C with a 4 KiB bound. */
   nir_shader *nir = build_agx3c_sampled_compute_nir(4, 1115);
   ASSERT_NE(nir, nullptr);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   key.optional_spills = true;
   key.max_scratch_B = 4096;
   key.no_preamble = true;
   key.reserved_preamble = 16;

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(nir, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_GT(part.info.nr_gprs, 160u);
   EXPECT_GT(part.info.scratch_size, 0u);

   free(part.binary);
   ralloc_free(nir);
}

/* ALU-heavy sampled compute shaders where capping to 160 halves forces
 * excessive optional spills drop the cap and compile from the whole file. */
TEST(SampledComputeRegisterCap, ExcessiveSpillsRetryUsesWholeFile)
{
   /* 4 texture loads and 100 live 32-bit values (~200 halves): fits uncapped
    * in G17C's 252-half file with zero scratch, whereas capping to 160 halves
    * exceeds both the spill-to-sample ratio and spill budget. */
   nir_shader *nir = build_agx3c_sampled_compute_nir(4, 100);
   ASSERT_NE(nir, nullptr);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   key.optional_spills = true;
   key.max_scratch_B = 4096;
   key.no_preamble = true;
   key.reserved_preamble = 16;

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(nir, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_GT(part.info.nr_gprs, 160u);
   EXPECT_EQ(part.info.scratch_size, 0u);

   free(part.binary);
   ralloc_free(nir);
}

/* Texture-heavy sampled compute shaders with moderate optional spills keep
 * the 160-half occupancy cap. */
TEST(SampledComputeRegisterCap, ModerateSpillsKeepSampledCap)
{
   /* 16 texture loads and 86 live 32-bit values (~172 halves): spills ~16
    * values (~64 B) under the 160-half cap, well within both the spill budget
    * and spill-to-sample ratio. */
   nir_shader *nir = build_agx3c_sampled_compute_nir(16, 86);
   ASSERT_NE(nir, nullptr);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   key.optional_spills = true;
   key.max_scratch_B = 4096;
   key.no_preamble = true;
   key.reserved_preamble = 16;

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(nir, &key, &part))
      << agx_last_compile_refusal();
   EXPECT_LE(part.info.nr_gprs, 160u);
   EXPECT_GT(part.info.scratch_size, 0u);

   free(part.binary);
   ralloc_free(nir);
}

/* Loop-weighted spill traffic outpacing outside texture ops rejects the cap,
 * whereas textures inside the same loop justify the same spills. Also verify:
 *  - multi-backedge loops count as a single nesting level,
 *  - nested loops compound nesting weight,
 *  - NIR scratch accesses below spill_base_B are ignored, and
 *  - multi-component (e.g. 64-bit / mask=0x3) spills count per scalar access
 *    against the 48-scalar ceiling. */
TEST(SampledComputeRegisterCap, LoopSpillsWeightedAgainstLoopTextures)
{
   for (bool tex_in_loop : {false, true}) {
      SCOPED_TRACE(tex_in_loop);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_context *ctx = b->shader;
      ctx->spill_base_B = 64;
      ctx->scratch_size_B = 80; /* 16 B RA spill frame above 64 B NIR scratch */

      agx_block *entry = agx_start_block(ctx);
      entry->index = 0;

      agx_block *loop_hdr = rzalloc(ctx, agx_block);
      loop_hdr->index = 1;
      loop_hdr->loop_header = true;
      list_inithead(&loop_hdr->instructions);
      list_addtail(&loop_hdr->link, &ctx->blocks);

      /* Two distinct backedges (continue + latch) to the same loop header. */
      agx_block *loop_latch = rzalloc(ctx, agx_block);
      loop_latch->index = 2;
      list_inithead(&loop_latch->instructions);
      list_addtail(&loop_latch->link, &ctx->blocks);

      agx_block_add_successor(entry, loop_hdr);
      agx_block_add_successor(loop_hdr, loop_hdr);
      agx_block_add_successor(loop_hdr, loop_latch);
      agx_block_add_successor(loop_latch, loop_hdr);

      /* NIR scratch accesses below spill_base_B (64 B) must not count. */
      b->cursor = agx_after_block(loop_latch);
      for (unsigned i = 0; i < 64; ++i) {
         agx_stack_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                         AGX_FORMAT_I32, 0xf);
         agx_stack_load_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                           AGX_FORMAT_I32, 0xf);
      }

      /* Emit 4 texture loads either outside or inside the loop. */
      b->cursor = agx_after_block(tex_in_loop ? loop_hdr : entry);
      for (unsigned i = 0; i < 4; ++i) {
         agx_texture_load_to(b, agx_register(0, AGX_SIZE_32),
                             agx_register(4, AGX_SIZE_32), agx_immediate(0),
                             agx_immediate(0), agx_immediate(0),
                             agx_immediate(0), agx_null(), AGX_DIM_2D,
                             AGX_LOD_MODE_LOD_MIN, 0x1, false, false);
      }

      /* Emit 2 scalar spills + 2 scalar fills in loop_hdr (single nesting
       * level despite two backedges -> weight 8 -> 32 weighted spill ops).
       * When textures are outside: 32 <= 4*8=32 (just fits!). Adding one more
       * 64-bit spill (mask=0x3 -> 2 scalar stores -> +16 weighted = 48 > 32)
       * must trip the ratio when textures are outside, while still passing when
       * textures are inside (48 <= 32*8=256). */
      b->cursor = agx_after_block(loop_hdr);
      for (unsigned i = 0; i < 2; ++i) {
         agx_stack_store(b, agx_register(0, AGX_SIZE_32),
                         agx_immediate(64 + i * 4), AGX_FORMAT_I32, 0x1);
         agx_stack_load_to(b, agx_register(0, AGX_SIZE_32),
                           agx_immediate(64 + i * 4), AGX_FORMAT_I32, 0x1);
      }
      /* Exactly at 32 == 4*8: passes regardless of tex_in_loop (verifies
       * multi-backedge did not double-count loop_hdr to weight 64). */
      EXPECT_FALSE(agx_sampled_cap_spills_excessive(ctx));

      agx_stack_store(b, agx_register(0, AGX_SIZE_64), agx_immediate(72),
                      AGX_FORMAT_I32, 0x3);
      EXPECT_EQ(agx_sampled_cap_spills_excessive(ctx), !tex_in_loop);
      ralloc_free(mem);
   }

   /* Nested loop (depth 2 -> weight 64) vs outer texture (depth 1 -> weight 8)
    * and 64-bit spill scalar boundary (48 ok, 50 rejected). */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_context *ctx = b->shader;
      ctx->spill_base_B = 32;
      ctx->scratch_size_B = 128; /* 96 B RA spill frame (<= 192 B) */

      agx_block *outer = agx_start_block(ctx);
      outer->index = 0;
      outer->loop_header = true;

      agx_block *inner = rzalloc(ctx, agx_block);
      inner->index = 1;
      inner->loop_header = true;
      list_inithead(&inner->instructions);
      list_addtail(&inner->link, &ctx->blocks);

      agx_block_add_successor(outer, inner);
      agx_block_add_successor(inner, inner);
      agx_block_add_successor(inner, outer);

      /* 2 outer textures (weight 8 -> 16) allows up to 128 weighted spill ops.
       * Two scalar stores in inner loop (weight 64 -> 128) fit; adding a third
       * (192 > 128) rejects. */
      b->cursor = agx_after_block(outer);
      for (unsigned i = 0; i < 2; ++i) {
         agx_texture_load_to(b, agx_register(0, AGX_SIZE_32),
                             agx_register(4, AGX_SIZE_32), agx_immediate(0),
                             agx_immediate(0), agx_immediate(0),
                             agx_immediate(0), agx_null(), AGX_DIM_2D,
                             AGX_LOD_MODE_LOD_MIN, 0x1, false, false);
      }
      b->cursor = agx_after_block(inner);
      agx_stack_store(b, agx_register(0, AGX_SIZE_64), agx_immediate(32),
                      AGX_FORMAT_I32, 0x3); /* 2 scalars * 64 = 128 */
      EXPECT_FALSE(agx_sampled_cap_spills_excessive(ctx));

      agx_stack_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(40),
                      AGX_FORMAT_I32, 0x1); /* +64 = 192 > 128 */
      EXPECT_TRUE(agx_sampled_cap_spills_excessive(ctx));
      ralloc_free(mem);
   }

   /* Scalar spill-count ceiling boundary with 64-bit spills (mask=0x3):
    * 24 * 2 = 48 scalar stores passes; 25 * 2 = 50 rejects even with ample
    * texture loads and small scratch footprint. */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_context *ctx = b->shader;
      ctx->spill_base_B = 16;
      ctx->scratch_size_B = 80; /* 64 B RA spill frame (<= 192 B) */

      agx_block *blk = agx_start_block(ctx);
      blk->index = 0;
      b->cursor = agx_after_block(blk);
      for (unsigned i = 0; i < 32; ++i) {
         agx_texture_load_to(b, agx_register(0, AGX_SIZE_32),
                             agx_register(4, AGX_SIZE_32), agx_immediate(0),
                             agx_immediate(0), agx_immediate(0),
                             agx_immediate(0), agx_null(), AGX_DIM_2D,
                             AGX_LOD_MODE_LOD_MIN, 0x1, false, false);
      }
      for (unsigned i = 0; i < 24; ++i) {
         agx_stack_store(b, agx_register(0, AGX_SIZE_64), agx_immediate(16),
                         AGX_FORMAT_I32, 0x3);
      }
      EXPECT_FALSE(agx_sampled_cap_spills_excessive(ctx));

      agx_stack_store(b, agx_register(0, AGX_SIZE_64), agx_immediate(16),
                      AGX_FORMAT_I32, 0x3); /* 50 > 48 */
      EXPECT_TRUE(agx_sampled_cap_spills_excessive(ctx));
      ralloc_free(mem);
   }
}
