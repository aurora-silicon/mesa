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

/* The cap follows from the register file and the occupancy steps, not from
 * the chip it was tuned on. */
TEST(SampledComputeRegisterCap, DerivedFromFileAndOccupancy)
{
   /* G17C (AGX3C): exactly the 160 halves it was tuned at. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3C), 160u);
   EXPECT_EQ(agx_occupancy_for_register_count(160).max_threads, 640u);
   agx_device_key g17c = {};
   g17c.encoding = AGX_ISA_ENCODING_AGX3C;
   g17c.chip = AGX_CHIP_G17C;
   EXPECT_EQ(agx_register_file_halves(&g17c), 252u);
   EXPECT_EQ(agx_occupancy_for_register_count(252).max_threads, 384u);

   /* The other files reach fewer threads than 640 uncapped. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX2), 160u);
   /* AGX3B reaches 512 threads uncapped: too small a step for the spills. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3B), 0u);

   /* AGX3D also has a 252-half file; launches without optional_spills keep
    * the cap off. */
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
   struct {
      enum agx_isa_encoding encoding;
      enum agx_chip chip;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G},
   };

   for (const auto &t : targets) {
      SCOPED_TRACE(t.chip);
      unsigned max_reg, scratch_B;

      ASSERT_NO_FATAL_FAILURE(
         allocate_sampled_compute(t.encoding, t.chip, 0, &max_reg, &scratch_B));
      EXPECT_LT(max_reg, 160u);
      EXPECT_GT(scratch_B, 0u);

      ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(t.encoding, t.chip, 16,
                                                       &max_reg, &scratch_B));
      EXPECT_GE(max_reg, 160u);
      EXPECT_EQ(scratch_B, 0u);
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

/* Compute programs below four texture reads keep the whole file. */
TEST(SampledComputeRegisterCap, NoCapBelowTextureThreshold)
{
   struct {
      enum agx_isa_encoding encoding;
      enum agx_chip chip;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G},
   };

   for (const auto &t : targets) {
      SCOPED_TRACE(t.chip);
      unsigned max_reg, scratch_B;

      ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(
         t.encoding, t.chip, 0, &max_reg, &scratch_B, true, 0));
      EXPECT_GE(max_reg, 160u);
      EXPECT_EQ(scratch_B, 0u);
   }
}

/* When the 160-half cap pushes spill demand past the RA_MEM bound (4 KiB on
 * AGX2), compilation drops the cap and retries from the whole register file
 * instead of refusing a program that fits there. */
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

TEST(SampledComputeRegisterCap, RetryOnSpillExhaustionUsesWholeFile)
{
   /* 1115 live 32-bit values (~2230 halves) exceed AGX2_MAX_SPILL_SLOTS (2048
    * halves) under the 160-half cap, but fit cleanly in the 256-half file. */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = NULL;
      agx_shader_key key = {};
      build_sampled_compute_air(mem, &b, &key, AGX_ISA_ENCODING_AGX2,
                                AGX_CHIP_G14G, false, 0, false, true, 4, 1115);
      agx_ra(b->shader);
      EXPECT_TRUE(b->shader->ra_failed);
      EXPECT_TRUE(b->shader->ra_mem_exhausted);
      EXPECT_TRUE(b->shader->sampled_cap_applied);
      ralloc_free(mem);

      mem = ralloc_context(NULL);
      build_sampled_compute_air(mem, &b, &key, AGX_ISA_ENCODING_AGX2,
                                AGX_CHIP_G14G, false, 0, true, true, 4, 1115);
      agx_ra(b->shader);
      EXPECT_FALSE(b->shader->ra_failed);
      EXPECT_FALSE(b->shader->ra_mem_exhausted);
      EXPECT_FALSE(b->shader->sampled_cap_applied);
      EXPECT_GE(b->shader->max_reg, 160u);
      EXPECT_GT(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
   }

   /* Full agx_compile_shader_nir() retry path on G14G (AGX2) and G17C (AGX3C
    * with a 4 KiB spill bound). */
   struct {
      enum agx_isa_encoding encoding;
      enum agx_chip chip;
      uint32_t max_scratch_B;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G, 0},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, 4096},
   };

   for (const auto &t : targets) {
      SCOPED_TRACE(t.chip);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "sampled cap spill retry");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *gid =
         nir_u2u64(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0));
      nir_def *addr =
         nir_iadd(&b, nir_imm_int64(&b, 0x10000), nir_imul_imm(&b, gid, 8));
      nir_def *base = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *delta = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4),
                                       .align_mul = 4);

      const bool is_agx3c = t.encoding == AGX_ISA_ENCODING_AGX3C;
      nir_def *tex_sum = nir_imm_float(&b, 0.0f);
      for (unsigned i = 0; i < 4; ++i) {
         nir_def *texture =
            nir_bindless_image_agx(&b, nir_imm_int(&b, i * 64), .desc_set = 0);
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
         tex->op = nir_texop_txf;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->coord_components = 2;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(
            is_agx3c ? nir_tex_src_backend1 : nir_tex_src_coord,
            nir_imm_ivec2(&b, i, 0));
         tex->src[1] = nir_tex_src_for_ssa(
            nir_tex_src_lod,
            is_agx3c ? nir_imm_intN_t(&b, 0, 16) : nir_imm_int(&b, 0));
         tex->src[2] =
            nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         tex_sum = nir_fadd(&b, tex_sum, nir_channel(&b, &tex->def, 0));
      }

      constexpr unsigned N = 1115;
      std::vector<nir_def *> vals(N);
      nir_def *prev = base;
      for (unsigned i = 0; i < N; ++i) {
         prev = nir_fadd(&b, prev, delta);
         vals[i] = prev;
      }
      /* Consume in reverse order so all N values stay live simultaneously. */
      nir_def *sum = tex_sum;
      for (int i = N - 1; i >= 0; --i)
         sum = nir_fadd(&b, nir_fmul_imm(&b, sum, 1.5f), vals[i]);
      nir_store_global(&b, sum, addr, .align_mul = 4);

      if (is_agx3c) {
         ASSERT_TRUE(agx_nir_lower_g17p_indexed_samplers(b.shader, true));
         NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
                  lower_test_descriptor_set, nir_metadata_control_flow, NULL);
      } else {
         agx_nir_lower_texture_for_generation(b.shader, 8);
      }
      agx_preprocess_nir(b.shader);

      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.has_scratch = true;
      key.optional_spills = true;
      key.max_scratch_B = t.max_scratch_B;
      key.no_preamble = true;
      key.reserved_preamble = 16;

      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      EXPECT_GT(part.info.nr_gprs, 160u);
      EXPECT_GT(part.info.scratch_size, 0u);

      free(part.binary);
      ralloc_free(b.shader);
   }
}
