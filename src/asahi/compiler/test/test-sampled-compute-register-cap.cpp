/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_test.h"

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

   /* G18G takes no scratch for compute, so it gets no cap. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3C, true), 0u);
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_AGX3D), 0u);

   /* No register file, no cap. */
   EXPECT_EQ(cap_for(AGX_ISA_ENCODING_UNKNOWN), 0u);
}

/* A G17C compute program with four texture reads and 200 halves live:
 * the first allocation spills to fit the cap; the retry after a failed
 * allocation (ra_budget_cut) works from the whole file and needs no spill. */
static void
allocate_sampled_compute(unsigned ra_budget_cut, unsigned *max_reg,
                         unsigned *scratch_B, bool optional_spills = true)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   key.optional_spills = optional_spills;
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, struct agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->nir->info.workgroup_size[0] = 32;
   b->shader->nir->info.workgroup_size[1] = 1;
   b->shader->nir->info.workgroup_size[2] = 1;
   b->shader->ra_budget_cut = ra_budget_cut;

   agx_index texels[4];
   for (unsigned i = 0; i < ARRAY_SIZE(texels); ++i) {
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

   agx_index values[84];
   /* Each value depends on the previous one, so none rematerializes. */
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = agx_fadd(b, i ? values[i - 1] : agx_uniform(160, AGX_SIZE_32),
                           agx_uniform(32 + 2 * (i % 64), AGX_SIZE_32));
   }
   for (agx_index value : values)
      agx_unit_test(b, value);
   for (agx_index texel : texels) {
      agx_instr *split = agx_split(b, 4, texel);
      agx_foreach_dest(split, d) {
         split->dest[d] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[d]);
      }
   }

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   *max_reg = b->shader->max_reg;
   *scratch_B = b->shader->scratch_size_B;
   ralloc_free(mem);
}

TEST(SampledComputeRegisterCap, RetryAfterFailedAllocationUsesWholeFile)
{
   unsigned max_reg, scratch_B;

   ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(0, &max_reg, &scratch_B));
   EXPECT_LT(max_reg, 160u);
   EXPECT_GT(scratch_B, 0u);

   ASSERT_NO_FATAL_FAILURE(allocate_sampled_compute(16, &max_reg, &scratch_B));
   EXPECT_GE(max_reg, 160u);
   EXPECT_EQ(scratch_B, 0u);
}

/* A launch that cannot take the cap's spills keeps the whole file. */
TEST(SampledComputeRegisterCap, NoCapWithoutOptionalSpills)
{
   unsigned max_reg, scratch_B;

   ASSERT_NO_FATAL_FAILURE(
      allocate_sampled_compute(0, &max_reg, &scratch_B, false));
   EXPECT_GE(max_reg, 160u);
   EXPECT_EQ(scratch_B, 0u);
}
