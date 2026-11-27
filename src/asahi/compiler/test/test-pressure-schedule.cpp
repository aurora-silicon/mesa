/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>

/* Regression test for wavefront-synchronized bottom-up pressure scheduling:
 * When an unrolled tap loop feeds a shallow scalar reduction chain (e.g. AO
 * sum with 1 post-op) alongside deeper vector reduction chains (e.g. bent
 * normal X/Y/Z with 10 post-ops), pure greedy min-delta unwinds the entire
 * shallow chain before reaching the deeper chains, keeping all N tap weights
 * and direction vectors live simultaneously. Reverse-wavefront fallback keeps
 * parallel chains synchronized and cuts peak register demand dramatically. */
TEST(PressureSchedule, WavefrontSynchronizesParallelReductionChains)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = agx_chip_isa_encoding(AGX_CHIP_G17S);

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *body = agx_start_block(b->shader);
   b->cursor = agx_after_block(body);

   agx_index base = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index scale = agx_preload(b, agx_register(2, AGX_SIZE_32));

   const unsigned taps = 24;
   agx_index w[taps], dx[taps], dy[taps], dz[taps];
   agx_index ao_sum = agx_fadd(b, base, scale);
   for (unsigned i = 0; i < taps; ++i) {
      agx_index t = agx_fadd(b, base, agx_immediate(i + 1));
      agx_index s = agx_fmul(b, t, scale);
      w[i] = agx_ffma(b, s, base, agx_immediate(i + 2));
      dx[i] = agx_ffma(b, s, scale, agx_immediate(i + 3));
      dy[i] = agx_ffma(b, t, scale, agx_immediate(i + 4));
      dz[i] = agx_ffma(b, t, base, agx_immediate(i + 5));
      ao_sum = agx_fadd(b, ao_sum, w[i]);
   }

   /* Separate X, Y, Z reduction chains emitted after all taps (matching
    * scalarized NIR). */
   agx_index sum_x = agx_fmul(b, w[0], dx[0]);
   for (unsigned i = 1; i < taps; ++i)
      sum_x = agx_ffma(b, w[i], dx[i], sum_x);

   agx_index sum_y = agx_fmul(b, w[0], dy[0]);
   for (unsigned i = 1; i < taps; ++i)
      sum_y = agx_ffma(b, w[i], dy[i], sum_y);

   agx_index sum_z = agx_fmul(b, w[0], dz[0]);
   for (unsigned i = 1; i < taps; ++i)
      sum_z = agx_ffma(b, w[i], dz[i], sum_z);

   /* Asymmetric post-processing: ao_sum has 1 op, while (sum_x, sum_y, sum_z)
    * has a deeper normalization chain. */
   agx_index ao_out = agx_fadd(b, ao_sum, agx_immediate(1));
   agx_index len2 = agx_ffma(b, sum_x, sum_x,
                             agx_ffma(b, sum_y, sum_y, agx_fmul(b, sum_z, sum_z)));
   agx_index inv = len2;
   for (unsigned k = 0; k < 8; ++k)
      inv = agx_ffma(b, inv, scale, base);
   agx_index nx = agx_fmul(b, sum_x, inv);
   agx_index ny = agx_fmul(b, sum_y, inv);
   agx_index nz = agx_fmul(b, sum_z, inv);
   agx_index combined = agx_fadd(b, agx_fadd(b, ao_out, nx),
                                 agx_fadd(b, ny, nz));
   agx_unit_test(b, combined);
   agx_stop(b);

   const unsigned before = agx_calc_register_demand(b->shader, false);
   agx_pressure_schedule(b->shader);
   const unsigned after = agx_calc_register_demand(b->shader, false);

   /* Without wavefront synchronization, greedy min-delta unwinds all 24 taps
    * (>140 halves). With wavefront fallback, demand drops below 96 halves. */
   EXPECT_LT(after, before);
   EXPECT_LT(after, 96u);

   ralloc_free(mem);
}
