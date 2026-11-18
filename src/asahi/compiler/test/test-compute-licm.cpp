/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* Compute shaders hoist loop invariants on every generation, bounded by that
 * generation's register file. A K loop that reads shared memory at a uniform
 * base plus per-lane terms plus the counter, as llama.cpp's coopmat mul_mm
 * does, compiles on each encoding.
 */
TEST(ComputeLicm, CompilesOnEachTarget)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G14G, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16S, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };
   for (const auto &t : targets) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "licm");
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.shared_size = 8192;

      nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *n = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *base = agx_test_load_preamble(&b, 1, 32, 6);
      nir_def *lane = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *row = nir_imul_imm(&b, nir_ushr_imm(&b, lane, 3), 256);
      nir_def *col = nir_ishl_imm(&b, nir_iand_imm(&b, lane, 7), 2);

      nir_variable *k = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
      nir_variable *acc =
         nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, k, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);
      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *kv = nir_load_var(&b, k);
         nir_break_if(&b, nir_uge(&b, kv, n));
         nir_def *off = nir_iadd(
            &b, nir_iadd(&b, nir_iadd(&b, base, nir_ishl_imm(&b, kv, 4)), row),
            col);
         nir_def *v = nir_load_shared(&b, 1, 32, off, .align_mul = 4);
         nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), v), 1);
         nir_store_var(&b, k, nir_iadd_imm(&b, kv, 1), 1);
      }
      nir_pop_loop(&b, loop);
      nir_store_global(&b, nir_load_var(&b, acc),
                       nir_iadd(&b, addr, nir_u2u64(&b, nir_imul_imm(&b, lane, 4))),
                       .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << "chip " << t.chip;
      free(part.binary);
      ralloc_free(b.shader);
   }
}
