/* SPDX-License-Identifier: MIT */
#include <map>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "asahi/lib/agx_abi.h"
}

TEST(ImageAtomicMetadata, AddressLoadsFollowDescriptorABI)
{
   for (unsigned generation : {9u, 10u}) {
      for (bool metadata : {false, true}) {
         SCOPED_TRACE(generation);
         SCOPED_TRACE(metadata);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "image atomic metadata");
         nir_def *coord = nir_pad_vector(
            &b, nir_load_global_invocation_id(&b, 32), 4);
         nir_bindless_image_atomic(
            &b, 32, nir_imm_int(&b, 1), coord, nir_imm_int(&b, 0),
            nir_imm_int(&b, 1), .image_dim = GLSL_SAMPLER_DIM_2D,
            .image_array = true, .format = PIPE_FORMAT_R32_UINT,
            .atomic_op = nir_atomic_op_iadd);

         if (metadata)
            agx_nir_lower_texture_with_buffer_metadata(b.shader, generation,
                                                       false, 0, false);
         else
            agx_nir_lower_texture_for_generation(b.shader, generation);

         /* Give the descriptor a known base so we can check the actual memory
          * reads produced by the address helper, including its high word.
          */
         const uint64_t base = 0x100000;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
                  b.cursor = nir_before_instr(instr);
                  nir_def_rewrite_uses(&intr->def, nir_imm_int64(&b, base));
               }
            }
         }
         while (nir_opt_constant_folding(b.shader)) {}

         std::map<uint64_t, unsigned> loads;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                   nir_src_is_const(intr->src[0]))
                  loads[nir_src_as_uint(intr->src[0])] =
                     intr->def.bit_size * intr->def.num_components;
            }
         }
         const unsigned offset = generation == 10 && !metadata
            ? AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET : AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE;
         EXPECT_EQ(loads.count(base + offset), 1u);
         EXPECT_EQ(loads[base + offset], 64u);
         const unsigned other = offset == 24 ? 40 : 24;
         EXPECT_EQ(loads.count(base + other), 0u);
         nir_validate_shader(b.shader, "image atomic descriptor addressing");
         ralloc_free(b.shader);
      }
   }
}

static unsigned
count_global_access(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             (nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_load_global ||
              nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_store_global))
            n++;
      }
   }
   return n;
}

static unsigned
lowered_global_access(unsigned generation, enum gl_access_qualifier access)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "coherent image");
   nir_def *coord =
      nir_pad_vector(&b, nir_load_global_invocation_id(&b, 32), 4);
   nir_def *v = nir_bindless_image_load(
      &b, 4, 32, nir_imm_int(&b, 1), coord, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .access = access,
      .dest_type = nir_type_uint32);
   nir_bindless_image_store(&b, nir_imm_int(&b, 2), coord, nir_imm_int(&b, 0),
                            v, nir_imm_int(&b, 0),
                            .image_dim = GLSL_SAMPLER_DIM_2D,
                            .format = PIPE_FORMAT_R32_UINT, .access = access,
                            .src_type = nir_type_uint32);

   agx_nir_lower_texture_for_generation(b.shader, generation);
   nir_validate_shader(b.shader, "coherent image lowering");
   unsigned n = count_global_access(b.shader);
   ralloc_free(b.shader);
   return n;
}

/* G13/G14 (descriptor generation 7): coherent and atomic image loads and
 * stores of plain 32-bit formats address the image memory directly, since the
 * texture cache and the PBE are not coherent across GPU clusters there. Plain
 * accesses and later generations keep the image instructions. */
TEST(ImageAtomicMetadata, CoherentImageAccessUsesMemoryOnLegacyGenerations)
{
   for (unsigned generation : {7u, 9u}) {
      SCOPED_TRACE(generation);
      const unsigned plain =
         lowered_global_access(generation, (enum gl_access_qualifier)0);
      const unsigned extra = generation < 9 ? 2 : 0;
      EXPECT_EQ(lowered_global_access(generation, ACCESS_COHERENT),
                plain + extra);
      EXPECT_EQ(lowered_global_access(generation, ACCESS_ATOMIC),
                plain + extra);
   }
}
