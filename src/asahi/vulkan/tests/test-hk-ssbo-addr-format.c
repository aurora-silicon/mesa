/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "vk_pipeline.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
 fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* An SSBO pointer is built by spirv_to_nir (hk_get_spirv_options()), has its
 * resource index rewritten by hk_nir_lower_descriptors() and is consumed by
 * nir_lower_explicit_io() (hk_lower_nir()). The three must use one address
 * format, or the explicit-I/O lowering sees a deref whose address has the
 * wrong number of components. Build what spirv_to_nir builds for a storage
 * buffer load and store, at a constant and at a dynamic array index, and run
 * both lowerings with the formats the driver passes them, for every chip with
 * a known descriptor generation. */
static void
check_chip(enum agx_chip chip, const struct vk_pipeline_robustness_state *rs)
{
   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(chip);

   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   CHECK(pdev);
   pdev->dev.chip = chip;
   const struct spirv_to_nir_options spirv =
      hk_get_spirv_options(&pdev->vk, MESA_SHADER_COMPUTE, rs);
   const nir_address_format format = hk_ssbo_addr_format(generation, rs);
   CHECK(spirv.ssbo_addr_format == format);
   free(pdev);

   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = generation;
   layout->binding_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .array_size = 4, .stride = sizeof(struct hk_buffer_address),
      .offset = 0,
   };
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "ssbo address format");
   const unsigned comps = nir_address_format_num_components(format);
   const unsigned bits = nir_address_format_bit_size(format);

   nir_def *indices[] = {nir_imm_int(&b, 1),
                         nir_load_local_invocation_index(&b)};
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < ARRAY_SIZE(indices); ++i) {
      nir_def *res = nir_vulkan_resource_index(&b, comps, bits, indices[i],
         .desc_set = 0, .binding = 0,
         .desc_type = nir_descriptor_type_storage_buffer);
      nir_def *desc = nir_load_vulkan_descriptor(&b, comps, bits, res,
         .desc_type = nir_descriptor_type_storage_buffer);
      nir_deref_instr *buf = nir_build_deref_cast(&b, desc, nir_var_mem_ssbo,
         glsl_array_type(glsl_uint_type(), 0, 4), 0);
      nir_deref_instr *elem = nir_build_deref_array(&b, buf,
         nir_load_local_invocation_index(&b));
      sum = nir_iadd(&b, sum, nir_load_deref(&b, elem));
      nir_store_deref(&b, elem, sum, 1);
   }

   CHECK(hk_nir_lower_descriptors(b.shader, rs, 1, layouts, false, false, 0,
      false, false, false, false, false));
   nir_validate_shader(b.shader, "after hk_nir_lower_descriptors");
   CHECK(nir_lower_explicit_io(b.shader, nir_var_mem_ssbo, format));
   nir_validate_shader(b.shader, "after nir_lower_explicit_io");

   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref)
               CHECK(!nir_deref_mode_is(nir_instr_as_deref(instr),
                                        nir_var_mem_ssbo));
         }
      }
   }

   ralloc_free(b.shader);
   free(layout);
}

int
main(void)
{
   const struct vk_pipeline_robustness_state robust = {
      .storage_buffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
      .uniform_buffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
      .vertex_inputs =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT,
      .null_storage_buffer_descriptor = true,
   };

   glsl_type_singleton_init_or_ref();
   unsigned map_chips = 0;
   for (unsigned c = 0; c < AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = c;
      const enum agx_descriptor_generation generation =
         agx_chip_descriptor_generation(chip);
      if (generation == AGX_DESCRIPTOR_GENERATION_UNKNOWN)
         continue;
      map_chips += hk_storage_address_has_map(generation);
      check_chip(chip, &vk_robustness_disabled);
      check_chip(chip, &robust);
   }
   /* The M5 parts and the M6 (G17G, G17S, G17C, G18G) carry the map word. */
   CHECK(map_chips == 4);
   CHECK(hk_storage_address_has_map(
      agx_chip_descriptor_generation(AGX_CHIP_G18G)));
   glsl_type_singleton_decref();
   return 0;
}
