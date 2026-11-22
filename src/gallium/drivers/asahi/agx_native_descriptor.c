/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_native_descriptor.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_pbe_agx3_core.h"

#include <string.h>

bool
agx_gallium_classify_standard_border(const uint32_t color[4], bool integer,
                                     enum agx_border_colour *out)
{
   if (!color || !out)
      return false;

   bool rgb_zero = true, rgba_one = true;
   for (unsigned i = 0; i < 4; ++i) {
      union {
         uint32_t u;
         float f;
      } value = {.u = color[i]};
      const bool zero = integer ? color[i] == 0 : value.f == 0.0f;
      const bool one = integer ? color[i] == 1 : value.f == 1.0f;
      rgb_zero &= i == 3 || zero;
      rgba_one &= one;
   }

   union {
      uint32_t u;
      float f;
   } alpha = {.u = color[3]};
   const bool alpha_zero = integer ? color[3] == 0 : alpha.f == 0.0f;
   const bool alpha_one = integer ? color[3] == 1 : alpha.f == 1.0f;

   if (rgb_zero && alpha_zero) {
      *out = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   } else if (rgb_zero && alpha_one) {
      *out = AGX_BORDER_COLOUR_OPAQUE_BLACK;
   } else if (rgba_one) {
      *out = AGX_BORDER_COLOUR_OPAQUE_WHITE;
   } else {
      return false;
   }

   return true;
}

bool
agx_gallium_g17p_sampler_handle(enum agx_chip chip, unsigned api_sampler,
                                uint16_t *native_handle)
{
   if (!agx_chip_agx3_path(chip) ||
       api_sampler >= AGX_GALLIUM_G17P_API_SAMPLERS || !native_handle)
      return false;

   *native_handle = api_sampler + 1;
   return true;
}

bool
agx_gallium_g17p_custom_border_handle(enum agx_chip chip, unsigned api_sampler,
                                      uint16_t *native_handle)
{
   if (!agx_chip_agx3_path(chip) ||
       api_sampler >= AGX_GALLIUM_G17P_API_SAMPLERS || !native_handle)
      return false;

   *native_handle = AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + api_sampler;
   return true;
}

/* Keep Gallium's narrower pre-refactor native PBE admission gate intact so
 * consolidating texture/sampler/G18G packing into asahi/lib introduces no
 * Gallium PBE behavior changes (widening Gallium's gate to match Borealis is
 * left to its own hardware-verified follow-up). */
bool
agx_gallium_pack_native_pbe(enum agx_descriptor_generation generation,
                            struct agx_gallium_pbe_descriptor *out,
                            const struct AGX_PBE *legacy)
{
   const unsigned descriptor_size = agx_gallium_pbe_descriptor_size(generation);

   if (!out || !legacy || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);
   if (!agx_gallium_native_pbe_supported(generation))
      return false;

   if (agx_gallium_descriptor_generation_is_agx2(generation)) {
      AGX_PBE_pack(out->opaque, legacy);
      return true;
   }

   const uint32_t agx3_generation =
      generation == AGX_DESCRIPTOR_GENERATION_9  ? AGX_PBE_AGX3_GENERATION_9
      : generation == AGX_DESCRIPTOR_GENERATION_10 ? AGX_PBE_AGX3_GENERATION_10
                                                   : 0;
   if (!agx3_generation)
      return false;

   const bool gen9 = agx3_generation == AGX_PBE_AGX3_GENERATION_9;
   const bool gen10_layered =
      agx3_generation == AGX_PBE_AGX3_GENERATION_10 &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
       legacy->dimension == AGX_TEXTURE_DIMENSION_3D);
   const bool array =
      (gen9 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY) ||
      (gen10_layered && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY);
   const bool array_multisample =
      gen9 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   const bool three_d =
      (gen9 && legacy->dimension == AGX_TEXTURE_DIMENSION_3D) ||
      (gen10_layered && legacy->dimension == AGX_TEXTURE_DIMENSION_3D);
   const bool layered = array || array_multisample || three_d;
   const bool multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      array_multisample;
   const bool compressed_2d =
      legacy->compressed && gen9 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D &&
      legacy->layout == AGX_LAYOUT_GPU && !multisample;
   const bool compressed_2d_ms =
      legacy->compressed && gen9 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU;

   if ((!layered && !multisample &&
        legacy->dimension != AGX_TEXTURE_DIMENSION_2D) ||
       (legacy->layout == AGX_LAYOUT_LINEAR &&
        (legacy->level != 0 || legacy->levels != 1 ||
         legacy->unk_mipmapped)) ||
       (legacy->layout != AGX_LAYOUT_LINEAR &&
        legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       (multisample ? (legacy->samples != AGX_SAMPLE_COUNT_2 &&
                       legacy->samples != AGX_SAMPLE_COUNT_4)
                    : legacy->samples != AGX_SAMPLE_COUNT_2) ||
       (legacy->compressed && !compressed_2d && !compressed_2d_ms) ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       (!layered && legacy->layers != 1) ||
       (multisample && (legacy->layout == AGX_LAYOUT_LINEAR ||
                        legacy->level != 0 || legacy->levels != 1 ||
                        legacy->unk_mipmapped ||
                        (gen9 && legacy->compressed && !compressed_2d_ms))) ||
       (!gen9 && (legacy->level != 0 || legacy->levels != 1 ||
                  legacy->unk_mipmapped)) ||
       (!gen9 && legacy->srgb) ||
       (legacy->extended != legacy->compressed) ||
       (legacy->layout == AGX_LAYOUT_LINEAR
           ? legacy->stride < 12 || legacy->stride > UINT32_MAX - 4
           : legacy->stride != 0) ||
       (!legacy->compressed && legacy->acceleration_buffer) ||
       legacy->depth_linear != 1 || legacy->layer_stride_linear ||
       legacy->level_offset_sw || legacy->aligned_width_msaa_sw ||
       legacy->sample_count_log2_sw || legacy->tile_width_sw != 1 ||
       legacy->tile_height_sw != 1 || legacy->layer_stride_sw ||
       legacy->buffer_offset_sw)
      return false;

   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = legacy->layout,
      .channels = legacy->channels,
      .type = legacy->type,
      .swizzle_r = legacy->swizzle_r,
      .swizzle_g = legacy->swizzle_g,
      .swizzle_b = legacy->swizzle_b,
      .swizzle_a = legacy->swizzle_a,
      .width = legacy->width,
      .height = legacy->height,
      .buffer = legacy->buffer,
      .compressed = legacy->compressed,
      .acceleration_buffer = legacy->acceleration_buffer,
      .stride_B =
         legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride + 4 : 0,
      .page_aligned_layers = legacy->page_aligned_layers,
      .srgb = legacy->srgb,
   };

   struct agx_pbe_agx3_descriptor packed;
   const unsigned samples = legacy->samples == AGX_SAMPLE_COUNT_4 ? 4u : 2u;
   const bool valid =
      multisample
         ? (array_multisample
               ? agx_pbe_agx3_2d_array_multisample_pack(
                    agx3_generation, &fields, legacy->layers, samples, &packed)
               : agx_pbe_agx3_2d_multisample_pack(agx3_generation, &fields,
                                                  samples, &packed))
      : layered
         ? (array ? agx_pbe_agx3_2d_array_mip_pack(
                       agx3_generation, &fields, legacy->layers, legacy->level,
                       legacy->levels, &packed)
                  : agx_pbe_agx3_3d_mip_pack(agx3_generation, &fields,
                                             legacy->layers, legacy->level,
                                             legacy->levels, &packed))
      : gen9 ? agx_pbe_agx3_2d_mip_pack(agx3_generation, &fields, legacy->level,
                                        legacy->levels, &packed)
             : agx_pbe_agx3_2d_pack(agx3_generation, &fields, &packed);
   if (!valid)
      return false;

   static_assert(sizeof(packed.bytes) == AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE,
                 "the AGX3 PBE descriptor is the AGX3 descriptor stride");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}
