/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_native_descriptor.h"

#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_pbe_agx3_core.h"
#include "util/u_math.h"

#include <math.h>
#include <string.h>

bool
agx_translate_native_texture_sample_count(
   enum agx_descriptor_generation generation, unsigned samples,
   enum agx_sample_count *out)
{
   if (!out || samples == 1 ||
       !agx_native_texture_sample_count_representable(generation, samples))
      return false;

   switch (samples) {
   case 2:
      *out = AGX_SAMPLE_COUNT_2;
      return true;
   case 4:
      *out = AGX_SAMPLE_COUNT_4;
      return true;
   case 8:
      *out = AGX_SAMPLE_COUNT_8;
      return true;
   default:
      return false;
   }
}

bool
agx_pack_native_texture(enum agx_descriptor_generation generation,
                        struct agx_native_texture_descriptor *out,
                        const struct AGX_TEXTURE *semantic)
{
   const unsigned descriptor_size = agx_texture_descriptor_size(generation);

   if (!out || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);

   if (!semantic)
      return false;

   if (agx_descriptor_generation_is_agx2(generation)) {
      AGX_TEXTURE_pack(out->opaque, semantic);
      return true;
   }

   const bool multisampled =
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;

   if (!agx_native_texture_supported(generation, multisampled,
                                     semantic->buffer_offset_sw))
      return false;

   /* The Apple Gen9/10 encoders accept the three recovered two-bit selector
    * values only. Reject malformed semantic input instead of truncating it.
    */
   if (multisampled) {
      unsigned samples;
      switch (semantic->samples) {
      case AGX_SAMPLE_COUNT_2:
         samples = 2;
         break;
      case AGX_SAMPLE_COUNT_4:
         samples = 4;
         break;
      case AGX_SAMPLE_COUNT_8:
         samples = 8;
         break;
      default:
         return false;
      }

      if (!agx_native_texture_sample_count_representable(generation, samples))
         return false;
   } else if (semantic->samples != AGX_SAMPLE_COUNT_2) {
      /* Sample Count has no 1x value, so zero/2x is the inert default for a
       * non-multisampled dimension. Do not silently accept other selectors
       * where the hardware field has no defined meaning.
       */
      return false;
   }

   const bool has_levels =
      semantic->first_level || semantic->last_level || semantic->unk_mipmapped;
   const bool extended = semantic->extended || has_levels;

   if ((semantic->address & 0xf) || semantic->address >> 48 ||
       (semantic->compressed &&
        ((semantic->acceleration_buffer & 0xf) ||
         semantic->acceleration_buffer >> 48)))
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_9) {
      if (semantic->compressed &&
          (!extended || !semantic->acceleration_buffer ||
           (multisampled && semantic->samples == AGX_SAMPLE_COUNT_8)))
         return false;

      if (!semantic->width || semantic->width > 16384 || !semantic->height ||
          semantic->height > 16384)
         return false;

      /* Stride and Depth share word1[61:46], sixteen bits, and the genxml
       * modifiers put stride >> 4 and depth - 1 there. util_bitpack_uint()
       * shifts without masking, so an out-of-range value would run into
       * "Page-aligned layers" (126) and "Extended" (127). Refuse instead.
       * On Gen9, Interchange uses the row-stride field just like Linear.
       */
      const bool strided = semantic->layout == AGX_LAYOUT_LINEAR ||
                           semantic->layout == AGX_LAYOUT_INTERCHANGE;
      if (strided ? ((semantic->stride >> 4) > 0xffffu)
                  : (semantic->depth > 0x10000u))
         return false;

      /* Interchange images are single-layer, single-level 2D; the layered
       * background descriptor every colour view also packs is a one-layer
       * 2D array.
       */
      if (semantic->layout == AGX_LAYOUT_INTERCHANGE &&
          ((semantic->dimension != AGX_TEXTURE_DIMENSION_2D &&
            semantic->dimension != AGX_TEXTURE_DIMENSION_2D_ARRAY) ||
           semantic->depth != 1 || has_levels))
         return false;

      /* On extended nonlinear Gen9 descriptors, word2 carries compression
       * metadata when compressed, or an optional caller-supplied range-end VA
       * (0 when unused, as in Borealis). Validate any non-zero VA rather
       * than silently truncating it.
       */
      if (extended && !strided && semantic->acceleration_buffer &&
          ((semantic->acceleration_buffer & 0xf) ||
           semantic->acceleration_buffer >> 48))
         return false;

      /* On Gen9, word1[62] must be clear for single-layer non-array 2D and
       * 2DMS GPU-tiled textures; this bit is not the inherited AIL
       * page-aligned-allocation flag. Keep this rule bounded to fully
       * represented, single-layer, one-level images; arrays, compression, and
       * non-normal image modes retain their existing policies.
       */
      const bool plain_2d =
         (semantic->dimension == AGX_TEXTURE_DIMENSION_2D ||
          semantic->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED) &&
         semantic->layout == AGX_LAYOUT_GPU && semantic->depth == 1 &&
         !semantic->compressed && !extended &&
         semantic->mode == AGX_IMAGE_MODE_NORMAL;

      struct AGX_TEXTURE_GEN9 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .has_levels = has_levels,
         .compressed = semantic->compressed,
         .state_control = semantic->mode | (semantic->compression << 2),
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = strided ? semantic->stride : 0,
         .depth = strided ? 1 : semantic->depth,
         .page_aligned_layers = plain_2d ? false : semantic->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            extended && (semantic->compressed || !strided)
               ? semantic->acceleration_buffer
               : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN9_pack(out->opaque, &cfg);
      return true;
   }

   if (generation == AGX_DESCRIPTOR_GENERATION_10) {
      if (!semantic->width || semantic->width > 32768 || !semantic->height ||
          semantic->height > 32768)
         return false;

      struct AGX_TEXTURE_GEN10 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .unk_mipmapped = semantic->unk_mipmapped,
         .compressed = semantic->compressed,
         .image_mode_enable = semantic->mode != AGX_IMAGE_MODE_NORMAL,
         /* The HAL300 texture encoder writes 1 for 16 KiB, 2 for 64 KiB and 3
          * for 256 KiB sparse pages. AIL uses 16 KiB pages.
          */
         .image_mode =
            semantic->mode == AGX_IMAGE_MODE_SPARSE ? 1 : semantic->mode,
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = semantic->layout == AGX_LAYOUT_LINEAR ? semantic->stride : 0,
         .depth = semantic->layout == AGX_LAYOUT_LINEAR ? 1 : semantic->depth,
         .page_aligned_layers = semantic->page_aligned_layers,
         .extended = extended,
         /* Gen10 keeps this word for compression metadata only. */
         .acceleration_buffer =
            semantic->compressed && extended ? semantic->acceleration_buffer
                                             : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN10_pack(out->opaque, &cfg);
      return true;
   }

   return false;
}

static bool
agx_pack_null_native_texture(enum agx_descriptor_generation generation,
                             struct agx_native_texture_descriptor *out)
{
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_0,
      .swizzle_g = AGX_CHANNEL_0,
      .swizzle_b = AGX_CHANNEL_0,
      .swizzle_a = AGX_CHANNEL_0,
      .address = AGX_ZERO_PAGE_ADDRESS,
   };

   return agx_pack_native_texture(generation, out, &semantic);
}

/* The null descriptor is a pure function of the generation, so pack it once
 * per generation. A refused generation is never cached.
 */
struct agx_native_texture_descriptor
   agx_null_native_texture_packed[AGX_NULL_NATIVE_TEXTURE_GENERATIONS];
uint32_t agx_null_native_texture_state[AGX_NULL_NATIVE_TEXTURE_GENERATIONS];

bool
agx_set_null_native_texture_slow(enum agx_descriptor_generation generation,
                                 struct agx_native_texture_descriptor *out)
{
   const unsigned size = agx_texture_descriptor_size(generation);

   if (!out || !size)
      return false;

   if ((unsigned)generation < AGX_NULL_NATIVE_TEXTURE_GENERATIONS &&
       p_atomic_read(&agx_null_native_texture_state[generation]) == 2) {
      memcpy(out, &agx_null_native_texture_packed[generation], size);
      return true;
   }

   struct agx_native_texture_descriptor tmp = {0};
   if (!agx_pack_null_native_texture(generation, &tmp)) {
      memset(out, 0, size);
      return false;
   }

   if ((unsigned)generation < AGX_NULL_NATIVE_TEXTURE_GENERATIONS &&
       p_atomic_cmpxchg(&agx_null_native_texture_state[generation], 0, 1) ==
          0) {
      agx_null_native_texture_packed[generation] = tmp;
      p_atomic_set(&agx_null_native_texture_state[generation], 2);
   }

   memcpy(out, &tmp, size);
   return true;
}

static uint32_t
agx_pbe_agx3_generation(enum agx_descriptor_generation generation)
{
   switch (generation) {
   case AGX_DESCRIPTOR_GENERATION_9:
      return AGX_PBE_AGX3_GENERATION_9;
   case AGX_DESCRIPTOR_GENERATION_10:
      return AGX_PBE_AGX3_GENERATION_10;
   default:
      return 0;
   }
}

bool
agx_pack_native_pbe(enum agx_descriptor_generation generation,
                    struct agx_native_pbe_descriptor *out,
                    const struct AGX_PBE *legacy)
{
   const unsigned descriptor_size = agx_pbe_descriptor_size(generation);

   if (!out || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);

   if (!legacy)
      return false;

   if (agx_descriptor_generation_is_agx2(generation)) {
      AGX_PBE_pack(out->opaque, legacy);
      return true;
   }

   const uint32_t agx3_generation = agx_pbe_agx3_generation(generation);
   if (!agx3_generation)
      return false;

   /* The recovered subset excludes fields whose Gen9/10 semantic placement
    * remains incomplete: sparse mode and the AGX2 software-addressing
    * sideband. Both Gen9 and Gen10 admit 2D (linear, twiddled, GPU, and on
    * Gen9 single-level Interchange), 2D array, 3D, 2D multisample, and 2D
    * array multisample, plus lossless GPU-tiled compression (2D/layered/2x-4x
    * MSAA on Gen9; 2D, 1-layer array, and qualified 2x-4x MSAA formats on
    * Gen10).
    */
   const bool gen9 = agx3_generation == AGX_PBE_AGX3_GENERATION_9;
   const bool gen10 = agx3_generation == AGX_PBE_AGX3_GENERATION_10;
   const bool array = legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY;
   const bool array_multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   const bool three_d = legacy->dimension == AGX_TEXTURE_DIMENSION_3D;
   const bool layered = array || array_multisample || three_d;
   const bool multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      array_multisample;
   const bool interchange = legacy->layout == AGX_LAYOUT_INTERCHANGE;
   const bool compressed_gen9 =
      legacy->compressed && gen9 &&
      (legacy->layout == AGX_LAYOUT_GPU ||
       (interchange && !layered && !multisample));
   const bool compressed_2d =
      legacy->compressed &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D ||
       (gen10 && array && legacy->layers == 1 && legacy->level == 0 &&
        legacy->levels == 1)) &&
      (legacy->layout == AGX_LAYOUT_GPU || (gen9 && interchange)) &&
      !multisample;
   /* Gen10 compressed 2x/4x MSAA is admitted only for the verified single-layer
    * 2D MSAA format subset (R32_UINT, RGBA16 F/UI, RGBA8 UNORM/sRGB).
    */
   const bool compressed_msaa_r32_uint =
      gen10 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R32 &&
      legacy->type == AGX_TEXTURE_TYPE_UINT && !legacy->srgb &&
      legacy->layers == 1 && legacy->level == 0 && legacy->levels == 1 &&
      !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   const bool compressed_msaa_rgba16 =
      gen10 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R16G16B16A16 &&
      (legacy->type == AGX_TEXTURE_TYPE_FLOAT ||
       legacy->type == AGX_TEXTURE_TYPE_UINT) &&
      !legacy->srgb && legacy->layers == 1 && legacy->level == 0 &&
      legacy->levels == 1 && !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   const bool compressed_msaa_rgba8 =
      gen10 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R8G8B8A8 &&
      legacy->type == AGX_TEXTURE_TYPE_UNORM && legacy->layers == 1 &&
      legacy->level == 0 && legacy->levels == 1 && !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   const bool compressed_gen10_msaa =
      compressed_msaa_r32_uint || compressed_msaa_rgba16 ||
      compressed_msaa_rgba8;

   if ((!layered && !multisample &&
        legacy->dimension != AGX_TEXTURE_DIMENSION_2D) ||
       (legacy->layout == AGX_LAYOUT_LINEAR &&
        (legacy->level != 0 || legacy->levels != 1 ||
         legacy->unk_mipmapped)) ||
       (legacy->layout != AGX_LAYOUT_LINEAR &&
        legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU &&
        !(interchange && gen9 && !layered && !multisample &&
          legacy->level == 0 && legacy->levels == 1)) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       (multisample
           ? (legacy->samples != AGX_SAMPLE_COUNT_2 &&
              legacy->samples != AGX_SAMPLE_COUNT_4 &&
              !(gen10 && legacy->samples == AGX_SAMPLE_COUNT_8))
           : legacy->samples != AGX_SAMPLE_COUNT_2) ||
       (legacy->compressed && !compressed_gen9 && !compressed_2d &&
        !compressed_gen10_msaa) ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       (!layered && legacy->layers != 1) ||
       (multisample &&
        (legacy->layout == AGX_LAYOUT_LINEAR || legacy->level != 0 ||
         legacy->levels != 1 || legacy->unk_mipmapped ||
         (legacy->compressed && !compressed_gen9 && !compressed_gen10_msaa))) ||
       (legacy->extended != legacy->compressed) ||
       (legacy->layout == AGX_LAYOUT_LINEAR || interchange
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
         legacy->layout == AGX_LAYOUT_LINEAR || interchange
            ? legacy->stride + 4
            : 0,
      .page_aligned_layers = legacy->page_aligned_layers,
      .srgb = legacy->srgb,
   };

   struct agx_pbe_agx3_descriptor packed;
   const unsigned samples = legacy->samples == AGX_SAMPLE_COUNT_8 ? 8u
                          : legacy->samples == AGX_SAMPLE_COUNT_4 ? 4u
                                                                  : 2u;
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
         : agx_pbe_agx3_2d_mip_pack(agx3_generation, &fields, legacy->level,
                                    legacy->levels, &packed);
   if (!valid)
      return false;

   static_assert(sizeof(out->opaque) == sizeof(packed.bytes),
                 "opaque descriptor matches the packed size");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}

bool
agx_pack_g18_native_pbe(struct agx_native_pbe_descriptor *out,
                        const struct AGX_PBE *legacy)
{
   if (!out || !legacy)
      return false;

   memset(out, 0, AGX_NATIVE_DESCRIPTOR_AGX3_SIZE);

   const bool array_multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   const bool multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      array_multisample;
   const bool g18_layered =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
      legacy->dimension == AGX_TEXTURE_DIMENSION_3D;
   const bool layered = g18_layered || array_multisample;

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
       (legacy->compressed && !multisample) ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       (!layered && legacy->layers != 1) ||
       (g18_layered && legacy->layout == AGX_LAYOUT_LINEAR) ||
       (multisample && (legacy->layout == AGX_LAYOUT_LINEAR ||
                        legacy->level != 0 || legacy->levels != 1 ||
                        legacy->unk_mipmapped)) ||
       legacy->srgb || (legacy->extended != legacy->compressed) ||
       (legacy->layout == AGX_LAYOUT_LINEAR
           ? legacy->stride < 12 || legacy->stride > UINT32_MAX - 4
           : legacy->stride != 0) ||
       (!legacy->compressed && !multisample && legacy->acceleration_buffer) ||
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
      g18_layered
         ? agx_pbe_g18_layered_mip_pack(&fields, legacy->dimension,
                                        legacy->layers, legacy->level,
                                        legacy->levels, &packed)
      : multisample
         ? (array_multisample
               ? agx_pbe_g18_2d_array_multisample_pack(&fields, legacy->layers,
                                                       samples, &packed)
               : agx_pbe_g18_2d_multisample_pack(&fields, samples, &packed))
         : agx_pbe_g18_2d_mip_pack(&fields, legacy->level, legacy->levels,
                                   &packed);
   if (!valid)
      return false;

   static_assert(sizeof(out->opaque) == sizeof(packed.bytes),
                 "opaque descriptor matches the packed size");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}

bool
agx_set_null_native_pbe(enum agx_descriptor_generation generation,
                        struct agx_native_pbe_descriptor *out)
{
   /* AGX_PBE_header already supplies the 1s for width, height, levels,
    * layers, depth_linear and the two software tile dimensions.
    */
   struct AGX_PBE semantic = {
      AGX_PBE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_R,
      .swizzle_b = AGX_CHANNEL_R,
      .swizzle_a = AGX_CHANNEL_R,
      .buffer = AGX_SCRATCH_PAGE_ADDRESS,
   };

   /* AGX3's recovered PBE geometry requires a 2D dimension; AGX2 keeps its
    * historical 1D null PBE encoding.
    */
   if (agx_descriptor_generation_is_agx3(generation))
      semantic.dimension = AGX_TEXTURE_DIMENSION_2D;

   return agx_pack_native_pbe(generation, out, &semantic);
}

bool
agx_native_sampler_reduction_supported(
   enum agx_sampler_generation generation,
   enum agx_native_sampler_reduction_mode reduction_mode)
{
   if (reduction_mode == AGX_NATIVE_SAMPLER_REDUCTION_WEIGHTED_AVERAGE)
      return generation != AGX_SAMPLER_GENERATION_UNKNOWN;

   return generation == AGX_SAMPLER_GENERATION_GEN5 &&
          (reduction_mode == AGX_NATIVE_SAMPLER_REDUCTION_MINIMUM ||
           reduction_mode == AGX_NATIVE_SAMPLER_REDUCTION_MAXIMUM);
}

static bool
agx_sampler_contract_valid(enum agx_sampler_generation generation,
                           unsigned element_size)
{
   switch (generation) {
   case AGX_SAMPLER_GENERATION_GEN4:
      return element_size == AGX_SAMPLER_LENGTH ||
             element_size == AGX_NATIVE_SAMPLER_MAX_ELEMENT_SIZE;
   case AGX_SAMPLER_GENERATION_GEN4_HAL200:
   case AGX_SAMPLER_GENERATION_GEN5:
      return element_size == AGX_NATIVE_SAMPLER_MAX_ELEMENT_SIZE;
   case AGX_SAMPLER_GENERATION_UNKNOWN:
   default:
      return false;
   }
}

bool
agx_pack_native_sampler(enum agx_chip chip,
                        struct agx_native_sampler_descriptor *out,
                        const struct agx_native_sampler_state *state)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));
   if (!state || !util_is_power_of_two_nonzero(state->maximum_anisotropy) ||
       state->maximum_anisotropy > 16)
      return false;

   const enum agx_sampler_generation generation =
      agx_chip_sampler_generation(chip);
   const unsigned element_size = agx_chip_sampler_heap_element_size(chip);
   if (!agx_sampler_contract_valid(generation, element_size) ||
       !agx_native_sampler_reduction_supported(generation,
                                               state->reduction_mode) ||
       (state->compare_enable &&
        state->reduction_mode !=
           AGX_NATIVE_SAMPLER_REDUCTION_WEIGHTED_AVERAGE))
      return false;

   switch (generation) {
   case AGX_SAMPLER_GENERATION_GEN4:
      agx_pack(out->opaque, SAMPLER, cfg) {
         cfg.minimum_lod = state->minimum_lod;
         cfg.maximum_lod = state->maximum_lod;
         cfg.maximum_anisotropy = state->maximum_anisotropy;
         cfg.magnify = state->magnify;
         cfg.minify = state->minify;
         cfg.mip_filter = state->mip_filter;
         cfg.wrap_s = state->wrap_s;
         cfg.wrap_t = state->wrap_t;
         cfg.wrap_r = state->wrap_r;
         cfg.pixel_coordinates = state->pixel_coordinates;
         if (state->compare_enable) {
            cfg.compare_func = state->compare_func;
            cfg.compare_enable = true;
         }
         cfg.border_colour = state->border_colour;
         cfg.seamful_cube_maps = state->seamful_cube_maps;
      }
      return true;

   case AGX_SAMPLER_GENERATION_GEN4_HAL200:
      agx_pack(out->opaque, SAMPLER_HAL200, cfg) {
         cfg.minimum_lod = state->minimum_lod;
         cfg.maximum_lod = state->maximum_lod;
         cfg.maximum_anisotropy = state->maximum_anisotropy;
         cfg.magnify = state->magnify;
         cfg.minify = state->minify;
         cfg.mip_filter = state->mip_filter;
         cfg.wrap_s = state->wrap_s;
         cfg.wrap_t = state->wrap_t;
         cfg.wrap_r = state->wrap_r;
         cfg.pixel_coordinates = state->pixel_coordinates;
         if (state->compare_enable) {
            cfg.compare_func = state->compare_func;
            cfg.compare_enable = true;
         }
         cfg.lod_bias = state->lod_bias;
         cfg.border_colour = state->border_colour;
         cfg.seamful_cube_maps = state->seamful_cube_maps;
      }
      return true;

   case AGX_SAMPLER_GENERATION_GEN5:
      agx_pack(out->opaque, SAMPLER_GEN5, cfg) {
         cfg.minimum_lod = state->minimum_lod;
         cfg.maximum_lod = state->maximum_lod;
         cfg.maximum_anisotropy = state->maximum_anisotropy;
         cfg.magnify = state->magnify;
         cfg.minify = state->minify;
         cfg.mip_filter = state->mip_filter;
         cfg.wrap_s = state->wrap_s;
         cfg.wrap_t = state->wrap_t;
         cfg.wrap_r = state->wrap_r;
         cfg.pixel_coordinates = state->pixel_coordinates;
         cfg.lod_bias = state->lod_bias;
         cfg.border_colour = state->border_colour;
         cfg.seamful_cube_maps = state->seamful_cube_maps;

         if (state->reduction_mode == AGX_NATIVE_SAMPLER_REDUCTION_MINIMUM) {
            cfg.reduction_mode = AGX_SAMPLER_REDUCTION_MODE_MINIMUM;
         } else if (state->reduction_mode ==
                    AGX_NATIVE_SAMPLER_REDUCTION_MAXIMUM) {
            cfg.reduction_mode = AGX_SAMPLER_REDUCTION_MODE_MAXIMUM;
         } else {
            /* Grounded by the Gen5 sampler constructor: bit 42 selects the
             * compare-function interpretation of the overlapping bits.
             */
            cfg.compare_mode = true;
            cfg.compare_func = state->compare_func;
         }
      }
      return true;

   case AGX_SAMPLER_GENERATION_UNKNOWN:
   default:
      return false;
   }
}

bool
agx_pack_native_txf_sampler(enum agx_chip chip,
                            struct agx_native_sampler_descriptor *out)
{
   const struct agx_native_sampler_state state = {
      .minimum_lod = 0.0f,
      .maximum_lod = INFINITY,
      .maximum_anisotropy = 1,
      .mip_filter = AGX_MIP_FILTER_NEAREST,
      .wrap_s = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_r = AGX_WRAP_CLAMP_TO_BORDER,
      .compare_func = AGX_COMPARE_FUNC_LEQUAL,
      .border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK,
      .reduction_mode = AGX_NATIVE_SAMPLER_REDUCTION_WEIGHTED_AVERAGE,
   };

   return agx_pack_native_sampler(chip, out, &state);
}
