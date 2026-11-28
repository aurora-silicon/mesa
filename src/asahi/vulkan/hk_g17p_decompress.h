/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
#include <stdint.h>

static inline bool
hk_g17p_decompress_image_type_supported(VkImageType image_type)
{
   /* 2D layer offsets and compression_layer_stride_B address 2D array layers,
    * not 3D depth slices; refuse 3D images here (note ail_layout::depth_px
    * holds arrayLayers for 2D arrays, so check the API image type directly). */
   return image_type != VK_IMAGE_TYPE_3D;
}

static inline bool
hk_g17p_decompress_pack_mode(enum agx_channels channels, uint32_t *out)
{
   switch (channels) {
   case AGX_CHANNELS_R8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8;
      return true;
   case AGX_CHANNELS_R8G8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8G8;
      return true;
   case AGX_CHANNELS_R16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16;
      return true;
   case AGX_CHANNELS_R5G6B5:
      *out = LIBAGX_AGX3_DECOMPRESS_R5G6B5;
      return true;
   case AGX_CHANNELS_R8G8B8A8:
      *out = LIBAGX_AGX3_DECOMPRESS_R8G8B8A8;
      return true;
   case AGX_CHANNELS_R10G10B10A2:
      *out = LIBAGX_AGX3_DECOMPRESS_R10G10B10A2;
      return true;
   case AGX_CHANNELS_R16G16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16G16;
      return true;
   case AGX_CHANNELS_R32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32;
      return true;
   case AGX_CHANNELS_R16G16B16A16:
      *out = LIBAGX_AGX3_DECOMPRESS_R16G16B16A16;
      return true;
   case AGX_CHANNELS_R32G32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32G32;
      return true;
   case AGX_CHANNELS_R32G32B32A32:
      *out = LIBAGX_AGX3_DECOMPRESS_R32G32B32A32;
      return true;
   case AGX_CHANNELS_R11G11B10:
      *out = LIBAGX_AGX3_DECOMPRESS_R11G11B10;
      return true;
   default:
      return false;
   }
}

static inline void
hk_agx3_decompress_texture_format(unsigned desc_gen,
                                  enum agx_channels channels,
                                  struct AGX_TEXTURE *cfg)
{
   if (desc_gen < 10) {
      /* Preserve the exact HAL200 (G16G/G17P) decompressor descriptor. */
      cfg->type = channels == AGX_CHANNELS_R5G6B5 ? AGX_TEXTURE_TYPE_UNORM
                                                  : AGX_TEXTURE_TYPE_UINT;
      cfg->swizzle_r = AGX_CHANNEL_R;
      cfg->swizzle_g = AGX_CHANNEL_G;
      cfg->swizzle_b = AGX_CHANNEL_B;
      cfg->swizzle_a = AGX_CHANNEL_A;
      return;
   }

   /* On Gen10 (M5), match swizzles to the format's channels (swizzling
    * absent channels faults on compressed 1-/2-channel targets) and decode
    * R11G11B10 as float. */
   cfg->type =
      channels == AGX_CHANNELS_R5G6B5     ? AGX_TEXTURE_TYPE_UNORM
      : channels == AGX_CHANNELS_R11G11B10 ? AGX_TEXTURE_TYPE_FLOAT
                                           : AGX_TEXTURE_TYPE_UINT;
   const bool has_g =
      channels != AGX_CHANNELS_R8 && channels != AGX_CHANNELS_R16 &&
      channels != AGX_CHANNELS_R32;
   const bool has_b =
      has_g && channels != AGX_CHANNELS_R8G8 &&
      channels != AGX_CHANNELS_R16G16 && channels != AGX_CHANNELS_R32G32;
   const bool has_a =
      has_b && channels != AGX_CHANNELS_R5G6B5 &&
      channels != AGX_CHANNELS_R11G11B10;
   cfg->swizzle_r = AGX_CHANNEL_R;
   cfg->swizzle_g = has_g ? AGX_CHANNEL_G : AGX_CHANNEL_0;
   cfg->swizzle_b = has_b ? AGX_CHANNEL_B : AGX_CHANNEL_0;
   cfg->swizzle_a = has_a ? AGX_CHANNEL_A : AGX_CHANNEL_1;
}
