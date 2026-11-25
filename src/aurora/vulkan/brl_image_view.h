/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_pack.h"
#include "brl_private.h"
#include "brl_native_descriptor_pack.h"
#include "vk_image.h"

struct brl_device;

#define BRL_MAX_IMAGE_DESCS (10 * BRL_MAX_PLANES)

struct brl_image_view {
   struct vk_image_view vk;

   uint8_t plane_count;
   /* Descriptor encodings of the view's fixed minimum LOD (relative to its
    * base level), computed once instead of on every descriptor write. */
   uint16_t sampled_min_lod_fp16;
   uint16_t sampled_min_lod_uint16;
   /* G17P sampled-descriptor format state of the view's logical format,
    * computed once: the integer conversion kind of a scaled format (0 when
    * not scaled) and whether it is USCALED with a normalized filter form. */
   uint16_t g17p_scaled_conversion;
   bool g17p_uscaled_normalized;
   struct {
      uint8_t image_plane;
      uint32_t sampled_heap_index;
      uint32_t storage_heap_index;
      uint32_t input_heap_index;
      uint32_t emrt_read_heap_index;
      uint32_t filterable_heap_index;

      struct brl_native_texture_descriptor ia, sampled, ro_storage, background,
         layered_background, emrt_texture, sampled_filterable;
      bool filterable_valid;

      struct brl_native_pbe_descriptor storage, eot, layered_eot, emrt_pbe;
      uint64_t storage_atomic_sw;
      /* M5 store window record (brl_native_descriptor.h), zero if unused. */
      uint32_t storage_window[4];
      uint64_t emrt_atomic_sw;
      uint32_t emrt_aligned_width;
      bool storage_pbe_valid, eot_pbe_valid, layered_eot_pbe_valid,
         emrt_pbe_valid;
   } planes[3];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(brl_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

/* VK_EXT_descriptor_buffer capture/replay data of an image view: the device
 * heap index of each of its textures, 0 for none. */
enum brl_view_heap {
   BRL_VIEW_HEAP_SAMPLED,
   BRL_VIEW_HEAP_FILTERABLE,
   BRL_VIEW_HEAP_STORAGE,
   BRL_VIEW_HEAP_INPUT,
   BRL_VIEW_HEAP_EMRT_READ,
   BRL_VIEW_HEAP_COUNT,
};

struct brl_image_view_capture {
   uint32_t index[BRL_MAX_PLANES][BRL_VIEW_HEAP_COUNT];
};
