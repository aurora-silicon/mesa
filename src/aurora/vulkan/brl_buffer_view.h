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

#include "vk_buffer_view.h"

struct brl_physical_device;

VkFormatFeatureFlags2
brl_get_buffer_format_features(struct brl_physical_device *pdevice,
                              VkFormat format);

struct brl_buffer_view {
   struct vk_buffer_view vk;
   struct brl_native_texture_descriptor tex;
   struct brl_native_pbe_descriptor pbe;
   /* Sampled-image device heap slot on AGX3/M5; zero on earlier chips, which
    * read texture descriptors inline.
    */
   uint32_t sampled_heap_index;
   struct brl_g17p_buffer_texture_metadata buffer_metadata;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(brl_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)

struct brl_device;

/* VK_EXT_descriptor_buffer texel buffers (G17P, see brl_buffer_view.c) */
VkResult brl_texel_windows_init(struct brl_device *dev);
void brl_texel_windows_finish(struct brl_device *dev);

/* Write the AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE bytes of a texel buffer
 * descriptor for vkGetDescriptorEXT() into zeroed memory; a null or
 * unrepresentable buffer gets the null descriptor. */
void brl_texel_buffer_descriptor(struct brl_device *dev,
                                const VkDescriptorAddressInfoEXT *info,
                                bool storage, void *out);
