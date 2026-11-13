/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hk_buffer.h"
#include "hk_private.h"

#include "vk_object.h"

/* VK_EXT_opacity_micromap object: a range of a buffer holding a micromap
 * image (libagx/micromap.h). */
struct hk_micromap {
   struct vk_object_base base;
   struct hk_buffer *buffer;
   uint64_t offset, size;
   VkMicromapTypeEXT type;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_micromap, base, VkMicromapEXT,
                               VK_OBJECT_TYPE_MICROMAP_EXT)

static inline uint64_t
hk_micromap_va(const struct hk_micromap *micromap)
{
   return micromap ? vk_buffer_address(&micromap->buffer->vk, micromap->offset)
                   : 0;
}
