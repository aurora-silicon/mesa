/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "asahi/lib/agx_bo.h"
#include "vk_device_memory.h"

#include "util/list.h"

struct hk_device;
struct hk_image_plane;

struct hk_device_memory {
   struct vk_device_memory vk;

   struct agx_bo *bo;

   void *map;

   /* map is this object's own mapping of map_size bytes rather than the
    * BO's shared one: the BO may be aliased by other memory objects. */
   size_t map_size;
   bool map_owned;

   /* Residency preference the kernel accepted (memory priority 1.0). */
   bool keep_resident;

   /* A host import in hk_device::host_alias.imports (host_alias.dynamic). */
   bool host_alias_tracked;
   /* An fd or dma-buf import, counted in hk_device::alias_live for its
    * lifetime (hk_device_alias_get()). */
   bool alias_fd;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

/* The device address of the start of the memory object */
static inline uint64_t
hk_device_memory_addr(const struct hk_device_memory *mem)
{
   return mem->bo->va->addr;
}

extern const VkExternalMemoryProperties hk_opaque_fd_mem_props;
extern const VkExternalMemoryProperties hk_dma_buf_mem_props;
extern const VkExternalMemoryProperties hk_host_ptr_mem_props;
