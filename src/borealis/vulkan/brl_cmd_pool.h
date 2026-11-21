/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "brl_private.h"
#include "util/macros.h"
#include "asahi/lib/agx_device.h"
#include "util/cache_ops.h"

#include "vk_command_pool.h"

#define BRL_CMD_BO_SIZE     1024 * 128
#define BRL_CMD_POOL_BO_MAX 32
#define BRL_G17P_CMD_POOL_MAIN_BO_MAX 256

/* A recording with no recycled BO starts each uploader on a span of a shared
 * slab: one tile upload writes under 8 KiB in total, and a whole BO per
 * uploader costs it five VM binds, five mmaps and 640 KiB of first touches.
 * A span is 16 KiB so it keeps the alignment a BO base gives its uploader. */
#define BRL_CMD_SPAN_SIZE (16 * 1024)
/* Two G17P render stream roots, or one compute root. */
#define BRL_CMD_STREAM_SPAN_SIZE (64 * 1024)
#define BRL_CMD_SLAB_MAX_SIZE (4 * 1024 * 1024)

enum brl_cmd_bo_type {
   BRL_CMD_BO_MAIN,
   BRL_CMD_BO_USC,
   BRL_CMD_BO_ROOT,
   BRL_CMD_BO_TYPE_COUNT,
};

/* Recyclable command buffer BO, used for both push buffers and upload */
struct brl_cmd_bo {
   /* Recording/pool ownership plus immutable in-flight submission pins. */
   int refcount;
   struct agx_bo *bo;

   void *map;

   /* What this object owns: all of bo, or span of a device slab. */
   uint64_t addr;
   uint32_t size;
   struct brl_vdm_allocation *span;

   /* Full VA of the G17P VDM alias in the architectural user window. */
   uint64_t g17p_vdm_alias;

   /* This recording allocated the BO only through uploader.streams. */
   bool stream_only;

   /** Link in a matching command-pool or command-buffer BO list. */
   struct list_head link;
};

struct brl_vdm_slab;

struct brl_vdm_allocation {
   int refcount;
   struct list_head link;
   struct brl_vdm_slab *slab;
   uint64_t addr;
   uint64_t alias;
   uint64_t size;
};

struct agx_device;
struct agx_ptr;
struct brl_device;

/* G17P command-buffer memory (main/USC pools, VDM slabs, Ioto arenas) is
 * CPU-cached; recording cleans what it wrote before submission
 * (brl_cmd_buffer_publish_cached_uploads()). */
static inline bool
brl_g17p_cached_cmd_memory(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P && util_has_cache_ops();
}

struct brl_cmd_pool;

void brl_cmd_pool_clean_vdm(const struct brl_vdm_allocation *allocation);

/* A VDM record's pointer is 40 bits and hardware reads bit 39 as the UAT
 * root select, adding 0x1000000000 when it is clear.  G17P's main heap sits
 * at 0x8000000000, so every main-heap pointer a VDM record carried would have
 * bit 39 set and be fetched from the upper root instead: the dVDM_SSD
 * unmapped read at (1<<42)|0x7ffee0400 hardware reported for the first PPP
 * state record after the root chunk (0x87ffee0400).  Native keeps these
 * payloads inside the architectural 0x1000000000 window, and so does G17P
 * recording. */
static inline bool
brl_g17p_vdm_low_window_enabled(const struct agx_device *dev)
{
   return agx_chip_native_agx3(dev->chip);
}

struct brl_cmd_pool {
   struct vk_command_pool vk;

   /** Recyclable brl_cmd_bo lists, kept separate by mapping and VA policy. */
   struct list_head free_bos[BRL_CMD_BO_TYPE_COUNT];
   uint32_t num_free_bos[BRL_CMD_BO_TYPE_COUNT];
   /** Spans likewise, so a warm small recording skips the device lock. */
   struct list_head free_spans[BRL_CMD_BO_TYPE_COUNT];
   uint32_t num_free_spans[BRL_CMD_BO_TYPE_COUNT];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(brl_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

static inline struct brl_device *
brl_cmd_pool_device(struct brl_cmd_pool *pool)
{
   return (struct brl_device *)pool->vk.base.device;
}

VkResult brl_cmd_pool_alloc_bo(struct brl_cmd_pool *pool,
                              enum brl_cmd_bo_type type,
                              struct brl_cmd_bo **bo_out);

void brl_cmd_pool_free_bo_list(struct brl_cmd_pool *pool,
                              enum brl_cmd_bo_type type,
                              struct list_head *bos);
VkResult brl_cmd_pool_alloc_bo_span(struct brl_cmd_pool *pool,
                                   enum brl_cmd_bo_type type, uint32_t size,
                                   struct brl_cmd_bo **bo_out);
VkResult brl_cmd_bo_ensure_vdm_alias(struct brl_cmd_pool *pool,
                                    struct brl_cmd_bo *bo);

VkResult brl_cmd_pool_alloc_span(struct brl_cmd_pool *pool,
                                enum brl_cmd_slab_class class, uint64_t size,
                                uint64_t alignment, struct agx_ptr *ptr,
                                struct brl_vdm_allocation **allocation);
void brl_device_free_span(struct brl_device *dev,
                         struct brl_vdm_allocation *allocation);
void brl_device_finish_vdm_slabs(struct brl_device *dev);

/* Command-memory retirement, independent of API semaphores: on native M5
 * for the memory of every submitted recording, elsewhere only for a
 * driver-owned buffer a submission reads (extra_bo, a packed compute
 * stream), which goes back to the BO cache only once the submission
 * retired. */
struct agx_bo;
struct brl_cmd_pending;
struct brl_queue;
struct vk_queue_submit;
VkResult brl_cmd_pending_begin(struct brl_device *dev,
                               const struct vk_queue_submit *submit,
                               unsigned chunks, struct agx_bo *extra_bo,
                               struct brl_cmd_pending **out);
uint32_t brl_cmd_pending_signal(struct brl_cmd_pending *pending);
void brl_cmd_pending_end(struct brl_device *dev, struct brl_cmd_pending *pending,
                        struct brl_queue *queue, VkResult result);
void brl_cmd_pending_reap(struct brl_device *dev);
void brl_cmd_pending_finish_queue(struct brl_device *dev, struct brl_queue *queue);
void brl_cmd_pending_finish(struct brl_device *dev);
