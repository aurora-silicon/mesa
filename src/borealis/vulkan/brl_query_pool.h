/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "brl_private.h"
#include "vk_query_pool.h"

struct agx_bo;

struct brl_query_pool {
   struct vk_query_pool vk;

   uint64_t query_start;
   uint64_t query_stride;

   struct agx_bo *bo;
   void *bo_map;

   /* For timestamp queries, the kernel-assigned timestamp buffer handle. On
    * the native AGX3 path, occlusion pools bind their availability slots the
    * same way, so the render that ends a query can mark it available through
    * its own retirement timestamp (0 if binding failed). Unused for all other
    * query types.
    */
   uint32_t handle;

   /* log2 of the availability slot size in 32-bit words: 1 for native AGX3
    * occlusion pools (64-bit timestamp slots), 0 otherwise. */
   uint8_t avail_shift;

   unsigned oq_queries;

   /* Occlusion queries: the queue and timeline point of the latest
    * submission that ended each query, point 0 if none since the last host
    * reset. */
   struct brl_oq_submit {
      struct brl_queue *queue;
      uint64_t point;
   } *oq_submit;
};

struct brl_cmd_buffer;
struct brl_queue;

struct brl_oq_end {
   struct brl_query_pool *pool;
   uint32_t query;
};

/* Record the queue timeline point that the submission of cmd signals. */
void brl_query_note_submit(struct brl_cmd_buffer *cmd, struct brl_queue *queue,
                          uint64_t point);

VK_DEFINE_NONDISP_HANDLE_CASTS(brl_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)
