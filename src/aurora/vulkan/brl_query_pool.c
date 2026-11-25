/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "brl_query_pool.h"
#include "brl_query_oq.h"
#include "brl_frame_anatomy.h"

#include <errno.h>
#include <xf86drm.h>

#include "agx_compile.h"
#include "agx_pack.h"
#include "brl_buffer.h"
#include "brl_cmd_buffer.h"
#include "brl_device.h"
#include "brl_entrypoints.h"
#include "brl_queue.h"
#include "brl_pipeline_statistics_core.h"
#include "brl_queue.h"
#include "brl_shader.h"
#include "brl_acceleration_structure.h"

#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"

#include "aurora/lib/agx_bo.h"
#include "aurora/libagx/libagx.h"
#include "aurora/libagx/query.h"
#include "aurora/libagx/gen1_as.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

#include "drm-uapi/asahi_drm.h"
#include "util/os_time.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

struct brl_query_report {
   uint64_t value;
};
static_assert(sizeof(struct brl_query_report) == sizeof(uint64_t),
              "CPU result ABI requires contiguous u64 reports");

static inline bool
brl_has_available(const struct brl_query_pool *pool)
{
   return pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP;
}

static inline bool
brl_is_acceleration_structure_query(VkQueryType type)
{
   return type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ||
          type == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR;
}

static uint16_t *
brl_pool_oq_index_ptr(const struct brl_query_pool *pool)
{
   /* A pool whose BO allocation failed (every BO creation fails once the VM
    * is lost) reaches here from brl_CreateQueryPool's error path. */
   if (!pool->bo)
      return NULL;
   void *map = agx_bo_map(pool->bo);
   return map ? (uint16_t *)((char *)map + pool->query_start) : NULL;
}

static uint32_t
brl_reports_per_query(struct brl_query_pool *pool)
{
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION:
   case VK_QUERY_TYPE_TIMESTAMP:
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      return 1;
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      return util_bitcount(pool->vk.pipeline_statistics);
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      // Primitives succeeded and primitives needed
      return 2;
   default:
      UNREACHABLE("Unsupported query type");
   }
}

static uint64_t
brl_query_available_size(const struct brl_query_pool *pool)
{
   return sizeof(uint32_t) << pool->avail_shift;
}

static uint64_t
brl_query_available_offset(struct brl_query_pool *pool, uint32_t query)
{
   assert(brl_has_available(pool));
   assert(query < pool->vk.query_count);
   return (uint64_t)query * brl_query_available_size(pool);
}

static uint64_t
brl_query_available_addr(struct brl_query_pool *pool, uint32_t query)
{
   return pool->bo->va->addr + brl_query_available_offset(pool, query);
}

/* The availability slot of a query, nonzero once available. A 64-bit slot
 * holds 1 when a queued store made the query available and a retirement
 * timestamp when the kernel did (brl_cmd_end_query_stamped()). */
static uint64_t
brl_query_available_value(struct brl_query_pool *pool, uint32_t query)
{
   void *slot = (char *)agx_bo_map(pool->bo) +
                brl_query_available_offset(pool, query);
   if (pool->avail_shift)
      return p_atomic_read((uint64_t *)slot);
   return p_atomic_read((uint32_t *)slot);
}

static void
brl_query_set_available_host(struct brl_query_pool *pool, uint32_t query,
                            bool available)
{
   void *slot = (char *)agx_bo_map(pool->bo) +
                brl_query_available_offset(pool, query);
   if (pool->avail_shift)
      *(uint64_t *)slot = available;
   else
      *(uint32_t *)slot = available;
}

static uint64_t
brl_query_offset(struct brl_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return pool->query_start + query * pool->query_stride;
}

static uint64_t
brl_query_report_addr(struct brl_device *dev, struct brl_query_pool *pool,
                     uint32_t query)
{
   if (pool->oq_queries) {
      uint16_t *oq_index = brl_pool_oq_index_ptr(pool);
      return dev->occlusion_queries.bo->va->addr +
             (oq_index[query] * sizeof(uint64_t));
   } else {
      return pool->bo->va->addr + brl_query_offset(pool, query);
   }
}

static struct brl_query_report *
brl_query_report_map(struct brl_device *dev, struct brl_query_pool *pool,
                    uint32_t query)
{
   if (pool->oq_queries) {
      uint64_t *queries = (uint64_t *)agx_bo_map(dev->occlusion_queries.bo);
      uint16_t *oq_index = brl_pool_oq_index_ptr(pool);

      if (oq_index[query] == LIBAGX_QUERY_INDEX_UNUSED)
         return NULL;

      return (struct brl_query_report *)&queries[oq_index[query]];
   } else {
      return (void *)((char *)agx_bo_map(pool->bo) +
                      brl_query_offset(pool, query));
   }
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   uint32_t kind;
   switch (queryType) {
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
      kind = 0;
      break;
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
      kind = 1;
      break;
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      kind = 2;
      break;
   default:
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if ((!dev->g17p_native_as && !dev->m5_native_as) || pool->vk.query_type != queryType ||
       firstQuery > pool->vk.query_count ||
       accelerationStructureCount > pool->vk.query_count - firstQuery) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!accelerationStructureCount || !brl_query_command_cs(cmd))
      return;
   for (uint32_t i = 0; i < accelerationStructureCount; ++i) {
      VK_FROM_HANDLE(vk_acceleration_structure, as, pAccelerationStructures[i]);
      if (dev->g17p_native_as) {
         uint64_t metadata = brl_gen1_as_metadata_addr(cmd, as);
         if (!metadata)
            return;
         libagx_gen1_as_property(cmd, agx_1d(1), AGX_BARRIER_ALL,
            vk_acceleration_structure_get_va(as),
            metadata, brl_query_report_addr(dev, pool, firstQuery + i),
            queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ? 0 :
            queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR ? 1 :
            queryType == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ? 2 : 3);
      } else {
         libagx_agx3_as_property(cmd, agx_1d(1), AGX_BARRIER_ALL,
            ALIGN_POT(vk_acceleration_structure_get_va(as), UINT64_C(1024)),
            brl_query_report_addr(dev, pool, firstQuery + i), kind);
      }
      if (vk_command_buffer_has_error(&cmd->vk))
         return;
      /* Acceleration-structure property pools have no kernel timestamp
       * binding (pool->handle), so this store touches nothing the kernel
       * publishes. */
      brl_queue_write(cmd, brl_query_available_addr(pool, firstQuery + i), 1, false);
   }
}

static void
brl_note_dispatched_imm_write_queries(struct brl_cs *cs)
{
   if (cs->imm_write_query_untracked)
      cs->query_access_untracked = true;
   util_dynarray_foreach(&cs->imm_write_query_ranges,
                         struct brl_query_access_range, r) {
      brl_cs_note_query_range(cs, r->handle, r->start_B, r->end_B - r->start_B);
   }
}

void
brl_dispatch_imm_writes(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   /* As soon as we mark a query available, it needs to be available system
    * wide, otherwise a CPU-side get result can query. As such, we cache flush
    * before and then let coherency works its magic. Without this barrier, we
    * get flakes in
    *
    * dEQP-VK.query_pool.occlusion_query.get_results_conservative_size_64_wait_query_without_availability_draw_triangles_discard
    */
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   brl_cdm_cache_flush(dev, cs);

   perf_debug(cmd, "Queued writes");

   uint32_t count =
      util_dynarray_num_elements(&cs->imm_writes, struct libagx_imm_write);
   assert(count > 0);

   uint64_t params =
      brl_pool_upload(cmd, cs->imm_writes.data, cs->imm_writes.size, 16);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   /* A deferred stream (brl_query_oq.c) is judged at the end of the command
    * buffer, when it is no longer current: a post-graphics dispatch would
    * open a new stream after every later command, a query copy or reset
    * included. Write into the stream itself, as native AGX3 does. */
   const bool pin = agx_chip_native_agx3(dev->dev.chip) ||
                    (cs != cmd->current_cs.post_gfx && cs != cmd->current_cs.cs);
   if (pin) {
      /* End-of-stream writes belong to this exact post-graphics stream.
       * Pin it while the common native helper obtains its CDM stream; it
       * must not move availability ahead of the VDM producer. */
      struct brl_cs *saved = cmd->current_cs.cs;
      bool pinned = cmd->sync.pin_cs;
      cmd->current_cs.cs = cs;
      cmd->sync.pin_cs = true;
      libagx_write_u32s(cmd, agx_1d(count), AGX_BARRIER_ALL, params);
      cmd->current_cs.cs = saved;
      cmd->sync.pin_cs = pinned;

      if (vk_command_buffer_has_error(&cmd->vk))
         return;
      util_dynarray_foreach(&cs->imm_writes, struct libagx_imm_write, write) {
         brl_drm_attachment_set_add(&cs->compute_attachments, write->address,
                                    sizeof(write->value));
      }
      brl_note_dispatched_imm_write_queries(cs);
      return;
   }

   libagx_write_u32s(cmd, agx_1d(count), AGX_BARRIER_ALL | AGX_POSTGFX, params);
   brl_note_dispatched_imm_write_queries(cs);
}

void
brl_record_stamp_copies(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                       const struct brl_stamp_copy *copies, unsigned count)
{
   assert(cs->type == BRL_CS_CDM && count > 0);
   assert(cs == cmd->current_cs.cs);

   perf_debug(cmd, "Timestamp copies");

   struct agx_ptr params =
      brl_pool_alloc(cmd, count * sizeof(struct libagx_timestamp_copy), 16);
   if (!params.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   struct libagx_timestamp_copy *pairs = params.cpu;
   for (unsigned i = 0; i < count; ++i) {
      pairs[i] = (struct libagx_timestamp_copy){
         .dest = copies[i].dst.addr,
         .src = copies[i].src.addr,
      };
   }

   /* One launch, then one flush for all of them. */
   libagx_copy_timestamps(cmd, agx_1d(count), AGX_BARRIER_ALL, params.gpu);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   /* Both words are query memory the launch accesses. */
   struct brl_cs *launch = cmd->current_cs.cs;
   for (unsigned i = 0; i < count; ++i) {
      brl_cmd_note_query_range(cmd, launch, copies[i].src.handle,
                              copies[i].src.offset_B, sizeof(uint64_t));
      brl_cmd_note_query_range(cmd, launch, copies[i].dst.handle,
                              copies[i].dst.offset_B, sizeof(uint64_t));
   }
}

void
brl_queue_write(struct brl_cmd_buffer *cmd, uint64_t address, uint32_t value,
               bool after_gfx)
{
   struct brl_cs *cs = brl_cmd_buffer_get_cs_general(
      cmd, after_gfx ? &cmd->current_cs.post_gfx : &cmd->current_cs.cs, true);
   if (!cs)
      return;

   /* TODO: Generalize this mechanism suitably */
   if (after_gfx) {
      struct libagx_imm_write imm = {.address = address, .value = value};

      if (!cs->imm_writes.data) {
         cs->imm_writes = UTIL_DYNARRAY_INIT;
      }

      util_dynarray_append(&cs->imm_writes, imm);
      return;
   }

   /* As soon as we mark a query available, it needs to be available system
    * wide, otherwise a CPU-side get result can query. As such, we cache flush
    * before and then let coherency works its magic. Without this barrier, we
    * get flakes in
    *
    * dEQP-VK.query_pool.occlusion_query.get_results_conservative_size_64_wait_query_without_availability_draw_triangles_discard
    */
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   brl_cdm_cache_flush(dev, cs);

   perf_debug(cmd, "Queued write");
   if (agx_chip_native_agx3(dev->dev.chip)) {
      brl_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs, address,
                             sizeof(value), value);
      return;
   }

   libagx_write_u32(cmd, agx_1d(1), AGX_BARRIER_ALL, address, value);
}

/* Publish @value after the pre-graphics helpers already recorded for the
 * current render, in the same stream. */
static void
brl_queue_write_pre_gfx(struct brl_cmd_buffer *cmd, uint64_t address,
                       uint32_t value)
{
   struct brl_cs *cs =
      brl_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
   if (!cs)
      return;

   /* Make the producers' counters visible before the availability word, as
    * brl_queue_write() does. */
   brl_cdm_cache_flush(brl_cmd_buffer_device(cmd), cs);
   libagx_write_u32(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, address,
                    value);
}

static void
brl_queue_query_available_write(struct brl_cmd_buffer *cmd,
                               struct brl_query_pool *pool, uint32_t query,
                               uint32_t value, bool after_gfx)
{
   brl_queue_write(cmd, brl_query_available_addr(pool, query), value, after_gfx);
   if (!pool->handle || vk_command_buffer_has_error(&cmd->vk))
      return;

   const uint64_t offset_B = brl_query_available_offset(pool, query);
   const uint64_t size_B = brl_query_available_size(pool);
   if (after_gfx) {
      brl_cmd_note_imm_write_query_range(cmd, cmd->current_cs.post_gfx,
                                        pool->handle, offset_B, size_B);
   } else {
      brl_cmd_note_query_range(cmd, cmd->current_cs.cs, pool->handle, offset_B,
                              size_B);
   }
}

static void
brl_cmd_end_query_stamped(struct brl_cmd_buffer *cmd, struct brl_query_pool *pool,
                         uint32_t query)
{
   struct brl_cs *render = cmd->current_cs.gfx;
   const uint64_t offset_B = brl_query_available_offset(pool, query);

   /* Fast path: stage up to 3 availability writes inline on the active render
    * stream so single-render command buffers never allocate and discard a
    * post_gfx CDM stream and its 64 KiB control-stream root. */
   if (!cmd->current_cs.post_gfx &&
       (!cmd->state.oq_pending_render ||
        cmd->state.oq_pending_render == render) &&
       render->oq_pending_count < ARRAY_SIZE(render->oq_pending) &&
       offset_B <= UINT32_MAX) {
      render->oq_pending[render->oq_pending_count++] = (struct brl_oq_pending){
         .avail_addr = brl_query_available_addr(pool, query),
         .handle = pool->handle,
         .offset_B = (uint32_t)offset_B,
         .avail_size = (uint8_t)brl_query_available_size(pool),
      };
      cmd->state.oq_pending_render = render;
      return;
   }

   brl_flush_oq_pending(cmd);

   const uint32_t render_seq = render->sync.seq;
   brl_queue_query_available_write(cmd, pool, query, 1, true);

   struct brl_cs *cs = cmd->current_cs.post_gfx;
   if (!cs || vk_command_buffer_has_error(&cmd->vk))
      return;

   if (offset_B > UINT32_MAX)
      return;

   if (!cs->oq_stamps.data)
      cs->oq_stamps = UTIL_DYNARRAY_INIT;

   const struct brl_oq_stamp stamp = {
      .handle = pool->handle,
      .offset_B = (uint32_t)offset_B,
      .render_seq = render_seq,
   };
   util_dynarray_append(&cs->oq_stamps, stamp);
}

/* Mark the stream that received a query copy or reset (query_access), and
 * record the words it touches in a kernel-bound timestamp buffer so
 * submission can still set DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS when none of
 * them has a host publication pending. Pools without a binding have no such
 * words. */
static void
brl_note_query_access(struct brl_cmd_buffer *cmd, struct brl_query_pool *pool,
                     uint32_t first_index, uint32_t num_queries)
{
   struct brl_cs *cs = cmd->current_cs.cs;
   if (!cs)
      return;

   cs->query_access = true;
   if (!pool->handle || !num_queries || !brl_cmd_tracks_query_ranges(cmd))
      return;

   if (brl_has_available(pool)) {
      brl_cs_note_query_range(
         cs, pool->handle, brl_query_available_offset(pool, first_index),
         (uint64_t)num_queries * brl_query_available_size(pool));
   } else if (pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP) {
      brl_cs_note_query_range(cs, pool->handle,
                             brl_query_offset(pool, first_index),
                             (uint64_t)num_queries * pool->query_stride);
   } else {
      cs->query_access_untracked = true;
   }
}

static void
emit_zero_queries_g17p(struct brl_cmd_buffer *cmd, struct brl_query_pool *pool,
                       uint32_t first_index, uint32_t num_queries,
                       bool set_available)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);

   if (first_index > pool->vk.query_count ||
       num_queries > pool->vk.query_count - first_index) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (num_queries == 0)
      return;

   if (pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP &&
       pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION &&
       pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS &&
       pool->vk.query_type != VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT &&
       pool->vk.query_type != VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT &&
       pool->vk.query_type != VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT &&
       !brl_is_acceleration_structure_query(pool->vk.query_type)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const uint32_t reports = brl_reports_per_query(pool);
   if (!reports || pool->query_stride !=
                      (uint64_t)reports * sizeof(struct brl_query_report) ||
       (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION &&
        pool->oq_queries != pool->vk.query_count)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* Use the same indexed batch as AGX2. The hardware counter heap is not
    * necessarily contiguous, so the kernel follows the pool's u16 remap.
    * Publishing availability is a separate operation when synthesizing
    * multiview results: the helper's strong barrier makes every zero report
    * visible before the following availability store. */
   const bool publish_available = set_available && brl_has_available(pool);
   struct libagx_reset_query_args info = {
      .availability = brl_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,
      .first_query = first_index,
      .reports_per_query = reports,
      .set_available = set_available && !publish_available,
      .availability_shift = pool->avail_shift,
   };
   struct brl_cs *cs = cmd->current_cs.cs;
   assert(set_available || cs);
   if (!set_available && !cs->query_reset_pending && cs->stats.cmds) {
      /* Earlier copies may still be reading this pool. Only consecutive
       * resets can omit this dependency. */
      brl_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      brl_cdm_cache_flush(dev, cs);
   }

   libagx_reset_query_struct(cmd, agx_1d(num_queries),
                             set_available ? AGX_BARRIER_ALL : AGX_BARRIER_NONE,
                             info);
   brl_note_query_access(cmd, pool, first_index, num_queries);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   cs = cmd->current_cs.cs;
   assert(cs);
   if (!set_available) {
      cs->query_reset_pending = true;
      cs->stats.query_reset_dispatches++;
      cs->stats.query_reset_queries += num_queries;
   }
   if (brl_has_available(pool)) {
      brl_drm_attachment_set_add(&cs->compute_attachments,
         brl_query_available_addr(pool, first_index),
         (uint64_t)num_queries * brl_query_available_size(pool));
   }
   if (pool->oq_queries) {
      /* A later-recorded command may assign a counter before this reset runs. */
      brl_drm_attachment_set_add(&cs->compute_attachments,
         dev->occlusion_queries.bo->va->addr,
         (uint64_t)AGX_MAX_OCCLUSION_QUERIES * sizeof(uint64_t));
   } else {
      brl_drm_attachment_set_add(&cs->compute_attachments,
         brl_query_report_addr(dev, pool, first_index),
         (uint64_t)num_queries * pool->query_stride);
   }

   if (publish_available) {
      brl_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs,
         brl_query_available_addr(pool, first_index),
         (uint64_t)num_queries * brl_query_available_size(pool), 1);
      if (pool->handle && !vk_command_buffer_has_error(&cmd->vk)) {
         brl_cmd_note_query_range(
            cmd, cmd->current_cs.cs, pool->handle,
            brl_query_available_offset(pool, first_index),
            (uint64_t)num_queries * brl_query_available_size(pool));
      }
   }
}

static void
emit_zero_queries(struct brl_cmd_buffer *cmd, struct brl_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries,
                  bool set_available)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   perf_debug(cmd, "Query pool zero");

   if (agx_chip_native_agx3(dev->dev.chip)) {
      emit_zero_queries_g17p(cmd, pool, first_index, num_queries,
                             set_available);
      return;
   }

   struct libagx_reset_query_args info = {
      .availability = brl_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,

      .first_query = first_index,
      .reports_per_query = brl_reports_per_query(pool),
      .set_available = set_available,
      .availability_shift = pool->avail_shift,
   };

   libagx_reset_query_struct(cmd, agx_1d(num_queries), AGX_BARRIER_ALL, info);
   brl_note_query_access(cmd, pool, first_index, num_queries);
}

static void
host_zero_queries(struct brl_device *dev, struct brl_query_pool *pool,
                  uint32_t first_index, uint32_t num_queries,
                  bool set_available)
{
   if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
      const struct brl_pipeline_statistics_layout layout = {
         .query_count = pool->vk.query_count,
         .reports_per_query = brl_reports_per_query(pool),
         .query_start_B = pool->query_start,
         .query_stride_B = pool->query_stride,
         .size_B = pool->query_start +
                   ((uint64_t)pool->vk.query_count * pool->query_stride),
      };
      brl_pipeline_statistics_reset_host(agx_bo_map(pool->bo), &layout,
                                        first_index, num_queries,
                                        set_available);
      return;
   }

   for (uint32_t i = 0; i < num_queries; i++) {
      struct brl_query_report *reports =
         brl_query_report_map(dev, pool, first_index + i);

      uint64_t value = 0;
      if (brl_has_available(pool)) {
         brl_query_set_available_host(pool, first_index + i, set_available);
      } else {
         value = set_available ? 0 : LIBAGX_QUERY_UNAVAILABLE;
      }

      if (reports) {
         for (unsigned j = 0; j < brl_reports_per_query(pool); ++j)
            reports[j].value = value;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
brl_CreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(brl_device, dev, device);
   struct brl_query_pool *pool;

   bool occlusion = pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION;
   bool timestamp = pCreateInfo->queryType == VK_QUERY_TYPE_TIMESTAMP;
   unsigned occlusion_queries = occlusion ? pCreateInfo->queryCount : 0;

   pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Occlusion availability lives in 64-bit slots of a bound timestamp
    * buffer (brl_cmd_end_query_stamped()) on chips whose kernel writes them at
    * render retirement (brl_oq_render_stamps_supported(): G17P and M5). */
   const bool stamped =
      occlusion && brl_oq_render_stamps_supported(dev->dev.chip);
   pool->avail_shift = stamped ? 1 : 0;

   /* We place the availability first and then data. Pipeline-statistics pools
    * use the independently tested 64-bit layout path so large query counts
    * cannot wrap a 32-bit BO size. */
   size_t bo_size = 0;
   if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
      struct brl_pipeline_statistics_layout layout;
      if (!brl_pipeline_statistics_layout_init(
             pool->vk.query_count, pool->vk.pipeline_statistics, &layout) ||
          layout.size_B > SIZE_MAX) {
         vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      pool->query_start = layout.query_start_B;
      pool->query_stride = layout.query_stride_B;
      bo_size = layout.size_B;
   } else {
      pool->query_start = 0;
      if (brl_has_available(pool)) {
         pool->query_start =
            align64((uint64_t)pool->vk.query_count *
                       brl_query_available_size(pool),
                    sizeof(struct brl_query_report));
      }

      const uint32_t reports_per_query = brl_reports_per_query(pool);
      pool->query_stride =
         (uint64_t)reports_per_query * sizeof(struct brl_query_report);
   }

   if (pool->vk.query_count > 0) {
      if (pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS) {
         bo_size = pool->query_start;

         /* For occlusion queries, we stick the query index remapping here */
         if (occlusion_queries)
            bo_size += sizeof(uint16_t) * (size_t)pool->vk.query_count;
         else
            bo_size += pool->query_stride * (size_t)pool->vk.query_count;
      }

      /* Kernel timestamp requests carry a 32-bit byte offset. Validate the
       * addressable range up front so a created timestamp pool never fails
       * recording later, and fall back to queued availability stores for
       * oversized occlusion pools. */
      if (timestamp &&
          brl_query_offset(pool, pool->vk.query_count - 1) > UINT32_MAX) {
         vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      const bool can_bind_stamped =
         stamped &&
         brl_query_available_offset(pool, pool->vk.query_count - 1) <=
            UINT32_MAX;

      /* The kernel requires that timestamp buffers are SHARED */
      enum agx_bo_flags flags = AGX_BO_WRITEBACK;
      if (timestamp || can_bind_stamped)
         flags |= AGX_BO_SHARED;

      pool->bo = agx_bo_create(&dev->dev, bo_size, 0, flags, "Query pool");
      if (!pool->bo || !agx_bo_map(pool->bo)) {
         brl_DestroyQueryPool(device, brl_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      /* Timestamp buffers must be explicitly bound as such before we can use
       * them.
       */
      if (timestamp) {
         int ret = agx_bind_timestamps(&dev->dev, pool->bo, &pool->handle);
         if (ret) {
            brl_DestroyQueryPool(device, brl_query_pool_to_handle(pool),
                                pAllocator);
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         }

         assert(pool->handle && "handles are nonzero");
         brl_frame_anatomy_bind_pool(dev, pool->handle, agx_bo_map(pool->bo),
                                    pool->bo->size);
      }

      /* Without a binding, every query end keeps the queued store. */
      if (can_bind_stamped &&
          agx_bind_timestamps(&dev->dev, pool->bo, &pool->handle))
         pool->handle = 0;
   }

   uint16_t *oq_index = brl_pool_oq_index_ptr(pool);
   assert(oq_index || !occlusion_queries);

   /* See brl_oq_submission_done(): only the native AGX3 path needs the
    * submission point. */
   if (occlusion_queries && agx_chip_native_agx3(dev->dev.chip)) {
      pool->oq_submit =
         vk_zalloc2(&dev->vk.alloc, pAllocator,
                    sizeof(*pool->oq_submit) * occlusion_queries, 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!pool->oq_submit) {
         brl_DestroyQueryPool(device, brl_query_pool_to_handle(pool), pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   /* Applications may reserve far more queries than they use. Keep the pool's
    * logical indices independent of the finite hardware counter heap, and
    * assign a counter only when a command needs one. */
   if (occlusion_queries)
      memset(oq_index, 0xff, occlusion_queries * sizeof(*oq_index));
   pool->oq_queries = occlusion_queries;

   if (pCreateInfo->flags & VK_QUERY_POOL_CREATE_RESET_BIT_KHR)
      host_zero_queries(dev, pool, 0, pool->vk.query_count, false);

   *pQueryPool = brl_query_pool_to_handle(pool);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
brl_DestroyQueryPool(VkDevice device, VkQueryPool queryPool,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(brl_device, dev, device);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);

   if (!pool)
      return;

   uint16_t *oq_index = brl_pool_oq_index_ptr(pool);

   for (unsigned i = 0; oq_index && i < pool->oq_queries; ++i) {
      if (oq_index[i] != LIBAGX_QUERY_INDEX_UNUSED)
         brl_descriptor_table_remove(dev, &dev->occlusion_queries, oq_index[i]);
   }

   if (pool->handle) {
      brl_query_stamps_unbind(dev, pool->handle);
      brl_frame_anatomy_unbind_pool(dev, pool->handle);
      dev->dev.ops.bo_unbind_object(&dev->dev, pool->handle);
   }

   agx_bo_unreference(&dev->dev, pool->bo);
   vk_free2(&dev->vk.alloc, pAllocator, pool->oq_submit);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

VKAPI_ATTR void VKAPI_CALL
brl_ResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                  uint32_t queryCount)
{
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);
   VK_FROM_HANDLE(brl_device, dev, device);

   host_zero_queries(dev, pool, firstQuery, queryCount, false);

   for (uint32_t i = 0; pool->oq_submit && i < queryCount; i++)
      p_atomic_set(&pool->oq_submit[firstQuery + i].point, 0);
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                     uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);

   perf_debug(cmd, "Reset query pool");

   /* The reset runs after every earlier use of the pool and before every
    * later one: it is a transfer operation (Vulkan 1.3, 17.1) whose result
    * every later query command on the queue observes without further
    * synchronization. Zeroing is compute on every generation, so the
    * ordering is one compute stream everything after waits on. */
   if (!queryCount)
      return;

   cmd->sync.kind_override = BRL_SYNC_ALL;
   cmd->state.query_reset_recording = true;
   struct brl_cs *cs = brl_query_command_cs(cmd);
   if (cs) {
      cmd->sync.has_gpu_query_reset = true;
      emit_zero_queries(cmd, pool, firstQuery, queryCount, false);
   }
   cmd->state.query_reset_recording = false;
   cmd->sync.kind_override = 0;
   if (!cs)
      return;

   brl_sync_after_stream(cmd, cmd->current_cs.cs ? cmd->current_cs.cs : cs,
                        BRL_SYNC_ALL);
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdWriteTimestamp2(VkCommandBuffer commandBuffer,
                      VkPipelineStageFlags2 stage, VkQueryPool queryPool,
                      uint32_t query)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   brl_m5_pass_merge_end_ordered(cmd, "timestamp");
   brl_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   if (unlikely(dev->frame_anatomy)) {
      brl_frame_anatomy_note(dev, "timestamp stage=0x%llx query=%u in_pass=%u",
                            (unsigned long long)stage, query,
                            cmd->state.gfx.render.tilebuffer.nr_samples != 0 &&
                               cmd->current_cs.gfx != NULL);
   }

   const uint64_t offset_B = brl_query_offset(pool, query);
   assert(!pool->handle || offset_B <= UINT32_MAX);
   const struct agx_timestamp_req req = {
      .addr = brl_query_report_addr(dev, pool, query),
      .handle = pool->handle,
      .offset_B = (uint32_t)offset_B,
   };

   /* The stamp rides on a command's native start or end timestamp; which
    * command, and what it must wait on, follows from the stage. */
   brl_sync_write_timestamp(cmd, stage, req);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   /* From the Vulkan spec:
    *
    *   "If vkCmdWriteTimestamp2 is called while executing a render pass
    *    instance that has multiview enabled, the timestamp uses N consecutive
    *    query indices in the query pool (starting at query) where N is the
    *    number of bits set in the view mask of the subpass the command is
    *    executed in. The resulting query values are determined by an
    *    implementation-dependent choice of one of the following behaviors:"
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries, as described in the first behavior listed in the
    * Vulkan spec:
    *
    *   "The first query is a timestamp value and (if more than one bit is set
    *   in the view mask) zero is written to the remaining queries."
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1)
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1, true);
   }
}

static VkResult
brl_ensure_occlusion_query(struct brl_device *dev, struct brl_query_pool *pool,
                          uint32_t query)
{
   uint16_t *entry = &brl_pool_oq_index_ptr(pool)[query];
   if (p_atomic_read(entry) != LIBAGX_QUERY_INDEX_UNUSED)
      return VK_SUCCESS;

   const uint64_t zero = 0;
   unsigned index;
   VkResult result = brl_descriptor_table_add(
      dev, &dev->occlusion_queries, &zero, sizeof(zero), &index);
   if (result != VK_SUCCESS)
      return result;

   /* Different command buffers may record the same query concurrently. Publish
    * one initialized counter and return any redundant allocation to the heap. */
   uint16_t previous =
      p_atomic_cmpxchg(entry, LIBAGX_QUERY_INDEX_UNUSED, (uint16_t)index);
   if (previous != LIBAGX_QUERY_INDEX_UNUSED)
      brl_descriptor_table_remove(dev, &dev->occlusion_queries, index);
   return VK_SUCCESS;
}

bool
brl_ensure_occlusion_query_for_draw(struct brl_cmd_buffer *cmd)
{
   struct brl_graphics_state *gfx = &cmd->state.gfx;
   struct brl_query_pool *pool = gfx->occlusion.pending_pool;
   if (!pool || gfx->occlusion.mode == AGX_VISIBILITY_MODE_NONE)
      return true;

   uint32_t query = gfx->occlusion.pending_query;
   VkResult result =
      brl_ensure_occlusion_query(brl_cmd_buffer_device(cmd), pool, query);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return false;
   }

   gfx->occlusion.index = brl_pool_oq_index_ptr(pool)[query];
   gfx->occlusion.pending_pool = NULL;
   gfx->dirty |= BRL_DIRTY_OCCLUSION;
   return true;
}

static void
brl_cmd_begin_end_query(struct brl_cmd_buffer *cmd, struct brl_query_pool *pool,
                       uint32_t query, uint32_t index,
                       VkQueryControlFlags flags, bool end)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   bool graphics = false, pre_gfx = false;

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      assert(query < pool->oq_queries);
      if (agx_chip_g17_m5(dev->dev.chip)) {
         /* A reset precedes each legal query use and zeros any assigned
          * counter at execution time. An empty Begin/End pair therefore
          * only needs to publish availability, without reserving a counter. */
         cmd->state.gfx.occlusion.pending_pool = end ? NULL : pool;
         cmd->state.gfx.occlusion.pending_query = query;
      } else {
         VkResult result = brl_ensure_occlusion_query(dev, pool, query);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
         uint16_t *oq_index = brl_pool_oq_index_ptr(pool);
         cmd->state.gfx.occlusion.index = oq_index[query];
      }

      if (end) {
         cmd->state.gfx.occlusion.mode = AGX_VISIBILITY_MODE_NONE;
      } else {
         cmd->state.gfx.occlusion.mode = flags & VK_QUERY_CONTROL_PRECISE_BIT
                                            ? AGX_VISIBILITY_MODE_COUNTING
                                            : AGX_VISIBILITY_MODE_BOOLEAN;
         cmd->state.gfx.occlusion.begin_render_seq =
            cmd->current_cs.gfx ? cmd->current_cs.gfx->sync.seq : 0;
      }

      cmd->state.gfx.dirty |= BRL_DIRTY_OCCLUSION;
      if (end && pool->oq_submit) {
         const struct brl_oq_end oq_end = {.pool = pool, .query = query};
         util_dynarray_append(&cmd->oq_ends, oq_end);
      }
      graphics = true;
      break;
   }

   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT: {
      uint64_t addr = brl_query_report_addr(dev, pool, query);
      cmd->state.gfx.xfb_query[index] = end ? 0 : addr;

      /* The counters are written by the geometry helpers in the
       * pre-graphics stream of the render that draws. The general compute
       * stream is not ordered after that stream: it stays open across
       * render begins and need not wait on later pre-graphics work. On M5,
       * publish availability after the helpers in the same pre-graphics
       * stream, or after the graphics work when the current render has no
       * such stream (or there is no current render). */
      if (agx_chip_g17_m5(dev->dev.chip)) {
         if (cmd->current_cs.gfx && cmd->current_cs.pre_gfx)
            pre_gfx = true;
         else
            graphics = true;
      }
      break;
   }

   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT: {
      /* Counted like the transform feedback query's primitives-needed
       * report: in the pre-graphics helpers of each draw. */
      uint64_t addr = brl_query_report_addr(dev, pool, query);
      cmd->state.gfx.prims_generated_query[index] = end ? 0 : addr;
      /* As for the pipeline statistics written by the same helpers, publish
       * availability after the graphics work. */
      graphics = true;
      break;
   }

   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT: {
      brl_root_mark_dirty(&cmd->state.gfx.descriptors);
      cmd->state.gfx.descriptors.root.mesh_prims_query =
         end ? 0 : brl_query_report_addr(dev, pool, query);
      graphics = true;
      break;
   }

   case VK_QUERY_TYPE_PIPELINE_STATISTICS: {
      struct brl_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
      brl_root_mark_dirty(&cmd->state.gfx.descriptors);

      root->draw.pipeline_stats =
         end ? 0 : brl_query_report_addr(dev, pool, query);
      root->draw.pipeline_stats_flags =
         end ? 0 : pool->vk.pipeline_statistics;

      /* IA/VS/clip counters may be written by the pre-graphics helper and FS
       * counters by the VDM workload itself, so any graphics statistic makes
       * availability a post-graphics write. A compute-only query increments
       * and publishes availability in the same ordered general CDM stream.
       * Mixed queries also select post-graphics; when no VDM is live that
       * stream is appended after the current compute stream. */
      graphics = pool->vk.pipeline_statistics &
                 ~VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
      break;
   }

   default:
      UNREACHABLE("Unsupported query type");
   }

   /* We need to set available=1 after the graphics work finishes. */
   if (end) {
      perf_debug(cmd, "Query ending, type %u", pool->vk.query_type);
      /* Renders of one queue may finish out of order: a query that spans
       * renders keeps the store, which waits on every render before it.
       * Only native AGX3 occlusion pools have a bound availability buffer
       * (pool->handle, see brl_CreateQueryPool()); M5 transform feedback
       * queries publish after their pre-graphics helpers. */
      if (pre_gfx) {
         brl_queue_write_pre_gfx(cmd, brl_query_available_addr(pool, query), 1);
         if (pool->handle && !vk_command_buffer_has_error(&cmd->vk)) {
            brl_cmd_note_query_range(cmd, cmd->current_cs.pre_gfx,
                                    pool->handle,
                                    brl_query_available_offset(pool, query),
                                    brl_query_available_size(pool));
         }
      } else if (pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION &&
                 pool->handle && cmd->current_cs.gfx &&
                 cmd->current_cs.gfx->sync.seq ==
                    cmd->state.gfx.occlusion.begin_render_seq) {
         brl_cmd_end_query_stamped(cmd, pool, query);
      } else {
         brl_queue_query_available_write(cmd, pool, query, 1, graphics);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdBeginQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t query, VkQueryControlFlags flags,
                           uint32_t index)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   brl_m5_pass_merge_end_ordered(cmd, "query");
   brl_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);

   brl_cmd_begin_end_query(cmd, pool, query, index, flags, false);
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdEndQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                         uint32_t query, uint32_t index)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   brl_m5_pass_merge_end_ordered(cmd, "query");
   brl_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);

   brl_cmd_begin_end_query(cmd, pool, query, index, 0, true);

   /* From the Vulkan spec:
    *
    *   "If queries are used while executing a render pass instance that has
    *    multiview enabled, the query uses N consecutive query indices in
    *    the query pool (starting at query) where N is the number of bits set
    *    in the view mask in the subpass the query is used in. How the
    *    numerical results of the query are distributed among the queries is
    *    implementation-dependent."
    *
    * In our case, only the first query is used, so we emit zeros for the
    * remaining queries.
    */
   if (cmd->state.gfx.render.view_mask != 0) {
      const uint32_t num_queries =
         util_bitcount(cmd->state.gfx.render.view_mask);
      if (num_queries > 1) {
         perf_debug(cmd, "Multiview query zeroing");
         emit_zero_queries(cmd, pool, query + 1, num_queries - 1, true);
      }
   }
}

static bool
brl_query_is_available(struct brl_device *dev, struct brl_query_pool *pool,
                      uint32_t query)
{
   if (brl_has_available(pool)) {
      return brl_query_available_value(pool, query) != 0;
   } else {
      const struct brl_query_report *report =
         brl_query_report_map(dev, pool, query);

      return report->value != LIBAGX_QUERY_UNAVAILABLE;
   }
}

#define BRL_QUERY_TIMEOUT 2000000000ull

void
brl_query_note_submit(struct brl_cmd_buffer *cmd, struct brl_queue *queue,
                     uint64_t point)
{
   util_dynarray_foreach(&cmd->oq_ends, struct brl_oq_end, e) {
      struct brl_oq_submit *s = &e->pool->oq_submit[e->query];
      /* Readers check the point again after the queue: clear it first so
       * they never pair this queue with the previous point. */
      p_atomic_set(&s->point, 0);
      p_atomic_set(&s->queue, queue);
      p_atomic_set(&s->point, point);
   }
}

struct brl_pending_query_stamp {
   uint32_t handle;
   uint64_t offset_B;
   struct brl_queue *queue;
   uint64_t point;
};

void
brl_query_stamps_init(struct brl_device *dev)
{
   simple_mtx_init(&dev->query_stamps.lock, mtx_plain);
   dev->query_stamps.pending = UTIL_DYNARRAY_INIT;
   dev->query_stamps.slot_index = NULL;
   dev->query_stamps.slot_cap = 0;
   dev->query_stamps.prune_threshold = 64;
   dev->query_stamps.oom = false;
   dev->query_stamps.ready = true;
}

void
brl_query_stamps_finish(struct brl_device *dev)
{
   if (!dev->query_stamps.ready)
      return;
   free(dev->query_stamps.slot_index);
   dev->query_stamps.slot_index = NULL;
   dev->query_stamps.slot_cap = 0;
   util_dynarray_fini(&dev->query_stamps.pending);
   simple_mtx_destroy(&dev->query_stamps.lock);
   dev->query_stamps.ready = false;
}

static uint64_t
brl_query_stamp_hash(uint32_t handle, uint64_t offset_B,
                    const struct brl_queue *queue)
{
   uint64_t h = (uint64_t)handle * UINT64_C(0x9e3779b97f4a7c15);
   h ^= offset_B + UINT64_C(0xbf58476d1ce4e5b9) + (h << 6) + (h >> 2);
   h ^= (uint64_t)(uintptr_t)queue + UINT64_C(0x94d049bb133111eb) +
        (h << 6) + (h >> 2);
   h ^= h >> 30;
   h *= UINT64_C(0xbf58476d1ce4e5b9);
   h ^= h >> 27;
   h *= UINT64_C(0x94d049bb133111eb);
   h ^= h >> 31;
   return h;
}

static bool
brl_query_stamps_rebuild_index_locked(struct brl_device *dev, size_t min_entries)
{
   const struct brl_pending_query_stamp *entries =
      dev->query_stamps.pending.data;
   const size_t count = util_dynarray_num_elements(
      &dev->query_stamps.pending, struct brl_pending_query_stamp);
   const size_t target = MAX2(count, min_entries);

   if (target == 0) {
      if (dev->query_stamps.slot_index) {
         if (dev->query_stamps.slot_cap > 1024) {
            free(dev->query_stamps.slot_index);
            dev->query_stamps.slot_index = NULL;
            dev->query_stamps.slot_cap = 0;
         } else {
            memset(dev->query_stamps.slot_index, 0,
                   dev->query_stamps.slot_cap * sizeof(uint32_t));
         }
      }
      return true;
   }

   if (count >= UINT32_MAX || target > SIZE_MAX / 2) {
      dev->query_stamps.oom = true;
      return false;
   }

   size_t want_cap = 64;
   while (want_cap < target * 2) {
      if (want_cap > SIZE_MAX / 2) {
         dev->query_stamps.oom = true;
         return false;
      }
      want_cap *= 2;
   }

   if (want_cap > dev->query_stamps.slot_cap) {
      if (want_cap > SIZE_MAX / sizeof(uint32_t)) {
         dev->query_stamps.oom = true;
         return false;
      }
      uint32_t *new_index = calloc(want_cap, sizeof(uint32_t));
      if (!new_index) {
         dev->query_stamps.oom = true;
         return false;
      }
      free(dev->query_stamps.slot_index);
      dev->query_stamps.slot_index = new_index;
      dev->query_stamps.slot_cap = want_cap;
   } else if (dev->query_stamps.slot_cap > 1024 &&
              want_cap * 4 <= dev->query_stamps.slot_cap) {
      uint32_t *shrunk = calloc(want_cap, sizeof(uint32_t));
      if (shrunk) {
         free(dev->query_stamps.slot_index);
         dev->query_stamps.slot_index = shrunk;
         dev->query_stamps.slot_cap = want_cap;
      } else {
         memset(dev->query_stamps.slot_index, 0,
                dev->query_stamps.slot_cap * sizeof(uint32_t));
      }
   } else {
      memset(dev->query_stamps.slot_index, 0,
             dev->query_stamps.slot_cap * sizeof(uint32_t));
   }

   const size_t mask = dev->query_stamps.slot_cap - 1;
   uint32_t *index = dev->query_stamps.slot_index;
   for (size_t i = 0; i < count; ++i) {
      size_t pos = (size_t)brl_query_stamp_hash(
         entries[i].handle, entries[i].offset_B, entries[i].queue) & mask;
      while (index[pos] != 0)
         pos = (pos + 1) & mask;
      index[pos] = (uint32_t)(i + 1);
   }
   return true;
}

static struct brl_pending_query_stamp *
brl_query_stamps_find_locked(struct brl_device *dev, struct brl_queue *queue,
                            uint32_t handle, uint64_t offset_B)
{
   if (!dev->query_stamps.slot_cap)
      return NULL;

   struct brl_pending_query_stamp *entries = dev->query_stamps.pending.data;
   const size_t mask = dev->query_stamps.slot_cap - 1;
   const uint32_t *index = dev->query_stamps.slot_index;
   size_t pos = (size_t)brl_query_stamp_hash(handle, offset_B, queue) & mask;
   for (;;) {
      const uint32_t slot = index[pos];
      if (!slot)
         return NULL;
      struct brl_pending_query_stamp *e = &entries[slot - 1];
      if (e->handle == handle && e->offset_B == offset_B && e->queue == queue)
         return e;
      pos = (pos + 1) & mask;
   }
}

/* A stamp is published once its queue point is reached: the kernel writes a
 * command's timestamps before it signals the command's fence, and the queue
 * syncobj point waits for every command of its submission. */
static bool
brl_query_stamp_done_locked(struct brl_device *dev,
                           const struct brl_pending_query_stamp *e,
                           bool allow_query)
{
   if (e->point <= p_atomic_read(&e->queue->drm.signalled_value))
      return true;
   if (!allow_query)
      return false;

   uint64_t reached = 0;
   if (drmSyncobjQuery(dev->dev.fd, &e->queue->drm.syncobj, &reached, 1) == 0) {
      brl_queue_note_signalled(e->queue, reached);
      if (e->point <= reached)
         return true;
   }
   return false;
}

struct brl_query_queue_set {
   struct brl_queue *inline_buf[32];
   struct brl_queue **items;
   size_t count;
   size_t cap;
   bool oom;
};

static void
brl_query_queue_set_init(struct brl_query_queue_set *set)
{
   set->items = set->inline_buf;
   set->count = 0;
   set->cap = ARRAY_SIZE(set->inline_buf);
   set->oom = false;
}

static void
brl_query_queue_set_fini(struct brl_query_queue_set *set)
{
   if (set->items != set->inline_buf)
      free(set->items);
}

static bool
brl_query_queue_set_take(struct brl_query_queue_set *set, struct brl_queue *queue)
{
   for (size_t q = 0; q < set->count; ++q) {
      if (set->items[q] == queue)
         return false;
   }
   if (set->oom)
      return false;
   if (set->count == set->cap) {
      if (set->cap > SIZE_MAX / 2 / sizeof(*set->items)) {
         set->oom = true;
         return false;
      }
      const size_t new_cap = set->cap * 2;
      struct brl_queue **new_items = malloc(new_cap * sizeof(*new_items));
      if (!new_items) {
         set->oom = true;
         return false;
      }
      memcpy(new_items, set->items, set->count * sizeof(*new_items));
      if (set->items != set->inline_buf)
         free(set->items);
      set->items = new_items;
      set->cap = new_cap;
   }
   set->items[set->count++] = queue;
   return true;
}

static void
brl_query_stamps_prune_locked(struct brl_device *dev, bool query_queues,
                             struct brl_queue *skip_queue, uint64_t skip_point)
{
   struct brl_pending_query_stamp *entries = dev->query_stamps.pending.data;
   const size_t old_count = util_dynarray_num_elements(
      &dev->query_stamps.pending, struct brl_pending_query_stamp);
   size_t count = old_count;
   struct brl_query_queue_set queried;
   brl_query_queue_set_init(&queried);

   size_t i = 0;
   while (i < count) {
      bool allow_query = false;
      if (query_queues &&
          !(entries[i].queue == skip_queue && entries[i].point == skip_point) &&
          entries[i].point >
             p_atomic_read(&entries[i].queue->drm.signalled_value)) {
         allow_query = brl_query_queue_set_take(&queried, entries[i].queue);
      }

      if (brl_query_stamp_done_locked(dev, &entries[i], allow_query)) {
         entries[i] = entries[--count];
      } else {
         i++;
      }
   }
   brl_query_queue_set_fini(&queried);

   dev->query_stamps.pending.size =
      count * sizeof(struct brl_pending_query_stamp);
   dev->query_stamps.prune_threshold = MAX2(64u, count * 2);
   if (count != old_count)
      brl_query_stamps_rebuild_index_locked(dev, count);
}

/* Drop the pending stamps of a destroyed pool (handle) or of a finished
 * queue: neither can be published to, or observed through, again. */
static void
brl_query_stamps_drop(struct brl_device *dev, uint32_t handle,
                     const struct brl_queue *queue)
{
   if (!dev->query_stamps.ready)
      return;

   simple_mtx_lock(&dev->query_stamps.lock);
   struct brl_pending_query_stamp *entries = dev->query_stamps.pending.data;
   const size_t old_count = util_dynarray_num_elements(
      &dev->query_stamps.pending, struct brl_pending_query_stamp);
   size_t count = old_count;
   size_t i = 0;
   while (i < count) {
      if ((handle && entries[i].handle == handle) ||
          (queue && entries[i].queue == queue))
         entries[i] = entries[--count];
      else
         i++;
   }
   dev->query_stamps.pending.size =
      count * sizeof(struct brl_pending_query_stamp);
   dev->query_stamps.prune_threshold =
      MIN2(dev->query_stamps.prune_threshold, MAX2(64u, count * 2));
   if (count != old_count)
      brl_query_stamps_rebuild_index_locked(dev, count);
   simple_mtx_unlock(&dev->query_stamps.lock);
}

void
brl_query_stamps_unbind(struct brl_device *dev, uint32_t handle)
{
   if (handle)
      brl_query_stamps_drop(dev, handle, NULL);
}

void
brl_query_stamps_clear_queue(struct brl_device *dev, struct brl_queue *queue)
{
   brl_query_stamps_drop(dev, 0, queue);
}

static bool
brl_query_stamp_overlaps_cs(const struct brl_pending_query_stamp *e,
                           const struct brl_cs *cs)
{
   const uint64_t stamp_start = e->offset_B;
   const uint64_t stamp_end = stamp_start + sizeof(uint64_t);

   util_dynarray_foreach(&cs->query_access_ranges, struct brl_query_access_range,
                         r) {
      if (r->handle == e->handle && stamp_start < r->end_B &&
          r->start_B < stamp_end)
         return true;
   }
   return false;
}

/* The conflict scan is linear: every pending stamp is compared with every
 * range of the stream, without the hash index. This caps that work at a fixed
 * number of comparisons under dev->query_stamps.lock. A stream whose scan
 * would exceed it fails closed: it gets no NO_QUERY_ACCESS, as if it
 * conflicted. */
#define BRL_QUERY_STAMPS_CONFLICT_BUDGET 4096u

/* Whether cs may touch a query word whose host publication is still pending,
 * so it must not get DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS. True whenever the
 * answer is unknown. A stream with no recorded words cannot conflict,
 * whether or not it copies or resets queries (query_access). */
bool
brl_query_stamps_conflict(struct brl_device *dev, struct brl_queue *queue,
                         uint64_t current_point, const struct brl_cs *cs)
{
   if (!(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS))
      return true;
   if (!dev->query_stamps.ready || cs->query_access_untracked)
      return true;
   if (!cs->query_access_ranges.size)
      return false;

   const size_t range_count = util_dynarray_num_elements(
      &cs->query_access_ranges, struct brl_query_access_range);

   simple_mtx_lock(&dev->query_stamps.lock);
   if (dev->query_stamps.oom) {
      simple_mtx_unlock(&dev->query_stamps.lock);
      return true;
   }

   struct brl_pending_query_stamp *entries = dev->query_stamps.pending.data;
   const size_t count = util_dynarray_num_elements(
      &dev->query_stamps.pending, struct brl_pending_query_stamp);
   if (count && range_count > BRL_QUERY_STAMPS_CONFLICT_BUDGET) {
      simple_mtx_unlock(&dev->query_stamps.lock);
      return true;
   }

   /* Keep conflict checks read-only on pending[] and slot_index[]: entry
    * retirement and O(N) hash-table rebuilds belong in the amortized prune
    * path so a budgeted conflict check stays strictly O(1) under the lock. */
   struct brl_query_queue_set queried;
   brl_query_queue_set_init(&queried);
   bool conflict = false;
   size_t work = 0;
   for (size_t i = 0; i < count; ++i) {
      if (work + range_count > BRL_QUERY_STAMPS_CONFLICT_BUDGET) {
         conflict = true;
         break;
      }
      work += range_count;
      if (!brl_query_stamp_overlaps_cs(&entries[i], cs))
         continue;
      if (entries[i].queue == queue && entries[i].point == current_point) {
         conflict = true;
         break;
      }
      bool allow_query = false;
      if (entries[i].point >
          p_atomic_read(&entries[i].queue->drm.signalled_value)) {
         allow_query = brl_query_queue_set_take(&queried, entries[i].queue);
      }
      if (!brl_query_stamp_done_locked(dev, &entries[i], allow_query)) {
         conflict = true;
         break;
      }
   }
   brl_query_queue_set_fini(&queried);
   simple_mtx_unlock(&dev->query_stamps.lock);
   return conflict;
}

static void
brl_query_stamps_note_one_locked(struct brl_device *dev, struct brl_queue *queue,
                                uint64_t point, uint32_t handle,
                                uint64_t offset_B)
{
   if (!handle || dev->query_stamps.oom)
      return;

   struct brl_pending_query_stamp *existing =
      brl_query_stamps_find_locked(dev, queue, handle, offset_B);
   if (existing) {
      if (point > existing->point)
         existing->point = point;
      return;
   }

   size_t count = util_dynarray_num_elements(&dev->query_stamps.pending,
                                             struct brl_pending_query_stamp);
   if (count >= dev->query_stamps.prune_threshold) {
      brl_query_stamps_prune_locked(dev, true, queue, point);
      count = util_dynarray_num_elements(&dev->query_stamps.pending,
                                         struct brl_pending_query_stamp);
   }

   if (count >= UINT32_MAX - 1) {
      dev->query_stamps.oom = true;
      return;
   }

   if (!dev->query_stamps.slot_cap ||
       (count + 1) * 2 > dev->query_stamps.slot_cap) {
      if (!brl_query_stamps_rebuild_index_locked(dev, count + 1))
         return;
   }

   struct brl_pending_query_stamp *slot = util_dynarray_grow_bytes(
      &dev->query_stamps.pending, 1, sizeof(*slot));
   if (!slot) {
      dev->query_stamps.oom = true;
      return;
   }

   *slot = (struct brl_pending_query_stamp){
      .handle = handle,
      .offset_B = offset_B,
      .queue = queue,
      .point = point,
   };

   const size_t mask = dev->query_stamps.slot_cap - 1;
   uint32_t *index = dev->query_stamps.slot_index;
   size_t pos = (size_t)brl_query_stamp_hash(handle, offset_B, queue) & mask;
   while (index[pos] != 0)
      pos = (pos + 1) & mask;
   index[pos] = (uint32_t)(count + 1);
}

void
brl_query_stamps_note_req(struct brl_device *dev, struct brl_queue *queue,
                         uint64_t point, uint32_t handle, uint32_t offset_B)
{
   if (!handle || !dev->query_stamps.ready ||
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS))
      return;

   simple_mtx_lock(&dev->query_stamps.lock);
   brl_query_stamps_note_one_locked(dev, queue, point, handle, offset_B);
   simple_mtx_unlock(&dev->query_stamps.lock);
}

void
brl_query_stamps_note_cs(struct brl_device *dev, struct brl_queue *queue,
                        uint64_t point, const struct brl_cs *cs)
{
   if (!dev->query_stamps.ready ||
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS))
      return;

   bool any = cs->timestamp.start.handle || cs->timestamp.end.handle;
   if (!any && cs->type == BRL_CS_VDM) {
      for (unsigned i = 0; i < ARRAY_SIZE(cs->oq_stamp); ++i) {
         if (cs->oq_stamp[i].handle) {
            any = true;
            break;
         }
      }
   }
   if (!any)
      return;

   simple_mtx_lock(&dev->query_stamps.lock);
   brl_query_stamps_note_one_locked(dev, queue, point,
                                   cs->timestamp.start.handle,
                                   cs->timestamp.start.offset_B);
   brl_query_stamps_note_one_locked(dev, queue, point,
                                   cs->timestamp.end.handle,
                                   cs->timestamp.end.offset_B);
   if (cs->type == BRL_CS_VDM) {
      for (unsigned i = 0; i < ARRAY_SIZE(cs->oq_stamp); ++i) {
         brl_query_stamps_note_one_locked(dev, queue, point,
                                         cs->oq_stamp[i].handle,
                                         cs->oq_stamp[i].offset_B);
      }
   }
   simple_mtx_unlock(&dev->query_stamps.lock);
}

/*
 * A render's occlusion counts reach memory with its end-of-render cache
 * maintenance. The availability word is written by a compute stream that
 * the firmware starts once the render's fragment work is done, which can be
 * before that maintenance: the CPU then reads availability 1 and a count of
 * 0 that is right shortly after. The kernel signals a submission only
 * after the maintenance of all its renders, so a CPU read of an occlusion
 * result also requires the submission that ended the query.
 *
 * This is the native AGX3 firmware path: A18 Pro and M4 Pro both return
 * early zero counts. G13 and G14 read correct counts as soon as the
 * availability word is set (0 of 51,200 reads wrong, where the A18 Pro
 * got 31), so they keep that cheaper and earlier check alone.
 */
static VkResult
brl_oq_submission_done(struct brl_device *dev, struct brl_query_pool *pool,
                      uint32_t query, bool wait, bool *done)
{
   *done = true;
   if (!pool->oq_submit)
      return VK_SUCCESS;

   struct brl_oq_submit *s = &pool->oq_submit[query];
   uint64_t point;
   struct brl_queue *queue;
   do {
      point = p_atomic_read(&s->point);
      queue = p_atomic_read(&s->queue);
   } while (point != p_atomic_read(&s->point));
   /* Timeline values only grow, so a point at or below one this queue has
    * been seen to reach is complete: one syncobj wait per submission, not
    * one per query read. */
   if (!point || point <= p_atomic_read(&queue->drm.signalled_value))
      return VK_SUCCESS;

   uint32_t syncobj = queue->drm.syncobj;

   /* Bounded waits so a lost device is noticed; a hung submission ends in
    * the kernel's own timeout. */
   for (;;) {
      int64_t timeout = wait ? (int64_t)os_time_get_absolute_timeout(100000000)
                             : 0;
      /* WAIT_FOR_SUBMIT: the point is noted just before the submit ioctl. */
      int ret = drmSyncobjTimelineWait(dev->dev.fd, &syncobj, &point, 1,
                                       timeout,
                                       DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                       NULL);
      if (ret == 0) {
         brl_queue_note_signalled(queue, point);
         return VK_SUCCESS;
      }
      if (ret != -ETIME)
         return vk_device_set_lost(&dev->vk,
                                   "occlusion query submission wait: %d", ret);
      if (!wait) {
         *done = false;
         return VK_SUCCESS;
      }

      VkResult status = vk_device_check_status(&dev->vk);
      if (status != VK_SUCCESS)
         return status;
   }
}

/* Whether every queue has completed all the work submitted to it: nothing
 * waits in a submit thread and each queue timeline reached its last point. */
static bool
brl_queues_idle(struct brl_device *dev)
{
   vk_foreach_queue(vk_queue, &dev->vk) {
      struct brl_queue *queue = container_of(vk_queue, struct brl_queue, vk);

      mtx_lock(&vk_queue->submit.mutex);
      const bool queued = !list_is_empty(&vk_queue->submit.submits);
      mtx_unlock(&vk_queue->submit.mutex);
      if (queued)
         return false;

      const uint64_t submitted = p_atomic_read(&queue->drm.timeline_value);
      uint64_t point = 0;
      if (!queue->drm.syncobj ||
          drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &point, 1) ||
          point < submitted)
         return false;
   }
   return true;
}

/* Wait until query becomes available. *never is set, with VK_SUCCESS, for a
 * query that no submitted work will make available.
 *
 * On M5 the kernel fails a command caught in another command's GPU recovery
 * as a victim: its fence completes with an error, the VM stays usable (its
 * status word stays zero; brl_check_status() loses the device otherwise),
 * and the command does not write the queries it would have ended or
 * timestamped. Vulkan only says a query that never becomes available "may"
 * lose the device (vkGetQueryPoolResults, VK_QUERY_RESULT_WAIT_BIT); losing
 * it here would turn the survivable victim into a device loss. So once
 * every queue is idle with the device intact, report the query unavailable
 * instead of waiting for the timeout. */
static VkResult
brl_query_wait_for_available(struct brl_device *dev, struct brl_query_pool *pool,
                            uint32_t query, bool *never)
{
   uint64_t abs_timeout_ns = os_time_get_absolute_timeout(BRL_QUERY_TIMEOUT);
   const bool victims = agx_chip_g17_m5(dev->dev.chip);

   *never = false;
   while (os_time_get_nano() < abs_timeout_ns) {
      if (brl_query_is_available(dev, pool, query))
         return VK_SUCCESS;

      VkResult status = vk_device_check_status(&dev->vk);
      if (status != VK_SUCCESS)
         return status;

      if (victims && brl_queues_idle(dev)) {
         /* The last producer may have retired after the check above. */
         if (brl_query_is_available(dev, pool, query))
            return VK_SUCCESS;
         status = vk_device_check_status(&dev->vk);
         if (status != VK_SUCCESS)
            return status;

         mesa_logw_once("query %u became unavailable for good: all submitted "
                        "work completed without writing it (a command the "
                        "kernel failed as a recovery victim)", query);
         *never = true;
         return VK_SUCCESS;
      }
   }

   return vk_device_set_lost(&dev->vk, "query timeout");
}

VKAPI_ATTR VkResult VKAPI_CALL
brl_GetQueryPoolResults(VkDevice device, VkQueryPool queryPool,
                       uint32_t firstQuery, uint32_t queryCount,
                       size_t dataSize, void *pData, VkDeviceSize stride,
                       VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(brl_device, dev, device);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   VkResult status = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t query = firstQuery + i;

      bool available = brl_query_is_available(dev, pool, query);

      bool never = false;
      if (!available && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         status = brl_query_wait_for_available(dev, pool, query, &never);
         if (status != VK_SUCCESS)
            return status;

         available = !never;
      }

      /* A retirement timestamp (any value but the queued store's 1) was
       * written after the end-of-render maintenance: no submission wait. */
      if (available &&
          !(pool->avail_shift && brl_query_available_value(pool, query) > 1)) {
         VkResult done = brl_oq_submission_done(
            dev, pool, query, flags & VK_QUERY_RESULT_WAIT_BIT, &available);
         if (done != VK_SUCCESS)
            return done;
      }

      const struct brl_query_report *src = brl_query_report_map(dev, pool, query);
      assert(i * stride < dataSize);
      void *dst = (char *)pData + i * stride;

      uint32_t reports = brl_reports_per_query(pool);
      if (never) {
         /* Zero results, reported unavailable: nothing was written. */
         uint64_t zero[32] = {0};
         assert(reports <= ARRAY_SIZE(zero));
         brl_query_write_cpu_result(dst, zero, reports, false,
                                   flags | VK_QUERY_RESULT_PARTIAL_BIT);
         continue;
      }
      const uint64_t zero = 0;
      if (!brl_query_write_cpu_result(dst, src ? &src[0].value : &zero,
                                     reports, available, flags))
         status = VK_NOT_READY;
   }

   /* Availability can already be set when entering this call. Check GPU abort
    * status after observing it too, including robust-index allocation failure. */
   VkResult device_status = vk_device_check_status(&dev->vk);
   return device_status == VK_SUCCESS ? status : device_status;
}

static void
brl_cmd_copy_query_pool_results(struct brl_cmd_buffer *cmd,
                               struct brl_query_pool *pool, uint32_t firstQuery,
                               uint32_t queryCount, struct brl_buffer *dst_buffer,
                               VkDeviceSize dstOffset, VkDeviceSize stride,
                               VkQueryResultFlags flags)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);

   if (!queryCount)
      return;

   /* Query availability writes may be deferred to the post-graphics stream.
    * Finish the producing streams before creating a waiting consumer so it
    * cannot spin ahead of its own availability write. Do this before
    * brl_query_command_cs attaches dependencies to the consumer stream. */
   if (agx_chip_native_agx3(dev->dev.chip) &&
       (flags & VK_QUERY_RESULT_WAIT_BIT)) {
      brl_cmd_buffer_end_graphics(cmd);
      brl_cmd_buffer_end_compute(cmd);
   }

   /* The copy reads results and availability every earlier stream may still
    * be producing. Ordering the destination buffer against later use is the
    * application's (it is a transfer operation, Vulkan 1.3, 17.2). */
   if (!brl_query_command_cs(cmd))
      return;

   perf_debug(cmd, "Query pool copy");

   if (agx_chip_native_agx3(dev->dev.chip)) {
      if ((pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP &&
           pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION &&
           pool->vk.query_type != VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT &&
           pool->vk.query_type != VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT &&
           pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS &&
           pool->vk.query_type != VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT &&
           !brl_is_acceleration_structure_query(pool->vk.query_type)) ||
          firstQuery > pool->vk.query_count ||
          queryCount > pool->vk.query_count - firstQuery) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      const unsigned reports = brl_reports_per_query(pool);
      const unsigned word_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;
      const uint64_t tuple_size = word_size *
         (reports + !!(flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
      if (!reports || dstOffset > dst_buffer->vk.size ||
          tuple_size > dst_buffer->vk.size - dstOffset ||
          (queryCount > 1 &&
           stride > (dst_buffer->vk.size - dstOffset - tuple_size) /
                       (queryCount - 1))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* The copy reads each counter through the pool's index table. Assign
       * the counters now even if their producers have not been recorded. */
      for (uint32_t i = 0; pool->oq_queries && i < queryCount; i++) {
         VkResult result = brl_ensure_occlusion_query(dev, pool, firstQuery + i);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
      }

      struct libagx_copy_query_g17p_args info = {
         .availability = brl_has_available(pool) ? pool->bo->va->addr : 0,
         .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                    : pool->bo->va->addr + pool->query_start,
         .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,
         .first_query = firstQuery,
         .dst_addr = brl_buffer_address_rw(dst_buffer, dstOffset),
         .dst_stride = stride,
         .reports_per_query = reports,
         .flags = flags,
         .availability_shift = pool->avail_shift,
      };

      struct brl_cs *cs = cmd->current_cs.cs;
      brl_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      brl_cdm_cache_flush(dev, cs);
      libagx_copy_query_g17p_struct(cmd, agx_1d(queryCount), AGX_BARRIER_NONE,
                                   info);
      brl_note_query_access(cmd, pool, firstQuery, queryCount);
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      cs = cmd->current_cs.cs;
      brl_drm_attachment_set_add(&cs->compute_attachments, info.dst_addr,
         (uint64_t)(queryCount - 1) * stride + tuple_size);
      cs->stats.query_copy_dispatches++;
      cs->stats.query_copy_queries += queryCount;
      cs->stats.query_copy_counters += (uint64_t)queryCount * reports;
      return;
   }

   struct libagx_copy_query_args info = {
      .availability = brl_has_available(pool) ? pool->bo->va->addr : 0,
      .results = pool->oq_queries ? dev->occlusion_queries.bo->va->addr
                                  : pool->bo->va->addr + pool->query_start,
      .oq_index = pool->oq_queries ? pool->bo->va->addr + pool->query_start : 0,

      .first_query = firstQuery,
      .dst_addr = brl_buffer_address_rw(dst_buffer, dstOffset),
      .dst_stride = stride,
      .reports_per_query = brl_reports_per_query(pool),
      .flags = flags,
      .availability_shift = pool->avail_shift,
   };

   libagx_copy_query_struct(cmd, agx_1d(queryCount), AGX_BARRIER_ALL, info);
   brl_note_query_access(cmd, pool, firstQuery, queryCount);
}

VKAPI_ATTR void VKAPI_CALL
brl_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                           uint32_t firstQuery, uint32_t queryCount,
                           VkBuffer dstBuffer, VkDeviceSize dstOffset,
                           VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(brl_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(brl_query_pool, pool, queryPool);
   VK_FROM_HANDLE(brl_buffer, dst_buffer, dstBuffer);

   cmd->sync.kind_override = BRL_SYNC_ALL;
   brl_cmd_copy_query_pool_results(cmd, pool, firstQuery, queryCount,
                                  dst_buffer, dstOffset, stride, flags);
   cmd->sync.kind_override = 0;
}
