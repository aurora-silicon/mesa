/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "brl_cmd_buffer.h"
#include "brl_device.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_ioto_program_core.h"

#include "util/cache_ops.h"
#include "util/log.h"
#include "util/u_dynarray.h"

#include <inttypes.h>
#include <string.h>

void
brl_ioto_arena_finish(struct brl_cmd_buffer *cmd)
{
   struct brl_ioto_arena *arena = &cmd->ioto;

   brl_device_free_span(brl_cmd_buffer_device(cmd), arena->span);
   memset(arena, 0, sizeof(*arena));
}

/*
 * Start a new chunk. The old one keeps whatever it already recorded -- those
 * addresses are still live in the recorded stream -- and stays owned by
 * large_bos.
 */
/* With cached command memory, programs are cleaned per chunk rather than per
 * program: everything below the cursor, which also covers reservations that
 * were never published. */
void
brl_ioto_arena_clean(struct brl_cmd_buffer *cmd)
{
   struct brl_ioto_arena *arena = &cmd->ioto;

   if (arena->map && arena->cursor &&
       brl_g17p_cached_cmd_memory(&brl_cmd_buffer_device(cmd)->dev))
      util_flush_range_no_fence(arena->map, arena->cursor);
}

static bool
brl_ioto_arena_grow(struct brl_cmd_buffer *cmd, size_t size, const char *label)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   struct brl_ioto_arena *arena = &cmd->ioto;

   brl_ioto_arena_clean(cmd);

   /* See BRL_CMD_SPAN_SIZE: one upload writes about 500 bytes of programs. */
   if (!arena->map && size <= BRL_CMD_SPAN_SIZE &&
       brl_g17p_cached_cmd_memory(&dev->dev)) {
      struct agx_ptr ptr;
      if (brl_cmd_pool_alloc_span(brl_cmd_buffer_pool(cmd), BRL_CMD_SLAB_IOTO,
                                 BRL_CMD_SPAN_SIZE, BRL_CMD_SPAN_SIZE, &ptr,
                                 &arena->span) != VK_SUCCESS)
         return false;

      arena->map = ptr.cpu;
      arena->base = ptr.gpu;
      arena->cursor = 0;
      arena->capacity = BRL_CMD_SPAN_SIZE;
      arena->chunk_bytes += BRL_CMD_SPAN_SIZE;
      return true;
   }

   const size_t chunk =
      brl_ioto_arena_chunk_size(size, AGX_IOTO_PROGRAM_ALIGNMENT);
   if (chunk == 0)
      return false;

   struct agx_bo *bo =
      agx_bo_create(&dev->dev, chunk, AGX_IOTO_PROGRAM_ALIGNMENT,
                    AGX_BO_EXEC | AGX_BO_LOW_VA |
                       (brl_g17p_cached_cmd_memory(&dev->dev) ? AGX_BO_WRITEBACK
                                                             : 0),
                    "G17P Ioto program arena");
   if (!bo)
      return false;

   void *map = agx_bo_map(bo);
   if (!map || !bo->va) {
      agx_bo_unreference(&dev->dev, bo);
      return false;
   }

   /* The chunk is retained exactly like the per-program BOs it replaces. */
   util_dynarray_append(&cmd->large_bos, bo);

   arena->bo = bo;
   arena->map = map;
   arena->base = bo->va->addr;
   arena->cursor = 0;
   arena->capacity = bo->size > UINT32_MAX ? UINT32_MAX : (uint32_t)bo->size;
   arena->chunk_bytes += bo->size;

   return true;
}

static_assert(BRL_IOTO_PROGRAM_ALIGNMENT == AGX_IOTO_PROGRAM_ALIGNMENT,
              "inline Ioto reservation alignment");

bool
brl_cmd_reserve_ioto_program_slow(struct brl_cmd_buffer *cmd, size_t size,
                                 const char *label,
                                 struct brl_ioto_reservation *reservation)
{
   struct brl_ioto_arena *arena = &cmd->ioto;
   struct brl_ioto_arena_span span;

   if (!reservation || size == 0) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (!arena->map ||
       !brl_ioto_arena_place(arena->cursor, arena->capacity, size,
                            AGX_IOTO_PROGRAM_ALIGNMENT, &span)) {
      if (!brl_ioto_arena_grow(cmd, size, label) ||
          !brl_ioto_arena_place(arena->cursor, arena->capacity, size,
                               AGX_IOTO_PROGRAM_ALIGNMENT, &span)) {
         /* Name what ran out. A bare VK_ERROR_OUT_OF_DEVICE_MEMORY from here
          * is indistinguishable from a texture allocation failing, and the
          * count is the number this recording actually needed. */
         mesa_loge("G17P Ioto arena exhausted requesting %zu bytes for %s "
                   "after %u programs (%" PRIu64 " program bytes "
                   "in %" PRIu64 " bytes of USC window)",
                   size, label ? label : "(program)", arena->programs,
                   arena->program_bytes, arena->chunk_bytes);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   *reservation = (struct brl_ioto_reservation){
      .map = arena->map + span.offset,
      .va = arena->base + span.offset,
   };
   arena->cursor = span.next_offset;
   return true;
}

uint64_t
brl_cmd_publish_ioto_program(struct brl_cmd_buffer *cmd,
                            const struct brl_ioto_reservation *reservation,
                            size_t length)
{
   struct brl_ioto_arena *arena = &cmd->ioto;

   if (!reservation || !reservation->map || length == 0 ||
       length > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   assert(reservation->va >= arena->base);
   const uint64_t offset = reservation->va - arena->base;
   assert(offset <= arena->capacity && length <= arena->capacity - offset);

   arena->programs++;

   /* Give back everything the reservation over-reserved: sites reserve the
    * composer's bound and compose something shorter. */
   arena->cursor = (uint32_t)offset + (uint32_t)length;

   /* Cached memory is cleaned with its chunk (brl_ioto_arena_clean()). */
   if (!brl_g17p_cached_cmd_memory(&brl_cmd_buffer_device(cmd)->dev))
      agx_bo_assert_range(arena->bo, offset, length);
   arena->program_bytes += length;
   return reservation->va;
}
