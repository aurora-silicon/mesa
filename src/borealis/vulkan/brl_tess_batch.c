/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "brl_tess_batch.h"
#include "asahi/libagx/tessellator.h"
#include "brl_cmd_buffer.h"
#include "brl_device.h"
#include "brl_shader.h"
#include "libagx_shaders.h"

bool
brl_tess_batch_open(struct brl_cmd_buffer *cmd)
{
   struct brl_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < BRL_TESS_STAGES; ++i) {
      if (!batch->stages[i])
         batch->stages[i] = brl_cs_create(cmd, true);

      if (!batch->stages[i]) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   return true;
}

/* No launch followed the last drop (an empty grid): the barrier, still in
 * place behind current, comes back before anything else follows. */
static void
brl_tess_batch_restore(struct brl_tess_batch *batch, enum brl_tess_stage stage)
{
   struct brl_cs *cs = batch->stages[stage];
   if (batch->dropped[stage] &&
       (uint8_t *)cs->current + AGX3_CDM_BARRIER_SIZE == batch->dropped[stage]) {
      cs->current = batch->dropped[stage];
      cs->weak_barrier_end = cs->current;
   }
   batch->dropped[stage] = NULL;
}

struct brl_cs *
brl_tess_batch_stage(struct brl_cmd_buffer *cmd, enum brl_tess_stage stage)
{
   struct brl_tess_batch *batch = &cmd->tess_batch;
   struct brl_cs *cs = batch->stages[stage];

   brl_tess_batch_restore(batch, stage);

   /* Before the tail goes: a chunk link must still follow a barrier. */
   brl_ensure_cs_has_space(cmd, cs, 0x2000);
   if (cs->stats.cmds && cs->current == cs->weak_barrier_end) {
      batch->dropped[stage] = cs->current;
      cs->current = (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
      cs->weak_barrier_end = NULL;
   }

   return cs;
}

void
brl_launch_tess_batched(struct brl_cmd_buffer *cmd, struct brl_shader *vs,
                       struct brl_shader *tcs, struct agx_grid grid_vs,
                       struct agx_grid grid_tcs, struct agx_grid grid_tess,
                       struct brl_tess_info info, uint64_t state)
{
   struct brl_tess_batch *batch = &cmd->tess_batch;
   if (!brl_tess_batch_open(cmd))
      return;

   /* As brl_dispatch_with_local_size(), without its flush. */
   if (!agx_is_shader_empty(&vs->b)) {
      struct brl_cs *cs = brl_tess_batch_stage(cmd, BRL_TESS_STAGE_VS);
      brl_reserve_scratch(cmd, cs, vs);
      brl_dispatch_g17p_native(cmd, cs, vs, grid_vs, agx_workgroup(64, 1, 1));
   }
   if (!agx_is_shader_empty(&tcs->b)) {
      struct brl_cs *cs = brl_tess_batch_stage(cmd, BRL_TESS_STAGE_TCS);
      brl_reserve_scratch(cmd, cs, tcs);
      brl_dispatch_g17p_native(
         cmd, cs, tcs, grid_tcs,
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
   }

   /* Precompiled kernels record into the pre-graphics stream: stand each
    * stage's side stream in for it. Batched draws count nothing, so the
    * prefix sum has no counters to increment. */
   struct brl_cs *pre_gfx = cmd->current_cs.pre_gfx;
   batch->recording = true;

   cmd->current_cs.pre_gfx = brl_tess_batch_stage(cmd, BRL_TESS_STAGE_COUNT);
   libagx_tessellate(cmd, grid_tess, AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_COUNT, state);
   cmd->current_cs.pre_gfx = brl_tess_batch_stage(cmd, BRL_TESS_STAGE_PREFIX_SUM);
   libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_PREGFX, state, 0, 0, 0,
                          false);
   cmd->current_cs.pre_gfx = brl_tess_batch_stage(cmd, BRL_TESS_STAGE_TESSELLATE);
   libagx_tessellate(cmd, grid_tess, AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   cmd->current_cs.pre_gfx = pre_gfx;
   batch->recording = false;
   batch->draws++;
}

void
brl_tess_batch_flush(struct brl_cmd_buffer *cmd)
{
   struct brl_tess_batch *batch = &cmd->tess_batch;
   if (!batch->draws)
      return;

   /* Created by the first batched draw, and kept open until after this. */
   struct brl_cs *cs = cmd->current_cs.pre_gfx;
   batch->draws = 0;

   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   for (unsigned i = 0; i < BRL_TESS_STAGES; ++i) {
      /* Every stage then ends in its last launch's weak barrier, which the
       * flush below strengthens in place. */
      brl_tess_batch_restore(batch, i);
      struct brl_cs *stage = batch->stages[i];
      batch->stages[i] = NULL;

      /* An empty stream has no barrier at its end to take over. */
      if (cs && stage->stats.cmds) {
         /* Copy a one-chunk stage into the current chunk, so its root takes
          * the next batch instead of five new roots on every flush.
          * brl_cs_merge_cdm() copies only into an unlinked stream, but the
          * copy needs no more than room in the chunk being written. */
         const bool copy = !stage->stream_linked;
         if (copy)
            brl_ensure_cs_has_space(
               cmd, cs, (uint8_t *)stage->current - (uint8_t *)stage->start);
         const bool linked = cs->stream_linked;
         cs->stream_linked &= !copy;
         brl_cs_merge_cdm(cs, stage);
         cs->stream_linked |= linked;

         /* The next stage reads what this one wrote. No chunk link may come
          * between the copy and this, which needs no room. */
         assert(cs->current == cs->weak_barrier_end);
         brl_cdm_cache_flush(dev, cs);
      }

      /* A stage the merge linked to instead is part of cs now. */
      const struct agx_ptr root = {.cpu = stage->start, .gpu = stage->addr};
      const bool reuse =
         !stage->stream_linked && (!cs || root.cpu != cs->chunk.cpu);
      brl_cs_destroy(stage);
      if (reuse)
         batch->stages[i] = brl_cs_create_at(cmd, true, root);
   }
}

void
brl_tess_batch_reset(struct brl_cmd_buffer *cmd)
{
   struct brl_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < BRL_TESS_STAGES; ++i) {
      if (batch->stages[i])
         brl_cs_destroy(batch->stages[i]);
   }

   *batch = (struct brl_tess_batch){0};
}
