/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "brl_cmd_buffer.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned flushes, reservations;

/* Exercise the production drain hook with command emission replaced by a
 * recorder. No device or BO allocation is needed to close an existing stream.
 */
void
brl_ensure_cs_has_space(struct brl_cmd_buffer *cmd, struct brl_cs *cs, size_t size)
{
   assert(cmd == cs->cmd);
   assert(size == AGX3_CDM_BARRIER_SIZE);
   reservations++;
}

void
brl_cdm_cache_flush(struct brl_device *dev, struct brl_cs *cs)
{
   (void)dev;
   assert(cs->query_reset_pending);
   assert(reservations == flushes + 1);
   flushes++;
}

/* All fixtures reuse a nonempty, already allocated general stream. These
 * paths must not allocate, change dependencies, discard a stream or emit
 * unrelated work while draining query resets. */
void
brl_flush_deferred_clears_why(struct brl_cmd_buffer *cmd, const char *why)
{
   (void)cmd;
   (void)why;
   abort();
}

void
brl_sync_check_cs(struct brl_cmd_buffer *cmd)
{
   (void)cmd;
   abort();
}

struct agx_ptr
brl_pool_cs_alloc(struct brl_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd;
   (void)size;
   (void)alignment;
   abort();
}

bool
brl_cs_init_graphics(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

void
brl_drm_attachment_set_init(struct brl_drm_attachment_set *set)
{
   (void)set;
   abort();
}

void
brl_sync_stream_created(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                       struct brl_cs **slot, bool compute)
{
   (void)cmd;
   (void)cs;
   (void)slot;
   (void)compute;
   abort();
}

void
brl_dispatch_imm_writes(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

bool
brl_defer_oq_available(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

void
brl_sync_forget_stream(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

/* These devices never map memory at a second address, so the real
 * function adds no flush either.
 */
void
brl_cdm_end_stream_flush(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
}

/* No tessellation is batched here. */
void
brl_tess_batch_flush(struct brl_cmd_buffer *cmd)
{
   (void)cmd;
   abort();
}

/* Query resets are not transfers. */
void
brl_sync_note_xfer_write(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

int
main(void)
{
   struct brl_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   assert(cmd);
   struct brl_cs cs = {.cmd = cmd, .type = BRL_CS_CDM};

   brl_flush_query_resets(cmd, &cs);
   assert(flushes == 0 && reservations == 0);

   /* The real getter defers drains during consecutive reset recording. */
   cs.query_reset_pending = true;
   cs.stats.query_reset_dispatches = 3;
   cs.stats.cmds = 3;
   cs.sync.kind = BRL_SYNC_ALL;
   cmd->current_cs.cs = &cs;
   cmd->state.query_reset_recording = true;
   assert(brl_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(cs.query_reset_pending && flushes == 0);

   /* The next consumer goes through that same getter and drains once. */
   cmd->state.query_reset_recording = false;
   assert(brl_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(!cs.query_reset_pending);
   assert(flushes == 1 && cs.stats.query_reset_flushes == 1);
   assert(brl_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(flushes == 1);

   /* Closing a stream also drains, even inside reset recording. */
   cmd->state.query_reset_recording = true;
   cs.query_reset_pending = true;
   brl_cmd_buffer_end_compute(cmd);
   assert(!cs.query_reset_pending && cmd->current_cs.cs == NULL);
   assert(flushes == 2 && cs.stats.query_reset_flushes == 2);
   assert(cs.stats.query_reset_dispatches == 3);
   brl_cmd_buffer_end_compute(cmd);
   assert(flushes == 2);
   free(cmd);
   puts(
      "query reset drains: empty, reset recording, consumer and stream end passed");
   return 0;
}
