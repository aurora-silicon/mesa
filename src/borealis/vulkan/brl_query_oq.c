/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

/* Deferred occlusion availability (brl_query_oq.h): the decisions made at the
 * end of a command buffer, kept apart from the rest of the query code so they
 * can be tested without a device. */

#include "brl_query_oq.h"

#include "brl_cmd_buffer.h"
#include "brl_device.h"
#include "brl_sync.h"
#include "asahi/libagx/query.h"
#include "util/list.h"
#include "util/u_dynarray.h"

bool
brl_defer_oq_available(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   const unsigned writes =
      util_dynarray_num_elements(&cs->imm_writes, struct libagx_imm_write);
   const unsigned stamps =
      util_dynarray_num_elements(&cs->oq_stamps, struct brl_oq_stamp);

   if (!stamps || stamps != writes || cs->stats.cmds ||
       cs->stats.flushes || cs->timestamp.start.handle ||
       cs->timestamp.end.handle || cs->query_reset_pending)
      return false;

   const struct brl_oq_stamp *s = cs->oq_stamps.data;
   for (unsigned i = 1; i < stamps; ++i) {
      if (s[i].render_seq != s[0].render_seq)
         return false;
   }

   cs->oq_avail_deferred = true;
   return true;
}

static bool
brl_dep_reaches(const struct brl_sync_dep *dep, const struct brl_cs *cs)
{
   return dep->cut && dep->cut->sync.seq >= cs->sync.seq;
}

enum brl_oq_resolution {
   /* Emit the store in place. */
   BRL_OQ_KEEP,
   /* The render carries the availability; drop the stream. */
   BRL_OQ_STAMPED,
   /* Nothing depends on the stream or (from compute) on its render, but the
    * render has too few free slots: the store may move to a later deferred
    * stream that waits on a later render. */
   BRL_OQ_MOVABLE,
};

/* Whether the render can carry the availability of the deferred stream post:
 * nothing later in the command buffer waits on post, no compute waits on the
 * render (or anything after it on the render subqueue), the command buffer's
 * tail requirement does not name it, and enough timestamp slots are free.
 */
static enum brl_oq_resolution
brl_stamp_oq_available(struct brl_cmd_buffer *cmd, struct brl_cs *post)
{
   const struct brl_oq_stamp *stamps = post->oq_stamps.data;
   const unsigned count =
      util_dynarray_num_elements(&post->oq_stamps, struct brl_oq_stamp);

   /* While deferred, the stream may have become the tail an API timestamp
    * rides on (brl_sync_write_timestamp()). It then stays, with its store. */
   if (post->timestamp.start.handle || post->timestamp.end.handle ||
       post->stats.cmds || post->stats.flushes)
      return BRL_OQ_KEEP;

   struct brl_cs *render = NULL;
   list_for_each_entry(struct brl_cs, it, &cmd->control_streams, node) {
      if (it == post)
         break;
      if (it->type == BRL_CS_VDM && it->sync.seq == stamps[0].render_seq)
         render = it;
   }
   if (!render || post->sync.dep[BRL_SYNC_VDM].cut != render)
      return BRL_OQ_KEEP;

   list_for_each_entry(struct brl_cs, it, &cmd->control_streams, node) {
      if (it == post || it == render)
         continue;
      if (brl_dep_reaches(&it->sync.dep[BRL_SYNC_CDM], post) ||
          brl_dep_reaches(&it->sync.dep_ta[BRL_SYNC_CDM], post))
         return BRL_OQ_KEEP;
      if (it->type == BRL_CS_CDM &&
          (brl_dep_reaches(&it->sync.dep[BRL_SYNC_VDM], render) ||
           brl_dep_reaches(&it->sync.dep_ta[BRL_SYNC_VDM], render)))
         return BRL_OQ_KEEP;
   }

   for (unsigned k = 0; k < BRL_SYNC_NUM_KINDS; ++k) {
      if (brl_dep_reaches(&cmd->sync.req[k][BRL_SYNC_VDM], render) ||
          brl_dep_reaches(&cmd->sync.req[k][BRL_SYNC_CDM], post))
         return BRL_OQ_KEEP;
   }

   /* The host-written slots only: vertex end and fragment start, which no
    * API timestamp uses, then vertex start. The fragment end is never used:
    * the kernel lets the firmware write it at the end of the pass, before the
    * end-of-render maintenance has made the counts visible
    * (brl_query_oq.h). A slot an API timestamp rides on stays with it. */
   const unsigned order[] = {1, 2, 0};
   unsigned free_slots[ARRAY_SIZE(order)], nfree = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
      const unsigned slot = order[i];
      if ((slot == 0 && render->timestamp.start.handle) ||
          render->oq_stamp[slot].handle)
         continue;
      free_slots[nfree++] = slot;
   }
   if (nfree < count)
      return BRL_OQ_MOVABLE;

   for (unsigned i = 0; i < count; ++i) {
      render->oq_stamp[free_slots[i]] = (struct agx_timestamp_req){
         .handle = stamps[i].handle,
         .offset_B = stamps[i].offset_B,
      };
   }
   return BRL_OQ_STAMPED;
}

static void
brl_drop_oq_stream(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   brl_sync_forget_stream(cmd, cs);
   list_del(&cs->node);
   brl_cs_destroy(cs);
}

/* Recording order: while an earlier stream is judged, later deferred streams
 * still wait on later renders, so only the last can ride on its render. */
static void
brl_resolve_oq_forward(struct brl_cmd_buffer *cmd)
{
   struct brl_cs *movable = NULL;

   list_for_each_entry_safe(struct brl_cs, cs, &cmd->control_streams, node) {
      if (cs->type != BRL_CS_CDM || !cs->oq_avail_deferred)
         continue;

      cs->oq_avail_deferred = false;
      switch (brl_stamp_oq_available(cmd, cs)) {
      case BRL_OQ_STAMPED:
         brl_drop_oq_stream(cmd, cs);
         break;
      case BRL_OQ_MOVABLE:
         /* The later stream's render wait covers every earlier render. */
         if (movable && movable->sync.dep[BRL_SYNC_VDM].cut->sync.seq >
                           cs->sync.dep[BRL_SYNC_VDM].cut->sync.seq) {
            brl_dispatch_imm_writes(cmd, movable);
            movable = NULL;
         }
         if (movable) {
            util_dynarray_append_dynarray(&cs->imm_writes, &movable->imm_writes);
            brl_drop_oq_stream(cmd, movable);
         }
         movable = cs;
         break;
      case BRL_OQ_KEEP:
         brl_dispatch_imm_writes(cmd, cs);
         break;
      }
   }

   if (movable)
      brl_dispatch_imm_writes(cmd, movable);
}

/* Whether the stores of the deferred stream post can join into, a later
 * movable stream: post carries nothing else, into waits on a render at or
 * after the one post waits on (renders complete in order, as
 * brl_resolve_oq_forward() also relies on), and nothing else in the command
 * buffer waits on post, so dropping it moves its stores later without moving
 * them past a consumer. Unlike in brl_stamp_oq_available(), compute waiting on
 * the render does not prevent this: into is that compute, and it is the one
 * command left for these stores. Stamping the render instead would make into
 * wait for the host-side retirement of the render. */
static bool
brl_oq_joins(struct brl_cmd_buffer *cmd, struct brl_cs *post, struct brl_cs *into)
{
   if (post->timestamp.start.handle || post->timestamp.end.handle ||
       post->stats.cmds || post->stats.flushes)
      return false;

   const struct brl_cs *render = post->sync.dep[BRL_SYNC_VDM].cut;
   const struct brl_cs *later = into->sync.dep[BRL_SYNC_VDM].cut;
   if (!render || !later || later->sync.seq < render->sync.seq)
      return false;

   list_for_each_entry(struct brl_cs, it, &cmd->control_streams, node) {
      if (it == post || it == into)
         continue;
      if (brl_dep_reaches(&it->sync.dep[BRL_SYNC_CDM], post) ||
          brl_dep_reaches(&it->sync.dep_ta[BRL_SYNC_CDM], post))
         return false;
   }

   for (unsigned k = 0; k < BRL_SYNC_NUM_KINDS; ++k) {
      if (brl_dep_reaches(&cmd->sync.req[k][BRL_SYNC_CDM], post))
         return false;
   }

   return true;
}

/* From the end of the command buffer. A later stream whose render carries its
 * stores is dropped before earlier streams are judged, so it no longer counts
 * as compute waiting on the render subqueue, and an earlier render can carry
 * its own stores too. A stream that stays (kept, or movable) still counts.
 * Once a movable stream remains, every earlier stream that can join it does
 * (brl_oq_joins()), whether or not its own render could have carried the
 * stores: the movable stream is emitted anyway and waits on all of those
 * renders, so one compute command serves the whole command buffer. */
static void
brl_resolve_oq_reverse(struct brl_cmd_buffer *cmd)
{
   struct brl_cs *movable = NULL;

   list_for_each_entry_safe_rev(struct brl_cs, cs, &cmd->control_streams, node) {
      if (cs->type != BRL_CS_CDM || !cs->oq_avail_deferred)
         continue;

      cs->oq_avail_deferred = false;
      if (movable && brl_oq_joins(cmd, cs, movable)) {
         util_dynarray_append_dynarray(&movable->imm_writes, &cs->imm_writes);
         brl_drop_oq_stream(cmd, cs);
         continue;
      }

      switch (brl_stamp_oq_available(cmd, cs)) {
      case BRL_OQ_STAMPED:
         brl_drop_oq_stream(cmd, cs);
         break;
      case BRL_OQ_MOVABLE:
         /* The later stream's render wait covers this stream's render. */
         if (movable && movable->sync.dep[BRL_SYNC_VDM].cut->sync.seq >=
                           cs->sync.dep[BRL_SYNC_VDM].cut->sync.seq) {
            util_dynarray_append_dynarray(&movable->imm_writes, &cs->imm_writes);
            brl_drop_oq_stream(cmd, cs);
            break;
         }
         if (movable)
            brl_dispatch_imm_writes(cmd, movable);
         movable = cs;
         break;
      case BRL_OQ_KEEP:
         brl_dispatch_imm_writes(cmd, cs);
         break;
      }
   }

   if (movable)
      brl_dispatch_imm_writes(cmd, movable);
}

bool
brl_oq_render_stamps_supported(enum agx_chip chip)
{
   /* G17P and M5 (G17G/G17S) both run the shared G17 kernel runtime: a render
    * completes only once its completion stamps cover it, after the
    * end-of-render maintenance, and the host then writes its four stage
    * timestamps into the requested slots before signalling the fence
    * (g17/queue/render/retirement.rs, g17/completion.rs). */
   return chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip);
}

void
brl_resolve_deferred_oq_available(struct brl_cmd_buffer *cmd)
{
   /* The reverse order is qualified on G17P; the other native AGX3 chips keep
    * recording order until it is measured there. */
   if (brl_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P)
      brl_resolve_oq_reverse(cmd);
   else
      brl_resolve_oq_forward(cmd);
}
