/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "brl_cmd_buffer.h"

/* Reset dispatches may overlap one another: they write the same reset value
 * even if their ranges overlap. Make them visible once before any consumer,
 * or before closing a stream that later streams/submissions can depend on.
 * Keep the existing queue dependencies; only the redundant per-reset cache
 * flush is deferred. Host resets and multiview availability publication do
 * not use this path.
 */
void
brl_flush_query_resets(struct brl_cmd_buffer *cmd, struct brl_cs *cs)
{
   if (!cs->query_reset_pending)
      return;

   brl_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
   brl_cdm_cache_flush(brl_cmd_buffer_device(cmd), cs);
   cs->query_reset_pending = false;
   cs->stats.query_reset_flushes++;
}
