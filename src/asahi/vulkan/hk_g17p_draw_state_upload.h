/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_g17p_draw_state_upload.h"

static inline bool
hk_g17p_upload_draw_indices(void *cpu, size_t capacity, uint64_t compact,
                            size_t scissor_bytes, size_t depth_bias_bytes,
                            unsigned viewport_count, uint32_t *offset)
{
   return agx_g17p_upload_draw_indices(cpu, capacity, compact, scissor_bytes,
                                       depth_bias_bytes, viewport_count,
                                       offset);
}
