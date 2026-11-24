/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The HAL200 (G15G/S/C, G16G, G17P) per-draw state block: the fragment,
 * output, face, cull and viewport records a draw's bind tail points at, and
 * the scissor and depth-bias records its draw-state selector indexes.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"
#include "agx_g17p_draw_state_upload.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_ppp_hal200_raster_core.h"
#include "agx_ppp_viewport_payload_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One face's stencil state. Operations and comparison use the native
 * numbering, which is Vulkan's. */
struct agx_hal200_stencil_face {
   bool enabled;
   uint8_t compare, fail, depth_fail, pass;
   uint8_t compare_mask, write_mask, reference;
};

struct agx_hal200_viewport {
   /* The viewport clipped by the framebuffer and the scissor, in pixels. */
   uint32_t min_x, min_y, max_x, max_y;
   /* Depth range, float bits. */
   uint32_t min_z_raw, max_z_raw;
   struct agx_ppp_viewport_transform_raw transform;
};

/*
 * Everything the state block is built from, in hardware terms. A driver
 * translates its API state into this; two descriptions that compare equal
 * produce the same bytes.
 *
 * The fields marked "driver policy" are where Borealis and the Gallium
 * driver decide differently today. The builder takes each driver's decision
 * as given and applies none of its own.
 */
struct agx_hal200_draw_state {
   /* Fragment stage. */
   /* Absolute 48-bit, 64-byte aligned stage-program address. */
   uint64_t fragment_program;
   /* Absolute 48-bit, 64-byte aligned VIT address, zero without varyings. */
   uint64_t varying_table;
   /* The linked FRAGMENT_CONTROL word: pass type, sample mask after
    * depth/stencil and tag write disable. Driver policy: Borealis passes
    * a punch-through pass promoted to translucent punch-through where it
    * wants whole quads, and keeps tag write disable; Gallium does neither. */
   uint32_t fragment_control;
   uint32_t nr_bindings;
   bool fragment_reads_z;
   /* enum agx_conservative_depth of the shader's depth layout. */
   uint32_t conservative_depth;
   bool early_fragment_tests;
   bool fragment_writes_memory;
   /* Driver policy: Gallium never sets thread-invariant resources. */
   bool fragment_ti_resources;
   uint32_t fragment_ti_spill_bytes;
   uint32_t fragment_ti_largest_ipr_bytes;
   /* Driver policy: which shaders and coverage states forbid triangle
    * merging. The primitive term is derived here, from object_type and
    * polygon_mode. */
   bool shader_tri_merge_disable;
   bool coverage_tri_merge_disable;

   /* Hardware vertex stage. */
   uint32_t vertex_output_select;
   uint32_t vertex_output_size;
   /* Words per primitive of a G17P mesh shader's per-primitive page. */
   uint32_t mesh_primitive_words;
   bool mesh_primitive_address;
   bool generate_primitive_id;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;

   /* Depth and stencil. */
   bool depth_test;
   bool depth_write;
   uint32_t depth_compare;
   bool stencil_test;
   struct agx_hal200_stencil_face stencil_front, stencil_back;

   /* Rasterizer. */
   uint32_t line_width;
   uint32_t polygon_mode;
   uint32_t object_type;
   uint32_t cull_mode;
   uint32_t provoking;
   bool front_face_ccw;
   bool depth_clip;
   bool rasterizer_discard;
   /* Driver policy: Gallium never sets the Bresenham line bit. */
   bool bresenham_lines;
   /* Driver policy: Gallium always scissors, Borealis elides a scissor
    * that covers the whole frame. */
   bool scissor_enable;
   bool depth_bias_enable;
   uint32_t depth_bias_raw, slope_scale_raw, depth_bias_clamp_raw;

   /* enum agx_visibility_mode and the counter index. */
   uint32_t occlusion_mode;
   uint32_t occlusion_index;

   /* Only the first viewport_count viewports are read. */
   uint32_t viewport_count;
   struct agx_hal200_viewport viewports[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
};

struct agx_hal200_draw_state_packed {
   uint8_t bind_group[AGX_G17P_BIND_GROUP_BYTES];
   struct agx_ppp_viewport_payload_core viewport;
   uint8_t scissors[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                   [AGX_PPP_HAL200_SCISSOR_RECORD_LENGTH];
   uint8_t depth_bias[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                     [AGX_PPP_HAL200_DEPTH_BIAS_RECORD_LENGTH];
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint8_t viewport_count;
};

/* Only the first viewport_count entries of scissors and depth_bias are
 * written. They are written, and out->viewport_count set, before anything
 * that can refuse a state is checked: the Gallium driver appends them to its
 * batch even for a draw it then drops. */
bool agx_hal200_draw_state_pack(enum agx_chip chip,
                                const struct agx_hal200_draw_state *state,
                                struct agx_hal200_draw_state_packed *out);

/* The fields of the fragment record, which is the first
 * AGX_PPP_HAL200_FS_LENGTH bytes of the bind group and the only record that
 * holds the fragment program address. A driver that keeps them can pack that
 * record again for another program with agx_ppp_hal200_fragment_core_pack()
 * and leave the rest of a state block as it is. */
bool agx_hal200_draw_state_fragment_fields(
   enum agx_chip chip, const struct agx_hal200_draw_state *state,
   struct agx_ppp_hal200_fragment_fields *fields);

/* Size of the state block a packed state uploads to; zero if invalid. */
static inline size_t
agx_hal200_draw_state_size(const struct agx_hal200_draw_state_packed *packed)
{
   return agx_g17p_bind_state_size(packed->viewport.length);
}

/* Write the state block at cpu, whose compact (VDM-relative) address is
 * given, and return the bind tail whose records point into it. */
static inline bool
agx_hal200_draw_state_upload(const struct agx_hal200_draw_state_packed *packed,
                             void *cpu, size_t capacity, uint64_t compact,
                             struct agx_vdm_hal200_bind_tail *tail)
{
   struct agx_g17p_bind_state_upload uploaded;
   if (!agx_g17p_upload_bind_state(cpu, capacity, compact, packed->bind_group,
                                   packed->viewport.bytes,
                                   packed->viewport.length,
                                   packed->varying_counts_32,
                                   packed->varying_counts_16, &uploaded))
      return false;

   *tail = uploaded.tail;
   return true;
}

#ifdef __cplusplus
}
#endif
