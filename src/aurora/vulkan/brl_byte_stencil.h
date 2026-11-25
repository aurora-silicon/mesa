/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef BRL_BYTE_STENCIL_H
#define BRL_BYTE_STENCIL_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"
#include "util/blake3/blake3.h"

struct brl_descriptor_state;
struct brl_device;
struct vk_descriptor_set_layout;

/* Maximum supported 1-byte stencil diameter (2 * 28 + 1 = 57). Keeps shared
 * memory staging compact (<= 7.75 KiB vertical, <= 3.0 KiB horizontal) so
 * threadgroup occupancy stays full. */
#define BRL_BYTE_STENCIL_MAX_DIAMETER  57u
#define BRL_BYTE_STENCIL_MAX_RADIUS    28u

/* The RGBA tiles include a halo of up to 64 pixels on either side.
 * Decoded horizontal rows need 8 KiB. Packed vertical columns need 12.25 KiB.
 */
#define BRL_RGBA8_STENCIL_MAX_DIAMETER 129u
#define BRL_RGBA8_STENCIL_MAX_RADIUS   64u

enum brl_byte_stencil_axis {
   BRL_BYTE_STENCIL_HORIZ = 0,
   BRL_BYTE_STENCIL_VERT = 1,
};

enum brl_byte_stencil_store_conv {
   BRL_BYTE_STENCIL_CONV_F2I32 = 0,
   BRL_BYTE_STENCIL_CONV_F2U32 = 1,
};

enum brl_byte_stencil_elem_kind {
   BRL_BYTE_STENCIL_ELEM_U8 = 0,
   BRL_BYTE_STENCIL_ELEM_RGBA8_CLAMPED = 1,
};

struct brl_byte_stencil_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t descriptor_offset_B;
   blake3_hash set_layout_hash;
};

struct brl_byte_stencil_plan {
   uint8_t axis;
   uint8_t store_conv;
   uint8_t is_grid_stride;
   uint8_t elem_kind;
   uint16_t tile_w;
   uint16_t tile_h;
   uint32_t orig_wg_size[3];
   uint16_t width_pc_offset_B;
   uint16_t height_pc_offset_B;
   uint16_t stride_pc_offset_B;
   uint16_t diameter_pc_offset_B;
   struct brl_byte_stencil_binding src;
   struct brl_byte_stencil_binding dst;
   struct brl_byte_stencil_binding weights;
};

bool brl_nir_match_byte_stencil(
   const nir_shader *orig_nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct brl_byte_stencil_plan *plan_out);

nir_shader *brl_nir_build_tiled_byte_stencil(
   void *mem_ctx, const nir_shader *orig_nir,
   const struct brl_byte_stencil_plan *plan,
   nir_address_format ssbo_addr_format);

bool brl_byte_stencil_validate_dispatch_core(
   bool alias_free, const struct brl_descriptor_state *desc,
   const struct brl_byte_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

bool brl_byte_stencil_validate_dispatch(
   const struct brl_device *dev, const struct brl_descriptor_state *desc,
   const struct brl_byte_stencil_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

#endif /* BRL_BYTE_STENCIL_H */
