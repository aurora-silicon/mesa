/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef BRL_COMPUTE_OPT_H
#define BRL_COMPUTE_OPT_H

#include <stdbool.h>
#include <stdint.h>

#include "nir.h"
#include "util/blake3/blake3.h"

struct brl_descriptor_state;
struct brl_device;
struct vk_descriptor_set_layout;

enum brl_compute_opt_kind {
   BRL_COMPUTE_OPT_RGBA8_TO_GRAY_U8 = 0,
   BRL_COMPUTE_OPT_CANNY_HYSTERESIS_U8 = 1,
   BRL_COMPUTE_OPT_CANNY_NMS_U8 = 2,
   BRL_COMPUTE_OPT_CANNY_FINALIZE_U8 = 3,
};

struct brl_compute_opt_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t descriptor_offset_B;
   blake3_hash set_layout_hash;
};

struct brl_compute_opt_plan {
   uint8_t kind;
   uint8_t flags;
   uint16_t tile_w;
   uint16_t tile_h;
   uint16_t pad0;
   uint32_t orig_wg_size[3];

   /* Push-constant byte offsets (all 32-bit fields). */
   uint16_t pc_off[6];

   /* Kernel-specific scalar parameters captured verbatim from matched NIR.
    * - RGBA8_TO_GRAY_U8:    u32[0..2] = (w_r, w_g, w_b), u32[3] = shr
    * - CANNY_NMS_U8:        u32[0] = dx_lut, u32[1] = dy_lut,
    *                        u32[2] = low_thresh bits, u32[3] = high_thresh bits
    * - CANNY_FINALIZE_U8:   u32[0] = state_thresh, flags bit 0 = is_f2u32
    */
   uint32_t params[8];

   /* Up to 4 SSBO bindings:
    * - RGBA8_TO_GRAY_U8:    [0]=src(u32), [1]=dst(u8)
    * - CANNY_HYSTERESIS_U8: [0]=src(u8), [1]=dst(u8)
    * - CANNY_NMS_U8:        [0]=dir(i32), [1]=mag(f32), [2]=dst(u8)
    * - CANNY_FINALIZE_U8:   [0]=state(u8), [1]=mag(f32), [2]=dst(u8)
    */
   struct brl_compute_opt_binding bufs[4];
};

bool brl_nir_match_compute_opt(
   const nir_shader *orig_nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct brl_compute_opt_plan *plan_out);

nir_shader *brl_nir_build_compute_opt(
   void *mem_ctx, const nir_shader *orig_nir,
   const struct brl_compute_opt_plan *plan,
   nir_address_format ssbo_addr_format);

bool brl_compute_opt_validate_dispatch_core(
   bool alias_free, const struct brl_descriptor_state *desc,
   const struct brl_compute_opt_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

bool brl_compute_opt_validate_dispatch(
   const struct brl_device *dev, const struct brl_descriptor_state *desc,
   const struct brl_compute_opt_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t *out_groups_x, uint32_t *out_groups_y);

#endif /* BRL_COMPUTE_OPT_H */
