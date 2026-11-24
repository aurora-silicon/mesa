/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_hal200_draw_state.h"
#include "agx_pack.h"
#include "agx_ppp_w_clamp_cull2_core.h"

#include "util/macros.h"

#include <string.h>

bool
agx_hal200_draw_state_fragment_fields(
   enum agx_chip chip, const struct agx_hal200_draw_state *state,
   struct agx_ppp_hal200_fragment_fields *fields)
{
   *fields = (struct agx_ppp_hal200_fragment_fields){
      /* The HAL200 fragment record puts the iteration count at bit 16. The
       * packer merges the complete VIT address, including its high bits. */
      .resource_counts = state->nr_bindings << 16,
      .varying_table = state->varying_table,
      .stage_program = state->fragment_program,
   };

   return !state->fragment_ti_resources ||
          agx_ppp_hal200_fragment_ti_resources(
             chip, state->fragment_ti_spill_bytes,
             state->fragment_ti_largest_ipr_bytes, fields);
}

bool
agx_hal200_draw_state_pack(enum agx_chip chip,
                           const struct agx_hal200_draw_state *state,
                           struct agx_hal200_draw_state_packed *out)
{
   const unsigned viewport_count = state->viewport_count;

   out->viewport_count = 0;
   if (viewport_count == 0 ||
       viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS)
      return false;

   /* The scissor and depth-bias records come first and do not depend on
    * what follows: see the note on agx_hal200_draw_state_pack(). */
   struct agx_ppp_viewport_payload_fields viewport_fields;
   viewport_fields.viewport_count = viewport_count;
   viewport_fields.viewport_control_raw = 0;
   uint32_t raster_control = 0;
   for (unsigned i = 0; i < viewport_count; ++i) {
      const struct agx_hal200_viewport *vp = &state->viewports[i];
      const struct agx_ppp_hal200_raster_fields raster_fields = {
         .scissor_enable = state->scissor_enable,
         .depth_bias_enable = state->depth_bias_enable,
         .max_x = MIN2(vp->max_x, UINT16_MAX),
         .min_x = MIN2(vp->min_x, UINT16_MAX),
         .max_y = MIN2(vp->max_y, UINT16_MAX),
         .min_y = MIN2(vp->min_y, UINT16_MAX),
         .min_z_raw = vp->min_z_raw,
         .max_z_raw = vp->max_z_raw,
         .depth_bias_raw = state->depth_bias_raw,
         .slope_scale_raw = state->slope_scale_raw,
         .clamp_raw = state->depth_bias_clamp_raw,
      };
      struct agx_ppp_hal200_raster_state raster;
      if (!agx_ppp_hal200_raster_state_build(chip, &raster_fields, &raster))
         return false;

      static_assert(sizeof(out->scissors[i]) == sizeof(raster.scissor), "");
      static_assert(sizeof(out->depth_bias[i]) == sizeof(raster.depth_bias),
                    "");
      memcpy(out->scissors[i], raster.scissor, sizeof(raster.scissor));
      memcpy(out->depth_bias[i], raster.depth_bias, sizeof(raster.depth_bias));
      if (i == 0)
         raster_control = raster.fragment_control;

      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = vp->min_x / 32;
         cfg.min_y = vp->min_y / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(vp->max_x, 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(vp->max_y, 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));
      viewport_fields.viewports[i] = vp->transform;
   }

   out->viewport_count = viewport_count;

   uint32_t output_size, output_select, viewport_count_word;
   if (!agx_ppp_hal200_output_size_pack(state->vertex_output_size, 0,
                                        &output_size) ||
       !agx_ppp_hal200_output_select_pack(
          chip, state->vertex_output_select, state->nr_bindings != 0,
          state->fragment_reads_z, &output_select) ||
       !agx_ppp_hal200_viewport_count_pack(chip, viewport_count,
                                           &viewport_count_word))
      return false;

   /* PRIMARY face words: stencil reference, line width, polygon mode and the
    * depth test of each face, each followed by that face's stencil word. */
   const struct agx_hal200_stencil_face *faces[2] = {
      &state->stencil_front, &state->stencil_back,
   };
   uint32_t face[2], stencil[2];
   for (unsigned i = 0; i < 2; ++i) {
      face[i] = faces[i]->reference;
      if (!agx_ppp_hal200_depth_face_pack(
             chip, state->depth_test, state->depth_write,
             state->depth_compare, face[i], &face[i]) ||
          !agx_ppp_hal200_fragment_face_raster_pack(
             chip, state->line_width, state->polygon_mode, face[i], &face[i]))
         return false;

      stencil[i] = agx_ppp_hal200_stencil_pack(
         faces[i]->enabled, faces[i]->compare, faces[i]->fail,
         faces[i]->depth_fail, faces[i]->pass, faces[i]->compare_mask,
         faces[i]->write_mask);
   }

   /* SECONDARY face word, the same for both faces: the depth write disable
    * and ALWAYS comparison that leave the test to the PRIMARY words, the
    * shader's depth contract and the object type. */
   uint32_t face_2 = UINT32_C(0x07200000);
   if (!agx_ppp_hal200_fragment_face_depth_pack(
          chip, state->conservative_depth, state->early_fragment_tests,
          state->fragment_writes_memory, face_2, &face_2) ||
       !agx_ppp_hal200_fragment_face_object_pack(chip, state->object_type,
                                                 face_2, &face_2))
      return false;

   uint32_t secondary_control = UINT32_C(0x00000200);
   uint32_t cull = UINT32_C(0x00000480);
   if (!agx_ppp_hal200_cull_mode_pack(chip, state->cull_mode,
                                      state->front_face_ccw, cull, &cull) ||
       !agx_ppp_hal200_cull_provoking_pack(chip, state->provoking, cull,
                                           &cull) ||
       !agx_ppp_hal200_depth_clip_pack(chip, state->depth_clip, cull, &cull) ||
       !agx_ppp_hal200_rasterizer_discard_pack(
          chip, state->rasterizer_discard, secondary_control, cull,
          &secondary_control, &cull) ||
       !agx_ppp_hal200_primitive_msaa_pack(chip, state->bresenham_lines, cull,
                                           &cull))
      return false;

   /* The pass type goes in the SECONDARY record's first payload word. Both
    * stencil-control bits stay in the same per-draw record as depth. */
   uint32_t fragment_control =
      raster_control | (state->stencil_test ? UINT32_C(0x000c0000) : 0);
   const struct agx_ppp_hal200_fragment_control_fields control_fields = {
      .linked_control = state->fragment_control,
      .shader_tri_merge_disable = state->shader_tri_merge_disable,
      .coverage_tri_merge_disable = state->coverage_tri_merge_disable,
      /* The ISP primitive type selects the runtime override for points,
       * lines and non-filled triangles, then applies it to both controls. */
      .primitive_tri_merge_disable =
         state->object_type != AGX_OBJECT_TYPE_TRIANGLE ||
         state->polygon_mode != AGX_POLYGON_MODE_FILL,
   };
   uint32_t occlusion_query;
   if (!agx_ppp_hal200_fragment_controls_pack(
          chip, &control_fields, fragment_control, secondary_control,
          &fragment_control, &secondary_control) ||
       !agx_ppp_hal200_occlusion_pack(
          chip, state->occlusion_mode, state->occlusion_index,
          fragment_control, &fragment_control, &occlusion_query))
      return false;

   const struct agx_ppp_hal200_graphics_fields output_fields = {
      .words = {output_select, state->mesh_primitive_words,
                viewport_count_word, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, fragment_control},
   };
   const struct agx_ppp_hal200_graphics_fields primary_fields = {
      .words = {fragment_control, face[0], stencil[0], face[1], stencil[1],
                occlusion_query},
   };
   const struct agx_ppp_hal200_graphics_fields secondary_fields = {
      .words = {secondary_control, face_2, face_2, UINT32_C(0x0001ffff)},
   };
   const struct agx_ppp_hal200_graphics_fields cull_fields = {
      .words = {cull},
   };
   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = 0,
      /* The generated primitive-ID coefficient needs its rasterizer input
       * enabled as well as the matching VDM producer and output slot. */
      .cull2_raw = UINT32_C(0x80) |
                   (state->generate_primitive_id ? UINT32_C(0x1000) : 0) |
                   (state->mesh_primitive_address ? UINT32_C(0x2000) : 0),
   };

   struct agx_ppp_hal200_fragment_fields fs_fields;
   struct agx_ppp_hal200_graphics_core fs_record;
   struct agx_ppp_hal200_graphics_core output_record, primary_record;
   struct agx_ppp_hal200_graphics_core secondary_record, cull_record;
   struct agx_ppp_w_clamp_cull2_core w_cull2_record;
   if (!agx_hal200_draw_state_fragment_fields(chip, state, &fs_fields) ||
       !agx_ppp_hal200_fragment_core_pack(chip, &fs_fields, &fs_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output_fields,
          &output_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, &primary_fields,
          &primary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, &secondary_fields,
          &secondary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          chip, AGX_PPP_HAL200_GRAPHICS_CULL, &cull_fields, &cull_record) ||
       !agx_ppp_w_clamp_cull2_core_pack(chip, &w_cull2_fields,
                                        &w_cull2_record) ||
       !agx_ppp_viewport_payload_core_pack(chip, &viewport_fields,
                                           &out->viewport))
      return false;

   /* The physical +0x00..+0x73 layout of a native state block. The 28-byte
    * fragment record occupies +0x00..+0x1b. The output window at +0x1c
    * overlaps the primary header and control at +0x30/+0x34 on purpose. */
   static_assert(AGX_PPP_HAL200_FS_LENGTH == 0x1c,
                 "HAL200 FS must end where the output record begins");
   memset(out->bind_group, 0, sizeof(out->bind_group));
   memcpy(&out->bind_group[0x00], fs_record.bytes, fs_record.length);
   memcpy(&out->bind_group[0x1c], output_record.bytes, output_record.length);
   memcpy(&out->bind_group[0x30], primary_record.bytes, primary_record.length);
   memcpy(&out->bind_group[0x4c], secondary_record.bytes,
          secondary_record.length);
   memcpy(&out->bind_group[0x60], w_cull2_record.bytes, w_cull2_record.length);
   memcpy(&out->bind_group[0x6c], cull_record.bytes, cull_record.length);

   out->varying_counts_32 = state->varying_counts_32;
   out->varying_counts_16 = state->varying_counts_16;
   return true;
}
