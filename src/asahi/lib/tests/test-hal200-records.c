/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Golden bytes for the HAL200 records the native AGX3 chips (G15G/S/C, G16G,
 * G17P) share: the PPP graphics records, the raster, viewport and W-clamp
 * records, the bind tail and its state block, and the direct, indexed and
 * indirect draws. Every case is packed for each chip agx_chip_native_agx3()
 * names and must give the same bytes; every other chip must be refused. The
 * expected bytes are what these packers produced on this tree before any of
 * them was touched (define HAL200_RECORDS_PRINT to print them).
 */

#include "agx_g17p_draw_state_upload.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_ppp_hal200_raster_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_vdm_hal200_direct_draw_core.h"

#include <stdio.h>
#include <string.h>

static unsigned failures;
#ifdef HAL200_RECORDS_PRINT
static bool printed;
#endif

#define CHECK(x)                                                              \
   do {                                                                       \
      if (!(x)) {                                                             \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);              \
         failures++;                                                          \
      }                                                                       \
   } while (0)

static void
expect(const char *name, const void *data, size_t size, const char *hex)
{
   const uint8_t *bytes = data;
   char got[2 * 1024 + 1];

   if (size > 1024) {
      fprintf(stderr, "%s: too long\n", name);
      failures++;
      return;
   }
   for (size_t i = 0; i < size; ++i)
      snprintf(&got[2 * i], 3, "%02x", bytes[i]);
   got[2 * size] = 0;
#ifdef HAL200_RECORDS_PRINT
   if (!printed)
      printf("   {\"%s\", \"%s\"},\n", name, got);
   (void)hex;
#else
   if (strcmp(got, hex)) {
      fprintf(stderr, "%s:\n  got  %s\n  want %s\n", name, got, hex);
      failures++;
   }
#endif
}

/* name -> bytes, in the order the cases below run. */
static const struct {
   const char *name, *hex;
} golden[] = {
#include "test-hal200-records-golden.h"
};

static const char *
want(const char *name)
{
   for (unsigned i = 0; i < sizeof(golden) / sizeof(golden[0]); ++i) {
      if (!strcmp(golden[i].name, name))
         return golden[i].hex;
   }
#ifndef HAL200_RECORDS_PRINT
   fprintf(stderr, "no golden bytes for %s\n", name);
   failures++;
#endif
   return "";
}

#define EXPECT(name, data, size) expect(name, data, size, want(name))
#define EXPECT_WORD(name, word)                                               \
   do {                                                                       \
      const uint32_t w_ = (word);                                             \
      const uint8_t b_[4] = {w_, w_ >> 8, w_ >> 16, w_ >> 24};                \
      EXPECT(name, b_, 4);                                                    \
   } while (0)

static void
field_packers(enum agx_chip chip)
{
   uint32_t w, w2;
   char name[64];

   for (unsigned cull = 0; cull < 4; ++cull) {
      for (unsigned ccw = 0; ccw < 2; ++ccw) {
         snprintf(name, sizeof(name), "cull-mode-%u-ccw-%u", cull, ccw);
         CHECK(agx_ppp_hal200_cull_mode_pack(chip, cull, ccw, 0x00000480, &w));
         EXPECT_WORD(name, w);
      }
   }
   CHECK(!agx_ppp_hal200_cull_mode_pack(chip, 4, false, 0x480, &w));

   for (unsigned selector = 0; selector < 3; ++selector) {
      snprintf(name, sizeof(name), "cull-provoking-%u", selector);
      CHECK(agx_ppp_hal200_cull_provoking_pack(chip, selector, 0x00010483,
                                               &w));
      EXPECT_WORD(name, w);
   }
   CHECK(!agx_ppp_hal200_cull_provoking_pack(chip, 3, 0x480, &w));

   CHECK(agx_ppp_hal200_fragment_face_raster_pack(chip, 0x2f, 1, 0x07200f00,
                                                  &w));
   EXPECT_WORD("face-raster-line", w);
   CHECK(agx_ppp_hal200_fragment_face_raster_pack(chip, 0xff, 2, 0x03000012,
                                                  &w));
   EXPECT_WORD("face-raster-point", w);
   CHECK(!agx_ppp_hal200_fragment_face_raster_pack(chip, 0x100, 0, 0, &w));
   CHECK(!agx_ppp_hal200_fragment_face_raster_pack(chip, 0, 3, 0, &w));

   static const unsigned objects[] = {0, 1, 2, 4};
   for (unsigned i = 0; i < 4; ++i) {
      snprintf(name, sizeof(name), "face-object-%u", objects[i]);
      CHECK(agx_ppp_hal200_fragment_face_object_pack(chip, objects[i],
                                                     0x07e00000, &w));
      EXPECT_WORD(name, w);
   }
   CHECK(!agx_ppp_hal200_fragment_face_object_pack(chip, 3, 0, &w));

   for (unsigned depth = 0; depth < 4; ++depth) {
      for (unsigned mode = 0; mode < 3; ++mode) {
         snprintf(name, sizeof(name), "face-depth-%u-%s", depth,
                  mode == 0 ? "plain" : mode == 1 ? "early" : "memory");
         CHECK(agx_ppp_hal200_fragment_face_depth_pack(
            chip, depth, mode == 1, mode == 2, 0x07e00000, &w));
         EXPECT_WORD(name, w);
      }
   }
   CHECK(!agx_ppp_hal200_fragment_face_depth_pack(chip, 4, false, false, 0,
                                                  &w));

   CHECK(agx_ppp_hal200_primitive_msaa_pack(chip, true, 0x480, &w));
   EXPECT_WORD("bresenham-on", w);
   CHECK(agx_ppp_hal200_primitive_msaa_pack(chip, false, 0x8480, &w));
   EXPECT_WORD("bresenham-off", w);

   CHECK(agx_ppp_hal200_depth_clip_pack(chip, true, 0x00000880, &w));
   EXPECT_WORD("depth-clip", w);
   CHECK(agx_ppp_hal200_depth_clip_pack(chip, false, 0x00000480, &w));
   EXPECT_WORD("depth-clamp", w);

   CHECK(agx_ppp_hal200_rasterizer_discard_pack(chip, true, 0x200, 0x480, &w,
                                                &w2));
   EXPECT_WORD("discard-on-secondary", w);
   EXPECT_WORD("discard-on-cull", w2);
   CHECK(agx_ppp_hal200_rasterizer_discard_pack(chip, false, 0x00200200,
                                                0x00020480, &w, &w2));
   EXPECT_WORD("discard-off-secondary", w);
   EXPECT_WORD("discard-off-cull", w2);

   /* Pass type, sample mask after depth/stencil and tag write disable come
    * from the linked control; the linker's own tri-merge bit does not. */
   for (unsigned pass = 0; pass < 4; ++pass) {
      for (unsigned policy = 0; policy < 4; ++policy) {
         const struct agx_ppp_hal200_fragment_control_fields fields = {
            .linked_control = (pass << 29) | 0x06200200,
            .shader_tri_merge_disable = policy == 1,
            .coverage_tri_merge_disable = policy == 2,
            .primitive_tri_merge_disable = policy == 3,
         };
         CHECK(agx_ppp_hal200_fragment_controls_pack(
            chip, &fields, 0x040f0200, 0x04000200, &w, &w2));
         snprintf(name, sizeof(name), "control-pass-%u-policy-%u-primary",
                  pass, policy);
         EXPECT_WORD(name, w);
         snprintf(name, sizeof(name), "control-pass-%u-policy-%u-secondary",
                  pass, policy);
         EXPECT_WORD(name, w2);
      }
   }
   const struct agx_ppp_hal200_fragment_control_fields bad_pass = {
      .linked_control = UINT32_C(4) << 29,
   };
   CHECK(!agx_ppp_hal200_fragment_controls_pack(chip, &bad_pass, 0, 0, &w,
                                                &w2));

   static const unsigned visibility[] = {0, 2, 3};
   for (unsigned i = 0; i < 3; ++i) {
      CHECK(agx_ppp_hal200_occlusion_pack(chip, visibility[i], 0x1234 * i,
                                          0x0005c200, &w, &w2));
      snprintf(name, sizeof(name), "occlusion-%u-control", visibility[i]);
      EXPECT_WORD(name, w);
      snprintf(name, sizeof(name), "occlusion-%u-offset", visibility[i]);
      EXPECT_WORD(name, w2);
   }
   CHECK(!agx_ppp_hal200_occlusion_pack(chip, 1, 0, 0, &w, &w2));
   CHECK(!agx_ppp_hal200_occlusion_pack(chip, 2, 32768, 0, &w, &w2));

   CHECK(agx_ppp_hal200_output_select_pack(chip, 0x001c0007, true, true, &w));
   EXPECT_WORD("output-select-all", w);
   CHECK(agx_ppp_hal200_output_select_pack(chip, 0, false, false, &w));
   EXPECT_WORD("output-select-none", w);
   CHECK(agx_ppp_hal200_output_select_pack(chip, 0x00040000, true, false, &w));
   EXPECT_WORD("output-select-psiz", w);
   /* A viewport target without a render target, and a foreign bit. */
   CHECK(!agx_ppp_hal200_output_select_pack(chip, 0x00080000, false, false,
                                            &w));
   CHECK(!agx_ppp_hal200_output_select_pack(chip, 0x00010000, false, false,
                                            &w));

   CHECK(agx_ppp_hal200_output_size_pack(0x2c, 0x05, &w));
   EXPECT_WORD("output-size", w);
   CHECK(!agx_ppp_hal200_output_size_pack(0x100, 0, &w));

   CHECK(agx_ppp_hal200_viewport_count_pack(chip, 1, &w));
   EXPECT_WORD("viewport-count-1", w);
   CHECK(agx_ppp_hal200_viewport_count_pack(chip, 16, &w));
   EXPECT_WORD("viewport-count-16", w);
   CHECK(!agx_ppp_hal200_viewport_count_pack(chip, 0, &w));
   CHECK(!agx_ppp_hal200_viewport_count_pack(chip, 17, &w));

   for (unsigned compare = 0; compare < 8; ++compare) {
      snprintf(name, sizeof(name), "depth-face-test-write-%u", compare);
      CHECK(agx_ppp_hal200_depth_face_pack(chip, true, true, compare,
                                           0x00042f7f, &w));
      EXPECT_WORD(name, w);
   }
   CHECK(agx_ppp_hal200_depth_face_pack(chip, true, false, 1, 0x07200f00, &w));
   EXPECT_WORD("depth-face-test-nowrite", w);
   CHECK(agx_ppp_hal200_depth_face_pack(chip, false, true, 1, 0x07200f00, &w));
   EXPECT_WORD("depth-face-notest", w);
   CHECK(!agx_ppp_hal200_depth_face_pack(chip, true, true, 8, 0, &w));
}

static void
records(enum agx_chip chip)
{
   static const struct agx_ppp_hal200_graphics_fields words = {
      .words = {0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555,
                0x66666666},
   };
   static const struct {
      enum agx_ppp_hal200_graphics_bundle bundle;
      const char *name;
      unsigned length;
   } bundles[] = {
      {AGX_PPP_HAL200_GRAPHICS_FS, "record-fs", AGX_PPP_HAL200_FS_LENGTH},
      {AGX_PPP_HAL200_GRAPHICS_OUTPUT, "record-output",
       AGX_PPP_HAL200_OUTPUT_LENGTH},
      {AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, "record-primary",
       AGX_PPP_HAL200_PRIMARY_FACE_LENGTH},
      {AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, "record-secondary",
       AGX_PPP_HAL200_SECONDARY_FACE_LENGTH},
      {AGX_PPP_HAL200_GRAPHICS_CULL, "record-cull",
       AGX_PPP_HAL200_CULL_LENGTH},
   };
   for (unsigned i = 0; i < sizeof(bundles) / sizeof(bundles[0]); ++i) {
      struct agx_ppp_hal200_graphics_core record;
      struct agx_ppp_hal200_graphics_fields back;
      CHECK(agx_ppp_hal200_graphics_core_pack(chip, bundles[i].bundle, &words,
                                              &record));
      CHECK(record.length == bundles[i].length);
      EXPECT(bundles[i].name, record.bytes, record.length);
      CHECK(agx_ppp_hal200_graphics_core_unpack(
         chip, bundles[i].bundle, record.bytes, record.length, &back));
      CHECK(!memcmp(back.words, words.words, record.length - 4));
      CHECK(!agx_ppp_hal200_graphics_core_unpack(
         chip, bundles[i].bundle, record.bytes, record.length - 4, &back));
   }

   struct agx_ppp_hal200_fragment_fields fs = {
      .resource_counts = 5 << 16,
      .varying_table = UINT64_C(0x1500ab4c0),
      .stage_program = UINT64_C(0x1007fed6000),
   };
   struct agx_ppp_hal200_graphics_core record;
   CHECK(agx_ppp_hal200_fragment_core_pack(chip, &fs, &record));
   EXPECT("fragment", record.bytes, record.length);
   CHECK(agx_ppp_hal200_fragment_ti_resources(chip, 70, 22, &fs));
   CHECK(agx_ppp_hal200_fragment_core_pack(chip, &fs, &record));
   EXPECT("fragment-ti", record.bytes, record.length);
   CHECK(!agx_ppp_hal200_fragment_ti_resources(chip, 0x100000, 0, &fs));
   fs.varying_table |= 0x20;
   CHECK(!agx_ppp_hal200_fragment_core_pack(chip, &fs, &record));
   fs.varying_table = 0;
   fs.stage_program |= 0x10;
   CHECK(!agx_ppp_hal200_fragment_core_pack(chip, &fs, &record));

   const uint32_t stencil = agx_ppp_hal200_stencil_pack(true, 5, 1, 2, 3, 0xa5,
                                                        0x3c);
   EXPECT_WORD("stencil-on", stencil);
   EXPECT_WORD("stencil-off",
               agx_ppp_hal200_stencil_pack(false, 5, 1, 2, 3, 0xa5, 0x3c));

   const struct agx_ppp_hal200_raster_fields raster_fields = {
      .scissor_enable = true,
      .depth_bias_enable = true,
      .max_x = 1920,
      .min_x = 16,
      .max_y = 1080,
      .min_y = 8,
      .min_z_raw = 0x3e800000,
      .max_z_raw = 0x3f400000,
      .depth_bias_raw = 0x40a00000,
      .slope_scale_raw = 0x3fc00000,
      .clamp_raw = 0x3c23d70a,
   };
   struct agx_ppp_hal200_raster_state raster;
   CHECK(agx_ppp_hal200_raster_state_build(chip, &raster_fields, &raster));
   EXPECT_WORD("raster-control", raster.fragment_control);
   EXPECT("raster-scissor", raster.scissor, sizeof(raster.scissor));
   EXPECT("raster-depth-bias", raster.depth_bias, sizeof(raster.depth_bias));
   const struct agx_ppp_hal200_raster_fields no_raster = {0};
   CHECK(agx_ppp_hal200_raster_state_build(chip, &no_raster, &raster));
   EXPECT_WORD("raster-control-plain", raster.fragment_control);

   struct agx_ppp_viewport_payload_fields viewports = {
      .viewport_count = 1,
      .region_clip_raw = {UINT64_C(0x0001002180000021)},
      .viewports = {{0x44700000, 0x44700000, 0x44070000, 0xc4070000,
                     0x00000000, 0x3f800000}},
   };
   struct agx_ppp_viewport_payload_core viewport;
   CHECK(agx_ppp_viewport_payload_core_pack(chip, &viewports, &viewport));
   CHECK(viewport.length == AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH);
   EXPECT("viewport-1", viewport.bytes, viewport.length);
   viewports.viewport_count = 3;
   viewports.region_clip_raw[1] = UINT64_C(0x0002000180030004);
   viewports.region_clip_raw[2] = UINT64_C(0x0000001f8000003b);
   viewports.viewports[1] = (struct agx_ppp_viewport_transform_raw){
      0x42000000, 0x42000000, 0x42800000, 0x42800000, 0x3dcccccd, 0x3f4ccccd,
   };
   viewports.viewports[2] = (struct agx_ppp_viewport_transform_raw){
      1, 2, 3, 4, 5, 6,
   };
   CHECK(agx_ppp_viewport_payload_core_pack(chip, &viewports, &viewport));
   CHECK(viewport.length == 8 + 3 * 32);
   EXPECT("viewport-3", viewport.bytes, viewport.length);
   struct agx_ppp_viewport_payload_fields back;
   CHECK(agx_ppp_viewport_payload_core_unpack(chip, viewport.bytes,
                                              viewport.length, &back));
   CHECK(back.viewport_count == 3 &&
         !memcmp(back.viewports, viewports.viewports,
                 3 * sizeof(viewports.viewports[0])) &&
         !memcmp(back.region_clip_raw, viewports.region_clip_raw,
                 3 * sizeof(viewports.region_clip_raw[0])));
   viewports.viewport_count = 0;
   CHECK(!agx_ppp_viewport_payload_core_pack(chip, &viewports, &viewport));
   viewports.viewport_count = 17;
   CHECK(!agx_ppp_viewport_payload_core_pack(chip, &viewports, &viewport));

   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = 0x322bcc77,
      .cull2_raw = 0x3080,
   };
   struct agx_ppp_w_clamp_cull2_core w_cull2;
   CHECK(agx_ppp_w_clamp_cull2_core_pack(chip, &w_cull2_fields, &w_cull2));
   EXPECT("w-clamp-cull2", w_cull2.bytes, w_cull2.length);
}

static void
bind_tail(enum agx_chip chip)
{
   const struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT] = {
      {0x00812ac0, 0x700}, {0x00812a00, 0x500}, {0x00812a1c, 0x700},
      {0x00812a30, 0x500}, {0x00812a4c, 0x0a00}, {0x00812a80, 0x300},
      {0x00812a60, 0x200}, {0x00812a6c, 0x200},
   };
   struct agx_vdm_hal200_bind_pair back[AGX_VDM_HAL200_BIND_COUNT];
   struct agx_vdm_hal200_bind_tail tail;
   CHECK(agx_vdm_hal200_bind_tail_pack(chip, pairs, &tail));
   CHECK(tail.length == AGX_VDM_HAL200_BIND_TAIL_LENGTH);
   EXPECT("bind-tail", tail.bytes, tail.length);
   CHECK(agx_vdm_hal200_bind_tail_unpack(chip, tail.bytes, tail.length, back));
   CHECK(!memcmp(back, pairs, sizeof(pairs)));

   char name[64];
   for (unsigned count = 1; count <= 16; count += 5) {
      snprintf(name, sizeof(name), "viewport-control-%u", count);
      EXPECT_WORD(name, agx_vdm_hal200_bind_viewport_control(8 + 32 * count));
   }
   CHECK(agx_vdm_hal200_bind_viewport_control(41) == 0);

   /* The state block and the tail that points into it, for one and for
    * sixteen viewports. */
   uint8_t group[AGX_G17P_BIND_GROUP_BYTES];
   uint8_t viewport[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   for (unsigned i = 0; i < sizeof(group); ++i)
      group[i] = (uint8_t)(i * 3 + 1);
   for (unsigned i = 0; i < sizeof(viewport); ++i)
      viewport[i] = (uint8_t)(i * 7 + 5);
   static const unsigned counts[] = {1, 16};
   for (unsigned i = 0; i < 2; ++i) {
      const size_t viewport_size = 8 + 32 * counts[i];
      const size_t size = agx_g17p_bind_state_size(viewport_size);
      uint8_t block[AGX_G17P_BIND_GROUP_BYTES +
                    AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH +
                    AGX_G17P_VARYING_RECORD_BYTES];
      struct agx_g17p_bind_state_upload upload;
      CHECK(size == sizeof(group) + viewport_size + 20);
      CHECK(agx_g17p_upload_bind_state(block, size, 0x00c40040, group, viewport,
                                       viewport_size, 0x00030201, 0x00000004,
                                       &upload));
      snprintf(name, sizeof(name), "state-block-%u", counts[i]);
      EXPECT(name, block, size);
      snprintf(name, sizeof(name), "state-tail-%u", counts[i]);
      EXPECT(name, upload.tail.bytes, upload.tail.length);
      CHECK(!agx_g17p_upload_bind_state(block, size - 1, 0x00c40040, group,
                                        viewport, viewport_size, 0, 0,
                                        &upload));
      CHECK(!agx_g17p_upload_bind_state(block, size, 0x00c40040, group,
                                        viewport, viewport_size, 0x01000000, 0,
                                        &upload));
      CHECK(!agx_g17p_upload_bind_state(block, size, 0x00000040, group,
                                        viewport, viewport_size, 0, 0,
                                        &upload));
   }
   CHECK(agx_g17p_bind_state_size(44) == 0);

   uint8_t indices[8];
   uint32_t offset = 0;
   CHECK(agx_g17p_upload_draw_indices(indices, sizeof(indices), 0x00c40100,
                                      16 * 21, 12 * 9, 4, &offset));
   CHECK(offset == 0x00c40100);
   EXPECT("draw-indices", indices, sizeof(indices));
   CHECK(!agx_g17p_upload_draw_indices(indices, sizeof(indices), 0x00c40100,
                                       16 * 3, 12 * 9, 4, &offset));
}

static void
draws(enum agx_chip chip)
{
   char name[64];

   for (unsigned primitive = 0; primitive < 16; ++primitive) {
      const struct agx_vdm_hal200_direct_draw_fields fields = {
         .draw_state_context_offset = 0x00c40100,
         .primitive = primitive,
         .vertex_count = 36 + primitive,
         .instance_count = 1 + primitive,
         .vertex_start = 7 * primitive,
      };
      struct agx_vdm_hal200_direct_draw draw;
      struct agx_vdm_hal200_direct_draw_body body;
      if (!agx_vdm_hal200_direct_draw_pack(chip, &fields, &draw)) {
         CHECK(!agx_vdm_hal200_direct_draw_body_pack(chip, &fields, &body));
         snprintf(name, sizeof(name), "direct-draw-%u-refused", primitive);
         EXPECT(name, "", 0);
         continue;
      }
      snprintf(name, sizeof(name), "direct-draw-%u", primitive);
      EXPECT(name, draw.bytes, draw.length);
      CHECK(agx_vdm_hal200_direct_draw_body_pack(chip, &fields, &body));
      CHECK(draw.length == AGX_VDM_HAL200_DIRECT_DRAW_LENGTH &&
            body.length == AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH &&
            !memcmp(body.bytes, &draw.bytes[4], body.length));

      /* The stream gets the selector only when render state was published. */
      uint8_t stream[32];
      size_t emitted = 0;
      CHECK(agx_vdm_hal200_draw_emit(chip, draw.bytes, draw.length, true,
                                     stream, sizeof(stream), &emitted));
      CHECK(emitted == draw.length && !memcmp(stream, draw.bytes, emitted));
      CHECK(agx_vdm_hal200_draw_emit(chip, draw.bytes, draw.length, false,
                                     stream, sizeof(stream), &emitted));
      CHECK(emitted == body.length && !memcmp(stream, body.bytes, emitted));
   }

   /* Indexed: every index size, starts that are and are not a multiple of
    * four bytes into the buffer, restart on and off, a negative base. */
   static const struct {
      uint8_t size_B;
      uint64_t buffer;
      uint32_t first, count;
      bool restart;
      int32_t base_vertex;
   } indexed[] = {
      {1, UINT64_C(0x15000c000), 0, 36, false, 0},
      {1, UINT64_C(0x15000c001), 0, 36, false, 0},
      {1, UINT64_C(0x15000c000), 3, 33, true, 5},
      {1, UINT64_C(0x15000c003), 2, 7, false, -3},
      {2, UINT64_C(0x15000c000), 0, 36, false, 0},
      {2, UINT64_C(0x15000c002), 0, 36, true, 0},
      {2, UINT64_C(0x15000c000), 1, 35, false, 100},
      {2, UINT64_C(0x15000c006), 5, 9, false, 0},
      {4, UINT64_C(0x15000c000), 0, 36, false, 0},
      {4, UINT64_C(0x15000c004), 9, 27, true, -1},
   };
   for (unsigned i = 0; i < sizeof(indexed) / sizeof(indexed[0]); ++i) {
      const struct agx_vdm_hal200_indexed_draw_fields fields = {
         .primitive = 3,
         .index_size_B = indexed[i].size_B,
         .restart_enable = indexed[i].restart,
         .restart_index = indexed[i].size_B == 1   ? 0xff
                          : indexed[i].size_B == 2 ? 0xffff
                                                   : 0xffffffff,
         .draw_state_context_offset = 0x00c40100,
         .index_buffer = indexed[i].buffer,
         .index_buffer_range_B = 4096,
         .first_index = indexed[i].first,
         .index_count = indexed[i].count,
         .instance_count = 2,
         .base_vertex = indexed[i].base_vertex,
      };
      struct agx_vdm_hal200_indexed_draw draw;
      struct agx_vdm_hal200_indexed_draw_body body;
      uint64_t relocated = 0;
      uint32_t size_word = 0;
      CHECK(agx_vdm_hal200_indexed_draw_pack(chip, &fields, &draw));
      snprintf(name, sizeof(name), "indexed-draw-%u", i);
      EXPECT(name, draw.bytes, draw.length);
      CHECK(agx_vdm_hal200_indexed_draw_body_pack(chip, &fields, &body));
      CHECK(draw.length == AGX_VDM_HAL200_INDEXED_DRAW_LENGTH &&
            body.length == AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH &&
            !memcmp(body.bytes, &draw.bytes[4], body.length));
      CHECK(agx_vdm_hal200_index_range(fields.index_buffer,
                                       fields.index_buffer_range_B,
                                       fields.first_index, fields.index_count,
                                       fields.index_size_B, &relocated,
                                       &size_word));
      snprintf(name, sizeof(name), "index-range-%u", i);
      const uint32_t range[3] = {(uint32_t)relocated,
                                 (uint32_t)(relocated >> 32), size_word};
      uint8_t range_bytes[12];
      for (unsigned b = 0; b < 12; ++b)
         range_bytes[b] = range[b / 4] >> (8 * (b % 4));
      EXPECT(name, range_bytes, sizeof(range_bytes));
   }
   uint64_t relocated;
   uint32_t size_word;
   CHECK(!agx_vdm_hal200_index_range(UINT64_C(0x15000c000), 64, 30, 6, 2,
                                     &relocated, &size_word));
   CHECK(!agx_vdm_hal200_index_range(UINT64_C(0x15000c000), 64, 0, 6, 3,
                                     &relocated, &size_word));
   /* An index buffer must be aligned to its index size. */
   CHECK(!agx_vdm_hal200_index_range(UINT64_C(0x15000c003), 64, 0, 6, 2,
                                     &relocated, &size_word));
   CHECK(!agx_vdm_hal200_index_range(UINT64_C(0x15000c002), 64, 0, 6, 4,
                                     &relocated, &size_word));

   for (unsigned i = 0; i < 4; ++i) {
      const bool is_indexed = i != 0;
      const struct agx_vdm_hal200_indirect_draw_fields fields = {
         .draw_state_context_offset = 0x00c40100,
         .primitive = 3,
         .indexed = is_indexed,
         .index_size_B = is_indexed ? 1 << (i - 1) : 0,
         .restart_enable = i == 2,
         .restart_index = 0xffff,
         .index_buffer = is_indexed ? UINT64_C(0x15000c000) : 0,
         .index_buffer_range_B = is_indexed ? 4096 : 0,
         .indirect_buffer = UINT64_C(0x150010040),
      };
      struct agx_vdm_hal200_indirect_draw draw;
      CHECK(agx_vdm_hal200_indirect_draw_pack(chip, &fields, &draw));
      CHECK(draw.length == (is_indexed
                               ? AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH
                               : AGX_VDM_HAL200_INDIRECT_NONINDEXED_DRAW_LENGTH));
      snprintf(name, sizeof(name), "indirect-draw-%u", i);
      EXPECT(name, draw.bytes, draw.length);
   }
}

static void
refused(enum agx_chip chip)
{
   const struct agx_ppp_hal200_graphics_fields words = {{0}};
   const struct agx_ppp_hal200_fragment_fields fs = {
      .stage_program = UINT64_C(0x1007fed6000),
   };
   const struct agx_ppp_hal200_raster_fields raster_fields = {0};
   const struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT] = {
      {0, 0x700}, {0, 0x500}, {0, 0x700}, {0, 0x500},
      {0, 0x0a00}, {0, 0x300}, {0, 0x200}, {0, 0x200},
   };
   const struct agx_vdm_hal200_direct_draw_fields direct = {
      .primitive = 3, .vertex_count = 3, .instance_count = 1,
   };
   const struct agx_vdm_hal200_indexed_draw_fields indexed = {
      .primitive = 3, .index_size_B = 2,
      .index_buffer = UINT64_C(0x15000c000), .index_buffer_range_B = 64,
      .index_count = 3, .instance_count = 1,
   };
   struct agx_ppp_hal200_graphics_core record;
   struct agx_ppp_hal200_raster_state raster;
   struct agx_vdm_hal200_bind_tail tail;
   struct agx_vdm_hal200_direct_draw direct_draw;
   struct agx_vdm_hal200_indexed_draw indexed_draw;
   uint32_t w, w2;

   CHECK(!agx_ppp_hal200_graphics_core_supported(chip));
   CHECK(!agx_ppp_hal200_graphics_core_pack(
      chip, AGX_PPP_HAL200_GRAPHICS_CULL, &words, &record));
   CHECK(!agx_ppp_hal200_fragment_core_pack(chip, &fs, &record));
   CHECK(!agx_ppp_hal200_cull_mode_pack(chip, 0, false, 0, &w));
   CHECK(!agx_ppp_hal200_cull_provoking_pack(chip, 0, 0, &w));
   CHECK(!agx_ppp_hal200_fragment_face_raster_pack(chip, 0, 0, 0, &w));
   CHECK(!agx_ppp_hal200_fragment_face_object_pack(chip, 0, 0, &w));
   CHECK(!agx_ppp_hal200_fragment_face_depth_pack(chip, 0, false, false, 0,
                                                  &w));
   CHECK(!agx_ppp_hal200_primitive_msaa_pack(chip, false, 0, &w));
   CHECK(!agx_ppp_hal200_depth_clip_pack(chip, true, 0, &w));
   CHECK(!agx_ppp_hal200_rasterizer_discard_pack(chip, false, 0, 0, &w, &w2));
   CHECK(!agx_ppp_hal200_occlusion_pack(chip, 0, 0, 0, &w, &w2));
   CHECK(!agx_ppp_hal200_output_select_pack(chip, 0, false, false, &w));
   CHECK(!agx_ppp_hal200_viewport_count_pack(chip, 1, &w));
   CHECK(!agx_ppp_hal200_depth_face_pack(chip, false, false, 0, 0, &w));
   CHECK(!agx_ppp_hal200_raster_state_build(chip, &raster_fields, &raster));
   CHECK(!agx_vdm_hal200_bind_tail_pack(chip, pairs, &tail));
   CHECK(!agx_vdm_hal200_direct_draw_pack(chip, &direct, &direct_draw));
   CHECK(!agx_vdm_hal200_indexed_draw_pack(chip, &indexed, &indexed_draw));
}

int
main(void)
{
   /* Which chips take these records is agx_chip_native_agx3()'s to say. */
   for (unsigned c = 0; c < AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = (enum agx_chip)c;
      if (!agx_chip_native_agx3(chip)) {
         refused(chip);
         continue;
      }

      field_packers(chip);
      records(chip);
      bind_tail(chip);
      draws(chip);
#ifdef HAL200_RECORDS_PRINT
      printed = true;
#endif
   }

   return failures != 0;
}
