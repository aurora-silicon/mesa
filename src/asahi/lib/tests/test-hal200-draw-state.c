/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx_hal200_draw_state_pack() against the state blocks Borealis ("vk") and
 * the Gallium driver ("gl") emitted before either used it. Each vector is a
 * description and the bytes the driver's own builder produced for it under
 * the DRM shim: the bind group, the viewport record, then the scissor and
 * depth-bias records. They were chosen from dEQP-VK and an offscreen GL test
 * so that every value each driver gave a field appears at least once.
 */

#include "agx_hal200_draw_state.h"

#include <stdio.h>
#include <string.h>

static const struct {
   /* The chip the driver ran as. */
   enum agx_chip chip;
   const char *driver;
   struct agx_hal200_draw_state state;
   const char *bytes;
} vectors[] = {
#include "test-hal200-draw-state-vectors.h"
};

static unsigned failures;

#define CHECK(x)                                                              \
   do {                                                                       \
      if (!(x)) {                                                             \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);              \
         failures++;                                                          \
      }                                                                       \
   } while (0)

static size_t
flatten(const struct agx_hal200_draw_state_packed *packed, uint8_t *out)
{
   size_t size = 0;
   memcpy(out, packed->bind_group, sizeof(packed->bind_group));
   size += sizeof(packed->bind_group);
   memcpy(out + size, packed->viewport.bytes, packed->viewport.length);
   size += packed->viewport.length;
   memcpy(out + size, packed->scissors,
          packed->viewport_count * sizeof(packed->scissors[0]));
   size += packed->viewport_count * sizeof(packed->scissors[0]);
   memcpy(out + size, packed->depth_bias,
          packed->viewport_count * sizeof(packed->depth_bias[0]));
   size += packed->viewport_count * sizeof(packed->depth_bias[0]);
   return size;
}

int
main(void)
{
   for (unsigned v = 0; v < sizeof(vectors) / sizeof(vectors[0]); ++v) {
      const struct agx_hal200_draw_state *state = &vectors[v].state;
      uint8_t bytes[sizeof(struct agx_hal200_draw_state_packed)];
      char hex[2 * sizeof(bytes) + 1];

      /* The chips agx_chip_native_agx3() names share every record, so the
       * bytes one of them was given are the bytes all of them get. Every
       * other chip is refused. */
      for (unsigned c = 0; c < AGX_CHIP_UNKNOWN; ++c) {
         const enum agx_chip chip = (enum agx_chip)c;
         struct agx_hal200_draw_state_packed packed;
         if (!agx_chip_native_agx3(chip)) {
            CHECK(!agx_hal200_draw_state_pack(chip, state, &packed));
            continue;
         }

         if (!agx_hal200_draw_state_pack(chip, state, &packed)) {
            fprintf(stderr, "vector %u (%s) refused on chip %u\n", v,
                    vectors[v].driver, chip);
            failures++;
            continue;
         }

         const size_t size = flatten(&packed, bytes);
         for (size_t i = 0; i < size; ++i)
            snprintf(&hex[2 * i], 3, "%02x", bytes[i]);
         if (strcmp(hex, vectors[v].bytes)) {
            fprintf(stderr, "vector %u (%s) on chip %u:\n  got  %s\n  want %s\n",
                    v, vectors[v].driver, chip, hex, vectors[v].bytes);
            failures++;
         }

         /* The fragment record packed again from its fields, as a driver
          * does for a new fragment program. */
         struct agx_ppp_hal200_fragment_fields fragment;
         struct agx_ppp_hal200_graphics_core record;
         CHECK(agx_hal200_draw_state_fragment_fields(chip, state,
                                                     &fragment));
         CHECK(agx_ppp_hal200_fragment_core_pack(chip, &fragment,
                                                 &record));
         CHECK(record.length == AGX_PPP_HAL200_FS_LENGTH &&
               !memcmp(record.bytes, packed.bind_group, record.length));

         /* The upload is the state block followed by the varying record,
          * with a tail that points into it. */
         uint8_t block[AGX_G17P_BIND_GROUP_BYTES +
                       AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH +
                       AGX_G17P_VARYING_RECORD_BYTES];
         struct agx_vdm_hal200_bind_tail tail;
         struct agx_vdm_hal200_bind_pair pairs[AGX_VDM_HAL200_BIND_COUNT];
         const size_t block_size = agx_hal200_draw_state_size(&packed);
         CHECK(block_size == sizeof(packed.bind_group) +
                                packed.viewport.length +
                                AGX_G17P_VARYING_RECORD_BYTES);
         CHECK(agx_hal200_draw_state_upload(&packed, block, block_size,
                                            0x00c40040, &tail));
         CHECK(!memcmp(block, bytes,
                       sizeof(packed.bind_group) + packed.viewport.length));
         CHECK(agx_vdm_hal200_bind_tail_unpack(chip, tail.bytes,
                                               tail.length, pairs));
         CHECK(pairs[1].context_offset == 0x00c40040 &&
               pairs[0].context_offset ==
                  0x00c40040 + block_size - AGX_G17P_VARYING_RECORD_BYTES);
      }

   }

   /* A state the records cannot express is refused, after the scissor and
    * depth-bias records were written. */
   struct agx_hal200_draw_state state = vectors[0].state;
   struct agx_hal200_draw_state_packed good, packed;
   CHECK(agx_hal200_draw_state_pack(vectors[0].chip, &state, &good));

   state.viewport_count = 0;
   CHECK(!agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed));
   CHECK(packed.viewport_count == 0);
   state.viewport_count = AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS + 1;
   CHECK(!agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed));
   CHECK(packed.viewport_count == 0);

   static const struct {
      size_t offset;
      uint32_t value;
   } bad[] = {
      {offsetof(struct agx_hal200_draw_state, fragment_control),
       UINT32_C(4) << 29},
      {offsetof(struct agx_hal200_draw_state, vertex_output_select), 0x80000},
      {offsetof(struct agx_hal200_draw_state, vertex_output_size), 0x100},
      {offsetof(struct agx_hal200_draw_state, conservative_depth), 4},
      {offsetof(struct agx_hal200_draw_state, depth_compare), 8},
      {offsetof(struct agx_hal200_draw_state, line_width), 0x100},
      {offsetof(struct agx_hal200_draw_state, polygon_mode), 3},
      {offsetof(struct agx_hal200_draw_state, object_type), 3},
      {offsetof(struct agx_hal200_draw_state, cull_mode), 4},
      {offsetof(struct agx_hal200_draw_state, provoking), 3},
      {offsetof(struct agx_hal200_draw_state, occlusion_mode), 1},
      {offsetof(struct agx_hal200_draw_state, occlusion_index), 32768},
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      state = vectors[0].state;
      memcpy((uint8_t *)&state + bad[i].offset, &bad[i].value,
             sizeof(bad[i].value));
      memset(&packed, 0xa5, sizeof(packed));
      if (agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed)) {
         fprintf(stderr, "bad state %u accepted\n", i);
         failures++;
      }
      CHECK(packed.viewport_count == good.viewport_count);
      CHECK(!memcmp(packed.scissors[0], good.scissors[0],
                    sizeof(good.scissors[0])));
      CHECK(!memcmp(packed.depth_bias[0], good.depth_bias[0],
                    sizeof(good.depth_bias[0])));
   }

   state = vectors[0].state;
   state.fragment_program |= 0x20;
   CHECK(!agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed));
   state = vectors[0].state;
   state.varying_table |= 0x20;
   CHECK(!agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed));
   state = vectors[0].state;
   state.fragment_ti_resources = true;
   state.fragment_ti_spill_bytes = 0x100000;
   CHECK(!agx_hal200_draw_state_pack(vectors[0].chip, &state, &packed));

   return failures != 0;
}
