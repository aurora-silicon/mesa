/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ppp_hal200_graphics_core.h"
#include "agx_bytes.h"
#include "agx_vdm_state_core.h"

#include <string.h>

static bool
bundle_layout(enum agx_ppp_hal200_graphics_bundle bundle, uint32_t *header,
              unsigned *payload_words)
{
   switch (bundle) {
   case AGX_PPP_HAL200_GRAPHICS_FS:
      *header = AGX_PPP_HAL200_FS_HEADER;
      *payload_words = 6;
      return true;
   case AGX_PPP_HAL200_GRAPHICS_OUTPUT:
      *header = AGX_PPP_HAL200_OUTPUT_HEADER;
      *payload_words = 6;
      return true;
   case AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE:
      *header = AGX_PPP_HAL200_PRIMARY_FACE_HEADER;
      *payload_words = 6;
      return true;
   case AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE:
      *header = AGX_PPP_HAL200_SECONDARY_FACE_HEADER;
      *payload_words = 4;
      return true;
   case AGX_PPP_HAL200_GRAPHICS_CULL:
      *header = AGX_PPP_HAL200_CULL_HEADER;
      *payload_words = 1;
      return true;
   default:
      *header = 0;
      *payload_words = 0;
      return false;
   }
}

/*
 * Pack one complete fixed-layout G17P record.  Each record has its own header
 * and outer VDM PPP_STATE pointer; records must never be concatenated behind a
 * single legacy selector header.
 *
 * Payload order and word count by bundle:
 *
 *   FS:             counts, VIT, unknown-0c, stage-program-high,
 *                   stage-program-low, flags
 *   OUTPUT:         output-select, varying-counts-16, output-unknown,
 *                   output-size, primary-face header, fragment-control
 *                   (the traced seven-word window deliberately overlaps the
 *                   first two words of the following primary-face record)
 *   PRIMARY_FACE:   fragment-control, front-face, front-stencil, back-face,
 *                   back-stencil, occlusion-query
 *   SECONDARY_FACE: fragment-control-2, front-face-2, back-face-2,
 *                   occlusion-query-2
 *   CULL:           cull
 */
bool
agx_ppp_hal200_graphics_core_supported(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip);
}

bool
agx_ppp_hal200_graphics_core_pack(
   enum agx_chip chip, enum agx_ppp_hal200_graphics_bundle bundle,
   const struct agx_ppp_hal200_graphics_fields *fields,
   struct agx_ppp_hal200_graphics_core *out)
{
   uint32_t header = 0;
   unsigned payload_words = 0;

   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || !agx_ppp_hal200_graphics_core_supported(chip) ||
       !bundle_layout(bundle, &header, &payload_words))
      return false;

   agx_store_le32(out->bytes, header);
   for (unsigned i = 0; i < payload_words; ++i)
      agx_store_le32(&out->bytes[4 + (4 * i)], fields->words[i]);

   out->length = (uint8_t)(4 + (4 * payload_words));
   return true;
}

bool
agx_ppp_hal200_graphics_core_unpack(
   enum agx_chip chip, enum agx_ppp_hal200_graphics_bundle bundle,
   const void *bytes, size_t size,
   struct agx_ppp_hal200_graphics_fields *fields)
{
   uint32_t header = 0;
   unsigned payload_words = 0;

   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!bytes || !fields ||
       !agx_ppp_hal200_graphics_core_supported(chip) ||
       !bundle_layout(bundle, &header, &payload_words) ||
       size != 4 + (4 * payload_words))
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(raw) != header)
      return false;

   for (unsigned i = 0; i < payload_words; ++i)
      fields->words[i] = agx_load_le32(&raw[4 + (4 * i)]);

   return true;
}

bool
agx_ppp_hal200_fragment_ti_resources(
   enum agx_chip chip, uint32_t spill_bytes, uint32_t largest_ipr_bytes,
   struct agx_ppp_hal200_fragment_fields *fields)
{
   const uint64_t spill_quads = ((uint64_t)spill_bytes + 3) >> 2;
   const uint64_t ipr_quads = ((((uint64_t)largest_ipr_bytes + 3) >> 2) + 1) &
                             ~UINT64_C(1);
   if (!agx_chip_native_agx3(chip) || !fields || spill_quads > 0x3ffff ||
       ipr_quads > 0xffe)
      return false;

   /* G17P fragment record: the TI spill quad count and the rounded TI
    * largest-IPR count go into the seven-word fragment record, IPR in word 3
    * and spill in word 4. */
   fields->unknown_0c = (fields->unknown_0c & ~UINT32_C(0xfff)) | ipr_quads;
   fields->stage_program_template =
      (fields->stage_program_template & ~UINT32_C(0x1ffff8)) | (spill_quads << 3);
   return true;
}

bool
agx_ppp_hal200_fragment_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal200_fragment_fields *fields,
   struct agx_ppp_hal200_graphics_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!fields || !out || (fields->varying_table & UINT64_C(0x3f)) != 0 ||
       fields->varying_table >= (UINT64_C(1) << 48) ||
       (fields->resource_counts & UINT32_C(0xffff)) != 0)
      return false;

   struct agx_vdm_state_vertex_pipeline_pair stage_program;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->stage_program, fields->stage_program_template,
          &stage_program))
      return false;

   const struct agx_ppp_hal200_graphics_fields raw = {
      .words = {
         /* The HAL200 fragment record merges the absolute VIT pair into its
          * first two words. */
         fields->resource_counts | (uint32_t)(fields->varying_table >> 32),
         (uint32_t)fields->varying_table,
         fields->unknown_0c,
         agx_load_le32(&stage_program.bytes[0]),
         agx_load_le32(&stage_program.bytes[4]),
         fields->flags,
      },
   };
   return agx_ppp_hal200_graphics_core_pack(
      chip, AGX_PPP_HAL200_GRAPHICS_FS, &raw, out);
}
