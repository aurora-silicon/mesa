/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded G17P/HAL200 fixed-layout PPP graphics-state records.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PPP_HAL200_FS_LENGTH 28u
#define AGX_PPP_HAL200_OUTPUT_LENGTH 28u
#define AGX_PPP_HAL200_PRIMARY_FACE_LENGTH 28u
#define AGX_PPP_HAL200_SECONDARY_FACE_LENGTH 20u
#define AGX_PPP_HAL200_CULL_LENGTH 8u
#define AGX_PPP_HAL200_GRAPHICS_MAX_LENGTH 28u

#define AGX_PPP_HAL200_FS_HEADER UINT32_C(0x00800000)
#define AGX_PPP_HAL200_OUTPUT_HEADER UINT32_C(0x0c0a0000)
#define AGX_PPP_HAL200_PRIMARY_FACE_HEADER UINT32_C(0x010000b4)
#define AGX_PPP_HAL200_SECONDARY_FACE_HEADER UINT32_C(0x02000048)
#define AGX_PPP_HAL200_CULL_HEADER UINT32_C(0x00200000)

enum agx_ppp_hal200_graphics_bundle {
   AGX_PPP_HAL200_GRAPHICS_FS,
   AGX_PPP_HAL200_GRAPHICS_OUTPUT,
   AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE,
   AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE,
   AGX_PPP_HAL200_GRAPHICS_CULL,
};

struct agx_ppp_hal200_graphics_fields {
   uint32_t words[6];
};

/* Native setCullMode/setFrontFacingWinding merge only bits0..1/16. */
static inline bool
agx_ppp_hal200_cull_mode_pack(enum agx_chip chip, unsigned cull_mode,
                              bool front_face_ccw, uint32_t original,
                              uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || cull_mode > 3 || !out)
      return false;

   *out = (original & ~UINT32_C(0x10003)) | cull_mode |
          ((uint32_t)front_face_ccw << 16);
   return true;
}

/* Provoking vertex: CULL bits7..8 encode raw vertex selector+1 independently
 * of VDM and the VIT. */
static inline bool
agx_ppp_hal200_cull_provoking_pack(enum agx_chip chip, unsigned selector,
                                   uint32_t original, uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || selector > 2 || !out)
      return false;

   *out = (original & ~UINT32_C(0x180)) | ((selector + 1) << 7);
   return true;
}

/* Native G17P face-state setters preserve these legacy semantic fields inside
 * the fixed HAL200 PRIMARY/SECONDARY records. */
static inline bool
agx_ppp_hal200_fragment_face_raster_pack(enum agx_chip chip,
                                         unsigned line_width,
                                         unsigned polygon_mode,
                                         uint32_t original, uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || line_width > 0xff || polygon_mode > 2 || !out)
      return false;

   /* The line width is unsigned 4:4 fixed point minus one. The triangle
    * fill mode is a two-bit field at bit 18 of both face words. */
   const uint32_t mask = UINT32_C(0x000cff00);
   *out = (original & ~mask) | (line_width << 8) | (polygon_mode << 18);
   return true;
}

static inline bool
agx_ppp_hal200_fragment_face_object_pack(enum agx_chip chip,
                                         unsigned object_type,
                                         uint32_t original, uint32_t *out)
{
   /* These are the four native Object Type encodings: triangle, line, and
    * the two point-sprite UV conventions. Reject every ungrounded nibble. */
   if (!agx_chip_native_agx3(chip) ||
       (object_type != 0 && object_type != 1 && object_type != 2 &&
        object_type != 4) ||
       !out)
      return false;

   *out = (original & ~UINT32_C(0xf0000000)) | (object_type << 28);
   return true;
}

/* SECONDARY front/back face bits22:23 carry conservative depth. Native
 * late-memory shaders select ANY even with a conservative depth qualifier;
 * explicit early tests select UNCHANGED and ignore shader depth writes. */
static inline unsigned
agx_ppp_fragment_depth_mode(unsigned conservative_depth, bool early_tests,
                           bool writes_memory)
{
   return early_tests ? 3 : writes_memory ? 0 : conservative_depth;
}

static inline bool
agx_ppp_hal200_fragment_face_depth_pack(enum agx_chip chip,
                                        unsigned conservative_depth,
                                        bool early_tests, bool writes_memory,
                                        uint32_t original, uint32_t *out)
{
   if (!out || !agx_chip_native_agx3(chip) || conservative_depth > 3)
      return false;

   /* The fragment program's metadata (root 2, field 4) selects this
    * mode, and it goes into both SECONDARY face words. Isolated native
    * witnesses give ANY=0, GREATER=1, LESS=2, UNCHANGED=3; adding a late
    * atomic to either conservative depth shader changes its mode to ANY. */
   unsigned mode = agx_ppp_fragment_depth_mode(conservative_depth, early_tests,
                                              writes_memory);
   *out = (original & ~UINT32_C(0x00c00000)) | (mode << 22);
   return true;
}

static inline bool
agx_ppp_hal200_primitive_msaa_pack(enum agx_chip chip, bool enable,
                                   uint32_t original, uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || !out)
      return false;

   *out = (original & ~UINT32_C(0x00008000)) |
          (enable ? UINT32_C(0x00008000) : 0);
   return true;
}

/* Native setDepthClipMode: CULL bits11:10 are 01 for clip and 10 for clamp. */
static inline bool
agx_ppp_hal200_depth_clip_pack(enum agx_chip chip, bool depth_clip,
                               uint32_t original, uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || !out)
      return false;

   *out = (original & ~UINT32_C(0xc00)) |
          (depth_clip ? UINT32_C(0x400) : UINT32_C(0x800));
   return true;
}

/* Rasterizer discard is a dual-record contract on HAL200. */
static inline bool
agx_ppp_hal200_rasterizer_discard_pack(enum agx_chip chip, bool discard,
                                       uint32_t secondary_original,
                                       uint32_t cull_original,
                                       uint32_t *secondary_out,
                                       uint32_t *cull_out)
{
   if (!agx_chip_native_agx3(chip) || !secondary_out || !cull_out)
      return false;

   const uint32_t secondary_mask = UINT32_C(1) << 21;
   const uint32_t cull_mask = UINT32_C(1) << 17;
   *secondary_out = (secondary_original & ~secondary_mask) |
                    (discard ? secondary_mask : 0);
   *cull_out = (cull_original & ~cull_mask) |
               (discard ? cull_mask : 0);
   return true;
}

/* Translate the linker's generation-independent fragment control into the
 * two native HAL200 controls.  Triangle merging is not ABI-compatible: the
 * linker's bit 26 describes the legacy helper-invocation policy, while G17P
 * derives its bit from native shader metadata, coverage, and primitive state.
 */
struct agx_ppp_hal200_fragment_control_fields {
   uint32_t linked_control;
   bool shader_tri_merge_disable;
   bool coverage_tri_merge_disable;
   bool primitive_tri_merge_disable;
};

static inline bool
agx_ppp_hal200_fragment_controls_pack(
   enum agx_chip chip,
   const struct agx_ppp_hal200_fragment_control_fields *fields,
   uint32_t primary_original, uint32_t secondary_original,
   uint32_t *primary_out, uint32_t *secondary_out)
{
   if (!agx_chip_native_agx3(chip) || !fields || !primary_out || !secondary_out)
      return false;

   const uint32_t pass_type = fields->linked_control >> 29;
   if (pass_type > 3)
      return false;

   const uint32_t tri_merge_disable = UINT32_C(1) << 26;

   /* These semantic fields go into the HAL200 SECONDARY word.  Bit 26 is
    * deliberately absent: Apple's G17P metadata leaves it clear for implicit
    * texture gradients, explicit gradients, storage writes, and shadow
    * sampling.  The generation-neutral linker sets its legacy bit for all of
    * those cases, so it cannot be copied as native shader metadata. */
   const uint32_t linked_native_mask = UINT32_C(0xe2600000);
   uint32_t secondary = (secondary_original & ~tri_merge_disable) |
                        (fields->linked_control & linked_native_mask);

   if (fields->shader_tri_merge_disable ||
       fields->coverage_tri_merge_disable ||
       fields->primitive_tri_merge_disable)
      secondary |= tri_merge_disable;

   /* encodeAndEmitRenderState applies only the primitive-derived override to
    * both controls.  Shader and coverage policy belong to SECONDARY. */
   uint32_t primary = primary_original & ~tri_merge_disable;
   if (fields->primitive_tri_merge_disable)
      primary |= tri_merge_disable;

   *primary_out = primary;
   *secondary_out = secondary;
   return true;
}

/* Native G17P visibility state: mode occupies fragment-control bits14..15;
 * the 64-bit counter byte offset is encoded as byteOffset<<14. */
static inline bool
agx_ppp_hal200_occlusion_pack(enum agx_chip chip, unsigned mode,
                              unsigned index,
                              uint32_t original_fragment_control,
                              uint32_t *fragment_control,
                              uint32_t *query_offset)
{
   if (!agx_chip_native_agx3(chip) || (mode != 0 && mode != 2 && mode != 3) ||
       index >= 32768 || !fragment_control || !query_offset)
      return false;

   *fragment_control =
      (original_fragment_control & ~UINT32_C(0xc000)) | (mode << 14);
   *query_offset = index << 17;
   return true;
}

struct agx_ppp_hal200_graphics_core {
   uint8_t bytes[AGX_PPP_HAL200_GRAPHICS_MAX_LENGTH];
   uint8_t length;
};

struct agx_ppp_hal200_fragment_fields {
   uint32_t resource_counts;
   /* Absolute 48-bit VIT address; its upper 16 bits occupy counts[15:0].
    * Zero when the fragment has no varyings. The bounded uploader aligns to 64 bytes. */
   uint64_t varying_table;
   uint32_t unknown_0c;
   /* Absolute 48-bit, 64-byte-aligned Ioto/ESL stage-program address. */
   uint64_t stage_program;
   /* Caller-preserved low 22 bits of the stage-program pair's high word. */
   uint32_t stage_program_template;
   uint32_t flags;
};

/* G17P fragment thread-invariant resources, separate from main SPECTPR.
 * Spill bytes become quads in stage-program-template bits 3..20. The native
 * thread_invariant_largest_ipr_bytes field becomes an even quad count in
 * unknown_0c bits 0..11. This is not a temporary-GPR count. Other template
 * bits are preserved; failure leaves fields unchanged. */
bool agx_ppp_hal200_fragment_ti_resources(
   enum agx_chip chip, uint32_t spill_bytes, uint32_t largest_ipr_bytes,
   struct agx_ppp_hal200_fragment_fields *fields);

bool agx_ppp_hal200_graphics_core_supported(enum agx_chip chip);

bool agx_ppp_hal200_graphics_core_pack(
   enum agx_chip chip, enum agx_ppp_hal200_graphics_bundle bundle,
   const struct agx_ppp_hal200_graphics_fields *fields,
   struct agx_ppp_hal200_graphics_core *out);

bool agx_ppp_hal200_graphics_core_unpack(
   enum agx_chip chip, enum agx_ppp_hal200_graphics_bundle bundle,
   const void *bytes, size_t size,
   struct agx_ppp_hal200_graphics_fields *fields);

/* Pack the VDM-referenced G17P fragment record by semantic field. */
bool agx_ppp_hal200_fragment_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal200_fragment_fields *fields,
   struct agx_ppp_hal200_graphics_core *out);

/* Update only native PRIMARY_FACE depth fields, preserving raster state.
 * Compare values use Metal/Vulkan ordering: NEVER=0 through ALWAYS=7.
 *
 * HAL200 depth state: inverted write-enable at bit21 and comparison at
 * bits24..26, in both front and back face state. The comparison field takes
 * the Metal comparison value directly; its values are the same eight
 * comparison operations as Vulkan. No stencil bits move.
 */
static inline bool
agx_ppp_hal200_depth_face_pack(enum agx_chip chip, bool test_enable,
                               bool write_enable, uint32_t compare,
                               uint32_t original, uint32_t *out)
{
   if (!out || !agx_chip_native_agx3(chip) || compare > 7)
      return false;

   const uint32_t mask = UINT32_C(0x07200000);
   const uint32_t function = test_enable ? compare : 7;
   const uint32_t disable_write = !(test_enable && write_enable);
   *out = (original & ~mask) | (function << 24) | (disable_write << 21);
   return true;
}

/* Native stencil word: Vulkan/Metal operation and comparison ordering.
 *
 * G17P stencil face layout, the same for both faces:
 * pass/depth-fail/fail/compare sit at bits 16/19/22/25, respectively. Disabled
 * stencil is ALWAYS+KEEP with zero masks, exactly the native0x0e000000.
 */
static inline uint32_t
agx_ppp_hal200_stencil_pack(bool enabled, unsigned compare, unsigned fail,
                            unsigned depth_fail, unsigned pass,
                            unsigned compare_mask, unsigned write_mask)
{
   if (!enabled)
      return UINT32_C(0x0e000000);

   return (write_mask & 0xff) | ((compare_mask & 0xff) << 8) |
          ((pass & 7) << 16) | ((depth_fail & 7) << 19) |
          ((fail & 7) << 22) | ((compare & 7) << 25);
}

/* Native VsStateConfig+0xc/+0x14 fields; the first includes all f32 UVS. */
static inline bool
agx_ppp_hal200_output_size_pack(uint32_t count_0, uint32_t count_2,
                                uint32_t *out)
{
   if (!out)
      return false;
   *out = 0;
   if (count_0 > 0xff || count_2 > 0xff)
      return false;

   /* The two vertex output counts share this word. The first field includes
    * all position and user f32 words, as proven by Apple vs0/vs2/vs4 metadata
    * giving (4,0)/(6,0)/(8,0). */
   *out = count_0 | (count_2 << 16);
   return true;
}

/* Merge the compiler's vertex OUTPUT_SELECT word with the fragment-owned
 * varying/depth inputs consumed by the HAL200 OUTPUT record. */
static inline bool
agx_ppp_hal200_output_select_pack(enum agx_chip chip,
                                  uint32_t vertex_output_select,
                                  bool fragment_varyings,
                                  bool fragment_reads_z, uint32_t *out)
{
   const uint32_t clip_distances = UINT32_C(0xffff);
   const uint32_t point_size = UINT32_C(1) << 18;
   const uint32_t viewport_target = UINT32_C(1) << 19;
   const uint32_t render_target = UINT32_C(1) << 20;
   const uint32_t vertex_mask = clip_distances | point_size |
                                viewport_target | render_target;

   if (!agx_chip_native_agx3(chip) || !out ||
       (vertex_output_select & ~vertex_mask) ||
       !!(vertex_output_select & viewport_target) !=
          !!(vertex_output_select & render_target))
      return false;

   *out = vertex_output_select |
          (fragment_varyings ? UINT32_C(1) << 16 : 0) |
          (fragment_reads_z ? UINT32_C(1) << 21 : 0);
   return true;
}

/* The vertex state holds viewport_count - 1, and the HAL200 output bundle
 * consumes that word unchanged. */
static inline bool
agx_ppp_hal200_viewport_count_pack(enum agx_chip chip,
                                   unsigned viewport_count, uint32_t *out)
{
   if (!agx_chip_native_agx3(chip) || viewport_count == 0 ||
       viewport_count > 16 || !out)
      return false;

   *out = viewport_count - 1;
   return true;
}

#ifdef __cplusplus
}
#endif
