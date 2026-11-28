/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_native_descriptor.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Gallium-specific descriptor constants, sampler bank layout, and thin
 * adapters over src/asahi/lib/agx_native_descriptor.h.
 */
#define AGX_GALLIUM_DESCRIPTOR_AGX2_SIZE AGX_NATIVE_DESCRIPTOR_AGX2_SIZE
#define AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE AGX_NATIVE_DESCRIPTOR_AGX3_SIZE
#define AGX_GALLIUM_DESCRIPTOR_MAX_SIZE  AGX_NATIVE_DESCRIPTOR_MAX_SIZE
#define AGX_GALLIUM_SAMPLER_AGX3_SIZE    AGX_NATIVE_SAMPLER_AGX3_SIZE
#define AGX_GALLIUM_G17P_API_SAMPLERS    32u

/* Custom borders are implemented like Honeykrisp: the ordinary API sampler
 * bank clamps to one and a second bank clamps to zero. The shader combines
 * those samples with the exact API border colour. */
#define AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE \
   (1u + AGX_GALLIUM_G17P_API_SAMPLERS)

/* LOD queries cannot carry the software bias used by ordinary samples.
 * Their private sampler bank keeps the API bias in the native descriptor. */
#define AGX_GALLIUM_G17P_QUERY_LOD_BASE \
   (AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + AGX_GALLIUM_G17P_API_SAMPLERS)

/* Shadow gathers need an unfiltered border mask, independent of the depth
 * comparison. The zero companion is already non-comparing; this is its
 * clamp-to-one partner. Keep the ordinary comparison sampler unchanged. */
#define AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE \
   (AGX_GALLIUM_G17P_QUERY_LOD_BASE + AGX_GALLIUM_G17P_API_SAMPLERS)

/* HAL200's immediate sampler operand names native slots 0..15. Gallium keeps
 * slot zero for texel fetch, leaving API samplers 0..14 directly encodable.
 * Later API samplers use an indirect native-table index instead. */
#define AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS 15u

#define agx_gallium_texture_descriptor agx_native_texture_descriptor
#define agx_gallium_pbe_descriptor     agx_native_pbe_descriptor
#define agx_gallium_sampler_descriptor agx_native_sampler_descriptor
#define agx_gallium_native_sampler     agx_native_sampler_state

struct agx_gallium_storage_descriptor_layout {
   uint8_t texture;
   uint8_t pbe;
   uint8_t stride;
};

static inline bool
agx_gallium_descriptor_generation_is_agx2(
   enum agx_descriptor_generation generation)
{
   return agx_descriptor_generation_is_agx2(generation);
}

static inline bool
agx_gallium_descriptor_generation_is_agx3(
   enum agx_descriptor_generation generation)
{
   return agx_descriptor_generation_is_agx3(generation);
}

static inline unsigned
agx_gallium_texture_descriptor_size(enum agx_descriptor_generation generation)
{
   return agx_texture_descriptor_size(generation);
}

static inline unsigned
agx_gallium_pbe_descriptor_size(enum agx_descriptor_generation generation)
{
   return agx_pbe_descriptor_size(generation);
}

static inline struct agx_gallium_storage_descriptor_layout
agx_gallium_storage_descriptor_layout(
   enum agx_descriptor_generation generation)
{
   const struct agx_storage_descriptor_layout base =
      agx_storage_descriptor_layout(generation);

   return (struct agx_gallium_storage_descriptor_layout){
      .texture = base.texture,
      .pbe = base.pbe,
      .stride = base.stride,
   };
}

static inline bool
agx_gallium_native_texture_supported(enum agx_descriptor_generation generation,
                                     bool multisampled,
                                     unsigned buffer_tail_offset)
{
   return agx_native_texture_supported(generation, multisampled,
                                       buffer_tail_offset);
}

static inline bool
agx_gallium_native_texture_sample_count_representable(
   enum agx_descriptor_generation generation, unsigned samples)
{
   return agx_native_texture_sample_count_representable(generation, samples);
}

/* Whether the generation has any recovered PBE descriptor encoding at all.
 *
 * This is a generation predicate, not a per-image one.  AGX3's recovered PBE
 * geometry is a bounded subset -- non-sparse 2D (on Gen9 also arrays, 3D,
 * mips and multisample, and the GPU-tiled compressed single-layer 2D and
 * 2x/4x multisample 2D packets) with no software addressing sideband -- and
 * that subset is enforced per descriptor by agx_gallium_pack_native_pbe(),
 * which fails closed for anything outside it.  Gallium still feeds AGX2
 * sideband state (image atomics, EMRT, spilled render targets) through the
 * same packet, so those images are refused rather than silently encoded with
 * the bits dropped.
 */
static inline bool
agx_gallium_native_pbe_supported(enum agx_descriptor_generation generation)
{
   return agx_native_pbe_hardware_supported(generation);
}

/* Whether Gallium may write any compressed image, shared ones included.
 *
 * Gallium decompresses a shared compressed image in place when a feedback
 * loop samples it (agx_decompress_inplace()). That kernel takes the AGX2
 * 24+24-byte descriptor pair and has no AGX3 build, so only AGX2 qualifies.
 * On AGX3 the compressed PBE itself is encoded (Gen9 GPU-tiled 2D and 2x/4x
 * multisample 2D, see agx_gallium_pack_native_pbe()); G17P compresses its
 * private images within agx_g17p_compression_color_format() and reallocates
 * them uncompressed via agx_decompress() if a framebuffer spills them or a
 * feedback loop/shader image binds them, while imported compressed images
 * stay sample-only (external) there.
 */
static inline bool
agx_gallium_shared_compression_supported(
   enum agx_descriptor_generation generation)
{
   return agx_descriptor_generation_is_agx2(generation);
}

/* Classify the three border colours represented directly by every native AGX
 * sampler generation. Returns false for colours that require shader-backed
 * custom-border emulation. */
bool agx_gallium_classify_standard_border(const uint32_t color[4], bool integer,
                                          enum agx_border_colour *out);

static inline bool
agx_gallium_pack_native_texture(enum agx_descriptor_generation generation,
                                struct agx_gallium_texture_descriptor *out,
                                const struct AGX_TEXTURE *semantic)
{
   return agx_pack_native_texture(generation, out, semantic);
}

/* Pack a native AGX3 per-stage sampler table entry (32-byte element). */
static inline bool
agx_gallium_pack_native_sampler(enum agx_chip chip,
                                struct agx_gallium_sampler_descriptor *out,
                                const struct agx_gallium_native_sampler *state)
{
   if (!out)
      return false;

   if (!agx_chip_native_agx3(chip) ||
       agx_chip_sampler_heap_element_size(chip) != sizeof(*out)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   return agx_pack_native_sampler(chip, out, state);
}

/* Exact sampler zero used by native AGX3 texture.read()/texel-fetch paths. */
static inline bool
agx_gallium_pack_g17p_txf_sampler(enum agx_chip chip,
                                  struct agx_gallium_sampler_descriptor *out)
{
   if (!out)
      return false;

   if (agx_chip_sampler_heap_element_size(chip) != sizeof(*out)) {
      memset(out, 0, sizeof(*out));
      return false;
   }

   return agx_pack_native_txf_sampler(chip, out);
}

/* Convert a Gallium API sampler slot to the index in the per-stage HAL200
 * table. This is deliberately not the legacy global-heap 28+index handle. */
bool agx_gallium_g17p_sampler_handle(enum agx_chip chip, unsigned api_sampler,
                                     uint16_t *native_handle);

bool agx_gallium_g17p_custom_border_handle(enum agx_chip chip,
                                           unsigned api_sampler,
                                           uint16_t *native_handle);

static inline bool
agx_gallium_set_null_native_texture(enum agx_descriptor_generation generation,
                                    struct agx_gallium_texture_descriptor *out)
{
   return agx_set_null_native_texture(generation, out);
}

bool agx_gallium_pack_native_pbe(enum agx_descriptor_generation generation,
                                 struct agx_gallium_pbe_descriptor *out,
                                 const struct AGX_PBE *semantic);

static inline bool
agx_gallium_pack_g18_native_pbe(struct agx_gallium_pbe_descriptor *out,
                                const struct AGX_PBE *semantic)
{
   return agx_pack_g18_native_pbe(out, semantic);
}

static inline bool
agx_gallium_set_null_native_pbe(enum agx_descriptor_generation generation,
                                struct agx_gallium_pbe_descriptor *out)
{
   return agx_set_null_native_pbe(generation, out);
}
