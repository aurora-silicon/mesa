/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_chip.h"
#include "util/macros.h"
#include "util/u_atomic.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* AGX2 (Gen7/8) texture and PBE descriptors are 24 bytes; AGX3 (Gen9/10)
 * grows both to 32 bytes. Keep the native allocations opaque so a legacy C
 * struct can never silently select the AGX2 stride on AGX3.
 */
#define AGX_NATIVE_DESCRIPTOR_AGX2_SIZE     24u
#define AGX_NATIVE_DESCRIPTOR_AGX3_SIZE     32u
#define AGX_NATIVE_DESCRIPTOR_MAX_SIZE      AGX_NATIVE_DESCRIPTOR_AGX3_SIZE
#define AGX_NATIVE_SAMPLER_AGX2_SIZE        8u
#define AGX_NATIVE_SAMPLER_AGX3_SIZE        32u
#define AGX_NATIVE_SAMPLER_MAX_ELEMENT_SIZE AGX_NATIVE_SAMPLER_AGX3_SIZE

#define AGX_NULL_NATIVE_TEXTURE_GENERATIONS 16u

struct agx_native_texture_descriptor {
   uint32_t opaque[AGX_NATIVE_DESCRIPTOR_MAX_SIZE / sizeof(uint32_t)];
};

struct agx_native_pbe_descriptor {
   uint32_t opaque[AGX_NATIVE_DESCRIPTOR_MAX_SIZE / sizeof(uint32_t)];
};

/* An AGX3 sampler heap element is sparse: its eight-byte packed state word
 * sits at offset zero and the remaining 24 bytes stay zero. Keeping the full
 * element here prevents confusing upload size with word size.
 */
struct agx_native_sampler_descriptor {
   uint32_t opaque[AGX_NATIVE_SAMPLER_MAX_ELEMENT_SIZE / sizeof(uint32_t)];
};

struct agx_storage_descriptor_layout {
   uint8_t texture;
   uint8_t pbe;
   uint8_t stride;
};

static inline bool
agx_descriptor_generation_is_agx2(enum agx_descriptor_generation generation)
{
   return generation == AGX_DESCRIPTOR_GENERATION_7 ||
          generation == AGX_DESCRIPTOR_GENERATION_8;
}

static inline bool
agx_descriptor_generation_is_agx3(enum agx_descriptor_generation generation)
{
   return generation == AGX_DESCRIPTOR_GENERATION_9 ||
          generation == AGX_DESCRIPTOR_GENERATION_10;
}

static inline unsigned
agx_texture_descriptor_size(enum agx_descriptor_generation generation)
{
   return agx_descriptor_generation_is_agx2(generation)
             ? AGX_NATIVE_DESCRIPTOR_AGX2_SIZE
          : agx_descriptor_generation_is_agx3(generation)
             ? AGX_NATIVE_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline unsigned
agx_pbe_descriptor_size(enum agx_descriptor_generation generation)
{
   return agx_texture_descriptor_size(generation);
}

static inline struct agx_storage_descriptor_layout
agx_storage_descriptor_layout(enum agx_descriptor_generation generation)
{
   const unsigned texture = agx_texture_descriptor_size(generation);
   const unsigned pbe = agx_pbe_descriptor_size(generation);

   if (!texture || !pbe)
      return (struct agx_storage_descriptor_layout){0};

   return (struct agx_storage_descriptor_layout){
      .texture = 0,
      .pbe = texture,
      .stride = texture + pbe,
   };
}

/* AGX3 buffer descriptors have no recovered equivalent for AGX2's software
 * tail offset. Keep those unavailable rather than aliasing unknown bits in a
 * native descriptor. Multisample descriptor representation is recovered on
 * both AGX3 generations; this is not by itself an execution/capability gate.
 */
static inline bool
agx_native_texture_supported(enum agx_descriptor_generation generation,
                             bool multisampled, unsigned buffer_tail_offset)
{
   if (agx_descriptor_generation_is_agx2(generation))
      return true;

   if (!agx_descriptor_generation_is_agx3(generation))
      return false;

   return buffer_tail_offset == 0;
}

/* Return whether the native sampled-texture descriptor has a recovered
 * representation for a numeric sample count. This is not an API capability
 * predicate: the live G17S corpus proves Gen10 descriptor bits but contains
 * no executed 8x sampling command, so public 8x exposure remains gated.
 */
static inline bool
agx_native_texture_sample_count_representable(
   enum agx_descriptor_generation generation, unsigned samples)
{
   if (samples == 1)
      return agx_texture_descriptor_size(generation) != 0;

   if (agx_descriptor_generation_is_agx2(generation))
      return samples == 2 || samples == 4;

   if (agx_descriptor_generation_is_agx3(generation))
      return samples == 2 || samples == 4 || samples == 8;

   return false;
}

/* Whether the generation has any recovered hardware PBE descriptor encoding.
 * Per-image admission is enforced by agx_pack_native_pbe().
 */
static inline bool
agx_native_pbe_hardware_supported(enum agx_descriptor_generation generation)
{
   return agx_pbe_descriptor_size(generation) != 0;
}

/* Translate a multisample count only when the native sampled-texture format
 * is recovered for the descriptor generation. There is no one-sample field in
 * the descriptor, so samples == 1 deliberately returns false.
 */
bool agx_translate_native_texture_sample_count(
   enum agx_descriptor_generation generation, unsigned samples,
   enum agx_sample_count *out);

/* Pack a native texture descriptor for `generation`. Writes exactly
 * agx_texture_descriptor_size(generation) bytes (24 for AGX2, 32 for AGX3),
 * leaving any trailing bytes in `out` untouched on AGX2 so callers packing
 * directly into 24-byte-strided arrays cannot overrun adjacent slots.
 */
bool agx_pack_native_texture(enum agx_descriptor_generation generation,
                             struct agx_native_texture_descriptor *out,
                             const struct AGX_TEXTURE *semantic);

/* Keep a 2D view's level-zero extent in the Gen10 descriptor's software word
 * (AGX_GEN10_IMAGE_EXTENT_OFFSET), for AGX_NIR_TEXTURE_LEVEL_ZERO_EXTENT.
 * Only for chips where agx_chip_texture_descriptor_spare_word() holds.
 */
static inline void
agx_native_texture_set_level_zero_extent(
   struct agx_native_texture_descriptor *out, uint32_t width, uint32_t height)
{
   /* Width and height are each at most 32768 (agx_pack_native_texture). */
   assert(width <= 0xffff && height <= 0xffff);
   out->opaque[AGX_GEN10_IMAGE_EXTENT_OFFSET / 4] = width | (height << 16);
}

/* The generation's robust null texture descriptor. It is a pure function of
 * the generation, so it is packed once per generation via an atomic-once
 * table and copied inline afterwards. Writes exactly
 * agx_texture_descriptor_size(generation) bytes.
 */
extern struct agx_native_texture_descriptor
   agx_null_native_texture_packed[AGX_NULL_NATIVE_TEXTURE_GENERATIONS];
/* 0: not packed yet, 1: being published, 2: packed valid. */
extern uint32_t
   agx_null_native_texture_state[AGX_NULL_NATIVE_TEXTURE_GENERATIONS];

bool agx_set_null_native_texture_slow(
   enum agx_descriptor_generation generation,
   struct agx_native_texture_descriptor *out);

static inline bool
agx_set_null_native_texture(enum agx_descriptor_generation generation,
                            struct agx_native_texture_descriptor *out)
{
   const unsigned size = agx_texture_descriptor_size(generation);
   if (likely(out && size &&
              (unsigned)generation < AGX_NULL_NATIVE_TEXTURE_GENERATIONS &&
              p_atomic_read(&agx_null_native_texture_state[generation]) == 2)) {
      memcpy(out, &agx_null_native_texture_packed[generation], size);
      return true;
   }
   return agx_set_null_native_texture_slow(generation, out);
}

/* Pack a native hardware PBE descriptor from a semantic struct AGX_PBE.
 * Writes exactly agx_pbe_descriptor_size(generation) bytes (24 for AGX2,
 * 32 for AGX3).
 */
bool agx_pack_native_pbe(enum agx_descriptor_generation generation,
                         struct agx_native_pbe_descriptor *out,
                         const struct AGX_PBE *semantic);

/* Pack a Gen10 PBE descriptor using the G18G-specific mip/layered/multisample
 * PBE encoders (preserving exact existing G18G Gallium behaviour).
 */
bool agx_pack_g18_native_pbe(struct agx_native_pbe_descriptor *out,
                             const struct AGX_PBE *semantic);

/* Construct the generation-native robust null PBE descriptor. Writes exactly
 * agx_pbe_descriptor_size(generation) bytes.
 */
bool agx_set_null_native_pbe(enum agx_descriptor_generation generation,
                             struct agx_native_pbe_descriptor *out);

enum agx_native_sampler_reduction_mode {
   AGX_NATIVE_SAMPLER_REDUCTION_WEIGHTED_AVERAGE = 0,
   AGX_NATIVE_SAMPLER_REDUCTION_MINIMUM = 1,
   AGX_NATIVE_SAMPLER_REDUCTION_MAXIMUM = 2,
};

/* Generation-independent sampler semantics. agx_pack_native_sampler() is the
 * single place that selects the Gen4, Gen4 HAL200, or Gen5 bit layout.
 */
struct agx_native_sampler_state {
   float minimum_lod;
   float maximum_lod;
   /* Applied only where the sampler word has a bias field (HAL200, Gen5);
    * on Gen4 the shader applies the bias. */
   float lod_bias;
   unsigned maximum_anisotropy;
   enum agx_filter magnify;
   enum agx_filter minify;
   enum agx_mip_filter mip_filter;
   enum agx_wrap wrap_s;
   enum agx_wrap wrap_t;
   enum agx_wrap wrap_r;
   enum agx_compare_func compare_func;
   enum agx_border_colour border_colour;
   enum agx_native_sampler_reduction_mode reduction_mode;
   bool pixel_coordinates;
   bool compare_enable;
   bool seamful_cube_maps;
};

bool agx_native_sampler_reduction_supported(
   enum agx_sampler_generation generation,
   enum agx_native_sampler_reduction_mode reduction_mode);

bool agx_pack_native_sampler(enum agx_chip chip,
                             struct agx_native_sampler_descriptor *out,
                             const struct agx_native_sampler_state *state);

bool agx_pack_native_txf_sampler(enum agx_chip chip,
                                 struct agx_native_sampler_descriptor *out);

#ifdef __cplusplus
}
#endif
