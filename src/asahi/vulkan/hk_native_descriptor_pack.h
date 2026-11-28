/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_native_descriptor.h"
#include "hk_native_descriptor.h"

#define hk_native_texture_descriptor agx_native_texture_descriptor
#define hk_native_pbe_descriptor     agx_native_pbe_descriptor

static inline bool
hk_pack_native_texture(unsigned generation,
                       struct hk_native_texture_descriptor *out,
                       const struct AGX_TEXTURE *legacy)
{
   const enum agx_descriptor_generation gen =
      (enum agx_descriptor_generation)generation;

   if (out && !agx_descriptor_generation_is_agx3(gen))
      memset(out, 0, sizeof(*out));

   return agx_pack_native_texture(gen, out, legacy);
}

static inline void
hk_native_texture_set_level_zero_extent(
   struct hk_native_texture_descriptor *out, uint32_t width, uint32_t height)
{
   agx_native_texture_set_level_zero_extent(out, width, height);
}

static inline bool
hk_translate_native_texture_sample_count(unsigned generation, unsigned samples,
                                         enum agx_sample_count *out)
{
   return agx_translate_native_texture_sample_count(
      (enum agx_descriptor_generation)generation, samples, out);
}

static inline bool
hk_pack_native_pbe(unsigned generation, struct hk_native_pbe_descriptor *out,
                   const struct AGX_PBE *legacy)
{
   const enum agx_descriptor_generation gen =
      (enum agx_descriptor_generation)generation;

   if (out && !agx_descriptor_generation_is_agx3(gen))
      memset(out, 0, sizeof(*out));

   return agx_pack_native_pbe(gen, out, legacy);
}

static inline bool
hk_set_null_native_texture(unsigned generation,
                           struct hk_native_texture_descriptor *out)
{
   const enum agx_descriptor_generation gen =
      (enum agx_descriptor_generation)generation;

   if (unlikely(out && !agx_descriptor_generation_is_agx3(gen)))
      memset(out, 0, sizeof(*out));

   return agx_set_null_native_texture(gen, out);
}

static inline bool
hk_set_null_native_pbe(unsigned generation,
                       struct hk_native_pbe_descriptor *out)
{
   const enum agx_descriptor_generation gen =
      (enum agx_descriptor_generation)generation;

   if (out && !agx_descriptor_generation_is_agx3(gen))
      memset(out, 0, sizeof(*out));

   return agx_set_null_native_pbe(gen, out);
}
