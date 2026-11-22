/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_native_descriptor.h"

#define HK_NATIVE_SAMPLER_MAX_ELEMENT_SIZE    AGX_NATIVE_SAMPLER_MAX_ELEMENT_SIZE
#define hk_sampler_reduction_mode             agx_native_sampler_reduction_mode
#define HK_SAMPLER_REDUCTION_WEIGHTED_AVERAGE AGX_NATIVE_SAMPLER_REDUCTION_WEIGHTED_AVERAGE
#define HK_SAMPLER_REDUCTION_MINIMUM          AGX_NATIVE_SAMPLER_REDUCTION_MINIMUM
#define HK_SAMPLER_REDUCTION_MAXIMUM          AGX_NATIVE_SAMPLER_REDUCTION_MAXIMUM
#define hk_native_sampler_state               agx_native_sampler_state
#define hk_native_sampler_descriptor          agx_native_sampler_descriptor

static inline bool
hk_sampler_reduction_supported(enum agx_sampler_generation generation,
                               enum hk_sampler_reduction_mode reduction_mode)
{
   return agx_native_sampler_reduction_supported(generation, reduction_mode);
}

static inline bool
hk_pack_native_sampler(enum agx_chip chip,
                       struct hk_native_sampler_descriptor *out,
                       const struct hk_native_sampler_state *state)
{
   return agx_pack_native_sampler(chip, out, state);
}

static inline bool
hk_pack_native_txf_sampler(enum agx_chip chip,
                           struct hk_native_sampler_descriptor *out)
{
   return agx_pack_native_txf_sampler(chip, out);
}
