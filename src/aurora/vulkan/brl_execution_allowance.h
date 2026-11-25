/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef BRL_EXECUTION_ALLOWANCE_H
#define BRL_EXECUTION_ALLOWANCE_H

#include <stdbool.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "aurora/lib/agx_chip.h"

/* Old kernels reject unknown compute flags. Never use scratch/IFT metadata
 * to disguise a build as a ray-query shader: builders do not traverse rays.
 */
static inline uint32_t
brl_as_build_flags(enum agx_chip chip, uint64_t features, bool as_build)
{
   return chip == AGX_CHIP_G17P && as_build && (features & DRM_ASAHI_FEATURE_BOUNDED_AS_BUILD)
             ? DRM_ASAHI_COMPUTE_AS_BUILD : 0;
}
#endif
