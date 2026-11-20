/* SPDX-License-Identifier: MIT */
#pragma once

#include "agx_chip.h"

/* G17P is the driver's normalized AGX3B/HAL200 profile. The M4 Pro kernel
 * reports raw G16S, which agx_chip_from_params maps to that profile. */
static inline bool
hk_query_wait_bench_profile_supported(
   const struct drm_asahi_params_global *params)
{
   const uint64_t required = DRM_ASAHI_FEATURE_VM_STATUS |
                             DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS |
                             DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   return agx_chip_from_params(params) == AGX_CHIP_G17P &&
          agx_params_driver_supported(params) &&
          agx_params_native_kernel_abi_supported(params) &&
          params->chip_id == 0x6040 && params->num_dies == 1 &&
          params->num_clusters_total == 2 && params->usc_generation == 3 &&
          params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200 &&
          params->command_timestamp_frequency_hz &&
          (params->features & required) == required;
}
