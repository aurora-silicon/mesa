/* SPDX-License-Identifier: MIT */
#include "hk-query-wait-bench-profile.h"

#include <stdio.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

int
main(void)
{
   const struct drm_asahi_params_global observed = {
      .features = UINT64_C(0x382001dc06),
      .chip_id = 0x6040,
      .gpu_generation = 16,
      .gpu_variant = 'S',
      .num_dies = 1,
      .num_clusters_total = 2,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
      .command_timestamp_frequency_hz = 1000000000,
   };
   CHECK(agx_chip_from_params(&observed) == AGX_CHIP_G17P);
   CHECK(agx_params_driver_supported(&observed));
   CHECK(hk_query_wait_bench_profile_supported(&observed));
   struct drm_asahi_params_global p = observed;
   p.chip_id = 0x6041;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.chip_id = 0x8140;
   p.gpu_generation = 17;
   p.gpu_variant = 'P';
   p.num_clusters_total = 1;
   CHECK(agx_params_driver_supported(&p));
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.usc_generation = 2;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.num_dies = 2;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.num_clusters_total = 3;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   p = observed;
   p.command_timestamp_frequency_hz = 0;
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   const uint64_t required[] = {
      DRM_ASAHI_FEATURE_VM_STATUS,
      DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS,
      DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS,
   };
   for (unsigned i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
      p = observed;
      p.features &= ~required[i];
      CHECK(!hk_query_wait_bench_profile_supported(&p));
   }
   p = observed;
   p.gpu_variant = 'G';
   CHECK(!hk_query_wait_bench_profile_supported(&p));
   puts(
      "PASS: observed raw G16S T6040 maps to G17P; wrong chip, USC, HAL, topology, clock, feature and architecture profiles rejected; no Vulkan/device calls");
   return 0;
}
