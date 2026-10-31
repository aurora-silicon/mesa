/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "agx_chip.h"
#include "util/macros.h"

static struct drm_asahi_params_global
g15_params(char variant, uint32_t chip_id, uint32_t clusters)
{
   struct drm_asahi_params_global params = {
      .gpu_generation = 15,
      .gpu_variant = variant,
      .chip_id = chip_id,
      .num_dies = 1,
      .num_clusters_total = clusters,
      .num_cores_per_cluster = 10,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
   };

   for (unsigned i = 0; i < clusters; ++i)
      params.core_masks[i] = 0x3ff;
   return params;
}

/* The M5 identities as the kernel reports them: HAL300, USC3 and the
 * sampled core masks. J775D (G17D) is eight clusters of ten slots with
 * eight enabled cores each (0x3db). */
static struct drm_asahi_params_global
m5_params(char variant, uint32_t chip_id, uint32_t dies, uint32_t clusters,
          uint32_t mask)
{
   struct drm_asahi_params_global params = {
      .gpu_generation = 17,
      .gpu_variant = variant,
      .gpu_revision = 0x11,
      .chip_id = chip_id,
      .num_dies = dies,
      .num_clusters_total = clusters,
      .num_cores_per_cluster = 10,
      .usc_generation = 3,
      .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
   };

   for (unsigned i = 0; i < clusters; ++i)
      params.core_masks[i] = mask;
   return params;
}

static void
check_m5(void)
{
   /* The comparison form libagx uses must agree with the HAL300 table. */
   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = (enum agx_chip)c;
      assert(agx_chip_hal300(chip) ==
             (agx_chip_command_stream_generation(chip) ==
              AGX_CS_GENERATION_HAL300));
      if (agx_chip_g17_m5(chip))
         assert(agx_chip_hal300(chip));
   }
   assert(agx_chip_g17_m5(AGX_CHIP_G17G));
   assert(agx_chip_g17_m5(AGX_CHIP_G17S));
   assert(agx_chip_g17_m5(AGX_CHIP_G17D));
   assert(!agx_chip_g17_m5(AGX_CHIP_G17C));
   assert(!agx_chip_g17_m5(AGX_CHIP_G17P));

   /* G17D shares G17S's ISA and command-architecture rows. */
   assert(agx_chip_isa(AGX_CHIP_G17D) == AGX_ISA_AGX3);
   assert(agx_chip_isa_encoding(AGX_CHIP_G17D) == AGX_ISA_ENCODING_AGX3C);
   assert(agx_chip_command_stream_generation(AGX_CHIP_G17D) ==
          agx_chip_command_stream_generation(AGX_CHIP_G17S));
   assert(agx_chip_descriptor_generation(AGX_CHIP_G17D) ==
          AGX_DESCRIPTOR_GENERATION_10);
   assert(agx_chip_sampler_generation(AGX_CHIP_G17D) ==
          AGX_SAMPLER_GENERATION_GEN5);
   assert(agx_chip_sampler_heap_element_size(AGX_CHIP_G17D) == 32);
   assert(agx_chip_codegen_supported(AGX_CHIP_G17D));
   assert(!agx_chip_native_agx3(AGX_CHIP_G17D));
   assert(!agx_chip_driver_supported(AGX_CHIP_G17D));
   assert(!agx_chip_uses_legacy_precompiled_programs(AGX_CHIP_G17D));
   assert(!strcmp(agx_chip_name(AGX_CHIP_G17D), "G17D"));
   assert(!strcmp(agx_chip_exact_product_name(AGX_CHIP_G17D), "M5 Ultra"));
   assert(agx_chip_exact_product_name(AGX_CHIP_G17S) == NULL);

   struct drm_asahi_params_global g17g = m5_params('G', 0x8142, 1, 1, 0x3ff);
   struct drm_asahi_params_global g17s = m5_params('S', 0x6050, 1, 2, 0x3ff);
   struct drm_asahi_params_global g17d = m5_params('D', 0x6050, 2, 8, 0x3db);
   assert(agx_chip_from_params(&g17g) == AGX_CHIP_G17G);
   assert(agx_chip_from_params(&g17s) == AGX_CHIP_G17S);
   assert(agx_chip_from_params(&g17d) == AGX_CHIP_G17D);

   /* M5 opens only through the experimental contract, never by default. */
   assert(!agx_params_m5_bringup_supported(NULL));
   for (unsigned i = 0; i < 3; ++i) {
      const struct drm_asahi_params_global *p =
         i == 0 ? &g17g : i == 1 ? &g17s : &g17d;
      assert(agx_params_m5_bringup_supported(p));
      assert(!agx_params_driver_supported(p));
   }

   /* Revision and binned masks are not part of the contract. */
   struct drm_asahi_params_global bad = g17d;
   bad.gpu_revision = 0x10;
   bad.core_masks[3] = 0x1db;
   assert(agx_params_m5_bringup_supported(&bad));

   /* The Ultra's exact topology: two dies, eight clusters, T6050. */
   const uint32_t clusters[] = {1, 2, 4, 7, 16};
   for (unsigned i = 0; i < ARRAY_SIZE(clusters); ++i) {
      bad = g17d;
      bad.num_clusters_total = clusters[i];
      assert(!agx_params_m5_bringup_supported(&bad));
   }
   const uint32_t dies[] = {0, 1, 4};
   for (unsigned i = 0; i < ARRAY_SIZE(dies); ++i) {
      bad = g17d;
      bad.num_dies = dies[i];
      assert(!agx_params_m5_bringup_supported(&bad));
   }
   bad = g17d;
   bad.chip_id = 0x6051;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17d;
   bad.chip_id = 0x8142;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17d;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17d;
   bad.usc_generation = 2;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17d;
   bad.gpu_generation = 16;
   assert(!agx_params_m5_bringup_supported(&bad));

   /* Neither profile reaches the other's identity: the Pro's contract stays
    * one-die, and an Ultra reported as variant S is refused. */
   bad = g17d;
   bad.gpu_variant = 'S';
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G17S);
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17s;
   bad.num_dies = 2;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17s;
   bad.gpu_variant = 'D';
   assert(!agx_params_m5_bringup_supported(&bad));

   /* G17G keeps its one-die, one-cluster T8142 contract. */
   bad = g17g;
   bad.num_clusters_total = 2;
   assert(!agx_params_m5_bringup_supported(&bad));
   bad = g17g;
   bad.num_dies = 2;
   assert(!agx_params_m5_bringup_supported(&bad));

   /* G17C shares the formats but has no contract. */
   bad = m5_params('C', 0x6051, 1, 4, 0x3ff);
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G17C);
   assert(!agx_params_m5_bringup_supported(&bad));
}

static void
check_native_g15(enum agx_chip chip)
{
   assert(agx_chip_native_agx3(chip));
   assert(agx_chip_native_g15(chip));
   assert(agx_chip_native_imageblock(chip));
   assert(agx_chip_native_indirect_compute(chip));
   assert(agx_chip_driver_supported(chip));
   assert(agx_chip_usc_helper_table_slots(chip, 0) == 10);
}

int
main(void)
{
   check_native_g15(AGX_CHIP_G15S);
   check_native_g15(AGX_CHIP_G15C);
   check_m5();

   /* The other M3 parts stay off the native path. */
   assert(!agx_chip_native_agx3(AGX_CHIP_G15G));
   assert(!agx_chip_native_agx3(AGX_CHIP_G15D));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15G));
   assert(!agx_chip_driver_supported(AGX_CHIP_G15D));

   /* T6030 and T6031 are each admitted with their own identity only. */
   struct drm_asahi_params_global g15s = g15_params('S', 0x6030, 2);
   struct drm_asahi_params_global g15c = g15_params('C', 0x6031, 4);
   assert(agx_chip_from_params(&g15s) == AGX_CHIP_G15S);
   assert(agx_chip_from_params(&g15c) == AGX_CHIP_G15C);
   assert(agx_params_driver_supported(&g15s));
   assert(agx_params_driver_supported(&g15c));

   /* A binned part reports fewer cores in its masks and is still admitted. */
   struct drm_asahi_params_global binned = g15c;
   binned.core_masks[0] = binned.core_masks[3] = 0x1ff;
   assert(agx_params_driver_supported(&binned));

   struct drm_asahi_params_global bad = g15c;
   bad.chip_id = 0x6030;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_clusters_total = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.num_dies = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(!agx_params_driver_supported(&bad));

   bad = g15c;
   bad.usc_generation = 2;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.chip_id = 0x6031;
   assert(!agx_params_driver_supported(&bad));

   bad = g15s;
   bad.num_clusters_total = 4;
   assert(!agx_params_driver_supported(&bad));

   /* The M4 Pro (T6040, G16S, AGX3, HAL200) runs as G17P with its two
    * clusters; other G16S identities stay G16S and unsupported. */
   struct drm_asahi_params_global t6040 = g15_params('S', 0x6040, 2);
   t6040.gpu_generation = 16;
   t6040.gpu_hal_generation = DRM_ASAHI_GPU_HAL_200;
   assert(agx_chip_from_params(&t6040) == AGX_CHIP_G17P);
   assert(agx_params_driver_supported(&t6040));

   bad = t6040;
   bad.num_clusters_total = 3;
   assert(!agx_params_driver_supported(&bad));

   bad = t6040;
   bad.chip_id = 0x6041;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);
   assert(!agx_params_driver_supported(&bad));

   bad = t6040;
   bad.gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY;
   assert(agx_chip_from_params(&bad) == AGX_CHIP_G16S);

   /* G17P helper tables have a slot per core of a cluster, never fewer than
    * the A18 Pro's six: six there, ten on the M4 Pro. */
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 0) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 6) == 6);
   assert(agx_chip_usc_helper_table_slots(AGX_CHIP_G17P, 10) == 10);

   printf("chip admission: ok\n");
   return 0;
}
