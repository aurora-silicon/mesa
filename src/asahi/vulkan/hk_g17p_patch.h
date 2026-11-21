/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_ioto_patch.h"

#define HK_G17P_PATCH_MAX_RANGES  AGX_IOTO_PATCH_MAX_RANGES
#define HK_G17P_PATCH_MAX_RELOCS  AGX_IOTO_PATCH_MAX_RELOCS
#define HK_G17P_PATCH_CLUSTER_GAP AGX_IOTO_PATCH_CLUSTER_GAP

#define hk_g17p_patch_reloc agx_ioto_patch_reloc
#define hk_g17p_patch_plan  agx_ioto_patch_plan

static inline bool
hk_g17p_patch_plan_init(struct hk_g17p_patch_plan *plan,
                        const struct agx3_ioto_uniform_range *ranges,
                        unsigned range_count,
                        const struct agx_ioto_load_relocation *relocs,
                        unsigned reloc_count, size_t length, bool independent)
{
   return agx_ioto_patch_plan_init(plan, ranges, range_count, relocs,
                                   reloc_count, length, independent);
}

static inline bool
hk_g17p_patch_plan_fits(const struct hk_g17p_patch_plan *plan,
                        const struct agx3_ioto_uniform_range *ranges)
{
   return agx_ioto_patch_plan_fits(plan, ranges);
}

static inline void
hk_g17p_patch_plan_apply(const struct hk_g17p_patch_plan *plan,
                         const struct agx3_ioto_uniform_range *ranges,
                         uint8_t *bytes)
{
   agx_ioto_patch_plan_apply(plan, ranges, bytes);
}

/* The LDIMMs plan patches, a low then a high half for each relocation: their
 * offsets in the program it was made from, and the immediates they take for
 * ranges (either may be NULL). Returns their count. */
static inline unsigned
hk_g17p_patch_plan_ldimms(const struct hk_g17p_patch_plan *plan,
                          const struct agx3_ioto_uniform_range *ranges,
                          unsigned *offsets, uint32_t *immediates)
{
   for (unsigned r = 0; r < plan->reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &plan->relocs[r];
      if (offsets) {
         offsets[2 * r] = rel->offset;
         offsets[2 * r + 1] = rel->offset + ((rel->short_mask & 1) ? 2 : 8);
      }
      if (immediates) {
         const uint64_t base = ranges[rel->range].address + rel->delta;
         immediates[2 * r] = (uint32_t)base;
         immediates[2 * r + 1] = (uint32_t)(base >> 32);
      }
   }
   return 2 * plan->reloc_count;
}
