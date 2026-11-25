/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef BRL_GEN1_RT_INDIRECT_CONTRACT_H
#define BRL_GEN1_RT_INDIRECT_CONTRACT_H
#include "aurora/lib/agx_scs_layout.h"
#include <stdint.h>
#include <stdbool.h>

#define BRL_GEN1_AS_TABLE_AS_SLOTS 2048u
#define BRL_GEN1_RQ_PHYSICAL_SLOTS 3072u
#define BRL_GEN1_AS_TABLE_ENTRIES (BRL_GEN1_AS_TABLE_AS_SLOTS + BRL_GEN1_RQ_PHYSICAL_SLOTS)
#define BRL_GEN1_AS_TABLE_ENTRY_BYTES 16u
#define BRL_GEN1_AS_TABLE_BYTES (BRL_GEN1_AS_TABLE_ENTRIES * BRL_GEN1_AS_TABLE_ENTRY_BYTES)
#define BRL_GEN1_RT_CANDIDATE_LAYOUT UINT64_C(0x0130000060010070)

struct brl_gen1_rt_indirect_contract {
   uint32_t physical_slots, cliques;
   uint32_t query_base, query_stride, driver_end, state_bytes;
   uint32_t kernel_scs_pages;
   uint32_t as_table_bytes;
};

/* Same resident-window contract as the direct Gen1 candidate-query path:
 * firmware's Gen1 window word carries 96 cliques and shaders address
 * SR59-resident slots by warp/lane, independent of logical launch dimensions.
 * Indirect RT remains default-off.
 */
static inline bool
brl_gen1_rt_indirect_contract_init(uint64_t packed,
                                struct brl_gen1_rt_indirect_contract *out)
{
   struct agx_scs_layout_gen2 scs;
   if (!out || packed != BRL_GEN1_RT_CANDIDATE_LAYOUT ||
       !agx_scs_layout_gen1_init(packed, &scs))
      return false;
   uint64_t cliques = (packed >> 24) & 255;
   uint64_t slots = cliques * 32;
   uint64_t stride = (packed >> 48) * 32;
   uint64_t end = (uint64_t)scs.end[5] + cliques * stride;
   if (cliques != 96 || slots != BRL_GEN1_RQ_PHYSICAL_SLOTS ||
       scs.end[5] != 0x2f8000 || stride != 304 * 32 ||
       end != scs.end[10] || end != scs.end[12] || end > UINT32_MAX)
      return false;
   /* Candidate records, instance pointers and mask/control fields fit every
    * physical warp's driver window, including the last lane/cursor. */
   if (1920u + 31u * 200u + 15u * 20u + 20u > stride ||
       9152u + 31u * 6u + 6u > stride ||
       8320u + 31u * 8u + 8u > stride)
      return false;
   uint64_t state = (end + 16383) & ~UINT64_C(16383);
   if (state < 0x60000 || state > UINT32_MAX ||
       BRL_GEN1_AS_TABLE_AS_SLOTS * 16u + slots * 16u > BRL_GEN1_AS_TABLE_BYTES)
      return false;
   *out = (struct brl_gen1_rt_indirect_contract){
      .physical_slots = slots, .cliques = cliques,
      .query_base = scs.end[5], .query_stride = stride,
      .driver_end = end, .state_bytes = state,
      /* Exact native scratch-page guard expression, not launch size. */
      .kernel_scs_pages = (end + 2 * 4096 - 1) >> 12,
      .as_table_bytes = BRL_GEN1_AS_TABLE_BYTES,
   };
   return true;
}
#endif
