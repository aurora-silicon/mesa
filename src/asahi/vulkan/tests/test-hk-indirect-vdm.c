/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_ioto_load_core.h"
#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_hal200_direct_draw_core.h"
#include "asahi/lib/agx_vdm_index_list_core.h"
#include "asahi/lib/agx_vdm_state_core.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/libagx/libagx_dgc.h"
#include "asahi/vulkan/hk_g17p_patch.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "line %u: %s\n", __LINE__, #cond);                    \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static void
test_ldimm_encoding(void)
{
   static const uint32_t samples[] = {
      0x00000000u, 0x00000001u, 0x0000007fu, 0x00000080u, 0x000007ffu,
      0x00001fffu, 0x0001ffffu, 0x01ffffffu, 0x12345678u, 0xdeadbeefu,
      0xfedcba98u, 0xffffffffu,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(samples); ++i) {
      CHECK(libagx_ioto_ldimm_long_immediate(samples[i]) ==
            agx_ioto_load_ldimm_long_immediate(samples[i]));
   }
   for (unsigned bit = 0; bit < 32; ++bit) {
      const uint32_t v = UINT32_C(1) << bit;
      CHECK(libagx_ioto_ldimm_long_immediate(v) ==
            agx_ioto_load_ldimm_long_immediate(v));
      CHECK(libagx_ioto_ldimm_long_immediate(~v) ==
            agx_ioto_load_ldimm_long_immediate(~v));
   }
}

static void
test_gpu_vdm_builder(enum agx_chip chip, bool indexed, bool republishes,
                     bool uses_draw_id, bool with_count, uint32_t max_draws,
                     uint32_t runtime_count)
{
   const uint32_t in_stride = indexed ? 24u : 20u;
   const uint32_t param_off = indexed ? 12u : 8u;
   const uint64_t in_va = UINT64_C(0x1400001000);
   const uint64_t index_va = UINT64_C(0x1500002000);
   const uint64_t return_compact = UINT64_C(0x0012345678);

   /* Build a representative VDM record template matching draw 0. */
   uint8_t vdm_tmpl[160] __attribute__((aligned(16)));
   memset(vdm_tmpl, 0, sizeof(vdm_tmpl));
   uint32_t pos = 0;
   uint32_t pipeline_word_off = UINT32_MAX;
   const uint32_t raw_template = 3u << 3;

   if (republishes) {
      /* 28-byte VDM state update prefix with pipeline pair at bytes 8..15. */
      struct agx_vdm_state_vertex_pipeline_pair pair;
      CHECK(agx_vdm_state_vertex_pipeline_pair_pack(
         chip, UINT64_C(0x1000040000), raw_template, &pair));
      for (unsigned i = 0; i < AGX_VDM_STATE_UPDATE_LENGTH; ++i)
         vdm_tmpl[i] = (uint8_t)(0xa0u + i);
      memcpy(vdm_tmpl + 8, pair.bytes, sizeof(pair.bytes));
      pipeline_word_off = 2u;
      pos += AGX_VDM_STATE_UPDATE_LENGTH;
   }

   const uint32_t body_off = pos;
   if (indexed) {
      struct agx_vdm_index_list_indirect_indexed pkt;
      const struct agx_vdm_index_list_indirect_indexed_info info = {
         .primitive = AGX_PRIMITIVE_TRIANGLES,
         .restart_enable = true,
         .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U32,
         .index_buffer = index_va,
         .indirect_buffer = in_va,
         .restart_index = UINT32_MAX,
         .index_buffer_size_word = 0x3ffu,
      };
      CHECK(agx_vdm_index_list_indirect_indexed_pack(chip, &info, &pkt));
      memcpy(vdm_tmpl + pos, pkt.bytes, pkt.length);
      pos += pkt.length;
   } else {
      struct agx_vdm_index_list_indirect_nonindexed pkt;
      const struct agx_vdm_index_list_indirect_nonindexed_info info = {
         .primitive = AGX_PRIMITIVE_TRIANGLES,
         .indirect_buffer = in_va,
      };
      CHECK(agx_vdm_index_list_indirect_nonindexed_pack(chip, &info, &pkt));
      memcpy(vdm_tmpl + pos, pkt.bytes, pkt.length);
      pos += pkt.length;
   }

   CHECK((pos % 4u) == 0);
   const uint32_t vdm_record_dwords = pos / 4u;
   const uint32_t vdm_indirect_word_off = body_off / 4u + (indexed ? 4u : 1u);

   /* Build a synthetic Ioto template with both long and short high LDIMMs. */
   uint8_t ioto_tmpl[96] __attribute__((aligned(16)));
   memset(ioto_tmpl, 0x55, sizeof(ioto_tmpl));
   const uint32_t ioto_len = 82u; /* deliberately non-dword-multiple */
   const uint32_t ioto_copy_dwords = ALIGN_POT(ioto_len, 16u) / 4u;
   const uint32_t ioto_stride_B = 128u;

   struct hk_g17p_patch_plan plan = {
      .range_count = 3,
      .reloc_count = 3,
      .relocs = {
         /* reloc 0: DRAW_PARAMS, 2-byte-aligned 8-byte low + 2-byte short high (< 128) */
         {
            .dest = {agx_ioto_load_ldimm_long_dest(4),
                     agx_ioto_load_ldimm_long_dest(6)},
            .delta = 4u,
            .offset = 10u,
            .range = 1,
            .short_mask = 2u,
         },
         /* reloc 1: DRAW_PARAMS, 4-byte-aligned 8-byte low + 8-byte high */
         {
            .dest = {agx_ioto_load_ldimm_long_dest(12),
                     agx_ioto_load_ldimm_long_dest(14)},
            .delta = 0u,
            .offset = 52u,
            .range = 1,
            .short_mask = 0u,
         },
         /* reloc 2: DRAW_ID, 2-byte-aligned 8-byte low + 8-byte high */
         {
            .dest = {agx_ioto_load_ldimm_long_dest(8),
                     agx_ioto_load_ldimm_long_dest(10)},
            .delta = 0u,
            .offset = 30u,
            .range = 2,
            .short_mask = 0u,
         },
      },
   };

   uint32_t draw_ids[16];
   memset(draw_ids, 0xee, sizeof(draw_ids));
   uint8_t ioto_out[16 * 128] __attribute__((aligned(64)));
   memset(ioto_out, 0xcc, sizeof(ioto_out));
   uint32_t sub_vdm[16 * 40 + 16] __attribute__((aligned(16)));
   memset(sub_vdm, 0xaa, sizeof(sub_vdm));

   struct libagx_indirect_vdm_desc desc = {
      .sub_vdm = (uintptr_t)sub_vdm,
      .vdm_template = (uintptr_t)vdm_tmpl,
      .in_buffer = in_va,
      .count_buffer = with_count ? (uintptr_t)&runtime_count : 0,
      .return_link_compact = return_compact,
      .ioto_out = republishes ? (uintptr_t)ioto_out : 0,
      .ioto_template = republishes ? (uintptr_t)ioto_tmpl : 0,
      .draw_ids_out = (republishes && uses_draw_id) ? (uintptr_t)draw_ids : 0,
      .max_draw_count = max_draws,
      .in_stride_B = in_stride,
      .vdm_record_dwords = vdm_record_dwords,
      .vdm_indirect_word_off = vdm_indirect_word_off,
      .vdm_pipeline_word_off = pipeline_word_off,
      .vdm_pipeline_raw_template = raw_template,
      .param_offset_B = param_off,
      .ioto_copy_dwords = ioto_copy_dwords,
      .ioto_stride_B = ioto_stride_B,
      .reloc_count = republishes ? (uses_draw_id ? 3u : 2u) : 0u,
   };
   for (unsigned r = 0; r < desc.reloc_count; ++r) {
      desc.relocs[r] = (struct libagx_indirect_ioto_reloc){
         .dest_lo = plan.relocs[r].dest[0],
         .dest_hi = plan.relocs[r].dest[1],
         .delta = plan.relocs[r].delta,
         .offset = plan.relocs[r].offset,
         .kind = plan.relocs[r].range == 1 ? 0u : 1u,
         .short_mask = plan.relocs[r].short_mask,
      };
   }

   for (uint32_t tid = 0;
        tid < (max_draws + 1u) * LIBAGX_INDIRECT_VDM_LANES_PER_DRAW; ++tid)
      libagx_build_indirect_vdm_sublane(&desc, tid);

   const uint32_t expected_count =
      with_count ? MIN2(runtime_count, max_draws) : max_draws;

   for (uint32_t i = 0; i < expected_count; ++i) {
      const uint8_t *rec =
         (const uint8_t *)&sub_vdm[i * vdm_record_dwords];
      const uint64_t expected_draw_va = in_va + (uint64_t)i * in_stride;

      if (republishes) {
         const uint64_t expected_prog_va =
            (uintptr_t)ioto_out + (uint64_t)i * ioto_stride_B;
         uint64_t decoded_prog = 0;
         uint32_t decoded_tmpl = 0;
         CHECK(agx_vdm_state_vertex_pipeline_pair_unpack(
            chip, rec + 8, 8, &decoded_prog, &decoded_tmpl));
         CHECK(decoded_prog == expected_prog_va);
         CHECK(decoded_tmpl == raw_template);

         if (uses_draw_id)
            CHECK(draw_ids[i] == i);

         /* Verify GPU-patched Ioto program matches CPU hk_g17p_patch_plan_apply(). */
         uint8_t expected_ioto[96];
         memcpy(expected_ioto, ioto_tmpl, ioto_len);
         struct hk_g17p_patch_plan cpu_plan = plan;
         cpu_plan.reloc_count = desc.reloc_count;
         struct agx3_ioto_uniform_range ranges[3] = {
            {.address = 0},
            {.address = expected_draw_va + param_off},
            {.address = (uintptr_t)draw_ids + (uint64_t)i * 4u},
         };
         hk_g17p_patch_plan_apply(&cpu_plan, ranges, expected_ioto);
         CHECK(!memcmp(ioto_out + i * ioto_stride_B, expected_ioto, ioto_len));
      }

      const uint8_t *body = rec + body_off;
      const size_t body_len = pos - body_off;
      if (indexed) {
         struct agx_vdm_index_list_indirect_indexed_info unpacked;
         CHECK(agx_vdm_index_list_indirect_indexed_unpack(
            chip, body, body_len, &unpacked));
         CHECK(unpacked.indirect_buffer == expected_draw_va);
         CHECK(unpacked.index_buffer == index_va);
      } else {
         struct agx_vdm_index_list_indirect_nonindexed_info unpacked;
         CHECK(agx_vdm_index_list_indirect_nonindexed_unpack(
            chip, body, body_len, &unpacked));
         CHECK(unpacked.indirect_buffer == expected_draw_va);
      }
   }

   /* Verify the terminating VDM Stream Link at slot expected_count. */
   const uint8_t *term =
      (const uint8_t *)&sub_vdm[expected_count * vdm_record_dwords];
   uint64_t decoded_target = 0;
   CHECK(agx_vdm_stream_link_core_unpack(chip, term, 8, &decoded_target));
   CHECK(decoded_target == return_compact);

   /* Culled slots after the stream link must remain untouched. */
   if (expected_count + 1u < max_draws) {
      CHECK(sub_vdm[(expected_count + 1u) * vdm_record_dwords] == 0xaaaaaaaau);
   }
}

static void
test_chip_gating(void)
{
   /* The builder admits every chip with the returning VDM Stream Link record,
    * whatever its command-stream generation (G15, HAL200, HAL300); AGX2 has
    * only its own non-returning link. */
   CHECK(agx_vdm_stream_link_core_supported(AGX_CHIP_G15G));
   CHECK(agx_vdm_stream_link_core_supported(AGX_CHIP_G15S));
   CHECK(agx_vdm_stream_link_core_supported(AGX_CHIP_G16G));
   CHECK(agx_vdm_stream_link_core_supported(AGX_CHIP_G17P));
   CHECK(agx_vdm_stream_link_core_supported(AGX_CHIP_G17S));
   CHECK(!agx_vdm_stream_link_core_supported(AGX_CHIP_G13G));
   CHECK(!agx_vdm_stream_link_core_supported(AGX_CHIP_G14X));
   CHECK(agx_chip_command_stream_generation(AGX_CHIP_G17P) ==
         AGX_CS_GENERATION_HAL200);
   CHECK(agx_chip_command_stream_generation(AGX_CHIP_G16G) ==
         AGX_CS_GENERATION_HAL200);
   CHECK(agx_chip_command_stream_generation(AGX_CHIP_G17S) ==
         AGX_CS_GENERATION_HAL300);
   CHECK(agx_chip_command_stream_generation(AGX_CHIP_G15G) ==
         AGX_CS_GENERATION_G15);
   CHECK(agx_chip_command_stream_generation(AGX_CHIP_G15S) ==
         AGX_CS_GENERATION_G15);

   /* Republishing draws write EXEC Ioto programs and require USC cache
    * invalidation on the render-start VDM barrier. */
   CHECK(agx_vdm_barrier_core_word(AGX_CHIP_G17P, false) &
         AGX_VDM_BARRIER_CORE_USC_INVAL);
   CHECK(agx_vdm_barrier_core_word(AGX_CHIP_G17S, false) &
         AGX_VDM_BARRIER_CORE_USC_INVAL);
   CHECK(!(agx_vdm_barrier_core_word(AGX_CHIP_G16G, false) &
           AGX_VDM_BARRIER_CORE_USC_INVAL));
   CHECK(agx_vdm_barrier_core_word(AGX_CHIP_G16G, true) &
         AGX_VDM_BARRIER_CORE_USC_INVAL);

   /* GPU indirect VDM slabs on G17P stay on their primary VA when both ends
    * (last dword at base + size - 4, not unaligned base + size - 1) fit in
    * the compact payload window. */
   const uint64_t slab_base = UINT64_C(0x4000000000);
   const uint64_t slab_size = UINT64_C(4) * 1024u * 1024u;
   uint64_t start_compact = 0, end_compact = 0;
   CHECK(agx_g17p_vdm_payload_compact(slab_base, &start_compact));
   CHECK(agx_g17p_vdm_payload_compact(slab_base + slab_size - 4u,
                                      &end_compact));
   CHECK(!agx_g17p_vdm_payload_compact(slab_base + slab_size - 1u,
                                       &end_compact));
   CHECK(start_compact < (UINT64_C(1) << 40));
   CHECK(end_compact < (UINT64_C(1) << 40));
}

static uint32_t
reference_coalesce(const struct libagx_indirect_cmd_u32 *in, uint32_t count,
                   struct libagx_indirect_cmd_u32 *out)
{
   uint32_t n = 0u;
   for (uint32_t i = 0; i < count; ++i) {
      if (!libagx_indirect_cmd_nonempty(in[i]))
         continue;
      if (n > 0u && libagx_indirect_cmd_can_merge(out[n - 1u], in[i]))
         out[n - 1u].instance_count += in[i].instance_count;
      else
         out[n++] = in[i];
   }
   return n;
}

static void
run_coalesce_case(enum agx_chip chip, bool indexed, bool republishes,
                  bool with_count, uint32_t max_draws, uint32_t runtime_count,
                  uint32_t in_stride, const struct libagx_indirect_cmd_u32 *cmds)
{
   const uint32_t cmd_size = indexed ? 20u : 16u;
   const uint32_t param_off = indexed ? 12u : 8u;
   const uint64_t index_va = UINT64_C(0x1500002000);
   const uint64_t return_compact = UINT64_C(0x0012345678);

   uint8_t *in_buf = calloc(MAX2(max_draws, 1u), in_stride);
   uint8_t *out_cmds = malloc(MAX2(max_draws, 1u) * cmd_size);
   CHECK(in_buf && out_cmds);
   memset(out_cmds, 0xdd, MAX2(max_draws, 1u) * cmd_size);

   for (uint32_t i = 0; i < max_draws; ++i) {
      uint32_t *w = (uint32_t *)(in_buf + (size_t)i * in_stride);
      w[0] = cmds[i].count;
      w[1] = cmds[i].instance_count;
      w[2] = cmds[i].first;
      if (indexed) {
         w[3] = cmds[i].bias;
         w[4] = cmds[i].first_instance;
      } else {
         w[3] = cmds[i].first_instance;
      }
   }

   uint8_t vdm_tmpl[160] __attribute__((aligned(16)));
   memset(vdm_tmpl, 0, sizeof(vdm_tmpl));
   uint32_t pos = 0;
   uint32_t pipeline_word_off = UINT32_MAX;
   const uint32_t raw_template = 3u << 3;

   if (republishes) {
      struct agx_vdm_state_vertex_pipeline_pair pair;
      CHECK(agx_vdm_state_vertex_pipeline_pair_pack(
         chip, UINT64_C(0x1000040000), raw_template, &pair));
      for (unsigned i = 0; i < AGX_VDM_STATE_UPDATE_LENGTH; ++i)
         vdm_tmpl[i] = (uint8_t)(0xa0u + i);
      memcpy(vdm_tmpl + 8, pair.bytes, sizeof(pair.bytes));
      pipeline_word_off = 2u;
      pos += AGX_VDM_STATE_UPDATE_LENGTH;
   }

   const uint32_t body_off = pos;
   const uint64_t dummy_in_va = UINT64_C(0x1400001000);
   if (indexed) {
      struct agx_vdm_index_list_indirect_indexed pkt;
      const struct agx_vdm_index_list_indirect_indexed_info info = {
         .primitive = AGX_PRIMITIVE_TRIANGLES,
         .restart_enable = false,
         .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U32,
         .index_buffer = index_va,
         .indirect_buffer = dummy_in_va,
         .restart_index = 0,
         .index_buffer_size_word = 0x3ffu,
      };
      CHECK(agx_vdm_index_list_indirect_indexed_pack(chip, &info, &pkt));
      memcpy(vdm_tmpl + pos, pkt.bytes, pkt.length);
      pos += pkt.length;
   } else {
      struct agx_vdm_index_list_indirect_nonindexed pkt;
      const struct agx_vdm_index_list_indirect_nonindexed_info info = {
         .primitive = AGX_PRIMITIVE_TRIANGLES,
         .indirect_buffer = dummy_in_va,
      };
      CHECK(agx_vdm_index_list_indirect_nonindexed_pack(chip, &info, &pkt));
      memcpy(vdm_tmpl + pos, pkt.bytes, pkt.length);
      pos += pkt.length;
   }

   CHECK((pos % 4u) == 0);
   const uint32_t vdm_record_dwords = pos / 4u;
   const uint32_t vdm_indirect_word_off = body_off / 4u + (indexed ? 4u : 1u);

   uint8_t ioto_tmpl[96] __attribute__((aligned(16)));
   memset(ioto_tmpl, 0x55, sizeof(ioto_tmpl));
   const uint32_t ioto_len = 82u;
   const uint32_t ioto_copy_dwords = ALIGN_POT(ioto_len, 16u) / 4u;
   const uint32_t ioto_stride_B = 128u;
   struct hk_g17p_patch_plan plan = {
      .range_count = 2,
      .reloc_count = 1,
      .relocs = {
         {
            .dest = {agx_ioto_load_ldimm_long_dest(12),
                     agx_ioto_load_ldimm_long_dest(14)},
            .delta = 0u,
            .offset = 32u,
            .range = 1,
            .short_mask = 0u,
         },
      },
   };

   const uint32_t num_wgs =
      DIV_ROUND_UP(MAX2(max_draws, 1u), LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE);
   const uint32_t chunk_draws =
      MIN2(MAX2(max_draws, 1u), LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE);
   const uint32_t chunk_vdm_stride_B =
      (uint32_t)ALIGN_POT((size_t)chunk_draws * pos +
                             AGX_VDM_STREAM_LINK_CORE_LENGTH,
                          16u);
   const size_t total_sub_vdm_B = (size_t)num_wgs * chunk_vdm_stride_B;
   const uint64_t sub_vdm_compact_base = UINT64_C(0x0040000000);

   uint8_t *ioto_out = aligned_alloc(64, MAX2(max_draws, 1u) * ioto_stride_B);
   uint8_t *sub_vdm = aligned_alloc(16, total_sub_vdm_B);
   CHECK(ioto_out && sub_vdm);
   memset(ioto_out, 0xcc, MAX2(max_draws, 1u) * ioto_stride_B);
   memset(sub_vdm, 0xaa, total_sub_vdm_B);

   struct libagx_indirect_vdm_coalesce_desc desc = {
      .base = {
         .sub_vdm = (uintptr_t)sub_vdm,
         .vdm_template = (uintptr_t)vdm_tmpl,
         .in_buffer = (uintptr_t)in_buf,
         .count_buffer = with_count ? (uintptr_t)&runtime_count : 0,
         .return_link_compact = return_compact,
         .ioto_out = republishes ? (uintptr_t)ioto_out : 0,
         .ioto_template = republishes ? (uintptr_t)ioto_tmpl : 0,
         .draw_ids_out = 0,
         .max_draw_count = max_draws,
         .in_stride_B = in_stride,
         .vdm_record_dwords = vdm_record_dwords,
         .vdm_indirect_word_off = vdm_indirect_word_off,
         .vdm_pipeline_word_off = pipeline_word_off,
         .vdm_pipeline_raw_template = raw_template,
         .param_offset_B = param_off,
         .ioto_copy_dwords = ioto_copy_dwords,
         .ioto_stride_B = ioto_stride_B,
         .reloc_count = republishes ? 1u : 0u,
      },
      .coalesced_cmds_out = (uintptr_t)out_cmds,
      .sub_vdm_compact = sub_vdm_compact_base,
      .chunk_vdm_stride_B = chunk_vdm_stride_B,
      .indexed = indexed ? 1u : 0u,
   };
   if (republishes) {
      desc.base.relocs[0] = (struct libagx_indirect_ioto_reloc){
         .dest_lo = plan.relocs[0].dest[0],
         .dest_hi = plan.relocs[0].dest[1],
         .delta = plan.relocs[0].delta,
         .offset = plan.relocs[0].offset,
         .kind = 0u,
         .short_mask = plan.relocs[0].short_mask,
      };
   }

   /* Simulate all dispatched 256-lane workgroups cleanly. */
   const uint32_t eff_count = libagx_indirect_vdm_coalesce_count(&desc);
   const uint32_t active_wgs =
      eff_count ? DIV_ROUND_UP(eff_count, LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE)
                : 1u;
   uint32_t *wg_runs = calloc(num_wgs, sizeof(uint32_t));
   CHECK(wg_runs);

   for (uint32_t wg = 0; wg < num_wgs; ++wg) {
      const uint32_t wg_start = wg * LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE;
      if (wg_start >= eff_count) {
         if (wg == 0u) {
            uint32_t *vdm0 = (uint32_t *)sub_vdm;
            vdm0[0] = 0x80000000u | (uint32_t)(return_compact >> 32);
            vdm0[1] = (uint32_t)return_compact;
         }
         continue;
      }

      struct libagx_indirect_cmd_u32 sh_cmd[LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE];
      uint16_t sh_run_start[LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE];
      uint16_t sh_run_end[LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE];
      uint32_t num_nonempty = 0u;

      for (uint32_t lid = 0; lid < LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE; ++lid) {
         struct libagx_indirect_cmd_u32 c = {0};
         if (libagx_indirect_vdm_coalesce_chunk_load(&desc, wg, lid, eff_count,
                                                     &c)) {
            sh_cmd[num_nonempty++] = c;
         }
      }

      uint32_t num_runs = 0u;
      if (num_nonempty > 0u) {
         for (uint32_t lid = 0; lid < num_nonempty; ++lid) {
            uint32_t is_start = 0u, is_end = 0u;
            libagx_indirect_vdm_coalesce_chunk_classify(
               sh_cmd, lid, num_nonempty, &is_start, &is_end);
            const uint32_t run_id = num_runs;
            if (is_start) {
               sh_run_start[run_id] = (uint16_t)lid;
               num_runs++;
            }
            if (is_end)
               sh_run_end[(run_id + is_start) - 1u] = (uint16_t)lid;
         }
      }

      wg_runs[wg] = num_runs;
      for (uint32_t lid = 0; lid < LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE; ++lid) {
         libagx_indirect_vdm_coalesce_chunk_emit(&desc, wg, lid, eff_count,
                                                 num_runs, sh_cmd, sh_run_start,
                                                 sh_run_end);
      }
   }

   /* Walk the VDM chunk STREAM_LINK chain from chunk 0 until return_compact,
    * verifying every chunk's coalesced draws, Ioto relocations, and link hop. */
   struct libagx_indirect_cmd_u32 expected_chunk[LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE];
   uint64_t cur_compact = sub_vdm_compact_base;
   for (uint32_t wg = 0; wg < active_wgs; ++wg) {
      CHECK(cur_compact ==
            sub_vdm_compact_base + (uint64_t)wg * chunk_vdm_stride_B);
      const uint8_t *chunk_vdm = sub_vdm + (size_t)wg * chunk_vdm_stride_B;
      const uint32_t wg_start = wg * LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE;
      const uint32_t wg_len =
         wg_start < eff_count
            ? MIN2(LIBAGX_INDIRECT_VDM_COALESCE_WG_SIZE, eff_count - wg_start)
            : 0u;
      const uint32_t exp_runs =
         wg_len ? reference_coalesce(cmds + wg_start, wg_len, expected_chunk)
                : 0u;
      CHECK(wg_runs[wg] == exp_runs);

      for (uint32_t r = 0; r < exp_runs; ++r) {
         const uint32_t global_slot = wg_start + r;
         struct libagx_indirect_cmd_u32 got = libagx_indirect_cmd_load(
            (uintptr_t)out_cmds, cmd_size, global_slot, indexed);
         CHECK(got.count == expected_chunk[r].count);
         CHECK(got.instance_count == expected_chunk[r].instance_count);
         CHECK(got.first == expected_chunk[r].first);
         CHECK(got.bias == (indexed ? expected_chunk[r].bias : 0u));
         CHECK(got.first_instance == expected_chunk[r].first_instance);

         const uint8_t *rec = chunk_vdm + (size_t)r * vdm_record_dwords * 4u;
         const uint64_t expected_draw_va =
            (uintptr_t)out_cmds + (uint64_t)global_slot * cmd_size;
         if (republishes) {
            const uint64_t expected_prog_va =
               (uintptr_t)ioto_out + (uint64_t)global_slot * ioto_stride_B;
            uint64_t decoded_prog = 0;
            uint32_t decoded_tmpl = 0;
            CHECK(agx_vdm_state_vertex_pipeline_pair_unpack(
               chip, rec + 8, 8, &decoded_prog, &decoded_tmpl));
            CHECK(decoded_prog == expected_prog_va);
            CHECK(decoded_tmpl == raw_template);

            uint8_t expected_ioto[96];
            memcpy(expected_ioto, ioto_tmpl, ioto_len);
            struct agx3_ioto_uniform_range ranges[2] = {
               {.address = 0},
               {.address = expected_draw_va + param_off},
            };
            hk_g17p_patch_plan_apply(&plan, ranges, expected_ioto);
            CHECK(!memcmp(ioto_out + (size_t)global_slot * ioto_stride_B,
                          expected_ioto, ioto_len));
         }

         const uint8_t *body = rec + body_off;
         const size_t body_len = pos - body_off;
         if (indexed) {
            struct agx_vdm_index_list_indirect_indexed_info unpacked;
            CHECK(agx_vdm_index_list_indirect_indexed_unpack(
               chip, body, body_len, &unpacked));
            CHECK(unpacked.indirect_buffer == expected_draw_va);
            CHECK(unpacked.index_buffer == index_va);
         } else {
            struct agx_vdm_index_list_indirect_nonindexed_info unpacked;
            CHECK(agx_vdm_index_list_indirect_nonindexed_unpack(
               chip, body, body_len, &unpacked));
            CHECK(unpacked.indirect_buffer == expected_draw_va);
         }
      }

      const uint8_t *term = chunk_vdm + (size_t)exp_runs * vdm_record_dwords * 4u;
      uint64_t decoded_target = 0;
      CHECK(agx_vdm_stream_link_core_unpack(chip, term, 8, &decoded_target));
      if (wg + 1u == active_wgs) {
         CHECK(decoded_target == return_compact);
      } else {
         CHECK(decoded_target ==
               sub_vdm_compact_base + (uint64_t)(wg + 1u) * chunk_vdm_stride_B);
      }
      cur_compact = decoded_target;
   }
   CHECK(cur_compact == return_compact);

   /* Inactive tail chunks past active_wgs must remain completely untouched. */
   for (uint32_t wg = active_wgs; wg < num_wgs; ++wg) {
      const uint8_t *chunk_vdm = sub_vdm + (size_t)wg * chunk_vdm_stride_B;
      CHECK(chunk_vdm[0] == 0xaa && chunk_vdm[7] == 0xaa);
   }

   free(wg_runs);
   free(sub_vdm);
   free(ioto_out);
   free(out_cmds);
   free(in_buf);
}

static void
test_gpu_vdm_coalescer(void)
{
   /* 1. 1024 contiguous draws collapse into 4 chunk runs (256 instances each). */
   const uint32_t max_test_draws = 16384u;
   struct libagx_indirect_cmd_u32 *buf = calloc(max_test_draws, sizeof(*buf));
   CHECK(buf);
   for (uint32_t i = 0; i < 1024u; ++i)
      buf[i] = (struct libagx_indirect_cmd_u32){6u, 1u, 0u, 0u, i};
   run_coalesce_case(AGX_CHIP_G16G, false, true, false, 1024u, 0u, 16u, buf);
   run_coalesce_case(AGX_CHIP_G17P, true, true, false, 1024u, 0u, 24u, buf);
   run_coalesce_case(AGX_CHIP_G17S, false, false, true, 1024u, 700u, 20u, buf);
   /* Runtime count == 0 jumps directly to return_compact from chunk 0. */
   run_coalesce_case(AGX_CHIP_G16G, false, true, true, 1024u, 0u, 16u, buf);

   /* 2. All empty draws and alternating empty/non-empty chunks. */
   for (uint32_t i = 0; i < 1024u; ++i)
      buf[i] = (struct libagx_indirect_cmd_u32){(i & 1u) ? 0u : 6u,
                                                (i & 1u) ? 1u : 0u, 0u, 0u, i};
   run_coalesce_case(AGX_CHIP_G16G, false, true, false, 512u, 0u, 16u, buf);
   run_coalesce_case(AGX_CHIP_G16G, true, true, true, 512u, 0u, 20u, buf);
   /* Chunk 0 and chunk 2 completely empty; chunks 1 and 3 populated. */
   for (uint32_t i = 256u; i < 512u; ++i)
      buf[i] = (struct libagx_indirect_cmd_u32){6u, 1u, 0u, 0u, i};
   for (uint32_t i = 768u; i < 900u; ++i)
      buf[i] = (struct libagx_indirect_cmd_u32){12u, 2u, 3u, 1u, (i - 768u) * 2u};
   run_coalesce_case(AGX_CHIP_G16G, true, true, false, 1024u, 0u, 20u, buf);

   /* 3. Mixed culled/live runs, interior splits, non-contiguous firstInstance,
    * varying vertex/bias, and u32 wraparound/overflow boundaries. */
   for (uint32_t i = 0; i < 2048u; ++i) {
      if ((i % 7u) == 0u) {
         buf[i] = (struct libagx_indirect_cmd_u32){6u, 0u, 0u, 0u, i};
      } else if (i < 600u) {
         buf[i] = (struct libagx_indirect_cmd_u32){
            6u + (i & 2u), 1u, (i / 13u) & 1u, (i / 17u) & 1u, i * 2u};
      } else if (i < 1500u) {
         buf[i] = (struct libagx_indirect_cmd_u32){36u, 1u, 12u, 5u, i};
      } else {
         buf[i] = (struct libagx_indirect_cmd_u32){12u, 2u, 0u, 0u, 10000u + (i - 1500u) * 2u};
      }
   }
   buf[2044] = (struct libagx_indirect_cmd_u32){6u, 16u, 0u, 0u, 0xfffffff0u};
   buf[2045] = (struct libagx_indirect_cmd_u32){6u, 1u, 0u, 0u, 0u};
   buf[2046] = (struct libagx_indirect_cmd_u32){6u, 0x80u, 0u, 0u, 0xffffff00u};
   buf[2047] = (struct libagx_indirect_cmd_u32){6u, 0x100u, 0u, 0u, 0xffffff80u};

   run_coalesce_case(AGX_CHIP_G16G, false, true, false, 2048u, 0u, 16u, buf);
   run_coalesce_case(AGX_CHIP_G16G, true, true, false, 2048u, 0u, 28u, buf);
   run_coalesce_case(AGX_CHIP_G17P, false, true, true, 2048u, 2048u, 16u, buf);
   run_coalesce_case(AGX_CHIP_G17S, true, false, true, 2048u, 1234u, 20u, buf);

   /* 4. Boundary draw counts + cull_mdi shape (4096 visible of 16384 max). */
   const uint32_t edge_counts[] = {2u, 255u, 256u, 257u, 16384u};
   for (unsigned e = 0; e < ARRAY_SIZE(edge_counts); ++e) {
      const uint32_t n = edge_counts[e];
      uint32_t inst = 0u;
      for (uint32_t i = 0; i < n; ++i) {
         if ((i % 19u) == 5u) {
            buf[i] = (struct libagx_indirect_cmd_u32){0u, 1u, 0u, 0u, inst};
         } else {
            const uint32_t ic = 1u + (i % 3u);
            if ((i % 11u) == 0u)
               inst += 7u; /* break contiguity */
            buf[i] = (struct libagx_indirect_cmd_u32){6u, ic, 0u, 0u, inst};
            inst += ic;
         }
      }
      run_coalesce_case(AGX_CHIP_G16G, (e & 1u) != 0u, true, false, n, 0u,
                        (e & 1u) ? 20u : 16u, buf);
   }
   run_coalesce_case(AGX_CHIP_G16G, true, true, true, 16384u, 4096u, 20u, buf);
   run_coalesce_case(AGX_CHIP_G16G, true, true, true, 16384u, 99999u, 20u, buf);

   free(buf);
}

int
main(void)
{
   test_ldimm_encoding();
   test_chip_gating();
   test_gpu_vdm_coalescer();

   const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G16G, AGX_CHIP_G17S};
   const uint32_t counts[] = {0u, 1u, 3u, 8u, 20u};

   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned indexed = 0; indexed < 2; ++indexed) {
         for (unsigned repub = 0; repub < 2; ++repub) {
            for (unsigned draw_id = 0; draw_id <= repub; ++draw_id) {
               test_gpu_vdm_builder(chips[c], indexed, repub, draw_id, false,
                                    8u, 0u);
               for (unsigned k = 0; k < ARRAY_SIZE(counts); ++k) {
                  test_gpu_vdm_builder(chips[c], indexed, repub, draw_id, true,
                                       8u, counts[k]);
               }
            }
         }
      }
   }

   printf("indirect VDM builder: %u checks passed\n", checks);
   return 0;
}
