/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/lib/agx_bg_eot.h"
#include "asahi/lib/agx_bg_eot_ioto.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "util/ralloc.h"

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static unsigned live_bos;
static unsigned total_bos_created;
static uint64_t next_exec_va = 0x10000000ull;
static uint64_t next_data_va = 0x200000000ull;

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
void __wrap_agx_bo_unreference(struct agx_device *, struct agx_bo *);

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, size_t size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   struct agx_va *va = calloc(1, sizeof(*va));
   assert(bo && va);
   bo->dev = dev;
   bo->size = size;
   bo->_map = calloc(1, size);
   bo->_map_size = size;
   assert(bo->_map);
   va->size_B = size;
   if (flags & AGX_BO_EXEC) {
      va->addr = next_exec_va;
      next_exec_va += size + 0x10000ull;
   } else {
      va->addr = next_data_va;
      next_data_va += size + 0x10000ull;
   }
   bo->va = va;
   live_bos++;
   total_bos_created++;
   return bo;
}

void
__wrap_agx_bo_unreference(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo)
      return;
   assert(live_bos > 0);
   live_bos--;
   free(bo->_map);
   free(bo->va);
   free(bo);
}

struct mock_build_ctx {
   unsigned calls;
   bool upload_data;
   bool fail;
   uint64_t rsrc_spec;
   struct agx_bo *shader_bo;
};

static bool
mock_build_cb(void *ctx, const struct agx_bg_eot_ioto_sink *sink,
              struct agx_bg_eot_ioto_result *out)
{
   struct mock_build_ctx *m = ctx;
   m->calls++;

   struct agx_bg_eot_ioto_reservation res = {0};
   if (!sink->reserve(sink->ctx, 192, "test ioto", &res))
      return false;

   struct agx_bo *data_bo = NULL;
   if (m->upload_data) {
      const uint32_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
      const uint64_t va =
         sink->upload(sink->ctx, payload, sizeof(payload), 32, &data_bo);
      if (!va)
         return false;
   }

   if (m->fail)
      return false;

   memset(res.map, 0xab, 128);
   const uint64_t published = sink->publish(sink->ctx, &res, 128);
   if (!published)
      return false;

   *out = (struct agx_bg_eot_ioto_result){
      .program_va = published,
      .rsrc_spec = m->rsrc_spec,
      .program_bo = res.bo,
      .data_bo = data_bo,
      .shader_bo = m->shader_bo,
   };
   return true;
}

static int
test_chip_gating_and_lifecycle(void)
{
   struct agx_device dev_g13 = {.chip = AGX_CHIP_G13G};
   struct agx_device dev_m5 = {.chip = AGX_CHIP_G17G};
   struct agx_device dev_g18 = {.chip = AGX_CHIP_G18G};
   struct agx_device dev_g17p = {.chip = AGX_CHIP_G17P};

   CHECK(agx_bg_eot_ioto_cache_create(&dev_g13) == NULL);
   CHECK(agx_bg_eot_ioto_cache_create(&dev_m5) == NULL);
   CHECK(agx_bg_eot_ioto_cache_create(&dev_g18) == NULL);

   struct agx_bg_eot_ioto_cache *dc = agx_bg_eot_ioto_cache_create(&dev_g17p);
   CHECK(dc != NULL);
   CHECK(live_bos == 0);
   agx_bg_eot_ioto_cache_destroy(&dev_g17p, dc);
   CHECK(live_bos == 0);
   return 0;
}

static int
test_multi_variant_and_single_shot_cache(void)
{
   struct agx_device dev = {.chip = AGX_CHIP_G17P};
   struct agx_bo fake_shader_bo = {.size = 4096};
   struct agx_bg_eot_ioto_cache *dc = agx_bg_eot_ioto_cache_create(&dev);
   CHECK(dc != NULL);

   /* 1. Multi-variant pass cache (Vulkan flow). */
   struct agx_bg_eot_ioto_cache_key k1;
   memset(&k1, 0, sizeof(k1));
   k1.color_attachment_count = 1;
   k1.clear[0] = 0x3f800000u;
   k1.eot[0] = 0x42;

   struct agx_bg_eot_ioto_result progs[AGX_BG_EOT_IOTO_CACHE_VARIANTS];
   uint8_t valid = 0xff;
   struct agx_bg_eot_ioto_cache_entry *e1 =
      agx_bg_eot_ioto_cache_get(dc, &k1, progs, &valid);
   CHECK(e1 != NULL);
   CHECK(valid == 0);

   struct mock_build_ctx b0 = {
      .upload_data = true,
      .rsrc_spec = 0x1111ull,
      .shader_bo = &fake_shader_bo,
   };
   struct agx_bg_eot_ioto_result r0 = {0};
   bool in_cache = false;
   CHECK(agx_bg_eot_ioto_cache_lookup_or_build(&dev, dc, e1, 0, mock_build_cb,
                                               &b0, &r0, &in_cache));
   CHECK(in_cache);
   CHECK(b0.calls == 1);
   CHECK(r0.program_va != 0);
   CHECK(r0.rsrc_spec == 0x1111ull);
   CHECK(r0.program_bo != NULL);
   CHECK(r0.data_bo != NULL);
   CHECK(r0.shader_bo == &fake_shader_bo);

   /* Lookup of variant 0 hits without calling build. */
   struct agx_bg_eot_ioto_result r0_hit = {0};
   CHECK(agx_bg_eot_ioto_cache_lookup_or_build(&dev, dc, e1, 0, mock_build_cb,
                                               &b0, &r0_hit, &in_cache));
   CHECK(in_cache);
   CHECK(b0.calls == 1);
   CHECK(r0_hit.program_va == r0.program_va);
   CHECK(r0_hit.program_bo == r0.program_bo);
   CHECK(r0_hit.data_bo == r0.data_bo);
   CHECK(r0_hit.shader_bo == r0.shader_bo);

   /* Failed build on variant 2 rolls back without caching. */
   struct mock_build_ctx b2_fail = {
      .upload_data = true,
      .fail = true,
      .rsrc_spec = 0x2222ull,
      .shader_bo = &fake_shader_bo,
   };
   struct agx_bg_eot_ioto_result r2_fail = {0};
   CHECK(agx_bg_eot_ioto_cache_lookup_or_build(&dev, dc, e1, 2, mock_build_cb,
                                               &b2_fail, &r2_fail, &in_cache));
   CHECK(!in_cache);
   CHECK(r2_fail.program_va == 0);
   CHECK(b2_fail.calls == 1);

   /* Subsequent valid build on variant 2 succeeds and caches. */
   struct mock_build_ctx b2 = {
      .upload_data = false,
      .rsrc_spec = 0x2222ull,
      .shader_bo = &fake_shader_bo,
   };
   struct agx_bg_eot_ioto_result r2 = {0};
   CHECK(agx_bg_eot_ioto_cache_lookup_or_build(&dev, dc, e1, 2, mock_build_cb,
                                               &b2, &r2, &in_cache));
   CHECK(in_cache);
   CHECK(b2.calls == 1);
   CHECK(r2.program_va != 0 && r2.program_va != r0.program_va);
   CHECK(r2.program_bo != NULL);
   CHECK(r2.data_bo == NULL); /* no data uploaded on variant 2 */

   /* Second pass lookup gets both variants pre-populated. */
   struct agx_bg_eot_ioto_cache_entry *e1_again =
      agx_bg_eot_ioto_cache_get(dc, &k1, progs, &valid);
   CHECK(e1_again == e1);
   CHECK(valid == (BITFIELD_BIT(0) | BITFIELD_BIT(2)));
   CHECK(progs[0].program_va == r0.program_va);
   CHECK(progs[2].program_va == r2.program_va);

   /* 2. Single-shot lookup+build (Gallium flow) with distinct key. */
   struct agx_bg_eot_ioto_cache_key k2 = k1;
   k2.flags = 0x101u;
   struct mock_build_ctx bg = {
      .upload_data = true,
      .rsrc_spec = 0x3333ull,
      .shader_bo = &fake_shader_bo,
   };
   struct agx_bg_eot_ioto_result rg1 = {0}, rg2 = {0};
   CHECK(agx_bg_eot_ioto_cache_get_or_build(&dev, dc, &k2, 0, mock_build_cb,
                                            &bg, &rg1, &in_cache));
   CHECK(in_cache && bg.calls == 1 && rg1.program_va != 0);
   CHECK(rg1.program_va != r0.program_va);

   /* Exact same key hits without calling builder. */
   CHECK(agx_bg_eot_ioto_cache_get_or_build(&dev, dc, &k2, 0, mock_build_cb,
                                            &bg, &rg2, &in_cache));
   CHECK(in_cache && bg.calls == 1);
   CHECK(rg2.program_va == rg1.program_va);
   CHECK(rg2.program_bo == rg1.program_bo);
   CHECK(rg2.data_bo == rg1.data_bo);
   CHECK(rg2.shader_bo == rg1.shader_bo);

   /* 3. Fill remaining entries to capacity and verify clean fallback. */
   for (unsigned i = 2; i < AGX_BG_EOT_IOTO_CACHE_ENTRIES; ++i) {
      struct agx_bg_eot_ioto_cache_key ki = k2;
      ki.clear[3] = i + 100u;
      struct mock_build_ctx bi = {
         .upload_data = true,
         .rsrc_spec = i,
         .shader_bo = &fake_shader_bo,
      };
      struct agx_bg_eot_ioto_result ri = {0};
      CHECK(agx_bg_eot_ioto_cache_get_or_build(&dev, dc, &ki, 0, mock_build_cb,
                                               &bi, &ri, &in_cache));
      CHECK(in_cache && bi.calls == 1 && ri.program_va != 0);
   }

   /* 257th distinct key returns false (caller falls back to pool). */
   struct agx_bg_eot_ioto_cache_key k_overflow = k2;
   k_overflow.clear[3] = 999999u;
   struct mock_build_ctx b_ov = {.rsrc_spec = 0x9999ull};
   struct agx_bg_eot_ioto_result r_ov = {0};
   CHECK(!agx_bg_eot_ioto_cache_get_or_build(&dev, dc, &k_overflow, 0,
                                             mock_build_cb, &b_ov, &r_ov,
                                             &in_cache));
   CHECK(!in_cache && b_ov.calls == 0);

   /* Existing entries still hit even at full capacity. */
   CHECK(agx_bg_eot_ioto_cache_get_or_build(&dev, dc, &k2, 0, mock_build_cb,
                                            &bg, &rg2, &in_cache));
   CHECK(in_cache && bg.calls == 1 && rg2.program_va == rg1.program_va);

   CHECK(live_bos > 0);
   agx_bg_eot_ioto_cache_destroy(&dev, dc);
   CHECK(live_bos == 0);
   return 0;
}

static int
test_helpers_and_m5_template_parity(void)
{
   /* Packed UNORM8 clear: clamp, NaN -> 0, exact 0..255 mapping. */
   const float c1[4] = {1.0f, 0.0f, 0.5f, NAN};
   const uint32_t p1 = agx_g17p_pack_unorm8_clear(c1);
   CHECK((p1 & 0xffu) == 0xffu);
   CHECK(((p1 >> 8) & 0xffu) == 0x00u);
   CHECK(((p1 >> 16) & 0xffu) == 0x80u);
   CHECK(((p1 >> 24) & 0xffu) == 0x00u);

   /* Uniform op check ignores NONE format holes. */
   struct agx_bg_eot_key ukey = {0};
   ukey.tib.logical_format[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
   ukey.tib.logical_format[2] = PIPE_FORMAT_R8G8B8A8_UNORM;
   ukey.op[0] = AGX_BG_CLEAR;
   ukey.op[2] = AGX_BG_CLEAR;
   CHECK(agx_g17p_bg_eot_ops_uniform(&ukey));
   ukey.op[2] = AGX_BG_LOAD;
   CHECK(!agx_g17p_bg_eot_ops_uniform(&ukey));

   /* M5 tile template cache parity vs fresh generation across shifted ranges. */
   struct agx_device dev = {.chip = AGX_CHIP_G17G, .debug = AGX_DBG_VERIFY};
   struct agx_bg_eot_cache cache = {0};
   cache.dev = &dev;
   simple_mtx_init(&cache.lock, mtx_plain);
   struct agx_bg_eot_shader *shader = rzalloc(NULL, struct agx_bg_eot_shader);
   CHECK(shader != NULL);

   struct agx3_ioto_uniform_range r1[2] = {
      {.start_halfs = 0, .size_halfs = 4, .address = 0x100001000ull},
      {.start_halfs = 4, .size_halfs = 4, .address = 0x100001008ull},
   };
   struct agx3_g17p_graphics_program_desc d = {
      .program_va = 0x10004000ull,
      .shader_va = 0x10008000ull,
      .shader_binary_size = 64,
      .nr_gprs = 8,
      .compiler_d_valid = true,
      .compiler_d = 16,
      .fragment = true,
      .imageblock = {.sample_size_B = 8, .nr_samples = 1, .tile_size_px = 1024},
      .uniform_ranges = r1,
      .uniform_range_count = 2,
   };
   struct agx3_g17p_graphics_program p_first = {0}, p_second = {0}, p_ref = {0};
   CHECK(agx_bg_eot_m5_tile_program(&cache, shader, &d, false, &p_first));
   CHECK(shader->m5_tile_templates != NULL);

   /* Shift both clustered ranges by the same delta and re-apply template. */
   struct agx3_ioto_uniform_range r2[2] = {
      {.start_halfs = 0, .size_halfs = 4, .address = 0x100025000ull},
      {.start_halfs = 4, .size_halfs = 4, .address = 0x100025008ull},
   };
   d.uniform_ranges = r2;
   CHECK(agx_bg_eot_m5_tile_program(&cache, shader, &d, false, &p_second));
   CHECK(agx3_g17g_tile_program_generate(AGX_CHIP_G17G, &d, false, &p_ref));
   CHECK(p_second.length == p_ref.length);
   CHECK(memcmp(p_second.bytes, p_ref.bytes, p_ref.length) == 0);

   ralloc_free(shader);
   simple_mtx_destroy(&cache.lock);
   return 0;
}

int
main(void)
{
   if (test_chip_gating_and_lifecycle())
      return 1;
   if (test_multi_variant_and_single_shot_cache())
      return 1;
   if (test_helpers_and_m5_template_parity())
      return 1;
   printf("PASS: test-agx-bg-eot-ioto-cache (created %u fake BOs, 0 leaked)\n",
          total_bos_created);
   return 0;
}
