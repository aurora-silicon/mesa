/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_drm_render_core.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The GPU mappings of one test: memory objects, each mapped writable as a
 * whole, and sparse spans mapped alone. */
struct mapping {
   uint64_t start, end;
};

static struct mapping maps[64];
static unsigned map_count;

/* Whether [start, end) lies in the union of the mappings. */
static bool
mapped(uint64_t start, uint64_t end)
{
   while (start < end) {
      uint64_t next = start;
      for (unsigned i = 0; i < map_count; ++i) {
         if (maps[i].start <= start && start < maps[i].end &&
             maps[i].end > next)
            next = maps[i].end;
      }
      if (next == start)
         return false;
      start = next;
   }
   return true;
}

static void
check(const struct hk_drm_attachment_set *set, const uint64_t *in_start,
      const uint64_t *in_end, unsigned in_count, uint32_t max)
{
   struct drm_asahi_attachment out[HK_DRM_MAX_FRAGMENT_ATTACHMENTS];
   bool superset;
   uint32_t count = hk_drm_attachment_set_fit(set, max, out, &superset);

   assert(count <= max || (!max && !count));
   assert(count <= set->count);
   /* The kernel refuses a span it does not find mapped writable. */
   for (uint32_t i = 0; i < count; ++i)
      assert(out[i].size && mapped(out[i].pointer, out[i].pointer + out[i].size));
   if (!superset) {
      assert(count == max);
      return;
   }
   /* A superset: every captured span lies in one emitted span. */
   for (unsigned i = 0; i < in_count; ++i) {
      bool covered = false;
      for (uint32_t j = 0; j < count && !covered; ++j)
         covered = out[j].pointer <= in_start[i] &&
                   in_end[i] <= out[j].pointer + out[j].size;
      assert(covered);
   }
}

static uint64_t
rnd(uint64_t n)
{
   return ((uint64_t)rand() << 31 ^ (uint64_t)rand()) % n;
}

int
main(void)
{
   struct hk_drm_attachment_set set;
   uint64_t start[HK_DRM_MAX_FRAGMENT_ATTACHMENTS];
   uint64_t end[HK_DRM_MAX_FRAGMENT_ATTACHMENTS];
   bool superset;
   struct drm_asahi_attachment out[HK_DRM_MAX_FRAGMENT_ATTACHMENTS];

   /* Eighteen spans in nine memory objects, two each: nine safe merges, so
    * sixteen fit as a superset. */
   hk_drm_attachment_set_init(&set);
   map_count = 0;
   for (unsigned i = 0; i < 9; ++i) {
      const uint64_t base = 0x100000000ull + i * 0x1000000ull;
      maps[map_count++] = (struct mapping){base, base + 0x40000};
      for (unsigned j = 0; j < 2; ++j) {
         start[2 * i + j] = base + j * 0x20000;
         end[2 * i + j] = start[2 * i + j] + 0x10000;
         assert(hk_drm_attachment_set_add_mapped(&set, start[2 * i + j],
                                                 0x10000, base + 0x40000));
      }
   }
   assert(set.count == 18);
   assert(hk_drm_attachment_set_fit(&set, 16, out, &superset) == 16 && superset);
   check(&set, start, end, 18, 16);
   assert(hk_drm_attachment_set_fit(&set, 9, out, &superset) == 9 && superset);
   check(&set, start, end, 18, 9);

   /* Eighteen spans in eighteen separate objects with unmapped gaps: no merge
    * stays inside writable mappings, so the set cannot fit as a superset. */
   hk_drm_attachment_set_init(&set);
   map_count = 0;
   for (unsigned i = 0; i < 18; ++i) {
      const uint64_t base = 0x100000000ull + i * 0x1000000ull;
      maps[map_count++] = (struct mapping){base, base + 0x10000};
      start[i] = base;
      end[i] = base + 0x10000;
      assert(hk_drm_attachment_set_add_mapped(&set, base, 0x10000,
                                              base + 0x10000));
   }
   assert(hk_drm_attachment_set_fit(&set, 16, out, &superset) == 16 && !superset);
   check(&set, start, end, 18, 16);
   /* Within the limit the set is unchanged. */
   assert(hk_drm_attachment_set_fit(&set, 18, out, &superset) == 18 && superset);
   assert(!memcmp(out, set.list, 18 * sizeof(out[0])));

   /* hk_drm_attachment_set_add() maps the span alone; coalescing keeps the
    * later mapping end. */
   hk_drm_attachment_set_init(&set);
   assert(hk_drm_attachment_set_add(&set, 0x10000, 0x1000));
   assert(set.mapped_end[0] == 0x11000);
   assert(hk_drm_attachment_set_add_mapped(&set, 0x11000, 0x1000, 0x40000));
   assert(set.count == 1 && set.list[0].pointer == 0x10000 &&
          set.list[0].size == 0x2000 && set.mapped_end[0] == 0x40000);
   hk_drm_attachment_set_init(&set);
   assert(hk_drm_attachment_set_add_mapped(&set, 0x11000, 0x1000, 0x40000));
   assert(hk_drm_attachment_set_add(&set, 0x10000, 0x1000));
   assert(set.count == 1 && set.list[0].pointer == 0x10000 &&
          set.list[0].size == 0x2000 && set.mapped_end[0] == 0x40000);

   /* Random sets: objects with spans inside, sparse spans, any limit. */
   srand(1);
   for (unsigned round = 0; round < 20000; ++round) {
      unsigned n = 0;

      hk_drm_attachment_set_init(&set);
      map_count = 0;
      uint64_t base = 0x100000000ull;
      while (n < HK_DRM_MAX_FRAGMENT_ATTACHMENTS && map_count < 64) {
         const uint64_t size = (1 + rnd(16)) * 0x4000;
         const bool sparse = !rnd(5);

         base += rnd(3) * 0x4000; /* adjacent, or an unmapped gap */
         if (!sparse)
            maps[map_count++] = (struct mapping){base, base + size};
         for (unsigned k = 1 + rnd(3); k && n < HK_DRM_MAX_FRAGMENT_ATTACHMENTS; --k) {
            uint64_t s = base + rnd(size / 0x1000) * 0x1000;
            uint64_t e = s + 0x1000 * (1 + rnd((base + size - s) / 0x1000));
            if (e > base + size)
               e = base + size;
            if (sparse)
               maps[map_count++] = (struct mapping){s, e};
            if (!hk_drm_attachment_set_add_mapped(&set, s, e - s,
                                                  sparse ? 0 : base + size))
               break;
            start[n] = s;
            end[n] = e;
            n++;
            if (sparse || map_count >= 64)
               break;
         }
         base += size;
         if (!rnd(4))
            break;
      }
      for (uint32_t max = 0; max <= HK_DRM_MAX_FRAGMENT_ATTACHMENTS; ++max)
         check(&set, start, end, n, max);
   }

   printf("hk_attachment_fit: ok\n");
   return 0;
}
