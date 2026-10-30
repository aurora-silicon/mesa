/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/vulkan/hk_g17p_compression.h"
#include "asahi/vulkan/hk_g17p_spill.h"
#include "vulkan/util/vk_format.h"

/* Deliberately independent of assert(): the release build defines NDEBUG. */
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                       \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

struct footprint {
   enum pipe_format format;
   unsigned align_B, size_B;
   bool compressed[3];
};

static struct footprint classes[PIPE_FORMAT_COUNT];
static unsigned nr_classes;
static uint64_t layouts[3][2], spilling[3][2];

static struct footprint
footprint(enum pipe_format format)
{
   const enum pipe_format phys = ail_pixel_format[format].renderable;
   unsigned align_B = util_format_get_blocksize(phys);
   unsigned size_B = align_B;
   if (util_format_get_nr_components(phys) == 1)
      size_B *= util_format_get_nr_components(format);
   return (
      struct footprint){.format = format, .align_B = align_B, .size_B = size_B};
}

static unsigned
class_for(enum pipe_format format)
{
   const struct footprint f = footprint(format);
   for (unsigned c = 0; c < nr_classes; ++c) {
      if (classes[c].align_B == f.align_B && classes[c].size_B == f.size_B)
         return c;
   }
   CHECK(nr_classes < ARRAY_SIZE(classes));
   classes[nr_classes] = f;
   return nr_classes++;
}

static bool
compresses(VkFormat format, unsigned samples)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {64, 64, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = (VkSampleCountFlagBits)samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   return hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, &info) ||
          hk_g17p_color_msaa_compression_eligible(AGX_CHIP_G17P, 0x8140, 0,
                                                   &info);
}

static void
init_classes(void)
{
   /* Include every AIL colour-renderable pipe format, a superset of native
    * Vulkan's padded/scaled formats. The allocator sees only alignment and
    * size. Discover both rather than assuming that API bytes equal tile bytes.
    */
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      if (ail_pixel_format[f].renderable != PIPE_FORMAT_NONE &&
          !util_format_is_depth_or_stencil((enum pipe_format)f))
         class_for((enum pipe_format)f);
   }

   /* A class is considered compressed if ANY admitted API format uses it.
    * Thus collapsing formats cannot hide an unsafe compressed attachment.
    * All currently admitted colour formats belong to the core enum range.
    */
   unsigned admitted = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      if (!hk_g17p_compression_color_format((VkFormat)f))
         continue;
      ++admitted;
      enum pipe_format p = vk_format_to_pipe_format((VkFormat)f);
      CHECK(p != PIPE_FORMAT_NONE);
      CHECK(ail_pixel_format[p].renderable != PIPE_FORMAT_NONE);
      unsigned c = class_for(p);
      for (unsigned s = 0; s < 3; ++s)
         classes[c].compressed[s] |= compresses((VkFormat)f, 1u << s);
   }
   CHECK(admitted == 46);
   printf("allocator footprint classes: %u, policy colour formats: %u\n",
          nr_classes, admitted);
   for (unsigned c = 0; c < nr_classes; ++c)
      printf("  alignment=%u size=%u representative=%s\n", classes[c].align_B,
             classes[c].size_B, util_format_name(classes[c].format));
}

static struct agx_tilebuffer_layout
layout(const enum pipe_format *formats, unsigned count, unsigned samples,
       bool layered, bool reload)
{
   return agx_build_tilebuffer_layout_with_limits(
      formats, count, samples, layered,
      reload ? AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE
             : AGX_G17P_MAX_BYTES_PER_SAMPLE,
      32768, 2048);
}

static unsigned
spill_mask(const struct agx_tilebuffer_layout *tib)
{
   unsigned mask = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      mask |= tib->spilled[rt] << rt;
   return mask;
}

static void
check_layout(const enum pipe_format *formats, unsigned count,
             const unsigned *ids)
{
   for (unsigned s = 0; s < 3; ++s) {
      const unsigned samples = 1u << s;
      for (unsigned reload = 0; reload <= (samples > 1); ++reload) {
         for (unsigned layered = 0; layered < 2; ++layered) {
            struct agx_tilebuffer_layout tib =
               layout(formats, count, samples, layered, reload);
            ++layouts[s][reload];
            CHECK(tib.sample_size_B <= (reload ? 8 : AGX_G17P_MAX_BYTES_PER_SAMPLE));
            CHECK(agx_tilebuffer_total_size(&tib) <= 32768);
            CHECK(tib.tile_size * samples <= 2048);
            if (agx_tilebuffer_spills(&tib)) {
               ++spilling[s][reload];
               CHECK(hk_g17p_transient_spill_mask(AGX_CHIP_G17P, 0x8140, &tib) ==
                     spill_mask(&tib));
               CHECK(!hk_g17p_transient_spill_mask(AGX_CHIP_G13G, 0x8140, &tib));
               CHECK(!hk_g17p_transient_spill_mask(AGX_CHIP_G14G, 0x8140, &tib));
               CHECK(!hk_g17p_transient_spill_mask(AGX_CHIP_G17P, 0x6040, &tib));
               for (unsigned rt = 0; rt < count; ++rt) {
                  /* Compression is per image and no longer excludes a
                   * footprint merely because this pass spills. */
                  CHECK(classes[ids[rt]].compressed[s] ==
                        classes[ids[rt]].compressed[0]);
               }
            }
         }
      }
   }
}

/* The allocator stable-sorts by descending alignment. Enumerate every
 * resulting sequence, preserving ALL orders within each alignment class.
 * Permuting different alignments or inserting NONE slots only renames the
 * output slots. Trailing NONEs represent every attachment count from 0 to 8.
 * This is exhaustive over allocator inputs, not a random format sample.
 */
static void
matrix(enum pipe_format *formats, unsigned *ids, unsigned count,
       unsigned max_align)
{
   check_layout(formats, count, ids);
   if (count == AGX_MAX_RENDER_TARGETS)
      return;
   for (unsigned c = 0; c < nr_classes; ++c) {
      if (classes[c].align_B > max_align)
         continue;
      formats[count] = classes[c].format;
      ids[count] = c;
      matrix(formats, ids, count + 1, classes[c].align_B);
   }
   formats[count] = PIPE_FORMAT_NONE;
}

static void
check_witnesses(void)
{
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      formats[rt] = PIPE_FORMAT_R16G16B16A16_UINT;
   struct agx_tilebuffer_layout tib = layout(formats, 8, 4, false, false);
   CHECK(spill_mask(&tib) == 0xf0);
   puts("4x eight RGBA16_UINT: spill mask 0xf0");

   /* Exercise every placement of a small attachment, including a resident
    * target in a spilling pass under the stored-LOAD/multiview limit.
    */
   for (unsigned slot = 0; slot < AGX_MAX_RENDER_TARGETS; ++slot) {
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
         formats[rt] = rt == slot ? PIPE_FORMAT_R8G8B8A8_UNORM
                                  : PIPE_FORMAT_R32G32B32A32_FLOAT;
      for (unsigned layered = 0; layered < 2; ++layered) {
         tib = layout(formats, 8, 2, layered, false);
         CHECK(tib.spilled[slot]);
         tib = layout(formats, 8, 1, layered, false);
         CHECK(!agx_tilebuffer_spills(&tib));
      }
      formats[slot] = PIPE_FORMAT_R8_UNORM;
      tib = layout(formats, 8, 2, false, true);
      CHECK(agx_tilebuffer_spills(&tib) && !tib.spilled[slot]);
   }
   puts("2x seven RGBA32F plus RGBA8: small target spills in every slot");
   puts("2x stored-LOAD/multiview: resident R8 in a spilling pass covered");
}

static int
probe(int argc, char **argv)
{
   if (argc < 6 || argc > 5 + AGX_MAX_RENDER_TARGETS)
      return 2;
   unsigned samples = !strcmp(argv[2], "1")   ? 1
                      : !strcmp(argv[2], "2") ? 2
                      : !strcmp(argv[2], "4") ? 4
                                              : 0;
   bool reload = !strcmp(argv[3], "reload") || !strcmp(argv[3], "multiview");
   bool layered = !strcmp(argv[4], "layered");
   if (!samples || (reload && samples == 1) ||
       (!reload && strcmp(argv[3], "normal")) ||
       (!layered && strcmp(argv[4], "plain")))
      return 2;
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
   for (int i = 5; i < argc; ++i) {
      bool found = false;
      for (unsigned f = 0; f < PIPE_FORMAT_COUNT; ++f) {
         if (!strcmp(argv[i], util_format_name((enum pipe_format)f))) {
            formats[i - 5] = (enum pipe_format)f;
            found = true;
            break;
         }
      }
      if (!found ||
          (formats[i - 5] != PIPE_FORMAT_NONE &&
           (ail_pixel_format[formats[i - 5]].renderable == PIPE_FORMAT_NONE ||
            util_format_is_depth_or_stencil(formats[i - 5])))) {
         fprintf(stderr, "not a colour tilebuffer format: %s\n", argv[i]);
         return 2;
      }
   }
   struct agx_tilebuffer_layout tib =
      layout(formats, argc - 5, samples, layered, reload);
   printf("samples=%u mode=%s layered=%u sample_bytes=%u tile_pixels=%u "
          "spill_mask=0x%02x transient_mask=0x%02x\n",
          samples, argv[3], layered, tib.sample_size_B, tib.tile_size,
          spill_mask(&tib), hk_g17p_transient_spill_mask(AGX_CHIP_G17P, 0x8140, &tib));
   for (int rt = 0; rt < argc - 5; ++rt)
      printf("slot=%d format=%s spilled=%u\n", rt,
             util_format_name(formats[rt]), tib.spilled[rt]);
   return 0;
}

int
main(int argc, char **argv)
{
   if (argc > 1) {
      if (!strcmp(argv[1], "--probe")) {
         int rc = probe(argc, argv);
         if (!rc)
            return 0;
      }
      fprintf(stderr,
              "usage: %s [--probe 1|2|4 normal|reload|multiview "
              "plain|layered PIPE_FORMAT_... ...]\n",
              argv[0]);
      return 2;
   }
   init_classes();
   check_witnesses();
   const VkRect2D area = {{3, 7}, {31, 19}};
   for (unsigned store = 0; store < 2; ++store) {
      for (unsigned layer = 0; layer < 32; ++layer) {
         VkImageCopy2 copy = hk_g17p_spill_region(area, 2, 5, layer, store);
         CHECK(copy.srcOffset.x == 3 && copy.dstOffset.x == 3);
         CHECK(copy.srcOffset.y == 7 && copy.dstOffset.y == 7);
         CHECK(copy.extent.width == 31 && copy.extent.height == 19);
         CHECK(copy.extent.depth == 1);
         CHECK(copy.srcSubresource.mipLevel == (store ? 0 : 2));
         CHECK(copy.dstSubresource.mipLevel == (store ? 2 : 0));
         CHECK(copy.srcSubresource.baseArrayLayer == layer + (store ? 0 : 5));
         CHECK(copy.dstSubresource.baseArrayLayer == layer + (store ? 5 : 0));
         CHECK(copy.srcSubresource.layerCount == 1 &&
               copy.dstSubresource.layerCount == 1);
      }
   }
   struct agx_tilebuffer_layout unproven = {.nr_samples = 8, .spilled = {true}};
   CHECK(!hk_g17p_transient_spill_mask(AGX_CHIP_G17P, 0x8140, &unproven));
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
   unsigned ids[AGX_MAX_RENDER_TARGETS] = {0};
   matrix(formats, ids, 0, ~0u);
   for (unsigned s = 0; s < 3; ++s) {
      for (unsigned reload = 0; reload <= (s > 0); ++reload) {
         CHECK(layouts[s][reload] > 0 && spilling[s][reload] > 0);
         printf("%ux %s: layouts=%" PRIu64 " spilling=%" PRIu64 "\n", 1u << s,
                reload ? "reload/multiview" : "normal", layouts[s][reload],
                spilling[s][reload]);
      }
   }
   puts(
      "G17P spill matrix: compressed attachments admitted, all spills use transient backing");
   return 0;
}
