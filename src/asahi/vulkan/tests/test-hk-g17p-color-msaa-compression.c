/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "asahi/vulkan/hk_g17p_compression.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
   fprintf(stderr, "line %u: %s\n", __LINE__, #x); exit(1); } } while (0)

static bool (*volatile eligible)(enum agx_chip, unsigned, unsigned,
                                const VkImageCreateInfo *) =
   hk_g17p_color_msaa_compression_eligible;

int main(void)
{
   const VkImageCreateInfo base = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {64, 64, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_2_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   const unsigned ids[] = {0x8140, 0x6040, 0x8103, 0x8112, 0x8132, 0};
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      for (unsigned id = 0; id < sizeof(ids) / sizeof(ids[0]); ++id)
         CHECK(eligible(chip, ids[id], 0, &base) ==
               (chip == AGX_CHIP_G17P && ids[id] == 0x8140));
   CHECK(!eligible(AGX_CHIP_G17P, 0x8140, 0, NULL));
   CHECK(!eligible(AGX_CHIP_G17P, 0x8140, 1, &base));
   unsigned formats = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      bool color = hk_g17p_compression_color_format(f);
      formats += color;
      for (unsigned samples = 1; samples <= 64; samples <<= 1) {
         for (unsigned layers = 0; layers <= 8; ++layers) {
            VkImageCreateInfo i = base;
            i.format = f; i.samples = samples; i.arrayLayers = layers;
            bool admitted = color && layers && (samples == 2 || samples == 4);
            CHECK(eligible(AGX_CHIP_G17P, 0x8140, 0, &i) == admitted);
            CHECK(!eligible(AGX_CHIP_G17P, 0x6040, 0, &i));
         }
      }
   }
   CHECK(formats == 46);
   VkImageCreateInfo i;
#define REJECT(field, value) do { i = base; i.field = (value); \
   CHECK(!eligible(AGX_CHIP_G17P, 0x8140, 0, &i)); } while (0)
   REJECT(imageType, VK_IMAGE_TYPE_3D);
   REJECT(extent.depth, 2);
   REJECT(extent.width, 15);
   REJECT(extent.height, 15);
   REJECT(mipLevels, 2);
   REJECT(tiling, VK_IMAGE_TILING_LINEAR);
   REJECT(flags, VK_IMAGE_CREATE_ALIAS_BIT);
   REJECT(flags, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
   REJECT(usage, base.usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
   REJECT(usage, base.usage | VK_IMAGE_USAGE_STORAGE_BIT);
   REJECT(usage, base.usage | VK_IMAGE_USAGE_HOST_TRANSFER_BIT);
   REJECT(usage, base.usage | VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
   };
   REJECT(pNext, &external);
   VkBaseInStructure wsi = {.sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA};
   REJECT(pNext, &wsi);
   i = base; i.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                          VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   CHECK(eligible(AGX_CHIP_G17P, 0x8140, 0, &i));
   i.usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
   CHECK(eligible(AGX_CHIP_G17P, 0x8140, 0, &i));
   printf("T8140 colour MSAA policy: %u checks, %u formats; T6040 excluded\n",
          checks, formats);
   return 0;
}
