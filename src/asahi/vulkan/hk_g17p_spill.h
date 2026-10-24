/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_chip.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "vulkan/vulkan_core.h"

/* Candidate backing mask for ordinary private 2D attachments, including
 * plain images. The caller retains direct storage for alias/input/feedback.
 * Keep the legacy path on AGX2 and leave 8x unproven. */
static inline unsigned
hk_g17p_transient_spill_mask(enum agx_chip chip,
                            const struct agx_tilebuffer_layout *tib)
{
   if (chip != AGX_CHIP_G17P ||
       (tib->nr_samples != 1 && tib->nr_samples != 2 && tib->nr_samples != 4))
      return 0;

   unsigned mask = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      mask |= tib->spilled[rt] << rt;
   return mask;
}

/* Copy just the API rectangle and active layers. In particular a sparse
 * multiview mask must not write intervening layers of the attachment. */
static inline VkImageCopy2
hk_g17p_spill_region(VkRect2D area, unsigned level, unsigned base_layer,
                     unsigned layer, bool store)
{
   const VkImageSubresourceLayers original = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = level,
      .baseArrayLayer = base_layer + layer,
      .layerCount = 1,
   };
   const VkImageSubresourceLayers backing = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseArrayLayer = layer,
      .layerCount = 1,
   };
   return (VkImageCopy2){
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = store ? backing : original,
      .dstSubresource = store ? original : backing,
      .srcOffset = {area.offset.x, area.offset.y, 0},
      .dstOffset = {area.offset.x, area.offset.y, 0},
      .extent = {area.extent.width, area.extent.height, 1},
   };
}
