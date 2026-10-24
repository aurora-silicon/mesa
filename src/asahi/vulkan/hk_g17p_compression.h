/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "asahi/lib/agx_chip.h"
#include "hk_g17p_meta_copy_core.h"
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
#include <string.h>

/* G17P lossless compression policy. Admitted images use texture reads and
 * PBE/ZLS writes or metadata fast clears. Storage and host access stay
 * excluded. Colour formats are bounded to 8 bytes per sample, so compressed
 * single-sample attachments cannot spill the 120-byte tilebuffer.
 *
 * Layered colour allocations currently use one level: the compressed PBE's
 * secondary-view stride aliases its mip fields. Keep the existing ordinary
 * 2D mip path separate until layered mip stores can be represented.
 * Multisample colour images are admitted for transfer/sampling use only.
 * Application colour attachments may spill at 2x/4x, which requires an
 * in-place decompression path before raw eMRT stores. G17P has no such kernel.
 * Depth/stencil has independent ZLS data and metadata layer strides and does
 * not spill. Its single-sample mip chains select compression per level.
 *
 * External memory, WSI and DRM modifiers retain their sharing contracts;
 * shared colour admission is handled separately below.
 */

/* Colour formats admitted on G17P: plain single-plane formats of 1-8 bytes
 * whose physical G17P storage equals the API format (no RGB padding) and
 * whose channel class has a bitwise copy view. hk_can_compress() additionally
 * requires ail_can_compress() (renderable, at least 16x16). */
static inline bool
hk_g17p_compression_color_format(VkFormat format)
{
   switch (format) {
   /* 1 byte */
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   /* 2 bytes */
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   /* 4 bytes */
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   /* Not B10G11R11_UFLOAT: G17P packs it into one tile word and stores it
    * through an R32_UINT PBE (agx_g17p_rt0_pbe_format), a different
    * compression class from the R11G11B10 texture that reads it back. */
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   /* 8 bytes */
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static inline bool
hk_g17p_compression_depth_format(VkFormat format)
{
   return format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
          format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

/* Mesa-private WSI image marker (src/vulkan/wsi/wsi_common_private.h). */
#define HK_G17P_WSI_IMAGE_CREATE_INFO_MESA ((VkStructureType)1000001002)

/* Whether plane `plane` of an image created with `info` is compressed on
 * `chip`. There is no opt-out: every eligible G17P image takes the compressed
 * path. The generic hk_can_compress() checks (sparse, mutable-format
 * compatibility, ail_can_compress, the AGX-wide nocompress debug flag) still
 * apply. */
static inline bool
hk_g17p_compression_eligible(enum agx_chip chip, unsigned plane,
                             const VkImageCreateInfo *info)
{
   if (chip != AGX_CHIP_G17P || !info ||
       (plane != 0 && !(plane == 1 &&
                        info->format == VK_FORMAT_D32_SFLOAT_S8_UINT)))
      return false;

   /* Shared images keep their plain layout contract. Read each header by
    * copy: the chain's structures have other types, and Mesa builds with
    * strict aliasing (gcc -O2 dropped the stores of a stack chain). */
   for (const void *next = info->pNext; next;) {
      VkBaseInStructure ext;
      memcpy(&ext, next, sizeof(ext));
      if (ext.sType == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO &&
          ((const VkExternalMemoryImageCreateInfo *)next)->handleTypes)
         return false;
      if (ext.sType == HK_G17P_WSI_IMAGE_CREATE_INFO_MESA)
         return false;
      next = ext.pNext;
   }

   VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   if (info->imageType == VK_IMAGE_TYPE_2D)
      allowed_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   if (info->imageType == VK_IMAGE_TYPE_3D)
      allowed_flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
                       VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
   /* Zink marks every depth attachment sample-location compatible, including
    * single-sample images. This changes sample-location tracking, not the
    * depth plane's storage class.
    * Refusing the flag also makes the host-copy performance query advertise
    * a plain device layout, so zink adds HOST_TRANSFER and keeps depth plain.
    */
   if (hk_g17p_compression_depth_format(info->format))
      allowed_flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   /* Only usages whose every access goes through the texture unit, a PBE or
    * ZLS. Storage and host-transfer access is raw; feedback loops read what
    * is being written; video, shading-rate and density-map units have their
    * own (unknown) compression contracts. */
   VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

   /* Zink depth feedback reads use the depth plane's ZLS layout. Color
    * feedback keeps the plain layout required by texture barriers. */
   if (hk_g17p_compression_depth_format(info->format))
      allowed_usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   const bool multisampled = info->samples != VK_SAMPLE_COUNT_1_BIT;
   const bool three_d = info->imageType == VK_IMAGE_TYPE_3D;
   if ((info->imageType != VK_IMAGE_TYPE_2D && !three_d) ||
       !info->extent.depth || !info->arrayLayers ||
       (three_d ? (info->arrayLayers != 1 || multisampled)
                : info->extent.depth != 1) ||
       (multisampled && ((info->samples != VK_SAMPLE_COUNT_2_BIT &&
                          info->samples != VK_SAMPLE_COUNT_4_BIT) ||
                         info->mipLevels != 1)) ||
       ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
        (multisampled || info->extent.width != info->extent.height ||
         info->arrayLayers % 6)) ||
       info->tiling != VK_IMAGE_TILING_OPTIMAL || info->mipLevels < 1 ||
       (info->flags & ~allowed_flags) || (info->usage & ~allowed_usage) ||
       info->extent.width < 16 || info->extent.height < 16)
      return false;

   if (hk_g17p_compression_color_format(info->format)) {
      if ((three_d || info->arrayLayers > 1) && info->mipLevels != 1)
         return false;
      if (multisampled &&
          (info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)))
         return false;

      /* An RGBA8/BGRA8 UNORM mip chain the application never renders to is
       * written only by uploads and mip blits. A plain image takes each
       * upload as one raw compute copy and each mip level as a raw compute
       * dispatch (hk_g17p_compute_blit). A compressed one takes each upload
       * as a render pass, since G17P has no compute store that writes
       * compressed tiles, and its mips through hk_g17p_compressed_blit.
       * For small images the render passes dominate: 200 128x128 textures
       * uploaded and mipmapped in one command buffer took 2.6 times as
       * long compressed. From the size where a render pass also beats
       * compute for a mip level (HK_G17P_MIP_PBE_MIN_TEXELS), compression
       * wins: on the A18 Pro a 2048x2048 texture uploads in 0.55-0.64 of
       * the time, generates its mips in 0.40-0.57 and samples in
       * 0.69-0.84. Keep only the small ones plain. Images that are also
       * attachments are compressed at any size. */
      if ((info->format == VK_FORMAT_R8G8B8A8_UNORM ||
           info->format == VK_FORMAT_B8G8R8A8_UNORM) &&
          info->mipLevels > 1 &&
          !(info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
          !hk_g17p_mip_level_through_pbe(info->extent.width,
                                         info->extent.height))
         return false;
      return true;
   }

   return (hk_g17p_compression_depth_format(info->format) ||
           info->format == VK_FORMAT_S8_UINT) &&
          !three_d && info->extent.width <= 16384 &&
          info->extent.height <= 16384 && info->mipLevels <= 16;
}

/* G17P colour images shared through DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED.
 *
 * The modifier is the sharing contract. With one level, one layer and one
 * sample, the G17P (legacy AIL policy) compressed allocation is a function of
 * the format's block size and the extent alone: GPU tiling, the image, then
 * its metadata, in one memory plane. Exporter and importer therefore derive
 * the same layout, and import already follows the modifier through
 * hk_map_compression(). The packets that read and write it are the ones the
 * private policy above uses for single-level colour images: the compressed
 * 2D PBE (end-of-tile stores, meta copies and uploads) and the compressed
 * texture descriptor, with clears through the metadata fast clear.
 *
 * The shape is the private colour policy's restricted to one level, without
 * its exclusion of external memory and WSI images and without a
 * colour-attachment requirement: hk_image_init() makes every transfer
 * destination renderable, so a swapchain image written only by blits or
 * copies also takes the compressed PBE. WSI creates every swapchain image
 * with ALIAS; aliasing images of identical parameters derive the same
 * layout. Storage and host-transfer usages stay excluded. Feedback-loop
 * usage is accepted on import only (zink adds it to every imported
 * render-target dma-buf) and never chosen at allocation (hk_can_compress()
 * refuses it). hk_image.c adds the PBE store-class check.
 */
static inline bool
hk_g17p_shared_compression_shape(enum agx_chip chip, VkFormat format,
                                 VkImageType type, VkImageCreateFlags flags,
                                 VkImageUsageFlags usage)
{
   const VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
      VK_IMAGE_CREATE_ALIAS_BIT;
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return chip == AGX_CHIP_G17P && type == VK_IMAGE_TYPE_2D &&
          !(flags & ~allowed_flags) && !(usage & ~allowed_usage) &&
          hk_g17p_compression_color_format(format);
}

static inline bool
hk_g17p_shared_compression_eligible(enum agx_chip chip, unsigned plane,
                                    const VkImageCreateInfo *info)
{
   return plane == 0 && info &&
          info->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
          hk_g17p_shared_compression_shape(chip, info->format,
                                           info->imageType, info->flags,
                                           info->usage) &&
          info->extent.depth == 1 && info->mipLevels == 1 &&
          info->arrayLayers == 1 && info->samples == VK_SAMPLE_COUNT_1_BIT &&
          info->extent.width >= 16 && info->extent.height >= 16;
}
