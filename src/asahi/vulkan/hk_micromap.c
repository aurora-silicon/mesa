/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_EXT_opacity_micromap objects, builds and copies. The image format is in
 * libagx/micromap.h; acceleration structures link micromaps through their
 * OMM tables (hk_acceleration_structure.c), and G17P ray queries look them
 * up (agx_nir_gen1_ray_query.c).
 */
#include "hk_micromap.h"

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include "asahi/libagx/libagx.h"
#include "asahi/libagx/micromap.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "vk_log.h"

struct hk_micromap_layout {
   uint32_t triangles;
   uint32_t data_offset;
   uint64_t bytes;
   uint64_t scratch_bytes;
   struct libagx_micromap_bucket buckets[LIBAGX_MICROMAP_BUCKETS];
};

/* Lay out an image from the build's usage counts alone, as both the size
 * query and the build must. Buckets follow (format, level) order. Usage
 * counts of formats or levels no micromap can hold take no space; triangles
 * using them get invalid records. */
static bool
hk_micromap_layout(const VkMicromapBuildInfoEXT *info,
                   struct hk_micromap_layout *out)
{
   uint64_t counts[LIBAGX_MICROMAP_BUCKETS] = {0}, total = 0;
   for (uint32_t i = 0; i < info->usageCountsCount; ++i) {
      const VkMicromapUsageEXT *usage =
         info->pUsageCounts ? &info->pUsageCounts[i] : info->ppUsageCounts[i];
      total += usage->count;
      if (usage->format < VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT ||
          usage->format > VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT ||
          usage->subdivisionLevel > LIBAGX_MICROMAP_MAX_LEVEL)
         continue;
      counts[libagx_micromap_bucket_index(usage->format,
                                          usage->subdivisionLevel)] += usage->count;
   }
   if (total > UINT32_MAX / LIBAGX_MICROMAP_RECORD_BYTES)
      return false;

   *out = (struct hk_micromap_layout){.triangles = total};
   uint64_t cursor = ALIGN_POT(LIBAGX_MICROMAP_RECORDS +
                               total * LIBAGX_MICROMAP_RECORD_BYTES, 16);
   out->data_offset = cursor;
   uint64_t first_slot = 0;
   for (uint32_t b = 0; b < LIBAGX_MICROMAP_BUCKETS; ++b) {
      const uint32_t format = b / (LIBAGX_MICROMAP_MAX_LEVEL + 1) + 1;
      const uint32_t level = b % (LIBAGX_MICROMAP_MAX_LEVEL + 1);
      const uint64_t slot_bytes =
         ALIGN_POT(libagx_micromap_data_bytes(format, level), 4);
      out->buckets[b] = (struct libagx_micromap_bucket){
         .data_word = cursor / 4,
         .slot_words = slot_bytes / 4,
         .first_slot = first_slot,
         .count = counts[b],
      };
      cursor += counts[b] * slot_bytes;
      first_slot += counts[b];
   }
   /* Records hold 32-bit word offsets. */
   if (cursor / 4 > UINT32_MAX)
      return false;
   out->bytes = ALIGN_POT(cursor, 16);
   out->scratch_bytes =
      ALIGN_POT(LIBAGX_MICROMAP_SCRATCH_SLOTS + total * 4, 256);
   return true;
}

VKAPI_ATTR void VKAPI_CALL
hk_GetMicromapBuildSizesEXT(VkDevice device,
                            VkAccelerationStructureBuildTypeKHR buildType,
                            const VkMicromapBuildInfoEXT *pBuildInfo,
                            VkMicromapBuildSizesInfoEXT *pSizeInfo)
{
   struct hk_micromap_layout layout;
   if (!hk_micromap_layout(pBuildInfo, &layout)) {
      pSizeInfo->micromapSize = UINT64_MAX;
      pSizeInfo->buildScratchSize = UINT64_MAX;
   } else {
      pSizeInfo->micromapSize = layout.bytes;
      pSizeInfo->buildScratchSize = layout.scratch_bytes;
   }
   /* Acceleration structures reference the micromap. */
   pSizeInfo->discardable = VK_FALSE;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateMicromapEXT(VkDevice device, const VkMicromapCreateInfoEXT *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkMicromapEXT *pMicromap)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_buffer, buffer, pCreateInfo->buffer);
   struct hk_micromap *micromap = vk_object_zalloc(
      &dev->vk, pAllocator, sizeof(*micromap), VK_OBJECT_TYPE_MICROMAP_EXT);
   if (!micromap)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   micromap->buffer = buffer;
   micromap->offset = pCreateInfo->offset;
   micromap->size = pCreateInfo->size;
   micromap->type = pCreateInfo->type;
   *pMicromap = hk_micromap_to_handle(micromap);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyMicromapEXT(VkDevice device, VkMicromapEXT micromap,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_micromap, object, micromap);
   if (object)
      vk_object_free(&dev->vk, pAllocator, object);
}

/* Data copies launch at most this many threads. */
#define HK_MICROMAP_COPY_THREADS (1u << 30)

static void
hk_cmd_build_micromap(struct hk_cmd_buffer *cmd, const VkMicromapBuildInfoEXT *info)
{
   VK_FROM_HANDLE(hk_micromap, dst, info->dstMicromap);
   struct hk_micromap_layout layout;
   if (!dst || !hk_micromap_layout(info, &layout) || layout.bytes > dst->size) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const uint64_t image = hk_micromap_va(dst);
   const uint64_t scratch = info->scratchData.deviceAddress;
   const uint64_t triangles = info->triangleArray.deviceAddress;
   const uint64_t data = info->data.deviceAddress;
   const uint64_t stride = info->triangleArrayStride;
   uint64_t buckets = hk_pool_upload(cmd, layout.buckets, sizeof(layout.buckets), 16);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   libagx_micromap_clear(cmd, agx_1d(LIBAGX_MICROMAP_BUCKETS), AGX_BARRIER_ALL,
                         scratch);
   libagx_micromap_assign(cmd, agx_1d(MAX2(layout.triangles, 1)), AGX_BARRIER_ALL,
                          image, triangles, stride, scratch, buckets,
                          layout.triangles, layout.bytes, layout.data_offset);
   for (uint32_t b = 0; b < LIBAGX_MICROMAP_BUCKETS; ++b) {
      const struct libagx_micromap_bucket *bucket = &layout.buckets[b];
      if (!bucket->count)
         continue;
      const uint32_t format = b / (LIBAGX_MICROMAP_MAX_LEVEL + 1) + 1;
      const uint32_t level = b % (LIBAGX_MICROMAP_MAX_LEVEL + 1);
      const uint32_t data_bytes = libagx_micromap_data_bytes(format, level);
      const uint32_t per_launch =
         MAX2(HK_MICROMAP_COPY_THREADS / bucket->slot_words, 1);
      for (uint32_t first = 0; first < bucket->count; first += per_launch) {
         const uint32_t slots = MIN2(per_launch, bucket->count - first);
         const uint32_t words = slots * bucket->slot_words;
         libagx_micromap_copy_data(cmd, agx_1d(words), AGX_BARRIER_ALL, image,
                                   triangles, stride, data, scratch, b,
                                   bucket->first_slot, first,
                                   bucket->data_word + first * bucket->slot_words,
                                   bucket->slot_words, data_bytes, words);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildMicromapsEXT(VkCommandBuffer commandBuffer, uint32_t infoCount,
                        const VkMicromapBuildInfoEXT *pInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   for (uint32_t i = 0; i < infoCount && !vk_command_buffer_has_error(&cmd->vk); ++i)
      hk_cmd_build_micromap(cmd, &pInfos[i]);
}

static uint64_t
hk_micromap_serial_header(struct hk_cmd_buffer *cmd)
{
   struct hk_physical_device *pdev = hk_device_physical(hk_cmd_buffer_device(cmd));
   const uint8_t format_uuid[VK_UUID_SIZE] = LIBAGX_MICROMAP_SERIAL_FORMAT_UUID_BYTES;
   uint8_t header[LIBAGX_MICROMAP_SERIAL_HEADER_BYTES];
   memcpy(header, pdev->vk.properties.driverUUID, VK_UUID_SIZE);
   memcpy(header + VK_UUID_SIZE, format_uuid, VK_UUID_SIZE);
   return hk_pool_upload(cmd, header, sizeof(header), 16);
}

/* Copy an image of at most limit bytes; the source header gives its size. */
static void
hk_cmd_copy_micromap_image(struct hk_cmd_buffer *cmd, uint64_t dst, uint64_t src,
                           uint64_t limit, uint64_t prefix_dst, uint64_t prefix)
{
   const uint64_t threads = MAX2(DIV_ROUND_UP(limit, 16),
                                 LIBAGX_MICROMAP_SERIAL_HEADER_BYTES / 4);
   if (threads > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   libagx_micromap_copy(cmd, agx_1d(threads), AGX_BARRIER_ALL, dst, src, limit,
                        prefix_dst, prefix);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyMicromapEXT(VkCommandBuffer commandBuffer,
                      const VkCopyMicromapInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_micromap, src, pInfo->src);
   VK_FROM_HANDLE(hk_micromap, dst, pInfo->dst);
   /* An image has no unused space, so compaction is a clone. */
   if (pInfo->mode != VK_COPY_MICROMAP_MODE_CLONE_EXT &&
       pInfo->mode != VK_COPY_MICROMAP_MODE_COMPACT_EXT) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   hk_cmd_copy_micromap_image(cmd, hk_micromap_va(dst), hk_micromap_va(src),
                              MIN2(src->size, dst->size), 0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyMicromapToMemoryEXT(VkCommandBuffer commandBuffer,
                              const VkCopyMicromapToMemoryInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_micromap, src, pInfo->src);
   if (pInfo->mode != VK_COPY_MICROMAP_MODE_SERIALIZE_EXT) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const uint64_t dst = pInfo->dst.deviceAddress;
   const uint64_t header = hk_micromap_serial_header(cmd);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   hk_cmd_copy_micromap_image(cmd, dst + LIBAGX_MICROMAP_SERIAL_HEADER_BYTES,
                              hk_micromap_va(src), src->size, dst, header);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyMemoryToMicromapEXT(VkCommandBuffer commandBuffer,
                              const VkCopyMemoryToMicromapInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_micromap, dst, pInfo->dst);
   if (pInfo->mode != VK_COPY_MICROMAP_MODE_DESERIALIZE_EXT) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   hk_cmd_copy_micromap_image(cmd, hk_micromap_va(dst),
                              pInfo->src.deviceAddress + LIBAGX_MICROMAP_SERIAL_HEADER_BYTES,
                              dst->size, 0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceMicromapCompatibilityEXT(VkDevice device,
                                     const VkMicromapVersionInfoEXT *pVersionInfo,
                                     VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   const uint8_t format_uuid[VK_UUID_SIZE] = LIBAGX_MICROMAP_SERIAL_FORMAT_UUID_BYTES;
   const bool compatible =
      !memcmp(pVersionInfo->pVersionData, pdev->vk.properties.driverUUID, VK_UUID_SIZE) &&
      !memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE, format_uuid, VK_UUID_SIZE);
   *pCompatibility = compatible ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                                : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/* Host micromap commands need micromapHostCommands, which is not advertised. */
VKAPI_ATTR VkResult VKAPI_CALL
hk_BuildMicromapsEXT(VkDevice device, VkDeferredOperationKHR deferredOperation,
                     uint32_t infoCount, const VkMicromapBuildInfoEXT *pInfos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMicromapEXT(VkDevice device, VkDeferredOperationKHR deferredOperation,
                   const VkCopyMicromapInfoEXT *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMicromapToMemoryEXT(VkDevice device, VkDeferredOperationKHR deferredOperation,
                           const VkCopyMicromapToMemoryInfoEXT *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMemoryToMicromapEXT(VkDevice device, VkDeferredOperationKHR deferredOperation,
                           const VkCopyMemoryToMicromapInfoEXT *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_WriteMicromapsPropertiesEXT(VkDevice device, uint32_t micromapCount,
                               const VkMicromapEXT *pMicromaps,
                               VkQueryType queryType, size_t dataSize,
                               void *pData, size_t stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
