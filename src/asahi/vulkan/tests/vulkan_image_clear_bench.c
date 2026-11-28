/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * GPU time of vkCmdClearColorImage on one layer of an uncompressed image.
 *
 * Per size: a 3-layer R32G32B32A32_UINT optimal-tiling image with transfer
 * and storage usage, cleared whole once, then 8 warm-up clears and 32 timed
 * clears of layer 1, with a transfer barrier between clears and GPU
 * timestamps around the timed ones. The image is then copied to a buffer and
 * every word of all three layers is checked: layer 1 must hold the last clear
 * colour, layers 0 and 2 the first one. Prints one line per size, clear_us
 * being the GPU time per clear.
 *
 * Arguments are image sides; a side S clears S*S*16 bytes per layer. The
 * default sides 16 128 512 1024 are 4 KiB, 256 KiB, 4 MiB and 16 MiB per
 * layer.
 *
 * To compare two drivers, run one process per sample with
 * VK_DRIVER_FILES naming each driver's ICD in turn, alternate which driver
 * goes first from round to round (A B, B A, ...), and compare the medians
 * per size over the rounds. A readback mismatch exits non-zero.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(call)                                                        \
   do {                                                                       \
      VkResult status = (call);                                               \
      if (status != VK_SUCCESS) {                                             \
         fprintf(stderr, "%s returned %d\n", #call, status);                  \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

static VkDevice device;
static VkPhysicalDevice gpu;
static VkQueue queue;
static VkCommandPool pool;
static VkQueryPool queries;
static float timestamp_period;
static uint32_t timestamp_bits;

static VkDeviceMemory
allocate(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(gpu, &props);
   for (unsigned i = 0; i < props.memoryTypeCount; i++) {
      if ((req.memoryTypeBits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & flags) == flags) {
         VkMemoryAllocateInfo info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = i,
         };
         VkDeviceMemory memory;
         VK_CHECK(vkAllocateMemory(device, &info, NULL, &memory));
         return memory;
      }
   }
   fprintf(stderr, "no suitable memory type\n");
   exit(1);
}

/* The clear colour of a step: distinct per step and component. */
static uint32_t
word(unsigned step, unsigned component)
{
   return (0x91ac35b7u + step * 0x636369du) ^ (0x93779b9u * (component + 1));
}

static void
clear(VkCommandBuffer cmd, VkImage image, unsigned first, unsigned layers,
      unsigned step)
{
   VkClearColorValue color;
   for (unsigned c = 0; c < 4; c++)
      color.uint32[c] = word(step, c);
   VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, first,
                                    layers};
   vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &color, 1,
                        &range);
}

static void
transfer_barrier(VkCommandBuffer cmd)
{
   VkMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask =
         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0,
                        NULL);
}

static void
measure(unsigned side)
{
   const unsigned layers = 3, warm = 8, timed = 32;
   const VkDeviceSize size = (VkDeviceSize)side * side * 16 * layers;

   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32G32B32A32_UINT,
      .extent = {side, side, 1},
      .mipLevels = 1,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      /* Storage usage keeps this 3-layer image uncompressed on every
       * generation, so clears take the fill paths, not a metadata clear. */
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   VK_CHECK(vkCreateImage(device, &ci, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, image, &req);
   VkDeviceMemory pixels = allocate(req, 0);
   VK_CHECK(vkBindImageMemory(device, image, pixels, 0));

   VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer out;
   VK_CHECK(vkCreateBuffer(device, &bi, NULL, &out));
   vkGetBufferMemoryRequirements(device, out, &req);
   VkDeviceMemory result =
      allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VK_CHECK(vkBindBufferMemory(device, out, result, 0));

   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

   VkImageMemoryBarrier layout = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &layout);
   vkCmdResetQueryPool(cmd, queries, 0, 2);

   clear(cmd, image, 0, layers, 0);
   for (unsigned step = 1; step <= warm + timed; step++) {
      transfer_barrier(cmd);
      if (step == warm + 1)
         vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             queries, 0);
      clear(cmd, image, 1, 1, step);
   }
   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);

   transfer_barrier(cmd);
   VkBufferImageCopy region = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
      .imageExtent = {side, side, 1},
   };
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, out, 1,
                          &region);
   VkMemoryBarrier host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0,
                        NULL);
   VK_CHECK(vkEndCommandBuffer(cmd));

   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   VK_CHECK(vkQueueWaitIdle(queue));

   uint64_t ts[2];
   VK_CHECK(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(ts), ts,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT |
                                     VK_QUERY_RESULT_WAIT_BIT));

   void *mapped;
   VK_CHECK(vkMapMemory(device, result, 0, size, 0, &mapped));
   const uint32_t *words = mapped;
   const size_t layer_words = (size_t)side * side * 4;
   for (unsigned layer = 0; layer < layers; layer++) {
      const unsigned step = layer == 1 ? warm + timed : 0;
      for (size_t i = 0; i < layer_words; i++) {
         if (words[layer * layer_words + i] != word(step, i % 4)) {
            fprintf(stderr, "readback failed at side %u layer %u word %zu\n",
                    side, layer, i);
            exit(1);
         }
      }
   }

   const uint64_t mask =
      timestamp_bits >= 64 ? UINT64_MAX : (UINT64_C(1) << timestamp_bits) - 1;
   const uint64_t elapsed = (ts[1] - ts[0]) & mask;
   if (!elapsed) {
      fprintf(stderr, "no elapsed time at side %u\n", side);
      exit(1);
   }
   printf("side=%u layer_bytes=%" PRIu64 " verified_words=%" PRIu64
          " clear_us=%.4f\n",
          side, (uint64_t)layer_words * 4, (uint64_t)(size / 4),
          elapsed * (double)timestamp_period / (1000.0 * timed));

   vkUnmapMemory(device, result);
   vkFreeCommandBuffers(device, pool, 1, &cmd);
   vkDestroyBuffer(device, out, NULL);
   vkFreeMemory(device, result, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, pixels, NULL);
}

int
main(int argc, char **argv)
{
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
   };
   VkInstance instance;
   VK_CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t ngpu = 1;
   VkResult res = vkEnumeratePhysicalDevices(instance, &ngpu, &gpu);
   if ((res != VK_SUCCESS && res != VK_INCOMPLETE) || !ngpu) {
      fprintf(stderr, "no Vulkan device\n");
      return 2;
   }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(gpu, &props);
   timestamp_period = props.limits.timestampPeriod;

   uint32_t count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   if (!families)
      return 2;
   vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families);
   unsigned family = 0;
   while (family < count &&
          (!(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) ||
           !families[family].timestampValidBits))
      family++;
   if (family == count) {
      fprintf(stderr, "no graphics queue with timestamps\n");
      return 2;
   }
   timestamp_bits = families[family].timestampValidBits;
   free(families);

   float priority = 1;
   VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qi,
   };
   VK_CHECK(vkCreateDevice(gpu, &di, NULL, &device));
   vkGetDeviceQueue(device, family, 0, &queue);
   VkCommandPoolCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VK_CHECK(vkCreateCommandPool(device, &pi, NULL, &pool));
   VkQueryPoolCreateInfo qc = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2,
   };
   VK_CHECK(vkCreateQueryPool(device, &qc, NULL, &queries));

   if (argc > 1) {
      for (int i = 1; i < argc; i++) {
         char *end;
         unsigned long side = strtoul(argv[i], &end, 0);
         if (*end || !side || side > 4096) {
            fprintf(stderr, "bad side %s\n", argv[i]);
            return 2;
         }
         measure((unsigned)side);
      }
   } else {
      static const unsigned sides[] = {16, 128, 512, 1024};
      for (unsigned i = 0; i < sizeof(sides) / sizeof(sides[0]); i++)
         measure(sides[i]);
   }

   vkDestroyQueryPool(device, queries, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
