/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strided storage-buffer word-field loads under VK_EXT_robustness2:
 * verifies shared-record lowering (buf[rec + 1], buf[rec + 3], plus buf[rec + 5]
 * in a dominated block) preserves bounds checking across:
 *   - a null storage-buffer descriptor (VK_NULL_HANDLE),
 *   - bounds ending mid-word and mid-record (every byte tail 1..80),
 *   - a record starting at the last in-bounds word,
 *   - records >= 2^30 whose scaled byte offset (rec << 2) wraps mod 2^32,
 *   - a nonzero binding offset across every byte tail 1..80 and VK_WHOLE_SIZE.
 *
 * usage: brl-vulkan-ssbo-word-fields [--shim]
 *   --shim   no-op DRM shim: compile, bind, record and submit without
 *            checking GPU readback
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "brl_vulkan_ssbo_word_fields_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define POISON_WORD   0xdeadbeefu
#define SENTINEL_WORD 0x55aa55aau

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool cmd_pool;
static VkCommandBuffer cmd;
static VkFence fence;
static VkDescriptorPool desc_pool;
static VkDescriptorSetLayout set_layout;
static VkDescriptorSet desc_set;
static VkPipelineLayout pipe_layout;
static VkPipeline pipe;

static unsigned failures, checks, launches;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         if (failures < 32) {                                                  \
            fprintf(stderr, "FAIL: " __VA_ARGS__);                             \
            fprintf(stderr, "\n");                                             \
         }                                                                     \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "no memory type for %#x\n", flags);
   exit(2);
}

static VkBuffer
create_host_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkDeviceMemory *mem_out, void **map_out)
{
   VkBuffer buf;
   CHECK(vkCreateBuffer(
      dev,
      &(VkBufferCreateInfo){
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = size,
         .usage = usage,
      },
      NULL, &buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buf, &req);
   CHECK(vkAllocateMemory(
      dev,
      &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = memory_type(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      },
      NULL, mem_out));
   CHECK(vkBindBufferMemory(dev, buf, *mem_out, 0));
   CHECK(vkMapMemory(dev, *mem_out, 0, size, 0, map_out));
   return buf;
}

/* 8-word-aligned record bases: small in-bounds/boundary records plus records
 * >= 2^30 where (rec << 2) wraps mod 2^32 to a small offset (0 or 32). */
static const uint32_t test_records[] = {
   0u,
   8u,
   16u,
   24u,
   0x3ffffff8u,
   0x40000000u,
   0x40000008u,
   0x80000000u,
   0xc0000000u,
   0xfffffff8u,
};

#define NUM_RECORDS (sizeof(test_records) / sizeof(test_records[0]))

static void
check_word(const char *label, bool is_null, VkDeviceSize offset,
           VkDeviceSize eff_range, VkDeviceSize robust_align,
           const uint32_t *data_words, uint32_t rec, unsigned field,
           uint32_t got)
{
   const uint32_t word_idx = rec + field;
   if (is_null) {
      EXPECT(got == 0,
             "%s rec=0x%08x+%u: got 0x%08x, expected 0 (null descriptor)",
             label, rec, field, got);
      return;
   }

   const uint64_t byte_start = (uint64_t)word_idx * 4u;
   const uint64_t byte_end = byte_start + 4u;
   if (byte_end <= (uint64_t)eff_range) {
      const uint32_t expected = data_words[(size_t)(offset / 4u) + word_idx];
      EXPECT(got == expected,
             "%s (off=%llu range=%llu) rec=0x%08x+%u: got 0x%08x, expected 0x%08x",
             label, (unsigned long long)offset, (unsigned long long)eff_range,
             rec, field, got, expected);
      return;
   }

   const uint64_t rounded_bytes =
      ((uint64_t)eff_range + robust_align - 1u) / robust_align * robust_align;
   if (byte_start >= rounded_bytes || robust_align == 1u) {
      /* Under robustStorageBufferAccessSizeAlignment == 1, a partial last word
       * or any out-of-bounds word takes the zero page and reads 0. */
      EXPECT(got == 0,
             "%s (off=%llu range=%llu) rec=0x%08x+%u: got 0x%08x, expected 0",
             label, (unsigned long long)offset, (unsigned long long)eff_range,
             rec, field, got);
      return;
   }

   const uint32_t expected = data_words[(size_t)(offset / 4u) + word_idx];
   EXPECT(got == expected || got == 0,
          "%s (off=%llu range=%llu) rec=0x%08x+%u: tail got 0x%08x, expected 0x%08x or 0",
          label, (unsigned long long)offset, (unsigned long long)eff_range,
          rec, field, got, expected);
}

static void
run_case(const char *label, VkBuffer data_buf, VkDeviceSize buf_size,
         VkDeviceSize offset, VkDeviceSize range, VkDeviceSize robust_align,
         uint32_t cond_mask, const uint32_t *data_words, uint32_t *rec_words,
         uint32_t *out_words, bool shim)
{
   rec_words[1] = cond_mask;

   const VkDescriptorBufferInfo data_info = {
      .buffer = data_buf,
      .offset = offset,
      .range = range,
   };
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &data_info,
   };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

   if (!shim) {
      for (size_t i = 0; i < NUM_RECORDS * 4u; i++)
         out_words[i] = POISON_WORD;
   }

   CHECK(vkResetCommandBuffer(cmd, 0));
   CHECK(vkBeginCommandBuffer(
      cmd, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
           }));
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0,
                           1, &desc_set, 0, NULL);
   vkCmdDispatch(cmd, (uint32_t)NUM_RECORDS, 1, 1);
   vkCmdPipelineBarrier(
      cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
      1,
      &(VkMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      },
      0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));

   CHECK(vkResetFences(dev, 1, &fence));
   CHECK(vkQueueSubmit(
      queue, 1,
      &(VkSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      },
      fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
   launches++;

   if (shim)
      return;

   const bool is_null = data_buf == VK_NULL_HANDLE;
   const VkDeviceSize eff_range =
      is_null ? 0 : (range == VK_WHOLE_SIZE ? buf_size - offset : range);

   for (size_t i = 0; i < NUM_RECORDS; i++) {
      const uint32_t rec = test_records[i];
      const uint32_t *got = &out_words[i * 4u];
      check_word(label, is_null, offset, eff_range, robust_align, data_words,
                 rec, 0u, got[0]);
      check_word(label, is_null, offset, eff_range, robust_align, data_words,
                 rec, 1u, got[1]);
      check_word(label, is_null, offset, eff_range, robust_align, data_words,
                 rec, 3u, got[2]);
      if ((cond_mask & (1u << (i & 31u))) != 0u) {
         check_word(label, is_null, offset, eff_range, robust_align, data_words,
                    rec, 5u, got[3]);
      } else {
         EXPECT(got[3] == SENTINEL_WORD,
                "%s (off=%llu range=%llu) rec=0x%08x+5 skipped: got 0x%08x, expected 0x%08x",
                label, (unsigned long long)offset,
                (unsigned long long)eff_range, rec, got[3], SENTINEL_WORD);
      }
   }
}

int
main(int argc, char **argv)
{
   bool shim = false;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else {
         fprintf(stderr, "unknown arg: %s\n", argv[i]);
         return 2;
      }
   }

   VkInstance instance;
   CHECK(vkCreateInstance(
      &(VkInstanceCreateInfo){
         .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pApplicationInfo =
            &(VkApplicationInfo){
               .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
               .apiVersion = VK_API_VERSION_1_3,
            },
      },
      NULL, &instance));

   uint32_t count = 1;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   VkPhysicalDeviceRobustness2FeaturesEXT rob2_avail = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 feats2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &rob2_avail,
   };
   vkGetPhysicalDeviceFeatures2(pdev, &feats2);
   if (!feats2.features.robustBufferAccess || !rob2_avail.robustBufferAccess2 ||
       !rob2_avail.nullDescriptor) {
      fprintf(stderr,
              "robustBufferAccess, robustBufferAccess2 or nullDescriptor unsupported\n");
      return 1;
   }

   VkPhysicalDeviceRobustness2PropertiesEXT rob2_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &rob2_props,
   };
   vkGetPhysicalDeviceProperties2(pdev, &props2);

   const VkDeviceSize ssbo_offset_align =
      props2.properties.limits.minStorageBufferOffsetAlignment;
   const VkDeviceSize robust_align =
      rob2_props.robustStorageBufferAccessSizeAlignment;
   if (robust_align == 0 || (robust_align & (robust_align - 1u)) != 0) {
      fprintf(stderr, "unexpected SSBO robust_align=%llu\n",
              (unsigned long long)robust_align);
      return 1;
   }

   uint32_t qf_count = 8;
   VkQueueFamilyProperties qf[8];
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qf);
   uint32_t qf_idx = 0;
   while (qf_idx < qf_count && !(qf[qf_idx].queueFlags & VK_QUEUE_COMPUTE_BIT))
      qf_idx++;
   if (qf_idx == qf_count) {
      fprintf(stderr, "no compute queue family\n");
      return 2;
   }

   const float prio = 1.0f;
   const char *exts[] = {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
   const VkPhysicalDeviceFeatures enabled_features = {
      .robustBufferAccess = VK_TRUE,
   };
   VkPhysicalDeviceRobustness2FeaturesEXT rob2_enable = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .robustBufferAccess2 = VK_TRUE,
      .nullDescriptor = VK_TRUE,
   };
   CHECK(vkCreateDevice(
      pdev,
      &(VkDeviceCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
         .pNext = &rob2_enable,
         .queueCreateInfoCount = 1,
         .pQueueCreateInfos =
            &(VkDeviceQueueCreateInfo){
               .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
               .queueFamilyIndex = qf_idx,
               .queueCount = 1,
               .pQueuePriorities = &prio,
            },
         .enabledExtensionCount = 1,
         .ppEnabledExtensionNames = exts,
         .pEnabledFeatures = &enabled_features,
      },
      NULL, &dev));
   vkGetDeviceQueue(dev, qf_idx, 0, &queue);

   CHECK(vkCreateCommandPool(
      dev,
      &(VkCommandPoolCreateInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
         .queueFamilyIndex = qf_idx,
      },
      NULL, &cmd_pool));
   CHECK(vkAllocateCommandBuffers(
      dev,
      &(VkCommandBufferAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = cmd_pool,
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      },
      &cmd));
   CHECK(vkCreateFence(
      dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
      NULL, &fence));

   const VkDeviceSize nonzero_offset =
      ssbo_offset_align > 256u ? ssbo_offset_align : 256u;
   const VkDeviceSize data_buf_size = nonzero_offset + 512u;
   VkDeviceMemory data_mem, rec_mem, out_mem;
   void *data_map, *rec_map, *out_map;
   VkBuffer data_buf = create_host_buffer(
      data_buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &data_mem, &data_map);
   VkBuffer rec_buf = create_host_buffer(
      (2u + NUM_RECORDS) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      &rec_mem, &rec_map);
   VkBuffer out_buf = create_host_buffer(
      NUM_RECORDS * 4u * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      &out_mem, &out_map);

   uint32_t *data_words = (uint32_t *)data_map;
   for (uint32_t i = 0; i < (uint32_t)(data_buf_size / 4u); i++)
      data_words[i] = 0xa5000000u | (i + 1u);

   uint32_t *rec_words = (uint32_t *)rec_map;
   rec_words[0] = (uint32_t)NUM_RECORDS;
   rec_words[1] = 0xffffffffu;
   memcpy(&rec_words[2], test_records, sizeof(test_records));

   const VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
   };
   CHECK(vkCreateDescriptorSetLayout(
      dev,
      &(VkDescriptorSetLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 3,
         .pBindings = bindings,
      },
      NULL, &set_layout));

   const VkDescriptorPoolSize pool_sizes[1] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
   };
   CHECK(vkCreateDescriptorPool(
      dev,
      &(VkDescriptorPoolCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 1,
         .pPoolSizes = pool_sizes,
      },
      NULL, &desc_pool));
   CHECK(vkAllocateDescriptorSets(
      dev,
      &(VkDescriptorSetAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = desc_pool,
         .descriptorSetCount = 1,
         .pSetLayouts = &set_layout,
      },
      &desc_set));

   const VkDescriptorBufferInfo rec_info = {rec_buf, 0, VK_WHOLE_SIZE};
   const VkDescriptorBufferInfo out_info = {out_buf, 0, VK_WHOLE_SIZE};
   const VkWriteDescriptorSet init_writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = desc_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &rec_info,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = desc_set,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &out_info,
      },
   };
   vkUpdateDescriptorSets(dev, 2, init_writes, 0, NULL);

   CHECK(vkCreatePipelineLayout(
      dev,
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &set_layout,
      },
      NULL, &pipe_layout));

   VkShaderModule shader;
   CHECK(vkCreateShaderModule(
      dev,
      &(VkShaderModuleCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(ssbo_word_fields_comp),
         .pCode = ssbo_word_fields_comp,
      },
      NULL, &shader));
   CHECK(vkCreateComputePipelines(
      dev, VK_NULL_HANDLE, 1,
      &(VkComputePipelineCreateInfo){
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage =
            {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
               .module = shader,
               .pName = "main",
            },
         .layout = pipe_layout,
      },
      NULL, &pipe));
   vkDestroyShaderModule(dev, shader, NULL);

   uint32_t *out_words = (uint32_t *)out_map;

   /* 1. Null descriptor. */
   run_case("null", VK_NULL_HANDLE, 0, 0, VK_WHOLE_SIZE, robust_align,
            0xffffffffu, data_words, rec_words, out_words, shim);
   run_case("null-conditional", VK_NULL_HANDLE, 0, 0, VK_WHOLE_SIZE, robust_align,
            0x55555555u, data_words, rec_words, out_words, shim);

   if (shim) {
      /* Under the no-op shim, submit representative descriptor shapes without
       * repeating all 166 launches through the same pipeline. */
      run_case("last-word", data_buf, data_buf_size, 0, 36, robust_align,
               0xffffffffu, data_words, rec_words, out_words, shim);
      run_case("mid-word", data_buf, data_buf_size, 0, 37, robust_align,
               0xffffffffu, data_words, rec_words, out_words, shim);
      run_case("mid-record", data_buf, data_buf_size, 0, 44, robust_align,
               0xffffffffu, data_words, rec_words, out_words, shim);
      run_case("nonzero-offset", data_buf, data_buf_size, nonzero_offset, 45,
               robust_align, 0x55555555u, data_words, rec_words, out_words,
               shim);
      run_case("whole", data_buf, data_buf_size, 0, VK_WHOLE_SIZE, robust_align,
               0xffffffffu, data_words, rec_words, out_words, shim);
      run_case("whole-nonzero-offset", data_buf, data_buf_size, nonzero_offset,
               VK_WHOLE_SIZE, robust_align, 0xffffffffu, data_words, rec_words,
               out_words, shim);
   } else {
      /* 2. Every byte range 1..80 at offset 0: covers mid-word tails,
       * mid-record bounds, and records starting at the last in-bounds word. */
      for (VkDeviceSize range = 1; range <= 80; range++) {
         run_case("tail-0", data_buf, data_buf_size, 0, range, robust_align,
                  0xffffffffu, data_words, rec_words, out_words, shim);
      }

      /* 3. Conditional dominated-block execution (taken and skipped) on key boundaries. */
      const VkDeviceSize div_ranges[] = {36, 37, 44, 53, 56, 64};
      for (size_t i = 0; i < sizeof(div_ranges) / sizeof(div_ranges[0]); i++) {
         run_case("conditional", data_buf, data_buf_size, 0, div_ranges[i],
                  robust_align, 0x55555555u, data_words, rec_words, out_words,
                  shim);
      }

      /* 4. Every byte range 1..80 at a nonzero binding offset. */
      for (VkDeviceSize range = 1; range <= 80; range++) {
         run_case("nonzero-offset", data_buf, data_buf_size, nonzero_offset,
                  range, robust_align, 0xffffffffu, data_words, rec_words,
                  out_words, shim);
      }

      /* 5. Full buffer range at offset 0 and nonzero offset (also verifies
       * wraparound records >= 2^30 against populated low words). */
      run_case("whole", data_buf, data_buf_size, 0, VK_WHOLE_SIZE, robust_align,
               0xffffffffu, data_words, rec_words, out_words, shim);
      run_case("whole-nonzero-offset", data_buf, data_buf_size, nonzero_offset,
               VK_WHOLE_SIZE, robust_align, 0xffffffffu, data_words, rec_words,
               out_words, shim);
   }

   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pipe_layout, NULL);
   vkDestroyDescriptorPool(dev, desc_pool, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroyBuffer(dev, out_buf, NULL);
   vkFreeMemory(dev, out_mem, NULL);
   vkDestroyBuffer(dev, rec_buf, NULL);
   vkFreeMemory(dev, rec_mem, NULL);
   vkDestroyBuffer(dev, data_buf, NULL);
   vkFreeMemory(dev, data_mem, NULL);
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, cmd_pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   printf("%u launches, %u checks, %u failures%s\n", launches, checks, failures,
          shim ? " (shim)" : "");
   return failures ? 1 : 0;
}
