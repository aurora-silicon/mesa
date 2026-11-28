/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Indexed vec4 uniform-buffer reads under VK_EXT_robustness2:
 * verifies element-limit rounding to robustUniformBufferAccessSizeAlignment
 * (64 bytes) and pre-masking bounds checks across:
 *   - a null UBO descriptor (VK_NULL_HANDLE),
 *   - every byte tail from 1 through 64 (and 65..128),
 *   - a nonzero binding offset with every byte tail from 1 through 64,
 *   - non-64-byte-aligned bounds up to the maximum 64 KiB UBO range,
 *   - dynamic indices both in bounds and >= 4096 within the declared array.
 *
 * usage: brl-vulkan-ubo-element-bounds [--shim]
 *   --shim   no-op DRM shim: compile, bind, record and submit without
 *            checking GPU readback
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "brl_vulkan_ubo_element_bounds_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define MAX_UBO_RANGE 65536u
#define POISON_WORD   0xdeadbeefu

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

/* Indices stay within the shader's declared a[8194] array (including i + 1)
 * while probing past the 64 KiB UBO descriptor range (indices >= 4096) so the
 * pre-masking bounds check is exercised under valid Vulkan shader semantics. */
static const uint32_t test_indices[] = {
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
   15, 16, 17, 63, 64,
   4091, 4092, 4093, 4094, 4095,
   4096, 4097, 4099, 8191, 8192,
};

#define NUM_INDICES (sizeof(test_indices) / sizeof(test_indices[0]))

static void
check_vec4(const char *label, bool is_null, VkDeviceSize offset,
           VkDeviceSize range, VkDeviceSize robust_align,
           const uint32_t *ubo_words, uint32_t idx, const uint32_t got[4])
{
   if (is_null) {
      for (unsigned c = 0; c < 4; c++) {
         EXPECT(got[c] == 0,
                "%s idx=%u comp=%u: got 0x%08x, expected 0 (null descriptor)",
                label, idx, c, got[c]);
      }
      return;
   }

   const uint64_t rounded_bytes =
      ((uint64_t)range + robust_align - 1u) / robust_align * robust_align;
   const uint64_t rounded_elems = rounded_bytes / 16u;

   if ((uint64_t)idx >= rounded_elems) {
      for (unsigned c = 0; c < 4; c++) {
         EXPECT(got[c] == 0,
                "%s (off=%llu range=%llu) idx=%u comp=%u: got 0x%08x, expected 0",
                label, (unsigned long long)offset, (unsigned long long)range,
                idx, c, got[c]);
      }
      return;
   }

   const uint32_t base_word = (uint32_t)(offset / 4u) + idx * 4u;
   for (unsigned c = 0; c < 4; c++) {
      const uint64_t word_end = (uint64_t)idx * 16u + (uint64_t)(c + 1u) * 4u;
      const uint32_t expected = ubo_words[base_word + c];
      if (word_end <= (uint64_t)range) {
         EXPECT(got[c] == expected,
                "%s (off=%llu range=%llu) idx=%u comp=%u: got 0x%08x, expected 0x%08x",
                label, (unsigned long long)offset, (unsigned long long)range,
                idx, c, got[c], expected);
      } else {
         EXPECT(got[c] == expected || got[c] == 0,
                "%s (off=%llu range=%llu) idx=%u comp=%u: tail got 0x%08x, expected 0x%08x or 0",
                label, (unsigned long long)offset, (unsigned long long)range,
                idx, c, got[c], expected);
      }
   }
}

static void
run_case(const char *label, VkBuffer ubo_buf, VkDeviceSize offset,
         VkDeviceSize range, VkDeviceSize robust_align,
         const uint32_t *ubo_words, uint32_t *out_words, bool shim)
{
   const VkDescriptorBufferInfo ubo_info = {
      .buffer = ubo_buf,
      .offset = offset,
      .range = range,
   };
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = desc_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &ubo_info,
   };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);

   if (!shim) {
      for (size_t i = 0; i < NUM_INDICES * 2 * 4; i++)
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
   vkCmdDispatch(cmd, (uint32_t)NUM_INDICES, 1, 1);
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

   const bool is_null = ubo_buf == VK_NULL_HANDLE;
   for (size_t i = 0; i < NUM_INDICES; i++) {
      const uint32_t idx0 = test_indices[i];
      const uint32_t idx1 = idx0 + 1u;
      check_vec4(label, is_null, offset, range, robust_align, ubo_words, idx0,
                 &out_words[(i * 2u) * 4u]);
      check_vec4(label, is_null, offset, range, robust_align, ubo_words, idx1,
                 &out_words[(i * 2u + 1u) * 4u]);
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

   const VkDeviceSize ubo_offset_align =
      props2.properties.limits.minUniformBufferOffsetAlignment;
   const VkDeviceSize robust_align =
      rob2_props.robustUniformBufferAccessSizeAlignment;
   if (props2.properties.limits.maxUniformBufferRange < MAX_UBO_RANGE ||
       robust_align == 0 || (robust_align & (robust_align - 1u)) != 0) {
      fprintf(stderr, "unexpected UBO limits (max=%u robust_align=%llu)\n",
              props2.properties.limits.maxUniformBufferRange,
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
      ubo_offset_align > 256u ? ubo_offset_align : 256u;
   const VkDeviceSize ubo_buf_size = nonzero_offset + MAX_UBO_RANGE + 4096u;
   VkDeviceMemory ubo_mem, idx_mem, out_mem;
   void *ubo_map, *idx_map, *out_map;
   VkBuffer ubo_buf = create_host_buffer(
      ubo_buf_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo_mem, &ubo_map);
   VkBuffer idx_buf = create_host_buffer(
      (1u + NUM_INDICES) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      &idx_mem, &idx_map);
   VkBuffer out_buf = create_host_buffer(
      NUM_INDICES * 2u * 4u * sizeof(uint32_t),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_mem, &out_map);

   uint32_t *ubo_words = (uint32_t *)ubo_map;
   for (uint32_t i = 0; i < (uint32_t)(ubo_buf_size / 4u); i++)
      ubo_words[i] = 0xa5000000u | (i + 1u);

   uint32_t *idx_words = (uint32_t *)idx_map;
   idx_words[0] = (uint32_t)NUM_INDICES;
   memcpy(&idx_words[1], test_indices, sizeof(test_indices));

   const VkDescriptorSetLayoutBinding bindings[3] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
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

   const VkDescriptorPoolSize pool_sizes[2] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
   };
   CHECK(vkCreateDescriptorPool(
      dev,
      &(VkDescriptorPoolCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 2,
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

   const VkDescriptorBufferInfo idx_info = {idx_buf, 0, VK_WHOLE_SIZE};
   const VkDescriptorBufferInfo out_info = {out_buf, 0, VK_WHOLE_SIZE};
   const VkWriteDescriptorSet init_writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = desc_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &idx_info,
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
         .codeSize = sizeof(ubo_element_bounds_comp),
         .pCode = ubo_element_bounds_comp,
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
   run_case("null", VK_NULL_HANDLE, 0, VK_WHOLE_SIZE, robust_align, ubo_words,
            out_words, shim);

   if (shim) {
      /* Under the no-op shim, submit representative descriptor shapes without
       * repeating all 198 launches through the same pipeline. */
      run_case("tail-0", ubo_buf, 0, 1, robust_align, ubo_words, out_words,
               shim);
      run_case("tail-64", ubo_buf, 0, 65, robust_align, ubo_words, out_words,
               shim);
      run_case("nonzero-offset", ubo_buf, nonzero_offset, 17, robust_align,
               ubo_words, out_words, shim);
      run_case("large", ubo_buf, 0, 65473, robust_align, ubo_words, out_words,
               shim);
      run_case("large", ubo_buf, 0, MAX_UBO_RANGE, robust_align, ubo_words,
               out_words, shim);
      run_case("large-nonzero-offset", ubo_buf, nonzero_offset, MAX_UBO_RANGE,
               robust_align, ubo_words, out_words, shim);
   } else {
      /* 2. Every byte tail 1..64 at offset 0. */
      for (VkDeviceSize range = 1; range <= 64; range++) {
         run_case("tail-0", ubo_buf, 0, range, robust_align, ubo_words,
                  out_words, shim);
      }

      /* 3. Every byte tail 1..64 above one 64-byte record (65..128) at offset 0. */
      for (VkDeviceSize range = 65; range <= 128; range++) {
         run_case("tail-64", ubo_buf, 0, range, robust_align, ubo_words,
                  out_words, shim);
      }

      /* 4. Every byte tail 1..64 at a nonzero binding offset. */
      for (VkDeviceSize range = 1; range <= 64; range++) {
         run_case("nonzero-offset", ubo_buf, nonzero_offset, range,
                  robust_align, ubo_words, out_words, shim);
      }

      /* 5. Near-64-KiB non-64-byte-aligned bounds and full 64 KiB range. */
      const VkDeviceSize large_ranges[] = {65472, 65473, 65535, MAX_UBO_RANGE};
      for (size_t i = 0; i < sizeof(large_ranges) / sizeof(large_ranges[0]);
           i++) {
         run_case("large", ubo_buf, 0, large_ranges[i], robust_align, ubo_words,
                  out_words, shim);
      }
      run_case("large-nonzero-offset", ubo_buf, nonzero_offset, MAX_UBO_RANGE,
               robust_align, ubo_words, out_words, shim);
   }

   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, pipe_layout, NULL);
   vkDestroyDescriptorPool(dev, desc_pool, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroyBuffer(dev, out_buf, NULL);
   vkFreeMemory(dev, out_mem, NULL);
   vkDestroyBuffer(dev, idx_buf, NULL);
   vkFreeMemory(dev, idx_mem, NULL);
   vkDestroyBuffer(dev, ubo_buf, NULL);
   vkFreeMemory(dev, ubo_mem, NULL);
   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, cmd_pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   printf("%u launches, %u checks, %u failures%s\n", launches, checks, failures,
          shim ? " (shim)" : "");
   return failures ? 1 : 0;
}
