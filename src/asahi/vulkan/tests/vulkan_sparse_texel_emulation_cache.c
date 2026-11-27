/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A program that creates its compute pipelines before its first sparse
 * resident buffer, as games that build pipelines on worker threads during
 * loading do. Until a buffer has a residency map, every texel buffer reads a
 * resident texel, so the shader that reads texel buffers is compiled without
 * the texel-buffer residency check and gets a checking twin built at the
 * first dispatch that binds it. The driver must remember that the program
 * needed the check, so that its next run compiles the shader with it from the
 * start and dispatches without building twins.
 *
 * The test starts from an empty cache directory and runs the same sequence
 * from two Vulkan instances in turn: create the pipelines, then a sparse
 * resident buffer of two blocks with only the first bound, then record a
 * dispatch that stores a value to every texel of the buffer through a storage
 * texel buffer and one that reads each texel back through a storage and a
 * uniform texel buffer, counting the shaders compiled while recording
 * (vulkan_compile_count.h). The stores to the unbound block land in the page
 * that backs unbound pages, so a read that skips the residency check returns
 * them. Where the device reports strict residency, the unbound block must
 * read zero and the bound one the value. The first recording must build a
 * twin, and the second must compile nothing and read the same.
 *
 * Usage: vulkan-sparse-texel-emulation-cache   Exit status 0 = pass, 77 = skip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "vulkan_compile_count.h"
#include "vulkan_sparse_texel_emulation_cache_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define VALUE 0x5a5a5a5au

static VkPhysicalDevice pdev;
static VkDevice dev;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkPipeline compute_pipeline(VkPipelineLayout pl, const uint32_t *code,
                                   size_t size) {
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size,
      .pCode = code}, NULL, &m));
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL},
      .layout = pl}, NULL, &pipe));
  vkDestroyShaderModule(dev, m, NULL);
  return pipe;
}

struct result {
  bool supported, strict;
  unsigned compiles, texels;
  double record_ms;
  uint64_t hash;
  /* Texels that do not read what they must: bound block, unbound block. */
  unsigned bad_bound, bad_unbound, leaked_unbound;
};

static void run(struct result *res) {
  memset(res, 0, sizeof(*res));
  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
          .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
          .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
  uint32_t n = 1;
  vkEnumeratePhysicalDevices(inst, &n, &pdev);

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(pdev, &features);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pdev, &props);
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, NULL);
  VkQueueFamilyProperties qfs[8];
  qf_count = qf_count > 8 ? 8 : qf_count;
  vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qfs);
  uint32_t family = UINT32_MAX;
  const VkQueueFlags want = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
  for (uint32_t i = 0; i < qf_count && family == UINT32_MAX; i++)
    if ((qfs[i].queueFlags & want) == want)
      family = i;
  if (!features.sparseBinding || !features.sparseResidencyBuffer ||
      family == UINT32_MAX) {
    vkDestroyInstance(inst, NULL);
    return;
  }
  res->supported = true;
  res->strict = props.sparseProperties.residencyNonResidentStrict;

  CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &(VkPhysicalDeviceVulkan13Features){
          .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
          .synchronization2 = VK_TRUE},
      .pEnabledFeatures = &(VkPhysicalDeviceFeatures){
          .sparseBinding = VK_TRUE, .sparseResidencyBuffer = VK_TRUE},
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = family, .queueCount = 1,
          .pQueuePriorities = (float[]){1}}}, NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, family, 0, &q);

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = (VkDescriptorSetLayoutBinding[]){
          {0, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1,
           VK_SHADER_STAGE_COMPUTE_BIT, NULL},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
           VK_SHADER_STAGE_COMPUTE_BIT, NULL},
          {2, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1,
           VK_SHADER_STAGE_COMPUTE_BIT, NULL}}}, NULL, &dsl));
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
          VK_SHADER_STAGE_COMPUTE_BIT, 0, 4}}, NULL, &pl));
  VkPipeline writer = compute_pipeline(pl, sparse_texel_write_comp,
                                       sizeof(sparse_texel_write_comp));
  VkPipeline reader = compute_pipeline(pl, sparse_texel_read_comp,
                                       sizeof(sparse_texel_read_comp));

  /* Only now the program's first sparse resident buffer: two blocks, the
   * second left unbound. */
  const VkBufferCreateInfo sparse_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
               VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
      .size = 1, .usage = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                          VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT};
  VkBuffer probe;
  CHECK(vkCreateBuffer(dev, &sparse_info, NULL, &probe));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, probe, &mr);
  vkDestroyBuffer(dev, probe, NULL);
  const VkDeviceSize block = mr.alignment;
  VkBufferCreateInfo info = sparse_info;
  info.size = 2 * block;
  VkBuffer sparse;
  CHECK(vkCreateBuffer(dev, &info, NULL, &sparse));
  vkGetBufferMemoryRequirements(dev, sparse, &mr);
  VkDeviceMemory block_mem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = block,
      .memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      NULL, &block_mem));
  VkFence fence;
  CHECK(vkCreateFence(dev, &(VkFenceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &fence));
  CHECK(vkQueueBindSparse(q, 1, &(VkBindSparseInfo){
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .bufferBindCount = 1,
      .pBufferBinds = &(VkSparseBufferMemoryBindInfo){
          .buffer = sparse, .bindCount = 1,
          .pBinds = &(VkSparseMemoryBind){
              .resourceOffset = 0, .size = block, .memory = block_mem}}},
      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  CHECK(vkResetFences(dev, 1, &fence));

  const uint32_t texels = (uint32_t)(info.size / 4);
  res->texels = texels;
  VkBufferView view;
  CHECK(vkCreateBufferView(dev, &(VkBufferViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = sparse,
      .format = VK_FORMAT_R32_UINT, .range = VK_WHOLE_SIZE}, NULL, &view));

  VkBuffer out;
  CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 2 * (VkDeviceSize)texels * 4,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, NULL, &out));
  vkGetBufferMemoryRequirements(dev, out, &mr);
  VkDeviceMemory out_mem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
      NULL, &out_mem));
  CHECK(vkBindBufferMemory(dev, out, out_mem, 0));

  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 3,
      .pPoolSizes = (VkDescriptorPoolSize[]){
          {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1},
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
          {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1}}}, NULL, &pool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl},
      &set));
  vkUpdateDescriptorSets(dev, 3, (VkWriteDescriptorSet[]){
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
       .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
       .pTexelBufferView = &view},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
       .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &(VkDescriptorBufferInfo){out, 0, VK_WHOLE_SIZE}},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
       .dstBinding = 2, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
       .pTexelBufferView = &view}}, 0, NULL);

  VkCommandPool cp;
  CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family}, NULL, &cp));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));

  struct compile_count count;
  struct timespec t0, t1;
  compile_count_begin(&count);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0,
                          NULL);
  const uint32_t value = VALUE;
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &value);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, writer);
  vkCmdDispatch(cb, texels / 64, 1, 1);
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
      .pMemoryBarriers = &(VkMemoryBarrier2){
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT}});
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, reader);
  vkCmdDispatch(cb, texels / 64, 1, 1);
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
      .pMemoryBarriers = &(VkMemoryBarrier2){
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
          .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
          .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}});
  CHECK(vkEndCommandBuffer(cb));
  clock_gettime(CLOCK_MONOTONIC, &t1);
  res->compiles = compile_count_end(&count);
  res->record_ms =
      (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

  CHECK(vkQueueSubmit(q, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
      .pCommandBuffers = &cb}, fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  const uint32_t *px;
  CHECK(vkMapMemory(dev, out_mem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < 2 * texels; i++) {
    h = (h ^ px[i]) * 1099511628211ull;
    const bool bound = (i % texels) < texels / 2;
    if (bound && px[i] != VALUE)
      res->bad_bound++;
    if (!bound && px[i] != 0)
      res->bad_unbound++;
    if (!bound && px[i] == VALUE)
      res->leaked_unbound++;
  }
  res->hash = h;
  vkUnmapMemory(dev, out_mem);

  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, cp, NULL);
  vkDestroyDescriptorPool(dev, pool, NULL);
  vkDestroyBufferView(dev, view, NULL);
  vkDestroyBuffer(dev, out, NULL);
  vkFreeMemory(dev, out_mem, NULL);
  vkDestroyBuffer(dev, sparse, NULL);
  vkFreeMemory(dev, block_mem, NULL);
  vkDestroyPipeline(dev, writer, NULL);
  vkDestroyPipeline(dev, reader, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  /* Destroying the device and instance flushes the disk cache writes. */
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
}

int main(void) {
  char dir[] = "/tmp/hk-sparse-texel-emulation-cache-XXXXXX";
  if (!mkdtemp(dir)) {
    perror("mkdtemp");
    return 2;
  }
  setenv("MESA_SHADER_CACHE_DIR", dir, 1);
  unsetenv("MESA_SHADER_CACHE_DISABLE");
  compile_count_enable();

  struct result first, second;
  run(&first);
  if (first.supported)
    run(&second);

  char cmd[96];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
  if (system(cmd) != 0)
    fprintf(stderr, "could not remove %s\n", dir);

  if (!first.supported) {
    printf("SKIP: no sparse resident buffers on a compute queue\n");
    return 77;
  }
  printf("%u texels, %s residency; recording with the checking twin built: "
         "%u compiles, %.2f ms; checking from the start: %u compiles, "
         "%.2f ms\n",
         first.texels, first.strict ? "strict" : "non-strict", first.compiles,
         first.record_ms, second.compiles, second.record_ms);

  const struct result *runs[2] = {&first, &second};
  for (unsigned r = 0; r < 2; r++) {
    if (runs[r]->bad_bound) {
      fprintf(stderr, "FAIL: run %u: %u reads of the bound block are not "
              "the stored value\n", r + 1, runs[r]->bad_bound);
      return 1;
    }
    if (first.strict && runs[r]->bad_unbound) {
      fprintf(stderr, "FAIL: run %u: %u reads of the unbound block are not "
              "zero (%u return the value stored to it)\n", r + 1,
              runs[r]->bad_unbound, runs[r]->leaked_unbound);
      return 1;
    }
  }
  if (first.hash != second.hash) {
    fprintf(stderr, "FAIL: the second run reads differently\n");
    return 1;
  }
  if (first.compiles == 0) {
    fprintf(stderr, "FAIL: the first recording compiled nothing, so the "
            "dispatch no longer builds a twin and this test checks nothing\n");
    return 1;
  }
  if (second.compiles != 0) {
    fprintf(stderr, "FAIL: the second recording compiled %u shaders: it "
            "built a checking twin again\n", second.compiles);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
