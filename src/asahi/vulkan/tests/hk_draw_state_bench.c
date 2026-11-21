/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host-only draw-recording benchmark. It records draws into one command
 * buffer and times the recording, per scenario, in nanoseconds per draw.
 * Nothing is submitted, so it runs under the DRM shim as well as on hardware
 * and never touches the GPU:
 *
 *   LD_PRELOAD=libasahi_noop_drm_shim.so ASAHI_SHIM_CHIP=g17p \
 *   VK_DRIVER_FILES=<icd.json> brl-draw-state-bench [draws] [rounds] [scenario]
 *                                                  [new-pass]
 *
 * The scenarios differ in what changes between two draws, which decides how
 * much of the per-draw render state a native AGX3 chip builds again:
 *
 *   plain            nothing
 *   base-instance    nothing, but the vertex shader reads gl_BaseInstance,
 *                    so every draw republishes its render state
 *   restart-enable   primitive restart enable: dynamic state the bind tail
 *                    does not read
 *   blend-constants  the blend constants, which live in the root
 *   push-constants   a push constant
 *   stencil-ref      the stencil reference, which the bind tail holds
 *   viewport         the viewport
 *   indexed          nothing, vkCmdDrawIndexed
 *
 * Each round records `draws` draws; the result is the fastest and the median
 * round. Compare two drivers by alternating runs of this program.
 *
 * With `all` for the scenario, more state changes follow, each on a pipeline
 * whose every state set here is dynamic. They are not timed rows: together
 * with ASAHI_MESA_DEBUG=perf, which prints what became of the bind tail of
 * each recording, they are the list run-draw-state-reuse.py checks. With
 * `new-pass` after the scenario, every draw of the loop is in a rendering
 * instance of its own. Naming a scenario enables the device features those
 * states need; a run without one creates the device the timed rows were
 * measured on.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "hk_draw_state_bench_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d\n", #call, r);                                \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define SIZE 256

enum scenario {
   PLAIN,
   BASE_INSTANCE,
   RESTART_ENABLE,
   BLEND_CONSTANTS,
   PUSH_CONSTANTS,
   STENCIL_REF,
   VIEWPORT,
   INDEXED,
   TIMED_SCENARIOS,
   /* Only with `all` or by name, on the all-dynamic pipeline. */
   SAME_VIEWPORT = TIMED_SCENARIOS,
   STENCIL_REF_BACK,
   SCISSOR,
   CULL_MODE,
   FRONT_FACE,
   DEPTH_COMPARE,
   DEPTH_TEST,
   DEPTH_WRITE,
   DEPTH_BIAS,
   STENCIL_OP,
   STENCIL_WRITE_MASK,
   STENCIL_COMPARE_MASK,
   TOPOLOGY,
   STENCIL_TEST,
   DEPTH_BIAS_ENABLE,
   LINE_WIDTH,
   POLYGON_MODE,
   DEPTH_CLAMP,
   DEPTH_CLIP,
   BLEND_EQUATION,
   SCENARIOS,
};

static const char *names[SCENARIOS] = {
   "plain",          "base-instance",  "restart-enable", "blend-constants",
   "push-constants", "stencil-ref",    "viewport",       "indexed",
   "same-viewport",  "stencil-ref-back", "scissor",      "cull-mode",
   "front-face",     "depth-compare",  "depth-test",     "depth-write",
   "depth-bias",     "stencil-op",     "stencil-write-mask",
   "stencil-compare-mask", "topology", "stencil-test", "depth-bias-enable",
   "line-width",     "polygon-mode",   "depth-clamp",    "depth-clip",
   "blend-equation",
};

static VkDevice dev;
static VkPhysicalDevice phys;

static uint64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + ts.tv_nsec;
}

static int
compare_u64(const void *a, const void *b)
{
   const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
   return x < y ? -1 : x > y;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(phys, &props);
   for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "no memory type\n");
   exit(2);
}

static VkImageView
attachment(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {SIZE, SIZE, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .usage = usage,
   };
   VkImage image;
   VkMemoryRequirements reqs;
   VkDeviceMemory memory;
   VkImageView view;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   vkGetImageMemoryRequirements(dev, image, &reqs);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type(reqs.memoryTypeBits, 0),
   };
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindImageMemory(dev, image, memory, 0));
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   CHECK(vkCreateImageView(dev, &view_info, NULL, &view));
   return view;
}

static VkShaderModule
module(const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &info, NULL, &m));
   return m;
}

static VkPipeline
pipeline(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs,
         bool all_dynamic)
{
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
   };
   const VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkStencilOpState stencil = {
      .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff, .writeMask = 0xff,
   };
   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
      .stencilTestEnable = VK_TRUE,
      .front = stencil,
      .back = stencil,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = 0xf,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
      VK_DYNAMIC_STATE_STENCIL_REFERENCE,
      VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
      /* The all-dynamic pipeline has these as well. */
      VK_DYNAMIC_STATE_CULL_MODE,
      VK_DYNAMIC_STATE_FRONT_FACE,
      VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
      VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
      VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
      VK_DYNAMIC_STATE_DEPTH_BIAS,
      VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
      VK_DYNAMIC_STATE_STENCIL_OP,
      VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
      VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
      VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
      VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
      VK_DYNAMIC_STATE_LINE_WIDTH,
      VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
      VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
      VK_DYNAMIC_STATE_DEPTH_CLIP_ENABLE_EXT,
      VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
   };
   const VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount =
         all_dynamic ? sizeof(dynamic_states) / sizeof(dynamic_states[0]) : 5,
      .pDynamicStates = dynamic_states,
   };
   const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &color_format,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .stencilAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &assembly,
      .pViewportState = &viewport,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic,
      .layout = layout,
   };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &info, NULL, &p));
   return p;
}

int
main(int argc, char **argv)
{
   const unsigned draws = argc > 1 ? strtoul(argv[1], NULL, 0) : 4096;
   const unsigned rounds = argc > 2 ? strtoul(argv[2], NULL, 0) : 51;
   const char *only = argc > 3 ? argv[3] : NULL;
   const bool new_pass = argc > 4 && !strcmp(argv[4], "new-pass");
   if (!draws || !rounds || rounds > 4096)
      return 2;

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &phys);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !count) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(phys, &props);

   /* What the all-dynamic pipeline needs, asked for with a scenario only. */
   VkPhysicalDeviceDepthClipEnableFeaturesEXT clip_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
      .depthClipEnable = VK_TRUE,
   };
   VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic3 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
      .pNext = &clip_features,
      .extendedDynamicState3DepthClampEnable = VK_TRUE,
      .extendedDynamicState3PolygonMode = VK_TRUE,
      .extendedDynamicState3ColorBlendEquation = VK_TRUE,
      .extendedDynamicState3DepthClipEnable = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures features = {
      .depthClamp = VK_TRUE,
      .fillModeNonSolid = VK_TRUE,
      .wideLines = VK_TRUE,
   };
   const char *const extensions[] = {
      VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
      VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = only ? &dynamic3 : NULL,
      .dynamicRendering = VK_TRUE,
   };
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = only ? 2 : 0,
      .ppEnabledExtensionNames = extensions,
      .pEnabledFeatures = only ? &features : NULL,
   };
   CHECK(vkCreateDevice(phys, &device_info, NULL, &dev));
#define EXT(name) const PFN_##name name = (PFN_##name)vkGetDeviceProcAddr(dev, #name)
   EXT(vkCmdSetPolygonModeEXT);
   EXT(vkCmdSetDepthClampEnableEXT);
   EXT(vkCmdSetDepthClipEnableEXT);
   EXT(vkCmdSetColorBlendEquationEXT);
#undef EXT

   const VkImageView color = attachment(
      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT);
   const VkImageView depth = attachment(
      VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

   /* Strip indices 0..3 repeated: the index buffer is only read by the GPU. */
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 4096,
      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
   };
   VkBuffer index_buffer;
   VkMemoryRequirements reqs;
   VkDeviceMemory index_memory;
   CHECK(vkCreateBuffer(dev, &buffer_info, NULL, &index_buffer));
   vkGetBufferMemoryRequirements(dev, index_buffer, &reqs);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = memory_type(reqs.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
   };
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &index_memory));
   CHECK(vkBindBufferMemory(dev, index_buffer, index_memory, 0));
   uint16_t *indices;
   CHECK(vkMapMemory(dev, index_memory, 0, VK_WHOLE_SIZE, 0, (void **)&indices));
   for (unsigned i = 0; i < 2048; ++i)
      indices[i] = i & 3;

   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .size = 16,
   };
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_range,
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &layout_info, NULL, &layout));
   const VkShaderModule fs = module(bench_frag, sizeof(bench_frag));
   const VkPipeline plain = pipeline(
      layout, module(bench_vert, sizeof(bench_vert)), fs, false);
   const VkPipeline base_instance = pipeline(
      layout, module(bench_base_vert, sizeof(bench_base_vert)), fs, false);
   const VkPipeline all_dynamic =
      only ? pipeline(layout, module(bench_vert, sizeof(bench_vert)), fs, true)
           : VK_NULL_HANDLE;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));
   const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cmd_info, &cmd));

   const VkRenderingAttachmentInfo color_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingAttachmentInfo depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {SIZE, SIZE}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_att,
      .pDepthAttachment = &depth_att,
      .pStencilAttachment = &depth_att,
   };
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   const VkRect2D scissor = {{8, 8}, {SIZE - 16, SIZE - 16}};
   /* The pipeline's equation, and one that differs in its operation. */
   const VkColorBlendEquationEXT blend_add = {
      .srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
   };
   VkColorBlendEquationEXT blend_subtract = blend_add;
   blend_subtract.colorBlendOp = VK_BLEND_OP_SUBTRACT;

   uint64_t *times = malloc(rounds * sizeof(*times));
   if (!times)
      return 2;

   printf("driver %s, %u draws per round, %u rounds\n", props.deviceName,
          draws, rounds);
   const bool all = only && !strcmp(only, "all");
   for (unsigned s = 0; s < SCENARIOS; ++s) {
      if (only ? !all && strcmp(only, names[s]) : s >= TIMED_SCENARIOS)
         continue;

      for (unsigned round = 0; round < rounds; ++round) {
         CHECK(vkResetCommandBuffer(cmd, 0));
         CHECK(vkBeginCommandBuffer(cmd, &begin));
         vkCmdBeginRendering(cmd, &rendering);
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           s == BASE_INSTANCE     ? base_instance
                           : s >= TIMED_SCENARIOS ? all_dynamic
                                                  : plain);
         const VkViewport viewport = {0, 0, SIZE - 1, SIZE, 0.0f, 1.0f};
         const float constants[4] = {0.5f, 0.5f, 0.5f, 0.5f};
         const float offset[4] = {0};
         vkCmdSetViewport(cmd, 0, 1, &viewport);
         vkCmdSetScissor(cmd, 0, 1, &scissor);
         vkCmdSetBlendConstants(cmd, constants);
         vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
         vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
         if (s >= TIMED_SCENARIOS) {
            /* What the plain pipeline has, but where a scenario alternates
             * between two values, the one of its odd draws: every draw of
             * the loop below then changes the state. */
            vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){{9, 8}, {SIZE - 16, SIZE - 16}});
            vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
            vkCmdSetFrontFace(cmd, VK_FRONT_FACE_CLOCKWISE);
            vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS);
            vkCmdSetDepthTestEnable(cmd, VK_TRUE);
            vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
            vkCmdSetDepthBiasEnable(cmd,
                                    s == DEPTH_BIAS || s == DEPTH_BIAS_ENABLE);
            vkCmdSetDepthBias(cmd, 1.0f, 0.0f, 1.0f);
            vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                              VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INVERT,
                              VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
            vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
            vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
            vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            vkCmdSetStencilTestEnable(cmd, VK_TRUE);
            vkCmdSetLineWidth(cmd, 1.0f);
            vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
            vkCmdSetDepthClampEnableEXT(cmd, VK_TRUE);
            vkCmdSetDepthClipEnableEXT(cmd, VK_TRUE);
            vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blend_add);
         }
         vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16,
                            offset);
         vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT16);
         /* The first draw of a recording pays for all of its state. */
         vkCmdDraw(cmd, 4, 1, 0, 0);

         const uint64_t start = now_ns();
         for (unsigned d = 0; d < draws; ++d) {
            if (new_pass) {
               vkCmdEndRendering(cmd);
               vkCmdBeginRendering(cmd, &rendering);
            }

            switch (s) {
            case RESTART_ENABLE:
               vkCmdSetPrimitiveRestartEnable(cmd, d & 1);
               break;
            case BLEND_CONSTANTS: {
               const float c[4] = {(d & 255) / 255.0f, 0.5f, 0.5f, 0.5f};
               vkCmdSetBlendConstants(cmd, c);
               break;
            }
            case PUSH_CONSTANTS: {
               const float o[4] = {(d & 63) / 256.0f, (d >> 6) / 256.0f, 0, 0};
               vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                                  16, o);
               break;
            }
            case STENCIL_REF:
               vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                        d & 255);
               break;
            case VIEWPORT: {
               const VkViewport v = {d & 63, 0, SIZE - 64, SIZE, 0.0f, 1.0f};
               vkCmdSetViewport(cmd, 0, 1, &v);
               break;
            }
            case SAME_VIEWPORT:
               vkCmdSetViewport(cmd, 0, 1, &viewport);
               break;
            case STENCIL_REF_BACK:
               /* Away and back: the same value, set twice. */
               vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 2);
               vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
               break;
            case SCISSOR:
               vkCmdSetScissor(cmd, 0, 1,
                               &(VkRect2D){{8 + (d & 1), 8}, {SIZE - 16, SIZE - 16}});
               break;
            case CULL_MODE:
               vkCmdSetCullMode(cmd, (d & 1) ? VK_CULL_MODE_BACK_BIT
                                             : VK_CULL_MODE_NONE);
               break;
            case FRONT_FACE:
               vkCmdSetFrontFace(cmd, (d & 1) ? VK_FRONT_FACE_CLOCKWISE
                                              : VK_FRONT_FACE_COUNTER_CLOCKWISE);
               break;
            case DEPTH_COMPARE:
               vkCmdSetDepthCompareOp(cmd, (d & 1) ? VK_COMPARE_OP_LESS
                                                   : VK_COMPARE_OP_LESS_OR_EQUAL);
               break;
            case DEPTH_TEST:
               vkCmdSetDepthTestEnable(cmd, d & 1);
               break;
            case DEPTH_WRITE:
               vkCmdSetDepthWriteEnable(cmd, d & 1);
               break;
            case DEPTH_BIAS:
               vkCmdSetDepthBias(cmd, (d & 1) ? 1.0f : 2.0f, 0.0f, 1.0f);
               break;
            case STENCIL_OP:
               vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                 VK_STENCIL_OP_KEEP,
                                 (d & 1) ? VK_STENCIL_OP_INVERT
                                         : VK_STENCIL_OP_REPLACE,
                                 VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
               break;
            case STENCIL_WRITE_MASK:
               vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                        1 + (d & 1));
               break;
            case STENCIL_COMPARE_MASK:
               vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                          1 + (d & 1));
               break;
            case TOPOLOGY:
               vkCmdSetPrimitiveTopology(
                  cmd, (d & 1) ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                               : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
               break;
            case STENCIL_TEST:
               vkCmdSetStencilTestEnable(cmd, d & 1);
               break;
            case DEPTH_BIAS_ENABLE:
               vkCmdSetDepthBiasEnable(cmd, d & 1);
               break;
            case LINE_WIDTH:
               vkCmdSetLineWidth(cmd, (d & 1) ? 1.0f : 2.0f);
               break;
            case POLYGON_MODE:
               vkCmdSetPolygonModeEXT(cmd, (d & 1) ? VK_POLYGON_MODE_FILL
                                                   : VK_POLYGON_MODE_LINE);
               break;
            case DEPTH_CLAMP:
               vkCmdSetDepthClampEnableEXT(cmd, d & 1);
               break;
            case DEPTH_CLIP:
               vkCmdSetDepthClipEnableEXT(cmd, d & 1);
               break;
            case BLEND_EQUATION:
               vkCmdSetColorBlendEquationEXT(
                  cmd, 0, 1, (d & 1) ? &blend_add : &blend_subtract);
               break;
            default:
               break;
            }

            if (s == INDEXED)
               vkCmdDrawIndexed(cmd, 4, 1, (d & 255) * 4, 0, 0);
            else if (s == BASE_INSTANCE)
               vkCmdDraw(cmd, 4, 1, 0, d & 15);
            else
               vkCmdDraw(cmd, 4, 1, 0, 0);
         }
         times[round] = now_ns() - start;

         vkCmdEndRendering(cmd);
         CHECK(vkEndCommandBuffer(cmd));
      }

      qsort(times, rounds, sizeof(*times), compare_u64);
      printf("%-16s min %8.1f  median %8.1f  ns/draw\n", names[s],
             (double)times[0] / draws, (double)times[rounds / 2] / draws);
   }

   return 0;
}
