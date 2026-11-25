/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Losslessly compressed depth arrays (M4 Pro): every unit that reads or
 * writes one layer must address that layer's metadata. Two 4-layer images
 * per format (D32_SFLOAT, D16_UNORM, D32_SFLOAT_S8_UINT) with the usage of a
 * shadow-map cascade go through
 *
 *  - buffer uploads (all layers in one region, and one region per layer),
 *  - vkCmdClearDepthStencilImage of a layer range (metadata fast clear),
 *  - a scissored draw into one layer through a 2D view (ZLS load + store),
 *  - a pass over all layers through a 2D-array view: load-op clear of a
 *    partial render area, plus a scissored draw into layer 0,
 *  - vkCmdCopyImage of single layers and of a layer range at offsets,
 *  - a 1:1 vkCmdBlitImage of one layer at offsets,
 *
 * and every layer is read back (texture reads) and compared with a CPU model
 * of the same operations, bit for bit.
 *
 * usage: g17p-vulkan-layered-depth [--shim] [--expect-compressed]
 *   --shim               no-op DRM shim: record and submit, skip readback
 *   --expect-compressed  require the driver's "compression enabled" message
 *                        for every image (needs ASAHI_MESA_DEBUG=perf)
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "g17p_vulkan_layered_depth_frag.h"
#include "g17p_vulkan_layered_depth_vert.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W      129
#define H      127
#define LAYERS 4
#define TEXELS (W * H)

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;

static FILE *driver_log;
static int saved_stderr = -1;

static void
restore_stderr(void)
{
   if (saved_stderr < 0)
      return;
   fflush(stderr);
   dup2(saved_stderr, STDERR_FILENO);
   close(saved_stderr);
   saved_stderr = -1;
}

static void
replay_driver_log(void)
{
   restore_stderr();
   if (!driver_log)
      return;
   rewind(driver_log);
   char line[1024];
   while (fgets(line, sizeof(line), driver_log))
      fputs(line, stderr);
   fclose(driver_log);
   driver_log = NULL;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "no memory type for %#x\n", flags);
   exit(2);
}

static VkBuffer
buffer_create(VkDeviceSize size, VkBufferUsageFlags usage, void **map)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &info, NULL, &buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buffer, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindBufferMemory(dev, buffer, memory, 0));
   CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, map));
   return buffer;
}

static VkCommandBuffer
begin(void)
{
   const VkCommandBufferAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &info, &cmd));
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &begin_info));
   return cmd;
}

static void
barrier(VkCommandBuffer cmd)
{
   const VkMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL,
                        0, NULL);
}

static void
submit(VkCommandBuffer cmd)
{
   CHECK(vkEndCommandBuffer(cmd));
   const VkSubmitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

struct fmt {
   VkFormat format;
   const char *name, *pipe_name;
   unsigned depth_B; /* 4 or 2 */
   bool stencil;
};

/* CPU model of one image: depth bits and stencil per layer and texel. */
struct model {
   uint32_t z[LAYERS][TEXELS];
   uint8_t s[LAYERS][TEXELS];
};

struct img {
   VkImage image;
   struct model m;
};

static VkImageAspectFlags
aspects(const struct fmt *f)
{
   return VK_IMAGE_ASPECT_DEPTH_BIT |
          (f->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

/* What the depth attachment stores for a float depth. */
static uint32_t
depth_bits(const struct fmt *f, float z)
{
   if (f->depth_B == 2)
      return (uint32_t)lrintf(z * 65535.0f);
   uint32_t u;
   memcpy(&u, &z, 4);
   return u;
}

/* Distinct per image, layer and texel. D32 patterns are exact floats in
 * [0, 1] (n < 2^24), so no conversion can alter them. */
static uint32_t
pattern_z(const struct fmt *f, unsigned img, unsigned layer, unsigned t)
{
   const uint32_t n = (layer * TEXELS + t + 1) * 3 + img;
   if (f->depth_B == 2)
      return (n * 7 + img * 13) & 0xffff;
   return depth_bits(f, (float)n / 16777216.0f);
}

static uint8_t
pattern_s(unsigned img, unsigned layer, unsigned t)
{
   return (uint8_t)((layer * TEXELS + t) * 5 + img * 3 + 1);
}

static VkImage
image_create(const struct fmt *f)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = f->format,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = LAYERS,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      /* Kyty's guest depth target usage. */
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindImageMemory(dev, image, memory, 0));
   return image;
}

static void
to_general(VkCommandBuffer cmd, const struct fmt *f, VkImage image)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {aspects(f), 0, 1, 0, LAYERS},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                        1, &b);
}

/* Upload the patterns: one region for all layers (A) or one per layer (B). */
static void
upload(const struct fmt *f, struct img *im, unsigned which)
{
   const VkDeviceSize z_size = (VkDeviceSize)f->depth_B * TEXELS * LAYERS;
   const VkDeviceSize s_size = f->stencil ? (VkDeviceSize)TEXELS * LAYERS : 0;
   uint8_t *map;
   VkBuffer staging = buffer_create(z_size + s_size,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    (void **)&map);
   for (unsigned l = 0; l < LAYERS; ++l) {
      for (unsigned t = 0; t < TEXELS; ++t) {
         im->m.z[l][t] = pattern_z(f, which, l, t);
         im->m.s[l][t] = f->stencil ? pattern_s(which, l, t) : 0;
         memcpy(map + (VkDeviceSize)f->depth_B * (l * TEXELS + t),
                &im->m.z[l][t], f->depth_B);
         if (f->stencil)
            map[z_size + l * TEXELS + t] = im->m.s[l][t];
      }
   }

   VkBufferImageCopy regions[2 * LAYERS];
   unsigned n = 0;
   for (unsigned a = 0; a < (f->stencil ? 2u : 1u); ++a) {
      const VkImageAspectFlags aspect =
         a ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
      const VkDeviceSize base = a ? z_size : 0;
      const VkDeviceSize layer_B = (a ? 1 : f->depth_B) * TEXELS;
      if (which == 0) {
         regions[n++] = (VkBufferImageCopy){
            .bufferOffset = base,
            .imageSubresource = {aspect, 0, 0, LAYERS},
            .imageExtent = {W, H, 1},
         };
      } else {
         for (unsigned l = 0; l < LAYERS; ++l) {
            regions[n++] = (VkBufferImageCopy){
               .bufferOffset = base + l * layer_B,
               .imageSubresource = {aspect, 0, l, 1},
               .imageExtent = {W, H, 1},
            };
         }
      }
   }

   VkCommandBuffer cmd = begin();
   to_general(cmd, f, im->image);
   vkCmdCopyBufferToImage(cmd, staging, im->image, VK_IMAGE_LAYOUT_GENERAL, n,
                          regions);
   barrier(cmd);
   submit(cmd);
   vkDestroyBuffer(dev, staging, NULL);
}

static unsigned failures;

static void
readback_check(const struct fmt *f, const struct img *im, const char *what,
               bool readback)
{
   const VkDeviceSize z_size = (VkDeviceSize)f->depth_B * TEXELS * LAYERS;
   const VkDeviceSize s_size = f->stencil ? (VkDeviceSize)TEXELS * LAYERS : 0;
   uint8_t *map;
   VkBuffer out = buffer_create(z_size + s_size,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                (void **)&map);
   memset(map, 0xcd, z_size + s_size);

   const VkBufferImageCopy regions[2] = {
      {
         .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, LAYERS},
         .imageExtent = {W, H, 1},
      },
      {
         .bufferOffset = z_size,
         .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, LAYERS},
         .imageExtent = {W, H, 1},
      },
   };
   VkCommandBuffer cmd = begin();
   vkCmdCopyImageToBuffer(cmd, im->image, VK_IMAGE_LAYOUT_GENERAL, out,
                          f->stencil ? 2 : 1, regions);
   barrier(cmd);
   submit(cmd);

   if (readback) {
      for (unsigned l = 0; l < LAYERS; ++l) {
         unsigned bad_z = 0, bad_s = 0;
         for (unsigned t = 0; t < TEXELS; ++t) {
            uint32_t z = 0;
            memcpy(&z, map + (VkDeviceSize)f->depth_B * (l * TEXELS + t),
                   f->depth_B);
            if (z != im->m.z[l][t]) {
               if (!bad_z)
                  fprintf(stderr,
                          "%s %s layer %u: depth (%u,%u) = %08x, want %08x\n",
                          f->name, what, l, t % W, t / W, z, im->m.z[l][t]);
               bad_z++;
            }
            if (f->stencil && map[z_size + l * TEXELS + t] != im->m.s[l][t]) {
               if (!bad_s)
                  fprintf(stderr,
                          "%s %s layer %u: stencil (%u,%u) = %02x, want %02x\n",
                          f->name, what, l, t % W, t / W,
                          map[z_size + l * TEXELS + t], im->m.s[l][t]);
               bad_s++;
            }
         }
         if (bad_z || bad_s) {
            fprintf(stderr, "  %u depth, %u stencil of %u texels wrong\n",
                    bad_z, bad_s, TEXELS);
            failures++;
         }
      }
   }
   vkDestroyBuffer(dev, out, NULL);
}

struct rect {
   int x, y;
   unsigned w, h;
};

static bool
inside(const struct rect *r, unsigned t)
{
   const int x = t % W, y = t / W;
   return x >= r->x && x < r->x + (int)r->w && y >= r->y &&
          y < r->y + (int)r->h;
}

static void
model_fill(const struct fmt *f, struct model *m, unsigned layer,
           const struct rect *r, uint32_t z, uint8_t s)
{
   for (unsigned t = 0; t < TEXELS; ++t) {
      if (r && !inside(r, t))
         continue;
      m->z[layer][t] = z;
      if (f->stencil)
         m->s[layer][t] = s;
   }
}

/* Copy a rectangle of src layers to dst layers in the model. */
static void
model_copy(struct model *dst, const struct model *src, unsigned src_layer,
           unsigned dst_layer, unsigned layers, struct rect s, int dx, int dy)
{
   for (unsigned l = 0; l < layers; ++l) {
      for (unsigned y = 0; y < s.h; ++y) {
         for (unsigned x = 0; x < s.w; ++x) {
            const unsigned st = (s.y + y) * W + s.x + x;
            const unsigned dt = (dy + y) * W + dx + x;
            dst->z[dst_layer + l][dt] = src->z[src_layer + l][st];
            dst->s[dst_layer + l][dt] = src->s[src_layer + l][st];
         }
      }
   }
}

static VkImageView
view_create(const struct fmt *f, VkImage image, VkImageViewType type,
            unsigned base, unsigned count)
{
   const VkImageViewCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = type,
      .format = f->format,
      .subresourceRange = {aspects(f), 0, 1, base, count},
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &info, NULL, &view));
   return view;
}

static VkShaderModule
shader(const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule module;
   CHECK(vkCreateShaderModule(dev, &info, NULL, &module));
   return module;
}

static VkPipelineLayout pipeline_layout;
static VkShaderModule vs, fs;

/* Depth (and stencil) always written; viewport and scissor dynamic. */
static VkPipeline
pipeline_create(const struct fmt *f)
{
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vs,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fs,
         .pName = "main",
      },
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkStencilOpState replace = {
      .failOp = VK_STENCIL_OP_REPLACE,
      .passOp = VK_STENCIL_OP_REPLACE,
      .depthFailOp = VK_STENCIL_OP_REPLACE,
      .compareOp = VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff,
      .writeMask = 0xff,
   };
   const VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = f->stencil,
      .front = replace,
      .back = replace,
   };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
   };
   const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_SCISSOR,
                                     VK_DYNAMIC_STATE_STENCIL_REFERENCE};
   const VkPipelineDynamicStateCreateInfo dy = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 3,
      .pDynamicStates = dynamic,
   };
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .depthAttachmentFormat = f->format,
      .stencilAttachmentFormat = f->stencil ? f->format : VK_FORMAT_UNDEFINED,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &ds,
      .pColorBlendState = &cb,
      .pDynamicState = &dy,
      .layout = pipeline_layout,
   };
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &info, NULL,
                                   &pipeline));
   return pipeline;
}

/* One dynamic-rendering pass on `view` (layer_count layers). With clear set
 * the attachment is cleared to (cz, cs) in `area`, else loaded; a draw at
 * (dz, ds) fills `scissor` of the pass's first layer. */
static void
pass(VkCommandBuffer cmd, const struct fmt *f, VkPipeline pipeline,
     VkImageView view, unsigned layer_count, struct rect area, bool clear,
     float cz, uint8_t cs, struct rect scissor, float dz, uint8_t dsv)
{
   const VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = {cz, cs},
   };
   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{area.x, area.y}, {area.w, area.h}},
      .layerCount = layer_count,
      .pDepthAttachment = &att,
      .pStencilAttachment = f->stencil ? &att : NULL,
   };
   vkCmdBeginRendering(cmd, &info);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkViewport viewport = {0, 0, W, H, 0.0f, 1.0f};
   vkCmdSetViewport(cmd, 0, 1, &viewport);
   const VkRect2D sc = {{scissor.x, scissor.y}, {scissor.w, scissor.h}};
   vkCmdSetScissor(cmd, 0, 1, &sc);
   vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, dsv);
   vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4,
                      &dz);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRendering(cmd);
}

static void
run_format(const struct fmt *f, bool readback)
{
   struct img *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
   a->image = image_create(f);
   b->image = image_create(f);

   upload(f, a, 0);
   upload(f, b, 1);
   readback_check(f, a, "upload (one region)", readback);
   readback_check(f, b, "upload (per layer)", readback);

   /* Clear layers 1-2 of A: the metadata fast clear of a layer range. */
   {
      VkCommandBuffer cmd = begin();
      const VkClearDepthStencilValue value = {0.75f, 0x5a};
      const VkImageSubresourceRange range = {aspects(f), 0, 1, 1, 2};
      vkCmdClearDepthStencilImage(cmd, a->image, VK_IMAGE_LAYOUT_GENERAL,
                                  &value, 1, &range);
      barrier(cmd);
      submit(cmd);
      for (unsigned l = 1; l <= 2; ++l)
         model_fill(f, &a->m, l, NULL, depth_bits(f, 0.75f), 0x5a);
      readback_check(f, a, "layer-range clear", readback);
   }

   VkPipeline pipeline = pipeline_create(f);
   const struct rect full = {0, 0, W, H};

   /* Layer 2 of A through a 2D view: ZLS load and store of one layer. */
   {
      VkImageView view = view_create(f, a->image, VK_IMAGE_VIEW_TYPE_2D, 2, 1);
      const struct rect scissor = {8, 20, 64, 70};
      VkCommandBuffer cmd = begin();
      pass(cmd, f, pipeline, view, 1, full, false, 0, 0, scissor, 1.0f, 0x3c);
      barrier(cmd);
      submit(cmd);
      model_fill(f, &a->m, 2, &scissor, depth_bits(f, 1.0f), 0x3c);
      readback_check(f, a, "2D-view draw", readback);
      vkDestroyImageView(dev, view, NULL);
   }

   /* All layers of B through a 2D-array view: a load-op clear of a partial
    * render area on every layer (the layered ZLS metadata stride), and a draw
    * into layer 0. */
   {
      VkImageView view =
         view_create(f, b->image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, LAYERS);
      const struct rect area = {32, 16, 40, 50};
      const struct rect scissor = {40, 30, 20, 20};
      VkCommandBuffer cmd = begin();
      pass(cmd, f, pipeline, view, LAYERS, area, true, 0.25f, 0xa7, scissor,
           0.0f, 0x11);
      barrier(cmd);
      submit(cmd);
      for (unsigned l = 0; l < LAYERS; ++l)
         model_fill(f, &b->m, l, &area, depth_bits(f, 0.25f), 0xa7);
      model_fill(f, &b->m, 0, &scissor, depth_bits(f, 0.0f), 0x11);
      readback_check(f, b, "2D-array-view clear and draw", readback);
      vkDestroyImageView(dev, view, NULL);
   }

   /* Copies: A layer 3 -> B layer 0 whole; A layers 0-1 -> B layers 2-3 at
    * an offset. */
   {
      const VkImageCopy regions[2] = {
         {
            .srcSubresource = {aspects(f), 0, 3, 1},
            .dstSubresource = {aspects(f), 0, 0, 1},
            .extent = {W, H, 1},
         },
         {
            .srcSubresource = {aspects(f), 0, 0, 2},
            .srcOffset = {5, 7, 0},
            .dstSubresource = {aspects(f), 0, 2, 2},
            .dstOffset = {17, 3, 0},
            .extent = {60, 50, 1},
         },
      };
      VkCommandBuffer cmd = begin();
      vkCmdCopyImage(cmd, a->image, VK_IMAGE_LAYOUT_GENERAL, b->image,
                     VK_IMAGE_LAYOUT_GENERAL, 2, regions);
      barrier(cmd);
      submit(cmd);
      model_copy(&b->m, &a->m, 3, 0, 1, full, 0, 0);
      model_copy(&b->m, &a->m, 0, 2, 2, (struct rect){5, 7, 60, 50}, 17, 3);
      readback_check(f, b, "image copy", readback);
   }

   /* A 1:1 blit of A layer 1 into B layer 1 at an offset. */
   {
      const VkImageBlit region = {
         .srcSubresource = {aspects(f), 0, 1, 1},
         .srcOffsets = {{2, 3, 0}, {66, 67, 1}},
         .dstSubresource = {aspects(f), 0, 1, 1},
         .dstOffsets = {{10, 11, 0}, {74, 75, 1}},
      };
      VkCommandBuffer cmd = begin();
      vkCmdBlitImage(cmd, a->image, VK_IMAGE_LAYOUT_GENERAL, b->image,
                     VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
      barrier(cmd);
      submit(cmd);
      model_copy(&b->m, &a->m, 1, 1, 1, (struct rect){2, 3, 64, 64}, 10, 11);
      readback_check(f, b, "blit", readback);
   }

   readback_check(f, a, "final", readback);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyImage(dev, a->image, NULL);
   vkDestroyImage(dev, b->image, NULL);
   free(a);
   free(b);
}

int
main(int argc, char **argv)
{
   bool shim = false, expect_compressed = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else if (!strcmp(argv[i], "--expect-compressed"))
         expect_compressed = true;
   }

   fflush(stderr);
   saved_stderr = dup(STDERR_FILENO);
   driver_log = tmpfile();
   if (saved_stderr < 0 || !driver_log ||
       atexit(replay_driver_log) != 0 ||
       dup2(fileno(driver_log), STDERR_FILENO) < 0) {
      replay_driver_log();
      fprintf(stderr, "could not capture driver log\n");
      return 2;
   }

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
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));

   const VkPushConstantRange push = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push,
   };
   CHECK(vkCreatePipelineLayout(dev, &layout_info, NULL, &pipeline_layout));
   vs = shader(g17p_vulkan_layered_depth_vert_spv,
               sizeof(g17p_vulkan_layered_depth_vert_spv));
   fs = shader(g17p_vulkan_layered_depth_frag_spv,
               sizeof(g17p_vulkan_layered_depth_frag_spv));

   const struct fmt formats[] = {
      {VK_FORMAT_D32_SFLOAT, "D32_SFLOAT", "z32_float", 4, false},
      {VK_FORMAT_D16_UNORM, "D16_UNORM", "z16_unorm", 2, false},
      {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", "z32_float", 4,
       true},
   };
   for (unsigned i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i)
      run_format(&formats[i], !shim);

   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyPipelineLayout(dev, pipeline_layout, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   restore_stderr();
   rewind(driver_log);
   unsigned compressed[3] = {0};
   char line[1024], want[96];
   while (fgets(line, sizeof(line), driver_log)) {
      fputs(line, stderr);
      for (unsigned i = 0; i < 3; ++i) {
         snprintf(want, sizeof(want), "compression enabled: %s %ux%u",
                  i == 0 ? "z32_float" : i == 1 ? "z16_unorm" : "s8_uint", W,
                  H);
         compressed[i] += strstr(line, want) != NULL;
      }
   }
   fclose(driver_log);
   driver_log = NULL;

   /* Two images per format: D32 and the D32S8 depth plane print z32_float
    * (4), D16 prints z16_unorm (2), the D32S8 stencil plane s8_uint (2). */
   if (expect_compressed &&
       (compressed[0] < 4 || compressed[1] < 2 || compressed[2] < 2)) {
      fprintf(stderr, "FAIL: depth arrays not compressed (%u z32, %u z16, "
                      "%u s8 messages)\n",
              compressed[0], compressed[1], compressed[2]);
      failures++;
   }
   printf("layered depth%s: %s\n", shim ? " (no readback)" : "",
          failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
