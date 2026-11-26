/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * borealis-sync: synchronization and visibility cases with exact readback
 * oracles.  Every case waits for its work to complete and then compares every
 * pixel it wrote, and every pixel it was required to preserve, with the value
 * the API guarantees.  A completed timeline with the wrong contents fails.
 *
 * Cases:
 *
 *   layer-disjoint-queues (strict)
 *      Two queues of one family render to different array layers of one
 *      image at the same time, each as 128 partial render areas that cover
 *      its layer in half-cell strips.  A store operation may touch the whole
 *      attachment, so each queue orders its own render pass instances with
 *      a barrier on its layer.  Different subresources are distinct memory
 *      locations, so no dependency is needed between the queues.
 *
 *   feedback-loop-outside-area (strict)
 *      One queue uploads a per-iteration pattern.  A graphics queue waits
 *      for it and renders into the left half of every 32x32 cell in the
 *      attachment feedback-loop layout, sampling the same image's right
 *      half, which lies outside the render area and is not written.  A
 *      barrier orders each render pass instance after the previous one.
 *      The result must hold the inverted right half on the left and the
 *      untouched upload on the right.
 *
 *   same-image-disjoint-tiles (compat)
 *      Two queues render texel-disjoint halves of every cell of one image
 *      with no dependency between them; each queue orders its own render
 *      pass instances with barriers.  Vulkan does not promise that
 *      unsynchronized writes from different queues to different texels of
 *      one optimally tiled subresource are free of hazards, so this is
 *      compatibility coverage: applications do it and expect both halves.
 *
 * Submissions from different queues are made from separate host threads
 * released together.  That makes the submissions concurrent; it does not by
 * itself prove that the GPU ran them at the same time.
 *
 * Usage: borealis-sync [--iterations N] [--seed N] [--shim] <case>
 *        borealis-sync --list
 *
 * --shim records, submits and waits as usual but does not check contents,
 * for the no-op DRM shim.  Exit status: 0 pass, 1 wrong contents, a timeout
 * or a lost device, 2 an API or setup error, 77 unsupported.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "borealis_sync_shaders.h"
#include "vulkan_render_area_shaders.h"

/* A lost device is a test failure; any other error is a setup error. */
#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s:%d: %s: %d\n", __FILE__, __LINE__, #call, r_);    \
         if (r_ == VK_ERROR_DEVICE_LOST) {                                     \
            printf("FAIL: device lost\n");                                     \
            exit(1);                                                           \
         }                                                                     \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W         256
#define H         256
#define CELL      32
#define HALF      16
#define LAYERS    2
#define WAIT_NS   (30ull * 1000 * 1000 * 1000)
#define EXIT_SKIP 77

/* Colours use only 0 and 255 per channel, which UNORM8 stores exactly. */
#define RED   0xff0000ffu
#define GREEN 0xff00ff00u
#define BLUE  0xffff0000u
#define WHITE 0xffffffffu

/* Written to the readback buffer before each iteration.  No case expects
 * it: every expected value is opaque.
 */
#define POISON 0x5a5a5a5au

#define MAX_ITERATIONS 1000000

struct ctx {
   const char *name;
   bool shim;
   unsigned iterations;
   uint32_t seed;

   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice dev;
   uint32_t family[2];
   VkQueue queue[2];
   VkCommandPool pool[2];
   /* Queue 0 supports graphics stages. */
   bool queue0_graphics;
};

static void
skip(const char *why)
{
   printf("SKIP: %s\n", why);
   exit(EXIT_SKIP);
}

static uint32_t
memory_type(struct ctx *c, uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(c->pdev, &props);

   for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }

   fprintf(stderr, "no memory type for %#x\n", flags);
   exit(2);
}

static VkDeviceMemory
allocate(struct ctx *c, VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(
      c->dev,
      &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = memory_type(c, req.memoryTypeBits, flags),
      },
      NULL, &memory));
   return memory;
}

struct host_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint32_t *map;
};

static struct host_buffer
host_buffer(struct ctx *c, VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct host_buffer b;
   CHECK(vkCreateBuffer(c->dev,
                        &(VkBufferCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = size,
                           .usage = usage,
                        },
                        NULL, &b.buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(c->dev, b.buffer, &req);
   b.memory = allocate(c, req,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(c->dev, b.buffer, b.memory, 0));
   CHECK(vkMapMemory(c->dev, b.memory, 0, VK_WHOLE_SIZE, 0, (void **)&b.map));
   return b;
}

/* Fills a readback buffer of `images` W x H images with POISON, so a copy
 * that never happened cannot pass.  The previous copy into it has completed.
 */
static void
poison(struct host_buffer *b, unsigned images)
{
   for (unsigned i = 0; i < images * W * H; ++i)
      b->map[i] = POISON;
}

static void
destroy_host_buffer(struct ctx *c, struct host_buffer *b)
{
   vkDestroyBuffer(c->dev, b->buffer, NULL);
   vkFreeMemory(c->dev, b->memory, NULL);
}

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view[LAYERS];
   unsigned layers;
};

static struct image
create_image(struct ctx *c, VkFormat format, unsigned layers,
             VkImageUsageFlags usage)
{
   struct image img = {.layers = layers};
   const bool concurrent = c->family[0] != c->family[1];
   CHECK(vkCreateImage(c->dev,
                       &(VkImageCreateInfo){
                          .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .imageType = VK_IMAGE_TYPE_2D,
                          .format = format,
                          .extent = {W, H, 1},
                          .mipLevels = 1,
                          .arrayLayers = layers,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .usage = usage,
                          .sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT
                                                    : VK_SHARING_MODE_EXCLUSIVE,
                          .queueFamilyIndexCount = concurrent ? 2 : 0,
                          .pQueueFamilyIndices = c->family,
                       },
                       NULL, &img.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(c->dev, img.image, &req);
   img.memory = allocate(c, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindImageMemory(c->dev, img.image, img.memory, 0));

   for (unsigned i = 0; i < layers; ++i) {
      CHECK(vkCreateImageView(
         c->dev,
         &(VkImageViewCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = img.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1},
         },
         NULL, &img.view[i]));
   }
   return img;
}

static void
destroy_image(struct ctx *c, struct image *img)
{
   for (unsigned i = 0; i < img->layers; ++i)
      vkDestroyImageView(c->dev, img->view[i], NULL);
   vkDestroyImage(c->dev, img->image, NULL);
   vkFreeMemory(c->dev, img->memory, NULL);
}

static VkSemaphore
timeline(struct ctx *c)
{
   VkSemaphore semaphore;
   CHECK(vkCreateSemaphore(
      c->dev,
      &(VkSemaphoreCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
         .pNext =
            &(VkSemaphoreTypeCreateInfo){
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
               .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            },
      },
      NULL, &semaphore));
   return semaphore;
}

static VkCommandBuffer
begin_commands(struct ctx *c, unsigned queue)
{
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(
      c->dev,
      &(VkCommandBufferAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = c->pool[queue],
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      },
      &cmd));
   CHECK(vkBeginCommandBuffer(
      cmd, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
           }));
   return cmd;
}

/* A graphics-only stage on a queue that may not support graphics. */
static VkPipelineStageFlags2
queue0_stage(struct ctx *c, VkPipelineStageFlags2 graphics_stage)
{
   return c->queue0_graphics ? graphics_stage
                             : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
}

static void
image_barrier(VkCommandBuffer cmd, VkImage image, unsigned layers,
              VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
              VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
              VkImageLayout old_layout, VkImageLayout new_layout)
{
   vkCmdPipelineBarrier2(
      cmd,
      &(VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers =
            &(VkImageMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = src_stage,
               .srcAccessMask = src_access,
               .dstStageMask = dst_stage,
               .dstAccessMask = dst_access,
               .oldLayout = old_layout,
               .newLayout = new_layout,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .image = image,
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers},
            },
      });
}

/* Orders a render pass instance on one layer after the previous one on the
 * same queue.  The layout does not change.
 */
static void
between_passes(VkCommandBuffer cmd, VkImage image, unsigned layer,
               VkImageLayout layout, VkPipelineStageFlags2 dst_stage,
               VkAccessFlags2 dst_access)
{
   vkCmdPipelineBarrier2(
      cmd,
      &(VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers =
            &(VkImageMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
               .dstStageMask =
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | dst_stage,
               .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                dst_access,
               .oldLayout = layout,
               .newLayout = layout,
               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
               .image = image,
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, layer, 1},
            },
      });
}

/* Copies every layer of the image to the buffer and makes it host-visible. */
static void
record_readback(VkCommandBuffer cmd, VkImage image, unsigned layers,
                VkBuffer buffer)
{
   VkBufferImageCopy regions[LAYERS];
   for (unsigned i = 0; i < layers; ++i) {
      regions[i] = (VkBufferImageCopy){
         .bufferOffset = (VkDeviceSize)i * W * H * sizeof(uint32_t),
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1},
         .imageExtent = {W, H, 1},
      };
   }
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          buffer, layers, regions);
   vkCmdPipelineBarrier2(
      cmd, &(VkDependencyInfo){
              .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
              .memoryBarrierCount = 1,
              .pMemoryBarriers =
                 &(VkMemoryBarrier2){
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                    .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                 },
           });
}

struct submission {
   VkQueue queue;
   VkCommandBuffer cmd;
   unsigned wait_count;
   VkSemaphore wait[2];
   uint64_t wait_value[2];
   VkPipelineStageFlags2 wait_stage;
   VkSemaphore signal;
   uint64_t signal_value;

   /* For submissions made from a host thread. */
   pthread_barrier_t *start;
   VkResult result;
};

static VkResult
submit(const struct submission *s)
{
   VkSemaphoreSubmitInfo waits[2];
   for (unsigned i = 0; i < s->wait_count; ++i) {
      waits[i] = (VkSemaphoreSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
         .semaphore = s->wait[i],
         .value = s->wait_value[i],
         .stageMask = s->wait_stage,
      };
   }
   const VkSemaphoreSubmitInfo signal = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = s->signal,
      .value = s->signal_value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   const VkCommandBufferSubmitInfo command = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = s->cmd,
   };
   return vkQueueSubmit2(s->queue, 1,
                         &(VkSubmitInfo2){
                            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                            .waitSemaphoreInfoCount = s->wait_count,
                            .pWaitSemaphoreInfos = waits,
                            .commandBufferInfoCount = 1,
                            .pCommandBufferInfos = &command,
                            .signalSemaphoreInfoCount = 1,
                            .pSignalSemaphoreInfos = &signal,
                         },
                         VK_NULL_HANDLE);
}

static void *
submit_thread(void *data)
{
   struct submission *s = data;
   pthread_barrier_wait(s->start);
   s->result = submit(s);
   return NULL;
}

/* Submits two command buffers to two queues from two threads at once. */
static void
submit_concurrently(struct submission s[2])
{
   pthread_barrier_t start;
   pthread_t threads[2];
   pthread_barrier_init(&start, NULL, 3);
   for (unsigned i = 0; i < 2; ++i) {
      s[i].start = &start;
      s[i].result = VK_ERROR_UNKNOWN;
      if (pthread_create(&threads[i], NULL, submit_thread, &s[i])) {
         fprintf(stderr, "pthread_create failed\n");
         exit(2);
      }
   }
   pthread_barrier_wait(&start);
   for (unsigned i = 0; i < 2; ++i)
      pthread_join(threads[i], NULL);
   pthread_barrier_destroy(&start);
   for (unsigned i = 0; i < 2; ++i) {
      if (s[i].result != VK_SUCCESS) {
         fprintf(stderr, "vkQueueSubmit2(queue %u): %d\n", i, s[i].result);
         exit(2);
      }
   }
}

static void
wait_value(struct ctx *c, VkSemaphore semaphore, uint64_t value)
{
   VkResult r =
      vkWaitSemaphores(c->dev,
                       &(VkSemaphoreWaitInfo){
                          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                          .semaphoreCount = 1,
                          .pSemaphores = &semaphore,
                          .pValues = &value,
                       },
                       WAIT_NS);
   if (r != VK_SUCCESS) {
      printf("FAIL: %s: iteration %" PRIu64 ": wait returned %d%s\n", c->name,
             value, r, r == VK_TIMEOUT ? " (timed out)" : "");
      exit(1);
   }
}

static VkPipeline
create_pipeline(struct ctx *c, VkPipelineLayout layout, VkFormat format,
                const uint32_t *frag, size_t frag_size,
                VkPipelineCreateFlags flags)
{
   VkShaderModule modules[2];
   const uint32_t *code[2] = {render_area_vert, frag};
   const size_t size[2] = {sizeof(render_area_vert), frag_size};
   for (unsigned i = 0; i < 2; ++i) {
      CHECK(vkCreateShaderModule(
         c->dev,
         &(VkShaderModuleCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size[i],
            .pCode = code[i],
         },
         NULL, &modules[i]));
   }

   const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_SCISSOR};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(
      c->dev, VK_NULL_HANDLE, 1,
      &(VkGraphicsPipelineCreateInfo){
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext =
            &(VkPipelineRenderingCreateInfo){
               .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
               .colorAttachmentCount = 1,
               .pColorAttachmentFormats = &format,
            },
         .flags = flags,
         .stageCount = 2,
         .pStages =
            (VkPipelineShaderStageCreateInfo[]){
               {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = modules[0],
                .pName = "main"},
               {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = modules[1],
                .pName = "main"},
            },
         .pVertexInputState =
            &(VkPipelineVertexInputStateCreateInfo){
               .sType =
                  VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            },
         .pInputAssemblyState =
            &(VkPipelineInputAssemblyStateCreateInfo){
               .sType =
                  VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
               .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            },
         .pViewportState =
            &(VkPipelineViewportStateCreateInfo){
               .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
               .viewportCount = 1,
               .scissorCount = 1,
            },
         .pRasterizationState =
            &(VkPipelineRasterizationStateCreateInfo){
               .sType =
                  VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
               .lineWidth = 1,
            },
         .pMultisampleState =
            &(VkPipelineMultisampleStateCreateInfo){
               .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            },
         .pColorBlendState =
            &(VkPipelineColorBlendStateCreateInfo){
               .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
               .attachmentCount = 1,
               .pAttachments =
                  &(VkPipelineColorBlendAttachmentState){
                     .colorWriteMask = 0xf,
                  },
            },
         .pDynamicState =
            &(VkPipelineDynamicStateCreateInfo){
               .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
               .dynamicStateCount = 2,
               .pDynamicStates = dynamic,
            },
         .layout = layout,
      },
      NULL, &pipeline));

   for (unsigned i = 0; i < 2; ++i)
      vkDestroyShaderModule(c->dev, modules[i], NULL);
   return pipeline;
}

/* A pipeline writing the 32-bit colour in its fragment push constant. */
struct fill {
   VkPipelineLayout layout;
   VkPipeline pipeline;
};

static struct fill
create_fill(struct ctx *c, VkFormat format, VkPipelineCreateFlags flags)
{
   struct fill f;
   CHECK(vkCreatePipelineLayout(
      c->dev,
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges =
            &(VkPushConstantRange){VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4},
      },
      NULL, &f.layout));
   f.pipeline = create_pipeline(c, f.layout, format, render_area_frag,
                                sizeof(render_area_frag), flags);
   return f;
}

static void
destroy_fill(struct ctx *c, struct fill *f)
{
   vkDestroyPipeline(c->dev, f->pipeline, NULL);
   vkDestroyPipelineLayout(c->dev, f->layout, NULL);
}

/* Renders one solid rectangle as its own render pass instance. */
static void
fill_rect(VkCommandBuffer cmd, const struct fill *f, VkImageView view,
          VkImageLayout layout, VkRect2D area, uint32_t colour)
{
   vkCmdBeginRendering(
      cmd, &(VkRenderingInfo){
              .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
              .renderArea = area,
              .layerCount = 1,
              .colorAttachmentCount = 1,
              .pColorAttachments =
                 &(VkRenderingAttachmentInfo){
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .imageView = view,
                    .imageLayout = layout,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                 },
           });
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, f->pipeline);
   vkCmdSetViewport(cmd, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
   vkCmdSetScissor(cmd, 0, 1, &area);
   vkCmdPushConstants(cmd, f->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4,
                      &colour);
   vkCmdDraw(cmd, 3, 1, 0, 0);
   vkCmdEndRendering(cmd);
}

static VkRect2D
half_cell(unsigned x, unsigned y, unsigned half)
{
   return (VkRect2D){{x + half * HALF, y}, {HALF, CELL}};
}

static unsigned
check_pixels(struct ctx *c, unsigned iteration, unsigned layer,
             const uint32_t *pixels,
             uint32_t (*expected)(void *, unsigned, unsigned), void *data)
{
   unsigned bad = 0;
   for (unsigned y = 0; y < H; ++y) {
      for (unsigned x = 0; x < W; ++x) {
         const uint32_t want = expected(data, x, y);
         const uint32_t got = pixels[y * W + x];
         if (got != want && bad++ < 8) {
            fprintf(stderr,
                    "%s: iteration %u layer %u (%u, %u): %08x, expected "
                    "%08x\n",
                    c->name, iteration, layer, x, y, got, want);
         }
      }
   }
   if (bad) {
      fprintf(stderr, "%s: iteration %u layer %u: %u of %u pixels wrong\n",
              c->name, iteration, layer, bad, W * H);
   }
   return bad;
}

/* Colours of the left and right half of every cell. */
struct halves {
   uint32_t left, right;
};

static uint32_t
expected_halves(void *data, unsigned x, unsigned y)
{
   const struct halves *h = data;
   return (x % CELL) < HALF ? h->left : h->right;
}

/* Variant 1 swaps the colours of variant 0, so stale contents from the
 * previous iteration never match.
 */
static struct halves
layer_colours(unsigned layer, unsigned variant)
{
   const uint32_t a = layer ? BLUE : RED;
   const uint32_t b = layer ? WHITE : GREEN;
   return variant ? (struct halves){b, a} : (struct halves){a, b};
}

static int
run_layer_disjoint_queues(struct ctx *c)
{
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   struct image img = create_image(c, format, LAYERS,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   struct host_buffer out =
      host_buffer(c, (VkDeviceSize)LAYERS * W * H * sizeof(uint32_t),
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   struct fill fill = create_fill(c, format, 0);

   /* Queue q renders layer q, both halves of every cell, as separate render
    * pass instances with disjoint render areas, each ordered after the last.
    */
   const VkImageLayout attachment = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   VkCommandBuffer work[2][2];
   for (unsigned q = 0; q < 2; ++q) {
      for (unsigned v = 0; v < 2; ++v) {
         const struct halves colours = layer_colours(q, v);
         VkCommandBuffer cmd = begin_commands(c, q);
         for (unsigned y = 0; y < H; y += CELL) {
            for (unsigned x = 0; x < W; x += CELL) {
               for (unsigned half = 0; half < 2; ++half) {
                  if (x || y || half)
                     between_passes(cmd, img.image, q, attachment, 0, 0);
                  fill_rect(cmd, &fill, img.view[q], attachment,
                            half_cell(x, y, half),
                            half ? colours.right : colours.left);
               }
            }
         }
         CHECK(vkEndCommandBuffer(cmd));
         work[q][v] = cmd;
      }
   }

   VkCommandBuffer init = begin_commands(c, 0);
   image_barrier(init, img.image, LAYERS, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 0, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   vkCmdClearColorImage(
      init, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      &(VkClearColorValue){.uint32 = {0, 0, 0, 0}}, 1,
      &(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, LAYERS});
   image_barrier(init, img.image, LAYERS, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
   CHECK(vkEndCommandBuffer(init));

   VkCommandBuffer readback = begin_commands(c, 0);
   image_barrier(readback, img.image, LAYERS,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   record_readback(readback, img.image, LAYERS, out.buffer);
   CHECK(vkEndCommandBuffer(readback));

   VkSemaphore init_done = timeline(c);
   VkSemaphore done[2] = {timeline(c), timeline(c)};
   VkSemaphore read_done = timeline(c);

   unsigned failed = 0;
   for (uint64_t n = 1; n <= c->iterations && !failed; ++n) {
      const unsigned v = n & 1;
      poison(&out, LAYERS);
      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = init,
         .signal = init_done,
         .signal_value = n,
      }));

      struct submission s[2];
      for (unsigned q = 0; q < 2; ++q) {
         s[q] = (struct submission){
            .queue = c->queue[q],
            .cmd = work[q][v],
            .wait_count = 1,
            .wait = {init_done},
            .wait_value = {n},
            .wait_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .signal = done[q],
            .signal_value = n,
         };
      }
      submit_concurrently(s);

      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = readback,
         .wait_count = 2,
         .wait = {done[0], done[1]},
         .wait_value = {n, n},
         .wait_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .signal = read_done,
         .signal_value = n,
      }));
      wait_value(c, read_done, n);

      if (c->shim)
         continue;
      for (unsigned layer = 0; layer < LAYERS; ++layer) {
         struct halves colours = layer_colours(layer, v);
         failed += check_pixels(c, n, layer, out.map + layer * W * H,
                                expected_halves, &colours);
      }
   }

   CHECK(vkDeviceWaitIdle(c->dev));
   vkDestroySemaphore(c->dev, init_done, NULL);
   vkDestroySemaphore(c->dev, done[0], NULL);
   vkDestroySemaphore(c->dev, done[1], NULL);
   vkDestroySemaphore(c->dev, read_done, NULL);
   destroy_fill(c, &fill);
   destroy_host_buffer(c, &out);
   destroy_image(c, &img);
   return failed ? 1 : 0;
}

static uint32_t
pattern(uint32_t seed, uint32_t n, unsigned x, unsigned y)
{
   uint32_t h = seed ^ (n * 0x9e3779b9u) ^ (y * W + x) * 0x85ebca6bu;
   h ^= h >> 16;
   h *= 0x7feb352du;
   h ^= h >> 15;
   h *= 0x846ca68bu;
   h ^= h >> 16;
   return h | 0xff000000u;
}

struct feedback_expect {
   uint32_t seed, n;
};

/* The left half holds the inverted colour of the texel HALF to its right;
 * the right half holds the upload.
 */
static uint32_t
expected_feedback(void *data, unsigned x, unsigned y)
{
   const struct feedback_expect *e = data;
   if ((x % CELL) < HALF)
      return pattern(e->seed, e->n, x + HALF, y) ^ 0x00ffffffu;
   return pattern(e->seed, e->n, x, y);
}

static int
run_feedback_loop_outside_area(struct ctx *c)
{
   const VkFormat format = VK_FORMAT_R8G8B8A8_UINT;
   const VkImageLayout feedback =
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
   struct image img = create_image(
      c, format, 1,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
         VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT |
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   const VkDeviceSize size = W * H * sizeof(uint32_t);
   struct host_buffer upload =
      host_buffer(c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
   struct host_buffer out =
      host_buffer(c, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   VkSampler sampler;
   CHECK(
      vkCreateSampler(c->dev,
                      &(VkSamplerCreateInfo){
                         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                         .magFilter = VK_FILTER_NEAREST,
                         .minFilter = VK_FILTER_NEAREST,
                         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                         .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                      },
                      NULL, &sampler));
   VkDescriptorSetLayout set_layout;
   CHECK(vkCreateDescriptorSetLayout(
      c->dev,
      &(VkDescriptorSetLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1,
         .pBindings =
            &(VkDescriptorSetLayoutBinding){
               .binding = 0,
               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               .descriptorCount = 1,
               .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
      },
      NULL, &set_layout));
   VkDescriptorPool descriptor_pool;
   CHECK(vkCreateDescriptorPool(
      c->dev,
      &(VkDescriptorPoolCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 1,
         .pPoolSizes = &(
            VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      },
      NULL, &descriptor_pool));
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(
      c->dev,
      &(VkDescriptorSetAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = descriptor_pool,
         .descriptorSetCount = 1,
         .pSetLayouts = &set_layout,
      },
      &set));
   vkUpdateDescriptorSets(
      c->dev, 1,
      &(VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo =
            &(VkDescriptorImageInfo){
               .sampler = sampler,
               .imageView = img.view[0],
               .imageLayout = feedback,
            },
      },
      0, NULL);
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(
      c->dev,
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount = 1,
         .pSetLayouts = &set_layout,
         .pushConstantRangeCount = 1,
         .pPushConstantRanges =
            &(VkPushConstantRange){VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8},
      },
      NULL, &layout));
   VkPipeline pipeline = create_pipeline(
      c, layout, format, borealis_sync_feedback_frag,
      sizeof(borealis_sync_feedback_frag),
      VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);

   /* Queue 0 uploads the pattern and hands the image over in the
    * feedback-loop layout.  The UNDEFINED source layout discards the
    * previous iteration; the execution dependency orders the write after
    * that iteration's readback on the same queue.
    */
   VkCommandBuffer produce = begin_commands(c, 0);
   image_barrier(produce, img.image, 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   vkCmdCopyBufferToImage(
      produce, upload.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1,
      &(VkBufferImageCopy){
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {W, H, 1},
      });
   image_barrier(
      produce, img.image, 1, VK_PIPELINE_STAGE_2_COPY_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT,
      queue0_stage(c, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT),
      c->queue0_graphics ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                              VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                         : 0,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, feedback);
   CHECK(vkEndCommandBuffer(produce));

   /* Queue 1 renders the left half of every cell, reading the right half. */
   VkCommandBuffer consume = begin_commands(c, 1);
   const int32_t offset[2] = {HALF, 0};
   for (unsigned y = 0; y < H; y += CELL) {
      for (unsigned x = 0; x < W; x += CELL) {
         const VkRect2D area = half_cell(x, y, 0);
         if (x || y) {
            between_passes(consume, img.image, 0, feedback,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
         }
         vkCmdBeginRendering(
            consume,
            &(VkRenderingInfo){
               .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
               .renderArea = area,
               .layerCount = 1,
               .colorAttachmentCount = 1,
               .pColorAttachments =
                  &(VkRenderingAttachmentInfo){
                     .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                     .imageView = img.view[0],
                     .imageLayout = feedback,
                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                  },
            });
         vkCmdBindPipeline(consume, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         vkCmdBindDescriptorSets(consume, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 layout, 0, 1, &set, 0, NULL);
         vkCmdSetViewport(consume, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
         vkCmdSetScissor(consume, 0, 1, &area);
         vkCmdPushConstants(consume, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(offset), offset);
         vkCmdDraw(consume, 3, 1, 0, 0);
         vkCmdEndRendering(consume);
      }
   }
   CHECK(vkEndCommandBuffer(consume));

   VkCommandBuffer readback = begin_commands(c, 0);
   image_barrier(
      readback, img.image, 1,
      queue0_stage(c, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT),
      c->queue0_graphics ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : 0,
      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, feedback,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   record_readback(readback, img.image, 1, out.buffer);
   CHECK(vkEndCommandBuffer(readback));

   VkSemaphore produced = timeline(c);
   VkSemaphore consumed = timeline(c);
   VkSemaphore read_done = timeline(c);

   unsigned failed = 0;
   for (uint64_t n = 1; n <= c->iterations && !failed; ++n) {
      /* The previous iteration's copy has completed: the host waited for
       * its readback, which waited for the copy's queue.
       */
      for (unsigned y = 0; y < H; ++y) {
         for (unsigned x = 0; x < W; ++x)
            upload.map[y * W + x] = pattern(c->seed, n, x, y);
      }
      poison(&out, 1);

      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = produce,
         .signal = produced,
         .signal_value = n,
      }));
      CHECK(submit(&(struct submission){
         .queue = c->queue[1],
         .cmd = consume,
         .wait_count = 1,
         .wait = {produced},
         .wait_value = {n},
         .wait_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .signal = consumed,
         .signal_value = n,
      }));
      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = readback,
         .wait_count = 1,
         .wait = {consumed},
         .wait_value = {n},
         .wait_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .signal = read_done,
         .signal_value = n,
      }));
      wait_value(c, read_done, n);

      if (c->shim)
         continue;
      struct feedback_expect e = {c->seed, n};
      failed += check_pixels(c, n, 0, out.map, expected_feedback, &e);
   }

   CHECK(vkDeviceWaitIdle(c->dev));
   vkDestroySemaphore(c->dev, produced, NULL);
   vkDestroySemaphore(c->dev, consumed, NULL);
   vkDestroySemaphore(c->dev, read_done, NULL);
   vkDestroyPipeline(c->dev, pipeline, NULL);
   vkDestroyPipelineLayout(c->dev, layout, NULL);
   vkDestroyDescriptorPool(c->dev, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(c->dev, set_layout, NULL);
   vkDestroySampler(c->dev, sampler, NULL);
   destroy_host_buffer(c, &upload);
   destroy_host_buffer(c, &out);
   destroy_image(c, &img);
   return failed ? 1 : 0;
}

static int
run_same_image_disjoint_tiles(struct ctx *c)
{
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkImageLayout feedback =
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
   struct image img = create_image(
      c, format, 1,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
         VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT |
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
         VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   struct host_buffer out = host_buffer(c, W * H * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   struct fill fill = create_fill(
      c, format, VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);

   /* Queue q renders half q of every cell of the one image, ordering its
    * own render pass instances; nothing orders the two queues.
    */
   VkCommandBuffer work[2][2];
   for (unsigned q = 0; q < 2; ++q) {
      for (unsigned v = 0; v < 2; ++v) {
         const struct halves colours = layer_colours(0, v);
         VkCommandBuffer cmd = begin_commands(c, q);
         for (unsigned y = 0; y < H; y += CELL) {
            for (unsigned x = 0; x < W; x += CELL) {
               if (x || y)
                  between_passes(cmd, img.image, 0, feedback, 0, 0);
               fill_rect(cmd, &fill, img.view[0], feedback, half_cell(x, y, q),
                         q ? colours.right : colours.left);
            }
         }
         CHECK(vkEndCommandBuffer(cmd));
         work[q][v] = cmd;
      }
   }

   VkCommandBuffer init = begin_commands(c, 0);
   image_barrier(init, img.image, 1, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                 VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   vkCmdClearColorImage(
      init, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      &(VkClearColorValue){.uint32 = {0, 0, 0, 0}}, 1,
      &(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
   image_barrier(init, img.image, 1, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, feedback);
   CHECK(vkEndCommandBuffer(init));

   VkCommandBuffer readback = begin_commands(c, 0);
   image_barrier(readback, img.image, 1,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                 feedback, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   record_readback(readback, img.image, 1, out.buffer);
   CHECK(vkEndCommandBuffer(readback));

   VkSemaphore init_done = timeline(c);
   VkSemaphore done[2] = {timeline(c), timeline(c)};
   VkSemaphore read_done = timeline(c);

   unsigned failed = 0;
   for (uint64_t n = 1; n <= c->iterations && !failed; ++n) {
      const unsigned v = n & 1;
      poison(&out, 1);
      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = init,
         .signal = init_done,
         .signal_value = n,
      }));

      struct submission s[2];
      for (unsigned q = 0; q < 2; ++q) {
         s[q] = (struct submission){
            .queue = c->queue[q],
            .cmd = work[q][v],
            .wait_count = 1,
            .wait = {init_done},
            .wait_value = {n},
            .wait_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .signal = done[q],
            .signal_value = n,
         };
      }
      submit_concurrently(s);

      CHECK(submit(&(struct submission){
         .queue = c->queue[0],
         .cmd = readback,
         .wait_count = 2,
         .wait = {done[0], done[1]},
         .wait_value = {n, n},
         .wait_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .signal = read_done,
         .signal_value = n,
      }));
      wait_value(c, read_done, n);

      if (c->shim)
         continue;
      struct halves colours = layer_colours(0, v);
      failed += check_pixels(c, n, 0, out.map, expected_halves, &colours);
   }

   CHECK(vkDeviceWaitIdle(c->dev));
   vkDestroySemaphore(c->dev, init_done, NULL);
   vkDestroySemaphore(c->dev, done[0], NULL);
   vkDestroySemaphore(c->dev, done[1], NULL);
   vkDestroySemaphore(c->dev, read_done, NULL);
   destroy_fill(c, &fill);
   destroy_host_buffer(c, &out);
   destroy_image(c, &img);
   return failed ? 1 : 0;
}

enum queue_need {
   /* Two queues of one graphics family. */
   TWO_GRAPHICS_QUEUES,
   /* A graphics queue plus any second queue that can copy. */
   GRAPHICS_AND_COPY_QUEUE,
};

struct test_case {
   const char *name;
   const char *tier;
   enum queue_need queues;
   bool feedback_loop;
   int (*run)(struct ctx *c);
   unsigned default_iterations;
};

static const struct test_case cases[] = {
   {"layer-disjoint-queues", "strict", TWO_GRAPHICS_QUEUES, false,
    run_layer_disjoint_queues, 200},
   {"feedback-loop-outside-area", "strict", GRAPHICS_AND_COPY_QUEUE, true,
    run_feedback_loop_outside_area, 200},
   {"same-image-disjoint-tiles", "compat", TWO_GRAPHICS_QUEUES, true,
    run_same_image_disjoint_tiles, 200},
};

static bool
has_extension(VkPhysicalDevice pdev, const char *name)
{
   uint32_t count = 0;
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &count, NULL));
   VkExtensionProperties *props = calloc(count, sizeof(*props));
   CHECK(vkEnumerateDeviceExtensionProperties(pdev, NULL, &count, props));
   bool found = false;
   for (uint32_t i = 0; i < count && !found; ++i)
      found = !strcmp(props[i].extensionName, name);
   free(props);
   return found;
}

static void
create_device(struct ctx *c, const struct test_case *tc)
{
   CHECK(vkCreateInstance(
      &(VkInstanceCreateInfo){
         .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pApplicationInfo =
            &(VkApplicationInfo){
               .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
               .pApplicationName = "borealis-sync",
               .apiVersion = VK_API_VERSION_1_3,
            },
      },
      NULL, &c->instance));

   /* An enumeration error is a driver failure, not a missing device. */
   uint32_t count = 1;
   VkResult r = vkEnumeratePhysicalDevices(c->instance, &count, &c->pdev);
   if (r < 0) {
      fprintf(stderr, "vkEnumeratePhysicalDevices: %d\n", r);
      exit(2);
   }
   if (!count)
      skip("no Vulkan device");

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(c->pdev, &props);
   if (props.apiVersion < VK_API_VERSION_1_3)
      skip("Vulkan 1.3 is required");

   vkGetPhysicalDeviceQueueFamilyProperties(c->pdev, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(c->pdev, &count, families);
   uint32_t graphics = UINT32_MAX, copy = UINT32_MAX;
   for (uint32_t i = 0; i < count; ++i) {
      const VkQueueFlags f = families[i].queueFlags;
      if (graphics == UINT32_MAX && (f & VK_QUEUE_GRAPHICS_BIT))
         graphics = i;
      else if (copy == UINT32_MAX &&
               (f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                     VK_QUEUE_TRANSFER_BIT)))
         copy = i;
   }
   const uint32_t graphics_queues =
      graphics == UINT32_MAX ? 0 : families[graphics].queueCount;
   free(families);

   VkDeviceQueueCreateInfo queue_info[2];
   unsigned queue_info_count = 1;
   const float priorities[2] = {1.0f, 1.0f};
   uint32_t index[2] = {0, 0};
   if (graphics_queues >= 2) {
      c->family[0] = c->family[1] = graphics;
      index[1] = 1;
      queue_info[0] = (VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = graphics,
         .queueCount = 2,
         .pQueuePriorities = priorities,
      };
      c->queue0_graphics = true;
   } else if (tc->queues == GRAPHICS_AND_COPY_QUEUE && graphics_queues &&
              copy != UINT32_MAX) {
      /* Queue 0 copies, queue 1 renders. */
      c->family[0] = copy;
      c->family[1] = graphics;
      for (unsigned i = 0; i < 2; ++i) {
         queue_info[i] = (VkDeviceQueueCreateInfo){
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = c->family[i],
            .queueCount = 1,
            .pQueuePriorities = priorities,
         };
      }
      queue_info_count = 2;
      c->queue0_graphics = false;
   } else if (tc->queues == TWO_GRAPHICS_QUEUES) {
      skip("two queues in one graphics family are required");
   } else {
      skip("a graphics queue and a second queue are required");
   }

   VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
   };
   const char *extensions[1];
   unsigned extension_count = 0;
   if (tc->feedback_loop) {
      if (!has_extension(c->pdev,
                         VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME))
         skip("VK_EXT_attachment_feedback_loop_layout is required");
      features13.pNext = &feedback;
      extensions[extension_count++] =
         VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
   }
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features12,
   };
   vkGetPhysicalDeviceFeatures2(c->pdev, &features2);
   if (!features12.timelineSemaphore || !features13.dynamicRendering ||
       !features13.synchronization2)
      skip("timeline semaphores, dynamic rendering and synchronization2 are "
           "required");
   if (tc->feedback_loop && !feedback.attachmentFeedbackLoopLayout)
      skip("the attachmentFeedbackLoopLayout feature is required");

   /* Enable only what the cases use. */
   features12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
      .timelineSemaphore = VK_TRUE,
   };
   features13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = tc->feedback_loop ? &feedback : NULL,
      .dynamicRendering = VK_TRUE,
      .synchronization2 = VK_TRUE,
   };
   feedback.pNext = NULL;
   feedback.attachmentFeedbackLoopLayout = VK_TRUE;

   CHECK(vkCreateDevice(c->pdev,
                        &(VkDeviceCreateInfo){
                           .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &features12,
                           .queueCreateInfoCount = queue_info_count,
                           .pQueueCreateInfos = queue_info,
                           .enabledExtensionCount = extension_count,
                           .ppEnabledExtensionNames = extensions,
                        },
                        NULL, &c->dev));

   for (unsigned i = 0; i < 2; ++i) {
      vkGetDeviceQueue(c->dev, c->family[i], index[i], &c->queue[i]);
      CHECK(vkCreateCommandPool(
         c->dev,
         &(VkCommandPoolCreateInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = c->family[i],
         },
         NULL, &c->pool[i]));
   }
}

static void
usage(const char *argv0)
{
   fprintf(stderr,
           "usage: %s [--iterations N] [--seed N] [--shim] <case>\n"
           "       %s --list\n",
           argv0, argv0);
   exit(2);
}

static uint32_t
parse_number(const char *argv0, const char *s, uint32_t max)
{
   char *end;
   errno = 0;
   unsigned long long v = strtoull(s, &end, 0);
   if (!*s || *end || *s == '-' || errno || v > max)
      usage(argv0);
   return v;
}

int
main(int argc, char **argv)
{
   struct ctx c = {.seed = 0x716c};
   const char *name = NULL;
   uint32_t iterations = 0;

   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--list")) {
         for (unsigned j = 0; j < sizeof(cases) / sizeof(cases[0]); ++j)
            printf("%s %s\n", cases[j].name, cases[j].tier);
         return 0;
      } else if (!strcmp(argv[i], "--shim")) {
         c.shim = true;
      } else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) {
         iterations = parse_number(argv[0], argv[++i], MAX_ITERATIONS);
         if (!iterations)
            usage(argv[0]);
      } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
         c.seed = parse_number(argv[0], argv[++i], UINT32_MAX);
      } else if (argv[i][0] != '-' && !name) {
         name = argv[i];
      } else {
         usage(argv[0]);
      }
   }
   if (!name)
      usage(argv[0]);

   const struct test_case *tc = NULL;
   for (unsigned j = 0; j < sizeof(cases) / sizeof(cases[0]); ++j) {
      if (!strcmp(cases[j].name, name))
         tc = &cases[j];
   }
   if (!tc) {
      fprintf(stderr, "unknown case: %s\n", name);
      return 2;
   }

   c.name = tc->name;
   /* Two iterations submit both recorded variants. */
   c.iterations = iterations ? iterations : c.shim ? 2 : tc->default_iterations;

   create_device(&c, tc);
   const int result = tc->run(&c);

   for (unsigned i = 0; i < 2; ++i)
      vkDestroyCommandPool(c.dev, c.pool[i], NULL);
   vkDestroyDevice(c.dev, NULL);
   vkDestroyInstance(c.instance, NULL);

   if (result)
      printf("FAIL: %s: readback did not match after completed waits\n",
             tc->name);
   else if (c.shim)
      printf("PASS: %s: %u iterations recorded and submitted (no checks)\n",
             tc->name, c.iterations);
   else
      printf("PASS: %s: %u iterations\n", tc->name, c.iterations);
   return result;
}
