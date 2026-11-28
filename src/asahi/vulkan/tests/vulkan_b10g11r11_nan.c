/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * NaN and infinity in B10G11R11 images, compressed or not.
 *
 * Vulkan converts a NaN to a NaN of the unsigned 10- and 11-bit float formats,
 * and a bit-exact copy keeps it. Every write path stores a pattern with NaN
 * and +inf texels in each channel (vulkan_b10g11r11_nan_pattern.glsl), and
 * every read path checks each texel: NaN reads back as a NaN, +inf as +inf,
 * every other value exactly.
 *
 * Images: "storage" (STORAGE | SAMPLED | TRANSFER | COLOR_ATTACHMENT, ALIAS,
 * the shape of vkd3d-proton's placed UAV/render targets) and "colour"
 * (SAMPLED | TRANSFER | COLOR_ATTACHMENT).
 * Writes: render (clear, full-screen draw, store), render2 (render, then a
 * load/store pass drawing one texel), store (compute imageStore, storage
 * image only) and upload (vkCmdCopyBufferToImage of the packed words).
 * Reads: copy (vkCmdCopyImageToBuffer), fetch (texelFetch) and load
 * (imageLoad, storage image only).
 *
 * Run it with and without the driver compressing the images
 * (ASAHI_MESA_DEBUG=nocompress); AGX_MESA_DEBUG=perf shows which are.
 *
 * Exit status 0 = every texel of every case as expected, 1 = a mismatch,
 * 2 = setup failure.
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_b10g11r11_nan_shaders.h"

#define CHECK(x)                                                                         \
   do {                                                                                  \
      VkResult r_ = (x);                                                                 \
      if (r_ != VK_SUCCESS) {                                                            \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);            \
         exit(2);                                                                        \
      }                                                                                  \
   } while (0)

#define W 128
#define H 96

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkPipeline p_store, p_read, p_read_tex, p_gfx;
static VkDescriptorPool dpool;
static VkSampler smp;

struct pc { uint32_t w, h, mode, nanbits; };

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

static VkCommandBuffer
begin(void)
{
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));
   CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
   return cb;
}

static void
submit(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb}, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void
barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

static void
to_general(VkCommandBuffer cb, VkImage img)
{
   VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib});
}

/* expected channel encodings: 0 normal (value), 1 NaN, 2 inf */
struct texel { unsigned kind[3]; uint32_t enc[3]; float f[3]; };

static struct texel
expect(uint32_t x, uint32_t y)
{
   struct texel t;
   uint32_t k = x & 15, j = y & 15, a = x & 7, b = y & 7;
   t.enc[0] = (13u << 6) | (k << 2);
   t.f[0] = 0.25f + k / 64.0f;
   t.enc[1] = 14u << 6;
   t.f[1] = 0.5f;
   t.enc[2] = (14u << 5) | (16u + j);
   t.f[2] = 0.75f + j / 64.0f;
   for (int c = 0; c < 3; ++c) t.kind[c] = 0;
   if (a == 3 && b == 5) t.kind[0] = 1;
   if (a == 1 && b == 1) t.kind[0] = 2;
   if (a == 5 && b == 3) t.kind[1] = 1;
   if (a == 7 && b == 0) t.kind[1] = 2;
   if (a == 6 && b == 6) t.kind[2] = 1;
   if (a == 2 && b == 7) t.kind[2] = 2;
   return t;
}

static uint32_t
pack_expected(uint32_t x, uint32_t y)
{
   struct texel t = expect(x, y);
   uint32_t e[3];
   for (int c = 0; c < 3; ++c) {
      unsigned mb = c == 2 ? 5 : 6;
      e[c] = t.kind[c] == 1 ? ((31u << mb) | (1u << (mb - 1))) : t.kind[c] == 2 ? (31u << mb) : t.enc[c];
   }
   return e[0] | (e[1] << 11) | (e[2] << 22);
}

struct tally { unsigned nan_ok, nan_n, inf_ok, inf_n, oth_ok, oth_n; char first[160]; };

static void
check_word(struct tally *T, uint32_t x, uint32_t y, uint32_t v)
{
   struct texel t = expect(x, y);
   uint32_t ch[3] = {v & 0x7ff, (v >> 11) & 0x7ff, v >> 22};
   for (int c = 0; c < 3; ++c) {
      unsigned mb = c == 2 ? 5 : 6;
      uint32_t ex = ch[c] >> mb, ma = ch[c] & ((1u << mb) - 1);
      bool ok;
      if (t.kind[c] == 1) { T->nan_n++; ok = ex == 31 && ma; T->nan_ok += ok; }
      else if (t.kind[c] == 2) { T->inf_n++; ok = ex == 31 && !ma; T->inf_ok += ok; }
      else { T->oth_n++; ok = ch[c] == t.enc[c]; T->oth_ok += ok; }
      if (!ok && !T->first[0])
         snprintf(T->first, sizeof(T->first), "%u,%u c%d word=0x%08x want=0x%08x", x, y, c, v, pack_expected(x, y));
   }
}

static void
check_float(struct tally *T, uint32_t x, uint32_t y, const uint32_t *v)
{
   struct texel t = expect(x, y);
   for (int c = 0; c < 3; ++c) {
      float f;
      memcpy(&f, &v[c], 4);
      bool ok;
      if (t.kind[c] == 1) { T->nan_n++; ok = isnan(f); T->nan_ok += ok; }
      else if (t.kind[c] == 2) { T->inf_n++; ok = isinf(f) && f > 0; T->inf_ok += ok; }
      else { T->oth_n++; ok = f == t.f[c]; T->oth_ok += ok; }
      if (!ok && !T->first[0])
         snprintf(T->first, sizeof(T->first), "%u,%u c%d got=0x%08x/0x%08x/0x%08x", x, y, c, v[0], v[1], v[2]);
   }
}

/* returns 1 if any texel of the case is wrong */
static unsigned
print_tally(const char *kind, const char *wr, const char *rd, const struct tally *T)
{
   const bool ok = T->nan_ok == T->nan_n && T->inf_ok == T->inf_n && T->oth_ok == T->oth_n;
   printf("RESULT %-7s %-7s %-5s nan=%u/%u inf=%u/%u other=%u/%u %s%s%s\n", kind, wr, rd, T->nan_ok,
          T->nan_n, T->inf_ok, T->inf_n, T->oth_ok, T->oth_n, ok ? "PASS" : "FAIL",
          T->first[0] ? " first " : "", T->first);
   return !ok;
}

static VkShaderModule
sm(const uint32_t *code, size_t size)
{
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code}, NULL, &m));
   return m;
}

static VkPipeline
cpipe(const uint32_t *code, size_t size)
{
   VkPipeline p;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm(code, size), .pName = "main"},
      .layout = pl}, NULL, &p));
   return p;
}

static VkPipeline
gpipe(void)
{
   VkPipeline p;
   VkFormat f = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = sm(nan_vert, sizeof(nan_vert)), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = sm(nan_frag, sizeof(nan_frag)), .pName = "main"},
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                .colorAttachmentCount = 1, .pColorAttachmentFormats = &f},
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .lineWidth = 1.0f},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}},
      .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2,
         .pDynamicStates = dyn},
      .layout = pl}, NULL, &p));
   return p;
}

static const struct pc PC = {W, H, 0, 0x7fc00000u};

static void
draw(VkCommandBuffer cb, VkImageView v, bool load, VkRect2D sc)
{
   vkCmdBeginRendering(cb, &(VkRenderingInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {W, H}}, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_gfx);
   vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

/* returns the number of failing read paths */
static unsigned
run(int kind, int wpath)
{
   static const char *wname[] = {"render", "render2", "store", "upload"};
   const bool storage = kind == 0;
   if (wpath == 2 && !storage)
      return 0;
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = storage ? VK_IMAGE_CREATE_ALIAS_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0) | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage img;
   CHECK(vkCreateImage(dev, &ci, NULL, &img));
   VkMemoryRequirements ir;
   vkGetImageMemoryRequirements(dev, img, &ir);
   VkBuffer alias;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = ir.size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &alias));
   VkMemoryRequirements ar;
   vkGetBufferMemoryRequirements(dev, alias, &ar);
   VkDeviceMemory im;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = ir.size > ar.size ? ir.size : ar.size,
      .memoryTypeIndex = mem_type(ir.memoryTypeBits & ar.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      NULL, &im));
   CHECK(vkBindImageMemory(dev, img, im, 0));
   CHECK(vkBindBufferMemory(dev, alias, im, 0));

   VkImageView v;
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ci.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &v));

   /* host buffer: [upload words | copy readback | float readback (16 B/texel) | raw dump] */
   const VkDeviceSize plane = (VkDeviceSize)W * H * 4, fplane = (VkDeviceSize)W * H * 16;
   const VkDeviceSize hsz = 2 * plane + fplane + ir.size;
   VkBuffer hb;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hsz,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, NULL, &hb));
   VkMemoryRequirements hr;
   vkGetBufferMemoryRequirements(dev, hb, &hr);
   VkDeviceMemory hm;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = hr.size, .memoryTypeIndex = mem_type(hr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &hm));
   CHECK(vkBindBufferMemory(dev, hb, hm, 0));
   uint8_t *hp;
   CHECK(vkMapMemory(dev, hm, 0, VK_WHOLE_SIZE, 0, (void **)&hp));
   memset(hp, 0, hsz);
   uint32_t *up = (uint32_t *)hp;
   for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
         up[y * W + x] = pack_expected(x, y);

   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
      .descriptorSetCount = 1, .pSetLayouts = &dsl}, &set));
   VkDescriptorImageInfo ii = {.sampler = smp, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   VkDescriptorBufferInfo bi = {hb, 2 * plane, fplane};
   VkWriteDescriptorSet wr[3] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi},
   };
   /* a storage descriptor of a non-storage image is invalid: the colour image binds only 1 and 2 */
   vkUpdateDescriptorSets(dev, storage ? 3 : 2, storage ? wr : wr + 1, 0, NULL);

   VkCommandBuffer cb = begin();
   to_general(cb, img);
   if (wpath <= 1) {
      draw(cb, v, false, (VkRect2D){{0, 0}, {W, H}});
      if (wpath == 1) {
         barrier(cb);
         draw(cb, v, true, (VkRect2D){{0, 0}, {1, 1}});
      }
   } else if (wpath == 2) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_store);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
      vkCmdDispatch(cb, W / 8, H / 8, 1);
   } else {
      vkCmdCopyBufferToImage(cb, hb, img, VK_IMAGE_LAYOUT_GENERAL, 1, &(VkBufferImageCopy){
         .bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}});
   }
   barrier(cb);
   vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &(VkBufferImageCopy){
      .bufferOffset = plane, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}});
   vkCmdCopyBuffer(cb, alias, hb, 1, &(VkBufferCopy){0, 2 * plane + fplane, ir.size});
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, storage ? p_read : p_read_tex);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(PC), &PC);
   vkCmdDispatch(cb, W / 8, H / 8, 1);
   submit(cb);

   const char *kn = storage ? "storage" : "colour";
   unsigned fails = 0;
   struct tally tc = {0}, tf = {0}, tl = {0};
   const uint32_t *cw = (const uint32_t *)(hp + plane), *fw = (const uint32_t *)(hp + 2 * plane);
   for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
         check_word(&tc, x, y, cw[y * W + x]);
         check_float(&tf, x, y, &fw[(y * W + x) * 4]);
      }
   fails += print_tally(kn, wname[wpath], "copy", &tc);
   fails += print_tally(kn, wname[wpath], "fetch", &tf);

   if (storage) {
      memset(hp + 2 * plane, 0, fplane);
      cb = begin();
      struct pc pc1 = PC;
      pc1.mode = 1;
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_read);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL, 0, sizeof(pc1), &pc1);
      vkCmdDispatch(cb, W / 8, H / 8, 1);
      submit(cb);
      for (uint32_t y = 0; y < H; ++y)
         for (uint32_t x = 0; x < W; ++x)
            check_float(&tl, x, y, &fw[(y * W + x) * 4]);
      fails += print_tally(kn, wname[wpath], "load", &tl);
   }

   vkUnmapMemory(dev, hm);
   vkDestroyBuffer(dev, hb, NULL);
   vkFreeMemory(dev, hm, NULL);
   vkDestroyImageView(dev, v, NULL);
   vkDestroyBuffer(dev, alias, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, im, NULL);
   CHECK(vkResetDescriptorPool(dev, dpool, 0));
   return fails;
}

static uint32_t
pack_expected_payload(uint32_t x, uint32_t y)
{
   /* Non-canonical NaN payloads (0x7c1, 0x7ff, 0x3e1, 0x3ff) alongside normal
    * and +inf encodings so bit-exact copy readbacks catch payload dropping. */
   struct texel t = expect(x, y);
   uint32_t e[3];
   const uint32_t sel = ((x >> 3) + (y >> 3)) & 3u;
   for (int c = 0; c < 3; ++c) {
      unsigned mb = c == 2 ? 5 : 6;
      if (t.kind[c] == 1) {
         uint32_t mant = sel == 0 ? 1u
                       : sel == 1 ? ((1u << mb) - 1u)
                       : sel == 2 ? (1u << (mb - 1))
                                  : (0x15u & ((1u << mb) - 1u));
         e[c] = (31u << mb) | mant;
      } else if (t.kind[c] == 2) {
         e[c] = 31u << mb;
      } else {
         e[c] = t.enc[c];
      }
   }
   uint32_t w = e[0] | (e[1] << 11) | (e[2] << 22);
   /* Explicitly hit the exact corner payloads on the first row. */
   if (y == 0 && x == 0) return 0x000007c1u;
   if (y == 0 && x == 1) return 0x000007ffu;
   if (y == 0 && x == 2) return 0x003e0800u;
   if (y == 0 && x == 3) return 0x003ff800u;
   if (y == 0 && x == 4) return 0xf8400000u;
   if (y == 0 && x == 5) return 0xffffffffu;
   return w;
}

static unsigned
check_exact_words(const char *tag, const uint32_t *got, const uint32_t *want,
                  uint32_t count)
{
   unsigned ok = 0;
   char first[160] = {0};
   for (uint32_t i = 0; i < count; ++i) {
      if (got[i] == want[i]) {
         ok++;
      } else if (!first[0]) {
         snprintf(first, sizeof(first), "%u,%u got=0x%08x want=0x%08x",
                  i % W, i / W, got[i], want[i]);
      }
   }
   const bool pass = ok == count;
   printf("RESULT %-21s exact=%u/%u %s%s%s\n", tag, ok, count,
          pass ? "PASS" : "FAIL", first[0] ? " first " : "", first);
   return !pass;
}

static VkImage
make_image(VkFormat fmt, VkImageUsageFlags usage, VkImageCreateFlags flags,
           VkDeviceMemory *mem_out, VkMemoryRequirements *mr_out)
{
   VkImage img;
   CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {W, H, 1},
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED}, NULL, &img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   if (mr_out)
      *mr_out = mr;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      NULL, mem_out));
   CHECK(vkBindImageMemory(dev, img, *mem_out, 0));
   return img;
}

/* Bit-exact copy contract for non-canonical NaN payloads across:
 *   1. upload -> CopyImageToBuffer (bit-exact)
 *   2. upload -> CopyImage(B10G11R11 -> R32_UINT) -> CopyImageToBuffer
 *   3. R32_UINT -> CopyImage(R32_UINT -> B10G11R11) -> CopyImageToBuffer
 *   4. B10G11R11 -> CopyImage(partial + full -> B10G11R11) -> CopyImageToBuffer
 *   5. render (compressed) -> partial non-aligned upload -> CopyImageToBuffer
 *   6. ALIAS_BIT garbage memory -> full upload -> CopyImageToBuffer
 *   7. vkCmdClearColorImage -> CopyImageToBuffer */
static unsigned
run_bitexact_copies(int kind)
{
   const bool storage = kind == 0;
   const char *kn = storage ? "storage" : "colour";
   const VkImageUsageFlags r11_usage =
      (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0) | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   const VkImageCreateFlags r11_flags = storage ? VK_IMAGE_CREATE_ALIAS_BIT : 0;

   VkDeviceMemory m_a, m_b, m_u32;
   VkMemoryRequirements mr_a;
   VkImage img_a = make_image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, r11_usage,
                              r11_flags, &m_a, &mr_a);
   VkImage img_b = make_image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, r11_usage,
                              r11_flags, &m_b, NULL);
   VkImage img_u32 = make_image(VK_FORMAT_R32_UINT,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                                0, &m_u32, NULL);

   VkImageView v_a;
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img_a,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &v_a));

   const VkDeviceSize plane = (VkDeviceSize)W * H * 4;
   const VkDeviceSize hsz = 6 * plane;
   VkBuffer hb;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hsz,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &hb));
   VkMemoryRequirements hr;
   vkGetBufferMemoryRequirements(dev, hb, &hr);
   VkDeviceMemory hm;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = hr.size,
      .memoryTypeIndex = mem_type(hr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &hm));
   CHECK(vkBindBufferMemory(dev, hb, hm, 0));
   uint32_t *hp;
   CHECK(vkMapMemory(dev, hm, 0, VK_WHOLE_SIZE, 0, (void **)&hp));
   memset(hp, 0, hsz);

   uint32_t *want = hp;
   for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
         want[y * W + x] = pack_expected_payload(x, y);

   unsigned fails = 0;
   char tag[48];
   VkBufferImageCopy full_bic = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {W, H, 1},
   };
   VkImageCopy full_ic = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent = {W, H, 1},
   };

   /* 1..4: upload -> copy2buf, -> R32_UINT, R32_UINT -> R11, partial R11 -> R11 */
   VkCommandBuffer cb = begin();
   to_general(cb, img_a);
   to_general(cb, img_b);
   to_general(cb, img_u32);
   /* Upload non-canonical payloads to img_a */
   full_bic.bufferOffset = 0;
   vkCmdCopyBufferToImage(cb, hb, img_a, VK_IMAGE_LAYOUT_GENERAL, 1, &full_bic);
   barrier(cb);
   /* Readback 1: img_a -> buf[1] */
   full_bic.bufferOffset = 1 * plane;
   vkCmdCopyImageToBuffer(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   /* Copy 2: img_a (compressed R11) -> img_u32 (R32_UINT) -> buf[2] */
   vkCmdCopyImage(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, img_u32,
                  VK_IMAGE_LAYOUT_GENERAL, 1, &full_ic);
   barrier(cb);
   full_bic.bufferOffset = 2 * plane;
   vkCmdCopyImageToBuffer(cb, img_u32, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   /* Copy 3: img_u32 (R32_UINT) -> img_b (compressed R11) -> buf[3] */
   vkCmdCopyImage(cb, img_u32, VK_IMAGE_LAYOUT_GENERAL, img_b,
                  VK_IMAGE_LAYOUT_GENERAL, 1, &full_ic);
   barrier(cb);
   full_bic.bufferOffset = 3 * plane;
   vkCmdCopyImageToBuffer(cb, img_b, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   submit(cb);

   snprintf(tag, sizeof(tag), "%s/up->c2b", kn);
   fails += check_exact_words(tag, hp + 1 * (W * H), want, W * H);
   snprintf(tag, sizeof(tag), "%s/r11->r32", kn);
   fails += check_exact_words(tag, hp + 2 * (W * H), want, W * H);
   snprintf(tag, sizeof(tag), "%s/r32->r11", kn);
   fails += check_exact_words(tag, hp + 3 * (W * H), want, W * H);

   /* 5: Render compressed img_a, then partial non-16-aligned upload (3,5,37,29)
    * of non-canonical NaN payloads, then CopyImageToBuffer: inside the rect
    * must be bit-exact payload, outside (including shared border tiles!) must
    * match rendered values. */
   const uint32_t rx = 3, ry = 5, rw = 37, rh = 29;
   uint32_t *sub_up = hp + 4 * (W * H);
   for (uint32_t y = 0; y < rh; ++y)
      for (uint32_t x = 0; x < rw; ++x)
         sub_up[y * rw + x] = pack_expected_payload(x, y);

   cb = begin();
   to_general(cb, img_a);
   draw(cb, v_a, false, (VkRect2D){{0, 0}, {W, H}});
   barrier(cb);
   vkCmdCopyBufferToImage(cb, hb, img_a, VK_IMAGE_LAYOUT_GENERAL, 1,
                          &(VkBufferImageCopy){
                             .bufferOffset = 4 * plane,
                             .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageOffset = {(int32_t)rx, (int32_t)ry, 0},
                             .imageExtent = {rw, rh, 1},
                          });
   barrier(cb);
   full_bic.bufferOffset = 5 * plane;
   vkCmdCopyImageToBuffer(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   submit(cb);

   {
      const uint32_t *got = hp + 5 * (W * H);
      unsigned inside_ok = 0, inside_n = 0;
      struct tally tout = {0};
      char first_in[160] = {0};
      for (uint32_t y = 0; y < H; ++y) {
         for (uint32_t x = 0; x < W; ++x) {
            uint32_t v = got[y * W + x];
            if (x >= rx && x < rx + rw && y >= ry && y < ry + rh) {
               uint32_t w = sub_up[(y - ry) * rw + (x - rx)];
               inside_n++;
               if (v == w) inside_ok++;
               else if (!first_in[0])
                  snprintf(first_in, sizeof(first_in), "%u,%u got=0x%08x want=0x%08x", x, y, v, w);
            } else {
               check_word(&tout, x, y, v);
            }
         }
      }
      bool ok_in = inside_ok == inside_n;
      bool ok_out = tout.nan_ok == tout.nan_n && tout.inf_ok == tout.inf_n &&
                    tout.oth_ok == tout.oth_n;
      printf("RESULT %s/partial-up  in=%u/%u out_nan=%u/%u out_oth=%u/%u %s%s%s%s%s\n",
             kn, inside_ok, inside_n, tout.nan_ok, tout.nan_n, tout.oth_ok,
             tout.oth_n, (ok_in && ok_out) ? "PASS" : "FAIL",
             first_in[0] ? " in:" : "", first_in,
             tout.first[0] ? " out:" : "", tout.first);
      if (!ok_in || !ok_out) fails++;
   }

   /* 6: Partial CopyImage (non-aligned rect from img_b -> rendered img_a) */
   cb = begin();
   to_general(cb, img_a);
   draw(cb, v_a, false, (VkRect2D){{0, 0}, {W, H}});
   barrier(cb);
   vkCmdCopyImage(cb, img_b, VK_IMAGE_LAYOUT_GENERAL, img_a,
                  VK_IMAGE_LAYOUT_GENERAL, 1,
                  &(VkImageCopy){
                     .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffset = {0, 0, 0},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .dstOffset = {(int32_t)rx, (int32_t)ry, 0},
                     .extent = {rw, rh, 1},
                  });
   barrier(cb);
   full_bic.bufferOffset = 5 * plane;
   vkCmdCopyImageToBuffer(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   submit(cb);
   {
      const uint32_t *got = hp + 5 * (W * H);
      unsigned inside_ok = 0, inside_n = 0;
      struct tally tout = {0};
      char first_in[160] = {0};
      for (uint32_t y = 0; y < H; ++y) {
         for (uint32_t x = 0; x < W; ++x) {
            uint32_t v = got[y * W + x];
            if (x >= rx && x < rx + rw && y >= ry && y < ry + rh) {
               uint32_t w = want[(y - ry) * W + (x - rx)];
               inside_n++;
               if (v == w) inside_ok++;
               else if (!first_in[0])
                  snprintf(first_in, sizeof(first_in), "%u,%u got=0x%08x want=0x%08x", x, y, v, w);
            } else {
               check_word(&tout, x, y, v);
            }
         }
      }
      bool ok_in = inside_ok == inside_n;
      bool ok_out = tout.nan_ok == tout.nan_n && tout.inf_ok == tout.inf_n &&
                    tout.oth_ok == tout.oth_n;
      printf("RESULT %s/partial-c2i in=%u/%u out_nan=%u/%u out_oth=%u/%u %s%s%s%s%s\n",
             kn, inside_ok, inside_n, tout.nan_ok, tout.nan_n, tout.oth_ok,
             tout.oth_n, (ok_in && ok_out) ? "PASS" : "FAIL",
             first_in[0] ? " in:" : "", first_in,
             tout.first[0] ? " out:" : "", tout.first);
      if (!ok_in || !ok_out) fails++;
   }

   /* 7: vkCmdClearColorImage on compressed B10G11R11 */
   cb = begin();
   to_general(cb, img_a);
   vkCmdClearColorImage(cb, img_a, VK_IMAGE_LAYOUT_GENERAL,
                        &(VkClearColorValue){.float32 = {1.0f, 0.5f, 2.0f, 0.0f}},
                        1, &(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
   barrier(cb);
   full_bic.bufferOffset = 5 * plane;
   vkCmdCopyImageToBuffer(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
   submit(cb);
   {
      const uint32_t exp_clear = 0x3c0u | (0x380u << 11) | (0x200u << 22);
      const uint32_t *got = hp + 5 * (W * H);
      uint32_t *wclr = hp + 4 * (W * H);
      for (uint32_t i = 0; i < W * H; ++i) wclr[i] = exp_clear;
      snprintf(tag, sizeof(tag), "%s/clear-img", kn);
      fails += check_exact_words(tag, got, wclr, W * H);
   }

   /* 8: compressed B10G11R11 <-> linear host-visible B10G11R11 via vkMapMemory */
   {
      VkImage img_lin;
      CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
         .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
         .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED}, NULL, &img_lin));
      VkMemoryRequirements lmr;
      vkGetImageMemoryRequirements(dev, img_lin, &lmr);
      VkDeviceMemory m_lin;
      CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = lmr.size,
         .memoryTypeIndex = mem_type(lmr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &m_lin));
      CHECK(vkBindImageMemory(dev, img_lin, m_lin, 0));
      VkSubresourceLayout sl;
      vkGetImageSubresourceLayout(dev, img_lin,
         &(VkImageSubresource){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, &sl);
      uint8_t *lp;
      CHECK(vkMapMemory(dev, m_lin, 0, VK_WHOLE_SIZE, 0, (void **)&lp));

      cb = begin();
      to_general(cb, img_a);
      to_general(cb, img_lin);
      full_bic.bufferOffset = 0;
      vkCmdCopyBufferToImage(cb, hb, img_a, VK_IMAGE_LAYOUT_GENERAL, 1, &full_bic);
      barrier(cb);
      vkCmdCopyImage(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, img_lin,
                     VK_IMAGE_LAYOUT_GENERAL, 1, &full_ic);
      submit(cb);

      uint32_t *got_lin = hp + 5 * (W * H);
      for (uint32_t y = 0; y < H; ++y)
         memcpy(got_lin + y * W, lp + sl.offset + y * sl.rowPitch, W * 4);
      snprintf(tag, sizeof(tag), "%s/host-linear", kn);
      fails += check_exact_words(tag, got_lin, want, W * H);

      vkUnmapMemory(dev, m_lin);
      vkDestroyImage(dev, img_lin, NULL);
      vkFreeMemory(dev, m_lin, NULL);
   }

   /* 9 (storage/ALIAS_BIT): fill backing memory with 0x5a garbage, then full
    * upload of non-canonical NaN payloads without clearing first. */
   if (storage) {
      VkBuffer alias_buf;
      CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = mr_a.size,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &alias_buf));
      CHECK(vkBindBufferMemory(dev, alias_buf, m_a, 0));
      cb = begin();
      vkCmdFillBuffer(cb, alias_buf, 0, mr_a.size, 0x5a5a5a5au);
      barrier(cb);
      to_general(cb, img_a);
      full_bic.bufferOffset = 0;
      vkCmdCopyBufferToImage(cb, hb, img_a, VK_IMAGE_LAYOUT_GENERAL, 1, &full_bic);
      barrier(cb);
      full_bic.bufferOffset = 5 * plane;
      vkCmdCopyImageToBuffer(cb, img_a, VK_IMAGE_LAYOUT_GENERAL, hb, 1, &full_bic);
      submit(cb);
      fails += check_exact_words("storage/alias-garb", hp + 5 * (W * H), want, W * H);
      vkDestroyBuffer(dev, alias_buf, NULL);
   }

   vkUnmapMemory(dev, hm);
   vkDestroyBuffer(dev, hb, NULL);
   vkFreeMemory(dev, hm, NULL);
   vkDestroyImageView(dev, v_a, NULL);
   vkDestroyImage(dev, img_a, NULL);
   vkDestroyImage(dev, img_b, NULL);
   vkDestroyImage(dev, img_u32, NULL);
   vkFreeMemory(dev, m_a, NULL);
   vkFreeMemory(dev, m_b, NULL);
   vkFreeMemory(dev, m_u32, NULL);
   return fails;
}

int
main(void)
{
   CHECK(vkCreateInstance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                               .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if (!n)
      return 2;
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
   uint32_t nq = 8;
   VkQueueFamilyProperties qf[8];
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qf);
   for (qfam = 0; qfam < nq && !(qf[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT); ++qfam)
      ;
   float prio = 1.0f;
   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
   CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1,
         .pQueuePriorities = &prio}}, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam}, NULL, &pool));
   VkDescriptorSetLayoutBinding b[3] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b}, NULL, &dsl));
   VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
                                 {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
   CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 3, .pPoolSizes = ps},
      NULL, &dpool));
   CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_ALL, 0, sizeof(struct pc)}}, NULL, &pl));
   CHECK(vkCreateSampler(dev, &(VkSamplerCreateInfo){.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST}, NULL, &smp));
   p_store = cpipe(store_comp, sizeof(store_comp));
   p_read = cpipe(load_comp, sizeof(load_comp));
   p_read_tex = cpipe(fetch_comp, sizeof(fetch_comp));
   p_gfx = gpipe();
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(pdev, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &fp);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   if ((fp.optimalTilingFeatures & need) != need) {
      printf("SKIP B10G11R11 lacks storage, sampled or colour attachment support\n");
      return 0;
   }
   printf("DEVICE %s\n", props.deviceName);
   unsigned fails = 0, cases = 0;
   for (int kind = 0; kind < 2; ++kind) {
      for (int wpath = 0; wpath < 4; ++wpath) {
         if (wpath == 2 && kind)
            continue;
         fails += run(kind, wpath);
         cases += kind ? 2 : 3;
      }
      fails += run_bitexact_copies(kind);
      cases += kind ? 7 : 8;
   }
   printf("SUMMARY %u/%u cases pass\n", cases - fails, cases);
   return fails ? 1 : 0;
}
