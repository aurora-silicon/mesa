/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify batched cache-clean publication and byte-for-byte descriptor parity
 * for descriptor array updates, templates, push descriptors, immutable sampler
 * allocation, mutable bindings, and vkGetDescriptorEXT across AGX2, Gen9, and
 * Gen10.
 */
#undef NDEBUG
#include "agx_nir.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "vk_acceleration_structure.h"
#include "vk_ycbcr_conversion.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FLUSH_RECORDS 128

struct flush_record {
   void *start;
   size_t size;
};

static struct flush_record flushes[MAX_FLUSH_RECORDS];
static unsigned flush_count;

void __wrap_util_flush_range_no_fence(void *start, size_t size);
void
__wrap_util_flush_range_no_fence(void *start, size_t size)
{
   if (flush_count < MAX_FLUSH_RECORDS) {
      flushes[flush_count].start = start;
      flushes[flush_count].size = size;
   }
   flush_count++;
}

static void
reset_flushes(void)
{
   flush_count = 0;
   memset(flushes, 0, sizeof(flushes));
}

struct test_env {
   struct hk_physical_device pdev;
   struct hk_device dev;
   struct hk_image img;
   struct hk_image_view views_1p[4];
   struct hk_image_view views_3p[4];
   struct hk_rc_sampler rc_samplers[4];
   struct hk_sampler samplers_1p[4];
   struct hk_sampler samplers_ycbcr[4];
   struct vk_ycbcr_conversion ycbcr;
   struct hk_buffer buffers[4];
   struct hk_buffer_view bviews[4];
   struct vk_acceleration_structure accels[4];
};

static void
fill_pattern(void *ptr, size_t size, uint8_t seed)
{
   uint8_t *p = ptr;
   for (size_t i = 0; i < size; ++i)
      p[i] = (uint8_t)(seed + i * 13);
}

static struct test_env *
create_env(enum agx_chip chip, bool m5_native_as)
{
   struct test_env *e = calloc(1, sizeof(*e));
   assert(e);

   e->pdev.dev.chip = chip;
   vk_object_base_init(NULL, &e->pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   e->dev.vk.physical = &e->pdev.vk;
   e->dev.dev.chip = chip;
   e->dev.m5_native_as = m5_native_as;
   e->dev.vk.alloc = *vk_default_allocator();
   vk_object_base_init(&e->dev.vk, &e->dev.vk.base, VK_OBJECT_TYPE_DEVICE);

   e->img.vk.samples = VK_SAMPLE_COUNT_1_BIT;

   e->ycbcr.state.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
   vk_object_base_init(NULL, &e->ycbcr.base,
                       VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION);

   for (unsigned i = 0; i < 4; ++i) {
      e->rc_samplers[i].index = 10 + i;
      fill_pattern(&e->rc_samplers[i].key, sizeof(e->rc_samplers[i].key),
                   0x10 + i);

      vk_object_base_init(&e->dev.vk, &e->samplers_1p[i].vk.base,
                          VK_OBJECT_TYPE_SAMPLER);
      e->samplers_1p[i].plane_count = 1;
      e->samplers_1p[i].planes[0].hw = &e->rc_samplers[i];
      e->samplers_1p[i].lod_bias_fp16 = 0x3c00 + i;
      e->samplers_1p[i].mip_lod_bias = 0.25f * (float)i;

      vk_object_base_init(&e->dev.vk, &e->samplers_ycbcr[i].vk.base,
                          VK_OBJECT_TYPE_SAMPLER);
      e->samplers_ycbcr[i].plane_count = 1;
      e->samplers_ycbcr[i].planes[0].hw = &e->rc_samplers[i];
      e->samplers_ycbcr[i].vk.ycbcr_conversion = &e->ycbcr;
      e->samplers_ycbcr[i].ycbcr = e->ycbcr.state;

      vk_object_base_init(&e->dev.vk, &e->views_1p[i].vk.base,
                          VK_OBJECT_TYPE_IMAGE_VIEW);
      e->views_1p[i].vk.image = &e->img.vk;
      e->views_1p[i].vk.format = VK_FORMAT_R8G8B8A8_UNORM;
      e->views_1p[i].vk.swizzle = (VkComponentMapping){
         .r = VK_COMPONENT_SWIZZLE_R,
         .g = VK_COMPONENT_SWIZZLE_G,
         .b = VK_COMPONENT_SWIZZLE_B,
         .a = VK_COMPONENT_SWIZZLE_A,
      };
      e->views_1p[i].plane_count = 1;
      e->views_1p[i].sampled_min_lod_fp16 = 0x1000 + i;
      e->views_1p[i].sampled_min_lod_uint16 = 1 + i;
      e->views_1p[i].planes[0].sampled_heap_index = 100 + i;
      e->views_1p[i].planes[0].input_heap_index = 200 + i;
      e->views_1p[i].planes[0].storage_heap_index = 300 + i;
      e->views_1p[i].planes[0].storage_pbe_valid = true;
      e->views_1p[i].planes[0].storage_atomic_sw = UINT64_C(0x1122334455660000) + i;
      for (unsigned w = 0; w < 4; ++w)
         e->views_1p[i].planes[0].storage_window[w] = 0x400 + i * 4 + w;
      fill_pattern(&e->views_1p[i].planes[0].sampled,
                   sizeof(e->views_1p[i].planes[0].sampled), 0x20 + i);
      fill_pattern(&e->views_1p[i].planes[0].ia,
                   sizeof(e->views_1p[i].planes[0].ia), 0x30 + i);
      fill_pattern(&e->views_1p[i].planes[0].ro_storage,
                   sizeof(e->views_1p[i].planes[0].ro_storage), 0x40 + i);
      fill_pattern(&e->views_1p[i].planes[0].storage,
                   sizeof(e->views_1p[i].planes[0].storage), 0x50 + i);

      vk_object_base_init(&e->dev.vk, &e->views_3p[i].vk.base,
                          VK_OBJECT_TYPE_IMAGE_VIEW);
      e->views_3p[i].vk.image = &e->img.vk;
      e->views_3p[i].vk.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
      e->views_3p[i].plane_count = 3;
      for (unsigned p = 0; p < 3; ++p) {
         e->views_3p[i].planes[p].sampled_heap_index = 500 + i * 3 + p;
         fill_pattern(&e->views_3p[i].planes[p].sampled,
                      sizeof(e->views_3p[i].planes[p].sampled),
                      0x60 + i * 3 + p);
      }

      vk_object_base_init(&e->dev.vk, &e->buffers[i].vk.base,
                          VK_OBJECT_TYPE_BUFFER);
      e->buffers[i].vk.device_address = UINT64_C(0x10000000) + i * 0x10000;
      e->buffers[i].vk.size = 4096 + i * 256;

      vk_object_base_init(&e->dev.vk, &e->bviews[i].vk.base,
                          VK_OBJECT_TYPE_BUFFER_VIEW);
      fill_pattern(&e->bviews[i].tex, sizeof(e->bviews[i].tex), 0x80 + i);
      fill_pattern(&e->bviews[i].pbe, sizeof(e->bviews[i].pbe), 0x90 + i);
      e->bviews[i].sampled_heap_index = 700 + i;
      e->bviews[i].buffer_metadata = (struct hk_g17p_buffer_texture_metadata){
         .elements = 256 + i,
         .offset_el = i * 4,
         .address = UINT64_C(0x20000000) + i * 0x1000,
      };

      vk_object_base_init(&e->dev.vk, &e->accels[i].base,
                          VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
      e->accels[i].addr = UINT64_C(0x30000000) + i * 0x2000 + 0x100;
   }

   return e;
}

static void
destroy_env(struct test_env *e)
{
   for (unsigned i = 0; i < 4; ++i) {
      vk_object_base_finish(&e->samplers_1p[i].vk.base);
      vk_object_base_finish(&e->samplers_ycbcr[i].vk.base);
      vk_object_base_finish(&e->views_1p[i].vk.base);
      vk_object_base_finish(&e->views_3p[i].vk.base);
      vk_object_base_finish(&e->buffers[i].vk.base);
      vk_object_base_finish(&e->bviews[i].vk.base);
      vk_object_base_finish(&e->accels[i].base);
   }
   vk_object_base_finish(&e->ycbcr.base);
   vk_object_base_finish(&e->dev.vk.base);
   vk_object_base_finish(&e->pdev.vk.base);
   free(e);
}

static struct hk_descriptor_set_layout *
create_layout(struct test_env *e, VkDescriptorSetLayoutCreateFlags flags,
              const VkDescriptorSetLayoutBinding *bindings,
              uint32_t binding_count, const void *pNext)
{
   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = pNext,
      .flags = flags,
      .bindingCount = binding_count,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout handle = VK_NULL_HANDLE;
   assert(hk_CreateDescriptorSetLayout(hk_device_to_handle(&e->dev), &info,
                                       NULL, &handle) == VK_SUCCESS);
   return hk_descriptor_set_layout_from_handle(handle);
}

static void
init_set(struct hk_descriptor_set *set, struct hk_descriptor_set_layout *layout,
         uint8_t *buf, uint32_t size, bool cpu_cached)
{
   memset(set, 0, sizeof(*set));
   vk_object_base_init(NULL, &set->base, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   set->layout = layout;
   set->mapped_ptr = buf;
   set->size = size;
   set->cpu_cached = cpu_cached;
}

static void
test_contiguous_array_types(enum agx_chip chip, bool m5_native_as)
{
   struct test_env *e = create_env(chip, m5_native_as);
   VkDevice dev = hk_device_to_handle(&e->dev);

   VkSampler imm_1p[4];
   for (unsigned i = 0; i < 4; ++i)
      imm_1p[i] = hk_sampler_to_handle(&e->samplers_1p[i]);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4, VK_SHADER_STAGE_ALL, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_ALL, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_ALL, imm_1p},
      {3, VK_DESCRIPTOR_TYPE_SAMPLER, 4, VK_SHADER_STAGE_ALL, NULL},
      {4, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 4, VK_SHADER_STAGE_ALL, NULL},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4, VK_SHADER_STAGE_ALL, NULL},
      {6, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {10, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4, VK_SHADER_STAGE_ALL, NULL},
   };
   struct hk_descriptor_set_layout *layout =
      create_layout(e, 0, bindings, ARRAY_SIZE(bindings), NULL);

   const uint32_t set_size = align64(layout->non_variable_descriptor_buffer_size,
                                     HK_MIN_UBO_ALIGNMENT);
   uint8_t *buf_batch = calloc(1, set_size + 64);
   uint8_t *buf_single = calloc(1, set_size + 64);
   uint8_t *buf_tmpl = calloc(1, set_size + 64);
   assert(buf_batch && buf_single && buf_tmpl);
   memset(buf_batch + set_size, 0xa5, 64);
   memset(buf_single + set_size, 0xa5, 64);
   memset(buf_tmpl + set_size, 0xa5, 64);

   struct hk_descriptor_set set_batch, set_single, set_tmpl;
   init_set(&set_batch, layout, buf_batch, set_size, true);
   init_set(&set_single, layout, buf_single, set_size, true);
   init_set(&set_tmpl, layout, buf_tmpl, set_size, true);

   VkDescriptorImageInfo img_infos[4];
   VkDescriptorBufferInfo buf_infos[4];
   VkBufferView bview_handles[4];
   VkAccelerationStructureKHR accel_handles[4];
   for (unsigned i = 0; i < 4; ++i) {
      img_infos[i] = (VkDescriptorImageInfo){
         .sampler = hk_sampler_to_handle(&e->samplers_1p[i]),
         .imageView = hk_image_view_to_handle(&e->views_1p[i]),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      buf_infos[i] = (VkDescriptorBufferInfo){
         .buffer = hk_buffer_to_handle(&e->buffers[i]),
         .offset = i * 64,
         .range = 1024 + i * 64,
      };
      bview_handles[i] = hk_buffer_view_to_handle(&e->bviews[i]);
      accel_handles[i] = vk_acceleration_structure_to_handle(&e->accels[i]);
   }
   VkWriteDescriptorSetAccelerationStructureKHR accel_ext = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 4,
      .pAccelerationStructures = accel_handles,
   };

   for (unsigned b = 0; b < ARRAY_SIZE(bindings); ++b) {
      const VkDescriptorType type = bindings[b].descriptorType;
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = hk_descriptor_set_to_handle(&set_batch),
         .dstBinding = b,
         .dstArrayElement = 0,
         .descriptorCount = 4,
         .descriptorType = type,
      };
      if (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
         w.pTexelBufferView = bview_handles;
      } else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                 type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
         w.pBufferInfo = buf_infos;
      } else if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
         w.pNext = &accel_ext;
      } else {
         w.pImageInfo = img_infos;
      }

      reset_flushes();
      hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
      assert(flush_count == 1);
      assert(flushes[0].start == buf_batch + layout->binding[b].offset);
      assert(flushes[0].size == (size_t)4 * layout->binding[b].stride);

      /* Compare byte-for-byte with four individual single-element updates. */
      for (unsigned j = 0; j < 4; ++j) {
         VkWriteDescriptorSet single = w;
         VkWriteDescriptorSetAccelerationStructureKHR single_accel = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &accel_handles[j],
         };
         single.dstSet = hk_descriptor_set_to_handle(&set_single);
         single.dstArrayElement = j;
         single.descriptorCount = 1;
         if (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
             type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
            single.pTexelBufferView = &bview_handles[j];
         else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                  type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            single.pBufferInfo = &buf_infos[j];
         else if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
            single.pNext = &single_accel;
         else
            single.pImageInfo = &img_infos[j];
         hk_UpdateDescriptorSets(dev, 1, &single, 0, NULL);
      }

      /* Also verify template update produces 1 flush and identical bytes. */
      struct vk_descriptor_update_template *tmpl =
         calloc(1, sizeof(*tmpl) + sizeof(tmpl->entries[0]));
      assert(tmpl);
      vk_object_base_init(&e->dev.vk, &tmpl->base,
                          VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE);
      tmpl->entry_count = 1;
      tmpl->entries[0] = (struct vk_descriptor_template_entry){
         .type = type,
         .binding = b,
         .array_element = 0,
         .array_count = 4,
         .offset = 0,
         .stride = (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                    type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
                      ? sizeof(VkBufferView)
                   : (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                      type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                      ? sizeof(VkDescriptorBufferInfo)
                   : (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                      ? sizeof(VkAccelerationStructureKHR)
                      : sizeof(VkDescriptorImageInfo),
      };
      const void *tmpl_data =
         (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
            ? (const void *)bview_handles
         : (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            ? (const void *)buf_infos
         : (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
            ? (const void *)accel_handles
            : (const void *)img_infos;

      reset_flushes();
      hk_UpdateDescriptorSetWithTemplate(
         dev, hk_descriptor_set_to_handle(&set_tmpl),
         vk_descriptor_update_template_to_handle(tmpl), tmpl_data);
      assert(flush_count == 1);
      assert(flushes[0].start == buf_tmpl + layout->binding[b].offset);
      assert(flushes[0].size == (size_t)4 * layout->binding[b].stride);
      vk_object_base_finish(&tmpl->base);
      free(tmpl);
   }

   assert(memcmp(buf_batch, buf_single, set_size) == 0);
   assert(memcmp(buf_tmpl, buf_single, set_size) == 0);
   for (unsigned i = 0; i < 64; ++i) {
      assert(buf_batch[set_size + i] == 0xa5);
      assert(buf_single[set_size + i] == 0xa5);
      assert(buf_tmpl[set_size + i] == 0xa5);
   }

   /* Push descriptor updates must produce zero cache cleans and identical bytes. */
   struct hk_descriptor_set_layout *push_layout = create_layout(
      e, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, bindings,
      ARRAY_SIZE(bindings), NULL);
   struct hk_push_descriptor_set push_set;
   memset(&push_set, 0, sizeof(push_set));
   VkWriteDescriptorSet push_writes[ARRAY_SIZE(bindings)];
   for (unsigned b = 0; b < ARRAY_SIZE(bindings); ++b) {
      const VkDescriptorType type = bindings[b].descriptorType;
      push_writes[b] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = b,
         .dstArrayElement = 0,
         .descriptorCount = 4,
         .descriptorType = type,
      };
      if (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
         push_writes[b].pTexelBufferView = bview_handles;
      else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
               type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
         push_writes[b].pBufferInfo = buf_infos;
      else if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
         push_writes[b].pNext = &accel_ext;
      else
         push_writes[b].pImageInfo = img_infos;
   }
   reset_flushes();
   assert(hk_push_descriptor_set_update(&push_set, push_layout,
                                        ARRAY_SIZE(push_writes), push_writes));
   assert(flush_count == 0);
   assert(memcmp(push_set.data, buf_single,
                 push_layout->non_variable_descriptor_buffer_size) == 0);

   /* vkGetDescriptorEXT must produce zero cache cleans and matching bytes. */
   uint8_t ext_buf[128];
   reset_flushes();
   VkDescriptorGetInfoEXT get_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
      .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .data.pSampledImage = &img_infos[0],
   };
   hk_GetDescriptorEXT(dev, &get_info, sizeof(ext_buf), ext_buf);
   assert(flush_count == 0);
   assert(memcmp(ext_buf, buf_single + layout->binding[0].offset,
                 layout->binding[0].stride) == 0);

   vk_descriptor_set_layout_unref(&e->dev.vk, &push_layout->vk);
   vk_descriptor_set_layout_unref(&e->dev.vk, &layout->vk);
   free(buf_batch);
   free(buf_single);
   free(buf_tmpl);
   destroy_env(e);
}

static void
test_mutable_and_ycbcr_gap_safety(void)
{
   struct test_env *e = create_env(AGX_CHIP_G17P, false);
   VkDevice dev = hk_device_to_handle(&e->dev);

   const VkDescriptorType mutable_types[] = {
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
   };
   const VkMutableDescriptorTypeListEXT mutable_lists[] = {
      {ARRAY_SIZE(mutable_types), mutable_types},
      {0, NULL},
   };
   const VkMutableDescriptorTypeCreateInfoEXT mutable_info = {
      .sType = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
      .mutableDescriptorTypeListCount = ARRAY_SIZE(mutable_lists),
      .pMutableDescriptorTypeLists = mutable_lists,
   };
   VkSampler ycbcr_samplers[4];
   for (unsigned i = 0; i < 4; ++i)
      ycbcr_samplers[i] = hk_sampler_to_handle(&e->samplers_ycbcr[i]);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_MUTABLE_EXT, 4, VK_SHADER_STAGE_ALL, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_ALL,
       ycbcr_samplers},
   };
   struct hk_descriptor_set_layout *layout =
      create_layout(e, 0, bindings, ARRAY_SIZE(bindings), &mutable_info);

   assert(layout->binding[0].stride == HK_MAX_DESCRIPTOR_SIZE);
   const uint32_t sampled_native =
      hk_sampled_image_descriptor_layout(layout->descriptor_generation).stride;
   assert(layout->binding[1].stride == 3 * sampled_native);

   const uint32_t set_size = align64(layout->non_variable_descriptor_buffer_size,
                                     HK_MIN_UBO_ALIGNMENT);
   uint8_t *buf = calloc(1, set_size);
   assert(buf);
   struct hk_descriptor_set set;
   init_set(&set, layout, buf, set_size, true);

   VkDescriptorImageInfo img_1p[4], img_3p[4];
   VkDescriptorBufferInfo buf_infos[4], null_buf_infos[4];
   VkBufferView bviews[4];
   VkAccelerationStructureKHR accels[4];
   for (unsigned i = 0; i < 4; ++i) {
      img_1p[i] = (VkDescriptorImageInfo){
         .imageView = hk_image_view_to_handle(&e->views_1p[i]),
      };
      img_3p[i] = (VkDescriptorImageInfo){
         .imageView = hk_image_view_to_handle(&e->views_3p[i]),
      };
      buf_infos[i] = (VkDescriptorBufferInfo){
         .buffer = hk_buffer_to_handle(&e->buffers[i]),
         .offset = 0,
         .range = 256,
      };
      null_buf_infos[i] = (VkDescriptorBufferInfo){
         .buffer = VK_NULL_HANDLE,
         .offset = 0,
         .range = VK_WHOLE_SIZE,
      };
      bviews[i] = hk_buffer_view_to_handle(&e->bviews[i]);
      accels[i] = vk_acceleration_structure_to_handle(&e->accels[i]);
   }

   /* Mutable SAMPLED_IMAGE (72B in 96B stride): 4 separate 72B flushes. */
   VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = hk_descriptor_set_to_handle(&set),
      .dstBinding = 0,
      .descriptorCount = 4,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = img_1p,
   };
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 4);
   for (unsigned i = 0; i < 4; ++i) {
      assert(flushes[i].start == buf + layout->binding[0].offset + i * 96);
      assert(flushes[i].size == 72);
   }

   /* Mutable non-null UNIFORM_BUFFER (16B in 96B stride): 4 separate 16B flushes. */
   w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
   w.pImageInfo = NULL;
   w.pBufferInfo = buf_infos;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 4);
   for (unsigned i = 0; i < 4; ++i) {
      assert(flushes[i].start == buf + layout->binding[0].offset + i * 96);
      assert(flushes[i].size == sizeof(struct hk_buffer_address));
   }

   /* Mutable null UNIFORM_BUFFER on G17P zeroes full 96B slot: 1 contiguous flush. */
   memset(buf, 0xdd, set_size);
   w.pBufferInfo = null_buf_infos;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 1);
   assert(flushes[0].start == buf + layout->binding[0].offset);
   assert(flushes[0].size == 4 * 96);

   /* Mutable ACCELERATION_STRUCTURE_KHR (8B in 96B stride): 4 separate 8B flushes. */
   VkWriteDescriptorSetAccelerationStructureKHR accel_ext = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 4,
      .pAccelerationStructures = accels,
   };
   w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
   w.pBufferInfo = NULL;
   w.pNext = &accel_ext;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 4);
   for (unsigned i = 0; i < 4; ++i) {
      assert(flushes[i].start == buf + layout->binding[0].offset + i * 96);
      assert(flushes[i].size == sizeof(uint64_t));
   }

   /* Mutable STORAGE_IMAGE writes full 96B stride: 1 contiguous flush. */
   w.pNext = NULL;
   w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   w.pImageInfo = img_1p;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 1);
   assert(flushes[0].start == buf + layout->binding[0].offset);
   assert(flushes[0].size == 4 * 96);

   /* Mutable UNIFORM_TEXEL_BUFFER writes full 96B stride: 1 contiguous flush. */
   w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
   w.pImageInfo = NULL;
   w.pTexelBufferView = bviews;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 1);
   assert(flushes[0].start == buf + layout->binding[0].offset);
   assert(flushes[0].size == 4 * 96);

   /* 3-plane YCbCr binding with 1-plane views (72B in 216B stride): 4 flushes. */
   w.dstBinding = 1;
   w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   w.pTexelBufferView = NULL;
   w.pImageInfo = img_1p;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 4);
   for (unsigned i = 0; i < 4; ++i) {
      assert(flushes[i].start ==
             buf + layout->binding[1].offset + i * 3 * sampled_native);
      assert(flushes[i].size == sampled_native);
   }

   /* 3-plane YCbCr binding with 3-plane views (216B in 216B stride): 1 flush. */
   w.pImageInfo = img_3p;
   reset_flushes();
   hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);
   assert(flush_count == 1);
   assert(flushes[0].start == buf + layout->binding[1].offset);
   assert(flushes[0].size == (size_t)4 * 3 * sampled_native);

   vk_descriptor_set_layout_unref(&e->dev.vk, &layout->vk);
   free(buf);
   destroy_env(e);
}

static void
test_allocate_immutable_samplers_single_flush(void)
{
   struct test_env *e = create_env(AGX_CHIP_G17P, false);
   VkDevice dev = hk_device_to_handle(&e->dev);

   VkSampler imm[8];
   for (unsigned i = 0; i < 8; ++i)
      imm[i] = hk_sampler_to_handle(&e->samplers_1p[i % 4]);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_SAMPLER, 8, VK_SHADER_STAGE_ALL, imm},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8, VK_SHADER_STAGE_ALL, imm},
   };
   struct hk_descriptor_set_layout *layout =
      create_layout(e, 0, bindings, ARRAY_SIZE(bindings), NULL);

   uint8_t *pool_mem = malloc(4096);
   assert(pool_mem);
   memset(pool_mem, 0xee, 4096);

   struct agx_va va = {.addr = UINT64_C(0x40000000), .size_B = 4096};
   struct agx_bo bo = {.va = &va, .size = 4096};
   struct hk_descriptor_pool pool;
   memset(&pool, 0, sizeof(pool));
   vk_object_base_init(&e->dev.vk, &pool.base, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   list_inithead(&pool.sets);
   pool.bo = &bo;
   pool.mapped_ptr = pool_mem;
   pool.cpu_cached = true;
   util_vma_heap_init(&pool.heap, va.addr, bo.size);

   VkDescriptorSetLayout layout_handle = hk_descriptor_set_layout_to_handle(layout);
   const VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = hk_descriptor_pool_to_handle(&pool),
      .descriptorSetCount = 1,
      .pSetLayouts = &layout_handle,
   };
   VkDescriptorSet set_handle = VK_NULL_HANDLE;

   reset_flushes();
   assert(hk_AllocateDescriptorSets(dev, &alloc_info, &set_handle) == VK_SUCCESS);
   struct hk_descriptor_set *set = hk_descriptor_set_from_handle(set_handle);

   /* Zeroing + 16 immutable samplers must publish in one whole-set clean. */
   assert(flush_count == 1);
   assert(flushes[0].start == set->mapped_ptr);
   assert(flushes[0].size == set->size);

   /* Verify the initialized sampler bytes match a direct SAMPLER write. */
   const uint32_t stride = layout->binding[0].stride;
   for (unsigned i = 0; i < 8; ++i) {
      const struct hk_sampled_image_descriptor *s0 =
         (const void *)((const uint8_t *)set->mapped_ptr +
                        layout->binding[0].offset + i * stride);
      const struct hk_sampled_image_descriptor *s1 =
         (const void *)((const uint8_t *)set->mapped_ptr +
                        layout->binding[1].offset + i * stride);
      assert(memcmp(s0, s1, sizeof(*s0)) == 0);
      assert(s0->sw.sampler_index ==
             hk_sampler_descriptor_index(layout, e->rc_samplers[i % 4].index));
      assert(s0->sw.clamp_0_sampler_index_or_negative == -1);
      assert(s0->sw.lod_bias_fp16 == e->samplers_1p[i % 4].lod_bias_fp16);
   }

   hk_FreeDescriptorSets(dev, hk_descriptor_pool_to_handle(&pool), 1,
                         &set_handle);
   util_vma_heap_finish(&pool.heap);
   vk_object_base_finish(&pool.base);
   vk_descriptor_set_layout_unref(&e->dev.vk, &layout->vk);
   free(pool_mem);
   destroy_env(e);
}

/* Reference single-element encoders matching aurora-wip base 8e288e5db so we
 * verify the rewritten encoders produce identical descriptor bytes across
 * AGX2, Gen9, and Gen10 (not merely self-consistency of the new code). */
static uint32_t
ref_8e288e5db_border_swizzle_word(const struct hk_image_view *view)
{
   const VkComponentSwizzle swizzle[4] = {view->vk.swizzle.r, view->vk.swizzle.g,
                                          view->vk.swizzle.b, view->vk.swizzle.a};
   uint32_t word = BITFIELD_BIT(31);
   for (unsigned c = 0; c < 4; ++c) {
      const unsigned code = swizzle[c] == VK_COMPONENT_SWIZZLE_ZERO ? 4
                            : swizzle[c] == VK_COMPONENT_SWIZZLE_ONE ? 5
                            : swizzle[c] - VK_COMPONENT_SWIZZLE_R;
      word |= code << (3 * c);
   }
   return word;
}

static void
ref_8e288e5db_write_sampled_one(const struct hk_descriptor_set_layout *layout,
                                uint32_t binding, uint32_t elem,
                                const VkDescriptorImageInfo *info,
                                VkDescriptorType descriptor_type,
                                uint8_t *dst, uint32_t dst_size)
{
   const unsigned generation = layout->descriptor_generation;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(generation);
   const struct hk_descriptor_set_binding_layout *binding_layout =
      &layout->binding[binding];

   struct hk_sampled_image_descriptor desc[3] = {};
   for (unsigned i = 0; i < 3; ++i) {
      if (generation == 9) {
         desc[i].sw.sampler_index = hk_sampler_descriptor_index(layout, 0);
         desc[i].sw.clamp_0_sampler_index_or_negative = -1;
      }
   }

   uint8_t plane_count = 1;
   uint8_t view_planes = 0;

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);
      plane_count = view->plane_count;
      view_planes = plane_count;
      for (uint8_t plane = 0; plane < plane_count; ++plane) {
         if (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
            desc[plane].tex = view->planes[plane].ia;
            desc[plane].sw.sampled_heap_index =
               view->planes[plane].input_heap_index;
         } else {
            desc[plane].tex = view->planes[plane].sampled;
            desc[plane].sw.sampled_heap_index =
               view->planes[plane].sampled_heap_index;
         }
         desc[plane].sw.min_lod_fp16 = view->sampled_min_lod_fp16;
         desc[plane].sw.min_lod_uint16 = view->sampled_min_lod_uint16;
         if (generation == 9 &&
             (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
              (descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT &&
               view->plane_count == 1 && !view->g17p_uscaled_normalized)))
            desc[plane].sw.border[0] = ref_8e288e5db_border_swizzle_word(view);
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      const struct hk_sampler *sampler =
         binding_layout->immutable_samplers
            ? binding_layout->immutable_samplers[elem]
            : hk_sampler_from_handle(info->sampler);
      plane_count = MAX2(plane_count, sampler->plane_count);
      for (uint8_t plane = 0; plane < plane_count; ++plane) {
         uint8_t sp = MIN2(plane, sampler->plane_count - 1);
         desc[plane].sw.sampler_index = hk_sampler_descriptor_index(
            layout, sampler->planes[sp].hw->index);
         if (generation == 9) {
            desc[plane].sw.g17p.unbiased_sampler_index =
               sampler->planes[sp].hw->index;
            desc[plane].sw.g17p.unbiased_clamp_0_sampler_index = 0;
            desc[plane].sw.g17p.lod_bias = sampler->mip_lod_bias;
         } else {
            desc[plane].sw.sampler = sampler->planes[sp].hw->key;
         }
         desc[plane].sw.lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].sw.clamp_0_sampler_index_or_negative = -1;
      }
   }

   for (unsigned plane = view_planes; plane < plane_count; ++plane)
      hk_set_null_native_texture(generation, &desc[plane].tex);

   const size_t desc_size = (size_t)native.stride * plane_count;
   assert(desc_size <= dst_size);
   uint8_t packed[3 * HK_MAX_DESCRIPTOR_SIZE];
   memset(packed, 0, desc_size);
   for (unsigned plane = 0; plane < plane_count; ++plane) {
      uint8_t *p = packed + plane * native.stride;
      memcpy(p + native.texture, &desc[plane].tex,
             hk_texture_descriptor_size(generation));
      memcpy(p + native.sampler, &desc[plane].sw, sizeof(desc[plane].sw));
   }
   memcpy(dst, packed, desc_size);
}

static void
ref_8e288e5db_write_storage_one(const struct hk_descriptor_set_layout *layout,
                                uint32_t binding,
                                const VkDescriptorImageInfo *info,
                                uint8_t *dst, uint32_t dst_size)
{
   const unsigned generation = layout->descriptor_generation;
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   const uint32_t stride = layout->binding[binding].stride;
   const uint32_t write_size = MIN2(stride, HK_MAX_DESCRIPTOR_SIZE);
   assert(write_size <= dst_size);

   struct hk_storage_image_descriptor desc = {};
   uint32_t store_window[4] = {0};
   bool supported = false;
   bool texture_supported = false;

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_image_view, view, info->imageView);
      const bool native_basic =
         hk_native_pbe_hardware_supported(generation) &&
         view->plane_count == 1 &&
         view->vk.image->samples == VK_SAMPLE_COUNT_1_BIT &&
         view->planes[0].storage_pbe_valid;
      supported = hk_native_pbe_supported(generation) || native_basic;
      texture_supported =
         supported || (hk_descriptor_generation_is_agx3(generation) &&
                       view->plane_count == 1);
      if (texture_supported) {
         desc.tex = view->planes[0].ro_storage;
         desc.texture_heap_index = view->planes[0].storage_heap_index;
      }
      if (supported) {
         desc.pbe = view->planes[0].storage;
         desc.atomic_sw = view->planes[0].storage_atomic_sw;
         memcpy(store_window, view->planes[0].storage_window,
                sizeof(store_window));
      }
   }

   if (!texture_supported)
      hk_set_null_native_texture(generation, &desc.tex);
   if (!supported)
      hk_set_null_native_pbe(generation, &desc.pbe);

   uint8_t packed[HK_MAX_DESCRIPTOR_SIZE] = {0};
   memcpy(packed + native.texture, &desc.tex,
          hk_texture_descriptor_size(generation));
   memcpy(packed + native.pbe, &desc.pbe, hk_pbe_descriptor_size(generation));
   if (stride >= AGX_G17P_STORAGE_HEAP_INDEX_OFFSET + sizeof(uint64_t))
      memcpy(packed + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET,
             &desc.texture_heap_index, sizeof(desc.texture_heap_index));
   if (stride >= AGX_G17P_IMAGE_ATOMIC_SW_OFFSET + sizeof(desc.atomic_sw))
      memcpy(packed + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET, &desc.atomic_sw,
             sizeof(desc.atomic_sw));
   if (stride >= HK_M5_STORE_WINDOW_OFFSET + HK_M5_STORE_WINDOW_SIZE)
      memcpy(packed + HK_M5_STORE_WINDOW_OFFSET, store_window,
             sizeof(store_window));
   memcpy(dst, packed, write_size);
}

static void
ref_8e288e5db_write_bview_one(const struct hk_descriptor_set_layout *layout,
                              uint32_t binding, VkBufferView bview,
                              bool storage, uint8_t *dst, uint32_t dst_size)
{
   const unsigned generation = layout->descriptor_generation;
   const struct hk_storage_image_descriptor_layout native =
      hk_storage_image_descriptor_layout(generation);
   const uint32_t stride = layout->binding[binding].stride;
   const uint32_t write_size = MIN2(stride, HK_MAX_DESCRIPTOR_SIZE);
   assert(write_size <= dst_size);

   struct hk_storage_image_descriptor desc = {};
   uint32_t heap_index = 0;
   if ((!storage || hk_descriptor_generation_is_agx3(generation) ||
        hk_native_pbe_supported(generation)) &&
       bview != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(hk_buffer_view, view, bview);
      desc.tex = view->tex;
      if (storage)
         desc.pbe = view->pbe;
      heap_index = view->sampled_heap_index;
   } else {
      if (generation != 9)
         hk_set_null_native_texture(generation, &desc.tex);
      hk_set_null_native_pbe(generation, &desc.pbe);
   }

   uint8_t packed[HK_MAX_DESCRIPTOR_SIZE] = {0};
   memcpy(packed + native.texture, &desc.tex,
          hk_texture_descriptor_size(generation));
   memcpy(packed + native.pbe, &desc.pbe, hk_pbe_descriptor_size(generation));
   if (heap_index &&
       stride >= AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET + sizeof(heap_index)) {
      memcpy(packed + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET, &heap_index,
             sizeof(heap_index));
      memcpy(packed + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET, &heap_index,
             sizeof(heap_index));
   }
   if (heap_index && stride >= AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE) {
      VK_FROM_HANDLE(hk_buffer_view, view, bview);
      memcpy(packed + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET,
             &view->buffer_metadata, sizeof(view->buffer_metadata));
   }
   memcpy(dst, packed, write_size);
}

static void
test_base_encoder_parity_vs_8e288e5db(enum agx_chip chip, bool m5_native_as)
{
   struct test_env *e = create_env(chip, m5_native_as);
   VkDevice dev = hk_device_to_handle(&e->dev);

   /* Exercise storage_pbe_valid = false on index 2 and VK_NULL_HANDLE on 3. */
   e->views_1p[2].planes[0].storage_pbe_valid = false;

   VkSampler imm_1p[4];
   for (unsigned i = 0; i < 4; ++i)
      imm_1p[i] = hk_sampler_to_handle(&e->samplers_1p[i]);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4, VK_SHADER_STAGE_ALL, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_ALL, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, VK_SHADER_STAGE_ALL, imm_1p},
      {3, VK_DESCRIPTOR_TYPE_SAMPLER, 4, VK_SHADER_STAGE_ALL, NULL},
      {4, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 4, VK_SHADER_STAGE_ALL, NULL},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4, VK_SHADER_STAGE_ALL, NULL},
      {6, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, VK_SHADER_STAGE_ALL, NULL},
      {10, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4, VK_SHADER_STAGE_ALL, NULL},
   };
   struct hk_descriptor_set_layout *layout =
      create_layout(e, 0, bindings, ARRAY_SIZE(bindings), NULL);

   const uint32_t set_size = align64(layout->non_variable_descriptor_buffer_size,
                                     HK_MIN_UBO_ALIGNMENT);
   uint8_t *buf_new = calloc(1, set_size);
   uint8_t *buf_ref = calloc(1, set_size);
   assert(buf_new && buf_ref);

   struct hk_descriptor_set set_new;
   init_set(&set_new, layout, buf_new, set_size, true);

   VkDescriptorImageInfo img_infos[4], stor_infos[4];
   VkDescriptorBufferInfo buf_infos[4];
   VkBufferView bview_handles[4];
   VkAccelerationStructureKHR accel_handles[4];
   for (unsigned i = 0; i < 4; ++i) {
      img_infos[i] = (VkDescriptorImageInfo){
         .sampler = hk_sampler_to_handle(&e->samplers_1p[i]),
         .imageView = hk_image_view_to_handle(&e->views_1p[i]),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      stor_infos[i] = (VkDescriptorImageInfo){
         .imageView = (i == 3) ? VK_NULL_HANDLE
                               : hk_image_view_to_handle(&e->views_1p[i]),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
      buf_infos[i] = (VkDescriptorBufferInfo){
         .buffer = hk_buffer_to_handle(&e->buffers[i]),
         .offset = i * 64,
         .range = 1024 + i * 64,
      };
      bview_handles[i] =
         (i == 3) ? VK_NULL_HANDLE : hk_buffer_view_to_handle(&e->bviews[i]);
      accel_handles[i] = vk_acceleration_structure_to_handle(&e->accels[i]);
   }
   VkWriteDescriptorSetAccelerationStructureKHR accel_ext = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 4,
      .pAccelerationStructures = accel_handles,
   };

   for (unsigned b = 0; b < ARRAY_SIZE(bindings); ++b) {
      const VkDescriptorType type = bindings[b].descriptorType;
      const uint32_t stride = layout->binding[b].stride;
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = hk_descriptor_set_to_handle(&set_new),
         .dstBinding = b,
         .dstArrayElement = 0,
         .descriptorCount = 4,
         .descriptorType = type,
      };
      if (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
         w.pTexelBufferView = bview_handles;
      else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
               type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
         w.pBufferInfo = buf_infos;
      else if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
         w.pNext = &accel_ext;
      else if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
         w.pImageInfo = stor_infos;
      else
         w.pImageInfo = img_infos;

      hk_UpdateDescriptorSets(dev, 1, &w, 0, NULL);

      for (unsigned j = 0; j < 4; ++j) {
         uint8_t *ref_dst = buf_ref + layout->binding[b].offset + j * stride;
         const uint32_t rem = set_size - (uint32_t)(ref_dst - buf_ref);
         if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            ref_8e288e5db_write_storage_one(layout, b, &stor_infos[j], ref_dst,
                                            rem);
         } else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                    type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            ref_8e288e5db_write_bview_one(
               layout, b, bview_handles[j],
               type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, ref_dst, rem);
         } else if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                    type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            const struct hk_buffer_address ba = {
               .base_addr = e->buffers[j].vk.device_address + buf_infos[j].offset,
               .size = buf_infos[j].range,
               .sparse_map_page = 0,
            };
            memcpy(ref_dst, &ba, sizeof(ba));
         } else if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
            uint64_t addr = e->accels[j].addr;
            if (m5_native_as)
               addr = ALIGN_POT(addr, UINT64_C(1024));
            memcpy(ref_dst, &addr, sizeof(addr));
         } else {
            ref_8e288e5db_write_sampled_one(layout, b, j, &img_infos[j], type,
                                            ref_dst, rem);
         }
      }
   }

   assert(memcmp(buf_new, buf_ref, set_size) == 0);

   /* Also verify immutable combined-image-sampler allocation (info == NULL). */
   uint8_t *pool_mem = malloc(set_size + 4096);
   assert(pool_mem);
   memset(pool_mem, 0x00, set_size + 4096);
   struct agx_va va = {.addr = UINT64_C(0x50000000), .size_B = set_size + 4096};
   struct agx_bo bo = {.va = &va, .size = set_size + 4096};
   struct hk_descriptor_pool pool = {0};
   vk_object_base_init(&e->dev.vk, &pool.base, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   list_inithead(&pool.sets);
   pool.bo = &bo;
   pool.mapped_ptr = pool_mem;
   pool.cpu_cached = true;
   util_vma_heap_init(&pool.heap, va.addr, bo.size);

   VkDescriptorSetLayout lh = hk_descriptor_set_layout_to_handle(layout);
   VkDescriptorSet alloc_handle = VK_NULL_HANDLE;
   const VkDescriptorSetAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = hk_descriptor_pool_to_handle(&pool),
      .descriptorSetCount = 1,
      .pSetLayouts = &lh,
   };
   assert(hk_AllocateDescriptorSets(dev, &alloc_info, &alloc_handle) == VK_SUCCESS);
   struct hk_descriptor_set *alloc_set =
      hk_descriptor_set_from_handle(alloc_handle);

   const uint32_t imm_stride = layout->binding[2].stride;
   uint8_t *imm_ref = calloc(1, 4 * imm_stride);
   assert(imm_ref);
   for (unsigned j = 0; j < 4; ++j) {
      ref_8e288e5db_write_sampled_one(
         layout, 2, j, NULL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         imm_ref + j * imm_stride, imm_stride);
   }
   assert(memcmp((const uint8_t *)alloc_set->mapped_ptr +
                    layout->binding[2].offset,
                 imm_ref, 4 * imm_stride) == 0);

   free(imm_ref);
   hk_FreeDescriptorSets(dev, hk_descriptor_pool_to_handle(&pool), 1,
                         &alloc_handle);
   util_vma_heap_finish(&pool.heap);
   vk_object_base_finish(&pool.base);
   free(pool_mem);
   vk_descriptor_set_layout_unref(&e->dev.vk, &layout->vk);
   free(buf_new);
   free(buf_ref);
   destroy_env(e);
}

int
main(void)
{
   test_contiguous_array_types(AGX_CHIP_G13G, false);
   test_contiguous_array_types(AGX_CHIP_G17P, false);
   test_contiguous_array_types(AGX_CHIP_G17C, true);
   test_base_encoder_parity_vs_8e288e5db(AGX_CHIP_G13G, false);
   test_base_encoder_parity_vs_8e288e5db(AGX_CHIP_G17P, false);
   test_base_encoder_parity_vs_8e288e5db(AGX_CHIP_G17C, true);
   test_mutable_and_ycbcr_gap_safety();
   test_allocate_immutable_samplers_single_flush();
   printf("PASS: batched descriptor array publication and 8e288e5db encoder parity across AGX2, Gen9, and Gen10\n");
   return 0;
}
