/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

/* Keep this header independent of Vulkan and generated GenXML headers.  Apart
 * from making the descriptor ABI easy to test on the host, this prevents a
 * recognised-but-disabled GPU generation from accidentally selecting the
 * legacy C struct layout.
 */
/* G13 uses Gen7, G14 uses Gen8, G15G/S/C/D, G16P/G/S/C, and G17P/A use
 * Gen9, while G17G/S/C uses Gen10.  G15P remains deliberately unknown. */
#define HK_TEXTURE_DESCRIPTOR_AGX2_SIZE 24
#define HK_TEXTURE_DESCRIPTOR_AGX3_SIZE 32
#define HK_PBE_DESCRIPTOR_AGX2_SIZE     24
#define HK_PBE_DESCRIPTOR_AGX3_SIZE     32
#define HK_SAMPLED_IMAGE_SW_SIZE        40
#define HK_G17P_STORAGE_SW_SIZE         16
#define HK_G17P_STORAGE_ALIGNMENT       32
#define HK_G17P_STORAGE_STRIDE          96

/* G17P imgwci indexes whole PBE records in the heap it addresses. */
#define HK_G17P_PBE_RECORD_SIZE         HK_PBE_DESCRIPTOR_AGX3_SIZE

/* The M5 and M5 Pro (G17G, G17S) store to a layer of an array as a 3D store
 * (agx3_pack_binary.c), and a 3D store addresses at most this many slices.
 * At mip level L, hk describes a view of N layers as N << L slices deep
 * (hk_image_view.c), so a level with N << L above the limit cannot be
 * described that way. Such a storage view instead gets a linear "store
 * window" PBE over its level, starting at its first layer, and this record in
 * the descriptor's spare bytes past the atomic word:
 *
 *    word 0: layer stride in elements; zero for every other descriptor
 *    word 1: log2 tile width [0:3], log2 tile height [4:7], in pixels;
 *            bit 31 (HK_M5_STORE_2D) instead marks a 2D PBE, see below
 *    word 2: level width [0:15], level height [16:31], in pixels
 *    word 3: layer count of the view (faces for a cube); zero for every
 *            descriptor without a record
 *
 * hk_nir_lower_descriptors() takes the record for stores when word 3 is
 * nonzero, and stores through the window at the texel's tiled element index
 * (the arithmetic image atomics use) unless word 1 has HK_M5_STORE_2D; it
 * takes it for atomics when word 0 is nonzero, updating the texel in memory.
 *
 * The packer admits a compressed layered PBE only as a one-layer 2D array,
 * never 3D, so a one-layer layered storage view of a compressed image (on
 * G17S, a single-layer 2D storage image, hk_image.c) cannot take the 3D
 * store. Its PBE is the image's compressed 2D PBE at the view's level, and
 * its record is word 0 zero, word 1 HK_M5_STORE_2D, word 2 the level size and
 * word 3 one: layered stores then store to that PBE at (x, y), and drop
 * layers other than 0. Atomics keep the native op, which the record's zero
 * word 0 selects: no compressed storage format has image atomics.
 */
#define HK_M5_STORE_WINDOW_OFFSET       80
#define HK_M5_STORE_WINDOW_SIZE         16
#define HK_M5_STORE_MAX_SLICES          2048
/* The window is a linear 2D PBE this many elements wide, as texel buffers. */
#define HK_M5_STORE_WINDOW_WIDTH        16384
/* Gen10 PBE heights are 15 bits wide. */
#define HK_M5_STORE_WINDOW_MAX_HEIGHT   32768
#define HK_M5_STORE_2D                  (1u << 31)

static_assert(HK_M5_STORE_WINDOW_OFFSET + HK_M5_STORE_WINDOW_SIZE <=
                 HK_G17P_STORAGE_STRIDE,
              "the store window record fits the storage descriptor stride");

/* Fill the store window record of a view of `layers` layers at a level of
 * width x height pixels with the given tile size, element size and layer
 * stride, whose level spans level_size_B bytes in each layer. Returns the
 * window height in rows, or zero if the window cannot address the view.
 */
static inline uint32_t
hk_pack_m5_store_window(uint64_t layer_stride_B, uint64_t level_size_B,
                        unsigned element_size_B, unsigned width_px,
                        unsigned height_px, unsigned tile_width_px,
                        unsigned tile_height_px, unsigned layers,
                        uint32_t out[4])
{
   if (!out || !element_size_B || !layers || !width_px || !height_px ||
       width_px > UINT16_MAX || height_px > UINT16_MAX ||
       (layer_stride_B % element_size_B) || !level_size_B ||
       tile_width_px == 0 || (tile_width_px & (tile_width_px - 1)) ||
       tile_height_px == 0 || (tile_height_px & (tile_height_px - 1)) ||
       tile_width_px > (1u << 15) || tile_height_px > (1u << 15))
      return 0;

   const uint64_t layer_stride_el = layer_stride_B / element_size_B;
   const uint64_t level_el =
      (level_size_B + element_size_B - 1) / element_size_B;
   const uint64_t span_el = (uint64_t)(layers - 1) * layer_stride_el + level_el;
   const uint64_t rows =
      (span_el + HK_M5_STORE_WINDOW_WIDTH - 1) / HK_M5_STORE_WINDOW_WIDTH;
   if (layer_stride_el == 0 || layer_stride_el > UINT32_MAX ||
       rows > HK_M5_STORE_WINDOW_MAX_HEIGHT)
      return 0;

   unsigned tile_width_log2 = 0, tile_height_log2 = 0;
   for (unsigned x = tile_width_px; x > 1; x >>= 1)
      ++tile_width_log2;
   for (unsigned x = tile_height_px; x > 1; x >>= 1)
      ++tile_height_log2;

   out[0] = (uint32_t)layer_stride_el;
   out[1] = tile_width_log2 | (tile_height_log2 << 4);
   out[2] = width_px | ((uint32_t)height_px << 16);
   out[3] = layers;
   return (uint32_t)rows;
}

/* Fill the record of a one-layer layered storage view whose PBE is 2D, at a
 * level of width x height pixels. Returns false if the record cannot hold
 * the level's size.
 */
static inline bool
hk_pack_m5_store_2d(unsigned width_px, unsigned height_px, uint32_t out[4])
{
   if (!out || !width_px || !height_px || width_px > UINT16_MAX ||
       height_px > UINT16_MAX)
      return false;

   out[0] = 0;
   out[1] = HK_M5_STORE_2D;
   out[2] = width_px | ((uint32_t)height_px << 16);
   out[3] = 1;
   return true;
}

static inline bool
hk_pack_g17p_image_atomic_sw(uint64_t level_offset_B, unsigned tile_width_px,
                             unsigned tile_height_px, uint64_t layer_stride_B,
                             unsigned samples, uint64_t *out)
{
   if (!out || (level_offset_B & 0x7f) || (layer_stride_B & 0x7f) ||
       tile_width_px == 0 || (tile_width_px & (tile_width_px - 1)) ||
       tile_height_px == 0 || (tile_height_px & (tile_height_px - 1)) ||
       (samples != 1 && samples != 2 && samples != 4 && samples != 8))
      return false;

   unsigned tile_width_log2 = 0, tile_height_log2 = 0, samples_log2 = 0;
   for (unsigned x = tile_width_px; x > 1; x >>= 1)
      ++tile_width_log2;
   for (unsigned x = tile_height_px; x > 1; x >>= 1)
      ++tile_height_log2;
   for (unsigned x = samples; x > 1; x >>= 1)
      ++samples_log2;

   const uint64_t level_offset = level_offset_B >> 7;
   const uint64_t layer_stride = layer_stride_B >> 7;
   if (level_offset >= (UINT64_C(1) << 27) ||
       layer_stride >= (UINT64_C(1) << 27) || tile_width_log2 >= 16 ||
       tile_height_log2 >= 16 || samples_log2 >= 4)
      return false;

   *out = level_offset | ((uint64_t)tile_width_log2 << 27) |
          ((uint64_t)tile_height_log2 << 31) | (layer_stride << 35) |
          ((uint64_t)samples_log2 << 62);
   return true;
}

struct hk_g17p_buffer_texture_metadata {
   uint32_t elements;
   uint32_t offset_el;
   uint64_t address;
   uint32_t rgb32;
   /* See AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET in agx_abi.h. */
   uint32_t sparse;
};

struct hk_sampled_image_descriptor_layout {
   uint8_t texture;
   uint8_t sampler;
   uint8_t sampler_index;
   uint8_t clamp_0_sampler_index;
   uint8_t lod_bias;
   uint8_t min_lod;
   uint8_t min_lod_uint;
   uint8_t border;
   uint8_t stride;
};

struct hk_storage_image_descriptor_layout {
   uint8_t texture;
   uint8_t pbe;
   uint8_t stride;
};

static inline bool
hk_descriptor_generation_is_agx2(unsigned generation)
{
   return generation == 7 || generation == 8;
}

static inline bool
hk_descriptor_generation_is_agx3(unsigned generation)
{
   return generation == 9 || generation == 10;
}

static inline unsigned
hk_texture_descriptor_size(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation)
             ? HK_TEXTURE_DESCRIPTOR_AGX2_SIZE
          : hk_descriptor_generation_is_agx3(generation)
             ? HK_TEXTURE_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline unsigned
hk_pbe_descriptor_size(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation)
             ? HK_PBE_DESCRIPTOR_AGX2_SIZE
          : hk_descriptor_generation_is_agx3(generation)
             ? HK_PBE_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline struct hk_sampled_image_descriptor_layout
hk_sampled_image_descriptor_layout(unsigned generation)
{
   const unsigned texture_size = hk_texture_descriptor_size(generation);

   if (!texture_size)
      return (struct hk_sampled_image_descriptor_layout){0};

   return (struct hk_sampled_image_descriptor_layout){
      .texture = 0,
      .sampler = texture_size,
      .sampler_index = texture_size + 8,
      .clamp_0_sampler_index = texture_size + 10,
      .lod_bias = texture_size + 12,
      .min_lod = texture_size + 14,
      .min_lod_uint = texture_size + 16,
      .border = texture_size + 20,
      .stride = texture_size + HK_SAMPLED_IMAGE_SW_SIZE,
   };
}

static inline struct hk_storage_image_descriptor_layout
hk_storage_image_descriptor_layout(unsigned generation)
{
   const unsigned texture_size = hk_texture_descriptor_size(generation);
   const unsigned pbe_size = hk_pbe_descriptor_size(generation);

   if (!texture_size || !pbe_size)
      return (struct hk_storage_image_descriptor_layout){0};

   return (struct hk_storage_image_descriptor_layout){
      .texture = 0,
      .pbe = texture_size,
      .stride = texture_size + pbe_size,
   };
}

/* AGX3 buffer descriptors have no recovered equivalent for AGX2's software
 * tail offset.  Keep those unavailable rather than aliasing unknown bits in a
 * native descriptor.  Multisample descriptor representation is recovered on
 * both AGX3 generations; this is not by itself an execution/capability gate.
 */
static inline bool
hk_native_texture_supported(unsigned generation, bool multisampled,
                            unsigned buffer_tail_offset)
{
   if (hk_descriptor_generation_is_agx2(generation))
      return true;

   if (!hk_descriptor_generation_is_agx3(generation))
      return false;

   return buffer_tail_offset == 0;
}

/* Return whether the sampled-texture descriptor has a recovered representation
 * for a sample count.  This is not a Vulkan capability predicate: the live
 * G17S corpus proves Gen10 creation/layout and descriptor bits but contains no
 * executed sampling command, so public 8x exposure remains gated.
 */
static inline bool
hk_native_texture_sample_count_representable(unsigned generation,
                                             unsigned samples)
{
   if (samples == 1)
      return hk_texture_descriptor_size(generation) != 0;

   if (hk_descriptor_generation_is_agx2(generation))
      return samples == 2 || samples == 4;

   if (generation == 9 || generation == 10)
      return samples == 2 || samples == 4 || samples == 8;

   return false;
}

/* Hardware PBE descriptors have an exact bounded Gen9/10 subset. */
static inline bool
hk_native_pbe_hardware_supported(unsigned generation)
{
   return hk_pbe_descriptor_size(generation) != 0;
}

/* Writable-image atomics, MSAA and buffers still rely on the AGX2 software
 * sideband decode. Keep those execution paths separate from ordinary native
 * PBE stores.
 */
static inline bool
hk_native_pbe_supported(unsigned generation)
{
   return hk_descriptor_generation_is_agx2(generation);
}
