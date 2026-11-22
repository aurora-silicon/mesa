/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#undef NDEBUG

#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_native_descriptor.h"
#include "asahi/lib/agx_pbe_agx3_core.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
test_generation_queries(void)
{
   assert(agx_descriptor_generation_is_agx2(AGX_DESCRIPTOR_GENERATION_7));
   assert(agx_descriptor_generation_is_agx2(AGX_DESCRIPTOR_GENERATION_8));
   assert(!agx_descriptor_generation_is_agx2(AGX_DESCRIPTOR_GENERATION_9));
   assert(!agx_descriptor_generation_is_agx2(AGX_DESCRIPTOR_GENERATION_10));
   assert(!agx_descriptor_generation_is_agx2(AGX_DESCRIPTOR_GENERATION_UNKNOWN));

   assert(!agx_descriptor_generation_is_agx3(AGX_DESCRIPTOR_GENERATION_7));
   assert(!agx_descriptor_generation_is_agx3(AGX_DESCRIPTOR_GENERATION_8));
   assert(agx_descriptor_generation_is_agx3(AGX_DESCRIPTOR_GENERATION_9));
   assert(agx_descriptor_generation_is_agx3(AGX_DESCRIPTOR_GENERATION_10));
   assert(!agx_descriptor_generation_is_agx3(AGX_DESCRIPTOR_GENERATION_UNKNOWN));

   assert(agx_texture_descriptor_size(AGX_DESCRIPTOR_GENERATION_7) == 24);
   assert(agx_texture_descriptor_size(AGX_DESCRIPTOR_GENERATION_8) == 24);
   assert(agx_texture_descriptor_size(AGX_DESCRIPTOR_GENERATION_9) == 32);
   assert(agx_texture_descriptor_size(AGX_DESCRIPTOR_GENERATION_10) == 32);
   assert(agx_texture_descriptor_size(AGX_DESCRIPTOR_GENERATION_UNKNOWN) == 0);

   assert(agx_pbe_descriptor_size(AGX_DESCRIPTOR_GENERATION_8) == 24);
   assert(agx_pbe_descriptor_size(AGX_DESCRIPTOR_GENERATION_9) == 32);
   assert(agx_pbe_descriptor_size(AGX_DESCRIPTOR_GENERATION_10) == 32);
   assert(agx_pbe_descriptor_size(AGX_DESCRIPTOR_GENERATION_UNKNOWN) == 0);

   const struct agx_storage_descriptor_layout l8 =
      agx_storage_descriptor_layout(AGX_DESCRIPTOR_GENERATION_8);
   assert(l8.texture == 0 && l8.pbe == 24 && l8.stride == 48);
   const struct agx_storage_descriptor_layout l9 =
      agx_storage_descriptor_layout(AGX_DESCRIPTOR_GENERATION_9);
   assert(l9.texture == 0 && l9.pbe == 32 && l9.stride == 64);
   const struct agx_storage_descriptor_layout l0 =
      agx_storage_descriptor_layout(AGX_DESCRIPTOR_GENERATION_UNKNOWN);
   assert(l0.texture == 0 && l0.pbe == 0 && l0.stride == 0);

   enum agx_sample_count sc;
   assert(!agx_translate_native_texture_sample_count(
      AGX_DESCRIPTOR_GENERATION_9, 1, &sc));
   assert(agx_translate_native_texture_sample_count(
             AGX_DESCRIPTOR_GENERATION_9, 2, &sc) &&
          sc == AGX_SAMPLE_COUNT_2);
   assert(agx_translate_native_texture_sample_count(
             AGX_DESCRIPTOR_GENERATION_9, 4, &sc) &&
          sc == AGX_SAMPLE_COUNT_4);
   assert(agx_translate_native_texture_sample_count(
             AGX_DESCRIPTOR_GENERATION_9, 8, &sc) &&
          sc == AGX_SAMPLE_COUNT_8);
   assert(!agx_translate_native_texture_sample_count(
      AGX_DESCRIPTOR_GENERATION_8, 8, &sc));
   assert(agx_translate_native_texture_sample_count(
             AGX_DESCRIPTOR_GENERATION_10, 8, &sc) &&
          sc == AGX_SAMPLE_COUNT_8);
}

static void
test_texture_reconciled_divergences(void)
{
   struct AGX_TEXTURE base = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 64,
      .height = 64,
      .depth = 1,
      .address = UINT64_C(0x10000080000),
   };
   struct agx_native_texture_descriptor out;

   /* Divergence 1: Gen9 uncompressed extended textures accept either zero
    * (Borealis) or an explicit 16B-aligned 48-bit range-end VA (Gallium)
    * in acceleration_buffer with no flags needed, while rejecting misaligned
    * or out-of-range VAs.
    */
   struct AGX_TEXTURE uncomp_mip = base;
   uncomp_mip.unk_mipmapped = true;
   uncomp_mip.last_level = 3;
   uncomp_mip.acceleration_buffer = 0;
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                  &uncomp_mip));
   struct AGX_TEXTURE_GEN9 decoded;
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)out.opaque, &decoded));
   assert(decoded.extended && !decoded.compressed &&
          decoded.acceleration_buffer == 0);

   uncomp_mip.acceleration_buffer = UINT64_C(0x10000090000);
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                  &uncomp_mip));
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)out.opaque, &decoded));
   assert(decoded.extended && !decoded.compressed &&
          decoded.acceleration_buffer == uncomp_mip.acceleration_buffer);

   uncomp_mip.acceleration_buffer = UINT64_C(0x10000090004);
   assert(!agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                   &uncomp_mip));

   /* Divergence 2: Gen9 stride/depth bitfield overflow guards and
    * AGX_LAYOUT_INTERCHANGE support.
    */
   struct AGX_TEXTURE overflow = base;
   overflow.depth = 0x10001u;
   assert(!agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                   &overflow));

   overflow = base;
   overflow.layout = AGX_LAYOUT_LINEAR;
   overflow.stride = 0x100000u; /* (stride >> 4) == 0x10000 > 0xffff */
   assert(!agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                   &overflow));

   struct AGX_TEXTURE interchange = base;
   interchange.layout = AGX_LAYOUT_INTERCHANGE;
   interchange.stride = 15u << 4;
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                  &interchange));
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)out.opaque, &decoded));
   assert(decoded.layout == AGX_LAYOUT_INTERCHANGE &&
          decoded.stride == interchange.stride);

   interchange.depth = 2;
   assert(!agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                   &interchange));
   interchange.depth = 1;
   interchange.last_level = 1;
   assert(!agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &out,
                                   &interchange));

   /* Divergence 3: Gen9 plain_2d word1[62] workaround covers both 2D and
    * 2D_MULTISAMPLED.
    */
   struct AGX_TEXTURE msaa = base;
   msaa.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   msaa.samples = AGX_SAMPLE_COUNT_4;
   struct agx_native_texture_descriptor ms_a, ms_b;
   msaa.page_aligned_layers = false;
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &ms_a, &msaa));
   msaa.page_aligned_layers = true;
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9, &ms_b, &msaa));
   assert(memcmp(&ms_a, &ms_b, sizeof(ms_a)) == 0);
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)ms_b.opaque,
                                  &decoded));
   assert(!decoded.page_aligned_layers);

   /* Divergence 4: Gen10 AGX_IMAGE_MODE_SPARSE remaps to 1 (16 KiB pages). */
   struct AGX_TEXTURE sparse = base;
   sparse.mode = AGX_IMAGE_MODE_SPARSE;
   assert(agx_pack_native_texture(AGX_DESCRIPTOR_GENERATION_10, &out, &sparse));
   struct AGX_TEXTURE_GEN10 decoded10;
   assert(AGX_TEXTURE_GEN10_unpack(NULL, (const uint8_t *)out.opaque,
                                   &decoded10));
   assert(decoded10.image_mode_enable && decoded10.image_mode == 1);

   /* Divergence 6: Null texture descriptor caching + AGX2 24B non-overrun. */
   struct agx_native_texture_descriptor null7, null9_first, null9_cached;
   memset(&null7, 0xa5, sizeof(null7));
   assert(agx_set_null_native_texture(AGX_DESCRIPTOR_GENERATION_7, &null7));
   assert(null7.opaque[6] == 0xa5a5a5a5 && null7.opaque[7] == 0xa5a5a5a5);

   /* Second call hits the atomic-once cached fast path and still preserves
    * bytes 24..31 on AGX2.
    */
   struct agx_native_texture_descriptor null7_cached;
   memset(&null7_cached, 0x5a, sizeof(null7_cached));
   assert(
      agx_set_null_native_texture(AGX_DESCRIPTOR_GENERATION_7, &null7_cached));
   assert(memcmp(&null7, &null7_cached, AGX_NATIVE_DESCRIPTOR_AGX2_SIZE) == 0);
   assert(null7_cached.opaque[6] == 0x5a5a5a5a &&
          null7_cached.opaque[7] == 0x5a5a5a5a);

   memset(&null9_first, 0xa5, sizeof(null9_first));
   assert(
      agx_set_null_native_texture(AGX_DESCRIPTOR_GENERATION_9, &null9_first));
   memset(&null9_cached, 0x5a, sizeof(null9_cached));
   assert(
      agx_set_null_native_texture(AGX_DESCRIPTOR_GENERATION_9, &null9_cached));
   assert(memcmp(&null9_first, &null9_cached, sizeof(null9_first)) == 0);
   assert(null9_cached.opaque[6] == 0 && null9_cached.opaque[7] == 0);
   assert(!agx_set_null_native_texture(AGX_DESCRIPTOR_GENERATION_UNKNOWN,
                                       &null9_cached));
}

static void
test_pbe_reconciled_profiles(void)
{
   /* Gen10 compressed 2x/4x MSAA admits only the verified single-layer 2D MSAA
    * formats (RGBA8, R32_UINT, RGBA16 F/UI) and refuses arbitrary formats and
    * compressed 2D MSAA arrays; G18G keeps its separate agx_pack_g18_native_pbe
    * encoder untouched.
    */
   struct AGX_PBE pbe = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 128,
      .height = 128,
      .buffer = UINT64_C(0x12345000),
      .compressed = true,
      .extended = true,
      .acceleration_buffer = UINT64_C(0x45678000),
      .samples = AGX_SAMPLE_COUNT_4,
   };
   struct agx_native_pbe_descriptor packed;
   assert(agx_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &packed, &pbe));

   /* Unverified compressed Gen10 MSAA formats and 2D MSAA arrays remain
    * refused on the generic Gen10 path; G18G keeps its own encoder path.
    */
   struct AGX_PBE msaa_arr = pbe;
   msaa_arr.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   msaa_arr.layers = 4;
   assert(!agx_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &packed,
                               &msaa_arr));
   assert(agx_pack_g18_native_pbe(&packed, &msaa_arr));
   msaa_arr.samples = AGX_SAMPLE_COUNT_8;
   assert(!agx_pack_g18_native_pbe(&packed, &msaa_arr));

   struct AGX_PBE unverified_fmt = pbe;
   unverified_fmt.channels = AGX_CHANNELS_R11G11B10;
   unverified_fmt.type = AGX_TEXTURE_TYPE_FLOAT;
   assert(!agx_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &packed,
                               &unverified_fmt));

   /* Gen10 admits sRGB stores and 16B-aligned uncompressed MSAA. */
   struct AGX_PBE srgb2d = pbe;
   srgb2d.dimension = AGX_TEXTURE_DIMENSION_2D;
   srgb2d.samples = AGX_SAMPLE_COUNT_2;
   srgb2d.compressed = false;
   srgb2d.extended = false;
   srgb2d.acceleration_buffer = 0;
   srgb2d.page_aligned_layers = false;
   srgb2d.srgb = true;
   assert(agx_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &packed, &srgb2d));

   /* Null PBE writes 24B on AGX2 (preserving tail) and 32B on AGX3. */
   memset(&packed, 0xa5, sizeof(packed));
   assert(agx_set_null_native_pbe(AGX_DESCRIPTOR_GENERATION_8, &packed));
   assert(packed.opaque[6] == 0xa5a5a5a5 && packed.opaque[7] == 0xa5a5a5a5);

   memset(&packed, 0xa5, sizeof(packed));
   assert(agx_set_null_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &packed));
   assert(packed.opaque[6] == 0 && packed.opaque[7] == 0);
   assert(agx_set_null_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &packed));
}

static void
test_samplers(void)
{
   struct agx_native_sampler_descriptor txf;
   static const uint8_t expected_hal200[32] = {
      0x00, 0x00, 0x0e, 0x68, 0x1b, 0x00, 0x00, 0x00,
   };
   static const uint8_t expected_gen5[32] = {
      0x00, 0x00, 0x0f, 0x68, 0x1b, 0x04, 0x00, 0x00,
   };
   assert(agx_pack_native_txf_sampler(AGX_CHIP_G13G, &txf));
   assert(memcmp(&txf, expected_hal200, sizeof(expected_hal200)) == 0);
   assert(agx_pack_native_txf_sampler(AGX_CHIP_G15S, &txf));
   assert(memcmp(&txf, expected_hal200, sizeof(expected_hal200)) == 0);
   assert(agx_pack_native_txf_sampler(AGX_CHIP_G17P, &txf));
   assert(memcmp(&txf, expected_hal200, sizeof(expected_hal200)) == 0);
   assert(agx_pack_native_txf_sampler(AGX_CHIP_G17G, &txf));
   assert(memcmp(&txf, expected_gen5, sizeof(expected_gen5)) == 0);
   assert(agx_pack_native_txf_sampler(AGX_CHIP_G18G, &txf));
   assert(memcmp(&txf, expected_gen5, sizeof(expected_gen5)) == 0);
   assert(!agx_pack_native_txf_sampler(AGX_CHIP_G15P, &txf));
   assert(!agx_pack_native_txf_sampler(AGX_CHIP_UNKNOWN, &txf));

   struct agx_native_sampler_state state = {
      .maximum_lod = 14.0f,
      .maximum_anisotropy = 4,
      .magnify = AGX_FILTER_LINEAR,
      .minify = AGX_FILTER_LINEAR,
      .reduction_mode = AGX_NATIVE_SAMPLER_REDUCTION_MINIMUM,
   };
   assert(agx_pack_native_sampler(AGX_CHIP_G17G, &txf, &state));
   assert(agx_pack_native_sampler(AGX_CHIP_G18G, &txf, &state));
   assert(!agx_pack_native_sampler(AGX_CHIP_G17P, &txf, &state));
   assert(!agx_pack_native_sampler(AGX_CHIP_G14G, &txf, &state));
}

int
main(void)
{
   test_generation_queries();
   test_texture_reconciled_divergences();
   test_pbe_reconciled_profiles();
   test_samplers();
   puts("asahi/lib native descriptor unification: passed");
   return 0;
}
