/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_pack.h"
#include "agx_vdm_mesh_core.h"
#include <cstring>
#include <array>

#include <gtest/gtest.h>

const struct {
   float f;
   uint16_t encoded;
   bool inexact;
} lod_cases[] = {
   /* Lower bound clamp */
   {-INFINITY, 0x00, true},
   {-0.1, 0x00, true},
   {-0.0, 0x00, true},

   /* Exact bounds */
   {0.0, 0x00},
   {14.0, 0x380},

   /* A positive clamp below one step still minifies */
   {0.01, 0x01, true},
   {1.0 / 64.0, 0x01},

   /* Upper bound clamp */
   {14.1, 0x380, true},
   {18.1, 0x380, true},
   {INFINITY, 0x380, true},
};

const struct {
   uint32_t group_size;
   uint32_t length;
   uint32_t value;
   uint32_t encoded;
} group_cases[] = {
   /* Groups of 16 in a 4-bit word */
   {16, 4, 0, 0x1},   {16, 4, 1, 0x1},   {16, 4, 16, 0x1},  {16, 4, 17, 0x2},
   {16, 4, 31, 0x2},  {16, 4, 32, 0x2},  {16, 4, 33, 0x3},  {16, 4, 239, 0xF},
   {16, 4, 240, 0xF}, {16, 4, 241, 0x0}, {16, 4, 255, 0x0}, {16, 4, 256, 0x0},
};

TEST(LODClamp, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i)
      ASSERT_EQ(__gen_pack_lod(lod_cases[i].f, 0, 9), lod_cases[i].encoded);
}

TEST(LODClamp, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(lod_cases); ++i) {
      if (lod_cases[i].inexact)
         continue;

      uint32_t cl;
      memcpy(&cl, &lod_cases[i].encoded, sizeof(lod_cases[i].encoded));

      ASSERT_EQ(__gen_unpack_lod(&cl, 0, 10), lod_cases[i].f);
   }
}

TEST(Groups, Encode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      ASSERT_EQ(__gen_to_groups(group_cases[i].value, group_cases[i].group_size,
                                group_cases[i].length),
                group_cases[i].encoded);
   }
}

TEST(Groups, Decode)
{
   for (unsigned i = 0; i < ARRAY_SIZE(group_cases); ++i) {
      unsigned expected =
         ALIGN_POT(group_cases[i].value, group_cases[i].group_size);

      /* Clamp to minimum encodable */
      if (group_cases[i].value == 0)
         expected = group_cases[i].group_size;

      ASSERT_EQ(
         __gen_from_groups(group_cases[i].encoded, group_cases[i].group_size,
                           group_cases[i].length),
         expected);
   }
}


TEST(M5MeshLimits, MatchesNativePackets)
{
   struct { uint32_t vertices, primitives, words, prefix, prim_limit, vert_limit; } cases[] = {
      {6, 2, 4, 0x504, 0x1, 0x9},
      {6, 2, 5, 0x605, 0x1, 0xb},
      {192, 64, 4, 0x504, 0x3003f, 0xc1},
   };
   for (const auto &c : cases) {
      agx_vdm_mesh_state_info info = {};
      ASSERT_TRUE(agx_vdm_m5_mesh_set_output_limits(&info, c.vertices, c.primitives, c.words));
      EXPECT_EQ(info.raw_prefix[0], c.prefix);
      EXPECT_EQ(info.raw_tail[1], c.prim_limit);
      EXPECT_EQ(info.raw_tail[4], c.vert_limit);
   }
}

/* The mesh dispatch record is an Index List block whose Primitive field
 * (bits 8..15) names the mesh output topology; native triangle meshes
 * carry 6 (0x70000600). */
TEST(M5MeshDispatch, PrimitiveFollowsTopology)
{
   for (const auto &c : {std::array<uint32_t, 2>{0, 0x70000000}, {1, 0x70000100},
                        {6, 0x70000600}}) {
      agx_vdm_mesh_dispatch_info info = {};
      for (unsigned i = 0; i < 3; ++i)
         info.threads[i] = info.local_size[i] = i ? 1 : 3;
      info.primitive = c[0];
      uint8_t bytes[AGX_VDM_MESH_DISPATCH_LENGTH];
      ASSERT_TRUE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G17S, &info, bytes));
      uint32_t header;
      memcpy(&header, bytes, 4);
      EXPECT_EQ(header, c[1]);
      agx_vdm_mesh_dispatch_info back;
      ASSERT_TRUE(agx_vdm_mesh_dispatch_unpack(AGX_CHIP_G17S, bytes, sizeof(bytes), &back));
      EXPECT_EQ(back.primitive, c[0]);
   }
   agx_vdm_mesh_dispatch_info info = {};
   for (unsigned i = 0; i < 3; ++i)
      info.threads[i] = info.local_size[i] = 1;
   info.primitive = 3; /* line strip: no mesh topology */
   uint8_t bytes[AGX_VDM_MESH_DISPATCH_LENGTH];
   EXPECT_FALSE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G17S, &info, bytes));
}

/* G16G writes G17P's HAL200 records: its mesh state and dispatch records
 * pack to the same bytes, and other AGX3B identities stay refused. */
TEST(Gen1Mesh, G16GPacksG17PRecords)
{
   agx_vdm_mesh_state_info state = {};
   ASSERT_TRUE(agx_vdm_m5_mesh_set_output_limits(&state, 64, 126, 8));
   state.raw_prefix[1] = 0x30;
   state.object_pipeline = UINT64_C(0x10000128000);
   state.mesh_pipeline = UINT64_C(0x10000129000);
   state.raw_mesh_template = 0x40;
   uint8_t g17p[AGX_VDM_MESH_STATE_LENGTH], g16g[AGX_VDM_MESH_STATE_LENGTH];
   ASSERT_TRUE(agx_vdm_mesh_state_pack(AGX_CHIP_G17P, &state, g17p));
   ASSERT_TRUE(agx_vdm_mesh_state_pack(AGX_CHIP_G16G, &state, g16g));
   EXPECT_EQ(memcmp(g17p, g16g, sizeof(g17p)), 0);
   agx_vdm_mesh_state_info back;
   ASSERT_TRUE(agx_vdm_mesh_state_unpack(AGX_CHIP_G16G, g16g, sizeof(g16g), &back));
   EXPECT_EQ(back.mesh_pipeline, state.mesh_pipeline);
   /* The GEN1 state requires the 0x10 prefix bit on both. */
   state.raw_prefix[1] = 0;
   EXPECT_FALSE(agx_vdm_mesh_state_pack(AGX_CHIP_G16G, &state, g16g));
   EXPECT_FALSE(agx_vdm_mesh_state_pack(AGX_CHIP_G17P, &state, g17p));

   agx_vdm_mesh_dispatch_info dispatch = {};
   dispatch.threads[0] = 96, dispatch.threads[1] = dispatch.threads[2] = 1;
   dispatch.local_size[0] = 32, dispatch.local_size[1] = dispatch.local_size[2] = 1;
   dispatch.primitive = 6;
   uint8_t d17[AGX_VDM_MESH_DISPATCH_LENGTH], d16[AGX_VDM_MESH_DISPATCH_LENGTH];
   ASSERT_TRUE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G17P, &dispatch, d17));
   ASSERT_TRUE(agx_vdm_mesh_dispatch_pack(AGX_CHIP_G16G, &dispatch, d16));
   EXPECT_EQ(memcmp(d17, d16, sizeof(d17)), 0);

   for (agx_chip chip : {AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G16S, AGX_CHIP_G13G}) {
      EXPECT_FALSE(agx_vdm_mesh_dispatch_pack(chip, &dispatch, d16));
      EXPECT_FALSE(agx_chip_gen1_mesh(chip));
   }
   EXPECT_TRUE(agx_chip_gen1_mesh(AGX_CHIP_G16G) && agx_chip_gen1_mesh(AGX_CHIP_G17P));
   for (agx_chip chip : {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G18G})
      EXPECT_FALSE(agx_chip_gen1_mesh(chip));
}

TEST(M5MeshLimits, RejectsInvalidLimitsWithoutMutation)
{
   for (const auto &c : {std::array<uint32_t, 3>{0, 1, 4}, {1, 0, 4},
                        {257, 1, 4}, {1, 257, 4}, {1, 1, 3}, {1, 1, 256}}) {
      agx_vdm_mesh_state_info info;
      memset(&info, 0x5a, sizeof(info));
      const auto before = info;
      EXPECT_FALSE(agx_vdm_m5_mesh_set_output_limits(&info, c[0], c[1], c[2]));
      EXPECT_EQ(memcmp(&info, &before, sizeof(info)), 0);
   }
}
