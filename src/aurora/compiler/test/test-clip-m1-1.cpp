/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"

namespace {

nir_intrinsic_instr *
store_pos(nir_builder *b, nir_def *v, unsigned component, unsigned mask)
{
   return nir_store_per_vertex_output(
      b, v, nir_imm_int(b, 0), nir_imm_int(b, 0), .write_mask = mask,
      .component = component, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
}

nir_builder
mesh_shader()
{
   return nir_builder_init_simple_shader(MESA_SHADER_MESH, &agx_nir_options,
                                         "clip m1_1");
}

/* The lowered z: the lerp where the coefficient is nonzero, else z itself. */
nir_alu_instr *
lowered_lerp(nir_def *store_src, unsigned comp, nir_scalar *orig_z)
{
   nir_scalar z = nir_scalar_chase_movs(nir_get_scalar(store_src, comp));
   if (!nir_scalar_is_alu(z) || nir_scalar_alu_op(z) != nir_op_bcsel)
      return NULL;
   *orig_z = nir_scalar_chase_movs(nir_scalar_chase_alu_src(z, 2));
   nir_scalar lerp = nir_scalar_chase_movs(nir_scalar_chase_alu_src(z, 1));
   if (!nir_scalar_is_alu(lerp) || nir_scalar_alu_op(lerp) != nir_op_ffma)
      return NULL;
   return nir_def_as_alu(lerp.def);
}

} // namespace

TEST(ClipM1_1, FullStoreIsLowered)
{
   nir_builder b = mesh_shader();
   nir_def *pos = agx_test_load_preamble(&b, 4, 32, 0);
   nir_intrinsic_instr *store = store_pos(&b, pos, 0, 0xf);

   EXPECT_TRUE(agx_nir_clip_m1_1_complete(b.shader));
   ASSERT_TRUE(agx_nir_lower_clip_m1_1(b.shader));
   nir_validate_shader(b.shader, "clip m1_1");

   nir_scalar orig_z;
   nir_alu_instr *lerp = lowered_lerp(store->src[0].ssa, 2, &orig_z);
   ASSERT_NE(lerp, nullptr);
   EXPECT_TRUE(nir_alu_instr_no_reassoc(lerp));
   EXPECT_TRUE(nir_scalar_equal(orig_z, nir_get_scalar(pos, 2)));
   ralloc_free(b.shader);
}

/* A z-only store has no w to lerp with: the pass leaves it and the driver
 * must refuse the shader. */
TEST(ClipM1_1, SplitZStoreIsIncomplete)
{
   nir_builder b = mesh_shader();
   store_pos(&b, agx_test_load_preamble(&b, 2, 32, 0), 0, 0x3);
   nir_intrinsic_instr *z = store_pos(&b, agx_test_load_preamble(&b, 1, 32, 4),
                                      2, 0x1);
   store_pos(&b, agx_test_load_preamble(&b, 1, 32, 6), 3, 0x1);
   nir_def *z_src = z->src[0].ssa;

   EXPECT_FALSE(agx_nir_clip_m1_1_complete(b.shader));
   EXPECT_FALSE(agx_nir_lower_clip_m1_1(b.shader));
   EXPECT_EQ(z->src[0].ssa, z_src);
   ralloc_free(b.shader);
}

/* x and y alone, then z and w in one store from component 2. */
TEST(ClipM1_1, ZwStoreIsLowered)
{
   nir_builder b = mesh_shader();
   store_pos(&b, agx_test_load_preamble(&b, 2, 32, 0), 0, 0x3);
   nir_intrinsic_instr *zw =
      store_pos(&b, agx_test_load_preamble(&b, 2, 32, 4), 2, 0x3);

   EXPECT_TRUE(agx_nir_clip_m1_1_complete(b.shader));
   ASSERT_TRUE(agx_nir_lower_clip_m1_1(b.shader));
   nir_validate_shader(b.shader, "clip m1_1");

   nir_scalar orig_z;
   ASSERT_NE(lowered_lerp(zw->src[0].ssa, 0, &orig_z), nullptr);
   ralloc_free(b.shader);
}

TEST(ClipM1_1, PositionReadBackIsIncomplete)
{
   nir_builder b = mesh_shader();
   store_pos(&b, agx_test_load_preamble(&b, 4, 32, 0), 0, 0xf);
   nir_load_per_vertex_output(
      &b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});

   EXPECT_FALSE(agx_nir_clip_m1_1_complete(b.shader));
   ralloc_free(b.shader);
}
