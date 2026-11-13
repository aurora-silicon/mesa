/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Body of agx3_refit_leaf.comp, for it and for shaders that run an
 * update's triangle leaf pass. Accessor macros (AGX3_REFIT_LEAF_*): ARGS,
 * GLOBAL_ID, ENTRY. Expects hk_build_interface.h, agx3_node.h and the
 * SignedZeroInfNanPreserve execution mode.
 */
#ifndef AGX3_REFIT_LEAF_ARGS
#define AGX3_REFIT_LEAF_ARGS args
#endif
#ifndef AGX3_REFIT_LEAF_GLOBAL_ID
#define AGX3_REFIT_LEAF_GLOBAL_ID gl_GlobalInvocationID
#endif
#ifndef AGX3_REFIT_LEAF_ENTRY
#define AGX3_REFIT_LEAF_ENTRY main
#endif

void
AGX3_REFIT_LEAF_ENTRY()
{
   uint id = AGX3_REFIT_LEAF_GLOBAL_ID.x;
   if (id >= AGX3_REFIT_LEAF_ARGS.primitive_count)
      return;
   hk_agx3_refit_state refit = DEREF(REF(hk_agx3_refit_state)(AGX3_REFIT_LEAF_ARGS.refit));
   uint64_t slots = refit.retained +
                    uint64_t(HK_AGX3_RETAINED_SLOTS(refit.job_count, AGX3_REFIT_LEAF_ARGS.leaf_bound));
   uint slot = DEREF(INDEX(uint32_t, slots, AGX3_REFIT_LEAF_ARGS.geometry.first_id + id));
   if (slot == ~0u)
      return;

   vk_bvh_geometry_data geom = AGX3_REFIT_LEAF_ARGS.geometry;
   triangle_vertices vertices =
      load_vertices(geom.data, load_indices(geom.indices, geom.index_format, id),
                    geom.vertex_format, geom.stride);
   if (geom.transform != 0ul) {
      mat4 transform = mat4(1.0);
      for (uint col = 0; col < 4; col++)
         for (uint row = 0; row < 3; row++)
            transform[col][row] = DEREF(INDEX(float, geom.transform, col + row * 4));
      for (uint i = 0; i < 3; i++)
         vertices.vertex[i] = transform * vertices.vertex[i];
   }
   /* An updatable structure keeps an inactive triangle as zero vertices
    * (VK_BUILD_FLAG_ALWAYS_ACTIVE). */
   bool inactive = false;
   for (uint i = 0; i < 3; i++)
      inactive = inactive || any(isnan(vertices.vertex[i]));
   uvec3 v[3];
   for (uint i = 0; i < 3; i++)
      v[i] = inactive ? uvec3(0u) : floatBitsToUint(vertices.vertex[i].xyz);

   uint64_t triangle = uint64_t(INDEX(hk_agx3_refit_triangle, AGX3_REFIT_LEAF_ARGS.triangles, slot));
   store128(triangle, uvec4(v[0], v[1].x));
   store128(triangle + 16ul, uvec4(v[1].yz, v[2].xy));
   store128(triangle + 32ul, uvec4(v[2].z, geom.geometry_id, id, 0u));
}
