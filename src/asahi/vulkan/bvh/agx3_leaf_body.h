/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Body of agx3_leaf.comp, for it and for shaders that run node encoding.
 * Accessor macros (AGX3_LEAF_*): ARGS, LOCAL_ID, GROUP_ID,
 * SUBGROUP_INVOCATION_ID, ENTRY. Expects hk_build_interface.h, agx3_node.h,
 * JOBS and VERTICES; honours the HK_AGX3_REFIT and HK_AGX3_LEAF_* variants.
 */
#ifndef AGX3_LEAF_ARGS
#define AGX3_LEAF_ARGS args
#endif
#ifndef AGX3_LEAF_LOCAL_ID
#define AGX3_LEAF_LOCAL_ID gl_LocalInvocationID
#endif
#ifndef AGX3_LEAF_GROUP_ID
#define AGX3_LEAF_GROUP_ID gl_WorkGroupID
#endif
#ifndef AGX3_LEAF_SUBGROUP_INVOCATION_ID
#define AGX3_LEAF_SUBGROUP_INVOCATION_ID gl_SubgroupInvocationID
#endif
#ifndef AGX3_LEAF_ENTRY
#define AGX3_LEAF_ENTRY main
#endif

bool
fit_qtb(vec3 lo, vec3 hi, out vec3 base, out uvec3 exponent)
{
   for (uint a = 0; a < 3; ++a) {
      base[a] = origin_down(lo[a]);
      precise float extent = hi[a] - base[a];
      if (!finite(base[a]) || !finite(extent))
         return false;
      uint e = (floatBitsToUint(extent) >> 23) & 255u;
      exponent[a] = e > 8u ? e - 7u : 1u;
   }
   return true;
}

/* A triangle record holds four vertices and two primitive words. The
 * hardware tests triangle A = (v0, v1, v2) and triangle B = (v2, v1, v3),
 * which shares A's edge v1-v2 with the opposite direction. Bits 30-31 of
 * each primitive word (above its 30-bit primitive index) give the rotation
 * of the application's vertex order against that stored order, which the
 * hardware applies to the barycentrics it reports: 0 for none, 2 when the
 * original order starts at the stored second vertex, 1 when it starts at
 * the third. An unpaired record repeats v2 as v3, so B is never hit.
 *
 * Store the pair (a, b) sharing edge ea of a and edge eb of b (see
 * hk_agx3_shared_edge): rotate a so its shared edge becomes v1-v2, then b
 * is (v2, v1, v3) rotated to start at its own vertex eb. */
#ifdef HK_AGX3_LEAF_PAIRS
void
arrange_pair(inout vec3 p[4], inout uint prim0, out uint prim1, vec3 b[3],
             uint edges, uint b_id)
{
   uint ea = edges & 3u, eb = edges >> 2;
   vec3 a[3] = vec3[3](p[0], p[1], p[2]);
   p[0] = a[(ea + 2u) % 3u];
   p[1] = a[ea];
   p[2] = a[(ea + 1u) % 3u];
   p[3] = b[(eb + 2u) % 3u];
   prim0 |= ((ea + 2u) % 3u) << 30;
   prim1 = b_id | (eb << 30);
}
#endif

uint
quantize_vertex(float value, float base, uint exponent)
{
   /* A binary32 subtraction can round an exact coordinate just below a
    * grid line onto that line. Retain its residual (TwoDiff) so truncation
    * agrees with the reference's binary64 arithmetic without shaderFloat64.
    * All operations must remain separate; contraction/reassociation would
    * discard the very residual we are recovering. */
   precise float delta = value - base;
   precise float virtual_base = value - delta;
   precise float virtual_value = delta + virtual_base;
   precise float base_error = virtual_base - base;
   precise float value_error = value - virtual_value;
   precise float error = value_error + base_error;
   precise float q = ldexp(delta, 127 - int(exponent));
   float whole = floor(q);
   if (q == whole && error < 0.0)
      whole -= 1.0;
   return uint(clamp(whole, 0.0, 255.0));
}

mat3x4
invert_affine(mat3x4 otw)
{
   vec3 r0 = vec3(otw[0][0], otw[0][1], otw[0][2]);
   vec3 r1 = vec3(otw[1][0], otw[1][1], otw[1][2]);
   vec3 r2 = vec3(otw[2][0], otw[2][1], otw[2][2]);
   vec3 t = vec3(otw[0][3], otw[1][3], otw[2][3]);

   vec3 c0 = cross(r1, r2);
   vec3 c1 = cross(r2, r0);
   vec3 c2 = cross(r0, r1);

   float det = dot(r0, c0);
   /* A singular instance transform is undefined behaviour in Vulkan.  Fall
    * back to something finite so the traversal cannot fault on a NaN
    * address computation. */
   float inv_det = det == 0.0 ? 0.0 : 1.0 / det;

   /* Rows of R^-1 are the columns of the cofactor matrix, scaled. */
   vec3 i0 = vec3(c0.x, c1.x, c2.x) * inv_det;
   vec3 i1 = vec3(c0.y, c1.y, c2.y) * inv_det;
   vec3 i2 = vec3(c0.z, c1.z, c2.z) * inv_det;

   mat3x4 wto;
   wto[0] = vec4(i0, -dot(i0, t));
   wto[1] = vec4(i1, -dot(i1, t));
   wto[2] = vec4(i2, -dot(i2, t));
   return wto;
}

void
store64(uint64_t address, uint64_t value)
{
   store32(address, uint(value));
   store32(address + 4ul, uint(value >> 32));
}

void
encode_instance(hk_agx3_leaf_job job, uint lane, uint64_t transforms,
                uint64_t pointers)
{
   REF(vk_ir_instance_node) src = REF(vk_ir_instance_node)(
      OFFSET(job.intermediate_bvh, job.triangle_offsets[lane]));
   uint64_t blas = DEREF(src).base_ptr;
   uint64_t record_offset = uint64_t(job.child_slot + 2u * lane) * 64ul;
   uint64_t record = job.output_bvh + record_offset;
   uint index = DEREF(src).instance_id;
   uint64_t transform = transforms + uint64_t(index) * 64ul;
   mat3x4 otw = DEREF(src).otw_matrix;
   mat3x4 wto = invert_affine(otw);
   for (uint i = 0; i < 32; ++i)
      store32(record + uint64_t(i) * 4ul, 0);
   store32(record, 26);
   store32(record + 4ul, DEREF(REF(uint32_t)(blas + 176ul)));
   store32(record + 8ul, DEREF(src).sbt_offset_and_flags >> 24);
   store32(record + 12ul, DEREF(src).custom_instance_and_mask >> 24);
   for (uint c = 0; c < 4; ++c)
      for (uint r = 0; r < 3; ++r) {
         uint offset = 4u * (c * 3u + r);
         store32(record + 16ul + uint64_t(offset), floatBitsToUint(wto[r][c]));
         store32(job.output_bvh + transform + uint64_t(offset),
                 floatBitsToUint(otw[r][c]));
      }
   for (uint i = 0; i < 6; ++i)
      store32(record + 64ul + uint64_t(i) * 4ul,
              DEREF(REF(uint32_t)(blas + 80ul + uint64_t(i) * 4ul)));
   store64(record + 88ul, blas);
   store32(record + 96ul, DEREF(src).custom_instance_and_mask & 0xffffffu);
   /* Native Metal user_instance_id reads this independent32-bit field. */
   store32(record + 100ul, index);
   /* Vulkan-only metadata lives outside the native128-byte instance record. */
   store32(job.output_bvh + transform + 48ul, DEREF(src).sbt_offset_and_flags);
   store64(record + 104ul, transform - record_offset);
   store64(job.output_bvh + pointers + 16ul * uint64_t(index), record_offset + 4ul);
   store64(job.output_bvh + pointers + 16ul * uint64_t(index) + 8ul, 1ul);
}

void
AGX3_LEAF_ENTRY()
{
   uint slot = AGX3_LEAF_LOCAL_ID.x / 8u, lane = AGX3_LEAF_LOCAL_ID.x % 8u;
   uint id = AGX3_LEAF_GROUP_ID.x * JOBS + slot;
   /* Every lane reaches every subgroup operation; a job that has no node to
    * encode, or fails a check, only stops doing work. */
#ifdef HK_AGX3_REFIT
   hk_agx3_refit_state refit = DEREF(REF(hk_agx3_refit_state)(AGX3_LEAF_ARGS.refit));
   uint jobs = refit.job_count;
   uint64_t records = refit.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   bool live = id < jobs;
   hk_agx3_leaf_job job;
   hk_agx3_retained_job record;
   uint count = 0;
   if (live) {
      /* Inner jobs are encoded from their children's bounds. */
      record = DEREF(INDEX(hk_agx3_retained_job, records, id));
      job.node_kind = (record.kind_count >> 4) & 15u;
      live = job.node_kind != HK_AGX3_NODE_INNER;
   }
   if (live) {
      uint64_t offsets = records + uint64_t(jobs) * uint64_t(HK_AGX3_RETAINED_JOB_BYTES);
      job.intermediate_bvh = AGX3_LEAF_ARGS.intermediate_bvh;
      job.output_bvh = AGX3_LEAF_ARGS.output_bvh;
      job.root_slot = record.root_slot;
      job.child_slot = record.child_slot;
      job.child_count = record.kind_count & 15u;
      /* Triangles at their slots need no child offsets. */
      for (uint c = 0; c < 8u; ++c)
         job.triangle_offsets[c] = c < job.child_count && AGX3_LEAF_ARGS.triangles == 0ul
            ? DEREF(INDEX(uint32_t, offsets, record.first + c)) : 0u;
      count = job.child_count;
#else
   uint jobs = AGX3_LEAF_ARGS.job_count;
   if (AGX3_LEAF_ARGS.counts != 0ul)
      jobs = min(jobs, DEREF(REF(uint32_t)(AGX3_LEAF_ARGS.counts)));
   bool live = id < jobs;
   hk_agx3_leaf_job job;
   uint count = 0;
   if (live) {
      job = DEREF(INDEX(hk_agx3_leaf_job, AGX3_LEAF_ARGS.jobs, id));
      count = job.child_count;
#endif
#ifdef HK_AGX3_LEAF_ONLY
      const bool kind_ok = job.node_kind == HK_AGX3_NODE_INNER ||
                           job.node_kind == HK_AGX3_LEAF_ONLY;
#else
      const bool kind_ok = job.node_kind <= HK_AGX3_NODE_AABBS;
#endif
      if (count < 1u || count > 8 || !kind_ok) {
         if (lane == 0)
            DEREF(INDEX(uint32_t, AGX3_LEAF_ARGS.status, id)) = HK_AGX3_LEAF_INVALID_COUNT;
         live = false;
      }
   }
   const bool valid = live;
#ifdef HK_AGX3_LEAF_ONLY
   const bool leaf = HK_AGX3_LEAF_ONLY == HK_AGX3_NODE_LEAF &&
                     live && job.node_kind == HK_AGX3_NODE_LEAF;
   const bool instances = HK_AGX3_LEAF_ONLY == HK_AGX3_NODE_INSTANCES &&
                          live && job.node_kind == HK_AGX3_NODE_INSTANCES;
   const bool aabbs = HK_AGX3_LEAF_ONLY == HK_AGX3_NODE_AABBS &&
                      live && job.node_kind == HK_AGX3_NODE_AABBS;
#else
   bool leaf = live && job.node_kind == HK_AGX3_NODE_LEAF;
   bool instances = live && job.node_kind == HK_AGX3_NODE_INSTANCES;
   bool aabbs = live && job.node_kind == HK_AGX3_NODE_AABBS;
#endif
   uint capacity = max(min(count, 4u), 1u), blocks = (count + capacity - 1u) / capacity;
   /* This lane's child: its bounds, record vertices and words. Lanes past
    * the last child keep empty bounds, which the reductions ignore. */
   bool mine = live && lane < count;
   vec3 my_lo = vec3(INFINITY), my_hi = vec3(-INFINITY);
   vec3 my_vertices[4];
   uint my_primitive = 0, my_primitive1 = 0, my_geometry_id = 0, my_type = 0;
   uint my_bad = 0;
   if (mine) {
      if (!leaf) {
         REF(vk_ir_node)
         src = REF(vk_ir_node)(
            OFFSET(job.intermediate_bvh, job.triangle_offsets[lane]));
         if (aabbs) {
            REF(vk_ir_aabb_node) box = REF(vk_ir_aabb_node)(src);
            my_geometry_id = DEREF(box).geometry_id_and_flags;
            my_primitive = DEREF(box).primitive_id;
         }
         my_lo = DEREF(src).aabb.min;
         my_hi = DEREF(src).aabb.max;
         for (uint a = 0; a < 3; ++a) {
            if (!finite(my_lo[a]) || !finite(my_hi[a]))
               my_bad |= HK_AGX3_LEAF_NONFINITE;
            if (my_lo[a] > my_hi[a])
               my_bad |= HK_AGX3_LEAF_RANGE;
         }
      } else {
#ifndef HK_AGX3_LEAF_PAIRS
         vec3 tri[3];
         uint geometry_flags;
#ifdef HK_AGX3_REFIT
         if (AGX3_LEAF_ARGS.triangles != 0ul) {
            REF(hk_agx3_refit_triangle) src =
               INDEX(hk_agx3_refit_triangle, AGX3_LEAF_ARGS.triangles, record.first + lane);
            geometry_flags = DEREF(src).geometry_id_and_flags;
            my_primitive = DEREF(src).triangle_id;
            for (uint v = 0; v < 3; ++v)
               for (uint a = 0; a < 3; ++a)
                  tri[v][a] = DEREF(src).coords[v][a];
         } else
#endif
         {
            REF(vk_ir_triangle_node)
            src = REF(vk_ir_triangle_node)(
               OFFSET(job.intermediate_bvh, job.triangle_offsets[lane]));
            geometry_flags = DEREF(src).geometry_id_and_flags;
            my_primitive = DEREF(src).triangle_id;
            for (uint v = 0; v < 3; ++v)
               for (uint a = 0; a < 3; ++a)
                  tri[v][a] = DEREF(src).coords[v][a];
         }
         my_geometry_id = geometry_flags & ~VK_GEOMETRY_OPAQUE;
         /* The Gen1 triangle record is the same for types 8 and 16;
          * type 8 marks the geometry opaque. */
         my_type = (geometry_flags & VK_GEOMETRY_OPAQUE) != 0 ? 8u : 16u;
         for (uint v = 0; v < 3; ++v) {
            for (uint a = 0; a < 3; ++a)
               if (!finite(tri[v][a]))
                  my_bad |= HK_AGX3_LEAF_NONFINITE;
            my_vertices[v] = tri[v];
         }
         my_vertices[3] = my_vertices[2];
#else
         /* A pair names the internal node over its two triangles. */
         uint offset = job.triangle_offsets[lane];
         bool paired = (offset & HK_AGX3_LEAF_PAIR_OFFSET) != 0u;
         offset &= ~3u;
         REF(vk_ir_triangle_node)
         src = REF(vk_ir_triangle_node)(OFFSET(job.intermediate_bvh, offset));
         REF(vk_ir_triangle_node) second = src;
         if (paired) {
            REF(vk_ir_box_node) box =
               REF(vk_ir_box_node)(OFFSET(job.intermediate_bvh, offset));
            src = REF(vk_ir_triangle_node)(OFFSET(job.intermediate_bvh,
               ir_id_to_offset(DEREF(box).children[0])));
            second = REF(vk_ir_triangle_node)(OFFSET(job.intermediate_bvh,
               ir_id_to_offset(DEREF(box).children[1])));
         }
         uint geometry_flags = DEREF(src).geometry_id_and_flags;
         my_geometry_id = geometry_flags & ~VK_GEOMETRY_OPAQUE;
         my_type = (geometry_flags & VK_GEOMETRY_OPAQUE) != 0 ? 8u : 16u;
         vec3 p[4];
         for (uint v = 0; v < 3; ++v)
            for (uint a = 0; a < 3; ++a)
               p[v][a] = DEREF(src).coords[v][a];
         /* An unpaired record repeats its last vertex. */
         p[3] = p[2];
         my_primitive = DEREF(src).triangle_id;
         my_primitive1 = my_primitive;
         if (paired) {
            vec3 a[3] = vec3[3](p[0], p[1], p[2]), b[3];
            for (uint v = 0; v < 3; ++v)
               for (uint k = 0; k < 3; ++k)
                  b[v][k] = DEREF(second).coords[v][k];
            uint edges = hk_agx3_shared_edge(a, b);
            if (edges == ~0u)
               my_bad |= HK_AGX3_LEAF_RANGE;
            else
               arrange_pair(p, my_primitive, my_primitive1, b, edges,
                            DEREF(second).triangle_id);
         }
         for (uint v = 0; v < 4; ++v) {
            for (uint a = 0; a < 3; ++a)
               if (!finite(p[v][a]))
                  my_bad |= HK_AGX3_LEAF_NONFINITE;
            my_vertices[v] = p[v];
         }
#endif
         my_lo = min(min(my_vertices[0], my_vertices[1]),
                     min(my_vertices[2], my_vertices[3]));
         my_hi = max(max(my_vertices[0], my_vertices[1]),
                     max(my_vertices[2], my_vertices[3]));
      }
   }

   /* Reductions over the node's eight lanes, and over each QTB block's four
    * (children 0-3 and 4-7; a node of four or fewer has one block). Every
    * lane takes part, in uniform control flow. */
   uint loaded_bad = subgroupClusteredOr(my_bad, 8u);
   vec3 node_lo = subgroupClusteredMin(my_lo, 8u);
   vec3 node_hi = subgroupClusteredMax(my_hi, 8u);
   vec3 block_lo = subgroupClusteredMin(my_lo, 4u);
   vec3 block_hi = subgroupClusteredMax(my_hi, 4u);
   uint type0 = subgroupShuffle(my_type, AGX3_LEAF_SUBGROUP_INVOCATION_ID & ~7u);

   /* Root word of child lane: its header bits, then its quantized bounds;
    * and its triangle's QTB vertex bytes, each in bits of its own. */
   vec3 base, qbase;
   uvec3 exponent, qexp;
   uint64_t my_root = 0ul;
   uint packed[3] = uint[3](0u, 0u, 0u);
   if (live && loaded_bad == 0u) {
      bool ok = fit_root(node_lo, node_hi, base, exponent);
      if (ok)
         my_root = root_word(lane, count, job.node_kind, type0, job.child_slot,
                             AGX3_LEAF_ARGS.slot_bias, base, exponent, my_lo, my_hi, ok);
      if (ok && leaf && lane < count) {
         ok = fit_qtb(block_lo, block_hi, qbase, qexp);
         /* The QTB holds the record's vertices in reverse order; an
          * unpaired record repeats v2 as v3. */
#ifdef HK_AGX3_LEAF_PAIRS
         const uint order[4] = uint[4](3, 2, 1, 0);
#else
         const uint order[4] = uint[4](2, 2, 1, 0);
#endif
         /* A triangle's twelve vertex bytes are three whole words of its
          * block. */
         for (uint v = 0; v < 4 && ok; ++v)
            for (uint a = 0; a < 3; ++a) {
               /* Power-of-two scaling avoids reciprocal overflow on small
                * axes, including the minimum normal binary32 exponent. */
               uint q = quantize_vertex(my_vertices[order[v]][a], qbase[a], qexp[a]);
               uint byte_offset = v * 3u + 2u - a;
               packed[byte_offset / 4u] |= q << ((byte_offset % 4u) * 8u);
            }
      }
      if (!ok)
         my_bad |= HK_AGX3_LEAF_RANGE;
   }
   uint node_bad = subgroupClusteredOr(my_bad, 8u);
   if (!valid)
      return;
   if (node_bad != 0u) {
      if (lane == 0) {
         DEREF(INDEX(uint32_t, AGX3_LEAF_ARGS.status, id)) = node_bad;
#ifdef HK_AGX3_REFIT
         store_aabb(uint64_t(INDEX(vk_aabb, AGX3_LEAF_ARGS.bounds, record.position)),
                    HK_AGX3_FAILED_BOUNDS, HK_AGX3_FAILED_BOUNDS);
#endif
      }
      return;
   }

   uint64_t children =
      uint64_t(job.output_bvh) + uint64_t(job.child_slot) * 64ul;
   if (aabbs && lane < count) {
      /* Native Gen1 marker24 selects the static bounding-box leaf. Its
       * payload starts at+4: xyz min/max then geometry/primitive words.
       * Bit31 of the geometry word is opacity; each root child uses64B. */
      uint64_t geometry = children + uint64_t(lane) * 64ul;
      uvec3 lo_bits = floatBitsToUint(my_lo), hi_bits = floatBitsToUint(my_hi);
      store128(geometry, uvec4(24u, lo_bits));
      store128(geometry + 16ul, uvec4(hi_bits, my_geometry_id));
      store128(geometry + 32ul, uvec4(my_primitive, 0u, 0u, 0u));
      store128(geometry + 48ul, uvec4(0u));
   }
#ifdef HK_AGX3_REFIT
   if (instances && lane < count) {
      /* As the envelope lays them out after the nodes. */
      uint64_t transforms = uint64_t(refit.end_slot) * 64ul;
      encode_instance(job, lane, transforms,
                      transforms + 64ul * uint64_t(AGX3_LEAF_ARGS.leaf_bound));
   }
#else
   if (instances && lane < count)
      encode_instance(job, lane, DEREF(REF(uint64_t)(job.output_bvh + 104ul)),
                      DEREF(REF(uint64_t)(job.output_bvh + 128ul)));
#endif
   if (leaf && lane < count) {
      /* The record: its header, the four x, y and z coordinates, then the
       * geometry and the two primitive words, as four 16-byte stores. */
      uint64_t geometry = children + uint64_t(blocks + lane) * 64ul;
      uvec3 p0 = floatBitsToUint(my_vertices[0]);
      uvec3 p1 = floatBitsToUint(my_vertices[1]);
      uvec3 p2 = floatBitsToUint(my_vertices[2]);
      uvec3 p3 = floatBitsToUint(my_vertices[VERTICES - 1u]);
#ifdef HK_AGX3_LEAF_PAIRS
      uint second = my_primitive1;
#else
      uint second = my_primitive;
#endif
      store128(geometry, uvec4(0x40u | my_type, p0.x, p1.x, p2.x));
      store128(geometry + 16ul, uvec4(p3.x, p0.y, p1.y, p2.y));
      store128(geometry + 32ul, uvec4(p3.y, p0.z, p1.z, p2.z));
      store128(geometry + 48ul, uvec4(p3.z, my_geometry_id, my_primitive, second));

      /* QTB block: the first lane of a block writes its header and clears
       * the slots of triangles it does not have; each lane writes its own
       * triangle's three words. */
      uint block = lane / capacity, in_block = lane % capacity;
      uint64_t qtb = children + uint64_t(block) * 64ul;
      if (in_block == 0u) {
         store128(qtb, uvec4(6u | (qexp.z << 8) | (floatBitsToUint(qbase.x) & 0xffff0000u),
                             (floatBitsToUint(qbase.y) >> 16) |
                                (floatBitsToUint(qbase.z) & 0xffff0000u),
                             qexp.x | (qexp.y << 8), 0u));
         uint present = min(capacity, count - block * capacity);
         for (uint w = 4u + present * 3u; w < 16u; ++w)
            store32(qtb + uint64_t(w) * 4ul, 0u);
      }
      for (uint w = 0; w < 3u; ++w)
         store32(qtb + 16ul + uint64_t(in_block * 12u + w * 4u), packed[w]);
   }
   /* Every lane stores its own root word, filling one 64-byte line. */
   store_word64(uint64_t(job.output_bvh) + uint64_t(job.root_slot) * 64ul +
                   uint64_t(lane) * 8ul, my_root);
   if (lane == 0) {
      DEREF(INDEX(uint32_t, AGX3_LEAF_ARGS.status, id)) = HK_AGX3_LEAF_SUCCESS;
#ifdef HK_AGX3_REFIT
      /* Its parent's encoding reads these (agx3_refit.comp). */
      store_aabb(uint64_t(INDEX(vk_aabb, AGX3_LEAF_ARGS.bounds, record.position)), node_lo, node_hi);
#endif
   }
}
