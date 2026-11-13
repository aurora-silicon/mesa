/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Body of agx3_plan.comp, for it and for shaders that run planning.
 * Accessor macros (AGX3_PLAN_*): ARGS, GLOBAL_ID, ENTRY.
 * Expects hk_build_interface.h; honours HK_AGX3_PLAN_PAIRS.
 */
#ifndef AGX3_PLAN_ARGS
#define AGX3_PLAN_ARGS args
#endif
#ifndef AGX3_PLAN_GLOBAL_ID
#define AGX3_PLAN_GLOBAL_ID gl_GlobalInvocationID
#endif
#ifndef AGX3_PLAN_ENTRY
#define AGX3_PLAN_ENTRY main
#endif

/* The common builder's internal node count, final before planning starts:
 * main() reads it once rather than every source_index() reloading it past
 * this invocation's own stores and atomics. */
uint internal_count;

void
fail(uint flag)
{
   atomicOr(DEREF(AGX3_PLAN_ARGS.state).error, flag);
}


uint
leaf_size()
{
   return AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR
             ? SIZEOF(vk_ir_instance_node)
             : AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR
                ? SIZEOF(vk_ir_aabb_node) : SIZEOF(vk_ir_triangle_node);
}

uint
source_index(uint ir_id)
{
   if (ir_id == VK_BVH_INVALID_NODE)
      return 0xffffffffu;
   uint offset = ir_id_to_offset(ir_id), kind = ir_id_to_type(ir_id);
   if (kind == (AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR ? vk_ir_node_instance : AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR ? vk_ir_node_aabb : vk_ir_node_triangle) && offset < AGX3_PLAN_ARGS.root_offset &&
       offset % leaf_size() == 0) {
      uint i = offset / leaf_size();
      if (i < AGX3_PLAN_ARGS.leaf_bound)
         return i;
   }
   if (kind == vk_ir_node_internal && offset >= AGX3_PLAN_ARGS.root_offset &&
       (offset - AGX3_PLAN_ARGS.root_offset) % SIZEOF(vk_ir_box_node) == 0) {
      uint i = (offset - AGX3_PLAN_ARGS.root_offset) / SIZEOF(vk_ir_box_node);
      if (i < internal_count &&
          i < AGX3_PLAN_ARGS.node_bound - AGX3_PLAN_ARGS.leaf_bound)
         return AGX3_PLAN_ARGS.leaf_bound + i;
   }
   fail(HK_AGX3_PLAN_INDEX);
   return 0xffffffffu;
}

uint
source_offset(uint index)
{
   return index < AGX3_PLAN_ARGS.leaf_bound
             ? index * leaf_size()
             : AGX3_PLAN_ARGS.root_offset +
                  (index - AGX3_PLAN_ARGS.leaf_bound) * SIZEOF(vk_ir_box_node);
}

uint
root_index()
{
   return AGX3_PLAN_ARGS.leaf_bound + internal_count - 1;
}

uvec2
children(uint index)
{
   REF(vk_ir_box_node)
   node =
      REF(vk_ir_box_node)(OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(index)));
   return uvec2(DEREF(node).children[0], DEREF(node).children[1]);
}

#ifdef HK_AGX3_PLAN_PAIRS
/* Triangle leaf index of an IR child reference, or ~0u. Unlike
 * source_index() this never records an error, so phase 0 may use it before
 * the plan state exists. */
uint
triangle_index(uint ir_id)
{
   if (ir_id == VK_BVH_INVALID_NODE || ir_id_to_type(ir_id) != vk_ir_node_triangle)
      return ~0u;
   uint offset = ir_id_to_offset(ir_id);
   if (offset >= AGX3_PLAN_ARGS.root_offset || offset % SIZEOF(vk_ir_triangle_node) != 0)
      return ~0u;
   uint i = offset / SIZEOF(vk_ir_triangle_node);
   return i < AGX3_PLAN_ARGS.leaf_bound ? i : ~0u;
}

/* Pair flags (HK_AGX3_PLAN_PAIR...) for internal node `index`: set when its
 * children are two triangles of one geometry and opacity sharing an edge,
 * which then make up one native record. */
uint
pair_flags(uint index)
{
   if (index < AGX3_PLAN_ARGS.leaf_bound ||
       index - AGX3_PLAN_ARGS.leaf_bound >= internal_count)
      return 0u;
   uvec2 pair = children(index);
   uint t0 = triangle_index(pair[0]), t1 = triangle_index(pair[1]);
   if (t0 == ~0u || t1 == ~0u)
      return 0u;
   REF(vk_ir_triangle_node) a = REF(vk_ir_triangle_node)(
      OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(t0)));
   REF(vk_ir_triangle_node) b = REF(vk_ir_triangle_node)(
      OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(t1)));
   uint flags = DEREF(a).geometry_id_and_flags;
   if (flags != DEREF(b).geometry_id_and_flags)
      return 0u;
   vec3 va[3], vb[3];
   for (uint v = 0; v < 3; ++v)
      for (uint c = 0; c < 3; ++c) {
         va[v][c] = DEREF(a).coords[v][c];
         vb[v][c] = DEREF(b).coords[v][c];
      }
   if (hk_agx3_shared_edge(va, vb) == ~0u)
      return 0u;
   return HK_AGX3_PLAN_PAIR |
          ((flags & VK_GEOMETRY_OPAQUE) != 0 ? HK_AGX3_PLAN_PAIR_OPAQUE : 0u);
}

bool
is_pair(uint index)
{
   return index >= AGX3_PLAN_ARGS.leaf_bound &&
          (DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, index)).depth &
           HK_AGX3_PLAN_PAIR) != 0u;
}
#endif

/* Every non-empty pending subtree contributes at least one leaf. Stop at
 * nine before pushing past this bounded stack, even for a skewed tree. A
 * triangle pair counts as one leaf. */
uint
count_leaves(uint start)
{
   uint stack[9], sp = 1, count = 0, steps = 0;
   uint opacity = 0;
   bool have_opacity = false;
   stack[0] = start;
   while (sp != 0) {
      if (++steps > AGX3_PLAN_ARGS.node_bound * 2u) {
         fail(HK_AGX3_PLAN_CYCLE);
         return 0;
      }
      uint index = stack[--sp];
#ifdef HK_AGX3_PLAN_PAIRS
      uint flags = index < AGX3_PLAN_ARGS.leaf_bound ? 0u
         : DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, index)).depth;
      if (index < AGX3_PLAN_ARGS.leaf_bound || (flags & HK_AGX3_PLAN_PAIR) != 0u) {
         if (AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            uint current_opacity;
            if (index < AGX3_PLAN_ARGS.leaf_bound) {
               REF(vk_ir_triangle_node) triangle = REF(vk_ir_triangle_node)(
                  OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(index)));
               current_opacity = DEREF(triangle).geometry_id_and_flags &
                                 VK_GEOMETRY_OPAQUE;
            } else {
               current_opacity = (flags & HK_AGX3_PLAN_PAIR_OPAQUE) != 0u
                                    ? VK_GEOMETRY_OPAQUE : 0u;
            }
#else
      if (index < AGX3_PLAN_ARGS.leaf_bound) {
         if (AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            REF(vk_ir_triangle_node) triangle = REF(vk_ir_triangle_node)(
               OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(index)));
            uint current_opacity = DEREF(triangle).geometry_id_and_flags &
                                   VK_GEOMETRY_OPAQUE;
#endif
            /* One native leaf cluster shares its child triangle type.
             * A mixed-opacity subtree must remain an inner node; retain
             * normal spatial collapse for every homogeneous subtree. */
            if (have_opacity && opacity != current_opacity)
               return 9;
            opacity = current_opacity;
            have_opacity = true;
         }
         ++count;
      } else {
         uvec2 pair = children(index);
         for (uint c = 0; c < 2; c++)
            if (pair[c] != VK_BVH_INVALID_NODE) {
               uint child = source_index(pair[c]);
               if (child == 0xffffffffu)
                  return 0;
               if (sp == 9)
                  return 9;
               stack[sp++] = child;
            }
      }
      if (count + sp > 8)
         return 9;
   }
   return count;
}

void
initialize(uint id)
{
   if (id < AGX3_PLAN_ARGS.node_bound) {
      hk_agx3_plan_node node;
      node.parent = 0xffffffffu;
      node.leaf_count = id < AGX3_PLAN_ARGS.leaf_bound ? 1u : 0u;
#ifdef HK_AGX3_PLAN_PAIRS
      node.depth = pair_flags(id);
      node.job = 0xffffffffu;
      if (node.depth != 0u)
         node.leaf_count = 1;
#else
      node.depth = 0;
      node.job = 0xffffffffu;
#endif
      DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id)) = node;
   }
   if (id == 0) {
      hk_agx3_plan_state state;
      state.job_count = 0;
      state.groups_y = 1;
      state.groups_z = 1;
      state.next_slot = 3;
      state.depth = 0;
      state.error = 0;
      state.saved_offsets = 0;
      state.saved_positions = 0;
      if (DEREF(AGX3_PLAN_ARGS.header).active_leaf_count == 0)
         state.error = HK_AGX3_PLAN_EMPTY;
      if (AGX3_PLAN_ARGS.leaf_bound > AGX3_PLAN_ARGS.node_bound || AGX3_PLAN_ARGS.node_bound > 0xffffffffu / 64u ||
          (DEREF(AGX3_PLAN_ARGS.header).active_leaf_count != 0 &&
           (DEREF(AGX3_PLAN_ARGS.header).ir_internal_node_count == 0 ||
            DEREF(AGX3_PLAN_ARGS.header).ir_internal_node_count >
               AGX3_PLAN_ARGS.node_bound - AGX3_PLAN_ARGS.leaf_bound)) ||
          DEREF(AGX3_PLAN_ARGS.header).active_leaf_count > AGX3_PLAN_ARGS.leaf_bound ||
          AGX3_PLAN_ARGS.root_offset != AGX3_PLAN_ARGS.leaf_bound * leaf_size() ||
          AGX3_PLAN_ARGS.output_bvh == 0 || (AGX3_PLAN_ARGS.output_bvh & 63ul) != 0 ||
          AGX3_PLAN_ARGS.output_bvh >= (1ul << 48) ||
          uint64_t(AGX3_PLAN_ARGS.slot_bound) * 64ul + 184ul > (1ul << 48) - AGX3_PLAN_ARGS.output_bvh ||
          AGX3_PLAN_ARGS.slot_bound < 5)
         state.error |= HK_AGX3_PLAN_INDEX;
      DEREF(AGX3_PLAN_ARGS.state) = state;
   }
}

void
link_and_count(uint id)
{
   if (id < AGX3_PLAN_ARGS.leaf_bound ||
       id >= AGX3_PLAN_ARGS.leaf_bound + internal_count)
      return;
   uvec2 pair = children(id);
   for (uint c = 0; c < 2; c++)
      if (pair[c] != VK_BVH_INVALID_NODE) {
         uint child = source_index(pair[c]);
         if (child == 0xffffffffu)
            return;
         uint old = atomicCompSwap(
            DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, child)).parent,
            0xffffffffu, id);
         if (old != 0xffffffffu)
            fail(HK_AGX3_PLAN_SHAPE);
      }
#ifdef HK_AGX3_PLAN_PAIRS
   if (!is_pair(id))
#endif
      DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id)).leaf_count =
         count_leaves(id);
}

void
classify(uint id)
{
   if (id >= AGX3_PLAN_ARGS.node_bound)
      return;
   hk_agx3_plan_node node = DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id));
   uint root = root_index();
   if (id != root && node.parent == 0xffffffffu)
      return; /* inactive IR leaf */
   uint cursor = id, depth = 0;
   while (cursor != root) {
      if (depth++ >= AGX3_PLAN_ARGS.node_bound) {
         fail(HK_AGX3_PLAN_CYCLE);
         return;
      }
      cursor = DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, cursor)).parent;
      if (cursor >= AGX3_PLAN_ARGS.node_bound) {
         fail(HK_AGX3_PLAN_INDEX);
         return;
      }
   }
#ifdef HK_AGX3_PLAN_PAIRS
   DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id)).depth =
      depth | (node.depth & (HK_AGX3_PLAN_PAIR | HK_AGX3_PLAN_PAIR_OPAQUE));
#else
   DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id)).depth = depth;
#endif
   bool selected = id == root;
   if (!selected)
      selected = node.leaf_count <= 8
                    ? DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, node.parent))
                            .leaf_count > 8
                    : depth % 3 == 0;
   if (!selected)
      return;
   uint job = atomicAdd(DEREF(AGX3_PLAN_ARGS.state).job_count, 1);
   if (job >= AGX3_PLAN_ARGS.node_bound) {
      fail(HK_AGX3_PLAN_ALLOCATION);
      return;
   }
   DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id)).job = job;
   hk_agx3_leaf_job output_job;
   output_job.intermediate_bvh = AGX3_PLAN_ARGS.intermediate_bvh;
   output_job.output_bvh = AGX3_PLAN_ARGS.output_bvh;
   output_job.root_slot = id == root ? 0u : 0xffffffffu;
   output_job.child_slot = 0;
   output_job.child_count = 0;
   output_job.node_kind =
      node.leaf_count <= 8
         ? (AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR
               ? HK_AGX3_NODE_INSTANCES
               : AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR
                  ? HK_AGX3_NODE_AABBS : HK_AGX3_NODE_LEAF)
         : HK_AGX3_NODE_INNER;
   for (uint i = 0; i < 8; i++)
      output_job.triangle_offsets[i] = 0;
   DEREF(INDEX(hk_agx3_leaf_job, AGX3_PLAN_ARGS.jobs, job)) = output_job;
   atomicMax(DEREF(AGX3_PLAN_ARGS.state).depth, (depth + 2u) / 3u + 1u);
}

void
emit(uint id)
{
   if (id >= AGX3_PLAN_ARGS.node_bound)
      return;
   hk_agx3_plan_node node = DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, id));
   if (node.job == 0xffffffffu)
      return;
   bool leaf = node.leaf_count <= 8;
   uint stack[9], level[9], selected[8], sp = 1, count = 0, steps = 0;
   stack[0] = id;
   level[0] = 0;
   while (sp != 0) {
      if (++steps > AGX3_PLAN_ARGS.node_bound * 2u) {
         fail(HK_AGX3_PLAN_CYCLE);
         return;
      }
      uint current = stack[--sp], d = level[sp];
#ifdef HK_AGX3_PLAN_PAIRS
      bool end = leaf ? current < AGX3_PLAN_ARGS.leaf_bound || is_pair(current)
                      : current != id &&
#else
      bool end = leaf ? current < AGX3_PLAN_ARGS.leaf_bound
                      : current != id &&
#endif
                           (DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, current))
                                  .leaf_count <= 8 ||
                            d == 3);
      if (end) {
         if (count == 8) {
            fail(HK_AGX3_PLAN_SHAPE);
            return;
         }
         selected[count++] = current;
      } else {
         uvec2 pair = children(current);
         /* Stack is LIFO; visit the common builder's left child first. */
         for (int c = 1; c >= 0; c--)
            if (pair[c] != VK_BVH_INVALID_NODE) {
               uint child = source_index(pair[c]);
               if (child == 0xffffffffu || sp == 9) {
                  fail(HK_AGX3_PLAN_SHAPE);
                  return;
               }
               stack[sp] = child;
               level[sp++] = d + 1;
            }
      }
   }
   if (count == 0 || (leaf && count != node.leaf_count)) {
      fail(HK_AGX3_PLAN_SHAPE);
      return;
   }
   uint slots = leaf ? (AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR
                           ? 2u * count
                           : AGX3_PLAN_ARGS.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR
                              ? count : count + (count + 3u) / 4u) : count;
   uint first = atomicAdd(DEREF(AGX3_PLAN_ARGS.state).next_slot, slots);
   if (slots > AGX3_PLAN_ARGS.slot_bound || first > AGX3_PLAN_ARGS.slot_bound - slots) {
      fail(HK_AGX3_PLAN_ALLOCATION);
      return;
   }
   REF(hk_agx3_leaf_job) job = INDEX(hk_agx3_leaf_job, AGX3_PLAN_ARGS.jobs, node.job);
   DEREF(job).child_slot = first;
   DEREF(job).child_count = count;
   for (uint c = 0; c < count; c++) {
#ifdef HK_AGX3_PLAN_PAIRS
      DEREF(job).triangle_offsets[c] =
         source_offset(selected[c]) |
         (leaf && selected[c] >= AGX3_PLAN_ARGS.leaf_bound ? HK_AGX3_LEAF_PAIR_OFFSET : 0u);
#else
      DEREF(job).triangle_offsets[c] = source_offset(selected[c]);
#endif
      if (!leaf) {
         uint child_job =
            DEREF(INDEX(hk_agx3_plan_node, AGX3_PLAN_ARGS.nodes, selected[c])).job;
         if (child_job >= DEREF(AGX3_PLAN_ARGS.state).job_count) {
            fail(HK_AGX3_PLAN_SHAPE);
            return;
         }
         DEREF(INDEX(hk_agx3_leaf_job, AGX3_PLAN_ARGS.jobs, child_job)).root_slot =
            first + c;
      }
   }
}

void
finish(uint id)
{
   if (id != 0)
      return;
   hk_agx3_envelope_job job;
   uint root = DEREF(AGX3_PLAN_ARGS.state).error == 0 ? root_index() : 0;
   job.bounds =
      REF(vk_ir_node)(OFFSET(AGX3_PLAN_ARGS.intermediate_bvh, source_offset(root)));
   job.output_bvh = AGX3_PLAN_ARGS.output_bvh;
   job.end_slot = DEREF(AGX3_PLAN_ARGS.state).next_slot;
   job.primitive_count =
      DEREF(AGX3_PLAN_ARGS.state).error == 0 ? DEREF(AGX3_PLAN_ARGS.header).active_leaf_count : 0;
   job.node_count = DEREF(AGX3_PLAN_ARGS.state).job_count;
   job.depth = DEREF(AGX3_PLAN_ARGS.state).depth;
   job.opaque_build_word = DEREF(AGX3_PLAN_ARGS.state).error == HK_AGX3_PLAN_EMPTY
                             ? HK_AGX3_PLAN_EMPTY : 0;
   job.resource_id = AGX3_PLAN_ARGS.resource_id;
   job.geometry_type = AGX3_PLAN_ARGS.geometry_type;
   job.leaf_bound = AGX3_PLAN_ARGS.leaf_bound;
   DEREF(AGX3_PLAN_ARGS.envelope) = job;
   uint jobs = DEREF(AGX3_PLAN_ARGS.state).error != 0 ? 0u : DEREF(AGX3_PLAN_ARGS.state).job_count;
   if (DEREF(AGX3_PLAN_ARGS.state).error != 0)
      DEREF(AGX3_PLAN_ARGS.state).job_count = 0;
   uint64_t dispatch = uint64_t(AGX3_PLAN_ARGS.state) + uint64_t(HK_AGX3_PLAN_ENCODE_DISPATCH);
   DEREF(REF(uint32_t)(dispatch)) =
      (jobs + HK_AGX3_NODE_JOBS_PER_GROUP - 1u) / HK_AGX3_NODE_JOBS_PER_GROUP;
   DEREF(REF(uint32_t)(dispatch + 4ul)) = 1u;
   DEREF(REF(uint32_t)(dispatch + 8ul)) = 1u;
}

void
AGX3_PLAN_ENTRY()
{
   uint id = AGX3_PLAN_GLOBAL_ID.x;
   internal_count = DEREF(AGX3_PLAN_ARGS.header).ir_internal_node_count;
   if (AGX3_PLAN_ARGS.phase == 0) {
      initialize(id);
      return;
   }
   if (AGX3_PLAN_ARGS.phase == 4) {
      finish(id);
      return;
   }
   if (DEREF(AGX3_PLAN_ARGS.state).error != 0)
      return;
   if (AGX3_PLAN_ARGS.phase == 1)
      link_and_count(id);
   if (AGX3_PLAN_ARGS.phase == 2)
      classify(id);
   if (AGX3_PLAN_ARGS.phase == 3)
      emit(id);
}
