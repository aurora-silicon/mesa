/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Body of agx3_refit.comp, for it and for shaders that run refit phases.
 * Accessor macros (AGX3_REFIT_*): ARGS, GLOBAL_ID, ENTRY.
 * Expects hk_build_interface.h and agx3_node.h; honours
 * HK_AGX3_REFIT_ONLY_PHASE.
 */
#ifndef AGX3_REFIT_ARGS
#define AGX3_REFIT_ARGS args
#endif
#ifndef AGX3_REFIT_GLOBAL_ID
#define AGX3_REFIT_GLOBAL_ID gl_GlobalInvocationID
#endif
#ifndef AGX3_REFIT_ENTRY
#define AGX3_REFIT_ENTRY main
#endif

uint
leaf_size()
{
   return AGX3_REFIT_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR
             ? SIZEOF(vk_ir_instance_node)
             : AGX3_REFIT_ARGS.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR
                ? SIZEOF(vk_ir_aabb_node) : SIZEOF(vk_ir_triangle_node);
}

/* The retained level table: counts, then fill cursors. */
uint64_t
level_table(uint64_t retained, uint jobs)
{
   return retained + uint64_t(HK_AGX3_RETAINED_LEVELS(jobs, AGX3_REFIT_ARGS.leaf_bound));
}

/* Index into the retained inner jobs by level of the first of `level`. */
uint
level_start(uint64_t table, uint level)
{
   uint start = 0;
   for (uint l = 0; l < level && l < HK_AGX3_REFIT_LEVELS; ++l)
      start += DEREF(INDEX(uint32_t, table, l));
   return start;
}

/* Plan node of the IR node at `offset` (an IR leaf or internal node). */
uint
node_index(uint offset)
{
   return offset < AGX3_REFIT_ARGS.root_offset
             ? offset / leaf_size()
             : AGX3_REFIT_ARGS.leaf_bound + (offset - AGX3_REFIT_ARGS.root_offset) / SIZEOF(vk_ir_box_node);
}

void
save(uint id)
{
   hk_agx3_plan_state plan = DEREF(REF(hk_agx3_plan_state)(AGX3_REFIT_ARGS.plan_state));
   uint jobs = plan.job_count;
   REF(hk_agx3_retained_state) state = REF(hk_agx3_retained_state)(AGX3_REFIT_ARGS.retained);
   uint64_t records = AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   uint64_t offsets = records + uint64_t(jobs) * uint64_t(HK_AGX3_RETAINED_JOB_BYTES);
   if (id == 0) {
      uint root = AGX3_REFIT_ARGS.leaf_bound + DEREF(AGX3_REFIT_ARGS.header).ir_internal_node_count - 1u;
      hk_agx3_retained_state saved;
      saved.job_count = jobs;
      saved.root_job = jobs != 0u && root < AGX3_REFIT_ARGS.node_bound
         ? DEREF(INDEX(hk_agx3_plan_node, AGX3_REFIT_ARGS.plan_nodes, root)).job : 0xffffffffu;
      saved.end_slot = plan.next_slot;
      saved.depth = plan.depth;
      saved.deepest_level = 0;
      saved.active_leaves = DEREF(AGX3_REFIT_ARGS.header).active_leaf_count;
      for (uint i = 0; i < 2; ++i)
         saved.reserved[i] = 0;
      DEREF(state) = saved;
      uint64_t table = level_table(AGX3_REFIT_ARGS.retained, jobs);
      for (uint i = 0; i < HK_AGX3_REFIT_LEVELS * 2u; ++i)
         DEREF(INDEX(uint32_t, table, i)) = 0u;
   }
   if (id < AGX3_REFIT_ARGS.leaf_bound)
      DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_SLOTS(jobs, AGX3_REFIT_ARGS.leaf_bound)),
                  id)) = ~0u;
   if (id >= jobs)
      return;
   hk_agx3_leaf_job job = DEREF(INDEX(hk_agx3_leaf_job, AGX3_REFIT_ARGS.plan_jobs, id));
   uint count = min(job.child_count, 8u);
   REF(hk_agx3_retained_job) record = INDEX(hk_agx3_retained_job, records, id);
   uint first;
   if (job.node_kind == HK_AGX3_NODE_INNER) {
      first = atomicAdd(DEREF(REF(hk_agx3_plan_state)(AGX3_REFIT_ARGS.plan_state)).saved_positions, count);
      for (uint c = 0; c < count; ++c) {
         hk_agx3_plan_node node = DEREF(INDEX(hk_agx3_plan_node, AGX3_REFIT_ARGS.plan_nodes,
                                              node_index(job.triangle_offsets[c])));
         uint child = node.job;
         if (child >= jobs)
            continue;
         REF(hk_agx3_retained_job) child_record = INDEX(hk_agx3_retained_job, records, child);
         DEREF(child_record).parent = id;
         DEREF(child_record).position = first + c;
         /* Inner jobs sit at depths of whole levels. */
         DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, child)) =
            (node.depth & ~(HK_AGX3_PLAN_PAIR | HK_AGX3_PLAN_PAIR_OPAQUE)) / 3u;
      }
   } else {
      first = atomicAdd(DEREF(REF(hk_agx3_plan_state)(AGX3_REFIT_ARGS.plan_state)).saved_offsets, count);
      for (uint c = 0; c < count; ++c)
         DEREF(INDEX(uint32_t, offsets, first + c)) = job.triangle_offsets[c];
   }
   DEREF(record).root_slot = job.root_slot;
   DEREF(record).child_slot = job.child_slot;
   DEREF(record).kind_count = job.child_count | (job.node_kind << 4);
   DEREF(record).first = first;
   if (job.root_slot == 0u) {
      DEREF(record).parent = 0xffffffffu;
      DEREF(record).position = jobs - 1u;
      DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, id)) = 0u;
   }
}

/* Give each inner job its level and count the inner jobs of each level;
 * give each triangle leaf its slot, the index of its child offset. */
void
count_levels(uint id)
{
   REF(hk_agx3_retained_state) state = REF(hk_agx3_retained_state)(AGX3_REFIT_ARGS.retained);
   uint jobs = DEREF(state).job_count;
   uint64_t records = AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   uint level = 0;
   bool inner = false;
   if (id < jobs) {
      REF(hk_agx3_retained_job) record = INDEX(hk_agx3_retained_job, records, id);
      uint kind_count = DEREF(record).kind_count;
      inner = (kind_count >> 4) == HK_AGX3_NODE_INNER;
      if ((kind_count >> 4) == HK_AGX3_NODE_LEAF) {
         uint first = DEREF(record).first;
         uint64_t offsets = AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_OFFSETS(jobs));
         uint64_t slots = AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_SLOTS(jobs, AGX3_REFIT_ARGS.leaf_bound));
         for (uint c = 0; c < min(kind_count & 15u, 8u); ++c) {
            uint leaf = DEREF(INDEX(uint32_t, offsets, first + c)) / leaf_size();
            if (leaf < AGX3_REFIT_ARGS.leaf_bound)
               DEREF(INDEX(uint32_t, slots, leaf)) = first + c;
         }
      }
      if (inner) {
         level = DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, id));
         DEREF(record).kind_count = kind_count | (level << 8);
         if (level < HK_AGX3_REFIT_LEVELS)
            atomicAdd(DEREF(INDEX(uint32_t, level_table(AGX3_REFIT_ARGS.retained, jobs), level)), 1u);
      }
   }
   uint deepest = subgroupMax(level);
   if (subgroupElect() && deepest != 0u)
      atomicMax(DEREF(state).deepest_level, deepest);
}

/* List the inner jobs of the first HK_AGX3_REFIT_LEVELS levels by level. */
void
list_levels(uint id)
{
   uint jobs = DEREF(REF(hk_agx3_retained_state)(AGX3_REFIT_ARGS.retained)).job_count;
   if (id >= jobs)
      return;
   uint64_t records = AGX3_REFIT_ARGS.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   uint kind_count = DEREF(INDEX(hk_agx3_retained_job, records, id)).kind_count;
   uint level = kind_count >> 8;
   if (((kind_count >> 4) & 15u) != HK_AGX3_NODE_INNER || level >= HK_AGX3_REFIT_LEVELS)
      return;
   uint64_t table = level_table(AGX3_REFIT_ARGS.retained, jobs);
   uint index = level_start(table, level) +
      atomicAdd(DEREF(INDEX(uint32_t, table, HK_AGX3_REFIT_LEVELS + level)), 1u);
   uint64_t list = table + uint64_t(HK_AGX3_RETAINED_LEVEL_BYTES);
   DEREF(INDEX(uint32_t, list, index)) = id;
}

void
prepare(uint id)
{
   uint64_t source = AGX3_REFIT_ARGS.retained;
   if (AGX3_REFIT_ARGS.metadata != 0ul) {
      libagx_gen1_as_metadata live = DEREF(REF(libagx_gen1_as_metadata)(AGX3_REFIT_ARGS.metadata));
      source = live.native_base + live.retained_offset;
   }
   hk_agx3_retained_state saved = DEREF(REF(hk_agx3_retained_state)(source));
   uint jobs = saved.job_count;
   if (AGX3_REFIT_ARGS.copy != 0u) {
      /* An update into another structure keeps it updatable: copy the state,
       * the records and every child offset the leaf jobs may use. */
      uint words = HK_AGX3_RETAINED_BYTES(jobs, AGX3_REFIT_ARGS.leaf_bound) / 4u;
      if (id < words)
         DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.retained, id)) = DEREF(INDEX(uint32_t, source, id));
   }
   if (id < jobs)
      DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.counters, id)) = 0u;

   bool instances = AGX3_REFIT_ARGS.geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR;
   uint64_t transforms = uint64_t(saved.end_slot) * 64ul;
   uint64_t pointers = transforms + 64ul * uint64_t(AGX3_REFIT_ARGS.leaf_bound);
   /* Instance encoding fills the pointer of every active instance. */
   if (instances && jobs != 0u && id < AGX3_REFIT_ARGS.leaf_bound) {
      uint64_t entry = AGX3_REFIT_ARGS.output_bvh + pointers + uint64_t(id) * 16ul;
      DEREF(REF(uint64_t)(entry)) = 0ul;
      DEREF(REF(uint64_t)(entry + 8ul)) = 0ul;
   }
   if (id != 0)
      return;
   hk_agx3_refit_state refit;
   refit.dispatch[0] = (jobs + HK_AGX3_NODE_JOBS_PER_GROUP - 1u) / HK_AGX3_NODE_JOBS_PER_GROUP;
   refit.dispatch[1] = 1u;
   refit.dispatch[2] = 1u;
   refit.job_count = jobs;
   refit.retained = AGX3_REFIT_ARGS.copy != 0u ? AGX3_REFIT_ARGS.retained : source;
   /* Propagation only for inner jobs past the levels. */
   refit.propagate[0] = saved.deepest_level >= AGX3_REFIT_ARGS.level
      ? (jobs + HK_AGX3_REFIT_GROUP_SIZE - 1u) / HK_AGX3_REFIT_GROUP_SIZE : 0u;
   refit.propagate[1] = 1u;
   refit.propagate[2] = 1u;
   refit.end_slot = saved.end_slot;
   DEREF(REF(hk_agx3_refit_state)(AGX3_REFIT_ARGS.refit)) = refit;
   /* Workgroups of HK_AGX3_REFIT_GROUP_SIZE / 8 jobs for each level the
    * update encodes by level, all from this thread: a small structure runs
    * fewer threads than there are levels, and a level left unwritten would
    * dispatch whatever the scratch held. */
   if (AGX3_REFIT_ARGS.levels != 0ul) {
      uint64_t table = level_table(source, jobs);
      const uint per_group = HK_AGX3_REFIT_GROUP_SIZE / 8u;
      for (uint l = 0; l < AGX3_REFIT_ARGS.level && l < HK_AGX3_REFIT_LEVELS; ++l) {
         uint count = min(DEREF(INDEX(uint32_t, table, l)), jobs);
         uint64_t dispatch = AGX3_REFIT_ARGS.levels + uint64_t(l) * 12ul;
         store32(dispatch, (count + per_group - 1u) / per_group);
         store32(dispatch + 4ul, 1u);
         store32(dispatch + 8ul, 1u);
      }
   }

   uint active_leaves = AGX3_REFIT_ARGS.no_leaves != 0u ? saved.active_leaves
                                             : DEREF(AGX3_REFIT_ARGS.header).active_leaf_count;
   bool empty = jobs == 0u || active_leaves == 0u;
   hk_agx3_envelope_job job;
   job.bounds = REF(vk_ir_node)(AGX3_REFIT_ARGS.bounds + uint64_t(jobs != 0u ? jobs - 1u : 0u) *
                                uint64_t(SIZEOF(vk_aabb)));
   job.output_bvh = AGX3_REFIT_ARGS.output_bvh;
   job.end_slot = saved.end_slot;
   job.primitive_count = empty ? 0u : active_leaves;
   job.node_count = jobs;
   job.depth = saved.depth;
   job.opaque_build_word = empty ? HK_AGX3_PLAN_EMPTY : 0u;
   job.resource_id = AGX3_REFIT_ARGS.resource_id;
   job.geometry_type = AGX3_REFIT_ARGS.geometry_type;
   job.leaf_bound = AGX3_REFIT_ARGS.leaf_bound;
   DEREF(REF(hk_agx3_envelope_job)(AGX3_REFIT_ARGS.envelope)) = job;
}

/* From leaf job `id`, whose bounds refit encoding left at its position:
 * arrive at the parent; while this was the parent's last child, encode the
 * parent's node from its children's bounds, leave the parent's bounds at
 * its position and arrive at its parent in turn. The arrival orders the
 * bounds stores of every child before the reads of the last. Parents of
 * the levels encoded by level are left to those. */
void
propagate(uint id)
{
   hk_agx3_refit_state refit = DEREF(REF(hk_agx3_refit_state)(AGX3_REFIT_ARGS.refit));
   if (id >= refit.job_count)
      return;
   uint64_t records = refit.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   hk_agx3_retained_job leaf = DEREF(INDEX(hk_agx3_retained_job, records, id));
   if (((leaf.kind_count >> 4) & 15u) == HK_AGX3_NODE_INNER)
      return;
   uint parent = leaf.parent;
   for (uint steps = 0; parent != 0xffffffffu && steps < 64u; ++steps) {
      hk_agx3_retained_job job = DEREF(INDEX(hk_agx3_retained_job, records, parent));
      if ((job.kind_count >> 8) < AGX3_REFIT_ARGS.level)
         return;
      uint count = job.kind_count & 15u;
      uint arrived = atomicAdd(DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.counters, parent)), 1u,
         gl_ScopeDevice, gl_StorageSemanticsBuffer,
         gl_SemanticsAcquireRelease | gl_SemanticsMakeAvailable | gl_SemanticsMakeVisible);
      if (arrived + 1u != count)
         return;
      vec3 child_lo[8], child_hi[8];
      vec3 lo = vec3(INFINITY), hi = vec3(-INFINITY);
      for (uint c = 0; c < count && c < 8u; ++c) {
         child_lo[c] = DEREF(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.first + c)).min;
         child_hi[c] = DEREF(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.first + c)).max;
         lo = min(lo, child_lo[c]);
         hi = max(hi, child_hi[c]);
      }
      vec3 base;
      uvec3 exponent;
      bool ok = count >= 1u && count <= 8u && fit_root(lo, hi, base, exponent);
      uint64_t words[8];
      for (uint t = 0; t < 8u && ok; ++t)
         words[t] = root_word(t, count, HK_AGX3_NODE_INNER, 0u, job.child_slot,
                              AGX3_REFIT_ARGS.slot_bias, base, exponent, child_lo[t], child_hi[t], ok);
      if (ok) {
         uint64_t node = AGX3_REFIT_ARGS.output_bvh + uint64_t(job.root_slot) * 64ul;
         for (uint t = 0; t < 8u; ++t)
            store_word64(node + uint64_t(t) * 8ul, words[t]);
      } else {
         lo = hi = HK_AGX3_FAILED_BOUNDS;
      }
      store_aabb(uint64_t(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.position)), lo, hi);
      DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, parent)) = ok ? HK_AGX3_LEAF_SUCCESS : HK_AGX3_LEAF_RANGE;
      parent = job.parent;
   }
}

/* Inner jobs of level args.level, whose children are encoded: eight lanes
 * per job, a lane per child for its bounds and root word. Every lane
 * reaches every subgroup operation. A job whose children's bounds do not
 * fit, as when a child failed, fails in turn. */
void
inner(uint id)
{
   uint slot = id / 8u, lane = id % 8u;
   hk_agx3_refit_state refit = DEREF(REF(hk_agx3_refit_state)(AGX3_REFIT_ARGS.refit));
   uint64_t records = refit.retained + uint64_t(HK_AGX3_RETAINED_STATE_BYTES);
   uint64_t table = level_table(refit.retained, refit.job_count);
   bool live = slot < min(DEREF(INDEX(uint32_t, table, AGX3_REFIT_ARGS.level)), refit.job_count);
   uint index = 0, count = 0;
   hk_agx3_retained_job job;
   vec3 my_lo = vec3(INFINITY), my_hi = vec3(-INFINITY);
   if (live) {
      uint64_t list = table + uint64_t(HK_AGX3_RETAINED_LEVEL_BYTES);
      index = DEREF(INDEX(uint32_t, list, level_start(table, AGX3_REFIT_ARGS.level) + slot));
      job = DEREF(INDEX(hk_agx3_retained_job, records, index));
      count = job.kind_count & 15u;
      if (lane < count) {
         my_lo = DEREF(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.first + lane)).min;
         my_hi = DEREF(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.first + lane)).max;
      }
   }
   vec3 lo = subgroupClusteredMin(my_lo, 8u);
   vec3 hi = subgroupClusteredMax(my_hi, 8u);
   vec3 base;
   uvec3 exponent;
   bool ok = live && count >= 1u && count <= 8u && fit_root(lo, hi, base, exponent);
   uint64_t word = 0ul;
   if (ok)
      word = root_word(lane, count, HK_AGX3_NODE_INNER, 0u, job.child_slot, AGX3_REFIT_ARGS.slot_bias,
                       base, exponent, my_lo, my_hi, ok);
   bool bad = subgroupClusteredOr(ok ? 0u : 1u, 8u) != 0u;
   if (!live)
      return;
   if (bad) {
      if (lane == 0u) {
         DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, index)) = HK_AGX3_LEAF_RANGE;
         store_aabb(uint64_t(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.position)),
                    HK_AGX3_FAILED_BOUNDS, HK_AGX3_FAILED_BOUNDS);
      }
      return;
   }
   store_word64(AGX3_REFIT_ARGS.output_bvh + uint64_t(job.root_slot) * 64ul + uint64_t(lane) * 8ul, word);
   if (lane == 0u) {
      store_aabb(uint64_t(INDEX(vk_aabb, AGX3_REFIT_ARGS.bounds, job.position)), lo, hi);
      DEREF(INDEX(uint32_t, AGX3_REFIT_ARGS.status, index)) = HK_AGX3_LEAF_SUCCESS;
   }
}

void
AGX3_REFIT_ENTRY()
{
   uint id = AGX3_REFIT_GLOBAL_ID.x;
   /* Built once per update phase that runs per level or per job
    * (HK_AGX3_REFIT_ONLY_PHASE), so those carry only their own registers,
    * and once for the other phases. */
#ifdef HK_AGX3_REFIT_ONLY_PHASE
   const uint phase = HK_AGX3_REFIT_ONLY_PHASE;
#else
   const uint phase = AGX3_REFIT_ARGS.phase;
#endif
   if (phase == HK_AGX3_REFIT_SAVE)
      save(id);
   else if (phase == HK_AGX3_REFIT_PREPARE)
      prepare(id);
   else if (phase == HK_AGX3_REFIT_PROPAGATE)
      propagate(id);
   else if (phase == HK_AGX3_REFIT_COUNT_LEVELS)
      count_levels(id);
   else if (phase == HK_AGX3_REFIT_LIST_LEVELS)
      list_levels(id);
   else
      inner(id);
}
