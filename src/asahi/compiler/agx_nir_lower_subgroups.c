/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/list.h"
#include "agx_nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

/* XXX: cribbed from nak, move to common */
static nir_def *
nir_udiv_round_up(nir_builder *b, nir_def *n, nir_def *d)
{
   return nir_udiv(b, nir_iadd(b, n, nir_iadd_imm(b, d, -1)), d);
}

struct agx_subgroup_lower_options {
   bool g17p;
   bool g17p_clustered_float;
};

/* Whether a clustered reduction takes the per-cluster branch in lower()
 * (full-cluster xor butterfly, partial-cluster serial gather): a scalar
 * reduce at a width the shuffles cover, over an aligned 8- or 16-lane
 * cluster.  The filter below must return false for exactly these so they
 * survive nir_lower_subgroups as intrinsics.
 */
static bool
agx_clustered_reduce_eligible(const nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_reduce ||
       !nir_intrinsic_has_cluster_size(intr))
      return false;

   unsigned cluster = nir_intrinsic_cluster_size(intr);
   if (cluster != 8 && cluster != 16)
      return false;

   return intr->def.num_components == 1 &&
          (intr->def.bit_size == 16 || intr->def.bit_size == 32);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_vote_any: {
      nir_def *ballot = nir_quad_ballot_agx(b, 16, intr->src[0].ssa);
      nir_def_rewrite_uses(&intr->def, nir_ine_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_quad_vote_all: {
      nir_def *ballot =
         nir_quad_ballot_agx(b, 16, nir_inot(b, intr->src[0].ssa));
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_elect: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, active_id, 0));
      return true;
   }

   case nir_intrinsic_first_invocation: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def *is_first = nir_ieq_imm(b, active_id, 0);
      nir_def *first_bit = nir_ballot(b, 1, 32, is_first);
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, first_bit));
      return true;
   }

   case nir_intrinsic_last_invocation: {
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, active_mask));
      return true;
   }

   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      /* The common lowering does:
       *
       *    vote_all(x == read_first(x))
       *
       * This is not optimal for AGX, since we have ufind_msb but not ctz, so
       * it's cheaper to read the last invocation than the first. So we do:
       *
       *    vote_all(x == read_last(x))
       *
       * implemented with lowered instructions as
       *
       *    ballot(x != broadcast(x, ffs(ballot(true)))) == 0
       */
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *active_bit = nir_ufind_msb(b, active_mask);
      nir_def *other = nir_read_invocation(b, intr->src[0].ssa, active_bit);
      nir_def *is_ne;

      if (intr->intrinsic == nir_intrinsic_vote_feq) {
         is_ne = nir_fneu(b, other, intr->src[0].ssa);
      } else {
         is_ne = nir_ine(b, other, intr->src[0].ssa);
      }

      nir_def *ballot = nir_ballot(b, 1, 32, is_ne);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_load_num_subgroups: {
      nir_def *workgroup_size = nir_load_workgroup_size(b);
      workgroup_size = nir_imul(b,
                                nir_imul(b, nir_channel(b, workgroup_size, 0),
                                         nir_channel(b, workgroup_size, 1)),
                                nir_channel(b, workgroup_size, 2));
      nir_def *subgroup_size = nir_imm_int(b, 32);
      nir_def *num_subgroups =
         nir_udiv_round_up(b, workgroup_size, subgroup_size);
      nir_def_rewrite_uses(&intr->def, num_subgroups);
      return true;
   }

   case nir_intrinsic_shuffle: {
      nir_def *data = intr->src[0].ssa;
      nir_def *target = intr->src[1].ssa;

      /* The hardware shuffle instruction chooses a single index within the
       * target quad to shuffle each source quad with. Consequently, the low
       * 2-bits of shuffle indices should not be quad divergent.  To implement
       * arbitrary shuffle, pull each low 2-bits index in the quad separately.
       */
      nir_def *quad_start = nir_iand_imm(b, target, 0x1c);
      nir_def *result = NULL;

      for (unsigned i = 0; i < 4; ++i) {
         nir_def *target_i = nir_iadd_imm(b, quad_start, i);
         nir_def *shuf = nir_read_invocation(b, data, target_i);

         if (result)
            result = nir_bcsel(b, nir_ieq(b, target, target_i), shuf, result);
         else
            result = shuf;
      }

      nir_def_rewrite_uses(&intr->def, result);
      return true;
   }

   case nir_intrinsic_inclusive_scan: {
      /* If we got here, we support the corresponding exclusive scan in
       * hardware, so just handle the last element.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      nir_def *data = intr->src[0].ssa;

      b->cursor = nir_after_instr(&intr->instr);
      intr->intrinsic = nir_intrinsic_exclusive_scan;
      nir_def *accum = nir_build_alu2(b, red_op, data, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, accum);
      return true;
   }

   case nir_intrinsic_reduce: {
      if (!agx_clustered_reduce_eligible(intr))
         return false;

      /* The common lowering wraps a full-subgroup butterfly in a
       * ballot==full branch whose partial-mask side walks the ballot one
       * buddy at a time; that masked walk costs hundreds of instructions
       * and dominates the binary.  An XOR with a delta below the cluster
       * size keeps the buddy inside the same aligned cluster, so a fixed
       * log2(cluster) butterfly covers the group whenever every lane in it
       * is active:
       *
       *    for i in {1 .. cluster / 2}:
       *       data = op(data, shuffle_xor(data, i))
       *
       * A partially active cluster cannot use it.  Inactive lanes execute
       * nothing, so their registers are stale and cannot be pre-charged
       * with the identity on the source side; guarding each merge on the
       * buddy's activity still loses values across mid-cluster holes
       * (active {0,1,2} of 8: lane 1 can only be reached by lane 2's data
       * through lane 0 with deltas 2-then-1, after step 1 already ran);
       * and sparse masks like {0,3} or {1,2,4} have no XOR path between
       * active lanes at all.  So branch per cluster -- the condition is
       * uniform within a cluster: full clusters take the butterfly and
       * partial ones gather the live lanes serially.  The gathers index
       * base+k with a constant k, whose low 2 bits are quad-uniform, so a
       * single read_invocation per lane suffices.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      unsigned cluster = nir_intrinsic_cluster_size(intr);
      nir_def *data = intr->src[0].ssa;

      nir_def *m = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *lane = nir_load_subgroup_invocation(b);
      nir_def *base = nir_iand_imm(b, lane, ~(cluster - 1));
      unsigned cmask_bits = BITFIELD_MASK(cluster);
      nir_def *cmask =
         nir_iand_imm(b, nir_ushr(b, m, base), cmask_bits);

      nir_if *nif = nir_push_if(b, nir_ieq_imm(b, cmask, cmask_bits));

      nir_def *acc = data;
      for (unsigned i = 1; i < cluster; i *= 2) {
         acc = nir_build_alu2(b, red_op, acc,
                              nir_shuffle_xor(b, acc, nir_imm_int(b, i)));
      }
      nir_def *fast = acc;

      nir_push_else(b, nif);

      /* -0.0 is the true identity for IEEE addition: x + -0.0 == x for
       * every x, whereas +0.0 would flip an all-(-0.0) cluster to +0.0.
       */
      nir_const_value ident = nir_alu_binop_identity(red_op, data->bit_size);
      if (red_op == nir_op_fadd) {
         assert(data->bit_size == 16 || data->bit_size == 32);
         if (data->bit_size == 16)
            ident.u16 = 0x8000;
         else
            ident.f32 = -0.0f;
      }

      acc = nir_build_imm(b, 1, data->bit_size, &ident);
      for (unsigned k = 0; k < cluster; ++k) {
         nir_def *j = nir_iadd_imm(b, base, k);
         nir_def *ok = nir_ine_imm(
            b, nir_iand_imm(b, nir_ushr(b, m, j), 1), 0);
         nir_def *v = nir_read_invocation(b, data, j);
         nir_def *add = nir_build_alu2(b, red_op, acc, v);
         acc = nir_bcsel(b, ok, add, acc);
      }

      nir_pop_if(b, nif);
      nir_def_rewrite_uses(&intr->def, nir_if_phi(b, fast, acc));
      return true;
   }

   case nir_intrinsic_ballot: {
      /* Optimize popcount(ballot(true)) to load_active_subgroup_count_agx().
       *
       * Not on G17P: Apple's compiler reads its active-count register (SR36)
       * only inside the helper-guarded region of a uniform atomic, and
       * compiles simd_sum(1u) to a reduction instead, so keep the ballot whose
       * lane membership every other subgroup operation shares.
       */
      const struct agx_subgroup_lower_options *agx_options = data;
      if (agx_options->g17p || !nir_src_is_const(intr->src[0]) ||
          !nir_src_as_bool(intr->src[0]) ||
          !list_is_singular(&intr->def.uses))
         return false;

      nir_src *use = list_first_entry(&intr->def.uses, nir_src, use_link);
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op != nir_op_bit_count)
         return false;

      nir_def_rewrite_uses(&alu->def,
                           nir_load_active_subgroup_count_agx(b, 32));
      return true;
   }

   default:
      return false;
   }
}



static bool
lower_subgroup_filter(const nir_intrinsic_instr *intr, const void *data)
{
   const struct agx_subgroup_lower_options *agx_options = data;
   unsigned cluster = 0;

   /* Use default behaviour for everything but scans */
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_reduce)
      return true;

   if (intr->def.num_components > 1 || intr->def.bit_size == 1)
      return true;

   /* Cluster-8/16 reductions take the guarded xor butterfly in lower();
    * they must stay intrinsics and reach it untouched.
    */
   if (agx_clustered_reduce_eligible(intr))
      return false;

   /* Hardware supports quad ops but no other support clustered reductions. */
   if (nir_intrinsic_has_cluster_size(intr)) {
      cluster = nir_intrinsic_cluster_size(intr);
      if (cluster && cluster != 4 && cluster < 32)
         return true;
   }

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imul:
      return true;

   case nir_op_fmin:
   case nir_op_fmax:
      /* Fresh G17P witnesses pin the native qRedfmin/qRedfmax forms. Keep
       * every other target and non-quad shape on the established common
       * lowering path. */
      if (agx_options->g17p_clustered_float && cluster == 4 &&
          intr->def.bit_size == 32)
         return false;
      return true;

   case nir_op_iadd:
   case nir_op_iand:
   case nir_op_ixor:
   case nir_op_ior:
      /* these have dedicated 64-bit lowering paths that use the 32-bit hardware
       * instructions so are likely better than the full lowering.
       */
      return false;

   default:
      /* otherwise, lower 64-bit, since the hw ops are at most 32-bit. */
      return intr->def.bit_size == 64;
   }
}

static bool
agx_nir_lower_subgroups_internal(nir_shader *s, bool g17p_clustered_float)
{
   const struct agx_subgroup_lower_options agx_options = {
      .g17p = g17p_clustered_float,
      .g17p_clustered_float = g17p_clustered_float,
   };

   /* First, do as much common lowering as we can */
   nir_lower_subgroups_options opts = {
      .filter = lower_subgroup_filter,
      .filter_data = &agx_options,
      .lower_read_first_invocation = true,
      .lower_inverse_ballot = true,
      .lower_to_scalar = true,
      .lower_relative_shuffle = false,
      .lower_rotate_to_shuffle = true,
      .lower_subgroup_masks = true,
      .lower_reduce = true,
      .lower_vote = true,
      .ballot_components = 1,
      .ballot_bit_size = 32,
      .subgroup_size = 32,
   };

   bool progress = nir_lower_subgroups(s, &opts);

   /* Then do AGX-only lowerings on top.  The clustered-reduce lowering may
    * wrap its butterfly in a per-cluster branch, so control flow metadata
    * cannot be preserved.
    */
   progress |= nir_shader_intrinsics_pass(s, lower, nir_metadata_none,
                                          (void *)&agx_options);

   return progress;
}

bool
agx_nir_lower_subgroups(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, false);
}

bool
agx_nir_lower_subgroups_g17p(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, true);
}
