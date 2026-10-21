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

/* Whether a scan is handled by the dense-guarded lowering in lower(): a
 * scalar scan at a width the shuffles cover, for an operation the
 * hardware prefix op does not implement over a partial subgroup.  The
 * filter below must return false for exactly these.
 *
 * 64-bit scans emit 64-bit combines and indexed reads; nir_lower_int64
 * (which runs after this pass and admits the 64-bit shuffle/intrinsic
 * splits in AGX's option mask) expands them into 32-bit pieces, so the
 * pointer-doubling chain itself needs no widening.  Integer min/max has
 * a prefix form through 32 bits and stays an intrinsic below 64; iadd
 * and the bitwise ops keep their dedicated 64-bit splittings, which are
 * cheaper than a dense sweep.
 *
 * 32-bit float min/max stays listed here so the common lowering leaves
 * it an intrinsic; on G17P lower() then scans the order-equivalent
 * integer key with the signed min/max prefix op instead (see
 * agx_f32_minmax_key), which is far cheaper than the dense sweep.
 */
static bool
agx_scan_eligible(const nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan)
      return false;

   if (intr->def.num_components != 1)
      return false;

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imul:
   case nir_op_fmin:
   case nir_op_fmax:
      return intr->def.bit_size == 16 || intr->def.bit_size == 32 ||
             intr->def.bit_size == 64;
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_imin:
   case nir_op_imax:
      return intr->def.bit_size == 64;
   default:
      return false;
   }
}

/* Order isomorphism between a float's bit pattern and signed integers,
 * and its own inverse: fold the sign bit into a magnitude mask so that
 * negative values sort below non-negative ones in signed order:
 *
 *    key = bits ^ ((bits >>s 31) & 0x7fffffff)
 *
 * Non-negative inputs (b < 0x80000000) keep their bits and stay positive
 * keys; negative inputs get their low 31 bits flipped and stay negative
 * keys, with -NaN at INT_MIN and -0.0 at -1, so signed min/max on the
 * keys is float min/max on the values.
 *
 * G17P's hardware prefix op covers sums and products only -- Apple's
 * compiler lowers float min/max scans to a shift-up sweep, whose fixed
 * distances read stale registers in inactive lanes (measured on M2 Pro:
 * simdshftup indexes raw lanes, not the active-lane ordering).  But the
 * signed integer min/max DOES have the prefix form (f53, the same
 * operand the witnessed integer-add prefix carries), so a float min/max
 * scan becomes three ALU ops to key, one prefix op, and three to
 * decode -- ~10 instructions where the dense pointer-doubling sweep
 * costs ~80.  Hardware-verified on G14S that the integer min/max prefix
 * scans the active-lane ordering under a partial mask and reports the
 * per-op identity for lanes with no active predecessor.
 */
static nir_def *
agx_f32_minmax_key(nir_builder *b, nir_def *x)
{
   return nir_ixor(b, x,
                   nir_iand_imm(b, nir_ishr_imm(b, x, 31), 0x7fffffff));
}

/* Shuffle read of data[index] for a per-lane index.
 *
 * The hardware shuffle instruction chooses a single index within the
 * target quad to shuffle each source quad with: the low two index bits
 * are accumulated across the quad, so they must not be quad divergent.
 * A raw read_invocation with an arbitrary per-lane index therefore reads
 * the wrong lane -- this is what broke the first version of the scan
 * lowering on hardware.  To implement an arbitrary index, pull each low
 * 2-bits index in the quad separately, as with nir_intrinsic_shuffle.
 */
static nir_def *
agx_subgroup_read(nir_builder *b, nir_def *data, nir_def *index)
{
   nir_def *quad_start = nir_iand_imm(b, index, 0x1c);
   nir_def *result = NULL;

   for (unsigned i = 0; i < 4; ++i) {
      nir_def *target_i = nir_iadd_imm(b, quad_start, i);
      nir_def *shuf = nir_read_invocation(b, data, target_i);

      if (result)
         result = nir_bcsel(b, nir_ieq(b, index, target_i), shuf, result);
      else
         result = shuf;
   }

   return result;
}

/* Exclusive signed-max prefix of `data`, emitted as the intrinsic so isel
 * reaches the hardware prefix op: 32-bit integer min/max has a prefix
 * encoding (f53 on sRedsmin/sRedsmax), hardware-verified to scan the
 * active-lane ordering under a partial mask and to report the per-op
 * identity -- INT_MIN here -- for lanes with no active predecessor.
 * The intrinsic survives this pass: agx_scan_eligible admits 32-bit imax
 * scans only at 64 bits, so it reaches isel untouched.
 */
static nir_def *
agx_exscan_smax(nir_builder *b, nir_def *data)
{
   nir_intrinsic_instr *scan =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_exclusive_scan);
   scan->num_components = 1;
   scan->src[0] = nir_src_for_ssa(data);
   nir_intrinsic_set_reduction_op(scan, nir_op_imax);
   nir_def_init(&scan->instr, &scan->def, 1, 32);
   nir_builder_instr_insert(b, &scan->instr);
   return &scan->def;
}

/* Gathered composition of a frontier vector with itself: sq[l] = f[f[l]],
 * i.e. twice as many chain steps back in one shuffle-read.  A dead
 * frontier (INT_MIN) clamps to lane 31 and could surface a live value, so
 * propagate the marker explicitly -- same guard the old per-step chain
 * carried.
 */
static nir_def *
agx_frontier_square(nir_builder *b, nir_def *f)
{
   nir_def *sq = agx_subgroup_read(b, f, nir_umin_imm(b, f, 31));
   return nir_bcsel(b, nir_ult_imm(b, f, 32), sq, f);
}

/* Subgroup scan over the active-lane predecessor chain.
 *
 * The common lowering guards a full-subgroup Kogge-Stone sweep with a
 * ballot==full branch whose partial-mask side walks the ballot one buddy
 * at a time; that masked walk costs hundreds of instructions.  But a
 * partially-active subgroup has stale registers in its inactive lanes,
 * not values a shuffle may read, so simply guarding each fixed-distance
 * buddy drops every contribution that lies below an inactive lane.  An
 * identity fill cannot fix that: inactive lanes never execute the fill.
 *
 * Instead, each lane finds its closest active predecessor once, and the
 * scan runs as pointer doubling down that chain: after step k each lane
 * has folded in its first 2^k predecessors and its frontier is the
 * 2^k-th predecessor.  Five steps cover a 32-lane subgroup and only
 * active lanes' registers are ever consumed.
 *
 * The chain itself is computed without touching the ballot: the closest
 * active predecessor of every lane is the exclusive signed-max prefix of
 * the lane indices, and scanning the resulting predecessor vector again
 * composes one more chain link (the prefix evaluates the vector at the
 * first predecessor, so a scan of f_k yields f_{k+1}).  The fold consumes
 * the frontiers at chain positions 1, 2, 4, 8 and 16: the first four
 * cost eight prefix ops between them, and f16 = f8[f8] is the one
 * squaring that beats eight more scans.  All of it is mask-only data,
 * so the whole apparatus commons up across scans sharing a block.
 */
static nir_def *
agx_subgroup_scan(nir_builder *b, nir_op red_op, nir_def *data,
                  bool inclusive)
{
   nir_def *lane = nir_load_subgroup_invocation(b);

   /* The frontiers the fold consumes: f1, f2, f4, f8 -- each a single
    * prefix op per extra chain link -- and f16 = f8[f8], the one squaring
    * that beats the next eight prefix ops.
    */
   nir_def *f[5];
   nir_def *link = agx_exscan_smax(b, lane);
   f[0] = link;
   for (unsigned i = 2; i <= 8; ++i) {
      link = agx_exscan_smax(b, link);
      if (i == 2)
         f[1] = link;
      else if (i == 4)
         f[2] = link;
   }
   f[3] = link;
   f[4] = agx_frontier_square(b, link);

   nir_def *v = data;
   for (unsigned i = 0; i < 5; ++i) {
      nir_def *ok = nir_ult_imm(b, f[i], 32);
      nir_def *idx = nir_umin_imm(b, f[i], 31);
      nir_def *buddy = agx_subgroup_read(b, v, idx);
      v = nir_bcsel(b, ok, nir_build_alu2(b, red_op, v, buddy), v);
   }

   if (inclusive)
      return v;

   /* The exclusive result is the inclusive value held by the closest
    * active predecessor, or the identity where there is none.
    */
   nir_def *ok = nir_ult_imm(b, f[0], 32);
   nir_def *idx = nir_umin_imm(b, f[0], 31);
   nir_def *buddy = agx_subgroup_read(b, v, idx);
   nir_const_value ident = nir_alu_binop_identity(red_op, data->bit_size);
   nir_def *identity = nir_build_imm(b, 1, data->bit_size, &ident);
   return nir_bcsel(b, ok, buddy, identity);
}

/* Whether the hardware quad reduce implements `op` at `bit_size` -- the
 * same shapes the subgroup filter admits to isel: everything but integer
 * multiply has a quad form, and the float min/max encodings are only
 * witnessed on G17P.  Ops without a quad form take the explicit
 * intra-quad combine steps in lower(), which work for any reduction op.
 */
static bool
agx_quad_reduce_supported(nir_op op, unsigned bit_size, bool g17p)
{
   switch (op) {
   case nir_op_iadd:
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      return true;
   case nir_op_fmin:
   case nir_op_fmax:
      return g17p && bit_size == 32;
   default:
      return false;
   }
}

/* Whether a clustered reduction takes the representative-read lowering
 * in lower() (guarded pair combine, or quad base plus per-block sibling
 * reads): a scalar reduce at a width the shuffles cover, over an aligned
 * 2-, 8- or 16-lane cluster, or a 4-lane cluster of an op the hardware
 * quad reduce does not implement -- that shape is exactly the two
 * intra-quad combine steps.  Cluster-4 reduces of quad-capable ops go
 * to the hardware quad reduce through the ordinary intrinsic path
 * instead.  The filter below must return false for exactly these so
 * they survive nir_lower_subgroups as intrinsics.
 */
static bool
agx_clustered_reduce_eligible(const nir_intrinsic_instr *intr, bool g17p)
{
   if (intr->intrinsic != nir_intrinsic_reduce ||
       !nir_intrinsic_has_cluster_size(intr))
      return false;

   unsigned cluster = nir_intrinsic_cluster_size(intr);
   if (cluster != 2 && cluster != 4 && cluster != 8 && cluster != 16)
      return false;

   if (intr->def.num_components != 1 ||
       (intr->def.bit_size != 16 && intr->def.bit_size != 32))
      return false;

   return cluster != 4 ||
          !agx_quad_reduce_supported(nir_intrinsic_reduction_op(intr),
                                     intr->def.bit_size, g17p);
}

/* Guarded xor-combine at distance 1: fold the pair buddy's value into
 * `v` when lane^1 is live, else keep `v`.  The shuffle itself is
 * unconditional -- an inactive buddy's register is stale, not
 * unreadable -- and the ballot guard drops its contribution, so the
 * step is exact under any activity mask.
 */
static nir_def *
agx_pair_combine(nir_builder *b, nir_op op, nir_def *m, nir_def *lane,
                 nir_def *v)
{
   nir_def *live = nir_ine_imm(
      b, nir_iand_imm(b, nir_ushr(b, m, nir_ixor(b, lane, nir_imm_int(b, 1))),
                      1),
      0);
   return nir_bcsel(b, live,
                    nir_build_alu2(b, op, v,
                                   nir_shuffle_xor(b, v, nir_imm_int(b, 1))),
                    v);
}

/* Whether a 64-bit min/max reduction takes the two-phase hi/lo lowering
 * in lower(): a scalar reduce over a subgroup shape the hardware reduce
 * covers natively (whole subgroup or quad cluster -- the clustered
 * lowering above owns 8/16, and those are the only shapes isel admits).
 * A 64-bit minmax is decided by the high word first; only lanes whose
 * high word ties the winner put their low word into the second reduce:
 *
 *    w_hi = reduce(hi, op)
 *    lo'  = bcsel(ieq(hi, w_hi), lo, identity)
 *    w_lo = reduce(lo', unsigned op)
 *    res  = pack64(w_hi, w_lo)
 *
 * The low word always compares unsigned, so the identity masking losing
 * lanes is the unsigned one -- ~0 for a min, 0 for a max -- regardless of
 * the signedness the high-word reduce used.  Masking is purely lane-local;
 * inactive lanes contribute nothing because the hardware reduce skips
 * them, the same assumption the native 32-bit reduce path makes.
 */
static bool
agx_minmax64_reduce_eligible(const nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_reduce ||
       !nir_intrinsic_has_cluster_size(intr))
      return false;

   unsigned cluster = nir_intrinsic_cluster_size(intr);
   if (cluster != 0 && cluster != 4 && cluster < 32)
      return false;

   if (intr->def.num_components != 1 || intr->def.bit_size != 64)
      return false;

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_imin:
   case nir_op_imax:
      return true;
   default:
      return false;
   }
}

/* Emit a scalar reduce intrinsic over `data`, preserving the cluster
 * shape so isel picks the same quad/full-subgroup form it would for a
 * native-width reduce.
 */
static nir_def *
agx_subgroup_reduce(nir_builder *b, nir_op op, unsigned cluster,
                    nir_def *data)
{
   nir_intrinsic_instr *red =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_reduce);
   red->num_components = data->num_components;
   red->src[0] = nir_src_for_ssa(data);
   nir_intrinsic_set_reduction_op(red, op);
   nir_intrinsic_set_cluster_size(red, cluster);
   nir_def_init(&red->instr, &red->def, data->num_components,
                data->bit_size);
   nir_builder_instr_insert(b, &red->instr);
   return &red->def;
}

/* The float compare whose complement is another float compare, for folding
 * a boolean not into a ballot's fused compare condition. */
static nir_op
agx_fcmp_complement(nir_op op)
{
   switch (op) {
   case nir_op_flt:  return nir_op_fge;
   case nir_op_fge:  return nir_op_flt;
   case nir_op_feq:  return nir_op_fneu;
   case nir_op_fneu: return nir_op_feq;
   default:          return nir_num_opcodes;
   }
}

/* Shape a ballot's predicate for G17P's fused float ballot, which compares
 * a register against a minifloat immediate or a second register: fold a
 * boolean not into the compare's condition and put a constant operand
 * second.  Each rewrite saves the predicate materialization (fcmpsel, the
 * not itself, and the widen) that an integer ballot of a 16-bit boolean
 * would otherwise need.
 */
static bool
agx_g17p_normalize_ballot_src(nir_builder *b, nir_intrinsic_instr *ballot)
{
   nir_def *pred = ballot->src[0].ssa;
   nir_instr *pred_instr = nir_def_instr(pred);
   if (pred_instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(pred_instr);

   bool changed = false;
   nir_op op = alu->op;
   if (op == nir_op_inot) {
      nir_def *x = alu->src[0].src.ssa;
      nir_instr *x_instr = nir_def_instr(x);
      if (x_instr->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *cmp = nir_instr_as_alu(x_instr);
      nir_op complement = agx_fcmp_complement(cmp->op);
      if (complement == nir_num_opcodes)
         return false;
      op = complement;
      alu = cmp;
      changed = true;
   } else if (agx_fcmp_complement(op) == nir_num_opcodes) {
      return false;
   }

   if (!changed)
      return false;

   nir_def *x = alu->src[0].src.ssa;
   nir_def *y = alu->src[1].src.ssa;

   nir_src_rewrite(&ballot->src[0], nir_build_alu2(b, op, x, y));
   return true;
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_subgroup_lower_options *options = data;
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
      nir_def *result =
         agx_subgroup_read(b, intr->src[0].ssa, intr->src[1].ssa);
      nir_def_rewrite_uses(&intr->def, result);
      return true;
   }

   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan: {
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      nir_def *data = intr->src[0].ssa;

      /* G17P: scan the order-equivalent signed key with the integer
       * min/max prefix op (agx_f32_minmax_key), since float min/max has
       * no prefix encoding.  The intrinsic keeps its place and becomes
       * the exclusive integer scan on the keyed source.
       */
      if (options->g17p && intr->def.num_components == 1 &&
          intr->def.bit_size == 32 &&
          (red_op == nir_op_fmin || red_op == nir_op_fmax)) {
         const bool inclusive =
            intr->intrinsic == nir_intrinsic_inclusive_scan;
         const nir_op int_op =
            red_op == nir_op_fmin ? nir_op_imin : nir_op_imax;

         nir_def *key = agx_f32_minmax_key(b, data);
         intr->intrinsic = nir_intrinsic_exclusive_scan;
         nir_src_rewrite(&intr->src[0], key);
         nir_intrinsic_set_reduction_op(intr, int_op);

         b->cursor = nir_after_instr(&intr->instr);

         if (inclusive) {
            /* Fold the lane's own key in; the prefix reports the
             * signed-min/max identity where there is no active
             * predecessor, which folds away cleanly. */
            nir_def *res = agx_f32_minmax_key(
               b, nir_build_alu2(b, int_op, &intr->def, key));
            nir_def_rewrite_uses_after(&intr->def, res);
         } else {
            /* The exclusive prefix reports the integer identity
             * (INT_MAX / INT_MIN) for a lane with no active predecessor,
             * which decodes to a NaN.  The float identity is +-inf, so
             * select it explicitly from the ballot. */
            nir_def *m = nir_ballot(b, 1, 32, nir_imm_true(b));
            nir_def *lane = nir_load_subgroup_invocation(b);
            nir_def *below =
               nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), lane), -1);
            nir_def *has_pred = nir_ine_imm(b, nir_iand(b, m, below), 0);
            nir_def *res = nir_bcsel(
               b, has_pred, agx_f32_minmax_key(b, &intr->def),
               nir_imm_int(b, red_op == nir_op_fmin ? 0x7f800000
                                                  : 0xff800000));
            nir_def_rewrite_uses_after(&intr->def, res);
         }
         return true;
      }

      if (agx_scan_eligible(intr)) {
         nir_def *res = agx_subgroup_scan(b, red_op, data,
            intr->intrinsic == nir_intrinsic_inclusive_scan);
         nir_def_rewrite_uses(&intr->def, res);
         return true;
      }

      if (intr->intrinsic == nir_intrinsic_exclusive_scan)
         return false;

      /* If we got here, we support the corresponding exclusive scan in
       * hardware, so just handle the last element.
       */
      b->cursor = nir_after_instr(&intr->instr);
      intr->intrinsic = nir_intrinsic_exclusive_scan;
      nir_def *accum = nir_build_alu2(b, red_op, data, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, accum);
      return true;
   }

   case nir_intrinsic_reduce: {
      if (agx_minmax64_reduce_eligible(intr)) {
         nir_op red_op = nir_intrinsic_reduction_op(intr);
         unsigned cluster = nir_intrinsic_cluster_size(intr);
         nir_def *x = intr->src[0].ssa;

         nir_def *lo = nir_unpack_64_2x32_split_x(b, x);
         nir_def *hi = nir_unpack_64_2x32_split_y(b, x);

         /* The high word decides with the source signedness. */
         nir_def *w_hi = agx_subgroup_reduce(b, red_op, cluster, hi);

         /* Only lanes tied on the high word put their low word into the
          * second reduce; the rest supply the identity of the low-word
          * op, which always compares unsigned. */
         nir_op lo_op = (red_op == nir_op_umin || red_op == nir_op_imin)
                           ? nir_op_umin
                           : nir_op_umax;
         nir_const_value ident = nir_alu_binop_identity(lo_op, 32);
         nir_def *identity = nir_build_imm(b, 1, 32, &ident);
         nir_def *tied = nir_bcsel(b, nir_ieq(b, hi, w_hi), lo, identity);

         nir_def *w_lo = agx_subgroup_reduce(b, lo_op, cluster, tied);
         nir_def_rewrite_uses(&intr->def,
                              nir_pack_64_2x32_split(b, w_lo, w_hi));
         return true;
      }

      if (!agx_clustered_reduce_eligible(intr,
                                         options->g17p_clustered_float))
         return false;

      /* Reduce over the active lanes of an aligned cluster, without
       * branching on the mask.  The hardware quad reduce skips inactive
       * lanes, so it is a valid 4-lane base under any activity mask.
       * Wider steps then fold each lane's accumulated d-block total
       * together with the sibling d-block's, which every live lane of
       * that block holds.  A fixed XOR buddy can be inactive -- the hole
       * that breaks a plain butterfly -- so the read must go through a
       * live lane of the sibling block instead.
       *
       * The register-lane shuffle does not take a fully per-lane index:
       * for a read of source quad Q the hardware merges the low two
       * index bits of Q's own lanes (see agx_subgroup_read), so the
       * effective lane is (idx & ~3) + p where p is whatever low bits
       * Q's active lanes present.  Publish p = the position of the last
       * live lane of each lane's own quad; every read of a quad then
       * lands on a live lane, which by induction holds that quad's
       * accumulated block total.
       *
       * An empty sibling block leaves the accumulated value untouched --
       * no identity element is ever materialised, which also keeps
       * fadd's -0.0 sign exact.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      unsigned cluster = nir_intrinsic_cluster_size(intr);
      nir_def *data = intr->src[0].ssa;

      nir_def *m = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *lane = nir_load_subgroup_invocation(b);

      /* A 2-lane cluster is the pair (l, l^1), which never straddles a
       * quad, so the guarded d=1 combine is the whole reduction.  It is
       * the whole reduction for every op: the quad form would fold in
       * the quad's other pair too.
       */
      if (cluster == 2) {
         nir_def_rewrite_uses(
            &intr->def, agx_pair_combine(b, red_op, m, lane, data));
         return true;
      }

      nir_def *acc;

      if (agx_quad_reduce_supported(red_op, data->bit_size,
                                    options->g17p_clustered_float)) {
         acc = agx_subgroup_reduce(b, red_op, 4, data);
      } else {
         /* No quad form (imul; fmin/fmax off G17P): run the two
          * intra-quad combine steps explicitly.  d = 1 folds in the
          * buddy lane itself through a guarded butterfly; d = 2 reads
          * the sibling pair through whichever of its lanes is live --
          * lane^2 when active, else lane^3, its pair-mate.
          */
         acc = agx_pair_combine(b, red_op, m, lane, data);

         nir_def *live2 = nir_ine_imm(
            b, nir_iand_imm(b,
                            nir_ushr(b, m, nir_ixor(b, lane, nir_imm_int(b, 2))), 1),
            0);
         nir_def *b2 = nir_bcsel(b, live2,
                                 nir_shuffle_xor(b, acc, nir_imm_int(b, 2)),
                                 nir_shuffle_xor(b, acc, nir_imm_int(b, 3)));
         nir_def *pair = nir_iand_imm(b, nir_ixor(b, lane, nir_imm_int(b, 2)), ~1u);
         nir_def *ok2 = nir_ine_imm(
            b, nir_iand_imm(b, nir_ushr(b, m, pair), 3), 0);
         acc = nir_bcsel(b, ok2, nir_build_alu2(b, red_op, acc, b2), acc);
      }

      /* A 4-lane cluster is exactly the quad: for the ops that get here
       * (no quad form -- the eligible filter keeps quad-capable cluster-4
       * reduces on the hardware path) the intra-quad steps are the whole
       * reduction.
       */
      if (cluster == 4) {
         nir_def_rewrite_uses(&intr->def, acc);
         return true;
      }

      /* p: the position of the last live lane of this lane's own quad.
       * The lane itself is active, so its quad always has one. */
      nir_def *qb = nir_iand_imm(b, lane, ~3u);
      nir_def *p = nir_ufind_msb(
         b, nir_iand_imm(b, nir_ushr(b, m, qb), 0xf));

      /* d = 4: the sibling block is exactly one quad.  Reading it lands
       * on its last live lane, which holds its total. */
      nir_def *q4 =
         nir_iand_imm(b, nir_ixor(b, lane, nir_imm_int(b, 4)), ~3u);
      nir_def *bits4 = nir_iand_imm(b, nir_ushr(b, m, q4), 0xf);
      nir_def *buddy4 = nir_read_invocation(b, acc, nir_ior(b, q4, p));
      acc = nir_bcsel(b, nir_ine_imm(b, bits4, 0),
                      nir_build_alu2(b, red_op, acc, buddy4), acc);

      /* d = 8: the sibling block spans two quads.  Read the quad holding
       * the block's last live lane, itself live and holding the total. */
      for (unsigned d = 8; d < cluster; d *= 2) {
         nir_def *base =
            nir_iand_imm(b, nir_ixor(b, lane, nir_imm_int(b, d)), ~(d - 1));
         nir_def *sib = nir_iand(
            b, m, nir_ishl(b, nir_imm_int(b, BITFIELD_MASK(d)), base));
         nir_def *rep = nir_umin_imm(b, nir_ufind_msb(b, sib), 31);
         nir_def *buddy = nir_read_invocation(
            b, acc, nir_ior(b, nir_iand_imm(b, rep, ~3u), p));
         acc = nir_bcsel(b, nir_ine_imm(b, sib, 0),
                         nir_build_alu2(b, red_op, acc, buddy), acc);
      }

      nir_def_rewrite_uses(&intr->def, acc);
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
      if (agx_options->g17p)
         return agx_g17p_normalize_ballot_src(b, intr);

      if (!nir_src_is_const(intr->src[0]) ||
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

   /* Scans and clustered reductions handled by the guarded lowerings in
    * lower() stay intrinsics and reach it untouched.
    */
   if (agx_scan_eligible(intr) ||
       agx_clustered_reduce_eligible(intr,
                                     agx_options->g17p_clustered_float) ||
       agx_minmax64_reduce_eligible(intr))
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
      /* Fresh G17P witnesses pin the native qRedfmin/qRedfmax forms, and the
       * same captures show sRedfmin/sRedfmax for full-subgroup reduces. Keep
       * every other target and non-quad clustered shape on the established
       * common lowering path. Float min/max scans have no prefix encoding;
       * G17P reroutes them through the signed-integer key prefix in lower(),
       * and anything else stays on the common lowering. */
      if (agx_options->g17p_clustered_float &&
          intr->def.bit_size == 32 &&
          intr->intrinsic == nir_intrinsic_reduce &&
          (cluster == 4 || cluster == 0 || cluster >= 32))
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

   /* Then do AGX-only lowerings on top. */
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
