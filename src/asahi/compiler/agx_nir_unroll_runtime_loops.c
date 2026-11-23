/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Guarded unrolling of counted loops whose trip count is only known at run
 * time, such as a filter window clipped at the image border:
 *
 *    for (i = lo; i <= hi; i++)
 *       body(i);
 *
 * becomes
 *
 *    for (; i <= hi - (F - 1); i += F)    // "main" loop, no exits inside
 *       body(i); body(i + 1); ... body(i + F - 1);
 *    for (; i <= hi; i++)                 // the original loop
 *       body(i);
 *
 * The main loop runs an iteration only when the original loop would run the
 * next F iterations, so every instruction it executes is one the original
 * executes, in the same order. Its guard is computed so that the iteration
 * variable cannot wrap inside those F iterations; when hi is so close to the
 * bottom of the range that hi - (F - 1) would wrap, the guard is never true
 * and the original loop does all the work.
 *
 * On a GPU that issues one instruction per lane per clock, the loop control
 * (compare, mask update, branch and induction increment) is a large share of
 * a short loop body, and the body's loads only overlap within one iteration.
 * Unrolling removes F - 1 of every F loop-control sequences, lets the
 * scheduler issue F iterations' loads together, and lets the memory
 * vectorizer merge their contiguous loads.
 *
 * Only innermost loops whose single exit is a test of a basic induction
 * variable against a loop-invariant bound at the top of the body qualify.
 * Loops with subgroup, quad or barrier operations are left alone: unrolling
 * changes which iterations of different invocations execute together.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "util/set.h"

enum cont_rel {
   REL_LT,
   REL_LE,
   REL_GT,
   REL_GE,
};

struct counted_loop {
   nir_loop *loop;
   nir_if *nif;
   bool break_in_then;
   /* The exit test ends the body (a rotated loop) instead of starting it. */
   bool bottom;
   nir_phi_instr *iv;
   /* The test reads iv + offset. */
   int64_t offset;
   nir_scalar bound;
   int64_t step;
   enum cont_rel rel;
   bool is_unsigned;
};

static bool
block_is_lone_break(nir_block *block)
{
   nir_instr *last = nir_block_last_instr(block);
   return last && last == nir_block_first_instr(block) &&
          last->type == nir_instr_type_jump &&
          nir_instr_as_jump(last)->type == nir_jump_break;
}

static bool
branch_is_lone_break(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          block_is_lone_break(nir_cf_node_as_block(node));
}

static bool
branch_is_empty(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(node)->instr_list);
}

/* By the control-flow tree, not block indices: those go stale once a loop
 * of the shader has been unrolled. */
static bool
def_outside_loop(nir_def *def, nir_loop *loop)
{
   for (nir_cf_node *node = nir_def_block(def)->cf_node.parent; node;
        node = node->parent) {
      if (node == &loop->cf_node)
         return false;
   }
   return true;
}

/* A value every iteration computes alike from values defined before the
 * loop, without side effects, in the first block: such as a bound read from a
 * uniform block. The first block runs before the loop can exit, so computing
 * the value once in the preheader reads nothing the loop would not read. */
static bool
header_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != nir_loop_first_block(loop) || depth == 0)
      return false;
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!header_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!nir_intrinsic_can_reorder(intr) ||
          !nir_intrinsic_infos[intr->intrinsic].has_dest)
         return false;
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         if (!header_invariant(intr->src[i].ssa, loop, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

/* Recompute a header_invariant() value at the builder's cursor. */
static nir_def *
rematerialize(nir_builder *b, nir_def *def, nir_loop *loop,
              struct hash_table *remap)
{
   if (def_outside_loop(def, loop))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         rematerialize(b, alu->src[i].src.ssa, loop, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         rematerialize(b, intr->src[i].ssa, loop, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *result = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, result);
   return result;
}

/* Operations whose result depends on which invocations run them together:
 * subgroup and quad operations, and the AGX forms the subgroup lowering has
 * already produced before this pass (elect, the active-lane count, quad
 * ballots), plus the AGX operations that act for the whole SIMD-group or
 * workgroup (ray traversal and its shared state, mesh payload barriers,
 * sample-mask demotes). A main loop and a remainder, or two copies of an
 * outer loop, run by different invocations at once would change them. */
static bool
lane_dependent(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_infos[intr->intrinsic].flags &
       (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
      return true;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_active_subgroup_invocation_agx:
   case nir_intrinsic_load_active_subgroup_count_agx:
   case nir_intrinsic_quad_ballot_agx:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_intersect_ray_agx:
   case nir_intrinsic_continue_ray_agx:
   case nir_intrinsic_load_scs_uniform_agx:
   case nir_intrinsic_store_scs_uniform_agx:
   case nir_intrinsic_mesh_payload_entry_barrier_agx:
   case nir_intrinsic_object_payload_exit_barrier_agx:
   case nir_intrinsic_fence_mesh_uvb_agx:
      return true;
   default:
      return false;
   }
}

/* Everything in the loop that is safe to execute F times in a row per
 * main-loop iteration. Returns the number of instructions, or -1. */
static int
count_unrollable(nir_loop *loop, nir_if *terminator)
{
   int count = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            /* Only the terminator's break. */
            if (block->cf_node.parent != &terminator->cf_node)
               return -1;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return -1;
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy_coarse:
               return -1;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return -1;
            break;
         case nir_instr_type_call:
            return -1;
         default:
            break;
         }
         if (instr->type != nir_instr_type_phi)
            count++;
      }
   }
   return count;
}

struct varying_state {
   BITSET_WORD *varying;
   bool any;
};

static bool
src_varies(nir_src *src, void *data)
{
   struct varying_state *st = data;
   st->any |= BITSET_TEST(st->varying, src->ssa->index);
   return true;
}

/* 32-bit registers one iteration's memory and texture reads fill, counting
 * the reads whose address depends on a value the loop carries (reads of
 * addresses that are the same every iteration, such as descriptors and the
 * bound's parameters, are shared by the unrolled copies). */
static unsigned
loaded_dwords(nir_function_impl *impl, nir_loop *loop)
{
   BITSET_WORD *varying =
      calloc(BITSET_WORDS(impl->ssa_alloc), sizeof(BITSET_WORD));
   nir_block *header = nir_loop_first_block(loop);
   unsigned dwords = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         struct varying_state st = {varying, false};
         if (instr->type == nir_instr_type_phi && block == header)
            st.any = true;
         else
            nir_foreach_src(instr, src_varies, &st);
         if (!st.any)
            continue;
         BITSET_SET(varying, def->index);

         bool load = instr->type == nir_instr_type_tex;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
            load = info->has_dest && nir_intrinsic_has_align_mul(intr) &&
                   !strncmp(info->name, "load_", 5);
         }
         if (load)
            dwords += def->num_components * DIV_ROUND_UP(def->bit_size, 32);
      }
   }
   free(varying);
   return dwords;
}

static bool
analyze_loop_opts(nir_loop *loop, struct counted_loop *out, bool allow_nested)
{
   if (nir_loop_has_continue_construct(loop) || loop->partially_unrolled ||
       loop->control == nir_loop_control_dont_unroll)
      return false;

   /* The single exit is a lone break in one branch of an if, either right
    * after the header block or as the last node before the final block. */
   bool bottom = false;
   nir_cf_node *first = nir_cf_node_next(&nir_loop_first_block(loop)->cf_node);
   nir_cf_node *last_if = nir_cf_node_prev(&nir_loop_last_block(loop)->cf_node);
   nir_if *nif = NULL;
   bool break_in_then = false;
   for (unsigned pass = 0; pass < 2 && !nif; ++pass) {
      /* The bottom form first: an exit after the whole body needs no
       * side-effect-free header. */
      nir_cf_node *node = pass == 0 ? last_if : first;
      if (!node || node->type != nir_cf_node_if)
         continue;
      if (pass == 0 &&
          !exec_list_is_empty(&nir_loop_last_block(loop)->instr_list))
         continue;
      nir_if *candidate = nir_cf_node_as_if(node);
      if (branch_is_lone_break(&candidate->then_list) &&
          branch_is_empty(&candidate->else_list))
         break_in_then = true;
      else if (branch_is_lone_break(&candidate->else_list) &&
               branch_is_empty(&candidate->then_list))
         break_in_then = false;
      else
         continue;
      nif = candidate;
      bottom = pass == 0;
   }
   if (!nif)
      return false;

   /* Innermost: nothing but blocks and ifs after the terminator. */
   if (!allow_nested) {
      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         nir_cf_node *parent = block->cf_node.parent;
         while (parent != &loop->cf_node) {
            if (parent->type == nir_cf_node_loop)
               return false;
            parent = parent->parent;
         }
      }
   }

   /* A top-tested main loop runs its header once more before it exits into
    * the original loop, which runs it again for the same iteration: the
    * header must have no side effects. A rotated loop's main loop exits
    * before any of its body. */
   nir_block *header = nir_loop_first_block(loop);
   nir_foreach_instr(instr, header) {
      if (bottom)
         break;
      if (instr->type == nir_instr_type_intrinsic &&
          (!(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].flags &
             NIR_INTRINSIC_CAN_ELIMINATE) ||
           (nir_intrinsic_has_access(nir_instr_as_intrinsic(instr)) &&
            (nir_intrinsic_access(nir_instr_as_intrinsic(instr)) & ACCESS_VOLATILE))))
         return false;
      if (instr->type == nir_instr_type_call)
         return false;
   }

   nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
   bool negate = false;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      negate = !negate;
      cond = nir_scalar_chase_alu_src(cond, 0);
   }
   if (!nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);
   if (op != nir_op_ilt && op != nir_op_ige && op != nir_op_ult &&
       op != nir_op_uge)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   if (a.def->bit_size != 32 || a.def->num_components != 1 ||
       b.def->num_components != 1)
      return false;

   /* One side is a phi of the header, or that phi plus a constant, the
    * other is invariant. */
   bool iv_left = false;
   nir_scalar ivs = {0}, bound = {0};
   int64_t offset = 0;
   for (unsigned side = 0; side < 2 && !ivs.def; ++side) {
      nir_scalar x = side == 0 ? a : b, y = side == 0 ? b : a;
      if (!nir_scalar_is_const(y) && !header_invariant(y.def, loop, 8))
         continue;
      int64_t c = 0;
      if (nir_scalar_is_alu(x) && nir_scalar_alu_op(x) == nir_op_iadd) {
         for (unsigned k = 0; k < 2; ++k) {
            nir_scalar kk = nir_scalar_chase_alu_src(x, k);
            nir_scalar other = nir_scalar_chase_alu_src(x, 1 - k);
            if (nir_scalar_is_const(kk) &&
                nir_def_instr(other.def)->type == nir_instr_type_phi) {
               c = nir_scalar_as_int(kk);
               x = other;
               break;
            }
         }
      }
      if (nir_def_instr(x.def)->type == nir_instr_type_phi &&
          nir_def_instr(x.def)->block == header && x.def->bit_size == 32 &&
          c >= -0x10000 && c <= 0x10000) {
         ivs = x;
         bound = y;
         offset = c;
         iv_left = side == 0;
      }
   }
   if (!ivs.def)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(ivs.def));
   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
   if (!back)
      return false;

   nir_scalar update = nir_scalar_resolved(back->src.ssa, 0);
   if (!nir_scalar_is_alu(update) || nir_scalar_alu_op(update) != nir_op_iadd)
      return false;

   int64_t step = 0;
   bool found = false;
   for (unsigned s = 0; s < 2; ++s) {
      nir_scalar src = nir_scalar_chase_alu_src(update, s);
      nir_scalar other = nir_scalar_chase_alu_src(update, 1 - s);
      if (other.def == &phi->def && nir_scalar_is_const(src)) {
         step = nir_scalar_as_int(src);
         found = true;
      }
   }
   if (!found || step == 0 || step > 0x10000 || step < -0x10000)
      return false;

   /* Normalize to the condition that keeps the loop running, written as
    * "iv REL bound". The break condition is cond (or !cond when negated)
    * when the break is in the then-branch, the opposite otherwise. */
   bool break_on_true = break_in_then != negate;
   /* cond is (a < b) for ilt/ult and (a >= b) for ige/uge. */
   bool less = op == nir_op_ilt || op == nir_op_ult;
   /* Continue condition on (a, b): */
   bool cont_less = break_on_true ? !less : less;
   /* cont_less: a < b; otherwise a >= b. */
   enum cont_rel rel;
   if (iv_left)
      rel = cont_less ? REL_LT : REL_GE;
   else
      rel = cont_less ? REL_GT : REL_LE; /* bound < iv / bound >= iv */

   if ((rel == REL_LT || rel == REL_LE) != (step > 0))
      return false;

   *out = (struct counted_loop){
      .loop = loop,
      .nif = nif,
      .break_in_then = break_in_then,
      .bottom = bottom,
      .iv = phi,
      .offset = offset,
      .bound = bound,
      .step = step,
      .rel = rel,
      .is_unsigned = op == nir_op_ult || op == nir_op_uge,
   };
   return true;
}

static bool
analyze_loop(nir_loop *loop, struct counted_loop *out)
{
   return analyze_loop_opts(loop, out, false);
}

/* The loop-invariant value G such that "iv < G" (REL_LT, REL_LE) or
 * "iv > G" (REL_GT, REL_GE) proves the next `factor` iterations run without
 * the induction variable wrapping. */
/* Check whether a dominating `if` proves a lower bound (for step > 0) on
 * `bound` sufficient to prevent `bound - span` underflow in guard_bound(),
 * and optionally whether it also proves `loop_continue_cond(iv0, bound)`. */
static void
check_dominating_bound_guard(nir_loop *loop, const struct counted_loop *cl,
                             nir_def *bound, nir_def *iv0, unsigned max_span,
                             bool *no_underflow, bool *iv0_enters)
{
   *no_underflow = false;
   *iv0_enters = false;

   if (cl->step <= 0 || (cl->rel != REL_LT && cl->rel != REL_LE) ||
       bound->bit_size != 32)
      return;

   const bool u = cl->is_unsigned;
   const int64_t min_val = u ? 0 : INT32_MIN;
   const int64_t need_min = min_val + (int64_t)max_span;
   nir_scalar b_s = nir_scalar_resolved(bound, 0);
   nir_scalar iv0_s = iv0 ? nir_scalar_resolved(iv0, 0) : (nir_scalar){0};

   for (nir_cf_node *node = &loop->cf_node; node->parent; node = node->parent) {
      if (node->parent->type != nir_cf_node_if)
         continue;
      nir_if *pif = nir_cf_node_as_if(node->parent);
      bool in_then = false;
      foreach_list_typed(nir_cf_node, c, node, &pif->then_list) {
         if (c == node) {
            in_then = true;
            break;
         }
      }

      nir_scalar cond = nir_scalar_resolved(pif->condition.ssa, 0);
      bool truth = in_then;
      while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
         cond = nir_scalar_chase_alu_src(cond, 0);
         truth = !truth;
      }
      if (!nir_scalar_is_alu(cond))
         continue;

      nir_op op = nir_scalar_alu_op(cond);
      nir_scalar s0 = nir_scalar_chase_alu_src(cond, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(cond, 1);

      /* Normalize to: lo < bound (strict=true) or lo <= bound (strict=false) */
      bool strict = false;
      nir_scalar lo = {0};
      bool matched = false;

      const nir_op lt_op = u ? nir_op_ult : nir_op_ilt;
      const nir_op ge_op = u ? nir_op_uge : nir_op_ige;

      if (truth && op == lt_op && nir_scalar_equal(s1, b_s)) {
         lo = s0;
         strict = true;
         matched = true;
      } else if (!truth && op == ge_op && nir_scalar_equal(s1, b_s)) {
         lo = s0;
         strict = true;
         matched = true;
      } else if (truth && op == ge_op && nir_scalar_equal(s0, b_s)) {
         lo = s1;
         strict = false;
         matched = true;
      } else if (!truth && op == lt_op && nir_scalar_equal(s0, b_s)) {
         lo = s1;
         strict = false;
         matched = true;
      }

      if (!matched)
         continue;

      if (nir_scalar_is_const(lo)) {
         int64_t c = u ? (int64_t)nir_scalar_as_uint(lo)
                       : (int64_t)nir_scalar_as_int(lo);
         int64_t lb = strict ? (c + 1) : c;
         if (lb >= need_min)
            *no_underflow = true;
      }

      if (iv0 && nir_scalar_equal(lo, iv0_s)) {
         if (strict || cl->rel == REL_LE)
            *iv0_enters = true;
      }
   }
}

static nir_def *
guard_bound(nir_builder *b, const struct counted_loop *cl, nir_def *bound,
            unsigned factor, bool no_underflow)
{
   const int64_t span = (int64_t)(factor - 1) * (cl->step > 0 ? cl->step : -cl->step);
   const bool u = cl->is_unsigned;
   const int64_t min = u ? 0 : INT32_MIN;
   const int64_t max = u ? UINT32_MAX : INT32_MAX;

   nir_def *valid, *adj, *never;
   switch (cl->rel) {
   case REL_LT:
   case REL_LE: {
      int64_t sub = cl->rel == REL_LT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, -sub);
      if (no_underflow)
         return adj;
      /* bound - span must not wrap below the minimum. */
      nir_def *lowest = nir_imm_int(b, (uint32_t)(min + span));
      valid = u ? nir_uge(b, bound, lowest) : nir_ige(b, bound, lowest);
      never = nir_imm_int(b, (uint32_t)min);
      break;
   }
   default: {
      nir_def *highest = nir_imm_int(b, (uint32_t)(max - span));
      valid = u ? nir_uge(b, highest, bound) : nir_ige(b, highest, bound);
      int64_t add = cl->rel == REL_GT ? span : span - 1;
      adj = nir_iadd_imm(b, bound, add);
      never = nir_imm_int(b, (uint32_t)max);
      break;
   }
   }
   return nir_bcsel(b, valid, adj, never);
}

static bool
block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node == &loop->cf_node)
         return true;
   }
   return false;
}

static nir_block *
use_block(nir_src *src)
{
   if (nir_src_is_if(src))
      return nir_cf_node_as_block(nir_cf_node_prev(&nir_src_use_if(src)->cf_node));
   return nir_src_use_instr(src)->block;
}

struct live_out {
   nir_def *def;
   /* The register of the header phi it feeds, or NULL for a loop-invariant
    * value. */
   nir_def *reg;
   bool invariant;
};

/* Computed from values defined before the loop only, by instructions that
 * may be repeated (as header_invariant(), in any block of the loop). */
static bool
loop_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!loop_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!nir_intrinsic_can_reorder(intr) ||
          !nir_intrinsic_infos[intr->intrinsic].has_dest)
         return false;
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         if (!loop_invariant(intr->src[i].ssa, loop, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

/* The loop's values used after it, when each is the back-edge value of a
 * header phi (a rotated loop's accumulators and induction variables) or
 * loop-invariant (recomputed after the loop). */
static bool
back_edge_live_outs(nir_loop *loop, struct live_out *outs, unsigned *count)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   /* Phis after the loop would need sources from the new if: keep the
    * plain form. */
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (nir_block_first_instr(after) &&
       nir_block_first_instr(after)->type == nir_instr_type_phi)
      return false;
   *count = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         bool outside = false;
         nir_foreach_use_including_if(src, def)
            outside |= !block_in_loop(use_block(src), loop);
         if (!outside)
            continue;
         /* A header phi used after the loop (its value at the start of the
          * last iteration) has no register holding it after the main loop;
          * nor does a header phi that is another phi's back-edge value. */
         if (instr->type == nir_instr_type_phi && block == header)
            return false;
         bool carried = false;
         nir_foreach_phi(phi, header) {
            nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
            carried |= back && back->src.ssa == def;
         }
         const bool invariant = !carried && loop_invariant(def, loop, 8);
         if ((!carried && !invariant) || *count == 16)
            return false;
         outs[(*count)++] = (struct live_out){def, NULL, invariant};
      }
   }
   return true;
}

/* The register a header phi became, from its write at the end of the
 * latch. */
static nir_def *
latch_register(nir_loop *loop, nir_def *value)
{
   nir_foreach_instr_reverse(instr, nir_loop_last_block(loop)) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
      if (store->intrinsic == nir_intrinsic_store_reg && store->src[0].ssa == value)
         return store->src[1].ssa;
   }
   return NULL;
}

static bool
cf_list_is_pure_alu_and_reg_stores(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type != nir_cf_node_block)
         return false;
      nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
         if (instr->type == nir_instr_type_alu ||
             instr->type == nir_instr_type_load_const ||
             instr->type == nir_instr_type_undef)
            continue;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_store_reg || op == nir_intrinsic_load_reg)
               continue;
         }
         return false;
      }
   }
   return true;
}

static bool
top_tested_only_phi_live_outs(nir_loop *loop, nir_def **later_bounds,
                              unsigned nlater)
{
   nir_block *header = nir_loop_first_block(loop);
   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (nir_block_first_instr(after) &&
       nir_block_first_instr(after)->type == nir_instr_type_phi)
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         for (unsigned j = 0; j < nlater; ++j) {
            if (later_bounds[j] == def)
               return false;
         }
         bool outside = false;
         nir_foreach_use_including_if(src, def)
            outside |= !block_in_loop(use_block(src), loop);
         if (outside && (instr->type != nir_instr_type_phi || block != header))
            return false;
      }
   }
   return true;
}

static void
emit_binary_tail_chunk(nir_builder *b, nir_def *cond, nir_cf_list *head,
                       nir_cf_list *rest, unsigned count)
{
   nir_if *chunk_if = nir_push_if(b, cond);
   nir_pop_if(b, chunk_if);
   for (unsigned i = 0; i < count; ++i) {
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list ch, cr;
      nir_cf_list_clone(&ch, head, &chunk_if->cf_node, remap);
      nir_cf_list_clone(&cr, rest, &chunk_if->cf_node, remap);
      _mesa_hash_table_destroy(remap, NULL);
      nir_cf_reinsert(&ch, nir_after_cf_list(&chunk_if->then_list));
      nir_cf_reinsert(&cr, nir_after_cf_list(&chunk_if->then_list));
   }
}

static nir_def *
loop_continue_cond(nir_builder *b, const struct counted_loop *cl, nir_def *v,
                   nir_def *bound)
{
   const bool u = cl->is_unsigned;
   switch (cl->rel) {
   case REL_LT: return u ? nir_ult(b, v, bound) : nir_ilt(b, v, bound);
   case REL_LE: return u ? nir_uge(b, bound, v) : nir_ige(b, bound, v);
   case REL_GT: return u ? nir_ult(b, bound, v) : nir_ilt(b, bound, v);
   default:     return u ? nir_uge(b, v, bound) : nir_ige(b, v, bound);
   }
}

static bool outer_invariant(nir_def *def, nir_loop *outer, unsigned depth);
static nir_def *rematerialize(nir_builder *b, nir_def *def, nir_loop *loop,
                              struct hash_table *remap);

/* Compute guard_bound(bound, k) at the outermost enclosing loop preheader
 * where `bound` is loop-invariant (or rematerializable from values defined
 * before that loop), so pure ALU guard computations are not repeated inside
 * outer pixel loops and can move to the uniform preamble. */
static nir_def *
guard_bound_hoisted(nir_function_impl *impl, nir_loop *loop, nir_loop *anchor,
                    const struct counted_loop *cl, nir_def *bound, unsigned k,
                    bool no_underflow)
{
   nir_loop *outermost = anchor;
   for (nir_cf_node *p = loop->cf_node.parent; p; p = p->parent) {
      if (p->type == nir_cf_node_loop) {
         nir_loop *pl = nir_cf_node_as_loop(p);
         if (outer_invariant(bound, pl, 8))
            outermost = pl;
         else
            break;
      }
   }
   nir_builder gb = nir_builder_at(nir_before_cf_node(&outermost->cf_node));
   nir_def *bdef = bound;
   if (outermost != anchor && !def_outside_loop(bound, outermost)) {
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      bdef = rematerialize(&gb, bound, outermost, remap);
      _mesa_hash_table_destroy(remap, NULL);
   }
   return guard_bound(&gb, cl, bdef, k, no_underflow);
}

/* later_bounds: the bounds computed for the loops unrolled after this one,
 * which may name a value this loop leaves (plain pointers, not uses). */
static void
unroll_loop(nir_function_impl *impl, struct counted_loop *cl, nir_def *bound,
            unsigned factor, unsigned max_instrs, nir_def **later_bounds,
            unsigned nlater, bool allow_binary_tail)
{
   nir_loop *loop = cl->loop;
   nir_builder b = nir_builder_create(impl);

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));

   /* A rotated loop runs its body before its test, so the original loop,
    * reached after the main loop, runs at least one more iteration. When
    * every value the loop leaves is a header phi's back-edge value and the
    * test reads the stepped induction variable, skip the original loop
    * instead once the test fails: the main loop then needs only `factor`
    * iterations left, not factor + 1 (a trip count of exactly `factor`,
    * such as 8 descriptor words, runs entirely in the main loop). */
   struct live_out outs[16];
   unsigned nouts = 0;
   nir_phi_src *init = nir_phi_get_src_from_block(cl->iv, pre);
   nir_phi_src *iv_back = nir_phi_get_src_from_block(cl->iv, nir_loop_last_block(loop));
   const bool exact_tail = cl->bottom && cl->offset == cl->step && init &&
                           iv_back && back_edge_live_outs(loop, outs, &nouts);
   nir_def *iv0 = exact_tail ? init->src.ssa : NULL;
   nir_def *iv_next = exact_tail ? iv_back->src.ssa : NULL;

   const unsigned max_span =
      (factor - 1) * (unsigned)(cl->step > 0 ? cl->step : -cl->step);
   bool no_underflow = false, iv0_enters = false;
   check_dominating_bound_guard(loop, cl, bound, iv0, max_span,
                                &no_underflow, &iv0_enters);

   /* The guard bound, hoisted to the outermost enclosing loop preheader where
    * `bound` is invariant. */
   b.cursor = nir_after_block_before_jump(pre);
   nir_def *g =
      guard_bound_hoisted(impl, loop, loop, cl, bound, factor, no_underflow);

   /* Compact step-1/-1 counted loops unrolled by 8 leave 0..7 remainder
    * iterations (unless the induction variable wraps on REL_LE/REL_GE at an
    * extreme bound or on an exact_tail first iteration at max/min, handled
    * below). Binary-peeling the tail (4/2/1, factor - 1 extra copies) removes
    * the scalar remainder backedge on the non-wrapping path and exposes 4x/2x
    * load vectorization and ILP while staying within 2 * max_instrs. */
   const int body_ops = count_unrollable(loop, cl->nif);
   bool binary_tail =
      allow_binary_tail && factor == 8 &&
      (cl->step == 1 || cl->step == -1) &&
      body_ops > 0 &&
      (unsigned)body_ops * (2 * factor - 1) <= 2 * max_instrs &&
      (exact_tail ||
       (!cl->bottom && cl->offset == 0 &&
        top_tested_only_phi_live_outs(loop, later_bounds, nlater)));

   /* The guard test, in the header after its phis. A top-tested loop runs
    * iteration k when its test passes on v + k * step, k = 0 .. factor - 1,
    * with v the tested value at the top. A rotated loop runs the iteration
    * after k when its test passes on v + k * step, k = 0 .. factor - 1: the
    * main loop's factor iterations, plus the first one of the original loop,
    * which runs its body before its test. The same guard covers both. */
   b.cursor = nir_after_phis(nir_loop_first_block(loop));
   nir_def *iv = &cl->iv->def;
   if (cl->offset && !exact_tail)
      iv = nir_iadd_imm(&b, iv, cl->offset);
   nir_def *cont;
   if (cl->rel == REL_LT || cl->rel == REL_LE)
      cont = cl->is_unsigned ? nir_ult(&b, iv, g) : nir_ilt(&b, iv, g);
   else
      cont = cl->is_unsigned ? nir_ult(&b, g, iv) : nir_ilt(&b, g, iv);
   nir_def *guard = cl->break_in_then ? nir_inot(&b, cont) : cont;
   /* Which source of the compare reads the induction variable: after the
    * header phis become registers, it is the load of the iv's register. */
   nir_alu_instr *cont_alu = nir_def_as_alu(cont);
   const unsigned iv_src = (cl->rel == REL_LT || cl->rel == REL_LE) ? 0 : 1;

   /* Leave SSA for the header's phis: both loops then read and write the
    * same registers and need no phis of their own. Write the registers at
    * the end of the latch, which every copy of the body includes: placed
    * higher, a value defined in the header would be written in the exit
    * test's continue branch, which the extra copies do not repeat. */
   nir_lower_phis_to_regs_block(nir_loop_first_block(loop), true);

   /* Build the main loop in front of the original. */
   nir_loop *main_loop = nir_loop_create(impl);
   nir_cf_node_insert(nir_before_cf_node(&loop->cf_node), &main_loop->cf_node);
   main_loop->partially_unrolled = true;
   loop->partially_unrolled = true;

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_cf_list(&loop->body),
                  nir_after_cf_list(&loop->body));
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_cf_list_clone_and_reinsert(&body, &main_loop->cf_node,
                                  nir_after_cf_list(&main_loop->body), remap);
   nir_cf_reinsert(&body, nir_after_cf_list(&loop->body));

   struct hash_entry *he = _mesa_hash_table_search(remap, guard);
   assert(he);
   nir_def *main_guard = he->data;
   _mesa_hash_table_destroy(remap, NULL);

   nir_if *main_if;
   if (!cl->bottom) {
      main_if = nir_cf_node_as_if(
         nir_cf_node_next(&nir_loop_first_block(main_loop)->cf_node));
      nir_src_rewrite(&main_if->condition, main_guard);
   } else {
      /* Test at the top instead, and drop the bottom test. Its continue
       * branch holds the register writes that replaced the header phis'
       * back-edge sources: keep them, after the dropped test. */
      nir_if *bottom_if = nir_cf_node_as_if(
         nir_cf_node_prev(&nir_loop_last_block(main_loop)->cf_node));
      struct exec_list *cont_list = cl->break_in_then ? &bottom_if->else_list
                                                      : &bottom_if->then_list;
      if (binary_tail && !cf_list_is_pure_alu_and_reg_stores(cont_list))
         binary_tail = false;
      nir_cf_list cont;
      nir_cf_extract(&cont, nir_before_cf_list(cont_list),
                     nir_after_cf_list(cont_list));
      nir_cf_reinsert(&cont, nir_after_cf_node(&bottom_if->cf_node));
      nir_cf_node_remove(&bottom_if->cf_node);
      b.cursor = nir_after_instr(nir_def_instr(main_guard));
      main_if = nir_push_if(&b, main_guard);
      if (cl->break_in_then)
         nir_jump(&b, nir_jump_break);
      nir_push_else(&b, main_if);
      if (!cl->break_in_then)
         nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, main_if);
   }
   /* Repeat the iteration factor - 1 more times inside the main loop. */
   nir_cf_list head, rest;
   nir_cf_extract(&head, nir_before_cf_list(&main_loop->body),
                  nir_before_cf_node(&main_if->cf_node));
   nir_cf_extract(&rest, nir_after_cf_node(&main_if->cf_node),
                  nir_after_cf_list(&main_loop->body));

   nir_cf_list copies[2 * 16];
   assert(factor <= 16);
   for (unsigned i = 1; i < factor; ++i) {
      remap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copies[2 * i], &head, &main_loop->cf_node, remap);
      nir_cf_list_clone(&copies[2 * i + 1], &rest, &main_loop->cf_node, remap);
      _mesa_hash_table_destroy(remap, NULL);
   }

   if (binary_tail) {
      nir_intrinsic_instr *iv_load =
         nir_def_as_intrinsic(cont_alu->src[iv_src].src.ssa);
      assert(iv_load->intrinsic == nir_intrinsic_load_reg);
      nir_def *iv_reg = iv_load->src[0].ssa;

      if (exact_tail) {
         for (unsigned i = 0; i < nouts; ++i) {
            if (outs[i].invariant)
               continue;
            outs[i].reg = latch_register(loop, outs[i].def);
            assert(outs[i].reg);
         }
      }

      const bool u = cl->is_unsigned;
      b.cursor = nir_before_cf_node(&loop->cf_node);
      for (unsigned sub = factor / 2; sub >= 2; sub /= 2) {
         nir_def *v = nir_load_reg(&b, iv_reg);
         nir_def *gs =
            guard_bound_hoisted(impl, loop, main_loop, cl, bound, sub,
                                no_underflow);
         nir_def *cond;
         if (cl->rel == REL_LT || cl->rel == REL_LE)
            cond = u ? nir_ult(&b, v, gs) : nir_ilt(&b, v, gs);
         else
            cond = u ? nir_ult(&b, gs, v) : nir_ilt(&b, gs, v);
         emit_binary_tail_chunk(&b, cond, &head, &rest, sub);
      }

      nir_def *v = nir_load_reg(&b, iv_reg);
      nir_def *more = loop_continue_cond(&b, cl, v, bound);
      nir_def *run = (exact_tail && !iv0_enters)
                        ? nir_ior(&b, nir_ieq(&b, v, iv0), more)
                        : more;
      emit_binary_tail_chunk(&b, run, &head, &rest, 1);

      /* Rotate main_loop to a guarded bottom-tested loop: when trip count <
       * factor we skip entering the loop CF nesting completely, and when
       * iterating the back-edge test is bottom-tested (enabling AGX3 early
       * loop compares without a top break_if + jmp_exec_none_after pair). */
      nir_cf_node_remove(&main_if->cf_node);
      nir_cf_reinsert(&head, nir_after_cf_list(&main_loop->body));
      nir_cf_reinsert(&rest, nir_after_cf_list(&main_loop->body));
      for (unsigned i = 1; i < factor; ++i) {
         nir_cf_reinsert(&copies[2 * i], nir_after_cf_list(&main_loop->body));
         nir_cf_reinsert(&copies[2 * i + 1], nir_after_cf_list(&main_loop->body));
      }

      b.cursor = nir_after_cf_list(&main_loop->body);
      nir_def *v_end = nir_load_reg(&b, iv_reg);
      if (cl->offset && !exact_tail)
         v_end = nir_iadd_imm(&b, v_end, cl->offset);
      nir_def *cont_end =
         (cl->rel == REL_LT || cl->rel == REL_LE)
            ? (u ? nir_ult(&b, v_end, g) : nir_ilt(&b, v_end, g))
            : (u ? nir_ult(&b, g, v_end) : nir_ilt(&b, g, v_end));
      nir_if *bot_if = nir_push_if(&b, nir_inot(&b, cont_end));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, bot_if);

      b.cursor = nir_before_cf_node(&main_loop->cf_node);
      nir_def *v_init = nir_load_reg(&b, iv_reg);
      if (cl->offset && !exact_tail)
         v_init = nir_iadd_imm(&b, v_init, cl->offset);
      nir_def *cont_init =
         (cl->rel == REL_LT || cl->rel == REL_LE)
            ? (u ? nir_ult(&b, v_init, g) : nir_ilt(&b, v_init, g))
            : (u ? nir_ult(&b, g, v_init) : nir_ilt(&b, g, v_init));
      nir_if *enter_if = nir_push_if(&b, cont_init);
      nir_pop_if(&b, enter_if);
      nir_cf_list main_list;
      nir_cf_extract(&main_list, nir_before_cf_node(&main_loop->cf_node),
                     nir_after_cf_node(&main_loop->cf_node));
      nir_cf_reinsert(&main_list, nir_before_cf_list(&enter_if->then_list));

      /* Top-tested REL_LT/REL_GT cannot wrap (every continuing iteration has
       * iv < bound <= max or iv > bound >= min), nor can top-tested
       * REL_LE/REL_GE with a constant bound away from max/min. Otherwise, an
       * extreme bound (REL_LE at max / REL_GE at min) or an exact_tail first
       * iteration at max/min can wrap after the 1-chunk; keep the original
       * scalar loop guarded by the loop condition so wrap semantics match. */
      bool needs_wrap_fallback = false;
      if (exact_tail) {
         /* Iteration 0 runs before testing; wrap on iteration 0 is only
          * possible if iv0 can equal the extreme in the step direction. */
         const int64_t step_extreme =
            cl->step > 0 ? (u ? (int64_t)UINT32_MAX : (int64_t)INT32_MAX)
                         : (u ? (int64_t)0 : (int64_t)INT32_MIN);
         nir_scalar iv0_s = nir_scalar_resolved(iv0, 0);
         needs_wrap_fallback =
            !nir_scalar_is_const(iv0_s) ||
            (u ? (int64_t)nir_scalar_as_uint(iv0_s) == step_extreme
               : nir_scalar_as_int(iv0_s) == step_extreme);
      }
      if (!needs_wrap_fallback && (cl->rel == REL_LE || cl->rel == REL_GE)) {
         const int64_t extreme =
            cl->rel == REL_LE ? (u ? (int64_t)UINT32_MAX : (int64_t)INT32_MAX)
                              : (u ? (int64_t)0 : (int64_t)INT32_MIN);
         needs_wrap_fallback =
            !nir_scalar_is_const(cl->bound) ||
            (u ? (int64_t)nir_scalar_as_uint(cl->bound) == extreme
               : nir_scalar_as_int(cl->bound) == extreme);
      }

      if (needs_wrap_fallback) {
         if (!exact_tail) {
            /* A top-tested loop already checks the exit condition before its
             * body, so leaving it bare avoids wrapping a loop in an if while
             * breaking immediately on the normal non-wrapping path. */
            return;
         }
         b.cursor = nir_before_cf_node(&loop->cf_node);
         nir_def *v_after = nir_load_reg(&b, iv_reg);
         nir_def *still_more = loop_continue_cond(&b, cl, v_after, bound);
         nir_if *wrap_if = nir_push_if(&b, still_more);
         nir_pop_if(&b, wrap_if);
         nir_cf_list moved;
         nir_cf_extract(&moved, nir_before_cf_node(&loop->cf_node),
                        nir_after_cf_node(&loop->cf_node));
         nir_cf_reinsert(&moved, nir_before_cf_list(&wrap_if->then_list));

         b.cursor = nir_after_cf_node(&wrap_if->cf_node);
         struct hash_table *again = _mesa_pointer_hash_table_create(NULL);
         for (unsigned i = 0; i < nouts; ++i) {
            nir_def *value = outs[i].invariant
                                ? rematerialize(&b, outs[i].def, loop, again)
                                : nir_load_reg(&b, outs[i].reg);
            nir_foreach_use_including_if_safe(src, outs[i].def) {
               if (!block_in_loop(use_block(src), loop))
                  nir_src_rewrite(src, value);
            }
            for (unsigned j = 0; j < nlater; ++j) {
               if (later_bounds[j] == outs[i].def)
                  later_bounds[j] = value;
            }
         }
         _mesa_hash_table_destroy(again, NULL);
         b.cursor = nir_after_cf_node(&loop->cf_node);
         for (unsigned i = 0; i < nouts; ++i) {
            if (!outs[i].invariant)
               nir_store_reg(&b, outs[i].def, outs[i].reg);
         }
         return;
      }

      b.cursor = nir_before_cf_node(&loop->cf_node);
      if (exact_tail) {
         struct hash_table *again = _mesa_pointer_hash_table_create(NULL);
         for (unsigned i = 0; i < nouts; ++i) {
            nir_def *value = outs[i].invariant
                                ? rematerialize(&b, outs[i].def, loop, again)
                                : nir_load_reg(&b, outs[i].reg);
            nir_foreach_use_including_if_safe(src, outs[i].def) {
               if (!block_in_loop(use_block(src), loop))
                  nir_src_rewrite(src, value);
            }
            for (unsigned j = 0; j < nlater; ++j) {
               if (later_bounds[j] == outs[i].def)
                  later_bounds[j] = value;
            }
         }
         _mesa_hash_table_destroy(again, NULL);
      }
      nir_foreach_instr(instr, nir_loop_first_block(loop)) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
         if (load->intrinsic != nir_intrinsic_load_reg)
            continue;
         bool outside = false;
         nir_foreach_use_including_if(src, &load->def)
            outside |= !block_in_loop(use_block(src), loop);
         if (!outside)
            continue;
         nir_def *value = nir_load_reg(&b, load->src[0].ssa);
         nir_foreach_use_including_if_safe(src, &load->def) {
            if (!block_in_loop(use_block(src), loop))
               nir_src_rewrite(src, value);
         }
      }
      nir_cf_node_remove(&loop->cf_node);
      return;
   }

   nir_cf_reinsert(&head, nir_before_cf_node(&main_if->cf_node));
   nir_cf_reinsert(&rest, nir_after_cf_list(&main_loop->body));
   for (unsigned i = 1; i < factor; ++i) {
      nir_cf_reinsert(&copies[2 * i], nir_after_cf_list(&main_loop->body));
      nir_cf_reinsert(&copies[2 * i + 1], nir_after_cf_list(&main_loop->body));
   }

   if (!exact_tail)
      return;

   /* Run the original loop only for its first iteration (no main
    * iteration ran: the induction variable is still its initial value) or
    * when the test the dropped iteration would have made passes. */
   nir_intrinsic_instr *iv_load =
      nir_def_as_intrinsic(cont_alu->src[iv_src].src.ssa);
   assert(iv_load->intrinsic == nir_intrinsic_load_reg);
   nir_def *iv_reg = iv_load->src[0].ssa;
   (void)iv_next;
   for (unsigned i = 0; i < nouts; ++i) {
      if (outs[i].invariant)
         continue;
      outs[i].reg = latch_register(loop, outs[i].def);
      assert(outs[i].reg);
   }
   b.cursor = nir_before_cf_node(&loop->cf_node);
   nir_def *v = nir_load_reg(&b, iv_reg);
   const bool u = cl->is_unsigned;
   nir_def *more;
   switch (cl->rel) {
   case REL_LT: more = u ? nir_ult(&b, v, bound) : nir_ilt(&b, v, bound); break;
   case REL_LE: more = u ? nir_uge(&b, bound, v) : nir_ige(&b, bound, v); break;
   case REL_GT: more = u ? nir_ult(&b, bound, v) : nir_ilt(&b, bound, v); break;
   default:     more = u ? nir_uge(&b, v, bound) : nir_ige(&b, v, bound); break;
   }
   nir_def *run = nir_ior(&b, nir_ieq(&b, v, iv0), more);
   nir_if *tail = nir_push_if(&b, run);
   nir_pop_if(&b, tail);
   nir_cf_list moved;
   nir_cf_extract(&moved, nir_before_cf_node(&loop->cf_node),
                  nir_after_cf_node(&loop->cf_node));
   nir_cf_reinsert(&moved, nir_before_cf_list(&tail->then_list));

   /* The values the loop leaves: after the if, from their registers. The
    * original loop exits before its latch writes them, so write them once
    * more after it. */
   b.cursor = nir_after_cf_node(&tail->cf_node);
   struct hash_table *again = _mesa_pointer_hash_table_create(NULL);
   for (unsigned i = 0; i < nouts; ++i) {
      nir_def *value = outs[i].invariant
                          ? rematerialize(&b, outs[i].def, loop, again)
                          : nir_load_reg(&b, outs[i].reg);
      nir_foreach_use_including_if_safe(src, outs[i].def) {
         if (!block_in_loop(use_block(src), loop))
            nir_src_rewrite(src, value);
      }
      for (unsigned j = 0; j < nlater; ++j) {
         if (later_bounds[j] == outs[i].def)
            later_bounds[j] = value;
      }
   }
   _mesa_hash_table_destroy(again, NULL);
   b.cursor = nir_after_cf_node(&loop->cf_node);
   for (unsigned i = 0; i < nouts; ++i) {
      if (!outs[i].invariant)
         nir_store_reg(&b, outs[i].def, outs[i].reg);
   }
}

static void collect_loops(struct exec_list *list, struct util_dynarray *loops);

/* The iterations per test for a counted loop, or 0 when it is not unrolled. */
static unsigned
choose_factor(nir_function_impl *impl, nir_loop *loop, struct counted_loop *cl,
              unsigned factor, unsigned max_instrs)
{
   int size = count_unrollable(loop, cl->nif);
   if (size < 0)
      return 0;
   /* A body too large for the full factor still gains from two
    * iterations per test when it fits the budget at half the factor. */
   unsigned f = factor;
   while (f > 2 && (unsigned)size * f > max_instrs)
      f /= 2;
   /* Beyond four iterations, keep the reads the unrolled body has in
    * flight within 16 registers: more only raises register pressure
    * (the scheduler issues them together) and can cost occupancy. */
   if (f > 4) {
      const unsigned dwords = loaded_dwords(impl, loop);
      while (f > 4 && dwords * f > 16)
         f /= 2;
   }
   return (unsigned)size * f > max_instrs ? 0 : f;
}

static nir_loop *
enclosing_loop(nir_cf_node *node)
{
   for (nir_cf_node *p = node->parent; p; p = p->parent) {
      if (p->type == nir_cf_node_loop)
         return nir_cf_node_as_loop(p);
   }
   return NULL;
}

/* Computable before `outer` from values defined before it, without memory
 * reads that could fault there (push constants, preamble values, system
 * values, and reads that may run speculatively from memory the shader does
 * not write, such as uniform buffers): a value every iteration of `outer`
 * sees alike. */
static bool
outer_invariant(nir_def *def, nir_loop *outer, unsigned depth)
{
   if (def_outside_loop(def, outer))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!outer_invariant(alu->src[i].src.ssa, outer, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      const bool speculatable_read =
         nir_intrinsic_can_reorder(intr) && nir_intrinsic_has_access(intr) &&
         (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE);
      const bool safe = intr->intrinsic == nir_intrinsic_load_push_constant ||
                        intr->intrinsic == nir_intrinsic_load_preamble ||
                        (info->num_srcs == 0 && nir_intrinsic_can_reorder(intr)) ||
                        speculatable_read;
      if (!safe || !info->has_dest)
         return false;
      for (unsigned i = 0; i < info->num_srcs; ++i)
         if (!outer_invariant(intr->src[i].ssa, outer, depth - 1))
            return false;
      return true;
   }
   default:
      return false;
   }
}

/* As rematerialize(), relative to `outer`. */
static nir_def *
remat_outer(nir_builder *b, nir_def *def, nir_loop *outer, struct hash_table *remap)
{
   if (def_outside_loop(def, outer))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         remat_outer(b, alu->src[i].src.ssa, outer, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         remat_outer(b, intr->src[i].ssa, outer, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *result = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, result);
   return result;
}

static unsigned
count_instrs(nir_loop *loop)
{
   unsigned n = 0;
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block)
         ++n;
   }
   return n;
}

/* Values the outer loop leaves to the code after it become phis of the two
 * copies' values. Fine for ALU results and constants; the backend reads some
 * intrinsic results from their defining instruction (bindless handles,
 * barycentrics), so any intrinsic or texture result used after the loop
 * keeps the outer loop as it is. */
static bool
live_outs_are_alu(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def || instr->type == nir_instr_type_alu ||
             instr->type == nir_instr_type_load_const ||
             instr->type == nir_instr_type_undef ||
             instr->type == nir_instr_type_phi)
            continue;
         nir_foreach_use_including_if(src, def) {
            if (!block_in_loop(use_block(src), loop))
               return false;
         }
      }
   }
   return true;
}

/* The outer loop may run under an if whose condition differs between
 * invocations: nothing in it may need uniform control flow or the other
 * invocations (barriers, subgroup and quad operations, derivatives), and only
 * its own breaks and continues may leave it. Derefs would need
 * rematerializing before the conversion to registers: refused. */
static bool
splittable_outer(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            if (nir_instr_as_jump(instr)->type != nir_jump_break &&
                nir_instr_as_jump(instr)->type != nir_jump_continue)
               return false;
            break;
         case nir_instr_type_call:
         case nir_instr_type_deref:
            return false;
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return false;
            /* A lane may wait on another lane of its subgroup through
             * memory: the two copies would run one after the other. Atomic
             * loads and stores of shared memory carry only ACCESS_ATOMIC. */
            if (strstr(nir_intrinsic_infos[intr->intrinsic].name, "atomic") ||
                (nir_intrinsic_has_access(intr) &&
                 (nir_intrinsic_access(intr) &
                  (ACCESS_COHERENT | ACCESS_VOLATILE | ACCESS_ATOMIC))))
               return false;
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy_coarse:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
               return false;
            default:
               break;
            }
            break;
         }
         default:
            break;
         }
      }
   }
   return true;
}

/*
 * A counted loop nested in another loop whose trip count the outer loop does
 * not change (a box filter's or a SAD window's x - r .. x + r): decide once,
 * before the outer loop, whether the unrolled form can run (at least
 * `factor` iterations), and otherwise run a copy of the outer loop with the
 * inner loop as it was. A 3-tap window then pays nothing for the unroll's
 * guard and structure on every outer iteration; an 11-tap one keeps it.
 * Both copies compute the same values, so the test only picks the faster.
 */
static bool
loop_has_nested_loop(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (loop_has_nested_loop(&nif->then_list) ||
             loop_has_nested_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
outer_peelable(nir_loop *outer, nir_if *terminator, nir_loop *inner)
{
   nir_foreach_block_in_cf_node(block, &outer->cf_node) {
      bool in_inner = false;
      for (nir_cf_node *p = block->cf_node.parent; p != &outer->cf_node;
           p = p->parent) {
         if (p == &inner->cf_node) {
            in_inner = true;
            break;
         }
      }
      if (in_inner)
         continue;

      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            if (block->cf_node.parent != &terminator->cf_node)
               return false;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent(intr))
               return false;
            switch (intr->intrinsic) {
            case nir_intrinsic_barrier:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy_coarse:
               return false;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            break;
         case nir_instr_type_call:
            return false;
         default:
            break;
         }
      }
   }
   return true;
}

static unsigned
count_nested_loops(struct exec_list *list)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         n += 1 + count_nested_loops(&nir_cf_node_as_loop(node)->body);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         n += count_nested_loops(&nif->then_list) +
              count_nested_loops(&nif->else_list);
      }
   }
   return n;
}

static nir_loop *
nth_loop_in_list(struct exec_list *list, unsigned *index)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         nir_loop *hit = nth_loop_in_list(&nif->then_list, index);
         if (hit)
            return hit;
         hit = nth_loop_in_list(&nif->else_list, index);
         if (hit)
            return hit;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(node);
         nir_loop *hit = nth_loop_in_list(&lp->body, index);
         if (hit)
            return hit;
         if (*index == 0)
            return lp;
         (*index)--;
      }
   }
   return NULL;
}

static bool
loop_ordinal_in_list(struct exec_list *list, nir_loop *target, unsigned *index)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (loop_ordinal_in_list(&nif->then_list, target, index) ||
             loop_ordinal_in_list(&nif->else_list, target, index))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(node);
         if (loop_ordinal_in_list(&lp->body, target, index))
            return true;
         if (lp == target)
            return true;
         (*index)++;
      }
   }
   return false;
}

/* Overflow-free exact trip-count equality predicate (trip == exact_n) for a
 * unit-step top-tested counted loop with offset == 0. */
static nir_def *
exact_trip_cond(nir_builder *b, const struct counted_loop *cl, nir_def *iv0,
                nir_def *bound, uint32_t exact_n)
{
   const bool u = cl->is_unsigned;
   const uint32_t max_u = u ? UINT32_MAX : (uint32_t)INT32_MAX;
   const uint32_t min_u = u ? 0u : (uint32_t)INT32_MIN;
   nir_def *ord, *diff, *no_wrap;

   switch (cl->rel) {
   case REL_LT:
      ord = u ? nir_ult(b, iv0, bound) : nir_ilt(b, iv0, bound);
      diff = nir_ieq_imm(b, nir_isub(b, bound, iv0), exact_n);
      return nir_iand(b, ord, diff);
   case REL_LE:
      ord = u ? nir_uge(b, bound, iv0) : nir_ige(b, bound, iv0);
      diff = nir_ieq_imm(b, nir_isub(b, bound, iv0), exact_n - 1);
      no_wrap = nir_ine_imm(b, bound, max_u);
      return nir_iand(b, nir_iand(b, ord, diff), no_wrap);
   case REL_GT:
      ord = u ? nir_ult(b, bound, iv0) : nir_ilt(b, bound, iv0);
      diff = nir_ieq_imm(b, nir_isub(b, iv0, bound), exact_n);
      return nir_iand(b, ord, diff);
   default: /* REL_GE */
      ord = u ? nir_uge(b, iv0, bound) : nir_ige(b, iv0, bound);
      diff = nir_ieq_imm(b, nir_isub(b, iv0, bound), exact_n - 1);
      no_wrap = nir_ine_imm(b, bound, min_u);
      return nir_iand(b, nir_iand(b, ord, diff), no_wrap);
   }
}

/* Peel `count` guaranteed straight-line iterations of a simple top-tested
 * counted loop. When `keep_tail` is false (exact trip == count), remove the
 * loop completely; when true (trip >= count), keep the original loop as the
 * compact remainder loop after the straight-line prefix. */
static bool
peel_straight_prefix(nir_function_impl *impl, nir_loop *loop, unsigned count,
                     bool keep_tail)
{
   struct counted_loop cl;
   if (!analyze_loop(loop, &cl))
      return false;
   if (count_unrollable(loop, cl.nif) < 0)
      return false;

   const bool rotated = cl.bottom;
   struct live_out outs[16];
   unsigned nouts = 0;
   nir_def *iv_back_def = NULL;
   nir_def *bound_def = NULL;
   if (rotated) {
      if (cl.offset != cl.step ||
          (cl.rel != REL_LT && cl.rel != REL_GT) ||
          !back_edge_live_outs(loop, outs, &nouts))
         return false;
      nir_phi_src *iv_back =
         nir_phi_get_src_from_block(cl.iv, nir_loop_last_block(loop));
      if (!iv_back)
         return false;
      iv_back_def = iv_back->src.ssa;
   }

   nir_foreach_block_in_cf_node(block, &loop->cf_node)
      nir_lower_phis_to_regs_block(block, true);
   nir_def *iv_reg =
      (rotated && keep_tail) ? latch_register(loop, iv_back_def) : NULL;
   if (rotated && keep_tail && !iv_reg) {
      nir_lower_reg_intrinsics_to_ssa_impl(impl);
      return false;
   }
   if (rotated && keep_tail) {
      nir_block *pre =
         nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
      nir_builder bb = nir_builder_at(nir_after_block_before_jump(pre));
      if (nir_scalar_is_const(cl.bound)) {
         bound_def = nir_imm_int(&bb, (uint32_t)nir_scalar_as_uint(cl.bound));
      } else {
         struct hash_table *bmap = _mesa_pointer_hash_table_create(NULL);
         nir_def *d = rematerialize(&bb, cl.bound.def, loop, bmap);
         _mesa_hash_table_destroy(bmap, NULL);
         bound_def = nir_channel(&bb, d, cl.bound.comp);
      }
   }
   nir_foreach_block_in_cf_node(block, &loop->cf_node)
      nir_lower_ssa_defs_to_regs_block(block);

   nir_cf_list head, rest;
   nir_cf_extract(&head, nir_before_cf_list(&loop->body),
                  nir_before_cf_node(&cl.nif->cf_node));
   nir_cf_extract(&rest, nir_after_cf_node(&cl.nif->cf_node),
                  nir_after_cf_list(&loop->body));

   nir_cf_list copies[2 * 4];
   assert(count >= 1 && count <= 4);
   for (unsigned i = 0; i < count; ++i) {
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      nir_cf_list_clone(&copies[2 * i], &head, loop->cf_node.parent, remap);
      nir_cf_list_clone(&copies[2 * i + 1], &rest, loop->cf_node.parent, remap);
      _mesa_hash_table_destroy(remap, NULL);
   }

   for (unsigned i = 0; i < count; ++i) {
      nir_cf_reinsert(&copies[2 * i], nir_before_cf_node(&loop->cf_node));
      nir_cf_reinsert(&copies[2 * i + 1], nir_before_cf_node(&loop->cf_node));
   }

   if (keep_tail) {
      nir_cf_reinsert(&head, nir_before_cf_node(&cl.nif->cf_node));
      nir_cf_reinsert(&rest, nir_after_cf_list(&loop->body));
      loop->partially_unrolled = true;
      if (rotated) {
         /* A rotated loop runs at least one iteration on entry, so guard it
          * with the continuation test after the straight-line prefix. */
         nir_builder b = nir_builder_at(nir_before_cf_node(&loop->cf_node));
         nir_def *cont = loop_continue_cond(&b, &cl, nir_load_reg(&b, iv_reg),
                                            bound_def);
         nir_if *tail_if = nir_push_if(&b, cont);
         nir_pop_if(&b, tail_if);
         nir_cf_list moved;
         nir_cf_extract(&moved, nir_before_cf_node(&loop->cf_node),
                        nir_after_cf_node(&loop->cf_node));
         nir_cf_reinsert(&moved, nir_before_cf_list(&tail_if->then_list));
      }
   } else if (rotated) {
      /* All `count` full iterations already ran and wrote every back-edge
       * live-out register; remove the loop. */
      nir_cf_delete(&head);
      nir_cf_delete(&rest);
      nir_cf_node_remove(&loop->cf_node);
   } else {
      /* On the exiting iteration (trip == count), the original top-tested loop
       * executes `head` once more before breaking, writing the final header-phi
       * values into their SSA-def registers. */
      nir_cf_reinsert(&head, nir_before_cf_node(&loop->cf_node));
      nir_cf_delete(&rest);
      nir_cf_node_remove(&loop->cf_node);
   }
   return true;
}

struct small_peel_rec {
   nir_loop *loop;
   unsigned count;
   bool keep_tail;
};

static bool
loop_is_compact_branchless_load_loop(nir_loop *loop, nir_if *exit_if)
{
   if (count_unrollable(loop, exit_if) <= 0)
      return false;
   unsigned ops = 0;
   bool has_load = false;
   foreach_list_typed(nir_cf_node, n, node, &loop->body) {
      if (n->type == nir_cf_node_loop)
         return false;
      if (n->type == nir_cf_node_if && nir_cf_node_as_if(n) != exit_if)
         return false;
   }
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_phi &&
             instr->type != nir_instr_type_load_const &&
             instr->type != nir_instr_type_undef &&
             instr->type != nir_instr_type_jump)
            ops++;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_load_global:
         case nir_intrinsic_load_global_constant:
         case nir_intrinsic_load_agx:
         case nir_intrinsic_load_constant_agx:
         case nir_intrinsic_load_ssbo:
            has_load = true;
            break;
         default:
            break;
         }
      }
   }
   /* Cap at 40 non-const ops so a 4x straight-line peel is <= 160 ops
    * (half of the 320-op runtime-unroll budget), bounding I-cache and GPRs. */
   return has_load && ops <= 40;
}

static bool
loop_has_8bit_load(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_global:
         case nir_intrinsic_load_global_constant:
         case nir_intrinsic_load_agx:
         case nir_intrinsic_load_constant_agx:
         case nir_intrinsic_load_ssbo:
            if (intr->def.bit_size == 8 && intr->def.num_components == 1)
               return true;
            break;
         default:
            break;
         }
      }
   }
   return false;
}

static bool
unswitch_nested(nir_function_impl *impl, unsigned factor, unsigned max_instrs,
                struct util_dynarray *small_peels,
                struct util_dynarray *outer3_peels,
                struct set *unswitched_inners)
{
   /* With maximal reconvergence the outer loop's invocations must stay
    * together; the split could separate them. */
   if (impl->function->shader->info.maximally_reconverges)
      return false;

   struct util_dynarray loops;
   util_dynarray_init(&loops, NULL);
   collect_loops(&impl->body, &loops);
   struct set *done = _mesa_pointer_set_create(NULL);
   bool progress = false;

   util_dynarray_foreach(&loops, nir_loop *, lp) {
      nir_loop *inner = *lp;
      nir_loop *outer = enclosing_loop(&inner->cf_node);
      struct counted_loop cl;
      if (!outer || _mesa_set_search(done, outer) || enclosing_loop(&outer->cf_node) ||
          outer->partially_unrolled || nir_loop_has_continue_construct(outer) ||
          !analyze_loop(inner, &cl))
         continue;
      const unsigned f = choose_factor(impl, inner, &cl, factor, max_instrs);
      nir_block *pre_inner = nir_cf_node_as_block(nir_cf_node_prev(&inner->cf_node));
      nir_phi_src *init = nir_phi_get_src_from_block(cl.iv, pre_inner);
      nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&outer->cf_node));
      if (f < 4 || !init ||
          (!nir_scalar_is_const(cl.bound) && !outer_invariant(cl.bound.def, outer, 8)))
         continue;
      if (!outer_invariant(init->src.ssa, outer, 8) || count_instrs(outer) > 256 ||
          !splittable_outer(outer) || !live_outs_are_alu(outer) ||
          (nir_block_first_instr(after) &&
           nir_block_first_instr(after)->type == nir_instr_type_phi))
         continue;

      /* The test, before the outer loop: would the main loop's first
       * iteration run? (For a rotated loop, the exact-tail form.) */
      nir_builder b = nir_builder_at(nir_before_cf_node(&outer->cf_node));
      struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
      nir_def *bound = nir_scalar_is_const(cl.bound)
                          ? nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(cl.bound))
                          : nir_channel(&b, remat_outer(&b, cl.bound.def, outer, remap),
                                        cl.bound.comp);
      nir_def *iv0 = remat_outer(&b, init->src.ssa, outer, remap);
      _mesa_hash_table_destroy(remap, NULL);
      if (cl.offset && !cl.bottom)
         iv0 = nir_iadd_imm(&b, iv0, cl.offset);
      nir_def *g = guard_bound(&b, &cl, bound, f, false);
      nir_def *cond;
      if (cl.rel == REL_LT || cl.rel == REL_LE)
         cond = cl.is_unsigned ? nir_ult(&b, iv0, g) : nir_ilt(&b, iv0, g);
      else
         cond = cl.is_unsigned ? nir_ult(&b, g, iv0) : nir_ilt(&b, g, iv0);

      /* For compact 2D stencils / box filters with f >= 8, also specialize
       * the small-window else arm (4 <= trip < 8 -> straight-line 4x + tail,
       * and trip == 3 -> straight-line 3x with zero inner loop) so 3x3..7x7
       * windows vectorize into u8x3/u8x4 loads instead of scalar byte loops. */
      struct live_out inner_outs[16];
      unsigned inner_nouts = 0;
      const bool inner_rot_ok =
         cl.bottom && cl.offset == cl.step &&
         (cl.rel == REL_LT || cl.rel == REL_GT) &&
         back_edge_live_outs(inner, inner_outs, &inner_nouts);
      const bool inner_peel_ok =
         (!cl.bottom && cl.offset == 0) || inner_rot_ok;
      const bool split_small =
         inner_peel_ok &&
         (cl.step == 1 || cl.step == -1) &&
         count_instrs(outer) <= 96 && !loop_has_nested_loop(&inner->body) &&
         ((f >= 8 && loop_has_8bit_load(inner)) ||
          (f >= 4 && inner_rot_ok &&
           (nir_src_is_const(init->src) && nir_src_as_int(init->src) == 0) &&
           count_instrs(outer) <= 80 &&
           loop_is_compact_branchless_load_loop(inner, cl.nif)));
      const bool split_peel4 = split_small && f >= 8;
      nir_def *cond4 = NULL, *cond3 = NULL, *cond3_outer = NULL;
      unsigned inner_ord = 0;
      if (split_small) {
         if (split_peel4) {
            nir_def *g4 = guard_bound(&b, &cl, bound, 4, false);
            if (cl.rel == REL_LT || cl.rel == REL_LE)
               cond4 = cl.is_unsigned ? nir_ult(&b, iv0, g4) : nir_ilt(&b, iv0, g4);
            else
               cond4 = cl.is_unsigned ? nir_ult(&b, g4, iv0) : nir_ilt(&b, g4, iv0);
         }
         cond3 = exact_trip_cond(&b, &cl, iv0, bound, 3);
         bool found_ord = loop_ordinal_in_list(&outer->body, inner, &inner_ord);
         assert(found_ord);

         struct counted_loop ocl;
         struct live_out outer_outs[16];
         unsigned outer_nouts = 0;
         nir_block *pre_outer = nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
         if (analyze_loop_opts(outer, &ocl, true) &&
             ((!ocl.bottom && ocl.offset == 0) ||
              (ocl.bottom && ocl.offset == ocl.step &&
               (ocl.rel == REL_LT || ocl.rel == REL_GT) &&
               back_edge_live_outs(outer, outer_outs, &outer_nouts))) &&
             (ocl.step == 1 || ocl.step == -1) &&
             count_nested_loops(&outer->body) == 1 &&
             /* Preserve the existing 64-op cap for top-tested 8-bit 3x3 nests;
              * rotated nests carry extra latch step/compare/init ops before
              * post-peel DCE, and 3 * 80 = 240 stays below the 256-op split cap. */
             count_instrs(outer) <= (ocl.bottom ? 80u : 64u) &&
             outer_peelable(outer, ocl.nif, inner)) {
            nir_phi_src *oinit = nir_phi_get_src_from_block(ocl.iv, pre_outer);
            if (oinit && (nir_scalar_is_const(ocl.bound) ||
                          outer_invariant(ocl.bound.def, outer, 8))) {
               struct hash_table *oremap = _mesa_pointer_hash_table_create(NULL);
               nir_def *obound =
                  nir_scalar_is_const(ocl.bound)
                     ? nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(ocl.bound))
                     : nir_channel(&b, remat_outer(&b, ocl.bound.def, outer, oremap),
                                   ocl.bound.comp);
               _mesa_hash_table_destroy(oremap, NULL);
               cond3_outer = exact_trip_cond(&b, &ocl, oinit->src.ssa, obound, 3);
            }
         }
      }

      /* Values of the outer loop through registers, so both copies write
       * the same ones and the code after reads them whichever ran. Phi
       * sources are written in their own predecessor: hoisted into the arms
       * of an if, an inner latch value would come back as a phi of the two
       * arms, which the inner loop's analysis does not take. */
      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_phis_to_regs_block(block, true);
      nir_foreach_block_in_cf_node(block, &outer->cf_node)
         nir_lower_ssa_defs_to_regs_block(block);

      /* The lowering writes the outer phis' initial values at the end of the
       * preheader, after cond: the if goes after those, right before the
       * outer loop, so both copies start from them. */
      b.cursor = nir_before_cf_node(&outer->cf_node);
      nir_if *nif = nir_push_if(&b, cond);
      nir_pop_if(&b, nif);
      nir_cf_list list, copy;
      nir_cf_extract(&list, nir_before_cf_node(&outer->cf_node),
                     nir_after_cf_node(&outer->cf_node));
      if (split_small) {
         b.cursor = nir_before_cf_list(&nif->else_list);
         nir_if *nif4 = NULL;
         if (split_peel4) {
            nif4 = nir_push_if(&b, cond4);
            nir_push_else(&b, nif4);
         }
         nir_if *nif3 = nir_push_if(&b, cond3);
         nir_pop_if(&b, nif3);
         if (nif4)
            nir_pop_if(&b, nif4);

         nir_cf_list copy4, copy3;
         if (nif4) {
            struct hash_table *cmap4 = _mesa_pointer_hash_table_create(NULL);
            nir_cf_list_clone(&copy4, &list, &nif4->cf_node, cmap4);
            _mesa_hash_table_destroy(cmap4, NULL);
         }

         struct hash_table *cmap3 = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy3, &list, &nif3->cf_node, cmap3);
         _mesa_hash_table_destroy(cmap3, NULL);

         inner->partially_unrolled = true;
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy, &list, &nif3->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);
         inner->partially_unrolled = false;

         nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
         if (nif4)
            nir_cf_reinsert(&copy4, nir_before_cf_list(&nif4->then_list));
         nir_cf_reinsert(&copy, nir_before_cf_list(&nif3->else_list));

         if (nif4) {
            unsigned idx4 = inner_ord;
            nir_loop *inner4 = nth_loop_in_list(&nif4->then_list, &idx4);
            assert(inner4);
            struct small_peel_rec r4 = {inner4, 4, true};
            util_dynarray_append(small_peels, r4);
         }

         if (cond3_outer) {
            b.cursor = nir_before_cf_list(&nif3->then_list);
            nir_if *nif33 = nir_push_if(&b, cond3_outer);
            nir_pop_if(&b, nif33);

            nir_cf_list copy33;
            struct hash_table *cmap33 = _mesa_pointer_hash_table_create(NULL);
            nir_cf_list_clone(&copy33, &copy3, &nif33->cf_node, cmap33);
            _mesa_hash_table_destroy(cmap33, NULL);

            nir_cf_reinsert(&copy33, nir_before_cf_list(&nif33->then_list));
            nir_cf_reinsert(&copy3, nir_before_cf_list(&nif33->else_list));

            unsigned idx33 = inner_ord;
            nir_loop *inner33 = nth_loop_in_list(&nif33->then_list, &idx33);
            assert(inner33);
            struct small_peel_rec r33 = {inner33, 3, false};
            util_dynarray_append(small_peels, r33);

            nir_loop *outer33 = enclosing_loop(&inner33->cf_node);
            assert(outer33);
            struct small_peel_rec ro33 = {outer33, 3, false};
            util_dynarray_append(outer3_peels, ro33);

            unsigned idx3 = inner_ord;
            nir_loop *inner3 = nth_loop_in_list(&nif33->else_list, &idx3);
            assert(inner3);
            struct small_peel_rec r3 = {inner3, 3, false};
            util_dynarray_append(small_peels, r3);
         } else {
            nir_cf_reinsert(&copy3, nir_before_cf_list(&nif3->then_list));

            unsigned idx3 = inner_ord;
            nir_loop *inner3 = nth_loop_in_list(&nif3->then_list, &idx3);
            assert(inner3);
            struct small_peel_rec r3 = {inner3, 3, false};
            util_dynarray_append(small_peels, r3);
         }
      } else {
         /* The copy keeps this inner loop as it is (clones copy the flag); its
          * other loops stay candidates. */
         inner->partially_unrolled = true;
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copy, &list, &nif->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);
         inner->partially_unrolled = false;
         nir_cf_reinsert(&list, nir_before_cf_list(&nif->then_list));
         nir_cf_reinsert(&copy, nir_before_cf_list(&nif->else_list));
      }
      _mesa_set_add(unswitched_inners, inner);
      _mesa_set_add(done, outer);
      progress = true;
   }

   _mesa_set_destroy(done, NULL);
   util_dynarray_fini(&loops);
   return progress;
}

static void
collect_loops(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops(&nif->then_list, loops);
         collect_loops(&nif->else_list, loops);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops(&loop->body, loops);
         util_dynarray_append(loops, loop);
         break;
      }
      default:
         break;
      }
   }
}

bool
agx_nir_unroll_runtime_loops(nir_shader *shader, unsigned factor,
                             unsigned max_instrs, unsigned *widest)
{
   bool any = false;
   assert(factor >= 2 && factor <= 16);

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      bool unswitched = false;
      struct util_dynarray small_peels, outer3_peels;
      util_dynarray_init(&small_peels, NULL);
      util_dynarray_init(&outer3_peels, NULL);
      struct set *unswitched_inners = _mesa_pointer_set_create(NULL);
      if (unswitch_nested(impl, factor, max_instrs, &small_peels, &outer3_peels,
                          unswitched_inners)) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
         bool peeled = false;
         util_dynarray_foreach(&small_peels, struct small_peel_rec, rec) {
            if (peel_straight_prefix(impl, rec->loop, rec->count,
                                     rec->keep_tail))
               peeled = true;
            else
               rec->loop->partially_unrolled = true;
         }
         if (peeled) {
            nir_progress(true, impl, nir_metadata_none);
            nir_lower_reg_intrinsics_to_ssa_impl(impl);
            nir_metadata_require(impl, nir_metadata_block_index);
            nir_index_ssa_defs(impl);
            bool outer_peeled = false;
            util_dynarray_foreach(&outer3_peels, struct small_peel_rec, rec) {
               if (peel_straight_prefix(impl, rec->loop, rec->count,
                                        rec->keep_tail))
                  outer_peeled = true;
            }
            if (outer_peeled) {
               nir_progress(true, impl, nir_metadata_none);
               nir_lower_reg_intrinsics_to_ssa_impl(impl);
               nir_metadata_require(impl, nir_metadata_block_index);
               nir_index_ssa_defs(impl);
            }
         }
         unswitched = true;
         any = true;
      }
      util_dynarray_fini(&small_peels);
      util_dynarray_fini(&outer3_peels);

      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops(&impl->body, &loops);

      struct counted_loop candidates[64];
      nir_def *bounds[64];
      unsigned factors[64];
      unsigned n = 0;
      util_dynarray_foreach(&loops, nir_loop *, lp) {
         struct counted_loop cl;
         if (n == ARRAY_SIZE(candidates) || !analyze_loop(*lp, &cl))
            continue;
         const unsigned f = choose_factor(impl, *lp, &cl, factor, max_instrs);
         if (!f)
            continue;

         /* The bound as seen from the preheader. */
         nir_scalar bound = cl.bound;
         nir_block *pre =
            nir_cf_node_as_block(nir_cf_node_prev(&(*lp)->cf_node));
         nir_builder b = nir_builder_at(nir_after_block_before_jump(pre));
         if (nir_scalar_is_const(bound)) {
            bounds[n] = nir_imm_int(&b, (uint32_t)nir_scalar_as_uint(bound));
         } else {
            /* Defined outside the loop and read in it, so it dominates the
             * preheader's end, or computed alike by every iteration: then
             * computed once there. */
            struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
            nir_def *def = rematerialize(&b, bound.def, *lp, remap);
            _mesa_hash_table_destroy(remap, NULL);
            bounds[n] = nir_channel(&b, def, bound.comp);
         }
         factors[n] = f;
         if (widest)
            *widest = MAX2(*widest, f);
         candidates[n++] = cl;
      }
      util_dynarray_fini(&loops);

      for (unsigned i = 0; i < n; ++i)
         unroll_loop(impl, &candidates[i], bounds[i], factors[i], max_instrs,
                     &bounds[i + 1], n - i - 1,
                     !_mesa_set_search(unswitched_inners, candidates[i].loop));
      _mesa_set_destroy(unswitched_inners, NULL);

      if (n) {
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         any = true;
      } else if (unswitched) {
         nir_progress(true, impl, nir_metadata_none);
      } else {
         nir_no_progress(impl);
      }
   }
   return any;
}

static bool
pure_loop_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!pure_loop_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_preamble ||
          intr->intrinsic == nir_intrinsic_load_push_constant ||
          (intr->intrinsic == nir_intrinsic_load_global_constant &&
           (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE))) {
         const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
         for (unsigned i = 0; i < info->num_srcs; ++i)
            if (!pure_loop_invariant(intr->src[i].ssa, loop, depth - 1))
               return false;
         return true;
      }
      return false;
   }
   default:
      return false;
   }
}

static bool
cf_list_has_jump(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         if (nir_block_ends_in_jump(nir_cf_node_as_block(node)))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_jump(&nif->then_list) ||
             cf_list_has_jump(&nif->else_list))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         if (cf_list_has_jump(&nir_cf_node_as_loop(node)->body))
            return true;
      }
   }
   return false;
}

static bool
cf_list_has_outer_jump(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         if (nir_block_ends_in_jump(nir_cf_node_as_block(node)))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_outer_jump(&nif->then_list) ||
             cf_list_has_outer_jump(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
outer_has_jump_outside(struct exec_list *list, nir_if *skip_term)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         if (nir_block_ends_in_jump(nir_cf_node_as_block(node)))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nif == skip_term)
            continue;
         if (outer_has_jump_outside(&nif->then_list, skip_term) ||
             outer_has_jump_outside(&nif->else_list, skip_term))
            return true;
      }
   }
   return false;
}

static void
count_arm_work(struct exec_list *list, nir_loop *loop, unsigned *total,
               unsigned *inv_alu)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type == nir_instr_type_load_const ||
                instr->type == nir_instr_type_undef ||
                instr->type == nir_instr_type_phi)
               continue;
            (*total)++;
            if (instr->type == nir_instr_type_alu &&
                pure_loop_invariant(&nir_instr_as_alu(instr)->def, loop, 8))
               (*inv_alu)++;
         }
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         count_arm_work(&nif->then_list, loop, total, inv_alu);
         count_arm_work(&nif->else_list, loop, total, inv_alu);
      } else if (node->type == nir_cf_node_loop) {
         count_arm_work(&nir_cf_node_as_loop(node)->body, loop, total, inv_alu);
      }
   }
}

/* Duplicating a loop around an invariant condition uses the same
 * extract/clone/reinsert transform as outer-loop splitting, so reuse its
 * structural guards: no derefs/barriers/atomics/subgroups/return/halt
 * (splittable_outer), no non-ALU live-outs (live_outs_are_alu), and no phis
 * in the loop successor block (which would keep stale break-block sources
 * once the loop moves into outer_if->then_list).
 *
 * The 192-instruction cap is 3/4 of the 256-instruction outer-loop split cap
 * so duplicating an innermost loop stays well within instruction-cache limits. */
static bool
loop_safe_to_unswitch_if(nir_loop *loop, bool allow_nested)
{
   if (!nir_loop_has_back_edge(loop) ||
       (!allow_nested && loop_has_nested_loop(&loop->body)))
      return false;
   if (nir_loop_has_continue_construct(loop) || count_instrs(loop) > 192)
      return false;

   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (nir_block_first_instr(after) &&
       nir_block_first_instr(after)->type == nir_instr_type_phi)
      return false;

   if (!splittable_outer(loop) || !live_outs_are_alu(loop))
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (nir_block_ends_in_return_or_halt(block))
         return false;
   }
   return true;
}

bool
agx_nir_opt_unswitch_invariant_if(nir_shader *shader)
{
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      struct util_dynarray loops;
      util_dynarray_init(&loops, NULL);
      collect_loops(&impl->body, &loops);

      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         nir_loop *loop = *lp;
         const bool has_inner_loop = loop_has_nested_loop(&loop->body);
         if (!loop_safe_to_unswitch_if(loop, true))
            continue;

         nir_if *target = NULL;
         unsigned target_ord = 0, ord = 0;
         foreach_list_typed(nir_cf_node, node, node, &loop->body) {
            if (node->type != nir_cf_node_if)
               continue;
            nir_if *nif = nir_cf_node_as_if(node);
            unsigned cur_ord = ord++;
            if (cf_list_has_outer_jump(&nif->then_list) ||
                cf_list_has_outer_jump(&nif->else_list))
               continue;
            /* Depth 8 matches outer_invariant() in outer-loop splitting. */
            if (!pure_loop_invariant(nif->condition.ssa, loop, 8))
               continue;

            unsigned t_tot = 0, t_inv = 0, e_tot = 0, e_inv = 0;
            count_arm_work(&nif->then_list, loop, &t_tot, &t_inv);
            count_arm_work(&nif->else_list, loop, &e_tot, &e_inv);
            const bool t_loop = loop_has_nested_loop(&nif->then_list);
            const bool e_loop = loop_has_nested_loop(&nif->else_list);

            if (has_inner_loop) {
               /* Never duplicate an inner loop: only unswitch around `nif`
                * when all nested loops in `loop` live inside a single arm of
                * `nif` and the loop-free arm is compact (<= 16 ops, e.g. a
                * border copy or zero-fill). */
               if (t_loop == e_loop)
                  continue;
               bool other_loop = false;
               foreach_list_typed(nir_cf_node, sib, node, &loop->body) {
                  if (sib == &nif->cf_node)
                     continue;
                  if (sib->type == nir_cf_node_loop) {
                     other_loop = true;
                     break;
                  }
                  if (sib->type == nir_cf_node_if) {
                     nir_if *snif = nir_cf_node_as_if(sib);
                     if (loop_has_nested_loop(&snif->then_list) ||
                         loop_has_nested_loop(&snif->else_list)) {
                        other_loop = true;
                        break;
                     }
                  }
               }
               if (other_loop)
                  continue;
               const unsigned small_tot = t_loop ? e_tot : t_tot;
               const unsigned big_tot = t_loop ? t_tot : e_tot;
               if (small_tot > 16 || big_tot < 8)
                  continue;
            } else {
               /* Require at least one substantial arm (>= 12 ops, so a small
                * select/copy is not worth duplicating the loop for) and at
                * least 2 loop-invariant ALU ops exposed across the arms for
                * subsequent LICM. */
               if ((t_tot < 12 && e_tot < 12) || (t_inv + e_inv) < 2)
                  continue;
            }

            target = nif;
            target_ord = cur_ord;
            break;
         }
         if (!target)
            continue;

         b.cursor = nir_before_cf_node(&loop->cf_node);
         struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
         nir_def *cond = remat_outer(&b, target->condition.ssa, loop, remap);
         _mesa_hash_table_destroy(remap, NULL);

         nir_foreach_block_in_cf_node(block, &loop->cf_node)
            nir_lower_phis_to_regs_block(block, true);
         nir_foreach_block_in_cf_node(block, &loop->cf_node)
            nir_lower_ssa_defs_to_regs_block(block);

         b.cursor = nir_before_cf_node(&loop->cf_node);
         nir_if *outer_if = nir_push_if(&b, cond);
         nir_pop_if(&b, outer_if);

         nir_cf_list orig_list, clone_list;
         nir_cf_extract(&orig_list, nir_before_cf_node(&loop->cf_node),
                        nir_after_cf_node(&loop->cf_node));
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&clone_list, &orig_list, &outer_if->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);

         nir_cf_reinsert(&orig_list, nir_before_cf_list(&outer_if->then_list));
         nir_cf_reinsert(&clone_list, nir_before_cf_list(&outer_if->else_list));

         for (unsigned arm = 0; arm < 2; ++arm) {
            struct exec_list *arm_list =
               arm == 0 ? &outer_if->then_list : &outer_if->else_list;
            nir_loop *arm_loop = NULL;
            foreach_list_typed(nir_cf_node, n, node, arm_list) {
               if (n->type == nir_cf_node_loop) {
                  arm_loop = nir_cf_node_as_loop(n);
                  break;
               }
            }
            assert(arm_loop);
            unsigned k = 0;
            foreach_list_typed(nir_cf_node, n, node, &arm_loop->body) {
               if (n->type != nir_cf_node_if)
                  continue;
               if (k++ == target_ord) {
                  nir_if *arm_if = nir_cf_node_as_if(n);
                  b.cursor = nir_before_cf_node(&arm_if->cf_node);
                  nir_src_rewrite(&arm_if->condition,
                                  nir_imm_bool(&b, arm == 0));
                  break;
               }
            }
         }

         impl_progress = true;
         /* Re-index after modifying CF before considering outer loops. */
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
      }

      util_dynarray_fini(&loops);
      if (impl_progress)
         any = true;
   }

   if (any) {
      NIR_PASS(_, shader, nir_opt_dead_cf);
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_dce);
      NIR_PASS(_, shader, nir_opt_cse);
   }

   return any;
}

/* Check whether `def` can be rematerialized inside a guard that is only taken
 * when iteration 0 of `loop` (with coord(0) = base >= 0) executes. Permits
 * pure ALU, preamble/push-constant loads, and read-only global constant loads.
 */
static bool
limit_loop_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         if (!limit_loop_invariant(alu->src[i].src.ssa, loop, depth - 1))
            return false;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_preamble ||
          intr->intrinsic == nir_intrinsic_load_push_constant ||
          (intr->intrinsic == nir_intrinsic_load_global_constant &&
           !(nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))) {
         const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
         for (unsigned i = 0; i < info->num_srcs; ++i)
            if (!limit_loop_invariant(intr->src[i].ssa, loop, depth - 1))
               return false;
         return true;
      }
      return false;
   }
   default:
      return false;
   }
}

static nir_def *
remat_limit(nir_builder *b, nir_def *def, nir_loop *loop,
            struct hash_table *remap)
{
   if (def_outside_loop(def, loop))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         remat_limit(b, alu->src[i].src.ssa, loop, remap);
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i)
         remat_limit(b, intr->src[i].ssa, loop, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *res = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, res);
   return res;
}

static bool
is_const_i32(nir_scalar s, int32_t val)
{
   return nir_scalar_is_const(s) && s.def->bit_size == 32 &&
          nir_scalar_as_int(s) == val;
}

static bool
is_const_bool(nir_scalar s, bool val)
{
   return nir_scalar_is_const(s) && s.def->bit_size == 1 &&
          (nir_scalar_as_uint(s) != 0) == val;
}

/* Affine 32-bit coordinate `coord = base_sum + iv * stride` in Z/2^32Z.
 * `base_terms[i]` are added; if `base_mul[i]` is non-NULL, the term is
 * `base_terms[i] * base_mul[i]`, else `base_terms[i]`. `stride == NULL` means 1.
 */
struct affine_iv {
   nir_def *base_terms[4];
   nir_def *base_mul[4];
   unsigned nbase;
   nir_def *stride;
};

static bool
affine_add_term(struct affine_iv *aff, nir_def *a, nir_def *b)
{
   if (aff->nbase >= ARRAY_SIZE(aff->base_terms))
      return false;
   aff->base_terms[aff->nbase] = a;
   aff->base_mul[aff->nbase] = b;
   aff->nbase++;
   return true;
}

static bool
match_affine_iv_rec(nir_scalar coord, nir_phi_instr *iv, nir_loop *loop,
                    bool allow_limit_inv, unsigned depth,
                    struct affine_iv *aff)
{
   if (depth == 0)
      return false;
   coord = nir_scalar_resolved(coord.def, coord.comp);
   if (coord.def->bit_size != 32 || coord.def->num_components != 1)
      return false;

   if (coord.def == &iv->def) {
      aff->stride = NULL;
      return true;
   }

   if (!nir_scalar_is_alu(coord))
      return false;

   nir_op op = nir_scalar_alu_op(coord);
   if (op == nir_op_imul) {
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar a = nir_scalar_chase_alu_src(coord, k);
         nir_scalar b = nir_scalar_chase_alu_src(coord, 1 - k);
         bool inv = allow_limit_inv ? limit_loop_invariant(b.def, loop, 8)
                                    : pure_loop_invariant(b.def, loop, 8);
         if (a.def == &iv->def && b.def->num_components == 1 && inv) {
            aff->stride = b.def;
            return true;
         }
      }
      return false;
   }

   if (op == nir_op_iadd) {
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar term = nir_scalar_chase_alu_src(coord, k);
         nir_scalar base = nir_scalar_chase_alu_src(coord, 1 - k);
         bool inv = allow_limit_inv ? limit_loop_invariant(base.def, loop, 8)
                                    : pure_loop_invariant(base.def, loop, 8);
         if (base.def->num_components != 1 || !inv)
            continue;
         struct affine_iv tmp = *aff;
         if (affine_add_term(&tmp, base.def, NULL) &&
             match_affine_iv_rec(term, iv, loop, allow_limit_inv, depth - 1,
                                 &tmp)) {
            *aff = tmp;
            return true;
         }
      }
      return false;
   }

   if (op == nir_op_imadshl_agx) {
      nir_scalar sh = nir_scalar_chase_alu_src(coord, 3);
      if (!is_const_i32(sh, 0))
         return false;
      nir_scalar a = nir_scalar_chase_alu_src(coord, 0);
      nir_scalar b = nir_scalar_chase_alu_src(coord, 1);
      nir_scalar c = nir_scalar_chase_alu_src(coord, 2);
      bool a_inv = a.def->num_components == 1 &&
                   (allow_limit_inv ? limit_loop_invariant(a.def, loop, 8)
                                    : pure_loop_invariant(a.def, loop, 8));
      bool b_inv = b.def->num_components == 1 &&
                   (allow_limit_inv ? limit_loop_invariant(b.def, loop, 8)
                                    : pure_loop_invariant(b.def, loop, 8));
      bool c_inv = c.def->num_components == 1 &&
                   (allow_limit_inv ? limit_loop_invariant(c.def, loop, 8)
                                    : pure_loop_invariant(c.def, loop, 8));

      /* Case 1: iv * inv + inv_c */
      if (c_inv) {
         if (a.def == &iv->def && b_inv) {
            if (!affine_add_term(aff, c.def, NULL))
               return false;
            aff->stride = b.def;
            return true;
         }
         if (b.def == &iv->def && a_inv) {
            if (!affine_add_term(aff, c.def, NULL))
               return false;
            aff->stride = a.def;
            return true;
         }
      }
      /* Case 2: inv_a * inv_b + affine(iv), e.g. oy * stride_y + (ky * dil_y - pad_y) */
      if (a_inv && b_inv) {
         struct affine_iv tmp = *aff;
         if (affine_add_term(&tmp, a.def, b.def) &&
             match_affine_iv_rec(c, iv, loop, allow_limit_inv, depth - 1,
                                 &tmp)) {
            *aff = tmp;
            return true;
         }
      }
   }

   return false;
}

static bool
match_affine_iv(nir_scalar coord, nir_phi_instr *iv, nir_loop *loop,
                bool allow_limit_inv, struct affine_iv *aff)
{
   memset(aff, 0, sizeof(*aff));
   return match_affine_iv_rec(coord, iv, loop, allow_limit_inv, 4, aff);
}

static bool
affine_invariant_in(const struct affine_iv *aff, nir_loop *outer,
                    bool allow_limit_inv)
{
   for (unsigned i = 0; i < aff->nbase; ++i) {
      bool ok = allow_limit_inv
                   ? limit_loop_invariant(aff->base_terms[i], outer, 8)
                   : pure_loop_invariant(aff->base_terms[i], outer, 8);
      if (!ok)
         return false;
      if (aff->base_mul[i]) {
         bool mok = allow_limit_inv
                       ? limit_loop_invariant(aff->base_mul[i], outer, 8)
                       : pure_loop_invariant(aff->base_mul[i], outer, 8);
         if (!mok)
            return false;
      }
   }
   if (aff->stride) {
      bool sok = allow_limit_inv
                    ? limit_loop_invariant(aff->stride, outer, 8)
                    : pure_loop_invariant(aff->stride, outer, 8);
      if (!sok)
         return false;
   }
   return true;
}

static nir_def *
emit_affine_base(nir_builder *b, const struct affine_iv *aff, nir_loop *loop,
                 struct hash_table *remap, bool use_limit_remat)
{
   if (aff->nbase == 0)
      return nir_imm_int(b, 0);
   nir_def *acc = NULL;
   for (unsigned i = 0; i < aff->nbase; ++i) {
      nir_def *a = use_limit_remat
                      ? remat_limit(b, aff->base_terms[i], loop, remap)
                      : remat_outer(b, aff->base_terms[i], loop, remap);
      nir_def *term = a;
      if (aff->base_mul[i]) {
         nir_def *m = use_limit_remat
                         ? remat_limit(b, aff->base_mul[i], loop, remap)
                         : remat_outer(b, aff->base_mul[i], loop, remap);
         term = nir_imul(b, a, m);
      }
      acc = acc ? nir_iadd(b, acc, term) : term;
   }
   return acc;
}

/* Match `coord >= 0` (returns +1) or `coord < 0` (returns -1), else 0. */
static int
match_ge_zero(nir_scalar cmp, nir_scalar *out_coord)
{
   bool neg = false;
   cmp = nir_scalar_resolved(cmp.def, cmp.comp);
   while (nir_scalar_is_alu(cmp) && nir_scalar_alu_op(cmp) == nir_op_inot) {
      neg = !neg;
      cmp = nir_scalar_chase_alu_src(cmp, 0);
   }
   if (!nir_scalar_is_alu(cmp))
      return 0;
   nir_op op = nir_scalar_alu_op(cmp);
   if (op != nir_op_ige && op != nir_op_ilt)
      return 0;
   nir_scalar a = nir_scalar_chase_alu_src(cmp, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cmp, 1);
   int dir = 0;
   if (is_const_i32(b, 0)) {
      *out_coord = a;
      dir = (op == nir_op_ige) ? 1 : -1;
   } else if (is_const_i32(a, 0)) {
      return 0;
   }
   return neg ? -dir : dir;
}

/* Match `coord < limit` (returns +1) or `coord >= limit` (returns -1) for
 * exact matching `coord`. */
static int
match_lt_limit(nir_scalar cmp, nir_scalar coord, nir_loop *loop,
               nir_def **out_limit)
{
   bool neg = false;
   cmp = nir_scalar_resolved(cmp.def, cmp.comp);
   while (nir_scalar_is_alu(cmp) && nir_scalar_alu_op(cmp) == nir_op_inot) {
      neg = !neg;
      cmp = nir_scalar_chase_alu_src(cmp, 0);
   }
   if (!nir_scalar_is_alu(cmp))
      return 0;
   nir_op op = nir_scalar_alu_op(cmp);
   if (op != nir_op_ilt && op != nir_op_ige)
      return 0;
   nir_scalar a = nir_scalar_resolved(nir_scalar_chase_alu_src(cmp, 0).def,
                                      nir_scalar_chase_alu_src(cmp, 0).comp);
   nir_scalar b = nir_scalar_resolved(nir_scalar_chase_alu_src(cmp, 1).def,
                                      nir_scalar_chase_alu_src(cmp, 1).comp);
   coord = nir_scalar_resolved(coord.def, coord.comp);

   int dir = 0;
   if (a.def == coord.def && a.comp == coord.comp &&
       b.def->num_components == 1 && limit_loop_invariant(b.def, loop, 8)) {
      *out_limit = b.def;
      dir = (op == nir_op_ilt) ? 1 : -1;
   }
   return neg ? -dir : dir;
}

struct interior_guard {
   nir_if *body_if;
   unsigned body_if_ord;
   bool body_in_then;
   bool has_sc_if;
   unsigned sc_if_ord;
   bool sc_if_val;
   struct affine_iv aff;
   nir_def *limit;
};

/* Match an in-bounds predicate proving `0 <= coord && coord < limit` where
 * `coord = base + iv * stride`. */
static bool
match_bounds_predicate(nir_scalar cond, bool want_true, nir_phi_instr *iv,
                       nir_loop *loop, bool allow_limit_inv,
                       struct affine_iv *out_aff, nir_def **out_limit,
                       nir_if **out_sc_if, bool *out_sc_val)
{
   cond = nir_scalar_resolved(cond.def, cond.comp);
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      want_true = !want_true;
      cond = nir_scalar_chase_alu_src(cond, 0);
   }

   /* Form 1: short-circuit diamond phi right before body_if */
   if (nir_def_instr(cond.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(cond.def));
      nir_block *pblk = phi->instr.block;
      if (nir_cf_node_is_first(&pblk->cf_node))
         return false;
      nir_cf_node *prev = nir_cf_node_prev(&pblk->cf_node);
      if (prev->type != nir_cf_node_if)
         return false;
      nir_if *sc_if = nir_cf_node_as_if(prev);
      nir_block *t_blk = nir_if_first_then_block(sc_if);
      nir_block *e_blk = nir_if_first_else_block(sc_if);
      if (nir_if_last_then_block(sc_if) != t_blk ||
          nir_if_last_else_block(sc_if) != e_blk)
         return false;

      nir_scalar coord;
      int ge0 = match_ge_zero(nir_get_scalar(sc_if->condition.ssa, 0), &coord);
      if (ge0 == 0)
         return false;
      if (!match_affine_iv(coord, iv, loop, allow_limit_inv, out_aff))
         return false;

      nir_phi_src *ts = nir_phi_get_src_from_block(phi, t_blk);
      nir_phi_src *es = nir_phi_get_src_from_block(phi, e_blk);
      if (!ts || !es)
         return false;

      nir_scalar nonneg_val = nir_get_scalar(ge0 > 0 ? ts->src.ssa : es->src.ssa, 0);
      nir_scalar neg_val = nir_get_scalar(ge0 > 0 ? es->src.ssa : ts->src.ssa, 0);
      if (!is_const_bool(neg_val, !want_true))
         return false;

      int lt = match_lt_limit(nonneg_val, coord, loop, out_limit);
      if (lt == (want_true ? 1 : -1)) {
         *out_sc_if = sc_if;
         *out_sc_val = (ge0 > 0);
         return true;
      }
      return false;
   }

   if (!nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);

   /* Form 2: bcsel(coord >= 0, coord < limit, false) or negated */
   if (op == nir_op_bcsel) {
      nir_scalar sel_c = nir_scalar_chase_alu_src(cond, 0);
      nir_scalar sel_t = nir_scalar_chase_alu_src(cond, 1);
      nir_scalar sel_e = nir_scalar_chase_alu_src(cond, 2);
      nir_scalar coord;
      int ge0 = match_ge_zero(sel_c, &coord);
      if (ge0 == 0)
         return false;
      if (!match_affine_iv(coord, iv, loop, allow_limit_inv, out_aff))
         return false;
      nir_scalar nonneg_val = ge0 > 0 ? sel_t : sel_e;
      nir_scalar neg_val = ge0 > 0 ? sel_e : sel_t;
      if (!is_const_bool(neg_val, !want_true))
         return false;
      int lt = match_lt_limit(nonneg_val, coord, loop, out_limit);
      return lt == (want_true ? 1 : -1);
   }

   /* Form 3: iand(coord >= 0, coord < limit) or ior(coord < 0, coord >= limit) */
   if ((want_true && op == nir_op_iand) || (!want_true && op == nir_op_ior)) {
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar s0 = nir_scalar_chase_alu_src(cond, k);
         nir_scalar s1 = nir_scalar_chase_alu_src(cond, 1 - k);
         nir_scalar coord;
         int ge0 = match_ge_zero(s0, &coord);
         if (ge0 != (want_true ? 1 : -1))
            continue;
         if (!match_affine_iv(coord, iv, loop, allow_limit_inv, out_aff))
            continue;
         int lt = match_lt_limit(s1, coord, loop, out_limit);
         if (lt == (want_true ? 1 : -1))
            return true;
      }
      return false;
   }

   /* Form 4: unsigned range check ult(coord, limit) / uge(coord, limit) */
   if ((want_true && op == nir_op_ult) || (!want_true && op == nir_op_uge)) {
      nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
      nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
      if (b.def->num_components == 1 && limit_loop_invariant(b.def, loop, 8) &&
          match_affine_iv(a, iv, loop, allow_limit_inv, out_aff)) {
         *out_limit = b.def;
         return true;
      }
   }

   return false;
}

/* Check that `def` is safe to evaluate once `bound > 0` (i.e. defined outside
 * `loop` or in `loop`'s header block before any nested CF, with only pure ALU
 * and read-only constant loads). */
static bool
header_safe_invariant(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (def_outside_loop(def, loop))
      return true;
   if (depth == 0)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != nir_loop_first_block(loop))
      return false;
   return limit_loop_invariant(def, loop, depth);
}

static bool
find_interior_guard(nir_loop *loop, struct counted_loop *cl,
                    bool allow_nested, bool allow_limit_inv,
                    struct interior_guard *out)
{
   if (!analyze_loop_opts(loop, cl, allow_nested))
      return false;
   if (cl->is_unsigned || cl->rel != REL_LT || cl->step != 1)
      return false;
   if ((!cl->bottom && cl->offset != 0) || (cl->bottom && cl->offset != 1))
      return false;
   if (!loop_safe_to_unswitch_if(loop, allow_nested))
      return false;

   /* Direct 2D nest cap (<= 120 non-const ops) keeps the simplified interior
    * fast path well below the 256-op splittable_outer limit with headroom for
    * downstream peeling/unrolling; 1D loops use 4..80 body ops (<= 1/4 of the
    * 320-op runtime-unroll budget). For 2D outer loops, also reject any jump
    * outside the loop terminator and inner loop so iteration 0 is guaranteed
    * to reach the outer guard before any hoisted non-speculatable loads. */
   if (allow_nested) {
      if (count_instrs(loop) > 120 ||
          outer_has_jump_outside(&loop->body, cl->nif))
         return false;
   } else {
      int body_ops = count_unrollable(loop, cl->nif);
      if (body_ops < 4 || body_ops > 80)
         return false;
   }

   nir_block *preheader =
      nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_phi_src *init_src = nir_phi_get_src_from_block(cl->iv, preheader);
   if (!init_src || !is_const_i32(nir_get_scalar(init_src->src.ssa, 0), 0))
      return false;

   if (!(allow_limit_inv ? limit_loop_invariant(cl->bound.def, loop, 8)
                         : pure_loop_invariant(cl->bound.def, loop, 8)))
      return false;

   unsigned ord = 0;
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      unsigned cur_ord = ord++;
      if (nif == cl->nif)
         continue;

      bool empty_else = branch_is_empty(&nif->else_list);
      bool empty_then = branch_is_empty(&nif->then_list);
      if (empty_else == empty_then)
         continue;

      bool body_in_then = empty_else;
      struct exec_list *work =
         body_in_then ? &nif->then_list : &nif->else_list;
      if (cf_list_has_outer_jump(work))
         continue;

      /* Skip trivial 1-3 op guarded assignments already handled by
       * peephole_select / flatten_loads without cloning the loop. */
      unsigned tot = 0, inv = 0;
      count_arm_work(work, loop, &tot, &inv);
      if (tot < 4)
         continue;

      struct affine_iv aff;
      nir_def *limit = NULL;
      nir_if *sc_if = NULL;
      bool sc_val = false;
      if (!match_bounds_predicate(nir_get_scalar(nif->condition.ssa, 0),
                                  body_in_then, cl->iv, loop, allow_limit_inv,
                                  &aff, &limit, &sc_if, &sc_val))
         continue;

      bool has_sc = false;
      unsigned sc_ord = 0;
      if (sc_if) {
         unsigned k = 0;
         foreach_list_typed(nir_cf_node, sn, node, &loop->body) {
            if (sn->type != nir_cf_node_if)
               continue;
            if (nir_cf_node_as_if(sn) == sc_if) {
               has_sc = true;
               sc_ord = k;
               break;
            }
            k++;
         }
      }

      *out = (struct interior_guard){
         .body_if = nif,
         .body_if_ord = cur_ord,
         .body_in_then = body_in_then,
         .has_sc_if = has_sc,
         .sc_if_ord = sc_ord,
         .sc_if_val = sc_val,
         .aff = aff,
         .limit = limit,
      };
      return true;
   }
   return false;
}

static struct hash_table *
clone_remap_table(struct hash_table *src)
{
   struct hash_table *dst = _mesa_pointer_hash_table_create(NULL);
   hash_table_foreach(src, entry) {
      _mesa_hash_table_insert(dst, entry->key, entry->data);
   }
   return dst;
}

static unsigned
count_impl_instrs(nir_function_impl *impl)
{
   unsigned n = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_phi &&
             instr->type != nir_instr_type_load_const &&
             instr->type != nir_instr_type_undef)
            n++;
      }
   }
   return n;
}

/* Emit exact zero-overflow interior test for `(bound, aff, limit)`.
 *
 * Fault safety (no soft-fault required):
 *  - `bound_out` is emitted before `bound > 0` (callers verify it is pure or
 *    header-safe invariant, so it runs on loop entry even when trip == 0).
 *  - `base_out` and `stride_out` are emitted strictly inside `if (bound > 0)`
 *    so any in-body `load_global_constant` only runs when iteration 0 is
 *    proved to execute.
 *  - `limit_out` is emitted strictly inside `if (stride >= 0 && base >= 0)`
 *    so any short-circuit `load_global_constant` only runs when coord(0) >= 0.
 *
 * SSA dominance: nested `if` blocks use cloned child remap tables so no clone
 * emitted inside `pos_if` or `chk_if` ever leaks back into the caller's
 * `remap` (preventing non-dominating reuse across sequential 2D checks). */
static nir_def *
emit_interior_check(nir_builder *b, nir_loop *loop, nir_def *bound_def,
                    const struct affine_iv *aff, nir_def *limit_def,
                    struct hash_table *remap, bool use_limit_remat)
{
   nir_def *bound_out = use_limit_remat
                           ? remat_limit(b, bound_def, loop, remap)
                           : remat_outer(b, bound_def, loop, remap);

   nir_if *pos_if = nir_push_if(b, nir_igt_imm(b, bound_out, 0));
   struct hash_table *pos_remap = clone_remap_table(remap);
   nir_def *base_out = emit_affine_base(b, aff, loop, pos_remap, use_limit_remat);
   nir_def *stride_out =
      aff->stride ? (use_limit_remat
                        ? remat_limit(b, aff->stride, loop, pos_remap)
                        : remat_outer(b, aff->stride, loop, pos_remap))
                  : nir_imm_int(b, 1);

   nir_def *can_check =
      nir_iand(b, nir_ige_imm(b, stride_out, 0), nir_ige_imm(b, base_out, 0));

   nir_if *chk_if = nir_push_if(b, can_check);
   struct hash_table *limit_remap = clone_remap_table(pos_remap);
   nir_def *limit_out = remat_limit(b, limit_def, loop, limit_remap);
   _mesa_hash_table_destroy(limit_remap, NULL);
   nir_def *in_0 = nir_ilt(b, base_out, limit_out);
   nir_def *n_minus_1 = nir_iadd_imm(b, bound_out, -1);
   nir_def *hi = nir_umul_high(b, n_minus_1, stride_out);
   nir_def *delta = nir_imul(b, n_minus_1, stride_out);
   nir_def *span = nir_isub(b, limit_out, base_out);
   nir_def *ok_then =
      nir_iand(b, in_0,
               nir_iand(b, nir_ieq_imm(b, hi, 0),
                        nir_ult(b, delta, span)));
   nir_push_else(b, chk_if);
   nir_def *ok_else = nir_imm_false(b);
   nir_pop_if(b, chk_if);
   nir_def *ok_pos = nir_if_phi(b, ok_then, ok_else);
   _mesa_hash_table_destroy(pos_remap, NULL);

   nir_push_else(b, pos_if);
   nir_def *ok_zero = nir_imm_false(b);
   nir_pop_if(b, pos_if);
   return nir_if_phi(b, ok_pos, ok_zero);
}

static void
mark_all_loops_partially_unrolled(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, n, node, list) {
      if (n->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(n);
         lp->partially_unrolled = true;
         mark_all_loops_partially_unrolled(&lp->body);
      } else if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         mark_all_loops_partially_unrolled(&nif->then_list);
         mark_all_loops_partially_unrolled(&nif->else_list);
      }
   }
}

static void
fold_nth_top_if(nir_builder *b, nir_loop *loop, unsigned ord, bool val)
{
   unsigned k = 0;
   foreach_list_typed(nir_cf_node, n, node, &loop->body) {
      if (n->type != nir_cf_node_if)
         continue;
      if (k++ == ord) {
         nir_if *arm_if = nir_cf_node_as_if(n);
         b->cursor = nir_before_cf_node(&arm_if->cf_node);
         nir_src_rewrite(&arm_if->condition, nir_imm_bool(b, val));
         break;
      }
   }
}

static void
collect_loops_preorder(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         util_dynarray_append(loops, loop);
         collect_loops_preorder(&loop->body, loops);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops_preorder(&nif->then_list, loops);
         collect_loops_preorder(&nif->else_list, loops);
      }
   }
}

/*
 * Unswitch per-tap affine interior bounds checks out of 1D and 2D counted
 * stencil loops:
 *
 *    for (int iv = 0; iv < N; iv++) {
 *       int coord = base + iv * stride;
 *       if (coord >= 0 && coord < limit) {
 *          body(iv, coord);
 *       }
 *    }
 *
 * When `N > 0 && stride >= 0 && base >= 0 && base < limit &&
 * umul_high(N - 1, stride) == 0 && (N - 1) * stride < (uint)(limit - base)`,
 * every iteration satisfies `0 <= coord < limit` without signed or unsigned
 * 32-bit overflow. For 2D nested stencil loops where both outer (y) and inner
 * (x) loops carry such guards and the inner guard's parameters are invariant
 * across the outer loop, a single combined `interior_y && interior_x` check
 * produces one completely branchless 2D fast path and keeps the border
 * fallback compact (`partially_unrolled = true`).
 */
bool
agx_nir_opt_unswitch_interior_bounds(nir_shader *shader)
{
   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      struct util_dynarray loops = UTIL_DYNARRAY_INIT;
      collect_loops_preorder(&impl->body, &loops);

      struct set *skip = _mesa_pointer_set_create(NULL);
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      /* Whole-function non-const op guardrail: chosen empirically above the
       * largest vision stencil shaders while preventing compounding unswitch
       * duplication across multi-loop megashaders before downstream unroll. */
      const unsigned max_fn_instrs = 1024;
      unsigned fn_instrs = count_impl_instrs(impl);

      util_dynarray_foreach(&loops, nir_loop *, lp) {
         nir_loop *loop = *lp;
         if (_mesa_set_search(skip, loop))
            continue;
         unsigned loop_ops = count_instrs(loop);
         if (fn_instrs + loop_ops > max_fn_instrs)
            continue;

         const unsigned nested = count_nested_loops(&loop->body);
         if (nested > 1)
            continue;

         if (nested == 1) {
            struct counted_loop ocl;
            struct interior_guard oguard;
            if (!find_interior_guard(loop, &ocl, true, false, &oguard)) {
               if (!find_interior_guard(loop, &ocl, true, true, &oguard))
                  continue;
               if (!pure_loop_invariant(ocl.bound.def, loop, 8))
                  continue;
               bool header_ok = true;
               for (unsigned i = 0; i < oguard.aff.nbase; ++i) {
                  if (!header_safe_invariant(oguard.aff.base_terms[i], loop, 8))
                     header_ok = false;
                  if (oguard.aff.base_mul[i] &&
                      !header_safe_invariant(oguard.aff.base_mul[i], loop, 8))
                     header_ok = false;
               }
               if (oguard.aff.stride &&
                   !header_safe_invariant(oguard.aff.stride, loop, 8))
                  header_ok = false;
               if (!header_ok)
                  continue;
            }

            struct exec_list *owork =
               oguard.body_in_then ? &oguard.body_if->then_list
                                   : &oguard.body_if->else_list;
            nir_loop *inner = NULL;
            nir_if *inner_entry_if = NULL;
            unsigned inner_entry_ord = 0;
            bool inner_entry_in_then = false;
            unsigned oif_ord = 0;
            foreach_list_typed(nir_cf_node, n, node, owork) {
               if (n->type == nir_cf_node_loop) {
                  inner = nir_cf_node_as_loop(n);
                  break;
               }
               if (n->type == nir_cf_node_if) {
                  nir_if *eif = nir_cf_node_as_if(n);
                  unsigned cur_eord = oif_ord++;
                  bool e_else_empty = branch_is_empty(&eif->else_list);
                  bool e_then_empty = branch_is_empty(&eif->then_list);
                  if (e_else_empty != e_then_empty &&
                      pure_loop_invariant(eif->condition.ssa, loop, 8)) {
                     struct exec_list *ework =
                        e_else_empty ? &eif->then_list : &eif->else_list;
                     foreach_list_typed(nir_cf_node, en, node, ework) {
                        if (en->type == nir_cf_node_loop) {
                           inner = nir_cf_node_as_loop(en);
                           inner_entry_if = eif;
                           inner_entry_ord = cur_eord;
                           inner_entry_in_then = e_else_empty;
                           break;
                        }
                     }
                     if (inner)
                        break;
                  }
               }
            }
            if (!inner)
               continue;

            struct counted_loop icl;
            struct interior_guard iguard;
            if (!find_interior_guard(inner, &icl, false, true, &iguard))
               continue;

            /* Inner bound is evaluated before testing inner_bound > 0, so it
             * must be header-safe in `inner` (no loads from inside `inner`'s
             * body that would not execute when top-tested inner trip == 0). */
            if (!limit_loop_invariant(icl.bound.def, loop, 8) ||
                !header_safe_invariant(icl.bound.def, inner, 8) ||
                !affine_invariant_in(&iguard.aff, loop, true) ||
                !limit_loop_invariant(iguard.limit, loop, 8))
               continue;

            b.cursor = nir_before_cf_node(&loop->cf_node);
            struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
            nir_def *obound_out = remat_outer(&b, ocl.bound.def, loop, remap);
            nir_def *pre_ok = nir_igt_imm(&b, obound_out, 0);
            if (inner_entry_if) {
               nir_def *econd =
                  remat_outer(&b, inner_entry_if->condition.ssa, loop, remap);
               if (!inner_entry_in_then)
                  econd = nir_inot(&b, econd);
               pre_ok = nir_iand(&b, pre_ok, econd);
            }
            nir_if *y_pos = nir_push_if(&b, pre_ok);
            nir_def *y_ok = emit_interior_check(&b, loop, ocl.bound.def,
                                                &oguard.aff, oguard.limit,
                                                remap, true);
            nir_if *x_chk = nir_push_if(&b, y_ok);
            nir_def *x_ok = emit_interior_check(&b, loop, icl.bound.def,
                                                &iguard.aff, iguard.limit,
                                                remap, true);
            nir_push_else(&b, x_chk);
            nir_def *x_no = nir_imm_false(&b);
            nir_pop_if(&b, x_chk);
            nir_def *both_then = nir_if_phi(&b, x_ok, x_no);
            nir_push_else(&b, y_pos);
            nir_def *both_else = nir_imm_false(&b);
            nir_pop_if(&b, y_pos);
            nir_def *interior = nir_if_phi(&b, both_then, both_else);
            _mesa_hash_table_destroy(remap, NULL);

            nir_foreach_block_in_cf_node(block, &loop->cf_node)
               nir_lower_phis_to_regs_block(block, true);
            nir_foreach_block_in_cf_node(block, &loop->cf_node)
               nir_lower_ssa_defs_to_regs_block(block);

            b.cursor = nir_before_cf_node(&loop->cf_node);
            nir_if *outer_if = nir_push_if(&b, interior);
            nir_pop_if(&b, outer_if);

            nir_cf_list orig_list, clone_list;
            nir_cf_extract(&orig_list, nir_before_cf_node(&loop->cf_node),
                           nir_after_cf_node(&loop->cf_node));
            struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
            nir_cf_list_clone(&clone_list, &orig_list, &outer_if->cf_node, cmap);
            _mesa_hash_table_destroy(cmap, NULL);

            nir_cf_reinsert(&clone_list, nir_before_cf_list(&outer_if->then_list));
            nir_cf_reinsert(&orig_list, nir_before_cf_list(&outer_if->else_list));

            mark_all_loops_partially_unrolled(&outer_if->else_list);

            nir_loop *fast_outer = NULL;
            foreach_list_typed(nir_cf_node, n, node, &outer_if->then_list) {
               if (n->type == nir_cf_node_loop) {
                  fast_outer = nir_cf_node_as_loop(n);
                  break;
               }
            }
            assert(fast_outer);
            unsigned inner_idx = 0;
            nir_loop *fast_inner = nth_loop_in_list(&fast_outer->body, &inner_idx);
            assert(fast_inner);
            if (inner_entry_if) {
               unsigned ok_ord = 0;
               foreach_list_typed(nir_cf_node, n, node, &fast_outer->body) {
                  if (n->type != nir_cf_node_if)
                     continue;
                  if (ok_ord++ == oguard.body_if_ord) {
                     nir_if *fob = nir_cf_node_as_if(n);
                     struct exec_list *fowork =
                        oguard.body_in_then ? &fob->then_list : &fob->else_list;
                     unsigned ek_ord = 0;
                     foreach_list_typed(nir_cf_node, en, node, fowork) {
                        if (en->type != nir_cf_node_if)
                           continue;
                        if (ek_ord++ == inner_entry_ord) {
                           nir_if *feif = nir_cf_node_as_if(en);
                           b.cursor = nir_before_cf_node(&feif->cf_node);
                           nir_src_rewrite(&feif->condition,
                                           nir_imm_bool(&b, inner_entry_in_then));
                           break;
                        }
                     }
                     break;
                  }
               }
            }
            if (oguard.has_sc_if)
               fold_nth_top_if(&b, fast_outer, oguard.sc_if_ord,
                               oguard.sc_if_val);
            fold_nth_top_if(&b, fast_outer, oguard.body_if_ord,
                            oguard.body_in_then);
            if (iguard.has_sc_if)
               fold_nth_top_if(&b, fast_inner, iguard.sc_if_ord,
                               iguard.sc_if_val);
            fold_nth_top_if(&b, fast_inner, iguard.body_if_ord,
                            iguard.body_in_then);

            _mesa_set_add(skip, inner);
            fn_instrs += loop_ops;
            impl_progress = true;
            nir_progress(true, impl, nir_metadata_none);
            nir_lower_reg_intrinsics_to_ssa_impl(impl);
            nir_metadata_require(impl, nir_metadata_block_index);
            nir_index_ssa_defs(impl);
            continue;
         }

         /* 1D innermost stencil loop */
         struct counted_loop cl;
         struct interior_guard guard;
         if (!find_interior_guard(loop, &cl, false, false, &guard))
            continue;

         b.cursor = nir_before_cf_node(&loop->cf_node);
         struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
         nir_def *interior = emit_interior_check(&b, loop, cl.bound.def,
                                                 &guard.aff, guard.limit,
                                                 remap, false);
         _mesa_hash_table_destroy(remap, NULL);

         nir_foreach_block_in_cf_node(block, &loop->cf_node)
            nir_lower_phis_to_regs_block(block, true);
         nir_foreach_block_in_cf_node(block, &loop->cf_node)
            nir_lower_ssa_defs_to_regs_block(block);

         b.cursor = nir_before_cf_node(&loop->cf_node);
         nir_if *outer_if = nir_push_if(&b, interior);
         nir_pop_if(&b, outer_if);

         nir_cf_list orig_list, clone_list;
         nir_cf_extract(&orig_list, nir_before_cf_node(&loop->cf_node),
                        nir_after_cf_node(&loop->cf_node));
         struct hash_table *cmap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&clone_list, &orig_list, &outer_if->cf_node, cmap);
         _mesa_hash_table_destroy(cmap, NULL);

         nir_cf_reinsert(&clone_list, nir_before_cf_list(&outer_if->then_list));
         nir_cf_reinsert(&orig_list, nir_before_cf_list(&outer_if->else_list));

         mark_all_loops_partially_unrolled(&outer_if->else_list);

         nir_loop *fast_loop = NULL;
         foreach_list_typed(nir_cf_node, n, node, &outer_if->then_list) {
            if (n->type == nir_cf_node_loop) {
               fast_loop = nir_cf_node_as_loop(n);
               break;
            }
         }
         assert(fast_loop);
         if (guard.has_sc_if)
            fold_nth_top_if(&b, fast_loop, guard.sc_if_ord, guard.sc_if_val);
         fold_nth_top_if(&b, fast_loop, guard.body_if_ord, guard.body_in_then);

         fn_instrs += loop_ops;
         impl_progress = true;
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);
      }

      _mesa_set_destroy(skip, NULL);
      util_dynarray_fini(&loops);
      if (impl_progress)
         any = true;
   }

   if (any) {
      NIR_PASS(_, shader, nir_opt_dead_cf);
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_constant_folding);
      NIR_PASS(_, shader, nir_opt_algebraic);
      NIR_PASS(_, shader, nir_opt_dce);
      NIR_PASS(_, shader, nir_opt_cse);
   }

   return any;
}
