/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/hash_table.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"

/*
 * Loop-invariant code motion that keeps register demand in check.
 *
 * nir_opt_licm hoists every invariant instruction of a loop, and each hoisted
 * result that the loop still reads stays live across the whole loop. Shaders
 * that address hundreds of memory locations from one loop-invariant base, as
 * DXVK compute shaders do for their shared-memory taps, then carry hundreds of
 * hoisted addresses through a loop that already fills the register file, and
 * fill them from spill slots on every iteration, which costs more than the
 * additions hoisting saved. Estimate each loop's peak register demand and the
 * demand hoisting would add, and only hoist if every loop still fits.
 *
 * The estimate counts the 16-bit halves of SSA values live at each point,
 * leaving out constants and preamble values, which immediates and uniforms
 * hold rather than registers.
 */

struct licm_estimate {
   uint8_t *halves;
   BITSET_WORD *invariant;
   void *mem_ctx;

   struct u_sparse_bitset live;
   unsigned live_halves;
};

static bool
def_is_zero_half_uniform(nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);

   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef ||
       (instr->type == nir_instr_type_intrinsic &&
        nir_instr_as_intrinsic(instr)->intrinsic ==
           nir_intrinsic_load_preamble))
      return true;

   /* Packing two preamble/constant halves into a 64-bit uniform pointer stays
    * in uniform registers on AGX rather than occupying GPRs. */
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_pack_64_2x32_split) {
         return def_is_zero_half_uniform(alu->src[0].src.ssa) &&
                def_is_zero_half_uniform(alu->src[1].src.ssa);
      }
   }

   return false;
}

static unsigned
def_halves(nir_def *def)
{
   if (def_is_zero_half_uniform(def))
      return 0;

   return def->num_components * DIV_ROUND_UP(def->bit_size, 16);
}

static bool
record_halves(nir_def *def, void *data)
{
   struct licm_estimate *est = data;
   est->halves[def->index] = MIN2(def_halves(def), UINT8_MAX);
   return true;
}

static void
set_live(struct licm_estimate *est, unsigned index)
{
   if (!u_sparse_bitset_test(&est->live, index)) {
      u_sparse_bitset_set(&est->live, index);
      est->live_halves += est->halves[index];
   }
}

static bool
kill_def(nir_def *def, void *data)
{
   struct licm_estimate *est = data;

   if (u_sparse_bitset_test(&est->live, def->index)) {
      u_sparse_bitset_clear(&est->live, def->index);
      est->live_halves -= est->halves[def->index];
   }

   return true;
}

static bool
gen_src(nir_src *src, void *data)
{
   set_live(data, src->ssa->index);
   return true;
}

/* The most halves live at any point of the loop */
static unsigned
loop_peak(struct licm_estimate *est, nir_loop *loop)
{
   unsigned peak = 0;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      u_sparse_bitset_dup_with_ctx(&est->live, &block->live_out, est->mem_ctx);
      est->live_halves = 0;
      U_SPARSE_BITSET_FOREACH_SET(&est->live, i)
         est->live_halves += est->halves[i];

      nir_if *following_if = nir_block_get_following_if(block);
      if (following_if)
         gen_src(&following_if->condition, est);

      peak = MAX2(peak, est->live_halves);

      nir_foreach_instr_reverse(instr, block) {
         if (instr->type == nir_instr_type_phi)
            break;

         /* Results are written while the sources are still read */
         nir_foreach_src(instr, gen_src, est);
         peak = MAX2(peak, est->live_halves);
         nir_foreach_def(instr, kill_def, est);
      }
   }

   return peak;
}

/* Estimate peak live halves inside `loop` after hoisting every instruction in
 * `est->invariant`: hoisted results that remain used in `loop` stay live across
 * the whole loop (counted once in `hoisted_live_in`, not double-counted inside
 * the loop), and hoisted instructions themselves are no longer executed in the
 * loop. */
static bool
gen_non_hoisted_src(nir_src *src, void *data)
{
   struct licm_estimate *est = data;
   if (!BITSET_TEST(est->invariant, src->ssa->index))
      set_live(est, src->ssa->index);
   return true;
}

static unsigned
loop_peak_after_hoist(struct licm_estimate *est, nir_loop *loop,
                      unsigned hoisted_live_in)
{
   unsigned peak = 0;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      u_sparse_bitset_dup_with_ctx(&est->live, &block->live_out, est->mem_ctx);
      est->live_halves = 0;
      U_SPARSE_BITSET_FOREACH_SET(&est->live, i) {
         if (BITSET_TEST(est->invariant, i))
            u_sparse_bitset_clear(&est->live, i);
         else
            est->live_halves += est->halves[i];
      }

      nir_if *following_if = nir_block_get_following_if(block);
      if (following_if &&
          !BITSET_TEST(est->invariant, following_if->condition.ssa->index))
         gen_non_hoisted_src(&following_if->condition, est);

      peak = MAX2(peak, est->live_halves + hoisted_live_in);

      nir_foreach_instr_reverse(instr, block) {
         if (instr->type == nir_instr_type_phi)
            break;

         nir_def *def = nir_instr_def(instr);
         if (def && BITSET_TEST(est->invariant, def->index))
            continue;

         nir_foreach_src(instr, gen_non_hoisted_src, est);
         peak = MAX2(peak, est->live_halves + hoisted_live_in);
         nir_foreach_def(instr, kill_def, est);
      }
   }

   return peak;
}

struct invariance {
   struct licm_estimate *est;
   nir_loop *loop;
};

static bool
src_invariant(nir_src *src, void *data)
{
   struct invariance *inv = data;
   return nir_def_block(src->ssa)->index <=
             nir_loop_predecessor_block(inv->loop)->index ||
          BITSET_TEST(inv->est->invariant, src->ssa->index);
}

/* What nir_opt_licm without a filter hoists out of the loop */
static bool
instr_invariant(nir_instr *instr, struct invariance *inv)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;

   case nir_instr_type_intrinsic:
      if (!nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
         return false;
      FALLTHROUGH;

   case nir_instr_type_alu:
   case nir_instr_type_tex:
   case nir_instr_type_deref:
      return nir_foreach_src(instr, src_invariant, inv);

   default:
      return false;
   }
}

/* Pure, side-effect-free instructions safe to hoist speculatively out of
 * loops whose body follows a header exit test (e.g. `if (i >= n) break;`).
 */
static bool
instr_speculatable(nir_instr *instr, nir_loop *loop)
{
   /* Only speculatively hoist from top-level loop blocks (e.g. loop body
    * following a header break), not from inside conditional branches.
    */
   if (instr->block->cf_node.parent != &loop->cf_node)
      return false;

   if (instr->type == nir_instr_type_intrinsic)
      return nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_load_preamble;

   if (instr->type != nir_instr_type_alu)
      return false;

   switch (nir_instr_as_alu(instr)->op) {
   case nir_op_idiv:
   case nir_op_udiv:
   case nir_op_imod:
   case nir_op_umod:
   case nir_op_irem:
      return false;
   default:
      return true;
   }
}

static bool
agx_licm_filter(nir_instr *instr, nir_loop *loop, bool block_dominates_exit)
{
   return block_dominates_exit || instr_speculatable(instr, loop);
}

static bool
loop_hoists(nir_loop *loop, bool *has_terminate)
{
   *has_terminate = false;

   if (!nir_loop_has_back_edge(loop))
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (nir_block_ends_in_return_or_halt(block))
         return false;

      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             (nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate ||
              nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate_if))
            *has_terminate = true;
      }
   }

   return true;
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

/* Whether a use keeps a hoisted result live in the loop */
static bool
use_stays_in_loop(struct licm_estimate *est, nir_src *use, nir_loop *loop)
{
   if (!block_in_loop(nir_src_get_block(use), loop))
      return false;

   if (nir_src_is_if(use))
      return true;

   nir_def *user = nir_instr_def(nir_src_use_instr(use));
   return !user || !BITSET_TEST(est->invariant, user->index);
}

/* Halves of hoisted results that instructions staying in the loop read */
static unsigned
loop_hoist_cost(struct licm_estimate *est, unsigned ssa_alloc, nir_loop *loop,
                bool has_terminate, bool speculative)
{
   struct invariance inv = {est, loop};
   nir_block *exit = nir_loop_successor_block(loop);
   unsigned cost = 0;

   memset(est->invariant, 0, BITSET_BYTES(ssa_alloc));

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      bool dom_exit = !has_terminate && nir_block_dominates(block, exit);

      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (def && instr_invariant(instr, &inv) &&
             (dom_exit || (speculative && instr_speculatable(instr, loop))))
            BITSET_SET(est->invariant, def->index);
      }
   }

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (!def || !BITSET_TEST(est->invariant, def->index))
            continue;

         nir_foreach_use_including_if(use, def) {
            if (use_stays_in_loop(est, use, loop)) {
               cost += est->halves[def->index];
               break;
            }
         }
      }
   }

   return cost;
}

static bool
loops_fit(struct licm_estimate *est, unsigned ssa_alloc,
          struct exec_list *list, unsigned budget, bool speculative)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!loops_fit(est, ssa_alloc, &nif->then_list, budget, speculative) ||
             !loops_fit(est, ssa_alloc, &nif->else_list, budget, speculative))
            return false;
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         bool has_terminate = false;
         if (loop_hoists(loop, &has_terminate) && (!has_terminate || speculative)) {
            unsigned cost =
               loop_hoist_cost(est, ssa_alloc, loop, has_terminate, speculative);
            if (cost && loop_peak(est, loop) + cost > budget)
               return false;
         }

         if (!loops_fit(est, ssa_alloc, &loop->body, budget, speculative))
            return false;
         break;
      }

      default:
         break;
      }
   }

   return true;
}

static bool
loop_is_pure_readonly(nir_loop *loop)
{
   bool has_terminate = false;
   if (!loop_hoists(loop, &has_terminate) || has_terminate)
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!(nir_intrinsic_infos[intr->intrinsic].flags &
               NIR_INTRINSIC_CAN_ELIMINATE))
            return false;
         if (strstr(nir_intrinsic_infos[intr->intrinsic].name, "atomic"))
            return false;
         if (nir_intrinsic_has_access(intr) &&
             (nir_intrinsic_access(intr) &
              (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_ATOMIC)))
            return false;
      }
   }
   return true;
}

static bool
is_hoistable_ro_load(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_agx &&
       intr->intrinsic != nir_intrinsic_load_constant_agx &&
       intr->intrinsic != nir_intrinsic_load_global &&
       intr->intrinsic != nir_intrinsic_load_global_constant)
      return false;
   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) &
        (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_ATOMIC)))
      return false;
   return true;
}

static bool
ro_instr_invariant(nir_instr *instr, struct invariance *inv)
{
   if (is_hoistable_ro_load(instr))
      return nir_foreach_src(instr, src_invariant, inv);
   return instr_invariant(instr, inv);
}

static void
collect_loops_postorder(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops_postorder(&nif->then_list, loops);
         collect_loops_postorder(&nif->else_list, loops);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops_postorder(&loop->body, loops);
         util_dynarray_append(loops, loop);
      }
   }
}

static bool
block_only_break(nir_block *block)
{
   nir_instr *first = nir_block_first_instr(block);
   if (!first || first != nir_block_last_instr(block) ||
       first->type != nir_instr_type_jump)
      return false;
   return nir_instr_as_jump(first)->type == nir_jump_break;
}

static bool
cf_list_only_break(struct exec_list *list)
{
   if (exec_list_is_empty(list) ||
       exec_list_get_head(list) != exec_list_get_tail(list))
      return false;
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          block_only_break(nir_cf_node_as_block(node));
}

static bool
cf_list_is_empty_block(struct exec_list *list)
{
   if (exec_list_is_empty(list) ||
       exec_list_get_head(list) != exec_list_get_tail(list))
      return false;
   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(node)->instr_list);
}

static bool
can_eval_at_entry(nir_def *def, nir_loop *loop, nir_block *hdr, nir_block *pre,
                  unsigned depth)
{
   nir_block *db = nir_def_block(def);
   if (!block_in_loop(db, loop))
      return true;
   if (depth == 0 || db != hdr)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_phi)
      return nir_phi_get_src_from_block(nir_instr_as_phi(instr), pre) != NULL;
   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef)
      return true;
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_idiv:
      case nir_op_udiv:
      case nir_op_imod:
      case nir_op_umod:
      case nir_op_irem:
         return false;
      default:
         break;
      }
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (!can_eval_at_entry(alu->src[i].src.ssa, loop, hdr, pre, depth - 1))
            return false;
      }
      return true;
   }
   return false;
}

static nir_def *
eval_at_entry(nir_builder *b, nir_def *def, nir_loop *loop, nir_block *hdr,
              nir_block *pre, struct hash_table *remap)
{
   if (!block_in_loop(nir_def_block(def), loop))
      return def;
   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_phi) {
      nir_phi_src *ps = nir_phi_get_src_from_block(nir_instr_as_phi(instr), pre);
      assert(ps);
      _mesa_hash_table_insert(remap, def, ps->src.ssa);
      return ps->src.ssa;
   }
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         eval_at_entry(b, alu->src[i].src.ssa, loop, hdr, pre, remap);
   }
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *res = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, res);
   return res;
}

static bool
all_live_outs_are_header_phis(nir_loop *loop, nir_block *hdr)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (block == hdr && instr->type == nir_instr_type_phi)
            continue;
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         nir_foreach_use_including_if(use, def) {
            if (!block_in_loop(nir_src_get_block(use), loop))
               return false;
         }
      }
   }
   return true;
}

static bool
block_in_cf_node(nir_block *block, nir_cf_node *ancestor)
{
   for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n == ancestor)
         return true;
   }
   return false;
}

static bool
hoist_pure_readonly_loop(nir_function_impl *impl, nir_loop *loop,
                         unsigned budget)
{
   if (!loop_is_pure_readonly(loop))
      return false;

   bool progress = false;

   /* Phase 1: hoist loop-invariant pure loads (including non-reorderable
    * load_agx/load_global) from blocks that dominate the loop exit. Because
    * the loop contains zero writes, atomics, barriers, calls, or early exits,
    * non-volatile/non-coherent loads are loop-invariant and execute at least
    * once whenever the loop is entered. */
   {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_live_defs);
      void *mem_ctx = ralloc_context(NULL);
      struct licm_estimate est = {
         .halves = rzalloc_array(mem_ctx, uint8_t, impl->ssa_alloc),
         .invariant =
            rzalloc_array(mem_ctx, BITSET_WORD, BITSET_WORDS(impl->ssa_alloc)),
         .mem_ctx = mem_ctx,
      };
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            nir_foreach_def(instr, record_halves, &est);
      }

      struct invariance inv = {&est, loop};
      nir_block *exit = nir_loop_successor_block(loop);
      bool has_ro_load = false;

      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         if (!nir_block_dominates(block, exit))
            continue;
         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (def && ro_instr_invariant(instr, &inv)) {
               BITSET_SET(est.invariant, def->index);
               if (is_hoistable_ro_load(instr))
                  has_ro_load = true;
            }
         }
      }

      if (has_ro_load) {
         unsigned cost = 0;
         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            if (!nir_block_dominates(block, exit))
               continue;
            nir_foreach_instr(instr, block) {
               nir_def *def = nir_instr_def(instr);
               if (!def || !BITSET_TEST(est.invariant, def->index))
                  continue;
               nir_foreach_use_including_if(use, def) {
                  if (use_stays_in_loop(&est, use, loop)) {
                     cost += est.halves[def->index];
                     break;
                  }
               }
            }
         }

         if (cost == 0 || loop_peak_after_hoist(&est, loop, cost) <= budget) {
            nir_cursor dest = nir_before_cf_node(&loop->cf_node);
            nir_foreach_block_in_cf_node(block, &loop->cf_node) {
               if (!nir_block_dominates(block, exit))
                  continue;
               nir_foreach_instr_safe(instr, block) {
                  nir_def *def = nir_instr_def(instr);
                  if (def && BITSET_TEST(est.invariant, def->index))
                     nir_instr_move(dest, instr);
               }
            }
            nir_progress(true, impl, nir_metadata_control_flow);
            progress = true;
         }
      }
      ralloc_free(mem_ctx);
   }

   /* Phase 2: zero-speculation guarded body-load hoisting. When a pure
    * read-only loop has the standard header-break form
    *   loop { hdr: phis; if (exit_cond) break; body: ... }
    * and `body` has loop-invariant pure loads (e.g. a descriptor row in a
    * nested search loop) whose register cost fits in `budget`, evaluate
    * `exit_cond` once at loop entry and hoist those loads into
    * `if (!first_exit) { hoisted; loop { ... } }` so they execute if and only
    * if iteration 0 actually runs `body`. */
   nir_block *hdr = nir_loop_first_block(loop);
   nir_cf_node *n1 = nir_cf_node_next(&hdr->cf_node);
   if (!n1 || n1->type != nir_cf_node_if)
      return progress;
   nir_if *brk_if = nir_cf_node_as_if(n1);
   bool break_in_then = cf_list_only_break(&brk_if->then_list) &&
                        cf_list_is_empty_block(&brk_if->else_list);
   bool break_in_else = !break_in_then &&
                        cf_list_only_break(&brk_if->else_list) &&
                        cf_list_is_empty_block(&brk_if->then_list);
   if (!break_in_then && !break_in_else)
      return progress;
   nir_cf_node *n2 = nir_cf_node_next(&brk_if->cf_node);
   if (!n2 || n2->type != nir_cf_node_block)
      return progress;
   nir_block *body = nir_cf_node_as_block(n2);
   nir_cf_node *prev_node = nir_cf_node_prev(&loop->cf_node);
   if (!prev_node || prev_node->type != nir_cf_node_block)
      return progress;
   nir_block *pre = nir_cf_node_as_block(prev_node);
   if (nir_block_ends_in_jump(pre) || nir_block_ends_in_jump(hdr))
      return progress;

   /* nir_cf_extract moves any successor-block phis into guard->then_list,
    * which would break dominance for exit phis (e.g. early-break searches). */
   nir_block *after_loop =
      nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node));
   if (nir_block_first_instr(after_loop) &&
       nir_block_first_instr(after_loop)->type == nir_instr_type_phi)
      return progress;

   if (!can_eval_at_entry(brk_if->condition.ssa, loop, hdr, pre, 12) ||
       !all_live_outs_are_header_phis(loop, hdr))
      return progress;

   nir_metadata_require(impl, nir_metadata_block_index |
                                 nir_metadata_dominance |
                                 nir_metadata_live_defs);
   void *mem_ctx = ralloc_context(NULL);
   struct licm_estimate est = {
      .halves = rzalloc_array(mem_ctx, uint8_t, impl->ssa_alloc),
      .invariant =
         rzalloc_array(mem_ctx, BITSET_WORD, BITSET_WORDS(impl->ssa_alloc)),
      .mem_ctx = mem_ctx,
   };
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         nir_foreach_def(instr, record_halves, &est);
   }

   struct invariance inv = {&est, loop};
   bool has_body_load = false;
   nir_foreach_instr(instr, body) {
      nir_def *def = nir_instr_def(instr);
      if (def && ro_instr_invariant(instr, &inv)) {
         BITSET_SET(est.invariant, def->index);
         if (is_hoistable_ro_load(instr))
            has_body_load = true;
      }
   }

   if (!has_body_load) {
      ralloc_free(mem_ctx);
      return progress;
   }

   unsigned cost = 0;
   nir_foreach_instr(instr, body) {
      nir_def *def = nir_instr_def(instr);
      if (!def || !BITSET_TEST(est.invariant, def->index))
         continue;
      nir_foreach_use_including_if(use, def) {
         if (use_stays_in_loop(&est, use, loop)) {
            cost += est.halves[def->index];
            break;
         }
      }
   }

   if (cost > 0 && loop_peak_after_hoist(&est, loop, cost) > budget) {
      ralloc_free(mem_ctx);
      return progress;
   }

   /* Collect live-out header phis and their preheader initial values before
    * restructuring CF. */
   struct live_phi_rec {
      nir_phi_instr *phi;
      nir_def *init;
   };
   struct util_dynarray live_phis;
   util_dynarray_init(&live_phis, mem_ctx);
   nir_foreach_phi(phi, hdr) {
      bool used_out = false;
      nir_foreach_use_including_if(use, &phi->def) {
         if (!block_in_loop(nir_src_get_block(use), loop)) {
            used_out = true;
            break;
         }
      }
      if (!used_out)
         continue;
      nir_phi_src *ps = nir_phi_get_src_from_block(phi, pre);
      if (!ps) {
         ralloc_free(mem_ctx);
         return progress;
      }
      struct live_phi_rec rec = {phi, ps->src.ssa};
      util_dynarray_append(&live_phis, rec);
   }

   /* Collect instructions in `body` to hoist in program order. */
   struct util_dynarray to_hoist;
   util_dynarray_init(&to_hoist, mem_ctx);
   nir_foreach_instr(instr, body) {
      nir_def *def = nir_instr_def(instr);
      if (def && BITSET_TEST(est.invariant, def->index))
         util_dynarray_append(&to_hoist, instr);
   }

   nir_builder b = nir_builder_at(nir_before_cf_node(&loop->cf_node));
   struct hash_table *remap = _mesa_pointer_hash_table_create(mem_ctx);
   nir_def *first_exit =
      eval_at_entry(&b, brk_if->condition.ssa, loop, hdr, pre, remap);
   nir_def *enter_cond = break_in_then ? nir_inot(&b, first_exit) : first_exit;
   nir_if *guard = nir_push_if(&b, enter_cond);
   nir_pop_if(&b, guard);

   nir_cf_list lcf;
   nir_cf_extract(&lcf, nir_before_cf_node(&loop->cf_node),
                  nir_after_cf_node(&loop->cf_node));
   nir_cf_reinsert(&lcf, nir_before_cf_list(&guard->then_list));

   nir_cursor hoist_dest = nir_before_cf_node(&loop->cf_node);
   util_dynarray_foreach(&to_hoist, nir_instr *, ip)
      nir_instr_move(hoist_dest, *ip);

   nir_block *after = nir_cf_node_as_block(nir_cf_node_next(&guard->cf_node));
   nir_block *then_end = nir_if_last_then_block(guard);
   nir_block *else_end = nir_if_last_else_block(guard);

   util_dynarray_foreach(&live_phis, struct live_phi_rec, rec) {
      nir_phi_instr *out_phi = nir_phi_instr_create(impl->function->shader);
      nir_def_init(&out_phi->instr, &out_phi->def,
                   rec->phi->def.num_components, rec->phi->def.bit_size);
      nir_phi_instr_add_src(out_phi, then_end, &rec->phi->def);
      nir_phi_instr_add_src(out_phi, else_end, rec->init);
      nir_instr_insert(nir_before_block(after), &out_phi->instr);

      nir_foreach_use_including_if_safe(use, &rec->phi->def) {
         if (nir_src_use_instr(use) == &out_phi->instr)
            continue;
         if (!block_in_cf_node(nir_src_get_block(use), &guard->cf_node))
            nir_src_rewrite(use, &out_phi->def);
      }
   }

   ralloc_free(mem_ctx);
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

/*
 * Run nir_opt_licm if no loop's estimated register demand would exceed budget
 * halves with the invariants hoisted, then hoist pure read-only loop loads
 * per-loop within budget.
 */
bool
agx_nir_opt_licm(nir_shader *shader, unsigned budget, bool speculative)
{
   bool fits_all = true;
   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_live_defs);

      void *mem_ctx = ralloc_context(NULL);
      struct licm_estimate est = {
         .halves = rzalloc_array(mem_ctx, uint8_t, impl->ssa_alloc),
         .invariant =
            rzalloc_array(mem_ctx, BITSET_WORD, BITSET_WORDS(impl->ssa_alloc)),
         .mem_ctx = mem_ctx,
      };

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            nir_foreach_def(instr, record_halves, &est);
      }

      if (!loops_fit(&est, impl->ssa_alloc, &impl->body, budget, speculative))
         fits_all = false;
      ralloc_free(mem_ctx);
   }

   bool progress = false;
   if (fits_all)
      progress |= nir_opt_licm(shader, speculative ? agx_licm_filter : NULL);

   if (fits_all && speculative) {
      nir_foreach_function_impl(impl, shader) {
         struct util_dynarray loops;
         util_dynarray_init(&loops, NULL);
         collect_loops_postorder(&impl->body, &loops);
         util_dynarray_foreach(&loops, nir_loop *, lp)
            progress |= hoist_pure_readonly_loop(impl, *lp, budget);
         util_dynarray_fini(&loops);
      }
   }

   return progress;
}
