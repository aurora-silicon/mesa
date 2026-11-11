/*
 * Copyright © 2024 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"

static bool
impl_has_intrinsic(nir_function_impl *impl, nir_intrinsic_op needle)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == needle)
            return true;
      }
   }

   return false;
}

/** Masks subgroup ballots against invocations terminated earlier
 *
 * Some hardware keeps demoted fragment lanes running as helpers so derivatives
 * and quad operations continue to see a full quad.  When terminate is lowered
 * to that same mechanism, terminated invocations must nevertheless stop
 * contributing to later subgroup ballots after reconvergence.
 *
 * This pass must run before nir_lower_terminate_to_demote(), while terminate
 * is still distinguishable from demote.  Track only termination, not helper
 * state: with maximal reconvergence, rasterizer helpers and invocations demoted
 * explicitly remain active for subgroup operations.  Quad operations are
 * deliberately left alone so full_quads retains their helper-lane values.
 */
bool
nir_mask_ballots_after_terminate(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      const bool has_terminate =
         impl_has_intrinsic(impl, nir_intrinsic_terminate) ||
         impl_has_intrinsic(impl, nir_intrinsic_terminate_if);
      if (!has_terminate ||
          !impl_has_intrinsic(impl, nir_intrinsic_ballot))
         continue;

      nir_variable *terminated = nir_local_variable_create(
         impl, glsl_bool_type(), "terminated_invocation");
      nir_builder b = nir_builder_at(nir_before_impl(impl));
      nir_store_var(&b, terminated, nir_imm_false(&b), 1);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            b.cursor = nir_before_instr(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_terminate:
               nir_store_var(&b, terminated, nir_imm_true(&b), 1);
               break;
            case nir_intrinsic_terminate_if:
               nir_store_var(&b, terminated,
                             nir_ior(&b, nir_load_var(&b, terminated),
                                     intr->src[0].ssa),
                             1);
               break;
            case nir_intrinsic_ballot: {
               nir_def *active = nir_inot(
                  &b, nir_load_var(&b, terminated));
               nir_src_rewrite(&intr->src[0],
                               nir_iand(&b, intr->src[0].ssa, active));
               break;
            }
            default:
               break;
            }
         }
      }

      nir_progress(true, impl, nir_metadata_none);
      progress = true;
   }

   return progress;
}

static bool
nir_lower_terminate_cf_list(nir_builder *b, struct exec_list *cf_list)
{
   bool progress = false;

   foreach_list_typed_safe(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *block = nir_cf_node_as_block(node);

         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_terminate: {
               /* Everything after the terminate is dead */
               nir_cf_list dead_cf;
               nir_cf_extract(&dead_cf, nir_after_instr(&intrin->instr),
                              nir_after_cf_list(cf_list));
               nir_cf_delete(&dead_cf);

               intrin->intrinsic = nir_intrinsic_demote;
               b->cursor = nir_after_instr(&intrin->instr);
               nir_jump(b, nir_jump_halt);

               /* We just removed the remainder of this list of CF nodes.
                * It's not safe to continue iterating.
                */
               return true;
            }

            case nir_intrinsic_terminate_if:
               /* We use demote_if instead of putting a demote in the if, since
                * the backend can likely optimize if { halt } via jump threading
                * or predication while an intervening demote may require real
                * control flow.
                */
               intrin->intrinsic = nir_intrinsic_demote_if;
               b->cursor = nir_after_instr(&intrin->instr);

               nir_push_if(b, intrin->src[0].ssa);
               {
                  nir_jump(b, nir_jump_halt);
               }
               progress = true;
               break;

            default:
               break;
            }
         }
         break;
      }

      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= nir_lower_terminate_cf_list(b, &nif->then_list);
         progress |= nir_lower_terminate_cf_list(b, &nif->else_list);
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= nir_lower_terminate_cf_list(b, &loop->body);
         progress |= nir_lower_terminate_cf_list(b, &loop->continue_list);
         break;
      }

      default:
         UNREACHABLE("Unknown CF node type");
      }
   }

   return progress;
}

static bool
nir_lower_terminate_impl(nir_function_impl *impl)
{
   nir_builder b = nir_builder_create(impl);
   bool progress = nir_lower_terminate_cf_list(&b, &impl->body);

   return nir_progress(progress, impl, nir_metadata_none);
}

/** Lowers nir_intrinsic_terminate to demote + halt
 *
 * The semantics of nir_intrinsic_terminate require that threads immediately
 * exit.  In SPIR-V, terminate is branch instruction even though it's only an
 * intrinsic in NIR.  This pass lowers terminate to demote + halt.  Since halt
 * is a jump instruction in NIR, this restores those semantics and NIR can
 * reason about dead threads after a halt.  It allows lets back-ends to only
 * implement nir_intrinsic_demote as long as they also implement nir_jump_halt.
 */
bool
nir_lower_terminate_to_demote(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      if (nir_lower_terminate_impl(impl))
         progress = true;
   }

   return progress;
}
