/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"

/* Fragment shaders with side effects require special handling to ensure the
 * side effects execute as intended. By default, they require late depth
 * testing, to ensure the side effects happen even for killed pixels. G17P's
 * native late-memory shaders provide pass-through depth feedback after the
 * side effects. MOVMSK is its explicit early-test path and does not by itself
 * defer hardware depth rejection. Keep the existing all-live sample test on
 * other devices.
 *
 * For side effects with force early testing forced, the sample mask is written
 * at the *beginning* of the shader.
 */

#define ALL_SAMPLES (0xFF)

static void
insert_late_test(nir_builder *b, bool g17p)
{
   if (g17p) {
      nir_store_output(b, nir_load_frag_coord_z(b), nir_imm_int(b, 0),
                       .io_semantics.location = FRAG_RESULT_DEPTH,
                       .src_type = nir_type_float32);
      b->shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
      return;
   }
   nir_def *all_samples = nir_imm_intN_t(b, ALL_SAMPLES, 16);
   nir_sample_mask_agx(b, all_samples, all_samples);
}

/*
 * This pass runs from agx_preprocess_nir, which drivers may call more than once
 * on the same shader: Honeykrisp preprocesses at shader creation
 * (hk_preprocess_nir) and again while lowering (hk_lower_nir). The late test
 * must be inserted exactly once. Two sample_mask instructions on the same path
 * run the depth/stencil test twice per sample, which the sample_mask contract
 * documented in agx_nir_lower_sample_mask.c forbids.
 *
 * The FragDepth lowering this pass replaced was self-idempotent, because it
 * recorded FRAG_RESULT_DEPTH in outputs_written and the writes_zs check below
 * caught the second run. A sample mask leaves no such trace in shader_info, so
 * look for the instruction itself.
 */
static bool
has_sample_mask(nir_shader *s)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            if (nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_sample_mask_agx)
               return true;
         }
      }
   }

   return false;
}

/*
 * The hardware skips a fragment shader with side effects outright when the
 * depth test is NEVER, so the driver does not program NEVER at all: hk forces
 * the depth comparison to ALWAYS whenever a side-effecting shader is drawn
 * under VK_COMPARE_OP_NEVER (hk_cmd_draw.c, draw.force_never_in_shader, in
 * both the common FRAGMENT_FACE path and the native HAL200 depth words) and
 * expects the shader to emulate the test instead.
 *
 * 7fb60c4c813 emulated it by adding NaN to the synthesized FragDepth write.
 * 53aac78d1aa replaced that write with an all-live sample mask and dropped the
 * NaN with it; df3c8c45508 brought a pass-through FragDepth write back for
 * G17P but not the bias. Nothing has emulated the test since, so every
 * fragment passed a test the API asked to fail and the colour was written.
 *
 * Demote at the very end of the shader instead, after every side effect has
 * executed: the stores land, the colour write does not. This does not depend
 * on NaN comparison semantics in the depth unit, and it is a no-op for every
 * other draw, where load_depth_never_agx reads zero.
 */
static void
emulate_depth_never(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_at(nir_after_impl(impl));

   nir_demote_if(&b, nir_ine_imm(&b, nir_load_depth_never_agx(&b), 0));
   s->info.fs.uses_discard = true;

   nir_progress(true, impl, nir_metadata_none);
}

/*
 * load_helper_invocation reads the lane's helper state.  Between two reads,
 * the state can change for a lane that keeps executing only at a demote or a
 * coverage write: on G17P's deferred-kill model, discard, terminate, sample
 * mask and Z/S feedback all leave rejected lanes running as helpers.  The
 * pipeline wraps every fragment memory write in a helper guard and runs
 * agx_preprocess_nir twice (shader creation, then variant lowering), so each
 * guard region seeds its own coverage read and the regions nest.  Reads with
 * no helper-state writer on any path between them always agree: reuse the
 * earlier read so nir_opt_if can collapse the duplicate guards.
 */
static bool
helper_load_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
   return op == nir_intrinsic_load_helper_invocation ||
          op == nir_intrinsic_is_helper_invocation;
}

static bool
helper_barrier_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_rq_terminate:
   case nir_intrinsic_sample_mask_agx:
   case nir_intrinsic_store_zs_agx:
      return true;
   case nir_intrinsic_store_output: {
      if (!nir_intrinsic_has_io_semantics(intr))
         return false;
      const unsigned location = nir_intrinsic_io_semantics(intr).location;
      return location == FRAG_RESULT_DEPTH ||
             location == FRAG_RESULT_STENCIL ||
             location == FRAG_RESULT_SAMPLE_MASK;
   }
   default:
      return false;
   }
}

/* CFG reachability over block successors, optionally skipping a block.
 * `from` reaches `to` when a path exists; a block reaches itself only
 * through a real cycle that avoids `avoid`. */
static bool
cfg_block_reaches_avoiding(nir_function_impl *impl, nir_block *from,
                           nir_block *to, const nir_block *avoid)
{
   if (from == avoid)
      return false;

   const unsigned n = impl->num_blocks;
   bool *seen = rzalloc_array(NULL, bool, n);
   nir_block **stack = ralloc_array(NULL, nir_block *, n);
   unsigned sp = 0;
   bool found = false;

   stack[sp++] = from;
   while (sp && !found) {
      nir_block *block = stack[--sp];
      for (unsigned i = 0; i < 2; ++i) {
         nir_block *succ = block->successors[i];
         if (!succ || succ == avoid || seen[succ->index])
            continue;
         if (succ == to) {
            found = true;
            break;
         }
         seen[succ->index] = true;
         stack[sp++] = succ;
      }
   }

   ralloc_free(stack);
   ralloc_free(seen);
   return found;
}

/* Whether a helper-state writer can come between two reads on any path: it
 * must run after the first read's point and before the second read's point.
 * A path that re-enters the first read's block re-evaluates it, so writers
 * reaching the second read only through the first block do not split the
 * pair; neither do writers reachable from the first read only through the
 * second read's block (those always follow a second-read evaluation). */
static bool
helper_barrier_between(nir_block *l1b, unsigned l1i, nir_block *l2b,
                       unsigned l2i, nir_block *bb, unsigned bi,
                       nir_function_impl *impl)
{
   if (bb == l1b && bi <= l1i)
      return false;
   if (bb == l2b && bi >= l2i)
      return cfg_block_reaches_avoiding(impl, l2b, l2b, l1b);
   if (bb == l1b || bb == l2b)
      return true;

   return cfg_block_reaches_avoiding(impl, l1b, bb, l2b) &&
          cfg_block_reaches_avoiding(impl, bb, l2b, l1b);
}

static bool
helper_load_dominates(nir_intrinsic_instr *a, unsigned ai,
                      nir_intrinsic_instr *b, unsigned bi)
{
   if (a->instr.block == b->instr.block)
      return ai < bi;
   return nir_block_dominates(a->instr.block, b->instr.block);
}

static bool
dedup_helper_loads_impl(nir_function_impl *impl)
{
   unsigned nr_loads = 0, nr_bars = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nr_loads += helper_load_instr(instr);
         nr_bars += helper_barrier_instr(instr);
      }
   }

   if (nr_loads < 2)
      return false;

   nir_intrinsic_instr **loads =
      ralloc_array(NULL, nir_intrinsic_instr *, nr_loads);
   unsigned *load_pos = ralloc_array(NULL, unsigned, nr_loads);
   nir_instr **bars = ralloc_array(NULL, nir_instr *, nr_bars);
   unsigned *bar_pos = ralloc_array(NULL, unsigned, nr_bars);

   unsigned l = 0, b = 0;
   nir_foreach_block(block, impl) {
      unsigned pos = 0;
      nir_foreach_instr(instr, block) {
         if (helper_load_instr(instr)) {
            loads[l] = nir_instr_as_intrinsic(instr);
            load_pos[l++] = pos;
         }
         if (helper_barrier_instr(instr)) {
            bars[b] = instr;
            bar_pos[b++] = pos;
         }
         pos++;
      }
   }
   assert(l == nr_loads && b == nr_bars);

   nir_metadata_require(impl, nir_metadata_dominance |
                                 nir_metadata_block_index);

   bool progress = false;
   for (unsigned i = 1; i < nr_loads; ++i) {
      for (unsigned j = 0; j < i; ++j) {
         if (!loads[j] || loads[j]->intrinsic != loads[i]->intrinsic ||
             !helper_load_dominates(loads[j], load_pos[j],
                                    loads[i], load_pos[i]) ||
             loads[j]->def.bit_size != loads[i]->def.bit_size)
            continue;

         bool barrier = false;
         for (unsigned k = 0; k < nr_bars && !barrier; ++k) {
            barrier = helper_barrier_between(
               loads[j]->instr.block, load_pos[j], loads[i]->instr.block,
               load_pos[i], bars[k]->block, bar_pos[k], impl);
         }
         if (barrier)
            continue;

         nir_def_rewrite_uses(&loads[i]->def, &loads[j]->def);
         nir_instr_remove(&loads[i]->instr);
         loads[i] = NULL;
         progress = true;
         break;
      }
   }

   ralloc_free(loads);
   ralloc_free(load_pos);
   ralloc_free(bars);
   ralloc_free(bar_pos);
   return nir_progress(progress, impl, nir_metadata_none);
}

bool
agx_nir_dedup_helper_loads(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   bool progress = false;
   nir_foreach_function_impl(impl, s)
      progress |= dedup_helper_loads_impl(impl);

   return progress;
}

struct sidefx_state {
   bool done;
   bool g17p;
};

static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   /* Only lower once */
   struct sidefx_state *state = data;
   if (state->done)
      return false;
   state->done = true;

   b->cursor = nir_before_instr(&intr->instr);
   insert_late_test(b, state->g17p);
   return true;
}

bool
agx_nir_lower_frag_sidefx(nir_shader *s, bool g17p)
{
   assert(s->info.stage == MESA_SHADER_FRAGMENT);

   /* If there are no side effects, there's nothing to lower */
   if (!s->info.writes_memory)
      return false;

   /* Lower writes from helper invocations with the common pass. The hardware
    * will predicate simple writes for us, but that fails with sample shading so
    * we do that in software too.
    */
   NIR_PASS(_, s, nir_lower_helper_writes,
            s->info.fs.uses_sample_shading || g17p);

   bool writes_zs =
      s->info.outputs_written &
      (BITFIELD64_BIT(FRAG_RESULT_STENCIL) | BITFIELD64_BIT(FRAG_RESULT_DEPTH));

   /* If the shader wants early fragment tests, the sample mask lowering pass
    * will trigger an early test at the beginning of the shader. This lets us
    * use a Passthrough punch type, instead of Opaque which may result in the
    * shader getting skipped incorrectly and then the side effects not kicking
    * in. But this happens there to avoid it happening twice with a discard.
    */
   if (s->info.fs.early_fragment_tests)
      return false;

   /* If depth/stencil feedback is already used, we're done */
   if (writes_zs)
      return false;

   /* If a previous run of this pass already inserted the late test, we're done.
    * Running it again would test every sample twice.
    */
   if (has_sample_mask(s))
      return false;

   struct sidefx_state state = {.g17p = g17p};
   nir_shader_intrinsics_pass(s, pass, nir_metadata_control_flow, &state);

   /* If there's no render targets written, just put the write at the end */
   if (!state.done) {
      nir_function_impl *impl = nir_shader_get_entrypoint(s);
      nir_builder b = nir_builder_at(nir_after_impl(impl));

      insert_late_test(&b, g17p);
   }

   if (g17p)
      emulate_depth_never(s);

   return true;
}
