/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019-2020 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/sparse_bitset.h"
#include "agx_compiler.h"

/* Liveness analysis is a backwards-may dataflow analysis pass. Within a block,
 * we compute live_out from live_in. The intrablock pass is linear-time. It
 * returns whether progress was made. */

/* live_in[s] = GEN[s] + (live_out[s] - KILL[s]) */

void
agx_liveness_ins_update(struct u_sparse_bitset *live, agx_instr *I)
{
   agx_foreach_ssa_dest(I, d)
      u_sparse_bitset_clear(live, I->dest[d].value);

   agx_foreach_ssa_src(I, s) {
      /* If the source is not live after this instruction, but becomes live
       * at this instruction, this is the use that kills the source
       */
      I->src[s].kill = !u_sparse_bitset_test(live, I->src[s].value);
      u_sparse_bitset_set(live, I->src[s].value);
   }
}

static inline void
liveness_ins_update_dense(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_ssa_dest(I, d)
      BITSET_CLEAR(live, I->dest[d].value);

   agx_foreach_ssa_src(I, s) {
      /* If the source is not live after this instruction, but becomes live
       * at this instruction, this is the use that kills the source
       */
      const unsigned v = I->src[s].value;
      I->src[s].kill = !BITSET_TEST(live, v);
      BITSET_SET(live, v);
   }
}

/* Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */

void
agx_compute_liveness(agx_context *ctx)
{
   /* Fast path for single-block shaders without self-loops: one dense
    * backwards pass with no worklist or sparse bitset churn.
    */
   if (list_is_singular(&ctx->blocks)) {
      agx_block *blk = list_first_entry(&ctx->blocks, agx_block, link);
      if (util_dynarray_num_elements(&blk->predecessors, agx_block *) == 0) {
         BITSET_DECLARE(stack_live, 4096);
         BITSET_WORD *dense_live =
            ctx->alloc <= 4096 ? stack_live : BITSET_CALLOC(ctx->alloc);

         u_sparse_bitset_free(&blk->live_in);
         u_sparse_bitset_free(&blk->live_out);
         u_sparse_bitset_init(&blk->live_in, ctx->alloc, blk);
         u_sparse_bitset_init(&blk->live_out, ctx->alloc, blk);

         if (dense_live == stack_live)
            memset(stack_live, 0, BITSET_BYTES(ctx->alloc));

         agx_foreach_instr_in_block_rev(blk, I) {
            if (I->op != AGX_OPCODE_PHI)
               liveness_ins_update_dense(dense_live, I);
         }

         u_sparse_bitset_from_dense(&blk->live_in, dense_live, ctx->alloc);

         if (dense_live != stack_live)
            free(dense_live);
         return;
      }
   }

   /* Fast path when the dense block x SSA bitset slab stays bounded (~4 MiB
    * at the 262144-word cap): precompute per-block gen/kill once, solve the
    * fixed-point with word-wide bitset ops (no per-iteration sparse bitset
    * dup/merge/free or repeated instruction walks), then run a single final
    * backwards pass per block to mark src.kill and populate sparse live sets.
    */
   const unsigned words = BITSET_WORDS(ctx->alloc);
   if (ctx->num_blocks > 0 && ctx->alloc > 0 &&
       (uint64_t)ctx->num_blocks * words <= 262144) {
      bool valid_indices = true;
      bool stack_seen[64];
      bool *seen = ctx->num_blocks <= ARRAY_SIZE(stack_seen)
                      ? stack_seen
                      : calloc(ctx->num_blocks, sizeof(bool));
      if (seen == stack_seen)
         memset(stack_seen, 0, ctx->num_blocks * sizeof(bool));
      if (seen) {
         agx_foreach_block(ctx, block) {
            if (block->index >= ctx->num_blocks || seen[block->index]) {
               valid_indices = false;
               break;
            }
            seen[block->index] = true;
         }
         if (seen != stack_seen)
            free(seen);
      } else {
         valid_indices = false;
      }

      if (valid_indices) {
         const size_t total = (size_t)ctx->num_blocks * words;
         const size_t slab_words = 4 * total + words;
         BITSET_WORD stack_slab[4096];
         bool stack_has_phi[64];
         BITSET_WORD *slab = slab_words <= ARRAY_SIZE(stack_slab)
                                ? stack_slab
                                : calloc(slab_words, sizeof(BITSET_WORD));
         bool *has_phi = ctx->num_blocks <= ARRAY_SIZE(stack_has_phi)
                            ? stack_has_phi
                            : calloc(ctx->num_blocks, sizeof(bool));
         if (slab == stack_slab)
            memset(stack_slab, 0, slab_words * sizeof(BITSET_WORD));
         if (has_phi == stack_has_phi)
            memset(stack_has_phi, 0, ctx->num_blocks * sizeof(bool));
         if (slab && has_phi) {
            BITSET_WORD *gen = slab;
            BITSET_WORD *kill = gen + total;
            BITSET_WORD *live_in = kill + total;
            BITSET_WORD *live_out = live_in + total;
            BITSET_WORD *tmp = live_out + total;

            const unsigned wl_cap = ctx->num_blocks;
            agx_block *stack_wl_entries[128];
            BITSET_DECLARE(stack_wl_present, 128);
            agx_block **wl_entries =
               wl_cap <= ARRAY_SIZE(stack_wl_entries)
                  ? stack_wl_entries
                  : malloc(wl_cap * sizeof(agx_block *));
            BITSET_WORD *wl_present =
               wl_cap <= 128 ? stack_wl_present : BITSET_CALLOC(wl_cap);
            if (!wl_entries || !wl_present) {
               if (wl_entries != stack_wl_entries)
                  free(wl_entries);
               if (wl_present != stack_wl_present)
                  free(wl_present);
               if (slab != stack_slab)
                  free(slab);
               if (has_phi != stack_has_phi)
                  free(has_phi);
               goto slow_path;
            }
            if (wl_present == stack_wl_present)
               memset(stack_wl_present, 0, BITSET_BYTES(wl_cap));
            unsigned wl_count = 0, wl_start = 0;

            agx_foreach_block(ctx, block) {
               const unsigned b = block->index;
               BITSET_WORD *gb = gen + (size_t)b * words;
               BITSET_WORD *kb = kill + (size_t)b * words;

               agx_foreach_instr_in_block_rev(block, I) {
                  if (I->op == AGX_OPCODE_PHI) {
                     has_phi[b] = true;
                     continue;
                  }
                  agx_foreach_ssa_dest(I, d) {
                     const unsigned v = I->dest[d].value;
                     BITSET_CLEAR(gb, v);
                     BITSET_SET(kb, v);
                  }
                  agx_foreach_ssa_src(I, s) {
                     const unsigned v = I->src[s].value;
                     BITSET_SET(gb, v);
                     BITSET_CLEAR(kb, v);
                  }
               }

               if (!BITSET_TEST(wl_present, b)) {
                  wl_start = (wl_start + wl_cap - 1) % wl_cap;
                  wl_entries[wl_start] = block;
                  wl_count++;
                  BITSET_SET(wl_present, b);
               }
            }

            while (wl_count > 0) {
               agx_block *blk = wl_entries[wl_start];
               wl_start = (wl_start + 1) % wl_cap;
               wl_count--;
               const unsigned b = blk->index;
               BITSET_CLEAR(wl_present, b);
               const BITSET_WORD *gb = gen + (size_t)b * words;
               const BITSET_WORD *kb = kill + (size_t)b * words;
               BITSET_WORD *in_b = live_in + (size_t)b * words;
               const BITSET_WORD *out_b = live_out + (size_t)b * words;

               for (unsigned w = 0; w < words; ++w)
                  in_b[w] = (out_b[w] & ~kb[w]) | gb[w];

               if (has_phi[b]) {
                  memcpy(tmp, in_b, words * sizeof(BITSET_WORD));
                  agx_foreach_phi_in_block(blk, phi) {
                     assert(phi->dest[0].type == AGX_INDEX_NORMAL);
                     BITSET_CLEAR(tmp, phi->dest[0].value);
                  }
               }

               const BITSET_WORD *prop = has_phi[b] ? tmp : in_b;
               agx_foreach_predecessor(blk, pred) {
                  const unsigned p = (*pred)->index;
                  BITSET_WORD *out_p = live_out + (size_t)p * words;
                  bool changed = false;

                  for (unsigned w = 0; w < words; ++w) {
                     BITSET_WORD old = out_p[w];
                     BITSET_WORD nw = old | prop[w];
                     if (nw != old) {
                        out_p[w] = nw;
                        changed = true;
                     }
                  }

                  if (has_phi[b]) {
                     unsigned pred_idx = agx_predecessor_index(blk, *pred);
                     agx_foreach_phi_in_block(blk, phi) {
                        agx_index operand = phi->src[pred_idx];
                        if (operand.type == AGX_INDEX_NORMAL) {
                           if (!BITSET_TEST(out_p, operand.value)) {
                              BITSET_SET(out_p, operand.value);
                              changed = true;
                           }
                           phi->src[pred_idx].kill = false;
                        }
                     }
                  }

                  if (changed && !BITSET_TEST(wl_present, p)) {
                     wl_entries[(wl_start + wl_count) % wl_cap] = *pred;
                     wl_count++;
                     BITSET_SET(wl_present, p);
                  }
               }
            }

            agx_foreach_block(ctx, block) {
               const unsigned b = block->index;
               const BITSET_WORD *in_b = live_in + (size_t)b * words;
               const BITSET_WORD *out_b = live_out + (size_t)b * words;

               u_sparse_bitset_free(&block->live_in);
               u_sparse_bitset_free(&block->live_out);
               u_sparse_bitset_init(&block->live_in, ctx->alloc, block);
               u_sparse_bitset_init(&block->live_out, ctx->alloc, block);

               memcpy(tmp, out_b, words * sizeof(BITSET_WORD));
               agx_foreach_instr_in_block_rev(block, I) {
                  if (I->op != AGX_OPCODE_PHI)
                     liveness_ins_update_dense(tmp, I);
               }
               assert(memcmp(tmp, in_b, words * sizeof(BITSET_WORD)) == 0);

               u_sparse_bitset_from_dense(&block->live_in, in_b, ctx->alloc);
               u_sparse_bitset_from_dense(&block->live_out, out_b, ctx->alloc);
            }

            if (wl_entries != stack_wl_entries)
               free(wl_entries);
            if (wl_present != stack_wl_present)
               free(wl_present);
            if (slab != stack_slab)
               free(slab);
            if (has_phi != stack_has_phi)
               free(has_phi);
            return;
         }
         if (slab != stack_slab)
            free(slab);
         if (has_phi != stack_has_phi)
            free(has_phi);
      }
   }

slow_path:;

   BITSET_DECLARE(stack_live, 4096);
   BITSET_WORD *dense_live =
      ctx->alloc <= 4096 ? stack_live : BITSET_CALLOC(ctx->alloc);

   u_worklist worklist;
   u_worklist_init(&worklist, ctx->num_blocks, NULL);

   /* Free any previous liveness, and allocate */
   agx_foreach_block(ctx, block) {
      u_sparse_bitset_free(&block->live_in);
      u_sparse_bitset_free(&block->live_out);

      u_sparse_bitset_init(&block->live_in, ctx->alloc, block);
      u_sparse_bitset_init(&block->live_out, ctx->alloc, block);

      agx_worklist_push_head(&worklist, block);
   }

   /* Iterate the work list */
   while (!u_worklist_is_empty(&worklist)) {
      /* Pop in reverse order since liveness is a backwards pass */
      agx_block *blk = agx_worklist_pop_head(&worklist);

      /* Update its liveness information using a dense bitset inside the block */
      memset(dense_live, 0, BITSET_BYTES(ctx->alloc));
      U_SPARSE_BITSET_FOREACH_SET(&blk->live_out, v)
         BITSET_SET(dense_live, v);

      agx_foreach_instr_in_block_rev(blk, I) {
         if (I->op != AGX_OPCODE_PHI)
            liveness_ins_update_dense(dense_live, I);
      }

      u_sparse_bitset_free(&blk->live_in);
      u_sparse_bitset_init(&blk->live_in, ctx->alloc, blk);
      u_sparse_bitset_from_dense(&blk->live_in, dense_live, ctx->alloc);

      bool has_phi = false;
      agx_foreach_phi_in_block(blk, phi) {
         has_phi = true;
         break;
      }

      /* Propagate the live in of the successor (blk) to the live out of
       * predecessors.
       *
       * Phi nodes are logically on the control flow edge and act in parallel.
       * To handle when propagating, we kill writes from phis and make live the
       * corresponding sources.
       */
      agx_foreach_predecessor(blk, pred) {
         if (!has_phi) {
            if (u_sparse_bitset_merge(&(*pred)->live_out, &blk->live_in))
               agx_worklist_push_tail(&worklist, *pred);
            continue;
         }

         struct u_sparse_bitset live;
         u_sparse_bitset_dup(&live, &blk->live_in);

         /* Kill write */
         agx_foreach_phi_in_block(blk, phi) {
            assert(phi->dest[0].type == AGX_INDEX_NORMAL);
            u_sparse_bitset_clear(&live, phi->dest[0].value);
         }

         /* Make live the corresponding source */
         unsigned pred_idx = agx_predecessor_index(blk, *pred);
         agx_foreach_phi_in_block(blk, phi) {
            agx_index operand = phi->src[pred_idx];
            if (operand.type == AGX_INDEX_NORMAL) {
               u_sparse_bitset_set(&live, operand.value);
               phi->src[pred_idx].kill = false;
            }
         }

         if (u_sparse_bitset_merge(&(*pred)->live_out, &live))
            agx_worklist_push_tail(&worklist, *pred);

         u_sparse_bitset_free(&live);
      }
   }

   u_worklist_fini(&worklist);
   if (dense_live != stack_live)
      free(dense_live);
}
