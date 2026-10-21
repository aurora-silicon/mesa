/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"

/*
 * Operands that cannot read the uniform file materialize their uniform into
 * a GPR with a `mov` per use site (agx_lower_uniform_sources). Programs fed
 * by a constant program read the same few preamble results from many
 * consumers, so the shader fills with identical `mov uN` copies.
 *
 * This pass CSEs those copies: a copy in a block dominated by an earlier
 * materialization of the same uniform reuses it.  Reuse adds no live range
 * the earlier copy did not already pay for, so it is a reliable win.
 *
 * Two range-extending variants were measured and dropped: materializing a
 * shared copy in a loop preheader, and lifting copies out of sibling
 * branches to their common dominator.  Both make one temporary live across
 * a large part of the function, and the interval-carrying cost through a
 * big kernel -- block-edge copies and index arithmetic, not register-file
 * spills -- exceeds the copies removed: on the merged vkquake RT pipeline
 * (post-RA bin3) hoisting cost +977 and the dominator lift +898, while
 * dominated reuse alone saves -653 (rg3: +1001 / -24 vs -465).  A dropped
 * per-use copy is the cheapest possible rematerialization -- the spiller
 * reloads the same uniform in one instruction -- so the extension buys
 * nothing under pressure either.
 */

static bool
is_uniform_copy(const agx_instr *I)
{
   return I->op == AGX_OPCODE_MOV && I->nr_srcs == 1 && I->nr_dests == 1 &&
          I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
          I->dest[0].type == AGX_INDEX_NORMAL && !I->dest[0].memory &&
          I->dest[0].size == I->src[0].size &&
          agx_channels(I->dest[0]) == 1 && agx_channels(I->src[0]) == 1;
}

static bool
same_uniform_source(agx_index a, agx_index b)
{
   return a.value == b.value && a.size == b.size && a.abs == b.abs &&
          a.neg == b.neg && a.channels_m1 == b.channels_m1;
}

/* Any write to the uniform file invalidates a cached copy of it. The cprog
 * preamble writes through its exports (UNIFORM_STORE) and native uniform
 * arithmetic (UNIFORM_*_AGX3 write the half pair at `imm`).
 */
static bool
writes_uniform_file(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_UNIFORM_STORE:
   case AGX_OPCODE_UNIFORM_FADD_AGX3:
   case AGX_OPCODE_UNIFORM_FMUL_AGX3:
   case AGX_OPCODE_UNIFORM_FFMA_AGX3:
      return true;
   default:
      break;
   }

   agx_foreach_dest(I, d) {
      if (I->dest[d].type == AGX_INDEX_UNIFORM)
         return true;
   }
   return false;
}

/* Halves of new materializations allowed to stay live across a block. Every
 * merged copy extends one temporary's live range across the whole reuse
 * region; the cap keeps that traded pressure bounded and refuses entirely
 * once a block's own liveness approaches the 256-half register file.
 */
#define UNIFORM_COPY_PRESSURE_LIMIT 232

struct pressure {
   unsigned *live;   /* live SSA values at block entry */
   unsigned *added;  /* halves this pass made live through the block */
   BITSET_WORD *dom; /* dominator sets, or NULL when uniforms are mutable */
   unsigned words;
   unsigned nb;
};

/* The shared copy defined in `def` is live across every dominated block up to
 * the last reuse. True if the block fits under the limit for all of them;
 * charges the halves when it does.
 */
static bool
pressure_charge(struct pressure *p, unsigned def, unsigned last_use,
                unsigned halves)
{
   for (unsigned b = 0; b < p->nb; ++b) {
      if (b <= last_use && BITSET_TEST(p->dom + b * p->words, def)) {
         if (2 * p->live[b] + p->added[b] + halves >
             UNIFORM_COPY_PRESSURE_LIMIT)
            return false;
      }
   }
   for (unsigned b = 0; b < p->nb; ++b) {
      if (b <= last_use && BITSET_TEST(p->dom + b * p->words, def))
         p->added[b] += halves;
   }
   return true;
}

void
agx_opt_uniform_copy_cse(agx_context *ctx)
{
   /* Size the replacement map for every temporary that could exist. */
   unsigned copies_total = 0;
   bool mutable_uniforms = false;
   agx_foreach_instr_global(ctx, I) {
      copies_total += is_uniform_copy(I);
      mutable_uniforms |= writes_uniform_file(I);
   }
   if (!copies_total)
      return;

   const unsigned replacement_size = ctx->alloc + copies_total;
   agx_index *replacement = calloc(replacement_size, sizeof(*replacement));

   /* With an immutable uniform file a materialization can serve every block
    * its definition dominates; compute dominators to find them. With mutable
    * uniforms (the preamble), falls back to per-block reuse behind the last
    * write. Live values at block entry come from liveness so merged copies
    * stop once a region approaches the register file's capacity.
    */
   const unsigned nb = ctx->num_blocks + 1;
   const unsigned words = BITSET_WORDS(nb);
   BITSET_WORD *dom =
      mutable_uniforms ? NULL : calloc(nb * words, sizeof(BITSET_WORD));

   struct pressure pressure = {
      .live = calloc(nb, sizeof(*pressure.live)),
      .added = calloc(nb, sizeof(*pressure.added)),
      .dom = dom,
      .words = words,
      .nb = nb,
   };
   agx_compute_liveness(ctx);
   agx_foreach_block(ctx, block)
      pressure.live[block->index] =
         u_sparse_bitset_count(&block->live_in);
   /* Liveness tagging is a side effect; leave the flags clean. */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         I->src[s].kill = false;
   }

   struct canon {
      agx_index src;
      unsigned block;
      agx_index dest;
      agx_instr *instr;
   } *canons = NULL;
   unsigned nr_canons = 0, canons_cap = 0;

   if (dom) {
      /* DOM[B] = {B} u (intersection of DOM over B's predecessors), iterated
       * to a fixpoint; back edges need a second pass over the loop.
       */
      BITSET_WORD *next = calloc(words, sizeof(BITSET_WORD));
      agx_foreach_block(ctx, block) {
         BITSET_WORD *d = dom + block->index * words;
         memset(d, agx_num_predecessors(block) ? 0xff : 0x00,
                words * sizeof(BITSET_WORD));
         if (!agx_num_predecessors(block))
            BITSET_SET(d, block->index);
      }
      bool changed = true;
      while (changed) {
         changed = false;
         agx_foreach_block(ctx, block) {
            BITSET_WORD *d = dom + block->index * words;
            memset(next, 0, words * sizeof(BITSET_WORD));
            bool first_pred = true;
            agx_foreach_predecessor(block, pred) {
               BITSET_WORD *p = dom + (*pred)->index * words;
               if (first_pred) {
                  memcpy(next, p, words * sizeof(BITSET_WORD));
                  first_pred = false;
               } else {
                  __bitset_and(next, next, p, words);
               }
            }
            BITSET_SET(next, block->index);
            if (memcmp(next, d, words * sizeof(BITSET_WORD))) {
               memcpy(d, next, words * sizeof(BITSET_WORD));
               changed = true;
            }
         }
      }
      free(next);
   }

   agx_foreach_block(ctx, block) {
      /* With a mutable uniform file there are no dominators to consult;
       * canons are block-local and die at the next uniform write. */
      if (mutable_uniforms)
         nr_canons = 0;
      unsigned block_canons = nr_canons;

      agx_foreach_instr_in_block_safe(block, I) {
         if (mutable_uniforms && writes_uniform_file(I))
            nr_canons = block_canons;

         if (!is_uniform_copy(I))
            continue;

         const agx_index src = I->src[0];
         agx_index canon_dest = agx_null();
         for (unsigned c = 0; c < nr_canons; ++c) {
            if (!same_uniform_source(canons[c].src, src))
               continue;
            if (canons[c].block == block->index) {
               canon_dest = canons[c].dest;
               break;
            }
            /* Cross-block reuse extends the canon's live range across every
             * dominated block up to here; refuse when tight.
             */
            if (dom &&
                BITSET_TEST(dom + block->index * words, canons[c].block) &&
                pressure_charge(&pressure, canons[c].block, block->index,
                                agx_index_size_16(src)))
               canon_dest = canons[c].dest;
         }
         if (!agx_is_null(canon_dest)) {
            replacement[I->dest[0].value] = canon_dest;
            agx_remove_instruction(I);
            continue;
         }

         if (nr_canons == canons_cap) {
            canons_cap = MAX2(16, canons_cap * 2);
            canons = realloc(canons, canons_cap * sizeof(*canons));
         }
         canons[nr_canons++] =
            (struct canon){.src = src,
                           .block = block->index,
                           .dest = I->dest[0],
                           .instr = I};
      }
   }

   free(canons);
   free(dom);
   free(pressure.live);
   free(pressure.added);

   /* Redirect every use of a removed copy to the shared destination, chasing
    * through a deduplicated hoisted copy to its surviving materialization. A
    * use may precede its removed definition in program order -- a loop
    * header phi reading the latch -- so this runs only after every
    * replacement is recorded.
    */
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_NORMAL)
            continue;
         unsigned v = I->src[s].value;
         agx_index repl = I->src[s];
         while (v < replacement_size &&
                replacement[v].type == AGX_INDEX_NORMAL &&
                replacement[v].value != v) {
            repl = replacement[v];
            v = repl.value;
         }
         if (v != I->src[s].value)
            agx_replace_src(I, s, repl);
      }
   }

   free(replacement);
}
