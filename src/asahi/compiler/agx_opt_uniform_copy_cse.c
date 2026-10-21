/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "util/u_worklist.h"

/*
 * Operands that cannot read the uniform file materialize their uniform into
 * a GPR with a `mov` per use site (agx_lower_uniform_sources). Programs fed
 * by a constant program read the same few preamble results from many
 * consumers, so the shader -- especially its loops -- fills with identical
 * `mov uN` copies.
 *
 * This pass CSEs those copies. A copy in a block dominated by an earlier
 * materialization of the same uniform reuses it. Within a loop, a uniform
 * copied at least twice is materialized once in the preheader instead of
 * once per use per iteration: one extra register stays live across the loop
 * in exchange for removing an instruction from every iteration. A copy that
 * is the only one in its loop keeps its short live range -- hoisting it
 * would buy nothing the spiller could not already recover by rematerializing
 * (agx_can_rematerialize_uniform).
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

/*
 * The natural loop of `header`: every block that can reach a back edge (a
 * predecessor at or past the header in source order) without passing through
 * the header. Returns false for an unstructured region -- a block the walk
 * absorbs that entry can still reach without the header means the region is
 * not dominated by it, so a preheader copy would not dominate every dynamic
 * use.
 */
static bool
loop_body(agx_context *ctx, agx_block *header, BITSET_WORD *body)
{
   const unsigned words = BITSET_WORDS(ctx->num_blocks + 1);
   BITSET_WORD *reach = calloc(words, sizeof(BITSET_WORD));
   memset(body, 0, words * sizeof(BITSET_WORD));
   bool ok = false;

   u_worklist work;
   agx_worklist_init(ctx, &work);

   /* Blocks reachable from the entry without passing through the header. */
   agx_worklist_push_tail(&work, agx_start_block(ctx));
   while (!u_worklist_is_empty(&work)) {
      agx_block *block = agx_worklist_pop_tail(&work);
      if (block == header || BITSET_TEST(reach, block->index))
         continue;
      BITSET_SET(reach, block->index);
      agx_foreach_successor(block, succ)
         agx_worklist_push_tail(&work, succ);
   }

   /* Backward from each back edge, stopping at the header. */
   BITSET_SET(body, header->index);
   agx_foreach_predecessor(header, pred) {
      if ((*pred)->index >= header->index &&
          !BITSET_TEST(body, (*pred)->index)) {
         BITSET_SET(body, (*pred)->index);
         agx_worklist_push_tail(&work, *pred);
      }
   }
   while (!u_worklist_is_empty(&work)) {
      agx_block *block = agx_worklist_pop_tail(&work);
      agx_foreach_predecessor(block, pred) {
         if (!BITSET_TEST(body, (*pred)->index)) {
            BITSET_SET(body, (*pred)->index);
            agx_worklist_push_tail(&work, *pred);
         }
      }
   }
   u_worklist_fini(&work);

   /* The region is canonical exactly when no absorbed block is reachable
    * without the header and the header has exactly one outside predecessor:
    * the preheader, which then dominates every block in `body`.
    */
   unsigned entries = 0;
   bool dominated = true;
   agx_foreach_block(ctx, block) {
      if (BITSET_TEST(body, block->index)) {
         if (block != header && BITSET_TEST(reach, block->index))
            dominated = false;
      } else {
         agx_foreach_successor(block, succ) {
            if (succ == header)
               ++entries;
         }
      }
   }
   ok = dominated && entries == 1;

   free(reach);
   return ok;
}

static agx_block *
loop_preheader(agx_block *header, const BITSET_WORD *body)
{
   agx_foreach_predecessor(header, pred) {
      if (!BITSET_TEST(body, (*pred)->index))
         return *pred;
   }
   UNREACHABLE("canonical loop has a preheader");
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


static void
hoist_loop_copies(agx_context *ctx, agx_index *replacement,
                  unsigned replacement_size, struct pressure *pressure)
{
   unsigned nr_headers = 0;
   agx_foreach_block(ctx, block)
      nr_headers += block->loop_header;
   if (!nr_headers)
      return;

   /* Innermost loops before their enclosing ones so a copy pulled into an
    * inner preheader can still merge with the outer loop's other copies.
    * Source order puts an inner loop's header past its enclosing one.
    */
   agx_block **headers = calloc(nr_headers, sizeof(*headers));
   unsigned n = 0;
   agx_foreach_block(ctx, block) {
      if (block->loop_header)
         headers[n++] = block;
   }
   for (unsigned i = 0; i + 1 < nr_headers; ++i) {
      for (unsigned j = i + 1; j < nr_headers; ++j) {
         if (headers[j]->index > headers[i]->index) {
            agx_block *swap = headers[i];
            headers[i] = headers[j];
            headers[j] = swap;
         }
      }
   }

   BITSET_WORD *body =
      calloc(BITSET_WORDS(ctx->num_blocks + 1), sizeof(BITSET_WORD));

   for (unsigned h = 0; h < nr_headers; ++h) {
      agx_block *header = headers[h];
      if (!loop_body(ctx, header, body))
         continue;

      agx_block *preheader = loop_preheader(header, body);

      /* A uniform written inside the loop is not loop-invariant, and the
       * writes are unordered relative to the reads within an iteration;
       * refuse the whole loop rather than reason about each index.
       */
      bool mutable_uniforms = false;
      unsigned copies = 0;
      agx_foreach_block(ctx, block) {
         if (!BITSET_TEST(body, block->index))
            continue;
         agx_foreach_instr_in_block(block, I) {
            mutable_uniforms |= writes_uniform_file(I);
            copies += is_uniform_copy(I);
         }
      }
      if (mutable_uniforms || copies < 2)
         continue;

      agx_instr **loop_copies = calloc(copies, sizeof(*loop_copies));
      unsigned *copy_blocks = calloc(copies, sizeof(*copy_blocks));
      unsigned nr_copies = 0;
      agx_foreach_block(ctx, block) {
         if (!BITSET_TEST(body, block->index))
            continue;
         agx_foreach_instr_in_block(block, I) {
            if (is_uniform_copy(I)) {
               copy_blocks[nr_copies] = block->index;
               loop_copies[nr_copies++] = I;
            }
         }
      }

      /* Group copies by uniform source; hoist groups of at least two. */
      for (unsigned i = 0; i < nr_copies; ++i) {
         if (!loop_copies[i])
            continue;
         const agx_index uniform = loop_copies[i]->src[0];
         unsigned count = 1;
         unsigned last_use = copy_blocks[i];
         for (unsigned j = i + 1; j < nr_copies; ++j) {
            if (loop_copies[j] &&
                same_uniform_source(uniform, loop_copies[j]->src[0])) {
               ++count;
               last_use = MAX2(last_use, copy_blocks[j]);
            }
         }
         if (count < 2)
            continue;

         /* The shared copy stays live from the preheader to the last copy's
          * block; refuse when a block it crosses is already tight.
          */
         if (pressure &&
             !pressure_charge(pressure, preheader->index, last_use,
                              agx_index_size_16(uniform)))
            continue;

         agx_index src = uniform;
         src.kill = src.cache = src.discard = false;
         agx_builder b =
            agx_init_builder(ctx, agx_after_block_logical(preheader));
         agx_index shared = agx_vec_temp(ctx, src.size, agx_channels(src));
         agx_mov_to(&b, shared, src);
         assert(shared.value < replacement_size);

         for (unsigned j = i; j < nr_copies; ++j) {
            agx_instr *dup = loop_copies[j];
            if (!dup || !same_uniform_source(src, dup->src[0]))
               continue;
            loop_copies[j] = NULL;
            replacement[dup->dest[0].value] = shared;
            agx_remove_instruction(dup);
         }
      }
      free(copy_blocks);
      free(loop_copies);
   }
   free(body);
   free(headers);
}

void
agx_opt_uniform_copy_cse(agx_context *ctx)
{
   /* The hoisting phase allocates fresh temporaries; size the replacement
    * map for every temporary that could exist.
    */
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

   hoist_loop_copies(ctx, replacement, replacement_size,
                     dom ? &pressure : NULL);

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

   /* A key whose surviving copies sit in blocks that do not dominate each
    * other -- the two sides of an if/else, or per-iteration bodies of an
    * unrolled check sequence -- still materializes the same uniform in each.
    * Pull one copy up to their nearest common dominator; the shared
    * definition then dominates every copy block and transitively every use.
    * The copy may execute on paths that never reach a consumer, a fair trade
    * for a side-effect-free uniform read.
    */
   if (dom) {
      BITSET_WORD *next_dom = calloc(words, sizeof(BITSET_WORD));
      unsigned *depth = calloc(ctx->num_blocks + 1, sizeof(*depth));
      agx_foreach_block(ctx, block) {
         const BITSET_WORD *d = dom + block->index * words;
         for (unsigned w = 0; w < words; ++w)
            depth[block->index] += util_bitcount(d[w]);
      }

      for (unsigned i = 0; i < nr_canons; ++i) {
         if (!canons[i].instr)
            continue;
         const agx_index uniform = canons[i].src;

         /* Common dominators of every copy block for this uniform. */
         bool have = false;
         unsigned count = 0;
         for (unsigned j = i; j < nr_canons; ++j) {
            if (!canons[j].instr ||
                !same_uniform_source(uniform, canons[j].src))
               continue;
            if (!have) {
               memcpy(next_dom, dom + canons[j].block * words,
                      words * sizeof(BITSET_WORD));
               have = true;
            } else {
               __bitset_and(next_dom, next_dom,
                            dom + canons[j].block * words, words);
            }
            ++count;
         }
         if (count < 2)
            continue;

         /* Deepest common dominator: the candidate dominated by all the
          * rest, i.e. the one with the most dominators of its own.
          */
         int best = -1;
         agx_foreach_block(ctx, block) {
            if (!BITSET_TEST(next_dom, block->index))
               continue;
            if (best < 0 || depth[block->index] > depth[best])
               best = block->index;
         }
         if (best < 0)
            continue;

         unsigned last_use = 0;
         for (unsigned j = i; j < nr_canons; ++j) {
            if (canons[j].instr &&
                same_uniform_source(uniform, canons[j].src))
               last_use = MAX2(last_use, canons[j].block);
         }
         if (!pressure_charge(&pressure, best, last_use,
                              agx_index_size_16(uniform)))
            continue;

         /* If the dominator is itself a copy block (layout order disagreed
          * with dominance), reuse its destination; otherwise materialize.
          */
         unsigned shared_from = UINT_MAX;
         agx_index shared = agx_null();
         for (unsigned j = i; j < nr_canons; ++j) {
            if (canons[j].instr &&
                same_uniform_source(uniform, canons[j].src) &&
                canons[j].block == (unsigned)best) {
               shared = canons[j].dest;
               shared_from = j;
            }
         }
         if (agx_is_null(shared)) {
            agx_block *dom_block = NULL;
            agx_foreach_block(ctx, block) {
               if (block->index == (unsigned)best) {
                  dom_block = block;
                  break;
               }
            }
            agx_index src = uniform;
            src.kill = src.cache = src.discard = false;
            agx_builder b =
               agx_init_builder(ctx, agx_after_block_logical(dom_block));
            shared = agx_vec_temp(ctx, src.size, agx_channels(src));
            agx_mov_to(&b, shared, src);
            assert(shared.value < replacement_size);
         }

         for (unsigned j = i; j < nr_canons; ++j) {
            if (!canons[j].instr ||
                !same_uniform_source(uniform, canons[j].src))
               continue;
            if (j != shared_from)
               replacement[canons[j].dest.value] = shared;
            if (j != shared_from)
               agx_remove_instruction(canons[j].instr);
            canons[j].instr = NULL;
         }
      }
      free(next_dom);
      free(depth);
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
