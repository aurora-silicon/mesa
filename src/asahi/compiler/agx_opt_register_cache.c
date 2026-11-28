/*
 * Copyright 2025 Valve Corporation
 * Copyright 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "asahi/isa/disasm.h"
#include "util/bitset.h"
#include "util/lut.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "shader_enums.h"


/* Remove bit-identical overwrites only after allocation, without extending
 * allocation live ranges. The cache pass below recomputes all use hints. */
static void
agx3_remove_redundant_physical_copies(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) || ctx->is_preamble)
      return;
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       ctx->stage != MESA_SHADER_FRAGMENT)
      return;

   uint32_t serial = 0;
   agx_foreach_block(ctx, block) {
      uint32_t value[AGX_NUM_REGS] = {0};
      agx_foreach_instr_in_block_safe(block, I) {
         const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
         /* Texture and image instructions can clobber undeclared physical
          * scratch GPRs during packing (for example AGX3B cube CEM scratch in
          * src[6]/+4/+6 and fixed r0/r7/r2 on immediate-descriptor cubes), so
          * always reset the physical-copy table across them. */
         if (!info->can_eliminate || !info->can_reorder ||
             info->schedule_class != AGX_SCHEDULE_CLASS_NONE ||
             !I->nr_dests ||
             I->op == AGX_OPCODE_TEXTURE_SAMPLE ||
             I->op == AGX_OPCODE_TEXTURE_LOAD ||
             I->op == AGX_OPCODE_IMAGE_LOAD) {
            memset(value, 0, sizeof(value));
            continue;
         }

         uint32_t copied[4] = {0}, old_value[4] = {0};
         bool copy = false;
         const agx_index dst = I->nr_dests == 1 ? I->dest[0] : agx_null();
         const unsigned count = agx_is_null(dst) ? 0 : agx_index_size_16(dst);
         const bool plain_dst = dst.type == AGX_INDEX_REGISTER && !dst.memory &&
            !dst.abs && !dst.neg && !I->saturate && !I->shift &&
            !I->invert_cond && !I->memory_is_packed &&
            count >= 1 && count <= 4 && dst.value + count <= AGX_NUM_REGS;
         if (plain_dst) {
            for (unsigned c = 0; c < count; ++c)
               old_value[c] = value[dst.value + c];
         }
         if (plain_dst && I->op == AGX_OPCODE_MOV_IMM && count <= 2) {
            for (unsigned c = 0; c < count; ++c)
               copied[c] = UINT32_C(0x80000000) |
                           ((I->imm >> (16 * c)) & 0xffff);
            copy = true;
         } else if (plain_dst && I->op == AGX_OPCODE_BITOP &&
                    I->truth_table == UTIL_LUT2(a) && I->nr_srcs == 2) {
            const agx_index src = I->src[0];
            if (!src.memory && !src.abs && !src.neg && src.size == dst.size &&
                agx_index_size_16(src) == count) {
               if (src.type == AGX_INDEX_REGISTER &&
                   src.value + count <= AGX_NUM_REGS) {
                  for (unsigned c = 0; c < count; ++c) {
                     if (!value[src.value + c])
                        value[src.value + c] = ++serial;
                     copied[c] = value[src.value + c];
                  }
                  copy = true;
               } else if (src.type == AGX_INDEX_UNIFORM &&
                          src.value + count <= 512) {
                  for (unsigned c = 0; c < count; ++c)
                     copied[c] = UINT32_C(0x40000000) | (src.value + c);
                  copy = true;
               }
            }
         }
         if (copy) {
            bool redundant = true;
            for (unsigned c = 0; c < count; ++c)
               redundant &= old_value[c] == copied[c];
            if (redundant) {
               agx_remove_instruction(I);
               continue;
            }
            for (unsigned c = 0; c < count; ++c)
               value[dst.value + c] = copied[c];
         } else {
            agx_foreach_dest(I, d) {
               const agx_index dest = I->dest[d];
               const unsigned n = agx_index_size_16(dest);
               if (dest.type != AGX_INDEX_REGISTER || dest.memory ||
                   dest.value + n > AGX_NUM_REGS) {
                  memset(value, 0, sizeof(value));
                  break;
               }
               for (unsigned c = 0; c < n; ++c)
                  value[dest.value + c] = ++serial;
            }
         }
      }
   }
}

/* agx_opt_break_if moves a conditional break into the block after its if,
 * but the edge to the break target stays on the emptied then block. The
 * target's live-in is live at the break itself. */
static agx_block *
fused_break_target(const agx_instr *I)
{
   return (I->op == AGX_OPCODE_BREAK_IF_ICMP ||
           I->op == AGX_OPCODE_BREAK_IF_FCMP)
             ? I->target
             : NULL;
}

static void
postra_liveness_ins(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_reg_dest(I, d) {
      unsigned reg = I->dest[d].value;
      BITSET_CLEAR_COUNT(live, reg, agx_index_size_16(I->dest[d]));
   }

   agx_foreach_reg_src(I, s) {
      unsigned reg = I->src[s].value;
      BITSET_SET_COUNT(live, reg, agx_index_size_16(I->src[s]));
   }
}

static bool
postra_block_gen_kill(agx_block *block, BITSET_WORD *gen, BITSET_WORD *kill)
{
   bool has_fused_break = false;
   agx_foreach_instr_in_block_rev(block, I) {
      if (fused_break_target(I))
         has_fused_break = true;

      agx_foreach_reg_dest(I, d) {
         unsigned reg = I->dest[d].value;
         unsigned count = agx_index_size_16(I->dest[d]);
         BITSET_CLEAR_COUNT(gen, reg, count);
         BITSET_SET_COUNT(kill, reg, count);
      }

      agx_foreach_reg_src(I, s) {
         unsigned reg = I->src[s].value;
         unsigned count = agx_index_size_16(I->src[s]);
         BITSET_SET_COUNT(gen, reg, count);
         BITSET_CLEAR_COUNT(kill, reg, count);
      }
   }
   return has_fused_break;
}

/*
 * Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */
static void
postra_liveness(agx_context *ctx)
{
   if (ctx->num_blocks == 1) {
      agx_block *blk = agx_start_block(ctx);
      if (!blk->successors[0] && !blk->successors[1]) {
         BITSET_ZERO(blk->reg_live_out);
         BITSET_DECLARE(live, AGX_NUM_REGS);
         BITSET_ZERO(live);
         agx_foreach_instr_in_block_rev(blk, ins) {
            assert(!fused_break_target(ins));
            postra_liveness_ins(live, ins);
         }
         memcpy(blk->reg_live_in, live, sizeof(live));
         return;
      }
   }

   u_worklist worklist;
   agx_worklist_init(ctx, &worklist);

   const unsigned n = ctx->num_blocks;
   const unsigned rw = BITSET_WORDS(AGX_NUM_REGS);
   BITSET_WORD *gen = n ? calloc((size_t)n * rw, sizeof(BITSET_WORD)) : NULL;
   BITSET_WORD *kill = n ? calloc((size_t)n * rw, sizeof(BITSET_WORD)) : NULL;
   bool *blk_fused = n ? calloc(n, sizeof(bool)) : NULL;
   bool *blk_seen = n ? calloc(n, sizeof(bool)) : NULL;
   bool use_summary = gen && kill && blk_fused && blk_seen;

   bool fused_breaks = false;
   agx_foreach_block(ctx, block) {
      BITSET_ZERO(block->reg_live_in);
      BITSET_ZERO(block->reg_live_out);

      if (use_summary && block->index < n && !blk_seen[block->index]) {
         blk_seen[block->index] = true;
         bool fb = postra_block_gen_kill(block, gen + (size_t)block->index * rw,
                                         kill + (size_t)block->index * rw);
         blk_fused[block->index] = fb;
         fused_breaks |= fb;
      } else {
         use_summary = false;
         agx_foreach_instr_in_block(block, I)
            fused_breaks |= fused_break_target(I) != NULL;
      }
   }
   free(blk_seen);

   /* A fused break's block is not a predecessor of its target, so with
    * fused breaks the whole analysis repeats until nothing changes. */
   bool changed;
   do {
      changed = false;
      agx_foreach_block(ctx, block)
         agx_worklist_push_tail(&worklist, block);

      while (!u_worklist_is_empty(&worklist)) {
         /* Pop off in reverse order since liveness is backwards */
         agx_block *blk = agx_worklist_pop_tail(&worklist);

         /* Calculate liveness locally */
         agx_foreach_successor(blk, succ) {
            BITSET_OR(blk->reg_live_out, blk->reg_live_out, succ->reg_live_in);
         }

         BITSET_DECLARE(live, AGX_NUM_REGS);
         if (use_summary && !blk_fused[blk->index]) {
            const BITSET_WORD *gb = gen + (size_t)blk->index * rw;
            const BITSET_WORD *kb = kill + (size_t)blk->index * rw;
            for (unsigned w = 0; w < rw; ++w)
               live[w] = (blk->reg_live_out[w] & ~kb[w]) | gb[w];
         } else {
            memcpy(live, blk->reg_live_out, sizeof(live));

            agx_foreach_instr_in_block_rev(blk, ins) {
               agx_block *target = fused_break_target(ins);
               if (target)
                  BITSET_OR(live, live, target->reg_live_in);

               postra_liveness_ins(live, ins);
            }
         }

         if (BITSET_EQUAL(blk->reg_live_in, live))
            continue;

         /* We made progress, so we need to reprocess the predecessors */
         memcpy(blk->reg_live_in, live, sizeof(live));
         changed = true;
         agx_foreach_predecessor(blk, pred) {
            agx_worklist_push_head(&worklist, *pred);
         }
      }
   } while (fused_breaks && changed);

   free(gen);
   free(kill);
   free(blk_fused);
   u_worklist_fini(&worklist);
}

/*
 * What the whole SIMD-group needs in divergent control flow.
 *
 * Divergent control flow runs every side of a branch, and every remaining
 * loop iteration, in program order with the other threads masked off. A
 * thread masked off in block B waits at a block that program order has not
 * reached yet: the target of an edge that jumps over B, or, inside a loop, a
 * block the loop leaves to and every remaining iteration jumps over. Call these B's pending blocks. Such a thread next
 * reads what the physical liveness into its pending block says, so B must
 * keep those registers (B's inactive set): its writes reach only the active
 * threads and do not end them, and no read in B is their last use.
 * Everything else in B is liveness as usual. Inactive sets and liveness feed
 * each other, so both are solved together; they only grow, so the iteration
 * ends.
 */
struct physical_live {
   unsigned words;              /* BITSET_WORDS(AGX_NUM_REGS) */
   unsigned *pos;               /* program position by block index */
   BITSET_WORD *inactive;       /* per position */
   BITSET_WORD *live_out;       /* per position, inactive set included */
   BITSET_WORD *live_in;        /* per position */
};

static void
physical_live_free(struct physical_live *p)
{
   free(p->pos);
   free(p->inactive);
   free(p->live_out);
   free(p->live_in);
   memset(p, 0, sizeof(*p));
}

/* Bounds the quadratic pending sets; larger shaders keep no divergent hints. */
#define PHYSICAL_LIVE_MAX_BLOCKS 4096

static bool
physical_liveness(agx_context *ctx, struct physical_live *p)
{
   const unsigned n = ctx->num_blocks;
   if (n == 0 || n > PHYSICAL_LIVE_MAX_BLOCKS)
      return false;

   const unsigned rw = BITSET_WORDS(AGX_NUM_REGS);
   const unsigned bw = BITSET_WORDS(n);
   agx_block **order = calloc(n, sizeof(*order));
   unsigned *loop_end = calloc(n, sizeof(unsigned));
   BITSET_WORD *pending = calloc((size_t)n * bw, sizeof(BITSET_WORD));
   BITSET_WORD *live_in = calloc((size_t)n * rw, sizeof(BITSET_WORD));
   BITSET_WORD *gen = calloc((size_t)n * rw, sizeof(BITSET_WORD));
   BITSET_WORD *kill = calloc((size_t)n * rw, sizeof(BITSET_WORD));
   bool *blk_fused = calloc(n, sizeof(bool));
   p->live_in = live_in;
   p->words = rw;
   p->pos = calloc(n, sizeof(unsigned));
   p->inactive = calloc((size_t)n * rw, sizeof(BITSET_WORD));
   p->live_out = calloc((size_t)n * rw, sizeof(BITSET_WORD));
   bool ok = order && loop_end && pending && live_in && gen && kill &&
             blk_fused && p->pos && p->inactive && p->live_out;

   unsigned count = 0;
   if (ok) {
      agx_foreach_block(ctx, block) {
         if (block->index >= n) {
            ok = false;
            break;
         }
         order[count] = block;
         blk_fused[count] =
            postra_block_gen_kill(block, gen + (size_t)count * rw,
                                  kill + (size_t)count * rw);
         p->pos[block->index] = count++;
      }
   }

   if (ok) {
      /* A loop runs again from its header up to its last back edge. */
      for (unsigned a = 0; a < count; ++a)
         loop_end[a] = a;
      for (unsigned a = 0; a < count; ++a) {
         agx_foreach_successor(order[a], succ) {
            const unsigned t = p->pos[succ->index];
            if (t <= a)
               loop_end[t] = MAX2(loop_end[t], a);
         }
      }

      /* An edge A -> T leaves its threads waiting at T until program order
       * reaches T: the blocks between them, or, for a back edge, the rest of
       * the loop before it runs again. */
      for (unsigned a = 0; a < count; ++a) {
         agx_foreach_successor(order[a], succ) {
            const unsigned t = p->pos[succ->index];
            const unsigned last = t > a ? t - 1 : loop_end[t];
            for (unsigned b = a + 1; b <= last && b < count; ++b)
               BITSET_SET(pending + (size_t)b * bw, t);
         }
      }

      /* Threads waiting outside a loop keep waiting while it runs again. */
      bool changed;
      do {
         changed = false;
         for (unsigned h = 0; h < count; ++h) {
            const unsigned l = loop_end[h];
            if (l == h)
               continue;
            const BITSET_WORD *tail = pending + (size_t)l * bw;
            unsigned t;
            BITSET_FOREACH_SET(t, tail, count) {
               if (t >= h && t <= l)
                  continue;
               for (unsigned b = h; b <= l; ++b) {
                  BITSET_WORD *set = pending + (size_t)b * bw;
                  if (!BITSET_TEST(set, t)) {
                     BITSET_SET(set, t);
                     changed = true;
                  }
               }
            }
         }
      } while (changed);

      do {
         changed = false;
         for (unsigned a = count; a-- > 0;) {
            agx_block *block = order[a];
            BITSET_WORD *inactive = p->inactive + (size_t)a * rw;
            BITSET_WORD *out = p->live_out + (size_t)a * rw;
            memset(inactive, 0, rw * sizeof(BITSET_WORD));

            if (block->cf_divergent) {
               unsigned t;
               BITSET_FOREACH_SET(t, pending + (size_t)a * bw, count) {
                  __bitset_or(inactive, inactive, live_in + (size_t)t * rw, rw);
               }
            }

            BITSET_DECLARE(live, AGX_NUM_REGS);
            memcpy(live, inactive, sizeof(live));
            agx_foreach_successor(block, succ) {
               __bitset_or(live, live,
                           live_in + (size_t)p->pos[succ->index] * rw, rw);
            }
            memcpy(out, live, sizeof(live));

            if (!blk_fused[a]) {
               const BITSET_WORD *gb = gen + (size_t)a * rw;
               const BITSET_WORD *kb = kill + (size_t)a * rw;
               for (unsigned w = 0; w < rw; ++w)
                  live[w] = (out[w] & ~kb[w]) | gb[w] | inactive[w];
            } else {
               agx_foreach_instr_in_block_rev(block, I) {
                  agx_block *target = fused_break_target(I);
                  if (target) {
                     __bitset_or(live, live,
                                 live_in + (size_t)p->pos[target->index] * rw, rw);
                  }
                  postra_liveness_ins(live, I);
                  __bitset_or(live, live, inactive, rw);
               }
            }

            BITSET_WORD *in = live_in + (size_t)a * rw;
            if (memcmp(in, live, sizeof(live))) {
               memcpy(in, live, sizeof(live));
               changed = true;
            }
         }
      } while (changed);
   }

   free(order);
   free(loop_end);
   free(pending);
   free(gen);
   free(kill);
   free(blk_fused);
   if (!ok)
      physical_live_free(p);
   return ok;
}

static bool
writes_reg(const agx_instr *I, unsigned reg)
{
   agx_foreach_reg_dest(I, d) {
      unsigned count = agx_index_size_16(I->dest[d]);

      if (reg >= I->dest[d].value && (reg - I->dest[d].value) < count)
         return true;
   }

   return false;
}

/* Last-use hints in divergent control flow, from physical liveness. On AGX3B
 * and AGX3C (Dynamic Caching, no declared register count), a source marked
 * as its last use releases the register for the whole SIMD-group. AGX2's
 * hint is a register-cache discard only, and AGX3D declares its register
 * count per dispatch again and has no hardware verification row yet; both
 * keep the old rule in divergent blocks.
 */
static bool
agx_divergent_last_use(const agx_context *ctx)
{
   return agx_isa_is_agx3(ctx->key->dev.encoding) && !ctx->key->dev.emit_agx3d;
}

static bool
reg_overlaps(unsigned a, unsigned an, unsigned b, unsigned bn)
{
   return a < b + bn && b < a + an;
}

static bool
instr_reads_reg_range(const agx_instr *I, unsigned reg, unsigned count)
{
   agx_foreach_reg_src(I, s) {
      if (I->src[s].memory)
         continue;
      if (reg_overlaps(I->src[s].value, agx_index_size_16(I->src[s]), reg, count))
         return true;
   }
   return false;
}

static bool
instr_writes_reg_range(const agx_instr *I, unsigned reg, unsigned count)
{
   agx_foreach_reg_dest(I, d) {
      if (I->dest[d].memory)
         continue;
      if (reg_overlaps(I->dest[d].value, agx_index_size_16(I->dest[d]), reg, count))
         return true;
   }
   return false;
}

static bool
instr_is_half_reg_copy(const agx_instr *I, unsigned src_half)
{
   if (I->op != AGX_OPCODE_BITOP || I->truth_table != UTIL_LUT2(a) ||
       I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate || I->shift ||
       I->invert_cond || I->agx3_tex_coord_stage)
      return false;

   const agx_index dst = I->dest[0], src = I->src[0], z = I->src[1];
   return dst.type == AGX_INDEX_REGISTER && !dst.memory &&
          dst.size == AGX_SIZE_16 && agx_channels(dst) == 1 && !dst.abs &&
          !dst.neg && dst.value < 256 &&
          src.type == AGX_INDEX_REGISTER && !src.memory &&
          src.size == AGX_SIZE_16 && agx_channels(src) == 1 && !src.abs &&
          !src.neg && src.value == src_half &&
          z.type == AGX_INDEX_IMMEDIATE && z.value == 0;
}

static bool
instr_blocks_shift_fold(const agx_instr *I)
{
   enum agx_schedule_class sc = agx_opcodes_info[I->op].schedule_class;
   return sc != AGX_SCHEDULE_CLASS_NONE && sc != AGX_SCHEDULE_CLASS_LOAD &&
          sc != AGX_SCHEDULE_CLASS_STORE && sc != AGX_SCHEDULE_CLASS_ATOMIC;
}

static bool
reg_is_live_after(const agx_block *block, const agx_instr *after, unsigned half)
{
   list_for_each_entry_from(agx_instr, J, after->link.next, &block->instructions, link) {
      if (instr_reads_reg_range(J, half, 1))
         return true;
      agx_block *target = fused_break_target(J);
      if (target && BITSET_TEST(target->reg_live_in, half))
         return true;
      if (instr_writes_reg_range(J, half, 1))
         return false;
   }
   return BITSET_TEST(block->reg_live_out, half);
}

/* Bound full-shader per-candidate fallback trials on AGX3B so large shaders
 * never go quadratic when the batch trial does not commit. */
#define AGX3B_MAX_PACK_TRIALS 16

struct shift_fold_candidate {
   agx_instr *S;
   agx_instr *c1;
   agx_instr *copies[8];
   unsigned nr_copies;
   agx_index dst;
   agx_index cmp;
   agx_index lo_half;
   agx_index hi_half;
   bool can_place_at_shift;
   bool can_place_at_copy;
};

struct shift_fold_edit {
   agx_instr *S;
   agx_instr *c1;
   agx_instr *isel;
   struct list_head *s_after;
   struct list_head *c1_after;
   bool c1_adjacent;
   agx_instr *copies[8];
   agx_index saved_src0[8];
   unsigned nr_copies;
};

static void
agx_save_cache_hints(const agx_context *ctx, struct util_dynarray *hints)
{
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_dest(I, d) {
         uint8_t h = I->dest[d].cache ? 1 : 0;
         util_dynarray_append(hints, h);
      }
      agx_foreach_src(I, s) {
         uint8_t h = (I->src[s].cache ? 1 : 0) | (I->src[s].discard ? 2 : 0);
         util_dynarray_append(hints, h);
      }
   }
}

static void
agx_restore_cache_hints(agx_context *ctx, const struct util_dynarray *hints)
{
   const uint8_t *p = (const uint8_t *)hints->data;
   const uint8_t *end = p + hints->size;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_dest(I, d) {
         assert(p < end);
         I->dest[d].cache = (*p++ & 1) != 0;
      }
      agx_foreach_src(I, s) {
         assert(p < end);
         uint8_t h = *p++;
         I->src[s].cache = (h & 1) != 0;
         I->src[s].discard = (h & 2) != 0;
      }
   }
   assert(p == end);
}

/* On AGX3B, an ICMPSEL waiting on more than one scoreboard slot expands into
 * one extra carrier instruction per waited slot (agx3_pack_binary.c), and
 * moving a wait can also perturb the 6-slot scoreboard allocator for later
 * selects. Measure the actual packed AGX3B instruction and byte counts so a
 * fold only commits when the real binary shrinks and never grows. */
static bool
agx3b_measure_packed(agx_context *ctx, bool recompute_cache,
                     unsigned *instrs_out, unsigned *bytes_out)
{
   if (recompute_cache)
      agx_compute_register_cache(ctx);

   const unsigned saved_max_reg = ctx->max_reg;
   const unsigned saved_staging_halfs = ctx->agx3_staging_halfs;
   const bool saved_staging_known = ctx->agx3_staging_known;
   const bool saved_fltnorm_start = ctx->agx3_fltnorm_start;
   const bool saved_g18_mixed = ctx->agx3_g18_mixed_protocol;
   const struct agx3_colour_link_info saved_colour_link = ctx->agx3_colour_link;
   const struct util_dynarray saved_uvs_relocs = ctx->agx3_uvs_relocs;
   const unsigned saved_uvs_reloc_count = ctx->agx3_uvs_reloc_count;
   const bool saved_sampled_fragment =
      ctx->out ? ctx->out->g17p_sampled_fragment : false;

   /* Give the speculative pack a private relocation buffer and count so it
    * never appends UVS relocation records to the live context. */
   ctx->agx3_uvs_relocs = UTIL_DYNARRAY_INIT;
   ctx->agx3_uvs_reloc_count = 0;

   struct util_dynarray bin = UTIL_DYNARRAY_INIT;
   const bool ok = agx3_pack_binary(ctx, &bin);

   util_dynarray_fini(&ctx->agx3_uvs_relocs);
   ctx->agx3_uvs_relocs = saved_uvs_relocs;
   ctx->agx3_uvs_reloc_count = saved_uvs_reloc_count;
   ctx->max_reg = saved_max_reg;
   ctx->agx3_staging_halfs = saved_staging_halfs;
   ctx->agx3_staging_known = saved_staging_known;
   ctx->agx3_fltnorm_start = saved_fltnorm_start;
   ctx->agx3_g18_mixed_protocol = saved_g18_mixed;
   ctx->agx3_colour_link = saved_colour_link;
   if (ctx->out)
      ctx->out->g17p_sampled_fragment = saved_sampled_fragment;

   if (!ok) {
      util_dynarray_fini(&bin);
      return false;
   }

   unsigned count = 0;
   size_t off = 0;
   const uint8_t *code = (const uint8_t *)bin.data;
   while (off < bin.size) {
      signed len = agx3b_instr_length(code + off, bin.size - off, NULL);
      if (len <= 0) {
         util_dynarray_fini(&bin);
         return false;
      }
      off += (size_t)len;
      ++count;
   }

   *instrs_out = count;
   *bytes_out = bin.size;
   util_dynarray_fini(&bin);
   return true;
}

static bool
agx_match_shift_fold_candidate(agx_block *block, agx_instr *S,
                               struct shift_fold_candidate *out)
{
   if (!S->shift_amount_zero_or_16 || S->nr_dests != 1 || S->saturate ||
       S->invert_cond || S->agx3_tex_coord_stage)
      return false;

   const agx_index sdst = S->dest[0];
   if (sdst.type != AGX_INDEX_REGISTER || sdst.memory ||
       sdst.size != AGX_SIZE_32 || agx_channels(sdst) != 1 ||
       (sdst.value & 1) || sdst.value + 2 > 256)
      return false;

   agx_index word, amt;
   if (S->op == AGX_OPCODE_BFEIL) {
      if (S->nr_srcs != 3 || S->bfi_mask != 0 ||
          S->src[0].type != AGX_INDEX_IMMEDIATE || S->src[0].value != 0 ||
          S->src[0].abs || S->src[0].neg || S->src[0].memory)
         return false;
      word = S->src[1];
      amt = S->src[2];
   } else if (S->op == AGX_OPCODE_ASR) {
      if (S->nr_srcs != 2)
         return false;
      word = S->src[0];
      amt = S->src[1];
   } else {
      return false;
   }

   if (word.type != AGX_INDEX_REGISTER || word.memory ||
       word.size != AGX_SIZE_32 || agx_channels(word) != 1 || word.abs ||
       word.neg || (word.value & 1) || word.value + 2 > 256)
      return false;

   if (amt.type != AGX_INDEX_REGISTER || amt.memory || agx_channels(amt) != 1 ||
       amt.abs || amt.neg)
      return false;
   if (amt.size == AGX_SIZE_32) {
      if ((amt.value & 1) || amt.value + 2 > 256)
         return false;
   } else if (amt.size == AGX_SIZE_16) {
      if (amt.value >= 256)
         return false;
   } else {
      return false;
   }

   /* Avoid reading the same register across both comparand and select
    * operands, where AGX2/AGX3B discard hints can conflict. */
   const unsigned amt_n = agx_index_size_16(amt);
   if (reg_overlaps(word.value, 2, amt.value, amt_n))
      return false;

   const unsigned lo = sdst.value;
   const unsigned hi = sdst.value + 1;
   unsigned nr_copies = 0;
   bool hi_overwritten = false;
   bool hi_overwritten_by_ck = false;
   bool ok = true;

   list_for_each_entry_from(agx_instr, J, S->link.next, &block->instructions,
                            link) {
      if (instr_blocks_shift_fold(J))
         break;

      if (!hi_overwritten && instr_reads_reg_range(J, hi, 1)) {
         ok = false;
         break;
      }

      if (instr_reads_reg_range(J, lo, 1)) {
         if (!instr_is_half_reg_copy(J, lo) || J->dest[0].value == lo ||
             nr_copies >= ARRAY_SIZE(out->copies)) {
            ok = false;
            break;
         }
         out->copies[nr_copies++] = J;
         if (instr_writes_reg_range(J, hi, 1))
            hi_overwritten = true;
         hi_overwritten_by_ck = hi_overwritten;
         continue;
      }

      if (instr_writes_reg_range(J, hi, 1))
         hi_overwritten = true;
      if (instr_writes_reg_range(J, lo, 1))
         break;
   }

   if (!ok || nr_copies == 0)
      return false;

   agx_instr *c1 = out->copies[0];
   agx_instr *ck = out->copies[nr_copies - 1];
   const agx_index d1 = c1->dest[0];

   if (reg_is_live_after(block, ck, lo))
      return false;
   if (!hi_overwritten_by_ck && reg_is_live_after(block, ck, hi))
      return false;

   if (nr_copies > 1) {
      list_for_each_entry_from(agx_instr, J, c1->link.next,
                               &block->instructions, link) {
         if (instr_writes_reg_range(J, d1.value, 1))
            return false;
         if (J == ck)
            break;
      }
   }

   bool can_place_at_shift = true;
   bool can_place_at_copy = true;
   list_for_each_entry_from(agx_instr, J, S->link.next, &block->instructions,
                            link) {
      if (J == c1)
         break;
      if (instr_reads_reg_range(J, d1.value, 1) ||
          instr_writes_reg_range(J, d1.value, 1))
         can_place_at_shift = false;
      if (instr_writes_reg_range(J, word.value, 2) ||
          instr_writes_reg_range(J, amt.value, amt_n))
         can_place_at_copy = false;
   }

   if (!can_place_at_shift && !can_place_at_copy)
      return false;

   out->S = S;
   out->c1 = c1;
   out->nr_copies = nr_copies;
   out->dst = agx_register(d1.value, AGX_SIZE_16);
   out->cmp = agx_register(amt.value, amt.size);
   out->lo_half = agx_register(word.value, AGX_SIZE_16);
   out->hi_half = agx_register(word.value + 1, AGX_SIZE_16);
   out->can_place_at_shift = can_place_at_shift;
   out->can_place_at_copy = can_place_at_copy;
   return true;
}

static struct shift_fold_edit
agx_apply_shift_fold(agx_context *ctx, const struct shift_fold_candidate *cand,
                     agx_instr *pos)
{
   struct shift_fold_edit edit = {
      .S = cand->S,
      .c1 = cand->c1,
      .s_after = cand->S->link.prev,
      .c1_after = cand->c1->link.prev,
      .c1_adjacent = cand->c1->link.prev == &cand->S->link,
      .nr_copies = cand->nr_copies,
   };
   for (unsigned i = 1; i < cand->nr_copies; ++i) {
      edit.copies[i] = cand->copies[i];
      edit.saved_src0[i] = cand->copies[i]->src[0];
   }

   agx_builder b = agx_init_builder(ctx, agx_before_instr(pos));
   edit.isel = agx_icmpsel_to(&b, cand->dst, cand->cmp, agx_zero(),
                              cand->lo_half, cand->hi_half, AGX_ICOND_UEQ);

   for (unsigned i = 1; i < cand->nr_copies; ++i)
      cand->copies[i]->src[0] = cand->dst;

   agx_remove_instruction(cand->S);
   agx_remove_instruction(cand->c1);
   return edit;
}

static void
agx_undo_shift_fold(const struct shift_fold_edit *edit)
{
   agx_remove_instruction(edit->isel);
   list_add(&edit->S->link, edit->s_after);
   list_add(&edit->c1->link,
            edit->c1_adjacent ? &edit->S->link : edit->c1_after);
   for (unsigned i = 1; i < edit->nr_copies; ++i)
      edit->copies[i]->src[0] = edit->saved_src0[i];
}

static bool
agx3b_packed_improved(unsigned new_instrs, unsigned new_bytes,
                      unsigned base_instrs, unsigned base_bytes)
{
   return new_instrs <= base_instrs && new_bytes <= base_bytes &&
          (new_instrs < base_instrs || new_bytes < base_bytes);
}

static bool
agx3b_try_fold(agx_context *ctx, const struct shift_fold_candidate *cand,
               agx_instr *pos, unsigned *base_instrs, unsigned *base_bytes)
{
   struct shift_fold_edit edit = agx_apply_shift_fold(ctx, cand, pos);

   unsigned new_instrs = 0, new_bytes = 0;
   if (agx3b_measure_packed(ctx, true, &new_instrs, &new_bytes) &&
       agx3b_packed_improved(new_instrs, new_bytes, *base_instrs,
                             *base_bytes)) {
      *base_instrs = new_instrs;
      *base_bytes = new_bytes;
      return true;
   }

   agx_undo_shift_fold(&edit);
   postra_liveness(ctx);
   return false;
}

/*
 * Post-RA peephole: fold a 32-bit right shift by a dynamic amount proven in
 * NIR to be 0 or 16, followed only by 16-bit register copy(s) of its low half
 * while the shift destination dies, into a single 16-bit icmpsel reading the
 * source word's two halves directly into the copy's destination. Because it
 * replaces a shift and a copy with one select once register assignment is
 * known, it reduces IR instruction count; only AGX3B checks packed binary size
 * before accepting a fold, while zero packed growth on the other encodings
 * (AGX2, AGX3C, AGX3D) was observed across the tested shapes.
 */
bool
agx_opt_select_shifted_half(agx_context *ctx)
{
   bool any = false;
   agx_foreach_instr_global(ctx, I) {
      if (I->shift_amount_zero_or_16) {
         any = true;
         break;
      }
   }
   if (!any)
      return false;

   postra_liveness(ctx);
   const bool agx3b =
      ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B;

   if (!agx3b) {
      bool any_folded = false;
      agx_foreach_block(ctx, block) {
         bool progress;
         do {
            progress = false;
            agx_foreach_instr_in_block_safe_rev(block, S) {
               struct shift_fold_candidate cand;
               if (!agx_match_shift_fold_candidate(block, S, &cand))
                  continue;
               agx_apply_shift_fold(
                  ctx, &cand, cand.can_place_at_shift ? cand.S : cand.c1);
               progress = true;
               any_folded = true;
            }
         } while (progress);
      }
      return any_folded;
   }

   /* On AGX3B, try committing all eligible folds in a single verified batch
    * first (two packs total regardless of candidate count). Only if the batch
    * fails or grows do we roll back and fall back to a bounded number of
    * per-candidate trials. */
   struct util_dynarray edits = UTIL_DYNARRAY_INIT;
   struct util_dynarray saved_hints = UTIL_DYNARRAY_INIT;
   bool have_base = false;
   unsigned base_instrs = 0, base_bytes = 0;

   agx_foreach_block(ctx, block) {
      bool progress;
      do {
         progress = false;
         agx_foreach_instr_in_block_safe_rev(block, S) {
            struct shift_fold_candidate cand;
            if (!agx_match_shift_fold_candidate(block, S, &cand))
               continue;
            if (!have_base) {
               if (!agx3b_measure_packed(ctx, false, &base_instrs,
                                         &base_bytes)) {
                  util_dynarray_fini(&edits);
                  util_dynarray_fini(&saved_hints);
                  return false;
               }
               agx_save_cache_hints(ctx, &saved_hints);
               have_base = true;
            }
            struct shift_fold_edit edit = agx_apply_shift_fold(
               ctx, &cand, cand.can_place_at_shift ? cand.S : cand.c1);
            util_dynarray_append(&edits, edit);
            progress = true;
         }
      } while (progress);
   }

   const unsigned n_edits =
      util_dynarray_num_elements(&edits, struct shift_fold_edit);
   if (n_edits == 0) {
      util_dynarray_fini(&edits);
      util_dynarray_fini(&saved_hints);
      return false;
   }

   unsigned batch_instrs = 0, batch_bytes = 0;
   if (agx3b_measure_packed(ctx, true, &batch_instrs, &batch_bytes) &&
       agx3b_packed_improved(batch_instrs, batch_bytes, base_instrs,
                             base_bytes)) {
      util_dynarray_fini(&edits);
      util_dynarray_fini(&saved_hints);
      return true;
   }

   /* Roll back the batch in LIFO order. */
   const struct shift_fold_edit *edit_arr =
      (const struct shift_fold_edit *)edits.data;
   for (unsigned i = n_edits; i > 0; --i)
      agx_undo_shift_fold(&edit_arr[i - 1]);
   util_dynarray_fini(&edits);
   postra_liveness(ctx);

   unsigned trials = 0;
   bool any_folded = false;
   bool need_cache_refresh = false;

   agx_foreach_block(ctx, block) {
      if (trials >= AGX3B_MAX_PACK_TRIALS)
         break;
      bool progress;
      do {
         progress = false;
         agx_foreach_instr_in_block_safe_rev(block, S) {
            if (trials >= AGX3B_MAX_PACK_TRIALS)
               break;
            struct shift_fold_candidate cand;
            if (!agx_match_shift_fold_candidate(block, S, &cand))
               continue;
            bool folded = false;
            if (cand.can_place_at_shift && trials < AGX3B_MAX_PACK_TRIALS) {
               trials++;
               folded = agx3b_try_fold(ctx, &cand, cand.S, &base_instrs,
                                       &base_bytes);
            }
            if (!folded && cand.can_place_at_copy &&
                trials < AGX3B_MAX_PACK_TRIALS) {
               trials++;
               folded = agx3b_try_fold(ctx, &cand, cand.c1, &base_instrs,
                                       &base_bytes);
            }
            if (!folded) {
               need_cache_refresh = true;
               continue;
            }
            progress = true;
            any_folded = true;
            need_cache_refresh = false;
         }
      } while (progress && trials < AGX3B_MAX_PACK_TRIALS);
   }

   if (!any_folded)
      agx_restore_cache_hints(ctx, &saved_hints);
   else if (need_cache_refresh)
      agx_compute_register_cache(ctx);

   util_dynarray_fini(&saved_hints);
   return any_folded;
}

void
agx_compute_register_cache(agx_context *ctx)
{
   /* A last-use hint releases the register for the whole SIMD-group (a
    * wrong one destroys the value), so divergent blocks take their last uses
    * from physical liveness (physical_liveness()), which knows what
    * masked-off threads still need. Last uses inside divergent control flow
    * are then marked as densely as outside it.
    */
   struct physical_live phys = {0};
   bool have_phys = false;
   if (agx_divergent_last_use(ctx)) {
      bool any = false;
      agx_foreach_block(ctx, block)
         any |= block->cf_divergent;

      have_phys = any && physical_liveness(ctx, &phys);
   }

   /* agx_lower_divergent_shuffle routes register lane indices through a
    * scratch register that must read zero in every inactive thread: the
    * shuffle adds the index up over the whole quad. Never release it. */
   const bool agx3 = ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   const unsigned scratch_lo = agx3 ? 2 : 1, scratch_hi = agx3 ? 3 : 1;
   const bool keep_scratch = have_phys && ctx->any_quad_divergent_shuffle;

   /* Analyze the shader globally */
   postra_liveness(ctx);

   agx_foreach_block(ctx, block) {
      /* Blocks outside divergent control flow keep logical liveness. */
      const bool physical = have_phys && block->cf_divergent;
      const BITSET_WORD *inactive =
         physical ? phys.inactive + (size_t)phys.pos[block->index] * phys.words
                  : NULL;

      /* Live-set at each point in the program */
      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, block->reg_live_out, sizeof(live));
      if (physical) {
         memcpy(live,
                phys.live_out + (size_t)phys.pos[block->index] * phys.words,
                sizeof(live));
      }

      /* Set of registers read "soon" by an ALU instruction. These are
       * candidates for the .cache bit. On AGX3B, bound the lookahead window to
       * 16 ALU instructions so long basic blocks do not evict hot short-lived
       * temporaries from the register cache for cold state read dozens or
       * hundreds of instructions later.
       */
      const bool bound_agx3b_cache =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B;
      const uint32_t agx3b_cache_window = 16;
      BITSET_DECLARE(alu_reads, AGX_NUM_REGS) = {0};
      uint32_t next_alu_read[AGX_NUM_REGS] = {0};
      uint32_t alu_pos = 1;

      agx_foreach_instr_in_block_rev(block, I) {
         agx_block *target = fused_break_target(I);
         if (target && physical) {
            __bitset_or(live, live,
                        phys.live_in + (size_t)phys.pos[target->index] * phys.words,
                        phys.words);
         } else if (target) {
            BITSET_OR(live, live, target->reg_live_in);
         }

         agx_foreach_reg_dest(I, d) {
            uint64_t reg = I->dest[d].value;
            unsigned nr = agx_index_size_16(I->dest[d]);

            if (bound_agx3b_cache) {
               uint32_t next = next_alu_read[reg];
               I->dest[d].cache = next != 0 && (alu_pos - next) <= agx3b_cache_window;
               for (unsigned c = 0; c < nr; ++c)
                  next_alu_read[reg + c] = 0;
            } else {
               I->dest[d].cache = BITSET_TEST(alu_reads, I->dest[d].value);
            }
            BITSET_CLEAR_COUNT(alu_reads, reg, nr);
         }

         agx_foreach_reg_src(I, s) {
            uint64_t reg = I->src[s].value;
            if (bound_agx3b_cache) {
               uint32_t next = next_alu_read[reg];
               I->src[s].cache = next != 0 && (alu_pos - next) <= agx3b_cache_window;
            } else {
               I->src[s].cache = BITSET_TEST(alu_reads, reg);
            }
         }

         const bool is_alu = agx_is_alu(I);
         agx_foreach_reg_src(I, s) {
            uint64_t reg = I->src[s].value;
            unsigned nr = agx_index_size_16(I->src[s]);

            /* If every half of the register is dead after this instruction,
             * it's the last use. That includes halves overwritten this cycle
             * (which won't show up in liveness analysis), unless masked-off
             * threads still need them (the inactive set).
             */
            bool lu = true;
            for (unsigned c = 0; c < nr; ++c) {
               const unsigned r = reg + c;
               if (BITSET_TEST(live, r) &&
                   (!writes_reg(I, r) ||
                    (inactive && BITSET_TEST(inactive, r)))) {
                  lu = false;
                  break;
               }
            }

            if (keep_scratch && reg <= scratch_hi && reg + nr > scratch_lo)
               lu = false;

            /* Without physical liveness, divergent blocks get no last-use
             * hints: another side of the branch, run later with these threads
             * masked off, may still read the register. Skipping hints never
             * affects correctness. Subgroup-uniform ifs/loops have no
             * physical divergence.
             */
            I->src[s].discard = lu && (!block->cf_divergent || physical);

            /* Mark any source read by an ALU instruction in the same block as
             * wanting a .cache hint.
             */
            if (is_alu) {
               BITSET_SET_COUNT(alu_reads, reg, nr);
               if (bound_agx3b_cache) {
                  for (unsigned c = 0; c < nr; ++c)
                     next_alu_read[reg + c] = alu_pos;
               }
            }

            assert(!(I->src[s].discard && I->src[s].cache));
         }
         if (is_alu)
            alu_pos++;

         /* M5 (AGX3C): a register read by more than one source of the same
          * instruction is not discarded by any of them. With both sources
          * discarding, allocate_uvb with the primitive and vertex counts in
          * one register (point meshes) read the wrong vertex count; the
          * discard appears to take effect before the later read. Not
          * setting a discard is always safe: it only gives up a register
          * cache saving. */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
            agx_foreach_reg_src(I, s) {
               if (!I->src[s].discard)
                  continue;
               const unsigned a = I->src[s].value;
               const unsigned an = agx_index_size_16(I->src[s]);
               agx_foreach_reg_src(I, t) {
                  if (t == s)
                     continue;
                  const unsigned b = I->src[t].value;
                  const unsigned bn = agx_index_size_16(I->src[t]);
                  if (a < b + bn && b < a + an) {
                     I->src[s].discard = false;
                     break;
                  }
               }
            }
         }

         postra_liveness_ins(live, I);
         if (inactive)
            __bitset_or(live, live, inactive, BITSET_WORDS(AGX_NUM_REGS));
      }
   }

   physical_live_free(&phys);

   /* Apple's G17P fragment loop test keeps
    * its counter operand in the register cache (the f11 hint on the icmp's
    * counter source). The block-local analysis above cannot: the counter's next
    * reads are at the top of the loop, across the back edge. Cache a loop
    * compare's register source when the loop header reads it with an ALU
    * instruction before writing it, unless this read is its last use. */
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_block(ctx, block) {
      agx_block *header = NULL;
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index)
            header = succ;
      }
      if (!header)
         continue;

      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_WHILE_ICMP &&
             I->op != AGX_OPCODE_BREAK_IF_ICMP)
            continue;
         agx_foreach_reg_src(I, s) {
            if (I->src[s].discard || I->src[s].cache ||
                I->src[s].type != AGX_INDEX_REGISTER)
               continue;
            const unsigned reg = I->src[s].value;
            const unsigned nr = agx_index_size_16(I->src[s]);
            bool read = false;
            agx_foreach_instr_in_block(header, H) {
               bool reads = false;
               agx_foreach_reg_src(H, hs) {
                  const unsigned r = H->src[hs].value;
                  const unsigned rn = agx_index_size_16(H->src[hs]);
                  reads |= r < reg + nr && reg < r + rn;
               }
               if (reads) {
                  read = agx_is_alu(H);
                  break;
               }
               bool overwrites = true;
               for (unsigned c = 0; c < nr; ++c)
                  overwrites &= writes_reg(H, reg + c);
               if (overwrites)
                  break;
            }
            I->src[s].cache = read;
         }
      }
   }
}

/*
 * Mark last-use sources to allow the hardware to discard from the register
 * cache. Last use information follows immediately from (post-RA) liveness
 * analysis: a register is dead immediately after its last use.
 *
 * Mark cache hints on sources/destinations to encourage the hardware to make
 * better use of the register cache. This is a simple local analysis.
 */
void
agx_opt_register_cache(agx_context *ctx)
{
   agx3_remove_redundant_physical_copies(ctx);
   agx_compute_register_cache(ctx);
}
