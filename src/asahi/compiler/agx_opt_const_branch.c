/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/list.h"
#include "asahi/isa/agx_minifloat.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_opcodes.h"

/*
 * Fold control-flow compares whose conditions are statically known, post-RA.
 *
 * Backend lowering and register allocation can leave an `if_*cmp` reading two
 * constants (or the same register twice) where the front-end had a folded
 * condition. The construct still owns an exec-mask level, so this is not a
 * peephole: for a constant-false `if` the whole then-path is dead, and for a
 * constant-true `if` the whole else-path is dead.
 *
 * When the canonical structurizer shape is present, excise the dead side:
 *
 *    if_*cmp <const-true>                blockB
 *      <then blocks>                       keeps its instructions
 *    else_*cmp -> end_else              <- else_ .. merge: emptied
 *      <else blocks>
 *    pop_exec n=1                        <- merge block: pop removed
 *
 * and symmetrically for constant-false. Removing the if/else/pop triple keeps
 * the exec-mask stack balanced; breaks inside the surviving side that unwind
 * through the removed level get their nest decremented.
 *
 * Because the pass removes instructions but no blocks, it can only run
 * post-RA where stale critical edges are acceptable (same constraint as
 * agx_opt_empty_else).
 */

static bool
is_cf_opcode(enum agx_opcode op)
{
   switch (op) {
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_IF_FCMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_BREAK:
   case AGX_OPCODE_BREAK_IF_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
   case AGX_OPCODE_POP_EXEC:
   case AGX_OPCODE_JMP_EXEC_ANY:
   case AGX_OPCODE_JMP_EXEC_NONE:
   case AGX_OPCODE_JMP_EXEC_NONE_AFTER:
      return true;
   default:
      return false;
   }
}

/*
 * Evaluate the raw compare of an if_*cmp with statically-known operands.
 * Returns the raw predicate result (0 or 1, before invert_cond), or -1 when
 * the condition is not statically decidable.
 */
static int
eval_const_compare(const agx_instr *I)
{
   assert(I->nr_srcs == 2);
   const agx_index x = I->src[0], y = I->src[1];
   const bool is_float = I->op == AGX_OPCODE_IF_FCMP;

   if (x.abs || x.neg || y.abs || y.neg)
      return -1;

   const bool both_imm = x.type == AGX_INDEX_IMMEDIATE &&
                         y.type == AGX_INDEX_IMMEDIATE;
   const bool same_reg = x.type == AGX_INDEX_REGISTER &&
                         y.type == AGX_INDEX_REGISTER &&
                         x.value == y.value && x.size == y.size &&
                         x.memory == y.memory;

   if (is_float) {
      if (both_imm) {
         /* Minifloat immediates have no NaN or infinity, so every condition
          * is exact. */
         const float fx = agx_minifloat_decode(x.value);
         const float fy = agx_minifloat_decode(y.value);

         switch (I->fcond) {
         case AGX_FCOND_EQ:  return fx == fy;
         case AGX_FCOND_LT:  return fx < fy;
         case AGX_FCOND_GT:  return fx > fy;
         case AGX_FCOND_LTN: return fx < fy;
         case AGX_FCOND_GE:  return fx >= fy;
         case AGX_FCOND_LE:  return fx <= fy;
         case AGX_FCOND_GTN: return fx > fy;
         default:            return -1;
         }
      }

      if (same_reg) {
         /* A register may hold NaN, so only the strict orderings are static
          * for a self-compare: x < x and x > x are never true. */
         switch (I->fcond) {
         case AGX_FCOND_LT:
         case AGX_FCOND_GT:
            return 0;
         default:
            return -1;
         }
      }

      return -1;
   }

   if (same_reg) {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         return 1;
      case AGX_ICOND_ULT:
      case AGX_ICOND_UGT:
      case AGX_ICOND_SLT:
      case AGX_ICOND_SGT:
         return 0;
      default:
         return -1;
      }
   }

   if (both_imm) {
      /* The compare width is no longer recorded; signed orderings are only
       * unambiguous below the narrower half-register's sign bit. Same rule as
       * agx_eval_immediate_icond().
       */
      const bool is_signed = I->icond == AGX_ICOND_SEQ ||
                             I->icond == AGX_ICOND_SLT ||
                             I->icond == AGX_ICOND_SGT;
      if (is_signed && (x.value >= 0x8000 || y.value >= 0x8000))
         return -1;

      switch (I->icond) {
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ: return x.value == y.value;
      case AGX_ICOND_ULT:
      case AGX_ICOND_SLT: return x.value < y.value;
      case AGX_ICOND_UGT:
      case AGX_ICOND_SGT: return x.value > y.value;
      default:            return -1;
      }
   }

   return -1;
}

static bool
in_range(const agx_block *b, unsigned lo, unsigned hi)
{
   return b->index >= lo && b->index <= hi;
}

/* agx_next_block() reads past the end of ctx->blocks; keep the terminator. */
static agx_block *
next_block_or_null(agx_context *ctx, agx_block *blk)
{
   return (blk->link.next == &ctx->blocks) ? NULL : agx_next_block(blk);
}

/*
 * Whether any instruction outside the [lo, hi] block-index range has a branch
 * target inside it. Block indices follow program order, so a dead range is a
 * contiguous index interval.
 */
static bool
branch_into_range(agx_context *ctx, unsigned lo, unsigned hi,
                  const agx_instr *ignore)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, J) {
         if (J == ignore)
            continue;

         if (!(agx_opcodes_info[J->op].immediates & AGX_IMMEDIATE_TARGET) ||
             !J->target)
            continue;

         if (in_range(J->target, lo, hi) &&
             !in_range(block, lo, hi))
            return true;
      }
   }

   return false;
}

/*
 * Scan the surviving side of a folded if before mutating. Tracks the number
 * of exec-mask levels opened inside the range. A pop_exec that would unwind
 * through the folded level, or a while_* form this pass does not reason
 * about, rejects the fold.
 */
static bool
check_surviving_range(agx_context *ctx, agx_block *from, unsigned hi)
{
   unsigned depth = 0;

   for (agx_block *blk = from; blk && blk->index < hi;
        blk = next_block_or_null(ctx, blk)) {
      agx_foreach_instr_in_block(blk, J) {
         switch (J->op) {
         case AGX_OPCODE_IF_ICMP:
         case AGX_OPCODE_IF_FCMP:
            depth++;
            break;
         case AGX_OPCODE_WHILE_ICMP:
         case AGX_OPCODE_WHILE_FCMP:
            return false;
         case AGX_OPCODE_POP_EXEC:
            if (J->nest > depth)
               return false;
            depth -= J->nest;
            break;
         case AGX_OPCODE_BEGIN_CF:
            return false;
         default:
            break;
         }
      }
   }

   return true;
}

/* Decrement the unwind depth of breaks that escape through the folded level. */
static void
fixup_surviving_range(agx_context *ctx, agx_block *from, unsigned hi)
{
   unsigned depth = 0;

   for (agx_block *blk = from; blk && blk->index < hi;
        blk = next_block_or_null(ctx, blk)) {
      agx_foreach_instr_in_block(blk, J) {
         switch (J->op) {
         case AGX_OPCODE_IF_ICMP:
         case AGX_OPCODE_IF_FCMP:
            depth++;
            break;
         case AGX_OPCODE_POP_EXEC:
            depth -= MIN2(depth, J->nest);
            break;
         case AGX_OPCODE_BREAK:
         case AGX_OPCODE_BREAK_IF_ICMP:
         case AGX_OPCODE_BREAK_IF_FCMP:
            /* A break unwinding only levels opened inside the surviving range
             * is unaffected; one escaping the folded if loses a level. */
            if (J->nest > depth)
               J->nest--;
            break;
         default:
            break;
         }
      }
   }
}

static void
empty_range(agx_context *ctx, agx_block *from, unsigned hi)
{
   for (agx_block *blk = from; blk && blk->index < hi;
        blk = next_block_or_null(ctx, blk)) {
      agx_foreach_instr_in_block_safe(blk, J)
         agx_remove_instruction(J);
   }
}

static bool
fold_const_if(agx_context *ctx, agx_block *block, agx_instr *I, bool then_taken)
{
   if (I->nest != 1 || I->target == NULL)
      return false;

   /* Before agx_opt_jmp_none the if closes its block. */
   if (I != agx_last_instr(block))
      return false;

   agx_block *then_ = block->successors[0];
   agx_block *else_ = block->successors[1];
   if (then_ != next_block_or_null(ctx, block) || else_ != I->target)
      return false;

   agx_instr *else_i = agx_first_instr(else_);
   if (!else_i || else_i->nest != 1 ||
       (else_i->op != AGX_OPCODE_ELSE_ICMP &&
        else_i->op != AGX_OPCODE_ELSE_FCMP))
      return false;

   agx_block *end_else = else_i->target;
   if (!end_else || end_else->index < else_->index)
      return false;

   agx_block *merge = next_block_or_null(ctx, end_else);
   if (!merge)
      return false;

   agx_instr *pop = agx_first_instr(merge);
   if (!pop || pop->op != AGX_OPCODE_POP_EXEC || pop->nest != 1)
      return false;

   /* The merge is where both sides reconverge. */
   bool merge_is_succ = false;
   agx_foreach_successor(end_else, succ)
      merge_is_succ |= succ == merge;
   if (!merge_is_succ)
      return false;

   /* [dead_lo, dead_hi] is emptied; [live_lo, live_hi) keeps its body with
    * the dead side's else_ boundary instruction handled separately. */
   const unsigned dead_lo = then_taken ? else_->index : then_->index;
   const unsigned dead_hi = then_taken ? merge->index - 1 : else_->index - 1;
   const unsigned live_hi = then_taken ? else_->index : merge->index;
   agx_block *live_from = then_taken ? then_ : else_;

   if (branch_into_range(ctx, dead_lo, dead_hi, I))
      return false;

   if (!check_surviving_range(ctx, live_from, live_hi))
      return false;

   /* Commit: drop the if, the else boundary instruction, the matching pop,
    * and every instruction in the dead side.
    */
   agx_remove_instruction(I);

   if (then_taken) {
      /* else_ .. merge are all dead, including else_i. */
      empty_range(ctx, else_, merge->index);
   } else {
      empty_range(ctx, then_, else_->index);
      agx_remove_instruction(else_i);
   }

   agx_remove_instruction(pop);
   fixup_surviving_range(ctx, live_from, live_hi);
   return true;
}

void
agx_opt_const_branch(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_IF_FCMP)
            continue;

         const int cond = eval_const_compare(I);
         if (cond < 0)
            continue;

         /* invert_cond flips the predicate before it selects the then side. */
         const bool then_taken = (cond != I->invert_cond);
         fold_const_if(ctx, block, I, then_taken);
      }
   }
}
