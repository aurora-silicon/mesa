/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/list.h"
#include "agx_builder.h"
#include "agx_compiler.h"

/*
 * Replaces conditional breaks:
 *
 *    if_cmp x, y, n=1 {
 *       break #value
 *    }
 *    pop_exec n=1
 *
 * with break_if_*cmp pseudo-instructions for further optimization. This
 * assumes agx_opt_empty_else has already run.
 *
 * It also folds the exit-value form the register allocator leaves:
 *
 *    if_cmp x, y, n=1 {
 *       dst = mov src   (zero or more register copies)
 *       break #value
 *    }
 *    pop_exec n=1
 *
 * Each copy is an exit-value assignment that only the breaking lanes may
 * perform, which is exactly a select on the if's own comparison:
 * dst = cond ? src : dst is a no-op for lanes that continue. The copies
 * become compare-selects in the merge block ahead of the fused break_if,
 * so the whole exit no longer needs a pushed region, a second compare for
 * the break, or its own nesting level.
 */

/* A then-block move that select-folding may lift out of the if. */
static bool
copy_ok_for_break_select(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_MOV || I->nr_dests != 1 || I->nr_srcs != 1)
      return false;
   const agx_index *dst = &I->dest[0];
   const agx_index *src = &I->src[0];
   if (dst->type != AGX_INDEX_REGISTER || dst->memory)
      return false;
   if (src->type != AGX_INDEX_REGISTER || src->memory || src->abs ||
       src->neg || src->kill)
      return false;
   if (dst->size != src->size ||
       (dst->size != AGX_SIZE_16 && dst->size != AGX_SIZE_32))
      return false;
   return true;
}

static void
match_block(agx_context *ctx, agx_block *block)
{
   agx_instr *if_ = agx_last_instr(block);
   if (!if_ ||
       (if_->op != AGX_OPCODE_IF_ICMP && if_->op != AGX_OPCODE_IF_FCMP) ||
       if_->nest != 1)
      return;

   /* If's fallthrough to the then */
   agx_block *then_block = agx_next_block(block);
   assert(block->successors[0] == then_block && "successors for if");

   /* We're searching for a single block then, so the next block is else */
   agx_block *else_block = agx_next_block(then_block);
   if (block->successors[1] != else_block)
      return;

   if (!list_is_singular(&else_block->instructions))
      return;

   /* The then block is the break plus any exit-value copies ahead of it.
    * Parallel-copy semantics apply to the group: a source read must see
    * the value from before the whole group, which a sequence of selects
    * violates if a later copy reads an earlier copy's destination. Refuse
    * rather than serialise incorrectly. */
   agx_foreach_instr_in_block(then_block, I) {
      if (I == agx_last_instr(then_block))
         break;
      if (!copy_ok_for_break_select(I))
         return;
      agx_foreach_instr_in_block(then_block, J) {
         if (J == I)
            break;
         if (I->src[0].type == J->dest[0].type &&
             I->src[0].value == J->dest[0].value &&
             I->src[0].size == J->dest[0].size)
            return;
      }
      /* The selects land ahead of the break_if's own compare. If a copy
       * touches a register the condition compares, the select's write would
       * change what the compare sees. Refuse rather than reorder. */
      for (unsigned s = 0; s < if_->nr_srcs; ++s) {
         if (if_->src[s].type == I->dest[0].type &&
             if_->src[s].value == I->dest[0].value)
            return;
         if (if_->src[s].type == I->src[0].type &&
             if_->src[s].value == I->src[0].value)
            return;
      }
   }

   agx_instr *break_ = agx_last_instr(then_block);
   agx_instr *pop = agx_last_instr(else_block);

   if (break_->op != AGX_OPCODE_BREAK || pop->op != AGX_OPCODE_POP_EXEC ||
       pop->nest != 1)
      return;

   /* Find the successor of the if */
   agx_block *after = else_block->successors[0];
   assert(else_block->successors[1] == NULL && "single successor");

   /* We are dropping one level of nesting (the if) when rewriting */
   assert(break_->nest >= 1 && "must at least break out of the if");
   unsigned new_nest = break_->nest - 1;

   /* Rewrite */
   agx_builder b = agx_init_builder(ctx, agx_before_block(after));

   /* Each exit-value copy becomes a compare-select on the if's own
    * condition: a lane that takes the break stores the incoming value, a
    * lane that continues keeps its destination. When the if is inverted
    * the body runs on the false side, so the alternatives swap. */
   agx_foreach_instr_in_block_safe(then_block, I) {
      if (I == break_)
         break;
      const agx_index t = if_->invert_cond ? I->dest[0] : I->src[0];
      const agx_index f = if_->invert_cond ? I->src[0] : I->dest[0];
      if (if_->op == AGX_OPCODE_IF_FCMP)
         agx_fcmpsel_to(&b, I->dest[0], if_->src[0], if_->src[1], t, f,
                        if_->fcond);
      else
         agx_icmpsel_to(&b, I->dest[0], if_->src[0], if_->src[1], t, f,
                        if_->icond);
   }

   if (if_->op == AGX_OPCODE_IF_FCMP) {
      agx_break_if_fcmp(&b, if_->src[0], if_->src[1], new_nest,
                        if_->invert_cond, if_->fcond, break_->target);
   } else {
      agx_break_if_icmp(&b, if_->src[0], if_->src[1], new_nest,
                        if_->invert_cond, if_->icond, break_->target);
   }

   agx_remove_instruction(if_);
   agx_remove_instruction(break_);
   agx_remove_instruction(pop);
   agx_foreach_instr_in_block_safe(then_block, I)
      agx_remove_instruction(I);
}

void
agx_opt_break_if(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      match_block(ctx, block);
   }
}
