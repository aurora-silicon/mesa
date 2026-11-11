/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_test.h"

class nir_lower_terminate_to_demote_test : public nir_test {
protected:
   nir_lower_terminate_to_demote_test();

   unsigned count_intrinsic(nir_intrinsic_op op);
   bool ballot_is_masked();
   bool ballot_source_is_const_true();
};

nir_lower_terminate_to_demote_test::nir_lower_terminate_to_demote_test()
   : nir_test::nir_test("nir_lower_terminate_to_demote_test",
                       MESA_SHADER_FRAGMENT)
{
}

unsigned
nir_lower_terminate_to_demote_test::count_intrinsic(nir_intrinsic_op op)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, b->shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }

   return count;
}

bool
nir_lower_terminate_to_demote_test::ballot_is_masked()
{
   nir_foreach_function_impl(impl, b->shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_ballot)
               continue;

            nir_alu_instr *and_instr =
               nir_def_as_alu_or_null(intr->src[0].ssa);
            if (!and_instr || and_instr->op != nir_op_iand)
               return false;

            for (unsigned i = 0; i < 2; ++i) {
               nir_alu_instr *not_instr =
                  nir_def_as_alu_or_null(and_instr->src[i].src.ssa);
               if (not_instr && not_instr->op == nir_op_inot &&
                   nir_def_is_intrinsic(not_instr->src[0].src.ssa) &&
                   nir_def_as_intrinsic(not_instr->src[0].src.ssa)->intrinsic ==
                      nir_intrinsic_load_deref)
                  return true;
            }
         }
      }
   }

   return false;
}

bool
nir_lower_terminate_to_demote_test::ballot_source_is_const_true()
{
   nir_foreach_function_impl(impl, b->shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_ballot)
               return nir_src_is_const(intr->src[0]) &&
                      nir_src_as_bool(intr->src[0]);
         }
      }
   }

   return false;
}

TEST_F(nir_lower_terminate_to_demote_test, mask_ballot)
{
   nir_def *cond = nir_ieq_imm(b, nir_load_subgroup_invocation(b), 0);
   nir_terminate_if(b, cond);
   nir_ballot(b, 4, 32, nir_imm_true(b));
   nir_quad_vote_any(b, 1, nir_imm_true(b));

   ASSERT_TRUE(nir_mask_ballots_after_terminate(b->shader));
   nir_validate_shader(b->shader, "after masking terminated ballots");
   EXPECT_TRUE(ballot_is_masked());
   EXPECT_EQ(count_intrinsic(nir_intrinsic_is_helper_invocation), 0u);
   EXPECT_EQ(count_intrinsic(nir_intrinsic_quad_vote_any), 1u);

   ASSERT_TRUE(nir_lower_terminate_to_demote(b->shader));
   ASSERT_TRUE(nir_lower_halt_to_return(b->shader));
   ASSERT_TRUE(nir_lower_returns(b->shader));
   ASSERT_TRUE(nir_lower_vars_to_ssa(b->shader));
   nir_opt_constant_folding(b->shader);
   nir_opt_algebraic(b->shader);
   nir_validate_shader(b->shader, "after lowering terminate and variables");
   EXPECT_FALSE(ballot_source_is_const_true());
}

TEST_F(nir_lower_terminate_to_demote_test, ignore_demote_only_shader)
{
   nir_def *cond = nir_ieq_imm(b, nir_load_subgroup_invocation(b), 0);
   nir_demote_if(b, cond);
   nir_ballot(b, 4, 32, nir_imm_true(b));

   EXPECT_FALSE(nir_mask_ballots_after_terminate(b->shader));
   nir_validate_shader(b->shader, "after ignoring demote-only shader");
   EXPECT_FALSE(ballot_is_masked());
   EXPECT_EQ(count_intrinsic(nir_intrinsic_is_helper_invocation), 0u);
}
