/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>
#include <string>

/* Scratch blocks on the M5 Ultra (agx_scratch_dynamic_in_order): a block's
 * independent scratch loads issue ahead of their consumers, since a thread's
 * scratch accesses need no wait between them there. Other chips keep the
 * pressure order, one load and then its consumer. */

namespace {

struct scratch_shader {
   void *mem;
   agx_shader_key key;
   agx_builder *b;
   agx_block *body;
};

static void
scratch_shader_init(scratch_shader *s, enum agx_chip chip)
{
   s->mem = ralloc_context(NULL);
   s->key = {};
   s->key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   s->key.dev.chip = chip;
   s->b = agx_test_builder(s->mem);
   s->b->shader->key = &s->key;
   s->b->shader->stage = MESA_SHADER_FRAGMENT;
   s->b->shader->nir = rzalloc(s->mem, nir_shader);
   s->b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* A loop body block, as a scratch loop unrolled by nir_opt_loop_unroll's
    * exact partial unroll leaves it. */
   agx_block *entry = agx_start_block(s->b->shader);
   s->body = agx_test_block(s->b->shader);
   agx_block *exit = agx_test_block(s->b->shader);
   s->body->loop_header = true;
   agx_block_add_successor(entry, s->body);
   agx_block_add_successor(s->body, s->body);
   agx_block_add_successor(s->body, exit);
}

/* n reads of a data-chosen word, each accumulated into one chain, in the
 * pressure order: each load right before its consumer. With a store, the
 * block writes the first location after the first load. */
static void
emit_reads(scratch_shader *s, unsigned n, bool store)
{
   agx_builder *b = s->b;
   agx_index base = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));

   b->cursor = agx_after_block(s->body);
   for (unsigned i = 0; i < n; ++i) {
      agx_index offset = agx_iadd(b, base, agx_immediate(4 * i), 0);
      agx_index word = agx_temp(b->shader, AGX_SIZE_32);
      agx_stack_load_to(b, word, offset, AGX_FORMAT_I32, 1);
      acc = agx_imad(b, acc, agx_immediate(31), word, 0);
      if (store && i == 0)
         agx_stack_store(b, acc, offset, AGX_FORMAT_I32, 1);
   }
   agx_stack_store(b, acc, base, AGX_FORMAT_I32, 1);
}

/* Loads issued ahead of the first consumer of any load. */
static unsigned
loads_before_first_use(agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_IMAD)
         break;
      loads += I->op == AGX_OPCODE_STACK_LOAD;
   }
   return loads;
}

/* Memory accesses in order: S for a store, L for a load. */
static std::string
memory_order(agx_block *block)
{
   std::string order;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_STACK_LOAD)
         order += 'L';
      else if (I->op == AGX_OPCODE_STACK_STORE)
         order += 'S';
   }
   return order;
}

} /* namespace */

TEST(M5ScratchSchedule, UltraIssuesLoadsAheadOfConsumers)
{
   scratch_shader s;
   scratch_shader_init(&s, AGX_CHIP_G17D);
   emit_reads(&s, 16, false);
   agx_ilp_schedule(s.b->shader);

   EXPECT_EQ(loads_before_first_use(s.body), 16u);
   EXPECT_EQ(memory_order(s.body), std::string(16, 'L') + "S");
   ralloc_free(s.mem);
}

TEST(M5ScratchSchedule, StoresKeepProgramOrder)
{
   /* A load after a store to scratch stays behind it. */
   scratch_shader s;
   scratch_shader_init(&s, AGX_CHIP_G17D);
   emit_reads(&s, 8, true);
   agx_ilp_schedule(s.b->shader);

   EXPECT_EQ(memory_order(s.body), "LS" + std::string(7, 'L') + "S");
   EXPECT_EQ(loads_before_first_use(s.body), 1u);
   ralloc_free(s.mem);
}

TEST(M5ScratchSchedule, OtherChipsUnchanged)
{
   const enum agx_chip chips[] = {AGX_CHIP_G17S, AGX_CHIP_G17G};
   for (enum agx_chip chip : chips) {
      scratch_shader s;
      scratch_shader_init(&s, chip);
      emit_reads(&s, 16, false);
      agx_ilp_schedule(s.b->shader);

      EXPECT_EQ(loads_before_first_use(s.body), 1u);
      ralloc_free(s.mem);
   }
}
