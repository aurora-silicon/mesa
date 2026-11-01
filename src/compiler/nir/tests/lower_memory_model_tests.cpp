/*
 * Copyright 2026 Jeff Queisser
 * SPDX-License-Identifier: MIT
 */

#include "nir_test.h"

class nir_lower_memory_model_test : public nir_test {
protected:
   nir_lower_memory_model_test()
      : nir_test::nir_test("nir_lower_memory_model_test")
   {
   }

   nir_intrinsic_instr *store_ssbo(enum gl_access_qualifier access)
   {
      nir_intrinsic_instr *store =
         nir_store_ssbo(b, nir_imm_int(b, 7), nir_imm_int(b, 0),
                        nir_imm_int(b, 0));
      nir_intrinsic_set_access(store, access);
      return store;
   }

   nir_intrinsic_instr *load_ssbo(enum gl_access_qualifier access)
   {
      nir_intrinsic_instr *load = nir_def_as_intrinsic(
         nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, 0)));
      nir_intrinsic_set_access(load, access);
      return load;
   }

   void barrier(nir_memory_semantics semantics)
   {
      nir_scoped_memory_barrier(b, SCOPE_DEVICE, semantics, nir_var_mem_ssbo);
   }

   void release() { barrier((nir_memory_semantics)(NIR_MEMORY_RELEASE |
                                                   NIR_MEMORY_MAKE_AVAILABLE)); }
   void acquire() { barrier((nir_memory_semantics)(NIR_MEMORY_ACQUIRE |
                                                   NIR_MEMORY_MAKE_VISIBLE)); }

   static bool coherent(nir_intrinsic_instr *intrin)
   {
      return nir_intrinsic_access(intrin) & ACCESS_COHERENT;
   }

   bool lower(bool non_readable_available)
   {
      nir_lower_memory_model_options opts = {};
      opts.non_readable_available = non_readable_available;
      return nir_lower_memory_model_with_options(b->shader, &opts);
   }
};

/* A store made available by a later release becomes coherent. */
TEST_F(nir_lower_memory_model_test, released_store_coherent)
{
   nir_intrinsic_instr *store = store_ssbo((enum gl_access_qualifier)0);
   release();

   ASSERT_TRUE(nir_lower_memory_model(b->shader));
   EXPECT_TRUE(coherent(store));
}

/* By default a write-only store keeps its exemption, also under a release:
 * nir_lower_memory_model() is unchanged for every driver that does not opt
 * in. (The pass still reports progress: it drops the barrier's
 * availability bit.) */
TEST_F(nir_lower_memory_model_test, non_readable_store_exempt_by_default)
{
   nir_intrinsic_instr *store = store_ssbo(ACCESS_NON_READABLE);
   release();

   nir_lower_memory_model(b->shader);
   EXPECT_FALSE(coherent(store));
}

/* So does passing no options, or the option clear. */
TEST_F(nir_lower_memory_model_test, non_readable_store_exempt_no_options)
{
   nir_intrinsic_instr *store = store_ssbo(ACCESS_NON_READABLE);
   release();

   nir_lower_memory_model_with_options(b->shader, NULL);
   EXPECT_FALSE(coherent(store));
}

TEST_F(nir_lower_memory_model_test, non_readable_store_exempt_option_clear)
{
   nir_intrinsic_instr *store = store_ssbo(ACCESS_NON_READABLE);
   release();

   lower(false);
   EXPECT_FALSE(coherent(store));
}

/* Opted in, a released write-only store becomes coherent. */
TEST_F(nir_lower_memory_model_test, non_readable_store_released)
{
   nir_intrinsic_instr *store = store_ssbo(ACCESS_NON_READABLE);
   release();

   ASSERT_TRUE(lower(true));
   EXPECT_TRUE(coherent(store));
}

/* Opted in, a write-only store that no release follows stays plain. */
TEST_F(nir_lower_memory_model_test, non_readable_store_unreleased)
{
   release();
   nir_intrinsic_instr *store = store_ssbo(ACCESS_NON_READABLE);

   lower(true);
   EXPECT_FALSE(coherent(store));
}

/* The option is about availability only: on the visibility side the
 * exemptions are unchanged: a read-only load after an acquire stays plain,
 * while a plain load becomes coherent as before. */
TEST_F(nir_lower_memory_model_test, visibility_unchanged)
{
   acquire();
   nir_intrinsic_instr *readonly = load_ssbo(ACCESS_NON_WRITEABLE);
   nir_intrinsic_instr *plain = load_ssbo((enum gl_access_qualifier)0);

   ASSERT_TRUE(lower(true));
   EXPECT_FALSE(coherent(readonly));
   EXPECT_TRUE(coherent(plain));
}
