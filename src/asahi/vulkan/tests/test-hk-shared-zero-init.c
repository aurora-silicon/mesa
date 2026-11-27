/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>

#include "hk_nir_shared_zero_init.h"
#include "nir_builder.h"

static unsigned tests, failures;

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else {                                                                 \
         printf("PASS %s\n", name);                                            \
      }                                                                        \
   } while (0)

static nir_builder
new_cs(uint16_t wx, uint16_t wy, uint16_t wz, uint32_t shared_size)
{
   static const nir_shader_compiler_options opts = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &opts,
                                                  "shared-zero-init-test");
   b.shader->info.workgroup_size[0] = wx;
   b.shader->info.workgroup_size[1] = wy;
   b.shader->info.workgroup_size[2] = wz;
   b.shader->info.shared_size = shared_size;
   b.shader->info.zero_initialize_shared_memory = true;
   return b;
}

static void
emit_wg_shared_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

/* Conv2D horizontal pattern: wg=(16,16,1), shared=2048B,
 * for (k = lid.x; k < 32; k += 16) shared[(lid.y * 32 + k) * 4] = val; */
static void
test_conv2d_horizontal(void)
{
   nir_builder b = new_cs(16, 16, 1, 2048);
   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, lid, 0);
   nir_def *ly = nir_channel(&b, lid, 1);

   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_store_var(&b, k_var, lx, 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_break_if(&b, nir_ige_imm(&b, k, 32));
      nir_def *off =
         nir_ishl_imm(&b, nir_iadd(&b, nir_imul_imm(&b, ly, 32), k), 2);
      nir_store_shared(&b, nir_imm_int(&b, 42), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 16), 0x1);
   }
   nir_pop_loop(&b, loop);
   emit_wg_shared_barrier(&b);

   nir_lower_vars_to_ssa(b.shader);
   T("conv2d horizontal loop covers full 2048B",
     hk_nir_shared_memory_fully_initialized(b.shader));
   ralloc_free(b.shader);
}

/* Conv2D vertical pattern: wg=(16,16,1), shared=2048B,
 * for (k = lid.y; k < 32; k += 16) shared[((k << 4) | lid.x) << 2] = val; */
static void
test_conv2d_vertical_ior(void)
{
   nir_builder b = new_cs(16, 16, 1, 2048);
   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, lid, 0);
   nir_def *ly = nir_channel(&b, lid, 1);

   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_store_var(&b, k_var, ly, 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_break_if(&b, nir_uge_imm(&b, k, 32));
      nir_def *off =
         nir_ishl_imm(&b, nir_ior(&b, nir_ishl_imm(&b, k, 4), lx), 2);
      nir_store_shared(&b, nir_imm_int(&b, 7), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 16), 0x1);
   }
   nir_pop_loop(&b, loop);
   emit_wg_shared_barrier(&b);

   nir_lower_vars_to_ssa(b.shader);
   T("conv2d vertical loop with ior covers full 2048B",
     hk_nir_shared_memory_fully_initialized(b.shader));
   ralloc_free(b.shader);
}

/* The vertical fill as DXVK emits it: the shared byte offset is
 * amul((lid.y << 5) + k, 4), with k counting from lid.x by 16 below 32. */
static void
test_conv2d_dxvk_amul(void)
{
   nir_builder b = new_cs(16, 16, 1, 2048);
   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, lid, 0);
   nir_def *ly = nir_channel(&b, lid, 1);

   nir_variable *k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_store_var(&b, k_var, lx, 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_break_if(&b, nir_ige_imm(&b, k, 32));
      nir_def *row = nir_iadd(&b, nir_ishl_imm(&b, ly, 5), k);
      nir_def *off = nir_iadd(&b, nir_imm_int(&b, 0),
                              nir_amul(&b, row, nir_imm_int(&b, 4)));
      nir_store_shared(&b, nir_imm_int(&b, 3), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 16), 0x1);
   }
   nir_pop_loop(&b, loop);
   emit_wg_shared_barrier(&b);

   nir_lower_vars_to_ssa(b.shader);
   T("DXVK amul-scaled fill covers full 2048B",
     hk_nir_shared_memory_fully_initialized(b.shader));
   ralloc_free(b.shader);

   /* Same shape with a 3-byte stride leaves gaps: keep the zero-fill. */
   b = new_cs(16, 16, 1, 2048);
   lid = nir_load_local_invocation_id(&b);
   lx = nir_channel(&b, lid, 0);
   ly = nir_channel(&b, lid, 1);
   k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_store_var(&b, k_var, lx, 0x1);
   loop = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_var);
      nir_break_if(&b, nir_ige_imm(&b, k, 32));
      nir_def *row = nir_iadd(&b, nir_ishl_imm(&b, ly, 5), k);
      nir_def *off = nir_amul(&b, row, nir_imm_int(&b, 2));
      nir_store_shared(&b, nir_imm_int(&b, 3), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 16), 0x1);
   }
   nir_pop_loop(&b, loop);
   emit_wg_shared_barrier(&b);
   nir_lower_vars_to_ssa(b.shader);
   T("amul fill with a short stride keeps zero-init",
     !hk_nir_shared_memory_fully_initialized(b.shader));
   ralloc_free(b.shader);
}

/* Straight-line vec4 init by local_invocation_index. */
static void
test_direct_vec4_init(void)
{
   nir_builder b = new_cs(64, 1, 1, 1024);
   nir_def *idx = nir_load_local_invocation_index(&b);
   nir_def *off = nir_ishl_imm(&b, idx, 4);
   nir_store_shared(&b, nir_imm_ivec4(&b, 1, 2, 3, 4), off, .write_mask = 0xf,
                    .align_mul = 16);
   emit_wg_shared_barrier(&b);

   T("direct vec4 local_invocation_index init covers 1024B",
     hk_nir_shared_memory_fully_initialized(b.shader));
   ralloc_free(b.shader);
}

/* Negative tests: partial coverage, pre-barrier read, atomic, conditional
 * store, missing barrier, non-shared barrier. */
static void
test_negative_cases(void)
{
   /* 1. Missing 4 bytes (shared=2052B, loop only covers 2048B). */
   {
      nir_builder b = new_cs(16, 16, 1, 2052);
      nir_def *lid = nir_load_local_invocation_id(&b);
      nir_def *lx = nir_channel(&b, lid, 0);
      nir_def *ly = nir_channel(&b, lid, 1);
      nir_variable *k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
      nir_store_var(&b, k_var, lx, 0x1);
      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, k_var);
         nir_break_if(&b, nir_ige_imm(&b, k, 32));
         nir_def *off =
            nir_ishl_imm(&b, nir_iadd(&b, nir_imul_imm(&b, ly, 32), k), 2);
         nir_store_shared(&b, nir_imm_int(&b, 1), off, .write_mask = 0x1,
                          .align_mul = 4);
         nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 16), 0x1);
      }
      nir_pop_loop(&b, loop);
      emit_wg_shared_barrier(&b);
      nir_lower_vars_to_ssa(b.shader);
      T("partial coverage keeps zero-init",
        !hk_nir_shared_memory_fully_initialized(b.shader));
      ralloc_free(b.shader);
   }

   /* 2. Read before barrier. */
   {
      nir_builder b = new_cs(64, 1, 1, 256);
      nir_def *idx = nir_load_local_invocation_index(&b);
      nir_def *off = nir_ishl_imm(&b, idx, 2);
      nir_def *old = nir_load_shared(&b, 1, 32, off, .align_mul = 4);
      nir_store_shared(&b, nir_iadd_imm(&b, old, 1), off, .write_mask = 0x1,
                       .align_mul = 4);
      emit_wg_shared_barrier(&b);
      T("load_shared before barrier keeps zero-init",
        !hk_nir_shared_memory_fully_initialized(b.shader));
      ralloc_free(b.shader);
   }

   /* 3. Atomic before barrier. */
   {
      nir_builder b = new_cs(64, 1, 1, 256);
      nir_def *idx = nir_load_local_invocation_index(&b);
      nir_def *off = nir_ishl_imm(&b, idx, 2);
      nir_shared_atomic(&b, 32, off, nir_imm_int(&b, 1),
                        .atomic_op = nir_atomic_op_iadd);
      nir_store_shared(&b, nir_imm_int(&b, 0), off, .write_mask = 0x1,
                       .align_mul = 4);
      emit_wg_shared_barrier(&b);
      T("shared_atomic before barrier keeps zero-init",
        !hk_nir_shared_memory_fully_initialized(b.shader));
      ralloc_free(b.shader);
   }

   /* 4. Conditional store. */
   {
      nir_builder b = new_cs(64, 1, 1, 256);
      nir_def *idx = nir_load_local_invocation_index(&b);
      nir_def *off = nir_ishl_imm(&b, idx, 2);
      nir_push_if(&b, nir_ult_imm(&b, idx, 64));
      nir_store_shared(&b, nir_imm_int(&b, 0), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_pop_if(&b, NULL);
      emit_wg_shared_barrier(&b);
      T("conditional store keeps zero-init",
        !hk_nir_shared_memory_fully_initialized(b.shader));
      ralloc_free(b.shader);
   }

   /* 5. Non-shared barrier only. */
   {
      nir_builder b = new_cs(64, 1, 1, 256);
      nir_def *idx = nir_load_local_invocation_index(&b);
      nir_def *off = nir_ishl_imm(&b, idx, 2);
      nir_store_shared(&b, nir_imm_int(&b, 0), off, .write_mask = 0x1,
                       .align_mul = 4);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_ssbo);
      T("non-shared barrier keeps zero-init",
        !hk_nir_shared_memory_fully_initialized(b.shader));
      ralloc_free(b.shader);
   }
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_conv2d_horizontal();
   test_conv2d_vertical_ior();
   test_conv2d_dxvk_amul();
   test_direct_vec4_init();
   test_negative_cases();
   glsl_type_singleton_decref();
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
