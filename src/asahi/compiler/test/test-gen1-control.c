/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_compile.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* The candidate control word a query with constant ray flags launches with:
 * the first word of the two-word store to query word 0. */
static uint32_t
control_word(unsigned ray_flags)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "Gen1 candidate control");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_variable *query = nir_local_variable_create(b.impl,
      glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10003000000ull),
      nir_imm_int(&b, ray_flags), nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 0.0001),
      nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 9));
   nir_store_global(&b, nir_b2i32(&b, nir_rq_proceed(&b, 1, rq)),
      nir_imm_int64(&b, 0x10004000000ull), .align_mul = 4, .write_mask = 1);
   nir_shader_gather_info(b.shader, b.impl);
   struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = 118, .bvh_uniform = 96, .status_uniform = 128,
      .context_uniform = 160, .as_table_uniform = 164, .as_table_first = 1,
      .launch_header = 0x210000, .instance_traversal = true,
      .candidates = true, .compute_kernel = true,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_dead_cf);
      NIR_PASS(progress, b.shader, nir_opt_dce);
   } while (progress);
   unsigned stores = 0;
   uint32_t control = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_scs_indexed_agx ||
             nir_intrinsic_base(intr) != 0 ||
             intr->src[0].ssa->num_components != 2)
            continue;
         nir_scalar word = nir_scalar_resolved(intr->src[0].ssa, 0);
         CHECK(nir_scalar_is_const(word));
         control = nir_scalar_as_uint(word);
         ++stores;
      }
   }
   CHECK(stores == 1);
   nir_validate_shader(b.shader, "candidate control");
   ralloc_free(b.shader);
   return control;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   /* Vulkan ray flags: Opaque 1, NoOpaque 2, TerminateOnFirstHit 4,
    * SkipClosestHitShader 8, CullBackFacing 0x10, CullFrontFacing 0x20,
    * CullOpaque 0x40, CullNoOpaque 0x80, SkipTriangles 0x100, SkipAABBs
    * 0x200. */
   static const struct { unsigned flags; uint32_t control; } cases[] = {
      /* Nothing software has to decide for an opaque triangle: the hardware
       * commits those and queues the rest. */
      {0, 0x0142}, {4, 0x0142}, {8, 0x0142}, {0x200, 0x0142}, {0x20c, 0x0142},
      /* Opaque rays commit every triangle hit: the hardware commits them
       * all and queues none. */
      {1, 0x4942}, {5, 0x4942}, {9, 0x4942}, {0x201, 0x4942}, {0x20d, 0x4942},
      /* Software has to see every triangle hit. */
      {2, 0x9142}, {3, 0x9142}, {0x10, 0x9142}, {0x11, 0x9142},
      {0x20, 0x9142}, {0x21, 0x9142}, {0x40, 0x9142}, {0x41, 0x9142},
      {0x80, 0x9142}, {0x81, 0x9142}, {0x100, 0x9142}, {0x101, 0x9142},
      {0x15, 0x9142}, {0x3ff, 0x9142},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i)
      CHECK(control_word(cases[i].flags) == cases[i].control);
   glsl_type_singleton_decref();
   printf("Gen1 candidate control: %zu passed\n", ARRAY_SIZE(cases));
   return 0;
}
