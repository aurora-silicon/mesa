/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A vertex program whose texture ops take the indexed heap moves its preamble
 * loads up by the heap-pointer halves (hk_nir_shift_vertex_preamble()). A
 * native ray query in that program has already read its state from fixed
 * halves below HK_NATIVE_QUERY_STATE_HALFS, where the driver publishes it, so
 * only the vertex ABI above the query state may move. Shifting the query
 * state too made every query of a VS, or of a GS raster replay, that also
 * sampled a texture read back as a miss on G17P.
 */
#include "hk_nir_vertex_preamble.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "nir_builder.h"
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return false; \
} } while (0)

/* Halves 0..7 hold the texture/sampler heap pointers (AGX_ABI_G17P_*). */
#define HEAP_POINTER_HALFS 8
/* The first vertex-ABI half the monolithic VBO lowering places above the
 * query state, which must move with the heap pointers. */
#define VERTEX_ABI_HALF HK_NATIVE_QUERY_STATE_HALFS
#define MAX_LOADS 64

struct preamble_loads {
   unsigned count;
   unsigned base[MAX_LOADS];
};

static bool
gather_loads(nir_shader *nir, struct preamble_loads *loads)
{
   loads->count = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_preamble)
               continue;
            CHECK(loads->count < MAX_LOADS);
            loads->base[loads->count++] =
               nir_intrinsic_base(nir_instr_as_intrinsic(instr));
         }
      }
   }
   return true;
}

/* An API vertex shader with one native ray query, lowered for the graphics
 * stage as hk_lower_nir() does, plus one vertex-ABI preamble load. */
static nir_shader *
build_query_vs(bool instances)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "VS native query");
   b.shader->info.io_lowered = true;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   nir_variable *query = nir_local_variable_create(
      b.impl, glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   nir_def *as = nir_load_global_constant(&b, 1, 64,
      nir_imm_int64(&b, 0x10003000000ull), .align_mul = 8);
   nir_rq_initialize(&b, rq, as, nir_imm_int(&b, 1), nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, 0), nir_imm_float(&b, 0),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_pop_loop(&b, NULL);
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, type), nir_imm_float(&b, 0),
      nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(b.shader, b.impl);
   if (b.shader->info.ray_queries != 1)
      return NULL;
   agx_preprocess_nir_g17p(b.shader);
   NIR_PASS(_, b.shader, nir_opt_ray_queries);
   NIR_PASS(_, b.shader, nir_opt_ray_query_ranges);

   /* hk_gen1_ray_query_layout() for an API vertex shader on G17P. */
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = instances ? 118 : 124,
      .bvh_uniform = 96,
      .status_uniform = 128,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .as_table_first = 1,
      .launch_header = instances ? 0x210000 : 0x110000,
      .instance_traversal = instances,
      .candidates = instances,
      .fragment_shader = true,
      .graphics_stage = true,
      .compute_kernel = true,
   };
   if (!agx_nir_lower_gen1_ray_query(b.shader, &layout))
      return NULL;

   /* A vertex-ABI load as the monolithic VBO lowering places it for a
    * query-bearing program (agx_nir_lower_non_monolithic_uniforms_at()). */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   b.cursor = nir_before_impl(impl);
   nir_def *vbo = nir_load_preamble(&b, 1, 32, .base = VERTEX_ABI_HALF);
   b.cursor = nir_after_impl(impl);
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, vbo), nir_imm_float(&b, 0),
      nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics.location = VARYING_SLOT_VAR0);
   b.shader->info.outputs_written |= VARYING_BIT_VAR(0);
   nir_validate_shader(b.shader, "VS native query");
   return b.shader;
}

static bool
check_shift(bool instances)
{
   nir_shader *nir = build_query_vs(instances);
   CHECK(nir);

   struct preamble_loads before, after;
   CHECK(gather_loads(nir, &before));
   unsigned query_loads = 0, abi_loads = 0;
   bool context = false;
   for (unsigned i = 0; i < before.count; ++i) {
      if (before.base[i] < HK_NATIVE_QUERY_STATE_HALFS) {
         ++query_loads;
      } else {
         CHECK(before.base[i] == VERTEX_ABI_HALF);
         ++abi_loads;
      }
      context |= before.base[i] == 160;
   }
   /* The query reads its context, BVH and payload state from the preamble. */
   CHECK(query_loads >= 3 && context && abi_loads == 1);

   /* Unguarded, the shift moves the query state: the context at 160 lands on
    * the heap pointers at 168. This is the configuration that failed. */
   nir_shader *unguarded = nir_shader_clone(NULL, nir);
   CHECK(hk_nir_shift_vertex_preamble(unguarded, HEAP_POINTER_HALFS, false));
   CHECK(gather_loads(unguarded, &after));
   CHECK(after.count == before.count);
   for (unsigned i = 0; i < before.count; ++i)
      CHECK(after.base[i] == before.base[i] + HEAP_POINTER_HALFS);
   ralloc_free(unguarded);

   /* A query-bearing program keeps its query state in place and moves only
    * the vertex ABI with the heap pointers. */
   CHECK(hk_nir_shift_vertex_preamble(nir, HEAP_POINTER_HALFS, true));
   nir_validate_shader(nir, "VS native query, shifted");
   CHECK(gather_loads(nir, &after));
   CHECK(after.count == before.count);
   for (unsigned i = 0; i < before.count; ++i) {
      if (before.base[i] < HK_NATIVE_QUERY_STATE_HALFS)
         CHECK(after.base[i] == before.base[i]);
      else
         CHECK(after.base[i] == before.base[i] + HEAP_POINTER_HALFS);
   }

   ralloc_free(nir);
   printf("VS query preamble shift: instances=%u query_loads=%u PASS\n",
          instances, query_loads);
   return true;
}

/* A program without a native query moves every preamble load, as before. */
static bool
check_query_free(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "VS without query");
   b.shader->info.io_lowered = true;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   nir_def *lo = nir_load_preamble(&b, 1, 32, .base = 0);
   nir_def *hi = nir_load_preamble(&b, 1, 32, .base = 6);
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, lo), nir_u2f32(&b, hi),
      nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics.location = VARYING_SLOT_POS);
   CHECK(hk_nir_shift_vertex_preamble(b.shader, HEAP_POINTER_HALFS, false));
   struct preamble_loads loads;
   CHECK(gather_loads(b.shader, &loads));
   CHECK(loads.count == 2 && loads.base[0] == 8 && loads.base[1] == 14);
   ralloc_free(b.shader);
   printf("VS query-free preamble shift: PASS\n");
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = check_query_free();
   valid &= check_shift(false);
   valid &= check_shift(true);
   glsl_type_singleton_decref();
   return valid ? 0 : 1;
}
