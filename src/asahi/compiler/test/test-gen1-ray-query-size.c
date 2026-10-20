/*
 * Gen1 native ray-query measurement harness (AGX3B / G17P).
 *
 * Prints main_size / instruction counts / binary hex for each compiled
 * ray-query shape. Two modes:
 *
 *   agx3_gen1_ray_query_size_test                 -> nir_builder query shapes
 *   agx3_gen1_ray_query_size_test file.spv ...    -> SPIR-V ray-query compute
 *
 * The .spv path runs spirv_to_nir, stubs descriptors out as fixed addresses,
 * then drives the same agx_nir_lower_gen1_ray_query + agx_preprocess_nir +
 * agx_compile_shader_nir pipeline the driver uses for a compute ray query.
 *
 * Output line format:
 *   RQSIZE <tag> main_size=<B> binary_size=<B> instrs=<n> gprs=<n> hex=<...>
 */
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/glsl_types.h"
#include "compiler/spirv/nir_spirv.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The layout honeykrisp uses on G17P with gen1_native_rq +
 * gen1_rq_instances + gen1_rq_candidates enabled (hk_shader.c). */
static const struct agx_gen1_ray_query_layout gen1_layout = {
   .payload_uniform = 118,
   .bvh_uniform = 96,
   .status_uniform = 128,
   .launch_header = 0x210000,
   .context_uniform = 160,
   .as_table_uniform = 164,
   .as_table_first = 2048, /* HK_GEN1_AS_TABLE_AS_SLOTS */
   .instance_traversal = true,
   .candidates = true,
};

/* Simpler no-candidates layout matching the legacy test shape. */
static const struct agx_gen1_ray_query_layout gen1_layout_plain = {
   .payload_uniform = 124,
   .bvh_uniform = 96,
   .status_uniform = 128,
   .launch_header = 0x110000,
   .context_uniform = 160,
   .as_table_uniform = 164,
   .as_table_first = 1,
   .instance_traversal = false,
   .candidates = false,
};

/* Instance traversal without the candidate queue. */
static const struct agx_gen1_ray_query_layout gen1_layout_instances = {
   .payload_uniform = 124,
   .bvh_uniform = 96,
   .status_uniform = 130,
   .launch_header = 0x110000,
   .context_uniform = 160,
   .as_table_uniform = 164,
   .as_table_first = 2048,
   .instance_traversal = true,
   .candidates = false,
};

static void
print_part(const char *tag, const struct agx_shader_part *part)
{
   printf("RQSIZE %s main_size=%u preamble_size=%u binary_size=%u "
          "instrs=%u alu=%u gprs=%u scratch=%u ray_traversal=%d hex=",
          tag, part->info.main_size, part->info.preamble_size,
          part->info.binary_size, part->info.stats.instrs,
          part->info.stats.alu, part->info.stats.gprs,
          part->info.scratch_size, part->info.uses_ray_traversal);
   const uint8_t *bin = part->binary;
   for (unsigned i = 0; i < part->info.binary_size; ++i)
      printf("%02x", bin[i]);
   printf("\n");
   fflush(stdout);
}

static bool
compile_and_report(nir_shader *nir, const char *tag)
{
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B},
      .no_preamble = true,
      .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(nir, &key, &part)) {
      printf("RQSIZE %s COMPILE_FAILED\n", tag);
      return false;
   }
   print_part(tag, &part);
   free(part.binary);
   return true;
}

/* One standard query, mirroring test-gen1-ray-query.c's shape. */
static void
measure_standard_ray_query(unsigned count, unsigned array, bool instances,
                           bool candidates)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "Gen1 standard ray query");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = count;
   nir_variable *queries[8];
   nir_def *rqs[8];
   assert(count && count <= 8);
   const struct glsl_type *array_type = array == 2 ?
      glsl_array_type(glsl_array_type(glsl_uint64_t_type(), 4, 0), 2, 0) :
      glsl_array_type(glsl_uint64_t_type(), count, 0);
   nir_variable *query_array = array ? nir_local_variable_create(
      b.impl, array_type, "queries") : NULL;
   if (query_array)
      query_array->data.ray_query = true;
   for (unsigned i = 0; i < count; ++i) {
      nir_variable *query = array ? query_array :
         nir_local_variable_create(b.impl, glsl_uint64_t_type(), "query");
      query->data.ray_query = true;
      queries[i] = query;
      nir_deref_instr *deref = nir_build_deref_var(&b, query);
      if (array) {
         nir_def *index = nir_iand_imm(&b,
            nir_iadd_imm(&b, nir_load_local_invocation_index(&b), i), count - 1);
         if (array == 2) {
            deref = nir_build_deref_array(&b, deref, nir_ushr_imm(&b, index, 2));
            index = nir_iand_imm(&b, index, 3);
         }
         deref = nir_build_deref_array(&b, deref, index);
      }
      nir_def *rq = &deref->def;
      rqs[i] = rq;
      nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10002000000ull),
                        candidates ? nir_ishl_imm(&b, nir_iand_imm(&b,
                           nir_load_local_invocation_index(&b), 1), 1)
                                   : nir_imm_int(&b, 1),
                        nir_imm_int(&b, 255),
                        nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
                        nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   }
   for (unsigned i = count; i-- > 0;) {
      nir_push_loop(&b);
      nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rqs[i])));
      if (candidates) {
         nir_def *candidate_id = nir_rq_load(&b, 1, 32, rqs[i],
            .ray_query_value = nir_ray_query_value_intersection_instance_id,
            .committed = false);
         nir_store_global(&b, candidate_id,
            nir_imm_int64(&b, 0x10003020000ull), .align_mul = 4,
            .write_mask = 1);
         nir_rq_confirm_intersection(&b, rqs[i]);
      }
      nir_pop_loop(&b, NULL);
   }
   nir_def *rq = rqs[0];
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   nir_def *distance = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_t,
      .committed = true);
   nir_store_global(&b, nir_vec2(&b, type, distance),
                    nir_imm_int64(&b, 0x10003010000ull),
                    .align_mul = 8, .write_mask = 3);
   if (instances) {
      nir_def *id = nir_rq_load(&b, 1, 32, rq,
         .ray_query_value = nir_ray_query_value_intersection_instance_id,
         .committed = true);
      nir_def *custom = nir_rq_load(&b, 1, 32, rq,
         .ray_query_value =
            nir_ray_query_value_intersection_instance_custom_index,
         .committed = true);
      nir_store_global(&b, nir_vec2(&b, id, custom),
         nir_imm_int64(&b, 0x10003010008ull), .align_mul = 8, .write_mask = 3);
   }
   if (candidates) {
      nir_def *object_origin = nir_rq_load(&b, 3, 32, rq,
         .ray_query_value = nir_ray_query_value_intersection_object_ray_origin,
         .committed = true);
      nir_def *object_direction = nir_rq_load(&b, 3, 32, rq,
         .ray_query_value =
            nir_ray_query_value_intersection_object_ray_direction,
         .committed = true);
      nir_def *sbt = nir_rq_load(&b, 1, 32, rq,
         .ray_query_value = nir_ray_query_value_intersection_instance_sbt_index,
         .committed = true);
      nir_store_global(&b, object_origin, nir_imm_int64(&b, 0x10003030000ull),
         .align_mul = 4, .write_mask = 7);
      nir_store_global(&b, object_direction,
         nir_imm_int64(&b, 0x10003030010ull), .align_mul = 4, .write_mask = 7);
      nir_store_global(&b, sbt, nir_imm_int64(&b, 0x10003030030ull),
         .align_mul = 4, .write_mask = 1);
   }

   const struct agx_gen1_ray_query_layout layout =
      candidates ? gen1_layout :
      instances ? gen1_layout_instances : gen1_layout_plain;
   if (!agx_nir_lower_gen1_ray_query(b.shader, &layout)) {
      printf("RQSIZE count=%u array=%u inst=%u cand=%u LOWER_FAILED\n",
             count, array, instances, candidates);
      ralloc_free(b.shader);
      return;
   }
   b.shader->info.ray_queries = 0;
   agx_preprocess_nir(b.shader);
   char tag[128];
   snprintf(tag, sizeof(tag), "nir_builder count=%u array=%u inst=%u cand=%u",
            count, array, instances, candidates);
   compile_and_report(b.shader, tag);
   ralloc_free(b.shader);
}

/* Replace descriptor loads with fixed 64-bit addresses so the Gen1 lowering
 * sees resolved pointers like it does after hk_nir_lower_descriptors. */
static bool
stub_descriptors_instr(nir_builder *b, nir_intrinsic_instr *intr, void *unused)
{
   if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   /* Base address varies by descriptor type so the BVH pointer and the result
    * buffer land in different windows. */
   nir_descriptor_type dt = nir_intrinsic_desc_type(intr);
   uint64_t base = dt == nir_descriptor_type_acceleration_structure
                      ? 0x10002000000ull : 0x10003000000ull;
   nir_def *c = nir_imm_int64(b, base);
   nir_def_replace(&intr->def, c);
   return true;
}

static bool
stub_resource_index_instr(nir_builder *b, nir_intrinsic_instr *intr,
                          void *unused)
{
   if (intr->intrinsic != nir_intrinsic_vulkan_resource_index)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *c = nir_imm_intN_t(b, 0, intr->def.bit_size);
   nir_def_replace(&intr->def, c);
   return true;
}

static void
measure_spv(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      printf("RQSIZE %s OPEN_FAILED\n", path);
      return;
   }
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   rewind(f);
   uint32_t *words = malloc(size);
   if (!words || fread(words, 1, size, f) != (size_t)size) {
      fclose(f);
      free(words);
      printf("RQSIZE %s READ_FAILED\n", path);
      return;
   }
   fclose(f);

   const struct spirv_to_nir_options spirv_opts = {
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = nir_address_format_64bit_global,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = nir_address_format_64bit_global,
      .shared_addr_format = nir_address_format_32bit_offset,
      .global_addr_format = nir_address_format_64bit_global,
      .min_ssbo_alignment = 16,
      .min_ubo_alignment = 16,
   };
   nir_shader *nir = spirv_to_nir(words, size / 4, NULL, MESA_SHADER_COMPUTE,
                                  "main", &spirv_opts, &agx_nir_options);
   free(words);
   if (!nir) {
      printf("RQSIZE %s SPIRV2NIR_FAILED\n", path);
      return;
   }
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);

   /* Mirror hk_preprocess_nir (hk_shader.c) for a compute shader. */
   NIR_PASS(_, nir, nir_lower_system_values);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);
   agx_preprocess_nir_g17p(nir);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);
   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   /* Resolve every descriptor to a fixed address, then drop derefs. This is
    * the stand-in for hk_nir_lower_descriptors. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, stub_descriptors_instr,
            nir_metadata_all, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, stub_resource_index_instr,
            nir_metadata_all, NULL);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   if (nir->info.ray_queries) {
      NIR_PASS(_, nir, nir_opt_ray_queries);
      NIR_PASS(_, nir, nir_opt_ray_query_ranges);
      if (!agx_nir_lower_gen1_ray_query(nir, &gen1_layout)) {
         printf("RQSIZE %s GEN1_LOWER_FAILED\n", path);
         ralloc_free(nir);
         return;
      }
      nir->info.ray_queries = 0;
   } else {
      printf("RQSIZE %s NO_RAY_QUERIES (compiling anyway)\n", path);
   }

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
   agx_preprocess_nir_g17p(nir);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);
   compile_and_report(nir, path);
   ralloc_free(nir);
}

int
main(int argc, char **argv)
{
   setbuf(stdout, NULL);
   glsl_type_singleton_init_or_ref();

   if (argc > 1) {
      for (int i = 1; i < argc; ++i)
         measure_spv(argv[i]);
   } else {
      measure_standard_ray_query(1, 0, false, false);
      measure_standard_ray_query(2, 0, false, false);
      measure_standard_ray_query(8, 0, false, false);
      measure_standard_ray_query(4, 1, false, false);
      measure_standard_ray_query(8, 1, false, false);
      measure_standard_ray_query(8, 2, false, false);
      measure_standard_ray_query(1, 0, true, false);
      measure_standard_ray_query(8, 2, true, false);
      measure_standard_ray_query(1, 0, true, true);
   }

   glsl_type_singleton_decref();
   return 0;
}
