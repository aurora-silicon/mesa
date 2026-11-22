/* SPDX-License-Identifier: GPL-2.0-only OR MIT */
/* Copyright (C) 2026 Eryk Wieliczko */
#include "nir_builder.h"
#include "poly_nir.h"
#include "nir_xfb_info.h"

#include <stdio.h>

static bool
check_outputs(unsigned components)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &options, "mixed geometry output components");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   const nir_alu_type types[] = {
      nir_type_float32, nir_type_uint32, nir_type_int32, nir_type_float32,
   };
   for (unsigned c = 0; c < components; ++c) {
      nir_store_output(&b, nir_imm_int(&b, 0x3f800000 + c), nir_imm_int(&b, 0),
                       .component = c, .src_type = types[c],
                       .io_semantics.location = VARYING_SLOT_VAR0);
   }
   nir_store_output(&b, nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0), nir_imm_int(&b, 0),
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);
   nir_end_primitive(&b, 0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_shader *count = NULL, *copy = NULL, *pre = NULL;
   struct poly_gs_info info;
   poly_nir_lower_gs(b.shader, &count, &copy, &pre, &info);
   bool valid = copy != NULL;
   unsigned mask = 0;
   unsigned position_stores = 0;
   if (copy) {
      nir_validate_shader(copy, "mixed geometry outputs");
      nir_foreach_block(block, nir_shader_get_entrypoint(copy)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output)
               continue;
            unsigned location = nir_intrinsic_io_semantics(intr).location;
            if (location == VARYING_SLOT_POS) {
               ++position_stores;
               valid &= intr->num_components == 4 &&
                        nir_intrinsic_write_mask(intr) == 0xf;
            } else if (location == VARYING_SLOT_VAR0) {
               valid &= nir_intrinsic_component(intr) == 0;
               unsigned write_mask = nir_intrinsic_write_mask(intr);
               u_foreach_bit(c, write_mask) {
                  valid &= c < components && nir_intrinsic_src_type(intr) == types[c];
                  mask |= BITFIELD_BIT(c);
               }
            }
         }
      }
   }
   valid &= mask == BITFIELD_MASK(components);
   valid &= position_stores == 1;
   if (!valid)
      fprintf(stderr, "Geometry output types were not preserved (%u components)\n", components);
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(copy);
   ralloc_free(pre);
   return valid;
}

/* A line-strip GS whose gl_PointSize transform feedback captures: the
 * rasterization shader must keep the point size, since its transform feedback
 * reads it from there (an undefined value was captured before #211). */
static bool
check_xfb_point_size(void)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &options, "line strip capturing point size");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_LINE_STRIP;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 2;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.xfb_stride[0] = 1;
   for (unsigned v = 0; v < 2; ++v) {
      nir_store_output(&b, nir_imm_vec4(&b, v, 0.0, 0.0, 1.0),
                       nir_imm_int(&b, 0), .src_type = nir_type_float32,
                       .io_semantics.location = VARYING_SLOT_POS);
      nir_store_output(&b, nir_imm_float(&b, 4.0), nir_imm_int(&b, 0),
                       .src_type = nir_type_float32,
                       .io_semantics.location = VARYING_SLOT_PSIZ,
                       .io_xfb.out[0] = {.num_components = 1});
      nir_emit_vertex(&b, 0);
   }
   nir_end_primitive(&b, 0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_gather_xfb_info_from_intrinsics(b.shader);

   nir_shader *count = NULL, *copy = NULL, *pre = NULL;
   struct poly_gs_info info;
   poly_nir_lower_gs(b.shader, &count, &copy, &pre, &info);
   bool valid = b.shader->xfb_info && copy &&
                (copy->info.outputs_written & VARYING_BIT_PSIZ);
   if (!valid)
      fprintf(stderr, "A captured line-strip point size was dropped\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(copy);
   ralloc_free(pre);
   return valid;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = true;
   for (unsigned n = 1; n <= 4; ++n)
      valid &= check_outputs(n);
   valid &= check_xfb_point_size();
   glsl_type_singleton_decref();
   return valid ? 0 : 1;
}
