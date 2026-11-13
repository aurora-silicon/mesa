/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"

/* Of a position store, which of z and w it writes (bits 2 and 3). */
static unsigned
zw_written(nir_intrinsic_instr *intr)
{
   return (nir_intrinsic_write_mask(intr) << nir_intrinsic_component(intr)) &
          0xc;
}

static bool
pos_access(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_has_io_semantics(intr) &&
          nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS;
}

/* Dynamic lowered I/O version of nir_lower_clip_halfz */
static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if ((intr->intrinsic != nir_intrinsic_store_output &&
        intr->intrinsic != nir_intrinsic_store_per_vertex_output) ||
       !pos_access(intr))
      return false;

   /* Output temporaries give a rasterizing stage one full store. A stage
    * linked to a geometry shader can have varyings packed into this slot by
    * nir_opt_varyings, but its hardware variant never rasterizes. Mesh
    * shaders have no output temporaries: their drivers check
    * agx_nir_clip_m1_1_complete() first.
    */
   if (zw_written(intr) != 0xc)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   /* Position math added after agx_nir_preserve_position_order must not be
    * reassociated either. */
   b->fp_math_ctrl = nir_fp_no_reassoc | nir_fp_no_transform;

   nir_def *pos = intr->src[0].ssa;
   const unsigned zi = 2 - nir_intrinsic_component(intr);
   nir_def *z = nir_channel(b, pos, zi);
   nir_def *w = nir_channel(b, pos, zi + 1);
   nir_def *c = nir_load_clip_z_coeff(b);

   /* Lerp. If c = 0, reduces to z. If c = 1/2, reduces to (z + w)/2 */
   nir_def *new_z = nir_ffma(b, z, nir_fsub(b, nir_imm_float(b, 1.0f), c),
                             nir_fmul(b, w, c));
   nir_src_rewrite(&intr->src[0], nir_vector_insert_imm(b, pos, new_z, zi));
   return true;
}

bool
agx_nir_lower_clip_m1_1(nir_shader *s)
{
   return nir_shader_intrinsics_pass(s, lower, nir_metadata_control_flow, NULL);
}

/* Whether agx_nir_lower_clip_m1_1 sees every position z of s: each position
 * store writes both z and w or neither, and nothing reads position back. */
bool
agx_nir_clip_m1_1_complete(nir_shader *s)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!pos_access(intr))
               continue;
            switch (intr->intrinsic) {
            case nir_intrinsic_store_output:
            case nir_intrinsic_store_per_vertex_output:
               if (zw_written(intr) == 0x4 || zw_written(intr) == 0x8)
                  return false;
               break;
            case nir_intrinsic_load_output:
            case nir_intrinsic_load_per_vertex_output:
               return false;
            default:
               break;
            }
         }
      }
   }
   return true;
}
