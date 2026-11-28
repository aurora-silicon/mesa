/* SPDX-License-Identifier: MIT */
#pragma once

#include "nir_builder.h"
#include "nir_builtin_builder.h"

/* HK-only flag, disjoint from the shared AGX texture flags. */
#define HK_TEXTURE_FLAG_COMBINED_BIAS (1u << 3)

/* A sample's sampler, when the shader's pipeline layout fixes it (an
 * immutable sampler). */
struct hk_known_sampler_bias {
   /* The LOD bias the sampler descriptor holds. */
   float bias;
   /* The bias is strictly hk_d3d_nearest_mip's +1/64 over an API bias of 0,
    * and the sampler's filtering state makes +1/64 a no-op at explicit level
    * 0 (nearest mip, equal mag/min filter, isotropic, non-comparison,
    * normalized coordinates, weighted-average reduction, non-YCbCr). */
   bool nearest_mip_adjustment_only;
};

/* Optional data for hk_lower_g17p_sampler_bias(): `known` returns true and
 * fills `out` for a sample whose sampler is known. */
struct hk_sampler_bias_lookup {
   bool (*known)(const nir_tex_instr *tex, const void *state,
                 struct hk_known_sampler_bias *out);
   const void *state;
};

static inline bool
hk_tex_level_is_zero(const nir_tex_instr *tex)
{
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   return tex->op == nir_texop_txl && lod >= 0 &&
          nir_src_is_const(tex->src[lod].src) &&
          nir_src_as_float(tex->src[lod].src) == 0.0;
}

static inline bool
hk_lower_g17p_sampler_bias(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct hk_sampler_bias_lookup *lookup = data;
   if ((tex->op != nir_texop_txb && tex->op != nir_texop_txl) ||
       tex->is_shadow || (tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS))
      return false;

   /* The sampler and instruction each truncate to signed 5:6 fixed point.
    * Adding those separately rounded values can exceed the filtering error
    * bound. Combine in float before the single instruction quantization,
    * selecting an otherwise identical zero-bias sampler for this sample. */
   b->cursor = nir_before_instr(&tex->instr);
   struct hk_known_sampler_bias known;
   nir_def *bias;
   if (lookup && lookup->known(tex, lookup->state, &known)) {
      /* The same value the sampler descriptor holds, as a constant: a
       * constant LOD then stays a constant, which the sample instruction
       * takes as an immediate instead of a per-sample staged register.
       *
       * At an explicit level of exactly 0, hk_d3d_nearest_mip's +1/64 only
       * moves the LOD to 1/64. A nearest-mip sampler still picks level 0 there
       * (and both clamp to the same level), and with one filter for
       * magnification and minification the mag/min choice can't change the
       * result. The adjustment is then a no-op: drop it, so the level stays
       * the constant 0. */
      const bool noop = known.nearest_mip_adjustment_only &&
                        hk_tex_level_is_zero(tex);
      bias = nir_imm_float(b, noop ? 0.0f : known.bias);
   } else {
      bias = nir_build_texture_query(b, tex, nir_texop_lod_bias, 1,
                                     nir_type_float32, false, false);
      nir_instr_as_tex(nir_def_instr(bias))->backend_flags |=
         HK_TEXTURE_FLAG_COMBINED_BIAS;
   }
   /* An explicit LOD takes the sampler's bias too, and the sum has to clamp as
    * one value: the hardware would clamp a negative LOD before biasing it. */
   nir_src *shader_bias =
      &tex->src[nir_tex_instr_src_index(tex, tex->op == nir_texop_txl
                                                ? nir_tex_src_lod
                                                : nir_tex_src_bias)].src;
   nir_src_rewrite(shader_bias,
                   nir_fadd(b, nir_f2f32(b, shader_bias->ssa),
                            nir_f2f32(b, bias)));
   tex->backend_flags |= HK_TEXTURE_FLAG_COMBINED_BIAS;
   return true;
}
