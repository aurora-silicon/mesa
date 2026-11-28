/* SPDX-License-Identifier: MIT */
#include "hk_nir_sampler_bias.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "compiler/glsl_types.h"
#include <stdio.h>

/* Keep this test's assert()s in NDEBUG builds. NIR's headers above must see
 * the build's NDEBUG, so that nir_validate_shader() matches what libnir
 * provides (an inline no-op in NDEBUG builds). */
#undef NDEBUG
#include <assert.h>

static const nir_shader_compiler_options options = {0};

static nir_tex_instr *
texture(nir_builder *b, nir_texop op, bool shadow)
{
   unsigned count = 3 + (op == nir_texop_txb || op == nir_texop_txl) +
                    2 * (op == nir_texop_txd) + shadow;
   unsigned n = 0;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, count);
   tex->op = op;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_shadow = shadow;
   tex->is_new_style_shadow = shadow;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_vec2(b, .5, .5));
   /* Distinct dynamic handles must survive the query and custom-border copy. */
   tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_texture_handle,
                         nir_load_uniform(b, 1, 32, nir_imm_int(b, 0)));
   tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle,
                         nir_load_uniform(b, 1, 32, nir_imm_int(b, 4)));
   if (op == nir_texop_txb)
      tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_bias, nir_imm_float(b, .590551f));
   if (op == nir_texop_txl)
      tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(b, 1));
   if (op == nir_texop_txd) {
      tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_imm_vec2(b, 1, 0));
      tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_imm_vec2(b, 0, 1));
   }
   if (shadow)
      tex->src[n++] = nir_tex_src_for_ssa(nir_tex_src_comparator, nir_imm_float(b, .5));
   nir_def_init(&tex->instr, &tex->def, nir_tex_instr_dest_size(tex), 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static unsigned
bias_queries(nir_shader *shader)
{
   unsigned queries = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            queries += instr->type == nir_instr_type_tex &&
                       nir_instr_as_tex(instr)->op == nir_texop_lod_bias;
         }
      }
   }
   return queries;
}

/* An immutable sampler of the test's pipeline layout. */
static bool
known_bias(const nir_tex_instr *tex, const void *state,
           struct hk_known_sampler_bias *out)
{
   (void)tex;
   *out = *(const struct hk_known_sampler_bias *)state;
   return true;
}

/* The constant level or bias a known sampler leaves a sample with. */
static float
folded_lod(nir_texop op, float level, struct hk_known_sampler_bias sampler)
{
   const struct hk_sampler_bias_lookup lookup = {
      .known = known_bias,
      .state = &sampler,
   };
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &options, "immutable sampler bias");
   nir_tex_instr *tex = texture(&b, op, false);
   if (op == nir_texop_txl)
      nir_src_rewrite(&tex->src[nir_tex_instr_src_index(tex, nir_tex_src_lod)].src,
                      nir_imm_float(&b, level));
   assert(nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                              nir_metadata_control_flow, (void *)&lookup));
   assert(!nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                               nir_metadata_control_flow, (void *)&lookup));
   assert(tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS);
   assert(bias_queries(b.shader) == 0);

   nir_opt_constant_folding(b.shader);
   const int src = nir_tex_instr_src_index(
      tex, op == nir_texop_txl ? nir_tex_src_lod : nir_tex_src_bias);
   assert(nir_src_is_const(tex->src[src].src));
   const float value = nir_src_as_float(tex->src[src].src);
   ralloc_free(b.shader);
   return value;
}


/* An immutable sampler lookup that reports the sampler as unknown. */
static bool
unknown_bias(const nir_tex_instr *tex, const void *state,
             struct hk_known_sampler_bias *out)
{
   (void)tex;
   (void)state;
   (void)out;
   return false;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const nir_texop ops[] = {nir_texop_tex, nir_texop_txb, nir_texop_txl,
                            nir_texop_txd, nir_texop_lod};
   unsigned checks = 0;
   for (unsigned shadow = 0; shadow < 2; ++shadow) {
      for (unsigned i = 0; i < ARRAY_SIZE(ops); ++i) {
         if (shadow && ops[i] == nir_texop_lod)
            continue;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &options, "sampler bias");
         nir_tex_instr *tex = texture(&b, ops[i], shadow);
         nir_def *handle = nir_get_tex_src(tex, nir_tex_src_sampler_handle);
         /* An explicit LOD takes the sampler's bias too (D3D LOD rules). */
         bool expected =
            (ops[i] == nir_texop_txb || ops[i] == nir_texop_txl) && !shadow;
         bool changed = nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                                            nir_metadata_control_flow, NULL);
         assert(changed == expected);
         assert(!nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                                    nir_metadata_control_flow, NULL));
         unsigned queries = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_tex)
                     continue;
                  nir_tex_instr *q = nir_instr_as_tex(instr);
                  if (q->op != nir_texop_lod_bias)
                     continue;
                  queries++;
                  assert(q->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS);
                  assert(nir_get_tex_src(q, nir_tex_src_sampler_handle) == handle);
               }
            }
         }
         assert(queries == expected);
         if (expected) {
            b.cursor = nir_after_instr(&tex->instr);
            nir_tex_instr *copy = nir_instr_as_tex(nir_instr_clone(b.shader, &tex->instr));
            copy->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;
            nir_builder_instr_insert(&b, &copy->instr);
            assert(copy->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS);
            assert(nir_get_tex_src(copy, nir_tex_src_sampler_handle) == handle);
            assert(!nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                                       nir_metadata_control_flow, NULL));
         }
         /* nir_validate_shader is compiled out of this NDEBUG build. */
         ralloc_free(b.shader);
         checks++;
      }
   }

   /* A known (immutable) sampler bias is added as a constant: no descriptor
    * query, the zero-bias sampler is still selected, and a constant level or
    * bias folds to the one constant the instruction takes as an immediate. */
   const struct hk_known_sampler_bias api_bias = {.bias = -1.0f};
   const struct hk_known_sampler_bias zero_bias = {.bias = 0.0f};
   assert(folded_lod(nir_texop_txl, 1.0f, api_bias) == 0.0f);
   assert(folded_lod(nir_texop_txl, 0.0f, zero_bias) == 0.0f);
   assert(folded_lod(nir_texop_txb, 0.0f, api_bias) == .590551f - 1.0f);
   assert(folded_lod(nir_texop_txb, 0.0f, zero_bias) == .590551f);
   checks += 4;

   /* hk_d3d_nearest_mip's +1/64 is dropped only where it is provably a no-op:
    * an explicit level of exactly 0 on an eligible nearest-mip sampler. At
    * 0.5 (the D3D tie-break boundary) or any other nonzero level, +1/64 stays. */
   const float d3d = 1.0f / 64.0f;
   const struct hk_known_sampler_bias nearest_only = {
      .bias = d3d,
      .nearest_mip_adjustment_only = true,
   };
   const struct hk_known_sampler_bias nearest = {.bias = d3d};
   assert(folded_lod(nir_texop_txl, 0.0f, nearest_only) == 0.0f);
   assert(folded_lod(nir_texop_txl, -0.0f, nearest_only) == 0.0f);
   assert(folded_lod(nir_texop_txl, 0.5f, nearest_only) == 0.5f + d3d);
   assert(folded_lod(nir_texop_txl, 1.0f, nearest_only) == 1.0f + d3d);
   assert(folded_lod(nir_texop_txl, -1.0f, nearest_only) == -1.0f + d3d);
   assert(folded_lod(nir_texop_txb, 0.0f, nearest_only) == .590551f + d3d);
   assert(folded_lod(nir_texop_txl, 0.0f, nearest) == d3d);
   checks += 7;

   /* Unknown samplers fall back to the descriptor query; shadow samples stay
    * untouched even when a lookup callback is provided. */
   {
      const struct hk_sampler_bias_lookup miss = {.known = unknown_bias};
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &options, "unknown sampler bias");
      nir_tex_instr *tex = texture(&b, nir_texop_txl, false);
      assert(nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                                 nir_metadata_control_flow, (void *)&miss));
      assert(bias_queries(b.shader) == 1);
      assert(tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS);
      ralloc_free(b.shader);

      const struct hk_sampler_bias_lookup hit = {
         .known = known_bias,
         .state = &nearest_only,
      };
      b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &options, "shadow immutable sampler");
      tex = texture(&b, nir_texop_txl, true);
      assert(!nir_shader_tex_pass(b.shader, hk_lower_g17p_sampler_bias,
                                  nir_metadata_control_flow, (void *)&hit));
      assert(bias_queries(b.shader) == 0);
      assert(!(tex->backend_flags & HK_TEXTURE_FLAG_COMBINED_BIAS));
      ralloc_free(b.shader);
      checks += 2;
   }

   glsl_type_singleton_decref();
   printf("%u sampler bias scope/idempotence/handle checks passed\n", checks);
   return 0;
}
