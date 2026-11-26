/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_native_descriptor.h"
#include "hk_native_sampler.h"
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_chip.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"
#include "vk_pipeline.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

enum desc_family {
   DESC_COMBINED_SET = 0,
   DESC_SEPARATE_SET = 1,
   DESC_COMBINED_BUFFER = 2,
   DESC_SEPARATE_PUSH = 3,
};

static bool
ref_compare(enum agx_compare_func func, float ref, float depth, bool is_fixed)
{
   if (is_fixed) {
      ref = CLAMP(ref, 0.0f, 1.0f);
      depth = CLAMP(depth, 0.0f, 1.0f);
   }
   switch (func) {
   case AGX_COMPARE_FUNC_NEVER:
      return false;
   case AGX_COMPARE_FUNC_LESS:
      return ref < depth;
   case AGX_COMPARE_FUNC_EQUAL:
      return ref == depth;
   case AGX_COMPARE_FUNC_LEQUAL:
      return ref <= depth;
   case AGX_COMPARE_FUNC_GREATER:
      return ref > depth;
   case AGX_COMPARE_FUNC_NOT_EQUAL:
      return ref != depth;
   case AGX_COMPARE_FUNC_GEQUAL:
      return ref >= depth;
   case AGX_COMPARE_FUNC_ALWAYS:
      return true;
   }
   abort();
}

static struct hk_descriptor_set_layout *
make_layout(enum agx_chip chip, enum desc_family family,
            unsigned generation_override)
{
   const unsigned gen = generation_override
                           ? generation_override
                           : agx_chip_descriptor_generation(chip);
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(gen);
   CHECK(native.stride);

   const bool separate =
      family == DESC_SEPARATE_SET || family == DESC_SEPARATE_PUSH;
   const uint32_t bindings = separate ? 2 : 1;
   const uint32_t stride = family == DESC_COMBINED_BUFFER ? 128 : native.stride;

   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + bindings * sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = gen;
   layout->native_sampler_indices = agx_chip_native_or_m5(chip);
   if (family == DESC_COMBINED_BUFFER)
      layout->vk.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
   if (family == DESC_SEPARATE_PUSH)
      layout->vk.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

   layout->binding_count = bindings;
   if (!separate) {
      layout->binding[0] = (struct hk_descriptor_set_binding_layout){
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .array_size = 1,
         .stride = stride,
         .offset = 0,
      };
   } else {
      layout->binding[0] = (struct hk_descriptor_set_binding_layout){
         .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .array_size = 1,
         .stride = stride,
         .offset = 0,
      };
      layout->binding[1] = (struct hk_descriptor_set_binding_layout){
         .type = VK_DESCRIPTOR_TYPE_SAMPLER,
         .array_size = 1,
         .stride = stride,
         .offset = stride,
      };
   }
   layout->non_variable_descriptor_buffer_size = bindings * stride;
   return layout;
}

static nir_shader *
build_shadow_gather(enum desc_family family, float ref, unsigned dest_bits)
{
   const bool separate =
      family == DESC_SEPARATE_SET || family == DESC_SEPARATE_PUSH;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "gen5 shadow gather border");

   nir_deref_instr *tex_deref;
   nir_deref_instr *samp_deref;
   if (!separate) {
      nir_variable *var = nir_variable_create(
         b.shader, nir_var_uniform,
         glsl_sampler_type(GLSL_SAMPLER_DIM_2D, true, false, GLSL_TYPE_FLOAT),
         "combined");
      var->data.descriptor_set = 0;
      var->data.binding = 0;
      tex_deref = nir_build_deref_var(&b, var);
      samp_deref = tex_deref;
   } else {
      nir_variable *tex_var = nir_variable_create(
         b.shader, nir_var_uniform,
         glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT), "tex");
      tex_var->data.descriptor_set = 0;
      tex_var->data.binding = 0;
      nir_variable *samp_var = nir_variable_create(
         b.shader, nir_var_uniform, glsl_bare_shadow_sampler_type(), "samp");
      samp_var->data.descriptor_set = 0;
      samp_var->data.binding = 1;
      tex_deref = nir_build_deref_var(&b, tex_var);
      samp_deref = nir_build_deref_var(&b, samp_var);
   }

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_tg4;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->is_shadow = true;
   tex->dest_type = dest_bits == 16 ? nir_type_float16 : nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &tex_deref->def);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &samp_deref->def);
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_vec2(&b, -0.05f, 0.5f));
   tex->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_comparator, nir_imm_float(&b, ref));
   nir_def_init(&tex->instr, &tex->def, 4, dest_bits);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *out = dest_bits == 16 ? nir_f2f32(&b, &tex->def) : &tex->def;
   nir_store_global(&b, out, nir_imm_int64(&b, 0x200000), .align_mul = 16);
   return b.shader;
}

static void
optimize(nir_shader *s)
{
   bool progress;
   do {
      progress = nir_opt_constant_folding(s);
      progress |= nir_opt_copy_prop(s);
      progress |= nir_opt_dead_cf(s);
      progress |= nir_opt_dce(s);
      progress |= nir_opt_algebraic(s);
   } while (progress);
}

static uint32_t
pack_gen10_tex_word0(enum agx_texture_type type)
{
   struct agx_texture_gen10_packed packed = {0};
   agx_pack(&packed, TEXTURE_GEN10, cfg) {
      cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.type = type;
      cfg.width = 4;
      cfg.height = 4;
      cfg.first_level = 0;
      cfg.last_level = 0;
   }
   return packed.opaque[0];
}

static void
evaluate_case(enum agx_chip chip, enum desc_family family,
              enum agx_texture_type tex_type, enum agx_compare_func func,
              float ref, float border, bool custom, unsigned dest_bits,
              bool interior_edges)
{
   struct hk_descriptor_set_layout *layout = make_layout(chip, family, 0);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(layout->descriptor_generation);

   nir_shader *s = build_shadow_gather(family, ref, dest_bits);
   CHECK(hk_nir_lower_descriptors(s, &vk_robustness_disabled, 1, layouts, false,
                                  false, 0, false, false, false, false, false,
                                  false, true));
   nir_validate_shader(s, "after hk_nir_lower_descriptors");

   /* Pack a real Gen5 sampler word for the target chip (M5 or G18G) and
    * populate the descriptor bytes exactly as hk_descriptor_set.c does. */
   struct hk_native_sampler_state state = {
      .minimum_lod = 0.0f,
      .maximum_lod = 15.0f,
      .maximum_anisotropy = 1,
      .magnify = AGX_FILTER_NEAREST,
      .minify = AGX_FILTER_NEAREST,
      .mip_filter = AGX_MIP_FILTER_NONE,
      .wrap_s = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_r = AGX_WRAP_CLAMP_TO_BORDER,
      .compare_func = func,
      .border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE,
      .reduction_mode = HK_SAMPLER_REDUCTION_WEIGHTED_AVERAGE,
      .compare_enable = true,
   };
   struct hk_native_sampler_descriptor samp_desc = {0};
   CHECK(hk_pack_native_sampler(chip, &samp_desc, &state));

   const uint32_t tex_word0 = pack_gen10_tex_word0(tex_type);
   const uint32_t samp_word1 = samp_desc.opaque[1];
   const uint16_t published_sampler_idx = hk_sampler_descriptor_index(layout, 3);
   const int16_t clamp_0_idx =
      custom ? (int16_t)hk_sampler_descriptor_index(layout, 4) : (int16_t)-1;
   const uint32_t border_u32 = fui(border);

   const uint64_t set0_base = 0x10000000ull;
   const uint32_t tex_base_off = layout->binding[0].offset;
   const uint32_t samp_base_off =
      layout->binding_count > 1 ? layout->binding[1].offset : tex_base_off;

   unsigned shadow_taps = 0, raw_one_taps = 0, raw_zero_taps = 0;
   const float ord_vals[4] = {0.25f, 0.75f, 0.5f, 0.125f};

   /* Verify no root descriptor / sampler_heap_address load is emitted (G18G has
    * +28 bias on sampler_index and must read the embedded Gen5 sampler word). */
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               CHECK(tex->op == nir_texop_tg4);
               nir_def *val;
               if (tex->is_shadow) {
                  ++shadow_taps;
                  if (dest_bits == 16) {
                     val = nir_vec4(&b, nir_imm_float16(&b, ord_vals[0]),
                                    nir_imm_float16(&b, ord_vals[1]),
                                    nir_imm_float16(&b, ord_vals[2]),
                                    nir_imm_float16(&b, ord_vals[3]));
                  } else {
                     val = nir_imm_vec4(&b, ord_vals[0], ord_vals[1],
                                        ord_vals[2], ord_vals[3]);
                  }
               } else {
                  CHECK(nir_tex_instr_src_index(tex, nir_tex_src_comparator) < 0);
                  const bool is_zero =
                     !!(tex->backend_flags & AGX_TEXTURE_FLAG_CLAMP_TO_0);
                  raw_zero_taps += is_zero;
                  raw_one_taps += !is_zero;
                  const float edge = is_zero ? 0.0f : 1.0f;
                  /* Mixed quad:
                   * - lane 0: interior NaN (0x7fc12345) or interior 0.0
                   * - lane 1: interior normal (0.4) or interior 1.0
                   * - lanes 2, 3: border (0.0 vs 1.0) */
                  nir_def *t0 = interior_edges ? nir_imm_float(&b, 0.0f)
                                               : nir_imm_int(&b, 0x7fc12345);
                  nir_def *t1 = interior_edges ? nir_imm_float(&b, 1.0f)
                                               : nir_imm_float(&b, 0.4f);
                  val = nir_vec4(&b, t0, t1, nir_imm_float(&b, edge),
                                 nir_imm_float(&b, edge));
               }
               nir_def_replace(&tex->def, val);
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               CHECK(intr->intrinsic != nir_intrinsic_load_root_agx);
               if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
                  CHECK(nir_intrinsic_desc_set(intr) == 0);
                  nir_def_replace(&intr->def, nir_imm_int64(&b, set0_base));
               }
            }
         }
      }
   }
   CHECK(shadow_taps == 1 && raw_one_taps == 1 && raw_zero_taps == 1);
   optimize(s);

   /* Substitute the descriptor memory loads with the packed Gen10 texture
    * word, Gen5 sampler word1, clamp-to-0 sentinel/index, and border[0]. */
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            uint64_t off = UINT64_MAX;
            if (intr->intrinsic == nir_intrinsic_load_global_constant_offset &&
                nir_src_is_const(intr->src[0]) &&
                nir_src_is_const(intr->src[1])) {
               CHECK(nir_src_as_uint(intr->src[0]) == set0_base);
               off = nir_src_as_uint(intr->src[1]);
            } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
                       nir_src_is_const(intr->src[0])) {
               uint64_t addr = nir_src_as_uint(intr->src[0]);
               CHECK(addr >= set0_base);
               off = addr - set0_base;
            } else {
               continue;
            }

            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_def *val = NULL;
            if (off == samp_base_off + native.clamp_0_sampler_index) {
               CHECK(intr->def.bit_size == 16);
               val = nir_imm_intN_t(&b, (uint16_t)clamp_0_idx, 16);
            } else if (off == samp_base_off + native.sampler_index) {
               CHECK(intr->def.bit_size == 16);
               val = nir_imm_intN_t(&b, published_sampler_idx, 16);
            } else if (off == samp_base_off + native.sampler + 4) {
               CHECK(intr->def.bit_size == 32);
               val = nir_imm_int(&b, samp_word1);
            } else if (off == samp_base_off + native.border) {
               CHECK(intr->def.bit_size == 32);
               val = nir_imm_int(&b, border_u32);
            } else if (off == tex_base_off + native.texture) {
               CHECK(intr->def.bit_size == 32);
               val = nir_imm_int(&b, tex_word0);
            }
            CHECK(val != NULL);
            nir_def_replace(&intr->def, val);
         }
      }
   }
   optimize(s);
   nir_validate_shader(s, "evaluated gen5 shadow gather border");

   const bool is_fixed = tex_type != AGX_TEXTURE_TYPE_FLOAT;
   const float expected_border =
      ref_compare(func, ref, border, is_fixed) ? 1.0f : 0.0f;

   unsigned stores = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_global)
               continue;
            CHECK(nir_src_is_const(intr->src[0]));
            /* Lanes 0 and 1 are interior (including interior NaN or 0/1) and
             * must preserve ordinary_tex's compare result untouched. */
            CHECK(nir_src_comp_as_float(intr->src[0], 0) == ord_vals[0]);
            CHECK(nir_src_comp_as_float(intr->src[0], 1) == ord_vals[1]);
            /* Lanes 2 and 3 are border and take expected_border iff custom. */
            CHECK(nir_src_comp_as_float(intr->src[0], 2) ==
                  (custom ? expected_border : ord_vals[2]));
            CHECK(nir_src_comp_as_float(intr->src[0], 3) ==
                  (custom ? expected_border : ord_vals[3]));
            ++stores;
         }
      }
   }
   CHECK(stores == 1);

   ralloc_free(s);
   free(layout);
}

static void
check_opt_out(enum agx_chip chip, unsigned generation_override,
              bool gen5_shadow_gather_border)
{
   struct hk_descriptor_set_layout *layout =
      make_layout(chip, DESC_COMBINED_SET, generation_override);
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   nir_shader *s = build_shadow_gather(DESC_COMBINED_SET, 0.5f, 32);

   CHECK(hk_nir_lower_descriptors(s, &vk_robustness_disabled, 1, layouts, false,
                                  false, 0, false, false, false, false, false,
                                  false, gen5_shadow_gather_border));
   nir_validate_shader(s, "opt-out shadow gather");

   unsigned tex_count = 0, if_count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
         if (prev && prev->type == nir_cf_node_if)
            ++if_count;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               ++tex_count;
         }
      }
   }
   CHECK(tex_count == 1 && if_count == 0);

   ralloc_free(s);
   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   /* Negative controls: disabled flag or non-Gen10 descriptors stay unlowered. */
   check_opt_out(AGX_CHIP_G17S, 0, false);
   check_opt_out(AGX_CHIP_G18G, 0, false);
   check_opt_out(AGX_CHIP_G17P, 9, true);

   const enum agx_chip chips[] = {
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
      AGX_CHIP_G18G,
   };
   const enum desc_family families[] = {
      DESC_COMBINED_SET,
      DESC_SEPARATE_SET,
      DESC_COMBINED_BUFFER,
      DESC_SEPARATE_PUSH,
   };
   const enum agx_compare_func funcs[] = {
      AGX_COMPARE_FUNC_NEVER,
      AGX_COMPARE_FUNC_LESS,
      AGX_COMPARE_FUNC_EQUAL,
      AGX_COMPARE_FUNC_LEQUAL,
      AGX_COMPARE_FUNC_GREATER,
      AGX_COMPARE_FUNC_NOT_EQUAL,
      AGX_COMPARE_FUNC_GEQUAL,
      AGX_COMPARE_FUNC_ALWAYS,
   };

   unsigned cases = 0;

   /* 1. Fixed-point depth (D16_UNORM -> AGX_TEXTURE_TYPE_UNORM) with Dref and
    * border inside and outside [0, 1] across all 8 compare ops and both M5 and
    * G18G chip families. */
   const float fixed_refs[] = {-0.5f, 0.0f, 0.5f, 1.0f, 1.25f, 1.5f};
   const float fixed_borders[] = {-0.25f, 0.0f, 0.25f, 0.75f, 1.0f, 1.5f};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned f = 0; f < ARRAY_SIZE(funcs); ++f) {
         for (unsigned r = 0; r < ARRAY_SIZE(fixed_refs); ++r) {
            for (unsigned b = 0; b < ARRAY_SIZE(fixed_borders); ++b) {
               evaluate_case(chips[c], families[(c + r + b) % ARRAY_SIZE(families)],
                             AGX_TEXTURE_TYPE_UNORM, funcs[f], fixed_refs[r],
                             fixed_borders[b], true, 32, false);
               ++cases;
            }
         }
      }
   }

   /* 2. Floating-point depth (D32_SFLOAT -> AGX_TEXTURE_TYPE_FLOAT) with
    * unclamped Dref/border, border NaN/Inf, interior NaN, and all 4 Vulkan
    * descriptor families on both G17S (M5) and G18G. */
   const float fp_refs[] = {-0.5f, 0.5f, 1.25f, 1.5f};
   const float fp_borders[] = {-0.25f, 0.25f, 1.25f, 1.5f, INFINITY, -INFINITY, NAN};
   for (unsigned f = 0; f < ARRAY_SIZE(funcs); ++f) {
      for (unsigned r = 0; r < ARRAY_SIZE(fp_refs); ++r) {
         for (unsigned b = 0; b < ARRAY_SIZE(fp_borders); ++b) {
            evaluate_case(AGX_CHIP_G17S, families[(r + b) % ARRAY_SIZE(families)],
                          AGX_TEXTURE_TYPE_FLOAT, funcs[f], fp_refs[r],
                          fp_borders[b], true, 32, false);
            evaluate_case(AGX_CHIP_G18G, families[(r + b + 1) % ARRAY_SIZE(families)],
                          AGX_TEXTURE_TYPE_FLOAT, funcs[f], fp_refs[r],
                          fp_borders[b], true, 32, false);
            cases += 2;
         }
      }
   }

   /* 3. Mixed interior 0.0/1.0 edge values, non-custom border bypass, and
    * 16-bit (RelaxedPrecision) destination across all descriptor families. */
   for (unsigned fam = 0; fam < ARRAY_SIZE(families); ++fam) {
      evaluate_case(AGX_CHIP_G17S, families[fam], AGX_TEXTURE_TYPE_UNORM,
                    AGX_COMPARE_FUNC_LEQUAL, 1.5f, 0.25f, true, 32, true);
      evaluate_case(AGX_CHIP_G18G, families[fam], AGX_TEXTURE_TYPE_FLOAT,
                    AGX_COMPARE_FUNC_EQUAL, 1.25f, 1.25f, true, 16, true);
      evaluate_case(AGX_CHIP_G17S, families[fam], AGX_TEXTURE_TYPE_UNORM,
                    AGX_COMPARE_FUNC_LESS, 0.5f, 0.75f, false, 32, false);
      evaluate_case(AGX_CHIP_G18G, families[fam], AGX_TEXTURE_TYPE_FLOAT,
                    AGX_COMPARE_FUNC_GREATER, 0.5f, 0.25f, false, 16, false);
      cases += 4;
   }

   printf("PASS: %u Gen5/Gen10 custom-border shadow gather cases (M5 + G18G, "
          "combined/separate/descriptor-buffer/push)\n", cases);
   glsl_type_singleton_decref();
   return 0;
}
