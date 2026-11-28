/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Regression test for AGX3 native sampler pre-packing and generation-gated
 * Gallium texture descriptor caching under the no-op DRM shim.
 */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_bo.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,      \
                 #expr);                                                       \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static const void *
batch_data(struct agx_batch *batch, uint64_t address, size_t size)
{
   util_dynarray_foreach(&batch->pool.bos, struct agx_bo *, bo) {
      if (address >= (*bo)->va->addr &&
          address - (*bo)->va->addr <= (*bo)->size &&
          size <= (*bo)->size - (address - (*bo)->va->addr)) {
         const uint8_t *data = agx_bo_map(*bo);
         CHECK(data);
         return data + (address - (*bo)->va->addr);
      }
   }
   CHECK(!"uploaded descriptor table must live in batch pool");
   return NULL;
}

static void *
create_vs(struct pipe_context *pctx)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "desc cache VS");
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   return pctx->create_vs_state(pctx, &state);
}

static nir_def *
emit_tex_sample(nir_builder *b, unsigned idx, bool shadow, bool query_lod)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, shadow ? 2 : 1);
   tex->op = query_lod ? nir_texop_lod : nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_shadow = shadow;
   tex->is_new_style_shadow = shadow;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = idx;
   tex->sampler_index = idx;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                     nir_imm_vec2(b, 0.25f, 0.75f));
   if (shadow) {
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                        nir_imm_float(b, 0.5f));
   }
   nir_def_init(&tex->instr, &tex->def,
                query_lod ? 2 : (shadow ? 1 : 4), 32);
   nir_builder_instr_insert(b, &tex->instr);

   if (query_lod) {
      return nir_vec4(b, nir_channel(b, &tex->def, 0),
                      nir_channel(b, &tex->def, 1),
                      nir_imm_float(b, 0.0f), nir_imm_float(b, 1.0f));
   }
   if (shadow) {
      return nir_vec4(b, &tex->def, &tex->def, &tex->def,
                      nir_imm_float(b, 1.0f));
   }
   return &tex->def;
}

static void *
create_fs_samplers(struct pipe_context *pctx, bool query_lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      query_lod ? "desc cache FS query_lod" : "desc cache FS sample");

   nir_def *c0 = emit_tex_sample(&b, 0, false, query_lod);
   nir_def *c1 = emit_tex_sample(&b, 1, false, false);
   nir_def *c2 = emit_tex_sample(&b, 2, true, false);
   nir_def *sum = nir_fadd(&b, nir_fadd(&b, c0, c1), c2);

   nir_store_output(&b, sum, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.textures_used[0] = 0x7;
   b.shader->info.samplers_used[0] = 0x7;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   return pctx->create_fs_state(pctx, &state);
}

static void *
create_fs_single_tex(struct pipe_context *pctx, bool is_buffer, bool is_uint)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      is_buffer ? "desc cache FS buf" : "desc cache FS single");

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, is_buffer ? 1 : 2);
   tex->op = is_buffer ? nir_texop_txf : nir_texop_tex;
   tex->sampler_dim = is_buffer ? GLSL_SAMPLER_DIM_BUF : GLSL_SAMPLER_DIM_2D;
   tex->dest_type = is_uint ? nir_type_uint32 : nir_type_float32;
   tex->coord_components = is_buffer ? 1 : 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   if (is_buffer) {
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_int(&b, 0));
   } else {
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                        nir_imm_vec2(&b, 0.5f, 0.5f));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                        nir_imm_float(&b, 0.0f));
      tex->op = nir_texop_txl;
   }
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *out = is_uint ? nir_u2f32(&b, &tex->def) : &tex->def;
   nir_store_output(&b, out, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.textures_used[0] = 0x1;
   b.shader->info.samplers_used[0] = is_buffer ? 0x0 : 0x1;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   return pctx->create_fs_state(pctx, &state);
}

struct pipeline_state {
   void *vs, *rast, *blend, *depth, *ve;
   struct pipe_resource *rt;
};

static void
setup_pipeline(struct pipe_context *pctx, struct pipe_screen *screen,
               struct pipeline_state *s)
{
   s->vs = create_vs(pctx);
   struct pipe_rasterizer_state rast = {
      .depth_clip_near = true,
      .depth_clip_far = true,
      .point_size = 1,
      .line_width = 1,
   };
   struct pipe_blend_state blend = {.rt[0].colormask = 0xf};
   struct pipe_depth_stencil_alpha_state depth = {0};
   s->rast = pctx->create_rasterizer_state(pctx, &rast);
   s->blend = pctx->create_blend_state(pctx, &blend);
   s->depth = pctx->create_depth_stencil_alpha_state(pctx, &depth);
   s->ve = pctx->create_vertex_elements_state(pctx, 0, NULL);
   CHECK(s->vs && s->rast && s->blend && s->depth && s->ve);

   struct pipe_resource rt_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 16,
      .height0 = 16,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   s->rt = screen->resource_create(screen, &rt_tmpl);
   CHECK(s->rt);

   pctx->bind_vs_state(pctx, s->vs);
   pctx->bind_rasterizer_state(pctx, s->rast);
   pctx->bind_blend_state(pctx, s->blend);
   pctx->bind_depth_stencil_alpha_state(pctx, s->depth);
   pctx->bind_vertex_elements_state(pctx, s->ve);

   struct pipe_framebuffer_state fb = {
      .width = 16,
      .height = 16,
      .layers = 1,
      .nr_cbufs = 1,
      .cbufs[0] = {
         .texture = s->rt,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      },
   };
   struct pipe_viewport_state vp = {
      .scale = {8, 8, 0.5f},
      .translate = {8, 8, 0.5f},
   };
   pctx->set_framebuffer_state(pctx, &fb);
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   pctx->set_sample_mask(pctx, ~0u);
}

static void
teardown_pipeline(struct pipe_context *pctx, struct pipeline_state *s)
{
   struct pipe_framebuffer_state empty_fb = {0};
   pctx->set_framebuffer_state(pctx, &empty_fb);
   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->delete_vs_state(pctx, s->vs);
   pctx->delete_rasterizer_state(pctx, s->rast);
   pctx->delete_blend_state(pctx, s->blend);
   pctx->delete_depth_stencil_alpha_state(pctx, s->depth);
   pctx->delete_vertex_elements_state(pctx, s->ve);
   pipe_resource_reference(&s->rt, NULL);
}

static void
draw_tri(struct pipe_context *pctx)
{
   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   struct pipe_draw_start_count_bias draw = {.count = 3};
   pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
}

static struct pipe_resource *
create_tex2d(struct pipe_screen *screen, enum pipe_format fmt, unsigned w,
             unsigned h, unsigned bind)
{
   struct pipe_resource tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = fmt,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .bind = bind,
   };
   struct pipe_resource *res = screen->resource_create(screen, &tmpl);
   CHECK(res);
   return res;
}

static struct pipe_sampler_view *
create_view(struct pipe_context *pctx, struct pipe_resource *res,
            enum pipe_format fmt)
{
   struct pipe_sampler_view tmpl = {
      .target = res->target,
      .format = fmt,
      .swizzle_r = PIPE_SWIZZLE_X,
      .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z,
      .swizzle_a = PIPE_SWIZZLE_W,
   };
   if (res->target == PIPE_BUFFER) {
      tmpl.u.buf.offset = 0;
      tmpl.u.buf.size = res->width0;
   }
   struct pipe_sampler_view *view = pctx->create_sampler_view(pctx, res, &tmpl);
   CHECK(view);
   return view;
}

static void
check_sampler_cache(struct pipe_screen *screen, unsigned ctx_flags)
{
   struct agx_device *dev = agx_device(screen);
   const enum agx_chip chip = dev->chip;
   const bool native = agx_chip_native_agx3(chip);
   const bool hal300 = agx_chip_hal300_path(chip);
   const bool agx3 = native || hal300;
   const bool has_lod_bias = agx_chip_sampler_has_lod_bias(chip);
   const bool shadow_gather = agx_shadow_gather_custom_border(chip);

   struct pipe_context *pctx = screen->context_create(screen, NULL, ctx_flags);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);

   struct pipeline_state pipe_state;
   setup_pipeline(pctx, screen, &pipe_state);

   /* 0: standard linear sampler with LOD bias and anisotropy */
   struct pipe_sampler_state s0_desc = {
      .wrap_s = PIPE_TEX_WRAP_REPEAT,
      .wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_LINEAR,
      .mag_img_filter = PIPE_TEX_FILTER_LINEAR,
      .min_mip_filter = PIPE_TEX_MIPFILTER_LINEAR,
      .min_lod = 0.0f,
      .max_lod = 10.0f,
      .lod_bias = 1.5f,
      .max_anisotropy = 8,
      .seamless_cube_map = true,
   };
   /* 1: custom border non-compare sampler */
   struct pipe_sampler_state s1_desc = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_LINEAR,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST,
      .min_lod = 1.0f,
      .max_lod = 8.0f,
      .lod_bias = -0.75f,
      .border_color_format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .border_color.f = {0.25f, 0.5f, 0.75f, 1.0f},
   };
   /* 2: custom border shadow compare sampler */
   struct pipe_sampler_state s2_desc = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_BORDER,
      .min_img_filter = PIPE_TEX_FILTER_LINEAR,
      .mag_img_filter = PIPE_TEX_FILTER_LINEAR,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
      .min_lod = 0.0f,
      .max_lod = 14.0f,
      .lod_bias = 2.0f,
      .compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE,
      .compare_func = PIPE_FUNC_LESS,
      .border_color_format = PIPE_FORMAT_Z32_FLOAT,
      .border_color.f = {0.5f, 0.0f, 0.0f, 1.0f},
   };

   struct agx_sampler_state *s0 = pctx->create_sampler_state(pctx, &s0_desc);
   struct agx_sampler_state *s1 = pctx->create_sampler_state(pctx, &s1_desc);
   struct agx_sampler_state *s2 = pctx->create_sampler_state(pctx, &s2_desc);
   CHECK(s0 && s1 && s2);
   CHECK(!s0->uses_custom_border);
   CHECK(s1->uses_custom_border);
   CHECK(s2->uses_custom_border);

   if (!agx3) {
      CHECK(s0->native_desc_valid == 0);
      CHECK(s1->native_desc_valid == 0);
      CHECK(s2->native_desc_valid == 0);
   } else {
      const uint8_t expected_std =
         AGX_NATIVE_SAMPLER_DESC |
         (native && has_lod_bias ? AGX_NATIVE_SAMPLER_QUERY_LOD : 0);
      const uint8_t expected_custom =
         expected_std | AGX_NATIVE_SAMPLER_CLAMP |
         (native && shadow_gather ? AGX_NATIVE_SAMPLER_PROBE : 0);

      CHECK(s0->native_desc_valid == expected_std);
      CHECK(s1->native_desc_valid == expected_custom);
      CHECK(s2->native_desc_valid == expected_custom);
   }

   struct pipe_resource *tex0 = create_tex2d(
      screen, PIPE_FORMAT_R8G8B8A8_UNORM, 32, 32, PIPE_BIND_SAMPLER_VIEW);
   struct pipe_resource *tex2 = create_tex2d(
      screen, PIPE_FORMAT_Z32_FLOAT, 32, 32,
      PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_DEPTH_STENCIL);
   struct pipe_sampler_view *views[3] = {
      create_view(pctx, tex0, PIPE_FORMAT_R8G8B8A8_UNORM),
      create_view(pctx, tex0, PIPE_FORMAT_R8G8B8A8_UNORM),
      create_view(pctx, tex2, PIPE_FORMAT_Z32_FLOAT),
   };
   void *samplers[4] = {s0, s1, s2, NULL};

   void *fs_sample = create_fs_samplers(pctx, false);
   void *fs_lod = create_fs_samplers(pctx, true);
   CHECK(fs_sample && fs_lod);

   /* Draw with regular sampling + custom borders + shadow compare.
    * With ASAHI_MESA_DEBUG=verify, agx_upload_samplers() asserts every
    * pre-packed descriptor matches a fresh pack byte-for-byte. */
   pctx->bind_fs_state(pctx, fs_sample);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 3, 0, views);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 4, samplers);
   draw_tri(pctx);

   if (agx3) {
      struct agx_batch *batch = ctx->batch;
      CHECK(batch && batch->samplers[MESA_SHADER_FRAGMENT]);
      const size_t stride = agx_chip_sampler_heap_element_size(chip);
      CHECK(stride == sizeof(struct agx_gallium_sampler_descriptor));
      const uint8_t *table = batch_data(
         batch, batch->samplers[MESA_SHADER_FRAGMENT],
         stride * batch->sampler_count[MESA_SHADER_FRAGMENT]);

      CHECK(!memcmp(table + stride * 1, &s0->native_desc, stride));
      CHECK(!memcmp(table + stride * 2, &s1->native_desc, stride));
      CHECK(!memcmp(table + stride * 3, &s2->native_desc, stride));
      CHECK(!memcmp(table + stride * (AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + 1),
                    &s1->native_clamp_desc, stride));
      CHECK(!memcmp(table + stride * (AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + 2),
                    &s2->native_clamp_desc, stride));
      if (native && shadow_gather) {
         CHECK(!memcmp(
            table + stride * (AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE + 1),
            &s1->native_probe_desc, stride));
         CHECK(!memcmp(
            table + stride * (AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE + 2),
            &s2->native_probe_desc, stride));
      }
   }

   /* Draw with textureQueryLOD() to verify the query-LOD sampler bank. */
   const unsigned skipped_before = ctx->skipped_no_program;
   pctx->bind_fs_state(pctx, fs_lod);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 4, samplers);
   draw_tri(pctx);

   if (native && ctx->skipped_no_program == skipped_before && ctx->batch &&
       ctx->batch->samplers[MESA_SHADER_FRAGMENT]) {
      struct agx_batch *batch = ctx->batch;
      const size_t stride = agx_chip_sampler_heap_element_size(chip);
      const uint8_t *table = batch_data(
         batch, batch->samplers[MESA_SHADER_FRAGMENT],
         stride * batch->sampler_count[MESA_SHADER_FRAGMENT]);
      const struct agx_gallium_sampler_descriptor *expected_q0 =
         has_lod_bias ? &s0->native_query_lod_desc : &s0->native_desc;
      CHECK(!memcmp(table + stride * (AGX_GALLIUM_G17P_QUERY_LOD_BASE + 0),
                    expected_q0, stride));
      if (has_lod_bias) {
         /* Non-compare sampler s0 has non-zero lod_bias=1.5: query-LOD
          * descriptor must differ from ordinary descriptor (which keeps
          * lod_bias=0 for software bias summation). */
         CHECK(memcmp(&s0->native_query_lod_desc, &s0->native_desc, stride) !=
               0);

         /* For an unbound (!sampler) slot (samplers[3] == NULL), copying
          * ordinary into the query-LOD bank must match unpacking ordinary as
          * SAMPLER_HAL200 and repacking with lod_bias = 0. */
         const uint8_t *null_ord = table + stride * 4;
         const uint8_t *null_qlod =
            table + stride * (AGX_GALLIUM_G17P_QUERY_LOD_BASE + 3);
         CHECK(!memcmp(null_qlod, null_ord, stride));

         struct agx_gallium_sampler_descriptor repacked = {0};
         agx_unpack(NULL, null_ord, SAMPLER_HAL200, cfg);
         cfg.lod_bias = 0.0f;
         AGX_SAMPLER_HAL200_pack(repacked.opaque, &cfg);
         CHECK(!memcmp(&repacked, null_ord, stride));
      }
   }

   pctx->flush(pctx, NULL, 0);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, 3, NULL);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 4,
                             (void *[]){NULL, NULL, NULL, NULL});
   pctx->bind_fs_state(pctx, NULL);
   pctx->delete_fs_state(pctx, fs_sample);
   pctx->delete_fs_state(pctx, fs_lod);
   pctx->delete_sampler_state(pctx, s0);
   pctx->delete_sampler_state(pctx, s1);
   pctx->delete_sampler_state(pctx, s2);
   for (unsigned i = 0; i < 3; ++i)
      pipe_sampler_view_reference(&views[i], NULL);
   pipe_resource_reference(&tex0, NULL);
   pipe_resource_reference(&tex2, NULL);
   teardown_pipeline(pctx, &pipe_state);
   pctx->destroy(pctx);
}

static void
check_texture_cache(struct pipe_screen *screen)
{
   struct agx_device *dev = agx_device(screen);
   const enum agx_descriptor_generation desc_gen =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned desc_size = agx_gallium_texture_descriptor_size(desc_gen);
   CHECK(desc_size > 0);

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);

   struct pipeline_state pipe_state;
   setup_pipeline(pctx, screen, &pipe_state);

   void *fs_tex = create_fs_single_tex(pctx, false, false);
   void *fs_buf = create_fs_single_tex(pctx, true, false);
   void *fs_u32 = create_fs_single_tex(pctx, false, true);
   CHECK(fs_tex && fs_buf && fs_u32);

   struct pipe_sampler_state s_desc = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
   };
   void *sampler = pctx->create_sampler_state(pctx, &s_desc);
   CHECK(sampler);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 1, &sampler);

   /* 1. Basic uncompressed 2D texture creation, cache miss -> hit, and sentinel
    * check. PIPE_BIND_SHADER_IMAGE keeps it uncompressed so texture_map()
    * exercises CPU mapping + agx_shadow() directly rather than a staging blit.
    */
   struct pipe_resource *tex_res = create_tex2d(
      screen, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 64,
      PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE);
   struct agx_resource *rsrc = agx_resource(tex_res);
   CHECK(rsrc->desc_generation == 1);

   /* Seed data_valid so a subsequent discard map takes agx_shadow(). */
   {
      struct pipe_box box = {.x = 0, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1};
      struct pipe_transfer *xfer = NULL;
      void *map = pctx->texture_map(pctx, tex_res, 0, PIPE_MAP_WRITE, &box, &xfer);
      CHECK(map && xfer);
      memset(map, 0x5a, xfer->stride * 64);
      pctx->texture_unmap(pctx, xfer);
   }

   struct pipe_sampler_view *pview =
      create_view(pctx, tex_res, PIPE_FORMAT_R8G8B8A8_UNORM);
   struct agx_sampler_view *view = (struct agx_sampler_view *)pview;
   CHECK(view->desc_packed_gen == AGX_DESCRIPTOR_GENERATION_UNKNOWN);
   CHECK(view->desc_generation == 0);

   pctx->bind_fs_state(pctx, fs_tex);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &pview);
   draw_tri(pctx);

   CHECK(view->desc_packed_gen == desc_gen);
   CHECK(view->desc_generation == 1);
   struct agx_gallium_texture_descriptor packed_v1 = view->desc;

   /* Prove a cache hit actually skips agx_pack_texture(): inject a sentinel
    * byte with VERIFY temporarily cleared, re-dirty and draw, and verify the
    * sentinel survives untouched into the uploaded table. */
   const uint32_t saved_debug = dev->debug;
   dev->debug &= ~AGX_DBG_VERIFY;
   view->desc.opaque[0] ^= 0x12345678u;
   const uint32_t sentinel_word = view->desc.opaque[0];
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &pview);
   draw_tri(pctx);
   CHECK(view->desc.opaque[0] == sentinel_word);
   {
      struct agx_batch *batch = ctx->batch;
      CHECK(batch && batch->stage_uniforms[MESA_SHADER_FRAGMENT].texture_base);
      const struct agx_gallium_texture_descriptor *uploaded = batch_data(
         batch, batch->stage_uniforms[MESA_SHADER_FRAGMENT].texture_base,
         desc_size);
      CHECK(uploaded->opaque[0] == sentinel_word);
   }
   view->desc = packed_v1;
   dev->debug = saved_debug;

   /* Cache hit with AGX_DBG_VERIFY confirms byte-for-byte identity. */
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &pview);
   draw_tri(pctx);
   CHECK(view->desc_generation == 1);

   /* 2. Invalidation on agx_shadow() (BO swap while active batch reads it). */
   struct agx_bo *old_bo = rsrc->bo;
   const uint64_t gen_before_shadow = rsrc->desc_generation;
   {
      struct pipe_box box = {.x = 0, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1};
      struct pipe_transfer *xfer = NULL;
      void *map = pctx->texture_map(
         pctx, tex_res, 0, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE,
         &box, &xfer);
      CHECK(map && xfer);
      pctx->texture_unmap(pctx, xfer);
   }
   CHECK(rsrc->bo != old_bo);
   CHECK(rsrc->desc_generation == gen_before_shadow + 1);

   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &pview);
   draw_tri(pctx);
   CHECK(view->desc_generation == rsrc->desc_generation);
   CHECK(memcmp(&view->desc, &packed_v1, desc_size) != 0);

   /* 3. Invalidation on agx_flush_resource() (transition_resource to shared). */
   struct agx_gallium_texture_descriptor packed_v2 = view->desc;
   const uint64_t gen_before_flush = rsrc->desc_generation;
   pctx->flush_resource(pctx, tex_res);
   CHECK(rsrc->desc_generation == gen_before_flush + 1);

   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &pview);
   draw_tri(pctx);
   CHECK(view->desc_generation == rsrc->desc_generation);
   CHECK(memcmp(&view->desc, &packed_v2, desc_size) != 0);

   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
   pipe_sampler_view_reference(&pview, NULL);
   pipe_resource_reference(&tex_res, NULL);

   /* 4. Invalidation on agx_decompress() (compressed -> uncompressed). */
   struct pipe_resource *comp_res = create_tex2d(
      screen, PIPE_FORMAT_R8G8B8A8_UNORM, 256, 256,
      PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET);
   struct agx_resource *comp_rsrc = agx_resource(comp_res);
   if (comp_rsrc->layout.compressed) {
      struct pipe_sampler_view *cview =
         create_view(pctx, comp_res, PIPE_FORMAT_R8G8B8A8_UNORM);
      struct agx_sampler_view *agx_cview = (struct agx_sampler_view *)cview;

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &cview);
      draw_tri(pctx);
      CHECK(agx_cview->desc_generation == 1);
      struct agx_gallium_texture_descriptor comp_desc = agx_cview->desc;

      agx_decompress(ctx, comp_rsrc, "desc cache unit test");
      CHECK(!comp_rsrc->layout.compressed);
      CHECK(comp_rsrc->desc_generation == 2);

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &cview);
      draw_tri(pctx);
      CHECK(agx_cview->desc_generation == 2);
      CHECK(memcmp(&agx_cview->desc, &comp_desc, desc_size) != 0);

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
      pipe_sampler_view_reference(&cview, NULL);
   }
   pipe_resource_reference(&comp_res, NULL);

   /* 5. Buffer texture (PIPE_BUFFER) shadowing invalidation. */
   {
      struct pipe_resource buf_tmpl = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = 4096,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *buf_res = screen->resource_create(screen, &buf_tmpl);
      CHECK(buf_res);
      struct agx_resource *buf_rsrc = agx_resource(buf_res);
      CHECK(buf_rsrc->desc_generation == 1);

      struct pipe_box box = {.x = 0, .y = 0, .z = 0, .width = 4096, .height = 1, .depth = 1};
      struct pipe_transfer *xfer = NULL;
      void *map = pctx->buffer_map(pctx, buf_res, 0, PIPE_MAP_WRITE, &box, &xfer);
      CHECK(map && xfer);
      memset(map, 0x33, 4096);
      pctx->buffer_unmap(pctx, xfer);

      struct pipe_sampler_view *bview =
         create_view(pctx, buf_res, PIPE_FORMAT_R32G32B32A32_FLOAT);
      struct agx_sampler_view *agx_bview = (struct agx_sampler_view *)bview;

      pctx->bind_fs_state(pctx, fs_buf);
      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &bview);
      draw_tri(pctx);
      CHECK(agx_bview->desc_packed_gen == desc_gen);
      CHECK(agx_bview->desc_generation == 1);
      struct agx_gallium_texture_descriptor buf_desc_v1 = agx_bview->desc;

      struct agx_bo *old_buf_bo = buf_rsrc->bo;
      map = pctx->buffer_map(
         pctx, buf_res, 0, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE,
         &box, &xfer);
      CHECK(map && xfer);
      pctx->buffer_unmap(pctx, xfer);
      CHECK(buf_rsrc->bo != old_buf_bo);
      CHECK(buf_rsrc->desc_generation == 2);

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &bview);
      draw_tri(pctx);
      CHECK(agx_bview->desc_generation == 2);
      CHECK(memcmp(&agx_bview->desc, &buf_desc_v1, desc_size) != 0);

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
      pipe_sampler_view_reference(&bview, NULL);
      pipe_resource_reference(&buf_res, NULL);
   }

   /* 6. Separate stencil sampler view tracks separate_stencil->desc_generation. */
   {
      struct pipe_resource *zs_res = create_tex2d(
         screen, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, 32, 32,
         PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_DEPTH_STENCIL);
      struct agx_resource *zs_rsrc = agx_resource(zs_res);
      CHECK(zs_rsrc->separate_stencil != NULL);
      CHECK(zs_rsrc->desc_generation >= 1);
      CHECK(zs_rsrc->separate_stencil->desc_generation == 1);

      struct pipe_sampler_view *sview =
         create_view(pctx, zs_res, PIPE_FORMAT_X32_S8X24_UINT);
      struct agx_sampler_view *agx_sview = (struct agx_sampler_view *)sview;
      CHECK(agx_sview->rsrc == zs_rsrc->separate_stencil);

      pctx->bind_fs_state(pctx, fs_u32);
      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &sview);
      draw_tri(pctx);
      CHECK(agx_sview->desc_packed_gen == desc_gen);
      CHECK(agx_sview->desc_generation == 1);

      /* Re-bind hits cache under AGX_DBG_VERIFY. */
      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 1, 0, &sview);
      draw_tri(pctx);
      CHECK(agx_sview->desc_generation == 1);

      pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
      pipe_sampler_view_reference(&sview, NULL);
      pipe_resource_reference(&zs_res, NULL);
   }

   pctx->flush(pctx, NULL, 0);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 1,
                             (void *[]){NULL});
   pctx->bind_fs_state(pctx, NULL);
   pctx->delete_fs_state(pctx, fs_tex);
   pctx->delete_fs_state(pctx, fs_buf);
   pctx->delete_fs_state(pctx, fs_u32);
   pctx->delete_sampler_state(pctx, sampler);
   teardown_pipeline(pctx, &pipe_state);
   pctx->destroy(pctx);
}

int
main(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip);

   glsl_type_singleton_init_or_ref();

   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {
      .options = &options,
      .options_info = &info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   CHECK(agx_device(screen)->debug & AGX_DBG_VERIFY);

   check_sampler_cache(screen, 0);
   check_sampler_cache(screen, PIPE_CONTEXT_NO_LOD_BIAS);
   check_texture_cache(screen);

   screen->destroy(screen);
   glsl_type_singleton_decref();
   printf("%s: descriptor cache ok\n", chip);
   return 0;
}
