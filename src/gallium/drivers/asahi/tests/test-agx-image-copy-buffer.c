/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Verify HAL200/AGX3B image_copy_buffer format mapping, alignment/tiling
 * geometry planners, and live drm-shim compute dispatch + direct-draw UBO
 * fast-path encoding. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "asahi/layout/layout.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static void
check_block_buf2img_formats(void)
{
   /* Supported 8-bit 1/4-channel (UNORM8 PBE) and 32-bit 1-channel (R32_UINT
    * PBE) formats map to their canonical PBE view format. */
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_UNORM) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_UINT) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8_SNORM) ==
         PIPE_FORMAT_R8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_UNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SRGB) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_B8G8R8A8_UNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SNORM) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_SINT) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8B8A8_UINT) ==
         PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_FLOAT) ==
         PIPE_FORMAT_R32_UINT);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_SINT) ==
         PIPE_FORMAT_R32_UINT);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32_UINT) ==
         PIPE_FORMAT_R32_UINT);

   /* Formats without an AGX3B explicit tile-store encoding, depth/stencil,
    * block-compressed, 16-byte, and packed formats decline cleanly. */
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R8G8_UNORM) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16G16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R16G16B16A16_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32G32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_Z32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_S8_UINT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_DXT1_RGBA) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R32G32B32A32_FLOAT) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_R10G10B10A2_UNORM) ==
         PIPE_FORMAT_NONE);
   CHECK(agx_g17p_block_buf2img_format(PIPE_FORMAT_B5G6R5_UNORM) ==
         PIPE_FORMAT_NONE);
}

static void
check_alignment_and_tiled_plan(void)
{
   struct pipe_box full_2048 = {.x = 0, .y = 0, .z = 0, .width = 2048, .height = 2048, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&full_2048, 2048, 2048));

   struct pipe_box full_npot = {.x = 0, .y = 0, .z = 0, .width = 300, .height = 200, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&full_npot, 300, 200));

   struct pipe_box sub_aligned = {.x = 32, .y = 64, .z = 0, .width = 96, .height = 64, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&sub_aligned, 256, 256));

   struct pipe_box sub_edge = {.x = 256, .y = 128, .z = 0, .width = 44, .height = 72, .depth = 1};
   CHECK(agx_g17p_block_buf2img_aligned(&sub_edge, 300, 200));

   struct pipe_box sub_unaligned_x = {.x = 5, .y = 0, .z = 0, .width = 64, .height = 64, .depth = 1};
   CHECK(!agx_g17p_block_buf2img_aligned(&sub_unaligned_x, 256, 256));

   struct pipe_box sub_unaligned_w = {.x = 0, .y = 0, .z = 0, .width = 50, .height = 64, .depth = 1};
   CHECK(!agx_g17p_block_buf2img_aligned(&sub_unaligned_w, 256, 256));

   /* Uncompressed GPU-tiled 2048x2048 RGBA8 layout. */
   struct ail_layout layout = {
      .width_px = 2048,
      .height_px = 2048,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .compressed = false,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&layout);

   uint32_t groups[3] = {0};
   uint32_t per_thread = 0;
   CHECK(agx_g17p_tiled_buf2img_plan(&layout, 0, &full_2048, 0, 2048 * 4,
                                     2048 * 2048 * 4, groups, &per_thread));
   CHECK(per_thread == 4);
   CHECK(groups[0] > 0 && groups[1] > 0 && groups[2] > 0);

   /* Unaligned buffer offset for 4B format is rejected cleanly. */
   CHECK(!agx_g17p_tiled_buf2img_plan(&layout, 0, &full_2048, 2, 2048 * 4,
                                      2048 * 2048 * 4, groups, &per_thread));

   /* Compressed level declines the uncompressed tiled planner. */
   struct ail_layout comp = layout;
   comp.compressed = true;
   ail_make_miptree(&comp);
   CHECK(!agx_g17p_tiled_buf2img_plan(&comp, 0, &full_2048, 0, 2048 * 4,
                                      2048 * 2048 * 4, groups, &per_thread));
}

static nir_shader *
build_ubo_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "ubo-fast-vs");
   b.shader->info.num_ubos = 2;

   nir_def *vid = nir_load_vertex_id(&b);
   /* Bind at Gallium cb[1], matching OpenGL GL_UNIFORM_BUFFER binding 0. */
   nir_def *off = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 1),
                               nir_imm_int(&b, 0), .align_mul = 4, .range = 16);
   nir_def *px = nir_u2f32(&b, nir_iand_imm(&b, vid, 1));
   nir_def *py = nir_u2f32(&b, nir_ushr_imm(&b, vid, 1));
   nir_def *pos = nir_vec4(&b, nir_fadd(&b, px, nir_channel(&b, off, 0)),
                           nir_fadd(&b, py, nir_channel(&b, off, 1)),
                           nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f));
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   b.shader->info.outputs_written = VARYING_BIT_POS;
   return b.shader;
}

static nir_shader *
build_solid_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "solid-fs");
   nir_def *color = nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f);
   nir_store_output(&b, color, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   return b.shader;
}

static void
check_shim_paths(void)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {
      .options = &cache,
      .options_info = &info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen != NULL);
   struct agx_device *dev = agx_device(screen);
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx != NULL);
   struct agx_context *ctx = agx_context(pctx);

   /* 1. Verify image_copy_buffer is wired on HAL200 and dispatches compute. */
   if (agx_chip_native_agx3(dev->chip)) {
      CHECK(pctx->image_copy_buffer != NULL);

      struct pipe_resource tex_templ = {
         .target = PIPE_TEXTURE_2D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 256,
         .height0 = 256,
         .depth0 = 1,
         .array_size = 1,
         .last_level = 0,
         .bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET,
      };
      struct pipe_resource *tex = screen->resource_create(screen, &tex_templ);
      CHECK(tex != NULL);

      struct pipe_resource buf_templ = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = 256 * 256 * 4,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *pbo = screen->resource_create(screen, &buf_templ);
      CHECK(pbo != NULL);

      /* Full aligned upload */
      struct pipe_box full = {.width = 256, .height = 256, .depth = 1};
      pctx->image_copy_buffer(pctx, tex, pbo, 0, 256 * 4, 0, 0, &full);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      uint8_t *cdm_after_full = ctx->batch->cdm.current;
      CHECK(cdm_after_full > (uint8_t *)agx_bo_map(ctx->batch->cdm.bo));
      CHECK(agx_resource_valid(agx_resource(tex), 0));

      /* Partial unaligned sub-rect upload into valid compressed texture */
      struct pipe_box partial = {.x = 7, .y = 9, .width = 45, .height = 37, .depth = 1};
      pctx->image_copy_buffer(pctx, tex, pbo, 0, 45 * 4, 0, 0, &partial);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      CHECK(ctx->batch->cdm.current > cdm_after_full);

      /* Uncompressed GPU-tiled 3D texture takes the Morton compute path. */
      struct pipe_resource tex3d_templ = {
         .target = PIPE_TEXTURE_3D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 64,
         .height0 = 64,
         .depth0 = 2,
         .array_size = 1,
         .last_level = 0,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_resource *tex3d = screen->resource_create(screen, &tex3d_templ);
      CHECK(tex3d != NULL);
      uint8_t *cdm_before_3d = ctx->batch->cdm.current;
      struct pipe_box box3d = {.width = 64, .height = 64, .depth = 2};
      pctx->image_copy_buffer(pctx, tex3d, pbo, 0, 64 * 4, 64 * 64 * 4, 0, &box3d);
      CHECK(ctx->batch != NULL && agx_batch_is_compute(ctx->batch));
      CHECK(ctx->batch->cdm.current > cdm_before_3d);
      CHECK(agx_resource_valid(agx_resource(tex3d), 0));

      pctx->flush(pctx, NULL, 0);
      pipe_resource_reference(&tex3d, NULL);
      pipe_resource_reference(&pbo, NULL);
      pipe_resource_reference(&tex, NULL);

      /* 2. Verify HAL200 direct-draw UBO fast path engages after draw 0 and
       *    invalidates on non-UBO state changes. */
      struct pipe_resource rt_templ = {
         .target = PIPE_TEXTURE_2D,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .width0 = 64,
         .height0 = 64,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_RENDER_TARGET,
      };
      struct pipe_resource *rt = screen->resource_create(screen, &rt_templ);
      CHECK(rt != NULL);

      struct pipe_framebuffer_state fb = {
         .width = 64,
         .height = 64,
         .nr_cbufs = 1,
      };
      fb.cbufs[0].texture = rt;
      fb.cbufs[0].format = rt->format;
      pctx->set_framebuffer_state(pctx, &fb);

      struct pipe_viewport_state vp = {
         .scale = {32.0f, 32.0f, 0.5f},
         .translate = {32.0f, 32.0f, 0.5f},
      };
      pctx->set_viewport_states(pctx, 0, 1, &vp);

      struct pipe_blend_state blend_s = {0};
      blend_s.rt[0].colormask = PIPE_MASK_RGBA;
      void *blend = pctx->create_blend_state(pctx, &blend_s);
      pctx->bind_blend_state(pctx, blend);

      struct pipe_rasterizer_state rs_s = {.half_pixel_center = true};
      void *rs = pctx->create_rasterizer_state(pctx, &rs_s);
      pctx->bind_rasterizer_state(pctx, rs);

      struct pipe_depth_stencil_alpha_state zsa_s = {0};
      void *zsa = pctx->create_depth_stencil_alpha_state(pctx, &zsa_s);
      pctx->bind_depth_stencil_alpha_state(pctx, zsa);

      void *velems = pctx->create_vertex_elements_state(pctx, 0, NULL);
      pctx->bind_vertex_elements_state(pctx, velems);

      struct pipe_shader_state vs_s = {.type = PIPE_SHADER_IR_NIR, .ir.nir = build_ubo_vs()};
      void *vs = pctx->create_vs_state(pctx, &vs_s);
      pctx->bind_vs_state(pctx, vs);

      struct pipe_shader_state fs_s = {.type = PIPE_SHADER_IR_NIR, .ir.nir = build_solid_fs()};
      void *fs = pctx->create_fs_state(pctx, &fs_s);
      pctx->bind_fs_state(pctx, fs);

      struct pipe_resource ubo_templ = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = 64 * 256,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_CONSTANT_BUFFER,
      };
      struct pipe_resource *ubo = screen->resource_create(screen, &ubo_templ);
      CHECK(ubo != NULL);

      struct pipe_resource ib_templ = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = 64,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_INDEX_BUFFER,
      };
      struct pipe_resource *ib = screen->resource_create(screen, &ib_templ);
      CHECK(ib != NULL);

      struct pipe_draw_info info = {
         .mode = MESA_PRIM_TRIANGLES,
         .index_size = 2,
         .instance_count = 1,
         .index.resource = ib,
      };
      struct pipe_draw_start_count_bias draw = {.start = 0, .count = 6};

      for (unsigned i = 0; i < 32; ++i) {
         struct pipe_constant_buffer cb = {
            .buffer = ubo,
            .buffer_offset = i * 256,
            .buffer_size = 16,
         };
         pctx->set_constant_buffer(pctx, MESA_SHADER_VERTEX, 1, &cb);
         pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
         CHECK(ctx->batch != NULL);
         CHECK(ctx->batch->draws == i + 1);
         if (i >= 4) {
            CHECK(ctx->batch->g17p_fast_vs.valid);
            CHECK(ctx->batch->g17p_fast_vs.ubo_slot == 1);
            CHECK(ctx->batch->g17p_fast_vs.ubo_range_idx != UINT32_MAX);
            CHECK(ctx->batch->g17p_fast_vs.draw_body_valid);
         }
      }

      /* A fragment program change must leave the fast path: only the full
       * draw path re-records the fragment program the fast state matches. */
      struct pipe_shader_state fs2_s = {.type = PIPE_SHADER_IR_NIR,
                                        .ir.nir = build_solid_fs()};
      void *fs2 = pctx->create_fs_state(pctx, &fs2_s);
      pctx->bind_fs_state(pctx, fs2);
      pctx->draw_vbo(pctx, &info, 0, NULL, &draw, 1);
      CHECK(ctx->batch != NULL && ctx->batch->draws == 33);
      CHECK(ctx->batch->g17p_fast_vs.fs == ctx->fs);

      pctx->flush(pctx, NULL, 0);
      pctx->bind_fs_state(pctx, fs);
      pctx->delete_fs_state(pctx, fs2);
      pipe_resource_reference(&ib, NULL);
      pipe_resource_reference(&ubo, NULL);
      pctx->delete_vs_state(pctx, vs);
      pctx->delete_fs_state(pctx, fs);
      pctx->delete_vertex_elements_state(pctx, velems);
      pctx->delete_depth_stencil_alpha_state(pctx, zsa);
      pctx->delete_rasterizer_state(pctx, rs);
      pctx->delete_blend_state(pctx, blend);
      pipe_resource_reference(&rt, NULL);
   }

   pctx->destroy(pctx);
   screen->destroy(screen);
   close(fd);
}

int
main(int argc, char **argv)
{
   check_block_buf2img_formats();
   check_alignment_and_tiled_plan();

   if (argc > 1 && !strcmp(argv[1], "--unit-only"))
      return 0;

   check_shim_paths();
   return 0;
}
