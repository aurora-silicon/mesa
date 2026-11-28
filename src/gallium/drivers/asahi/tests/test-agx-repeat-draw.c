/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* agx_draw_vbo_repeat(): on native AGX3 chips, a direct draw that follows a
 * recorded draw with only vertex constant buffers rebound is replayed as a
 * vertex state update and a draw body. Check that the replay
 *  - points the vertex program at a stage table holding the new binding,
 *  - emits the same commands as the full path for the same draw, apart from
 *    the vertex program address,
 *  - emits only the draw body when nothing changed, and
 *  - leaves draws with other state changes, or bound queries, to the full
 *    path.
 * On other chips nothing is recorded. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "asahi/lib/agx_vdm_state_core.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

#define UBO_STRIDE 256
#define UBO_SLOTS  32

/* A triangle from a vertex buffer, offset by a vec4 from constant buffer 0,
 * as in a draw loop that rebinds a per-draw uniform block range. */
static void *
create_vs(struct pipe_context *pctx)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "repeat-draw VS");
   b.shader->info.io_lowered = true;

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *rect = nir_load_ubo(&b, 4, 32, zero, zero, .align_mul = 16,
                                .align_offset = 0, .range = 16);
   nir_def *in = nir_load_input(&b, 2, 32, zero, .base = 0,
                                .io_semantics.location = VERT_ATTRIB_GENERIC0,
                                .io_semantics.num_slots = 1);
   nir_def *pos = nir_vec4(
      &b, nir_fadd(&b, nir_channel(&b, rect, 0), nir_channel(&b, in, 0)),
      nir_fadd(&b, nir_channel(&b, rect, 1), nir_channel(&b, in, 1)),
      nir_channel(&b, rect, 2), nir_imm_float(&b, 1.0f));
   nir_store_output(&b, pos, zero, .base = 0, .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   b.shader->info.inputs_read = VERT_BIT_GENERIC0;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   b.shader->info.num_ubos = 1;
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *vs = pctx->create_vs_state(pctx, &state);
   CHECK(vs);
   return vs;
}

static void *
create_fs(struct pipe_context *pctx)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "repeat-draw FS");
   b.shader->info.io_lowered = true;
   nir_store_output(&b, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));

   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   void *fs = pctx->create_fs_state(pctx, &state);
   CHECK(fs);
   return fs;
}

static void
bind_ubo(struct pipe_context *pctx, mesa_shader_stage stage,
         struct pipe_resource *ubo, unsigned slot)
{
   const struct pipe_constant_buffer cb = {
      .buffer = ubo,
      .buffer_offset = slot * UBO_STRIDE,
      .buffer_size = 16,
   };
   pctx->set_constant_buffer(pctx, stage, 0, &cb);
}

/* Draw, returning the VDM bytes the draw added to the current batch. */
static size_t
draw(struct pipe_context *pctx, uint8_t **start)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_batch *batch = agx_get_batch(ctx);
   uint8_t *before = batch->vdm.current;

   const struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   const struct pipe_draw_start_count_bias range = {.count = 3};
   pctx->draw_vbo(pctx, &info, 0, NULL, &range, 1);

   CHECK(ctx->batch == batch);
   if (start)
      *start = before;
   return batch->vdm.current - before;
}

/* The ranges the replay patched read the tables now current. */
static void
check_recorded_ranges(struct agx_batch *batch)
{
   CHECK(batch->repeat.valid && batch->repeat.program_ok);
   CHECK(batch->repeat.vdm_end == batch->vdm.current);

   bool stage_range = false;
   u_foreach_bit(i, batch->repeat.table_mask) {
      const unsigned table = batch->repeat.range_table[i];
      CHECK(batch->repeat.ranges[i].address ==
            batch->uniforms.tables[table] + batch->repeat.range_offset[i]);
      stage_range |= table == AGX_SYSVAL_STAGE(MESA_SHADER_VERTEX);
   }
   CHECK(stage_range);
}

static void
check_repeat(void)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription options[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, cache;
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {.options = &cache, .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);
   const bool native = agx_chip_native_agx3(agx_device(screen)->chip);

   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);

   const struct pipe_resource rt_templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 128,
      .height0 = 64,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *rt = screen->resource_create(screen, &rt_templ);
   CHECK(rt);
   struct pipe_framebuffer_state fb = {
      .width = 128,
      .height = 64,
      .nr_cbufs = 1,
   };
   fb.cbufs[0].texture = rt;
   fb.cbufs[0].format = rt->format;
   pctx->set_framebuffer_state(pctx, &fb);

   const struct pipe_rasterizer_state rs_desc = {
      .fill_front = PIPE_POLYGON_MODE_FILL,
      .fill_back = PIPE_POLYGON_MODE_FILL,
      .depth_clip_near = true,
      .depth_clip_far = true,
   };
   const struct pipe_blend_state bl_desc = {
      .rt[0] = {.colormask = PIPE_MASK_RGBA},
   };
   const struct pipe_depth_stencil_alpha_state zs_desc = {0};
   const struct pipe_viewport_state vp = {
      .scale = {64.0f, 32.0f, 0.5f},
      .translate = {64.0f, 32.0f, 0.5f},
   };
   void *rs = pctx->create_rasterizer_state(pctx, &rs_desc);
   void *bl = pctx->create_blend_state(pctx, &bl_desc);
   void *zs = pctx->create_depth_stencil_alpha_state(pctx, &zs_desc);
   const struct pipe_vertex_element velem = {
      .src_format = PIPE_FORMAT_R32G32_FLOAT,
      .src_stride = 2 * sizeof(float),
   };
   void *ve = pctx->create_vertex_elements_state(pctx, 1, &velem);
   static const float verts[6] = {0, 0, 1, 0, 0, 1};
   struct pipe_resource *vb = pipe_buffer_create(
      screen, PIPE_BIND_VERTEX_BUFFER, PIPE_USAGE_DEFAULT, sizeof(verts));
   CHECK(vb);
   pipe_buffer_write(pctx, vb, 0, sizeof(verts), verts);
   const struct pipe_vertex_buffer vbuf = {.buffer.resource = vb};
   pctx->set_vertex_buffers(pctx, 1, &vbuf);
   void *vs = create_vs(pctx);
   void *fs = create_fs(pctx);
   pctx->bind_rasterizer_state(pctx, rs);
   pctx->bind_blend_state(pctx, bl);
   pctx->bind_depth_stencil_alpha_state(pctx, zs);
   pctx->bind_vertex_elements_state(pctx, ve);
   pctx->bind_vs_state(pctx, vs);
   pctx->bind_fs_state(pctx, fs);
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   pctx->set_sample_mask(pctx, 0xf);

   struct pipe_resource *ubo =
      pipe_buffer_create(screen, PIPE_BIND_CONSTANT_BUFFER, PIPE_USAGE_DEFAULT,
                         UBO_SLOTS * UBO_STRIDE);
   CHECK(ubo);
   const uint64_t ubo_va = agx_resource(ubo)->bo->va->addr;

   /* The first draws generate vertex program templates until one has a
    * patch plan that fits the moving ranges. The first draw patched from a
    * plan is recorded. */
   struct agx_batch *batch = agx_get_batch(ctx);
   unsigned warmup = 0;
   do {
      bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, warmup);
      draw(pctx, NULL);
   } while (!batch->repeat.valid && ++warmup < 8);
   if (batch->repeat.valid != native) {
      fprintf(stderr, "recorded %u, program_ok %u, draw_id %u, base_param %u, "
              "bind tail %u/%u\n", batch->repeat.valid,
              batch->repeat.program_ok, ctx->vs->b.info.uses_draw_id,
              ctx->linked.vs->uses_base_param, batch->g17p_bind_tail_latched,
              batch->g17p_have_latched_vs);
   }
   CHECK(batch->repeat.valid == native);

   if (!native) {
      for (unsigned k = 8; k < 16; ++k) {
         bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, k);
         draw(pctx, NULL);
         CHECK(!batch->repeat.valid);
      }
      goto out;
   }

   /* Replayed draws: a state update and a draw body each, with the new
    * binding in the stage table the vertex program reads. */
   size_t replay_len = 0;
   uint8_t replay[256];
   for (unsigned k = warmup + 1; k < warmup + 11; ++k) {
      const unsigned draws = batch->draws;
      bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, k);
      uint8_t *start;
      const size_t len = draw(pctx, &start);
      CHECK(batch->draws == draws + 1);
      CHECK(len > 0 && len <= sizeof(replay));
      CHECK(!replay_len || len == replay_len);
      replay_len = len;
      memcpy(replay, start, len);
      CHECK(batch->stage_uniforms[MESA_SHADER_VERTEX].ubo_base[0] ==
            ubo_va + k * UBO_STRIDE);
      CHECK(batch->stage_uniforms[MESA_SHADER_VERTEX].ubo_size[0] == 16);
      check_recorded_ranges(batch);
   }

   /* The full path, for the same kind of draw, emits the same commands but
    * for the vertex program address. */
   batch->repeat.valid = false;
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 20);
   uint8_t *full;
   const size_t full_len = draw(pctx, &full);
   CHECK(full_len == replay_len);
   const size_t pair_start = 8;
   const size_t pair_end = pair_start + AGX_VDM_STATE_VERTEX_PIPELINE_PAIR_LENGTH;
   CHECK(!memcmp(full, replay, pair_start));
   CHECK(!memcmp(full + pair_end, replay + pair_end, full_len - pair_end));
   CHECK(memcmp(full + pair_start, replay + pair_start, pair_end - pair_start));
   check_recorded_ranges(batch);

   /* Nothing changed: the draw body alone. */
   const size_t body_len = draw(pctx, NULL);
   CHECK(body_len > 0 && body_len < replay_len);
   CHECK(batch->repeat.valid);
   CHECK(draw(pctx, NULL) == body_len);

   /* A fragment constant buffer change takes the full path, which also
    * emits the bind tail, and records the draw again. */
   bind_ubo(pctx, MESA_SHADER_FRAGMENT, ubo, 3);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 21);
   CHECK(draw(pctx, NULL) > replay_len);
   CHECK(batch->repeat.valid);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 22);
   CHECK(draw(pctx, NULL) == replay_len);

   /* A draw counted by a query is not recorded, nor replayed. */
   struct pipe_query *oq =
      pctx->create_query(pctx, PIPE_QUERY_OCCLUSION_COUNTER, 0);
   CHECK(oq && pctx->begin_query(pctx, oq));
   batch = agx_get_batch(ctx);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 23);
   draw(pctx, NULL);
   CHECK(!batch->repeat.valid);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 24);
   draw(pctx, NULL);
   CHECK(!batch->repeat.valid);
   CHECK(pctx->end_query(pctx, oq));
   batch = agx_get_batch(ctx);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 25);
   draw(pctx, NULL);
   CHECK(batch->repeat.valid);
   bind_ubo(pctx, MESA_SHADER_VERTEX, ubo, 26);
   CHECK(draw(pctx, NULL) == replay_len);
   check_recorded_ranges(batch);
   pctx->destroy_query(pctx, oq);

   /* A new batch starts without a recorded draw. */
   pctx->flush(pctx, NULL, 0);
   batch = agx_get_batch(ctx);
   CHECK(!batch->repeat.valid);

out:
   pctx->flush(pctx, NULL, 0);
   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->set_constant_buffer(pctx, MESA_SHADER_VERTEX, 0, NULL);
   pctx->set_constant_buffer(pctx, MESA_SHADER_FRAGMENT, 0, NULL);
   pctx->delete_vs_state(pctx, vs);
   pctx->delete_fs_state(pctx, fs);
   pctx->delete_vertex_elements_state(pctx, ve);
   pctx->delete_rasterizer_state(pctx, rs);
   pctx->delete_blend_state(pctx, bl);
   pctx->delete_depth_stencil_alpha_state(pctx, zs);
   pctx->set_vertex_buffers(pctx, 0, NULL);
   pipe_resource_reference(&vb, NULL);
   pipe_resource_reference(&ubo, NULL);
   pipe_resource_reference(&rt, NULL);
   pctx->destroy(pctx);
   screen->destroy(screen);
   driDestroyOptionInfo(&info);
   close(fd);
}

int
main(void)
{
   check_repeat();
   printf("repeat-draw: ok\n");
   return 0;
}
