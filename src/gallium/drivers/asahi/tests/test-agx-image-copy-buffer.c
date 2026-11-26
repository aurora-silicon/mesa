/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Under the no-op DRM shim, check that pipe_context::image_copy_buffer
 * renders a pixel-buffer upload where the screen takes blit-based texture
 * transfers: one draw into a batch that reads the buffer, with a fragment
 * program the backend encodes and a render target the native descriptor
 * profile accepts. Covers a whole single-layer image, a box inside a larger
 * level, and three layers of an array in one layered draw, for 8-, 32-, 64-
 * and 128-bit texels. The bound state an application set is restored. This
 * test does not execute GPU work or compare rendered pixels.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/u_inlines.h"
#include "util/xmlconfig.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr) do { \
   if (!(expr)) { \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
      exit(1); \
   } \
} while (0)

/* The one active batch that reads bo, checked for a single accepted draw. */
static void
check_upload_batch(struct agx_context *ctx, struct agx_bo *bo)
{
   unsigned found = 0;
   unsigned idx;

   BITSET_FOREACH_SET(idx, ctx->batches.active, AGX_MAX_BATCHES) {
      struct agx_batch *batch = &ctx->batches.slots[idx];

      CHECK(!batch->native_rejected && !batch->draws_refused);
      if (agx_batch_uses_bo(batch, bo)) {
         CHECK(batch->draws == 1);
         found++;
      }
   }

   CHECK(found == 1);
}

static void
copy_case(struct pipe_screen *screen, struct pipe_context *pipe,
          enum pipe_format format, enum pipe_texture_target target,
          unsigned width, unsigned height, unsigned layers,
          const struct pipe_box *box)
{
   struct agx_context *ctx = agx_context(pipe);
   const unsigned bpp = util_format_get_blocksize(format);
   const unsigned stride = box->width * bpp;
   const struct pipe_resource templ = {
      .target = target,
      .format = format,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = layers,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *texture = screen->resource_create(screen, &templ);
   CHECK(texture);

   const unsigned size = stride * box->height * box->depth;
   struct pipe_resource *pbo = pipe_buffer_create(
      screen, PIPE_BIND_SAMPLER_VIEW, PIPE_USAGE_STREAM, size);
   CHECK(pbo);

   /* State the copy must leave as it found it. */
   const uint32_t app_constants[4] = {1, 2, 3, 4};
   struct pipe_constant_buffer app_cb = {
      .buffer_size = sizeof(app_constants),
      .user_buffer = app_constants,
   };
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, &app_cb);
   struct pipe_resource *app_ssbo_buf = pipe_buffer_create(
      screen, PIPE_BIND_SHADER_BUFFER, PIPE_USAGE_DEFAULT, 256);
   CHECK(app_ssbo_buf);
   struct pipe_shader_buffer app_ssbo = {
      .buffer = app_ssbo_buf,
      .buffer_offset = 64,
      .buffer_size = 128,
   };
   pipe->set_shader_buffers(pipe, MESA_SHADER_FRAGMENT, 0, 1, &app_ssbo, 1);

   const unsigned refused = ctx->skipped_no_program;
   pipe->image_copy_buffer(pipe, texture, pbo, 0, stride,
                           stride * box->height, 0, box);
   CHECK(ctx->skipped_no_program == refused);
   check_upload_batch(ctx, agx_resource(pbo)->bo);

   struct agx_stage *fs = &ctx->stage[MESA_SHADER_FRAGMENT];
   CHECK(fs->ssbo[0].buffer == app_ssbo_buf);
   CHECK(fs->ssbo[0].buffer_offset == 64 && fs->ssbo[0].buffer_size == 128);
   CHECK(fs->ssbo_writable_mask & BITFIELD_BIT(0));

   pipe->flush(pipe, NULL, 0);
   pipe->set_shader_buffers(pipe, MESA_SHADER_FRAGMENT, 0, 1, NULL, 0);
   pipe->set_constant_buffer(pipe, MESA_SHADER_FRAGMENT, 0, NULL);
   pipe_resource_reference(&app_ssbo_buf, NULL);
   pipe_resource_reference(&pbo, NULL);
   pipe_resource_reference(&texture, NULL);
}

int
main(void)
{
   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);

   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {.options = &options,
                                       .options_info = &info};
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);

   /* Elsewhere the copy takes the CPU maps, as st/mesa's fallback would. */
   if (!(screen->caps.texture_transfer_modes & PIPE_TEXTURE_TRANSFER_BLIT)) {
      printf("no blit-based texture transfers: nothing to check\n");
      screen->destroy(screen);
      return 0;
   }

   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   CHECK(pipe && pipe->image_copy_buffer);

   const struct pipe_box whole = {.width = 64, .height = 64, .depth = 1};
   const struct pipe_box inside = {
      .x = 8, .y = 16, .width = 24, .height = 20, .depth = 1};
   const struct pipe_box layers = {
      .z = 1, .width = 32, .height = 32, .depth = 3};

   copy_case(screen, pipe, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_TEXTURE_2D, 64,
             64, 1, &whole);
   copy_case(screen, pipe, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_TEXTURE_2D, 64,
             64, 1, &inside);
   copy_case(screen, pipe, PIPE_FORMAT_B8G8R8A8_SRGB, PIPE_TEXTURE_2D, 64,
             64, 1, &inside);
   copy_case(screen, pipe, PIPE_FORMAT_R8_UNORM, PIPE_TEXTURE_2D, 64, 64, 1,
             &inside);
   copy_case(screen, pipe, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_TEXTURE_2D,
             64, 64, 1, &whole);
   copy_case(screen, pipe, PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_TEXTURE_2D,
             64, 64, 1, &inside);
   copy_case(screen, pipe, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_TEXTURE_2D_ARRAY,
             32, 32, 4, &layers);

   pipe->destroy(pipe);
   screen->destroy(screen);
   printf("image_copy_buffer: every upload rendered\n");
   return 0;
}
