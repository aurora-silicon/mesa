/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* The fused 3x3 Sobel + Canny NMS kernel (hk_nir_build_sobel_nms()) against
 * the tiled Sobel and the tiled NMS it replaces, run on the CPU NIR evaluator:
 * every byte of dir, mag and the NMS output must match. Then the submission
 * pairing rule (hk_compute_pair_canny_match()): the pair GB6's Canny records
 * matches, and each broken precondition keeps the two dispatches apart. */

#include "hk_2d_stencil.h"
#include "hk_compute_opt.h"
#include "hk_compute_pair_interval.h"

#include "compiler/glsl_types.h"
#include "nir_builder.h"

#include "hk_nir_eval_test.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const nir_shader_compiler_options test_options = {
   .has_cs_global_id = true,
};

/* Push layout of the matched shaders: width, height, stride. */
enum { PC_W = 0, PC_H = 4, PC_STRIDE = 8 };

static uint32_t
lut_of(const int off[8])
{
   uint32_t lut = 0;
   for (unsigned d = 0; d < 8; ++d)
      lut |= (uint32_t)(off[d] + 1) << (2 * d);
   return lut;
}

/* GB6 6.7.1's Canny NMS direction tables and thresholds. */
static const int gb6_xp[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int gb6_yp[8] = {0, -1, -1, -1, 0, 1, 1, 1};

static uint32_t
f2u(float f)
{
   uint32_t u;
   memcpy(&u, &f, sizeof(u));
   return u;
}

static struct hk_2d_stencil_plan
sobel_plan(void)
{
   struct hk_2d_stencil_plan plan = {
      .kind = HK_2D_STENCIL_SOBEL_3X3_U8,
      .tile_w = 128,
      .tile_h = 8,
      .orig_wg_size = {32, 1, 1},
      .width_pc_offset_B = PC_W,
      .height_pc_offset_B = PC_H,
      .stride_pc_offset_B = PC_STRIDE,
   };
   for (unsigned i = 0; i < 3; ++i)
      plan.bufs[i] = (struct hk_2d_stencil_binding){.set = 0, .binding = i};
   return plan;
}

static struct hk_compute_opt_plan
nms_plan(uint32_t dx_lut, uint32_t dy_lut, float lo, float hi)
{
   struct hk_compute_opt_plan plan = {
      .kind = HK_COMPUTE_OPT_CANNY_NMS_U8,
      .tile_w = 256,
      .tile_h = 4,
      .orig_wg_size = {32, 1, 1},
      .pc_off = {PC_W, PC_H},
      .params = {dx_lut, dy_lut, f2u(lo), f2u(hi)},
   };
   for (unsigned i = 0; i < 3; ++i)
      plan.bufs[i] = (struct hk_compute_opt_binding){.set = 0, .binding = i};
   return plan;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint32_t
rnd(void)
{
   rng ^= rng << 13;
   rng ^= rng >> 7;
   rng ^= rng << 17;
   return (uint32_t)(rng >> 24);
}

static uint8_t
pattern_byte(int pattern, int x, int y)
{
   switch (pattern) {
   case 0: return rnd();                                       /* noise */
   case 1: return (uint8_t)((x * 3 + y * 2) / 4 + (rnd() & 7)); /* ramp */
   case 2: return ((x / 7 + y / 5) & 1) ? 230 : (rnd() & 15);   /* blocks */
   case 3: return (rnd() & 1) ? 255 : 0;                        /* extremes */
   default: return 77;                                          /* flat */
   }
}

struct canny_push {
   int32_t w, h, stride;
   uint32_t pad[(HK_SOBEL_NMS_PUSH_OFFSET_B - 12) / 4];
   struct hk_sobel_nms_push nms;
};

static unsigned cases_run;

static bool
cf_has_loop(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_has_loop(&nif->then_list) || cf_has_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

/* The evaluator takes workgroup memory derefs only: no function locals. */
static void
prepare(nir_shader *s)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_lower_vars_to_ssa);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_function_temp, NULL);
   nir_foreach_function_impl(impl, s)
      HK_EVAL_CHECK(exec_list_is_empty(&impl->locals));
}

/* One image through Sobel then NMS, and through the fused kernel; every
 * output byte must agree. */
static void
check_case(nir_shader *sobel, nir_shader *nms, nir_shader *fused,
           const struct hk_compute_opt_plan *np, int w, int h, int stride,
           int pattern)
{
   const size_t src_size = (size_t)(h - 1) * stride + w;
   const size_t px = (size_t)w * h;
   uint8_t *src = malloc(src_size);
   uint32_t *dir_a = malloc(px * 4), *mag_a = malloc(px * 4);
   uint32_t *dir_b = malloc(px * 4), *mag_b = malloc(px * 4);
   uint8_t *dst_a = malloc(px), *dst_b = malloc(px);
   HK_EVAL_CHECK(src && dir_a && mag_a && dir_b && mag_b && dst_a && dst_b);
   for (size_t i = 0; i < src_size; ++i)
      src[i] = pattern_byte(pattern, (int)(i % stride), (int)(i / stride));
   /* Bytes neither kernel writes must stay as they were. */
   memset(dir_a, 0x5a, px * 4);
   memset(mag_a, 0x5a, px * 4);
   memset(dir_b, 0x5a, px * 4);
   memset(mag_b, 0x5a, px * 4);
   memset(dst_a, 0xcd, px);
   memset(dst_b, 0xcd, px);

   struct canny_push push = {.w = w, .h = h, .stride = stride};
   const uint32_t sobel_grid[3] = {(w + 127) / 128, (h + 7) / 8, 1};
   struct hk_eval_ssbo s_bufs[3] = {
      {0, 0, src, src_size, 0},
      {0, 1, dir_a, px * 4, 0},
      {0, 2, mag_a, px * 4, 0},
   };
   hk_eval_dispatch_wg(sobel, sobel_grid, s_bufs, 3, &push, sizeof(push));

   const uint32_t nms_grid[3] = {(w + 255) / 256, (h + 3) / 4, 1};
   struct hk_eval_ssbo n_bufs[3] = {
      {0, 0, dir_a, px * 4, 0},
      {0, 1, mag_a, px * 4, 0},
      {0, 2, dst_a, px, 0},
   };
   hk_eval_dispatch_wg(nms, nms_grid, n_bufs, 3, &push, sizeof(push));

   /* The NMS output goes through a global address: SSBO 3 of the evaluator. */
   const uint64_t dst_va = hk_eval_make_vaddr(3, 0);
   push.nms = (struct hk_sobel_nms_push){
      .dst_va_lo = (uint32_t)dst_va,
      .dst_va_hi = (uint32_t)(dst_va >> 32),
      .dx_lut = np->params[0],
      .dy_lut = np->params[1],
      .low_bits = np->params[2],
      .high_bits = np->params[3],
   };
   struct hk_eval_ssbo f_bufs[4] = {
      {0, 0, src, src_size, 0},
      {0, 1, dir_b, px * 4, 0},
      {0, 2, mag_b, px * 4, 0},
      {0, 3, dst_b, px, 0},
   };
   hk_eval_dispatch_wg(fused, sobel_grid, f_bufs, 4, &push, sizeof(push));

   if (memcmp(dir_a, dir_b, px * 4) || memcmp(mag_a, mag_b, px * 4) ||
       memcmp(dst_a, dst_b, px)) {
      fprintf(stderr, "mismatch %dx%d stride %d pattern %d\n", w, h, stride,
              pattern);
      for (size_t i = 0; i < px; ++i) {
         if (dir_a[i] != dir_b[i] || mag_a[i] != mag_b[i] ||
             dst_a[i] != dst_b[i]) {
            fprintf(stderr, "  first at (%zu,%zu): dir %x/%x mag %x/%x nms %u/%u\n",
                    i % w, i / w, dir_a[i], dir_b[i], mag_a[i], mag_b[i],
                    dst_a[i], dst_b[i]);
            break;
         }
      }
      abort();
   }

   /* The reference must not be trivial: noise always has edges. */
   if ((pattern == 0 || pattern == 3) && w >= 8 && h >= 8) {
      bool any = false;
      for (size_t i = 0; i < px && !any; ++i)
         any = dst_a[i] != 0;
      if (!any) {
         size_t nz_mag = 0, nz_dir = 0, cd = 0;
         float mx = 0;
         for (size_t i = 0; i < px; ++i) {
            float f;
            memcpy(&f, &mag_a[i], 4);
            nz_mag += mag_a[i] != 0 && mag_a[i] != 0x5a5a5a5au;
            nz_dir += dir_a[i] != 0x5a5a5a5au;
            cd += dst_a[i] == 0xcd;
            if (mag_a[i] != 0x5a5a5a5au && f > mx)
               mx = f;
         }
         fprintf(stderr, "no edge %dx%d s%d p%d: mag nonzero %zu max %g, dir written %zu, nms untouched %zu, src[0..3] %u %u %u %u mag[w+1] %08x\n",
                 w, h, stride, pattern, nz_mag, mx, nz_dir, cd, src[0], src[1], src[2], src[3], mag_a[w + 1]);
      }
      HK_EVAL_CHECK(any);
   }

   cases_run++;
   free(src);
   free(dir_a);
   free(mag_a);
   free(dir_b);
   free(mag_b);
   free(dst_a);
   free(dst_b);
}

static void
test_fused_kernel_matches(void)
{
   nir_shader *orig = nir_shader_create(NULL, MESA_SHADER_COMPUTE, &test_options);
   const struct hk_2d_stencil_plan sp = sobel_plan();
   HK_EVAL_CHECK(hk_2d_stencil_sobel_nms_supported(&sp));

   nir_shader *sobel = hk_nir_build_tiled_2d_stencil(
      orig, orig, &sp, nir_address_format_32bit_index_offset);
   nir_shader *fused =
      hk_nir_build_sobel_nms(orig, orig, &sp, nir_address_format_32bit_index_offset);
   HK_EVAL_CHECK(sobel && fused);
   prepare(sobel);
   prepare(fused);
   /* Loop-free, so the queue's per-run loop budget never limits fusion. */
   HK_EVAL_CHECK(!cf_has_loop(&nir_shader_get_entrypoint(fused)->body));
   HK_EVAL_CHECK(fused->info.workgroup_size[0] == sobel->info.workgroup_size[0] &&
                 fused->info.workgroup_size[1] == sobel->info.workgroup_size[1] &&
                 fused->info.workgroup_size[2] == sobel->info.workgroup_size[2]);

   /* GB6's tables and thresholds, a transposed table pair and other
    * thresholds. */
   const struct hk_compute_opt_plan nps[] = {
      nms_plan(lut_of(gb6_xp), lut_of(gb6_yp), 11.0f, 33.0f),
      nms_plan(lut_of(gb6_yp), lut_of(gb6_xp), 0.0f, 400.5f),
      nms_plan(lut_of(gb6_xp), lut_of(gb6_yp), 100.0f, 100.0f),
   };

   static const int sizes[][3] = {
      {4, 1, 4},     {4, 4, 4},       {8, 3, 8},      {8, 8, 9},
      {12, 5, 12},   {16, 16, 20},    {124, 9, 124},  {128, 8, 128},
      {132, 10, 133}, {136, 17, 136}, {256, 4, 259},  {260, 9, 264},
      {132, 24, 132}, {388, 13, 391},
      /* Widths that are not a multiple of 4 (GB6's Edge image is 5959 x
       * 3952, stride 6144). */
      {5, 5, 5},     {13, 7, 13},     {101, 11, 104}, {131, 9, 136},
      {259, 17, 272},
   };
   for (unsigned n = 0; n < ARRAY_SIZE(nps); ++n) {
      nir_shader *nms = hk_nir_build_compute_opt(
         orig, orig, &nps[n], nir_address_format_32bit_index_offset);
      HK_EVAL_CHECK(nms);
      prepare(nms);
      for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
         for (int pat = 0; pat < 5; ++pat) {
            if (n && pat > 1)
               continue;
            check_case(sobel, nms, fused, &nps[n], sizes[i][0], sizes[i][1],
                       sizes[i][2], pat);
         }
      }
   }
   ralloc_free(orig);
}

/* --- Pairing rule ------------------------------------------------------- */

enum { W = 64, H = 32, STRIDE = 72 };

static struct hk_shader *const fake_fused = (struct hk_shader *)(uintptr_t)0x1000;

static struct hk_storage_backing
backing(uint32_t handle, uint64_t bo_off, uint64_t va, uint32_t size)
{
   return (struct hk_storage_backing){
      .address = va, .offset = bo_off, .size = size, .handle = handle};
}

/* A Sobel and its NMS as GB6 records them: src, dir, mag and dst in four
 * BOs. */
static void
make_pair(struct hk_compute_pair_dispatch *s, struct hk_compute_pair_dispatch *n)
{
   const uint32_t px = W * H, src = (H - 1) * STRIDE + W;
   memset(s, 0, sizeof(*s));
   memset(n, 0, sizeof(*n));
   s->info.role = HK_COMPUTE_PAIR_SOBEL3X3;
   s->root_va = 0x7000000;
   s->extra.canny = (struct hk_compute_pair_canny){
      .width = W, .height = H, .stride = STRIDE, .fused = fake_fused};
   s->bindings[0] = backing(1, 0, 0x100000, src);
   s->bindings[1] = backing(2, 256, 0x200100, px * 4);
   s->bindings[2] = backing(3, 0, 0x300000, px * 4);

   n->info.role = HK_COMPUTE_PAIR_CANNY_NMS;
   n->extra.canny = (struct hk_compute_pair_canny){
      .width = W, .height = H,
      .params = {lut_of(gb6_xp), lut_of(gb6_yp), f2u(11.0f), f2u(33.0f)}};
   n->bindings[0] = s->bindings[1];
   n->bindings[1] = s->bindings[2];
   n->bindings[2] = backing(4, 0, 0x400000, px);
}

static void
test_pairing_rule(void)
{
   struct hk_compute_pair_dispatch s, n;
   struct hk_sobel_nms_push push;

   make_pair(&s, &n);
   HK_EVAL_CHECK(hk_compute_pair_canny_match(&s, &n, &push));
   HK_EVAL_CHECK(push.dst_va_lo == 0x400000 && push.dst_va_hi == 0 &&
                 push.dx_lut == lut_of(gb6_xp) && push.dy_lut == lut_of(gb6_yp) &&
                 push.low_bits == f2u(11.0f) && push.high_bits == f2u(33.0f));

   /* Reversed order, other roles, no fused shader, no root. */
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&n, &s, NULL));
   make_pair(&s, &n);
   n.info.role = HK_COMPUTE_PAIR_CLAMP;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   s.info.role = HK_COMPUTE_PAIR_GEMM;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   s.extra.canny.fused = NULL;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   s.root_va = 0;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));

   /* Image size: the NMS must cover the Sobel's image exactly. */
   make_pair(&s, &n);
   n.extra.canny.width = W - 4;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.extra.canny.height = H + 1;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   /* Any width, and an output at any byte (GB6's Edge image is 5959 wide);
    * the kernel picks whole-word stores where they are aligned. */
   make_pair(&s, &n);
   s.extra.canny.width = n.extra.canny.width = W - 1;
   HK_EVAL_CHECK(hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.bindings[2].address += 1;
   HK_EVAL_CHECK(hk_compute_pair_canny_match(&s, &n, &push) &&
                 push.dst_va_lo == 0x400001);
   make_pair(&s, &n);
   s.extra.canny.stride = W - 1;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));

   /* The NMS must read what the Sobel wrote: the same dir and mag. */
   make_pair(&s, &n);
   n.bindings[0] = s.bindings[2];
   n.bindings[1] = s.bindings[1];
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.bindings[0].offset += 4;
   n.bindings[0].address += 4;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.bindings[1].handle = 9;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.bindings[1].address += 0x1000;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));

   /* Buffers too small for the image. */
   make_pair(&s, &n);
   s.bindings[0].size -= 1;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.bindings[2].size -= 1;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   s.bindings[2].size -= 4;
   n.bindings[1].size -= 4;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));

   /* The NMS output may not alias the Sobel source, dir or mag: in the same
    * BO, or at an overlapping address. */
   for (unsigned b = 0; b < 3; ++b) {
      make_pair(&s, &n);
      n.bindings[2].handle = s.bindings[b].handle;
      n.bindings[2].offset = s.bindings[b].offset + 4;
      HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
      make_pair(&s, &n);
      n.bindings[2].address = s.bindings[b].address + 8;
      HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   }
   /* Same BO but a disjoint range is fine. */
   make_pair(&s, &n);
   n.bindings[2].handle = s.bindings[0].handle;
   n.bindings[2].offset = s.bindings[0].offset + s.bindings[0].size;
   HK_EVAL_CHECK(hk_compute_pair_canny_match(&s, &n, NULL));

   /* A direction table entry outside [-1, 1] needs a wider ring. */
   make_pair(&s, &n);
   n.extra.canny.params[0] |= 3u << 6;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));
   make_pair(&s, &n);
   n.extra.canny.params[1] |= 3u << 14;
   HK_EVAL_CHECK(!hk_compute_pair_canny_match(&s, &n, NULL));

   /* Only the 8 Sobel directions matter. */
   make_pair(&s, &n);
   n.extra.canny.params[0] |= 0xffff0000u;
   HK_EVAL_CHECK(hk_compute_pair_canny_match(&s, &n, NULL));

   /* Plans the fused kernel does not take. */
   struct hk_2d_stencil_plan sp = sobel_plan();
   sp.stride_pc_offset_B = HK_SOBEL_NMS_PUSH_OFFSET_B;
   HK_EVAL_CHECK(!hk_2d_stencil_sobel_nms_supported(&sp));
   sp = sobel_plan();
   sp.kind = HK_2D_STENCIL_BOX_BLUR_U8;
   HK_EVAL_CHECK(!hk_2d_stencil_sobel_nms_supported(&sp));
   sp = sobel_plan();
   sp.tile_h = 16;
   HK_EVAL_CHECK(!hk_2d_stencil_sobel_nms_supported(&sp));
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_pairing_rule();
   test_fused_kernel_matches();
   glsl_type_singleton_decref();
   printf("hk_canny_fuse: %u image cases, OK\n", cases_run);
   return 0;
}
