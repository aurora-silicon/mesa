/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hk_2d_stencil.h"
#include "hk_compute_pair.h"

/* A byte range [start, end) of one BO, as seen through a storage backing. Two
 * descriptors alias exactly when they name the same BO handle and their ranges
 * intersect, whatever their buffer offsets or sizes. */
struct hk_compute_pair_interval {
   uint64_t start, end;
   uint32_t handle;
};

static inline bool
compute_pair_interval(const struct hk_storage_backing *backing,
                      uint64_t start, uint64_t end, struct hk_compute_pair_interval *out)
{
   if (!backing->handle || start >= end || end > backing->size ||
       backing->offset > UINT64_MAX - end)
      return false;
   *out = (struct hk_compute_pair_interval){
      backing->offset + start, backing->offset + end, backing->handle,
   };
   return true;
}

static inline bool
compute_pair_same_interval(const struct hk_compute_pair_interval *a,
                           const struct hk_compute_pair_interval *b)
{
   return a->handle == b->handle && a->start == b->start && a->end == b->end;
}

static inline bool
compute_pair_intervals_disjoint(const struct hk_compute_pair_interval *a,
                                const struct hk_compute_pair_interval *b)
{
   return a->handle != b->handle || a->end <= b->start || b->end <= a->start;
}

/* Whether the linear FADD `producer` (bindings 0 and 1 read, binding 2
 * written, all over the same element range) may also write the copy4
 * `copy4`'s destination as a second output. The copy source must be exactly
 * the producer's output, and the copy destination must not overlap the output
 * or either producer read; otherwise the two dispatches stay separate. */
static inline bool
hk_compute_pair_fadd_copy4_match(const struct hk_compute_pair_dispatch *producer,
                                 const struct hk_compute_pair_dispatch *copy4)
{
   const struct hk_compute_pair_copy4 *fp = &producer->extra.copy4;
   const struct hk_compute_pair_copy4 *cp = &copy4->extra.copy4;
   if (producer->info.role != HK_COMPUTE_PAIR_FADD ||
       producer->alt_kind != HK_COMPUTE_PAIR_ALT_LINEAR_FADD ||
       copy4->info.role != HK_COMPUTE_PAIR_COPY4 ||
       !copy4->bindings[1].address ||
       copy4->bindings[1].address > UINT64_MAX - cp->dst_start ||
       cp->src_start > UINT64_MAX - cp->bytes ||
       cp->dst_start > UINT64_MAX - cp->bytes ||
       hk_dual_write_push_offset(producer->info.fadd.push_bytes) +
          sizeof(struct hk_dual_copy_push) > HK_MAX_PUSH_SIZE ||
       ((copy4->bindings[1].address + cp->dst_start) % 16) != 0 ||
       fp->src_start > UINT64_MAX - fp->bytes ||
       fp->dst_start > UINT64_MAX - fp->bytes)
      return false;

   struct hk_compute_pair_interval output, src, dst;
   if (!compute_pair_interval(&producer->bindings[2], fp->dst_start,
                              fp->dst_start + fp->bytes, &output) ||
       !compute_pair_interval(&copy4->bindings[0], cp->src_start,
                              cp->src_start + cp->bytes, &src) ||
       !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                              cp->dst_start + cp->bytes, &dst) ||
       !compute_pair_same_interval(&output, &src) ||
       !compute_pair_intervals_disjoint(&output, &dst))
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], fp->src_start,
                                 fp->src_start + fp->bytes, &read) ||
          !compute_pair_intervals_disjoint(&read, &dst))
         return false;
   }
   return true;
}

static inline bool
compute_pair_va_disjoint(uint64_t a, uint64_t a_len, uint64_t b, uint64_t b_len)
{
   return a <= UINT64_MAX - a_len && b <= UINT64_MAX - b_len &&
          (a + a_len <= b || b + b_len <= a);
}

/* Whether the Sobel `sobel` and the Canny NMS `nms`, the dispatch right after
 * it in the same submission, may run as one hk_nir_build_sobel_nms() dispatch
 * that writes what both write, and its NMS inputs if so.
 *
 * The NMS must read exactly the dir and mag the Sobel writes (the same
 * buffers over the same image size), so the fused kernel can take them from
 * its own registers and workgroup memory. Its output must be clear of the
 * Sobel's source, dir and mag, by BO range and by address, so that no
 * Sobel input or output changes under it. Its direction tables must stay within
 * the one-pixel magnitude ring (hk_sobel_nms_lut_in_ring()). */
static inline bool
hk_compute_pair_canny_match(const struct hk_compute_pair_dispatch *sobel,
                            const struct hk_compute_pair_dispatch *nms,
                            struct hk_sobel_nms_push *push_out)
{
   if (!sobel || !nms || sobel->info.role != HK_COMPUTE_PAIR_SOBEL3X3 ||
       nms->info.role != HK_COMPUTE_PAIR_CANNY_NMS ||
       !sobel->extra.canny.fused || !sobel->root_va)
      return false;

   const struct hk_compute_pair_canny *sc = &sobel->extra.canny;
   const struct hk_compute_pair_canny *nc = &nms->extra.canny;
   if (sc->width < 1 || sc->height < 1 || sc->stride < sc->width ||
       nc->width != sc->width || nc->height != sc->height ||
       !hk_sobel_nms_lut_in_ring(nc->params[0]) ||
       !hk_sobel_nms_lut_in_ring(nc->params[1]))
      return false;

   const uint64_t px = (uint64_t)sc->width * (uint64_t)sc->height;
   const uint64_t out4 = px * 4u;
   const uint64_t src_bytes =
      (uint64_t)(sc->height - 1) * (uint64_t)sc->stride + (uint64_t)sc->width;
   if (out4 > (uint64_t)INT32_MAX || src_bytes > (uint64_t)INT32_MAX)
      return false;

   struct hk_compute_pair_interval src, s_dir, s_mag, n_dir, n_mag, dst;
   if (!compute_pair_interval(&sobel->bindings[0], 0, src_bytes, &src) ||
       !compute_pair_interval(&sobel->bindings[1], 0, out4, &s_dir) ||
       !compute_pair_interval(&sobel->bindings[2], 0, out4, &s_mag) ||
       !compute_pair_interval(&nms->bindings[0], 0, out4, &n_dir) ||
       !compute_pair_interval(&nms->bindings[1], 0, out4, &n_mag) ||
       !compute_pair_interval(&nms->bindings[2], 0, px, &dst))
      return false;

   if (!compute_pair_same_interval(&s_dir, &n_dir) ||
       !compute_pair_same_interval(&s_mag, &n_mag) ||
       sobel->bindings[1].address != nms->bindings[0].address ||
       sobel->bindings[2].address != nms->bindings[1].address)
      return false;

   const uint64_t dst_va = nms->bindings[2].address;
   if (!dst_va ||
       !compute_pair_intervals_disjoint(&dst, &src) ||
       !compute_pair_intervals_disjoint(&dst, &s_dir) ||
       !compute_pair_intervals_disjoint(&dst, &s_mag) ||
       !compute_pair_va_disjoint(dst_va, px, sobel->bindings[0].address, src_bytes) ||
       !compute_pair_va_disjoint(dst_va, px, sobel->bindings[1].address, out4) ||
       !compute_pair_va_disjoint(dst_va, px, sobel->bindings[2].address, out4))
      return false;

   if (push_out) {
      *push_out = (struct hk_sobel_nms_push){
         .dst_va_lo = (uint32_t)dst_va,
         .dst_va_hi = (uint32_t)(dst_va >> 32),
         .dx_lut = nc->params[0],
         .dy_lut = nc->params[1],
         .low_bits = nc->params[2],
         .high_bits = nc->params[3],
      };
   }
   return true;
}
