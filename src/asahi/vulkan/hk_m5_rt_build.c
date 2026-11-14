/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_m5_rt_runtime.h"

#include <stdlib.h>
#include <string.h>
#include "asahi/compiler/agx3_compose/agx3_compose.h"

enum { HK_M5_RT_ZERO = 0, HK_M5_RT_ROW = 1, HK_M5_RT_RAW = 2 };

static bool
read_varint(const uint8_t *s, size_t n, size_t *p, uint64_t *v)
{
   uint64_t out = 0;
   for (unsigned shift = 0; shift < 64; shift += 7) {
      if (*p >= n)
         return false;
      const uint8_t b = s[(*p)++];
      out |= (uint64_t)(b & 0x7f) << shift;
      if (!(b & 0x80)) {
         *v = out;
         return true;
      }
   }
   return false;
}

/* Tokens: varint literal count, literals, varint match length (0 ends),
 * varint distance. Returns a malloc'd buffer of exactly out_size bytes. */
static uint8_t *
hk_m5_rt_lz_decode(const uint8_t *in, size_t in_size, size_t out_size)
{
   uint8_t *out = malloc(out_size ? out_size : 1);
   if (!out)
      return NULL;
   size_t p = 0, o = 0;
   for (;;) {
      uint64_t lit, len, dist;
      if (!read_varint(in, in_size, &p, &lit) || lit > in_size - p ||
          lit > out_size - o)
         goto fail;
      memcpy(out + o, in + p, lit);
      p += lit;
      o += lit;
      if (!read_varint(in, in_size, &p, &len))
         goto fail;
      if (!len)
         break;
      if (!read_varint(in, in_size, &p, &dist) || !dist || dist > o ||
          len > out_size - o)
         goto fail;
      for (uint64_t i = 0; i < len; ++i, ++o)
         out[o] = out[o - dist];
   }
   if (o == out_size)
      return out;
fail:
   free(out);
   return NULL;
}

static uint64_t
hk_m5_rt_digest(const uint8_t *data, size_t size)
{
   uint64_t h = 0xcbf29ce484222325ull;
   for (size_t i = 0; i < size; ++i)
      h = (h ^ data[i]) * 0x100000001b3ull;
   return h;
}

static bool
hk_m5_rt_build(const uint8_t *lz, size_t lz_size, size_t stream_size,
               uint8_t *dst, bool rows_allowed, uint64_t digest)
{
   uint8_t *s = hk_m5_rt_lz_decode(lz, lz_size, stream_size);
   if (!s)
      return false;
   memset(dst, 0, HK_M5_RT_IMAGE_SIZE);
   size_t p = 0, at = 0;
   bool ok = true;
   while (ok && p < stream_size) {
      const uint8_t tag = s[p++];
      uint64_t v;
      if (!read_varint(s, stream_size, &p, &v)) {
         ok = false;
      } else if (tag == HK_M5_RT_ZERO) {
         ok = v <= HK_M5_RT_IMAGE_SIZE - at;
         at += v;
      } else if (tag == HK_M5_RT_RAW) {
         ok = v <= HK_M5_RT_IMAGE_SIZE - at && v <= stream_size - p;
         if (ok) {
            memcpy(dst + at, s + p, v);
            p += v;
            at += v;
         }
      } else if (tag == HK_M5_RT_ROW && rows_allowed &&
                 v < hk_m5_rt_shape_count &&
                 hk_m5_rt_shapes[v].form < agx3c_table.num_forms &&
                 hk_m5_rt_shapes[v].field_off <= hk_m5_rt_field_id_count &&
                 hk_m5_rt_shapes[v].nfields <=
                    hk_m5_rt_field_id_count - hk_m5_rt_shapes[v].field_off) {
         const struct hk_m5_rt_shape *shape = &hk_m5_rt_shapes[v];
         uint64_t fields[AGX3_MAX_FIELDS] = {0};
         for (unsigned i = 0; ok && i < shape->nfields; ++i) {
            const uint8_t id = hk_m5_rt_field_ids[shape->field_off + i];
            ok = id < AGX3_MAX_FIELDS &&
                 read_varint(s, stream_size, &p, &fields[id]);
         }
         const unsigned size = agx3c_table.forms[shape->form].size;
         ok = ok && size <= HK_M5_RT_IMAGE_SIZE - at &&
              agx3_compose(&agx3c_table, shape->form, fields, dst + at);
         at += size;
      } else {
         ok = false;
      }
   }
   free(s);
   return ok && at == HK_M5_RT_IMAGE_SIZE &&
          hk_m5_rt_digest(dst, HK_M5_RT_IMAGE_SIZE) == digest;
}

bool
hk_m5_rt_build_code(uint8_t *dst)
{
   return hk_m5_rt_build(hk_m5_rt_code_lz, hk_m5_rt_code_lz_size,
                         hk_m5_rt_code_stream_size, dst, true,
                         hk_m5_rt_code_digest);
}

bool
hk_m5_rt_build_user_image(uint8_t *dst)
{
   return hk_m5_rt_build(hk_m5_rt_user_lz, hk_m5_rt_user_lz_size,
                         hk_m5_rt_user_stream_size, dst, false,
                         hk_m5_rt_user_digest);
}
