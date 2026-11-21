/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot_ioto.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx3_reencode_cache.h"
#include "asahi/compiler/agx3_transcode.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_chip.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_ioto_patch.h"
#include "asahi/lib/agx_ioto_program_core.h"
#include "util/bitscan.h"
#include "util/format/format_utils.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

static_assert(sizeof(struct agx_bg_eot_ioto_cache_key) % sizeof(uint64_t) == 0,
              "word-addressable BG/EOT device cache key");

struct agx_bg_eot_ioto_cache_entry {
   struct agx_bg_eot_ioto_cache_key key;
   uint64_t hash;
   struct agx_bg_eot_ioto_result program[AGX_BG_EOT_IOTO_CACHE_VARIANTS];
   uint8_t valid_variants;
};

struct agx_bg_eot_ioto_arena {
   struct agx_bo *bo;
   uint8_t *map;
   uint64_t base;
   uint32_t cursor;
   uint32_t capacity;
};

struct agx_bg_eot_ioto_cache {
   simple_mtx_t lock;
   struct agx_bg_eot_ioto_arena program;
   struct agx_bg_eot_ioto_arena data;
   struct agx_bo *bos[2 * AGX_BG_EOT_IOTO_CACHE_MAX_CHUNKS];
   unsigned bo_count;
   unsigned count;
   bool full;
   /* Open-addressed index: entry number + 1, or 0 when free. */
   uint16_t index[2 * AGX_BG_EOT_IOTO_CACHE_ENTRIES];
   struct agx_bg_eot_ioto_cache_entry entries[AGX_BG_EOT_IOTO_CACHE_ENTRIES];
};

static uint64_t
agx_bg_eot_ioto_cache_hash(const struct agx_bg_eot_ioto_cache_key *dk)
{
   const uint8_t *bytes = (const uint8_t *)dk;
   uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
   for (unsigned i = 0; i < sizeof(*dk) / sizeof(uint64_t); ++i) {
      uint64_t word;
      memcpy(&word, bytes + i * sizeof(word), sizeof(word));
      hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
   }
   return hash;
}

static struct agx_bg_eot_ioto_cache_entry *
agx_bg_eot_ioto_cache_find(struct agx_bg_eot_ioto_cache *dc,
                           const struct agx_bg_eot_ioto_cache_key *dk,
                           uint64_t hash)
{
   const unsigned mask = ARRAY_SIZE(dc->index) - 1;
   static_assert(
      (ARRAY_SIZE(((struct agx_bg_eot_ioto_cache *)0)->index) &
       (ARRAY_SIZE(((struct agx_bg_eot_ioto_cache *)0)->index) - 1)) == 0,
      "power-of-two index");
   for (unsigned i = hash & mask;; i = (i + 1) & mask) {
      if (!dc->index[i])
         return NULL;
      struct agx_bg_eot_ioto_cache_entry *e = &dc->entries[dc->index[i] - 1];
      if (e->hash == hash && !memcmp(&e->key, dk, sizeof(*dk)))
         return e;
   }
}

/* The index is twice the entry bound, so a free slot always exists. */
static struct agx_bg_eot_ioto_cache_entry *
agx_bg_eot_ioto_cache_insert(struct agx_bg_eot_ioto_cache *dc,
                             const struct agx_bg_eot_ioto_cache_key *dk,
                             uint64_t hash)
{
   if (dc->count >= AGX_BG_EOT_IOTO_CACHE_ENTRIES)
      return NULL;
   const unsigned mask = ARRAY_SIZE(dc->index) - 1;
   unsigned i = hash & mask;
   while (dc->index[i])
      i = (i + 1) & mask;
   struct agx_bg_eot_ioto_cache_entry *e = &dc->entries[dc->count];
   memcpy(&e->key, dk, sizeof(*dk));
   e->hash = hash;
   e->valid_variants = 0;
   memset(e->program, 0, sizeof(e->program));
   dc->index[i] = (uint16_t)(++dc->count);
   return e;
}

struct agx_bg_eot_ioto_cache *
agx_bg_eot_ioto_cache_create(struct agx_device *dev)
{
   if (!dev || !agx_chip_native_agx3(dev->chip))
      return NULL;
   struct agx_bg_eot_ioto_cache *dc = calloc(1, sizeof(*dc));
   if (!dc)
      return NULL;
   simple_mtx_init(&dc->lock, mtx_plain);
   return dc;
}

void
agx_bg_eot_ioto_cache_destroy(struct agx_device *dev,
                              struct agx_bg_eot_ioto_cache *dc)
{
   if (!dc)
      return;
   for (unsigned i = 0; i < dc->bo_count; ++i)
      agx_bo_unreference(dev, dc->bos[i]);
   simple_mtx_destroy(&dc->lock);
   free(dc);
}

static bool
agx_bg_eot_ioto_arena_room(struct agx_device *dev,
                           struct agx_bg_eot_ioto_cache *dc,
                           struct agx_bg_eot_ioto_arena *a, bool exec,
                           uint32_t need, uint32_t align)
{
   if (a->map && align_uintptr(a->cursor, align) + need <= a->capacity)
      return true;
   if (dc->bo_count >= ARRAY_SIZE(dc->bos))
      return false;
   struct agx_bo *bo = agx_bo_create(
      dev, AGX_BG_EOT_IOTO_CACHE_CHUNK,
      exec ? AGX_IOTO_PROGRAM_ALIGNMENT : 0,
      exec ? (AGX_BO_EXEC | AGX_BO_LOW_VA) : 0,
      exec ? "G17P BG/EOT device programs" : "G17P BG/EOT device arguments");
   if (!bo)
      return false;
   void *map = agx_bo_map(bo);
   if (!map || !bo->va) {
      agx_bo_unreference(dev, bo);
      return false;
   }
   dc->bos[dc->bo_count++] = bo;
   a->bo = bo;
   a->map = map;
   a->base = bo->va->addr;
   a->cursor = 0;
   a->capacity = bo->size > UINT32_MAX ? UINT32_MAX : (uint32_t)bo->size;
   return align_uintptr(0, align) + need <= a->capacity;
}

static struct agx_ptr
agx_bg_eot_ioto_arena_take(struct agx_bg_eot_ioto_arena *a, uint32_t size,
                           uint32_t align)
{
   const uint32_t offset = (uint32_t)align_uintptr(a->cursor, align);
   assert(a->map && offset + size <= a->capacity);
   a->cursor = offset + size;
   return (struct agx_ptr){.cpu = a->map + offset, .gpu = a->base + offset};
}

struct agx_bg_eot_cache_sink_ctx {
   struct agx_bg_eot_ioto_cache *dc;
   struct agx_bo *used_data_bo;
};

static uint64_t
agx_bg_eot_cache_sink_upload(void *ctx, const void *data, uint32_t size,
                             uint32_t alignment, struct agx_bo **out_bo)
{
   struct agx_bg_eot_cache_sink_ctx *sc = ctx;
   struct agx_bg_eot_ioto_arena *a = &sc->dc->data;
   struct agx_ptr t = agx_bg_eot_ioto_arena_take(a, size, alignment);
   memcpy(t.cpu, data, size);
   sc->used_data_bo = a->bo;
   if (out_bo)
      *out_bo = a->bo;
   return t.gpu;
}

static bool
agx_bg_eot_cache_sink_reserve(void *ctx, size_t capacity, const char *label,
                              struct agx_bg_eot_ioto_reservation *out)
{
   struct agx_bg_eot_cache_sink_ctx *sc = ctx;
   struct agx_bg_eot_ioto_arena *a = &sc->dc->program;
   if (capacity > AGX_BG_EOT_IOTO_CACHE_PROGRAM_MAX)
      return false;
   struct agx_ptr t = agx_bg_eot_ioto_arena_take(a, (uint32_t)capacity,
                                                 AGX_IOTO_PROGRAM_ALIGNMENT);
   *out = (struct agx_bg_eot_ioto_reservation){
      .map = t.cpu,
      .va = t.gpu,
      .bo = a->bo,
   };
   return true;
}

static uint64_t
agx_bg_eot_cache_sink_publish(void *ctx,
                              const struct agx_bg_eot_ioto_reservation *r,
                              size_t length)
{
   struct agx_bg_eot_cache_sink_ctx *sc = ctx;
   struct agx_bg_eot_ioto_arena *a = &sc->dc->program;
   assert(r->va >= a->base && length > 0);
   const uint64_t offset = r->va - a->base;
   /* Return any over-reservation back to the arena. */
   a->cursor = (uint32_t)(offset + length);
   agx_bo_assert_range(a->bo, offset, length);
   return r->va;
}

struct agx_bg_eot_ioto_cache_entry *
agx_bg_eot_ioto_cache_get(
   struct agx_bg_eot_ioto_cache *dc,
   const struct agx_bg_eot_ioto_cache_key *key,
   struct agx_bg_eot_ioto_result out_programs[AGX_BG_EOT_IOTO_CACHE_VARIANTS],
   uint8_t *out_valid_variants)
{
   if (out_valid_variants)
      *out_valid_variants = 0;
   if (!dc || !key)
      return NULL;

   const uint64_t hash = agx_bg_eot_ioto_cache_hash(key);
   simple_mtx_lock(&dc->lock);
   struct agx_bg_eot_ioto_cache_entry *de =
      agx_bg_eot_ioto_cache_find(dc, key, hash);
   if (!de && !dc->full)
      de = agx_bg_eot_ioto_cache_insert(dc, key, hash);
   if (de) {
      if (out_programs)
         memcpy(out_programs, de->program, sizeof(de->program));
      if (out_valid_variants)
         *out_valid_variants = de->valid_variants;
   }
   simple_mtx_unlock(&dc->lock);
   return de;
}

static bool
agx_bg_eot_ioto_cache_build_locked(struct agx_device *dev,
                                   struct agx_bg_eot_ioto_cache *dc,
                                   struct agx_bg_eot_ioto_cache_entry *entry,
                                   unsigned variant,
                                   agx_bg_eot_ioto_build_fn build, void *ctx,
                                   struct agx_bg_eot_ioto_result *out,
                                   bool *out_in_cache)
{
   simple_mtx_assert_locked(&dc->lock);
   assert(variant < AGX_BG_EOT_IOTO_CACHE_VARIANTS);

   if (entry->valid_variants & BITFIELD_BIT(variant)) {
      *out = entry->program[variant];
      if (out_in_cache)
         *out_in_cache = true;
      return true;
   }

   if (!agx_bg_eot_ioto_arena_room(dev, dc, &dc->program, true,
                                   AGX_BG_EOT_IOTO_CACHE_PROGRAM_MAX,
                                   AGX_IOTO_PROGRAM_ALIGNMENT) ||
       !agx_bg_eot_ioto_arena_room(dev, dc, &dc->data, false,
                                   AGX_BG_EOT_IOTO_CACHE_DATA_MAX, 32)) {
      dc->full = true;
      if (out_in_cache)
         *out_in_cache = false;
      return false;
   }

   const uint32_t prog_cursor_before = dc->program.cursor;
   const uint32_t data_cursor_before = dc->data.cursor;
   struct agx_bg_eot_cache_sink_ctx sink_ctx = {.dc = dc};
   const struct agx_bg_eot_ioto_sink sink = {
      .ctx = &sink_ctx,
      .upload = agx_bg_eot_cache_sink_upload,
      .reserve = agx_bg_eot_cache_sink_reserve,
      .publish = agx_bg_eot_cache_sink_publish,
   };

   struct agx_bg_eot_ioto_result built = {0};
   const bool ok = build(ctx, &sink, &built);
   if (ok && built.program_va) {
      built.program_bo = dc->program.bo;
      built.data_bo = sink_ctx.used_data_bo;
      entry->program[variant] = built;
      entry->valid_variants |= BITFIELD_BIT(variant);
      *out = built;
      if (out_in_cache)
         *out_in_cache = true;
      return true;
   }

   /* Roll back arena cursors if build failed partway through. */
   dc->program.cursor = prog_cursor_before;
   dc->data.cursor = data_cursor_before;
   *out = (struct agx_bg_eot_ioto_result){0};
   if (out_in_cache)
      *out_in_cache = false;
   return true;
}

bool
agx_bg_eot_ioto_cache_lookup_or_build(
   struct agx_device *dev, struct agx_bg_eot_ioto_cache *dc,
   struct agx_bg_eot_ioto_cache_entry *entry, unsigned variant,
   agx_bg_eot_ioto_build_fn build, void *ctx,
   struct agx_bg_eot_ioto_result *out, bool *out_in_cache)
{
   if (out_in_cache)
      *out_in_cache = false;
   if (!dc || !entry || variant >= AGX_BG_EOT_IOTO_CACHE_VARIANTS)
      return false;

   simple_mtx_lock(&dc->lock);
   const bool handled = agx_bg_eot_ioto_cache_build_locked(
      dev, dc, entry, variant, build, ctx, out, out_in_cache);
   simple_mtx_unlock(&dc->lock);
   return handled;
}

bool
agx_bg_eot_ioto_cache_get_or_build(
   struct agx_device *dev, struct agx_bg_eot_ioto_cache *dc,
   const struct agx_bg_eot_ioto_cache_key *key, unsigned variant,
   agx_bg_eot_ioto_build_fn build, void *ctx,
   struct agx_bg_eot_ioto_result *out, bool *out_in_cache)
{
   if (out_in_cache)
      *out_in_cache = false;
   if (!dc || !key || variant >= AGX_BG_EOT_IOTO_CACHE_VARIANTS)
      return false;

   const uint64_t hash = agx_bg_eot_ioto_cache_hash(key);
   simple_mtx_lock(&dc->lock);
   struct agx_bg_eot_ioto_cache_entry *de =
      agx_bg_eot_ioto_cache_find(dc, key, hash);
   if (!de && !dc->full)
      de = agx_bg_eot_ioto_cache_insert(dc, key, hash);
   bool handled = false;
   if (de) {
      handled = agx_bg_eot_ioto_cache_build_locked(
         dev, dc, de, variant, build, ctx, out, out_in_cache);
   }
   simple_mtx_unlock(&dc->lock);
   return handled;
}

bool
agx_bg_eot_shader_supported(const struct agx_bg_eot_shader *shader)
{
   return shader && shader->bo && shader->bo->va && shader->ptr != 0 &&
          shader->info.binary_size > 0 &&
          shader->info.main_offset < shader->info.binary_size &&
          shader->info.main_size <=
             shader->info.binary_size - shader->info.main_offset &&
          !shader->info.has_preamble && shader->info.rodata.size_16 == 0 &&
          shader->info.nr_gprs > 0;
}

bool
agx_g17p_bg_eot_ops_uniform(const struct agx_bg_eot_key *key)
{
   bool seen = false;
   enum agx_bg_eot_op first = AGX_BG_EOT_NONE;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->tib.logical_format); ++rt) {
      if (key->tib.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      if (!seen) {
         first = key->op[rt];
         seen = true;
      } else if (key->op[rt] != first) {
         return false;
      }
   }
   return true;
}

static uint32_t
agx_g17p_unorm8(float x)
{
   return x == x ? (uint32_t)_mesa_float_to_unorm(x, 8) : 0;
}

uint32_t
agx_g17p_pack_unorm8_clear(const float rgba[4])
{
   return agx_g17p_unorm8(rgba[0]) | (agx_g17p_unorm8(rgba[1]) << 8) |
          (agx_g17p_unorm8(rgba[2]) << 16) | (agx_g17p_unorm8(rgba[3]) << 24);
}

bool
agx_bg_eot_build_g17p_mixed(struct agx_bg_eot_cache *cache,
                            const struct agx_bg_eot_g17p_mixed_desc *in,
                            const struct agx_bg_eot_ioto_sink *sink,
                            struct agx_bg_eot_ioto_result *out)
{
   *out = (struct agx_bg_eot_ioto_result){0};
   struct agx_device *dev = cache->dev;
   struct agx_bg_eot_key key = in->key;

   const bool clears = !in->store && in->count > in->reloads;
   if (clears)
      key.reserved_preamble = 4 + 8 * in->count;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(cache, &key);
   if (!agx_bg_eot_shader_supported(shader) ||
       (!in->store && (shader->info.push_count != key.reserved_preamble ||
                       shader->info.uses_txf != (in->reloads > 0)))) {
      mesa_loge("G17P %s shader declined: supported=%u push=%u want=%u "
                "uses_txf=%u want=%u count=%u reloads=%u samples=%u "
                "layered=%u sample_size=%uB",
                in->store ? "EOT" : "background",
                agx_bg_eot_shader_supported(shader),
                shader ? shader->info.push_count : 0, key.reserved_preamble,
                shader ? shader->info.uses_txf : 0, in->reloads > 0, in->count,
                in->reloads, key.tib.nr_samples, key.tib.layered,
                key.tib.sample_size_B);
      return false;
   }

   const enum agx3_g17p_rt0_bg_op bg_op =
      in->count == 0            ? AGX3_G17P_RT0_BG_NONE
      : in->reloads == in->count ? AGX3_G17P_RT0_BG_RELOAD
      : in->reloads             ? AGX3_G17P_RT0_BG_CLEAR_RELOAD
                                : AGX3_G17P_RT0_BG_CLEAR;
   const size_t program_capacity =
      in->count == 0 ? AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
      : in->store    ? (in->count > 4 ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                                      : AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE)
      : in->count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
      : bg_op == AGX3_G17P_RT0_BG_CLEAR_RELOAD
         ? AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE
         : AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;

   struct agx_bg_eot_ioto_reservation reservation = {0};
   if (!sink->reserve(sink->ctx, program_capacity,
                      "G17P mixed MRT Ioto program", &reservation))
      return false;

   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = key.tib.sample_size_B,
      .nr_samples = key.tib.nr_samples,
      .tile_size_px = key.tib.tile_size,
   };
   const uint8_t *program_bytes = NULL;
   size_t program_length = 0;
   uint64_t resource_spec = 0;
   struct agx_bo *data_bo = NULL;
   struct agx3_g17p_graphics_program empty_program;
   struct agx3_g17p_rt0_eot_program eot_program;
   struct agx3_g17p_rt0_bg_program bg_program;

   if (in->count == 0) {
      const struct agx3_g17p_graphics_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = imageblock,
      };
      if (!agx3_g17p_graphics_program_generate(dev->chip, &desc,
                                               &empty_program) ||
          empty_program.length > program_capacity ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, bg_op, &resource_spec))
         return false;
      program_bytes = empty_program.bytes;
      program_length = empty_program.length;
   } else if (in->store) {
      const uint64_t pbe_argument_va = sink->upload(
         sink->ctx, in->pbe_descs,
         in->count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE, 32, &data_bo);
      if (!pbe_argument_va)
         return false;

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = key.tib._offset_B[0],
         .attachment_count = in->count,
         .imageblock_offset1_B = key.tib._offset_B[1],
         .imageblock_offset2_B = key.tib._offset_B[2],
         .imageblock_offset3_B = key.tib._offset_B[3],
         .imageblock_offset4_B = key.tib._offset_B[4],
         .imageblock_offset5_B = key.tib._offset_B[5],
         .imageblock_offset6_B = key.tib._offset_B[6],
         .imageblock_offset7_B = key.tib._offset_B[7],
         .mixed_mrt = in->count > 1,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                             ? shader->info.agx3_staging_halfs
                             : 0,
         .imageblock = imageblock,
      };
      if (!agx3_g17p_rt0_eot_program_generate(dev->chip, &desc, &eot_program) ||
          eot_program.length > program_capacity)
         return false;
      program_bytes = eot_program.bytes;
      program_length = eot_program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .clear_start_halfs = 4 + 8 * in->reloads,
         .staging_known = shader->info.agx3_staging_known,
         .op = bg_op,
         .imageblock = imageblock,
      };

      if (clears) {
         desc.clear_value_va = sink->upload(
            sink->ctx, in->clear_values, 16 * (in->count - in->reloads), 16,
            &data_bo);
         if (!desc.clear_value_va)
            return false;
      }

      if (in->reloads) {
         /* Target n reads native texture index 2n; the odd entries stay zero. */
         uint8_t textures[2 * 8 - 1][AGX_BG_EOT_DESC_SIZE] = {0};
         for (unsigned rt = 0; rt < in->reloads; ++rt)
            memcpy(textures[2 * rt], in->tex_descs[rt], AGX_BG_EOT_DESC_SIZE);
         const uint64_t texture_va = sink->upload(
            sink->ctx, textures,
            (2 * in->reloads - 1) * AGX_BG_EOT_DESC_SIZE,
            AGX_BG_EOT_DESC_SIZE, &data_bo);
         uint64_t sampler_va = in->sampler_va;
         if (!sampler_va && in->sampler_data && in->sampler_size) {
            sampler_va = sink->upload(sink->ctx, in->sampler_data,
                                      in->sampler_size, 64, &data_bo);
         }
         if (!texture_va || !sampler_va)
            return false;
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table =
            sink->upload(sink->ctx, pointers, sizeof(pointers), 8, &data_bo);
         if (!table)
            return false;
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      if (!agx3_g17p_rt0_bg_program_generate(dev->chip, &desc, &bg_program) ||
          bg_program.length > program_capacity ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, bg_op, &resource_spec))
         return false;
      program_bytes = bg_program.bytes;
      program_length = bg_program.length;
   }

   memcpy(reservation.map, program_bytes, program_length);
   const uint64_t published =
      sink->publish(sink->ctx, &reservation, program_length);
   if (!published)
      return false;

   *out = (struct agx_bg_eot_ioto_result){
      .program_va = published,
      .rsrc_spec = resource_spec,
      .program_bo = reservation.bo,
      .data_bo = data_bo,
      .shader_bo = shader->bo,
   };
   return true;
}

bool
agx_bg_eot_build_g17p_uniform(struct agx_bg_eot_cache *cache,
                              const struct agx_bg_eot_g17p_uniform_desc *in,
                              const struct agx_bg_eot_ioto_sink *sink,
                              struct agx_bg_eot_ioto_result *out)
{
   *out = (struct agx_bg_eot_ioto_result){0};
   struct agx_device *dev = cache->dev;
   struct agx_bg_eot_key key = in->key;

   const bool wboit_mrt = in->mrt == AGX_G17P_MRT_WBOIT;
   const bool rgba8_mrt = in->mrt == AGX_G17P_MRT_RGBA8;
   const bool r11g11b10_mrt = in->mrt == AGX_G17P_MRT_R11G11B10;

   if (!in->store && in->op == AGX_BG_CLEAR) {
      key.reserved_preamble =
         in->wide_mrt || in->multi_mrt_active
            ? 4 + 8 * in->slot_count
         : rgba8_mrt && in->dual_mrt_active
            ? AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
         : r11g11b10_mrt && in->dual_mrt_active
            ? AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS
         : wboit_mrt && in->dual_mrt_active
            ? AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
            : AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
   }

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(cache, &key);
   const bool size_ok =
      !in->max_shader_binary_size ||
      (shader && shader->info.binary_size <= in->max_shader_binary_size);
   const bool resources_match =
      shader &&
      (in->store ||
       (in->op == AGX_BG_EOT_NONE
           ? shader->info.push_count == 0 && !shader->info.uses_txf
           : agx3_g17p_rt0_bg_shader_resources_match(
                in->op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                       : AGX3_G17P_RT0_BG_RELOAD,
                shader->info.push_count, shader->info.uses_txf)));
   if (!agx_bg_eot_shader_supported(shader) || !size_ok || !resources_match)
      return false;

   size_t capacity;
   const char *label;
   if (in->op == AGX_BG_EOT_NONE) {
      capacity = AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE;
      label = "G17P empty tile Ioto program";
   } else if (in->wide_mrt || key.store_and_resolve) {
      capacity = in->store ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                           : AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE;
      label = in->store ? "G17P wide EOT Ioto program"
                        : "G17P wide background Ioto program";
   } else if (in->store) {
      capacity = in->multi_mrt_active ? AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE
                                      : AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE;
      label = "G17P RT0 EOT Ioto program";
   } else if (in->op == AGX_BG_CLEAR) {
      capacity =
         in->multi_mrt_active ? AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE
         : (rgba8_mrt || r11g11b10_mrt) && in->dual_mrt_active
            ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
         : wboit_mrt && in->dual_mrt_active
            ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
            : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE;
      label = "G17P RT0 clear Ioto program";
   } else {
      capacity = AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
      label = "G17P RT0 reload Ioto program";
   }

   struct agx_bg_eot_ioto_reservation reservation = {0};
   if (!sink->reserve(sink->ctx, capacity, label, &reservation))
      return false;

   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = key.tib.sample_size_B,
      .nr_samples = key.tib.nr_samples,
      .tile_size_px = key.tib.tile_size,
   };
   size_t program_length = 0;
   uint64_t resource_spec = 0;
   struct agx_bo *data_bo = NULL;

   if (in->store && in->op == AGX_EOT_STORE) {
      const size_t pbe_size =
         in->attachment_count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE;
      const uint64_t pbe_va =
         sink->upload(sink->ctx, in->pbe_descs, pbe_size, 32, &data_bo);
      if (!pbe_va)
         return false;

      struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_va,
         .imageblock_offset_B = agx_tilebuffer_offset_B(&key.tib, in->first_rt),
         .attachment_count = in->attachment_count,
         .mixed_mrt = in->attachment_count >= 2,
         .imageblock_offset1_B =
            key.store_and_resolve
               ? agx_tilebuffer_offset_B(&key.tib, in->first_rt)
               : 0,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                             ? shader->info.agx3_staging_halfs
                             : 0,
         .imageblock = imageblock,
      };
      if (!key.store_and_resolve) {
         uint8_t *const offsets[] = {
            &desc.imageblock_offset1_B, &desc.imageblock_offset2_B,
            &desc.imageblock_offset3_B, &desc.imageblock_offset4_B,
            &desc.imageblock_offset5_B, &desc.imageblock_offset6_B,
            &desc.imageblock_offset7_B,
         };
         unsigned offset_index = 0;
         u_foreach_bit(rt, in->attachment_mask & ~BITFIELD_BIT(in->first_rt)) {
            if (offset_index >= ARRAY_SIZE(offsets))
               break;
            *offsets[offset_index++] = agx_tilebuffer_offset_B(&key.tib, rt);
         }
      }

      struct agx3_g17p_rt0_eot_program native;
      if (!agx3_g17p_rt0_eot_program_generate(dev->chip, &desc, &native) ||
          native.length > capacity)
         return false;
      memcpy(reservation.map, native.bytes, native.length);
      program_length = native.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else if (in->op == AGX_BG_EOT_NONE) {
      const struct agx3_g17p_graphics_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = imageblock,
      };
      struct agx3_g17p_graphics_program native;
      if (!agx3_g17p_graphics_program_generate(dev->chip, &desc, &native) ||
          native.length > capacity ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, AGX3_G17P_RT0_BG_NONE,
                                          &resource_spec))
         return false;
      memcpy(reservation.map, native.bytes, native.length);
      program_length = native.length;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .staging_known = shader->info.agx3_staging_known,
         .op = in->op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                      : AGX3_G17P_RT0_BG_RELOAD,
         .imageblock = imageblock,
      };

      if (in->op == AGX_BG_CLEAR) {
         if (in->clear_value_va) {
            desc.clear_value_va = in->clear_value_va;
         } else {
            uint32_t packed_buf[4];
            const void *clear_src = in->clear_values;
            if (key.g17p_packed_clear) {
               float rgba[4];
               memcpy(rgba, in->clear_values[0], sizeof(rgba));
               memset(packed_buf, 0, sizeof(packed_buf));
               packed_buf[0] = agx_g17p_pack_unorm8_clear(rgba);
               clear_src = packed_buf;
            }
            const size_t clear_size =
               in->wide_mrt || in->multi_mrt_active ? 16 * in->slot_count
               : rgba8_mrt && in->dual_mrt_active   ? 32
               : r11g11b10_mrt && in->dual_mrt_active ? 28
               : wboit_mrt && in->dual_mrt_active   ? 20
                                                    : 16;
            desc.clear_value_va =
               sink->upload(sink->ctx, clear_src, clear_size, 16, &data_bo);
            if (!desc.clear_value_va)
               return false;
         }
      } else {
         if (key.preserve_resolve_outside) {
            desc.resolve_bounds_va = sink->upload(
               sink->ctx, in->resolve_bounds, sizeof(in->resolve_bounds), 16,
               &data_bo);
            if (!desc.resolve_bounds_va)
               return false;
         }
         const size_t texture_size = in->tex_desc_count * AGX_BG_EOT_DESC_SIZE;
         const uint64_t texture_va = sink->upload(
            sink->ctx, in->tex_descs, texture_size, AGX_BG_EOT_DESC_SIZE,
            &data_bo);
         uint64_t sampler_va = in->sampler_va;
         if (!sampler_va && in->sampler_data && in->sampler_size) {
            sampler_va = sink->upload(sink->ctx, in->sampler_data,
                                      in->sampler_size, 64, &data_bo);
         }
         if (!texture_va || !sampler_va)
            return false;
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table =
            sink->upload(sink->ctx, pointers, sizeof(pointers), 8, &data_bo);
         if (!table)
            return false;
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      struct agx3_g17p_rt0_bg_program native;
      if (!agx3_g17p_rt0_bg_program_generate(dev->chip, &desc, &native) ||
          native.length > capacity ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, desc.op, &resource_spec))
         return false;
      memcpy(reservation.map, native.bytes, native.length);
      program_length = native.length;
   }

   const uint64_t published =
      sink->publish(sink->ctx, &reservation, program_length);
   if (!published)
      return false;

   *out = (struct agx_bg_eot_ioto_result){
      .program_va = published,
      .rsrc_spec = resource_spec,
      .program_bo = reservation.bo,
      .data_bo = data_bo,
      .shader_bo = shader->bo,
   };
   return true;
}

/*
 * Device-owned immutable M5 tile launch template, ralloc child of the
 * `agx_bg_eot_shader`. Preserves the normal uniform load schedule and patches
 * base immediates when new uniform range addresses fit the template's plan.
 */
struct agx_m5_tile_template {
   struct agx_m5_tile_template *next;
   struct agx3_g17p_graphics_program_desc desc;
   struct agx3_ioto_uniform_range ranges[AGX_IOTO_PATCH_MAX_RANGES];
   struct agx_ioto_patch_plan plan;
   bool store;
   uint16_t length;
   uint8_t bytes[];
};

static bool
agx_m5_tile_matches(const struct agx_m5_tile_template *t,
                    const struct agx3_g17p_graphics_program_desc *d, bool store)
{
   const struct agx3_g17p_graphics_program_desc *a = &t->desc;
   if (t->store != store || a->shader_va != d->shader_va ||
       a->shader_binary_size != d->shader_binary_size ||
       a->scratch_size != d->scratch_size || a->nr_gprs != d->nr_gprs ||
       a->compiler_d_valid != d->compiler_d_valid ||
       a->compiler_d != d->compiler_d ||
       a->imageblock.sample_size_B != d->imageblock.sample_size_B ||
       a->imageblock.nr_samples != d->imageblock.nr_samples ||
       a->imageblock.tile_size_px != d->imageblock.tile_size_px ||
       a->uniform_range_count != d->uniform_range_count)
      return false;
   for (unsigned i = 0; i < d->uniform_range_count; ++i) {
      if (t->ranges[i].start_halfs != d->uniform_ranges[i].start_halfs ||
          t->ranges[i].size_halfs != d->uniform_ranges[i].size_halfs)
         return false;
   }
   return agx_ioto_patch_plan_fits(&t->plan, d->uniform_ranges);
}

bool
agx_bg_eot_m5_tile_program(struct agx_bg_eot_cache *cache,
                           struct agx_bg_eot_shader *shader,
                           const struct agx3_g17p_graphics_program_desc *d,
                           bool store,
                           struct agx3_g17p_graphics_program *program)
{
   struct agx_device *dev = cache->dev;
   if (!d->fragment || d->writes_sample_mask || d->colour_classes ||
       d->preloaded_root || d->patchable || d->threadgroup_memory_size_B ||
       d->constant_program.code_size ||
       (d->cprog_inputs && d->cprog_inputs->count) ||
       d->uniform_range_count > AGX_IOTO_PATCH_MAX_RANGES)
      return agx3_g17g_tile_program_generate(dev->chip, d, store, program);

   simple_mtx_lock(&cache->lock);
   struct agx_m5_tile_template *selected = NULL;
   unsigned count = 0;
   for (struct agx_m5_tile_template *t = shader->m5_tile_templates; t;
        t = t->next) {
      ++count;
      if (agx_m5_tile_matches(t, d, store)) {
         selected = t;
         break;
      }
   }
   /* Bound memory even for layouts whose addresses change incompatibly. */
   if (!selected && count < 8) {
      struct agx3_g17p_graphics_program_template *fresh =
         calloc(1, sizeof(*fresh));
      if (fresh && agx3_g17g_tile_program_template_generate(dev->chip, d, store,
                                                            fresh)) {
         struct agx_m5_tile_template *t =
            rzalloc_size(shader, sizeof(*t) + fresh->program.length);
         if (t && agx_ioto_patch_plan_init(
                     &t->plan, d->uniform_ranges, d->uniform_range_count,
                     fresh->relocations, fresh->relocation_count,
                     fresh->program.length, false)) {
            t->desc = *d;
            t->desc.uniform_ranges = t->ranges;
            t->desc.cprog_inputs = NULL;
            if (d->uniform_range_count)
               memcpy(t->ranges, d->uniform_ranges,
                      d->uniform_range_count * sizeof(t->ranges[0]));
            t->store = store;
            t->length = fresh->program.length;
            memcpy(t->bytes, fresh->program.bytes, t->length);
            t->next = shader->m5_tile_templates;
            shader->m5_tile_templates = selected = t;
         } else {
            ralloc_free(t);
         }
      }
      free(fresh);
   }
   simple_mtx_unlock(&cache->lock);

   if (!selected)
      return agx3_g17g_tile_program_generate(dev->chip, d, store, program);

   program->length = selected->length;
   memcpy(program->bytes, selected->bytes, selected->length);
   agx_ioto_patch_plan_apply(&selected->plan, d->uniform_ranges,
                             program->bytes);
   if (dev->debug & AGX_DBG_VERIFY) {
      struct agx3_g17p_graphics_program fresh;
      if (!agx3_g17g_tile_program_generate(dev->chip, d, store, &fresh) ||
          fresh.length != program->length ||
          memcmp(fresh.bytes, program->bytes, program->length)) {
         mesa_loge("M5 cached tile launch differs from fresh composition");
         return false;
      }
   }
   return true;
}

static struct agx3_reencode_cache agx_bg_eot_g18g_reencode_cache =
   AGX3_REENCODE_CACHE_INITIALIZER;

static void
agx_bg_eot_g18g_reencode_refused(const struct agx_device *dev,
                                 const struct agx3_g17p_graphics_program *p,
                                 const char *why, unsigned at)
{
   mesa_loge("asahi: G18G tile Ioto not re-encoded: %s at +0x%x",
             why ? why : "longer than the program buffer", at);
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr, "G18G_IOTO_REFUSED length=%u data=", p->length);
      for (unsigned i = 0; i < p->length; ++i)
         fprintf(stderr, "%02x", p->bytes[i]);
      fputc('\n', stderr);
   }
}

static bool
agx_bg_eot_g18g_reencode_tile(const struct agx_device *dev,
                              struct agx3_g17p_graphics_program *program,
                              unsigned shader_gprs)
{
   const struct agx3_transcode_ioto ioto = {
      .api_program = false,
      .shader_gprs = shader_gprs,
   };
   const char *why = NULL;
   unsigned at = 0;
   const bool verify = dev->debug & AGX_DBG_VERIFY;
   const unsigned length = program->length;
   struct agx3_g17p_graphics_program agx3c;

   if (verify)
      agx3c = *program;

   switch (agx3_reencode_cached(&agx_bg_eot_g18g_reencode_cache, program, &ioto,
                                &why, &at)) {
   case AGX3_REENCODE_HIT:
      if (verify && (!agx3_reencode_program(&agx3c, &ioto, &why, &at) ||
                     agx3c.length != program->length ||
                     memcmp(agx3c.bytes, program->bytes, program->length)))
         mesa_loge("asahi: G18G tile re-encoding cache hit differs from "
                   "a full re-encoding (length %u)", length);
      return true;
   case AGX3_REENCODE_MISS:
      return true;
   case AGX3_REENCODE_REFUSED:
      agx_bg_eot_g18g_reencode_refused(dev, program, why, at);
      return false;
   case AGX3_REENCODE_UNCACHEABLE:
      break;
   }

   if (agx3_reencode_program(program, &ioto, &why, &at))
      return true;
   agx_bg_eot_g18g_reencode_refused(dev, program, why, at);
   return false;
}

bool
agx_bg_eot_emit_m5_tile_ioto(struct agx_bg_eot_cache *cache,
                             struct agx_bg_eot_shader *shader,
                             const struct agx_tilebuffer_layout *tib,
                             const struct agx3_ioto_uniform_range *ranges,
                             unsigned range_count, bool store,
                             const char *label,
                             const struct agx_bg_eot_ioto_sink *sink,
                             struct agx_bg_eot_ioto_result *out,
                             struct agx3_g17p_graphics_program *out_program_opt)
{
   *out = (struct agx_bg_eot_ioto_result){0};
   struct agx_device *dev = cache->dev;
   const bool agx3d = agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX3D;
   const enum agx_chip ioto_chip = agx3d ? AGX_CHIP_G17G : dev->chip;

   size_t capacity = 0;
   if (!agx3_g17g_graphics_program_bound(ioto_chip, ranges, range_count,
                                         &capacity))
      return false;
   if (agx3d)
      capacity *= 2;

   struct agx_bg_eot_ioto_reservation reservation = {0};
   if (!sink->reserve(sink->ctx, capacity, label, &reservation))
      return false;

   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va,
      .shader_va = shader->ptr + shader->info.main_offset,
      .shader_binary_size = shader->info.binary_size,
      .scratch_size = shader->info.scratch_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = shader->info.agx3_staging_known,
      .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
      .fragment = true,
      .imageblock = {tib->sample_size_B, MAX2(tib->nr_samples, 1),
                     tib->tile_size},
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
   };

   struct agx3_g17p_graphics_program program;
   const bool generated =
      agx3d ? agx3_g17g_tile_program_generate(ioto_chip, &desc, store, &program)
            : agx_bg_eot_m5_tile_program(cache, shader, &desc, store, &program);
   if (!generated)
      return false;

   if (agx3d && !agx_bg_eot_g18g_reencode_tile(dev, &program,
                                               shader->info.nr_gprs))
      return false;

   if (program.length > capacity)
      return false;

   memcpy(reservation.map, program.bytes, program.length);
   const uint64_t published =
      sink->publish(sink->ctx, &reservation, program.length);
   if (!published)
      return false;

   if (out_program_opt)
      *out_program_opt = program;

   *out = (struct agx_bg_eot_ioto_result){
      .program_va = published,
      .primary_storage_B = desc.scratch_size,
      .staging_storage_B = desc.compiler_d,
      .program_bo = reservation.bo,
      .shader_bo = shader->bo,
   };
   return true;
}
