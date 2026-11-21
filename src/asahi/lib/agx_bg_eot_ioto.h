/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/lib/agx_bg_eot.h"
#include "asahi/lib/agx_bg_eot_key.h"
#include "asahi/lib/agx_tilebuffer.h"

struct agx_bo;
struct agx_device;
struct agx_bg_eot_ioto_cache;
struct agx_bg_eot_ioto_cache_entry;

#define AGX_BG_EOT_DESC_SIZE              32u
#define AGX_BG_EOT_IOTO_CACHE_ENTRIES     256u
#define AGX_BG_EOT_IOTO_CACHE_CHUNK       (64u * 1024u)
#define AGX_BG_EOT_IOTO_CACHE_MAX_CHUNKS  8u
#define AGX_BG_EOT_IOTO_CACHE_PROGRAM_MAX 512u
#define AGX_BG_EOT_IOTO_CACHE_DATA_MAX    512u
#define AGX_BG_EOT_IOTO_CACHE_VARIANTS    4u

/*
 * Result of building or retrieving a BG/EOT Ioto launch program.
 *
 * `program_va` is the raw 64-bit GPU VA inside the USC window: Vulkan uses it
 * directly, while Gallium converts it with `agx_usc_addr(dev, res.program_va)`.
 * `program_bo`, `data_bo`, and `shader_bo` record every BO referenced by the
 * emitted program so frontends with per-batch BO lists (Gallium) or command
 * buffer retention (Vulkan) can retain them on both hits and misses.
 */
struct agx_bg_eot_ioto_result {
   uint64_t program_va;
   uint64_t rsrc_spec;
   uint32_t primary_storage_B;
   uint32_t staging_storage_B;
   struct agx_bo *program_bo;
   struct agx_bo *data_bo;
   struct agx_bo *shader_bo;
};

struct agx_bg_eot_ioto_reservation {
   void *map;
   uint64_t va;
   struct agx_bo *bo;
};

/*
 * Memory sink for Ioto program bytes and their argument/descriptor tables.
 * Backed either by the device-level `agx_bg_eot_ioto_cache` arenas or by a
 * frontend's per-recording / per-batch pool.
 */
struct agx_bg_eot_ioto_sink {
   void *ctx;
   uint64_t (*upload)(void *ctx, const void *data, uint32_t size,
                      uint32_t alignment, struct agx_bo **out_bo);
   bool (*reserve)(void *ctx, size_t capacity, const char *label,
                   struct agx_bg_eot_ioto_reservation *out);
   uint64_t (*publish)(void *ctx, const struct agx_bg_eot_ioto_reservation *r,
                       size_t length);
};

/*
 * Value key for the device-level BG/EOT Ioto cache.
 * Frontends zero-initialize the whole struct (including padding) and fill the
 * tilebuffer layout, operation metadata, clear value, and packed 32-byte
 * hardware descriptors that determine the generated Ioto program and data.
 */
struct agx_bg_eot_ioto_cache_key {
   _Alignas(8) struct agx_tilebuffer_layout tib;
   uint32_t layers;
   uint32_t color_attachment_count;
   uint32_t flags;
   uint32_t view;
   uint32_t samples;
   uint32_t format;
   uint32_t load_op;
   uint32_t store_op;
   uint32_t clear[4];
   uint8_t eot[AGX_BG_EOT_DESC_SIZE];
   uint8_t layered_eot[AGX_BG_EOT_DESC_SIZE];
   uint8_t background[AGX_BG_EOT_DESC_SIZE];
   uint8_t layered_background[AGX_BG_EOT_DESC_SIZE];
   uint32_t pbe_valid;
   uint32_t reserved;
};

typedef bool (*agx_bg_eot_ioto_build_fn)(
   void *ctx, const struct agx_bg_eot_ioto_sink *sink,
   struct agx_bg_eot_ioto_result *out);

struct agx_bg_eot_ioto_cache *
agx_bg_eot_ioto_cache_create(struct agx_device *dev);

void agx_bg_eot_ioto_cache_destroy(struct agx_device *dev,
                                   struct agx_bg_eot_ioto_cache *dc);

/*
 * Look up or insert a pass entry in the device-level cache.
 * Returns the immutable entry pointer (or NULL if the cache is full), copying
 * any already-populated variants into `out_programs` / `out_valid_variants`.
 */
struct agx_bg_eot_ioto_cache_entry *agx_bg_eot_ioto_cache_get(
   struct agx_bg_eot_ioto_cache *dc,
   const struct agx_bg_eot_ioto_cache_key *key,
   struct agx_bg_eot_ioto_result out_programs[AGX_BG_EOT_IOTO_CACHE_VARIANTS],
   uint8_t *out_valid_variants);

/*
 * Retrieve or build one variant (0..AGX_BG_EOT_IOTO_CACHE_VARIANTS-1) of an
 * already-attached cache entry. Returns true if `out` was populated (either
 * from cache or via `build`); returns false if the cache arenas are full and
 * the caller should fall back to a per-recording/per-batch sink.
 */
bool agx_bg_eot_ioto_cache_lookup_or_build(
   struct agx_device *dev, struct agx_bg_eot_ioto_cache *dc,
   struct agx_bg_eot_ioto_cache_entry *entry, unsigned variant,
   agx_bg_eot_ioto_build_fn build, void *ctx,
   struct agx_bg_eot_ioto_result *out, bool *out_in_cache);

/*
 * Single-shot lookup + build by value key (used by frontends without an L1
 * command-buffer cache, such as Gallium). Returns true if `out` was populated
 * from the device cache (hit or newly inserted), or false if the cache is full
 * and the caller should fall back to a per-batch sink.
 */
bool agx_bg_eot_ioto_cache_get_or_build(
   struct agx_device *dev, struct agx_bg_eot_ioto_cache *dc,
   const struct agx_bg_eot_ioto_cache_key *key, unsigned variant,
   agx_bg_eot_ioto_build_fn build, void *ctx,
   struct agx_bg_eot_ioto_result *out, bool *out_in_cache);

bool agx_bg_eot_shader_supported(const struct agx_bg_eot_shader *shader);

bool agx_g17p_bg_eot_ops_uniform(const struct agx_bg_eot_key *key);

uint32_t agx_g17p_pack_unorm8_clear(const float rgba[4]);

/*
 * Inputs to the shared G17P mixed-MRT BG/EOT Ioto program builder.
 * `key` must already be compacted in LOAD -> CLEAR -> STORE order with
 * `key.mixed_mrt = true`.
 */
struct agx_bg_eot_g17p_mixed_desc {
   struct agx_bg_eot_key key;
   bool store;
   unsigned count;
   unsigned reloads;
   /* For store: `count` packed 32-byte PBE descriptors. */
   const uint8_t (*pbe_descs)[AGX_BG_EOT_DESC_SIZE];
   /* For !store && reloads > 0: `reloads` packed 32-byte texture descriptors. */
   const uint8_t (*tex_descs)[AGX_BG_EOT_DESC_SIZE];
   /* Either pre-uploaded sampler VA (Vulkan rodata) or sampler bytes to upload. */
   uint64_t sampler_va;
   const void *sampler_data;
   uint32_t sampler_size;
   /* For !store && count > reloads: `count - reloads` 16-byte clear values. */
   const uint32_t (*clear_values)[4];
};

bool agx_bg_eot_build_g17p_mixed(struct agx_bg_eot_cache *cache,
                                 const struct agx_bg_eot_g17p_mixed_desc *in,
                                 const struct agx_bg_eot_ioto_sink *sink,
                                 struct agx_bg_eot_ioto_result *out);

/*
 * Inputs to the shared G17P uniform BG/EOT Ioto program builder (covering
 * empty/attachmentless, single-target RT0, dual WBOIT/RGBA8/R11G11B10, multi
 * RGBA32/R32_QUAD/RGBA8, wide RGBA8 MRT, and store_and_resolve).
 */
struct agx_bg_eot_g17p_uniform_desc {
   struct agx_bg_eot_key key;
   bool store;
   enum agx_bg_eot_op op;
   enum agx_g17p_mrt_layout mrt;
   unsigned attachment_count;
   unsigned slot_count;
   uint32_t attachment_mask;
   unsigned first_rt;
   bool dual_mrt_active;
   bool multi_mrt_active;
   bool wide_mrt;
   /* Optional upper bound on compiled shader binary size (0 = no extra bound). */
   uint32_t max_shader_binary_size;
   /* For store && op == AGX_EOT_STORE: `attachment_count` packed 32B PBEs. */
   const uint8_t (*pbe_descs)[AGX_BG_EOT_DESC_SIZE];
   /* For !store && op == AGX_BG_CLEAR: pre-uploaded VA or `slot_count` 16B values. */
   uint64_t clear_value_va;
   const uint32_t (*clear_values)[4];
   /* For !store && op == AGX_BG_LOAD: `tex_desc_count` packed 32B descriptors
    * (sparse at 2*rt, or resolve target at slot 2 for preserve_resolve_outside). */
   const uint8_t (*tex_descs)[AGX_BG_EOT_DESC_SIZE];
   unsigned tex_desc_count;
   uint32_t resolve_bounds[4];
   uint64_t sampler_va;
   const void *sampler_data;
   uint32_t sampler_size;
};

bool agx_bg_eot_build_g17p_uniform(
   struct agx_bg_eot_cache *cache,
   const struct agx_bg_eot_g17p_uniform_desc *in,
   const struct agx_bg_eot_ioto_sink *sink,
   struct agx_bg_eot_ioto_result *out);

/*
 * Cached M5 tile launch program composer (caches immutable relocatable
 * templates on `shader->m5_tile_templates` under `cache->lock`).
 */
bool agx_bg_eot_m5_tile_program(
   struct agx_bg_eot_cache *cache, struct agx_bg_eot_shader *shader,
   const struct agx3_g17p_graphics_program_desc *desc, bool store,
   struct agx3_g17p_graphics_program *program);

/*
 * Shared M5 / G18G tile Ioto emitter: computes program size bound, reserves via
 * `sink`, composes (using cached M5 templates on M5 or G17G->AGX3D transcode
 * on G18G), copies, and publishes. Optionally returns the emitted bytes in
 * `out_program_opt` for debug tracing.
 */
bool agx_bg_eot_emit_m5_tile_ioto(
   struct agx_bg_eot_cache *cache, struct agx_bg_eot_shader *shader,
   const struct agx_tilebuffer_layout *tib,
   const struct agx3_ioto_uniform_range *ranges, unsigned range_count,
   bool store, const char *label, const struct agx_bg_eot_ioto_sink *sink,
   struct agx_bg_eot_ioto_result *out,
   struct agx3_g17p_graphics_program *out_program_opt);
