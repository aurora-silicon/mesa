/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host-only unit and concurrency test for the lock-free fast-link cache on
 * hk_shader::linked (hk_shader_fast_link_lookup_lockfree,
 * hk_shader_fast_link_publish_locked, and hk_get_fast_linked).
 */
#include "hk_device.h"
#include "hk_shader.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#undef NDEBUG
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "line %d: check failed: %s\n", __LINE__, #cond);      \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

#define NUM_KEYS 12
#define NUM_THREADS 8
#define NUM_ROUNDS 32
#define CANARY_MAGIC 0x5a3c96a5u

static struct hk_fast_link_key_vs
make_vs_key(unsigned idx)
{
   struct hk_fast_link_key_vs key;
   memset(&key, 0, sizeof(key));
   key.uvs_flat = (uint64_t)idx << 1;
   key.uvs_linear = (uint64_t)(idx & 3u) << 8;
   key.prolog.hw = (idx & 1u) == 0;
   key.prolog.uniform_base = (uint8_t)(idx * 2u);
   key.prolog.static_vi = (idx & 2u) == 0;
   key.prolog.sw_index_size_B = (uint8_t)(idx % 5u);
   key.prolog.attribs[idx % AGX_MAX_VBUFS].stride = (uint16_t)(16u + idx * 4u);
   key.prolog.attribs[idx % AGX_MAX_VBUFS].format = (uint8_t)(idx + 1u);
   return key;
}

static struct hk_fast_link_key_fs
make_fs_key(unsigned idx)
{
   struct hk_fast_link_key_fs key;
   memset(&key, 0, sizeof(key));
   key.nr_samples_shaded = idx & 1u ? 4u : 0u;
   key.prolog.api_sample_mask = (uint8_t)(0xffu - idx);
   key.prolog.statistics = (idx & 2u) != 0;
   key.epilog.nr_samples = (uint8_t)(1u << (idx % 3u));
   key.epilog.rt_formats[0] = (enum pipe_format)(PIPE_FORMAT_R8G8B8A8_UNORM + idx);
   key.epilog.blend.rt[0].colormask = (idx + 1u) & 0xfu;
   key.epilog.blend_constant_base = (uint16_t)(idx * 8u);
   key.epilog.emrt_heap_base = (uint16_t)(idx * 4u);
   return key;
}

static struct hk_linked_shader *
make_fake_linked(mesa_shader_stage stage, unsigned idx)
{
   struct hk_linked_shader *linked = rzalloc(NULL, struct hk_linked_shader);
   CHECK(linked != NULL);
   linked->b.size = (size_t)CANARY_MAGIC + idx;
   linked->b.nr_gprs = (uint16_t)(32u + idx);
   linked->g17p_blend_constant_base = (uint16_t)(idx * 8u);
   linked->g17p_emrt_heap_base = (uint16_t)(idx * 4u);
   linked->sw_indexing = (stage == MESA_SHADER_VERTEX) && ((idx & 1u) != 0);
   linked->fs_color_written = (uint8_t)((idx + 1u) & 0xffu);
   return linked;
}

static void
verify_linked(const struct hk_linked_shader *linked, mesa_shader_stage stage,
              unsigned idx)
{
   CHECK(linked != NULL);
   CHECK(linked->b.size == (size_t)CANARY_MAGIC + idx);
   CHECK(linked->b.nr_gprs == (uint16_t)(32u + idx));
   CHECK(linked->g17p_blend_constant_base == (uint16_t)(idx * 8u));
   CHECK(linked->g17p_emrt_heap_base == (uint16_t)(idx * 4u));
   CHECK(linked->sw_indexing ==
         ((stage == MESA_SHADER_VERTEX) && ((idx & 1u) != 0)));
   CHECK(linked->fs_color_written == (uint8_t)((idx + 1u) & 0xffu));
}

static void
init_shader(struct hk_shader *shader, mesa_shader_stage stage, bool lazy_ht)
{
   memset(shader, 0, sizeof(*shader));
   shader->info.stage = stage;
   simple_mtx_init(&shader->linked.lock, mtx_plain);
   shader->linked.lazy_ht = lazy_ht;
}

static void
destroy_shader(struct hk_shader *shader)
{
   simple_mtx_destroy(&shader->linked.lock);
   memset(shader->linked.fast, 0, sizeof(shader->linked.fast));
   shader->linked.fast_count = 0;
   if (shader->linked.ht) {
      hash_table_foreach(shader->linked.ht, entry) {
         ralloc_free(entry->data);
      }
      _mesa_hash_table_destroy(shader->linked.ht, NULL);
      shader->linked.ht = NULL;
   }
}

struct lock_hold_ctx {
   struct hk_device *dev;
   struct hk_shader *shader;
   mesa_shader_stage stage;
   struct hk_linked_shader *expected[HK_SHADER_FAST_LINK_SLOTS];
   pthread_barrier_t ready;
   atomic_bool done;
};

static void *
lock_free_reader_worker(void *arg)
{
   struct lock_hold_ctx *ctx = arg;
   pthread_barrier_wait(&ctx->ready);

   for (unsigned i = 0; i < HK_SHADER_FAST_LINK_SLOTS; ++i) {
      struct hk_linked_shader *got;
      if (ctx->stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         got = hk_get_fast_linked(ctx->dev, ctx->shader, &key);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         got = hk_get_fast_linked(ctx->dev, ctx->shader, &key);
      }
      CHECK(got == ctx->expected[i]);
      verify_linked(got, ctx->stage, i);
   }

   atomic_store_explicit(&ctx->done, true, memory_order_release);
   return NULL;
}

static void
test_stage_basic_and_lockfree(struct hk_device *dev, mesa_shader_stage stage)
{
   const size_t key_size = hk_fast_link_key_size(stage);
   struct hk_shader shader;

   /* Empty shader with neither ht nor lazy_ht: both lock-free and full lookup
    * must return NULL cleanly. */
   init_shader(&shader, stage, false);
   if (stage == MESA_SHADER_VERTEX) {
      struct hk_fast_link_key_vs key = make_vs_key(0);
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      CHECK(hk_get_fast_linked(dev, &shader, &key) == NULL);
   } else {
      struct hk_fast_link_key_fs key = make_fs_key(0);
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      CHECK(hk_get_fast_linked(dev, &shader, &key) == NULL);
   }
   destroy_shader(&shader);

   /* Populate NUM_KEYS variants (first HK_SHADER_FAST_LINK_SLOTS publish into
    * shader->linked.fast[], the rest live only in shader->linked.ht). */
   init_shader(&shader, stage, true);
   struct hk_linked_shader *created[NUM_KEYS];

   simple_mtx_lock(&shader.linked.lock);
   CHECK(hk_shader_ensure_link_ht_locked(&shader));
   for (unsigned i = 0; i < NUM_KEYS; ++i) {
      created[i] = make_fake_linked(stage, i);
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         hk_shader_fast_link_publish_locked(&shader, created[i], &key, key_size,
                                            true);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         hk_shader_fast_link_publish_locked(&shader, created[i], &key, key_size,
                                            true);
      }
   }
   CHECK(shader.linked.fast_count == HK_SHADER_FAST_LINK_SLOTS);
   simple_mtx_unlock(&shader.linked.lock);

   /* Prove hk_get_fast_linked() hits fast[0..HK_SHADER_FAST_LINK_SLOTS-1]
    * without acquiring shader->linked.lock by holding the mutex on this thread
    * while a second thread executes hk_get_fast_linked(). */
   struct lock_hold_ctx ctx = {
      .dev = dev,
      .shader = &shader,
      .stage = stage,
   };
   for (unsigned i = 0; i < HK_SHADER_FAST_LINK_SLOTS; ++i)
      ctx.expected[i] = created[i];
   atomic_init(&ctx.done, false);
   CHECK(pthread_barrier_init(&ctx.ready, NULL, 2) == 0);

   pthread_t reader;
   CHECK(pthread_create(&reader, NULL, lock_free_reader_worker, &ctx) == 0);

   simple_mtx_lock(&shader.linked.lock);
   pthread_barrier_wait(&ctx.ready);
   /* Wait for reader to finish ALL lock-free lookups while we hold the lock. */
   CHECK(pthread_join(reader, NULL) == 0);
   CHECK(atomic_load_explicit(&ctx.done, memory_order_acquire));
   simple_mtx_unlock(&shader.linked.lock);
   pthread_barrier_destroy(&ctx.ready);

   /* Verify lock-free lookup hits 0..SLOTS-1, misses SLOTS..NUM_KEYS-1, and
    * hk_get_fast_linked() hits ALL 0..NUM_KEYS-1 via fast path or hash table. */
   for (unsigned i = 0; i < NUM_KEYS; ++i) {
      struct hk_linked_shader *lf, *full;
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         lf = hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size);
         full = hk_get_fast_linked(dev, &shader, &key);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         lf = hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size);
         full = hk_get_fast_linked(dev, &shader, &key);
      }
      if (i < HK_SHADER_FAST_LINK_SLOTS)
         CHECK(lf == created[i]);
      else
         CHECK(lf == NULL);
      CHECK(full == created[i]);
      verify_linked(full, stage, i);
   }

   /* Single-bit key perturbation must miss the lock-free cache. */
   if (stage == MESA_SHADER_VERTEX) {
      struct hk_fast_link_key_vs key = make_vs_key(0);
      key.uvs_flat ^= 1ull << 63;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      key = make_vs_key(0);
      key.prolog.hw = !key.prolog.hw;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      key = make_vs_key(0);
      key.prolog.attribs[0].stride ^= 1u;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
   } else {
      struct hk_fast_link_key_fs key = make_fs_key(0);
      key.nr_samples_shaded ^= 1u;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      key = make_fs_key(0);
      key.prolog.statistics = !key.prolog.statistics;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
      key = make_fs_key(0);
      key.epilog.blend.rt[0].colormask ^= 1u;
      CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) == NULL);
   }

   destroy_shader(&shader);
}

struct stress_ctx {
   struct hk_device *dev;
   struct hk_shader shader;
   mesa_shader_stage stage;
   pthread_barrier_t start;
   atomic_uint created_count;
   struct hk_linked_shader *seen[NUM_THREADS][NUM_KEYS];
};

struct stress_arg {
   struct stress_ctx *ctx;
   unsigned tid;
};

static struct hk_linked_shader *
lookup_or_create_test(struct stress_ctx *ctx, unsigned idx)
{
   struct hk_shader *shader = &ctx->shader;
   const size_t key_size = hk_fast_link_key_size(ctx->stage);

   union {
      struct hk_fast_link_key_vs vs;
      struct hk_fast_link_key_fs fs;
   } key;
   if (ctx->stage == MESA_SHADER_VERTEX)
      key.vs = make_vs_key(idx);
   else
      key.fs = make_fs_key(idx);

   struct hk_linked_shader *linked =
      hk_shader_fast_link_lookup_lockfree(shader, &key, key_size);
   if (linked)
      return linked;

   simple_mtx_lock(&shader->linked.lock);
   CHECK(hk_shader_ensure_link_ht_locked(shader));
   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, &key);
   if (ent) {
      linked = ent->data;
   } else {
      linked = make_fake_linked(ctx->stage, idx);
      hk_shader_fast_link_publish_locked(shader, linked, &key, key_size, true);
      atomic_fetch_add_explicit(&ctx->created_count, 1, memory_order_relaxed);
   }
   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static void *
stress_worker(void *arg_)
{
   struct stress_arg *arg = arg_;
   struct stress_ctx *ctx = arg->ctx;
   const unsigned tid = arg->tid;

   pthread_barrier_wait(&ctx->start);

   /* Pass 1: concurrent miss/publish race across all NUM_KEYS in per-thread
    * order so multiple threads collide on both fast[] slots and ht entries. */
   for (unsigned iter = 0; iter < 64; ++iter) {
      for (unsigned k = 0; k < NUM_KEYS; ++k) {
         unsigned idx = (k + tid * 3u + iter) % NUM_KEYS;
         struct hk_linked_shader *linked = lookup_or_create_test(ctx, idx);
         verify_linked(linked, ctx->stage, idx);
         if (!ctx->seen[tid][idx])
            ctx->seen[tid][idx] = linked;
         else
            CHECK(ctx->seen[tid][idx] == linked);
      }
   }

   /* Pass 2: verify real hk_get_fast_linked() agrees for all published keys. */
   for (unsigned idx = 0; idx < NUM_KEYS; ++idx) {
      struct hk_linked_shader *got;
      if (ctx->stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(idx);
         got = hk_get_fast_linked(ctx->dev, &ctx->shader, &key);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(idx);
         got = hk_get_fast_linked(ctx->dev, &ctx->shader, &key);
      }
      CHECK(got == ctx->seen[tid][idx]);
      verify_linked(got, ctx->stage, idx);
   }

   return NULL;
}

static void
test_stage_concurrent(struct hk_device *dev, mesa_shader_stage stage)
{
   struct stress_ctx *ctx = calloc(1, sizeof(*ctx));
   CHECK(ctx != NULL);

   for (unsigned r = 0; r < NUM_ROUNDS; ++r) {
      memset(ctx, 0, sizeof(*ctx));
      ctx->dev = dev;
      ctx->stage = stage;
      init_shader(&ctx->shader, stage, true);
      atomic_init(&ctx->created_count, 0);
      CHECK(pthread_barrier_init(&ctx->start, NULL, NUM_THREADS) == 0);

      pthread_t threads[NUM_THREADS];
      struct stress_arg args[NUM_THREADS];
      for (unsigned t = 0; t < NUM_THREADS; ++t) {
         args[t] = (struct stress_arg){ .ctx = ctx, .tid = t };
         CHECK(pthread_create(&threads[t], NULL, stress_worker, &args[t]) == 0);
      }
      for (unsigned t = 0; t < NUM_THREADS; ++t)
         CHECK(pthread_join(threads[t], NULL) == 0);

      pthread_barrier_destroy(&ctx->start);

      CHECK(atomic_load_explicit(&ctx->created_count, memory_order_acquire) ==
            NUM_KEYS);
      CHECK(ctx->shader.linked.fast_count == HK_SHADER_FAST_LINK_SLOTS);
      for (unsigned k = 0; k < NUM_KEYS; ++k) {
         struct hk_linked_shader *canonical = ctx->seen[0][k];
         verify_linked(canonical, stage, k);
         for (unsigned t = 1; t < NUM_THREADS; ++t)
            CHECK(ctx->seen[t][k] == canonical);
      }

      destroy_shader(&ctx->shader);
   }

   free(ctx);
}

static void
test_stage_prelink_promotion(struct hk_device *dev, mesa_shader_stage stage)
{
   const size_t key_size = hk_fast_link_key_size(stage);
   struct hk_shader shader;
   init_shader(&shader, stage, true);

   /* Prelink HK_SHADER_FAST_LINK_SLOTS + 2 variants into ht without promoting
    * them into fast[] (simulating pipeline/shader-object prelinks that may
    * never be drawn). */
   enum { N = HK_SHADER_FAST_LINK_SLOTS + 2 };
   struct hk_linked_shader *created[N];
   simple_mtx_lock(&shader.linked.lock);
   CHECK(hk_shader_ensure_link_ht_locked(&shader));
   for (unsigned i = 0; i < N; ++i) {
      created[i] = make_fake_linked(stage, i);
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         hk_shader_fast_link_publish_locked(&shader, created[i], &key, key_size,
                                            false);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         hk_shader_fast_link_publish_locked(&shader, created[i], &key, key_size,
                                            false);
      }
   }
   CHECK(shader.linked.fast_count == 0);
   simple_mtx_unlock(&shader.linked.lock);

   /* None of the unused prelinks should occupy fast[]. */
   for (unsigned i = 0; i < N; ++i) {
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               NULL);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               NULL);
      }
   }

   /* Drawing the later variants (indices 2..5) via hk_get_fast_linked() must
    * promote ONLY those drawn variants from ht into fast[]. */
   for (unsigned i = 2; i < N; ++i) {
      struct hk_linked_shader *got;
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         got = hk_get_fast_linked(dev, &shader, &key);
         CHECK(got == created[i]);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               created[i]);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         got = hk_get_fast_linked(dev, &shader, &key);
         CHECK(got == created[i]);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               created[i]);
      }
   }
   CHECK(shader.linked.fast_count == HK_SHADER_FAST_LINK_SLOTS);

   /* Unused prelinks 0 and 1 still resolve through ht without evicting fast[]. */
   for (unsigned i = 0; i < 2; ++i) {
      if (stage == MESA_SHADER_VERTEX) {
         struct hk_fast_link_key_vs key = make_vs_key(i);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               NULL);
         CHECK(hk_get_fast_linked(dev, &shader, &key) == created[i]);
      } else {
         struct hk_fast_link_key_fs key = make_fs_key(i);
         CHECK(hk_shader_fast_link_lookup_lockfree(&shader, &key, key_size) ==
               NULL);
         CHECK(hk_get_fast_linked(dev, &shader, &key) == created[i]);
      }
   }

   destroy_shader(&shader);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(dev != NULL);
   dev->dev.chip = AGX_CHIP_G17P;

   test_stage_basic_and_lockfree(dev, MESA_SHADER_VERTEX);
   test_stage_basic_and_lockfree(dev, MESA_SHADER_FRAGMENT);
   test_stage_prelink_promotion(dev, MESA_SHADER_VERTEX);
   test_stage_prelink_promotion(dev, MESA_SHADER_FRAGMENT);
   test_stage_concurrent(dev, MESA_SHADER_VERTEX);
   test_stage_concurrent(dev, MESA_SHADER_FRAGMENT);

   free(dev);
   puts("PASS: lock-free fast-link cache VS/FS hit/miss, prelink promotion, overflow, and 8-thread stress verified");
   return 0;
}
