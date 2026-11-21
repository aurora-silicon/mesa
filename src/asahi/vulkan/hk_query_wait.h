/* SPDX-License-Identifier: MIT */
#pragma once

#include "util/cnd_monotonic.h"
#include "util/list.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

/* Match the existing HK completion/loss-check interval. This is a bound on
 * one blocking operation, not a deadline for producing a query result. */
#define HK_QUERY_WAIT_INTERVAL_NS UINT64_C(100000000)

/* One initial spin per host wait, never per completion timeout. The poll cap
 * also bounds work if the clock does not advance. These are candidate tuning
 * parameters, not measured hardware latency/CPU guarantees. */
#define HK_QUERY_WAIT_SPIN_NS UINT64_C(50000)
#define HK_QUERY_WAIT_SPIN_POLLS 1024

/* A queue timeline point covers the entire submission, not just the query
 * write. Recheck availability sooner while that point is still pending. */
#define HK_QUERY_WAIT_COMPLETION_NS UINT64_C(1000000)

struct hk_query_wait_pool;
struct hk_query_wait_batch;
struct hk_query_wait_producer;

struct hk_query_wait_device {
   mtx_t mutex;
   struct u_cnd_monotonic changed;
   struct list_head producers;
   uint64_t serial;
};

struct hk_query_wait_op {
   struct hk_query_wait_pool *pool;
   uint32_t first, count;
   bool reset;
};

bool hk_query_wait_device_init(struct hk_query_wait_device *dev);
void hk_query_wait_device_finish(struct hk_query_wait_device *dev);
struct hk_query_wait_pool *
hk_query_wait_pool_create(struct hk_query_wait_device *dev, uint32_t count);
void hk_query_wait_pool_destroy(struct hk_query_wait_pool *pool);
void hk_query_wait_host_reset(struct hk_query_wait_pool *pool, uint32_t first,
                              uint32_t count);
bool hk_query_wait_record(struct util_dynarray *ops,
                          struct hk_query_wait_pool *pool, uint32_t first,
                          uint32_t count, bool reset);
void hk_query_wait_recording_finish(struct util_dynarray *ops);

struct hk_query_wait_batch *hk_query_wait_batch_create(void);
void hk_query_wait_batch_finish(struct hk_query_wait_device *dev,
                                struct hk_query_wait_batch *batch, bool failed);
bool hk_query_wait_enqueue(struct hk_query_wait_device *dev, void *queue,
                           const void *recording,
                           const struct util_dynarray *ops,
                           struct hk_query_wait_batch *batch);
bool hk_query_wait_claim(struct hk_query_wait_device *dev, void *queue,
                         const void *recording, const struct util_dynarray *ops,
                         struct hk_query_wait_producer **out);
void hk_query_wait_publish(struct hk_query_wait_device *dev,
                           struct hk_query_wait_producer *producer,
                           uint64_t point, bool accepted);
void hk_query_wait_prune(struct hk_query_wait_device *dev, void *queue,
                         uint64_t completed);
void hk_query_wait_finish_queue(struct hk_query_wait_device *dev, void *queue);
bool hk_query_wait_quiescent(struct hk_query_wait_device *dev,
                             uint64_t *serial);

/* Completion is only a blocking opportunity. Availability is authoritative,
 * including when it precedes later host-dependent work in the submission.
 * The clock and condition-wait overrides are for deterministic host tests. */
struct hk_query_wait_ops {
   bool (*available)(void *data);
   VkResult (*status)(void *data);
   bool (*idle)(void *data);
   int (*completion)(void *data, void *queue, uint64_t point,
                     uint64_t deadline);
   VkResult (*error)(void *data, int error);
   uint64_t (*now)(void *data);
   int (*changed)(void *data, struct hk_query_wait_device *dev,
                  uint64_t deadline);
};

VkResult hk_query_wait_available(struct hk_query_wait_pool *pool,
                                 uint32_t query, bool victims, bool *never,
                                 const struct hk_query_wait_ops *ops,
                                 void *data);
