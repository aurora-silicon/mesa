/* SPDX-License-Identifier: MIT */
#include "hk_query_wait.h"

#include <errno.h>
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_math.h"

struct hk_query_wait_query {
   struct list_head producers;
   uint64_t epoch;
};

struct hk_query_wait_pool {
   struct hk_query_wait_device *dev;
   unsigned refs;
   uint32_t count;
   bool destroyed;
   struct hk_query_wait_query queries[];
};

struct hk_query_wait_batch {
   unsigned refs;
};

struct hk_query_wait_link {
   struct list_head link;
   struct hk_query_wait_producer *producer;
   struct hk_query_wait_pool *pool;
   uint32_t query;
   uint64_t epoch;
};

enum hk_query_wait_state {
   HK_QUERY_REGISTERED,
   HK_QUERY_QUEUED,
   HK_QUERY_RESERVED,
   HK_QUERY_ACCEPTED,
   HK_QUERY_RETIRED,
   HK_QUERY_FAILED,
};

struct hk_query_wait_producer {
   struct list_head link;
   unsigned refs;
   enum hk_query_wait_state state;
   void *queue;
   const void *recording;
   struct hk_query_wait_batch *batch;
   uint64_t point;
   size_t count;
   struct hk_query_wait_link queries[];
};

static void
pool_unref(struct hk_query_wait_pool *pool)
{
   if (p_atomic_dec_zero(&pool->refs))
      free(pool);
}

static void
batch_unref(struct hk_query_wait_batch *batch)
{
   if (batch && p_atomic_dec_zero(&batch->refs))
      free(batch);
}

static void
producer_unref(struct hk_query_wait_producer *p)
{
   if (p_atomic_dec_zero(&p->refs))
      free(p);
}

static void
unlink_query(struct hk_query_wait_link *q)
{
   if (!q->pool)
      return;
   list_del(&q->link);
   pool_unref(q->pool);
   q->pool = NULL;
}

/* Caller holds dev->mutex. Waiters keep their own producer reference while
 * outside the lock; queue identity and its syncobj live until device teardown.
 */
static void
retire(struct hk_query_wait_device *dev, struct hk_query_wait_producer *p,
       enum hk_query_wait_state state)
{
   p->state = state;
   for (size_t i = 0; i < p->count; i++)
      unlink_query(&p->queries[i]);
   list_del(&p->link);
   batch_unref(p->batch);
   p->batch = NULL;
   producer_unref(p);
   dev->serial++;
   u_cnd_monotonic_broadcast(&dev->changed);
}

bool
hk_query_wait_device_init(struct hk_query_wait_device *dev)
{
   if (mtx_init(&dev->mutex, mtx_plain) != thrd_success)
      return false;
   if (u_cnd_monotonic_init(&dev->changed) != thrd_success) {
      mtx_destroy(&dev->mutex);
      return false;
   }
   list_inithead(&dev->producers);
   dev->serial = 0;
   return true;
}

void
hk_query_wait_device_finish(struct hk_query_wait_device *dev)
{
   assert(list_is_empty(&dev->producers));
   u_cnd_monotonic_destroy(&dev->changed);
   mtx_destroy(&dev->mutex);
}

struct hk_query_wait_pool *
hk_query_wait_pool_create(struct hk_query_wait_device *dev, uint32_t count)
{
   if (count > (SIZE_MAX - sizeof(struct hk_query_wait_pool)) /
                  sizeof(struct hk_query_wait_query))
      return NULL;
   struct hk_query_wait_pool *pool =
      calloc(1, sizeof(*pool) + count * sizeof(pool->queries[0]));
   if (!pool)
      return NULL;
   pool->dev = dev;
   pool->refs = 1;
   pool->count = count;
   for (uint32_t i = 0; i < count; i++)
      list_inithead(&pool->queries[i].producers);
   return pool;
}

static void
reset_locked(struct hk_query_wait_pool *pool, uint32_t first, uint32_t count)
{
   struct hk_query_wait_device *dev = pool->dev;
   assert(first <= pool->count && count <= pool->count - first);
   for (uint32_t i = first; i < first + count; i++) {
      struct hk_query_wait_query *q = &pool->queries[i];
      list_for_each_entry_safe(struct hk_query_wait_link, p, &q->producers,
                               link)
         unlink_query(p);
      q->epoch++;
   }
   list_for_each_entry_safe(struct hk_query_wait_producer, p, &dev->producers,
                            link) {
      if (p->state != HK_QUERY_ACCEPTED)
         continue;
      bool used = false;
      for (size_t i = 0; i < p->count; i++)
         used |= p->queries[i].pool != NULL;
      if (!used)
         retire(dev, p, HK_QUERY_RETIRED);
   }
   dev->serial++;
   u_cnd_monotonic_broadcast(&dev->changed);
}

void
hk_query_wait_host_reset(struct hk_query_wait_pool *pool, uint32_t first,
                         uint32_t count)
{
   struct hk_query_wait_device *dev = pool->dev;
   mtx_lock(&dev->mutex);
   reset_locked(pool, first, count);
   mtx_unlock(&dev->mutex);
}

void
hk_query_wait_pool_destroy(struct hk_query_wait_pool *pool)
{
   if (!pool)
      return;
   mtx_lock(&pool->dev->mutex);
   reset_locked(pool, 0, pool->count);
   pool->destroyed = true;
   mtx_unlock(&pool->dev->mutex);
   pool_unref(pool);
}

bool
hk_query_wait_record(struct util_dynarray *ops, struct hk_query_wait_pool *pool,
                     uint32_t first, uint32_t count, bool reset)
{
   assert(first <= pool->count && count <= pool->count - first);
   if (!count)
      return true;
   struct hk_query_wait_op *op =
      util_dynarray_grow(ops, struct hk_query_wait_op, 1);
   if (!op)
      return false;
   p_atomic_inc(&pool->refs);
   *op = (struct hk_query_wait_op){pool, first, count, reset};
   return true;
}

void
hk_query_wait_recording_finish(struct util_dynarray *ops)
{
   util_dynarray_foreach(ops, struct hk_query_wait_op, op)
      pool_unref(op->pool);
   util_dynarray_fini(ops);
}

struct hk_query_wait_batch *
hk_query_wait_batch_create(void)
{
   struct hk_query_wait_batch *batch = calloc(1, sizeof(*batch));
   if (batch)
      batch->refs = 1;
   return batch;
}

void
hk_query_wait_batch_finish(struct hk_query_wait_device *dev,
                           struct hk_query_wait_batch *batch, bool failed)
{
   mtx_lock(&dev->mutex);
   list_for_each_entry_safe(struct hk_query_wait_producer, p, &dev->producers,
                            link) {
      if (p->batch == batch &&
          (p->state == HK_QUERY_REGISTERED || p->state == HK_QUERY_QUEUED)) {
         if (failed)
            retire(dev, p, HK_QUERY_FAILED);
         else
            p->state = HK_QUERY_QUEUED;
      }
   }
   mtx_unlock(&dev->mutex);
   batch_unref(batch);
}

bool
hk_query_wait_enqueue(struct hk_query_wait_device *dev, void *queue,
                      const void *recording, const struct util_dynarray *ops,
                      struct hk_query_wait_batch *batch)
{
   size_t count = 0;
   util_dynarray_foreach(ops, struct hk_query_wait_op, op) {
      if (!op->reset) {
         if (op->count > (SIZE_MAX - sizeof(struct hk_query_wait_producer)) /
                               sizeof(struct hk_query_wait_link) -
                            count)
            return false;
         count += op->count;
      }
   }
   struct hk_query_wait_producer *p =
      calloc(1, sizeof(*p) + count * sizeof(p->queries[0]));
   if (!p)
      return false;
   p->refs = 1;
   p->queue = queue;
   p->recording = recording;
   p->batch = batch;
   p->count = count;
   p->state = HK_QUERY_REGISTERED;
   if (batch)
      p_atomic_inc(&batch->refs);

   mtx_lock(&dev->mutex);
   size_t n = 0;
   util_dynarray_foreach(ops, struct hk_query_wait_op, op) {
      struct hk_query_wait_pool *pool = op->pool;
      if (pool->destroyed)
         continue;
      for (uint32_t i = op->first; i < op->first + op->count; i++) {
         struct hk_query_wait_query *q = &pool->queries[i];
         if (op->reset) {
            /* Recording/queuing a reset must not erase an available value,
             * or discard an older epoch whose GPU reset has not executed. */
            q->epoch++;
         } else {
            struct hk_query_wait_link *link = &p->queries[n++];
            *link = (struct hk_query_wait_link){
               .producer = p,
               .pool = pool,
               .query = i,
               .epoch = q->epoch,
            };
            p_atomic_inc(&pool->refs);
            list_addtail(&link->link, &q->producers);
         }
      }
   }
   list_addtail(&p->link, &dev->producers);
   dev->serial++;
   u_cnd_monotonic_broadcast(&dev->changed);
   mtx_unlock(&dev->mutex);
   return true;
}

bool
hk_query_wait_claim(struct hk_query_wait_device *dev, void *queue,
                    const void *recording, const struct util_dynarray *ops,
                    struct hk_query_wait_producer **out)
{
   *out = NULL;
   if (!ops->size)
      return true;
   for (;;) {
      mtx_lock(&dev->mutex);
      list_for_each_entry(struct hk_query_wait_producer, p, &dev->producers,
                          link) {
         if ((p->state == HK_QUERY_REGISTERED || p->state == HK_QUERY_QUEUED) &&
             p->queue == queue && p->recording == recording) {
            p->state = HK_QUERY_RESERVED;
            p_atomic_inc(&p->refs);
            *out = p;
            mtx_unlock(&dev->mutex);
            return true;
         }
      }
      mtx_unlock(&dev->mutex);
      /* QueueSubmit2 may have failed after queuing an earlier part of the
       * call. Recreate that part from its still-live original recording. */
      if (!hk_query_wait_enqueue(dev, queue, recording, ops, NULL))
         return false;
   }
}

void
hk_query_wait_publish(struct hk_query_wait_device *dev,
                      struct hk_query_wait_producer *p, uint64_t point,
                      bool accepted)
{
   if (!p)
      return;
   mtx_lock(&dev->mutex);
   assert(p->state == HK_QUERY_RESERVED);
   bool used = false;
   for (size_t i = 0; i < p->count; i++)
      used |= p->queries[i].pool != NULL;
   if (accepted && used) {
      assert(point);
      p->point = point;
      p->state = HK_QUERY_ACCEPTED;
      batch_unref(p->batch);
      p->batch = NULL;
      dev->serial++;
      u_cnd_monotonic_broadcast(&dev->changed);
   } else {
      retire(dev, p, accepted ? HK_QUERY_RETIRED : HK_QUERY_FAILED);
   }
   mtx_unlock(&dev->mutex);
   producer_unref(p);
}

void
hk_query_wait_prune(struct hk_query_wait_device *dev, void *queue,
                    uint64_t completed)
{
   mtx_lock(&dev->mutex);
   list_for_each_entry_safe(struct hk_query_wait_producer, p, &dev->producers,
                            link) {
      if (p->queue == queue && p->state == HK_QUERY_ACCEPTED &&
          p->point <= completed)
         retire(dev, p, HK_QUERY_RETIRED);
   }
   mtx_unlock(&dev->mutex);
}

void
hk_query_wait_finish_queue(struct hk_query_wait_device *dev, void *queue)
{
   mtx_lock(&dev->mutex);
   list_for_each_entry_safe(struct hk_query_wait_producer, p, &dev->producers,
                            link) {
      if (p->queue == queue)
         retire(dev, p, HK_QUERY_FAILED);
   }
   mtx_unlock(&dev->mutex);
}

bool
hk_query_wait_quiescent(struct hk_query_wait_device *dev, uint64_t *serial)
{
   mtx_lock(&dev->mutex);
   *serial = dev->serial;
   bool idle = true;
   list_for_each_entry(struct hk_query_wait_producer, p, &dev->producers,
                       link) {
      if (p->state == HK_QUERY_REGISTERED || p->state == HK_QUERY_QUEUED ||
          p->state == HK_QUERY_RESERVED) {
         idle = false;
         break;
      }
   }
   mtx_unlock(&dev->mutex);
   return idle;
}

VkResult
hk_query_wait_available(struct hk_query_wait_pool *pool, uint32_t query,
                        bool victims, bool *never,
                        const struct hk_query_wait_ops *ops, void *data)
{
   struct hk_query_wait_device *dev = pool->dev;
   *never = false;
   for (;;) {
      if (ops->available(data))
         return VK_SUCCESS;
      VkResult status = ops->status(data);
      if (status != VK_SUCCESS)
         return status;
      if (victims && ops->idle(data)) {
         if (ops->available(data))
            return VK_SUCCESS;
         status = ops->status(data);
         if (status != VK_SUCCESS)
            return status;
         *never = true;
         return VK_SUCCESS;
      }

      const uint64_t now = ops->now ? ops->now(data) : os_time_get_nano();
      const uint64_t deadline =
         MIN2(now, INT64_MAX - HK_QUERY_WAIT_INTERVAL_NS) +
         HK_QUERY_WAIT_INTERVAL_NS;
      struct hk_query_wait_producer *producer = NULL;
      mtx_lock(&dev->mutex);
      list_for_each_entry(struct hk_query_wait_link, q,
                          &pool->queries[query].producers, link) {
         if (q->producer->state == HK_QUERY_ACCEPTED)
            producer = q->producer;
         if (producer)
            break;
      }
      if (producer) {
         p_atomic_inc(&producer->refs);
         mtx_unlock(&dev->mutex);
         const int ret =
            ops->completion(data, producer->queue, producer->point, deadline);
         mtx_lock(&dev->mutex);
         if (!ret && producer->state == HK_QUERY_ACCEPTED)
            retire(dev, producer, HK_QUERY_RETIRED);
         mtx_unlock(&dev->mutex);
         producer_unref(producer);
         if (ret && ret != -ETIME && ret != -EINTR)
            return ops->error(data, ret);
      } else {
         int ret;
         if (ops->changed) {
            ret = ops->changed(data, dev, deadline);
         } else {
            const struct timespec ts = {
               .tv_sec = deadline / 1000000000,
               .tv_nsec = deadline % 1000000000,
            };
            ret = u_cnd_monotonic_timedwait(&dev->changed, &dev->mutex, &ts);
         }
         mtx_unlock(&dev->mutex);
         if (ret != thrd_success && ret != thrd_timedout)
            return ops->error(data, -EINVAL);
      }
      /* Error fences can signal successfully. Do not turn a newly visible
       * result into success without checking the device after the wake. */
      status = ops->status(data);
      if (status != VK_SUCCESS)
         return status;
   }
}
