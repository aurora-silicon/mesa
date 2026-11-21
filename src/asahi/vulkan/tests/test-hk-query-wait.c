/* SPDX-License-Identifier: MIT */
#include "hk_query_wait.h"

#include <errno.h>
#include <stdio.h>
#include "util/os_time.h"

#undef NDEBUG
#include <assert.h>

static int allocation_failure;
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *ptr, size_t size);
void *__wrap_calloc(size_t count, size_t size);
void *__wrap_realloc(void *ptr, size_t size);

void *
__wrap_calloc(size_t count, size_t size)
{
   if (allocation_failure && --allocation_failure == 0)
      return NULL;
   return __real_calloc(count, size);
}

void *
__wrap_realloc(void *ptr, size_t size)
{
   if (allocation_failure && --allocation_failure == 0)
      return NULL;
   return __real_realloc(ptr, size);
}

struct fixture {
   struct hk_query_wait_device dev;
   struct hk_query_wait_pool *pool;
   struct util_dynarray recording;
   int queue[2];
   void *order_queue[2];
   uint64_t point[2], completion_at[2];
   uint64_t now, ready_at, loss_at, publish_at;
   uint64_t watchdog_at;
   uint64_t clock_step;
   unsigned initial_polls;
   int completion_error, changed_error;
   unsigned available, statuses, completions, changes, errors;
   bool race_ready, race_loss, idle;
   bool publish_pending;
   unsigned publish_queue;
   bool idle_after_completion;
   uint64_t event_at;
   void (*event)(struct fixture *f);
   void (*deadline_event)(struct fixture *f);
   unsigned deadline_statuses;
   struct hk_query_wait_producer *reserved;
   struct hk_query_wait_batch *batch;
};

static void
init(struct fixture *f)
{
   memset(f, 0, sizeof(*f));
   assert(hk_query_wait_device_init(&f->dev));
   f->order_queue[0] = &f->queue[0];
   f->order_queue[1] = &f->queue[1];
   f->pool = hk_query_wait_pool_create(&f->dev, 4);
   assert(f->pool);
   f->ready_at = f->loss_at = f->publish_at = UINT64_MAX;
   f->watchdog_at = UINT64_MAX;
   f->event_at = UINT64_MAX;
   f->completion_at[0] = f->completion_at[1] = UINT64_MAX;
   assert(hk_query_wait_record(&f->recording, f->pool, 0, 1, false));
}

static void
finish(struct fixture *f)
{
   hk_query_wait_pool_destroy(f->pool);
   hk_query_wait_recording_finish(&f->recording);
   hk_query_wait_finish_queue(&f->dev, &f->queue[0]);
   hk_query_wait_finish_queue(&f->dev, &f->queue[1]);
   assert(list_is_empty(&f->dev.producers));
   hk_query_wait_device_finish(&f->dev);
}

static void
enqueue(struct fixture *f, unsigned q)
{
   struct hk_query_wait_batch *b = hk_query_wait_batch_create();
   assert(b);
   assert(hk_query_wait_enqueue(&f->dev, &f->queue[q], &f->recording,
                                &f->recording, b, &f->queue[q]));
   hk_query_wait_batch_finish(&f->dev, b, false);
}

static struct hk_query_wait_producer *
claim(struct fixture *f, unsigned q)
{
   struct hk_query_wait_producer *p;
   assert(hk_query_wait_claim(&f->dev, &f->queue[q], &f->recording,
                              &f->recording, &p, f->order_queue[q]));
   assert(p);
   return p;
}

static void
accept(struct fixture *f, unsigned q, uint64_t point)
{
   f->point[q] = point;
   hk_query_wait_publish(&f->dev, claim(f, q), point, true);
}

static bool
available(void *data)
{
   struct fixture *f = data;
   f->available++;
   if (!f->completions && !f->changes)
      f->initial_polls++;
   assert(f->now <= UINT64_C(5000000000));
   return f->now >= f->ready_at;
}

static VkResult
status(void *data)
{
   struct fixture *f = data;
   assert(mtx_trylock(&f->dev.mutex) == thrd_success);
   mtx_unlock(&f->dev.mutex);
   f->statuses++;
   if (f->now == HK_QUERY_WAIT_RETIRED_NS && ++f->deadline_statuses == 3 &&
       f->deadline_event) {
      void (*event)(struct fixture *) = f->deadline_event;
      f->deadline_event = NULL;
      event(f);
   }
   /* A diagnostic stop, deliberately distinct from device loss. */
   if (f->now >= f->watchdog_at)
      return VK_ERROR_UNKNOWN;
   return f->now >= f->loss_at ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static bool
idle(void *data)
{
   struct fixture *f = data;
   uint64_t serial;
   if (!hk_query_wait_quiescent(&f->dev, &serial) || !f->idle)
      return false;
   if (f->race_ready)
      f->ready_at = 0;
   if (f->race_loss)
      f->loss_at = 0;
   return true;
}

static uint64_t
now(void *data)
{
   struct fixture *f = data;
   uint64_t value = f->now;
   f->now += f->clock_step;
   return value;
}

static int
completion(void *data, void *queue, uint64_t point, uint64_t deadline)
{
   struct fixture *f = data;
   unsigned q = queue == &f->queue[0] ? 0 : 1;
   assert(queue == &f->queue[q]);
   assert(point == f->point[q] && point != 0);
   assert(deadline > f->now);
   assert(deadline - f->now <= UINT64_C(1000000));
   f->completions++;
   if (f->completion_error) {
      if (f->completion_error == -EINTR)
         f->loss_at = f->now;
      return f->completion_error;
   }
   if (f->completion_at[q] <= deadline) {
      f->now = f->completion_at[q];
      if (f->idle_after_completion)
         f->idle = true;
      return 0;
   }
   f->now = deadline;
   return -ETIME;
}

static int
changed(void *data, struct hk_query_wait_device *dev, uint64_t deadline)
{
   struct fixture *f = data;
   assert(deadline > f->now);
   assert(deadline - f->now <= HK_QUERY_WAIT_INTERVAL_NS);
   f->changes++;
   if (f->changed_error)
      return f->changed_error;
   if (f->event && deadline >= f->event_at) {
      f->now = f->event_at;
      void (*event)(struct fixture *) = f->event;
      f->event = NULL;
      mtx_unlock(&dev->mutex);
      event(f);
      mtx_lock(&dev->mutex);
      return thrd_success;
   }
   if (f->publish_pending && deadline >= f->publish_at) {
      f->now = f->publish_at;
      f->publish_pending = false;
      mtx_unlock(&dev->mutex);
      accept(f, f->publish_queue, f->point[f->publish_queue] + 7);
      mtx_lock(&dev->mutex);
      return thrd_success;
   }
   f->now = deadline;
   return thrd_timedout;
}

static VkResult
error(void *data, int err)
{
   struct fixture *f = data;
   assert(err);
   assert(mtx_trylock(&f->dev.mutex) == thrd_success);
   mtx_unlock(&f->dev.mutex);
   f->errors++;
   f->loss_at = f->now;
   return VK_ERROR_DEVICE_LOST;
}

static const struct hk_query_wait_ops ops = {
   .available = available,
   .status = status,
   .idle = idle,
   .completion = completion,
   .error = error,
   .now = now,
   .changed = changed,
};

static VkResult
wait(struct fixture *f, bool victims, bool *never)
{
   return hk_query_wait_available(f->pool, 0, victims, never, &ops, f);
}

static void
test_waits(void)
{
   struct fixture f;
   bool never;
   init(&f);
   f.ready_at = 0;
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.available == 1 && !f.completions && !f.changes);
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.ready_at = f.completion_at[0] = UINT64_C(2200000000);
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.now == f.ready_at && f.completions == 2200 && !f.errors);
   assert(f.initial_polls == 1);
   assert(f.available <= f.initial_polls + f.completions);
   printf("delayed producer: success at 2200 ms, 2200 bounded DRM waits\n");
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 11);
   f.ready_at = 1;
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.now == UINT64_C(1000000) && f.completions == 1);
   assert(f.completion_at[0] == UINT64_MAX);
   printf("availability before host-dependent tail: extra 999999 ns\n");
   finish(&f);

   for (unsigned state = 0; state < 3; state++) {
      init(&f);
      enqueue(&f, 0);
      struct hk_query_wait_producer *reserved = NULL;
      if (state == 1)
         reserved = claim(&f, 0);
      else if (state == 2)
         accept(&f, 0, 3);
      f.loss_at = 1;
      assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
      assert(f.now == UINT64_C(1000000) && !f.errors);
      assert(f.completions == (state == 2));
      if (reserved)
         hk_query_wait_publish(&f.dev, reserved, 0, false);
      finish(&f);
   }
   printf("queued/reserved/accepted loss: <1 ms (fake time)\n");

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.ready_at = f.loss_at = f.completion_at[0] = 1;
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
   assert(f.completions == 1 && f.now == 1);
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   f.publish_pending = true;
   f.publish_at = UINT64_C(250000000);
   f.ready_at = f.completion_at[0] = UINT64_C(2200000000);
   assert(wait(&f, false, &never) == VK_SUCCESS);
   assert(f.changes == 250 && f.completions == 1950 && !f.errors);
   assert(f.now == f.ready_at);
   finish(&f);
}

/* Availability is checked after blocking, without an initial or repeated
 * spin. Advancing and frozen clocks both exercise the production helper. */
static void
test_rechecks_and_errors(void)
{
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   f.clock_step = 1000;
   f.ready_at = 20000;
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(!f.completions && f.changes == 1);
   assert(f.now == UINT64_C(1000000) && f.initial_polls == 1);
   printf("queued availability: returned after blocking at %llu ns, no spin\n",
          (unsigned long long)f.now);
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.clock_step = 1000;
   f.ready_at = f.completion_at[0] = UINT64_C(2200000000);
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.initial_polls == 1);
   /* One availability test per subsequent wait, no repeated spinning. */
   assert(f.available <= f.initial_polls + f.completions);
   assert(f.now >= f.ready_at && f.now <= f.ready_at + 1000);
   printf("long wait: %u initial polls, %u completion waits, %u total polls\n",
          f.initial_polls, f.completions, f.available);
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   f.ready_at = UINT64_C(2200000000);
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.now == f.ready_at && f.changes == 2200 && !f.completions);
   assert(f.initial_polls == 1 && f.available == f.changes + 1);
   puts("long queued work: 2200 condition waits, no spin or total deadline");
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   f.clock_step = 1000;
   f.ready_at = f.loss_at = 20000;
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
   assert(!f.completions && f.changes == 1);
   finish(&f);

   for (unsigned mode = 0; mode < 3; mode++) {
      init(&f);
      if (mode != 2) {
         enqueue(&f, 0);
         accept(&f, 0, 1);
         f.completion_error = mode ? -EINTR : -EIO;
      } else {
         f.changed_error = thrd_error;
      }
      /* EINTR retries and then observes loss; fatal errors use error(). */
      assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
      assert(f.errors == (mode != 1));
      finish(&f);
   }
}

/* A live writer can publish availability before the submit callback has
 * published its accepted completion point. Test each unaccepted state with
 * a frozen clock, including loss arriving with availability on the wake. */
static void
test_unaccepted_availability(unsigned state)
{
   for (unsigned lost = 0; lost < 2; lost++) {
      struct fixture f;
      bool never;
      init(&f);
      struct hk_query_wait_batch *batch = NULL;
      struct hk_query_wait_producer *reserved = NULL;
      if (state == 0) {
         batch = hk_query_wait_batch_create();
         assert(batch);
         assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording,
                                      &f.recording, batch, &f.queue[0]));
      } else {
         enqueue(&f, 0);
         if (state == 2)
            reserved = claim(&f, 0);
      }
      f.ready_at = 1;
      if (lost)
         f.loss_at = 1;
      assert(wait(&f, false, &never) ==
             (lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS));
      assert(!never && f.now == UINT64_C(1000000));
      assert(f.changes == 1 && !f.completions && !f.errors);
      assert(f.initial_polls == 1);
      if (reserved)
         hk_query_wait_publish(&f.dev, reserved, 0, false);
      if (batch)
         hk_query_wait_batch_finish(&f.dev, batch, false);
      finish(&f);
   }
   printf("%s availability/loss: 999999 ns extra blocking, no accepted point\n",
          state == 0 ? "registered" : state == 1 ? "queued" : "reserved");
}

static void
test_idle_rechecks(void)
{
   for (unsigned unrelated = 0; unrelated < 2; unrelated++) {
      struct fixture f;
      bool never;
      init(&f);
      if (unrelated) {
         hk_query_wait_recording_finish(&f.recording);
         assert(hk_query_wait_record(&f.recording, f.pool, 1, 1, false));
         enqueue(&f, 0);
         accept(&f, 0, 1);
      }
      f.ready_at = 1;
      assert(wait(&f, false, &never) == VK_SUCCESS && !never);
      assert(f.now == HK_QUERY_WAIT_INTERVAL_NS && f.changes == 1);
      assert(!f.completions && f.initial_polls == 1);
      finish(&f);
   }
   puts("idle/unrelated-slot producer: 100 ms fallback, no spin or false loss");
}

static void
test_future_producers(void)
{
   struct fixture f;
   bool never;
   init(&f);
   f.idle = true;
   f.publish_pending = true;
   f.publish_at = UINT64_C(2500000000);
   f.ready_at = f.completion_at[0] = UINT64_C(2600000000);
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.now == f.ready_at && f.changes == 25 && f.completions == 100);
   printf("initially idle: future producer registered at 2500 ms, success at 2600 ms\n");
   finish(&f);

   init(&f);
   f.idle = true;
   f.publish_pending = true;
   f.publish_at = UINT64_C(2500000000);
   f.ready_at = UINT64_C(2600000000);
   /* Applying the existing victim rule universally would return before the
    * exact same future producer. This models the proposed rule, not a GPU. */
   assert(wait(&f, true, &never) == VK_SUCCESS && never);
   assert(f.now == 0 && f.publish_pending && !f.completions && !f.changes);
   printf("universal idle rule counterexample: never=true before future writer\n");
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.completion_at[0] = 1;
   f.idle_after_completion = true;
   f.loss_at = UINT64_C(300000000);
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST && !never);
   assert(f.now > f.completion_at[0] && f.completions == 1 && f.changes == 3);
   assert(!f.errors);
   printf("signaled producer without availability: waits; sticky loss still propagates\n");
   finish(&f);
}

static void
test_reset_only(void)
{
   struct fixture f;
   init(&f);
   struct util_dynarray resets = {0};
   /* Retain reset ranges, but enqueue/claim must allocate no producer and
    * terminate even on repeated submit/fallback paths. */
   assert(hk_query_wait_record(&resets, f.pool, 0, 4, true));
   assert(resets.size && !hk_query_wait_has_writers(&resets));
   allocation_failure = 1;
   for (unsigned i = 0; i < 2; i++) {
      assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &resets, &resets, NULL,
                                   &f.queue[0]));
      struct hk_query_wait_producer *p = NULL;
      assert(hk_query_wait_claim(&f.dev, &f.queue[0], &resets, &resets, &p,
                                 &f.queue[0]));
      assert(!p && list_is_empty(&f.dev.producers) && allocation_failure == 1);
      hk_query_wait_publish(&f.dev, p, 1, true);
   }
   allocation_failure = 0;
   hk_query_wait_recording_finish(&resets);
   finish(&f);
   puts("reset-only: range metadata retained; no producer allocation; repeated enqueue/claim terminates");
}

/* Model a completed write, then an ordered, completed GPU reset-only submit,
 * then a host reader before another thread submits the next writer. The GPU
 * reset's execution is modelled by changing ready_at, not by executing a GPU.
 * Submitting the reset op must allocate no producer. Whether the old
 * writer was already pruned or is retired by this reader must not turn that
 * old completion into evidence that the new generation lost a write.
 *
 * Replay the same writer and reset recording: a recording is not a generation.
 */
static void
test_retired_reset_future(void)
{
   for (unsigned pruned = 0; pruned < 2; pruned++) {
      struct fixture f;
      bool never;
      init(&f);
      struct util_dynarray resets = {0};
      assert(hk_query_wait_record(&resets, f.pool, 0, 1, true));
      assert(resets.size && !hk_query_wait_has_writers(&resets));
      for (unsigned replay = 0; replay < 3; replay++) {
         f.now = 0;
         hk_query_wait_host_reset(f.pool, 0, 1);
         f.completion_at[0] = 0;
         enqueue(&f, 0);
         accept(&f, 0, 1 + replay);
         f.ready_at = 0;
         assert(available(&f));
         if (pruned)
            hk_query_wait_prune(&f.dev, &f.queue[0], 1 + replay);

         /* The reset invalidates evidence, without a writer allocation. */
         allocation_failure = 1;
         assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &resets, &resets,
                                      NULL, &f.queue[0]));
         uint64_t serial = f.dev.serial;
         struct hk_query_wait_producer *p = NULL;
         assert(hk_query_wait_claim(&f.dev, &f.queue[0], &resets, &resets, &p,
                                    &f.queue[0]));
         assert(!p && f.dev.serial == serial && allocation_failure == 1);
         allocation_failure = 0;
         f.ready_at = UINT64_MAX; /* completed GPU reset */

         f.publish_pending = true;
         f.publish_queue = 1;
         f.publish_at = UINT64_C(2500000000);
         f.ready_at = UINT64_C(2600000000);
         f.completion_at[1] = f.ready_at;
         unsigned completions = f.completions, changes = f.changes;
         assert(wait(&f, false, &never) == VK_SUCCESS && !never);
         assert(f.now == f.ready_at && !f.publish_pending);
         assert(f.changes - changes == 25);
         assert(f.completions - completions == 100 + !pruned);
         assert(list_is_empty(&f.dev.producers));
      }
      hk_query_wait_recording_finish(&resets);
      finish(&f);
   }
   puts("GPU reset-only after completed/pruned writer: future replay at 2500 ms succeeds at 2600 ms (CPU model)");
}

/* Missing-write regression. The
 * status callback stops fake time at 3 s and an external process watchdog
 * should bound the invocation too. UNKNOWN means the helper had no exit of
 * its own; this is a fault model, not a valid Vulkan/GPU failure reproduction.
 */
static int
probe_retired_missing(bool pruned)
{
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.completion_at[0] = 0;
   if (pruned)
      hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   f.watchdog_at = UINT64_C(3000000000);
   VkResult ret = wait(&f, false, &never);
   printf("retired-missing (%s): result=%d never=%d fake_ms=%llu completion_waits=%u condition_waits=%u live=%u\n",
          pruned ? "prune" : "completion", ret, never,
          (unsigned long long)(f.now / 1000000), f.completions, f.changes,
          list_length(&f.dev.producers));
   bool finite_exit = ret == VK_ERROR_DEVICE_LOST && !never;
   if (finite_exit)
      assert(f.now == HK_QUERY_WAIT_RETIRED_NS && f.errors == 1);
   puts(finite_exit ? "PASS: helper exited before watchdog" :
                     "UNRESOLVED: only the test watchdog ended the wait");
   finish(&f);
   return finite_exit ? 0 : 1;
}

/* A small order matrix distinguishes reset position, range and API queue
 * from completion queue. Each row runs both completion-observation paths. */
static void
test_ordered_resets(void)
{
   static const struct {
      const char *ops;
      bool separate, other_queue, redirected, loss;
   } cases[] = {
      {"rw", false, false, false, true},   {"wr", false, false, false, false},
      {"rwr", false, false, false, false}, {"rwrw", false, false, false, true},
      {"rwR", false, false, false, true}, /* R resets another slot */
      {"rw", true, false, false, true},    {"rw", true, true, false, false},
      {"rw", true, true, true, false}, /* two API queues, one real queue */
   };
   for (unsigned c = 0; c < ARRAY_SIZE(cases); c++) {
      for (unsigned pruned = 0; pruned < 2; pruned++) {
         struct fixture f;
         bool never;
         init(&f);
         hk_query_wait_recording_finish(&f.recording);
         for (const char *op = cases[c].ops; *op; op++) {
            bool reset = *op != 'w';
            assert(hk_query_wait_record(&f.recording, f.pool, *op == 'R', 1,
                                        reset));
            if (cases[c].separate && reset) {
               assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording,
                                            &f.recording, NULL, &f.queue[0]));
               hk_query_wait_recording_finish(&f.recording);
            }
         }
         unsigned real = cases[c].other_queue && !cases[c].redirected;
         void *logical = &f.queue[cases[c].other_queue];
         assert(hk_query_wait_enqueue(&f.dev, &f.queue[real], &f.recording,
                                      &f.recording, NULL, logical));
         f.order_queue[real] = logical;
         accept(&f, real, 1);
         f.completion_at[real] = 0;
         if (pruned)
            hk_query_wait_prune(&f.dev, &f.queue[real], 1);
         f.watchdog_at = UINT64_C(3000000000);
         VkResult result = wait(&f, false, &never);
         assert(result ==
                (cases[c].loss ? VK_ERROR_DEVICE_LOST : VK_ERROR_UNKNOWN));
         assert(!never && f.now == (cases[c].loss ? HK_QUERY_WAIT_RETIRED_NS
                                                  : f.watchdog_at));
         finish(&f);
      }
   }

   /* A later queued reset must invalidate an earlier writer even when the
    * earlier callback runs afterward. Claim must not replay/reset generations.
    * A canceled latest writer cannot revive evidence from the older one. */
   for (unsigned cancel = 0; cancel < 2; cancel++) {
      struct fixture f;
      bool never;
      init(&f);
      hk_query_wait_recording_finish(&f.recording);
      assert(hk_query_wait_record(&f.recording, f.pool, 0, 1, true));
      assert(hk_query_wait_record(&f.recording, f.pool, 0, 1, false));
      enqueue(&f, 0);
      struct util_dynarray later = {0};
      assert(hk_query_wait_record(&later, f.pool, 0, 1, true));
      if (cancel)
         assert(hk_query_wait_record(&later, f.pool, 0, 1, false));
      assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &later, &later, NULL,
                                   &f.queue[0]));
      accept(&f, 0, 1);
      hk_query_wait_prune(&f.dev, &f.queue[0], 1);
      if (cancel) {
         struct hk_query_wait_producer *p;
         assert(hk_query_wait_claim(&f.dev, &f.queue[0], &later, &later, &p,
                                    &f.queue[0]));
         hk_query_wait_publish(&f.dev, p, 0, false);
      }
      f.watchdog_at = UINT64_C(3000000000);
      assert(wait(&f, false, &never) == VK_ERROR_UNKNOWN && !f.errors);
      hk_query_wait_recording_finish(&later);
      finish(&f);
   }
   /* Cross-queue uncertainty survives a later same-queue reset and replay;
    * it must not be overwritten by the last reset's queue identity. */
   struct fixture f;
   bool never;
   init(&f);
   struct util_dynarray resets = {0};
   assert(hk_query_wait_record(&resets, f.pool, 0, 1, true));
   for (unsigned replay = 0; replay < 2; replay++) {
      for (unsigned n = 0; n < 3; n++) {
         unsigned q = n == 1;
         assert(hk_query_wait_enqueue(&f.dev, &f.queue[q], &resets, &resets,
                                      NULL, &f.queue[q]));
      }
      enqueue(&f, 0);
      accept(&f, 0, replay + 1);
      hk_query_wait_prune(&f.dev, &f.queue[0], replay + 1);
      f.now = 0;
      f.watchdog_at = UINT64_C(3000000000);
      assert(wait(&f, false, &never) == VK_ERROR_UNKNOWN && !f.errors);
   }
   hk_query_wait_recording_finish(&resets);
   finish(&f);
   /* Identical recording, two logical queues, one completion queue. Register
    * B then A but invoke A's callback first. Failure must belong to its actual
    * origin: stealing B's registration could turn canceled A into evidence. */
   for (unsigned fail_a = 0; fail_a < 2; fail_a++) {
      init(&f);
      struct util_dynarray reset = {0};
      assert(hk_query_wait_record(&reset, f.pool, 0, 1, true));
      assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &reset, &reset, NULL,
                                   &f.queue[0]));
      assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording,
                                   &f.recording, NULL, &f.queue[1]));
      enqueue(&f, 0);
      struct hk_query_wait_producer *a, *b;
      assert(hk_query_wait_claim(&f.dev, &f.queue[0], &f.recording,
                                 &f.recording, &a, &f.queue[0]));
      assert(hk_query_wait_claim(&f.dev, &f.queue[0], &f.recording,
                                 &f.recording, &b, &f.queue[1]));
      assert(a != b);
      hk_query_wait_publish(&f.dev, a, fail_a ? 0 : 1, !fail_a);
      hk_query_wait_publish(&f.dev, b, fail_a ? 1 : 0, fail_a);
      hk_query_wait_prune(&f.dev, &f.queue[0], 1);
      f.watchdog_at = UINT64_C(3000000000);
      assert(wait(&f, false, &never) ==
             (fail_a ? VK_ERROR_UNKNOWN : VK_ERROR_DEVICE_LOST));
      hk_query_wait_recording_finish(&reset);
      finish(&f);
   }
   puts(
      "ordered GPU resets: position, subsets, separate submits, cross/redirected queues, deferred claims and canceled latest passed");
}

static void
unrelated_churn(struct fixture *f)
{
   struct util_dynarray ops = {0};
   assert(hk_query_wait_record(&ops, f->pool, 1, 1, false));
   struct hk_query_wait_producer *p;
   assert(hk_query_wait_enqueue(&f->dev, &f->queue[1], &ops, &ops, NULL,
                                &f->queue[1]));
   assert(
      hk_query_wait_claim(&f->dev, &f->queue[1], &ops, &ops, &p, &f->queue[1]));
   hk_query_wait_publish(&f->dev, p, ++f->point[1], true);
   hk_query_wait_prune(&f->dev, &f->queue[1], f->point[1]);
   hk_query_wait_recording_finish(&ops);
   f->event = unrelated_churn;
   f->event_at = f->now + HK_QUERY_WAIT_INTERVAL_NS;
}

static void
new_writer(struct fixture *f)
{
   enqueue(f, 0);
   accept(f, 0, f->point[0] + 1);
   f->ready_at = f->completion_at[0] = UINT64_C(4500000000);
}

static void
new_registered(struct fixture *f)
{
   f->batch = hk_query_wait_batch_create();
   assert(f->batch);
   assert(hk_query_wait_enqueue(&f->dev, &f->queue[0], &f->recording,
                                &f->recording, f->batch, &f->queue[0]));
   f->ready_at = UINT64_C(4500000000);
}

static void
new_queued(struct fixture *f)
{
   enqueue(f, 0);
   f->ready_at = UINT64_C(4500000000);
}

static void
new_reserved(struct fixture *f)
{
   new_queued(f);
   f->reserved = claim(f, 0);
}

static void
host_reset(struct fixture *f)
{
   hk_query_wait_host_reset(f->pool, 0, 1);
   f->publish_pending = true;
   f->publish_at = UINT64_C(2500000000);
   f->ready_at = f->completion_at[0] = UINT64_C(2600000000);
}

static void
gpu_reset(struct fixture *f)
{
   struct util_dynarray ops = {0};
   assert(hk_query_wait_record(&ops, f->pool, 0, 1, true));
   assert(hk_query_wait_enqueue(&f->dev, &f->queue[1], &ops, &ops, NULL,
                                &f->queue[1]));
   hk_query_wait_recording_finish(&ops);
   f->publish_pending = true;
   f->publish_at = UINT64_C(2500000000);
   f->ready_at = f->completion_at[0] = UINT64_C(4500000000);
}

static void
deadline_ready(struct fixture *f)
{
   f->ready_at = f->now;
}

static void
deadline_loss(struct fixture *f)
{
   f->loss_at = f->now;
}

static void
test_retired_generations(void)
{
   /* Churn on another slot/queue must not restart the queried slot's grace. */
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   f.event = unrelated_churn;
   f.event_at = HK_QUERY_WAIT_INTERVAL_NS;
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST && !never);
   assert(f.now == HK_QUERY_WAIT_RETIRED_NS && f.errors == 1);
   finish(&f);

   void (*events[])(struct fixture *) = {
      new_writer, host_reset, gpu_reset, new_registered, new_queued, new_reserved,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(events); i++) {
      init(&f);
      enqueue(&f, 0);
      accept(&f, 0, 1);
      hk_query_wait_prune(&f.dev, &f.queue[0], 1);
      f.event = events[i];
      f.event_at = UINT64_C(1900000000);
      assert(wait(&f, false, &never) == VK_SUCCESS && !never);
      assert(f.now == f.ready_at && !f.errors);
      if (f.reserved)
         hk_query_wait_publish(&f.dev, f.reserved, 0, false);
      if (f.batch)
         hk_query_wait_batch_finish(&f.dev, f.batch, true);
      finish(&f);
   }

   /* Exercise the unlocked final status callback itself: registration or
    * reset during it invalidates the slot snapshot. Availability and sticky
    * loss on that boundary must win over the missing-write diagnostic. */
   void (*final_events[])(struct fixture *) = {
      new_writer, host_reset, gpu_reset, deadline_ready, deadline_loss,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(final_events); i++) {
      init(&f);
      enqueue(&f, 0);
      accept(&f, 0, 1);
      hk_query_wait_prune(&f.dev, &f.queue[0], 1);
      f.deadline_event = final_events[i];
      assert(wait(&f, false, &never) ==
             (i == 4 ? VK_ERROR_DEVICE_LOST : VK_SUCCESS));
      assert(!never && !f.errors && !f.deadline_event);
      finish(&f);
   }

   /* Host reset unlinks an accepted writer even if it has not completed.
    * Failure/cancel and queue teardown also provide no completion evidence.
    * These are helper lifetime tests, not valid concurrent Vulkan resets. */
   for (unsigned i = 0; i < 4; i++) {
      init(&f);
      enqueue(&f, 0);
      if (i == 0) {
         accept(&f, 0, 1);
         hk_query_wait_host_reset(f.pool, 0, 1);
      } else if (i == 1) {
         hk_query_wait_publish(&f.dev, claim(&f, 0), 0, false);
      } else if (i == 2) {
         accept(&f, 0, 1);
         hk_query_wait_finish_queue(&f.dev, &f.queue[0]);
      } else {
         accept(&f, 0, 1);
         hk_query_wait_prune(&f.dev, &f.queue[0], 1);
         enqueue(&f, 0);
         hk_query_wait_publish(&f.dev, claim(&f, 0), 0, false);
      }
      f.watchdog_at = UINT64_C(3000000000);
      assert(wait(&f, false, &never) == VK_ERROR_UNKNOWN && !never);
      assert(!f.errors);
      finish(&f);
   }

   /* The older completed instance cannot poison a newer pending writer. */
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   enqueue(&f, 1);
   accept(&f, 1, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   f.ready_at = f.completion_at[1] = UINT64_C(4500000000);
   assert(wait(&f, false, &never) == VK_SUCCESS && !never && !f.errors);
   finish(&f);

   /* A host reset recovers diagnostic coverage after GPU-reset uncertainty. */
   init(&f);
   gpu_reset(&f);
   hk_query_wait_host_reset(f.pool, 0, 1);
   f.publish_pending = false;
   f.ready_at = UINT64_MAX;
   enqueue(&f, 0);
   accept(&f, 0, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST && !never);
   assert(f.now == HK_QUERY_WAIT_RETIRED_NS && f.errors == 1);
   finish(&f);

   /* Explicitly retain the unresolved GPU-reset-generation case. */
   init(&f);
   gpu_reset(&f);
   f.publish_pending = false;
   f.ready_at = UINT64_MAX;
   enqueue(&f, 0);
   accept(&f, 0, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   f.watchdog_at = UINT64_C(3000000000);
   assert(wait(&f, false, &never) == VK_ERROR_UNKNOWN && !never && !f.errors);
   finish(&f);

   /* Merely recording an unsubmitted reset does not suppress diagnostics. */
   init(&f);
   struct util_dynarray unsubmitted = {0};
   assert(hk_query_wait_record(&unsubmitted, f.pool, 0, 1, true));
   enqueue(&f, 0);
   accept(&f, 0, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 1);
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST && !never);
   assert(f.now == HK_QUERY_WAIT_RETIRED_NS && f.errors == 1);
   hk_query_wait_recording_finish(&unsubmitted);
   finish(&f);
   puts("retired grace: per-slot churn, newer writers, resets, cancellation, final status races and host-reset recovery passed");
}

static void
test_retirement(void)
{
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 9000);
   enqueue(&f, 1);
   accept(&f, 1, 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 9000);
   assert(list_length(&f.dev.producers) == 1);
   f.ready_at = f.completion_at[1] = 1;
   assert(wait(&f, false, &never) == VK_SUCCESS);
   assert(f.completions == 1 && list_is_empty(&f.dev.producers));
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   f.completion_at[0] = 0;
   f.loss_at = HK_QUERY_WAIT_INTERVAL_NS;
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
   assert(f.completions == 1 && f.changes == 1);
   finish(&f);

   init(&f);
   for (unsigned i = 1; i <= 4096; i++) {
      enqueue(&f, 0);
      accept(&f, 0, i);
      /* Same queue-level retirement hook used at the next query submit. */
      hk_query_wait_prune(&f.dev, &f.queue[0], i);
      assert(list_is_empty(&f.dev.producers));
   }
   assert(!f.available && !f.completions && !f.changes);
   printf("GPU-only reuse: 4096 submissions retired with no host query wait\n");
   finish(&f);
}

static void
test_epochs_and_failure(void)
{
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 1);
   assert(hk_query_wait_record(&f.recording, f.pool, 0, 1, true));
   assert(hk_query_wait_record(&f.recording, f.pool, 0, 4, false));
   /* Repeated/secondary replay records ordered operations. Registration
    * preserves the still-available old epoch until the GPU reset executes. */
   enqueue(&f, 0);
   accept(&f, 0, 2);
   f.ready_at = 0;
   assert(wait(&f, false, &never) == VK_SUCCESS && !f.completions);
   assert(list_length(&f.dev.producers) == 2);
   hk_query_wait_host_reset(f.pool, 0, 1);
   assert(list_length(&f.dev.producers) == 1);
   hk_query_wait_host_reset(f.pool, 1, 3);
   assert(list_is_empty(&f.dev.producers));
   finish(&f);

   init(&f);
   struct hk_query_wait_batch *batch = hk_query_wait_batch_create();
   assert(batch);
   assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording, &f.recording,
                                batch, &f.queue[0]));
   assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording, &f.recording,
                                batch, &f.queue[0]));
   struct hk_query_wait_producer *p = claim(&f, 0);
   hk_query_wait_publish(&f.dev, p, 17, true);
   /* A partially queued API call fails after its earlier part was accepted. */
   hk_query_wait_batch_finish(&f.dev, batch, true);
   assert(list_length(&f.dev.producers) == 1);
   hk_query_wait_prune(&f.dev, &f.queue[0], 17);
   assert(list_is_empty(&f.dev.producers));
   /* An earlier queued part can still reach the callback after the error.
    * Claim reconstructs it without publishing a point before acceptance. */
   p = claim(&f, 0);
   hk_query_wait_publish(&f.dev, p, 0, false);
   assert(list_is_empty(&f.dev.producers));
   f.loss_at = 1;
   assert(wait(&f, false, &never) == VK_ERROR_DEVICE_LOST);
   assert(!f.completions && f.changes == 1);
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   p = claim(&f, 0);
   hk_query_wait_pool_destroy(f.pool);
   f.pool = NULL;
   hk_query_wait_recording_finish(&f.recording);
   hk_query_wait_publish(&f.dev, p, 0, false);
   assert(list_is_empty(&f.dev.producers));
   finish(&f);
}

static void
test_allocation_failure(void)
{
   struct fixture f;
   init(&f);
   allocation_failure = 1;
   assert(!hk_query_wait_pool_create(&f.dev, 4));
   allocation_failure = 1;
   assert(!hk_query_wait_batch_create());
   allocation_failure = 1;
   assert(!hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording,
                                 &f.recording, NULL, &f.queue[0]));
   assert(list_is_empty(&f.dev.producers));
   struct util_dynarray failed_recording = {0};
   allocation_failure = 1;
   assert(!hk_query_wait_record(&failed_recording, f.pool, 0, 1, false));
   assert(!failed_recording.size);
   allocation_failure = 1;
   assert(!hk_query_wait_record(&failed_recording, f.pool, 0, 1, true));
   assert(!failed_recording.size);
   hk_query_wait_recording_finish(&failed_recording);
   struct hk_query_wait_producer *p;
   allocation_failure = 1;
   assert(!hk_query_wait_claim(&f.dev, &f.queue[0], &f.recording, &f.recording,
                               &p, &f.queue[0]));
   assert(!p && list_is_empty(&f.dev.producers));
   finish(&f);
}

static void
test_victims(void)
{
   for (unsigned mode = 0; mode < 4; mode++) {
      struct fixture f;
      bool never;
      init(&f);
      f.idle = true;
      f.race_ready = mode == 1;
      f.race_loss = mode == 2;
      if (mode == 3) {
         enqueue(&f, 0);
         f.loss_at = HK_QUERY_WAIT_INTERVAL_NS;
      }
      VkResult ret = wait(&f, true, &never);
      assert(ret == (mode >= 2 ? VK_ERROR_DEVICE_LOST : VK_SUCCESS));
      assert(never == (mode == 0));
      assert(f.completions == 0);
      finish(&f);
   }
   struct fixture f;
   bool never;
   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 5);
   f.completion_at[0] = 10;
   f.idle_after_completion = true;
   assert(wait(&f, true, &never) == VK_SUCCESS && never);
   assert(f.now == 10 && f.completions == 1 && !f.changes);
   finish(&f);
}

/* The completion mock deliberately has no loss notification, just as an
 * unrelated/unpublished DRM fence does not observe a Mesa status word. */
struct threaded {
   struct fixture f;
   mtx_t lock;
   struct u_cnd_monotonic changed;
   unsigned entered, returned;
   unsigned waiting;
   bool lost;
   bool ready;
   unsigned polls, statuses;
   uint64_t start, delay, signalled;
};

static bool
thread_ready(void *data)
{
   struct threaded *t = data;
   mtx_lock(&t->lock);
   t->polls++;
   bool ready = t->ready;
   mtx_unlock(&t->lock);
   return ready;
}

static VkResult
thread_status(void *data)
{
   struct threaded *t = data;
   mtx_lock(&t->lock);
   t->statuses++;
   bool lost = t->lost;
   mtx_unlock(&t->lock);
   return lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static int
thread_completion(void *data, void *queue, uint64_t point, uint64_t deadline)
{
   struct threaded *t = data;
   assert(queue == &t->f.queue[0] && point == 1);
   mtx_lock(&t->lock);
   t->entered++;
   u_cnd_monotonic_broadcast(&t->changed);
   struct timespec ts = {deadline / 1000000000, deadline % 1000000000};
   /* No wake on the lost/ready flag; only timeout or mock completion. */
   while (!t->ready) {
      int ret = u_cnd_monotonic_timedwait(&t->changed, &t->lock, &ts);
      if (ret == thrd_timedout)
         break;
      assert(ret == thrd_success);
   }
   bool ready = t->ready;
   mtx_unlock(&t->lock);
   return ready ? 0 : -ETIME;
}

static int
reader(void *data)
{
   struct threaded *t = data;
   const struct hk_query_wait_ops thread_ops = {
      .available = thread_ready,
      .status = thread_status,
      .completion = thread_completion,
   };
   bool never;
   assert(hk_query_wait_available(t->f.pool, 0, false, &never, &thread_ops,
                                  t) == VK_ERROR_DEVICE_LOST);
   mtx_lock(&t->lock);
   t->returned++;
   u_cnd_monotonic_broadcast(&t->changed);
   mtx_unlock(&t->lock);
   return 0;
}

static void
test_threads(void)
{
   struct threaded t = {0};
   init(&t.f);
   assert(mtx_init(&t.lock, mtx_plain) == thrd_success);
   assert(u_cnd_monotonic_init(&t.changed) == thrd_success);
   enqueue(&t.f, 0);
   accept(&t.f, 0, 1);
   thrd_t a, b;
   assert(thrd_create(&a, reader, &t) == thrd_success);
   assert(thrd_create(&b, reader, &t) == thrd_success);
   mtx_lock(&t.lock);
   while (t.entered < 2)
      assert(u_cnd_monotonic_wait(&t.changed, &t.lock) == thrd_success);
   t.lost = true;
   uint64_t start = os_time_get_nano();
   /* Concurrent retirement must not free a producer used by either waiter. */
   hk_query_wait_prune(&t.f.dev, &t.f.queue[0], 1);
   while (t.returned < 2)
      assert(u_cnd_monotonic_wait(&t.changed, &t.lock) == thrd_success);
   uint64_t elapsed = os_time_get_nano() - start;
   mtx_unlock(&t.lock);
   thrd_join(a, NULL);
   thrd_join(b, NULL);
   assert(elapsed < UINT64_C(2000000000));
   printf("two live waiters noticed loss without fence notification: %.3f ms\n",
          elapsed / 1e6);
   u_cnd_monotonic_destroy(&t.changed);
   mtx_destroy(&t.lock);
   finish(&t.f);
}

static int
registration_changed(void *data, struct hk_query_wait_device *dev,
                     uint64_t deadline)
{
   struct threaded *t = data;
   assert(deadline <= os_time_get_nano() + HK_QUERY_WAIT_INTERVAL_NS);
   mtx_lock(&t->lock);
   t->waiting++;
   u_cnd_monotonic_broadcast(&t->changed);
   mtx_unlock(&t->lock);
   /* The producer registers after the announcement above, contending on the
    * still-held tracking mutex. A longer test guard distinguishes a missed
    * notification from the normal bounded predicate-check fallback. */
   deadline = os_time_get_nano() + UINT64_C(2000000000);
   struct timespec ts = {deadline / 1000000000, deadline % 1000000000};
   int ret = u_cnd_monotonic_timedwait(&dev->changed, &dev->mutex, &ts);
   assert(ret == thrd_success);
   return ret;
}

static int
registration_reader(void *data)
{
   struct threaded *t = data;
   const struct hk_query_wait_ops thread_ops = {
      .available = thread_ready,
      .status = thread_status,
      .completion = thread_completion,
      .changed = registration_changed,
   };
   bool never;
   assert(hk_query_wait_available(t->f.pool, 0, false, &never, &thread_ops,
                                  t) == VK_SUCCESS);
   assert(!never);
   return 0;
}

static void
test_registration_race(void)
{
   struct threaded t = {0};
   init(&t.f);
   assert(mtx_init(&t.lock, mtx_plain) == thrd_success);
   assert(u_cnd_monotonic_init(&t.changed) == thrd_success);
   thrd_t thread;
   assert(thrd_create(&thread, registration_reader, &t) == thrd_success);
   mtx_lock(&t.lock);
   while (!t.waiting)
      assert(u_cnd_monotonic_wait(&t.changed, &t.lock) == thrd_success);
   mtx_unlock(&t.lock);
   enqueue(&t.f, 0);
   accept(&t.f, 0, 1);
   mtx_lock(&t.lock);
   while (!t.entered)
      assert(u_cnd_monotonic_wait(&t.changed, &t.lock) == thrd_success);
   t.ready = true;
   u_cnd_monotonic_broadcast(&t.changed);
   mtx_unlock(&t.lock);
   thrd_join(thread, NULL);
   assert(list_is_empty(&t.f.dev.producers));
   u_cnd_monotonic_destroy(&t.changed);
   mtx_destroy(&t.lock);
   finish(&t.f);
}

static uint64_t
clock_ns(clockid_t clock)
{
   struct timespec ts;
   assert(!clock_gettime(clock, &ts));
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static int
ready_later(void *data)
{
   struct threaded *t = data;
   mtx_lock(&t->lock);
   while (!t->start)
      assert(u_cnd_monotonic_wait(&t->changed, &t->lock) == thrd_success);
   uint64_t at = t->start + t->delay;
   mtx_unlock(&t->lock);
   struct timespec ts = {at / 1000000000, at % 1000000000};
   int ret;
   do {
      ret = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
   } while (ret == EINTR);
   assert(!ret);
   mtx_lock(&t->lock);
   t->ready = true;
   t->signalled = os_time_get_nano();
   u_cnd_monotonic_broadcast(&t->changed);
   mtx_unlock(&t->lock);
   return 0;
}

static void
measure(unsigned delay_ms)
{
   struct threaded t = {0};
   init(&t.f);
   assert(mtx_init(&t.lock, mtx_plain) == thrd_success);
   assert(u_cnd_monotonic_init(&t.changed) == thrd_success);
   enqueue(&t.f, 0);
   accept(&t.f, 0, 1);
   t.delay = (uint64_t)delay_ms * 1000000;
   thrd_t thread;
   assert(thrd_create(&thread, ready_later, &t) == thrd_success);
   const struct hk_query_wait_ops real_ops = {
      .available = thread_ready,
      .status = thread_status,
      .completion = thread_completion,
   };
   bool never;
   uint64_t process0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
   uint64_t cpu0 = clock_ns(CLOCK_THREAD_CPUTIME_ID);
   mtx_lock(&t.lock);
   uint64_t wall0 = t.start = os_time_get_nano();
   u_cnd_monotonic_broadcast(&t.changed);
   mtx_unlock(&t.lock);
   VkResult ret =
      hk_query_wait_available(t.f.pool, 0, false, &never, &real_ops, &t);
   uint64_t wall1 = os_time_get_nano();
   uint64_t cpu1 = clock_ns(CLOCK_THREAD_CPUTIME_ID);
   uint64_t process1 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
   thrd_join(thread, NULL);
   assert(ret == VK_SUCCESS && !never);
   printf("{\"strategy\":\"production-helper\",\"delay_ms\":%u,"
          "\"result\":%d,\"never\":false,\"wall_ms\":%.6f,"
          "\"process_cpu_ms\":%.6f,\"thread_cpu_ms\":%.6f,"
          "\"availability_polls\":%u,\"status_polls\":%u,"
          "\"completion_waits\":%u,\"signal_after_ms\":%.6f}\n",
          delay_ms, ret, (wall1 - wall0) / 1e6, (process1 - process0) / 1e6,
          (cpu1 - cpu0) / 1e6, t.polls, t.statuses, t.entered,
          (t.signalled - wall0) / 1e6);
   u_cnd_monotonic_destroy(&t.changed);
   mtx_destroy(&t.lock);
   finish(&t.f);
}

int
main(int argc, char **argv)
{
   if (argc == 3 && !strcmp(argv[1], "--probe-retired-missing")) {
      assert(!strcmp(argv[2], "completion") || !strcmp(argv[2], "prune"));
      return probe_retired_missing(!strcmp(argv[2], "prune"));
   }
   if (argc == 3 && !strcmp(argv[1], "--measure-delayed")) {
      unsigned delay = strtoul(argv[2], NULL, 10);
      assert(delay == 150 || delay == 2200);
      measure(delay);
      return 0;
   }
   if (argc == 3 && !strcmp(argv[1], "--case")) {
      if (!strcmp(argv[2], "reset"))
         test_reset_only();
      else if (!strcmp(argv[2], "retired-reset-future"))
         test_retired_reset_future();
      else if (!strcmp(argv[2], "rechecks"))
         test_rechecks_and_errors();
      else if (!strcmp(argv[2], "waits"))
         test_waits();
      else if (!strcmp(argv[2], "registered"))
         test_unaccepted_availability(0);
      else if (!strcmp(argv[2], "queued"))
         test_unaccepted_availability(1);
      else if (!strcmp(argv[2], "reserved"))
         test_unaccepted_availability(2);
      else
         assert(!"unknown case");
      return 0;
   }
   assert(argc == 1);
   test_reset_only();
   test_retired_reset_future();
   assert(probe_retired_missing(false) == 0);
   assert(probe_retired_missing(true) == 0);
   test_ordered_resets();
   test_retired_generations();
   test_rechecks_and_errors();
   for (unsigned state = 0; state < 3; state++)
      test_unaccepted_availability(state);
   test_idle_rechecks();
   test_future_producers();
   test_waits();
   test_retirement();
   test_epochs_and_failure();
   test_allocation_failure();
   test_victims();
   test_threads();
   test_registration_race();
   puts("PASS: production query wait/tracking helpers; no GPU/device access");
   return 0;
}
