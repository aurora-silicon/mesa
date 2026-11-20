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
   uint64_t point[2], completion_at[2];
   uint64_t now, ready_at, loss_at, publish_at;
   unsigned available, statuses, completions, changes, errors;
   bool race_ready, race_loss, idle;
   bool publish_pending;
};

static void
init(struct fixture *f)
{
   memset(f, 0, sizeof(*f));
   assert(hk_query_wait_device_init(&f->dev));
   f->pool = hk_query_wait_pool_create(&f->dev, 4);
   assert(f->pool);
   f->ready_at = f->loss_at = f->publish_at = UINT64_MAX;
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
                                &f->recording, b));
   hk_query_wait_batch_finish(&f->dev, b, false);
}

static struct hk_query_wait_producer *
claim(struct fixture *f, unsigned q)
{
   struct hk_query_wait_producer *p;
   assert(hk_query_wait_claim(&f->dev, &f->queue[q], &f->recording,
                              &f->recording, &p));
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
   assert(f->now <= UINT64_C(5000000000));
   return f->now >= f->ready_at;
}

static VkResult
status(void *data)
{
   struct fixture *f = data;
   f->statuses++;
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
   return ((struct fixture *)data)->now;
}

static int
completion(void *data, void *queue, uint64_t point, uint64_t deadline)
{
   struct fixture *f = data;
   unsigned q = queue == &f->queue[0] ? 0 : 1;
   assert(queue == &f->queue[q]);
   assert(point == f->point[q] && point != 0);
   assert(deadline > f->now);
   assert(deadline - f->now <= HK_QUERY_WAIT_INTERVAL_NS);
   f->completions++;
   if (f->completion_at[q] <= deadline) {
      f->now = f->completion_at[q];
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
   if (f->publish_pending && deadline >= f->publish_at) {
      f->now = f->publish_at;
      f->publish_pending = false;
      mtx_unlock(&dev->mutex);
      accept(f, 0, 7);
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
   f->errors++;
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
   assert(f.now == f.ready_at && f.completions == 22 && !f.errors);
   printf("delayed producer: success at 2200 ms, 22 bounded DRM waits\n");
   finish(&f);

   init(&f);
   enqueue(&f, 0);
   accept(&f, 0, 11);
   f.ready_at = 1;
   assert(wait(&f, false, &never) == VK_SUCCESS && !never);
   assert(f.now == HK_QUERY_WAIT_INTERVAL_NS && f.completions == 1);
   assert(f.completion_at[0] == UINT64_MAX);
   printf("availability before host-dependent tail: extra 99999999 ns\n");
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
      assert(f.now == HK_QUERY_WAIT_INTERVAL_NS && !f.errors);
      assert(f.completions == (state == 2));
      if (reserved)
         hk_query_wait_publish(&f.dev, reserved, 0, false);
      finish(&f);
   }
   printf("queued/reserved/accepted loss: extra 99999999 ns in each state\n");

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
   assert(f.changes == 3 && f.completions == 20 && !f.errors);
   assert(f.now == f.ready_at);
   finish(&f);
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
                                batch));
   assert(hk_query_wait_enqueue(&f.dev, &f.queue[0], &f.recording, &f.recording,
                                batch));
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
                                 &f.recording, NULL));
   assert(list_is_empty(&f.dev.producers));
   struct util_dynarray failed_recording = {0};
   allocation_failure = 1;
   assert(!hk_query_wait_record(&failed_recording, f.pool, 0, 1, false));
   assert(!failed_recording.size);
   hk_query_wait_recording_finish(&failed_recording);
   struct hk_query_wait_producer *p;
   allocation_failure = 1;
   assert(!hk_query_wait_claim(&f.dev, &f.queue[0], &f.recording, &f.recording,
                               &p));
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
}

/* The completion mock deliberately has no loss notification, just as an
 * unrelated/unpublished DRM fence does not observe a Mesa status word. */
struct threaded {
   struct fixture f;
   mtx_t lock;
   struct u_cnd_monotonic changed;
   unsigned entered, returned;
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
   if (argc == 3 && !strcmp(argv[1], "--measure-delayed")) {
      unsigned delay = strtoul(argv[2], NULL, 10);
      assert(delay == 150 || delay == 2200);
      measure(delay);
      return 0;
   }
   assert(argc == 1);
   test_waits();
   test_retirement();
   test_epochs_and_failure();
   test_allocation_failure();
   test_victims();
   test_threads();
   puts("PASS: production query wait/tracking helpers; no GPU/device access");
   return 0;
}
