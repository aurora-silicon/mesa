/*
 * Copyright 1999-2006 Brian Paul
 * Copyright 2008 VMware, Inc.
 * Copyright 2022 Yonggang Luo
 * SPDX-License-Identifier: MIT
 */

#include "util/u_thread.h"
#include "util/perf/u_perfetto.h"

#include "macros.h"

#ifdef HAVE_PTHREAD
#include <signal.h>
#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif
#endif

#ifdef __HAIKU__
#include <OS.h>
#endif

#if DETECT_OS_LINUX && !DETECT_OS_ANDROID
#include <sched.h>
#elif defined(_WIN32) && !defined(HAVE_PTHREAD)
#include <windows.h>
#endif

#if DETECT_OS_LINUX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

int
util_get_current_cpu(void)
{
#if DETECT_OS_LINUX && !DETECT_OS_ANDROID
   return sched_getcpu();

#elif defined(_WIN32) && !defined(HAVE_PTHREAD)
   return GetCurrentProcessorNumber();

#else
   return -1;
#endif
}

#if DETECT_OS_LINUX && defined(SYS_sched_getattr) && defined(SYS_sched_setattr)
/* struct sched_attr of sched_setattr(2), up to the utilisation clamps. */
struct util_sched_attr {
   uint32_t size;
   uint32_t sched_policy;
   uint64_t sched_flags;
   int32_t sched_nice;
   uint32_t sched_priority;
   uint64_t sched_runtime;
   uint64_t sched_deadline;
   uint64_t sched_period;
   uint32_t sched_util_min;
   uint32_t sched_util_max;
};

#define UTIL_SCHED_FLAG_RESET_ON_FORK  0x01
#define UTIL_SCHED_FLAG_KEEP_POLICY    0x08
#define UTIL_SCHED_FLAG_KEEP_PARAMS    0x10
#define UTIL_SCHED_FLAG_UTIL_CLAMP_MIN 0x20

/* Whether the calling thread is known to run without a seccomp filter.
 *
 * Seccomp filters are per thread, so this reads the thread's own status, not
 * the process's (/proc/self is the main thread). When the status cannot be
 * read (no /proc, or a sandbox that denies the open) the answer is no.
 */
static bool
util_thread_has_no_seccomp(void)
{
   FILE *f = fopen("/proc/thread-self/status", "re");
   bool none = false;
   char line[128];

   if (!f)
      return false;

   while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "Seccomp:", 8) == 0) {
         char *end;
         long mode = strtol(line + 8, &end, 10);

         none = end != line + 8 && mode == 0;
         break;
      }
   }

   fclose(f);
   return none;
}
#endif

bool
util_thread_raise_min_util_clamp(unsigned util_min)
{
   if (!util_min)
      return true;

#if DETECT_OS_LINUX && defined(SYS_sched_getattr) && defined(SYS_sched_setattr)
   /* Chromium's GPU process sends sched_getattr and sched_setattr to SIGSYS
    * even for the calling thread, and a queue can create a thread after the
    * sandbox is up.
    */
   if (!util_thread_has_no_seccomp())
      return false;

   struct util_sched_attr attr;
   memset(&attr, 0, sizeof(attr));
   if (syscall(SYS_sched_getattr, 0, &attr, sizeof(attr), 0) != 0)
      return false;

   if (attr.sched_util_min >= util_min)
      return true;

   /* Keep the policy and its parameters (the ones just read, so that an RT
    * or deadline thread passes the parameter checks too) and only change the
    * minimum clamp. With KEEP_POLICY the kernel also keeps the thread's
    * reset-on-fork setting, so threads it creates later inherit what they
    * did before. The kernel rejects a minimum above the thread's maximum
    * clamp, which is then kept as well.
    */
   attr.size = sizeof(attr);
   attr.sched_flags = (attr.sched_flags & UTIL_SCHED_FLAG_RESET_ON_FORK) |
                      UTIL_SCHED_FLAG_KEEP_POLICY |
                      UTIL_SCHED_FLAG_KEEP_PARAMS |
                      UTIL_SCHED_FLAG_UTIL_CLAMP_MIN;
   attr.sched_util_min = util_min;

   /* Kernels without CONFIG_UCLAMP_TASK return EOPNOTSUPP. */
   return syscall(SYS_sched_setattr, 0, &attr, 0) == 0;
#else
   return false;
#endif
}

struct u_thread_param {
   int (*routine)(void *);
   void *arg;
};

static int
thread_routine(void *arg)
{
   struct u_thread_param *paramp = arg;
   struct u_thread_param param = *paramp;

   free(paramp);

   int ret = param.routine(param.arg);

   util_perfetto_thread_flush();

   return ret;
}

int u_thread_create(thrd_t *thrd, int (*routine)(void *), void *param)
{
   struct u_thread_param *paramp = malloc(sizeof(*paramp));
   int ret = thrd_error;

   paramp->routine = routine;
   paramp->arg = param;

#if defined(HAVE_PTHREAD) && !DETECT_OS_FUCHIA
   sigset_t saved_set, new_set;

   sigfillset(&new_set);
   sigdelset(&new_set, SIGSYS);

   /* SIGSEGV is commonly used by Vulkan API tracing layers in order to track
    * accesses in device memory mapped to user space. Blocking the signal hinders
    * that tracking mechanism.
    */
   sigdelset(&new_set, SIGSEGV);
   pthread_sigmask(SIG_BLOCK, &new_set, &saved_set);
   ret = thrd_create(thrd, thread_routine, paramp);
   pthread_sigmask(SIG_SETMASK, &saved_set, NULL);
#else
   ret = thrd_create(thrd, thread_routine, paramp);
#endif

   return ret;
}

void u_thread_setname( const char *name )
{
#if defined(HAVE_PTHREAD)
#if DETECT_OS_LINUX || DETECT_OS_CYGWIN || DETECT_OS_SOLARIS || defined(__GLIBC__) || DETECT_OS_MANAGARM || DETECT_OS_FUCHSIA
   int ret = pthread_setname_np(pthread_self(), name);
   if (ret == ERANGE) {
      char buf[16];
      const size_t len = MIN2(strlen(name), ARRAY_SIZE(buf) - 1);
      memcpy(buf, name, len);
      buf[len] = '\0';
      pthread_setname_np(pthread_self(), buf);
   }
#elif DETECT_OS_FREEBSD || DETECT_OS_OPENBSD
   pthread_set_name_np(pthread_self(), name);
#elif DETECT_OS_NETBSD
   pthread_setname_np(pthread_self(), "%s", (void *)name);
#elif DETECT_OS_APPLE
   pthread_setname_np(name);
#elif DETECT_OS_HAIKU
   rename_thread(find_thread(NULL), name);
#else
#warning Not sure how to call pthread_setname_np
#endif
#endif
   (void)name;
}

bool
util_set_thread_affinity(thrd_t thread,
                         const uint32_t *mask,
                         uint32_t *old_mask,
                         unsigned num_mask_bits)
{
#if defined(HAVE_PTHREAD_SETAFFINITY)
   cpu_set_t cpuset;

   if (old_mask) {
      if (pthread_getaffinity_np(thread, sizeof(cpuset), &cpuset) != 0)
         return false;

      memset(old_mask, 0, num_mask_bits / 8);
      for (unsigned i = 0; i < num_mask_bits && i < CPU_SETSIZE; i++) {
         if (CPU_ISSET(i, &cpuset))
            old_mask[i / 32] |= 1u << (i % 32);
      }
   }

   CPU_ZERO(&cpuset);
   for (unsigned i = 0; i < num_mask_bits && i < CPU_SETSIZE; i++) {
      if (mask[i / 32] & (1u << (i % 32)))
         CPU_SET(i, &cpuset);
   }
   return pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset) == 0;

#elif defined(_WIN32) && !defined(HAVE_PTHREAD)
   DWORD_PTR m = mask[0];

   if (sizeof(m) > 4 && num_mask_bits > 32)
      m |= (uint64_t)mask[1] << 32;

   m = SetThreadAffinityMask(thread.handle, m);
   if (!m)
      return false;

   if (old_mask) {
      memset(old_mask, 0, num_mask_bits / 8);

      old_mask[0] = m;
#ifdef _WIN64
      old_mask[1] = m >> 32;
#endif
   }

   return true;
#else
   return false;
#endif
}

int64_t
util_thread_get_time_nano(thrd_t thread)
{
#if defined(HAVE_PTHREAD) && !defined(__APPLE__) && !defined(__HAIKU__) && !defined(__managarm__)
   struct timespec ts;
   clockid_t cid;

   pthread_getcpuclockid(thread, &cid);
   clock_gettime(cid, &ts);
   return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
#elif defined(_WIN32)
   union {
      FILETIME time;
      ULONGLONG value;
   } kernel_time, user_time;
   GetThreadTimes((HANDLE)thread.handle, NULL, NULL, &kernel_time.time, &user_time.time);
   return (kernel_time.value + user_time.value) * 100;
#else
   (void)thread;
   return 0;
#endif
}

#if defined(HAVE_PTHREAD) && !defined(__APPLE__) && !defined(__HAIKU__)

void util_barrier_init(util_barrier *barrier, unsigned count)
{
   pthread_barrier_init(barrier, NULL, count);
}

void util_barrier_destroy(util_barrier *barrier)
{
   pthread_barrier_destroy(barrier);
}

bool util_barrier_wait(util_barrier *barrier)
{
   return pthread_barrier_wait(barrier) == PTHREAD_BARRIER_SERIAL_THREAD;
}

#else /* If the OS doesn't have its own, implement barriers using a mutex and a condvar */

void util_barrier_init(util_barrier *barrier, unsigned count)
{
   barrier->count = count;
   barrier->waiters = 0;
   barrier->sequence = 0;
   (void) mtx_init(&barrier->mutex, mtx_plain);
   cnd_init(&barrier->condvar);
}

void util_barrier_destroy(util_barrier *barrier)
{
   assert(barrier->waiters == 0);
   mtx_destroy(&barrier->mutex);
   cnd_destroy(&barrier->condvar);
}

bool util_barrier_wait(util_barrier *barrier)
{
   mtx_lock(&barrier->mutex);

   assert(barrier->waiters < barrier->count);
   barrier->waiters++;

   if (barrier->waiters < barrier->count) {
      uint64_t sequence = barrier->sequence;

      do {
         cnd_wait(&barrier->condvar, &barrier->mutex);
      } while (sequence == barrier->sequence);
   } else {
      barrier->waiters = 0;
      barrier->sequence++;
      cnd_broadcast(&barrier->condvar);
   }

   mtx_unlock(&barrier->mutex);

   return true;
}

#endif
