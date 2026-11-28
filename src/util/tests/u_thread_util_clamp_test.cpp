/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

/* util_thread_raise_min_util_clamp() and UTIL_QUEUE_INIT_PREFER_BIG_CPUS.
 *
 * Each case runs in a forked child, so that the scheduler settings and seccomp
 * filters it applies do not reach the other tests. The child exits with 0 on
 * success, SKIP when the test itself runs under a seccomp filter (a container)
 * or the kernel does not support utilisation clamps, and
 * the number of the failed check otherwise.
 */

#include <gtest/gtest.h>

#include <errno.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>

#include "util/u_cpu_detect.h"
#include "util/u_queue.h"
#include "util/u_thread.h"

#define SKIP 77

namespace {

struct sched_attr_v1 {
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

constexpr uint64_t FLAG_RESET_ON_FORK = 0x01;
constexpr uint64_t FLAG_KEEP_ALL = 0x18;
constexpr uint64_t FLAG_UTIL_CLAMP_MIN = 0x20;
constexpr uint64_t FLAG_UTIL_CLAMP_MAX = 0x40;

/* The capacity the queue's threads ask for in these tests. */
constexpr unsigned TARGET = 512;

bool
get_attr(sched_attr_v1 *attr)
{
   memset(attr, 0, sizeof(*attr));
   return syscall(SYS_sched_getattr, 0, attr, sizeof(*attr), 0) == 0;
}

/* Set the calling thread's clamps, keeping its policy and parameters. */
bool
set_clamps(uint64_t flags, unsigned util_min, unsigned util_max)
{
   sched_attr_v1 attr;
   if (!get_attr(&attr))
      return false;
   attr.size = sizeof(attr);
   attr.sched_flags = FLAG_KEEP_ALL | flags;
   attr.sched_util_min = util_min;
   attr.sched_util_max = util_max;
   return syscall(SYS_sched_setattr, 0, &attr, 0) == 0;
}

/* Whether the calling thread runs under a seccomp filter (a container's, for
 * example), in which case util_thread_raise_min_util_clamp() does nothing.
 */
bool
under_seccomp(void)
{
   FILE *f = fopen("/proc/thread-self/status", "re");
   char line[128];
   bool filtered = true;

   if (!f)
      return true;
   while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "Seccomp:", 8) == 0) {
         filtered = strtol(line + 8, NULL, 10) != 0;
         break;
      }
   }
   fclose(f);
   return filtered;
}

/* Whether this thread can have its minimum clamp raised: no seccomp filter,
 * and a kernel that lets it set a minimum clamp.
 */
bool
have_util_clamp(void)
{
   sched_attr_v1 attr;
   return !under_seccomp() && get_attr(&attr) &&
          set_clamps(FLAG_UTIL_CLAMP_MIN, attr.sched_util_min, 0);
}

/* Make the CPU look like one with small CPUs to the queue. */
void
pretend_small_cpus(void)
{
   const struct util_cpu_caps_t *caps = util_get_cpu_caps();
   const_cast<struct util_cpu_caps_t *>(caps)->big_cpu_capacity = TARGET;
}

struct worker_attr {
   bool ok;
   sched_attr_v1 attr;
   int nice;
};

void
read_worker_attr(void *job, void *gdata, int thread_index)
{
   worker_attr *w = (worker_attr *)job;
   errno = 0;
   w->nice = getpriority(PRIO_PROCESS, 0);
   w->ok = get_attr(&w->attr) && errno == 0;
}

/* Create a PREFER_BIG_CPUS queue and read its thread's scheduler settings. */
bool
run_worker(worker_attr *w)
{
   struct util_queue queue;
   struct util_queue_fence fence;

   memset(w, 0, sizeof(*w));
   if (!util_queue_init(&queue, "uclamp", 4, 1,
                        UTIL_QUEUE_INIT_PREFER_BIG_CPUS, NULL))
      return false;
   util_queue_fence_init(&fence);
   util_queue_add_job(&queue, w, &fence, read_worker_attr, NULL, 0);
   util_queue_fence_wait(&fence);
   util_queue_fence_destroy(&fence);
   util_queue_destroy(&queue);
   return w->ok;
}

/* A filter that kills the process with SIGSYS on sched_getattr and
 * sched_setattr, whatever the target, as Chromium's GPU process policy does.
 */
bool
install_sched_attr_filter(void)
{
   struct sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_sched_getattr, 2, 0),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_sched_setattr, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
   };
   struct sock_fprog prog = {
      .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
      .filter = filter,
   };

   return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
          syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) == 0;
}

/* Run fn in a child process and return its wait status. */
template <typename F>
int
in_child(F fn)
{
   fflush(NULL);
   pid_t pid = fork();
   if (pid == 0)
      _exit(fn());
   int status = 0;
   if (pid < 0 || waitpid(pid, &status, 0) != pid)
      return -1;
   return status;
}

void
expect_exit_ok(int status)
{
   ASSERT_NE(status, -1);
   ASSERT_FALSE(WIFSIGNALED(status))
      << "child killed by signal " << WTERMSIG(status);
   ASSERT_TRUE(WIFEXITED(status));
   if (WEXITSTATUS(status) == SKIP)
      GTEST_SKIP() << "the thread runs under a seccomp filter or the kernel does not "
                      "support utilisation clamps";
   EXPECT_EQ(WEXITSTATUS(status), 0) << "check " << WEXITSTATUS(status) << " failed";
}

} /* namespace */

/* Under a filter that kills for sched_getattr/sched_setattr, neither the
 * helper nor a PREFER_BIG_CPUS queue thread may make those calls.
 */
TEST(util_thread_util_clamp, seccomp_filter_is_not_tripped)
{
   int status = in_child([] {
      pretend_small_cpus();
      if (!install_sched_attr_filter())
         return SKIP;

      if (util_thread_raise_min_util_clamp(TARGET))
         return 1;

      struct util_queue queue;
      struct util_queue_fence fence;
      if (!util_queue_init(&queue, "seccomp", 4, 1,
                           UTIL_QUEUE_INIT_PREFER_BIG_CPUS, NULL))
         return 2;
      util_queue_fence_init(&fence);
      util_queue_add_job(&queue, &queue, &fence,
                         [](void *, void *, int) {}, NULL, 0);
      util_queue_fence_wait(&fence);
      util_queue_fence_destroy(&fence);
      util_queue_destroy(&queue);
      return 0;
   });

   ASSERT_NE(status, -1);
   ASSERT_FALSE(WIFSIGNALED(status) && WTERMSIG(status) == SIGSYS)
      << "a scheduler call ran under the seccomp filter";
   ASSERT_FALSE(WIFSIGNALED(status)) << "child killed by signal " << WTERMSIG(status);
   ASSERT_TRUE(WIFEXITED(status));
   if (WEXITSTATUS(status) == SKIP)
      GTEST_SKIP() << "cannot install a seccomp filter here";
   EXPECT_EQ(WEXITSTATUS(status), 0) << "check " << WEXITSTATUS(status) << " failed";
}

/* Without a stronger clamp, the queue thread gets the target. */
TEST(util_thread_util_clamp, queue_thread_gets_target)
{
   expect_exit_ok(in_child([] {
      if (!have_util_clamp() || !set_clamps(FLAG_UTIL_CLAMP_MIN, 0, 0))
         return SKIP;
      pretend_small_cpus();

      worker_attr w;
      if (!run_worker(&w))
         return 1;
      if (w.attr.sched_util_min != TARGET)
         return 2;
      return 0;
   }));
}

/* A stronger minimum clamp the thread inherited is never lowered. */
TEST(util_thread_util_clamp, stronger_min_clamp_is_kept)
{
   expect_exit_ok(in_child([] {
      if (!have_util_clamp() || !set_clamps(FLAG_UTIL_CLAMP_MIN, 800, 0))
         return SKIP;
      pretend_small_cpus();

      worker_attr w;
      if (!run_worker(&w))
         return 1;
      if (w.attr.sched_util_min != 800)
         return 2;

      if (!util_thread_raise_min_util_clamp(TARGET))
         return 3;
      sched_attr_v1 attr;
      if (!get_attr(&attr) || attr.sched_util_min != 800)
         return 4;

      if (!set_clamps(FLAG_UTIL_CLAMP_MIN, 1024, 0) ||
          !util_thread_raise_min_util_clamp(TARGET) ||
          !get_attr(&attr) || attr.sched_util_min != 1024)
         return 5;
      return 0;
   }));
}

/* A maximum clamp below the target is kept, and so is the minimum. */
TEST(util_thread_util_clamp, lower_max_clamp_is_kept)
{
   expect_exit_ok(in_child([] {
      if (!have_util_clamp() ||
          !set_clamps(FLAG_UTIL_CLAMP_MIN | FLAG_UTIL_CLAMP_MAX, 100, 300))
         return SKIP;
      pretend_small_cpus();

      worker_attr w;
      if (!run_worker(&w))
         return 1;
      if (w.attr.sched_util_min != 100 || w.attr.sched_util_max != 300)
         return 2;
      return 0;
   }));
}

/* The thread's reset-on-fork setting is kept as it was, set or not, and so are
 * its policy and nice value.
 */
TEST(util_thread_util_clamp, reset_on_fork_is_kept)
{
   expect_exit_ok(in_child([] {
      if (!have_util_clamp() || !set_clamps(FLAG_UTIL_CLAMP_MIN, 0, 0))
         return SKIP;
      pretend_small_cpus();
      /* Raising the nice value is always allowed. */
      errno = 0;
      int nice = getpriority(PRIO_PROCESS, 0);
      if (errno)
         return 1;
      nice = nice < 19 ? nice + 1 : 19;
      if (setpriority(PRIO_PROCESS, 0, nice) != 0)
         return 1;

      sched_attr_v1 attr;
      if (!get_attr(&attr) || (attr.sched_flags & FLAG_RESET_ON_FORK))
         return 2;

      /* Not set on the queue's thread: it stays unset. */
      worker_attr w;
      if (!run_worker(&w))
         return 3;
      if (w.attr.sched_util_min != TARGET)
         return 4;
      if (w.attr.sched_flags & FLAG_RESET_ON_FORK)
         return 5;
      if (w.attr.sched_policy != SCHED_OTHER || w.nice != nice)
         return 6;

      /* Not set on the calling thread: it stays unset. */
      if (!util_thread_raise_min_util_clamp(TARGET) ||
          !get_attr(&attr) || attr.sched_util_min != TARGET ||
          (attr.sched_flags & FLAG_RESET_ON_FORK))
         return 7;

      /* Set on the calling thread: it stays set. */
      if (!set_clamps(FLAG_UTIL_CLAMP_MIN, 0, 0))
         return 8;
      struct sched_param param = {};
      if (sched_setscheduler(0, SCHED_OTHER | SCHED_RESET_ON_FORK, &param) != 0)
         return 9;
      if (!util_thread_raise_min_util_clamp(TARGET) ||
          !get_attr(&attr) || attr.sched_util_min != TARGET ||
          !(attr.sched_flags & FLAG_RESET_ON_FORK) ||
          getpriority(PRIO_PROCESS, 0) != nice)
         return 10;
      return 0;
   }));
}
