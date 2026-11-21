/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/macros.h"
#include "util/os_misc.h"
#include "util/u_atomic.h"

/*
 * Borealis environment variables are named BRL_<NAME>. Before the driver was
 * renamed they were HK_<NAME>; that spelling is still read when BRL_<NAME> is
 * not set, so existing scripts keep working.
 *
 * Like os_get_option(), the result points into the environment: use it
 * before the environment can change, or copy it.
 */
static inline const char *
brl_env(const char *name)
{
   char key[64];

   snprintf(key, sizeof(key), "BRL_%s", name);
   const char *value = os_get_option(key);
   if (value)
      return value;

   snprintf(key, sizeof(key), "HK_%s", name);
   return os_get_option(key);
}

/* An environment entry ("NAME=value") under either Borealis prefix. */
static inline bool
brl_env_is_driver_var(const char *entry)
{
   return !strncmp(entry, "BRL_", 4) || !strncmp(entry, "HK_", 3);
}

/* Whether an environment entry is BRL_<name> or HK_<name>, comparing the
 * first len characters of entry (its name, without "=value"). */
static inline bool
brl_env_entry_is(const char *entry, size_t len, const char *name)
{
   if (len > 4 && !strncmp(entry, "BRL_", 4))
      return strlen(name) == len - 4 && !strncmp(entry + 4, name, len - 4);
   if (len > 3 && !strncmp(entry, "HK_", 3))
      return strlen(name) == len - 3 && !strncmp(entry + 3, name, len - 3);
   return false;
}

/* DEBUG_GET_ONCE_OPTION for brl_env(): read once, keep a private copy. */
#define BRL_GET_ONCE_ENV(fn, name)                                            \
   static const char *fn(void)                                                \
   {                                                                          \
      static bool initialized = false;                                        \
      static const char *value;                                               \
      if (unlikely(!p_atomic_read_relaxed(&initialized))) {                   \
         const char *str = brl_env(name);                                     \
         p_atomic_set(&value, str ? strdup(str) : NULL);                      \
         p_atomic_set(&initialized, true);                                    \
      }                                                                       \
      return value;                                                           \
   }
