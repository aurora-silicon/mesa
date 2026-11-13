/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "util/u_debug.h"
#include "util/u_math.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Independent, default-off traversal experiments. Keep these bit assignments
 * stable: benchmark scripts record the numeric mask. This is a code-generation
 * option and must participate in the Vulkan shader-cache identity.
 */
enum agx_g17p_rt_perf_arm {
   /* Allow ForceOpaque through the existing hardware-commit predicate. */
   AGX_G17P_RT_HW_OPAQUE = 1u << 0,
   /* Return after a burst fills eight of the ten allocated records. */
   AGX_G17P_RT_QUEUE_EIGHT = 1u << 1,
   /* Read the queue header once per batch, rather than per candidate. */
   AGX_G17P_RT_CACHE_CONTROL = 1u << 2,
   /* Publish tmax only when the software committed bound changed. */
   AGX_G17P_RT_DIRTY_TMAX = 1u << 3,
   /* Use the polling continuation publication form for queue refills. */
   AGX_G17P_RT_REFILL_RESUME = 1u << 4,
   /* Pad resident query slots without changing their internal fields. */
   AGX_G17P_RT_PAD_QUERY = 1u << 5,
   /* Test the runtime status byte directly before the polling loop. */
   AGX_G17P_RT_STATUS_FIRST = 1u << 6,
   /* Clear the hardware hit tag/ID packet before each refill. */
   AGX_G17P_RT_CLEAR_COMMIT = 1u << 7,
   /* Publish the closest bound to both RayCore banks. */
   AGX_G17P_RT_MIRROR_TMAX = 1u << 8,
   AGX_G17P_RT_PERF_ALL = (1u << 9) - 1,
};

static inline unsigned
agx_g17p_rt_perf_arms(bool a18_pro)
{
   return a18_pro ? debug_get_num_option("ASAHI_G17P_RT_PERF_ARM", 0) &
                       AGX_G17P_RT_PERF_ALL : 0;
}

/* RayCore banks each contain 4096 32-byte ray records. */
#define AGX_GEN1_RT_CORE_BANK_BYTES (4096u * 32u)

static inline unsigned
agx_gen1_rt_driver_bytes(unsigned arms)
{
   /* Padding changes only the spacing of resident warp slots. All lane
    * fields and the ten-record physical candidate buffer retain their ABI. */
   return (arms & AGX_G17P_RT_PAD_QUERY) ? 320 : 304;
}

static inline uint64_t
agx_gen1_rt_candidate_layout(unsigned arms)
{
   const uint64_t stack_depth = 112, levels = 1, cliques = 96;
   return stack_depth | (levels << 16) | (cliques << 24) |
          ((uint64_t)agx_gen1_rt_driver_bytes(arms) << 48);
}

static inline unsigned
agx_gen1_rt_queue_capacity(unsigned arms)
{
   return (arms & AGX_G17P_RT_QUEUE_EIGHT) ? 8 : 10;
}

/* The enqueue preflight reserves an entire eight-candidate burst. Changing
 * this threshold makes a completed batch return earlier; it never truncates
 * a produced queue or changes the 200-byte per-lane backing store.
 */
static inline void
agx_gen1_rt_queue_reserve(uint8_t *constants, unsigned word, unsigned arms)
{
   uint32_t reserve = util_cpu_to_le32(
      (agx_gen1_rt_queue_capacity(arms) - 8) << 21);
   memcpy(constants + word * sizeof(reserve), &reserve, sizeof(reserve));
}
