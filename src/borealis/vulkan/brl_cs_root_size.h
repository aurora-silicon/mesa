/* SPDX-License-Identifier: MIT */
#pragma once

#include "asahi/lib/agx_chip.h"

#include <stdbool.h>
#include <stddef.h>

#define BRL_CS_ROOT_SIZE       65536u
/* Native G17P allocates its ordinary graphics control-stream root as one
 * 0x8000-byte object and concatenates draws in that root. When it fills, an
 * in-render VDM stream link continues into an upload chunk; a root-capacity
 * boundary is not a retained-render boundary.
 */
#define BRL_G17P_VDM_ROOT_SIZE    32768u
#define BRL_G17P_VDM_INITIAL_SIZE 12u
#define BRL_G17P_VDM_UPLOAD_CHUNK_SIZE 0x1000u
#define BRL_GFX_DRAW_RESERVATION  0x2000u
#define BRL_CS_SPACE_PADDING      0x800u

/* M5 roots stay small, so the few streams a command buffer opens share its
 * first command BO and a stream that outgrows its root links on. */
#define BRL_M5_CDM_ROOT_SIZE   16384u

static inline size_t
brl_cs_root_size(enum agx_chip chip, bool compute)
{
   if (agx_chip_g17_m5(chip))
      return compute ? BRL_M5_CDM_ROOT_SIZE : BRL_G17P_VDM_ROOT_SIZE;

   return agx_chip_native_agx3(chip) && !compute ? BRL_G17P_VDM_ROOT_SIZE
                                                 : BRL_CS_ROOT_SIZE;
}

static inline bool
brl_cs_space_fits(size_t remaining, size_t requested, size_t link_length)
{
   return requested <= remaining && link_length <= remaining - requested &&
          BRL_CS_SPACE_PADDING <= remaining - requested - link_length;
}

static inline bool
brl_graphics_cs_publishable(enum agx_chip chip, bool initialized,
                           bool active_context_selector_valid)
{
   return initialized &&
          (!agx_chip_native_agx3(chip) || active_context_selector_valid);
}
