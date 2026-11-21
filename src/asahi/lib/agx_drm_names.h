/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Whether the asahi Gallium driver refuses a GPU, named by its device tree
 * compatible or by its chip name (agx_chip_name()); NULL for a name not
 * given. GL on such a GPU runs through zink over Borealis.
 *
 * The table is the one list of these GPUs, for everything that needs to know
 * them: agx_screen_create() checks the chip of the device it has opened,
 * natively or under virtio, and code that has not opened the device can
 * match the compatible. This header needs only the C library.
 */
static inline bool
agx_gallium_refuses_gpu(const char *compatible, const char *chip)
{
   static const struct {
      const char *compatible;
      const char *chip;
   } gpus[] = {
      {"apple,agx-g17d", "G17D"}, /* M5 Ultra */
   };

   for (size_t i = 0; i < sizeof(gpus) / sizeof(gpus[0]); i++) {
      if ((compatible && !strcmp(compatible, gpus[i].compatible)) ||
          (chip && !strcmp(chip, gpus[i].chip)))
         return true;
   }

   return false;
}
