/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* DRM name of the AGX kernel driver. Kernels before the Aurora rename
 * report "asahi", so both are accepted and Mesa and the kernel can update in
 * either order. Every place that recognises the AGX GPU by its kernel name
 * (the driver, the loader, the pipe-loader, kmsro) takes it from here, so a
 * rename reaches all of them at once.
 */
#define AGX_DRM_DRIVER_NAME        "aurora"
#define AGX_DRM_DRIVER_NAME_LEGACY "asahi"

/* Both names, newest first, for lists of kernel driver names. */
#define AGX_DRM_DRIVER_NAMES AGX_DRM_DRIVER_NAME, AGX_DRM_DRIVER_NAME_LEGACY

static inline bool
agx_is_native_drm_driver(const char *name)
{
   return name && (!strcmp(name, AGX_DRM_DRIVER_NAME) ||
                   !strcmp(name, AGX_DRM_DRIVER_NAME_LEGACY));
}

/* Whether the asahi Gallium driver refuses a GPU, named by its device tree
 * compatible or by its chip name (agx_chip_name()); NULL for a name not
 * given. GL on such a GPU runs through zink over Borealis.
 *
 * The table is the one list of these GPUs. The loader matches a GPU's
 * compatible, read from sysfs without opening the device, and picks zink for
 * it (asahi_zink_predicate()); agx_screen_create() checks the chip of the
 * device it has opened, natively or under virtio. A GPU added here or removed
 * is refused or driven by both at once.
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
