/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The generated M5 intersection runtime builds, and both images match their
 * digests: every code row composes with agx3c_table, so a change to the
 * table or to agx3_compose() that alters the image fails here instead of at
 * device creation on an M5. The per-process copies devices use match a
 * fresh build and are built only once.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_m5_rt_runtime.h"

static int
check(const char *what, bool (*build)(uint8_t *),
      const uint8_t *(*cached)(void), uint8_t *image)
{
   if (!build(image)) {
      fprintf(stderr, "M5 intersection runtime: %s image does not build\n",
              what);
      return 1;
   }
   const uint8_t *copy = cached();
   if (!copy || memcmp(copy, image, HK_M5_RT_IMAGE_SIZE)) {
      fprintf(stderr, "M5 intersection runtime: cached %s image differs\n",
              what);
      return 1;
   }
   if (cached() != copy) {
      fprintf(stderr, "M5 intersection runtime: %s image is not kept\n",
              what);
      return 1;
   }
   return 0;
}

int
main(void)
{
   uint8_t *image = malloc(HK_M5_RT_IMAGE_SIZE);
   if (!image)
      return 77;

   int failures = 0;
   failures += check("code", hk_m5_rt_build_code, hk_m5_rt_code_image, image);
   failures += check("user", hk_m5_rt_build_user_image, hk_m5_rt_user_image,
                     image);
   free(image);
   return failures ? 1 : 0;
}
