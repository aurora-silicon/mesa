/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The generated M5 intersection runtime builds, and both images match their
 * digests: every code row composes with agx3c_table, so a change to the
 * table or to agx3_compose() that alters the image fails here instead of at
 * device creation on an M5.
 */
#include <stdio.h>
#include <stdlib.h>
#include "hk_m5_rt_runtime.h"

int
main(void)
{
   uint8_t *image = malloc(HK_M5_RT_IMAGE_SIZE);
   if (!image)
      return 77;

   int failures = 0;
   if (!hk_m5_rt_build_code(image)) {
      fprintf(stderr, "M5 intersection runtime: code image does not build\n");
      failures++;
   }
   if (!hk_m5_rt_build_user_image(image)) {
      fprintf(stderr, "M5 intersection runtime: user image does not build\n");
      failures++;
   }
   free(image);
   return failures ? 1 : 0;
}
