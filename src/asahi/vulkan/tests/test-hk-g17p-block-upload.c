/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Which buffer rows the whole-level RGBA8 upload through the compressed PBE
 * (hk_g17p_block_upload) addresses: the row pitch it takes, and the copies it
 * leaves to the render-pass upload.
 */
#include "hk_g17p_meta_copy_core.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

static unsigned failures;

static void
expect_pitch(uint32_t width, uint32_t row_length, bool ok, uint32_t pitch)
{
   uint32_t got = 0xdeadbeef;
   const bool got_ok = hk_g17p_block_upload_row_pitch(width, row_length, &got);
   if (got_ok != ok || (ok && got != pitch)) {
      fprintf(stderr, "width %u row length %u: got %s %u, want %s %u\n", width,
              row_length, got_ok ? "pitch" : "refused", got,
              ok ? "pitch" : "refused", pitch);
      failures++;
   }
}

int
main(void)
{
   /* Tightly packed rows, and a row length equal to the width */
   expect_pitch(2048, 0, true, 8192);
   expect_pitch(2048, 2048, true, 8192);
   expect_pitch(1, 0, true, 4);

   /* Padded rows */
   expect_pitch(129, 1024, true, 4096);
   expect_pitch(3840, 4096, true, 16384);

   /* A row length below the width is not a valid copy */
   expect_pitch(2048, 2047, false, 0);
   expect_pitch(0, 0, false, 0);

   /* A pitch whose row offsets pass 4 GiB: rows 8192 and up of a 16384-texel
    * level with 131072-texel rows start past 2^32 bytes, which the shader's
    * 64-bit row product reaches. */
   expect_pitch(16384, 131072, true, 524288);
   if ((uint64_t)16383 * 524288 <= UINT32_MAX) {
      fprintf(stderr, "the 16384 x 131072 case no longer passes 4 GiB\n");
      failures++;
   }

   /* The largest pitch the 32-bit push constant holds, and the first that
    * it does not: those copies keep the render-pass upload. */
   expect_pitch(1, (1u << 30) - 1, true, UINT32_MAX - 3);
   expect_pitch(1, 1u << 30, false, 0);
   expect_pitch(4096, UINT32_MAX, false, 0);

   if (failures)
      fprintf(stderr, "%u failures\n", failures);
   else
      printf("block upload row pitch: PASS\n");
   return failures ? 1 : 0;
}
