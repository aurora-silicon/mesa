/*
 * Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */

/* Structs that cross an ABI boundary must have one layout on every host.
 * asahi_clc writes agx_precompiled_kernel_info on the build host for drivers
 * that may be 32-bit, and libagx structs are read by the GPU. i386 aligns
 * 8-byte members to only 4 inside structs, so such a struct only keeps its
 * layout there if no member offset and no tail padding needs more than 4-byte
 * alignment. Recompute each layout with i386 alignment and compare it with this
 * build's, so the mismatch shows up in a 64-bit build too.
 */

#include <assert.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>

#include "libagx/query.h"
#include "agx_compile.h"

struct member {
   const char *name;
   size_t offset, size, align;
};

#define M(S, m)                                                                \
   {#m, offsetof(S, m), sizeof(((S *)0)->m), alignof(__typeof__(((S *)0)->m))}

static unsigned failures;

static size_t
i386_align(size_t align)
{
   return align < 4 ? align : 4;
}

static size_t
align_up(size_t x, size_t align)
{
   return (x + align - 1) / align * align;
}

/* Members must be listed in declaration order; a missing one shows as a gap.
 * Member sizes are taken from this build, so a member that is itself a struct
 * needs its own CHECK, as agx_rodata has.
 */
static void
check(const char *name, size_t size, size_t align, const struct member *m,
      unsigned count)
{
   size_t end = 0;
   for (unsigned i = 0; i < count; ++i) {
      size_t i386_offset = align_up(end, i386_align(m[i].align));
      if (m[i].offset != i386_offset) {
         fprintf(stderr, "%s.%s: offset %zu here, %zu on i386\n", name,
                 m[i].name, m[i].offset, i386_offset);
         failures++;
      }
      end = m[i].offset + m[i].size;
   }

   size_t i386_size = align_up(end, i386_align(align));
   if (size != i386_size) {
      fprintf(stderr, "%s: %zu bytes here, %zu on i386\n", name, size,
              i386_size);
      failures++;
   }
}

#define CHECK(S, ...)                                                          \
   do {                                                                        \
      const struct member members[] = {__VA_ARGS__};                           \
      check(#S, sizeof(S), alignof(S), members,                                \
            sizeof(members) / sizeof(members[0]));                             \
   } while (0)

int
main(void)
{
   typedef struct libagx_imm_write imm;
   CHECK(imm, M(imm, address), M(imm, value), M(imm, pad));

   typedef struct agx_rodata rodata;
   CHECK(rodata, M(rodata, offset), M(rodata, base_uniform),
         M(rodata, size_16));

   typedef struct agx_precompiled_kernel_info info;
   CHECK(info, M(info, preamble_offset), M(info, main_offset),
         M(info, main_size), M(info, preamble_size), M(info, binary_size),
         M(info, rodata), M(info, nr_gprs), M(info, nr_preamble_gprs),
         M(info, push_count), M(info, workgroup_size), M(info, local_size),
         M(info, imageblock_stride), M(info, scratch_size),
         M(info, agx3_spectpr_scratch_size), M(info, uses_txf),
         M(info, agx3_staging_known), M(info, agx3_staging_halfs),
         M(info, agx3_preamble_staging_known),
         M(info, agx3_preamble_staging_halfs), M(info, printf_uniform),
         M(info, agx3_tps_size));

   if (failures)
      fprintf(stderr, "%u layout mismatches\n", failures);
   return failures ? 1 : 0;
}
