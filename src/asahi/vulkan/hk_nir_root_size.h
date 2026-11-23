/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "nir.h"
#include "util/macros.h"

/* Required prefix of the root table. Unknown/escaping accesses require full_size. */
unsigned hk_nir_root_size(const nir_shader *nir, unsigned full_size);

static inline uint32_t
hk_shader_root_descriptor_size(const nir_shader *nir,
                               mesa_shader_stage sw_stage,
                               unsigned set_count, unsigned sets_offset,
                               unsigned push_offset, unsigned full_size)
{
   unsigned bytes = hk_nir_root_size(nir, full_size);
   bytes = MAX2(bytes, sets_offset + set_count * (unsigned)sizeof(uint64_t));
   if (sw_stage != MESA_SHADER_COMPUTE)
      bytes = MAX2(bytes, push_offset);
   return MIN2(ALIGN_POT(bytes, 16), full_size);
}

static inline size_t
hk_gfx_root_size_from_stages(uint32_t vs_size, bool has_fs, uint32_t fs_size,
                             size_t full_size)
{
   if (!vs_size)
      return full_size;
   size_t size = vs_size;
   if (has_fs) {
      if (!fs_size)
         return full_size;
      size = MAX2(size, (size_t)fs_size);
   }
   return MIN2(size, full_size);
}

static inline bool
hk_root_upload_needs_growth(uint32_t uploaded_root_size,
                            size_t required_root_size)
{
   return uploaded_root_size < required_root_size;
}
