/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "compiler/nir/nir.h"
#include "drm-uapi/asahi_drm.h"

/* Use the same external-write predicate as nir_shader_gather_info, without
 * mutating shader info or instructions. Geometry clones can retain stale info
 * after their API stores are stripped, so inspect the selected program itself.
 * Vulkan inlines ordinary calls; follow any remaining direct helper calls too. */
static inline bool
hk_nir_function_writes_external_memory(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_intrinsic_writes_external_memory(nir_instr_as_intrinsic(instr)))
            return true;
         if (instr->type == nir_instr_type_call) {
            nir_function *callee = nir_instr_as_call(instr)->callee;
            if (callee->impl && hk_nir_function_writes_external_memory(callee->impl))
               return true;
         }
      }
   }
   return false;
}

/* XFB intent precedes its explicit buffer-store lowering. Native mesh/task
 * callers retain this classification before adding private transport stores. */
static inline bool
hk_shader_writes_external_memory(const nir_shader *nir)
{
   return nir->info.has_transform_feedback_varyings ||
          hk_nir_function_writes_external_memory(nir_shader_get_entrypoint(nir));
}

static inline void
hk_m5_accumulate_shader_writes(bool *command_writes, bool shader_writes)
{
   *command_writes |= shader_writes;
}

static inline uint32_t
hk_m5_shader_write_class_flags(uint64_t features, bool command_writes)
{
   return command_writes &&
          (features & DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES)
             ? DRM_ASAHI_RENDER_SHADER_WRITES : 0;
}
