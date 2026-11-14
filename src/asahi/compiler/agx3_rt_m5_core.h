/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"

/* intersect_ray operands: a constant AS uses f63/f69=3; an AS selected per
 * invocation uses f62/f69=2 and f73 marks its last use. The first operand is
 * a GPR pair, copied from the runtime's uniform pair. Keep its name neutral
 * until the complete runtime pointer contract is mapped. All register
 * indices are in half-register units. */
struct agx3_rt_m5_intersect_desc {
   uint16_t address_register;
   uint16_t bvh_base;
   bool bvh_uniform;
   bool bvh_kill;
};

/* N simultaneously live queries use consecutive allocation indices
 * 4..4+N-1, observed for N = 2, 3, 4, 5 and 8. */
#define AGX3_RT_M5_MAX_LIVE_QUERIES 8

/* Live queries the lowering admits: every observed allocation index.
 *
 * On M5 hardware the only CTS cases with 3 to 8 simultaneously live
 * queries are ray_query.multiple_ray_queries.* (3, 4 and 7),
 * ray_query.stress.ahit_shader.* (3) and
 * ray_query.builtin.rayqueryterminate.* outside any-hit and intersection
 * shaders (8, from a split rayQueryEXT[8]). All 34 pass with this limit and
 * are refused with a limit of 2; other ray query, ray tracing pipeline and
 * acceleration-structure cases are unaffected. This is the number of native slots the
 * candidate-API lowering uses: logical queries beyond it share a slot
 * (agx_nir_m5_ray_query.c, m5_rq_slot), which is how the any-hit and
 * intersection rayqueryterminate variants (9 live queries) and
 * ray_query.misc.dynamic_indexing (48 elements, one selected per thread) are
 * compiled. Override with -DAGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES=N
 * (N <= 8). */
#ifndef AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES
#define AGX3_RT_M5_MAX_QUALIFIED_LIVE_QUERIES AGX3_RT_M5_MAX_LIVE_QUERIES
#endif

/* Native query stack entry: f2 is the 32-bit GPR result (an even half index)
 * and f24 the allocation index (4 + live query number). */
bool agx3_rt_m5_encode_enter_query(enum agx_isa_encoding encoding,
                                   uint16_t destination,
                                   uint16_t allocation_index, uint8_t out[16],
                                   uint8_t *length);

bool agx3_rt_m5_encode_intersect(enum agx_isa_encoding encoding,
                                const struct agx3_rt_m5_intersect_desc *desc,
                                uint8_t out[16], uint8_t *length);

/* The continue_ray instruction. Its fields do not specify the surrounding
 * polling/barrier protocol. */
bool agx3_rt_m5_encode_continue(enum agx_isa_encoding encoding,
                               uint8_t out[16], uint8_t *length);

/* Native Gen3 IFT preamble through LDSHDR (not the complete entry):
 * reload the context words and per-function constants,
 * then enter the separately allocated shader body. No shader body is embedded.
 * The two observed context layouts use constants at half24 or half28. */
#define AGX3_RT_M5_IFT_LOADER_MAX_SIZE 192u
struct agx3_rt_m5_ift_loader_desc {
   uint64_t constants_va;
   uint64_t shader_va;
   uint8_t constant_dwords;
   uint8_t fetch_span;
   bool context48;
};

bool agx3_rt_m5_ift_loader_generate(
   const struct agx3_rt_m5_ift_loader_desc *desc,
   uint8_t out[AGX3_RT_M5_IFT_LOADER_MAX_SIZE], size_t *length);

#define AGX3_RT_M5_IFT_ENTRY_MAX_SIZE 512u
#define AGX3_RT_M5_IFT_MAX_UNIFORMS 36u
struct agx3_rt_m5_ift_uniform {
   uint16_t uniform_half;
   uint16_t register_half;
};
struct agx3_rt_m5_ift_entry_desc {
   struct agx3_rt_m5_ift_loader_desc loader;
   const struct agx3_rt_m5_ift_uniform *uniforms;
   uint8_t uniform_count;
   /* The two observed TPR fields vary together; their semantics remain open. */
   bool tpr20;
   bool final_copy16;
};

/* Complete entry including resource state, call setup, uniform publication,
 * and END. The caller owns the separately allocated function body/constants. */
bool agx3_rt_m5_ift_entry_generate(
   const struct agx3_rt_m5_ift_entry_desc *desc,
   uint8_t out[AGX3_RT_M5_IFT_ENTRY_MAX_SIZE], size_t *length);

/* Native IFT type-1 hit commit. Uniform indices and all register operands use
 * half-register units. The two context uniforms retain numeric names until
 * their complete lifetime contract is known. */
#define AGX3_RT_M5_HIT_COMMIT_MAX_SIZE 256u
struct agx3_rt_m5_hit_commit_desc {
   uint16_t payload_uniform;
   uint16_t context6_uniform;
   uint16_t context8_uniform;
};
bool agx3_rt_m5_hit_commit_generate(
   const struct agx3_rt_m5_hit_commit_desc *desc,
   uint8_t out[AGX3_RT_M5_HIT_COMMIT_MAX_SIZE], size_t *length);
