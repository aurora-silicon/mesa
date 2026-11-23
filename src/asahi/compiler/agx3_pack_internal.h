/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Internal declarations and shared packer state for the AGX3 binary packer
 * modules.
 */

#ifndef AGX3_PACK_INTERNAL_H
#define AGX3_PACK_INTERNAL_H

#include "agx3_alu_core.h"
#include "agx3_pack.h"
#include "agx3_resource_ir.h"
#include "agx3_select_core.h"
#include "agx3_slot_core.h"
#include "agx3_staging.h"
#include "agx3_writeout_core.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "util/u_dynarray.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define AGX3_M5_FETCH_BANKS            8u
#define AGX3_M5_MS_FETCH_BANKS         8u
#define AGX3_TILE_ALL_SAMPLES          0xFFu

enum agx3_pack_result {
   AGX3_PACK_UNHANDLED = 0,
   AGX3_PACK_EMITTED = 1,
   AGX3_PACK_ERROR = -1,
};

struct agx3_branch_fixup {
   agx_block *block;
   off_t offset;
   bool skip_to_end;
   bool loop_exit;
   bool is_bn;
};

/* Early-compare bookkeeping: where each instruction's bytes start, and every
 * emitted control-flow compare. */
struct agx3_instr_span {
   const agx_instr *I;
   agx_block *block;
   off_t start;
   uint16_t wait;
   uint8_t slot;
};

struct agx3_early_compare {
   const agx_instr *I;
   agx_block *block;
   off_t instr_start, compare_start, compare_end;
   /* Spans recorded up to and including I's own. */
   unsigned span_count;
};

struct agx3_fetch_group {
   bool reuse;
   bool keep;
   uint16_t coord;
};

struct agx3_g17p_mrt_protocol {
   bool valid;
   bool raw32_store;
   bool has_loads;
   bool native_mixed;
   bool multisample;
   bool multisample_combined;
   uint8_t ms_tile_stride;
   uint16_t combined_classes;
   uint8_t count;
   uint8_t classes;
   uint8_t shared_first, shared_last;
   bool shared_loaded;
   uint8_t shared_load_first;
   uint8_t f13_stores;
   uint16_t offsets[AGX_MAX_RTS];
   uint8_t targets[AGX_MAX_RTS];
   uint8_t loads[AGX_MAX_RTS];
   uint8_t stores[AGX_MAX_RTS];
};

/*
 * Mutable packing state threaded through agx3_pack_binary() and each
 * per-category instruction emitter.
 */
struct agx3_pack_state {
   /* Core context */
   agx_context *ctx;
   struct util_dynarray *emission;
   enum agx_isa_encoding enc;
   unsigned initial_size;
   unsigned nslots;

   /* Scheduled instruction, slot, and resource tables */
   struct agx3_slot_instr *sl;
   struct agx3_slot_result *res;
   agx_instr **ordered;
   uint8_t *native_coord_slots;
   uint32_t *native_coord_values;
   struct agx3_resource_value *resources;
   struct agx3_resource_use *resource_uses;
   uint16_t *resource_bases;
   uint32_t *publication_values;
   uint32_t *resource_last_use;
   const bool *tps_rereads;
   unsigned count;
   unsigned resource_capacity;

   /* Branch fixups and early-compare rewriting */
   struct util_dynarray *fixups;
   struct util_dynarray *spans;
   struct util_dynarray *compares;
   bool early_compare;

   /* Live scoreboard and staging banks */
   uint16_t outstanding;
   uint16_t outstanding_here;
   uint16_t image_write_slots;
   struct agx3_staging_banks staging_banks;
   uint8_t next_uvs_slot;
   uint16_t fixed_staging_floor;
   uint16_t image_write_bank_floor;

   /* Current block and instruction cursor */
   agx_block *block;
   const agx_instr *I;
   unsigned instr_index;
   uint8_t slot;
   uint16_t wait;
   uint8_t native_coord_slot;
   off_t span_start;
   bool after_loop_backedge;
   bool after_loop_exit_test;
   struct agx3_fetch_group fetch_group;
   unsigned fixed_gradient_staging;
   unsigned loop_depth;

   /* Texture / image / scratch protocol state */
   unsigned texture_ops_left;
   unsigned image_writes_left;
   unsigned texture_samples_left;
   unsigned texture_samples_remaining;
   unsigned sample_group_members_left;
   unsigned sample_group_members_remaining;
   bool g17p_quad_op_after_group;
   bool quad_consumer_after_last_member;
   bool g17p_sample_group;
   bool g17g_fragment_group;
   bool g17p_storage_image_protocol;
   bool g18g_compact_image_banks;
   bool g18g_image_write_placed;
   bool g18g_image_read_bank0;
   uint16_t g18g_output_staging_base;
   uint16_t g18g_image_write_floor;
   uint16_t g18g_other_staging_end;
   uint16_t g18g_read_staging_end;
   uint16_t cem_scratch[3];
   uint16_t ms_array_scratch;
   uint16_t g17g_lod_relay;

   /* Fragment / tile / pixel / MRT / ZS protocol state */
   unsigned tile_loads_left;
   unsigned tile_loads_seen;
   unsigned tile_stores_left;
   bool has_coverage_write;
   bool has_memory_write;
   bool has_ray_traversal;
   bool has_zs_emit;
   bool zs_sample_loop;
   bool g17p_fragment_helper_protocol;
   bool fragment_input_started;
   bool interleaved_zs;
   bool g17p_multi_background;
   unsigned g17p_msaa_background_targets;
   bool g17p_msaa_background_stores;
   bool g18_ms_mrt_background;
   bool g17p_sample_colour_stores;
   bool g18_sample_colour_stores;
   bool g17p_colour_protocol;
   bool g17p_blend_protocol;
   bool g17p_pixel_protocol;
   bool g17p_colourless_fragment;
   bool fragment_ray_query;
   bool ray_query_acquired;
   bool g17p_mrt_acquired;
   bool g17p_colour_open;
   bool m5_sidefx_entry;
   unsigned g17p_lone_pair;
   unsigned g17p_link_packs_seen;
   unsigned g17p_mrt_stores_seen;
   unsigned g17p_zs_writes;
   unsigned g17p_zs_staging;
   const agx_instr *g17p_final_release;
   const agx_instr *g17g_final_signal;
   bool g17g_store_after_final_signal;
   const agx_instr *g17p_deferred_write;
   const agx_instr *g17g_deferred_write;
   const agx_instr *g17p_deferred_zs_wait;
   const agx_instr *m5_coverage_entry_before;
   struct agx3_g17p_mrt_protocol g17p_mrt;

   /* Vertex / mesh / memory / cprog state */
   unsigned device_loads_left;
   unsigned vertex_loads_seen;
   bool g17p_position_protocol;
   bool g17p_legacy_single_source;
   bool g17p_terminal_outputs;
   bool g17p_firefox_dense_outputs;
   bool g17p_user_output_protocol;
   bool g17p_vertex_load_policy;
   unsigned g17p_positions_staged;
   unsigned g17p_output_count;
   unsigned g17p_outputs_staged;
   uint8_t g17p_output_order[AGX3_MAX_VERTEX_OUTPUTS];
   bool gen1_mesh;
   bool mesh_allocated;
   bool mesh_indices_fenced;
   bool mesh_exports_pending;
   unsigned mesh_vertex_fences;
   bool cprog_imports;
   bool uniform_half_used[512];
};

/* Refusal reporting (implemented in agx3_pack_binary.c). */
static inline const char *
agx3_basename(const char *path)
{
   const char *slash = strrchr(path, '/');
   return slash ? slash + 1 : path;
}

void agx3_report_refusal_why(const agx_instr *I, const char *file,
                             unsigned line, const char *why);
void agx3_report_refusal(const agx_instr *I, const char *file, unsigned line);
bool agx3_tex_decline(const agx_instr *I, const char *file, unsigned line);

#define AGX3_REFUSE(I)                                                         \
   do {                                                                        \
      agx3_report_refusal((I), __FILE__, __LINE__);                            \
      goto fail;                                                               \
   } while (0)

#define AGX3_REFUSE_WHY(I, why)                                                \
   do {                                                                        \
      agx3_report_refusal_why((I), __FILE__, __LINE__, (why));                 \
      goto fail;                                                               \
   } while (0)

#define AGX3_REFUSE_RET(I)                                                     \
   do {                                                                        \
      agx3_report_refusal((I), __FILE__, __LINE__);                            \
      return AGX3_PACK_ERROR;                                                  \
   } while (0)

#define AGX3_REFUSE_WHY_RET(I, why)                                            \
   do {                                                                        \
      agx3_report_refusal_why((I), __FILE__, __LINE__, (why));                 \
      return AGX3_PACK_ERROR;                                                  \
   } while (0)

/* Shared inline low-level helpers. */
static inline bool
agx3_append(struct util_dynarray *emission, const uint8_t *bytes, uint8_t len)
{
   memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
   return true;
}

static inline bool
agx3_src_is_register(agx_index s)
{
   return s.type == AGX_INDEX_REGISTER;
}

static inline bool
agx3_src_is_zero_imm(agx_index s)
{
   return s.type == AGX_INDEX_IMMEDIATE && s.value == 0 && !s.memory &&
          !s.abs && !s.neg;
}

static inline bool
agx3_half_reg(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.size == AGX_SIZE_16 &&
          s.value < 256 && agx_channels(s) == 1;
}

static inline bool
agx3_vary_reg32(agx_index index)
{
   return agx3_src_is_register(index) && !index.memory &&
          index.size == AGX_SIZE_32 && (index.value & 1) == 0 &&
          index.value < 256 && !index.abs && !index.neg;
}

static inline bool
agx3_plain_immediate(agx_index x)
{
   return x.type == AGX_INDEX_IMMEDIATE && !x.memory && !x.kill && !x.cache &&
          !x.discard;
}

static inline bool
agx3_plain_uniform(agx_index x)
{
   return x.type == AGX_INDEX_UNIFORM && !x.memory && !x.abs && !x.neg &&
          !x.kill && !x.cache && !x.discard && x.value < 256 &&
          (x.size == AGX_SIZE_16 ||
           (x.size == AGX_SIZE_32 && !(x.value & 1)));
}

static inline bool
agx3_plain_uniform_wide(agx_index x)
{
   return x.type == AGX_INDEX_UNIFORM && !x.memory && !x.abs && !x.neg &&
          !x.kill && !x.cache && !x.discard && x.value < 512 &&
          (x.size == AGX_SIZE_16 ||
           (x.size == AGX_SIZE_32 && !(x.value & 1)));
}

static inline bool
agx3_mod_src_ok(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.size == AGX_SIZE_32 &&
          (s.value & 1) == 0 && s.value < 256;
}

static inline bool
agx3_mod_src_any_width(agx_index s)
{
   return agx3_src_is_register(s) && !s.memory && s.value < 256 &&
          agx_channels(s) == 1 &&
          ((s.size == AGX_SIZE_32 && (s.value & 1) == 0) ||
           s.size == AGX_SIZE_16);
}

static inline enum agx3_alu_hint
agx3_post_ra_hint(agx_index index)
{
   return index.discard ? AGX3_ALU_HINT_DISCARD
          : index.cache ? AGX3_ALU_HINT_CACHE
                        : AGX3_ALU_HINT_NONE;
}

static inline bool
agx3_tex_plain_reg(agx_index index)
{
   return agx3_src_is_register(index) && !index.memory && !index.abs &&
          !index.neg;
}

static inline bool
agx3_tex_imm(agx_index index, uint16_t value)
{
   return index.type == AGX_INDEX_IMMEDIATE && index.value == value &&
          !index.memory && !index.abs && !index.neg;
}

static inline bool
agx3_tex_plain_uniform64(agx_index index)
{
   return index.type == AGX_INDEX_UNIFORM && index.size == AGX_SIZE_64 &&
          agx_channels(index) == 1 && !(index.value & 1) &&
          index.value <= 252 && !index.memory && !index.abs && !index.neg &&
          !index.kill && !index.cache && !index.discard;
}

static inline bool
agx3_subgroup_plain(agx_index index, enum agx_index_type type)
{
   return index.type == type && !index.memory && !index.abs && !index.neg &&
          !index.kill && !(index.cache && index.discard) &&
          index.size == AGX_SIZE_32 && agx_channels(index) == 1 &&
          (index.value & 1) == 0 && index.value < 256;
}

static inline bool
agx3_stage_runs_as_kernel(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_COMPUTE || stage == MESA_SHADER_GEOMETRY ||
          stage == MESA_SHADER_TESS_CTRL || stage == MESA_SHADER_TESS_EVAL;
}

/* Control flow, wait retirement, fixups, and post-passes (agx3_pack_cf.c). */
void agx3_carrier_init(bool carrier1140);
bool agx3_carrier_is_1140(void);
void agx3_carrier_update_r0(uint16_t issued, uint16_t wait);
bool agx3_band_carrier(enum agx_isa_encoding enc, uint16_t reg, uint16_t mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
bool agx3_retire_slots_noop(struct util_dynarray *emission,
                            enum agx_isa_encoding enc, uint16_t mask);
bool agx3_retire_wait_on(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, agx_index reg,
                         uint16_t mask);
bool agx3_retire_result_wait(struct util_dynarray *emission,
                             enum agx_isa_encoding enc, const agx_instr *I,
                             uint16_t mask);
bool agx3_allocate_staging(struct util_dynarray *emission,
                           enum agx_isa_encoding enc, const agx_instr *I,
                           struct agx3_staging_banks *banks,
                           uint16_t *outstanding, uint8_t slot, unsigned span,
                           unsigned avoid_end, uint16_t *base);
bool agx3_is_cf_op(const agx_instr *I);
bool agx3_fixup_branch(struct util_dynarray *emission,
                       enum agx_isa_encoding enc,
                       struct agx3_branch_fixup fixup);
off_t agx3_fixup_target(const struct agx3_branch_fixup *f);
unsigned agx3_apply_loop_head_hints(agx_context *ctx,
                                    struct util_dynarray *emission);
unsigned agx3_apply_early_compares(agx_context *ctx,
                                   struct util_dynarray *emission,
                                   size_t initial_size,
                                   const struct util_dynarray *spans,
                                   const struct util_dynarray *compares,
                                   const struct util_dynarray *fixups);
enum agx3_pack_result agx3_pack_cf_instr(struct agx3_pack_state *st);

/* Texture / image / coord staging (agx3_pack_tex.c). */
bool agx3_tex_cube_groups_result_slot(const agx_instr *I);
bool agx3_is_staged_3d_bias_min(const agx_instr *I);
bool agx3_cube_own_scratch(const agx_instr *I);
const uint16_t *agx3_cube_scratch(const agx_instr *I, const uint16_t shared[3],
                                  uint16_t own[3]);
bool agx3_g17g_cube_gather(const agx_instr *I);
bool agx3_g17g_staged_block(const agx_instr *I, bool g18g);
bool agx3_g17g_register_lod(const agx_instr *I);
bool agx3_reusable_cube_scratch(agx_context *ctx, unsigned maximum,
                                uint16_t scratch[3]);
enum agx3_pack_result agx3_pack_tex_pre_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_tex_pub_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_tex_instr(struct agx3_pack_state *st);

/* Fragment varyings, UVSW vertex outputs, tilebuffer, pixel/MRT/ZS (agx3_pack_frag.c). */
bool agx3_g18_msaa4_tile_shader(const agx_context *ctx);
bool agx3_is_g17p_float2_position_vs(agx_context *ctx,
                                     enum agx_isa_encoding enc);
unsigned agx3_g17p_vertex_output_count(agx_context *ctx,
                                       enum agx_isa_encoding enc);
bool agx3_outputs_are_terminal(agx_context *ctx, unsigned count,
                               uint8_t order[AGX3_MAX_VERTEX_OUTPUTS]);
bool agx3_is_firefox_dense_output_suffix(agx_context *ctx, unsigned count);
bool agx3_outputs_leave_staging_alone(agx_context *ctx);
bool agx3_vertex_outputs_share_source(agx_context *ctx);
bool agx3_g17p_8norm_format(enum agx_format format);
const agx_instr *agx3_g17p_deferred_write_wait(agx_context *ctx, bool blend);
const agx_instr *agx3_g17g_deferred_colour_wait(agx_context *ctx);
bool agx3_g17p_lone_colour_pair(agx_context *ctx, unsigned *pair);
unsigned agx3_g17p_sample_store_protocol(agx_context *ctx, bool colour);
bool agx3_g18_ms_mrt_background_protocol(agx_context *ctx);
struct agx3_g17p_mrt_protocol agx3_g17p_mrt_protocol(agx_context *ctx);
bool agx3_g17p_multi_background_protocol(const agx_context *ctx);
void agx3_g17p_colour_link_info(agx_context *ctx, bool opaque_protocol);
enum agx3_pack_result agx3_pack_vary_uvsw_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_frag_pixel_instr(struct agx3_pack_state *st);

/* Memory, atomics, SR, RT/SCS/UVB, stack/spill (agx3_pack_mem.c). */
unsigned agx3_expand_regs(agx_index idx, uint16_t *arr, unsigned n);
bool *agx3_static_reload_rereads(agx_context *ctx, agx_instr **ordered,
                                 unsigned count);
bool agx3_dynamic_stack_load(const agx_instr *I);
bool agx3_append_static_spill_tokens(const agx_instr *I,
                                     struct agx3_slot_instr *slot_instr,
                                     enum agx_isa_encoding enc);
enum agx3_pack_result agx3_pack_mesh_fence_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_mem_instr(struct agx3_pack_state *st);

/* Subgroups, scoped barriers, and matrix/coopmat (agx3_pack_subgroup.c). */
bool agx3_m5_rendezvous_barrier(const agx_context *ctx,
                                enum agx_isa_encoding enc, unsigned scope);
enum agx3_pack_result agx3_pack_matrix_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_subgroup_instr(struct agx3_pack_state *st);

/* Float/int ALU, compares/selects, bitops, conversions, funops (agx3_pack_alu.c). */
bool agx3_emit(struct util_dynarray *emission, enum agx_isa_encoding enc,
               const struct agx3_sel_instr *S);
enum agx3_pack_result agx3_pack_uniform_float_instr(struct agx3_pack_state *st);
enum agx3_pack_result agx3_pack_alu_instr(struct agx3_pack_state *st);

#endif /* AGX3_PACK_INTERNAL_H */
