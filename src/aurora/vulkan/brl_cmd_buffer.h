/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/macros.h"

#include "util/list.h"
#include "agx_abi.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "agx_uvs.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"

#include "aurora/compiler/agx3_ioto_uniform_program.h"
#include "brl_agx3_ioto_program_core.h"
#include "aurora/genxml/agx3_cmdstream.h"
#include "aurora/lib/agx3_compute_program_core.h"
#include "aurora/lib/agx_ppp_hal200_graphics_core.h"
#include "aurora/lib/agx_ppp_hal300_graphics_core.h"
#include "aurora/lib/agx_ppp_viewport_payload_core.h"
#include "aurora/lib/agx_vdm_hal200_bind_tail_core.h"
#include "aurora/lib/agx_va_runtime_hole.h"
#include "aurora/libagx/query.h"
#include "brl_query_oq.h"

#include "brl_private.h"
#include "brl_barrier.h"
#include "brl_conditional_render_core.h"
#include "brl_shader.h"
#include "brl_compute_pair.h"
#include "brl_sync.h"
#include "brl_tess_batch.h"

#include "brl_cmd_pool.h"
#include "brl_cs_root_size.h"
#include "brl_descriptor_set.h"
#include "brl_drm_render_core.h"
#include "brl_g17p_patch.h"
#include "brl_ioto_arena_core.h"
#include "brl_nir_root_size.h"
#include "brl_pipeline_statistics_core.h"

#include "aurora/lib/agx_nir_lower_vbo.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "vk_command_buffer.h"

#include <stdio.h>

/* Largest per-thread scratch frame of a kernel that may overlap the
 * application dispatches around it. 64 unordered dispatches of 4096 threads
 * take as long as ordered ones with 512 bytes per thread and five times as
 * long with 1 KB. */
#define BRL_M5_OVERLAP_MAX_SCRATCH_B 512

void *brl_cmd_buffer_emit_cdm_terminate(struct brl_cmd_buffer *cmd,
                                       void *current);
void *brl_cmd_buffer_emit_vdm_terminate(struct brl_cmd_buffer *cmd,
                                       void *current);

struct set;
struct brl_buffer;
struct brl_cmd_bo;
struct brl_cmd_pool;
struct brl_image_view;
struct brl_push_descriptor_set;
struct brl_ray_tracing_pipeline;
struct brl_api_shader;
struct brl_shader;
struct brl_linked_shader;
struct agx_usc_builder;
struct vk_shader;

/* Dynamic graphics state subset consumed by brl_g17p_build_native_bind_tail().
 * Kept separate from dyn->generation so unrelated dynamic state changes
 * (blend constants, vertex input, color write masks, etc.) never thrash the
 * HAL200 bind tail or its memo. */
struct brl_g17p_bind_tail_dyn_state {
   VkViewport viewports[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   VkRect2D scissors[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   struct vk_stencil_test_face_state stencil_front;
   struct vk_stencil_test_face_state stencil_back;
   uint32_t rasterization_stream;
   uint32_t line_width_raw;
   uint32_t depth_bias_constant_raw;
   uint32_t depth_bias_clamp_raw;
   uint32_t depth_bias_slope_raw;
   uint8_t depth_bias_representation;
   uint8_t viewport_count;
   uint8_t scissor_count;
   uint8_t depth_clip_enable;
   uint8_t polygon_mode;
   uint8_t cull_mode;
   uint8_t front_face;
   uint8_t line_mode;
   uint8_t depth_compare_op;
   uint8_t domain_origin;
   uint8_t rasterization_samples;
   bool depth_clip_negative_one_to_one;
   bool rasterizer_discard_enable;
   bool depth_clamp_enable;
   bool depth_bias_enable;
   bool depth_bias_exact;
   bool depth_test_enable;
   bool depth_write_enable;
   bool depth_bounds_test_enable;
   bool stencil_test_enable;
};

/* A complete but unpublished HAL200 state update.  The referenced VDM state
 * is command-buffer-owned.  The scissor/depth records stay inline until the
 * draw chooses its live stream, so preparation cannot perturb stream-local
 * indices or make firmware-visible bytes reachable early. */
struct brl_g17p_prepared_bind_tail {
   /* Hot per-draw validation fields grouped at the front so
    * brl_g17p_flush_dynamic_state_fast() and brl_g17p_emit_native_bind_tail()
    * stay in adjacent hot fields on the steady-state draw path. */
   uint64_t dynamic_generation;
   uint64_t descriptor_generation;
   uint64_t non_push_descriptor_generation;
   uint64_t vertex_buffer_generation;
   uint64_t root;
   /* The fragment shader variant the tail launches, its published stage
    * program address, and the uniform-DMA set base its program was built
    * with (brl_g17p_dma_set_base()). */
   const struct brl_shader *fragment_shader;
   uint64_t fragment_program;
   uint64_t fragment_dma_set_base;
   struct brl_linked_shader *linked_vs;
   struct brl_linked_shader *linked_fs;
   uint32_t cull_word;
   uint32_t render_view_mask;
   uint8_t render_color_count;
   uint8_t render_samples;
   uint8_t viewport_count;
   /* Cached set index returned by brl_g17p_dma_set_index(fragment_shader), or
    * -1 when the fragment shader has no set-sourced uniform DMA span. */
   int8_t fragment_dma_set_index;
   bool in_meta;
   /* Metadata fields above (including fragment_program and cull_word) match
    * the active state. */
   bool valid;
   /* prepared->tail's embedded FS and CULL records still match
    * fragment_program and cull_word (false after an incremental standalone
    * PPP update until a full bind tail is rebuilt). */
   bool fs_record_in_tail_valid;

   /* Cold state compared only on full bind-tail checks or rebuilds. */
   struct brl_image_view *render_color0;
   struct brl_image_view *render_depth;
   struct brl_image_view *render_stencil;
   VkRect2D render_area;
   VkFormat render_depth_format;
   VkFormat render_stencil_format;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t render_layer_count;
   struct brl_api_shader *shaders[MESA_SHADER_MESH + 1];
   uint64_t varyings;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint32_t occlusion_index;
   uint8_t topology;
   uint8_t object_type;
   uint8_t provoking;
   uint8_t occlusion_mode;
   uint8_t effective_fragment_pass_type;
   bool generate_primitive_id;
   bool force_never_in_shader;
   bool rasterizer_discard;
   struct agx_vdm_hal200_bind_tail tail;
   uint8_t scissors[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                   [AGX_SCISSOR_LENGTH];
   uint8_t depth_bias[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                    [AGX_DEPTH_BIAS_LENGTH];
};

/** Root descriptor table. */
struct brl_root_descriptor_table {
   /* Address of this descriptor itself. Must be first for reflection. */
   uint64_t root_desc_addr;

   /* Descriptor set base addresses. Must follow root_desc_addr to match our
    * push layout.
    */
   uint64_t sets[BRL_MAX_SETS];

   union {
      struct {
         uint32_t view_index;
         uint32_t ppp_multisamplectl;

         /* Vertex input state, indexed by compacted attribute slot. The
          * bases live at the end of this arm (attrib_base below): the clamp
          * and stride tables at 32 entries exactly fill the 256 bytes the
          * three 16-entry tables used to, which keeps every later field --
          * notably the pinned pipeline_stats offsets -- where it was.
          */
         uint32_t attrib_clamps[AGX_MAX_VBUFS];
         uint32_t attrib_strides[AGX_MAX_VBUFS];

         /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
         uint64_t vertex_outputs;

         /* Address of vertex param buffer if geom/tess is used, else 0 */
         uint64_t vertex_params;

         /* Address of tessellation param buffer if tessellation used, else 0 */
         uint64_t tess_params;

         /* Address of geometry param buffer if GS is used, else 0 */
         uint64_t geometry_params;

         /* Pipeline statistics queries. This is a base address with flags. */
         uint64_t pipeline_stats;
         VkQueryPipelineStatisticFlags pipeline_stats_flags;

         float blend_constant[4];
         uint16_t no_epilog_discard;
         uint16_t _pad1;
         uint16_t api_sample_mask;
         uint16_t _pad2;
         uint16_t force_never_in_shader;
         uint16_t _pad3;
         uint16_t provoking;
         uint16_t _pad4;

         /* True if there is an API geometry shader. If false, there may still
          * be a geometry shader in use (notably for transform feedback) but it
          * should not contribute to pipeline statistics.
          */
         uint16_t api_gs;
         uint16_t _pad5;

         uint16_t rasterization_stream;
         uint16_t _pad6;

         /* Alpha-to-coverage enable for fragment shaders that write the
          * sample mask; they apply it before their own coverage update.
          * Sysval fields must sit at 4-byte offsets (root table loads). */
         uint16_t alpha_to_coverage;

         /* The pass has no attachments, so no storage backs its layers: the
          * hardware sees layer zero, while a fragment shader reading the layer
          * still gets what the vertex stage wrote. */
         uint16_t no_attachments;

         /* Mapping from varying slots written by the last vertex stage to UVS
          * indices. This mapping must be compatible with the fragment shader.
          */
         uint8_t uvs_index[VARYING_SLOT_MAX];

         /* GPU addresses of the immutable NIR constant-data blobs of the
          * bound vertex and fragment shaders, or zero if they have none. */
         uint64_t vs_constant_data_addr;
         uint64_t fs_constant_data_addr;

         /* Vertex attribute base addresses, indexed by compacted slot. */
         uint64_t attrib_base[AGX_MAX_VBUFS];
      } draw;
      struct {
         /* GPU address of the immutable NIR constant-data blob for the
          * currently bound compute shader. */
         uint64_t constant_data_addr;
         uint64_t group_count_addr;
         uint32_t base_group[3];

         /* Native G17P indirect dispatch only.  The grid helper reads the
          * threadgroup counts at group_count_addr, multiplies them by
          * grid_helper_local_size, and publishes the six-u32 launch geometry
          * at grid_helper_geometry_addr for the main record to consume.  For
          * an indirect-local grid, group_count_addr instead holds a launch
          * geometry, which the geometry copy helper publishes as it is.  Both
          * are zero for every direct dispatch and for every other chip. */
         uint64_t grid_helper_geometry_addr;

         /* Native G17P indirect dispatch whose argument buffer is a sparse
          * residency buffer: the base of that buffer's residency bitmap (a
          * struct brl_sparse_buffer_map_header followed by one bit per page),
          * and zero otherwise.  When it is set, group_count_addr names the
          * PRIMARY mapping rather than the read-only shadow, and the grid
          * helper reads the bitmap to decide whether the arguments are
          * resident, publishing a zero grid when they are not.  See
          * brl_CmdDispatchIndirect() for why the shadow cannot be read
          * directly here. */
         uint64_t grid_helper_sparse_map_addr;

         uint32_t grid_helper_local_size[3];
         uint32_t _pad_grid_helper;

         /* VK_KHR_ray_tracing_pipeline: device address of the
          * VkTraceRaysIndirectCommand2KHR record the megakernel reads its
          * shader binding tables and launch size from.  Zero for every
          * ordinary compute dispatch; the ray-tracing megakernel is the only
          * shader that loads it. */
         uint64_t trace_rays_addr;
      } cs;
   };

   /* Client push constants */
   uint8_t push[BRL_MAX_PUSH_SIZE];

   /* Dynamic buffer bindings */
   struct brl_buffer_address dynamic_buffers[BRL_MAX_DYNAMIC_BUFFERS];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[BRL_MAX_SETS];

   /* Process-local on G17P; loaded through the root instead of cached code. */
   uint64_t printf_buffer_address;
   uint32_t mesh_draw_id;
   uint64_t mesh_grid_addr;
   uint64_t mesh_prims_query;
   uint64_t sampler_heap_address;
   uint64_t physical_sparse_map;
   uint64_t physical_sparse_base;
   uint64_t physical_sparse_size;

   /* Upper four sample positions on M5. It follows every agx3b root field,
    * so the draw arm, and with it every field after the union, keeps the
    * agx3b layout. */
   uint32_t ppp_multisamplectl_hi;

   /* M5 task draws whose grid, count or predicate live in GPU memory:
    * address of the word libagx_m5_task_dispatch writes (nonzero = execute);
    * zero for every other draw. Appended after every agx3b field. */
   uint64_t mesh_task_flag;

   /* VK_EXT_depth_clip_control: 1/2 maps clip z from [-w, w] to [0, w] in the
    * last pre-rasterization stage, 0 leaves it (agx_nir_lower_clip_m1_1). */
   float clip_z_coeff;

   /* G17P: which launch of a mesh draw this is (BRL_MESH_LAUNCH_WGS). */
   uint32_t mesh_launch;

   /* G17P and M5 mesh lines: 1/64 pixel along +y in NDC for viewport 0,
    * signed by its y direction (agx_nir_lower_gen1_mesh, m5_mesh_line_bias). */
   float line_y_bias;

   /* G17P mesh-only draw: nonzero when the draw's grid is known at recording
    * and fits one launch as it is (brl_mesh_launch_whole()), so the mesh
    * workgroup IDs are the API's without reading the grid. */
   uint32_t mesh_whole;
};

/* The graphics arm is the larger of the two, so everything after the union --
 * push constants, dynamic buffers, the printf address -- keeps its offset when
 * a compute field is added. Growing the compute arm past it would move all of
 * them at once, which no single change should do silently. */
static_assert(sizeof(((struct brl_root_descriptor_table *)0)->cs) <=
                 sizeof(((struct brl_root_descriptor_table *)0)->draw),
              "compute root fields must not grow the root descriptor table");

/* helper macro for computing root descriptor byte offsets */
#define brl_root_descriptor_offset(member)                                      \
   offsetof(struct brl_root_descriptor_table, member)

/* Counts only these CPU upload paths, enabled by ASAHI_MESA_DEBUG=perf. */
struct brl_upload_stats {
   uint64_t draws, descriptor_binds, descriptor_noops;
   uint64_t root_uploads, root_bytes, root_reuses;
   uint64_t draw_param_uploads, draw_param_bytes, draw_param_reuses;
   uint64_t draw_id_uploads, draw_id_bytes, draw_id_reuses;
   uint64_t constant_uploads, constant_bytes, constant_reuses;
   uint64_t xfer_desc_walks;
};

#define brl_upload_stat(cmd, field, value)                                      \
   do {                                                                       \
      if (unlikely(brl_cmd_buffer_device(cmd)->dev.debug & AGX_DBG_PERF))         \
         (cmd)->state.upload_stats.field += (value);                            \
   } while (0)

/* Root bytes the fragment stage reads only through these classes of access,
 * recorded per shader in brl_shader_info::agx3_root_reads. */
enum brl_root_reads {
   /* The root's own address: the shader can then reach any root byte. */
   BRL_ROOT_READS_ADDRESS = BITFIELD_BIT(0),
   /* Vertex attribute bases, clamps and strides, and the vertex shader's
    * constant data. */
   BRL_ROOT_READS_VERTEX_INPUT = BITFIELD_BIT(1),
   BRL_ROOT_READS_DYNAMIC_BUFFERS = BITFIELD_BIT(2),
   /* Push constants: vertex-only push updates leave fragment-visible root
    * bytes unchanged unless the fragment shader reads push constants. */
   BRL_ROOT_READS_PUSH = BITFIELD_BIT(3),
   /* Compute-only positive proof (inverted polarity so older cached binaries
    * default to 0 = conservative): API workgroup has <= 256 invocations (a
    * multiple of 32), depends only on global invocation ID / zero base_group
    * (no shared/tile/scratch/ray/matrix/barrier/subgroup/local/workgroup
    * builtins, and never observes group_count_addr), so direct zero-base
    * dispatches may launch in coalesced hardware workgroups up to 256/512
    * threads that evenly divide the thread grid. */
   BRL_ROOT_PROVEN_WG_COALESCE_SAFE = BITFIELD_BIT(4),
};

/* Every root byte a draw may republish without the fragment stage observing
 * it unless the fragment shader reads that class (brl_agx3_fragment_root()), as
 * X(from, to, class) spans in ascending order. The vertex-input spans hold
 * just their fields: the clamp and stride tables are adjacent and followed by
 * vertex_outputs, and attrib_base ends the graphics arm. */
#define BRL_ROOT_FRAGMENT_HIDDEN_SPANS(X)                                       \
   X(root_desc_addr, sets, BRL_ROOT_READS_ADDRESS)                              \
   X(draw.attrib_clamps, draw.vertex_outputs, BRL_ROOT_READS_VERTEX_INPUT)      \
   X(draw.vs_constant_data_addr, draw.fs_constant_data_addr,                   \
     BRL_ROOT_READS_VERTEX_INPUT)                                               \
   X(draw.attrib_base, push, BRL_ROOT_READS_VERTEX_INPUT)                       \
   X(push, dynamic_buffers, BRL_ROOT_READS_PUSH)                                \
   X(dynamic_buffers, set_dynamic_buffer_start, BRL_ROOT_READS_DYNAMIC_BUFFERS)

/* The BRL_ROOT_READS_* classes a read of root bytes [start_B, end_B) falls
 * in. */
static inline uint8_t
brl_root_reads_of(uint64_t start_B, uint64_t end_B)
{
   uint8_t reads = 0;
#define BRL_ROOT_CLASSIFY(from, to, class)                                      \
   if (start_B < brl_root_descriptor_offset(to) &&                              \
       end_B > brl_root_descriptor_offset(from))                                \
      reads |= (class);
   BRL_ROOT_FRAGMENT_HIDDEN_SPANS(BRL_ROOT_CLASSIFY)
#undef BRL_ROOT_CLASSIFY
   return reads;
}

/* Compare root bytes [start, end) without reading past either published size;
 * bytes beyond a trimmed HAL200 root's size are treated as zero. */
static inline bool
brl_root_span_matches(const uint8_t *a, size_t a_size, const uint8_t *b,
                     size_t b_size, size_t start, size_t end)
{
   const size_t a_end = start < a_size ? MIN2(end, a_size) : start;
   const size_t b_end = start < b_size ? MIN2(end, b_size) : start;
   const size_t common = MIN2(a_end, b_end);
   if (common > start && memcmp(a + start, b + start, common - start))
      return false;
   for (size_t i = common; i < a_end; ++i) {
      if (a[i] != 0)
         return false;
   }
   for (size_t i = common; i < b_end; ++i) {
      if (b[i] != 0)
         return false;
   }
   return true;
}

/* Whether two roots agree on every byte outside the hidden spans. */
static inline bool
brl_root_matches_for_fragment(const struct brl_root_descriptor_table *a,
                             size_t a_size,
                             const struct brl_root_descriptor_table *b,
                             size_t b_size)
{
   const uint8_t *ab = (const uint8_t *)a;
   const uint8_t *bb = (const uint8_t *)b;
   const size_t a_lim = MIN2(a_size, sizeof(*a));
   const size_t b_lim = MIN2(b_size, sizeof(*b));
   size_t at = 0;
#define BRL_ROOT_COMPARE_BEFORE(from, to, class)                                \
   if (!brl_root_span_matches(ab, a_lim, bb, b_lim, at,                         \
                             brl_root_descriptor_offset(from)))                 \
      return false;                                                            \
   at = brl_root_descriptor_offset(to);
   BRL_ROOT_FRAGMENT_HIDDEN_SPANS(BRL_ROOT_COMPARE_BEFORE)
#undef BRL_ROOT_COMPARE_BEFORE
   return brl_root_span_matches(ab, a_lim, bb, b_lim, at, sizeof(*a));
}

/* Whether two roots agree on their push-constant bytes. */
static inline bool
brl_root_matches_push(const struct brl_root_descriptor_table *a, size_t a_size,
                     const struct brl_root_descriptor_table *b, size_t b_size)
{
   return brl_root_span_matches((const uint8_t *)a, MIN2(a_size, sizeof(*a)),
                               (const uint8_t *)b, MIN2(b_size, sizeof(*b)),
                               brl_root_descriptor_offset(push),
                               brl_root_descriptor_offset(dynamic_buffers));
}

/* Build the same visible-byte sequence as brl_root_matches_for_fragment().
 * Keeping this derived from the shared span list includes future root fields
 * and padding without duplicating the root ABI. Push constants sit alongside
 * bytes so shaders without BRL_ROOT_READS_PUSH skip them while push-reading
 * shaders still reuse roots across vertex-input and dynamic-buffer updates. */
#define BRL_ROOT_FRAGMENT_SUBTRACT(from, to, class)                             \
   -(brl_root_descriptor_offset(to) - brl_root_descriptor_offset(from))
enum {
   BRL_ROOT_FRAGMENT_VISIBLE_SIZE = sizeof(struct brl_root_descriptor_table)
      BRL_ROOT_FRAGMENT_HIDDEN_SPANS(BRL_ROOT_FRAGMENT_SUBTRACT),
};
#undef BRL_ROOT_FRAGMENT_SUBTRACT

struct brl_root_fragment_snapshot {
   uint8_t bytes[BRL_ROOT_FRAGMENT_VISIBLE_SIZE];
   uint8_t push[BRL_MAX_PUSH_SIZE];
};

static inline void
brl_root_fragment_snapshot_store(struct brl_root_fragment_snapshot *snapshot,
                                const struct brl_root_descriptor_table *root,
                                uint32_t root_size)
{
   const size_t limit = MIN2((size_t)root_size, sizeof(*root));
   size_t at = 0, out = 0;
#define BRL_ROOT_FRAGMENT_COPY(from, to, class)                                 \
   do {                                                                        \
      const size_t end = brl_root_descriptor_offset(from);                      \
      const size_t size = end - at;                                            \
      const size_t copy = at < limit ? MIN2(end, limit) - at : 0;              \
      if (copy)                                                                \
         memcpy(snapshot->bytes + out, (const uint8_t *)root + at, copy);      \
      if (copy < size)                                                         \
         memset(snapshot->bytes + out + copy, 0, size - copy);                 \
      out += size;                                                             \
      at = brl_root_descriptor_offset(to);                                      \
   } while (0);
   BRL_ROOT_FRAGMENT_HIDDEN_SPANS(BRL_ROOT_FRAGMENT_COPY)
#undef BRL_ROOT_FRAGMENT_COPY
   const size_t tail_size = sizeof(*root) - at;
   const size_t tail_copy = at < limit ? limit - at : 0;
   if (tail_copy)
      memcpy(snapshot->bytes + out, (const uint8_t *)root + at, tail_copy);
   if (tail_copy < tail_size)
      memset(snapshot->bytes + out + tail_copy, 0, tail_size - tail_copy);
   assert(out + tail_size == sizeof(snapshot->bytes));

   const size_t push_off = brl_root_descriptor_offset(push);
   const size_t push_copy =
      push_off < limit ? MIN2(sizeof(snapshot->push), limit - push_off) : 0;
   if (push_copy)
      memcpy(snapshot->push, (const uint8_t *)root + push_off, push_copy);
   if (push_copy < sizeof(snapshot->push))
      memset(snapshot->push + push_copy, 0, sizeof(snapshot->push) - push_copy);
}

static_assert(brl_root_descriptor_offset(draw.attrib_strides) ==
                    brl_root_descriptor_offset(draw.attrib_clamps) +
                       sizeof(((struct brl_root_descriptor_table *)0)
                                 ->draw.attrib_clamps) &&
                 brl_root_descriptor_offset(draw.vertex_outputs) ==
                    brl_root_descriptor_offset(draw.attrib_strides) +
                       sizeof(((struct brl_root_descriptor_table *)0)
                                 ->draw.attrib_strides),
              "only the clamp and stride tables precede vertex_outputs");
static_assert(brl_root_descriptor_offset(draw.fs_constant_data_addr) ==
                 brl_root_descriptor_offset(draw.vs_constant_data_addr) +
                    sizeof(uint64_t),
              "the fragment constant data follows the vertex constant data");
static_assert(brl_root_descriptor_offset(push) ==
                 brl_root_descriptor_offset(draw.attrib_base) +
                    sizeof(((struct brl_root_descriptor_table *)0)
                              ->draw.attrib_base),
              "attrib_base ends the graphics arm");
static_assert(brl_root_descriptor_offset(dynamic_buffers) ==
                 brl_root_descriptor_offset(push) +
                    sizeof(((struct brl_root_descriptor_table *)0)->push),
              "push constants are followed by dynamic buffers");
static_assert(brl_root_descriptor_offset(set_dynamic_buffer_start) ==
                 brl_root_descriptor_offset(dynamic_buffers) +
                    sizeof(((struct brl_root_descriptor_table *)0)
                              ->dynamic_buffers),
              "the dynamic buffer table is followed by its set starts");

struct brl_descriptor_state {
   /* How root differs from the last uploaded root: read and written only
    * through brl_root_dirty() and the other brl_root_*() helpers below.
    * push_only is set while every change since that upload was to push
    * constants, push_launched while the vertex launch last published reads
    * the current push constants from an upload of their own. */
   struct {
      bool dirty;
      bool push_only;
      bool push_launched;
   } root_change;
   uint64_t generation;
   /* Advances on every descriptor state change except brl_push_constants()
    * (set binds, dynamic offsets, push descriptors, descriptor buffers, meta
    * restores), so push-only launches and xfer_clean can track descriptor
    * resource mutations in O(1). */
   uint64_t non_push_generation;
   uint64_t identity_copy4_push_mask;
   struct brl_root_descriptor_table root;
   /* Immutable command-buffer storage; reset with descriptor state. Only the
    * uploaded head and tail are compared, excluding unused dynamic slots. */
   struct agx_ptr uploaded_root;
   uint32_t uploaded_root_size;
   uint64_t uploaded_root_generation;
   uint8_t uploaded_dynamic_buffer_end;

   /* One past the highest root.dynamic_buffers entry any bind wrote. Later
    * entries are unread by every bound layout and stay out of uploads. */
   uint8_t dynamic_buffer_end;

   uint32_t set_sizes[BRL_MAX_SETS];
   struct brl_descriptor_set *sets[BRL_MAX_SETS];
   uint32_t set_update_seq[BRL_MAX_SETS];
   uint32_t sets_dirty;

   struct brl_push_descriptor_set *push[BRL_MAX_SETS];
   uint32_t push_dirty;
   uint32_t push_active;

   /* Sets bound from a descriptor buffer (or embedded immutable samplers):
    * root.sets[] holds their address, and their contents may change until
    * the commands execute, so recording never reads them. */
   uint32_t buffer_active;

   /* Bitmask of 32-bit push-constant DWORDs written in this command buffer. */
   uint64_t push_written_dwords;
};

/* Whether root differs from the last uploaded root. */
static inline bool
brl_root_dirty(const struct brl_descriptor_state *desc)
{
   return desc->root_change.dirty;
}

/* Whether root differs from the last uploaded root in push constants alone. */
static inline bool
brl_root_push_only_dirty(const struct brl_descriptor_state *desc)
{
   return desc->root_change.dirty && desc->root_change.push_only;
}

/* Whether the vertex launch last published reads every root byte a draw
 * needs: the root is as uploaded, or only push constants changed since and
 * that launch reads them from their own upload (brl_root_mark_push_launched()).
 * push_launch_fits says whether the draw could take such a launch
 * (brl_agx3_try_push_only_vs_program()). The root itself stays dirty either
 * way, so every path that reads it from the uploaded copy uploads it first. */
static inline bool
brl_root_launch_current(const struct brl_descriptor_state *desc,
                       bool push_launch_fits)
{
   return !desc->root_change.dirty ||
          (push_launch_fits && desc->root_change.push_launched);
}

/* Root changed in anything but push constants, or in unknown bytes. */
static inline void
brl_root_mark_dirty(struct brl_descriptor_state *desc)
{
   desc->root_change.dirty = true;
   desc->root_change.push_only = false;
   desc->root_change.push_launched = false;
}

/* Root changed in its push constants. An earlier change of any other kind
 * since the last upload keeps the root fully dirty. */
static inline void
brl_root_mark_push_dirty(struct brl_descriptor_state *desc)
{
   if (!desc->root_change.dirty) {
      desc->root_change.dirty = true;
      desc->root_change.push_only = true;
   }
   desc->root_change.push_launched = false;
}

/* The vertex program was launched over the current push constants alone,
 * and the uploaded root, which still holds older ones, was kept. */
static inline void
brl_root_mark_push_launched(struct brl_descriptor_state *desc)
{
   assert(brl_root_push_only_dirty(desc));
   desc->root_change.push_launched = true;
}

/* Root was uploaded as it is. */
static inline void
brl_root_mark_clean(struct brl_descriptor_state *desc)
{
   desc->root_change.dirty = false;
   desc->root_change.push_only = false;
   desc->root_change.push_launched = false;
}

struct brl_attachment {
   VkRenderingAttachmentFlagBitsKHR flags;

   VkFormat vk_format;
   struct brl_image_view *iview;

   /* The API load and store operations and the layout the pass was begun
    * with. */
   VkAttachmentLoadOp load_op;
   VkAttachmentStoreOp store_op;
   VkImageLayout layout;

   VkResolveModeFlagBits resolve_mode;
   struct brl_image_view *resolve_iview;

   bool clear;
   uint32_t clear_colour[4];
};

struct brl_bg_eot {
   uint64_t usc;
   uint32_t primary_storage_B;
   uint32_t staging_storage_B;
   union {
      struct agx_counts_packed counts;
      uint64_t rsrc_spec;
   };
   /* Frame anatomy profiler (brl_frame_anatomy.h): per-RT op summary of the
    * program. Written only while HK_FRAME_ANATOMY is set; otherwise zero and
    * never read. Not part of any hardware or kernel structure. */
   uint8_t anatomy_ops[BRL_MAX_RTS];
};

/* BG/EOT programs are self-addressed and may reference command-buffer pool
 * uploads, so reuse them only inside the command buffer which owns both. One
 * exact render-pass key owns its main/partial BG/EOT variants; hash collisions
 * remain harmless misses. */
#define BRL_G17P_BG_EOT_CACHE_ENTRIES 16
#define BRL_G17P_BG_EOT_CACHE_KEY_MAX 768
struct brl_g17p_bg_eot_dev_entry;

struct brl_g17p_bg_eot_cache_entry {
   uint8_t key[BRL_G17P_BG_EOT_CACHE_KEY_MAX];
   struct brl_bg_eot program[4];
   /* frg-codegen v4: the device-lifetime entry this pass fills or reuses,
    * or NULL. */
   struct brl_g17p_bg_eot_dev_entry *dev_entry;
   uint64_t hash;
   uint16_t key_size;
   uint8_t valid_variants;
   uint8_t direct_color_resolve_variants;
   bool valid;
};

struct brl_render_registers {
   uint32_t width, height, layers;
   uint32_t zls_width, zls_height;
   uint32_t isp_bgobjdepth;
   uint32_t isp_bgobjvals;
   struct agx_zls_control_packed zls_control, zls_control_partial;
   uint32_t depth_dimensions;
   bool process_empty_tiles;
   /* The render area is every attachment's full extent: the tile stores
    * write no pixel outside it. */
   bool whole_attachments;
   enum u_tristate dbias_is_int;

   struct {
      uint32_t dimensions;
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } depth;

   struct {
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } stencil;

   struct {
      struct brl_bg_eot main;
      struct brl_bg_eot partial;
   } bg;

   struct {
      struct brl_bg_eot main;
      struct brl_bg_eot partial;
   } eot;

   /* Frame anatomy profiler only: bit i set when colour attachment i's image
    * layout is compressed. Zero unless HK_FRAME_ANATOMY is set. */
   uint8_t color_compressed_mask;
};

struct brl_rendering_state {
   VkRenderingFlagBits flags;

   /* May have written storage memory/XFB anywhere in this API pass, including
    * before a physical render split. Used only for conservative barriers. */
   bool barrier_shader_writes;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;
   /* Attachmentless rendering pass with no multiview. */
   bool no_attachments;

   /* The main G17P EOT writes the single-sample resolve target directly. */
   bool g17p_direct_color_resolve;

   /* Exact admission result for the current physical tilebuffer layout.
    * The layout is immutable after BeginRendering except for the one
    * attachmentless late-sample transition, which refreshes this value. */
   bool g17p_rt0_layout_supported;

   uint32_t color_att_count;
   struct brl_attachment color_att[BRL_MAX_RTS];
   struct brl_attachment depth_att;
   struct brl_attachment stencil_att;

   struct agx_tilebuffer_layout tilebuffer;
   struct brl_render_registers cr;

   /* G17P precise depth store: the ZLS words for the
    * stream that ends the pass, when it is also the first stream (only) or
    * a later one (last). Depth honours storeOp there; stencil keeps the
    * forced store. zls_first_cs is the stream BeginRendering created. */
   bool zls_precise;
   struct agx_zls_control_packed zls_control_only, zls_control_last;
   const void *zls_first_cs;

   /* Track depth and stencil independently: an untouched aspect can be
    * removed from every stream even when the other aspect is cleared or
    * used. NONE/NONE aspects must retain their memory contents; eliding
    * untouched LOAD/STORE aspects also avoids redundant traffic.
    */
   VkImageAspectFlags g17p_zls_elide_mask;
   VkImageAspectFlags g17p_zls_touched;

   /* Of the touched aspects, those a draw or clear may have changed. A pass
    * that only tests an elidable aspect leaves its memory contents as they
    * were, so it keeps the load and drops the store
    * (brl_g17p_zls_elide_store()).
    */
   VkImageAspectFlags g17p_zls_written;

   /* M5: an attachment of the pass can alias another resource (aliased,
    * sparse or linear image), so its stores count as memory writes for
    * dependency tracking (BRL_SYNC_GFX_MEM). */
   bool m5_attachments_reach_memory;

   /* AGX3 dead depth store: the "only" ZLS word with the depth store
    * dropped, and whether the pass processes empty tiles without that store,
    * for a later pass that makes the store dead (brl_take_pending_z_store()).
    * Valid when no_z_store_ok. */
   bool no_z_store_ok;
   bool process_empty_tiles_without_z;
   struct agx_zls_control_packed zls_control_only_no_z_store;

   /* G17P: colour attachments a draw or clear of the pass may write, and
    * whether a fragment shader reads the tilebuffer. See
    * brl_g17p_elide_untouched_color(). */
   uint8_t color_written;
   bool color_read;

   /* G17P: the pass's first draw was seen, and the colour attachments it
    * overwrote at every sample before anything could read them. See
    * brl_g17p_note_first_draw(). */
   bool first_draw_seen;
   uint8_t color_covered;

   /* An in-pass BY_REGION barrier with INPUT_ATTACHMENT_READ_BIT was kept in
    * this physical stream assuming tile-memory reads. Any subsequent draw that
    * falls back to texture-backed input attachment reads must split first. */
   bool kept_local_read_barrier;

   /* Partial-MRT HSR background liveness for the first stream of the pass:
    * bg_live_mask marks bound colour attachments whose initial BG contents
    * must be preserved (LOAD_OP_LOAD or LOAD_OP_CLEAR), and bg_clear_fill_rt
    * encodes 2-bit agx_epilog_clear_fill codes for LOAD_OP_CLEAR attachments
    * whose clear value can be materialized directly in the fragment epilog. */
   uint8_t bg_live_mask;
   uint16_t bg_clear_fill_rt;
};

struct brl_index_buffer_state {
   struct brl_addr_range buffer;
   enum agx_index_size size;
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum brl_dirty {
   BRL_DIRTY_VB = BITFIELD_BIT(0),
   BRL_DIRTY_OCCLUSION = BITFIELD_BIT(1),
   BRL_DIRTY_PROVOKING = BITFIELD_BIT(2),
   BRL_DIRTY_VARYINGS = BITFIELD_BIT(3),
};

/* Where a G17P graphics uniform range's source address comes from. */
enum brl_g17p_range_source {
   /* Fixed for the shader and device (heap pointers, promoted constants). */
   BRL_G17P_RANGE_SOURCE_CONSTANT = 0,
   /* The graphics root plus a fixed offset. */
   BRL_G17P_RANGE_SOURCE_ROOT = 1,
   BRL_G17P_RANGE_SOURCE_DRAW_PARAMS = 2,
   BRL_G17P_RANGE_SOURCE_DRAW_ID = 3,
   /* A bound descriptor set's address plus a fixed offset, or the zero page
    * when the set is not bound (uniform DMA spans). */
   BRL_G17P_RANGE_SOURCE_SET = 4,
   /* A whole uniform DMA buffer span: the buffer whose descriptor the root
    * (dynamic buffers) or a static descriptor set holds at a fixed byte, plus
    * a fixed offset, while that buffer covers the span. */
   BRL_G17P_RANGE_SOURCE_BUFFER = 5,
   /* Hoisted AGX3 vertex preamble output buffer (computed once in PREGFX). */
   BRL_G17P_RANGE_SOURCE_PREAMBLE_OUT = 6,
   /* Push-constant slice inside the graphics root (uniform DMA span inside
    * root.push[]). Replays from root + offset in full-root mode, or from a
    * narrow uploaded push slice when push_only_root is set. */
   BRL_G17P_RANGE_SOURCE_PUSH = 7,
   /* Uploaded while collecting the ranges (combined-sampler pointers,
    * spilled render-target descriptors), or a buffer span the buffer does
    * not wholly cover: not re-derivable. */
   BRL_G17P_RANGE_SOURCE_UPLOADED,
};

struct brl_g17p_gfx_loader;

/* How a stage's uniform ranges derive from the draw state, recorded from a
 * collection by brl_g17p_graphics_uniform_ranges(), and the loader they were
 * published with. A draw with the same shader, linked executable, imageblock,
 * range-selecting state (brl_g17p_launch_flags()) and, for a fragment shader,
 * admission inputs derives its ranges from it. Every BO the loader names was
 * retained when it was recorded. */
#define BRL_G17P_LAUNCH_SLOTS 8
#define BRL_G17P_LAUNCH_ROOT_SET 0xff

struct brl_g17p_launch {
   const struct brl_shader *s;
   const struct brl_linked_shader *linked;
   const struct brl_g17p_gfx_loader *loader;
   const struct brl_shader *hw_vs;
   uint16_t tile_size_px;
   uint8_t sample_size_B;
   uint8_t nr_samples;
   uint8_t topology;
   uint8_t provoking;
   uint8_t flags;
   uint8_t count;
   bool valid;
   /* Vertex launch whose only root-backed uniform ranges lie inside
    * root.push[push_min_B .. push_min_B + push_size_B). */
   bool push_only_root;
   uint16_t push_min_B;
   uint16_t push_size_B;
   uint8_t source[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Descriptor set of a SET or BUFFER range; BRL_G17P_LAUNCH_ROOT_SET for a
    * BUFFER range whose descriptor is in the root. */
   uint8_t set[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t start_halfs[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t size_halfs[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Byte of a BUFFER range's descriptor in its root or set. */
   uint16_t descriptor_B[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
   /* Root/set/buffer offset, or the address of a constant range. */
   uint64_t offset[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];
};

struct brl_g17p_graphics_ioto_cache_entry {
   /* The loader of the last draw, the program published from it and the
    * range addresses written into it (brl_g17p_publish_gfx_loader()). */
   const struct brl_g17p_gfx_loader *loader;
   const struct brl_linked_shader *loader_linked;
   uint64_t loader_program;
   struct agx3_ioto_uniform_range
      loader_ranges[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES];

   /* Inputs of the last admitted program (brl_upload_usc_words_internal()). */
   struct {
      const struct brl_shader *s;
      const struct brl_linked_shader *linked;
      const struct brl_shader *hw_vs;
      bool spills;
      uint8_t topology;
      uint8_t provoking;
   } admit;
};

/* The inputs of a G17P VertexProgramVariant state update other than the
 * pipeline address (brl_agx3_publish_vertex_state()). */
struct brl_g17p_vdm_update_key {
   const struct brl_shader *hw_vs;
   uint32_t selector;
   uint8_t draw_path;
   uint8_t provoking;
   bool rasterization_enabled;
   bool primitive_id;
   bool ordinary;
   bool selector_valid;
   bool hoisted_preamble;
};

struct brl_graphics_state {
   /* A bound graphics program writes external memory. */
   bool shaders_write_memory;

   struct brl_rendering_state render;
   struct brl_descriptor_state descriptors;

   enum brl_dirty dirty;
   /* CmdBindVertexBuffers is not vk_dynamic_graphics_state and therefore does
    * not advance its generation.  Prepared G17P roots snapshot this separate
    * identity so a post-BeginRendering bind cannot publish stale attrib VAs. */
   uint64_t vertex_buffer_generation;
   /* The vertex_buffer_generation that descriptors.root.draw.attrib_* were
    * last derived from (brl_flush_dynamic_state_slow()'s vertex-buffer
    * block).  A prepared bind tail records this, not the live generation:
    * a tail rebuilt outside a full flush (vkCmdBindIndexBuffer) must not
    * vouch for a root whose attrib VAs predate the latest vertex binding. */
   uint64_t root_vertex_buffer_generation;

   uint64_t root;
   /* CPU copy and published byte size of the root uploaded at root
    * (brl_cmd_buffer_upload_root()). */
   const struct brl_root_descriptor_table *root_cpu;
   uint32_t root_size;
   /* The root the fragment program was last published with, its CPU copy,
    * and its published byte size (brl_agx3_fragment_root()). */
   uint64_t fs_root;
   const struct brl_root_descriptor_table *fs_root_cpu;
   uint32_t fs_root_size;

   /* A successful equality check between the immutable root snapshots at
    * fs_root and this address. Cleared whenever fs_root is replaced.
    * fs_root_checked_push_match is true only when push[] was also proven
    * equal for this root pair. */
   uint64_t fs_root_checked_root;
   bool fs_root_checked_push_match;

   /* CPU-cached copies of exactly the fragment-visible published bytes. The
    * pointer keys name the immutable GPU-mapped copies, never their contents. */
   struct brl_root_fragment_snapshot root_fragment_snapshot;
   struct brl_root_fragment_snapshot fs_root_fragment_snapshot;
   const struct brl_root_descriptor_table *root_fragment_snapshot_cpu;
   const struct brl_root_descriptor_table *fs_root_fragment_snapshot_cpu;

   uint64_t draw_params;
   uint64_t draw_id_ptr;
   /* Hoisted AGX3 vertex preamble output VA active for the current multi-draw
    * batch, and whether the current multi-draw requested hoisting. */
   uint64_t g17p_hoisted_preamble_va;
   bool g17p_hoist_preamble_requested;
   uint32_t g17p_draw_params_value[2];
   uint32_t g17p_draw_id_value;
   bool g17p_draw_params_host_valid;
   bool g17p_draw_id_host_valid;

   struct brl_g17p_prepared_bind_tail g17p_bind_tail;
   /* The last complete build of g17p_bind_tail, whose other fields still
    * describe its inputs. A rebuild whose only changed inputs are the root,
    * descriptor or vertex-buffer identity (a descriptor push or vertex-buffer
    * bind) reads exactly the same state for every record except the fragment
    * record's stage program, which is recomputed from fs_fields; see
    * brl_g17p_build_native_bind_tail(). */
   struct {
      struct agx_ppp_hal200_fragment_fields fs_fields;
      /* The state block of the last build or memo rebuild: CPU copy, its
       * compact address and size. */
      const uint8_t *state_cpu;
      uint32_t state_compact;
      uint32_t state_size;
      uint32_t cull_word;
      uint32_t raster_fragment_control;
      uint16_t viewport_length;
      const struct brl_shader *fs;
      const struct brl_shader *hw_vs;
      bool valid;
      /* Cached viewport payload (at state_cpu + AGX_G17P_BIND_GROUP_BYTES),
       * raster_fragment_control, and prepared->scissors/depth_bias are valid
       * for reuse across material switches when viewport/scissor/depth-bias
       * state is unchanged. */
      bool viewport_valid;
   } g17p_bind_tail_memo;

   /* Per vertex-attribute slot: the last API format seen and the pipe format
    * and interchange block size brl_flush_dynamic_state_slow() derives from
    * it, which are pure functions of the format. */
   struct {
      VkFormat vk_format;
      enum pipe_format format;
      unsigned interchange_align;
   } vb_format_memo[AGX_MAX_VBUFS];

   /* The last VertexProgramVariant state update this command buffer encoded,
    * keyed by every encoder input except the pipeline address; see
    * brl_agx3_publish_vertex_state(). */
   struct {
      struct brl_g17p_vdm_update_key key;
      struct agx_vdm_state_update encoded;
      uint64_t pipeline_program;
      bool valid;
   } g17p_vdm_update_memo;

   /* The last M5 HAL300 graphics core agx_ppp_hal300_graphics_core_pack()
    * packed, a pure function of its fields (zero-initialised, so they compare
    * bytewise); see brl_flush_ppp_state(). */
   struct {
      struct agx_ppp_hal300_graphics_fields fields;
      struct agx_ppp_hal300_graphics_core state;
      bool valid;
   } m5_ppp_pack_memo;

   uint32_t shaders_dirty;
   struct brl_api_shader *shaders[MESA_SHADER_MESH + 1];

   /* Stages bound to a shader compiled without custom-border emulation that
    * has an emulating twin to switch to once the device needs it. */
   uint32_t border_free_stages;

   /* G17P: the twin of the bound fragment shader that reads its per-primitive
    * inputs from the bound mesh shader's records (brl_mesh_fs_twin()), or
    * NULL to draw with the bound fragment shader itself. */
   struct brl_api_shader *mesh_fs;

   /* The zs_self_dep twin of the bound fragment shader while a dynamic
    * depth/stencil feedback loop is enabled (brl_fbl_twin()), or NULL. */
   struct brl_api_shader *fbl_fs;

   /* Vertex buffers */
   struct brl_addr_range vb[AGX_MAX_VBUFS];
   /* Bitmask of vb[] slots with a non-zero range currently bound. */
   uint32_t bound_vb_mask;

   /* Transform feedback buffers */
   struct brl_addr_range xfb[4];

   /* Is transform feedback enabled? */
   bool xfb_enabled;

   /* Internal transform feedback offset vec4.
    *
    * TODO: Strictly could be global.
    */
   uint64_t xfb_offsets;

   /* Pointer to the GPU memory backing active transform feedback queries,
    * per-stream. Zero if no query is bound.
    */
   uint64_t xfb_query[4];
   /* Active VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT reports, per stream. */
   uint64_t prims_generated_query[4];

   struct brl_index_buffer_state index;
   enum agx_primitive topology;
   enum agx_object_type object_type;

   /* Provoking vertex 0, 1, or 2. Usually 0 or 2 for FIRST/LAST. 1 can only be
    * set for tri fans.
    */
   uint8_t provoking;

   struct {
      enum agx_visibility_mode mode;

      /* If enabled, index of the current occlusion query in the occlusion heap.
       * There can only be one active at a time (hardware contraint).
       */
      uint16_t index;

      /* On M5, an empty query needs only its logical availability word.
       * Assign its physical counter when the first draw needs it. */
      struct brl_query_pool *pending_pool;
      uint32_t pending_query;

      /* Recording order (brl_sync_stream::seq) of the render the active query
       * began in, 0 if none: only a query that also ends there may become
       * available with that render's retirement. */
      uint32_t begin_render_seq;
   } occlusion;

   /* Fast linked shader data structures */
   uint64_t varyings;
   struct agx_varyings_vs linked_varyings;
   /* brl_g17p_smooth_varyings_supported() of the last (linked FS, hardware
    * VS, topology, provoking) it evaluated against the linked_varyings
    * above. Cleared whenever linked_varyings is reassigned. */
   struct {
      const void *linked;
      const void *vs;
      uint32_t topology;
      uint32_t provoking;
      bool valid;
      bool result;
   } g17p_smooth_memo;

   uint32_t linked_dirty;
   struct brl_linked_shader *linked[MESA_SHADER_STAGES];

   /* Last complete fast-link selection for this command buffer. The shader
    * identity is part of the cache key: identical bytes owned by a different
    * shader may have a different lifetime. Graphics-state invalidation clears
    * these records with the rest of gfx state. */
   struct {
      struct brl_shader *shader;
      struct brl_fast_link_key_vs key;
      bool valid;
   } fast_link_vs;
   struct {
      struct brl_shader *shader;
      struct brl_fast_link_key_fs key;
      uint64_t desc_gen;
      bool valid;
      /* Colour attachments the linked fragment program writes (including any
       * epilog clear-fill targets), those written by the application shader
       * itself, the stream clear-fill code snapshot used to build `key`, and
       * whether the linked program reads the tilebuffer. */
      uint8_t color_written;
      uint8_t app_color_written;
      uint16_t bg_clear_fill_rt;
      bool reads_tib;
   } fast_link_fs;

   /* Effective AGX_PASS_TYPE_* selected for the active draw (brl_partial_mrt_hsr). */
   uint8_t effective_fragment_pass_type;

   /* Keep these five native graphics caches contiguous: AGX2 resets leave
    * their zero-initialized storage untouched.
    *
    * Exact last VS/FS loader identities. These are recording-only selectors;
    * the published programs and every referenced BO remain command-buffer-
    * owned. Submission never mutates or consults this cache. */
   struct brl_g17p_graphics_ioto_cache_entry g17p_ioto[2];
   struct brl_g17p_launch g17p_launch[2][BRL_G17P_LAUNCH_SLOTS];
   /* Exact command-buffer-owned uploads for combined texture/sampler pointer
    * pairs.  Shader/material switches may revisit the same pair thousands of
    * times; a fresh upload address would otherwise force a fresh Ioto program
    * even though the GPU-visible values are identical. */
   struct {
      /* Key: descriptor-set texture word address and sampler state address. */
      uint64_t pointers[2];
      /* The uploaded pair; its texture pointer is 32-byte aligned
       * (brl_g17p_texture_state_pointer). */
      uint64_t published[2];
      uint64_t address;
      bool valid;
   } g17p_combined_sampler_pointers[32];

   /* Dynamic graphics state snapshot for g17p_bind_tail (kept out of
    * struct brl_g17p_prepared_bind_tail so hot graphics fields stay dense).
    * Read only when g17p_bind_tail.valid or g17p_bind_tail_memo.valid /
    * viewport_valid is true. */
   struct brl_g17p_bind_tail_dyn_state g17p_bind_tail_dyn;

   /* Exact immutable eMRT descriptor-table upload used by every fragment
    * program in one spilled render configuration. */
   struct {
      uint8_t entries[BRL_MAX_RTS][AGX_G17P_EMRT_DESCRIPTOR_SIZE];
      uint64_t address;
      uint64_t pointer;
      uint8_t color_att_count;
      bool valid;
   } g17p_spilled_rt_descs;

   bool generate_primitive_id;

   /* Tessellation state */
   struct {
      uint64_t out_draws;
      uint64_t grids;
      /* Direct thread count bounding the TES-before-GS kernel of this draw,
       * or 0 to use the indirect grid (brl_tes_direct_query_threads()). */
      uint32_t tes_direct_threads;
      struct brl_tess_info info;
      enum mesa_prim prim;
   } tess;

   /* Native mesh draw state (VK_EXT_mesh_shader, native TA path). */
   struct {
      struct agx_ptr params;
      uint64_t grid, object_program, object_params;
      /* Direct-draw API task grid, used for the object dispatch when an
       * task shader is bound. */
      uint32_t task_grid[3];
   } mesh;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

/* The CPU source is mutable. Snapshot its visible bytes at publication,
 * before copying into the immutable GPU-mapped root. */
static inline void
brl_root_fragment_snapshot_publish(struct brl_graphics_state *gfx,
                                  const struct brl_root_descriptor_table *cpu,
                                  const struct brl_root_descriptor_table *source,
                                  uint32_t root_size)
{
   brl_root_fragment_snapshot_store(&gfx->root_fragment_snapshot, source,
                                   root_size);
   gfx->root_fragment_snapshot_cpu = cpu;
}

/* Select the same immutable root as the fresh byte comparison. Hidden reads
 * always use the current root; snapshots and the positive identity memo only
 * accelerate the equality check for shaders that cannot observe those bytes. */
static inline uint64_t
brl_agx3_fragment_root_select(struct brl_graphics_state *gfx, uint8_t root_reads,
                             uint32_t required_size, bool use_memo,
                             bool use_shadow)
{
   const uint8_t hidden = BRL_ROOT_READS_ADDRESS | BRL_ROOT_READS_VERTEX_INPUT |
                          BRL_ROOT_READS_DYNAMIC_BUFFERS;
   const bool reads_push = (root_reads & BRL_ROOT_READS_PUSH) != 0;
   const uint32_t full = sizeof(struct brl_root_descriptor_table);
   const uint32_t req = required_size ? MIN2(required_size, full) : full;
   const uint32_t cur_size = gfx->root_size ? MIN2(gfx->root_size, full) : full;
   const uint32_t prev_size =
      gfx->fs_root_size ? MIN2(gfx->fs_root_size, full) : full;

   const bool shadow = use_shadow &&
                       gfx->root_fragment_snapshot_cpu == gfx->root_cpu &&
                       gfx->fs_root_fragment_snapshot_cpu == gfx->fs_root_cpu;

   if (gfx->fs_root && gfx->fs_root != gfx->root && gfx->root_cpu &&
       prev_size >= req && !(root_reads & hidden)) {
      const bool memo_bytes = use_memo && gfx->fs_root_checked_root &&
                              gfx->fs_root_checked_root == gfx->root;
      if (memo_bytes && (!reads_push || gfx->fs_root_checked_push_match))
         return gfx->fs_root;

      const bool bytes_match =
         memo_bytes ||
         (shadow
             ? !memcmp(gfx->fs_root_fragment_snapshot.bytes,
                       gfx->root_fragment_snapshot.bytes,
                       sizeof(gfx->root_fragment_snapshot.bytes))
             : brl_root_matches_for_fragment(gfx->fs_root_cpu, prev_size,
                                            gfx->root_cpu, cur_size));
      if (bytes_match) {
         const bool push_match =
            !reads_push ||
            (shadow
                ? !memcmp(gfx->fs_root_fragment_snapshot.push,
                          gfx->root_fragment_snapshot.push,
                          sizeof(gfx->root_fragment_snapshot.push))
                : brl_root_matches_push(gfx->fs_root_cpu, prev_size,
                                       gfx->root_cpu, cur_size));
         if (push_match) {
            if (use_memo) {
               if (gfx->fs_root_checked_root != gfx->root)
                  gfx->fs_root_checked_push_match = false;
               gfx->fs_root_checked_root = gfx->root;
               if (reads_push)
                  gfx->fs_root_checked_push_match = true;
            }
            return gfx->fs_root;
         }
      }
   }

   if (use_shadow && gfx->root_cpu &&
       gfx->root_fragment_snapshot_cpu == gfx->root_cpu) {
      gfx->fs_root_fragment_snapshot = gfx->root_fragment_snapshot;
      gfx->fs_root_fragment_snapshot_cpu = gfx->root_cpu;
   } else {
      gfx->fs_root_fragment_snapshot_cpu = NULL;
   }

   gfx->fs_root_checked_root = 0;
   gfx->fs_root_checked_push_match = false;
   gfx->fs_root = gfx->root;
   gfx->fs_root_cpu = gfx->root_cpu;
   gfx->fs_root_size = gfx->root_size;
   return gfx->root;
}

struct brl_compute_state {
   struct brl_descriptor_state descriptors;
   struct brl_api_shader *shader;
   /* vkCmdPipelineBarrier2 dependency not yet consumed by a CDM launch.
    * Lives on the command buffer so it survives CS splits and
    * merge_control_streams. */
   bool cdm_barrier_pending;
};

/*
 * VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR state.  Descriptor sets and push
 * constants bound to the ray-tracing bind point are separate from the compute
 * ones, even though the pipeline executes as a compute shader: an application
 * may bind a compute pipeline's sets between vkCmdBindPipeline and
 * vkCmdTraceRaysKHR and neither may disturb the other.
 */
struct brl_ray_tracing_state {
   struct brl_descriptor_state descriptors;
   struct brl_ray_tracing_pipeline *pipeline;
};

#define BRL_RT_SHADER_STAGES                                                    \
   (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |         \
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |       \
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR)

struct brl_cmd_push {
   void *map;
   uint64_t addr;
   uint32_t range;
   bool no_prefetch;
};

struct brl_scratch_req {
   bool main;
   bool preamble;
};

/* struct agx_timestamp_req lives in brl_sync.h. */

/*
 * brl_cs represents a single control stream, to be enqueued either to the
 * CDM or VDM for compute/3D respectively.
 */
enum brl_cs_type {
   BRL_CS_CDM,
   BRL_CS_VDM,
};

/* Fragment programs a control stream records for the frame anatomy
 * profiler (brl_frame_anatomy.h). */
#define BRL_FA_FS_MAX 4

struct brl_compute_pair_dispatch;


struct brl_cs {
   struct list_head node;

   struct brl_compute_pair_dispatch *compute_pair;

   /* Parent command buffer. Convenience. */
   struct brl_cmd_buffer *cmd;

   /* Data master */
   enum brl_cs_type type;

   /* Address of the root control stream for the job */
   uint64_t addr;

   /* Fat pointer to the start of the current chunk of the control stream */
   struct agx_ptr chunk;

   /* Start pointer of the root control stream */
   void *start;

   /* Size of the root control stream allocation */
   uint32_t root_size;

   /* Current pointer within the control stream */
   void *current;

   /* End pointer of the current chunk of the control stream */
   void *end;

   /* AGX3 CDM: the value of current just after a strong cache flush, so a
    * flush requested with nothing recorded since it can be elided. */
   void *strong_flush_end;

   /* Likewise for the weak ordering barrier attached to a native launch. A
    * later cache-flush request upgrades this record in place instead of
    * emitting two adjacent barriers. */
   void *weak_barrier_end;

   /* AGX3 CDM: the value of current just after a strong flush that also has
    * memory scope (AGX3_CDM_BARRIER_FLUSH_MEMORY): on G17P and M5 the
    * flushes brl_cdm_visibility_flush() asks for, and on M5 also the driver's
    * own (brl_cdm_cache_flush()). A patch site written later
    * (struct brl_alias_site) does not move it. */
   void *memory_flush_end;

   /* The vertex stages read the hoisted preamble after this stream. */
   bool pre_gfx_writes_preamble;

   /* M5: patch sites recorded in this stream (brl_cmd_buffer::alias_sites),
    * which a merge moves with its bytes. */
   uint32_t alias_sites;

   /* AGX2: sparse patch sites recorded in this stream
    * (brl_cmd_buffer::sparse_sites), which a merge moves with its bytes. */
   uint32_t sparse_sites;

   /* AGX3 CDM: the value of current just after a standalone barrier that is
    * the stream's *first* record. Apple's emitter only opens a stream with a
    * barrier (the strong flush that makes a preceding render visible), so a
    * stream is canonical on its own; but concatenating it after a stream that
    * already ends in a barrier leaves two adjacent barriers with no launch
    * between them, which Apple never emits. brl_cs_merge_cdm() uses this to
    * fold the leading record into the destination's trailing one. */
   void *leading_barrier_end;

   /* AGX3 CDM: weak_barrier_end when that barrier ends an application
    * dispatch. Where agx_chip_elide_app_dispatch_barrier() holds, a following
    * application dispatch overwrites it, so consecutive application dispatches
    * carry no barrier between them; any other record still follows it. */
   void *app_dispatch_end;

   /* Whether there is more than just the root chunk */
   bool stream_linked;

   /* Active-context selector consumed by AGX3 VertexProgramVariant. */
   bool agx3_active_context_selector_valid;
   uint32_t agx3_raw_active_context_selector;
   /* G18G: a terminated M5 dispatch owns this stream; no record may follow
    * it, and the submission carries the stream's compute resources. */
   bool g18_dispatch_sealed;
   struct drm_asahi_g18_compute_resources g18_compute_resources;

   /* Whether the sampler heap is required. Although we always must maintain the
    * heap for correctness, it's often not necessary since we can push lots of
    * samplers (especially for GL/DX11-era engines).
    */
   bool uses_sampler_heap;
   bool gen1_native_rq;
   bool gen1_native_fragment_rq;
   bool gen1_native_vertex_rq;
   bool m5_matrix_ops;

   /* Threads of the application's direct dispatches in this stream, and
    * whether it holds an indirect one (brl_queue::sib only takes streams with
    * enough work to overlap). */
   uint64_t sib_threads;
   bool sib_indirect;

   /* Submit packing (brl_pack_runs()): how many of the stream's launches are
    * direct application dispatches, their threads, and the sum over them of
    * threads times static native instructions. These bound the work one
    * packed command may hold. The sums saturate at UINT64_MAX. */
   uint32_t pack_dispatches;
   uint64_t pack_threads;
   uint64_t pack_static_work;
   /* A dispatch in this stream runs a program with global atomics. */
   bool uses_global_atomics;
   /* A dispatch in this stream runs a program with a loop: pack_static_work
    * does not bound its run time. */
   bool pack_loops;
   /* The stream calls another stream through a link with return; its bytes
    * must stay where they were recorded. */
   bool calls_stream;

   /* At least one AS-build dispatch or compute transfer is in this stream. */
   bool as_build;
   /* M5: a pre-rasterization program recorded in this CS keeps state in
    * thread-private storage (a spill frame or constant-program inputs), so
    * its tiling must not overlap earlier renders' fragment work
    * (brl_sync_resolve_submit()). */
   bool m5_vertex_tps;

   /* M5: a later render pass instance continues in this stream
    * (brl_m5_pass_merge_begin()); its draws may not share the stream's
    * committed sample locations (brl_flush_gfx_state()). */
   bool m5_continued_pass;

   /* A graphics program recorded in this stream writes external memory. */
   bool m5_shader_writes_memory;

   /* Per-render-stream memo of hoisted AGX3 vertex preamble outputs already
    * computed by a PREGFX dispatch for a given (vertex shader, root VA). */
   struct {
      const struct brl_shader *vs;
      uint64_t root;
      uint64_t out_va;
   } preamble_hoist[4];
   uint8_t preamble_hoist_next;

   /* M5: the continued instance was ordered after this stream's earlier work
    * by a dependency that is not framebuffer-local (brl_m5_pass_merge_begin()).
    * While this is the open render stream, every draw recorded into it must
    * leave the merge memory-equivalent to the two separate instances, or the
    * stream ends first (brl_m5_ordered_draw_hazard()). */
   bool m5_ordered;

   /* M5: every attachment of the continued instances is in an
    * attachment-only layout, so no descriptor can access one of them and
    * descriptor sets bound for graphics need not end an ordered continuation
    * (brl_m5_attachments_exclude_descriptors()). */
   bool m5_layout_proof;

   /* Scratch requirements */
   struct {
      union {
         struct brl_scratch_req vs;
         struct brl_scratch_req cs;
      };

      struct brl_scratch_req fs;
   } scratch;

   /* Immediate writes, type libagx_imm_write. These happen at the end of the
    * control stream (parallel on AGX2, native ordered stores on G17P). This
    * accelerates queries. Implies CDM.
    */
   struct util_dynarray imm_writes;
   /* Bound query-pool byte ranges (struct brl_query_access_range) written by
    * imm_writes above, kept separate from oq_stamps so fallback availability
    * stores and deferred occlusion joins preserve GPU query access tracking. */
   struct util_dynarray imm_write_query_ranges;
   bool imm_write_query_untracked;

   /* Native private-memory profiles for every dispatch in this stream. */
   struct util_dynarray compute_resources;
   /* Fields of the M5 HAL300 graphics core this stream last emitted, and the
    * records they packed to. The PPP latches each record for the following
    * draws of the stream, so an identical core is not emitted again and a
    * changed one emits only its changed records; any other PPP state emitted
    * into the stream clears m5_ppp_emitted_valid. See brl_flush_ppp_state(). */
   struct agx_ppp_hal300_graphics_fields m5_ppp_emitted;
   struct agx_ppp_hal300_graphics_core m5_ppp_emitted_state;
   bool m5_ppp_emitted_valid;

   struct util_dynarray render_resources;
   /* The render_resources entry the last draw matched or appended, see
    * brl_m5_note_render_resources(). */
   uint8_t m5_render_resources_last[3 * sizeof(struct drm_asahi_render_program_v2)];
   size_t m5_render_resources_last_size;
   struct drm_asahi_render_program tile_resources;
   /* Native AGX3: struct brl_oq_stamp for each immediate write that makes an
    * occlusion query available and could instead ride on the retirement
    * timestamp of the render that ended the query. While oq_avail_deferred,
    * the writes are not emitted yet: the end of the command buffer decides
    * (brl_resolve_deferred_oq_available()). Implies CDM. */
   struct util_dynarray oq_stamps;
   bool oq_avail_deferred;

   /* Native AGX3: occlusion availability writes staged inline on a render
    * stream so single-render command buffers stamp availability directly at
    * EndCommandBuffer without allocating and discarding a post_gfx CDM stream
    * and its GPU control-stream root. Materialized into post_gfx on demand by
    * brl_flush_oq_pending() before any later stream or sync record. */
   struct brl_oq_pending {
      uint64_t avail_addr;
      uint32_t handle;
      uint32_t offset_B;
      uint8_t avail_size;
   } oq_pending[3];
   uint8_t oq_pending_count;

   /* Number of successfully emitted draws whose last raster producer is
    * positively known to target only hardware Layer=0, without multiview.
    * Compare with stats.cmds: untracked emitters conservatively disqualify.
    * The wider proof count does not wrap with stats.cmds.
    */
   uint64_t layer_zero_draws;
   bool layer_zero_unknown;

   /* Number of successfully emitted draws whose active hardware viewport and
    * scissor bounds were recorded into attachmentless_max_{x,y}. Compare with
    * stats.cmds: untracked emitters conservatively fall back to cs->cr size.
    */
   uint64_t attachmentless_bounded_draws;
   uint32_t attachmentless_max_x;
   uint32_t attachmentless_max_y;

   /* An upper bound of the primitives every draw recorded into this G17P
    * render stream submits (brl_draw_primitive_bound()), saturating at
    * UINT64_MAX, which also stands for "not bounded when recorded". Secondary
    * command buffers replay into the primary's stream and add to it there. */
   uint64_t g17p_primitive_bound;

   /* This stream begins (vkCmdBeginRendering without RESUMING) and ends
    * (vkCmdEndRendering without SUSPENDING) its render pass: it is the whole
    * pass, not a part split off or continued. */
   bool g17p_pass_begins, g17p_pass_ends;

   /* Partial-MRT HSR per-stream tracking:
    * - bg_live_mask: bound colour attachments whose stream BG shader (bg.main
    *   or bg.partial) holds live contents that ISP HSR must not skip.
    * - bg_clear_fill_rt: 2-bit agx_epilog_clear_fill codes for LOAD_OP_CLEAR
    *   attachments on the first stream of a non-resuming pass (0 on later
    *   streams where bg.main reloads partial results).
    * - user_color_written_mask: colour attachments written by application
    *   draws or mid-pass vkCmdClearAttachments in this stream.
    * - opaque_color_written: colour attachments written by pending
    *   opaque/punch-through draws in the current HSR batch. */
   uint8_t bg_live_mask;
   uint16_t bg_clear_fill_rt;
   uint8_t user_color_written_mask;
   uint8_t opaque_color_written;

   /* A G17P mesh launch is recorded into this render stream. The A18 Pro
    * loses the vertex output of the mesh workgroup in flight whenever the
    * tiler heap runs out mid-draw, so the submission asks the kernel to size
    * the heap for mesh output first (DRM_ASAHI_RENDER_MESH_OUTPUT). */
   bool g17p_mesh_output;

   /* Consecutive GPU query resets share one strong flush, at the first
    * consumer or the end of the stream. G13 never sets this. */
   bool query_reset_pending;

   /* A query copy or reset was recorded into this stream. */
   bool query_access;

   /* Query words in kernel-bound timestamp buffers this stream reads or
    * writes (struct brl_query_access_range): copies, resets, availability
    * stores, acceleration-structure properties and timestamp copies. These
    * decide DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS (brl_query_stamps_conflict());
    * query_access above does not. untracked: a word could not be recorded,
    * so the flag is omitted. */
   bool query_access_untracked;
   struct util_dynarray query_access_ranges;

   /* The stream ends with a strong, memory-scoped flush
    * (DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION).
    */
   bool visible_completion;

   /* Statistics */
   struct {
      uint32_t calls, cmds, flushes, merged;
      uint64_t query_copy_dispatches, query_copy_queries, query_copy_counters;
      uint64_t query_reset_dispatches, query_reset_queries, query_reset_flushes;
      uint32_t robust_index_draws, robust_index_dispatches,
         robust_index_batches;
   } stats;

   /* Frame anatomy profiler only: the first fragment programs this stream
    * draws with and their draw counts; later programs are only counted.
    * Zero unless HK_FRAME_ANATOMY is set. */
   struct {
      const struct brl_shader *fs;
      uint32_t draws;
      /* Compute streams: total threads and workgroups of the direct
       * dispatches, and how many were indirect. */
      uint64_t threads, groups;
      uint32_t indirect;
   } anatomy_fs[BRL_FA_FS_MAX];
   uint32_t anatomy_fs_other;

   /* Likewise for the hardware vertex programs. */
   struct {
      const struct brl_shader *vs;
      uint32_t draws;
   } anatomy_vs[BRL_FA_FS_MAX];
   uint32_t anatomy_vs_other;

   /* Frame anatomy profiler only: driver-internal (precompiled) kernels this
    * stream dispatches, with counts and total threads (UINT64_MAX once any
    * dispatch is indirect). */
   struct {
      uint16_t program;
      uint16_t dispatches;
      uint64_t threads;
   } anatomy_precomp[BRL_FA_FS_MAX];
   uint32_t anatomy_precomp_other;

   /* Frame anatomy profiler only: the API pipeline barriers recorded between
    * the previous stream's creation and this one's (count, and the union of
    * their source and destination stage masks). */
   uint32_t anatomy_api_barriers;
   uint16_t anatomy_api_mem, anatomy_api_buf, anatomy_api_img;
   /* Frame anatomy profiler only: draws that republished render state,
    * vertex-program state updates and how many named a program other than
    * the stream's previous one, and PPP state records emitted. */
   uint32_t anatomy_republish, anatomy_vdm, anatomy_vdm_new, anatomy_ppp;
   /* Colour attachments whose load the first draw made unnecessary. */
   uint8_t anatomy_covered;
   /* Frame anatomy profiler only, M5: draws by the latched fragment pass type
    * (opaque, translucent, punch-through, translucent punch-through), and
    * draws with triangle merging disabled, with depth writes, and with a
    * depth mode other than UNCHANGED (shader depth or side effects). */
   uint32_t anatomy_pass[4], anatomy_nomerge, anatomy_zwrite, anatomy_late_z;
   /* Frame anatomy profiler only: the render area of a pass that does not
    * cover its whole framebuffer (zero extent otherwise). */
   VkRect2D anatomy_area;
   uint64_t anatomy_last_program;
   uint64_t anatomy_api_src, anatomy_api_dst;
   uint64_t anatomy_api_src_access, anatomy_api_dst_access;

   /* Timestamp writes: the start of the command (compute start, render
    * vertex start) and its end (compute end, render fragment end). The
    * kernel writes both from the firmware's native pass record.
    */
   struct {
      struct agx_timestamp_req start;
      struct agx_timestamp_req end;
   } timestamp;

   /* G17P and M5: timestamp.start came from a leading TOP_OF_PIPE write on a
    * chip whose start stamps latch before barriers resolve. Submission either
    * keeps it on this command or moves it to a preceding end-stamp carrier. */
   bool unordered_start_ts;

   /* Render only: occlusion availability slots written by the kernel at
    * retirement, indexed vertex start, vertex end, fragment start, fragment
    * end. Slots 0 and 3 are only used when timestamp.start/end are not. */
   struct agx_timestamp_req oq_stamp[4];

   /* G17P query commands observe timestamp writes from earlier streams.
    * Firmware start timestamps reach query memory only at retirement, so a
    * producer cannot be merged across this consumer boundary.
    */
   bool g17p_timestamp_query;

   /* Recorded dependencies, see brl_sync.h. */
   struct brl_sync_stream sync;

   /* Writable destinations known while recording meta compute dispatches. */
   struct brl_drm_attachment_set compute_attachments;

   /* Remaining state is for graphics only, ignored for compute */
   struct agx_tilebuffer_layout tib;

   /* Bottom-right corner of the render area, or 0 where it is not known (a
    * secondary command buffer continuing a render pass). Tiles past it hold
    * nothing the render may change, so the render need not visit them. */
   uint32_t area_x1, area_y1;

   struct util_dynarray scissor, depth_bias;
   uint64_t uploaded_scissor, uploaded_zbias;

   /* G17P: the last uploaded draw-state selector record and the scissor /
    * depth-bias array sizes and viewport count it selects for. */
   struct {
      uint32_t offset;
      uint32_t scissor_size, depth_bias_size;
      uint8_t viewport_count;
   } g17p_draw_state;

   /* G17P / HAL200: the last complete bind tail published into this VDM
    * stream.  Every record in that tail latches independently in PPP state,
    * so a subsequent draw whose only changed tail inputs are the fragment
    * program and/or cull/front-face mode can elide the tail (0 bytes) or emit
    * standalone 8-byte PPP_STATE_UPDATE records for the 28-byte FS and/or
    * 8-byte CULL records. */
   struct {
      uint64_t fs_program;
      uint32_t state_compact;
      uint32_t cull_word;
      uint32_t scissor_size, depth_bias_size;
      uint8_t viewport_count;
      bool valid;
   } g17p_bind_tail;

   /* We can only set ppp_multisamplectl once per batch. has_sample_locations
    * tracks if we've committed to a set of sample locations yet. vk_meta
    * operations do not set has_sample_locations since they don't care and it
    * would interfere with the app-provided samples.
    *
    */
   bool has_sample_locations;
   uint64_t ppp_multisamplectl;

   struct brl_render_registers cr;

   /* Snapshot of the writable image ranges for this exact VDM command. Render
    * state may change before queue submission, so the synthetic fragment
    * attachment command must travel with the control stream it describes. */
   struct brl_drm_attachment_set fragment_attachments;

   /* The restart index last emitted to this stream, valid once
    * restart_index_set. VK_EXT_primitive_restart_index makes every value,
    * zero included, a legal custom index, so no value can mean "unset".
    * This is used in place of dirty tracking, because dirty tracking
    * restart indices is complicated and just checking the saved value is cheap.
    */
   uint32_t restart_index;
   bool restart_index_set;
};

static inline unsigned
brl_sync_sq(const struct brl_cs *cs)
{
   return cs->type == BRL_CS_VDM ? BRL_SYNC_VDM : BRL_SYNC_CDM;
}

/* The later of two streams in list order; NULL is earlier than anything. */
static inline struct brl_cs *
brl_sync_later(struct brl_cs *a, struct brl_cs *b)
{
   if (!a)
      return b;
   if (!b)
      return a;
   if (b->sync.seq != a->sync.seq)
      return b->sync.seq > a->sync.seq ? b : a;
   return b->sync.pre_seq > a->sync.pre_seq ? b : a;
}

/* Whether the vertex program of a draw is published as an AGX3 VDM State
 * Update (brl_agx3_publish_vertex_state()): the record
 * agx_vdm_state_update_supported() admits. */
static inline bool
brl_agx3_vertex_state(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_STATE);
}

/* Whether brl_agx3_publish_vertex_state() appends the native bind tail
 * (brl_g17p_emit_native_bind_tail()) after the VDM State Update: the native
 * AGX3 path. */
static inline bool
brl_vertex_state_has_bind_tail(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip);
}

/* The bind-tail result of a draw that published its vertex state with
 * brl_agx3_publish_vertex_state(). Every flush that publishes vertex state
 * reports this, so none drops an emitted tail. */
static inline bool
brl_vertex_state_bind_tail(enum agx_chip chip, bool tail_emitted)
{
   return brl_vertex_state_has_bind_tail(chip) && tail_emitted;
}

/* Whether a draw that changes only root descriptors may take the root-only
 * flush of brl_flush_dynamic_state(): the stream has latched a HAL300 graphics
 * core (brl_flush_ppp_state()), so a new fragment program needs only its
 * program record, and the vertex program is published as an AGX3 VDM State
 * Update with no bind tail. A chip whose publication appends a tail takes
 * brl_flush_dynamic_state_slow(), which also republishes the PPP state the
 * tail belongs to. */
static inline bool
brl_root_only_flush_supported(enum agx_chip chip, const struct brl_cs *cs)
{
   return cs->m5_ppp_emitted_valid && brl_agx3_vertex_state(chip) &&
          !brl_vertex_state_has_bind_tail(chip);
}

void *brl_cmd_buffer_emit_cs_jump(struct brl_cmd_buffer *cmd, void *current,
                                 uint64_t target, bool vdm);

/* Moves the patch sites of cmd in [from, from_end), copied to `to` without
 * their first `skip` bytes, which were folded into the record just before
 * `to` (brl_cs_merge_cdm_until()). */
void brl_cmd_buffer_move_alias_sites(struct brl_cmd_buffer *cmd,
                                    const void *from, const void *from_end,
                                    uint8_t *to, size_t skip);

/* As brl_cmd_buffer_move_alias_sites(), for the AGX2 sparse patch sites. */
void brl_cmd_buffer_move_sparse_sites(struct brl_cmd_buffer *cmd,
                                     const void *from, const void *from_end,
                                     uint8_t *to, size_t skip);

/*
 * Helper to merge two compute control streams, concatenating the second control
 * stream, up to b_current, to the first one. Must sync with brl_cs.
 *
 * Submit packing (brl_pack_runs()) merges streams of other command buffers
 * into a submission-owned stream, stopping before their terminate; it only
 * merges inline, so the link path, which allocates from a's command buffer,
 * is never taken for it.
 */
static inline void
brl_cs_invalidate_compute_pair(struct brl_cs *cs)
{
   if (!cs || !cs->compute_pair)
      return;
   brl_compute_pair_producer_unref(cs->compute_pair->producer);
   free(cs->compute_pair);
   cs->compute_pair = NULL;
}

static inline void
brl_cs_merge_cdm_until(struct brl_cs *a, const struct brl_cs *b, void *b_current)
{
   assert(a->type == BRL_CS_CDM && b->type == BRL_CS_CDM);
   brl_cs_invalidate_compute_pair(a);
   assert(!a->timestamp.end.handle);
   a->m5_matrix_ops |= b->m5_matrix_ops;

   const size_t b_size = (uint8_t *)b_current - (uint8_t *)b->start;
   const size_t remaining = (uint8_t *)a->end - (uint8_t *)a->current;
   const bool inline_b = !a->stream_linked && !b->stream_linked &&
                         brl_cs_space_fits(remaining, b_size,
                                          AGX_CDM_STREAM_LINK_LENGTH);

   if (inline_b) {
      uint8_t *dst = a->current;

      /* Two barriers with no launch between them is not a shape Apple's CDM
       * emitter produces, and the G17P stream oracle rejects it. Within one
       * stream brl_cdm_cache_flush() already strengthens the trailing record
       * in place; do the same across a concatenation. b's leading barrier is
       * the strong flush, so copying its word over a's trailing barrier is
       * exactly that upgrade -- no encoder needed, and it removes a record
       * rather than adding one. */
      const bool b_opens_with_barrier =
         b->leading_barrier_end &&
         (uint8_t *)b->leading_barrier_end ==
            (uint8_t *)b->start + AGX3_CDM_BARRIER_SIZE;
      const bool a_ends_with_barrier =
         ((uint8_t *)a->current == (uint8_t *)a->strong_flush_end ||
          (uint8_t *)a->current == (uint8_t *)a->weak_barrier_end) &&
         (uint8_t *)a->current - (uint8_t *)a->start >= AGX3_CDM_BARRIER_SIZE;
      const size_t b_skip = (b_opens_with_barrier && a_ends_with_barrier)
                               ? AGX3_CDM_BARRIER_SIZE
                               : 0;
      const bool a_memory = (uint8_t *)a->current == (uint8_t *)a->memory_flush_end;
      if (b_skip) {
         /* b's strong flush must not drop the memory scope a's trailing
          * record carried for the work before it. */
         uint32_t word = agx3_cmdstream_load_le32(b->start);
         if (a_memory)
            word |= AGX3_CDM_BARRIER_FLUSH_MEMORY;
         agx3_cmdstream_store_le32(dst - AGX3_CDM_BARRIER_SIZE, word);
      }

      memcpy(dst, (const uint8_t *)b->start + b_skip, b_size - b_skip);
      a->current = dst + (b_size - b_skip);
      /* b's patch sites are in the copy now. Submit packing copies other
       * command buffers' streams, whose sites stay with their recording. */
      if (b->alias_sites && a->cmd == b->cmd)
         brl_cmd_buffer_move_alias_sites(a->cmd, b->start, b_current, dst,
                                        b_skip);
      if (b->sparse_sites && a->cmd == b->cmd)
         brl_cmd_buffer_move_sparse_sites(a->cmd, b->start, b_current, dst,
                                         b_skip);

      /* These markers point just past an emitted barrier. Preserve them in
       * the destination root so another adjacent merge still has an exact
       * view of the stream tail. A marker on the folded record maps onto the
       * trailing record that absorbed it. */
      a->strong_flush_end =
         b->strong_flush_end
            ? dst + ((uint8_t *)b->strong_flush_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
      a->weak_barrier_end =
         b->weak_barrier_end
            ? dst + ((uint8_t *)b->weak_barrier_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
      a->memory_flush_end =
         b->memory_flush_end
            ? dst + ((uint8_t *)b->memory_flush_end - (uint8_t *)b->start) -
                 b_skip
         : (a_memory && b_skip && a->current == dst) ? dst
                                                     : NULL;
      a->app_dispatch_end = NULL;
   } else {
      void *link_end =
         brl_cmd_buffer_emit_cs_jump(a->cmd, a->current, b->addr, false);
      const bool linked = link_end != a->current;
      a->current = b_current;
      a->end = b->end;
      a->chunk = b->chunk;
      a->stream_linked |= linked || b->stream_linked;
      a->strong_flush_end = b->strong_flush_end;
      a->weak_barrier_end = b->weak_barrier_end;
      a->memory_flush_end = b->memory_flush_end;
      a->app_dispatch_end = NULL;
   }

   a->alias_sites += b->alias_sites;
   a->sparse_sites += b->sparse_sites;
   a->uses_sampler_heap |= b->uses_sampler_heap;
   a->gen1_native_rq |= b->gen1_native_rq;
   a->as_build |= b->as_build;
   a->uses_global_atomics |= b->uses_global_atomics;
   a->pack_loops |= b->pack_loops;
   a->calls_stream |= b->calls_stream;
   a->pack_dispatches += b->pack_dispatches;
   a->pack_threads = b->pack_threads > UINT64_MAX - a->pack_threads
                        ? UINT64_MAX : a->pack_threads + b->pack_threads;
   a->pack_static_work =
      b->pack_static_work > UINT64_MAX - a->pack_static_work
         ? UINT64_MAX : a->pack_static_work + b->pack_static_work;
   a->scratch.cs.main |= b->scratch.cs.main;
   a->scratch.cs.preamble |= b->scratch.cs.preamble;

   /* Submission sees only the merged root. Keep the destinations recorded
    * by both streams without changing their inline/link batching policy. */
   for (unsigned i = 0; i < b->compute_attachments.count; ++i) {
      const struct drm_asahi_attachment *attachment =
         &b->compute_attachments.list[i];
      brl_drm_attachment_set_add(&a->compute_attachments,
                                attachment->pointer, attachment->size);
   }
   a->compute_attachments.complete &= b->compute_attachments.complete;

   /* The merged command ends where b ended. It starts where a started; a
    * start stamp b carried was a top-of-pipe write, which any earlier value
    * also satisfies. The caller refuses the merge when both have one. */
   a->timestamp.end = b->timestamp.end;
   if (!a->timestamp.start.handle) {
      a->timestamp.start = b->timestamp.start;
      a->unordered_start_ts = b->unordered_start_ts;
   }

   util_dynarray_append_array(&a->compute_resources, uint8_t,
                             b->compute_resources.data, b->compute_resources.size);

   a->g17p_timestamp_query |= b->g17p_timestamp_query;

   a->stats.calls += b->stats.calls;
   a->stats.cmds += b->stats.cmds;
   a->stats.flushes += b->stats.flushes;
   a->stats.robust_index_draws += b->stats.robust_index_draws;
   a->stats.robust_index_dispatches += b->stats.robust_index_dispatches;
   a->stats.robust_index_batches += b->stats.robust_index_batches;
   a->stats.merged++;
   a->stats.query_copy_dispatches += b->stats.query_copy_dispatches;
   a->stats.query_copy_queries += b->stats.query_copy_queries;
   a->stats.query_copy_counters += b->stats.query_copy_counters;
   a->stats.query_reset_dispatches += b->stats.query_reset_dispatches;
   a->stats.query_reset_queries += b->stats.query_reset_queries;
   a->stats.query_reset_flushes += b->stats.query_reset_flushes;
   a->query_access |= b->query_access;
   a->query_access_untracked |= b->query_access_untracked;
   if (b->query_access_ranges.size) {
      if (!a->query_access_ranges.data)
         a->query_access_ranges = UTIL_DYNARRAY_INIT;
      void *dst = util_dynarray_grow_bytes(&a->query_access_ranges, 1,
                                           b->query_access_ranges.size);
      if (dst)
         memcpy(dst, b->query_access_ranges.data, b->query_access_ranges.size);
      else
         a->query_access_untracked = true;
   }
   assert(!a->query_reset_pending && !b->query_reset_pending);

   /* Frame anatomy: the merged command runs b's programs too. Without them
    * its whole duration is charged to a's programs alone. */
   for (unsigned j = 0; j < BRL_FA_FS_MAX && b->anatomy_fs[j].fs; ++j) {
      unsigned i = 0;
      while (i < BRL_FA_FS_MAX && a->anatomy_fs[i].fs &&
             a->anatomy_fs[i].fs != b->anatomy_fs[j].fs)
         ++i;
      if (i == BRL_FA_FS_MAX) {
         a->anatomy_fs_other += b->anatomy_fs[j].draws;
         continue;
      }
      a->anatomy_fs[i].fs = b->anatomy_fs[j].fs;
      a->anatomy_fs[i].draws += b->anatomy_fs[j].draws;
      a->anatomy_fs[i].threads += b->anatomy_fs[j].threads;
      a->anatomy_fs[i].groups += b->anatomy_fs[j].groups;
      a->anatomy_fs[i].indirect += b->anatomy_fs[j].indirect;
   }
   a->anatomy_fs_other += b->anatomy_fs_other;

   for (unsigned j = 0; j < BRL_FA_FS_MAX && b->anatomy_precomp[j].dispatches;
        ++j) {
      unsigned i = 0;
      while (i < BRL_FA_FS_MAX && a->anatomy_precomp[i].dispatches &&
             a->anatomy_precomp[i].program != b->anatomy_precomp[j].program)
         ++i;
      if (i == BRL_FA_FS_MAX) {
         a->anatomy_precomp_other += b->anatomy_precomp[j].dispatches;
         continue;
      }
      const uint64_t t = b->anatomy_precomp[j].threads;
      a->anatomy_precomp[i].threads =
         a->anatomy_precomp[i].dispatches == 0 ? t
         : (t == UINT64_MAX || a->anatomy_precomp[i].threads == UINT64_MAX)
            ? UINT64_MAX
            : a->anatomy_precomp[i].threads + t;
      a->anatomy_precomp[i].program = b->anatomy_precomp[j].program;
      a->anatomy_precomp[i].dispatches += b->anatomy_precomp[j].dispatches;
   }
   a->anatomy_precomp_other += b->anatomy_precomp_other;
}

static inline void
brl_cs_merge_cdm(struct brl_cs *a, const struct brl_cs *b)
{
   assert(a->cmd == b->cmd);
   brl_cs_merge_cdm_until(a, b, b->current);
}

static inline uint64_t
brl_cs_current_addr(struct brl_cs *cs)
{
   return cs->chunk.gpu + ((uint8_t *)cs->current - (uint8_t *)cs->chunk.cpu);
}

struct brl_uploader {
   /** List of brl_cmd_bo */
   struct list_head bos;

   /* Current addresses */
   uint8_t *map;
   uint64_t base;
   uint32_t offset;
   uint32_t capacity;

   /* VDM uploader only: low alias of base (its brl_vdm_allocation's alias). */
   uint64_t vdm_alias;

   /* Current backing BO is mapped CPU write-back (AGX_BO_WRITEBACK). */
   bool cpu_cached;
};

/*
 * Bump arena for the G17P Ioto programs one command buffer records. Semantic
 * program reuse happens before reservation; resource programs embed their
 * reserved address, so post-composition byte hashing cannot deduplicate them.
 * See brl_ioto_arena_core.h for why programs cannot each have a BO.
 */
struct brl_ioto_arena {
   /* Chunk currently being filled. Owned by brl_cmd_buffer::large_bos, unless
    * it is the span below, which has no bo of its own. */
   struct agx_bo *bo;
   struct brl_vdm_allocation *span;
   uint8_t *map;
   uint64_t base;
   uint32_t cursor;
   uint32_t capacity;

   /* Recording statistics, reported when the arena cannot grow. */
   uint32_t programs;
   uint64_t program_bytes;
   uint64_t chunk_bytes;
};

#define BRL_MAX_DEFERRED_CLEARS 8

struct brl_cmd_buffer {
   struct vk_command_buffer vk;

   /* Frame anatomy profiler only: API pipeline barriers recorded since the
    * last stream was created (see brl_cs::anatomy_api_barriers). */
   struct {
      uint32_t count;
      uint16_t mem, buf, img;
      uint64_t src, dst, src_access, dst_access;
   } anatomy_bar;

   /* frg-codegen v4: the per-recording BG/EOT cache (13.5 KiB) lives outside
    * `state` so the full-state memset of every reset no longer clears it;
    * brl_reset_cmd_buffer() invalidates the entries only when used. Entries are
    * only read when valid, so this is the same cache contract. */
   struct brl_g17p_bg_eot_cache_entry
      g17p_bg_eot_cache[BRL_G17P_BG_EOT_CACHE_ENTRIES];
   uint8_t g17p_bg_eot_cache_next;
   bool g17p_bg_eot_cache_used;

   /* Set when any of the four contiguous native graphics caches inside
    * state.gfx (g17p_ioto..g17p_spilled_rt_descs) is populated, so AGX2 and
    * compute-only recordings skip clearing their 8.5 KiB on reset. */
   bool g17p_gfx_caches_used;

   /* Ray-tracing bind point state lives outside `state` so graphics and
    * compute recordings skip clearing and tearing down its descriptors. */
   struct brl_ray_tracing_state rt;
   bool used_rt;

   /* M5: the render pass instance that ended last, while the next one may
    * continue in its render stream instead of storing and reloading the
    * attachments (brl_m5_pass_merge_begin()). Kept outside `state` because its
    * fields are read only when `valid` is set. */
   struct brl_pass_merge_candidate {
      bool valid;
      /* The instance's single render stream, where its draws ended and
       * where its terminator left the stream and chunk. */
      struct brl_cs *cs;
      void *resume_current;
      void *after_current;
      void *end;
      /* The tracked state when the instance ended; the next instance may
       * continue only if none of it moved. */
      uint32_t next_seq;
      uint32_t sync_epoch;
      struct brl_cs *open_cs;
      void *open_cs_current;
      /* A recorded dependency orders the next instance after this one
       * beyond attachment operations; continuing then needs the stronger
       * conditions of brl_m5_ordered_refusal(). */
      bool ordered;
      /* The render state the instance ended with. */
      struct brl_rendering_state render;
   } pass_merge;

   /* Per-recording compute group-count cache (384 B) lives outside `state` so
    * empty and graphics-only command buffer resets skip zeroing it. */
   struct brl_group_count_cache_entry {
      uint32_t x, y, z, valid;
      uint64_t addr;
   } group_count_cache[16];
   bool used_group_count_cache;

   /* Set by brl_reset_cmd_buffer, cleared by vkBeginCommandBuffer. */
   bool g17p_pristine;

   struct {
      struct brl_upload_stats upload_stats;
      bool compute_pair_simultaneous;
      struct {
         uint64_t id, addr;
      } constant_data[MESA_SHADER_STAGES];
      /* vkCmdBindDescriptorBuffersEXT() */
      uint64_t descriptor_buffers[BRL_MAX_DESCRIPTOR_BUFFERS];
      struct brl_graphics_state gfx;
      struct brl_compute_state cs;
      struct brl_conditional_render_state conditional;

      /* Whole-image clears not yet recorded, see
       * brl_flush_deferred_clears(). */
      struct brl_deferred_clear {
         struct brl_image *image;
         VkImageAspectFlags aspects;
         VkClearValue value;
      } deferred_clear[BRL_MAX_DEFERRED_CLEARS];
      uint8_t deferred_clear_count;
      bool deferred_clear_flushing;
      /* Recording a buffer-only command that does not touch their memory. */
      bool deferred_clear_hold;

      /* Suppress consumption of pending resets while recording another reset. */
      bool query_reset_recording;

      /* An event is set, reset or waited on: the host or another queue can
       * observe this recording before its submission completes. */
      bool uses_events;

      /* Recording brl_CmdWaitEvents(): the event waits stay recorded for the
       * pipeline barrier that carries its barriers. */
      bool brl_legacy_event_wait;

      /* Scoped around direct/indirect AS builds, including their meta blits. */
      bool as_build_recording;

      /* Render stream holding unstamped inline occlusion availability records
       * (cs->oq_pending). Cleared by brl_flush_oq_pending() or
       * brl_resolve_pending_oq_available(). */
      struct brl_cs *oq_pending_render;

      /* AGX3: the last render pass instance stored a whole depth-only image
       * that nothing recorded since may have read. A later instance that
       * clears the whole image, or does not care about its contents, makes
       * that store dead (brl_take_pending_z_store()). Kept across exactly the
       * commands deferred clears survive (deferred_clear_hold). */
      struct brl_pending_z_store {
         struct brl_cs *cs;
         const struct brl_image *image;
         struct agx_zls_control_packed zls_control;
         bool process_empty_tiles;
      } pending_z_store;

      /* M5: the open render stream is an ordered continuation (brl_cs::
       * m5_ordered) and a command recorded since may read the instance's
       * attachments through memory; the next draw ends the stream first. */
      bool m5_ordered_split;

      /* Cumulative GPU-built indirect Ioto bytes reserved in this recording,
       * bounding LOW_VA pressure across repeated indirect-count commands. */
      size_t indirect_vdm_ioto_bytes;

      /* Per recording; reported at submission so reused command buffers are
       * counted each time. Reasons overlap, total counts physical splits. */
      struct {
         uint64_t kept, splits;
         uint64_t reason[BRL_BARRIER_SPLIT_COUNT];
      } barrier_stats;
   } state;

   struct {
      struct brl_uploader main, usc, root, vdm, streams;
   } uploader;

   /* Command-pool slab spans leased by this command buffer's VDM uploader. */
   struct list_head vdm_allocations;
   /* Most recently addressed span. Recording often references the current
    * 4 KiB VDM chunk many times before moving to the next one. */
   struct brl_vdm_allocation *g17p_vdm_last_allocation;

   /* List of all recorded control streams */
   struct list_head control_streams;

   /* Current recorded control stream */
   struct {
      /* VDM stream for 3D */
      struct brl_cs *gfx;

      /* CDM stream for compute */
      struct brl_cs *cs;

      /* CDM stream that executes immediately before the current graphics
       * control stream. Used for geometry shading, tessellation, etc.
       */
      struct brl_cs *pre_gfx;

      /* CDM stream that will execute after the current graphics control stream
       * finishes. Used for queries.
       */
      struct brl_cs *post_gfx;
   } current_cs;

   /* Tessellation stages of the current render's draws, not yet in pre_gfx */
   struct brl_tess_batch tess_batch;

   /* Recorded dependencies between control streams, see brl_sync.h. */
   struct brl_sync_state sync;

   /* Tile hazards: struct brl_sync_pass_record for every recorded pass
    * instance footprint, in recording order. Cleared by brl_sync_reset. */
   struct util_dynarray sync_pass_records;

   /* struct brl_oq_end for every occlusion query this command buffer ends. */
   struct util_dynarray oq_ends;

   /* Are we currently inside a vk_meta operation? This alters sample location
    * behaviour.
    */
   bool in_meta;

   /* Flags passed to vkBeginCommandBuffer(). */
   VkCommandBufferUsageFlags usage_flags;

   /* CmdClearAttachments is the sole vk_meta operation affected by
    * conditional rendering. */
   bool conditional_meta;

   /* True while recording the meta draw inside vkCmdClearAttachments. */
   bool in_clear_attachments;

   /* True while brl_begin_rendering() records its pass-start clear draw. */
   bool in_begin_rendering_clear;

   /* Single-kick vkCmdTraceRays* on M5: while set, brl_dispatch_m5_native
    * writes the kernel's launch record (40-byte direct launch + barrier) here
    * instead of into the control stream. */
   uint8_t *m5_rt_capture;
   bool m5_rt_captured;

   /* XXX: move me?
    *
    * Indirect draw generated by the indirect GS translator.
    */
   uint64_t geom_index_buffer;
   uint32_t geom_index_count;
   uint32_t geom_instance_count;

   /* Does the command buffer use the geometry heap? */
   bool uses_heap;

   /* The push descriptor sets brl_meta_begin() saves, each only as long as
    * its layout, stacked in the order nested meta operations begin. */
   struct util_dynarray meta_push_saves;

   /* Owned BOs and retained references needed by recorded commands. */
   struct util_dynarray large_bos;
   /* Direct-mapped temporal cache avoids rescanning large_bos for the linked
    * vertex/fragment shader, setup and rodata BOs repeated by every draw. */
   struct agx_bo *recent_retained_bos[16];
   /* Open-addressed pointer set of the BOs brl_cmd_buffer_retain_bo() has
    * appended to large_bos (capacity a power of two, or zero). A recording
    * retains one BO per distinct shader/setup/program it publishes, so a
    * linear scan of large_bos on every temporal-cache miss grew quadratically
    * with the number of pipelines a command buffer uses. */
   struct agx_bo **retained_set;
   uint32_t retained_set_capacity;
   uint32_t retained_set_count;

   /* M5: every attachment-capable image (brl_image) a graphics push descriptor
    * has referenced since recording began. Push descriptors stay valid until
    * overwritten, so an image recorded here may still be readable by the
    * bound shaders (brl_m5_ordered_refusal()). Created on first use. */
   struct set *m5_pushed_images;

   /* Executable USC-window arena the G17P Ioto programs recorded by this
    * command buffer are bump-allocated from. Its chunks are owned by
    * large_bos, so it dies with the recording like every other program
    * reference. */
   struct brl_ioto_arena ioto;

   /* M5 with brl_device::host_alias.dynamic: the application dependencies
    * recorded without memory scope (struct brl_alias_site), written once an
    * import may alias (brl_cdm_alias_scope()). alias_patched (atomic) once
    * they are; alias_registered (atomic) while alias_link is in
    * brl_device::host_alias.cmd_buffers. */
   struct util_dynarray alias_sites;
   uint32_t alias_patched;
   uint32_t alias_registered;
   struct list_head alias_link;

   /* AGX2 reduced-word CDM barrier records emitted while no sparse resource
    * existed (struct brl_sparse_site), upgraded to the full word at submit
    * when dev->sparse_used is set (brl_submit_patch_sparse_sites()). */
   struct util_dynarray sparse_sites;
};

/* A patch site: the stream word of a barrier record an application
 * dependency was recorded with, without memory scope, because no host import
 * could alias then (brl_cdm_alias_scope()). `value` is the memory-scoped
 * strong flush (brl_device::host_alias.flush_word), which only strengthens
 * the record: a weak ordering record, or a strong flush, becomes the strong
 * flush with memory scope, as brl_cdm_flush() upgrades in place. A
 * site stays a barrier record for the life of the recording: an application
 * dispatch after it no longer drops it, and a merge that folds a leading
 * barrier over it writes a barrier. */
struct brl_alias_site {
   uint32_t *word;
   uint32_t value;
};

/* A patch site: the stream word of an AGX2 barrier record emitted with the
 * reduced flag word while the device held no sparse resource. `value` is
 * the full-word record, written at submit when a sparse resource exists by
 * then (brl_submit_patch_sparse_sites()). */
struct brl_sparse_site {
   uint32_t *word;
   uint32_t value;
};

/* Whether an application dependency recorded now needs memory scope: as
 * brl_cdm_alias_scope(), and also, with host_alias.dynamic, in a command
 * buffer that uses events. */
bool brl_cmd_alias_scope(struct brl_cmd_buffer *cmd);

/* After an application dependency was recorded into cs without memory scope
 * because brl_cmd_alias_scope() was false: with host_alias.dynamic, records
 * the stream's last record as a patch site, or, without such a record,
 * flushes with memory scope now. */
void brl_cdm_note_alias_site(struct brl_cmd_buffer *cmd, struct brl_cs *cs);

/* Writes the command buffer's patch sites, once. Callers ensure no reset
 * runs concurrently; two writers write the same words. */
void brl_cmd_buffer_patch_alias_sites(struct brl_cmd_buffer *cmd);

/* Records the just-emitted reduced-word AGX2 barrier record at `word` as a
 * sparse patch site in cs's stream (brl_cdm_flush()). */
void brl_cdm_note_sparse_site(struct brl_cs *cs, uint32_t *word);

/* Writes each submitted command buffer's sparse patch sites, once
 * dev->sparse_used is set: an upgrade only strengthens the record. */
void brl_submit_patch_sparse_sites(struct brl_device *dev,
                                  struct vk_command_buffer *const *cmds,
                                  unsigned count);

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT and
 * its sub-stages). Nesting is fine: meta helpers re-enter through the
 * dispatch table. */
static inline void
brl_sync_xfer_begin(struct brl_cmd_buffer *cmd)
{
   /* An outermost transfer without a known destination may write anything
    * (brl_sync_note_xfer_write()). */
   if (cmd->sync.xfer_depth++ == 0)
      cmd->sync.xfer_dst = (struct brl_sync_xfer_dst){0};
}

/* As brl_sync_xfer_begin(), for a transfer whose writes stay inside the GPU
 * VA range [lo, hi). device_address says shaders may also reach the range
 * through a buffer device address. A nested transfer widens the range of the
 * outermost one. */
static inline void
brl_sync_xfer_begin_dst(struct brl_cmd_buffer *cmd, uint64_t lo, uint64_t hi,
                       bool device_address)
{
   struct brl_sync_xfer_dst *dst = &cmd->sync.xfer_dst;
   if (cmd->sync.xfer_depth++ == 0) {
      *dst = (struct brl_sync_xfer_dst){
         .lo = lo,
         .hi = hi,
         .valid = lo < hi,
         .device_address = device_address,
      };
   } else if (dst->valid && lo < hi) {
      dst->lo = MIN2(dst->lo, lo);
      dst->hi = MAX2(dst->hi, hi);
      dst->device_address |= device_address;
   }
}

static inline void
brl_sync_xfer_end(struct brl_cmd_buffer *cmd)
{
   assert(cmd->sync.xfer_depth > 0);
   if (--cmd->sync.xfer_depth == 0)
      cmd->sync.xfer_dst = (struct brl_sync_xfer_dst){0};
}

/*
 * One reservation inside the Ioto arena: where the program is to be composed,
 * and the address the composer must be given for it.
 */
struct brl_ioto_reservation {
   void *map;
   uint64_t va;
};

/*
 * Reserve `size` bytes of executable USC-window space for one Ioto program.
 * On failure the command buffer error is already set.
 */
bool brl_cmd_reserve_ioto_program_slow(struct brl_cmd_buffer *cmd, size_t size,
                                      const char *label,
                                      struct brl_ioto_reservation *reservation);

/* AGX_IOTO_PROGRAM_ALIGNMENT, checked in brl_ioto_arena.c. */
#define BRL_IOTO_PROGRAM_ALIGNMENT 64u

static inline bool
brl_cmd_reserve_ioto_program(struct brl_cmd_buffer *cmd, size_t size,
                            const char *label,
                            struct brl_ioto_reservation *reservation)
{
   /* The placement brl_ioto_arena_place() computes, in the current chunk. */
   struct brl_ioto_arena *arena = &cmd->ioto;
   const uint32_t mask = BRL_IOTO_PROGRAM_ALIGNMENT - 1;
   if (likely(reservation && size != 0 && size <= UINT32_MAX && arena->map &&
              arena->cursor <= arena->capacity &&
              arena->cursor <= UINT32_MAX - mask)) {
      const uint32_t offset = (arena->cursor + mask) & ~mask;
      if (likely(offset <= arena->capacity &&
                 (uint32_t)size <= arena->capacity - offset)) {
         *reservation = (struct brl_ioto_reservation){
            .map = arena->map + offset,
            .va = arena->base + offset,
         };
         arena->cursor = offset + (uint32_t)size;
         return true;
      }
   }

   return brl_cmd_reserve_ioto_program_slow(cmd, size, label, reservation);
}

/*
 * Publish the `length` bytes composed at reservation->map and return the
 * reserved address for the launch record.
 */
uint64_t brl_cmd_publish_ioto_program(struct brl_cmd_buffer *cmd,
                                     const struct brl_ioto_reservation *
                                        reservation,
                                     size_t length);

/* Clean the current chunk's programs from the CPU cache (cached command
 * memory); brl_cmd_buffer_publish_cached_uploads() fences it. */
bool brl_ioto_arena_clean(struct brl_cmd_buffer *cmd);

/* Release the arena's bookkeeping. The chunks themselves are in large_bos. */
void brl_ioto_arena_finish(struct brl_cmd_buffer *cmd);

VK_DEFINE_HANDLE_CASTS(brl_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops brl_cmd_buffer_ops;

static inline struct brl_cmd_pool *
brl_cmd_buffer_pool(struct brl_cmd_buffer *cmd)
{
   return (struct brl_cmd_pool *)cmd->vk.pool;
}

/* Whether the current blend state reads the blend constants. Derived from
 * the dynamic graphics state at each use rather than cached: the fragment
 * stage program (its blend-constant uniform range) is built by bind-time and
 * BeginRendering preparation as well as by the draw, and that preparation
 * runs before the draw-time dirty-state flush. A cached copy refreshed only
 * by that flush was stale there, so the first draw after a blend-equation
 * change published a fragment program without the blend-constant range and
 * the epilog read unwritten uniforms. Blend-constant value changes dirty and
 * re-upload the root (invalidating prepared->valid and moving the root
 * address), while changes in whether an epilog reads blend constants change
 * linked_fs. */
static inline bool
brl_uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                    VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                 "constant factors are contiguous");

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      const unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

static inline bool
brl_cmd_uses_blend_constant(const struct brl_cmd_buffer *cmd)
{
   return brl_uses_blend_constant(&cmd->vk.dynamic_graphics_state.cb);
}

/*
 * The hardware vertex shader is supplied by the last geometry stage. The
 * geometry pipeline is vertex->tess->geometry so we search backwards.
 */
static inline struct brl_shader *
brl_bound_hw_vs(struct brl_graphics_state *gfx)
{
   struct brl_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct brl_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct brl_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct brl_api_shader *mesh = gfx->shaders[MESA_SHADER_MESH];

   /* A mesh pipeline has no vertex, tessellation or geometry stage; its
    * native mesh variant is the hardware vertex program. */
   if (mesh)
      return &mesh->variants[BRL_MESH_VARIANT_MAIN];
   else if (gs)
      return &gs->variants[BRL_GS_VARIANT_RAST];
   else if (tes)
      return &tes->variants[BRL_VS_VARIANT_HW];
   else
      return &vs->variants[BRL_VS_VARIANT_HW];
}

static inline struct brl_shader *
brl_bound_sw_vs(struct brl_graphics_state *gfx)
{
   struct brl_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct brl_shader *hw_vs = brl_bound_hw_vs(gfx);

   /* Mesh pipelines: the raster variant is both the hardware and the only
    * vertex program (no attributes, no prolog remap). */
   if (!vs)
      return hw_vs;

   if (hw_vs == &vs->variants[BRL_VS_VARIANT_HW])
      return hw_vs;
   else
      return &vs->variants[BRL_VS_VARIANT_SW];
}

static inline struct brl_shader *
brl_bound_sw_vs_before_gs(struct brl_graphics_state *gfx)
{
   struct brl_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct brl_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct brl_api_shader *api = tes ?: vs;

   /* Mesh pipelines have no software vertex stage. */
   if (!api)
      return NULL;

   return &api->variants[BRL_VS_VARIANT_SW];
}

struct agx_ptr brl_pool_alloc_from_slow(struct brl_cmd_buffer *cmd,
                                       struct brl_uploader *uploader,
                                       uint32_t size, uint32_t alignment,
                                       enum brl_cmd_bo_type type);

/* Bump allocation from the uploader's current BO; a new BO or a large
 * allocation takes brl_pool_alloc_from_slow(). */
static inline struct agx_ptr
brl_pool_alloc_from(struct brl_cmd_buffer *cmd, struct brl_uploader *uploader,
                   uint32_t size, uint32_t alignment,
                   enum brl_cmd_bo_type type)
{
   assert(alignment > 0);
   const uint32_t offset = align(uploader->offset, alignment);
   if (likely(uploader->map != NULL && offset <= uploader->capacity &&
              size <= uploader->capacity - offset)) {
      uploader->offset = offset + size;
      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   return brl_pool_alloc_from_slow(cmd, uploader, size, alignment, type);
}

struct agx_ptr brl_pool_cs_alloc(struct brl_cmd_buffer *cmd, uint32_t size,
                                uint32_t alignment);
struct agx_ptr brl_pool_vdm_primary_alloc(struct brl_cmd_buffer *cmd,
                                         uint32_t size, uint32_t alignment);

uint64_t brl_g17p_vdm_compact(struct brl_cmd_buffer *cmd, uint64_t primary);
uint64_t brl_g17p_vdm_alias(struct brl_cmd_buffer *cmd, uint64_t primary);

uint64_t brl_pool_upload(struct brl_cmd_buffer *cmd, const void *data,
                        uint32_t size, uint32_t alignment);

/*
 * G17P does not honour the low five bits of the texture-table pointer that
 * the combined-sampler ABIs publish in uniform halves 0..3: the 32-byte
 * texture state is read as if the pointer were 32-byte aligned.  Sampled-image
 * descriptors only have 8-byte alignment and a 72-byte stride in a descriptor
 * set, so set address + binding offset is generally not a valid table
 * pointer.  Measured on G17P with vkCmdBlitImage of D32_SFLOAT_S8_UINT, whose
 * pushed set holds the depth and stencil texture words at offsets 72 and 144:
 * every depth and stencil texel read back wrong, and when the qword before
 * the stencil texture word (the depth descriptor's last qword) was nonzero the
 * texture unit's read faulted (VM error at 0x40000000000, VK_ERROR_DEVICE_LOST
 * in Pokemon Scarlet).  The same descriptor bytes published at a 32-byte
 * aligned address -- including 32 mod 64 -- read back exactly and never
 * faulted, while 8 and 16 mod 32 still read back wrong.
 *
 * Returns texture_va when it is already aligned, otherwise the address of a
 * 32-byte-aligned command-buffer copy of the descriptor record (texture word
 * plus the software fields after it, which descriptor crawls through the
 * same pointer may read), or 0 on allocation failure.  The ABIs using this
 * that need a copy are not selected when sampled-image update-after-bind is
 * enabled.  The direct txf ABI may still publish an already aligned live set
 * pointer in that case.  The combined-sampler ABIs also snapshot the
 * descriptor's sampler index at record time, so their copies hold what the set
 * holds when the commands execute.
 */
#define BRL_G17P_TEXTURE_STATE_ALIGN 32

uint64_t brl_g17p_texture_state_pointer(struct brl_cmd_buffer *cmd,
                                       uint64_t texture_va,
                                       const void *descriptor,
                                       uint32_t available_B);
uint64_t brl_pool_usc_upload(struct brl_cmd_buffer *cmd, const void *data,
                            uint32_t size, uint32_t alignment);

/* Keep externally owned storage referenced by recorded GPU commands alive for
 * the command-buffer lifetime. Repeated uses of the same BO are deduplicated.
 */
void brl_cmd_buffer_retain_bo(struct brl_cmd_buffer *cmd, struct agx_bo *bo);
void brl_cmd_buffer_release_bos(struct brl_cmd_buffer *cmd,
                               struct agx_device *dev);

static inline struct agx_ptr
brl_pool_alloc(struct brl_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return brl_pool_alloc_from(cmd, &cmd->uploader.main, size, alignment,
                             BRL_CMD_BO_MAIN);
}

static inline struct agx_ptr
brl_pool_usc_alloc(struct brl_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return brl_pool_alloc_from(cmd, &cmd->uploader.usc, size, alignment,
                             BRL_CMD_BO_USC);
}

/*
 * Allocate a payload that a VDM record references through its 40-bit pointer.
 * HAL200 draw-state fields are narrower than the VDM control-stream pointer,
 * so those payloads retain a device-global low alias. Control-stream roots
 * themselves use the constrained main heap directly.
 */
static inline struct agx_ptr
brl_pool_vdm_alloc(struct brl_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);

   /* Bump-allocate from the current VDM chunk and derive its compact pointer
    * from the chunk's alias.  Allocations never overlap, so this is the
    * address brl_g17p_vdm_compact() resolves through the chunk's
    * brl_vdm_allocation; anything else takes the general path below. */
   struct brl_uploader *vdm = &cmd->uploader.vdm;
   const uint32_t vdm_offset = align(vdm->offset, alignment);
   uint32_t vdm_compact;
   if (likely(vdm->map && vdm_offset <= vdm->capacity &&
              size <= vdm->capacity - vdm_offset &&
              brl_g17p_vdm_low_window_enabled(&dev->dev) &&
              agx_g17p_vdm_alias_compact_inline(vdm->vdm_alias + vdm_offset,
                                                &vdm_compact) &&
              vdm_compact != 0)) {
      vdm->offset = vdm_offset + size;
      return (struct agx_ptr){
         .gpu = vdm_compact,
         .cpu = vdm->map + vdm_offset,
      };
   }

   struct agx_ptr T = brl_pool_vdm_primary_alloc(cmd, size, alignment);

   if (T.gpu && brl_g17p_vdm_low_window_enabled(&dev->dev)) {
      T.gpu = brl_g17p_vdm_compact(cmd, T.gpu);
      if (!T.gpu) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         T.cpu = NULL;
      }
   }

   return T;
}

bool brl_cs_init_graphics(struct brl_cmd_buffer *cmd, struct brl_cs *cs);
/* A depth/stencil clear touched the pass's attachment (ZLS elision). */
void brl_g17p_note_depth_clear(struct brl_cmd_buffer *cmd,
                              VkImageAspectFlags aspects);

uint64_t brl_default_sample_positions(unsigned nr_samples);

void brl_flush_deferred_clears_why(struct brl_cmd_buffer *cmd, const char *why);
#define brl_flush_deferred_clears(cmd) brl_flush_deferred_clears_why(cmd, __func__)

/* The depth image of the pending store is overwritten as a whole before
 * anything read it: the store is dead (see brl_take_pending_z_store()). */
static inline void
brl_kill_pending_z_store(struct brl_cmd_buffer *cmd)
{
   struct brl_pending_z_store *p = &cmd->state.pending_z_store;
   p->cs->cr.zls_control = p->zls_control;
   p->cs->cr.process_empty_tiles = p->process_empty_tiles;
   p->cs = NULL;
}
void brl_deferred_clears_hold_buffer(struct brl_cmd_buffer *cmd, uint64_t addr,
                                    uint64_t size);
bool brl_defer_clear(struct brl_cmd_buffer *cmd, struct brl_image *image,
                    VkImageAspectFlags aspects, const VkClearValue *value,
                    uint32_t range_count, const VkImageSubresourceRange *ranges);
const VkRenderingInfo *
brl_take_deferred_clears(struct brl_cmd_buffer *cmd, const VkRenderingInfo *info,
                        VkRenderingInfo *copy,
                        VkRenderingAttachmentInfo *color,
                        VkRenderingAttachmentInfo *depth,
                        VkRenderingAttachmentInfo *stencil);

void brl_flush_query_resets(struct brl_cmd_buffer *cmd, struct brl_cs *cs);

/* A control stream outside the command buffer's list of streams, on a root
 * of brl_cs_root_size() bytes. */
static inline struct brl_cs *
brl_cs_create_at(struct brl_cmd_buffer *cmd, bool compute, struct agx_ptr root)
{
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   struct brl_cmd_pool *pool = cmd->vk.pool ? brl_cmd_buffer_pool(cmd) : NULL;
   size_t initial_size = brl_cs_root_size(dev->dev.chip, compute);

   /* Recycle a pool-owned brl_cs (keeping scissor/depth_bias storage warm)
    * when available to avoid per-pass heap traffic on steady-state recordings. */
   struct brl_cs *cs;
   struct util_dynarray scissor = UTIL_DYNARRAY_INIT;
   struct util_dynarray depth_bias = UTIL_DYNARRAY_INIT;
   if (pool && !list_is_empty(&pool->free_cs)) {
      cs = list_first_entry(&pool->free_cs, struct brl_cs, node);
      list_del(&cs->node);
      pool->num_free_cs--;
      scissor = cs->scissor;
      depth_bias = cs->depth_bias;
   } else {
      cs = malloc(sizeof(*cs));
      if (!cs)
         return NULL;
   }
   *cs = (struct brl_cs){
      .cmd = cmd,
      .type = compute ? BRL_CS_CDM : BRL_CS_VDM,
      .addr = root.gpu,
      .start = root.cpu,
      .root_size = initial_size,
      .chunk = root,
      .current = root.cpu,
      .end = root.cpu + initial_size,
      .scissor = scissor,
      .depth_bias = depth_bias,
   };

   util_dynarray_init(&cs->compute_resources, NULL);
   util_dynarray_init(&cs->render_resources, NULL);
   list_inithead(&cs->node);
   brl_drm_attachment_set_init(&cs->compute_attachments);
   /* Only known meta destinations are collected so far. */
   cs->compute_attachments.complete = false;
   return cs;
}

static inline struct brl_cs *
brl_cs_create(struct brl_cmd_buffer *cmd, bool compute)
{
   /* Allocate root control stream */
   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   struct agx_ptr root =
      brl_pool_cs_alloc(cmd, brl_cs_root_size(dev->dev.chip, compute), 1024);
   return root.cpu ? brl_cs_create_at(cmd, compute, root) : NULL;
}

static inline void brl_flush_oq_pending(struct brl_cmd_buffer *cmd);
static inline void brl_cs_destroy(struct brl_cs *cs);

static inline struct brl_cs *
brl_cmd_buffer_get_cs_general(struct brl_cmd_buffer *cmd, struct brl_cs **ptr,
                             bool compute)
{
   /* Any recorded work may read an image with a deferred clear, or the
    * depth image of a pending store. */
   if (unlikely(cmd->state.deferred_clear_count) &&
       !cmd->state.deferred_clear_flushing && !cmd->state.deferred_clear_hold)
      brl_flush_deferred_clears(cmd);
   if (unlikely(cmd->state.pending_z_store.cs) &&
       !cmd->state.deferred_clear_hold)
      cmd->state.pending_z_store.cs = NULL;
   if (unlikely(cmd->state.oq_pending_render) &&
       (ptr == &cmd->current_cs.post_gfx ||
        cmd->current_cs.gfx != cmd->state.oq_pending_render))
      brl_flush_oq_pending(cmd);

   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   if (*ptr && (*ptr)->g18_dispatch_sealed)
      *ptr = NULL;

   /* A recorded dependency may have ordered the open general compute stream
    * after work that follows it in the list. It is then ended, and the work
    * goes into a stream positioned after that requirement. Work of a new
    * stage class must also update the reused stream's kind and dependencies,
    * so later barriers include transfers and meta graphics emitted there. */
   if (*ptr && ptr == &cmd->current_cs.cs &&
       (cmd->sync.cs_check_pending ||
        (brl_sync_general_cs_kind(cmd) & ~(*ptr)->sync.kind)))
      brl_sync_check_cs(cmd);

   /* Work for the pre-graphics stream follows the batched tessellation of the
    * draws recorded before it. */
   if (ptr == &cmd->current_cs.pre_gfx && cmd->tess_batch.draws &&
       !cmd->tess_batch.recording)
      brl_tess_batch_flush(cmd);

   if ((*ptr) == NULL) {
      struct brl_device *dev = brl_cmd_buffer_device(cmd);
      struct brl_cs *cs = brl_cs_create(cmd, compute);
      if (!cs)
         return NULL;

      bool before_gfx = (ptr == &cmd->current_cs.pre_gfx);

      if (!compute) {
         const bool initialized = brl_cs_init_graphics(cmd, cs);
         if (!brl_graphics_cs_publishable(
                dev->dev.chip, initialized,
                cs->agx3_active_context_selector_valid)) {
            brl_cs_destroy(cs);
            return NULL;
         }
      }

      /* Insert into the command buffer. We usually append to the end of the
       * command buffer, except for pre-graphics streams which go right before
       * the graphics workload. (This implies a level of out-of-order processing
       * that's allowed by Vulkan and required for efficient
       * geometry/tessellation shaders.)
       */
      if (before_gfx && cmd->current_cs.gfx) {
         list_addtail(&cs->node, &cmd->current_cs.gfx->node);
      } else {
         list_addtail(&cs->node, &cmd->control_streams);
      }

      *ptr = cs;

      brl_sync_stream_created(cmd, cs, ptr, compute);
   }

   assert(*ptr != NULL);
   if (compute && unlikely((*ptr)->query_reset_pending) &&
       !cmd->state.query_reset_recording)
      brl_flush_query_resets(cmd, *ptr);
   if (compute && cmd->state.as_build_recording)
      (*ptr)->as_build = true;
   /* Transfer entry points also reuse an open CDM stream, so note each use,
    * not just stream creation (brl_sync_note_xfer_write()). */
   if (compute && unlikely(cmd->sync.xfer_depth) && !cmd->sync.in_pass &&
       brl_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P)
      brl_sync_note_xfer_write(cmd, *ptr);
   return *ptr;
}

static inline struct brl_cs *
brl_cmd_buffer_get_cs(struct brl_cmd_buffer *cmd, bool compute)
{
   struct brl_cs **ptr = compute ? &cmd->current_cs.cs : &cmd->current_cs.gfx;
   return brl_cmd_buffer_get_cs_general(cmd, ptr, compute);
}

void brl_ensure_cs_has_space(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                            size_t space);

static inline uint64_t
brl_cs_alloc_for_indirect(struct brl_cs *cs, size_t size_B)
{
   brl_ensure_cs_has_space(cs->cmd, cs, size_B);

   uint64_t addr = brl_cs_current_addr(cs);
   cs->current += size_B;
   return addr;
}

static void
brl_cmd_buffer_dirty_all(struct brl_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct brl_graphics_state *gfx = &cmd->state.gfx;

   vk_dynamic_graphics_state_dirty_all(dyn);
   gfx->dirty = ~0;
   gfx->shaders_dirty = ~0;
   gfx->linked_dirty = ~0;
   brl_root_mark_dirty(&gfx->descriptors);
   gfx->g17p_bind_tail.valid = false;
   gfx->g17p_bind_tail.fs_record_in_tail_valid = false;
   gfx->g17p_bind_tail_memo.valid = false;
   gfx->g17p_bind_tail_memo.viewport_valid = false;
   if (cmd->current_cs.gfx)
      cmd->current_cs.gfx->g17p_bind_tail.valid = false;
}

static inline struct brl_api_shader *
brl_bound_fs_obj(struct brl_graphics_state *gfx)
{
   if (gfx->mesh_fs)
      return gfx->mesh_fs;
   return gfx->fbl_fs ? gfx->fbl_fs : gfx->shaders[MESA_SHADER_FRAGMENT];
}

static inline struct brl_shader *
brl_bound_fs(struct brl_cmd_buffer *cmd)
{
   return brl_fs_variant(brl_bound_fs_obj(&cmd->state.gfx),
                        cmd->vk.dynamic_graphics_state.ms.rasterization_samples);
}

static inline size_t
brl_gfx_root_descriptor_size(const struct brl_cmd_buffer *cmd)
{
   const size_t full = sizeof(struct brl_root_descriptor_table);
   const struct brl_graphics_state *gfx = &cmd->state.gfx;
   if (gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH] ||
       !gfx->shaders[MESA_SHADER_VERTEX])
      return full;

   const struct brl_shader *vs =
      &gfx->shaders[MESA_SHADER_VERTEX]->variants[BRL_VS_VARIANT_HW];
   if (!vs->compiled)
      return full;

   struct brl_api_shader *fs_obj =
      gfx->mesh_fs ? gfx->mesh_fs
      : (gfx->fbl_fs ? gfx->fbl_fs : gfx->shaders[MESA_SHADER_FRAGMENT]);
   uint32_t fs_size = 0;
   if (fs_obj) {
      const struct brl_shader *fs = brl_fs_variant(
         fs_obj, cmd->vk.dynamic_graphics_state.ms.rasterization_samples);
      if (!fs || !fs->compiled)
         return full;
      fs_size = fs->root_descriptor_size;
   }

   return brl_gfx_root_size_from_stages(vs->root_descriptor_size, fs_obj != NULL,
                                       fs_size, full);
}

struct brl_query_access_range {
   uint32_t handle;
   uint64_t start_B;
   uint64_t end_B;
};

/* Record a bound query-pool byte range in ranges. Handle 0 (no kernel
 * timestamp binding) and empty ranges have nothing the kernel publishes and
 * record nothing; a range that cannot be recorded sets *untracked, which
 * fails closed (no DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS). */
static inline void
brl_query_ranges_note(struct util_dynarray *ranges, bool *untracked,
                     uint32_t handle, uint64_t start_B, uint64_t size_B)
{
   if (!handle || !size_B)
      return;

   if (size_B > UINT64_MAX - start_B) {
      *untracked = true;
      return;
   }

   if (!ranges->data)
      *ranges = UTIL_DYNARRAY_INIT;

   struct brl_query_access_range *slot =
      util_dynarray_grow_bytes(ranges, 1, sizeof(*slot));
   if (!slot) {
      *untracked = true;
      return;
   }

   *slot = (struct brl_query_access_range){
      .handle = handle,
      .start_B = start_B,
      .end_B = start_B + size_B,
   };
}

/* A query word this stream reads or writes. Only decides
 * DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS; it does not set query_access, which
 * records copies and resets alone. Recording paths use the brl_cmd_ forms
 * below. */
static inline void
brl_cs_note_query_range(struct brl_cs *cs, uint32_t handle, uint64_t start_B,
                       uint64_t size_B)
{
   if (cs)
      brl_query_ranges_note(&cs->query_access_ranges,
                           &cs->query_access_untracked, handle, start_B,
                           size_B);
}

/* A query word written by a queued immediate write (imm_writes). Moves to
 * query_access_ranges when the writes are dispatched. */
static inline void
brl_cs_note_imm_write_query_range(struct brl_cs *cs, uint32_t handle,
                                 uint64_t start_B, uint64_t size_B)
{
   if (cs)
      brl_query_ranges_note(&cs->imm_write_query_ranges,
                           &cs->imm_write_query_untracked, handle, start_B,
                           size_B);
}

/* Words are recorded only where the kernel accepts
 * DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS: elsewhere nothing reads them. */
static inline bool
brl_cmd_tracks_query_ranges(const struct brl_cmd_buffer *cmd)
{
   return brl_cmd_buffer_device(cmd)->dev.params.features &
          DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
}

static inline void
brl_cmd_note_query_range(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                        uint32_t handle, uint64_t start_B, uint64_t size_B)
{
   if (brl_cmd_tracks_query_ranges(cmd))
      brl_cs_note_query_range(cs, handle, start_B, size_B);
}

static inline void
brl_cmd_note_imm_write_query_range(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                                  uint32_t handle, uint64_t start_B,
                                  uint64_t size_B)
{
   if (brl_cmd_tracks_query_ranges(cmd))
      brl_cs_note_imm_write_query_range(cs, handle, start_B, size_B);
}

static inline void
brl_cs_destroy(struct brl_cs *cs)
{
   brl_cs_invalidate_compute_pair(cs);
   util_dynarray_fini(&cs->compute_resources);
   util_dynarray_fini(&cs->render_resources);
   util_dynarray_fini(&cs->scissor);
   util_dynarray_fini(&cs->depth_bias);
   util_dynarray_fini(&cs->imm_writes);
   util_dynarray_fini(&cs->imm_write_query_ranges);
   util_dynarray_fini(&cs->oq_stamps);
   util_dynarray_fini(&cs->query_access_ranges);

   free(cs);
}

#define BRL_CMD_POOL_CS_CACHE_MAX 4u

static inline void
brl_cs_recycle(struct brl_cmd_pool *pool, struct brl_cs *cs)
{
   if (!pool || pool->num_free_cs >= BRL_CMD_POOL_CS_CACHE_MAX) {
      brl_cs_destroy(cs);
      return;
   }

   brl_cs_invalidate_compute_pair(cs);
   util_dynarray_fini(&cs->compute_resources);
   util_dynarray_fini(&cs->render_resources);
   util_dynarray_fini(&cs->imm_writes);
   util_dynarray_fini(&cs->imm_write_query_ranges);
   util_dynarray_fini(&cs->oq_stamps);
   util_dynarray_fini(&cs->query_access_ranges);
   util_dynarray_clear(&cs->scissor);
   util_dynarray_clear(&cs->depth_bias);
   list_add(&cs->node, &pool->free_cs);
   pool->num_free_cs++;
}

void brl_dispatch_imm_writes(struct brl_cmd_buffer *cmd, struct brl_cs *cs);
bool brl_defer_oq_available(struct brl_cmd_buffer *cmd, struct brl_cs *cs);
void brl_resolve_pending_oq_available(struct brl_cmd_buffer *cmd);
void brl_resolve_deferred_oq_available(struct brl_cmd_buffer *cmd);

bool brl_g17p_smooth_varyings_supported(
   struct brl_cmd_buffer *cmd, const struct brl_linked_shader *linked);

void *brl_cmd_buffer_emit_vdm_terminate(struct brl_cmd_buffer *cmd,
                                       void *current);

/* Ends a compute stream holding launches with a flush that has memory scope
 * while brl_cmd_alias_scope() asks for it; otherwise its last record is a
 * patch site (brl_cdm_note_alias_site()). */
void brl_cdm_end_stream_flush(struct brl_cmd_buffer *cmd, struct brl_cs *cs);

static void
brl_cmd_buffer_end_compute_internal(struct brl_cmd_buffer *cmd,
                                   struct brl_cs **ptr)
{
   if (*ptr) {
      struct brl_cs *cs = *ptr;
      if (unlikely(cs->query_reset_pending))
         brl_flush_query_resets(cmd, cs);

      /* This control stream may write immediates as it ends. Queue the writes
       * now that we're done emitting everything else. Occlusion availability
       * alone may instead wait for the end of the command buffer, which
       * keeps the (still empty) stream in the list.
       */
      if (cs->imm_writes.size) {
         if (brl_defer_oq_available(cmd, cs)) {
            *ptr = NULL;
            return;
         }
         brl_dispatch_imm_writes(cmd, cs);
      }

      brl_cdm_end_stream_flush(cmd, cs);

      /* A transformed draw may reserve a pre-graphics stream and then prove
       * that it has no vertices to process. Do not leave an empty CDM job in
       * the submission list. A stream that only carries a timestamp stays:
       * that is the one job the stamp can ride on. */
      if (cs->stats.cmds == 0 && cs->stats.flushes == 0 &&
          !cs->timestamp.end.handle && !cs->timestamp.start.handle) {
         brl_sync_forget_stream(cmd, cs);
         list_del(&cs->node);
         brl_cs_destroy(cs);
      }
   }

   *ptr = NULL;
}

static inline void
brl_flush_oq_pending(struct brl_cmd_buffer *cmd)
{
   struct brl_cs *render = cmd->state.oq_pending_render;
   if (!render)
      return;

   cmd->state.oq_pending_render = NULL;
   const uint8_t count = render->oq_pending_count;
   render->oq_pending_count = 0;
   if (!count)
      return;

   struct brl_cs *cs =
      brl_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.post_gfx, true);
   if (!cs || vk_command_buffer_has_error(&cmd->vk))
      return;

   if (!cs->imm_writes.data)
      cs->imm_writes = UTIL_DYNARRAY_INIT;
   if (!cs->oq_stamps.data)
      cs->oq_stamps = UTIL_DYNARRAY_INIT;

   for (unsigned i = 0; i < count; ++i) {
      const struct brl_oq_pending *p = &render->oq_pending[i];
      const struct libagx_imm_write imm = {
         .address = p->avail_addr,
         .value = 1,
      };
      util_dynarray_append(&cs->imm_writes, imm);
      brl_cmd_note_imm_write_query_range(cmd, cs, p->handle, p->offset_B,
                                        p->avail_size);
      const struct brl_oq_stamp stamp = {
         .handle = p->handle,
         .offset_B = p->offset_B,
         .render_seq = render->sync.seq,
      };
      util_dynarray_append(&cs->oq_stamps, stamp);
   }

   if (cmd->current_cs.gfx != render)
      brl_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
}

static void
brl_cmd_buffer_end_compute(struct brl_cmd_buffer *cmd)
{
   brl_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.cs);
}

/* Return true when the VDM was replaced/removed, false when it needs normal
 * graphics finalization (for example, a native background clear). */
bool brl_optimize_empty_vdm(struct brl_cmd_buffer *cmd);

/* Return whether a physical render was finalized (empty renders can vanish). */
static bool
brl_cmd_buffer_end_graphics(struct brl_cmd_buffer *cmd)
{
   brl_tess_batch_flush(cmd);

   struct brl_cs *cs = cmd->current_cs.gfx;

   if (cs && cs->stats.cmds == 0 && brl_optimize_empty_vdm(cmd))
      cs = NULL;

   if (cs) {
      struct brl_device *dev = brl_cmd_buffer_device(cmd);
      const bool full_usc_state = agx_chip_native_agx3(dev->dev.chip);
      /* Scissor and depth bias arrays are staged to dynamic arrays on the
       * CPU. When we end the control stream, they're done growing and are
       * ready for upload. G17P's ISP consumes these as full bit-40-set GART
       * addresses, matching native's USC-window allocations; a main-heap
       * address has bit 39 set and incorrectly selects the upper root.
       */
      cs->uploaded_scissor =
         full_usc_state
            ? brl_pool_usc_upload(cmd, cs->scissor.data, cs->scissor.size, 64)
            : brl_pool_upload(cmd, cs->scissor.data, cs->scissor.size, 64);

      cs->uploaded_zbias =
         full_usc_state
            ? brl_pool_usc_upload(cmd, cs->depth_bias.data,
                                 cs->depth_bias.size, 64)
            : brl_pool_upload(cmd, cs->depth_bias.data,
                             cs->depth_bias.size, 64);

      /* TODO: maybe free scissor/depth_bias now? */
      cs->current = brl_cmd_buffer_emit_vdm_terminate(cmd, cs->current);
   }

   cmd->current_cs.gfx = NULL;
   cmd->state.gfx.render.kept_local_read_barrier = false;

   brl_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.pre_gfx);
   brl_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   assert(cmd->current_cs.pre_gfx == NULL);
   assert(cmd->current_cs.post_gfx == NULL);

   /* We just flushed out the heap use. If we want to use it again, we'll need
    * to queue a free for it again.
    */
   cmd->uses_heap = false;
   return cs != NULL;
}

static inline uint64_t
brl_pipeline_stat_addr(struct brl_cmd_buffer *cmd,
                      VkQueryPipelineStatisticFlagBits stat)
{
   struct brl_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   VkQueryPipelineStatisticFlags flags = root->draw.pipeline_stats_flags;

   if (flags & stat) {
      assert(!cmd->in_meta && "queries paused for meta");
      assert(util_bitcount(stat) == 1 && "by construction");

      uint32_t index;
      const bool present =
         brl_pipeline_statistics_report_index(flags, stat, &index);
      assert(present && "enabled statistic has a compact report index");

      return root->draw.pipeline_stats + (sizeof(uint64_t) * index);
   } else {
      /* Query disabled */
      return AGX_SCRATCH_PAGE_ADDRESS;
   }
}

static inline bool
brl_stat_enabled(uint64_t addr)
{
   return addr != AGX_SCRATCH_PAGE_ADDRESS;
}

/* AGX3 render pass continuation (brl_cmd_draw.c): a pipeline barrier recorded
 * while an ended instance is still continuable decides whether the next one
 * may continue in its stream. */
void brl_m5_pass_merge_barrier(struct brl_cmd_buffer *cmd,
                              const VkDependencyInfo *info);

/* An ordered continuation (brl_cs::m5_ordered) ends at a command that may
 * break its memory equivalence (reason names the command). */
void brl_m5_pass_merge_end_ordered(struct brl_cmd_buffer *cmd,
                                  const char *reason);

/* Graphics descriptor updates while an ordered continuation is open, and
 * bookkeeping of the images graphics push descriptors reference. */
void brl_m5_pass_merge_note_push(struct brl_cmd_buffer *cmd,
                                struct brl_image_view *iview);
void brl_m5_pass_merge_note_set_bind(struct brl_cmd_buffer *cmd);

/* The vertex stage program of record draw_id > 0 of a multi-draw
 * (brl_cmd_buffer.c): only the draw parameters and draw index may differ from
 * the program the previous record of the same multi-draw published. */
uint64_t brl_m5_multi_draw_vertex_program(struct brl_cmd_buffer *cmd,
                                         struct brl_shader *s,
                                         struct brl_linked_shader *linked);

struct brl_agx3_multi_draw_ioto_reloc {
   uint64_t dest[2];
   uint32_t delta;
   uint16_t offset;
   uint8_t short_mask;
   bool is_draw_id;
};

/* Pre-patched vertex-stage Ioto template for a multi-draw indirect loop.
 * Static uniform ranges are patched into `bytes` once; each record only
 * reserves arena space, copies `bytes`, and patches `dyn_relocs`. */
struct brl_agx3_multi_draw_ioto_tpl {
   uint8_t bytes[BRL_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
   bool cached_cmd_mem;
   uint8_t dyn_reloc_count;
   uint32_t draw_params_mask;
   uint32_t draw_id_mask;
   struct brl_agx3_multi_draw_ioto_reloc dyn_relocs[BRL_G17P_PATCH_MAX_RELOCS];
};

bool brl_agx3_multi_draw_ioto_prepare(struct brl_cmd_buffer *cmd,
                                     struct brl_shader *s,
                                     struct brl_linked_shader *linked,
                                     bool uses_base_param, bool uses_draw_id,
                                     struct brl_agx3_multi_draw_ioto_tpl *tpl);

void brl_agx3_multi_draw_ioto_finish(struct brl_cmd_buffer *cmd,
                                    const struct brl_agx3_multi_draw_ioto_tpl *tpl,
                                    uint64_t pipeline_program,
                                    uint64_t draw_params, uint64_t draw_id_ptr);

struct libagx_indirect_ioto_reloc;

/* Prepare a GPU-patchable Ioto vertex-program template and its varying
 * relocations (draw parameters / draw ID) for a multi-draw indirect VDM
 * builder. Returns false when the active vertex launch cannot be patched
 * cleanly across [0, max_draw_count). */
bool brl_agx3_indirect_ioto_prepare(
   struct brl_cmd_buffer *cmd, struct brl_shader *s,
   struct brl_linked_shader *linked, uint64_t in_va, uint32_t stride_B,
   uint32_t param_offset_B, uint64_t draw_ids_va, uint32_t max_draw_count,
   uint64_t *ioto_template_gpu, uint32_t *ioto_copy_dwords,
   uint32_t *ioto_stride_B, struct libagx_indirect_ioto_reloc *relocs,
   uint32_t *reloc_count);

/* Dispatch a 1-thread PREGFX compute kernel to materialize a hoisted AGX3
 * vertex preamble from root_ptr into out_ptr before the render pass runs. */
bool brl_dispatch_preamble_cs(struct brl_cmd_buffer *cmd,
                             struct brl_shader *preamble_cs, uint64_t root_ptr,
                             uint64_t out_ptr);

void brl_cmd_buffer_begin_graphics(struct brl_cmd_buffer *cmd,
                                  const VkCommandBufferBeginInfo *pBeginInfo);
void brl_cmd_buffer_begin_compute(struct brl_cmd_buffer *cmd,
                                 const VkCommandBufferBeginInfo *pBeginInfo);

void brl_cmd_invalidate_graphics_state(struct brl_cmd_buffer *cmd);
void brl_cmd_invalidate_compute_state(struct brl_cmd_buffer *cmd);
/* Non-static for unit tests (brl_cpu_upload_bench). */
void brl_xfer_check_draw_slow(struct brl_cmd_buffer *cmd, uint64_t extra_lo,
                             uint64_t extra_hi);

void brl_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                         const mesa_shader_stage *stages,
                         struct vk_shader **const shaders);

void brl_cmd_bind_graphics_shader(struct brl_cmd_buffer *cmd,
                                 const mesa_shader_stage stage,
                                 struct brl_api_shader *shader);

void brl_cmd_bind_compute_shader(struct brl_cmd_buffer *cmd,
                                struct brl_api_shader *shader);

void brl_cmd_bind_vertex_buffer(struct brl_cmd_buffer *cmd, uint32_t vb_idx,
                               struct brl_addr_range addr_range);

static inline struct brl_descriptor_state *
brl_get_descriptors_state(struct brl_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
      cmd->used_rt = true;
      return &cmd->rt.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void brl_cmd_buffer_flush_push_descriptors(struct brl_cmd_buffer *cmd,
                                          struct brl_descriptor_state *desc);

void brl_meta_resolve_rendering(struct brl_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

void brl_cmd_upload_draw_params(struct brl_cmd_buffer *cmd, uint32_t first,
                               uint32_t first_instance);
void brl_cmd_upload_draw_id(struct brl_cmd_buffer *cmd, uint32_t draw_id);
uint64_t brl_cmd_upload_constants(struct brl_cmd_buffer *cmd,
                                 const struct brl_shader *shader,
                                 mesa_shader_stage stage);
uint64_t brl_cmd_upload_group_count(struct brl_cmd_buffer *cmd,
                                   VkDispatchIndirectCommand group_count);
void brl_cmd_print_upload_stats(struct brl_cmd_buffer *cmd);

uint64_t brl_cmd_buffer_upload_root(struct brl_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);

void brl_reserve_scratch_slow(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                             struct brl_shader *s);

static inline void
brl_reserve_scratch(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                   struct brl_shader *s)
{
   /* Not scratch but this is the most convenient place for this... */
   cs->uses_sampler_heap |= s->b.info.uses_sampler_heap;

   /* G17P and M5 publish a shader's spill frame as thread-private storage in their
    * SPECTPR. There is no helper-program stack, and the kernel rejects a
    * command that names one. */
   if ((s->b.info.scratch_size == 0 && s->b.info.preamble_scratch_size == 0) ||
       agx_chip_native_or_m5(brl_cmd_buffer_device(cmd)->dev.chip))
      return;

   brl_reserve_scratch_slow(cmd, cs, s);
}

/* Whether image barriers of the command buffer order only the vertex work
 * that reads images (brl_sync.h, BRL_SYNC_GFX_IMG). */
static inline bool
brl_sync_image_class(const struct brl_cmd_buffer *cmd)
{
   const enum agx_chip chip = brl_cmd_buffer_device(cmd)->dev.chip;
   return agx_chip_g17p_or_m5(chip);
}

/* AGX2 returns its shader-base-relative USC offset.  G17P returns the absolute
 * 48-bit Ioto program VA consumed by HAL200 VDM state. */
struct agx3_ioto_uniform_range;

bool brl_m5_compute_uniform_dma_ranges(
   struct brl_cmd_buffer *cmd, const struct brl_shader *s, uint64_t root_ptr,
   struct agx3_ioto_uniform_range *ranges, size_t capacity, size_t *count);

int8_t brl_g17p_dma_set_index(const struct brl_shader *s);
uint64_t brl_g17p_dma_set_base(const struct brl_cmd_buffer *cmd,
                              const struct brl_shader *s);

static inline uint64_t
brl_g17p_dma_set_base_from_index(const struct brl_cmd_buffer *cmd, int8_t set_idx)
{
   if (set_idx < 0)
      return 0;

   const uint64_t base = cmd->state.gfx.descriptors.root.sets[set_idx];
   return base ? base : AGX_ZERO_PAGE_ADDRESS;
}

struct agx3_ioto_uniform_range;
bool brl_g17p_dma_ranges(const struct brl_descriptor_state *desc,
                        const struct brl_shader *s, uint64_t root_ptr,
                        struct agx3_ioto_uniform_range *ranges,
                        size_t max_ranges, size_t *count_io,
                        uint8_t *sources);
bool brl_g17p_graphics_uniform_ranges(
   struct brl_cmd_buffer *cmd, struct brl_shader *s,
   const struct brl_linked_shader *linked, uint64_t root_ptr,
   struct agx3_ioto_uniform_range ranges[BRL_AGX3_G17P_GRAPHICS_MAX_RANGES],
   size_t *range_count, uint8_t *sources);

uint64_t brl_upload_native_mesh_program(struct brl_cmd_buffer *cmd,
                                      struct brl_shader *s,
                                      struct brl_linked_shader *linked, bool object,
                                      bool task);
uint64_t brl_upload_usc_words(struct brl_cmd_buffer *cmd, struct brl_shader *s,
                             struct brl_linked_shader *linked);
uint64_t brl_agx3_try_push_only_vs_program(struct brl_cmd_buffer *cmd,
                                          struct brl_shader *s,
                                          struct brl_linked_shader *linked);

void brl_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                                    struct brl_cmd_buffer *cmd);

void brl_cdm_cache_flush(struct brl_device *dev, struct brl_cs *cs);

/* Initializes the metadata of the compressed storage images a dependency
 * moves out of UNDEFINED, before a shader store can meet stale metadata: by
 * clearing the discarded subresources as their layout transition
 * (brl_sync_transition_begin()). This depends only on the image being
 * compressed for storage, whatever the chip. brl_discard_init_prepare()
 * returns whether the dependency has such an image and takes its first scope;
 * call it before the dependency's brl_sync_barrier(), and
 * brl_discard_init_record() with the returned second scope after it. */
bool brl_discard_init_prepare(struct brl_cmd_buffer *cmd,
                              const VkDependencyInfo *info,
                              uint8_t *dst_kinds);
void brl_discard_init_record(struct brl_cmd_buffer *cmd,
                             const VkDependencyInfo *info, uint8_t dst_kinds);

/* As brl_cdm_cache_flush(), for a flush an application dependency asks for.
 * With `memory`, the flush also has memory scope on G17P and M5: needed for
 * host availability, and for device dependencies once the same memory may be
 * reached through more than one virtual address (brl_cdm_alias_scope()). A
 * trailing strong flush without memory scope is upgraded in place. The caller
 * ensures space for one barrier record. */
void brl_cdm_visibility_flush(struct brl_device *dev, struct brl_cs *cs,
                             bool memory);

/* The AGX3 strong flush record brl_cdm_visibility_flush(dev, cs, memory)
 * appends, for a stream the driver builds outside a command buffer. False
 * where the chip has no grounded encoding. */
bool brl_cdm_pack_visibility_flush(const struct brl_device *dev, bool memory,
                                  uint8_t packed[AGX3_CDM_BARRIER_SIZE]);

/* Whether brl_cdm_visibility_flush(dev, cs, memory) would record anything. */
bool brl_cdm_visibility_flush_needed(const struct brl_device *dev,
                                    const struct brl_cs *cs, bool memory);

void brl_dispatch_with_usc_launch(struct brl_device *dev, struct brl_cs *cs,
                                 struct agx_cdm_launch_word_0_packed launch,
                                 uint32_t usc, struct agx_grid grid,
                                 struct agx_workgroup local_size);

void brl_dispatch_with_usc(struct brl_device *dev, struct brl_cs *cs,
                          struct agx_shader_info *info, uint32_t usc,
                          struct agx_grid grid,
                          struct agx_workgroup local_size);

void brl_dispatch_m5_native(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                            struct brl_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size);

void brl_dispatch_g17p_native(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                             struct brl_shader *s, struct agx_grid grid,
                             struct agx_workgroup local_size);
void brl_dispatch_g17g_native(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                             struct brl_shader *s, struct agx_grid grid,
                             struct agx_workgroup local_size);

bool brl_gen1_upload_fragment_query_state(struct brl_cmd_buffer *cmd, bool vertex_stage,
   struct agx3_ioto_uniform_range *range);
bool brl_gen1_upload_rq_uniforms(struct brl_cmd_buffer *cmd, uint32_t *context,
                                unsigned context_words, bool profile,
                                struct agx3_ioto_uniform_range *range);

void brl_g17p_prewarm_linked_compute_shader(struct brl_device *dev,
                                            struct brl_shader *s,
                                            struct brl_linked_shader *linked);
void brl_g17p_linked_compute_cache_finish(struct brl_linked_shader *linked);
void brl_g17p_linked_graphics_cache_finish(struct brl_device *dev,
                                          struct brl_linked_shader *linked);

/*
 * Saved application state around a driver-internal meta operation.  The
 * acceleration-structure builder needs this as much as a blit does: the
 * common build shaders bind their own compute pipeline and push constants,
 * so a build recorded between two application dispatches would otherwise
 * silently clobber them.
 */
struct brl_meta_save {
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
   struct vk_dynamic_graphics_state dynamic;
   struct brl_api_shader *shaders[MESA_SHADER_MESH + 1];
   struct brl_addr_range vb0;
   struct brl_descriptor_set *desc0;
   uint32_t desc0_update_seq;
   /* Nested meta must retain the outer query/conditional-rendering scope. */
   bool in_meta;
   bool has_push_desc0;
   bool push_active0;
   /* Set 0 bound from a descriptor buffer */
   bool buffer_active0;
   uint64_t buffer_set0;
   uint32_t buffer_set_size0;
   enum agx_visibility_mode occlusion;
   /* Where in brl_cmd_buffer::meta_push_saves the saved push set 0 starts */
   uint32_t push_desc0_offset;
   VkQueryPipelineStatisticFlags pipeline_stats_flags;
   uint8_t push[BRL_MAX_PUSH_SIZE];
   /* The runtime's record of stages bound from a pipeline. */
   VkShaderStageFlags pipeline_shader_stages;
};

void brl_meta_begin(struct brl_cmd_buffer *cmd, struct brl_meta_save *save,
                   VkPipelineBindPoint bind_point);
void brl_meta_end(struct brl_cmd_buffer *cmd, struct brl_meta_save *save,
                 VkPipelineBindPoint bind_point);

void brl_cmd_dispatch_rays(struct brl_cmd_buffer *cmd,
                          struct brl_api_shader *shader, uint64_t record_addr,
                          uint64_t group_count_addr, struct agx_grid grid);

void brl_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer,
                               uint32_t invocations_x, uint32_t invocations_y,
                               uint32_t invocations_z);

/* Driver-owned, resident indirect arguments addressed through their primary
 * GPU VA. Sparse application buffers must use the public entrypoint. */
void brl_cmd_dispatch_indirect_addr(VkCommandBuffer command_buffer,
                                   VkDeviceAddress address);

static inline void
brl_dispatch_with_local_size(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                            struct brl_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size)
{
   if (agx_is_shader_empty(&s->b))
      return;

   struct brl_device *dev = brl_cmd_buffer_device(cmd);
   if (dev->dev.chip == AGX_CHIP_G18G) {
      brl_dispatch_g17g_native(cmd, cs, s, grid, local_size);
      return;
   }
   if (agx_chip_g17_m5(dev->dev.chip)) {
      brl_dispatch_m5_native(cmd, cs, s, grid, local_size);
      return;
   }
   if (agx_chip_native_agx3(dev->dev.chip)) {
      brl_reserve_scratch(cmd, cs, s);
      brl_dispatch_g17p_native(cmd, cs, s, grid, local_size);
      /* Software graphics stages communicate through memory: VS outputs,
       * geometry counts and XFB bases are consumed by the next dispatch.
       * The application cannot supply barriers between these internal
       * stages. Preserve the AGX2 flush-after-launch contract, including
       * indirect launches, by strengthening the native ordering barrier. */
      if (s->info.stage != MESA_SHADER_COMPUTE) {
         brl_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
         brl_cdm_cache_flush(dev, cs);
      }
      return;
   }

   uint64_t uploaded = brl_upload_usc_words(cmd, s, s->only_linked);
   if (uploaded > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t usc = uploaded;

   brl_reserve_scratch(cmd, cs, s);
   brl_dispatch_with_usc(dev, cs, &s->b.info, usc, grid, local_size);
}

void brl_dispatch_precomp(struct brl_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         void *data, size_t data_size);
void brl_dispatch_precomp_m5(struct brl_cmd_buffer *cmd, struct agx_grid grid,
                            enum agx_barrier barrier, enum libagx_program idx,
                            const void *data, size_t data_size);
bool brl_g18_emit_indirect_launch(
   struct brl_cmd_buffer *cmd, struct brl_cs **target,
   const struct agx3_compute_program *launch, struct agx_grid grid,
   struct agx_workgroup local_size,
   const struct drm_asahi_g18_compute_resources *target_res);
void brl_dispatch_precomp_g17p(struct brl_cmd_buffer *cmd, struct agx_grid grid,
                              enum agx_barrier barrier,
                              enum libagx_program idx, const void *data,
                              size_t data_size);

/* Turn a three/four/five-word indirect argument record into an execution-time
 * conditional record. `barrier` selects the ordered compute stream which must
 * precede its consumer. */
uint64_t brl_apply_conditional_rendering(struct brl_cmd_buffer *cmd,
                                        uint64_t source, uint32_t words,
                                        enum agx_barrier barrier);

#define MESA_DISPATCH_PRECOMP brl_dispatch_precomp

void brl_queue_write(struct brl_cmd_buffer *cmd, uint64_t address, uint32_t value,
                    bool after_gfx);
/* Launch libagx_copy_timestamps for count copies in cs, the open
 * timestamp-only compute stream, which is ordered after the stamps they
 * read. */
void brl_record_stamp_copies(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                            const struct brl_stamp_copy *copies,
                            unsigned count);
bool brl_ensure_occlusion_query_for_draw(struct brl_cmd_buffer *cmd);

/* Dispatch the native G17P raw-address fill meta shader into an explicitly
 * selected compute stream.  The target may be the ordinary or post-graphics
 * stream; the latter preserves query-availability ordering. */
void brl_g17p_meta_fill_addr(struct brl_cmd_buffer *cmd, struct brl_cs **target,
                            uint64_t address, uint64_t size, uint32_t value);

/* Poll a naturally aligned 32-bit device address from a native G17P compute
 * stream until it reaches value. */
void brl_g17p_meta_wait_addr(struct brl_cmd_buffer *cmd, uint64_t address,
                            uint32_t value);

void agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                         uint64_t attribs_read, struct agx_velem_key *keys);

bool brl_needs_index_robustness(struct brl_cmd_buffer *cmd,
                               struct agx_draw *draw,
                               const struct agx_draw *conditional_direct);

bool brl_robust_index_rewrite_admitted(const struct brl_device *dev,
                                      struct agx_draw draw,
                                      enum agx_primitive topology);

struct agx_robustness brl_prolog_robustness(struct brl_device *dev);

bool brl_attachmentless_render(const struct brl_cs *cs);

void brl_attachmentless_vdm_bounds(const struct brl_cs *cs, uint32_t *width_px,
                                  uint32_t *height_px, uint32_t *layers);
