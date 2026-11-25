/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "aurora/lib/agx_chip.h"
#include "util/u_dynarray.h"
#include "brl_pack_loop_bound.h"
#include "brl_sync.h"

struct agx_bo;

struct brl_cmd_buffer;
struct brl_cs;
struct brl_device;
struct brl_sync_queue;

/* Submit packing (brl_queue.c, brl_pack_runs()): a run of consecutive command
 * buffers that each record one plain compute stream is submitted as one
 * command. These are the parts that decide what a packet may hold and how
 * its dependencies are recorded; they touch no device or kernel state.
 *
 * A packet covers at most BRL_PACK_MAX_DISPATCHES launches and the thread
 * and static work bounds; the measured 165-, 200- and 242-dispatch image and
 * compute graphs fit. A program with loops runs for longer than its static
 * work says, so a packet holds at most one stream with a looping program.
 * The static work bound is 2^35 thread-instructions, about 18 ms at the M4's
 * ALU issue rate; memory-bound programs run slower per instruction, so for
 * them the bound limits a packet's length only roughly.
 */
#define BRL_PACK_MAX_DISPATCHES 256u
#define BRL_PACK_MAX_THREADS (UINT64_C(1) << 28)
#define BRL_PACK_MAX_STATIC_WORK (UINT64_C(1) << 35)
#define BRL_PACK_CDM_TERMINATE_MAX 8u

/* One command buffer of a submission, as the planner sees it. */
struct brl_pack_member {
   /* Its single compute stream, or NULL: it stays an ordinary command. */
   const struct brl_cs *cs;
   /* Stream bytes before the terminate. */
   size_t bytes;
   /* Synchronization kinds of the stream. */
   uint8_t kinds;
   /* Render (VDM) kinds the stream itself waits for, and per stream kind k
    * those the command buffer's tail makes later streams of kind k wait for
    * (req[k][VDM]), both only of kinds that resolve to a render wait before
    * this command buffer (brl_sync_queue_kinds(), brl_sync_cmd_kinds()): other
    * kinds are no wait. */
   uint8_t render_waits;
   uint8_t render_tail[BRL_SYNC_NUM_KINDS];
   /* The render subqueue ran work before this command buffer. */
   bool render_ran;
};

struct brl_pack_limits {
   /* Compute attachments one command may carry. */
   uint32_t max_attachments;
   /* Bytes a packet needs beyond its members' streams. */
   size_t overhead;
};

struct brl_pack_run {
   unsigned first, count;
   size_t offset, bytes;
};

/* How a member's end publishes its writes to the members after it, which
 * decides whether a device packs at all. */
enum brl_pack_boundary {
   /* No boundary is defined: the device does not pack. */
   BRL_PACK_BOUNDARY_NONE,
   /* The kernel takes each stream's own completion flush, the strong
    * memory-scoped flush (DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION, M5
    * streams): a member ends with a strong flush, a boundary keeps the
    * memory scope only where memory may have a second address
    * (brl_cdm_alias_scope()), and the packet's end gets it. */
   BRL_PACK_BOUNDARY_VISIBLE_COMPLETION,
   /* HAL200 streams whose strong flush has no memory-scoped form (G16): the
    * kernel publishes a command's writes when it completes, and a boundary
    * makes the member's last ordering record the strong flush, the record an
    * in-stream compute dependency gets. */
   BRL_PACK_BOUNDARY_STRONG_FLUSH,
};

enum brl_pack_boundary brl_pack_boundary(const struct brl_device *dev);

/* Packs the CDM stream-terminate record for `chip` into `out` and returns its
 * byte length (8 for AGX2, 4 for AGX3), or 0 if unsupported. */
size_t brl_pack_cdm_terminate(enum agx_chip chip,
                             uint8_t out[BRL_PACK_CDM_TERMINATE_MAX]);

bool brl_pack_account_dispatch_work(uint64_t threads, uint64_t instrs,
                                   bool has_loops, uint64_t loop_bound,
                                   uint64_t *work_out);

bool brl_pack_account_dispatch_work_eff(uint64_t threads, uint64_t instrs,
                                       bool has_loops, uint64_t eff_instrs,
                                       uint64_t *work_out);

bool brl_pack_fusion_fits_run(uint64_t run_work, unsigned run_loops,
                             uint64_t replaced_work, unsigned replaced_loops,
                             uint64_t fused_work, bool fused_loops,
                             uint64_t *new_work_out, unsigned *new_loops_out);

/* Whether a member may end with a strong flush without memory scope, as an
 * application barrier at the end of a command buffer leaves it: at every
 * boundary. The HAL200 boundary upgrades it in place; at the
 * visible-completion boundary the packet's end carries the memory-scoped
 * flush the kernel takes as its completion (brl_pack_runs()).
 * brl_pack_run_member() takes the answer from here, and the packing trace
 * reports it (strong_tails=), so tests know which barrier-ended command
 * buffers must pack. */
bool brl_pack_admits_strong_tail(enum brl_pack_boundary boundary);

/* Whether streams with global atomics may be packed, before the check of the
 * device's other kernel queues: with `serial_kernel` (G16G, whose kernel
 * runs a VM's compute commands one after another on one firmware queue) by
 * default and unless HK_PACK_ATOMICS (`opt`) says no; otherwise only when
 * `opt` says yes and the queue has a single kernel queue (`queues`). `opt`
 * takes debug_parse_bool_option()'s values plus "on" and "off"; any other
 * value keeps the default. */
bool brl_pack_atomics_choice(bool serial_kernel, const char *opt,
                            unsigned queues);

/* Whether `cs` is a clean single-dispatch compute stream eligible to
 * participate as producer or consumer in submit-time compute pair fusion. */
bool brl_compute_pair_stream_eligible(const struct brl_cs *cs);

/* The single compute stream of `cmd` that a packet may copy, or NULL. */
const struct brl_cs *brl_pack_run_member(const struct brl_device *dev,
                                       const struct brl_cmd_buffer *cmd,
                                       const uint8_t *terminate, bool atomics);

/* Fills members[0..n) for the command buffers of a submission to the queue
 * whose synchronization state is `sync`, as brl_pack_plan() takes them, and
 * returns how many may be packed. */
unsigned brl_pack_describe_submit(struct brl_pack_member *members,
                                 const struct brl_device *dev,
                                 const struct brl_sync_queue *sync,
                                 struct brl_cmd_buffer *const *cmds, unsigned n,
                                 const uint8_t *terminate, bool atomics);

/* Groups members into runs of two or more: runs[] gets them in order, with
 * byte offsets in one buffer of *total bytes. Returns the run count. */
unsigned brl_pack_plan(const struct brl_pack_member *members, unsigned n,
                      const struct brl_pack_limits *limits,
                      struct brl_pack_run *runs, unsigned *packed_members,
                      size_t *total);

/* Initializes a reusable packet's command buffer and compute control stream,
 * resetting only the compute-relevant state and preserving dynarray backing
 * storage across submissions. */
void brl_pack_init_packet(struct brl_cmd_buffer *cmd, struct brl_cs *cs,
                         struct brl_device *dev, uint8_t *start, uint64_t addr,
                         size_t bytes, bool visible_completion);

/* Records member `item` (the stream of `source`) in the packet's
 * synchronization: what it waits for, what it makes later command buffers
 * wait for, and, unless it is the last member, its tail as a requirement of
 * the members after it. `preceding_kinds` accumulates over the members. */
void brl_pack_fold_member(struct brl_cmd_buffer *packet, struct brl_cs *cs,
                         const struct brl_cmd_buffer *source,
                         const struct brl_cs *item, bool last,
                         uint8_t *preceding_kinds);

/* The packet's stream is its command buffer's latest of every kind. */
void brl_pack_finish_sync(struct brl_cmd_buffer *packet, struct brl_cs *cs);

/* Adds `bo` once to `bos`, the buffers a packed submission keeps alive until
 * it completes (brl_cmd_pending_begin()). A NULL `bo` adds nothing. Returns
 * false if the list cannot grow. */
bool brl_pack_add_bo(struct util_dynarray *bos, struct agx_bo *bo);

#define BRL_PACK_FOOTPRINT_MAX_READS 8u
#define BRL_PACK_FOOTPRINT_MAX_WRITES 6u
#define BRL_PACK_HAZARD_MAX_INTERVALS 64u
#define BRL_PACK_LIVENESS_MAX_INTERVALS 64u

struct brl_pack_interval {
   uint64_t start;
   uint64_t end;
   uint32_t handle;
};

struct brl_pack_footprint {
   struct brl_pack_interval reads[BRL_PACK_FOOTPRINT_MAX_READS];
   struct brl_pack_interval writes[BRL_PACK_FOOTPRINT_MAX_WRITES];
   uint8_t num_reads;
   uint8_t num_writes;
   bool known;
};

enum brl_pack_boundary_action {
   BRL_PACK_BOUNDARY_ELIDE_BARRIER,
   BRL_PACK_BOUNDARY_KEEP_WEAK,
   BRL_PACK_BOUNDARY_STRONG,
};

struct brl_pack_hazard_state {
   struct brl_pack_interval dirty_writes[BRL_PACK_HAZARD_MAX_INTERVALS];
   struct brl_pack_interval dirty_reads[BRL_PACK_HAZARD_MAX_INTERVALS];
   uint8_t num_dirty_writes;
   uint8_t num_dirty_reads;
   bool write_unknown;
   bool read_unknown;
};

void brl_pack_hazard_reset_all(struct brl_pack_hazard_state *st);
void brl_pack_hazard_record(struct brl_pack_hazard_state *st,
                           const struct brl_pack_footprint *fp,
                           bool strong_flushed);
enum brl_pack_boundary_action
brl_pack_hazard_decide(struct brl_pack_hazard_state *st,
                      const struct brl_pack_footprint *next_fp,
                      bool can_elide_weak);

/* Backward dead-interval tracker within a single packed run.
 * Every interval in `dead[]` is guaranteed to be completely overwritten by
 * later members of the same run before any read or end-of-run; coalesced
 * disjoint intervals per handle. Fails closed (empties `dead[]`) on any
 * unknown footprint or invalid read. */
struct brl_pack_liveness_state {
   struct brl_pack_interval dead[BRL_PACK_LIVENESS_MAX_INTERVALS];
   uint8_t num_dead;
};

void brl_pack_liveness_reset(struct brl_pack_liveness_state *st);
bool brl_pack_liveness_covers(const struct brl_pack_liveness_state *st,
                             const struct brl_pack_interval *iv);
void brl_pack_liveness_add_overwrite(struct brl_pack_liveness_state *st,
                                    const struct brl_pack_interval *iv);
void brl_pack_liveness_kill_read(struct brl_pack_liveness_state *st,
                                const struct brl_pack_interval *iv);
void brl_pack_liveness_step_backward(struct brl_pack_liveness_state *st,
                                    const struct brl_pack_footprint *fp);
