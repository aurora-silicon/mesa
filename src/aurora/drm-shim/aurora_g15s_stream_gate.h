/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "drm-uapi/aurora_drm.h"

/*
 * Host mirror of the T6030 (M3 Pro, G15S) kernel's submission gates.
 *
 * The M3 kernel refuses some packets and command-stream records outright and
 * the G15 firmware turns others into GPU faults, which on that machine are
 * fail-stop until reboot. Reproducing the kernel's checks here lets a G15S
 * command stream be qualified on the host before it ever reaches hardware.
 */

struct aurora_g15s_gate_memory {
   void *data;
   bool (*read)(void *data, uint64_t address, void *dst, size_t size);
   bool verbose;
};

struct aurora_g15s_gate_stats {
   unsigned renders;
   unsigned computes;
   unsigned launches;
   unsigned indirect_launches;
   unsigned barriers;
   unsigned wide_barriers;
   unsigned links;
};

/* Kernel payload rules for one command. features is the advertised feature
 * word; size is header.size. Returns false and prints the reason on
 * refusal. */
bool aurora_g15s_render_packet_validate(const struct drm_aurora_cmd_render *render,
                                       const void *payload, size_t size,
                                       uint64_t features);
bool aurora_g15s_compute_packet_validate(
   const struct drm_aurora_cmd_compute *compute, const void *payload,
   size_t size, uint64_t features);

/* Kernel stream_gate: the VDM root's first word and the CDM record walk.
 * usc_base is the queue's USC window base; every program must lie in its
 * 4 GiB window. */
bool aurora_g15s_vdm_gate(const struct aurora_g15s_gate_memory *memory,
                         uint64_t root);
bool aurora_g15s_cdm_gate(const struct aurora_g15s_gate_memory *memory,
                         uint64_t usc_base, uint64_t base,
                         struct aurora_g15s_gate_stats *stats);
