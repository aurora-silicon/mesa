/* SPDX-License-Identifier: MIT */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "asahi/lib/agx_chip.h"

/* intersect_ray operands: a constant AS uses f63/f69=3; an AS selected per
 * invocation uses f62/f69=2 and f73 marks its last use. The first operand is
 * a GPR pair. All register indices are in half-register units. */
struct agx3_rt_m5_intersect_desc {
   uint16_t address_register;
   uint16_t bvh_base;
   bool bvh_uniform;
   bool bvh_kill;
};

/* N simultaneously live queries use consecutive allocation indices
 * 4..4+N-1. */
#define AGX3_RT_M5_MAX_LIVE_QUERIES 8

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
