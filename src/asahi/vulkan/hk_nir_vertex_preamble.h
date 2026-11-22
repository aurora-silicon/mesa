/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_NIR_VERTEX_PREAMBLE_H
#define HK_NIR_VERTEX_PREAMBLE_H

#include "nir.h"

/* A native ray query owns uniform halves 0..167 of the program that runs it;
 * the vertex ABI and the indexed texture/sampler heaps of such a vertex
 * program start here. */
#define HK_NATIVE_QUERY_STATE_HALFS 168

/* Move a vertex program's preamble loads up by halves, making room for the
 * indexed texture/sampler heap pointers in front of the vertex ABI that the
 * monolithic VBO lowering may already have materialized. With native_query,
 * the query state below HK_NATIVE_QUERY_STATE_HALFS stays at the fixed halves
 * the driver publishes it to and only the vertex ABI above it moves. Returns
 * progress.
 */
bool hk_nir_shift_vertex_preamble(nir_shader *nir, unsigned halves,
                                  bool native_query);

#endif
