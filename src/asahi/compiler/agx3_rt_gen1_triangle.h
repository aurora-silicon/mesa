#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx3_rt_gen1_entry.h"
/* Constant-table word holding (queue capacity - burst size) in count units. */
#define AGX3_RT_GEN1_TRIANGLE_QUEUE_RESERVE_WORD 4u
#define AGX3_RT_GEN1_TRIANGLE_SLOT 101u
bool agx3_rt_gen1_triangle_body_generate(uint8_t *out, size_t capacity, size_t *length);
void agx3_rt_gen1_triangle_constants(const uint8_t **data, size_t *length);
bool agx3_rt_gen1_triangle_entry_generate(uint64_t slot_body_va, uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE], size_t *length);
