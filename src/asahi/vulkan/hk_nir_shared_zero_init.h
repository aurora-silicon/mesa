/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>

struct nir_shader;

/* Return true when every dword of [0, nir->info.shared_size) is proven to be
 * unconditionally written by the workgroup before any shared read, atomic, or
 * early exit, and before a dominating top-level workgroup shared barrier. In
 * that case VK_KHR_zero_initialize_workgroup_memory's zero-fill and initial
 * barrier are dead and may be skipped. */
bool hk_nir_shared_memory_fully_initialized(const struct nir_shader *nir);
