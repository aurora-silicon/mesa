/*
 * Copyright 2026 Dylan Ironside
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "nir.h"

/* Whether `use` of a value the host evaluates for a G16 draw reads it in a
 * way that does not depend on the host matching the GPU's NaN payloads and
 * signed zeros bit for bit. Needs divergence information. */
bool hk_g16_host_float_use(const nir_src *use);
