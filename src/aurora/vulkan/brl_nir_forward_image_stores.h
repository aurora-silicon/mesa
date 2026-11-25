/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

struct nir_shader;

bool brl_nir_forward_image_stores(struct nir_shader *nir, bool robust_images);
