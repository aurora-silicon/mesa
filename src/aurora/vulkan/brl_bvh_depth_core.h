/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Depth analysis and depth-limiting for an encoded Borealis BVH.
 *
 * Encoded output nodes are topologically ordered (parent index < child
 * indices), so depth and subtree height/leaf counts compute in single forward
 * and backward sweeps over caller-provided scratch without recursion.
 */

#ifndef BRL_BVH_DEPTH_CORE_H
#define BRL_BVH_DEPTH_CORE_H

#include "brl_bvh.h"

#include <stdbool.h>
#include <stdint.h>

#endif /* BRL_BVH_DEPTH_CORE_H */
