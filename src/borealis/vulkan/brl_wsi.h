/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "brl_physical_device.h"

VkResult brl_init_wsi(struct brl_physical_device *pdev);
void brl_finish_wsi(struct brl_physical_device *pdev);

VkResult brl_wsi_allocate_memory(VkDevice device, const VkMemoryAllocateInfo *info,
   const VkAllocationCallbacks *alloc, VkDeviceMemory *out);
