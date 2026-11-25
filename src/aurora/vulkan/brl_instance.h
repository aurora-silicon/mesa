/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "brl_drirc.h"
#include "vk_instance.h"

struct brl_instance {
   struct vk_instance vk;

   struct brl_drirc drirc;

   uint8_t driver_build_sha[BLAKE3_KEY_LEN];
};

VK_DEFINE_HANDLE_CASTS(brl_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
