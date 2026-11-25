/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef BRL_ACCELERATION_STRUCTURE_H
#define BRL_ACCELERATION_STRUCTURE_H

#include "brl_agx3_as_format.h"
#include "brl_bvh.h"

#include "vk_acceleration_structure.h"

struct brl_device;
struct brl_cmd_buffer;
struct agx_scs_layout_gen2;
struct vk_buffer;
bool brl_gen1_publish_buffer_as_addresses(struct brl_device *dev,
                                         const struct vk_buffer *buffer);
uint64_t brl_gen1_upload_as_table(struct brl_cmd_buffer *cmd);
uint64_t brl_gen1_as_metadata_addr(struct brl_cmd_buffer *cmd,
                                  struct vk_acceleration_structure *as);
struct brl_agx3_plan_args;

void brl_gen1_rq_finish(struct brl_device *dev);
VkResult brl_device_init_accel_struct_state(struct brl_device *device);
void brl_device_finish_accel_struct_state(struct brl_device *device);
void brl_m5_as_buffer_bound(struct brl_device *dev, struct vk_buffer *buffer);

/* Plans, encodes and finalizes one triangle AS from common build IR, with
 * internal phase barriers and a GPU-written indirect node count. */
VkResult brl_cmd_build_agx3_from_ir(VkCommandBuffer command_buffer,
                                 const struct brl_agx3_plan_args *args,
                                 VkDeviceAddress node_status,
                                 VkDeviceAddress envelope_status,
                                 uint32_t slot_bias);

/* The SPIR-V that the node encoding and refit pipelines of each native AS
 * format are created from; brl_agx3_as_format_for_chip() selects the format.
 * GEN1 also has a triangle-pair encoder, M5 a batched one. */
const uint32_t *brl_agx3_node_encoder_spirv(enum agx_chip chip, size_t *size_B);
const uint32_t *brl_agx3_refit_spirv(enum agx_chip chip, size_t *size_B);
const uint32_t *brl_gen1_agx3_node_encoder_spirv(bool pairs, size_t *size_B);
const uint32_t *brl_gen1_agx3_refit_spirv(size_t *size_B);
const uint32_t *brl_gen1_agx3_refit_top_spirv(size_t *size_B);
const uint32_t *brl_gen1_agx3_refit_encode_direct_spirv(size_t *size_B);
const uint32_t *brl_m5_agx3_node_encoder_spirv(size_t *size_B);
const uint32_t *brl_m5_agx3_refit_spirv(size_t *size_B);

uint64_t brl_gen1_ray_query_scs_layout(const struct brl_device *dev);
uint64_t brl_gen1_query_state(struct brl_cmd_buffer *cmd,
                            const struct agx_scs_layout_gen2 *scs);

#endif /* BRL_ACCELERATION_STRUCTURE_H */
