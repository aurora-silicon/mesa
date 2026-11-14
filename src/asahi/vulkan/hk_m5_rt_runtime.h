/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The M5 (G17S) intersection runtime that native ray queries call through the
 * intersection function table, generated into hk_m5_rt_runtime.c.
 *
 * The code image is a stream of rows. A row is one instruction: a shape (a
 * form of agx3c_table and the fields it sets) and the field values. The device
 * composes every row with agx3_compose() at its offset. Bytes the table does
 * not reproduce are raw rows, and zero runs are skipped. The user image is raw
 * data in the same stream format. Both streams are packed with a small LZ77
 * (hk_m5_rt_lz_decode()).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define HK_M5_RT_IMAGE_SIZE 0x800000u

struct hk_m5_rt_shape {
   uint16_t form;
   uint8_t nfields;
   uint32_t field_off; /* into hk_m5_rt_field_ids */
};

extern const struct hk_m5_rt_shape hk_m5_rt_shapes[];
extern const uint32_t hk_m5_rt_shape_count;
extern const uint8_t hk_m5_rt_field_ids[];
extern const uint32_t hk_m5_rt_field_id_count;
extern const uint8_t *const hk_m5_rt_code_lz;
extern const uint32_t hk_m5_rt_code_lz_size, hk_m5_rt_code_stream_size;
extern const uint8_t *const hk_m5_rt_user_lz;
extern const uint32_t hk_m5_rt_user_lz_size, hk_m5_rt_user_stream_size;
/* FNV-1a 64 of the images the streams must build. */
extern const uint64_t hk_m5_rt_code_digest, hk_m5_rt_user_digest;
extern const uint32_t hk_m5_rt_uniforms[74];
extern const uint32_t hk_m5_rt_instance_uniforms[80];

/* Fill dst (HK_M5_RT_IMAGE_SIZE bytes) with the code image, composed row by
 * row; false if a row fails to compose, the stream is malformed or the image
 * does not match hk_m5_rt_code_digest. */
bool hk_m5_rt_build_code(uint8_t *dst);

/* Fill dst (HK_M5_RT_IMAGE_SIZE bytes) with the user image; false as above,
 * against hk_m5_rt_user_digest. */
bool hk_m5_rt_build_user_image(uint8_t *dst);
