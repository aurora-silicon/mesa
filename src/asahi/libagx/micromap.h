/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef LIBAGX_MICROMAP_H
#define LIBAGX_MICROMAP_H

#include <stdint.h>

/* Opacity micromap image (VkMicromapEXT storage). It is position
 * independent, so clone, compact, serialize and deserialize copy its words.
 *
 *  - header, LIBAGX_MICROMAP_HEADER_BYTES: the words below;
 *  - one 8-byte record per triangle at LIBAGX_MICROMAP_RECORDS: the word
 *    offset of its data from the image start, then its subdivision level
 *    (low 16 bits) and format (high 16 bits, 0 for an invalid triangle);
 *  - data: one bucket per (format, level). A triangle's data is its
 *    application bytes padded to whole words, so lookups read 32-bit words.
 */
#define LIBAGX_MICROMAP_MAGIC 0x4d4f4b48u /* "HKOM" */
#define LIBAGX_MICROMAP_VERSION 1u
#define LIBAGX_MICROMAP_HEADER_BYTES 256u
#define LIBAGX_MICROMAP_RECORDS LIBAGX_MICROMAP_HEADER_BYTES
#define LIBAGX_MICROMAP_RECORD_BYTES 8u

/* Header words. */
#define LIBAGX_MICROMAP_WORD_MAGIC 0u
#define LIBAGX_MICROMAP_WORD_VERSION 1u
#define LIBAGX_MICROMAP_WORD_TRIANGLES 2u
#define LIBAGX_MICROMAP_WORD_BYTES_LO 4u
#define LIBAGX_MICROMAP_WORD_BYTES_HI 5u
#define LIBAGX_MICROMAP_WORD_DATA 6u

/* Highest subdivision level any micromap can hold. The advertised limits
 * may be lower. */
#define LIBAGX_MICROMAP_MAX_LEVEL 15u
#define LIBAGX_MICROMAP_BUCKETS (2u * (LIBAGX_MICROMAP_MAX_LEVEL + 1u))

/* Serialized micromaps: driverUUID, compatibility UUID, then the image. The
 * compatibility UUID names this image format. */
#define LIBAGX_MICROMAP_SERIAL_HEADER_BYTES 32u
#define LIBAGX_MICROMAP_SERIAL_FORMAT_UUID_BYTES \
   {0x48,0x4b,0x4f,0x4d,0x7a,0x31,0x4e,0x2e, \
    0x9c,0x58,0x0b,0x6e,0x6d,0x6d,0x00,0x01}

/* One bucket of a build: where its slots start, how many words each takes,
 * its first entry in the scratch slot-to-triangle map, and its capacity. */
struct libagx_micromap_bucket {
   uint32_t data_word;
   uint32_t slot_words;
   uint32_t first_slot;
   uint32_t count;
};

/* Scratch: a counter per bucket, then the slot-to-triangle map. */
#define LIBAGX_MICROMAP_SCRATCH_SLOTS (LIBAGX_MICROMAP_BUCKETS * 4u)

static inline uint32_t
libagx_micromap_bucket_index(uint32_t format, uint32_t level)
{
   return (format - 1u) * (LIBAGX_MICROMAP_MAX_LEVEL + 1u) + level;
}

/* Bytes of one triangle's data: 4^level elements of 1 (2-state) or
 * 2 (4-state) bits, rounded up to whole bytes. */
static inline uint64_t
libagx_micromap_data_bytes(uint32_t format, uint32_t level)
{
   return ((UINT64_C(1) << (2u * level)) * format + 7u) / 8u;
}

/* Acceleration-structure OMM table, after a triangle BLAS's native image:
 *
 *  - header, LIBAGX_OMM_TABLE_HEADER_BYTES: magic, geometry count, primitive
 *    count;
 *  - one LIBAGX_OMM_TABLE_GEOMETRY_BYTES entry per geometry: the micromap
 *    address (0 for a null handle), the byte offset of the geometry's index
 *    array from the table, and flags (LIBAGX_OMM_GEOMETRY_LINKED when the
 *    geometry has opacity micromap linkage);
 *  - per primitive, its micromap triangle index with baseTriangle applied,
 *    or a VkOpacityMicromapSpecialIndexEXT value (-1 .. -4) unchanged.
 *
 * Traversal finds the table through the OMM directory: one 16-byte entry
 * per AS id holding the table address (0 for none), geometry count and
 * primitive count.
 */
#define LIBAGX_OMM_TABLE_MAGIC 0x544f4b48u /* "HKOT" */
#define LIBAGX_OMM_TABLE_HEADER_BYTES 16u
#define LIBAGX_OMM_TABLE_GEOMETRY_BYTES 16u
#define LIBAGX_OMM_GEOMETRY_LINKED 1u
#define LIBAGX_OMM_DIRECTORY_ENTRY_BYTES 16u

/* A TLAS's directory entry has no table. Each build or update of the TLAS
 * writes a fresh stamp (LIBAGX_OMM_TLAS_STAMP_BIT set) to word 2, and to
 * word 3 too when an instance not disabling micromaps references a BLAS with
 * an OMM table. Queries on a TLAS whose words 2 and 3 differ skip micromap
 * lookups and keep hardware commits. */
#define LIBAGX_OMM_TLAS_STAMP_BIT 0x80000000u
/* VkGeometryInstanceFlagBitsKHR in a native instance record's flags word. */
#define LIBAGX_OMM_INSTANCE_DISABLE 0x20u

/* Index types of struct libagx_omm_geometry_desc. */
#define LIBAGX_OMM_INDEX_NONE 0u
#define LIBAGX_OMM_INDEX_U16 1u
#define LIBAGX_OMM_INDEX_U32 2u

/* Build-time description of one geometry's OMM linkage (uploaded). count is
 * the geometry's primitive count; an indirect build passes its maximum and
 * the device ranges give the count. Padded to a power-of-two size: AGX3
 * cannot encode the multiply-add a 40-byte array stride takes in 64-bit
 * address arithmetic. */
struct libagx_omm_geometry_desc {
   uint64_t micromap;
   uint64_t indices;
   uint64_t stride;
   uint32_t index_type;
   uint32_t base_triangle;
   uint32_t flags;
   uint32_t count;
   uint64_t reserved[3];
};

#endif
