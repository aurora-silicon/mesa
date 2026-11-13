/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Opacity micromap build, copy and properties (see micromap.h), and the
 * per-primitive index arrays of an acceleration structure's OMM table.
 */
#include "compiler/libcl/libcl.h"
#include "gen1_as.h"
#include "micromap.h"

KERNEL(32)
libagx_micromap_clear(global uint32_t *counters)
{
   if (cl_global_id.x < LIBAGX_MICROMAP_BUCKETS)
      counters[cl_global_id.x] = 0;
}

/* One thread per triangle: claim a slot in its bucket and write its record.
 * Thread 0 also writes the header. A triangle whose format or level is
 * invalid, or whose bucket is full, gets an invalid record (format 0).
 */
KERNEL(32)
libagx_micromap_assign(global uint32_t *image, global uint8_t *triangles,
                       uint64_t stride, global uint32_t *scratch,
                       global struct libagx_micromap_bucket *buckets,
                       uint32_t triangle_count, uint64_t image_bytes,
                       uint32_t data_offset)
{
   uint32_t i = cl_global_id.x;
   if (i == 0) {
      image[LIBAGX_MICROMAP_WORD_MAGIC] = LIBAGX_MICROMAP_MAGIC;
      image[LIBAGX_MICROMAP_WORD_VERSION] = LIBAGX_MICROMAP_VERSION;
      image[LIBAGX_MICROMAP_WORD_TRIANGLES] = triangle_count;
      image[3] = 0;
      image[LIBAGX_MICROMAP_WORD_BYTES_LO] = (uint32_t)image_bytes;
      image[LIBAGX_MICROMAP_WORD_BYTES_HI] = (uint32_t)(image_bytes >> 32);
      image[LIBAGX_MICROMAP_WORD_DATA] = data_offset;
      for (uint32_t w = LIBAGX_MICROMAP_WORD_DATA + 1;
           w < LIBAGX_MICROMAP_HEADER_BYTES / 4; ++w)
         image[w] = 0;
   }
   if (i >= triangle_count)
      return;

   global uint32_t *src = (global uint32_t *)(triangles + (uint64_t)i * stride);
   uint32_t level_format = src[1];
   uint32_t level = level_format & 0xffffu, format = level_format >> 16;
   global uint32_t *record =
      image + (LIBAGX_MICROMAP_RECORDS + (uint64_t)i * LIBAGX_MICROMAP_RECORD_BYTES) / 4;
   record[0] = 0;
   record[1] = 0;
   if (format < 1 || format > 2 || level > LIBAGX_MICROMAP_MAX_LEVEL)
      return;

   uint32_t b = libagx_micromap_bucket_index(format, level);
   struct libagx_micromap_bucket bucket = buckets[b];
   uint32_t slot = atomic_inc((volatile global uint32_t *)&scratch[b]);
   if (slot >= bucket.count)
      return;
   record[0] = bucket.data_word + slot * bucket.slot_words;
   record[1] = level_format;
   scratch[LIBAGX_MICROMAP_SCRATCH_SLOTS / 4 + bucket.first_slot + slot] = i;
}

/* One thread per data word of a run of one bucket's slots, from slot
 * chunk_first on. Slots no triangle claimed (usage counts may overstate a
 * bucket) are left alone. The application's dataOffset has no alignment, so
 * words are assembled from bytes. */
KERNEL(32)
libagx_micromap_copy_data(global uint32_t *image, global uint8_t *triangles,
                          uint64_t stride, global uint8_t *data,
                          global uint32_t *scratch, uint32_t bucket,
                          uint32_t first_slot, uint32_t chunk_first,
                          uint32_t first_word, uint32_t slot_words,
                          uint32_t data_bytes, uint32_t words)
{
   uint32_t id = cl_global_id.x;
   if (id >= words)
      return;
   uint32_t s = chunk_first + id / slot_words, w = id % slot_words;
   if (s >= scratch[bucket])
      return;
   uint32_t tri = scratch[LIBAGX_MICROMAP_SCRATCH_SLOTS / 4 + first_slot + s];
   uint32_t offset = *(global uint32_t *)(triangles + (uint64_t)tri * stride);
   global uint8_t *src = data + offset;
   uint32_t value = 0;
   for (uint32_t k = 0; k < 4; ++k) {
      uint32_t byte = w * 4 + k;
      if (byte < data_bytes)
         value |= (uint32_t)src[byte] << (8 * k);
   }
   image[(uint64_t)first_word + id] = value;
}

/* Copy an image, sized by its own header, bounded by limit_bytes. With a
 * prefix, the first threads also copy LIBAGX_MICROMAP_SERIAL_HEADER_BYTES
 * from prefix to prefix_dst (serialization). */
KERNEL(32)
libagx_micromap_copy(global uint32_t *dst, global uint32_t *src,
                     uint64_t limit_bytes, global uint32_t *prefix_dst,
                     global uint32_t *prefix)
{
   uint64_t bytes = (uint64_t)src[LIBAGX_MICROMAP_WORD_BYTES_LO] |
                    ((uint64_t)src[LIBAGX_MICROMAP_WORD_BYTES_HI] << 32);
   if (bytes > limit_bytes)
      bytes = limit_bytes;
   uint64_t first = (uint64_t)cl_global_id.x * 4;
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < bytes)
         dst[word] = src[word];
   }
   if (prefix_dst && cl_global_id.x < LIBAGX_MICROMAP_SERIAL_HEADER_BYTES / 4)
      prefix_dst[cl_global_id.x] = prefix[cl_global_id.x];
}

/* serialized 0: compacted size; 1: serialization size. */
KERNEL(1)
libagx_micromap_property(global uint32_t *image, global uint64_t *report,
                         uint32_t serialized)
{
   uint64_t bytes = (uint64_t)image[LIBAGX_MICROMAP_WORD_BYTES_LO] |
                    ((uint64_t)image[LIBAGX_MICROMAP_WORD_BYTES_HI] << 32);
   *report = bytes + (serialized ? LIBAGX_MICROMAP_SERIAL_HEADER_BYTES : 0);
}

/* Lay out the table: the header and every geometry's entry, whose index array
 * offset is the running primitive count. Then publish it in the directory.
 * Serial over the geometries: the table is built once per build. */
KERNEL(1)
libagx_omm_table_init(global uint32_t *table, uint64_t table_va,
                      global struct libagx_omm_geometry_desc *desc,
                      uint32_t geometry_count, uint32_t primitive_bound,
                      global uint8_t *ranges, uint32_t range_stride,
                      global uint32_t *directory_entry)
{
   uint32_t first = (LIBAGX_OMM_TABLE_HEADER_BYTES +
                     geometry_count * LIBAGX_OMM_TABLE_GEOMETRY_BYTES) / 4;
   uint32_t running = 0;
   global uint32_t *entry = table + LIBAGX_OMM_TABLE_HEADER_BYTES / 4;
   global uint8_t *range = ranges;
   for (uint32_t g = 0; g < geometry_count; ++g) {
      uint32_t count = desc->count;
      if (ranges) {
         count = min(count, *(global uint32_t *)range);
         range += range_stride;
      }
      count = min(count, primitive_bound - running);
      uint64_t micromap = desc->micromap;
      entry[0] = (uint32_t)micromap;
      entry[1] = (uint32_t)(micromap >> 32);
      entry[2] = (first + running) * 4;
      entry[3] = desc->flags;
      running += count;
      entry += LIBAGX_OMM_TABLE_GEOMETRY_BYTES / 4;
      desc++;
   }
   table[0] = LIBAGX_OMM_TABLE_MAGIC;
   table[1] = geometry_count;
   table[2] = running;
   table[3] = 0;
   directory_entry[0] = (uint32_t)table_va;
   directory_entry[1] = (uint32_t)(table_va >> 32);
   directory_entry[2] = geometry_count;
   directory_entry[3] = running;
}

KERNEL(1)
libagx_omm_directory_write(global uint32_t *directory_entry, uint64_t table_va,
                           uint32_t geometry_count, uint32_t primitives)
{
   directory_entry[0] = (uint32_t)table_va;
   directory_entry[1] = (uint32_t)(table_va >> 32);
   directory_entry[2] = geometry_count;
   directory_entry[3] = primitives;
}

/* One thread per primitive of the build: find its geometry (binary search on
 * the index array offsets), then store its micromap triangle index with
 * base_triangle applied, or its special index. UINT16 special indices are
 * the 16-bit casts of -1 .. -4. */
KERNEL(32)
libagx_omm_table_resolve(global uint32_t *table,
                         global struct libagx_omm_geometry_desc *desc,
                         uint32_t primitive_bound)
{
   uint32_t t = cl_global_id.x;
   uint32_t geometry_count = table[1], primitives = table[2];
   if (t >= primitive_bound || t >= primitives || !geometry_count)
      return;
   uint32_t first = (LIBAGX_OMM_TABLE_HEADER_BYTES +
                     geometry_count * LIBAGX_OMM_TABLE_GEOMETRY_BYTES) / 4;
   uint32_t lo = 0, hi = geometry_count - 1;
   while (lo < hi) {
      uint32_t mid = (lo + hi + 1) / 2;
      uint32_t start = table[(LIBAGX_OMM_TABLE_HEADER_BYTES +
                              mid * LIBAGX_OMM_TABLE_GEOMETRY_BYTES) / 4 + 2] / 4 - first;
      if (start <= t)
         lo = mid;
      else
         hi = mid - 1;
   }
   global uint32_t *entry =
      table + (LIBAGX_OMM_TABLE_HEADER_BYTES + lo * LIBAGX_OMM_TABLE_GEOMETRY_BYTES) / 4;
   uint32_t i = t - (entry[2] / 4 - first);
   struct libagx_omm_geometry_desc d = desc[lo];
   uint32_t value = 0xffffffffu;
   if (d.flags & LIBAGX_OMM_GEOMETRY_LINKED) {
      value = i;
      if (d.index_type == LIBAGX_OMM_INDEX_U16) {
         value = *(global uint16_t *)(d.indices + (uint64_t)i * d.stride);
         if (value >= 0xfffcu)
            value |= 0xffff0000u;
      } else if (d.index_type == LIBAGX_OMM_INDEX_U32) {
         value = *(global uint32_t *)(d.indices + (uint64_t)i * d.stride);
      }
      if (value < 0xfffffffcu)
         value += d.base_triangle;
   }
   table[first + t] = value;
}

/* Copy an OMM table (acceleration-structure clone), sized by its header and
 * bounded by limit_bytes, and publish the copy. */
KERNEL(32)
libagx_omm_table_copy(global uint32_t *dst, global uint32_t *src,
                      uint64_t limit_bytes, global uint32_t *directory_entry,
                      uint64_t table_va)
{
   uint32_t geometry_count = src[1], primitives = src[2];
   uint64_t bytes = LIBAGX_OMM_TABLE_HEADER_BYTES +
                    (uint64_t)geometry_count * LIBAGX_OMM_TABLE_GEOMETRY_BYTES +
                    (uint64_t)primitives * 4;
   if (bytes > limit_bytes)
      bytes = limit_bytes;
   uint64_t first = (uint64_t)cl_global_id.x * 4;
   for (uint32_t i = 0; i < 4; ++i) {
      uint64_t word = first + i;
      if (word * 4 < bytes)
         dst[word] = src[word];
   }
   if (cl_global_id.x == 0) {
      directory_entry[0] = (uint32_t)table_va;
      directory_entry[1] = (uint32_t)(table_va >> 32);
      directory_entry[2] = geometry_count;
      directory_entry[3] = primitives;
   }
}

/* A compacted-size report of a BLAS with an OMM table: the table follows the
 * compacted image, 16-byte aligned. */
KERNEL(1)
libagx_omm_size_adjust(global uint64_t *report, uint64_t table_bytes)
{
   *report = ((*report + 15) & ~(uint64_t)15) + table_bytes;
}

/* One thread per instance slot of a TLAS: mark the TLAS's directory entry
 * when an instance needs micromaps (see LIBAGX_OMM_TLAS_STAMP_BIT). The
 * instance pointers start at the image's LIBAGX_GEN1_AS_POINTER_WORD offset
 * and end where its descriptor (word 14) starts; each 16-byte entry holds the
 * record's offset plus 4, or 0 for an inactive instance. Each record stores
 * its BLAS's AS id at +4 and its instance flags at +8. A copy passes the
 * destination's metadata (only TLASes are scanned) and no instance bound. */
KERNEL(32)
libagx_omm_tlas_scan(global uint32_t *entry, global uint8_t *tlas,
                     global uint32_t *directory, global uint32_t *metadata,
                     uint32_t instance_bound, uint32_t stamp)
{
   if (metadata && ((global struct libagx_gen1_as_metadata *)metadata)->geometry_type != 2u)
      return;
   global uint64_t *words = (global uint64_t *)tlas;
   if (instance_bound == 0xffffffffu)
      instance_bound = words[LIBAGX_GEN1_AS_SIZE_WORD]
         ? (uint32_t)((words[14] - words[LIBAGX_GEN1_AS_POINTER_WORD]) / 16) : 0;
   uint32_t i = cl_global_id.x;
   if (i == 0) {
      entry[0] = 0;
      entry[1] = 0;
      entry[2] = stamp;
   }
   if (i >= instance_bound)
      return;
   uint64_t record = *(global uint64_t *)(tlas + words[LIBAGX_GEN1_AS_POINTER_WORD] +
                                          (uint64_t)i * 16);
   if (!record)
      return;
   global uint32_t *instance = (global uint32_t *)(tlas + record - 4);
   if (instance[2] & LIBAGX_OMM_INSTANCE_DISABLE)
      return;
   global uint32_t *blas = directory + (uint64_t)instance[1] *
                                       (LIBAGX_OMM_DIRECTORY_ENTRY_BYTES / 4);
   if (blas[0] | blas[1])
      entry[3] = stamp;
}
