/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 system-register reads, device/local loads and stores, atomics,
 * ray-tracing/SCS/mesh UVB operations, constant-program imports, and stack
 * spills/reloads.
 */

#include "agx3_pack_internal.h"
#include "agx3_atomic_core.h"
#include "agx3_cprog_input.h"
#include "agx3_local_tps_core.h"
#include "agx3_mem_core.h"
#include "agx3_mesh_core.h"
#include "agx3_pixel_core.h"
#include "agx3_rt_core.h"
#include "agx3_rt_gen1_traversal.h"
#include "agx3_rt_m5_core.h"
#include "agx3_scs_core.h"
#include "agx3_sr_core.h"
#include "agx3_stage_core.h"
#include "agx3_static_spill_core.h"
#include "agx_mesh_abi.h"

#include <limits.h>

/* Map a Mesa special register to the corpus-witnessed Apple SR number and
 * destination width. Only registers tied to a source-level builtin by an
 * isolating probe are mapped; everything else -- including any (SR, width)
 * combination Apple never emitted -- fails closed so the shader is honestly
 * rejected, not miscompiled. */
static bool
agx3_map_sr(enum agx_isa_encoding enc, enum agx_sr sr, uint8_t *apple,
            enum agx3_sr_width *width)
{
#define SR(mesa, num, w)                                                       \
   case mesa:                                                                  \
      *apple = (num);                                                          \
      *width = (w);                                                            \
      return true
   switch (sr) {
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_X, 26, AGX3_SR_W32);
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_Y, 27, AGX3_SR_W32);
      SR(AGX_SR_THREADGROUP_POSITION_IN_GRID_Z, 28, AGX3_SR_W32);
      SR(AGX_SR_THREADS_PER_THREADGROUP_X, 17, AGX3_SR_W16);
      SR(AGX_SR_THREADS_PER_THREADGROUP_Y, 18, AGX3_SR_W16);
      SR(AGX_SR_THREADS_PER_THREADGROUP_Z, 19, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X, 20, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y, 21, AGX3_SR_W16);
      SR(AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z, 22, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_X, 9, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y, 10, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z, 11, AGX3_SR_W16);
      SR(AGX_SR_THREAD_INDEX_IN_THREADGROUP, 13, AGX3_SR_W16);
      SR(AGX_SR_SCS_THREAD_INDEX, 59, AGX3_SR_W16);
      SR(AGX_SR_THREAD_INDEX_IN_SUBGROUP, 7, AGX3_SR_W16);
      SR(AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP, 6, AGX3_SR_W16);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_X, 23, AGX3_SR_W32);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_Y, 24, AGX3_SR_W32);
      SR(AGX_SR_THREAD_POSITION_IN_GRID_Z, 25, AGX3_SR_W32);
   case AGX_SR_MESH_PAYLOAD_LO:
   case AGX_SR_MESH_PAYLOAD_HI:
      if (enc != AGX_ISA_ENCODING_AGX3C) return false;
      *apple = sr == AGX_SR_MESH_PAYLOAD_LO ? 62 : 63;
      *width = sr == AGX_SR_MESH_PAYLOAD_LO ? AGX3_SR_W32 : AGX3_SR_W16;
      return true;
   case AGX_SR_MESH_PRIMITIVE_STORAGE_LO:
   case AGX_SR_MESH_PRIMITIVE_STORAGE_HI:
      

      if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
         return false;
      *apple = sr == AGX_SR_MESH_PRIMITIVE_STORAGE_LO ? 64 : 65;
      *width = sr == AGX_SR_MESH_PRIMITIVE_STORAGE_LO ? AGX3_SR_W32 : AGX3_SR_W16;
      return true;
   case AGX_SR_INPUT_SAMPLE_MASK:
      /* G17G and G18G read [[sample_mask]] from the same W16 SR39, as the
       * first, entry-marked read of the fragment (metal-nt applegpu_g17g and
       * applegpu_g18g, m6-sample-mask-in at 1x, 2x and 4x: rld f21 = 39,
       * f48 = 1, then srl #8). */
      if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
         return false;
      *apple = 39;
      *width = AGX3_SR_W16;
      return true;
      SR(AGX_SR_COVERAGE_MASK, 39, AGX3_SR_W16);
      SR(AGX_SR_IS_ACTIVE_THREAD, 2, AGX3_SR_W16);
   case AGX_SR_BACKFACING:
      if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
         return false;
      *apple = 41;
      *width = AGX3_SR_W16;
      return true;
   case AGX_SR_BACKGROUND_LAYER_ID:
      /* Native G17G llvm.agx2.get.layerid reads the same SR12/W16
       * tile-layer input as G17P (background-load internal witness).
       * G18G's 21 84 43 00 00 40 is the same field vector as G17G's read. */
      if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
         return false;
      *apple = 12;
      *width = AGX3_SR_W16;
      return true;
      /* Apple's uniform-address atomic election reads the active-thread
       * index (SR 34) and count (SR 36) of the SIMD-group, and quad_is_first
       * reads SR 33, all into half registers in the short form, also after
       * discard_fragment. */
      SR(AGX_SR_ACTIVE_THREAD_INDEX_IN_QUAD, 33, AGX3_SR_W16);
      SR(AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP, 34, AGX3_SR_W16);
      SR(AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP, 36, AGX3_SR_W16);
      /* The invocation enums carry hardware SR bytes (0xd8/0xdd), while
       * agx3_sr_core consumes Apple's normalized f21 value. */
      SR(AGX_SR_INSTANCE_ID, 46, AGX3_SR_W32);
      SR(AGX_SR_TESS_PATCH_ID, 47, AGX3_SR_W32);
      SR(AGX_SR_TESS_COORD_U, 48, AGX3_SR_W32);
      SR(AGX_SR_TESS_COORD_V, 49, AGX3_SR_W32);
      SR(AGX_SR_VERTEX_ID, 51, AGX3_SR_W32);
   default:
      return false;
   }
#undef SR
}

/* Translate a get_sr into the normalized SR-read descriptor. The destination
 * size must match the witnessed width of the register: Apple reads 16-bit SRs
 * into a half register (consumers or an explicit zero-extend handle the rest),
 * and no witness reads one at another width. */
static bool
agx3_translate_sr(enum agx_isa_encoding enc, const agx_instr *I, uint8_t slot,
                  uint16_t wait_mask, struct agx3_sr_desc *d)
{
   memset(d, 0, sizeof(*d));

   if (I->nr_dests < 1 || I->dest[0].type == AGX_INDEX_NULL)
      return false;
   const agx_index dst = I->dest[0];
   if (!agx3_src_is_register(dst) || dst.memory || dst.value >= 256)
      return false;

   uint8_t apple;
   enum agx3_sr_width width;
   if (!agx3_map_sr(enc, I->sr, &apple, &width))
      return false;
   if (width == AGX3_SR_W32 ? dst.size != AGX_SIZE_32 : dst.size != AGX_SIZE_16)
      return false;

   d->dest = dst.value;
   d->width = width;
   d->sr = apple;
   d->slot = slot;
   d->wait_mask = wait_mask;
   return true;
}

/* Append the half-registers an operand occupies (empty for immediates,
 * uniforms, or null). */
unsigned
agx3_expand_regs(agx_index idx, uint16_t *arr, unsigned n)
{
   if (!agx3_src_is_register(idx) || idx.memory)
      return n;
   const unsigned halves = agx_size_align_16(idx.size) * agx_channels(idx);
   /* Never silently omit the rest of an operand or later operands. In
    * particular, texture/sampler handles follow the large gradient tuple. */
   if (n > AGX3_SLOT_MAX_REGS || halves > AGX3_SLOT_MAX_REGS - n ||
       idx.value >= AGX3_SLOT_GPR_TOKENS ||
       halves > AGX3_SLOT_GPR_TOKENS - idx.value)
      return AGX3_SLOT_MAX_REGS + 1;
   for (unsigned i = 0; i < halves; ++i)
      arr[n++] = (uint16_t)(idx.value + i);
   return n;
}

static int
agx3_fmt_to_width(enum agx_format f)
{
   switch (f) {
   case AGX_FORMAT_I8:
      return AGX3_MEM_W8;
   case AGX_FORMAT_I16:
   case AGX_FORMAT_F16:
      return AGX3_MEM_W16;
   case AGX_FORMAT_I32:
      return AGX3_MEM_W32;
   default:
      return -1; /* norm/packed formats have no AGX3 device-op witness */
   }
}

static bool
agx3_scs_plain_index(agx_index index)
{
   return !index.kill && !index.cache && !index.discard && !index.abs &&
          !index.neg && !index.memory && !index.has_reg &&
          agx_channels(index) == 1;
}

static bool
agx3_translate_scs(enum agx_isa_encoding enc, const agx_instr *I, bool is_store, bool indexed,
                   bool register_indexed, bool register_base,
                   uint8_t slot, uint16_t wait_mask,
                   struct agx3_scs_desc *d)
{
   memset(d, 0, sizeof(*d));
   const int width = agx3_fmt_to_width(I->format);
   const int64_t byte_offset = (int64_t)I->imm;
   const unsigned expected_srcs = register_base ? (is_store ? 2 : 1) :
                                   indexed ? (is_store ? 3 : 2) : (is_store ? 1 : 0);
   if (width < 0 || byte_offset < INT16_MIN || byte_offset > INT16_MAX ||
       I->nr_dests != (is_store ? 0 : 1) ||
       I->nr_srcs != expected_srcs ||
       ((is_store || !indexed) &&
        (I->scs_zero_extend || I->scs_c_load_marker)))
      return false;

   agx_index data = is_store ? I->src[0] : I->dest[0];
   /* Post-RA register-cache hints are not SCS operand modifiers. Its
    * witnessed final-use bit is carried separately in the SCS descriptor. */
   const bool data_kill = enc == AGX_ISA_ENCODING_AGX3C &&
                          is_store && (data.kill || data.discard);
   data.kill = data.cache = data.discard = false;
   agx_index scalar_data = data;
   scalar_data.channels_m1 = 0;
   const enum agx_size expected =
      (width == AGX3_MEM_W32 || I->scs_zero_extend) ? AGX_SIZE_32 : AGX_SIZE_16;
   if (!agx3_src_is_register(data) || !agx3_scs_plain_index(scalar_data) ||
       data.size != expected || I->mask > 15 ||
       agx_channels(data) != util_last_bit(I->mask ? I->mask : 1) ||
       (data.value & (agx_size_align_16(expected) - 1)) ||
       data.value + agx_size_align_16(expected) * agx_channels(data) > 256)
      return false;

   *d = (struct agx3_scs_desc){
      .data = data.value,
      .width = (enum agx3_mem_width)width,
      .byte_offset = (int16_t)byte_offset,
      .address_kind = indexed ? AGX3_SCS_ADDRESS_UNIFORM_INDEXED
                              : AGX3_SCS_ADDRESS_DIRECT,
      .slot = is_store ? 0 : slot,
      .wait_mask = wait_mask | (I->scs_rt_wait ? 32 : 0),
      .value_kill = data_kill,
      .component_mask = I->mask ? I->mask : 1,
   };

   if (register_indexed || register_base) {
      agx_index base = I->src[is_store ? 1 : 0];
      agx_index index = register_base ? agx_register(0, AGX_SIZE_32) :
                                       I->src[is_store ? 2 : 1];
      const bool base_kill = enc == AGX_ISA_ENCODING_AGX3C &&
                             (base.kill || base.discard);
      const bool index_kill = enc == AGX_ISA_ENCODING_AGX3C &&
                              (index.kill || index.discard);
      base.kill = base.cache = base.discard = false;
      index.kill = index.cache = index.discard = false;
      if (!agx3_src_is_register(base) || base.size != AGX_SIZE_32 ||
          !agx3_scs_plain_index(base) || (base.value & 1) || base.value > 254 ||
          !agx3_src_is_register(index) || index.size != AGX_SIZE_32 ||
          !agx3_scs_plain_index(index) || (index.value & 1) || index.value > 254)
         return false;
      d->address_kind = register_base ? AGX3_SCS_ADDRESS_REGISTER_BASE :
                                        AGX3_SCS_ADDRESS_REGISTER_INDEXED;
      d->base_register = base.value;
      d->index = index.value;
      d->has_address_selector = true;
      d->address_selector = I->scs_address_selector;
      d->base_kill = base_kill;
      d->index_kill = !register_base && index_kill;
      d->value_kill = data_kill;
      d->c_status_load = enc == AGX_ISA_ENCODING_AGX3C &&
                         !is_store && I->scs_c_status_load;
      d->b_status_load = enc == AGX_ISA_ENCODING_AGX3B &&
                         !is_store && I->scs_c_status_load;
      d->c_load_marker = !is_store && I->scs_c_load_marker;
      if (I->scs_rt_completion) {
         if (is_store || width != AGX3_MEM_W8 ||
             !(register_indexed ? I->scs_address_selector == 3 &&
                                  I->scs_c_load_marker
                                : register_base && I->scs_address_selector == 4))
            return false;
         d->wait_mask |= 0x80;
      }
      return true;
   }

   if (indexed) {
      agx_index base = I->src[is_store ? 1 : 0];
      agx_index index = I->src[is_store ? 2 : 1];
      const bool index_kill = enc == AGX_ISA_ENCODING_AGX3C &&
                              (index.kill || index.discard);
      base.kill = base.cache = base.discard = false;
      index.kill = index.cache = index.discard = false;

      /* The pseudo itself preserves the address-space provenance.  Require
       * the physical operands to preserve it too: a scalar 64-bit uniform
       * base and a scalar aligned 32-bit GPR index.  In particular, no
       * register base or arithmetic source modifier is silently encoded.
       * Component masks and final index use have independent native witnesses.
       */
      if (base.type != AGX_INDEX_UNIFORM || base.size != AGX_SIZE_64 ||
          !agx3_scs_plain_index(base) || base.value >= 512 ||
          !agx3_src_is_register(index) || index.size != AGX_SIZE_32 ||
          !agx3_scs_plain_index(index) || (index.value & 1) ||
          index.value + agx_size_align_16(index.size) > 256)
         return false;

      d->uniform_base = base.value;
      d->index = index.value;
      d->index_kill = index_kill;
      d->has_address_selector = true;
      d->address_selector = I->scs_address_selector;
      d->zero_extend = !is_store && I->scs_zero_extend;
      d->c_load_marker = !is_store && I->scs_c_load_marker;
   }

   return true;
}

/* The destination-less SCS load: a plain 32-bit register base and, in the
 * indexed form, the low half of a plain 32-bit register as its 16-bit index.
 * Post-RA cache hints are not operands of the form. */
static bool
agx3_translate_scs_touch(enum agx_isa_encoding enc, const agx_instr *I,
                         struct agx3_scs_touch_desc *d)
{
   const bool indexed = I->op == AGX_OPCODE_SCS_TOUCH_REGISTER_INDEXED_AGX3;
   const int64_t byte_offset = (int64_t)I->imm;
   memset(d, 0, sizeof(*d));
   if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_dests != 0 ||
       I->nr_srcs != (indexed ? 2u : 1u) || byte_offset < INT16_MIN ||
       byte_offset > INT16_MAX || I->mask > 15 ||
       (!indexed && I->scs_address_selector))
      return false;

   agx_index base = I->src[0];
   base.kill = base.cache = base.discard = false;
   if (!agx3_src_is_register(base) || base.size != AGX_SIZE_32 ||
       !agx3_scs_plain_index(base))
      return false;

   *d = (struct agx3_scs_touch_desc){
      .base_register = base.value,
      .has_index = indexed,
      .byte_offset = (int16_t)byte_offset,
      .component_mask = I->mask,
   };

   if (indexed) {
      agx_index index = I->src[1];
      index.kill = index.cache = index.discard = false;
      if (!agx3_src_is_register(index) || index.size != AGX_SIZE_32 ||
          !agx3_scs_plain_index(index) || (index.value & 1))
         return false;
      d->index = index.value;
      d->address_selector = I->scs_address_selector;
   }

   /* agx3_scs_encode_touch bounds the registers, mask and selector. */
   return true;
}

/* Translate a device_load/device_store into the normalized memory descriptor.
 * Returns false for any operand shape outside the grounded domain. */
static bool
agx3_translate_mem(enum agx_isa_encoding enc, const agx_instr *I, bool is_store,
                   uint8_t slot, uint16_t wait_mask, struct agx3_mem_desc *d)
{
   memset(d, 0, sizeof(*d));

   const int w = agx3_fmt_to_width(I->format);
   if (w < 0 || I->mask < 1 || I->mask > 15 || I->shift > 4)
      return false;
   d->width = (enum agx3_mem_width)w;
   d->mask = (uint8_t)I->mask;
   d->shift = (uint8_t)I->shift;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const bool packed_bytes = w == AGX3_MEM_W8 && I->memory_is_packed;
   const bool packed_byte_store = is_store && packed_bytes;
   /* Native scalar byte loads write an aligned low half. Stores can also
    * address individual halves: Apple's byte_halves_odd_zero stores
    * ushort2.y from half3 with W8/mask1. A packed vector byte store likewise
    * names its physical 16-bit payload in half-register units, regardless of
    * the logical component mask. */
   const bool scalar_half = data.size == AGX_SIZE_16 &&
      (w == AGX3_MEM_W16 || w == AGX3_MEM_W8) &&
      I->mask == 1 && agx_channels(data) == 1;
   const bool byte_load_half = !is_store && w == AGX3_MEM_W8 && scalar_half;
   const unsigned data_halfs = packed_bytes
      ? DIV_ROUND_UP(util_bitcount(I->mask), 2)
      : agx_size_align_16(data.size) * util_bitcount(I->mask);
   if (!agx3_src_is_register(data) || data.memory ||
       ((!scalar_half || byte_load_half) && !packed_byte_store &&
        (data.value & 1)) ||
       (packed_byte_store &&
        (data.size != AGX_SIZE_16 || agx_channels(data) != data_halfs)) ||
       data.value + data_halfs > 256)
      return false;
   d->data = data.value;

   const agx_index base = I->src[is_store ? 1 : 0];
   if (base.type == AGX_INDEX_UNIFORM) {
      d->base_is_uniform = true;
      d->base = base.value;
   } else if (agx3_src_is_register(base) && !base.memory) {
      d->base_is_uniform = false;
      d->base = base.value;
      d->base_kill = base.kill;
   } else {
      return false;
   }

   const agx_index index = I->src[is_store ? 2 : 1];

   /*
    * The index -- immediate or register -- counts ELEMENTS, and
    * agx_nir_lower_address() has already subtracted the format shift from
    * I->shift (match.shift = shift - format_shift, the AGX2 convention where
    * the hardware rescales). AGX3 does not rescale, so the true scale is
    * I->shift + w, where w is log2(element bytes).
    */
   const unsigned total_shift = (unsigned)I->shift +
      (I->memory_index_is_bytes ? 0 : (unsigned)w);

   if (index.type == AGX_INDEX_IMMEDIATE) {
      d->idx_kind = AGX3_MEM_IDX_IMM;
      /*
       * A byte offset is absolute, so it must carry the FULL scale. Using
       * I->shift alone addressed element 9 of the root descriptor table at
       * byte 9 instead of byte 36 -- invisible on every probe whose only
       * immediate index is 0, and the reason the constant-data (rodata)
       * pointer came back wrong.
       */
      int32_t bytes;
      if (agx3_mem_immediate_byte_offset(I, index.value, &bytes)) {
         d->idx_scaled = false;
         d->byte_off = bytes;
         d->shift = 0;
      } else if (index.value <= 0xffff && total_shift <= 4 &&
                 !I->memory_byte_offset) {
         d->idx_scaled = true;
         d->elem_idx = index.value;
         d->shift = (uint8_t)total_shift;
      } else {
         return false; /* needs address-math legalization */
      }
   } else if (agx3_src_is_register(index) && !index.memory) {
      d->idx_kind = AGX3_MEM_IDX_REG;
      if ((index.value & 1) != 0 ||
          index.value >= (enc == AGX_ISA_ENCODING_AGX3C
                             ? AGX3C_NUM_REGS : AGX3B_NUM_REGS))
         return false;
      d->idx_reg = index.value;
      /* Keep the IR's own liveness: forcing this bit to 1 (so the store's
       * index operand cannot decode as "operand absent") made no difference
       * on hardware. */
      d->idx_kill = index.kill;
      /* M5 C45: NIR abs requests zero extension; plain indices are signed. */
      d->idx_signed = !index.abs;

      /*
       * A REGISTER index is scaled by the shift alone, not by the access
       * format: shift == log2(element bytes) plus any vector stride, and
       * shift 0 only for immediate byte offsets. agx_nir_lower_address()
       * subtracted the format shift (the AGX2 convention), so add it back;
       * `w` is log2(element bytes).
       */
      if (total_shift > 4)
         return false;
      d->shift = (uint8_t)total_shift;
      d->byte_off = I->memory_byte_offset;
   } else {
      return false;
   }

   d->slot = slot;
   d->wait_mask = wait_mask;
   d->coherent = I->coherent;
   return true;
}

/* Translate the shared-memory shape Mesa currently emits. AGX3 preprocessing
 * converts byte offsets to scalar-element indices, matching Apple's grounded
 * ldtg/sttg register-index convention. */
static bool
agx3_translate_local(enum agx_isa_encoding enc, const agx_instr *I,
                     bool is_store, uint8_t slot, uint16_t wait_mask,
                     struct agx3_local_desc *d)
{
   memset(d, 0, sizeof(*d));

   const int w = agx3_fmt_to_width(I->format);
   if (w < 0 || I->mask < 1 || I->mask > 15 ||
       I->mask != BITFIELD_MASK(util_bitcount(I->mask)))
      return false;
   d->width = (enum agx3_mem_width)w;
   d->mask = (uint8_t)I->mask;
   /* The index counts scalar elements even for vector accesses. Scaling by
    * the total vector width makes successive vec4 stores stride four times
    * too far and breaks explicitly aliased shared-memory views. */
   d->shift = d->width;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const enum agx_size expected =
      d->width == AGX3_MEM_W32 ? AGX_SIZE_32 : AGX_SIZE_16;
   const bool packed_bytes = d->width == AGX3_MEM_W8 && I->memory_is_packed;
   const unsigned halves = packed_bytes ? DIV_ROUND_UP(util_bitcount(I->mask), 2)
      : agx_size_align_16(expected) * util_bitcount(I->mask);
   if (!agx3_src_is_register(data) || data.memory || data.size != expected ||
       (data.value & (agx_size_align_16(expected) - 1)) ||
       data.value + halves > 256)
      return false;
   d->data = data.value;
   d->value_kill = is_store && data.kill;

   const agx_index base = I->src[is_store ? 1 : 0];
   if (base.size != AGX_SIZE_16 || base.memory || base.abs || base.neg)
      return false;
   if (base.type == AGX_INDEX_IMMEDIATE && base.value <= 255) {
      /* An immediate base is a constant byte offset folded off the address
       * by agx_nir_lower_shared_element_index(): it travels in the
       * instruction's immediate byte-offset field (f71, witnessed in Apple's
       * CmAntiFirefly ldtg at 16/32/48/64 next to a register base) and the
       * workgroup-relative base stays zero.
       */
      d->base_kind = AGX3_LOCAL_BASE_IMMEDIATE;
      d->byte_offset = base.value;
   } else if (base.type == AGX_INDEX_UNIFORM && base.value < 512) {
      d->base_kind = AGX3_LOCAL_BASE_UNIFORM;
      d->base = base.value;
   } else if (agx3_src_is_register(base) && base.value < 256) {
      d->base_kind = AGX3_LOCAL_BASE_REGISTER;
      d->base = base.value;
      d->base_kill = base.kill;
   } else {
      return false;
   }

   const agx_index index = I->src[is_store ? 2 : 1];
   if (index.size != AGX_SIZE_16 || index.memory || index.abs || index.neg)
      return false;
   if (index.type == AGX_INDEX_IMMEDIATE && index.value < 256) {
      d->index = index.value;
   } else if (agx3_src_is_register(index) && index.value < 256) {
      d->index_is_register = true;
      /* The ldtg/sttg index is a 16-bit scalar, and its field names the
       * source in half-register units just like the compiler IR. Native
       * witnesses include odd indices (for example 5), so converting this
       * to a full-register number selects the wrong half and collapses
       * dynamic stores onto an unrelated value, commonly zero.
       */
      d->index = index.value;
      d->index_kill = index.discard;
   } else {
      return false;
   }


   /* Native M5 byte vectors occupy packed byte lanes, just as on G17P.
    * Metal535 uchar3 LDTG/STTG both leave f11 clear. */
   d->narrow_extend = false;
   d->slot = slot;
   d->wait_mask = wait_mask;
   return true;
}

static bool
agx3_atomic_plain_reg(agx_index index, enum agx_size size, unsigned channels)
{
   return agx3_src_is_register(index) && !index.memory && !index.abs &&
          !index.neg && !index.kill && !(index.cache && index.discard) &&
          index.size == size &&
          agx_channels(index) == channels &&
          (index.value & (agx_size_align_16(size) - 1)) == 0 &&
          index.value + agx_size_align_16(size) * channels <= 256;
}

static bool
agx3_map_atomic_op(enum agx_atomic_opc in, enum agx3_atomic_op *out)
{
   switch (in) {
#define OP(x)                                                                  \
   case AGX_ATOMIC_OPC_##x:                                                    \
      *out = AGX3_ATOMIC_##x;                                                  \
      return true
      OP(ADD);
      OP(SUB);
      OP(XCHG);
      OP(CMPXCHG);
      OP(UMIN);
      OP(IMIN);
      OP(UMAX);
      OP(IMAX);
      OP(AND);
      OP(OR);
      OP(XOR);
      OP(FADD);
#undef OP
   }

   return false;
}

static bool
agx3_translate_device_atomic(const agx_instr *I,
                             enum agx_isa_encoding enc, bool exec_masked,
                             uint8_t slot,
                             uint16_t wait_mask,
                             struct agx3_device_atomic_desc *d)
{
   memset(d, 0, sizeof(*d));
   if (I->op != AGX_OPCODE_ATOMIC || I->nr_dests != 1 || I->nr_srcs != 3 ||
       !agx3_map_atomic_op(I->atomic_opc, &d->op))
      return false;

   d->exec_masked = exec_masked;
   if (d->op == AGX3_ATOMIC_FADD && enc != AGX_ISA_ENCODING_AGX3B)
      return false;

   const unsigned value_channels =
      d->op == AGX3_ATOMIC_CMPXCHG ? 2 : 1;
   const agx_index value = I->src[0];
   if (!agx3_atomic_plain_reg(value, AGX_SIZE_32, value_channels))
      return false;
   d->value = value.value;
   d->value_kill = value.discard;

   const agx_index result = I->dest[0];
   d->returns_value = !agx_is_null(result);
   if (d->returns_value) {
      if (!agx3_atomic_plain_reg(result, AGX_SIZE_32, 1) ||
          (enc == AGX_ISA_ENCODING_AGX3B && result.value != value.value))
         return false;
      d->result = result.value;
      d->slot = slot;
   } else if (slot != 0) {
      return false;
   }

   const agx_index base = I->src[1];
   if (base.memory || base.abs || base.neg || base.kill ||
       (base.cache && base.discard) || base.size != AGX_SIZE_64 ||
       agx_channels(base) != 1) {
      return false;
   } else if (base.type == AGX_INDEX_UNIFORM) {
      d->base_kind = AGX3_ATOMIC_BASE_UNIFORM;
      d->base = base.value;
   } else if (agx3_src_is_register(base)) {
      d->base_kind = AGX3_ATOMIC_BASE_REGISTER;
      d->base = base.value;
      /* atomicdev has no base cache/discard field. As on AGX2, silently
       * dropping these allocator hints is semantics-preserving. */
   } else {
      return false;
   }

   const agx_index index = I->src[2];
   /* Immediate inlining uses abs to mark the wider 16-bit literal field,
    * including atomic element indices. It is not an address modifier. */
   if (index.memory || (index.abs && index.type != AGX_INDEX_IMMEDIATE) ||
       index.neg || index.kill || (index.cache && index.discard) ||
       agx_channels(index) != 1)
      return false;

   if (index.type == AGX_INDEX_IMMEDIATE) {
      if (index.kill || index.cache || index.discard ||
          index.value > UINT16_MAX)
         return false;
      const uint64_t byte_offset = (uint64_t)index.value << 2;
      if (byte_offset <= INT16_MAX) {
         d->index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
         d->byte_offset = byte_offset;
      } else {
         d->index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
         d->index = index.value;
         d->index_shift = 2;
      }
   } else if (agx3_src_is_register(index) && index.size == AGX_SIZE_32 &&
              !index.kill && !(index.cache && index.discard) &&
              (index.value & 1) == 0) {
      d->index_kind = AGX3_ATOMIC_INDEX_REGISTER;
      d->index = index.value;
      d->index_kill = index.discard;
      d->index_shift = 2;
   } else if (index.type == AGX_INDEX_UNIFORM && index.size == AGX_SIZE_32 &&
              index.value < 512 && !index.kill && !index.cache &&
              !index.discard) {
      d->index_kind = AGX3_ATOMIC_INDEX_UNIFORM;
      d->index = index.value;
      d->index_shift = 2;
   } else {
      return false;
   }

   d->wait_mask = wait_mask;
   return true;
}

/* AGX3b accepts either a literal byte displacement or a 16-bit GPR containing
 * the byte address.  Vulkan shared atomics use both: static accesses keep the
 * displacement, while dynamically indexed returned operations arrive with a
 * register address after scalar lowering. */
static bool
agx3_translate_local_atomic(const agx_instr *I, enum agx_isa_encoding enc,
                            uint8_t slot, uint16_t wait_mask,
                            struct agx3_local_atomic_desc *d)
{
   memset(d, 0, sizeof(*d));
   if (I->op != AGX_OPCODE_LOCAL_ATOMIC || I->nr_dests != 1 ||
       I->nr_srcs != 3 ||
       !agx3_map_atomic_op(I->atomic_opc, &d->op))
      return false;

   const unsigned value_channels =
      d->op == AGX3_ATOMIC_CMPXCHG ? 2 : 1;
   if (!agx3_atomic_plain_reg(I->src[0], AGX_SIZE_32, value_channels))
      return false;

   d->returns_value = !agx_is_null(I->dest[0]);
   if (d->returns_value) {
      if (!agx3_atomic_plain_reg(I->dest[0], AGX_SIZE_32, 1))
         return false;
      d->result = I->dest[0].value;
      d->slot = slot;
   } else if (slot != 0) {
      return false;
   }

   const agx_index index = I->src[2];
   if (index.type != AGX_INDEX_IMMEDIATE || index.value != 0 ||
       index.size != AGX_SIZE_16 || index.memory || index.abs || index.neg ||
       index.cache || index.discard || index.kill)
      return false;

   const agx_index base = I->src[1];
   if (base.size != AGX_SIZE_16 || base.memory || base.abs || base.neg)
      return false;
   if (base.type == AGX_INDEX_IMMEDIATE && base.value <= UINT16_MAX &&
       (base.value & 3) == 0 && !base.cache && !base.discard && !base.kill) {
      d->address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE;
      d->address = base.value;
   } else if (agx3_src_is_register(base) &&
              base.value <= (agx_encoding_is_g17_class(enc) ? 255 : 191) &&
              !(base.cache && base.discard)) {
      d->address_kind = AGX3_ATOMIC_INDEX_REGISTER;
      d->address = base.value;
      d->address_kill = base.kill;
   } else {
      return false;
   }

   d->value = I->src[0].value;
   d->value_kill = I->src[0].discard;
   d->wait_mask = wait_mask;
   return true;
}

/* Mesa's scalar 16-bit or 32-bit stack operation is an exact match for
 * Apple's static copy/rld family: the immediate byte offset names a TPS
 * half-register and the single data operand names one half or one aligned GPR
 * pair, at W16 or W32. Constant vectors are split after RA and dynamic reads
 * use LDTPS. Other modifiers stay on the fail-closed path. */
static bool
agx3_translate_static_spill(const agx_instr *I, bool is_store, uint8_t slot,
                            uint16_t wait_mask, bool tps_live,
                            struct agx3_static_spill_store_desc *store,
                            struct agx3_static_spill_load_desc *load)
{
   if (I->op != (is_store ? AGX_OPCODE_STACK_STORE : AGX_OPCODE_STACK_LOAD) ||
       I->nr_dests != (is_store ? 0 : 1) ||
       I->nr_srcs != (is_store ? 2 : 1) ||
       (I->format != AGX_FORMAT_I16 && I->format != AGX_FORMAT_I32) ||
       I->mask != 1)
      return false;

   const bool wide = I->format == AGX_FORMAT_I32;

   const agx_index data = is_store ? I->src[0] : I->dest[0];
   const agx_index offset = I->src[is_store ? 1 : 0];
   /*
    * A spill store's source is normally DEAD after it -- that is the whole
    * point of spilling -- so agx_opt_register_cache() marks it `.discard`,
    * and refusing that rejected essentially every real spill. f27, which
    * `src_kill` drives, is exactly that last-use bit: the same field the ALU
    * forms set for AGX3_ALU_HINT_DISCARD. Carry it rather than refuse it.
    *
    * Neither form has a cache field. Apple's reloads never set f11 even when
    * an ALU consumes the value next (60 live float4 temporaries under
    * applegpu_g17p), so a cached reload destination is the same instruction.
    * A cached spill source is still live after the copy, which the clear
    * last-use bit already preserves.
    */
   if (!agx3_src_is_register(data) || data.memory || data.abs || data.neg ||
       (data.cache && data.discard) || (!is_store && data.discard) ||
       data.size != (wide ? AGX_SIZE_32 : AGX_SIZE_16) ||
       agx_channels(data) != 1 || (wide && (data.value & 1)) ||
       data.value > (wide ? 254 : 255) ||
       (!is_store && data.kill) || offset.type != AGX_INDEX_IMMEDIATE ||
       /* abs on an immediate only marks the 16-bit form (offsets >= 256) */
       offset.memory || offset.neg || offset.cache ||
       offset.discard || offset.kill || offset.size != AGX_SIZE_16 ||
       agx_channels(offset) != 1 || (offset.value & (wide ? 3 : 1)))
      return false;

   const uint32_t tps_half = offset.value >> 1;
   if (tps_half + (wide ? 2 : 1) > AGX3_SLOT_TPS_HALF_TOKENS)
      return false;

   if (is_store) {
      if (store == NULL)
         return false;
      *store = (struct agx3_static_spill_store_desc){
         .tps_half = tps_half,
         .src = data.value,
         .width = wide ? AGX3_MEM_W32 : AGX3_MEM_W16,
         /* Both spellings of "dead after this use" drive the same f27. */
         .src_kill = data.kill || data.discard,
         .slot = slot,
         .wait_mask = wait_mask,
      };
   } else {
      if (load == NULL)
         return false;
      *load = (struct agx3_static_spill_load_desc){
         .dest = data.value,
         .tps_half = tps_half,
         .width = wide ? AGX3_MEM_W32 : AGX3_MEM_W16,
         .slot = slot,
         .slot_mirror = slot,
         .wait_mask = wait_mask,
         .tps_live = tps_live,
      };
   }

   return true;
}

/*
 * Mark the static reloads whose TPS cell may be read again.
 *
 * A reload's f27 ends the lifetime of the cell it reads. applegpu_g17p sets it
 * only when the cell is written, or never touched, before its next read:
 * across kernels of live ushort3, ushort and half temporaries, each of the 18
 * reloads followed by another reload of the same cell clears it, and all 530
 * other reloads set it. Mesa's spiller can reload one spilled value several times, so a reload
 * keeps its cell live whenever another reload of any of its halves comes
 * later in the linearized program, or when it sits in a loop, whose next
 * iteration may read the cell again. Later stores are not taken as ending
 * the lifetime: on another path the value may still be read.
 */
bool *
agx3_static_reload_rereads(agx_context *ctx, agx_instr **ordered,
                           unsigned count)
{
   unsigned blocks = 1;
   agx_foreach_block(ctx, block)
      blocks = MAX2(blocks, block->index + 1);

   bool *live = rzalloc_array(ctx, bool, count ? count : 1);
   bool *in_loop = rzalloc_array(ctx, bool, blocks);
   if (!live || !in_loop)
      return NULL;

   /* Structured loops are contiguous: a header and every block up to its
    * last backedge predecessor. */
   agx_foreach_block(ctx, header) {
      if (!header->loop_header)
         continue;
      unsigned last = header->index;
      agx_foreach_predecessor(header, pred)
         last = MAX2(last, (*pred)->index);
      agx_foreach_block(ctx, block) {
         if (block->index >= header->index && block->index <= last)
            in_loop[block->index] = true;
      }
   }

   uint32_t halves = 0;
   for (unsigned i = 0; i < count; ++i) {
      const agx_instr *I = ordered[i];
      if (I->op == AGX_OPCODE_STACK_LOAD &&
          I->src[0].type == AGX_INDEX_IMMEDIATE)
         halves = MAX2(halves, (I->src[0].value >> 1) + 2);
   }

   BITSET_WORD *read_later =
      rzalloc_array(ctx, BITSET_WORD, BITSET_WORDS(halves ? halves : 1));
   if (!read_later)
      return NULL;

   unsigned k = count;
   bool input_read_later[512] = {false};
   agx_foreach_block_rev(ctx, block) {
      agx_foreach_instr_in_block_rev(block, I) {
         if (I->op == AGX_OPCODE_STOP)
            continue;
         --k;
         if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
            if (I->nr_dests != 1 || I->imm >= 512 ||
                (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32) ||
                agx_channels(I->dest[0]) != 1)
               return NULL;
            unsigned n = agx_size_align_16(I->dest[0].size);
            if (n > 512 - I->imm)
               return NULL;
            live[k] = in_loop[block->index];
            for (unsigned h = I->imm; h < I->imm + n; ++h) {
               live[k] |= input_read_later[h];
               input_read_later[h] = true;
            }
            continue;
         }
         if (I->op != AGX_OPCODE_STACK_LOAD ||
             I->src[0].type != AGX_INDEX_IMMEDIATE)
            continue;

         const unsigned first = I->src[0].value >> 1;
         const unsigned n = I->format == AGX_FORMAT_I32 ? 2 : 1;
         bool reread = in_loop[block->index];
         for (unsigned h = first; h < first + n; ++h) {
            reread |= BITSET_TEST(read_later, h);
            BITSET_SET(read_later, h);
         }
         live[k] = reread;
      }
   }
   assert(k == 0);
   return live;
}

bool
agx3_dynamic_stack_load(const agx_instr *I)
{
   return I->op == AGX_OPCODE_STACK_LOAD && I->nr_srcs == 1 &&
          (I->src[0].type == AGX_INDEX_REGISTER ||
           (I->format == AGX_FORMAT_I8 && I->memory_is_packed));
}

static bool
agx3_translate_byte_stack(const agx_instr *I, bool store, uint8_t slot,
                           uint16_t wait, struct agx3_tps_desc *d)
{
   const agx_index data = store ? I->src[0] : I->dest[0];
   const agx_index index = I->src[store ? 1 : 0];
   const bool immediate = index.type == AGX_INDEX_IMMEDIATE;
   const unsigned halves = DIV_ROUND_UP(util_bitcount(I->mask), 2);
   if (I->format != AGX_FORMAT_I8 || !I->memory_is_packed ||
       !I->mask || (I->mask & ~15) || !agx3_tex_plain_reg(data) ||
       data.size != AGX_SIZE_16 || agx_channels(data) != halves ||
       data.value + halves > 256 || (store && data.cache && data.discard) ||
       (immediate ? index.value > UINT16_MAX || index.neg || index.memory ||
                      index.cache || index.discard || index.kill
                  : !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
                      agx_channels(index) != 1 || (index.value & 1) ||
                      index.value > 254 || (index.cache && index.discard)))
      return false;
   *d = (struct agx3_tps_desc){
      .data = data.value, .width = AGX3_MEM_W8, .mask = I->mask,
      .index = immediate ? 0 : index.value, .index_immediate = immediate,
      .index_kill = !immediate && index.discard,
      .byte_offset = immediate ? index.value : 0,
      .frame_region = 6, .value_kill = store && data.discard,
      .retain_tps = !store, .slot = store ? 0 : slot, .wait_mask = wait,
   };
   return true;
}

static bool
agx3_translate_dynamic_stack_load(const agx_instr *I, uint8_t slot,
                                  struct agx3_tps_desc *d)
{
   if (I->format == AGX_FORMAT_I8)
      return agx3_translate_byte_stack(I, false, slot, 0, d);

   /* A half vector covers whole registers and the same bytes as the word
    * load the private-array witness grounds: load the bits as words. Both
    * halves of every word must be selected, so only an even, contiguous
    * vector takes this form; an odd or sparse one stays refused. */
   if (I->format == AGX_FORMAT_I16) {
      if (!agx3_dynamic_stack_load(I) || I->nr_dests != 1 || !I->mask ||
          (I->mask & ~15))
         return false;
      const agx_index dst = I->dest[0], index = I->src[0];
      const unsigned channels = util_bitcount(I->mask);
      if (dst.size != AGX_SIZE_16 || agx_channels(dst) != channels ||
          I->mask != (1u << channels) - 1 || (channels & 1) ||
          !agx3_tex_plain_reg(dst) || (dst.value & 1) ||
          dst.value + channels > 256 ||
          !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
          agx_channels(index) != 1 || (index.value & 1) || index.value > 254 ||
          (index.cache && index.discard))
         return false;
      *d = (struct agx3_tps_desc){
         .data = dst.value, .width = AGX3_MEM_W32,
         .mask = (uint8_t)((1u << (channels / 2)) - 1),
         .index = index.value, .index_kill = index.discard,
         .frame_region = 1, .retain_tps = true, .slot = slot,
      };
      return true;
   }

   if (!agx3_dynamic_stack_load(I) || I->nr_dests != 1 ||
       I->format != AGX_FORMAT_I32 || !I->mask || (I->mask & ~15))
      return false;
   const agx_index dst = I->dest[0], index = I->src[0];
   if (!agx3_tex_plain_reg(dst) || dst.size != AGX_SIZE_32 ||
       agx_channels(dst) != util_bitcount(I->mask) || (dst.value & 1) ||
       dst.value + 2 * agx_channels(dst) > 256 ||
       !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
       agx_channels(index) != 1 || (index.value & 1) || index.value > 254 ||
       (index.cache && index.discard))
      return false;
   /* G16G native uint4 private-array witness: ldtps frame1, byte index,
    * mask15. Static COPY half8 and dynamic byte offset16 name the same data. */
   *d = (struct agx3_tps_desc){
      .data = dst.value, .width = AGX3_MEM_W32, .mask = I->mask,
      .index = index.value, .index_kill = index.discard,
      .frame_region = 1, .retain_tps = true, .slot = slot,
   };
   return true;
}

static bool
agx3_dynamic_stack_store(const agx_instr *I)
{
   return I->op == AGX_OPCODE_STACK_STORE && I->nr_srcs == 2 &&
          (I->src[1].type == AGX_INDEX_REGISTER ||
           (I->format == AGX_FORMAT_I8 && I->memory_is_packed));
}

/* G16G private arrays use STTPS frame 1 with a byte index. On M5, STTPS
 * completion is observed through slot 1: Apple's fixed-index array reads use
 * RLD with wait 1 after STTPS, and a byte-exact hardware replay requires that
 * wait to see the new value. Track this producer even though STTPS has no
 * explicit slot operand. */
static bool
agx3_translate_dynamic_stack_store(const agx_instr *I, uint16_t wait_mask,
                                   struct agx3_tps_desc *d)
{
   if (I->format == AGX_FORMAT_I8)
      return agx3_translate_byte_stack(I, true, 0, wait_mask, d);

   if (!agx3_dynamic_stack_store(I) || I->nr_dests != 0 ||
       I->format != AGX_FORMAT_I32 || !I->mask || (I->mask & ~15))
      return false;
   const agx_index value = I->src[0], index = I->src[1];
   if (!agx3_tex_plain_reg(value) || value.size != AGX_SIZE_32 ||
       agx_channels(value) != util_bitcount(I->mask) || (value.value & 1) ||
       value.value + 2 * agx_channels(value) > 256 ||
       (value.cache && value.discard) ||
       !agx3_tex_plain_reg(index) || index.size != AGX_SIZE_32 ||
       agx_channels(index) != 1 || (index.value & 1) || index.value > 254 ||
       (index.cache && index.discard))
      return false;
   *d = (struct agx3_tps_desc){
      .data = value.value, .width = AGX3_MEM_W32, .mask = I->mask,
      .index = index.value, .index_kill = index.discard,
      .frame_region = 1, .value_kill = value.discard, .slot = 0,
      .wait_mask = wait_mask,
   };
   return true;
}

bool
agx3_append_static_spill_tokens(const agx_instr *I,
                                struct agx3_slot_instr *slot_instr,
                                enum agx_isa_encoding enc)
{
   if (agx3_dynamic_stack_load(I)) {
      struct agx3_tps_desc desc;
      if (!agx3_translate_dynamic_stack_load(I, 1, &desc) ||
          slot_instr->nr_dsts >= AGX3_SLOT_MAX_REGS)
         return false;
      slot_instr->dst[slot_instr->nr_dsts++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
      return true;
   }
   if (agx3_dynamic_stack_store(I)) {
      /* M5 STTPS produces an implicit slot-1 completion. Preserve the
       * alias read as well, so earlier dynamic reads retire before a write. */
      struct agx3_tps_desc desc;
      if (!agx3_translate_dynamic_stack_store(I, 0, &desc) ||
          slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS)
         return false;
      slot_instr->src[slot_instr->nr_srcs++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
      if (enc == AGX_ISA_ENCODING_AGX3C) {
         if (slot_instr->nr_dsts >= AGX3_SLOT_MAX_REGS)
            return false;
         slot_instr->dst[slot_instr->nr_dsts++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
         slot_instr->preferred_slot = 1;
      }
      return true;
   }
   if (slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS)
      return false;
   /* A dynamically addressed read can alias any static write. */
   slot_instr->src[slot_instr->nr_srcs++] = AGX3_SLOT_DYNAMIC_TPS_TOKEN;
   const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
   struct agx3_static_spill_store_desc store;
   struct agx3_static_spill_load_desc load;
   if (!agx3_translate_static_spill(I, is_store, 1, 0, false, &store, &load))
      return false;

   const uint32_t tps_half = is_store ? store.tps_half : load.tps_half;
   const enum agx3_mem_width width = is_store ? store.width : load.width;
   for (unsigned i = 0; i < (width == AGX3_MEM_W32 ? 2 : 1); ++i) {
      if (slot_instr->nr_srcs >= AGX3_SLOT_MAX_REGS ||
          slot_instr->nr_dsts >= AGX3_SLOT_MAX_REGS)
         return false;
      const uint16_t token =
         (uint16_t)(AGX3_SLOT_TPS_TOKEN_BASE + tps_half + i);
      slot_instr->src[slot_instr->nr_srcs++] = token;
      slot_instr->dst[slot_instr->nr_dsts++] = token;
   }
   return true;
}

enum agx3_pack_result
agx3_pack_mesh_fence_instr(struct agx3_pack_state *st)
{
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_FENCE_UVB_GEN1) {
      if (!st->gen1_mesh || !st->mesh_allocated || I->nr_srcs || I->nr_dests ||
          st->slot || I->imm > AGX_MESH_EXPORT_FENCE_VERTEX_BATCH ||
          (I->imm == AGX_MESH_EXPORT_FENCE_INDICES
              ? st->mesh_indices_fenced : !st->mesh_indices_fenced))
         AGX3_REFUSE_RET(I);
      const uint16_t retire = st->wait | st->outstanding_here;
      const uint16_t phase = I->imm == AGX_MESH_EXPORT_FENCE_INDICES ? 2 : 4;
      const uint16_t src[] = {0, 0};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      /* Drain ALL modeled explicit producers, including any pending
       * writer of either half of r0, before this live identity reads
       * r0. The source-less AIR fence has drain semantics in the slot
       * pass; the emission ledger independently covers any residual.
       * Use the exact native bitop waits, without retire-noop's extra
       * slot6 wait. No comparison flag, nesting state or value changes.
       * Phase masks are the G17P exporter protocol at these source
       * boundaries, not universal UVST producer-slot assignments. */
      if (!agx3_alu_bit_reg_native(enc, AGX3_ALU_BAND, 0, src, hints,
                                  false, retire | phase, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding = 0;
      st->image_write_slots &= (uint16_t)~retire;
      agx3_staging_forget(&st->staging_banks, retire);
      st->mesh_exports_pending = false;
      if (I->imm == AGX_MESH_EXPORT_FENCE_INDICES)
         st->mesh_indices_fenced = true;
      else
         ++st->mesh_vertex_fences;
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}

enum agx3_pack_result
agx3_pack_mem_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_GET_SR) {
      struct agx3_sr_desc d;
      uint8_t bytes[AGX3_SR_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_sr(enc, I, st->slot, st->wait, &d))
         AGX3_REFUSE_RET(I);
      /* The coverage word is a fragment input. Apple's programs that
       * open with it put the entry marker on that read. */
      const bool coverage_input = ctx->stage == MESA_SHADER_FRAGMENT &&
         ((enc == AGX_ISA_ENCODING_AGX3B &&
           I->sr == AGX_SR_INPUT_SAMPLE_MASK) ||
          (ctx->key->dev.emit_agx3d &&
           I->sr == AGX_SR_INPUT_SAMPLE_MASK) ||
          (enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
           !ctx->key->fs.ignore_tib_dependencies &&
           (I->sr == AGX_SR_INPUT_SAMPLE_MASK ||
            I->sr == AGX_SR_COVERAGE_MASK)));
      d.first = coverage_input &&
                emission->size == st->initial_size &&
                !st->fragment_input_started &&
                !st->g17p_fragment_helper_protocol;
      if (!agx3_sr_encode(enc, &d, bytes, &len))
         AGX3_REFUSE_RET(I);
      memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
      st->fragment_input_started |= coverage_input || d.first;
      return AGX3_PACK_EMITTED;
   }

   /* Device memory. */
   if (I->op == AGX_OPCODE_DEVICE_LOAD ||
       I->op == AGX_OPCODE_DEVICE_STORE) {
      const bool is_store = I->op == AGX_OPCODE_DEVICE_STORE;
      struct agx3_mem_desc d;
      uint8_t bytes[AGX3_MEM_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_mem(enc, I, is_store, st->slot, st->wait, &d))
         AGX3_REFUSE_RET(I);
      if (st->g17p_vertex_load_policy && !is_store) {
         const bool first = st->vertex_loads_seen++ == 0;
         const bool last = --st->device_loads_left == 0;
         if (first && d.slot != 6)
            AGX3_REFUSE_RET(I);
         d.agx3b_last_load = last;
      } else if (ctx->key->dev.emit_agx3d && !is_store) {
         /* Apple's G17G compiler marks the final LDDEV of every
          * program regardless of stage (gpu/m5-xfb-alu-reference
          * and the fragment/vertex witnesses); the marked form is
          * the 14-byte ep2. Carried for G18G. */
         d.last_load = --st->device_loads_left == 0;
      }
      if (st->g17p_position_protocol && !is_store) {
         /* These assertions are the descriptor-level half of the shape
          * guard above. Refuse if a later lowering changes it. */
         if (d.data != 0 || d.width != AGX3_MEM_W32 || d.mask != 3 ||
             d.shift != 3 || !d.base_is_uniform || d.base != 0 ||
             d.idx_kind != AGX3_MEM_IDX_REG || d.idx_reg != 0 ||
             d.idx_signed || d.slot != 6 || d.wait_mask != 1 ||
             !d.agx3b_last_load)
            AGX3_REFUSE_RET(I);
         d.idx_kill = true;
      }
      const bool ok = is_store
                         ? agx3_mem_encode_store(enc, &d, bytes, &len)
                         : agx3_mem_encode_load(enc, &d, bytes, &len);
      if (!ok)
         AGX3_REFUSE_RET(I);
      memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ATOMIC) {
      struct agx3_device_atomic_desc d;
      uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH];
      uint8_t len = 0;
      const bool returned = I->nr_dests == 1 &&
                            !agx_is_null(I->dest[0]);
      const bool g17p_returned =
         enc == AGX_ISA_ENCODING_AGX3B && returned;
      const uint8_t atomic_slot = g17p_returned ? 1 : st->slot;
      /* A colourless fragment's returning atomic runs under pixel
       * class 4 and is followed by a barrier; see
       * agx3_pixel_atomic_acquire(). With a colour store, Apple takes
       * colour partition 0's write class instead, and the store then
       * acquires only its read class; that shared protocol is not
       * encoded, so colour-writing fragments keep neither. */
      const bool pixel_bracket = g17p_returned &&
         st->g17p_colourless_fragment;
      uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len = 0;
      /* The old value must be published above the spill frame */
      const unsigned return_tps_half =
         g17p_returned ? ctx->agx3_atomic_return_tps_half : 0;
      if ((g17p_returned &&
           (st->slot != 6 || return_tps_half * 2 < ctx->scratch_size_B)) ||
          !agx3_translate_device_atomic(I, enc, st->block->divergent,
                                        atomic_slot, st->wait, &d) ||
          (pixel_bracket &&
           (!agx3_pixel_atomic_acquire(enc, pixel, &pixel_len) ||
            !agx3_append(emission, pixel, pixel_len))))
         AGX3_REFUSE_RET(I);
      d.result_tps_half = return_tps_half;
      if (!agx3_atomic_encode_device(enc, &d, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);

      if (g17p_returned) {
         /* Exact applegpu_g17p return sequence. For destination r0,
          * this composes to 0c8019a500200000, the eight-byte native
          * rld immediately following atomicdev. */
         const struct agx3_static_spill_load_desc reload = {
            .dest = I->dest[0].value,
            .tps_half = return_tps_half,
            .width = AGX3_MEM_W32,
            .slot = 6,
            .slot_mirror = 6,
            .wait_mask = 1u << 0,
         };
         uint8_t reload_bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
         uint8_t reload_len = 0;
         if (!agx3_static_spill_encode_load_form(
                enc, &reload, 1, reload_bytes, &reload_len) ||
             !agx3_append(emission, reload_bytes, reload_len) ||
             (pixel_bracket &&
              (!agx3_pixel_atomic_barrier(enc, pixel, &pixel_len) ||
               !agx3_append(emission, pixel, pixel_len))))
            AGX3_REFUSE_RET(I);
      }
      return AGX3_PACK_EMITTED;
   }

   /* Threadgroup memory. AGX3 preprocessing has converted the byte
    * address to the element index consumed by the native ldtg/sttg
    * forms. */
   if (I->op == AGX_OPCODE_LOCAL_LOAD ||
       I->op == AGX_OPCODE_LOCAL_STORE) {
      const bool is_store = I->op == AGX_OPCODE_LOCAL_STORE;
      struct agx3_local_desc d;
      uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_local(enc, I, is_store, st->slot, st->wait, &d))
         AGX3_REFUSE_WHY_RET(I, "local: translate declined (operand shape)");
      const bool ok =
         is_store ? agx3_local_encode_store(enc, &d, bytes, &len)
                  : agx3_local_encode_load(enc, &d, bytes, &len);
      if (!ok)
         AGX3_REFUSE_WHY_RET(I, "local: no grounded form for this "
                            "width/mask/base combination");
      memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_LOCAL_ATOMIC) {
      struct agx3_local_atomic_desc d;
      uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_local_atomic(I, enc, st->slot, st->wait, &d) ||
          !(agx_is_g18g(&ctx->key->dev)
               ? agx3_atomic_encode_local_g18g(enc, &d, bytes, &len)
               : agx3_atomic_encode_local(enc, &d, bytes, &len)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 ||
       I->op == AGX_OPCODE_CONTINUE_RAY_GEN1) {
      uint8_t bytes[16], len;
      if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_dests || st->slot)
         AGX3_REFUSE_RET(I);
      if (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait))
         AGX3_REFUSE_RET(I);
      if (I->op == AGX_OPCODE_INTERSECT_RAY_GEN1) {
         agx_index bvh=I->src[0], request=I->src[1];
         bvh.kill=bvh.cache=bvh.discard=false;
         request.kill=request.cache=request.discard=false;
         if (I->nr_srcs != 4 || bvh.size != AGX_SIZE_64 ||
             request.size != AGX_SIZE_64 ||
             !agx3_scs_plain_index(bvh) || !agx3_scs_plain_index(request))
            AGX3_REFUSE_RET(I);
         const struct agx3_rt_gen1_intersect_desc d = {
            .bvh=bvh.value, .request=request.value,
            .bvh_uniform=bvh.type==AGX_INDEX_UNIFORM,
            .request_uniform=request.type==AGX_INDEX_UNIFORM,
            .context_uniform=I->imm,
         };
         if (!agx3_rt_gen1_encode_intersect(enc,&d,bytes,&len))
            AGX3_REFUSE_RET(I);
         for (unsigned j=0;j<2;j++) {
            agx_index word=I->src[2+j];
            uint8_t copy[16], n;
            if (!agx3_src_is_register(word) || word.size!=AGX_SIZE_32 ||
                word.memory || word.abs || word.neg || (word.value&1) ||
                !agx3_rt_gen1_encode_launch_word(j,word.value,copy,&n) ||
                !agx3_append(emission,copy,n))
               AGX3_REFUSE_RET(I);
         }
         ctx->agx3_staging_halfs=MAX2(ctx->agx3_staging_halfs,4);
      } else {
         if (I->nr_srcs != 1) AGX3_REFUSE_RET(I);
         agx_index id = I->src[0];
         uint8_t resume[32], resume_len;
         if (!agx3_src_is_register(id) || id.size != AGX_SIZE_32 ||
             id.memory || id.abs || id.neg || (id.value & 1) ||
             !(I->imm == 0 ?
               agx3_rt_gen1_encode_resume(enc, id.value, resume, &resume_len) :
               I->imm == 1 && agx3_rt_gen1_encode_refill(enc, id.value, resume, &resume_len)) ||
             !agx3_append(emission, resume, resume_len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 6);
         return AGX3_PACK_EMITTED;
      }
      uint8_t fence[AGX3_RT_CORE_MAX_LENGTH], fence_len;
      if (!agx3_rt_traversal_barrier(enc,
            I->op == AGX_OPCODE_INTERSECT_RAY_GEN1 ? AGX3_RT_BARRIER_ENTRY
                                                 : AGX3_RT_BARRIER_IN_LOOP,
            fence, &fence_len) || !agx3_append(emission,fence,fence_len))
         AGX3_REFUSE_RET(I);
      if (st->fragment_ray_query && !st->ray_query_acquired) {
         uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len;
         const bool ok = st->g17p_colourless_fragment
            ? agx3_pixel_atomic_acquire(enc, pixel, &pixel_len)
            : agx3_pixel_wait(enc, 4, 0, pixel, &pixel_len);
         if (!ok || !agx3_append(emission, pixel, pixel_len)) AGX3_REFUSE_RET(I);
         st->ray_query_acquired = true;
         ctx->agx3_colour_link.rt_query_write_acquired = !st->g17p_colourless_fragment;
      }
      if (!agx3_append(emission,bytes,len)) AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_RT_BARRIER_GEN1) {
      /* Nothing stays in flight across the fence: what follows it has
       * no scoreboard slot and nothing can wait on it afterwards. */
      const uint16_t retire = st->wait | st->outstanding_here;
      uint8_t fence[AGX3_RT_CORE_MAX_LENGTH], fence_len;
      if (enc != AGX_ISA_ENCODING_AGX3B || I->nr_dests || I->nr_srcs ||
          st->slot ||
          (retire && !agx3_retire_slots_noop(emission, enc, retire)) ||
          !agx3_rt_traversal_barrier(
             enc, (enum agx3_rt_barrier_scope)I->imm, fence,
             &fence_len) ||
          !agx3_append(emission, fence, fence_len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= (uint16_t)~retire;
      st->image_write_slots &= (uint16_t)~retire;
      agx3_staging_forget(&st->staging_banks, retire);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_SCS_TOUCH_REGISTER_INDEXED_AGX3 ||
       I->op == AGX_OPCODE_SCS_TOUCH_REGISTER_BASE_AGX3) {
      /* No destination and no slot. The form's wait fields are not
       * established, so its address registers are completed first. */
      struct agx3_scs_touch_desc d;
      uint8_t bytes[AGX3_SCS_MAX_LENGTH], len = 0;
      if (st->slot || !agx3_translate_scs_touch(enc, I, &d) ||
          !agx3_scs_encode_touch(enc, &d, bytes, &len) ||
          (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ENTER_QUERY_M5) {
      /* The native entry writes a 32-bit query pointer to its f2
       * register (r92/r94 in the native witnesses). It must be a real,
       * allocated destination: the old fixed encoding wrote r0 behind
       * the register allocator's back. */
      uint8_t enter[16], length;
      if (I->nr_dests != 1 || I->nr_srcs ||
          !agx3_src_is_register(I->dest[0]) ||
          I->dest[0].size != AGX_SIZE_32 || I->dest[0].memory ||
          I->dest[0].abs || I->dest[0].neg || st->slot ||
          !agx3_rt_m5_encode_enter_query(enc, I->dest[0].value, I->imm,
                                         enter, &length) ||
          (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait)) ||
          !agx3_append(emission, enter, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_INTERSECT_RAY_M5 ||
       I->op == AGX_OPCODE_CONTINUE_RAY_M5) {
      uint8_t bytes[16], len = 0;
      if (enc != AGX_ISA_ENCODING_AGX3C || I->nr_dests || st->slot)
         AGX3_REFUSE_RET(I);
      if (I->op == AGX_OPCODE_INTERSECT_RAY_M5) {
         if (I->nr_srcs != 5)
            AGX3_REFUSE_RET(I);
         agx_index address = I->src[0], bvh = I->src[1];
         const bool bvh_kill = bvh.kill || bvh.discard;
         address.kill = address.cache = address.discard = false;
         bvh.kill = bvh.cache = bvh.discard = false;
         if (!agx3_src_is_register(address) ||
             address.size != AGX_SIZE_64 ||
             !agx3_scs_plain_index(address) ||
             (bvh.type != AGX_INDEX_UNIFORM &&
              !agx3_src_is_register(bvh)) ||
             bvh.size != AGX_SIZE_64 || !agx3_scs_plain_index(bvh))
            AGX3_REFUSE_RET(I);
         const struct agx3_rt_m5_intersect_desc d = {
            .address_register = address.value,
            .bvh_base = bvh.value,
            .bvh_uniform = bvh.type == AGX_INDEX_UNIFORM,
            .bvh_kill = bvh_kill,
         };
         if (!agx3_rt_m5_encode_intersect(enc, &d, bytes, &len))
            AGX3_REFUSE_RET(I);
         /* Native M5 main +0x4a8/+0x4c4/+0x4cc publishes three
          * launch words before traversal. GPR values and the SCS packet
          * alone do not provide these staging operands. Keep publication
          * fused with its consumer so texture staging cannot intervene.
          */
         if (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait))
            AGX3_REFUSE_RET(I);
         for (unsigned w = 0; w < 3; ++w) {
            agx_index word = I->src[2 + w];
            if (!agx3_src_is_register(word) || word.size != AGX_SIZE_32 ||
                word.memory || word.abs || word.neg || (word.value & 1))
               AGX3_REFUSE_RET(I);
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], copy_len;
            if (!agx3_stage_copy_liveness(enc, 2 * w, word.value,
                   false, word.discard, 1, 0, copy, &copy_len) ||
                !agx3_append(emission, copy, copy_len))
               AGX3_REFUSE_RET(I);
         }
         const uint8_t publish[] = {0x17, 0x11, 0, 6, 0, 0};
         if (!agx3_append(emission, publish, sizeof(publish)))
            AGX3_REFUSE_RET(I);
         if (st->fragment_ray_query && !st->ray_query_acquired) {
            uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len;
            const bool ok = st->g17p_colourless_fragment
               ? agx3_pixel_atomic_acquire(enc, pixel, &pixel_len)
               : agx3_pixel_wait(enc, 4, 0, pixel, &pixel_len);
            if (!ok || !agx3_append(emission, pixel, pixel_len))
               AGX3_REFUSE_RET(I);
            st->ray_query_acquired = true;
            ctx->agx3_colour_link.rt_query_write_acquired = !st->g17p_colourless_fragment;
         }
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 6);
      } else {
         if (I->nr_srcs != 1 ||
             !agx3_rt_m5_encode_continue(enc, bytes, &len))
            AGX3_REFUSE_RET(I);
         agx_index resident_id = I->src[0];
         if (!agx3_src_is_register(resident_id) ||
             resident_id.size != AGX_SIZE_32 || resident_id.memory ||
             resident_id.abs || resident_id.neg || (resident_id.value & 1))
            AGX3_REFUSE_RET(I);
         /* Native query main +0x658 stages SR59 at half8 before
          * continue_ray consumes it. Staging0/2/4 belongs to the
          * initial intersection and cannot identify a continuation. */
         if (st->wait && !agx3_retire_slots_noop(emission, enc, st->wait))
            AGX3_REFUSE_RET(I);
         st->wait = 0;
         uint8_t copy[AGX3_STAGE_MAX_LENGTH], copy_len;
         if (!agx3_stage_copy_liveness(enc, 8, resident_id.value,
                false, resident_id.discard, 1, 0, copy, &copy_len) ||
             !agx3_append(emission, copy, copy_len))
            AGX3_REFUSE_RET(I);
         ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 10);
      }
      if (I->op == AGX_OPCODE_CONTINUE_RAY_M5) {
         /* Publish the resident ID and status before resuming. */
         const uint8_t resume_barrier[] = {0x17, 0x09, 0, 6, 0, 0};
         if (!agx3_append(emission, resume_barrier, sizeof(resume_barrier)))
            AGX3_REFUSE_RET(I);
      }
      /* Native RT forms have no ordinary scoreboard wait field.
       * Retire only their actual input dependencies. The following
       * query-status loads and continuation remain explicit IR. */
      if ((I->op == AGX_OPCODE_CONTINUE_RAY_M5 && st->wait &&
           !agx3_retire_slots_noop(emission, enc, st->wait)) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ALLOCATE_UVB_M5 ||
       I->op == AGX_OPCODE_STORE_UVB_M5) {
      uint8_t bytes[8], length;
      bool encoded;
      if (I->nr_dests || st->slot || (enc != AGX_ISA_ENCODING_AGX3C &&
                                enc != AGX_ISA_ENCODING_AGX3B))
         AGX3_REFUSE_RET(I);
      if (I->op == AGX_OPCODE_ALLOCATE_UVB_M5) {
         if (I->nr_srcs != 2 ||
             !agx3_src_is_register(I->src[0]) ||
             !agx3_src_is_register(I->src[1]) ||
             I->src[0].size != AGX_SIZE_16 ||
             I->src[1].size != AGX_SIZE_16 ||
             agx_channels(I->src[0]) != 1 ||
             agx_channels(I->src[1]) != 1 ||
             I->src[0].memory || I->src[1].memory ||
             I->src[0].abs || I->src[1].abs ||
             I->src[0].neg || I->src[1].neg)
            AGX3_REFUSE_RET(I);
         encoded = agx3_mesh_allocate_uvb(enc, I->src[0].value,
                                          I->src[1].value, I->src[0].kill || I->src[0].discard,
                                          bytes, &length);
      } else {
         if (I->nr_srcs != 1 || I->imm > 1 ||
             !agx3_src_is_register(I->src[0]) ||
             I->src[0].size != AGX_SIZE_32 ||
             agx_channels(I->src[0]) != 2 ||
             I->src[0].memory || I->src[0].abs || I->src[0].neg)
            AGX3_REFUSE_RET(I);
         encoded = agx3_mesh_uvst(enc, I->src[0].value, I->imm,
                                   I->src[0].kill || I->src[0].discard,
                                   bytes, &length);
      }
      /* allocate_uvb completes on scoreboard slot 1: every native
       * write_uvb helper waits slot 1 on the instruction right after it
       * (cndst f4: 1e00002e on G17G, 0f155401 on G16S). UVSTs issued
       * before that completion lost the exports of alternate mesh
       * workgroups on M5 (workgroups 2, 4, 6 of 8) and the later
       * vertices of some workgroups on G16S. Retire any older slot-1
       * generation before the allocation claims the slot, then wait
       * for the allocation itself. */
      const bool alloc_slot = I->op == AGX_OPCODE_ALLOCATE_UVB_M5;
      if (st->gen1_mesh) {
         if (alloc_slot) {
            if (st->mesh_allocated)
               AGX3_REFUSE_RET(I);
            st->mesh_allocated = true;
         } else {
            if (!st->mesh_allocated || st->outstanding_here ||
                (I->imm ? st->mesh_indices_fenced : !st->mesh_indices_fenced))
               AGX3_REFUSE_RET(I);
            st->mesh_exports_pending = true;
         }
      }
      const uint16_t pre_wait = alloc_slot ? (st->wait | (st->outstanding_here & 1u)) : st->wait;
      if (!encoded || (pre_wait && !agx3_retire_slots_noop(emission, enc, pre_wait)) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      if (alloc_slot) {
         if (!agx3_retire_slots_noop(emission, enc, 1u))
            AGX3_REFUSE_RET(I);
         st->outstanding &= (uint16_t)~1u;
      }
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_STORE_OBJECT_CHILDCOUNT_M5) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], length;
      if ((enc != AGX_ISA_ENCODING_AGX3C && enc != AGX_ISA_ENCODING_AGX3B) ||
          I->nr_dests || I->nr_srcs != 1 ||
          !agx3_vary_reg32(I->src[0]))
         AGX3_REFUSE_RET(I);
      const uint16_t stage_wait = st->wait | (st->outstanding & 1u);
      if (!agx3_stage_copy_source(enc, 0, I->src[0].value,
           I->src[0].kill || I->src[0].discard, 1, stage_wait, bytes, &length) ||
          !agx3_append(emission, bytes, length) ||
          !agx3_mesh_object_childcount(enc, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      ctx->agx3_staging_halfs = MAX2(ctx->agx3_staging_halfs, 2);
      st->outstanding &= (uint16_t)~(stage_wait | 1u);
      return AGX3_PACK_EMITTED;
   }
   if (I->op == AGX_OPCODE_SCS_LOAD_AGX3 ||
       I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
       I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
       I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3 ||
       I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
       I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3 ||
       I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
       I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3) {
      const bool is_store = I->op == AGX_OPCODE_SCS_STORE_AGX3 ||
                            I->op ==
                               AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3 ||
                            I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3 ||
                            I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3;
      const bool register_indexed =
         I->op == AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3 ||
         I->op == AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3;
      const bool register_base =
         I->op == AGX_OPCODE_SCS_LOAD_REGISTER_BASE_AGX3 ||
         I->op == AGX_OPCODE_SCS_STORE_REGISTER_BASE_AGX3;
      const bool indexed = register_base || register_indexed ||
         I->op == AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3 ||
         I->op == AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3;
      struct agx3_scs_desc d;
      uint8_t bytes[AGX3_SCS_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_scs(enc, I, is_store, indexed, register_indexed,
                              register_base, st->slot, st->wait, &d))
         AGX3_REFUSE_RET(I);
      const bool ok =
         is_store ? agx3_scs_encode_store(enc, &d, bytes, &len)
                  : agx3_scs_encode_load(enc, &d, bytes, &len);
      if (!ok)
         AGX3_REFUSE_RET(I);
      memcpy(util_dynarray_grow_bytes(emission, 1, len), bytes, len);
      return AGX3_PACK_EMITTED;
   }

   if (agx3_dynamic_stack_load(I)) {
      struct agx3_tps_desc desc;
      uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH], len;
      /* The address is unknown at compile time. Complete outstanding
       * static writes before reading; retain this read's alias token
       * until later static accesses have waited for it. */
      const uint16_t retire = st->wait | st->outstanding_here;
      if (!agx3_translate_dynamic_stack_load(I, st->slot, &desc) ||
          (retire && !agx3_retire_slots_noop(emission, enc, retire)) ||
          !agx3_tps_encode_load(enc, &desc, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= ~retire;
      st->outstanding |= 1u << (st->slot - 1);
      return AGX3_PACK_EMITTED;
   }

   if (agx3_dynamic_stack_store(I)) {
      struct agx3_tps_desc desc;
      uint8_t bytes[AGX3_LOCAL_TPS_MAX_LENGTH], len;
      /* The address is unknown at compile time: every outstanding
       * static copy or reload may touch the same location, so the
       * store waits for all of them in its own wait field (Apple's
       * STTPS carries the wait of the preceding static copies). */
      const uint16_t retire = st->wait | st->outstanding_here;
      if (!agx3_translate_dynamic_stack_store(I, retire, &desc) ||
          !agx3_tps_encode_store(enc, &desc, bytes, &len) ||
          !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      st->outstanding &= ~retire;
      if (enc == AGX_ISA_ENCODING_AGX3C) {
         assert(st->slot == 1);
         st->outstanding |= 1u;
      }
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_LOAD_CPROG_AGX3) {
      if (!st->cprog_imports || I->nr_dests != 1 || I->nr_srcs ||
          I->saturate || I->imm >= 512 ||
          !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          I->dest[0].abs || I->dest[0].neg || I->dest[0].kill ||
          I->dest[0].discard || agx_channels(I->dest[0]) != 1 ||
          (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32))
         AGX3_REFUSE_RET(I);
      const struct agx3_cprog_import import = {
         .uniform_half = I->imm, .dest_half = I->dest[0].value,
         .width = I->dest[0].size == AGX_SIZE_16 ? AGX3_MEM_W16 : AGX3_MEM_W32,
         .slot = st->slot, .wait_mask = st->wait,
         .tps_live = st->tps_rereads[st->instr_index],
      };
      uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
      size_t length;
      if (!agx3_cprog_input_import_encoding(enc, &ctx->out->agx3_preamble_inputs,
                                  &import, bytes, sizeof(bytes), &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Static TPS spill/reload. Both copy and rld signal scoreboard slots;
    * their synthetic TPS dependency tokens make a reload wait the copy
    * that produced the same stack location. */
   if (I->op == AGX_OPCODE_STACK_LOAD ||
       I->op == AGX_OPCODE_STACK_STORE) {
      const bool is_store = I->op == AGX_OPCODE_STACK_STORE;
      struct agx3_static_spill_store_desc store;
      struct agx3_static_spill_load_desc load;
      uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
      uint8_t len = 0;
      if (!agx3_translate_static_spill(I, is_store, st->slot, st->wait,
                                       st->tps_rereads[st->instr_index], &store,
                                       &load))
         AGX3_REFUSE_RET(I);
      const bool ok =
         is_store ? agx3_static_spill_encode_store(enc, &store, bytes,
                                                    &len)
                  : agx3_static_spill_encode_load(enc, &load, bytes,
                                                  &len);
      if (!ok || !agx3_append(emission, bytes, len))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}
