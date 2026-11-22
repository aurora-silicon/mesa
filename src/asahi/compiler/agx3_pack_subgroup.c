/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 subgroup shuffles/broadcasts/reductions/ballots, scoped barriers, and
 * matrix/coopmat instructions.
 */

#include "agx3_pack_internal.h"
#include "agx3_matrix_core.h"
#include "agx3_subgroup_core.h"

/* The 16-bit counterpart of agx3_subgroup_plain(), in half units. */
static bool
agx3_subgroup_plain_half(agx_index index)
{
   return index.type == AGX_INDEX_REGISTER && !index.memory &&
          index.size == AGX_SIZE_16 && agx_channels(index) == 1 &&
          !index.abs && !index.neg && !index.kill &&
          !(index.cache && index.discard) && index.value < 256;
}

/* A plain immediate operand: an unmodified single-channel constant. */
static bool
agx3_subgroup_imm_lane(agx_index x)
{
   return x.type == AGX_INDEX_IMMEDIATE && !x.memory && !x.abs && !x.neg &&
          agx_channels(x) == 1;
}

static bool
agx3_emit_simd_broadcast(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t wait_mask, bool g18g)
{
   if (I->op != AGX_OPCODE_SHUFFLE || I->nr_dests != 1 || I->nr_srcs != 2 ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_imm_lane(I->src[1]) || I->saturate)
      return false;

   /* G17P puts the lane straight into f29 for every lane the field holds, so
    * route AGX3b through the bounded immediate encoder and let it carry the
    * producer's wait. See agx3_subgroup_core.c for Apple's per-lane sweep. */
   if (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         /* Every other shuffle site publishes this; leaving it out here left
          * the field defaulted to "final read" on EVERY immediate-lane
          * broadcast, and the hardware then destroyed the source register. */
         .src_live = !I->src[0].discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return (g18g
               ? agx3_subgroup_encode_shuffle_g18g
               : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   /* AGX3c keeps its own two pinned broadcast words; nothing else about that
    * encoding has been witnessed, so it stays where it was. */
   /* G18G: the two legacy lane0/3 words fix the source to its final read and
    * cannot carry a destination hint. Preserve live sources and cached
    * destinations through the already bounded immediate-lane encoder. */
   if (enc == AGX_ISA_ENCODING_AGX3C && g18g &&
       (wait_mask != 0 || !I->src[0].discard || I->dest[0].cache ||
        (I->src[1].value != 0 && I->src[1].value != 3))) {
      const struct agx3_subgroup_shuffle_desc d = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .dest_hint = I->dest[0].cache,
         .src_live = !I->src[0].discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      uint8_t retire_bytes[AGX3_ALU_MAX_LENGTH], retire_length = 0;
      return (g18g
               ? agx3_subgroup_encode_shuffle_g18g
               : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
             (!wait_mask ||
              (agx3_band_carrier(enc, I->dest[0].value, wait_mask,
                                 retire_bytes, &retire_length) &&
               agx3_append(emission, retire_bytes, retire_length))) &&
             agx3_append(emission, bytes, length);
   }
   if (wait_mask != 0 ||
       (I->src[1].value != 0 && I->src[1].value != 3))
      return false;

   struct agx3_encoded_simd_broadcast_f32_imm encoded;
   return agx3_pack_simd_broadcast_f32_imm(
             enc, I->dest[0].value, I->src[0].value, I->src[1].value,
             &encoded) &&
          agx3_append(emission, encoded.bytes, encoded.length);
}

static bool
agx3_map_shuffle(unsigned op, bool *quad,
                 enum agx3_subgroup_shuffle_op *shuffle)
{
   *quad = false;
   switch (op) {
   case AGX_OPCODE_SHUFFLE:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
      return true;
   case AGX_OPCODE_SHUFFLE_XOR:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_XOR;
      return true;
   case AGX_OPCODE_SHUFFLE_UP:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_UP;
      return true;
   case AGX_OPCODE_SHUFFLE_DOWN:
      *shuffle = AGX3_SUBGROUP_SHUFFLE_DOWN;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_BROADCAST;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_XOR:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_XOR;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_UP:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_UP;
      return true;
   case AGX_OPCODE_QUAD_SHUFFLE_DOWN:
      *quad = true;
      *shuffle = AGX3_SUBGROUP_SHUFFLE_DOWN;
      return true;
   default:
      return false;
   }
}

static bool
agx3_emit_subgroup_shuffle(struct util_dynarray *emission,
                           enum agx_isa_encoding enc, const agx_instr *I,
                           uint16_t wait_mask, bool g18g)
{
   bool quad;
   enum agx3_subgroup_shuffle_op op;
   if (!agx3_map_shuffle(I->op, &quad, &op))
      return false;

   /* The SIMD shuffles from a register lane at either data width, and the
    * 16-bit broadcast from an immediate lane, carry the operands' lifetime
    * hints and the source wait. The lane is read from the low half of its
    * register. */
   const agx_index lane = I->nr_srcs == 2 ? I->src[1] : agx_null();
   const bool register_lane =
      lane.type == AGX_INDEX_REGISTER && !lane.memory &&
      (lane.size == AGX_SIZE_16 || lane.size == AGX_SIZE_32) &&
      agx_channels(lane) == 1 && !lane.abs && !lane.neg && !lane.kill &&
      !(lane.cache && lane.discard) &&
      lane.value < (enc == AGX_ISA_ENCODING_AGX3C ? 256 : 192);
   /* G18G keeps j873's register-lane forms, quad ones included. */
   if ((enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
       (g18g ? register_lane && I->nr_dests == 1 &&
                  I->dest[0].size == AGX_SIZE_32
             : !quad) && I->nr_dests == 1 &&
       I->nr_srcs == 2 && !I->saturate &&
       (register_lane ||
        (I->op == AGX_OPCODE_SHUFFLE && I->dest[0].size == AGX_SIZE_16))) {
      const agx_index dest = I->dest[0], src = I->src[0];
      const bool half = dest.size == AGX_SIZE_16;
      if (!(half ? agx3_subgroup_plain_half(dest) &&
                      agx3_subgroup_plain_half(src)
                 : agx3_subgroup_plain(dest, AGX_INDEX_REGISTER) &&
                      agx3_subgroup_plain(src, AGX_INDEX_REGISTER)) ||
          !(register_lane || agx3_subgroup_imm_lane(lane)))
         return false;

      const struct agx3_subgroup_shuffle_desc d = {
         .quad = quad,
         .op = op,
         .lane_kind = register_lane ? AGX3_SUBGROUP_LANE_REGISTER
                                    : AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = dest.value,
         .src = src.value,
         .lane = lane.value,
         .wait_mask = wait_mask,
         .dest_hint = dest.cache,
         .src_live = !src.discard,
         .half = half,
         .src_cache = src.cache,
         .src_discard = src.discard,
         .lane_cache = register_lane && lane.cache,
         .lane_discard = register_lane && lane.discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return (g18g
               ? agx3_subgroup_encode_shuffle_g18g
               : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (I->op == AGX_OPCODE_SHUFFLE && I->nr_srcs == 2 &&
       I->src[1].type == AGX_INDEX_IMMEDIATE)
      return agx3_emit_simd_broadcast(emission, enc, I, wait_mask, g18g);

   /* Every immediate-lane shuffle -- quad or SIMD, broadcast, butterfly or
    * shift -- is the same encoding point with the lane in f29, and Apple's
    * own words for all six mnemonics carry the producer wait in the f44 bank.
    * Route them all here rather than pushing the three non-broadcast SIMD
    * shapes down to the register-lane path, which cannot take a wait and so
    * had to spend a whole synchronous prelude retiring one. */
   if (I->nr_srcs == 2 && agx3_subgroup_imm_lane(I->src[1]) &&
       (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C)) {
      /* 16-bit data narrows this encoding point's width pair, a field
       * separate from the quad selector, so a half broadcast or butterfly
       * encodes for quad as for SIMD (agx3_subgroup_encode_shuffle()). The
       * shifts keep the 32-bit predicate. */
      const bool half = I->nr_dests == 1 &&
                        I->dest[0].size == AGX_SIZE_16 &&
                        (op == AGX3_SUBGROUP_SHUFFLE_BROADCAST ||
                         op == AGX3_SUBGROUP_SHUFFLE_XOR);
      if (I->nr_dests != 1 || I->saturate ||
          !(half ? agx3_subgroup_plain_half(I->dest[0]) &&
                      agx3_subgroup_plain_half(I->src[0])
                 : agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) &&
                      agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER)))
         return false;

      const struct agx3_subgroup_shuffle_desc d = {
         .quad = quad,
         .op = op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         .src_live = !I->src[0].discard,
         .half = half,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return (g18g
               ? agx3_subgroup_encode_shuffle_g18g
               : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (quad && I->nr_srcs == 2 && I->src[1].type == AGX_INDEX_IMMEDIATE) {
      if (I->nr_dests != 1 || I->saturate ||
          !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
          !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER) ||
          !agx3_subgroup_imm_lane(I->src[1]))
         return false;

      /* AGX3c has no witnessed wait on this form: settle any producer first. */
      if (op != AGX3_SUBGROUP_SHUFFLE_BROADCAST && wait_mask != 0) {
         if (!agx3_retire_result_wait(emission, enc, I, wait_mask))
            return false;
         wait_mask = 0;
      }

      const struct agx3_subgroup_shuffle_desc d = {
         .quad = true,
         .op = op,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = I->dest[0].value,
         .src = I->src[0].value,
         .lane = I->src[1].value,
         .wait_mask = wait_mask,
         .dest_hint = I->dest[0].cache,
         .src_live = !I->src[0].discard,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return (g18g
               ? agx3_subgroup_encode_shuffle_g18g
               : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       !agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER))
      return false;

   enum agx3_subgroup_lane_kind lane_kind;
   if (agx3_subgroup_plain(I->src[1], AGX_INDEX_REGISTER))
      lane_kind = AGX3_SUBGROUP_LANE_REGISTER;
   else if (agx3_subgroup_plain(I->src[1], AGX_INDEX_UNIFORM))
      lane_kind = AGX3_SUBGROUP_LANE_UNIFORM;
   else
      return false;

   /* The register/uniform-lane native shuffle forms have no witnessed
    * scoreboard field.  Public shuffles commonly consume values directly
    * from an asynchronous SSBO/device load, so retire only their concrete
    * producer slots through the same synchronous identity prelude used by
    * other no-wait subgroup forms. */
   if (wait_mask != 0 &&
       !agx3_retire_result_wait(emission, enc, I, wait_mask))
      return false;

   const struct agx3_subgroup_shuffle_desc d = {
      .quad = quad,
      .op = op,
      .lane_kind = lane_kind,
      .dest = I->dest[0].value,
      .src = I->src[0].value,
      .lane = I->src[1].value,
      .wait_mask = 0,
      .dest_hint = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .lane_live = !I->src[1].discard,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return (g18g
            ? agx3_subgroup_encode_shuffle_g18g
            : agx3_subgroup_encode_shuffle)(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_map_reduce_op(enum agx_simd_op in, enum agx3_subgroup_reduce_op *out)
{
   switch (in) {
   case AGX_SIMD_OP_AND:  *out = AGX3_SUBGROUP_REDUCE_AND; return true;
   case AGX_SIMD_OP_FADD: *out = AGX3_SUBGROUP_REDUCE_FADD; return true;
   case AGX_SIMD_OP_OR:   *out = AGX3_SUBGROUP_REDUCE_OR; return true;
   case AGX_SIMD_OP_FMUL: *out = AGX3_SUBGROUP_REDUCE_FMUL; return true;
   case AGX_SIMD_OP_XOR:  *out = AGX3_SUBGROUP_REDUCE_XOR; return true;
   case AGX_SIMD_OP_FMIN: *out = AGX3_SUBGROUP_REDUCE_FMIN; return true;
   case AGX_SIMD_OP_FMAX: *out = AGX3_SUBGROUP_REDUCE_FMAX; return true;
   case AGX_SIMD_OP_IADD: *out = AGX3_SUBGROUP_REDUCE_IADD; return true;
   case AGX_SIMD_OP_SMIN: *out = AGX3_SUBGROUP_REDUCE_SMIN; return true;
   case AGX_SIMD_OP_SMAX: *out = AGX3_SUBGROUP_REDUCE_SMAX; return true;
   case AGX_SIMD_OP_UMIN: *out = AGX3_SUBGROUP_REDUCE_UMIN; return true;
   case AGX_SIMD_OP_UMAX: *out = AGX3_SUBGROUP_REDUCE_UMAX; return true;
   default: return false;
   }
}

static bool
agx3_emit_subgroup_reduce(struct util_dynarray *emission,
                          enum agx_isa_encoding enc, const agx_instr *I,
                          uint16_t wait_mask, bool g18g)
{
   const bool quad = I->op == AGX_OPCODE_QUAD_REDUCE ||
                     I->op == AGX_OPCODE_QUAD_PREFIX;
   const bool prefix = I->op == AGX_OPCODE_SIMD_PREFIX ||
                       I->op == AGX_OPCODE_QUAD_PREFIX;
   const bool half = I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_16;
   if (I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate ||
       !(half ? agx3_subgroup_plain_half(I->dest[0]) &&
                   agx3_subgroup_plain_half(I->src[0])
              : agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) &&
                   agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER)))
      return false;

   enum agx3_subgroup_reduce_op op;
   if (!agx3_map_reduce_op(I->simd_op, &op))
      return false;
   /* G18G only has a witnessed embedded wait for full-width SIMD iadd.
    * Retire the full wait on our destination, whose pending producers and
    * readers are covered by the scheduler. An identity preserves an in-place
    * source; an unrelated r0 could still have an unretired producer. M6
    * only: the M5 keeps the embedded wait. */
   const bool retire_wait = g18g && enc == AGX_ISA_ENCODING_AGX3C && wait_mask &&
      (quad || half || I->simd_op != AGX_SIMD_OP_IADD);
   const struct agx3_subgroup_reduce_desc d = {
      .quad = quad,
      .prefix = prefix,
      .op = op,
      .dest = I->dest[0].value,
      .src = I->src[0].value,
      .wait_mask = retire_wait ? 0 : wait_mask,
      .dest_hint = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .half = half,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   uint8_t retire_bytes[AGX3_ALU_MAX_LENGTH], retire_length = 0;
   return (g18g
            ? agx3_subgroup_encode_reduce_g18g
            : agx3_subgroup_encode_reduce)(enc, &d, bytes, &length) &&
          (!retire_wait ||
           (agx3_band_carrier(enc, I->dest[0].value, wait_mask,
                              retire_bytes, &retire_length) &&
            agx3_append(emission, retire_bytes, retire_length))) &&
          agx3_append(emission, bytes, length);
}

/* G18G (agx3d) keeps j873's whole AGX3c ballot emitter: its immediate
 * equality with the invert bit, simd_ballot(true) and the unsigned register
 * compare, each retiring producers first. The other agx3c GPUs use
 * agx3_emit_subgroup_ballot() below. */
static bool
agx3_emit_subgroup_ballot_g18g(struct util_dynarray *emission,
                               enum agx_isa_encoding enc,
                               const agx_instr *I, uint16_t wait_mask)
{
   const bool is_float = I->op == AGX_OPCODE_FCMP_BALLOT ||
                         I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   const bool quad = I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
                     I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   /* A 16-bit source is grounded for the immediate integer form only; the
    * encoder refuses every other shape. */
   const agx_index src = I->src[0];
   const bool src_half =
      !is_float && src.type == AGX_INDEX_REGISTER && !src.memory &&
      src.size == AGX_SIZE_16 && agx_channels(src) == 1 && !src.abs &&
      !src.neg && !src.kill && !(src.cache && src.discard);

   /* AGX3c SIMD ballots the forms below cannot take: simd_ballot(true) --
    * `#1 != #0`, both operands immediate -- and an unsigned less-than of two
    * registers. Each is Apple's own instruction for the same predicate
    * (agx3_subgroup_encode_ballot_true_gpu1/_ult_reg_gpu1). Neither witness
    * carries a wait, so any producer retires first, as below. */
   if (enc == AGX_ISA_ENCODING_AGX3C && !quad && !is_float &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER)) {
      const agx_index a = I->src[0], b = I->src[1];
      const bool always =
         I->icond == AGX_ICOND_UEQ && I->invert_cond &&
         agx3_plain_immediate(a) && !a.abs && !a.neg && a.value == 1 &&
         agx3_plain_immediate(b) && !b.abs && !b.neg && b.value == 0;
      const bool ult_regs =
         I->icond == AGX_ICOND_ULT && !I->invert_cond &&
         agx3_subgroup_plain(a, AGX_INDEX_REGISTER) && a.discard &&
         agx3_subgroup_plain(b, AGX_INDEX_REGISTER) && b.discard;
      if (always || ult_regs) {
         uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
         if (wait_mask != 0 &&
             !agx3_retire_slots_noop(emission, enc, wait_mask))
            return false;
         return (always ? agx3_subgroup_encode_ballot_true_gpu1(
                             I->dest[0].value, I->dest[0].cache, bytes,
                             &length)
                        : agx3_subgroup_encode_ballot_ult_reg_gpu1(
                             I->dest[0].value, I->dest[0].cache, a.value,
                             b.value, bytes, &length)) &&
                agx3_append(emission, bytes, length);
      }
   }

   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       I->src[1].type != AGX_INDEX_IMMEDIATE ||
       I->src[1].memory || I->src[1].abs || I->src[1].neg ||
       I->src[1].cache || I->src[1].discard || I->src[1].kill)
      return false;

   /* The 16-bit quad ballot against zero carries its own wait. */
   if (quad && !is_float && I->dest[0].size == AGX_SIZE_16) {
      if (!agx3_subgroup_plain_half(I->dest[0]) || I->src[1].value != 0 ||
          !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)) ||
          (I->icond != AGX_ICOND_UEQ && I->icond != AGX_ICOND_SEQ))
         return false;

      const struct agx3_subgroup_ballot_desc d = {
         .quad = true,
         .type = AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = I->dest[0].value,
         .src = src.value,
         .src_half = src_half,
         .dest_half = true,
         .dest_cache = I->dest[0].cache,
         .src_live = !src.discard,
         .condition = 4,
         .invert = I->invert_cond,
         .wait_mask = wait_mask,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_ballot_g18g(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (!agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)))
      return false;

   bool compare_immediate = false;
   /* Only AGX3c's immediate equality keeps an invert bit; AGX3b's immediate
    * forms fold the sense into the condition instead. */
   bool immediate_invert = false;
   uint32_t compare_value = I->src[1].value;
   uint8_t condition;
   if (is_float) {
      if (compare_value != 0)
         return false;
      condition = I->fcond;
   } else {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
         /* The direct predicate uses Apple's immediate equality form. The
          * register-zero equality form miscomputes on G17P (wrong mask and a
          * lost execution mask), so `x != 0` goes through the immediate form
          * as the unsigned `x > 0`. A nonzero immediate cannot be inverted
          * this way (agx_optimizer ballots the opposite compare instead);
          * the 16-bit quad ballot returned above. AGX3B only.
          */
         if (I->invert_cond) {
            if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
               if (compare_value != 0)
                  return false;
               compare_immediate = true;
               condition = AGX_ICOND_UGT;
               break;
            }
            /* AGX3c: Apple's own simd_ballot(x != 0) is the immediate
             * equality with its invert bit, for 32- and 16-bit x alike
             * (metal-nt applegpu_g17g/g18g, g18ref-out/probe/m6-libagx-forms
             * lk_ballot_ne32/ne16: simdiballot f30=0 f38=1 f39=1 f54=4
             * f55=1, f24 the source width), not the register form. */
            if (!quad && enc == AGX_ISA_ENCODING_AGX3C) {
               compare_immediate = true;
               immediate_invert = true;
               condition = 4;
               break;
            }
            /* The 16-bit quad ballot keeps the form it had. */
            compare_immediate = false;
            condition = 4;
            break;
         }
         compare_immediate = true;
         condition = 4;
         break;
      case AGX_ICOND_SEQ:
         if (compare_value != 0)
            return false;
         /* Signed equality against zero is unsigned equality against zero, so
          * the same remap applies rather than the broken register-zero form. */
         if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
            compare_immediate = true;
            condition = I->invert_cond ? AGX_ICOND_UGT : 4;
            break;
         }
         if (!quad && enc == AGX_ISA_ENCODING_AGX3C) {
            compare_immediate = true;
            immediate_invert = I->invert_cond;
            condition = 4;
            break;
         }
         condition = 4;
         break;
      case AGX_ICOND_SLT:
         if (compare_value != 0)
            return false;
         condition = 5;
         break;
      case AGX_ICOND_SGT:
         if (compare_value != 0)
            return false;
         condition = 6;
         break;
      case AGX_ICOND_ULT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x < N) is x > N - 1.  Apple emits this canonical native
             * form for subgroup votes, and the native instruction has no
             * need for an invert bit. */
            if (compare_value == 0)
               return false;
            compare_value--;
            condition = AGX_ICOND_UGT;
         } else {
            condition = AGX_ICOND_ULT;
         }
         break;
      case AGX_ICOND_UGT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x > N) is x < N + 1. */
            if (compare_value == UINT16_MAX)
               return false;
            compare_value++;
            condition = AGX_ICOND_ULT;
         } else {
            condition = AGX_ICOND_UGT;
         }
         break;
      default: return false;
      }
      if (compare_immediate && compare_value > UINT16_MAX)
         return false;
   }

   /* The native ballot itself has no witnessed scoreboard field.  A ballot
    * predicate may still come directly from an asynchronous SSBO/device
    * load, so retire those exact producer slots with the existing synchronous
    * identity prelude before issuing the unchanged no-wait ballot. */
   if (wait_mask != 0 &&
       !agx3_retire_result_wait(emission, enc, I, wait_mask))
      return false;

   const struct agx3_subgroup_ballot_desc d = {
      .quad = quad,
      .type = is_float ? AGX3_SUBGROUP_BALLOT_FLOAT
                       : AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = I->dest[0].value,
      .src = src.value,
      .compare_immediate = compare_immediate,
      .compare_value = compare_value,
      .src_half = src_half,
      .dest_cache = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .src_cache = enc == AGX_ISA_ENCODING_AGX3B && !quad && !is_float &&
                   !compare_immediate && !src_half && src.cache,
      .condition = condition,
      .invert = compare_immediate ? immediate_invert : I->invert_cond,
      .wait_mask = 0,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_ballot_g18g(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_emit_subgroup_ballot(struct util_dynarray *emission,
                          enum agx_isa_encoding enc, const agx_instr *I,
                          uint16_t wait_mask, bool g18g)
{
   if (g18g)
      return agx3_emit_subgroup_ballot_g18g(emission, enc, I, wait_mask);
   const bool is_float = I->op == AGX_OPCODE_FCMP_BALLOT ||
                         I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   const bool quad = I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
                     I->op == AGX_OPCODE_FCMP_QUAD_BALLOT;
   if (enc == AGX_ISA_ENCODING_AGX3C && !is_float && !quad &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate && wait_mask <= 255 &&
       agx3_subgroup_plain(I->dest[0],AGX_INDEX_REGISTER) &&
       agx3_subgroup_imm_lane(I->src[0]) && I->src[0].value == 1 &&
       agx3_src_is_zero_imm(I->src[1]) && I->icond == AGX_ICOND_UEQ &&
       I->invert_cond) {
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length;
      return agx3_subgroup_encode_active_mask(I->dest[0].value,
                I->dest[0].cache,wait_mask,bytes,&length) &&
             agx3_append(emission,bytes,length);
   }
   /* A 16-bit quad ballot of a boolean immediate (a quad vote whose operand
    * NIR proved constant, e.g. after terminateInvocation). Native G17G keeps
    * it as a ballot: `v == 0` is the inverted ballot of `!v`. */
   if (enc == AGX_ISA_ENCODING_AGX3C && quad && !is_float &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       wait_mask <= 255 && agx3_subgroup_plain_half(I->dest[0]) &&
       agx3_subgroup_imm_lane(I->src[0]) && I->src[0].value <= 1 &&
       agx3_src_is_zero_imm(I->src[1]) &&
       (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ)) {
      const uint16_t nonzero = I->invert_cond ? I->src[0].value
                                              : (I->src[0].value == 0);
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length;
      return agx3_subgroup_encode_quad_ballot_imm(
                I->dest[0].value, I->dest[0].cache, nonzero, wait_mask, bytes,
                &length) &&
             agx3_append(emission, bytes, length);
   }
   if (enc == AGX_ISA_ENCODING_AGX3C && !quad &&
       I->nr_dests == 1 && I->nr_srcs == 2 && !I->saturate &&
       agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) &&
       (agx3_subgroup_plain_half(I->src[0]) ||
        agx3_subgroup_plain(I->src[0], AGX_INDEX_REGISTER)) &&
       (agx3_subgroup_plain_half(I->src[1]) ||
        agx3_subgroup_plain(I->src[1], AGX_INDEX_REGISTER)) &&
       (is_float || I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ ||
        I->icond == AGX_ICOND_ULT || I->icond == AGX_ICOND_UGT ||
        I->icond == AGX_ICOND_SLT || I->icond == AGX_ICOND_SGT)) {
      const struct agx3_subgroup_ballot_desc d = {
         .type = is_float ? AGX3_SUBGROUP_BALLOT_FLOAT : AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = I->dest[0].value,
         .dest_cache = I->dest[0].cache,
         .src = I->src[0].value,
         .src_half = I->src[0].size == AGX_SIZE_16,
         .src_live = !I->src[0].discard,
         .src_cache = I->src[0].cache,
         .compare_register = true,
         .compare_value = I->src[1].value,
         .compare_half = I->src[1].size == AGX_SIZE_16,
         .compare_cache = I->src[1].cache,
         .compare_discard = I->src[1].discard,
         .condition = is_float ? I->fcond :
            (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ) ? 4 :
            I->icond == AGX_ICOND_SLT ? 5 :
            I->icond == AGX_ICOND_SGT ? 6 :
            I->icond == AGX_ICOND_ULT ? 1 : 2,
         .invert = I->invert_cond,
         .wait_mask = wait_mask,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length;
      return agx3_subgroup_encode_ballot(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }
   /* A 16-bit source is grounded for the immediate integer form only; the
    * encoder refuses every other shape. */
   const agx_index src = I->src[0];
   const bool src_half =
      !is_float && src.type == AGX_INDEX_REGISTER && !src.memory &&
      src.size == AGX_SIZE_16 && agx_channels(src) == 1 && !src.abs &&
      !src.neg && !src.kill && !(src.cache && src.discard);
   if (I->nr_dests != 1 || I->nr_srcs != 2 || I->saturate ||
       I->src[1].type != AGX_INDEX_IMMEDIATE ||
       I->src[1].memory || I->src[1].abs || I->src[1].neg ||
       I->src[1].cache || I->src[1].discard || I->src[1].kill)
      return false;

   /* The 16-bit quad ballot against zero carries its own wait. */
   if (quad && !is_float && I->dest[0].size == AGX_SIZE_16) {
      if (!agx3_subgroup_plain_half(I->dest[0]) || I->src[1].value != 0 ||
          !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)) ||
          (I->icond != AGX_ICOND_UEQ && I->icond != AGX_ICOND_SEQ))
         return false;

      const struct agx3_subgroup_ballot_desc d = {
         .quad = true,
         .type = AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = I->dest[0].value,
         .src = src.value,
         .src_half = src_half,
         .dest_half = true,
         .dest_cache = I->dest[0].cache,
         .src_live = !src.discard,
         .condition = 4,
         .invert = I->invert_cond,
         .wait_mask = wait_mask,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      return agx3_subgroup_encode_ballot(enc, &d, bytes, &length) &&
             agx3_append(emission, bytes, length);
   }

   if (!agx3_subgroup_plain(I->dest[0], AGX_INDEX_REGISTER) ||
       !(src_half || agx3_subgroup_plain(src, AGX_INDEX_REGISTER)))
      return false;

   bool compare_immediate = false;
   uint32_t compare_value = I->src[1].value;
   uint8_t condition;
   if (is_float) {
      if (compare_value != 0)
         return false;
      condition = I->fcond;
   } else {
      switch (I->icond) {
      case AGX_ICOND_UEQ:
         /* The direct predicate uses Apple's immediate equality form. The
          * register-zero equality form miscomputes on G17P (wrong mask and a
          * lost execution mask), so `x != 0` goes through the immediate form
          * as the unsigned `x > 0`. A nonzero immediate cannot be inverted
          * this way (agx_optimizer ballots the opposite compare instead);
          * the 16-bit quad ballot returned above. AGX3B only.
          */
         if (I->invert_cond) {
            if (!quad && enc == AGX_ISA_ENCODING_AGX3C) {
               compare_immediate = true;
               condition = 4;
               break;
            }
            if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
               if (compare_value != 0)
                  return false;
               compare_immediate = true;
               condition = AGX_ICOND_UGT;
               break;
            }
            /* The 16-bit quad ballot and AGX3C keep the form they had. */
            compare_immediate = false;
            condition = 4;
            break;
         }
         compare_immediate = true;
         condition = 4;
         break;
      case AGX_ICOND_SEQ:
         if (compare_value != 0)
            return false;
         /* Signed equality against zero is unsigned equality against zero, so
          * the same remap applies rather than the broken register-zero form. */
         if (!quad && enc == AGX_ISA_ENCODING_AGX3B) {
            compare_immediate = true;
            condition = I->invert_cond ? AGX_ICOND_UGT : 4;
            break;
         }
         condition = 4;
         break;
      case AGX_ICOND_SLT:
         if (compare_value != 0)
            return false;
         condition = 5;
         break;
      case AGX_ICOND_SGT:
         if (compare_value != 0)
            return false;
         condition = 6;
         break;
      case AGX_ICOND_ULT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x < N) is x > N - 1.  Apple emits this canonical native
             * form for subgroup votes, and the native instruction has no
             * need for an invert bit. */
            if (compare_value == 0)
               return false;
            compare_value--;
            condition = AGX_ICOND_UGT;
         } else {
            condition = AGX_ICOND_ULT;
         }
         break;
      case AGX_ICOND_UGT:
         compare_immediate = true;
         if (I->invert_cond) {
            /* !(x > N) is x < N + 1. */
            if (compare_value == UINT16_MAX)
               return false;
            compare_value++;
            condition = AGX_ICOND_ULT;
         } else {
            condition = AGX_ICOND_UGT;
         }
         break;
      default: return false;
      }
      if (compare_immediate && compare_value > UINT16_MAX)
         return false;
   }

   /* The native ballot itself has no witnessed scoreboard field.  A ballot
    * predicate may still come directly from an asynchronous SSBO/device
    * load, so retire those exact producer slots with the existing synchronous
    * identity prelude before issuing the unchanged no-wait ballot. */
   const bool direct_wait = enc == AGX_ISA_ENCODING_AGX3C && !is_float && !quad;
   if (!direct_wait && wait_mask != 0 &&
       !agx3_retire_result_wait(emission, enc, I, wait_mask))
      return false;

   const struct agx3_subgroup_ballot_desc d = {
      .quad = quad,
      .type = is_float ? AGX3_SUBGROUP_BALLOT_FLOAT
                       : AGX3_SUBGROUP_BALLOT_INTEGER,
      .dest = I->dest[0].value,
      .src = src.value,
      .compare_immediate = compare_immediate,
      .compare_value = compare_value,
      .src_half = src_half,
      .dest_cache = I->dest[0].cache,
      .src_live = !I->src[0].discard,
      .condition = condition,
      .invert = compare_immediate && !direct_wait ? false : I->invert_cond,
      .wait_mask = direct_wait ? wait_mask : 0,
   };
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   return agx3_subgroup_encode_ballot(enc, &d, bytes, &length) &&
          agx3_append(emission, bytes, length);
}

static bool
agx3_m5_texture_barrier(enum agx_isa_encoding enc, unsigned scope)
{
   return enc == AGX_ISA_ENCODING_AGX3C &&
          (scope == AGX3_BARRIER_SCOPE_TEXTURE ||
           scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT ||
           scope == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP);
}

/* Whether an M5 (G17) barrier is a workgroup barrier at which every load
 * still in flight is retired first: every invocation of the workgroup stops
 * there until all of them arrive. EXECUTION and THREADGROUP are the
 * workgroup barriers without and with threadgroup memory. TEXTURE also
 * carries the rendezvous, but the compiler emits it only right after a
 * DEVICE barrier, which already retires everything (a workgroup barrier over
 * image memory, fence_mem_to_tex_agx), and as the fragment shader's pixel
 * fence (fence_pbe_to_tex_pixel_agx), which has no workgroup. G18G, which
 * compiles with the same encoding, keeps its barriers as they are. */
bool
agx3_m5_rendezvous_barrier(const agx_context *ctx, enum agx_isa_encoding enc,
                           unsigned scope)
{
   return enc == AGX_ISA_ENCODING_AGX3C && !ctx->key->dev.emit_agx3d &&
          (scope == AGX3_BARRIER_SCOPE_EXECUTION ||
           scope == AGX3_BARRIER_SCOPE_THREADGROUP);
}

static bool
agx3_emit_scoped_barrier(struct util_dynarray *emission,
                         enum agx_isa_encoding enc, const agx_instr *I,
                         uint16_t wait_mask)
{
   if (I->op != AGX_OPCODE_SCOPED_BARRIER_AGX3 ||
       I->nr_dests != 0 || I->nr_srcs != 0 ||
       I->imm > AGX3_BARRIER_SCOPE_OBJECT_PAYLOAD_EXIT)
      return false;

   struct agx3_encoded_scoped_barrier encoded;
   /* AGX3C texture barriers wait in their own first word, as on G17G. */
   if (agx3_m5_texture_barrier(enc, I->imm))
      return agx3_pack_scoped_barrier_wait(enc, I->imm, wait_mask, &encoded) &&
             agx3_append(emission, encoded.bytes, encoded.length);

   return (!wait_mask || agx3_retire_slots_noop(emission, enc, wait_mask)) &&
          agx3_pack_scoped_barrier(enc, I->imm, &encoded) &&
          agx3_append(emission, encoded.bytes, encoded.length);
}

enum agx3_pack_result
agx3_pack_matrix_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   if (I->op == AGX_OPCODE_MATMAD16_AGX3) {
      if (enc != AGX_ISA_ENCODING_AGX3C || ctx->is_preamble ||
          ctx->stage != MESA_SHADER_COMPUTE || I->nr_dests != 1 ||
          I->nr_srcs != 3 || I->saturate || st->wait > 0xff ||
          !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          I->dest[0].size != AGX_SIZE_32 ||
          agx_channels(I->dest[0]) != 8 ||
          I->dest[0].value != I->src[2].value)
         AGX3_REFUSE_RET(I);
      for (unsigned s = 0; s < 3; ++s) {
         const agx_index src = I->src[s];
         if (!agx3_src_is_register(src) || src.memory || src.abs ||
             src.neg || agx_channels(src) != 8 ||
             src.size != (s == 2 ? AGX_SIZE_32 : AGX_SIZE_16))
            AGX3_REFUSE_RET(I);
      }
      struct agx3_matrix_gemm16_desc d = {
         .dest = I->dest[0].value,
         .a = I->src[0].value,
         .b = I->src[1].value,
         .input_type = I->matrix_type & 15,
         .b_input_type = (I->matrix_type >> 4) & 15,
         .wait_mask = st->wait,
         .preserve_a = !I->src[0].discard,
         .preserve_b = !I->src[1].discard,
      };
      uint8_t bytes[10], length;
      if (!agx3_matrix_encode_gemm16(enc, &d, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_MATMAD8_AGX3) {
      if ((I->matrix_type & 15) != ((I->matrix_type >> 4) & 15))
         AGX3_REFUSE_RET(I);
      /* 8x8 matmad is available on AGX3B (G15/G16/G17P) as well as M5
       * (AGX3C); the encoder packs the right form for each. The 16x16
       * GEMM above stays AGX3C only. */
      const unsigned wait_max =
         enc == AGX_ISA_ENCODING_AGX3B ? 0x3f : 0xff;
      if ((enc != AGX_ISA_ENCODING_AGX3C &&
           enc != AGX_ISA_ENCODING_AGX3B) || ctx->is_preamble ||
          ctx->stage != MESA_SHADER_COMPUTE || I->nr_dests != 1 ||
          I->nr_srcs != 3 || I->saturate || st->wait > wait_max ||
          !agx3_src_is_register(I->dest[0]) || I->dest[0].memory ||
          I->dest[0].size != AGX_SIZE_32 ||
          agx_channels(I->dest[0]) != 2)
         AGX3_REFUSE_RET(I);

      struct agx3_matrix_mma8_desc d = {
         .dest = I->dest[0].value,
         .a = I->src[0].value,
         .b = I->src[1].value,
         .c = I->src[2].value,
         .dest_cache = I->dest[0].cache,
         .wait_mask = st->wait,
         .input_type = I->matrix_type & 15,
      };
      for (unsigned s = 0; s < 3; ++s) {
         const agx_index src = I->src[s];
         if (!agx3_src_is_register(src) || src.memory || src.abs ||
             src.neg || agx_channels(src) != 2 ||
             src.size != (s == 2 || (I->matrix_type & 15) == AGX3_MATRIX_INPUT_F32
                            ? AGX_SIZE_32 : AGX_SIZE_16))
            AGX3_REFUSE_RET(I);
         d.cache[s] = src.cache;
         d.discard[s] = src.discard;
      }
      uint8_t bytes[14], length;
      if (!agx3_matrix_encode_mma8(enc, &d, bytes, &length) ||
          !agx3_append(emission, bytes, length))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   /* Give every control-flow boundary an empty scoreboard, rather
    * than refusing the shader when something is still in flight. */
   return AGX3_PACK_UNHANDLED;
}

enum agx3_pack_result
agx3_pack_subgroup_instr(struct agx3_pack_state *st)
{
   agx_context *ctx = st->ctx;
   struct util_dynarray *emission = st->emission;
   const enum agx_isa_encoding enc = st->enc;
   const agx_instr *I = st->I;

   bool subgroup_quad;
   enum agx3_subgroup_shuffle_op subgroup_shuffle;
   if (agx3_map_shuffle(I->op, &subgroup_quad, &subgroup_shuffle)) {
      if (!agx3_emit_subgroup_shuffle(emission, enc, I, st->wait,
                                      agx_is_g18g(&ctx->key->dev)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_SIMD_REDUCE ||
       I->op == AGX_OPCODE_QUAD_REDUCE ||
       I->op == AGX_OPCODE_SIMD_PREFIX ||
       I->op == AGX_OPCODE_QUAD_PREFIX) {
      if (!agx3_emit_subgroup_reduce(emission, enc, I, st->wait,
                                     agx_is_g18g(&ctx->key->dev)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_ICMP_BALLOT ||
       I->op == AGX_OPCODE_ICMP_QUAD_BALLOT ||
       I->op == AGX_OPCODE_FCMP_BALLOT ||
       I->op == AGX_OPCODE_FCMP_QUAD_BALLOT) {
      if (!agx3_emit_subgroup_ballot(emission, enc, I, st->wait,
                                     agx_is_g18g(&ctx->key->dev)))
         AGX3_REFUSE_RET(I);
      return AGX3_PACK_EMITTED;
   }

   if (I->op == AGX_OPCODE_SCOPED_BARRIER_AGX3) {
      /* Native G18 texture barriers wait for the preceding IMGWCI's
       * slot before making that write visible to a subsequent fetch.
       * Image writes add emitter-owned slots to outstanding, without
       * an SSA destination for the earlier register-slot planner.
       * Waiting only on that planner's mask lets the visibility
       * barrier overtake the write (observed as a one-write-old VS
       * imageBuffer read). Retire the live generations on the grounded
       * identity carrier before issuing the existing barrier pair. */
      const bool g18g_texture_barrier = ctx->key->dev.emit_agx3d &&
         I->imm == AGX3_BARRIER_SCOPE_TEXTURE;
      /* Image-write slots are assigned by the staging allocator after
       * the general scheduler. A device fence must also complete these
       * issued writes, which the scheduler's drain cannot see. */
      const bool device = I->imm == AGX3_BARRIER_SCOPE_DEVICE ||
                          I->imm == AGX3_BARRIER_SCOPE_DEVICE_FENCE;
      /* An M5 workgroup barrier retires everything in flight, as the
       * planner's drain does for the operations it tracks. */
      const bool rendezvous =
         agx3_m5_rendezvous_barrier(ctx, enc, I->imm);
      /* The native PBE-to-texture fence includes a wait on slot1.
       * Heap image writes rotate through the six completion slots,
       * however, so retire their other live generations before the
       * fence invalidates the texture cache. Waiting after the fence
       * is too late: a following read can still observe the old texel.
       * Unrelated memory/texture producers do not join this wait. */
      const bool texture_wait =
         I->imm == AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT;
      /* AGX3C has no implied slot: every G17G texture barrier waits,
       * in its first word, on the result slots of all image writes
       * still in flight (port/fencewit k_fence, k_fence2,
       * k_fence_twice, k_tgbar_tex, k_atf_tg, k_atf_dev), and on no
       * unrelated producer (k_fence_ld). G18G keeps j873's texture
       * barrier, which retires everything outstanding. */
      const bool m5_texture = agx3_m5_texture_barrier(enc, I->imm);
      const uint16_t image_wait =
         ctx->key->dev.emit_agx3d ? 0 :
         m5_texture ? st->image_write_slots & st->outstanding_here :
         texture_wait ? st->image_write_slots & st->outstanding_here & ~1u : 0;
      const uint16_t retire =
         st->wait | image_wait |
         ((device || rendezvous || g18g_texture_barrier)
             ? st->outstanding_here : 0);
      if (!agx3_emit_scoped_barrier(emission, enc, I, retire))
         AGX3_REFUSE_RET(I);
      const uint16_t completed =
         retire | (texture_wait && !m5_texture ? 1u : 0);
      st->outstanding &= (uint16_t)~completed;
      st->image_write_slots &= (uint16_t)~completed;
      agx3_staging_forget(&st->staging_banks, completed);
      return AGX3_PACK_EMITTED;
   }

   return AGX3_PACK_UNHANDLED;
}
