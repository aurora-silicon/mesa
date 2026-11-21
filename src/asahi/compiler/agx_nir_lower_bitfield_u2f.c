/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "agx_nir.h"

/*
 * Convert a narrow unsigned bitfield to float through the mantissa.
 *
 * If v = (x & mask) >> s and mask lies in bits [s, 23), then
 *
 *    uintBitsToFloat((x & mask) | ((150 - s) << 23)) = 2^(23 - s) + v
 *
 * exactly: the exponent makes bit s of the mantissa weigh 1. Subtracting
 * 2^(23 - s) is exact too, so u2f32(v) becomes an and, an or and an add,
 * and an fadd of a constant that reads the conversion takes the bias into
 * its constant when that stays exact. The result is bit-identical on every
 * generation, so the pass runs on all of them.
 *
 * Cost model: the rewrite trades a bitfield extract and an int-to-float
 * conversion for bitwise ops and an FP add. Measured on G17P only, where
 * extracts and conversions issue at a fraction of the rate of bitwise ops
 * and FP adds: in a throughput loop, 4-bit fields unpack 1.8x faster this
 * way (bitfield extract, u2f32, fadd -8 as in llama.cpp's Q4_0), 1.6x
 * without the fadd, and masked fields that need no shift 1.25x. Other
 * generations are unmeasured. The rewrite issues up to one instruction more
 * per field, which is a loss where extracts and conversions run at full rate.
 */

struct field {
   nir_def *masked; /* x & mask, if it exists already */
   nir_scalar x;
   uint32_t mask;
   unsigned shift;
};

static bool
match_field(nir_shader *shader, struct hash_table *ranges, nir_scalar v,
            struct field *f)
{
   if (!nir_scalar_is_alu(v))
      return false;

   /* Each field op below has at least two sources; other ops may have one. */
   const nir_op op = nir_scalar_alu_op(v);
   if (op != nir_op_ubitfield_extract && op != nir_op_ushr &&
       op != nir_op_iand)
      return false;

   nir_scalar src0 = nir_scalar_chase_alu_src(v, 0);
   nir_scalar src1 = nir_scalar_chase_alu_src(v, 1);

   switch (op) {
   case nir_op_ubitfield_extract: {
      nir_scalar bits = nir_scalar_chase_alu_src(v, 2);
      if (!nir_scalar_is_const(src1) || !nir_scalar_is_const(bits))
         return false;
      unsigned offset = nir_scalar_as_uint(src1) & 31;
      unsigned width = nir_scalar_as_uint(bits);
      if (width == 0 || offset + width > 23)
         return false;
      *f = (struct field){NULL, src0, BITFIELD_MASK(width) << offset, offset};
      return true;
   }

   case nir_op_ushr: {
      if (!nir_scalar_is_const(src1) ||
          nir_unsigned_upper_bound(shader, ranges, src0) >= (1u << 23))
         return false;
      unsigned shift = nir_scalar_as_uint(src1) & 31;
      if (shift >= 23)
         return false;
      *f = (struct field){NULL, src0, BITFIELD_MASK(23) & ~BITFIELD_MASK(shift),
                          shift};
      return true;
   }

   case nir_op_iand: {
      /* The mask stays in place: this saves the conversion only. */
      if (nir_scalar_is_const(src0)) {
         nir_scalar t = src0;
         src0 = src1;
         src1 = t;
      }
      if (!nir_scalar_is_const(src1) ||
          nir_scalar_as_uint(src1) >= (1u << 23) || v.comp != 0 ||
          v.def->num_components != 1)
         return false;
      *f = (struct field){v.def, src0, nir_scalar_as_uint(src1), 0};
      return true;
   }

   default:
      return false;
   }
}

/* The fadd of a constant that is the conversion's only use, if any. */
static nir_alu_instr *
only_constant_add(nir_def *def, unsigned *src)
{
   if (!list_is_singular(&def->uses))
      return NULL;

   nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return NULL;

   nir_instr *instr = nir_src_use_instr(use);
   if (instr->type != nir_instr_type_alu)
      return NULL;

   nir_alu_instr *add = nir_instr_as_alu(instr);
   if (add->op != nir_op_fadd || add->def.num_components != 1 ||
       add->def.bit_size != 32)
      return NULL;

   *src = (nir_alu_src *)container_of(use, nir_alu_src, src) - add->src;
   nir_scalar c = nir_scalar_chase_alu_src(nir_get_scalar(&add->def, 0),
                                           1 - *src);
   return nir_scalar_is_const(c) ? add : NULL;
}

static bool
lower(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_u2f32 || alu->def.num_components != 1 ||
       nir_src_bit_size(alu->src[0].src) != 32)
      return false;

   struct field f;
   if (!match_field(b->shader, data, nir_get_scalar(alu->src[0].src.ssa,
                                                    alu->src[0].swizzle[0]),
                    &f))
      return false;

   const float bias = (float)(1u << (23 - f.shift));
   unsigned s = 0;
   nir_alu_instr *add = only_constant_add(&alu->def, &s);
   double folded = 0;
   if (add) {
      nir_scalar c = nir_scalar_chase_alu_src(nir_get_scalar(&add->def, 0),
                                              1 - s);
      /* Fold only an exact c - bias: TwoSum's error term is zero, and the
       * difference is a float. */
      const double k = nir_scalar_as_float(c), nb = -(double)bias;
      folded = k + nb;
      const double bv = folded - k, av = folded - bv;
      if ((k - av) + (nb - bv) != 0 || (double)(float)folded != folded)
         add = NULL;
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *masked =
      f.masked ? f.masked
               : nir_iand_imm(b, nir_mov_scalar(b, f.x), f.mask);
   nir_def *biased = nir_ior_imm(b, masked, (150 - f.shift) << 23);

   if (add) {
      b->cursor = nir_before_instr(&add->instr);
      nir_src_rewrite(&add->src[s].src, biased);
      add->src[s].swizzle[0] = 0;
      nir_src_rewrite(&add->src[1 - s].src, nir_imm_float(b, (float)folded));
      add->src[1 - s].swizzle[0] = 0;
      nir_instr_remove(&alu->instr);
   } else {
      nir_def_replace(&alu->def, nir_fadd_imm(b, biased, -bias));
   }
   return true;
}

bool
agx_nir_lower_bitfield_u2f(nir_shader *shader)
{
   struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
   bool progress = nir_shader_alu_pass(shader, lower, nir_metadata_control_flow,
                                       ranges);
   _mesa_hash_table_destroy(ranges, NULL);
   return progress;
}
