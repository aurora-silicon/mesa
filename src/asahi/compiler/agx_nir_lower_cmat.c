/*
 * Copyright © 2023 Bas Nieuwenhuizen
 * Copyright © 2024 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */


#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"

static const struct glsl_type *
remap_matrix_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(mapping, orig);

   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;

   if (glsl_type_is_cmat(orig)) {
      struct glsl_cmat_description matrix_desc =
         *glsl_get_cmat_description(orig);

      new_type = glsl_vector_type(matrix_desc.element_type,
                                  matrix_desc.rows * matrix_desc.cols / 32);
   } else if (glsl_type_is_array(orig)) {
      const struct glsl_type *elem_type = glsl_get_array_element(orig);
      const struct glsl_type *new_elem_type =
         remap_matrix_type(mapping, elem_type);

      if (elem_type != new_elem_type) {
         new_type = glsl_array_type(new_elem_type, glsl_get_length(orig),
                                    glsl_get_explicit_stride(orig));
      }
   } else if (glsl_type_is_struct(orig)) {
      unsigned i;
      for (i = 0; i < orig->length; i++) {
         const struct glsl_type *field_type = glsl_get_struct_field(orig, i);
         const struct glsl_type *new_field_type =
            remap_matrix_type(mapping, field_type);

         if (field_type != new_field_type) {
            break;
         }
      }

      
      if (i < orig->length) {
         struct glsl_struct_field *fields =
            malloc(sizeof(struct glsl_struct_field) * orig->length);

         
         memcpy(fields, orig->fields.structure,
                sizeof(struct glsl_struct_field) * i);

         
         for (; i < orig->length; i++) {
            fields[i] = *glsl_get_struct_field_data(orig, i);
            fields[i].type = remap_matrix_type(mapping, fields[i].type);
         }

         new_type =
            glsl_struct_type(fields, orig->length, glsl_get_type_name(orig),
                             glsl_struct_type_is_packed(orig));

         free(fields);
      }
   }

   _mesa_hash_table_insert(mapping, orig, (void *)new_type);
   return new_type;
}

static struct glsl_cmat_description
matrix_desc(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static nir_def *
load_matrix(nir_builder *b, nir_src src)
{
   struct glsl_cmat_description desc = matrix_desc(src);
   assert((desc.rows == 8 || desc.rows == 16) &&
          desc.cols == desc.rows && desc.scope == SCOPE_SUBGROUP);
   return nir_build_load_deref(b, desc.rows * desc.cols / 32, glsl_base_type_bit_size(desc.element_type),
                               src.ssa, 0);
}

static void
store_matrix(nir_builder *b, nir_src dst, nir_def *value)
{
   nir_build_store_deref(b, dst.ssa, value, .write_mask = BITFIELD_MASK(value->num_components));
}

static nir_def *
matrix_row(nir_builder *b, nir_def *lane,
           struct glsl_cmat_description desc, unsigned element)
{
   if (desc.rows == 16)
      return nir_ior_imm(b, nir_ior(b, nir_iand_imm(b, lane, 6),
                 nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 8)), element >> 2);
   return nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 3),
                    nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 4));
}

static nir_def *
matrix_col(nir_builder *b, nir_def *lane,
           struct glsl_cmat_description desc, unsigned element)
{
   if (desc.cols == 16) {
      if (desc.use == GLSL_CMAT_USE_A)
         return nir_ior_imm(b, nir_ior(b,
            nir_iand_imm(b, nir_ushr_imm(b, lane, 3), 1),
            nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 3)),
            (element & 3) << 1);
      return nir_ior_imm(b, nir_ior(b,
         nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 2),
         nir_iand_imm(b, lane, 8)), element & 3);
   }
   return nir_ior_imm(b, nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, lane, 1), 1),
                                   nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 4)),
                         element);
}

/* Where a matrix's elements are: bit i of an element's index within its lane
 * and bit j of its lane's subgroup invocation each hold one bit of its row
 * (ROW(k)) or of its column (COL(k)), as matrix_row() and matrix_col() place
 * them. The accumulator and B layouts of a 16x16 matrix are the same; A
 * differs only in where its column bits are. All uses of an 8x8 matrix share
 * one layout.
 */
#define ROW(k)       (k)
#define COL(k)       (8 + (k))
#define COORD_BIT(c) ((c) & 7)
#define TRANSPOSED(c) ((c) ^ 8)

struct cmat_layout {
   unsigned element_bits;
   uint8_t element[3];
   uint8_t lane[5];
};

static const struct cmat_layout layout_16_a = {
   3, {COL(1), COL(2), ROW(0)}, {COL(3), ROW(1), ROW(2), COL(0), ROW(3)},
};
static const struct cmat_layout layout_16 = {
   3, {COL(0), COL(1), ROW(0)}, {COL(2), ROW(1), ROW(2), COL(3), ROW(3)},
};
static const struct cmat_layout layout_8 = {
   1, {COL(0)}, {COL(1), ROW(0), ROW(1), COL(2), ROW(2)},
};

static const struct cmat_layout *
matrix_layout(struct glsl_cmat_description desc)
{
   assert((desc.rows == 8 || desc.rows == 16) && desc.cols == desc.rows);
   if (desc.rows == 8)
      return &layout_8;
   return desc.use == GLSL_CMAT_USE_A ? &layout_16_a : &layout_16;
}

static int
find_element_bit(const struct cmat_layout *l, unsigned coord)
{
   for (unsigned i = 0; i < l->element_bits; ++i) {
      if (l->element[i] == coord)
         return i;
   }
   return -1;
}

static int
find_lane_bit(const struct cmat_layout *l, unsigned coord)
{
   for (unsigned j = 0; j < 5; ++j) {
      if (l->lane[j] == coord)
         return j;
   }
   return -1;
}

/* out[i] = subgroupShuffleXor(in[i], mask). The 16-bit values go two to a
 * 32-bit shuffle, which is one instruction for a constant mask.
 */
static void
shuffle_xor_values(nir_builder *b, nir_def **in, nir_def **out, unsigned count,
                   unsigned mask)
{
   nir_def *m = nir_imm_int(b, mask);
   for (unsigned i = 0; i < count; ++i) {
      if (in[i]->bit_size == 32) {
         out[i] = nir_shuffle_xor(b, in[i], m);
      } else if (i + 1 < count) {
         nir_def *pair = nir_shuffle_xor(b, nir_pack_32_2x16_split(b, in[i], in[i + 1]), m);
         out[i] = nir_unpack_32_2x16_split_x(b, pair);
         out[++i] = nir_unpack_32_2x16_split_y(b, pair);
      } else {
         out[i] = nir_u2u16(b, nir_shuffle_xor(b, nir_u2u32(b, in[i]), m));
      }
   }
}

/* Swap what element bit i and lane bit j of the matrix in v hold: of each pair
 * of elements differing in bit i, a lane keeps the one that stays and trades
 * the other with the lane differing in bit j.
 */
static void
exchange_bits(nir_builder *b, nir_def **v, struct cmat_layout *l, unsigned i,
              unsigned j, nir_def *lane)
{
   unsigned count = 1 << l->element_bits, pairs = 0;
   nir_def *upper = nir_i2b(b, nir_iand_imm(b, lane, 1 << j));
   nir_def *send[4], *recv[4];
   for (unsigned e = 0; e < count; ++e) {
      if (!(e & (1 << i)))
         send[pairs++] = nir_bcsel(b, upper, v[e], v[e | (1 << i)]);
   }
   shuffle_xor_values(b, send, recv, pairs, 1 << j);
   pairs = 0;
   for (unsigned e = 0; e < count; ++e) {
      if (e & (1 << i))
         continue;
      v[e] = nir_bcsel(b, upper, recv[pairs], v[e]);
      v[e | (1 << i)] = nir_bcsel(b, upper, v[e | (1 << i)], recv[pairs]);
      pairs++;
   }
   uint8_t tmp = l->element[i];
   l->element[i] = l->lane[j];
   l->lane[j] = tmp;
}

/* Move a matrix's elements from layout `from` to layout `to` (with the same
 * coordinates). Each lane bit that holds the wrong coordinate trades it with
 * the element bit that holds the right one; a coordinate that is in another
 * lane bit goes through an element bit first. What is left is a renaming of
 * the elements within each lane.
 */
static nir_def *
permute_matrix(nir_builder *b, nir_def *value, struct cmat_layout from,
               const struct cmat_layout *to)
{
   nir_def *v[8], *out[8];
   unsigned count = 1 << from.element_bits;
   assert(value->num_components == count && to->element_bits == from.element_bits);
   for (unsigned e = 0; e < count; ++e)
      v[e] = nir_channel(b, value, e);

   nir_def *lane = nir_load_subgroup_invocation(b);
   for (unsigned j = 0; j < 5; ++j) {
      while (from.lane[j] != to->lane[j]) {
         int i = find_element_bit(&from, to->lane[j]);
         if (i >= 0) {
            exchange_bits(b, v, &from, i, j, lane);
            continue;
         }
         /* Bring it in from lane bit k, through the element bit whose
          * coordinate lane bit k wants if there is one.
          */
         int k = find_lane_bit(&from, to->lane[j]);
         assert(k > (int)j);
         i = find_element_bit(&from, to->lane[k]);
         exchange_bits(b, v, &from, i >= 0 ? i : 0, k, lane);
      }
   }
   for (unsigned e = 0; e < count; ++e) {
      unsigned src = 0;
      for (unsigned i = 0; i < from.element_bits; ++i) {
         if (e & (1 << i))
            src |= 1 << find_element_bit(&from, to->element[i]);
      }
      out[e] = v[src];
   }
   return nir_vec(b, out, count);
}

/* The memory of matrix element `element` of this lane, or of the one `skip`
 * elements after it along its row (a column, for a column-major layout).
 */
static nir_deref_instr *
matrix_memory_at(nir_builder *b, nir_intrinsic_instr *intr, unsigned element,
                 unsigned skip)
{
   const bool store = intr->intrinsic == nir_intrinsic_cmat_store;
   struct glsl_cmat_description desc = matrix_desc(intr->src[store ? 1 : 0]);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[store ? 0 : 1]);
   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *row = matrix_row(b, lane, desc, element),
           *col = matrix_col(b, lane, desc, element);
   if (nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
      nir_def *tmp = row; row = col; col = tmp;
   }
   unsigned bits = ptr->def.bit_size;
   nir_def *stride = nir_u2uN(b, intr->src[2].ssa, bits);
   nir_def *row_offset = nir_imul(b, nir_u2uN(b, row, bits), stride);
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes, ptr->type,
                              glsl_get_vector_elements(ptr->type) *
                              glsl_get_bit_size(ptr->type) / 8);
   ptr = nir_build_deref_ptr_as_array(b, ptr, row_offset);
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes,
                              glsl_scalar_type(desc.element_type),
                              glsl_base_type_bit_size(desc.element_type) / 8);
   return nir_build_deref_ptr_as_array(
      b, ptr, nir_u2uN(b, nir_iadd_imm(b, col, skip), bits));
}

static nir_deref_instr *
matrix_memory(nir_builder *b, nir_intrinsic_instr *intr, unsigned element)
{
   return matrix_memory_at(b, intr, element, 0);
}

static nir_def *
convert_matrix_value(nir_builder *b, nir_def *value,
                     enum glsl_base_type src, enum glsl_base_type dst)
{
   if (src == dst)
      return value;
   if (src == GLSL_TYPE_BFLOAT16) {
      value = nir_bf2f(b, value);
      src = GLSL_TYPE_FLOAT;
   }
   if (dst == GLSL_TYPE_BFLOAT16) {
      if (src != GLSL_TYPE_FLOAT)
         value = nir_f2f32(b, value);
      return nir_f2bf(b, value);
   }
   nir_op op = nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src),
                                      nir_get_nir_type_for_glsl_base_type(dst),
                                      nir_rounding_mode_undef);
   return nir_build_alu1(b, op, value);
}

static bool
lower_matrix_instr(nir_builder *b, nir_instr *instr, struct hash_table *types)
{
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      const glsl_type *type = remap_matrix_type(types, deref->type);
      bool changed = type != deref->type;
      deref->type = type;
      return changed;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct:
      store_matrix(b, intr->src[0], nir_replicate(b, intr->src[1].ssa,
         matrix_desc(intr->src[0]).rows * matrix_desc(intr->src[0]).cols / 32));
      break;
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store: {
      bool store = intr->intrinsic == nir_intrinsic_cmat_store;
      struct glsl_cmat_description desc = matrix_desc(intr->src[store ? 1 : 0]);
      nir_def *data = store ? load_matrix(b, intr->src[1]) : NULL;
      nir_def *values[8];
      unsigned count = desc.rows * desc.cols / 32;
      bool row_major = nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_ROW_MAJOR;
      unsigned width = desc.rows == 8 ? (row_major ? 2 : 1)
         : (!row_major ? 2 : desc.use == GLSL_CMAT_USE_A ? 1 : 4);
      unsigned component_step = desc.rows == 16 && !row_major ? 4 : 1;
      unsigned component_mask = (width - 1) * component_step;
      /* A lane's four elements of a row of a 16x16 A operand are every other
       * one of seven consecutive columns (matrix_col), so they are in two
       * four-element loads that start at the first and at the fourth of the
       * seven: the first load's elements 0 and 2, the second's 1 and 3.
       * Neither reads outside the lane's seven columns. A store cannot
       * overlap its neighbours' columns and stays scalar.
       */
      if (!store && desc.rows == 16 && row_major &&
          desc.use == GLSL_CMAT_USE_A) {
         unsigned bytes = glsl_base_type_bit_size(desc.element_type) / 8;
         for (unsigned i = 0; i < count; i += 4) {
            for (unsigned half = 0; half < 2; ++half) {
               nir_deref_instr *ptr = matrix_memory_at(b, intr, i, half * 3);
               ptr = nir_build_deref_cast_with_alignment(
                  b, &ptr->def, ptr->modes,
                  glsl_vector_type(desc.element_type, 4), 0, bytes, 0);
               nir_def *loaded = nir_load_deref(b, ptr);
               values[i + half * 2] = nir_channel(b, loaded, half);
               values[i + half * 2 + 1] = nir_channel(b, loaded, half + 2);
            }
         }
         store_matrix(b, intr->src[0], nir_vec(b, values, count));
         break;
      }

      for (unsigned i = 0; i < count; ++i) {
         if (i & component_mask)
            continue;
         nir_deref_instr *ptr = matrix_memory(b, intr, i);
         if (width > 1) {
            unsigned align = width * glsl_base_type_bit_size(desc.element_type) / 8;
            ptr = nir_build_deref_cast_with_alignment(b, &ptr->def, ptr->modes,
               glsl_vector_type(desc.element_type, width), 0, align, 0);
         }
         if (store) {
            nir_def *channels[4];
            for (unsigned j = 0; j < width; ++j)
               channels[j] = nir_channel(b, data, i + j * component_step);
            nir_store_deref(b, ptr, nir_vec(b, channels, width), ~0);
         } else {
            nir_def *loaded = nir_load_deref(b, ptr);
            for (unsigned j = 0; j < width; ++j)
               values[i + j * component_step] = nir_channel(b, loaded, j);
         }
      }
      if (!store)
         store_matrix(b, intr->src[0], nir_vec(b, values, count));
      break;
   }
   case nir_intrinsic_cmat_length: {
      struct glsl_cmat_description desc = nir_intrinsic_cmat_desc(intr);
      nir_def_replace(&intr->def, nir_imm_int(b, desc.rows * desc.cols / 32));
      return true;
   }
   case nir_intrinsic_cmat_muladd: {
      assert(!nir_intrinsic_saturate(intr));
      enum glsl_base_type input = matrix_desc(intr->src[1]).element_type;
      enum glsl_base_type input_b = matrix_desc(intr->src[2]).element_type;
      enum glsl_base_type accum = matrix_desc(intr->src[3]).element_type;
      enum glsl_base_type result = matrix_desc(intr->src[0]).element_type;
      assert((input == GLSL_TYPE_FLOAT16 || input == GLSL_TYPE_BFLOAT16 ||
              (input == GLSL_TYPE_FLOAT && matrix_desc(intr->src[1]).rows == 8)) &&
             (input_b == input || (matrix_desc(intr->src[1]).rows == 16 &&
                ((input == GLSL_TYPE_FLOAT16 && input_b == GLSL_TYPE_BFLOAT16) ||
                 (input == GLSL_TYPE_BFLOAT16 && input_b == GLSL_TYPE_FLOAT16)))) &&
             (accum == GLSL_TYPE_FLOAT || accum == GLSL_TYPE_BFLOAT16) && result == accum);
      nir_def *a = load_matrix(b, intr->src[1]);
      nir_def *bb = load_matrix(b, intr->src[2]);
      nir_def *c = convert_matrix_value(b, load_matrix(b, intr->src[3]),
                                         accum, GLSL_TYPE_FLOAT);
      nir_def *value = nir_cmat_muladd_agx(b, a, bb, c, .src_type =
         input == GLSL_TYPE_BFLOAT16 ? nir_type_uint16 :
         input == GLSL_TYPE_FLOAT ? nir_type_float32 : nir_type_float16,
         .src_type2 = input_b == GLSL_TYPE_BFLOAT16 ? nir_type_uint16 :
         input_b == GLSL_TYPE_FLOAT ? nir_type_float32 : nir_type_float16);
      store_matrix(b, intr->src[0], convert_matrix_value(b, value,
                                                      GLSL_TYPE_FLOAT, result));
      break;
   }
   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op: {
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      nir_op op = nir_intrinsic_alu_op(intr);
      enum glsl_base_type type = matrix_desc(intr->src[1]).element_type;
      nir_def *a = load_matrix(b, intr->src[1]);
      if (type == GLSL_TYPE_BFLOAT16)
         a = nir_bf2f(b, a);
      nir_def *value;
      if (intr->intrinsic == nir_intrinsic_cmat_unary_op) {
         value = nir_build_alu1(b, op, a);
      } else {
         nir_def *bb = intr->intrinsic == nir_intrinsic_cmat_scalar_op
            ? intr->src[2].ssa : load_matrix(b, intr->src[2]);
         if (type == GLSL_TYPE_BFLOAT16)
            bb = nir_bf2f(b, bb);
         value = nir_build_alu2(b, op, a, bb);
      }
      if (type == GLSL_TYPE_BFLOAT16)
         value = nir_f2bf(b, value);
      store_matrix(b, intr->src[0], value);
      b->fp_math_ctrl = nir_fp_fast_math;
      break;
   }
   case nir_intrinsic_cmat_convert:
   case nir_intrinsic_cmat_transpose: {
      struct glsl_cmat_description src = matrix_desc(intr->src[1]);
      struct glsl_cmat_description dst = matrix_desc(intr->src[0]);
      struct cmat_layout from = *matrix_layout(src);
      const struct cmat_layout *to = matrix_layout(dst);
      bool transpose = intr->intrinsic == nir_intrinsic_cmat_transpose;
      if (transpose) {
         for (unsigned i = 0; i < from.element_bits; ++i)
            from.element[i] = TRANSPOSED(from.element[i]);
         for (unsigned j = 0; j < 5; ++j)
            from.lane[j] = TRANSPOSED(from.lane[j]);
      }
      bool move = memcmp(&from, to, sizeof(from)) != 0;
      /* Convert where the elements are narrower */
      bool convert_first = glsl_base_type_bit_size(dst.element_type) <
                           glsl_base_type_bit_size(src.element_type);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      nir_def *value = load_matrix(b, intr->src[1]);
      if (convert_first)
         value = convert_matrix_value(b, value, src.element_type, dst.element_type);
      if (move)
         value = permute_matrix(b, value, from, to);
      if (!convert_first)
         value = convert_matrix_value(b, value, src.element_type, dst.element_type);
      store_matrix(b, intr->src[0], value);
      b->fp_math_ctrl = nir_fp_fast_math;
      break;
   }
   case nir_intrinsic_cmat_bitcast:
      store_matrix(b, intr->src[0], load_matrix(b, intr->src[1]));
      break;
   case nir_intrinsic_cmat_extract:
      nir_def_replace(&intr->def, nir_vector_extract(b, load_matrix(b, intr->src[0]), intr->src[1].ssa));
      return true;
   case nir_intrinsic_cmat_insert:
      store_matrix(b, intr->src[0], nir_vector_insert(b, load_matrix(b, intr->src[2]),
                                                     intr->src[1].ssa, intr->src[3].ssa));
      break;
   case nir_intrinsic_cmat_copy:
      store_matrix(b, intr->src[0], load_matrix(b, intr->src[1]));
      break;
   default:
      return false;
   }
   nir_instr_remove(instr);
   return true;
}

bool
agx_nir_lower_cmat(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || !nir->info.cs.has_cooperative_matrix)
      return false;
   struct hash_table *types = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp)
      var->type = remap_matrix_type(types, var->type);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl)
         var->type = remap_matrix_type(types, var->type);
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block_reverse_safe(block, impl) {
         nir_foreach_instr_reverse_safe(instr, block)
            lower_matrix_instr(&b, instr, types);
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   _mesa_hash_table_destroy(types, NULL);
   return true;
}

static bool
type_is_supported(const struct glsl_type *type)
{
   if (glsl_type_is_cmat(type)) {
      const struct glsl_cmat_description *desc = glsl_get_cmat_description(type);
      return (desc->rows == 8 || desc->rows == 16) && desc->cols == desc->rows &&
             desc->scope == SCOPE_SUBGROUP;
   } else if (glsl_type_is_array(type)) {
      return type_is_supported(glsl_get_array_element(type));
   } else if (glsl_type_is_struct(type)) {
      for (unsigned i = 0; i < glsl_get_length(type); i++) {
         if (!type_is_supported(glsl_get_struct_field(type, i)))
            return false;
      }
   }
   return true;
}

/* Whether the lowering can take every cooperative matrix of the shader: only
 * subgroup-scope 8x8 and 16x16 ones, and no matrix call left
 * (agx_nir_lower_cmat_calls() leaves those it cannot lower, and none of the
 * others is lowered at all).
 */
bool
agx_nir_cmat_supported(nir_shader *nir)
{
   if (!nir->info.cs.has_cooperative_matrix)
      return true;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref &&
                !type_is_supported(nir_instr_as_deref(instr)->type))
               return false;
            if (instr->type == nir_instr_type_cmat_call)
               return false;
         }
      }
   }
   return true;
}
