/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_shared_zero_init.h"

#include "nir.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"

#define HK_SHARED_ZERO_INIT_MAX_BYTES   (64u * 1024u)
#define HK_SHARED_ZERO_INIT_MAX_DWORDS  (HK_SHARED_ZERO_INIT_MAX_BYTES / 4u)
#define HK_SHARED_ZERO_INIT_MAX_THREADS 1024u
#define HK_SHARED_ZERO_INIT_MAX_ITERS   64u
#define HK_SHARED_ZERO_INIT_MAX_DEPTH   16u

struct count_loop {
   nir_def *iv;
   nir_def *init_def;
   uint32_t step;
   int64_t limit;
   bool is_unsigned;
};

/* Evaluate a pure workgroup-local integer expression for thread (x, y, z, idx)
 * and optional counting-loop induction variable iv_val, matching exact 32-bit
 * GPU wrap-around semantics. The result is the value zero-extended from
 * def->bit_size. Arithmetic is only evaluated at 32 bits. Returns false on any
 * non-local or unsupported op. */
static bool
eval_u32(nir_def *def, uint8_t swiz, nir_def *iv, uint32_t x, uint32_t y,
         uint32_t z, uint32_t idx, uint32_t iv_val, unsigned depth,
         uint32_t *out)
{
   if (!def || swiz >= def->num_components ||
       depth > HK_SHARED_ZERO_INIT_MAX_DEPTH)
      return false;

   if (iv && def == iv) {
      if (swiz != 0)
         return false;
      *out = iv_val;
      return true;
   }

   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const: {
      nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      if (def->bit_size > 32)
         return false;
      *out = (uint32_t)nir_const_value_as_uint(lc->value[swiz], def->bit_size);
      return true;
   }

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      /* Local IDs and indices are below 1024, so 16 and 32 bits are exact. */
      if (def->bit_size != 16 && def->bit_size != 32)
         return false;
      if (intr->intrinsic == nir_intrinsic_load_local_invocation_index) {
         if (swiz != 0)
            return false;
         *out = idx;
         return true;
      }
      if (intr->intrinsic == nir_intrinsic_load_local_invocation_id) {
         if (swiz == 0)
            *out = x;
         else if (swiz == 1)
            *out = y;
         else if (swiz == 2)
            *out = z;
         else
            return false;
         return true;
      }
      return false;
   }

   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_vec2 || alu->op == nir_op_vec3 ||
          alu->op == nir_op_vec4) {
         return eval_u32(alu->src[swiz].src.ssa, alu->src[swiz].swizzle[0], iv,
                         x, y, z, idx, iv_val, depth + 1, out);
      }

      const unsigned nsrc = nir_op_infos[alu->op].num_inputs;
      if (nsrc == 0 || nsrc > 3)
         return false;

      uint32_t s[3] = {0, 0, 0};
      for (unsigned i = 0; i < nsrc; ++i) {
         if (!eval_u32(alu->src[i].src.ssa, alu->src[i].swizzle[swiz], iv, x, y,
                       z, idx, iv_val, depth + 1, &s[i]))
            return false;
      }

      /* Conversions extend from the source width. Everything else is only
       * evaluated at 32 bits, where uint32_t arithmetic matches the GPU. */
      const unsigned src_bits = alu->src[0].src.ssa->bit_size;
      switch (alu->op) {
      case nir_op_mov:
         *out = s[0];
         return true;
      case nir_op_u2u32:
         if (src_bits > 32)
            return false;
         *out = s[0];
         return true;
      case nir_op_i2i32:
         if (src_bits > 32)
            return false;
         *out = (uint32_t)util_sign_extend(s[0], src_bits);
         return true;
      case nir_op_u2u16:
         if (src_bits > 32)
            return false;
         *out = s[0] & 0xffffu;
         return true;
      default:
         break;
      }

      if (alu->def.bit_size != 32)
         return false;

      switch (alu->op) {
      case nir_op_iadd:
         *out = s[0] + s[1];
         return true;
      case nir_op_isub:
         *out = s[0] - s[1];
         return true;
      case nir_op_imul:
      case nir_op_amul:
         *out = s[0] * s[1];
         return true;
      case nir_op_imad:
         *out = s[0] * s[1] + s[2];
         return true;
      case nir_op_ishl:
         *out = s[0] << (s[1] & 31u);
         return true;
      case nir_op_ushr:
         *out = s[0] >> (s[1] & 31u);
         return true;
      case nir_op_iand:
         *out = s[0] & s[1];
         return true;
      case nir_op_ior:
         *out = s[0] | s[1];
         return true;
      default:
         return false;
      }
   }

   default:
      return false;
   }
}

/* Quick structural check that def depends only on constants, local thread
 * coordinates, and optional iv before running per-thread evaluation. */
static bool
expr_is_local(nir_def *def, nir_def *iv)
{
   uint32_t dummy;
   return eval_u32(def, 0, iv, 0, 0, 0, 0, 0, 0, &dummy);
}

static bool
instr_blocks_elision(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_jump || instr->type == nir_instr_type_call)
      return true;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_shared_ir3:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
      return true;
   default:
      return false;
   }
}

static bool
is_workgroup_shared_barrier(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_barrier)
      return false;
   return nir_intrinsic_execution_scope(intr) >= SCOPE_WORKGROUP &&
          nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP &&
          (nir_intrinsic_memory_modes(intr) & nir_var_mem_shared) &&
          (nir_intrinsic_memory_semantics(intr) & NIR_MEMORY_ACQUIRE) &&
          (nir_intrinsic_memory_semantics(intr) & NIR_MEMORY_RELEASE);
}

static nir_scalar
chase_mov(nir_scalar s)
{
   for (unsigned depth = 0; depth < HK_SHARED_ZERO_INIT_MAX_DEPTH; ++depth) {
      if (!s.def || s.comp >= s.def->num_components)
         break;
      nir_alu_instr *alu = nir_def_as_alu_or_null(s.def);
      /* Only plain moves: a conversion would change the width or signedness
       * the caller reads a constant at. */
      if (!alu || alu->op != nir_op_mov)
         return s;

      s = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[s.comp]);
   }
   return (nir_scalar){0};
}

/* The matched comparisons and iadd are per-component ALU operations. */
static nir_scalar
chase_alu_src(const nir_alu_src *src, unsigned comp)
{
   return chase_mov(nir_get_scalar(src->src.ssa, src->swizzle[comp]));
}

static bool
match_iv_const(nir_alu_instr *alu, unsigned comp, nir_def *iv,
               bool *iv_lhs, nir_scalar *constant)
{
   const nir_scalar lhs = chase_alu_src(&alu->src[0], comp);
   const nir_scalar rhs = chase_alu_src(&alu->src[1], comp);

   *iv_lhs = lhs.def == iv && lhs.comp == 0;
   if (!*iv_lhs && (rhs.def != iv || rhs.comp != 0))
      return false;

   *constant = *iv_lhs ? rhs : lhs;
   return constant->def && constant->def->bit_size == iv->bit_size &&
          nir_scalar_is_const(*constant);
}

static bool
match_break_cond(nir_def *cond, bool invert, nir_def *iv, int64_t *limit_out,
                 bool *unsigned_out)
{
   nir_scalar s = chase_mov(nir_get_scalar(cond, 0));
   nir_alu_instr *alu;
   for (unsigned depth = 0; ; ++depth) {
      if (depth >= HK_SHARED_ZERO_INIT_MAX_DEPTH || !s.def)
         return false;
      alu = nir_def_as_alu_or_null(s.def);
      if (!alu)
         return false;
      if (alu->op != nir_op_inot)
         break;

      invert = !invert;
      s = chase_alu_src(&alu->src[0], s.comp);
   }

   if (alu->op != nir_op_ige && alu->op != nir_op_uge &&
       alu->op != nir_op_ilt && alu->op != nir_op_ult)
      return false;

   bool iv_lhs;
   nir_scalar constant;
   if (!match_iv_const(alu, s.comp, iv, &iv_lhs, &constant))
      return false;

   /* Break condition:
    *   iv >= limit:
    *     !invert: ige(iv, L), uge(iv, L), ilt(L-1, iv), ult(L-1, iv)
    *      invert: ilt(iv, L), ult(iv, L), ige(L-1, iv), uge(L-1, iv)
    */
   bool is_unsigned = false;
   int64_t adj = 0;
   if (iv_lhs) {
      if (!invert && (alu->op == nir_op_ige || alu->op == nir_op_uge)) {
         is_unsigned = (alu->op == nir_op_uge);
      } else if (invert && (alu->op == nir_op_ilt || alu->op == nir_op_ult)) {
         is_unsigned = (alu->op == nir_op_ult);
      } else {
         return false;
      }
   } else {
      if (!invert && (alu->op == nir_op_ilt || alu->op == nir_op_ult)) {
         is_unsigned = (alu->op == nir_op_ult);
         adj = 1;
      } else if (invert && (alu->op == nir_op_ige || alu->op == nir_op_uge)) {
         is_unsigned = (alu->op == nir_op_uge);
         adj = 1;
      } else {
         return false;
      }
   }

   const int64_t lim =
      (is_unsigned ? (int64_t)nir_scalar_as_uint(constant)
                   : nir_scalar_as_int(constant)) +
      adj;
   if (lim <= 0 || lim > (int64_t)HK_SHARED_ZERO_INIT_MAX_BYTES)
      return false;

   *limit_out = lim;
   *unsigned_out = is_unsigned;
   return true;
}

static bool
match_counting_loop(nir_loop *loop, struct count_loop *out)
{
   if (nir_loop_has_continue_construct(loop))
      return false;

   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);

   nir_cf_node *second = nir_cf_node_next(&header->cf_node);
   if (!second || second->type != nir_cf_node_if)
      return false;
   nir_if *break_if = nir_cf_node_as_if(second);

   nir_block *then_blk = nir_if_first_then_block(break_if);
   nir_block *else_blk = nir_if_first_else_block(break_if);
   if (then_blk != nir_if_last_then_block(break_if) ||
       else_blk != nir_if_last_else_block(break_if))
      return false;

   bool invert = false;
   if (nir_block_ends_in_break(then_blk) &&
       exec_list_is_empty(&else_blk->instr_list)) {
      invert = false;
   } else if (nir_block_ends_in_break(else_blk) &&
              exec_list_is_empty(&then_blk->instr_list)) {
      invert = true;
   } else {
      return false;
   }

   nir_block *brk_blk = invert ? else_blk : then_blk;
   nir_foreach_instr(instr, brk_blk) {
      if (instr->type != nir_instr_type_jump)
         return false;
   }

   nir_cf_node *after = nir_cf_node_next(&break_if->cf_node);
   if (!after || after != &latch->cf_node || nir_cf_node_next(after) != NULL)
      return false;

   nir_foreach_instr(instr, header) {
      if (instr_blocks_elision(instr) || is_workgroup_shared_barrier(instr))
         return false;
   }
   nir_foreach_instr(instr, latch) {
      if (instr_blocks_elision(instr) || is_workgroup_shared_barrier(instr))
         return false;
   }

   nir_foreach_phi(phi, header) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32)
         continue;
      if (exec_list_length(&phi->srcs) != 2)
         continue;

      nir_def *init_def = NULL;
      nir_def *latch_def = NULL;
      nir_foreach_phi_src(src, phi) {
         if (src->pred == latch)
            latch_def = src->src.ssa;
         else
            init_def = src->src.ssa;
      }
      if (!init_def || !latch_def)
         continue;

      int64_t limit = 0;
      bool is_unsigned = false;
      if (!match_break_cond(break_if->condition.ssa, invert, &phi->def, &limit,
                            &is_unsigned))
         continue;

      const nir_scalar next = chase_mov(nir_get_scalar(latch_def, 0));
      nir_alu_instr *step_alu = next.def ? nir_def_as_alu_or_null(next.def)
                                       : NULL;
      if (!step_alu || step_alu->op != nir_op_iadd)
         continue;

      bool iv_lhs;
      nir_scalar constant;
      if (!match_iv_const(step_alu, next.comp, &phi->def, &iv_lhs, &constant))
         continue;
      const int64_t step = nir_scalar_as_int(constant);
      if (step <= 0 || step > (int64_t)HK_SHARED_ZERO_INIT_MAX_BYTES)
         continue;

      if (!expr_is_local(init_def, NULL))
         continue;

      *out = (struct count_loop){
         .iv = &phi->def,
         .init_def = init_def,
         .step = (uint32_t)step,
         .limit = limit,
         .is_unsigned = is_unsigned,
      };
      return true;
   }

   return false;
}

static void
mark_store_coverage(nir_intrinsic_instr *intr, const struct count_loop *loop,
                    const uint16_t wg[3], uint32_t shared_size,
                    uint64_t *covered)
{
   if (intr->intrinsic != nir_intrinsic_store_shared)
      return;

   const unsigned bit_size = intr->src[0].ssa->bit_size;
   if (bit_size != 8 && bit_size != 16 && bit_size != 32)
      return;
   const nir_component_mask_t wrmask = nir_intrinsic_write_mask(intr);
   if (!wrmask)
      return;

   const unsigned comps = util_bitcount(wrmask);
   if (wrmask != BITFIELD_MASK(comps))
      return;
   const uint32_t store_bytes = (comps * bit_size) / 8u;
   if (store_bytes == 0 || (store_bytes & 3u))
      return;

   nir_def *off_def = intr->src[1].ssa;
   if (!expr_is_local(off_def, loop ? loop->iv : NULL))
      return;

   const uint32_t base = (uint32_t)nir_intrinsic_base(intr);
   const uint32_t shared_dwords = shared_size / 4u;
   const uint32_t store_dwords = store_bytes / 4u;

   for (uint32_t z = 0; z < wg[2]; ++z) {
      for (uint32_t y = 0; y < wg[1]; ++y) {
         for (uint32_t x = 0; x < wg[0]; ++x) {
            const uint32_t idx = (z * wg[1] + y) * wg[0] + x;
            if (!loop) {
               uint32_t rel = 0;
               if (!eval_u32(off_def, 0, NULL, x, y, z, idx, 0, 0, &rel))
                  return;
               const uint64_t byte_off = (uint64_t)rel + base;
               if ((byte_off & 3u) || byte_off + store_bytes > shared_size)
                  continue;
               const uint32_t dw = (uint32_t)(byte_off / 4u);
               for (uint32_t d = 0; d < store_dwords && dw + d < shared_dwords;
                    ++d)
                  covered[(dw + d) / 64u] |= BITFIELD64_BIT((dw + d) % 64u);
            } else {
               uint32_t iv_u = 0;
               if (!eval_u32(loop->init_def, 0, NULL, x, y, z, idx, 0, 0,
                             &iv_u))
                  return;
               int64_t iv = loop->is_unsigned ? (int64_t)iv_u
                                              : (int64_t)(int32_t)iv_u;
               for (unsigned iter = 0; iter < HK_SHARED_ZERO_INIT_MAX_ITERS;
                    ++iter) {
                  if (iv >= loop->limit)
                     break;
                  uint32_t rel = 0;
                  if (!eval_u32(off_def, 0, loop->iv, x, y, z, idx,
                                (uint32_t)iv, 0, &rel))
                     return;
                  const uint64_t byte_off = (uint64_t)rel + base;
                  if (!(byte_off & 3u) &&
                      byte_off + store_bytes <= shared_size) {
                     const uint32_t dw = (uint32_t)(byte_off / 4u);
                     for (uint32_t d = 0;
                          d < store_dwords && dw + d < shared_dwords; ++d)
                        covered[(dw + d) / 64u] |=
                           BITFIELD64_BIT((dw + d) % 64u);
                  }
                  iv += loop->step;
               }
            }
         }
      }
   }
}

static bool
all_dwords_covered(const uint64_t *covered, uint32_t shared_dwords)
{
   const uint32_t full_words = shared_dwords / 64u;
   for (uint32_t w = 0; w < full_words; ++w) {
      if (covered[w] != UINT64_MAX)
         return false;
   }
   const uint32_t rem = shared_dwords % 64u;
   if (rem &&
       (covered[full_words] & BITFIELD64_MASK(rem)) != BITFIELD64_MASK(rem))
      return false;
   return true;
}

bool
hk_nir_shared_memory_fully_initialized(const nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       nir->info.workgroup_size_variable || nir->info.shared_size == 0 ||
       (nir->info.shared_size & 3u) ||
       nir->info.shared_size > HK_SHARED_ZERO_INIT_MAX_BYTES)
      return false;

   const uint16_t wg[3] = {
      nir->info.workgroup_size[0],
      nir->info.workgroup_size[1],
      nir->info.workgroup_size[2],
   };
   const uint32_t local_count = (uint32_t)wg[0] * wg[1] * wg[2];
   if (!local_count || local_count > HK_SHARED_ZERO_INIT_MAX_THREADS)
      return false;

   const nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
   if (!impl)
      return false;

   uint64_t covered[HK_SHARED_ZERO_INIT_MAX_DWORDS / 64u] = {0};
   const uint32_t shared_size = nir->info.shared_size;
   const uint32_t shared_dwords = shared_size / 4u;

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (is_workgroup_shared_barrier(instr))
               return all_dwords_covered(covered, shared_dwords);
            if (instr_blocks_elision(instr))
               return false;
            if (instr->type == nir_instr_type_intrinsic) {
               mark_store_coverage(nir_instr_as_intrinsic(instr), NULL, wg,
                                   shared_size, covered);
            }
         }
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         struct count_loop cl;
         if (!match_counting_loop(loop, &cl))
            return false;

         for (uint32_t z = 0; z < wg[2]; ++z) {
            for (uint32_t y = 0; y < wg[1]; ++y) {
               for (uint32_t x = 0; x < wg[0]; ++x) {
                  const uint32_t idx = (z * wg[1] + y) * wg[0] + x;
                  uint32_t iv_u = 0;
                  if (!eval_u32(cl.init_def, 0, NULL, x, y, z, idx, 0, 0,
                                &iv_u))
                     return false;
                  const int64_t iv0 = cl.is_unsigned
                                         ? (int64_t)iv_u
                                         : (int64_t)(int32_t)iv_u;
                  if (iv0 < cl.limit) {
                     const int64_t span = cl.limit - iv0;
                     const int64_t iters = (span + cl.step - 1) / cl.step;
                     if (iters > (int64_t)HK_SHARED_ZERO_INIT_MAX_ITERS)
                        return false;
                  }
               }
            }
         }

         nir_block *latch = nir_loop_last_block(loop);
         nir_foreach_instr(instr, latch) {
            if (instr->type == nir_instr_type_intrinsic) {
               mark_store_coverage(nir_instr_as_intrinsic(instr), &cl, wg,
                                   shared_size, covered);
            }
         }
         break;
      }

      default:
         return false;
      }
   }

   return false;
}
