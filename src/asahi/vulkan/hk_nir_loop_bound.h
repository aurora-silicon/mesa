/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string.h>
#include "compiler/nir/nir.h"
#include "hk_pack_loop_bound.h"

struct hk_nir_loop_param_binding {
   nir_loop *loop;
   enum hk_pack_loop_param param;
};

#define HK_NIR_LOOP_MAX_BINDINGS 128u

struct hk_nir_loop_binding_list {
   struct hk_nir_loop_param_binding entries[HK_NIR_LOOP_MAX_BINDINGS];
   unsigned count;
   bool overflow;
};

static inline void
hk_nir_loop_binding_add(struct hk_nir_loop_binding_list *list,
                        nir_loop *loop, enum hk_pack_loop_param param)
{
   if (!list || !loop)
      return;
   if (list->count >= HK_NIR_LOOP_MAX_BINDINGS) {
      list->overflow = true;
      return;
   }
   list->entries[list->count++] = (struct hk_nir_loop_param_binding){
      .loop = loop,
      .param = param,
   };
}

struct hk_nir_lb_builder {
   const nir_shader *nir;
   struct hk_pack_loop_op *ops;
   uint8_t *count;
   uint8_t max_ops;
   const struct hk_nir_loop_param_binding *bindings;
   unsigned num_bindings;
   bool failed;
};

static inline int
hk_nir_lb_emit(struct hk_nir_lb_builder *b, struct hk_pack_loop_op op)
{
   if (b->failed)
      return -1;
   for (uint8_t i = 0; i < *b->count; ++i) {
      const struct hk_pack_loop_op *e = &b->ops[i];
      if (e->kind == op.kind && e->lhs == op.lhs && e->rhs == op.rhs &&
          e->imm == op.imm)
         return (int)i;
   }
   if (*b->count >= b->max_ops) {
      b->failed = true;
      return -1;
   }
   const int idx = (int)(*b->count)++;
   b->ops[idx] = op;
   return idx;
}

static inline int
hk_nir_lb_const(struct hk_nir_lb_builder *b, uint32_t imm)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_CONST, .imm = imm,
   });
}

static inline int
hk_nir_lb_param(struct hk_nir_lb_builder *b, enum hk_pack_loop_param param)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_PARAM, .imm = (uint32_t)param,
   });
}

static inline bool
hk_nir_lb_is_const(const struct hk_nir_lb_builder *b, int idx, uint32_t *imm_out)
{
   if (idx < 0 || b->failed)
      return false;
   const struct hk_pack_loop_op op = b->ops[idx];
   if (op.kind != HK_PACK_LOOP_OP_CONST)
      return false;
   if (imm_out)
      *imm_out = op.imm;
   return true;
}

static inline int
hk_nir_lb_add(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc && (uint64_t)a + (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a + c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_ADD, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_mul(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if ((ca && a == 0) || (cc && c == 0))
      return hk_nir_lb_const(b, 0);
   if (ca && a == 1)
      return rhs;
   if (cc && c == 1)
      return lhs;
   if (ca && cc && (uint64_t)a * (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a * c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MUL, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_max(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   if (lhs == rhs)
      return lhs;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc)
      return hk_nir_lb_const(b, a >= c ? a : c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MAX, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

/* Affine symbolic value: pure constant `c` when `node < 0`, or `ops[node] + c`
 * when `node >= 0`. Keeping additive constants out-of-band avoids emitting
 * dead intermediate CONST ops during CF walks, preventing op-budget overflows
 * on multi-branch shaders. */
struct hk_nir_lb_val {
   int node;
   uint64_t c;
};

static inline struct hk_nir_lb_val
hk_nir_lb_val_const(uint64_t c)
{
   return (struct hk_nir_lb_val){ .node = -1, .c = c };
}

static inline int
hk_nir_lb_val_materialize(struct hk_nir_lb_builder *b, struct hk_nir_lb_val v)
{
   if (b->failed)
      return -1;
   if (v.c > UINT32_MAX) {
      b->failed = true;
      return -1;
   }
   if (v.node < 0)
      return hk_nir_lb_const(b, (uint32_t)v.c);
   if (v.c == 0)
      return v.node;
   return hk_nir_lb_add(b, v.node, hk_nir_lb_const(b, (uint32_t)v.c));
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_add(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   uint64_t sum_c = 0;
   if (__builtin_add_overflow(a.c, d.c, &sum_c) || sum_c > UINT32_MAX) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   if (a.node < 0 && d.node < 0)
      return hk_nir_lb_val_const(sum_c);
   if (a.node < 0)
      return (struct hk_nir_lb_val){ .node = d.node, .c = sum_c };
   if (d.node < 0)
      return (struct hk_nir_lb_val){ .node = a.node, .c = sum_c };
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_add(b, a.node, d.node),
      .c = sum_c,
   };
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_max(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   if (a.node < 0 && d.node < 0)
      return hk_nir_lb_val_const(a.c >= d.c ? a.c : d.c);
   if (a.node == d.node)
      return (struct hk_nir_lb_val){ .node = a.node, .c = a.c >= d.c ? a.c : d.c };
   /* All symbolic ops are non-negative, so ops[node] + c >= c. */
   if (a.node >= 0 && d.node < 0 && d.c <= a.c)
      return a;
   if (d.node >= 0 && a.node < 0 && a.c <= d.c)
      return d;
   const uint64_t min_c = a.c < d.c ? a.c : d.c;
   const int la = hk_nir_lb_val_materialize(
      b, (struct hk_nir_lb_val){ .node = a.node, .c = a.c - min_c });
   const int ld = hk_nir_lb_val_materialize(
      b, (struct hk_nir_lb_val){ .node = d.node, .c = d.c - min_c });
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_max(b, la, ld),
      .c = min_c,
   };
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_mul(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   if ((a.node < 0 && a.c == 0) || (d.node < 0 && d.c == 0))
      return hk_nir_lb_val_const(0);
   if (a.node < 0 && d.node < 0) {
      uint64_t prod = 0;
      if (__builtin_mul_overflow(a.c, d.c, &prod) || prod > UINT32_MAX) {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
      return hk_nir_lb_val_const(prod);
   }
   if (a.node < 0 && a.c == 1)
      return d;
   if (d.node < 0 && d.c == 1)
      return a;
   if (a.node < 0 && a.c <= UINT32_MAX) {
      uint64_t prod_c = 0;
      if (!__builtin_mul_overflow(a.c, d.c, &prod_c) && prod_c <= UINT32_MAX) {
         return (struct hk_nir_lb_val){
            .node = hk_nir_lb_mul(b, hk_nir_lb_const(b, (uint32_t)a.c), d.node),
            .c = prod_c,
         };
      }
   }
   if (d.node < 0 && d.c <= UINT32_MAX) {
      uint64_t prod_c = 0;
      if (!__builtin_mul_overflow(d.c, a.c, &prod_c) && prod_c <= UINT32_MAX) {
         return (struct hk_nir_lb_val){
            .node = hk_nir_lb_mul(b, hk_nir_lb_const(b, (uint32_t)d.c), a.node),
            .c = prod_c,
         };
      }
   }
   const int la = hk_nir_lb_val_materialize(b, a);
   const int ld = hk_nir_lb_val_materialize(b, d);
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_mul(b, la, ld),
      .c = 0,
   };
}

static inline bool
hk_nir_lb_list_is_lone_break(struct exec_list *list)
{
   if (exec_list_is_empty(list) ||
       exec_list_get_head(list) != exec_list_get_tail(list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *tb = nir_cf_node_as_block(n);
   if (exec_list_is_empty(&tb->instr_list) ||
       exec_list_get_head(&tb->instr_list) !=
          exec_list_get_tail(&tb->instr_list))
      return false;
   nir_instr *ti =
      exec_node_data(nir_instr, exec_list_get_head(&tb->instr_list), node);
   return ti->type == nir_instr_type_jump &&
          nir_instr_as_jump(ti)->type == nir_jump_break;
}

static inline bool
hk_nir_lb_list_is_empty_block(struct exec_list *list)
{
   if (exec_list_is_empty(list) ||
       exec_list_get_head(list) != exec_list_get_tail(list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return n->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(n)->instr_list);
}

static inline bool
hk_nir_lb_is_simple_break_if(nir_if *nif, bool *break_in_then)
{
   if (hk_nir_lb_list_is_lone_break(&nif->then_list) &&
       hk_nir_lb_list_is_empty_block(&nif->else_list)) {
      *break_in_then = true;
      return true;
   }
   if (hk_nir_lb_list_is_lone_break(&nif->else_list) &&
       hk_nir_lb_list_is_empty_block(&nif->then_list)) {
      *break_in_then = false;
      return true;
   }
   return false;
}

static inline bool
hk_nir_lb_is_zero_when_base0(nir_scalar s, unsigned depth)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_scalar_is_const(s))
      return nir_scalar_as_uint(s) == 0;
   if (!depth)
      return false;
   if (nir_scalar_is_intrinsic(s))
      return nir_scalar_intrinsic_op(s) == nir_intrinsic_load_base_workgroup_id;
   if (!nir_scalar_is_alu(s))
      return false;
   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl)
      return hk_nir_lb_is_zero_when_base0(nir_scalar_chase_alu_src(s, 0), depth - 1);
   if (op == nir_op_imul || op == nir_op_amul) {
      return hk_nir_lb_is_zero_when_base0(nir_scalar_chase_alu_src(s, 0), depth - 1) ||
             hk_nir_lb_is_zero_when_base0(nir_scalar_chase_alu_src(s, 1), depth - 1);
   }
   return false;
}

static inline bool
hk_nir_lb_is_wg_id_axis(nir_scalar s, unsigned *axis_out, unsigned depth)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_scalar_is_intrinsic(s) &&
       nir_scalar_intrinsic_op(s) == nir_intrinsic_load_workgroup_id &&
       s.comp < 3) {
      *axis_out = s.comp;
      return true;
   }
   if (depth && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      if (hk_nir_lb_is_zero_when_base0(s0, depth - 1))
         return hk_nir_lb_is_wg_id_axis(s1, axis_out, depth - 1);
      if (hk_nir_lb_is_zero_when_base0(s1, depth - 1))
         return hk_nir_lb_is_wg_id_axis(s0, axis_out, depth - 1);
   }
   return false;
}

static inline bool
hk_nir_lb_is_num_wg_axis(nir_scalar s, unsigned axis)
{
   s = nir_scalar_resolved(s.def, s.comp);
   return nir_scalar_is_intrinsic(s) &&
          nir_scalar_intrinsic_op(s) == nir_intrinsic_load_num_workgroups &&
          s.comp == axis;
}

static inline bool
hk_nir_lb_eval_const_u32(nir_scalar s, uint64_t *val_out, unsigned depth)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_scalar_is_const(s)) {
      *val_out = nir_scalar_as_uint(s);
      return true;
   }
   if (depth && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      uint64_t a = 0, c = 0;
      if (hk_nir_lb_eval_const_u32(nir_scalar_chase_alu_src(s, 0), &a, depth - 1) &&
          hk_nir_lb_eval_const_u32(nir_scalar_chase_alu_src(s, 1), &c, depth - 1)) {
         *val_out = (uint32_t)(a + c);
         return true;
      }
   }
   return false;
}

static inline bool
hk_nir_lb_push_offset(nir_scalar s, uint32_t *off_out)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (!nir_scalar_is_intrinsic(s))
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(s.def));
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   uint64_t src_off = 0;
   if (!hk_nir_lb_eval_const_u32(nir_get_scalar(intr->src[0].ssa, 0), &src_off, 4))
      return false;
   const uint64_t off = (uint64_t)nir_intrinsic_base(intr) + src_off +
                        (uint64_t)s.comp * 4ull;
   if (off > 252ull || (off % 4ull) != 0)
      return false;
   *off_out = (uint32_t)off;
   return true;
}

static inline bool
hk_nir_lb_same_scalar_or_push(nir_scalar a, nir_scalar b, unsigned depth)
{
   a = nir_scalar_resolved(a.def, a.comp);
   b = nir_scalar_resolved(b.def, b.comp);
   if (nir_scalar_equal(a, b))
      return true;
   if (nir_scalar_is_const(a) && nir_scalar_is_const(b))
      return nir_scalar_as_uint(a) == nir_scalar_as_uint(b);
   uint32_t oa = 0, ob = 0;
   if (hk_nir_lb_push_offset(a, &oa) && hk_nir_lb_push_offset(b, &ob))
      return oa == ob;
   if (depth && nir_scalar_is_alu(a) && nir_scalar_is_alu(b) &&
       nir_scalar_alu_op(a) == nir_scalar_alu_op(b)) {
      const nir_op op = nir_scalar_alu_op(a);
      if (op == nir_op_iadd || op == nir_op_imul || op == nir_op_amul) {
         nir_scalar a0 = nir_scalar_chase_alu_src(a, 0);
         nir_scalar a1 = nir_scalar_chase_alu_src(a, 1);
         nir_scalar b0 = nir_scalar_chase_alu_src(b, 0);
         nir_scalar b1 = nir_scalar_chase_alu_src(b, 1);
         return (hk_nir_lb_same_scalar_or_push(a0, b0, depth - 1) &&
                 hk_nir_lb_same_scalar_or_push(a1, b1, depth - 1)) ||
                (hk_nir_lb_same_scalar_or_push(a0, b1, depth - 1) &&
                 hk_nir_lb_same_scalar_or_push(a1, b0, depth - 1));
      }
   }
   return false;
}

/* Return true when `init_s` is guaranteed >= 0 at dispatch time (with
 * base_group == 0 and grid * wg_size <= INT32_MAX). */
static inline bool
hk_nir_lb_init_nonneg(nir_scalar init_s, nir_scalar step_s, bool is_unsigned,
                      unsigned depth)
{
   if (is_unsigned)
      return true;
   init_s = nir_scalar_resolved(init_s.def, init_s.comp);
   if (nir_scalar_is_const(init_s))
      return nir_scalar_as_int(init_s) >= 0;
   if (nir_scalar_is_intrinsic(init_s)) {
      switch (nir_scalar_intrinsic_op(init_s)) {
      case nir_intrinsic_load_local_invocation_id:
      case nir_intrinsic_load_local_invocation_index:
      case nir_intrinsic_load_workgroup_id:
      case nir_intrinsic_load_global_invocation_id:
      case nir_intrinsic_load_base_workgroup_id:
      case nir_intrinsic_load_subgroup_invocation:
      case nir_intrinsic_load_subgroup_id:
         return true;
      default:
         return false;
      }
   }
   if (!depth || !nir_scalar_is_alu(init_s))
      return false;
   const nir_op op = nir_scalar_alu_op(init_s);
   if (op == nir_op_iadd) {
      nir_scalar s0 = nir_scalar_chase_alu_src(init_s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(init_s, 1);
      if (hk_nir_lb_is_zero_when_base0(s0, depth - 1))
         return hk_nir_lb_init_nonneg(s1, step_s, false, depth - 1);
      if (hk_nir_lb_is_zero_when_base0(s1, depth - 1))
         return hk_nir_lb_init_nonneg(s0, step_s, false, depth - 1);
   }
   /* Grid-stride scaled init: wg_id[axis] * F with step == num_wg[axis] * F.
    * Since OP_CEIL_DIV_I32 verifies step = num_wg[axis] * F <= INT32_MAX and
    * F >= 1, 0 <= wg_id[axis] * F < step <= INT32_MAX cannot wrap negative. */
   if (op == nir_op_imul || op == nir_op_amul) {
      step_s = nir_scalar_resolved(step_s.def, step_s.comp);
      if (nir_scalar_is_alu(step_s) &&
          (nir_scalar_alu_op(step_s) == nir_op_imul ||
           nir_scalar_alu_op(step_s) == nir_op_amul)) {
         nir_scalar i0 = nir_scalar_chase_alu_src(init_s, 0);
         nir_scalar i1 = nir_scalar_chase_alu_src(init_s, 1);
         nir_scalar st0 = nir_scalar_chase_alu_src(step_s, 0);
         nir_scalar st1 = nir_scalar_chase_alu_src(step_s, 1);
         for (unsigned a = 0; a < 2; ++a) {
            nir_scalar wg = a == 0 ? i0 : i1;
            nir_scalar fac = a == 0 ? i1 : i0;
            unsigned axis = 0;
            if (!hk_nir_lb_is_wg_id_axis(wg, &axis, 4))
               continue;
            for (unsigned c = 0; c < 2; ++c) {
               nir_scalar nwg = c == 0 ? st0 : st1;
               nir_scalar sfac = c == 0 ? st1 : st0;
               if (hk_nir_lb_is_num_wg_axis(nwg, axis) &&
                   hk_nir_lb_same_scalar_or_push(fac, sfac, 2))
                  return true;
            }
         }
      }
   }
   return false;
}

static inline bool
hk_nir_lb_is_const_div_or_shr(nir_scalar s, nir_scalar *num_out,
                              uint32_t *div_out, bool *is_signed_out)
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (!nir_scalar_is_alu(s) || s.def->bit_size != 32)
      return false;
   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ushr || op == nir_op_ishr) {
      nir_scalar v = nir_scalar_chase_alu_src(s, 0);
      nir_scalar k = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(k)) {
         const uint64_t sh = nir_scalar_as_uint(k);
         if (sh >= 1 && sh < 31) {
            /* Standard nir_opt_idiv_const lowering:
             * ishr(iadd(x, iand(ishr(x, 31), (1<<k)-1)), k) or
             * ishr(iadd(x, ushr(x, 31)), 1). */
            if (op == nir_op_ishr && nir_scalar_is_alu(v) &&
                nir_scalar_alu_op(v) == nir_op_iadd) {
               nir_scalar a0 = nir_scalar_chase_alu_src(v, 0);
               nir_scalar a1 = nir_scalar_chase_alu_src(v, 1);
               for (unsigned i = 0; i < 2; ++i) {
                  nir_scalar x = i == 0 ? a0 : a1;
                  nir_scalar bias = i == 0 ? a1 : a0;
                  if (sh == 1 && nir_scalar_is_alu(bias) &&
                      nir_scalar_alu_op(bias) == nir_op_ushr &&
                      nir_scalar_is_const(nir_scalar_chase_alu_src(bias, 1)) &&
                      nir_scalar_as_uint(nir_scalar_chase_alu_src(bias, 1)) == 31 &&
                      nir_scalar_equal(nir_scalar_chase_alu_src(bias, 0), x)) {
                     *num_out = x;
                     *div_out = 2u;
                     *is_signed_out = true;
                     return true;
                  }
               }
            }
            *num_out = v;
            *div_out = 1u << sh;
            *is_signed_out = (op == nir_op_ishr);
            return true;
         }
      }
      return false;
   }
   if (op == nir_op_idiv || op == nir_op_udiv) {
      nir_scalar v = nir_scalar_chase_alu_src(s, 0);
      nir_scalar d = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(d)) {
         const int64_t dv = op == nir_op_idiv ? nir_scalar_as_int(d)
                                              : (int64_t)nir_scalar_as_uint(d);
         if (dv >= 2 && dv <= UINT32_MAX) {
            *num_out = v;
            *div_out = (uint32_t)dv;
            *is_signed_out = (op == nir_op_idiv);
            return true;
         }
      }
      return false;
   }
   /* Lowered signed pow2 idiv: imul(imin(imax(x, -1), 1), ushr(imax(x, ineg(x)), k)). */
   if (op == nir_op_imul || op == nir_op_amul) {
      nir_scalar m0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar m1 = nir_scalar_chase_alu_src(s, 1);
      for (unsigned i = 0; i < 2; ++i) {
         nir_scalar sgn = i == 0 ? m0 : m1;
         nir_scalar mag = i == 0 ? m1 : m0;
         if (!nir_scalar_is_alu(mag) || nir_scalar_alu_op(mag) != nir_op_ushr)
            continue;
         nir_scalar k = nir_scalar_chase_alu_src(mag, 1);
         if (!nir_scalar_is_const(k))
            continue;
         const uint64_t sh = nir_scalar_as_uint(k);
         if (sh < 1 || sh >= 31)
            continue;
         nir_scalar absv = nir_scalar_chase_alu_src(mag, 0);
         nir_scalar x = {0};
         if (nir_scalar_is_alu(absv) && nir_scalar_alu_op(absv) == nir_op_iabs) {
            x = nir_scalar_chase_alu_src(absv, 0);
         } else if (nir_scalar_is_alu(absv) && nir_scalar_alu_op(absv) == nir_op_imax) {
            nir_scalar b0 = nir_scalar_chase_alu_src(absv, 0);
            nir_scalar b1 = nir_scalar_chase_alu_src(absv, 1);
            if (nir_scalar_is_alu(b1) && nir_scalar_alu_op(b1) == nir_op_ineg &&
                nir_scalar_equal(nir_scalar_chase_alu_src(b1, 0), b0))
               x = b0;
            else if (nir_scalar_is_alu(b0) && nir_scalar_alu_op(b0) == nir_op_ineg &&
                     nir_scalar_equal(nir_scalar_chase_alu_src(b0, 0), b1))
               x = b1;
         }
         if (!x.def)
            continue;
         /* Verify sgn == isign(x) = imin(imax(x, -1), 1) or isign(x). */
         bool ok_sgn = false;
         if (nir_scalar_is_alu(sgn) && nir_scalar_alu_op(sgn) == nir_op_isign &&
             nir_scalar_equal(nir_scalar_chase_alu_src(sgn, 0), x)) {
            ok_sgn = true;
         } else if (nir_scalar_is_alu(sgn) && nir_scalar_alu_op(sgn) == nir_op_imin) {
            nir_scalar n0 = nir_scalar_chase_alu_src(sgn, 0);
            nir_scalar n1 = nir_scalar_chase_alu_src(sgn, 1);
            nir_scalar inner = {0};
            if (nir_scalar_is_const(n1) && nir_scalar_as_int(n1) == 1)
               inner = n0;
            else if (nir_scalar_is_const(n0) && nir_scalar_as_int(n0) == 1)
               inner = n1;
            if (inner.def && nir_scalar_is_alu(inner) &&
                nir_scalar_alu_op(inner) == nir_op_imax) {
               nir_scalar p0 = nir_scalar_chase_alu_src(inner, 0);
               nir_scalar p1 = nir_scalar_chase_alu_src(inner, 1);
               if ((nir_scalar_is_const(p1) && nir_scalar_as_int(p1) == -1 &&
                    nir_scalar_equal(p0, x)) ||
                   (nir_scalar_is_const(p0) && nir_scalar_as_int(p0) == -1 &&
                    nir_scalar_equal(p1, x)))
                  ok_sgn = true;
            }
         }
         if (ok_sgn) {
            *num_out = x;
            *div_out = 1u << sh;
            *is_signed_out = true;
            return true;
         }
      }
   }
   return false;
}

static inline bool
hk_nir_lb_is_shr_of(nir_scalar cand, nir_scalar base)
{
   cand = nir_scalar_resolved(cand.def, cand.comp);
   if (!nir_scalar_is_alu(cand))
      return false;
   const nir_op op = nir_scalar_alu_op(cand);
   if ((op == nir_op_ishr || op == nir_op_ushr) &&
       nir_scalar_is_const(nir_scalar_chase_alu_src(cand, 1)) &&
       nir_scalar_equal(nir_scalar_chase_alu_src(cand, 0), base))
      return true;
   /* NIR lowers ishr(x, 16) to extract_i16(x, 1) or i2i32(unpack_32_2x16_split_y(x)). */
   if ((op == nir_op_extract_i16 || op == nir_op_extract_u16) &&
       nir_scalar_is_const(nir_scalar_chase_alu_src(cand, 1)) &&
       nir_scalar_as_uint(nir_scalar_chase_alu_src(cand, 1)) == 1 &&
       nir_scalar_equal(nir_scalar_chase_alu_src(cand, 0), base))
      return true;
   if ((op == nir_op_i2i32 || op == nir_op_u2u32)) {
      nir_scalar u = nir_scalar_chase_alu_src(cand, 0);
      if (nir_scalar_is_alu(u) &&
          nir_scalar_alu_op(u) == nir_op_unpack_32_2x16_split_y &&
          nir_scalar_equal(nir_scalar_chase_alu_src(u, 0), base))
         return true;
   }
   return false;
}

/* Match bit-smear `(base - 1) | (>>1) | ... | (>>16) == next_pow2(base) - 1`. */
static inline bool
hk_nir_lb_match_pow2_smear(nir_scalar sm, nir_scalar *base_out)
{
   sm = nir_scalar_resolved(sm.def, sm.comp);
   unsigned steps = 0;
   while (steps < 6 && nir_scalar_is_alu(sm) &&
          nir_scalar_alu_op(sm) == nir_op_ior) {
      nir_scalar o0 = nir_scalar_chase_alu_src(sm, 0);
      nir_scalar o1 = nir_scalar_chase_alu_src(sm, 1);
      if (hk_nir_lb_is_shr_of(o1, o0))
         sm = o0;
      else if (hk_nir_lb_is_shr_of(o0, o1))
         sm = o1;
      else
         break;
      steps++;
   }
   if (steps == 0 || !nir_scalar_is_alu(sm) ||
       nir_scalar_alu_op(sm) != nir_op_iadd)
      return false;
   nir_scalar b0 = nir_scalar_chase_alu_src(sm, 0);
   nir_scalar b1 = nir_scalar_chase_alu_src(sm, 1);
   if (nir_scalar_is_const(b0) && nir_scalar_as_int(b0) == -1) {
      *base_out = b1;
      return true;
   }
   if (nir_scalar_is_const(b1) && nir_scalar_as_int(b1) == -1) {
      *base_out = b0;
      return true;
   }
   return false;
}

static inline int
hk_nir_lb_emit_push_or_grid_expr(struct hk_nir_lb_builder *b, nir_scalar s,
                                 bool is_unsigned, bool is_limit,
                                 bool require_nonneg, unsigned depth)
{
   if (b->failed || !depth)
      return -1;
   s = nir_scalar_resolved(s.def, s.comp);
   if (s.def->bit_size != 32)
      return -1;

   if (nir_scalar_is_const(s)) {
      if (is_unsigned) {
         const uint32_t u = (uint32_t)nir_scalar_as_uint(s);
         if (!is_limit && u == 0)
            return -1;
         return hk_nir_lb_const(b, u);
      } else {
         const int32_t v = (int32_t)nir_scalar_as_int(s);
         if (!is_limit && v <= 0)
            return -1;
         if (require_nonneg && v < 0)
            return -1;
         return hk_nir_lb_const(b, v > 0 ? (uint32_t)v : 0u);
      }
   }

   if (nir_scalar_is_intrinsic(s)) {
      uint32_t off = 0;
      if (hk_nir_lb_push_offset(s, &off)) {
         return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
            .kind = is_unsigned
                       ? HK_PACK_LOOP_OP_PUSH_U32
                       : (require_nonneg ? HK_PACK_LOOP_OP_PUSH_I32_NONNEG
                                         : HK_PACK_LOOP_OP_PUSH_I32_POS),
            .imm = off,
         });
      }
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(s.def));
      if (intr->intrinsic == nir_intrinsic_load_num_workgroups && s.comp < 3) {
         return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
            .kind = HK_PACK_LOOP_OP_GRID_GROUPS,
            .imm = s.comp,
         });
      }
      if (intr->intrinsic == nir_intrinsic_load_workgroup_size && s.comp < 3 &&
          b->nir && !b->nir->info.workgroup_size_variable) {
         const uint32_t wg = b->nir->info.workgroup_size[s.comp];
         if (!is_limit && wg == 0)
            return -1;
         return hk_nir_lb_const(b, wg);
      }
      return -1;
   }

   if (!nir_scalar_is_alu(s))
      return -1;
   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_ishl) {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(sh))
         return -1;
      const uint64_t k = nir_scalar_as_uint(sh);
      if (k >= 31)
         return -1;
      const int lhs = hk_nir_lb_emit_push_or_grid_expr(
         b, nir_scalar_chase_alu_src(s, 0), is_unsigned, is_limit,
         require_nonneg, depth - 1);
      if (lhs < 0)
         return -1;
      return hk_nir_lb_mul(b, lhs, hk_nir_lb_const(b, 1u << k));
   }
   /* Constant division / right shift (e.g. radius = kernel_size / 2). */
   {
      nir_scalar num = {0};
      uint32_t div = 0;
      bool div_signed = false;
      if (hk_nir_lb_is_const_div_or_shr(s, &num, &div, &div_signed) &&
          div >= 2 && (!div_signed || !is_unsigned)) {
         const int num_op = hk_nir_lb_emit_push_or_grid_expr(
            b, num, !div_signed, is_limit, true, depth - 1);
         if (num_op >= 0) {
            return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_DIV_CONST,
               .lhs = (uint8_t)num_op,
               .imm = div,
            });
         }
      }
   }
   if (op == nir_op_imul || op == nir_op_amul) {
      const int lhs = hk_nir_lb_emit_push_or_grid_expr(
         b, nir_scalar_chase_alu_src(s, 0), is_unsigned, is_limit,
         require_nonneg, depth - 1);
      const int rhs = hk_nir_lb_emit_push_or_grid_expr(
         b, nir_scalar_chase_alu_src(s, 1), is_unsigned, is_limit,
         require_nonneg, depth - 1);
      if (lhs < 0 || rhs < 0)
         return -1;
      return hk_nir_lb_mul(b, lhs, rhs);
   }
   /* Upper-bound next_pow2(C) - 1 and next_pow2(C) <= 2*C (requiring C >= 0). */
   if (is_limit && !is_unsigned) {
      nir_scalar base = {0};
      if (op == nir_op_ior && hk_nir_lb_match_pow2_smear(s, &base)) {
         const int c_op = hk_nir_lb_emit_push_or_grid_expr(
            b, base, false, true, true, depth - 1);
         if (c_op >= 0)
            return hk_nir_lb_mul(b, c_op, hk_nir_lb_const(b, 2));
      }
      if (op == nir_op_iadd) {
         nir_scalar a0 = nir_scalar_chase_alu_src(s, 0);
         nir_scalar a1 = nir_scalar_chase_alu_src(s, 1);
         nir_scalar sm = {0};
         if (nir_scalar_is_const(a0) && nir_scalar_as_int(a0) == 1)
            sm = a1;
         else if (nir_scalar_is_const(a1) && nir_scalar_as_int(a1) == 1)
            sm = a0;
         if (sm.def && hk_nir_lb_match_pow2_smear(sm, &base)) {
            const int c_op = hk_nir_lb_emit_push_or_grid_expr(
               b, base, false, true, true, depth - 1);
            if (c_op >= 0)
               return hk_nir_lb_mul(b, c_op, hk_nir_lb_const(b, 2));
         }
      }
      /* Upper-bound `A - local_invocation_id` (or `A + ineg(lid)`) by upper
       * bound of `A` (since 0 <= lid < wg_size <= 1024, subtracting lid cannot
       * underflow INT32_MIN when A >= -1). */
      if ((op == nir_op_isub || op == nir_op_iadd) &&
          b->nir && !b->nir->info.workgroup_size_variable) {
         nir_scalar a = {0}, sub = {0};
         if (op == nir_op_isub) {
            a = nir_scalar_chase_alu_src(s, 0);
            sub = nir_scalar_chase_alu_src(s, 1);
         } else {
            nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
            nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
            if (nir_scalar_is_alu(s1) && nir_scalar_alu_op(s1) == nir_op_ineg) {
               a = s0;
               sub = nir_scalar_chase_alu_src(s1, 0);
            } else if (nir_scalar_is_alu(s0) && nir_scalar_alu_op(s0) == nir_op_ineg) {
               a = s1;
               sub = nir_scalar_chase_alu_src(s0, 0);
            }
         }
         if (sub.def && nir_scalar_is_intrinsic(sub) &&
             (nir_scalar_intrinsic_op(sub) == nir_intrinsic_load_local_invocation_id ||
              nir_scalar_intrinsic_op(sub) == nir_intrinsic_load_local_invocation_index)) {
            const int a_op = hk_nir_lb_emit_push_or_grid_expr(
               b, a, false, true, true, depth - 1);
            if (a_op >= 0)
               return a_op;
         }
      }
   }
   /* Non-negative additive limit expressions (e.g. `2 * radius + TILE_W` in
    * tiled stencil loaders). Overflow is checked at dispatch evaluation time
    * by HK_PACK_LOOP_OP_ADD and the enclosing CEIL_DIV_I32/U32. */
   if (is_limit && op == nir_op_iadd) {
      const uint8_t saved_count = *b->count;
      const bool saved_failed = b->failed;
      const int lhs = hk_nir_lb_emit_push_or_grid_expr(
         b, nir_scalar_chase_alu_src(s, 0), is_unsigned, true, true, depth - 1);
      const int rhs = lhs >= 0
                         ? hk_nir_lb_emit_push_or_grid_expr(
                              b, nir_scalar_chase_alu_src(s, 1), is_unsigned,
                              true, true, depth - 1)
                         : -1;
      if (lhs >= 0 && rhs >= 0 && !b->failed)
         return hk_nir_lb_add(b, lhs, rhs);
      *b->count = saved_count;
      b->failed = saved_failed;
   }
   return -1;
}

static inline int
hk_nir_lb_emit_grid_coord_max(struct hk_nir_lb_builder *b, nir_scalar s,
                              unsigned depth)
{
   if (b->failed || !depth)
      return -1;
   s = nir_scalar_resolved(s.def, s.comp);
   if (s.def->bit_size != 32)
      return -1;
   if (nir_scalar_is_const(s)) {
      const int32_t v = (int32_t)nir_scalar_as_int(s);
      return v >= 0 ? hk_nir_lb_const(b, (uint32_t)v) : -1;
   }
   if (nir_scalar_is_intrinsic(s)) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(s.def));
      switch (intr->intrinsic) {
      case nir_intrinsic_load_workgroup_id:
         if (s.comp < 3) {
            return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_GRID_GROUPS,
               .imm = s.comp,
            });
         }
         return -1;
      case nir_intrinsic_load_global_invocation_id:
         if (s.comp < 3 && b->nir && !b->nir->info.workgroup_size_variable) {
            const uint32_t wg = b->nir->info.workgroup_size[s.comp];
            const int g = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_GRID_GROUPS,
               .imm = s.comp,
            });
            return hk_nir_lb_mul(b, g, hk_nir_lb_const(b, wg));
         }
         return -1;
      case nir_intrinsic_load_local_invocation_id:
         if (s.comp < 3 && b->nir && !b->nir->info.workgroup_size_variable)
            return hk_nir_lb_const(b, b->nir->info.workgroup_size[s.comp]);
         return -1;
      case nir_intrinsic_load_local_invocation_index:
         if (b->nir && !b->nir->info.workgroup_size_variable) {
            const uint64_t sz = (uint64_t)b->nir->info.workgroup_size[0] *
                                b->nir->info.workgroup_size[1] *
                                b->nir->info.workgroup_size[2];
            if (sz <= UINT32_MAX)
               return hk_nir_lb_const(b, (uint32_t)sz);
         }
         return -1;
      case nir_intrinsic_load_base_workgroup_id:
         return hk_nir_lb_const(b, 0);
      default:
         return -1;
      }
   }
   if (!nir_scalar_is_alu(s))
      return -1;
   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_iadd) {
      nir_scalar s0 = nir_scalar_chase_alu_src(s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(s, 1);
      if (hk_nir_lb_is_zero_when_base0(s0, depth - 1))
         return hk_nir_lb_emit_grid_coord_max(b, s1, depth - 1);
      if (hk_nir_lb_is_zero_when_base0(s1, depth - 1))
         return hk_nir_lb_emit_grid_coord_max(b, s0, depth - 1);
      const int m0 = hk_nir_lb_emit_grid_coord_max(b, s0, depth - 1);
      const int m1 = hk_nir_lb_emit_grid_coord_max(b, s1, depth - 1);
      if (m0 >= 0 && m1 >= 0)
         return hk_nir_lb_add(b, m0, m1);
   }
   if (op == nir_op_imul || op == nir_op_amul) {
      const int m0 = hk_nir_lb_emit_grid_coord_max(b, nir_scalar_chase_alu_src(s, 0), depth - 1);
      const int m1 = hk_nir_lb_emit_grid_coord_max(b, nir_scalar_chase_alu_src(s, 1), depth - 1);
      if (m0 >= 0 && m1 >= 0)
         return hk_nir_lb_mul(b, m0, m1);
   }
   if (op == nir_op_ishl) {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) < 31) {
         const int m0 = hk_nir_lb_emit_grid_coord_max(b, nir_scalar_chase_alu_src(s, 0), depth - 1);
         if (m0 >= 0)
            return hk_nir_lb_mul(b, m0, hk_nir_lb_const(b, 1u << nir_scalar_as_uint(sh)));
      }
   }
   return -1;
}

static inline bool
hk_nir_lb_split_window_lo(nir_scalar s, nir_scalar *base_out,
                          nir_scalar *rad_out, uint32_t *const_rad_out)
{
   s = nir_scalar_resolved(s.def, s.comp);
   *rad_out = (nir_scalar){0};
   *const_rad_out = 0;
   if (nir_scalar_is_alu(s)) {
      const nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_isub) {
         *base_out = nir_scalar_chase_alu_src(s, 0);
         *rad_out = nir_scalar_chase_alu_src(s, 1);
         return true;
      }
      if (op == nir_op_iadd) {
         nir_scalar a0 = nir_scalar_chase_alu_src(s, 0);
         nir_scalar a1 = nir_scalar_chase_alu_src(s, 1);
         if (nir_scalar_is_alu(a1) && nir_scalar_alu_op(a1) == nir_op_ineg) {
            *base_out = a0;
            *rad_out = nir_scalar_chase_alu_src(a1, 0);
            return true;
         }
         if (nir_scalar_is_alu(a0) && nir_scalar_alu_op(a0) == nir_op_ineg) {
            *base_out = a1;
            *rad_out = nir_scalar_chase_alu_src(a0, 0);
            return true;
         }
         if (nir_scalar_is_const(a1) && nir_scalar_as_int(a1) <= 0 &&
             nir_scalar_as_int(a1) > INT32_MIN) {
            *base_out = a0;
            *const_rad_out = (uint32_t)(-nir_scalar_as_int(a1));
            return true;
         }
         if (nir_scalar_is_const(a0) && nir_scalar_as_int(a0) <= 0 &&
             nir_scalar_as_int(a0) > INT32_MIN) {
            *base_out = a1;
            *const_rad_out = (uint32_t)(-nir_scalar_as_int(a0));
            return true;
         }
      }
   }
   *base_out = s;
   return true;
}

/* Match a 2-source if-phi that implements signed min(A, B) (want_max=false)
 * or signed max(A, B) (want_max=true) prior to nir_opt_peephole_select. */
static inline bool
hk_nir_lb_extract_if_phi_minmax(nir_scalar s, bool want_max,
                                nir_scalar out_cands[2])
{
   s = nir_scalar_resolved(s.def, s.comp);
   if (nir_def_instr(s.def)->type != nir_instr_type_phi)
      return false;
   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(s.def));
   nir_block *merge = phi->instr.block;
   nir_cf_node *prev = nir_cf_node_prev(&merge->cf_node);
   if (!prev || prev->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(prev);
   nir_block *then_last = nir_if_last_then_block(nif);
   nir_block *else_last = nir_if_last_else_block(nif);
   nir_scalar then_v = {0}, else_v = {0};
   unsigned count = 0;
   nir_foreach_phi_src(ps, phi) {
      count++;
      if (ps->pred == then_last)
         then_v = nir_scalar_resolved(ps->src.ssa, s.comp);
      else if (ps->pred == else_last)
         else_v = nir_scalar_resolved(ps->src.ssa, s.comp);
   }
   if (count != 2 || !then_v.def || !else_v.def)
      return false;

   nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
   bool neg = false;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      neg = !neg;
      cond = nir_scalar_chase_alu_src(cond, 0);
   }
   if (!nir_scalar_is_alu(cond))
      return false;
   const nir_op cop = nir_scalar_alu_op(cond);
   if (cop != nir_op_ilt && cop != nir_op_ige)
      return false;
   nir_scalar c0 = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar c1 = nir_scalar_chase_alu_src(cond, 1);
   /* cond_lt == true means the then-branch runs when c0 < c1. */
   const bool cond_lt = (cop == nir_op_ilt) != neg;
   /* For min: when c0 < c1, then picks c0 and else picks c1 (or when c0 >= c1, then picks c1 and else picks c0).
    * For max: when c0 < c1, then picks c1 and else picks c0 (or when c0 >= c1, then picks c0 and else picks c1). */
   const bool then_picks_c0 = (cond_lt != want_max);
   nir_scalar expect_then = then_picks_c0 ? c0 : c1;
   nir_scalar expect_else = then_picks_c0 ? c1 : c0;
   if (hk_nir_lb_same_scalar_or_push(then_v, expect_then, 2) &&
       hk_nir_lb_same_scalar_or_push(else_v, expect_else, 2)) {
      out_cands[0] = then_v;
      out_cands[1] = else_v;
      return true;
   }
   return false;
}

/* Try bounding a signed sliding-window/stencil loop:
 *   init = [max(] base - rad_lo [, L)]
 *   lim  = [min(] base + rad_hi [, U)]
 * with base a non-negative grid coordinate. Verifies at dispatch time that
 * neither base - rad_lo nor base + rad_hi + strict_gt + step can wrap 32-bit
 * signed arithmetic, giving trips <= ceil((rad_lo + rad_hi + strict_gt) / step). */
static inline int
hk_nir_lb_try_window_trips(struct hk_nir_lb_builder *b, nir_scalar init_s,
                           nir_scalar limit_s, nir_scalar step_s,
                           bool strict_gt, bool bottom_tested)
{
   init_s = nir_scalar_resolved(init_s.def, init_s.comp);
   limit_s = nir_scalar_resolved(limit_s.def, limit_s.comp);

   nir_scalar lo_cands[2] = { init_s, {0} };
   unsigned n_lo = 1;
   if (nir_scalar_is_alu(init_s) && nir_scalar_alu_op(init_s) == nir_op_imax) {
      lo_cands[0] = nir_scalar_chase_alu_src(init_s, 0);
      lo_cands[1] = nir_scalar_chase_alu_src(init_s, 1);
      n_lo = 2;
   } else if (hk_nir_lb_extract_if_phi_minmax(init_s, true, lo_cands)) {
      n_lo = 2;
   }

   nir_scalar hi_cands[2] = { limit_s, {0} };
   unsigned n_hi = 1;
   if (nir_scalar_is_alu(limit_s) && nir_scalar_alu_op(limit_s) == nir_op_imin) {
      hi_cands[0] = nir_scalar_chase_alu_src(limit_s, 0);
      hi_cands[1] = nir_scalar_chase_alu_src(limit_s, 1);
      n_hi = 2;
   } else if (hk_nir_lb_extract_if_phi_minmax(limit_s, false, hi_cands)) {
      n_hi = 2;
   }

   for (unsigned li = 0; li < n_lo; ++li) {
      nir_scalar lo_base = {0}, rad_lo_s = {0};
      uint32_t rad_lo_c = 0;
      if (!hk_nir_lb_split_window_lo(lo_cands[li], &lo_base, &rad_lo_s, &rad_lo_c))
         continue;
      lo_base = nir_scalar_resolved(lo_base.def, lo_base.comp);

      for (unsigned hi = 0; hi < n_hi; ++hi) {
         nir_scalar hc = nir_scalar_resolved(hi_cands[hi].def, hi_cands[hi].comp);
         nir_scalar rad_hi_s = {0};
         bool matched = false;
         if (nir_scalar_equal(hc, lo_base)) {
            matched = true;
         } else if (nir_scalar_is_alu(hc) && nir_scalar_alu_op(hc) == nir_op_iadd) {
            nir_scalar h0 = nir_scalar_chase_alu_src(hc, 0);
            nir_scalar h1 = nir_scalar_chase_alu_src(hc, 1);
            if (nir_scalar_equal(h0, lo_base)) {
               rad_hi_s = h1;
               matched = true;
            } else if (nir_scalar_equal(h1, lo_base)) {
               rad_hi_s = h0;
               matched = true;
            }
         }
         if (!matched)
            continue;

         const uint8_t saved_count = *b->count;
         const bool saved_failed = b->failed;
         const int base_max = hk_nir_lb_emit_grid_coord_max(b, lo_base, 6);
         const int rlo = rad_lo_s.def
                            ? hk_nir_lb_emit_push_or_grid_expr(
                                 b, rad_lo_s, false, true, true, 8)
                            : hk_nir_lb_const(b, rad_lo_c);
         const int rhi = rad_hi_s.def
                            ? hk_nir_lb_emit_push_or_grid_expr(
                                 b, rad_hi_s, false, true, true, 8)
                            : hk_nir_lb_const(b, 0);
         const int stp = hk_nir_lb_emit_push_or_grid_expr(
            b, step_s, false, false, true, 8);
         if (base_max >= 0 && rlo >= 0 && rhi >= 0 && stp >= 0 && !b->failed) {
            const int chk_lo = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_ASSERT_I32,
               .lhs = (uint8_t)rlo,
            });
            int hi_sum = hk_nir_lb_add(b, base_max, rhi);
            if (strict_gt)
               hi_sum = hk_nir_lb_add(b, hi_sum, hk_nir_lb_const(b, 1));
            hi_sum = hk_nir_lb_add(b, hi_sum, stp);
            const int chk_hi = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_ASSERT_I32,
               .lhs = (uint8_t)hi_sum,
            });
            int span = hk_nir_lb_add(b, rlo, rhi);
            if (strict_gt)
               span = hk_nir_lb_add(b, span, hk_nir_lb_const(b, 1));
            /* Keep both overflow assertions live in the expression DAG (each
             * evaluates to 0 when in range and fails closed otherwise). */
            span = hk_nir_lb_add(b, span, chk_lo);
            span = hk_nir_lb_add(b, span, chk_hi);
            int trips = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
               .kind = HK_PACK_LOOP_OP_CEIL_DIV_I32,
               .lhs = (uint8_t)span,
               .rhs = (uint8_t)stp,
            });
            if (trips >= 0 && bottom_tested)
               trips = hk_nir_lb_max(b, trips, hk_nir_lb_const(b, 1));
            if (trips >= 0 && !b->failed)
               return trips;
         }
         *b->count = saved_count;
         b->failed = saved_failed;
      }
   }
   return -1;
}

static inline bool
hk_nir_lb_is_halving_update(nir_scalar update, const nir_def *iv_def)
{
   nir_scalar num = {0};
   uint32_t div = 0;
   bool is_signed = false;
   if (!hk_nir_lb_is_const_div_or_shr(update, &num, &div, &is_signed))
      return false;
   return num.def == iv_def && div >= 2;
}

static inline bool
hk_nir_lb_is_halving_break(struct hk_nir_lb_builder *b, nir_scalar cond,
                           bool break_on_true, nir_block *hdr, nir_block *pre,
                           uint64_t *const_trips_out, int *dyn_trips_out)
{
   *const_trips_out = 0;
   *dyn_trips_out = -1;
   cond = nir_scalar_resolved(cond.def, cond.comp);
   if (!nir_scalar_is_alu(cond))
      return false;
   const nir_op op = nir_scalar_alu_op(cond);
   if (op != nir_op_ilt && op != nir_op_ige && op != nir_op_ult &&
       op != nir_op_uge && op != nir_op_ieq && op != nir_op_ine)
      return false;
   nir_scalar s0 = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar s1 = nir_scalar_chase_alu_src(cond, 1);
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return false;

   /* Normalize to: breaks when `cand <= 0` or `cand == 0`. */
   nir_scalar cand = {0};
   bool ok_rel = false;
   if (nir_scalar_is_const(s1)) {
      const int64_t c = nir_scalar_as_int(s1);
      if (break_on_true)
         ok_rel = (op == nir_op_ieq && c == 0) || (op == nir_op_ilt && c == 1);
      else
         ok_rel = (op == nir_op_ine && c == 0) || (op == nir_op_ige && c == 1);
      if (ok_rel)
         cand = s0;
   } else if (nir_scalar_is_const(s0)) {
      const int64_t c = nir_scalar_as_int(s0);
      if (break_on_true)
         ok_rel = (op == nir_op_ieq && c == 0) || (op == nir_op_ige && c == 0);
      else
         ok_rel = (op == nir_op_ine && c == 0) || (op == nir_op_ilt && c == 0);
      if (ok_rel)
         cand = s1;
   }
   if (!ok_rel || !cand.def)
      return false;

   /* cand is either the header phi itself (top-tested) or its halving update (bottom-tested). */
   nir_phi_instr *iv = NULL;
   const nir_src *init_src = NULL;
   const nir_src *back_src = NULL;
   bool bottom_tested = false;
   if (nir_def_instr(cand.def)->type == nir_instr_type_phi &&
       nir_def_instr(cand.def)->block == hdr) {
      iv = nir_instr_as_phi(nir_def_instr(cand.def));
      unsigned src_count = 0;
      nir_foreach_phi_src(ps, iv) {
         src_count++;
         if (ps->pred == pre)
            init_src = &ps->src;
         else
            back_src = &ps->src;
      }
      if (src_count != 2 || !init_src || !back_src)
         return false;
   } else {
      bottom_tested = true;
      nir_foreach_phi(phi, hdr) {
         const nir_src *i_s = NULL, *b_s = NULL;
         unsigned src_count = 0;
         nir_foreach_phi_src(ps, phi) {
            src_count++;
            if (ps->pred == pre)
               i_s = &ps->src;
            else
               b_s = &ps->src;
         }
         if (src_count == 2 && i_s && b_s && b_s->ssa == cand.def) {
            iv = phi;
            init_src = i_s;
            back_src = b_s;
            break;
         }
      }
      if (!iv)
         return false;
   }
   if (!hk_nir_lb_is_halving_update(nir_get_scalar(back_src->ssa, 0), &iv->def))
      return false;
   nir_scalar init_s = nir_scalar_resolved(init_src->ssa, 0);
   if (nir_scalar_is_const(init_s)) {
      const int64_t v = nir_scalar_as_int(init_s);
      if (v > 0 && v <= INT32_MAX) {
         uint64_t t = 0;
         for (uint32_t u = (uint32_t)v; u > 0; u >>= 1)
            t++;
         *const_trips_out = t > 0 ? t : 1ull;
         return true;
      }
      if (v <= 0) {
         *const_trips_out = bottom_tested ? 1ull : 0ull;
         return true;
      }
   }
   /* Try dynamic floor(log2(init_upper)) + 1 when init is push/grid-bounded. */
   const uint8_t saved_count = *b->count;
   const bool saved_failed = b->failed;
   const int init_op = hk_nir_lb_emit_push_or_grid_expr(
      b, init_s, false, true, true, 8);
   if (init_op >= 0 && !b->failed) {
      int lg = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
         .kind = HK_PACK_LOOP_OP_LOG2_PLUS1,
         .lhs = (uint8_t)init_op,
      });
      if (lg >= 0 && bottom_tested)
         lg = hk_nir_lb_max(b, lg, hk_nir_lb_const(b, 1));
      if (lg >= 0 && !b->failed) {
         *dyn_trips_out = lg;
         return true;
      }
   }
   *b->count = saved_count;
   b->failed = saved_failed;
   *const_trips_out = 32ull;
   return true;
}

static inline int
hk_nir_lb_loop_trips_node(struct hk_nir_lb_builder *b, nir_loop *loop)
{
   nir_cf_node *prev_cf = nir_cf_node_prev(&loop->cf_node);
   if (!prev_cf || prev_cf->type != nir_cf_node_block) {
      b->failed = true;
      return -1;
   }
   nir_block *pre = nir_cf_node_as_block(prev_cf);
   if (exec_list_is_empty(&loop->body)) {
      b->failed = true;
      return -1;
   }
   nir_cf_node *first_cf =
      exec_node_data(nir_cf_node, exec_list_get_head(&loop->body), node);
   if (first_cf->type != nir_cf_node_block) {
      b->failed = true;
      return -1;
   }
   nir_block *hdr = nir_cf_node_as_block(first_cf);
   nir_cf_node *last_cf =
      exec_node_data(nir_cf_node, exec_list_get_tail(&loop->body), node);

   /* Find the loop's single simple break-if: either immediately after the header
    * block (top-tested or compact bottom-tested) or immediately before an empty
    * latch block (general bottom-tested loop with inner control flow). */
   nir_if *brk_if = NULL;
   bool break_in_then = true;
   nir_cf_node *second_cf = nir_cf_node_next(first_cf);
   if (second_cf && second_cf->type == nir_cf_node_if &&
       hk_nir_lb_is_simple_break_if(nir_cf_node_as_if(second_cf),
                                    &break_in_then)) {
      brk_if = nir_cf_node_as_if(second_cf);
   } else if (last_cf && last_cf->type == nir_cf_node_block &&
              exec_list_is_empty(&nir_cf_node_as_block(last_cf)->instr_list)) {
      nir_cf_node *penult = nir_cf_node_prev(last_cf);
      if (penult && penult->type == nir_cf_node_if &&
          hk_nir_lb_is_simple_break_if(nir_cf_node_as_if(penult),
                                       &break_in_then)) {
         brk_if = nir_cf_node_as_if(penult);
      }
   }
   if (!brk_if) {
      b->failed = true;
      return -1;
   }

   nir_scalar top_cond = nir_scalar_resolved(brk_if->condition.ssa, 0);
   bool break_on_true = break_in_then;
   while (nir_scalar_is_alu(top_cond) &&
          nir_scalar_alu_op(top_cond) == nir_op_inot) {
      break_on_true = !break_on_true;
      top_cond = nir_scalar_chase_alu_src(top_cond, 0);
   }

   /* If breaking occurs on a disjunction (A || B when break_on_true, or
    * !(A && B) when !break_on_true), proving a trip bound from any disjunct
    * is a sound upper bound. */
   nir_scalar cands[2] = { top_cond, {0} };
   unsigned num_cands = 1;
   if (nir_scalar_is_alu(top_cond) &&
       ((break_on_true && nir_scalar_alu_op(top_cond) == nir_op_ior) ||
        (!break_on_true && nir_scalar_alu_op(top_cond) == nir_op_iand))) {
      cands[0] = nir_scalar_chase_alu_src(top_cond, 0);
      cands[1] = nir_scalar_chase_alu_src(top_cond, 1);
      num_cands = 2;
   }

   bool found_canonical = false;
   bool found_const = false;
   uint64_t best_const_trips = UINT64_MAX;
   int dyn_trips_op = -1;

   for (unsigned ci = 0; ci < num_cands; ++ci) {
      nir_scalar cond_s = nir_scalar_resolved(cands[ci].def, cands[ci].comp);
      bool bot = break_on_true;
      while (nir_scalar_is_alu(cond_s) &&
             nir_scalar_alu_op(cond_s) == nir_op_inot) {
         bot = !bot;
         cond_s = nir_scalar_chase_alu_src(cond_s, 0);
      }
      if (!cond_s.def || !nir_scalar_is_alu(cond_s))
         continue;

      /* Halving loops (s = s >> 1 or s = s / 2 until s <= 0) terminate in
       * at most floor(log2(init)) + 1 (or 32) iterations. */
      uint64_t halving_trips = 0;
      int halving_dyn_op = -1;
      if (b->num_bindings == 0 &&
          hk_nir_lb_is_halving_break(b, cond_s, bot, hdr, pre,
                                     &halving_trips, &halving_dyn_op)) {
         found_canonical = true;
         if (halving_dyn_op >= 0) {
            if (dyn_trips_op < 0)
               dyn_trips_op = halving_dyn_op;
         } else if (halving_trips < best_const_trips) {
            best_const_trips = halving_trips;
            found_const = true;
         }
         continue;
      }

      const nir_op op = nir_scalar_alu_op(cond_s);
      if (op != nir_op_ige && op != nir_op_uge &&
          op != nir_op_ilt && op != nir_op_ult)
         continue;
      const bool is_unsigned = (op == nir_op_uge || op == nir_op_ult);
      const bool is_ge = (op == nir_op_ige || op == nir_op_uge);

      nir_scalar s0 = nir_scalar_chase_alu_src(cond_s, 0);
      nir_scalar s1 = nir_scalar_chase_alu_src(cond_s, 1);

      /* Try both operands as the IV (handles both counting-up `iv >= limit` /
       * `iv > limit` and counting-down `iv < limit` / `iv <= limit` loops). */
      for (unsigned side = 0; side < 2; ++side) {
         nir_scalar cmp_iv_s = side == 0 ? s0 : s1;
         nir_scalar limit_s = side == 0 ? s1 : s0;
         /* Determine whether breaking stops an increasing (counting_down=false)
          * or decreasing (counting_down=true) IV, and whether the boundary is
          * strict (`>` for up, `<=` for down). */
         bool counting_down = false;
         bool strict_bound = false;
         if (side == 0) {
            /* Comparing iv <op> limit.
             * Up breaks when iv >= limit: (bot && is_ge) || (!bot && !is_ge).
             * Down breaks when iv < limit: (bot && !is_ge) || (!bot && is_ge). */
            counting_down = (bot != is_ge);
            strict_bound = false;
         } else {
            /* Comparing limit <op> iv.
             * Up breaks when limit < iv (i.e. iv > limit): (bot && !is_ge) || (!bot && is_ge).
             * Down breaks when limit >= iv (i.e. iv <= limit): (bot && is_ge) || (!bot && !is_ge). */
            counting_down = (bot == is_ge);
            strict_bound = true;
         }

         const nir_def *cmp_lhs = cmp_iv_s.def;
         if (!cmp_lhs || cmp_lhs->num_components != 1 || cmp_lhs->bit_size != 32)
            continue;

         /* Accept either a top-tested loop comparing the header IV phi directly
          * or a bottom-tested loop comparing the post-update back-edge IV. */
         const nir_def *iv_def = cmp_lhs;
         bool bottom_tested = false;
         if (nir_def_instr(cmp_lhs)->type == nir_instr_type_alu) {
            nir_alu_instr *lhs_alu = nir_instr_as_alu(nir_def_instr(cmp_lhs));
            if (lhs_alu->op != nir_op_iadd && lhs_alu->op != nir_op_isub)
               continue;
            const nir_def *a0 = nir_scalar_chase_alu_src(cmp_iv_s, 0).def;
            const nir_def *a1 = nir_scalar_chase_alu_src(cmp_iv_s, 1).def;
            if (a0 && nir_def_instr(a0)->type == nir_instr_type_phi &&
                nir_def_instr(a0)->block == hdr) {
               iv_def = a0;
               bottom_tested = true;
            } else if (lhs_alu->op == nir_op_iadd &&
                       a1 && nir_def_instr(a1)->type == nir_instr_type_phi &&
                       nir_def_instr(a1)->block == hdr) {
               iv_def = a1;
               bottom_tested = true;
            } else {
               continue;
            }
         } else if (nir_def_instr(iv_def)->type != nir_instr_type_phi ||
                    nir_def_instr(iv_def)->block != hdr) {
            continue;
         }

         nir_phi_instr *iv = nir_instr_as_phi(nir_def_instr(iv_def));
         const nir_src *init_src = NULL;
         const nir_src *back_src = NULL;
         unsigned src_count = 0;
         nir_foreach_phi_src(ps, iv) {
            src_count++;
            if (ps->pred == pre)
               init_src = &ps->src;
            else
               back_src = &ps->src;
         }
         if (src_count != 2 || !init_src || !back_src)
            continue;
         nir_scalar back_s = nir_scalar_resolved(back_src->ssa, 0);
         if (!nir_scalar_is_alu(back_s))
            continue;
         if (bottom_tested && back_s.def != cmp_lhs)
            continue;

         nir_alu_instr *step_alu = nir_instr_as_alu(nir_def_instr(back_s.def));
         if (step_alu->op != nir_op_iadd && step_alu->op != nir_op_isub)
            continue;
         bool step_negated = false;
         nir_scalar step_s = {0};
         if (nir_scalar_chase_alu_src(back_s, 0).def == iv_def) {
            step_s = nir_scalar_chase_alu_src(back_s, 1);
            step_negated = (step_alu->op == nir_op_isub);
         } else if (step_alu->op == nir_op_iadd &&
                    nir_scalar_chase_alu_src(back_s, 1).def == iv_def) {
            step_s = nir_scalar_chase_alu_src(back_s, 0);
         } else {
            continue;
         }

         nir_scalar init_s = nir_scalar_resolved(init_src->ssa, 0);
         limit_s = nir_scalar_resolved(limit_s.def, limit_s.comp);
         step_s = nir_scalar_resolved(step_s.def, step_s.comp);

         if (counting_down) {
            if (is_unsigned || b->num_bindings != 0 || !nir_scalar_is_const(step_s))
               continue;
            const int64_t raw_step = nir_scalar_as_int(step_s);
            const int64_t mag = step_negated ? raw_step : -raw_step;
            if (mag <= 0 || mag > INT32_MAX)
               continue;
            found_canonical = true;
            if (nir_scalar_is_const(init_s) && nir_scalar_is_const(limit_s)) {
               const int64_t init_i = nir_scalar_as_int(init_s);
               const int64_t lim_i = nir_scalar_as_int(limit_s);
               /* Breaks when iv < lim_i (strict_bound=false) or iv <= lim_i (strict_bound=true). */
               const int64_t stop_i = lim_i + (strict_bound ? 1 : 0);
               if (stop_i >= (int64_t)INT32_MIN + mag) {
                  uint64_t trips = 0;
                  if (init_i >= stop_i)
                     trips = ((uint64_t)(init_i - stop_i) + (uint64_t)mag) / (uint64_t)mag;
                  if (bottom_tested && trips == 0)
                     trips = 1;
                  if (trips <= UINT32_MAX && trips < best_const_trips) {
                     best_const_trips = trips;
                     found_const = true;
                  }
               }
            } else if (dyn_trips_op < 0) {
               const uint8_t saved_count = *b->count;
               const bool saved_failed = b->failed;
               /* Require limit >= 0 (prevents signed underflow below INT32_MIN). */
               int lim_chk = -1;
               if (hk_nir_lb_init_nonneg(limit_s, step_s, false, 4)) {
                  lim_chk = hk_nir_lb_const(b, 0);
               } else {
                  const int l_op = hk_nir_lb_emit_push_or_grid_expr(
                     b, limit_s, false, true, true, 8);
                  if (l_op >= 0) {
                     lim_chk = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
                        .kind = HK_PACK_LOOP_OP_ASSERT_I32,
                        .lhs = (uint8_t)l_op,
                     });
                  }
               }
               int init_op = lim_chk >= 0
                                ? hk_nir_lb_emit_push_or_grid_expr(
                                     b, init_s, false, true, true, 8)
                                : -1;
               if (lim_chk >= 0 && init_op >= 0 && !b->failed) {
                  int span = strict_bound
                                ? init_op
                                : hk_nir_lb_add(b, init_op, hk_nir_lb_const(b, 1));
                  span = hk_nir_lb_add(b, span, lim_chk);
                  int cdiv = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
                     .kind = HK_PACK_LOOP_OP_CEIL_DIV_I32,
                     .lhs = (uint8_t)span,
                     .rhs = (uint8_t)hk_nir_lb_const(b, (uint32_t)mag),
                  });
                  if (cdiv >= 0 && bottom_tested)
                     cdiv = hk_nir_lb_max(b, cdiv, hk_nir_lb_const(b, 1));
                  if (cdiv >= 0 && !b->failed)
                     dyn_trips_op = cdiv;
               }
               if (dyn_trips_op < 0) {
                  *b->count = saved_count;
                  b->failed = saved_failed;
               }
            }
            continue;
         }

         /* Counting-up loop (`step > 0`). */
         if (step_negated)
            continue;
         const bool strict_gt = strict_bound;
         found_canonical = true;

         if (nir_scalar_is_const(init_s) && nir_scalar_is_const(step_s) &&
             nir_scalar_is_const(limit_s)) {
            uint64_t trips = 0;
            bool ok = true;
            if (!is_unsigned) {
               const int64_t init_i = nir_scalar_as_int(init_s);
               const int64_t step_i = nir_scalar_as_int(step_s);
               const int64_t limit_i = nir_scalar_as_int(limit_s) + (strict_gt ? 1 : 0);
               if (step_i <= 0 || limit_i > (int64_t)INT32_MAX + 1)
                  ok = false;
               else if (limit_i > init_i)
                  trips = ((uint64_t)(limit_i - init_i) + (uint64_t)step_i - 1ull) / (uint64_t)step_i;
            } else {
               const uint64_t init_u = nir_scalar_as_uint(init_s);
               const uint64_t step_u = nir_scalar_as_uint(step_s);
               const uint64_t limit_u = nir_scalar_as_uint(limit_s) + (strict_gt ? 1ull : 0ull);
               if (step_u == 0 || (strict_gt && nir_scalar_as_uint(limit_s) == UINT32_MAX))
                  ok = false;
               else if (limit_u > init_u) {
                  const uint64_t span = limit_u - init_u;
                  if (span > UINT64_MAX - (step_u - 1ull))
                     ok = false;
                  else
                     trips = (span + step_u - 1ull) / step_u;
               }
            }
            if (ok && bottom_tested && trips == 0)
               trips = 1;
            if (ok && trips <= UINT32_MAX && trips < best_const_trips) {
               best_const_trips = trips;
               found_const = true;
            }
         } else if (b->num_bindings == 0 && dyn_trips_op < 0) {
            /* Prefer tight sliding-window / stencil span bound (base - rad_lo .. base + rad_hi)
             * when applicable, before falling back to 0..limit. */
            if (!is_unsigned) {
               const int win_op = hk_nir_lb_try_window_trips(
                  b, init_s, limit_s, step_s, strict_gt, bottom_tested);
               if (win_op >= 0) {
                  dyn_trips_op = win_op;
                  continue;
               }
            }
            if (hk_nir_lb_init_nonneg(init_s, step_s, is_unsigned, 4)) {
               const uint8_t saved_count = *b->count;
               const bool saved_failed = b->failed;
               int lim_op = hk_nir_lb_emit_push_or_grid_expr(
                  b, limit_s, is_unsigned, true, false, 8);
               int stp_op = lim_op >= 0
                               ? hk_nir_lb_emit_push_or_grid_expr(
                                    b, step_s, is_unsigned, false, true, 8)
                               : -1;
               if (lim_op >= 0 && stp_op >= 0 && !b->failed) {
                  if (strict_gt)
                     lim_op = hk_nir_lb_add(b, lim_op, hk_nir_lb_const(b, 1));
                  int cdiv = hk_nir_lb_emit(b, (struct hk_pack_loop_op){
                     .kind = is_unsigned ? HK_PACK_LOOP_OP_CEIL_DIV_U32
                                         : HK_PACK_LOOP_OP_CEIL_DIV_I32,
                     .lhs = (uint8_t)lim_op,
                     .rhs = (uint8_t)stp_op,
                  });
                  if (cdiv >= 0 && bottom_tested)
                     cdiv = hk_nir_lb_max(b, cdiv, hk_nir_lb_const(b, 1));
                  if (cdiv >= 0 && !b->failed)
                     dyn_trips_op = cdiv;
               }
               if (dyn_trips_op < 0) {
                  *b->count = saved_count;
                  b->failed = saved_failed;
               }
            }
         }
      }
   }

   if (!found_canonical) {
      b->failed = true;
      return -1;
   }

   /* Check explicit dynamic parameter binding first (structure already verified). */
   for (unsigned i = 0; i < b->num_bindings; ++i) {
      if (b->bindings[i].loop == loop)
         return hk_nir_lb_param(b, b->bindings[i].param);
   }

   if (found_const)
      return hk_nir_lb_const(b, (uint32_t)best_const_trips);
   if (dyn_trips_op >= 0)
      return dyn_trips_op;

   b->failed = true;
   return -1;
}

static inline struct hk_nir_lb_val
hk_nir_lb_loop_trips(struct hk_nir_lb_builder *b, nir_loop *loop)
{
   const int node = hk_nir_lb_loop_trips_node(b, loop);
   if (node < 0 || b->failed)
      return hk_nir_lb_val_const(0);
   uint32_t c = 0;
   if (hk_nir_lb_is_const(b, node, &c))
      return hk_nir_lb_val_const(c);
   return (struct hk_nir_lb_val){ .node = node, .c = 0 };
}

static inline struct hk_nir_lb_val
hk_nir_lb_walk_list(struct hk_nir_lb_builder *b, const struct exec_list *cf_list)
{
   struct hk_nir_lb_val total = hk_nir_lb_val_const(0);
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      if (b->failed)
         return hk_nir_lb_val_const(0);
      if (cf->type == nir_cf_node_block) {
         continue;
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         const struct hk_nir_lb_val t = hk_nir_lb_walk_list(b, &nif->then_list);
         const struct hk_nir_lb_val e = hk_nir_lb_walk_list(b, &nif->else_list);
         total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_max(b, t, e));
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         const struct hk_nir_lb_val trips = hk_nir_lb_loop_trips(b, loop);
         const struct hk_nir_lb_val body = hk_nir_lb_walk_list(b, &loop->body);
         const struct hk_nir_lb_val per_iter =
            hk_nir_lb_val_add(b, hk_nir_lb_val_const(1), body);
         const struct hk_nir_lb_val loop_edges =
            hk_nir_lb_val_mul(b, trips, per_iter);
         total = hk_nir_lb_val_add(b, total, loop_edges);
      } else {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
   }
   return total;
}

static inline uint32_t
hk_nir_lb_block_weight(const nir_block *blk)
{
   uint32_t w = 0;
   nir_foreach_instr(instr, blk) {
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_tex:
      case nir_instr_type_jump:
         w++;
         break;
      case nir_instr_type_intrinsic: {
         const nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic == nir_intrinsic_cmat_muladd_agx ||
             intrin->intrinsic == nir_intrinsic_barrier)
            w += 4;
         else
            w++;
         break;
      }
      default:
         break;
      }
   }
   return w;
}

static inline struct hk_nir_lb_val
hk_nir_lb_walk_work_list(struct hk_nir_lb_builder *b,
                         const struct exec_list *cf_list)
{
   struct hk_nir_lb_val total = hk_nir_lb_val_const(0);
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      if (b->failed)
         return hk_nir_lb_val_const(0);
      if (cf->type == nir_cf_node_block) {
         const uint32_t w = hk_nir_lb_block_weight(nir_cf_node_as_block(cf));
         if (w > 0)
            total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_const(w));
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         const struct hk_nir_lb_val t = hk_nir_lb_walk_work_list(b, &nif->then_list);
         const struct hk_nir_lb_val e = hk_nir_lb_walk_work_list(b, &nif->else_list);
         total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_max(b, t, e));
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         const struct hk_nir_lb_val trips = hk_nir_lb_loop_trips(b, loop);
         const struct hk_nir_lb_val body = hk_nir_lb_walk_work_list(b, &loop->body);
         const nir_block *hdr = nir_loop_first_block(loop);
         const uint32_t hdr_w = hk_nir_lb_block_weight(hdr) + 1u;
         const struct hk_nir_lb_val loop_work =
            hk_nir_lb_val_add(b, hk_nir_lb_val_mul(b, trips, body),
                              hk_nir_lb_val_const(hdr_w));
         total = hk_nir_lb_val_add(b, total, loop_work);
      } else {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
   }
   return total;
}

/* Analyze the emitted NIR control-flow tree and populate `out` with an
 * expression computing worst-case (total loop back-edges + 1), plus a
 * CF-weighted dynamic executable NIR op bound (`work_ops` / `static_nir_ops`).
 * Fails closed (`out->valid = false`) if any loop cannot be bounded statically
 * or via a verified dynamic parameter binding. */
static inline bool
hk_nir_compute_pack_loop_bound(nir_shader *nir,
                               const struct hk_nir_loop_param_binding *bindings,
                               unsigned num_bindings,
                               struct hk_pack_loop_bound *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!nir)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
   if (!impl)
      return false;

   struct hk_nir_lb_builder b = {
      .nir = nir,
      .ops = out->ops,
      .count = &out->count,
      .max_ops = HK_PACK_LOOP_MAX_OPS,
      .bindings = bindings,
      .num_bindings = num_bindings,
      .failed = false,
   };

   const struct hk_nir_lb_val edges = hk_nir_lb_walk_list(&b, &impl->body);
   const struct hk_nir_lb_val bound_v =
      hk_nir_lb_val_add(&b, edges, hk_nir_lb_val_const(1));
   const int bound = hk_nir_lb_val_materialize(&b, bound_v);
   if (b.failed || bound < 0 || out->count == 0) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   /* Ensure the final expression value sits at ops[count - 1]. */
   if (bound != (int)out->count - 1) {
      if (out->count >= HK_PACK_LOOP_MAX_OPS) {
         memset(out, 0, sizeof(*out));
         return false;
      }
      out->ops[out->count] = out->ops[bound];
      out->count++;
   }
   out->valid = true;

   /* Best-effort CF-weighted executable NIR op bound (falls back cleanly to
    * back-edges + 1 if work_ops exceeds HK_PACK_LOOP_WORK_MAX_OPS). */
   uint64_t static_ops = 0;
   nir_foreach_block(blk, impl) {
      static_ops += hk_nir_lb_block_weight(blk);
   }
   if (static_ops > 0 && static_ops <= UINT32_MAX) {
      struct hk_nir_lb_builder wb = {
         .nir = nir,
         .ops = out->work_ops,
         .count = &out->work_count,
         .max_ops = HK_PACK_LOOP_WORK_MAX_OPS,
         .bindings = bindings,
         .num_bindings = num_bindings,
         .failed = false,
      };
      const struct hk_nir_lb_val work_v =
         hk_nir_lb_walk_work_list(&wb, &impl->body);
      const int work = hk_nir_lb_val_materialize(&wb, work_v);
      if (!wb.failed && work >= 0 && out->work_count > 0) {
         bool ok = true;
         if (work != (int)out->work_count - 1) {
            if (out->work_count >= HK_PACK_LOOP_WORK_MAX_OPS) {
               ok = false;
            } else {
               out->work_ops[out->work_count] = out->work_ops[work];
               out->work_count++;
            }
         }
         if (ok) {
            out->static_nir_ops = (uint32_t)static_ops;
            out->work_valid = true;
         } else {
            out->work_count = 0;
         }
      } else {
         out->work_count = 0;
      }
   }
   return true;
}
