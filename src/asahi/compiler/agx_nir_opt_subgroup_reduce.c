/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * AGX3C intra-subgroup shared-memory tree reduction lowering and single-lane
 * guarded select sinking.
 *
 * 1. agx_nir_opt_shared_tree_reduce():
 *    Recognizes provably intra-subgroup (stride S = 2^k <= 16, halving down to
 *    1) shared-memory tree reductions in convergent compute control flow where
 *    intermediate slots 1..2*S-1 are provably dead outside the reduction.
 *    Lowers associative-commutative 32-bit integer reductions to native
 *    nir_intrinsic_reduce (single sRed* instruction) when all 2*S lanes of the
 *    subgroup participate, and lowers floating-point (and partial-subgroup)
 *    reductions to an exact shuffle_xor butterfly (simdbfly with immediate
 *    masks S, S/2, ..., 1) preserving the exact pairwise evaluation order,
 *    operand order, and float-math controls of the original tree bit-for-bit.
 *
 * 2. agx_nir_opt_sink_single_lane_select():
 *    In multi-subgroup compute shaders (workgroup invocations > 32), when a
 *    large pure ALU dependency cone feeds only bcsel(tid == c, val, fallback)
 *    instructions (typically produced by peephole-select flattening an
 *    if (tid == 0) prologue), sinks the private cone back into an if (tid == c)
 *    block so all other subgroups skip it via jmp_exec_none.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"

#define MAX_REDUCE_ARRAYS 8
#define MAX_REDUCE_STEPS  5 /* strides 16, 8, 4, 2, 1 */
#define SINGLE_LANE_SINK_MIN_ALU 12

struct tid_info {
   bool one_d;
   unsigned invocations;
};

static bool
is_tid_scalar(const struct tid_info *t, nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   while (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_u2u32 ||
                                   nir_scalar_alu_op(s) == nir_op_u2u16 ||
                                   nir_scalar_alu_op(s) == nir_op_i2i32))
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));

   if (!nir_scalar_is_intrinsic(s))
      return false;

   nir_intrinsic_op op = nir_scalar_intrinsic_op(s);
   if (op == nir_intrinsic_load_local_invocation_index)
      return true;
   return op == nir_intrinsic_load_local_invocation_id && t->one_d && s.comp == 0;
}

/* Parse a linear function of tid: (scale * tid + constant). */
static bool
parse_tid_linear(const struct tid_info *t, nir_scalar s, unsigned depth,
                 uint64_t *scale_out, uint64_t *const_out)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      *scale_out = 0;
      *const_out = nir_scalar_as_uint(s);
      return true;
   }
   if (is_tid_scalar(t, s)) {
      *scale_out = 1;
      *const_out = 0;
      return true;
   }
   if (!depth || !nir_scalar_is_alu(s))
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   switch (op) {
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_i2i32: {
      uint64_t sc, co;
      if (!parse_tid_linear(t, nir_scalar_chase_alu_src(s, 0), depth - 1, &sc, &co))
         return false;
      if (op == nir_op_u2u16) {
         uint64_t max_val = co + sc * (t->invocations ? t->invocations - 1 : 0);
         if (max_val > UINT16_MAX)
            return false;
      }
      *scale_out = sc;
      *const_out = co;
      return true;
   }
   case nir_op_iadd: {
      uint64_t s0, c0, s1, c1;
      if (!parse_tid_linear(t, nir_scalar_chase_alu_src(s, 0), depth - 1, &s0, &c0) ||
          !parse_tid_linear(t, nir_scalar_chase_alu_src(s, 1), depth - 1, &s1, &c1))
         return false;
      if (s0 + s1 > UINT32_MAX || c0 + c1 > UINT32_MAX)
         return false;
      *scale_out = s0 + s1;
      *const_out = c0 + c1;
      return true;
   }
   case nir_op_ishl: {
      nir_scalar sh = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(sh))
         return false;
      uint64_t k = nir_scalar_as_uint(sh);
      if (k >= 32)
         return false;
      uint64_t sc, co;
      if (!parse_tid_linear(t, nir_scalar_chase_alu_src(s, 0), depth - 1, &sc, &co))
         return false;
      if ((sc << k) > UINT32_MAX || (co << k) > UINT32_MAX)
         return false;
      *scale_out = sc << k;
      *const_out = co << k;
      return true;
   }
   case nir_op_imul:
   case nir_op_amul: {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(a) && !nir_scalar_is_const(b)) {
         nir_scalar tmp = a;
         a = b;
         b = tmp;
      }
      if (!nir_scalar_is_const(b))
         return false;
      uint64_t m = nir_scalar_as_uint(b);
      uint64_t sc, co;
      if (!parse_tid_linear(t, a, depth - 1, &sc, &co))
         return false;
      if (sc * m > UINT32_MAX || co * m > UINT32_MAX)
         return false;
      *scale_out = sc * m;
      *const_out = co * m;
      return true;
   }
   case nir_op_imadshl_agx: {
      /* a * b + (c << sh) */
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      nir_scalar c = nir_scalar_chase_alu_src(s, 2);
      nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
      if (!nir_scalar_is_const(sh))
         return false;
      uint64_t k = nir_scalar_as_uint(sh);
      if (k >= 32)
         return false;
      uint64_t sm = 0, cm = 0;
      if ((nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 0) ||
          (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 0)) {
         sm = 0;
         cm = 0;
      } else {
         if (nir_scalar_is_const(a) && !nir_scalar_is_const(b)) {
            nir_scalar tmp = a;
            a = b;
            b = tmp;
         }
         if (!nir_scalar_is_const(b))
            return false;
         uint64_t m = nir_scalar_as_uint(b);
         if (!parse_tid_linear(t, a, depth - 1, &sm, &cm))
            return false;
         if (sm * m > UINT32_MAX || cm * m > UINT32_MAX)
            return false;
         sm *= m;
         cm *= m;
      }
      uint64_t sc, cc;
      if (!parse_tid_linear(t, c, depth - 1, &sc, &cc))
         return false;
      if ((sc << k) > UINT32_MAX || (cc << k) > UINT32_MAX)
         return false;
      if (sm + (sc << k) > UINT32_MAX || cm + (cc << k) > UINT32_MAX)
         return false;
      *scale_out = sm + (sc << k);
      *const_out = cm + (cc << k);
      return true;
   }
   default:
      return false;
   }
}

struct shared_addr {
   uint32_t base_bytes;
   uint32_t tid_stride_bytes;
   uint32_t access_bytes;
   uint8_t num_components;
   uint8_t bit_size;
};

static bool
parse_shared_access(const nir_shader *shader, const struct tid_info *t,
                    const nir_intrinsic_instr *intr, struct shared_addr *out)
{
   bool is_load = intr->intrinsic == nir_intrinsic_load_shared;
   bool is_store = intr->intrinsic == nir_intrinsic_store_shared;
   if (!is_load && !is_store)
      return false;
   if (nir_intrinsic_access(intr) != 0)
      return false;

   unsigned bit_size = is_load ? intr->def.bit_size : nir_src_bit_size(intr->src[0]);
   unsigned comps = is_load ? intr->def.num_components : nir_src_num_components(intr->src[0]);
   if (bit_size != 32 || comps < 1 || comps > 4)
      return false;
   if (is_store && nir_intrinsic_write_mask(intr) != BITFIELD_MASK(comps))
      return false;

   int base = nir_intrinsic_base(intr);
   if (base < 0)
      return false;

   nir_def *idx_def = is_load ? intr->src[0].ssa : intr->src[1].ssa;
   uint64_t scale = 0, c = 0;
   if (!parse_tid_linear(t, nir_get_scalar(idx_def, 0), 8, &scale, &c))
      return false;

   uint64_t unit_bytes = shader->info.shared_element_index_agx ? (bit_size / 8) : 1;
   uint64_t byte_off = (uint64_t)base + c * unit_bytes;
   uint64_t stride_bytes = scale * unit_bytes;
   uint64_t access_bytes = (uint64_t)comps * (bit_size / 8);

   if (byte_off > UINT32_MAX || stride_bytes > UINT32_MAX ||
       byte_off + access_bytes > UINT32_MAX)
      return false;

   *out = (struct shared_addr){
      .base_bytes = (uint32_t)byte_off,
      .tid_stride_bytes = (uint32_t)stride_bytes,
      .access_bytes = (uint32_t)access_bytes,
      .num_components = (uint8_t)comps,
      .bit_size = (uint8_t)bit_size,
   };
   return true;
}

/* Exact check: cond == value iff tid < bound (with 1 <= bound <= 32). */
static bool
cond_exact_tid_lt(const struct tid_info *t, nir_scalar cond, bool value,
                  unsigned depth, uint32_t *bound_out)
{
   cond = nir_scalar_chase_movs(cond);
   if (!depth || !nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_inot)
      return cond_exact_tid_lt(t, nir_scalar_chase_alu_src(cond, 0), !value,
                               depth - 1, bound_out);
   if (nir_op_infos[op].num_inputs != 2)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   bool tid_a = is_tid_scalar(t, a);
   bool tid_b = is_tid_scalar(t, b);
   if ((!tid_a || !nir_scalar_is_const(b)) &&
       (!tid_b || !nir_scalar_is_const(a)))
      return false;

   uint64_t c = nir_scalar_as_uint(tid_a ? b : a);
   if (c > 32)
      return false;

   switch (op) {
   case nir_op_ieq:
      if (value && c == 0) {
         *bound_out = 1;
         return true;
      }
      return false;
   case nir_op_ine:
      if (!value && c == 0) {
         *bound_out = 1;
         return true;
      }
      return false;
   case nir_op_ult:
   case nir_op_ilt:
      if (tid_a && value && c >= 1 && c <= 32) {
         *bound_out = (uint32_t)c;
         return true;
      }
      if (!tid_a && !value && c < 32) {
         *bound_out = (uint32_t)(c + 1);
         return true;
      }
      return false;
   case nir_op_uge:
   case nir_op_ige:
      if (tid_a && !value && c >= 1 && c <= 32) {
         *bound_out = (uint32_t)c;
         return true;
      }
      if (!tid_a && value && c < 32) {
         *bound_out = (uint32_t)(c + 1);
         return true;
      }
      return false;
   default:
      return false;
   }
}

/* Exact check: cond == value iff tid == lane (with lane < invocations). */
static bool
cond_exact_tid_eq(const struct tid_info *t, nir_scalar cond, bool value,
                  unsigned depth, uint32_t *lane_out)
{
   cond = nir_scalar_chase_movs(cond);
   if (!depth || !nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_inot)
      return cond_exact_tid_eq(t, nir_scalar_chase_alu_src(cond, 0), !value,
                               depth - 1, lane_out);
   if (nir_op_infos[op].num_inputs != 2)
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(cond, 0);
   nir_scalar b = nir_scalar_chase_alu_src(cond, 1);
   bool tid_a = is_tid_scalar(t, a);
   bool tid_b = is_tid_scalar(t, b);
   if ((!tid_a || !nir_scalar_is_const(b)) &&
       (!tid_b || !nir_scalar_is_const(a)))
      return false;

   uint64_t c = nir_scalar_as_uint(tid_a ? b : a);
   if (c >= t->invocations)
      return false;

   if ((op == nir_op_ieq && value) || (op == nir_op_ine && !value)) {
      *lane_out = (uint32_t)c;
      return true;
   }
   /* tid < 1 is also tid == 0. */
   if (c == 1 && tid_a &&
       ((value && (op == nir_op_ult || op == nir_op_ilt)) ||
        (!value && (op == nir_op_uge || op == nir_op_ige)))) {
      *lane_out = 0;
      return true;
   }
   if (c == 0 && !tid_a &&
       ((!value && (op == nir_op_ult || op == nir_op_ilt)) ||
        (value && (op == nir_op_uge || op == nir_op_ige)))) {
      *lane_out = 0;
      return true;
   }
   return false;
}

static bool
is_guarded_tid_eq0(const struct tid_info *t, const nir_instr *instr)
{
   const nir_cf_node *child = &instr->block->cf_node;
   for (const nir_cf_node *node = child->parent; node;
        child = node, node = node->parent) {
      if (node->type != nir_cf_node_if)
         continue;
      const nir_if *nif = nir_cf_node_as_if(node);
      bool in_then = false;
      foreach_list_typed(const nir_cf_node, n, node, &nif->then_list)
         in_then |= (n == child);
      uint32_t lane = 0;
      if (cond_exact_tid_eq(t, nir_get_scalar(nif->condition.ssa, 0), in_then,
                            6, &lane) &&
          lane == 0)
         return true;
   }
   return false;
}

static bool
is_plain_shared_barrier(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(intr) >= SCOPE_SUBGROUP &&
          nir_intrinsic_memory_scope(intr) >= SCOPE_SUBGROUP &&
          nir_intrinsic_memory_modes(intr) == nir_var_mem_shared &&
          nir_intrinsic_memory_semantics(intr) != 0;
}

static bool
is_tile_or_unknown_shared(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_pixel_agx:
   case nir_intrinsic_store_local_pixel_agx:
   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
   case nir_intrinsic_image_deref_store_block_agx:
      return true;
   default:
      return nir_is_shared_access(intr) &&
             intr->intrinsic != nir_intrinsic_load_shared &&
             intr->intrinsic != nir_intrinsic_store_shared;
   }
}

static bool
is_supported_reduce_op(nir_op op, bool *is_int_assoc)
{
   switch (op) {
   case nir_op_iadd:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
      *is_int_assoc = true;
      return true;
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_fmin:
   case nir_op_fmax:
      *is_int_assoc = false;
      return true;
   default:
      return false;
   }
}

static bool
is_single_block_empty_else(nir_if *nif, nir_block **then_out)
{
   if (!exec_list_is_singular(&nif->then_list) ||
       !exec_list_is_singular(&nif->else_list))
      return false;

   nir_cf_node *tn = exec_node_data(nir_cf_node, exec_list_get_head(&nif->then_list), node);
   nir_cf_node *en = exec_node_data(nir_cf_node, exec_list_get_head(&nif->else_list), node);
   if (tn->type != nir_cf_node_block || en->type != nir_cf_node_block)
      return false;

   nir_block *eb = nir_cf_node_as_block(en);
   if (!exec_list_is_empty(&eb->instr_list))
      return false;

   *then_out = nir_cf_node_as_block(tn);
   return true;
}

static bool
def_used_only_in_block(const nir_def *def, const nir_block *block)
{
   nir_foreach_use(src, def) {
      if (nir_src_use_instr(src)->block != block)
         return false;
   }
   nir_foreach_if_use(src, def) {
      return false;
   }
   return true;
}

static bool
ranges_overlap(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
{
   return a0 < b1 && b0 < a1;
}

struct red_array {
   uint32_t base_bytes;
   uint32_t elem_bytes;
   uint8_t num_components;
   nir_intrinsic_instr *prod_store;
   nir_def *init_val;
   nir_op op;
   bool is_int_assoc;
   bool op_set;
   bool lhs_is_self[4];
   unsigned fp_math_ctrl[4];
   bool needs_slot0_store;
   struct util_dynarray slot0_readers; /* nir_intrinsic_instr * */
};

struct red_match {
   nir_block *prod_block;     /* block holding prod_stores */
   nir_if *prod_if;           /* NULL if unconditional (wg == 2*S), else if (tid < 2*S) */
   unsigned start_stride;     /* S: 1, 2, 4, 8, or 16 */
   unsigned num_steps;        /* log2(S) + 1 */
   nir_intrinsic_instr *step_barriers[MAX_REDUCE_STEPS];
   nir_if *step_ifs[MAX_REDUCE_STEPS];
   nir_intrinsic_instr *trailing_barrier; /* optional barrier after stride==1 */
   unsigned num_arrays;
   struct red_array arrays[MAX_REDUCE_ARRAYS];
};

static bool
addr_matches_slot(const struct shared_addr *ad, const struct red_array *a,
                  unsigned stride_off, unsigned step_stride)
{
   return ad->base_bytes == a->base_bytes + stride_off * a->elem_bytes &&
          ad->access_bytes == a->elem_bytes &&
          ad->num_components == a->num_components &&
          (ad->tid_stride_bytes == a->elem_bytes ||
           (step_stride == 1 && ad->tid_stride_bytes == 0));
}

/* On stride == 1 (tid == 0), nir_opt_load_store_vectorize can merge adjacent
 * constant-offset loads of slot 0 and slot 1 into one 2*N-component load at
 * slot 0 when 2*N <= 4 (channels 0..N-1 are slot 0, channels N..2*N-1 are
 * slot 1). */
static bool
addr_matches_stride1_pair(const struct shared_addr *ad,
                          const struct red_array *a, unsigned step_stride)
{
   return step_stride == 1 && ad->tid_stride_bytes == 0 &&
          ad->base_bytes == a->base_bytes &&
          ad->num_components == 2 * a->num_components &&
          ad->access_bytes == 2 * a->elem_bytes;
}

static bool
match_step_block(const nir_shader *shader, const struct tid_info *t,
                 nir_block *blk, unsigned stride, struct red_match *m)
{
   /* Track which arrays this step updates. */
   bool seen_array[MAX_REDUCE_ARRAYS] = {false};
   nir_intrinsic_instr *lo_loads[MAX_REDUCE_ARRAYS] = {NULL};
   nir_intrinsic_instr *hi_loads[MAX_REDUCE_ARRAYS] = {NULL};
   unsigned total_loads = 0;
   unsigned expected_loads = 0;

   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_alu ||
          instr->type == nir_instr_type_load_const ||
          instr->type == nir_instr_type_undef)
         continue;
      if (instr->type != nir_instr_type_intrinsic)
         return false;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_shared) {
         total_loads++;
         if (!def_used_only_in_block(&intr->def, blk))
            return false;
         continue;
      }
      if (intr->intrinsic != nir_intrinsic_store_shared)
         return false;

      struct shared_addr saddr;
      if (!parse_shared_access(shader, t, intr, &saddr))
         return false;

      /* Match against one of the producer arrays. */
      int idx = -1;
      for (unsigned k = 0; k < m->num_arrays; k++) {
         const struct red_array *a = &m->arrays[k];
         if (saddr.num_components != a->num_components ||
             saddr.access_bytes != a->elem_bytes ||
             saddr.base_bytes != a->base_bytes)
            continue;
         bool ok_stride = (saddr.tid_stride_bytes == a->elem_bytes) ||
                          (stride == 1 && saddr.tid_stride_bytes == 0);
         if (ok_stride) {
            idx = (int)k;
            break;
         }
      }
      if (idx < 0 || seen_array[idx])
         return false;

      struct red_array *a = &m->arrays[idx];
      nir_def *stored = intr->src[0].ssa;

      /* Check every component of the stored value is op(lo.c, hi.c). */
      nir_intrinsic_instr *step_lo = NULL;
      nir_intrinsic_instr *step_hi = NULL;
      bool used_pair_load = false;
      for (unsigned c = 0; c < a->num_components; c++) {
         nir_scalar sc = nir_scalar_resolved(stored, c);
         if (!nir_scalar_is_alu(sc) || nir_def_instr(sc.def)->block != blk)
            return false;
         nir_alu_instr *alu = nir_def_as_alu(sc.def);
         bool is_int_assoc = false;
         if (!is_supported_reduce_op(alu->op, &is_int_assoc))
            return false;
         if (!def_used_only_in_block(&alu->def, blk))
            return false;

         nir_scalar s0 = nir_scalar_chase_alu_src(sc, 0);
         nir_scalar s1 = nir_scalar_chase_alu_src(sc, 1);
         if (!nir_scalar_is_intrinsic(s0) || !nir_scalar_is_intrinsic(s1))
            return false;

         nir_intrinsic_instr *l0 = nir_def_as_intrinsic(s0.def);
         nir_intrinsic_instr *l1 = nir_def_as_intrinsic(s1.def);
         if (l0->intrinsic != nir_intrinsic_load_shared ||
             l1->intrinsic != nir_intrinsic_load_shared ||
             l0->instr.block != blk || l1->instr.block != blk)
            return false;

         struct shared_addr a0, a1;
         if (!parse_shared_access(shader, t, l0, &a0) ||
             !parse_shared_access(shader, t, l1, &a1))
            return false;

         bool lhs_self;
         bool pair_c = false;
         nir_intrinsic_instr *lo_c, *hi_c;
         if (l0 == l1 && addr_matches_stride1_pair(&a0, a, stride)) {
            pair_c = true;
            if (s0.comp == c && s1.comp == c + a->num_components) {
               lhs_self = true;
            } else if (s1.comp == c && s0.comp == c + a->num_components) {
               lhs_self = false;
            } else {
               return false;
            }
            lo_c = l0;
            hi_c = l0;
         } else if (s0.comp == c && s1.comp == c &&
                    addr_matches_slot(&a0, a, 0, stride) &&
                    addr_matches_slot(&a1, a, stride, stride)) {
            lhs_self = true;
            lo_c = l0;
            hi_c = l1;
         } else if (s0.comp == c && s1.comp == c &&
                    addr_matches_slot(&a1, a, 0, stride) &&
                    addr_matches_slot(&a0, a, stride, stride)) {
            lhs_self = false;
            lo_c = l1;
            hi_c = l0;
         } else {
            return false;
         }

         if (c == 0) {
            step_lo = lo_c;
            step_hi = hi_c;
            used_pair_load = pair_c;
         } else if (lo_c != step_lo || hi_c != step_hi ||
                    pair_c != used_pair_load) {
            return false;
         }

         if (!a->op_set) {
            if (c == 0) {
               a->op = alu->op;
               a->is_int_assoc = is_int_assoc;
            } else if (alu->op != a->op) {
               return false;
            }
            a->lhs_is_self[c] = lhs_self;
            a->fp_math_ctrl[c] = alu->fp_math_ctrl;
         } else {
            if (alu->op != a->op || a->lhs_is_self[c] != lhs_self ||
                a->fp_math_ctrl[c] != alu->fp_math_ctrl)
               return false;
         }
      }

      /* Loads must precede this store and not be reused by another store. */
      if (!step_lo || !step_hi || (!used_pair_load && step_lo == step_hi))
         return false;
      a->op_set = true;
      seen_array[idx] = true;
      lo_loads[idx] = step_lo;
      hi_loads[idx] = step_hi;
      expected_loads += used_pair_load ? 1 : 2;
   }

   unsigned matched_count = 0;
   for (unsigned k = 0; k < m->num_arrays; k++) {
      if (seen_array[k])
         matched_count++;
   }
   /* Every array must be updated, and every load in the block must be accounted for. */
   if (matched_count != m->num_arrays || total_loads != expected_loads)
      return false;

   /* Verify no two arrays share a load instruction. */
   for (unsigned i = 0; i < m->num_arrays; i++) {
      for (unsigned j = i + 1; j < m->num_arrays; j++) {
         if (lo_loads[i] == lo_loads[j] || lo_loads[i] == hi_loads[j] ||
             hi_loads[i] == lo_loads[j] || hi_loads[i] == hi_loads[j])
            return false;
      }
   }

   return true;
}

static bool
block_has_single_shared_barrier(nir_block *blk, nir_intrinsic_instr **bar_out)
{
   nir_intrinsic_instr *bar = NULL;
   nir_foreach_instr(instr, blk) {
      if (instr->type == nir_instr_type_alu ||
          instr->type == nir_instr_type_load_const ||
          instr->type == nir_instr_type_undef)
         continue;
      if (instr->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (is_plain_shared_barrier(intr)) {
         if (bar)
            return false;
         bar = intr;
         continue;
      }
      if (nir_is_shared_access(intr) || is_tile_or_unknown_shared(intr))
         return false;
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      if (!(info->flags & NIR_INTRINSIC_CAN_ELIMINATE))
         return false;
   }
   if (!bar)
      return false;
   *bar_out = bar;
   return true;
}

static bool
block_in_loop(const nir_block *blk)
{
   for (const nir_cf_node *n = &blk->cf_node; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         return true;
   }
   return false;
}

static bool
instr_in_cf_subtree(const nir_block *blk, const nir_cf_node *root)
{
   for (const nir_cf_node *n = &blk->cf_node; n; n = n->parent) {
      if (n == root)
         return true;
   }
   return false;
}

static bool
is_step_instr(const struct red_match *m, const nir_instr *instr)
{
   for (unsigned s = 0; s < m->num_steps; s++) {
      if (instr_in_cf_subtree(instr->block, &m->step_ifs[s]->cf_node))
         return true;
   }
   return false;
}

static bool
is_prod_store(const struct red_match *m, const nir_intrinsic_instr *intr)
{
   for (unsigned k = 0; k < m->num_arrays; k++) {
      if (m->arrays[k].prod_store == intr)
         return true;
   }
   return false;
}

static void
classify_non_shared(nir_intrinsic_instr *intr, bool *writes, bool *reads)
{
   const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
   if (!(info->flags & NIR_INTRINSIC_CAN_ELIMINATE))
      *writes = true;
   else if (info->has_dest && !nir_intrinsic_can_reorder(intr))
      *reads = true;
}

static bool
audit_and_lower_reduction(nir_shader *shader, nir_function_impl *impl,
                          const struct tid_info *t, struct red_match *m)
{
   nir_metadata_require(impl, nir_metadata_dominance);
   nir_index_blocks(impl);
   nir_index_instrs(impl);

   const unsigned prod_blk_idx = m->prod_block->index;
   unsigned min_prod_store_idx = UINT_MAX;
   unsigned max_prod_store_idx = 0;
   for (unsigned k = 0; k < m->num_arrays; k++) {
      min_prod_store_idx =
         MIN2(min_prod_store_idx, m->arrays[k].prod_store->instr.index);
      max_prod_store_idx =
         MAX2(max_prod_store_idx, m->arrays[k].prod_store->instr.index);
   }

   nir_block *export_block = m->prod_if
      ? nir_cf_node_as_block(nir_cf_node_next(&m->prod_if->cf_node))
      : m->prod_block;

   nir_block *last_step_then = NULL;
   is_single_block_empty_else(m->step_ifs[m->num_steps - 1], &last_step_then);
   const unsigned last_step_blk_idx = last_step_then->index;

   /* Find trailing barrier if present right after the last step if. */
   nir_cf_node *after_last = nir_cf_node_next(&m->step_ifs[m->num_steps - 1]->cf_node);
   nir_block *after_last_blk = (after_last && after_last->type == nir_cf_node_block)
                                  ? nir_cf_node_as_block(after_last)
                                  : NULL;
   m->trailing_barrier = NULL;
   if (after_last_blk) {
      nir_foreach_instr(instr, after_last_blk) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (is_plain_shared_barrier(intr)) {
               m->trailing_barrier = intr;
               break;
            }
            if (nir_is_shared_access(intr) || is_tile_or_unknown_shared(intr))
               break;
         }
      }
   }

   /* Track non-shared memory ordering between the last kept WG barrier before
    * the reduction and the region after the reduction (needed when wg > 32 and
    * we remove workgroup-scope barriers), as well as the strongest scope and
    * semantics among the step barriers so disjoint shared traffic stays ordered. */
   bool pre_writes = false, pre_reads = false;
   bool post_writes = false, post_reads = false;
   mesa_scope max_step_exec_scope = SCOPE_NONE;
   mesa_scope max_step_mem_scope = SCOPE_NONE;
   unsigned combined_step_semantics = 0;
   for (unsigned s = 0; s < m->num_steps; s++) {
      nir_intrinsic_instr *sb = m->step_barriers[s];
      mesa_scope es = nir_intrinsic_execution_scope(sb);
      mesa_scope ms = nir_intrinsic_memory_scope(sb);
      if (es > max_step_exec_scope)
         max_step_exec_scope = es;
      if (ms > max_step_mem_scope)
         max_step_mem_scope = ms;
      combined_step_semantics |= nir_intrinsic_memory_semantics(sb);
   }
   bool any_deleted_wg_barrier = (max_step_exec_scope == SCOPE_WORKGROUP);
   bool other_shared_before = false;
   bool other_shared_after = false;
   bool written_after_red[MAX_REDUCE_ARRAYS] = {false};

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (is_tile_or_unknown_shared(intr))
            return false;

         if (is_step_instr(m, instr) || is_prod_store(m, intr))
            continue;

         bool is_deleted_bar = (intr == m->trailing_barrier);
         for (unsigned s = 0; s < m->num_steps; s++)
            is_deleted_bar |= (intr == m->step_barriers[s]);
         if (is_deleted_bar)
            continue;

         bool strictly_before_prod =
            (block->index < prod_blk_idx) ||
            (block == m->prod_block && instr->index < max_prod_store_idx);
         bool strictly_after_red = (block->index > last_step_blk_idx);

         if (intr->intrinsic == nir_intrinsic_barrier) {
            bool strictly_before_all_prods =
               (block->index < prod_blk_idx) ||
               (block == m->prod_block && instr->index < min_prod_store_idx);
            if (strictly_before_all_prods &&
                nir_block_dominates(block, m->prod_block) &&
                !block_in_loop(block) &&
                nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
               /* A kept dominating workgroup barrier resets pre-reduction ordering. */
               pre_writes = false;
               pre_reads = false;
               if (nir_intrinsic_memory_scope(intr) >= SCOPE_WORKGROUP &&
                   (nir_intrinsic_memory_modes(intr) & nir_var_mem_shared) &&
                   nir_intrinsic_memory_semantics(intr) != 0)
                  other_shared_before = false;
            }
            continue;
         }

         if (!nir_is_shared_access(intr)) {
            if (strictly_before_prod)
               classify_non_shared(intr, &pre_writes, &pre_reads);
            else if (strictly_after_red)
               classify_non_shared(intr, &post_writes, &post_reads);
            else
               return false; /* unexpected side-effecting op inside reduction range */
            continue;
         }

         /* Shared load or store outside the matched reduction. */
         struct shared_addr addr;
         bool parsed = parse_shared_access(shader, t, intr, &addr);
         uint32_t lo_b = 0, hi_b = UINT32_MAX;
         if (parsed) {
            lo_b = addr.base_bytes;
            uint64_t span = (uint64_t)(t->invocations - 1) * addr.tid_stride_bytes +
                            addr.access_bytes;
            hi_b = (lo_b + span > UINT32_MAX) ? UINT32_MAX : (uint32_t)(lo_b + span);
         }

         /* In prod_block, an access is before array k's producer store iff its
          * index is strictly less than that array's prod_store index. */
         if (block->index < prod_blk_idx || block == m->prod_block) {
            bool contained_in_any = false;
            for (unsigned k = 0; k < m->num_arrays; k++) {
               uint32_t a0 = m->arrays[k].base_bytes;
               uint32_t a1 = a0 + 2 * m->start_stride * m->arrays[k].elem_bytes;
               if (!ranges_overlap(lo_b, hi_b, a0, a1))
                  continue;
               if (parsed && lo_b >= a0 && hi_b <= a1)
                  contained_in_any = true;
               if (block == m->prod_block &&
                   instr->index > m->arrays[k].prod_store->instr.index)
                  return false;
            }
            if (!contained_in_any)
               other_shared_before = true;
            continue;
         }

         /* Not strictly before prod: must be strictly after reduction, OR
          * strictly disjoint from every reduced array! */
         for (unsigned k = 0; k < m->num_arrays; k++) {
            struct red_array *a = &m->arrays[k];
            uint32_t a0 = a->base_bytes;
            uint32_t a1 = a0 + 2 * m->start_stride * a->elem_bytes;
            if (!ranges_overlap(lo_b, hi_b, a0, a1))
               continue;

            if (!strictly_after_red)
               return false;

            if (intr->intrinsic == nir_intrinsic_store_shared) {
               /* A later write to A_k is only allowed if it doesn't precede a
                * reader we replace, and only after a barrier or overwrite. */
               written_after_red[k] = true;
               continue;
            }

            /* Shared load overlapping A_k after reduction: must be exact slot 0
             * (and not after any post-reduction write to A_k). */
            if (!parsed || addr.tid_stride_bytes != 0 ||
                addr.base_bytes != a->base_bytes ||
                addr.access_bytes != a->elem_bytes ||
                addr.num_components != a->num_components ||
                written_after_red[k])
               return false;

            /* Only rewrite a tid==0 reader to exported_vals[k] when the
             * reduction's export block dominates the reader (not e.g. in a
             * sibling else-branch or after an enclosing uniform-if merge). */
            if (is_guarded_tid_eq0(t, instr) &&
                nir_block_dominates(export_block, block)) {
               util_dynarray_append(&a->slot0_readers, intr);
            } else {
               a->needs_slot0_store = true;
            }
         }

         if (strictly_after_red) {
            bool contained_in_any = false;
            for (unsigned k = 0; k < m->num_arrays; k++) {
               uint32_t a0 = m->arrays[k].base_bytes;
               uint32_t a1 = a0 + 2 * m->start_stride * m->arrays[k].elem_bytes;
               if (parsed && lo_b >= a0 && hi_b <= a1)
                  contained_in_any = true;
            }
            if (!contained_in_any)
               other_shared_after = true;
         }
      }
   }

   for (unsigned k = 0; k < m->num_arrays; k++) {
      if (!written_after_red[k])
         continue;
      util_dynarray_foreach(&m->arrays[k].slot0_readers, nir_intrinsic_instr *, rptr) {
         if (block_in_loop((*rptr)->instr.block))
            return false;
      }
   }

   /* Non-shared ordering check for multi-subgroup workgroups when we would
    * remove a workgroup-scope barrier without another kept WG barrier between
    * pre and post non-shared memory accesses. */
   const bool non_shared_ordered =
      (pre_writes && (post_writes || post_reads)) ||
      (post_writes && pre_reads);
   if (t->invocations > 32 && any_deleted_wg_barrier && non_shared_ordered)
      return false;

   /* If disjoint shared accesses occur both before and after the reduction,
    * deleting all step barriers would drop the synchronization between them.
    * Keep either a strong-enough trailing barrier or one step barrier covering
    * the strongest scope/semantics among the deleted step barriers. */
   nir_intrinsic_instr *strongest_step_bar = NULL;
   for (unsigned s = 0; s < m->num_steps; s++) {
      nir_intrinsic_instr *sb = m->step_barriers[s];
      if (nir_intrinsic_execution_scope(sb) >= max_step_exec_scope &&
          nir_intrinsic_memory_scope(sb) >= max_step_mem_scope &&
          (nir_intrinsic_memory_semantics(sb) & combined_step_semantics) ==
             combined_step_semantics) {
         strongest_step_bar = sb;
         break;
      }
   }

   const bool trailing_covers_steps =
      m->trailing_barrier &&
      nir_intrinsic_execution_scope(m->trailing_barrier) >= max_step_exec_scope &&
      nir_intrinsic_memory_scope(m->trailing_barrier) >= max_step_mem_scope &&
      (nir_intrinsic_memory_semantics(m->trailing_barrier) &
       combined_step_semantics) == combined_step_semantics;

   const bool need_other_shared_barrier =
      other_shared_before && other_shared_after;
   nir_intrinsic_instr *kept_step_bar = NULL;
   if (need_other_shared_barrier && !trailing_covers_steps) {
      if (!strongest_step_bar)
         return false;
      kept_step_bar = strongest_step_bar;
   }

   bool any_slot0_store = false;
   for (unsigned k = 0; k < m->num_arrays; k++)
      any_slot0_store |= m->arrays[k].needs_slot0_store;

   /* If a non-tid==0 lane reads slot 0, we need a trailing barrier after
    * storing slot 0 (unless one already exists). */
   if (any_slot0_store && !m->trailing_barrier)
      return false;

   /* Emit the subgroup reductions at the producer block (right before the first
    * producer store, or right after the last init_val). */
   nir_builder b = nir_builder_at(nir_after_block(m->prod_block));
   nir_def *reduced_vals[MAX_REDUCE_ARRAYS] = {NULL};
   nir_def *reduced_comps[MAX_REDUCE_ARRAYS][4] = {{NULL}};

   const bool full_subgroup_active =
      (m->start_stride == 16) || (t->invocations == 2 * m->start_stride);

   for (unsigned k = 0; k < m->num_arrays; k++) {
      struct red_array *a = &m->arrays[k];
      b.cursor = nir_before_instr(&a->prod_store->instr);
      for (unsigned c = 0; c < a->num_components; c++) {
         nir_def *val = nir_channel(&b, a->init_val, c);
         nir_def *acc = val;
         if (a->is_int_assoc && full_subgroup_active) {
            acc = nir_reduce(&b, val, .reduction_op = a->op, .cluster_size = 0);
         } else {
            for (unsigned stride = m->start_stride; stride >= 1; stride >>= 1) {
               nir_def *other = nir_shuffle_xor(&b, acc, nir_imm_int(&b, stride));
               b.fp_math_ctrl = a->fp_math_ctrl[c];
               acc = nir_build_alu2(&b, a->op,
                                    a->lhs_is_self[c] ? acc : other,
                                    a->lhs_is_self[c] ? other : acc);
               b.fp_math_ctrl = 0;
            }
         }
         reduced_comps[k][c] = acc;
      }
      if (a->num_components == 1)
         reduced_vals[k] = reduced_comps[k][0];
      else
         reduced_vals[k] = nir_vec(&b, reduced_comps[k], a->num_components);
   }

   /* If producer was inside if (tid < 2*S), thread scalar phis to the merge block. */
   nir_def *exported_vals[MAX_REDUCE_ARRAYS] = {NULL};
   if (m->prod_if) {
      nir_block *then_blk = m->prod_block;
      nir_block *else_blk = nir_if_first_else_block(m->prod_if);
      nir_block *merge_blk = nir_cf_node_as_block(nir_cf_node_next(&m->prod_if->cf_node));

      nir_builder b_else = nir_builder_at(nir_after_block(else_blk));
      nir_builder b_merge = nir_builder_at(nir_after_phis(merge_blk));

      for (unsigned k = 0; k < m->num_arrays; k++) {
         struct red_array *a = &m->arrays[k];
         nir_def *phi_defs[4] = {NULL};
         for (unsigned c = 0; c < a->num_components; c++) {
            nir_phi_instr *phi = nir_phi_instr_create(shader);
            nir_def_init(&phi->instr, &phi->def, 1, 32);
            nir_phi_instr_add_src(phi, then_blk, reduced_comps[k][c]);
            nir_phi_instr_add_src(phi, else_blk, nir_undef(&b_else, 1, 32));
            nir_instr_insert(nir_before_block(merge_blk), &phi->instr);
            phi_defs[c] = &phi->def;
         }
         if (a->num_components == 1)
            exported_vals[k] = phi_defs[0];
         else
            exported_vals[k] = nir_vec(&b_merge, phi_defs, a->num_components);
      }
   } else {
      for (unsigned k = 0; k < m->num_arrays; k++)
         exported_vals[k] = reduced_vals[k];
   }

   /* If any array needs slot 0 stored for non-tid==0 readers, emit
    * if (tid == 0) { store_shared(exported_vals[k], base_k); } right before
    * the trailing barrier. */
   if (any_slot0_store) {
      b.cursor = nir_before_instr(&m->trailing_barrier->instr);
      nir_def *tid = t->one_d
                        ? nir_channel(&b, nir_load_local_invocation_id(&b), 0)
                        : nir_load_local_invocation_index(&b);
      nir_push_if(&b, nir_ieq_imm(&b, tid, 0));
      for (unsigned k = 0; k < m->num_arrays; k++) {
         struct red_array *a = &m->arrays[k];
         if (!a->needs_slot0_store)
            continue;
         nir_def *idx = shader->info.shared_element_index_agx
                           ? nir_imm_intN_t(&b, a->base_bytes / 4, 16)
                           : nir_imm_int(&b, a->base_bytes);
         nir_store_shared(&b, exported_vals[k], idx,
                          .base = 0,
                          .write_mask = BITFIELD_MASK(a->num_components),
                          .align_mul = a->elem_bytes);
      }
      nir_pop_if(&b, NULL);
   }

   /* Rewrite tid==0 slot-0 readers directly to the SSA reduced values. */
   for (unsigned k = 0; k < m->num_arrays; k++) {
      struct red_array *a = &m->arrays[k];
      util_dynarray_foreach(&a->slot0_readers, nir_intrinsic_instr *, rptr) {
         nir_intrinsic_instr *reader = *rptr;
         nir_def_rewrite_uses(&reader->def, exported_vals[k]);
         nir_instr_remove(&reader->instr);
      }
   }

   /* Remove producer stores, step ifs, and intermediate barriers. */
   for (unsigned k = 0; k < m->num_arrays; k++)
      nir_instr_remove(&m->arrays[k].prod_store->instr);

   for (unsigned s = 0; s < m->num_steps; s++) {
      if (m->step_barriers[s] != kept_step_bar)
         nir_instr_remove(&m->step_barriers[s]->instr);
      nir_cf_node_remove(&m->step_ifs[s]->cf_node);
   }

   /* Remove trailing barrier if no slot-0 store was emitted and no other shared
    * or non-shared traffic needs it. */
   if (m->trailing_barrier && !any_slot0_store &&
       !(need_other_shared_barrier && !kept_step_bar) &&
       !(t->invocations > 32 &&
         nir_intrinsic_execution_scope(m->trailing_barrier) == SCOPE_WORKGROUP &&
         non_shared_ordered)) {
      nir_instr_remove(&m->trailing_barrier->instr);
   }

   return true;
}

static bool
try_match_in_cf_list(nir_shader *shader, nir_function_impl *impl,
                     const struct tid_info *t, struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         /* Only recurse into convergent ifs so all lanes of subgroup 0 enter. */
         if (!nir_src_is_divergent(&nif->condition)) {
            if (try_match_in_cf_list(shader, impl, t, &nif->then_list) ||
                try_match_in_cf_list(shader, impl, t, &nif->else_list))
               return true;
         }
      }

      /* Candidate producer: either a block (unconditional, wg == 2*S) or
       * an if (tid < 2*S) with empty else. */
      nir_block *prod_block = NULL;
      nir_if *prod_if = NULL;
      uint32_t prod_bound = t->invocations;

      if (node->type == nir_cf_node_block) {
         prod_block = nir_cf_node_as_block(node);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!is_single_block_empty_else(nif, &prod_block))
            continue;
         if (!cond_exact_tid_lt(t, nir_get_scalar(nif->condition.ssa, 0), true,
                                6, &prod_bound))
            continue;
         if (prod_bound < 2 || prod_bound > 32 ||
             prod_bound > t->invocations ||
             !util_is_power_of_two_nonzero(prod_bound))
            continue;
         prod_if = nif;
      } else {
         continue;
      }

      /* Collect producer store_shared candidates in prod_block. */
      struct red_match m = {
         .prod_block = prod_block,
         .prod_if = prod_if,
      };
      bool ok = true;
      nir_foreach_instr(instr, prod_block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_shared)
            continue;
         struct shared_addr saddr;
         if (!parse_shared_access(shader, t, intr, &saddr))
            continue;
         if (saddr.tid_stride_bytes != saddr.access_bytes)
            continue;
         if (m.num_arrays == MAX_REDUCE_ARRAYS) {
            ok = false;
            break;
         }
         /* Reject duplicate base in producer block. */
         bool dup = false;
         for (unsigned k = 0; k < m.num_arrays; k++) {
            if (m.arrays[k].base_bytes == saddr.base_bytes)
               dup = true;
         }
         if (dup)
            continue;

         struct red_array *a = &m.arrays[m.num_arrays++];
         *a = (struct red_array){
            .base_bytes = saddr.base_bytes,
            .elem_bytes = saddr.access_bytes,
            .num_components = saddr.num_components,
            .prod_store = intr,
            .init_val = intr->src[0].ssa,
         };
         util_dynarray_init(&a->slot0_readers, NULL);
      }

      if (!ok || m.num_arrays == 0) {
         for (unsigned k = 0; k < m.num_arrays; k++)
            util_dynarray_fini(&m.arrays[k].slot0_readers);
         continue;
      }

      /* First step immediately follows producer: barrier block + if (tid < S). */
      nir_cf_node *cur = nir_cf_node_next(node);
      nir_intrinsic_instr *first_bar = NULL;
      if (prod_if) {
         if (!cur || cur->type != nir_cf_node_block ||
             !block_has_single_shared_barrier(nir_cf_node_as_block(cur), &first_bar))
            goto cleanup_fail;
         cur = nir_cf_node_next(cur);
      } else {
         /* Unconditional producer: barrier is in prod_block AFTER all prod_stores. */
         bool seen_all = false;
         unsigned seen_stores = 0;
         nir_foreach_instr(instr, prod_block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (is_prod_store(&m, intr)) {
               seen_stores++;
               if (seen_stores == m.num_arrays)
                  seen_all = true;
               continue;
            }
            if (seen_all) {
               if (is_plain_shared_barrier(intr) && !first_bar) {
                  first_bar = intr;
               } else if (nir_is_shared_access(intr) ||
                          is_tile_or_unknown_shared(intr) ||
                          is_plain_shared_barrier(intr) ||
                          !(nir_intrinsic_infos[intr->intrinsic].flags &
                            NIR_INTRINSIC_CAN_ELIMINATE)) {
                  first_bar = NULL;
                  break;
               }
            }
         }
         if (!first_bar)
            goto cleanup_fail;
      }

      if (!cur || cur->type != nir_cf_node_if)
         goto cleanup_fail;
      nir_if *first_if = nir_cf_node_as_if(cur);
      nir_block *first_then = NULL;
      uint32_t start_stride = 0;
      if (!is_single_block_empty_else(first_if, &first_then) ||
          !cond_exact_tid_lt(t, nir_get_scalar(first_if->condition.ssa, 0), true,
                             6, &start_stride))
         goto cleanup_fail;

      if (start_stride < 1 || start_stride > 16 ||
          !util_is_power_of_two_nonzero(start_stride) ||
          2 * start_stride != prod_bound)
         goto cleanup_fail;

      /* Verify pairwise disjointness of [base_k, base_k + 2*S*elem_bytes). */
      for (unsigned i = 0; i < m.num_arrays; i++) {
         uint64_t end_i = (uint64_t)m.arrays[i].base_bytes +
                          2ull * start_stride * m.arrays[i].elem_bytes;
         if (end_i > shader->info.shared_size)
            goto cleanup_fail;
         for (unsigned j = i + 1; j < m.num_arrays; j++) {
            uint64_t end_j = (uint64_t)m.arrays[j].base_bytes +
                             2ull * start_stride * m.arrays[j].elem_bytes;
            if (ranges_overlap(m.arrays[i].base_bytes, (uint32_t)end_i,
                               m.arrays[j].base_bytes, (uint32_t)end_j))
               goto cleanup_fail;
         }
      }

      m.start_stride = start_stride;
      m.num_steps = 0;

      /* Match all halving steps: S, S/2, ..., 1. */
      unsigned stride = start_stride;
      nir_intrinsic_instr *step_bar = first_bar;
      nir_cf_node *step_node = cur;
      bool matched_all = true;
      while (true) {
         if (!step_node || step_node->type != nir_cf_node_if ||
             m.num_steps >= MAX_REDUCE_STEPS) {
            matched_all = false;
            break;
         }
         nir_if *sif = nir_cf_node_as_if(step_node);
         nir_block *sthen = NULL;
         uint32_t sbound = 0;
         if (!is_single_block_empty_else(sif, &sthen) ||
             !cond_exact_tid_lt(t, nir_get_scalar(sif->condition.ssa, 0), true,
                                6, &sbound) ||
             sbound != stride ||
             !match_step_block(shader, t, sthen, stride, &m)) {
            matched_all = false;
            break;
         }

         m.step_barriers[m.num_steps] = step_bar;
         m.step_ifs[m.num_steps] = sif;
         m.num_steps++;

         if (stride == 1)
            break;

         stride >>= 1;
         nir_cf_node *bar_node = nir_cf_node_next(step_node);
         if (!bar_node || bar_node->type != nir_cf_node_block ||
             !block_has_single_shared_barrier(nir_cf_node_as_block(bar_node),
                                              &step_bar)) {
            matched_all = false;
            break;
         }
         step_node = nir_cf_node_next(bar_node);
      }

      if (matched_all && audit_and_lower_reduction(shader, impl, t, &m)) {
         for (unsigned k = 0; k < m.num_arrays; k++)
            util_dynarray_fini(&m.arrays[k].slot0_readers);
         return true;
      }

   cleanup_fail:
      for (unsigned k = 0; k < m.num_arrays; k++)
         util_dynarray_fini(&m.arrays[k].slot0_readers);
   }

   return false;
}

/* Remove a leading plain shared-memory barrier that has zero shared-memory
 * accesses before it (from entrypoint) and no non-shared memory write-read /
 * write-write / read-write ordering conflict across it. */
static bool
remove_dead_leading_shared_barrier(nir_function_impl *impl, unsigned invocations)
{
   nir_intrinsic_instr *first_bar = NULL;
   bool shared_before = false;
   bool pre_writes = false, pre_reads = false;
   bool post_writes = false, post_reads = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!first_bar) {
            if (is_plain_shared_barrier(intr)) {
               /* Must not be inside a loop. */
               for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
                  if (n->type == nir_cf_node_loop)
                     return false;
               }
               first_bar = intr;
               continue;
            }
            if (nir_is_shared_access(intr) || is_tile_or_unknown_shared(intr))
               shared_before = true;
            else if (intr->intrinsic == nir_intrinsic_barrier)
               return false;
            else
               classify_non_shared(intr, &pre_writes, &pre_reads);
         } else {
            if (intr->intrinsic == nir_intrinsic_barrier)
               break;
            if (!nir_is_shared_access(intr) && !is_tile_or_unknown_shared(intr))
               classify_non_shared(intr, &post_writes, &post_reads);
         }
      }
   }

   if (!first_bar || shared_before)
      return false;

   const bool ordered = (pre_writes && (post_writes || post_reads)) ||
                        (post_writes && pre_reads);
   if (invocations > 32 && ordered)
      return false;

   nir_instr_remove(&first_bar->instr);
   return true;
}

static bool
shader_has_any_shared_access(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (nir_is_shared_access(intr) || is_tile_or_unknown_shared(intr))
            return true;
      }
   }
   return false;
}

bool
agx_nir_opt_shared_tree_reduce(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable ||
       !shader->info.shared_size)
      return false;

   const unsigned invocations = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];
   if (invocations < 2)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_divergence_analysis(shader);

   const struct tid_info t = {
      .one_d = shader->info.workgroup_size[1] == 1 &&
               shader->info.workgroup_size[2] == 1,
      .invocations = invocations,
   };

   bool progress = false;
   while (try_match_in_cf_list(shader, impl, &t, &impl->body)) {
      progress = true;
      nir_progress(true, impl, nir_metadata_none);
      nir_divergence_analysis(shader);
   }

   if (progress) {
      if (remove_dead_leading_shared_barrier(impl, invocations))
         nir_progress(true, impl, nir_metadata_none);
      if (!shader_has_any_shared_access(impl))
         shader->info.shared_size = 0;
   }

   nir_foreach_function_impl(other, shader) {
      if (other != impl)
         nir_no_progress(other);
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

/*
 * Sink a large pure ALU dependency cone feeding only bcsel(tid == c, val, else)
 * instructions into an explicit if (tid == c) block in multi-subgroup compute
 * shaders, so all other subgroups skip the computation via jmp_exec_none.
 */
static bool
is_pure_sinkable_instr(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_alu ||
       instr->type == nir_instr_type_load_const)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
      return op == nir_intrinsic_load_preamble;
   }
   return false;
}

static bool
is_nontrivial_alu(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_op op = nir_instr_as_alu(instr)->op;
   return op != nir_op_mov && op != nir_op_vec2 && op != nir_op_vec3 &&
          op != nir_op_vec4;
}

static bool
try_sink_selects_in_block(nir_block *block, const struct tid_info *t)
{
   if (block->divergent)
      return false;

   /* Do not split blocks inside loops (could add a branch inside a hot loop). */
   for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n->type == nir_cf_node_loop)
         return false;
   }

   /* Index instructions inside this block (0..N-1). */
   unsigned count = 0;
   nir_foreach_instr(instr, block)
      instr->index = count++;
   if (count < SINGLE_LANE_SINK_MIN_ALU + 2)
      return false;

   nir_instr **instrs = malloc(count * sizeof(*instrs));
   bool *in_cone = calloc(count, sizeof(*in_cone));
   bool *is_root = calloc(count, sizeof(*is_root));
   if (!instrs || !in_cone || !is_root) {
      free(instrs);
      free(in_cone);
      free(is_root);
      return false;
   }
   unsigned idx = 0;
   nir_foreach_instr(instr, block)
      instrs[idx++] = instr;

   bool progress = false;
   for (unsigned i = 0; i < count; i++) {
      if (instrs[i]->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *first = nir_instr_as_alu(instrs[i]);
      if (first->op != nir_op_bcsel || first->def.num_components != 1)
         continue;

      nir_def *cond = first->src[0].src.ssa;
      uint32_t lane = 0;
      if (cond->num_components != 1 ||
          !cond_exact_tid_eq(t, nir_get_scalar(cond, 0), true, 6, &lane))
         continue;

      /* Group contiguous bcsel instructions sharing the exact same cond whose
       * fallback (src[2]) is already defined before `first`. */
      memset(is_root, 0, count * sizeof(*is_root));
      unsigned last_root = i;
      unsigned num_roots = 0;
      for (unsigned j = i; j < count; j++) {
         if (instrs[j]->type != nir_instr_type_alu)
            break;
         nir_alu_instr *cand = nir_instr_as_alu(instrs[j]);
         if (cand->op != nir_op_bcsel || cand->def.num_components != 1 ||
             cand->src[0].src.ssa != cond)
            break;
         nir_instr *else_parent = nir_def_instr(cand->src[2].src.ssa);
         if (else_parent->block == block && else_parent->index >= i)
            break;
         nir_instr *then_parent = nir_def_instr(cand->src[1].src.ssa);
         if (then_parent->block == block && then_parent->index >= i)
            break;
         is_root[j] = true;
         last_root = j;
         num_roots++;
      }
      if (num_roots == 0)
         continue;

      /* Reverse-order closure in [0..i-1]: mark any pure instruction whose ALL
       * uses are inside the cone or are src[1] (then_val) of a root bcsel. */
      memset(in_cone, 0, count * sizeof(*in_cone));
      unsigned alu_count = 0;
      for (int k = (int)i - 1; k >= 0; k--) {
         nir_instr *cand = instrs[k];
         if (!is_pure_sinkable_instr(cand))
            continue;
         nir_def *def = nir_instr_def(cand);
         if (!def || list_is_empty(&def->uses))
            continue;

         bool has_if_use = false;
         nir_foreach_if_use(src, def) {
            has_if_use = true;
            break;
         }
         if (has_if_use)
            continue;

         bool all_private = true;
         nir_foreach_use(src, def) {
            nir_instr *user = nir_src_use_instr(src);
            if (user->block != block) {
               all_private = false;
               break;
            }
            unsigned uidx = user->index;
            if (in_cone[uidx])
               continue;
            if (is_root[uidx]) {
               nir_alu_instr *ralu = nir_instr_as_alu(user);
               /* Must feed ONLY then_val (src[1]), not cond or else_val. */
               if (src == &ralu->src[1].src &&
                   &ralu->src[0].src != src && &ralu->src[2].src != src)
                  continue;
            }
            all_private = false;
            break;
         }

         if (all_private) {
            in_cone[k] = true;
            if (is_nontrivial_alu(cand))
               alu_count++;
         }
      }

      if (alu_count < SINGLE_LANE_SINK_MIN_ALU) {
         i = last_root;
         continue;
      }

      /* Split block right before `first`, move cone into then_block, replace
       * each root bcsel with a scalar nir_if_phi. */
      nir_builder b = nir_builder_at(nir_before_instr(&first->instr));
      nir_if *nif = nir_push_if(&b, cond);
      nir_block *then_blk = nir_if_first_then_block(nif);

      for (unsigned k = 0; k < i; k++) {
         if (in_cone[k])
            nir_instr_move(nir_after_block(then_blk), instrs[k]);
      }

      nir_pop_if(&b, nif);

      for (unsigned j = i; j <= last_root; j++) {
         nir_alu_instr *ralu = nir_instr_as_alu(instrs[j]);
         nir_def *then_val;
         nir_def *else_val;
         if (ralu->src[1].src.ssa->num_components > 1) {
            nir_builder bt = nir_builder_at(nir_after_block(then_blk));
            then_val = nir_channel(&bt, ralu->src[1].src.ssa,
                                   ralu->src[1].swizzle[0]);
         } else {
            then_val = ralu->src[1].src.ssa;
         }
         if (ralu->src[2].src.ssa->num_components > 1) {
            nir_builder be = nir_builder_at(nir_after_block(nir_if_first_else_block(nif)));
            else_val = nir_channel(&be, ralu->src[2].src.ssa,
                                   ralu->src[2].swizzle[0]);
         } else {
            else_val = ralu->src[2].src.ssa;
         }

         nir_def *phi = nir_if_phi(&b, then_val, else_val);
         nir_def_rewrite_uses(&ralu->def, phi);
         nir_instr_remove(&ralu->instr);
      }

      progress = true;
      break; /* Block was split; caller will re-scan if needed. */
   }

   free(instrs);
   free(in_cone);
   free(is_root);
   return progress;
}

bool
agx_nir_opt_sink_single_lane_select(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable)
      return false;

   const unsigned invocations = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];
   /* Single-subgroup workgroups always execute lane 0 in subgroup 0, so
    * wrapping a bcsel in an if would not skip any subgroup. */
   if (invocations <= 32)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_divergence_analysis(shader);

   const struct tid_info t = {
      .one_d = shader->info.workgroup_size[1] == 1 &&
               shader->info.workgroup_size[2] == 1,
      .invocations = invocations,
   };

   bool progress = false;
   bool iter;
   do {
      iter = false;
      nir_foreach_block_safe(block, impl) {
         if (try_sink_selects_in_block(block, &t)) {
            iter = true;
            progress = true;
            nir_progress(true, impl, nir_metadata_none);
            nir_divergence_analysis(shader);
            break;
         }
      }
   } while (iter);

   nir_foreach_function_impl(other, shader) {
      if (other != impl)
         nir_no_progress(other);
   }
   return nir_progress(progress, impl, nir_metadata_none);
}
