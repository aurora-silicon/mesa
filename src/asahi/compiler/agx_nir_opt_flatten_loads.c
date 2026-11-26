/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "util/hash_table.h"
#include "util/set.h"

/*
 * Flatten small uniform ifs whose arms load from memory.
 *
 * nir_opt_peephole_select turns a small if into selects only when every
 * instruction of its arms may run unconditionally, which excludes loads from
 * device and threadgroup memory: an untaken arm's address need not be valid.
 * Such ifs are common in compute shaders. A parameter that arrives either as
 * a value or through an address is
 *
 *    if (by_value) x = value; else x = load_global(addr);
 *
 * and a choice between a value already in registers and a load is the same
 * shape. On AGX3 the if costs more than its arms: a compare, an exec-mask
 * push, a skip branch, an else flip and a pop, and every control-flow
 * boundary waits for all loads in flight, so a load issued before the if, or
 * in one arm, cannot overlap anything after it. A kernel that reads eight
 * parameters this way waits for eight loads one after another before it does
 * any work.
 *
 * Every load of an arm is made safe to run unconditionally by pointing it at
 * memory that is readable when its arm is not taken: a global load at the
 * read-only zero page every VM maps at AGX_ZERO_PAGE_ADDRESS, a threadgroup
 * load at an element index inside the shader's threadgroup allocation. Other
 * instructions must be safe to speculate as they are
 * (nir_instr_can_speculate), and loads marked speculatable keep their
 * address. Then the arms are moved before the if and its phis become
 * selects, as peephole select does:
 *
 *    l = load_global(by_value ? AGX_ZERO_PAGE_ADDRESS : addr);
 *    x = by_value ? value : l;
 *
 * A load that ran in its arm still runs with the same address and in the same
 * order relative to every other memory access (the arms hold no stores,
 * atomics or barriers, and the instructions only move above the if's
 * condition), so the result is unchanged; what the untaken arm's loads return
 * is discarded by the selects.
 *
 * Flattening handles two cases:
 *  - Uniform ifs whose arms are single blocks of at most `limit` counted
 *    instructions and `max_loads` memory reads, both arms together: the
 *    untaken arm's work is executed too, so a large arm costs more than the
 *    branch saves.
 *  - Bursts of at least 4 sibling single-load divergent ifs in the same CF
 *    list (each with one empty arm, one non-constant-buffer data load, and at
 *    most MIN2(limit, 24) counted ops), where flattening replaces >= 4 serial
 *    control-flow load-wait barriers with back-to-back overlapping loads.
 */

struct flatten_options {
   unsigned limit;
   unsigned max_loads;
};

/* The bytes a load reads from its address on. */
static unsigned
load_bytes(nir_intrinsic_instr *intr)
{
   return intr->def.num_components * DIV_ROUND_UP(intr->def.bit_size, 8);
}

/* The smallest offset not below `from` that the load's alignment allows. */
static uint64_t
aligned_from(nir_intrinsic_instr *intr, uint64_t from)
{
   const uint64_t mul = MAX2(nir_intrinsic_align_mul(intr), 1);
   const uint64_t offset = nir_intrinsic_align_offset(intr);
   return from + ((offset - from) & (mul - 1));
}

/* Whether a load may be redirected: a global load with a 64-bit scalar
 * address or a threadgroup load, neither volatile (it must not run more
 * often than written) nor atomic. */
static bool
is_redirectable_load(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      if (intr->src[0].ssa->bit_size != 64 ||
          intr->src[0].ssa->num_components != 1)
         return false;
      FALLTHROUGH;
   case nir_intrinsic_load_shared:
      return !(nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC));
   default:
      return false;
   }
}

/*
 * A global load's address is redirected in one of two ways. The cheap one
 * keeps the address's shape: the address is a base plus offsets, an offset a
 * 32-bit value widened by u2u64 or i2i64, a constant, or a sum or constant
 * multiple or shift of offsets (amul is the multiply of an array index by its
 * stride). The base is selected against the zero page and every widened value
 * against zero before it is widened, so the load keeps its base + index form:
 *
 *    base + u2u64(i) * 4   ->   (t ? base : Z) + u2u64(t ? i : 0) * 4
 *
 * The constants stay, so untaken the address is Z plus the residual of the
 * offsets with each widened value zero. The selects go below the widening, so
 * the 32-bit arithmetic that address lowering may treat as a valid access
 * index is exactly zero; the 64-bit arithmetic above it computes the residual
 * modulo 2^64, as evaluated here. This form is used only when that residual,
 * the load's whole footprint and its declared alignment fit the zero page.
 * Otherwise the whole address is selected against a zero page address of the
 * load's alignment.
 *
 * The walk only follows scalar terms, so every source is its def's only
 * component.
 */
static bool
is_scalar_offset(nir_def *def)
{
   if (def->num_components != 1 || def->bit_size != 64)
      return false;

   if (nir_def_is_const(def))
      return true;

   nir_alu_instr *alu = nir_def_as_alu_or_null(def);
   if (!alu)
      return false;

   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return true;
   case nir_op_ishl:
   case nir_op_imul:
   case nir_op_amul:
      return nir_src_is_const(alu->src[1].src) &&
             is_scalar_offset(alu->src[0].src.ssa);
   case nir_op_iadd:
      return is_scalar_offset(alu->src[0].src.ssa) &&
             is_scalar_offset(alu->src[1].src.ssa);
   default:
      return false;
   }
}

/* The offset `def`, or its residual unless `taken`. */
static nir_def *
offset_unless(nir_builder *b, nir_def *def, nir_def *taken)
{
   if (nir_def_is_const(def))
      return def;

   nir_alu_instr *alu = nir_def_as_alu(def);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return nir_build_alu1(b, alu->op,
                            nir_bcsel(b, taken, x, nir_imm_zero(b, 1, x->bit_size)));
   case nir_op_ishl:
   case nir_op_imul:
   case nir_op_amul:
      return nir_build_alu2(b, alu->op, offset_unless(b, x, taken),
                            nir_ssa_for_alu_src(b, alu, 1));
   default:
      return nir_iadd(b, offset_unless(b, x, taken),
                      offset_unless(b, nir_ssa_for_alu_src(b, alu, 1), taken));
   }
}

/* The value offset_unless() gives `def` when not taken, modulo 2^64. */
static uint64_t
offset_residual(nir_def *def)
{
   if (nir_def_is_const(def))
      return nir_scalar_as_uint(nir_get_scalar(def, 0));

   nir_alu_instr *alu = nir_def_as_alu(def);
   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return 0;
   case nir_op_ishl:
      return offset_residual(alu->src[0].src.ssa)
             << (nir_alu_src_as_uint(alu->src[1]) & 63);
   case nir_op_imul:
   case nir_op_amul:
      return offset_residual(alu->src[0].src.ssa) *
             nir_alu_src_as_uint(alu->src[1]);
   default:
      return offset_residual(alu->src[0].src.ssa) +
             offset_residual(alu->src[1].src.ssa);
   }
}

/* The source of the scalar address sum `alu` that is an offset, or -1. */
static int
offset_source(nir_alu_instr *alu)
{
   if (alu->op != nir_op_iadd || alu->def.num_components != 1 ||
       alu->src[0].src.ssa->num_components != 1 ||
       alu->src[1].src.ssa->num_components != 1)
      return -1;

   for (unsigned i = 0; i < 2; ++i) {
      if (is_scalar_offset(alu->src[i].src.ssa))
         return i;
   }

   return -1;
}

/* The residual of the global address `addr` split into base and offsets. */
static uint64_t
address_residual(nir_def *addr)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(addr);
   int i = alu ? offset_source(alu) : -1;
   if (i < 0)
      return 0;

   return address_residual(alu->src[1 - i].src.ssa) +
          offset_residual(alu->src[i].src.ssa);
}

/* The global address `addr`, or the zero page plus its residual unless
 * `taken`. */
static nir_def *
address_unless(nir_builder *b, nir_def *addr, nir_def *taken)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(addr);
   int i = alu ? offset_source(alu) : -1;
   if (i < 0)
      return nir_bcsel(b, taken, addr, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));

   return nir_iadd(b, address_unless(b, alu->src[1 - i].src.ssa, taken),
                   offset_unless(b, alu->src[i].src.ssa, taken));
}

/* Whether the zero page holds the untaken load at offset `offset`. */
static bool
fits_zero_page(nir_intrinsic_instr *intr, uint64_t offset)
{
   return offset <= AGX_ZERO_PAGE_SIZE - load_bytes(intr) &&
          offset == aligned_from(intr, offset);
}

/*
 * The index an untaken threadgroup load reads, if one is inside the shader's
 * threadgroup allocation. By the time this pass runs on AGX3 a load_shared's
 * source counts scalar elements of the load's bit size (lowered by
 * agx_nir_lower_shared_bitsize in the frontend or by the
 * agx_nir_lower_shared_element_index backstop; shared_element_index_agx), and
 * the load reads from byte BASE + index * scalar bytes, which ALIGN_MUL and
 * ALIGN_OFFSET still describe. With powers of two for both, the byte offsets
 * BASE + k * scalar bytes meet the declared alignment only when the gap from
 * BASE to it, modulo ALIGN_MUL, is a multiple of min(scalar bytes, ALIGN_MUL)
 * (so zero when ALIGN_MUL is smaller than a scalar), and the smallest is
 * k = gap / scalar bytes. A load
 * whose source still counts bytes is never redirected.
 */
static bool
shared_index(nir_shader *shader, nir_intrinsic_instr *intr, uint64_t *out)
{
   const int base = nir_intrinsic_base(intr);
   if (!shader->info.shared_element_index_agx || base < 0 ||
       intr->def.bit_size < 8)
      return false;

   const uint64_t scalar = intr->def.bit_size / 8;
   const uint64_t mul = MAX2(nir_intrinsic_align_mul(intr), 1);
   const uint64_t gap =
      ((uint64_t)nir_intrinsic_align_offset(intr) - base) & (mul - 1);
   if (gap % MIN2(scalar, mul))
      return false;

   const uint64_t k = gap / scalar;
   const unsigned index_bits = intr->src[0].ssa->bit_size;
   if (index_bits < 64 && (k >> index_bits))
      return false;

   *out = k;
   return base + k * scalar + load_bytes(intr) <= shader->info.shared_size;
}

/* A load that may run unconditionally only once redirected. One already
 * marked speculatable (a push constant read from the root table, say) is
 * safe as it is and keeps its address. */
static bool
load_needs_redirect(nir_shader *shader, nir_intrinsic_instr *intr)
{
   if (!is_redirectable_load(intr) ||
       (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE))
      return false;

   uint64_t index;
   if (intr->intrinsic == nir_intrinsic_load_shared)
      return shared_index(shader, intr, &index);
   else
      return fits_zero_page(intr, aligned_from(intr, 0));
}

/* Whether an intrinsic reads memory, for the arms' budget of loads. */
static bool
reads_memory(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_infos[intr->intrinsic].has_dest &&
          nir_intrinsic_has_access(intr);
}

struct arm_cost {
   unsigned count, loads, redirects, data_loads;
   bool has_const_global;
};

/* Whether every instruction of `block` may run with its if's condition false
 * once its loads are redirected. Adds its instructions, memory reads and
 * loads to redirect to `cost`. */
static bool
arm_is_flattenable(nir_shader *shader, nir_block *block, struct arm_cost *cost)
{
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
         break;

      case nir_instr_type_alu:
         cost->count++;
         break;

      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (nir_intrinsic_has_access(intr) &&
             (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))
            return false;

         if (is_redirectable_load(intr) &&
             intr->intrinsic != nir_intrinsic_load_global_constant)
            cost->data_loads++;

         if (load_needs_redirect(shader, intr)) {
            cost->redirects++;
            if (intr->intrinsic == nir_intrinsic_load_global_constant)
               cost->has_const_global = true;
         } else if (!(is_redirectable_load(intr) &&
                      (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE)) &&
                    !nir_instr_can_speculate(instr)) {
            /* Stores, atomics, barriers, subgroup operations (which depend
             * on the active lanes), discards, and memory reads not known to
             * be safe at an untaken arm's address. */
            return false;
         }

         if (reads_memory(intr)) {
            cost->count++;
            cost->loads++;
         }
         break;
      }

      default:
         return false;
      }
   }

   return true;
}

/* Point the loads of `block` at readable memory unless `taken`. */
static void
redirect_loads(nir_builder *b, nir_block *block, nir_def *taken)
{
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!load_needs_redirect(b->shader, intr))
         continue;

      b->cursor = nir_before_instr(instr);
      nir_def *addr = intr->src[0].ssa;
      if (intr->intrinsic == nir_intrinsic_load_shared) {
         uint64_t index;
         ASSERTED bool fits = shared_index(b->shader, intr, &index);
         assert(fits);
         addr = nir_bcsel(b, taken, addr,
                          nir_imm_intN_t(b, index, addr->bit_size));
      } else if (fits_zero_page(intr, address_residual(addr))) {
         addr = address_unless(b, addr, taken);
      } else {
         addr = nir_bcsel(b, taken, addr,
                          nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS +
                                              aligned_from(intr, 0)));
      }
      nir_src_rewrite(&intr->src[0], addr);
   }
}

static bool
def_outside_if_arms(nir_def *def, nir_block *then_block, nir_block *else_block)
{
   nir_block *blk = nir_def_instr(def)->block;
   return blk != then_block && blk != else_block;
}

static bool
alu_op_is_commutative_int(nir_op op)
{
   switch (op) {
   case nir_op_iadd:
   case nir_op_imul:
   case nir_op_amul:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
      return true;
   default:
      return false;
   }
}

static bool
same_pure_expr(nir_def *a, nir_def *b, nir_block *then_block,
               nir_block *else_block, unsigned depth, unsigned *budget)
{
   if (*budget == 0)
      return false;
   (*budget)--;

   if (a == b)
      return def_outside_if_arms(a, then_block, else_block);

   if (depth == 0 || a->bit_size != b->bit_size ||
       a->num_components != b->num_components)
      return false;

   nir_instr *ia = nir_def_instr(a);
   nir_instr *ib = nir_def_instr(b);
   if (ia->type != ib->type)
      return false;

   if (ia->type == nir_instr_type_load_const) {
      nir_load_const_instr *ca = nir_instr_as_load_const(ia);
      nir_load_const_instr *cb = nir_instr_as_load_const(ib);
      for (unsigned i = 0; i < a->num_components; ++i) {
         if (nir_const_value_as_uint(ca->value[i], a->bit_size) !=
             nir_const_value_as_uint(cb->value[i], b->bit_size))
            return false;
      }
      return true;
   }

   if (ia->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *aa = nir_instr_as_alu(ia);
   nir_alu_instr *ab = nir_instr_as_alu(ib);
   if (aa->op != ab->op || aa->fp_math_ctrl != ab->fp_math_ctrl ||
       aa->no_signed_wrap != ab->no_signed_wrap ||
       aa->no_unsigned_wrap != ab->no_unsigned_wrap)
      return false;

   const unsigned n = nir_op_infos[aa->op].num_inputs;
   bool direct = true;
   for (unsigned i = 0; i < n; ++i) {
      unsigned comps = nir_ssa_alu_instr_src_components(aa, i);
      if (memcmp(aa->src[i].swizzle, ab->src[i].swizzle, comps) != 0 ||
          !same_pure_expr(aa->src[i].src.ssa, ab->src[i].src.ssa, then_block,
                          else_block, depth - 1, budget)) {
         direct = false;
         break;
      }
   }
   if (direct)
      return true;

   if (n == 2 && alu_op_is_commutative_int(aa->op)) {
      unsigned c0 = nir_ssa_alu_instr_src_components(aa, 0);
      unsigned c1 = nir_ssa_alu_instr_src_components(aa, 1);
      if (c0 == c1 &&
          memcmp(aa->src[0].swizzle, ab->src[1].swizzle, c0) == 0 &&
          memcmp(aa->src[1].swizzle, ab->src[0].swizzle, c1) == 0 &&
          same_pure_expr(aa->src[0].src.ssa, ab->src[1].src.ssa, then_block,
                         else_block, depth - 1, budget) &&
          same_pure_expr(aa->src[1].src.ssa, ab->src[0].src.ssa, then_block,
                         else_block, depth - 1, budget))
         return true;
   }

   return false;
}

static nir_def *
clone_pure_expr(nir_builder *b, nir_def *def, nir_block *then_block,
                nir_block *else_block, struct hash_table *remap)
{
   if (def_outside_if_arms(def, then_block, else_block))
      return def;

   struct hash_entry *he = _mesa_hash_table_search(remap, def);
   if (he)
      return he->data;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
         clone_pure_expr(b, alu->src[i].src.ssa, then_block, else_block, remap);
   }

   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, remap);
   nir_builder_instr_insert(b, clone);
   nir_def *res = nir_instr_def(clone);
   _mesa_hash_table_insert(remap, def, res);
   return res;
}

static void
dce_pure_arm(nir_block *block)
{
   nir_foreach_instr_reverse_safe(instr, block) {
      if (instr->type != nir_instr_type_alu &&
          instr->type != nir_instr_type_load_const &&
          instr->type != nir_instr_type_undef)
         continue;
      nir_def *def = nir_instr_def(instr);
      if (def && list_is_empty(&def->uses))
         nir_instr_remove(instr);
   }
}

/* Sink matching terminal store_shared instructions out of a single-block
 * if/else diamond into one store_shared(phi(v_then, v_else), saddr) after the
 * if. This removes duplicate shared-address ALU and leaves the if with only
 * its load arm, allowing burst flattening or peephole-select to remove the CF.
 */
static bool
sink_common_store_shared(nir_builder *b, nir_block *block)
{
   if (nir_cf_node_is_first(&block->cf_node))
      return false;

   nir_cf_node *prev_node = nir_cf_node_prev(&block->cf_node);
   if (prev_node->type != nir_cf_node_if)
      return false;

   nir_block *prev_block = nir_cf_node_as_block(nir_cf_node_prev(prev_node));
   if (nir_block_ends_in_jump(prev_block))
      return false;

   nir_if *nif = nir_cf_node_as_if(prev_node);
   nir_block *then_block = nir_if_first_then_block(nif);
   nir_block *else_block = nir_if_first_else_block(nif);
   if (nir_if_last_then_block(nif) != then_block ||
       nir_if_last_else_block(nif) != else_block ||
       nir_block_ends_in_jump(then_block) ||
       nir_block_ends_in_jump(else_block))
      return false;

   nir_instr *then_last = nir_block_last_instr(then_block);
   nir_instr *else_last = nir_block_last_instr(else_block);
   if (!then_last || !else_last ||
       then_last->type != nir_instr_type_intrinsic ||
       else_last->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *ts = nir_instr_as_intrinsic(then_last);
   nir_intrinsic_instr *es = nir_instr_as_intrinsic(else_last);
   if (ts->intrinsic != nir_intrinsic_store_shared ||
       es->intrinsic != nir_intrinsic_store_shared)
      return false;

   if ((nir_intrinsic_access(ts) | nir_intrinsic_access(es)) &
       (ACCESS_VOLATILE | ACCESS_ATOMIC))
      return false;

   if (nir_intrinsic_base(ts) != nir_intrinsic_base(es) ||
       nir_intrinsic_write_mask(ts) != nir_intrinsic_write_mask(es) ||
       nir_intrinsic_align_mul(ts) != nir_intrinsic_align_mul(es) ||
       nir_intrinsic_align_offset(ts) != nir_intrinsic_align_offset(es) ||
       ts->src[0].ssa->bit_size != es->src[0].ssa->bit_size ||
       ts->src[0].ssa->num_components != es->src[0].ssa->num_components ||
       ts->src[1].ssa->bit_size != es->src[1].ssa->bit_size ||
       ts->src[1].ssa->num_components != es->src[1].ssa->num_components)
      return false;

   unsigned expr_budget = 64;
   if (!same_pure_expr(ts->src[1].ssa, es->src[1].ssa, then_block, else_block,
                       8, &expr_budget))
      return false;

   b->cursor = nir_before_cf_node(&nif->cf_node);
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_def *hoisted_addr =
      clone_pure_expr(b, ts->src[1].ssa, then_block, else_block, remap);
   _mesa_hash_table_destroy(remap, NULL);

   nir_phi_instr *phi = nir_phi_instr_create(b->shader);
   nir_def_init(&phi->instr, &phi->def, ts->src[0].ssa->num_components,
                ts->src[0].ssa->bit_size);
   nir_phi_instr_add_src(phi, then_block, ts->src[0].ssa);
   nir_phi_instr_add_src(phi, else_block, es->src[0].ssa);
   b->cursor = nir_before_block(block);
   nir_builder_instr_insert(b, &phi->instr);

   b->cursor = nir_after_phis(block);
   nir_store_shared(b, &phi->def, hoisted_addr,
                    .base = nir_intrinsic_base(ts),
                    .write_mask = nir_intrinsic_write_mask(ts),
                    .align_mul = nir_intrinsic_align_mul(ts),
                    .align_offset = nir_intrinsic_align_offset(ts),
                    .access = nir_intrinsic_access(ts) &
                              nir_intrinsic_access(es));

   nir_instr_remove(&ts->instr);
   nir_instr_remove(&es->instr);
   dce_pure_arm(then_block);
   dce_pure_arm(else_block);
   return true;
}

/* Divergent small single-load guarded if eligible as part of a sibling
 * tile-load burst (>= 4 in one CF list). Excludes load_global_constant so UBO
 * reads remain intact for agx3_nir_lower_uniform_dma. */
static bool
is_divergent_burst_if(nir_shader *shader, nir_if *nif,
                      const struct flatten_options *options)
{
   if (nif->control == nir_selection_control_dont_flatten ||
       !nir_src_is_divergent(&nif->condition))
      return false;

   nir_block *prev_block = nir_cf_node_as_block(nir_cf_node_prev(&nif->cf_node));
   if (nir_block_ends_in_jump(prev_block))
      return false;

   nir_block *then_block = nir_if_first_then_block(nif);
   nir_block *else_block = nir_if_first_else_block(nif);
   if (nir_if_last_then_block(nif) != then_block ||
       nir_if_last_else_block(nif) != else_block)
      return false;

   struct arm_cost tcost = {0}, ecost = {0};
   if (!arm_is_flattenable(shader, then_block, &tcost) ||
       !arm_is_flattenable(shader, else_block, &ecost))
      return false;

   unsigned total_loads = tcost.loads + ecost.loads;
   unsigned total_data_loads = tcost.data_loads + ecost.data_loads;
   unsigned total_count = tcost.count + ecost.count;
   bool has_const_global = tcost.has_const_global || ecost.has_const_global;

   /* One data load (plus at most one speculatable descriptor load). Cap the
    * arm at 24 counted ops (3/8 of the 64-op uniform limit): enough for a
    * 2D/3D index + byte offset + bounds/descriptor setup (~8-16 ops) while
    * bounding speculative untaken ALU across a multi-if burst. */
   return total_data_loads == 1 && total_loads <= 2 &&
          total_loads <= options->max_loads &&
          total_count <= MIN2(options->limit, 24u) && !has_const_global &&
          (tcost.count == 0 || ecost.count == 0);
}

static bool
block_breaks_burst_run(nir_block *block)
{
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
      case nir_instr_type_phi:
         break;
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_shared) {
            if (nir_intrinsic_access(intr) & ACCESS_VOLATILE)
               return true;
            break;
         }
         if (nir_intrinsic_can_reorder(intr))
            break;
         return true;
      }
      default:
         return true;
      }
   }
   return false;
}

static void
flush_burst_run(struct set *burst_ifs, nir_if **run, unsigned *run_len)
{
   /* Require a contiguous run of >= 4 guarded loads (separated only by pure
    * ALU or non-volatile store_shared, with no barriers or other CF nodes)
    * before speculatively flattening divergent ifs: isolated 1-3 loads rarely
    * amortize zero-page redirection on cold lanes, whereas >= 4 consecutive
    * tile loads remove >= 4 serial load-wait barriers so global latencies
    * overlap back-to-back and common tile-address ALU can hoist. */
   if (*run_len >= 4) {
      for (unsigned i = 0; i < *run_len; ++i)
         _mesa_set_add(burst_ifs, run[i]);
   }
   *run_len = 0;
}

static void
collect_burst_ifs(nir_shader *shader, struct exec_list *cf_list,
                  const struct flatten_options *options, struct set *burst_ifs)
{
   nir_if *run[64];
   unsigned run_len = 0;

   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_block) {
         if (block_breaks_burst_run(nir_cf_node_as_block(node)))
            flush_burst_run(burst_ifs, run, &run_len);
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (is_divergent_burst_if(shader, nif, options)) {
            if (run_len < ARRAY_SIZE(run)) {
               run[run_len++] = nif;
            } else {
               _mesa_set_add(burst_ifs, nif);
            }
         } else {
            flush_burst_run(burst_ifs, run, &run_len);
         }
         collect_burst_ifs(shader, &nif->then_list, options, burst_ifs);
         collect_burst_ifs(shader, &nif->else_list, options, burst_ifs);
      } else if (node->type == nir_cf_node_loop) {
         flush_burst_run(burst_ifs, run, &run_len);
         collect_burst_ifs(shader, &nir_cf_node_as_loop(node)->body, options,
                           burst_ifs);
      }
   }
   flush_burst_run(burst_ifs, run, &run_len);
}

/* Match val == base + delta via 1..4 chained constant-offset iadd ops. */
static bool
match_iv_step(nir_def *val, nir_def *base, int64_t *delta_out)
{
   int64_t total = 0;
   nir_def *cur = val;
   for (unsigned depth = 0; depth < 4 && cur != base; ++depth) {
      nir_instr *instr = nir_def_instr(cur);
      if (instr->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_iadd)
         return false;
      nir_scalar s0 =
         nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
      nir_scalar s1 =
         nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
      if (nir_scalar_is_const(s1)) {
         total += nir_scalar_as_int(s1);
         cur = alu->src[0].src.ssa;
      } else if (nir_scalar_is_const(s0)) {
         total += nir_scalar_as_int(s0);
         cur = alu->src[1].src.ssa;
      } else {
         return false;
      }
   }
   if (cur != base || total == 0)
      return false;
   *delta_out = total;
   return true;
}

static bool
cond_compares_iv(nir_def *cond, nir_def *iv, unsigned depth)
{
   nir_instr *ci = nir_def_instr(cond);
   if (ci->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(ci);
   switch (alu->op) {
   case nir_op_ilt:
   case nir_op_ult:
   case nir_op_ige:
   case nir_op_uge:
   case nir_op_ieq:
   case nir_op_ine:
      return alu->src[0].src.ssa == iv || alu->src[1].src.ssa == iv;
   case nir_op_ior:
   case nir_op_iand:
      return depth > 0 &&
             (cond_compares_iv(alu->src[0].src.ssa, iv, depth - 1) ||
              cond_compares_iv(alu->src[1].src.ssa, iv, depth - 1));
   case nir_op_inot:
      return depth > 0 && cond_compares_iv(alu->src[0].src.ssa, iv, depth - 1);
   default:
      return false;
   }
}

static bool
is_uniform_single_then_if(nir_if *nif)
{
   if (nir_src_is_divergent(&nif->condition))
      return false;
   nir_block *tb = nir_if_first_then_block(nif);
   nir_block *eb = nir_if_first_else_block(nif);
   if (nir_if_last_then_block(nif) != tb || nir_if_last_else_block(nif) != eb)
      return false;
   return exec_list_is_empty(&eb->instr_list);
}

static bool
match_tail_iv_step_if(nir_if *nif, nir_def *expected_iv,
                      nir_def **iv_out_out, int64_t *delta_out)
{
   if (!is_uniform_single_then_if(nif))
      return false;

   nir_block *tb = nir_if_first_then_block(nif);
   nir_block *eb = nir_if_first_else_block(nif);
   nir_block *merge = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
   nir_foreach_phi(phi, merge) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32)
         continue;
      nir_phi_src *ts = nir_phi_get_src_from_block(phi, tb);
      nir_phi_src *es = nir_phi_get_src_from_block(phi, eb);
      if (!ts || !es)
         continue;
      nir_def *vin = es->src.ssa;
      if (expected_iv && vin != expected_iv)
         continue;
      if (!cond_compares_iv(nif->condition.ssa, vin, 2))
         continue;
      int64_t delta = 0;
      if (!match_iv_step(ts->src.ssa, vin, &delta))
         continue;
      int64_t ad = delta < 0 ? -delta : delta;
      if (ad == 4 || ad == 2 || ad == 1) {
         *iv_out_out = &phi->def;
         *delta_out = delta;
         return true;
      }
   }
   return false;
}

/* Binary-tail cascades emitted by agx_nir_unroll_runtime_loops consist of
 * >= 2 sequential single-arm uniform ifs comparing a live induction variable
 * and stepping it by halving power-of-2 deltas (+/-4 -> +/-2 -> +/-1, with
 * the final 1-chunk allowed to drop the dead IV phi). Skip them locally inside
 * load flattening without marking nir_selection_control_dont_flatten, so
 * nir_opt_peephole_select stays free to fold pure-ALU / speculatable tails. */
static void
collect_binary_tail_ifs(struct exec_list *cf_list, struct set *skip_ifs)
{
   struct {
      nir_if *ifs[4];
      unsigned nifs;
      nir_def *iv_out;
      int64_t last_delta;
   } chains[8];
   unsigned nchains = 0;

   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_binary_tail_ifs(&nif->then_list, skip_ifs);
         collect_binary_tail_ifs(&nif->else_list, skip_ifs);

         bool extended = false;
         for (unsigned c = 0; c < nchains; ++c) {
            if (!chains[c].iv_out || chains[c].nifs >= ARRAY_SIZE(chains[c].ifs))
               continue;
            int64_t abs_prev = chains[c].last_delta < 0 ? -chains[c].last_delta
                                                        : chains[c].last_delta;
            if (abs_prev != 4 && abs_prev != 2)
               continue;
            nir_def *vout = NULL;
            int64_t delta = 0;
            if (match_tail_iv_step_if(nif, chains[c].iv_out, &vout, &delta) &&
                delta * 2 == chains[c].last_delta) {
               chains[c].ifs[chains[c].nifs++] = nif;
               chains[c].iv_out = vout;
               chains[c].last_delta = delta;
               extended = true;
               break;
            }
            if (abs_prev == 2 && is_uniform_single_then_if(nif) &&
                cond_compares_iv(nif->condition.ssa, chains[c].iv_out, 2)) {
               chains[c].ifs[chains[c].nifs++] = nif;
               chains[c].iv_out = NULL;
               chains[c].last_delta = chains[c].last_delta / 2;
               extended = true;
               break;
            }
         }
         if (!extended && nchains < ARRAY_SIZE(chains)) {
            nir_def *vout = NULL;
            int64_t delta = 0;
            if (match_tail_iv_step_if(nif, NULL, &vout, &delta)) {
               int64_t ad = delta < 0 ? -delta : delta;
               if (ad == 4 || ad == 2) {
                  chains[nchains].ifs[0] = nif;
                  chains[nchains].nifs = 1;
                  chains[nchains].iv_out = vout;
                  chains[nchains].last_delta = delta;
                  nchains++;
               }
            }
         }
      } else if (node->type == nir_cf_node_loop) {
         collect_binary_tail_ifs(&nir_cf_node_as_loop(node)->body, skip_ifs);
      }
   }

   for (unsigned c = 0; c < nchains; ++c) {
      if (chains[c].nifs < 2)
         continue;
      for (unsigned i = 0; i < chains[c].nifs; ++i)
         _mesa_set_add(skip_ifs, chains[c].ifs[i]);
   }
}

static bool
flatten_block(nir_builder *b, nir_block *block,
              const struct flatten_options *options,
              const struct set *burst_ifs, const struct set *skip_ifs)
{
   if (nir_cf_node_is_first(&block->cf_node))
      return false;

   nir_cf_node *prev_node = nir_cf_node_prev(&block->cf_node);
   if (prev_node->type != nir_cf_node_if)
      return false;

   nir_block *prev_block = nir_cf_node_as_block(nir_cf_node_prev(prev_node));
   if (nir_block_ends_in_jump(prev_block))
      return false;

   /* Flatten branches every lane takes the same way, plus bursts of >= 4
    * sibling small single-load guarded ifs (such as tile-loading loops) where
    * per-load control-flow barriers serialize memory latency. */
   nir_if *nif = nir_cf_node_as_if(prev_node);
   if (nif->control == nir_selection_control_dont_flatten ||
       _mesa_set_search(skip_ifs, nif) != NULL)
      return false;

   const bool in_burst = _mesa_set_search(burst_ifs, nif) != NULL;
   if (nir_src_is_divergent(&nif->condition) && !in_burst)
      return false;

   nir_block *then_block = nir_if_first_then_block(nif);
   nir_block *else_block = nir_if_first_else_block(nif);
   if (nir_if_last_then_block(nif) != then_block ||
       nir_if_last_else_block(nif) != else_block)
      return false;

   struct arm_cost cost = {0};
   if (!arm_is_flattenable(b->shader, then_block, &cost) ||
       !arm_is_flattenable(b->shader, else_block, &cost))
      return false;

   /* Uniform ifs without loads to redirect are peephole select's; burst ifs
    * may hold speculatable data loads that peephole select skips when
    * indirect_load_ok is false. */
   if ((!in_burst && cost.redirects == 0) || cost.loads > options->max_loads ||
       cost.count > options->limit)
      return false;

   nir_def *cond = nif->condition.ssa;
   b->cursor = nir_after_block(prev_block);
   nir_def *not_cond = nir_inot(b, cond);
   redirect_loads(b, then_block, cond);
   redirect_loads(b, else_block, not_cond);

   nir_foreach_instr_safe(instr, then_block) {
      exec_node_remove(&instr->node);
      instr->block = prev_block;
      exec_list_push_tail(&prev_block->instr_list, &instr->node);
   }

   nir_foreach_instr_safe(instr, else_block) {
      exec_node_remove(&instr->node);
      instr->block = prev_block;
      exec_list_push_tail(&prev_block->instr_list, &instr->node);
   }

   nir_foreach_phi_safe(phi, block) {
      nir_def *then_def = NULL, *else_def = NULL;
      nir_foreach_phi_src(src, phi) {
         if (src->pred == then_block)
            then_def = src->src.ssa;
         else
            else_def = src->src.ssa;
      }

      b->cursor = nir_after_block(prev_block);
      nir_def_replace(&phi->def, nir_bcsel(b, cond, then_def, else_def));
   }

   nir_cf_node_remove(&nif->cf_node);
   return true;
}

bool
agx_nir_opt_flatten_loads(nir_shader *shader, unsigned limit,
                          unsigned max_loads)
{
   const struct flatten_options options = {limit, max_loads};
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool sunk = false;
      nir_foreach_block_safe(block, impl)
         sunk |= sink_common_store_shared(&b, block);
      if (sunk)
         nir_progress(true, impl, nir_metadata_control_flow);

      nir_metadata_require(impl, nir_metadata_divergence);
      struct set *burst_ifs = _mesa_pointer_set_create(NULL);
      struct set *skip_ifs = _mesa_pointer_set_create(NULL);
      collect_burst_ifs(shader, &impl->body, &options, burst_ifs);
      collect_binary_tail_ifs(&impl->body, skip_ifs);

      bool flattened = false;
      nir_foreach_block_safe(block, impl)
         flattened |= flatten_block(&b, block, &options, burst_ifs, skip_ifs);

      _mesa_set_destroy(skip_ifs, NULL);
      _mesa_set_destroy(burst_ifs, NULL);
      progress |= nir_progress(sunk || flattened, impl, nir_metadata_none);
   }

   return progress;
}
