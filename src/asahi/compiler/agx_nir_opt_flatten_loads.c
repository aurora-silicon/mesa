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
 * Only ifs with a uniform condition whose arms are single blocks of at most
 * `limit` counted instructions and `max_loads` memory reads, both arms
 * together, are flattened: the untaken arm's work is executed too, and a
 * large arm costs more than the branch saves.
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
   unsigned count, loads, redirects;
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

         if (load_needs_redirect(shader, intr)) {
            cost->redirects++;
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
flatten_block(nir_builder *b, nir_block *block,
              const struct flatten_options *options)
{
   if (nir_cf_node_is_first(&block->cf_node))
      return false;

   nir_cf_node *prev_node = nir_cf_node_prev(&block->cf_node);
   if (prev_node->type != nir_cf_node_if)
      return false;

   nir_block *prev_block = nir_cf_node_as_block(nir_cf_node_prev(prev_node));
   if (nir_block_ends_in_jump(prev_block))
      return false;

   /* Only branches every lane takes the same way: a divergent if may run
    * neither arm or one arm for a whole subgroup, and flattening it would run
    * both every time. */
   nir_if *nif = nir_cf_node_as_if(prev_node);
   if (nif->control == nir_selection_control_dont_flatten ||
       nir_src_is_divergent(&nif->condition))
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

   /* Ifs without loads to redirect are peephole select's. */
   if (cost.redirects == 0 || cost.loads > options->max_loads ||
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
      nir_metadata_require(impl, nir_metadata_divergence);
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block_safe(block, impl)
         impl_progress |= flatten_block(&b, block, &options);

      progress |= nir_progress(impl_progress, impl, nir_metadata_none);
   }

   return progress;
}
