/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

/*
 * Store-to-load forwarding within a block:
 *
 *    store_deref (d, %v)
 *    ...
 *    %l = load_deref (d)
 *
 * When the load reads exactly the footprint a preceding same-block store
 * wrote, its result is the stored value: replace the load's uses with %v
 * and drop the load.  Within the invocation nothing between the store and
 * the load may have changed the location, and across invocations a
 * conflicting write visible without a barrier would be a data race in the
 * Vulkan memory model, which is undefined.  The same holds for the lowered
 * memory intrinsics (store_ssbo followed by load_ssbo at the same
 * index/offset, and the shared/global/task_payload equivalents).
 *
 * Only full coverage is forwarded: the derefs must compare equal and the
 * stored value must match the load's component count and bit size.  Partial
 * overlap is left alone.
 *
 * Everything between the store and the load must be free of side effects on
 * memory and of any barrier that could publish another thread's write to
 * the address.  Loads and other eliminatable intrinsics are harmless: they
 * do not write, and they cannot make a foreign write visible to this
 * invocation's program order in a race-free program.
 */

static bool
has_volatile_access(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & ACCESS_VOLATILE);
}

/* May the instruction change, or let another thread change, memory that a
 * following load in the same block could observe?  ALU, derefs, undefs and
 * phis are pure.  Eliminatable intrinsics are loads and queries, which
 * cannot publish a write.  Anything else -- stores, atomics, barriers,
 * fences, discard -- is conservatively treated as a hazard.
 */
static bool
is_memory_hazard(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_deref:
   case nir_instr_type_undef:
   case nir_instr_type_tex:
   case nir_instr_type_load_const:
      return false;
   case nir_instr_type_intrinsic:
      return !(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic]
               .flags & NIR_INTRINSIC_CAN_ELIMINATE);
   default:
      return true;
   }
}

static bool
same_scalar(nir_scalar a, nir_scalar b)
{
   if (a.def == b.def && a.comp == b.comp)
      return true;

   if (nir_scalar_is_const(a) && nir_scalar_is_const(b) &&
       a.def->bit_size == b.def->bit_size &&
       nir_scalar_as_uint(a) == nir_scalar_as_uint(b))
      return true;

   return false;
}

/* Does the load read exactly the footprint the store wrote: provably the
 * same address in the same memory space, and a value of the same component
 * count and bit size?  Anything less -- partial or unproven coverage -- is
 * not forwarded.
 */
static bool
store_covers_load(nir_intrinsic_instr *store, nir_intrinsic_instr *load)
{
   /* store_deref is src[] = { deref, value }; the lowered memory stores are
    * src[] = { value, ... }.
    */
   const bool is_deref = store->intrinsic == nir_intrinsic_store_deref;
   nir_def *value = store->src[is_deref ? 1 : 0].ssa;

   if (value->num_components != load->def.num_components ||
       value->bit_size != load->def.bit_size)
      return false;

   switch (store->intrinsic) {

   case nir_intrinsic_store_deref: {
      if (load->intrinsic != nir_intrinsic_load_deref)
         return false;

      nir_deref_instr *store_deref = nir_src_as_deref(store->src[0]);
      nir_deref_instr *load_deref = nir_src_as_deref(load->src[0]);
      if (!store_deref || !load_deref)
         return false;

      const nir_variable_mode modes =
         nir_var_mem_ssbo | nir_var_mem_shared | nir_var_mem_global |
         nir_var_mem_task_payload;
      if (!nir_deref_mode_may_be(store_deref, modes))
         return false;

      if (!(nir_compare_derefs(store_deref, load_deref) &
            nir_derefs_equal_bit))
         return false;

      return true;
   }

   case nir_intrinsic_store_ssbo: {
      /* src[] = { value, buffer_index, offset } */
      if (load->intrinsic != nir_intrinsic_load_ssbo)
         return false;
      if (nir_intrinsic_offset_shift(store) !=
          nir_intrinsic_offset_shift(load))
         return false;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)) ||
          !same_scalar(nir_scalar_resolved(store->src[2].ssa, 0),
                       nir_scalar_resolved(load->src[1].ssa, 0)))
         return false;
      return true;
   }

   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_task_payload: {
      nir_intrinsic_op match = store->intrinsic == nir_intrinsic_store_shared
                               ? nir_intrinsic_load_shared
                               : nir_intrinsic_load_task_payload;
      if (load->intrinsic != match)
         return false;
      if (nir_intrinsic_base(store) != nir_intrinsic_base(load))
         return false;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)))
         return false;
      return true;
   }

   case nir_intrinsic_store_global: {
      /* src[] = { value, address } */
      if (load->intrinsic != nir_intrinsic_load_global)
         return false;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)))
         return false;
      return true;
   }

   default:
      return false;
   }
}

static bool
is_supported_store(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_task_payload:
   case nir_intrinsic_store_global:
      return !has_volatile_access(intr);
   default:
      return false;
   }
}

/* Forward the stored value into every matching load that follows the store
 * in the same block, up to the first memory hazard.  One store may feed
 * several loads.
 */
static bool
forward_from_store(nir_intrinsic_instr *store)
{
   const bool is_deref = store->intrinsic == nir_intrinsic_store_deref;
   nir_def *value = store->src[is_deref ? 1 : 0].ssa;
   bool progress = false;

   for (nir_instr *instr = nir_instr_next(&store->instr),
                  *next = instr ? nir_instr_next(instr) : NULL;
        instr;
        instr = next, next = instr ? nir_instr_next(instr) : NULL) {
      if (is_memory_hazard(instr))
         break;

      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
      if (has_volatile_access(load) ||
          !store_covers_load(store, load))
         continue;

      nir_def_rewrite_uses(&load->def, value);
      nir_instr_remove(instr);
      progress = true;
   }

   return progress;
}

static bool
load_after_store_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (is_supported_store(store))
            progress |= forward_from_store(store);
      }
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_none);

   return progress;
}

bool
nir_opt_load_after_store(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader)
      progress |= load_after_store_impl(impl);

   return progress;
}
