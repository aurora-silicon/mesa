/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

/*
 * Eliminate stores that write back exactly the value already held at the
 * destination:
 *
 *    %v = load_deref  (d)
 *    store_deref (%v, d)
 *
 * A non-atomic, non-volatile store of the value just loaded from the same
 * address is unobservable: within the invocation nothing between the load and
 * the store may change the location, and across invocations a conflicting
 * write would be a data race in the Vulkan memory model, which is undefined.
 * The same holds for the lowered memory intrinsics (store_ssbo of a
 * load_ssbo at the same index/offset, and the shared/global/task_payload
 * equivalents).
 *
 * Everything between the load and the store must be free of side effects on
 * memory and of any barrier that could publish another thread's write to the
 * address.  Loads and other eliminatable intrinsics are harmless: they do
 * not write, and they cannot make a foreign write visible to this
 * invocation's program order in a race-free program.
 */

static bool
has_volatile_access(nir_intrinsic_instr *intr)
{
   return nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & ACCESS_VOLATILE);
}

/* May the instruction change, or let another thread change, memory that a
 * preceding load in the same block could have observed?  ALU, derefs, undefs
 * and phis are pure.  Eliminatable intrinsics are loads and queries, which
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

/* Both load and store must live in the same block and only harmless
 * instructions may sit between them.
 */
static bool
clean_span(nir_intrinsic_instr *load, nir_intrinsic_instr *store)
{
   if (load->instr.block != store->instr.block)
      return false;

   for (nir_instr *instr = nir_instr_next(&load->instr);
        instr && instr != &store->instr;
        instr = nir_instr_next(instr)) {
      if (is_memory_hazard(instr))
         return false;
   }
   return true;
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

/* Match a store intrinsic to the load intrinsic producing its value.  On
 * success returns the load, or NULL if the store is not a self-write. */
static nir_intrinsic_instr *
matching_load(nir_intrinsic_instr *store)
{
   switch (store->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_task_payload:
   case nir_intrinsic_store_global:
      break;
   default:
      return NULL;
   }

   /* store_deref is src[] = { deref, value }; the lowered memory stores are
    * src[] = { value, ... }. */
   const bool is_deref = store->intrinsic == nir_intrinsic_store_deref;
   nir_def *value = store->src[is_deref ? 1 : 0].ssa;
   nir_instr *value_instr = nir_def_instr(value);
   if (value_instr->type != nir_instr_type_intrinsic)
      return NULL;

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(value_instr);

   if (has_volatile_access(load) || has_volatile_access(store))
      return NULL;

   if (is_deref) {
      if (load->intrinsic != nir_intrinsic_load_deref)
         return NULL;

      nir_deref_instr *store_deref = nir_src_as_deref(store->src[0]);
      nir_deref_instr *load_deref = nir_src_as_deref(load->src[0]);
      if (!store_deref || !load_deref)
         return NULL;

      const nir_variable_mode modes =
         nir_var_mem_ssbo | nir_var_mem_shared | nir_var_mem_global |
         nir_var_mem_task_payload;
      if (!nir_deref_mode_may_be(store_deref, modes))
         return NULL;

      if (!(nir_compare_derefs(store_deref, load_deref) &
            nir_derefs_equal_bit))
         return NULL;

      return load;
   }

   switch (store->intrinsic) {

   case nir_intrinsic_store_ssbo: {
      /* src[] = { value, buffer_index, offset } */
      if (load->intrinsic != nir_intrinsic_load_ssbo)
         return NULL;
      if (nir_intrinsic_offset_shift(store) !=
          nir_intrinsic_offset_shift(load))
         return NULL;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)) ||
          !same_scalar(nir_scalar_resolved(store->src[2].ssa, 0),
                       nir_scalar_resolved(load->src[1].ssa, 0)))
         return NULL;
      return load;
   }

   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_task_payload: {
      nir_intrinsic_op match = store->intrinsic == nir_intrinsic_store_shared
                               ? nir_intrinsic_load_shared
                               : nir_intrinsic_load_task_payload;
      if (load->intrinsic != match)
         return NULL;
      if (nir_intrinsic_base(store) != nir_intrinsic_base(load))
         return NULL;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)))
         return NULL;
      return load;
   }

   case nir_intrinsic_store_global: {
      /* src[] = { value, address } */
      if (load->intrinsic != nir_intrinsic_load_global)
         return NULL;
      if (!same_scalar(nir_scalar_resolved(store->src[1].ssa, 0),
                       nir_scalar_resolved(load->src[0].ssa, 0)))
         return NULL;
      return load;
   }

   default:
      return NULL;
   }
}

static bool
dead_self_stores_impl(nir_function_impl *impl)
{
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         nir_intrinsic_instr *load = matching_load(store);
         if (!load || !clean_span(load, store))
            continue;

         /* Drop only the store.  The load may have other uses; if it does
          * not, ordinary DCE reclaims it. */
         nir_instr_remove(&store->instr);
         progress = true;
      }
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_none);

   return progress;
}

bool
nir_opt_dead_self_stores(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader)
      progress |= dead_self_stores_impl(impl);

   return progress;
}
