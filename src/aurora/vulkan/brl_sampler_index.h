/*
 * Sampler heap index from the index Borealis publishes in the descriptor set.
 *
 * brl_sampler_descriptor_index() (brl_descriptor_set_layout.h) stores
 * heap_index + 28 on every chip except G17P (the AGX2 bias).  The G17P/G18G
 * combined-sampler pointer path addresses the heap as
 * sampler_heap_va + index * 32 with NO bias, so it must be given the heap
 * index, not the published one.  Returns UINT16_MAX when the published index
 * is below the bias; every caller bounds-checks against sampler_count, which
 * then refuses it.
 *
 * Pure C on purpose: the offline test links the production code.
 */
#ifndef BRL_SAMPLER_INDEX_H
#define BRL_SAMPLER_INDEX_H

#include <stdbool.h>
#include <stdint.h>

#define BRL_SAMPLER_AGX2_BIAS 28u

static inline uint16_t
brl_sampler_heap_index(bool native_indices, uint16_t published)
{
   if (native_indices)
      return published;
   return published >= BRL_SAMPLER_AGX2_BIAS
             ? (uint16_t)(published - BRL_SAMPLER_AGX2_BIAS)
             : UINT16_MAX;
}

#endif
