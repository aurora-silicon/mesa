/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_slot_core.h"
#include "util/bitscan.h"
#include "util/macros.h"

#include <stdlib.h>
#include <string.h>

#define AGX3_SLOT_TRACK_CAP 32u

/* Most regions touch only GPRs. Do not scan the entire static TPS address
 * space every time a scoreboard generation retires. Every writer alias and
 * payload reader below comes from a source or destination in this region,
 * so tokens beyond this bound remain empty throughout the assignment.
 *
 * Leave malformed-input handling to the assignment pass, including which
 * prefix of the output it has written when it rejects an instruction. */
static unsigned
agx3_slot_token_count(const struct agx3_slot_instr *instrs, unsigned n)
{
   unsigned count = 1;
   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return AGX3_SLOT_NUM_TOKENS;

      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->src[s] >= count)
            count = I->src[s] + 1u;
      }
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->dst[d] >= count)
            count = I->dst[d] + 1u;
      }
   }
   return count;
}

/* One assignment state, as a structured level saves it. */
struct agx3_slot_state {
   uint16_t pending, entry_pending, active_reader_slots;
   bool waited[9], reissuable[9];
   uint8_t join_class[9], join_count[9];
   uint16_t reg_count[9], reader_count[9];
   uint16_t *reg_slots;
   uint16_t *reader_slots;
};

/* A structured if (or loop) level: the state at the if, at the end of its
 * first arm once an else has started the second, and, for a loop, the join
 * of the states its breaks left. */
struct agx3_slot_level {
   bool arm, has_else, loop, has_break;
   struct agx3_slot_state at_if, then_end, breaks;
};

/* Two paths' join class for one slot, when both leave it pending with
 * different queues: no producer may join it, and any reuse waits first. */
#define AGX3_SLOT_JOIN_MIXED UINT8_MAX

struct agx3_slot_levels {
   struct agx3_slot_level *levels;
   unsigned max_depth;
   /* Where a break keeps the state it continues with while it joins that
    * state into its loop's. */
   struct agx3_slot_state scratch;
};

static bool
agx3_slot_assign_core(const struct agx3_slot_instr *instrs, unsigned n,
                      unsigned nslots, const struct agx3_slot_entry *entry,
                      unsigned entry_count,
                      const struct agx3_slot_live_range *live,
                      unsigned live_count, struct agx3_slot_result *out,
                      struct agx3_slot_levels *lv, unsigned token_count)
{
   if (instrs == NULL || out == NULL || nslots < 1 || nslots > 8 ||
       (entry_count && entry == NULL) || entry_count > AGX3_SLOT_NUM_TOKENS ||
       (live_count && !live))
      return false;

   for (unsigned i = 0; i < entry_count; ++i) {
      if (entry[i].token >= AGX3_SLOT_NUM_TOKENS ||
          entry[i].slot < 1 || entry[i].slot > nslots)
         return false;
      if (entry[i].token >= token_count)
         token_count = entry[i].token + 1u;
   }

   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      if (!range->count || !range->slot || range->slot > nslots ||
          range->first >= AGX3_SLOT_NUM_TOKENS ||
          range->count > AGX3_SLOT_NUM_TOKENS - range->first)
         return false;
      if (range->first + range->count > token_count)
         token_count = range->first + range->count;
   }

   /* reg_slots[r] = the slots (bit s - 1 for slot s) of the async ops whose
    * result may currently live in half-register r, or 0 if r was last written
    * by a synchronous op (or never). One path names one slot; after the arms
    * of an if join, r may hold the result of either arm's op. */
   uint16_t stack_reg_slots[1024];
   uint16_t stack_reader_slots[1024];
   uint16_t *reg_slots = token_count <= ARRAY_SIZE(stack_reg_slots)
                            ? stack_reg_slots
                            : malloc(token_count * sizeof(uint16_t));
   uint16_t *reader_slots = token_count <= ARRAY_SIZE(stack_reader_slots)
                               ? stack_reader_slots
                               : malloc(token_count * sizeof(uint16_t));
   if (!reg_slots || !reader_slots) {
      if (reg_slots != stack_reg_slots)
         free(reg_slots);
      if (reader_slots != stack_reader_slots)
         free(reader_slots);
      return false;
   }
   bool ok = false;
   memset(reg_slots, 0, token_count * sizeof(reg_slots[0]));
   uint16_t slot_reg_count[9] = {0};
   uint16_t slot_reg_tokens[9][AGX3_SLOT_TRACK_CAP];
   uint8_t slot_reg_len[9] = {0};
   uint16_t pending = 0;
   for (unsigned i = 0; i < entry_count; ++i) {
      /* One token cannot name two independently pending generations. */
      if (reg_slots[entry[i].token])
         goto out_free;
      const uint8_t s = entry[i].slot;
      const uint16_t r = entry[i].token;
      reg_slots[r] = (uint16_t)(1u << (s - 1));
      slot_reg_count[s]++;
      if (slot_reg_len[s] < AGX3_SLOT_TRACK_CAP)
         slot_reg_tokens[s][slot_reg_len[s]++] = r;
      else
         slot_reg_len[s] = UINT8_MAX;
      pending |= (uint16_t)(1u << (s - 1));
   }
   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      const uint8_t s = range->slot;
      const uint16_t bit = (uint16_t)(1u << (s - 1));
      for (unsigned r = range->first; r < range->first + range->count; ++r) {
         if (reg_slots[r])
            goto out_free;
         reg_slots[r] = bit;
         slot_reg_count[s]++;
         if (slot_reg_len[s] < AGX3_SLOT_TRACK_CAP)
            slot_reg_tokens[s][slot_reg_len[s]++] = (uint16_t)r;
         else
            slot_reg_len[s] = UINT8_MAX;
      }
      pending |= bit;
   }
   uint16_t entry_pending = pending;

   /* reader_slots[r] = the slots of the in-flight stores that latch
    * half-register r as payload. A store has no destination, so the result
    * bookkeeping above never covers its payload: the transform feedback copy
    * program reuses one payload quad for all sixteen captured varyings, and
    * the next varying's setup raced the outstanding store. */
   bool reader_slots_init = false;
   uint16_t slot_reader_count[9] = {0};
   uint16_t slot_reader_tokens[9][AGX3_SLOT_TRACK_CAP];
   uint8_t slot_reader_len[9] = {0};
   uint16_t active_reader_slots = 0;

   /* slot_waited[s] = the slot has been synchronised since it was last
    * allocated, so later consumers need not wait it again. */
   bool slot_waited[9] = {false};

   /* slot_reissuable[s] = the slot's outstanding generation was issued by an
    * instruction that reissues live slots itself. */
   bool slot_reissuable[9] = {false};
   uint8_t slot_join_class[9] = {0};
   uint8_t slot_join_count[9] = {0};

   unsigned next = 0; /* round-robin cursor into 0..nslots-1 */
   unsigned depth = 0; /* open structured control-flow levels */

#define AGX3_CLEAR_REG_SLOT(S)                                                \
   do {                                                                       \
      const unsigned _s = (S);                                                \
      if (slot_reg_count[_s]) {                                               \
         const uint16_t _mask = (uint16_t)~(1u << (_s - 1));                  \
         if (slot_reg_len[_s] != UINT8_MAX) {                                 \
            for (unsigned _k = 0; _k < slot_reg_len[_s]; ++_k)                \
               reg_slots[slot_reg_tokens[_s][_k]] &= _mask;                   \
         } else {                                                             \
            for (unsigned _r = 0; _r < token_count; ++_r)                     \
               reg_slots[_r] &= _mask;                                        \
         }                                                                    \
         slot_reg_count[_s] = 0;                                              \
      }                                                                       \
      slot_reg_len[_s] = 0;                                                   \
   } while (0)

#define AGX3_CLEAR_READER_MASK(MASK)                                          \
   do {                                                                       \
      const uint16_t _cm = (MASK) & active_reader_slots;                      \
      if (_cm) {                                                              \
         bool _need_full = false;                                             \
         u_foreach_bit(_b, _cm) {                                             \
            if (slot_reader_count[_b + 1] &&                                  \
                slot_reader_len[_b + 1] == UINT8_MAX) {                       \
               _need_full = true;                                             \
               break;                                                         \
            }                                                                 \
         }                                                                    \
         if (_need_full) {                                                    \
            for (unsigned _r = 0; _r < token_count; ++_r)                     \
               reader_slots[_r] &= (uint16_t)~_cm;                            \
            u_foreach_bit(_b, _cm) {                                          \
               slot_reader_count[_b + 1] = 0;                                 \
               slot_reader_len[_b + 1] = 0;                                   \
            }                                                                 \
         } else {                                                             \
            u_foreach_bit(_b, _cm) {                                          \
               const unsigned _s = _b + 1;                                    \
               const uint16_t _rm = (uint16_t)~(1u << _b);                    \
               for (unsigned _k = 0; _k < slot_reader_len[_s]; ++_k)          \
                  reader_slots[slot_reader_tokens[_s][_k]] &= _rm;            \
               slot_reader_count[_s] = 0;                                     \
               slot_reader_len[_s] = 0;                                       \
            }                                                                 \
         }                                                                    \
         active_reader_slots &= (uint16_t)~_cm;                               \
      }                                                                       \
   } while (0)

   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         goto out_free;
      if (I->preferred_slot > nslots || (!I->is_async && I->preferred_slot))
         goto out_free;
      if (I->coalesce_src > I->nr_srcs || (!I->is_async && I->coalesce_src))
         goto out_free;
      if ((I->join_class && (!I->is_async || !I->join_limit || I->join_limit > 8)) ||
          (!I->join_class && (I->join_limit || I->prefer_join)))
         goto out_free;
      const uint16_t valid_slots = (uint16_t)((1u << nslots) - 1u);
      if ((I->nr_srcs < 32 && (I->late_src_mask >> I->nr_srcs)) ||
          (!I->is_async && I->late_src_mask))
         goto out_free;
      if ((I->transient_slots & ~valid_slots) ||
          (I->preferred_slot &&
           (I->transient_slots & (1u << (I->preferred_slot - 1)))))
         goto out_free;

      uint16_t wait_mask = 0;
      if (I->drain) {
         wait_mask = pending;
         for (unsigned slot = 1; slot <= nslots; ++slot)
            if (pending & (1u << (slot - 1)))
               slot_waited[slot] = true;
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         const uint16_t r = I->src[s];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            goto out_free;
         u_foreach_bit(b, reg_slots[r]) {
            if (!slot_waited[b + 1]) {
               wait_mask |= (uint16_t)(1u << b);
               slot_waited[b + 1] = true;
            }
         }
      }

      /*
       * Write-after-write. An async result that is still in flight for this
       * register lands LATER than this instruction, so a synchronous write
       * here is silently overwritten by the load when it retires -- and any
       * consumer that waits the slot then reads the loaded word instead of
       * the computed one. Mesa's own AGX2 pass checks this case explicitly
       * (agx_insert_waits.c, "Check for write-after-write"); omitting it here
       * produced no diagnostic, just wrong values on hardware.
       *
       * This runs before the bookkeeping below, so reg_slots[] still names the
       * previous writers rather than this instruction.
       */
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         const uint16_t r = I->dst[d];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            goto out_free;
         u_foreach_bit(b, reg_slots[r]) {
            if (!slot_waited[b + 1]) {
               wait_mask |= (uint16_t)(1u << b);
               slot_waited[b + 1] = true;
            }
         }

         /* Write-after-read: retire every async op still reading this
          * register before redefining it. */
         if (active_reader_slots) {
            uint16_t readers = reader_slots[r];
            while (readers) {
               const unsigned bit = (unsigned)(readers & (~readers + 1u));
               const uint8_t reader = (uint8_t)(__builtin_ctz(bit) + 1);
               readers &= (uint16_t)~bit;
               if (!slot_waited[reader]) {
                  wait_mask |= (uint16_t)bit;
                  slot_waited[reader] = true;
               }
            }
         }
      }

      /* A compound instruction may use a scoreboard slot internally, for
       * example PACK_SMPIR/COPY producers followed by a TEX which waits for
       * them. Retire a prior generation before reusing the slot, then forget
       * its aliases: the prewait has made those GPR values synchronous and
       * the internal generation is consumed before this logical instruction
       * returns. */
      if (I->transient_slots) {
         u_foreach_bit(b, I->transient_slots) {
            const unsigned slot = b + 1;
            const uint16_t bit = (uint16_t)(1u << b);
            const bool live_slot =
               slot_reg_count[slot] != 0 || slot_reader_count[slot] != 0;
            AGX3_CLEAR_REG_SLOT(slot);
            if (live_slot && !slot_waited[slot])
               wait_mask |= bit;
            slot_waited[slot] = true;
            pending &= (uint16_t)~bit;
         }
      }

      /* Every slot this instruction waited is now retired, so the registers
       * its async ops were reading are free again. */
      AGX3_CLEAR_READER_MASK(wait_mask);

      if (I->is_async) {
         /*
          * A slot names one generation at a time. Issuing it while an earlier
          * generation is still outstanding for a live result leaves that
          * result's consumer waiting on the new generation instead, so it can
          * read its register before the older operation has written it. The
          * round-robin allocation therefore skips such slots, and when every
          * slot is in that state it waits the old generation out first.
          *
          * A preferred slot keeps its witnessed protocol: Apple's fragment
          * tile loads, for one, issue slot 6 again for each tile read. So
          * does an instruction that reissues live slots, such as a fragment
          * coefficient load, which has no wait field: it only skips a slot
          * whose live generation some other kind of instruction issued, and
          * when every slot is in that state it takes the next one as before.
          */
         uint8_t slot = I->preferred_slot;
         const unsigned cap =
            I->max_slot && I->max_slot < nslots ? I->max_slot : nslots;
         if (!slot) {
            if (I->coalesce_src) {
               const unsigned source = I->src[I->coalesce_src - 1];
               /* Only a source with one possible producer. */
               const uint16_t producers = reg_slots[source];
               const uint8_t producer_slot =
                  util_bitcount(producers) == 1
                     ? (uint8_t)(__builtin_ctz(producers) + 1) : 0;
               if (producer_slot && producer_slot <= cap &&
                   slot_waited[producer_slot] &&
                   !(I->transient_slots & (1u << (producer_slot - 1))))
                  slot = producer_slot;
            }
            if (I->prefer_join && !slot) {
               for (unsigned t = 0; t < nslots && !slot; ++t) {
                  const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
                  if (candidate <= cap && !slot_waited[candidate] &&
                      slot_join_class[candidate] == I->join_class &&
                      slot_join_count[candidate] < I->join_limit)
                     slot = candidate;
               }
            }
            /* Prefer an already-waited or unused slot before joining or
             * reissuing a live generation, so independent instructions do not
             * unnecessarily serialize onto a single scoreboard slot. */
            for (unsigned t = 0; t < nslots && !slot; ++t) {
               const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
               if (candidate > cap)
                  continue;
               if (slot_waited[candidate] ||
                   (slot_reg_count[candidate] == 0 &&
                    slot_reader_count[candidate] == 0))
                  slot = candidate;
            }
            if (!slot && (I->reissues_live_slot || I->join_class)) {
               for (unsigned t = 0; t < nslots && !slot; ++t) {
                  const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
                  if (candidate > cap)
                     continue;
                  if ((I->reissues_live_slot && slot_reissuable[candidate]) ||
                      (I->join_class &&
                       slot_join_class[candidate] == I->join_class &&
                       slot_join_count[candidate] < I->join_limit))
                     slot = candidate;
               }
            }
            if (!slot) {
               slot = (uint8_t)(next + 1);
               if (slot > cap)
                  slot = 1;
               if (!I->reissues_live_slot) {
                  wait_mask |= (uint16_t)(1u << (slot - 1));
                  slot_waited[slot] = true;
               }
            }
            next = slot % nslots;
         }

         /* A preferred/reissuable native instruction may normally join an
          * in-region generation. The external loader's generation is not
          * part of that protocol: retire it before any slot reuse. */
         if ((entry_pending & (1u << (slot - 1))) && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* A joined queue must complete before an incompatible producer
          * replaces it. In particular, an SCS completion or a texture
          * publication must not inherit a pending device-read generation. */
         if (!slot_waited[slot] && slot_join_count[slot] &&
             ((slot_join_class[slot] != I->join_class) ||
              (I->join_class && slot_join_count[slot] >= I->join_limit))) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         out[i].wait_mask = wait_mask;
         out[i].slot = slot;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         pending |= (uint16_t)(1u << (slot - 1));

         /* Allocation can add a wait when every generation is occupied.
          * Retire its payload readers before recording this instruction. */
         AGX3_CLEAR_READER_MASK(wait_mask);

         /* Reusing an already-synchronized slot starts a new dependency
          * generation. Retire the old generation's register aliases before
          * publishing the new result, otherwise an unrelated later write can
          * consume the new generation's wait and hide it from its consumer. */
         const bool new_join = slot_waited[slot] ||
                               slot_join_class[slot] != I->join_class;
         if (slot_waited[slot])
            AGX3_CLEAR_REG_SLOT(slot);
         slot_waited[slot] = false; /* freshly outstanding */
         slot_reissuable[slot] = I->reissues_live_slot;
         slot_join_count[slot] = I->join_class ?
            (new_join ? 1 : slot_join_count[slot] + 1) : 0;
         slot_join_class[slot] = I->join_class;
         const uint16_t slot_bit = (uint16_t)(1u << (slot - 1));
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            const uint16_t r = I->dst[d];
            if (r >= AGX3_SLOT_NUM_TOKENS)
               goto out_free;
            const uint16_t old = reg_slots[r];
            if (old) {
               u_foreach_bit(ob, old)
                  slot_reg_count[ob + 1]--;
            }
            reg_slots[r] = slot_bit;
            slot_reg_count[slot]++;
            if (slot_reg_len[slot] < AGX3_SLOT_TRACK_CAP)
               slot_reg_tokens[slot][slot_reg_len[slot]++] = r;
            else
               slot_reg_len[slot] = UINT8_MAX;
         }
         if ((I->latches_srcs || I->late_src_mask) && I->nr_srcs) {
            if (!reader_slots_init) {
               memset(reader_slots, 0, token_count * sizeof(reader_slots[0]));
               reader_slots_init = true;
            }
            active_reader_slots |= slot_bit;
            for (unsigned s = 0; s < I->nr_srcs; ++s) {
               if (I->latches_srcs || (I->late_src_mask & (UINT32_C(1) << s))) {
                  const uint16_t r = I->src[s];
                  if (!(reader_slots[r] & slot_bit)) {
                     reader_slots[r] |= slot_bit;
                     slot_reader_count[slot]++;
                     if (slot_reader_len[slot] < AGX3_SLOT_TRACK_CAP)
                        slot_reader_tokens[slot][slot_reader_len[slot]++] = r;
                     else
                        slot_reader_len[slot] = UINT8_MAX;
                  }
               }
            }
         }
      } else {
         out[i].wait_mask = wait_mask;
         out[i].slot = 0;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            const uint16_t r = I->dst[d];
            if (r >= AGX3_SLOT_NUM_TOKENS)
               goto out_free;
            const uint16_t old = reg_slots[r];
            if (old) {
               u_foreach_bit(ob, old)
                  slot_reg_count[ob + 1]--;
               reg_slots[r] = 0; /* synchronous result, no wait needed */
            }
         }
      }

      /* Emission retires every outstanding generation before a draining
       * control-flow boundary. Start the code after it from that empty
       * scoreboard, so allocation neither skips nor waits for generations
       * the drain already retired. */
      if (I->cf != AGX3_SLOT_CF_NONE) {
         for (unsigned slot = 1; slot <= nslots; ++slot)
            AGX3_CLEAR_REG_SLOT(slot);
         AGX3_CLEAR_READER_MASK(active_reader_slots);
         pending = 0;
         entry_pending = 0;
         for (unsigned slot = 1; slot <= nslots; ++slot) {
            slot_waited[slot] = true;
            slot_reissuable[slot] = false;
            slot_join_count[slot] = 0;
            slot_join_class[slot] = 0;
         }
      }

      /* The arms of an if join (enum agx3_slot_cf): each starts from the
       * state at the if, an else's arm also from what the first arm left in
       * flight, and the code after the pop from both arms' ends. A path no
       * lane takes still contributes its state, since its waits did not
       * run. */
#define AGX3_SLOT_SAVE(S)                                                     \
   do {                                                                       \
      memcpy((S)->reg_slots, reg_slots, token_count * sizeof(reg_slots[0])); \
      if (active_reader_slots)                                                \
         memcpy((S)->reader_slots, reader_slots,                              \
                token_count * sizeof(reader_slots[0]));                       \
      (S)->pending = pending;                                                \
      (S)->entry_pending = entry_pending;                                    \
      (S)->active_reader_slots = active_reader_slots;                        \
      memcpy((S)->waited, slot_waited, sizeof(slot_waited));                 \
      memcpy((S)->reissuable, slot_reissuable, sizeof(slot_reissuable));     \
      memcpy((S)->join_class, slot_join_class, sizeof(slot_join_class));     \
      memcpy((S)->join_count, slot_join_count, sizeof(slot_join_count));     \
      memcpy((S)->reg_count, slot_reg_count, sizeof(slot_reg_count));        \
      memcpy((S)->reader_count, slot_reader_count,                           \
             sizeof(slot_reader_count));                                      \
   } while (0)
#define AGX3_SLOT_RESTORE(S)                                                  \
   do {                                                                       \
      memcpy(reg_slots, (S)->reg_slots, token_count * sizeof(reg_slots[0])); \
      if ((S)->active_reader_slots) {                                         \
         memcpy(reader_slots, (S)->reader_slots,                              \
                token_count * sizeof(reader_slots[0]));                       \
         reader_slots_init = true;                                            \
      } else if (active_reader_slots) {                                       \
         memset(reader_slots, 0, token_count * sizeof(reader_slots[0]));      \
      }                                                                       \
      pending = (S)->pending;                                                \
      entry_pending = (S)->entry_pending;                                    \
      active_reader_slots = (S)->active_reader_slots;                        \
      memcpy(slot_waited, (S)->waited, sizeof(slot_waited));                 \
      memcpy(slot_reissuable, (S)->reissuable, sizeof(slot_reissuable));     \
      memcpy(slot_join_class, (S)->join_class, sizeof(slot_join_class));     \
      memcpy(slot_join_count, (S)->join_count, sizeof(slot_join_count));     \
      memcpy(slot_reg_count, (S)->reg_count, sizeof(slot_reg_count));        \
      memcpy(slot_reader_count, (S)->reader_count,                           \
             sizeof(slot_reader_count));                                      \
      for (unsigned _s = 1; _s <= nslots; ++_s) {                             \
         slot_reg_len[_s] = slot_reg_count[_s] ? UINT8_MAX : 0;               \
         slot_reader_len[_s] = slot_reader_count[_s] ? UINT8_MAX : 0;         \
      }                                                                       \
   } while (0)
#define AGX3_SLOT_JOIN(S)                                                     \
   do {                                                                       \
      for (unsigned r = 0; r < token_count; ++r) {                           \
         const uint16_t _old = reg_slots[r];                                 \
         const uint16_t _add = (S)->reg_slots[r] & (uint16_t)~_old;          \
         if (_add) {                                                         \
            reg_slots[r] = (uint16_t)(_old | _add);                          \
            u_foreach_bit(_b, _add)                                          \
               slot_reg_count[_b + 1]++;                                     \
         }                                                                   \
      }                                                                      \
      if ((S)->active_reader_slots) {                                        \
         if (!reader_slots_init) {                                           \
            memset(reader_slots, 0, token_count * sizeof(reader_slots[0]));  \
            reader_slots_init = true;                                        \
         }                                                                   \
         for (unsigned r = 0; r < token_count; ++r) {                        \
            const uint16_t _old = reader_slots[r];                           \
            const uint16_t _add = (S)->reader_slots[r] & (uint16_t)~_old;    \
            if (_add) {                                                      \
               reader_slots[r] = (uint16_t)(_old | _add);                    \
               u_foreach_bit(_b, _add)                                       \
                  slot_reader_count[_b + 1]++;                               \
            }                                                                \
         }                                                                   \
      }                                                                      \
      for (unsigned slot = 1; slot <= nslots; ++slot) {                      \
         const uint16_t bit = (uint16_t)(1u << (slot - 1));                  \
         const bool here = pending & bit, there = (S)->pending & bit;        \
         if (here && there) {                                                \
            slot_reissuable[slot] &= (S)->reissuable[slot];                  \
            if (slot_join_class[slot] != (S)->join_class[slot]) {            \
               slot_join_class[slot] = AGX3_SLOT_JOIN_MIXED;                 \
               slot_join_count[slot] = 1;                                    \
            } else {                                                         \
               slot_join_count[slot] =                                       \
                  MAX2(slot_join_count[slot], (S)->join_count[slot]);        \
            }                                                                \
         } else if (there) {                                                 \
            slot_reissuable[slot] = (S)->reissuable[slot];                   \
            slot_join_class[slot] = (S)->join_class[slot];                   \
            slot_join_count[slot] = (S)->join_count[slot];                   \
         }                                                                   \
         /* Synchronised exactly when no joined path left it pending. */    \
         slot_waited[slot] = !here && !there;                                \
         if (slot_reg_count[slot])                                           \
            slot_reg_len[slot] = UINT8_MAX;                                  \
         if (slot_reader_count[slot])                                        \
            slot_reader_len[slot] = UINT8_MAX;                               \
      }                                                                      \
      pending |= (S)->pending;                                               \
      entry_pending |= (S)->entry_pending;                                   \
      active_reader_slots |= (S)->active_reader_slots;                       \
   } while (0)
      /* A break's lanes resume at its loop's pop with what they left in
       * flight; the lanes that stay continue with the same state. */
      if (I->cf_break) {
         if (I->cf_break > depth)
            goto out_free;
         struct agx3_slot_level *L = &lv->levels[depth - I->cf_break];
         if (!L->loop)
            goto out_free;
         if (!L->has_break) {
            AGX3_SLOT_SAVE(&L->breaks);
            L->has_break = true;
         } else {
            AGX3_SLOT_SAVE(&lv->scratch);
            AGX3_SLOT_JOIN(&L->breaks);
            AGX3_SLOT_SAVE(&L->breaks);
            AGX3_SLOT_RESTORE(&lv->scratch);
         }
      }
      if (I->cf_else) {
         if (!depth)
            goto out_free;
         struct agx3_slot_level *L = &lv->levels[depth - 1];
         if (L->arm) {
            AGX3_SLOT_SAVE(&L->then_end);
            L->has_else = true;
            AGX3_SLOT_JOIN(&L->at_if);
         }
      }
      for (unsigned p = 0; p < I->cf_pop; ++p) {
         if (!depth)
            goto out_free;
         const struct agx3_slot_level *L = &lv->levels[--depth];
         /* An arm, or a loop's breaks, keep state only when its pop closes
          * it alone. */
         if ((L->arm || L->has_break) && I->cf_pop != 1)
            goto out_free;
         if (L->arm)
            AGX3_SLOT_JOIN(L->has_else ? &L->then_end : &L->at_if);
         if (L->has_break)
            AGX3_SLOT_JOIN(&L->breaks);
      }
      for (unsigned p = 0; p < I->cf_push; ++p) {
         if (depth >= lv->max_depth)
            goto out_free;
         struct agx3_slot_level *L = &lv->levels[depth++];
         L->arm = I->cf_arm;
         L->has_else = false;
         L->loop = I->cf_loop;
         L->has_break = false;
         if (L->arm)
            AGX3_SLOT_SAVE(&L->at_if);
      }
#undef AGX3_SLOT_JOIN
#undef AGX3_SLOT_RESTORE
#undef AGX3_SLOT_SAVE
#undef AGX3_CLEAR_READER_MASK
#undef AGX3_CLEAR_REG_SLOT
   }

   ok = true;

out_free:
   if (reg_slots != stack_reg_slots)
      free(reg_slots);
   if (reader_slots != stack_reader_slots)
      free(reader_slots);
   return ok;
}

/* Size the per-level snapshots like agx3_slot_assign_core() sizes its token
 * domain, run it, and free them. */
static bool
agx3_slot_assign_initial(const struct agx3_slot_instr *instrs, unsigned n,
                         unsigned nslots, const struct agx3_slot_entry *entry,
                         unsigned entry_count,
                         const struct agx3_slot_live_range *live,
                         unsigned live_count, struct agx3_slot_result *out)
{
   if (instrs == NULL)
      return false;

   unsigned depth = 0, max_depth = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (instrs[i].cf_pop > depth)
         return false;
      depth = depth - instrs[i].cf_pop + instrs[i].cf_push;
      max_depth = MAX2(max_depth, depth);
   }

   unsigned stride = agx3_slot_token_count(instrs, n);
   for (unsigned i = 0; entry && i < entry_count; ++i)
      stride = MAX2(stride, (unsigned)entry[i].token + 1u);
   for (unsigned i = 0; live && i < live_count; ++i)
      stride = MAX2(stride, (unsigned)live[i].first + live[i].count);
   stride = MIN2(stride, AGX3_SLOT_NUM_TOKENS);

   struct agx3_slot_level stack_levels[8];
   uint16_t stack_arena[4096];
   struct agx3_slot_levels lv = {.max_depth = max_depth};
   uint16_t *arena = NULL;
   if (max_depth) {
      const size_t arena_elems = (size_t)(3 * max_depth + 1) * 2 * stride;
      lv.levels = max_depth <= ARRAY_SIZE(stack_levels)
                     ? stack_levels
                     : calloc(max_depth, sizeof(*lv.levels));
      arena = arena_elems <= ARRAY_SIZE(stack_arena)
                 ? stack_arena
                 : calloc(arena_elems, sizeof(uint16_t));
      if (!lv.levels || !arena) {
         if (lv.levels != stack_levels)
            free(lv.levels);
         if (arena != stack_arena)
            free(arena);
         return false;
      }
      if (lv.levels == stack_levels)
         memset(stack_levels, 0, max_depth * sizeof(*lv.levels));
      uint16_t *base = arena;
      lv.scratch.reg_slots = base;
      lv.scratch.reader_slots = base + stride;
      base += 2 * stride;
      for (unsigned d = 0; d < max_depth; ++d) {
         struct agx3_slot_state *level[] = {
            &lv.levels[d].at_if, &lv.levels[d].then_end, &lv.levels[d].breaks,
         };
         for (unsigned i = 0; i < ARRAY_SIZE(level); ++i, base += 2 * stride) {
            level[i]->reg_slots = base;
            level[i]->reader_slots = base + stride;
         }
      }
   }

   const bool ok = agx3_slot_assign_core(instrs, n, nslots, entry, entry_count,
                                         live, live_count, out, &lv, stride);
   if (lv.levels != stack_levels)
      free(lv.levels);
   if (arena != stack_arena)
      free(arena);
   return ok;
}

bool
agx3_slot_assign_with_entry(const struct agx3_slot_instr *instrs, unsigned n,
                            unsigned nslots, const struct agx3_slot_entry *entry,
                            unsigned entry_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, entry, entry_count,
                                   NULL, 0, out);
}

bool
agx3_slot_assign_with_live(const struct agx3_slot_instr *instrs, unsigned n,
                           unsigned nslots,
                           const struct agx3_slot_live_range *live,
                           unsigned live_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, NULL, 0,
                                   live, live_count, out);
}

bool
agx3_slot_assign(const struct agx3_slot_instr *instrs, unsigned n,
                 unsigned nslots, struct agx3_slot_result *out)
{
   return agx3_slot_assign_with_entry(instrs, n, nslots, NULL, 0, out);
}

void
agx3_slot_retire_at_cf(const struct agx3_slot_instr *instrs, unsigned n,
                       uint16_t entry_wait, struct agx3_slot_result *res,
                       bool *drained)
{
   /* Assignment already starts from the state each drain leaves, so no
    * wait names a generation a drain retired; only report the drains,
    * following the same arm joins. */
   struct level {
      bool arm, has_else, loop;
      uint16_t at_if, then_end, breaks;
   };
   struct level stack_levels[64];
   struct level *levels = n <= ARRAY_SIZE(stack_levels)
                             ? stack_levels
                             : calloc(n ? n : 1, sizeof(*levels));
   if (levels == stack_levels && n)
      memset(stack_levels, 0, n * sizeof(*levels));
   uint16_t live = entry_wait;
   unsigned depth = 0;
   for (unsigned i = 0; i < n; ++i) {
      live &= (uint16_t)~res[i].wait_mask;
      if (res[i].slot)
         live |= (uint16_t)(1u << (res[i].slot - 1));

      if (instrs[i].cf != AGX3_SLOT_CF_NONE) {
         if (drained)
            drained[i] = live != 0;
         live = 0;
      }
      if (!levels)
         continue;
      if (instrs[i].cf_break && instrs[i].cf_break <= depth)
         levels[depth - instrs[i].cf_break].breaks |= live;
      if (instrs[i].cf_else && depth && levels[depth - 1].arm) {
         levels[depth - 1].then_end = live;
         levels[depth - 1].has_else = true;
         live |= levels[depth - 1].at_if;
      }
      for (unsigned p = 0; p < instrs[i].cf_pop && depth; ++p) {
         const struct level *L = &levels[--depth];
         if (L->arm)
            live |= L->has_else ? L->then_end : L->at_if;
         if (L->loop)
            live |= L->breaks;
      }
      for (unsigned p = 0; p < instrs[i].cf_push && depth < n; ++p) {
         levels[depth++] = (struct level){
            .arm = instrs[i].cf_arm, .loop = instrs[i].cf_loop, .at_if = live,
         };
      }
   }
   if (levels != stack_levels)
      free(levels);
}
