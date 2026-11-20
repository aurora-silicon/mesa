/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_query_pool.h"
#include "hk_queue.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
init_fake_queue(struct hk_queue *queue, uint64_t signalled)
{
   memset(queue, 0, sizeof(*queue));
   queue->drm.syncobj = 0;
   queue->drm.signalled_value = signalled;
}

static void
init_cdm_cs(struct hk_cs *cs)
{
   memset(cs, 0, sizeof(*cs));
   cs->type = HK_CS_CDM;
}

static void
init_vdm_cs(struct hk_cs *cs)
{
   memset(cs, 0, sizeof(*cs));
   cs->type = HK_CS_VDM;
}

int
main(void)
{
   struct hk_device dev;
   memset(&dev, 0, sizeof(dev));
   dev.dev.fd = -1;
   hk_query_stamps_init(&dev);

   struct hk_queue q1, q2;
   init_fake_queue(&q1, 10);
   init_fake_queue(&q2, 20);

   /* Pending CDM end timestamp on handle 7, offset 16 (bytes [16, 24)), point 11. */
   struct hk_cs prod_cdm;
   init_cdm_cs(&prod_cdm);
   prod_cdm.timestamp.end = (struct agx_timestamp_req){
      .handle = 7,
      .offset_B = 16,
   };
   hk_query_stamps_note_cs(&dev, &q1, 11, &prod_cdm);

   /* Pending VDM oq_stamp on handle 8, offset 32 (bytes [32, 40)), point 21. */
   struct hk_cs prod_vdm;
   init_vdm_cs(&prod_vdm);
   prod_vdm.oq_stamp[2] = (struct agx_timestamp_req){
      .handle = 8,
      .offset_B = 32,
   };
   hk_query_stamps_note_cs(&dev, &q2, 21, &prod_vdm);

   /* Disjoint range on same handle -> no conflict. */
   struct hk_cs cons;
   init_cdm_cs(&cons);
   hk_cs_note_query_access(&cons, 7, 0, 16);
   hk_cs_note_query_access(&cons, 7, 24, 16);
   assert(cons.query_access);
   assert(!cons.query_access_untracked);
   assert(!hk_query_stamps_conflict(&dev, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Unbound pool (handle 0) -> query_access set, no conflict. */
   init_cdm_cs(&cons);
   hk_cs_note_query_access(&cons, 0, 0, 0);
   assert(cons.query_access);
   assert(!hk_query_stamps_conflict(&dev, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Overlapping range on handle 7 -> conflict while point 11 > signalled 10. */
   init_cdm_cs(&cons);
   hk_cs_note_query_access(&cons, 7, 8, 16);
   assert(hk_query_stamps_conflict(&dev, &cons));

   /* Once q1 advances past point 11, the pending entry retires and clears. */
   p_atomic_set(&q1.drm.signalled_value, 11);
   assert(!hk_query_stamps_conflict(&dev, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Overlapping VDM oq_stamp on handle 8 -> conflict until unbound or cleared. */
   init_cdm_cs(&cons);
   hk_cs_note_query_access(&cons, 8, 32, 8);
   assert(hk_query_stamps_conflict(&dev, &cons));

   /* Unbinding handle 8 removes its pending entries. */
   hk_query_stamps_unbind(&dev, 8);
   assert(!hk_query_stamps_conflict(&dev, &cons));

   /* Re-add handle 8 on q2 and clear q2. */
   hk_query_stamps_note_cs(&dev, &q2, 22, &prod_vdm);
   assert(hk_query_stamps_conflict(&dev, &cons));
   hk_query_stamps_clear_queue(&dev, &q2);
   assert(!hk_query_stamps_conflict(&dev, &cons));

   /* Untracked access always fails closed. */
   cons.query_access_untracked = true;
   assert(hk_query_stamps_conflict(&dev, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Merging CDM streams preserves query_access_ranges and untracked flag. */
   struct hk_cs a, b;
   init_cdm_cs(&a);
   init_cdm_cs(&b);
   uint8_t dummy_a[4096] = {0}, dummy_b[64] = {0};
   a.start = a.current = dummy_a;
   a.end = dummy_a + sizeof(dummy_a);
   b.start = dummy_b;
   b.current = dummy_b + 16;
   b.end = dummy_b + sizeof(dummy_b);
   hk_query_stamps_note_cs(&dev, &q2, 23, &prod_vdm);
   hk_cs_note_query_access(&b, 8, 32, 8);
   hk_cs_merge_cdm_until(&a, &b, b.current);
   assert(a.query_access);
   assert(hk_query_stamps_conflict(&dev, &a));
   util_dynarray_fini(&a.query_access_ranges);
   util_dynarray_fini(&b.query_access_ranges);
   util_dynarray_fini(&a.compute_resources);
   util_dynarray_fini(&b.compute_resources);

   hk_query_stamps_finish(&dev);
   puts("hk_query_stamps: ok");
   return 0;
}
