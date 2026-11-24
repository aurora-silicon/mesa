/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"

static unsigned tests, failures;

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (name));      \
      }                                                                        \
   } while (0)

struct test_fixture {
   struct hk_device dev;
   struct hk_cmd_buffer cmd;
   struct hk_cs cs;
   uint8_t buf[0x8000];
};

static void
fixture_init(struct test_fixture *f, enum agx_chip chip, uint32_t perftest)
{
   memset(f, 0, sizeof(*f));
   f->dev.vk.base.type = VK_OBJECT_TYPE_DEVICE;
   f->dev.dev.chip = chip;
   f->dev.perftest = perftest;
   f->cmd.vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   f->cmd.vk.base.device = &f->dev.vk;
   list_inithead(&f->cmd.control_streams);
   util_dynarray_init(&f->cmd.sync_pass_records, NULL);

   f->cs.cmd = &f->cmd;
   f->cs.type = HK_CS_CDM;
   f->cs.sync.seq = 1;
   f->cs.start = f->buf;
   f->cs.current = f->buf;
   f->cs.end = f->buf + sizeof(f->buf);
   list_addtail(&f->cs.node, &f->cmd.control_streams);
   f->cmd.current_cs.cs = &f->cs;
}

static VkDependencyInfo
compute_memory_dep(bool host_read)
{
   static VkMemoryBarrier2 bar;
   bar = (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = host_read ? VK_PIPELINE_STAGE_2_HOST_BIT
                                : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = host_read ? VK_ACCESS_2_HOST_READ_BIT
                                 : VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
   };
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &bar,
   };
}

static void
test_default_chip_matrix(void)
{
   T("G13G gen 13 defaults tracked",
     hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G13G, 13));
   T("G13X gen 13 defaults tracked",
     hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G13X, 13));
   T("G14X gen 14 defaults tracked",
     hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G14X, 14));
   T("G14G gen 14 stays per-launch by default",
     !hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G14G, 14));
   T("G14X gen 15 stays per-launch by default",
     !hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G14X, 15));
   T("G15G stays off AGX2 default",
     !hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G15G, 15));
   T("G17G stays off AGX2 default",
     !hk_agx2_tracked_cdm_barrier_default(AGX_CHIP_G17G, 17));
}

static void
test_tracked_compute_stream(void)
{
   struct test_fixture f;
   fixture_init(&f, AGX_CHIP_G14X, 0);

   struct agx_cdm_launch_word_0_packed launch = {0};
   struct agx_grid grid = agx_1d(1);
   struct agx_workgroup wg = agx_workgroup(32, 1, 1);

   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x100, grid, wg);
   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x200, grid, wg);
   T("unordered launches emit 0 flushes",
     f.cs.stats.cmds == 2 && f.cs.stats.flushes == 0);
   T("unordered launches leave no strong flush tail",
     f.cs.strong_flush_end == NULL);

   const VkDependencyInfo compute_dep = compute_memory_dep(false);
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &compute_dep);
   T("compute-to-compute barrier sets pending flag and keeps batch open",
     f.cmd.state.cs.cdm_barrier_pending && f.cmd.current_cs.cs == &f.cs &&
        f.cs.stats.flushes == 0);

   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x300, grid, wg);
   T("next launch consumes pending flag and emits 1 flush",
     f.cs.stats.cmds == 3 && f.cs.stats.flushes == 1 &&
        !f.cmd.state.cs.cdm_barrier_pending);

   /* Pending barrier flushed on compute stream end */
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &compute_dep);
   T("second compute barrier keeps stream open",
     f.cmd.state.cs.cdm_barrier_pending && f.cmd.current_cs.cs == &f.cs);

   hk_cmd_buffer_end_compute(&f.cmd);
   T("ending compute flushes trailing pending barrier and marks strong tail",
     f.cs.stats.flushes == 2 && !f.cmd.state.cs.cdm_barrier_pending &&
        f.cmd.current_cs.cs == NULL && f.cs.strong_flush_end == f.cs.current);

   /* Calling end_stream_flush again must be idempotent */
   hk_cdm_end_stream_flush(&f.cmd, &f.cs);
   T("repeated end_stream_flush does not double-flush", f.cs.stats.flushes == 2);

   /* Host dependency still splits compute batch on AGX2 after flushing */
   fixture_init(&f, AGX_CHIP_G13X, 0);
   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x100, grid, wg);
   const VkDependencyInfo host_dep = compute_memory_dep(true);
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &host_dep);
   T("host barrier splits AGX2 compute batch and flushes pending barrier",
     f.cmd.current_cs.cs == NULL && !f.cmd.state.cs.cdm_barrier_pending &&
        f.cs.stats.flushes == 1 && f.cs.strong_flush_end == f.cs.current);
}

static void
test_non_main_stream_and_alwayscdmbarrier(void)
{
   struct test_fixture f;
   fixture_init(&f, AGX_CHIP_G14X, 0);

   struct hk_cs aux = {
      .cmd = &f.cmd,
      .type = HK_CS_CDM,
      .start = f.buf + 0x4000,
      .current = f.buf + 0x4000,
      .end = f.buf + sizeof(f.buf),
   };
   f.cmd.state.cs.cdm_barrier_pending = true;

   struct agx_cdm_launch_word_0_packed launch = {0};
   struct agx_grid grid = agx_1d(1);
   struct agx_workgroup wg = agx_workgroup(32, 1, 1);

   hk_dispatch_with_usc_launch(&f.dev, &aux, launch, 0x400, grid, wg);
   T("aux stream launch still flushes and preserves main pending flag",
     aux.stats.cmds == 1 && aux.stats.flushes == 1 &&
        aux.strong_flush_end == aux.current &&
        f.cmd.state.cs.cdm_barrier_pending);

   fixture_init(&f, AGX_CHIP_G14G, HK_PERF_ALWAYSCDMBARRIER);
   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x100, grid, wg);
   hk_dispatch_with_usc_launch(&f.dev, &f.cs, launch, 0x200, grid, wg);
   T("alwayscdmbarrier flushes every launch",
     f.cs.stats.cmds == 2 && f.cs.stats.flushes == 2 &&
        f.cs.strong_flush_end == f.cs.current);

   const VkDependencyInfo compute_dep = compute_memory_dep(false);
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &compute_dep);
   T("alwayscdmbarrier splits batch on pipeline barrier",
     f.cmd.current_cs.cs == NULL && !f.cmd.state.cs.cdm_barrier_pending);
}

int
main(void)
{
   test_default_chip_matrix();
   test_tracked_compute_stream();
   test_non_main_stream_and_alwayscdmbarrier();

   printf("hk_agx2_cdm_barrier: %u/%u passed\n", tests - failures, tests);
   return failures ? 1 : 0;
}
