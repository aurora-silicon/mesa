/* SPDX-License-Identifier: MIT */
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_query_pool.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"

#include <stdio.h>
#undef NDEBUG
#include <assert.h>

/* Only GPU emission is replaced. Recording entrypoints, secondary command
 * queue expansion, QueueSubmit2 registration and the common runtime are real.
 */
__typeof__(hk_sync_write_timestamp) __wrap_hk_sync_write_timestamp;
__typeof__(hk_sync_flush_pending_timestamp)
   __wrap_hk_sync_flush_pending_timestamp;
__typeof__(hk_sync_end_cs_if_timestamped) __wrap_hk_sync_end_cs_if_timestamped;
__typeof__(hk_sync_src_dep) __wrap_hk_sync_src_dep;
__typeof__(hk_sync_get_cs_after) __wrap_hk_sync_get_cs_after;
__typeof__(hk_flush_deferred_clears_why) __wrap_hk_flush_deferred_clears_why;
__typeof__(hk_cmd_invalidate_graphics_state)
   __wrap_hk_cmd_invalidate_graphics_state;
__typeof__(hk_cmd_invalidate_compute_state)
   __wrap_hk_cmd_invalidate_compute_state;

static unsigned stamps, submissions;

void
__wrap_hk_sync_write_timestamp(struct hk_cmd_buffer *cmd,
                               VkPipelineStageFlags2 stage,
                               struct agx_timestamp_req req)
{
   assert(cmd->query_wait_ops.size && req.addr);
   assert(stage == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
   stamps++;
}

void
__wrap_hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
}

void
__wrap_hk_sync_end_cs_if_timestamped(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
}

void
__wrap_hk_sync_src_dep(struct hk_cmd_buffer *cmd, uint8_t kinds,
                       struct hk_cs *self,
                       struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   (void)cmd;
   (void)kinds;
   (void)self;
   memset(dep, 0, sizeof(*dep) * HK_SYNC_NUM_SQ);
}

struct hk_cs *
__wrap_hk_sync_get_cs_after(struct hk_cmd_buffer *cmd,
                            const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   (void)cmd;
   (void)dep;
   return NULL;
}

void
__wrap_hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   (void)cmd;
   (void)why;
}

void
__wrap_hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
}

void
__wrap_hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
}

static VkResult
submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct hk_device *dev = (void *)vk_queue->base.device;
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   uint64_t serial;
   /* This assertion fails if the production API entrypoint omits its early
    * registration, even though claim() supports a late fallback. */
   assert(!hk_query_wait_quiescent(&dev->query_wait, &serial));
   assert(list_length(&dev->query_wait.producers) == submissions + 1);
   assert(submit->command_buffer_count == 1);
   struct hk_cmd_buffer *cmd = (void *)submit->command_buffers[0];
   assert(util_dynarray_num_elements(&cmd->query_wait_ops,
                                     struct hk_query_wait_op) == 6);
   struct hk_query_wait_producer *p;
   assert(hk_query_wait_claim(&dev->query_wait, queue, cmd,
                              &cmd->query_wait_ops, &p));
   assert(p);
   hk_query_wait_publish(&dev->query_wait, p, 36 + submissions++, true);
   return VK_SUCCESS;
}

static void
init_cmd(struct hk_cmd_buffer *cmd, struct hk_device *dev)
{
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   list_inithead(&cmd->control_streams);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *physical = calloc(1, sizeof(*physical));
   struct hk_cmd_buffer *primary = calloc(1, sizeof(*primary));
   struct hk_cmd_buffer *secondary = calloc(1, sizeof(*secondary));
   assert(dev && physical && primary && secondary);
   dev->dev.fd = -1;
   dev->dev.chip = AGX_CHIP_G13G;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->vk.alloc = *vk_default_allocator();
   dev->vk.physical = &physical->vk;
   dev->vk.submit_mode = VK_QUEUE_SUBMIT_MODE_IMMEDIATE;
   dev->vk.timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   list_inithead(&dev->vk.queues);
   assert(hk_query_wait_device_init(&dev->query_wait));
   const struct vk_device_dispatch_table dispatch = {
      .CmdResetQueryPool = hk_CmdResetQueryPool,
      .CmdWriteTimestamp2 = hk_CmdWriteTimestamp2,
   };
   dev->vk.command_dispatch_table = &dispatch;
   init_cmd(primary, dev);
   init_cmd(secondary, dev);
   struct vk_command_pool command_pool = {.queue_family_index = 0};
   primary->vk.pool = secondary->vk.pool = &command_pool;
   vk_cmd_queue_init(&secondary->vk.cmd_queue);
   struct agx_va va = {.addr = 0x100000};
   uint64_t reports[4] = {17, 23, 29, 31};
   struct agx_bo bo = {.va = &va, ._map = reports};
   struct hk_query_pool pool = {.bo = &bo, .query_stride = 8};
   vk_object_base_init(&dev->vk, &pool.vk.base, VK_OBJECT_TYPE_QUERY_POOL);
   pool.vk.query_type = VK_QUERY_TYPE_TIMESTAMP;
   pool.vk.query_count = 4;
   pool.wait = hk_query_wait_pool_create(&dev->query_wait, 4);
   assert(pool.wait);
   VkQueryPool handle = hk_query_pool_to_handle(&pool);
   VkCommandBuffer cmd = hk_cmd_buffer_to_handle(primary);

   hk_CmdResetQueryPool(cmd, handle, 0, 1);
   hk_CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, handle,
                         0);
   assert(
      vk_enqueue_cmd_reset_query_pool(&secondary->vk.cmd_queue, handle, 1, 1));
   assert(vk_enqueue_cmd_write_timestamp2(
      &secondary->vk.cmd_queue, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, handle,
      1));
   VkCommandBuffer secondary_handle = hk_cmd_buffer_to_handle(secondary);
   hk_CmdExecuteCommands(cmd, 1, &secondary_handle);
   hk_CmdExecuteCommands(cmd, 1, &secondary_handle);
   assert(stamps == 3);
   unsigned n = 0;
   util_dynarray_foreach(&primary->query_wait_ops, struct hk_query_wait_op,
                         op) {
      assert(op->pool == pool.wait && op->count == 1);
      assert(op->first == (n < 2 ? 0 : 1));
      assert(op->reset == !(n++ & 1));
   }
   assert(n == 6 && reports[0] == 17 && reports[1] == 23);
   uint64_t output[2] = {0};
   assert(hk_GetQueryPoolResults(
             hk_device_to_handle(dev), handle, 0, 1, sizeof(output), output,
             sizeof(output),
             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ==
          VK_SUCCESS);
   assert(output[0] == 17 && output[1] == 1);

   struct hk_queue queue = {0};
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
   };
   assert(vk_queue_init(&queue.vk, &dev->vk, &qi, 0) == VK_SUCCESS);
   queue.vk.driver_submit = submit;
   const VkCommandBufferSubmitInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
   };
   const VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &ci,
   };
   for (unsigned i = 0; i < 2; i++)
      assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &si,
                             VK_NULL_HANDLE) == VK_SUCCESS);
   assert(submissions == 2);
   hk_query_wait_prune(&dev->query_wait, &queue, 36);
   assert(list_length(&dev->query_wait.producers) == 1);
   hk_ResetQueryPool(hk_device_to_handle(dev), handle, 0, 2);
   assert(reports[0] == UINT64_MAX && reports[1] == UINT64_MAX);
   assert(list_is_empty(&dev->query_wait.producers));
   output[1] = 99;
   assert(hk_GetQueryPoolResults(
             hk_device_to_handle(dev), handle, 0, 1, sizeof(output), output,
             sizeof(output),
             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ==
          VK_NOT_READY);
   assert(output[1] == 0);

   vk_queue_finish(&queue.vk);
   hk_query_wait_pool_destroy(pool.wait);
   hk_query_wait_recording_finish(&primary->query_wait_ops);
   hk_query_wait_recording_finish(&secondary->query_wait_ops);
   vk_cmd_queue_finish(&secondary->vk.cmd_queue);
   hk_query_wait_device_finish(&dev->query_wait);
   vk_object_base_finish(&pool.vk.base);
   vk_object_base_finish(&primary->vk.base);
   vk_object_base_finish(&secondary->vk.base);
   vk_object_base_finish(&dev->vk.base);
   free(secondary);
   free(primary);
   free(physical);
   free(dev);
   puts(
      "PASS: real timestamp/reset recording, secondary replay, repeated API submission and host reset/readback; no GPU");
   return 0;
}
