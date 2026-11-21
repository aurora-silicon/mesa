/* SPDX-License-Identifier: MIT */
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_query_pool.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"

#include <errno.h>
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

static unsigned stamps, submissions, resets;
static bool reset_only;
static unsigned expected_live;
static bool fail_submission;
static bool tracking_expected = true;
static bool legacy_clock;
static uint64_t legacy_now, legacy_timeout;
static unsigned legacy_losses;

__typeof__(os_time_get_nano) __real_os_time_get_nano;
__typeof__(os_time_get_nano) __wrap_os_time_get_nano;
__typeof__(os_time_get_absolute_timeout) __real_os_time_get_absolute_timeout;
__typeof__(os_time_get_absolute_timeout) __wrap_os_time_get_absolute_timeout;
__typeof__(_vk_device_set_lost) __wrap__vk_device_set_lost;

int64_t
__wrap_os_time_get_nano(void)
{
   if (legacy_clock)
      return legacy_now += 100000000;
   return __real_os_time_get_nano();
}

int64_t
__wrap_os_time_get_absolute_timeout(uint64_t timeout)
{
   if (legacy_clock) {
      legacy_timeout = timeout;
      return legacy_now + timeout;
   }
   return __real_os_time_get_absolute_timeout(timeout);
}

VkResult
__wrap__vk_device_set_lost(struct vk_device *dev, const char *file, int line,
                           const char *msg, ...)
{
   (void)dev;
   (void)file;
   (void)line;
   (void)msg;
   assert(legacy_clock);
   legacy_losses++;
   return VK_ERROR_DEVICE_LOST;
}

void
__wrap_hk_sync_write_timestamp(struct hk_cmd_buffer *cmd,
                               VkPipelineStageFlags2 stage,
                               struct agx_timestamp_req req)
{
   assert((!tracking_expected || hk_query_wait_has_writers(&cmd->query_wait_ops)) && req.addr);
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
   assert(cmd->state.query_reset_recording);
   assert(cmd->sync.kind_override == HK_SYNC_ALL);
   resets++;
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
   if (!tracking_expected || reset_only) {
      assert(hk_query_wait_quiescent(&dev->query_wait, &serial));
      assert(list_length(&dev->query_wait.producers) == expected_live);
      if (reset_only) {
         struct hk_cmd_buffer *cmd = (void *)submit->command_buffers[0];
         assert(!hk_query_wait_has_writers(&cmd->query_wait_ops));
         struct hk_query_wait_producer *p;
         assert(hk_query_wait_claim(&dev->query_wait, queue, cmd,
                                    &cmd->query_wait_ops, &p,
                                    submit->api_queue));
         assert(!p);
      }
      return VK_SUCCESS;
   }
   /* This assertion fails if the production API entrypoint omits its early
    * registration, even though claim() supports a late fallback. */
   assert(!hk_query_wait_quiescent(&dev->query_wait, &serial));
   assert(list_length(&dev->query_wait.producers) == expected_live);
   assert(submit->command_buffer_count == 1);
   struct hk_cmd_buffer *cmd = (void *)submit->command_buffers[0];
   assert(util_dynarray_num_elements(&cmd->query_wait_ops,
                                     struct hk_query_wait_op) == 6);
   if (fail_submission)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct hk_query_wait_producer *p;
   assert(hk_query_wait_claim(&dev->query_wait, queue, cmd,
                              &cmd->query_wait_ops, &p, submit->api_queue));
   assert(p);
   hk_query_wait_publish(&dev->query_wait, p, 36 + submissions++, true);
   return VK_SUCCESS;
}

/* Fault model after real recording/replay and API registration. Only GPU
 * emission/completion/availability and the clock are mocked; no device opens.
 * The watchdog makes the ce1 sticky-reset failure a finite red result. */
static uint64_t fault_now;

static bool
fault_available(void *data)
{
   return false;
}
static VkResult
fault_status(void *data)
{
   return fault_now >= UINT64_C(3000000000) ? VK_ERROR_UNKNOWN : VK_SUCCESS;
}
static uint64_t
fault_clock(void *data)
{
   return fault_now;
}
static int
fault_changed(void *data, struct hk_query_wait_device *dev, uint64_t deadline)
{
   fault_now = deadline;
   return thrd_timedout;
}
static VkResult
fault_error(void *data, int error)
{
   assert(error == -ETIMEDOUT);
   return VK_ERROR_DEVICE_LOST;
}

static int
probe_ordered_reset(struct hk_query_pool *pool)
{
   const struct hk_query_wait_ops ops = {
      .available = fault_available,
      .status = fault_status,
      .now = fault_clock,
      .changed = fault_changed,
      .error = fault_error,
   };
   bool never;
   VkResult result =
      hk_query_wait_available(pool->wait, 0, false, &never, &ops, NULL);
   printf("reset+timestamp real API wiring, mocked missing write: result=%d "
          "fake_ms=%llu never=%d\n",
          result, (unsigned long long)(fault_now / 1000000), never);
   return result == VK_ERROR_DEVICE_LOST && !never &&
                fault_now == HK_QUERY_WAIT_RETIRED_NS
             ? 0
             : 1;
}

static void
init_cmd(struct hk_cmd_buffer *cmd, struct hk_device *dev)
{
   vk_object_base_init(&dev->vk, &cmd->vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd->vk.state = MESA_VK_COMMAND_BUFFER_STATE_EXECUTABLE;
   list_inithead(&cmd->control_streams);
}

static void
test_loss_capability(struct hk_device *dev)
{
   int32_t mirror = 0;
   uint32_t async = 0;
   for (unsigned mask = 0; mask < 16; mask++) {
      dev->vm_status_mirror.word = mask & 1 ? &mirror : NULL;
      dev->dev.async_errors = mask & 2 ? &async : NULL;
      dev->dev.is_virtio = mask & 4;
      dev->dev.params.features = mask & 8 ? DRM_ASAHI_FEATURE_VM_STATUS : 0;
      bool supported = (mask & 3) || (!(mask & 4) && (mask & 8));
      assert(hk_query_wait_has_loss_status(dev) == supported);
   }
   dev->vm_status_mirror.word = NULL;
   dev->dev.async_errors = NULL;
   dev->dev.is_virtio = false;
   dev->dev.params.features = DRM_ASAHI_FEATURE_VM_STATUS_MIRROR;
   assert(!hk_query_wait_has_loss_status(dev));
   dev->dev.params.features = DRM_ASAHI_FEATURE_VM_STATUS;
}

int
main(int argc, char **argv)
{
   bool probe = argc == 2 && !strcmp(argv[1], "--probe-ordered-reset");
   assert(argc == 1 || probe);
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
   test_loss_capability(dev);
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
   reset_only = true;
   hk_CmdResetQueryPool(cmd, handle, 0, 1);
   assert(resets == 1 && primary->query_wait_ops.size);
   assert(!hk_query_wait_has_writers(&primary->query_wait_ops));
   assert(!primary->state.query_reset_recording && !primary->sync.kind_override);
   for (unsigned i = 0; i < 2; i++)
      assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &si,
                             VK_NULL_HANDLE) == VK_SUCCESS);
   reset_only = false;
   hk_query_wait_recording_finish(&primary->query_wait_ops);

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
   assert(stamps == 3 && resets == 4);
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

   for (unsigned i = 0; i < 2; i++) {
      expected_live = i + 1;
      assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &si,
                             VK_NULL_HANDLE) == VK_SUCCESS);
   }
   assert(submissions == 2);
   if (probe) {
      hk_query_wait_prune(&dev->query_wait, &queue, 37);
      int result = probe_ordered_reset(&pool);
      /* Process exit is bounded, but still exercise normal tracker teardown. */
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
      return result;
   }
   hk_query_wait_prune(&dev->query_wait, &queue, 36);
   assert(list_length(&dev->query_wait.producers) == 1);

   /* A real reset-only entrypoint/submit/replay does not invalidate the
    * earlier accepted writer. GPU emission is replaced above, so model its
    * completed memory write explicitly. This proves the metadata gap, not
    * actual GPU execution or a lost write. */
   struct hk_cmd_buffer *reset_cmd = calloc(1, sizeof(*reset_cmd));
   assert(reset_cmd);
   init_cmd(reset_cmd, dev);
   reset_cmd->vk.pool = &command_pool;
   hk_CmdResetQueryPool(hk_cmd_buffer_to_handle(reset_cmd), handle, 0, 2);
   assert(reset_cmd->query_wait_ops.size);
   assert(!hk_query_wait_has_writers(&reset_cmd->query_wait_ops));
   VkCommandBufferSubmitInfo reset_ci = ci;
   reset_ci.commandBuffer = hk_cmd_buffer_to_handle(reset_cmd);
   VkSubmitInfo2 reset_si = si;
   reset_si.pCommandBufferInfos = &reset_ci;
   reset_only = true;
   expected_live = 1;
   for (unsigned replay = 0; replay < 2; replay++) {
      assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &reset_si,
                             VK_NULL_HANDLE) == VK_SUCCESS);
      reports[0] = reports[1] = UINT64_MAX; /* model reset execution */
      assert(hk_GetQueryPoolResults(
                hk_device_to_handle(dev), handle, 0, 1, sizeof(output), output,
                sizeof(output), VK_QUERY_RESULT_64_BIT |
                                   VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ==
             VK_NOT_READY);
      assert(output[1] == 0 && list_length(&dev->query_wait.producers) == 1);
   }
   reset_only = false;
   hk_query_wait_recording_finish(&reset_cmd->query_wait_ops);
   vk_object_base_finish(&reset_cmd->vk.base);
   free(reset_cmd);
   puts("reset-only API submit/replay retains invalidation ranges without a writer; reset memory is modelled");

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

   fail_submission = true;
   expected_live = 1;
   assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &si,
                          VK_NULL_HANDLE) == VK_ERROR_OUT_OF_HOST_MEMORY);
   assert(list_is_empty(&dev->query_wait.producers));
   assert(!vk_device_is_lost(&dev->vk));

   /* The original finite wait is preserved when no usable loss source is
    * available, without recording or submitting new tracking metadata. */
   hk_query_wait_recording_finish(&primary->query_wait_ops);
   struct hk_query_wait_pool *saved_wait = pool.wait;
   pool.wait = NULL;
   dev->dev.params.features = 0;
   expected_live = 0;
   tracking_expected = false;
   fail_submission = false;
   hk_CmdResetQueryPool(cmd, handle, 0, 1);
   hk_CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, handle,
                         0);
   assert(!primary->query_wait_ops.size);
   assert(hk_QueueSubmit2(vk_queue_to_handle(&queue.vk), 1, &si,
                          VK_NULL_HANDLE) == VK_SUCCESS);
   legacy_clock = true;
   assert(hk_GetQueryPoolResults(
             hk_device_to_handle(dev), handle, 0, 1, sizeof(output), output,
             sizeof(output),
             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
          VK_ERROR_DEVICE_LOST);
   assert(legacy_timeout == UINT64_C(2000000000));
   assert(legacy_now == UINT64_C(2000000000) && legacy_losses == 1);
   legacy_clock = false;
   pool.wait = saved_wait;

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
      "PASS: recording/replay, early registration, readback/reset, failure cleanup and observable-loss/finite-legacy guards; no GPU");
   return 0;
}
