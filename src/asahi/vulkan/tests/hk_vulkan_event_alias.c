/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A command buffer that waits for an event the host sets must take memory
 * scope throughout at vkEndCommandBuffer on G17P and M5: memory that first
 * aliases while it waits is too late to patch, the CDM may have read the
 * records past the wait. A vkSetEvent wait may have no barriers at all, or
 * barriers with no source stage (VUID-vkCmdWaitEvents2-pEvents-03839). Each
 * case records transfer work, the wait and a barrier, ends the command
 * buffer, then creates a sparse buffer so that memory first aliases after
 * recording. The driver's alias trace says how many command buffers were
 * still waiting for that to be patched: none for a host wait, the one
 * recorded for an event the device sets. Nothing
 * executes on the GPU, so the test runs under the no-op DRM shim.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "FAIL: %s: %d\n", #call, r_);                         \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkBuffer dst;
static VkEvent event;

enum wait_kind { WAIT_EMPTY, WAIT_NONE_STAGE, WAIT_HOST_STAGE, WAIT_DEVICE_SET };

static const VkMemoryBarrier2 xfer_barrier = {
   .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
   .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
   .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
};
static const VkMemoryBarrier2 host_barrier = {
   .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
   .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
   .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
   .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
   .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
};

/* An empty first scope also holds only host operations. */
static const VkMemoryBarrier2 none_barrier = {
   .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
   .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
   .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
   .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
};

static VkDependencyInfo
dep(const VkMemoryBarrier2 *b)
{
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = b ? 1 : 0,
      .pMemoryBarriers = b,
   };
}

/* Records and ends the case, then returns the patched= count the trace
 * reports when the first aliasing resource appears. */
static unsigned
run_case(enum wait_kind kind)
{
   VkCommandBuffer cb;
   const VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(cb, &bi));

   vkCmdFillBuffer(cb, dst, 0, 4096, 1);
   const VkDependencyInfo xfer = dep(&xfer_barrier);
   VkDependencyInfo wait = dep(NULL);
   if (kind == WAIT_NONE_STAGE)
      wait = dep(&none_barrier);
   if (kind == WAIT_HOST_STAGE)
      wait = dep(&host_barrier);
   if (kind == WAIT_DEVICE_SET) {
      vkCmdSetEvent2(cb, event, &xfer);
      wait = xfer;
   }
   vkCmdWaitEvents2(cb, 1, &event, &wait);
   vkCmdPipelineBarrier2(cb, &xfer);
   vkCmdFillBuffer(cb, dst, 4096, 4096, 2);
   vkCmdPipelineBarrier2(cb, &xfer);
   vkCmdFillBuffer(cb, dst, 8192, 4096, 3);
   CHECK(vkEndCommandBuffer(cb));

   /* The first aliasing resource, after recording; the trace goes to stderr. */
   fflush(stderr);
   FILE *log = tmpfile();
   const int saved = dup(2);
   if (!log || saved < 0 || dup2(fileno(log), 2) < 0) {
      fprintf(stderr, "FAIL: redirect stderr\n");
      exit(1);
   }
   const VkBufferCreateInfo sbi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
      .size = 1 << 20,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer sparse;
   const VkResult r = vkCreateBuffer(dev, &sbi, NULL, &sparse);
   fflush(stderr);
   dup2(saved, 2);
   close(saved);
   CHECK(r);

   unsigned patched = ~0u;
   char line[256];
   rewind(log);
   while (fgets(line, sizeof(line), log)) {
      if (sscanf(line, "M5_HOST_ALIAS began cause=sparse patched=%u",
                 &patched) == 1)
         break;
   }
   fclose(log);

   if (kind != WAIT_DEVICE_SET)
      CHECK(vkSetEvent(dev, event));
   vkDestroyBuffer(dev, sparse, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cb);
   CHECK(vkResetEvent(dev, event));
   return patched;
}

int
main(void)
{
   setenv("ASAHI_M5_HOST_ALIAS_TRACE", "1", 1);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pdev;
   const VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n == 0) {
      fprintf(stderr, "FAIL: no physical device\n");
      return 1;
   }

   VkPhysicalDeviceFeatures supported;
   vkGetPhysicalDeviceFeatures(pdev, &supported);
   if (!supported.sparseBinding) {
      printf("SKIP: no sparseBinding\n");
      return 77;
   }

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures f = {.sparseBinding = VK_TRUE};
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &f13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qi,
      .pEnabledFeatures = &f,
   };
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);

   const VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   const VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
   CHECK(vkCreateEvent(dev, &eci, NULL, &event));

   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 16384,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &dst));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, dst, &req);
   const VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = __builtin_ctz(req.memoryTypeBits),
   };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, dst, mem, 0));

   static const struct {
      enum wait_kind kind;
      const char *name;
      unsigned patched;
   } cases[] = {
      {WAIT_EMPTY, "host event, no barriers", 0},
      {WAIT_NONE_STAGE, "host event, NONE srcStageMask", 0},
      {WAIT_HOST_STAGE, "host event, HOST in srcStageMask", 0},
      {WAIT_DEVICE_SET, "event set by vkCmdSetEvent2", 1},
   };
   int fails = 0;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const unsigned patched = run_case(cases[i].kind);
      const bool ok = patched == cases[i].patched;
      printf("%s: %s: patched=%d, want %u\n", ok ? "ok" : "FAIL",
             cases[i].name, (int)patched, cases[i].patched);
      fails += !ok;
   }

   vkDestroyBuffer(dev, dst, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyEvent(dev, event, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return fails ? 1 : 0;
}
