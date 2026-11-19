/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Buffer device address capture/replay without a second mapping: a
 * capture/replay buffer takes its memory's address plus the bind offset,
 * memory allocated again on another device at its captured address gets it
 * back and so does the buffer, and an address still in use is refused with
 * VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS. With VK_EXT_buffer_device_address,
 * whose memory need not be capture/replay memory, the buffer's own address
 * replays. Nothing executes on the GPU, so the test also runs under the
 * no-op DRM shim.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "FAIL: %s: %d\n", #call, r_);                         \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

#define EXPECT(cond)                                                           \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL: %s\n", #cond);                                 \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static const VkDeviceSize offset = 64 * 1024, size = 256 * 1024;

static VkDevice
make_device(VkPhysicalDevice pdev, bool ext)
{
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .bufferDeviceAddress = VK_TRUE,
      .bufferDeviceAddressCaptureReplay = VK_TRUE,
   };
   const VkPhysicalDeviceBufferDeviceAddressFeaturesEXT fext = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
      .bufferDeviceAddress = VK_TRUE,
      .bufferDeviceAddressCaptureReplay = VK_TRUE,
   };
   const char *ext_name = VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = ext ? (const void *)&fext : &f12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = ext,
      .ppEnabledExtensionNames = &ext_name,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   return dev;
}

static VkResult
alloc(VkDevice dev, uint64_t replay_addr, VkDeviceMemory *mem)
{
   const VkMemoryOpaqueCaptureAddressAllocateInfo capture = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
      .opaqueCaptureAddress = replay_addr,
   };
   const VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = &capture,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT |
               VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
   };
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags,
      .allocationSize = size,
   };
   return vkAllocateMemory(dev, &info, NULL, mem);
}

/* A capture/replay buffer bound at `offset` in `mem`; returns its address. */
static uint64_t
bind_buffer(VkDevice dev, VkDeviceMemory mem, uint64_t replay_addr,
            VkBuffer *buf)
{
   const VkBufferOpaqueCaptureAddressCreateInfo capture = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
      .opaqueCaptureAddress = replay_addr,
   };
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &capture,
      .flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
      .size = 4096,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
   };
   CHECK(vkCreateBuffer(dev, &info, NULL, buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, *buf, &req);
   EXPECT((req.memoryTypeBits & 1) && offset % req.alignment == 0);
   CHECK(vkBindBufferMemory(dev, *buf, mem, offset));

   const VkBufferDeviceAddressInfo addr_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = *buf,
   };
   const uint64_t addr = vkGetBufferDeviceAddress(dev, &addr_info);
   EXPECT(vkGetBufferOpaqueCaptureAddress(dev, &addr_info) == addr);
   return addr;
}

static uint64_t
memory_addr(VkDevice dev, VkDeviceMemory mem)
{
   const VkDeviceMemoryOpaqueCaptureAddressInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO,
      .memory = mem,
   };
   return vkGetDeviceMemoryOpaqueCaptureAddress(dev, &info);
}

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   /* Capture. */
   VkDevice dev = make_device(pdev, false);
   VkDeviceMemory mem, taken;
   VkBuffer buf;
   CHECK(alloc(dev, 0, &mem));
   const uint64_t mem_addr = memory_addr(dev, mem);
   const uint64_t buf_addr = bind_buffer(dev, mem, 0, &buf);
   EXPECT(mem_addr != 0 && buf_addr == mem_addr + offset);
   EXPECT(alloc(dev, mem_addr, &taken) ==
          VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   vkDestroyBuffer(dev, buf, NULL);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyDevice(dev, NULL);

   /* Replay, on a new device. */
   dev = make_device(pdev, false);
   CHECK(alloc(dev, mem_addr, &mem));
   EXPECT(memory_addr(dev, mem) == mem_addr);
   EXPECT(bind_buffer(dev, mem, buf_addr, &buf) == buf_addr);
   vkDestroyBuffer(dev, buf, NULL);
   vkFreeMemory(dev, mem, NULL);

   /* Freed, the address is free again. */
   CHECK(alloc(dev, mem_addr, &mem));
   vkFreeMemory(dev, mem, NULL);
   vkDestroyDevice(dev, NULL);

   /* VK_EXT_buffer_device_address: ordinary memory, the buffer replays its
    * own address. The replay device allocates other memory first, so that
    * ordinary addresses differ from the capture's. */
   const VkMemoryAllocateInfo plain = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
   };
   uint64_t ext_addr = 0;
   for (unsigned replay = 0; replay < 2; replay++) {
      dev = make_device(pdev, true);
      PFN_vkGetBufferDeviceAddressEXT get_addr =
         (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(
            dev, "vkGetBufferDeviceAddressEXT");
      EXPECT(get_addr != NULL);
      VkDeviceMemory other = VK_NULL_HANDLE;
      if (replay)
         CHECK(vkAllocateMemory(dev, &plain, NULL, &other));
      CHECK(vkAllocateMemory(dev, &plain, NULL, &mem));
      const VkBufferDeviceAddressCreateInfoEXT replay_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT,
         .deviceAddress = ext_addr,
      };
      const VkBufferCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .pNext = &replay_info,
         .flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
         .size = 4096,
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      };
      CHECK(vkCreateBuffer(dev, &info, NULL, &buf));
      CHECK(vkBindBufferMemory(dev, buf, mem, offset));
      const VkBufferDeviceAddressInfo addr_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
         .buffer = buf,
      };
      const uint64_t addr = get_addr(dev, &addr_info);
      EXPECT(addr != 0 && (!replay || addr == ext_addr));
      ext_addr = addr;
      vkDestroyBuffer(dev, buf, NULL);
      vkFreeMemory(dev, mem, NULL);
      vkFreeMemory(dev, other, NULL);
      vkDestroyDevice(dev, NULL);
   }

   vkDestroyInstance(instance, NULL);
   printf("PASS\n");
   return 0;
}
