/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>

#include "drm-uapi/asahi_drm.h"
#include "hk-query-wait-bench-profile.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult result = (call);                                                \
      if (result != VK_SUCCESS) {                                              \
         fprintf(stderr, "%s returned %d\n", #call, result);                   \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static void
require(bool ok, const char *message)
{
   if (!ok) {
      fprintf(stderr, "%s\n", message);
      exit(1);
   }
}

static uint64_t
ns(clockid_t clock)
{
   struct timespec ts;
   require(!clock_gettime(clock, &ts), "clock_gettime failed");
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void
json_string(const char *s)
{
   putchar('"');
   for (; *s; s++) {
      unsigned char c = *s;
      if (c == '"' || c == '\\')
         putchar('\\');
      if (c < 32)
         printf("\\u%04x", c);
      else
         putchar(c);
   }
   putchar('"');
}

struct library_check {
   char expected[PATH_MAX];
   char observed[PATH_MAX];
   bool found;
};

static int
loaded_library(struct dl_phdr_info *info, size_t size, void *data)
{
   (void)size;
   struct library_check *check = data;
   char path[PATH_MAX];
   if (realpath(info->dlpi_name, path)) {
      const char *name = strrchr(path, '/');
      const bool exact = !strcmp(path, check->expected);
      if (exact ||
          (!check->found && name && !strcmp(name + 1, "libvulkan_asahi.so")))
         snprintf(check->observed, sizeof(check->observed), "%s", path);
      if (exact)
         check->found = true;
   }
   return 0;
}

static void
kernel_preflight(const VkPhysicalDeviceDrmPropertiesEXT *drm)
{
   require(drm->hasRender, "the benchmark requires a native render node");
   char node[96];
   snprintf(node, sizeof(node), "/dev/char/%lld:%lld",
            (long long)drm->renderMajor, (long long)drm->renderMinor);
   int fd = open(node, O_RDWR | O_CLOEXEC);
   require(fd >= 0, "cannot open the enumerated render node for preflight");
   drmVersionPtr version = drmGetVersion(fd);
   require(version && (!strcmp(version->name, "aurora") ||
                       !strcmp(version->name, "asahi")),
           "the benchmark requires the native Aurora/Asahi kernel driver");
   struct drm_asahi_params_global params = {0};
   struct drm_asahi_get_params get = {
      .pointer = (uintptr_t)&params,
      .size = sizeof(params),
   };
   require(!drmIoctl(fd, DRM_IOCTL_ASAHI_GET_PARAMS, &get),
           "kernel GET_PARAMS failed");
   printf("{\"kind\":\"kernel_contract\",\"driver\":");
   json_string(version->name);
   printf(
      ",\"major\":%d,\"minor\":%d,\"patch\":%d,\"features\":%llu,"
      "\"chip_id\":%u,\"generation\":%u,\"variant\":%u,\"dies\":%u,"
      "\"clusters\":%u,\"usc\":%u,\"hal\":%u,\"timestamp_hz\":%llu,"
      "\"normalized_profile\":\"%s\",\"driver_supported\":%s,\"native_abi_supported\":%s}\n",
      version->version_major, version->version_minor,
      version->version_patchlevel, (unsigned long long)params.features,
      params.chip_id, params.gpu_generation, params.gpu_variant,
      params.num_dies, params.num_clusters_total, params.usc_generation,
      params.gpu_hal_generation,
      (unsigned long long)params.command_timestamp_frequency_hz,
      agx_chip_name(agx_chip_from_params(&params)),
      agx_params_driver_supported(&params) ? "true" : "false",
      agx_params_native_kernel_abi_supported(&params) ? "true" : "false");
   drmFreeVersion(version);
   close(fd);
   /* Deliberately narrower than the driver: this workload's first physical
    * comparison is specified for the M4 Pro and its native HAL200 contract. */
   require(
      hk_query_wait_bench_profile_supported(&params),
      "benchmark preflight differs from the planned M4 Pro contract; review before running");
}

struct release {
   VkDevice device;
   VkSemaphore semaphore;
   pthread_mutex_t mutex;
   pthread_cond_t condition;
   bool entered;
   uint64_t start, delay, signal_at;
   VkResult result;
};

static void *
signal_later(void *data)
{
   struct release *r = data;
   pthread_mutex_lock(&r->mutex);
   while (!r->entered)
      pthread_cond_wait(&r->condition, &r->mutex);
   uint64_t target = r->start + r->delay;
   pthread_mutex_unlock(&r->mutex);
   const struct timespec ts = {target / 1000000000, target % 1000000000};
   int ret;
   do {
      ret = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
   } while (ret == EINTR);
   require(!ret, "clock_nanosleep failed");
   const VkSemaphoreSignalInfo signal = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = r->semaphore,
      .value = 1,
   };
   r->signal_at = ns(CLOCK_MONOTONIC);
   r->result = vkSignalSemaphore(r->device, &signal);
   return NULL;
}

int
main(int argc, char **argv)
{
   bool run = false, probe = false, submit_only = false, reset_only = false;
   bool gpu_reset = false;
   unsigned timestamps = 1;
   const char *library = NULL;
   unsigned delay_ms = 2200;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--help")) {
         puts(
            "hk-query-wait-bench --probe|--run --library ABSOLUTE_PRIVATE_ICD_LIBRARY [--delay-ms 2200] [--timestamps 1..1024] [--submit-only] [--reset-only] [--gpu-reset]\n"
            "Requires one absolute process-local VK_DRIVER_FILES manifest.\n"
            "--probe enumerates hardware; --run submits one command buffer.\n"
            "--submit-only skips WAIT readback and requires --delay-ms 0.\n"
            "--reset-only records GPU resets and requires --submit-only.\n"
            "--gpu-reset records a reset before the timestamp writers.\n"
            "The old driver may report device loss before the host release.\n"
            "Neither hardware mode is a build/test step. Use a separately approved test machine.\n"
            "Output is JSON Lines; each invocation is one fresh-device measurement.");
         return 0;
      } else if (!strcmp(argv[i], "--run")) {
         run = true;
      } else if (!strcmp(argv[i], "--probe")) {
         probe = true;
      } else if (!strcmp(argv[i], "--submit-only")) {
         submit_only = true;
      } else if (!strcmp(argv[i], "--reset-only")) {
         reset_only = true;
      } else if (!strcmp(argv[i], "--gpu-reset")) {
         gpu_reset = true;
      } else if (!strcmp(argv[i], "--timestamps") && i + 1 < argc) {
         char *end;
         unsigned long value = strtoul(argv[++i], &end, 10);
         require(*argv[i] && !*end && value >= 1 && value <= 1024,
                 "timestamp count must be an integer from 1 through 1024");
         timestamps = value;
      } else if (!strcmp(argv[i], "--library") && i + 1 < argc) {
         library = argv[++i];
      } else if (!strcmp(argv[i], "--delay-ms") && i + 1 < argc) {
         char *end;
         unsigned long value = strtoul(argv[++i], &end, 10);
         require(*argv[i] && !*end && value <= 5000,
                 "delay must be an integer from zero through 5000 ms");
         delay_ms = value;
      } else {
         require(false, "unknown/incomplete option; use --help");
      }
   }
   require(run != probe && library && library[0] == '/',
           "choose --probe or --run and an absolute --library; use --help");
   require(!submit_only || delay_ms == 0,
           "--submit-only requires --delay-ms 0");
   require(!reset_only || submit_only,
           "--reset-only requires --submit-only (reset is not a writer)");
   require(!gpu_reset || !reset_only,
           "--gpu-reset and --reset-only are separate workloads");
   const char *icd = getenv("VK_DRIVER_FILES");
   require(icd && icd[0] == '/' && !strchr(icd, ':'),
           "set VK_DRIVER_FILES to one absolute private ICD manifest");
   struct library_check loaded = {0};
   require(realpath(library, loaded.expected),
           "private library does not exist");

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "HK query wait benchmark",
      .applicationVersion = 4,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   require(count == 1, "private ICD must expose exactly one physical device");
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   dl_iterate_phdr(loaded_library, &loaded);

   VkPhysicalDeviceDrmPropertiesEXT drm = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
   };
   VkPhysicalDeviceDriverProperties driver = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
      .pNext = &drm,
   };
   VkPhysicalDeviceProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &driver,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties);
   printf("{\"kind\":\"identity\",\"workload_version\":5,\"vendor_id\":%u,"
          "\"device_id\":%u,\"driver_id\":%u,\"expected_library\":",
          properties.properties.vendorID, properties.properties.deviceID,
          driver.driverID);
   json_string(loaded.expected);
   printf(",\"observed_library\":");
   json_string(loaded.observed);
   printf(",\"expected_library_loaded\":%s,\"driver_name\":",
          loaded.found ? "true" : "false");
   json_string(driver.driverName);
   printf(",\"driver_info\":");
   json_string(driver.driverInfo);
   puts("}");
   fflush(stdout);
   require(loaded.found, "expected private ICD library was not loaded");
   /* HK reports Mesa's Vulkan vendor ID by default, independently of its
    * native Apple GPU/kernel identity checked below. */
   require(properties.properties.vendorID == VK_VENDOR_ID_MESA &&
              driver.driverID == VK_DRIVER_ID_AURORA_BOREALIS,
           "expected Mesa vendor ID and the Aurora Borealis driver");
   require(properties.properties.apiVersion >= VK_API_VERSION_1_3,
           "Vulkan 1.3 is required");
   kernel_preflight(&drm);
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceVulkan12Features f12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &f12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   require(f12.timelineSemaphore && f12.hostQueryReset && f13.synchronization2,
           "timelineSemaphore, hostQueryReset and synchronization2 required");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   require(families != NULL, "out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
   uint32_t family;
   for (family = 0; family < count; family++) {
      if (families[family].timestampValidBits &&
          (families[family].queueFlags &
           (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
         break;
   }
   require(family < count, "no timestamp-capable graphics/compute queue");
   printf("{\"kind\":\"configuration\",\"workload_version\":5,\"device\":");
   json_string(properties.properties.deviceName);
   printf(",\"driver\":");
   json_string(driver.driverInfo);
   printf(",\"library\":");
   json_string(loaded.expected);
   printf(
      ",\"queue_family\":%u,\"timestamp_bits\":%u,\"timestamp_period_ns\":%.9g,"
      "\"timeline\":true,\"host_reset\":true,\"synchronization2\":true}\n",
      family, families[family].timestampValidBits,
      properties.properties.limits.timestampPeriod);
   free(families);
   fflush(stdout);
   if (probe) {
      vkDestroyInstance(instance, NULL);
      return 0;
   }

   f13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = true,
   };
   f12 = (VkPhysicalDeviceVulkan12Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13,
      .timelineSemaphore = true,
      .hostQueryReset = true,
   };
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &f12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qi,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &di, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   const VkQueryPoolCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = timestamps,
   };
   VkQueryPool pool;
   CHECK(vkCreateQueryPool(device, &pi, NULL, &pool));
   vkResetQueryPool(device, pool, 0, timestamps);
   const VkSemaphoreTypeCreateInfo type = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
   };
   const VkSemaphoreCreateInfo si = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &type,
   };
   VkSemaphore semaphore;
   CHECK(vkCreateSemaphore(device, &si, NULL, &semaphore));
   const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool commands;
   CHECK(vkCreateCommandPool(device, &ci, NULL, &commands));
   const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commands,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
   const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &bi));
   if (reset_only || gpu_reset)
      vkCmdResetQueryPool(cmd, pool, 0, timestamps);
   if (!reset_only)
      for (unsigned i = 0; i < timestamps; i++)
         vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, pool, i);
   CHECK(vkEndCommandBuffer(cmd));
   const VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(device, &fi, NULL, &fence));
   const VkSemaphoreSubmitInfo wi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = semaphore,
      .value = 1,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   const VkCommandBufferSubmitInfo cbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
      .deviceMask = 1,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &wi,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cbi,
   };
   uint64_t submit_cpu0 = ns(CLOCK_THREAD_CPUTIME_ID);
   uint64_t submit_wall0 = ns(CLOCK_MONOTONIC);
   CHECK(vkQueueSubmit2(queue, 1, &submit, fence));
   uint64_t submit_wall = ns(CLOCK_MONOTONIC) - submit_wall0;
   uint64_t submit_cpu = ns(CLOCK_THREAD_CPUTIME_ID) - submit_cpu0;
   struct release release = {
      .device = device,
      .semaphore = semaphore,
      .mutex = PTHREAD_MUTEX_INITIALIZER,
      .condition = PTHREAD_COND_INITIALIZER,
      .delay = (uint64_t)delay_ms * 1000000,
   };
   pthread_t thread;
   require(!pthread_create(&thread, NULL, signal_later, &release),
           "release thread creation failed");
   uint64_t result[2] = {0};
   uint64_t cpu0 = ns(CLOCK_THREAD_CPUTIME_ID);
   uint64_t process0 = ns(CLOCK_PROCESS_CPUTIME_ID);
   pthread_mutex_lock(&release.mutex);
   uint64_t start = release.start = ns(CLOCK_MONOTONIC);
   release.entered = true;
   pthread_cond_signal(&release.condition);
   pthread_mutex_unlock(&release.mutex);
   VkResult ret = VK_SUCCESS;
   if (!submit_only)
      ret = vkGetQueryPoolResults(
         device, pool, 0, 1, sizeof(result), result, sizeof(result),
         VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT |
            VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   uint64_t end = ns(CLOCK_MONOTONIC);
   uint64_t process1 = ns(CLOCK_PROCESS_CPUTIME_ID);
   uint64_t cpu1 = ns(CLOCK_THREAD_CPUTIME_ID);
   /* Query zero may be visible while the remaining timestamps are pending.
    * Fence status is observed after the timed reader region, not a timestamp
    * of the availability write. NOT_READY proves a pending tail; SUCCESS does
    * not prove that the fence was already signaled at reader return. */
   VkResult fence_at_return = vkGetFenceStatus(device, fence);
   pthread_join(thread, NULL);
   VkResult retired = vkWaitForFences(device, 1, &fence, true, 5000000000);
   printf(
      "{\"kind\":\"submit_measurement\",\"query_count\":%u,"
      "\"reset_only\":%s,\"gpu_reset\":%s,\"submit_only\":%s,\"submit_wall_ns\":%llu,"
      "\"submit_cpu_ns\":%llu,\"fence_at_return\":%d}\n",
      timestamps, reset_only ? "true" : "false", gpu_reset ? "true" : "false",
      submit_only ? "true" : "false", (unsigned long long)submit_wall,
      (unsigned long long)submit_cpu, fence_at_return);
   printf("{\"kind\":\"measurement\",\"delay_ms\":%u,\"result\":%d,"
          "\"availability\":%llu,\"timestamp\":%llu,\"wall_ns\":%llu,"
          "\"reader_cpu_ns\":%llu,\"process_cpu_ns\":%llu,"
          "\"signal_after_ns\":%llu,\"signal_result\":%d,\"retirement\":%d}\n",
          delay_ms, ret, (unsigned long long)result[1],
          (unsigned long long)result[0], (unsigned long long)(end - start),
          (unsigned long long)(cpu1 - cpu0),
          (unsigned long long)(process1 - process0),
          (unsigned long long)(release.signal_at - start), release.result,
          retired);
   fflush(stdout);
   /* A timeout is not permission to destroy resources still used by the GPU. */
   if (retired != VK_SUCCESS && retired != VK_ERROR_DEVICE_LOST)
      _Exit(2);
   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, commands, NULL);
   vkDestroySemaphore(device, semaphore, NULL);
   vkDestroyQueryPool(device, pool, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   pthread_cond_destroy(&release.condition);
   pthread_mutex_destroy(&release.mutex);
   return ret == VK_SUCCESS && (submit_only || result[1] != 0) &&
                release.result == VK_SUCCESS &&
                retired == VK_SUCCESS
             ? 0
             : 1;
}
