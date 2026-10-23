// SPDX-License-Identifier: GPL-2.0-only
#include "vkutil.h"
#include "../common/json.h"
#include <stdlib.h>
#include <string.h>

int vk_pipeline_markers;

void vk_marker(const char *name)
{
   if (vk_pipeline_markers) {
      fprintf(stderr, "GB_PIPELINE %s\n", name);
      fflush(stderr);
   }
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
   for (uint32_t i = 0; i < n; i++)
      if (!strcmp(e[i].extensionName, name)) return 1;
   return 0;
}

static VkResult make_instance(VkInstance *inst, int window)
{
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "gpubench-rt",
                            .apiVersion = VK_API_VERSION_1_3};
   const char *exts[2] = {"VK_KHR_surface", "VK_KHR_wayland_surface"};
   VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &app,
                              .enabledExtensionCount = window ? 2 : 0,
                              .ppEnabledExtensionNames = exts};
   return vkCreateInstance(&ci, NULL, inst);
}

static VkPhysicalDevice pick(VkInstance inst, int index)
{
   uint32_t n = 0;
   vkEnumeratePhysicalDevices(inst, &n, NULL);
   if (!n || index < 0 || (uint32_t)index >= n) return VK_NULL_HANDLE;
   VkPhysicalDevice *d = calloc(n, sizeof(*d));
   vkEnumeratePhysicalDevices(inst, &n, d);
   VkPhysicalDevice r = d[index];
   free(d);
   return r;
}

static void query_props(vk_ctx *c)
{
   c->asprops = (VkPhysicalDeviceAccelerationStructurePropertiesKHR){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
   c->rtprops = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
   c->driver = (VkPhysicalDeviceDriverProperties){.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   c->driver.pNext = c->has_as ? (void *)&c->asprops : NULL;
   if (c->has_as) c->asprops.pNext = c->has_rtp ? (void *)&c->rtprops : NULL;
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &c->driver};
   vkGetPhysicalDeviceProperties2(c->phys, &p2);
   c->props = p2.properties;
   vkGetPhysicalDeviceMemoryProperties(c->phys, &c->memprops);
}

static void query_exts(vk_ctx *c)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(c->phys, NULL, &n, NULL);
   VkExtensionProperties *e = calloc(n, sizeof(*e));
   vkEnumerateDeviceExtensionProperties(c->phys, NULL, &n, e);
   c->has_as = has_ext(e, n, "VK_KHR_acceleration_structure") && has_ext(e, n, "VK_KHR_deferred_host_operations");
   c->has_rq = has_ext(e, n, "VK_KHR_ray_query");
   c->has_rtp = has_ext(e, n, "VK_KHR_ray_tracing_pipeline");
   c->has_exec_props = has_ext(e, n, "VK_KHR_pipeline_executable_properties");
   free(e);
}

VkResult vk_ctx_init(vk_ctx *c, int device_index, int want_rtp, int want_stats, int window)
{
   memset(c, 0, sizeof(*c));
   VKC(make_instance(&c->inst, window));
   c->phys = pick(c->inst, device_index);
   if (!c->phys) {
      fprintf(stderr, "no Vulkan device %d\n", device_index);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   query_exts(c);
   query_props(c);

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
   VkQueueFamilyProperties *qf = calloc(nq, sizeof(*qf));
   vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qf);
   c->qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq; i++)
      if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
         c->qfam = i;
         break;
      }
   free(qf);
   if (c->qfam == UINT32_MAX) return VK_ERROR_INITIALIZATION_FAILED;

   const char *exts[16];
   uint32_t ne = 0;
   int rt = c->has_as && c->has_rq;
   if (rt) {
      exts[ne++] = "VK_KHR_acceleration_structure";
      exts[ne++] = "VK_KHR_deferred_host_operations";
      exts[ne++] = "VK_KHR_ray_query";
   }
   int rtp = rt && want_rtp && c->has_rtp;
   if (rtp) exts[ne++] = "VK_KHR_ray_tracing_pipeline";
   c->exec_props_enabled = want_stats && c->has_exec_props;
   if (c->exec_props_enabled) exts[ne++] = "VK_KHR_pipeline_executable_properties";
   if (window) exts[ne++] = "VK_KHR_swapchain";

   VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fexec = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
      .pipelineExecutableInfo = VK_TRUE};
   VkPhysicalDeviceRayTracingPipelineFeaturesKHR frtp = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, .rayTracingPipeline = VK_TRUE};
   VkPhysicalDeviceRayQueryFeaturesKHR frq = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
                                              .rayQuery = VK_TRUE};
   VkPhysicalDeviceAccelerationStructureFeaturesKHR fas = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, .accelerationStructure = VK_TRUE};
   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE,
                                           .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &f13,
                                           .bufferDeviceAddress = VK_TRUE,
                                           .hostQueryReset = VK_TRUE,
                                           .timelineSemaphore = VK_TRUE};
   void **tail = &f13.pNext;
   if (rt) {
      *tail = &fas; tail = &fas.pNext;
      *tail = &frq; tail = &frq.pNext;
   }
   if (rtp) { *tail = &frtp; tail = &frtp.pNext; }
   if (c->exec_props_enabled) { *tail = &fexec; tail = &fexec.pNext; }
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = c->qfam,
                                  .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f2,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .enabledExtensionCount = ne,
                             .ppEnabledExtensionNames = exts};
   VKC(vkCreateDevice(c->phys, &dci, NULL, &c->dev));
   vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);
   c->has_rtp = rtp;
   if (!rt) c->has_as = c->has_rq = 0;
#define GP(field, name) c->field = (void *)vkGetDeviceProcAddr(c->dev, name)
   if (rt) {
      GP(CreateAS, "vkCreateAccelerationStructureKHR");
      GP(DestroyAS, "vkDestroyAccelerationStructureKHR");
      GP(GetASBuildSizes, "vkGetAccelerationStructureBuildSizesKHR");
      GP(CmdBuildAS, "vkCmdBuildAccelerationStructuresKHR");
      GP(GetASAddress, "vkGetAccelerationStructureDeviceAddressKHR");
      GP(CmdWriteASProps, "vkCmdWriteAccelerationStructuresPropertiesKHR");
   }
   if (rtp) {
      GP(CreateRTPipelines, "vkCreateRayTracingPipelinesKHR");
      GP(GetRTGroupHandles, "vkGetRayTracingShaderGroupHandlesKHR");
      GP(CmdTraceRays, "vkCmdTraceRaysKHR");
   }
   if (c->exec_props_enabled) {
      GP(GetExecProps, "vkGetPipelineExecutablePropertiesKHR");
      GP(GetExecStats, "vkGetPipelineExecutableStatisticsKHR");
   }
#undef GP
   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                  .queueFamilyIndex = c->qfam};
   VKC(vkCreateCommandPool(c->dev, &pci, NULL, &c->pool));
   return VK_SUCCESS;
}

void vk_ctx_destroy(vk_ctx *c)
{
   if (c->dev) {
      vkDeviceWaitIdle(c->dev);
      vkDestroyCommandPool(c->dev, c->pool, NULL);
      vkDestroyDevice(c->dev, NULL);
   }
   if (c->inst) vkDestroyInstance(c->inst, NULL);
   memset(c, 0, sizeof(*c));
}

void vk_print_info(int device_index)
{
   VkInstance inst;
   rt_json j;
   rt_json_init(&j);
   rt_json_obj(&j, NULL);
   rt_json_str(&j, "api", "vulkan");
   if (make_instance(&inst, 0) != VK_SUCCESS) {
      rt_json_str(&j, "error", "vkCreateInstance failed");
      rt_json_end(&j);
      printf("%s\n", j.buf);
      rt_json_free(&j);
      return;
   }
   vk_ctx c = {.inst = inst};
   c.phys = pick(inst, device_index);
   if (c.phys) {
      query_exts(&c);
      query_props(&c);
      char ver[64];
      rt_json_str(&j, "device", c.props.deviceName);
      rt_json_obj(&j, "driver");
      rt_json_str(&j, "name", c.driver.driverName);
      rt_json_str(&j, "info", c.driver.driverInfo);
      snprintf(ver, sizeof(ver), "%u.%u.%u", VK_API_VERSION_MAJOR(c.props.driverVersion),
               VK_API_VERSION_MINOR(c.props.driverVersion), VK_API_VERSION_PATCH(c.props.driverVersion));
      rt_json_str(&j, "version", ver);
      snprintf(ver, sizeof(ver), "%u.%u.%u", VK_API_VERSION_MAJOR(c.props.apiVersion),
               VK_API_VERSION_MINOR(c.props.apiVersion), VK_API_VERSION_PATCH(c.props.apiVersion));
      rt_json_str(&j, "api_version", ver);
      const char *g = strstr(c.driver.driverInfo, "git-");
      if (g) {
         char sha[16];
         snprintf(sha, sizeof(sha), "%.*s", (int)strcspn(g + 4, ")"), g + 4);
         rt_json_str(&j, "mesa_sha", sha);
      }
      rt_json_end(&j);
      rt_json_obj(&j, "gpu");
      rt_json_str(&j, "name", c.props.deviceName);
      rt_json_int(&j, "vendor_id", c.props.vendorID);
      rt_json_int(&j, "device_id", c.props.deviceID);
      rt_json_end(&j);
      rt_json_obj(&j, "rt");
      rt_json_bool(&j, "acceleration_structure", c.has_as);
      rt_json_bool(&j, "ray_query", c.has_rq);
      rt_json_bool(&j, "ray_tracing_pipeline", c.has_rtp);
      rt_json_bool(&j, "pipeline_executable_properties", c.has_exec_props);
      if (c.has_as) {
         rt_json_int(&j, "max_primitive_count", (int64_t)c.asprops.maxPrimitiveCount);
         rt_json_int(&j, "max_instance_count", (int64_t)c.asprops.maxInstanceCount);
         rt_json_int(&j, "scratch_alignment", c.asprops.minAccelerationStructureScratchOffsetAlignment);
      }
      if (c.has_rtp) {
         rt_json_int(&j, "max_recursion", c.rtprops.maxRayRecursionDepth);
         rt_json_int(&j, "group_handle_size", c.rtprops.shaderGroupHandleSize);
      }
      rt_json_end(&j);
      rt_json_num(&j, "timestamp_period_ns", c.props.limits.timestampPeriod);
   } else {
      rt_json_str(&j, "error", "no such device");
   }
   rt_json_end(&j);
   printf("%s\n", j.buf);
   rt_json_free(&j);
   vkDestroyInstance(inst, NULL);
}

static int32_t mem_type(vk_ctx *c, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
   for (uint32_t i = 0; i < c->memprops.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (c->memprops.memoryTypes[i].propertyFlags & want) == want &&
          !(c->memprops.memoryTypes[i].propertyFlags & avoid))
         return (int32_t)i;
   return -1;
}

static VkResult alloc_mem(vk_ctx *c, VkMemoryRequirements req, int host, int addr, VkDeviceMemory *mem)
{
   int32_t t;
   if (host) {
      t = mem_type(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
      if (t < 0)
         t = mem_type(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
   } else {
      t = mem_type(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      if (t < 0) t = mem_type(c, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
      if (t < 0) t = mem_type(c, req.memoryTypeBits, 0, 0);
   }
   if (t < 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                   .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .pNext = addr ? &fi : NULL,
                              .allocationSize = req.size,
                              .memoryTypeIndex = (uint32_t)t};
   return vkAllocateMemory(c->dev, &ai, NULL, mem);
}

VkResult vk_buf_create(vk_ctx *c, vk_buf *b, VkDeviceSize size, VkBufferUsageFlags usage, int host)
{
   memset(b, 0, sizeof(*b));
   if (!size) size = 16;
   int addr = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
   VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
   VKC(vkCreateBuffer(c->dev, &ci, NULL, &b->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(c->dev, b->buf, &req);
   VKC(alloc_mem(c, req, host, addr, &b->mem));
   VKC(vkBindBufferMemory(c->dev, b->buf, b->mem, 0));
   b->size = size;
   if (host) VKC(vkMapMemory(c->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
   if (addr) {
      VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buf};
      b->addr = vkGetBufferDeviceAddress(c->dev, &ai);
   }
   return VK_SUCCESS;
}

void vk_buf_destroy(vk_ctx *c, vk_buf *b)
{
   if (b->buf) vkDestroyBuffer(c->dev, b->buf, NULL);
   if (b->mem) vkFreeMemory(c->dev, b->mem, NULL);
   memset(b, 0, sizeof(*b));
}

VkResult vk_img_create(vk_ctx *c, vk_img *im, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                       VkImageAspectFlags aspect)
{
   memset(im, 0, sizeof(*im));
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                           .imageType = VK_IMAGE_TYPE_2D,
                           .format = fmt,
                           .extent = {w, h, 1},
                           .mipLevels = 1,
                           .arrayLayers = 1,
                           .samples = VK_SAMPLE_COUNT_1_BIT,
                           .tiling = VK_IMAGE_TILING_OPTIMAL,
                           .usage = usage,
                           .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VKC(vkCreateImage(c->dev, &ci, NULL, &im->img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(c->dev, im->img, &req);
   VKC(alloc_mem(c, req, 0, 0, &im->mem));
   VKC(vkBindImageMemory(c->dev, im->img, im->mem, 0));
   VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                               .image = im->img,
                               .viewType = VK_IMAGE_VIEW_TYPE_2D,
                               .format = fmt,
                               .subresourceRange = {aspect, 0, 1, 0, 1}};
   VKC(vkCreateImageView(c->dev, &vi, NULL, &im->view));
   im->fmt = fmt;
   im->w = w;
   im->h = h;
   return VK_SUCCESS;
}

void vk_img_destroy(vk_ctx *c, vk_img *im)
{
   if (im->view) vkDestroyImageView(c->dev, im->view, NULL);
   if (im->img) vkDestroyImage(c->dev, im->img, NULL);
   if (im->mem) vkFreeMemory(c->dev, im->mem, NULL);
   memset(im, 0, sizeof(*im));
}

VkResult vk_shader(vk_ctx *c, const uint32_t *code, size_t size, VkShaderModule *out)
{
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
   return vkCreateShaderModule(c->dev, &ci, NULL, out);
}

VkResult vk_oneshot_begin(vk_ctx *c, VkCommandBuffer *cb)
{
   VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                     .commandPool = c->pool,
                                     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                     .commandBufferCount = 1};
   VKC(vkAllocateCommandBuffers(c->dev, &ai, cb));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   return vkBeginCommandBuffer(*cb, &bi);
}

VkResult vk_oneshot_end(vk_ctx *c, VkCommandBuffer cb)
{
   VKC(vkEndCommandBuffer(cb));
   VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence f;
   VKC(vkCreateFence(c->dev, &fi, NULL, &f));
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
   VkResult r = vkQueueSubmit(c->queue, 1, &si, f);
   if (r == VK_SUCCESS) r = vkWaitForFences(c->dev, 1, &f, VK_TRUE, 300ull * 1000000000ull);
   vkDestroyFence(c->dev, f, NULL);
   vkFreeCommandBuffers(c->dev, c->pool, 1, &cb);
   return r;
}

void vk_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 srca, VkPipelineStageFlags2 dst,
                VkAccessFlags2 dsta)
{
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                          .srcStageMask = src,
                          .srcAccessMask = srca,
                          .dstStageMask = dst,
                          .dstAccessMask = dsta};
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
   vkCmdPipelineBarrier2(cb, &di);
}

void vk_image_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from,
                      VkImageLayout to, VkPipelineStageFlags2 src, VkAccessFlags2 srca,
                      VkPipelineStageFlags2 dst, VkAccessFlags2 dsta)
{
   VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = src,
                               .srcAccessMask = srca,
                               .dstStageMask = dst,
                               .dstAccessMask = dsta,
                               .oldLayout = from,
                               .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img,
                               .subresourceRange = {aspect, 0, 1, 0, 1}};
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &ib};
   vkCmdPipelineBarrier2(cb, &di);
}

void vk_pipeline_stats_json(vk_ctx *c, VkPipeline p, rt_json *j, const char *key)
{
   if (!c->exec_props_enabled || !p) return;
   VkPipelineInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = p};
   uint32_t n = 0;
   if (c->GetExecProps(c->dev, &pi, &n, NULL) != VK_SUCCESS || !n) return;
   VkPipelineExecutablePropertiesKHR *ep = calloc(n, sizeof(*ep));
   for (uint32_t i = 0; i < n; i++) ep[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
   c->GetExecProps(c->dev, &pi, &n, ep);
   rt_json_obj(j, key);
   for (uint32_t e = 0; e < n; e++) {
      VkPipelineExecutableInfoKHR ei = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                        .pipeline = p,
                                        .executableIndex = e};
      uint32_t ns = 0;
      c->GetExecStats(c->dev, &ei, &ns, NULL);
      VkPipelineExecutableStatisticKHR *st = calloc(ns ? ns : 1, sizeof(*st));
      for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      c->GetExecStats(c->dev, &ei, &ns, st);
      char name[300];
      snprintf(name, sizeof(name), "%u:%s", e, ep[e].name);
      rt_json_obj(j, name);
      for (uint32_t i = 0; i < ns; i++) {
         switch (st[i].format) {
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: rt_json_bool(j, st[i].name, st[i].value.b32); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: rt_json_int(j, st[i].name, st[i].value.i64); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: rt_json_int(j, st[i].name, (int64_t)st[i].value.u64); break;
         default: rt_json_num(j, st[i].name, st[i].value.f64); break;
         }
      }
      rt_json_end(j);
      free(st);
   }
   rt_json_end(j);
   free(ep);
}
