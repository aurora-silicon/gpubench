// SPDX-License-Identifier: GPL-2.0-only
#define _GNU_SOURCE
#include "gb_vk.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GPU_TIMEOUT_NS (10ull * 1000000000ull)

bool gb_vk_want_pipeline_stats = false;

/* ---------------- instance / device ---------------- */

static VkInstance
make_instance(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "gpubench",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst;
   VkResult r = vkCreateInstance(&ci, NULL, &inst);
   if (r != VK_SUCCESS)
      gb_die("vkCreateInstance failed: %d", (int)r);
   return inst;
}

static VkPhysicalDevice
pick_pdev(VkInstance inst, int index)
{
   uint32_t n = 0;
   vkEnumeratePhysicalDevices(inst, &n, NULL);
   if (!n)
      gb_die("no Vulkan devices");
   VkPhysicalDevice *p = calloc(n, sizeof(*p));
   vkEnumeratePhysicalDevices(inst, &n, p);
   if (index < 0 || (uint32_t)index >= n)
      gb_die("device index %d out of range (%u devices)", index, n);
   VkPhysicalDevice d = p[index];
   free(p);
   return d;
}

static uint32_t
pick_family(VkPhysicalDevice pd, uint32_t *count)
{
   uint32_t n = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
   VkQueueFamilyProperties *q = calloc(n, sizeof(*q));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
   for (uint32_t i = 0; i < n; i++) {
      VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
      if ((q[i].queueFlags & need) == need) {
         *count = q[i].queueCount;
         free(q);
         return i;
      }
   }
   gb_die("no graphics+compute queue family");
}

int
gb_vk_probe_queue_count(int device_index)
{
   VkInstance inst = make_instance();
   uint32_t count = 0;
   pick_family(pick_pdev(inst, device_index), &count);
   vkDestroyInstance(inst, NULL);
   return (int)count;
}

int
gb_vk_create(struct gb_vk *vk, int device_index, int nqueues)
{
   memset(vk, 0, sizeof(*vk));
   vk->instance = make_instance();
   vk->pdev = pick_pdev(vk->instance, device_index);
   vk->qfam = pick_family(vk->pdev, &vk->family_queue_count);

   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
   vkGetPhysicalDeviceProperties2(vk->pdev, &p2);
   vk->props = p2.properties;
   snprintf(vk->driver_name, sizeof(vk->driver_name), "%s", drv.driverName);
   snprintf(vk->driver_info, sizeof(vk->driver_info), "%s", drv.driverInfo);
   vk->ts_period_ns = vk->props.limits.timestampPeriod;
   vkGetPhysicalDeviceMemoryProperties(vk->pdev, &vk->memprops);

   uint32_t nq;
   vkGetPhysicalDeviceQueueFamilyProperties(vk->pdev, &nq, NULL);
   VkQueueFamilyProperties *qp = calloc(nq, sizeof(*qp));
   vkGetPhysicalDeviceQueueFamilyProperties(vk->pdev, &nq, qp);
   uint32_t bits = qp[vk->qfam].timestampValidBits;
   vk->ts_mask = bits >= 64 ? ~0ull : bits ? ((1ull << bits) - 1) : 0;
   free(qp);

   if (nqueues > (int)vk->family_queue_count)
      return -1;
   if (nqueues > GB_MAX_QUEUES)
      nqueues = GB_MAX_QUEUES;
   vk->nqueues = nqueues;

   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
   VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &f13};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
   vkGetPhysicalDeviceFeatures2(vk->pdev, &f2);
   if (!f12.timelineSemaphore || !f12.hostQueryReset || !f13.synchronization2 || !f13.dynamicRendering)
      gb_die("device lacks timelineSemaphore/hostQueryReset/synchronization2/dynamicRendering");

   VkPhysicalDeviceVulkan13Features e13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE,
                                           .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceVulkan12Features e12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &e13,
                                           .timelineSemaphore = VK_TRUE,
                                           .hostQueryReset = VK_TRUE};
   VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pef = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
      .pipelineExecutableInfo = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 e2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12};
   const char *exts[1] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
   if (gb_vk_want_pipeline_stats)
      e13.pNext = &pef;

   float prio[GB_MAX_QUEUES];
   for (int i = 0; i < GB_MAX_QUEUES; i++)
      prio[i] = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = vk->qfam,
      .queueCount = (uint32_t)nqueues,
      .pQueuePriorities = prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &e2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = gb_vk_want_pipeline_stats ? 1 : 0,
      .ppEnabledExtensionNames = exts,
   };
   GB_VK_CHECK(vkCreateDevice(vk->pdev, &dci, NULL, &vk->dev));
   for (int i = 0; i < nqueues; i++)
      vkGetDeviceQueue(vk->dev, vk->qfam, i, &vk->queues[i]);

   /* memory types */
   vk->mem_host = vk->mem_dev = UINT32_MAX;
   const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++) {
      VkMemoryPropertyFlags f = vk->memprops.memoryTypes[i].propertyFlags;
      if ((f & hv) == hv && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && vk->mem_host == UINT32_MAX)
         vk->mem_host = i;
      if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && vk->mem_dev == UINT32_MAX)
         vk->mem_dev = i;
   }
   if (vk->mem_host == UINT32_MAX)
      for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++)
         if ((vk->memprops.memoryTypes[i].propertyFlags & hv) == hv) {
            vk->mem_host = i;
            break;
         }
   if (vk->mem_dev == UINT32_MAX)
      vk->mem_dev = vk->mem_host;

   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = vk->qfam,
   };
   GB_VK_CHECK(vkCreateCommandPool(vk->dev, &pci, NULL, &vk->oneshot_pool));
   return 0;
}

void
gb_vk_destroy(struct gb_vk *vk)
{
   if (vk->dev) {
      vkDeviceWaitIdle(vk->dev);
      vkDestroyCommandPool(vk->dev, vk->oneshot_pool, NULL);
      vkDestroyDevice(vk->dev, NULL);
   }
   if (vk->instance)
      vkDestroyInstance(vk->instance, NULL);
   memset(vk, 0, sizeof(*vk));
}

int
gb_vk_print_info(int device_index)
{
   struct gb_vk vk;
   gb_vk_create(&vk, device_index, 1);
   struct gb_json j;
   gb_json_init(&j);
   gb_json_obj_begin(&j, NULL);
   gb_json_str(&j, "schema", "gpubench-info/1");
   gb_json_str(&j, "api", "vulkan");
   gb_json_str(&j, "device_name", vk.props.deviceName);
   gb_json_str(&j, "driver_name", vk.driver_name);
   gb_json_str(&j, "driver_info", vk.driver_info);
   char b[64];
   snprintf(b, sizeof(b), "%u.%u.%u", VK_API_VERSION_MAJOR(vk.props.apiVersion),
            VK_API_VERSION_MINOR(vk.props.apiVersion), VK_API_VERSION_PATCH(vk.props.apiVersion));
   gb_json_str(&j, "api_version", b);
   snprintf(b, sizeof(b), "0x%x", vk.props.driverVersion);
   gb_json_str(&j, "driver_version", b);
   snprintf(b, sizeof(b), "0x%x", vk.props.vendorID);
   gb_json_str(&j, "vendor_id", b);
   snprintf(b, sizeof(b), "0x%x", vk.props.deviceID);
   gb_json_str(&j, "device_id", b);
   gb_json_num(&j, "timestamp_period_ns", vk.ts_period_ns);
   gb_json_int(&j, "queue_count", vk.family_queue_count);
   VkDeviceSize heap = 0;
   for (uint32_t i = 0; i < vk.memprops.memoryHeapCount; i++)
      if (vk.memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
         heap += vk.memprops.memoryHeaps[i].size;
   gb_json_int(&j, "device_local_heap_bytes", (int64_t)heap);
   gb_json_obj_end(&j);
   puts(j.buf);
   gb_json_free(&j);
   gb_vk_destroy(&vk);
   return 0;
}

/* ---------------- resources ---------------- */

static uint32_t
mem_type_for(struct gb_vk *vk, uint32_t bits, uint32_t preferred)
{
   if (bits & (1u << preferred))
      return preferred;
   for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++)
      if (bits & (1u << i))
         return i;
   gb_die("no memory type for bits 0x%x", bits);
}

void
gb_buf_create(struct gb_vk *vk, VkDeviceSize size, VkBufferUsageFlags usage, bool host, struct gb_buf *b)
{
   memset(b, 0, sizeof(*b));
   b->size = size;
   VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   GB_VK_CHECK(vkCreateBuffer(vk->dev, &ci, NULL, &b->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(vk->dev, b->buf, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mem_type_for(vk, req.memoryTypeBits, host ? vk->mem_host : vk->mem_dev),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &b->mem));
   GB_VK_CHECK(vkBindBufferMemory(vk->dev, b->buf, b->mem, 0));
   if (host)
      GB_VK_CHECK(vkMapMemory(vk->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
}

void
gb_buf_destroy(struct gb_vk *vk, struct gb_buf *b)
{
   if (b->map)
      vkUnmapMemory(vk->dev, b->mem);
   vkDestroyBuffer(vk->dev, b->buf, NULL);
   vkFreeMemory(vk->dev, b->mem, NULL);
   memset(b, 0, sizeof(*b));
}

void
gb_img_create(struct gb_vk *vk, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, struct gb_img *im)
{
   memset(im, 0, sizeof(*im));
   im->w = w;
   im->h = h;
   im->fmt = fmt;
   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   GB_VK_CHECK(vkCreateImage(vk->dev, &ci, NULL, &im->img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(vk->dev, im->img, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mem_type_for(vk, req.memoryTypeBits, vk->mem_dev),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &im->mem));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, im->img, im->mem, 0));
   if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) {
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = im->img,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = fmt,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      GB_VK_CHECK(vkCreateImageView(vk->dev, &vi, NULL, &im->view));
   }
}

void
gb_img_destroy(struct gb_vk *vk, struct gb_img *im)
{
   if (im->view)
      vkDestroyImageView(vk->dev, im->view, NULL);
   vkDestroyImage(vk->dev, im->img, NULL);
   vkFreeMemory(vk->dev, im->mem, NULL);
   memset(im, 0, sizeof(*im));
}

void
gb_img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
               VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
               VkAccessFlags2 dst_access)
{
   VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

VkCommandBuffer
gb_oneshot_begin(struct gb_vk *vk)
{
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = vk->oneshot_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   GB_VK_CHECK(vkAllocateCommandBuffers(vk->dev, &ai, &cb));
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   GB_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
   return cb;
}

void
gb_oneshot_end(struct gb_vk *vk, VkCommandBuffer cb)
{
   GB_VK_CHECK(vkEndCommandBuffer(cb));
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
   };
   GB_VK_CHECK(vkQueueSubmit(vk->queues[0], 1, &si, VK_NULL_HANDLE));
   GB_VK_CHECK(vkQueueWaitIdle(vk->queues[0]));
   vkFreeCommandBuffers(vk->dev, vk->oneshot_pool, 1, &cb);
}

/* ---------------- pipelines ---------------- */

VkShaderModule
gb_shader(struct gb_vk *vk, const uint32_t *code, size_t bytes)
{
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes,
      .pCode = code,
   };
   VkShaderModule m;
   GB_VK_CHECK(vkCreateShaderModule(vk->dev, &ci, NULL, &m));
   return m;
}

static void
make_layout(struct gb_vk *vk, int nbuf, VkShaderStageFlags push_stages, uint32_t push_size, bool texture, struct gb_pipe *p)
{
   VkDescriptorSetLayoutBinding binds[8];
   for (int i = 0; i < nbuf; i++)
      binds[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = texture ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = texture ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_COMPUTE_BIT,
      };
   VkDescriptorSetLayoutCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nbuf,
      .pBindings = binds,
   };
   if (nbuf) {
      GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &p->dsl));
      VkDescriptorPoolSize ps = {texture ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nbuf};
      VkDescriptorPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1,
         .poolSizeCount = 1,
         .pPoolSizes = &ps,
      };
      GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &p->dpool));
      VkDescriptorSetAllocateInfo sai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = p->dpool,
         .descriptorSetCount = 1,
         .pSetLayouts = &p->dsl,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, &p->set));
   }
   VkPushConstantRange pr = {push_stages, 0, push_size};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = nbuf ? 1 : 0,
      .pSetLayouts = &p->dsl,
      .pushConstantRangeCount = push_size ? 1 : 0,
      .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &p->layout));
   p->push_stages = push_stages;
   p->push_size = push_size;
}

void
gb_compute_pipe(struct gb_vk *vk, const uint32_t *code, size_t bytes, int nbuf, uint32_t push_size,
                struct gb_pipe *p)
{
   memset(p, 0, sizeof(*p));
   make_layout(vk, nbuf, VK_SHADER_STAGE_COMPUTE_BIT, push_size, false, p);
   VkShaderModule m = gb_shader(vk, code, bytes);
   VkComputePipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = m,
         .pName = "main",
      },
      .layout = p->layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe));
   vkDestroyShaderModule(vk->dev, m, NULL);
}

void
gb_pipe_set_buffers(struct gb_vk *vk, struct gb_pipe *p, struct gb_buf *const *bufs, int n)
{
   VkDescriptorBufferInfo bi[8];
   VkWriteDescriptorSet w[8];
   for (int i = 0; i < n; i++) {
      bi[i] = (VkDescriptorBufferInfo){bufs[i]->buf, 0, VK_WHOLE_SIZE};
      w[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = p->set,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi[i],
      };
   }
   vkUpdateDescriptorSets(vk->dev, n, w, 0, NULL);
}

static void
graphics_pipe(struct gb_vk *vk, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
                 VkFormat fmt, enum gb_blend blend, uint32_t push_size, bool texture, struct gb_pipe *p)
{
   memset(p, 0, sizeof(*p));
   make_layout(vk, texture ? 1 : 0, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, push_size, texture, p);
   VkShaderModule mv = gb_shader(vk, vs, vs_bytes), mf = gb_shader(vk, fs, fs_bytes);
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = mv,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = mf,
       .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState ba = {
      .colorWriteMask = 0xf,
   };
   if (blend == GB_BLEND_ALPHA) {
      ba.blendEnable = VK_TRUE;
      ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      ba.colorBlendOp = VK_BLEND_OP_ADD;
      ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      ba.alphaBlendOp = VK_BLEND_OP_ADD;
   }
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dyn,
   };
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &fmt,
   };
   VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
      .flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
      .stageCount = 2,
      .pStages = st,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = p->layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
}

void
gb_graphics_pipe(struct gb_vk *vk, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
                 VkFormat fmt, enum gb_blend blend, uint32_t push_size, struct gb_pipe *p)
{
   graphics_pipe(vk, vs, vs_bytes, fs, fs_bytes, fmt, blend, push_size, false, p);
}

void
gb_texture_pipe(struct gb_vk *vk, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
                VkFormat fmt, struct gb_pipe *p)
{
   graphics_pipe(vk, vs, vs_bytes, fs, fs_bytes, fmt, GB_BLEND_ALPHA, 16, true, p);
}

void
gb_pipe_destroy(struct gb_vk *vk, struct gb_pipe *p)
{
   if (p->pipe)
      vkDestroyPipeline(vk->dev, p->pipe, NULL);
   if (p->layout)
      vkDestroyPipelineLayout(vk->dev, p->layout, NULL);
   if (p->dpool)
      vkDestroyDescriptorPool(vk->dev, p->dpool, NULL);
   if (p->dsl)
      vkDestroyDescriptorSetLayout(vk->dev, p->dsl, NULL);
   memset(p, 0, sizeof(*p));
}

void
gb_begin_render(VkCommandBuffer cb, const struct gb_img *im, const float clear[4])
{
   VkRenderingAttachmentInfo ca = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = im->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {clear[0], clear[1], clear[2], clear[3]},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {im->w, im->h}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ca,
   };
   vkCmdBeginRendering(cb, &ri);
   VkViewport v = {0, 0, (float)im->w, (float)im->h, 0, 1};
   VkRect2D s = {{0, 0}, {im->w, im->h}};
   vkCmdSetViewport(cb, 0, 1, &v);
   vkCmdSetScissor(cb, 0, 1, &s);
}

/* ---------------- streams ---------------- */

void
gb_stream_init(struct gb_stream *s, struct gb_vk *vk, int queue_index, int depth, gb_record_fn record,
               gb_done_fn done, void *user)
{
   memset(s, 0, sizeof(*s));
   s->vk = vk;
   s->queue = vk->queues[queue_index];
   s->depth = depth;
   s->record = record;
   s->done = done;
   s->user = user;
   s->pools = calloc(depth, sizeof(VkCommandPool));
   s->cbs = calloc(depth, sizeof(VkCommandBuffer));
   s->submit_ns = calloc(depth, sizeof(uint64_t));
   for (int i = 0; i < depth; i++) {
      VkCommandPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
         .queueFamilyIndex = vk->qfam,
      };
      GB_VK_CHECK(vkCreateCommandPool(vk->dev, &pci, NULL, &s->pools[i]));
      VkCommandBufferAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = s->pools[i],
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = 1,
      };
      GB_VK_CHECK(vkAllocateCommandBuffers(vk->dev, &ai, &s->cbs[i]));
   }
   VkSemaphoreTypeCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
   };
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
   GB_VK_CHECK(vkCreateSemaphore(vk->dev, &sci, NULL, &s->tl));
   if (vk->ts_mask) {
      VkQueryPoolCreateInfo qci = {
         .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType = VK_QUERY_TYPE_TIMESTAMP,
         .queryCount = 2 * depth,
      };
      GB_VK_CHECK(vkCreateQueryPool(vk->dev, &qci, NULL, &s->qp));
      vkResetQueryPool(vk->dev, s->qp, 0, 2 * depth);
   }
}

void
gb_stream_finish(struct gb_stream *s)
{
   VkSemaphoreWaitInfo wi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &s->tl,
      .pValues = &s->value,
   };
   vkWaitSemaphores(s->vk->dev, &wi, GPU_TIMEOUT_NS);
   for (int i = 0; i < s->depth; i++)
      vkDestroyCommandPool(s->vk->dev, s->pools[i], NULL);
   vkDestroySemaphore(s->vk->dev, s->tl, NULL);
   if (s->qp)
      vkDestroyQueryPool(s->vk->dev, s->qp, NULL);
   free(s->pools);
   free(s->cbs);
   free(s->submit_ns);
   memset(s, 0, sizeof(*s));
}

static void
record_slot(struct gb_stream *s, int slot, uint64_t k, bool timed)
{
   VkCommandBuffer cb = s->cbs[slot];
   vkResetCommandPool(s->vk->dev, s->pools[slot], 0);
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   vkBeginCommandBuffer(cb, &bi);
   if (timed)
      vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s->qp, 2 * slot);
   s->record(s->user, cb, k);
   if (timed)
      vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, 2 * slot + 1);
   vkEndCommandBuffer(cb);
}

static VkResult
submit_slot(struct gb_stream *s, int slot, uint64_t value)
{
   VkCommandBufferSubmitInfo cbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = s->cbs[slot],
   };
   VkSemaphoreSubmitInfo sig = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = s->tl,
      .value = value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cbi,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &sig,
   };
   return vkQueueSubmit2(s->queue, 1, &si, VK_NULL_HANDLE);
}

static int
wait_value(struct gb_stream *s, uint64_t value)
{
   VkSemaphoreWaitInfo wi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &s->tl,
      .pValues = &value,
   };
   VkResult r = vkWaitSemaphores(s->vk->dev, &wi, GPU_TIMEOUT_NS);
   if (r != VK_SUCCESS) {
      gb_log("timeline wait for %llu failed: %d", (unsigned long long)value, (int)r);
      s->failed = true;
      return -1;
   }
   return 0;
}

double
gb_stream_once(struct gb_stream *s)
{
   uint64_t v = s->value + 1;
   record_slot(s, 0, v, false);
   uint64_t t0 = gb_now_ns();
   if (submit_slot(s, 0, v) != VK_SUCCESS)
      return -1;
   if (wait_value(s, v))
      return -1;
   uint64_t t1 = gb_now_ns();
   s->value = v;
   if (s->done)
      s->done(s->user, v, 0);
   return (t1 - t0) / 1e3;
}

int
gb_stream_run(struct gb_stream *s, bool timed, uint64_t t0, uint64_t warmup_ns, int nwin, uint64_t win_ns,
              struct gb_pass *out)
{
   memset(out, 0, sizeof(*out));
   out->nwin = nwin;
   out->win_s = win_ns / 1e9;
   out->timed = timed = timed && s->qp;
   const uint64_t tm0 = t0 + warmup_ns, t_end = tm0 + (uint64_t)nwin * win_ns;
   const uint64_t base = s->value;
   const int depth = s->depth;
   uint64_t k = 0, done = 0, last_end = 0;
   bool have_last = false;

   gb_sleep_until_ns(t0);
   for (;;) {
      uint64_t now = gb_now_ns();
      bool full = k - done >= (uint64_t)depth;
      if (!full && now >= t_end)
         break;
      if (full || (done < k && now >= t_end)) {
         /* complete the oldest submission */
         int slot = done % depth;
         if (wait_value(s, base + done + 1))
            break;
         uint64_t t = gb_now_ns();
         bool measured = t >= tm0 && t < t_end;
         if (measured) {
            gb_pass_count(out, t, tm0, win_ns);
            gb_samples_push(&out->latency, (t - s->submit_ns[slot]) / 1e3);
         }
         if (timed) {
            uint64_t ts[2];
            if (vkGetQueryPoolResults(s->vk->dev, s->qp, 2 * slot, 2, sizeof(ts), ts, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
               uint64_t a = ts[0] & s->vk->ts_mask, b = ts[1] & s->vk->ts_mask;
               if (measured) {
                  /* A driver may write the top-of-pipe stamp as soon as the
                   * queue reaches it, before earlier submissions finish, so
                   * b - a would include their work at depth > 1. Count this
                   * submission from the later of its stamp and the previous
                   * end. gpu_gap_us keeps the raw difference. */
                  uint64_t start = have_last && last_end > a ? last_end : a;
                  gb_samples_push(&out->gpu, b > start ? (double)(b - start) * s->vk->ts_period_ns / 1e3 : 0);
                  if (have_last)
                     gb_samples_push(&out->gap, ((double)a - (double)last_end) * s->vk->ts_period_ns / 1e3);
               }
               last_end = b;
               have_last = true;
            }
            vkResetQueryPool(s->vk->dev, s->qp, 2 * slot, 2);
         }
         if (s->done)
            s->done(s->user, base + done + 1, slot);
         done++;
         continue;
      }
      int slot = k % depth;
      uint64_t r0 = gb_now_ns();
      record_slot(s, slot, base + k + 1, timed);
      uint64_t r1 = gb_now_ns();
      s->submit_ns[slot] = r1;
      VkResult vr = submit_slot(s, slot, base + k + 1);
      uint64_t r2 = gb_now_ns();
      if (vr != VK_SUCCESS) {
         gb_log("vkQueueSubmit2 failed: %d", (int)vr);
         s->failed = true;
         break;
      }
      if (r0 >= tm0 && r0 < t_end) {
         gb_samples_push(&out->record_cpu, (r1 - r0) / 1e3);
         gb_samples_push(&out->submit_cpu, (r2 - r1) / 1e3);
      }
      k++;
   }
   /* drain whatever is still in flight (not counted) */
   if (!s->failed && k > done)
      wait_value(s, base + k);
   if (timed && k > done)
      vkResetQueryPool(s->vk->dev, s->qp, 0, 2 * depth);
   if (s->done)
      for (; done < k; done++)
         s->done(s->user, base + done + 1, done % depth);
   s->value = base + k;
   for (int i = 0; i < nwin; i++)
      out->submits += out->win_count[i];
   out->failed = s->failed;
   return s->failed ? -1 : 0;
}

/* ---------------- stream test driver ---------------- */

struct thr {
   struct gb_stream *s;
   bool timed;
   uint64_t t0, warm, win;
   int nwin;
   struct gb_pass pass;
   pthread_t th;
};

static void *
thr_main(void *p)
{
   struct thr *t = p;
   gb_stream_run(t->s, t->timed, t->t0, t->warm, t->nwin, t->win, &t->pass);
   return NULL;
}

static void
run_threads(struct gb_stream *streams, int n, bool timed, uint64_t warm, int nwin, uint64_t win,
            struct gb_pass *out)
{
   struct thr *t = calloc(n, sizeof(*t));
   uint64_t t0 = gb_now_ns() + 20000000ull;
   for (int i = 0; i < n; i++) {
      t[i] = (struct thr){.s = &streams[i], .timed = timed, .t0 = t0, .warm = warm, .win = win, .nwin = nwin};
      pthread_create(&t[i].th, NULL, thr_main, &t[i]);
   }
   memset(out, 0, sizeof(*out));
   for (int i = 0; i < n; i++) {
      pthread_join(t[i].th, NULL);
      gb_pass_merge(out, &t[i].pass);
      gb_pass_free(&t[i].pass);
   }
   free(t);
}

static void
finish_result(const struct gb_opts *o, const struct gb_stream_test *t, enum gb_mode mode, int depth,
              struct gb_pass *u, struct gb_pass *ti, double first_us, int units)
{
   struct gb_result r;
   gb_result_begin(&r, o, mode, depth, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (u->failed || (ti && ti->failed)) {
      gb_result_finish(&r, "fail", NULL, "GPU wait timed out or submit failed");
      return;
   }
   if (!u->submits) {
      gb_result_finish(&r, "fail", NULL, "no completions in the measured windows");
      return;
   }
   gb_result_stream_metrics(&r, u, ti);
   if (first_us >= 0)
      gb_result_metric1(&r, "first_us", "us", true, first_us);
   if (t->derive) {
      struct gb_samples us = {0};
      gb_pass_rate(u, &us);
      t->derive(o, &r, &us);
      gb_samples_free(&us);
   }
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "submits", (int64_t)u->submits);
   if (mode != GB_MODE_QUEUE)
      gb_json_int(&r.j, mode == GB_MODE_PROCS ? "procs" : "queues", units);
   gb_json_obj_end(&r.j);
   gb_result_finish(&r, "ok", "us_per_submit", NULL);
}

static uint64_t
timed_warm(const struct gb_opts *o)
{
   uint64_t w = (uint64_t)o->warmup_ms * 1000000ull;
   return w < 100000000ull ? w : 100000000ull;
}

static int
run_child(const struct gb_opts *o, const struct gb_stream_test *t)
{
   struct gb_vk vk;
   if (gb_vk_create(&vk, o->device, 1))
      return 1;
   char err[256] = "";
   void *st = t->setup(&vk, o, err, sizeof(err));
   if (!st) {
      gb_log("child setup failed: %s", err);
      return 1;
   }
   struct gb_stream s;
   gb_stream_init(&s, &vk, 0, 1, t->record, t->done, st);
   struct gb_pass prime;
   gb_stream_run(&s, false, gb_now_ns(), 0, 1, 30000000ull, &prime); /* build contexts before READY */
   gb_pass_free(&prime);
   gb_child_ready();
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   for (int pass = 0;; pass++) {
      uint64_t t0 = gb_child_recv_u64();
      if (!t0)
         break;
      bool timed = pass == 1;
      struct gb_pass p;
      gb_stream_run(&s, timed, t0, timed ? timed_warm(o) : W, timed ? o->timed_runs : o->runs, R, &p);
      gb_pass_put(1, &p);
      gb_pass_free(&p);
   }
   gb_stream_finish(&s);
   t->teardown(&vk, st);
   gb_vk_destroy(&vk);
   return 0;
}

static void
run_procs(const struct gb_opts *o, const struct gb_stream_test *t, const int *depths, int nd)
{
   for (int di = 0; di < nd; di++) {
      int n = depths[di];
      struct gb_child *ch = calloc(n, sizeof(*ch));
      if (gb_spawn_children(o, n, ch)) {
         gb_result_simple(o, GB_MODE_PROCS, n, NULL, "fail", "child process failed to start or set up");
         for (int i = 0; i < n; i++)
            if (ch[i].pid)
               gb_child_reap(&ch[i]);
         free(ch);
         continue;
      }
      struct gb_pass u = {0}, ti = {0};
      bool bad = false;
      const bool do_timed = o->timestamps && !t->no_timed;
      for (int pass = 0; pass < (do_timed ? 2 : 1); pass++) {
         uint64_t t0 = gb_now_ns() + 50000000ull;
         for (int i = 0; i < n; i++)
            gb_child_send_u64(&ch[i], t0);
         for (int i = 0; i < n; i++) {
            struct gb_pass p;
            if (gb_pass_get(ch[i].from_child, &p)) {
               bad = true;
               continue;
            }
            gb_pass_merge(pass ? &ti : &u, &p);
            gb_pass_free(&p);
         }
      }
      for (int i = 0; i < n; i++) {
         gb_child_send_u64(&ch[i], 0);
         gb_child_reap(&ch[i]);
      }
      if (bad)
         u.failed = true;
      finish_result(o, t, GB_MODE_PROCS, n, &u, do_timed ? &ti : NULL, -1, n);
      gb_pass_free(&u);
      gb_pass_free(&ti);
      free(ch);
   }
}

int
gb_run_stream_test(const struct gb_opts *o, const struct gb_stream_test *t)
{
   if (!gb_check_params(o, t->known_params)) {
      gb_result_simple(o, o->mode, 0, NULL, "fail", "unknown parameter");
      return 1;
   }
   if (o->child)
      return run_child(o, t);
   const int *depths = o->ndepths ? o->depths : t->default_depths;
   int nd = o->ndepths ? o->ndepths : t->ndefault;
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;

   if (o->mode == GB_MODE_PROCS) {
      run_procs(o, t, depths, nd);
      return 0;
   }

   int need = 1;
   if (o->mode == GB_MODE_QUEUES)
      for (int i = 0; i < nd; i++)
         need = depths[i] > need ? depths[i] : need;
   int have = gb_vk_probe_queue_count(o->device);
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, need < have ? need : have);
   char err[256] = "";
   void *st = t->setup(&vk, o, err, sizeof(err));
   if (!st) {
      for (int i = 0; i < nd; i++)
         gb_result_simple(o, o->mode, depths[i], NULL, "skip", err[0] ? err : "setup failed");
      gb_vk_destroy(&vk);
      return 0;
   }
   bool first = t->report_first;
   const bool do_timed = o->timestamps && !t->no_timed;
   for (int di = 0; di < nd; di++) {
      int n = depths[di];
      struct gb_pass u = {0}, ti = {0};
      double first_us = -1;
      if (o->mode == GB_MODE_QUEUE) {
         struct gb_stream s;
         gb_stream_init(&s, &vk, 0, n, t->record, t->done, st);
         if (first) {
            first_us = gb_stream_once(&s);
            first = false;
         }
         gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &u);
         if (do_timed && !s.failed)
            gb_stream_run(&s, true, gb_now_ns(), timed_warm(o), o->timed_runs, R, &ti);
         gb_stream_finish(&s);
      } else {
         if (n > vk.nqueues) {
            char why[96];
            snprintf(why, sizeof(why), "device has %d queue(s)", (int)vk.family_queue_count);
            gb_result_simple(o, o->mode, n, NULL, "skip", why);
            continue;
         }
         struct gb_stream *ss = calloc(n, sizeof(*ss));
         for (int i = 0; i < n; i++)
            gb_stream_init(&ss[i], &vk, i, 1, t->record, t->done, st);
         run_threads(ss, n, false, W, o->runs, R, &u);
         if (do_timed && !u.failed)
            run_threads(ss, n, true, timed_warm(o), o->timed_runs, R, &ti);
         for (int i = 0; i < n; i++)
            gb_stream_finish(&ss[i]);
         free(ss);
      }
      finish_result(o, t, o->mode, n, &u, do_timed ? &ti : NULL, first_us, n);
      gb_pass_free(&u);
      gb_pass_free(&ti);
   }
   t->teardown(&vk, st);
   gb_vk_destroy(&vk);
   return 0;
}
