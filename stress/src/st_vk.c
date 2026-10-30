// SPDX-License-Identifier: GPL-2.0-only
/* Vulkan helpers for the stress lane. Unlike the benchmark executors, every call returns its
 * VkResult: a stress test must report a lost device or an allocation failure as a verdict, not
 * exit on it. */
#include "st.h"
#include "st_rq.h"

#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                                      \
   do {                                                                                            \
      VkResult r_ = (x);                                                                           \
      if (r_ != VK_SUCCESS)                                                                        \
         return r_;                                                                                \
   } while (0)

const char *
st_vkr(VkResult r)
{
   switch (r) {
   case VK_SUCCESS: return "VK_SUCCESS";
   case VK_NOT_READY: return "VK_NOT_READY";
   case VK_TIMEOUT: return "VK_TIMEOUT";
   case VK_INCOMPLETE: return "VK_INCOMPLETE";
   case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
   case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
   case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
   case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
   case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
   case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
   case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
   case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
   case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
   case VK_ERROR_NOT_PERMITTED_KHR: return "VK_ERROR_NOT_PERMITTED";
   case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
   case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
   default: return "VkResult(other)";
   }
}

/* ---------------- instance / device ---------------- */

static VkResult
make_instance(VkInstance *inst)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "gpubench-stress",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   return vkCreateInstance(&ci, NULL, inst);
}

static VkResult
pick(VkInstance inst, int index, VkPhysicalDevice *pd)
{
   uint32_t n = 0;
   CK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   if (!n || index < 0 || (uint32_t)index >= n)
      return VK_ERROR_INITIALIZATION_FAILED;
   VkPhysicalDevice *all = calloc(n, sizeof(*all));
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, all);
   if (r == VK_SUCCESS || r == VK_INCOMPLETE) {
      *pd = all[index];
      r = VK_SUCCESS;
   }
   free(all);
   return r;
}

static bool
has_ext(VkPhysicalDevice pd, const char *name)
{
   uint32_t n = 0;
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
   VkExtensionProperties *e = calloc(n ? n : 1, sizeof(*e));
   vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
   bool found = false;
   for (uint32_t i = 0; i < n && !found; i++)
      found = !strcmp(e[i].extensionName, name);
   free(e);
   return found;
}

static void
fill_caps(VkPhysicalDevice pd, struct st_caps *c, uint32_t *qf)
{
   memset(c, 0, sizeof(*c));
   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceMaintenance3Properties m3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
                                                .pNext = &drv};
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3};
   vkGetPhysicalDeviceProperties2(pd, &p2);
   snprintf(c->name, sizeof(c->name), "%s", p2.properties.deviceName);
   snprintf(c->driver, sizeof(c->driver), "%s", drv.driverName);
   snprintf(c->driver_info, sizeof(c->driver_info), "%s", drv.driverInfo);
   c->api = p2.properties.apiVersion;
   c->max_alloc = m3.maxMemoryAllocationSize;

   uint32_t n = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
   VkQueueFamilyProperties *q = calloc(n ? n : 1, sizeof(*q));
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
   *qf = UINT32_MAX;
   for (uint32_t i = 0; i < n; i++) {
      VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
      if ((q[i].queueFlags & need) == need) {
         *qf = i;
         c->queues = q[i].queueCount;
         c->sparse_queue = q[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT;
         break;
      }
   }
   free(q);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
      if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
         c->heap += mp.memoryHeaps[i].size;

   VkPhysicalDeviceRayQueryFeaturesKHR rqf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
   VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
   VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
   bool rq_ext = has_ext(pd, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                 has_ext(pd, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                 has_ext(pd, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
   if (rq_ext) {
      f12.pNext = &asf;
      asf.pNext = &rqf;
   }
   vkGetPhysicalDeviceFeatures2(pd, &f2);
   c->bda = f12.bufferDeviceAddress;
   c->rq = rq_ext && rqf.rayQuery && asf.accelerationStructure && f12.bufferDeviceAddress;
   c->sparse = f2.features.sparseBinding && f2.features.sparseResidencyBuffer && c->sparse_queue;
   c->frag_stores = f2.features.fragmentStoresAndAtomics;
   c->gprio = has_ext(pd, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) || has_ext(pd, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME);
}

VkResult
st_probe(int device_index, struct st_caps *c)
{
   VkInstance inst;
   CK(make_instance(&inst));
   VkPhysicalDevice pd;
   VkResult r = pick(inst, device_index, &pd);
   if (r == VK_SUCCESS) {
      uint32_t qf;
      fill_caps(pd, c, &qf);
      if (qf == UINT32_MAX)
         r = VK_ERROR_FEATURE_NOT_PRESENT;
   }
   vkDestroyInstance(inst, NULL);
   return r;
}

VkResult
st_dev_create(struct st_dev *d, int device_index, const struct st_req *req)
{
   memset(d, 0, sizeof(*d));
   VkResult r = make_instance(&d->inst);
   if (r != VK_SUCCESS) {
      d->inst = VK_NULL_HANDLE;
      return r;
   }
   if ((r = pick(d->inst, device_index, &d->pd)) != VK_SUCCESS)
      goto fail;
   fill_caps(d->pd, &d->caps, &d->qf);
   if (d->qf == UINT32_MAX) {
      r = VK_ERROR_FEATURE_NOT_PRESENT;
      goto fail;
   }
   vkGetPhysicalDeviceProperties(d->pd, &d->props);
   vkGetPhysicalDeviceMemoryProperties(d->pd, &d->mp);

   int nq = req && req->nqueues > 0 ? req->nqueues : 1;
   if (nq > (int)d->caps.queues || nq > ST_MAX_QUEUES) {
      r = VK_ERROR_FEATURE_NOT_PRESENT;
      goto fail;
   }
   d->nq = nq;
   d->bda = req && (req->bda || req->rq) && d->caps.bda;
   d->rq = req && req->rq && d->caps.rq;
   d->sparse = req && req->sparse && d->caps.sparse;
   if ((req && req->rq && !d->rq) || (req && req->bda && !d->bda) || (req && req->sparse && !d->sparse)) {
      r = VK_ERROR_FEATURE_NOT_PRESENT;
      goto fail;
   }

   VkPhysicalDeviceRayQueryFeaturesKHR rqf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
                                              .rayQuery = VK_TRUE};
   VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rqf,
      .accelerationStructure = VK_TRUE};
   VkPhysicalDeviceVulkan13Features e13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE,
                                           .dynamicRendering = VK_TRUE,
                                           .pNext = d->rq ? &asf : NULL};
   VkPhysicalDeviceVulkan12Features e12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                           .pNext = &e13,
                                           .timelineSemaphore = VK_TRUE,
                                           .bufferDeviceAddress = d->bda};
   VkPhysicalDeviceFeatures2 e2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12};
   e2.features.sparseBinding = d->sparse;
   e2.features.sparseResidencyBuffer = d->sparse;
   e2.features.fragmentStoresAndAtomics = d->caps.frag_stores;

   const char *exts[4];
   uint32_t next = 0;
   if (d->rq) {
      exts[next++] = VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
      exts[next++] = VK_KHR_RAY_QUERY_EXTENSION_NAME;
      exts[next++] = VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
   }
   bool want_prio = req && req->gprio && d->caps.gprio;
   if (want_prio)
      exts[next++] = has_ext(d->pd, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) ? VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME
                                                                           : VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;

   /* Global priority applies to every queue of one create info (a family appears once), so a
    * test mixes global levels across devices and relative priorities within one. */
   float prio[ST_MAX_QUEUES];
   for (int i = 0; i < nq; i++)
      prio[i] = req && req->mixed ? (float)(i % 3) / 2.0f : 1.0f;
   VkDeviceQueueGlobalPriorityCreateInfoKHR gp = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
      .globalPriority = want_prio ? (VkQueueGlobalPriorityKHR)req->gprio : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
   };
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = want_prio ? &gp : NULL,
      .queueFamilyIndex = d->qf,
      .queueCount = (uint32_t)nq,
      .pQueuePriorities = prio,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &e2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = next,
      .ppEnabledExtensionNames = exts,
   };
   r = vkCreateDevice(d->pd, &dci, NULL, &d->dev);
   if (r == VK_ERROR_NOT_PERMITTED_KHR && want_prio) {
      /* Levels above medium may need privileges: fall back to the default level. */
      qci.pNext = NULL;
      dci.enabledExtensionCount = next - 1;
      r = vkCreateDevice(d->pd, &dci, NULL, &d->dev);
   }
   if (r != VK_SUCCESS) {
      d->dev = VK_NULL_HANDLE;
      goto fail;
   }
   d->gprio = want_prio && qci.pNext;
   for (int i = 0; i < nq; i++)
      vkGetDeviceQueue(d->dev, d->qf, i, &d->q[i]);

   d->mem_host = d->mem_dev = UINT32_MAX;
   const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < d->mp.memoryTypeCount; i++) {
      VkMemoryPropertyFlags f = d->mp.memoryTypes[i].propertyFlags;
      if ((f & hv) == hv && d->mem_host == UINT32_MAX)
         d->mem_host = i;
      if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && d->mem_dev == UINT32_MAX)
         d->mem_dev = i;
   }
   if (d->mem_dev == UINT32_MAX)
      d->mem_dev = d->mem_host;
   for (int i = 0; i < nq; i++) {
      VkCommandPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
         .queueFamilyIndex = d->qf,
      };
      if ((r = vkCreateCommandPool(d->dev, &pci, NULL, &d->pool[i])) != VK_SUCCESS)
         goto fail;
   }
   if (d->rq && (r = st_rq_load(d)) != VK_SUCCESS)
      goto fail;
   return VK_SUCCESS;
fail:
   st_dev_destroy(d);
   return r;
}

bool
st_dev_destroy(struct st_dev *d)
{
   bool idle = true;
   if (d->dev) {
      /* A device that cannot go idle is left to process exit; destroying it would block. */
      for (int i = 0; i < d->nq; i++)
         if (vkQueueWaitIdle(d->q[i]) != VK_SUCCESS)
            idle = false;
      for (int i = 0; i < d->nq; i++)
         if (d->pool[i])
            vkDestroyCommandPool(d->dev, d->pool[i], NULL);
      free(d->rqf);
      vkDestroyDevice(d->dev, NULL);
   }
   if (d->inst)
      vkDestroyInstance(d->inst, NULL);
   memset(d, 0, sizeof(*d));
   return idle;
}

/* ---------------- resources ---------------- */

static uint32_t
mem_type(struct st_dev *d, uint32_t bits, uint32_t preferred, VkMemoryPropertyFlags need)
{
   if (preferred != UINT32_MAX && (bits & (1u << preferred)) &&
       (d->mp.memoryTypes[preferred].propertyFlags & need) == need)
      return preferred;
   for (uint32_t i = 0; i < d->mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (d->mp.memoryTypes[i].propertyFlags & need) == need)
         return i;
   return UINT32_MAX;
}

VkResult
st_buf_create(struct st_dev *d, VkDeviceSize size, VkBufferUsageFlags usage, bool host, struct st_buf *b)
{
   memset(b, 0, sizeof(*b));
   b->size = size;
   if (d->bda)
      usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
   VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
   CK(vkCreateBuffer(d->dev, &ci, NULL, &b->buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(d->dev, b->buf, &req);
   VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                   .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
   uint32_t mt = host ? mem_type(d, req.memoryTypeBits, d->mem_host,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                      : mem_type(d, req.memoryTypeBits, d->mem_dev, 0);
   VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
   if (mt == UINT32_MAX)
      goto fail;
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = d->bda ? &fi : NULL,
      .allocationSize = req.size,
      .memoryTypeIndex = mt,
   };
   if ((r = vkAllocateMemory(d->dev, &ai, NULL, &b->mem)) != VK_SUCCESS)
      goto fail;
   if ((r = vkBindBufferMemory(d->dev, b->buf, b->mem, 0)) != VK_SUCCESS)
      goto fail;
   if (host && (r = vkMapMemory(d->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map)) != VK_SUCCESS)
      goto fail;
   if (d->bda) {
      VkBufferDeviceAddressInfo ai2 = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b->buf};
      b->addr = vkGetBufferDeviceAddress(d->dev, &ai2);
   }
   return VK_SUCCESS;
fail:
   st_buf_destroy(d, b);
   return r;
}

void
st_buf_destroy(struct st_dev *d, struct st_buf *b)
{
   if (b->map)
      vkUnmapMemory(d->dev, b->mem);
   if (b->buf)
      vkDestroyBuffer(d->dev, b->buf, NULL);
   if (b->mem)
      vkFreeMemory(d->dev, b->mem, NULL);
   memset(b, 0, sizeof(*b));
}

VkResult
st_img_create(struct st_dev *d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, struct st_img *im)
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
   CK(vkCreateImage(d->dev, &ci, NULL, &im->img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(d->dev, im->img, &req);
   VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
   uint32_t mt = mem_type(d, req.memoryTypeBits, d->mem_dev, 0);
   if (mt == UINT32_MAX)
      goto fail;
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                              .memoryTypeIndex = mt};
   if ((r = vkAllocateMemory(d->dev, &ai, NULL, &im->mem)) != VK_SUCCESS)
      goto fail;
   if ((r = vkBindImageMemory(d->dev, im->img, im->mem, 0)) != VK_SUCCESS)
      goto fail;
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = im->img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = fmt,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   if ((r = vkCreateImageView(d->dev, &vi, NULL, &im->view)) != VK_SUCCESS)
      goto fail;
   return VK_SUCCESS;
fail:
   st_img_destroy(d, im);
   return r;
}

void
st_img_destroy(struct st_dev *d, struct st_img *im)
{
   if (im->view)
      vkDestroyImageView(d->dev, im->view, NULL);
   if (im->img)
      vkDestroyImage(d->dev, im->img, NULL);
   if (im->mem)
      vkFreeMemory(d->dev, im->mem, NULL);
   memset(im, 0, sizeof(*im));
}

/* ---------------- pipelines ---------------- */

static VkResult
make_layout(struct st_dev *d, int nbuf, VkShaderStageFlags stages, uint32_t push_size, struct st_pipe *p)
{
   VkDescriptorSetLayoutBinding binds[8];
   for (int i = 0; i < nbuf; i++)
      binds[i] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = stages,
      };
   if (nbuf) {
      VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                             .bindingCount = (uint32_t)nbuf, .pBindings = binds};
      CK(vkCreateDescriptorSetLayout(d->dev, &dci, NULL, &p->dsl));
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbuf};
      VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                        .poolSizeCount = 1, .pPoolSizes = &ps};
      CK(vkCreateDescriptorPool(d->dev, &pci, NULL, &p->dpool));
      VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = p->dpool, .descriptorSetCount = 1,
                                         .pSetLayouts = &p->dsl};
      CK(vkAllocateDescriptorSets(d->dev, &sai, &p->set));
   }
   VkPushConstantRange pr = {stages, 0, push_size};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = nbuf ? 1 : 0,
      .pSetLayouts = &p->dsl,
      .pushConstantRangeCount = push_size ? 1 : 0,
      .pPushConstantRanges = &pr,
   };
   CK(vkCreatePipelineLayout(d->dev, &lci, NULL, &p->layout));
   p->stages = stages;
   return VK_SUCCESS;
}

static VkResult
module(struct st_dev *d, const uint32_t *code, size_t bytes, VkShaderModule *m)
{
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes,
                                  .pCode = code};
   return vkCreateShaderModule(d->dev, &ci, NULL, m);
}

VkResult
st_compute_pipe(struct st_dev *d, const uint32_t *spv, size_t bytes, int nbuf, uint32_t push_size, struct st_pipe *p)
{
   memset(p, 0, sizeof(*p));
   p->bind = VK_PIPELINE_BIND_POINT_COMPUTE;
   VkResult r = make_layout(d, nbuf, VK_SHADER_STAGE_COMPUTE_BIT, push_size, p);
   if (r != VK_SUCCESS)
      return r;
   VkShaderModule m;
   CK(module(d, spv, bytes, &m));
   VkComputePipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
      .layout = p->layout,
   };
   r = vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe);
   vkDestroyShaderModule(d->dev, m, NULL);
   return r;
}

VkResult
st_graphics_pipe(struct st_dev *d, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs, size_t fs_bytes,
                 VkFormat fmt, enum st_blend blend, int nbuf, uint32_t push_size, struct st_pipe *p)
{
   memset(p, 0, sizeof(*p));
   p->bind = VK_PIPELINE_BIND_POINT_GRAPHICS;
   VkResult r = make_layout(d, nbuf, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, push_size, p);
   if (r != VK_SUCCESS)
      return r;
   VkShaderModule mv, mf;
   CK(module(d, vs, vs_bytes, &mv));
   if ((r = module(d, fs, fs_bytes, &mf)) != VK_SUCCESS) {
      vkDestroyShaderModule(d->dev, mv, NULL);
      return r;
   }
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = mf, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                           .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                              .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   if (blend == ST_BLEND_ADD) {
      ba.blendEnable = VK_TRUE;
      ba.srcColorBlendFactor = ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.srcAlphaBlendFactor = ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.colorBlendOp = ba.alphaBlendOp = VK_BLEND_OP_ADD;
   }
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                             .attachmentCount = 1, .pAttachments = &ba};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                          .dynamicStateCount = 2, .pDynamicStates = dyn};
   VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                       .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
   VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
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
   r = vkCreateGraphicsPipelines(d->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p->pipe);
   vkDestroyShaderModule(d->dev, mv, NULL);
   vkDestroyShaderModule(d->dev, mf, NULL);
   return r;
}

void
st_pipe_buffers(struct st_dev *d, struct st_pipe *p, struct st_buf *const *bufs, int n)
{
   VkDescriptorBufferInfo bi[8];
   VkWriteDescriptorSet w[8];
   for (int i = 0; i < n; i++) {
      bi[i] = (VkDescriptorBufferInfo){bufs[i]->buf, 0, VK_WHOLE_SIZE};
      w[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = p->set,
         .dstBinding = (uint32_t)i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi[i],
      };
   }
   vkUpdateDescriptorSets(d->dev, (uint32_t)n, w, 0, NULL);
}

void
st_pipe_destroy(struct st_dev *d, struct st_pipe *p)
{
   if (p->pipe)
      vkDestroyPipeline(d->dev, p->pipe, NULL);
   if (p->layout)
      vkDestroyPipelineLayout(d->dev, p->layout, NULL);
   if (p->dpool)
      vkDestroyDescriptorPool(d->dev, p->dpool, NULL);
   if (p->dsl)
      vkDestroyDescriptorSetLayout(d->dev, p->dsl, NULL);
   memset(p, 0, sizeof(*p));
}

/* ---------------- commands ---------------- */

VkResult
st_cb_alloc(struct st_dev *d, int qi, VkCommandBuffer *cb)
{
   VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                     .commandPool = d->pool[qi], .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                     .commandBufferCount = 1};
   return vkAllocateCommandBuffers(d->dev, &ai, cb);
}

VkResult
st_cb_begin(VkCommandBuffer cb)
{
   CK(vkResetCommandBuffer(cb, 0));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   return vkBeginCommandBuffer(cb, &bi);
}

void
st_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
           VkAccessFlags2 da)
{
   VkMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                         .dstStageMask = dst, .dstAccessMask = da};
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                          .pMemoryBarriers = &b};
   vkCmdPipelineBarrier2(cb, &di);
}

void
st_img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 src,
               VkAccessFlags2 sa, VkPipelineStageFlags2 dst, VkAccessFlags2 da)
{
   VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src,
      .srcAccessMask = sa,
      .dstStageMask = dst,
      .dstAccessMask = da,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
   vkCmdPipelineBarrier2(cb, &di);
}

void
st_viewport(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
   VkViewport v = {(float)x, (float)y, (float)w, (float)h, 0, 1};
   VkRect2D s = {{(int32_t)x, (int32_t)y}, {w, h}};
   vkCmdSetViewport(cb, 0, 1, &v);
   vkCmdSetScissor(cb, 0, 1, &s);
}

void
st_begin_render(VkCommandBuffer cb, const struct st_img *im, VkAttachmentLoadOp load, const VkClearColorValue *clear)
{
   VkRenderingAttachmentInfo ca = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = im->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = load,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   if (clear)
      ca.clearValue.color = *clear;
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {im->w, im->h}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ca,
   };
   vkCmdBeginRendering(cb, &ri);
   st_viewport(cb, 0, 0, im->w, im->h);
}

VkResult
st_submit(VkQueue q, VkCommandBuffer cb, const struct st_sem_op *waits, int nwait, const struct st_sem_op *sigs,
          int nsig, VkFence fence)
{
   VkSemaphoreSubmitInfo w[16], s[16];
   if (nwait > 16 || nsig > 16)
      return VK_ERROR_TOO_MANY_OBJECTS;
   for (int i = 0; i < nwait; i++)
      w[i] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = waits[i].sem,
                                     .value = waits[i].value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   for (int i = 0; i < nsig; i++)
      s[i] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = sigs[i].sem,
                                     .value = sigs[i].value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   VkCommandBufferSubmitInfo c = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
   VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = (uint32_t)nwait,
      .pWaitSemaphoreInfos = w,
      .commandBufferInfoCount = cb ? 1 : 0,
      .pCommandBufferInfos = &c,
      .signalSemaphoreInfoCount = (uint32_t)nsig,
      .pSignalSemaphoreInfos = s,
   };
   return vkQueueSubmit2(q, 1, &si, fence);
}

VkResult
st_timeline(struct st_dev *d, uint64_t initial, VkSemaphore *s)
{
   VkSemaphoreTypeCreateInfo t = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                  .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = initial};
   VkSemaphoreCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &t};
   return vkCreateSemaphore(d->dev, &ci, NULL, s);
}

VkResult
st_binary(struct st_dev *d, VkSemaphore *s)
{
   VkSemaphoreCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   return vkCreateSemaphore(d->dev, &ci, NULL, s);
}

VkResult
st_wait_value(struct st_dev *d, VkSemaphore s, uint64_t v, uint64_t timeout_ns)
{
   VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                             .pSemaphores = &s, .pValues = &v};
   return vkWaitSemaphores(d->dev, &wi, timeout_ns);
}

VkResult
st_run_cb(struct st_dev *d, int qi, VkCommandBuffer cb, uint64_t timeout_ns)
{
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence f;
   CK(vkCreateFence(d->dev, &fci, NULL, &f));
   VkResult r = st_submit(d->q[qi], cb, NULL, 0, NULL, 0, f);
   if (r == VK_SUCCESS)
      r = vkWaitForFences(d->dev, 1, &f, VK_TRUE, timeout_ns);
   if (r != VK_TIMEOUT)
      vkDestroyFence(d->dev, f, NULL);
   return r;
}
