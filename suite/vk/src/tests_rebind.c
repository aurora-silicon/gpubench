// SPDX-License-Identifier: GPL-2.0-only
/* Queue rebind & context-churn Vulkan benchmarks:
 *   1. queue.rebind — Multi-process, multi-queue mixed compute+render
 *                     submissions (explicit compute dispatch + graphics draw +
 *                     checked compute and render readback) chained by binary
 *                     semaphores with multiple in-flight batches per queue,
 *                     exercising physical queue rebinding and shared
 *                     compute/render work-state lifetimes.
 *   2. queue.churn  — Single-engine (compute or render) device/queue create ->
 *                     light checked submit -> destroy lifecycle churn,
 *                     isolating context and queue allocation/teardown overhead.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gb_params.h"
#include "gb_vk.h"
#include "shaders.h"
#include "tests.h"

#define REBIND_MAX_WORKERS   8
#define REBIND_MAX_QUEUES    8
#define REBIND_MAX_INFLIGHT  4
#define REBIND_MAX_SAMPLES   2048
#define REBIND_TIMEOUT_NS    (8ull * 1000000000ull)
#define REBIND_LIGHT_DIM     64u
#define REBIND_WORDS         32u
#define REBIND_PIX_W         4u
#define REBIND_PIX_H         4u
#define REBIND_PIX_COUNT     (REBIND_PIX_W * REBIND_PIX_H)
#define REBIND_SLICE_BYTES   256u
#define REBIND_SETUP_MS      15000
#define REBIND_EXTRA_MS      25000

struct shared_inst {
   VkInstance inst;
   VkPhysicalDevice pdev;
   uint32_t qfam;
   uint32_t family_queues;
   uint32_t mem_host;
   uint32_t mem_dev;
   bool has_maintenance4;
   bool ok;
   char err[128];
};

static bool
parse_strict_long(const struct gb_opts *o, const char *key, long def, long *out)
{
   const char *v = gb_param(o, key, NULL);
   if (!v) {
      *out = def;
      return true;
   }
   if (!*v)
      return false;
   errno = 0;
   char *end = NULL;
   long x = strtol(v, &end, 10);
   if (errno == ERANGE || !end || *end != '\0')
      return false;
   *out = x;
   return true;
}

static bool
validate_run_controls(const struct gb_opts *o, char *err, size_t err_sz)
{
   if (o->warmup_ms < 0) {
      snprintf(err, err_sz, "warmup-ms must be >= 0 (got %d)", o->warmup_ms);
      return false;
   }
   if (o->run_ms <= 0) {
      snprintf(err, err_sz, "run-ms must be > 0 (got %d)", o->run_ms);
      return false;
   }
   if (o->runs < 1 || o->runs > GB_MAX_WIN) {
      snprintf(err, err_sz, "runs must be 1..%d (got %d)", GB_MAX_WIN, o->runs);
      return false;
   }
   return true;
}

static bool
shared_inst_init(struct shared_inst *si, int device_index)
{
   memset(si, 0, sizeof(*si));
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "gpubench",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst = VK_NULL_HANDLE;
   VkResult vr = vkCreateInstance(&ici, NULL, &inst);
   if (vr != VK_SUCCESS) {
      snprintf(si->err, sizeof(si->err), "vkCreateInstance failed: %d", (int)vr);
      return false;
   }
   si->inst = inst;

   uint32_t npd = 0;
   vr = vkEnumeratePhysicalDevices(si->inst, &npd, NULL);
   if ((vr != VK_SUCCESS && vr != VK_INCOMPLETE) || !npd ||
       device_index < 0 || (uint32_t)device_index >= npd) {
      snprintf(si->err, sizeof(si->err), "device index %d out of range (%u, vr=%d)",
               device_index, npd, (int)vr);
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   VkPhysicalDevice *pds = calloc(npd, sizeof(*pds));
   if (!pds) {
      snprintf(si->err, sizeof(si->err), "out of host memory");
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   uint32_t got = npd;
   vr = vkEnumeratePhysicalDevices(si->inst, &got, pds);
   if ((vr != VK_SUCCESS && vr != VK_INCOMPLETE) || (uint32_t)device_index >= got) {
      free(pds);
      snprintf(si->err, sizeof(si->err), "vkEnumeratePhysicalDevices failed (%d, count=%u)",
               (int)vr, got);
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   si->pdev = pds[device_index];
   free(pds);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(si->pdev, &nqf, NULL);
   VkQueueFamilyProperties *qfp = nqf ? calloc(nqf, sizeof(*qfp)) : NULL;
   if (!qfp) {
      snprintf(si->err, sizeof(si->err), "no queue families available");
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   vkGetPhysicalDeviceQueueFamilyProperties(si->pdev, &nqf, qfp);
   bool found = false;
   for (uint32_t i = 0; i < nqf; i++) {
      VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
      if ((qfp[i].queueFlags & need) == need && qfp[i].queueCount > 0) {
         if (!found || qfp[i].queueCount > si->family_queues) {
            si->qfam = i;
            si->family_queues = qfp[i].queueCount;
            found = true;
         }
      }
   }
   free(qfp);
   if (!found) {
      snprintf(si->err, sizeof(si->err), "no graphics+compute queue family");
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }

   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
   vkGetPhysicalDeviceFeatures2(si->pdev, &f2);
   if (!f13.synchronization2 || !f13.dynamicRendering) {
      snprintf(si->err, sizeof(si->err), "device lacks synchronization2/dynamicRendering");
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   si->has_maintenance4 = (f13.maintenance4 == VK_TRUE);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(si->pdev, &mp);
   si->mem_host = UINT32_MAX;
   si->mem_dev = UINT32_MAX;
   const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
      VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
      if ((f & hv) == hv && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && si->mem_host == UINT32_MAX)
         si->mem_host = i;
      if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && si->mem_dev == UINT32_MAX)
         si->mem_dev = i;
   }
   if (si->mem_host == UINT32_MAX) {
      for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
         if ((mp.memoryTypes[i].propertyFlags & hv) == hv) {
            si->mem_host = i;
            break;
         }
      }
   }
   if (si->mem_dev == UINT32_MAX)
      si->mem_dev = si->mem_host;
   if (si->mem_host == UINT32_MAX) {
      snprintf(si->err, sizeof(si->err), "no host-visible coherent memory type");
      vkDestroyInstance(si->inst, NULL);
      si->inst = VK_NULL_HANDLE;
      return false;
   }
   si->ok = true;
   return true;
}

static void
shared_inst_destroy(struct shared_inst *si)
{
   if (si->inst)
      vkDestroyInstance(si->inst, NULL);
   memset(si, 0, sizeof(*si));
}

static uint32_t
find_mem_type(const struct shared_inst *si, uint32_t type_bits, bool host)
{
   uint32_t pref = host ? si->mem_host : si->mem_dev;
   if (type_bits & (1u << pref))
      return pref;
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(si->pdev, &mp);
   VkMemoryPropertyFlags need = host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & need) == need)
         return i;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if (type_bits & (1u << i))
         return i;
   return UINT32_MAX;
}

static bool
write_full(int fd, const void *buf, size_t n)
{
   const uint8_t *p = (const uint8_t *)buf;
   while (n > 0) {
      ssize_t r = write(fd, p, n);
      if (r < 0) {
         if (errno == EINTR)
            continue;
         return false;
      }
      p += (size_t)r;
      n -= (size_t)r;
   }
   return true;
}

static bool
read_full_deadline(int fd, void *buf, size_t n, uint64_t deadline_ns)
{
   uint8_t *p = (uint8_t *)buf;
   while (n > 0) {
      uint64_t now = gb_now_ns();
      if (now >= deadline_ns)
         return false;
      uint64_t rem_ms = (deadline_ns - now + 999999ull) / 1000000ull;
      int timeout = rem_ms > (uint64_t)INT_MAX ? INT_MAX : (int)rem_ms;
      struct pollfd pfd = {.fd = fd, .events = POLLIN};
      int pr = poll(&pfd, 1, timeout);
      if (pr < 0) {
         if (errno == EINTR)
            continue;
         return false;
      }
      if (pr == 0)
         return false;
      ssize_t r = read(fd, p, n);
      if (r < 0) {
         if (errno == EINTR)
            continue;
         return false;
      }
      if (r == 0)
         return false;
      p += (size_t)r;
      n -= (size_t)r;
   }
   return true;
}

static uint64_t
xorshift64(uint64_t *state)
{
   uint64_t x = *state;
   x ^= x << 13;
   x ^= x >> 7;
   x ^= x << 17;
   *state = x;
   return x;
}

static VkResult
create_compute_pipeline(VkDevice dev, VkDescriptorSetLayout *out_dsl,
                        VkPipelineLayout *out_layout, VkPipeline *out_pipe)
{
   VkDescriptorSetLayoutBinding b = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &b,
   };
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkResult vr = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl);
   if (vr != VK_SUCCESS)
      return vr;
   *out_dsl = dsl;

   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pr,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   vr = vkCreatePipelineLayout(dev, &plci, NULL, &layout);
   if (vr != VK_SUCCESS)
      return vr;
   *out_layout = layout;

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = light_comp_size,
      .pCode = light_comp,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   vr = vkCreateShaderModule(dev, &smci, NULL, &sm);
   if (vr != VK_SUCCESS)
      return vr;

   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = sm,
         .pName = "main",
      },
      .layout = layout,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   vr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
   vkDestroyShaderModule(dev, sm, NULL);
   if (vr != VK_SUCCESS) {
      if (pipe != VK_NULL_HANDLE)
         vkDestroyPipeline(dev, pipe, NULL);
      return vr;
   }
   *out_pipe = pipe;
   return VK_SUCCESS;
}

static VkResult
create_graphics_pipeline(VkDevice dev, VkFormat fmt,
                         VkPipelineLayout *out_layout, VkPipeline *out_pipe)
{
   VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pr,
   };
   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkResult vr = vkCreatePipelineLayout(dev, &plci, NULL, &layout);
   if (vr != VK_SUCCESS)
      return vr;
   *out_layout = layout;

   VkShaderModuleCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = fs_vert_size,
      .pCode = fs_vert,
   };
   VkShaderModuleCreateInfo fci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = color_frag_size,
      .pCode = color_frag,
   };
   VkShaderModule mv = VK_NULL_HANDLE, mf = VK_NULL_HANDLE;
   vr = vkCreateShaderModule(dev, &vci, NULL, &mv);
   if (vr != VK_SUCCESS)
      return vr;
   vr = vkCreateShaderModule(dev, &fci, NULL, &mf);
   if (vr != VK_SUCCESS) {
      vkDestroyShaderModule(dev, mv, NULL);
      return vr;
   }
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
      .stageCount = 2, .pStages = st,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
      .layout = layout,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   vr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe);
   vkDestroyShaderModule(dev, mv, NULL);
   vkDestroyShaderModule(dev, mf, NULL);
   if (vr != VK_SUCCESS) {
      if (pipe != VK_NULL_HANDLE)
         vkDestroyPipeline(dev, pipe, NULL);
      return vr;
   }
   *out_pipe = pipe;
   return VK_SUCCESS;
}

static VkResult
alloc_host_buffer(const struct shared_inst *si, VkDevice dev, VkDeviceSize size,
                  VkBufferUsageFlags usage, VkBuffer *out_buf,
                  VkDeviceMemory *out_mem, void **out_map)
{
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   VkBuffer buf = VK_NULL_HANDLE;
   VkResult vr = vkCreateBuffer(dev, &bci, NULL, &buf);
   if (vr != VK_SUCCESS)
      return vr;

   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);
   uint32_t mt = find_mem_type(si, mr.memoryTypeBits, true);
   if (mt == UINT32_MAX) {
      vkDestroyBuffer(dev, buf, NULL);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem = VK_NULL_HANDLE;
   vr = vkAllocateMemory(dev, &mai, NULL, &mem);
   if (vr != VK_SUCCESS) {
      vkDestroyBuffer(dev, buf, NULL);
      return vr;
   }
   vr = vkBindBufferMemory(dev, buf, mem, 0);
   if (vr != VK_SUCCESS) {
      vkDestroyBuffer(dev, buf, NULL);
      vkFreeMemory(dev, mem, NULL);
      return vr;
   }
   void *ptr = NULL;
   vr = vkMapMemory(dev, mem, 0, size, 0, &ptr);
   if (vr != VK_SUCCESS) {
      vkDestroyBuffer(dev, buf, NULL);
      vkFreeMemory(dev, mem, NULL);
      return vr;
   }
   memset(ptr, 0, (size_t)size);
   *out_buf = buf;
   *out_mem = mem;
   *out_map = ptr;
   return VK_SUCCESS;
}

static VkResult
alloc_color_image(const struct shared_inst *si, VkDevice dev, VkFormat fmt,
                  VkImageUsageFlags usage, VkImage *out_img,
                  VkDeviceMemory *out_mem, VkImageView *out_view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {REBIND_LIGHT_DIM, REBIND_LIGHT_DIM, 1},
      .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   VkImage img = VK_NULL_HANDLE;
   VkResult vr = vkCreateImage(dev, &ici, NULL, &img);
   if (vr != VK_SUCCESS)
      return vr;

   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   uint32_t mt = find_mem_type(si, mr.memoryTypeBits, false);
   if (mt == UINT32_MAX) {
      vkDestroyImage(dev, img, NULL);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = mt,
   };
   VkDeviceMemory mem = VK_NULL_HANDLE;
   vr = vkAllocateMemory(dev, &mai, NULL, &mem);
   if (vr != VK_SUCCESS) {
      vkDestroyImage(dev, img, NULL);
      return vr;
   }
   vr = vkBindImageMemory(dev, img, mem, 0);
   if (vr != VK_SUCCESS) {
      vkDestroyImage(dev, img, NULL);
      vkFreeMemory(dev, mem, NULL);
      return vr;
   }
   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = fmt,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view = VK_NULL_HANDLE;
   vr = vkCreateImageView(dev, &ivci, NULL, &view);
   if (vr != VK_SUCCESS) {
      vkDestroyImage(dev, img, NULL);
      vkFreeMemory(dev, mem, NULL);
      return vr;
   }
   *out_img = img;
   *out_mem = mem;
   *out_view = view;
   return VK_SUCCESS;
}

static void
seed_to_rgba8(uint32_t seed, uint8_t rgba[4], float fcolor[4])
{
   /* Exact multiples of 1/255 round cleanly to the same u8 in UNORM8. */
   rgba[0] = (uint8_t)(((seed * 37u + 17u) & 0xfe) | 0x01);
   rgba[1] = (uint8_t)((((seed >> 8) * 53u + 41u) & 0xfe) | 0x01);
   rgba[2] = (uint8_t)((((seed >> 16) * 71u + 89u) & 0xfe) | 0x01);
   rgba[3] = 255u;
   fcolor[0] = (float)rgba[0] / 255.0f;
   fcolor[1] = (float)rgba[1] / 255.0f;
   fcolor[2] = (float)rgba[2] / 255.0f;
   fcolor[3] = 1.0f;
}

/* =========================================================================
 * Case 1: queue.rebind — multi-process mixed compute+render queue rebind
 * ========================================================================= */

struct rebind_go_msg {
   uint64_t t_start;
   bool abort;
};

struct rebind_child_out {
   uint64_t win_iters[GB_MAX_WIN];
   uint64_t measured_iters;
   double iter_us[REBIND_MAX_SAMPLES];
   size_t n_samples;
   uint32_t last_got_crc;
   uint32_t last_exp_crc;
   int fail_priority; /* 3=device_lost, 2=check_bad, 1=other/timeout */
   bool failed;
   bool check_bad;
   char err[128];
};

static void
rebind_child_run(int wid, int device_index, int nqueues, int inflight,
                 uint64_t warm_ns, uint64_t win_ns, int nwin,
                 int ready_fd, int go_fd, int out_fd)
{
   struct rebind_child_out out;
   memset(&out, 0, sizeof(out));

   struct shared_inst si;
   if (!shared_inst_init(&si, device_index)) {
      out.failed = true;
      out.fail_priority = 1;
      snprintf(out.err, sizeof(out.err), "%s", si.err);
      uint8_t ready = 0;
      write_full(ready_fd, &ready, 1);
      write_full(out_fd, &out, sizeof(out));
      _exit(0);
   }

   float prios[REBIND_MAX_QUEUES];
   for (int i = 0; i < nqueues; i++)
      prios[i] = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = si.qfam,
      .queueCount = (uint32_t)nqueues,
      .pQueuePriorities = prios,
   };
   VkPhysicalDeviceVulkan13Features e13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
      .maintenance4 = si.has_maintenance4 ? VK_TRUE : VK_FALSE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &e13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice dev = VK_NULL_HANDLE;
   VkResult vr = vkCreateDevice(si.pdev, &dci, NULL, &dev);
   if (vr != VK_SUCCESS) {
      out.failed = true;
      out.fail_priority = (vr == VK_ERROR_DEVICE_LOST) ? 3 : 1;
      snprintf(out.err, sizeof(out.err), "vkCreateDevice failed: %d", (int)vr);
      uint8_t ready = 0;
      write_full(ready_fd, &ready, 1);
      write_full(out_fd, &out, sizeof(out));
      shared_inst_destroy(&si);
      _exit(0);
   }

   const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   VkPipelineLayout comp_layout = VK_NULL_HANDLE, gfx_layout = VK_NULL_HANDLE;
   VkPipeline comp_pipe = VK_NULL_HANDLE, gfx_pipe = VK_NULL_HANDLE;

   VkQueue qs[REBIND_MAX_QUEUES] = {0};
   VkCommandPool cps[REBIND_MAX_QUEUES] = {0};
   VkCommandBuffer cbs[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkFence fs[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkSemaphore sems[REBIND_MAX_QUEUES] = {0};

   /* Disjoint per-(q, k) compute outputs (pass A & pass B) and render targets + readback. */
   VkBuffer comp_buf[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkDeviceMemory comp_mem[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   uint8_t *comp_map[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkDescriptorSet dset_a[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkDescriptorSet dset_b[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};

   VkImage imgs[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkDeviceMemory img_mems[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkImageView img_views[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkBuffer pix_buf[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   VkDeviceMemory pix_mem[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};
   uint8_t *pix_map[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT] = {{0}};

   bool submitted_pending = false;
   bool device_lost = false;
   bool ready_sent = false;

   if ((vr = create_compute_pipeline(dev, &dsl, &comp_layout, &comp_pipe)) != VK_SUCCESS)
      goto setup_fail;
   if ((vr = create_graphics_pipeline(dev, fmt, &gfx_layout, &gfx_pipe)) != VK_SUCCESS)
      goto setup_fail;

   {
      uint32_t nsets = (uint32_t)(nqueues * inflight * 2);
      VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nsets};
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = nsets,
         .poolSizeCount = 1,
         .pPoolSizes = &dps,
      };
      VkDescriptorPool tmp_dp = VK_NULL_HANDLE;
      if ((vr = vkCreateDescriptorPool(dev, &dpci, NULL, &tmp_dp)) != VK_SUCCESS)
         goto setup_fail;
      dpool = tmp_dp;
   }

   for (int q = 0; q < nqueues; q++) {
      vkGetDeviceQueue(dev, si.qfam, (uint32_t)q, &qs[q]);
      VkCommandPoolCreateInfo cpci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
         .queueFamilyIndex = si.qfam,
      };
      VkCommandPool tmp_cp = VK_NULL_HANDLE;
      if ((vr = vkCreateCommandPool(dev, &cpci, NULL, &tmp_cp)) != VK_SUCCESS)
         goto setup_fail;
      cps[q] = tmp_cp;

      VkCommandBufferAllocateInfo cbai = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool = cps[q],
         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
         .commandBufferCount = (uint32_t)inflight,
      };
      if ((vr = vkAllocateCommandBuffers(dev, &cbai, cbs[q])) != VK_SUCCESS)
         goto setup_fail;

      VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VkSemaphore tmp_sem = VK_NULL_HANDLE;
      if ((vr = vkCreateSemaphore(dev, &sci, NULL, &tmp_sem)) != VK_SUCCESS)
         goto setup_fail;
      sems[q] = tmp_sem;

      for (int k = 0; k < inflight; k++) {
         VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
         VkFence tmp_f = VK_NULL_HANDLE;
         if ((vr = vkCreateFence(dev, &fci, NULL, &tmp_f)) != VK_SUCCESS)
            goto setup_fail;
         fs[q][k] = tmp_f;

         /* Two disjoint 256-byte slices in comp_buf[q][k]: [0..255] for pre-render
          * compute dispatch, [256..511] for post-render compute dispatch. */
         void *cmap = NULL;
         if ((vr = alloc_host_buffer(&si, dev, 2u * REBIND_SLICE_BYTES,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     &comp_buf[q][k], &comp_mem[q][k], &cmap)) != VK_SUCCESS)
            goto setup_fail;
         comp_map[q][k] = (uint8_t *)cmap;

         VkDescriptorSetLayout layouts[2] = {dsl, dsl};
         VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = dpool,
            .descriptorSetCount = 2,
            .pSetLayouts = layouts,
         };
         VkDescriptorSet dpair[2] = {0};
         if ((vr = vkAllocateDescriptorSets(dev, &dsai, dpair)) != VK_SUCCESS)
            goto setup_fail;
         dset_a[q][k] = dpair[0];
         dset_b[q][k] = dpair[1];

         VkDescriptorBufferInfo dbia = {comp_buf[q][k], 0, REBIND_SLICE_BYTES};
         VkDescriptorBufferInfo dbib = {comp_buf[q][k], REBIND_SLICE_BYTES, REBIND_SLICE_BYTES};
         VkWriteDescriptorSet ws[2] = {
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstSet = dset_a[q][k], .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbia},
            {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             .dstSet = dset_b[q][k], .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbib},
         };
         vkUpdateDescriptorSets(dev, 2, ws, 0, NULL);

         if ((vr = alloc_color_image(&si, dev, fmt,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                     &imgs[q][k], &img_mems[q][k], &img_views[q][k])) != VK_SUCCESS)
            goto setup_fail;

         void *pmap = NULL;
         if ((vr = alloc_host_buffer(&si, dev, REBIND_PIX_COUNT * 4u,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     &pix_buf[q][k], &pix_mem[q][k], &pmap)) != VK_SUCCESS)
            goto setup_fail;
         pix_map[q][k] = (uint8_t *)pmap;
      }
   }

   /* Signal parent that all queues and device resources are open. */
   {
      uint8_t ready = 1;
      ready_sent = true;
      if (!write_full(ready_fd, &ready, 1))
         goto cleanup;
   }
   struct rebind_go_msg go = {0};
   if (!read_full_deadline(go_fd, &go, sizeof(go), gb_now_ns() + (uint64_t)REBIND_SETUP_MS * 1000000ull) ||
       go.abort)
      goto cleanup;

   const uint64_t t_meas = go.t_start + warm_ns;
   const uint64_t t_end = t_meas + (uint64_t)nwin * win_ns;
   gb_sleep_until_ns(go.t_start);

   uint64_t iter = 0;
   uint64_t rng = 0x9e3779b97f4a7c15ull ^ ((uint64_t)(wid + 1) * 0xbf58476d1ce4e5b9ull);
   uint64_t seen_samples = 0;

   while (gb_now_ns() < t_end) {
      uint32_t seeds[REBIND_MAX_QUEUES][REBIND_MAX_INFLIGHT];
      uint64_t ts0 = gb_now_ns();

      for (int q = 0; q < nqueues; q++) {
         for (int k = 0; k < inflight; k++) {
            uint32_t seed = (uint32_t)((iter * 131u) ^ ((uint64_t)wid << 20) ^
                                       ((uint64_t)q << 12) ^ ((uint64_t)k << 4)) + 1u;
            uint32_t pre_seed = seed ^ 0x55555555u;
            seeds[q][k] = seed;

            if ((vr = vkResetCommandBuffer(cbs[q][k], 0)) != VK_SUCCESS)
               goto run_fail;
            VkCommandBufferBeginInfo bi = {
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
               .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            if ((vr = vkBeginCommandBuffer(cbs[q][k], &bi)) != VK_SUCCESS)
               goto run_fail;

            /* 1. Pre-render compute dispatch into disjoint slice A. */
            vkCmdBindPipeline(cbs[q][k], VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipe);
            vkCmdBindDescriptorSets(cbs[q][k], VK_PIPELINE_BIND_POINT_COMPUTE, comp_layout,
                                    0, 1, &dset_a[q][k], 0, NULL);
            vkCmdPushConstants(cbs[q][k], comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pre_seed), &pre_seed);
            vkCmdDispatch(cbs[q][k], 1, 1, 1);

            /* 2. Compute -> render execution barrier + image transition for disjoint img[q][k]. */
            VkMemoryBarrier2 c2r = {
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            };
            VkImageMemoryBarrier2 to_att = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
               .srcAccessMask = 0,
               .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
               .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               .image = imgs[q][k],
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            VkDependencyInfo di1 = {
               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
               .memoryBarrierCount = 1, .pMemoryBarriers = &c2r,
               .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_att,
            };
            vkCmdPipelineBarrier2(cbs[q][k], &di1);

            /* 3. Explicit graphics draw pass with deterministic push-constant RGBA color. */
            uint8_t want_rgba[4];
            float fcolor[4];
            seed_to_rgba8(seed, want_rgba, fcolor);
            VkRenderingAttachmentInfo ca = {
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = img_views[q][k],
               .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f},
            };
            VkRenderingInfo ri = {
               .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
               .renderArea = {{0, 0}, {REBIND_LIGHT_DIM, REBIND_LIGHT_DIM}},
               .layerCount = 1,
               .colorAttachmentCount = 1,
               .pColorAttachments = &ca,
            };
            vkCmdBeginRendering(cbs[q][k], &ri);
            VkViewport vp = {0, 0, (float)REBIND_LIGHT_DIM, (float)REBIND_LIGHT_DIM, 0.0f, 1.0f};
            VkRect2D sc = {{0, 0}, {REBIND_LIGHT_DIM, REBIND_LIGHT_DIM}};
            vkCmdSetViewport(cbs[q][k], 0, 1, &vp);
            vkCmdSetScissor(cbs[q][k], 0, 1, &sc);
            vkCmdBindPipeline(cbs[q][k], VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_pipe);
            vkCmdPushConstants(cbs[q][k], gfx_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(fcolor), fcolor);
            vkCmdDraw(cbs[q][k], 3, 1, 0, 0);
            vkCmdEndRendering(cbs[q][k]);

            /* 4. Render -> transfer+compute barrier and copy a 4x4 patch for pixel verification. */
            VkImageMemoryBarrier2 to_src = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
               .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               .image = imgs[q][k],
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            VkDependencyInfo di2 = {
               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
               .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_src,
            };
            vkCmdPipelineBarrier2(cbs[q][k], &di2);

            VkBufferImageCopy region = {
               .bufferOffset = 0,
               .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
               .imageOffset = {REBIND_LIGHT_DIM / 2, REBIND_LIGHT_DIM / 2, 0},
               .imageExtent = {REBIND_PIX_W, REBIND_PIX_H, 1},
            };
            vkCmdCopyImageToBuffer(cbs[q][k], imgs[q][k], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   pix_buf[q][k], 1, &region);

            /* 5. Post-render compute dispatch into disjoint slice B. */
            vkCmdBindPipeline(cbs[q][k], VK_PIPELINE_BIND_POINT_COMPUTE, comp_pipe);
            vkCmdBindDescriptorSets(cbs[q][k], VK_PIPELINE_BIND_POINT_COMPUTE, comp_layout,
                                    0, 1, &dset_b[q][k], 0, NULL);
            vkCmdPushConstants(cbs[q][k], comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seed), &seed);
            vkCmdDispatch(cbs[q][k], 1, 1, 1);

            /* 6. Host visibility barriers for both compute slices and pixel readback buffer. */
            VkBufferMemoryBarrier2 host_bmbs[2] = {
               {
                  .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                  .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                  .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                  .buffer = comp_buf[q][k],
                  .offset = 0,
                  .size = 2u * REBIND_SLICE_BYTES,
               },
               {
                  .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                  .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                  .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                  .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                  .buffer = pix_buf[q][k],
                  .offset = 0,
                  .size = REBIND_PIX_COUNT * 4u,
               },
            };
            VkDependencyInfo di3 = {
               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
               .bufferMemoryBarrierCount = 2, .pBufferMemoryBarriers = host_bmbs,
            };
            vkCmdPipelineBarrier2(cbs[q][k], &di3);

            if ((vr = vkEndCommandBuffer(cbs[q][k])) != VK_SUCCESS)
               goto run_fail;
            if ((vr = vkResetFences(dev, 1, &fs[q][k])) != VK_SUCCESS)
               goto run_fail;

            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo sub = {
               .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
               .commandBufferCount = 1,
               .pCommandBuffers = &cbs[q][k],
            };
            if (k == 0 && q > 0) {
               sub.waitSemaphoreCount = 1;
               sub.pWaitSemaphores = &sems[q - 1];
               sub.pWaitDstStageMask = &wait_stage;
            }
            if (k == inflight - 1 && q + 1 < nqueues) {
               sub.signalSemaphoreCount = 1;
               sub.pSignalSemaphores = &sems[q];
            }
            if ((vr = vkQueueSubmit(qs[q], 1, &sub, fs[q][k])) != VK_SUCCESS) {
               if (vr == VK_ERROR_DEVICE_LOST)
                  submitted_pending = true;
               goto run_fail;
            }
            submitted_pending = true;
         }
      }

      VkFence all_f[REBIND_MAX_QUEUES * REBIND_MAX_INFLIGHT];
      int nf = 0;
      for (int q = 0; q < nqueues; q++)
         for (int k = 0; k < inflight; k++)
            all_f[nf++] = fs[q][k];

      vr = vkWaitForFences(dev, (uint32_t)nf, all_f, VK_TRUE, REBIND_TIMEOUT_NS);
      if (vr != VK_SUCCESS)
         goto run_fail;
      submitted_pending = false;
      uint64_t ts1 = gb_now_ns();

      /* Verify both compute outputs (pre & post render) and rendered pixels. */
      for (int q = 0; q < nqueues; q++) {
         for (int k = 0; k < inflight; k++) {
            uint32_t seed = seeds[q][k];
            uint32_t pre_seed = seed ^ 0x55555555u;
            uint32_t want_a[REBIND_WORDS], want_b[REBIND_WORDS];
            for (uint32_t i = 0; i < REBIND_WORDS; i++) {
               want_a[i] = pre_seed + i;
               want_b[i] = seed + i;
            }
            uint8_t want_rgba[4];
            float funused[4];
            seed_to_rgba8(seed, want_rgba, funused);
            uint8_t want_pix[REBIND_PIX_COUNT * 4u];
            for (uint32_t p = 0; p < REBIND_PIX_COUNT; p++)
               memcpy(&want_pix[p * 4u], want_rgba, 4u);

            const uint8_t *got_a = comp_map[q][k];
            const uint8_t *got_b = comp_map[q][k] + REBIND_SLICE_BYTES;
            const uint8_t *got_p = pix_map[q][k];

            uint32_t got_combo[3] = {
               gb_crc32(got_a, sizeof(want_a)),
               gb_crc32(got_b, sizeof(want_b)),
               gb_crc32(got_p, sizeof(want_pix)),
            };
            uint32_t exp_combo[3] = {
               gb_crc32(want_a, sizeof(want_a)),
               gb_crc32(want_b, sizeof(want_b)),
               gb_crc32(want_pix, sizeof(want_pix)),
            };
            uint32_t got_crc = gb_crc32(got_combo, sizeof(got_combo));
            uint32_t exp_crc = gb_crc32(exp_combo, sizeof(exp_combo));
            out.last_got_crc = got_crc;
            out.last_exp_crc = exp_crc;

            bool bad_a = memcmp(got_a, want_a, sizeof(want_a)) != 0;
            bool bad_b = memcmp(got_b, want_b, sizeof(want_b)) != 0;
            bool bad_p = memcmp(got_p, want_pix, sizeof(want_pix)) != 0;
            if (bad_a || bad_b || bad_p) {
               out.failed = true;
               out.check_bad = true;
               out.fail_priority = 2;
               snprintf(out.err, sizeof(out.err),
                        "worker %d q=%d k=%d iter=%llu %s mismatch (crc %08x != %08x)",
                        wid, q, k, (unsigned long long)iter,
                        bad_p ? "render" : "compute", got_crc, exp_crc);
               goto cleanup;
            }
         }
      }

      iter++;
      if (ts1 >= t_meas && ts1 < t_end) {
         int wi = (int)((ts1 - t_meas) / win_ns);
         if (wi >= 0 && wi < nwin) {
            out.win_iters[wi]++;
            out.measured_iters++;
            double lat_us = (double)(ts1 - ts0) / 1e3;
            seen_samples++;
            if (out.n_samples < REBIND_MAX_SAMPLES) {
               out.iter_us[out.n_samples++] = lat_us;
            } else {
               uint64_t j = xorshift64(&rng) % seen_samples;
               if (j < REBIND_MAX_SAMPLES)
                  out.iter_us[j] = lat_us;
            }
         }
      }
   }
   goto cleanup;

setup_fail:
   out.failed = true;
   out.fail_priority = (vr == VK_ERROR_DEVICE_LOST) ? 3 : 1;
   snprintf(out.err, sizeof(out.err), "worker %d setup failed: %d", wid, (int)vr);
   device_lost = (vr == VK_ERROR_DEVICE_LOST);
   if (!ready_sent) {
      uint8_t ready = 0;
      write_full(ready_fd, &ready, 1);
   }
   goto cleanup;

run_fail:
   out.failed = true;
   device_lost = (vr == VK_ERROR_DEVICE_LOST);
   out.fail_priority = device_lost ? 3 : 1;
   snprintf(out.err, sizeof(out.err), "%s (%d) on worker %d after %llu iters",
            device_lost ? "VK_ERROR_DEVICE_LOST"
            : vr == VK_TIMEOUT ? "GPU wait timed out"
            : "Vulkan call failed",
            (int)vr, wid, (unsigned long long)iter);

cleanup:
   /* Only tear down Vulkan handles when no submissions remain pending. After
    * VK_TIMEOUT or VK_ERROR_DEVICE_LOST with submitted work, exit the worker
    * process directly rather than destroying resources with pending state. */
   if (!submitted_pending) {
      if (dev) {
         if (!device_lost)
            vkDeviceWaitIdle(dev);
         for (int q = 0; q < nqueues; q++) {
            for (int k = 0; k < inflight; k++) {
               if (fs[q][k]) vkDestroyFence(dev, fs[q][k], NULL);
               if (img_views[q][k]) vkDestroyImageView(dev, img_views[q][k], NULL);
               if (imgs[q][k]) vkDestroyImage(dev, imgs[q][k], NULL);
               if (img_mems[q][k]) vkFreeMemory(dev, img_mems[q][k], NULL);
               if (pix_map[q][k]) vkUnmapMemory(dev, pix_mem[q][k]);
               if (pix_buf[q][k]) vkDestroyBuffer(dev, pix_buf[q][k], NULL);
               if (pix_mem[q][k]) vkFreeMemory(dev, pix_mem[q][k], NULL);
               if (comp_map[q][k]) vkUnmapMemory(dev, comp_mem[q][k]);
               if (comp_buf[q][k]) vkDestroyBuffer(dev, comp_buf[q][k], NULL);
               if (comp_mem[q][k]) vkFreeMemory(dev, comp_mem[q][k], NULL);
            }
            if (sems[q]) vkDestroySemaphore(dev, sems[q], NULL);
            if (cps[q]) vkDestroyCommandPool(dev, cps[q], NULL);
         }
         if (gfx_pipe) vkDestroyPipeline(dev, gfx_pipe, NULL);
         if (gfx_layout) vkDestroyPipelineLayout(dev, gfx_layout, NULL);
         if (comp_pipe) vkDestroyPipeline(dev, comp_pipe, NULL);
         if (comp_layout) vkDestroyPipelineLayout(dev, comp_layout, NULL);
         if (dpool) vkDestroyDescriptorPool(dev, dpool, NULL);
         if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, NULL);
         vkDestroyDevice(dev, NULL);
      }
      shared_inst_destroy(&si);
   }
   write_full(out_fd, &out, sizeof(out));
   _exit(0);
}

static int
waitpid_retry(pid_t pid, int *status)
{
   for (;;) {
      pid_t r = waitpid(pid, status, 0);
      if (r < 0 && errno == EINTR)
         continue;
      return r < 0 ? -1 : 0;
   }
}

static int
run_queue_rebind_procs(const struct gb_opts *o, int nqueues, int inflight)
{
   struct shared_inst si;
   if (!shared_inst_init(&si, o->device)) {
      char pq[32], pi[32];
      snprintf(pq, sizeof(pq), "queues=%d", nqueues);
      snprintf(pi, sizeof(pi), "inflight=%d", inflight);
      const char *const extra[] = {pq, pi, NULL};
      gb_result_simple(o, GB_MODE_PROCS, nqueues * inflight, extra, "fail", si.err);
      return 1;
   }
   uint32_t family_queues = si.family_queues;
   shared_inst_destroy(&si);

   int depths[GB_MAX_DEPTHS];
   int nd = 0;
   if (o->ndepths > 0) {
      for (int i = 0; i < o->ndepths; i++)
         depths[nd++] = o->depths[i];
   } else {
      depths[0] = 1;
      depths[1] = 6;
      nd = 2;
   }

   const uint64_t warm_ns = (uint64_t)o->warmup_ms * 1000000ull;
   const uint64_t win_ns = (uint64_t)o->run_ms * 1000000ull;
   const double win_us = (double)o->run_ms * 1000.0;
   void (*prev_sigpipe)(int) = signal(SIGPIPE, SIG_IGN);

   for (int di = 0; di < nd; di++) {
      int nprocs = depths[di];
      char pq[32], pi[32], pp[32];
      snprintf(pq, sizeof(pq), "queues=%d", nqueues);
      snprintf(pi, sizeof(pi), "inflight=%d", inflight);
      snprintf(pp, sizeof(pp), "procs=%d", nprocs);
      const char *const extra_params[] = {pq, pi, pp, NULL};

      if (nprocs < 1 || nprocs > REBIND_MAX_WORKERS) {
         gb_result_simple(o, GB_MODE_PROCS, nprocs, extra_params, "fail", "depth (procs) must be 1..8");
         continue;
      }
      const int total_depth = nprocs * nqueues * inflight;
      if ((uint32_t)nqueues > family_queues) {
         char why[64];
         snprintf(why, sizeof(why), "device has %u queues (need %d)", family_queues, nqueues);
         gb_result_simple(o, GB_MODE_PROCS, total_depth, extra_params, "skip", why);
         continue;
      }

      pid_t pids[REBIND_MAX_WORKERS] = {0};
      int ready_p[REBIND_MAX_WORKERS][2];
      int go_p[REBIND_MAX_WORKERS][2];
      int out_p[REBIND_MAX_WORKERS][2];
      for (int i = 0; i < nprocs; i++) {
         ready_p[i][0] = ready_p[i][1] = -1;
         go_p[i][0] = go_p[i][1] = -1;
         out_p[i][0] = out_p[i][1] = -1;
      }

      bool spawn_ok = true;
      int spawned = 0;
      for (int i = 0; i < nprocs; i++) {
         if (pipe(ready_p[i]) != 0 || pipe(go_p[i]) != 0 || pipe(out_p[i]) != 0) {
            spawn_ok = false;
            break;
         }
         fflush(stdout);
         fflush(stderr);
         pid_t pid = fork();
         if (pid < 0) {
            spawn_ok = false;
            break;
         }
         if (pid == 0) {
            for (int j = 0; j < i; j++) {
               close(ready_p[j][0]);
               close(go_p[j][1]);
               close(out_p[j][0]);
            }
            close(ready_p[i][0]);
            close(go_p[i][1]);
            close(out_p[i][0]);
            rebind_child_run(i, o->device, nqueues, inflight, warm_ns, win_ns, o->runs,
                             ready_p[i][1], go_p[i][0], out_p[i][1]);
            _exit(0);
         }
         pids[i] = pid;
         spawned++;
         close(ready_p[i][1]); ready_p[i][1] = -1;
         close(go_p[i][0]);    go_p[i][0] = -1;
         close(out_p[i][1]);   out_p[i][1] = -1;
      }

      bool child_ready[REBIND_MAX_WORKERS] = {0};
      bool all_ready = spawn_ok;
      uint64_t setup_deadline = gb_now_ns() + (uint64_t)REBIND_SETUP_MS * 1000000ull;
      for (int i = 0; i < spawned; i++) {
         uint8_t rdy = 0;
         if (read_full_deadline(ready_p[i][0], &rdy, 1, setup_deadline) && rdy == 1)
            child_ready[i] = true;
         else
            all_ready = false;
         close(ready_p[i][0]); ready_p[i][0] = -1;
      }

      struct rebind_go_msg go = {
         .t_start = gb_now_ns() + 20000000ull,
         .abort = !all_ready,
      };
      for (int i = 0; i < spawned; i++) {
         if (child_ready[i])
            write_full(go_p[i][1], &go, sizeof(go));
         close(go_p[i][1]); go_p[i][1] = -1;
      }

      struct rebind_child_out outs[REBIND_MAX_WORKERS];
      memset(outs, 0, sizeof(outs));
      bool read_ok[REBIND_MAX_WORKERS] = {0};
      bool timed_out_child[REBIND_MAX_WORKERS] = {0};
      uint64_t total_run_ms = (uint64_t)o->warmup_ms + (uint64_t)o->runs * (uint64_t)o->run_ms + REBIND_EXTRA_MS;
      uint64_t result_deadline = gb_now_ns() + total_run_ms * 1000000ull;

      for (int i = 0; i < spawned; i++) {
         if (read_full_deadline(out_p[i][0], &outs[i], sizeof(outs[i]), result_deadline)) {
            read_ok[i] = true;
         } else {
            timed_out_child[i] = (gb_now_ns() >= result_deadline);
            if (pids[i] > 0)
               kill(pids[i], SIGKILL);
         }
         close(out_p[i][0]); out_p[i][0] = -1;
      }
      for (int i = 0; i < nprocs; i++) {
         if (ready_p[i][0] >= 0) close(ready_p[i][0]);
         if (ready_p[i][1] >= 0) close(ready_p[i][1]);
         if (go_p[i][0] >= 0) close(go_p[i][0]);
         if (go_p[i][1] >= 0) close(go_p[i][1]);
         if (out_p[i][0] >= 0) close(out_p[i][0]);
         if (out_p[i][1] >= 0) close(out_p[i][1]);
      }

      bool failed = !spawn_ok;
      bool check_bad = false;
      int best_prio = 0;
      char err[128] = {0};
      if (!spawn_ok)
         snprintf(err, sizeof(err), "fork/pipe failed");

      uint64_t measured_iters = 0;
      uint32_t got_crc = 0, exp_crc = 0;
      uint32_t bad_got_crc = 0, bad_exp_crc = 0;
      struct gb_samples iter_rate = {0}, submit_rate = {0}, iter_lat = {0};

      for (int i = 0; i < spawned; i++) {
         int st = 0;
         waitpid_retry(pids[i], &st);
         if (!read_ok[i]) {
            failed = true;
            if (best_prio < 1) {
               best_prio = 1;
               if (timed_out_child[i])
                  snprintf(err, sizeof(err), "worker %d exceeded parent deadline and was killed", i);
               else if (WIFSIGNALED(st))
                  snprintf(err, sizeof(err), "worker %d killed by signal %d", i, WTERMSIG(st));
               else
                  snprintf(err, sizeof(err), "worker %d exited without result (status %d)",
                           i, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
            }
            continue;
         }
         measured_iters += outs[i].measured_iters;
         if (outs[i].last_exp_crc) {
            got_crc = outs[i].last_got_crc;
            exp_crc = outs[i].last_exp_crc;
         }
         if (outs[i].failed) {
            failed = true;
            if (outs[i].check_bad && !check_bad) {
               check_bad = true;
               bad_got_crc = outs[i].last_got_crc;
               bad_exp_crc = outs[i].last_exp_crc;
            }
            if (outs[i].fail_priority > best_prio) {
               best_prio = outs[i].fail_priority;
               snprintf(err, sizeof(err), "%s", outs[i].err);
            }
         }
         for (size_t s = 0; s < outs[i].n_samples && s < REBIND_MAX_SAMPLES; s++)
            gb_samples_push(&iter_lat, outs[i].iter_us[s]);
      }

      for (int wi = 0; wi < o->runs; wi++) {
         uint64_t c = 0;
         for (int i = 0; i < spawned; i++)
            if (read_ok[i])
               c += outs[i].win_iters[wi];
         if (c > 0) {
            double us_it = win_us / (double)c;
            gb_samples_push(&iter_rate, us_it);
            gb_samples_push(&submit_rate, us_it / (double)(nqueues * inflight));
         }
      }

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_PROCS, total_depth, extra_params);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (failed || !iter_rate.n) {
         if (!err[0])
            snprintf(err, sizeof(err), "no completed iterations in measured windows");
         gb_result_end_metrics(&r);
         if (check_bad) {
            struct gb_check chk = {.done = true, .valid = false, .crc = bad_got_crc, .expected = bad_exp_crc};
            gb_result_check(&r, &chk);
         }
         gb_result_finish(&r, "fail", NULL, err);
      } else {
         gb_result_metric(&r, "us_per_iter", "us", true, &iter_rate, true);
         gb_result_metric(&r, "us_per_submit", "us", true, &submit_rate, true);
         gb_result_metric(&r, "iter_us", "us", true, &iter_lat, false);
         gb_result_metric1(&r, "iters", "count", false, (double)measured_iters);
         gb_result_end_metrics(&r);
         struct gb_check chk = {.done = true, .valid = true, .crc = got_crc, .expected = exp_crc};
         gb_result_check(&r, &chk);
         gb_result_finish(&r, "ok", "us_per_iter", NULL);
      }

      gb_samples_free(&iter_rate);
      gb_samples_free(&submit_rate);
      gb_samples_free(&iter_lat);
   }

   signal(SIGPIPE, prev_sigpipe);
   return 0;
}

/* =========================================================================
 * Case 2: queue.churn — single-engine device/queue lifecycle churn
 * ========================================================================= */

struct cycle_dev {
   VkDevice dev;
   int nqueues;
   bool want_render;
   bool want_compute;
   VkQueue queues[REBIND_MAX_QUEUES];
   VkCommandPool pools[REBIND_MAX_QUEUES];
   VkFence fences[REBIND_MAX_QUEUES];

   /* compute resources */
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkPipelineLayout comp_layout;
   VkPipeline comp_pipe;
   VkBuffer bufs[REBIND_MAX_QUEUES];
   VkDeviceMemory buf_mems[REBIND_MAX_QUEUES];
   uint32_t *buf_maps[REBIND_MAX_QUEUES];
   VkDescriptorSet dsets[REBIND_MAX_QUEUES];

   /* render resources */
   VkPipelineLayout gfx_layout;
   VkPipeline gfx_pipe;
   VkImage imgs[REBIND_MAX_QUEUES];
   VkDeviceMemory img_mems[REBIND_MAX_QUEUES];
   VkImageView img_views[REBIND_MAX_QUEUES];
};

static void
cycle_dev_destroy(struct cycle_dev *cd, bool skip_wait)
{
   if (!cd->dev)
      return;
   if (!skip_wait)
      vkDeviceWaitIdle(cd->dev);
   for (int q = 0; q < cd->nqueues; q++) {
      if (cd->fences[q]) vkDestroyFence(cd->dev, cd->fences[q], NULL);
      if (cd->pools[q]) vkDestroyCommandPool(cd->dev, cd->pools[q], NULL);
      if (cd->img_views[q]) vkDestroyImageView(cd->dev, cd->img_views[q], NULL);
      if (cd->imgs[q]) vkDestroyImage(cd->dev, cd->imgs[q], NULL);
      if (cd->img_mems[q]) vkFreeMemory(cd->dev, cd->img_mems[q], NULL);
      if (cd->buf_maps[q]) vkUnmapMemory(cd->dev, cd->buf_mems[q]);
      if (cd->bufs[q]) vkDestroyBuffer(cd->dev, cd->bufs[q], NULL);
      if (cd->buf_mems[q]) vkFreeMemory(cd->dev, cd->buf_mems[q], NULL);
   }
   if (cd->gfx_pipe) vkDestroyPipeline(cd->dev, cd->gfx_pipe, NULL);
   if (cd->gfx_layout) vkDestroyPipelineLayout(cd->dev, cd->gfx_layout, NULL);
   if (cd->comp_pipe) vkDestroyPipeline(cd->dev, cd->comp_pipe, NULL);
   if (cd->comp_layout) vkDestroyPipelineLayout(cd->dev, cd->comp_layout, NULL);
   if (cd->dpool) vkDestroyDescriptorPool(cd->dev, cd->dpool, NULL);
   if (cd->dsl) vkDestroyDescriptorSetLayout(cd->dev, cd->dsl, NULL);
   vkDestroyDevice(cd->dev, NULL);
   memset(cd, 0, sizeof(*cd));
}

static VkResult
cycle_dev_create(const struct shared_inst *si, int nqueues, bool want_render, bool want_compute,
                 struct cycle_dev *cd)
{
   memset(cd, 0, sizeof(*cd));
   cd->nqueues = nqueues;
   cd->want_render = want_render;
   cd->want_compute = want_compute;

   float prios[REBIND_MAX_QUEUES];
   for (int i = 0; i < nqueues; i++)
      prios[i] = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = si->qfam,
      .queueCount = (uint32_t)nqueues,
      .pQueuePriorities = prios,
   };
   VkPhysicalDeviceVulkan13Features e13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
      .maintenance4 = si->has_maintenance4 ? VK_TRUE : VK_FALSE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &e13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice dev = VK_NULL_HANDLE;
   VkResult vr = vkCreateDevice(si->pdev, &dci, NULL, &dev);
   if (vr != VK_SUCCESS)
      return vr;
   cd->dev = dev;

   for (int q = 0; q < nqueues; q++) {
      vkGetDeviceQueue(cd->dev, si->qfam, (uint32_t)q, &cd->queues[q]);
      VkCommandPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
         .queueFamilyIndex = si->qfam,
      };
      VkCommandPool tmp_cp = VK_NULL_HANDLE;
      if ((vr = vkCreateCommandPool(cd->dev, &pci, NULL, &tmp_cp)) != VK_SUCCESS)
         goto fail;
      cd->pools[q] = tmp_cp;

      VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VkFence tmp_f = VK_NULL_HANDLE;
      if ((vr = vkCreateFence(cd->dev, &fci, NULL, &tmp_f)) != VK_SUCCESS)
         goto fail;
      cd->fences[q] = tmp_f;
   }

   if (want_compute) {
      if ((vr = create_compute_pipeline(cd->dev, &cd->dsl, &cd->comp_layout, &cd->comp_pipe)) != VK_SUCCESS)
         goto fail;
      VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nqueues};
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = (uint32_t)nqueues,
         .poolSizeCount = 1,
         .pPoolSizes = &dps,
      };
      VkDescriptorPool tmp_dp = VK_NULL_HANDLE;
      if ((vr = vkCreateDescriptorPool(cd->dev, &dpci, NULL, &tmp_dp)) != VK_SUCCESS)
         goto fail;
      cd->dpool = tmp_dp;

      for (int q = 0; q < nqueues; q++) {
         void *ptr = NULL;
         if ((vr = alloc_host_buffer(si, cd->dev, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     &cd->bufs[q], &cd->buf_mems[q], &ptr)) != VK_SUCCESS)
            goto fail;
         cd->buf_maps[q] = (uint32_t *)ptr;

         VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = cd->dpool,
            .descriptorSetCount = 1,
            .pSetLayouts = &cd->dsl,
         };
         VkDescriptorSet tmp_ds = VK_NULL_HANDLE;
         if ((vr = vkAllocateDescriptorSets(cd->dev, &dsai, &tmp_ds)) != VK_SUCCESS)
            goto fail;
         cd->dsets[q] = tmp_ds;

         VkDescriptorBufferInfo dbi = {cd->bufs[q], 0, VK_WHOLE_SIZE};
         VkWriteDescriptorSet w = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = cd->dsets[q],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi,
         };
         vkUpdateDescriptorSets(cd->dev, 1, &w, 0, NULL);
      }
   }

   if (want_render) {
      VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
      if ((vr = create_graphics_pipeline(cd->dev, fmt, &cd->gfx_layout, &cd->gfx_pipe)) != VK_SUCCESS)
         goto fail;
      for (int q = 0; q < nqueues; q++) {
         if ((vr = alloc_color_image(si, cd->dev, fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                     &cd->imgs[q], &cd->img_mems[q], &cd->img_views[q])) != VK_SUCCESS)
            goto fail;
      }
   }

   return VK_SUCCESS;

fail:
   cycle_dev_destroy(cd, true);
   return vr;
}

static void
record_render_pass(VkCommandBuffer cb, const struct cycle_dev *cd, int q, uint32_t seed)
{
   VkImageMemoryBarrier2 ib = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .image = cd->imgs[q],
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &ib,
   };
   vkCmdPipelineBarrier2(cb, &di);

   VkRenderingAttachmentInfo ca = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = cd->img_views[q],
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.1f, 0.2f, 0.3f, 1.0f},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {REBIND_LIGHT_DIM, REBIND_LIGHT_DIM}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ca,
   };
   vkCmdBeginRendering(cb, &ri);
   VkViewport vp = {0, 0, (float)REBIND_LIGHT_DIM, (float)REBIND_LIGHT_DIM, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {REBIND_LIGHT_DIM, REBIND_LIGHT_DIM}};
   vkCmdSetViewport(cb, 0, 1, &vp);
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, cd->gfx_pipe);
   float color[4] = {(float)((seed & 0xffu) / 255.0), 0.5f, 0.25f, 1.0f};
   vkCmdPushConstants(cb, cd->gfx_layout,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(color), color);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

static void
record_compute_dispatch(VkCommandBuffer cb, const struct cycle_dev *cd, int q, uint32_t seed)
{
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cd->comp_pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cd->comp_layout, 0, 1, &cd->dsets[q], 0, NULL);
   vkCmdPushConstants(cb, cd->comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seed), &seed);
   vkCmdDispatch(cb, 1, 1, 1);
   VkBufferMemoryBarrier2 bmb = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
      .buffer = cd->bufs[q],
      .offset = 0,
      .size = REBIND_WORDS * sizeof(uint32_t),
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &bmb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

enum churn_kind {
   CHURN_COMPUTE = 0,
   CHURN_RENDER,
};

struct churn_worker {
   pthread_t th;
   int tid;
   const struct shared_inst *si;
   enum churn_kind kind;
   int nqueues;
   uint64_t t_start;
   uint64_t warm_ns;
   uint64_t win_ns;
   int nwin;
   atomic_bool *abort_flag;

   uint64_t win_cycles[GB_MAX_WIN];
   struct gb_samples create_us;
   struct gb_samples submit_us;
   struct gb_samples destroy_us;
   uint64_t measured_cycles;
   bool failed;
   bool check_bad;
   bool leaked_dev;
   uint32_t last_got_crc;
   uint32_t last_exp_crc;
   char err[128];
};

static void *
churn_worker_main(void *arg)
{
   struct churn_worker *w = (struct churn_worker *)arg;
   const bool want_render = (w->kind == CHURN_RENDER);
   const bool want_compute = (w->kind == CHURN_COMPUTE);
   const uint64_t t_meas = w->t_start + w->warm_ns;
   const uint64_t t_end = t_meas + (uint64_t)w->nwin * w->win_ns;

   gb_sleep_until_ns(w->t_start);

   uint64_t cycle = 0;
   while (!atomic_load_explicit(w->abort_flag, memory_order_relaxed)) {
      if (gb_now_ns() >= t_end)
         break;

      struct cycle_dev cd;
      uint64_t tc0 = gb_now_ns();
      VkResult vr = cycle_dev_create(w->si, w->nqueues, want_render, want_compute, &cd);
      uint64_t tc1 = gb_now_ns();
      if (vr != VK_SUCCESS) {
         snprintf(w->err, sizeof(w->err), "vkCreateDevice/setup failed: %d after %llu cycles",
                  (int)vr, (unsigned long long)cycle);
         w->failed = true;
         atomic_store_explicit(w->abort_flag, true, memory_order_relaxed);
         break;
      }

      VkCommandBuffer cbs[REBIND_MAX_QUEUES] = {0};
      uint32_t seeds[REBIND_MAX_QUEUES] = {0};
      bool submitted_pending = false;
      bool device_lost = false;
      int submitted_queues = 0;

      for (int q = 0; q < w->nqueues; q++) {
         VkCommandBufferAllocateInfo cbai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cd.pools[q],
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
         };
         vr = vkAllocateCommandBuffers(cd.dev, &cbai, &cbs[q]);
         if (vr != VK_SUCCESS) {
            snprintf(w->err, sizeof(w->err), "vkAllocateCommandBuffers failed: %d", (int)vr);
            w->failed = true;
            device_lost = (vr == VK_ERROR_DEVICE_LOST);
            break;
         }
         uint32_t seed = (uint32_t)((cycle * 131u) ^ ((uint64_t)w->tid << 20) ^ ((uint64_t)q << 12)) + 1u;
         seeds[q] = seed;
         VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
         };
         if ((vr = vkBeginCommandBuffer(cbs[q], &bi)) != VK_SUCCESS) {
            snprintf(w->err, sizeof(w->err), "vkBeginCommandBuffer failed: %d", (int)vr);
            w->failed = true;
            device_lost = (vr == VK_ERROR_DEVICE_LOST);
            break;
         }
         if (want_compute)
            record_compute_dispatch(cbs[q], &cd, q, seed);
         else
            record_render_pass(cbs[q], &cd, q, seed);
         if ((vr = vkEndCommandBuffer(cbs[q])) != VK_SUCCESS) {
            snprintf(w->err, sizeof(w->err), "vkEndCommandBuffer failed: %d", (int)vr);
            w->failed = true;
            device_lost = (vr == VK_ERROR_DEVICE_LOST);
            break;
         }
      }

      uint64_t ts0 = gb_now_ns();
      if (!w->failed) {
         for (int q = 0; q < w->nqueues; q++) {
            VkSubmitInfo si = {
               .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
               .commandBufferCount = 1,
               .pCommandBuffers = &cbs[q],
            };
            vr = vkQueueSubmit(cd.queues[q], 1, &si, cd.fences[q]);
            if (vr != VK_SUCCESS) {
               snprintf(w->err, sizeof(w->err), "vkQueueSubmit failed: %d after %llu cycles",
                        (int)vr, (unsigned long long)cycle);
               w->failed = true;
               device_lost = (vr == VK_ERROR_DEVICE_LOST);
               if (device_lost)
                  submitted_pending = true;
               break;
            }
            submitted_queues++;
            submitted_pending = true;
         }
      }
      if (!w->failed) {
         vr = vkWaitForFences(cd.dev, (uint32_t)submitted_queues, cd.fences, VK_TRUE, REBIND_TIMEOUT_NS);
         if (vr != VK_SUCCESS) {
            device_lost = (vr == VK_ERROR_DEVICE_LOST);
            snprintf(w->err, sizeof(w->err), "%s (%d) after %llu cycles",
                     device_lost ? "VK_ERROR_DEVICE_LOST"
                     : vr == VK_TIMEOUT ? "GPU wait timed out"
                     : "vkWaitForFences failed",
                     (int)vr, (unsigned long long)cycle);
            w->failed = true;
         } else {
            submitted_pending = false;
         }
      } else if (submitted_pending && !device_lost) {
         /* Drain earlier successful submissions before destroying resources. */
         VkResult wr = vkWaitForFences(cd.dev, (uint32_t)submitted_queues, cd.fences, VK_TRUE, REBIND_TIMEOUT_NS);
         if (wr == VK_SUCCESS)
            submitted_pending = false;
         else if (wr == VK_ERROR_DEVICE_LOST)
            device_lost = true;
      }
      uint64_t ts1 = gb_now_ns();

      if (!w->failed && want_compute) {
         for (int q = 0; q < w->nqueues; q++) {
            uint32_t want[REBIND_WORDS];
            for (uint32_t i = 0; i < REBIND_WORDS; i++)
               want[i] = seeds[q] + i;
            uint32_t got_crc = gb_crc32(cd.buf_maps[q], sizeof(want));
            uint32_t exp_crc = gb_crc32(want, sizeof(want));
            w->last_got_crc = got_crc;
            w->last_exp_crc = exp_crc;
            if (got_crc != exp_crc || memcmp(cd.buf_maps[q], want, sizeof(want)) != 0) {
               snprintf(w->err, sizeof(w->err), "compute output mismatch on q=%d cycle=%llu (crc %08x != %08x)",
                        q, (unsigned long long)cycle, got_crc, exp_crc);
               w->check_bad = true;
               w->failed = true;
               break;
            }
         }
      }

      uint64_t td0 = gb_now_ns();
      if (!submitted_pending) {
         cycle_dev_destroy(&cd, device_lost);
      } else if (device_lost) {
         /* Per Vulkan 1.3 lost-device rules, vkDeviceWaitIdle establishes
          * completion of pending submissions before destroying handles. */
         (void)vkDeviceWaitIdle(cd.dev);
         cycle_dev_destroy(&cd, true);
      } else {
         /* Pending work with unknown completion (e.g. VK_TIMEOUT): leave the
          * device and its parent VkInstance alive until process exit. */
         w->leaked_dev = true;
      }
      uint64_t td1 = gb_now_ns();

      if (w->failed) {
         atomic_store_explicit(w->abort_flag, true, memory_order_relaxed);
         break;
      }

      cycle++;
      if (td1 >= t_meas && td1 < t_end) {
         int wi = (int)((td1 - t_meas) / w->win_ns);
         if (wi >= 0 && wi < w->nwin) {
            w->win_cycles[wi]++;
            w->measured_cycles++;
            gb_samples_push(&w->create_us, (double)(tc1 - tc0) / 1e3);
            gb_samples_push(&w->submit_us, (double)(ts1 - ts0) / 1e3);
            gb_samples_push(&w->destroy_us, (double)(td1 - td0) / 1e3);
         }
      }
   }

   return NULL;
}

static int
run_queue_churn(const struct gb_opts *o, enum churn_kind kind, const char *kind_s, int nqueues)
{
   struct shared_inst si;
   if (!shared_inst_init(&si, o->device)) {
      char pk[32], pq[32];
      snprintf(pk, sizeof(pk), "kind=%s", kind_s);
      snprintf(pq, sizeof(pq), "queues=%d", nqueues);
      const char *const extra[] = {pk, pq, NULL};
      gb_result_simple(o, nqueues > 1 ? GB_MODE_QUEUES : GB_MODE_QUEUE, nqueues, extra, "fail", si.err);
      return 1;
   }

   int depths[GB_MAX_DEPTHS];
   int nd = 0;
   if (o->ndepths > 0) {
      for (int i = 0; i < o->ndepths; i++)
         depths[nd++] = o->depths[i];
   } else {
      depths[0] = 1;
      depths[1] = 4;
      nd = 2;
   }

   const uint64_t warm_ns = (uint64_t)o->warmup_ms * 1000000ull;
   const uint64_t win_ns = (uint64_t)o->run_ms * 1000000ull;
   const double win_us = (double)o->run_ms * 1000.0;
   bool any_leaked = false;

   for (int di = 0; di < nd; di++) {
      int nthreads = depths[di];
      char pk[32], pq[32], pw[32];
      snprintf(pk, sizeof(pk), "kind=%s", kind_s);
      snprintf(pq, sizeof(pq), "queues=%d", nqueues);
      snprintf(pw, sizeof(pw), "workers=%d", nthreads);
      const char *const extra_params[] = {pk, pq, pw, NULL};

      if (nthreads < 1 || nthreads > REBIND_MAX_WORKERS) {
         gb_result_simple(o, GB_MODE_QUEUES, nthreads, extra_params, "fail", "depth (workers) must be 1..8");
         continue;
      }
      const int total_depth = nthreads * nqueues;
      const enum gb_mode mode = (total_depth > 1) ? GB_MODE_QUEUES : GB_MODE_QUEUE;
      if ((uint32_t)nqueues > si.family_queues) {
         char why[64];
         snprintf(why, sizeof(why), "device has %u queues (need %d)", si.family_queues, nqueues);
         gb_result_simple(o, mode, total_depth, extra_params, "skip", why);
         continue;
      }

      struct churn_worker ws[REBIND_MAX_WORKERS];
      memset(ws, 0, sizeof(ws));
      atomic_bool abort_flag = false;
      uint64_t t_start = gb_now_ns() + 20000000ull;

      for (int t = 0; t < nthreads; t++) {
         ws[t].tid = t;
         ws[t].si = &si;
         ws[t].kind = kind;
         ws[t].nqueues = nqueues;
         ws[t].t_start = t_start;
         ws[t].warm_ns = warm_ns;
         ws[t].win_ns = win_ns;
         ws[t].nwin = o->runs;
         ws[t].abort_flag = &abort_flag;
      }

      bool spawn_failed = false;
      if (nthreads == 1) {
         churn_worker_main(&ws[0]);
      } else {
         int created = 0;
         for (int t = 0; t < nthreads; t++) {
            if (pthread_create(&ws[t].th, NULL, churn_worker_main, &ws[t]) != 0) {
               spawn_failed = true;
               atomic_store_explicit(&abort_flag, true, memory_order_relaxed);
               break;
            }
            created++;
         }
         for (int t = 0; t < created; t++)
            pthread_join(ws[t].th, NULL);
      }

      bool failed = spawn_failed;
      bool check_bad = false;
      char err[128] = {0};
      if (spawn_failed)
         snprintf(err, sizeof(err), "pthread_create failed");
      uint32_t got_crc = 0, exp_crc = 0;
      uint32_t bad_got_crc = 0, bad_exp_crc = 0;
      uint64_t measured_cycles = 0;
      struct gb_samples cycle_rate = {0}, queue_rate = {0};
      struct gb_samples create_all = {0}, submit_all = {0}, destroy_all = {0};

      for (int t = 0; t < nthreads; t++) {
         measured_cycles += ws[t].measured_cycles;
         if (ws[t].leaked_dev)
            any_leaked = true;
         if (ws[t].last_exp_crc) {
            got_crc = ws[t].last_got_crc;
            exp_crc = ws[t].last_exp_crc;
         }
         if (ws[t].failed) {
            if (ws[t].check_bad && !check_bad) {
               check_bad = true;
               bad_got_crc = ws[t].last_got_crc;
               bad_exp_crc = ws[t].last_exp_crc;
            }
            if (!failed) {
               failed = true;
               snprintf(err, sizeof(err), "%s", ws[t].err);
            }
         }
         gb_samples_append(&create_all, &ws[t].create_us);
         gb_samples_append(&submit_all, &ws[t].submit_us);
         gb_samples_append(&destroy_all, &ws[t].destroy_us);
      }

      for (int wi = 0; wi < o->runs; wi++) {
         uint64_t c = 0;
         for (int t = 0; t < nthreads; t++)
            c += ws[t].win_cycles[wi];
         if (c > 0) {
            double us_cyc = win_us / (double)c;
            gb_samples_push(&cycle_rate, us_cyc);
            gb_samples_push(&queue_rate, us_cyc / (double)nqueues);
         }
      }

      struct gb_result r;
      gb_result_begin(&r, o, mode, total_depth, extra_params);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (failed || !cycle_rate.n) {
         if (!err[0])
            snprintf(err, sizeof(err), "no completed cycles in measured windows");
         gb_result_end_metrics(&r);
         if (check_bad) {
            struct gb_check chk = {.done = true, .valid = false, .crc = bad_got_crc, .expected = bad_exp_crc};
            gb_result_check(&r, &chk);
         }
         gb_result_finish(&r, "fail", NULL, err);
      } else {
         gb_result_metric(&r, "us_per_cycle", "us", true, &cycle_rate, true);
         gb_result_metric(&r, "us_per_queue", "us", true, &queue_rate, true);
         gb_result_metric(&r, "create_us", "us", true, &create_all, false);
         gb_result_metric(&r, "submit_us", "us", true, &submit_all, false);
         gb_result_metric(&r, "destroy_us", "us", true, &destroy_all, false);
         gb_result_metric1(&r, "cycles", "count", false, (double)measured_cycles);
         gb_result_end_metrics(&r);
         if (kind == CHURN_COMPUTE) {
            struct gb_check chk = {.done = true, .valid = true, .crc = got_crc, .expected = exp_crc};
            gb_result_check(&r, &chk);
         }
         gb_result_finish(&r, "ok", "us_per_cycle", NULL);
      }

      gb_samples_free(&cycle_rate);
      gb_samples_free(&queue_rate);
      gb_samples_free(&create_all);
      gb_samples_free(&submit_all);
      gb_samples_free(&destroy_all);
      for (int t = 0; t < nthreads; t++) {
         gb_samples_free(&ws[t].create_us);
         gb_samples_free(&ws[t].submit_us);
         gb_samples_free(&ws[t].destroy_us);
      }
      if (any_leaked)
         break;
   }

   if (!any_leaked)
      shared_inst_destroy(&si);
   return 0;
}

int
gb_run_queue_rebind(const struct gb_opts *o)
{
   static const char *const known[] = {"queues", "inflight", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_PROCS, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   char err[128];
   if (!validate_run_controls(o, err, sizeof(err))) {
      gb_result_simple(o, GB_MODE_PROCS, 1, NULL, "fail", err);
      return 1;
   }
   if (o->mode == GB_MODE_QUEUES) {
      gb_result_simple(o, GB_MODE_PROCS, 1, NULL, "fail", "queue.rebind runs in procs mode only");
      return 1;
   }
   long queues = 0, inflight = 0;
   if (!parse_strict_long(o, "queues", 7, &queues) || queues < 1 || queues > REBIND_MAX_QUEUES) {
      gb_result_simple(o, GB_MODE_PROCS, 1, NULL, "fail", "queues must be an integer 1..8");
      return 1;
   }
   if (!parse_strict_long(o, "inflight", 2, &inflight) || inflight < 1 || inflight > REBIND_MAX_INFLIGHT) {
      gb_result_simple(o, GB_MODE_PROCS, 1, NULL, "fail", "inflight must be an integer 1..4");
      return 1;
   }
   return run_queue_rebind_procs(o, (int)queues, (int)inflight);
}

int
gb_run_queue_churn(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "queues", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   char err[128];
   if (!validate_run_controls(o, err, sizeof(err))) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", err);
      return 1;
   }
   if (o->mode == GB_MODE_PROCS) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "queue.churn does not support procs mode");
      return 1;
   }
   const char *kind_s = gb_param(o, "kind", "compute");
   enum churn_kind kind;
   if (!strcmp(kind_s, "compute"))
      kind = CHURN_COMPUTE;
   else if (!strcmp(kind_s, "render"))
      kind = CHURN_RENDER;
   else {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be compute or render");
      return 1;
   }
   long queues = 0;
   if (!parse_strict_long(o, "queues", 4, &queues) || queues < 1 || queues > REBIND_MAX_QUEUES) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "queues must be an integer 1..8");
      return 1;
   }
   return run_queue_churn(o, kind, kind_s, (int)queues);
}
