// SPDX-License-Identifier: GPL-2.0-only
/* Vulkan helpers for the rt executor: device bring-up with the ray-tracing features, buffers,
 * images, acceleration-structure entry points, one-shot submits and pipeline statistics. */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <vulkan/vulkan.h>

#define VKC(x)                                                                                   \
   do {                                                                                          \
      VkResult r_ = (x);                                                                         \
      if (r_ != VK_SUCCESS) {                                                                    \
         fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, r_);                        \
         return r_;                                                                              \
      }                                                                                          \
   } while (0)

typedef struct vk_buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   VkDeviceSize size;
   void *map;
   VkDeviceAddress addr;
} vk_buf;

typedef struct vk_img {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   VkFormat fmt;
   uint32_t w, h;
} vk_img;

typedef struct vk_ctx {
   VkInstance inst;
   VkPhysicalDevice phys;
   VkDevice dev;
   VkQueue queue;
   uint32_t qfam;
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties memprops;
   VkPhysicalDeviceAccelerationStructurePropertiesKHR asprops;
   VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops;
   VkPhysicalDeviceDriverProperties driver;
   int has_as, has_rq, has_rtp, has_exec_props, has_wayland;
   int exec_props_enabled;
   VkCommandPool pool;  /* for one-shot submits */
   PFN_vkCreateAccelerationStructureKHR CreateAS;
   PFN_vkDestroyAccelerationStructureKHR DestroyAS;
   PFN_vkGetAccelerationStructureBuildSizesKHR GetASBuildSizes;
   PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAS;
   PFN_vkGetAccelerationStructureDeviceAddressKHR GetASAddress;
   PFN_vkCmdWriteAccelerationStructuresPropertiesKHR CmdWriteASProps;
   PFN_vkCreateRayTracingPipelinesKHR CreateRTPipelines;
   PFN_vkGetRayTracingShaderGroupHandlesKHR GetRTGroupHandles;
   PFN_vkCmdTraceRaysKHR CmdTraceRays;
   PFN_vkGetPipelineExecutablePropertiesKHR GetExecProps;
   PFN_vkGetPipelineExecutableStatisticsKHR GetExecStats;
} vk_ctx;

/* Creates the instance and the device. want_rtp: also enable the ray-tracing pipeline;
 * want_stats: enable pipeline executable statistics when available; window: Wayland WSI. */
VkResult vk_ctx_init(vk_ctx *c, int device_index, int want_rtp, int want_stats, int window);
void vk_ctx_destroy(vk_ctx *c);
/* Lists devices and features as one JSON object (the executor's --info). */
void vk_print_info(int device_index);

VkResult vk_buf_create(vk_ctx *c, vk_buf *b, VkDeviceSize size, VkBufferUsageFlags usage, int host);
void vk_buf_destroy(vk_ctx *c, vk_buf *b);
VkResult vk_img_create(vk_ctx *c, vk_img *im, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                       VkImageAspectFlags aspect);
void vk_img_destroy(vk_ctx *c, vk_img *im);
VkResult vk_shader(vk_ctx *c, const uint32_t *code, size_t size, VkShaderModule *out);

VkResult vk_oneshot_begin(vk_ctx *c, VkCommandBuffer *cb);
VkResult vk_oneshot_end(vk_ctx *c, VkCommandBuffer cb);   /* submits and waits */

/* Global memory barrier helper (sync2). */
void vk_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 srca, VkPipelineStageFlags2 dst,
                VkAccessFlags2 dsta);
void vk_image_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from,
                      VkImageLayout to, VkPipelineStageFlags2 src, VkAccessFlags2 srca,
                      VkPipelineStageFlags2 dst, VkAccessFlags2 dsta);

/* Pipeline statistics (VK_KHR_pipeline_executable_properties) as a JSON object under key. */
struct rt_json;
void vk_pipeline_stats_json(vk_ctx *c, VkPipeline p, struct rt_json *j, const char *key);

/* With --shader-stats: print "GB_PIPELINE <name>" on stderr before creating a pipeline, so the
 * compiler's AGX_MESA_DEBUG=shaders dump can be attributed (same convention as the suite). */
extern int vk_pipeline_markers;
void vk_marker(const char *name);

/* Embedded SPIR-V (generated at build time). */
typedef struct { const char *name; const uint32_t *code; size_t size; } vk_spv;
const vk_spv *vk_find_spv(const char *name);
