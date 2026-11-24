// SPDX-License-Identifier: GPL-2.0-only
/* gb_vk: Vulkan context, resource helpers and the submission-stream engine
 * shared by the gpubench Vulkan executors. */
#pragma once

#include <vulkan/vulkan.h>

/* Compatibility shims for distro Vulkan headers older than 1.3.279 (e.g. Ubuntu
 * 24.04 libvulkan-dev 1.3.275), which predate VK_KHR_dynamic_rendering_local_read. */
#ifndef VK_KHR_dynamic_rendering_local_read
#define VK_KHR_dynamic_rendering_local_read 1
#define VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_SPEC_VERSION 1
#define VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME "VK_KHR_dynamic_rendering_local_read"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR ((VkStructureType)1000232000)
#define VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR ((VkStructureType)1000232001)
#define VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR ((VkStructureType)1000232002)
#define VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR ((VkImageLayout)1000232000)
typedef struct VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR {
   VkStructureType sType;
   void *pNext;
   VkBool32 dynamicRenderingLocalRead;
} VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR;
typedef struct VkRenderingAttachmentLocationInfoKHR {
   VkStructureType sType;
   const void *pNext;
   uint32_t colorAttachmentCount;
   const uint32_t *pColorAttachmentLocations;
} VkRenderingAttachmentLocationInfoKHR;
typedef struct VkRenderingInputAttachmentIndexInfoKHR {
   VkStructureType sType;
   const void *pNext;
   uint32_t colorAttachmentCount;
   const uint32_t *pColorAttachmentInputIndices;
   const uint32_t *pDepthInputAttachmentIndex;
   const uint32_t *pStencilInputAttachmentIndex;
} VkRenderingInputAttachmentIndexInfoKHR;
typedef void (VKAPI_PTR *PFN_vkCmdSetRenderingAttachmentLocationsKHR)(
   VkCommandBuffer commandBuffer, const VkRenderingAttachmentLocationInfoKHR *pLocationInfo);
typedef void (VKAPI_PTR *PFN_vkCmdSetRenderingInputAttachmentIndicesKHR)(
   VkCommandBuffer commandBuffer, const VkRenderingInputAttachmentIndexInfoKHR *pInputAttachmentIndexInfo);
#endif

#include "gb_common.h"

#define GB_MAX_QUEUES 16

struct gb_vk {
   VkInstance instance;
   VkPhysicalDevice pdev;
   VkDevice dev;
   uint32_t qfam;
   uint32_t family_queue_count;
   int nqueues;
   VkQueue queues[GB_MAX_QUEUES];
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties memprops;
   double ts_period_ns;
   uint64_t ts_mask;
   char driver_name[VK_MAX_DRIVER_NAME_SIZE];
   char driver_info[VK_MAX_DRIVER_INFO_SIZE];
   uint32_t mem_host; /* HOST_VISIBLE | HOST_COHERENT, device-local when available */
   uint32_t mem_dev;  /* DEVICE_LOCAL */
   VkCommandPool oneshot_pool;
   /* Optional Vulkan 1.2/1.3/1.4 features queried and enabled when present. */
   bool has_local_read;
   bool has_draw_indirect_count;
   bool has_descriptor_indexing;
   bool has_float16;
   bool has_8bit_storage;
   bool has_buffer_device_address;
   bool has_descriptor_buffer;
   bool has_subgroup_ballot;
   bool has_subgroup_arithmetic;
   uint32_t subgroup_size;
   uint32_t max_sampled_images;
   VkDeviceSize desc_buf_offset_align;
   size_t ubo_desc_size;
   PFN_vkCmdSetRenderingAttachmentLocationsKHR cmd_set_attachment_locations;
   PFN_vkCmdSetRenderingInputAttachmentIndicesKHR cmd_set_input_attachment_indices;
#ifdef VK_EXT_descriptor_buffer
   PFN_vkGetDescriptorSetLayoutSizeEXT get_dsl_size;
   PFN_vkGetDescriptorSetLayoutBindingOffsetEXT get_dsl_binding_offset;
   PFN_vkGetDescriptorEXT get_descriptor;
   PFN_vkCmdBindDescriptorBuffersEXT cmd_bind_desc_bufs;
   PFN_vkCmdSetDescriptorBufferOffsetsEXT cmd_set_desc_buf_offsets;
#endif
};

#define GB_VK_CHECK(call)                                                        \
   do {                                                                          \
      VkResult r_ = (call);                                                      \
      if (r_ != VK_SUCCESS)                                                      \
         gb_die("%s:%d: %s failed: %d", __FILE__, __LINE__, #call, (int)r_);   \
   } while (0)

/* Set before gb_vk_create to enable VK_KHR_pipeline_executable_properties. */
extern bool gb_vk_want_pipeline_stats;

/* Creates instance + device with `nqueues` queues of the first graphics+compute
 * family. Returns 0, or -1 when the device has fewer queues (vk is then
 * still valid with family_queue_count set and no device). */
int gb_vk_create(struct gb_vk *vk, int device_index, int nqueues);
void gb_vk_destroy(struct gb_vk *vk);
/* Prints one JSON object describing the device and driver. */
int gb_vk_print_info(int device_index);
/* Queue count of the family without creating a device. */
int gb_vk_probe_queue_count(int device_index);

/* ---- resources ---- */
struct gb_buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   VkDeviceSize size;
   void *map;
   VkDeviceAddress addr;
};
void gb_buf_create(struct gb_vk *vk, VkDeviceSize size, VkBufferUsageFlags usage, bool host,
                   struct gb_buf *b);
void gb_buf_create_addr(struct gb_vk *vk, VkDeviceSize size, VkBufferUsageFlags usage, bool host,
                        struct gb_buf *b);
void gb_buf_destroy(struct gb_vk *vk, struct gb_buf *b);

struct gb_img {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   uint32_t w, h;
   VkFormat fmt;
};
void gb_img_create(struct gb_vk *vk, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                   struct gb_img *im);
void gb_img_create_ex(struct gb_vk *vk, uint32_t w, uint32_t h, VkFormat fmt, VkSampleCountFlagBits samples,
                      VkImageUsageFlags usage, VkImageAspectFlags aspect, struct gb_img *im);
void gb_img_destroy(struct gb_vk *vk, struct gb_img *im);
/* Readback helpers (one-shot transfer to host buffer, waits on queue 0). */
void gb_buf_readback(struct gb_vk *vk, const struct gb_buf *src, void *dst, size_t bytes);
void gb_buf_upload(struct gb_vk *vk, const void *src, struct gb_buf *dst, size_t bytes);
void gb_img_readback(struct gb_vk *vk, const struct gb_img *im, VkImageLayout cur_layout, void *dst, size_t bpp);
/* Image barrier helper (sync2). */
void gb_img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

VkCommandBuffer gb_oneshot_begin(struct gb_vk *vk);
void gb_oneshot_end(struct gb_vk *vk, VkCommandBuffer cb); /* submits on queue 0 and waits */

/* ---- pipelines ---- */
struct gb_pipe {
   VkPipeline pipe;
   VkPipelineLayout layout;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkShaderStageFlags push_stages;
   uint32_t push_size;
};
VkShaderModule gb_shader(struct gb_vk *vk, const uint32_t *code, size_t bytes);
/* Compute pipeline with `nbuf` storage buffers at set 0, bindings 0..nbuf-1. */
void gb_compute_pipe(struct gb_vk *vk, const uint32_t *code, size_t bytes, int nbuf,
                     uint32_t push_size, struct gb_pipe *p);
void gb_pipe_set_buffers(struct gb_vk *vk, struct gb_pipe *p, struct gb_buf *const *bufs, int n);
enum gb_blend { GB_BLEND_NONE = 0, GB_BLEND_ALPHA };
/* Graphics pipeline for dynamic rendering into one color attachment; no
 * vertex input, dynamic viewport/scissor, push constants to VS+FS. */
void gb_graphics_pipe(struct gb_vk *vk, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs,
                      size_t fs_bytes, VkFormat fmt, enum gb_blend blend, uint32_t push_size,
                      struct gb_pipe *p);
/* Graphics variant with a combined image sampler at set 0, binding 0. */
void gb_texture_pipe(struct gb_vk *vk, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs,
                     size_t fs_bytes, VkFormat fmt, struct gb_pipe *p);
void gb_pipe_destroy(struct gb_vk *vk, struct gb_pipe *p);
/* Begin dynamic rendering with a cleared color attachment, set viewport/scissor. */
void gb_begin_render(VkCommandBuffer cb, const struct gb_img *im, const float clear[4]);

/* ---- submission streams ----
 * A stream is a ring of `depth` command buffers on one queue with a timeline
 * semaphore. Each submission is recorded fresh. See suite/README.md for the
 * timing method. */
typedef void (*gb_record_fn)(void *user, VkCommandBuffer cb, uint64_t k);
typedef void (*gb_done_fn)(void *user, uint64_t k, int slot);

struct gb_stream {
   struct gb_vk *vk;
   VkQueue queue;
   int depth;
   VkCommandPool *pools;
   VkCommandBuffer *cbs;
   uint64_t *submit_ns;
   VkSemaphore tl;
   uint64_t value;
   VkQueryPool qp;
   gb_record_fn record;
   gb_done_fn done;
   void *user;
   bool failed;
};



void gb_stream_init(struct gb_stream *s, struct gb_vk *vk, int queue_index, int depth,
                    gb_record_fn record, gb_done_fn done, void *user);
void gb_stream_finish(struct gb_stream *s);
/* Runs the stream from absolute time t0: warm-up, then nwin windows. Returns
 * 0, or -1 on a GPU timeout (pass->failed). */
int gb_stream_run(struct gb_stream *s, bool timed, uint64_t t0, uint64_t warmup_ns, int nwin,
                  uint64_t win_ns, struct gb_pass *out);
/* One submission, round trip in microseconds (for cold first-use costs). */
double gb_stream_once(struct gb_stream *s);

/* ---- stream tests ----
 * A stream test supplies setup/teardown and the per-submission recording;
 * gb_run_stream_test runs it over the depth list in the requested mode. */
struct gb_stream_test {
   const char *name;
   const char *const *known_params;
   int default_depths[GB_MAX_DEPTHS];
   int ndefault;
   bool report_first; /* add first_us (cold first submission) to the first result */
   void *(*setup)(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t errlen);
   void (*teardown)(struct gb_vk *vk, void *state);
   gb_record_fn record;
   gb_done_fn done;
   /* derived throughput metrics from per-window us_per_submit */
   void (*derive)(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us_per_submit);
   bool no_timed; /* never run the timed pass (the test has no GPU work to time) */
   /* Optional: compare the output of the submissions just run with the
    * test's reference. Called with the queue idle, once per result. */
   void (*check)(struct gb_vk *vk, void *state, struct gb_check *chk);
};
int gb_run_stream_test(const struct gb_opts *o, const struct gb_stream_test *t);
