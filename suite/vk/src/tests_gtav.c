// SPDX-License-Identifier: GPL-2.0-only
/* GTA V / D3D12-on-Vulkan driver bottleneck microbenchmarks:
 *   render.storage_meta_discard  (repeated UNDEFINED->GENERAL compressed storage imageStore)
 *   render.aliased_heap_pipeline (VK_IMAGE_CREATE_ALIAS_BIT placed targets + layout decay + ALL_COMMANDS) */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>

#include "gb_vk.h"
#include "shaders.h"
#include "tests.h"

static bool
parse_gtav_res(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "native")) *w = 3024, *h = 1964;
   else if (!strcmp(s, "half")) *w = 1512, *h = 982;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else return false;
   return true;
}

static uint32_t
find_mem_type(struct gb_vk *vk, uint32_t type_bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++) {
      if ((type_bits & (1u << i)) &&
          (vk->memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   gb_die("no matching memory type (bits=0x%x want=0x%x)", type_bits, want);
   return 0;
}

/* ==================== render.storage_meta_discard ==================== */

struct meta_discard_state {
   uint32_t w, h, passes;
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   struct gb_pipe pipe;
};

static void *
meta_discard_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *res_s = gb_param(o, "res", "half");
   long passes = gb_param_int(o, "passes", 16);
   uint32_t w, h;
   if (!parse_gtav_res(res_s, &w, &h)) {
      snprintf(err, n, "bad res %s (native|half|1080p|720p)", res_s);
      return NULL;
   }
   if (passes < 1 || passes > 128) {
      snprintf(err, n, "passes must be 1..128");
      return NULL;
   }
   struct meta_discard_state *st = calloc(1, sizeof(*st));
   st->w = w;
   st->h = h;
   st->passes = (uint32_t)passes;

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(vk->dev, st->img, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = find_mem_type(vk, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &mai, NULL, &st->mem));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->img, st->mem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = st->img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   GB_VK_CHECK(vkCreateImageView(vk->dev, &ivci, NULL, &st->view));

   VkDescriptorSetLayoutBinding b0 = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &b0,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->pipe.dsl));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &st->pipe.dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &plci, NULL, &st->pipe.layout));
   VkShaderModule sm = gb_shader(vk, GB_SPV(compress_store_comp));
   VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = sm,
         .pName = "main",
      },
      .layout = st->pipe.layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &st->pipe.pipe));
   vkDestroyShaderModule(vk->dev, sm, NULL);

   VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &dps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->pipe.dpool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = st->pipe.dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &st->pipe.dsl,
   };
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->pipe.set));
   VkDescriptorImageInfo dii = {
      .imageView = st->view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = st->pipe.set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &dii,
   };
   vkUpdateDescriptorSets(vk->dev, 1, &wds, 0, NULL);
   return st;
}

static void
meta_discard_record(void *user, VkCommandBuffer cb, uint64_t k)
{
   struct meta_discard_state *st = user;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipe.layout,
                           0, 1, &st->pipe.set, 0, NULL);
   uint32_t gx = (st->w + 15u) / 16u;
   uint32_t gy = (st->h + 15u) / 16u;
   for (uint32_t p = 0; p < st->passes; p++) {
      gb_img_barrier(cb, st->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
      struct { int w; int h; float k; float pad; } pc = {
         (int)st->w, (int)st->h, (float)((k + p) & 15u) * 0.0625f, 0.0f
      };
      vkCmdPushConstants(cb, st->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, gx, gy, 1);
   }
}

static void
meta_discard_teardown(struct gb_vk *vk, void *user)
{
   struct meta_discard_state *st = user;
   gb_pipe_destroy(vk, &st->pipe);
   vkDestroyImageView(vk->dev, st->view, NULL);
   vkDestroyImage(vk->dev, st->img, NULL);
   vkFreeMemory(vk->dev, st->mem, NULL);
   free(st);
}

static void
meta_discard_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long passes = gb_param_int(o, "passes", 16);
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, us->v[i] / (double)passes);
   gb_result_metric(r, "us_per_pass", "us", true, &per, false);
   gb_samples_free(&per);
}

static const char *const meta_discard_params[] = {"res", "passes", NULL};
static const struct gb_stream_test t_storage_meta_discard = {
   .name = "render.storage_meta_discard",
   .known_params = meta_discard_params,
   .default_depths = {1, 2},
   .ndefault = 2,
   .setup = meta_discard_setup,
   .teardown = meta_discard_teardown,
   .record = meta_discard_record,
   .derive = meta_discard_derive,
};

int
gb_run_render_storage_meta_discard(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_storage_meta_discard);
}

/* ==================== render.aliased_heap_pipeline ==================== */

struct aliased_heap_state {
   uint32_t w, h, passes;
   VkDeviceMemory heap;
   VkImage imgs[2];
   VkImageView views[2];
   struct gb_pipe pipe;
};

static void *
aliased_heap_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *res_s = gb_param(o, "res", "1080p");
   long passes = gb_param_int(o, "passes", 16);
   uint32_t w, h;
   if (!parse_gtav_res(res_s, &w, &h)) {
      snprintf(err, n, "bad res %s (native|half|1080p|720p)", res_s);
      return NULL;
   }
   if (passes < 2 || passes > 128) {
      snprintf(err, n, "passes must be 2..128");
      return NULL;
   }
   struct aliased_heap_state *st = calloc(1, sizeof(*st));
   st->w = w;
   st->h = h;
   st->passes = (uint32_t)passes;

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkMemoryRequirements mr[2];
   for (int i = 0; i < 2; i++) {
      GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->imgs[i]));
      vkGetImageMemoryRequirements(vk->dev, st->imgs[i], &mr[i]);
   }
   VkDeviceSize align = mr[1].alignment > 0 ? mr[1].alignment : 16384;
   VkDeviceSize off1 = (mr[0].size + align - 1) & ~(align - 1);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = off1 + mr[1].size,
      .memoryTypeIndex = find_mem_type(vk, mr[0].memoryTypeBits & mr[1].memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &mai, NULL, &st->heap));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->imgs[0], st->heap, 0));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->imgs[1], st->heap, off1));

   for (int i = 0; i < 2; i++) {
      VkImageViewCreateInfo ivci = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = st->imgs[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      GB_VK_CHECK(vkCreateImageView(vk->dev, &ivci, NULL, &st->views[i]));
   }
   gb_graphics_pipe(vk, GB_SPV(fs_vert), GB_SPV(color_frag),
                    VK_FORMAT_R8G8B8A8_UNORM, GB_BLEND_ALPHA, 16, &st->pipe);

   VkCommandBuffer cb = gb_oneshot_begin(vk);
   for (int i = 0; i < 2; i++) {
      gb_img_barrier(cb, st->imgs[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   }
   gb_oneshot_end(vk, cb);
   return st;
}

static void
aliased_heap_record(void *user, VkCommandBuffer cb, uint64_t k)
{
   struct aliased_heap_state *st = user;
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};
   for (uint32_t p = 0; p < st->passes; p++) {
      uint32_t idx = p & 1u;
      /* vkd3d-proton layout decay: GENERAL -> COLOR_ATTACHMENT before pass,
       * COLOR_ATTACHMENT -> GENERAL after pass, plus ALL_COMMANDS memory barrier. */
      gb_img_barrier(cb, st->imgs[idx], VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      VkRenderingAttachmentInfo att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->views[idx],
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = p < 2 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo ri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {st->w, st->h}},
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &att,
      };
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &vp);
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe.pipe);
      float col[4] = {(float)((k + p) & 7u) * 0.1f, 0.25f, 0.5f, 0.5f};
      vkCmdPushConstants(cb, st->pipe.layout, st->pipe.push_stages, 0, sizeof(col), col);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
      gb_img_barrier(cb, st->imgs[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
   }
}

static void
aliased_heap_teardown(struct gb_vk *vk, void *user)
{
   struct aliased_heap_state *st = user;
   gb_pipe_destroy(vk, &st->pipe);
   for (int i = 0; i < 2; i++) {
      vkDestroyImageView(vk->dev, st->views[i], NULL);
      vkDestroyImage(vk->dev, st->imgs[i], NULL);
   }
   vkFreeMemory(vk->dev, st->heap, NULL);
   free(st);
}

static void
aliased_heap_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long passes = gb_param_int(o, "passes", 16);
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, us->v[i] / (double)passes);
   gb_result_metric(r, "us_per_pass", "us", true, &per, false);
   gb_samples_free(&per);
}

static const char *const aliased_heap_params[] = {"res", "passes", NULL};
static const struct gb_stream_test t_aliased_heap_pipeline = {
   .name = "render.aliased_heap_pipeline",
   .known_params = aliased_heap_params,
   .default_depths = {1, 2},
   .ndefault = 2,
   .setup = aliased_heap_setup,
   .teardown = aliased_heap_teardown,
   .record = aliased_heap_record,
   .derive = aliased_heap_derive,
};

int
gb_run_render_aliased_heap_pipeline(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_aliased_heap_pipeline);
}

/* ==================== render.gbuf_pre_gfx ==================== */
/* Multi-CB 4-MRT + D32F G-buffer submission in a single vkQueueSubmit2:
 *   kind=direct:   N CBs continuing the same 4-MRT+D32F pass with LOAD_OP_LOAD
 *                  + STORE_OP_STORE (measures cross-CB render packing & union EOT/ZLS)
 *   kind=skinning: each CB runs a buffer-only compute dispatch + COMPUTE->VERTEX
 *                  memory barrier before continuing the same 4-MRT+D32F G-buffer pass
 *                  (measures pre-graphics / skinning compute hoisting across CBs). */

static VkPipeline
make_mrt4_depth_pipe(struct gb_vk *vk, VkPipelineLayout layout, bool depth_write)
{
   VkShaderModule vs = gb_shader(vk, GB_SPV(fs_vert));
   VkShaderModule fs = gb_shader(vk, GB_SPV(compress_mrt4_frag));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS};
   VkPipelineColorBlendAttachmentState cba[4];
   for (int i = 0; i < 4; i++) {
      cba[i] = (VkPipelineColorBlendAttachmentState){
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
   }
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 4, .pAttachments = cba};
   VkDynamicState dyn_s[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn_s};
   VkFormat fmts[4] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                       VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT};
   VkPipelineRenderingCreateInfo rci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 4, .pColorAttachmentFormats = fmts,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT};
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cb, .pDynamicState = &dyn, .layout = layout};
   VkPipeline pipe = VK_NULL_HANDLE;
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
   vkDestroyShaderModule(vk->dev, vs, NULL);
   vkDestroyShaderModule(vk->dev, fs, NULL);
   return pipe;
}

int
gb_run_render_gbuf_pre_gfx(const struct gb_opts *o)
{
   static const char *const known[] = {"res", "cbs", "kind", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "unknown param");
      return 0;
   }
   const char *res_s = gb_param(o, "res", "1080p");
   const char *kind = gb_param(o, "kind", "skinning");
   long cbs_l = gb_param_int(o, "cbs", 6);
   uint32_t w, h;
   if (!parse_gtav_res(res_s, &w, &h)) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "bad res (native|half|1080p|720p)");
      return 0;
   }
   bool do_skin = false;
   if (!strcmp(kind, "skinning")) do_skin = true;
   else if (!strcmp(kind, "direct")) do_skin = false;
   else {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "bad kind (direct|skinning)");
      return 0;
   }
   if (cbs_l < 2 || cbs_l > 16) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "cbs must be 2..16");
      return 0;
   }
   if (o->mode != GB_MODE_QUEUE) {
      gb_result_simple(o, o->mode, 1, NULL, "skip",
                       "render.gbuf_pre_gfx supports mode=queue only");
      return 0;
   }
   uint32_t ncbs = (uint32_t)cbs_l;
   int def_depths[2] = {1, 2};
   const int *depths = o->ndepths ? o->depths : def_depths;
   int ndepths = o->ndepths ? o->ndepths : 2;

   struct gb_vk vk;
   if (gb_vk_create(&vk, o->device, 1) != 0) {
      gb_result_simple(o, o->mode, 1, NULL, "skip", "device init failed");
      return 0;
   }

   struct gb_img rt[4], depth_img;
   const VkFormat rt_fmts[4] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT};
   for (int i = 0; i < 4; i++)
      gb_img_create(&vk, w, h, rt_fmts[i],
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &rt[i]);
   gb_img_create_ex(&vk, w, h, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &depth_img);
   VkCommandBuffer init_cb = gb_oneshot_begin(&vk);
   for (int i = 0; i < 4; i++)
      gb_img_barrier(init_cb, rt[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   VkImageMemoryBarrier2 db = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
      .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .image = depth_img.img,
      .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
   };
   VkDependencyInfo dep_init = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &db};
   vkCmdPipelineBarrier2(init_cb, &dep_init);
   gb_oneshot_end(&vk, init_cb);

   VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout gfx_layout;
   GB_VK_CHECK(vkCreatePipelineLayout(vk.dev, &plci, NULL, &gfx_layout));
   VkPipeline gfx_pipe_zw = make_mrt4_depth_pipe(&vk, gfx_layout, true);
   VkPipeline gfx_pipe_nozw = make_mrt4_depth_pipe(&vk, gfx_layout, false);

   struct gb_pipe skin_pipe = {0};
   struct gb_buf skin_buf = {0};
   if (do_skin) {
      gb_buf_create(&vk, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &skin_buf);
      gb_compute_pipe(&vk, GB_SPV(light_comp), 1, 4, &skin_pipe);
      struct gb_buf *bs[1] = {&skin_buf};
      gb_pipe_set_buffers(&vk, &skin_pipe, bs, 1);
   }

   uint64_t W = (uint64_t)o->warmup_ms * 1000000ull;
   uint64_t R = (uint64_t)o->run_ms * 1000000ull;
   for (int di = 0; di < ndepths; di++) {
      int depth = depths[di];
      if (depth < 1 || depth > 8) continue;
      VkCommandPool pools[8];
      VkCommandBuffer cbs[8 * 16];
      VkSemaphore tl;
      VkSemaphoreTypeCreateInfo stci = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
      VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &stci};
      GB_VK_CHECK(vkCreateSemaphore(vk.dev, &sci, NULL, &tl));
      for (int s = 0; s < depth; s++) {
         VkCommandPoolCreateInfo cpci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = vk.qfam};
         GB_VK_CHECK(vkCreateCommandPool(vk.dev, &cpci, NULL, &pools[s]));
         VkCommandBufferAllocateInfo cbai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pools[s], .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = ncbs};
         GB_VK_CHECK(vkAllocateCommandBuffers(vk.dev, &cbai, &cbs[(size_t)s * ncbs]));
      }

      struct gb_pass pass = {.nwin = o->runs, .win_s = (double)R / 1e9};
      uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
      uint64_t slot_val[8] = {0}, slot_sub_ns[8] = {0};
      uint64_t k = 0;
      bool bad = false;
      VkViewport vp = {0, 0, (float)w, (float)h, 0.0f, 1.0f};
      VkRect2D sc = {{0, 0}, {w, h}};

      while (!bad) {
         int s = (int)(k % (uint64_t)depth);
         if (slot_val[s] > 0) {
            VkSemaphoreWaitInfo wi = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
               .semaphoreCount = 1, .pSemaphores = &tl, .pValues = &slot_val[s]};
            if (vkWaitSemaphores(vk.dev, &wi, 10ull * 1000000000ull) != VK_SUCCESS) {
               bad = true;
               break;
            }
            uint64_t done_ns = gb_now_ns();
            if (done_ns >= tm0 && done_ns < t_end) {
               int widx = (int)((done_ns - tm0) / R);
               if (widx >= 0 && widx < pass.nwin) pass.win_count[widx]++;
               gb_samples_push(&pass.latency, (double)(done_ns - slot_sub_ns[s]) / 1e3);
               pass.submits++;
            }
            if (done_ns >= t_end) break;
         }

         uint64_t tr0 = gb_now_ns();
         GB_VK_CHECK(vkResetCommandPool(vk.dev, pools[s], 0));
         VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
         VkCommandBufferSubmitInfo cbsis[16];
         for (uint32_t c = 0; c < ncbs; c++) {
            VkCommandBuffer cb = cbs[(size_t)s * ncbs + c];
            GB_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
            if (do_skin) {
               uint32_t sv = (uint32_t)(k + c);
               vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, skin_pipe.pipe);
               vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, skin_pipe.layout,
                                       0, 1, &skin_pipe.set, 0, NULL);
               vkCmdPushConstants(cb, skin_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &sv);
               vkCmdDispatch(cb, 4, 1, 1);
               VkMemoryBarrier2 mb = {
                  .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                  .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  .dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                   VK_ACCESS_2_SHADER_READ_BIT |
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT};
               VkDependencyInfo di2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                       .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
               vkCmdPipelineBarrier2(cb, &di2);
            }
            /* Attachment dependency between consecutive CBs (c > 0) and across
             * back-to-back submissions (c == 0) sharing the G-buffer targets. */
            VkMemoryBarrier2 att_mb = {
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
               .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
               .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
            VkDependencyInfo att_di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                       .memoryBarrierCount = 1, .pMemoryBarriers = &att_mb};
            vkCmdPipelineBarrier2(cb, &att_di);
            VkRenderingAttachmentInfo catts[4];
            for (int i = 0; i < 4; i++) {
               catts[i] = (VkRenderingAttachmentInfo){
                  .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                  .imageView = rt[i].view,
                  .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  .loadOp = c == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                  .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               };
            }
            VkRenderingAttachmentInfo datt = {
               .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
               .imageView = depth_img.view,
               .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               .loadOp = c == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
               .clearValue = {.depthStencil = {1.0f, 0}},
            };
            VkRenderingInfo ri = {
               .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
               .renderArea = {{0, 0}, {w, h}},
               .layerCount = 1,
               .colorAttachmentCount = 4, .pColorAttachments = catts,
               .pDepthAttachment = &datt};
            vkCmdBeginRendering(cb, &ri);
            vkCmdSetViewport(cb, 0, 1, &vp);
            vkCmdSetScissor(cb, 0, 1, &sc);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              (c & 1u) ? gfx_pipe_nozw : gfx_pipe_zw);
            float pc[4] = {(float)((k + c) & 15u) * 0.0625f, 0.25f, 0.5f, 1.0f};
            vkCmdPushConstants(cb, gfx_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
            vkCmdDraw(cb, 3, 1, 0, 0);
            vkCmdEndRendering(cb);
            GB_VK_CHECK(vkEndCommandBuffer(cb));
            cbsis[c] = (VkCommandBufferSubmitInfo){
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
         }
         uint64_t tr1 = gb_now_ns();
         uint64_t sig_v = ++k;
         VkSemaphoreSubmitInfo sinfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = tl, .value = sig_v,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
         VkSubmitInfo2 si2 = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount = ncbs, .pCommandBufferInfos = cbsis,
            .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &sinfo};
         GB_VK_CHECK(vkQueueSubmit2(vk.queues[0], 1, &si2, VK_NULL_HANDLE));
         uint64_t ts1 = gb_now_ns();
         slot_val[s] = sig_v;
         slot_sub_ns[s] = tr1;
         if (ts1 >= tm0 && ts1 < t_end) {
            gb_samples_push(&pass.record_cpu, (double)(tr1 - tr0) / 1e3);
            gb_samples_push(&pass.submit_cpu, (double)(ts1 - tr1) / 1e3);
         }
      }
      for (int s = 0; s < depth; s++) {
         if (slot_val[s] > 0) {
            VkSemaphoreWaitInfo wi = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
               .semaphoreCount = 1, .pSemaphores = &tl, .pValues = &slot_val[s]};
            vkWaitSemaphores(vk.dev, &wi, 10ull * 1000000000ull);
         }
         vkDestroyCommandPool(vk.dev, pools[s], NULL);
      }
      vkDestroySemaphore(vk.dev, tl, NULL);

      struct gb_result res;
      gb_result_begin(&res, o, GB_MODE_QUEUE, depth, NULL);
      gb_result_run_info(&res, o->warmup_ms, o->runs, o->run_ms);
      if (bad) {
         gb_result_finish(&res, "fail", NULL, "GPU timeout");
      } else {
         gb_result_stream_metrics(&res, &pass, NULL);
         struct gb_samples us = {0}, per = {0};
         gb_pass_rate(&pass, &us);
         for (size_t i = 0; i < us.n; i++)
            gb_samples_push(&per, us.v[i] / (double)ncbs);
         gb_result_metric(&res, "us_per_cb", "us", true, &per, false);
         gb_samples_free(&us);
         gb_samples_free(&per);
         gb_result_end_metrics(&res);
         gb_result_finish(&res, "ok", "us_per_submit", NULL);
      }
      gb_pass_free(&pass);
   }

   if (do_skin) {
      gb_pipe_destroy(&vk, &skin_pipe);
      gb_buf_destroy(&vk, &skin_buf);
   }
   vkDestroyPipeline(vk.dev, gfx_pipe_zw, NULL);
   vkDestroyPipeline(vk.dev, gfx_pipe_nozw, NULL);
   vkDestroyPipelineLayout(vk.dev, gfx_layout, NULL);
   gb_img_destroy(&vk, &depth_img);
   for (int i = 0; i < 4; i++)
      gb_img_destroy(&vk, &rt[i]);
   gb_vk_destroy(&vk);
   return 0;
}
