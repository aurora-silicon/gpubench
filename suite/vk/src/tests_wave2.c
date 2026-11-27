// SPDX-License-Identifier: GPL-2.0-only
/* 90%-coverage Wave-2 heavy-game / AAA surface Vulkan benchmarks:
 *   render.shadow, compute.cull_mdi, render.bindless_ubo,
 *   compute.fp16, transfer.ops, render.postfx
 * Mirrors suite/metal/main.m and suite/gl/gl_wave2.c with identical
 * dimensions (gb_params.h) and shared CPU reference oracles (gb_wave2_ref.h). */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>

#include "gb_params.h"
#include "gb_vk.h"
#include "gb_wave2_ref.h"
#include "shaders.h"
#include "tests.h"

static bool
parse_res_w2(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static void
img_barrier_sub(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect,
                uint32_t base_mip, uint32_t mip_cnt, uint32_t base_layer, uint32_t layer_cnt,
                VkImageLayout from, VkImageLayout to,
                VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
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
      .subresourceRange = {aspect, base_mip, mip_cnt, base_layer, layer_cnt},
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
            VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static uint32_t
find_mem_type(struct gb_vk *vk, uint32_t bits, uint32_t preferred)
{
   if (bits & (1u << preferred))
      return preferred;
   for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++)
      if (bits & (1u << i))
         return i;
   gb_die("no memory type for bits 0x%x", bits);
}

static VkPipeline
make_gfx_pipe_w2(struct gb_vk *vk, const uint32_t *vs, size_t vs_b, const uint32_t *fs, size_t fs_b,
                 VkPipelineLayout layout, uint32_t color_cnt, const VkFormat *color_fmts, VkFormat depth_fmt,
                 VkSampleCountFlagBits samples, bool depth_test, bool additive_blend)
{
   VkShaderModule mv = gb_shader(vk, vs, vs_b);
   VkShaderModule mf = fs ? gb_shader(vk, fs, fs_b) : VK_NULL_HANDLE;
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
      .rasterizationSamples = samples,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth_test ? VK_TRUE : VK_FALSE,
      .depthWriteEnable = depth_test ? VK_TRUE : VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
   if (additive_blend) {
      ba.blendEnable = VK_TRUE;
      ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.colorBlendOp = VK_BLEND_OP_ADD;
      ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      ba.alphaBlendOp = VK_BLEND_OP_ADD;
   }
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = color_cnt,
      .pAttachments = color_cnt ? &ba : NULL,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = color_cnt,
      .pColorAttachmentFormats = color_fmts,
      .depthAttachmentFormat = depth_fmt,
   };
   VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
      .stageCount = fs ? 2u : 1u,
      .pStages = st,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &dss,
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = layout,
   };
   VkPipeline pipe;
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   if (mf)
      vkDestroyShaderModule(vk->dev, mf, NULL);
   return pipe;
}

static int
emit_skip_depths(const struct gb_opts *o, const int *defd, int ndef, const char *reason)
{
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : ndef;
   for (int i = 0; i < nd; i++)
      gb_result_simple(o, o->mode, depths[i], NULL, "skip", reason);
   return 0;
}

/* ======================= 1. render.shadow ======================= */

struct shadow_state {
   bool alpha_test;
   uint32_t res, inst;
   VkFormat depth_fmt;
   VkImage depth_img;
   VkDeviceMemory depth_mem;
   VkImageView layer_views[GB_SHADOW_CASCADES];
   VkImageView array_view;
   VkSampler cmp_sampler;
   struct gb_img recv;
   VkPipelineLayout depth_layout;
   VkPipeline depth_pipe;
   VkDescriptorSetLayout recv_dsl;
   VkDescriptorPool recv_pool;
   VkDescriptorSet recv_set;
   VkPipelineLayout recv_layout;
   VkPipeline recv_pipe;
};

static void *
shadow_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "depth_only");
   bool alpha_test = !strcmp(mode, "alpha_test");
   if (!alpha_test && strcmp(mode, "depth_only")) {
      snprintf(err, n, "mode must be depth_only or alpha_test");
      return NULL;
   }
   const char *fmt_s = gb_param(o, "fmt", "d32f");
   VkFormat dfmt;
   if (!strcmp(fmt_s, "d32f")) dfmt = VK_FORMAT_D32_SFLOAT;
   else if (!strcmp(fmt_s, "d16")) dfmt = VK_FORMAT_D16_UNORM;
   else {
      snprintf(err, n, "fmt must be d32f or d16");
      return NULL;
   }
   long res = gb_param_int(o, "res", 2048);
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   if (res < 256 || res > 4096 || inst < 1024 || inst > 16384) {
      snprintf(err, n, "res 256..4096, inst 1024..16384");
      return NULL;
   }

   struct shadow_state *st = calloc(1, sizeof(*st));
   st->alpha_test = alpha_test;
   st->res = (uint32_t)res;
   st->inst = (uint32_t)inst;
   st->depth_fmt = dfmt;

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = dfmt,
      .extent = {st->res, st->res, 1},
      .mipLevels = 1,
      .arrayLayers = GB_SHADOW_CASCADES,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->depth_img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(vk->dev, st->depth_img, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_mem_type(vk, req.memoryTypeBits, vk->mem_dev),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &st->depth_mem));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->depth_img, st->depth_mem, 0));

   for (uint32_t c = 0; c < GB_SHADOW_CASCADES; c++) {
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = st->depth_img,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = dfmt,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, c, 1},
      };
      GB_VK_CHECK(vkCreateImageView(vk->dev, &vi, NULL, &st->layer_views[c]));
   }
   VkImageViewCreateInfo avi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = st->depth_img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
      .format = dfmt,
      .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, GB_SHADOW_CASCADES},
   };
   GB_VK_CHECK(vkCreateImageView(vk->dev, &avi, NULL, &st->array_view));

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .compareEnable = VK_TRUE,
      .compareOp = VK_COMPARE_OP_LESS,
      .maxLod = 0.0f,
   };
   GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->cmp_sampler));

   gb_img_create(vk, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->recv);
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, st->recv.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);

   VkPushConstantRange dpr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 8};
   VkPipelineLayoutCreateInfo dlci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &dpr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &dlci, NULL, &st->depth_layout));
   st->depth_pipe = make_gfx_pipe_w2(
      vk, GB_SPV(shadow_depth_vert),
      alpha_test ? shadow_alpha_frag : NULL, alpha_test ? shadow_alpha_frag_size : 0,
      st->depth_layout, 0, NULL, dfmt, VK_SAMPLE_COUNT_1_BIT, true, false);

   VkDescriptorSetLayoutBinding bind = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &bind,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->recv_dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->recv_pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = st->recv_pool, .descriptorSetCount = 1, .pSetLayouts = &st->recv_dsl,
   };
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->recv_set));
   VkDescriptorImageInfo dii = {st->cmp_sampler, st->array_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   VkWriteDescriptorSet wr = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = st->recv_set, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii,
   };
   vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);

   VkPushConstantRange rpr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4};
   VkPipelineLayoutCreateInfo rlci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->recv_dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &rpr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &rlci, NULL, &st->recv_layout));
   VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   st->recv_pipe = make_gfx_pipe_w2(vk, GB_SPV(fs_vert), GB_SPV(shadow_pcf_frag),
                                    st->recv_layout, 1, &cfmt, VK_FORMAT_UNDEFINED,
                                    VK_SAMPLE_COUNT_1_BIT, false, false);
   return st;
}

static void
shadow_teardown(struct gb_vk *vk, void *p)
{
   struct shadow_state *st = p;
   vkDestroyPipeline(vk->dev, st->depth_pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->depth_layout, NULL);
   vkDestroyPipeline(vk->dev, st->recv_pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->recv_layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->recv_pool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->recv_dsl, NULL);
   vkDestroySampler(vk->dev, st->cmp_sampler, NULL);
   vkDestroyImageView(vk->dev, st->array_view, NULL);
   for (uint32_t c = 0; c < GB_SHADOW_CASCADES; c++)
      vkDestroyImageView(vk->dev, st->layer_views[c], NULL);
   vkDestroyImage(vk->dev, st->depth_img, NULL);
   vkFreeMemory(vk->dev, st->depth_mem, NULL);
   gb_img_destroy(vk, &st->recv);
   free(st);
}

static void
shadow_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct shadow_state *st = p;
   img_barrier_sub(cb, st->depth_img, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, GB_SHADOW_CASCADES,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

   VkViewport svp = {0, 0, (float)st->res, (float)st->res, 0.0f, 1.0f};
   VkRect2D ssc = {{0, 0}, {st->res, st->res}};
   for (uint32_t c = 0; c < GB_SHADOW_CASCADES; c++) {
      VkRenderingAttachmentInfo da = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->layer_views[c],
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil = {1.0f, 0},
      };
      VkRenderingInfo ri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = ssc,
         .layerCount = 1,
         .pDepthAttachment = &da,
      };
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &svp);
      vkCmdSetScissor(cb, 0, 1, &ssc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->depth_pipe);
      uint32_t pc[2] = {c, st->res};
      vkCmdPushConstants(cb, st->depth_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, pc);
      vkCmdDraw(cb, 6, st->inst, 0, 0);
      vkCmdEndRendering(cb);
   }

   img_barrier_sub(cb, st->depth_img, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, GB_SHADOW_CASCADES,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

   float clr[4] = {0, 0, 0, 1};
   gb_begin_render(cb, &st->recv, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->recv_pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->recv_layout, 0, 1, &st->recv_set, 0, NULL);
   float inv_res = 1.0f / (float)st->res;
   vkCmdPushConstants(cb, st->recv_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &inv_res);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

static void
shadow_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   long res = gb_param_int(o, "res", 2048);
   double tris = (double)GB_SHADOW_CASCADES * (double)inst * 2.0;
   double pix = (double)GB_SHADOW_CASCADES * (double)res * (double)res +
                (double)GB_SHADOW_RECV_DIM * (double)GB_SHADOW_RECV_DIM;
   gb_derive_rate(r, "mtri_s", "Mtri/s", us, tris);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, pix / 1e3);
}

static void
shadow_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct shadow_state *st = p;
   uint8_t *got = malloc((size_t)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM * 4u);
   gb_img_readback(vk, &st->recv, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, 4u);
   gb_shadow_check_fb(got, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, st->alpha_test, chk);
   free(got);
}

static const char *const shadow_params[] = {"mode", "fmt", "res", "inst", NULL};
static const struct gb_stream_test t_render_shadow = {
   .name = "render.shadow", .known_params = shadow_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = shadow_setup, .teardown = shadow_teardown, .record = shadow_record,
   .derive = shadow_derive, .check = shadow_check,
};

int
gb_run_render_shadow(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_shadow);
}

/* ======================= 2. compute.cull_mdi ======================= */

struct cull_state {
   uint32_t objs, visible_pct, expected_vis, expected_crc;
   struct gb_buf objs_buf, cmds_buf, cnt_buf, ib_buf;
   struct gb_img fb;
   struct gb_pipe comp_pipe;
   VkDescriptorSetLayout gfx_dsl;
   VkDescriptorPool gfx_pool;
   VkDescriptorSet gfx_set;
   VkPipelineLayout gfx_layout;
   VkPipeline gfx_pipe;
};

static void *
cull_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long objs = gb_param_int(o, "objs", GB_CULL_MAX_OBJS);
   long vis = gb_param_int(o, "visible", 25);
   if (objs < 64 || objs > (long)GB_CULL_MAX_OBJS || vis < 0 || vis > 100) {
      snprintf(err, n, "objs 64..16384, visible 0..100");
      return NULL;
   }
   struct cull_state *st = calloc(1, sizeof(*st));
   st->objs = (uint32_t)objs;
   st->visible_pct = (uint32_t)vis;

   struct gb_cull_obj *host_objs = calloc(st->objs, sizeof(*host_objs));
   st->expected_vis = gb_cull_mdi_fill_objects(st->objs, st->visible_pct, host_objs);
   uint32_t *ref_fb = malloc((size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   st->expected_crc = gb_cull_mdi_ref_fb(host_objs, st->objs, ref_fb);
   free(ref_fb);

   gb_buf_create(vk, (VkDeviceSize)st->objs * sizeof(struct gb_cull_obj),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->objs_buf);
   gb_buf_upload(vk, host_objs, &st->objs_buf, (size_t)st->objs * sizeof(struct gb_cull_obj));
   free(host_objs);

   gb_buf_create(vk, (VkDeviceSize)st->objs * sizeof(VkDrawIndexedIndirectCommand),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, false, &st->cmds_buf);
   gb_buf_create(vk, 16,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 false, &st->cnt_buf);
   static const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
   gb_buf_create(vk, sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 false, &st->ib_buf);
   gb_buf_upload(vk, idx, &st->ib_buf, sizeof(idx));

   gb_img_create(vk, GB_CULL_DIM, GB_CULL_DIM, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->fb);
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, st->fb.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);

   gb_compute_pipe(vk, GB_SPV(cull_mdi_comp), 3, 4, &st->comp_pipe);
   struct gb_buf *cbufs[3] = {&st->objs_buf, &st->cmds_buf, &st->cnt_buf};
   gb_pipe_set_buffers(vk, &st->comp_pipe, cbufs, 3);

   VkDescriptorSetLayoutBinding bind = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &bind,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->gfx_dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->gfx_pool));
   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = st->gfx_pool, .descriptorSetCount = 1, .pSetLayouts = &st->gfx_dsl,
   };
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->gfx_set));
   VkDescriptorBufferInfo bi = {st->objs_buf.buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet wr = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = st->gfx_set, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi,
   };
   vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->gfx_dsl,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &plci, NULL, &st->gfx_layout));
   VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   st->gfx_pipe = make_gfx_pipe_w2(vk, GB_SPV(cull_mdi_vert), GB_SPV(cull_mdi_frag),
                                   st->gfx_layout, 1, &cfmt, VK_FORMAT_UNDEFINED,
                                   VK_SAMPLE_COUNT_1_BIT, false, false);
   return st;
}

static void
cull_teardown(struct gb_vk *vk, void *p)
{
   struct cull_state *st = p;
   vkDestroyPipeline(vk->dev, st->gfx_pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->gfx_layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->gfx_pool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->gfx_dsl, NULL);
   gb_pipe_destroy(vk, &st->comp_pipe);
   gb_img_destroy(vk, &st->fb);
   gb_buf_destroy(vk, &st->ib_buf);
   gb_buf_destroy(vk, &st->cnt_buf);
   gb_buf_destroy(vk, &st->cmds_buf);
   gb_buf_destroy(vk, &st->objs_buf);
   free(st);
}

static void
cull_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct cull_state *st = p;
   mem_barrier(cb, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
               VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
   vkCmdFillBuffer(cb, st->cnt_buf.buf, 0, 4, 0);
   mem_barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->comp_pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->comp_pipe.layout, 0, 1, &st->comp_pipe.set, 0, NULL);
   vkCmdPushConstants(cb, st->comp_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &st->objs);
   vkCmdDispatch(cb, (st->objs + GB_CULL_LOCAL - 1u) / GB_CULL_LOCAL, 1, 1);

   mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
               VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

   float clr[4] = {0, 0, 0, 1};
   gb_begin_render(cb, &st->fb, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->gfx_pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->gfx_layout, 0, 1, &st->gfx_set, 0, NULL);
   vkCmdBindIndexBuffer(cb, st->ib_buf.buf, 0, VK_INDEX_TYPE_UINT16);
   vkCmdDrawIndexedIndirectCount(cb, st->cmds_buf.buf, 0, st->cnt_buf.buf, 0,
                                 st->objs, sizeof(VkDrawIndexedIndirectCommand));
   vkCmdEndRendering(cb);
}

static void
cull_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long objs = gb_param_int(o, "objs", GB_CULL_MAX_OBJS);
   long vis = gb_param_int(o, "visible", 25);
   gb_derive_rate(r, "mobjs_s", "Mobjs/s", us, (double)objs);
   gb_derive_rate(r, "mdraws_s", "Mdraws/s", us, (double)objs * (double)vis / 100.0);
}

static void
set_crc_check(struct gb_check *chk, uint32_t got, uint32_t expected)
{
   chk->done = true;
   chk->crc = got;
   chk->expected = expected;
   chk->valid = (got == expected);
}

static void
cull_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct cull_state *st = p;
   uint32_t got_cnt = 0;
   gb_buf_readback(vk, &st->cnt_buf, &got_cnt, sizeof(got_cnt));
   uint8_t *fb = malloc((size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   gb_img_readback(vk, &st->fb, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, fb, 4u);
   uint32_t got_crc = gb_crc32(fb, (size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   free(fb);
   set_crc_check(chk, got_crc, st->expected_crc);
   if (got_cnt != st->expected_vis) {
      chk->valid = false;
      gb_log("compute.cull_mdi visible count mismatch: got=%u expected=%u", got_cnt, st->expected_vis);
   }
}

static const char *const cull_params[] = {"objs", "visible", NULL};
static const struct gb_stream_test t_compute_cull_mdi = {
   .name = "compute.cull_mdi", .known_params = cull_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = cull_setup, .teardown = cull_teardown, .record = cull_record,
   .derive = cull_derive, .check = cull_check,
};

int
gb_run_compute_cull_mdi(const struct gb_opts *o)
{
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   bool ok = vk.has_draw_indirect_count && vk.has_subgroup_ballot;
   gb_vk_destroy(&vk);
   if (!ok)
      return emit_skip_depths(o, t_compute_cull_mdi.default_depths, t_compute_cull_mdi.ndefault,
                              "device lacks drawIndirectCount or subgroupBallot");
   return gb_run_stream_test(o, &t_compute_cull_mdi);
}

/* ======================= 3. render.bindless_ubo ======================= */

struct bubo_state {
   bool desc_buf_mode;
   uint32_t draws, expected_crc;
   struct gb_buf banks[GB_BUBO_BANKS];
   struct gb_buf desc_buf;
   VkDeviceMemory alias_mem;
   struct gb_img rt_alias, rt_final;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkPipelineLayout layout;
   VkPipeline pipe;
   struct gb_vk *vk;
};

static void
create_aliased_rts(struct gb_vk *vk, struct bubo_state *st)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {GB_BUBO_DIM, GB_BUBO_DIM, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->rt_alias.img));
   GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->rt_final.img));
   VkMemoryRequirements ra, rb;
   vkGetImageMemoryRequirements(vk->dev, st->rt_alias.img, &ra);
   vkGetImageMemoryRequirements(vk->dev, st->rt_final.img, &rb);
   VkDeviceSize sz = ra.size > rb.size ? ra.size : rb.size;
   uint32_t bits = ra.memoryTypeBits & rb.memoryTypeBits;
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = sz,
      .memoryTypeIndex = find_mem_type(vk, bits, vk->mem_dev),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &st->alias_mem));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->rt_alias.img, st->alias_mem, 0));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->rt_final.img, st->alias_mem, 0));

   struct gb_img *rts[2] = {&st->rt_alias, &st->rt_final};
   for (int i = 0; i < 2; i++) {
      rts[i]->w = GB_BUBO_DIM;
      rts[i]->h = GB_BUBO_DIM;
      rts[i]->fmt = VK_FORMAT_R8G8B8A8_UNORM;
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = rts[i]->img,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = rts[i]->fmt,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      GB_VK_CHECK(vkCreateImageView(vk->dev, &vi, NULL, &rts[i]->view));
   }
}

static void *
bubo_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "ubo_vec4");
   bool desc_buf_mode = !strcmp(mode, "desc_buf");
   if (!desc_buf_mode && strcmp(mode, "ubo_vec4")) {
      snprintf(err, n, "mode must be ubo_vec4 or desc_buf");
      return NULL;
   }
   long draws = gb_param_int(o, "draws", GB_BUBO_MAX_DRAWS);
   if (draws < 1 || draws > (long)GB_BUBO_MAX_DRAWS) {
      snprintf(err, n, "draws must be 1..2048");
      return NULL;
   }
   struct bubo_state *st = calloc(1, sizeof(*st));
   st->desc_buf_mode = desc_buf_mode;
   st->draws = (uint32_t)draws;
   st->vk = vk;

   uint32_t *ref = malloc((size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   st->expected_crc = gb_bubo_ref_fb(st->draws, ref);
   free(ref);

   float *tmp = malloc(GB_BUBO_ENTRIES * 4u * sizeof(float));
   VkBufferUsageFlags ubo_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
   if (desc_buf_mode)
      ubo_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
   for (uint32_t b = 0; b < GB_BUBO_BANKS; b++) {
      gb_bubo_fill_bank(b, tmp);
      gb_buf_create(vk, GB_BUBO_ENTRIES * 16u, ubo_usage, false, &st->banks[b]);
      gb_buf_upload(vk, tmp, &st->banks[b], GB_BUBO_ENTRIES * 16u);
   }
   free(tmp);

   create_aliased_rts(vk, st);

   VkDescriptorSetLayoutBinding bind = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = GB_BUBO_BANKS,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
#ifdef VK_EXT_descriptor_buffer
      .flags = desc_buf_mode ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0u,
#endif
      .bindingCount = 1,
      .pBindings = &bind,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->dsl));

   if (desc_buf_mode) {
#ifdef VK_EXT_descriptor_buffer
      VkDeviceSize layout_size = 0, bind_off = 0;
      vk->get_dsl_size(vk->dev, st->dsl, &layout_size);
      vk->get_dsl_binding_offset(vk->dev, st->dsl, 0, &bind_off);
      VkDeviceSize align = vk->desc_buf_offset_align ? vk->desc_buf_offset_align : 64;
      VkDeviceSize buf_sz = ((layout_size + align - 1) / align) * align;
      if (buf_sz < 256)
         buf_sz = 256;
      gb_buf_create_addr(vk, buf_sz, VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT, true, &st->desc_buf);
      memset(st->desc_buf.map, 0, (size_t)buf_sz);
      for (uint32_t b = 0; b < GB_BUBO_BANKS; b++) {
         VkDescriptorAddressInfoEXT dai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .address = st->banks[b].addr,
            .range = GB_BUBO_ENTRIES * 16u,
         };
         VkDescriptorGetInfoEXT dgi = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .data.pUniformBuffer = &dai,
         };
         uint8_t *dst = (uint8_t *)st->desc_buf.map + bind_off + (size_t)b * vk->ubo_desc_size;
         vk->get_descriptor(vk->dev, &dgi, vk->ubo_desc_size, dst);
      }
#endif
   } else {
      VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, GB_BUBO_BANKS};
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps,
      };
      GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->dpool));
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->set));
      VkDescriptorBufferInfo bi[GB_BUBO_BANKS];
      for (uint32_t b = 0; b < GB_BUBO_BANKS; b++)
         bi[b] = (VkDescriptorBufferInfo){st->banks[b].buf, 0, GB_BUBO_ENTRIES * 16u};
      VkWriteDescriptorSet wr = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = st->set, .dstBinding = 0, .descriptorCount = GB_BUBO_BANKS,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = bi,
      };
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   }

   VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &plci, NULL, &st->layout));

   /* Create graphics pipeline (with DESCRIPTOR_BUFFER_BIT_EXT when enabled). */
   VkShaderModule mv = gb_shader(vk, GB_SPV(bubo_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(bubo_frag));
   VkPipelineShaderStageCreateInfo stages[2] = {
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
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1,
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
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &ba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
#ifdef VK_EXT_descriptor_buffer
      .flags = desc_buf_mode ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0u,
#endif
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cbs,
      .pDynamicState = &ds, .layout = st->layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &st->pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
   return st;
}

static void
bubo_teardown(struct gb_vk *vk, void *p)
{
   struct bubo_state *st = p;
   vkDestroyPipeline(vk->dev, st->pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   if (st->dpool)
      vkDestroyDescriptorPool(vk->dev, st->dpool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl, NULL);
   if (st->desc_buf.buf)
      gb_buf_destroy(vk, &st->desc_buf);
   for (uint32_t b = 0; b < GB_BUBO_BANKS; b++)
      gb_buf_destroy(vk, &st->banks[b]);
   vkDestroyImageView(vk->dev, st->rt_alias.view, NULL);
   vkDestroyImageView(vk->dev, st->rt_final.view, NULL);
   vkDestroyImage(vk->dev, st->rt_alias.img, NULL);
   vkDestroyImage(vk->dev, st->rt_final.img, NULL);
   vkFreeMemory(vk->dev, st->alias_mem, NULL);
   free(st);
}

static void
bubo_bind_desc(struct bubo_state *st, VkCommandBuffer cb)
{
   if (st->desc_buf_mode) {
#ifdef VK_EXT_descriptor_buffer
      VkDescriptorBufferBindingInfoEXT bi = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
         .address = st->desc_buf.addr,
         .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
      };
      uint32_t buf_idx = 0;
      VkDeviceSize off = 0;
      st->vk->cmd_bind_desc_bufs(cb, 1, &bi);
      st->vk->cmd_set_desc_buf_offsets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &buf_idx, &off);
#endif
   } else {
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &st->set, 0, NULL);
   }
}

static void
bubo_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct bubo_state *st = p;
   float clr[4] = {0, 0, 0, 1};

   /* Pass 1: exercise aliased placement target rt_alias (offset 0 of shared memory). */
   gb_img_barrier(cb, st->rt_alias.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_begin_render(cb, &st->rt_alias, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe);
   bubo_bind_desc(st, cb);
   uint32_t warmup_draws = st->draws > 64u ? 64u : st->draws;
   for (uint32_t d = 0; d < warmup_draws; d++) {
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &d);
      vkCmdDraw(cb, 6, 1, 0, 0);
   }
   vkCmdEndRendering(cb);

   /* Aliasing barrier: transition rt_final (same VkDeviceMemory, offset 0) with LOAD_OP_CLEAR. */
   gb_img_barrier(cb, st->rt_final.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_begin_render(cb, &st->rt_final, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe);
   bubo_bind_desc(st, cb);
   for (uint32_t d = 0; d < st->draws; d++) {
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &d);
      vkCmdDraw(cb, 6, 1, 0, 0);
   }
   vkCmdEndRendering(cb);
}

static void
bubo_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long draws = gb_param_int(o, "draws", GB_BUBO_MAX_DRAWS);
   gb_derive_rate(r, "kdraws_s", "kdraws/s", us, (double)draws * 1e3);
}

static void
bubo_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct bubo_state *st = p;
   uint8_t *fb = malloc((size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   gb_img_readback(vk, &st->rt_final, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, fb, 4u);
   uint32_t got = gb_crc32(fb, (size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   free(fb);
   set_crc_check(chk, got, st->expected_crc);
}

static const char *const bubo_params[] = {"mode", "draws", NULL};
static const struct gb_stream_test t_render_bindless_ubo = {
   .name = "render.bindless_ubo", .known_params = bubo_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = bubo_setup, .teardown = bubo_teardown, .record = bubo_record,
   .derive = bubo_derive, .check = bubo_check,
};

int
gb_run_render_bindless_ubo(const struct gb_opts *o)
{
   const char *mode = gb_param(o, "mode", "ubo_vec4");
   if (!strcmp(mode, "desc_buf")) {
      struct gb_vk vk;
      gb_vk_create(&vk, o->device, 1);
      bool ok = vk.has_descriptor_buffer;
      gb_vk_destroy(&vk);
      if (!ok)
         return emit_skip_depths(o, t_render_bindless_ubo.default_depths, t_render_bindless_ubo.ndefault,
                                 "device lacks VK_EXT_descriptor_buffer");
   }
   return gb_run_stream_test(o, &t_render_bindless_ubo);
}

/* ======================= 4. compute.fp16 ======================= */

struct fp16_state {
   bool gemv;
   uint32_t expected_crc;
   struct gb_pipe pipe;
   struct gb_buf w_buf, x_buf, out_buf;
};

static void *
fp16_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "gemv");
   bool gemv = !strcmp(mode, "gemv");
   if (!gemv && strcmp(mode, "alu_mix")) {
      snprintf(err, n, "mode must be gemv or alu_mix");
      return NULL;
   }
   struct fp16_state *st = calloc(1, sizeof(*st));
   st->gemv = gemv;
   if (gemv) {
      size_t w_bytes = (size_t)GB_FP16_GEMV_M * GB_FP16_GEMV_K;
      size_t x_bytes = (size_t)GB_FP16_GEMV_K * 2u;
      size_t y_bytes = (size_t)GB_FP16_GEMV_M * sizeof(float);
      uint32_t *w_host = malloc(w_bytes);
      uint32_t *x_host = malloc(x_bytes);
      float *y_ref = malloc(y_bytes);
      gb_fp16_gemv_fill(w_host, x_host);
      st->expected_crc = gb_fp16_gemv_ref(w_host, x_host, y_ref);
      free(y_ref);

      gb_buf_create(vk, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->w_buf);
      gb_buf_create(vk, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->x_buf);
      gb_buf_create(vk, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out_buf);
      gb_buf_upload(vk, w_host, &st->w_buf, w_bytes);
      gb_buf_upload(vk, x_host, &st->x_buf, x_bytes);
      free(w_host);
      free(x_host);

      gb_compute_pipe(vk, GB_SPV(fp16_gemv_comp), 3, 0, &st->pipe);
      struct gb_buf *b[3] = {&st->w_buf, &st->x_buf, &st->out_buf};
      gb_pipe_set_buffers(vk, &st->pipe, b, 3);
   } else {
      uint32_t nthr = GB_FP16_ALUMIX_GROUPS * GB_FP16_ALUMIX_LOCAL;
      float *ref = malloc((size_t)nthr * sizeof(float));
      st->expected_crc = gb_fp16_alumix_ref(nthr, GB_FP16_ALUMIX_ITERS, -1.0f, 0.25f, ref);
      free(ref);

      gb_buf_create(vk, (size_t)nthr * sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out_buf);
      gb_compute_pipe(vk, GB_SPV(fp16_alumix_comp), 1, 12, &st->pipe);
      struct gb_buf *b[1] = {&st->out_buf};
      gb_pipe_set_buffers(vk, &st->pipe, b, 1);
   }
   return st;
}

static void
fp16_teardown(struct gb_vk *vk, void *p)
{
   struct fp16_state *st = p;
   gb_pipe_destroy(vk, &st->pipe);
   if (st->w_buf.buf) gb_buf_destroy(vk, &st->w_buf);
   if (st->x_buf.buf) gb_buf_destroy(vk, &st->x_buf);
   gb_buf_destroy(vk, &st->out_buf);
   free(st);
}

static void
fp16_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct fp16_state *st = p;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipe.layout, 0, 1, &st->pipe.set, 0, NULL);
   if (st->gemv) {
      vkCmdDispatch(cb, GB_FP16_GEMV_M / 4u, 1, 1);
   } else {
      struct { uint32_t iters; float a, b; } pc = {GB_FP16_ALUMIX_ITERS, -1.0f, 0.25f};
      vkCmdPushConstants(cb, st->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, GB_FP16_ALUMIX_GROUPS, 1, 1);
   }
}

static void
fp16_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   if (!strcmp(gb_param(o, "mode", "gemv"), "gemv")) {
      double bytes = (double)GB_FP16_GEMV_M * GB_FP16_GEMV_K + (double)GB_FP16_GEMV_K * 2.0 +
                     (double)GB_FP16_GEMV_M * 4.0;
      double flops = 2.0 * (double)GB_FP16_GEMV_M * (double)GB_FP16_GEMV_K;
      gb_derive_rate(r, "gbps", "GB/s", us, bytes / 1e3);
      gb_derive_rate(r, "gflops", "GFLOP/s", us, flops / 1e3);
   } else {
      gb_derive_rate(r, "gflops", "GFLOP/s", us, GB_FP16_ALUMIX_FLOP / 1e3);
   }
}

static void
fp16_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct fp16_state *st = p;
   size_t bytes = st->gemv ? (size_t)GB_FP16_GEMV_M * sizeof(float)
                           : (size_t)GB_FP16_ALUMIX_GROUPS * GB_FP16_ALUMIX_LOCAL * sizeof(float);
   void *got = malloc(bytes);
   gb_buf_readback(vk, &st->out_buf, got, bytes);
   set_crc_check(chk, gb_crc32(got, bytes), st->expected_crc);
   free(got);
}

static const char *const fp16_params[] = {"mode", NULL};
static const struct gb_stream_test t_compute_fp16 = {
   .name = "compute.fp16", .known_params = fp16_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = fp16_setup, .teardown = fp16_teardown, .record = fp16_record,
   .derive = fp16_derive, .check = fp16_check,
};

int
gb_run_compute_fp16(const struct gb_opts *o)
{
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   bool ok = vk.has_float16;
   gb_vk_destroy(&vk);
   if (!ok)
      return emit_skip_depths(o, t_compute_fp16.default_depths, t_compute_fp16.ndefault,
                              "device lacks shaderFloat16");
   return gb_run_stream_test(o, &t_compute_fp16);
}

/* ======================= 5. transfer.ops ======================= */

enum xfer_kind { XF_CLEAR_4M, XF_CLEAR_16M, XF_COPY_B2I, XF_MIPCHAIN };

struct xfer_state {
   enum xfer_kind kind;
   uint32_t w, h, mips, expected_crc;
   struct gb_buf staging;
   VkImage img;
   VkDeviceMemory mem;
};

static void *
xfer_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *k = gb_param(o, "kind", "clear_4m");
   struct xfer_state *st = calloc(1, sizeof(*st));
   if (!strcmp(k, "clear_4m")) {
      st->kind = XF_CLEAR_4M; st->w = GB_XFER_4M_DIM; st->h = GB_XFER_4M_DIM; st->mips = 1;
      st->expected_crc = gb_transfer_clear_ref_crc(st->w, st->h);
   } else if (!strcmp(k, "clear_16m")) {
      st->kind = XF_CLEAR_16M; st->w = GB_XFER_16M_DIM; st->h = GB_XFER_16M_DIM; st->mips = 1;
      st->expected_crc = gb_transfer_clear_ref_crc(st->w, st->h);
   } else if (!strcmp(k, "copy_b2i")) {
      st->kind = XF_COPY_B2I; st->w = GB_XFER_16M_DIM; st->h = GB_XFER_16M_DIM; st->mips = 1;
      size_t bytes = (size_t)st->w * st->h * 4u;
      gb_buf_create(vk, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, &st->staging);
      gb_transfer_fill_rgba(st->w, st->h, st->staging.map);
      st->expected_crc = gb_crc32(st->staging.map, bytes);
   } else if (!strcmp(k, "mipchain")) {
      st->kind = XF_MIPCHAIN; st->w = GB_XFER_MIP_W; st->h = GB_XFER_MIP_H; st->mips = GB_XFER_MIP_LEVELS;
   } else {
      free(st);
      snprintf(err, n, "kind must be clear_4m, clear_16m, copy_b2i or mipchain");
      return NULL;
   }

   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {st->w, st->h, 1},
      .mipLevels = st->mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   GB_VK_CHECK(vkCreateImage(vk->dev, &ici, NULL, &st->img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(vk->dev, st->img, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_mem_type(vk, req.memoryTypeBits, vk->mem_dev),
   };
   GB_VK_CHECK(vkAllocateMemory(vk->dev, &ai, NULL, &st->mem));
   GB_VK_CHECK(vkBindImageMemory(vk->dev, st->img, st->mem, 0));

   VkCommandBuffer cb = gb_oneshot_begin(vk);
   if (st->kind == XF_MIPCHAIN) {
      size_t bytes = (size_t)st->w * st->h * 4u;
      struct gb_buf up;
      gb_buf_create(vk, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, &up);
      gb_transfer_fill_mip0(st->w, st->h, up.map);
      img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, 0, st->mips, 0, 1,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkBufferImageCopy r = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {st->w, st->h, 1}};
      vkCmdCopyBufferToImage(cb, up.buf, st->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      gb_oneshot_end(vk, cb);
      gb_buf_destroy(vk, &up);
   } else {
      img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      gb_oneshot_end(vk, cb);
   }
   return st;
}

static void
xfer_teardown(struct gb_vk *vk, void *p)
{
   struct xfer_state *st = p;
   if (st->staging.buf)
      gb_buf_destroy(vk, &st->staging);
   vkDestroyImage(vk->dev, st->img, NULL);
   vkFreeMemory(vk->dev, st->mem, NULL);
   free(st);
}

static void
xfer_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct xfer_state *st = p;
   if (st->kind == XF_CLEAR_4M || st->kind == XF_CLEAR_16M) {
      VkClearColorValue cv = {.float32 = {192.0f / 255.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f}};
      VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cb, st->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &rng);
   } else if (st->kind == XF_COPY_B2I) {
      VkBufferImageCopy r = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {st->w, st->h, 1}};
      vkCmdCopyBufferToImage(cb, st->staging.buf, st->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
   } else {
      img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, 1, st->mips - 1u, 0, 1,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      int32_t mw = (int32_t)st->w, mh = (int32_t)st->h;
      for (uint32_t i = 1; i < st->mips; i++) {
         int32_t nw = mw > 1 ? mw / 2 : 1;
         int32_t nh = mh > 1 ? mh / 2 : 1;
         VkImageBlit bl = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1u, 0, 1},
            .srcOffsets = {{0, 0, 0}, {mw, mh, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1},
            .dstOffsets = {{0, 0, 0}, {nw, nh, 1}},
         };
         vkCmdBlitImage(cb, st->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        st->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
         if (i + 1u < st->mips) {
            img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
         }
         mw = nw;
         mh = nh;
      }
   }
}

static void
xfer_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   const char *k = gb_param(o, "kind", "clear_4m");
   double bytes = 0.0;
   if (!strcmp(k, "clear_4m")) bytes = (double)GB_XFER_4M_DIM * GB_XFER_4M_DIM * 4.0;
   else if (!strcmp(k, "clear_16m") || !strcmp(k, "copy_b2i"))
      bytes = (double)GB_XFER_16M_DIM * GB_XFER_16M_DIM * 4.0;
   else {
      /* Sum of src read + dst write across all 11 downsample blits. */
      uint32_t mw = GB_XFER_MIP_W, mh = GB_XFER_MIP_H;
      for (uint32_t i = 1; i < GB_XFER_MIP_LEVELS; i++) {
         uint32_t nw = mw > 1 ? mw / 2 : 1, nh = mh > 1 ? mh / 2 : 1;
         bytes += ((double)mw * mh + (double)nw * nh) * 4.0;
         mw = nw; mh = nh;
      }
   }
   gb_derive_rate(r, "gbps", "GB/s", us, bytes / 1e3);
}

static void
xfer_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct xfer_state *st = p;
   if (st->kind != XF_MIPCHAIN) {
      struct gb_img tmp = {.img = st->img, .w = st->w, .h = st->h};
      size_t bytes = (size_t)st->w * st->h * 4u;
      uint8_t *got = malloc(bytes);
      gb_img_readback(vk, &tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, got, 4u);
      set_crc_check(chk, gb_crc32(got, bytes), st->expected_crc);
      free(got);
   } else {
      uint32_t m4w = st->w >> 4, m4h = st->h >> 4;
      struct gb_buf b4, blast;
      gb_buf_create(vk, (size_t)m4w * m4h * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &b4);
      gb_buf_create(vk, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &blast);
      VkCommandBuffer cb = gb_oneshot_begin(vk);
      /* Mip 4 is already TRANSFER_SRC_OPTIMAL; mip 11 is TRANSFER_DST_OPTIMAL. */
      img_barrier_sub(cb, st->img, VK_IMAGE_ASPECT_COLOR_BIT, st->mips - 1u, 1, 0, 1,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy r4 = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 4, 0, 1},
                              .imageExtent = {m4w, m4h, 1}};
      VkBufferImageCopy rl = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, st->mips - 1u, 0, 1},
                              .imageExtent = {1, 1, 1}};
      vkCmdCopyImageToBuffer(cb, st->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b4.buf, 1, &r4);
      vkCmdCopyImageToBuffer(cb, st->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, blast.buf, 1, &rl);
      gb_oneshot_end(vk, cb);
      gb_transfer_check_mipchain(b4.map, m4w, m4h, blast.map, chk);
      gb_buf_destroy(vk, &b4);
      gb_buf_destroy(vk, &blast);
   }
}

static const char *const xfer_params[] = {"kind", NULL};
static const struct gb_stream_test t_transfer_ops = {
   .name = "transfer.ops", .known_params = xfer_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = xfer_setup, .teardown = xfer_teardown, .record = xfer_record,
   .derive = xfer_derive, .check = xfer_check,
};

int
gb_run_transfer_ops(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_transfer_ops);
}

/* ======================= 6. render.postfx ======================= */

struct postfx_state {
   uint32_t w, h, msaa;
   struct gb_img hdr_ms, hdr_res, depth_ms;
   struct gb_img bloom[GB_POSTFX_BLOOM_MIPS];
   struct gb_img final_img;
   VkSampler linear_smp;
   VkPipelineLayout scene_layout, down_layout, up_layout, tone_layout;
   VkPipeline scene_pipe, down_pipe, up_pipe, tone_pipe;
   VkDescriptorSetLayout dsl1, dsl2;
   VkDescriptorPool dpool;
   VkDescriptorSet down_sets[GB_POSTFX_BLOOM_MIPS];
   VkDescriptorSet up_sets[GB_POSTFX_BLOOM_MIPS - 1u];
   VkDescriptorSet tone_set;
};

static void *
postfx_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   uint32_t w = 0, h = 0;
   if (!parse_res_w2(gb_param(o, "res", "1440p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   long msaa = gb_param_int(o, "msaa", 4);
   if (msaa != 1 && msaa != 4) {
      snprintf(err, n, "msaa must be 1 or 4");
      return NULL;
   }
   struct postfx_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->msaa = (uint32_t)msaa;
   VkSampleCountFlagBits sc = msaa == 4 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;

   if (msaa == 4) {
      gb_img_create_ex(vk, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, sc,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT, &st->hdr_ms);
   }
   gb_img_create(vk, w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->hdr_res);
   gb_img_create_ex(vk, w, h, VK_FORMAT_D32_SFLOAT, sc,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, &st->depth_ms);
   uint32_t bw = w, bh = h;
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      bw = bw > 1 ? bw / 2 : 1;
      bh = bh > 1 ? bh / 2 : 1;
      gb_img_create(vk, bw, bh, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &st->bloom[i]);
   }
   gb_img_create(vk, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->final_img);

   VkCommandBuffer cb = gb_oneshot_begin(vk);
   if (msaa == 4) {
      gb_img_barrier(cb, st->hdr_ms.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   }
   img_barrier_sub(cb, st->depth_ms.img, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
   gb_img_barrier(cb, st->final_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->linear_smp));

   VkDescriptorSetLayoutBinding b1 = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutBinding b2[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
   };
   VkDescriptorSetLayoutCreateInfo d1 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b1,
   };
   VkDescriptorSetLayoutCreateInfo d2 = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b2,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &d1, NULL, &st->dsl1));
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &d2, NULL, &st->dsl2));

   uint32_t nsets = GB_POSTFX_BLOOM_MIPS + (GB_POSTFX_BLOOM_MIPS - 1u) + 1u;
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nsets + 2u};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = nsets, .poolSizeCount = 1, .pPoolSizes = &ps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->dpool));

   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      VkDescriptorSetAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl1,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &ai, &st->down_sets[i]));
      VkImageView srcv = (i == 0) ? st->hdr_res.view : st->bloom[i - 1u].view;
      VkDescriptorImageInfo ii = {st->linear_smp, srcv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = st->down_sets[i], .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii,
      };
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   }
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS - 1u; i++) {
      VkDescriptorSetAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl1,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &ai, &st->up_sets[i]));
      VkDescriptorImageInfo ii = {st->linear_smp, st->bloom[i + 1u].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = st->up_sets[i], .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii,
      };
      vkUpdateDescriptorSets(vk->dev, 1, &wr, 0, NULL);
   }
   {
      VkDescriptorSetAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl2,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &ai, &st->tone_set));
      VkDescriptorImageInfo ii[2] = {
         {st->linear_smp, st->hdr_res.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
         {st->linear_smp, st->bloom[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      };
      VkWriteDescriptorSet wr[2] = {
         {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->tone_set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
         {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->tone_set, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[1]},
      };
      vkUpdateDescriptorSets(vk->dev, 2, wr, 0, NULL);
   }

   VkPipelineLayoutCreateInfo slci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &slci, NULL, &st->scene_layout));
   VkPushConstantRange pr_down = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 20};
   VkPipelineLayoutCreateInfo dlci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl1,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr_down,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &dlci, NULL, &st->down_layout));
   VkPushConstantRange pr_up = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
   VkPipelineLayoutCreateInfo ulci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl1,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr_up,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &ulci, NULL, &st->up_layout));
   VkPushConstantRange pr_tone = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
   VkPipelineLayoutCreateInfo tlci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl2,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr_tone,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &tlci, NULL, &st->tone_layout));

   VkFormat hdr_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
   VkFormat ldr_fmt = VK_FORMAT_R8G8B8A8_UNORM;
   st->scene_pipe = make_gfx_pipe_w2(vk, GB_SPV(postfx_scene_vert), GB_SPV(postfx_scene_frag),
                                     st->scene_layout, 1, &hdr_fmt, VK_FORMAT_D32_SFLOAT, sc, true, false);
   st->down_pipe = make_gfx_pipe_w2(vk, GB_SPV(fs_vert), GB_SPV(postfx_down_frag),
                                    st->down_layout, 1, &hdr_fmt, VK_FORMAT_UNDEFINED,
                                    VK_SAMPLE_COUNT_1_BIT, false, false);
   st->up_pipe = make_gfx_pipe_w2(vk, GB_SPV(fs_vert), GB_SPV(postfx_up_frag),
                                  st->up_layout, 1, &hdr_fmt, VK_FORMAT_UNDEFINED,
                                  VK_SAMPLE_COUNT_1_BIT, false, true);
   st->tone_pipe = make_gfx_pipe_w2(vk, GB_SPV(fs_vert), GB_SPV(postfx_tonemap_frag),
                                    st->tone_layout, 1, &ldr_fmt, VK_FORMAT_UNDEFINED,
                                    VK_SAMPLE_COUNT_1_BIT, false, false);
   return st;
}

static void
postfx_teardown(struct gb_vk *vk, void *p)
{
   struct postfx_state *st = p;
   vkDestroyPipeline(vk->dev, st->scene_pipe, NULL);
   vkDestroyPipeline(vk->dev, st->down_pipe, NULL);
   vkDestroyPipeline(vk->dev, st->up_pipe, NULL);
   vkDestroyPipeline(vk->dev, st->tone_pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->scene_layout, NULL);
   vkDestroyPipelineLayout(vk->dev, st->down_layout, NULL);
   vkDestroyPipelineLayout(vk->dev, st->up_layout, NULL);
   vkDestroyPipelineLayout(vk->dev, st->tone_layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->dpool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl1, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl2, NULL);
   vkDestroySampler(vk->dev, st->linear_smp, NULL);
   if (st->hdr_ms.img) gb_img_destroy(vk, &st->hdr_ms);
   gb_img_destroy(vk, &st->hdr_res);
   gb_img_destroy(vk, &st->depth_ms);
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++)
      gb_img_destroy(vk, &st->bloom[i]);
   gb_img_destroy(vk, &st->final_img);
   free(st);
}

static void
postfx_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct postfx_state *st = p;

   /* 1. HDR scene pass (+ HW MSAA resolve when msaa == 4). */
   gb_img_barrier(cb, st->hdr_res.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   VkRenderingAttachmentInfo ca = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->msaa == 4u ? st->hdr_ms.view : st->hdr_res.view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolveMode = st->msaa == 4u ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
      .resolveImageView = st->msaa == 4u ? st->hdr_res.view : VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = st->msaa == 4u ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = {0.04f, 0.05f, 0.08f, 1.0f},
   };
   VkRenderingAttachmentInfo da = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->depth_ms.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {st->w, st->h}},
      .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &ca,
      .pDepthAttachment = &da,
   };
   vkCmdBeginRendering(cb, &ri);
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};
   vkCmdSetViewport(cb, 0, 1, &vp);
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->scene_pipe);
   vkCmdDraw(cb, GB_POSTFX_QUADS * 6u, 1, 0, 0);
   vkCmdEndRendering(cb);

   gb_img_barrier(cb, st->hdr_res.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

   /* 2. 5-level bloom downsample chain. */
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      struct gb_img *dst = &st->bloom[i];
      uint32_t sw = (i == 0) ? st->w : st->bloom[i - 1u].w;
      uint32_t sh = (i == 0) ? st->h : st->bloom[i - 1u].h;
      gb_img_barrier(cb, dst->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      /* The full-screen triangle writes every texel, so the previous contents are not needed: DONT_CARE, as
       * the Metal suite's downsample passes (MTLLoadActionDontCare). */
      float z[4] = {0, 0, 0, 1};
      gb_begin_render_op(cb, dst, z, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->down_pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->down_layout, 0, 1, &st->down_sets[i], 0, NULL);
      struct { float iw, ih, siw, sih; uint32_t bp; } pc = {
         1.0f / (float)dst->w, 1.0f / (float)dst->h, 1.0f / (float)sw, 1.0f / (float)sh, i == 0 ? 1u : 0u
      };
      vkCmdPushConstants(cb, st->down_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
      gb_img_barrier(cb, dst->img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }

   /* 3. 4-level additive tent upsample chain (bloom[4] -> bloom[0]). */
   for (int i = (int)GB_POSTFX_BLOOM_MIPS - 2; i >= 0; i--) {
      struct gb_img *dst = &st->bloom[i];
      struct gb_img *src = &st->bloom[i + 1];
      gb_img_barrier(cb, dst->img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      VkRenderingAttachmentInfo uca = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = dst->view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo uri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {dst->w, dst->h}},
         .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &uca,
      };
      vkCmdBeginRendering(cb, &uri);
      VkViewport uvp = {0, 0, (float)dst->w, (float)dst->h, 0.0f, 1.0f};
      VkRect2D usc = {{0, 0}, {dst->w, dst->h}};
      vkCmdSetViewport(cb, 0, 1, &uvp);
      vkCmdSetScissor(cb, 0, 1, &usc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->up_pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->up_layout, 0, 1, &st->up_sets[i], 0, NULL);
      float pc[4] = {1.0f / (float)dst->w, 1.0f / (float)dst->h, 1.0f / (float)src->w, 1.0f / (float)src->h};
      vkCmdPushConstants(cb, st->up_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
      gb_img_barrier(cb, dst->img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }

   /* 4. Final ACES tonemap + vignette into RGBA8 (full-screen, DONT_CARE as in the Metal suite). */
   float z[4] = {0, 0, 0, 1};
   gb_begin_render_op(cb, &st->final_img, z, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->tone_pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->tone_layout, 0, 1, &st->tone_set, 0, NULL);
   float tpc[2] = {1.0f / (float)st->w, 1.0f / (float)st->h};
   vkCmdPushConstants(cb, st->tone_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(tpc), tpc);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

static void
postfx_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 2560, h = 1440;
   parse_res_w2(gb_param(o, "res", "1440p"), &w, &h);
   gb_derive_rate(r, "fps", "fps", us, 1e6);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h / 1e3);
}

static void
postfx_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct postfx_state *st = p;
   uint8_t *got = malloc((size_t)st->w * st->h * 4u);
   gb_img_readback(vk, &st->final_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, 4u);
   gb_postfx_check_fb(got, st->w, st->h, chk);
   free(got);
}

static const char *const postfx_params[] = {"res", "msaa", NULL};
static const struct gb_stream_test t_render_postfx = {
   .name = "render.postfx", .known_params = postfx_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = postfx_setup, .teardown = postfx_teardown, .record = postfx_record,
   .derive = postfx_derive, .check = postfx_check,
};

int
gb_run_render_postfx(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_postfx);
}
