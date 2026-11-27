// SPDX-License-Identifier: GPL-2.0-only
/*
 * AAA native-resolution Vulkan benchmarks (chunk aaa-native in
 * suite/plan-90pct-w2.txt). Each models a cost measured in a real title at the
 * panel's native resolution, where the gap shows.
 *
 *   render.pass_split (G1)   one G-buffer pass recorded as `splits` render
 *                            pass instances that each load and store every
 *                            attachment, in one command buffer or one per
 *                            instance in a single vkQueueSubmit2 (GTA V
 *                            Enhanced records its G-buffer this way, ~21
 *                            command buffers per submission)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gb_vk.h"
#include "shaders.h"
#include "tests.h"

/* ======================= render.pass_split (G1) ======================= */

#define PS_MAX_TARGETS 5u
#define PS_MAX_SPLITS 64u
#define PS_MAX_DRAWS 4096u
#define PS_HASH_ROUNDS 32u /* pass_split.frag */

enum ps_mode { PS_ONE_CB, PS_MULTI_CB };
/* What is recorded between two instances: nothing; a framebuffer-local
 * attachment barrier; a vkd3d-style UAV barrier (shader writes to shader
 * reads of the pre-rasterization, fragment and compute stages, the shape
 * GTA V records between its splits); vkd3d-proton's render-target layout
 * decay (every attachment to GENERAL after an instance, back before the
 * next); a 64-thread compute dispatch. */
enum ps_gap { PS_GAP_NONE, PS_GAP_BARRIER, PS_GAP_UAV, PS_GAP_DECAY, PS_GAP_COMPUTE };

struct ps_push {
   int32_t rect[4];
   float scale[2];
   float z;
   uint32_t draw;
};

struct ps_state {
   uint32_t w, h, targets, splits, draws, bpp;
   VkFormat fmt;
   enum ps_mode mode;
   enum ps_gap gap;
   struct gb_img color[PS_MAX_TARGETS];
   struct gb_img depth;
   VkPipelineLayout layout;
   VkPipeline pipe;
   struct gb_pipe gap_pipe;
   struct gb_buf scratch;
   struct ps_push *list;
   double pixels; /* rectangle pixels per submission */
   /* The expected contents of every target, built on first check. */
   uint32_t *want;
   uint32_t want_crc;
   size_t words;
};

static bool
ps_parse_res(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "native")) /* the 14" MacBook Pro panel */
      *w = 3024, *h = 1964;
   else if (!strcmp(s, "1080p"))
      *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p"))
      *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p"))
      *w = 3840, *h = 2160;
   else if (sscanf(s, "%ux%u", w, h) != 2 || !*w || !*h || *w > 16384 || *h > 16384)
      return false;
   return true;
}

/* The draws of one submission: rectangles covering most of the target, at
 * pixel bounds, so coverage is exact. Even draws come nearer each time and
 * win where they cover; odd draws lie behind and win only where no earlier
 * draw covered, so the image depends on depth surviving every split. */
static void
ps_build_draws(struct ps_state *st)
{
   const uint32_t mx = st->w / 8, my = st->h / 8;
   st->pixels = 0;
   for (uint32_t i = 0; i < st->draws; i++) {
      struct ps_push *d = &st->list[i];
      d->rect[0] = (int32_t)((i * 97u) % mx);
      d->rect[1] = (int32_t)((i * 61u) % my);
      d->rect[2] = (int32_t)(st->w - (i * 37u) % mx);
      d->rect[3] = (int32_t)(st->h - (i * 53u) % my);
      d->scale[0] = 2.0f / (float)st->w;
      d->scale[1] = 2.0f / (float)st->h;
      d->z = (i & 1) ? 0.75f + 0.2f * (float)i / (float)st->draws
                     : 0.5f - 0.4f * (float)i / (float)st->draws;
      d->draw = i;
      st->pixels += (double)(d->rect[2] - d->rect[0]) * (double)(d->rect[3] - d->rect[1]);
   }
}

static uint32_t
ps_mix(uint32_t h)
{
   h ^= h >> 16;
   h *= 0x7feb352du;
   h ^= h >> 15;
   h *= 0x846ca68bu;
   h ^= h >> 16;
   return h;
}

/* pass_split.frag's pattern for target t. */
static uint32_t
ps_pattern(uint32_t h, uint32_t t)
{
   switch (t) {
   case 0: return h;
   case 1: return h ^ 0x5a5a5a5au;
   case 2: return h * 0x01000193u;
   case 3: return h + 0x3c6ef372u;
   default: return ~h;
   }
}

static void
ps_layout(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout to,
          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
   VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
ps_begin(VkCommandBuffer cb, const struct ps_state *st, VkAttachmentLoadOp load)
{
   VkRenderingAttachmentInfo ca[PS_MAX_TARGETS];
   for (uint32_t t = 0; t < st->targets; t++)
      ca[t] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->color[t].view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = load,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
   VkRenderingAttachmentInfo da = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->depth.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = load,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = {1.0f, 0},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {st->w, st->h}},
      .layerCount = 1,
      .colorAttachmentCount = st->targets,
      .pColorAttachments = ca,
      .pDepthAttachment = &da,
   };
   vkCmdBeginRendering(cb, &ri);
}

static VkPipeline
ps_pipeline(struct gb_vk *vk, const struct ps_state *st)
{
   VkShaderModule mv = gb_shader(vk, GB_SPV(pass_split_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(pass_split_frag));
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
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
   };
   VkPipelineColorBlendAttachmentState ba[PS_MAX_TARGETS];
   for (uint32_t t = 0; t < st->targets; t++)
      ba[t] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = st->targets, .pAttachments = ba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkFormat fmts[PS_MAX_TARGETS];
   for (uint32_t t = 0; t < st->targets; t++)
      fmts[t] = st->fmt;
   VkPipelineRenderingCreateInfo rci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = st->targets, .pColorAttachmentFormats = fmts,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rci,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &dss,
      .pColorBlendState = &cb, .pDynamicState = &ds,
      .layout = st->layout,
   };
   VkPipeline pipe;
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
   return pipe;
}

static void *
ps_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t errlen)
{
   struct ps_state *st = calloc(1, sizeof(*st));
   const char *res = gb_param(o, "res", "native");
   const char *fmt = gb_param(o, "fmt", "rgba8");
   const char *mode = gb_param(o, "mode", "multi_cb");
   const char *gap = gb_param(o, "gap", "none");
   long targets = gb_param_int(o, "targets", 5);
   long splits = gb_param_int(o, "splits", 8);
   long draws = gb_param_int(o, "draws", 88);
   if (!ps_parse_res(res, &st->w, &st->h)) {
      snprintf(err, errlen, "res=%s: native, 1080p, 1440p, 2160p or WxH", res);
      goto fail;
   }
   if (!strcmp(fmt, "rgba8"))
      st->fmt = VK_FORMAT_R8G8B8A8_UNORM, st->bpp = 4;
   else if (!strcmp(fmt, "rgba16"))
      st->fmt = VK_FORMAT_R16G16B16A16_UNORM, st->bpp = 8;
   else {
      snprintf(err, errlen, "fmt=%s: rgba8 or rgba16", fmt);
      goto fail;
   }
   if (!strcmp(mode, "one_cb"))
      st->mode = PS_ONE_CB;
   else if (!strcmp(mode, "multi_cb"))
      st->mode = PS_MULTI_CB;
   else {
      snprintf(err, errlen, "mode=%s: one_cb or multi_cb", mode);
      goto fail;
   }
   if (!strcmp(gap, "none"))
      st->gap = PS_GAP_NONE;
   else if (!strcmp(gap, "barrier"))
      st->gap = PS_GAP_BARRIER;
   else if (!strcmp(gap, "uav"))
      st->gap = PS_GAP_UAV;
   else if (!strcmp(gap, "decay"))
      st->gap = PS_GAP_DECAY;
   else if (!strcmp(gap, "compute"))
      st->gap = PS_GAP_COMPUTE;
   else {
      snprintf(err, errlen, "gap=%s: none, barrier, uav, decay or compute", gap);
      goto fail;
   }
   if (targets < 1 || targets > (long)PS_MAX_TARGETS || splits < 1 ||
       splits > (long)PS_MAX_SPLITS || draws < splits || draws > (long)PS_MAX_DRAWS) {
      snprintf(err, errlen, "targets 1..%u, splits 1..%u, draws splits..%u", PS_MAX_TARGETS,
               PS_MAX_SPLITS, PS_MAX_DRAWS);
      goto fail;
   }
   st->targets = (uint32_t)targets;
   st->splits = (uint32_t)splits;
   st->draws = (uint32_t)draws;

   for (uint32_t t = 0; t < st->targets; t++)
      gb_img_create(vk, st->w, st->h, st->fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    &st->color[t]);
   gb_img_create_ex(vk, st->w, st->h, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
                    &st->depth);

   VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                             sizeof(struct ps_push)};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &st->layout));
   st->pipe = ps_pipeline(vk, st);
   if (st->gap == PS_GAP_COMPUTE) {
      gb_buf_create(vk, 64ull * 4 * st->splits, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->scratch);
      gb_compute_pipe(vk, GB_SPV(pass_split_gap_comp), 1, 4, &st->gap_pipe);
      struct gb_buf *bufs[1] = {&st->scratch};
      gb_pipe_set_buffers(vk, &st->gap_pipe, bufs, 1);
   }

   st->list = calloc(st->draws, sizeof(*st->list));
   ps_build_draws(st);

   /* Every submission loads what the one before it stored and repaints it
    * the same (LESS_OR_EQUAL depth), so the contents after any number of
    * submissions are those of one, from this clear. */
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   for (uint32_t t = 0; t < st->targets; t++)
      ps_layout(cb, st->color[t].img, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   ps_layout(cb, st->depth.img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
   ps_begin(cb, st, VK_ATTACHMENT_LOAD_OP_CLEAR);
   vkCmdEndRendering(cb);
   gb_oneshot_end(vk, cb);
   return st;

fail:
   free(st);
   return NULL;
}

static void
ps_teardown(struct gb_vk *vk, void *p)
{
   struct ps_state *st = p;
   vkDestroyPipeline(vk->dev, st->pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   if (st->gap == PS_GAP_COMPUTE) {
      gb_pipe_destroy(vk, &st->gap_pipe);
      gb_buf_destroy(vk, &st->scratch);
   }
   for (uint32_t t = 0; t < st->targets; t++)
      gb_img_destroy(vk, &st->color[t]);
   gb_img_destroy(vk, &st->depth);
   free(st->list);
   free(st->want);
   free(st);
}

/* gap=decay: every attachment between its attachment layout and GENERAL,
 * as vkd3d-proton's render pass transitions record it (attachment writes to
 * shader reads after an instance, and back before the next). */
static void
ps_decay(const struct ps_state *st, VkCommandBuffer cb, bool to_general)
{
   VkImageMemoryBarrier2 b[PS_MAX_TARGETS + 1];
   const VkPipelineStageFlags2 shader = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
   for (uint32_t t = 0; t <= st->targets; t++) {
      const bool depth = t == st->targets;
      const VkImageLayout att = depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                      : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      const VkPipelineStageFlags2 att_stages =
         depth ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
               : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      const VkAccessFlags2 att_access =
         depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
               : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      b[t] = (VkImageMemoryBarrier2){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = to_general ? att_stages : shader,
         .srcAccessMask = to_general ? att_access : 0,
         .dstStageMask = to_general ? shader : att_stages,
         .dstAccessMask = to_general ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : att_access,
         .oldLayout = to_general ? att : VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = to_general ? VK_IMAGE_LAYOUT_GENERAL : att,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = depth ? st->depth.img : st->color[t].img,
         .subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
   }
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = st->targets + 1,
      .pImageMemoryBarriers = b,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
ps_pass(const struct ps_state *st, VkCommandBuffer cb, uint32_t pass)
{
   if (st->gap == PS_GAP_DECAY && pass > 0)
      ps_decay(st, cb, false);
   ps_begin(cb, st, VK_ATTACHMENT_LOAD_OP_LOAD);
   VkViewport v = {0, 0, (float)st->w, (float)st->h, 0, 1};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};
   vkCmdSetViewport(cb, 0, 1, &v);
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe);
   const uint32_t first = pass * st->draws / st->splits;
   const uint32_t end = (pass + 1) * st->draws / st->splits;
   for (uint32_t d = first; d < end; d++) {
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(struct ps_push), &st->list[d]);
      vkCmdDraw(cb, 6, 1, 0, 0);
   }
   vkCmdEndRendering(cb);
}

static void
ps_gap(const struct ps_state *st, VkCommandBuffer cb, uint32_t pass)
{
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
   switch (st->gap) {
   case PS_GAP_NONE:
      return;
   case PS_GAP_BARRIER:
      mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
      mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      mb.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      mb.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      break;
   case PS_GAP_UAV:
      mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      break;
   case PS_GAP_DECAY:
      ps_decay(st, cb, true);
      return;
   case PS_GAP_COMPUTE:
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->gap_pipe.pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->gap_pipe.layout, 0, 1,
                              &st->gap_pipe.set, 0, NULL);
      vkCmdPushConstants(cb, st->gap_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &pass);
      vkCmdDispatch(cb, 1, 1, 1);
      return;
   }
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

/* mode=one_cb: every instance in the one command buffer. */
static void
ps_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   const struct ps_state *st = p;
   for (uint32_t pass = 0; pass < st->splits; pass++) {
      ps_pass(st, cb, pass);
      if (pass + 1 < st->splits)
         ps_gap(st, cb, pass);
   }
}

/* mode=multi_cb: instance `part` and the gap after it in a command buffer of
 * their own. */
static void
ps_record_part(void *p, VkCommandBuffer cb, uint64_t k, int part)
{
   (void)k;
   const struct ps_state *st = p;
   ps_pass(st, cb, (uint32_t)part);
   if ((uint32_t)part + 1 < st->splits)
      ps_gap(st, cb, (uint32_t)part);
}

static int
ps_parts(void *p)
{
   const struct ps_state *st = p;
   return st->mode == PS_MULTI_CB ? (int)st->splits : 0;
}

static void
ps_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long splits = gb_param_int(o, "splits", 8);
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, us->v[i] / (double)(splits > 0 ? splits : 1));
   gb_result_metric(r, "us_per_pass", "us", true, &per, true);
   gb_samples_free(&per);
}

/* The CPU reference: per pixel, the last draw to pass the depth test of
 * those covering it, as in-order rasterization with LESS_OR_EQUAL gives;
 * its hash per target; clear (zero) where no draw covers. */
static void
ps_reference(struct ps_state *st)
{
   const size_t px = (size_t)st->w * st->h;
   uint16_t *winner = calloc(px, sizeof(*winner));
   float *zb = malloc(px * sizeof(*zb));
   for (size_t i = 0; i < px; i++)
      zb[i] = 1.0f;
   for (uint32_t d = 0; d < st->draws; d++) {
      const struct ps_push *dr = &st->list[d];
      for (int32_t y = dr->rect[1]; y < dr->rect[3]; y++) {
         float *zr = &zb[(size_t)y * st->w];
         uint16_t *wr = &winner[(size_t)y * st->w];
         for (int32_t x = dr->rect[0]; x < dr->rect[2]; x++) {
            if (dr->z <= zr[x]) {
               zr[x] = dr->z;
               wr[x] = (uint16_t)(d + 1);
            }
         }
      }
   }
   free(zb);

   const size_t per_px = st->bpp / 4;
   st->words = px * per_px * st->targets;
   st->want = calloc(st->words, sizeof(uint32_t));
   for (uint32_t y = 0; y < st->h; y++) {
      for (uint32_t x = 0; x < st->w; x++) {
         const size_t i = (size_t)y * st->w + x;
         if (!winner[i])
            continue;
         uint32_t h = (x * 0x9e3779b1u) ^ (y * 0x85ebca77u) ^ ((uint32_t)(winner[i] - 1) * 0xc2b2ae3du);
         for (uint32_t r = 0; r < PS_HASH_ROUNDS; r++)
            h = ps_mix(h + r);
         for (uint32_t t = 0; t < st->targets; t++) {
            const uint32_t v = ps_pattern(h, t);
            uint32_t *dst = &st->want[(t * px + i) * per_px];
            if (per_px == 1) {
               dst[0] = v;
            } else {
               /* UNORM16: k / 255 stores as 257 k. */
               dst[0] = ((v & 0xffu) * 257u) | (((v >> 8) & 0xffu) * 257u) << 16;
               dst[1] = (((v >> 16) & 0xffu) * 257u) | ((v >> 24) * 257u) << 16;
            }
         }
      }
   }
   free(winner);
   st->want_crc = gb_crc32(st->want, st->words * sizeof(uint32_t));
}

static void
ps_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct ps_state *st = p;
   if (!st->want)
      ps_reference(st);
   const size_t px = (size_t)st->w * st->h;
   uint32_t *got = malloc(st->words * sizeof(uint32_t));
   for (uint32_t t = 0; t < st->targets; t++)
      gb_img_readback(vk, &st->color[t], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      &got[t * px * (st->bpp / 4)], st->bpp);
   gb_check_words(chk, got, st->want, st->want_crc, st->words);
   free(got);
}

static const char *const ps_params[] = {"targets", "fmt", "splits", "draws", "mode", "gap", "res", NULL};
static const struct gb_stream_test t_render_pass_split = {
   .name = "render.pass_split", .known_params = ps_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = ps_setup, .teardown = ps_teardown, .record = ps_record,
   .parts = ps_parts, .record_part = ps_record_part,
   .derive = ps_derive, .check = ps_check,
};

int
gb_run_render_pass_split(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_pass_split);
}
