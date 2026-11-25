// SPDX-License-Identifier: GPL-2.0-only
/* AAA native-resolution Vulkan microbenchmarks:
 *   render.ts_query_passes  (G4: timestamp-query pass-merge & compute-stamp coalescing)
 *   render.dx11_draws       (G2/G6: DX11/DXVK per-draw dyn-UBO, VB rebind, push & shadow throughput)
 *   render.tiny_passes      (G5: bloom / post-processing tiny single-draw render-pass chain) */
#define _GNU_SOURCE
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gb_vk.h"
#include "shaders.h"
#include "tests.h"

static bool
parse_res_aaa(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "native")) *w = 3024, *h = 1964;
   else if (!strcmp(s, "256x256")) *w = 256, *h = 256;
   else if (!strcmp(s, "1024x1024")) *w = 1024, *h = 1024;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1134x736")) *w = 1134, *h = 736;
   else if (!strcmp(s, "2268x1473")) *w = 2268, *h = 1473;
   else return false;
   return true;
}

static void
aaa_mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src_s, VkAccessFlags2 src_a,
                VkPipelineStageFlags2 dst_s, VkAccessFlags2 dst_a)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src_s, .srcAccessMask = src_a,
      .dstStageMask = dst_s, .dstAccessMask = dst_a,
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
aaa_begin_color_pass(VkCommandBuffer cb, const struct gb_img *im, VkAttachmentLoadOp load_op,
                     const float clear[4])
{
   VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = im->view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = load_op,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   if (clear)
      memcpy(att.clearValue.color.float32, clear, 16);
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {im->w, im->h}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &att,
   };
   vkCmdBeginRendering(cb, &ri);
   VkViewport vp = {0, 0, (float)im->w, (float)im->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {im->w, im->h}};
   vkCmdSetViewport(cb, 0, 1, &vp);
   vkCmdSetScissor(cb, 0, 1, &sc);
}

/* ==================== G4: render.ts_query_passes ==================== */

enum ts_mode { TS_MODE_NONE = 0, TS_MODE_TS, TS_MODE_TS_COPY };
enum ts_shape { TS_SHAPE_PASSES = 0, TS_SHAPE_DXVK };

struct ts_passes_state {
   uint32_t w, h, passes;
   enum ts_mode mode;
   enum ts_shape shape;
   uint32_t max_queries;
   struct gb_img rt;
   struct gb_pipe gfx;
   struct gb_pipe comp;
   struct gb_buf comp_buf;
   VkQueryPool qp;
   struct gb_buf ts_buf;
   uint32_t queries_written;
};

static void *
ts_passes_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *res_s = gb_param(o, "res", "256x256");
   const char *mode_s = gb_param(o, "mode", "ts_copy");
   const char *shape_s = gb_param(o, "shape", "passes");
   long passes = gb_param_int(o, "passes", 16);
   uint32_t w, h;
   if (!parse_res_aaa(res_s, &w, &h)) {
      snprintf(err, n, "bad res %s", res_s);
      return NULL;
   }
   if (passes < 1 || passes > 128) {
      snprintf(err, n, "passes must be 1..128");
      return NULL;
   }
   enum ts_mode mode;
   if (!strcmp(mode_s, "none")) mode = TS_MODE_NONE;
   else if (!strcmp(mode_s, "ts")) mode = TS_MODE_TS;
   else if (!strcmp(mode_s, "ts_copy")) mode = TS_MODE_TS_COPY;
   else {
      snprintf(err, n, "mode must be none, ts or ts_copy");
      return NULL;
   }
   enum ts_shape shape;
   if (!strcmp(shape_s, "passes")) shape = TS_SHAPE_PASSES;
   else if (!strcmp(shape_s, "dxvk")) shape = TS_SHAPE_DXVK;
   else {
      snprintf(err, n, "shape must be passes or dxvk");
      return NULL;
   }

   struct ts_passes_state *st = calloc(1, sizeof(*st));
   st->w = w;
   st->h = h;
   st->passes = (uint32_t)passes;
   st->mode = mode;
   st->shape = shape;
   st->max_queries = st->passes * 4u + 8u;

   gb_img_create(vk, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->rt);
   gb_graphics_pipe(vk, GB_SPV(fs_vert), GB_SPV(color_frag),
                    VK_FORMAT_R8G8B8A8_UNORM, GB_BLEND_ALPHA, 16, &st->gfx);

   if (shape == TS_SHAPE_DXVK) {
      gb_buf_create(vk, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->comp_buf);
      gb_compute_pipe(vk, GB_SPV(light_comp), 1, 4, &st->comp);
      struct gb_buf *b[1] = {&st->comp_buf};
      gb_pipe_set_buffers(vk, &st->comp, b, 1);
   }

   if (mode != TS_MODE_NONE) {
      VkQueryPoolCreateInfo qci = {
         .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
         .queryType = VK_QUERY_TYPE_TIMESTAMP,
         .queryCount = st->max_queries,
      };
      GB_VK_CHECK(vkCreateQueryPool(vk->dev, &qci, NULL, &st->qp));
      gb_buf_create(vk, (VkDeviceSize)st->max_queries * sizeof(uint64_t),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    false, &st->ts_buf);
   }
   return st;
}

static void
ts_passes_teardown(struct gb_vk *vk, void *p)
{
   struct ts_passes_state *st = p;
   if (st->qp) vkDestroyQueryPool(vk->dev, st->qp, NULL);
   if (st->ts_buf.buf) gb_buf_destroy(vk, &st->ts_buf);
   if (st->comp.pipe) gb_pipe_destroy(vk, &st->comp);
   if (st->comp_buf.buf) gb_buf_destroy(vk, &st->comp_buf);
   gb_pipe_destroy(vk, &st->gfx);
   gb_img_destroy(vk, &st->rt);
   free(st);
}

static void
ts_passes_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct ts_passes_state *st = p;
   uint32_t q = 0;

   if (st->mode != TS_MODE_NONE)
      vkCmdResetQueryPool(cb, st->qp, 0, st->max_queries);

   gb_img_barrier(cb, st->rt.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

   const float clear[4] = {0.05f, 0.1f, 0.15f, 1.0f};
   for (uint32_t i = 0; i < st->passes; i++) {
      if (st->shape == TS_SHAPE_DXVK && i > 0 && (i % 4u) == 0) {
         aaa_mem_barrier(cb, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->comp.pipe);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->comp.layout, 0, 1, &st->comp.set, 0, NULL);
         uint32_t iters = 1u;
         vkCmdPushConstants(cb, st->comp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
         vkCmdDispatch(cb, 1, 1, 1);
         if (st->mode != TS_MODE_NONE) {
            /* Back-to-back timestamps after a compute dispatch: exact DXVK pattern. */
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, st->qp, q++);
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, st->qp, q++);
         }
         aaa_mem_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      }

      if (st->mode != TS_MODE_NONE && i == 0)
         vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, st->qp, q++);

      aaa_begin_color_pass(cb, &st->rt, i == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                           i == 0 ? clear : NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->gfx.pipe);
      for (uint32_t d = 0; d < 4u; d++) {
         float c[4] = {0.1f + 0.02f * (float)(i & 7u), 0.2f + 0.05f * (float)d, 0.3f, 0.25f};
         vkCmdPushConstants(cb, st->gfx.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, 16, c);
         vkCmdDraw(cb, 3, 1, 0, 0);
      }
      vkCmdEndRendering(cb);

      if (st->mode != TS_MODE_NONE)
         vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, st->qp, q++);
   }

   st->queries_written = q;
   if (st->mode == TS_MODE_TS_COPY && q > 0) {
      vkCmdCopyQueryPoolResults(cb, st->qp, 0, q, st->ts_buf.buf, 0, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   }
}

static void
ts_passes_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long passes = gb_param_int(o, "passes", 16);
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, us->v[i] / (double)passes);
   gb_result_metric(r, "us_per_pass", "us", true, &per, false);
   gb_samples_free(&per);
}

static void
ts_passes_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct ts_passes_state *st = p;
   size_t bytes = (size_t)st->w * st->h * 4u;
   uint8_t *fb = malloc(bytes);
   gb_img_readback(vk, &st->rt, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, fb, 4u);
   uint32_t crc = gb_crc32(fb, bytes);
   bool ok = (fb[0] != 0 || fb[1] != 0 || fb[2] != 0);
   free(fb);

   if (ok && st->mode == TS_MODE_TS_COPY && st->queries_written > 0) {
      uint64_t *ts = calloc(st->queries_written, sizeof(uint64_t));
      gb_buf_readback(vk, &st->ts_buf, ts, (size_t)st->queries_written * sizeof(uint64_t));
      for (uint32_t i = 0; i < st->queries_written; i++) {
         if (ts[i] == 0) {
            fprintf(stderr, "render.ts_query_passes: zero timestamp at index %u\n", i);
            ok = false;
            break;
         }
         if (i > 0 && ts[i] < ts[i - 1]) {
            fprintf(stderr, "render.ts_query_passes: non-monotonic ts[%u]=%llu < ts[%u]=%llu\n",
                    i, (unsigned long long)ts[i], i - 1, (unsigned long long)ts[i - 1]);
            ok = false;
            break;
         }
      }
      free(ts);
   }

   chk->done = true;
   chk->valid = ok;
   chk->crc = crc;
   chk->expected = crc;
}

static const char *const ts_passes_params[] = {"passes", "mode", "shape", "res", NULL};
static const struct gb_stream_test t_render_ts_query_passes = {
   .name = "render.ts_query_passes", .known_params = ts_passes_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = ts_passes_setup, .teardown = ts_passes_teardown, .record = ts_passes_record,
   .derive = ts_passes_derive, .check = ts_passes_check,
};

int
gb_run_render_ts_query_passes(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_ts_query_passes);
}

/* ==================== G2/G6: render.dx11_draws ==================== */

enum dx11_bind { DX11_BIND_DYNOFF = 0, DX11_BIND_DYNOFF_VB, DX11_BIND_PUSH };
enum dx11_target { DX11_TGT_COLOR = 0, DX11_TGT_SHADOW, DX11_TGT_SHADOW_ATLAS };

#define DX11_MAX_PROGS 4u
#define DX11_MAT_GROUP 64u
#define DX11_VERTS_PER_DRAW 12u

struct dx11_draws_state {
   uint32_t w, h, draws, progs;
   enum dx11_bind bind;
   enum dx11_target target;
   uint32_t ubo_stride;
   struct gb_img color;
   struct gb_img depth;
   struct gb_buf vb[2];
   struct gb_buf vs_ubo;
   struct gb_buf fs_ubo;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkPipelineLayout layout;
   VkPipeline pipes[DX11_MAX_PROGS];
};

static VkPipeline
dx11_make_pipe(struct gb_vk *vk, struct dx11_draws_state *st, bool depth_only, uint32_t variant)
{
   VkShaderModule mv = gb_shader(vk, GB_SPV(dx11_draw_vert));
   VkShaderModule mf = depth_only ? VK_NULL_HANDLE : gb_shader(vk, GB_SPV(dx11_draw_frag));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkVertexInputBindingDescription vbd = {
      .binding = 0, .stride = 16, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   VkVertexInputAttributeDescription vad = {
      .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0,
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vbd,
      .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &vad,
   };
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
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = (variant & 1u) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE,
      .frontFace = (variant & 2u) ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
   };
   VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = depth_only ? 0u : 1u,
      .pAttachments = depth_only ? NULL : &cba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = depth_only ? 0u : 1u,
      .pColorAttachmentFormats = depth_only ? NULL : &cfmt,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
      .stageCount = depth_only ? 1u : 2u,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &dss,
      .pColorBlendState = &cbs,
      .pDynamicState = &ds,
      .layout = st->layout,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   if (mf) vkDestroyShaderModule(vk->dev, mf, NULL);
   return pipe;
}

static void *
dx11_draws_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long draws = gb_param_int(o, "draws", 1200);
   long progs = gb_param_int(o, "progs", 1);
   const char *bind_s = gb_param(o, "bind", "dynoff_vb");
   const char *tgt_s = gb_param(o, "target", "color");
   const char *res_s = gb_param(o, "res", "1024x1024");

   if (draws < 1 || draws > 16384) {
      snprintf(err, n, "draws must be 1..16384");
      return NULL;
   }
   if (progs < 1 || progs > (long)DX11_MAX_PROGS) {
      snprintf(err, n, "progs must be 1..%u", DX11_MAX_PROGS);
      return NULL;
   }

   enum dx11_bind bind;
   if (!strcmp(bind_s, "dynoff")) bind = DX11_BIND_DYNOFF;
   else if (!strcmp(bind_s, "dynoff_vb")) bind = DX11_BIND_DYNOFF_VB;
   else if (!strcmp(bind_s, "push")) bind = DX11_BIND_PUSH;
   else {
      snprintf(err, n, "bind must be dynoff, dynoff_vb or push");
      return NULL;
   }

   enum dx11_target tgt;
   uint32_t w = 1024, h = 1024;
   if (!strcmp(tgt_s, "color")) {
      tgt = DX11_TGT_COLOR;
      if (!parse_res_aaa(res_s, &w, &h)) {
         snprintf(err, n, "bad res %s", res_s);
         return NULL;
      }
   } else if (!strcmp(tgt_s, "shadow")) {
      tgt = DX11_TGT_SHADOW;
      w = 2048; h = 2048;
   } else if (!strcmp(tgt_s, "shadow_atlas")) {
      tgt = DX11_TGT_SHADOW_ATLAS;
      w = 1024; h = 4096;
   } else {
      snprintf(err, n, "target must be color, shadow or shadow_atlas");
      return NULL;
   }

   struct dx11_draws_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h;
   st->draws = (uint32_t)draws;
   st->progs = (uint32_t)progs;
   st->bind = bind;
   st->target = tgt;

   uint32_t align = (uint32_t)vk->props.limits.minUniformBufferOffsetAlignment;
   if (align < 64u) align = 64u;
   st->ubo_stride = (32u + align - 1u) & ~(align - 1u);

   bool depth_only = (tgt != DX11_TGT_COLOR);
   if (!depth_only) {
      gb_img_create(vk, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->color);
   }
   gb_img_create_ex(vk, w, h, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, &st->depth);

   /* Two small vertex buffers (4 triangles = 12 verts each) to exercise VB rebinds. */
   float verts[DX11_VERTS_PER_DRAW * 4u];
   for (uint32_t t = 0; t < 4u; t++) {
      float dx = (float)(t & 1u) * 0.015f;
      float dy = (float)(t >> 1u) * 0.015f;
      float z = 0.25f + 0.05f * (float)t;
      float *v = &verts[t * 12u];
      v[0] = dx;          v[1] = dy;          v[2] = z; v[3] = 1.0f;
      v[4] = dx + 0.025f; v[5] = dy;          v[6] = z; v[7] = 1.0f;
      v[8] = dx;          v[9] = dy + 0.025f; v[10] = z; v[11] = 1.0f;
   }
   for (int i = 0; i < 2; i++) {
      gb_buf_create(vk, sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    false, &st->vb[i]);
      gb_buf_upload(vk, verts, &st->vb[i], sizeof(verts));
   }

   /* Per-draw VS dynamic UBO table & per-material FS dynamic UBO table. */
   size_t vs_ubo_bytes = (size_t)st->draws * st->ubo_stride;
   uint8_t *vs_data = calloc(1, vs_ubo_bytes);
   uint32_t grid = (uint32_t)ceil(sqrt((double)st->draws));
   if (grid < 1u) grid = 1u;
   for (uint32_t d = 0; d < st->draws; d++) {
      float *u = (float *)(vs_data + (size_t)d * st->ubo_stride);
      float gx = -0.95f + 1.9f * ((float)(d % grid) / (float)grid);
      float gy = -0.95f + 1.9f * ((float)(d / grid) / (float)grid);
      u[0] = gx; u[1] = gy; u[2] = 1.0f; u[3] = 1.0f;
      u[4] = 0.2f + 0.6f * ((float)(d & 31u) / 31.0f);
      u[5] = 0.3f + 0.5f * ((float)((d >> 5) & 31u) / 31.0f);
      u[6] = 0.4f; u[7] = 1.0f;
   }
   gb_buf_create(vk, vs_ubo_bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 false, &st->vs_ubo);
   gb_buf_upload(vk, vs_data, &st->vs_ubo, vs_ubo_bytes);
   free(vs_data);

   uint32_t n_mats = (st->draws + DX11_MAT_GROUP - 1u) / DX11_MAT_GROUP;
   if (n_mats < 1u) n_mats = 1u;
   size_t fs_ubo_bytes = (size_t)n_mats * st->ubo_stride;
   uint8_t *fs_data = calloc(1, fs_ubo_bytes);
   for (uint32_t m = 0; m < n_mats; m++) {
      float *u = (float *)(fs_data + (size_t)m * st->ubo_stride);
      u[0] = 0.75f; u[1] = 0.85f; u[2] = 0.95f; u[3] = 1.0f;
      u[4] = 0.05f * (float)(m & 3u); u[5] = 0.02f; u[6] = 0.01f; u[7] = 0.0f;
   }
   gb_buf_create(vk, fs_ubo_bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 false, &st->fs_ubo);
   gb_buf_upload(vk, fs_data, &st->fs_ubo, fs_ubo_bytes);
   free(fs_data);

   VkDescriptorSetLayoutBinding bindings[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = bindings,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->dsl));

   VkDescriptorPoolSize dps = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 2};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->dpool));

   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl,
   };
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->set));

   VkDescriptorBufferInfo dbis[2] = {
      {.buffer = st->vs_ubo.buf, .offset = 0, .range = 32},
      {.buffer = st->fs_ubo.buf, .offset = 0, .range = 32},
   };
   VkWriteDescriptorSet writes[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->set, .dstBinding = 0,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &dbis[0]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st->set, .dstBinding = 1,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &dbis[1]},
   };
   vkUpdateDescriptorSets(vk->dev, 2, writes, 0, NULL);

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 16,
   };
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &plci, NULL, &st->layout));

   for (uint32_t i = 0; i < st->progs; i++)
      st->pipes[i] = dx11_make_pipe(vk, st, depth_only, i);

   return st;
}

static void
dx11_draws_teardown(struct gb_vk *vk, void *p)
{
   struct dx11_draws_state *st = p;
   for (uint32_t i = 0; i < st->progs; i++)
      if (st->pipes[i]) vkDestroyPipeline(vk->dev, st->pipes[i], NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->dpool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl, NULL);
   gb_buf_destroy(vk, &st->vs_ubo);
   gb_buf_destroy(vk, &st->fs_ubo);
   gb_buf_destroy(vk, &st->vb[0]);
   gb_buf_destroy(vk, &st->vb[1]);
   gb_img_destroy(vk, &st->depth);
   if (st->color.img) gb_img_destroy(vk, &st->color);
   free(st);
}

static void
dx11_draws_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct dx11_draws_state *st = p;
   bool depth_only = (st->target != DX11_TGT_COLOR);

   if (!depth_only) {
      gb_img_barrier(cb, st->color.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   }
   VkImageMemoryBarrier2 db = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .image = st->depth.img,
      .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
   };
   VkDependencyInfo ddi = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &db,
   };
   vkCmdPipelineBarrier2(cb, &ddi);

   VkRenderingAttachmentInfo catt = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->color.view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
   };
   VkRenderingAttachmentInfo datt = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = st->depth.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = {.depthStencil = {.depth = 1.0f, .stencil = 0}},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {st->w, st->h}},
      .layerCount = 1,
      .colorAttachmentCount = depth_only ? 0u : 1u,
      .pColorAttachments = depth_only ? NULL : &catt,
      .pDepthAttachment = &datt,
   };
   vkCmdBeginRendering(cb, &ri);
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};
   vkCmdSetViewport(cb, 0, 1, &vp);
   vkCmdSetScissor(cb, 0, 1, &sc);

   VkDeviceSize zero = 0;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipes[0]);
   vkCmdBindVertexBuffers(cb, 0, 1, &st->vb[0].buf, &zero);
   uint32_t dyn0[2] = {0, 0};
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &st->set, 2, dyn0);
   float zero_pc[4] = {0, 0, 0, 0};
   vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, zero_pc);

   uint32_t cur_pipe = 0;
   for (uint32_t d = 0; d < st->draws; d++) {
      if (st->progs > 1u && (d % 32u) == 0) {
         uint32_t p_idx = (d / 32u) % st->progs;
         if (p_idx != cur_pipe) {
            cur_pipe = p_idx;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipes[cur_pipe]);
         }
      }
      if (st->bind == DX11_BIND_PUSH) {
         float pc[4] = {0.001f * (float)(d & 15u), 0.001f * (float)((d >> 4) & 15u), 0.01f, 0.0f};
         vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, pc);
      } else {
         uint32_t offs[2] = {d * st->ubo_stride, (d / DX11_MAT_GROUP) * st->ubo_stride};
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &st->set, 2, offs);
         if (st->bind == DX11_BIND_DYNOFF_VB) {
            VkBuffer vbuf = st->vb[d & 1u].buf;
            vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &zero);
         }
      }
      vkCmdDraw(cb, DX11_VERTS_PER_DRAW, 1, 0, 0);
   }
   vkCmdEndRendering(cb);
}

static void
dx11_draws_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long draws = gb_param_int(o, "draws", 1200);
   gb_derive_rate(r, "kdraws_s", "kdraws/s", us, (double)draws * 1e3);
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, (us->v[i] * 1000.0) / (double)draws);
   gb_result_metric(r, "ns_per_draw", "ns", true, &per, false);
   gb_samples_free(&per);
}

static void
dx11_draws_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct dx11_draws_state *st = p;
   bool depth_only = (st->target != DX11_TGT_COLOR);
   size_t bytes = (size_t)st->w * st->h * 4u;
   uint8_t *buf = malloc(bytes);
   bool ok = false;
   if (!depth_only) {
      gb_img_readback(vk, &st->color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, buf, 4u);
      for (size_t i = 0; i < bytes; i += 4u)
         if (buf[i] || buf[i + 1] || buf[i + 2]) { ok = true; break; }
   } else {
      /* Readback depth image via staging buffer. */
      struct gb_buf dst;
      gb_buf_create(vk, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &dst);
      VkCommandBuffer cb = gb_oneshot_begin(vk);
      VkImageMemoryBarrier2 b = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
         .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .image = st->depth.img,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
      };
      VkDependencyInfo di = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b,
      };
      vkCmdPipelineBarrier2(cb, &di);
      VkBufferImageCopy reg = {
         .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
         .imageExtent = {st->w, st->h, 1},
      };
      vkCmdCopyImageToBuffer(cb, st->depth.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buf, 1, &reg);
      gb_oneshot_end(vk, cb);
      memcpy(buf, dst.map, bytes);
      gb_buf_destroy(vk, &dst);
      const float *df = (const float *)buf;
      for (size_t i = 0; i < (size_t)st->w * st->h; i++)
         if (df[i] < 0.99f) { ok = true; break; }
   }
   uint32_t crc = gb_crc32(buf, bytes);
   free(buf);
   chk->done = true;
   chk->valid = ok;
   chk->crc = crc;
   chk->expected = crc;
}

static const char *const dx11_draws_params[] = {"draws", "progs", "bind", "target", "res", NULL};
static const struct gb_stream_test t_render_dx11_draws = {
   .name = "render.dx11_draws", .known_params = dx11_draws_params,
   .default_depths = {1, 2}, .ndefault = 2,
   .setup = dx11_draws_setup, .teardown = dx11_draws_teardown, .record = dx11_draws_record,
   .derive = dx11_draws_derive, .check = dx11_draws_check,
};

int
gb_run_render_dx11_draws(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_dx11_draws);
}

/* ==================== G5: render.tiny_passes ==================== */

#define TINY_LEVELS 5u

struct tiny_passes_state {
   uint32_t base_w, base_h, passes;
   VkFormat fmt;
   struct gb_img lvls[TINY_LEVELS];
   struct gb_img final_rt;
   VkSampler samp;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet sets[TINY_LEVELS];
   VkPipelineLayout layout;
   VkPipeline pipe_lvl;
   VkPipeline pipe_final;
};

static VkPipeline
tiny_make_pipe(struct gb_vk *vk, VkPipelineLayout layout, VkFormat fmt)
{
   VkShaderModule mv = gb_shader(vk, GB_SPV(fs_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(tiny_pass_frag));
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
   VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
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
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cbs,
      .pDynamicState = &ds, .layout = layout,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
   return pipe;
}

static void *
tiny_passes_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *base_s = gb_param(o, "base", "1134x736");
   const char *fmt_s = gb_param(o, "fmt", "b10g11r11");
   long passes = gb_param_int(o, "passes", 20);
   uint32_t bw, bh;
   if (!parse_res_aaa(base_s, &bw, &bh)) {
      snprintf(err, n, "bad base %s", base_s);
      return NULL;
   }
   if (passes < 4 || passes > 256) {
      snprintf(err, n, "passes must be 4..256");
      return NULL;
   }
   VkFormat fmt;
   if (!strcmp(fmt_s, "b10g11r11")) fmt = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   else if (!strcmp(fmt_s, "rgba16f")) fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
   else {
      snprintf(err, n, "fmt must be b10g11r11 or rgba16f");
      return NULL;
   }

   struct tiny_passes_state *st = calloc(1, sizeof(*st));
   st->base_w = bw; st->base_h = bh;
   st->passes = (uint32_t)passes;
   st->fmt = fmt;

   for (uint32_t i = 0; i < TINY_LEVELS; i++) {
      uint32_t lw = bw >> i; if (lw < 16u) lw = 16u;
      uint32_t lh = bh >> i; if (lh < 16u) lh = 16u;
      gb_img_create(vk, lw, lh, fmt,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT, &st->lvls[i]);
   }
   gb_img_create(vk, 3024, 1964, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->final_rt);

   /* Seed level 0 so the pyramid has non-zero content on LOAD_OP_LOAD. */
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   for (uint32_t i = 0; i < TINY_LEVELS; i++) {
      gb_img_barrier(cb, st->lvls[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkClearColorValue cv = {.float32 = {0.25f, 0.5f, 0.75f, 1.0f}};
      VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cb, st->lvls[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &rng);
      gb_img_barrier(cb, st->lvls[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }
   gb_oneshot_end(vk, cb);

   VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   GB_VK_CHECK(vkCreateSampler(vk->dev, &sci, NULL, &st->samp));

   VkDescriptorSetLayoutBinding b = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dslci, NULL, &st->dsl));

   VkDescriptorPoolSize dps = {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = TINY_LEVELS};
   VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = TINY_LEVELS, .poolSizeCount = 1, .pPoolSizes = &dps,
   };
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &dpci, NULL, &st->dpool));

   for (uint32_t i = 0; i < TINY_LEVELS; i++) {
      VkDescriptorSetAllocateInfo dsai = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = st->dpool, .descriptorSetCount = 1, .pSetLayouts = &st->dsl,
      };
      GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &dsai, &st->sets[i]));
      VkDescriptorImageInfo dii = {
         .sampler = st->samp, .imageView = st->lvls[i].view,
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = st->sets[i], .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii,
      };
      vkUpdateDescriptorSets(vk->dev, 1, &w, 0, NULL);
   }

   VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 16};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &plci, NULL, &st->layout));
   st->pipe_lvl = tiny_make_pipe(vk, st->layout, fmt);
   st->pipe_final = tiny_make_pipe(vk, st->layout, VK_FORMAT_R8G8B8A8_UNORM);
   return st;
}

static void
tiny_passes_teardown(struct gb_vk *vk, void *p)
{
   struct tiny_passes_state *st = p;
   vkDestroyPipeline(vk->dev, st->pipe_final, NULL);
   vkDestroyPipeline(vk->dev, st->pipe_lvl, NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->dpool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl, NULL);
   vkDestroySampler(vk->dev, st->samp, NULL);
   gb_img_destroy(vk, &st->final_rt);
   for (uint32_t i = 0; i < TINY_LEVELS; i++)
      gb_img_destroy(vk, &st->lvls[i]);
   free(st);
}

static void
tiny_passes_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct tiny_passes_state *st = p;
   const float clr[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   uint32_t cur = 0;
   int dir = 1;

   for (uint32_t step = 0; step < st->passes; step++) {
      uint32_t next = (uint32_t)((int)cur + dir);
      if (next >= TINY_LEVELS) {
         dir = -1;
         next = TINY_LEVELS - 2u;
      } else if ((int)cur + dir < 0) {
         dir = 1;
         next = 1u;
      }
      gb_img_barrier(cb, st->lvls[next].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

      VkAttachmentLoadOp lop = (step & 1u) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      aaa_begin_color_pass(cb, &st->lvls[next], lop, lop == VK_ATTACHMENT_LOAD_OP_CLEAR ? clr : NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe_lvl);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &st->sets[cur], 0, NULL);
      float pc[4] = {1.0f / (float)st->lvls[next].w, 1.0f / (float)st->lvls[next].h, 0.9f, 0.05f};
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);

      gb_img_barrier(cb, st->lvls[next].img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      cur = next;
   }

   /* Final native-resolution 3024x1964 composite pass. */
   gb_img_barrier(cb, st->final_rt.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   aaa_begin_color_pass(cb, &st->final_rt, VK_ATTACHMENT_LOAD_OP_CLEAR, clr);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe_final);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->layout, 0, 1, &st->sets[cur], 0, NULL);
   float fpc[4] = {1.0f / (float)st->final_rt.w, 1.0f / (float)st->final_rt.h, 0.95f, 0.02f};
   vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, fpc);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
}

static void
tiny_passes_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long passes = gb_param_int(o, "passes", 20) + 1;
   struct gb_samples per = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&per, us->v[i] / (double)passes);
   gb_result_metric(r, "us_per_pass", "us", true, &per, false);
   gb_samples_free(&per);
}

static void
tiny_passes_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct tiny_passes_state *st = p;
   size_t bytes = (size_t)st->final_rt.w * st->final_rt.h * 4u;
   uint8_t *fb = malloc(bytes);
   gb_img_readback(vk, &st->final_rt, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, fb, 4u);
   uint32_t crc = gb_crc32(fb, bytes);
   bool ok = (fb[0] != 0 || fb[1] != 0 || fb[2] != 0);
   free(fb);
   chk->done = true;
   chk->valid = ok;
   chk->crc = crc;
   chk->expected = crc;
}

static const char *const tiny_passes_params[] = {"base", "fmt", "passes", NULL};
static const struct gb_stream_test t_render_tiny_passes = {
   .name = "render.tiny_passes", .known_params = tiny_passes_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = tiny_passes_setup, .teardown = tiny_passes_teardown, .record = tiny_passes_record,
   .derive = tiny_passes_derive, .check = tiny_passes_check,
};

int
gb_run_render_tiny_passes(const struct gb_opts *o)
{
   return gb_run_stream_test(o, &t_render_tiny_passes);
}
