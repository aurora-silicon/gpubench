// SPDX-License-Identifier: GPL-2.0-only
/* Wave-3 AGX3/M4-M5 compute & MSAA Vulkan benchmarks:
 *   1. compute.blur_fp32   — Direct-global uint8_t RGBA8 -> FP32 vec4 separable blur
 *   2. compute.corner_ring — 16-tap radius-3 circular stencil corner detector (FAST-16)
 *   3. compute.desc_search — Nested-loop vec4 descriptor search (LICM + idiv/imod)
 *   4. compute.chain       — Back-to-back dependent compute dispatch chain
 *   5. render.msaa_merge   — Multi-pass 2x/4x MSAA color-compressed render target merge
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>

#include "gb_params.h"
#include "gb_vk.h"
#include "gb_wave3_ref.h"
#include "shaders.h"
#include "tests.h"

#define W3_GPU_TIMEOUT_NS (10ull * 1000000000ull)

static bool
parse_res_w3(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static void
compute_barrier_w3(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
compute_pipe_spec_w3(struct gb_vk *vk, const uint32_t *code, size_t bytes, int nbuf,
                     uint32_t push_size, const VkSpecializationInfo *si, struct gb_pipe *p)
{
   memset(p, 0, sizeof(*p));
   p->push_stages = VK_SHADER_STAGE_COMPUTE_BIT;
   p->push_size = push_size;
   VkDescriptorSetLayoutBinding bs[8];
   for (int i = 0; i < nbuf; i++)
      bs[i] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   VkDescriptorSetLayoutCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = (uint32_t)nbuf,
      .pBindings = bs,
   };
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &p->dsl));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbuf};
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
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &p->dsl,
      .pushConstantRangeCount = push_size ? 1u : 0u,
      .pPushConstantRanges = push_size ? &pr : NULL,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &p->layout));
   VkShaderModule m = gb_shader(vk, code, bytes);
   VkComputePipelineCreateInfo cci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = m,
         .pName = "main",
         .pSpecializationInfo = si,
      },
      .layout = p->layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cci, NULL, &p->pipe));
   vkDestroyShaderModule(vk->dev, m, NULL);
}

/* =========================================================================
 * 1. compute.blur_fp32 (direct-global uint8_t RGBA8 -> FP32 separable blur)
 * ========================================================================= */

struct blur_fp32_state {
   uint32_t w, h, taps;
   struct gb_pipe ph, pv;
   struct gb_buf in, mid, out;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
blur_fp32_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   if (!vk->has_8bit_storage) {
      snprintf(err, n, "device lacks storageBuffer8BitAccess/shaderInt8");
      return NULL;
   }
   long taps = gb_param_int(o, "taps", 17);
   if (taps != 9 && taps != 17) {
      snprintf(err, n, "taps must be 9 or 17");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w3(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   struct blur_fp32_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->taps = (uint32_t)taps;
   size_t bytes = (size_t)w * h * 4u;
   uint8_t *in_cpu = malloc(bytes);
   gb_blur_fp32_fill_input(in_cpu, w, h);
   st->ref = malloc(bytes);
   st->ref_crc = gb_blur_fp32_ref(st->taps, in_cpu, (uint8_t *)st->ref, w, h);

   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->in);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->mid);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out);
   gb_buf_upload(vk, in_cpu, &st->in, bytes);
   free(in_cpu);

   uint32_t spec_val = st->taps;
   VkSpecializationMapEntry me = {.constantID = 0, .offset = 0, .size = sizeof(uint32_t)};
   VkSpecializationInfo si = {.mapEntryCount = 1, .pMapEntries = &me, .dataSize = sizeof(spec_val), .pData = &spec_val};
   compute_pipe_spec_w3(vk, GB_SPV(blur_fp32_h_comp), 2, 8, &si, &st->ph);
   compute_pipe_spec_w3(vk, GB_SPV(blur_fp32_v_comp), 2, 8, &si, &st->pv);
   struct gb_buf *bh[2] = {&st->in, &st->mid};
   struct gb_buf *bv[2] = {&st->mid, &st->out};
   gb_pipe_set_buffers(vk, &st->ph, bh, 2);
   gb_pipe_set_buffers(vk, &st->pv, bv, 2);
   return st;
}

static void
blur_fp32_teardown(struct gb_vk *vk, void *p)
{
   struct blur_fp32_state *st = p;
   gb_pipe_destroy(vk, &st->ph);
   gb_pipe_destroy(vk, &st->pv);
   gb_buf_destroy(vk, &st->in);
   gb_buf_destroy(vk, &st->mid);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
blur_fp32_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct blur_fp32_state *st = p;
   struct { int32_t w, h; } pc = {(int32_t)st->w, (int32_t)st->h};
   uint32_t h_gx = (((st->w + 1u) / 2u) + GB_BLUR_LOCAL - 1u) / GB_BLUR_LOCAL;
   uint32_t h_gy = (st->h + GB_BLUR_LOCAL - 1u) / GB_BLUR_LOCAL;
   uint32_t v_gx = (st->w + GB_BLUR_LOCAL - 1u) / GB_BLUR_LOCAL;
   uint32_t v_gy = (((st->h + 1u) / 2u) + GB_BLUR_LOCAL - 1u) / GB_BLUR_LOCAL;

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->ph.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->ph.layout, 0, 1, &st->ph.set, 0, NULL);
   vkCmdPushConstants(cb, st->ph.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, h_gx, h_gy, 1);

   compute_barrier_w3(cb);

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pv.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pv.layout, 0, 1, &st->pv.set, 0, NULL);
   vkCmdPushConstants(cb, st->pv.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, v_gx, v_gy, 1);
}

static void
blur_fp32_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w3(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
   /* Logical image bytes read + written across H and V passes (4 * w * h * 4 B) */
   gb_derive_rate(r, "gb_s", "GB/s", us, (double)w * h * 16.0 / 1e3);
}

static void
blur_fp32_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct blur_fp32_state *st = p;
   size_t words = (size_t)st->w * st->h;
   uint32_t *got = malloc(words * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, words * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, words);
   free(got);
}

static const char *const blur_fp32_params[] = {"taps", "res", NULL};
static const struct gb_stream_test t_compute_blur_fp32 = {
   .name = "compute.blur_fp32",
   .known_params = blur_fp32_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = blur_fp32_setup,
   .teardown = blur_fp32_teardown,
   .record = blur_fp32_record,
   .derive = blur_fp32_derive,
   .check = blur_fp32_check,
};

int gb_run_compute_blur_fp32(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_blur_fp32); }

/* =========================================================================
 * 2. compute.corner_ring (16-tap radius-3 circular stencil corner detector)
 * ========================================================================= */

struct corner_ring_state {
   uint32_t w, h;
   int32_t thresh;
   bool dense_ring;
   struct gb_pipe p;
   struct gb_buf in, out;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
corner_ring_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   if (!vk->has_8bit_storage) {
      snprintf(err, n, "device lacks storageBuffer8BitAccess/shaderInt8");
      return NULL;
   }
   const char *mode = gb_param(o, "mode", "fast16");
   bool dense_ring = false;
   if (!strcmp(mode, "dense_ring")) dense_ring = true;
   else if (strcmp(mode, "fast16")) {
      snprintf(err, n, "mode must be fast16 or dense_ring");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w3(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   long thresh = gb_param_int(o, "thresh", 24);
   if (thresh < 1 || thresh > 128) {
      snprintf(err, n, "thresh must be 1..128");
      return NULL;
   }
   struct corner_ring_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h; st->thresh = (int32_t)thresh; st->dense_ring = dense_ring;
   size_t in_bytes = (size_t)w * h;
   size_t out_bytes = (size_t)w * h * sizeof(uint32_t);
   uint8_t *in_cpu = malloc(in_bytes);
   gb_corner_ring_fill_input(in_cpu, w, h);
   st->ref = malloc(out_bytes);
   st->ref_crc = gb_corner_ring_ref(dense_ring, st->thresh, in_cpu, st->ref, w, h);

   gb_buf_create(vk, in_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->in);
   gb_buf_create(vk, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out);
   gb_buf_upload(vk, in_cpu, &st->in, in_bytes);
   free(in_cpu);

   uint32_t spec_val = dense_ring ? 1u : 0u;
   VkSpecializationMapEntry me = {.constantID = 0, .offset = 0, .size = sizeof(uint32_t)};
   VkSpecializationInfo si = {.mapEntryCount = 1, .pMapEntries = &me, .dataSize = sizeof(spec_val), .pData = &spec_val};
   compute_pipe_spec_w3(vk, GB_SPV(corner_ring_comp), 2, 12, &si, &st->p);
   struct gb_buf *b[2] = {&st->in, &st->out};
   gb_pipe_set_buffers(vk, &st->p, b, 2);
   return st;
}

static void
corner_ring_teardown(struct gb_vk *vk, void *p)
{
   struct corner_ring_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->in);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
corner_ring_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct corner_ring_state *st = p;
   struct { int32_t w, h, thresh; } pc = {(int32_t)st->w, (int32_t)st->h, st->thresh};
   uint32_t gx = (st->w + GB_CORNER_LOCAL - 1u) / GB_CORNER_LOCAL;
   uint32_t gy = (st->h + GB_CORNER_LOCAL - 1u) / GB_CORNER_LOCAL;

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, gy, 1);
}

static void
corner_ring_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w3(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
}

static void
corner_ring_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct corner_ring_state *st = p;
   size_t count = (size_t)st->w * st->h;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, count);
   free(got);
}

static const char *const corner_ring_params[] = {"mode", "res", "thresh", NULL};
static const struct gb_stream_test t_compute_corner_ring = {
   .name = "compute.corner_ring",
   .known_params = corner_ring_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = corner_ring_setup,
   .teardown = corner_ring_teardown,
   .record = corner_ring_record,
   .derive = corner_ring_derive,
   .check = corner_ring_check,
};

int gb_run_compute_corner_ring(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_corner_ring); }

/* =========================================================================
 * 3. compute.desc_search (nested read-only LICM + signed idiv/imod search)
 * ========================================================================= */

struct desc_search_state {
   uint32_t num_queries, num_cands, stride, flags;
   struct gb_pipe p;
   struct gb_buf queries, cands, weights, out;
   uint32_t *ref;
   uint32_t ref_crc;
};

static void *
desc_search_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long queries = gb_param_int(o, "queries", 8192);
   long cands = gb_param_int(o, "cands", 256);
   long stride = gb_param_int(o, "stride", 24);
   if (queries < 64 || queries > 65536) {
      snprintf(err, n, "queries must be 64..65536");
      return NULL;
   }
   if (cands < 16 || cands > 4096 || (cands & 15) != 0) {
      snprintf(err, n, "cands must be 16..4096 and a multiple of 16");
      return NULL;
   }
   if (stride < 4 || stride > 256) {
      snprintf(err, n, "stride must be 4..256");
      return NULL;
   }
   struct desc_search_state *st = calloc(1, sizeof(*st));
   st->num_queries = (uint32_t)queries;
   st->num_cands = (uint32_t)cands;
   st->stride = (uint32_t)stride;
   st->flags = 1u;

   uint32_t atlas_rows = (st->num_cands + st->stride - 1u) / st->stride;
   uint32_t atlas_pitch = st->stride + 4u;
   size_t q_bytes = (size_t)st->num_queries * GB_DESC_DIMS * sizeof(float);
   size_t c_bytes = (size_t)atlas_rows * atlas_pitch * GB_DESC_DIMS * sizeof(float);
   size_t w_bytes = 16u * GB_DESC_DIMS * sizeof(float);
   size_t o_bytes = (size_t)st->num_queries * sizeof(uint32_t);

   float *q_cpu = malloc(q_bytes);
   float *c_cpu = malloc(c_bytes);
   float *w_cpu = malloc(w_bytes);
   gb_desc_search_fill(st->num_queries, st->num_cands, st->stride, q_cpu, c_cpu, w_cpu);
   st->ref = malloc(o_bytes);
   st->ref_crc = gb_desc_search_ref(st->num_queries, st->num_cands, st->stride, st->flags,
                                    q_cpu, c_cpu, w_cpu, st->ref);

   gb_buf_create(vk, q_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->queries);
   gb_buf_create(vk, c_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->cands);
   gb_buf_create(vk, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->weights);
   gb_buf_create(vk, o_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->out);
   gb_buf_upload(vk, q_cpu, &st->queries, q_bytes);
   gb_buf_upload(vk, c_cpu, &st->cands, c_bytes);
   gb_buf_upload(vk, w_cpu, &st->weights, w_bytes);
   free(q_cpu);
   free(c_cpu);
   free(w_cpu);

   gb_compute_pipe(vk, GB_SPV(desc_search_comp), 4, 16, &st->p);
   struct gb_buf *b[4] = {&st->queries, &st->cands, &st->weights, &st->out};
   gb_pipe_set_buffers(vk, &st->p, b, 4);
   return st;
}

static void
desc_search_teardown(struct gb_vk *vk, void *p)
{
   struct desc_search_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->queries);
   gb_buf_destroy(vk, &st->cands);
   gb_buf_destroy(vk, &st->weights);
   gb_buf_destroy(vk, &st->out);
   free(st->ref);
   free(st);
}

static void
desc_search_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct desc_search_state *st = p;
   struct { int32_t num_queries, num_cands, stride, flags; } pc = {
      (int32_t)st->num_queries, (int32_t)st->num_cands, (int32_t)st->stride, (int32_t)st->flags,
   };
   uint32_t gx = (st->num_queries + GB_DESC_LOCAL - 1u) / GB_DESC_LOCAL;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, gx, 1, 1);
}

static void
desc_search_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long queries = gb_param_int(o, "queries", 8192);
   long cands = gb_param_int(o, "cands", 256);
   gb_derive_rate(r, "mpairs_s", "Mpairs/s", us, (double)queries * (double)cands);
}

static void
desc_search_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct desc_search_state *st = p;
   uint32_t *got = malloc((size_t)st->num_queries * sizeof(uint32_t));
   gb_buf_readback(vk, &st->out, got, (size_t)st->num_queries * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, st->num_queries);
   free(got);
}

static const char *const desc_search_params[] = {"queries", "cands", "stride", NULL};
static const struct gb_stream_test t_compute_desc_search = {
   .name = "compute.desc_search",
   .known_params = desc_search_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = desc_search_setup,
   .teardown = desc_search_teardown,
   .record = desc_search_record,
   .derive = desc_search_derive,
   .check = desc_search_check,
};

int gb_run_compute_desc_search(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_desc_search); }

/* =========================================================================
 * 4. compute.chain (back-to-back dependent compute dispatch chain)
 * ========================================================================= */

struct chain_state {
   uint32_t count, passes;
   VkPipeline pipe;
   VkPipelineLayout layout;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set_init; /* init -> buf[0] */
   VkDescriptorSet set_01;   /* buf[0] -> buf[1] */
   VkDescriptorSet set_10;   /* buf[1] -> buf[0] */
   struct gb_buf init_buf, buf[2];
   uint32_t *ref;
   uint32_t ref_crc;
};

static void
write_pair_set(struct gb_vk *vk, VkDescriptorSet set, const struct gb_buf *src, const struct gb_buf *dst)
{
   VkDescriptorBufferInfo bi[2] = {
      {src->buf, 0, src->size},
      {dst->buf, 0, dst->size},
   };
   VkWriteDescriptorSet wr[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1]},
   };
   vkUpdateDescriptorSets(vk->dev, 2, wr, 0, NULL);
}

static bool
parse_chain_params(const struct gb_opts *o, uint32_t *count, uint32_t *passes, bool *split, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "single_cb");
   if (!strcmp(mode, "single_cb")) *split = false;
   else if (!strcmp(mode, "split_submits")) *split = true;
   else { snprintf(err, n, "mode must be single_cb or split_submits"); return false; }

   const char *sz = gb_param(o, "size", "small");
   if (!strcmp(sz, "small")) *count = GB_CHAIN_SMALL_WORDS;
   else if (!strcmp(sz, "medium")) *count = GB_CHAIN_MED_WORDS;
   else { snprintf(err, n, "size must be small or medium"); return false; }

   long ps = gb_param_int(o, "passes", 32);
   if (ps < 2 || ps > 128) {
      snprintf(err, n, "passes must be 2..128");
      return false;
   }
   *passes = (uint32_t)ps;
   return true;
}

static void *
chain_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   uint32_t count = 0, passes = 0;
   bool split = false;
   if (!parse_chain_params(o, &count, &passes, &split, err, n))
      return NULL;

   struct chain_state *st = calloc(1, sizeof(*st));
   st->count = count;
   st->passes = passes;
   size_t bytes = (size_t)count * sizeof(uint32_t);
   uint32_t *init_cpu = malloc(bytes);
   gb_chain_fill_init(count, init_cpu);
   st->ref = malloc(bytes);
   st->ref_crc = gb_chain_ref(count, passes, init_cpu, st->ref);

   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->init_buf);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->buf[0]);
   gb_buf_create(vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->buf[1]);
   gb_buf_upload(vk, init_cpu, &st->init_buf, bytes);
   free(init_cpu);

   VkDescriptorSetLayoutBinding bs[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bs};
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk->dev, &dci, NULL, &st->dsl));

   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
   VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 3, .poolSizeCount = 1, .pPoolSizes = &ps};
   GB_VK_CHECK(vkCreateDescriptorPool(vk->dev, &pci, NULL, &st->dpool));

   VkDescriptorSetLayout dsls[3] = {st->dsl, st->dsl, st->dsl};
   VkDescriptorSet sets[3];
   VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = st->dpool, .descriptorSetCount = 3, .pSetLayouts = dsls};
   GB_VK_CHECK(vkAllocateDescriptorSets(vk->dev, &sai, sets));
   st->set_init = sets[0];
   st->set_01 = sets[1];
   st->set_10 = sets[2];
   write_pair_set(vk, st->set_init, &st->init_buf, &st->buf[0]);
   write_pair_set(vk, st->set_01, &st->buf[0], &st->buf[1]);
   write_pair_set(vk, st->set_10, &st->buf[1], &st->buf[0]);

   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &st->dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &st->layout));
   VkShaderModule m = gb_shader(vk, GB_SPV(chain_step_comp));
   VkComputePipelineCreateInfo cci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
      .layout = st->layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk->dev, VK_NULL_HANDLE, 1, &cci, NULL, &st->pipe));
   vkDestroyShaderModule(vk->dev, m, NULL);
   return st;
}

static void
chain_teardown(struct gb_vk *vk, void *p)
{
   struct chain_state *st = p;
   vkDestroyPipeline(vk->dev, st->pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   vkDestroyDescriptorPool(vk->dev, st->dpool, NULL);
   vkDestroyDescriptorSetLayout(vk->dev, st->dsl, NULL);
   gb_buf_destroy(vk, &st->init_buf);
   gb_buf_destroy(vk, &st->buf[0]);
   gb_buf_destroy(vk, &st->buf[1]);
   free(st->ref);
   free(st);
}

static void
chain_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct chain_state *st = p;
   uint32_t gx = (st->count + GB_CHAIN_LOCAL - 1u) / GB_CHAIN_LOCAL;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipe);
   for (uint32_t pass = 0; pass < st->passes; pass++) {
      if (pass > 0u)
         compute_barrier_w3(cb);
      VkDescriptorSet set = (pass == 0u) ? st->set_init : ((pass & 1u) ? st->set_01 : st->set_10);
      struct { uint32_t count, mask, salt; } pc = {st->count, st->count - 1u, 0x9e3779b9u * (pass + 1u)};
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->layout, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, gx, 1, 1);
   }
}

static void
chain_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long passes = gb_param_int(o, "passes", 32);
   gb_derive_rate(r, "kpass_s", "kpass/s", us, (double)passes * 1e3);
   struct gb_samples upp = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&upp, us->v[i] / (double)passes);
   gb_result_metric(r, "us_per_pass", "us", true, &upp, false);
   gb_samples_free(&upp);
}

static void
chain_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct chain_state *st = p;
   const struct gb_buf *final_buf = &st->buf[(st->passes - 1u) & 1u];
   uint32_t *got = malloc((size_t)st->count * sizeof(uint32_t));
   gb_buf_readback(vk, final_buf, got, (size_t)st->count * sizeof(uint32_t));
   gb_check_words(chk, got, st->ref, st->ref_crc, st->count);
   free(got);
}

static const char *const chain_params[] = {"mode", "passes", "size", NULL};
static const struct gb_stream_test t_compute_chain = {
   .name = "compute.chain",
   .known_params = chain_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = chain_setup,
   .teardown = chain_teardown,
   .record = chain_record,
   .derive = chain_derive,
   .check = chain_check,
};

/* Split-submits variant: each pass in the dependent chain is submitted in its
 * own vkQueueSubmit2 chained via a timeline semaphore, with per-slot ping-pong
 * buffers so depth > 1 pipelines independent chains without cross-chain hazards. */
static int
run_chain_split(const struct gb_opts *o, uint32_t count, uint32_t passes)
{
   const int default_depths[2] = {1, 4};
   const int *depths = o->ndepths ? o->depths : default_depths;
   int nd = o->ndepths ? o->ndepths : 2;
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull;
   const uint64_t R = (uint64_t)o->run_ms * 1000000ull;

   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);

   size_t bytes = (size_t)count * sizeof(uint32_t);
   uint32_t *init_cpu = malloc(bytes);
   gb_chain_fill_init(count, init_cpu);
   uint32_t *ref = malloc(bytes);
   uint32_t ref_crc = gb_chain_ref(count, passes, init_cpu, ref);

   struct gb_buf init_buf;
   gb_buf_create(&vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &init_buf);
   gb_buf_upload(&vk, init_cpu, &init_buf, bytes);
   free(init_cpu);

   VkDescriptorSetLayoutBinding bs[2] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   VkDescriptorSetLayout dsl;
   VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bs};
   GB_VK_CHECK(vkCreateDescriptorSetLayout(vk.dev, &dci, NULL, &dsl));
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
   VkPipelineLayout layout;
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk.dev, &lci, NULL, &layout));
   VkShaderModule mod = gb_shader(&vk, GB_SPV(chain_step_comp));
   VkPipeline pipe;
   VkComputePipelineCreateInfo cci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"},
      .layout = layout,
   };
   GB_VK_CHECK(vkCreateComputePipelines(vk.dev, VK_NULL_HANDLE, 1, &cci, NULL, &pipe));
   vkDestroyShaderModule(vk.dev, mod, NULL);

   uint32_t gx = (count + GB_CHAIN_LOCAL - 1u) / GB_CHAIN_LOCAL;

   for (int di = 0; di < nd; di++) {
      int depth = depths[di];
      if (depth < 1) depth = 1;

      struct gb_buf (*slot_bufs)[2] = calloc((size_t)depth, sizeof(*slot_bufs));
      VkCommandPool *pools = calloc((size_t)depth, sizeof(*pools));
      VkCommandBuffer *cbs = calloc((size_t)depth * passes, sizeof(*cbs));
      VkDescriptorSet *sets = calloc((size_t)depth * 3u, sizeof(*sets));
      uint64_t *slot_end_val = calloc((size_t)depth, sizeof(*slot_end_val));
      uint64_t *slot_submit_ns = calloc((size_t)depth, sizeof(*slot_submit_ns));
      VkSemaphore *slot_tl = calloc((size_t)depth, sizeof(*slot_tl));

      VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)depth * 6u};
      VkDescriptorPool dpool;
      VkDescriptorPoolCreateInfo dpci = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets = (uint32_t)depth * 3u, .poolSizeCount = 1, .pPoolSizes = &dps,
      };
      GB_VK_CHECK(vkCreateDescriptorPool(vk.dev, &dpci, NULL, &dpool));

      for (int s = 0; s < depth; s++) {
         gb_buf_create(&vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &slot_bufs[s][0]);
         gb_buf_create(&vk, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &slot_bufs[s][1]);
         VkDescriptorSetLayout dsls[3] = {dsl, dsl, dsl};
         VkDescriptorSetAllocateInfo sai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = dpool, .descriptorSetCount = 3, .pSetLayouts = dsls,
         };
         GB_VK_CHECK(vkAllocateDescriptorSets(vk.dev, &sai, &sets[s * 3]));
         write_pair_set(&vk, sets[s * 3 + 0], &init_buf, &slot_bufs[s][0]);
         write_pair_set(&vk, sets[s * 3 + 1], &slot_bufs[s][0], &slot_bufs[s][1]);
         write_pair_set(&vk, sets[s * 3 + 2], &slot_bufs[s][1], &slot_bufs[s][0]);

         VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = vk.qfam};
         GB_VK_CHECK(vkCreateCommandPool(vk.dev, &cpci, NULL, &pools[s]));
         VkCommandBufferAllocateInfo cbai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pools[s], .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = passes,
         };
         GB_VK_CHECK(vkAllocateCommandBuffers(vk.dev, &cbai, &cbs[(size_t)s * passes]));

         VkSemaphoreTypeCreateInfo stci = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0,
         };
         VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &stci};
         GB_VK_CHECK(vkCreateSemaphore(vk.dev, &sci, NULL, &slot_tl[s]));
      }

      struct gb_pass pass_res = {.nwin = o->runs, .win_s = (double)R / 1e9};
      uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
      uint64_t k = 0;
      int last_slot = 0;
      bool bad = false;

      while (!bad) {
         int s = (int)(k % (uint64_t)depth);
         if (slot_end_val[s] > 0) {
            VkSemaphoreWaitInfo wi = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
               .semaphoreCount = 1, .pSemaphores = &slot_tl[s], .pValues = &slot_end_val[s],
            };
            if (vkWaitSemaphores(vk.dev, &wi, W3_GPU_TIMEOUT_NS) != VK_SUCCESS) {
               bad = true;
               break;
            }
            uint64_t done_ns = gb_now_ns();
            if (done_ns >= tm0 && done_ns < t_end) {
               int w = (int)((done_ns - tm0) / R);
               if (w >= 0 && w < pass_res.nwin)
                  pass_res.win_count[w]++;
               gb_samples_push(&pass_res.latency, (double)(done_ns - slot_submit_ns[s]) / 1e3);
               pass_res.submits++;
            }
            if (done_ns >= t_end)
               break;
         }

         uint64_t tr0 = gb_now_ns();
         GB_VK_CHECK(vkResetCommandPool(vk.dev, pools[s], 0));
         VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
         };
         for (uint32_t p = 0; p < passes; p++) {
            VkCommandBuffer cb = cbs[(size_t)s * passes + p];
            GB_VK_CHECK(vkBeginCommandBuffer(cb, &bi));
            if (p > 0u)
               compute_barrier_w3(cb);
            VkDescriptorSet set = (p == 0u) ? sets[s * 3 + 0] : ((p & 1u) ? sets[s * 3 + 1] : sets[s * 3 + 2]);
            struct { uint32_t count, mask, salt; } pc = {count, count - 1u, 0x9e3779b9u * (p + 1u)};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, gx, 1, 1);
            GB_VK_CHECK(vkEndCommandBuffer(cb));
         }
         uint64_t tr1 = gb_now_ns();

         uint64_t base_v = slot_end_val[s];
         slot_submit_ns[s] = tr1;
         for (uint32_t p = 0; p < passes; p++) {
            VkCommandBufferSubmitInfo cbsi = {
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
               .commandBuffer = cbs[(size_t)s * passes + p],
            };
            VkSemaphoreSubmitInfo winfo = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
               .semaphore = slot_tl[s], .value = base_v + p,
               .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            };
            VkSemaphoreSubmitInfo sinfo = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
               .semaphore = slot_tl[s], .value = base_v + p + 1u,
               .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            };
            VkSubmitInfo2 si2 = {
               .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
               .waitSemaphoreInfoCount = p > 0u ? 1u : 0u,
               .pWaitSemaphoreInfos = p > 0u ? &winfo : NULL,
               .commandBufferInfoCount = 1,
               .pCommandBufferInfos = &cbsi,
               .signalSemaphoreInfoCount = 1,
               .pSignalSemaphoreInfos = &sinfo,
            };
            GB_VK_CHECK(vkQueueSubmit2(vk.queues[0], 1, &si2, VK_NULL_HANDLE));
         }
         uint64_t ts1 = gb_now_ns();
         slot_end_val[s] = base_v + passes;
         last_slot = s;
         if (ts1 >= tm0 && ts1 < t_end) {
            gb_samples_push(&pass_res.record_cpu, (double)(tr1 - tr0) / 1e3);
            gb_samples_push(&pass_res.submit_cpu, (double)(ts1 - tr1) / 1e3);
         }
         k++;
      }

      /* Drain all in-flight slots before readback and teardown */
      for (int s = 0; s < depth; s++) {
         if (slot_end_val[s] > 0) {
            VkSemaphoreWaitInfo wi = {
               .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
               .semaphoreCount = 1, .pSemaphores = &slot_tl[s], .pValues = &slot_end_val[s],
            };
            if (vkWaitSemaphores(vk.dev, &wi, W3_GPU_TIMEOUT_NS) != VK_SUCCESS)
               bad = true;
         }
      }

      struct gb_check chk = {0};
      if (!bad) {
         uint32_t *got = malloc(bytes);
         gb_buf_readback(&vk, &slot_bufs[last_slot][(passes - 1u) & 1u], got, bytes);
         gb_check_words(&chk, got, ref, ref_crc, count);
         free(got);
      }

      struct gb_samples us_rate = {0};
      gb_pass_rate(&pass_res, &us_rate);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depth, NULL);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad || !us_rate.n) {
         gb_result_finish(&r, "fail", NULL, bad ? "GPU wait timed out" : "no completions");
      } else {
         gb_result_stream_metrics(&r, &pass_res, NULL);
         chain_derive(o, &r, &us_rate);
         gb_result_end_metrics(&r);
         gb_json_obj_begin(&r.j, "counts");
         gb_json_int(&r.j, "submits", (int64_t)pass_res.submits);
         gb_json_obj_end(&r.j);
         gb_result_check(&r, &chk);
         gb_result_finish(&r, chk.valid ? "ok" : "fail", "us_per_submit",
                          chk.valid ? NULL : "output mismatch");
      }
      gb_samples_free(&us_rate);
      gb_pass_free(&pass_res);

      for (int s = 0; s < depth; s++) {
         vkDestroySemaphore(vk.dev, slot_tl[s], NULL);
         vkDestroyCommandPool(vk.dev, pools[s], NULL);
         gb_buf_destroy(&vk, &slot_bufs[s][0]);
         gb_buf_destroy(&vk, &slot_bufs[s][1]);
      }
      vkDestroyDescriptorPool(vk.dev, dpool, NULL);
      free(slot_tl);
      free(slot_submit_ns);
      free(slot_end_val);
      free(sets);
      free(cbs);
      free(pools);
      free(slot_bufs);
   }

   vkDestroyPipeline(vk.dev, pipe, NULL);
   vkDestroyPipelineLayout(vk.dev, layout, NULL);
   vkDestroyDescriptorSetLayout(vk.dev, dsl, NULL);
   gb_buf_destroy(&vk, &init_buf);
   free(ref);
   gb_vk_destroy(&vk);
   return 0;
}

int
gb_run_compute_chain(const struct gb_opts *o)
{
   if (!gb_check_params(o, chain_params)) {
      gb_result_simple(o, o->mode, 0, NULL, "fail", "unknown parameter");
      return 1;
   }
   uint32_t count = 0, passes = 0;
   bool split = false;
   char err[128] = "";
   if (!parse_chain_params(o, &count, &passes, &split, err, sizeof(err))) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", err);
      return 1;
   }
   if (split) {
      if (o->mode != GB_MODE_QUEUE) {
         gb_result_simple(o, o->mode, 1, NULL, "skip", "split_submits supports mode=queue only");
         return 0;
      }
      return run_chain_split(o, count, passes);
   }
   return gb_run_stream_test(o, &t_compute_chain);
}

/* =========================================================================
 * 5. render.msaa_merge (2x/4x MSAA multi-pass color-compressed target merge)
 * ========================================================================= */

struct msaa_merge_state {
   uint32_t w, h, samples, passes;
   bool is_fp16;
   VkFormat fmt;
   struct gb_img msaa_img, res_img;
   VkPipelineLayout layout;
   VkPipeline pipe;
   void *ref;
   uint32_t ref_crc;
};

static void *
msaa_merge_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long samples = gb_param_int(o, "samples", 4);
   if (samples != 2 && samples != 4) {
      snprintf(err, n, "samples must be 2 or 4");
      return NULL;
   }
   const char *fmt_s = gb_param(o, "fmt", "rgba8");
   bool is_fp16 = false;
   if (!strcmp(fmt_s, "rgba16f")) is_fp16 = true;
   else if (strcmp(fmt_s, "rgba8")) {
      snprintf(err, n, "fmt must be rgba8 or rgba16f");
      return NULL;
   }
   long passes = gb_param_int(o, "passes", 4);
   if (passes < 2 || passes > 8) {
      snprintf(err, n, "passes must be 2..8");
      return NULL;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res_w3(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }

   VkSampleCountFlagBits sc = (samples == 2) ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_4_BIT;
   if ((vk->props.limits.framebufferColorSampleCounts & sc) == 0) {
      snprintf(err, n, "device lacks %ldx color MSAA", samples);
      return NULL;
   }

   struct msaa_merge_state *st = calloc(1, sizeof(*st));
   st->w = w; st->h = h;
   st->samples = (uint32_t)samples;
   st->passes = (uint32_t)passes;
   st->is_fp16 = is_fp16;
   st->fmt = is_fp16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

   if (is_fp16) {
      st->ref = malloc((size_t)w * h * 8u);
      st->ref_crc = gb_msaa_merge_ref_rgba16f(w, h, st->passes, (uint16_t *)st->ref);
   } else {
      st->ref = malloc((size_t)w * h * 4u);
      st->ref_crc = gb_msaa_merge_ref_rgba8(w, h, st->passes, (uint32_t *)st->ref);
   }

   gb_img_create_ex(vk, w, h, st->fmt, sc, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &st->msaa_img);
   gb_img_create(vk, w, h, st->fmt,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &st->res_img);

   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, st->msaa_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_img_barrier(cb, st->res_img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);

   VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   GB_VK_CHECK(vkCreatePipelineLayout(vk->dev, &lci, NULL, &st->layout));

   VkShaderModule mv = gb_shader(vk, GB_SPV(fs_vert));
   VkShaderModule mf = gb_shader(vk, GB_SPV(msaa_merge_frag));
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
      .rasterizationSamples = sc,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = 0xf,
   };
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dyns = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &st->fmt,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs, .pDynamicState = &dyns,
      .layout = st->layout,
   };
   GB_VK_CHECK(vkCreateGraphicsPipelines(vk->dev, VK_NULL_HANDLE, 1, &gci, NULL, &st->pipe));
   vkDestroyShaderModule(vk->dev, mv, NULL);
   vkDestroyShaderModule(vk->dev, mf, NULL);
   return st;
}

static void
msaa_merge_teardown(struct gb_vk *vk, void *p)
{
   struct msaa_merge_state *st = p;
   vkDestroyPipeline(vk->dev, st->pipe, NULL);
   vkDestroyPipelineLayout(vk->dev, st->layout, NULL);
   gb_img_destroy(vk, &st->msaa_img);
   gb_img_destroy(vk, &st->res_img);
   free(st->ref);
   free(st);
}

static void
msaa_merge_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct msaa_merge_state *st = p;
   VkViewport vp = {0, 0, (float)st->w, (float)st->h, 0.0f, 1.0f};
   VkRect2D sc = {{0, 0}, {st->w, st->h}};
   float scale = st->is_fp16 ? (1.0f / 256.0f) : (1.0f / 255.0f);

   for (uint32_t pass = 0; pass < st->passes; pass++) {
      if (pass > 0u) {
         /* Framebuffer-local attachment dependency between consecutive MSAA passes
          * on the same color attachment (eligible for TBDR MSAA pass merging). */
         VkMemoryBarrier2 mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
         };
         VkDependencyInfo di = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
         };
         vkCmdPipelineBarrier2(cb, &di);
      }
      bool last = (pass + 1u == st->passes);
      VkRenderingAttachmentInfo ca = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = st->msaa_img.view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .resolveMode = last ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
         .resolveImageView = last ? st->res_img.view : VK_NULL_HANDLE,
         .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = (pass == 0u) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = last ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 0.0f},
      };
      VkRenderingInfo ri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {st->w, st->h}},
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &ca,
      };
      vkCmdBeginRendering(cb, &ri);
      vkCmdSetViewport(cb, 0, 1, &vp);
      vkCmdSetScissor(cb, 0, 1, &sc);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->pipe);
      struct { uint32_t pass_idx, passes; float scale; uint32_t pad; } pc = {pass, st->passes, scale, 0u};
      vkCmdPushConstants(cb, st->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
   }
}

static void
msaa_merge_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res_w3(gb_param(o, "res", "1080p"), &w, &h);
   long passes = gb_param_int(o, "passes", 4);
   gb_derive_rate(r, "fps", "fps", us, 1e6);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h * (double)passes / 1e3);
}

static void
msaa_merge_check(struct gb_vk *vk, void *p, struct gb_check *chk)
{
   struct msaa_merge_state *st = p;
   size_t bpp = st->is_fp16 ? 8u : 4u;
   size_t bytes = (size_t)st->w * st->h * bpp;
   uint32_t *got = malloc(bytes);
   gb_img_readback(vk, &st->res_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, got, bpp);
   gb_check_words(chk, got, (const uint32_t *)st->ref, st->ref_crc, bytes / sizeof(uint32_t));
   free(got);
}

static const char *const msaa_merge_params[] = {"samples", "fmt", "passes", "res", NULL};
static const struct gb_stream_test t_render_msaa_merge = {
   .name = "render.msaa_merge",
   .known_params = msaa_merge_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = msaa_merge_setup,
   .teardown = msaa_merge_teardown,
   .record = msaa_merge_record,
   .derive = msaa_merge_derive,
   .check = msaa_merge_check,
};

int gb_run_render_msaa_merge(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_msaa_merge); }
