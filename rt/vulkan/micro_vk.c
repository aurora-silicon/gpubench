// SPDX-License-Identifier: GPL-2.0-only
/* RT micro-benchmarks on Vulkan.
 *   rt.asbuild: one acceleration-structure build per submission (kind=blas: BLAS build of a
 *               `tris`-triangle mesh; kind=refit: BLAS update of the same; kind=tlas: TLAS build of
 *               `tris` instances), build=fasttrace|fastbuild.
 *   rt.rays:    one dispatch of `rays` closest-hit ray queries per submission against the static
 *               scene, kind=coherent|incoherent.
 *   rt.trav:    the same against synthetic geometry that isolates one traversal step
 *               (common/trav.h).
 * Both stream at depth N (ring of N submissions in flight, each slot with its own resources, no
 * dependencies between submissions). Timed pass: timestamps at the start and the end of each
 * command buffer. */
#include "../common/result.h"
#include "../common/scene.h"
#include "../common/trav.h"
#include "../common/util.h"
#include "rtvk.h"
#include "vkstream.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct micro {
   vk_ctx *c;
   const rt_params *p;
   int test;
   VkQueryPool qp;
   int ts_ok;
   double ts_ns;
   uint64_t last_end;
   rt_vec gpu_us, gap_us;
   /* asbuild */
   vk_buf vbuf, ibuf, inst, blas_buf, scratch[VK_STREAM_MAX];
   VkAccelerationStructureKHR blas, as[VK_STREAM_MAX];
   vk_buf asbuf[VK_STREAM_MAX];
   VkAccelerationStructureGeometryKHR geom;
   VkAccelerationStructureBuildRangeInfoKHR range;
   VkAccelerationStructureBuildSizesInfoKHR sizes;
   VkBuildAccelerationStructureFlagsKHR flags;
   VkAccelerationStructureTypeKHR type;
   int refit;
   uint32_t prims, nverts;
   /* rays */
   struct scn *scene;
   VkDescriptorSetLayout dsl;
   VkPipelineLayout pl;
   VkPipeline pipe;
   VkDescriptorPool dp;
   VkDescriptorSet set[VK_STREAM_MAX];
   vk_buf out[VK_STREAM_MAX];
   struct {
      float eye[4], fwd[4], right[4], up[4];
      uint32_t mode, nrays, seed, width;
   } pc;
   uint32_t nslots;
   uint64_t compacted;
   /* trav */
   rt_trav_rays tpc;
   VkAccelerationStructureKHR tlas;
   vk_buf tlas_buf;
   VkDeviceSize blas_bytes;
} micro;

/* Compacted size of slot 0's structure (after the run, so it is a finished build). */
static void query_compacted(micro *m)
{
   vk_ctx *c = m->c;
   VkQueryPool qp;
   VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                               .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                               .queryCount = 1};
   if (vkCreateQueryPool(c->dev, &qi, NULL, &qp) != VK_SUCCESS) return;
   vkResetQueryPool(c->dev, qp, 0, 1);
   VkCommandBuffer cb;
   if (vk_oneshot_begin(c, &cb) == VK_SUCCESS) {
      vk_barrier(cb, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
      c->CmdWriteASProps(cb, 1, &m->as[0], VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, qp, 0);
      if (vk_oneshot_end(c, cb) == VK_SUCCESS)
         vkGetQueryPoolResults(c->dev, qp, 0, 1, 8, &m->compacted, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   }
   vkDestroyQueryPool(c->dev, qp, NULL);
}

static void ts_begin(micro *m, VkCommandBuffer cb, uint32_t slot, int timed)
{
   if (timed && m->ts_ok) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m->qp, slot * 2);
}
static void ts_end(micro *m, VkCommandBuffer cb, uint32_t slot, int timed)
{
   if (timed && m->ts_ok) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m->qp, slot * 2 + 1);
}

static void complete(void *user, uint32_t slot, uint64_t k, int timed)
{
   (void)k;
   micro *m = user;
   if (!timed || !m->ts_ok) { m->last_end = 0; return; }
   uint64_t ts[2];
   if (vkGetQueryPoolResults(m->c->dev, m->qp, slot * 2, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      rt_vec_push(&m->gpu_us, (ts[1] - ts[0]) * m->ts_ns / 1e3);
      if (m->last_end && ts[0] > m->last_end) rt_vec_push(&m->gap_us, (ts[0] - m->last_end) * m->ts_ns / 1e3);
      else if (m->last_end) rt_vec_push(&m->gap_us, 0);
      m->last_end = ts[1];
   }
   vkResetQueryPool(m->c->dev, m->qp, slot * 2, 2);
}

/* ---- rt.asbuild ---- */

static VkResult record_build(void *user, uint32_t slot, uint64_t k, VkCommandBuffer cb, int timed)
{
   (void)k;
   micro *m = user;
   ts_begin(m, cb, slot, timed);
   VkAccelerationStructureBuildGeometryInfoKHR bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = m->type,
      .flags = m->flags,
      .mode = m->refit ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .srcAccelerationStructure = m->refit ? m->as[slot] : VK_NULL_HANDLE,
      .dstAccelerationStructure = m->as[slot],
      .geometryCount = 1,
      .pGeometries = &m->geom,
      .scratchData.deviceAddress = m->scratch[slot].addr};
   const VkAccelerationStructureBuildRangeInfoKHR *rp = &m->range;
   m->c->CmdBuildAS(cb, 1, &bi, &rp);
   ts_end(m, cb, slot, timed);
   return VK_SUCCESS;
}

static VkResult as_create(micro *m, vk_buf *buf, VkAccelerationStructureKHR *as, VkDeviceSize size,
                          VkAccelerationStructureTypeKHR type)
{
   vk_ctx *c = m->c;
   VKC(vk_buf_create(c, buf, size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
   VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                              .buffer = buf->buf,
                                              .size = size,
                                              .type = type};
   return c->CreateAS(c->dev, &ci, NULL, as);
}

static VkResult setup_asbuild(micro *m, uint32_t depth)
{
   vk_ctx *c = m->c;
   const rt_params *p = m->p;
   const VkBufferUsageFlags geo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   int tlas = !strcmp(p->kind, "tlas");
   m->refit = !strcmp(p->kind, "refit");
   int fastbuild = !strcmp(p->build, "fastbuild");
   m->flags = fastbuild ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
                        : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
   if (m->refit) m->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
   if (p->compact) m->flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
   rt_vertex *v;
   uint32_t *idx, nv, ni;
   rt_gen_mesh(RT_KIND_ROCK, tlas ? 1024 : p->tris, &v, &nv, &idx, &ni);
   VKC(vk_buf_create(c, &m->vbuf, (VkDeviceSize)nv * sizeof(rt_vertex), geo, 1));
   memcpy(m->vbuf.map, v, (size_t)nv * sizeof(rt_vertex));
   VKC(vk_buf_create(c, &m->ibuf, (VkDeviceSize)ni * 4, geo, 1));
   memcpy(m->ibuf.map, idx, (size_t)ni * 4);
   free(v);
   free(idx);
   m->nverts = nv;
   VkAccelerationStructureGeometryKHR tri = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
      .geometry.triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                             .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                             .vertexData.deviceAddress = m->vbuf.addr,
                             .vertexStride = sizeof(rt_vertex),
                             .maxVertex = nv - 1,
                             .indexType = VK_INDEX_TYPE_UINT32,
                             .indexData.deviceAddress = m->ibuf.addr}};
   if (tlas) {
      /* one 1K-triangle BLAS instanced `tris` times on a jittered grid */
      VkAccelerationStructureBuildGeometryInfoKHR bi = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
         .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
         .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
         .geometryCount = 1,
         .pGeometries = &tri};
      uint32_t np = ni / 3;
      VkAccelerationStructureBuildSizesInfoKHR bs = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
      c->GetASBuildSizes(c->dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &np, &bs);
      VKC(as_create(m, &m->blas_buf, &m->blas, bs.accelerationStructureSize, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR));
      vk_buf scr;
      VKC(vk_buf_create(c, &scr, bs.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
      bi.dstAccelerationStructure = m->blas;
      bi.scratchData.deviceAddress = scr.addr;
      VkAccelerationStructureBuildRangeInfoKHR r1 = {.primitiveCount = np};
      const VkAccelerationStructureBuildRangeInfoKHR *rp = &r1;
      VkCommandBuffer cb;
      VKC(vk_oneshot_begin(c, &cb));
      c->CmdBuildAS(cb, 1, &bi, &rp);
      VKC(vk_oneshot_end(c, cb));
      vk_buf_destroy(c, &scr);
      VkAccelerationStructureDeviceAddressInfoKHR ai = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .accelerationStructure = m->blas};
      VkDeviceAddress ba = c->GetASAddress(c->dev, &ai);
      uint32_t n = p->tris;
      VKC(vk_buf_create(c, &m->inst, (VkDeviceSize)n * sizeof(VkAccelerationStructureInstanceKHR),
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 1));
      VkAccelerationStructureInstanceKHR *in = m->inst.map;
      uint32_t side = (uint32_t)ceil(sqrt((double)n));
      for (uint32_t i = 0; i < n; i++) {
         memset(&in[i], 0, sizeof(in[i]));
         float x = (float)(i % side) * 3.0f, z = (float)(i / side) * 3.0f;
         float a = (float)((i * 2654435761u) >> 8) * (6.2831853f / 16777216.0f);
         in[i].transform.matrix[0][0] = cosf(a); in[i].transform.matrix[0][2] = sinf(a); in[i].transform.matrix[0][3] = x;
         in[i].transform.matrix[1][1] = 1; in[i].transform.matrix[1][3] = (float)(i % 7) * 0.5f;
         in[i].transform.matrix[2][0] = -sinf(a); in[i].transform.matrix[2][2] = cosf(a); in[i].transform.matrix[2][3] = z;
         in[i].instanceCustomIndex = i;
         in[i].mask = 0xff;
         in[i].accelerationStructureReference = ba;
      }
      m->geom = (VkAccelerationStructureGeometryKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
         .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
         .geometry.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                .data.deviceAddress = m->inst.addr}};
      m->type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
      m->prims = n;
   } else {
      m->geom = tri;
      m->type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      m->prims = ni / 3;
   }
   m->range.primitiveCount = m->prims;
   VkAccelerationStructureBuildGeometryInfoKHR bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = m->type,
      .flags = m->flags,
      .geometryCount = 1,
      .pGeometries = &m->geom};
   m->sizes = (VkAccelerationStructureBuildSizesInfoKHR){.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
   c->GetASBuildSizes(c->dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &m->prims, &m->sizes);
   VkDeviceSize scr = m->refit ? m->sizes.updateScratchSize : m->sizes.buildScratchSize;
   if (m->sizes.buildScratchSize > scr) scr = m->sizes.buildScratchSize;
   m->nslots = depth;
   for (uint32_t i = 0; i < depth; i++) {
      VKC(as_create(m, &m->asbuf[i], &m->as[i], m->sizes.accelerationStructureSize, m->type));
      VKC(vk_buf_create(c, &m->scratch[i], scr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
   }
   if (m->refit) {
      /* the source of every update is a full build */
      VkCommandBuffer cb;
      VKC(vk_oneshot_begin(c, &cb));
      int save = m->refit;
      m->refit = 0;
      for (uint32_t i = 0; i < depth; i++) record_build(m, i, 0, cb, 0);
      m->refit = save;
      VKC(vk_oneshot_end(c, cb));
   }
   return VK_SUCCESS;
}

/* ---- rt.rays ---- */

static VkResult record_rays(void *user, uint32_t slot, uint64_t k, VkCommandBuffer cb, int timed)
{
   micro *m = user;
   ts_begin(m, cb, slot, timed);
   m->pc.seed = (uint32_t)k;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m->pl, 0, 1, &m->set[slot], 0, NULL);
   vkCmdPushConstants(cb, m->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m->pc), &m->pc);
   vkCmdDispatch(cb, (m->p->rays + 63) / 64, 1, 1);
   ts_end(m, cb, slot, timed);
   return VK_SUCCESS;
}

/* The ray-query pipeline `shader` (push constants of `pc_bytes`) and a descriptor set per slot
 * naming `tl` and the slot's result buffer. */
static VkResult rays_pipeline(micro *m, uint32_t depth, const char *shader, uint32_t pc_bytes,
                              VkAccelerationStructureKHR tl)
{
   vk_ctx *c = m->c;
   const rt_params *p = m->p;
   VkDescriptorSetLayoutBinding b[2] = {
      {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
   VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 2, .pBindings = b};
   VKC(vkCreateDescriptorSetLayout(c->dev, &dci, NULL, &m->dsl));
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1, .pSetLayouts = &m->dsl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
   VKC(vkCreatePipelineLayout(c->dev, &pli, NULL, &m->pl));
   const vk_spv *x = vk_find_spv(shader);
   VkShaderModule mod;
   VKC(vk_shader(c, x->code, x->size, &mod));
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                     .flags = c->exec_props_enabled ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0,
                                     .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                               .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"},
                                     .layout = m->pl};
   vk_marker(m->test == RT_TEST_TRAV ? "trav" : "rays");
   VKC(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &ci, NULL, &m->pipe));
   vk_marker("internal");
   vkDestroyShaderModule(c->dev, mod, NULL);
   VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, depth},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, depth}};
   VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = depth, .poolSizeCount = 2, .pPoolSizes = ps};
   VKC(vkCreateDescriptorPool(c->dev, &pci, NULL, &m->dp));
   m->nslots = depth;
   for (uint32_t i = 0; i < depth; i++) {
      VKC(vk_buf_create(c, &m->out[i], (VkDeviceSize)p->rays * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0));
      VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = m->dp, .descriptorSetCount = 1, .pSetLayouts = &m->dsl};
      VKC(vkAllocateDescriptorSets(c->dev, &ai, &m->set[i]));
      VkWriteDescriptorSetAccelerationStructureKHR wa = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
         .accelerationStructureCount = 1, .pAccelerationStructures = &tl};
      VkDescriptorBufferInfo bo = {m->out[i].buf, 0, VK_WHOLE_SIZE};
      VkWriteDescriptorSet w[2] = {
         {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &wa, .dstSet = m->set[i], .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
         {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m->set[i], .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bo}};
      vkUpdateDescriptorSets(c->dev, 2, w, 0, NULL);
   }
   return VK_SUCCESS;
}

static VkResult setup_rays(micro *m, uint32_t depth)
{
   vk_ctx *c = m->c;
   const rt_params *p = m->p;
   m->scene = rt_vk_static_scene(c, p);
   if (!m->scene) return VK_ERROR_INITIALIZATION_FAILED;
   VKC(rays_pipeline(m, depth, "rays.comp", sizeof(m->pc), rt_vk_static_scene_tlas(m->scene)));
   const rt_scene *sc = rt_vk_static_scene_desc(m->scene);
   uint32_t width = 1024, h = (p->rays + width - 1) / width;
   rt_scene_camera(sc, 0, (float)width / (float)h, m->pc.eye, m->pc.fwd, m->pc.right, m->pc.up);
   m->pc.eye[3] = p->tmax;
   m->pc.mode = !strcmp(p->kind, "incoherent");
   m->pc.nrays = p->rays;
   m->pc.width = width;
   return VK_SUCCESS;
}

/* ---- rt.trav ---- */

static VkResult record_trav(void *user, uint32_t slot, uint64_t k, VkCommandBuffer cb, int timed)
{
   (void)k;
   micro *m = user;
   ts_begin(m, cb, slot, timed);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m->pl, 0, 1, &m->set[slot], 0, NULL);
   vkCmdPushConstants(cb, m->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m->tpc), &m->tpc);
   vkCmdDispatch(cb, (m->tpc.nrays + 63) / 64, 1, 1);
   ts_end(m, cb, slot, timed);
   return VK_SUCCESS;
}

/* Build `bi` (sizes queried for `count` primitives) into a new structure. */
static VkResult build_once(micro *m, VkAccelerationStructureBuildGeometryInfoKHR *bi, uint32_t count, vk_buf *buf,
                           VkAccelerationStructureKHR *as, VkDeviceSize *bytes)
{
   vk_ctx *c = m->c;
   VkAccelerationStructureBuildSizesInfoKHR bs = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
   c->GetASBuildSizes(c->dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, bi, &count, &bs);
   VKC(as_create(m, buf, as, bs.accelerationStructureSize, bi->type));
   vk_buf scr;
   VKC(vk_buf_create(c, &scr, bs.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
   bi->dstAccelerationStructure = *as;
   bi->scratchData.deviceAddress = scr.addr;
   VkAccelerationStructureBuildRangeInfoKHR r1 = {.primitiveCount = count};
   const VkAccelerationStructureBuildRangeInfoKHR *rp = &r1;
   VkCommandBuffer cb;
   VKC(vk_oneshot_begin(c, &cb));
   c->CmdBuildAS(cb, 1, bi, &rp);
   VKC(vk_oneshot_end(c, cb));
   vk_buf_destroy(c, &scr);
   if (bytes) *bytes = bs.accelerationStructureSize;
   return VK_SUCCESS;
}

static VkResult setup_trav(micro *m, uint32_t depth)
{
   vk_ctx *c = m->c;
   const rt_params *p = m->p;
   float *pos;
   if (rt_trav_geometry(p, &pos, &m->prims, &m->tpc)) return VK_ERROR_INITIALIZATION_FAILED;
   const VkBufferUsageFlags geo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   m->nverts = m->prims * 3;
   VKC(vk_buf_create(c, &m->vbuf, (VkDeviceSize)m->nverts * 12, geo, 1));
   memcpy(m->vbuf.map, pos, (size_t)m->nverts * 12);
   free(pos);
   VkAccelerationStructureGeometryKHR tri = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
      .geometry.triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                             .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                             .vertexData.deviceAddress = m->vbuf.addr,
                             .vertexStride = 12,
                             .maxVertex = m->nverts - 1,
                             .indexType = VK_INDEX_TYPE_NONE_KHR}};
   VkAccelerationStructureBuildGeometryInfoKHR bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .geometryCount = 1,
      .pGeometries = &tri};
   VKC(build_once(m, &bi, m->prims, &m->blas_buf, &m->blas, &m->blas_bytes));
   VkAccelerationStructureDeviceAddressInfoKHR ai = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .accelerationStructure = m->blas};
   VKC(vk_buf_create(c, &m->inst, sizeof(VkAccelerationStructureInstanceKHR),
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 1));
   VkAccelerationStructureInstanceKHR *in = m->inst.map;
   memset(in, 0, sizeof(*in));
   in->transform.matrix[0][0] = in->transform.matrix[1][1] = in->transform.matrix[2][2] = 1.0f;
   in->mask = 0xff;
   in->accelerationStructureReference = c->GetASAddress(c->dev, &ai);
   VkAccelerationStructureGeometryKHR ig = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                             .data.deviceAddress = m->inst.addr}};
   VkAccelerationStructureBuildGeometryInfoKHR ti = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .geometryCount = 1,
      .pGeometries = &ig};
   VKC(build_once(m, &ti, 1, &m->tlas_buf, &m->tlas, NULL));
   return rays_pipeline(m, depth, "trav.comp", sizeof(m->tpc), m->tlas);
}

static void teardown(micro *m)
{
   vk_ctx *c = m->c;
   vkDeviceWaitIdle(c->dev);
   for (uint32_t i = 0; i < VK_STREAM_MAX; i++) {
      if (m->as[i]) c->DestroyAS(c->dev, m->as[i], NULL);
      vk_buf_destroy(c, &m->asbuf[i]);
      vk_buf_destroy(c, &m->scratch[i]);
      vk_buf_destroy(c, &m->out[i]);
   }
   if (m->blas) c->DestroyAS(c->dev, m->blas, NULL);
   vk_buf_destroy(c, &m->blas_buf);
   if (m->tlas) c->DestroyAS(c->dev, m->tlas, NULL);
   vk_buf_destroy(c, &m->tlas_buf);
   vk_buf_destroy(c, &m->vbuf);
   vk_buf_destroy(c, &m->ibuf);
   vk_buf_destroy(c, &m->inst);
   if (m->pipe) vkDestroyPipeline(c->dev, m->pipe, NULL);
   if (m->pl) vkDestroyPipelineLayout(c->dev, m->pl, NULL);
   if (m->dsl) vkDestroyDescriptorSetLayout(c->dev, m->dsl, NULL);
   if (m->dp) vkDestroyDescriptorPool(c->dev, m->dp, NULL);
   if (m->scene) rt_vk_static_scene_free(m->scene);
   if (m->qp) vkDestroyQueryPool(c->dev, m->qp, NULL);
   rt_vec_free(&m->gpu_us);
   rt_vec_free(&m->gap_us);
}

static int run_one(vk_ctx *c, const rt_params *p, const rt_run *r, int test, uint32_t depth)
{
   micro m;
   memset(&m, 0, sizeof(m));
   m.c = c;
   m.p = p;
   m.test = test;
   if (depth > VK_STREAM_MAX) depth = VK_STREAM_MAX;
   m.ts_ok = c->props.limits.timestampComputeAndGraphics && c->props.limits.timestampPeriod > 0;
   m.ts_ns = c->props.limits.timestampPeriod;
   uint64_t t0 = rt_now_ns();
   VkResult vr = test == RT_TEST_ASBUILD ? setup_asbuild(&m, depth)
               : test == RT_TEST_TRAV ? setup_trav(&m, depth) : setup_rays(&m, depth);
   if (vr == VK_SUCCESS) {
      VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                  .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = depth * 2};
      vr = vkCreateQueryPool(c->dev, &qi, NULL, &m.qp);
      if (vr == VK_SUCCESS) vkResetQueryPool(c->dev, m.qp, 0, depth * 2);
   }
   if (vr != VK_SUCCESS) {
      char e[96];
      snprintf(e, sizeof(e), "setup failed (Vulkan error %d)", vr);
      rt_result_fail(test, p, r->mode, depth, e);
      teardown(&m);
      return -1;
   }
   vk_stream st;
   vk_stream_ops ops = {&m, test == RT_TEST_ASBUILD ? record_build : test == RT_TEST_TRAV ? record_trav : record_rays,
                        complete};
   vr = vk_stream_init(&st, c, depth, ops);
   vk_stream_stats un = {0}, tm = {0};
   if (vr == VK_SUCCESS) vr = vk_stream_phase(&st, 0, r->warmup_ms, 0, 0, NULL);
   if (vr == VK_SUCCESS) vr = vk_stream_phase(&st, r->runs, r->run_ms, r->frames, 0, &un);
   if (vr == VK_SUCCESS && r->timestamps && m.ts_ok) vr = vk_stream_phase(&st, 3, r->run_ms, r->frames, 1, &tm);
   if (vr == VK_SUCCESS) vr = vk_stream_drain(&st);
   vk_stream_destroy(&st);
   if (vr == VK_SUCCESS && test == RT_TEST_ASBUILD && p->compact) query_compacted(&m);
   if (vr != VK_SUCCESS) {
      char e[96];
      snprintf(e, sizeof(e), "Vulkan error %d during the run", vr);
      rt_result_fail(test, p, r->mode, depth, e);
      teardown(&m);
      vk_stream_stats_free(&un);
      vk_stream_stats_free(&tm);
      return -1;
   }
   rt_json j;
   rt_result_begin(&j, test, p, r->mode, depth);
   rt_json_str(&j, "status", "ok");
   rt_result_windows(&j, r);
   int tlas = test == RT_TEST_ASBUILD && !strcmp(p->kind, "tlas");
   int rays = test == RT_TEST_RAYS || test == RT_TEST_TRAV;
   const char *rate_name = rays ? "mrays_s" : tlas ? "minst_s" : "mtri_s";
   rt_json_str(&j, "primary", rays ? "mrays_s" : (m.ts_ok && r->timestamps) ? rate_name : "us_per_submit");
   rt_json_obj(&j, "metrics");
   rt_vec ups = {0}, ups_t = {0}, rate = {0};
   double work = rays ? (double)p->rays : (double)m.prims;
   for (size_t i = 0; i < un.rate.n; i++) rt_vec_push(&ups, 1e6 / un.rate.v[i]);
   for (size_t i = 0; i < tm.rate.n; i++) rt_vec_push(&ups_t, 1e6 / tm.rate.v[i]);
   if (rays)
      for (size_t i = 0; i < un.rate.n; i++) rt_vec_push(&rate, un.rate.v[i] * work / 1e6);
   else
      for (size_t i = 0; i < m.gpu_us.n; i++) if (m.gpu_us.v[i] > 0) rt_vec_push(&rate, work / m.gpu_us.v[i]);
   rt_metric(&j, "us_per_submit", "us", "lower", ups.v, ups.n, 1);
   rt_metric(&j, rate_name, rays ? "Mray/s" : tlas ? "Minst/s" : "Mtri/s", "higher", rate.v, rate.n, rays);
   rt_metric(&j, "latency_us", "us", "lower", un.latency_us.v, un.latency_us.n, 0);
   rt_metric(&j, "record_cpu_us", "us", "lower", un.record_us.v, un.record_us.n, 0);
   rt_metric(&j, "submit_cpu_us", "us", "lower", un.submit_us.v, un.submit_us.n, 0);
   rt_metric(&j, "gpu_us", "us", "lower", m.gpu_us.v, m.gpu_us.n, 0);
   rt_metric(&j, "gpu_gap_us", "us", "lower", m.gap_us.v, m.gap_us.n, 0);
   rt_metric(&j, "us_per_submit_timed", "us", "lower", ups_t.v, ups_t.n, 1);
   if (rays && m.gpu_us.n) {
      rt_vec gr = {0};
      for (size_t i = 0; i < m.gpu_us.n; i++) if (m.gpu_us.v[i] > 0) rt_vec_push(&gr, work / m.gpu_us.v[i]);
      rt_metric(&j, "mrays_s_gpu", "Mray/s", "higher", gr.v, gr.n, 0);
      rt_vec_free(&gr);
   }
   rt_json_end(&j);
   rt_vec_free(&ups);
   rt_vec_free(&ups_t);
   rt_vec_free(&rate);
   rt_json_obj(&j, "counts");
   rt_json_int(&j, "submits", (int64_t)un.completions);
   rt_json_int(&j, "submits_timed", (int64_t)tm.completions);
   rt_json_end(&j);
   rt_json_obj(&j, "info");
   rt_json_str(&j, "api", "vulkan");
   rt_json_str(&j, "device", c->props.deviceName);
   rt_json_str(&j, "driver", c->driver.driverInfo);
   if (test == RT_TEST_ASBUILD) {
      rt_json_int(&j, tlas ? "instances" : "triangles", m.prims);
      if (!tlas) rt_json_int(&j, "vertices", m.nverts);
      rt_json_int(&j, "as_bytes", (int64_t)m.sizes.accelerationStructureSize);
      rt_json_int(&j, "build_scratch_bytes", (int64_t)m.sizes.buildScratchSize);
      rt_json_int(&j, "update_scratch_bytes", (int64_t)m.sizes.updateScratchSize);
      if (p->compact) rt_json_int(&j, "compacted_bytes", (int64_t)m.compacted);
   } else if (test == RT_TEST_TRAV) {
      rt_json_int(&j, "triangles", m.prims);
      rt_json_int(&j, "blas_bytes", (int64_t)m.blas_bytes);
   } else {
      const rt_scene *sc = rt_vk_static_scene_desc(m.scene);
      rt_json_int(&j, "scene_unique_tris", (int64_t)sc->unique_tris);
      rt_json_int(&j, "scene_instanced_tris", (int64_t)sc->instanced_tris);
      rt_json_int(&j, "scene_instances", sc->ninst);
      rt_json_num(&j, "scene_build_gpu_ms", rt_vk_static_scene_build_ms(m.scene));
   }
   if (r->shader_stats && m.pipe) {
      rt_json_obj(&j, "shader_stats");
      vk_pipeline_stats_json(c, m.pipe, &j, test == RT_TEST_TRAV ? "trav" : "rays");
      rt_json_end(&j);
   }
   rt_json_end(&j);
   rt_json_num(&j, "elapsed_s", (rt_now_ns() - t0) / 1e9);
   rt_result_print(&j);
   teardown(&m);
   vk_stream_stats_free(&un);
   vk_stream_stats_free(&tm);
   return 0;
}

static int run_test(vk_ctx *c, const rt_params *p, const rt_run *r, int test)
{
   if (strcmp(r->mode, "queue")) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(test, p, r->mode, r->depths[d], "rt micro-tests run only in mode=queue");
      return 0;
   }
   if (!c->has_as || !c->has_rq) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(test, p, r->mode, r->depths[d], "device lacks VK_KHR_acceleration_structure/ray_query");
      return 0;
   }
   if (test == RT_TEST_ASBUILD && strcmp(p->kind, "blas") && strcmp(p->kind, "refit") && strcmp(p->kind, "tlas")) {
      rt_result_fail(test, p, r->mode, r->depths[0], "kind must be blas, refit or tlas");
      return 0;
   }
   if (test == RT_TEST_RAYS && strcmp(p->kind, "coherent") && strcmp(p->kind, "incoherent")) {
      rt_result_fail(test, p, r->mode, r->depths[0], "kind must be coherent or incoherent");
      return 0;
   }
   if (test == RT_TEST_TRAV && strcmp(p->kind, "miss") && strcmp(p->kind, "single") && strcmp(p->kind, "comb") &&
       strcmp(p->kind, "grid") && strcmp(p->kind, "gap")) {
      rt_result_fail(test, p, r->mode, r->depths[0], "kind must be miss, single, comb, grid or gap");
      return 0;
   }
   for (uint32_t d = 0; d < r->ndepths; d++)
      if (run_one(c, p, r, test, r->depths[d])) break;
   return 0;
}

int rt_vk_asbuild(vk_ctx *c, const rt_params *p, const rt_run *r) { return run_test(c, p, r, RT_TEST_ASBUILD); }
int rt_vk_rays(vk_ctx *c, const rt_params *p, const rt_run *r) { return run_test(c, p, r, RT_TEST_RAYS); }
int rt_vk_trav(vk_ctx *c, const rt_params *p, const rt_run *r) { return run_test(c, p, r, RT_TEST_TRAV); }
