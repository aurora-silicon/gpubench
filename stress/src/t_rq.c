// SPDX-License-Identifier: GPL-2.0-only
/* Ray-query traversal stress: a BLAS of separate flat quads at moving heights, refit or rebuilt
 * every iteration, under a TLAS of 4096 moving instances rebuilt every iteration, and 128K rays
 * per dispatch whose hit instance, primitive and distance are all known on the CPU. */
#include "st.h"
#include "st_rq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RQ_Q 16u                 /* quads per BLAS side: 512 triangles */
#define RQ_G 64u                 /* instances per TLAS side: 4096 */
#define RQ_RAYS (128u * 1024u)
#define RQ_PITCH 1.5f            /* instance spacing; the BLAS spans [0,1]^2 */
#define RQ_ORIGIN_Z (-100.0f)

struct as {
   VkAccelerationStructureKHR as;
   struct st_buf storage, scratch;
   VkDeviceAddress addr;
   VkDeviceSize build_scratch, update_scratch;
};

static float
quad_z(uint64_t it, uint32_t u, uint32_t v)
{
   return 0.25f * sinf((float)it * 0.7f + (float)u * 0.3f + (float)v * 0.5f);
}

static float
inst_z(uint64_t it, uint32_t i, uint32_t j)
{
   return 2.0f * sinf((float)it * 0.3f + (float)i * 0.1f + (float)j * 0.2f);
}

static VkResult
as_create(struct st_dev *d, VkAccelerationStructureTypeKHR type, const VkAccelerationStructureBuildGeometryInfoKHR *bi,
          uint32_t prims, struct as *a)
{
   memset(a, 0, sizeof(*a));
   VkAccelerationStructureBuildSizesInfoKHR sz = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
   d->rqf->sizes(d->dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, bi, &prims, &sz);
   a->build_scratch = sz.buildScratchSize;
   a->update_scratch = sz.updateScratchSize;
   VkDeviceSize scratch = sz.buildScratchSize > sz.updateScratchSize ? sz.buildScratchSize : sz.updateScratchSize;
   VkResult vr;
   if ((vr = st_buf_create(d, sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                           false, &a->storage)) ||
       (vr = st_buf_create(d, scratch + 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &a->scratch)))
      return vr;
   VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                              .buffer = a->storage.buf, .size = sz.accelerationStructureSize,
                                              .type = type};
   if ((vr = d->rqf->create(d->dev, &ci, NULL, &a->as)))
      return vr;
   VkAccelerationStructureDeviceAddressInfoKHR ai = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .accelerationStructure = a->as};
   a->addr = d->rqf->addr(d->dev, &ai);
   return VK_SUCCESS;
}

static void
as_destroy(struct st_dev *d, struct as *a)
{
   if (a->as)
      d->rqf->destroy(d->dev, a->as, NULL);
   st_buf_destroy(d, &a->storage);
   st_buf_destroy(d, &a->scratch);
}

static VkDeviceAddress
aligned(VkDeviceAddress a)
{
   return (a + 255) & ~(VkDeviceAddress)255;
}

int
t_rq_stress(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (!caps.rq)
      return st_skip(r, "ray query is not supported");
   struct st_dev d;
   struct st_req req = {.nqueues = 1, .rq = true};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   const VkBufferUsageFlags in = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   const uint32_t ntri = RQ_Q * RQ_Q * 2, ninst = RQ_G * RQ_G;
   struct st_buf vtx, inst, rays, out;
   if ((vr = st_buf_create(&d, (VkDeviceSize)ntri * 3 * 12, in, true, &vtx)) ||
       (vr = st_buf_create(&d, (VkDeviceSize)ninst * sizeof(VkAccelerationStructureInstanceKHR), in, true, &inst)) ||
       (vr = st_buf_create(&d, (VkDeviceSize)RQ_RAYS * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &rays)) ||
       (vr = st_buf_create(&d, (VkDeviceSize)RQ_RAYS * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &out)))
      return st_fail(r, "buffers: %s", st_vkr(vr));

   VkAccelerationStructureGeometryKHR blas_geom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry.triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                             .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                             .vertexData.deviceAddress = vtx.addr,
                             .vertexStride = 12,
                             .maxVertex = ntri * 3 - 1,
                             .indexType = VK_INDEX_TYPE_NONE_KHR},
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
   };
   VkAccelerationStructureBuildGeometryInfoKHR blas_bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
               VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries = &blas_geom,
   };
   VkAccelerationStructureGeometryKHR tlas_geom = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                             .data.deviceAddress = inst.addr},
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
   };
   VkAccelerationStructureBuildGeometryInfoKHR tlas_bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries = &tlas_geom,
   };
   struct as blas, tlas;
   if ((vr = as_create(&d, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &blas_bi, ntri, &blas)) ||
       (vr = as_create(&d, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &tlas_bi, ninst, &tlas)))
      return st_fail(r, "acceleration structures: %s", st_vkr(vr));

   /* Ray-query pipeline: binding 0 the TLAS, 1 rays, 2 results. */
   VkDescriptorSetLayoutBinding binds[3] = {
      {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .bindingCount = 3, .pBindings = binds};
   VkDescriptorSetLayout dsl;
   VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
   VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                     .poolSizeCount = 2, .pPoolSizes = ps};
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
   VkPipelineLayout layout;
   VkPipeline pipe;
   VkShaderModule sm;
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = rq_comp_size,
                                   .pCode = rq_comp};
   if ((vr = vkCreateDescriptorSetLayout(d.dev, &dli, NULL, &dsl)) ||
       (vr = vkCreateDescriptorPool(d.dev, &dpi, NULL, &dpool)))
      return st_fail(r, "descriptors: %s", st_vkr(vr));
   VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
   if ((vr = vkAllocateDescriptorSets(d.dev, &sai, &set)) || (vr = vkCreatePipelineLayout(d.dev, &pli, NULL, &layout)) ||
       (vr = vkCreateShaderModule(d.dev, &smi, NULL, &sm)))
      return st_fail(r, "pipeline layout: %s", st_vkr(vr));
   VkComputePipelineCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = "main"},
      .layout = layout,
   };
   if ((vr = vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe)))
      return st_fail(r, "ray-query pipeline: %s", st_vkr(vr));
   vkDestroyShaderModule(d.dev, sm, NULL);
   VkWriteDescriptorSetAccelerationStructureKHR wa = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, .accelerationStructureCount = 1,
      .pAccelerationStructures = &tlas.as};
   VkDescriptorBufferInfo rbi = {rays.buf, 0, VK_WHOLE_SIZE}, obi = {out.buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet w[3] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &wa, .dstSet = set, .dstBinding = 0,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &rbi},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &obi},
   };
   vkUpdateDescriptorSets(d.dev, 3, w, 0, NULL);
   VkCommandBuffer cb;
   if ((vr = st_cb_alloc(&d, 0, &cb)))
      return st_fail(r, "command buffer: %s", st_vkr(vr));

   /* Expected result per ray: instance, primitive and t, or a miss. */
   struct expect {
      uint32_t inst, prim;
      float t;
      bool hit;
   } *ex = calloc(RQ_RAYS, sizeof(*ex));
   uint64_t refits = 0, rebuilds = 0, rays_checked = 0;
   for (uint64_t it = 0; st_time_left(r) && !r->errors; it++) {
      /* Geometry: separate flat quads at per-quad heights. */
      float *vp = vtx.map;
      for (uint32_t v = 0; v < RQ_Q; v++)
         for (uint32_t u = 0; u < RQ_Q; u++) {
            float x0 = (float)u / RQ_Q, x1 = (float)(u + 1) / RQ_Q, y0 = (float)v / RQ_Q, y1 = (float)(v + 1) / RQ_Q;
            float z = quad_z(it, u, v);
            float tri[18] = {x0, y0, z, x1, y0, z, x1, y1, z, x0, y0, z, x1, y1, z, x0, y1, z};
            memcpy(vp + (size_t)(v * RQ_Q + u) * 18, tri, sizeof(tri));
         }
      VkAccelerationStructureInstanceKHR *ip = inst.map;
      for (uint32_t j = 0; j < RQ_G; j++)
         for (uint32_t i = 0; i < RQ_G; i++) {
            VkAccelerationStructureInstanceKHR *x = &ip[j * RQ_G + i];
            memset(x, 0, sizeof(*x));
            x->transform.matrix[0][0] = x->transform.matrix[1][1] = x->transform.matrix[2][2] = 1.0f;
            x->transform.matrix[0][3] = (float)i * RQ_PITCH;
            x->transform.matrix[1][3] = (float)j * RQ_PITCH;
            x->transform.matrix[2][3] = inst_z(it, i, j);
            x->instanceCustomIndex = j * RQ_G + i;
            x->mask = 0xff;
            x->flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            x->accelerationStructureReference = blas.addr;
         }
      /* Rays: most aim inside a known triangle, some into the gaps between instances. */
      float *rp = rays.map;
      for (uint32_t k = 0; k < RQ_RAYS; k++) {
         uint32_t i = st_rand_range(r, 0, RQ_G - 1), j = st_rand_range(r, 0, RQ_G - 1);
         float x, y;
         struct expect *e = &ex[k];
         if (st_rand_range(r, 0, 7) == 0) {
            x = (float)i * RQ_PITCH + 1.1f + 0.3f * (st_rand(r) & 0xff) / 256.0f;
            y = (float)j * RQ_PITCH + 0.5f;
            e->hit = false;
         } else {
            uint32_t u = st_rand_range(r, 0, RQ_Q - 1), v = st_rand_range(r, 0, RQ_Q - 1);
            bool upper = st_rand_range(r, 0, 1); /* (0.25, 0.75) is in the second triangle */
            float fx = upper ? 0.25f : 0.75f, fy = upper ? 0.75f : 0.25f;
            x = (float)i * RQ_PITCH + ((float)u + fx) / RQ_Q;
            y = (float)j * RQ_PITCH + ((float)v + fy) / RQ_Q;
            e->hit = true;
            e->inst = j * RQ_G + i;
            e->prim = (v * RQ_Q + u) * 2 + (upper ? 1 : 0);
            e->t = inst_z(it, i, j) + quad_z(it, u, v) - RQ_ORIGIN_Z;
         }
         rp[k * 4 + 0] = x;
         rp[k * 4 + 1] = y;
         rp[k * 4 + 2] = RQ_ORIGIN_Z;
         rp[k * 4 + 3] = 1000.0f;
      }
      memset(out.map, 0xff, out.size);

      bool update = it > 0 && it % 4 != 0;
      blas_bi.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      blas_bi.srcAccelerationStructure = update ? blas.as : VK_NULL_HANDLE;
      blas_bi.dstAccelerationStructure = blas.as;
      blas_bi.scratchData.deviceAddress = aligned(blas.scratch.addr);
      tlas_bi.dstAccelerationStructure = tlas.as;
      tlas_bi.scratchData.deviceAddress = aligned(tlas.scratch.addr);
      VkAccelerationStructureBuildRangeInfoKHR br = {.primitiveCount = ntri}, tr = {.primitiveCount = ninst};
      const VkAccelerationStructureBuildRangeInfoKHR *brp = &br, *trp = &tr;

      st_cb_begin(cb);
      d.rqf->build(cb, 1, &blas_bi, &brp);
      st_barrier(cb, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
      d.rqf->build(cb, 1, &tlas_bi, &trp);
      st_barrier(cb, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);
      uint32_t n = RQ_RAYS;
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
      vkCmdDispatch(cb, RQ_RAYS / 64, 1, 1);
      /* The next iteration's builds overwrite what this dispatch read. */
      st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
      vkEndCommandBuffer(cb);
      uint64_t t0 = st_now();
      if ((vr = st_run_cb(&d, 0, cb, 20 * ST_NS)))
         return st_fail(r, "iteration %llu (%s BLAS, TLAS of %u): %s", (unsigned long long)it,
                        update ? "refit" : "rebuilt", ninst, st_vkr(vr));
      st_metric_max(r, "max_iteration_ms", (st_now() - t0) / 1e6);
      update ? refits++ : rebuilds++;
      const uint32_t *o = out.map;
      uint64_t bad = 0;
      for (uint32_t k = 0; k < RQ_RAYS; k++) {
         const uint32_t *q = o + k * 4;
         const struct expect *e = &ex[k];
         bool ok;
         if (!e->hit) {
            ok = q[3] == 0;
         } else {
            float t;
            memcpy(&t, &q[2], 4);
            ok = q[3] == 1 && q[0] == e->inst && q[1] == e->prim && fabsf(t - e->t) <= 1e-3f * (1.0f + fabsf(e->t));
         }
         if (!ok) {
            if (!bad) {
               float t;
               memcpy(&t, &q[2], 4);
               st_fail(r,
                       "iteration %llu (%s BLAS) ray %u: got hit=%u instance %u primitive %u t %.4f, expected "
                       "hit=%d instance %u primitive %u t %.4f",
                       (unsigned long long)it, update ? "refit" : "rebuilt", k, q[3], q[0], q[1], t, e->hit, e->inst,
                       e->prim, e->t);
            }
            bad++;
         }
      }
      if (bad)
         st_metric(r, "wrong_rays", (double)bad);
      rays_checked += RQ_RAYS;
      r->checks += RQ_RAYS;
      r->iterations++;
   }
   st_metric(r, "refits", (double)refits);
   st_metric(r, "rebuilds", (double)rebuilds);
   st_metric(r, "mrays_checked", rays_checked / 1e6);
   free(ex);
   if (!r->errors) {
      vkDestroyPipeline(d.dev, pipe, NULL);
      vkDestroyPipelineLayout(d.dev, layout, NULL);
      vkDestroyDescriptorPool(d.dev, dpool, NULL);
      vkDestroyDescriptorSetLayout(d.dev, dsl, NULL);
      as_destroy(&d, &tlas);
      as_destroy(&d, &blas);
      st_buf_destroy(&d, &vtx);
      st_buf_destroy(&d, &inst);
      st_buf_destroy(&d, &rays);
      st_buf_destroy(&d, &out);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}
