// SPDX-License-Identifier: GPL-2.0-only
/* rt.scene on Vulkan: G-buffer raster pass, ray-query reflections and shadows in compute, composite.
 * Per frame: [update] water deformation, [as] BLAS refit/rebuild + TLAS build, [raster] G-buffer,
 * [rt] reflections + shadows, [composite]. The Metal executor (rt/metal/main.m) does the same
 * passes with the same data. */
#include "../common/result.h"
#include "../common/scene.h"
#include "../common/util.h"
#include "rtvk.h"
#include "rtwin.h"
#include "vkstream.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { P_UPDATE, P_AS, P_RASTER, P_RT, P_COMPOSITE, NPASS };
static const char *pass_metric[NPASS] = {"gpu_update_us", "gpu_as_us", "gpu_raster_us", "gpu_rt_us",
                                         "gpu_composite_us"};
#define NTS (NPASS + 1)

typedef struct as_t {
   VkAccelerationStructureKHR as;
   vk_buf buf;
   VkDeviceAddress addr;
   VkAccelerationStructureBuildSizesInfoKHR sizes;
   VkAccelerationStructureGeometryKHR geom;
   VkAccelerationStructureBuildRangeInfoKHR range;
   VkBuildAccelerationStructureFlagsKHR flags;
   VkDeviceSize scratch_off;
} as_t;

typedef struct slot_t {
   vk_buf ubo, xf, inst;
   VkDescriptorSet gb, rt, comp;
} slot_t;

typedef struct scn {
   vk_ctx *c;
   const rt_params *p;
   const rt_run *r;
   rt_scene sc;
   vk_buf vbuf, ibuf, info, scratch, counters, readback;
   vk_img albedo, normal, depth, rtimg, out;
   VkSampler samp;
   as_t *blas;
   as_t tlas;
   VkDeviceSize init_scratch, frame_scratch;
   VkDescriptorSetLayout l_gb, l_rt, l_comp, l_def;
   VkPipelineLayout pl_gb, pl_rt, pl_comp, pl_def;
   VkPipeline p_gb_opaque, p_gb_sail, p_rt, p_comp, p_def;
   VkDescriptorPool dpool;
   VkDescriptorSet set_def;
   slot_t slots[VK_STREAM_MAX];
   uint32_t nslots;
   VkQueryPool qp;
   int ts_ok;
   double ts_ns;
   uint64_t ts_mask;
   uint32_t frame;
   int census;
   rt_vec pass_us[NPASS], frame_us;
   double init_ms[3], init_wall_ms;
   rt_vtrace vtrace;  /* ray-tracing pipeline variant (trace=pipeline) */
   rt_params pcopy;   /* rt.rays: the static variant of the parameters */
   int nort;          /* RTB_NO_RT=1 debug: no AS, stub RT pass (devices without ray queries) */
} scn;

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

static void tlas_instance(scn *s, uint32_t i, float t, VkAccelerationStructureInstanceKHR *o, rt_xform *xf)
{
   rt_scene_xform(&s->sc, i, t, xf);
   memset(o, 0, sizeof(*o));
   memcpy(o->transform.matrix, xf->r, sizeof(xf->r));
   o->instanceCustomIndex = i;
   o->mask = 0xff;
   o->instanceShaderBindingTableRecordOffset = 0;
   o->flags = 0;
   o->accelerationStructureReference = s->blas ? s->blas[s->sc.inst[i].mesh].addr : 0;
}

/* Writes the per-frame host data of slot for frame f. all: also the static instances. */
static void slot_update(scn *s, uint32_t slot, uint32_t f, int all)
{
   slot_t *sl = &s->slots[slot];
   rt_frame_ubo u;
   rt_scene_frame_ubo(&s->sc, s->p, f, 1, s->census, &u);
   memcpy(sl->ubo.map, &u, sizeof(u));
   float t = rt_frame_time(f);
   VkAccelerationStructureInstanceKHR *in = sl->inst.map;
   rt_xform *xf = sl->xf.map;
   for (uint32_t i = 0; i < s->sc.ninst; i++)
      if (all || (s->sc.inst[i].flags & RT_FLAG_MOVING)) tlas_instance(s, i, t, &in[i], &xf[i]);
}

static VkResult make_as(scn *s, as_t *a, VkAccelerationStructureTypeKHR type, uint32_t count)
{
   vk_ctx *c = s->c;
   VkAccelerationStructureBuildGeometryInfoKHR bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = type,
      .flags = a->flags,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .geometryCount = 1,
      .pGeometries = &a->geom};
   a->sizes = (VkAccelerationStructureBuildSizesInfoKHR){.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
   c->GetASBuildSizes(c->dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &count, &a->sizes);
   VKC(vk_buf_create(c, &a->buf, a->sizes.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
   VkAccelerationStructureCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                                              .buffer = a->buf.buf,
                                              .size = a->sizes.accelerationStructureSize,
                                              .type = type};
   VKC(c->CreateAS(c->dev, &ci, NULL, &a->as));
   VkAccelerationStructureDeviceAddressInfoKHR ai = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, .accelerationStructure = a->as};
   a->addr = c->GetASAddress(c->dev, &ai);
   a->range.primitiveCount = count;
   return VK_SUCCESS;
}

static VkBuildAccelerationStructureFlagsKHR as_flags(int mode)
{
   if (mode == RT_AS_REFIT) return VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
   if (mode == RT_AS_REBUILD) return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
   return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
}

static VkResult create_as(scn *s)
{
   vk_ctx *c = s->c;
   if (s->nort) return VK_SUCCESS;
   rt_scene *sc = &s->sc;
   VkDeviceSize al = c->asprops.minAccelerationStructureScratchOffsetAlignment;
   if (!al) al = 256;
   s->blas = calloc(sc->nmeshes, sizeof(as_t));
   VkDeviceSize init = 0, frame = 0;
   for (uint32_t m = 0; m < sc->nmeshes; m++) {
      rt_mesh *me = &sc->meshes[m];
      as_t *a = &s->blas[m];
      a->flags = as_flags(me->dynamic ? s->p->blas_mode : RT_AS_STATIC);
      a->geom = (VkAccelerationStructureGeometryKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
         .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
         .flags = me->alpha ? VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR : VK_GEOMETRY_OPAQUE_BIT_KHR,
         .geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData.deviceAddress = s->vbuf.addr + (VkDeviceSize)me->base_vertex * sizeof(rt_vertex),
            .vertexStride = sizeof(rt_vertex),
            .maxVertex = me->vertex_count - 1,
            .indexType = VK_INDEX_TYPE_UINT32,
            .indexData.deviceAddress = s->ibuf.addr + (VkDeviceSize)me->first_index * 4}};
      VKC(make_as(s, a, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, me->index_count / 3));
      init += align_up(a->sizes.buildScratchSize, al);
      if (me->dynamic) {
         a->scratch_off = frame;
         frame += align_up(s->p->blas_mode == RT_AS_REFIT ? a->sizes.updateScratchSize : a->sizes.buildScratchSize, al);
      }
   }
   s->tlas.flags = as_flags(s->p->tlas_mode);
   s->tlas.geom = (VkAccelerationStructureGeometryKHR){
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                             .arrayOfPointers = VK_FALSE,
                             .data.deviceAddress = s->slots[0].inst.addr}};
   VKC(make_as(s, &s->tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sc->ninst));
   s->tlas.scratch_off = frame;
   if (s->p->tlas_mode != RT_AS_STATIC)
      frame += align_up(s->p->tlas_mode == RT_AS_REFIT ? s->tlas.sizes.updateScratchSize : s->tlas.sizes.buildScratchSize, al);
   VkDeviceSize need = init;
   if (s->tlas.sizes.buildScratchSize > need) need = s->tlas.sizes.buildScratchSize;
   if (frame > need) need = frame;
   s->init_scratch = init;
   s->frame_scratch = frame;
   VKC(vk_buf_create(c, &s->scratch, need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0));
   return VK_SUCCESS;
}

static void build_blas(scn *s, VkCommandBuffer cb, int init)
{
   rt_scene *sc = &s->sc;
   VkDeviceSize al = s->c->asprops.minAccelerationStructureScratchOffsetAlignment;
   if (!al) al = 256;
   VkAccelerationStructureBuildGeometryInfoKHR bi[64 + RT_WATER_PATCHES];
   const VkAccelerationStructureBuildRangeInfoKHR *rp[64 + RT_WATER_PATCHES];
   uint32_t n = 0;
   VkDeviceSize off = 0;
   for (uint32_t m = 0; m < sc->nmeshes && n < 64 + RT_WATER_PATCHES; m++) {
      as_t *a = &s->blas[m];
      int upd = !init && s->p->blas_mode == RT_AS_REFIT;
      if (!init && !sc->meshes[m].dynamic) continue;
      bi[n] = (VkAccelerationStructureBuildGeometryInfoKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
         .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
         .flags = a->flags,
         .mode = upd ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
         .srcAccelerationStructure = upd ? a->as : VK_NULL_HANDLE,
         .dstAccelerationStructure = a->as,
         .geometryCount = 1,
         .pGeometries = &a->geom,
         .scratchData.deviceAddress = s->scratch.addr + (init ? off : a->scratch_off)};
      if (init) off += align_up(a->sizes.buildScratchSize, al);
      rp[n] = &a->range;
      n++;
   }
   /* More than 80 meshes: build the rest in further calls. */
   if (n) s->c->CmdBuildAS(cb, n, bi, rp);
   if (init && sc->nmeshes > n) {
      for (uint32_t m = n; m < sc->nmeshes; m++) {
         as_t *a = &s->blas[m];
         VkAccelerationStructureBuildGeometryInfoKHR b1 = bi[0];
         b1.flags = a->flags;
         b1.dstAccelerationStructure = a->as;
         b1.pGeometries = &a->geom;
         b1.scratchData.deviceAddress = s->scratch.addr + off;
         off += align_up(a->sizes.buildScratchSize, al);
         const VkAccelerationStructureBuildRangeInfoKHR *r1 = &a->range;
         s->c->CmdBuildAS(cb, 1, &b1, &r1);
      }
   }
}

static void build_tlas(scn *s, VkCommandBuffer cb, uint32_t slot, int init)
{
   int upd = !init && s->p->tlas_mode == RT_AS_REFIT;
   s->tlas.geom.geometry.instances.data.deviceAddress = s->slots[slot].inst.addr;
   VkAccelerationStructureBuildGeometryInfoKHR bi = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = s->tlas.flags,
      .mode = upd ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .srcAccelerationStructure = upd ? s->tlas.as : VK_NULL_HANDLE,
      .dstAccelerationStructure = s->tlas.as,
      .geometryCount = 1,
      .pGeometries = &s->tlas.geom,
      .scratchData.deviceAddress = s->scratch.addr + (init ? 0 : s->tlas.scratch_off)};
   const VkAccelerationStructureBuildRangeInfoKHR *rp = &s->tlas.range;
   s->c->CmdBuildAS(cb, 1, &bi, &rp);
}

static void deform(scn *s, VkCommandBuffer cb, uint32_t nslots, float t)
{
   rt_deform_pc pc = {.vbase = s->sc.water_vbase, .grid = s->sc.water_grid, .first_slot = 0, .nslots = nslots,
                      .size = s->sc.water_size, .time = t};
   uint32_t vpp = (s->sc.water_grid + 1) * (s->sc.water_grid + 1);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->p_def);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_def, 0, 1, &s->set_def, 0, NULL);
   vkCmdPushConstants(cb, s->pl_def, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, (vpp + 63) / 64, nslots, 1);
}

#define AS_RW (VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR)
#define ST_AS VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR

/* Deform all water at t = 0 and build every BLAS and the TLAS; records GPU and wall times. */
static VkResult init_build(scn *s)
{
   vk_ctx *c = s->c;
   VkQueryPool qp;
   VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                               .queryType = VK_QUERY_TYPE_TIMESTAMP,
                               .queryCount = 4};
   VKC(vkCreateQueryPool(c->dev, &qi, NULL, &qp));
   vkResetQueryPool(c->dev, qp, 0, 4);
   for (uint32_t i = 0; i < s->nslots; i++) slot_update(s, i, 0, 1);
   VkCommandBuffer cb;
   uint64_t t0 = rt_now_ns();
   VKC(vk_oneshot_begin(c, &cb));
   vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, qp, 0);
   /* RTB_SPLIT=1 submits each stage separately (debugging) */
   int split = getenv("RTB_SPLIT") != NULL;
#define SPLIT(name) if (split) { fprintf(stderr, "init: %s\n", name); VKC(vk_oneshot_end(c, cb)); VKC(vk_oneshot_begin(c, &cb)); }
   deform(s, cb, RT_WATER_PATCHES, 0.0f);
   vk_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, ST_AS | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
   vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 1);
   SPLIT("deform");
   if (!s->nort) build_blas(s, cb, 1);
   vk_barrier(cb, ST_AS, AS_RW, ST_AS, AS_RW);
   vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 2);
   SPLIT("blas");
   if (!s->nort) build_tlas(s, cb, 0, 1);
   vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, qp, 3);
   vk_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
   SPLIT("tlas");
#undef SPLIT
   VKC(vk_oneshot_end(c, cb));
   s->init_wall_ms = (rt_now_ns() - t0) / 1e6;
   uint64_t ts[4];
   if (vkGetQueryPoolResults(c->dev, qp, 0, 4, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      for (int i = 0; i < 3; i++) s->init_ms[i] = ((ts[i + 1] - ts[i]) & s->ts_mask) * s->ts_ns / 1e6;
   }
   vkDestroyQueryPool(c->dev, qp, NULL);
   return VK_SUCCESS;
}

static VkResult create_pipelines(scn *s)
{
   vk_ctx *c = s->c;
   VkShaderModule m_def, m_vs, m_fs, m_rt, m_comp;
#define SH(var, name) do { const vk_spv *x = vk_find_spv(name); VKC(vk_shader(c, x->code, x->size, &var)); } while (0)
   SH(m_def, "deform.comp");
   SH(m_vs, "gbuffer.vert");
   SH(m_fs, "gbuffer.frag");
   SH(m_rt, s->nort ? "rtstub.comp" : "rtreflect.comp");
   SH(m_comp, "composite.comp");
#undef SH
   VkPipelineCreateFlags pflags = c->exec_props_enabled ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;

   /* compute pipelines */
   struct { VkShaderModule m; VkPipelineLayout l; VkPipeline *out; const char *name; } cp[3] = {
      {m_def, s->pl_def, &s->p_def, "deform"}, {m_rt, s->pl_rt, &s->p_rt, "rt_reflect"},
      {m_comp, s->pl_comp, &s->p_comp, "composite"}};
   for (int i = 0; i < 3; i++) {
      vk_marker(cp[i].name);
      VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                        .flags = pflags,
                                        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                  .module = cp[i].m,
                                                  .pName = "main"},
                                        .layout = cp[i].l};
      VKC(vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &ci, NULL, cp[i].out));
   }

   /* G-buffer pipelines: opaque (back-face culled) and sail (two-sided, alpha-tested with --alpha 1) */
   VkVertexInputBindingDescription vb = {0, sizeof(rt_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription va[3] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                              {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                                              {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}};
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                              .vertexBindingDescriptionCount = 1,
                                              .pVertexBindingDescriptions = &vb,
                                              .vertexAttributeDescriptionCount = 3,
                                              .pVertexAttributeDescriptions = va};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                           .viewportCount = 1,
                                           .scissorCount = 1};
   VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                .polygonMode = VK_POLYGON_MODE_FILL,
                                                .cullMode = VK_CULL_MODE_BACK_BIT,
                                                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                              .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                               .depthTestEnable = VK_TRUE,
                                               .depthWriteEnable = VK_TRUE,
                                               .depthCompareOp = VK_COMPARE_OP_GREATER};
   VkPipelineColorBlendAttachmentState ba[2] = {{.colorWriteMask = 0xf}, {.colorWriteMask = 0xf}};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                             .attachmentCount = 2,
                                             .pAttachments = ba};
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                          .dynamicStateCount = 2,
                                          .pDynamicStates = dyn};
   VkFormat cfmt[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT};
   VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                       .colorAttachmentCount = 2,
                                       .pColorAttachmentFormats = cfmt,
                                       .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT};
   for (int sail = 0; sail < 2; sail++) {
      VkBool32 alpha = sail && s->p->alpha;
      VkSpecializationMapEntry me = {0, 0, sizeof(VkBool32)};
      VkSpecializationInfo si = {1, &me, sizeof(alpha), &alpha};
      VkPipelineShaderStageCreateInfo st[2] = {
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = m_vs, .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = m_fs, .pName = "main", .pSpecializationInfo = &si}};
      rs.cullMode = sail ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      VkGraphicsPipelineCreateInfo gi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                         .pNext = &ri,
                                         .flags = pflags,
                                         .stageCount = 2,
                                         .pStages = st,
                                         .pVertexInputState = &vi,
                                         .pInputAssemblyState = &ia,
                                         .pViewportState = &vp,
                                         .pRasterizationState = &rs,
                                         .pMultisampleState = &ms,
                                         .pDepthStencilState = &ds,
                                         .pColorBlendState = &cb,
                                         .pDynamicState = &dy,
                                         .layout = s->pl_gb};
      vk_marker(sail ? (alpha ? "gbuffer_alpha" : "gbuffer_sail_opaque") : "gbuffer_opaque");
      VKC(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1, &gi, NULL, sail ? &s->p_gb_sail : &s->p_gb_opaque));
   }
   vkDestroyShaderModule(c->dev, m_def, NULL);
   vkDestroyShaderModule(c->dev, m_vs, NULL);
   vkDestroyShaderModule(c->dev, m_fs, NULL);
   vkDestroyShaderModule(c->dev, m_rt, NULL);
   vkDestroyShaderModule(c->dev, m_comp, NULL);
   vk_marker("internal");  /* later binaries: driver-internal (AS build) pipelines */
   return VK_SUCCESS;
}

static VkResult make_layout(vk_ctx *c, const VkDescriptorSetLayoutBinding *b, uint32_t n, VkDescriptorSetLayout *l,
                            uint32_t pc_size, VkShaderStageFlags pc_stages, VkPipelineLayout *pl)
{
   VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .bindingCount = n,
                                         .pBindings = b};
   VKC(vkCreateDescriptorSetLayout(c->dev, &ci, NULL, l));
   VkPushConstantRange pr = {pc_stages, 0, pc_size};
   VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                    .setLayoutCount = 1,
                                    .pSetLayouts = l,
                                    .pushConstantRangeCount = pc_size ? 1 : 0,
                                    .pPushConstantRanges = &pr};
   return vkCreatePipelineLayout(c->dev, &pi, NULL, pl);
}

static VkResult create_layouts(scn *s)
{
   vk_ctx *c = s->c;
   const VkShaderStageFlags CS = VK_SHADER_STAGE_COMPUTE_BIT;
   VkShaderStageFlags RTS = CS;
   if (c->has_rtp)
      RTS |= VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
             VK_SHADER_STAGE_MISS_BIT_KHR;
   VkDescriptorSetLayoutBinding def[1] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, CS, NULL}};
   VKC(make_layout(c, def, 1, &s->l_def, sizeof(rt_deform_pc), CS, &s->pl_def));
   VkDescriptorSetLayoutBinding gb[3] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
   VKC(make_layout(c, gb, 3, &s->l_gb, 0, 0, &s->pl_gb));
   VkDescriptorSetLayoutBinding rt[11] = {
      {0, s->nort ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, RTS, NULL},
      {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, RTS, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, RTS, NULL},
      {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, RTS, NULL},
      {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, RTS, NULL},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, RTS, NULL},
      {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, RTS, NULL},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, RTS, NULL},
      {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, RTS, NULL},
      {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, RTS, NULL},
      {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, RTS, NULL}};
   VKC(make_layout(c, rt, 11, &s->l_rt, 0, 0, &s->pl_rt));
   VkDescriptorSetLayoutBinding cp[6] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, CS, NULL},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL},
      {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL},
      {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL},
      {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, CS, NULL},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, CS, NULL}};
   VKC(make_layout(c, cp, 6, &s->l_comp, 0, 0, &s->pl_comp));
   return VK_SUCCESS;
}

static VkResult create_descriptors(scn *s)
{
   vk_ctx *c = s->c;
   uint32_t n = s->nslots;
   VkDescriptorPoolSize ps[5] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 * n},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * n + 1},
                                 {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7 * n},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * n},
                                 {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, n}};
   VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = 3 * n + 1,
                                     .poolSizeCount = 5,
                                     .pPoolSizes = ps};
   VKC(vkCreateDescriptorPool(c->dev, &pci, NULL, &s->dpool));
   VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                     .descriptorPool = s->dpool,
                                     .descriptorSetCount = 1};
   ai.pSetLayouts = &s->l_def;
   VKC(vkAllocateDescriptorSets(c->dev, &ai, &s->set_def));
   VkDescriptorBufferInfo vbi = {s->vbuf.buf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s->set_def, .dstBinding = 0,
                             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &vbi};
   vkUpdateDescriptorSets(c->dev, 1, &w, 0, NULL);
   for (uint32_t i = 0; i < n; i++) {
      slot_t *sl = &s->slots[i];
      ai.pSetLayouts = &s->l_gb;
      VKC(vkAllocateDescriptorSets(c->dev, &ai, &sl->gb));
      ai.pSetLayouts = &s->l_rt;
      VKC(vkAllocateDescriptorSets(c->dev, &ai, &sl->rt));
      ai.pSetLayouts = &s->l_comp;
      VKC(vkAllocateDescriptorSets(c->dev, &ai, &sl->comp));
      VkDescriptorBufferInfo ubo = {sl->ubo.buf, 0, VK_WHOLE_SIZE}, xf = {sl->xf.buf, 0, VK_WHOLE_SIZE},
                             info = {s->info.buf, 0, VK_WHOLE_SIZE}, ib = {s->ibuf.buf, 0, VK_WHOLE_SIZE},
                             cnt = {s->counters.buf, 0, VK_WHOLE_SIZE};
      VkDescriptorImageInfo alb = {s->samp, s->albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                            nrm = {s->samp, s->normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                            dep = {s->samp, s->depth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                            rts = {s->samp, s->rtimg.view, VK_IMAGE_LAYOUT_GENERAL},
                            rtw = {VK_NULL_HANDLE, s->rtimg.view, VK_IMAGE_LAYOUT_GENERAL},
                            outw = {VK_NULL_HANDLE, s->out.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSetAccelerationStructureKHR wa = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
         .accelerationStructureCount = 1,
         .pAccelerationStructures = &s->tlas.as};
#define WB(set, b, type, bi) {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = b, \
                              .descriptorCount = 1, .descriptorType = type, .pBufferInfo = bi}
#define WI(set, b, type, ii) {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = b, \
                              .descriptorCount = 1, .descriptorType = type, .pImageInfo = ii}
      VkWriteDescriptorSet ws[] = {
         WB(sl->gb, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &ubo),
         WB(sl->gb, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &xf),
         WB(sl->gb, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &info),
         s->nort ? (VkWriteDescriptorSet)WB(sl->rt, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &cnt)
                 : (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &wa, .dstSet = sl->rt,
                                          .dstBinding = 0, .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
         WB(sl->rt, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &ubo),
         WI(sl->rt, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &alb),
         WI(sl->rt, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &nrm),
         WI(sl->rt, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dep),
         WB(sl->rt, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vbi),
         WB(sl->rt, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &ib),
         WB(sl->rt, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &info),
         WB(sl->rt, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &xf),
         WI(sl->rt, 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rtw),
         WB(sl->rt, 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &cnt),
         WB(sl->comp, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &ubo),
         WI(sl->comp, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &alb),
         WI(sl->comp, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &nrm),
         WI(sl->comp, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dep),
         WI(sl->comp, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rts),
         WI(sl->comp, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outw)};
#undef WB
#undef WI
      vkUpdateDescriptorSets(c->dev, sizeof(ws) / sizeof(ws[0]), ws, 0, NULL);
   }
   return VK_SUCCESS;
}

static VkResult create_resources(scn *s, uint32_t nslots)
{
   vk_ctx *c = s->c;
   rt_scene *sc = &s->sc;
   const VkBufferUsageFlags geo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
   VKC(vk_buf_create(c, &s->vbuf, (VkDeviceSize)sc->nverts * sizeof(rt_vertex), geo | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 1));
   memcpy(s->vbuf.map, sc->verts, (size_t)sc->nverts * sizeof(rt_vertex));
   VKC(vk_buf_create(c, &s->ibuf, (VkDeviceSize)sc->nindices * 4, geo | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 1));
   memcpy(s->ibuf.map, sc->indices, (size_t)sc->nindices * 4);
   VKC(vk_buf_create(c, &s->info, (VkDeviceSize)sc->ninst * sizeof(rt_instinfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1));
   memcpy(s->info.map, sc->info, (size_t)sc->ninst * sizeof(rt_instinfo));
   VKC(vk_buf_create(c, &s->counters, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1));
   memset(s->counters.map, 0, 16);
   uint32_t W = s->p->width, H = s->p->height;
   VKC(vk_buf_create(c, &s->readback, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1));
   s->nslots = nslots;
   for (uint32_t i = 0; i < nslots; i++) {
      slot_t *sl = &s->slots[i];
      VKC(vk_buf_create(c, &sl->ubo, sizeof(rt_frame_ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 1));
      VKC(vk_buf_create(c, &sl->xf, (VkDeviceSize)sc->ninst * sizeof(rt_xform), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1));
      VKC(vk_buf_create(c, &sl->inst, (VkDeviceSize)sc->ninst * sizeof(VkAccelerationStructureInstanceKHR),
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, 1));
   }
   const VkImageUsageFlags att = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   VKC(vk_img_create(c, &s->albedo, W, H, VK_FORMAT_R8G8B8A8_UNORM, att, VK_IMAGE_ASPECT_COLOR_BIT));
   VKC(vk_img_create(c, &s->normal, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, att, VK_IMAGE_ASPECT_COLOR_BIT));
   VKC(vk_img_create(c, &s->depth, W, H, VK_FORMAT_D32_SFLOAT,
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
   VKC(vk_img_create(c, &s->rtimg, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
   VKC(vk_img_create(c, &s->out, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
   VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                             .magFilter = VK_FILTER_NEAREST,
                             .minFilter = VK_FILTER_NEAREST,
                             .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                             .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
   VKC(vkCreateSampler(c->dev, &si, NULL, &s->samp));
   /* rt and out images live in GENERAL */
   VkCommandBuffer cb;
   VKC(vk_oneshot_begin(c, &cb));
   vk_image_barrier(cb, s->rtimg.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
   vk_image_barrier(cb, s->out.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
   VKC(vk_oneshot_end(c, cb));
   VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                               .queryType = VK_QUERY_TYPE_TIMESTAMP,
                               .queryCount = nslots * NTS};
   VKC(vkCreateQueryPool(c->dev, &qi, NULL, &s->qp));
   vkResetQueryPool(c->dev, s->qp, 0, nslots * NTS);
   return VK_SUCCESS;
}

static void attachment_barriers(VkCommandBuffer cb, scn *s, int to_read)
{
   VkImageMemoryBarrier2 b[3];
   VkImage imgs[3] = {s->albedo.img, s->normal.img, s->depth.img};
   for (int i = 0; i < 3; i++) {
      int d = i == 2;
      VkImageLayout att = d ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      VkPipelineStageFlags2 ast = d ? (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
                                    : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkAccessFlags2 aac = d ? (VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
                             : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      b[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                     .image = imgs[i],
                                     .subresourceRange = {d ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      if (to_read) {
         b[i].srcStageMask = ast; b[i].srcAccessMask = aac;
         b[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
         b[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
         b[i].oldLayout = att; b[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      } else {
         /* previous frame's readers are covered by the frame-start barrier; contents are discarded */
         b[i].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; b[i].srcAccessMask = 0;
         b[i].dstStageMask = ast; b[i].dstAccessMask = aac;
         b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b[i].newLayout = att;
      }
   }
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                          .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                          .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .memoryBarrierCount = to_read ? 0 : 1,
                          .pMemoryBarriers = &mb,
                          .imageMemoryBarrierCount = 3,
                          .pImageMemoryBarriers = b};
   vkCmdPipelineBarrier2(cb, &di);
}

static VkResult record_frame(void *user, uint32_t slot, uint64_t k, VkCommandBuffer cb, int timed)
{
   (void)k;
   scn *s = user;
   const rt_params *p = s->p;
   uint32_t f = s->frame++;
   float t = rt_frame_time(f);
   slot_update(s, slot, f, 0);
   timed = timed && s->ts_ok;
   int ps = timed && !s->r->stamps_frame;  /* per-pass stamps */
   uint32_t q0 = slot * NTS;
   int dyn = s->sc.dyn_patches > 0, tdyn = p->tlas_mode != RT_AS_STATIC && !s->nort;

   attachment_barriers(cb, s, 0);
   if (timed) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s->qp, q0 + 0);
   if (dyn) {
      deform(s, cb, s->sc.dyn_patches, t);
      vk_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                 ST_AS | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
   }
   if (ps) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, q0 + 1);
   if (dyn && !s->nort) {
      build_blas(s, cb, 0);
      vk_barrier(cb, ST_AS, AS_RW, ST_AS, AS_RW);
   }
   if (tdyn) build_tlas(s, cb, slot, 0);
   if (dyn || tdyn)
      vk_barrier(cb, ST_AS, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
   if (ps) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, q0 + 2);

   /* G-buffer */
   VkRenderingAttachmentInfo ca[2] = {
      {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = s->albedo.view,
       .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_STORE},
      {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = s->normal.view,
       .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_STORE}};
   VkRenderingAttachmentInfo da = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = s->depth.view,
                                   .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                   .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                   .clearValue.depthStencil = {0.0f, 0}};
   VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                         .renderArea = {{0, 0}, {p->width, p->height}},
                         .layerCount = 1,
                         .colorAttachmentCount = 2,
                         .pColorAttachments = ca,
                         .pDepthAttachment = &da};
   vkCmdBeginRendering(cb, &ri);
   VkViewport vp = {0, 0, (float)p->width, (float)p->height, 0, 1};
   VkRect2D sc = {{0, 0}, {p->width, p->height}};
   vkCmdSetViewport(cb, 0, 1, &vp);
   vkCmdSetScissor(cb, 0, 1, &sc);
   VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cb, 0, 1, &s->vbuf.buf, &zero);
   vkCmdBindIndexBuffer(cb, s->ibuf.buf, 0, VK_INDEX_TYPE_UINT32);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s->pl_gb, 0, 1, &s->slots[slot].gb, 0, NULL);
   VkPipeline bound = VK_NULL_HANDLE;
   for (uint32_t d = 0; d < s->sc.ndraws; d++) {
      rt_draw *dr = &s->sc.draws[d];
      rt_mesh *m = &s->sc.meshes[dr->mesh];
      VkPipeline want = dr->twosided ? s->p_gb_sail : s->p_gb_opaque;
      if (want != bound) {
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, want);
         bound = want;
      }
      vkCmdDrawIndexed(cb, m->index_count, dr->instance_count, m->first_index, (int32_t)m->base_vertex, dr->first_instance);
   }
   vkCmdEndRendering(cb);
   attachment_barriers(cb, s, 1);
   if (ps) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, q0 + 3);

   /* reflections and shadows */
   if (p->trace_mode == RT_TRACE_PIPELINE) {
      rt_vtrace_record(&s->vtrace, cb, s->slots[slot].rt, p->width, p->height);
   } else {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->p_rt);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_rt, 0, 1, &s->slots[slot].rt, 0, NULL);
      vkCmdDispatch(cb, (p->width + 7) / 8, (p->height + 7) / 8, 1);
   }
   vk_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   if (ps) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, q0 + 4);

   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->p_comp);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl_comp, 0, 1, &s->slots[slot].comp, 0, NULL);
   vkCmdDispatch(cb, (p->width + 7) / 8, (p->height + 7) / 8, 1);
   if (timed) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, s->qp, q0 + 5);

   if (s->census) {
      vk_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy rg = {.bufferOffset = 0,
                              .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageExtent = {p->width, p->height, 1}};
      vkCmdCopyImageToBuffer(cb, s->out.img, VK_IMAGE_LAYOUT_GENERAL, s->readback.buf, 1, &rg);
      vk_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                 VK_ACCESS_2_HOST_READ_BIT);
   }
   return VK_SUCCESS;
}

static void complete_frame(void *user, uint32_t slot, uint64_t k, int timed)
{
   (void)k;
   scn *s = user;
   if (!timed || !s->ts_ok) return;
   uint64_t ts[NTS];
   if (s->r && s->r->stamps_frame) {
      uint64_t a, b;
      if (vkGetQueryPoolResults(s->c->dev, s->qp, slot * NTS, 1, 8, &a, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
          vkGetQueryPoolResults(s->c->dev, s->qp, slot * NTS + NPASS, 1, 8, &b, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
         rt_vec_push(&s->frame_us, ((b - a) & s->ts_mask) * s->ts_ns / 1e3);
   } else if (vkGetQueryPoolResults(s->c->dev, s->qp, slot * NTS, NTS, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      for (int i = 0; i < NPASS; i++) rt_vec_push(&s->pass_us[i], ((ts[i + 1] - ts[i]) & s->ts_mask) * s->ts_ns / 1e3);
      rt_vec_push(&s->frame_us, ((ts[NPASS] - ts[0]) & s->ts_mask) * s->ts_ns / 1e3);
   }
   vkResetQueryPool(s->c->dev, s->qp, slot * NTS, NTS);
}

static void destroy(scn *s)
{
   vk_ctx *c = s->c;
   if (!c || !c->dev) return;
   vkDeviceWaitIdle(c->dev);
   rt_vtrace_destroy(&s->vtrace, c);
   VkPipeline ps[] = {s->p_gb_opaque, s->p_gb_sail, s->p_rt, s->p_comp, s->p_def};
   for (unsigned i = 0; i < sizeof(ps) / sizeof(ps[0]); i++) if (ps[i]) vkDestroyPipeline(c->dev, ps[i], NULL);
   VkPipelineLayout pls[] = {s->pl_gb, s->pl_rt, s->pl_comp, s->pl_def};
   for (unsigned i = 0; i < 4; i++) if (pls[i]) vkDestroyPipelineLayout(c->dev, pls[i], NULL);
   VkDescriptorSetLayout ls[] = {s->l_gb, s->l_rt, s->l_comp, s->l_def};
   for (unsigned i = 0; i < 4; i++) if (ls[i]) vkDestroyDescriptorSetLayout(c->dev, ls[i], NULL);
   if (s->dpool) vkDestroyDescriptorPool(c->dev, s->dpool, NULL);
   if (s->qp) vkDestroyQueryPool(c->dev, s->qp, NULL);
   if (s->samp) vkDestroySampler(c->dev, s->samp, NULL);
   if (s->blas) {
      for (uint32_t m = 0; m < s->sc.nmeshes; m++) {
         if (s->blas[m].as) c->DestroyAS(c->dev, s->blas[m].as, NULL);
         vk_buf_destroy(c, &s->blas[m].buf);
      }
      free(s->blas);
   }
   if (s->tlas.as && c->DestroyAS) c->DestroyAS(c->dev, s->tlas.as, NULL);
   vk_buf_destroy(c, &s->tlas.buf);
   vk_buf *bs[] = {&s->vbuf, &s->ibuf, &s->info, &s->scratch, &s->counters, &s->readback};
   for (unsigned i = 0; i < 6; i++) vk_buf_destroy(c, bs[i]);
   for (uint32_t i = 0; i < s->nslots; i++) {
      vk_buf_destroy(c, &s->slots[i].ubo);
      vk_buf_destroy(c, &s->slots[i].xf);
      vk_buf_destroy(c, &s->slots[i].inst);
   }
   vk_img *is[] = {&s->albedo, &s->normal, &s->depth, &s->rtimg, &s->out};
   for (unsigned i = 0; i < 5; i++) vk_img_destroy(c, is[i]);
   rt_scene_free(&s->sc);
}

static void emit_us(rt_json *j, const char *name, rt_vec *v)
{
   rt_metric(j, name, "us", "lower", v->v, v->n, 0);
}

static int cmp_desc(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? 1 : x > y ? -1 : 0;
}

/* "1% low": the mean of the slowest 1% of frame intervals (at least one), as FPS. */
static double low1(const double *iv, size_t n)
{
   if (!n) return 0;
   double *s = malloc(n * sizeof(double));
   memcpy(s, iv, n * sizeof(double));
   qsort(s, n, sizeof(double), cmp_desc);
   size_t m = n / 100;
   if (m < 1) m = 1;
   double sum = 0;
   for (size_t i = 0; i < m; i++) sum += s[i];
   free(s);
   return 1e6 / (sum / (double)m);
}

static int run_depth(scn *s, uint32_t depth)
{
   const rt_run *r = s->r;
   const rt_params *p = s->p;
   vk_ctx *c = s->c;
   char err[256];
   uint64_t t_start = rt_now_ns();
   for (int i = 0; i < NPASS; i++) rt_vec_free(&s->pass_us[i]);
   rt_vec_free(&s->frame_us);
   s->frame = 0;
   s->census = 0;
   char therm0[2048], therm1[2048];
   rt_thermal_snapshot(therm0, sizeof(therm0));
   if (init_build(s) != VK_SUCCESS) {
      rt_result_fail(RT_TEST_SCENE, p, r->mode, depth, "initial acceleration-structure build failed");
      return -1;
   }
   vk_stream st;
   vk_stream_ops ops = {s, record_frame, complete_frame};
   if (vk_stream_init(&st, c, depth, ops) != VK_SUCCESS) {
      rt_result_fail(RT_TEST_SCENE, p, r->mode, depth, "stream init failed");
      return -1;
   }
   vk_stream_stats un = {0}, tm = {0};
   rt_vec win_low = {0};
   VkResult vr = vk_stream_phase(&st, 0, r->warmup_ms, 0, 0, NULL);
   size_t mark = 0;
   for (uint32_t w = 0; vr == VK_SUCCESS && w < r->runs; w++) {
      vr = vk_stream_phase(&st, 1, r->run_ms, r->frames, 0, &un);
      rt_vec_push(&win_low, low1(un.interval_us.v + mark, un.interval_us.n - mark));
      mark = un.interval_us.n;
   }
   if (vr == VK_SUCCESS && r->timestamps && s->ts_ok) vr = vk_stream_phase(&st, 3, r->run_ms, r->frames, 1, &tm);
   if (vr == VK_SUCCESS) vr = vk_stream_drain(&st);
   /* census frame: ray counts and the image */
   uint32_t cnt[4] = {0};
   double refl_frac = -1;
   uint32_t crc = 0;
   if (vr == VK_SUCCESS) {
      memset(s->counters.map, 0, 16);
      s->census = 1;
      s->frame = 0; /* the census always renders frame 0: same image and rays on every device */
      vr = vk_stream_submit(&st, 0);
      if (vr == VK_SUCCESS) vr = vk_stream_drain(&st);
      s->census = 0;
      if (vr == VK_SUCCESS) {
         memcpy(cnt, s->counters.map, 16);
         refl_frac = rt_alpha_fraction(s->readback.map, p->width, p->height, p->width * 4);
         crc = rt_crc32(0, s->readback.map, (size_t)p->width * p->height * 4);
         if (r->png[0]) {
            char path[600];
            if (r->ndepths > 1) snprintf(path, sizeof(path), "%.*s-d%u.png", (int)strcspn(r->png, "."), r->png, depth);
            else snprintf(path, sizeof(path), "%s", r->png);
            if (rt_write_png(path, s->readback.map, p->width, p->height, p->width * 4)) fprintf(stderr, "cannot write %s\n", path);
            else fprintf(stderr, "wrote %s\n", path);
         }
      }
   }
   vk_stream_destroy(&st);
   rt_thermal_snapshot(therm1, sizeof(therm1));
   if (vr != VK_SUCCESS) {
      snprintf(err, sizeof(err), "Vulkan error %d during the run", vr);
      rt_result_fail(RT_TEST_SCENE, p, r->mode, depth, err);
      vk_stream_stats_free(&un);
      vk_stream_stats_free(&tm);
      rt_vec_free(&win_low);
      return -1;
   }

   rt_json j;
   rt_result_begin(&j, RT_TEST_SCENE, p, r->mode, depth);
   rt_json_str(&j, "status", "ok");
   rt_result_windows(&j, r);
   rt_json_str(&j, "primary", "fps");
   rt_json_obj(&j, "metrics");
   rt_metric(&j, "fps", "fps", "higher", un.rate.v, un.rate.n, 1);
   rt_metric(&j, "fps_1pct_low", "fps", "higher", win_low.v, win_low.n, 1);
   rt_vec fm = {0};
   for (size_t i = 0; i < un.interval_us.n; i++) rt_vec_push(&fm, un.interval_us.v[i] / 1e3);
   rt_metric(&j, "frame_ms", "ms", "lower", fm.v, fm.n, 0);
   rt_vec_free(&fm);
   emit_us(&j, "latency_us", &un.latency_us);
   emit_us(&j, "record_cpu_us", &un.record_us);
   emit_us(&j, "submit_cpu_us", &un.submit_us);
   rt_metric(&j, "fps_timed", "fps", "higher", tm.rate.v, tm.rate.n, 1);
   emit_us(&j, "gpu_us", &s->frame_us);
   for (int i = 0; i < NPASS; i++) emit_us(&j, pass_metric[i], &s->pass_us[i]);
   uint64_t rays = (uint64_t)cnt[0] + cnt[1] + cnt[2];
   if (rays && s->pass_us[P_RT].n) {
      rt_vec mr = {0};
      for (size_t i = 0; i < s->pass_us[P_RT].n; i++)
         if (s->pass_us[P_RT].v[i] > 0) rt_vec_push(&mr, (double)rays / s->pass_us[P_RT].v[i]);
      rt_metric(&j, "mrays_s", "Mray/s", "higher", mr.v, mr.n, 0);
      rt_vec_free(&mr);
   }
   rt_json_end(&j);
   rt_json_obj(&j, "counts");
   rt_json_int(&j, "frames", (int64_t)un.completions);
   rt_json_int(&j, "frames_timed", (int64_t)tm.completions);
   rt_json_end(&j);
   rt_json_obj(&j, "info");
   rt_json_str(&j, "api", "vulkan");
   rt_json_str(&j, "device", c->props.deviceName);
   rt_json_str(&j, "driver", c->driver.driverInfo);
   rt_json_obj(&j, "scene");
   rt_json_int(&j, "unique_tris", (int64_t)s->sc.unique_tris);
   rt_json_int(&j, "instanced_tris", (int64_t)s->sc.instanced_tris);
   rt_json_int(&j, "vertices", s->sc.nverts);
   rt_json_int(&j, "blas", s->sc.nmeshes);
   rt_json_int(&j, "tlas_instances", s->sc.ninst);
   rt_json_int(&j, "moving_instances", s->sc.nmoving);
   rt_json_int(&j, "dyn_patches", s->sc.dyn_patches);
   rt_json_int(&j, "draws", s->sc.ndraws);
   rt_json_end(&j);
   rt_json_obj(&j, "as");
   uint64_t blas_bytes = 0;
   for (uint32_t m = 0; s->blas && m < s->sc.nmeshes; m++) {
      blas_bytes += s->blas[m].sizes.accelerationStructureSize;
   }
   rt_json_int(&j, "blas_bytes", (int64_t)blas_bytes);
   rt_json_int(&j, "tlas_bytes", (int64_t)s->tlas.sizes.accelerationStructureSize);
   rt_json_int(&j, "init_scratch_bytes", (int64_t)s->init_scratch);
   rt_json_int(&j, "frame_scratch_bytes", (int64_t)s->frame_scratch);
   rt_json_int(&j, "tlas_build_scratch_bytes", (int64_t)s->tlas.sizes.buildScratchSize);
   rt_json_num(&j, "init_deform_gpu_ms", s->init_ms[0]);
   rt_json_num(&j, "init_blas_gpu_ms", s->init_ms[1]);
   rt_json_num(&j, "init_tlas_gpu_ms", s->init_ms[2]);
   rt_json_num(&j, "init_wall_ms", s->init_wall_ms);
   rt_json_str(&j, "blas_flags", rt_as_mode_name(p->blas_mode));
   rt_json_str(&j, "tlas_flags", rt_as_mode_name(p->tlas_mode));
   rt_json_end(&j);
   rt_json_obj(&j, "rays");
   rt_json_int(&j, "primary_shadow", cnt[0]);
   rt_json_int(&j, "reflection", cnt[1]);
   rt_json_int(&j, "secondary_shadow", cnt[2]);
   rt_json_int(&j, "per_frame", (int64_t)rays);
   rt_json_num(&j, "reflective_pixel_frac", refl_frac);
   rt_json_end(&j);
   char crcs[16];
   snprintf(crcs, sizeof(crcs), "%08x", crc);
   rt_json_str(&j, "image_crc32", crcs);
   rt_json_str(&j, "timestamps", s->ts_ok ? (r->timestamps ? "on" : "off") : "unsupported");
   rt_json_end(&j);
   if (r->shader_stats && c->exec_props_enabled) {
      rt_json_obj(&j, "shader_stats");
      vk_pipeline_stats_json(c, s->p_def, &j, "deform");
      vk_pipeline_stats_json(c, s->p_gb_opaque, &j, "gbuffer_opaque");
      vk_pipeline_stats_json(c, s->p_gb_sail, &j, "gbuffer_sail");
      if (p->trace_mode == RT_TRACE_PIPELINE) vk_pipeline_stats_json(c, s->vtrace.pipe, &j, "rt_pipeline");
      else vk_pipeline_stats_json(c, s->p_rt, &j, "rt_reflect");
      vk_pipeline_stats_json(c, s->p_comp, &j, "composite");
      rt_json_end(&j);
   }
   rt_json_raw(&j, "thermal_before", therm0);
   rt_json_raw(&j, "thermal_after", therm1);
   rt_json_num(&j, "elapsed_s", (rt_now_ns() - t_start) / 1e9);
   rt_result_print(&j);
   vk_stream_stats_free(&un);
   vk_stream_stats_free(&tm);
   rt_vec_free(&win_low);
   return 0;
}

static void window_record(void *user, uint32_t slot, VkCommandBuffer cb)
{
   record_frame(user, slot, 0, cb, 0);
}

static int run_window(scn *s)
{
#ifdef RT_HAVE_WAYLAND
   const rt_params *p = s->p;
   if (init_build(s) != VK_SUCCESS) return -1;
   char title[128];
   snprintf(title, sizeof(title), "gpubench-rt %s (Vulkan)", p->preset);
   rt_window *w = rt_window_open(s->c, p->width, p->height, title);
   if (!w) return -1;
   int n = rt_window_run(w, s->r->window_s, window_record, s, s->out.img);
   rt_window_close(w);
   fprintf(stderr, "window: %d frames\n", n);
   rt_result_skip(RT_TEST_SCENE, p, s->r->mode, RT_WIN_FRAMES, "interactive window mode: no record");
   return 0;
#else
   (void)s;
   fprintf(stderr, "built without Wayland support\n");
   return -1;
#endif
}

int rt_vk_scene(vk_ctx *c, const rt_params *p, const rt_run *r)
{
   if (strcmp(r->mode, "queue")) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "rt.scene runs only in mode=queue");
      return 0;
   }
   int nort = getenv("RTB_NO_RT") != NULL;
   if ((!c->has_as || !c->has_rq) && !nort) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "device lacks VK_KHR_acceleration_structure/ray_query");
      return 0;
   }
   if (nort) fprintf(stderr, "RTB_NO_RT: debug run without acceleration structures or rays\n");
   if (p->trace_mode == RT_TRACE_PIPELINE && !c->has_rtp) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "device lacks VK_KHR_ray_tracing_pipeline");
      return 0;
   }
   scn *s = calloc(1, sizeof(scn));
   s->c = c;
   s->p = p;
   s->r = r;
   s->nort = nort;
   uint32_t maxd = r->window ? RT_WIN_FRAMES : 1;
   for (uint32_t d = 0; d < r->ndepths; d++) if (r->depths[d] > maxd) maxd = r->depths[d];
   if (maxd > VK_STREAM_MAX) maxd = VK_STREAM_MAX;
   rt_scene_build(&s->sc, p);
   uint32_t vb = c->props.limits.timestampComputeAndGraphics ? 64 : 0;
   s->ts_ok = vb && c->props.limits.timestampPeriod > 0;
   s->ts_ns = c->props.limits.timestampPeriod;
   s->ts_mask = ~0ull;
   fprintf(stderr, "rt.scene: %u unique tris, %llu instanced tris, %u instances, %u BLAS\n", (unsigned)s->sc.unique_tris,
           (unsigned long long)s->sc.instanced_tris, s->sc.ninst, s->sc.nmeshes);
   int rc = 0;
   VkResult vr = create_resources(s, maxd);
   if (vr == VK_SUCCESS) vr = create_as(s);
   if (vr == VK_SUCCESS) vr = create_layouts(s);
   if (vr == VK_SUCCESS) vr = create_descriptors(s);
   uint64_t tp = rt_now_ns();
   if (vr == VK_SUCCESS) vr = create_pipelines(s);
   if (vr == VK_SUCCESS && p->trace_mode == RT_TRACE_PIPELINE) vr = rt_vtrace_create(&s->vtrace, c, s->pl_rt);
   if (vr == VK_SUCCESS) fprintf(stderr, "pipelines created in %.1f ms\n", (rt_now_ns() - tp) / 1e6);
   if (vr != VK_SUCCESS) {
      char e[128];
      snprintf(e, sizeof(e), "setup failed (Vulkan error %d)", vr);
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_fail(RT_TEST_SCENE, p, r->mode, r->depths[d], e);
      rc = 0;
   } else if (r->window) {
      rc = run_window(s) ? 1 : 0;
   } else {
      for (uint32_t d = 0; d < r->ndepths; d++) {
         if (run_depth(s, r->depths[d] > maxd ? maxd : r->depths[d])) break;
      }
   }
   destroy(s);
   free(s);
   return rc;
}

/* rt.rays: the scene geometry, static, built at t = 0 (small render targets, one slot). */
scn *rt_vk_static_scene(vk_ctx *c, const rt_params *p)
{
   scn *s = calloc(1, sizeof(scn));
   s->c = c;
   s->pcopy = *p;
   s->pcopy.blas_mode = RT_AS_STATIC;
   s->pcopy.tlas_mode = RT_AS_STATIC;
   s->pcopy.dyn_patches = 0;
   s->pcopy.moving = 0;
   s->pcopy.width = s->pcopy.height = 64;
   s->pcopy.trace_mode = RT_TRACE_QUERY;
   s->p = &s->pcopy;
   s->ts_ok = c->props.limits.timestampComputeAndGraphics && c->props.limits.timestampPeriod > 0;
   s->ts_ns = c->props.limits.timestampPeriod;
   s->ts_mask = ~0ull;
   rt_scene_build(&s->sc, s->p);
   VkResult vr = create_resources(s, 1);
   if (vr == VK_SUCCESS) vr = create_as(s);
   if (vr == VK_SUCCESS) vr = create_layouts(s);
   if (vr == VK_SUCCESS) vr = create_descriptors(s);
   if (vr == VK_SUCCESS) vr = create_pipelines(s);
   if (vr == VK_SUCCESS) vr = init_build(s);
   if (vr != VK_SUCCESS) {
      destroy(s);
      free(s);
      return NULL;
   }
   return s;
}

VkAccelerationStructureKHR rt_vk_static_scene_tlas(scn *s) { return s->tlas.as; }
const rt_scene *rt_vk_static_scene_desc(scn *s) { return &s->sc; }
double rt_vk_static_scene_build_ms(scn *s) { return s->init_ms[1] + s->init_ms[2]; }
void rt_vk_static_scene_free(scn *s)
{
   destroy(s);
   free(s);
}
