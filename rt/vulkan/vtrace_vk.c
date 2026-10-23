// SPDX-License-Identifier: GPL-2.0-only
/* trace=pipeline: the reflection pass as a ray-tracing pipeline (raygen + closest hit + any hit +
 * two miss shaders), with the same per-pixel shading as the ray-query compute shader. */
#include "rtvk.h"
#include <string.h>

static VkPipelineLayout g_layout;

VkResult rt_vtrace_create(rt_vtrace *v, vk_ctx *c, VkPipelineLayout layout)
{
   memset(v, 0, sizeof(*v));
   g_layout = layout;
   static const char *names[5] = {"rtpipe.rgen", "rtpipe.rmiss", "rtpipe_shadow.rmiss", "rtpipe.rchit", "rtpipe.rahit"};
   static const VkShaderStageFlagBits stages[5] = {VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_SHADER_STAGE_MISS_BIT_KHR,
                                                   VK_SHADER_STAGE_MISS_BIT_KHR, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                   VK_SHADER_STAGE_ANY_HIT_BIT_KHR};
   VkShaderModule mods[5];
   VkPipelineShaderStageCreateInfo st[5];
   for (int i = 0; i < 5; i++) {
      const vk_spv *x = vk_find_spv(names[i]);
      VKC(vk_shader(c, x->code, x->size, &mods[i]));
      st[i] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = stages[i],
                                                .module = mods[i],
                                                .pName = "main"};
   }
   VkRayTracingShaderGroupCreateInfoKHR g[4];
   for (int i = 0; i < 4; i++)
      g[i] = (VkRayTracingShaderGroupCreateInfoKHR){.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                                    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                                    .generalShader = VK_SHADER_UNUSED_KHR,
                                                    .closestHitShader = VK_SHADER_UNUSED_KHR,
                                                    .anyHitShader = VK_SHADER_UNUSED_KHR,
                                                    .intersectionShader = VK_SHADER_UNUSED_KHR};
   g[0].generalShader = 0;
   g[1].generalShader = 1;
   g[2].generalShader = 2;
   g[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
   g[3].closestHitShader = 3;
   g[3].anyHitShader = 4;
   VkRayTracingPipelineCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                                           .flags = c->exec_props_enabled ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0,
                                           .stageCount = 5,
                                           .pStages = st,
                                           .groupCount = 4,
                                           .pGroups = g,
                                           .maxPipelineRayRecursionDepth = 1,
                                           .layout = layout};
   vk_marker("rt_pipeline");
   VkResult r = c->CreateRTPipelines(c->dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, NULL, &v->pipe);
   for (int i = 0; i < 5; i++) vkDestroyShaderModule(c->dev, mods[i], NULL);
   vk_marker("internal");
   if (r != VK_SUCCESS) return r;

   uint32_t hs = c->rtprops.shaderGroupHandleSize, ha = c->rtprops.shaderGroupHandleAlignment,
            ba = c->rtprops.shaderGroupBaseAlignment;
   uint32_t stride = (hs + ha - 1) / ha * ha;
   uint32_t region = (stride + ba - 1) / ba * ba;
   uint8_t handles[4 * 64];
   VKC(c->GetRTGroupHandles(c->dev, v->pipe, 0, 4, 4 * hs, handles));
   /* regions: raygen | miss (2) | hit */
   VkDeviceSize miss_sz = (2 * stride + ba - 1) / ba * ba;
   VKC(vk_buf_create(c, &v->sbt, region + miss_sz + region,
                     VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 1));
   uint8_t *m = v->sbt.map;
   memcpy(m, handles, hs);
   memcpy(m + region, handles + hs, hs);
   memcpy(m + region + stride, handles + 2 * hs, hs);
   memcpy(m + region + miss_sz, handles + 3 * hs, hs);
   v->rgen = (VkStridedDeviceAddressRegionKHR){v->sbt.addr, region, region};
   v->miss = (VkStridedDeviceAddressRegionKHR){v->sbt.addr + region, stride, 2 * stride};
   v->hit = (VkStridedDeviceAddressRegionKHR){v->sbt.addr + region + miss_sz, stride, stride};
   v->call = (VkStridedDeviceAddressRegionKHR){0, 0, 0};
   return VK_SUCCESS;
}

void rt_vtrace_record(rt_vtrace *v, VkCommandBuffer cb, VkDescriptorSet set, uint32_t w, uint32_t h)
{
   extern PFN_vkCmdTraceRaysKHR rt_vk_trace_rays;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, v->pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, g_layout, 0, 1, &set, 0, NULL);
   rt_vk_trace_rays(cb, &v->rgen, &v->miss, &v->hit, &v->call, w, h, 1);
}

void rt_vtrace_destroy(rt_vtrace *v, vk_ctx *c)
{
   if (v->pipe) vkDestroyPipeline(c->dev, v->pipe, NULL);
   vk_buf_destroy(c, &v->sbt);
   memset(v, 0, sizeof(*v));
}
