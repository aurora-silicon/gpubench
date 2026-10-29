// SPDX-License-Identifier: GPL-2.0-only
/* Entry points of the Vulkan rt executor. */
#pragma once
#include "../common/params.h"
#include "vkutil.h"

int rt_vk_scene(vk_ctx *c, const rt_params *p, const rt_run *r);
int rt_vk_asbuild(vk_ctx *c, const rt_params *p, const rt_run *r);
int rt_vk_rays(vk_ctx *c, const rt_params *p, const rt_run *r);
int rt_vk_trav(vk_ctx *c, const rt_params *p, const rt_run *r);

/* Static scene geometry for rt.rays (scene_vk.c). */
struct scn;
struct rt_scene;
struct scn *rt_vk_static_scene(vk_ctx *c, const rt_params *p);
VkAccelerationStructureKHR rt_vk_static_scene_tlas(struct scn *s);
const struct rt_scene *rt_vk_static_scene_desc(struct scn *s);
double rt_vk_static_scene_build_ms(struct scn *s);
void rt_vk_static_scene_free(struct scn *s);

/* Ray-tracing pipeline variant of the reflection pass (trace=pipeline). */
typedef struct rt_vtrace {
   VkPipeline pipe;
   vk_buf sbt;
   VkStridedDeviceAddressRegionKHR rgen, miss, hit, call;
} rt_vtrace;
VkResult rt_vtrace_create(rt_vtrace *v, vk_ctx *c, VkPipelineLayout layout);
void rt_vtrace_record(rt_vtrace *v, VkCommandBuffer cb, VkDescriptorSet set, uint32_t w, uint32_t h);
void rt_vtrace_destroy(rt_vtrace *v, vk_ctx *c);
