// SPDX-License-Identifier: GPL-2.0-only
// Resources of the reflection pass (set 0), shared by rtreflect.comp and the ray-tracing
// pipeline shaders.
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1) uniform Frame { FRAME_UBO_BODY } fr;
layout(set = 0, binding = 2) uniform sampler2D g_albedo;
layout(set = 0, binding = 3) uniform sampler2D g_normal;
layout(set = 0, binding = 4) uniform sampler2D g_depth;
layout(std430, set = 0, binding = 5) readonly buffer VB { Vertex verts[]; };
layout(std430, set = 0, binding = 6) readonly buffer IB { uint indices[]; };
layout(std430, set = 0, binding = 7) readonly buffer IN { InstInfo info[]; };
layout(std430, set = 0, binding = 8) readonly buffer XF { Xform xf[]; };
layout(set = 0, binding = 9, rgba16f) uniform writeonly image2D o_rt;
layout(std430, set = 0, binding = 10) buffer CNT { uint counters[4]; };

// Interpolated UV of triangle prim of instance inst (alpha test and shading).
vec2 hit_uv(InstInfo ii, uint prim, vec2 b)
{
   uint base = ii.first_index + prim * 3u;
   Vertex v0 = verts[indices[base] + ii.base_vertex];
   Vertex v1 = verts[indices[base + 1u] + ii.base_vertex];
   Vertex v2 = verts[indices[base + 2u] + ii.base_vertex];
   float w = 1.0 - b.x - b.y;
   return vec2(v0.u, v0.v) * w + vec2(v1.u, v1.v) * b.x + vec2(v2.u, v2.v) * b.y;
}

// Non-opaque candidates are only the alpha-tested sails: accept unless the UV falls in a hole.
bool candidate_opaque(uint inst, uint prim, vec2 b)
{
   return !alpha_hole(hit_uv(info[inst], prim, b));
}
