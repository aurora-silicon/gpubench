// SPDX-License-Identifier: GPL-2.0-only
#version 460
// G-buffer fragment shader. RT0 = albedo, roughness; RT1 = world normal, reflectivity.
// ALPHA (specialization constant 0) enables the alpha test of the sails.
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(constant_id = 0) const bool ALPHA = false;
layout(set = 0, binding = 0) uniform Frame { FRAME_UBO_BODY } fr;
layout(std430, set = 0, binding = 2) readonly buffer IN { InstInfo info[]; };

layout(location = 0) in vec3 i_wpos;
layout(location = 1) in vec3 i_nrm;
layout(location = 2) in vec2 i_uv;
layout(location = 3) flat in uint i_inst;
layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_normal;

void main()
{
   InstInfo ii = info[i_inst];
   if (ALPHA && alpha_hole(i_uv)) discard;
   vec3 n = normalize(i_nrm);
   if ((ii.flags & FLAG_TWOSIDED) != 0u && dot(n, fr.cam_pos.xyz - i_wpos) < 0.0) n = -n;
   o_albedo = vec4(surface_albedo(ii, i_uv), ii.params.y);
   o_normal = vec4(n, ii.params.x);
}
