// SPDX-License-Identifier: GPL-2.0-only
#version 460
// G-buffer vertex shader: per-instance 3x4 transform from a storage buffer, indexed by the instance
// index (draws are instanced per mesh, firstInstance = first instance of the mesh).
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_nrm;
layout(location = 2) in vec2 in_uv;
layout(set = 0, binding = 0) uniform Frame { FRAME_UBO_BODY } fr;
layout(std430, set = 0, binding = 1) readonly buffer XF { Xform xf[]; };

layout(location = 0) out vec3 o_wpos;
layout(location = 1) out vec3 o_nrm;
layout(location = 2) out vec2 o_uv;
layout(location = 3) flat out uint o_inst;

void main()
{
   Xform t = xf[gl_InstanceIndex];
   vec3 w = xform_point(t, in_pos);
   gl_Position = fr.view_proj * vec4(w, 1.0);
   o_wpos = w;
   o_nrm = xform_dir(t, in_nrm);
   o_uv = in_uv;
   o_inst = uint(gl_InstanceIndex);
}
