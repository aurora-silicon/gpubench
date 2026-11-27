// SPDX-License-Identifier: GPL-2.0-only
#version 450
// Fullscreen affine texture step for render.pass_chain.
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(push_constant) uniform PC { vec2 inv_size; float scale; float bias; } pc;
layout(location = 0) out vec4 o;
void main()
{
   vec2 uv = gl_FragCoord.xy * pc.inv_size;
   vec4 s = textureLod(tex, uv, 0.0);
   o = vec4(s.rgb * pc.scale + pc.bias, 1.0);
}
