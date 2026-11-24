// SPDX-License-Identifier: GPL-2.0-only
#version 450

// render.msaa_merge: Compressible 16x16 tiled additive fragment shader for
// multi-pass 2x/4x MSAA color-compressed render target merging (RGBA8 / RGBA16F).
layout(location = 0) out vec4 out_color;
layout(push_constant) uniform PC {
   uint pass_idx;
   uint passes;
   float scale;
   uint pad;
} pc;

void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   uvec2 b = p >> 4u;
   uint h = (b.x * 73u + b.y * 151u + pc.pass_idx * 37u) & 31u;
   uint r = (h + pc.pass_idx * 3u) & 31u;
   uint g = ((h ^ (b.x & 15u)) + pc.pass_idx * 5u) & 31u;
   uint bl = ((h ^ (b.y & 15u)) + pc.pass_idx * 7u) & 31u;
   out_color = vec4(float(r), float(g), float(bl), 31.0) * pc.scale;
}
