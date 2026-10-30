// SPDX-License-Identifier: GPL-2.0-only
#version 450
// Render link of a cross-queue chain: reads the value the previous (compute) link wrote for
// this pixel and outputs v * 5 + k + (idx ^ 0x55) into an R32_UINT target.
// p.x = base, p.y = width, p.z = k.
layout(push_constant) uniform PC { uvec4 p; vec4 color; } pc;
layout(set = 0, binding = 0, std430) readonly buffer D { uint v[]; };
layout(location = 0) out uint o;
void main()
{
   uint x = uint(gl_FragCoord.x), y = uint(gl_FragCoord.y);
   uint idx = y * pc.p.y + x;
   o = v[pc.p.x + idx] * 5u + pc.p.z + (idx ^ 0x55u);
}
