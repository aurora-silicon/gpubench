// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.pass_split: a G-buffer fragment program. 32 rounds of an integer
// hash of the pixel and the draw (about 230 instructions), then one 8-bit
// pattern per render target, so the CPU reference is exact for UNORM8 and
// UNORM16 targets alike (k/255 stores as k and as 257k).
layout(push_constant) uniform PC {
   ivec4 rect;
   vec2 scale;
   float z;
   uint draw;
} pc;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;

uint mix32(uint h)
{
   h ^= h >> 16;
   h *= 0x7feb352du;
   h ^= h >> 15;
   h *= 0x846ca68bu;
   h ^= h >> 16;
   return h;
}

void main()
{
   uvec2 p = uvec2(gl_FragCoord.xy);
   uint h = (p.x * 0x9e3779b1u) ^ (p.y * 0x85ebca77u) ^ (pc.draw * 0xc2b2ae3du);
   for (uint i = 0u; i < 32u; ++i)
      h = mix32(h + i);
   o0 = unpackUnorm4x8(h);
   o1 = unpackUnorm4x8(h ^ 0x5a5a5a5au);
   o2 = unpackUnorm4x8(h * 0x01000193u);
   o3 = unpackUnorm4x8(h + 0x3c6ef372u);
   o4 = unpackUnorm4x8(~h);
}
