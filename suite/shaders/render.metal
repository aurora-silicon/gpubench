// SPDX-License-Identifier: GPL-2.0-only
// Render shaders: the same algorithms as fs.vert, color.frag, geom.vert and
// tvb.vert.
#include <metal_stdlib>
using namespace metal;

// One triangle that covers the whole viewport.
vertex float4 fs_vert(uint vid [[vertex_id]])
{
   float2 p = float2(float((vid << 1) & 2), float(vid & 2));
   return float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
}

// Flat color from the bound constant.
fragment float4 color_frag(constant float4 &color [[buffer(0)]])
{
   return color;
}

// 512x512 grid of half-cell triangles (262144).
vertex float4 geom_vert(uint vid [[vertex_id]])
{
   uint tri = vid / 3u, c = vid - tri * 3u;
   uint gx = tri & 511u, gy = tri >> 9u;
   float cell = 2.0f / 512.0f;
   float2 p = float2(float(gx), float(gy)) * cell - 1.0f;
   if (c == 1u) p.x += cell;
   if (c == 2u) p.y += cell;
   return float4(p, 0.0f, 1.0f);
}

// Full-width slivers (about 1 px tall) at hashed heights.
vertex float4 tvb_vert(uint vid [[vertex_id]])
{
   uint tri = vid / 3u, c = vid - tri * 3u;
   uint h = tri * 2654435761u;
   h ^= h >> 16u;
   float y = float(h & 65535u) * (1.0f / 32768.0f) - 1.0f;
   float2 p = float2(-1.0f, y);
   if (c == 1u) p = float2(1.0f, y);
   if (c == 2u) p = float2(1.0f, y + 2.0f / 1080.0f);
   return float4(p, 0.0f, 1.0f);
}
