// SPDX-License-Identifier: GPL-2.0-only
#version 450
// Tiler stress: quad q covers exactly one framebuffer row, chosen by a hash, across the full
// width, so every quad is binned into every tile along its row. p.x = rows, p.y = seed.
layout(push_constant) uniform PC { uvec4 p; vec4 color; } pc;
uint row_of(uint q)
{
   uint h = (q + pc.p.y) * 2654435761u;
   h ^= h >> 15u;
   h *= 2246822519u;
   h ^= h >> 13u;
   return h % pc.p.x;
}
void main()
{
   uint vid = uint(gl_VertexIndex);
   uint q = vid / 6u, c = vid - q * 6u;
   uint r = row_of(q);
   uint cx = (c == 1u || c == 2u || c == 4u) ? 1u : 0u;
   uint cy = (c == 2u || c == 4u || c == 5u) ? 1u : 0u;
   float y = (float(r + cy) / float(pc.p.x)) * 2.0 - 1.0;
   gl_Position = vec4(float(cx) * 2.0 - 1.0, y, 0.0, 1.0);
}
