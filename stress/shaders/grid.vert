// SPDX-License-Identifier: GPL-2.0-only
#version 450
// p.x x p.x cells, two triangles each, tiling the viewport exactly once (6 vertices per cell).
layout(push_constant) uniform PC { uvec4 p; vec4 color; } pc;
void main()
{
   uint vid = uint(gl_VertexIndex);
   uint cell = vid / 6u, c = vid - cell * 6u;
   uint g = pc.p.x;
   uint gx = cell % g, gy = cell / g;
   // corners: 0 (0,0) 1 (1,0) 2 (1,1) | 3 (0,0) 4 (1,1) 5 (0,1)
   uint cx = (c == 1u || c == 2u || c == 4u) ? 1u : 0u;
   uint cy = (c == 2u || c == 4u || c == 5u) ? 1u : 0u;
   vec2 q = vec2(float(gx + cx), float(gy + cy)) / float(g);
   gl_Position = vec4(q * 2.0 - 1.0, 0.0, 1.0);
}
