// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.heavy kind=geom: 512x512 grid of half-cell triangles (262144).
void main()
{
   uint vid = uint(gl_VertexIndex);
   uint tri = vid / 3u, c = vid - tri * 3u;
   uint gx = tri & 511u, gy = tri >> 9u;
   float cell = 2.0 / 512.0;
   vec2 p = vec2(float(gx), float(gy)) * cell - 1.0;
   if (c == 1u) p.x += cell;
   if (c == 2u) p.y += cell;
   gl_Position = vec4(p, 0.0, 1.0);
}
