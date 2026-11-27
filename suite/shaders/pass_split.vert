// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.pass_split: one screen-space rectangle per draw, pixel bounds and
// depth from push constants (two triangles, no vertex buffer).
layout(push_constant) uniform PC {
   ivec4 rect;    // x0, y0, x1, y1 in pixels
   vec2 scale;    // 2 / width, 2 / height
   float z;
   uint draw;
} pc;
void main()
{
   const int cx[6] = int[](0, 1, 0, 1, 1, 0);
   const int cy[6] = int[](0, 0, 1, 0, 1, 1);
   vec2 px = vec2(cx[gl_VertexIndex] != 0 ? pc.rect.z : pc.rect.x,
                  cy[gl_VertexIndex] != 0 ? pc.rect.w : pc.rect.y);
   gl_Position = vec4(px * pc.scale - 1.0, pc.z, 1.0);
}
