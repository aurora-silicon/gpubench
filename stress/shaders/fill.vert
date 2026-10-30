// SPDX-License-Identifier: GPL-2.0-only
#version 450
// One triangle that covers the whole viewport.
layout(push_constant) uniform PC { uvec4 p; vec4 color; } pc;
void main()
{
   vec2 q = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
   gl_Position = vec4(q * 2.0 - 1.0, 0.0, 1.0);
}
