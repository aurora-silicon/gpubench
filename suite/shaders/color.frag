// SPDX-License-Identifier: GPL-2.0-only
#version 450
// Flat color from push constants.
layout(push_constant) uniform PC { vec4 color; } pc;
layout(location = 0) out vec4 o;
void main()
{
   o = pc.color;
}
