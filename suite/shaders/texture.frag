// SPDX-License-Identifier: GPL-2.0-only
#version 450
// One nearest-filtered, repeating RGBA8 lookup per fragment, mip level zero.
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(push_constant) uniform PC { vec4 color; } pc;
layout(location = 0) out vec4 o;
void main()
{
   o = textureLod(tex, gl_FragCoord.xy / 256.0, 0.0) * pc.color;
}
