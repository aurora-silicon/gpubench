// SPDX-License-Identifier: GPL-2.0-only
#version 450
/* DX11/DXVK-style draw vertex shader: per-draw dynamic UBO transform +
 * vertex buffer attribute input + optional push-constant offset. */
layout(location = 0) in vec4 in_pos;
layout(set = 0, binding = 0) uniform DrawUBO {
   vec4 offset_scale;
   vec4 tint;
} u_draw;
layout(push_constant) uniform PC {
   vec4 push_off;
} pc;
layout(location = 0) out vec4 v_color;
void main()
{
   vec2 xy = in_pos.xy * u_draw.offset_scale.zw + u_draw.offset_scale.xy + pc.push_off.xy;
   gl_Position = vec4(xy, in_pos.z, 1.0);
   v_color = u_draw.tint + vec4(pc.push_off.zw, 0.0, 0.0);
}
