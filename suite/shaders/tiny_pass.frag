// SPDX-License-Identifier: GPL-2.0-only
#version 450
/* Tiny fullscreen post/bloom pass shader: 4-tap box filter + push weight. */
layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(push_constant) uniform PC {
   vec4 params; /* xy = inv_size, z = weight, w = bias */
} pc;
layout(location = 0) out vec4 o_color;
void main()
{
   vec2 uv = gl_FragCoord.xy * pc.params.xy;
   vec2 d = pc.params.xy * 0.5;
   vec4 s = 0.25 * (texture(u_src, uv + vec2(-d.x, -d.y)) +
                    texture(u_src, uv + vec2( d.x, -d.y)) +
                    texture(u_src, uv + vec2(-d.x,  d.y)) +
                    texture(u_src, uv + vec2( d.x,  d.y)));
   o_color = vec4(s.rgb * pc.params.z + vec3(pc.params.w), 1.0);
}
