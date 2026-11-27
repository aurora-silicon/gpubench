// SPDX-License-Identifier: GPL-2.0-only
#version 450
/* Material fragment shader reading its own dynamic UBO slot (binding 1),
 * separate from the per-draw vertex dynamic UBO (binding 0). */
layout(set = 0, binding = 1) uniform MatUBO {
   vec4 scale;
   vec4 bias;
} u_mat;
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main()
{
   vec4 c = clamp(v_color * u_mat.scale + u_mat.bias, 0.0, 1.0);
   o_color = c;
}
