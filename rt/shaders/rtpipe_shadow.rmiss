// SPDX-License-Identifier: GPL-2.0-only
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT uint shadow_hit_;

void main()
{
   shadow_hit_ = 0u;
}
