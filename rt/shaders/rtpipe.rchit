// SPDX-License-Identifier: GPL-2.0-only
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpayload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hp_;
hitAttributeEXT vec2 bary_;

void main()
{
   hp_.t = gl_HitTEXT;
   hp_.inst = uint(gl_InstanceID);
   hp_.prim = uint(gl_PrimitiveID);
   hp_.bary = bary_;
}
