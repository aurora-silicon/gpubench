// SPDX-License-Identifier: GPL-2.0-only
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rtpayload.glsl"

layout(location = 0) rayPayloadInEXT HitPayload hp_;

void main()
{
   hp_.t = -1.0;
}
