// SPDX-License-Identifier: GPL-2.0-only
// One nearest-filtered, repeating RGBA8 lookup per fragment, mip level zero.
Texture2D<float4> tex : register(t0);
SamplerState smp : register(s0);
cbuffer ColorCB : register(b0)
{
   float4 color;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   return tex.SampleLevel(smp, pos.xy / 256.0, 0.0) * color;
}
