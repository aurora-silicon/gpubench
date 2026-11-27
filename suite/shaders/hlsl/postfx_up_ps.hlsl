// SPDX-License-Identifier: GPL-2.0-only
// render.postfx 4-tap additive tent upsample.
Texture2D<float4> src_tex : register(t0);
SamplerState linear_smp : register(s0);
cbuffer UpCB : register(b0)
{
   float inv_w;
   float inv_h;
   float src_inv_w;
   float src_inv_h;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   float2 uv = pos.xy * float2(inv_w, inv_h);
   float2 d = float2(src_inv_w, src_inv_h);
   float3 c = (src_tex.SampleLevel(linear_smp, uv + float2(-d.x, -d.y), 0.0).rgb
             + src_tex.SampleLevel(linear_smp, uv + float2( d.x, -d.y), 0.0).rgb
             + src_tex.SampleLevel(linear_smp, uv + float2(-d.x,  d.y), 0.0).rgb
             + src_tex.SampleLevel(linear_smp, uv + float2( d.x,  d.y), 0.0).rgb) * 0.25;
   return float4(c * 0.5, 1.0);
}
