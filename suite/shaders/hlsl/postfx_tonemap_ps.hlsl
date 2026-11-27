// SPDX-License-Identifier: GPL-2.0-only
// render.postfx ACES filmic tonemap + vignette -> RGBA8.
Texture2D<float4> hdr_tex : register(t0);
Texture2D<float4> bloom_tex : register(t1);
SamplerState linear_smp : register(s0);
cbuffer ToneCB : register(b0)
{
   float inv_w;
   float inv_h;
   float2 _pad;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   float2 uv = pos.xy * float2(inv_w, inv_h);
   float3 hdr = hdr_tex.SampleLevel(linear_smp, uv, 0.0).rgb
              + 0.35 * bloom_tex.SampleLevel(linear_smp, uv, 0.0).rgb;
   float2 dv = uv - float2(0.5, 0.5);
   float vig = clamp(1.0 - dot(dv, dv) * 1.4, 0.25, 1.0);
   hdr *= vig;
   float3 mapped = saturate((hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14));
   return float4(pow(mapped, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2)), 1.0);
}
