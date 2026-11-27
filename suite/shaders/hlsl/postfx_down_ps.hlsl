// SPDX-License-Identifier: GPL-2.0-only
// render.postfx 5-tap bloom downsample + optional bright-pass threshold.
Texture2D<float4> src_tex : register(t0);
SamplerState linear_smp : register(s0);
cbuffer DownCB : register(b0)
{
   float inv_w;
   float inv_h;
   float src_inv_w;
   float src_inv_h;
   uint bright_pass;
   uint3 _pad;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   float2 uv = pos.xy * float2(inv_w, inv_h);
   float2 d = float2(src_inv_w, src_inv_h);
   float3 c = src_tex.SampleLevel(linear_smp, uv, 0.0).rgb * 0.5
            + src_tex.SampleLevel(linear_smp, uv + float2(-d.x, -d.y), 0.0).rgb * 0.125
            + src_tex.SampleLevel(linear_smp, uv + float2( d.x, -d.y), 0.0).rgb * 0.125
            + src_tex.SampleLevel(linear_smp, uv + float2(-d.x,  d.y), 0.0).rgb * 0.125
            + src_tex.SampleLevel(linear_smp, uv + float2( d.x,  d.y), 0.0).rgb * 0.125;
   if (bright_pass != 0u)
      c = max(c - float3(1.0, 1.0, 1.0), float3(0.0, 0.0, 0.0));
   return float4(c, 1.0);
}
