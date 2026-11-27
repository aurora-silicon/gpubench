// SPDX-License-Identifier: GPL-2.0-only
// transfer.ops kind=mipchain bilinear mip downsample pass.
Texture2D<float4> src_tex : register(t0);
SamplerState linear_smp : register(s0);
cbuffer MipCB : register(b0)
{
   float inv_w;
   float inv_h;
   float src_mip;
   float _pad;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   float2 uv = pos.xy * float2(inv_w, inv_h);
   return src_tex.SampleLevel(linear_smp, uv, src_mip);
}
