// SPDX-License-Identifier: GPL-2.0-only
// render.shadow 3x3 hardware PCF receiver PS.
Texture2DArray<float> smap : register(t0);
SamplerComparisonState cmp_smp : register(s0);
cbuffer PcfCB : register(b0)
{
   float inv_res;
   float3 _pad;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   float2 uv = pos.xy * (1.0 / 1024.0);
   float qx = (uv.x >= 0.5) ? 1.0 : 0.0;
   float qy = (uv.y >= 0.5) ? 1.0 : 0.0;
   float layer = qx + 2.0 * qy;
   float2 luv = frac(uv * 2.0);

   float sum = 0.0;
   [unroll]
   for (int dy = -1; dy <= 1; dy++) {
      [unroll]
      for (int dx = -1; dx <= 1; dx++) {
         float2 suv = saturate(luv + float2((float)dx, (float)dy) * inv_res);
         sum += smap.SampleCmpLevelZero(cmp_smp, float3(suv, layer), 0.70);
      }
   }
   float sh = sum * (1.0 / 9.0);
   return float4(0.2 + 0.7 * sh, 0.25 + 0.5 * sh, 0.35 + 0.4 * sh, 1.0);
}
