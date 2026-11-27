// SPDX-License-Identifier: GPL-2.0-only
// render.draws fragment shader.
struct PSIn {
   float4 pos : SV_Position;
   float3 col : COLOR0;
};

float4 main(PSIn i) : SV_Target
{
   return float4(i.col, 1.0);
}
