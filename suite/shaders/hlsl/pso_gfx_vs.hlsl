// SPDX-License-Identifier: GPL-2.0-only
// compile.pso kind=graphics VS: distinct compiled variant per VAR_ID.
#ifndef VAR_ID
#define VAR_ID 1u
#endif

struct VSOut {
   float4 pos : SV_Position;
   float4 uv : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID)
{
   float2 p = float2((float)((vid << 1) & 2u), (float)(vid & 2u));
   float s = (float)((VAR_ID * 37u) & 255u) / 255.0;
   VSOut o;
   o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.5, 1.0);
   o.uv = float4(p, s, 1.0 - s);
   return o;
}
