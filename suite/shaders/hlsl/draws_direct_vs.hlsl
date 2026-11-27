// SPDX-License-Identifier: GPL-2.0-only
// render.draws mode=direct: per-draw constant buffer rect & color (10x10 quads in 1024x1024).
cbuffer DrawCB : register(b0)
{
   float4 rect_col;
};

struct VSOut {
   float4 pos : SV_Position;
   float3 col : COLOR0;
};

VSOut main(float2 in_pos : POSITION)
{
   VSOut o;
   float2 px = rect_col.xy + in_pos * 10.0;
   float2 uv = px / 1024.0;
   o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
   o.col = float3(rect_col.zw, 0.5);
   return o;
}
