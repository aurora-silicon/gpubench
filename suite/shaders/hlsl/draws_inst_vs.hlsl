// SPDX-License-Identifier: GPL-2.0-only
// render.draws mode=instanced|indirect: per-instance draw_id from input slot 1
// (PER_INSTANCE_DATA so StartInstanceLocation in ExecuteIndirect / DrawIndexedInstancedIndirect
// selects the exact draw index in both D3D11 and D3D12).
struct VSOut {
   float4 pos : SV_Position;
   float3 col : COLOR0;
};

VSOut main(float2 in_pos : POSITION, uint draw_id : TEXCOORD0)
{
   uint d = draw_id % 10000u;
   uint cx = d % 100u;
   uint cy = d / 100u;
   float2 px = float2((float)(cx * 10u), (float)(cy * 10u)) + in_pos * 10.0;
   float2 uv = px / 1024.0;
   VSOut o;
   o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
   o.col = float3((float)((d * 37u) & 255u) / 255.0, (float)((d * 73u) & 255u) / 255.0, 0.5);
   return o;
}
