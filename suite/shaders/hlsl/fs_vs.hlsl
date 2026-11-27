// SPDX-License-Identifier: GPL-2.0-only
// Fullscreen triangle covering [0, W] x [0, H] in D3D Y-up clip space.
float4 main(uint vid : SV_VertexID) : SV_Position
{
   float2 p = float2((float)((vid << 1) & 2u), (float)(vid & 2u));
   return float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}
