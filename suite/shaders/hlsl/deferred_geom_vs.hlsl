// SPDX-License-Identifier: GPL-2.0-only
// render.deferred geometry pass VS: 4096 quads = 4 depth layers x 32x32 grid.
// Drawn back-to-front (layer 3 -> 0) so depth test LESS lets layer 0 win at z=0.25.
// Supports base_quad offset via cbuffer for multi-command-list split G-buffer passes.
cbuffer DeferredCB : register(b0)
{
   uint base_quad;
   uint3 _pad;
};

struct VSOut {
   float4 pos : SV_Position;
   nointerpolation uint v_qid : TEXCOORD0;
   nointerpolation float v_linz : TEXCOORD1;
};

VSOut main(uint vid : SV_VertexID)
{
   uint quad = base_quad + (vid / 6u);
   uint corner = vid - (vid / 6u) * 6u;
   uint layer = 3u - (quad >> 10u);
   uint cell = quad & 1023u;
   uint qx = cell & 31u;
   uint qy = cell >> 5u;

   float2 off;
   if (corner == 0u) off = float2(0.0, 0.0);
   else if (corner == 1u) off = float2(1.0, 0.0);
   else if (corner == 2u) off = float2(0.0, 1.0);
   else if (corner == 3u) off = float2(1.0, 0.0);
   else if (corner == 4u) off = float2(1.0, 1.0);
   else off = float2(0.0, 1.0);

   float2 uv = (float2((float)qx, (float)qy) + off) * (1.0 / 32.0);
   float z = 0.25 + (float)layer * 0.20;
   VSOut o;
   o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
   o.v_qid = cell + layer * 1024u;
   o.v_linz = z;
   return o;
}
