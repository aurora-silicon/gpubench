// SPDX-License-Identifier: GPL-2.0-only
// render.postfx HDR scene VS: 64x64 grid of HDR quads with bright center sun.
struct VSOut {
   float4 pos : SV_Position;
   float4 hdr : COLOR0;
};

VSOut main(uint vid : SV_VertexID)
{
   uint q = vid / 6u;
   uint v = vid - q * 6u;
   uint gx = q & 63u;
   uint gy = (q >> 6u) & 63u;

   static const float2 corners[6] = {
      float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
      float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)
   };
   float2 uv = (float2((float)gx, (float)gy) + corners[v]) * (1.0 / 64.0);
   float dx = (float)gx - 31.5;
   float dy = (float)gy - 31.5;
   float d2 = dx * dx + dy * dy;
   VSOut o;
   if (d2 < 100.0) {
      float boost = 3.5 - d2 * 0.02;
      o.hdr = float4(boost, boost * 0.65, boost * 0.30, 1.0);
   } else {
      o.hdr = float4(0.15 + 0.55 * uv.x, 0.15 + 0.55 * uv.y, 0.35, 1.0);
   }
   float z = 0.2 + 0.6 * (float)((gx ^ gy) & 15u) / 15.0;
   o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
   return o;
}
