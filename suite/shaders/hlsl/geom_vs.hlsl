// SPDX-License-Identifier: GPL-2.0-only
// render.heavy kind=geom: 512x512 grid of half-cell triangles (262144).
// Y is flipped to D3D clip space so (gx=0, gy=0) covers top-left pixel (0,0).
float4 main(uint vid : SV_VertexID) : SV_Position
{
   uint tri = vid / 3u;
   uint c = vid - tri * 3u;
   uint gx = tri & 511u;
   uint gy = tri >> 9u;
   float cell = 2.0 / 512.0;
   float2 p = float2((float)gx, (float)gy) * cell - 1.0;
   if (c == 1u) p.x += cell;
   if (c == 2u) p.y += cell;
   return float4(p.x, -p.y, 0.0, 1.0);
}
