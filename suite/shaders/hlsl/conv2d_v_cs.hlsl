// SPDX-License-Identifier: GPL-2.0-only
// compute.conv2d Pass 2 (Vertical 17-tap separable Gaussian / Bilateral)
// Cooperative 16x32 shared-memory halo tile for 16x16 workgroup.
#ifndef BILATERAL
#define BILATERAL 0
#endif

cbuffer ConvCB : register(b0)
{
   int w;
   int h;
   int2 _pad;
};
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

static const uint W17[17] = {
   16u, 86u, 342u, 1052u, 2554u, 4996u, 7914u, 10242u,
   11132u,
   10242u, 7914u, 4996u, 2554u, 1052u, 342u, 86u, 16u
};

groupshared uint tile[32 * 16];

[numthreads(16, 16, 1)]
void main(uint3 lid : SV_GroupThreadID, uint3 wid : SV_GroupID)
{
   int lx = (int)lid.x;
   int ly = (int)lid.y;
   int gx = (int)(wid.x * 16u) + lx;
   int gy = (int)(wid.y * 16u);
   int sx = clamp(gx, 0, w - 1);

   for (int dy = ly; dy < 32; dy += 16) {
      int sy = clamp(gy + dy - 8, 0, h - 1);
      tile[dy * 16 + lx] = src.Load((uint)(sy * w + sx) * 4u);
   }
   GroupMemoryBarrierWithGroupSync();

   int y = gy + ly;
   if (gx >= w || y >= h)
      return;

   uint cpx = tile[(ly + 8) * 16 + lx];
   uint4 cc = (uint4(cpx, cpx, cpx, cpx) >> uint4(0u, 8u, 16u, 24u)) & 255u;
   int cluma = (int)((cc.r + 2u * cc.g + cc.b) >> 2u);
   uint4 acc = uint4(0u, 0u, 0u, 0u);
   uint wsum = 0u;
   for (int t = 0; t < 17; t++) {
      uint p = tile[(ly + t) * 16 + lx];
      uint4 ch = (uint4(p, p, p, p) >> uint4(0u, 8u, 16u, 24u)) & 255u;
      uint wt = W17[t];
#if BILATERAL
      int nluma = (int)((ch.r + 2u * ch.g + ch.b) >> 2u);
      int diff = abs(cluma - nluma);
      uint rw = diff < 64 ? (256u - (uint)diff * 4u) : 1u;
      wt = (wt >> 4u) * rw;
#endif
      acc += ch * wt;
      wsum += wt;
   }
   uint4 res;
#if BILATERAL
   uint4 halfw = uint4(wsum >> 1u, wsum >> 1u, wsum >> 1u, wsum >> 1u);
   res = (acc + halfw) / uint4(wsum, wsum, wsum, wsum);
#else
   res = (acc + uint4(32768u, 32768u, 32768u, 32768u)) >> uint4(16u, 16u, 16u, 16u);
#endif
   res = (res & 255u) << uint4(0u, 8u, 16u, 24u);
   dst.Store((uint)(y * w + gx) * 4u, res.r | res.g | res.b | res.a);
}
