// SPDX-License-Identifier: GPL-2.0-only
// compute.heavy kind=alu: 8 independent FMA chains, 4 steps each per
// iteration (32 FMA per iteration), `iters` iterations.
cbuffer AluCB : register(b0)
{
   uint iters;
   float a;
   float b;
   uint _pad;
};
RWByteAddressBuffer dst : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
   uint id = gid.x;
   float x0 = (float)id * 1e-6;
   float x1 = x0 + 1.0, x2 = x0 + 2.0, x3 = x0 + 3.0;
   float x4 = x0 + 4.0, x5 = x0 + 5.0, x6 = x0 + 6.0, x7 = x0 + 7.0;
   float ma = a, mb = b;
   [loop]
   for (uint i = 0u; i < iters; i++) {
      x0 = mad(x0, ma, mb); x1 = mad(x1, ma, mb); x2 = mad(x2, ma, mb); x3 = mad(x3, ma, mb);
      x4 = mad(x4, ma, mb); x5 = mad(x5, ma, mb); x6 = mad(x6, ma, mb); x7 = mad(x7, ma, mb);
      x0 = mad(x0, ma, mb); x1 = mad(x1, ma, mb); x2 = mad(x2, ma, mb); x3 = mad(x3, ma, mb);
      x4 = mad(x4, ma, mb); x5 = mad(x5, ma, mb); x6 = mad(x6, ma, mb); x7 = mad(x7, ma, mb);
      x0 = mad(x0, ma, mb); x1 = mad(x1, ma, mb); x2 = mad(x2, ma, mb); x3 = mad(x3, ma, mb);
      x4 = mad(x4, ma, mb); x5 = mad(x5, ma, mb); x6 = mad(x6, ma, mb); x7 = mad(x7, ma, mb);
      x0 = mad(x0, ma, mb); x1 = mad(x1, ma, mb); x2 = mad(x2, ma, mb); x3 = mad(x3, ma, mb);
      x4 = mad(x4, ma, mb); x5 = mad(x5, ma, mb); x6 = mad(x6, ma, mb); x7 = mad(x7, ma, mb);
   }
   float res = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
   dst.Store(id * 4u, asuint(res));
}
