// SPDX-License-Identifier: GPL-2.0-only
// compute.heavy kind=alu: 8 independent FMA chains, 4 steps each per
// iteration (32 FMA per iteration), `iters` iterations.
#include <metal_stdlib>
using namespace metal;
struct AluPC { uint iters; float a; float b; };
kernel void alu_main(device float *o [[buffer(0)]], constant AluPC &pc [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
   float x0 = float(gid) * 1e-6f;
   float x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
   float x4 = x0 + 4.0f, x5 = x0 + 5.0f, x6 = x0 + 6.0f, x7 = x0 + 7.0f;
   float a = pc.a, b = pc.b;
   for (uint i = 0u; i < pc.iters; i++) {
      x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
      x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
      x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
      x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
      x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
      x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
      x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
      x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
   }
   o[gid] = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
}
