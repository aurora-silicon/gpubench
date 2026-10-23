// SPDX-License-Identifier: GPL-2.0-only
// compute.heavy kind=bw: stream copy, one float4 per thread.
#include <metal_stdlib>
using namespace metal;
kernel void bw_main(const device float4 *s [[buffer(0)]], device float4 *d [[buffer(1)]],
                    uint i [[thread_position_in_grid]])
{
   d[i] = s[i];
}
