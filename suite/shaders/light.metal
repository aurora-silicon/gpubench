// SPDX-License-Identifier: GPL-2.0-only
// compute.light: one threadgroup of 32 threads, one store each.
#include <metal_stdlib>
using namespace metal;
kernel void light_main(device uint *o [[buffer(0)]], constant uint &v [[buffer(1)]],
                       uint gid [[thread_position_in_grid]])
{
   o[gid] = v + gid;
}
