// SPDX-License-Identifier: GPL-2.0-only
// compute.heavy kind=bw: stream copy, one float4 (16 bytes) per thread.
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
   uint off = gid.x * 16u;
   dst.Store4(off, src.Load4(off));
}
