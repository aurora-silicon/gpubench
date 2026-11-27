// SPDX-License-Identifier: GPL-2.0-only
// compute.light: one workgroup of 32 threads, one uint store each.
cbuffer LightCB : register(b0)
{
   uint v;
   uint3 _pad;
};
RWByteAddressBuffer dst : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
   dst.Store(gid.x * 4u, v + gid.x);
}
