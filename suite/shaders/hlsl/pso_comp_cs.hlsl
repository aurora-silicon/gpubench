// SPDX-License-Identifier: GPL-2.0-only
// compile.pso kind=compute: parameterized compute shader variant compiled with
// distinct VAR_ID/LOOP_N/MASK macros into distinct DXBC/DXIL bytecode blobs.
#ifndef VAR_ID
#define VAR_ID 1u
#endif
#ifndef LOOP_N
#define LOOP_N 8u
#endif
#ifndef MASK
#define MASK 3u
#endif

RWByteAddressBuffer dst : register(u0);
groupshared uint scratch[128];

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 ltid : SV_GroupThreadID)
{
   uint id = gid.x;
   uint lid = ltid.x;
   uint acc = (id * 2654435761u) ^ (VAR_ID * 0x9e3779b9u) ^ 0x51a71234u;
   scratch[lid] = acc;
   scratch[lid + 64u] = acc ^ 0xa5a5a5a5u;
   GroupMemoryBarrierWithGroupSync();
   [unroll]
   for (uint i = 0u; i < LOOP_N; i++) {
      uint s = scratch[(lid + i + (VAR_ID & 7u)) & 127u];
      if (((i + VAR_ID) & MASK) == 0u)
         acc = (acc * 16777619u) ^ (s >> ((VAR_ID & 7u) + 1u));
      else
         acc = (acc + s + (VAR_ID * 131u)) ^ (acc << ((VAR_ID & 3u) + 1u));
   }
   dst.Store(id * 4u, acc);
}
