// SPDX-License-Identifier: GPL-2.0-only
// compile.pso kind=graphics PS: multi-light shader variant per VAR_ID/LOOP_N/MASK.
#ifndef VAR_ID
#define VAR_ID 1u
#endif
#ifndef LOOP_N
#define LOOP_N 6u
#endif
#ifndef MASK
#define MASK 3u
#endif

struct PSIn {
   float4 pos : SV_Position;
   float4 uv : TEXCOORD0;
};

float4 main(PSIn inp) : SV_Target
{
   float3 n = normalize(float3(inp.uv.xy * 2.0 - 1.0, 0.8 + 0.1 * inp.uv.z));
   float3 albedo = inp.uv.xyz;
   float3 col = albedo * 0.1;
   float phase = (float)((VAR_ID * 17u) & 1023u) * 0.006;
   [unroll]
   for (uint i = 0u; i < LOOP_N; i++) {
      float a = phase + (float)i * 0.7;
      float3 l = normalize(float3(sin(a), cos(a), 0.7));
      float ndl = max(dot(n, l), 0.0);
      float3 h = normalize(l + float3(0.0, 0.0, 1.0));
      float spec = pow(max(dot(n, h), 0.0), 8.0 + (float)((i + VAR_ID) & MASK) * 6.0);
      col += (albedo * ndl + float3(spec * 0.3, spec * 0.3, spec * 0.3)) * (0.25 / (float)LOOP_N);
   }
   return float4(saturate(col), 1.0);
}
