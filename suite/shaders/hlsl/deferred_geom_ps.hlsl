// SPDX-License-Identifier: GPL-2.0-only
// render.deferred geometry pass PS: writes 4 MRT G-buffer attachments.
struct PSIn {
   float4 pos : SV_Position;
   nointerpolation uint v_qid : TEXCOORD0;
   nointerpolation float v_linz : TEXCOORD1;
};

struct PSOut {
   float4 g0 : SV_Target0;
   float4 g1 : SV_Target1;
   float4 g2 : SV_Target2;
   float  g3 : SV_Target3;
};

PSOut main(PSIn i)
{
   uint qid = i.v_qid;
   float3 albedo = float3(
      0.2 + 0.7 * (float)((qid * 37u) & 255u) / 255.0,
      0.2 + 0.7 * (float)((qid * 73u + 51u) & 255u) / 255.0,
      0.2 + 0.7 * (float)((qid * 131u + 17u) & 255u) / 255.0
   );
   float rough = 0.15 + 0.7 * (float)((qid * 19u) & 255u) / 255.0;
   float3 n = normalize(float3(
      (float)((qid * 53u) & 255u) / 255.0 * 0.8 - 0.4,
      (float)((qid * 97u) & 255u) / 255.0 * 0.8 - 0.4,
      1.0
   ));
   float metal = ((qid & 3u) == 0u) ? 0.8 : 0.1;
   float3 emissive = float3(
      0.05 * (float)((qid * 11u) & 255u) / 255.0,
      0.04 * (float)((qid * 23u) & 255u) / 255.0,
      0.06 * (float)((qid * 41u) & 255u) / 255.0
   );
   float ao = 0.6 + 0.4 * (float)((qid * 29u) & 255u) / 255.0;

   PSOut o;
   o.g0 = float4(albedo, rough);
   o.g1 = float4(n, metal);
   o.g2 = float4(emissive, ao);
   o.g3 = i.v_linz;
   return o;
}
