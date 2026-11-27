// SPDX-License-Identifier: GPL-2.0-only
// render.deferred (4x MSAA) lighting pass PS: per-sample fetch + resolve average.
Texture2DMS<float4, 4> tex_g0 : register(t0);
Texture2DMS<float4, 4> tex_g1 : register(t1);
Texture2DMS<float4, 4> tex_g2 : register(t2);
Texture2DMS<float4, 4> tex_g3 : register(t3);

static const float3 LDIR[4] = {
   float3( 0.57735027,  0.57735027, 0.57735027),
   float3(-0.53452248,  0.26726124, 0.80178373),
   float3( 0.26726124, -0.53452248, 0.80178373),
   float3(-0.40824829, -0.40824829, 0.81649658)
};
static const float3 LCOL[4] = {
   float3(0.95, 0.90, 0.85),
   float3(0.40, 0.55, 0.85),
   float3(0.75, 0.45, 0.30),
   float3(0.35, 0.70, 0.50)
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   int2 p = int2((int)pos.x, (int)pos.y);
   float3 total = float3(0.0, 0.0, 0.0);
   [unroll]
   for (int s = 0; s < 4; s++) {
      float4 g0 = tex_g0.Load(p, s);
      float4 g1 = tex_g1.Load(p, s);
      float4 g2 = tex_g2.Load(p, s);
      float linz = tex_g3.Load(p, s).r;

      float3 albedo = g0.rgb;
      float rough = g0.a;
      float3 n = normalize(g1.xyz);
      float metal = g1.w;
      float3 emissive = g2.rgb;
      float ao = g2.a;

      float3 col = albedo * 0.12 * ao + emissive;
      float shin = 8.0 + (1.0 - rough) * 24.0;
      float atten = 1.0 / (1.0 + linz * linz);
      [unroll]
      for (int i = 0; i < 4; i++) {
         float ndl = max(dot(n, LDIR[i]), 0.0);
         float3 h = normalize(LDIR[i] + float3(0.0, 0.0, 1.0));
         float ndh = max(dot(n, h), 0.0);
         float spec = pow(ndh, shin) * (0.2 + 0.8 * metal);
         float diff = ndl * (1.0 - 0.5 * metal);
         col += (albedo * diff + float3(spec, spec, spec)) * LCOL[i] * atten;
      }
      total += saturate(col);
   }
   return float4(total * 0.25, 1.0);
}
