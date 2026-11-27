// SPDX-License-Identifier: GPL-2.0-only
// render.shadow depth pass VS: 4 cascades, instanced 32x32 shadow-caster quads.
cbuffer ShadowCB : register(b0)
{
   uint cascade;
   uint res;
   uint2 _pad;
};

struct VSOut {
   float4 pos : SV_Position;
   float2 uv : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID, uint inst : SV_InstanceID)
{
   uint cell = inst & 1023u;
   uint gx = cell & 31u;
   uint gy = cell >> 5u;

   static const float2 corners[6] = {
      float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
      float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)
   };
   float2 corner = corners[vid % 6u];
   float2 uv = (float2((float)gx, (float)gy) + corner) * (1.0 / 32.0);
   bool caster = (((gx + gy + cascade) & 1u) == 0u);
   float z = caster ? ((inst < 1024u) ? (0.30 + 0.05 * (float)cascade)
                                      : (0.60 + 0.05 * (float)cascade))
                    : 0.85;
   VSOut o;
   o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
   o.uv = corner;
   return o;
}
