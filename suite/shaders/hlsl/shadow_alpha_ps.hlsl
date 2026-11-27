// SPDX-License-Identifier: GPL-2.0-only
// render.shadow alpha-tested depth pass PS.
struct PSIn {
   float4 pos : SV_Position;
   float2 uv : TEXCOORD0;
};

void main(PSIn i)
{
   float m = sin(i.uv.x * 18.8495559) * sin(i.uv.y * 18.8495559);
   clip(m);
}
