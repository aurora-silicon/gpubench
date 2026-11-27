// SPDX-License-Identifier: GPL-2.0-only
// render.postfx HDR scene PS.
struct PSIn {
   float4 pos : SV_Position;
   float4 hdr : COLOR0;
};

float4 main(PSIn i) : SV_Target
{
   return i.hdr;
}
