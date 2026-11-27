// SPDX-License-Identifier: GPL-2.0-only
// Flat color from constant buffer.
cbuffer ColorCB : register(b0)
{
   float4 color;
};

float4 main(float4 pos : SV_Position) : SV_Target
{
   return color;
}
