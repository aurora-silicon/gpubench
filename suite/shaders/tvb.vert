// SPDX-License-Identifier: GPL-2.0-only
#version 450
// render.tvb: full-width slivers (about 1 px tall) at hashed heights, so each
// triangle is binned into every tile along a row.
void main()
{
   uint vid = uint(gl_VertexIndex);
   uint tri = vid / 3u, c = vid - tri * 3u;
   uint h = tri * 2654435761u;
   h ^= h >> 16u;
   float y = float(h & 65535u) * (1.0 / 32768.0) - 1.0;
   vec2 p = vec2(-1.0, y);
   if (c == 1u) p = vec2(1.0, y);
   if (c == 2u) p = vec2(1.0, y + 2.0 / 1080.0);
   gl_Position = vec4(p, 0.0, 1.0);
}
