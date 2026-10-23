// SPDX-License-Identifier: GPL-2.0-only
// Shared declarations for the rtbench GLSL shaders. Layouts mirror rt/common/scene.h and
// rt/shaders/rt.metal; every function here has a line-for-line MSL twin in rt.metal.

#define PI 3.14159265358979

struct Vertex { float px, py, pz, nx, ny, nz, u, v; };
struct InstInfo { uint first_index, base_vertex, flags, mesh; vec4 albedo; vec4 params; };
struct Xform { vec4 r0, r1, r2; };

#define FLAG_ALPHA 1u
#define FLAG_WATER 2u
#define FLAG_MOVING 4u
#define FLAG_TWOSIDED 8u
#define UBO_SHADOWS 1u
#define UBO_ALPHA 2u
#define UBO_COUNT 4u

#define FRAME_UBO_BODY \
   mat4 view_proj; mat4 inv_view_proj; vec4 cam_pos; vec4 sun_dir; vec4 params; uvec4 ip; uvec4 ip2;

uint pcg(uint v)
{
   uint state = v * 747796405u + 2891336453u;
   uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
   return (word >> 22u) ^ word;
}

float u01(inout uint s)
{
   s = pcg(s);
   return float(s >> 8) * (1.0 / 16777216.0);
}

vec3 rand_dir(inout uint s)
{
   float z = u01(s) * 2.0 - 1.0;
   float a = u01(s) * (2.0 * PI);
   float r = sqrt(max(0.0, 1.0 - z * z));
   return vec3(r * cos(a), r * sin(a), z);
}

vec3 sky(vec3 d, vec3 sun)
{
   float up = clamp(d.y, 0.0, 1.0);
   vec3 c = mix(vec3(0.62, 0.70, 0.82), vec3(0.16, 0.32, 0.62), sqrt(up));
   if (d.y < 0.0) c = vec3(0.10, 0.12, 0.15);
   float s = max(dot(d, sun), 0.0);
   return c + vec3(8.0, 7.0, 5.0) * pow(s, 256.0);
}

vec3 surface_albedo(InstInfo ii, vec2 uv)
{
   uint pat = uint(ii.params.z);
   float k = 1.0;
   if (pat == 1u) {
      vec2 c = floor(uv * 8.0);
      float m = c.x + c.y;
      k = (m - 2.0 * floor(m * 0.5)) < 0.5 ? 1.0 : 0.72;
   } else if (pat == 2u) {
      k = fract(uv.y * 12.0) < 0.5 ? 1.0 : 0.6;
   }
   return ii.albedo.rgb * k;
}

bool alpha_hole(vec2 uv)
{
   vec2 f = fract(uv * vec2(6.0, 10.0)) - 0.5;
   return dot(f, f) < 0.09;
}

vec3 xform_point(Xform t, vec3 p)
{
   vec4 q = vec4(p, 1.0);
   return vec3(dot(t.r0, q), dot(t.r1, q), dot(t.r2, q));
}

vec3 xform_dir(Xform t, vec3 d)
{
   return vec3(dot(t.r0.xyz, d), dot(t.r1.xyz, d), dot(t.r2.xyz, d));
}
