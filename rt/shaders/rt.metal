// SPDX-License-Identifier: GPL-2.0-only
// rtbench Metal shaders: line-for-line twins of the GLSL in this directory (common.glsl,
// deform.comp, gbuffer.vert/.frag, rtres.glsl + rtshade.glsl + rtreflect.comp, composite.comp,
// rays.comp). Layouts mirror rt/common/scene.h. Ray queries use metal::raytracing
// intersection_query, the counterpart of GL_EXT_ray_query.
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

#define PI 3.14159265358979f

struct Vertex { float px, py, pz, nx, ny, nz, u, v; };
struct InstInfo { uint first_index, base_vertex, flags, mesh; float4 albedo; float4 params; };
struct Xform { float4 r0, r1, r2; };
struct Frame {
   float4x4 view_proj;
   float4x4 inv_view_proj;
   float4 cam_pos;
   float4 sun_dir;
   float4 params;
   uint4 ip;
   uint4 ip2;
};

#define FLAG_ALPHA 1u
#define FLAG_WATER 2u
#define FLAG_MOVING 4u
#define FLAG_TWOSIDED 8u
#define UBO_SHADOWS 1u
#define UBO_ALPHA 2u
#define UBO_COUNT 4u

static uint pcg(uint v)
{
   uint state = v * 747796405u + 2891336453u;
   uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
   return (word >> 22u) ^ word;
}

static float u01(thread uint &s)
{
   s = pcg(s);
   return float(s >> 8) * (1.0f / 16777216.0f);
}

static float3 rand_dir(thread uint &s)
{
   float z = u01(s) * 2.0f - 1.0f;
   float a = u01(s) * (2.0f * PI);
   float r = sqrt(max(0.0f, 1.0f - z * z));
   return float3(r * cos(a), r * sin(a), z);
}

static float3 sky(float3 d, float3 sun)
{
   float up = clamp(d.y, 0.0f, 1.0f);
   float3 c = mix(float3(0.62f, 0.70f, 0.82f), float3(0.16f, 0.32f, 0.62f), sqrt(up));
   if (d.y < 0.0f) c = float3(0.10f, 0.12f, 0.15f);
   float s = max(dot(d, sun), 0.0f);
   return c + float3(8.0f, 7.0f, 5.0f) * pow(s, 256.0f);
}

static float3 surface_albedo(InstInfo ii, float2 uv)
{
   uint pat = uint(ii.params.z);
   float k = 1.0f;
   if (pat == 1u) {
      float2 c = floor(uv * 8.0f);
      float m = c.x + c.y;
      k = (m - 2.0f * floor(m * 0.5f)) < 0.5f ? 1.0f : 0.72f;
   } else if (pat == 2u) {
      k = fract(uv.y * 12.0f) < 0.5f ? 1.0f : 0.6f;
   }
   return ii.albedo.rgb * k;
}

static bool alpha_hole(float2 uv)
{
   float2 f = fract(uv * float2(6.0f, 10.0f)) - 0.5f;
   return dot(f, f) < 0.09f;
}

static float3 xform_point(Xform t, float3 p)
{
   float4 q = float4(p, 1.0f);
   return float3(dot(t.r0, q), dot(t.r1, q), dot(t.r2, q));
}

static float3 xform_dir(Xform t, float3 d)
{
   return float3(dot(t.r0.xyz, d), dot(t.r1.xyz, d), dot(t.r2.xyz, d));
}

// ---- deform.comp ----

struct DeformPC { uint vbase; uint grid; uint first_slot; uint nslots; float size; float time; float pad0, pad1; };

constant uint water_cell[16] = {5u, 6u, 9u, 10u, 1u, 2u, 4u, 7u, 8u, 11u, 13u, 14u, 0u, 3u, 12u, 15u};

kernel void deform(device Vertex *verts [[buffer(0)]], constant DeformPC &pc [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]])
{
   uint g1 = pc.grid + 1u;
   uint vpp = g1 * g1;
   uint id = gid.x;
   uint slot = pc.first_slot + gid.y;
   if (id >= vpp) return;
   uint cell = water_cell[slot];
   uint i = id % g1, j = id / g1;
   float ps = pc.size * 0.25f;
   float fu = float(i) / float(pc.grid), fv = float(j) / float(pc.grid);
   float x = -0.5f * pc.size + float(cell % 4u) * ps + fu * ps;
   float z = -0.5f * pc.size + float(cell / 4u) * ps + fv * ps;
   float t = pc.time;

   float h0 = 0.12f * sin(0.31f * x + 0.7f) * sin(0.27f * z + 0.3f);
   float h0x = 0.12f * 0.31f * cos(0.31f * x + 0.7f) * sin(0.27f * z + 0.3f);
   float h0z = 0.12f * 0.27f * sin(0.31f * x + 0.7f) * cos(0.27f * z + 0.3f);
   float hd = 0.35f * sin(0.23f * x + 1.3f * t) + 0.25f * sin(0.19f * z - 1.1f * t + 0.5f) +
              0.12f * sin(0.41f * (x + z) + 2.1f * t);
   float hdx = 0.35f * 0.23f * cos(0.23f * x + 1.3f * t) + 0.12f * 0.41f * cos(0.41f * (x + z) + 2.1f * t);
   float hdz = 0.25f * 0.19f * cos(0.19f * z - 1.1f * t + 0.5f) + 0.12f * 0.41f * cos(0.41f * (x + z) + 2.1f * t);
   float w = sin(PI * fu) * sin(PI * fv);
   float wx = PI / ps * cos(PI * fu) * sin(PI * fv);
   float wz = PI / ps * sin(PI * fu) * cos(PI * fv);
   float h = h0 + w * hd;
   float hx = h0x + w * hdx + wx * hd;
   float hz = h0z + w * hdz + wz * hd;
   float3 n = normalize(float3(-hx, 1.0f, -hz));
   Vertex o;
   o.px = x; o.py = h; o.pz = z;
   o.nx = n.x; o.ny = n.y; o.nz = n.z;
   o.u = x * 0.1f; o.v = z * 0.1f;
   verts[pc.vbase + slot * vpp + id] = o;
}

// ---- gbuffer.vert / gbuffer.frag ----

struct VSIn {
   float3 pos [[attribute(0)]];
   float3 nrm [[attribute(1)]];
   float2 uv [[attribute(2)]];
};
struct VSOut {
   float4 pos [[position]];
   float3 wpos;
   float3 nrm;
   float2 uv;
   uint inst [[flat]];
};

vertex VSOut gbuffer_vs(VSIn in [[stage_in]], uint iid [[instance_id]], constant Frame &fr [[buffer(1)]],
                        const device Xform *xf [[buffer(2)]])
{
   Xform t = xf[iid];
   float3 w = xform_point(t, in.pos);
   VSOut o;
   o.pos = fr.view_proj * float4(w, 1.0f);
   o.wpos = w;
   o.nrm = xform_dir(t, in.nrm);
   o.uv = in.uv;
   o.inst = iid;
   return o;
}

constant bool ALPHA [[function_constant(0)]];

struct GBOut {
   float4 albedo [[color(0)]];
   float4 normal [[color(1)]];
};

fragment GBOut gbuffer_fs(VSOut in [[stage_in]], constant Frame &fr [[buffer(1)]],
                          const device InstInfo *info [[buffer(3)]])
{
   InstInfo ii = info[in.inst];
   if (ALPHA && alpha_hole(in.uv)) discard_fragment();
   float3 n = normalize(in.nrm);
   if ((ii.flags & FLAG_TWOSIDED) != 0u && dot(n, fr.cam_pos.xyz - in.wpos) < 0.0f) n = -n;
   GBOut o;
   o.albedo = float4(surface_albedo(ii, in.uv), ii.params.y);
   o.normal = float4(n, ii.params.x);
   return o;
}

// ---- rtres.glsl + rtreflect.comp + rtshade.glsl ----

struct RtRes {
   constant Frame *fr;
   const device Vertex *verts;
   const device uint *indices;
   const device InstInfo *info;
   const device Xform *xf;
};

static float2 hit_uv(thread const RtRes &R, InstInfo ii, uint prim, float2 b)
{
   uint base = ii.first_index + prim * 3u;
   Vertex v0 = R.verts[R.indices[base] + ii.base_vertex];
   Vertex v1 = R.verts[R.indices[base + 1u] + ii.base_vertex];
   Vertex v2 = R.verts[R.indices[base + 2u] + ii.base_vertex];
   float w = 1.0f - b.x - b.y;
   return float2(v0.u, v0.v) * w + float2(v1.u, v1.v) * b.x + float2(v2.u, v2.v) * b.y;
}

static bool candidate_opaque(thread const RtRes &R, uint inst, uint prim, float2 b)
{
   return !alpha_hole(hit_uv(R, R.info[inst], prim, b));
}

static bool trace_closest(instance_acceleration_structure tlas, thread const RtRes &R, float3 o, float3 d, float tmax,
                          thread float &t, thread uint &inst, thread uint &prim, thread float2 &bary)
{
   ray r(o, d, 0.0f, tmax);
   intersection_params params;
   intersection_query<triangle_data, instancing> q;
   q.reset(r, tlas, 0xFFu, params);
   while (q.next()) {
      if (q.get_candidate_intersection_type() == intersection_type::triangle) {
         if (candidate_opaque(R, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                              q.get_candidate_triangle_barycentric_coord()))
            q.commit_triangle_intersection();
      }
   }
   if (q.get_committed_intersection_type() == intersection_type::none)
      return false;
   t = q.get_committed_distance();
   inst = q.get_committed_instance_id();
   prim = q.get_committed_primitive_id();
   bary = q.get_committed_triangle_barycentric_coord();
   return true;
}

static bool trace_shadow(instance_acceleration_structure tlas, thread const RtRes &R, float3 o, float3 d, float tmax)
{
   ray r(o, d, 0.0f, tmax);
   intersection_params params;
   params.accept_any_intersection(true);
   intersection_query<triangle_data, instancing> q;
   q.reset(r, tlas, 0xFFu, params);
   while (q.next()) {
      if (q.get_candidate_intersection_type() == intersection_type::triangle) {
         if (candidate_opaque(R, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                              q.get_candidate_triangle_barycentric_coord()))
            q.commit_triangle_intersection();
      }
   }
   return q.get_committed_intersection_type() != intersection_type::none;
}

kernel void rt_reflect(instance_acceleration_structure tlas [[buffer(0)]], constant Frame &fr [[buffer(1)]],
                       texture2d<float, access::read> g_albedo [[texture(0)]],
                       texture2d<float, access::read> g_normal [[texture(1)]],
                       depth2d<float, access::read> g_depth [[texture(2)]],
                       const device Vertex *verts [[buffer(2)]], const device uint *indices [[buffer(3)]],
                       const device InstInfo *info [[buffer(4)]], const device Xform *xf [[buffer(5)]],
                       texture2d<float, access::write> o_rt [[texture(3)]],
                       device atomic_uint *counters [[buffer(6)]], uint2 px [[thread_position_in_grid]])
{
   RtRes R = {&fr, verts, indices, info, xf};
   uint W = fr.ip.x, H = fr.ip.y;
   if (px.x >= W || px.y >= H) return;
   float depth = g_depth.read(px);
   if (depth == 0.0f) {
      o_rt.write(float4(0.0f, 0.0f, 0.0f, 1.0f), px);
      return;
   }
   float4 nr = g_normal.read(px);
   float rough = g_albedo.read(px).w;
   float2 ndc = (float2(px) + 0.5f) / float2(float(W), float(H)) * 2.0f - 1.0f;
   ndc.y = -ndc.y;  // Metal NDC y points up, the framebuffer's y down
   float4 wp = fr.inv_view_proj * float4(ndc, depth, 1.0f);
   float3 P = wp.xyz / wp.w;
   float3 N = normalize(nr.xyz);
   float refl = nr.w;
   float3 cam = fr.cam_pos.xyz;
   float3 V = normalize(P - cam);
   float3 sun = fr.sun_dir.xyz;
   float tmax = fr.params.x;
   float eps = fr.params.y * max(1.0f, length(P - cam));
   float3 origin = P + N * eps;
   uint flags = fr.ip.w;
   uint n_shadow = 0u, n_refl = 0u, n_shadow2 = 0u;

   float shadow = 1.0f;
   if ((flags & UBO_SHADOWS) != 0u) {
      n_shadow++;
      shadow = trace_shadow(tlas, R, origin, sun, tmax) ? 0.0f : 1.0f;
   }

   float3 acc = float3(0.0f);
   if (refl > 0.0f) {
      uint rpp = fr.ip2.x, bounces = fr.ip2.y;
      for (uint s = 0u; s < rpp; s++) {
         uint seed = pcg(px.x ^ pcg(px.y ^ pcg(fr.ip.z * 64u + s)));
         float3 d = reflect(V, N);
         if (rough > 0.0f) d = normalize(d + rough * rand_dir(seed));
         if (dot(d, N) <= 0.0f) d = reflect(d, N);
         float3 o = origin;
         float3 thr = float3(1.0f);
         for (uint b = 0u; b < bounces; b++) {
            n_refl++;
            float t;
            uint inst, prim;
            float2 bary;
            if (!trace_closest(tlas, R, o, d, tmax, t, inst, prim, bary)) {
               acc += thr * sky(d, sun);
               break;
            }
            InstInfo ii = info[inst];
            Xform x = xf[inst];
            uint base = ii.first_index + prim * 3u;
            Vertex v0 = verts[indices[base] + ii.base_vertex];
            Vertex v1 = verts[indices[base + 1u] + ii.base_vertex];
            Vertex v2 = verts[indices[base + 2u] + ii.base_vertex];
            float w = 1.0f - bary.x - bary.y;
            float3 on = float3(v0.nx, v0.ny, v0.nz) * w + float3(v1.nx, v1.ny, v1.nz) * bary.x +
                        float3(v2.nx, v2.ny, v2.nz) * bary.y;
            float2 uv = float2(v0.u, v0.v) * w + float2(v1.u, v1.v) * bary.x + float2(v2.u, v2.v) * bary.y;
            float3 hn = normalize(xform_dir(x, on));
            if (dot(hn, d) > 0.0f) hn = -hn;
            float3 hp = o + d * t;
            float3 alb = surface_albedo(ii, uv);
            float hrefl = ii.params.x;
            float3 ho = hp + hn * (fr.params.y * max(1.0f, t));
            float sh = 1.0f;
            if ((flags & UBO_SHADOWS) != 0u) {
               n_shadow2++;
               sh = trace_shadow(tlas, R, ho, sun, tmax) ? 0.0f : 1.0f;
            }
            float3 direct = alb * (fr.params.z + fr.sun_dir.w * max(dot(hn, sun), 0.0f) * sh);
            acc += thr * direct * (1.0f - hrefl);
            if (hrefl <= 0.0f) break;
            thr *= hrefl;
            d = reflect(d, hn);
            o = ho;
            if (b + 1u == bounces) acc += thr * sky(d, sun);
         }
      }
      acc /= float(rpp);
   }
   o_rt.write(float4(acc, shadow), px);
   if ((flags & UBO_COUNT) != 0u) {
      atomic_fetch_add_explicit(&counters[0], n_shadow, memory_order_relaxed);
      atomic_fetch_add_explicit(&counters[1], n_refl, memory_order_relaxed);
      atomic_fetch_add_explicit(&counters[2], n_shadow2, memory_order_relaxed);
   }
}

// ---- composite.comp ----

kernel void composite(constant Frame &fr [[buffer(0)]], texture2d<float, access::read> g_albedo [[texture(0)]],
                      texture2d<float, access::read> g_normal [[texture(1)]],
                      depth2d<float, access::read> g_depth [[texture(2)]],
                      texture2d<float, access::read> g_rt [[texture(3)]],
                      texture2d<float, access::write> o_color [[texture(4)]], uint2 px [[thread_position_in_grid]])
{
   uint W = fr.ip.x, H = fr.ip.y;
   if (px.x >= W || px.y >= H) return;
   float depth = g_depth.read(px);
   float3 sun = fr.sun_dir.xyz;
   float3 c;
   float refl = 0.0f;
   if (depth == 0.0f) {
      float2 ndc = (float2(px) + 0.5f) / float2(float(W), float(H)) * 2.0f - 1.0f;
      ndc.y = -ndc.y;
      float4 wp = fr.inv_view_proj * float4(ndc, 0.5f, 1.0f);
      c = sky(normalize(wp.xyz / wp.w - fr.cam_pos.xyz), sun);
   } else {
      float4 ar = g_albedo.read(px);
      float4 nr = g_normal.read(px);
      float4 rt = g_rt.read(px);
      float3 N = normalize(nr.xyz);
      refl = nr.w;
      float3 direct = ar.rgb * (fr.params.z + fr.sun_dir.w * max(dot(N, sun), 0.0f) * rt.a);
      c = direct * (1.0f - refl) + rt.rgb * refl;
   }
   c = c / (1.0f + c);
   c = pow(c, float3(1.0f / 2.2f));
   o_color.write(float4(c, refl > 0.0f ? 1.0f : 0.0f), px);
}

// ---- rays.comp ----

struct RaysPC {
   float4 eye;
   float4 fwd;
   float4 right;
   float4 up;
   uint mode, nrays, seed, width;
};

kernel void rays(instance_acceleration_structure tlas [[buffer(0)]], device float *result [[buffer(1)]],
                 constant RaysPC &pc [[buffer(2)]], uint id [[thread_position_in_grid]])
{
   if (id >= pc.nrays) return;
   float3 o, d;
   if (pc.mode == 0u) {
      uint x = id % pc.width, y = id / pc.width;
      uint h = (pc.nrays + pc.width - 1u) / pc.width;
      float2 ndc = (float2(float(x), float(y)) + 0.5f) / float2(float(pc.width), float(h)) * 2.0f - 1.0f;
      o = pc.eye.xyz;
      d = normalize(pc.fwd.xyz + ndc.x * pc.right.xyz - ndc.y * pc.up.xyz);
   } else {
      uint s = pcg(id ^ pcg(pc.seed));
      o = float3(u01(s) * 120.0f - 60.0f, 0.5f + u01(s) * 8.0f, u01(s) * 120.0f - 60.0f);
      d = rand_dir(s);
   }
   ray r(o, d, 0.0f, pc.eye.w);
   intersection_params params;
   params.force_opacity(forced_opacity::opaque);
   intersection_query<triangle_data, instancing> q;
   q.reset(r, tlas, 0xFFu, params);
   while (q.next()) {
   }
   float t = -1.0f;
   if (q.get_committed_intersection_type() != intersection_type::none)
      t = q.get_committed_distance();
   result[id] = t;
}

/* rt.trav (common/trav.h): GLSL twin shaders/trav.comp. */
struct TravPC {
   float x0, y0, dx, dy;
   uint width, nrays;
   float tmax;
   uint pad;
};

kernel void trav(instance_acceleration_structure tlas [[buffer(0)]], device float *result [[buffer(1)]],
                 constant TravPC &pc [[buffer(2)]], uint id [[thread_position_in_grid]])
{
   if (id >= pc.nrays) return;
   float3 o = float3(pc.x0 + (float(id % pc.width) + 0.5f) * pc.dx, pc.y0 + (float(id / pc.width) + 0.5f) * pc.dy, -1.0f);
   ray r(o, float3(0.0f, 0.0f, 1.0f), 0.0f, pc.tmax);
   intersection_params params;
   params.force_opacity(forced_opacity::opaque);
   intersection_query<triangle_data, instancing> q;
   q.reset(r, tlas, 0xFFu, params);
   while (q.next()) {
   }
   float t = -1.0f;
   if (q.get_committed_intersection_type() != intersection_type::none)
      t = q.get_committed_distance();
   result[id] = t;
}
