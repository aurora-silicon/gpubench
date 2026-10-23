// SPDX-License-Identifier: GPL-2.0-only
// Per-pixel reflection + shadow shading shared by rtreflect.comp (ray queries) and rtpipe.rgen
// (ray-tracing pipeline). The includer defines the resources (rtres.glsl), trace_closest() and
// trace_shadow() before including this file. MSL twin: shade_pixel in rt.metal.

void shade_pixel(uvec2 px)
{
   uint W = fr.ip.x, H = fr.ip.y;
   if (px.x >= W || px.y >= H) return;
   float depth = texelFetch(g_depth, ivec2(px), 0).r;
   if (depth == 0.0) {
      imageStore(o_rt, ivec2(px), vec4(0.0, 0.0, 0.0, 1.0));
      return;
   }
   vec4 nr = texelFetch(g_normal, ivec2(px), 0);
   float rough = texelFetch(g_albedo, ivec2(px), 0).w;
   vec2 ndc = (vec2(px) + 0.5) / vec2(float(W), float(H)) * 2.0 - 1.0;
   vec4 wp = fr.inv_view_proj * vec4(ndc, depth, 1.0);
   vec3 P = wp.xyz / wp.w;
   vec3 N = normalize(nr.xyz);
   float refl = nr.w;
   vec3 cam = fr.cam_pos.xyz;
   vec3 V = normalize(P - cam);
   vec3 sun = fr.sun_dir.xyz;
   float tmax = fr.params.x;
   float eps = fr.params.y * max(1.0, length(P - cam));
   vec3 origin = P + N * eps;
   uint flags = fr.ip.w;
   uint n_shadow = 0u, n_refl = 0u, n_shadow2 = 0u;

   float shadow = 1.0;
   if ((flags & UBO_SHADOWS) != 0u) {
      n_shadow++;
      shadow = trace_shadow(origin, sun, tmax) ? 0.0 : 1.0;
   }

   vec3 acc = vec3(0.0);
   if (refl > 0.0) {
      uint rpp = fr.ip2.x, bounces = fr.ip2.y;
      for (uint s = 0u; s < rpp; s++) {
         uint seed = pcg(px.x ^ pcg(px.y ^ pcg(fr.ip.z * 64u + s)));
         vec3 d = reflect(V, N);
         if (rough > 0.0) d = normalize(d + rough * rand_dir(seed));
         if (dot(d, N) <= 0.0) d = reflect(d, N);
         vec3 o = origin;
         vec3 thr = vec3(1.0);
         for (uint b = 0u; b < bounces; b++) {
            n_refl++;
            float t;
            uint inst, prim;
            vec2 bary;
            if (!trace_closest(o, d, tmax, t, inst, prim, bary)) {
               acc += thr * sky(d, sun);
               break;
            }
            InstInfo ii = info[inst];
            Xform x = xf[inst];
            uint base = ii.first_index + prim * 3u;
            Vertex v0 = verts[indices[base] + ii.base_vertex];
            Vertex v1 = verts[indices[base + 1u] + ii.base_vertex];
            Vertex v2 = verts[indices[base + 2u] + ii.base_vertex];
            float w = 1.0 - bary.x - bary.y;
            vec3 on = vec3(v0.nx, v0.ny, v0.nz) * w + vec3(v1.nx, v1.ny, v1.nz) * bary.x +
                      vec3(v2.nx, v2.ny, v2.nz) * bary.y;
            vec2 uv = vec2(v0.u, v0.v) * w + vec2(v1.u, v1.v) * bary.x + vec2(v2.u, v2.v) * bary.y;
            vec3 hn = normalize(xform_dir(x, on));
            if (dot(hn, d) > 0.0) hn = -hn;
            vec3 hp = o + d * t;
            vec3 alb = surface_albedo(ii, uv);
            float hrefl = ii.params.x;
            vec3 ho = hp + hn * (fr.params.y * max(1.0, t));
            float sh = 1.0;
            if ((flags & UBO_SHADOWS) != 0u) {
               n_shadow2++;
               sh = trace_shadow(ho, sun, tmax) ? 0.0 : 1.0;
            }
            vec3 direct = alb * (fr.params.z + fr.sun_dir.w * max(dot(hn, sun), 0.0) * sh);
            acc += thr * direct * (1.0 - hrefl);
            if (hrefl <= 0.0) break;
            thr *= hrefl;
            d = reflect(d, hn);
            o = ho;
            if (b + 1u == bounces) acc += thr * sky(d, sun);
         }
      }
      acc /= float(rpp);
   }
   imageStore(o_rt, ivec2(px), vec4(acc, shadow));
   if ((flags & UBO_COUNT) != 0u) {
      atomicAdd(counters[0], n_shadow);
      atomicAdd(counters[1], n_refl);
      atomicAdd(counters[2], n_shadow2);
   }
}
