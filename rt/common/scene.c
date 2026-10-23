// SPDX-License-Identifier: GPL-2.0-only
#include "scene.h"
#include "mathx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979f

/* Water slot s covers cell water_cell[s] of the 4x4 grid (the four centre cells first, so
 * --dyn-patches 4 animates the middle of the bay); the table lives in the deform shaders. */

static uint32_t hash32(uint32_t x)
{
   x ^= x >> 16; x *= 0x7feb352dU;
   x ^= x >> 15; x *= 0x846ca68bU;
   x ^= x >> 16;
   return x;
}
static float hash01(uint32_t x) { return (hash32(x) >> 8) * (1.0f / 16777216.0f); }

typedef struct { uint64_t s; } lcg;
static float lcg01(lcg *g)
{
   g->s = g->s * 6364136223846793005ULL + 1442695040888963407ULL;
   return (float)((g->s >> 40) & 0xffffff) * (1.0f / 16777216.0f);
}

typedef struct {
   rt_vertex *v;
   uint32_t nv, cv;
   uint32_t *i;
   uint32_t ni, ci;
} builder;

static uint32_t addv(builder *b, v3 p, v3 n, float u, float vv)
{
   if (b->nv == b->cv) {
      b->cv = b->cv ? b->cv * 2 : 65536;
      b->v = realloc(b->v, b->cv * sizeof(rt_vertex));
   }
   rt_vertex *o = &b->v[b->nv];
   o->px = p.x; o->py = p.y; o->pz = p.z;
   o->nx = n.x; o->ny = n.y; o->nz = n.z;
   o->u = u; o->v = vv;
   return b->nv++;
}

static void addi(builder *b, uint32_t x)
{
   if (b->ni == b->ci) {
      b->ci = b->ci ? b->ci * 2 : 65536 * 3;
      b->i = realloc(b->i, b->ci * sizeof(uint32_t));
   }
   b->i[b->ni++] = x;
}

static v3 vpos(const builder *b, uint32_t k) { return v3m(b->v[k].px, b->v[k].py, b->v[k].pz); }
static v3 vnrm(const builder *b, uint32_t k) { return v3m(b->v[k].nx, b->v[k].ny, b->v[k].nz); }

/* Adds triangle (a, b, c) wound counter-clockwise around the vertex normals; drops degenerate ones. */
static void tri(builder *b, uint32_t a, uint32_t c1, uint32_t c2)
{
   v3 pa = vpos(b, a), pb = vpos(b, c1), pc = vpos(b, c2);
   v3 fn = v3cross(v3sub(pb, pa), v3sub(pc, pa));
   if (v3dot(fn, fn) < 1e-14f)
      return;
   v3 vn = v3add(v3add(vnrm(b, a), vnrm(b, c1)), vnrm(b, c2));
   addi(b, a);
   if (v3dot(fn, vn) >= 0) { addi(b, c1); addi(b, c2); }
   else { addi(b, c2); addi(b, c1); }
}

/* Triangulates a (cols+1) x (rows+1) vertex grid starting at vertex base. */
static void grid(builder *b, uint32_t base, uint32_t cols, uint32_t rows)
{
   for (uint32_t j = 0; j < rows; j++)
      for (uint32_t i = 0; i < cols; i++) {
         uint32_t a = base + j * (cols + 1) + i, bb = a + 1, d = a + cols + 1, c = d + 1;
         tri(b, a, c, bb);
         tri(b, a, d, c);
      }
}

/* Recomputes smooth normals for vertices [v0, nv) from the triangles in [i0, ni). */
static void smooth_normals(builder *b, uint32_t v0, uint32_t i0)
{
   for (uint32_t k = v0; k < b->nv; k++) b->v[k].nx = b->v[k].ny = b->v[k].nz = 0;
   for (uint32_t t = i0; t + 2 < b->ni; t += 3) {
      uint32_t a = b->i[t], c1 = b->i[t + 1], c2 = b->i[t + 2];
      v3 fn = v3cross(v3sub(vpos(b, c1), vpos(b, a)), v3sub(vpos(b, c2), vpos(b, a)));
      uint32_t idx[3] = {a, c1, c2};
      for (int q = 0; q < 3; q++) {
         rt_vertex *o = &b->v[idx[q]];
         o->nx += fn.x; o->ny += fn.y; o->nz += fn.z;
      }
   }
   for (uint32_t k = v0; k < b->nv; k++) {
      v3 n = v3norm(vnrm(b, k));
      b->v[k].nx = n.x; b->v[k].ny = n.y; b->v[k].nz = n.z;
   }
}

static uint32_t isqrt_at_least(double x, uint32_t lo)
{
   double r = sqrt(x);
   uint32_t v = (uint32_t)(r + 0.5);
   return v < lo ? lo : v;
}

static void gen_sphere(builder *b, uint32_t T, int rock, uint32_t variant)
{
   uint32_t rings = isqrt_at_least(T / 4.0, 4), segs = rings * 2;
   uint32_t v0 = b->nv, i0 = b->ni;
   float f1 = 2.0f + hash01(variant * 3 + 1) * 2.0f, f2 = 2.0f + hash01(variant * 3 + 2) * 3.0f;
   for (uint32_t j = 0; j <= rings; j++)
      for (uint32_t i = 0; i <= segs; i++) {
         float th = PI * j / rings, ph = 2 * PI * i / segs;
         v3 d = v3m(sinf(th) * cosf(ph), cosf(th), sinf(th) * sinf(ph));
         float r = 1.0f;
         if (rock)
            r = 1.0f + 0.18f * sinf(f1 * d.x + 1.7f) * sinf(f2 * d.y + 0.4f) * sinf(2.9f * d.z + 2.2f) * 2.0f +
                0.06f * sinf(7.0f * d.x + 5.0f * d.z);
         addv(b, v3scale(d, r), d, (float)i / segs, (float)j / rings);
      }
   grid(b, v0, segs, rings);
   if (rock) smooth_normals(b, v0, i0);
}

static void gen_tower(builder *b, uint32_t T, uint32_t variant)
{
   uint32_t n = isqrt_at_least(T / 2.0, 8), h = (T > n) ? (T - n) / (2 * n) : 1;
   if (h < 1) h = 1;
   float H = 5.0f + 3.0f * hash01(variant * 5 + 7), r0 = 0.7f, r1 = 0.35f + 0.25f * hash01(variant * 5 + 8);
   uint32_t v0 = b->nv;
   for (uint32_t j = 0; j <= h; j++)
      for (uint32_t i = 0; i <= n; i++) {
         float t = (float)j / h, ph = 2 * PI * i / n;
         float r = r0 + (r1 - r0) * t;
         v3 d = v3m(cosf(ph), 0, sinf(ph));
         v3 nn = v3norm(v3m(d.x, (r0 - r1) / H, d.z));
         addv(b, v3m(d.x * r, t * H, d.z * r), nn, (float)i / n, t);
      }
   grid(b, v0, n, h);
   uint32_t c = addv(b, v3m(0, H, 0), v3m(0, 1, 0), 0.5f, 1.0f);
   uint32_t ring0 = b->nv;
   for (uint32_t i = 0; i <= n; i++) {
      float ph = 2 * PI * i / n;
      addv(b, v3m(cosf(ph) * r1, H, sinf(ph) * r1), v3m(0, 1, 0), (float)i / n, 1.0f);
   }
   for (uint32_t i = 0; i < n; i++) tri(b, c, ring0 + i, ring0 + i + 1);
}

static void gen_torus(builder *b, uint32_t T)
{
   uint32_t bs = isqrt_at_least(T / 8.0, 6), as = T / (2 * bs);
   if (as < 8) as = 8;
   float R = 1.4f, r = 0.35f;
   uint32_t v0 = b->nv;
   for (uint32_t j = 0; j <= bs; j++)
      for (uint32_t i = 0; i <= as; i++) {
         float u = 2 * PI * i / as, v = 2 * PI * j / bs;
         v3 c = v3m(cosf(u) * R, sinf(u) * R, 0);
         v3 n = v3m(cosf(u) * cosf(v), sinf(u) * cosf(v), sinf(v));
         addv(b, v3add(c, v3scale(n, r)), n, (float)i / as * 4.0f, (float)j / bs);
      }
   grid(b, v0, as, bs);
}

static void gen_sail(builder *b, uint32_t T)
{
   uint32_t cols = isqrt_at_least(T / 2.0 * 0.6, 2), rows = T / (2 * cols);
   if (rows < 2) rows = 2;
   float W = 3.0f, H = 5.0f, bend = 0.6f;
   uint32_t v0 = b->nv, i0 = b->ni;
   for (uint32_t j = 0; j <= rows; j++)
      for (uint32_t i = 0; i <= cols; i++) {
         float u = (float)i / cols, v = (float)j / rows;
         float x = (u - 0.5f) * W;
         float z = bend * (1.0f - (2 * u - 1) * (2 * u - 1)) * (0.6f + 0.4f * v);
         addv(b, v3m(x, v * H, z), v3m(0, 0, -1), u, v);
      }
   grid(b, v0, cols, rows);
   smooth_normals(b, v0, i0);
}

void rt_scene_build(rt_scene *s, const rt_params *p)
{
   memset(s, 0, sizeof(*s));
   builder b = {0};
   s->water_size = 160.0f;
   s->water_grid = isqrt_at_least(p->water_tris / (double)RT_WATER_PATCHES / 2.0, 2);
   s->dyn_patches = p->blas_mode == RT_AS_STATIC ? 0 : p->dyn_patches;
   s->nmeshes = RT_WATER_PATCHES + p->meshes;
   s->meshes = calloc(s->nmeshes, sizeof(rt_mesh));

   /* Water slots: flat grids; the deform shader writes the real positions and normals. */
   uint32_t g = s->water_grid;
   s->water_vbase = 0;
   for (uint32_t slot = 0; slot < RT_WATER_PATCHES; slot++) {
      rt_mesh *m = &s->meshes[slot];
      m->first_index = b.ni;
      m->base_vertex = b.nv;
      for (uint32_t j = 0; j <= g; j++)
         for (uint32_t i = 0; i <= g; i++)
            addv(&b, v3m((float)i, 0, (float)j), v3m(0, 1, 0), 0, 0);
      /* local indices: base_vertex is applied by the draw and by the BLAS geometry offset */
      uint32_t i0 = b.ni;
      grid(&b, m->base_vertex, g, g);
      for (uint32_t k = i0; k < b.ni; k++) b.i[k] -= m->base_vertex;
      m->index_count = b.ni - m->first_index;
      m->vertex_count = b.nv - m->base_vertex;
      m->water = 1;
      m->dynamic = slot < s->dyn_patches;
   }
   for (uint32_t k = 0; k < p->meshes; k++) {
      rt_mesh *m = &s->meshes[RT_WATER_PATCHES + k];
      m->kind = k % RT_KINDS;
      m->first_index = b.ni;
      m->base_vertex = b.nv;
      uint32_t i0 = b.ni;
      switch (m->kind) {
      case RT_KIND_SPHERE: gen_sphere(&b, p->mesh_tris, 0, k); break;
      case RT_KIND_TOWER: gen_tower(&b, p->mesh_tris, k); break;
      case RT_KIND_TORUS: gen_torus(&b, p->mesh_tris); break;
      case RT_KIND_ROCK: gen_sphere(&b, p->mesh_tris, 1, k); break;
      case RT_KIND_SAIL: gen_sail(&b, p->mesh_tris); break;
      }
      for (uint32_t q = i0; q < b.ni; q++) b.i[q] -= m->base_vertex;
      m->index_count = b.ni - m->first_index;
      m->vertex_count = b.nv - m->base_vertex;
      m->alpha = m->kind == RT_KIND_SAIL && p->alpha;
      m->twosided = m->kind == RT_KIND_SAIL;
   }
   s->verts = b.v;
   s->nverts = b.nv;
   s->indices = b.i;
   s->nindices = b.ni;

   /* Instances: water slots, then objects grouped by mesh. */
   s->ninst = RT_WATER_PATCHES + p->instances;
   s->inst = calloc(s->ninst, sizeof(rt_instance));
   s->info = calloc(s->ninst, sizeof(rt_instinfo));
   s->draws = calloc(s->nmeshes, sizeof(rt_draw));
   lcg rng = {p->seed * 2654435761ULL + 12345};
   for (uint32_t slot = 0; slot < RT_WATER_PATCHES; slot++) {
      rt_instance *in = &s->inst[slot];
      rt_instinfo *ii = &s->info[slot];
      in->mesh = slot;
      in->flags = RT_FLAG_WATER;
      in->scale = 1;
      int refl = hash01(p->seed + slot * 13 + 5) < p->reflective;
      ii->flags = RT_FLAG_WATER;
      if (refl) {
         ii->albedo[0] = 0.02f; ii->albedo[1] = 0.07f; ii->albedo[2] = 0.09f;
         ii->params[0] = 0.75f; ii->params[1] = 0.02f;
      } else {
         ii->albedo[0] = 0.05f; ii->albedo[1] = 0.22f; ii->albedo[2] = 0.27f;
         ii->params[0] = 0.0f; ii->params[1] = 1.0f;
      }
      ii->albedo[3] = 1;
      ii->params[2] = 0;
      s->draws[s->ndraws++] = (rt_draw){slot, slot, 1, 0};
   }
   static const float palette[RT_KINDS][3] = {
      {0.85f, 0.35f, 0.20f}, {0.75f, 0.75f, 0.72f}, {0.95f, 0.75f, 0.20f}, {0.45f, 0.40f, 0.36f}, {0.90f, 0.90f, 0.85f}};
   uint32_t ii_ = RT_WATER_PATCHES;
   float rmin = 7.0f, rmax = 0.42f * s->water_size;
   for (uint32_t k = 0; k < p->meshes; k++) {
      uint32_t count = p->instances / p->meshes + (k < p->instances % p->meshes ? 1 : 0);
      if (!count) continue;
      rt_mesh *m = &s->meshes[RT_WATER_PATCHES + k];
      s->draws[s->ndraws++] = (rt_draw){RT_WATER_PATCHES + k, ii_, count, m->twosided};
      for (uint32_t c = 0; c < count; c++, ii_++) {
         rt_instance *in = &s->inst[ii_];
         rt_instinfo *inf = &s->info[ii_];
         in->mesh = RT_WATER_PATCHES + k;
         float rr = sqrtf(rmin * rmin + (rmax * rmax - rmin * rmin) * lcg01(&rng));
         float an = 2 * PI * lcg01(&rng);
         float sc = 0.8f + 1.0f * lcg01(&rng);
         float yaw = 2 * PI * lcg01(&rng);
         float y = 0;
         switch (m->kind) {
         case RT_KIND_SPHERE: y = 1.0f + 2.5f * lcg01(&rng); sc *= 1.2f; break;
         case RT_KIND_TOWER: y = -0.6f; break;
         case RT_KIND_TORUS: y = 1.2f * sc; break;
         case RT_KIND_ROCK: y = -0.2f; sc *= 1.4f; break;
         case RT_KIND_SAIL: y = 0.1f; break;
         }
         in->pos[0] = cosf(an) * rr; in->pos[1] = y; in->pos[2] = sinf(an) * rr;
         in->scale = sc;
         in->yaw = yaw;
         in->phase = 2 * PI * lcg01(&rng);
         float sp = 0.3f + 0.7f * lcg01(&rng);
         in->spin = lcg01(&rng) < 0.5f ? -sp : sp;
         in->bob = 0.6f;
         float hm = lcg01(&rng);
         if (hm < p->moving) { in->flags |= RT_FLAG_MOVING; s->nmoving++; }
         uint32_t id = ii_;
         int refl = hash01(p->seed * 7 + id * 7 + 1) < p->reflective;
         float var = 0.75f + 0.25f * hash01(id * 11 + 3);
         inf->albedo[0] = palette[m->kind][0] * var;
         inf->albedo[1] = palette[m->kind][1] * var;
         inf->albedo[2] = palette[m->kind][2] * var;
         inf->albedo[3] = 1;
         inf->flags = in->flags | (m->alpha ? RT_FLAG_ALPHA : 0) | (m->twosided ? RT_FLAG_TWOSIDED : 0);
         if (refl) {
            inf->params[0] = 0.45f + 0.45f * hash01(id * 17 + 9);
            inf->params[1] = 0.12f * hash01(id * 19 + 4);
         } else {
            inf->params[0] = 0; inf->params[1] = 1;
         }
         inf->params[2] = (float)(m->kind == RT_KIND_TOWER ? 2 : m->kind == RT_KIND_SAIL ? 0 : 1);
      }
   }
   for (uint32_t i = 0; i < s->ninst; i++) {
      rt_mesh *m = &s->meshes[s->inst[i].mesh];
      s->info[i].first_index = m->first_index;
      s->info[i].base_vertex = m->base_vertex;
      s->info[i].mesh = s->inst[i].mesh;
      s->instanced_tris += m->index_count / 3;
   }
   for (uint32_t k = 0; k < s->nmeshes; k++) s->unique_tris += s->meshes[k].index_count / 3;
}

void rt_scene_free(rt_scene *s)
{
   free(s->verts); free(s->indices); free(s->meshes); free(s->inst); free(s->info); free(s->draws);
   memset(s, 0, sizeof(*s));
}

void rt_scene_xform(const rt_scene *s, uint32_t i, float t, rt_xform *o)
{
   const rt_instance *in = &s->inst[i];
   memset(o, 0, sizeof(*o));
   if (in->flags & RT_FLAG_WATER) {
      o->r[0] = o->r[5] = o->r[10] = 1;
      return;
   }
   float yaw = in->yaw, y = in->pos[1];
   if (in->flags & RT_FLAG_MOVING) {
      yaw += t * in->spin;
      y += in->bob * sinf(t * 1.7f + in->phase);
   }
   float c = cosf(yaw) * in->scale, sn = sinf(yaw) * in->scale;
   /* rotation about Y, uniform scale */
   o->r[0] = c;   o->r[1] = 0;         o->r[2] = sn;  o->r[3] = in->pos[0];
   o->r[4] = 0;   o->r[5] = in->scale; o->r[6] = 0;   o->r[7] = y;
   o->r[8] = -sn; o->r[9] = 0;         o->r[10] = c;  o->r[11] = in->pos[2];
}

static void camera(const rt_scene *s, uint32_t frame, v3 *eye, v3 *at)
{
   float t = rt_frame_time(frame);
   float ang = 0.6f + 0.05f * t, rad = 0.30f * s->water_size;
   *eye = v3m(cosf(ang) * rad, 9.0f + 1.5f * sinf(0.21f * t), sinf(ang) * rad);
   *at = v3m(-cosf(ang) * 8.0f, 3.0f, -sinf(ang) * 8.0f);
}

#define FOVY (55.0f * PI / 180.0f)

void rt_scene_camera(const rt_scene *s, uint32_t frame, float aspect, float eye[3], float fwd[3], float right[3],
                     float up[3])
{
   v3 e, at;
   camera(s, frame, &e, &at);
   v3 f = v3norm(v3sub(at, e));
   v3 r = v3norm(v3cross(f, v3m(0, 1, 0)));
   v3 u = v3cross(r, f);
   float th = tanf(FOVY * 0.5f);
   r = v3scale(r, th * aspect);
   u = v3scale(u, th);
   eye[0] = e.x; eye[1] = e.y; eye[2] = e.z;
   fwd[0] = f.x; fwd[1] = f.y; fwd[2] = f.z;
   right[0] = r.x; right[1] = r.y; right[2] = r.z;
   up[0] = u.x; up[1] = u.y; up[2] = u.z;
}

void rt_scene_frame_ubo(const rt_scene *s, const rt_params *p, uint32_t frame, int flip_y, int count_rays,
                        rt_frame_ubo *u)
{
   memset(u, 0, sizeof(*u));
   float t = rt_frame_time(frame);
   v3 eye, at;
   /* A fixed camera keeps the work per frame the same however many frames a device renders in a
    * window; water and moving objects still animate with the frame index. */
   camera(s, p->camera_orbit ? frame : 0, &eye, &at);
   float view[16], proj[16];
   m4lookat(view, eye, at, v3m(0, 1, 0));
   m4persp_revz(proj, FOVY, (float)p->width / (float)p->height, 0.1f, flip_y);
   m4mul(u->view_proj, proj, view);
   m4invert(u->inv_view_proj, u->view_proj);
   u->cam_pos[0] = eye.x; u->cam_pos[1] = eye.y; u->cam_pos[2] = eye.z; u->cam_pos[3] = t;
   v3 sun = v3norm(v3m(0.45f, 0.42f, 0.30f));
   u->sun_dir[0] = sun.x; u->sun_dir[1] = sun.y; u->sun_dir[2] = sun.z; u->sun_dir[3] = 3.0f;
   u->params[0] = p->tmax;
   u->params[1] = 1e-3f;
   u->params[2] = 0.22f;
   u->ip[0] = p->width; u->ip[1] = p->height; u->ip[2] = frame;
   u->ip[3] = (p->shadows ? RT_UBO_SHADOWS : 0) | (p->alpha ? RT_UBO_ALPHA : 0) | (count_rays ? RT_UBO_COUNT : 0);
   u->ip2[0] = p->rpp; u->ip2[1] = p->bounces; u->ip2[2] = s->ninst;
}

void rt_gen_mesh(int kind, uint32_t tris, rt_vertex **v, uint32_t *nv, uint32_t **idx, uint32_t *ni)
{
   builder b = {0};
   switch (kind) {
   case RT_KIND_SPHERE: gen_sphere(&b, tris, 0, 0); break;
   case RT_KIND_TOWER: gen_tower(&b, tris, 0); break;
   case RT_KIND_TORUS: gen_torus(&b, tris); break;
   case RT_KIND_ROCK: gen_sphere(&b, tris, 1, 0); break;
   default: gen_sail(&b, tris); break;
   }
   *v = b.v; *nv = b.nv; *idx = b.i; *ni = b.ni;
}
