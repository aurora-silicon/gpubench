// SPDX-License-Identifier: GPL-2.0-only
#include "trav.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float *tri(float *o, float ax, float ay, float bx, float by, float cx, float cy, float z)
{
   float v[9] = {ax, ay, z, bx, by, z, cx, cy, z};
   memcpy(o, v, sizeof(v));
   return o + 9;
}

int rt_trav_geometry(const rt_params *p, float **pos, uint32_t *ntris, rt_trav_rays *r)
{
   const uint32_t n = p->tris ? p->tris : 1;
   memset(r, 0, sizeof(*r));
   r->nrays = p->rays;
   r->width = 512;
   uint32_t h = (r->nrays + r->width - 1) / r->width;
   r->tmax = 1000.0f;
   float *o;
   if (!strcmp(p->kind, "miss") || !strcmp(p->kind, "single")) {
      int miss = !strcmp(p->kind, "miss");
      *ntris = 1;
      o = *pos = malloc(9 * sizeof(float));
      if (miss) tri(o, 0, 0, 1, 0, 0, 1, 0);
      else tri(o, -4, -4, 12, -4, -4, 12, 0);
      r->x0 = miss ? 10.0f : 0.0f;
      r->dx = 1.0f / r->width;
      r->y0 = 0.0f;
      r->dy = 1.0f / h;
      return 0;
   }
   if (!strcmp(p->kind, "comb")) {
      /* lower-right halves of [-1,1]^2; the rays pass through the upper-left halves */
      *ntris = n;
      o = *pos = malloc((size_t)n * 9 * sizeof(float));
      for (uint32_t k = 0; k < n; k++) o = tri(o, -1, -1, 1, -1, 1, 1, (float)(k + 1));
      r->x0 = -0.9f;
      r->dx = 0.8f / r->width;
      r->y0 = 0.1f;
      r->dy = 0.8f / h;
      r->tmax = (float)n + 10.0f;
      return 0;
   }
   if (!strcmp(p->kind, "grid")) {
      uint32_t g = (uint32_t)ceil(sqrt((double)n / 2.0));
      if (!g) g = 1;
      float c = 1.0f / g;
      *ntris = 2 * g * g;
      o = *pos = malloc((size_t)*ntris * 9 * sizeof(float));
      for (uint32_t j = 0; j < g; j++)
         for (uint32_t i = 0; i < g; i++) {
            float x = i * c, y = j * c;
            o = tri(o, x, y, x + c, y, x + c, y + c, 0);
            o = tri(o, x, y, x + c, y + c, x, y + c, 0);
         }
      r->x0 = 0.0f;
      r->dx = 1.0f / r->width;
      r->y0 = 0.0f;
      r->dy = 1.0f / h;
      return 0;
   }
   if (!strcmp(p->kind, "gap")) {
      /* boxes x in [eps, 1] and [-1, -eps] on every plane; every ray runs along x = 0 */
      float e = p->eps;
      *ntris = 2 * n;
      o = *pos = malloc((size_t)*ntris * 9 * sizeof(float));
      for (uint32_t k = 0; k < n; k++) {
         float z = (float)(k + 1);
         o = tri(o, e, -1, 1, -1, 1, 1, z);
         o = tri(o, -1, -1, -e, -1, -e, 1, z);
      }
      r->width = 1;
      r->x0 = 0.0f;
      r->dx = 0.0f;
      r->y0 = -0.9f;
      r->dy = 1.8f / r->nrays;
      r->tmax = (float)n + 10.0f;
      return 0;
   }
   *pos = NULL;
   *ntris = 0;
   return -1;
}
