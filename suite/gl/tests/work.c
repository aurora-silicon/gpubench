// SPDX-License-Identifier: GPL-2.0-only
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gb_gl_work.h"
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "line %d: %s\n", __LINE__, #expr); exit(1); } } while (0)

static struct gb_opts
options(const char *test, const char *kind, const char *res)
{
   struct gb_opts o = {.test = test};
   if (kind) {
      strcpy(o.params[o.nparams].key, "kind");
      strcpy(o.params[o.nparams++].val, kind);
   }
   if (res) {
      strcpy(o.params[o.nparams].key, "res");
      strcpy(o.params[o.nparams++].val, res);
   }
   return o;
}

int main(void)
{
   struct gb_gl_work w; const char *err;
   struct gb_opts o = options("render.heavy", NULL, NULL);
   CHECK(gb_gl_work(&o, &w, &err));
   CHECK(w.kind == GB_GL_FILL && w.w == 1920 && w.h == 1080 && !w.compute);
   CHECK(w.work_e6 == 1920.0 * 1080 * GB_FILL_LAYERS / 1e3);
   const char *res[] = {"720p", "1080p", "1440p", "2160p"};
   const unsigned widths[] = {1280, 1920, 2560, 3840}, heights[] = {720, 1080, 1440, 2160};
   const char *kinds[] = {"fill", "opaque", "texture", "geom"};
   for (int i = 0; i < 4; i++) for (int k = 0; k < 4; k++) {
      o = options("render.heavy", kinds[k], res[i]);
      CHECK(gb_gl_work(&o, &w, &err));
      CHECK(w.w == widths[i] && w.h == heights[i]);
      CHECK(w.work_e6 == (k == 3 ? GB_GEOM_TRIS : (double)w.w * w.h * GB_FILL_LAYERS / 1e3));
   }
   o = options("compute.heavy", NULL, NULL);
   CHECK(gb_gl_work(&o, &w, &err) && w.compute && w.kind == GB_GL_ALU);
   CHECK(w.work_e6 == GB_ALU_FLOP / 1e3);
   o = options("compute.heavy", "bw", NULL);
   CHECK(gb_gl_work(&o, &w, &err) && w.work_e6 == 2.0 * GB_BW_BYTES / 1e3);
   o = options("render.light", NULL, NULL);
   CHECK(gb_gl_work(&o, &w, &err) && w.w == GB_LIGHT_DIM && w.h == GB_LIGHT_DIM);
   o = options("compute.light", NULL, NULL);
   CHECK(gb_gl_work(&o, &w, &err) && w.kind == GB_GL_LIGHT);
   const char *bad[][3] = {{"render.heavy", "bogus", NULL}, {"render.heavy", "fill", "oops"},
      {"compute.light", "alu", NULL}, {"compute.heavy", "texture", NULL}, {"no-test", NULL, NULL},
      {"render.light", "fill", NULL}, {"render.light", "tri", "720p"}};
   for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); i++) {
      o = options(bad[i][0], bad[i][1], bad[i][2]);
      CHECK(!gb_gl_work(&o, &w, &err));
   }
   /* Reuse the suite's rate conversion and percentile implementation. */
   struct gb_pass p = {.nwin = 3, .win_s = 0.25, .win_count = {250, 500, 1000}};
   struct gb_samples rates = {0}; gb_pass_rate(&p, &rates);
   CHECK(rates.n == 3 && rates.v[0] == 1000 && rates.v[1] == 500 && rates.v[2] == 250);
   struct gb_stat stat = gb_stats(&rates);
   CHECK(stat.n == 3 && stat.median == 500 && stat.min == 250 && stat.max == 1000);
   gb_samples_free(&rates);
   puts("workload and shared statistics checks passed");
   return 0;
}
