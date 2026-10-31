// SPDX-License-Identifier: GPL-2.0-only
#include <string.h>
#include "gb_gl_work.h"

bool
gb_gl_work(const struct gb_opts *o, struct gb_gl_work *w, const char **error)
{
   static const char *const none[] = {NULL};
   static const char *const kind[] = {"kind", NULL};
   static const char *const render[] = {"kind", "res", NULL};
   memset(w, 0, sizeof(*w));
   const char *k = gb_param(o, "kind", "");
   const char *const *params = none;
   *error = "unknown test or kind";
   if (!strcmp(o->test, "compute.light")) {
      w->kind = GB_GL_LIGHT;
      w->compute = true;
   } else if (!strcmp(o->test, "compute.heavy")) {
      params = kind;
      w->compute = true;
      if (!*k || !strcmp(k, "alu")) {
         w->kind = GB_GL_ALU;
         w->metric = "gflops"; w->unit = "GFLOP/s";
         w->work_e6 = GB_ALU_FLOP / 1e3;
      } else if (!strcmp(k, "bw")) {
         w->kind = GB_GL_BW;
         w->metric = "gbps"; w->unit = "GB/s";
         w->work_e6 = 2.0 * GB_BW_BYTES / 1e3;
      } else return false;
   } else if (!strcmp(o->test, "render.light")) {
      params = kind;
      w->w = w->h = GB_LIGHT_DIM;
      if (!*k || !strcmp(k, "tri")) w->kind = GB_GL_TRI;
      else if (!strcmp(k, "clear")) w->kind = GB_GL_CLEAR;
      else return false;
   } else if (!strcmp(o->test, "render.heavy")) {
      params = render;
      if (!*k || !strcmp(k, "fill")) w->kind = GB_GL_FILL;
      else if (!strcmp(k, "opaque")) w->kind = GB_GL_OPAQUE;
      else if (!strcmp(k, "texture")) w->kind = GB_GL_TEXTURE;
      else if (!strcmp(k, "geom")) w->kind = GB_GL_GEOM;
      else return false;
      const char *r = gb_param(o, "res", "1080p");
      if (!strcmp(r, "720p")) w->w = 1280, w->h = 720;
      else if (!strcmp(r, "1080p")) w->w = 1920, w->h = 1080;
      else if (!strcmp(r, "1440p")) w->w = 2560, w->h = 1440;
      else if (!strcmp(r, "2160p")) w->w = 3840, w->h = 2160;
      else { *error = "res must be 720p, 1080p, 1440p or 2160p"; return false; }
      w->metric = w->kind == GB_GL_GEOM ? "mtri_s" : "gpix_s";
      w->unit = w->kind == GB_GL_GEOM ? "Mtri/s" : "Gpix/s";
      w->work_e6 = w->kind == GB_GL_GEOM ? GB_GEOM_TRIS : (double)w->w * w->h * GB_FILL_LAYERS / 1e3;
   } else return false;
   if (!gb_check_params(o, params)) { *error = "unknown parameter"; return false; }
   return true;
}
