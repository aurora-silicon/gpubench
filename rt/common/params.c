// SPDX-License-Identifier: GPL-2.0-only
#include "params.h"
#include "json.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *test_names[RT_TEST_COUNT] = {"rt.scene", "rt.asbuild", "rt.rays"};

const char *rt_test_name(int t) { return t >= 0 && t < RT_TEST_COUNT ? test_names[t] : "?"; }

const char *rt_as_mode_name(int m)
{
   return m == RT_AS_REFIT ? "refit" : m == RT_AS_REBUILD ? "rebuild" : "static";
}

void rt_params_defaults(rt_params *p, int test)
{
   memset(p, 0, sizeof(*p));
   strcpy(p->preset, "medium");
   p->width = 1920;
   p->height = 1080;
   p->mesh_tris = 20000;
   p->meshes = 10;
   p->instances = 256;
   p->water_tris = 32768;
   p->blas_mode = RT_AS_STATIC;
   p->dyn_patches = 4;
   p->tlas_mode = RT_AS_STATIC;
   p->moving = 0.1f;
   p->reflective = 0.5f;
   p->rpp = 1;
   p->bounces = 1;
   p->shadows = 1;
   p->tmax = 400.0f;
   p->alpha = 1;
   p->trace_mode = RT_TRACE_QUERY;
   p->seed = 1234;
   if (test == RT_TEST_ASBUILD) {
      strcpy(p->kind, "blas");
      strcpy(p->build, "fasttrace");
      p->tris = 65536;
   } else if (test == RT_TEST_RAYS) {
      strcpy(p->kind, "coherent");
      p->rays = 1u << 18;
   }
}

static int parse_as_mode(const char *v, int *out)
{
   if (!strcmp(v, "static") || !strcmp(v, "none")) *out = RT_AS_STATIC;
   else if (!strcmp(v, "refit") || !strcmp(v, "update")) *out = RT_AS_REFIT;
   else if (!strcmp(v, "rebuild") || !strcmp(v, "build")) *out = RT_AS_REBUILD;
   else return -1;
   return 0;
}

static int parse_bool(const char *v, int *out)
{
   if (!strcmp(v, "0") || !strcmp(v, "off") || !strcmp(v, "false") || !strcmp(v, "no")) *out = 0;
   else if (!strcmp(v, "1") || !strcmp(v, "on") || !strcmp(v, "true") || !strcmp(v, "yes")) *out = 1;
   else return -1;
   return 0;
}

static int parse_u32(const char *v, uint32_t *out)
{
   char *e;
   unsigned long x = strtoul(v, &e, 0);
   if (*e == 'k' || *e == 'K') { x <<= 10; e++; }
   else if (*e == 'M' || *e == 'm') { x <<= 20; e++; }
   if (e == v || *e) return -1;
   *out = (uint32_t)x;
   return 0;
}

static int parse_f(const char *v, float *out)
{
   char *e;
   *out = strtof(v, &e);
   return (e == v || *e) ? -1 : 0;
}

int rt_params_set(rt_params *p, const char *k, const char *v)
{
#define U(name, field) if (!strcmp(k, name)) return parse_u32(v, &p->field);
#define F(name, field) if (!strcmp(k, name)) return parse_f(v, &p->field);
#define B(name, field) if (!strcmp(k, name)) return parse_bool(v, &p->field);
#define S(name, field) if (!strcmp(k, name)) { snprintf(p->field, sizeof(p->field), "%s", v); return 0; }
   if (!strcmp(k, "res")) {
      unsigned w, h;
      if (!strcmp(v, "720p")) { p->width = 1280; p->height = 720; return 0; }
      if (!strcmp(v, "1080p")) { p->width = 1920; p->height = 1080; return 0; }
      if (!strcmp(v, "1440p")) { p->width = 2560; p->height = 1440; return 0; }
      if (!strcmp(v, "2160p")) { p->width = 3840; p->height = 2160; return 0; }
      if (sscanf(v, "%ux%u", &w, &h) != 2) return -1;
      p->width = w; p->height = h;
      return 0;
   }
   U("width", width) U("height", height)
   U("mesh-tris", mesh_tris) U("meshes", meshes) U("instances", instances) U("water-tris", water_tris)
   if (!strcmp(k, "blas")) return parse_as_mode(v, &p->blas_mode);
   if (!strcmp(k, "tlas")) return parse_as_mode(v, &p->tlas_mode);
   U("dyn-patches", dyn_patches) F("moving", moving)
   F("reflective", reflective) U("rpp", rpp) U("bounces", bounces) B("shadows", shadows)
   F("tmax", tmax) B("alpha", alpha) U("seed", seed)
   if (!strcmp(k, "camera")) {
      if (!strcmp(v, "fixed")) p->camera_orbit = 0;
      else if (!strcmp(v, "orbit")) p->camera_orbit = 1;
      else return -1;
      return 0;
   }
   if (!strcmp(k, "trace")) {
      if (!strcmp(v, "query")) p->trace_mode = RT_TRACE_QUERY;
      else if (!strcmp(v, "pipeline")) p->trace_mode = RT_TRACE_PIPELINE;
      else return -1;
      return 0;
   }
   S("kind", kind) S("build", build) U("tris", tris) U("rays", rays) B("compact", compact)
#undef U
#undef F
#undef B
#undef S
   return -1;
}

static int apply_ini(rt_params *p, const char *text, const char *name)
{
   const char *s = text;
   int in = 0, found = 0;
   char line[512];
   while (*s) {
      size_t n = strcspn(s, "\n");
      if (n >= sizeof(line)) n = sizeof(line) - 1;
      memcpy(line, s, n);
      line[n] = 0;
      s += n;
      if (*s == '\n') s++;
      char *l = line;
      while (isspace((unsigned char)*l)) l++;
      char *e = l + strlen(l);
      while (e > l && isspace((unsigned char)e[-1])) *--e = 0;
      if (!*l || *l == '#' || *l == ';') continue;
      if (*l == '[') {
         char *r = strchr(l, ']');
         if (r) *r = 0;
         in = !strcmp(l + 1, name);
         found |= in;
         continue;
      }
      if (!in) continue;
      char *eq = strchr(l, '=');
      if (!eq) continue;
      *eq = 0;
      if (rt_params_set(p, l, eq + 1)) {
         fprintf(stderr, "preset %s: bad key %s=%s\n", name, l, eq + 1);
         return -1;
      }
   }
   return found ? 0 : -1;
}

static char *read_file(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f) return NULL;
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   char *b = malloc(n + 1);
   if (fread(b, 1, n, f) != (size_t)n) n = 0;
   b[n] = 0;
   fclose(f);
   return b;
}

int rt_params_apply_preset(rt_params *p, const char *name, const char *file)
{
   char *text = file && *file ? read_file(file) : NULL;
   if (file && *file && !text) {
      fprintf(stderr, "cannot read preset file %s\n", file);
      return -1;
   }
   int r = apply_ini(p, text ? text : rt_presets_ini, name);
   free(text);
   if (r) {
      fprintf(stderr, "unknown preset %s\n", name);
      return -1;
   }
   snprintf(p->preset, sizeof(p->preset), "%s", name);
   return 0;
}

static void usage(const char *exe)
{
   fprintf(stderr,
      "%s: ray-tracing benchmarks (Solar Bay-style scene and RT micro-tests)\n\n"
      "Executor interface:\n"
      "  --info | --list | --test rt.scene|rt.asbuild|rt.rays\n"
      "  --depths 1,2,..  --mode queue  --param key=value (repeatable)\n"
      "  --warmup-ms N  --runs N  --run-ms N  --timestamps on|off  --device N\n"
      "Run controls: --frames N (fixed submissions per window), --png PATH, --shader-stats, --tag T,\n"
      "  --stamps pass|frame (rt.scene timed pass: per-pass stamps, default, or frame start/end only),\n"
      "  --window [--window-seconds N], --verbose, --preset-file PATH, --list-presets\n"
      "Scene parameters (--param k=v or --k v):\n"
      "  preset=raster|light|medium|solarbay|heavy|extreme  res=WxH|720p|1080p|1440p|2160p\n"
      "  mesh-tris meshes instances water-tris  blas=static|refit|rebuild dyn-patches\n"
      "  tlas=static|refit|rebuild moving  reflective rpp bounces shadows tmax alpha\n"
      "  trace=query|pipeline camera=fixed|orbit seed\n"
      "rt.asbuild: kind=blas|refit|tlas build=fasttrace|fastbuild tris=N compact=0|1\n"
      "rt.rays: kind=coherent|incoherent rays=N (+ scene geometry parameters)\n", exe);
}

static int parse_depths(rt_run *r, const char *v)
{
   r->ndepths = 0;
   const char *s = v;
   while (*s && r->ndepths < 16) {
      char *e;
      unsigned long d = strtoul(s, &e, 10);
      if (e == s || !d) return -1;
      r->depths[r->ndepths++] = (uint32_t)d;
      s = *e == ',' ? e + 1 : e;
      if (*e && *e != ',') return -1;
   }
   return 0;
}

int rt_parse(rt_params *p, rt_run *r, int argc, char **argv, const char *exe)
{
   memset(r, 0, sizeof(*r));
   r->test = RT_TEST_SCENE;
   r->timestamps = 1;
   r->warmup_ms = r->runs = r->run_ms = UINT32_MAX;  /* unset: the test's default */
   strcpy(r->mode, "queue");
   const char *preset = NULL;
   /* first pass: test, preset and preset file (they decide the defaults) */
   for (int i = 1; i < argc; i++) {
      const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
      if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(exe); return 1; }
      if (!strcmp(a, "--list-presets")) { fputs(rt_presets_ini, stdout); return 1; }
      if (!strcmp(a, "--test") && v) {
         int t;
         for (t = 0; t < RT_TEST_COUNT; t++) if (!strcmp(v, test_names[t])) break;
         if (t == RT_TEST_COUNT) { fprintf(stderr, "unknown test %s\n", v); return -1; }
         r->test = t;
      }
      if (!strcmp(a, "--preset") && v) preset = v;
      if (!strcmp(a, "--param") && v && !strncmp(v, "preset=", 7)) preset = v + 7;
      if (!strcmp(a, "--preset-file") && v) snprintf(r->preset_file, sizeof(r->preset_file), "%s", v);
   }
   rt_params_defaults(p, r->test);
   if (rt_params_apply_preset(p, preset ? preset : "medium", r->preset_file)) return -1;
   for (int i = 1; i < argc; i++) {
      const char *a = argv[i];
      const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define NEEDV() do { if (!v) { fprintf(stderr, "missing value for %s\n", a); return -1; } i++; } while (0)
      if (!strcmp(a, "--info")) { r->info = 1; continue; }
      if (!strcmp(a, "--list")) { r->list = 1; continue; }
      if (!strcmp(a, "--shader-stats")) { r->shader_stats = 1; continue; }
      if (!strcmp(a, "--window")) { r->window = 1; continue; }
      if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) { r->verbose = 1; continue; }
      if (!strcmp(a, "--test") || !strcmp(a, "--preset") || !strcmp(a, "--preset-file")) { NEEDV(); continue; }
      if (!strcmp(a, "--depths")) { NEEDV(); if (parse_depths(r, v)) { fprintf(stderr, "bad --depths\n"); return -1; } continue; }
      if (!strcmp(a, "--mode")) { NEEDV(); snprintf(r->mode, sizeof(r->mode), "%s", v); continue; }
      if (!strcmp(a, "--warmup-ms")) { NEEDV(); r->warmup_ms = (uint32_t)strtoul(v, NULL, 0); continue; }
      if (!strcmp(a, "--runs")) { NEEDV(); r->runs = (uint32_t)strtoul(v, NULL, 0); continue; }
      if (!strcmp(a, "--run-ms")) { NEEDV(); r->run_ms = (uint32_t)strtoul(v, NULL, 0); continue; }
      if (!strcmp(a, "--frames")) { NEEDV(); r->frames = (uint32_t)strtoul(v, NULL, 0); continue; }
      if (!strcmp(a, "--device")) { NEEDV(); r->device = atoi(v); continue; }
      if (!strcmp(a, "--png")) { NEEDV(); snprintf(r->png, sizeof(r->png), "%s", v); continue; }
      if (!strcmp(a, "--window-seconds")) { NEEDV(); r->window = 1; r->window_s = (uint32_t)strtoul(v, NULL, 0); continue; }
      if (!strcmp(a, "--dump-archive")) { NEEDV(); snprintf(r->dump_archive, sizeof(r->dump_archive), "%s", v); continue; }
      if (!strcmp(a, "--stamps")) {
         NEEDV();
         if (!strcmp(v, "frame")) r->stamps_frame = 1;
         else if (!strcmp(v, "pass")) r->stamps_frame = 0;
         else { fprintf(stderr, "--stamps pass|frame\n"); return -1; }
         continue;
      }
      if (!strcmp(a, "--tag")) { NEEDV(); snprintf(r->tag, sizeof(r->tag), "%s", v); continue; }
      if (!strcmp(a, "--timestamps")) {
         NEEDV();
         if (parse_bool(v, &r->timestamps)) { fprintf(stderr, "bad --timestamps %s\n", v); return -1; }
         continue;
      }
      char key[64];
      const char *val;
      if (!strcmp(a, "--param")) {
         NEEDV();
         const char *eq = strchr(v, '=');
         if (!eq) { fprintf(stderr, "--param needs key=value\n"); return -1; }
         snprintf(key, sizeof(key), "%.*s", (int)(eq - v), v);
         val = eq + 1;
      } else if (!strncmp(a, "--", 2)) {
         const char *eq = strchr(a + 2, '=');
         if (eq) {
            snprintf(key, sizeof(key), "%.*s", (int)(eq - a - 2), a + 2);
            val = eq + 1;
         } else {
            NEEDV();
            snprintf(key, sizeof(key), "%s", a + 2);
            val = v;
         }
      } else {
         fprintf(stderr, "unexpected argument %s\n", a);
         return -1;
      }
      if (!strcmp(key, "preset")) continue;
      if (rt_params_set(p, key, val)) { fprintf(stderr, "unknown or bad parameter %s=%s\n", key, val); return -1; }
#undef NEEDV
   }
   if (p->bounces < 1) p->bounces = 1;
   if (p->bounces > 8) p->bounces = 8;
   if (p->rpp < 1) p->rpp = 1;
   if (p->rpp > 64) p->rpp = 64;
   if (p->dyn_patches > 16) p->dyn_patches = 16;
   if (p->blas_mode == RT_AS_STATIC) p->dyn_patches = 0;
   if (p->tlas_mode == RT_AS_STATIC) p->moving = 0;
   if (p->meshes < 1) p->meshes = 1;
   if (p->mesh_tris < 64) p->mesh_tris = 64;
   if (p->water_tris < 32) p->water_tris = 32;
   return 0;
}

void rt_run_defaults(rt_run *r)
{
   if (!r->ndepths) {
      r->ndepths = 1;
      r->depths[0] = r->test == RT_TEST_SCENE ? 2 : 1;
   }
   int sc = r->test == RT_TEST_SCENE;
   if (r->warmup_ms == UINT32_MAX) r->warmup_ms = sc ? 1500 : 300;
   if (r->runs == UINT32_MAX) r->runs = 5;
   if (r->run_ms == UINT32_MAX) r->run_ms = sc ? 1000 : 250;
   if (!r->runs) r->runs = 1;
}

void rt_params_json(rt_json *j, const rt_params *p, int test)
{
   char res[32];
   rt_json_obj(j, "params");
   if (test == RT_TEST_ASBUILD) {
      rt_json_str(j, "kind", p->kind);
      rt_json_str(j, "build", p->build);
      rt_json_int(j, "tris", p->tris);
      if (p->compact) rt_json_bool(j, "compact", 1);
      rt_json_end(j);
      return;
   }
   if (test == RT_TEST_RAYS) {
      rt_json_str(j, "kind", p->kind);
      rt_json_int(j, "rays", p->rays);
   }
   rt_json_str(j, "preset", p->preset);
   if (test == RT_TEST_SCENE) {
      snprintf(res, sizeof(res), "%ux%u", p->width, p->height);
      rt_json_str(j, "res", res);
   }
   rt_json_int(j, "mesh-tris", p->mesh_tris);
   rt_json_int(j, "meshes", p->meshes);
   rt_json_int(j, "instances", p->instances);
   rt_json_int(j, "water-tris", p->water_tris);
   rt_json_int(j, "seed", p->seed);
   rt_json_num(j, "tmax", p->tmax);
   rt_json_bool(j, "alpha", p->alpha);
   if (test == RT_TEST_SCENE) {
      rt_json_str(j, "blas", rt_as_mode_name(p->blas_mode));
      rt_json_int(j, "dyn-patches", p->dyn_patches);
      rt_json_str(j, "tlas", rt_as_mode_name(p->tlas_mode));
      rt_json_num(j, "moving", p->moving);
      rt_json_num(j, "reflective", p->reflective);
      rt_json_int(j, "rpp", p->rpp);
      rt_json_int(j, "bounces", p->bounces);
      rt_json_bool(j, "shadows", p->shadows);
      rt_json_str(j, "trace", p->trace_mode == RT_TRACE_PIPELINE ? "pipeline" : "query");
      rt_json_str(j, "camera", p->camera_orbit ? "orbit" : "fixed");
   }
   rt_json_end(j);
}
