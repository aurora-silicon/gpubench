// SPDX-License-Identifier: GPL-2.0-only
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put(rt_json *j, const char *s, size_t n)
{
   if (j->len + n + 1 > j->cap) {
      j->cap = (j->len + n + 1) * 2 + 256;
      j->buf = realloc(j->buf, j->cap);
   }
   memcpy(j->buf + j->len, s, n);
   j->len += n;
   j->buf[j->len] = 0;
}
static void puts_(rt_json *j, const char *s) { put(j, s, strlen(s)); }

static void quoted(rt_json *j, const char *s)
{
   put(j, "\"", 1);
   for (; *s; s++) {
      unsigned char c = (unsigned char)*s;
      if (c == '"' || c == '\\') { char e[2] = {'\\', (char)c}; put(j, e, 2); }
      else if (c == '\n') puts_(j, "\\n");
      else if (c == '\t') puts_(j, "\\t");
      else if (c < 0x20) { char e[8]; snprintf(e, sizeof(e), "\\u%04x", c); puts_(j, e); }
      else put(j, (const char *)&c, 1);
   }
   put(j, "\"", 1);
}

static void sep(rt_json *j, const char *key)
{
   if (j->depth > 0) {
      if (!j->first[j->depth]) put(j, ",", 1);
      j->first[j->depth] = 0;
   }
   if (key) { quoted(j, key); put(j, ":", 1); }
}

void rt_json_init(rt_json *j) { memset(j, 0, sizeof(*j)); put(j, "", 0); }
void rt_json_free(rt_json *j) { free(j->buf); memset(j, 0, sizeof(*j)); }
void rt_json_obj(rt_json *j, const char *key) { sep(j, key); put(j, "{", 1); j->first[++j->depth] = 1; }
void rt_json_end(rt_json *j) { put(j, "}", 1); j->depth--; }
void rt_json_arr(rt_json *j, const char *key) { sep(j, key); put(j, "[", 1); j->first[++j->depth] = 1; }
void rt_json_end_arr(rt_json *j) { put(j, "]", 1); j->depth--; }
void rt_json_str(rt_json *j, const char *key, const char *v)
{
   sep(j, key);
   if (v) quoted(j, v); else puts_(j, "null");
}
void rt_json_num(rt_json *j, const char *key, double v)
{
   char b[64];
   sep(j, key);
   if (!isfinite(v)) { puts_(j, "null"); return; }
   double a = fabs(v);
   if (a != 0 && (a < 1e-3 || a >= 1e12)) snprintf(b, sizeof(b), "%.6g", v);
   else if (a >= 1000) snprintf(b, sizeof(b), "%.2f", v);
   else snprintf(b, sizeof(b), "%.4g", v);
   puts_(j, b);
}
void rt_json_int(rt_json *j, const char *key, int64_t v)
{
   char b[32];
   sep(j, key);
   snprintf(b, sizeof(b), "%lld", (long long)v);
   puts_(j, b);
}
void rt_json_bool(rt_json *j, const char *key, int v) { sep(j, key); puts_(j, v ? "true" : "false"); }
void rt_json_null(rt_json *j, const char *key) { sep(j, key); puts_(j, "null"); }
void rt_json_raw(rt_json *j, const char *key, const char *raw) { sep(j, key); puts_(j, raw && *raw ? raw : "null"); }

static int cmpd(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

double rt_quantile(const double *v, size_t n, double q)
{
   if (!n) return NAN;
   double *s = malloc(n * sizeof(double));
   memcpy(s, v, n * sizeof(double));
   qsort(s, n, sizeof(double), cmpd);
   double pos = q * (double)(n - 1);
   size_t i = (size_t)pos;
   double f = pos - (double)i, r = s[i];
   if (i + 1 < n) r = s[i] * (1 - f) + s[i + 1] * f;
   free(s);
   return r;
}

double rt_mean(const double *v, size_t n)
{
   double s = 0;
   for (size_t i = 0; i < n; i++) s += v[i];
   return n ? s / (double)n : NAN;
}

void rt_metric(rt_json *j, const char *name, const char *unit, const char *better, const double *v, size_t n,
               int samples)
{
   if (!n) return;
   rt_json_obj(j, name);
   rt_json_str(j, "unit", unit);
   rt_json_str(j, "better", better);
   rt_json_num(j, "median", rt_quantile(v, n, 0.5));
   rt_json_num(j, "p5", rt_quantile(v, n, 0.05));
   rt_json_num(j, "p95", rt_quantile(v, n, 0.95));
   rt_json_int(j, "n", (int64_t)n);
   if (samples) {
      rt_json_arr(j, "samples");
      for (size_t i = 0; i < n; i++) rt_json_num(j, NULL, v[i]);
      rt_json_end_arr(j);
   }
   rt_json_end(j);
}

void rt_vec_push(rt_vec *a, double x)
{
   if (a->n == a->cap) {
      a->cap = a->cap ? a->cap * 2 : 1024;
      a->v = realloc(a->v, a->cap * sizeof(double));
   }
   a->v[a->n++] = x;
}
void rt_vec_free(rt_vec *a) { free(a->v); memset(a, 0, sizeof(*a)); }
