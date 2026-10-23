// SPDX-License-Identifier: GPL-2.0-only
/* Minimal JSON writer and the gpubench-result/1 metric helper. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct rt_json {
   char *buf;
   size_t len, cap;
   int depth;
   int first[32];
} rt_json;

void rt_json_init(rt_json *j);
void rt_json_free(rt_json *j);
void rt_json_obj(rt_json *j, const char *key);   /* key NULL inside arrays or at top level */
void rt_json_end(rt_json *j);                    /* closes the innermost object */
void rt_json_arr(rt_json *j, const char *key);
void rt_json_end_arr(rt_json *j);
void rt_json_str(rt_json *j, const char *key, const char *v);
void rt_json_num(rt_json *j, const char *key, double v);
void rt_json_int(rt_json *j, const char *key, int64_t v);
void rt_json_bool(rt_json *j, const char *key, int v);
void rt_json_null(rt_json *j, const char *key);
/* Inserts already formatted JSON (an object or value) under key. */
void rt_json_raw(rt_json *j, const char *key, const char *raw);

/* Sorts a copy of v and returns the linearly interpolated quantile q in [0, 1]. */
double rt_quantile(const double *v, size_t n, double q);
double rt_mean(const double *v, size_t n);
/* Emits {"unit","better","median","p5","p95","n"[,"samples"]} under key name. Skips n == 0. */
void rt_metric(rt_json *j, const char *name, const char *unit, const char *better, const double *v, size_t n,
               int samples);
/* Appends to a growable double vector. */
typedef struct { double *v; size_t n, cap; } rt_vec;
void rt_vec_push(rt_vec *a, double x);
void rt_vec_free(rt_vec *a);
