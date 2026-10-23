// SPDX-License-Identifier: GPL-2.0-only
/* gb_common: CLI, timing, statistics, JSON result lines and child processes
 * shared by every gpubench executor (Vulkan side). No Vulkan dependency. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define GB_MAX_PARAMS 16
#define GB_MAX_DEPTHS 16
#define GB_MAX_WIN 64

/* ---- time ---- */
uint64_t gb_now_ns(void);
void gb_sleep_until_ns(uint64_t t);
void gb_spin_ns(uint64_t ns);

/* ---- options ---- */
enum gb_mode { GB_MODE_QUEUE = 0, GB_MODE_QUEUES, GB_MODE_PROCS };

struct gb_opts {
   const char *test;
   const char *lane;          /* "suite" by default */
   int depths[GB_MAX_DEPTHS];
   int ndepths;               /* 0 = test default */
   enum gb_mode mode;
   struct { char key[32]; char val[64]; } params[GB_MAX_PARAMS];
   int nparams;
   int warmup_ms, runs, run_ms;
   int timed_runs;            /* windows in the timed pass */
   bool timestamps;
   int device;
   bool info, list;
   /* child-process protocol (mode=procs) */
   bool child;
   int child_depth;
   const char *argv0;
   int argc; char **argv;     /* original argv, re-used to spawn children */
};

int gb_parse_opts(struct gb_opts *o, int argc, char **argv);
const char *gb_param(const struct gb_opts *o, const char *key, const char *def);
long gb_param_int(const struct gb_opts *o, const char *key, long def);
/* Error out on --param keys the test does not know. `known` is NULL-terminated. */
bool gb_check_params(const struct gb_opts *o, const char *const *known);
const char *gb_mode_name(enum gb_mode m);

/* ---- samples and statistics ---- */
struct gb_samples {
   double *v;
   size_t n, cap;
};
void gb_samples_push(struct gb_samples *s, double v);
void gb_samples_append(struct gb_samples *d, const struct gb_samples *s);
void gb_samples_free(struct gb_samples *s);
void gb_samples_clear(struct gb_samples *s);
struct gb_stat {
   double median, p5, p95, mean, min, max;
   size_t n;
};
struct gb_stat gb_stats(const struct gb_samples *s);

/* ---- JSON result line ---- */
struct gb_json {
   char *buf;
   size_t len, cap;
   int depth;
   bool need_comma[16];
};
void gb_json_init(struct gb_json *j);
void gb_json_free(struct gb_json *j);
void gb_json_obj_begin(struct gb_json *j, const char *key);   /* key NULL at top level / in arrays */
void gb_json_obj_end(struct gb_json *j);
void gb_json_arr_begin(struct gb_json *j, const char *key);
void gb_json_arr_end(struct gb_json *j);
void gb_json_str(struct gb_json *j, const char *key, const char *v);
void gb_json_num(struct gb_json *j, const char *key, double v);
void gb_json_int(struct gb_json *j, const char *key, int64_t v);
void gb_json_bool(struct gb_json *j, const char *key, bool v);
void gb_json_null(struct gb_json *j, const char *key);

/* A result being assembled for one configuration. */
struct gb_result {
   struct gb_json j;
   const struct gb_opts *o;
   bool metrics_open;
   uint64_t t_start;
};
/* Starts the object and writes schema, lane, test, params (from opts plus
 * `extra` "k=v" pairs, NULL-terminated, may be NULL), mode and depth. */
void gb_result_begin(struct gb_result *r, const struct gb_opts *o, enum gb_mode mode, int depth,
                     const char *const *extra);
void gb_result_run_info(struct gb_result *r, int warmup_ms, int runs, int run_ms);
/* Metric from samples: rate metrics pass include_samples=true. */
void gb_result_metric(struct gb_result *r, const char *name, const char *unit, bool lower_better,
                      const struct gb_samples *s, bool include_samples);
/* Metric with a single value (n = 1). */
void gb_result_metric1(struct gb_result *r, const char *name, const char *unit, bool lower_better,
                       double v);
void gb_result_end_metrics(struct gb_result *r);
/* Finish with status "ok", "skip" or "fail"; `primary` and `why` may be NULL. */
void gb_result_finish(struct gb_result *r, const char *status, const char *primary, const char *why);
/* Shorthand: print a skip/fail line with no metrics. */
void gb_result_simple(const struct gb_opts *o, enum gb_mode mode, int depth, const char *const *extra,
                      const char *status, const char *why);

/* ---- thread barrier (portable) ---- */
struct gb_barrier;
struct gb_barrier *gb_barrier_new(int n);
void gb_barrier_wait(struct gb_barrier *b);
void gb_barrier_free(struct gb_barrier *b);

/* ---- child processes for mode=procs ----
 * The parent re-executes itself with --child; each child is started only
 * after the previous one has reported READY (setup done, contexts primed).
 * Then every child receives the same start time (monotonic ns) and runs the
 * same window schedule; results come back over the pipe as gb_wire blobs. */
struct gb_child {
   int pid;
   int to_child, from_child;
};
int gb_spawn_children(const struct gb_opts *o, int n, struct gb_child *out);
int gb_child_wait_ready(struct gb_child *c);
int gb_child_send_u64(struct gb_child *c, uint64_t v);
int gb_child_reap(struct gb_child *c);
/* child side */
void gb_child_ready(void);
uint64_t gb_child_recv_u64(void);

/* binary wire helpers (child stdout -> parent) */
int gb_write_all(int fd, const void *p, size_t n);
int gb_read_all(int fd, void *p, size_t n);
int gb_wire_put_samples(int fd, const struct gb_samples *s);
int gb_wire_get_samples(int fd, struct gb_samples *s);

/* ---- one measured pass of a submission stream ---- */
struct gb_pass {
   int nwin;
   double win_s;
   uint64_t win_count[GB_MAX_WIN];
   struct gb_samples latency, submit_cpu, record_cpu, gpu, gap;
   uint64_t submits;
   bool timed, failed;
};
void gb_pass_free(struct gb_pass *p);
void gb_pass_merge(struct gb_pass *dst, const struct gb_pass *src);
/* us_per_submit per window */
void gb_pass_rate(const struct gb_pass *p, struct gb_samples *us_per_submit);
int gb_pass_put(int fd, const struct gb_pass *p);
int gb_pass_get(int fd, struct gb_pass *p);
/* Writes the standard metrics of an untimed and (optional) timed pass. */
void gb_result_stream_metrics(struct gb_result *r, const struct gb_pass *untimed,
                              const struct gb_pass *timed);
/* Per-window conversion helper: out[i] = work_e6 / us[i], where work_e6 is
 * the work per submission in output units times 1e6 (GFLOP per submission x
 * 1e6 gives GFLOP/s). */
void gb_derive_rate(struct gb_result *r, const char *name, const char *unit,
                    const struct gb_samples *us, double work_e6);
static inline double gb_median(const struct gb_samples *s) { return gb_stats(s).median; }
/* Delay before a synchronized start (threads, processes). */
#define GB_SYNC_START_NS 50000000ull

void gb_die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void gb_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
