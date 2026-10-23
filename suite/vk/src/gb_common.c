// SPDX-License-Identifier: GPL-2.0-only
#define _GNU_SOURCE
#include "gb_common.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

extern char **environ;

/* ---------------- time ---------------- */

uint64_t
gb_now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void
gb_sleep_until_ns(uint64_t t)
{
   for (;;) {
      uint64_t now = gb_now_ns();
      if (now >= t)
         return;
      uint64_t d = t - now;
      if (d > 2000000) {
         struct timespec ts = {.tv_sec = (d - 1000000) / 1000000000ull,
                               .tv_nsec = (d - 1000000) % 1000000000ull};
         nanosleep(&ts, NULL);
      }
   }
}

void
gb_spin_ns(uint64_t ns)
{
   uint64_t end = gb_now_ns() + ns;
   while (gb_now_ns() < end)
      ;
}

/* ---------------- log ---------------- */

void
gb_die(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fprintf(stderr, "gpubench: ");
   vfprintf(stderr, fmt, ap);
   fprintf(stderr, "\n");
   va_end(ap);
   exit(2);
}

void
gb_log(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fprintf(stderr, "gpubench: ");
   vfprintf(stderr, fmt, ap);
   fprintf(stderr, "\n");
   va_end(ap);
}

/* ---------------- options ---------------- */

const char *
gb_mode_name(enum gb_mode m)
{
   switch (m) {
   case GB_MODE_QUEUES: return "queues";
   case GB_MODE_PROCS: return "procs";
   default: return "queue";
   }
}

static int
parse_list(const char *s, int *out, int max)
{
   int n = 0;
   while (*s && n < max) {
      char *end;
      long v = strtol(s, &end, 10);
      if (end == s || v <= 0)
         gb_die("bad depth list near '%s'", s);
      out[n++] = (int)v;
      s = *end == ',' ? end + 1 : end;
      if (*end && *end != ',')
         gb_die("bad depth list near '%s'", end);
   }
   return n;
}

int
gb_parse_opts(struct gb_opts *o, int argc, char **argv)
{
   memset(o, 0, sizeof(*o));
   o->lane = "suite";
   o->warmup_ms = 300;
   o->runs = 5;
   o->run_ms = 250;
   o->timed_runs = 3;
   o->timestamps = true;
   o->argc = argc;
   o->argv = argv;
   o->argv0 = argv[0];

   const char *e;
   if ((e = getenv("GB_WARMUP_MS"))) o->warmup_ms = atoi(e);
   if ((e = getenv("GB_RUNS"))) o->runs = atoi(e);
   if ((e = getenv("GB_RUN_MS"))) o->run_ms = atoi(e);
   if ((e = getenv("GB_TIMED_RUNS"))) o->timed_runs = atoi(e);

   for (int i = 1; i < argc; i++) {
      const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (gb_die("%s needs a value", a), (char *)0))
      if (!strcmp(a, "--info")) o->info = true;
      else if (!strcmp(a, "--list")) o->list = true;
      else if (!strcmp(a, "--test")) o->test = NEXT();
      else if (!strcmp(a, "--lane")) o->lane = NEXT();
      else if (!strcmp(a, "--depths")) o->ndepths = parse_list(NEXT(), o->depths, GB_MAX_DEPTHS);
      else if (!strcmp(a, "--mode")) {
         const char *m = NEXT();
         if (!strcmp(m, "queue")) o->mode = GB_MODE_QUEUE;
         else if (!strcmp(m, "queues")) o->mode = GB_MODE_QUEUES;
         else if (!strcmp(m, "procs")) o->mode = GB_MODE_PROCS;
         else gb_die("unknown mode %s", m);
      } else if (!strcmp(a, "--param")) {
         const char *kv = NEXT();
         const char *eq = strchr(kv, '=');
         if (!eq || o->nparams >= GB_MAX_PARAMS)
            gb_die("bad --param %s", kv);
         size_t kl = eq - kv;
         if (kl >= sizeof(o->params[0].key) || strlen(eq + 1) >= sizeof(o->params[0].val))
            gb_die("--param too long: %s", kv);
         memcpy(o->params[o->nparams].key, kv, kl);
         o->params[o->nparams].key[kl] = 0;
         strcpy(o->params[o->nparams].val, eq + 1);
         o->nparams++;
      } else if (!strcmp(a, "--warmup-ms")) o->warmup_ms = atoi(NEXT());
      else if (!strcmp(a, "--runs")) o->runs = atoi(NEXT());
      else if (!strcmp(a, "--run-ms")) o->run_ms = atoi(NEXT());
      else if (!strcmp(a, "--timed-runs")) o->timed_runs = atoi(NEXT());
      else if (!strcmp(a, "--timestamps")) {
         const char *v = NEXT();
         o->timestamps = !strcmp(v, "on") || !strcmp(v, "1");
      } else if (!strcmp(a, "--device")) o->device = atoi(NEXT());
      else if (!strcmp(a, "--child")) o->child = true;
      else if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
         printf("usage: %s --info | --list | --test NAME [--depths 1,2,4] [--mode queue|queues|procs]\n"
                "       [--param k=v]... [--warmup-ms N] [--runs N] [--run-ms N] [--timed-runs N]\n"
                "       [--timestamps on|off] [--device N]\n", argv[0]);
         exit(0);
      } else
         gb_die("unknown argument %s", a);
#undef NEXT
   }
   if (o->runs < 1 || o->runs > GB_MAX_WIN || o->timed_runs < 0 || o->timed_runs > GB_MAX_WIN)
      gb_die("runs must be 1..%d", GB_MAX_WIN);
   return 0;
}

const char *
gb_param(const struct gb_opts *o, const char *key, const char *def)
{
   for (int i = 0; i < o->nparams; i++)
      if (!strcmp(o->params[i].key, key))
         return o->params[i].val;
   return def;
}

long
gb_param_int(const struct gb_opts *o, const char *key, long def)
{
   const char *v = gb_param(o, key, NULL);
   return v ? strtol(v, NULL, 0) : def;
}

bool
gb_check_params(const struct gb_opts *o, const char *const *known)
{
   for (int i = 0; i < o->nparams; i++) {
      bool ok = false;
      for (const char *const *k = known; k && *k; k++)
         if (!strcmp(*k, o->params[i].key))
            ok = true;
      if (!ok) {
         gb_log("test %s: unknown --param %s", o->test, o->params[i].key);
         return false;
      }
   }
   return true;
}

/* ---------------- samples ---------------- */

void
gb_samples_push(struct gb_samples *s, double v)
{
   if (s->n == s->cap) {
      s->cap = s->cap ? s->cap * 2 : 1024;
      s->v = realloc(s->v, s->cap * sizeof(double));
      if (!s->v)
         gb_die("out of memory");
   }
   s->v[s->n++] = v;
}

void
gb_samples_append(struct gb_samples *d, const struct gb_samples *s)
{
   for (size_t i = 0; i < s->n; i++)
      gb_samples_push(d, s->v[i]);
}

void
gb_samples_free(struct gb_samples *s)
{
   free(s->v);
   memset(s, 0, sizeof(*s));
}

void
gb_samples_clear(struct gb_samples *s)
{
   s->n = 0;
}

static int
cmp_double(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static double
pct(const double *v, size_t n, double p)
{
   if (n == 1)
      return v[0];
   double x = p * (n - 1);
   size_t i = (size_t)x;
   if (i + 1 >= n)
      return v[n - 1];
   double f = x - i;
   return v[i] * (1 - f) + v[i + 1] * f;
}

struct gb_stat
gb_stats(const struct gb_samples *s)
{
   struct gb_stat st = {0};
   st.n = s->n;
   if (!s->n)
      return st;
   double *c = malloc(s->n * sizeof(double));
   memcpy(c, s->v, s->n * sizeof(double));
   qsort(c, s->n, sizeof(double), cmp_double);
   st.median = pct(c, s->n, 0.5);
   st.p5 = pct(c, s->n, 0.05);
   st.p95 = pct(c, s->n, 0.95);
   st.min = c[0];
   st.max = c[s->n - 1];
   double sum = 0;
   for (size_t i = 0; i < s->n; i++)
      sum += c[i];
   st.mean = sum / s->n;
   free(c);
   return st;
}

/* ---------------- JSON ---------------- */

static void
jput(struct gb_json *j, const char *s, size_t n)
{
   if (j->len + n + 1 > j->cap) {
      while (j->len + n + 1 > j->cap)
         j->cap = j->cap ? j->cap * 2 : 4096;
      j->buf = realloc(j->buf, j->cap);
   }
   memcpy(j->buf + j->len, s, n);
   j->len += n;
   j->buf[j->len] = 0;
}

static void
jputs(struct gb_json *j, const char *s)
{
   jput(j, s, strlen(s));
}

static void
jstr(struct gb_json *j, const char *s)
{
   jputs(j, "\"");
   for (; *s; s++) {
      char c = *s;
      if (c == '"' || c == '\\') {
         char b[2] = {'\\', c};
         jput(j, b, 2);
      } else if ((unsigned char)c < 0x20) {
         char b[8];
         snprintf(b, sizeof(b), "\\u%04x", c);
         jputs(j, b);
      } else
         jput(j, &c, 1);
   }
   jputs(j, "\"");
}

static void
jkey(struct gb_json *j, const char *key)
{
   if (j->need_comma[j->depth])
      jputs(j, ",");
   j->need_comma[j->depth] = true;
   if (key) {
      jstr(j, key);
      jputs(j, ":");
   }
}

void
gb_json_init(struct gb_json *j)
{
   memset(j, 0, sizeof(*j));
}

void
gb_json_free(struct gb_json *j)
{
   free(j->buf);
   memset(j, 0, sizeof(*j));
}

void
gb_json_obj_begin(struct gb_json *j, const char *key)
{
   jkey(j, key);
   jputs(j, "{");
   j->need_comma[++j->depth] = false;
}

void
gb_json_obj_end(struct gb_json *j)
{
   j->depth--;
   jputs(j, "}");
}

void
gb_json_arr_begin(struct gb_json *j, const char *key)
{
   jkey(j, key);
   jputs(j, "[");
   j->need_comma[++j->depth] = false;
}

void
gb_json_arr_end(struct gb_json *j)
{
   j->depth--;
   jputs(j, "]");
}

void
gb_json_str(struct gb_json *j, const char *key, const char *v)
{
   jkey(j, key);
   if (v)
      jstr(j, v);
   else
      jputs(j, "null");
}

void
gb_json_num(struct gb_json *j, const char *key, double v)
{
   jkey(j, key);
   char b[64];
   if (!isfinite(v))
      snprintf(b, sizeof(b), "null");
   else if (v != 0 && fabs(v) < 0.01)
      snprintf(b, sizeof(b), "%.4g", v);
   else
      snprintf(b, sizeof(b), "%.4f", v);
   /* trim trailing zeros of fixed notation */
   if (strchr(b, '.') && !strchr(b, 'e')) {
      size_t l = strlen(b);
      while (l > 1 && b[l - 1] == '0')
         b[--l] = 0;
      if (b[l - 1] == '.')
         b[--l] = 0;
   }
   jputs(j, b);
}

void
gb_json_int(struct gb_json *j, const char *key, int64_t v)
{
   jkey(j, key);
   char b[32];
   snprintf(b, sizeof(b), "%lld", (long long)v);
   jputs(j, b);
}

void
gb_json_bool(struct gb_json *j, const char *key, bool v)
{
   jkey(j, key);
   jputs(j, v ? "true" : "false");
}

void
gb_json_null(struct gb_json *j, const char *key)
{
   jkey(j, key);
   jputs(j, "null");
}

/* ---------------- results ---------------- */

void
gb_result_begin(struct gb_result *r, const struct gb_opts *o, enum gb_mode mode, int depth,
                const char *const *extra)
{
   memset(r, 0, sizeof(*r));
   r->o = o;
   r->t_start = gb_now_ns();
   gb_json_init(&r->j);
   struct gb_json *j = &r->j;
   gb_json_obj_begin(j, NULL);
   gb_json_str(j, "schema", "gpubench-result/1");
   gb_json_str(j, "lane", o->lane);
   gb_json_str(j, "test", o->test);
   gb_json_obj_begin(j, "params");
   for (int i = 0; i < o->nparams; i++)
      gb_json_str(j, o->params[i].key, o->params[i].val);
   for (const char *const *e = extra; e && *e; e++) {
      const char *eq = strchr(*e, '=');
      char k[64];
      size_t kl = eq ? (size_t)(eq - *e) : strlen(*e);
      if (kl >= sizeof(k))
         kl = sizeof(k) - 1;
      memcpy(k, *e, kl);
      k[kl] = 0;
      if (!gb_param(o, k, NULL))
         gb_json_str(j, k, eq ? eq + 1 : "");
   }
   gb_json_obj_end(j);
   gb_json_str(j, "mode", gb_mode_name(mode));
   gb_json_int(j, "depth", depth);
}

void
gb_result_run_info(struct gb_result *r, int warmup_ms, int runs, int run_ms)
{
   gb_json_int(&r->j, "warmup_ms", warmup_ms);
   gb_json_int(&r->j, "runs", runs);
   gb_json_int(&r->j, "run_ms", run_ms);
}

static void
open_metrics(struct gb_result *r)
{
   if (!r->metrics_open) {
      gb_json_obj_begin(&r->j, "metrics");
      r->metrics_open = true;
   }
}

void
gb_result_metric(struct gb_result *r, const char *name, const char *unit, bool lower_better,
                 const struct gb_samples *s, bool include_samples)
{
   if (!s->n)
      return;
   open_metrics(r);
   struct gb_json *j = &r->j;
   struct gb_stat st = gb_stats(s);
   gb_json_obj_begin(j, name);
   gb_json_str(j, "unit", unit);
   gb_json_str(j, "better", lower_better ? "lower" : "higher");
   gb_json_num(j, "median", st.median);
   gb_json_num(j, "p5", st.p5);
   gb_json_num(j, "p95", st.p95);
   gb_json_int(j, "n", (int64_t)st.n);
   if (include_samples) {
      gb_json_arr_begin(j, "samples");
      for (size_t i = 0; i < s->n; i++)
         gb_json_num(j, NULL, s->v[i]);
      gb_json_arr_end(j);
   }
   gb_json_obj_end(j);
}

void
gb_result_metric1(struct gb_result *r, const char *name, const char *unit, bool lower_better, double v)
{
   struct gb_samples s = {0};
   gb_samples_push(&s, v);
   open_metrics(r);
   struct gb_json *j = &r->j;
   gb_json_obj_begin(j, name);
   gb_json_str(j, "unit", unit);
   gb_json_str(j, "better", lower_better ? "lower" : "higher");
   gb_json_num(j, "median", v);
   gb_json_num(j, "p5", v);
   gb_json_num(j, "p95", v);
   gb_json_int(j, "n", 1);
   gb_json_obj_end(j);
   gb_samples_free(&s);
}

void
gb_result_end_metrics(struct gb_result *r)
{
   if (r->metrics_open) {
      gb_json_obj_end(&r->j);
      r->metrics_open = false;
   }
}

void
gb_result_finish(struct gb_result *r, const char *status, const char *primary, const char *why)
{
   gb_result_end_metrics(r);
   struct gb_json *j = &r->j;
   gb_json_str(j, "status", status);
   if (primary)
      gb_json_str(j, "primary", primary);
   if (why)
      gb_json_str(j, !strcmp(status, "skip") ? "reason" : "error", why);
   gb_json_obj_end(j);
   fputs(j->buf, stdout);
   fputc('\n', stdout);
   fflush(stdout);
   gb_json_free(j);
}

void
gb_result_simple(const struct gb_opts *o, enum gb_mode mode, int depth, const char *const *extra,
                 const char *status, const char *why)
{
   struct gb_result r;
   gb_result_begin(&r, o, mode, depth, extra);
   gb_result_finish(&r, status, NULL, why);
}

/* ---------------- barrier ---------------- */

struct gb_barrier {
   pthread_mutex_t m;
   pthread_cond_t c;
   int n, waiting;
   unsigned gen;
};

struct gb_barrier *
gb_barrier_new(int n)
{
   struct gb_barrier *b = calloc(1, sizeof(*b));
   pthread_mutex_init(&b->m, NULL);
   pthread_cond_init(&b->c, NULL);
   b->n = n;
   return b;
}

void
gb_barrier_wait(struct gb_barrier *b)
{
   pthread_mutex_lock(&b->m);
   unsigned gen = b->gen;
   if (++b->waiting == b->n) {
      b->waiting = 0;
      b->gen++;
      pthread_cond_broadcast(&b->c);
   } else {
      while (gen == b->gen)
         pthread_cond_wait(&b->c, &b->m);
   }
   pthread_mutex_unlock(&b->m);
}

void
gb_barrier_free(struct gb_barrier *b)
{
   pthread_mutex_destroy(&b->m);
   pthread_cond_destroy(&b->c);
   free(b);
}

/* ---------------- wire ---------------- */

int
gb_write_all(int fd, const void *p, size_t n)
{
   const char *c = p;
   while (n) {
      ssize_t w = write(fd, c, n);
      if (w < 0 && errno == EINTR)
         continue;
      if (w <= 0)
         return -1;
      c += w;
      n -= w;
   }
   return 0;
}

int
gb_read_all(int fd, void *p, size_t n)
{
   char *c = p;
   while (n) {
      ssize_t r = read(fd, c, n);
      if (r < 0 && errno == EINTR)
         continue;
      if (r <= 0)
         return -1;
      c += r;
      n -= r;
   }
   return 0;
}

int
gb_wire_put_samples(int fd, const struct gb_samples *s)
{
   uint64_t n = s->n;
   if (gb_write_all(fd, &n, sizeof(n)))
      return -1;
   return n ? gb_write_all(fd, s->v, n * sizeof(double)) : 0;
}

int
gb_wire_get_samples(int fd, struct gb_samples *s)
{
   uint64_t n;
   if (gb_read_all(fd, &n, sizeof(n)))
      return -1;
   for (uint64_t i = 0; i < n; i++) {
      double v;
      if (gb_read_all(fd, &v, sizeof(v)))
         return -1;
      gb_samples_push(s, v);
   }
   return 0;
}

/* ---------------- children ---------------- */

static const char *
self_exe(const struct gb_opts *o)
{
   static char path[4096];
#ifdef __APPLE__
   uint32_t sz = sizeof(path);
   if (_NSGetExecutablePath(path, &sz) == 0)
      return path;
   ssize_t n = -1;
#else
   ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
#endif
   if (n > 0) {
      path[n] = 0;
      return path;
   }
   return o->argv0;
}

int
gb_spawn_children(const struct gb_opts *o, int n, struct gb_child *out)
{
   const char *exe = self_exe(o);
   char **argv = calloc(o->argc + 2, sizeof(char *));
   argv[0] = (char *)exe;
   for (int i = 1; i < o->argc; i++)
      argv[i] = o->argv[i];
   argv[o->argc] = "--child";
   argv[o->argc + 1] = NULL;

   for (int i = 0; i < n; i++) {
      int to[2], from[2];
      if (pipe(to) || pipe(from))
         return -1;
      posix_spawn_file_actions_t fa;
      posix_spawn_file_actions_init(&fa);
      posix_spawn_file_actions_adddup2(&fa, to[0], 0);
      posix_spawn_file_actions_adddup2(&fa, from[1], 1);
      posix_spawn_file_actions_addclose(&fa, to[1]);
      posix_spawn_file_actions_addclose(&fa, from[0]);
      pid_t pid;
      int rc = posix_spawn(&pid, exe, &fa, NULL, argv, environ);
      posix_spawn_file_actions_destroy(&fa);
      close(to[0]);
      close(from[1]);
      if (rc)
         return -1;
      out[i].pid = pid;
      out[i].to_child = to[1];
      out[i].from_child = from[0];
      /* Stagger: the next child starts only after this one is set up. */
      if (gb_child_wait_ready(&out[i]))
         return -1;
   }
   free(argv);
   return 0;
}

int
gb_child_wait_ready(struct gb_child *c)
{
   char b;
   if (gb_read_all(c->from_child, &b, 1) || b != 'R')
      return -1;
   return 0;
}

int
gb_child_send_u64(struct gb_child *c, uint64_t v)
{
   return gb_write_all(c->to_child, &v, sizeof(v));
}

int
gb_child_reap(struct gb_child *c)
{
   int st = 0;
   close(c->to_child);
   close(c->from_child);
   waitpid(c->pid, &st, 0);
   return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
}

void
gb_child_ready(void)
{
   char b = 'R';
   gb_write_all(1, &b, 1);
}

uint64_t
gb_child_recv_u64(void)
{
   uint64_t v = 0;
   if (gb_read_all(0, &v, sizeof(v)))
      return 0;
   return v;
}

/* ---------------- passes ---------------- */

void
gb_pass_free(struct gb_pass *p)
{
   gb_samples_free(&p->latency);
   gb_samples_free(&p->submit_cpu);
   gb_samples_free(&p->record_cpu);
   gb_samples_free(&p->gpu);
   gb_samples_free(&p->gap);
}

void
gb_pass_merge(struct gb_pass *d, const struct gb_pass *s)
{
   if (!d->nwin) {
      d->nwin = s->nwin;
      d->win_s = s->win_s;
      d->timed = s->timed;
   }
   for (int i = 0; i < s->nwin && i < GB_MAX_WIN; i++)
      d->win_count[i] += s->win_count[i];
   gb_samples_append(&d->latency, &s->latency);
   gb_samples_append(&d->submit_cpu, &s->submit_cpu);
   gb_samples_append(&d->record_cpu, &s->record_cpu);
   gb_samples_append(&d->gpu, &s->gpu);
   gb_samples_append(&d->gap, &s->gap);
   d->submits += s->submits;
   d->failed |= s->failed;
}

void
gb_pass_rate(const struct gb_pass *p, struct gb_samples *us)
{
   for (int i = 0; i < p->nwin; i++)
      if (p->win_count[i])
         gb_samples_push(us, p->win_s * 1e6 / p->win_count[i]);
}

int
gb_pass_put(int fd, const struct gb_pass *p)
{
   int32_t hdr[4] = {p->nwin, p->timed, p->failed, 0};
   if (gb_write_all(fd, hdr, sizeof(hdr)) || gb_write_all(fd, &p->win_s, sizeof(double)) ||
       gb_write_all(fd, p->win_count, sizeof(p->win_count)) || gb_write_all(fd, &p->submits, sizeof(uint64_t)))
      return -1;
   if (gb_wire_put_samples(fd, &p->latency) || gb_wire_put_samples(fd, &p->submit_cpu) ||
       gb_wire_put_samples(fd, &p->record_cpu) || gb_wire_put_samples(fd, &p->gpu) ||
       gb_wire_put_samples(fd, &p->gap))
      return -1;
   return 0;
}

int
gb_pass_get(int fd, struct gb_pass *p)
{
   memset(p, 0, sizeof(*p));
   int32_t hdr[4];
   if (gb_read_all(fd, hdr, sizeof(hdr)) || gb_read_all(fd, &p->win_s, sizeof(double)) ||
       gb_read_all(fd, p->win_count, sizeof(p->win_count)) || gb_read_all(fd, &p->submits, sizeof(uint64_t)))
      return -1;
   p->nwin = hdr[0];
   p->timed = hdr[1];
   p->failed = hdr[2];
   if (gb_wire_get_samples(fd, &p->latency) || gb_wire_get_samples(fd, &p->submit_cpu) ||
       gb_wire_get_samples(fd, &p->record_cpu) || gb_wire_get_samples(fd, &p->gpu) ||
       gb_wire_get_samples(fd, &p->gap))
      return -1;
   return 0;
}

void
gb_result_stream_metrics(struct gb_result *r, const struct gb_pass *u, const struct gb_pass *t)
{
   struct gb_samples us = {0};
   gb_pass_rate(u, &us);
   gb_result_metric(r, "us_per_submit", "us", true, &us, true);
   gb_result_metric(r, "latency_us", "us", true, &u->latency, false);
   gb_result_metric(r, "submit_cpu_us", "us", true, &u->submit_cpu, false);
   gb_result_metric(r, "record_cpu_us", "us", true, &u->record_cpu, false);
   gb_samples_free(&us);
   if (t && t->timed) {
      struct gb_samples ust = {0};
      gb_pass_rate(t, &ust);
      gb_result_metric(r, "gpu_us", "us", true, &t->gpu, false);
      gb_result_metric(r, "gpu_gap_us", "us", true, &t->gap, false);
      gb_result_metric(r, "us_per_submit_timed", "us", true, &ust, true);
      gb_samples_free(&ust);
   }
}

void
gb_derive_rate(struct gb_result *r, const char *name, const char *unit, const struct gb_samples *us,
               double work_e6)
{
   struct gb_samples d = {0};
   for (size_t i = 0; i < us->n; i++)
      gb_samples_push(&d, work_e6 / us->v[i]);
   gb_result_metric(r, name, unit, false, &d, true);
   gb_samples_free(&d);
}
