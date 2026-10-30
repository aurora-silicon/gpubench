// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-stress-vk: runs the stress tests, one process per test, and prints one JSON line per
 * test on stdout (human-readable progress on stderr). See stress/README.md.
 *
 * Exit status: 0 every selected test passed or was skipped; 1 at least one test failed
 * (including new kernel-log errors with --kernel-log); 2 usage error, no Vulkan device or the
 * kernel log could not be read; 3 a test process could not be killed after its deadline
 * (the GPU or driver is likely wedged) and the run was aborted. */
#include "st.h"
#include "klog.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEFAULT_SECONDS_PER_WEIGHT 20.0
#define DEFAULT_MAX_SECONDS 300.0
#define MIN_TEST_SECONDS 4.0

const struct st_test st_tests[] = {
   {"xq.order", "producer/consumer pairs on two queues: timeline, binary, wait-before-signal and host signals",
    "cross-queue ordering and visibility", false, 1.0, NULL, t_xq_order},
   {"xq.chain", "8-link chains alternating a render queue and a compute queue, data checked against a CPU model",
    "chained data between render and compute queues", false, 1.0, NULL, t_xq_chain},
   {"xq.exit", "processes exit or are killed with dependent cross-queue work in flight; a fresh client checks the GPU after",
    "teardown of in-flight cross-queue work", false, 1.0, NULL, t_xq_exit},
   {"cr.concurrent", "render passes in flight on two queues beside compute on two others, every result checked",
    "concurrent render and compute", false, 1.5, NULL, t_cr_concurrent},
   {"render.first", "fresh devices whose first submission is a heavy render, then renders after idle gaps",
    "renders that finish but never complete (late or missing completion), first-frame setup", false, 1.0, NULL,
    t_render_first},
   {"render.tiler", "1080p frames of up to 128K full-width row quads (every tile of a row), cold and warm, exact counts",
    "tiler buffer growth on the first big frame, lost fragment work", false, 1.5, NULL, t_render_tiler},
   {"compute.long", "solo ~1.5 s dispatches, then long dispatches beside render, compute and ray-query rings",
    "watchdog timeouts under contention; legitimate long jobs killed", false, 1.5, NULL, t_compute_long},
   {"life.churn", "device create/destroy loops and short-lived processes that exit or are killed mid-work",
    "context teardown races (exit and kill with work in flight)", false, 1.0, NULL, t_life_churn},
   {"life.fault", "a client that page-faults the GPU over and over beside a healthy render+compute client",
    "a faulting client taking innocent clients down", false, 1.0,
    "(page fault|fault (addr|info|status|at)|unhandled fault|mmu fault|gpu fault|faulted|fault on|"
    "translation fault|work of a client failed)",
    t_life_fault},
   {"queue.exhaust", "opens devices with every queue until the driver refuses; older contexts must keep working",
    "queue/context exhaustion handling", false, 0.5, NULL, t_queue_exhaust},
   {"mem.limit", "allocations beyond the heap must fail cleanly; holds memory near a safe cap while checked work runs",
    "allocation-limit handling (clean errors, no hang), submissions under memory pressure", false, 1.0, NULL,
    t_mem_limit},
   {"mem.churn", "thousands of small buffers allocated, mapped, written by CPU and GPU, unmapped and freed",
    "buffer object and mapping churn", false, 1.0, NULL, t_mem_churn},
   {"sync.dense", "thousands of semaphores and fences in a random dependency graph across queues; blocked work at exit",
    "dense synchronisation and cancellation of pending work", false, 1.0, NULL, t_sync_dense},
   {"idle.wake", "checked bursts after 2, 20 and 200 ms of idle, repeated",
    "power-down/wake transitions losing or corrupting work", false, 1.0, NULL, t_idle_wake},
   {"queue.depth", "128 submissions in flight on one queue; every queue at once with mixed priorities",
    "queue-depth and priority extremes", false, 1.0, NULL, t_queue_depth},
   {"sparse.churn", "sparse buffer pages bound, written, read back and unbound in loops (skips without sparse)",
    "sparse binding churn", false, 0.5, NULL, t_sparse_churn},
   {"rq.stress", "4096 instances, refit/rebuild loops and 128K rays per dispatch, every hit checked (skips without ray query)",
    "ray-query traversal and acceleration-structure updates", false, 1.0, NULL, t_rq_stress},
   {"cr.pairs", "4 and 8 processes released at the same instant, each creating a context and rendering at once",
    "simultaneous render-context construction (has hard-hung a machine)", true, 1.0, NULL, t_cr_pairs},
   {"compute.starve", "1.5 s dispatches in one process beside another's 4 compute queues and clients killed mid-work",
    "watchdog starvation and kills beside long work (has crashed GPU firmware)", true, 1.0, NULL, t_compute_starve},
   {"mix.wedge", "five concurrent lanes: process churn, heavy render, multi-queue render, long ALU, ray query",
    "combined-load wedges (has hard-hung a machine)", true, 2.0, NULL, t_mix_wedge},
   {"mem.exhaust", "allocates and touches device memory until the driver refuses",
    "out-of-memory handling at the real limit (may invoke the kernel OOM killer)", true, 0.5, NULL, t_mem_exhaust},
   {NULL},
};

const struct st_test *
st_find_test(const char *name)
{
   for (const struct st_test *t = st_tests; t->name; t++)
      if (!strcmp(t->name, name))
         return t;
   return NULL;
}

static void
usage(FILE *f)
{
   fprintf(f,
           "usage: gpubench-stress-vk [--all | --test NAME[,NAME...]] [options]\n"
           "\n"
           "  --all                 every test except the dangerous ones (the default)\n"
           "  --test NAME[,...]     selected tests (repeatable)\n"
           "  --dangerous           allow dangerous tests; with --all, include them\n"
           "  --duration SECONDS    total time budget, split by test weight (default: 20 s per unit\n"
           "                        of weight, at most 300 s)\n"
           "  --kernel-log          fail a test on new GPU error lines in the kernel log (Linux)\n"
           "  --kernel-log-pattern RE    add an error pattern (POSIX ERE, case-insensitive; repeatable)\n"
           "  --kernel-log-ignore RE     ignore matching lines (repeatable)\n"
           "  --kernel-log-no-defaults   use only --kernel-log-pattern patterns\n"
           "  --device N            Vulkan physical device index (default 0, or GB_DEVICE)\n"
           "  --seed N              random seed (default: time based, printed)\n"
           "  --verbose             per-step progress on stderr\n"
           "  --no-fork             run tests in this process, without the watchdog (debugging)\n"
           "  --list                list the tests\n"
           "  --help\n"
           "\n"
           "Exit status: 0 pass, 1 a test failed, 2 usage/setup error, 3 a test process could not be\n"
           "killed after its deadline (run aborted; the GPU is probably wedged).\n");
}

static void
list_tests(void)
{
   for (const struct st_test *t = st_tests; t->name; t++)
      printf("%-15s %s%s\n                protects: %s\n", t->name, t->dangerous ? "[dangerous] " : "", t->summary,
             t->protects);
}

/* ---------------- device probe ---------------- */

struct probe_msg {
   int ok;
   struct st_caps caps;
};

static int
probe(int device, struct st_caps *caps)
{
   int fd[2];
   if (pipe(fd))
      return -1;
   pid_t pid = fork();
   if (pid == 0) {
      close(fd[0]);
      struct probe_msg m = {0};
      m.ok = st_probe(device, &m.caps) == VK_SUCCESS;
      gb_write_all(fd[1], &m, sizeof(m));
      _exit(0);
   }
   close(fd[1]);
   struct probe_msg m = {0};
   int rc = gb_read_all(fd[0], &m, sizeof(m));
   close(fd[0]);
   waitpid(pid, NULL, 0);
   if (rc || !m.ok)
      return -1;
   *caps = m.caps;
   return 0;
}

/* ---------------- one test ---------------- */

struct outcome {
   int status;
   struct st_run r;
   double wall_s;
   bool hung, unkillable;
   int signal;
};

/* Kills a process group and reaps everything that the subreaper inherits. Returns false if
 * some process could not be reaped within the grace period. */
static bool
reap_group(pid_t pgid, uint64_t grace_ns)
{
   kill(-pgid, SIGKILL);
   uint64_t end = st_now() + grace_ns;
   for (;;) {
      int status;
      pid_t w = waitpid(-1, &status, WNOHANG);
      if (w < 0 && errno == ECHILD)
         return true;
      if (w <= 0) {
         if (st_now() >= end)
            return false;
         st_sleep_ns(20 * ST_MS);
      }
   }
}

static bool g_no_fork;

static void
run_one(const struct st_opts *o, const struct st_test *t, double budget_s, uint64_t seed, struct outcome *out)
{
   memset(out, 0, sizeof(*out));
   if (g_no_fork) {
      uint64_t t0 = st_now();
      out->r.o = o;
      out->r.t = t;
      out->r.t0 = t0;
      out->r.budget_ns = (uint64_t)(budget_s * 1e9);
      out->r.rng = seed ? seed : 0x9e3779b97f4a7c15ull;
      t->run(&out->r);
      out->wall_s = (st_now() - t0) / 1e9;
      out->status = out->r.errors ? ST_FAIL : out->r.skipped ? ST_SKIP : ST_PASS;
      return;
   }
   int fd[2];
   if (pipe(fd)) {
      out->status = ST_FAIL;
      snprintf(out->r.reason, sizeof(out->r.reason), "pipe: %s", strerror(errno));
      return;
   }
   uint64_t t0 = st_now();
   fflush(stdout);
   fflush(stderr);
   pid_t pid = fork();
   if (pid == 0) {
      close(fd[0]);
      setpgid(0, 0);
      struct st_run r = {0};
      r.o = o;
      r.t = t;
      r.t0 = st_now();
      r.budget_ns = (uint64_t)(budget_s * 1e9);
      r.rng = seed ? seed : 0x9e3779b97f4a7c15ull;
      int rc = t->run(&r);
      if (rc == ST_FAIL && !r.errors)
         st_fail(&r, "failed");
      if (rc == ST_SKIP && !r.errors)
         r.skipped = true;
      r.o = NULL;
      r.t = NULL;
      gb_write_all(fd[1], &r, sizeof(r));
      close(fd[1]);
      fflush(stderr);
      _exit(0);
   }
   close(fd[1]);
   if (pid < 0) {
      close(fd[0]);
      out->status = ST_FAIL;
      snprintf(out->r.reason, sizeof(out->r.reason), "fork: %s", strerror(errno));
      return;
   }
   setpgid(pid, pid);
   /* Deadline: the budget plus a generous allowance for setup and teardown. */
   double allow = budget_s > 60 ? budget_s : 60;
   int sig = 0;
   int rc = st_wait_pid(pid, (uint64_t)((budget_s + allow) * 1e9), &sig);
   out->wall_s = (st_now() - t0) / 1e9;
   if (rc == -2) {
      out->hung = true;
      out->status = ST_FAIL;
   }
   bool got = false;
   if (rc >= 0) {
      got = !gb_read_all(fd[0], &out->r, sizeof(out->r));
   }
   close(fd[0]);
   /* Leftover helper processes of the test (and the test itself on a timeout). */
   if (!reap_group(pid, 15 * ST_NS))
      out->unkillable = true;
   if (out->hung) {
      snprintf(out->r.reason, sizeof(out->r.reason),
               "test did not finish within %.0f s (budget %.0f s): GPU work hung or teardown blocked%s", budget_s + allow,
               budget_s, out->unkillable ? "; a process could not be killed" : "");
      out->r.errors = out->r.errors ? out->r.errors : 1;
      return;
   }
   if (rc == -1) {
      out->signal = sig;
      out->status = ST_FAIL;
      snprintf(out->r.reason, sizeof(out->r.reason), "test process died (signal %d: %s)", sig, strsignal(sig));
      out->r.errors = 1;
      return;
   }
   if (!got) {
      out->status = ST_FAIL;
      snprintf(out->r.reason, sizeof(out->r.reason), "test process exited (%d) without a result", rc);
      out->r.errors = 1;
      return;
   }
   out->status = out->r.errors ? ST_FAIL : out->r.skipped ? ST_SKIP : ST_PASS;
}

static const char *
status_name(int s)
{
   return s == ST_PASS ? "pass" : s == ST_SKIP ? "skip" : "fail";
}

/* ---------------- main ---------------- */

int
main(int argc, char **argv)
{
   struct st_opts o = {.device = 0};
   struct klog_opts ko = {0};
   bool all = false, kernel_log = false, have_seed = false, no_fork = false;
   const struct st_test *sel[64];
   int nsel = 0;
   double duration = 0;
   const char *dev = getenv("GB_DEVICE");
   if (dev)
      o.device = atoi(dev);

   for (int i = 1; i < argc; i++) {
      const char *a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (fprintf(stderr, "%s needs a value\n", a), usage(stderr), exit(2), (char *)0))
      if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
         usage(stdout);
         return 0;
      } else if (!strcmp(a, "--list")) {
         list_tests();
         return 0;
      } else if (!strcmp(a, "--all")) {
         all = true;
      } else if (!strcmp(a, "--test")) {
         char *list = strdup(NEXT());
         for (char *s = strtok(list, ","); s; s = strtok(NULL, ",")) {
            const struct st_test *t = st_find_test(s);
            if (!t) {
               fprintf(stderr, "unknown test %s (see --list)\n", s);
               return 2;
            }
            if (nsel < 64)
               sel[nsel++] = t;
         }
      } else if (!strcmp(a, "--dangerous")) {
         o.dangerous = true;
      } else if (!strcmp(a, "--duration")) {
         duration = atof(NEXT());
         if (duration <= 0) {
            fprintf(stderr, "--duration must be positive\n");
            return 2;
         }
      } else if (!strcmp(a, "--kernel-log")) {
         kernel_log = true;
      } else if (!strcmp(a, "--kernel-log-pattern")) {
         if (ko.npatterns < KLOG_MAX_PATTERNS)
            ko.patterns[ko.npatterns++] = NEXT();
         kernel_log = true;
      } else if (!strcmp(a, "--kernel-log-ignore")) {
         if (ko.nignore < KLOG_MAX_PATTERNS)
            ko.ignore[ko.nignore++] = NEXT();
         kernel_log = true;
      } else if (!strcmp(a, "--kernel-log-no-defaults")) {
         ko.no_defaults = true;
         kernel_log = true;
      } else if (!strcmp(a, "--device")) {
         o.device = atoi(NEXT());
      } else if (!strcmp(a, "--seed")) {
         o.seed = strtoull(NEXT(), NULL, 0);
         have_seed = true;
      } else if (!strcmp(a, "--no-fork")) {
         no_fork = true;
      } else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) {
         o.verbose = true;
      } else {
         fprintf(stderr, "unknown argument %s\n", a);
         usage(stderr);
         return 2;
      }
#undef NEXT
   }
   if (!nsel || all) {
      for (const struct st_test *t = st_tests; t->name; t++) {
         if (t->dangerous && !o.dangerous)
            continue;
         bool dup = false;
         for (int i = 0; i < nsel; i++)
            dup |= sel[i] == t;
         if (!dup && nsel < 64)
            sel[nsel++] = t;
      }
   }
   for (int i = 0; i < nsel; i++)
      if (sel[i]->dangerous && !o.dangerous) {
         fprintf(stderr, "%s is dangerous (it has hung or crashed GPUs); pass --dangerous to run it\n", sel[i]->name);
         return 2;
      }
   if (!have_seed)
      o.seed = st_now() ^ ((uint64_t)getpid() << 32);

   double wsum = 0;
   for (int i = 0; i < nsel; i++)
      wsum += sel[i]->weight;
   if (duration <= 0) {
      duration = DEFAULT_SECONDS_PER_WEIGHT * wsum;
      if (duration > DEFAULT_MAX_SECONDS)
         duration = DEFAULT_MAX_SECONDS;
   }
   o.duration_s = duration;

   g_no_fork = no_fork;
   signal(SIGPIPE, SIG_IGN);
   /* Orphaned helpers of a test are reparented here so they can be reaped and checked. */
   prctl(PR_SET_CHILD_SUBREAPER, 1);

   struct st_caps caps;
   if (probe(o.device, &caps)) {
      fprintf(stderr, "gpubench-stress-vk: no usable Vulkan device %d (graphics+compute queue family)\n", o.device);
      return 2;
   }

   struct klog *klog = NULL;
   if (kernel_log) {
      char err[512];
      klog = klog_open(&ko, err, sizeof(err));
      if (!klog) {
         fprintf(stderr, "gpubench-stress-vk: %s\n", err);
         return 2;
      }
   }

   char api[32];
   snprintf(api, sizeof(api), "%u.%u.%u", VK_API_VERSION_MAJOR(caps.api), VK_API_VERSION_MINOR(caps.api),
            VK_API_VERSION_PATCH(caps.api));
   struct gb_json j;
   gb_json_init(&j);
   gb_json_obj_begin(&j, NULL);
   gb_json_str(&j, "schema", "gpubench-stress-run/1");
   gb_json_str(&j, "suite_sha", GB_SUITE_SHA);
   gb_json_str(&j, "device_name", caps.name);
   gb_json_str(&j, "driver_name", caps.driver);
   gb_json_str(&j, "driver_info", caps.driver_info);
   gb_json_str(&j, "api_version", api);
   gb_json_int(&j, "queues", caps.queues);
   gb_json_obj_begin(&j, "features");
   gb_json_bool(&j, "buffer_device_address", caps.bda);
   gb_json_bool(&j, "ray_query", caps.rq);
   gb_json_bool(&j, "sparse_residency_buffer", caps.sparse);
   gb_json_bool(&j, "global_priority", caps.gprio);
   gb_json_obj_end(&j);
   gb_json_int(&j, "device_local_heap_bytes", (int64_t)caps.heap);
   gb_json_num(&j, "duration_s", duration);
   char seedbuf[32];
   snprintf(seedbuf, sizeof(seedbuf), "%llu", (unsigned long long)o.seed);
   gb_json_str(&j, "seed", seedbuf); /* a string: JSON numbers lose 64-bit precision */
   gb_json_bool(&j, "dangerous", o.dangerous);
   gb_json_str(&j, "kernel_log", klog ? klog_source(klog) : NULL);
   gb_json_arr_begin(&j, "tests");
   for (int i = 0; i < nsel; i++)
      gb_json_str(&j, NULL, sel[i]->name);
   gb_json_arr_end(&j);
   gb_json_obj_end(&j);
   puts(j.buf);
   fflush(stdout);
   gb_json_free(&j);
   fprintf(stderr, "gpubench-stress-vk: %s (%s %s), %d tests, %.0f s budget, seed %llu%s\n", caps.name, caps.driver,
           caps.driver_info, nsel, duration, (unsigned long long)o.seed, klog ? ", kernel log checked" : "");

   int passed = 0, failed = 0, skipped = 0, code = 0;
   uint64_t run0 = st_now();
   for (int i = 0; i < nsel; i++) {
      const struct st_test *t = sel[i];
      double budget = duration * t->weight / wsum;
      if (budget < MIN_TEST_SECONDS)
         budget = MIN_TEST_SECONDS;
      fprintf(stderr, "[%2d/%d] %-15s %.0f s ...\n", i + 1, nsel, t->name, budget);
      struct klog_lines kl = {0};
      bool kl_err = false;
      if (klog)
         klog_collect(klog, NULL, &kl); /* lines from before the test belong to nobody */
      klog_lines_free(&kl);
      struct outcome out;
      run_one(&o, t, budget, o.seed + (uint64_t)i * 0x9e3779b97f4a7c15ull, &out);
      if (klog) {
         st_sleep_ns(500 * ST_MS); /* late lines of this test */
         if (klog_collect(klog, t->kernel_expect, &kl)) {
            kl_err = true;
         } else if (kl.nbad) {
            if (out.status != ST_FAIL)
               snprintf(out.r.reason, sizeof(out.r.reason), "new kernel-log error: %s", kl.bad[0]);
            out.status = ST_FAIL;
         }
      }
      if (out.status == ST_PASS)
         passed++;
      else if (out.status == ST_SKIP)
         skipped++;
      else
         failed++;

      gb_json_init(&j);
      gb_json_obj_begin(&j, NULL);
      gb_json_str(&j, "schema", "gpubench-stress/1");
      gb_json_str(&j, "lane", "stress");
      gb_json_str(&j, "test", t->name);
      gb_json_str(&j, "status", status_name(out.status));
      gb_json_bool(&j, "dangerous", t->dangerous);
      gb_json_num(&j, "budget_s", budget);
      gb_json_num(&j, "wall_s", out.wall_s);
      gb_json_int(&j, "iterations", (int64_t)out.r.iterations);
      gb_json_int(&j, "checks", (int64_t)out.r.checks);
      gb_json_int(&j, "errors", (int64_t)out.r.errors);
      gb_json_str(&j, "reason", out.r.reason[0] ? out.r.reason : NULL);
      gb_json_obj_begin(&j, "metrics");
      for (int m = 0; m < out.r.nm && m < ST_MAX_METRICS; m++) {
         out.r.m[m].name[sizeof(out.r.m[m].name) - 1] = 0;
         gb_json_num(&j, out.r.m[m].name, out.r.m[m].v);
      }
      gb_json_obj_end(&j);
      if (klog) {
         gb_json_obj_begin(&j, "kernel_log");
         gb_json_bool(&j, "read", !kl_err);
         gb_json_int(&j, "lines", kl.total);
         gb_json_int(&j, "expected_errors", kl.expected);
         gb_json_int(&j, "ignored_errors", kl.ignored);
         gb_json_arr_begin(&j, "errors");
         for (int b = 0; b < kl.nbad && b < 20; b++)
            gb_json_str(&j, NULL, kl.bad[b]);
         gb_json_arr_end(&j);
         gb_json_obj_end(&j);
      }
      gb_json_obj_end(&j);
      puts(j.buf);
      fflush(stdout);
      gb_json_free(&j);

      fprintf(stderr, "[%2d/%d] %-15s %s  %.1f s, %llu iterations, %llu checks%s%s\n", i + 1, nsel, t->name,
              out.status == ST_PASS ? "PASS" : out.status == ST_SKIP ? "SKIP" : "FAIL", out.wall_s,
              (unsigned long long)out.r.iterations, (unsigned long long)out.r.checks, out.r.reason[0] ? ": " : "",
              out.r.reason);
      for (int b = 0; b < kl.nbad && b < 5; b++)
         fprintf(stderr, "        kernel: %s\n", kl.bad[b]);
      if (kl.expected)
         fprintf(stderr, "        kernel: %d expected line(s) from deliberate faults\n", kl.expected);
      klog_lines_free(&kl);

      if (out.unkillable) {
         fprintf(stderr, "gpubench-stress-vk: a process of %s could not be killed; aborting the run\n", t->name);
         code = 3;
         break;
      }
   }
   if (code != 3)
      code = failed ? 1 : 0;

   gb_json_init(&j);
   gb_json_obj_begin(&j, NULL);
   gb_json_str(&j, "schema", "gpubench-stress-summary/1");
   gb_json_int(&j, "passed", passed);
   gb_json_int(&j, "failed", failed);
   gb_json_int(&j, "skipped", skipped);
   gb_json_num(&j, "wall_s", (st_now() - run0) / 1e9);
   gb_json_int(&j, "exit_code", code);
   gb_json_obj_end(&j);
   puts(j.buf);
   gb_json_free(&j);
   fprintf(stderr, "gpubench-stress-vk: %d passed, %d failed, %d skipped in %.0f s -> exit %d\n", passed, failed,
           skipped, (st_now() - run0) / 1e9, code);
   klog_close(klog);
   return code;
}
