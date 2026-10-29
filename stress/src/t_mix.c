// SPDX-License-Identifier: GPL-2.0-only
/* Concurrent lanes (separate processes) and the five-lane mix (dangerous). */
#include "st.h"
#include "st_client.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct lane_arg {
   const struct st_lane *lane;
   struct st_run *r;
   int fd;
};

static int
lane_main(void *arg)
{
   struct lane_arg *a = arg;
   struct st_run sub = *a->r;
   sub.iterations = sub.checks = sub.errors = 0;
   sub.reason[0] = 0;
   sub.skipped = false;
   sub.nm = 0;
   sub.rng ^= (uint64_t)getpid() * 0x9e3779b97f4a7c15ull;
   a->lane->fn(&sub, a->lane->arg);
   sub.o = NULL;
   sub.t = NULL;
   gb_write_all(a->fd, &sub, sizeof(sub));
   return 0;
}

int
st_run_lanes(struct st_run *r, const struct st_lane *lanes, int n)
{
   pid_t pid[8];
   int fd[8];
   for (int i = 0; i < n && i < 8; i++) {
      int p[2];
      if (pipe(p))
         return st_fail(r, "pipe");
      struct lane_arg a = {&lanes[i], r, p[1]};
      pid[i] = st_fork(lane_main, &a);
      close(p[1]);
      fd[i] = p[0];
      st_sleep_ns(50 * ST_MS); /* staggered starts: simultaneous context creation is its own test */
   }
   uint64_t left = r->budget_ns > st_elapsed(r) ? r->budget_ns - st_elapsed(r) : 0;
   for (int i = 0; i < n && i < 8; i++) {
      int sig = 0;
      int rc = st_wait_pid(pid[i], left + 120 * ST_NS, &sig);
      struct st_run sub;
      bool got = rc == 0 && !gb_read_all(fd[i], &sub, sizeof(sub));
      close(fd[i]);
      if (!got) {
         st_fail(r, "lane %s: %s %d", lanes[i].name,
                 rc == -2 ? "did not finish (GPU hang or blocked teardown), timeout" : rc == -1 ? "died, signal" : "exit",
                 rc == -1 ? sig : rc);
         continue;
      }
      r->iterations += sub.iterations;
      r->checks += sub.checks;
      if (sub.errors) {
         if (!r->errors)
            snprintf(r->reason, sizeof(r->reason), "lane %s: %.480s", lanes[i].name, sub.reason);
         r->errors += sub.errors;
      }
      for (int m = 0; m < sub.nm; m++) {
         char name[48];
         snprintf(name, sizeof(name), "%s.%s", lanes[i].name, sub.m[m].name);
         name[39] = 0;
         st_metric(r, name, sub.m[m].v);
      }
      if (sub.skipped && !sub.errors) {
         char name[48];
         snprintf(name, sizeof(name), "%s.skipped", lanes[i].name);
         st_metric(r, name, 1);
      }
   }
   return st_verdict(r);
}

/* ======================= mix.wedge (dangerous) ======================= */

static int
lane_churn(struct st_run *r, void *arg)
{
   (void)arg;
   return t_life_churn(r);
}

static int
lane_render(struct st_run *r, void *arg)
{
   (void)arg;
   return t_render_tiler(r);
}

static int
lane_render2(struct st_run *r, void *arg)
{
   (void)arg;
   return t_cr_concurrent(r);
}

static int
lane_long(struct st_run *r, void *arg)
{
   (void)arg;
   return st_lane_longalu(r, 1500);
}

static int
lane_rq(struct st_run *r, void *arg)
{
   (void)arg;
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) == VK_SUCCESS && caps.rq)
      return t_rq_stress(r);
   return t_sync_dense(r);
}

int
t_mix_wedge(struct st_run *r)
{
   struct st_lane lanes[5] = {
      {"churn", lane_churn, NULL},     {"render", lane_render, NULL}, {"render2", lane_render2, NULL},
      {"longalu", lane_long, NULL},    {"rq", lane_rq, NULL},
   };
   return st_run_lanes(r, lanes, 5);
}
