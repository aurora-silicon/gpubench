// SPDX-License-Identifier: GPL-2.0-only
/* Long compute: legitimate long dispatches alone, long dispatches beside render, compute and
 * ray-query work, and the starvation mix that has crashed GPU firmware (dangerous). */
#include "st.h"
#include "st_client.h"
#include "st_ring.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

#define LONG_GROUPS 256u

/* Long ALU ring on queue 0 beside a render ring and a compute ring on other queues, in one
 * process. Runs until r's budget ends. */
int
st_lane_longalu(struct st_run *r, double target_ms)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   int nq = caps.queues < 3 ? (int)caps.queues : 3;
   struct st_dev d;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_lcg cal;
   if ((vr = st_lcg_init(&d, LONG_GROUPS, &cal)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   double ms = 0;
   uint32_t iters = st_lcg_calibrate(&d, 0, &cal, target_ms, &ms);
   st_lcg_destroy(&d, &cal);
   if (!iters)
      return st_fail(r, "calibration of a %.0f ms dispatch failed", target_ms);
   struct st_ring g[3];
   int ng = 0;
   if ((vr = st_ring_init(&g[ng++], &d, 0, 2, ST_RING_LCG, LONG_GROUPS, iters)))
      return st_fail(r, "long ring: %s", st_vkr(vr));
   if (nq > 1 && (vr = st_ring_init(&g[ng++], &d, 1, 8, ST_RING_RENDER, 256, 256)))
      return st_fail(r, "render ring: %s", st_vkr(vr));
   if (nq > 2 && (vr = st_ring_init(&g[ng++], &d, 2, 8, ST_RING_CHAIN, 65536, 0)))
      return st_fail(r, "compute ring: %s", st_vkr(vr));
   /* Short work may wait behind a long dispatch; it must not wait forever. */
   uint64_t stall = 10 * ST_NS + (uint64_t)(4 * target_ms * ST_MS);
   while (st_time_left(r) && !r->errors) {
      bool idle = true;
      for (int i = 0; i < ng; i++) {
         uint64_t before = g[i].checked;
         if ((vr = st_ring_pump(&g[i], r, true, stall)))
            return st_fail(r, "%s: %s after %llu completions", g[i].name,
                           vr == VK_TIMEOUT ? "no completion within the stall limit" : st_vkr(vr),
                           (unsigned long long)g[i].checked);
         idle &= g[i].checked == before;
      }
      if (idle)
         st_sleep_ns(200 * 1000ull);
   }
   for (int i = 0; i < ng && !r->errors; i++)
      if ((vr = st_ring_drain(&g[i], r, stall)))
         return st_fail(r, "%s drain: %s", g[i].name, st_vkr(vr));
   st_metric(r, "long_dispatches", (double)g[0].checked);
   st_metric(r, "long_dispatch_alone_ms", target_ms);
   (void)ms;
   st_metric_max(r, "max_long_latency_ms", g[0].max_ms);
   for (int i = 1; i < ng; i++)
      st_metric_max(r, "max_short_latency_ms", g[i].max_ms);
   if (!r->errors) {
      for (int i = 0; i < ng; i++)
         st_ring_destroy(&g[i]);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}

static int
lane_longalu_400(struct st_run *r, void *arg)
{
   (void)arg;
   return st_lane_longalu(r, 400);
}

static int
lane_rq(struct st_run *r, void *arg)
{
   (void)arg;
   return t_rq_stress(r);
}

int
t_compute_long(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   /* Phase 1: legitimate long dispatches alone must complete with the right result. */
   {
      struct st_dev d;
      struct st_req req = {.nqueues = 1};
      VkResult vr = st_dev_create(&d, r->o->device, &req);
      if (vr)
         return st_fail(r, "device: %s", st_vkr(vr));
      struct st_lcg l;
      VkCommandBuffer cb;
      if ((vr = st_lcg_init(&d, LONG_GROUPS, &l)) || (vr = st_cb_alloc(&d, 0, &cb)))
         return st_fail(r, "setup: %s", st_vkr(vr));
      double cal_ms = 0;
      uint32_t iters = st_lcg_calibrate(&d, 0, &l, 1500, &cal_ms);
      if (!iters)
         return st_fail(r, "calibration failed");
      for (int i = 0; i < 2; i++) {
         uint32_t seed = st_rand(r);
         st_cb_begin(cb);
         st_lcg_record(cb, &l, iters, seed);
         vkEndCommandBuffer(cb);
         uint64_t t0 = st_now();
         vr = st_run_cb(&d, 0, cb, 60 * ST_NS);
         double ms = (st_now() - t0) / 1e6;
         if (vr)
            return st_fail(r, "solo long dispatch %d (about 1.5 s): %s after %.0f ms", i, st_vkr(vr), ms);
         st_metric_max(r, "solo_dispatch_ms", ms);
         char why[256];
         r->checks++;
         if (st_lcg_verify(&l, iters, seed, 16, why, sizeof(why)))
            st_fail(r, "solo long dispatch %d: %s", i, why);
         r->iterations++;
      }
      st_lcg_destroy(&d, &l);
      st_dev_destroy(&d);
      if (r->errors)
         return ST_FAIL;
   }
   /* Phase 2: long dispatches beside render and compute queues, and beside a ray-query client. */
   struct st_lane lanes[2] = {{"longalu", lane_longalu_400, NULL}, {"rq", lane_rq, NULL}};
   return st_run_lanes(r, lanes, caps.rq ? 2 : 1);
}

/* ======================= compute.starve (dangerous) ======================= */

static int
lane_longalu_solo(struct st_run *r, void *arg)
{
   (void)arg;
   struct st_dev d;
   struct st_req req = {.nqueues = 1};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_lcg cal;
   if ((vr = st_lcg_init(&d, LONG_GROUPS, &cal)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   uint32_t iters = st_lcg_calibrate(&d, 0, &cal, 1500, NULL);
   st_lcg_destroy(&d, &cal);
   if (!iters)
      return st_fail(r, "calibration failed");
   struct st_ring g;
   if ((vr = st_ring_init(&g, &d, 0, 1, ST_RING_LCG, LONG_GROUPS, iters)))
      return st_fail(r, "ring: %s", st_vkr(vr));
   while (st_time_left(r) && !r->errors) {
      if ((vr = st_ring_pump(&g, r, true, 30 * ST_NS)))
         return st_fail(r, "long dispatch: %s", vr == VK_TIMEOUT ? "no completion for 30 s" : st_vkr(vr));
      st_wait_value(&d, g.tl, g.checked + 1, 100 * ST_MS);
   }
   if ((vr = st_ring_drain(&g, r, 30 * ST_NS)))
      return st_fail(r, "long dispatch drain: %s", st_vkr(vr));
   st_metric_max(r, "max_long_latency_ms", g.max_ms);
   st_ring_destroy(&g);
   st_dev_destroy(&d);
   return st_verdict(r);
}

/* `nqueues` compute rings (and a render ring when `render`) in one process. */
int
st_lane_queues(struct st_run *r, int nqueues, bool render)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   int nq = nqueues + (render ? 1 : 0);
   if (nq > (int)caps.queues)
      nq = (int)caps.queues;
   struct st_dev d;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_ring g[ST_MAX_QUEUES];
   int ng = 0;
   for (int q = 0; q < nq; q++) {
      bool rend = render && q == nq - 1;
      if ((vr = st_ring_init(&g[ng++], &d, q, 4, rend ? ST_RING_RENDER : ST_RING_CHAIN, rend ? 256 : 16384, 0)))
         return st_fail(r, "ring %d: %s", q, st_vkr(vr));
   }
   while (st_time_left(r) && !r->errors) {
      bool idle = true;
      for (int i = 0; i < ng; i++) {
         uint64_t before = g[i].checked;
         if ((vr = st_ring_pump(&g[i], r, true, 30 * ST_NS)))
            return st_fail(r, "%s: %s", g[i].name, vr == VK_TIMEOUT ? "no completion for 30 s" : st_vkr(vr));
         idle &= g[i].checked == before;
      }
      if (idle)
         st_sleep_ns(200 * 1000ull);
   }
   for (int i = 0; i < ng && !r->errors; i++)
      if ((vr = st_ring_drain(&g[i], r, 30 * ST_NS)))
         return st_fail(r, "%s drain: %s", g[i].name, st_vkr(vr));
   for (int i = 0; i < ng; i++)
      st_metric_max(r, "max_latency_ms", g[i].max_ms);
   if (!r->errors) {
      for (int i = 0; i < ng; i++)
         st_ring_destroy(&g[i]);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}

static int
lane_queues4(struct st_run *r, void *arg)
{
   (void)arg;
   return st_lane_queues(r, 4, true);
}

/* Clients killed with SIGKILL a few milliseconds into their work. */
static int
lane_kills(struct st_run *r, void *arg)
{
   (void)arg;
   uint64_t kills = 0;
   while (st_elapsed(r) + 2 * ST_NS < r->budget_ns && !r->errors) {
      int fd[2];
      if (pipe(fd))
         return st_fail(r, "pipe");
      struct st_client_cfg cfg = st_client_defaults(r);
      cfg.seed = ((uint64_t)st_rand(r) << 32) | st_rand(r);
      cfg.render = true;
      cfg.spin = 400000;
      cfg.until_ns = st_now() + 30 * ST_NS;
      cfg.report_fd = fd[1];
      pid_t pid = st_fork(st_client_main, &cfg);
      close(fd[1]);
      char b;
      struct timeval tv = {.tv_sec = 30};
      fd_set fs;
      FD_ZERO(&fs);
      FD_SET(fd[0], &fs);
      bool started = select(fd[0] + 1, &fs, NULL, NULL, &tv) > 0 && read(fd[0], &b, 1) == 1;
      close(fd[0]);
      if (started)
         st_sleep_ns(st_rand_range(r, 0, 50000) * 1000ull);
      kill(pid, SIGKILL);
      uint64_t t0 = st_now();
      int sig = 0, rc = st_wait_pid(pid, 30 * ST_NS, &sig);
      st_metric_max(r, "max_kill_teardown_ms", (st_now() - t0) / 1e6);
      r->checks++;
      if (rc == -2)
         st_fail(r, "killed client did not exit within 30 s");
      else if (!started)
         st_fail(r, "client never started its work (exit %d)", rc);
      kills++;
      st_sleep_ns(st_rand_range(r, 50, 400) * ST_MS);
   }
   st_metric(r, "kills", (double)kills);
   return st_verdict(r);
}

int
t_compute_starve(struct st_run *r)
{
   struct st_lane lanes[3] = {
      {"long1500", lane_longalu_solo, NULL},
      {"queues4", lane_queues4, NULL},
      {"kills", lane_kills, NULL},
   };
   return st_run_lanes(r, lanes, 3);
}
