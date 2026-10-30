// SPDX-License-Identifier: GPL-2.0-only
/* Queue-depth extremes: 128 submissions in flight on one queue; several devices with different
 * global priorities, each with mixed relative priorities on every queue. */
#include "st.h"
#include "st_ring.h"

#include <stdlib.h>
#include <string.h>

static VkResult
pump_all(struct st_run *r, struct st_ring *g, int n, uint64_t until, uint64_t stall)
{
   VkResult vr;
   while (st_now() < until && !r->errors) {
      bool idle = true;
      for (int i = 0; i < n; i++) {
         uint64_t before = g[i].checked;
         if ((vr = st_ring_pump(&g[i], r, true, stall)))
            return vr;
         idle &= g[i].checked == before;
      }
      if (idle)
         st_sleep_ns(100 * 1000ull);
   }
   for (int i = 0; i < n && !r->errors; i++)
      if ((vr = st_ring_drain(&g[i], r, stall)))
         return vr;
   return VK_SUCCESS;
}

int
t_queue_depth(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   VkResult vr;
   /* Phase 1: deep rings on one queue. */
   {
      struct st_dev d;
      struct st_req req = {.nqueues = 1};
      if ((vr = st_dev_create(&d, r->o->device, &req)))
         return st_fail(r, "device: %s", st_vkr(vr));
      struct st_ring g[2];
      if ((vr = st_ring_init(&g[0], &d, 0, 128, ST_RING_CHAIN, 256, 0)) ||
          (vr = st_ring_init(&g[1], &d, 0, 64, ST_RING_RENDER, 64, 64)))
         return st_fail(r, "deep rings: %s", st_vkr(vr));
      vr = pump_all(r, g, 2, st_now() + st_part(r, 0.4, 2 * ST_NS), 15 * ST_NS);
      if (vr)
         return st_fail(r, "depth 128+64 on one queue: %s", vr == VK_TIMEOUT ? "no completion for 15 s" : st_vkr(vr));
      st_metric(r, "deep_completions", (double)(g[0].checked + g[1].checked));
      st_metric_max(r, "deep_max_latency_ms", g[0].max_ms > g[1].max_ms ? g[0].max_ms : g[1].max_ms);
      if (r->errors)
         return ST_FAIL;
      st_ring_destroy(&g[0]);
      st_ring_destroy(&g[1]);
      st_dev_destroy(&d);
   }
   /* Phase 2: three devices (low, medium, high global priority when supported), every queue
    * busy with relative priorities 0, 0.5 and 1 in turn. */
   static const int levels[3] = {VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
                                 VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR};
   int nq = caps.queues < 4 ? (int)caps.queues : 4;
   struct st_dev d[3];
   struct st_ring g[12];
   int ng = 0, granted = 0;
   for (int i = 0; i < 3; i++) {
      struct st_req req = {.nqueues = nq, .mixed = true, .gprio = caps.gprio ? levels[i] : 0};
      if ((vr = st_dev_create(&d[i], r->o->device, &req)))
         return st_fail(r, "device %d: %s", i, st_vkr(vr));
      granted += d[i].gprio;
      for (int q = 0; q < nq; q++)
         if ((vr = st_ring_init(&g[ng++], &d[i], q, 16, (q + i) % 2 ? ST_RING_RENDER : ST_RING_CHAIN,
                                (q + i) % 2 ? 128 : 16384, 0)))
            return st_fail(r, "ring: %s", st_vkr(vr));
   }
   vr = pump_all(r, g, ng, r->t0 + r->budget_ns, 15 * ST_NS);
   if (vr)
      return st_fail(r, "%d queues with mixed priorities: %s", ng, vr == VK_TIMEOUT ? "a queue starved for 15 s" : st_vkr(vr));
   double minc = 1e18, maxc = 0;
   for (int i = 0; i < ng; i++) {
      if (g[i].checked < minc)
         minc = (double)g[i].checked;
      if (g[i].checked > maxc)
         maxc = (double)g[i].checked;
      st_metric_max(r, "mixed_max_latency_ms", g[i].max_ms);
   }
   st_metric(r, "mixed_queues", ng);
   st_metric(r, "global_priorities_granted", granted);
   st_metric(r, "mixed_min_completions", minc);
   st_metric(r, "mixed_max_completions", maxc);
   if (!r->errors) {
      for (int i = 0; i < ng; i++)
         st_ring_destroy(&g[i]);
      for (int i = 0; i < 3; i++)
         st_dev_destroy(&d[i]);
   }
   return st_verdict(r);
}
