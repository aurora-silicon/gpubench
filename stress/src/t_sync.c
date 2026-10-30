// SPDX-License-Identifier: GPL-2.0-only
/* Dense synchronisation (random dependency graphs over many semaphores and fences, pending
 * work cancelled at process exit) and idle/wake transitions. */
#include "st.h"
#include "st_ring.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ======================= sync.dense ======================= */

#define SD_NODES 384
#define SD_DEPS 3

struct sd_node {
   int q;
   int ndep;
   int dep[SD_DEPS + 1];
   bool bin_in, bin_out; /* binary edge from node - 1 / to node + 1 */
   uint32_t depth;
};

struct sd {
   struct st_dev d;
   struct st_buf buf;
   struct st_pipe pipe;
   VkCommandBuffer cb[SD_NODES];
   VkSemaphore tl[SD_NODES], bin[SD_NODES];
   VkFence fence[SD_NODES];
   uint64_t semaphores_created;
};

static VkResult
sd_semaphores(struct sd *s, bool recreate)
{
   VkResult vr;
   for (int i = 0; i < SD_NODES; i++) {
      if (recreate) {
         vkDestroySemaphore(s->d.dev, s->tl[i], NULL);
         vkDestroySemaphore(s->d.dev, s->bin[i], NULL);
      }
      if ((vr = st_timeline(&s->d, 0, &s->tl[i])) || (vr = st_binary(&s->d, &s->bin[i])))
         return vr;
      s->semaphores_created += 2;
   }
   return VK_SUCCESS;
}

static void
sd_record(struct sd *s, VkCommandBuffer cb, int i, const struct sd_node *n)
{
   uint32_t pc[6] = {(uint32_t)i, (uint32_t)n->ndep, 0, 0, 0, 0};
   for (int k = 0; k < n->ndep; k++)
      pc[2 + k] = (uint32_t)n->dep[k];
   st_cb_begin(cb);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe.layout, 0, 1, &s->pipe.set, 0, NULL);
   vkCmdPushConstants(cb, s->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, 1, 1, 1);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
   vkEndCommandBuffer(cb);
}

static VkResult
sd_submit(struct sd *s, int i, const struct sd_node *n, uint64_t round)
{
   struct st_sem_op w[SD_DEPS + 1], sig[2];
   int nw = 0, ns = 0;
   for (int k = 0; k < n->ndep; k++)
      if (!(n->bin_in && n->dep[k] == i - 1))
         w[nw++] = (struct st_sem_op){s->tl[n->dep[k]], round};
   if (n->bin_in)
      w[nw++] = (struct st_sem_op){s->bin[i - 1], 0};
   sig[ns++] = (struct st_sem_op){s->tl[i], round};
   if (n->bin_out)
      sig[ns++] = (struct st_sem_op){s->bin[i], 0};
   return st_submit(s->d.q[n->q], s->cb[i], w, nw, sig, ns, s->fence[i]);
}

struct sd_exit_arg {
   int device;
   int nq;
};

/* Child: a long chain blocked on a semaphore nobody signals, across every queue, then exit. */
static int
sd_pending_child(void *arg)
{
   struct sd_exit_arg *a = arg;
   struct st_dev d;
   struct st_req req = {.nqueues = a->nq};
   if (st_dev_create(&d, a->device, &req))
      return 2;
   VkSemaphore gate, tl;
   VkCommandBuffer cb;
   if (st_timeline(&d, 0, &gate) || st_timeline(&d, 0, &tl) || st_cb_alloc(&d, 0, &cb))
      return 2;
   st_cb_begin(cb);
   vkEndCommandBuffer(cb);
   struct st_sem_op w = {gate, 1}, s = {tl, 1};
   if (st_submit(d.q[0], cb, &w, 1, &s, 1, VK_NULL_HANDLE))
      return 2;
   for (uint64_t v = 1; v < 64; v++) {
      VkCommandBuffer c;
      if (st_cb_alloc(&d, 0, &c))
         return 2;
      st_cb_begin(c);
      vkEndCommandBuffer(c);
      struct st_sem_op w2 = {tl, v}, s2 = {tl, v + 1};
      VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VkFence f;
      vkCreateFence(d.dev, &fci, NULL, &f);
      if (st_submit(d.q[v % d.nq], c, &w2, 1, &s2, 1, f))
         return 2;
   }
   _exit(0); /* everything above is still pending */
}

int
t_sync_dense(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   struct sd *s = calloc(1, sizeof(*s));
   int nq = caps.queues < 4 ? (int)caps.queues : 4;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&s->d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   if ((vr = st_buf_create(&s->d, SD_NODES * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &s->buf)) ||
       (vr = st_compute_pipe(&s->d, ST_SPV(link_comp), 1, 24, &s->pipe)) || (vr = sd_semaphores(s, false)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   struct st_buf *bp = &s->buf;
   st_pipe_buffers(&s->d, &s->pipe, &bp, 1);
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   for (int i = 0; i < SD_NODES; i++)
      if ((vr = st_cb_alloc(&s->d, 0, &s->cb[i])) || (vr = vkCreateFence(s->d.dev, &fci, NULL, &s->fence[i])))
         return st_fail(r, "setup: %s", st_vkr(vr));

   struct sd_node *n = calloc(SD_NODES, sizeof(*n));
   int order[SD_NODES];
   uint64_t round = 0, waited = 0, pending_exits = 0;
   uint64_t next_exit = 0;
   while (st_time_left(r) && !r->errors) {
      /* Now and then a child leaves pending work behind for the kernel to cancel. */
      if (st_elapsed(r) >= next_exit) {
         struct sd_exit_arg a = {r->o->device, nq};
         int sig = 0, rc = st_wait_pid(st_fork(sd_pending_child, &a), 30 * ST_NS, &sig);
         st_log(r, "pending-work child exited: %d (signal %d)", rc, sig);
         r->checks++;
         if (rc != 0)
            st_fail(r, "process exit with 64 blocked submissions: %s %d", rc == -2 ? "did not exit within 30 s" : "exit",
                    rc);
         pending_exits++;
         next_exit += r->budget_ns / 4;
      }
      round++;
      if (round % 8 == 0 && (vr = sd_semaphores(s, true)))
         return st_fail(r, "semaphore churn: %s", st_vkr(vr));
      if (round % 8 == 0)
         round = 1; /* fresh semaphores start at 0 again */
      for (int i = 0; i < SD_NODES; i++) {
         struct sd_node *x = &n[i];
         memset(x, 0, sizeof(*x));
         x->q = (int)st_rand_range(r, 0, nq - 1);
         int nd = i ? (int)st_rand_range(r, 0, SD_DEPS) : 0;
         for (int k = 0; k < nd; k++) {
            int dep = (int)st_rand_range(r, i > 16 ? i - 16 : 0, i - 1);
            bool dup = false;
            for (int m = 0; m < x->ndep; m++)
               dup |= x->dep[m] == dep;
            if (!dup)
               x->dep[x->ndep++] = dep;
         }
         if (i && !n[i - 1].bin_out && st_rand_range(r, 0, 3) == 0) {
            n[i - 1].bin_out = true;
            x->bin_in = true;
            bool have = false;
            for (int m = 0; m < x->ndep; m++)
               have |= x->dep[m] == i - 1;
            if (!have)
               x->dep[x->ndep++] = i - 1;
         }
         uint32_t depth = 0;
         for (int m = 0; m < x->ndep; m++)
            if (n[x->dep[m]].depth > depth)
               depth = n[x->dep[m]].depth;
         x->depth = depth + 1;
      }
      memset(s->buf.map, 0, s->buf.size);
      if ((vr = vkResetFences(s->d.dev, SD_NODES, s->fence)))
         return st_fail(r, "reset fences: %s", st_vkr(vr));
      for (int i = 0; i < SD_NODES; i++)
         sd_record(s, s->cb[i], i, &n[i]);
      /* Each queue runs its nodes in index order (a queue executes in order, so a later
       * submission must never be what an earlier one on the same queue waits for). The queues'
       * streams are merged at random, so many waits are submitted before the signal they wait
       * for on another queue. A binary wait goes in only once its signal and everything that
       * signal depends on are submitted (every lower-indexed node), as Vulkan requires. */
      int head[ST_MAX_QUEUES], no = 0, prefix = 0;
      bool submitted[SD_NODES];
      memset(submitted, 0, sizeof(submitted));
      for (int q = 0; q < nq; q++) {
         head[q] = 0;
         while (head[q] < SD_NODES && n[head[q]].q != q)
            head[q]++;
      }
      while (no < SD_NODES) {
         int cand[ST_MAX_QUEUES], nc = 0;
         for (int q = 0; q < nq; q++)
            if (head[q] < SD_NODES && (!n[head[q]].bin_in || prefix >= head[q]))
               cand[nc++] = q;
         int q = cand[st_rand_range(r, 0, nc - 1)];
         int i = head[q];
         order[no++] = i;
         submitted[i] = true;
         while (prefix < SD_NODES && submitted[prefix])
            prefix++;
         do
            head[q]++;
         while (head[q] < SD_NODES && n[head[q]].q != q);
      }
      st_log(r, "round %llu: submitting", (unsigned long long)round);
      for (int k = 0; k < SD_NODES; k++)
         if ((vr = sd_submit(s, order[k], &n[order[k]], round)))
            return st_fail(r, "round %llu node %d: submit %s", (unsigned long long)round, order[k], st_vkr(vr));
      st_log(r, "round %llu: %d submissions in", (unsigned long long)round, SD_NODES);
      /* Wait for any of a few, then for all fences. */
      VkSemaphore any[8];
      uint64_t anyv[8];
      for (int k = 0; k < 8; k++) {
         int i = (int)st_rand_range(r, 0, SD_NODES - 1);
         any[k] = s->tl[i];
         anyv[k] = round;
      }
      VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                .flags = VK_SEMAPHORE_WAIT_ANY_BIT, .semaphoreCount = 8,
                                .pSemaphores = any, .pValues = anyv};
      if ((vr = vkWaitSemaphores(s->d.dev, &wi, 10 * ST_NS)))
         return st_fail(r, "round %llu: wait-any %s", (unsigned long long)round, st_vkr(vr));
      st_log(r, "round %llu: wait-any done", (unsigned long long)round);
      uint64_t t0 = st_now();
      if ((vr = vkWaitForFences(s->d.dev, SD_NODES, s->fence, VK_TRUE, 20 * ST_NS)))
         return st_fail(r, "round %llu: waiting for %d fences: %s", (unsigned long long)round, SD_NODES, st_vkr(vr));
      st_metric_max(r, "max_round_ms", (st_now() - t0) / 1e6);
      waited += SD_NODES;
      const uint32_t *b = s->buf.map;
      for (int i = 0; i < SD_NODES; i++) {
         r->checks++;
         if (b[i] != n[i].depth) {
            st_fail(r, "round %llu node %d (queue %d, %d deps%s): depth %u, expected %u: a dependency did not order",
                    (unsigned long long)round, i, n[i].q, n[i].ndep, n[i].bin_in ? ", binary" : "", b[i], n[i].depth);
            break;
         }
      }
      r->iterations++;
   }
   st_metric(r, "fences_waited", (double)waited);
   st_metric(r, "semaphores_created", (double)s->semaphores_created);
   st_metric(r, "pending_exits", (double)pending_exits);
   free(n);
   if (!r->errors)
      st_dev_destroy(&s->d);
   return st_verdict(r);
}

/* ======================= idle.wake ======================= */

int
t_idle_wake(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   struct st_dev d;
   struct st_req req = {.nqueues = caps.queues > 1 ? 2 : 1};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_ring rend, comp;
   if ((vr = st_ring_init(&rend, &d, 0, 2, ST_RING_RENDER, 512, 512)) ||
       (vr = st_ring_init(&comp, &d, d.nq - 1, 2, ST_RING_CHAIN, 262144, 0)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   static const uint32_t gaps[] = {2, 20, 200};
   double maxms[3] = {0};
   uint64_t bursts = 0;
   while (st_time_left(r) && !r->errors) {
      int gi = (int)st_rand_range(r, 0, 2);
      st_sleep_ns(gaps[gi] * ST_MS);
      uint64_t t0 = st_now();
      /* A burst: render and compute together, sometimes several back to back. */
      int n = (int)st_rand_range(r, 1, 3);
      for (int k = 0; k < n && !r->errors; k++) {
         if ((vr = st_ring_pump(&rend, r, true, 2 * ST_NS)) || (vr = st_ring_pump(&comp, r, true, 2 * ST_NS)))
            return st_fail(r, "burst after %u ms idle: %s", gaps[gi], st_vkr(vr));
         if ((vr = st_ring_drain(&rend, r, 2 * ST_NS)) || (vr = st_ring_drain(&comp, r, 2 * ST_NS)))
            return st_fail(r, "burst after %u ms idle: %s (no completion within 2 s of waking)", gaps[gi],
                           vr == VK_TIMEOUT ? "timeout" : st_vkr(vr));
      }
      double ms = (st_now() - t0) / 1e6;
      if (ms > maxms[gi])
         maxms[gi] = ms;
      bursts++;
   }
   st_metric(r, "max_burst_ms_after_2ms", maxms[0]);
   st_metric(r, "max_burst_ms_after_20ms", maxms[1]);
   st_metric(r, "max_burst_ms_after_200ms", maxms[2]);
   st_metric(r, "bursts", (double)bursts);
   if (!r->errors) {
      st_ring_destroy(&rend);
      st_ring_destroy(&comp);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}
