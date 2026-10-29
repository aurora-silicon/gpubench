// SPDX-License-Identifier: GPL-2.0-only
#include "st_ring.h"

#include <stdlib.h>
#include <string.h>

VkResult
st_ring_init(struct st_ring *g, struct st_dev *d, int qi, int depth, enum st_ring_kind kind, uint32_t a, uint32_t b)
{
   memset(g, 0, sizeof(*g));
   g->d = d;
   g->qi = qi;
   g->depth = depth > ST_RING_MAX ? ST_RING_MAX : depth;
   g->kind = kind;
   snprintf(g->name, sizeof(g->name), "%s q%d d%d",
            kind == ST_RING_RENDER ? "render" : kind == ST_RING_CHAIN ? "compute" : "long-alu", qi, g->depth);
   VkResult r = st_timeline(d, 0, &g->tl);
   if (r)
      return r;
   for (int i = 0; i < g->depth; i++)
      if ((r = st_cb_alloc(d, qi, &g->cb[i])))
         return r;
   switch (kind) {
   case ST_RING_RENDER:
      g->rc = calloc(g->depth, sizeof(*g->rc));
      for (int i = 0; i < g->depth; i++)
         if ((r = st_rcheck_init(d, a, b ? b : a, &g->rc[i])))
            return r;
      break;
   case ST_RING_CHAIN:
      g->chain_n = (a + 63) & ~63u;
      if ((r = st_buf_create(d, (VkDeviceSize)g->chain_n * g->depth * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                             &g->chain_buf)))
         return r;
      if ((r = st_chain_init(d, &g->chain_buf, &g->chain)))
         return r;
      break;
   case ST_RING_LCG:
      g->lcg = calloc(g->depth, sizeof(*g->lcg));
      g->lcg_iters = b;
      for (int i = 0; i < g->depth; i++)
         if ((r = st_lcg_init(d, a, &g->lcg[i])))
            return r;
      break;
   }
   g->last_progress = st_now();
   return VK_SUCCESS;
}

static void
record(struct st_ring *g, VkCommandBuffer cb, int slot, uint32_t k)
{
   switch (g->kind) {
   case ST_RING_RENDER:
      st_rcheck_record(cb, &g->rc[slot], k);
      break;
   case ST_RING_CHAIN: {
      uint32_t *v = (uint32_t *)g->chain_buf.map + (size_t)slot * g->chain_n;
      for (uint32_t i = 0; i < g->chain_n; i++)
         v[i] = k ^ i;
      st_chain_record(cb, &g->chain, slot * g->chain_n, g->chain_n, k);
      st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                 VK_ACCESS_2_HOST_READ_BIT);
      break;
   }
   case ST_RING_LCG:
      st_lcg_record(cb, &g->lcg[slot], g->lcg_iters, k);
      break;
   }
}

static uint64_t
verify(struct st_ring *g, int slot, uint32_t k, char *why, size_t n)
{
   switch (g->kind) {
   case ST_RING_RENDER:
      return st_rcheck_verify(&g->rc[slot], k, why, n);
   case ST_RING_CHAIN: {
      const uint32_t *v = (const uint32_t *)g->chain_buf.map + (size_t)slot * g->chain_n;
      for (uint32_t i = 0; i < g->chain_n; i++)
         if (v[i] != st_chain_model(k ^ i, k, i)) {
            snprintf(why, n, "compute k=%u element %u = %08x, expected %08x", k, i, v[i], st_chain_model(k ^ i, k, i));
            return 1;
         }
      return 0;
   }
   case ST_RING_LCG:
      return st_lcg_verify(&g->lcg[slot], g->lcg_iters, k, 6, why, n);
   }
   return 0;
}

VkResult
st_ring_pump(struct st_ring *g, struct st_run *r, bool submit, uint64_t stall_ns)
{
   uint64_t c = 0;
   VkResult vr = vkGetSemaphoreCounterValue(g->d->dev, g->tl, &c);
   if (vr)
      return vr;
   uint64_t now = st_now();
   while (g->checked < c) {
      int slot = (int)(g->checked % g->depth);
      char why[256] = "";
      double ms = (now - g->submit_ns[slot]) / 1e6;
      if (ms > g->max_ms)
         g->max_ms = ms;
      r->checks++;
      if (verify(g, slot, g->k[slot], why, sizeof(why)))
         st_fail(r, "%s, submission %llu: %s", g->name, (unsigned long long)g->checked + 1, why);
      g->checked++;
      g->last_progress = now;
   }
   while (submit && g->next - g->checked < (uint64_t)g->depth) {
      int slot = (int)(g->next % g->depth);
      uint32_t k = st_rand(r);
      VkCommandBuffer cb = g->cb[slot];
      if ((vr = st_cb_begin(cb)))
         return vr;
      record(g, cb, slot, k);
      if ((vr = vkEndCommandBuffer(cb)))
         return vr;
      struct st_sem_op s = {g->tl, g->next + 1};
      g->k[slot] = k;
      g->submit_ns[slot] = st_now();
      if ((vr = st_submit(g->d->q[g->qi], cb, NULL, 0, &s, 1, VK_NULL_HANDLE)))
         return vr;
      if (g->next == g->checked)
         g->last_progress = st_now(); /* the clock starts when the ring becomes busy */
      g->next++;
      r->iterations++;
   }
   if (g->checked < g->next && st_now() - g->last_progress > stall_ns)
      return VK_TIMEOUT;
   return VK_SUCCESS;
}

VkResult
st_ring_drain(struct st_ring *g, struct st_run *r, uint64_t timeout_ns)
{
   while (st_ring_busy(g)) {
      VkResult vr = st_ring_pump(g, r, false, timeout_ns);
      if (vr)
         return vr;
      if (st_ring_busy(g)) {
         uint64_t want = g->checked + 1;
         st_wait_value(g->d, g->tl, want, 50 * ST_MS);
      }
   }
   return VK_SUCCESS;
}

void
st_ring_destroy(struct st_ring *g)
{
   struct st_dev *d = g->d;
   if (!d)
      return;
   if (g->rc) {
      for (int i = 0; i < g->depth; i++)
         if (g->rc[i].pipe.pipe)
            st_rcheck_destroy(d, &g->rc[i]);
      free(g->rc);
   }
   if (g->lcg) {
      for (int i = 0; i < g->depth; i++)
         st_lcg_destroy(d, &g->lcg[i]);
      free(g->lcg);
   }
   if (g->chain.pipe.pipe)
      st_chain_destroy(d, &g->chain);
   if (g->chain_buf.buf)
      st_buf_destroy(d, &g->chain_buf);
   for (int i = 0; i < g->depth; i++)
      if (g->cb[i])
         vkFreeCommandBuffers(d->dev, d->pool[g->qi], 1, &g->cb[i]);
   if (g->tl)
      vkDestroySemaphore(d->dev, g->tl, NULL);
   memset(g, 0, sizeof(*g));
}
