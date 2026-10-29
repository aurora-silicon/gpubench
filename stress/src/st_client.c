// SPDX-License-Identifier: GPL-2.0-only
/* A small checked GPU client: per iteration, a cross-queue producer/consumer pair, a render
 * check and a chained compute pass, every result verified. Child processes of several tests run
 * it (process churn, exit/kill with work in flight, the healthy client beside a faulting one). */
#include "st_client.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHAIN_N 4096u
#define ORD_WORDS 8u

struct client {
   struct st_dev d;
   struct st_buf buf;    /* [0, ORD_WORDS) order words, then CHAIN_N chain words */
   struct st_pipe order;
   struct st_chain chain;
   struct st_rcheck rc;
   bool have_rc;
   VkCommandBuffer prod, cons, rend;
   VkSemaphore tl, done;
   uint64_t value;
   int qa, qb;
};

static VkResult
setup(struct client *c, const struct st_client_cfg *cfg)
{
   struct st_req req = {.nqueues = cfg->nqueues > 0 ? cfg->nqueues : 2};
   struct st_caps caps;
   VkResult r = st_probe(cfg->device, &caps);
   if (r != VK_SUCCESS)
      return r;
   if ((uint32_t)req.nqueues > caps.queues)
      req.nqueues = (int)caps.queues;
   if ((r = st_dev_create(&c->d, cfg->device, &req)) != VK_SUCCESS)
      return r;
   c->qa = 0;
   c->qb = c->d.nq > 1 ? 1 : 0;
   if ((r = st_buf_create(&c->d, (ORD_WORDS + CHAIN_N) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &c->buf)))
      return r;
   if ((r = st_compute_pipe(&c->d, ST_SPV(order_comp), 1, 16, &c->order)))
      return r;
   struct st_buf *b = &c->buf;
   st_pipe_buffers(&c->d, &c->order, &b, 1);
   if ((r = st_chain_init(&c->d, &c->buf, &c->chain)))
      return r;
   if (cfg->render) {
      if ((r = st_rcheck_init(&c->d, cfg->rw ? cfg->rw : 128, cfg->rh ? cfg->rh : 128, &c->rc)))
         return r;
      c->have_rc = true;
   }
   if ((r = st_cb_alloc(&c->d, c->qa, &c->prod)) || (r = st_cb_alloc(&c->d, c->qb, &c->cons)) ||
       (r = st_cb_alloc(&c->d, c->qa, &c->rend)))
      return r;
   if ((r = st_timeline(&c->d, 0, &c->tl)) || (r = st_timeline(&c->d, 0, &c->done)))
      return r;
   return VK_SUCCESS;
}

static void
record_order(struct client *c, VkCommandBuffer cb, uint32_t role, uint32_t spin, uint32_t mark)
{
   uint32_t pc[4] = {0, role, spin, mark};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->order.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->order.layout, 0, 1, &c->order.set, 0, NULL);
   vkCmdPushConstants(cb, c->order.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, 1, 1, 1);
}

/* Submits one iteration; *done_value is the value of c->done that completes it. */
static VkResult
submit_iter(struct client *c, const struct st_client_cfg *cfg, uint32_t k, uint32_t mark, uint64_t *done_value)
{
   uint32_t *v = c->buf.map;
   memset(v, 0, ORD_WORDS * 4);
   for (uint32_t i = 0; i < CHAIN_N; i++)
      v[ORD_WORDS + i] = k ^ i;
   VkResult r;
   if ((r = st_cb_begin(c->prod)))
      return r;
   record_order(c, c->prod, 0, cfg->spin, mark);
   st_barrier(c->prod, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
   st_chain_record(c->prod, &c->chain, ORD_WORDS, CHAIN_N, k);
   if ((r = vkEndCommandBuffer(c->prod)))
      return r;
   if ((r = st_cb_begin(c->cons)))
      return r;
   record_order(c, c->cons, 1, 0, 0);
   st_barrier(c->cons, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
   st_chain_record(c->cons, &c->chain, ORD_WORDS, CHAIN_N, k + 1);
   st_barrier(c->cons, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
   if ((r = vkEndCommandBuffer(c->cons)))
      return r;
   uint64_t v1 = ++c->value;
   struct st_sem_op sig = {c->tl, v1}, wait = {c->tl, v1};
   if ((r = st_submit(c->d.q[c->qa], c->prod, NULL, 0, &sig, 1, VK_NULL_HANDLE)))
      return r;
   struct st_sem_op done = {c->done, v1 * 2};
   if ((r = st_submit(c->d.q[c->qb], c->cons, &wait, 1, &done, 1, VK_NULL_HANDLE)))
      return r;
   if (c->have_rc) {
      if ((r = st_cb_begin(c->rend)))
         return r;
      st_rcheck_record(c->rend, &c->rc, k);
      if ((r = vkEndCommandBuffer(c->rend)))
         return r;
      struct st_sem_op w2 = {c->done, v1 * 2}, s2 = {c->done, v1 * 2 + 1};
      if ((r = st_submit(c->d.q[c->qa], c->rend, &w2, 1, &s2, 1, VK_NULL_HANDLE)))
         return r;
      *done_value = v1 * 2 + 1;
   } else {
      *done_value = v1 * 2;
   }
   return VK_SUCCESS;
}

static int
verify_iter(struct client *c, uint32_t k, uint32_t mark, char *why, size_t n)
{
   const uint32_t *v = c->buf.map;
   if (v[1] != 0 || v[2] != 1 || v[3] != 2 || v[4] != 3 || v[5] != mark || v[6] != mark) {
      snprintf(why, n, "cross-queue order k=%u: tickets %u,%u/%u,%u marker %08x/%08x (expected 0,1/2,3 %08x)", k, v[1],
               v[2], v[3], v[4], v[5], v[6], mark);
      return 1;
   }
   for (uint32_t i = 0; i < CHAIN_N; i++) {
      uint32_t want = st_chain_model(st_chain_model(k ^ i, k, i), k + 1, i);
      if (v[ORD_WORDS + i] != want) {
         snprintf(why, n, "cross-queue chain k=%u element %u = %08x, expected %08x", k, i, v[ORD_WORDS + i], want);
         return 1;
      }
   }
   if (c->have_rc && st_rcheck_verify(&c->rc, k, why, n))
      return 1;
   return 0;
}

int
st_client(const struct st_client_cfg *cfg, struct st_client_result *res)
{
   memset(res, 0, sizeof(*res));
   struct client c;
   memset(&c, 0, sizeof(c));
   VkResult r = setup(&c, cfg);
   if (r != VK_SUCCESS) {
      res->vk = r;
      snprintf(res->why, sizeof(res->why), "setup: %s", st_vkr(r));
      return 2;
   }
   uint64_t timeout = cfg->timeout_ns ? cfg->timeout_ns : 10 * ST_NS;
   uint64_t seed = cfg->seed;
   for (int it = 0; cfg->iters <= 0 || it < cfg->iters; it++) {
      if (cfg->until_ns && st_now() >= cfg->until_ns && it > 0)
         break;
      seed = seed * 6364136223846793005ull + 1442695040888963407ull;
      uint32_t k = (uint32_t)(seed >> 33), mark = (uint32_t)(seed >> 7) | 1u;
      uint64_t dv;
      uint64_t t0 = st_now();
      if ((r = submit_iter(&c, cfg, k, mark, &dv))) {
         res->vk = r;
         snprintf(res->why, sizeof(res->why), "submit: %s", st_vkr(r));
         return 2;
      }
      if (cfg->report_fd >= 0 && it == 0) {
         char b = 'S';
         if (write(cfg->report_fd, &b, 1) < 0) {
         }
      }
      if (cfg->exit_after_submit) {
         /* Leave with the producer, the dependent consumer and the render in flight. */
         _exit(0);
      }
      r = st_wait_value(&c.d, c.done, dv, timeout);
      if (r != VK_SUCCESS) {
         res->vk = r;
         snprintf(res->why, sizeof(res->why), "wait for iteration %d: %s after %.1f s", it, st_vkr(r),
                  (st_now() - t0) / 1e9);
         if (r == VK_TIMEOUT)
            _exit(3); /* the device cannot be destroyed with work stuck in it */
         return 2;
      }
      double ms = (st_now() - t0) / 1e6;
      if (ms > res->max_ms)
         res->max_ms = ms;
      res->iterations++;
      if (verify_iter(&c, k, mark, res->why, sizeof(res->why))) {
         res->bad++;
         return 1;
      }
      res->checks++;
      if (cfg->exit_after_wait && (cfg->iters <= 0 || it + 1 >= cfg->iters))
         _exit(0); /* process exit right after a fence wait returns */
      if (cfg->gap_ns)
         st_sleep_ns(cfg->gap_ns);
   }
   if (cfg->no_teardown)
      return 0;
   if (c.have_rc)
      st_rcheck_destroy(&c.d, &c.rc);
   vkDestroySemaphore(c.d.dev, c.tl, NULL);
   vkDestroySemaphore(c.d.dev, c.done, NULL);
   st_chain_destroy(&c.d, &c.chain);
   st_pipe_destroy(&c.d, &c.order);
   st_buf_destroy(&c.d, &c.buf);
   if (!st_dev_destroy(&c.d)) {
      snprintf(res->why, sizeof(res->why), "device did not go idle at teardown");
      return 2;
   }
   return 0;
}

/* Child-process entry: runs the client and exits 0 (pass), 1 (wrong data), 2 (Vulkan error) or
 * 3 (GPU timeout). The result is written to cfg->result_fd when set. */
int
st_client_main(void *arg)
{
   const struct st_client_cfg *cfg = arg;
   struct st_client_result res;
   int rc = st_client(cfg, &res);
   if (cfg->result_fd >= 0) {
      res.rc = rc;
      gb_write_all(cfg->result_fd, &res, sizeof(res));
   }
   if (rc && cfg->verbose)
      fprintf(stderr, "    client %d: %s\n", (int)getpid(), res.why);
   return rc;
}
