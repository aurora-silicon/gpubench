// SPDX-License-Identifier: GPL-2.0-only
/* Cross-queue ordering, chained render/compute data, and exit with cross-queue work in flight. */
#include "st.h"
#include "st_client.h"

#include <signal.h>
#include <stdlib.h>
#include <sys/select.h>
#include <string.h>
#include <unistd.h>

/* ======================= xq.order ======================= */

#define ORD_SLOTS 16
#define ORD_STRIDE 8

enum ord_mode { ORD_TIMELINE, ORD_BINARY, ORD_WAIT_FIRST, ORD_HOST, ORD_MODES };
static const char *const ord_names[] = {"timeline", "binary", "wait_before_signal", "host_signal"};

struct ord {
   struct st_dev d;
   struct st_buf buf;
   struct st_pipe pipe;
   VkCommandBuffer prod[ST_MAX_QUEUES][ORD_SLOTS], cons[ST_MAX_QUEUES][ORD_SLOTS];
   VkSemaphore tl[ORD_SLOTS], bin[ORD_SLOTS], host[ORD_SLOTS], done[ORD_SLOTS];
   uint64_t tlv[ORD_SLOTS], hostv[ORD_SLOTS], donev[ORD_SLOTS];
};

static void
ord_record(struct ord *s, VkCommandBuffer cb, uint32_t base, uint32_t role, uint32_t spin, uint32_t mark, bool host)
{
   st_cb_begin(cb);
   if (host)
      st_barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);
   uint32_t pc[4] = {base, role, spin, mark};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipe.layout, 0, 1, &s->pipe.set, 0, NULL);
   vkCmdPushConstants(cb, s->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, 1, 1, 1);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
   vkEndCommandBuffer(cb);
}

int
t_xq_order(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (caps.queues < 2)
      return st_skip(r, "needs 2 queues, the device has %u", caps.queues);
   struct ord *s = calloc(1, sizeof(*s));
   int nq = caps.queues < 4 ? (int)caps.queues : 4;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&s->d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   if ((vr = st_buf_create(&s->d, ORD_SLOTS * ORD_STRIDE * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &s->buf)) ||
       (vr = st_compute_pipe(&s->d, ST_SPV(order_comp), 1, 16, &s->pipe)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   struct st_buf *bp = &s->buf;
   st_pipe_buffers(&s->d, &s->pipe, &bp, 1);
   for (int q = 0; q < nq; q++)
      for (int i = 0; i < ORD_SLOTS; i++)
         if ((vr = st_cb_alloc(&s->d, q, &s->prod[q][i])) || (vr = st_cb_alloc(&s->d, q, &s->cons[q][i])))
            return st_fail(r, "command buffers: %s", st_vkr(vr));
   for (int i = 0; i < ORD_SLOTS; i++)
      if ((vr = st_timeline(&s->d, 0, &s->tl[i])) || (vr = st_binary(&s->d, &s->bin[i])) ||
          (vr = st_timeline(&s->d, 0, &s->host[i])) || (vr = st_timeline(&s->d, 0, &s->done[i])))
         return st_fail(r, "semaphores: %s", st_vkr(vr));

   uint32_t *v = s->buf.map;
   uint64_t per_mode[ORD_MODES] = {0};
   for (uint64_t batch = 0; st_time_left(r) && !r->errors; batch++) {
      enum ord_mode mode = (enum ord_mode)(batch % ORD_MODES);
      int qa = (int)st_rand_range(r, 0, nq - 1), qb = (qa + 1 + (int)st_rand_range(r, 0, nq - 2)) % nq;
      uint32_t mark[ORD_SLOTS];
      memset(v, 0, s->buf.size);
      for (int i = 0; i < ORD_SLOTS; i++) {
         mark[i] = st_rand(r) | 1u;
         uint32_t spin = st_rand_range(r, 0, 3) ? st_rand_range(r, 0, 200000) : 0;
         ord_record(s, s->prod[qa][i], i * ORD_STRIDE, 0, spin, mark[i], false);
         ord_record(s, s->cons[qb][i], i * ORD_STRIDE, mode == ORD_HOST ? 2 : 1, 0, 0, mode == ORD_HOST);
      }
      VkResult sr = VK_SUCCESS;
      for (int i = 0; i < ORD_SLOTS && !sr; i++) {
         uint64_t tv = ++s->tlv[i], dv = ++s->donev[i];
         struct st_sem_op done = {s->done[i], dv};
         if (mode == ORD_TIMELINE) {
            struct st_sem_op sig = {s->tl[i], tv};
            sr = st_submit(s->d.q[qa], s->prod[qa][i], NULL, 0, &sig, 1, VK_NULL_HANDLE);
            if (!sr)
               sr = st_submit(s->d.q[qb], s->cons[qb][i], &sig, 1, &done, 1, VK_NULL_HANDLE);
         } else if (mode == ORD_BINARY) {
            struct st_sem_op b = {s->bin[i], 0};
            sr = st_submit(s->d.q[qa], s->prod[qa][i], NULL, 0, &b, 1, VK_NULL_HANDLE);
            if (!sr)
               sr = st_submit(s->d.q[qb], s->cons[qb][i], &b, 1, &done, 1, VK_NULL_HANDLE);
         } else if (mode == ORD_WAIT_FIRST) {
            struct st_sem_op w = {s->tl[i], tv};
            sr = st_submit(s->d.q[qb], s->cons[qb][i], &w, 1, &done, 1, VK_NULL_HANDLE);
         } else {
            struct st_sem_op w = {s->host[i], ++s->hostv[i]};
            sr = st_submit(s->d.q[qb], s->cons[qb][i], &w, 1, &done, 1, VK_NULL_HANDLE);
         }
      }
      if (!sr && mode == ORD_WAIT_FIRST) {
         /* Producers after every consumer, in reverse order, some after a pause. */
         for (int i = ORD_SLOTS - 1; i >= 0 && !sr; i--) {
            if (st_rand_range(r, 0, 3) == 0)
               st_sleep_ns(st_rand_range(r, 0, 2000) * 1000ull);
            struct st_sem_op sig = {s->tl[i], s->tlv[i]};
            sr = st_submit(s->d.q[qa], s->prod[qa][i], NULL, 0, &sig, 1, VK_NULL_HANDLE);
         }
      }
      if (!sr && mode == ORD_HOST) {
         for (int i = 0; i < ORD_SLOTS && !sr; i++) {
            if (st_rand_range(r, 0, 3) == 0)
               st_sleep_ns(st_rand_range(r, 0, 2000) * 1000ull);
            v[i * ORD_STRIDE + 5] = mark[i];
            VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = s->host[i],
                                        .value = s->hostv[i]};
            sr = vkSignalSemaphore(s->d.dev, &si);
         }
      }
      if (sr)
         return st_fail(r, "%s batch %llu: submit/signal %s", ord_names[mode], (unsigned long long)batch, st_vkr(sr));
      VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = ORD_SLOTS,
                                .pSemaphores = s->done, .pValues = s->donev};
      uint64_t t0 = st_now();
      VkResult wr = vkWaitSemaphores(s->d.dev, &wi, 10 * ST_NS);
      st_metric_max(r, "max_batch_ms", (st_now() - t0) / 1e6);
      if (wr != VK_SUCCESS) /* the device is left to process exit */
         return st_fail(r, "%s batch %llu: wait %s", ord_names[mode], (unsigned long long)batch, st_vkr(wr));
      for (int i = 0; i < ORD_SLOTS; i++) {
         const uint32_t *w = v + i * ORD_STRIDE;
         r->checks++;
         if (mode == ORD_HOST) {
            if (w[6] != mark[i])
               st_fail(r, "host_signal batch %llu slot %d: consumer read %08x, host wrote %08x before the signal",
                       (unsigned long long)batch, i, w[6], mark[i]);
         } else if (w[1] != 0 || w[2] != 1 || w[3] != 2 || w[4] != 3 || w[5] != mark[i] || w[6] != mark[i] ||
                    w[0] != 4) {
            st_fail(r,
                    "%s batch %llu slot %d (queue %d -> %d): producer tickets %u,%u consumer %u,%u marker %08x "
                    "copied %08x (expected 0,1 2,3 %08x)",
                    ord_names[mode], (unsigned long long)batch, i, qa, qb, w[1], w[2], w[3], w[4], w[5], w[6], mark[i]);
         }
      }
      per_mode[mode]++;
      r->iterations++;
   }
   for (int m = 0; m < ORD_MODES; m++) {
      char n[40];
      snprintf(n, sizeof(n), "%s_batches", ord_names[m]);
      st_metric(r, n, (double)per_mode[m]);
   }
   st_metric(r, "pairs_per_batch", ORD_SLOTS);
   if (!r->errors)
      st_dev_destroy(&s->d);
   return st_verdict(r);
}

/* ======================= xq.chain ======================= */

#define XC_W 128u
#define XC_H 128u
#define XC_N (XC_W * XC_H)
#define XC_MAX_LINKS 16

struct xpc {
   uint32_t p[4];
   float color[4];
};

int
t_xq_chain(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (caps.queues < 2)
      return st_skip(r, "needs 2 queues, the device has %u", caps.queues);
   struct st_dev d;
   int nq = caps.queues < 4 ? (int)caps.queues : 4;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_buf data;
   struct st_img img;
   struct st_chain chain;
   struct st_pipe gp;
   if ((vr = st_buf_create(&d, XC_N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                           &data)) ||
       (vr = st_img_create(&d, XC_W, XC_H, VK_FORMAT_R32_UINT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &img)) ||
       (vr = st_chain_init(&d, &data, &chain)) ||
       (vr = st_graphics_pipe(&d, ST_SPV(fill_vert), ST_SPV(xdata_frag), VK_FORMAT_R32_UINT, ST_BLEND_NONE, 1,
                              sizeof(struct xpc), &gp)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   struct st_buf *bp = &data;
   st_pipe_buffers(&d, &gp, &bp, 1);
   VkCommandBuffer cbs[ST_MAX_QUEUES][XC_MAX_LINKS];
   for (int q = 0; q < nq; q++)
      for (int i = 0; i < XC_MAX_LINKS; i++)
         if ((vr = st_cb_alloc(&d, q, &cbs[q][i])))
            return st_fail(r, "command buffers: %s", st_vkr(vr));
   VkSemaphore tl;
   if ((vr = st_timeline(&d, 0, &tl)))
      return st_fail(r, "timeline: %s", st_vkr(vr));
   uint32_t *v = data.map;
   uint32_t *model = malloc(XC_N * 4);
   uint64_t value = 0, links = 0, renders = 0;
   while (st_time_left(r) && !r->errors) {
      int nlinks = (int)st_rand_range(r, 2, XC_MAX_LINKS);
      int qr = (int)st_rand_range(r, 0, nq - 1), qc = (qr + 1 + (int)st_rand_range(r, 0, nq - 2)) % nq;
      uint32_t seed = st_rand(r);
      for (uint32_t i = 0; i < XC_N; i++)
         model[i] = v[i] = seed ^ (i * 2654435761u);
      uint64_t first = value;
      VkResult sr = VK_SUCCESS;
      for (int l = 0; l < nlinks && !sr; l++) {
         bool render = (l % 2 == 0) ? st_rand_range(r, 0, 4) != 0 : st_rand_range(r, 0, 4) == 0;
         uint32_t k = st_rand(r);
         int q = render ? qr : qc;
         VkCommandBuffer cb = cbs[q][l];
         st_cb_begin(cb);
         if (render) {
            st_img_barrier(cb, img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            st_begin_render(cb, &img, VK_ATTACHMENT_LOAD_OP_DONT_CARE, NULL);
            struct xpc pc = {{0, XC_W, k, 0}, {0, 0, 0, 0}};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gp.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gp.layout, 0, 1, &gp.set, 0, NULL);
            vkCmdPushConstants(cb, gp.layout, gp.stages, 0, sizeof(pc), &pc);
            vkCmdDraw(cb, 3, 1, 0, 0);
            vkCmdEndRendering(cb);
            /* The copy overwrites what the fragment shader read: write-after-read. */
            st_barrier(cb, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            st_img_barrier(cb, img.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                     .imageExtent = {XC_W, XC_H, 1}};
            vkCmdCopyImageToBuffer(cb, img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data.buf, 1, &reg);
            st_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                       VK_ACCESS_2_HOST_READ_BIT);
            for (uint32_t i = 0; i < XC_N; i++)
               model[i] = model[i] * 5u + k + (i ^ 0x55u);
            renders++;
         } else {
            st_chain_record(cb, &chain, 0, XC_N, k);
            st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            for (uint32_t i = 0; i < XC_N; i++)
               model[i] = st_chain_model(model[i], k, i);
         }
         vkEndCommandBuffer(cb);
         struct st_sem_op w = {tl, value}, s = {tl, value + 1};
         sr = st_submit(d.q[q], cb, l ? &w : NULL, l ? 1 : 0, &s, 1, VK_NULL_HANDLE);
         value++;
         links++;
      }
      if (sr)
         return st_fail(r, "submit: %s", st_vkr(sr));
      uint64_t t0 = st_now();
      VkResult wr = st_wait_value(&d, tl, value, 10 * ST_NS);
      st_metric_max(r, "max_chain_ms", (st_now() - t0) / 1e6);
      if (wr)
         return st_fail(r, "chain of %d links (values %llu..%llu): wait %s", nlinks, (unsigned long long)first + 1,
                        (unsigned long long)value, st_vkr(wr));
      for (uint32_t i = 0; i < XC_N; i++)
         if (v[i] != model[i]) {
            st_fail(r, "chain of %d links (render queue %d, compute queue %d): element %u = %08x, expected %08x",
                    nlinks, qr, qc, i, v[i], model[i]);
            break;
         }
      r->checks += XC_N;
      r->iterations++;
   }
   st_metric(r, "links", (double)links);
   st_metric(r, "render_links", (double)renders);
   free(model);
   if (!r->errors)
      st_dev_destroy(&d);
   return st_verdict(r);
}

/* ======================= xq.exit ======================= */

enum exit_mode { EX_IN_FLIGHT, EX_AFTER_WAIT, EX_KILLED, EX_CLEAN, EX_MODES };
static const char *const ex_names[] = {"exit_in_flight", "exit_after_wait", "killed", "clean_exit"};

struct exit_child {
   pid_t pid;
   enum exit_mode mode;
   int report[2];
   uint64_t t_start, t_report;
   bool reported, killed;
   uint64_t kill_at;
};

static int
exit_child_main(void *arg)
{
   struct st_client_cfg *cfg = arg;
   struct st_client_result res;
   int rc = st_client(cfg, &res);
   if (rc && cfg->verbose)
      fprintf(stderr, "    exit child %d: %s\n", (int)getpid(), res.why);
   return rc;
}

/* Starts one client child. */
static void
exit_start(struct st_run *r, struct exit_child *c, enum exit_mode mode)
{
   memset(c, 0, sizeof(*c));
   c->mode = mode;
   if (pipe(c->report))
      c->report[0] = c->report[1] = -1;
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.seed = ((uint64_t)st_rand(r) << 32) | st_rand(r);
   cfg.render = true;
   cfg.spin = st_rand_range(r, 50000, 2000000); /* a slow producer keeps the pair in flight */
   cfg.report_fd = c->report[1];
   cfg.iters = mode == EX_KILLED ? 0 : (int)st_rand_range(r, 1, 3);
   cfg.until_ns = mode == EX_KILLED ? st_now() + 20 * ST_NS : 0;
   cfg.exit_after_submit = mode == EX_IN_FLIGHT;
   cfg.exit_after_wait = mode == EX_AFTER_WAIT;
   c->t_start = st_now();
   c->pid = st_fork(exit_child_main, &cfg);
   close(c->report[1]);
   c->report[1] = -1;
}

int
t_xq_exit(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (caps.queues < 2)
      return st_skip(r, "needs 2 queues, the device has %u", caps.queues);
   uint64_t per_mode[EX_MODES] = {0};
   uint64_t verify_ns = 3 * ST_NS;
   while (st_elapsed(r) + verify_ns < r->budget_ns && !r->errors) {
      /* Two children at a time, in different modes. */
      struct exit_child c[2];
      enum exit_mode m0 = (enum exit_mode)st_rand_range(r, 0, EX_MODES - 1);
      enum exit_mode m1 = (enum exit_mode)((m0 + 1 + st_rand_range(r, 0, EX_MODES - 2)) % EX_MODES);
      exit_start(r, &c[0], m0);
      exit_start(r, &c[1], m1);
      for (int i = 0; i < 2; i++) {
         if (c[i].pid < 0) {
            st_fail(r, "fork failed");
            continue;
         }
         /* Wait for the first submission (the client writes one byte), bounded. */
         char b;
         struct timeval tv = {.tv_sec = 20};
         fd_set fs;
         FD_ZERO(&fs);
         FD_SET(c[i].report[0], &fs);
         if (select(c[i].report[0] + 1, &fs, NULL, NULL, &tv) > 0 && read(c[i].report[0], &b, 1) == 1) {
            c[i].reported = true;
            c[i].t_report = st_now();
         }
         if (c[i].mode == EX_KILLED && c[i].reported) {
            st_sleep_ns(st_rand_range(r, 0, 30000) * 1000ull);
            kill(c[i].pid, SIGKILL);
            c[i].killed = true;
            c[i].t_report = st_now();
         }
      }
      for (int i = 0; i < 2; i++) {
         if (c[i].pid < 0)
            continue;
         int sig = 0;
         int rc = st_wait_pid(c[i].pid, 30 * ST_NS, &sig);
         double exit_ms = c[i].reported ? (st_now() - c[i].t_report) / 1e6 : 0;
         close(c[i].report[0]);
         r->checks++;
         if (rc == -2) {
            st_fail(r, "%s child did not exit within 30 s (teardown blocked with work in flight)", ex_names[c[i].mode]);
            continue;
         }
         if (!c[i].reported) {
            st_fail(r, "%s child never submitted (exit %d, signal %d)", ex_names[c[i].mode], rc, sig);
            continue;
         }
         if (c[i].mode == EX_KILLED) {
            if (!(rc == -1 && sig == SIGKILL) && rc != 0)
               st_fail(r, "killed child: exit %d signal %d before the kill", rc, sig);
         } else if (rc != 0) {
            st_fail(r, "%s child failed: exit %d%s", ex_names[c[i].mode], rc,
                    rc == 1   ? " (wrong data)"
                    : rc == 3 ? " (GPU timeout)"
                    : rc == -1 ? " (signal)"
                               : " (Vulkan error)");
         }
         if (c[i].mode == EX_IN_FLIGHT || c[i].mode == EX_KILLED)
            st_metric_max(r, "max_teardown_ms", exit_ms);
         per_mode[c[i].mode]++;
      }
      r->iterations++;
   }
   for (int m = 0; m < EX_MODES; m++)
      st_metric(r, ex_names[m], (double)per_mode[m]);
   /* A fresh client must still get correct results. */
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.render = true;
   cfg.iters = 8;
   cfg.spin = 100000;
   pid_t pid = st_fork(st_client_main, &cfg);
   int rc = st_wait_pid(pid, 60 * ST_NS, NULL);
   r->checks++;
   if (rc != 0)
      st_fail(r, "the verification client after the exit loop failed (exit %d)", rc);
   return st_verdict(r);
}
