// SPDX-License-Identifier: GPL-2.0-only
/* Context lifecycle: device and process churn, a faulting client beside a healthy one, and
 * queue/context exhaustion. */
#include "st.h"
#include "st_client.h"
#include "st_ring.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

/* ======================= life.churn ======================= */

enum churn_mode { CH_AFTER_WAIT, CH_CLEAN, CH_IN_FLIGHT, CH_KILLED, CH_MODES };
static const char *const ch_names[] = {"exit_after_wait", "clean_exit", "exit_in_flight", "killed"};

/* One short-lived client process in the given mode. */
static void
churn_one(struct st_run *r, enum churn_mode mode)
{
   int fd[2];
   if (pipe(fd)) {
      st_fail(r, "pipe");
      return;
   }
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.seed = ((uint64_t)st_rand(r) << 32) | st_rand(r);
   cfg.render = st_rand_range(r, 0, 1);
   cfg.rw = cfg.rh = 64;
   cfg.nqueues = (int)st_rand_range(r, 1, 2);
   cfg.iters = mode == CH_KILLED ? 0 : 1;
   cfg.until_ns = mode == CH_KILLED ? st_now() + 20 * ST_NS : 0;
   cfg.spin = mode == CH_KILLED || mode == CH_IN_FLIGHT ? 300000 : 0;
   cfg.exit_after_wait = mode == CH_AFTER_WAIT;
   cfg.exit_after_submit = mode == CH_IN_FLIGHT;
   cfg.report_fd = fd[1];
   pid_t pid = st_fork(st_client_main, &cfg);
   close(fd[1]);
   if (pid < 0) {
      close(fd[0]);
      st_fail(r, "fork");
      return;
   }
   char b;
   struct timeval tv = {.tv_sec = 20};
   fd_set fs;
   FD_ZERO(&fs);
   FD_SET(fd[0], &fs);
   bool started = select(fd[0] + 1, &fs, NULL, NULL, &tv) > 0 && read(fd[0], &b, 1) == 1;
   close(fd[0]);
   uint64_t t0 = st_now();
   if (mode == CH_KILLED) {
      st_sleep_ns(st_rand_range(r, 0, 5000) * 1000ull);
      kill(pid, SIGKILL);
   }
   int sig = 0, rc = st_wait_pid(pid, 30 * ST_NS, &sig);
   st_metric_max(r, "max_exit_ms", (st_now() - t0) / 1e6);
   r->checks++;
   if (rc == -2)
      st_fail(r, "%s client did not exit within 30 s", ch_names[mode]);
   else if (!started)
      st_fail(r, "%s client never submitted (exit %d, signal %d)", ch_names[mode], rc, sig);
   else if (mode == CH_KILLED ? !(rc == 0 || (rc == -1 && sig == SIGKILL)) : rc != 0)
      st_fail(r, "%s client: exit %d signal %d%s", ch_names[mode], rc, sig,
              rc == 1 ? " (wrong data)" : rc == 3 ? " (GPU timeout)" : "");
   r->iterations++;
}

static int
lane_exitloop(struct st_run *r, void *arg)
{
   (void)arg;
   uint64_t n[CH_MODES] = {0};
   while (st_elapsed(r) + ST_NS < r->budget_ns && !r->errors) {
      enum churn_mode m = (enum churn_mode)st_rand_range(r, 0, CH_MODES - 1);
      churn_one(r, m);
      n[m]++;
   }
   for (int m = 0; m < CH_MODES; m++)
      st_metric(r, ch_names[m], (double)n[m]);
   return st_verdict(r);
}

static int
lane_devchurn(struct st_run *r, void *arg)
{
   (void)arg;
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   uint64_t devices = 0;
   while (st_time_left(r) && !r->errors) {
      int nq = (int)st_rand_range(r, 1, caps.queues < 4 ? caps.queues : 4);
      struct st_dev d;
      struct st_req req = {.nqueues = nq};
      VkResult vr = st_dev_create(&d, r->o->device, &req);
      if (vr)
         return st_fail(r, "device %llu: %s", (unsigned long long)devices, st_vkr(vr));
      struct st_ring rr, cr;
      int qr = (int)st_rand_range(r, 0, nq - 1), qc = (int)st_rand_range(r, 0, nq - 1);
      if ((vr = st_ring_init(&rr, &d, qr, 2, ST_RING_RENDER, 64, 64)) ||
          (vr = st_ring_init(&cr, &d, qc, 2, ST_RING_CHAIN, 4096, 0)))
         return st_fail(r, "device %llu rings: %s", (unsigned long long)devices, st_vkr(vr));
      if ((vr = st_ring_pump(&rr, r, true, 10 * ST_NS)) || (vr = st_ring_pump(&cr, r, true, 10 * ST_NS)) ||
          (vr = st_ring_drain(&rr, r, 10 * ST_NS)) || (vr = st_ring_drain(&cr, r, 10 * ST_NS)))
         return st_fail(r, "device %llu work: %s", (unsigned long long)devices, st_vkr(vr));
      st_ring_destroy(&rr);
      st_ring_destroy(&cr);
      if (!st_dev_destroy(&d))
         return st_fail(r, "device %llu did not go idle", (unsigned long long)devices);
      devices++;
   }
   st_metric(r, "devices", (double)devices);
   return st_verdict(r);
}

int
t_life_churn(struct st_run *r)
{
   struct st_lane lanes[3] = {
      {"exitloop1", lane_exitloop, NULL},
      {"exitloop2", lane_exitloop, NULL},
      {"devices", lane_devchurn, NULL},
   };
   return st_run_lanes(r, lanes, 3);
}

/* ======================= life.fault ======================= */

enum { FAULT_LOST = 0, FAULT_IGNORED = 10 };

struct fault_cfg {
   int device;
   uint64_t addr;
   uint32_t value;
};

static int
fault_child(void *arg)
{
   struct fault_cfg *f = arg;
   struct st_dev d;
   struct st_req req = {.nqueues = 1, .bda = true};
   if (st_dev_create(&d, f->device, &req))
      return 2;
   struct st_pipe p;
   VkCommandBuffer cb;
   if (st_compute_pipe(&d, ST_SPV(fault_comp), 0, 12, &p) || st_cb_alloc(&d, 0, &cb))
      return 2;
   st_cb_begin(cb);
   uint32_t pc[3] = {(uint32_t)f->addr, (uint32_t)(f->addr >> 32), f->value};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
   vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, 4, 1, 1);
   vkEndCommandBuffer(cb);
   VkResult vr = st_run_cb(&d, 0, cb, 15 * ST_NS);
   /* The context is gone either way; exit without teardown. */
   if (vr == VK_ERROR_DEVICE_LOST)
      _exit(FAULT_LOST);
   if (vr == VK_SUCCESS)
      _exit(FAULT_IGNORED);
   _exit(vr == VK_TIMEOUT ? 3 : 2);
}

static int
lane_faulter(struct st_run *r, void *arg)
{
   (void)arg;
   uint64_t lost = 0, ignored = 0;
   st_sleep_ns(500 * ST_MS); /* let the healthy client get going */
   while (st_elapsed(r) + 2 * ST_NS < r->budget_ns && !r->errors) {
      static const uint64_t addrs[] = {0xdead0000ull, 0x10000ull, 0x7fdead0000ull};
      struct fault_cfg f = {.device = r->o->device, .addr = addrs[st_rand_range(r, 0, 2)], .value = st_rand(r)};
      pid_t pid = st_fork(fault_child, &f);
      int sig = 0, rc = st_wait_pid(pid, 30 * ST_NS, &sig);
      r->checks++;
      if (rc == FAULT_LOST)
         lost++;
      else if (rc == FAULT_IGNORED)
         ignored++;
      else
         st_fail(r, "faulting client (address 0x%llx): %s %d", (unsigned long long)f.addr,
                 rc == -2   ? "did not finish within 30 s; timeout"
                 : rc == 3  ? "its submission never completed; exit"
                 : rc == -1 ? "died with signal"
                            : "Vulkan error; exit",
                 rc == -1 ? sig : rc);
      r->iterations++;
      st_sleep_ns(st_rand_range(r, 100, 400) * ST_MS);
   }
   st_metric(r, "faults_device_lost", (double)lost);
   st_metric(r, "faults_ignored", (double)ignored);
   if (!lost && !ignored && !r->errors)
      return st_fail(r, "no faulting submission ran");
   return st_verdict(r);
}

static int
lane_healthy(struct st_run *r, void *arg)
{
   (void)arg;
   return st_lane_queues(r, 2, true);
}

int
t_life_fault(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (!caps.bda)
      return st_skip(r, "needs bufferDeviceAddress to fault on purpose");
   struct st_lane lanes[2] = {{"healthy", lane_healthy, NULL}, {"faulter", lane_faulter, NULL}};
   st_run_lanes(r, lanes, 2);
   /* A fresh client must still work. */
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.render = true;
   cfg.iters = 4;
   int rc = st_wait_pid(st_fork(st_client_main, &cfg), 60 * ST_NS, NULL);
   r->checks++;
   if (rc != 0)
      st_fail(r, "a fresh client after the faults failed (exit %d)", rc);
   return st_verdict(r);
}

/* ======================= queue.exhaust ======================= */

#define EX_MAX_DEVICES 32

struct exdev {
   struct st_dev d;
   struct st_ring ring[ST_MAX_QUEUES];
   struct st_ring rend;
   int nr;
};

/* One checked submission on every queue of the device (and a render on queue 0). */
static VkResult
exdev_round(struct st_run *r, struct exdev *e)
{
   VkResult vr;
   for (int i = 0; i < e->nr; i++)
      if ((vr = st_ring_pump(&e->ring[i], r, true, 10 * ST_NS)))
         return vr;
   if ((vr = st_ring_pump(&e->rend, r, true, 10 * ST_NS)))
      return vr;
   for (int i = 0; i < e->nr; i++)
      if ((vr = st_ring_drain(&e->ring[i], r, 10 * ST_NS)))
         return vr;
   return st_ring_drain(&e->rend, r, 10 * ST_NS);
}

int
t_queue_exhaust(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   int nq = caps.queues > ST_MAX_QUEUES ? ST_MAX_QUEUES : (int)caps.queues;
   struct exdev *ex = calloc(EX_MAX_DEVICES, sizeof(*ex));
   int n = 0;
   const char *stop = "device cap reached";
   char stopbuf[160];
   while (n < EX_MAX_DEVICES && st_time_left(r) && !r->errors) {
      struct exdev *e = &ex[n];
      struct st_req req = {.nqueues = nq};
      VkResult vr = st_dev_create(&e->d, r->o->device, &req);
      if (vr) {
         snprintf(stopbuf, sizeof(stopbuf), "device %d refused at creation: %s", n + 1, st_vkr(vr));
         stop = stopbuf;
         if (vr == VK_ERROR_DEVICE_LOST)
            st_fail(r, "%s", stop);
         break;
      }
      e->nr = nq;
      for (int q = 0; q < nq && !vr; q++)
         vr = st_ring_init(&e->ring[q], &e->d, q, 1, ST_RING_CHAIN, 4096, 0);
      if (!vr)
         vr = st_ring_init(&e->rend, &e->d, 0, 1, ST_RING_RENDER, 64, 64);
      if (!vr)
         vr = exdev_round(r, e);
      if (vr) {
         /* Refusing new work cleanly is fine; hanging is not. */
         snprintf(stopbuf, sizeof(stopbuf), "device %d refused its first work: %s", n + 1, st_vkr(vr));
         stop = stopbuf;
         if (vr == VK_TIMEOUT)
            st_fail(r, "device %d: work never completed (no clean refusal)", n + 1);
         n++; /* keep it for the checks below; it is not expected to work */
         ex[n - 1].nr = -1;
         break;
      }
      n++;
      /* Every earlier context must keep working. */
      for (int i = 0; i < n - 1 && !r->errors; i++)
         if ((vr = exdev_round(r, &ex[i])))
            st_fail(r, "device %d stopped working after device %d was created: %s", i + 1, n, st_vkr(vr));
      r->iterations++;
   }
   int working = 0;
   for (int i = 0; i < n; i++)
      working += ex[i].nr > 0;
   st_metric(r, "devices", working);
   st_metric(r, "queues", (double)working * nq);
   if (!r->errors) {
      /* Once more on every working device, then release everything. */
      for (int i = 0; i < n && !r->errors; i++)
         if (ex[i].nr > 0) {
            VkResult vr = exdev_round(r, &ex[i]);
            if (vr)
               st_fail(r, "device %d after the limit: %s", i + 1, st_vkr(vr));
         }
   }
   if (r->errors)
      return ST_FAIL; /* leave the devices to process exit */
   for (int i = n - 1; i >= 0; i--) {
      if (ex[i].nr > 0) {
         for (int q = 0; q < ex[i].nr; q++)
            st_ring_destroy(&ex[i].ring[q]);
         st_ring_destroy(&ex[i].rend);
      }
      st_dev_destroy(&ex[i].d);
   }
   free(ex);
   fprintf(stderr, "  [queue.exhaust] %d devices x %d queues worked; stop: %s\n", working, nq, stop);
   /* After release, a fresh client must work. */
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.render = true;
   cfg.iters = 4;
   int rc = st_wait_pid(st_fork(st_client_main, &cfg), 60 * ST_NS, NULL);
   r->checks++;
   if (rc != 0)
      st_fail(r, "a fresh client after releasing every context failed (exit %d)", rc);
   return st_verdict(r);
}
