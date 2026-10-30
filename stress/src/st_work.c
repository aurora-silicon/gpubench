// SPDX-License-Identifier: GPL-2.0-only
/* Run helpers, child processes and the verified workloads shared by several tests. */
#include "st.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ---------------- run helpers ---------------- */

uint64_t
st_now(void)
{
   return gb_now_ns();
}

void
st_sleep_ns(uint64_t ns)
{
   struct timespec ts = {.tv_sec = ns / ST_NS, .tv_nsec = ns % ST_NS};
   while (nanosleep(&ts, &ts) && errno == EINTR)
      ;
}

uint64_t
st_part(const struct st_run *r, double f, uint64_t min_ns)
{
   uint64_t v = (uint64_t)(r->budget_ns * f);
   return v < min_ns ? min_ns : v;
}

uint32_t
st_rand(struct st_run *r)
{
   /* xorshift64* */
   uint64_t x = r->rng;
   x ^= x >> 12;
   x ^= x << 25;
   x ^= x >> 27;
   r->rng = x;
   return (uint32_t)((x * 2685821657736338717ull) >> 32);
}

uint32_t
st_rand_range(struct st_run *r, uint32_t lo, uint32_t hi)
{
   return lo + st_rand(r) % (hi - lo + 1);
}

static void
vlog(struct st_run *r, const char *kind, const char *fmt, va_list ap)
{
   fprintf(stderr, "  [%s] %s", r->t ? r->t->name : "?", kind);
   vfprintf(stderr, fmt, ap);
   fputc('\n', stderr);
}

int
st_fail(struct st_run *r, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   if (!r->errors)
      vsnprintf(r->reason, sizeof(r->reason), fmt, ap);
   va_end(ap);
   r->errors++;
   if (r->errors <= 5) {
      va_start(ap, fmt);
      vlog(r, "FAIL: ", fmt, ap);
      va_end(ap);
   }
   return ST_FAIL;
}

int
st_skip(struct st_run *r, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(r->reason, sizeof(r->reason), fmt, ap);
   va_end(ap);
   r->skipped = true;
   return ST_SKIP;
}

void
st_log(struct st_run *r, const char *fmt, ...)
{
   if (!r->o->verbose)
      return;
   va_list ap;
   va_start(ap, fmt);
   vlog(r, "", fmt, ap);
   va_end(ap);
}

static struct st_metric *
metric(struct st_run *r, const char *name, bool *fresh)
{
   for (int i = 0; i < r->nm; i++)
      if (!strcmp(r->m[i].name, name)) {
         *fresh = false;
         return &r->m[i];
      }
   if (r->nm == ST_MAX_METRICS)
      return NULL;
   struct st_metric *m = &r->m[r->nm++];
   snprintf(m->name, sizeof(m->name), "%s", name);
   *fresh = true;
   return m;
}

void
st_metric(struct st_run *r, const char *name, double v)
{
   bool fresh;
   struct st_metric *m = metric(r, name, &fresh);
   if (m)
      m->v = v;
}

void
st_metric_max(struct st_run *r, const char *name, double v)
{
   bool fresh;
   struct st_metric *m = metric(r, name, &fresh);
   if (m && (fresh || v > m->v))
      m->v = v;
}

void
st_metric_add(struct st_run *r, const char *name, double v)
{
   bool fresh;
   struct st_metric *m = metric(r, name, &fresh);
   if (m)
      m->v = (fresh ? 0 : m->v) + v;
}

/* ---------------- processes ---------------- */

pid_t
st_fork(int (*fn)(void *arg), void *arg)
{
   fflush(stdout);
   fflush(stderr);
   pid_t pid = fork();
   if (pid == 0) {
      signal(SIGPIPE, SIG_DFL);
      int rc = fn(arg);
      fflush(stderr);
      _exit(rc);
   }
   return pid;
}

int
st_wait_pid(pid_t pid, uint64_t timeout_ns, int *sig)
{
   uint64_t end = st_now() + timeout_ns;
   for (;;) {
      int status;
      pid_t w = waitpid(pid, &status, WNOHANG);
      if (w == pid) {
         if (WIFEXITED(status))
            return WEXITSTATUS(status);
         if (sig)
            *sig = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
         return -1;
      }
      if (w < 0)
         return -1;
      if (st_now() >= end) {
         kill(pid, SIGKILL);
         /* A process blocked in the kernel may not die at once; give it a bounded grace. */
         uint64_t grace = st_now() + 10 * ST_NS;
         while (st_now() < grace) {
            if (waitpid(pid, &status, WNOHANG) == pid)
               return -2;
            st_sleep_ns(10 * ST_MS);
         }
         return -2;
      }
      st_sleep_ns(2 * ST_MS);
   }
}

void
st_gate_init(struct st_gate *g)
{
   if (pipe(g->fd))
      g->fd[0] = g->fd[1] = -1;
}

void
st_gate_wait(struct st_gate *g)
{
   close(g->fd[1]);
   char c;
   /* Returns when the parent closes its write end (EOF releases every reader at once). */
   while (read(g->fd[0], &c, 1) < 0 && errno == EINTR)
      ;
   close(g->fd[0]);
}

void
st_gate_open(struct st_gate *g)
{
   close(g->fd[0]);
   close(g->fd[1]);
}

/* ---------------- chain compute ---------------- */

VkResult
st_chain_init(struct st_dev *d, struct st_buf *data, struct st_chain *c)
{
   VkResult r = st_compute_pipe(d, ST_SPV(chain_comp), 1, 12, &c->pipe);
   if (r == VK_SUCCESS)
      st_pipe_buffers(d, &c->pipe, &data, 1);
   return r;
}

void
st_chain_record(VkCommandBuffer cb, struct st_chain *c, uint32_t base, uint32_t n, uint32_t k)
{
   uint32_t pc[3] = {base, n, k};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe.layout, 0, 1, &c->pipe.set, 0, NULL);
   vkCmdPushConstants(cb, c->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, (n + 63) / 64, 1, 1);
}

void
st_chain_destroy(struct st_dev *d, struct st_chain *c)
{
   st_pipe_destroy(d, &c->pipe);
}

/* ---------------- render check ---------------- */

struct st_pc {
   uint32_t p[4];
   float color[4];
};

static void
rc_colors(uint32_t k, uint8_t a[4], uint8_t b[4])
{
   a[0] = (uint8_t)k;
   a[1] = 0x11;
   a[2] = (uint8_t)(k >> 8);
   a[3] = 0xff;
   b[0] = 0x33;
   b[1] = (uint8_t)(k * 7);
   b[2] = 0x44;
   b[3] = 0xff;
}

VkResult
st_rcheck_init(struct st_dev *d, uint32_t w, uint32_t h, struct st_rcheck *rc)
{
   memset(rc, 0, sizeof(*rc));
   rc->w = w;
   rc->h = h;
   VkResult r = st_graphics_pipe(d, ST_SPV(fill_vert), ST_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, ST_BLEND_NONE, 0,
                                 sizeof(struct st_pc), &rc->pipe);
   if (r != VK_SUCCESS)
      return r;
   r = st_img_create(d, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &rc->img);
   if (r != VK_SUCCESS)
      return r;
   return st_buf_create(d, (VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &rc->rb);
}

void
st_rcheck_record(VkCommandBuffer cb, struct st_rcheck *rc, uint32_t k)
{
   uint8_t a[4], b[4];
   rc_colors(k, a, b);
   memset(rc->rb.map, 0xcd, rc->rb.size); /* poison: a lost copy shows */
   st_img_barrier(cb, rc->img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   VkClearColorValue clear = {.float32 = {a[0] / 255.0f, a[1] / 255.0f, a[2] / 255.0f, a[3] / 255.0f}};
   st_begin_render(cb, &rc->img, VK_ATTACHMENT_LOAD_OP_CLEAR, &clear);
   struct st_pc pc = {{0, 0, 0, 0}, {b[0] / 255.0f, b[1] / 255.0f, b[2] / 255.0f, b[3] / 255.0f}};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rc->pipe.pipe);
   vkCmdPushConstants(cb, rc->pipe.layout, rc->pipe.stages, 0, sizeof(pc), &pc);
   VkRect2D sc = {{0, 0}, {rc->w / 2, rc->h}};
   vkCmdSetScissor(cb, 0, 1, &sc);
   vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
   st_img_barrier(cb, rc->img.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
   VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {rc->w, rc->h, 1}};
   vkCmdCopyImageToBuffer(cb, rc->img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rc->rb.buf, 1, &reg);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
}

uint64_t
st_rcheck_verify(const struct st_rcheck *rc, uint32_t k, char *why, size_t n)
{
   uint8_t a[4], b[4];
   rc_colors(k, a, b);
   uint32_t wa, wb;
   memcpy(&wa, a, 4);
   memcpy(&wb, b, 4);
   const uint32_t *px = rc->rb.map;
   uint64_t bad = 0;
   for (uint32_t y = 0; y < rc->h; y++)
      for (uint32_t x = 0; x < rc->w; x++) {
         uint32_t want = x < rc->w / 2 ? wb : wa, got = px[(size_t)y * rc->w + x];
         if (got != want) {
            if (!bad && why)
               snprintf(why, n, "render %ux%u k=%u: pixel (%u,%u) = %08x, expected %08x", rc->w, rc->h, k, x, y, got,
                        want);
            bad++;
         }
      }
   return bad;
}

void
st_rcheck_destroy(struct st_dev *d, struct st_rcheck *rc)
{
   st_buf_destroy(d, &rc->rb);
   st_img_destroy(d, &rc->img);
   st_pipe_destroy(d, &rc->pipe);
}

/* ---------------- long ALU ---------------- */

VkResult
st_lcg_init(struct st_dev *d, uint32_t groups, struct st_lcg *l)
{
   memset(l, 0, sizeof(*l));
   l->groups = groups;
   VkResult r = st_buf_create(d, (VkDeviceSize)groups * 64 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &l->out);
   if (r != VK_SUCCESS)
      return r;
   if ((r = st_compute_pipe(d, ST_SPV(lcg_comp), 1, 8, &l->pipe)) != VK_SUCCESS)
      return r;
   struct st_buf *b = &l->out;
   st_pipe_buffers(d, &l->pipe, &b, 1);
   return VK_SUCCESS;
}

void
st_lcg_record(VkCommandBuffer cb, struct st_lcg *l, uint32_t iters, uint32_t seed)
{
   uint32_t pc[2] = {iters, seed};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, l->pipe.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, l->pipe.layout, 0, 1, &l->pipe.set, 0, NULL);
   vkCmdPushConstants(cb, l->pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, l->groups, 1, 1);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
}

uint32_t
st_lcg_model(uint32_t seed, uint32_t gid, uint32_t iters)
{
   uint32_t x = seed ^ (gid * 2654435761u);
   for (uint32_t i = 0; i < iters; i++)
      x = (x ^ (x >> 13)) * 1664525u + 1013904223u;
   return x;
}

uint64_t
st_lcg_verify(const struct st_lcg *l, uint32_t iters, uint32_t seed, int samples, char *why, size_t n)
{
   const uint32_t *o = l->out.map;
   uint32_t total = l->groups * 64;
   uint64_t bad = 0;
   for (int s = 0; s < samples; s++) {
      uint32_t gid = (uint32_t)(((uint64_t)s * 7919u + seed) % total);
      if (s == 0)
         gid = 0;
      else if (s == 1)
         gid = total - 1;
      uint32_t want = st_lcg_model(seed, gid, iters);
      if (o[gid] != want) {
         if (!bad && why)
            snprintf(why, n, "long ALU result gid %u = %08x, expected %08x (iters %u)", gid, o[gid], want, iters);
         bad++;
      }
   }
   return bad;
}

uint32_t
st_lcg_calibrate(struct st_dev *d, int qi, struct st_lcg *l, double target_ms, double *measured_ms)
{
   VkCommandBuffer cb;
   if (st_cb_alloc(d, qi, &cb) != VK_SUCCESS)
      return 0;
   uint32_t iters = 4096;
   double ms = 0;
   for (int step = 0; step < 12; step++) {
      if (st_cb_begin(cb) != VK_SUCCESS)
         return 0;
      st_lcg_record(cb, l, iters, 1234);
      if (vkEndCommandBuffer(cb) != VK_SUCCESS)
         return 0;
      uint64_t t = st_now();
      if (st_run_cb(d, qi, cb, 30 * ST_NS) != VK_SUCCESS)
         return 0;
      ms = (st_now() - t) / 1e6;
      if (ms >= target_ms * 0.25 || iters >= (1u << 30))
         break;
      iters *= ms < target_ms / 64 ? 8 : 2;
   }
   if (ms <= 0)
      return 0;
   double scaled = iters * (target_ms / ms);
   if (scaled > 2e9)
      scaled = 2e9;
   if (measured_ms)
      *measured_ms = ms;
   return (uint32_t)scaled;
}

void
st_lcg_destroy(struct st_dev *d, struct st_lcg *l)
{
   st_pipe_destroy(d, &l->pipe);
   st_buf_destroy(d, &l->out);
}
