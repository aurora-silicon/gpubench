// SPDX-License-Identifier: GPL-2.0-only
/* Render completion and concurrency: concurrent render+compute rings, the first heavy render
 * of fresh contexts, tiler-buffer growth, and simultaneous context construction (dangerous). */
#include "st.h"
#include "st_client.h"
#include "st_ring.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct rpc {
   uint32_t p[4];
   float color[4];
};

/* ======================= cr.concurrent ======================= */

int
t_cr_concurrent(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   int nq = caps.queues < 4 ? (int)caps.queues : 4;
   struct st_dev d;
   struct st_req req = {.nqueues = nq};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   /* Queues 0 and 1 render (one ring of large targets, one of small), the others compute.
    * With fewer queues, rings share them. */
   struct st_ring g[4];
   int ng = 0;
   if ((vr = st_ring_init(&g[ng++], &d, 0, 8, ST_RING_RENDER, 1024, 1024)) ||
       (vr = st_ring_init(&g[ng++], &d, 1 % nq, 16, ST_RING_RENDER, 256, 256)) ||
       (vr = st_ring_init(&g[ng++], &d, 2 % nq, 8, ST_RING_CHAIN, 65536, 0)) ||
       (vr = st_ring_init(&g[ng++], &d, 3 % nq, 8, ST_RING_CHAIN, 16384, 0)))
      return st_fail(r, "rings: %s", st_vkr(vr));
   while (st_time_left(r) && !r->errors) {
      bool idle = true;
      for (int i = 0; i < ng; i++) {
         uint64_t before = g[i].checked;
         vr = st_ring_pump(&g[i], r, true, 10 * ST_NS);
         if (vr)
            return st_fail(r, "%s: %s after %llu completions", g[i].name,
                           vr == VK_TIMEOUT ? "no completion for 10 s" : st_vkr(vr),
                           (unsigned long long)g[i].checked);
         idle &= g[i].checked == before;
      }
      if (idle)
         st_wait_value(&d, g[0].tl, g[0].checked + 1, 200 * 1000ull);
   }
   for (int i = 0; i < ng && !r->errors; i++)
      if ((vr = st_ring_drain(&g[i], r, 10 * ST_NS)))
         return st_fail(r, "%s drain: %s", g[i].name, st_vkr(vr));
   for (int i = 0; i < ng; i++) {
      char n[48];
      snprintf(n, sizeof(n), "ring%d_completions", i);
      st_metric(r, n, (double)g[i].checked);
      st_metric_max(r, "max_latency_ms", g[i].max_ms);
   }
   st_metric(r, "queues", nq);
   if (!r->errors) {
      for (int i = 0; i < ng; i++)
         st_ring_destroy(&g[i]);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}

/* ======================= render.first ======================= */

#define RF_W 1920u
#define RF_H 1080u
#define RF_GRID 256u    /* 256 x 256 cells, 131072 triangles */
#define RF_LAYERS 8u

struct heavy {
   struct st_pipe grid, add;
   struct st_img img;
   struct st_buf rb;
};

static VkResult
heavy_init(struct st_dev *d, struct heavy *h)
{
   VkResult vr;
   if ((vr = st_graphics_pipe(d, ST_SPV(grid_vert), ST_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, ST_BLEND_NONE, 0,
                              sizeof(struct rpc), &h->grid)) ||
       (vr = st_graphics_pipe(d, ST_SPV(fill_vert), ST_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, ST_BLEND_ADD, 0,
                              sizeof(struct rpc), &h->add)) ||
       (vr = st_img_create(d, RF_W, RF_H, VK_FORMAT_R8G8B8A8_UNORM,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &h->img)))
      return vr;
   return st_buf_create(d, (VkDeviceSize)RF_W * RF_H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &h->rb);
}

static void
heavy_destroy(struct st_dev *d, struct heavy *h)
{
   st_buf_destroy(d, &h->rb);
   st_img_destroy(d, &h->img);
   st_pipe_destroy(d, &h->grid);
   st_pipe_destroy(d, &h->add);
}

/* clear (0x10,0x20,0x30), grid in base colour, then RF_LAYERS additive layers of (8,4,2,0). */
static void
heavy_record(VkCommandBuffer cb, struct heavy *h, uint8_t base)
{
   memset(h->rb.map, 0xcd, h->rb.size);
   st_img_barrier(cb, h->img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   VkClearColorValue clear = {.float32 = {0x10 / 255.0f, 0x20 / 255.0f, 0x30 / 255.0f, 1}};
   st_begin_render(cb, &h->img, VK_ATTACHMENT_LOAD_OP_CLEAR, &clear);
   struct rpc pc = {{RF_GRID, 0, 0, 0}, {base / 255.0f, (base + 0x10) / 255.0f, (base + 0x20) / 255.0f, 1}};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, h->grid.pipe);
   vkCmdPushConstants(cb, h->grid.layout, h->grid.stages, 0, sizeof(pc), &pc);
   vkCmdDraw(cb, RF_GRID * RF_GRID * 6, 1, 0, 0);
   struct rpc lpc = {{0, 0, 0, 0}, {8 / 255.0f, 4 / 255.0f, 2 / 255.0f, 0}};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, h->add.pipe);
   vkCmdPushConstants(cb, h->add.layout, h->add.stages, 0, sizeof(lpc), &lpc);
   for (uint32_t l = 0; l < RF_LAYERS; l++)
      vkCmdDraw(cb, 3, 1, 0, 0);
   vkCmdEndRendering(cb);
   st_img_barrier(cb, h->img.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
   VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {RF_W, RF_H, 1}};
   vkCmdCopyImageToBuffer(cb, h->img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, h->rb.buf, 1, &reg);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
}

static uint64_t
heavy_verify(const struct heavy *h, uint8_t base, char *why, size_t n)
{
   uint8_t want[4] = {(uint8_t)(base + 8 * RF_LAYERS), (uint8_t)(base + 0x10 + 4 * RF_LAYERS),
                      (uint8_t)(base + 0x20 + 2 * RF_LAYERS), 0xff};
   uint32_t w;
   memcpy(&w, want, 4);
   const uint32_t *px = h->rb.map;
   uint64_t bad = 0;
   for (size_t i = 0; i < (size_t)RF_W * RF_H; i++)
      if (px[i] != w) {
         if (!bad)
            snprintf(why, n, "heavy 1080p render: pixel (%zu,%zu) = %08x, expected %08x", i % RF_W, i / RF_W, px[i], w);
         bad++;
      }
   return bad;
}

int
t_render_first(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   /* A render that takes longer than this has lost its completion (the GPU work is small). */
   const uint64_t late_ns = 5 * ST_NS;
   static const uint32_t gaps_ms[] = {0, 5, 30, 150, 600};
   uint64_t devices = 0, late = 0;
   while (st_time_left(r) && !r->errors) {
      int nq = (int)st_rand_range(r, 1, caps.queues < 4 ? caps.queues : 4);
      struct st_dev d;
      struct st_req req = {.nqueues = nq};
      VkResult vr = st_dev_create(&d, r->o->device, &req);
      if (vr)
         return st_fail(r, "device %llu: %s", (unsigned long long)devices, st_vkr(vr));
      struct heavy h;
      struct st_rcheck rc;
      VkCommandBuffer cb[ST_MAX_QUEUES];
      if ((vr = heavy_init(&d, &h)) || (vr = st_rcheck_init(&d, 256, 256, &rc)))
         return st_fail(r, "setup: %s", st_vkr(vr));
      for (int q = 0; q < nq; q++)
         if ((vr = st_cb_alloc(&d, q, &cb[q])))
            return st_fail(r, "command buffer: %s", st_vkr(vr));
      /* The first submission of this context is the heavy render, on a random queue. */
      int frames = (int)st_rand_range(r, 3, 8);
      for (int f = 0; f < frames && !r->errors; f++) {
         int q = (int)st_rand_range(r, 0, nq - 1);
         bool heavy_frame = f == 0 || st_rand_range(r, 0, 1);
         uint8_t base = (uint8_t)st_rand_range(r, 0x20, 0x60);
         uint32_t k = st_rand(r);
         if (f) {
            uint32_t gap = gaps_ms[st_rand_range(r, 0, sizeof(gaps_ms) / sizeof(gaps_ms[0]) - 1)];
            st_sleep_ns(gap * ST_MS);
         }
         st_cb_begin(cb[q]);
         if (heavy_frame)
            heavy_record(cb[q], &h, base);
         else
            st_rcheck_record(cb[q], &rc, k);
         vkEndCommandBuffer(cb[q]);
         uint64_t t0 = st_now();
         vr = st_run_cb(&d, q, cb[q], 30 * ST_NS);
         uint64_t dt = st_now() - t0;
         if (vr)
            return st_fail(r, "device %llu frame %d (%s, queue %d): %s after %.1f s", (unsigned long long)devices, f,
                           heavy_frame ? "heavy" : "light", q, st_vkr(vr), dt / 1e9);
         if (f == 0)
            st_metric_max(r, "max_first_frame_ms", dt / 1e6);
         else
            st_metric_max(r, "max_frame_ms", dt / 1e6);
         if (dt > late_ns) {
            late++;
            st_fail(r, "device %llu frame %d (%s) completed only after %.1f s: completion was late or lost",
                    (unsigned long long)devices, f, heavy_frame ? "heavy" : "light", dt / 1e9);
         }
         char why[256];
         r->checks++;
         if (heavy_frame ? heavy_verify(&h, base, why, sizeof(why)) : st_rcheck_verify(&rc, k, why, sizeof(why)))
            st_fail(r, "device %llu frame %d: %s", (unsigned long long)devices, f, why);
         r->iterations++;
      }
      if (r->errors)
         break;
      st_rcheck_destroy(&d, &rc);
      heavy_destroy(&d, &h);
      if (!st_dev_destroy(&d))
         return st_fail(r, "device %llu did not go idle at destruction", (unsigned long long)devices);
      devices++;
   }
   st_metric(r, "devices", (double)devices);
   st_metric(r, "late_frames", (double)late);
   return st_verdict(r);
}

/* ======================= render.tiler ======================= */

#define RT_W 1920u
#define RT_H 1080u
static const uint32_t tiler_quads[] = {2048, 16384, 65536, 131072};
#define RT_SIZES (sizeof(tiler_quads) / sizeof(tiler_quads[0]))

static uint32_t
row_of(uint32_t q, uint32_t seed, uint32_t rows)
{
   uint32_t h = (q + seed) * 2654435761u;
   h ^= h >> 15;
   h *= 2246822519u;
   h ^= h >> 13;
   return h % rows;
}

struct tframe {
   struct st_img img;
   struct st_buf rb;
   uint32_t quads, seed;
};

static void
tiler_record(VkCommandBuffer cb, struct st_pipe *p, struct tframe *f)
{
   memset(f->rb.map, 0xcd, f->rb.size);
   st_img_barrier(cb, f->img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   VkClearColorValue clear = {.float32 = {0, 0x40 / 255.0f, 0, 1}};
   st_begin_render(cb, &f->img, VK_ATTACHMENT_LOAD_OP_CLEAR, &clear);
   struct rpc pc = {{RT_H, f->seed, 0, 0}, {1 / 255.0f, 0, 0, 0}};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipe);
   vkCmdPushConstants(cb, p->layout, p->stages, 0, sizeof(pc), &pc);
   vkCmdDraw(cb, f->quads * 6, 1, 0, 0);
   vkCmdEndRendering(cb);
   st_img_barrier(cb, f->img.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
   VkBufferImageCopy reg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {RT_W, RT_H, 1}};
   vkCmdCopyImageToBuffer(cb, f->img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f->rb.buf, 1, &reg);
   st_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
}

/* Every pixel of row y holds (min(count[y], 255), 0x40, 0, 0xff). */
static uint64_t
tiler_verify(const struct tframe *f, uint32_t *count, char *why, size_t n)
{
   memset(count, 0, RT_H * 4);
   for (uint32_t q = 0; q < f->quads; q++)
      count[row_of(q, f->seed, RT_H)]++;
   const uint32_t *px = f->rb.map;
   uint64_t bad = 0;
   for (uint32_t y = 0; y < RT_H; y++) {
      uint32_t c = count[y] > 255 ? 255 : count[y];
      uint32_t want = c | 0x40u << 8 | 0xffu << 24;
      for (uint32_t x = 0; x < RT_W; x++)
         if (px[(size_t)y * RT_W + x] != want) {
            if (!bad)
               snprintf(why, n, "%u quads seed %u: pixel (%u,%u) = %08x, expected %08x (row covered %u times)",
                        f->quads, f->seed, x, y, px[(size_t)y * RT_W + x], want, count[y]);
            bad++;
         }
   }
   return bad;
}

int
t_render_tiler(struct st_run *r)
{
   uint32_t *count = malloc(RT_H * 4);
   uint64_t devices = 0, frames = 0, tris = 0;
   while (st_time_left(r) && !r->errors) {
      struct st_dev d;
      struct st_req req = {.nqueues = 1};
      VkResult vr = st_dev_create(&d, r->o->device, &req);
      if (vr)
         return st_fail(r, "device: %s", st_vkr(vr));
      struct st_pipe p;
      struct tframe f[2];
      VkCommandBuffer cb[2];
      memset(f, 0, sizeof(f));
      if ((vr = st_graphics_pipe(&d, ST_SPV(rows_vert), ST_SPV(color_frag), VK_FORMAT_R8G8B8A8_UNORM, ST_BLEND_ADD, 0,
                                 sizeof(struct rpc), &p)))
         return st_fail(r, "pipeline: %s", st_vkr(vr));
      for (int i = 0; i < 2; i++)
         if ((vr = st_img_create(&d, RT_W, RT_H, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &f[i].img)) ||
             (vr = st_buf_create(&d, (VkDeviceSize)RT_W * RT_H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &f[i].rb)) ||
             (vr = st_cb_alloc(&d, 0, &cb[i])))
            return st_fail(r, "targets: %s", st_vkr(vr));
      /* A light frame, then the largest one cold (first big frame on a fresh context), then
       * random sizes, sometimes two frames in flight. */
      int nframes = (int)st_rand_range(r, 4, 10);
      for (int fi = 0; fi < nframes && !r->errors; fi++) {
         int inflight = fi >= 2 && st_rand_range(r, 0, 1) ? 2 : 1;
         VkFence fence[2];
         uint64_t t0 = st_now();
         for (int j = 0; j < inflight; j++) {
            f[j].quads = fi == 0 ? tiler_quads[0] : fi == 1 ? tiler_quads[RT_SIZES - 1]
                                                           : tiler_quads[st_rand_range(r, 0, RT_SIZES - 1)];
            f[j].seed = st_rand(r);
            st_cb_begin(cb[j]);
            tiler_record(cb[j], &p, &f[j]);
            vkEndCommandBuffer(cb[j]);
            VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(d.dev, &fci, NULL, &fence[j]);
            if ((vr = st_submit(d.q[0], cb[j], NULL, 0, NULL, 0, fence[j])))
               return st_fail(r, "submit: %s", st_vkr(vr));
         }
         vr = vkWaitForFences(d.dev, inflight, fence, VK_TRUE, 30 * ST_NS);
         uint64_t dt = st_now() - t0;
         if (vr)
            return st_fail(r, "frame %d on device %llu (%u quads%s): %s after %.1f s", fi, (unsigned long long)devices,
                           f[0].quads, inflight > 1 ? ", two in flight" : "", st_vkr(vr), dt / 1e9);
         for (int j = 0; j < inflight; j++)
            vkDestroyFence(d.dev, fence[j], NULL);
         st_metric_max(r, fi == 1 ? "max_cold_big_frame_ms" : "max_frame_ms", dt / 1e6);
         if (dt > 10 * ST_NS)
            st_fail(r, "frame %d (%u quads) took %.1f s: completion late or lost", fi, f[0].quads, dt / 1e9);
         for (int j = 0; j < inflight; j++) {
            char why[256];
            r->checks++;
            if (tiler_verify(&f[j], count, why, sizeof(why)))
               st_fail(r, "device %llu frame %d: %s", (unsigned long long)devices, fi, why);
            tris += 2ull * f[j].quads;
            frames++;
         }
         r->iterations++;
      }
      if (r->errors)
         break;
      for (int i = 0; i < 2; i++) {
         st_buf_destroy(&d, &f[i].rb);
         st_img_destroy(&d, &f[i].img);
      }
      st_pipe_destroy(&d, &p);
      if (!st_dev_destroy(&d))
         return st_fail(r, "device did not go idle at destruction");
      devices++;
   }
   free(count);
   st_metric(r, "devices", (double)devices);
   st_metric(r, "frames", (double)frames);
   st_metric(r, "mtris", tris / 1e6);
   return st_verdict(r);
}

/* ======================= cr.pairs (dangerous) ======================= */

struct pair_child {
   struct st_gate *gate;
   struct st_client_cfg cfg;
};

static int
pair_child_main(void *arg)
{
   struct pair_child *p = arg;
   st_gate_wait(p->gate);
   return st_client_main(&p->cfg);
}

int
t_cr_pairs(struct st_run *r)
{
   static const int sizes[] = {4, 8};
   int round = 0;
   while (st_time_left(r) && !r->errors && round < 40) {
      int n = sizes[round % 2];
      struct st_gate gate;
      st_gate_init(&gate);
      pid_t pids[8];
      for (int i = 0; i < n; i++) {
         struct pair_child pc = {.gate = &gate, .cfg = st_client_defaults(r)};
         pc.cfg.seed = ((uint64_t)st_rand(r) << 32) | st_rand(r);
         pc.cfg.nqueues = 1;
         pc.cfg.render = true;
         pc.cfg.rw = pc.cfg.rh = 256;
         pc.cfg.iters = 20;
         pids[i] = st_fork(pair_child_main, &pc);
      }
      st_sleep_ns(100 * ST_MS); /* every child is blocked at the gate */
      st_gate_open(&gate);
      for (int i = 0; i < n; i++) {
         int sig = 0, rc = st_wait_pid(pids[i], 60 * ST_NS, &sig);
         r->checks++;
         if (rc != 0)
            st_fail(r, "round %d (%d processes at once): client %d %s %d", round, n, i,
                    rc == -2 ? "timed out" : rc == -1 ? "died with signal" : "exited", rc == -1 ? sig : rc);
      }
      r->iterations++;
      round++;
   }
   st_metric(r, "rounds", round);
   return st_verdict(r);
}
