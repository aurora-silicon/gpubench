// SPDX-License-Identifier: GPL-2.0-only
/* Memory: refusal of impossible allocations, checked work while memory is held near a safe
 * cap, buffer/mapping churn, and real exhaustion (dangerous). */
#include "st.h"
#include "st_client.h"
#include "st_ring.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHUNK (128ull << 20)

/* MemAvailable and MemTotal in bytes (0 when unknown). */
static void
meminfo(uint64_t *avail, uint64_t *total)
{
   *avail = *total = 0;
   FILE *f = fopen("/proc/meminfo", "r");
   if (!f)
      return;
   char line[256];
   while (fgets(line, sizeof(line), f)) {
      unsigned long long kb;
      if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
         *avail = kb << 10;
      else if (sscanf(line, "MemTotal: %llu kB", &kb) == 1)
         *total = kb << 10;
   }
   fclose(f);
}

static uint32_t
mem_type(struct st_dev *d, uint32_t bits, uint32_t preferred)
{
   if (bits & (1u << preferred))
      return preferred;
   for (uint32_t i = 0; i < d->mp.memoryTypeCount; i++)
      if (bits & (1u << i))
         return i;
   return 0;
}

/* An allocation the device cannot satisfy must fail promptly with an out-of-memory error. */
static void
over_ask(struct st_run *r, struct st_dev *d, VkDeviceSize size, const char *what)
{
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size,
                              .memoryTypeIndex = d->mem_dev};
   VkDeviceMemory m = VK_NULL_HANDLE;
   uint64_t t0 = st_now();
   VkResult vr = vkAllocateMemory(d->dev, &ai, NULL, &m);
   double ms = (st_now() - t0) / 1e6;
   st_metric_max(r, "max_refusal_ms", ms);
   r->checks++;
   if (vr == VK_SUCCESS) {
      /* Lazily backed memory may be granted; never touch it. */
      vkFreeMemory(d->dev, m, NULL);
      st_metric_add(r, "over_asks_granted", 1);
   } else if (vr != VK_ERROR_OUT_OF_DEVICE_MEMORY && vr != VK_ERROR_OUT_OF_HOST_MEMORY) {
      st_fail(r, "%s (%.1f GiB): %s, expected an out-of-memory error", what, size / 1073741824.0, st_vkr(vr));
   }
   if (ms > 5000)
      st_fail(r, "%s (%.1f GiB): took %.1f s to answer", what, size / 1073741824.0, ms / 1e3);
}

struct chunk {
   VkBuffer buf;
   VkDeviceMemory mem;
   uint32_t pattern;
};

/* Allocates one device-local chunk and has the GPU fill it (so every page is backed). */
static VkResult
chunk_alloc(struct st_dev *d, VkCommandBuffer cb, struct st_buf *probe, struct chunk *c, uint32_t pattern,
            uint64_t *bad)
{
   memset(c, 0, sizeof(*c));
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = CHUNK,
                            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
   VkResult vr = vkCreateBuffer(d->dev, &bi, NULL, &c->buf);
   if (vr)
      return vr;
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(d->dev, c->buf, &req);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                              .memoryTypeIndex = mem_type(d, req.memoryTypeBits, d->mem_dev)};
   if ((vr = vkAllocateMemory(d->dev, &ai, NULL, &c->mem)) || (vr = vkBindBufferMemory(d->dev, c->buf, c->mem, 0))) {
      vkDestroyBuffer(d->dev, c->buf, NULL);
      if (c->mem)
         vkFreeMemory(d->dev, c->mem, NULL);
      memset(c, 0, sizeof(*c));
      return vr;
   }
   c->pattern = pattern;
   /* Fill, then copy 16 probes of 4 KiB spread over the chunk back to the host. */
   memset(probe->map, 0, probe->size);
   st_cb_begin(cb);
   vkCmdFillBuffer(cb, c->buf, 0, VK_WHOLE_SIZE, pattern);
   st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
   VkBufferCopy reg[16];
   for (int i = 0; i < 16; i++)
      reg[i] = (VkBufferCopy){.srcOffset = (CHUNK - 4096) / 15 * i & ~4095ull, .dstOffset = 4096ull * i, .size = 4096};
   vkCmdCopyBuffer(cb, c->buf, probe->buf, 16, reg);
   st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
   vkEndCommandBuffer(cb);
   if ((vr = st_run_cb(d, 0, cb, 20 * ST_NS)))
      return vr;
   const uint32_t *p = probe->map;
   for (size_t i = 0; i < 16 * 1024; i++)
      if (p[i] != pattern) {
         (*bad)++;
         break;
      }
   return VK_SUCCESS;
}

static void
chunk_free(struct st_dev *d, struct chunk *c)
{
   if (c->buf)
      vkDestroyBuffer(d->dev, c->buf, NULL);
   if (c->mem)
      vkFreeMemory(d->dev, c->mem, NULL);
   memset(c, 0, sizeof(*c));
}

/* Checked render and compute while memory is held. */
static void
work_while_holding(struct st_run *r, struct st_dev *d, uint64_t ns)
{
   struct st_ring g[2];
   VkResult vr;
   if ((vr = st_ring_init(&g[0], d, 0, 4, ST_RING_RENDER, 256, 256)) ||
       (vr = st_ring_init(&g[1], d, d->nq > 1 ? 1 : 0, 4, ST_RING_CHAIN, 65536, 0))) {
      /* Running out of memory for these is a clean refusal; failing work is not. */
      st_metric_add(r, "work_setup_refused", 1);
      return;
   }
   uint64_t end = st_now() + ns;
   while (st_now() < end && !r->errors)
      for (int i = 0; i < 2; i++)
         if ((vr = st_ring_pump(&g[i], r, true, 10 * ST_NS))) {
            st_fail(r, "%s under memory pressure: %s", g[i].name, st_vkr(vr));
            return;
         }
   for (int i = 0; i < 2 && !r->errors; i++)
      if ((vr = st_ring_drain(&g[i], r, 10 * ST_NS)))
         st_fail(r, "%s drain under memory pressure: %s", g[i].name, st_vkr(vr));
   for (int i = 0; i < 2; i++)
      st_ring_destroy(&g[i]);
}

int
t_mem_limit(struct st_run *r)
{
   struct st_dev d;
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   struct st_req req = {.nqueues = caps.queues > 1 ? 2 : 1};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   uint64_t avail, total;
   meminfo(&avail, &total);
   VkDeviceSize heap = caps.heap;
   st_metric(r, "heap_gib", heap / 1073741824.0);

   /* Impossible requests. */
   over_ask(r, &d, heap * 2 + (1ull << 30), "allocation twice the heap");
   if (caps.max_alloc && caps.max_alloc < heap * 2)
      over_ask(r, &d, caps.max_alloc + (64ull << 20), "allocation above maxMemoryAllocationSize");

   /* A cap that leaves the rest of the system alone. */
   uint64_t reserve = total / 4 > (1536ull << 20) ? total / 4 : (1536ull << 20);
   uint64_t cap = heap / 10 * 6;
   if (avail && avail > reserve && avail - reserve < cap)
      cap = avail - reserve;
   else if (avail && avail <= reserve)
      cap = 0;
   st_metric(r, "hold_cap_gib", cap / 1073741824.0);

   struct st_buf probe;
   VkCommandBuffer cb;
   if ((vr = st_buf_create(&d, 16 * 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &probe)) ||
       (vr = st_cb_alloc(&d, 0, &cb)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   size_t maxc = cap / CHUNK + 1;
   struct chunk *c = calloc(maxc, sizeof(*c));
   uint64_t cycles = 0, peak = 0;
   while (st_time_left(r) && !r->errors && cap >= CHUNK) {
      size_t n = 0;
      uint64_t bad = 0;
      while (n < maxc && (n + 1) * CHUNK <= cap) {
         vr = chunk_alloc(&d, cb, &probe, &c[n], st_rand(r), &bad);
         if (vr == VK_ERROR_OUT_OF_DEVICE_MEMORY || vr == VK_ERROR_OUT_OF_HOST_MEMORY) {
            st_metric_add(r, "refused_below_cap", 1);
            break;
         }
         if (vr) {
            st_fail(r, "chunk %zu (%.2f GiB held): %s", n, n * CHUNK / 1073741824.0, st_vkr(vr));
            break;
         }
         if (bad)
            st_fail(r, "chunk %zu: GPU fill did not read back", n);
         r->checks++;
         n++;
      }
      if (n * CHUNK > peak)
         peak = n * CHUNK;
      if (!r->errors)
         work_while_holding(r, &d, st_part(r, 0.15, 2 * ST_NS));
      for (size_t i = 0; i < n; i++)
         chunk_free(&d, &c[i]);
      cycles++;
      r->iterations++;
   }
   st_metric(r, "held_gib", peak / 1073741824.0);
   st_metric(r, "cycles", (double)cycles);
   free(c);
   if (!r->errors) {
      work_while_holding(r, &d, ST_NS); /* and after everything is freed */
      st_buf_destroy(&d, &probe);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}

int
t_mem_exhaust(struct st_run *r)
{
   struct st_dev d;
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   struct st_req req = {.nqueues = 1};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   struct st_buf probe;
   VkCommandBuffer cb;
   if ((vr = st_buf_create(&d, 16 * 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &probe)) ||
       (vr = st_cb_alloc(&d, 0, &cb)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   size_t maxc = caps.heap * 2 / CHUNK + 1;
   struct chunk *c = calloc(maxc, sizeof(*c));
   size_t n = 0;
   uint64_t bad = 0;
   VkResult stop = VK_SUCCESS;
   while (n < maxc && st_time_left(r)) {
      stop = chunk_alloc(&d, cb, &probe, &c[n], st_rand(r), &bad);
      if (stop)
         break;
      r->checks++;
      n++;
   }
   st_metric(r, "held_gib", n * CHUNK / 1073741824.0);
   if (stop == VK_TIMEOUT)
      return st_fail(r, "a fill at %.2f GiB never completed", n * CHUNK / 1073741824.0);
   if (stop && stop != VK_ERROR_OUT_OF_DEVICE_MEMORY && stop != VK_ERROR_OUT_OF_HOST_MEMORY)
      st_fail(r, "at %.2f GiB: %s, expected an out-of-memory error", n * CHUNK / 1073741824.0, st_vkr(stop));
   if (bad)
      st_fail(r, "%llu chunks read back wrong", (unsigned long long)bad);
   for (size_t i = 0; i < n; i++)
      chunk_free(&d, &c[i]);
   free(c);
   if (r->errors)
      return ST_FAIL;
   st_buf_destroy(&d, &probe);
   st_dev_destroy(&d);
   struct st_client_cfg cfg = st_client_defaults(r);
   cfg.render = true;
   cfg.iters = 4;
   int rc = st_wait_pid(st_fork(st_client_main, &cfg), 60 * ST_NS, NULL);
   r->checks++;
   if (rc != 0)
      st_fail(r, "a fresh client after exhaustion failed (exit %d)", rc);
   r->iterations = 1;
   return st_verdict(r);
}

/* ======================= mem.churn ======================= */

struct small {
   VkBuffer buf;
   VkDeviceMemory mem;
   VkDeviceSize size;
   uint32_t *map;
   uint32_t seed;
};

static uint32_t
pat(uint32_t seed, uint32_t i)
{
   return (seed ^ (i * 2246822519u)) + i;
}

int
t_mem_churn(struct st_run *r)
{
   struct st_dev d;
   struct st_req req = {.nqueues = 1};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   uint32_t max_n = 2048;
   if (d.props.limits.maxMemoryAllocationCount < max_n + 64)
      max_n = d.props.limits.maxMemoryAllocationCount > 128 ? d.props.limits.maxMemoryAllocationCount - 64 : 64;
   struct small *s = calloc(max_n, sizeof(*s));
   struct st_buf result;
   VkCommandBuffer cb;
   if ((vr = st_buf_create(&d, (VkDeviceSize)max_n * 128, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &result)) ||
       (vr = st_cb_alloc(&d, 0, &cb)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   uint64_t allocs = 0, maps = 0;
   while (st_time_left(r) && !r->errors) {
      uint32_t n = st_rand_range(r, max_n / 4, max_n);
      for (uint32_t i = 0; i < n && !r->errors; i++) {
         struct small *b = &s[i];
         b->size = (VkDeviceSize)st_rand_range(r, 1, 64) * 4096;
         b->seed = st_rand(r);
         VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = b->size,
                                  .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
         if ((vr = vkCreateBuffer(d.dev, &bi, NULL, &b->buf)))
            return st_fail(r, "buffer %u: %s", i, st_vkr(vr));
         VkMemoryRequirements mr;
         vkGetBufferMemoryRequirements(d.dev, b->buf, &mr);
         VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                    .memoryTypeIndex = mem_type(&d, mr.memoryTypeBits, d.mem_host)};
         if ((vr = vkAllocateMemory(d.dev, &ai, NULL, &b->mem)) || (vr = vkBindBufferMemory(d.dev, b->buf, b->mem, 0)))
            return st_fail(r, "allocation %u of %u (%llu KiB): %s", i, n, (unsigned long long)(b->size >> 10),
                           st_vkr(vr));
         void *p;
         if ((vr = vkMapMemory(d.dev, b->mem, 0, VK_WHOLE_SIZE, 0, &p)))
            return st_fail(r, "map %u: %s", i, st_vkr(vr));
         b->map = p;
         for (uint32_t w = 0; w < b->size / 4; w++)
            b->map[w] = pat(b->seed, w);
         /* Half are unmapped before the GPU uses them, and mapped again to check. */
         if (st_rand_range(r, 0, 1)) {
            vkUnmapMemory(d.dev, b->mem);
            b->map = NULL;
         }
         maps++;
         allocs++;
      }
      /* The GPU reads the first and last 64 bytes of each buffer and overwrites bytes 64..127. */
      memset(result.map, 0, result.size);
      st_cb_begin(cb);
      for (uint32_t i = 0; i < n; i++) {
         VkBufferCopy reg[2] = {{0, (VkDeviceSize)i * 128, 64}, {s[i].size - 64, (VkDeviceSize)i * 128 + 64, 64}};
         vkCmdCopyBuffer(cb, s[i].buf, result.buf, 2, reg);
      }
      st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      for (uint32_t i = 0; i < n; i++)
         vkCmdFillBuffer(cb, s[i].buf, 64, 64, s[i].seed ^ 0xa5a5a5a5u);
      st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                 VK_ACCESS_2_HOST_READ_BIT);
      vkEndCommandBuffer(cb);
      if ((vr = st_run_cb(&d, 0, cb, 20 * ST_NS)))
         return st_fail(r, "copies from %u buffers: %s", n, st_vkr(vr));
      const uint32_t *res = result.map;
      for (uint32_t i = 0; i < n && !r->errors; i++) {
         uint32_t last = (uint32_t)(s[i].size / 4) - 16;
         for (uint32_t w = 0; w < 16; w++) {
            if (res[i * 32 + w] != pat(s[i].seed, w) || res[i * 32 + 16 + w] != pat(s[i].seed, last + w)) {
               st_fail(r, "buffer %u (%llu KiB): GPU read %08x/%08x, CPU wrote %08x/%08x", i,
                       (unsigned long long)(s[i].size >> 10), res[i * 32 + w], res[i * 32 + 16 + w], pat(s[i].seed, w),
                       pat(s[i].seed, last + w));
               break;
            }
         }
         if (!s[i].map) {
            void *p;
            if ((vr = vkMapMemory(d.dev, s[i].mem, 0, VK_WHOLE_SIZE, 0, &p)))
               return st_fail(r, "map %u again: %s", i, st_vkr(vr));
            s[i].map = p;
            maps++;
         }
         for (uint32_t w = 16; w < 32; w++)
            if (s[i].map[w] != (s[i].seed ^ 0xa5a5a5a5u)) {
               st_fail(r, "buffer %u: GPU fill not visible to the CPU (word %u = %08x)", i, w, s[i].map[w]);
               break;
            }
         r->checks++;
      }
      /* Free in random order. */
      for (uint32_t i = n; i > 0; i--) {
         uint32_t j = st_rand_range(r, 0, i - 1);
         struct small t = s[j];
         s[j] = s[i - 1];
         s[i - 1] = t;
         if (s[i - 1].map)
            vkUnmapMemory(d.dev, s[i - 1].mem);
         vkDestroyBuffer(d.dev, s[i - 1].buf, NULL);
         vkFreeMemory(d.dev, s[i - 1].mem, NULL);
         memset(&s[i - 1], 0, sizeof(s[0]));
      }
      r->iterations++;
   }
   /* Map/unmap churn on one allocation. */
   struct st_buf one;
   if (!r->errors && !(vr = st_buf_create(&d, 1 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &one))) {
      vkUnmapMemory(d.dev, one.mem);
      one.map = NULL;
      for (uint32_t i = 0; i < 2000 && !r->errors; i++) {
         void *p;
         VkDeviceSize off = (VkDeviceSize)st_rand_range(r, 0, 255) * 4096;
         if ((vr = vkMapMemory(d.dev, one.mem, off, 4096, 0, &p))) {
            st_fail(r, "map/unmap churn %u: %s", i, st_vkr(vr));
            break;
         }
         ((volatile uint32_t *)p)[0] = i;
         if (((volatile uint32_t *)p)[0] != i)
            st_fail(r, "map/unmap churn %u: readback mismatch", i);
         vkUnmapMemory(d.dev, one.mem);
         maps++;
      }
      st_buf_destroy(&d, &one);
   }
   st_metric(r, "allocations", (double)allocs);
   st_metric(r, "maps", (double)maps);
   free(s);
   if (!r->errors) {
      st_buf_destroy(&d, &result);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}
