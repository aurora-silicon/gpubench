// SPDX-License-Identifier: GPL-2.0-only
/* Sparse binding churn: pages of a sparse buffer bound to random pages of a memory pool,
 * filled through the sparse buffer, read back through it and through a plain buffer aliasing
 * the pool, then unbound. The sizes stay small on purpose. */
#include "st.h"

#include <stdlib.h>
#include <string.h>

#define SP_BUFFER (64ull << 20)
#define SP_POOL_PAGES 256u
#define SP_BOUND 64u

static uint32_t
spval(uint32_t page, uint64_t iter)
{
   return (page * 2654435761u) ^ (uint32_t)(iter * 40503u) ^ 0x5a5a0000u;
}

/* After a pool mismatch: which pool pages hold the value instead (diagnostic). */
static void
pool_scan(struct st_run *r, struct st_dev *d, VkBuffer alias, VkDeviceSize page, uint32_t want)
{
   struct st_buf all;
   VkCommandBuffer cb;
   if (st_buf_create(d, page * SP_POOL_PAGES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &all) || st_cb_alloc(d, 0, &cb))
      return;
   st_cb_begin(cb);
   VkBufferCopy c = {0, 0, page * SP_POOL_PAGES};
   vkCmdCopyBuffer(cb, alias, all.buf, 1, &c);
   st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
              VK_ACCESS_2_HOST_READ_BIT);
   vkEndCommandBuffer(cb);
   if (st_run_cb(d, 0, cb, 10 * ST_NS) == VK_SUCCESS) {
      const uint32_t *w = all.map;
      int found = 0;
      for (uint32_t p = 0; p < SP_POOL_PAGES; p++)
         if (w[p * (page / 4)] == want) {
            fprintf(stderr, "  [sparse.churn] value found at pool page %u\n", p);
            found++;
         }
      fprintf(stderr, "  [sparse.churn] %d pool pages hold the value\n", found);
   }
   st_buf_destroy(d, &all);
   (void)r;
}

int
t_sparse_churn(struct st_run *r)
{
   struct st_caps caps;
   if (st_probe(r->o->device, &caps) != VK_SUCCESS)
      return st_fail(r, "probe failed");
   if (!caps.sparse)
      return st_skip(r, "sparse binding with buffer residency is not supported");
   struct st_dev d;
   struct st_req req = {.nqueues = 1, .sparse = true};
   VkResult vr = st_dev_create(&d, r->o->device, &req);
   if (vr)
      return st_fail(r, "device: %s", st_vkr(vr));
   VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
      .size = SP_BUFFER,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer sb;
   if ((vr = vkCreateBuffer(d.dev, &bi, NULL, &sb)))
      return st_fail(r, "sparse buffer: %s", st_vkr(vr));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(d.dev, sb, &mr);
   VkDeviceSize page = mr.alignment;
   uint32_t npages = (uint32_t)(SP_BUFFER / page);
   uint32_t mt = UINT32_MAX;
   for (uint32_t i = 0; i < d.mp.memoryTypeCount; i++)
      if (mr.memoryTypeBits & (1u << i)) {
         mt = i;
         if (d.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            break;
      }
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = page * SP_POOL_PAGES,
                              .memoryTypeIndex = mt};
   VkDeviceMemory pool;
   if (mt == UINT32_MAX || (vr = vkAllocateMemory(d.dev, &ai, NULL, &pool)))
      return st_fail(r, "pool memory: %s", st_vkr(vr));
   /* A plain buffer over the whole pool, to read what landed in each physical page. */
   VkBufferCreateInfo ab = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = page * SP_POOL_PAGES,
                            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
   VkBuffer alias = VK_NULL_HANDLE;
   bool have_alias = !vkCreateBuffer(d.dev, &ab, NULL, &alias);
   if (have_alias) {
      VkMemoryRequirements amr;
      vkGetBufferMemoryRequirements(d.dev, alias, &amr);
      have_alias = (amr.memoryTypeBits & (1u << mt)) && !vkBindBufferMemory(d.dev, alias, pool, 0);
   }
   struct st_buf rb;
   VkCommandBuffer cb;
   VkSemaphore tl;
   if ((vr = st_buf_create(&d, SP_BOUND * 2 * 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, &rb)) ||
       (vr = st_cb_alloc(&d, 0, &cb)) || (vr = st_timeline(&d, 0, &tl)))
      return st_fail(r, "setup: %s", st_vkr(vr));
   uint32_t bpage[SP_BOUND] = {0}, ppage[SP_BOUND] = {0};
   uint32_t nbound = 0;
   uint64_t value = 0, binds = 0;
   VkSparseMemoryBind mb[2 * SP_BOUND];
   for (uint64_t iter = 0; st_time_left(r) && !r->errors; iter++) {
      /* Unbind the previous set and bind a new one in one operation. */
      int nb = 0;
      for (uint32_t i = 0; i < nbound; i++)
         mb[nb++] = (VkSparseMemoryBind){.resourceOffset = bpage[i] * page, .size = page, .memory = VK_NULL_HANDLE};
      uint32_t used[SP_POOL_PAGES / 32] = {0};
      /* New pages differ from the ones being unbound, so the result does not depend on the
       * order of the ranges within one bind. */
      uint32_t prev[SP_BOUND], nprev = nbound;
      memcpy(prev, bpage, sizeof(prev));
      nbound = st_rand_range(r, 1, SP_BOUND);
      for (uint32_t i = 0; i < nbound; i++) {
         uint32_t b, p;
         bool dup;
         do {
            b = st_rand_range(r, 0, npages - 1);
            dup = false;
            for (uint32_t k = 0; k < i; k++)
               dup |= bpage[k] == b;
            for (uint32_t k = 0; k < nprev; k++)
               dup |= prev[k] == b;
         } while (dup);
         do
            p = st_rand_range(r, 0, SP_POOL_PAGES - 1);
         while (used[p / 32] & (1u << (p % 32)));
         used[p / 32] |= 1u << (p % 32);
         bpage[i] = b;
         ppage[i] = p;
         mb[nb++] = (VkSparseMemoryBind){.resourceOffset = b * page, .size = page, .memory = pool,
                                         .memoryOffset = p * page};
      }
      VkSparseBufferMemoryBindInfo sbi = {.buffer = sb, .bindCount = (uint32_t)nb, .pBinds = mb};
      uint64_t bv = ++value;
      VkTimelineSemaphoreSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                          .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &bv};
      VkBindSparseInfo bsi = {.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .pNext = &ts, .bufferBindCount = 1,
                              .pBufferBinds = &sbi, .signalSemaphoreCount = 1, .pSignalSemaphores = &tl};
      if ((vr = vkQueueBindSparse(d.q[0], 1, &bsi, VK_NULL_HANDLE)))
         return st_fail(r, "iteration %llu: bind %d ranges: %s", (unsigned long long)iter, nb, st_vkr(vr));
      binds += nb;
      /* Fill each bound page, then read 256 bytes back through the sparse buffer and through the
       * alias of its pool page. */
      memset(rb.map, 0, rb.size);
      st_cb_begin(cb);
      for (uint32_t i = 0; i < nbound; i++)
         vkCmdFillBuffer(cb, sb, bpage[i] * page, page, spval(bpage[i], iter));
      st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      for (uint32_t i = 0; i < nbound; i++) {
         VkBufferCopy c = {.srcOffset = bpage[i] * page + page - 256, .dstOffset = i * 512ull, .size = 256};
         vkCmdCopyBuffer(cb, sb, rb.buf, 1, &c);
         if (have_alias) {
            VkBufferCopy a = {.srcOffset = ppage[i] * page, .dstOffset = i * 512ull + 256, .size = 256};
            vkCmdCopyBuffer(cb, alias, rb.buf, 1, &a);
         }
      }
      st_barrier(cb, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
                 VK_ACCESS_2_HOST_READ_BIT);
      vkEndCommandBuffer(cb);
      struct st_sem_op w = {tl, bv}, s = {tl, bv + 1};
      value++;
      if ((vr = st_submit(d.q[0], cb, &w, 1, &s, 1, VK_NULL_HANDLE)))
         return st_fail(r, "iteration %llu: submit %s", (unsigned long long)iter, st_vkr(vr));
      if ((vr = st_wait_value(&d, tl, value, 10 * ST_NS)))
         return st_fail(r, "iteration %llu: wait %s", (unsigned long long)iter, st_vkr(vr));
      const uint32_t *v = rb.map;
      for (uint32_t i = 0; i < nbound && !r->errors; i++) {
         uint32_t want = spval(bpage[i], iter);
         for (uint32_t w2 = 0; w2 < (have_alias ? 128u : 64u); w2++)
            if (v[i * 128 + w2] != want) {
               st_fail(r, "iteration %llu: buffer page %u (pool page %u) %s word %u = %08x, expected %08x",
                       (unsigned long long)iter, bpage[i], ppage[i], w2 < 64 ? "via the sparse buffer" : "via the pool",
                       w2 % 64, v[i * 128 + w2], want);
               if (w2 >= 64)
                  pool_scan(r, &d, alias, page, want);
               break;
            }
         r->checks++;
      }
      r->iterations++;
   }
   st_metric(r, "bind_ranges", (double)binds);
   st_metric(r, "page_kib", page / 1024.0);
   st_metric(r, "pool_alias_checked", have_alias);
   if (!r->errors) {
      vkQueueWaitIdle(d.q[0]);
      if (alias)
         vkDestroyBuffer(d.dev, alias, NULL);
      vkDestroyBuffer(d.dev, sb, NULL);
      vkFreeMemory(d.dev, pool, NULL);
      vkDestroySemaphore(d.dev, tl, NULL);
      st_buf_destroy(&d, &rb);
      st_dev_destroy(&d);
   }
   return st_verdict(r);
}
