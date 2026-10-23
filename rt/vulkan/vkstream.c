// SPDX-License-Identifier: GPL-2.0-only
#include "vkstream.h"
#include "../common/util.h"
#include <string.h>

VkResult vk_stream_init(vk_stream *s, vk_ctx *c, uint32_t depth, vk_stream_ops ops)
{
   memset(s, 0, sizeof(*s));
   if (depth < 1) depth = 1;
   if (depth > VK_STREAM_MAX) depth = VK_STREAM_MAX;
   s->c = c;
   s->depth = depth;
   s->ops = ops;
   s->next = 1;
   VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                   .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
   VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
   VKC(vkCreateSemaphore(c->dev, &si, NULL, &s->tl));
   for (uint32_t i = 0; i < depth; i++) {
      VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                     .queueFamilyIndex = c->qfam};
      VKC(vkCreateCommandPool(c->dev, &pci, NULL, &s->pool[i]));
      VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = s->pool[i],
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1};
      VKC(vkAllocateCommandBuffers(c->dev, &ai, &s->cb[i]));
   }
   return VK_SUCCESS;
}

void vk_stream_destroy(vk_stream *s)
{
   if (!s->c) return;
   vk_stream_drain(s);
   for (uint32_t i = 0; i < s->depth; i++)
      if (s->pool[i]) vkDestroyCommandPool(s->c->dev, s->pool[i], NULL);
   if (s->tl) vkDestroySemaphore(s->c->dev, s->tl, NULL);
   memset(s, 0, sizeof(*s));
}

static VkResult wait_value(vk_stream *s, uint64_t v)
{
   VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                             .semaphoreCount = 1,
                             .pSemaphores = &s->tl,
                             .pValues = &v};
   VkResult r = vkWaitSemaphores(s->c->dev, &wi, 30ull * 1000000000ull);
   if (r == VK_TIMEOUT) fprintf(stderr, "timeline wait for %llu timed out (GPU hang?)\n", (unsigned long long)v);
   return r;
}

/* Waits for submission k and records its observation. */
static VkResult observe(vk_stream *s, uint64_t k)
{
   VkResult r = wait_value(s, k);
   if (r != VK_SUCCESS) return r;
   uint64_t now = rt_now_ns();
   uint32_t slot = (uint32_t)(k % s->depth);
   if (s->stats) {
      rt_vec_push(&s->stats->latency_us, (now - s->submit_ns[slot]) / 1e3);
      if (s->last_done_ns) rt_vec_push(&s->stats->interval_us, (now - s->last_done_ns) / 1e3);
      s->stats->completions++;
   }
   s->last_done_ns = now;
   s->done = k;
   if (s->ops.complete) s->ops.complete(s->ops.user, slot, k, s->timed[slot]);
   return VK_SUCCESS;
}

VkResult vk_stream_submit(vk_stream *s, int timed)
{
   uint64_t k = s->next;
   uint32_t slot = (uint32_t)(k % s->depth);
   if (k > s->depth && s->done < k - s->depth) VKC(observe(s, k - s->depth));
   uint64_t t0 = rt_now_ns();
   VKC(vkResetCommandPool(s->c->dev, s->pool[slot], 0));
   VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   VKC(vkBeginCommandBuffer(s->cb[slot], &bi));
   VKC(s->ops.record(s->ops.user, slot, k, s->cb[slot], timed));
   VKC(vkEndCommandBuffer(s->cb[slot]));
   uint64_t t1 = rt_now_ns();
   VkTimelineSemaphoreSubmitInfo ts = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                       .signalSemaphoreValueCount = 1,
                                       .pSignalSemaphoreValues = &k};
   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .pNext = &ts,
                      .commandBufferCount = 1,
                      .pCommandBuffers = &s->cb[slot],
                      .signalSemaphoreCount = 1,
                      .pSignalSemaphores = &s->tl};
   s->submit_ns[slot] = rt_now_ns();
   s->timed[slot] = timed;
   VKC(vkQueueSubmit(s->c->queue, 1, &si, VK_NULL_HANDLE));
   uint64_t t2 = rt_now_ns();
   if (s->stats) {
      rt_vec_push(&s->stats->record_us, (t1 - t0) / 1e3);
      rt_vec_push(&s->stats->submit_us, (t2 - s->submit_ns[slot]) / 1e3);
   }
   s->next++;
   return VK_SUCCESS;
}

VkResult vk_stream_drain(vk_stream *s)
{
   while (s->done + 1 < s->next) VKC(observe(s, s->done + 1));
   return VK_SUCCESS;
}

VkResult vk_stream_phase(vk_stream *s, uint32_t windows, uint32_t ms, uint32_t count, int timed,
                         vk_stream_stats *st)
{
   if (!windows) {
      s->stats = NULL;
      uint64_t end = rt_now_ns() + (uint64_t)ms * 1000000ull;
      while (rt_now_ns() < end) VKC(vk_stream_submit(s, timed));
      return VK_SUCCESS;
   }
   s->stats = st;
   for (uint32_t w = 0; w < windows; w++) {
      uint64_t start = rt_now_ns(), done0 = st->completions;
      if (count) {
         while (st->completions - done0 < count) VKC(vk_stream_submit(s, timed));
      } else {
         /* a window lasts run_ms, and at least until two completions were observed, so frames
          * longer than the window still give a rate */
         uint64_t end = start + (uint64_t)ms * 1000000ull;
         while (rt_now_ns() < end || st->completions - done0 < 2) VKC(vk_stream_submit(s, timed));
      }
      double secs = (rt_now_ns() - start) / 1e9;
      rt_vec_push(&st->rate, (double)(st->completions - done0) / secs);
   }
   s->stats = NULL;
   return VK_SUCCESS;
}

void vk_stream_stats_free(vk_stream_stats *st)
{
   rt_vec_free(&st->rate);
   rt_vec_free(&st->interval_us);
   rt_vec_free(&st->latency_us);
   rt_vec_free(&st->record_us);
   rt_vec_free(&st->submit_us);
   memset(st, 0, sizeof(*st));
}
