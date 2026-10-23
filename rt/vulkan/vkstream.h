// SPDX-License-Identifier: GPL-2.0-only
/* Submission stream: one queue, a ring of `depth` slots (command pool + buffer each), one timeline
 * semaphore. Submission k signals value k; before recording k the CPU waits for value k - depth and
 * that wait's return is the observed completion of k - depth (suite/README.md, "Depth"). */
#pragma once
#include "../common/json.h"
#include "vkutil.h"

#define VK_STREAM_MAX 64

typedef struct vk_stream_ops {
   void *user;
   /* Record submission k into cb (already begun). timed: write GPU timestamps. */
   VkResult (*record)(void *user, uint32_t slot, uint64_t k, VkCommandBuffer cb, int timed);
   /* Submission k completed (called in order). */
   void (*complete)(void *user, uint32_t slot, uint64_t k, int timed);
} vk_stream_ops;

typedef struct vk_stream_stats {
   rt_vec rate;        /* completions per second, one value per window */
   rt_vec interval_us; /* observed completion to completion */
   rt_vec latency_us;  /* submit to observed completion */
   rt_vec record_us;
   rt_vec submit_us;
   uint64_t completions;
} vk_stream_stats;

typedef struct vk_stream {
   vk_ctx *c;
   uint32_t depth;
   VkSemaphore tl;
   VkCommandPool pool[VK_STREAM_MAX];
   VkCommandBuffer cb[VK_STREAM_MAX];
   uint64_t submit_ns[VK_STREAM_MAX];
   int timed[VK_STREAM_MAX];
   uint64_t next, done, last_done_ns;
   vk_stream_ops ops;
   vk_stream_stats *stats;  /* where observations go (NULL = discard) */
} vk_stream;

VkResult vk_stream_init(vk_stream *s, vk_ctx *c, uint32_t depth, vk_stream_ops ops);
void vk_stream_destroy(vk_stream *s);
VkResult vk_stream_submit(vk_stream *s, int timed);
VkResult vk_stream_drain(vk_stream *s);
/* Runs `windows` windows of `ms` milliseconds (or of `count` submissions when count != 0) on the
 * continuing stream; window rates go to st->rate. windows = 0 with ms = warm-up length discards. */
VkResult vk_stream_phase(vk_stream *s, uint32_t windows, uint32_t ms, uint32_t count, int timed,
                         vk_stream_stats *st);
void vk_stream_stats_free(vk_stream_stats *st);
