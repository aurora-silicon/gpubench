// SPDX-License-Identifier: GPL-2.0-only
/* A ring of `depth` checked submissions on one queue with one timeline semaphore: submission i
 * signals i + 1 and uses slot i % depth. Each completed slot is verified before it is reused. */
#pragma once
#include "st.h"

#define ST_RING_MAX 256

enum st_ring_kind {
   ST_RING_RENDER, /* render check (st_rcheck) per slot */
   ST_RING_CHAIN,  /* chain compute over a per-slot region, initialised by the CPU */
   ST_RING_LCG,    /* long ALU dispatch per slot */
};

struct st_ring {
   struct st_dev *d;
   int qi, depth;
   enum st_ring_kind kind;
   VkSemaphore tl;
   uint64_t next, checked;
   VkCommandBuffer cb[ST_RING_MAX];
   uint32_t k[ST_RING_MAX];
   uint64_t submit_ns[ST_RING_MAX];
   uint64_t last_progress;
   double max_ms;
   /* kind resources */
   struct st_rcheck *rc;   /* render: depth checks */
   struct st_buf chain_buf;
   struct st_chain chain;
   uint32_t chain_n;       /* elements per slot */
   struct st_lcg *lcg;     /* lcg: depth outputs */
   uint32_t lcg_iters;
   char name[32];
};

/* render: w x h targets; chain: n elements per slot; lcg: groups x 64 invocations, iters. */
VkResult st_ring_init(struct st_ring *g, struct st_dev *d, int qi, int depth, enum st_ring_kind kind, uint32_t a,
                      uint32_t b);
/* Verifies what completed and refills free slots (when `submit`). Returns VK_SUCCESS, the first
 * submission error, or VK_TIMEOUT when work is outstanding and nothing completed for stall_ns.
 * Wrong results are recorded in r (st_fail) and counted as checks. */
VkResult st_ring_pump(struct st_ring *g, struct st_run *r, bool submit, uint64_t stall_ns);
/* Pumps without submitting until everything is verified or timeout_ns passes without progress. */
VkResult st_ring_drain(struct st_ring *g, struct st_run *r, uint64_t timeout_ns);
static inline bool st_ring_busy(const struct st_ring *g) { return g->checked < g->next; }
void st_ring_destroy(struct st_ring *g);
