// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include "st.h"

struct st_client_cfg {
   int device;
   int nqueues;          /* 0 = 2 (cross-queue pairs need two; one queue still works) */
   int iters;            /* <= 0: until until_ns */
   uint64_t until_ns;    /* absolute st_now() deadline for the loop (0: none) */
   uint64_t seed;
   uint32_t spin;        /* producer LCG steps: makes the producer slow */
   bool render;          /* add a render check per iteration */
   uint32_t rw, rh;      /* render size (default 128 x 128) */
   uint64_t gap_ns;      /* idle between iterations */
   uint64_t timeout_ns;  /* per-iteration GPU wait (default 10 s) */
   bool exit_after_submit; /* _exit(0) with the first iteration in flight */
   bool exit_after_wait;   /* _exit(0) right after the last wait returns, without teardown */
   bool no_teardown;
   bool verbose;
   int report_fd;        /* writes 'S' after the first submission (-1: none) */
   int result_fd;        /* st_client_main writes struct st_client_result here (-1: none) */
};

struct st_client_result {
   int rc;
   VkResult vk;
   uint64_t iterations, checks, bad;
   double max_ms;
   char why[256];
};

static inline struct st_client_cfg
st_client_defaults(const struct st_run *r)
{
   struct st_client_cfg c = {0};
   c.device = r->o->device;
   c.report_fd = -1;
   c.result_fd = -1;
   c.verbose = r->o->verbose;
   c.seed = r->rng;
   return c;
}

int st_client(const struct st_client_cfg *cfg, struct st_client_result *res);
int st_client_main(void *arg);
