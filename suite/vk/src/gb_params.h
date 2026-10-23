// SPDX-License-Identifier: GPL-2.0-only
/* Workload dimensions shared by the Vulkan and Metal executors. Changing a
 * value here changes the benchmark: bump the result's meaning in README. */
#pragma once

#define GB_LIGHT_LOCAL 32u              /* compute.light: 1 group of 32 */
#define GB_ALU_GROUPS 1024u             /* compute.heavy kind=alu */
#define GB_ALU_LOCAL 256u
#define GB_ALU_ITERS 128u               /* x 32 FMA per iteration */
#define GB_ALU_A 0.999f
#define GB_ALU_B 0.001f
#define GB_ALU_FLOP ((double)GB_ALU_GROUPS * GB_ALU_LOCAL * GB_ALU_ITERS * 32.0 * 2.0)
#define GB_BW_BYTES (64ull << 20)       /* compute.heavy kind=bw: read + write */
#define GB_BW_LOCAL 256u
#define GB_LIGHT_DIM 64u                /* render.light target */
#define GB_FILL_LAYERS 16u              /* render.heavy kind=fill */
#define GB_GEOM_TRIS 262144u            /* render.heavy kind=geom */
#define GB_TVB_W 1920u                  /* render.tvb target */
#define GB_TVB_H 1080u
#define GB_COPY_BYTES (64ull << 20)     /* copy.bw */
#define GB_COPY_DIM 4096u               /* copy.bw kind=buf2img, RGBA8 */
#define GB_QRING 128u                   /* query ring */
#define GB_HOST_SIGNAL_DELAY_NS 200000ull /* sync.signal kind=host: park time before the signal */
#define GB_PING_RING 8                  /* sync.signal kind=queue: rounds in flight */
#define GB_CR_DEPTH 2                   /* concurrent.cr mode=queues: per-queue depth */
#define GB_IL_DEPTH 4                   /* concurrent.cr mode=interleave */
#define GB_TOUCH_STRIDE 16384           /* mem.*: one write per 16 KiB page */
#define GB_IDLE_BUDGET_MS 1500           /* idle.wake: samples = budget / gap_ms, clamped */
#define GB_IDLE_MIN_SAMPLES 6
#define GB_IDLE_MAX_SAMPLES 40
