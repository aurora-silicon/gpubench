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
#define GB_TEX_DIM 256u                 /* render.heavy kind=texture */
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

/* ---- 90%-coverage Wave-1 workload dimensions ---- */
#define GB_UBER_GROUPS 256u              /* compute.uber: 256 x 256 = 65536 invocations */
#define GB_UBER_LOCAL 256u
#define GB_UBER_THREADS (GB_UBER_GROUPS * GB_UBER_LOCAL)
#define GB_UBER_ITERS 48u                /* narrow post-branch loop iterations */
#define GB_UBER_ILP_WORDS (4u << 20)     /* 16 MiB input table for kind=ilp */
#define GB_UBER_SEED 0x13579bdfu

#define GB_CONV_LOCAL 16u                /* compute.conv2d: 16x16 cooperative halo tile */
#define GB_CONV_RADIUS 8u                /* 17-tap separable filter */

#define GB_STEREO_W 450u                 /* compute.stereo: matches Geekbench Stereo */
#define GB_STEREO_H 375u
#define GB_STEREO_PITCH 512u
#define GB_STEREO_RADIUS 5u              /* 11x11 SAD window */
#define GB_STEREO_LOCAL 16u

#define GB_EDGE_LOCAL 16u                /* compute.edge: 16x16 + 1px halo = 18x18 tile */

#define GB_DEFERRED_QUADS 4096u          /* render.deferred: 4096 quads = 8192 tris, 4 layers */

#define GB_MDI_DIM 1024u                 /* render.mdi: 1024x1024 RGBA8 target */
#define GB_MDI_TEX_DIM 16u               /* 16x16 bindless textures */
#define GB_MDI_MAX_BINDLESS 1024u
#define GB_MDI_MAX_DRAWS 10000u

#define GB_PSO_DEFAULT_COUNT 32u         /* compile.pso: default variant batch */

/* ---- 90%-coverage Wave-2 heavy-game / AAA workload dimensions ---- */
#define GB_SHADOW_CASCADES 4u            /* render.shadow: 4 CSM cascades */
#define GB_SHADOW_RECV_DIM 1024u         /* render.shadow: 1024x1024 PCF receiver */
#define GB_SHADOW_DEFAULT_INST 2048u     /* render.shadow: instances per cascade */

#define GB_CULL_DIM 1024u                /* compute.cull_mdi: 1024x1024 target */
#define GB_CULL_GRID 128u                /* 128x128 = 16384 disjoint 8x8 quads */
#define GB_CULL_MAX_OBJS 16384u
#define GB_CULL_LOCAL 64u

#define GB_BUBO_DIM 1024u                /* render.bindless_ubo: 1024x1024 target */
#define GB_BUBO_BANKS 16u                /* 16 UBO banks of vec4[1024] (256 KiB total) */
#define GB_BUBO_ENTRIES 1024u
#define GB_BUBO_MAX_DRAWS 2048u

#define GB_FP16_GEMV_M 4096u             /* compute.fp16 mode=gemv: 4096x4096 u8->fp16 */
#define GB_FP16_GEMV_K 4096u
#define GB_FP16_GEMV_LOCAL 64u           /* 4 rows/workgroup x 16 lanes/row */
#define GB_FP16_ALUMIX_GROUPS 256u       /* compute.fp16 mode=alu_mix: 65536 threads */
#define GB_FP16_ALUMIX_LOCAL 256u
#define GB_FP16_ALUMIX_ITERS 256u        /* 96 FLOPs/iter (64 FP16 + 32 FP32) */
#define GB_FP16_ALUMIX_FLOP ((double)GB_FP16_ALUMIX_GROUPS * GB_FP16_ALUMIX_LOCAL * GB_FP16_ALUMIX_ITERS * 96.0)

#define GB_XFER_4M_DIM 1024u             /* transfer.ops kind=clear_4m (4 MiB) */
#define GB_XFER_16M_DIM 2048u            /* transfer.ops kind=clear_16m / copy_b2i (16 MiB) */
#define GB_XFER_MIP_W 2560u              /* transfer.ops kind=mipchain (1440p -> 1x1) */
#define GB_XFER_MIP_H 1440u
#define GB_XFER_MIP_LEVELS 12u

#define GB_POSTFX_QUADS 4096u            /* render.postfx: 64x64 HDR scene quads */
#define GB_POSTFX_BLOOM_MIPS 5u          /* 5-level bloom pyramid (res/2 .. res/32) */

/* ---- Wave-3 AGX3 / M4-M5 compute & MSAA workload dimensions ---- */
#define GB_BLUR_LOCAL 16u                /* compute.blur_fp32: 16x16 workgroup, 2 pix/thread */
#define GB_CORNER_LOCAL 16u              /* compute.corner_ring: 16x16 workgroup */
#define GB_DESC_LOCAL 64u                /* compute.desc_search: 64 threads/workgroup */
#define GB_DESC_DIMS 32u                 /* 32-dim (8x vec4) feature descriptors */
#define GB_CHAIN_LOCAL 64u               /* compute.chain: 64 threads/workgroup */
#define GB_CHAIN_SMALL_WORDS 1024u       /* compute.chain size=small (16 workgroups) */
#define GB_CHAIN_MED_WORDS 16384u        /* compute.chain size=medium (256 workgroups) */
