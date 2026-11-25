// SPDX-License-Identifier: GPL-2.0-only
/* Shared input generators and deterministic CPU reference oracles for the
 * Wave-3 compute and MSAA benchmarks (compute.blur_fp32, compute.corner_ring,
 * compute.desc_search, compute.chain, render.msaa_merge). */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gb_common.h"

/* ---- 1. compute.blur_fp32 ---- */
void gb_blur_fp32_fill_input(uint8_t *rgba, uint32_t w, uint32_t h);
uint32_t gb_blur_fp32_ref(uint32_t taps, const uint8_t *in_rgba, uint8_t *out_rgba, uint32_t w, uint32_t h);

/* ---- 2. compute.corner_ring ---- */
void gb_corner_ring_fill_input(uint8_t *gray, uint32_t w, uint32_t h);
uint32_t gb_corner_ring_ref(bool dense_ring, int32_t thresh, const uint8_t *gray, uint32_t *out_scores,
                            uint32_t w, uint32_t h);

/* ---- 3. compute.desc_search ---- */
void gb_desc_search_fill(uint32_t num_queries, uint32_t num_cands, uint32_t stride,
                         float *queries, float *cands_atlas, float *weights);
uint32_t gb_desc_search_ref(uint32_t num_queries, uint32_t num_cands, uint32_t stride, uint32_t flags,
                            const float *queries, const float *cands_atlas, const float *weights,
                            uint32_t *out_res);

/* ---- 4. compute.chain ---- */
void gb_chain_fill_init(uint32_t count, uint32_t *init_buf);
uint32_t gb_chain_ref(uint32_t count, uint32_t passes, const uint32_t *init_buf, uint32_t *out_buf);

/* ---- 5. render.msaa_merge ---- */
uint32_t gb_msaa_merge_ref_rgba8(uint32_t w, uint32_t h, uint32_t passes, uint32_t *out_rgba8);
uint32_t gb_msaa_merge_ref_rgba16f(uint32_t w, uint32_t h, uint32_t passes, uint16_t *out_rgba16f);

/* ---- 6. compute.deferred_light ---- */
struct gb_dlight_input {
   uint32_t w, h;
   uint32_t max_lights; /* slots per tile list, terminator included */
   uint32_t shadow_dim;
   uint32_t *g[4];      /* w * h words each: four G-buffer layers */
   uint32_t *depth;     /* w * h words; 0 = sky */
   uint32_t *lights;    /* GB_DLIGHT_TILE_LISTS * max_lights * 2 uvec4s */
   uint32_t *shadow;    /* shadow_dim^2 words */
};
/* Allocates and fills `in`; `lights` is the per-tile list length without the
 * terminator. */
void gb_deferred_light_fill(struct gb_dlight_input *in, uint32_t w, uint32_t h, uint32_t lights);
void gb_deferred_light_free(struct gb_dlight_input *in);
/* Writes w * h uvec2s to `out` and returns their CRC-32. */
uint32_t gb_deferred_light_ref(const struct gb_dlight_input *in, uint32_t mats, uint32_t *out);
