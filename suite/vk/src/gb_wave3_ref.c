// SPDX-License-Identifier: GPL-2.0-only
#include "gb_wave3_ref.h"
#include "gb_params.h"

#include <stdlib.h>
#include <string.h>

static inline int32_t
clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
   return v < lo ? lo : (v > hi ? hi : v);
}

/* =========================================================================
 * 1. compute.blur_fp32 CPU reference
 * ========================================================================= */

static const uint32_t BLUR_W9[9] = {
   1024u, 3072u, 7168u, 12288u, 18432u, 12288u, 7168u, 3072u, 1024u,
};

static const uint32_t BLUR_W17[17] = {
   16u, 86u, 342u, 1052u, 2554u, 4996u, 7914u, 10242u,
   11132u,
   10242u, 7914u, 4996u, 2554u, 1052u, 342u, 86u, 16u,
};

void
gb_blur_fp32_fill_input(uint8_t *rgba, uint32_t w, uint32_t h)
{
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         uint32_t s = (x * 73856093u) ^ (y * 19349663u) ^ 0x9e3779b9u;
         s ^= s >> 13;
         s *= 1664525u;
         uint32_t edge = (((x >> 4) ^ (y >> 4)) & 1u) ? 180u : 40u;
         uint32_t r = (edge + (s & 63u)) & 255u;
         uint32_t g = ((x * 3u + y * 5u + ((s >> 8) & 31u)) & 255u);
         uint32_t b = (((x ^ y) * 7u + ((s >> 16) & 63u)) & 255u);
         uint32_t a = (192u + ((s >> 24) & 63u)) & 255u;
         size_t off = ((size_t)y * w + x) * 4u;
         rgba[off + 0] = (uint8_t)r;
         rgba[off + 1] = (uint8_t)g;
         rgba[off + 2] = (uint8_t)b;
         rgba[off + 3] = (uint8_t)a;
      }
   }
}

uint32_t
gb_blur_fp32_ref(uint32_t taps, const uint8_t *in_rgba, uint8_t *out_rgba, uint32_t w, uint32_t h)
{
   const uint32_t *wt = (taps == 9u) ? BLUR_W9 : BLUR_W17;
   const int32_t rad = (taps == 9u) ? 4 : 8;
   const int32_t wi = (int32_t)w, hi = (int32_t)h;
   size_t bytes = (size_t)w * h * 4u;
   uint8_t *mid = malloc(bytes);

   /* Horizontal pass */
   for (int32_t y = 0; y < hi; y++) {
      for (int32_t x = 0; x < wi; x++) {
         uint32_t acc[4] = {0, 0, 0, 0};
         for (int32_t k = 0; k < (int32_t)taps; k++) {
            int32_t sx = clamp_i32(x + k - rad, 0, wi - 1);
            const uint8_t *p = in_rgba + ((size_t)y * w + (size_t)sx) * 4u;
            uint32_t wv = wt[k];
            acc[0] += (uint32_t)p[0] * wv;
            acc[1] += (uint32_t)p[1] * wv;
            acc[2] += (uint32_t)p[2] * wv;
            acc[3] += (uint32_t)p[3] * wv;
         }
         uint8_t *d = mid + ((size_t)y * w + (size_t)x) * 4u;
         for (int c = 0; c < 4; c++) {
            uint32_t v = (acc[c] + 32768u) >> 16;
            d[c] = (uint8_t)(v > 255u ? 255u : v);
         }
      }
   }

   /* Vertical pass */
   for (int32_t y = 0; y < hi; y++) {
      for (int32_t x = 0; x < wi; x++) {
         uint32_t acc[4] = {0, 0, 0, 0};
         for (int32_t k = 0; k < (int32_t)taps; k++) {
            int32_t sy = clamp_i32(y + k - rad, 0, hi - 1);
            const uint8_t *p = mid + ((size_t)sy * w + (size_t)x) * 4u;
            uint32_t wv = wt[k];
            acc[0] += (uint32_t)p[0] * wv;
            acc[1] += (uint32_t)p[1] * wv;
            acc[2] += (uint32_t)p[2] * wv;
            acc[3] += (uint32_t)p[3] * wv;
         }
         uint8_t *d = out_rgba + ((size_t)y * w + (size_t)x) * 4u;
         for (int c = 0; c < 4; c++) {
            uint32_t v = (acc[c] + 32768u) >> 16;
            d[c] = (uint8_t)(v > 255u ? 255u : v);
         }
      }
   }

   free(mid);
   return gb_crc32(out_rgba, bytes);
}

/* =========================================================================
 * 2. compute.corner_ring CPU reference
 * ========================================================================= */

static const int32_t RING16_X[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
static const int32_t RING16_Y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
static const int32_t RING8_X[8] = {0, 2, 2, 2, 0, -2, -2, -2};
static const int32_t RING8_Y[8] = {-2, -2, 0, 2, 2, 2, 0, -2};

void
gb_corner_ring_fill_input(uint8_t *gray, uint32_t w, uint32_t h)
{
   /* Synthetic scene mixing smooth gradients, 16x16 high-contrast checker
    * corners, small feature blobs, and deterministic noise so ~25-35% of
    * pixels pass the FAST compass check and ~8-15% trigger full arc scoring. */
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         uint32_t s = (x * 2654435761u) ^ (y * 2246822519u) ^ 0x517cc1b7u;
         s = (s ^ (s >> 16)) * 0x45d9f3bu;
         s ^= s >> 16;

         uint32_t v;
         uint32_t cell_x = x & 15u, cell_y = y & 15u;
         if (((x >> 4) ^ (y >> 4)) & 1u) {
            /* Bright tile with dark corner cutouts and small interior spots */
            v = 192u + (s & 15u);
            if (cell_x < 4u && cell_y < 4u)
               v = 48u + (s & 15u);
            else if (cell_x >= 9u && cell_x <= 11u && cell_y >= 9u && cell_y <= 11u)
               v = 36u + (s & 15u);
         } else {
            /* Dark tile with diagonal band and bright feature dots */
            v = 56u + (s & 15u);
            if (cell_x + cell_y < 5u)
               v = 208u + (s & 15u);
            else if (cell_x >= 6u && cell_x <= 8u && cell_y >= 10u && cell_y <= 12u)
               v = 220u + (s & 15u);
         }
         /* Modulate every 3rd band into smooth mid-tone to exercise early-reject */
         if (((x >> 5) + (y >> 5)) % 3u == 0u)
            v = 120u + ((x + y) & 15u);
         gray[(size_t)y * w + x] = (uint8_t)(v & 255u);
      }
   }
}

uint32_t
gb_corner_ring_ref(bool dense_ring, int32_t thresh, const uint8_t *gray, uint32_t *out_scores,
                   uint32_t w, uint32_t h)
{
   const int32_t wi = (int32_t)w, hi = (int32_t)h;
   for (int32_t y = 0; y < hi; y++) {
      for (int32_t x = 0; x < wi; x++) {
         size_t out_idx = (size_t)y * w + (size_t)x;
         if (x < 3 || y < 3 || x >= wi - 3 || y >= hi - 3) {
            out_scores[out_idx] = 0u;
            continue;
         }
         int32_t c = (int32_t)gray[out_idx];
         int32_t hi_t = c + thresh;
         int32_t lo_t = c - thresh;

         int32_t p0  = (int32_t)gray[(size_t)(y + RING16_Y[0])  * w + (size_t)(x + RING16_X[0])];
         int32_t p4  = (int32_t)gray[(size_t)(y + RING16_Y[4])  * w + (size_t)(x + RING16_X[4])];
         int32_t p8  = (int32_t)gray[(size_t)(y + RING16_Y[8])  * w + (size_t)(x + RING16_X[8])];
         int32_t p12 = (int32_t)gray[(size_t)(y + RING16_Y[12]) * w + (size_t)(x + RING16_X[12])];

         if (!dense_ring) {
            int32_t cb = (p0 > hi_t) + (p4 > hi_t) + (p8 > hi_t) + (p12 > hi_t);
            int32_t cd = (p0 < lo_t) + (p4 < lo_t) + (p8 < lo_t) + (p12 < lo_t);
            if (cb < 3 && cd < 3) {
               out_scores[out_idx] = 0u;
               continue;
            }
         }

         int32_t r[16];
         for (int i = 0; i < 16; i++)
            r[i] = (int32_t)gray[(size_t)(y + RING16_Y[i]) * w + (size_t)(x + RING16_X[i])];

         int32_t run_b = 0, max_b = 0;
         int32_t run_d = 0, max_d = 0;
         for (int i = 0; i < 24; i++) {
            int32_t v = r[i & 15];
            run_b = (v > hi_t) ? (run_b + 1) : 0;
            run_d = (v < lo_t) ? (run_d + 1) : 0;
            if (run_b > max_b) max_b = run_b;
            if (run_d > max_d) max_d = run_d;
         }

         int32_t score = 0;
         if (max_b >= 9) {
            int32_t best_b = 0;
            for (int k = 0; k < 16; k++) {
               int32_t m = 255;
               for (int j = 0; j < 9; j++) {
                  int32_t d = r[(k + j) & 15] - c;
                  if (d < m) m = d;
               }
               if (m > best_b) best_b = m;
            }
            if (best_b > score) score = best_b;
         }
         if (max_d >= 9) {
            int32_t best_d = 0;
            for (int k = 0; k < 16; k++) {
               int32_t m = 255;
               for (int j = 0; j < 9; j++) {
                  int32_t d = c - r[(k + j) & 15];
                  if (d < m) m = d;
               }
               if (m > best_d) best_d = m;
            }
            if (best_d > score) score = best_d;
         }

         if (dense_ring) {
            int32_t compass_contrast = abs(p0 - p8) + abs(p4 - p12);
            if (score > 0 || compass_contrast > thresh * 2) {
               int32_t in8[8];
               for (int j = 0; j < 8; j++)
                  in8[j] = (int32_t)gray[(size_t)(y + RING8_Y[j]) * w + (size_t)(x + RING8_X[j])];
               int32_t dc = abs(in8[0] - in8[4]) + abs(in8[1] - in8[5]) +
                            abs(in8[2] - in8[6]) + abs(in8[3] - in8[7]);
               int32_t rg = 0;
               for (int j = 0; j < 8; j++)
                  rg += abs(in8[j] - c);
               int32_t refine = clamp_i32(dc + (rg >> 2), 0, 1023);
               score = score * 16 + refine;
            }
         }

         out_scores[out_idx] = (uint32_t)(score > 0 ? score : 0);
      }
   }
   return gb_crc32(out_scores, (size_t)w * h * sizeof(uint32_t));
}

/* =========================================================================
 * 3. compute.desc_search CPU reference
 * ========================================================================= */

void
gb_desc_search_fill(uint32_t num_queries, uint32_t num_cands, uint32_t stride,
                    float *queries, float *cands_atlas, float *weights)
{
   uint32_t s = 0x13579bdfu;
   for (uint32_t i = 0; i < num_queries * GB_DESC_DIMS; i++) {
      s = s * 1664525u + 1013904223u;
      queries[i] = (float)((s >> 28) & 15u);
   }
   uint32_t atlas_rows = (num_cands + stride - 1u) / stride;
   uint32_t atlas_pitch = stride + 4u;
   memset(cands_atlas, 0, (size_t)atlas_rows * atlas_pitch * GB_DESC_DIMS * sizeof(float));
   for (uint32_t c = 0; c < num_cands; c++) {
      uint32_t row = c / stride;
      uint32_t col = c % stride;
      float *dst = cands_atlas + ((size_t)row * atlas_pitch + col) * GB_DESC_DIMS;
      for (uint32_t d = 0; d < GB_DESC_DIMS; d++) {
         s = s * 1664525u + 1013904223u;
         dst[d] = (float)((s >> 28) & 15u);
      }
   }
   for (uint32_t i = 0; i < 16u * GB_DESC_DIMS; i++) {
      s = s * 1664525u + 1013904223u;
      weights[i] = (float)(1u + ((s >> 31) & 1u));
   }
}

uint32_t
gb_desc_search_ref(uint32_t num_queries, uint32_t num_cands, uint32_t stride, uint32_t flags,
                   const float *queries, const float *cands_atlas, const float *weights,
                   uint32_t *out_res)
{
   const uint32_t atlas_pitch = stride + 4u;
   for (uint32_t q = 0; q < num_queries; q++) {
      const float *qv = queries + (size_t)q * GB_DESC_DIMS;
      const float *wv = weights + (size_t)(q & 15u) * GB_DESC_DIMS;
      float best_dist = 1e9f;
      uint32_t best_id = 0u;
      for (uint32_t c = 0; c < num_cands; c++) {
         uint32_t row = c / stride;
         uint32_t col = c % stride;
         const float *cv = cands_atlas + ((size_t)row * atlas_pitch + col) * GB_DESC_DIMS;
         float dist = 0.0f;
         for (uint32_t d = 0; d < GB_DESC_DIMS; d++) {
            float wt = (flags & 1u) ? wv[d] : 1.0f;
            float diff = (qv[d] - cv[d]) * wt;
            dist += diff * diff;
         }
         if (dist < best_dist) {
            best_dist = dist;
            best_id = c;
         }
      }
      out_res[q] = (best_id & 0xffffu) | (((uint32_t)best_dist & 0xffffu) << 16);
   }
   return gb_crc32(out_res, (size_t)num_queries * sizeof(uint32_t));
}

/* =========================================================================
 * 4. compute.chain CPU reference
 * ========================================================================= */

void
gb_chain_fill_init(uint32_t count, uint32_t *init_buf)
{
   for (uint32_t i = 0; i < count; i++)
      init_buf[i] = (i * 2654435761u) ^ 0x243f6a88u;
}

uint32_t
gb_chain_ref(uint32_t count, uint32_t passes, const uint32_t *init_buf, uint32_t *out_buf)
{
   uint32_t mask = count - 1u;
   uint32_t *a = malloc((size_t)count * sizeof(uint32_t));
   uint32_t *b = malloc((size_t)count * sizeof(uint32_t));
   const uint32_t *src = init_buf;
   uint32_t *dst = a;

   for (uint32_t p = 0; p < passes; p++) {
      uint32_t salt = 0x9e3779b9u * (p + 1u);
      for (uint32_t i = 0; i < count; i++) {
         uint32_t va = src[i];
         uint32_t vb = src[(i + 1u) & mask];
         uint32_t vc = src[(i + 7u) & mask];
         dst[i] = (va * 1664525u + vb + salt) ^ ((vc >> 13) | (va << 19));
      }
      src = dst;
      dst = (dst == a) ? b : a;
   }
   memcpy(out_buf, src, (size_t)count * sizeof(uint32_t));
   free(a);
   free(b);
   return gb_crc32(out_buf, (size_t)count * sizeof(uint32_t));
}

/* =========================================================================
 * 5. render.msaa_merge CPU reference
 * ========================================================================= */

static inline uint16_t
u8_div256_to_f16(uint32_t v)
{
   if (v == 0u)
      return 0u;
   uint32_t msb = 0u;
   for (uint32_t t = v >> 1; t != 0u; t >>= 1)
      msb++;
   uint32_t exp = 7u + msb;
   uint32_t mant = (v << (10u - msb)) & 0x3ffu;
   return (uint16_t)((exp << 10) | mant);
}

uint32_t
gb_msaa_merge_ref_rgba8(uint32_t w, uint32_t h, uint32_t passes, uint32_t *out_rgba8)
{
   for (uint32_t y = 0; y < h; y++) {
      uint32_t by = y >> 4;
      for (uint32_t x = 0; x < w; x++) {
         uint32_t bx = x >> 4;
         uint32_t sr = 0u, sg = 0u, sb = 0u, sa = 0u;
         for (uint32_t p = 0; p < passes; p++) {
            uint32_t hv = (bx * 73u + by * 151u + p * 37u) & 31u;
            sr += (hv + p * 3u) & 31u;
            sg += ((hv ^ (bx & 15u)) + p * 5u) & 31u;
            sb += ((hv ^ (by & 15u)) + p * 7u) & 31u;
            sa += 31u;
         }
         if (sr > 255u) sr = 255u;
         if (sg > 255u) sg = 255u;
         if (sb > 255u) sb = 255u;
         if (sa > 255u) sa = 255u;
         out_rgba8[(size_t)y * w + x] = sr | (sg << 8) | (sb << 16) | (sa << 24);
      }
   }
   return gb_crc32(out_rgba8, (size_t)w * h * sizeof(uint32_t));
}

uint32_t
gb_msaa_merge_ref_rgba16f(uint32_t w, uint32_t h, uint32_t passes, uint16_t *out_rgba16f)
{
   for (uint32_t y = 0; y < h; y++) {
      uint32_t by = y >> 4;
      for (uint32_t x = 0; x < w; x++) {
         uint32_t bx = x >> 4;
         uint32_t sr = 0u, sg = 0u, sb = 0u, sa = 0u;
         for (uint32_t p = 0; p < passes; p++) {
            uint32_t hv = (bx * 73u + by * 151u + p * 37u) & 31u;
            sr += (hv + p * 3u) & 31u;
            sg += ((hv ^ (bx & 15u)) + p * 5u) & 31u;
            sb += ((hv ^ (by & 15u)) + p * 7u) & 31u;
            sa += 31u;
         }
         uint16_t *d = out_rgba16f + ((size_t)y * w + x) * 4u;
         d[0] = u8_div256_to_f16(sr);
         d[1] = u8_div256_to_f16(sg);
         d[2] = u8_div256_to_f16(sb);
         d[3] = u8_div256_to_f16(sa);
      }
   }
   return gb_crc32(out_rgba16f, (size_t)w * h * 8u);
}

/* =========================================================================
 * 6. compute.deferred_light CPU reference
 *
 * Mirrors suite/shaders/deferred_light.comp operation for operation. Every
 * float value is an integer below 2^24, so each operation is exact and the
 * result matches the GPU bit for bit whatever the contraction or order.
 * ========================================================================= */

static uint32_t
dlight_hash(uint32_t v)
{
   v ^= v >> 16;
   v *= 0x7feb352du;
   v ^= v >> 15;
   v *= 0x846ca68bu;
   v ^= v >> 16;
   return v;
}

void
gb_deferred_light_fill(struct gb_dlight_input *in, uint32_t w, uint32_t h, uint32_t lights)
{
   memset(in, 0, sizeof(*in));
   in->w = w;
   in->h = h;
   in->max_lights = lights + 1u;
   in->shadow_dim = GB_DLIGHT_SHADOW_DIM;
   size_t px = (size_t)w * h;
   for (int l = 0; l < 4; l++) {
      in->g[l] = malloc(px * sizeof(uint32_t));
      for (size_t i = 0; i < px; i++)
         in->g[l][i] = dlight_hash((uint32_t)i * 4u + (uint32_t)l + 0x9e3779b9u);
   }
   in->depth = malloc(px * sizeof(uint32_t));
   for (size_t i = 0; i < px; i++) {
      uint32_t r = dlight_hash((uint32_t)i ^ 0x51ed270bu);
      /* About 1 pixel in 16 is sky. */
      in->depth[i] = (r & 15u) == 0u ? 0u : 1u + ((r >> 4) & 254u);
   }
   size_t slots = (size_t)GB_DLIGHT_TILE_LISTS * in->max_lights;
   in->lights = calloc(slots * 2u * 4u, sizeof(uint32_t));
   for (uint32_t t = 0; t < GB_DLIGHT_TILE_LISTS; t++) {
      /* Tile lists end 0..3 lights early, so the loop's early break varies
       * between workgroups. */
      uint32_t count = lights > 3u ? lights - (t & 3u) : lights;
      for (uint32_t i = 0; i < count; i++) {
         uint32_t *l = in->lights + ((size_t)t * in->max_lights + i) * 8u;
         uint32_t r0 = dlight_hash(t * 977u + i * 131u + 1u);
         uint32_t r1 = dlight_hash(r0 + 0x2545f491u);
         l[0] = r0 & 63u;
         l[1] = (r0 >> 6) & 63u;
         l[2] = (r0 >> 12) & 63u;
         l[3] = 256u + ((r0 >> 18) & 4095u) % 3841u; /* radius^2, 256..4096 */
         l[4] = r1 & 15u;
         l[5] = (r1 >> 4) & 15u;
         l[6] = (r1 >> 8) & 15u;
         l[7] = 1u + ((r1 >> 12) & 3u);              /* nonzero: not the terminator */
      }
      /* Slots from `count` on stay zero: l1.w == 0 is the terminator. */
   }
   size_t sp = (size_t)in->shadow_dim * in->shadow_dim;
   in->shadow = malloc(sp * sizeof(uint32_t));
   for (size_t i = 0; i < sp; i++)
      in->shadow[i] = dlight_hash((uint32_t)i + 0x68e31da4u) & 63u;
}

void
gb_deferred_light_free(struct gb_dlight_input *in)
{
   for (int l = 0; l < 4; l++)
      free(in->g[l]);
   free(in->depth);
   free(in->lights);
   free(in->shadow);
   memset(in, 0, sizeof(*in));
}

static void
dlight_bytes4(uint32_t v, float out[4])
{
   out[0] = (float)(v & 15u);
   out[1] = (float)((v >> 8) & 15u);
   out[2] = (float)((v >> 16) & 15u);
   out[3] = (float)((v >> 24) & 15u);
}

static float
dlight_shadow_tap(const struct gb_dlight_input *in, int32_t sx, int32_t sy, float ref)
{
   int32_t dim = (int32_t)in->shadow_dim;
   int32_t cx = clamp_i32(sx, 0, dim - 1), cy = clamp_i32(sy, 0, dim - 1);
   return (float)in->shadow[(size_t)cy * in->shadow_dim + (size_t)cx] >= ref ? 1.0f : 0.0f;
}

uint32_t
gb_deferred_light_ref(const struct gb_dlight_input *in, uint32_t mats, uint32_t *out)
{
   for (uint32_t y = 0; y < in->h; y++) {
      for (uint32_t x = 0; x < in->w; x++) {
         size_t idx = (size_t)y * in->w + x;
         uint32_t d = in->depth[idx];
         if (d == 0u) {
            out[idx * 2u + 0u] = 0u;
            out[idx * 2u + 1u] = 0u;
            continue;
         }
         float g[4][4];
         for (int l = 0; l < 4; l++)
            dlight_bytes4(in->g[l][idx], g[l]);
         const float *a = g[0], *b = g[1], *c = g[2], *e = g[3];
         float pos[3] = {(float)(x & 63u), (float)(y & 63u), (float)(d & 63u)};
         float n[3] = {a[0] - 7.0f, a[1] - 7.0f, a[2] - 7.0f};

         float mat[GB_DLIGHT_MAX_MATS][4], acc[GB_DLIGHT_MAX_MATS][4];
         for (uint32_t k = 0; k < mats; k++) {
            const float *src = (k & 3u) == 0u ? b : (k & 3u) == 1u ? c : (k & 3u) == 2u ? e : a;
            mat[k][0] = src[k & 3u];
            mat[k][1] = src[(k + 1u) & 3u];
            mat[k][2] = src[(k + 2u) & 3u];
            mat[k][3] = (float)k;
            acc[k][0] = acc[k][1] = acc[k][2] = acc[k][3] = 0.0f;
         }

         uint32_t tile = ((y / GB_DLIGHT_LOCAL) * 7u + (x / GB_DLIGHT_LOCAL)) % GB_DLIGHT_TILE_LISTS;
         size_t base = (size_t)tile * in->max_lights * 2u;
         int32_t sx = (int32_t)x >> 2, sy = (int32_t)y >> 2;
         float ref = (float)(d & 63u);

         for (uint32_t i = 0; i < in->max_lights; i++) {
            const uint32_t *l0 = in->lights + (base + 2u * i) * 4u;
            const uint32_t *l1 = l0 + 4u;
            if (l1[3] == 0u)
               break;
            float dv[3] = {(float)l0[0] - pos[0], (float)l0[1] - pos[1], (float)l0[2] - pos[2]};
            float dist2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
            float r2 = (float)l0[3];
            if (dist2 < r2) {
               float ndl = n[0] * dv[0] + n[1] * dv[1] + n[2] * dv[2];
               if (ndl < 0.0f)
                  ndl = 0.0f;
               /* floor() of a value >= 0 is its truncation. A -0 here (ndl = -0)
                * only ever adds zeros, so its sign cannot reach the output. */
               float term = (float)(uint32_t)((r2 - dist2) * ndl * (1.0f / 65536.0f));
               int32_t o = (int32_t)(l1[3] & 3u) - 1;
               float vis = dlight_shadow_tap(in, sx + o, sy, ref) + dlight_shadow_tap(in, sx - o, sy, ref) +
                           dlight_shadow_tap(in, sx, sy + o, ref) + dlight_shadow_tap(in, sx, sy - o, ref);
               float tv = term * vis;
               float col[4] = {(float)l1[0] * tv, (float)l1[1] * tv, (float)l1[2] * tv, tv};
               for (uint32_t k = 0; k < mats; k++)
                  for (int ch = 0; ch < 4; ch++)
                     acc[k][ch] += col[ch] * mat[k][ch];
            }
         }

         uint32_t sum[4] = {0, 0, 0, 0};
         for (uint32_t k = 0; k < mats; k++)
            for (int ch = 0; ch < 4; ch++)
               sum[ch] ^= (uint32_t)acc[k][ch] << (k & 7u);
         out[idx * 2u + 0u] = (sum[0] & 0xffffu) | (sum[1] << 16);
         out[idx * 2u + 1u] = (sum[2] & 0xffffu) | (sum[3] << 16);
      }
   }
   return gb_crc32(out, (size_t)in->w * in->h * 2u * sizeof(uint32_t));
}
