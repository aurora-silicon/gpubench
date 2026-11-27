// SPDX-License-Identifier: GPL-2.0-only
/* Shared helpers, workload math, and deterministic CPU reference checks for
 * the D3D11 (DXVK) and D3D12 (vkd3d-proton) gpubench executors. */
#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "gb_common.h"
#include "gb_params.h"
#include "gb_wave1_ref.h"
#include "gb_wave2_ref.h"
#ifdef __cplusplus
}
#endif

static inline bool
gb_d3d_parse_res(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) { *w = 1280u; *h = 720u; return true; }
   if (!strcmp(s, "1080p")) { *w = 1920u; *h = 1080u; return true; }
   if (!strcmp(s, "1440p")) { *w = 2560u; *h = 1440u; return true; }
   if (!strcmp(s, "2160p")) { *w = 3840u; *h = 2160u; return true; }
   return false;
}

static inline uint32_t
gb_d3d_draws_ref_fb(uint32_t draws, uint32_t *fb_rgba8)
{
   const uint32_t dim = 1024u;
   for (uint32_t i = 0; i < dim * dim; i++)
      fb_rgba8[i] = 0xff000000u;
   for (uint32_t d = 0; d < draws; d++) {
      uint32_t cell = d % 10000u;
      uint32_t cx = cell % 100u;
      uint32_t cy = cell / 100u;
      uint32_t r = (d * 37u) & 255u;
      uint32_t g = (d * 73u) & 255u;
      uint32_t b = 128u; /* 0.5 * 255 = 127.5 -> 128 in UNORM8 */
      uint32_t col = r | (g << 8u) | (b << 16u) | 0xff000000u;
      uint32_t x0 = cx * 10u, y0 = cy * 10u;
      for (uint32_t y = y0; y < y0 + 10u; y++)
         for (uint32_t x = x0; x < x0 + 10u; x++)
            fb_rgba8[y * dim + x] = col;
   }
   return gb_crc32(fb_rgba8, (size_t)dim * dim * sizeof(uint32_t));
}

static inline void
gb_d3d_check_draws_fb(const uint8_t *got_rgba8, uint32_t draws, struct gb_check *chk)
{
   const uint32_t dim = 1024u;
   uint32_t *ref = (uint32_t *)malloc((size_t)dim * dim * sizeof(uint32_t));
   if (!ref) gb_die("out of memory");
   uint32_t expected_crc = gb_d3d_draws_ref_fb(draws, ref);
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(got_rgba8, (size_t)dim * dim * 4u);
   chk->expected = expected_crc;
   chk->valid = true;
   const uint8_t *ref_b = (const uint8_t *)ref;
   /* Allow +-1 rounding on 0.5 -> 127/128 in UNORM8. */
   for (size_t i = 0; i < (size_t)dim * dim * 4u; i++) {
      int diff = (int)got_rgba8[i] - (int)ref_b[i];
      if (diff < 0) diff = -diff;
      if (diff > 1) {
         gb_log("render.draws mismatch at byte %zu: got %u want %u", i, got_rgba8[i], ref_b[i]);
         chk->valid = false;
         break;
      }
   }
   if (chk->valid && chk->crc != expected_crc)
      chk->expected = chk->crc;
   free(ref);
}

static inline void
gb_d3d_check_compute_light(const uint32_t *got, uint32_t k, struct gb_check *chk)
{
   uint32_t want[GB_LIGHT_LOCAL];
   for (uint32_t i = 0; i < GB_LIGHT_LOCAL; i++)
      want[i] = k + i;
   uint32_t ecrc = gb_crc32(want, sizeof(want));
   gb_check_words(chk, got, want, ecrc, GB_LIGHT_LOCAL);
}

static inline void
gb_d3d_check_compute_alu(const float *got, struct gb_check *chk)
{
   const size_t n = (size_t)GB_ALU_GROUPS * GB_ALU_LOCAL;
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(got, n * sizeof(float));
   chk->expected = chk->crc;
   chk->valid = true;
   for (size_t i = 0; i < n; i += GB_ALU_LOCAL - 1u) {
      float x[8];
      for (int j = 0; j < 8; j++) x[j] = (float)i * 1e-6f + (float)j;
      for (uint32_t step = 0; step < GB_ALU_ITERS * 4u; step++)
         for (int j = 0; j < 8; j++) x[j] = fmaf(x[j], GB_ALU_A, GB_ALU_B);
      float expected = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
      if (!isfinite(got[i]) || fabsf(got[i] - expected) > 0.002f) {
         gb_log("compute.heavy alu mismatch at %zu: got %.6f want %.6f", i, got[i], expected);
         chk->valid = false;
         chk->expected = 0u;
         return;
      }
   }
}

static inline void
gb_d3d_check_compute_bw(const float *got, struct gb_check *chk)
{
   const size_t n = (size_t)(GB_BW_BYTES / sizeof(float));
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(got, GB_BW_BYTES);
   chk->expected = chk->crc;
   chk->valid = true;
   for (size_t i = 0; i < n; i += 257u) {
      float want = (float)(i & 255u);
      if (got[i] != want) {
         gb_log("compute.heavy bw mismatch at %zu: got %.1f want %.1f", i, got[i], want);
         chk->valid = false;
         chk->expected = 0u;
         return;
      }
   }
}

/* Verify render.light / render.heavy RGBA8 framebuffer pixel + CRC. */
static inline void
gb_d3d_check_render_basic(const uint8_t *rgba8, uint32_t w, uint32_t h,
                          const char *kind, uint32_t k, struct gb_check *chk)
{
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(rgba8, (size_t)w * h * 4u);
   chk->expected = chk->crc;
   chk->valid = true;

   bool is_tex = !strcmp(kind, "texture");
   bool is_fill = !strcmp(kind, "fill");
   bool is_clear = !strcmp(kind, "clear");
   uint32_t sx = is_tex ? 13u : 0u;
   uint32_t sy = is_tex ? 29u : 0u;
   if (sx >= w) sx = 0;
   if (sy >= h) sy = 0;
   const uint8_t *px = rgba8 + ((size_t)sy * w + sx) * 4u;

   const float clear[4] = {0.1f, 0.2f, 0.3f, 1.0f};
   const float color[4] = {(float)(k & 255u) / 255.0f, 0.5f, 0.25f, 0.5f};
   for (int i = 0; i < 4; i++) {
      float expected = is_clear ? clear[i] : color[i];
      if (is_tex && i < 3) {
         float tc = (float)(i == 0 ? sx : (i == 1 ? sy : (sx ^ sy))) / 255.0f;
         expected = tc * color[i];
      }
      if (is_fill || is_tex) {
         /* 16 layers of SRC_ALPHA / INV_SRC_ALPHA with alpha=0.5 */
         expected = (i == 3) ? 1.0f : (expected + (clear[i] - expected) / 65536.0f);
      }
      int want = (int)lroundf(expected * 255.0f);
      int diff = (int)px[i] - want;
      if (diff < 0) diff = -diff;
      if (diff > 3) {
         gb_log("render (%s) mismatch at (%u,%u) ch %d: got %u want %d", kind, sx, sy, i, px[i], want);
         chk->valid = false;
         chk->expected = 0u;
         return;
      }
   }
}

static inline void
gb_d3d_strip_verify_flag(int *argc, char **argv, bool *do_verify)
{
   *do_verify = false;
   for (int i = 1; i < *argc;) {
      if (!strcmp(argv[i], "--verify")) {
         *do_verify = true;
         memmove(argv + i, argv + i + 1, (size_t)(*argc - i) * sizeof(*argv));
         (*argc)--;
      } else {
         i++;
      }
   }
}

static inline double
gb_d3d_mipchain_bytes(void)
{
   double bytes = 0.0;
   uint32_t mw = GB_XFER_MIP_W, mh = GB_XFER_MIP_H;
   for (uint32_t i = 1; i < GB_XFER_MIP_LEVELS; i++) {
      uint32_t nw = mw > 1u ? mw / 2u : 1u;
      uint32_t nh = mh > 1u ? mh / 2u : 1u;
      bytes += ((double)mw * mh + (double)nw * nh) * 4.0;
      mw = nw;
      mh = nh;
   }
   return bytes;
}
