// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include "gb_common.h"

struct gb_test {
   const char *name;
   int (*run)(const struct gb_opts *o);
};
extern const struct gb_test gb_tests[];
/* Builds every suite pipeline and prints VK_KHR_pipeline_executable_properties
 * statistics as JSON lines (used for suite/FAIRNESS.md). */
int gb_shader_stats(const struct gb_opts *o);

/* Wave-1 90%-coverage Vulkan benchmarks (tests_wave1.c). */
int gb_run_compute_uber(const struct gb_opts *o);
int gb_run_compute_conv2d(const struct gb_opts *o);
int gb_run_compute_stereo(const struct gb_opts *o);
int gb_run_compute_edge(const struct gb_opts *o);
int gb_run_render_deferred(const struct gb_opts *o);
int gb_run_concurrent_sram(const struct gb_opts *o);
int gb_run_render_mdi(const struct gb_opts *o);
int gb_run_compile_pso(const struct gb_opts *o);
int gb_run_compress_bw(const struct gb_opts *o);
int gb_run_overlap_passes(const struct gb_opts *o);

/* Wave-2 90%-coverage heavy-game / AAA Vulkan benchmarks (tests_wave2.c). */
int gb_run_render_shadow(const struct gb_opts *o);
int gb_run_compute_cull_mdi(const struct gb_opts *o);
int gb_run_render_bindless_ubo(const struct gb_opts *o);
int gb_run_compute_fp16(const struct gb_opts *o);
int gb_run_transfer_ops(const struct gb_opts *o);
int gb_run_render_postfx(const struct gb_opts *o);

/* Wave-3 AGX3 / M4-M5 compute & MSAA Vulkan benchmarks (tests_wave3.c). */
int gb_run_compute_blur_fp32(const struct gb_opts *o);
int gb_run_compute_corner_ring(const struct gb_opts *o);
int gb_run_compute_desc_search(const struct gb_opts *o);
int gb_run_compute_chain(const struct gb_opts *o);
int gb_run_render_msaa_merge(const struct gb_opts *o);
