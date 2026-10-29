// SPDX-License-Identifier: GPL-2.0-only
/* Parameters and run controls, shared by the Vulkan and Metal executors.
 *
 * Executor interface (common to suite/ and rt/, see suite/README.md):
 *   --info | --list | --test NAME [--depths 1,2] [--mode queue] [--param k=v]... [--warmup-ms N]
 *   [--runs N] [--run-ms N] [--timestamps on|off] [--device N]
 * Convenience spellings: --preset NAME == --param preset=NAME, and --<key> V == --param <key>=V for
 * every test parameter. Without --test, rt.scene is run. */
#pragma once
#include <stdint.h>

enum rt_as_mode { RT_AS_STATIC = 0, RT_AS_REFIT = 1, RT_AS_REBUILD = 2 };
enum rt_trace_mode { RT_TRACE_QUERY = 0, RT_TRACE_PIPELINE = 1 };
enum rt_test { RT_TEST_SCENE = 0, RT_TEST_ASBUILD = 1, RT_TEST_RAYS = 2, RT_TEST_TRAV = 3, RT_TEST_COUNT };

typedef struct rt_params {
   char preset[64];
   uint32_t width, height;
   /* geometry */
   uint32_t mesh_tris;     /* target triangles per unique object mesh */
   uint32_t meshes;        /* unique object meshes (one BLAS each) */
   uint32_t instances;     /* object instances in the TLAS (water patches not counted) */
   uint32_t water_tris;    /* triangles in the water surface (16 patches, one BLAS each) */
   /* dynamics */
   int blas_mode;          /* animated water patches: static, refit or rebuild every frame */
   uint32_t dyn_patches;   /* animated water patches (0..16) when blas_mode != static */
   int tlas_mode;          /* static, refit or rebuild every frame */
   float moving;           /* fraction of object instances that move when tlas_mode != static */
   /* rays */
   float reflective;       /* fraction of surfaces (instances and water patches) that reflect */
   uint32_t rpp;           /* reflection rays per reflective pixel */
   uint32_t bounces;       /* reflection bounces, 1..8 */
   int shadows;            /* shadow rays at the primary surface and at every reflection hit */
   float tmax;             /* ray length for reflection and shadow rays */
   int alpha;              /* alpha-tested sails (non-opaque geometry, any-hit path) */
   int trace_mode;         /* enum rt_trace_mode */
   int camera_orbit;       /* 0: camera fixed at its frame-0 pose (default), 1: orbits with time */
   uint32_t seed;
   /* micro-benchmarks */
   char kind[32];          /* asbuild: blas | refit | tlas; rays: coherent | incoherent;
                              trav: miss | single | comb | grid | gap (common/trav.h) */
   char build[16];         /* asbuild: fasttrace | fastbuild */
   uint32_t tris;          /* asbuild: triangles (blas, refit) or instances (tlas); trav: triangles
                              (comb, grid) or planes (gap) */
   int compact;            /* asbuild: build with compaction allowed and report the compacted size */
   uint32_t rays;          /* rays, trav: rays per submission */
   float eps;              /* trav gap: gap between the rays and every triangle's box */
} rt_params;

typedef struct rt_run {
   int test;               /* enum rt_test */
   int info, list;
   uint32_t depths[16];
   uint32_t ndepths;       /* 0 = the test's default */
   char mode[16];
   uint32_t warmup_ms, runs, run_ms;   /* UINT32_MAX until rt_run_defaults() */
   int timestamps;
   int stamps_frame;       /* --stamps frame: only frame start/end stamps (no per-pass splits) */
   int device;
   uint32_t frames;        /* nonzero: fixed submissions per window instead of run_ms (short checks) */
   int shader_stats;
   int window;             /* --window: interactive Wayland/NSWindow view, no record */
   uint32_t window_s;      /* --window-seconds: close the window after N s (0 = when closed) */
   int verbose;
   char png[512];
   char preset_file[512];
   char tag[128];          /* free-form label copied into every result line */
   char dump_archive[512]; /* Metal: serialize an MTLBinaryArchive of every pipeline here and exit */
} rt_run;

const char *rt_test_name(int t);
const char *rt_as_mode_name(int m);
void rt_params_defaults(rt_params *p, int test);
int rt_params_apply_preset(rt_params *p, const char *name, const char *file);
/* Sets one test parameter. Returns 0 on success. */
int rt_params_set(rt_params *p, const char *key, const char *val);
/* Parses argv. Returns 0 to run, 1 when the caller should exit 0, -1 on a usage error. */
int rt_parse(rt_params *p, rt_run *r, int argc, char **argv, const char *exe);
/* Applies the test's default depths and windows to whatever was not given. */
void rt_run_defaults(rt_run *r);
/* The record's "params": only what changes the work, for the given test. Appends to a JSON writer. */
struct rt_json;
void rt_params_json(struct rt_json *j, const rt_params *p, int test);

extern const char rt_presets_ini[];
