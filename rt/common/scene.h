// SPDX-License-Identifier: GPL-2.0-only
/* Procedural "bay" scene shared by both executors: a water surface split into 16 patch BLASes
 * (some animated), and object instances (spheres, towers, tori, rocks, alpha-tested sails) drawn
 * from a set of unique meshes. Everything is deterministic from the parameters and the seed, so
 * Vulkan and Metal build byte-identical buffers. GPU struct layouts below are mirrored in
 * shaders/common.glsl and shaders/rt.metal. */
#pragma once
#include <stdint.h>
#include "params.h"

#define RT_WATER_PATCHES 16
#define RT_KINDS 5
enum { RT_KIND_SPHERE, RT_KIND_TOWER, RT_KIND_TORUS, RT_KIND_ROCK, RT_KIND_SAIL };
enum { RT_FLAG_ALPHA = 1, RT_FLAG_WATER = 2, RT_FLAG_MOVING = 4, RT_FLAG_TWOSIDED = 8 };

typedef struct { float px, py, pz, nx, ny, nz, u, v; } rt_vertex;            /* 32 bytes */
typedef struct {
   uint32_t first_index, base_vertex, flags, mesh;
   float albedo[4];
   float params[4];        /* reflectivity, roughness, pattern, 0 */
} rt_instinfo;                                                               /* 48 bytes */
typedef struct { float r[12]; } rt_xform;   /* 3x4 row-major object-to-world, 48 bytes */

typedef struct {
   float view_proj[16];
   float inv_view_proj[16];
   float cam_pos[4];       /* xyz, w = time */
   float sun_dir[4];       /* xyz (towards the sun), w = intensity */
   float params[4];        /* tmax, origin offset scale, ambient, 0 */
   uint32_t ip[4];         /* width, height, frame, flags (RT_UBO_*) */
   uint32_t ip2[4];        /* rpp, bounces, instance count, 0 */
} rt_frame_ubo;                                                              /* 192 bytes */
enum { RT_UBO_SHADOWS = 1, RT_UBO_ALPHA = 2, RT_UBO_COUNT = 4 };

typedef struct {
   uint32_t vbase;         /* first vertex of water slot 0 */
   uint32_t grid;          /* quads per patch side */
   uint32_t first_slot;    /* first water slot to process */
   uint32_t nslots;        /* slots to process */
   float size;             /* water side length */
   float time;
   float pad[2];
} rt_deform_pc;                                                              /* 32 bytes */

typedef struct {
   uint32_t first_index, index_count, base_vertex, vertex_count;
   uint32_t kind, alpha, water, dynamic, twosided;
} rt_mesh;

typedef struct {
   uint32_t mesh, flags;
   float pos[3], scale, yaw, spin, bob, phase;
} rt_instance;

/* twosided: drawn with the no-cull pipeline (the sails), which also alpha-tests when --alpha 1 */
typedef struct { uint32_t mesh, first_instance, instance_count, twosided; } rt_draw;

typedef struct rt_scene {
   rt_vertex *verts;
   uint32_t nverts;
   uint32_t *indices;
   uint32_t nindices;
   rt_mesh *meshes;         /* water slots 0..15 first, then object meshes */
   uint32_t nmeshes;
   rt_instance *inst;       /* water slots 0..15 first, then objects grouped by mesh */
   rt_instinfo *info;
   uint32_t ninst;
   rt_draw *draws;
   uint32_t ndraws;
   uint32_t water_grid, water_vbase, dyn_patches, nmoving;
   float water_size;
   uint64_t unique_tris, instanced_tris;
} rt_scene;

void rt_scene_build(rt_scene *s, const rt_params *p);
void rt_scene_free(rt_scene *s);
/* Object-to-world transform of instance i at time t (3x4 row-major). */
void rt_scene_xform(const rt_scene *s, uint32_t i, float t, rt_xform *out);
/* Fills the per-frame constants. flip_y: 1 for Vulkan, 0 for Metal. */
void rt_scene_frame_ubo(const rt_scene *s, const rt_params *p, uint32_t frame, int flip_y, int count_rays,
                        rt_frame_ubo *u);
/* Pinhole camera of frame: eye, forward, right * tan(fov/2) * aspect, up * tan(fov/2). */
void rt_scene_camera(const rt_scene *s, uint32_t frame, float aspect, float eye[3], float fwd[3], float right[3],
                     float up[3]);
/* One standalone mesh of the given kind (used by the AS build micro-benchmark); caller frees. */
void rt_gen_mesh(int kind, uint32_t tris, rt_vertex **v, uint32_t *nv, uint32_t **idx, uint32_t *ni);
/* Fixed time step: frame k is rendered at time k / 60 s regardless of wall-clock time. */
static inline float rt_frame_time(uint32_t frame) { return (float)frame / 60.0f; }
