// SPDX-License-Identifier: GPL-2.0-only
/* rt.trav: traversal micro-benchmarks on synthetic geometry, shared by both executors. One opaque
 * triangle BLAS (non-indexed float3 positions) under one identity instance; parallel rays along +z
 * from z = -1 on a grid. Kinds isolate one traversal step each:
 *   miss    rays miss the TLAS root box (one box test per ray)
 *   single  one large triangle that every ray hits (TLAS root, instance, BLAS root, one triangle)
 *   comb    `tris` triangles stacked along the rays; every ray is inside every box and misses every
 *           triangle, so it visits every node and tests every triangle
 *   grid    `tris` small triangles tiling a square; every ray hits exactly one (path length cost)
 *   gap     `tris` planes of two triangles whose boxes leave a gap of `eps` on each side of the
 *           rays: with exact child boxes no ray reaches a triangle, so time over the gap sweep
 *           measures how loosely child boxes are encoded
 * Mirrored by shaders/trav.comp and the trav kernel in shaders/rt.metal. */
#pragma once
#include <stdint.h>
#include "params.h"

typedef struct {
   float x0, y0, dx, dy;   /* ray i starts at (x0 + (i % width + 0.5) dx, y0 + (i / width + 0.5) dy, -1) */
   uint32_t width, nrays;
   float tmax;
   uint32_t pad;
} rt_trav_rays;            /* 32 bytes */

/* Allocates *pos (3 float3 vertices per triangle) and fills the ray grid. Returns 0, or -1 for an
 * unknown kind. */
int rt_trav_geometry(const rt_params *p, float **pos, uint32_t *ntris, rt_trav_rays *rays);
