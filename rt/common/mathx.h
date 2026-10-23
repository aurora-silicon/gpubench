// SPDX-License-Identifier: GPL-2.0-only
/* Small vector/matrix helpers shared by the Vulkan and Metal executors.
 * Matrices are column-major float[16] (m[col * 4 + row]), matching GLSL and MSL. */
#pragma once
#include <math.h>
#include <string.h>

typedef struct { float x, y, z; } v3;

static inline v3 v3m(float x, float y, float z) { v3 r = {x, y, z}; return r; }
static inline v3 v3add(v3 a, v3 b) { return v3m(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline v3 v3sub(v3 a, v3 b) { return v3m(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline v3 v3scale(v3 a, float s) { return v3m(a.x * s, a.y * s, a.z * s); }
static inline float v3dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline v3 v3cross(v3 a, v3 b)
{
   return v3m(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float v3len(v3 a) { return sqrtf(v3dot(a, a)); }
static inline v3 v3norm(v3 a)
{
   float l = v3len(a);
   return l > 0 ? v3scale(a, 1.0f / l) : v3m(0, 1, 0);
}

static inline void m4identity(float *m)
{
   memset(m, 0, 16 * sizeof(float));
   m[0] = m[5] = m[10] = m[15] = 1;
}

static inline void m4mul(float *r, const float *a, const float *b)
{
   float t[16];
   for (int c = 0; c < 4; c++)
      for (int rr = 0; rr < 4; rr++) {
         float s = 0;
         for (int k = 0; k < 4; k++)
            s += a[k * 4 + rr] * b[c * 4 + k];
         t[c * 4 + rr] = s;
      }
   memcpy(r, t, sizeof(t));
}

/* Right-handed look-at (camera looks down -Z in view space). */
static inline void m4lookat(float *m, v3 eye, v3 at, v3 up)
{
   v3 f = v3norm(v3sub(at, eye));
   v3 s = v3norm(v3cross(f, up));
   v3 u = v3cross(s, f);
   m4identity(m);
   m[0] = s.x; m[4] = s.y; m[8] = s.z;
   m[1] = u.x; m[5] = u.y; m[9] = u.z;
   m[2] = -f.x; m[6] = -f.y; m[10] = -f.z;
   m[12] = -v3dot(s, eye);
   m[13] = -v3dot(u, eye);
   m[14] = v3dot(f, eye);
}

/* Reversed-Z perspective with infinite far plane: depth 1 at the near plane, 0 at infinity.
 * flip_y = 1 for Vulkan (NDC y points down), 0 for Metal (NDC y points up). */
static inline void m4persp_revz(float *m, float fovy, float aspect, float znear, int flip_y)
{
   float f = 1.0f / tanf(fovy * 0.5f);
   memset(m, 0, 16 * sizeof(float));
   m[0] = f / aspect;
   m[5] = flip_y ? -f : f;
   m[11] = -1.0f;
   m[14] = znear;
}

static inline int m4invert(float *inv, const float *m)
{
   float t[16];
   t[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
   t[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
   t[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
   t[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
   t[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
   t[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
   t[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
   t[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
   t[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
   t[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
   t[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
   t[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
   t[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
   t[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
   t[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
   t[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
   float det = m[0] * t[0] + m[1] * t[4] + m[2] * t[8] + m[3] * t[12];
   if (det == 0)
      return 0;
   det = 1.0f / det;
   for (int i = 0; i < 16; i++)
      inv[i] = t[i] * det;
   return 1;
}
