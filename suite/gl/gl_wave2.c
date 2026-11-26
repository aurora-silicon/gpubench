// SPDX-License-Identifier: GPL-2.0-only
/* OpenGL 90%-coverage Wave-2 parity benchmarks for gpubench-gl:
 *   render.shadow (depth_only | alpha_test, 4-cascade 2D_ARRAY + 3x3 PCF)
 *   transfer.ops  (clear_4m | clear_16m | copy_b2i | mipchain)
 *   render.postfx (4x MSAA RGBA16F HDR -> resolve -> 5-level bloom -> ACES) */
#include "gl_wave2.h"
#include "gb_params.h"
#include "gb_wave2_ref.h"

#include <epoxy/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
gl2_check(const char *where)
{
   GLenum e = glGetError();
   if (e != GL_NO_ERROR)
      gb_die("%s: GL error 0x%x", where, e);
}

static bool
gl2_parse_res(const char *s, unsigned *w, unsigned *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static GLuint
gl2_compile(const char *vs_src, const char *fs_src)
{
   GLuint p = glCreateProgram();
   GLuint vs = glCreateShader(GL_VERTEX_SHADER);
   GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(vs, 1, &vs_src, NULL);
   glShaderSource(fs, 1, &fs_src, NULL);
   glCompileShader(vs);
   glCompileShader(fs);
   GLuint sh[2] = {vs, fs};
   for (int i = 0; i < 2; i++) {
      GLint ok = 0;
      glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[2048];
         glGetShaderInfoLog(sh[i], sizeof(log), NULL, log);
         gb_die("GL wave2 shader compile failed: %s", log);
      }
      glAttachShader(p, sh[i]);
   }
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      gb_die("GL wave2 program link failed: %s", log);
   }
   glDeleteShader(vs);
   glDeleteShader(fs);
   return p;
}

typedef void (*gl2_rec_fn)(void *user, uint64_t k);

static void
gl2_run_stream(int depth, int warm_ms, int runs, int run_ms, gl2_rec_fn rec, void *user, struct gb_pass *p)
{
   struct { GLsync fence; uint64_t submit; } slots[64] = {0};
   memset(p, 0, sizeof(*p));
   p->nwin = runs;
   p->win_s = run_ms / 1e3;
   const uint64_t start = gb_now_ns() + (uint64_t)warm_ms * 1000000ull;
   const uint64_t win = (uint64_t)run_ms * 1000000ull;
   const uint64_t end = start + (uint64_t)runs * win;
   uint64_t sent = 0, done = 0;

   while (true) {
      uint64_t now = gb_now_ns();
      bool full = sent - done >= (uint64_t)depth;
      if (!full && now >= end)
         break;
      if (full) {
         GLenum st = glClientWaitSync(slots[done % depth].fence, 0, 10000000000ull);
         if (st != GL_ALREADY_SIGNALED && st != GL_CONDITION_SATISFIED) {
            p->failed = true;
            break;
         }
         uint64_t t = gb_now_ns();
         if (t >= start && t < end) {
            p->win_count[(t - start) / win]++;
            gb_samples_push(&p->latency, (t - slots[done % depth].submit) / 1e3);
         }
         glDeleteSync(slots[done % depth].fence);
         slots[done % depth].fence = NULL;
         done++;
         continue;
      }
      uint64_t r0 = gb_now_ns();
      rec(user, sent + 1);
      uint64_t r1 = gb_now_ns();
      slots[sent % depth].fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      glFlush();
      uint64_t r2 = gb_now_ns();
      slots[sent % depth].submit = r0;
      if (r0 >= start && r0 < end) {
         gb_samples_push(&p->record_cpu, (r1 - r0) / 1e3);
         gb_samples_push(&p->submit_cpu, (r2 - r1) / 1e3);
      }
      sent++;
   }
   while (done < sent) {
      if (!p->failed) {
         GLenum st = glClientWaitSync(slots[done % depth].fence, 0, 10000000000ull);
         if (st != GL_ALREADY_SIGNALED && st != GL_CONDITION_SATISFIED)
            p->failed = true;
         else {
            uint64_t t = gb_now_ns();
            if (t >= start && t < end)
               p->win_count[(t - start) / win]++;
         }
      }
      glDeleteSync(slots[done % depth].fence);
      done++;
   }
   for (int i = 0; i < runs; i++)
      p->submits += p->win_count[i];
}

static void
gl2_emit_result(const struct gl_ctx_info *ci, const struct gb_opts *o, int depth,
                const struct gb_pass *u, const struct gb_check *chk,
                const char *metric, const char *unit, double work_e6, const char *primary)
{
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, depth, NULL);
   gb_json_obj_begin(&r.j, "executor");
   ci->emit_identity(&r.j, ci->ctx);
   gb_json_obj_end(&r.j);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (u->failed || !u->submits) {
      gb_result_finish(&r, "fail", NULL, "GL wait/error or no measured completions");
      return;
   }
   gb_result_stream_metrics(&r, u, NULL);
   if (metric) {
      struct gb_samples us = {0};
      gb_pass_rate(u, &us);
      gb_derive_rate(&r, metric, unit, &us, work_e6);
      gb_samples_free(&us);
   }
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "submits", u->submits);
   gb_json_obj_end(&r.j);
   if (chk && chk->done)
      gb_result_check(&r, chk);
   gb_result_finish(&r, (chk && chk->done && !chk->valid) ? "fail" : "ok",
                    primary ? primary : "us_per_submit",
                    (chk && chk->done && !chk->valid) ? "output check failed" : NULL);
}

static const char *GL2_FS_VS =
   "#version 330 core\n"
   "void main() {\n"
   "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
   "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
   "}\n";

/* ======================= 1. render.shadow ======================= */

static const char *GL2_SHAD_VS =
   "#version 330 core\n"
   "uniform uint u_cascade;\n"
   "out vec2 v_uv;\n"
   "void main() {\n"
   "  uint i = uint(gl_InstanceID);\n"
   "  uint cell = i & 1023u; uint gx = cell & 31u; uint gy = cell >> 5;\n"
   "  const vec2 corners[6] = vec2[6](\n"
   "    vec2(0.0,0.0), vec2(1.0,0.0), vec2(1.0,1.0),\n"
   "    vec2(0.0,0.0), vec2(1.0,1.0), vec2(0.0,1.0));\n"
   "  vec2 corner = corners[gl_VertexID % 6];\n"
   "  vec2 uv = (vec2(float(gx), float(gy)) + corner) * (1.0 / 32.0);\n"
   "  bool caster = (((gx + gy + u_cascade) & 1u) == 0u);\n"
   "  float z = caster ? ((i < 1024u) ? (0.30 + 0.05 * float(u_cascade))\n"
   "                                  : (0.60 + 0.05 * float(u_cascade))) : 0.85;\n"
   "  gl_Position = vec4(uv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);\n"
   "  v_uv = corner;\n"
   "}\n";

static const char *GL2_SHAD_EMPTY_FS =
   "#version 330 core\n"
   "void main() {}\n";

static const char *GL2_SHAD_ALPHA_FS =
   "#version 330 core\n"
   "in vec2 v_uv;\n"
   "void main() {\n"
   "  float m = sin(v_uv.x * 18.8495559) * sin(v_uv.y * 18.8495559);\n"
   "  if (m < 0.0) discard;\n"
   "}\n";

static const char *GL2_SHAD_PCF_FS =
   "#version 330 core\n"
   "uniform sampler2DArrayShadow smap;\n"
   "uniform float u_inv_res;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  vec2 uv = gl_FragCoord.xy * (1.0 / 1024.0);\n"
   "  float qx = uv.x >= 0.5 ? 1.0 : 0.0; float qy = uv.y >= 0.5 ? 1.0 : 0.0;\n"
   "  float layer = qx + 2.0 * qy; vec2 luv = fract(uv * 2.0);\n"
   "  float sum = 0.0;\n"
   "  for (int dy = -1; dy <= 1; dy++)\n"
   "    for (int dx = -1; dx <= 1; dx++)\n"
   "      sum += texture(smap, vec4(clamp(luv + vec2(float(dx), float(dy)) * u_inv_res, 0.0, 1.0), layer, 0.70));\n"
   "  float sh = sum * (1.0 / 9.0);\n"
   "  o = vec4(0.2 + 0.7 * sh, 0.25 + 0.5 * sh, 0.35 + 0.4 * sh, 1.0);\n"
   "}\n";

struct gl2_shadow {
   bool alpha;
   unsigned res, inst;
   GLuint vao, depth_prog, pcf_prog, depth_tex, depth_fbos[GB_SHADOW_CASCADES], recv_tex, recv_fbo;
   GLint loc_cascade, loc_inv_res;
};

static void
gl2_shadow_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl2_shadow *st = u;
   glBindVertexArray(st->vao);
   glEnable(GL_DEPTH_TEST);
   glDepthMask(GL_TRUE);
   glDepthFunc(GL_LESS);
   glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
   glViewport(0, 0, (GLsizei)st->res, (GLsizei)st->res);
   glUseProgram(st->depth_prog);
   for (unsigned c = 0; c < GB_SHADOW_CASCADES; c++) {
      glBindFramebuffer(GL_FRAMEBUFFER, st->depth_fbos[c]);
      glClear(GL_DEPTH_BUFFER_BIT);
      glUniform1ui(st->loc_cascade, c);
      glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)st->inst);
   }
   glDisable(GL_DEPTH_TEST);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glBindFramebuffer(GL_FRAMEBUFFER, st->recv_fbo);
   glViewport(0, 0, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM);
   glClearColor(0, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(st->pcf_prog);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D_ARRAY, st->depth_tex);
   glUniform1f(st->loc_inv_res, 1.0f / (float)st->res);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static int
gl2_run_shadow(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"mode", "fmt", "res", "inst", NULL};
   if (!gb_check_params(o, known))
      gb_die("unknown parameter for render.shadow");
   if (ci->es) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "render.shadow requires desktop GL");
      return 0;
   }
   const char *mode = gb_param(o, "mode", "depth_only");
   bool alpha = !strcmp(mode, "alpha_test");
   if (!alpha && strcmp(mode, "depth_only"))
      gb_die("mode must be depth_only or alpha_test");
   const char *fmt_s = gb_param(o, "fmt", "d32f");
   GLenum dfmt;
   if (!strcmp(fmt_s, "d32f")) dfmt = GL_DEPTH_COMPONENT32F;
   else if (!strcmp(fmt_s, "d16")) dfmt = GL_DEPTH_COMPONENT16;
   else gb_die("fmt must be d32f or d16");
   long res = gb_param_int(o, "res", 2048);
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   if (res < 256 || res > 4096 || inst < 1024 || inst > 16384)
      gb_die("res 256..4096, inst 1024..16384");

   struct gl2_shadow st = {.alpha = alpha, .res = (unsigned)res, .inst = (unsigned)inst};
   glGenVertexArrays(1, &st.vao);
   st.depth_prog = gl2_compile(GL2_SHAD_VS, alpha ? GL2_SHAD_ALPHA_FS : GL2_SHAD_EMPTY_FS);
   st.pcf_prog = gl2_compile(GL2_FS_VS, GL2_SHAD_PCF_FS);
   st.loc_cascade = glGetUniformLocation(st.depth_prog, "u_cascade");
   st.loc_inv_res = glGetUniformLocation(st.pcf_prog, "u_inv_res");
   glUseProgram(st.pcf_prog);
   glUniform1i(glGetUniformLocation(st.pcf_prog, "smap"), 0);

   glGenTextures(1, &st.depth_tex);
   glBindTexture(GL_TEXTURE_2D_ARRAY, st.depth_tex);
   glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, (GLint)dfmt, (GLsizei)st.res, (GLsizei)st.res,
                GB_SHADOW_CASCADES, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
   glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LESS);

   glGenFramebuffers(GB_SHADOW_CASCADES, st.depth_fbos);
   for (unsigned c = 0; c < GB_SHADOW_CASCADES; c++) {
      glBindFramebuffer(GL_FRAMEBUFFER, st.depth_fbos[c]);
      glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, st.depth_tex, 0, (GLint)c);
      glDrawBuffer(GL_NONE);
      glReadBuffer(GL_NONE);
   }

   glGenTextures(1, &st.recv_tex);
   glBindTexture(GL_TEXTURE_2D, st.recv_tex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   glGenFramebuffers(1, &st.recv_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, st.recv_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, st.recv_tex, 0);
   gl2_check("shadow setup");

   struct gb_check chk = {0};
   if (do_verify) {
      gl2_shadow_rec(&st, 0);
      glFinish();
      uint8_t *got = malloc((size_t)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM * 4u);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, st.recv_fbo);
      glReadPixels(0, 0, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, GL_RGBA, GL_UNSIGNED_BYTE, got);
      gb_shadow_check_fb(got, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, alpha, &chk);
      free(got);
   }

   static const int defd[] = {1, 2};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   double tris = (double)GB_SHADOW_CASCADES * (double)st.inst * 2.0;
   for (int i = 0; i < nd; i++) {
      struct gb_pass u;
      gl2_run_stream(depths[i], o->warmup_ms, o->runs, o->run_ms, gl2_shadow_rec, &st, &u);
      gl2_emit_result(ci, o, depths[i], &u, &chk, "mtri_s", "Mtri/s", tris, "mtri_s");
      gb_pass_free(&u);
   }
   glDeleteFramebuffers(1, &st.recv_fbo);
   glDeleteTextures(1, &st.recv_tex);
   glDeleteFramebuffers(GB_SHADOW_CASCADES, st.depth_fbos);
   glDeleteTextures(1, &st.depth_tex);
   glDeleteProgram(st.depth_prog);
   glDeleteProgram(st.pcf_prog);
   glDeleteVertexArrays(1, &st.vao);
   return (chk.done && !chk.valid) ? 1 : 0;
}

/* ======================= 2. transfer.ops ======================= */

struct gl2_xfer {
   int kind; /* 0=clear_4m, 1=clear_16m, 2=copy_b2i, 3=mipchain */
   unsigned w, h;
   GLuint tex, fbo, pbo;
   uint32_t expected_crc;
};

static void
gl2_xfer_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl2_xfer *st = u;
   if (st->kind <= 1) {
      glBindFramebuffer(GL_FRAMEBUFFER, st->fbo);
      glViewport(0, 0, (GLsizei)st->w, (GLsizei)st->h);
      glClearColor(192.0f / 255.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
   } else if (st->kind == 2) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, st->pbo);
      glBindTexture(GL_TEXTURE_2D, st->tex);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)st->w, (GLsizei)st->h, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
   } else {
      glBindTexture(GL_TEXTURE_2D, st->tex);
      glGenerateMipmap(GL_TEXTURE_2D);
   }
}

static int
gl2_run_xfer(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known))
      gb_die("unknown parameter for transfer.ops");
   if (ci->es) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "transfer.ops requires desktop GL");
      return 0;
   }
   const char *k = gb_param(o, "kind", "clear_4m");
   struct gl2_xfer st = {0};
   if (!strcmp(k, "clear_4m")) { st.kind = 0; st.w = GB_XFER_4M_DIM; st.h = GB_XFER_4M_DIM; }
   else if (!strcmp(k, "clear_16m")) { st.kind = 1; st.w = GB_XFER_16M_DIM; st.h = GB_XFER_16M_DIM; }
   else if (!strcmp(k, "copy_b2i")) { st.kind = 2; st.w = GB_XFER_16M_DIM; st.h = GB_XFER_16M_DIM; }
   else if (!strcmp(k, "mipchain")) { st.kind = 3; st.w = GB_XFER_MIP_W; st.h = GB_XFER_MIP_H; }
   else gb_die("kind must be clear_4m, clear_16m, copy_b2i or mipchain");

   glGenTextures(1, &st.tex);
   glBindTexture(GL_TEXTURE_2D, st.tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, st.kind == 3 ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   size_t bytes = (size_t)st.w * st.h * 4u;
   if (st.kind <= 1) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)st.w, (GLsizei)st.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glGenFramebuffers(1, &st.fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, st.fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, st.tex, 0);
      st.expected_crc = gb_transfer_clear_ref_crc(st.w, st.h);
   } else if (st.kind == 2) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)st.w, (GLsizei)st.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      uint32_t *src = malloc(bytes);
      gb_transfer_fill_rgba(st.w, st.h, src);
      st.expected_crc = gb_crc32(src, bytes);
      glGenBuffers(1, &st.pbo);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, st.pbo);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)bytes, src, GL_STATIC_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      free(src);
   } else {
      uint32_t *mip0 = malloc(bytes);
      gb_transfer_fill_mip0(st.w, st.h, mip0);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)st.w, (GLsizei)st.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, mip0);
      free(mip0);
      glGenerateMipmap(GL_TEXTURE_2D);
   }
   gl2_check("transfer setup");

   struct gb_check chk = {0};
   if (do_verify) {
      gl2_xfer_rec(&st, 0);
      glFinish();
      if (st.kind != 3) {
         uint8_t *got = malloc(bytes);
         glBindTexture(GL_TEXTURE_2D, st.tex);
         glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, got);
         uint32_t got_crc = gb_crc32(got, bytes);
         chk.done = true;
         chk.crc = got_crc;
         chk.expected = st.expected_crc;
         chk.valid = (got_crc == st.expected_crc);
         free(got);
      } else {
         unsigned m4w = st.w >> 4, m4h = st.h >> 4;
         uint8_t *b4 = malloc((size_t)m4w * m4h * 4u);
         uint8_t blast[4] = {0};
         glBindTexture(GL_TEXTURE_2D, st.tex);
         glGetTexImage(GL_TEXTURE_2D, 4, GL_RGBA, GL_UNSIGNED_BYTE, b4);
         glGetTexImage(GL_TEXTURE_2D, (GLint)(GB_XFER_MIP_LEVELS - 1u), GL_RGBA, GL_UNSIGNED_BYTE, blast);
         gb_transfer_check_mipchain(b4, m4w, m4h, blast, &chk);
         free(b4);
      }
   }

   double xbytes = (double)bytes;
   if (st.kind == 3) {
      xbytes = 0.0;
      unsigned mw = st.w, mh = st.h;
      for (unsigned i = 1; i < GB_XFER_MIP_LEVELS; i++) {
         unsigned nw = mw > 1 ? mw / 2 : 1, nh = mh > 1 ? mh / 2 : 1;
         xbytes += ((double)mw * mh + (double)nw * nh) * 4.0;
         mw = nw; mh = nh;
      }
   }
   static const int defd[] = {1, 4};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int i = 0; i < nd; i++) {
      struct gb_pass u;
      gl2_run_stream(depths[i], o->warmup_ms, o->runs, o->run_ms, gl2_xfer_rec, &st, &u);
      gl2_emit_result(ci, o, depths[i], &u, &chk, "gbps", "GB/s", xbytes / 1e3, "gbps");
      gb_pass_free(&u);
   }
   if (st.fbo) glDeleteFramebuffers(1, &st.fbo);
   if (st.pbo) glDeleteBuffers(1, &st.pbo);
   glDeleteTextures(1, &st.tex);
   return (chk.done && !chk.valid) ? 1 : 0;
}

/* ======================= 3. render.postfx ======================= */

static const char *GL2_POST_SCENE_VS =
   "#version 330 core\n"
   "out vec4 v_hdr;\n"
   "void main() {\n"
   "  uint q = uint(gl_VertexID) / 6u; uint v = uint(gl_VertexID) % 6u;\n"
   "  uint gx = q & 63u; uint gy = (q >> 6) & 63u;\n"
   "  const vec2 corners[6] = vec2[6](\n"
   "    vec2(0.0,0.0), vec2(1.0,0.0), vec2(1.0,1.0),\n"
   "    vec2(0.0,0.0), vec2(1.0,1.0), vec2(0.0,1.0));\n"
   "  vec2 uv = (vec2(float(gx), float(gy)) + corners[v]) * (1.0 / 64.0);\n"
   "  float dx = float(gx) - 31.5; float dy = float(gy) - 31.5; float d2 = dx*dx + dy*dy;\n"
   "  if (d2 < 100.0) { float b = 3.5 - d2 * 0.02; v_hdr = vec4(b, b*0.65, b*0.30, 1.0); }\n"
   "  else v_hdr = vec4(0.15 + 0.55*uv.x, 0.15 + 0.55*uv.y, 0.35, 1.0);\n"
   "  float z = 0.2 + 0.6 * float((gx ^ gy) & 15u) / 15.0;\n"
   "  gl_Position = vec4(uv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);\n"
   "}\n";

static const char *GL2_POST_SCENE_FS =
   "#version 330 core\n"
   "in vec4 v_hdr; out vec4 o;\n"
   "void main() { o = v_hdr; }\n";

static const char *GL2_POST_DOWN_FS =
   "#version 330 core\n"
   "uniform sampler2D src_tex; uniform vec4 u_inv; uniform int u_bp;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  vec2 uv = gl_FragCoord.xy * u_inv.xy; vec2 d = u_inv.zw;\n"
   "  vec3 c = texture(src_tex, uv).rgb * 0.5\n"
   "         + texture(src_tex, uv + vec2(-d.x,-d.y)).rgb * 0.125\n"
   "         + texture(src_tex, uv + vec2( d.x,-d.y)).rgb * 0.125\n"
   "         + texture(src_tex, uv + vec2(-d.x, d.y)).rgb * 0.125\n"
   "         + texture(src_tex, uv + vec2( d.x, d.y)).rgb * 0.125;\n"
   "  if (u_bp != 0) c = max(c - vec3(1.0), vec3(0.0));\n"
   "  o = vec4(c, 1.0);\n"
   "}\n";

static const char *GL2_POST_UP_FS =
   "#version 330 core\n"
   "uniform sampler2D src_tex; uniform vec4 u_inv;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  vec2 uv = gl_FragCoord.xy * u_inv.xy; vec2 d = u_inv.zw;\n"
   "  vec3 c = (texture(src_tex, uv + vec2(-d.x,-d.y)).rgb +\n"
   "            texture(src_tex, uv + vec2( d.x,-d.y)).rgb +\n"
   "            texture(src_tex, uv + vec2(-d.x, d.y)).rgb +\n"
   "            texture(src_tex, uv + vec2( d.x, d.y)).rgb) * 0.25;\n"
   "  o = vec4(c * 0.5, 1.0);\n"
   "}\n";

static const char *GL2_POST_TONE_FS =
   "#version 330 core\n"
   "uniform sampler2D hdr_tex; uniform sampler2D bloom_tex; uniform vec2 u_inv;\n"
   "out vec4 o;\n"
   "void main() {\n"
   "  vec2 uv = gl_FragCoord.xy * u_inv;\n"
   "  vec3 hdr = texture(hdr_tex, uv).rgb + 0.35 * texture(bloom_tex, uv).rgb;\n"
   "  vec2 dv = uv - vec2(0.5);\n"
   "  hdr *= clamp(1.0 - dot(dv, dv) * 1.4, 0.25, 1.0);\n"
   "  vec3 m = clamp((hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14), 0.0, 1.0);\n"
   "  o = vec4(pow(m, vec3(1.0 / 2.2)), 1.0);\n"
   "}\n";

struct gl2_postfx {
   unsigned w, h, msaa;
   unsigned bw[GB_POSTFX_BLOOM_MIPS], bh[GB_POSTFX_BLOOM_MIPS];
   GLuint vao, scene_p, down_p, up_p, tone_p;
   GLuint ms_fbo, ms_col, ms_dep, hdr_fbo, hdr_tex, final_fbo, final_tex;
   GLuint bloom_fbo[GB_POSTFX_BLOOM_MIPS], bloom_tex[GB_POSTFX_BLOOM_MIPS];
   GLint loc_down_inv, loc_down_bp, loc_up_inv, loc_tone_inv;
};

static void
gl2_postfx_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl2_postfx *st = u;
   glBindVertexArray(st->vao);

   glBindFramebuffer(GL_FRAMEBUFFER, st->msaa == 4 ? st->ms_fbo : st->hdr_fbo);
   glViewport(0, 0, (GLsizei)st->w, (GLsizei)st->h);
   glEnable(GL_DEPTH_TEST);
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);
   glClearColor(0.04f, 0.05f, 0.08f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glUseProgram(st->scene_p);
   glDrawArrays(GL_TRIANGLES, 0, GB_POSTFX_QUADS * 6);
   glDisable(GL_DEPTH_TEST);

   if (st->msaa == 4) {
      glBindFramebuffer(GL_READ_FRAMEBUFFER, st->ms_fbo);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, st->hdr_fbo);
      glBlitFramebuffer(0, 0, (GLint)st->w, (GLint)st->h, 0, 0, (GLint)st->w, (GLint)st->h,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST);
      /* Discard the multisample colour and depth after the resolve, as the
       * Vulkan and Metal executors do with DONT_CARE store operations. */
      static const GLenum ms_att[2] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
      glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 2, ms_att);
   }

   glUseProgram(st->down_p);
   glActiveTexture(GL_TEXTURE0);
   for (unsigned i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      unsigned sw = i == 0 ? st->w : st->bw[i - 1];
      unsigned sh = i == 0 ? st->h : st->bh[i - 1];
      glBindFramebuffer(GL_FRAMEBUFFER, st->bloom_fbo[i]);
      glViewport(0, 0, (GLsizei)st->bw[i], (GLsizei)st->bh[i]);
      glBindTexture(GL_TEXTURE_2D, i == 0 ? st->hdr_tex : st->bloom_tex[i - 1]);
      glUniform4f(st->loc_down_inv, 1.0f / (float)st->bw[i], 1.0f / (float)st->bh[i],
                  1.0f / (float)sw, 1.0f / (float)sh);
      glUniform1i(st->loc_down_bp, i == 0 ? 1 : 0);
      glDrawArrays(GL_TRIANGLES, 0, 3);
   }

   glEnable(GL_BLEND);
   glBlendFunc(GL_ONE, GL_ONE);
   glUseProgram(st->up_p);
   for (int i = (int)GB_POSTFX_BLOOM_MIPS - 2; i >= 0; i--) {
      glBindFramebuffer(GL_FRAMEBUFFER, st->bloom_fbo[i]);
      glViewport(0, 0, (GLsizei)st->bw[i], (GLsizei)st->bh[i]);
      glBindTexture(GL_TEXTURE_2D, st->bloom_tex[i + 1]);
      glUniform4f(st->loc_up_inv, 1.0f / (float)st->bw[i], 1.0f / (float)st->bh[i],
                  1.0f / (float)st->bw[i + 1], 1.0f / (float)st->bh[i + 1]);
      glDrawArrays(GL_TRIANGLES, 0, 3);
   }
   glDisable(GL_BLEND);

   glBindFramebuffer(GL_FRAMEBUFFER, st->final_fbo);
   glViewport(0, 0, (GLsizei)st->w, (GLsizei)st->h);
   glUseProgram(st->tone_p);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, st->hdr_tex);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, st->bloom_tex[0]);
   glUniform2f(st->loc_tone_inv, 1.0f / (float)st->w, 1.0f / (float)st->h);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static GLuint
make_tex2d_linear(GLint ifmt, unsigned w, unsigned h, GLenum fmt, GLenum ty)
{
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexImage2D(GL_TEXTURE_2D, 0, ifmt, (GLsizei)w, (GLsizei)h, 0, fmt, ty, NULL);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   return t;
}

static int
gl2_run_postfx(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"res", "msaa", NULL};
   if (!gb_check_params(o, known))
      gb_die("unknown parameter for render.postfx");
   if (ci->es) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "render.postfx requires desktop GL");
      return 0;
   }
   unsigned w = 0, h = 0;
   if (!gl2_parse_res(gb_param(o, "res", "1440p"), &w, &h))
      gb_die("res must be 720p, 1080p, 1440p or 2160p");
   long msaa = gb_param_int(o, "msaa", 4);
   if (msaa != 1 && msaa != 4)
      gb_die("msaa must be 1 or 4");

   struct gl2_postfx st = {.w = w, .h = h, .msaa = (unsigned)msaa};
   glGenVertexArrays(1, &st.vao);
   st.scene_p = gl2_compile(GL2_POST_SCENE_VS, GL2_POST_SCENE_FS);
   st.down_p = gl2_compile(GL2_FS_VS, GL2_POST_DOWN_FS);
   st.up_p = gl2_compile(GL2_FS_VS, GL2_POST_UP_FS);
   st.tone_p = gl2_compile(GL2_FS_VS, GL2_POST_TONE_FS);
   st.loc_down_inv = glGetUniformLocation(st.down_p, "u_inv");
   st.loc_down_bp = glGetUniformLocation(st.down_p, "u_bp");
   st.loc_up_inv = glGetUniformLocation(st.up_p, "u_inv");
   st.loc_tone_inv = glGetUniformLocation(st.tone_p, "u_inv");
   glUseProgram(st.down_p); glUniform1i(glGetUniformLocation(st.down_p, "src_tex"), 0);
   glUseProgram(st.up_p); glUniform1i(glGetUniformLocation(st.up_p, "src_tex"), 0);
   glUseProgram(st.tone_p);
   glUniform1i(glGetUniformLocation(st.tone_p, "hdr_tex"), 0);
   glUniform1i(glGetUniformLocation(st.tone_p, "bloom_tex"), 1);

   st.hdr_tex = make_tex2d_linear(GL_RGBA16F, w, h, GL_RGBA, GL_HALF_FLOAT);
   glGenFramebuffers(1, &st.hdr_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, st.hdr_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, st.hdr_tex, 0);

   glGenRenderbuffers(1, &st.ms_dep);
   glBindRenderbuffer(GL_RENDERBUFFER, st.ms_dep);
   if (msaa == 4) {
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT32F, (GLsizei)w, (GLsizei)h);
      glGenRenderbuffers(1, &st.ms_col);
      glBindRenderbuffer(GL_RENDERBUFFER, st.ms_col);
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA16F, (GLsizei)w, (GLsizei)h);
      glGenFramebuffers(1, &st.ms_fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, st.ms_fbo);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, st.ms_col);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, st.ms_dep);
   } else {
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, (GLsizei)w, (GLsizei)h);
      glBindFramebuffer(GL_FRAMEBUFFER, st.hdr_fbo);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, st.ms_dep);
   }

   unsigned bw = w, bh = h;
   glGenFramebuffers(GB_POSTFX_BLOOM_MIPS, st.bloom_fbo);
   for (unsigned i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      bw = bw > 1 ? bw / 2 : 1;
      bh = bh > 1 ? bh / 2 : 1;
      st.bw[i] = bw; st.bh[i] = bh;
      st.bloom_tex[i] = make_tex2d_linear(GL_RGBA16F, bw, bh, GL_RGBA, GL_HALF_FLOAT);
      glBindFramebuffer(GL_FRAMEBUFFER, st.bloom_fbo[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, st.bloom_tex[i], 0);
   }

   st.final_tex = make_tex2d_linear(GL_RGBA8, w, h, GL_RGBA, GL_UNSIGNED_BYTE);
   glGenFramebuffers(1, &st.final_fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, st.final_fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, st.final_tex, 0);
   gl2_check("postfx setup");

   struct gb_check chk = {0};
   if (do_verify) {
      gl2_postfx_rec(&st, 0);
      glFinish();
      uint8_t *got = malloc((size_t)w * h * 4u);
      glBindFramebuffer(GL_READ_FRAMEBUFFER, st.final_fbo);
      glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, got);
      gb_postfx_check_fb(got, w, h, &chk);
      free(got);
   }

   static const int defd[] = {1, 2};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int i = 0; i < nd; i++) {
      struct gb_pass u;
      gl2_run_stream(depths[i], o->warmup_ms, o->runs, o->run_ms, gl2_postfx_rec, &st, &u);
      gl2_emit_result(ci, o, depths[i], &u, &chk, "fps", "fps", 1e6, "fps");
      gb_pass_free(&u);
   }

   glDeleteFramebuffers(1, &st.final_fbo);
   glDeleteTextures(1, &st.final_tex);
   glDeleteFramebuffers(GB_POSTFX_BLOOM_MIPS, st.bloom_fbo);
   glDeleteTextures(GB_POSTFX_BLOOM_MIPS, st.bloom_tex);
   if (st.ms_fbo) glDeleteFramebuffers(1, &st.ms_fbo);
   if (st.ms_col) glDeleteRenderbuffers(1, &st.ms_col);
   glDeleteRenderbuffers(1, &st.ms_dep);
   glDeleteFramebuffers(1, &st.hdr_fbo);
   glDeleteTextures(1, &st.hdr_tex);
   glDeleteProgram(st.scene_p);
   glDeleteProgram(st.down_p);
   glDeleteProgram(st.up_p);
   glDeleteProgram(st.tone_p);
   glDeleteVertexArrays(1, &st.vao);
   return (chk.done && !chk.valid) ? 1 : 0;
}

/* ======================= dispatch ======================= */

bool
gb_gl_is_wave2(const char *test)
{
   if (!test)
      return false;
   return !strcmp(test, "render.shadow") || !strcmp(test, "transfer.ops") || !strcmp(test, "render.postfx");
}

int
gb_gl_run_wave2(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify)
{
   if (!strcmp(o->test, "render.shadow")) return gl2_run_shadow(ci, o, do_verify);
   if (!strcmp(o->test, "transfer.ops"))  return gl2_run_xfer(ci, o, do_verify);
   if (!strcmp(o->test, "render.postfx")) return gl2_run_postfx(ci, o, do_verify);
   gb_die("unknown wave2 GL test %s", o->test);
}
