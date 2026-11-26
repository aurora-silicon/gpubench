// SPDX-License-Identifier: GPL-2.0-only
/* OpenGL 90%-coverage parity benchmarks for gpubench-gl:
 *   compress.bw (mrt4 | rgba32f | msaa4 | storage)
 *   render.draws (direct VAO+UBO vs glMultiDrawElementsIndirect)
 *   compute.uber (divergent | uniform, CRC-verified against gb_uber_ref)
 *   compute.image (2D imageLoad + imageStore)
 *   compile.pso (GLSL shader compile + program link cold latency)
 *   sync.finish (glFlush + glClientWaitSync vs glFinish) */
#include "gl_wave1.h"
#include "gb_params.h"
#include "gb_wave1_ref.h"

#include <epoxy/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void
gl_die_err(const char *where)
{
   GLenum e = glGetError();
   if (e != GL_NO_ERROR)
      gb_die("%s: GL error 0x%x", where, e);
}

static bool
gl_parse_res(const char *s, unsigned *w, unsigned *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static GLuint
gl_compile_prog(const char *vs_src, const char *fs_src, const char *cs_src)
{
   GLuint p = glCreateProgram();
   GLuint sh[2] = {0, 0};
   int n = 0;
   if (cs_src) {
      sh[n] = glCreateShader(GL_COMPUTE_SHADER);
      glShaderSource(sh[n], 1, &cs_src, NULL);
      glCompileShader(sh[n]);
      n++;
   } else {
      sh[n] = glCreateShader(GL_VERTEX_SHADER);
      glShaderSource(sh[n], 1, &vs_src, NULL);
      glCompileShader(sh[n]);
      n++;
      sh[n] = glCreateShader(GL_FRAGMENT_SHADER);
      glShaderSource(sh[n], 1, &fs_src, NULL);
      glCompileShader(sh[n]);
      n++;
   }
   for (int i = 0; i < n; i++) {
      GLint ok = 0;
      glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[2048]; glGetShaderInfoLog(sh[i], sizeof(log), NULL, log);
         gb_die("GL shader compile failed: %s", log);
      }
      glAttachShader(p, sh[i]);
   }
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[2048]; glGetProgramInfoLog(p, sizeof(log), NULL, log);
      gb_die("GL program link failed: %s", log);
   }
   for (int i = 0; i < n; i++) glDeleteShader(sh[i]);
   return p;
}

typedef void (*gl_rec_fn)(void *user, uint64_t k);

static void
gl_run_stream(int depth, int warm_ms, int runs, int run_ms, gl_rec_fn rec, void *user, bool use_finish,
              struct gb_pass *p)
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
      if (!full && now >= end) break;
      if (full) {
         GLenum st = glClientWaitSync(slots[done % depth].fence, 0, 10000000000ull);
         if (st != GL_ALREADY_SIGNALED && st != GL_CONDITION_SATISFIED) { p->failed = true; break; }
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
      if (use_finish) {
         glFinish();
         uint64_t r2 = gb_now_ns();
         if (r0 >= start && r0 < end) {
            p->win_count[(r2 - start) / win < (uint64_t)runs ? (r2 - start) / win : (uint64_t)(runs - 1)]++;
            gb_samples_push(&p->record_cpu, (r1 - r0) / 1e3);
            gb_samples_push(&p->submit_cpu, (r2 - r1) / 1e3);
            gb_samples_push(&p->latency, (r2 - r1) / 1e3);
         }
         sent++; done++;
         continue;
      }
      slots[sent % depth].submit = r1;
      slots[sent % depth].fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      glFlush();
      uint64_t r2 = gb_now_ns();
      if (!slots[sent % depth].fence) { p->failed = true; break; }
      if (r0 >= start && r0 < end) {
         gb_samples_push(&p->record_cpu, (r1 - r0) / 1e3);
         gb_samples_push(&p->submit_cpu, (r2 - r1) / 1e3);
      }
      sent++;
   }
   glFinish();
   for (int i = 0; i < depth; i++)
      if (slots[i].fence) glDeleteSync(slots[i].fence);
   if (glGetError() != GL_NO_ERROR) p->failed = true;
   for (int i = 0; i < runs; i++) p->submits += p->win_count[i];
}

static void
emit_skip(const struct gl_ctx_info *ci, const struct gb_opts *o, int depth, const char *why)
{
   struct gb_result r;
   gb_result_begin(&r, o, o->mode, depth, NULL);
   gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
   gb_result_finish(&r, "skip", NULL, why);
}

/* ---------------- 1. compress.bw (GL) ---------------- */

static const char *GL_FS_VS =
   "#version 330 core\n"
   "void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
static const char *GL_CMP_MRT4_FS =
   "#version 330 core\n"
   "uniform float u_k;\n"
   "layout(location=0) out vec4 g0; layout(location=1) out vec4 g1;\n"
   "layout(location=2) out vec4 g2; layout(location=3) out float g3;\n"
   "void main() {\n"
   "  vec2 p = floor(gl_FragCoord.xy * 0.0625) * 0.015625;\n"
   "  g0 = vec4(fract(p.x + u_k), fract(p.y + 0.25), fract(p.x + p.y), 1.0);\n"
   "  g1 = vec4(fract(p.y + u_k), fract(p.x + 0.50), 0.75, 0.5);\n"
   "  g2 = vec4(0.25, fract(p.x * 0.5 + u_k), fract(p.y * 0.5), 1.0);\n"
   "  g3 = 0.25 + 0.5 * fract(p.x + u_k);\n"
   "}\n";
static const char *GL_CMP_W1_FS =
   "#version 330 core\n"
   "uniform float u_k; layout(location=0) out vec4 o;\n"
   "void main() { vec2 p = floor(gl_FragCoord.xy * 0.0625) * 0.015625; o = vec4(fract(p.x + u_k), fract(p.y + 0.25), fract(p.x + p.y + 0.5), 1.0); }\n";
static const char *GL_CMP_S4_FS =
   "#version 330 core\n"
   "uniform sampler2D t0, t1, t2, t3; layout(location=0) out vec4 o;\n"
   "void main() { ivec2 p = ivec2(gl_FragCoord.xy); vec3 s = (texelFetch(t0,p,0).rgb + texelFetch(t1,p,0).rgb + texelFetch(t2,p,0).rgb)*0.25 + vec3(texelFetch(t3,p,0).r*0.25); o = vec4(clamp(s,0.0,1.0),1.0); }\n";
static const char *GL_CMP_S1_FS =
   "#version 330 core\n"
   "uniform sampler2D t0; layout(location=0) out vec4 o;\n"
   "void main() { o = vec4(clamp(texelFetch(t0, ivec2(gl_FragCoord.xy), 0).rgb, 0.0, 1.0), 1.0); }\n";
static const char *GL_CMP_STORE_CS =
   "#version 430 core\n"
   "layout(local_size_x=16, local_size_y=16) in;\n"
   "layout(rgba8, binding=0) writeonly uniform image2D dst;\n"
   "uniform int u_w, u_h; uniform float u_k;\n"
   "void main() {\n"
   "  ivec2 p = ivec2(gl_GlobalInvocationID.xy); if (p.x >= u_w || p.y >= u_h) return;\n"
   "  vec2 q = floor(vec2(p) * 0.0625) * 0.015625;\n"
   "  imageStore(dst, p, vec4(fract(q.x + u_k), fract(q.y + 0.25), fract(q.x + q.y), 1.0));\n"
   "}\n";

struct gl_cmp_st {
   int kind; /* 0=mrt4, 1=rgba32f, 2=msaa4, 3=storage */
   unsigned w, h;
   GLuint vao, fbo_w, fbo_r, fbo_res, rb_ms;
   GLuint tex[4], final_tex;
   GLuint pw, pr;
   GLint loc_k;
};

static void
gl_cmp_rec(void *u, uint64_t k)
{
   struct gl_cmp_st *s = u;
   float kf = (float)(k & 15u) * 0.03125f;
   glBindVertexArray(s->vao);
   if (s->kind == 3) {
      glUseProgram(s->pw);
      glUniform1f(s->loc_k, kf);
      glBindImageTexture(0, s->tex[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
      glDispatchCompute((s->w + 15u) / 16u, (s->h + 15u) / 16u, 1);
      glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
   } else {
      glBindFramebuffer(GL_FRAMEBUFFER, s->fbo_w);
      glViewport(0, 0, (GLsizei)s->w, (GLsizei)s->h);
      glUseProgram(s->pw);
      glUniform1f(s->loc_k, kf);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      if (s->kind == 2) {
         glBindFramebuffer(GL_READ_FRAMEBUFFER, s->fbo_w);
         glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s->fbo_res);
         glBlitFramebuffer(0, 0, (GLint)s->w, (GLint)s->h, 0, 0, (GLint)s->w, (GLint)s->h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
         /* The Vulkan and Metal executors discard the samples after the
          * resolve (DONT_CARE / MultisampleResolve). Without this, GL has to
          * store all four samples for a later pass that never reads them. */
         static const GLenum ms_att = GL_COLOR_ATTACHMENT0;
         glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 1, &ms_att);
      }
   }
   glBindFramebuffer(GL_FRAMEBUFFER, s->fbo_r);
   glViewport(0, 0, (GLsizei)s->w, (GLsizei)s->h);
   glUseProgram(s->pr);
   int nt = s->kind == 0 ? 4 : 1;
   for (int i = 0; i < nt; i++) {
      glActiveTexture(GL_TEXTURE0 + (GLenum)i);
      glBindTexture(GL_TEXTURE_2D, s->tex[i]);
   }
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static int
run_gl_compress(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "res", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "mrt4");
   int k = !strcmp(kind, "mrt4") ? 0 : !strcmp(kind, "rgba32f") ? 1 : !strcmp(kind, "msaa4") ? 2 : !strcmp(kind, "storage") ? 3 : -1;
   if (k < 0) gb_die("kind must be mrt4, rgba32f, msaa4 or storage");
   unsigned w = 0, h = 0;
   if (!gl_parse_res(gb_param(o, "res", "1440p"), &w, &h)) gb_die("bad res");
   if (ci->es || (k == 3 && !ci->compute)) {
      emit_skip(ci, o, 1, "requires desktop GL 4.3");
      return 0;
   }

   struct gl_cmp_st s = {.kind = k, .w = w, .h = h};
   glGenVertexArrays(1, &s.vao); glBindVertexArray(s.vao);
   glGenFramebuffers(1, &s.fbo_r);
   glGenTextures(1, &s.final_tex);
   glBindTexture(GL_TEXTURE_2D, s.final_tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
   glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_r);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.final_tex, 0);

   double bpp_total = 0;
   if (k == 0) {
      bpp_total = 44.0;
      const GLenum fmts[4] = {GL_RGBA8, GL_RGBA16F, GL_RGBA8, GL_R32F};
      const GLenum bufs[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
      glGenFramebuffers(1, &s.fbo_w); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_w);
      glGenTextures(4, s.tex);
      for (int i = 0; i < 4; i++) {
         glBindTexture(GL_TEXTURE_2D, s.tex[i]);
         glTexStorage2D(GL_TEXTURE_2D, 1, fmts[i], (GLsizei)w, (GLsizei)h);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
         glFramebufferTexture2D(GL_FRAMEBUFFER, bufs[i], GL_TEXTURE_2D, s.tex[i], 0);
      }
      glDrawBuffers(4, bufs);
      s.pw = gl_compile_prog(GL_FS_VS, GL_CMP_MRT4_FS, NULL);
      s.pr = gl_compile_prog(GL_FS_VS, GL_CMP_S4_FS, NULL);
      glUseProgram(s.pr);
      glUniform1i(glGetUniformLocation(s.pr, "t0"), 0);
      glUniform1i(glGetUniformLocation(s.pr, "t1"), 1);
      glUniform1i(glGetUniformLocation(s.pr, "t2"), 2);
      glUniform1i(glGetUniformLocation(s.pr, "t3"), 3);
   } else if (k == 1) {
      bpp_total = 36.0;
      glGenFramebuffers(1, &s.fbo_w); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_w);
      glGenTextures(1, &s.tex[0]); glBindTexture(GL_TEXTURE_2D, s.tex[0]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, (GLsizei)w, (GLsizei)h);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.tex[0], 0);
      s.pw = gl_compile_prog(GL_FS_VS, GL_CMP_W1_FS, NULL);
      s.pr = gl_compile_prog(GL_FS_VS, GL_CMP_S1_FS, NULL);
      glUseProgram(s.pr); glUniform1i(glGetUniformLocation(s.pr, "t0"), 0);
   } else if (k == 2) {
      bpp_total = 28.0;
      glGenFramebuffers(1, &s.fbo_w); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_w);
      glGenRenderbuffers(1, &s.rb_ms); glBindRenderbuffer(GL_RENDERBUFFER, s.rb_ms);
      glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, (GLsizei)w, (GLsizei)h);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, s.rb_ms);
      glGenFramebuffers(1, &s.fbo_res); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_res);
      glGenTextures(1, &s.tex[0]); glBindTexture(GL_TEXTURE_2D, s.tex[0]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.tex[0], 0);
      s.pw = gl_compile_prog(GL_FS_VS, GL_CMP_W1_FS, NULL);
      s.pr = gl_compile_prog(GL_FS_VS, GL_CMP_S1_FS, NULL);
      glUseProgram(s.pr); glUniform1i(glGetUniformLocation(s.pr, "t0"), 0);
   } else {
      bpp_total = 12.0;
      glGenTextures(1, &s.tex[0]); glBindTexture(GL_TEXTURE_2D, s.tex[0]);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      s.pw = gl_compile_prog(NULL, NULL, GL_CMP_STORE_CS);
      glUseProgram(s.pw);
      glUniform1i(glGetUniformLocation(s.pw, "u_w"), (GLint)w);
      glUniform1i(glGetUniformLocation(s.pw, "u_h"), (GLint)h);
      s.pr = gl_compile_prog(GL_FS_VS, GL_CMP_S1_FS, NULL);
      glUseProgram(s.pr); glUniform1i(glGetUniformLocation(s.pr, "t0"), 0);
   }
   s.loc_k = glGetUniformLocation(s.pw, "u_k");
   glFinish(); gl_die_err("compress.bw setup");

   const int defd[2] = {1, 4};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int di = 0; di < nd; di++) {
      struct gb_pass u = {0};
      gl_run_stream(depths[di], o->warmup_ms, o->runs, o->run_ms, gl_cmp_rec, &s, false, &u);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depths[di], NULL);
      gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (u.failed || !u.submits) {
         gb_result_finish(&r, "fail", NULL, "GL error or no completions");
      } else {
         gb_result_stream_metrics(&r, &u, NULL);
         struct gb_samples us = {0}; gb_pass_rate(&u, &us);
         gb_derive_rate(&r, "gbps", "GB/s", &us, bpp_total * (double)w * h / 1e9 * 1e6);
         gb_samples_free(&us);
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      gb_pass_free(&u);
   }
   glDeleteProgram(s.pw); glDeleteProgram(s.pr);
   glDeleteTextures(4, s.tex); glDeleteTextures(1, &s.final_tex);
   if (s.rb_ms) glDeleteRenderbuffers(1, &s.rb_ms);
   if (s.fbo_w) glDeleteFramebuffers(1, &s.fbo_w);
   if (s.fbo_res) glDeleteFramebuffers(1, &s.fbo_res);
   glDeleteFramebuffers(1, &s.fbo_r);
   glDeleteVertexArrays(1, &s.vao);
   return 0;
}

/* ---------------- 2. render.draws (direct VAO+UBO vs MDI) ---------------- */

static const char *GL_DRAW_DIR_VS =
   "#version 330 core\n"
   "layout(location=0) in vec2 in_pos;\n"
   "layout(std140) uniform DrawUBO { vec4 rect_col; };\n"
   "out vec3 v_col;\n"
   "void main() { vec2 px = rect_col.xy + in_pos * 10.0; gl_Position = vec4((px / 1024.0) * 2.0 - 1.0, 0.0, 1.0); v_col = vec3(rect_col.zw, 0.5); }\n";
static const char *GL_DRAW_MDI_VS =
   "#version 430 core\n"
   "#extension GL_ARB_shader_draw_parameters : require\n"
   "layout(location=0) in vec2 in_pos;\n"
   "out vec3 v_col;\n"
   "void main() {\n"
   "  uint d = uint(gl_DrawIDARB); uint cx = (d % 10000u) % 100u; uint cy = (d % 10000u) / 100u;\n"
   "  vec2 px = vec2(float(cx * 10u), float(cy * 10u)) + in_pos * 10.0;\n"
   "  gl_Position = vec4((px / 1024.0) * 2.0 - 1.0, 0.0, 1.0);\n"
   "  v_col = vec3(float((d * 37u) & 255u) / 255.0, float((d * 73u) & 255u) / 255.0, 0.5);\n"
   "}\n";
static const char *GL_DRAW_FS =
   "#version 330 core\n"
   "in vec3 v_col; layout(location=0) out vec4 o;\n"
   "void main() { o = vec4(v_col, 1.0); }\n";

struct gl_draws_st {
   bool mdi;
   unsigned draws;
   GLuint prog, fbo, target, vao[2], vb, ib, ubo, indirect;
   unsigned ubo_stride;
};

static void
gl_draws_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl_draws_st *s = u;
   glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
   glViewport(0, 0, 1024, 1024);
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(s->prog);
   if (s->mdi) {
      glBindVertexArray(s->vao[0]);
      glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s->indirect);
      glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, NULL, (GLsizei)s->draws, 0);
   } else {
      for (unsigned d = 0; d < s->draws; d++) {
         glBindVertexArray(s->vao[d & 1u]);
         glBindBufferRange(GL_UNIFORM_BUFFER, 0, s->ubo, (GLintptr)(d * s->ubo_stride), 16);
         glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
      }
   }
}

static int
run_gl_draws(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"draws", "mode", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   long draws = gb_param_int(o, "draws", 2048);
   const char *mode = gb_param(o, "mode", "direct");
   bool mdi = !strcmp(mode, "mdi");
   if (!mdi && strcmp(mode, "direct")) gb_die("mode must be direct or mdi");
   if (draws < 1 || draws > 10000) gb_die("draws must be 1..10000");
   if (ci->es || (mdi && !ci->compute)) {
      emit_skip(ci, o, 1, "requires desktop GL 4.3");
      return 0;
   }

   struct gl_draws_st s = {.mdi = mdi, .draws = (unsigned)draws};
   s.prog = gl_compile_prog(mdi ? GL_DRAW_MDI_VS : GL_DRAW_DIR_VS, GL_DRAW_FS, NULL);
   if (!mdi) {
      GLuint bi = glGetUniformBlockIndex(s.prog, "DrawUBO");
      glUniformBlockBinding(s.prog, bi, 0);
   }
   glGenFramebuffers(1, &s.fbo); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
   glGenTextures(1, &s.target); glBindTexture(GL_TEXTURE_2D, s.target);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1024, 1024);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.target, 0);

   const float verts[8] = {0, 0, 1, 0, 1, 1, 0, 1};
   const uint16_t idxs[6] = {0, 1, 2, 0, 2, 3};
   glGenBuffers(1, &s.vb); glBindBuffer(GL_ARRAY_BUFFER, s.vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
   glGenBuffers(1, &s.ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.ib);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxs), idxs, GL_STATIC_DRAW);
   glGenVertexArrays(2, s.vao);
   for (int i = 0; i < 2; i++) {
      glBindVertexArray(s.vao[i]);
      glBindBuffer(GL_ARRAY_BUFFER, s.vb);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.ib);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
   }

   if (mdi) {
      struct { uint32_t count, inst, first_idx, base_vtx, base_inst; } *cmds = calloc((size_t)draws, 20);
      for (unsigned d = 0; d < (unsigned)draws; d++) { cmds[d].count = 6; cmds[d].inst = 1; cmds[d].base_inst = d; }
      glGenBuffers(1, &s.indirect); glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s.indirect);
      glBufferData(GL_DRAW_INDIRECT_BUFFER, (GLsizeiptr)((size_t)draws * 20), cmds, GL_STATIC_DRAW);
      free(cmds);
   } else {
      GLint align = 256; glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &align);
      s.ubo_stride = (unsigned)((align + 15) & ~15);
      uint8_t *buf = calloc((size_t)draws, s.ubo_stride);
      for (unsigned d = 0; d < (unsigned)draws; d++) {
         float *f = (float *)(buf + (size_t)d * s.ubo_stride);
         f[0] = (float)(((d % 10000u) % 100u) * 10u);
         f[1] = (float)(((d % 10000u) / 100u) * 10u);
         f[2] = (float)((d * 37u) & 255u) / 255.0f;
         f[3] = (float)((d * 73u) & 255u) / 255.0f;
      }
      glGenBuffers(1, &s.ubo); glBindBuffer(GL_UNIFORM_BUFFER, s.ubo);
      glBufferData(GL_UNIFORM_BUFFER, (GLsizeiptr)((size_t)draws * s.ubo_stride), buf, GL_STATIC_DRAW);
      free(buf);
   }
   glFinish(); gl_die_err("render.draws setup");

   const int defd[2] = {1, 4};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int di = 0; di < nd; di++) {
      struct gb_pass u = {0};
      gl_run_stream(depths[di], o->warmup_ms, o->runs, o->run_ms, gl_draws_rec, &s, false, &u);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depths[di], NULL);
      gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (u.failed || !u.submits) {
         gb_result_finish(&r, "fail", NULL, "GL error or no completions");
      } else {
         gb_result_stream_metrics(&r, &u, NULL);
         struct gb_samples us = {0}; gb_pass_rate(&u, &us);
         gb_derive_rate(&r, "mdraws_s", "Mdraws/s", &us, (double)draws);
         gb_samples_free(&us);
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      gb_pass_free(&u);
   }
   glDeleteProgram(s.prog); glDeleteBuffers(1, &s.vb); glDeleteBuffers(1, &s.ib);
   if (s.ubo) glDeleteBuffers(1, &s.ubo);
   if (s.indirect) glDeleteBuffers(1, &s.indirect);
   glDeleteVertexArrays(2, s.vao); glDeleteTextures(1, &s.target); glDeleteFramebuffers(1, &s.fbo);
   return 0;
}

/* ---------------- 3. compute.uber (GL, CRC-checked) ---------------- */

static const char *GL_UBER_BODY =
   "#version 430 core\n"
   "layout(local_size_x=256) in;\n"
   "layout(std430, binding=0) writeonly buffer Out { uint o[]; };\n"
   "uniform uint u_seed, u_iters, u_uni;\n"
   "#define U_B(X, p, a, b, c, d) X(a, p) X(b, a) X(c, b) X(d, c)\n"
   "#define U_ALL(X) U_B(X,a23,a0,a1,a2,a3) U_B(X,a3,a4,a5,a6,a7) U_B(X,a7,a8,a9,a10,a11) U_B(X,a11,a12,a13,a14,a15) U_B(X,a15,a16,a17,a18,a19) U_B(X,a19,a20,a21,a22,a23)\n"
   "#define U_DECL(a, p) sd = sd * uvec4(747796405u) + uvec4(2891336453u); uvec4 a = (sd ^ (sd >> uvec4(15u))) + ks;\n"
   "#define U_STEP(a, p) a += p ^ (a >> uvec4(9u));\n"
   "#define U_FOLD(a) sd = (sd ^ (a >> uvec4(sd.x & 15u))) * uvec4(16777619u);\n"
   "#define U_NARROW(ALT) { U_DECL(b0,b3) U_DECL(b1,b0) U_DECL(b2,b1) U_DECL(b3,b2) for (uint i=0u; i<u_iters; i++) { if (ALT) { b0+=b3^(b0<<uvec4(7u)); b1+=b0^(b1<<uvec4(7u)); b2+=b1^(b2<<uvec4(7u)); b3+=b2^(b3<<uvec4(7u)); } else { b0+=b3^(b0>>uvec4(9u)); b1+=b0^(b1>>uvec4(9u)); b2+=b1^(b2>>uvec4(9u)); b3+=b2^(b3>>uvec4(9u)); } } U_FOLD(b0) U_FOLD(b1) U_FOLD(b2) U_FOLD(b3) }\n"
   "void main() {\n"
   "  uint id = gl_GlobalInvocationID.x; uint grp = gl_WorkGroupID.x;\n"
   "  uvec4 ks = uvec4(u_seed); uvec4 sd = ((uvec4(id) + uvec4(1u,2u,3u,4u)) * uvec4(2654435761u)) ^ ks;\n"
   "  U_ALL(U_DECL) for (uint s=0u; s<4u; s++) { U_ALL(U_STEP) }\n"
   "  uint sel = u_uni != 0u ? ((grp ^ u_seed) & 3u) : (((id ^ (id >> 2u)) ^ u_seed) & 3u);\n"
   "  if (sel == 0u) { U_FOLD(a0) U_FOLD(a1) U_FOLD(a2) U_FOLD(a3) U_FOLD(a4) U_FOLD(a5) U_NARROW(false) sd ^= uvec4(0x243f6a88u); }\n"
   "  else if (sel == 1u) { U_FOLD(a6) U_FOLD(a7) U_FOLD(a8) U_FOLD(a9) U_FOLD(a10) U_FOLD(a11) U_NARROW(true) sd ^= uvec4(0x85a308d3u); }\n"
   "  else if (sel == 2u) { U_FOLD(a12) U_FOLD(a13) U_FOLD(a14) U_FOLD(a15) U_FOLD(a16) U_FOLD(a17) U_NARROW(false) sd ^= uvec4(0x13198a2eu); }\n"
   "  else { U_FOLD(a18) U_FOLD(a19) U_FOLD(a20) U_FOLD(a21) U_FOLD(a22) U_FOLD(a23) U_NARROW(true) sd ^= uvec4(0x03707344u); }\n"
   "  o[id] = sd.x ^ sd.y ^ sd.z ^ sd.w;\n"
   "}\n";

static void
gl_uber_rec(void *u, uint64_t k)
{
   (void)k;
   GLuint prog = *(GLuint *)u;
   glUseProgram(prog);
   glDispatchCompute(GB_UBER_GROUPS, 1, 1);
}

static int
run_gl_uber(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "divergent");
   bool uni = !strcmp(kind, "uniform");
   if (!uni && strcmp(kind, "divergent")) gb_die("kind must be divergent or uniform");
   if (ci->es || !ci->compute) { emit_skip(ci, o, 1, "requires desktop GL 4.3 compute"); return 0; }

   GLuint prog = gl_compile_prog(NULL, NULL, GL_UBER_BODY);
   glUseProgram(prog);
   glUniform1ui(glGetUniformLocation(prog, "u_seed"), GB_UBER_SEED);
   glUniform1ui(glGetUniformLocation(prog, "u_iters"), GB_UBER_ITERS);
   glUniform1ui(glGetUniformLocation(prog, "u_uni"), uni ? 1u : 0u);
   GLuint ssbo; glGenBuffers(1, &ssbo); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
   size_t bytes = (size_t)GB_UBER_THREADS * sizeof(uint32_t);
   glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, NULL, GL_DYNAMIC_COPY);
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

   uint32_t *ref = malloc(bytes);
   uint32_t ref_crc = gb_uber_ref(uni ? GB_UBER_UNI : GB_UBER_DIV, NULL, ref, GB_UBER_THREADS);

   const int defd[2] = {1, 4};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int di = 0; di < nd; di++) {
      struct gb_pass u = {0};
      gl_run_stream(depths[di], o->warmup_ms, o->runs, o->run_ms, gl_uber_rec, &prog, false, &u);
      struct gb_check chk = {0};
      if (!u.failed && u.submits) {
         glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
         glFinish();
         uint32_t *got = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, GL_MAP_READ_BIT);
         if (got) {
            gb_check_words(&chk, got, ref, ref_crc, GB_UBER_THREADS);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
         }
      }
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depths[di], NULL);
      gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (u.failed || !u.submits) {
         gb_result_finish(&r, "fail", NULL, "GL error or no completions");
      } else {
         gb_result_stream_metrics(&r, &u, NULL);
         struct gb_samples us = {0}; gb_pass_rate(&u, &us);
         gb_derive_rate(&r, "mthreads_s", "Mthreads/s", &us, (double)GB_UBER_THREADS);
         gb_samples_free(&us);
         gb_result_end_metrics(&r);
         gb_result_check(&r, &chk);
         gb_result_finish(&r, chk.valid ? "ok" : "fail", "us_per_submit",
                          chk.valid ? NULL : "output does not match the reference");
      }
      gb_pass_free(&u);
   }
   free(ref); glDeleteBuffers(1, &ssbo); glDeleteProgram(prog);
   return 0;
}

/* ---------------- 4. compute.image (GL 4.3 imageLoad + imageStore) ---------------- */

static const char *GL_IMG_CS =
   "#version 430 core\n"
   "layout(local_size_x=16, local_size_y=16) in;\n"
   "layout(rgba8, binding=0) readonly uniform image2D src;\n"
   "layout(rgba8, binding=1) writeonly uniform image2D dst;\n"
   "uniform int u_w, u_h;\n"
   "void main() {\n"
   "  ivec2 p = ivec2(gl_GlobalInvocationID.xy); if (p.x >= u_w || p.y >= u_h) return;\n"
   "  vec4 c = imageLoad(src, p) * 0.5 + imageLoad(src, ivec2(min(p.x+1,u_w-1), p.y)) * 0.25 + imageLoad(src, ivec2(p.x, min(p.y+1,u_h-1))) * 0.25;\n"
   "  imageStore(dst, p, vec4(c.rgb, 1.0));\n"
   "}\n";

struct gl_img_st { GLuint prog, t_in, t_out; unsigned w, h; };

static void
gl_img_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl_img_st *s = u;
   glUseProgram(s->prog);
   glBindImageTexture(0, s->t_in, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
   glBindImageTexture(1, s->t_out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
   glDispatchCompute((s->w + 15u) / 16u, (s->h + 15u) / 16u, 1);
   glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

static int
run_gl_image(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"res", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   unsigned w = 0, h = 0;
   if (!gl_parse_res(gb_param(o, "res", "1080p"), &w, &h)) gb_die("bad res");
   if (ci->es || !ci->compute) { emit_skip(ci, o, 1, "requires desktop GL 4.3 compute"); return 0; }

   struct gl_img_st s = {.w = w, .h = h};
   s.prog = gl_compile_prog(NULL, NULL, GL_IMG_CS);
   glUseProgram(s.prog);
   glUniform1i(glGetUniformLocation(s.prog, "u_w"), (GLint)w);
   glUniform1i(glGetUniformLocation(s.prog, "u_h"), (GLint)h);
   uint32_t *init = malloc((size_t)w * h * 4u);
   gb_conv2d_fill_input(init, w, h);
   glGenTextures(1, &s.t_in); glBindTexture(GL_TEXTURE_2D, s.t_in);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, init);
   free(init);
   glGenTextures(1, &s.t_out); glBindTexture(GL_TEXTURE_2D, s.t_out);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
   glFinish(); gl_die_err("compute.image setup");

   const int defd[2] = {1, 4};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   for (int di = 0; di < nd; di++) {
      struct gb_pass u = {0};
      gl_run_stream(depths[di], o->warmup_ms, o->runs, o->run_ms, gl_img_rec, &s, false, &u);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depths[di], NULL);
      gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (u.failed || !u.submits) gb_result_finish(&r, "fail", NULL, "GL error or no completions");
      else {
         gb_result_stream_metrics(&r, &u, NULL);
         struct gb_samples us = {0}; gb_pass_rate(&u, &us);
         gb_derive_rate(&r, "mpix_s", "Mpix/s", &us, (double)w * h);
         gb_samples_free(&us);
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      gb_pass_free(&u);
   }
   glDeleteTextures(1, &s.t_in); glDeleteTextures(1, &s.t_out); glDeleteProgram(s.prog);
   return 0;
}

/* ---------------- 5. compile.pso (GLSL compile + link) ---------------- */

static int
run_gl_compile(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "count", "salt", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "uber");
   bool is_gfx = !strcmp(kind, "graphics");
   bool is_uber = !strcmp(kind, "uber");
   if (!is_gfx && !is_uber && strcmp(kind, "compute")) gb_die("kind must be compute, graphics or uber");
   long count = gb_param_int(o, "count", (long)GB_PSO_DEFAULT_COUNT);
   long salt = gb_param_int(o, "salt", 1);
   if (count < 1 || count > 256) gb_die("count 1..256");
   if (ci->es || (!is_gfx && !ci->compute)) { emit_skip(ci, o, 1, "requires desktop GL 4.3"); return 0; }

   uint32_t nonce = salt ? (uint32_t)((gb_now_ns() ^ ((uint64_t)getpid() << 16)) | 1u) : 0x51a71234u;
   struct gb_samples cold_s = {0};
   uint64_t tb0 = gb_now_ns();
   char src[2048];
   for (long i = 0; i < count; i++) {
      uint32_t vid = (uint32_t)(i + 1), nn = nonce + (uint32_t)i * 2654435761u;
      uint32_t ln = is_gfx ? (4u + (uint32_t)(i & 7)) : (8u + (uint32_t)(i & 15));
      uint64_t t0 = gb_now_ns();
      GLuint p = 0;
      if (is_gfx) {
         snprintf(src, sizeof(src),
            "#version 330 core\nlayout(location=0) out vec4 o;\n"
            "void main() { vec3 n=normalize(vec3(gl_FragCoord.xy*0.01,0.8)); vec3 c=vec3(0.1);\n"
            "  float ph=float((%uu^%uu)&1023u)*0.006;\n"
            "  for(uint i=0u;i<%uu;i++){float a=ph+float(i)*0.7; vec3 l=normalize(vec3(sin(a),cos(a),0.7)); c+=vec3(max(dot(n,l),0.0))*0.2;}\n"
            "  o=vec4(clamp(c,0.0,1.0),1.0); }\n", nn, vid * 17u, ln);
         p = gl_compile_prog(GL_FS_VS, src, NULL);
      } else {
         snprintf(src, sizeof(src),
            "#version 430 core\nlayout(local_size_x=64) in;\nlayout(std430,binding=0) buffer Out{uint o[];};\n"
            "void main(){ uint id=gl_GlobalInvocationID.x; uvec4 v0=uvec4(id,id+1u,id+2u,id+3u)^uvec4(%uu); uvec4 v1=v0*uvec4(747796405u)+uvec4(%uu);\n"
            "  for(uint i=0u;i<%uu;i++){ if(((id+i+%uu)&%uu)==0u) v0+=v1^(v0>>uvec4(3u)); else v1=(v1^v0)*uvec4(16777619u)+uvec4(%uu); }\n"
            "  o[id]=v0.x^v1.y^v0.z^v1.w; }\n", nn, vid, ln, vid, is_uber ? 7u : 3u, nn);
         p = gl_compile_prog(NULL, NULL, src);
      }
      uint64_t t1 = gb_now_ns();
      glDeleteProgram(p);
      gb_samples_push(&cold_s, (double)(t1 - t0) / 1e6);
   }
   double cold_total = (double)(gb_now_ns() - tb0) / 1e6;

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
   gb_result_metric(&r, "cold_per_pso_ms", "ms", true, &cold_s, true);
   gb_result_metric1(&r, "cold_total_ms", "ms", true, cold_total);
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts"); gb_json_int(&r.j, "pipelines", count); gb_json_obj_end(&r.j);
   gb_result_finish(&r, "ok", "cold_per_pso_ms", NULL);
   gb_samples_free(&cold_s);
   return 0;
}

/* ---------------- 6. sync.finish (fence vs glFinish) ---------------- */

struct gl_sync_st { GLuint prog, fbo, tex, vao; };

static void
gl_sync_rec(void *u, uint64_t k)
{
   (void)k;
   struct gl_sync_st *s = u;
   glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
   glViewport(0, 0, 64, 64);
   glUseProgram(s->prog);
   glBindVertexArray(s->vao);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static int
run_gl_sync(const struct gl_ctx_info *ci, const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "finish");
   bool use_finish = !strcmp(kind, "finish");
   if (!use_finish && strcmp(kind, "fence")) gb_die("kind must be fence or finish");
   if (ci->es) { emit_skip(ci, o, 1, "desktop GL only"); return 0; }

   struct gl_sync_st s = {0};
   s.prog = gl_compile_prog(GL_FS_VS, GL_CMP_W1_FS, NULL);
   glGenVertexArrays(1, &s.vao);
   glGenFramebuffers(1, &s.fbo); glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
   glGenTextures(1, &s.tex); glBindTexture(GL_TEXTURE_2D, s.tex);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 64);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.tex, 0);
   glFinish();

   struct gb_pass u = {0};
   gl_run_stream(1, o->warmup_ms, o->runs, o->run_ms, gl_sync_rec, &s, use_finish, &u);
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_json_obj_begin(&r.j, "executor"); ci->emit_identity(&r.j, ci->ctx); gb_json_obj_end(&r.j);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (u.failed || !u.submits) gb_result_finish(&r, "fail", NULL, "GL error or no completions");
   else {
      gb_result_stream_metrics(&r, &u, NULL);
      gb_result_finish(&r, "ok", "us_per_submit", NULL);
   }
   gb_pass_free(&u);
   glDeleteProgram(s.prog); glDeleteTextures(1, &s.tex);
   glDeleteFramebuffers(1, &s.fbo); glDeleteVertexArrays(1, &s.vao);
   return 0;
}

bool
gb_gl_is_wave1(const char *test)
{
   return test && (!strcmp(test, "compress.bw") || !strcmp(test, "render.draws") ||
                   !strcmp(test, "compute.uber") || !strcmp(test, "compute.image") ||
                   !strcmp(test, "compile.pso") || !strcmp(test, "sync.finish"));
}

int
gb_gl_run_wave1(const struct gl_ctx_info *ci, const struct gb_opts *o, bool do_verify)
{
   (void)do_verify;
   if (!strcmp(o->test, "compress.bw")) return run_gl_compress(ci, o);
   if (!strcmp(o->test, "render.draws")) return run_gl_draws(ci, o);
   if (!strcmp(o->test, "compute.uber")) return run_gl_uber(ci, o);
   if (!strcmp(o->test, "compute.image")) return run_gl_image(ci, o);
   if (!strcmp(o->test, "compile.pso")) return run_gl_compile(ci, o);
   if (!strcmp(o->test, "sync.finish")) return run_gl_sync(ci, o);
   return 1;
}
