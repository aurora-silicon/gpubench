// SPDX-License-Identifier: GPL-2.0-only
/* Headless GL/GLES executor. See README.md for API and timing limits. */
#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "gb_gl_work.h"
#include "shaders.h"

struct context {
   EGLDisplay display;
   EGLContext context;
   EGLSurface surface;
   bool es, compute, fma, timer;
};
struct assets {
   GLuint program, vao, fbo, target, tex, buffers[2];
   GLint color_location, value_location;
};

static void
check_gl(void)
{
   GLenum error = glGetError();
   if (error != GL_NO_ERROR) gb_die("GL error 0x%x", error);
}

static void
create_context(struct context *c, bool es)
{
   memset(c, 0, sizeof(*c));
   c->es = es;
   const char *ext = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
   if (ext && strstr(ext, "EGL_MESA_platform_surfaceless"))
      c->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   else c->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (c->display == EGL_NO_DISPLAY || !eglInitialize(c->display, NULL, NULL))
      gb_die("EGL initialization failed: 0x%x", eglGetError());
   if (!eglBindAPI(es ? EGL_OPENGL_ES_API : EGL_OPENGL_API)) gb_die("EGL API unavailable");
   EGLint attrs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
      es ? EGL_OPENGL_ES3_BIT : EGL_OPENGL_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
   EGLConfig config;
   EGLint n;
   if (!eglChooseConfig(c->display, attrs, &config, 1, &n) || !n) gb_die("no EGL RGBA8 config");
   for (int attempt = 0; attempt < 2; attempt++) {
      EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION, es ? 3 : (attempt ? 3 : 4),
         EGL_CONTEXT_MINOR_VERSION, es ? (attempt ? 0 : 1) : 3,
         EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
      if (es) ca[4] = EGL_NONE;
      c->context = eglCreateContext(c->display, config, EGL_NO_CONTEXT, ca);
      if (c->context != EGL_NO_CONTEXT) break;
   }
   if (c->context == EGL_NO_CONTEXT) gb_die("need desktop GL 3.3 or GLES 3.0");
   EGLint pa[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   c->surface = eglCreatePbufferSurface(c->display, config, pa);
   if (c->surface == EGL_NO_SURFACE ||
       !eglMakeCurrent(c->display, c->surface, c->surface, c->context)) gb_die("EGL make-current failed");
   c->compute = epoxy_gl_version() >= (es ? 31 : 43);
   c->fma = !es || epoxy_gl_version() >= 32 || epoxy_has_gl_extension("GL_EXT_gpu_shader5");
   c->timer = !es || epoxy_has_gl_extension("GL_EXT_disjoint_timer_query");
   glDisable(GL_DITHER);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   check_gl();
}

static void
destroy_context(struct context *c)
{
   eglMakeCurrent(c->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroySurface(c->display, c->surface);
   eglDestroyContext(c->display, c->context);
   eglTerminate(c->display);
}

static GLuint
shader(const struct context *c, GLenum stage, const char *body)
{
   const char *version = c->es ? (stage == GL_COMPUTE_SHADER ? (epoxy_gl_version() >= 32 ? "#version 320 es\n" : "#version 310 es\n") : "#version 300 es\n") :
                                 (stage == GL_COMPUTE_SHADER ? "#version 430 core\n" : "#version 330 core\n");
   const char *parts[] = {version,
      c->es && stage == GL_COMPUTE_SHADER && body == alu_comp && epoxy_gl_version() < 32 ? "#extension GL_EXT_gpu_shader5 : require\n" : "",
      c->es ? "precision highp float;\nprecision highp int;\nprecision highp sampler2D;\n" : "", body};
   GLuint s = glCreateShader(stage);
   glShaderSource(s, 4, parts, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[4096]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
      gb_die("shader compilation: %s", log);
   }
   return s;
}

static void
setup(const struct context *c, const struct gb_gl_work *w, struct assets *a)
{
   memset(a, 0, sizeof(*a));
   a->program = glCreateProgram();
   GLuint s[2] = {0};
   if (w->compute) {
      s[0] = shader(c, GL_COMPUTE_SHADER, w->kind == GB_GL_LIGHT ? light_comp : w->kind == GB_GL_ALU ? alu_comp : bw_comp);
   } else {
      s[0] = shader(c, GL_VERTEX_SHADER, w->kind == GB_GL_GEOM ? geom_vert : fs_vert);
      s[1] = shader(c, GL_FRAGMENT_SHADER, w->kind == GB_GL_TEXTURE ? texture_frag : color_frag);
   }
   glAttachShader(a->program, s[0]);
   if (s[1]) glAttachShader(a->program, s[1]);
   glLinkProgram(a->program);
   GLint ok;
   glGetProgramiv(a->program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[4096]; glGetProgramInfoLog(a->program, sizeof(log), NULL, log);
      gb_die("program link: %s", log);
   }
   glDeleteShader(s[0]);
   if (s[1]) glDeleteShader(s[1]);
   glUseProgram(a->program);
   a->color_location = glGetUniformLocation(a->program, "pc.color");
   a->value_location = glGetUniformLocation(a->program, "pc.v");
   if (w->compute) {
      glGenBuffers(2, a->buffers);
      for (int i = 0; i < (w->kind == GB_GL_BW ? 2 : 1); i++) {
         size_t size = w->kind == GB_GL_LIGHT ? 4096 : w->kind == GB_GL_ALU ? GB_ALU_GROUPS * GB_ALU_LOCAL * 4 : GB_BW_BYTES;
         glBindBuffer(GL_SHADER_STORAGE_BUFFER, a->buffers[i]);
         glBufferData(GL_SHADER_STORAGE_BUFFER, size, NULL, GL_DYNAMIC_COPY);
         glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, a->buffers[i]);
         /* Define input outside timing, also making bandwidth readback deterministic. */
         if (w->kind == GB_GL_BW && i == 0) {
            float *data = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
            if (!data) gb_die("buffer initialization map failed");
            for (size_t j = 0; j < size / sizeof(float); j++) data[j] = (float)(j & 255);
            if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)) gb_die("buffer initialization invalidated");
         }
      }
      glUniform1ui(glGetUniformLocation(a->program, "pc.iters"), GB_ALU_ITERS);
      glUniform1f(glGetUniformLocation(a->program, "pc.a"), GB_ALU_A);
      glUniform1f(glGetUniformLocation(a->program, "pc.b"), GB_ALU_B);
   } else {
      glGenVertexArrays(1, &a->vao); glBindVertexArray(a->vao);
      glGenFramebuffers(1, &a->fbo); glBindFramebuffer(GL_FRAMEBUFFER, a->fbo);
      glGenTextures(1, &a->target); glBindTexture(GL_TEXTURE_2D, a->target);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w->w, w->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, a->target, 0);
      if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) gb_die("incomplete RGBA8 framebuffer");
      glViewport(0, 0, w->w, w->h);
      if (w->kind == GB_GL_FILL) {
         glEnable(GL_BLEND);
         glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      } else glDisable(GL_BLEND);
      if (w->kind == GB_GL_TEXTURE) {
         unsigned char data[GB_TEX_DIM * GB_TEX_DIM * 4];
         for (unsigned y = 0; y < GB_TEX_DIM; y++)
            for (unsigned x = 0; x < GB_TEX_DIM; x++) {
               unsigned i = (y * GB_TEX_DIM + x) * 4;
               data[i] = x; data[i+1] = y; data[i+2] = x ^ y; data[i+3] = 255;
            }
         glGenTextures(1, &a->tex); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, a->tex);
         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GB_TEX_DIM, GB_TEX_DIM, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
         glUniform1i(glGetUniformLocation(a->program, "tex"), 0);
      }
   }
   glFinish(); check_gl();
}

static void
record(const struct gb_gl_work *w, const struct assets *a, uint64_t k)
{
   if (w->compute) {
      if (w->kind == GB_GL_LIGHT) glUniform1ui(a->value_location, (GLuint)k);
      GLuint groups = w->kind == GB_GL_LIGHT ? 1 : w->kind == GB_GL_ALU ? GB_ALU_GROUPS : GB_BW_BYTES / 16 / GB_BW_LOCAL;
      glDispatchCompute(groups, 1, 1);
   } else {
      glClearColor(0.1f, 0.2f, 0.3f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
      if (w->kind == GB_GL_CLEAR) return;
      glUniform4f(a->color_location, (float)(k & 255) / 255.0f, 0.5f, 0.25f, 0.5f);
      if (w->kind == GB_GL_GEOM) glDrawArrays(GL_TRIANGLES, 0, 3 * GB_GEOM_TRIS);
      else glDrawArraysInstanced(GL_TRIANGLES, 0, 3, w->kind == GB_GL_TRI ? 1 : GB_FILL_LAYERS);
   }
}

/* Verification happens outside all timing windows. */
static void
verify(const struct gb_gl_work *w, const struct assets *a)
{
   record(w, a, 255);
   if (w->compute) glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
   glFinish();
   if (w->compute) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, a->buffers[w->kind == GB_GL_BW ? 1 : 0]);
      size_t size = w->kind == GB_GL_LIGHT ? GB_LIGHT_LOCAL * 4 : w->kind == GB_GL_ALU ? GB_ALU_GROUPS * GB_ALU_LOCAL * 4 : GB_BW_BYTES;
      void *data = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
      if (!data) gb_die("verification map failed");
      for (size_t i = 0; i < size / 4; i += w->kind == GB_GL_ALU ? (GB_ALU_LOCAL - 1) : 1) {
         if (w->kind == GB_GL_LIGHT) {
            if (((unsigned *)data)[i] != 255 + i) gb_die("compute.light readback mismatch");
         } else {
            float expected = (float)(i & 255);
            if (w->kind == GB_GL_ALU) {
               float x[8];
               for (int j = 0; j < 8; j++) x[j] = (float)i * 1e-6f + j;
               for (unsigned step = 0; step < GB_ALU_ITERS * 4; step++)
                  for (int j = 0; j < 8; j++) x[j] = fmaf(x[j], GB_ALU_A, GB_ALU_B);
               expected = ((x[0]+x[1])+(x[2]+x[3]))+((x[4]+x[5])+(x[6]+x[7]));
            }
            if (fabsf(((float *)data)[i] - expected) > 0.001f) gb_die("compute readback mismatch at %zu", i);
         }
      }
      if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)) gb_die("verification map invalidated");
   } else {
      unsigned char pixel[4];
      /* Bottom-left sample is covered by the geometry grid's first triangle. */
      glReadPixels(w->kind == GB_GL_TEXTURE ? 13 : 0, w->kind == GB_GL_TEXTURE ? 29 : 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
      const float clear[] = {0.1f, 0.2f, 0.3f, 1.0f}, color[] = {1, 0.5f, 0.25f, 0.5f};
      for (int i = 0; i < 4; i++) {
         float expected = w->kind == GB_GL_CLEAR ? clear[i] : w->kind == GB_GL_TEXTURE && i < 3 ? 0 : color[i];
         if (w->kind == GB_GL_TEXTURE && i < 3) expected = (i == 0 ? 13.0f : i == 1 ? 29.0f : (13 ^ 29)) / 255.0f * color[i];
         if (w->kind == GB_GL_FILL) expected = i == 3 ? 1 : color[i] + (clear[i] - color[i]) / 65536.0f;
         if (abs((int)pixel[i] - (int)lroundf(expected * 255)) > 2) gb_die("render readback mismatch channel %d: %u", i, pixel[i]);
      }
   }
   check_gl();
}

static void
cleanup(struct assets *a)
{
   glDeleteProgram(a->program); glDeleteBuffers(2, a->buffers);
   glDeleteTextures(1, &a->tex); glDeleteTextures(1, &a->target);
   glDeleteFramebuffers(1, &a->fbo); glDeleteVertexArrays(1, &a->vao);
}

static void
stamp(const struct context *c, GLuint q)
{
   if (c->es) glQueryCounterEXT(q, GL_TIMESTAMP_EXT);
   else glQueryCounter(q, GL_TIMESTAMP);
}
static GLuint64
query(const struct context *c, GLuint q)
{
   GLuint64 value;
   if (c->es) glGetQueryObjectui64vEXT(q, GL_QUERY_RESULT_EXT, &value);
   else glGetQueryObjectui64v(q, GL_QUERY_RESULT, &value);
   return value;
}

static void
run_pass(const struct context *c, const struct gb_gl_work *w, const struct assets *a,
         int depth, bool timed, int warm_ms, int runs, int run_ms, struct gb_pass *p)
{
   struct slot { GLsync fence; uint64_t submit; GLuint q[2]; } slots[64] = {0};
   memset(p, 0, sizeof(*p));
   p->nwin = runs; p->win_s = run_ms / 1e3; p->timed = timed;
   if (timed) {
      for (int i = 0; i < depth; i++) {
         if (c->es) glGenQueriesEXT(2, slots[i].q);
         else glGenQueries(2, slots[i].q);
      }
      if (c->es) { GLint disjoint; glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint); }
   }
   const uint64_t start = gb_now_ns() + (uint64_t)warm_ms * 1000000;
   const uint64_t win = (uint64_t)run_ms * 1000000, end = start + (uint64_t)runs * win;
   uint64_t sent = 0, done = 0, last_end = 0;
   while (true) {
      uint64_t now = gb_now_ns();
      bool full = sent - done >= (unsigned)depth;
      if (!full && now >= end) break;
      if (full) {
         struct slot *s = &slots[done % depth];
         GLenum status = glClientWaitSync(s->fence, 0, 10000000000ull);
         if (status != GL_ALREADY_SIGNALED && status != GL_CONDITION_SATISFIED) { p->failed = true; break; }
         uint64_t t = gb_now_ns();
         bool measured = t >= start && t < end;
         if (measured) {
            p->win_count[(t - start) / win]++;
            gb_samples_push(&p->latency, (t - s->submit) / 1e3);
         }
         if (timed) {
            uint64_t b = query(c, s->q[1]), begin = query(c, s->q[0]);
            uint64_t effective = begin > last_end ? begin : last_end;
            if (measured) {
               gb_samples_push(&p->gpu, b > effective ? (b - effective) / 1e3 : 0);
               if (last_end) gb_samples_push(&p->gap, ((double)begin - (double)last_end) / 1e3);
            }
            last_end = b;
         }
         glDeleteSync(s->fence); s->fence = NULL; done++;
         continue;
      }
      struct slot *s = &slots[sent % depth];
      uint64_t r0 = gb_now_ns();
      if (timed) stamp(c, s->q[0]);
      record(w, a, sent + 1);
      if (timed) stamp(c, s->q[1]);
      uint64_t r1 = gb_now_ns();
      s->submit = r1;
      s->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      glFlush();
      uint64_t r2 = gb_now_ns();
      if (!s->fence) { p->failed = true; break; }
      if (r0 >= start && r0 < end) {
         gb_samples_push(&p->record_cpu, (r1 - r0) / 1e3);
         gb_samples_push(&p->submit_cpu, (r2 - r1) / 1e3);
      }
      sent++;
   }
   glFinish();
   for (int i = 0; i < depth; i++) {
      if (slots[i].fence) glDeleteSync(slots[i].fence);
      if (timed) {
         if (c->es) glDeleteQueriesEXT(2, slots[i].q);
         else glDeleteQueries(2, slots[i].q);
      }
   }
   if (timed && c->es) {
      GLint disjoint; glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
      if (disjoint) p->failed = true;
   }
   if (glGetError() != GL_NO_ERROR) p->failed = true;
   for (int i = 0; i < runs; i++) p->submits += p->win_count[i];
}

static void
identity(struct gb_json *j, const struct context *c)
{
   gb_json_str(j, "api", c->es ? "gles" : "gl");
   gb_json_str(j, "renderer", (const char *)glGetString(GL_RENDERER));
   gb_json_str(j, "vendor", (const char *)glGetString(GL_VENDOR));
   gb_json_str(j, "version", (const char *)glGetString(GL_VERSION));
   gb_json_str(j, "suite_sha", GB_SUITE_SHA);
   gb_json_str(j, "loader_driver_override", getenv("MESA_LOADER_DRIVER_OVERRIDE") ? getenv("MESA_LOADER_DRIVER_OVERRIDE") : "");
   gb_json_bool(j, "compute", c->compute);
   gb_json_bool(j, "timer_queries", c->timer);
   gb_json_bool(j, "fma", c->fma);
}

int
main(int argc, char **argv)
{
   bool es = false, do_verify = false;
   for (int i = 1; i < argc;) {
      if (!strcmp(argv[i], "--gles") || !strcmp(argv[i], "--verify")) {
         if (!strcmp(argv[i], "--gles")) es = true; else do_verify = true;
         memmove(argv + i, argv + i + 1, (argc - i) * sizeof(*argv)); argc--;
      } else i++;
   }
   struct gb_opts o; gb_parse_opts(&o, argc, argv);
   if (o.list) { puts("compute.light\ncompute.heavy\nrender.light\nrender.heavy"); return 0; }
   if (o.warmup_ms < 0 || o.run_ms <= 0) gb_die("warmup-ms must be nonnegative and run-ms positive");
   if (o.device || o.child) gb_die("GL supports the EGL default device and mode=queue only");
   struct gb_gl_work w;
   const char *error;
   if (!o.info && (!o.test || !gb_gl_work(&o, &w, &error))) gb_die("%s", o.test ? error : "need --test, --info or --list");
   struct context c; create_context(&c, es);
   if (o.info) {
      struct gb_json j; gb_json_init(&j); gb_json_obj_begin(&j, NULL);
      identity(&j, &c); gb_json_obj_end(&j); puts(j.buf); gb_json_free(&j);
      destroy_context(&c); return 0;
   }
   const int defaults[] = {1, 2, 4, 8, 16, 32, 64};
   const int *depths = o.ndepths ? o.depths : defaults;
   int count = o.ndepths ? o.ndepths : 7;
   const char *skip = o.mode != GB_MODE_QUEUE ? "GL supports mode=queue only" :
                      w.compute && !c.compute ? "compute requires desktop GL 4.3 or GLES 3.1" : NULL;
   if (!skip && w.kind == GB_GL_ALU && !c.fma)
      skip = "ALU requires GLES 3.2 or GL_EXT_gpu_shader5 for matched FMA";
   if (!skip && w.compute) {
      GLint max_local, max_invocations;
      glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &max_local);
      glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_invocations);
      int local = w.kind == GB_GL_LIGHT ? GB_LIGHT_LOCAL : GB_ALU_LOCAL;
      if (max_local < local || max_invocations < local)
         skip = "matched compute local size exceeds device limits";
   }
   struct assets a = {0};
   if (!skip) { setup(&c, &w, &a); if (do_verify) verify(&w, &a); }
   int rc = 0;
   for (int i = 0; i < count; i++) {
      int depth = depths[i];
      struct gb_result r; gb_result_begin(&r, &o, o.mode, depth, NULL);
      gb_json_obj_begin(&r.j, "executor"); identity(&r.j, &c); gb_json_obj_end(&r.j);
      if (skip || depth < 1 || depth > 64) {
         gb_result_finish(&r, "skip", NULL, skip ? skip : "depth must be 1..64"); continue;
      }
      gb_result_run_info(&r, o.warmup_ms, o.runs, o.run_ms);
      bool timed = o.timestamps && c.timer && o.timed_runs > 0;
      if (o.timestamps && !c.timer) gb_json_str(&r.j, "timed_pass_skip", "timer queries unavailable");
      struct gb_pass u = {0}, t = {0};
      run_pass(&c, &w, &a, depth, false, o.warmup_ms, o.runs, o.run_ms, &u);
      if (timed && !u.failed) run_pass(&c, &w, &a, depth, true,
                                     o.warmup_ms < 100 ? o.warmup_ms : 100, o.timed_runs, o.run_ms, &t);
      if (u.failed || t.failed || !u.submits || (timed && !t.submits)) {
         gb_result_finish(&r, "fail", NULL, "GPU wait/error, disjoint timer or no measured completions"); rc = 1;
      } else {
         gb_result_stream_metrics(&r, &u, timed ? &t : NULL);
         if (w.metric) {
            struct gb_samples us = {0}; gb_pass_rate(&u, &us);
            gb_derive_rate(&r, w.metric, w.unit, &us, w.work_e6); gb_samples_free(&us);
         }
         gb_result_end_metrics(&r);
         gb_json_obj_begin(&r.j, "counts"); gb_json_int(&r.j, "submits", u.submits); gb_json_obj_end(&r.j);
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      gb_pass_free(&u); gb_pass_free(&t);
   }
   if (!skip) cleanup(&a);
   destroy_context(&c);
   return rc;
}
