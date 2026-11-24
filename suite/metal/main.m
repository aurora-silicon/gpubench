// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-mtl: the standard suite's Metal executor. It mirrors suite/vk test
 * for test: same dimensions (gb_params.h), same stream engine semantics, same
 * result lines (gb_common). See suite/README.md. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "gb_common.h"
#include "gb_params.h"
#include "gb_wave1_ref.h"
#include "gb_wave2_ref.h"
#include "msl_sources.h"

/* ======================= context ======================= */

static id<MTLDevice> g_dev;
static id<MTLLibrary> g_lib;

static void
ctx_init(void)
{
   if (g_dev)
      return;
   g_dev = MTLCreateSystemDefaultDevice();
   if (!g_dev)
      gb_die("no Metal device");
   MTLCompileOptions *opts = [MTLCompileOptions new];
   NSError *err = nil;
   g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:gb_msl_source] options:opts error:&err];
   if (!g_lib)
      gb_die("MSL compile failed: %s", err.localizedDescription.UTF8String);
}

static id<MTLCommandQueue>
new_queue(int depth)
{
   int max = depth + 16 > 64 ? depth + 16 : 64;
   return [g_dev newCommandQueueWithMaxCommandBufferCount:max];
}

static id<MTLComputePipelineState>
cpipe(const char *fn)
{
   NSError *err = nil;
   id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
   if (!f)
      gb_die("no function %s", fn);
   id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&err];
   if (!p)
      gb_die("compute pipeline %s: %s", fn, err.localizedDescription.UTF8String);
   return p;
}

static id<MTLRenderPipelineState>
rpipe(const char *vs, const char *fs, bool blend)
{
   MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
   d.vertexFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:vs]];
   d.fragmentFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
   d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   if (blend) {
      MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
      c.blendingEnabled = YES;
      c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      c.rgbBlendOperation = MTLBlendOperationAdd;
      c.sourceAlphaBlendFactor = MTLBlendFactorOne;
      c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      c.alphaBlendOperation = MTLBlendOperationAdd;
   }
   NSError *err = nil;
   id<MTLRenderPipelineState> p = [g_dev newRenderPipelineStateWithDescriptor:d error:&err];
   if (!p)
      gb_die("render pipeline %s/%s: %s", vs, fs, err.localizedDescription.UTF8String);
   return p;
}

static const MTLResourceOptions SHARED = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
static const MTLResourceOptions PRIVATE = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;

static id<MTLBuffer>
buf(size_t size, bool host)
{
   id<MTLBuffer> b = [g_dev newBufferWithLength:size options:host ? SHARED : PRIVATE];
   if (!b)
      gb_die("buffer of %zu bytes failed", size);
   return b;
}

static id<MTLTexture>
tex(uint32_t w, uint32_t h, MTLTextureUsage usage)
{
   MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                width:w
                                                                               height:h
                                                                            mipmapped:NO];
   d.storageMode = MTLStorageModePrivate;
   d.usage = usage;
   d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
   id<MTLTexture> t = [g_dev newTextureWithDescriptor:d];
   if (!t)
      gb_die("texture %ux%u failed", w, h);
   return t;
}

static MTLRenderPassDescriptor *
pass_desc(id<MTLTexture> t)
{
   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture = t;
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.2, 0.3, 1.0);
   return rp;
}

static int
print_info(void)
{
   ctx_init();
   struct gb_json j;
   gb_json_init(&j);
   gb_json_obj_begin(&j, NULL);
   gb_json_str(&j, "schema", "gpubench-info/1");
   gb_json_str(&j, "api", "metal");
   gb_json_str(&j, "device_name", g_dev.name.UTF8String);
   gb_json_str(&j, "os_version", [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
   char build[64] = "";
   size_t bl = sizeof(build);
   sysctlbyname("kern.osversion", build, &bl, NULL, 0);
   gb_json_str(&j, "os_build", build);
   char model[64] = "";
   size_t ml = sizeof(model);
   sysctlbyname("hw.model", model, &ml, NULL, 0);
   gb_json_str(&j, "hw_model", model);
   const char *fam = "unknown";
   if (@available(macOS 14.0, *)) {
      if ([g_dev supportsFamily:MTLGPUFamilyApple9]) fam = "apple9";
      else if ([g_dev supportsFamily:MTLGPUFamilyApple8]) fam = "apple8";
      else if ([g_dev supportsFamily:MTLGPUFamilyApple7]) fam = "apple7";
   }
   gb_json_str(&j, "gpu_family", fam);
   gb_json_int(&j, "recommended_working_set_bytes", (int64_t)g_dev.recommendedMaxWorkingSetSize);
   gb_json_bool(&j, "unified_memory", g_dev.hasUnifiedMemory);
   gb_json_bool(&j, "counter_sampling_stage_boundary",
                [g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]);
   gb_json_obj_end(&j);
   puts(j.buf);
   gb_json_free(&j);
   return 0;
}

/* ======================= streams ======================= */

typedef void (^GBEncode)(id<MTLCommandBuffer> cb, uint64_t k);
typedef void (^GBDone)(uint64_t k);

@interface GBStream : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic) int depth;
@property(nonatomic, strong) NSMutableArray *cbs;
@property(nonatomic, copy) GBEncode encode;
@property(nonatomic, copy) GBDone done;
@property(nonatomic) uint64_t value;
@property(nonatomic) BOOL failed;
@property(nonatomic) uint64_t *submit_ns;
@end

@implementation GBStream
- (instancetype)initWithQueue:(id<MTLCommandQueue>)q depth:(int)d encode:(GBEncode)e done:(GBDone)dn
{
   self = [super init];
   _queue = q;
   _depth = d;
   _encode = e;
   _done = dn;
   _cbs = [NSMutableArray arrayWithCapacity:d];
   for (int i = 0; i < d; i++)
      [_cbs addObject:[NSNull null]];
   _submit_ns = calloc(d, sizeof(uint64_t));
   return self;
}
- (void)dealloc
{
   free(_submit_ns);
}
@end

static BOOL
cb_ok(id<MTLCommandBuffer> cb)
{
   if (cb.status == MTLCommandBufferStatusError) {
      gb_log("command buffer error: %s", cb.error.localizedDescription.UTF8String);
      return NO;
   }
   return YES;
}

static id<MTLCommandBuffer>
new_cb(id<MTLCommandQueue> q)
{
   return [q commandBufferWithUnretainedReferences];
}

static double
stream_once(GBStream *s)
{
   uint64_t v = s.value + 1;
   double us;
   @autoreleasepool {
      id<MTLCommandBuffer> cb = new_cb(s.queue);
      s.encode(cb, v);
      uint64_t t0 = gb_now_ns();
      [cb commit];
      [cb waitUntilCompleted];
      uint64_t t1 = gb_now_ns();
      if (!cb_ok(cb))
         return -1;
      us = (t1 - t0) / 1e3;
   }
   s.value = v;
   if (s.done)
      s.done(v);
   return us;
}

static int
stream_run(GBStream *s, bool timed, uint64_t t0, uint64_t warmup_ns, int nwin, uint64_t win_ns, struct gb_pass *out)
{
   memset(out, 0, sizeof(*out));
   out->nwin = nwin;
   out->win_s = win_ns / 1e9;
   out->timed = timed;
   const uint64_t tm0 = t0 + warmup_ns, t_end = tm0 + (uint64_t)nwin * win_ns;
   const uint64_t base = s.value;
   const int depth = s.depth;
   uint64_t k = 0, done = 0;
   double last_end = 0;
   bool have_last = false;
   NSMutableArray *cbs = s.cbs;
   uint64_t *submit_ns = s.submit_ns;

   gb_sleep_until_ns(t0);
   for (;;) {
      @autoreleasepool {
         uint64_t now = gb_now_ns();
         bool full = k - done >= (uint64_t)depth;
         if (!full && now >= t_end)
            break;
         if (full) {
            int slot = done % depth;
            id<MTLCommandBuffer> cb = cbs[slot];
            [cb waitUntilCompleted];
            uint64_t t = gb_now_ns();
            if (!cb_ok(cb)) {
               s.failed = YES;
               break;
            }
            bool measured = t >= tm0 && t < t_end;
            if (measured) {
               gb_pass_count(out, t, tm0, win_ns);
               gb_samples_push(&out->latency, (t - submit_ns[slot]) / 1e3);
            }
            if (timed) {
               double a = cb.GPUStartTime, b = cb.GPUEndTime;
               if (measured) {
                  gb_samples_push(&out->gpu, (b - a) * 1e6);
                  if (have_last)
                     gb_samples_push(&out->gap, (a - last_end) * 1e6);
               }
               last_end = b;
               have_last = true;
            }
            cbs[slot] = [NSNull null];
            if (s.done)
               s.done(base + done + 1);
            done++;
            continue;
         }
         int slot = k % depth;
         uint64_t r0 = gb_now_ns();
         id<MTLCommandBuffer> cb = new_cb(s.queue);
         s.encode(cb, base + k + 1);
         uint64_t r1 = gb_now_ns();
         submit_ns[slot] = r1;
         [cb commit];
         uint64_t r2 = gb_now_ns();
         cbs[slot] = cb;
         if (r0 >= tm0 && r0 < t_end) {
            gb_samples_push(&out->record_cpu, (r1 - r0) / 1e3);
            gb_samples_push(&out->submit_cpu, (r2 - r1) / 1e3);
         }
         k++;
      }
   }
   /* drain (not counted) */
   for (; done < k; done++) {
      @autoreleasepool {
         id<MTLCommandBuffer> cb = cbs[done % depth];
         if ([cb isKindOfClass:[NSNull class]])
            continue;
         [cb waitUntilCompleted];
         cbs[done % depth] = [NSNull null];
         if (s.done)
            s.done(base + done + 1);
      }
   }
   s.value = base + k;
   for (int i = 0; i < nwin; i++)
      out->submits += out->win_count[i];
   out->failed = s.failed;
   return s.failed ? -1 : 0;
}

/* ======================= stream-test driver (mirrors gb_vk.c) ======================= */

struct mtest {
   const char *name;
   const char *const *known;
   int depths[GB_MAX_DEPTHS];
   int nd;
   bool report_first, no_timed;
   id (*setup)(const struct gb_opts *o, char *err, size_t n);
   void (*encode)(id state, id<MTLCommandBuffer> cb, uint64_t k);
   void (*derive)(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us);
   void (*check)(id state, struct gb_check *chk);
};

static GBEncode
make_encode(const struct mtest *t, id st)
{
   void (*fn)(id, id<MTLCommandBuffer>, uint64_t) = t->encode;
   return ^(id<MTLCommandBuffer> cb, uint64_t k) { fn(st, cb, k); };
}

static uint64_t
timed_warm(const struct gb_opts *o)
{
   uint64_t w = (uint64_t)o->warmup_ms * 1000000ull;
   return w < 100000000ull ? w : 100000000ull;
}

struct thr {
   void *s; /* GBStream, bridged */
   bool timed;
   uint64_t t0, warm, win;
   int nwin;
   struct gb_pass pass;
   pthread_t th;
};

static void *
thr_main(void *p)
{
   struct thr *t = p;
   @autoreleasepool {
      stream_run((__bridge GBStream *)t->s, t->timed, t->t0, t->warm, t->nwin, t->win, &t->pass);
   }
   return NULL;
}

static void
run_threads(NSArray *streams, bool timed, uint64_t warm, int nwin, uint64_t win, struct gb_pass *out)
{
   int n = (int)streams.count;
   struct thr *t = calloc(n, sizeof(*t));
   uint64_t t0 = gb_now_ns() + 20000000ull;
   for (int i = 0; i < n; i++) {
      t[i] = (struct thr){.s = (__bridge void *)streams[i], .timed = timed, .t0 = t0, .warm = warm,
                          .win = win, .nwin = nwin};
      pthread_create(&t[i].th, NULL, thr_main, &t[i]);
   }
   memset(out, 0, sizeof(*out));
   for (int i = 0; i < n; i++) {
      pthread_join(t[i].th, NULL);
      gb_pass_merge(out, &t[i].pass);
      gb_pass_free(&t[i].pass);
   }
   free(t);
}

static void
finish_result(const struct gb_opts *o, const struct mtest *t, enum gb_mode mode, int depth, struct gb_pass *u,
              struct gb_pass *ti, double first_us, int units, const struct gb_check *chk)
{
   struct gb_result r;
   gb_result_begin(&r, o, mode, depth, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (u->failed || (ti && ti->failed)) {
      gb_result_finish(&r, "fail", NULL, "command buffer error");
      return;
   }
   if (!u->submits) {
      gb_result_finish(&r, "fail", NULL, "no completions in the measured windows");
      return;
   }
   gb_result_stream_metrics(&r, u, ti);
   if (first_us >= 0)
      gb_result_metric1(&r, "first_us", "us", true, first_us);
   if (t->derive) {
      struct gb_samples us = {0};
      gb_pass_rate(u, &us);
      t->derive(o, &r, &us);
      gb_samples_free(&us);
   }
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "submits", (int64_t)u->submits);
   if (mode != GB_MODE_QUEUE)
      gb_json_int(&r.j, mode == GB_MODE_PROCS ? "procs" : "queues", units);
   gb_json_obj_end(&r.j);
   if (chk && chk->done) {
      gb_result_check(&r, chk);
      if (!chk->valid) {
         gb_result_finish(&r, "fail", NULL, "output does not match the reference");
         return;
      }
   }
   gb_result_finish(&r, "ok", "us_per_submit", NULL);
}

static int
run_child(const struct gb_opts *o, const struct mtest *t)
{
   ctx_init();
   char err[256] = "";
   id st = t->setup(o, err, sizeof(err));
   if (!st) {
      gb_log("child setup failed: %s", err);
      return 1;
   }
   GBStream *s = [[GBStream alloc] initWithQueue:new_queue(1) depth:1 encode:make_encode(t, st) done:nil];
   struct gb_pass prime;
   stream_run(s, false, gb_now_ns(), 0, 1, 30000000ull, &prime);
   gb_pass_free(&prime);
   if (t->check) {
      struct gb_check chk = {0};
      t->check(st, &chk);
      if (chk.done && !chk.valid) {
         gb_log("child output does not match the reference");
         return 1;
      }
   }
   gb_child_ready();
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   for (int pass = 0;; pass++) {
      uint64_t t0 = gb_child_recv_u64();
      if (!t0)
         break;
      bool timed = pass == 1;
      struct gb_pass p;
      stream_run(s, timed, t0, timed ? timed_warm(o) : W, timed ? o->timed_runs : o->runs, R, &p);
      gb_pass_put(1, &p);
      gb_pass_free(&p);
   }
   return 0;
}

static int
run_stream_test(const struct gb_opts *o, const struct mtest *t)
{
   if (!gb_check_params(o, t->known)) {
      gb_result_simple(o, o->mode, 0, NULL, "fail", "unknown parameter");
      return 1;
   }
   if (o->child)
      return run_child(o, t);
   const int *depths = o->ndepths ? o->depths : t->depths;
   int nd = o->ndepths ? o->ndepths : t->nd;
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   const bool do_timed = o->timestamps && !t->no_timed;

   if (o->mode == GB_MODE_PROCS) {
      for (int di = 0; di < nd; di++) {
         int n = depths[di];
         struct gb_child *ch = calloc(n, sizeof(*ch));
         if (gb_spawn_children(o, n, ch)) {
            gb_result_simple(o, GB_MODE_PROCS, n, NULL, "fail", "child process failed to start or set up");
            for (int i = 0; i < n; i++)
               if (ch[i].pid)
                  gb_child_reap(&ch[i]);
            free(ch);
            continue;
         }
         struct gb_pass u = {0}, ti = {0};
         bool bad = false;
         for (int pass = 0; pass < (do_timed ? 2 : 1); pass++) {
            uint64_t t0 = gb_now_ns() + GB_SYNC_START_NS;
            for (int i = 0; i < n; i++)
               gb_child_send_u64(&ch[i], t0);
            for (int i = 0; i < n; i++) {
               struct gb_pass p;
               if (gb_pass_get(ch[i].from_child, &p)) {
                  bad = true;
                  continue;
               }
               gb_pass_merge(pass ? &ti : &u, &p);
               gb_pass_free(&p);
            }
         }
         for (int i = 0; i < n; i++) {
            gb_child_send_u64(&ch[i], 0);
            gb_child_reap(&ch[i]);
         }
         if (bad)
            u.failed = true;
         finish_result(o, t, GB_MODE_PROCS, n, &u, do_timed ? &ti : NULL, -1, n, NULL);
         gb_pass_free(&u);
         gb_pass_free(&ti);
         free(ch);
      }
      return 0;
   }

   ctx_init();
   char err[256] = "";
   id st = t->setup(o, err, sizeof(err));
   if (!st) {
      for (int i = 0; i < nd; i++)
         gb_result_simple(o, o->mode, depths[i], NULL, "skip", err[0] ? err : "setup failed");
      return 0;
   }
   GBEncode enc = make_encode(t, st);
   bool first = t->report_first;
   for (int di = 0; di < nd; di++) {
      @autoreleasepool {
         int n = depths[di];
         struct gb_pass u = {0}, ti = {0};
         double first_us = -1;
         if (o->mode == GB_MODE_QUEUE) {
            GBStream *s = [[GBStream alloc] initWithQueue:new_queue(n) depth:n encode:enc done:nil];
            if (first) {
               first_us = stream_once(s);
               first = false;
            }
            stream_run(s, false, gb_now_ns(), W, o->runs, R, &u);
            if (do_timed && !s.failed)
               stream_run(s, true, gb_now_ns(), timed_warm(o), o->timed_runs, R, &ti);
         } else {
            NSMutableArray *ss = [NSMutableArray array];
            for (int i = 0; i < n; i++)
               [ss addObject:[[GBStream alloc] initWithQueue:new_queue(1) depth:1 encode:enc done:nil]];
            run_threads(ss, false, W, o->runs, R, &u);
            if (do_timed && !u.failed)
               run_threads(ss, true, timed_warm(o), o->timed_runs, R, &ti);
         }
         struct gb_check chk = {0};
         if (t->check && !u.failed && !ti.failed)
            t->check(st, &chk);
         finish_result(o, t, o->mode, n, &u, do_timed ? &ti : NULL, first_us, n, &chk);
         gb_pass_free(&u);
         gb_pass_free(&ti);
      }
   }
   return 0;
}

/* ======================= tests ======================= */

static const char *const no_params[] = {NULL};
static const char *const kind_params[] = {"kind", NULL};
static const char *const rh_params[] = {"kind", "res", NULL};
static const char *const tvb_params[] = {"tris", NULL};
#define DEPTHS_ALL {1, 2, 4, 8, 16, 32, 64}, 7

static bool
parse_res(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

/* ---- state objects ---- */

@interface GBState : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> cp;
@property(nonatomic, strong) id<MTLRenderPipelineState> rp;
@property(nonatomic, strong) id<MTLBuffer> a, b;
@property(nonatomic, strong) id<MTLTexture> t;
@property(nonatomic) int kind;
@property(nonatomic) uint32_t n;
@end
@implementation GBState
@end

enum { K_ALU, K_BW, K_TRI, K_CLEAR, K_FILL, K_GEOM, K_TVB, K_B2B, K_B2I, K_FILLBUF };

static void
encode_light(GBState *st, id<MTLCommandBuffer> cb, uint64_t k)
{
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   uint32_t v = (uint32_t)k;
   [e setComputePipelineState:st.cp];
   [e setBuffer:st.a offset:0 atIndex:0];
   [e setBytes:&v length:4 atIndex:1];
   [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(GB_LIGHT_LOCAL, 1, 1)];
   [e endEncoding];
}

static GBState *
light_state(void)
{
   GBState *st = [GBState new];
   st.cp = cpipe("light_main");
   st.a = buf(4096, true);
   return st;
}

static void
encode_render(GBState *st, id<MTLCommandBuffer> cb, uint64_t k)
{
   id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass_desc(st.t)];
   if (st.kind != K_CLEAR) {
      float color[4] = {(float)(k & 255) / 255.0f, 0.5f, 0.25f, 0.5f};
      [e setRenderPipelineState:st.rp];
      [e setFragmentBytes:color length:16 atIndex:0];
      switch (st.kind) {
      case K_TRI: [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; break;
      case K_FILL:
         [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:GB_FILL_LAYERS];
         break;
      case K_GEOM: [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 * GB_GEOM_TRIS]; break;
      case K_TVB: [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 * st.n]; break;
      default: break;
      }
   }
   [e endEncoding];
}

static GBState *
render_state(int kind, uint32_t w, uint32_t h, uint32_t tris)
{
   GBState *st = [GBState new];
   st.kind = kind;
   st.n = tris;
   st.t = tex(w, h, MTLTextureUsageRenderTarget);
   switch (kind) {
   case K_TRI: st.rp = rpipe("fs_vert", "color_frag", false); break;
   case K_FILL: st.rp = rpipe("fs_vert", "color_frag", true); break;
   case K_GEOM: st.rp = rpipe("geom_vert", "color_frag", false); break;
   case K_TVB: st.rp = rpipe("tvb_vert", "color_frag", false); break;
   default: break;
   }
   return st;
}

/* ---- submit.empty ---- */
static id
empty_setup(const struct gb_opts *o, char *e, size_t n)
{
   return [NSObject new];
}

static void
empty_encode(id st, id<MTLCommandBuffer> cb, uint64_t k)
{
}

static const struct mtest t_submit_empty = {.name = "submit.empty", .known = no_params, .depths = DEPTHS_ALL,
                                            .no_timed = true, .setup = empty_setup, .encode = empty_encode};

/* ---- compute.light ---- */
static id
cl_setup(const struct gb_opts *o, char *e, size_t n)
{
   return light_state();
}

static void
cl_encode(id st, id<MTLCommandBuffer> cb, uint64_t k)
{
   encode_light(st, cb, k);
}

static const struct mtest t_compute_light = {.name = "compute.light", .known = no_params, .depths = DEPTHS_ALL,
                                             .setup = cl_setup, .encode = cl_encode};

/* ---- compute.heavy ---- */
static void
ch_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   if (!strcmp(gb_param(o, "kind", "alu"), "bw"))
      gb_derive_rate(r, "gbps", "GB/s", us, 2.0 * GB_BW_BYTES / 1e9 * 1e6);
   else
      gb_derive_rate(r, "gflops", "GFLOP/s", us, GB_ALU_FLOP / 1e9 * 1e6);
}

static GBState *
heavy_state(const char *kind)
{
   GBState *st = [GBState new];
   if (!strcmp(kind, "alu")) {
      st.kind = K_ALU;
      st.cp = cpipe("alu_main");
      st.a = buf(GB_ALU_GROUPS * GB_ALU_LOCAL * 4, false);
   } else if (!strcmp(kind, "bw")) {
      st.kind = K_BW;
      st.cp = cpipe("bw_main");
      st.a = buf(GB_BW_BYTES, false);
      st.b = buf(GB_BW_BYTES, false);
   } else
      return nil;
   return st;
}

static void
encode_heavy(GBState *s, id<MTLCommandBuffer> cb)
{
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:s.cp];
   [e setBuffer:s.a offset:0 atIndex:0];
   if (s.kind == K_BW) {
      [e setBuffer:s.b offset:0 atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(GB_BW_BYTES / 16 / GB_BW_LOCAL, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(GB_BW_LOCAL, 1, 1)];
   } else {
      struct { uint32_t iters; float a, b; } pc = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B};
      [e setBytes:&pc length:sizeof(pc) atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(GB_ALU_GROUPS, 1, 1) threadsPerThreadgroup:MTLSizeMake(GB_ALU_LOCAL, 1, 1)];
   }
   [e endEncoding];
}

static id
ch_setup(const struct gb_opts *o, char *e, size_t n)
{
   GBState *st = heavy_state(gb_param(o, "kind", "alu"));
   if (!st)
      snprintf(e, n, "kind must be alu or bw");
   return st;
}

static void
ch_encode(id st, id<MTLCommandBuffer> cb, uint64_t k)
{
   encode_heavy(st, cb);
}

static const struct mtest t_compute_heavy = {.name = "compute.heavy", .known = kind_params, .depths = DEPTHS_ALL,
                                             .setup = ch_setup, .encode = ch_encode, .derive = ch_derive};

/* ---- render.light / render.heavy / render.tvb ---- */
static void
r_encode(id st, id<MTLCommandBuffer> cb, uint64_t k)
{
   encode_render(st, cb, k);
}

static id
rl_setup(const struct gb_opts *o, char *e, size_t n)
{
   const char *kind = gb_param(o, "kind", "tri");
   if (!strcmp(kind, "tri"))
      return render_state(K_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
   if (!strcmp(kind, "clear"))
      return render_state(K_CLEAR, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
   snprintf(e, n, "kind must be tri or clear");
   return nil;
}

static const struct mtest t_render_light = {.name = "render.light", .known = kind_params, .depths = DEPTHS_ALL,
                                            .setup = rl_setup, .encode = r_encode};

static void
rh_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res(gb_param(o, "res", "1080p"), &w, &h);
   if (!strcmp(gb_param(o, "kind", "fill"), "geom"))
      gb_derive_rate(r, "mtri_s", "Mtri/s", us, GB_GEOM_TRIS / 1e6 * 1e6);
   else
      gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h * GB_FILL_LAYERS / 1e9 * 1e6);
}

static id
rh_setup(const struct gb_opts *o, char *e, size_t n)
{
   uint32_t w, h;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(e, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   const char *kind = gb_param(o, "kind", "fill");
   if (!strcmp(kind, "fill"))
      return render_state(K_FILL, w, h, 0);
   if (!strcmp(kind, "geom"))
      return render_state(K_GEOM, w, h, 0);
   snprintf(e, n, "kind must be fill or geom");
   return nil;
}

static const struct mtest t_render_heavy = {.name = "render.heavy", .known = rh_params, .depths = DEPTHS_ALL,
                                            .setup = rh_setup, .encode = r_encode, .derive = rh_derive};

static void
tvb_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   gb_derive_rate(r, "mtri_s", "Mtri/s", us, gb_param_int(o, "tris", 262144) / 1e6 * 1e6);
}

static id
tvb_setup(const struct gb_opts *o, char *e, size_t n)
{
   long tris = gb_param_int(o, "tris", 262144);
   if (tris < 1 || tris > (16 << 20)) {
      snprintf(e, n, "tris out of range");
      return nil;
   }
   return render_state(K_TVB, GB_TVB_W, GB_TVB_H, (uint32_t)tris);
}

static const struct mtest t_render_tvb = {.name = "render.tvb", .known = tvb_params, .depths = {1, 4}, .nd = 2,
                                          .report_first = true, .setup = tvb_setup, .encode = r_encode,
                                          .derive = tvb_derive};

/* ---- copy.bw ---- */
static void
copy_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   gb_derive_rate(r, "gbps", "GB/s", us, GB_COPY_BYTES / 1e9 * 1e6);
}

static id
copy_setup(const struct gb_opts *o, char *e, size_t n)
{
   const char *kind = gb_param(o, "kind", "buf2buf");
   GBState *st = [GBState new];
   if (!strcmp(kind, "buf2buf")) {
      st.kind = K_B2B;
      st.a = buf(GB_COPY_BYTES, false);
      st.b = buf(GB_COPY_BYTES, false);
   } else if (!strcmp(kind, "buf2img")) {
      st.kind = K_B2I;
      st.a = buf(GB_COPY_BYTES, false);
      st.t = tex(GB_COPY_DIM, GB_COPY_DIM, MTLTextureUsageShaderRead);
   } else if (!strcmp(kind, "fill")) {
      st.kind = K_FILLBUF;
      st.b = buf(GB_COPY_BYTES, false);
   } else {
      snprintf(e, n, "kind must be buf2buf, buf2img or fill");
      return nil;
   }
   return st;
}

static void
copy_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   GBState *s = state;
   id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
   switch (s.kind) {
   case K_B2B:
      [e copyFromBuffer:s.a sourceOffset:0 toBuffer:s.b destinationOffset:0 size:GB_COPY_BYTES];
      break;
   case K_B2I:
      [e copyFromBuffer:s.a
               sourceOffset:0
          sourceBytesPerRow:GB_COPY_DIM * 4
        sourceBytesPerImage:GB_COPY_BYTES
                 sourceSize:MTLSizeMake(GB_COPY_DIM, GB_COPY_DIM, 1)
                  toTexture:s.t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
      break;
   default:
      [e fillBuffer:s.b range:NSMakeRange(0, GB_COPY_BYTES) value:0x3f];
      break;
   }
   [e endEncoding];
}

static const struct mtest t_copy_bw = {.name = "copy.bw", .known = kind_params, .depths = {1, 4}, .nd = 2,
                                       .setup = copy_setup, .encode = copy_encode, .derive = copy_derive};

/* ---- diag.fill: copy.bw kind=fill with a variable size and an optional pre-touch ---- */
static const char *const diag_fill_params[] = {"size_mib", "init", "storage", NULL};

static uint64_t
diag_fill_bytes(const struct gb_opts *o)
{
   return (uint64_t)gb_param_int(o, "size_mib", 64) << 20;
}

static id
diag_fill_setup(const struct gb_opts *o, char *e, size_t n)
{
   long mib = gb_param_int(o, "size_mib", 64), init = gb_param_int(o, "init", 0);
   const char *storage = gb_param(o, "storage", "private");
   bool shared = !strcmp(storage, "shared");
   if (mib < 1 || mib > 1024 || (init != 0 && init != 1) || (!shared && strcmp(storage, "private"))) {
      snprintf(e, n, "size_mib 1..1024, init 0|1, storage private|shared");
      return nil;
   }
   GBState *st = [GBState new];
   st.kind = K_FILLBUF;
   st.n = (uint32_t)mib;
   st.b = buf((size_t)mib << 20, shared);
   if (init) {
      /* One fill before timing: later fills overwrite pages the GPU has already written. */
      id<MTLCommandQueue> q = [g_dev newCommandQueue];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b fillBuffer:st.b range:NSMakeRange(0, (NSUInteger)mib << 20) value:0];
      [b endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
   }
   return st;
}

static void
diag_fill_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   GBState *s = state;
   id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
   [e fillBuffer:s.b range:NSMakeRange(0, (NSUInteger)s.n << 20) value:0x3f];
   [e endEncoding];
}

static void
diag_fill_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   gb_derive_rate(r, "gbps", "GB/s", us, diag_fill_bytes(o) / 1e9 * 1e6);
}

static const struct mtest t_diag_fill = {.name = "diag.fill", .known = diag_fill_params, .depths = {1, 4},
                                         .nd = 2, .setup = diag_fill_setup, .encode = diag_fill_encode,
                                         .derive = diag_fill_derive};

/* ---- query ---- */
static int
run_query(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "timestamp");
   bool occl = !strcmp(kind, "occlusion");
   if (!occl && strcmp(kind, "timestamp")) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "kind must be timestamp or occlusion");
      return 1;
   }
   ctx_init();
   static const int defd[] = {1, 16};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   GBState *base_st = occl ? render_state(K_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0) : light_state();
   id<MTLCounterSampleBuffer> sb = nil;
   id<MTLBuffer> vis = nil;
   if (!occl) {
      if (![g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
         gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "no stage-boundary counter sampling");
         return 0;
      }
      id<MTLCounterSet> ts = nil;
      for (id<MTLCounterSet> cs in g_dev.counterSets)
         if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp])
            ts = cs;
      if (!ts) {
         gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "no timestamp counter set");
         return 0;
      }
      MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
      d.counterSet = ts;
      d.storageMode = MTLStorageModeShared;
      d.sampleCount = 2 * GB_QRING;
      NSError *err = nil;
      sb = [g_dev newCounterSampleBufferWithDescriptor:d error:&err];
      if (!sb) {
         gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "skip", "counter sample buffer failed");
         return 0;
      }
   } else {
      vis = buf(8 * GB_QRING, true);
   }
   __block struct gb_samples readback = {0};
   __block bool collect = false;
   GBEncode qenc;
   if (!occl) {
      qenc = ^(id<MTLCommandBuffer> cb, uint64_t k) {
         uint32_t i = (uint32_t)(k % GB_QRING);
         MTLComputePassDescriptor *cpd = [MTLComputePassDescriptor computePassDescriptor];
         cpd.sampleBufferAttachments[0].sampleBuffer = sb;
         cpd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
         cpd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
         id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDescriptor:cpd];
         uint32_t v = (uint32_t)k;
         [e setComputePipelineState:base_st.cp];
         [e setBuffer:base_st.a offset:0 atIndex:0];
         [e setBytes:&v length:4 atIndex:1];
         [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(GB_LIGHT_LOCAL, 1, 1)];
         [e endEncoding];
      };
   } else {
      qenc = ^(id<MTLCommandBuffer> cb, uint64_t k) {
         uint32_t i = (uint32_t)(k % GB_QRING);
         MTLRenderPassDescriptor *rp = pass_desc(base_st.t);
         rp.visibilityResultBuffer = vis;
         id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
         float color[4] = {1, 1, 1, 1};
         [e setRenderPipelineState:base_st.rp];
         [e setFragmentBytes:color length:16 atIndex:0];
         [e setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:8 * i];
         [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
         [e setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
         [e endEncoding];
      };
   }
   GBDone qdone = ^(uint64_t k) {
      uint32_t i = (uint32_t)(k % GB_QRING);
      uint64_t t0 = gb_now_ns();
      volatile uint64_t sink = 0;
      if (!occl) {
         NSData *d = [sb resolveCounterRange:NSMakeRange(2 * i, 2)];
         if (d.length >= 16)
            sink = ((const uint64_t *)d.bytes)[1];
      } else {
         sink = ((volatile uint64_t *)vis.contents)[i];
      }
      (void)sink;
      uint64_t t1 = gb_now_ns();
      if (collect)
         gb_samples_push(&readback, (t1 - t0) / 1e3);
   };
   GBEncode benc = occl ? ^(id<MTLCommandBuffer> cb, uint64_t k) { encode_render(base_st, cb, k); }
                        : ^(id<MTLCommandBuffer> cb, uint64_t k) { encode_light(base_st, cb, k); };
   for (int di = 0; di < nd; di++) {
      @autoreleasepool {
         int d = depths[di];
         struct gb_pass bp = {0}, qp = {0};
         gb_samples_clear(&readback);
         /* Base and query windows alternate in ABBA order, as on Vulkan: a drift of the GPU's or the system's
          * state over the row falls on both alike instead of showing as an overhead of either sign. */
         for (int run = 0; run < o->runs; run++) {
            for (int half = 0; half < 2; half++) {
               bool with_query = (half == 1) == (run % 2 == 0);
               struct gb_pass one = {0};
               GBStream *s = with_query
                  ? [[GBStream alloc] initWithQueue:new_queue(d) depth:d encode:qenc done:qdone]
                  : [[GBStream alloc] initWithQueue:new_queue(d) depth:d encode:benc done:nil];
               collect = with_query;
               stream_run(s, false, gb_now_ns(), W, 1, R, &one);
               collect = false;
               gb_pass_append(with_query ? &qp : &bp, &one);
               gb_pass_free(&one);
            }
         }
         struct gb_result r;
         gb_result_begin(&r, o, GB_MODE_QUEUE, d, NULL);
         gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
         if (bp.failed || qp.failed || !qp.submits || !bp.submits) {
            gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
         } else {
            gb_result_stream_metrics(&r, &qp, NULL);
            struct gb_samples bus = {0}, qus = {0};
            gb_pass_rate(&bp, &bus);
            gb_pass_rate(&qp, &qus);
            gb_result_metric(&r, "base_us_per_submit", "us", true, &bus, true);
            gb_result_metric1(&r, "overhead_us", "us", true, gb_median(&qus) - gb_median(&bus));
            gb_result_metric(&r, "readback_us", "us", true, &readback, false);
            gb_samples_free(&bus);
            gb_samples_free(&qus);
            gb_result_finish(&r, "ok", "overhead_us", NULL);
         }
         gb_pass_free(&bp);
         gb_pass_free(&qp);
      }
   }
   gb_samples_free(&readback);
   return 0;
}

/* ---- concurrent.cr ---- */
struct cr_thr {
   void *s;
   uint64_t t0, warm, win;
   int nwin;
   struct gb_pass pass;
   pthread_t th;
};

static void *
cr_thr_main(void *p)
{
   struct cr_thr *t = p;
   @autoreleasepool {
      stream_run((__bridge GBStream *)t->s, false, t->t0, t->warm, t->nwin, t->win, &t->pass);
   }
   return NULL;
}

static int
run_concurrent(const struct gb_opts *o)
{
   static const char *const known[] = {"mode", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   ctx_init();
   const char *mode = gb_param(o, "mode", "queues");
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   if (!strcmp(mode, "queues")) {
      const int depth = GB_CR_DEPTH;
      GBState *ch = heavy_state("alu");
      GBState *rh = render_state(K_FILL, 1920, 1080, 0);
      GBStream *sc = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                              encode:^(id<MTLCommandBuffer> cb, uint64_t k) { encode_heavy(ch, cb); }
                                                done:nil];
      GBStream *sr = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                              encode:^(id<MTLCommandBuffer> cb, uint64_t k) { encode_render(rh, cb, k); }
                                                done:nil];
      struct gb_pass ca, ra;
      stream_run(sc, false, gb_now_ns(), W, o->runs, R, &ca);
      stream_run(sr, false, gb_now_ns(), W, o->runs, R, &ra);
      uint64_t t0 = gb_now_ns() + 20000000ull;
      struct cr_thr th[2] = {{.s = (__bridge void *)sc, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
                             {.s = (__bridge void *)sr, .t0 = t0, .warm = W, .win = R, .nwin = o->runs}};
      for (int i = 0; i < 2; i++)
         pthread_create(&th[i].th, NULL, cr_thr_main, &th[i]);
      for (int i = 0; i < 2; i++)
         pthread_join(th[i].th, NULL);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth,
                      (const char *const[]){"compute=compute.heavy:alu", "render=render.heavy:fill:1080p", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples s[4] = {0};
      gb_pass_rate(&ca, &s[0]);
      gb_pass_rate(&ra, &s[1]);
      gb_pass_rate(&th[0].pass, &s[2]);
      gb_pass_rate(&th[1].pass, &s[3]);
      bool bad = ca.failed || ra.failed || th[0].pass.failed || th[1].pass.failed;
      for (int i = 0; i < 4; i++)
         bad |= !s[i].n;
      if (bad) {
         gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
      } else {
         gb_result_metric(&r, "compute_alone_us", "us", true, &s[0], true);
         gb_result_metric(&r, "render_alone_us", "us", true, &s[1], true);
         gb_result_metric(&r, "compute_conc_us", "us", true, &s[2], true);
         gb_result_metric(&r, "render_conc_us", "us", true, &s[3], true);
         double ov = gb_median(&s[0]) / gb_median(&s[2]) + gb_median(&s[1]) / gb_median(&s[3]);
         gb_result_metric1(&r, "overlap", "x", false, ov);
         gb_result_finish(&r, "ok", "overlap", NULL);
      }
      for (int i = 0; i < 4; i++)
         gb_samples_free(&s[i]);
      gb_pass_free(&ca);
      gb_pass_free(&ra);
      gb_pass_free(&th[0].pass);
      gb_pass_free(&th[1].pass);
      return 0;
   }
   if (!strcmp(mode, "interleave")) {
      const int depth = GB_IL_DEPTH;
      GBState *cl = light_state();
      GBState *rl = render_state(K_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
      struct gb_pass pc, pr, pi;
      GBStream *s = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                             encode:^(id<MTLCommandBuffer> cb, uint64_t k) { encode_light(cl, cb, k); }
                                               done:nil];
      stream_run(s, false, gb_now_ns(), W, o->runs, R, &pc);
      s = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                   encode:^(id<MTLCommandBuffer> cb, uint64_t k) { encode_render(rl, cb, k); }
                                     done:nil];
      stream_run(s, false, gb_now_ns(), W, o->runs, R, &pr);
      s = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                   encode:^(id<MTLCommandBuffer> cb, uint64_t k) {
                                      if (k & 1)
                                         encode_light(cl, cb, k);
                                      else
                                         encode_render(rl, cb, k);
                                   }
                                     done:nil];
      stream_run(s, false, gb_now_ns(), W, o->runs, R, &pi);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depth,
                      (const char *const[]){"compute=compute.light", "render=render.light:tri", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples s3[3] = {0};
      gb_pass_rate(&pc, &s3[0]);
      gb_pass_rate(&pr, &s3[1]);
      gb_pass_rate(&pi, &s3[2]);
      if (pc.failed || pr.failed || pi.failed || !s3[0].n || !s3[1].n || !s3[2].n) {
         gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
      } else {
         gb_result_metric(&r, "us_per_submit", "us", true, &s3[2], true);
         gb_result_metric(&r, "latency_us", "us", true, &pi.latency, false);
         gb_result_metric(&r, "compute_only_us", "us", true, &s3[0], true);
         gb_result_metric(&r, "render_only_us", "us", true, &s3[1], true);
         gb_result_metric1(&r, "switch_us", "us", true,
                           gb_median(&s3[2]) - 0.5 * (gb_median(&s3[0]) + gb_median(&s3[1])));
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      for (int i = 0; i < 3; i++)
         gb_samples_free(&s3[i]);
      gb_pass_free(&pc);
      gb_pass_free(&pr);
      gb_pass_free(&pi);
      return 0;
   }
   gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "mode must be queues or interleave");
   return 1;
}

/* ---- sync.signal ---- */
static int
run_sync(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "sema", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   ctx_init();
   const char *kind = gb_param(o, "kind", "host");
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   GBState *cl = light_state();

   if (!strcmp(kind, "host")) {
      id<MTLCommandQueue> q = new_queue(1);
      id<MTLSharedEvent> ev = [g_dev newSharedEvent];
      struct gb_samples lat = {0}, base = {0};
      uint64_t v = 0;
      bool bad = false;
      for (int phase = 0; phase < 2 && !bad; phase++) {
         uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
         while (gb_now_ns() < t_end) {
            @autoreleasepool {
               v++;
               id<MTLCommandBuffer> cb = new_cb(q);
               if (phase == 1)
                  [cb encodeWaitForEvent:ev value:v];
               encode_light(cl, cb, v);
               uint64_t a;
               if (phase == 0) {
                  a = gb_now_ns();
                  [cb commit];
               } else {
                  [cb commit];
                  gb_spin_ns(GB_HOST_SIGNAL_DELAY_NS);
                  a = gb_now_ns();
                  ev.signaledValue = v;
               }
               [cb waitUntilCompleted];
               uint64_t b = gb_now_ns();
               if (!cb_ok(cb)) {
                  bad = true;
                  break;
               }
               if (a >= tm0)
                  gb_samples_push(phase ? &lat : &base, (b - a) / 1e3);
            }
         }
      }
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad || !lat.n) {
         gb_result_finish(&r, "fail", NULL, "command buffer error");
      } else {
         gb_result_metric(&r, "latency_us", "us", true, &lat, false);
         gb_result_metric(&r, "base_latency_us", "us", true, &base, false);
         gb_result_metric1(&r, "added_us", "us", true, gb_median(&lat) - gb_median(&base));
         gb_result_end_metrics(&r);
         gb_json_str(&r.j, "metal_sync", "MTLSharedEvent");
         gb_result_finish(&r, "ok", "latency_us", NULL);
      }
      gb_samples_free(&lat);
      gb_samples_free(&base);
      return 0;
   }
   if (!strcmp(kind, "queue")) {
      const char *sema = gb_param(o, "sema", "timeline");
      if (strcmp(sema, "binary") && strcmp(sema, "timeline")) {
         gb_result_simple(o, GB_MODE_QUEUES, 2, NULL, "fail", "sema must be timeline or binary");
         return 1;
      }
      enum { RING = GB_PING_RING };
      id<MTLCommandQueue> qa = new_queue(RING), qb = new_queue(RING);
      id<MTLEvent> eS = [g_dev newEvent], eT = [g_dev newEvent];
      NSMutableArray *ring = [NSMutableArray arrayWithCapacity:RING];
      for (int i = 0; i < RING; i++)
         [ring addObject:[NSNull null]];
      uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
      uint64_t k = 0, seen = 0, win[GB_MAX_WIN] = {0};
      bool bad = false;
      for (;;) {
         @autoreleasepool {
            uint64_t now = gb_now_ns();
            if (k - seen >= RING || (now >= t_end && seen < k)) {
               id<MTLCommandBuffer> cb = ring[(seen + 1) % RING];
               [cb waitUntilCompleted];
               if (!cb_ok(cb)) {
                  bad = true;
                  break;
               }
               uint64_t t = gb_now_ns();
               if (t >= tm0 && t < t_end)
                  win[(t - tm0) / R]++;
               ring[(seen + 1) % RING] = [NSNull null];
               seen++;
               continue;
            }
            if (now >= t_end)
               break;
            k++;
            id<MTLCommandBuffer> a = new_cb(qa);
            [a encodeWaitForEvent:eT value:k - 1];
            encode_light(cl, a, k);
            [a encodeSignalEvent:eS value:k];
            [a commit];
            id<MTLCommandBuffer> b = new_cb(qb);
            [b encodeWaitForEvent:eS value:k];
            encode_light(cl, b, k);
            [b encodeSignalEvent:eT value:k];
            [b commit];
            ring[k % RING] = b;
         }
      }
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2, (const char *const[]){"work=compute.light", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples hop = {0};
      for (int w = 0; w < o->runs; w++)
         if (win[w])
            gb_samples_push(&hop, R / 1e3 / (2.0 * win[w]));
      if (bad || !hop.n) {
         gb_result_finish(&r, "fail", NULL, "command buffer error");
      } else {
         gb_result_metric(&r, "us_per_hop", "us", true, &hop, true);
         gb_result_end_metrics(&r);
         gb_json_str(&r.j, "metal_sync", "MTLEvent");
         gb_result_finish(&r, "ok", "us_per_hop", NULL);
      }
      gb_samples_free(&hop);
      return 0;
   }
   gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be host or queue");
   return 1;
}

/* ---- idle.wake ---- */
static int
run_idle(const struct gb_opts *o)
{
   static const char *const known[] = {"gap_ms", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   long gap = gb_param_int(o, "gap_ms", 20);
   if (gap < 0 || gap > 5000) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "gap_ms out of range");
      return 1;
   }
   int n = (int)(GB_IDLE_BUDGET_MS / (gap ? gap : 1));
   n = n < GB_IDLE_MIN_SAMPLES ? GB_IDLE_MIN_SAMPLES : n > GB_IDLE_MAX_SAMPLES ? GB_IDLE_MAX_SAMPLES : n;
   ctx_init();
   GBState *cl = light_state();
   id<MTLCommandQueue> q = new_queue(1);
   struct gb_samples lat = {0}, hot = {0};
   bool bad = false;
   uint64_t v = 0;
   for (int i = -2; i < n && !bad; i++) {
      @autoreleasepool {
         id<MTLCommandBuffer> cb = new_cb(q);
         encode_light(cl, cb, ++v);
         gb_sleep_until_ns(gb_now_ns() + (uint64_t)gap * 1000000ull);
         uint64_t a = gb_now_ns();
         [cb commit];
         [cb waitUntilCompleted];
         uint64_t b = gb_now_ns();
         id<MTLCommandBuffer> cb2 = new_cb(q);
         encode_light(cl, cb2, ++v);
         uint64_t c = gb_now_ns();
         [cb2 commit];
         [cb2 waitUntilCompleted];
         uint64_t d = gb_now_ns();
         if (!cb_ok(cb) || !cb_ok(cb2))
            bad = true;
         if (i >= 0) {
            gb_samples_push(&lat, (b - a) / 1e3);
            gb_samples_push(&hot, (d - c) / 1e3);
         }
      }
   }
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", NULL});
   if (bad || !lat.n) {
      gb_result_finish(&r, "fail", NULL, "command buffer error");
   } else {
      gb_result_metric(&r, "wake_latency_us", "us", true, &lat, false);
      gb_result_metric(&r, "awake_latency_us", "us", true, &hot, false);
      gb_result_metric1(&r, "wake_cost_us", "us", true, gb_median(&lat) - gb_median(&hot));
      gb_result_finish(&r, "ok", "wake_latency_us", NULL);
   }
   gb_samples_free(&lat);
   gb_samples_free(&hot);
   return 0;
}

/* ---- mem.buffer / mem.image ---- */
static int
run_mem(const struct gb_opts *o, bool image)
{
   static const char *const kb[] = {"size", NULL}, *const ki[] = {"dim", NULL};
   if (!gb_check_params(o, image ? ki : kb)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   long size = gb_param_int(o, "size", 1 << 20);
   long dim = gb_param_int(o, "dim", 1024);
   if (size < 4096 || size > (1l << 30) || dim < 16 || dim > 16384) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "size/dim out of range");
      return 1;
   }
   ctx_init();
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   struct gb_samples s_alloc = {0}, s_map = {0}, s_touch = {0}, s_free = {0}, s_cycle = {0};
   id<MTLCommandQueue> q = new_queue(1);
   bool bad = false;

   uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
   while (gb_now_ns() < t_end) {
      @autoreleasepool {
         uint64_t a = gb_now_ns();
         bool m = a >= tm0;
         if (!image) {
            id<MTLBuffer> b = [g_dev newBufferWithLength:size options:SHARED];
            uint64_t t1 = gb_now_ns();
            void *p = b.contents;
            uint64_t t2 = gb_now_ns();
            for (long off = 0; off < size; off += GB_TOUCH_STRIDE)
               ((volatile uint32_t *)p)[off / 4] = (uint32_t)off;
            uint64_t t3 = gb_now_ns();
            b = nil;
            uint64_t t4 = gb_now_ns();
            if (m) {
               gb_samples_push(&s_alloc, (t1 - a) / 1e3);
               gb_samples_push(&s_map, (t2 - t1) / 1e3);
               gb_samples_push(&s_touch, (t3 - t2) / 1e3);
               gb_samples_push(&s_free, (t4 - t3) / 1e3);
            }
         } else {
            id<MTLTexture> t = tex((uint32_t)dim, (uint32_t)dim, MTLTextureUsageRenderTarget);
            uint64_t t1 = gb_now_ns();
            if (!t)
               bad = true;
            t = nil;
            uint64_t t2 = gb_now_ns();
            if (m) {
               gb_samples_push(&s_alloc, (t1 - a) / 1e3);
               gb_samples_push(&s_free, (t2 - t1) / 1e3);
            }
         }
      }
   }

   t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
   while (!bad && gb_now_ns() < t_end) {
      @autoreleasepool {
         uint64_t a = gb_now_ns();
         id<MTLCommandBuffer> cb = new_cb(q);
         if (!image) {
            id<MTLBuffer> b = [g_dev newBufferWithLength:size options:PRIVATE];
            id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
            [e fillBuffer:b range:NSMakeRange(0, size) value:0];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            b = nil;
         } else {
            id<MTLTexture> t = tex((uint32_t)dim, (uint32_t)dim, MTLTextureUsageRenderTarget);
            id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass_desc(t)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            t = nil;
         }
         if (!cb_ok(cb))
            bad = true;
         cb = nil;
         uint64_t z = gb_now_ns();
         if (a >= tm0)
            gb_samples_push(&s_cycle, (z - a) / 1e3);
      }
   }

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "command buffer error");
   } else {
      gb_result_metric(&r, "alloc_us", "us", true, &s_alloc, false);
      gb_result_metric(&r, "map_us", "us", true, &s_map, false);
      gb_result_metric(&r, "touch_us", "us", true, &s_touch, false);
      gb_result_metric(&r, "free_us", "us", true, &s_free, false);
      gb_result_metric(&r, "cycle_us", "us", true, &s_cycle, false);
      gb_result_finish(&r, "ok", "cycle_us", NULL);
   }
   gb_samples_free(&s_alloc);
   gb_samples_free(&s_map);
   gb_samples_free(&s_touch);
   gb_samples_free(&s_free);
   gb_samples_free(&s_cycle);
   return 0;
}

/* ======================= native code dump (FAIRNESS.md) ======================= */

/* --dump-archive DIR: one binary archive per suite pipeline, serialized by
 * this machine's own Metal compiler. tools/fairness decodes them. */
static int
dump_archives(const char *dir)
{
   ctx_init();
   [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:dir]
                             withIntermediateDirectories:YES
                                              attributes:nil
                                                   error:nil];
   struct { const char *name, *cs, *vs, *fs; bool blend; } pl[] = {
      {"light", "light_main", NULL, NULL, false},
      {"alu", "alu_main", NULL, NULL, false},
      {"bw", "bw_main", NULL, NULL, false},
      {"fs_color", NULL, "fs_vert", "color_frag", false},
      {"fs_color_blend", NULL, "fs_vert", "color_frag", true},
      {"geom_color", NULL, "geom_vert", "color_frag", false},
      {"tvb_color", NULL, "tvb_vert", "color_frag", false},
   };
   int bad = 0;
   for (unsigned i = 0; i < sizeof(pl) / sizeof(pl[0]); i++) {
      @autoreleasepool {
         NSError *err = nil;
         MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
         id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:ad error:&err];
         BOOL ok;
         if (pl[i].cs) {
            MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
            cd.computeFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:pl[i].cs]];
            ok = [ar addComputePipelineFunctionsWithDescriptor:cd error:&err];
         } else {
            MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
            d.vertexFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:pl[i].vs]];
            d.fragmentFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:pl[i].fs]];
            d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            if (pl[i].blend) {
               MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
               c.blendingEnabled = YES;
               c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
               c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
               c.sourceAlphaBlendFactor = MTLBlendFactorOne;
               c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            ok = [ar addRenderPipelineFunctionsWithDescriptor:d error:&err];
         }
         NSString *path = [NSString stringWithFormat:@"%s/%s.bin", dir, pl[i].name];
         if (!ok || ![ar serializeToURL:[NSURL fileURLWithPath:path] error:&err]) {
            fprintf(stderr, "%s: %s\n", pl[i].name, err.localizedDescription.UTF8String);
            bad++;
         } else
            printf("%s\n", path.UTF8String);
      }
   }
   return bad ? 1 : 0;
}

/* ======================= 90%-coverage Wave-1 tests ======================= */

static void
mtl_buf_upload(id<MTLBuffer> dst, const void *src, size_t bytes)
{
   id<MTLBuffer> stg = buf(bytes, true);
   memcpy(stg.contents, src, bytes);
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
   [b copyFromBuffer:stg sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
   [b endEncoding];
   [cb commit];
   [cb waitUntilCompleted];
}

static void
mtl_buf_readback(id<MTLBuffer> src, void *dst, size_t bytes)
{
   id<MTLBuffer> stg = buf(bytes, true);
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
   [b copyFromBuffer:src sourceOffset:0 toBuffer:stg destinationOffset:0 size:bytes];
   [b endEncoding];
   [cb commit];
   [cb waitUntilCompleted];
   memcpy(dst, stg.contents, bytes);
}

static void
mtl_tex_readback(id<MTLTexture> src, void *dst, uint32_t w, uint32_t h, size_t bpp)
{
   size_t bytes = (size_t)w * h * bpp;
   id<MTLBuffer> stg = buf(bytes, true);
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
   [b copyFromTexture:src
          sourceSlice:0
          sourceLevel:0
         sourceOrigin:MTLOriginMake(0, 0, 0)
           sourceSize:MTLSizeMake(w, h, 1)
             toBuffer:stg
    destinationOffset:0
 destinationBytesPerRow:w * bpp
 destinationBytesPerImage:bytes];
   [b endEncoding];
   [cb commit];
   [cb waitUntilCompleted];
   memcpy(dst, stg.contents, bytes);
}

static id<MTLBuffer>
buf_tracked(size_t size, bool host)
{
   MTLResourceOptions opts = (host ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate) |
                             MTLResourceHazardTrackingModeTracked;
   id<MTLBuffer> b = [g_dev newBufferWithLength:size options:opts];
   if (!b)
      gb_die("tracked buffer of %zu bytes failed", size);
   return b;
}

static id<MTLTexture>
tex_ex_opt(uint32_t w, uint32_t h, MTLPixelFormat fmt, NSUInteger samples, MTLStorageMode sm,
           MTLTextureUsage usage, bool tracked)
{
   MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
   if (samples > 1) {
      d.textureType = MTLTextureType2DMultisample;
      d.sampleCount = samples;
   }
   d.storageMode = sm;
   d.usage = usage;
   d.hazardTrackingMode = tracked ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
   id<MTLTexture> t = [g_dev newTextureWithDescriptor:d];
   if (!t)
      gb_die("texture %ux%u (samples=%lu) failed", w, h, (unsigned long)samples);
   return t;
}

static id<MTLTexture>
tex_ex(uint32_t w, uint32_t h, MTLPixelFormat fmt, NSUInteger samples, MTLStorageMode sm, MTLTextureUsage usage)
{
   return tex_ex_opt(w, h, fmt, samples, sm, usage, false);
}

static id<MTLComputePipelineState>
cpipe_const_u32(const char *fn, NSUInteger idx, uint32_t val)
{
   NSError *err = nil;
   MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
   [fc setConstantValue:&val type:MTLDataTypeUInt atIndex:idx];
   id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fn] constantValues:fc error:&err];
   if (!f)
      gb_die("function %s const failed: %s", fn, err.localizedDescription.UTF8String);
   id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&err];
   if (!p)
      gb_die("compute pipeline %s: %s", fn, err.localizedDescription.UTF8String);
   return p;
}

/* ---- 1. compute.uber ---- */

@interface GBUberState : NSObject
@property(nonatomic) enum gb_uber_kind kind;
@property(nonatomic, strong) id<MTLComputePipelineState> cp;
@property(nonatomic, strong) id<MTLBuffer> inBuf, outBuf;
@property(nonatomic) uint32_t *ref;
@property(nonatomic) uint32_t refCrc;
@end
@implementation GBUberState
- (void)dealloc { free(_ref); }
@end

static GBUberState *
uber_make_state(enum gb_uber_kind kind)
{
   GBUberState *st = [GBUberState new];
   st.kind = kind;
   size_t out_bytes = (size_t)GB_UBER_THREADS * sizeof(uint32_t);
   st.outBuf = buf(out_bytes, false);
   uint32_t *ilp_cpu = NULL;
   if (kind == GB_UBER_ILP) {
      size_t in_bytes = (size_t)GB_UBER_ILP_WORDS * sizeof(uint32_t);
      ilp_cpu = malloc(in_bytes);
      gb_uber_fill_ilp_input(ilp_cpu, GB_UBER_ILP_WORDS);
      st.inBuf = buf(in_bytes, false);
      mtl_buf_upload(st.inBuf, ilp_cpu, in_bytes);
      st.cp = cpipe("uber_ilp_main");
   } else {
      const char *fn = kind == GB_UBER_DIV ? "uber_div_main"
                       : kind == GB_UBER_UNI ? "uber_uni_main" : "uber_sram_main";
      st.cp = cpipe(fn);
   }
   st.ref = malloc(out_bytes);
   st.refCrc = gb_uber_ref(kind, ilp_cpu, st.ref, GB_UBER_THREADS);
   free(ilp_cpu);
   return st;
}

static id
uber_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "divergent");
   if (!strcmp(kind, "divergent")) return uber_make_state(GB_UBER_DIV);
   if (!strcmp(kind, "uniform"))   return uber_make_state(GB_UBER_UNI);
   if (!strcmp(kind, "ilp"))       return uber_make_state(GB_UBER_ILP);
   snprintf(err, n, "kind must be divergent, uniform or ilp");
   return nil;
}

static void
uber_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBUberState *st = state;
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:st.cp];
   struct { uint32_t a, b; } pc = {
      GB_UBER_SEED,
      st.kind == GB_UBER_ILP ? (GB_UBER_ILP_WORDS - 1u) : GB_UBER_ITERS,
   };
   if (st.kind == GB_UBER_ILP) {
      [e setBuffer:st.inBuf offset:0 atIndex:0];
      [e setBuffer:st.outBuf offset:0 atIndex:1];
      [e setBytes:&pc length:sizeof(pc) atIndex:2];
   } else {
      [e setBuffer:st.outBuf offset:0 atIndex:0];
      [e setBytes:&pc length:sizeof(pc) atIndex:1];
   }
   [e dispatchThreadgroups:MTLSizeMake(GB_UBER_GROUPS, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(GB_UBER_LOCAL, 1, 1)];
   [e endEncoding];
}

static void
uber_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   (void)o;
   gb_derive_rate(r, "mthreads_s", "Mthreads/s", us, (double)GB_UBER_THREADS);
}

static void
uber_check(id state, struct gb_check *chk)
{
   GBUberState *st = state;
   uint32_t *got = malloc((size_t)GB_UBER_THREADS * sizeof(uint32_t));
   mtl_buf_readback(st.outBuf, got, (size_t)GB_UBER_THREADS * sizeof(uint32_t));
   gb_check_words(chk, got, st.ref, st.refCrc, GB_UBER_THREADS);
   free(got);
}

static const char *const uber_params[] = {"kind", NULL};
static const struct mtest t_compute_uber = {
   .name = "compute.uber", .known = uber_params, .depths = {1, 4}, .nd = 2,
   .setup = uber_setup, .encode = uber_encode, .derive = uber_derive, .check = uber_check,
};

/* ---- 2. compute.conv2d ---- */

@interface GBConvState : NSObject
@property(nonatomic) uint32_t w, h;
@property(nonatomic, strong) id<MTLComputePipelineState> ph, pv;
@property(nonatomic, strong) id<MTLBuffer> inBuf, midBuf, outBuf;
@property(nonatomic) uint32_t *ref;
@property(nonatomic) uint32_t refCrc;
@end
@implementation GBConvState
- (void)dealloc { free(_ref); }
@end

static id
conv_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "gaussian");
   bool bilateral = !strcmp(kind, "bilateral");
   if (!bilateral && strcmp(kind, "gaussian")) {
      snprintf(err, n, "kind must be gaussian or bilateral");
      return nil;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   GBConvState *st = [GBConvState new];
   st.w = w; st.h = h;
   size_t bytes = (size_t)w * h * sizeof(uint32_t);
   uint32_t *in_cpu = malloc(bytes);
   gb_conv2d_fill_input(in_cpu, w, h);
   st.ref = malloc(bytes);
   st.refCrc = gb_conv2d_ref(bilateral, in_cpu, st.ref, w, h);
   st.inBuf = buf(bytes, false);
   st.midBuf = buf(bytes, false);
   st.outBuf = buf(bytes, false);
   mtl_buf_upload(st.inBuf, in_cpu, bytes);
   free(in_cpu);
   st.ph = cpipe_const_u32("conv2d_h_main", 10, bilateral ? 1u : 0u);
   st.pv = cpipe_const_u32("conv2d_v_main", 10, bilateral ? 1u : 0u);
   return st;
}

static void
conv_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBConvState *st = state;
   struct { int32_t w, h; } pc = {(int32_t)st.w, (int32_t)st.h};
   MTLSize tg = MTLSizeMake((st.w + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL,
                            (st.h + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL, 1);
   MTLSize loc = MTLSizeMake(GB_CONV_LOCAL, GB_CONV_LOCAL, 1);

   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:st.ph];
   [e setBuffer:st.inBuf offset:0 atIndex:0];
   [e setBuffer:st.midBuf offset:0 atIndex:1];
   [e setBytes:&pc length:sizeof(pc) atIndex:2];
   [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
   [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
   [e setComputePipelineState:st.pv];
   [e setBuffer:st.midBuf offset:0 atIndex:0];
   [e setBuffer:st.outBuf offset:0 atIndex:1];
   [e setBytes:&pc length:sizeof(pc) atIndex:2];
   [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
   [e endEncoding];
}

static void
conv_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
}

static void
conv_check(id state, struct gb_check *chk)
{
   GBConvState *st = state;
   size_t count = (size_t)st.w * st.h;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   mtl_buf_readback(st.outBuf, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st.ref, st.refCrc, count);
   free(got);
}

static const char *const conv_params[] = {"kind", "res", NULL};
static const struct mtest t_compute_conv2d = {
   .name = "compute.conv2d", .known = conv_params, .depths = {1, 4}, .nd = 2,
   .setup = conv_setup, .encode = conv_encode, .derive = conv_derive, .check = conv_check,
};

/* ---- 3. compute.stereo ---- */

@interface GBStereoState : NSObject
@property(nonatomic) uint32_t disp;
@property(nonatomic, strong) id<MTLComputePipelineState> cp;
@property(nonatomic, strong) id<MTLBuffer> leftBuf, rightBuf, outBuf;
@property(nonatomic) uint32_t *ref;
@property(nonatomic) uint32_t refCrc;
@end
@implementation GBStereoState
- (void)dealloc { free(_ref); }
@end

static id
stereo_setup(const struct gb_opts *o, char *err, size_t n)
{
   long disp = gb_param_int(o, "disp", 64);
   if (disp < 8 || disp > 128) {
      snprintf(err, n, "disp must be 8..128");
      return nil;
   }
   GBStereoState *st = [GBStereoState new];
   st.disp = (uint32_t)disp;
   size_t in_bytes = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   size_t out_words = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   size_t out_bytes = out_words * sizeof(uint32_t);
   uint32_t *l_cpu = calloc(1, in_bytes);
   uint32_t *r_cpu = calloc(1, in_bytes);
   gb_stereo_fill_inputs(l_cpu, r_cpu, st.disp);
   st.ref = calloc(out_words, sizeof(uint32_t));
   st.refCrc = gb_stereo_ref(l_cpu, r_cpu, st.disp, st.ref);

   st.leftBuf = buf(in_bytes, false);
   st.rightBuf = buf(in_bytes, false);
   st.outBuf = buf(out_bytes, false);
   mtl_buf_upload(st.leftBuf, l_cpu, in_bytes);
   mtl_buf_upload(st.rightBuf, r_cpu, in_bytes);
   free(l_cpu);
   free(r_cpu);

   /* Zero output padding columns once so CRC matches CPU reference. */
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
   [b fillBuffer:st.outBuf range:NSMakeRange(0, out_bytes) value:0];
   [b endEncoding];
   [cb commit];
   [cb waitUntilCompleted];

   st.cp = cpipe("stereo_main");
   return st;
}

static void
stereo_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBStereoState *st = state;
   struct { int32_t w, h, pitch, disp; } pc = {
      (int32_t)GB_STEREO_W, (int32_t)GB_STEREO_H, (int32_t)GB_STEREO_PITCH, (int32_t)st.disp
   };
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:st.cp];
   [e setBuffer:st.leftBuf offset:0 atIndex:0];
   [e setBuffer:st.rightBuf offset:0 atIndex:1];
   [e setBuffer:st.outBuf offset:0 atIndex:2];
   [e setBytes:&pc length:sizeof(pc) atIndex:3];
   [e dispatchThreadgroups:MTLSizeMake((GB_STEREO_W + GB_STEREO_LOCAL - 1u) / GB_STEREO_LOCAL,
                                       (GB_STEREO_H + GB_STEREO_LOCAL - 1u) / GB_STEREO_LOCAL, 1)
     threadsPerThreadgroup:MTLSizeMake(GB_STEREO_LOCAL, GB_STEREO_LOCAL, 1)];
   [e endEncoding];
}

static void
stereo_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long disp = gb_param_int(o, "disp", 64);
   gb_derive_rate(r, "mde_s", "Mdisp/s", us, (double)GB_STEREO_W * GB_STEREO_H * (double)disp);
}

static void
stereo_check(id state, struct gb_check *chk)
{
   GBStereoState *st = state;
   size_t count = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   mtl_buf_readback(st.outBuf, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st.ref, st.refCrc, count);
   free(got);
}

static const char *const stereo_params[] = {"disp", NULL};
static const struct mtest t_compute_stereo = {
   .name = "compute.stereo", .known = stereo_params, .depths = {1, 2}, .nd = 2,
   .setup = stereo_setup, .encode = stereo_encode, .derive = stereo_derive, .check = stereo_check,
};

/* ---- 4. compute.edge ---- */

@interface GBEdgeState : NSObject
@property(nonatomic) uint32_t w, h;
@property(nonatomic) BOOL harris;
@property(nonatomic, strong) id<MTLComputePipelineState> p1, p2, p3;
@property(nonatomic, strong) id<MTLBuffer> inBuf, gradBuf, magBuf, respBuf, outBuf, cntBuf;
@property(nonatomic) uint32_t *ref;
@property(nonatomic) uint32_t refCrc, refCorners;
@end
@implementation GBEdgeState
- (void)dealloc { free(_ref); }
@end

static id
edge_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "sobel");
   bool harris = !strcmp(kind, "harris");
   if (!harris && strcmp(kind, "sobel")) {
      snprintf(err, n, "kind must be sobel or harris");
      return nil;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   GBEdgeState *st = [GBEdgeState new];
   st.w = w; st.h = h; st.harris = harris;
   size_t bytes = (size_t)w * h * sizeof(uint32_t);
   uint32_t *in_cpu = malloc(bytes);
   gb_edge_fill_input(in_cpu, w, h);
   st.ref = malloc(bytes);
   uint32_t ref_corners = 0;
   st.refCrc = gb_edge_ref(harris, in_cpu, st.ref, &ref_corners, w, h);
   st.refCorners = ref_corners;

   st.inBuf = buf(bytes, false);
   st.gradBuf = buf(bytes, false);
   st.magBuf = buf(bytes, false);
   st.outBuf = buf(bytes, false);
   mtl_buf_upload(st.inBuf, in_cpu, bytes);
   free(in_cpu);

   st.p1 = cpipe("edge_sobel_main");
   if (!harris) {
      st.p2 = cpipe("edge_nms_main");
   } else {
      st.respBuf = buf(bytes, false);
      st.cntBuf = buf_tracked(64, false);
      st.p2 = cpipe("edge_harris_resp_main");
      st.p3 = cpipe("edge_harris_nms_main");
   }
   return st;
}

static void
edge_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBEdgeState *st = state;
   struct { int32_t w, h; } pc = {(int32_t)st.w, (int32_t)st.h};
   MTLSize tg = MTLSizeMake((st.w + GB_EDGE_LOCAL - 1u) / GB_EDGE_LOCAL,
                            (st.h + GB_EDGE_LOCAL - 1u) / GB_EDGE_LOCAL, 1);
   MTLSize loc = MTLSizeMake(GB_EDGE_LOCAL, GB_EDGE_LOCAL, 1);

   if (st.harris) {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b fillBuffer:st.cntBuf range:NSMakeRange(0, 4) value:0];
      [b endEncoding];
   }
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:st.p1];
   [e setBuffer:st.inBuf offset:0 atIndex:0];
   [e setBuffer:st.gradBuf offset:0 atIndex:1];
   [e setBuffer:st.magBuf offset:0 atIndex:2];
   [e setBytes:&pc length:sizeof(pc) atIndex:3];
   [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
   [e memoryBarrierWithScope:MTLBarrierScopeBuffers];

   if (!st.harris) {
      [e setComputePipelineState:st.p2];
      [e setBuffer:st.gradBuf offset:0 atIndex:0];
      [e setBuffer:st.magBuf offset:0 atIndex:1];
      [e setBuffer:st.outBuf offset:0 atIndex:2];
      [e setBytes:&pc length:sizeof(pc) atIndex:3];
      [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
   } else {
      [e setComputePipelineState:st.p2];
      [e setBuffer:st.gradBuf offset:0 atIndex:0];
      [e setBuffer:st.respBuf offset:0 atIndex:1];
      [e setBytes:&pc length:sizeof(pc) atIndex:2];
      [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
      [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [e setComputePipelineState:st.p3];
      [e setBuffer:st.respBuf offset:0 atIndex:0];
      [e setBuffer:st.outBuf offset:0 atIndex:1];
      [e setBuffer:st.cntBuf offset:0 atIndex:2];
      [e setBytes:&pc length:sizeof(pc) atIndex:3];
      [e dispatchThreadgroups:tg threadsPerThreadgroup:loc];
   }
   [e endEncoding];
}

static void
edge_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "mpix_s", "Mpix/s", us, (double)w * h);
}

static void
edge_check(id state, struct gb_check *chk)
{
   GBEdgeState *st = state;
   size_t count = (size_t)st.w * st.h;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   mtl_buf_readback(st.outBuf, got, count * sizeof(uint32_t));
   gb_check_words(chk, got, st.ref, st.refCrc, count);
   free(got);
   if (st.harris && chk->valid) {
      uint32_t cnt = 0;
      mtl_buf_readback(st.cntBuf, &cnt, sizeof(cnt));
      if (cnt != st.refCorners) {
         gb_log("harris corner count mismatch: got %u want %u", cnt, st.refCorners);
         chk->valid = false;
      }
   }
}

static const char *const edge_params[] = {"kind", "res", NULL};
static const struct mtest t_compute_edge = {
   .name = "compute.edge", .known = edge_params, .depths = {1, 4}, .nd = 2,
   .setup = edge_setup, .encode = edge_encode, .derive = edge_derive, .check = edge_check,
};

/* ---- 5. render.deferred ---- */

@interface GBDeferredState : NSObject
@property(nonatomic) uint32_t w, h, msaa;
@property(nonatomic) BOOL tile;
@property(nonatomic, strong) id<MTLTexture> g0, g1, g2, g3, depthTex, finalTex, finalMs;
@property(nonatomic, strong) id<MTLRenderPipelineState> pGeom, pLight;
@property(nonatomic, strong) id<MTLDepthStencilState> dsLess, dsNone;
@end
@implementation GBDeferredState
@end

static GBDeferredState *
deferred_make_state(uint32_t w, uint32_t h, uint32_t msaa, bool tile)
{
   GBDeferredState *st = [GBDeferredState new];
   st.w = w; st.h = h; st.msaa = msaa; st.tile = tile;
   MTLStorageMode gsm = tile ? MTLStorageModeMemoryless : MTLStorageModePrivate;
   MTLTextureUsage gusage = tile ? MTLTextureUsageRenderTarget
                                 : (MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
   bool gtrack = !tile;
   st.g0 = tex_ex_opt(w, h, MTLPixelFormatRGBA8Unorm, msaa, gsm, gusage, gtrack);
   st.g1 = tex_ex_opt(w, h, MTLPixelFormatRGBA16Float, msaa, gsm, gusage, gtrack);
   st.g2 = tex_ex_opt(w, h, MTLPixelFormatRGBA8Unorm, msaa, gsm, gusage, gtrack);
   st.g3 = tex_ex_opt(w, h, MTLPixelFormatR32Float, msaa, gsm, gusage, gtrack);
   st.depthTex = tex_ex(w, h, MTLPixelFormatDepth32Float, msaa, MTLStorageModeMemoryless, MTLTextureUsageRenderTarget);
   st.finalTex = tex_ex(w, h, MTLPixelFormatRGBA8Unorm, 1, MTLStorageModePrivate,
                        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
   if (tile && msaa == 4u)
      st.finalMs = tex_ex(w, h, MTLPixelFormatRGBA8Unorm, msaa, MTLStorageModeMemoryless, MTLTextureUsageRenderTarget);

   MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
   dsd.depthCompareFunction = MTLCompareFunctionLess;
   dsd.depthWriteEnabled = YES;
   st.dsLess = [g_dev newDepthStencilStateWithDescriptor:dsd];
   dsd.depthCompareFunction = MTLCompareFunctionAlways;
   dsd.depthWriteEnabled = NO;
   st.dsNone = [g_dev newDepthStencilStateWithDescriptor:dsd];

   NSError *err = nil;
   MTLRenderPipelineDescriptor *gd = [MTLRenderPipelineDescriptor new];
   gd.vertexFunction = [g_lib newFunctionWithName:@"deferred_geom_vert"];
   gd.fragmentFunction = [g_lib newFunctionWithName:@"deferred_geom_frag"];
   gd.rasterSampleCount = msaa;
   gd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   gd.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
   gd.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA8Unorm;
   gd.colorAttachments[3].pixelFormat = MTLPixelFormatR32Float;
   if (tile) {
      gd.colorAttachments[4].pixelFormat = MTLPixelFormatRGBA8Unorm;
      gd.colorAttachments[4].writeMask = MTLColorWriteMaskNone;
   }
   gd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
   st.pGeom = [g_dev newRenderPipelineStateWithDescriptor:gd error:&err];
   if (!st.pGeom)
      gb_die("deferred geom pipeline failed: %s", err.localizedDescription.UTF8String);

   MTLRenderPipelineDescriptor *ld = [MTLRenderPipelineDescriptor new];
   ld.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   if (tile) {
      ld.fragmentFunction = [g_lib newFunctionWithName:(msaa == 4u ? @"deferred_tile_msaa_frag" : @"deferred_tile_frag")];
      ld.rasterSampleCount = msaa;
      ld.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      ld.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
      ld.colorAttachments[2].pixelFormat = MTLPixelFormatRGBA8Unorm;
      ld.colorAttachments[3].pixelFormat = MTLPixelFormatR32Float;
      for (int i = 0; i < 4; i++)
         ld.colorAttachments[i].writeMask = MTLColorWriteMaskNone;
      ld.colorAttachments[4].pixelFormat = MTLPixelFormatRGBA8Unorm;
      ld.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
   } else {
      ld.fragmentFunction = [g_lib newFunctionWithName:(msaa == 4u ? @"deferred_dram_msaa_frag" : @"deferred_dram_frag")];
      ld.rasterSampleCount = 1;
      ld.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   }
   st.pLight = [g_dev newRenderPipelineStateWithDescriptor:ld error:&err];
   if (!st.pLight)
      gb_die("deferred light pipeline failed: %s", err.localizedDescription.UTF8String);
   return st;
}

static id
deferred_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "tile");
   bool tile = !strcmp(mode, "tile");
   if (!tile && strcmp(mode, "dram")) {
      snprintf(err, n, "mode must be tile or dram");
      return nil;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   long msaa = gb_param_int(o, "msaa", 1);
   if (msaa != 1 && msaa != 4) {
      snprintf(err, n, "msaa must be 1 or 4");
      return nil;
   }
   return deferred_make_state(w, h, (uint32_t)msaa, tile);
}

static void
deferred_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBDeferredState *st = state;
   if (st.tile) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      id<MTLTexture> gs[4] = {st.g0, st.g1, st.g2, st.g3};
      for (int i = 0; i < 4; i++) {
         rp.colorAttachments[i].texture = gs[i];
         rp.colorAttachments[i].loadAction = MTLLoadActionClear;
         rp.colorAttachments[i].storeAction = MTLStoreActionDontCare;
      }
      if (st.msaa == 4u) {
         rp.colorAttachments[4].texture = st.finalMs;
         rp.colorAttachments[4].resolveTexture = st.finalTex;
         rp.colorAttachments[4].loadAction = MTLLoadActionClear;
         rp.colorAttachments[4].storeAction = MTLStoreActionMultisampleResolve;
      } else {
         rp.colorAttachments[4].texture = st.finalTex;
         rp.colorAttachments[4].loadAction = MTLLoadActionClear;
         rp.colorAttachments[4].storeAction = MTLStoreActionStore;
      }
      rp.depthAttachment.texture = st.depthTex;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.storeAction = MTLStoreActionDontCare;
      rp.depthAttachment.clearDepth = 1.0;

      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      [e setRenderPipelineState:st.pGeom];
      [e setDepthStencilState:st.dsLess];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:GB_DEFERRED_QUADS * 6u];
      [e setRenderPipelineState:st.pLight];
      [e setDepthStencilState:st.dsNone];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
   } else {
      MTLRenderPassDescriptor *rp1 = [MTLRenderPassDescriptor renderPassDescriptor];
      id<MTLTexture> gs[4] = {st.g0, st.g1, st.g2, st.g3};
      for (int i = 0; i < 4; i++) {
         rp1.colorAttachments[i].texture = gs[i];
         rp1.colorAttachments[i].loadAction = MTLLoadActionClear;
         rp1.colorAttachments[i].storeAction = MTLStoreActionStore;
      }
      rp1.depthAttachment.texture = st.depthTex;
      rp1.depthAttachment.loadAction = MTLLoadActionClear;
      rp1.depthAttachment.storeAction = MTLStoreActionDontCare;
      rp1.depthAttachment.clearDepth = 1.0;

      id<MTLRenderCommandEncoder> e1 = [cb renderCommandEncoderWithDescriptor:rp1];
      [e1 setRenderPipelineState:st.pGeom];
      [e1 setDepthStencilState:st.dsLess];
      [e1 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:GB_DEFERRED_QUADS * 6u];
      [e1 endEncoding];

      MTLRenderPassDescriptor *rp2 = [MTLRenderPassDescriptor renderPassDescriptor];
      rp2.colorAttachments[0].texture = st.finalTex;
      rp2.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp2.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e2 = [cb renderCommandEncoderWithDescriptor:rp2];
      [e2 setRenderPipelineState:st.pLight];
      [e2 setFragmentTexture:st.g0 atIndex:0];
      [e2 setFragmentTexture:st.g1 atIndex:1];
      [e2 setFragmentTexture:st.g2 atIndex:2];
      [e2 setFragmentTexture:st.g3 atIndex:3];
      [e2 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e2 endEncoding];
   }
}

static void
deferred_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res(gb_param(o, "res", "1080p"), &w, &h);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h / 1e9 * 1e6);
   gb_derive_rate(r, "fps", "fps", us, 1e6);
}

static void
deferred_check(id state, struct gb_check *chk)
{
   GBDeferredState *st = state;
   uint8_t *got = malloc((size_t)st.w * st.h * 4u);
   mtl_tex_readback(st.finalTex, got, st.w, st.h, 4u);
   gb_deferred_check_fb(got, st.w, st.h, chk);
   free(got);
}

static const char *const deferred_params[] = {"mode", "res", "msaa", NULL};
static const struct mtest t_render_deferred = {
   .name = "render.deferred", .known = deferred_params, .depths = {1, 2}, .nd = 2,
   .setup = deferred_setup, .encode = deferred_encode, .derive = deferred_derive, .check = deferred_check,
};

/* ---- 6. concurrent.sram ---- */

static int
run_concurrent_sram(const struct gb_opts *o)
{
   static const char *const known[] = {"res", "msaa", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "res must be 720p, 1080p, 1440p or 2160p");
      return 1;
   }
   long msaa = gb_param_int(o, "msaa", 4);
   if (msaa != 1 && msaa != 4) {
      gb_result_simple(o, GB_MODE_QUEUES, 1, NULL, "fail", "msaa must be 1 or 4");
      return 1;
   }
   ctx_init();
   const int depth = GB_CR_DEPTH;
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   GBUberState *ub = uber_make_state(GB_UBER_SRAM);
   GBDeferredState *def = deferred_make_state(w, h, (uint32_t)msaa, true);
   GBStream *sc = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                           encode:^(id<MTLCommandBuffer> cb, uint64_t k) { uber_encode(ub, cb, k); }
                                             done:nil];
   GBStream *sr = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                           encode:^(id<MTLCommandBuffer> cb, uint64_t k) { deferred_encode(def, cb, k); }
                                             done:nil];
   struct gb_pass ca = {0}, ra = {0};
   stream_run(sc, false, gb_now_ns(), W, o->runs, R, &ca);
   stream_run(sr, false, gb_now_ns(), W, o->runs, R, &ra);
   uint64_t t0 = gb_now_ns() + 20000000ull;
   struct cr_thr th[2] = {{.s = (__bridge void *)sc, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
                          {.s = (__bridge void *)sr, .t0 = t0, .warm = W, .win = R, .nwin = o->runs}};
   for (int i = 0; i < 2; i++) pthread_create(&th[i].th, NULL, cr_thr_main, &th[i]);
   for (int i = 0; i < 2; i++) pthread_join(th[i].th, NULL);

   struct gb_check chk = {0};
   bool bad = ca.failed || ra.failed || th[0].pass.failed || th[1].pass.failed;
   if (!bad)
      uber_check(ub, &chk);

   struct gb_result r;
   const char *extra[3] = {"compute=uber.sram(16KB)", "render=deferred:tile", NULL};
   gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth, extra);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   struct gb_samples s[4] = {0};
   gb_pass_rate(&ca, &s[0]);
   gb_pass_rate(&ra, &s[1]);
   gb_pass_rate(&th[0].pass, &s[2]);
   gb_pass_rate(&th[1].pass, &s[3]);
   for (int i = 0; i < 4; i++) bad |= !s[i].n;
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
   } else {
      gb_result_metric(&r, "compute_alone_us", "us", true, &s[0], true);
      gb_result_metric(&r, "render_alone_us", "us", true, &s[1], true);
      gb_result_metric(&r, "compute_conc_us", "us", true, &s[2], true);
      gb_result_metric(&r, "render_conc_us", "us", true, &s[3], true);
      double ov = gb_median(&s[0]) / gb_median(&s[2]) + gb_median(&s[1]) / gb_median(&s[3]);
      gb_result_metric1(&r, "overlap", "x", false, ov);
      gb_result_end_metrics(&r);
      gb_result_check(&r, &chk);
      gb_result_finish(&r, chk.valid ? "ok" : "fail", "overlap",
                       chk.valid ? NULL : "compute output corrupted under SRAM contention");
   }
   for (int i = 0; i < 4; i++) gb_samples_free(&s[i]);
   gb_pass_free(&ca); gb_pass_free(&ra);
   gb_pass_free(&th[0].pass); gb_pass_free(&th[1].pass);
   return 0;
}

/* ---- 7. render.mdi (Metal ICB + bindless argument buffer) ---- */

@interface GBMdiState : NSObject
@property(nonatomic) uint32_t draws, bindless;
@property(nonatomic, strong) id<MTLTexture> target;
@property(nonatomic, strong) NSMutableArray<id<MTLTexture>> *texs;
@property(nonatomic, strong) id<MTLBuffer> vb, ib, argBuf;
@property(nonatomic, strong) id<MTLIndirectCommandBuffer> icb;
@property(nonatomic, strong) id<MTLRenderPipelineState> rp;
@property(nonatomic) const __unsafe_unretained id<MTLResource> *resPtrs;
@property(nonatomic) uint32_t *ref;
@property(nonatomic) uint32_t refCrc;
@end
@implementation GBMdiState
- (void)dealloc { free((void *)_resPtrs); free(_ref); }
@end

static id
mdi_setup(const struct gb_opts *o, char *err, size_t n)
{
   long draws = gb_param_int(o, "draws", 4096);
   long bindless = gb_param_int(o, "bindless", 1024);
   if (draws < 1 || draws > (long)GB_MDI_MAX_DRAWS) {
      snprintf(err, n, "draws must be 1..%u", GB_MDI_MAX_DRAWS);
      return nil;
   }
   if (bindless < 1 || bindless > (long)GB_MDI_MAX_BINDLESS) {
      snprintf(err, n, "bindless must be 1..%u", GB_MDI_MAX_BINDLESS);
      return nil;
   }
   GBMdiState *st = [GBMdiState new];
   st.draws = (uint32_t)draws;
   st.bindless = (uint32_t)bindless;
   st.target = tex_ex(GB_MDI_DIM, GB_MDI_DIM, MTLPixelFormatRGBA8Unorm, 1, MTLStorageModePrivate,
                      MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);

   const float verts[8] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
   const uint16_t idxs[6] = {0, 1, 2, 0, 2, 3};
   st.vb = buf(sizeof(verts), false);
   st.ib = buf(sizeof(idxs), false);
   mtl_buf_upload(st.vb, verts, sizeof(verts));
   mtl_buf_upload(st.ib, idxs, sizeof(idxs));

   st.texs = [NSMutableArray arrayWithCapacity:st.bindless];
   __unsafe_unretained id<MTLResource> *ptrs = (__unsafe_unretained id<MTLResource> *)calloc(st.bindless, sizeof(id<MTLResource>));
   size_t one_tex = GB_MDI_TEX_DIM * GB_MDI_TEX_DIM * sizeof(uint32_t);
   id<MTLBuffer> stg = buf(st.bindless * one_tex, true);
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
   for (uint32_t t = 0; t < st.bindless; t++) {
      id<MTLTexture> tx = tex_ex(GB_MDI_TEX_DIM, GB_MDI_TEX_DIM, MTLPixelFormatRGBA8Unorm, 1,
                                 MTLStorageModePrivate, MTLTextureUsageShaderRead);
      gb_mdi_fill_tex(t, (uint32_t *)((uint8_t *)stg.contents + t * one_tex));
      [bl copyFromBuffer:stg sourceOffset:t * one_tex
       sourceBytesPerRow:GB_MDI_TEX_DIM * 4 sourceBytesPerImage:one_tex
              sourceSize:MTLSizeMake(GB_MDI_TEX_DIM, GB_MDI_TEX_DIM, 1)
               toTexture:tx destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
      [st.texs addObject:tx];
      ptrs[t] = tx;
   }
   [bl endEncoding];
   [cb commit];
   [cb waitUntilCompleted];
   st.resPtrs = ptrs;

   id<MTLFunction> vf = [g_lib newFunctionWithName:@"mdi_vert"];
   id<MTLFunction> ff = [g_lib newFunctionWithName:@"mdi_frag"];
   id<MTLArgumentEncoder> ae = [ff newArgumentEncoderWithBufferIndex:0];
   st.argBuf = buf(ae.encodedLength, true);
   [ae setArgumentBuffer:st.argBuf offset:0];
   for (uint32_t t = 0; t < st.bindless; t++)
      [ae setTexture:st.texs[t] atIndex:t];

   MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
   vd.attributes[0].format = MTLVertexFormatFloat2;
   vd.attributes[0].offset = 0;
   vd.attributes[0].bufferIndex = 0;
   vd.layouts[0].stride = 8;

   MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
   rpd.vertexFunction = vf;
   rpd.fragmentFunction = ff;
   rpd.vertexDescriptor = vd;
   rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   rpd.supportIndirectCommandBuffers = YES;
   NSError *merr = nil;
   st.rp = [g_dev newRenderPipelineStateWithDescriptor:rpd error:&merr];
   if (!st.rp)
      gb_die("mdi pipeline failed: %s", merr.localizedDescription.UTF8String);

   MTLIndirectCommandBufferDescriptor *icbd = [MTLIndirectCommandBufferDescriptor new];
   icbd.commandTypes = MTLIndirectCommandTypeDrawIndexed;
   icbd.inheritBuffers = YES;
   icbd.inheritPipelineState = YES;
   st.icb = [g_dev newIndirectCommandBufferWithDescriptor:icbd maxCommandCount:st.draws options:SHARED];
   for (uint32_t d = 0; d < st.draws; d++) {
      id<MTLIndirectRenderCommand> cmd = [st.icb indirectRenderCommandAtIndex:d];
      [cmd drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:6
                       indexType:MTLIndexTypeUInt16
                     indexBuffer:st.ib
               indexBufferOffset:0
                   instanceCount:1
                      baseVertex:0
                    baseInstance:d];
   }

   st.ref = malloc((size_t)GB_MDI_DIM * GB_MDI_DIM * sizeof(uint32_t));
   st.refCrc = gb_mdi_ref_fb(st.draws, st.bindless, st.ref);
   return st;
}

static void
mdi_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBMdiState *st = state;
   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture = st.target;
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
   id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
   [e setRenderPipelineState:st.rp];
   [e setVertexBuffer:st.vb offset:0 atIndex:0];
   uint32_t bcnt = st.bindless;
   [e setVertexBytes:&bcnt length:4 atIndex:1];
   [e setFragmentBuffer:st.argBuf offset:0 atIndex:0];
   [e useResource:st.ib usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
   [e useResources:st.resPtrs count:st.bindless usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
   [e executeCommandsInBuffer:st.icb withRange:NSMakeRange(0, st.draws)];
   [e endEncoding];
}

static void
mdi_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long draws = gb_param_int(o, "draws", 4096);
   gb_derive_rate(r, "mdraws_s", "Mdraws/s", us, (double)draws);
}

static void
mdi_check(id state, struct gb_check *chk)
{
   GBMdiState *st = state;
   size_t count = (size_t)GB_MDI_DIM * GB_MDI_DIM;
   uint32_t *got = malloc(count * sizeof(uint32_t));
   mtl_tex_readback(st.target, got, GB_MDI_DIM, GB_MDI_DIM, 4u);
   gb_check_words(chk, got, st.ref, st.refCrc, count);
   free(got);
}

static const char *const mdi_params[] = {"draws", "bindless", NULL};
static const struct mtest t_render_mdi = {
   .name = "render.mdi", .known = mdi_params, .depths = {1, 4}, .nd = 2,
   .setup = mdi_setup, .encode = mdi_encode, .derive = mdi_derive, .check = mdi_check,
};

/* ---- 8. compile.pso ---- */

static int
run_compile_pso(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "count", "salt", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "uber");
   bool is_comp = !strcmp(kind, "compute");
   bool is_uber = !strcmp(kind, "uber");
   bool is_gfx = !strcmp(kind, "graphics");
   if (!is_comp && !is_uber && !is_gfx) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be compute, graphics or uber");
      return 1;
   }
   long count = gb_param_int(o, "count", (long)GB_PSO_DEFAULT_COUNT);
   long salt = gb_param_int(o, "salt", 1);
   if (count < 1 || count > 256 || (salt != 0 && salt != 1)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "count 1..256, salt 0|1");
      return 1;
   }
   ctx_init();
   uint32_t nonce = salt ? (uint32_t)((gb_now_ns() ^ ((uint64_t)getpid() << 16)) | 1u) : 0x51a71234u;

   NSError *err = nil;
   MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
   id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:ad error:&err];
   if (!ar)
      gb_die("MTLBinaryArchive failed: %s", err.localizedDescription.UTF8String);

   struct gb_samples cold_s = {0}, warm_s = {0};
   double cold_total = 0, warm_total = 0;
   id<MTLComputePipelineState> last_cp = nil;
   id<MTLRenderPipelineState> last_rp = nil;

   for (int phase = 0; phase < 2; phase++) {
      uint64_t t_batch0 = gb_now_ns();
      for (long i = 0; i < count; i++) {
         @autoreleasepool {
            uint32_t v0 = (uint32_t)(i + 1);
            uint32_t v1 = (uint32_t)(is_gfx ? (4u + (uint32_t)(i & 7)) : (8u + (uint32_t)(i & 15)));
            uint32_t v2 = (uint32_t)(is_uber ? 7u : 3u);
            uint32_t v3 = nonce + (uint32_t)i * 2654435761u;
            MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
            [fc setConstantValue:&v0 type:MTLDataTypeUInt atIndex:20];
            [fc setConstantValue:&v1 type:MTLDataTypeUInt atIndex:21];
            [fc setConstantValue:&v2 type:MTLDataTypeUInt atIndex:22];
            [fc setConstantValue:&v3 type:MTLDataTypeUInt atIndex:23];

            uint64_t t0 = gb_now_ns();
            if (!is_gfx) {
               id<MTLFunction> fn = [g_lib newFunctionWithName:(is_uber ? @"pso_uber_main" : @"pso_comp_main")
                                                constantValues:fc error:&err];
               if (!fn)
                  gb_die("compile.pso compute function failed: %s", err.localizedDescription.UTF8String);
               MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
               cd.computeFunction = fn;
               if (phase == 0) {
                  if (![ar addComputePipelineFunctionsWithDescriptor:cd error:&err])
                     gb_die("compile.pso addComputePipeline failed: %s", err.localizedDescription.UTF8String);
               } else {
                  cd.binaryArchives = @[ar];
                  last_cp = [g_dev newComputePipelineStateWithDescriptor:cd options:MTLPipelineOptionNone reflection:nil error:&err];
                  if (!last_cp)
                     gb_die("compile.pso warm compute pipeline failed: %s", err.localizedDescription.UTF8String);
               }
            } else {
               id<MTLFunction> vf = [g_lib newFunctionWithName:@"pso_gfx_vert" constantValues:fc error:&err];
               id<MTLFunction> ff = [g_lib newFunctionWithName:@"pso_gfx_frag" constantValues:fc error:&err];
               if (!vf || !ff)
                  gb_die("compile.pso graphics function failed: %s", err.localizedDescription.UTF8String);
               MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
               rd.vertexFunction = vf;
               rd.fragmentFunction = ff;
               rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
               if (phase == 0) {
                  if (![ar addRenderPipelineFunctionsWithDescriptor:rd error:&err])
                     gb_die("compile.pso addRenderPipeline failed: %s", err.localizedDescription.UTF8String);
               } else {
                  rd.binaryArchives = @[ar];
                  last_rp = [g_dev newRenderPipelineStateWithDescriptor:rd error:&err];
                  if (!last_rp)
                     gb_die("compile.pso warm render pipeline failed: %s", err.localizedDescription.UTF8String);
               }
            }
            uint64_t t1 = gb_now_ns();
            gb_samples_push(phase == 0 ? &cold_s : &warm_s, (double)(t1 - t0) / 1e6);
         }
      }
      uint64_t t_batch1 = gb_now_ns();
      if (phase == 0) cold_total = (double)(t_batch1 - t_batch0) / 1e6;
      else warm_total = (double)(t_batch1 - t_batch0) / 1e6;
   }

   /* Smoke-execute the last compiled pipeline. */
   id<MTLCommandQueue> q = [g_dev newCommandQueue];
   id<MTLCommandBuffer> cb = [q commandBuffer];
   if (!is_gfx && last_cp) {
      id<MTLBuffer> ob = buf(4096, true);
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:last_cp];
      [e setBuffer:ob offset:0 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [e endEncoding];
   } else if (is_gfx && last_rp) {
      id<MTLTexture> ot = tex(64, 64, MTLTextureUsageRenderTarget);
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass_desc(ot)];
      [e setRenderPipelineState:last_rp];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
   }
   [cb commit];
   [cb waitUntilCompleted];

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_result_metric(&r, "cold_per_pso_ms", "ms", true, &cold_s, true);
   gb_result_metric1(&r, "cold_total_ms", "ms", true, cold_total);
   gb_result_metric(&r, "warm_per_pso_ms", "ms", true, &warm_s, true);
   gb_result_metric1(&r, "warm_total_ms", "ms", true, warm_total);
   gb_result_metric1(&r, "cache_speedup", "x", false, warm_total > 0.0 ? cold_total / warm_total : 0.0);
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "pipelines", count);
   gb_json_obj_end(&r.j);
   gb_result_finish(&r, "ok", "cold_per_pso_ms", NULL);

   gb_samples_free(&cold_s);
   gb_samples_free(&warm_s);
   return 0;
}

/* ---- 9. compress.bw (mrt4 | rgba32f | msaa4 | array | storage) ---- */

enum { MCMP_MRT4, MCMP_RGBA32F, MCMP_MSAA4, MCMP_ARRAY, MCMP_STORAGE };

@interface GBCmpState : NSObject
@property(nonatomic) int kind;
@property(nonatomic) uint32_t w, h;
@property(nonatomic, strong) id<MTLTexture> g0, g1, g2, g3, msaaTex, arrTex, finalTex;
@property(nonatomic, strong) id<MTLRenderPipelineState> rpWrite, rpRead;
@property(nonatomic, strong) id<MTLComputePipelineState> cpWrite;
@end
@implementation GBCmpState
@end

static id<MTLRenderPipelineState>
cmp_rpipe(const char *fs, int n_att, const MTLPixelFormat *fmts, NSUInteger samples)
{
   MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
   d.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   d.fragmentFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
   d.rasterSampleCount = samples;
   for (int i = 0; i < n_att; i++)
      d.colorAttachments[i].pixelFormat = fmts[i];
   NSError *err = nil;
   id<MTLRenderPipelineState> p = [g_dev newRenderPipelineStateWithDescriptor:d error:&err];
   if (!p) gb_die("compress render pipeline %s: %s", fs, err.localizedDescription.UTF8String);
   return p;
}

static id
cmp_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "mrt4");
   int k;
   if (!strcmp(kind, "mrt4")) k = MCMP_MRT4;
   else if (!strcmp(kind, "rgba32f")) k = MCMP_RGBA32F;
   else if (!strcmp(kind, "msaa4")) k = MCMP_MSAA4;
   else if (!strcmp(kind, "array")) k = MCMP_ARRAY;
   else if (!strcmp(kind, "storage")) k = MCMP_STORAGE;
   else { snprintf(err, n, "kind must be mrt4, rgba32f, msaa4, array or storage"); return nil; }

   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1440p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   GBCmpState *st = [GBCmpState new];
   st.kind = k;
   if (k == MCMP_ARRAY) { st.w = w >= 2560u ? 1024u : 512u; st.h = st.w; }
   else { st.w = w; st.h = h; }

   st.finalTex = tex_ex(st.w, st.h, MTLPixelFormatRGBA8Unorm, 1, MTLStorageModePrivate,
                        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
   MTLPixelFormat rgba8 = MTLPixelFormatRGBA8Unorm;
   MTLPixelFormat rgba32f = MTLPixelFormatRGBA32Float;
   MTLPixelFormat mrt4[4] = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float,
                             MTLPixelFormatRGBA8Unorm, MTLPixelFormatR32Float};
   MTLTextureUsage rt_read = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

   if (k == MCMP_MRT4) {
      st.g0 = tex_ex_opt(st.w, st.h, mrt4[0], 1, MTLStorageModePrivate, rt_read, true);
      st.g1 = tex_ex_opt(st.w, st.h, mrt4[1], 1, MTLStorageModePrivate, rt_read, true);
      st.g2 = tex_ex_opt(st.w, st.h, mrt4[2], 1, MTLStorageModePrivate, rt_read, true);
      st.g3 = tex_ex_opt(st.w, st.h, mrt4[3], 1, MTLStorageModePrivate, rt_read, true);
      st.rpWrite = cmp_rpipe("compress_mrt4_frag", 4, mrt4, 1);
      st.rpRead = cmp_rpipe("compress_sample4_frag", 1, &rgba8, 1);
   } else if (k == MCMP_RGBA32F) {
      st.g0 = tex_ex_opt(st.w, st.h, rgba32f, 1, MTLStorageModePrivate, rt_read, true);
      st.rpWrite = cmp_rpipe("compress_write1_frag", 1, &rgba32f, 1);
      st.rpRead = cmp_rpipe("compress_sample1_frag", 1, &rgba8, 1);
   } else if (k == MCMP_MSAA4) {
      st.msaaTex = tex_ex(st.w, st.h, rgba8, 4, MTLStorageModeMemoryless, MTLTextureUsageRenderTarget);
      st.g0 = tex_ex_opt(st.w, st.h, rgba8, 1, MTLStorageModePrivate, rt_read, true);
      st.rpWrite = cmp_rpipe("compress_write1_frag", 1, &rgba8, 4);
      st.rpRead = cmp_rpipe("compress_sample1_frag", 1, &rgba8, 1);
   } else if (k == MCMP_ARRAY) {
      MTLTextureDescriptor *ad = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:rgba8 width:st.w height:st.h mipmapped:NO];
      ad.textureType = MTLTextureType2DArray;
      ad.arrayLength = 6;
      ad.storageMode = MTLStorageModePrivate;
      ad.usage = rt_read;
      ad.hazardTrackingMode = MTLHazardTrackingModeTracked;
      st.arrTex = [g_dev newTextureWithDescriptor:ad];
      if (!st.arrTex) gb_die("compress array texture failed");
      st.rpWrite = cmp_rpipe("compress_write1_frag", 1, &rgba8, 1);
      st.rpRead = cmp_rpipe("compress_array_frag", 1, &rgba8, 1);
   } else {
      st.g0 = tex_ex_opt(st.w, st.h, rgba8, 1, MTLStorageModePrivate,
                         MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead, true);
      st.cpWrite = cpipe("compress_store_main");
      st.rpRead = cmp_rpipe("compress_sample1_frag", 1, &rgba8, 1);
   }
   return st;
}

static void
cmp_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   GBCmpState *st = state;
   float kf = (float)(k & 15u) * 0.03125f;
   if (st.kind == MCMP_MRT4) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      id<MTLTexture> gs[4] = {st.g0, st.g1, st.g2, st.g3};
      for (int i = 0; i < 4; i++) {
         rp.colorAttachments[i].texture = gs[i];
         rp.colorAttachments[i].loadAction = MTLLoadActionDontCare;
         rp.colorAttachments[i].storeAction = MTLStoreActionStore;
      }
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      float pc[4] = {kf, 0.25f, 0.5f, 0.0f};
      [e setRenderPipelineState:st.rpWrite];
      [e setFragmentBytes:pc length:16 atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
   } else if (st.kind == MCMP_RGBA32F || st.kind == MCMP_MSAA4) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      if (st.kind == MCMP_MSAA4) {
         rp.colorAttachments[0].texture = st.msaaTex;
         rp.colorAttachments[0].resolveTexture = st.g0;
         rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
         rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      } else {
         rp.colorAttachments[0].texture = st.g0;
         rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
         rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      float pc[4] = {kf, 0.25f, 0.5f, 0.0f};
      [e setRenderPipelineState:st.rpWrite];
      [e setFragmentBytes:pc length:16 atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
   } else if (st.kind == MCMP_ARRAY) {
      for (uint32_t l = 0; l < 6; l++) {
         MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
         rp.colorAttachments[0].texture = st.arrTex;
         rp.colorAttachments[0].slice = l;
         rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
         rp.colorAttachments[0].storeAction = MTLStoreActionStore;
         id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
         float pc[4] = {kf, (float)l * 0.125f, (float)l * 0.0625f, 0.0f};
         [e setRenderPipelineState:st.rpWrite];
         [e setFragmentBytes:pc length:16 atIndex:0];
         [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
         [e endEncoding];
      }
   } else {
      struct { int32_t w, h; float k, pad; } pc = {(int32_t)st.w, (int32_t)st.h, kf, 0.0f};
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:st.cpWrite];
      [e setTexture:st.g0 atIndex:0];
      [e setBytes:&pc length:sizeof(pc) atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake((st.w + 15u) / 16u, (st.h + 15u) / 16u, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [e endEncoding];
   }

   MTLRenderPassDescriptor *rf = [MTLRenderPassDescriptor renderPassDescriptor];
   rf.colorAttachments[0].texture = st.finalTex;
   rf.colorAttachments[0].loadAction = MTLLoadActionDontCare;
   rf.colorAttachments[0].storeAction = MTLStoreActionStore;
   id<MTLRenderCommandEncoder> ef = [cb renderCommandEncoderWithDescriptor:rf];
   [ef setRenderPipelineState:st.rpRead];
   if (st.kind == MCMP_MRT4) {
      [ef setFragmentTexture:st.g0 atIndex:0];
      [ef setFragmentTexture:st.g1 atIndex:1];
      [ef setFragmentTexture:st.g2 atIndex:2];
      [ef setFragmentTexture:st.g3 atIndex:3];
   } else if (st.kind == MCMP_ARRAY) {
      [ef setFragmentTexture:st.arrTex atIndex:0];
   } else {
      [ef setFragmentTexture:st.g0 atIndex:0];
   }
   [ef drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
   [ef endEncoding];
}

static void
cmp_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   const char *kind = gb_param(o, "kind", "mrt4");
   uint32_t w = 2560, h = 1440;
   parse_res(gb_param(o, "res", "1440p"), &w, &h);
   double b = 0;
   if (!strcmp(kind, "mrt4")) b = 44.0 * w * h;
   else if (!strcmp(kind, "rgba32f")) b = 36.0 * w * h;
   else if (!strcmp(kind, "msaa4")) b = 28.0 * w * h;
   else if (!strcmp(kind, "array")) { double d = w >= 2560u ? 1024.0 : 512.0; b = 52.0 * d * d; }
   else b = 12.0 * w * h;
   gb_derive_rate(r, "gbps", "GB/s", us, b / 1e9 * 1e6);
}

static void
cmp_check(id state, struct gb_check *chk)
{
   GBCmpState *st = state;
   size_t bytes = (size_t)st.w * st.h * 4u;
   uint8_t *got = malloc(bytes);
   mtl_tex_readback(st.finalTex, got, st.w, st.h, 4u);
   chk->done = true;
   chk->crc = gb_crc32(got, bytes);
   chk->expected = chk->crc;
   chk->valid = true;
   uint32_t nonzero = 0;
   for (uint32_t y = 0; y < st.h; y += st.h / 8u)
      for (uint32_t x = 0; x < st.w; x += st.w / 8u) {
         const uint8_t *px = got + ((size_t)y * st.w + x) * 4u;
         if (px[3] != 255u) chk->valid = false;
         if (px[0] || px[1] || px[2]) nonzero++;
      }
   if (nonzero < 32u) { chk->valid = false; chk->expected = 0u; }
   free(got);
}

static const char *const cmp_params[] = {"kind", "res", NULL};
static const struct mtest t_compress_bw = {
   .name = "compress.bw", .known = cmp_params, .depths = {1, 4}, .nd = 2,
   .setup = cmp_setup, .encode = cmp_encode, .derive = cmp_derive, .check = cmp_check,
};

/* ---- 10. overlap.passes ---- */

static void
ov_enc_comp(id<MTLComputePipelineState> cp, id<MTLBuffer> b, id<MTLComputeCommandEncoder> e)
{
   struct { uint32_t iters; float a, b; } pc = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B};
   [e setComputePipelineState:cp];
   [e setBuffer:b offset:0 atIndex:0];
   [e setBytes:&pc length:sizeof(pc) atIndex:1];
   [e dispatchThreadgroups:MTLSizeMake(GB_ALU_GROUPS / 2u, 1, 1) threadsPerThreadgroup:MTLSizeMake(GB_ALU_LOCAL, 1, 1)];
}

static void
ov_enc_render(id<MTLRenderPipelineState> rp, id<MTLTexture> t, id<MTLCommandBuffer> cb, uint64_t k,
              id<MTLFence> waitF, id<MTLFence> sigF)
{
   float color[4] = {(float)(k & 255u) / 255.0f, 0.5f, 0.25f, 0.5f};
   id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass_desc(t)];
   if (waitF) [e waitForFence:waitF beforeStages:MTLRenderStageVertex];
   [e setRenderPipelineState:rp];
   [e setFragmentBytes:color length:16 atIndex:0];
   [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:GB_FILL_LAYERS / 2u];
   if (sigF) [e updateFence:sigF afterStages:MTLRenderStageFragment];
   [e endEncoding];
}

static int
run_overlap_passes(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "comp_comp");
   int k = !strcmp(kind, "comp_comp") ? 0 : !strcmp(kind, "comp_render") ? 1
           : !strcmp(kind, "render_render") ? 2 : !strcmp(kind, "async_cr") ? 3 : -1;
   if (k < 0) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be comp_comp, comp_render, render_render or async_cr");
      return 1;
   }
   ctx_init();
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   id<MTLComputePipelineState> cp = cpipe("alu_main");
   id<MTLRenderPipelineState> rp = rpipe("fs_vert", "color_frag", true);
   size_t cbytes = (size_t)(GB_ALU_GROUPS / 2u) * GB_ALU_LOCAL * 4u;
   id<MTLBuffer> ba = buf(cbytes, false), bb = buf(cbytes, false);
   id<MTLTexture> ia = tex(1920, 1080, MTLTextureUsageRenderTarget);
   id<MTLTexture> ib = tex(1920, 1080, MTLTextureUsageRenderTarget);
   NSMutableArray<id<MTLFence>> *fences = [NSMutableArray arrayWithCapacity:8];
   for (int i = 0; i < 8; i++)
      [fences addObject:[g_dev newFence]];

   struct gb_pass pa = {0}, pb = {0}, pnb = {0}, pbar = {0};
   if (k == 3) {
      const int depth = GB_CR_DEPTH;
      GBStream *sa = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                              encode:^(id<MTLCommandBuffer> cb, uint64_t seq) {
         id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
         ov_enc_comp(cp, ba, e); [e endEncoding];
      } done:nil];
      GBStream *sb = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                              encode:^(id<MTLCommandBuffer> cb, uint64_t seq) {
         ov_enc_render(rp, ib, cb, seq, nil, nil);
      } done:nil];
      GBStream *sbar = [[GBStream alloc] initWithQueue:new_queue(depth) depth:depth
                                                encode:^(id<MTLCommandBuffer> cb, uint64_t seq) {
         id<MTLFence> f = fences[seq & 7];
         id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
         ov_enc_comp(cp, ba, e); [e updateFence:f]; [e endEncoding];
         ov_enc_render(rp, ib, cb, seq, f, nil);
      } done:nil];
      stream_run(sa, false, gb_now_ns(), W, o->runs, R, &pa);
      stream_run(sb, false, gb_now_ns(), W, o->runs, R, &pb);
      uint64_t t0 = gb_now_ns() + 20000000ull;
      struct cr_thr th[2] = {{.s = (__bridge void *)sa, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
                             {.s = (__bridge void *)sb, .t0 = t0, .warm = W, .win = R, .nwin = o->runs}};
      for (int i = 0; i < 2; i++) pthread_create(&th[i].th, NULL, cr_thr_main, &th[i]);
      for (int i = 0; i < 2; i++) pthread_join(th[i].th, NULL);
      stream_run(sbar, false, gb_now_ns(), W, o->runs, R, &pbar);

      struct gb_samples ra_s = {0}, rb_s = {0}, ca_s = {0}, cb_s = {0}, bar_s = {0};
      gb_pass_rate(&pa, &ra_s); gb_pass_rate(&pb, &rb_s);
      gb_pass_rate(&th[0].pass, &ca_s); gb_pass_rate(&th[1].pass, &cb_s);
      gb_pass_rate(&pbar, &bar_s);
      bool bad = pa.failed || pb.failed || th[0].pass.failed || th[1].pass.failed || pbar.failed ||
                 !ra_s.n || !rb_s.n || !ca_s.n || !cb_s.n || !bar_s.n;
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth, NULL);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad) {
         gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
      } else {
         double ma = gb_median(&ra_s), mb = gb_median(&rb_s);
         double mca = gb_median(&ca_s), mcb = gb_median(&cb_s);
         double mnb = 1.0 / ((1.0 / mca) * (ma / (ma + mb)) + (1.0 / mcb) * (mb / (ma + mb)));
         double ov = ma / mca + mb / mcb;
         double mbar = gb_median(&bar_s);
         gb_result_metric(&r, "solo_a_us", "us", true, &ra_s, true);
         gb_result_metric(&r, "solo_b_us", "us", true, &rb_s, true);
         gb_result_metric(&r, "conc_a_us", "us", true, &ca_s, true);
         gb_result_metric(&r, "conc_b_us", "us", true, &cb_s, true);
         gb_result_metric1(&r, "nobarrier_us", "us", true, mnb);
         gb_result_metric(&r, "barrier_us", "us", true, &bar_s, true);
         gb_result_metric1(&r, "overlap_ratio", "x", false, ov);
         gb_result_metric1(&r, "barrier_penalty_us", "us", true, mbar - mnb);
         gb_result_metric1(&r, "serialization_gap", "x", false, mnb > 0.0 ? mbar / mnb : 0.0);
         gb_result_finish(&r, "ok", "overlap_ratio", NULL);
      }
      gb_samples_free(&ra_s); gb_samples_free(&rb_s); gb_samples_free(&ca_s);
      gb_samples_free(&cb_s); gb_samples_free(&bar_s);
      gb_pass_free(&pa); gb_pass_free(&pb); gb_pass_free(&th[0].pass);
      gb_pass_free(&th[1].pass); gb_pass_free(&pbar);
      return 0;
   }

   __block int phase = 0;
   GBStream *s = [[GBStream alloc] initWithQueue:new_queue(2) depth:2
                                          encode:^(id<MTLCommandBuffer> cb, uint64_t seq) {
      bool run_a = phase != 1, run_b = phase != 0, bar = phase == 3;
      if (k == 0) {
         id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:(bar ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent)];
         if (run_a) ov_enc_comp(cp, ba, e);
         if (bar) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
         if (run_b) ov_enc_comp(cp, bb, e);
         [e endEncoding];
      } else if (k == 1) {
         id<MTLFence> f = bar ? fences[seq & 7] : nil;
         if (run_a) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            ov_enc_comp(cp, ba, e);
            if (f) [e updateFence:f];
            [e endEncoding];
         }
         if (run_b) ov_enc_render(rp, ib, cb, seq, f, nil);
      } else {
         id<MTLFence> f = bar ? fences[seq & 7] : nil;
         if (run_a) ov_enc_render(rp, ia, cb, seq, nil, f);
         if (run_b) ov_enc_render(rp, ib, cb, seq + 1, f, nil);
      }
   } done:nil];
   phase = 0; stream_run(s, false, gb_now_ns(), W, o->runs, R, &pa);
   phase = 1; stream_run(s, false, gb_now_ns(), W, o->runs, R, &pb);
   phase = 2; stream_run(s, false, gb_now_ns(), W, o->runs, R, &pnb);
   phase = 3; stream_run(s, false, gb_now_ns(), W, o->runs, R, &pbar);

   struct gb_samples sa_s = {0}, sb_s = {0}, snb_s = {0}, sbar_s = {0};
   gb_pass_rate(&pa, &sa_s); gb_pass_rate(&pb, &sb_s);
   gb_pass_rate(&pnb, &snb_s); gb_pass_rate(&pbar, &sbar_s);
   bool bad = pa.failed || pb.failed || pnb.failed || pbar.failed || !sa_s.n || !sb_s.n || !snb_s.n || !sbar_s.n;

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 2, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "command buffer error or no completions");
   } else {
      double ma = gb_median(&sa_s), mb = gb_median(&sb_s);
      double mnb = gb_median(&snb_s), mbar = gb_median(&sbar_s);
      gb_result_metric(&r, "solo_a_us", "us", true, &sa_s, true);
      gb_result_metric(&r, "solo_b_us", "us", true, &sb_s, true);
      gb_result_metric(&r, "nobarrier_us", "us", true, &snb_s, true);
      gb_result_metric(&r, "barrier_us", "us", true, &sbar_s, true);
      gb_result_metric1(&r, "overlap_ratio", "x", false, mnb > 0.0 ? (ma + mb) / mnb : 0.0);
      gb_result_metric1(&r, "barrier_penalty_us", "us", true, mbar - mnb);
      gb_result_metric1(&r, "serialization_gap", "x", false, mnb > 0.0 ? mbar / mnb : 0.0);
      gb_result_finish(&r, "ok", "overlap_ratio", NULL);
   }
   gb_samples_free(&sa_s); gb_samples_free(&sb_s); gb_samples_free(&snb_s); gb_samples_free(&sbar_s);
   gb_pass_free(&pa); gb_pass_free(&pb); gb_pass_free(&pnb); gb_pass_free(&pbar);
   return 0;
}

/* ======================= 90%-coverage Wave-2 benchmarks ======================= */

static void
mtl_set_crc_check(struct gb_check *chk, uint32_t got, uint32_t expected)
{
   chk->done = true;
   chk->crc = got;
   chk->expected = expected;
   chk->valid = (got == expected);
}

/* ---- W2.1 render.shadow ---- */

@interface GBShadowState : NSObject
@property(nonatomic) bool alphaTest;
@property(nonatomic) uint32_t res, inst;
@property(nonatomic, strong) id<MTLTexture> depthTex, recvTex;
@property(nonatomic, strong) id<MTLSamplerState> cmpSampler;
@property(nonatomic, strong) id<MTLDepthStencilState> dsLess;
@property(nonatomic, strong) id<MTLRenderPipelineState> pDepth, pRecv;
@end
@implementation GBShadowState @end

static id
shadow_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "depth_only");
   bool alpha = !strcmp(mode, "alpha_test");
   if (!alpha && strcmp(mode, "depth_only")) {
      snprintf(err, n, "mode must be depth_only or alpha_test");
      return nil;
   }
   const char *fmt_s = gb_param(o, "fmt", "d32f");
   MTLPixelFormat dfmt;
   if (!strcmp(fmt_s, "d32f")) dfmt = MTLPixelFormatDepth32Float;
   else if (!strcmp(fmt_s, "d16")) dfmt = MTLPixelFormatDepth16Unorm;
   else {
      snprintf(err, n, "fmt must be d32f or d16");
      return nil;
   }
   long res = gb_param_int(o, "res", 2048);
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   if (res < 256 || res > 4096 || inst < 1024 || inst > 16384) {
      snprintf(err, n, "res 256..4096, inst 1024..16384");
      return nil;
   }

   GBShadowState *st = [GBShadowState new];
   st.alphaTest = alpha;
   st.res = (uint32_t)res;
   st.inst = (uint32_t)inst;

   MTLTextureDescriptor *dd = [MTLTextureDescriptor new];
   dd.textureType = MTLTextureType2DArray;
   dd.pixelFormat = dfmt;
   dd.width = st.res;
   dd.height = st.res;
   dd.arrayLength = GB_SHADOW_CASCADES;
   dd.storageMode = MTLStorageModePrivate;
   dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
   dd.hazardTrackingMode = MTLHazardTrackingModeTracked;
   st.depthTex = [g_dev newTextureWithDescriptor:dd];
   if (!st.depthTex) gb_die("shadow depth array failed");

   st.recvTex = tex_ex_opt(GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, MTLPixelFormatRGBA8Unorm, 1,
                           MTLStorageModePrivate, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, true);

   MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
   sd.minFilter = MTLSamplerMinMagFilterLinear;
   sd.magFilter = MTLSamplerMinMagFilterLinear;
   sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
   sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
   sd.compareFunction = MTLCompareFunctionLess;
   st.cmpSampler = [g_dev newSamplerStateWithDescriptor:sd];

   MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
   dsd.depthCompareFunction = MTLCompareFunctionLess;
   dsd.depthWriteEnabled = YES;
   st.dsLess = [g_dev newDepthStencilStateWithDescriptor:dsd];

   NSError *merr = nil;
   MTLRenderPipelineDescriptor *dp = [MTLRenderPipelineDescriptor new];
   dp.vertexFunction = [g_lib newFunctionWithName:@"shadow_depth_vert"];
   dp.fragmentFunction = alpha ? [g_lib newFunctionWithName:@"shadow_alpha_frag"] : nil;
   dp.depthAttachmentPixelFormat = dfmt;
   st.pDepth = [g_dev newRenderPipelineStateWithDescriptor:dp error:&merr];
   if (!st.pDepth) gb_die("shadow depth pipe failed: %s", merr.localizedDescription.UTF8String);

   MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
   rp.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   rp.fragmentFunction = [g_lib newFunctionWithName:@"shadow_pcf_frag"];
   rp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   st.pRecv = [g_dev newRenderPipelineStateWithDescriptor:rp error:&merr];
   if (!st.pRecv) gb_die("shadow recv pipe failed: %s", merr.localizedDescription.UTF8String);
   return st;
}

static void
shadow_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBShadowState *st = state;
   for (uint32_t c = 0; c < GB_SHADOW_CASCADES; c++) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = st.depthTex;
      rp.depthAttachment.slice = c;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.depthAttachment.clearDepth = 1.0;
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      [e setRenderPipelineState:st.pDepth];
      [e setDepthStencilState:st.dsLess];
      uint32_t pc[2] = {c, st.res};
      [e setVertexBytes:pc length:sizeof(pc) atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:st.inst];
      [e endEncoding];
   }
   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture = st.recvTex;
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
   id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
   [e setRenderPipelineState:st.pRecv];
   [e setFragmentTexture:st.depthTex atIndex:0];
   [e setFragmentSamplerState:st.cmpSampler atIndex:0];
   float inv_res = 1.0f / (float)st.res;
   [e setFragmentBytes:&inv_res length:sizeof(inv_res) atIndex:0];
   [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
   [e endEncoding];
}

static void
shadow_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   long res = gb_param_int(o, "res", 2048);
   double tris = (double)GB_SHADOW_CASCADES * (double)inst * 2.0;
   double pix = (double)GB_SHADOW_CASCADES * (double)res * (double)res +
                (double)GB_SHADOW_RECV_DIM * (double)GB_SHADOW_RECV_DIM;
   gb_derive_rate(r, "mtri_s", "Mtri/s", us, tris);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, pix / 1e3);
}

static void
shadow_check(id state, struct gb_check *chk)
{
   GBShadowState *st = state;
   uint8_t *got = malloc((size_t)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM * 4u);
   mtl_tex_readback(st.recvTex, got, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, 4u);
   gb_shadow_check_fb(got, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, st.alphaTest, chk);
   free(got);
}

static const char *const shadow_params[] = {"mode", "fmt", "res", "inst", NULL};
static const struct mtest t_render_shadow = {
   .name = "render.shadow", .known = shadow_params, .depths = {1, 2}, .nd = 2,
   .setup = shadow_setup, .encode = shadow_encode, .derive = shadow_derive, .check = shadow_check,
};

/* ---- W2.2 compute.cull_mdi ---- */

@interface GBCullMdiState : NSObject
@property(nonatomic) uint32_t objs, visiblePct, expectedVis, expectedCrc;
@property(nonatomic, strong) id<MTLComputePipelineState> pCull;
@property(nonatomic, strong) id<MTLRenderPipelineState> pDraw;
@property(nonatomic, strong) id<MTLBuffer> objsBuf, icbArgBuf, rangeBuf, ibBuf;
@property(nonatomic, strong) id<MTLIndirectCommandBuffer> icb;
@property(nonatomic, strong) id<MTLTexture> fbTex;
@end
@implementation GBCullMdiState @end

static id
cull_mdi_setup(const struct gb_opts *o, char *err, size_t n)
{
   long objs = gb_param_int(o, "objs", GB_CULL_MAX_OBJS);
   long vis = gb_param_int(o, "visible", 25);
   if (objs < 64 || objs > (long)GB_CULL_MAX_OBJS || vis < 0 || vis > 100) {
      snprintf(err, n, "objs 64..16384, visible 0..100");
      return nil;
   }
   GBCullMdiState *st = [GBCullMdiState new];
   st.objs = (uint32_t)objs;
   st.visiblePct = (uint32_t)vis;

   struct gb_cull_obj *host = calloc(st.objs, sizeof(*host));
   st.expectedVis = gb_cull_mdi_fill_objects(st.objs, st.visiblePct, host);
   uint32_t *ref = malloc((size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   st.expectedCrc = gb_cull_mdi_ref_fb(host, st.objs, ref);
   free(ref);

   st.objsBuf = buf_tracked((size_t)st.objs * sizeof(struct gb_cull_obj), false);
   mtl_buf_upload(st.objsBuf, host, (size_t)st.objs * sizeof(struct gb_cull_obj));
   free(host);

   static const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
   st.ibBuf = buf_tracked(sizeof(idx), false);
   mtl_buf_upload(st.ibBuf, idx, sizeof(idx));
   st.rangeBuf = buf_tracked(16, false);

   MTLIndirectCommandBufferDescriptor *idsc = [MTLIndirectCommandBufferDescriptor new];
   idsc.commandTypes = MTLIndirectCommandTypeDrawIndexed;
   idsc.inheritBuffers = YES;
   idsc.inheritPipelineState = YES;
   st.icb = [g_dev newIndirectCommandBufferWithDescriptor:idsc
                                          maxCommandCount:st.objs
                                                  options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
   if (!st.icb) gb_die("cull_mdi ICB allocation failed");

   id<MTLFunction> cf = [g_lib newFunctionWithName:@"cull_mdi_comp"];
   if (!cf) gb_die("missing cull_mdi_comp");
   NSError *merr = nil;
   st.pCull = [g_dev newComputePipelineStateWithFunction:cf error:&merr];
   if (!st.pCull) gb_die("cull_mdi compute pipe failed: %s", merr.localizedDescription.UTF8String);

   id<MTLArgumentEncoder> ae = [cf newArgumentEncoderWithBufferIndex:1];
   st.icbArgBuf = buf_tracked(ae.encodedLength, true);
   [ae setArgumentBuffer:st.icbArgBuf offset:0];
   [ae setIndirectCommandBuffer:st.icb atIndex:0];

   MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
   rp.vertexFunction = [g_lib newFunctionWithName:@"cull_mdi_vert"];
   rp.fragmentFunction = [g_lib newFunctionWithName:@"cull_mdi_frag"];
   rp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   rp.supportIndirectCommandBuffers = YES;
   st.pDraw = [g_dev newRenderPipelineStateWithDescriptor:rp error:&merr];
   if (!st.pDraw) gb_die("cull_mdi render pipe failed: %s", merr.localizedDescription.UTF8String);

   st.fbTex = tex_ex_opt(GB_CULL_DIM, GB_CULL_DIM, MTLPixelFormatRGBA8Unorm, 1,
                         MTLStorageModePrivate, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, true);
   return st;
}

static void
cull_mdi_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBCullMdiState *st = state;
   id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
   [b fillBuffer:st.rangeBuf range:NSMakeRange(0, 8) value:0];
   [b resetCommandsInBuffer:st.icb withRange:NSMakeRange(0, st.objs)];
   [b endEncoding];

   id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
   [ce setComputePipelineState:st.pCull];
   [ce useResource:st.icb usage:MTLResourceUsageWrite];
   [ce setBuffer:st.objsBuf offset:0 atIndex:0];
   [ce setBuffer:st.icbArgBuf offset:0 atIndex:1];
   [ce setBuffer:st.rangeBuf offset:0 atIndex:2];
   [ce setBuffer:st.ibBuf offset:0 atIndex:3];
   uint32_t n_objs = st.objs;
   [ce setBytes:&n_objs length:4 atIndex:4];
   [ce dispatchThreadgroups:MTLSizeMake((st.objs + GB_CULL_LOCAL - 1u) / GB_CULL_LOCAL, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(GB_CULL_LOCAL, 1, 1)];
   [ce endEncoding];

   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture = st.fbTex;
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
   id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
   [re setRenderPipelineState:st.pDraw];
   [re setVertexBuffer:st.objsBuf offset:0 atIndex:0];
   [re useResource:st.icb usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
   [re useResource:st.ibBuf usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
   [re executeCommandsInBuffer:st.icb indirectBuffer:st.rangeBuf indirectBufferOffset:0];
   [re endEncoding];
}

static void
cull_mdi_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long objs = gb_param_int(o, "objs", GB_CULL_MAX_OBJS);
   long vis = gb_param_int(o, "visible", 25);
   gb_derive_rate(r, "mobjs_s", "Mobjs/s", us, (double)objs);
   gb_derive_rate(r, "mdraws_s", "Mdraws/s", us, (double)objs * (double)vis / 100.0);
}

static void
cull_mdi_check(id state, struct gb_check *chk)
{
   GBCullMdiState *st = state;
   uint32_t rng[2] = {0, 0};
   mtl_buf_readback(st.rangeBuf, rng, sizeof(rng));
   uint8_t *fb = malloc((size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   mtl_tex_readback(st.fbTex, fb, GB_CULL_DIM, GB_CULL_DIM, 4u);
   uint32_t got = gb_crc32(fb, (size_t)GB_CULL_DIM * GB_CULL_DIM * 4u);
   free(fb);
   mtl_set_crc_check(chk, got, st.expectedCrc);
   if (rng[1] != st.expectedVis) {
      chk->valid = false;
      gb_log("compute.cull_mdi visible count mismatch: got=%u expected=%u", rng[1], st.expectedVis);
   }
}

static const char *const cull_mdi_params[] = {"objs", "visible", NULL};
static const struct mtest t_compute_cull_mdi = {
   .name = "compute.cull_mdi", .known = cull_mdi_params, .depths = {1, 2}, .nd = 2,
   .setup = cull_mdi_setup, .encode = cull_mdi_encode, .derive = cull_mdi_derive, .check = cull_mdi_check,
};

/* ---- W2.3 render.bindless_ubo ---- */

@interface GBBuboState : NSObject
@property(nonatomic) uint32_t draws, expectedCrc;
@property(nonatomic, strong) id<MTLRenderPipelineState> pipe;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *banks;
@property(nonatomic, strong) id<MTLBuffer> argBuf;
@property(nonatomic, strong) id<MTLHeap> aliasHeap;
@property(nonatomic, strong) id<MTLTexture> rtAlias, rtFinal;
@end
@implementation GBBuboState @end

static id
bubo_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "ubo_vec4");
   if (strcmp(mode, "ubo_vec4") && strcmp(mode, "desc_buf")) {
      snprintf(err, n, "mode must be ubo_vec4 or desc_buf");
      return nil;
   }
   long draws = gb_param_int(o, "draws", GB_BUBO_MAX_DRAWS);
   if (draws < 1 || draws > (long)GB_BUBO_MAX_DRAWS) {
      snprintf(err, n, "draws must be 1..2048");
      return nil;
   }
   GBBuboState *st = [GBBuboState new];
   st.draws = (uint32_t)draws;
   uint32_t *ref = malloc((size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   st.expectedCrc = gb_bubo_ref_fb(st.draws, ref);
   free(ref);

   NSMutableArray<id<MTLBuffer>> *banks = [NSMutableArray arrayWithCapacity:GB_BUBO_BANKS];
   float *tmp = malloc(GB_BUBO_ENTRIES * 4u * sizeof(float));
   for (uint32_t b = 0; b < GB_BUBO_BANKS; b++) {
      gb_bubo_fill_bank(b, tmp);
      id<MTLBuffer> ub = buf_tracked(GB_BUBO_ENTRIES * 16u, false);
      mtl_buf_upload(ub, tmp, GB_BUBO_ENTRIES * 16u);
      [banks addObject:ub];
   }
   free(tmp);
   st.banks = banks;

   id<MTLFunction> ff = [g_lib newFunctionWithName:@"bubo_frag"];
   id<MTLArgumentEncoder> ae = [ff newArgumentEncoderWithBufferIndex:0];
   st.argBuf = buf_tracked(ae.encodedLength, true);
   [ae setArgumentBuffer:st.argBuf offset:0];
   for (uint32_t b = 0; b < GB_BUBO_BANKS; b++)
      [ae setBuffer:banks[b] offset:0 atIndex:b];

   MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:GB_BUBO_DIM
                                                                                height:GB_BUBO_DIM
                                                                             mipmapped:NO];
   td.storageMode = MTLStorageModePrivate;
   td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
   td.hazardTrackingMode = MTLHazardTrackingModeTracked;
   MTLSizeAndAlign sa = [g_dev heapTextureSizeAndAlignWithDescriptor:td];
   MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
   hd.type = MTLHeapTypePlacement;
   hd.size = sa.size + sa.align;
   hd.storageMode = MTLStorageModePrivate;
   hd.hazardTrackingMode = MTLHazardTrackingModeTracked;
   st.aliasHeap = [g_dev newHeapWithDescriptor:hd];
   if (!st.aliasHeap) gb_die("bubo alias heap failed");
   st.rtAlias = [st.aliasHeap newTextureWithDescriptor:td offset:0];
   st.rtFinal = [st.aliasHeap newTextureWithDescriptor:td offset:0];
   if (!st.rtAlias || !st.rtFinal) gb_die("bubo aliased textures failed");

   MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
   rp.vertexFunction = [g_lib newFunctionWithName:@"bubo_vert"];
   rp.fragmentFunction = ff;
   rp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   NSError *merr = nil;
   st.pipe = [g_dev newRenderPipelineStateWithDescriptor:rp error:&merr];
   if (!st.pipe) gb_die("bubo pipe failed: %s", merr.localizedDescription.UTF8String);
   return st;
}

static void
bubo_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBBuboState *st = state;
   __unsafe_unretained id<MTLResource> res[GB_BUBO_BANKS];
   for (uint32_t b = 0; b < GB_BUBO_BANKS; b++)
      res[b] = st.banks[b];

   MTLRenderPassDescriptor *rp1 = [MTLRenderPassDescriptor renderPassDescriptor];
   rp1.colorAttachments[0].texture = st.rtAlias;
   rp1.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp1.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp1.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
   id<MTLRenderCommandEncoder> e1 = [cb renderCommandEncoderWithDescriptor:rp1];
   [e1 setRenderPipelineState:st.pipe];
   [e1 setFragmentBuffer:st.argBuf offset:0 atIndex:0];
   [e1 useResources:res count:GB_BUBO_BANKS usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
   uint32_t wdraws = st.draws > 64u ? 64u : st.draws;
   for (uint32_t d = 0; d < wdraws; d++) {
      [e1 setVertexBytes:&d length:4 atIndex:0];
      [e1 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
   }
   [e1 endEncoding];

   MTLRenderPassDescriptor *rp2 = [MTLRenderPassDescriptor renderPassDescriptor];
   rp2.colorAttachments[0].texture = st.rtFinal;
   rp2.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp2.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp2.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
   id<MTLRenderCommandEncoder> e2 = [cb renderCommandEncoderWithDescriptor:rp2];
   [e2 setRenderPipelineState:st.pipe];
   [e2 setFragmentBuffer:st.argBuf offset:0 atIndex:0];
   [e2 useResources:res count:GB_BUBO_BANKS usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
   for (uint32_t d = 0; d < st.draws; d++) {
      [e2 setVertexBytes:&d length:4 atIndex:0];
      [e2 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
   }
   [e2 endEncoding];
}

static void
bubo_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   long draws = gb_param_int(o, "draws", GB_BUBO_MAX_DRAWS);
   gb_derive_rate(r, "kdraws_s", "kdraws/s", us, (double)draws * 1e3);
}

static void
bubo_check(id state, struct gb_check *chk)
{
   GBBuboState *st = state;
   uint8_t *fb = malloc((size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   mtl_tex_readback(st.rtFinal, fb, GB_BUBO_DIM, GB_BUBO_DIM, 4u);
   uint32_t got = gb_crc32(fb, (size_t)GB_BUBO_DIM * GB_BUBO_DIM * 4u);
   free(fb);
   mtl_set_crc_check(chk, got, st.expectedCrc);
}

static const char *const bubo_params[] = {"mode", "draws", NULL};
static const struct mtest t_render_bindless_ubo = {
   .name = "render.bindless_ubo", .known = bubo_params, .depths = {1, 2}, .nd = 2,
   .setup = bubo_setup, .encode = bubo_encode, .derive = bubo_derive, .check = bubo_check,
};

/* ---- W2.4 compute.fp16 ---- */

@interface GBFp16State : NSObject
@property(nonatomic) bool gemv;
@property(nonatomic) uint32_t expectedCrc;
@property(nonatomic, strong) id<MTLComputePipelineState> cp;
@property(nonatomic, strong) id<MTLBuffer> wBuf, xBuf, outBuf;
@end
@implementation GBFp16State @end

static id
fp16_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *mode = gb_param(o, "mode", "gemv");
   bool gemv = !strcmp(mode, "gemv");
   if (!gemv && strcmp(mode, "alu_mix")) {
      snprintf(err, n, "mode must be gemv or alu_mix");
      return nil;
   }
   GBFp16State *st = [GBFp16State new];
   st.gemv = gemv;
   if (gemv) {
      size_t wb = (size_t)GB_FP16_GEMV_M * GB_FP16_GEMV_K;
      size_t xb = (size_t)GB_FP16_GEMV_K * 2u;
      size_t yb = (size_t)GB_FP16_GEMV_M * sizeof(float);
      uint32_t *wh = malloc(wb), *xh = malloc(xb);
      float *yref = malloc(yb);
      gb_fp16_gemv_fill(wh, xh);
      st.expectedCrc = gb_fp16_gemv_ref(wh, xh, yref);
      free(yref);
      st.wBuf = buf(wb, false);
      st.xBuf = buf(xb, false);
      st.outBuf = buf(yb, false);
      mtl_buf_upload(st.wBuf, wh, wb);
      mtl_buf_upload(st.xBuf, xh, xb);
      free(wh); free(xh);
      st.cp = cpipe("fp16_gemv_main");
   } else {
      uint32_t nthr = GB_FP16_ALUMIX_GROUPS * GB_FP16_ALUMIX_LOCAL;
      float *ref = malloc((size_t)nthr * sizeof(float));
      st.expectedCrc = gb_fp16_alumix_ref(nthr, GB_FP16_ALUMIX_ITERS, -1.0f, 0.25f, ref);
      free(ref);
      st.outBuf = buf((size_t)nthr * sizeof(float), false);
      st.cp = cpipe("fp16_alumix_main");
   }
   return st;
}

static void
fp16_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBFp16State *st = state;
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:st.cp];
   if (st.gemv) {
      [e setBuffer:st.wBuf offset:0 atIndex:0];
      [e setBuffer:st.xBuf offset:0 atIndex:1];
      [e setBuffer:st.outBuf offset:0 atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(GB_FP16_GEMV_M / 4u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GB_FP16_GEMV_LOCAL, 1, 1)];
   } else {
      struct { uint32_t iters; float a, b; } pc = {GB_FP16_ALUMIX_ITERS, -1.0f, 0.25f};
      [e setBuffer:st.outBuf offset:0 atIndex:0];
      [e setBytes:&pc length:sizeof(pc) atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(GB_FP16_ALUMIX_GROUPS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GB_FP16_ALUMIX_LOCAL, 1, 1)];
   }
   [e endEncoding];
}

static void
fp16_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   if (!strcmp(gb_param(o, "mode", "gemv"), "gemv")) {
      double bytes = (double)GB_FP16_GEMV_M * GB_FP16_GEMV_K + (double)GB_FP16_GEMV_K * 2.0 +
                     (double)GB_FP16_GEMV_M * 4.0;
      double flops = 2.0 * (double)GB_FP16_GEMV_M * (double)GB_FP16_GEMV_K;
      gb_derive_rate(r, "gbps", "GB/s", us, bytes / 1e3);
      gb_derive_rate(r, "gflops", "GFLOP/s", us, flops / 1e3);
   } else {
      gb_derive_rate(r, "gflops", "GFLOP/s", us, GB_FP16_ALUMIX_FLOP / 1e3);
   }
}

static void
fp16_check(id state, struct gb_check *chk)
{
   GBFp16State *st = state;
   size_t bytes = st.gemv ? (size_t)GB_FP16_GEMV_M * sizeof(float)
                          : (size_t)GB_FP16_ALUMIX_GROUPS * GB_FP16_ALUMIX_LOCAL * sizeof(float);
   void *got = malloc(bytes);
   mtl_buf_readback(st.outBuf, got, bytes);
   mtl_set_crc_check(chk, gb_crc32(got, bytes), st.expectedCrc);
   free(got);
}

static const char *const fp16_params[] = {"mode", NULL};
static const struct mtest t_compute_fp16 = {
   .name = "compute.fp16", .known = fp16_params, .depths = {1, 4}, .nd = 2,
   .setup = fp16_setup, .encode = fp16_encode, .derive = fp16_derive, .check = fp16_check,
};

/* ---- W2.5 transfer.ops ---- */

@interface GBXferState : NSObject
@property(nonatomic) int kind;
@property(nonatomic) uint32_t w, h, mips, expectedCrc;
@property(nonatomic, strong) id<MTLBuffer> staging;
@property(nonatomic, strong) id<MTLTexture> tex;
@end
@implementation GBXferState @end

static id
xfer_setup(const struct gb_opts *o, char *err, size_t n)
{
   const char *k = gb_param(o, "kind", "clear_4m");
   GBXferState *st = [GBXferState new];
   if (!strcmp(k, "clear_4m")) {
      st.kind = 0; st.w = GB_XFER_4M_DIM; st.h = GB_XFER_4M_DIM; st.mips = 1;
      st.expectedCrc = gb_transfer_clear_ref_crc(st.w, st.h);
   } else if (!strcmp(k, "clear_16m")) {
      st.kind = 1; st.w = GB_XFER_16M_DIM; st.h = GB_XFER_16M_DIM; st.mips = 1;
      st.expectedCrc = gb_transfer_clear_ref_crc(st.w, st.h);
   } else if (!strcmp(k, "copy_b2i")) {
      st.kind = 2; st.w = GB_XFER_16M_DIM; st.h = GB_XFER_16M_DIM; st.mips = 1;
      size_t bytes = (size_t)st.w * st.h * 4u;
      st.staging = buf_tracked(bytes, true);
      gb_transfer_fill_rgba(st.w, st.h, st.staging.contents);
      st.expectedCrc = gb_crc32(st.staging.contents, bytes);
   } else if (!strcmp(k, "mipchain")) {
      st.kind = 3; st.w = GB_XFER_MIP_W; st.h = GB_XFER_MIP_H; st.mips = GB_XFER_MIP_LEVELS;
   } else {
      snprintf(err, n, "kind must be clear_4m, clear_16m, copy_b2i or mipchain");
      return nil;
   }
   MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:st.w
                                                                                height:st.h
                                                                             mipmapped:(st.mips > 1)];
   td.mipmapLevelCount = st.mips;
   td.storageMode = MTLStorageModePrivate;
   td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
   td.hazardTrackingMode = MTLHazardTrackingModeTracked;
   st.tex = [g_dev newTextureWithDescriptor:td];
   if (!st.tex) gb_die("transfer.ops texture failed");
   if (st.kind == 3) {
      size_t bytes = (size_t)st.w * st.h * 4u;
      id<MTLBuffer> up = buf_tracked(bytes, true);
      gb_transfer_fill_mip0(st.w, st.h, up.contents);
      id<MTLCommandQueue> q = new_queue(2);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b copyFromBuffer:up sourceOffset:0 sourceBytesPerRow:(size_t)st.w * 4u
       sourceBytesPerImage:bytes sourceSize:MTLSizeMake(st.w, st.h, 1)
                 toTexture:st.tex destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
      [b endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
   }
   return st;
}

static void
xfer_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBXferState *st = state;
   if (st.kind <= 1) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = st.tex;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.colorAttachments[0].clearColor = MTLClearColorMake(192.0 / 255.0, 128.0 / 255.0, 64.0 / 255.0, 1.0);
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      [e endEncoding];
   } else if (st.kind == 2) {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      size_t row = (size_t)st.w * 4u;
      [b copyFromBuffer:st.staging sourceOffset:0 sourceBytesPerRow:row
       sourceBytesPerImage:row * st.h sourceSize:MTLSizeMake(st.w, st.h, 1)
                 toTexture:st.tex destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
      [b endEncoding];
   } else {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b generateMipmapsForTexture:st.tex];
      [b endEncoding];
   }
}

static void
xfer_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   const char *k = gb_param(o, "kind", "clear_4m");
   double bytes = 0.0;
   if (!strcmp(k, "clear_4m")) bytes = (double)GB_XFER_4M_DIM * GB_XFER_4M_DIM * 4.0;
   else if (!strcmp(k, "clear_16m") || !strcmp(k, "copy_b2i"))
      bytes = (double)GB_XFER_16M_DIM * GB_XFER_16M_DIM * 4.0;
   else {
      uint32_t mw = GB_XFER_MIP_W, mh = GB_XFER_MIP_H;
      for (uint32_t i = 1; i < GB_XFER_MIP_LEVELS; i++) {
         uint32_t nw = mw > 1 ? mw / 2 : 1, nh = mh > 1 ? mh / 2 : 1;
         bytes += ((double)mw * mh + (double)nw * nh) * 4.0;
         mw = nw; mh = nh;
      }
   }
   gb_derive_rate(r, "gbps", "GB/s", us, bytes / 1e3);
}

static void
xfer_check(id state, struct gb_check *chk)
{
   GBXferState *st = state;
   if (st.kind != 3) {
      size_t bytes = (size_t)st.w * st.h * 4u;
      uint8_t *got = malloc(bytes);
      mtl_tex_readback(st.tex, got, st.w, st.h, 4u);
      mtl_set_crc_check(chk, gb_crc32(got, bytes), st.expectedCrc);
      free(got);
   } else {
      uint32_t m4w = st.w >> 4, m4h = st.h >> 4;
      size_t b4_bytes = (size_t)m4w * m4h * 4u;
      id<MTLBuffer> b4 = buf_tracked(b4_bytes, true);
      id<MTLBuffer> bl = buf_tracked(16, true);
      id<MTLCommandQueue> q = new_queue(2);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b copyFromTexture:st.tex sourceSlice:0 sourceLevel:4 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(m4w, m4h, 1) toBuffer:b4 destinationOffset:0
       destinationBytesPerRow:(size_t)m4w * 4u destinationBytesPerImage:b4_bytes];
      [b copyFromTexture:st.tex sourceSlice:0 sourceLevel:(st.mips - 1u) sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(1, 1, 1) toBuffer:bl destinationOffset:0
       destinationBytesPerRow:4 destinationBytesPerImage:4];
      [b endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      gb_transfer_check_mipchain(b4.contents, m4w, m4h, bl.contents, chk);
   }
}

static const char *const xfer_params[] = {"kind", NULL};
static const struct mtest t_transfer_ops = {
   .name = "transfer.ops", .known = xfer_params, .depths = {1, 4}, .nd = 2,
   .setup = xfer_setup, .encode = xfer_encode, .derive = xfer_derive, .check = xfer_check,
};

/* ---- W2.6 render.postfx ---- */

@interface GBPostfxState : NSObject
@property(nonatomic) uint32_t w, h, msaa;
@property(nonatomic, strong) id<MTLTexture> hdrMs, hdrRes, depthMs, finalTex;
@property(nonatomic, strong) NSArray<id<MTLTexture>> *bloom;
@property(nonatomic, strong) id<MTLSamplerState> linSampler;
@property(nonatomic, strong) id<MTLDepthStencilState> dsLess;
@property(nonatomic, strong) id<MTLRenderPipelineState> pScene, pDown, pUp, pTone;
@end
@implementation GBPostfxState @end

static id
postfx_setup(const struct gb_opts *o, char *err, size_t n)
{
   uint32_t w = 0, h = 0;
   if (!parse_res(gb_param(o, "res", "1440p"), &w, &h)) {
      snprintf(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return nil;
   }
   long msaa = gb_param_int(o, "msaa", 4);
   if (msaa != 1 && msaa != 4) {
      snprintf(err, n, "msaa must be 1 or 4");
      return nil;
   }
   GBPostfxState *st = [GBPostfxState new];
   st.w = w; st.h = h; st.msaa = (uint32_t)msaa;
   if (msaa == 4)
      st.hdrMs = tex_ex_opt(w, h, MTLPixelFormatRGBA16Float, 4, MTLStorageModeMemoryless, MTLTextureUsageRenderTarget, false);
   st.hdrRes = tex_ex_opt(w, h, MTLPixelFormatRGBA16Float, 1,
                          MTLStorageModePrivate, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, true);
   st.depthMs = tex_ex_opt(w, h, MTLPixelFormatDepth32Float, (uint32_t)msaa, MTLStorageModeMemoryless, MTLTextureUsageRenderTarget, false);
   NSMutableArray<id<MTLTexture>> *bloom = [NSMutableArray arrayWithCapacity:GB_POSTFX_BLOOM_MIPS];
   uint32_t bw = w, bh = h;
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      bw = bw > 1 ? bw / 2 : 1;
      bh = bh > 1 ? bh / 2 : 1;
      [bloom addObject:tex_ex_opt(bw, bh, MTLPixelFormatRGBA16Float, 1,
                                  MTLStorageModePrivate, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, true)];
   }
   st.bloom = bloom;
   st.finalTex = tex_ex_opt(w, h, MTLPixelFormatRGBA8Unorm, 1,
                            MTLStorageModePrivate, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, true);

   MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
   sd.minFilter = MTLSamplerMinMagFilterLinear;
   sd.magFilter = MTLSamplerMinMagFilterLinear;
   sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
   sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
   st.linSampler = [g_dev newSamplerStateWithDescriptor:sd];

   MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
   dsd.depthCompareFunction = MTLCompareFunctionLess;
   dsd.depthWriteEnabled = YES;
   st.dsLess = [g_dev newDepthStencilStateWithDescriptor:dsd];

   NSError *merr = nil;
   MTLRenderPipelineDescriptor *sp = [MTLRenderPipelineDescriptor new];
   sp.vertexFunction = [g_lib newFunctionWithName:@"postfx_scene_vert"];
   sp.fragmentFunction = [g_lib newFunctionWithName:@"postfx_scene_frag"];
   sp.rasterSampleCount = st.msaa;
   sp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
   sp.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
   st.pScene = [g_dev newRenderPipelineStateWithDescriptor:sp error:&merr];
   if (!st.pScene) gb_die("postfx scene pipe failed: %s", merr.localizedDescription.UTF8String);

   MTLRenderPipelineDescriptor *dp = [MTLRenderPipelineDescriptor new];
   dp.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   dp.fragmentFunction = [g_lib newFunctionWithName:@"postfx_down_frag"];
   dp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
   st.pDown = [g_dev newRenderPipelineStateWithDescriptor:dp error:&merr];
   if (!st.pDown) gb_die("postfx down pipe failed: %s", merr.localizedDescription.UTF8String);

   MTLRenderPipelineDescriptor *up = [MTLRenderPipelineDescriptor new];
   up.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   up.fragmentFunction = [g_lib newFunctionWithName:@"postfx_up_frag"];
   up.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
   up.colorAttachments[0].blendingEnabled = YES;
   up.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
   up.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
   up.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
   up.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
   up.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
   up.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
   st.pUp = [g_dev newRenderPipelineStateWithDescriptor:up error:&merr];
   if (!st.pUp) gb_die("postfx up pipe failed: %s", merr.localizedDescription.UTF8String);

   MTLRenderPipelineDescriptor *tp = [MTLRenderPipelineDescriptor new];
   tp.vertexFunction = [g_lib newFunctionWithName:@"fs_vert"];
   tp.fragmentFunction = [g_lib newFunctionWithName:@"postfx_tonemap_frag"];
   tp.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
   st.pTone = [g_dev newRenderPipelineStateWithDescriptor:tp error:&merr];
   if (!st.pTone) gb_die("postfx tone pipe failed: %s", merr.localizedDescription.UTF8String);
   return st;
}

static void
postfx_encode(id state, id<MTLCommandBuffer> cb, uint64_t k)
{
   (void)k;
   GBPostfxState *st = state;
   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   if (st.msaa == 4u) {
      rp.colorAttachments[0].texture = st.hdrMs;
      rp.colorAttachments[0].resolveTexture = st.hdrRes;
      rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
   } else {
      rp.colorAttachments[0].texture = st.hdrRes;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   }
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0.04, 0.05, 0.08, 1.0);
   rp.depthAttachment.texture = st.depthMs;
   rp.depthAttachment.loadAction = MTLLoadActionClear;
   rp.depthAttachment.storeAction = MTLStoreActionDontCare;
   rp.depthAttachment.clearDepth = 1.0;
   id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
   [e setRenderPipelineState:st.pScene];
   [e setDepthStencilState:st.dsLess];
   [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:GB_POSTFX_QUADS * 6u];
   [e endEncoding];

   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      id<MTLTexture> dst = st.bloom[i];
      id<MTLTexture> src = (i == 0) ? st.hdrRes : st.bloom[i - 1u];
      MTLRenderPassDescriptor *drp = [MTLRenderPassDescriptor renderPassDescriptor];
      drp.colorAttachments[0].texture = dst;
      drp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      drp.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> de = [cb renderCommandEncoderWithDescriptor:drp];
      [de setRenderPipelineState:st.pDown];
      [de setFragmentTexture:src atIndex:0];
      [de setFragmentSamplerState:st.linSampler atIndex:0];
      struct { float iw, ih, siw, sih; uint32_t bp; } pc = {
         1.0f / (float)dst.width, 1.0f / (float)dst.height,
         1.0f / (float)src.width, 1.0f / (float)src.height, i == 0 ? 1u : 0u
      };
      [de setFragmentBytes:&pc length:sizeof(pc) atIndex:0];
      [de drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [de endEncoding];
   }

   for (int i = (int)GB_POSTFX_BLOOM_MIPS - 2; i >= 0; i--) {
      id<MTLTexture> dst = st.bloom[i];
      id<MTLTexture> src = st.bloom[i + 1];
      MTLRenderPassDescriptor *urp = [MTLRenderPassDescriptor renderPassDescriptor];
      urp.colorAttachments[0].texture = dst;
      urp.colorAttachments[0].loadAction = MTLLoadActionLoad;
      urp.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> ue = [cb renderCommandEncoderWithDescriptor:urp];
      [ue setRenderPipelineState:st.pUp];
      [ue setFragmentTexture:src atIndex:0];
      [ue setFragmentSamplerState:st.linSampler atIndex:0];
      float pc[4] = {1.0f / (float)dst.width, 1.0f / (float)dst.height,
                     1.0f / (float)src.width, 1.0f / (float)src.height};
      [ue setFragmentBytes:pc length:sizeof(pc) atIndex:0];
      [ue drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [ue endEncoding];
   }

   MTLRenderPassDescriptor *trp = [MTLRenderPassDescriptor renderPassDescriptor];
   trp.colorAttachments[0].texture = st.finalTex;
   trp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
   trp.colorAttachments[0].storeAction = MTLStoreActionStore;
   id<MTLRenderCommandEncoder> te = [cb renderCommandEncoderWithDescriptor:trp];
   [te setRenderPipelineState:st.pTone];
   [te setFragmentTexture:st.hdrRes atIndex:0];
   [te setFragmentTexture:st.bloom[0] atIndex:1];
   [te setFragmentSamplerState:st.linSampler atIndex:0];
   float tpc[2] = {1.0f / (float)st.w, 1.0f / (float)st.h};
   [te setFragmentBytes:tpc length:sizeof(tpc) atIndex:0];
   [te drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
   [te endEncoding];
}

static void
postfx_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 2560, h = 1440;
   parse_res(gb_param(o, "res", "1440p"), &w, &h);
   gb_derive_rate(r, "fps", "fps", us, 1e6);
   gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h / 1e3);
}

static void
postfx_check(id state, struct gb_check *chk)
{
   GBPostfxState *st = state;
   uint8_t *got = malloc((size_t)st.w * st.h * 4u);
   mtl_tex_readback(st.finalTex, got, st.w, st.h, 4u);
   gb_postfx_check_fb(got, st.w, st.h, chk);
   free(got);
}

static const char *const postfx_params[] = {"res", "msaa", NULL};
static const struct mtest t_render_postfx = {
   .name = "render.postfx", .known = postfx_params, .depths = {1, 2}, .nd = 2,
   .setup = postfx_setup, .encode = postfx_encode, .derive = postfx_derive, .check = postfx_check,
};

/* ======================= table ======================= */

static int st_empty(const struct gb_opts *o) { return run_stream_test(o, &t_submit_empty); }
static int st_cl(const struct gb_opts *o) { return run_stream_test(o, &t_compute_light); }
static int st_ch(const struct gb_opts *o) { return run_stream_test(o, &t_compute_heavy); }
static int st_rl(const struct gb_opts *o) { return run_stream_test(o, &t_render_light); }
static int st_rh(const struct gb_opts *o) { return run_stream_test(o, &t_render_heavy); }
static int st_tvb(const struct gb_opts *o) { return run_stream_test(o, &t_render_tvb); }
static int st_copy(const struct gb_opts *o) { return run_stream_test(o, &t_copy_bw); }
static int st_diag_fill(const struct gb_opts *o) { return run_stream_test(o, &t_diag_fill); }
static int st_mem_buffer(const struct gb_opts *o) { return run_mem(o, false); }
static int st_mem_image(const struct gb_opts *o) { return run_mem(o, true); }
static int st_uber(const struct gb_opts *o) { return run_stream_test(o, &t_compute_uber); }
static int st_conv2d(const struct gb_opts *o) { return run_stream_test(o, &t_compute_conv2d); }
static int st_stereo(const struct gb_opts *o) { return run_stream_test(o, &t_compute_stereo); }
static int st_edge(const struct gb_opts *o) { return run_stream_test(o, &t_compute_edge); }
static int st_deferred(const struct gb_opts *o) { return run_stream_test(o, &t_render_deferred); }
static int st_mdi(const struct gb_opts *o) { return run_stream_test(o, &t_render_mdi); }
static int st_cmp(const struct gb_opts *o) { return run_stream_test(o, &t_compress_bw); }
static int st_shadow(const struct gb_opts *o) { return run_stream_test(o, &t_render_shadow); }
static int st_cull_mdi(const struct gb_opts *o) { return run_stream_test(o, &t_compute_cull_mdi); }
static int st_bubo(const struct gb_opts *o) { return run_stream_test(o, &t_render_bindless_ubo); }
static int st_fp16(const struct gb_opts *o) { return run_stream_test(o, &t_compute_fp16); }
static int st_xfer(const struct gb_opts *o) { return run_stream_test(o, &t_transfer_ops); }
static int st_postfx(const struct gb_opts *o) { return run_stream_test(o, &t_render_postfx); }

static const struct {
   const char *name;
   int (*run)(const struct gb_opts *o);
} tests[] = {
   {"submit.empty", st_empty},
   {"compute.light", st_cl},
   {"compute.heavy", st_ch},
   {"render.light", st_rl},
   {"render.heavy", st_rh},
   {"render.tvb", st_tvb},
   {"sync.signal", run_sync},
   {"mem.buffer", st_mem_buffer},
   {"mem.image", st_mem_image},
   {"copy.bw", st_copy},
   {"diag.fill", st_diag_fill},
   {"concurrent.cr", run_concurrent},
   {"query", run_query},
   {"idle.wake", run_idle},
   /* 90%-coverage Wave-1 benchmarks */
   {"compute.uber", st_uber},
   {"compute.conv2d", st_conv2d},
   {"compute.stereo", st_stereo},
   {"compute.edge", st_edge},
   {"render.deferred", st_deferred},
   {"concurrent.sram", run_concurrent_sram},
   {"render.mdi", st_mdi},
   {"compile.pso", run_compile_pso},
   {"compress.bw", st_cmp},
   {"overlap.passes", run_overlap_passes},
   /* 90%-coverage Wave-2 heavy-game / AAA benchmarks */
   {"render.shadow", st_shadow},
   {"compute.cull_mdi", st_cull_mdi},
   {"render.bindless_ubo", st_bubo},
   {"compute.fp16", st_fp16},
   {"transfer.ops", st_xfer},
   {"render.postfx", st_postfx},
   {NULL, NULL},
};

int
main(int argc, char **argv)
{
   @autoreleasepool {
      if (argc == 3 && !strcmp(argv[1], "--dump-archive"))
         return dump_archives(argv[2]);
      struct gb_opts o;
      gb_parse_opts(&o, argc, argv);
      if (o.info)
         return print_info();
      if (o.list) {
         for (int i = 0; tests[i].name; i++)
            puts(tests[i].name);
         return 0;
      }
      if (!o.test)
         gb_die("need --test NAME, --list or --info (see --help)");
      for (int i = 0; tests[i].name; i++)
         if (!strcmp(tests[i].name, o.test))
            return tests[i].run(&o);
      gb_die("unknown test %s", o.test);
   }
}
