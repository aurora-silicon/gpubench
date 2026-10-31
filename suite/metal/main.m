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

#include "gb_common.h"
#include "gb_params.h"
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
              struct gb_pass *ti, double first_us, int units)
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
         finish_result(o, t, GB_MODE_PROCS, n, &u, do_timed ? &ti : NULL, -1, n);
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
         finish_result(o, t, o->mode, n, &u, do_timed ? &ti : NULL, first_us, n);
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
