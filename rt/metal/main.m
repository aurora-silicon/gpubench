// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-rt-mtl: the Metal executor of the ray-tracing lane. Same tests, parameters, scene,
 * data layouts, passes and timing method as gpubench-rt-vk (rt/vulkan):
 *   rt.scene   [update] water deformation, [as] BLAS refit/rebuild + TLAS build, [raster]
 *              G-buffer, [rt] intersection-query reflections + shadows, [composite]
 *   rt.asbuild one acceleration-structure build per submission
 *   rt.rays    one dispatch of closest-hit intersection queries per submission
 * Depth N = a ring of N command buffers in flight: before committing k the CPU waits for k - N.
 * Per-pass GPU times come from counter sample buffers at encoder (stage) boundaries; gpu_us is
 * GPUEndTime - GPUStartTime of each command buffer. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include "../common/json.h"
#include "../common/params.h"
#include "../common/result.h"
#include "../common/scene.h"
#include "../common/util.h"

extern const char rt_metal_src[];

#define MAXD 64
enum { P_UPDATE, P_AS, P_RASTER, P_RT, P_COMPOSITE, NPASS };
static const char *pass_metric[NPASS] = {"gpu_update_us", "gpu_as_us", "gpu_raster_us", "gpu_rt_us",
                                         "gpu_composite_us"};
/* sample indices per frame slot: start/end of each encoder */
enum { S_DEF0, S_DEF1, S_BLAS0, S_BLAS1, S_TLAS0, S_TLAS1, S_GB0, S_GB1, S_RT0, S_RT1, S_COMP0, S_COMP1, NSAMP };

/* ---------------- device ---------------- */

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_lib;
static id<MTLCounterSet> g_tsset;
static int g_stage_sampling;
/* GPU tick -> ns calibration */
static MTLTimestamp g_cpu0, g_gpu0;

static int metal_init(const rt_run *r)
{
   (void)r;
   g_dev = MTLCreateSystemDefaultDevice();
   if (!g_dev) return -1;
   g_queue = [g_dev newCommandQueueWithMaxCommandBufferCount:MAXD + 4];
   for (id<MTLCounterSet> cs in g_dev.counterSets)
      if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) g_tsset = cs;
   g_stage_sampling = g_tsset && [g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
   [g_dev sampleTimestamps:&g_cpu0 gpuTimestamp:&g_gpu0];
   NSError *err = nil;
   NSString *exe = [[NSProcessInfo processInfo].arguments[0] stringByResolvingSymlinksInPath];
   NSString *lib = [exe stringByAppendingString:@".metallib"];
   if ([[NSFileManager defaultManager] fileExistsAtPath:lib]) {
      g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:lib] error:&err];
      if (g_lib) fprintf(stderr, "shaders: %s\n", lib.UTF8String);
   }
   if (!g_lib) {
      MTLCompileOptions *o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion3_0;
      g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:rt_metal_src] options:o error:&err];
      if (!g_lib) {
         fprintf(stderr, "MSL compile failed: %s\n", err.localizedDescription.UTF8String);
         return -1;
      }
      fprintf(stderr, "shaders: compiled from embedded MSL\n");
   }
   return 0;
}

static double tick_ns(void)
{
   MTLTimestamp c1, g1;
   [g_dev sampleTimestamps:&c1 gpuTimestamp:&g1];
   if (g1 <= g_gpu0 || c1 <= g_cpu0) return 1.0;
   return (double)(c1 - g_cpu0) / (double)(g1 - g_gpu0);
}

static void os_version(char *buf, size_t n)
{
   NSOperatingSystemVersion v = [NSProcessInfo processInfo].operatingSystemVersion;
   char build[64] = "";
   size_t bl = sizeof(build);
   sysctlbyname("kern.osversion", build, &bl, NULL, 0);
   snprintf(buf, n, "%ld.%ld.%ld (%s)", (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, build);
}

static void print_info(void)
{
   rt_json j;
   rt_json_init(&j);
   rt_json_obj(&j, NULL);
   rt_json_str(&j, "api", "metal");
   id<MTLDevice> d = MTLCreateSystemDefaultDevice();
   if (!d) {
      rt_json_str(&j, "error", "no Metal device");
   } else {
      char os[128];
      os_version(os, sizeof(os));
      rt_json_str(&j, "device", d.name.UTF8String);
      rt_json_obj(&j, "driver");
      rt_json_str(&j, "name", "Metal");
      rt_json_str(&j, "info", os);
      rt_json_str(&j, "version", os);
      rt_json_end(&j);
      rt_json_obj(&j, "gpu");
      rt_json_str(&j, "name", d.name.UTF8String);
      rt_json_bool(&j, "apple9", [d supportsFamily:MTLGPUFamilyApple9]);
      rt_json_int(&j, "max_working_set", (int64_t)d.recommendedMaxWorkingSetSize);
      rt_json_end(&j);
      rt_json_obj(&j, "rt");
      rt_json_bool(&j, "acceleration_structure", d.supportsRaytracing);
      rt_json_bool(&j, "ray_query", d.supportsRaytracing);
      rt_json_bool(&j, "ray_tracing_pipeline", 0);
      rt_json_bool(&j, "stage_boundary_counters", [d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]);
      rt_json_end(&j);
   }
   rt_json_end(&j);
   printf("%s\n", j.buf);
   rt_json_free(&j);
}

static const char *thermal_state(void)
{
   switch ([NSProcessInfo processInfo].thermalState) {
   case NSProcessInfoThermalStateNominal: return "nominal";
   case NSProcessInfoThermalStateFair: return "fair";
   case NSProcessInfoThermalStateSerious: return "serious";
   case NSProcessInfoThermalStateCritical: return "critical";
   }
   return "?";
}

static id<MTLBuffer> buf(size_t n, int shared, int tracked)
{
   MTLResourceOptions o = shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
   if (!tracked) o |= MTLResourceHazardTrackingModeUntracked;
   return [g_dev newBufferWithLength:n ? n : 16 options:o];
}

/* ---------------- submission stream ---------------- */

typedef struct mtl_stream_stats {
   rt_vec rate, interval_us, latency_us, record_us, submit_us, gpu_us, gap_us;
   uint64_t completions;
} mtl_stream_stats;

typedef struct mtl_stream {
   uint32_t depth;
   id<MTLCommandBuffer> cb[MAXD];
   uint64_t submit_ns[MAXD];
   int timed[MAXD];
   uint64_t next, done, last_done_ns;
   double last_gpu_end;
   int unretained;
   void *user;
   int (*record)(void *user, uint32_t slot, uint64_t k, id<MTLCommandBuffer> cb, int timed);
   void (*complete)(void *user, uint32_t slot, uint64_t k, int timed, id<MTLCommandBuffer> cb);
   mtl_stream_stats *stats;
   int error;
} mtl_stream;

static int observe(mtl_stream *s, uint64_t k)
{
   uint32_t slot = (uint32_t)(k % s->depth);
   id<MTLCommandBuffer> cb = s->cb[slot];
   [cb waitUntilCompleted];
   uint64_t now = rt_now_ns();
   if (cb.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "command buffer %llu failed: %s\n", (unsigned long long)k, cb.error.localizedDescription.UTF8String);
      s->error = 1;
   }
   if (s->stats) {
      rt_vec_push(&s->stats->latency_us, (now - s->submit_ns[slot]) / 1e3);
      if (s->last_done_ns) rt_vec_push(&s->stats->interval_us, (now - s->last_done_ns) / 1e3);
      s->stats->completions++;
      if (s->timed[slot]) {
         rt_vec_push(&s->stats->gpu_us, (cb.GPUEndTime - cb.GPUStartTime) * 1e6);
         if (s->last_gpu_end > 0) rt_vec_push(&s->stats->gap_us, fmax(0, (cb.GPUStartTime - s->last_gpu_end) * 1e6));
      }
   }
   s->last_gpu_end = s->timed[slot] ? cb.GPUEndTime : 0;
   s->last_done_ns = now;
   s->done = k;
   if (s->complete) s->complete(s->user, slot, k, s->timed[slot], cb);
   s->cb[slot] = nil;
   return s->error ? -1 : 0;
}

static int stream_submit(mtl_stream *s, int timed)
{
   uint64_t k = s->next;
   uint32_t slot = (uint32_t)(k % s->depth);
   if (k > s->depth && s->done < k - s->depth && observe(s, k - s->depth)) return -1;
   uint64_t t0 = rt_now_ns();
   @autoreleasepool {
      id<MTLCommandBuffer> cb = s->unretained ? [g_queue commandBufferWithUnretainedReferences] : [g_queue commandBuffer];
      if (s->record(s->user, slot, k, cb, timed)) return -1;
      uint64_t t1 = rt_now_ns();
      s->submit_ns[slot] = rt_now_ns();
      s->timed[slot] = timed;
      s->cb[slot] = cb;
      [cb commit];
      uint64_t t2 = rt_now_ns();
      if (s->stats) {
         rt_vec_push(&s->stats->record_us, (t1 - t0) / 1e3);
         rt_vec_push(&s->stats->submit_us, (t2 - s->submit_ns[slot]) / 1e3);
      }
   }
   s->next++;
   return 0;
}

static int stream_drain(mtl_stream *s)
{
   while (s->done + 1 < s->next)
      if (observe(s, s->done + 1)) return -1;
   return 0;
}

static int stream_phase(mtl_stream *s, uint32_t windows, uint32_t ms, uint32_t count, int timed, mtl_stream_stats *st)
{
   if (!windows) {
      s->stats = NULL;
      uint64_t end = rt_now_ns() + (uint64_t)ms * 1000000ull;
      while (rt_now_ns() < end)
         if (stream_submit(s, timed)) return -1;
      return 0;
   }
   s->stats = st;
   for (uint32_t w = 0; w < windows; w++) {
      uint64_t start = rt_now_ns(), d0 = st->completions;
      if (count) {
         while (st->completions - d0 < count)
            if (stream_submit(s, timed)) return -1;
      } else {
         /* a window lasts run_ms, and at least until two completions were observed, so frames
          * longer than the window still give a rate */
         uint64_t end = start + (uint64_t)ms * 1000000ull;
         while (rt_now_ns() < end || st->completions - d0 < 2)
            if (stream_submit(s, timed)) return -1;
      }
      rt_vec_push(&st->rate, (double)(st->completions - d0) / ((rt_now_ns() - start) / 1e9));
   }
   s->stats = NULL;
   return 0;
}

static void stats_free(mtl_stream_stats *st)
{
   rt_vec_free(&st->rate); rt_vec_free(&st->interval_us); rt_vec_free(&st->latency_us); rt_vec_free(&st->record_us);
   rt_vec_free(&st->submit_us); rt_vec_free(&st->gpu_us); rt_vec_free(&st->gap_us);
   memset(st, 0, sizeof(*st));
}

static void emit_us(rt_json *j, const char *name, rt_vec *v) { rt_metric(j, name, "us", "lower", v->v, v->n, 0); }

static int cmp_desc(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? 1 : x > y ? -1 : 0;
}
static double low1(const double *iv, size_t n)
{
   if (!n) return 0;
   double *s = malloc(n * sizeof(double));
   memcpy(s, iv, n * sizeof(double));
   qsort(s, n, sizeof(double), cmp_desc);
   size_t m = n / 100;
   if (m < 1) m = 1;
   double sum = 0;
   for (size_t i = 0; i < m; i++) sum += s[i];
   free(s);
   return 1e6 / (sum / (double)m);
}

/* ---------------- scene ---------------- */

typedef struct mscene {
   const rt_params *p;
   const rt_run *r;
   rt_scene sc;
   id<MTLBuffer> vbuf, ibuf, info, scratch, counters, readback;
   id<MTLTexture> albedo, normal, depth, rtimg, out;
   NSMutableArray<id<MTLAccelerationStructure>> *blas;
   MTLPrimitiveAccelerationStructureDescriptor *__strong *bdesc;
   MTLAccelerationStructureSizes *bsizes;
   size_t *bscratch_off;
   id<MTLAccelerationStructure> tlas;
   MTLInstanceAccelerationStructureDescriptor *tdesc;
   MTLAccelerationStructureSizes tsizes;
   size_t tscratch_off, init_scratch, frame_scratch;
   id<MTLBuffer> ubo[MAXD], xf[MAXD], inst[MAXD];
   uint32_t nslots;
   id<MTLComputePipelineState> p_def, p_rt, p_comp;
   id<MTLRenderPipelineState> p_gb_opaque, p_gb_sail;
   id<MTLDepthStencilState> ds;
   id<MTLCounterSampleBuffer> sb;
   int ts_ok;
   double tick;
   uint32_t frame;
   int census;
   rt_vec pass_us[NPASS], frame_us;
   double init_ms[3], init_wall_ms;
   int frame_mode;   /* --stamps frame: sample only the first encoder's start and the last one's end */
   int first_sample; /* S_DEF0 / S_TLAS0 / S_GB0: the first encoder of the frame being recorded */
} mscene;

static void slot_update(mscene *s, uint32_t slot, uint32_t f, int all)
{
   rt_frame_ubo u;
   rt_scene_frame_ubo(&s->sc, s->p, f, 0, s->census, &u);
   memcpy(s->ubo[slot].contents, &u, sizeof(u));
   float t = rt_frame_time(f);
   MTLAccelerationStructureInstanceDescriptor *in = s->inst[slot].contents;
   rt_xform *xf = s->xf[slot].contents;
   for (uint32_t i = 0; i < s->sc.ninst; i++) {
      if (!all && !(s->sc.inst[i].flags & RT_FLAG_MOVING)) continue;
      rt_scene_xform(&s->sc, i, t, &xf[i]);
      MTLAccelerationStructureInstanceDescriptor *d = &in[i];
      memset(d, 0, sizeof(*d));
      /* MTLPackedFloat4x3: four columns of packed float3; our rows are [R | t] */
      for (int c = 0; c < 4; c++)
         for (int rr = 0; rr < 3; rr++) d->transformationMatrix.columns[c].elements[rr] = xf[i].r[rr * 4 + c];
      d->options = MTLAccelerationStructureInstanceOptionNone;
      d->mask = 0xff;
      d->intersectionFunctionTableOffset = 0;
      d->accelerationStructureIndex = s->sc.inst[i].mesh;
   }
}

static MTLAccelerationStructureUsage as_usage(int mode)
{
   if (mode == RT_AS_REFIT) return MTLAccelerationStructureUsageRefit;
   if (mode == RT_AS_REBUILD) return MTLAccelerationStructureUsagePreferFastBuild;
   return MTLAccelerationStructureUsageNone;
}

static size_t al256(size_t v) { return (v + 255) & ~(size_t)255; }

static int create_scene(mscene *s, uint32_t nslots)
{
   rt_scene *sc = &s->sc;
   const rt_params *p = s->p;
   s->vbuf = buf((size_t)sc->nverts * sizeof(rt_vertex), 1, 1);
   memcpy(s->vbuf.contents, sc->verts, (size_t)sc->nverts * sizeof(rt_vertex));
   s->ibuf = buf((size_t)sc->nindices * 4, 1, 1);
   memcpy(s->ibuf.contents, sc->indices, (size_t)sc->nindices * 4);
   s->info = buf((size_t)sc->ninst * sizeof(rt_instinfo), 1, 1);
   memcpy(s->info.contents, sc->info, (size_t)sc->ninst * sizeof(rt_instinfo));
   s->counters = buf(16, 1, 1);
   memset(s->counters.contents, 0, 16);
   s->readback = buf((size_t)p->width * p->height * 4, 1, 1);
   s->nslots = nslots;
   for (uint32_t i = 0; i < nslots; i++) {
      s->ubo[i] = buf(sizeof(rt_frame_ubo), 1, 1);
      s->xf[i] = buf((size_t)sc->ninst * sizeof(rt_xform), 1, 1);
      s->inst[i] = buf((size_t)sc->ninst * sizeof(MTLAccelerationStructureInstanceDescriptor), 1, 1);
   }
   MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:p->width height:p->height mipmapped:NO];
   td.storageMode = MTLStorageModePrivate;
   td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
   s->albedo = [g_dev newTextureWithDescriptor:td];
   td.pixelFormat = MTLPixelFormatRGBA16Float;
   s->normal = [g_dev newTextureWithDescriptor:td];
   td.pixelFormat = MTLPixelFormatDepth32Float;
   s->depth = [g_dev newTextureWithDescriptor:td];
   td.pixelFormat = MTLPixelFormatRGBA16Float;
   td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
   s->rtimg = [g_dev newTextureWithDescriptor:td];
   td.pixelFormat = MTLPixelFormatRGBA8Unorm;
   s->out = [g_dev newTextureWithDescriptor:td];

   /* acceleration structures */
   s->blas = [NSMutableArray arrayWithCapacity:sc->nmeshes];
   s->bdesc = (__strong MTLPrimitiveAccelerationStructureDescriptor **)calloc(sc->nmeshes, sizeof(void *));
   s->bsizes = calloc(sc->nmeshes, sizeof(MTLAccelerationStructureSizes));
   s->bscratch_off = calloc(sc->nmeshes, sizeof(size_t));
   size_t init = 0, frame = 0;
   for (uint32_t m = 0; m < sc->nmeshes; m++) {
      rt_mesh *me = &sc->meshes[m];
      MTLAccelerationStructureTriangleGeometryDescriptor *g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
      g.vertexBuffer = s->vbuf;
      g.vertexBufferOffset = (NSUInteger)me->base_vertex * sizeof(rt_vertex);
      g.vertexStride = sizeof(rt_vertex);
      g.vertexFormat = MTLAttributeFormatFloat3;
      g.indexBuffer = s->ibuf;
      g.indexBufferOffset = (NSUInteger)me->first_index * 4;
      g.indexType = MTLIndexTypeUInt32;
      g.triangleCount = me->index_count / 3;
      g.opaque = !me->alpha;
      g.allowDuplicateIntersectionFunctionInvocation = !me->alpha;
      MTLPrimitiveAccelerationStructureDescriptor *d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      d.geometryDescriptors = @[ g ];
      d.usage = as_usage(me->dynamic ? p->blas_mode : RT_AS_STATIC);
      s->bdesc[m] = d;
      s->bsizes[m] = [g_dev accelerationStructureSizesWithDescriptor:d];
      id<MTLAccelerationStructure> as = [g_dev newAccelerationStructureWithSize:s->bsizes[m].accelerationStructureSize];
      if (!as) return -1;
      [s->blas addObject:as];
      init += al256(s->bsizes[m].buildScratchBufferSize);
      if (me->dynamic) {
         s->bscratch_off[m] = frame;
         frame += al256(p->blas_mode == RT_AS_REFIT ? s->bsizes[m].refitScratchBufferSize : s->bsizes[m].buildScratchBufferSize);
      }
   }
   s->tdesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
   s->tdesc.instancedAccelerationStructures = s->blas;
   s->tdesc.instanceCount = sc->ninst;
   s->tdesc.instanceDescriptorBuffer = s->inst[0];
   s->tdesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
   s->tdesc.usage = as_usage(p->tlas_mode);
   s->tsizes = [g_dev accelerationStructureSizesWithDescriptor:s->tdesc];
   s->tlas = [g_dev newAccelerationStructureWithSize:s->tsizes.accelerationStructureSize];
   s->tscratch_off = frame;
   if (p->tlas_mode != RT_AS_STATIC)
      frame += al256(p->tlas_mode == RT_AS_REFIT ? s->tsizes.refitScratchBufferSize : s->tsizes.buildScratchBufferSize);
   size_t need = init;
   if (s->tsizes.buildScratchBufferSize > need) need = s->tsizes.buildScratchBufferSize;
   if (frame > need) need = frame;
   s->init_scratch = init;
   s->frame_scratch = frame;
   s->scratch = buf(need, 0, 1);

   /* pipelines */
   NSError *err = nil;
   s->p_def = [g_dev newComputePipelineStateWithFunction:[g_lib newFunctionWithName:@"deform"] error:&err];
   s->p_rt = [g_dev newComputePipelineStateWithFunction:[g_lib newFunctionWithName:@"rt_reflect"] error:&err];
   s->p_comp = [g_dev newComputePipelineStateWithFunction:[g_lib newFunctionWithName:@"composite"] error:&err];
   if (!s->p_def || !s->p_rt || !s->p_comp) {
      fprintf(stderr, "compute pipeline: %s\n", err.localizedDescription.UTF8String);
      return -1;
   }
   MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
   vd.attributes[0].format = MTLVertexFormatFloat3; vd.attributes[0].offset = 0; vd.attributes[0].bufferIndex = 0;
   vd.attributes[1].format = MTLVertexFormatFloat3; vd.attributes[1].offset = 12; vd.attributes[1].bufferIndex = 0;
   vd.attributes[2].format = MTLVertexFormatFloat2; vd.attributes[2].offset = 24; vd.attributes[2].bufferIndex = 0;
   vd.layouts[0].stride = sizeof(rt_vertex);
   for (int sail = 0; sail < 2; sail++) {
      MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
      bool alpha = sail && p->alpha;
      [fc setConstantValue:&alpha type:MTLDataTypeBool atIndex:0];
      MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
      rd.vertexFunction = [g_lib newFunctionWithName:@"gbuffer_vs"];
      rd.fragmentFunction = [g_lib newFunctionWithName:@"gbuffer_fs" constantValues:fc error:&err];
      rd.vertexDescriptor = vd;
      rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      rd.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
      rd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:rd error:&err];
      if (!ps) {
         fprintf(stderr, "render pipeline: %s\n", err.localizedDescription.UTF8String);
         return -1;
      }
      if (sail) s->p_gb_sail = ps; else s->p_gb_opaque = ps;
   }
   MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
   dsd.depthCompareFunction = MTLCompareFunctionGreater;
   dsd.depthWriteEnabled = YES;
   s->ds = [g_dev newDepthStencilStateWithDescriptor:dsd];
   if (g_stage_sampling) {
      MTLCounterSampleBufferDescriptor *cd = [MTLCounterSampleBufferDescriptor new];
      cd.counterSet = g_tsset;
      cd.storageMode = MTLStorageModeShared;
      cd.sampleCount = nslots * NSAMP + 8;
      s->sb = [g_dev newCounterSampleBufferWithDescriptor:cd error:&err];
   }
   s->ts_ok = s->sb != nil;
   return 0;
}

/* Sample attachment helpers: index or MTLCounterDontSample. */
static NSUInteger si(mscene *s, uint32_t slot, int timed, int idx)
{
   if (!timed || !s->ts_ok) return MTLCounterDontSample;
   if (s->frame_mode && idx != s->first_sample && idx != S_COMP1) return MTLCounterDontSample;
   return (NSUInteger)(slot * NSAMP + idx);
}

static id<MTLComputeCommandEncoder> compute_enc(mscene *s, id<MTLCommandBuffer> cb, uint32_t slot, int timed, int i0)
{
   MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
   d.dispatchType = MTLDispatchTypeSerial;
   if (timed && s->ts_ok) {
      d.sampleBufferAttachments[0].sampleBuffer = s->sb;
      d.sampleBufferAttachments[0].startOfEncoderSampleIndex = si(s, slot, timed, i0);
      d.sampleBufferAttachments[0].endOfEncoderSampleIndex = si(s, slot, timed, i0 + 1);
   }
   return [cb computeCommandEncoderWithDescriptor:d];
}

static id<MTLAccelerationStructureCommandEncoder> as_enc(mscene *s, id<MTLCommandBuffer> cb, uint32_t slot, int timed, int i0)
{
   MTLAccelerationStructurePassDescriptor *d = [MTLAccelerationStructurePassDescriptor accelerationStructurePassDescriptor];
   if (timed && s->ts_ok) {
      d.sampleBufferAttachments[0].sampleBuffer = s->sb;
      d.sampleBufferAttachments[0].startOfEncoderSampleIndex = si(s, slot, timed, i0);
      d.sampleBufferAttachments[0].endOfEncoderSampleIndex = si(s, slot, timed, i0 + 1);
   }
   return [cb accelerationStructureCommandEncoderWithDescriptor:d];
}

static void enc_deform(mscene *s, id<MTLComputeCommandEncoder> e, uint32_t nslots, float t)
{
   rt_deform_pc pc = {.vbase = s->sc.water_vbase, .grid = s->sc.water_grid, .first_slot = 0, .nslots = nslots,
                      .size = s->sc.water_size, .time = t};
   uint32_t vpp = (s->sc.water_grid + 1) * (s->sc.water_grid + 1);
   [e setComputePipelineState:s->p_def];
   [e setBuffer:s->vbuf offset:0 atIndex:0];
   [e setBytes:&pc length:sizeof(pc) atIndex:1];
   [e dispatchThreadgroups:MTLSizeMake((vpp + 63) / 64, nslots, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void enc_blas(mscene *s, id<MTLAccelerationStructureCommandEncoder> e, int init)
{
   size_t off = 0;
   for (uint32_t m = 0; m < s->sc.nmeshes; m++) {
      if (!init && !s->sc.meshes[m].dynamic) continue;
      if (!init && s->p->blas_mode == RT_AS_REFIT) {
         [e refitAccelerationStructure:s->blas[m] descriptor:s->bdesc[m] destination:nil scratchBuffer:s->scratch
                   scratchBufferOffset:s->bscratch_off[m]];
      } else {
         [e buildAccelerationStructure:s->blas[m] descriptor:s->bdesc[m] scratchBuffer:s->scratch
                   scratchBufferOffset:init ? off : s->bscratch_off[m]];
      }
      if (init) off += al256(s->bsizes[m].buildScratchBufferSize);
   }
}

static void enc_tlas(mscene *s, id<MTLAccelerationStructureCommandEncoder> e, uint32_t slot, int init)
{
   s->tdesc.instanceDescriptorBuffer = s->inst[slot];
   if (!init && s->p->tlas_mode == RT_AS_REFIT)
      [e refitAccelerationStructure:s->tlas descriptor:s->tdesc destination:nil scratchBuffer:s->scratch
                scratchBufferOffset:s->tscratch_off];
   else
      [e buildAccelerationStructure:s->tlas descriptor:s->tdesc scratchBuffer:s->scratch
                scratchBufferOffset:init ? 0 : s->tscratch_off];
}

static int init_build(mscene *s)
{
   int fm = s->frame_mode;
   s->frame_mode = 0; /* the initial build samples every encoder */
   for (uint32_t i = 0; i < s->nslots; i++) slot_update(s, i, 0, 1);
   uint64_t t0 = rt_now_ns();
   id<MTLCommandBuffer> cb = [g_queue commandBuffer];
   int timed = 1;
   id<MTLComputeCommandEncoder> ce = compute_enc(s, cb, 0, timed, S_DEF0);
   enc_deform(s, ce, RT_WATER_PATCHES, 0.0f);
   [ce endEncoding];
   id<MTLAccelerationStructureCommandEncoder> ae = as_enc(s, cb, 0, timed, S_BLAS0);
   enc_blas(s, ae, 1);
   [ae endEncoding];
   ae = as_enc(s, cb, 0, timed, S_TLAS0);
   enc_tlas(s, ae, 0, 1);
   [ae endEncoding];
   [cb commit];
   [cb waitUntilCompleted];
   s->frame_mode = fm;
   s->init_wall_ms = (rt_now_ns() - t0) / 1e6;
   if (cb.status == MTLCommandBufferStatusError) {
      fprintf(stderr, "initial build failed: %s\n", cb.error.localizedDescription.UTF8String);
      return -1;
   }
   if (s->ts_ok) {
      NSData *d = [s->sb resolveCounterRange:NSMakeRange(0, NSAMP)];
      const MTLCounterResultTimestamp *ts = d.bytes;
      double tk = tick_ns();
      if (ts) {
         s->init_ms[0] = (ts[S_DEF1].timestamp - ts[S_DEF0].timestamp) * tk / 1e6;
         s->init_ms[1] = (ts[S_BLAS1].timestamp - ts[S_BLAS0].timestamp) * tk / 1e6;
         s->init_ms[2] = (ts[S_TLAS1].timestamp - ts[S_TLAS0].timestamp) * tk / 1e6;
      }
   }
   return 0;
}

static int record_frame(void *user, uint32_t slot, uint64_t k, id<MTLCommandBuffer> cb, int timed)
{
   (void)k;
   mscene *s = user;
   const rt_params *p = s->p;
   uint32_t f = s->frame++;
   float t = rt_frame_time(f);
   slot_update(s, slot, f, 0);
   int dyn = s->sc.dyn_patches > 0, tdyn = p->tlas_mode != RT_AS_STATIC;
   s->first_sample = dyn ? S_DEF0 : tdyn ? S_TLAS0 : S_GB0;
   if (dyn) {
      id<MTLComputeCommandEncoder> e = compute_enc(s, cb, slot, timed, S_DEF0);
      enc_deform(s, e, s->sc.dyn_patches, t);
      [e endEncoding];
      id<MTLAccelerationStructureCommandEncoder> a = as_enc(s, cb, slot, timed, S_BLAS0);
      enc_blas(s, a, 0);
      [a endEncoding];
   }
   if (tdyn) {
      id<MTLAccelerationStructureCommandEncoder> a = as_enc(s, cb, slot, timed, S_TLAS0);
      enc_tlas(s, a, slot, 0);
      [a endEncoding];
   }
   /* G-buffer */
   MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture = s->albedo;
   rp.colorAttachments[0].loadAction = MTLLoadActionClear;
   rp.colorAttachments[0].storeAction = MTLStoreActionStore;
   rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
   rp.colorAttachments[1].texture = s->normal;
   rp.colorAttachments[1].loadAction = MTLLoadActionClear;
   rp.colorAttachments[1].storeAction = MTLStoreActionStore;
   rp.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0);
   rp.depthAttachment.texture = s->depth;
   rp.depthAttachment.loadAction = MTLLoadActionClear;
   rp.depthAttachment.storeAction = MTLStoreActionStore;
   rp.depthAttachment.clearDepth = 0.0;
   if (timed && s->ts_ok) {
      rp.sampleBufferAttachments[0].sampleBuffer = s->sb;
      rp.sampleBufferAttachments[0].startOfVertexSampleIndex = si(s, slot, timed, S_GB0);
      rp.sampleBufferAttachments[0].endOfVertexSampleIndex = MTLCounterDontSample;
      rp.sampleBufferAttachments[0].startOfFragmentSampleIndex = MTLCounterDontSample;
      rp.sampleBufferAttachments[0].endOfFragmentSampleIndex = si(s, slot, timed, S_GB1);
   }
   id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
   [re setDepthStencilState:s->ds];
   [re setFrontFacingWinding:MTLWindingCounterClockwise];
   [re setVertexBuffer:s->vbuf offset:0 atIndex:0];
   [re setVertexBuffer:s->ubo[slot] offset:0 atIndex:1];
   [re setVertexBuffer:s->xf[slot] offset:0 atIndex:2];
   [re setFragmentBuffer:s->ubo[slot] offset:0 atIndex:1];
   [re setFragmentBuffer:s->info offset:0 atIndex:3];
   id<MTLRenderPipelineState> bound = nil;
   for (uint32_t d = 0; d < s->sc.ndraws; d++) {
      rt_draw *dr = &s->sc.draws[d];
      rt_mesh *m = &s->sc.meshes[dr->mesh];
      id<MTLRenderPipelineState> want = dr->twosided ? s->p_gb_sail : s->p_gb_opaque;
      if (want != bound) {
         [re setRenderPipelineState:want];
         [re setCullMode:dr->twosided ? MTLCullModeNone : MTLCullModeBack];
         bound = want;
      }
      [re drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:m->index_count indexType:MTLIndexTypeUInt32
                    indexBuffer:s->ibuf indexBufferOffset:(NSUInteger)m->first_index * 4 instanceCount:dr->instance_count
                     baseVertex:(NSInteger)m->base_vertex baseInstance:dr->first_instance];
   }
   [re endEncoding];
   /* reflections and shadows */
   id<MTLComputeCommandEncoder> e = compute_enc(s, cb, slot, timed, S_RT0);
   [e setComputePipelineState:s->p_rt];
   [e setAccelerationStructure:s->tlas atBufferIndex:0];
   for (id<MTLAccelerationStructure> b in s->blas) [e useResource:b usage:MTLResourceUsageRead];
   [e setBuffer:s->ubo[slot] offset:0 atIndex:1];
   [e setTexture:s->albedo atIndex:0];
   [e setTexture:s->normal atIndex:1];
   [e setTexture:s->depth atIndex:2];
   [e setBuffer:s->vbuf offset:0 atIndex:2];
   [e setBuffer:s->ibuf offset:0 atIndex:3];
   [e setBuffer:s->info offset:0 atIndex:4];
   [e setBuffer:s->xf[slot] offset:0 atIndex:5];
   [e setTexture:s->rtimg atIndex:3];
   [e setBuffer:s->counters offset:0 atIndex:6];
   MTLSize tg = MTLSizeMake(8, 8, 1), grid = MTLSizeMake((p->width + 7) / 8, (p->height + 7) / 8, 1);
   [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
   [e endEncoding];
   e = compute_enc(s, cb, slot, timed, S_COMP0);
   [e setComputePipelineState:s->p_comp];
   [e setBuffer:s->ubo[slot] offset:0 atIndex:0];
   [e setTexture:s->albedo atIndex:0];
   [e setTexture:s->normal atIndex:1];
   [e setTexture:s->depth atIndex:2];
   [e setTexture:s->rtimg atIndex:3];
   [e setTexture:s->out atIndex:4];
   [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
   [e endEncoding];
   if (s->census) {
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b copyFromTexture:s->out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(p->width, p->height, 1) toBuffer:s->readback destinationOffset:0
       destinationBytesPerRow:p->width * 4 destinationBytesPerImage:(NSUInteger)p->width * p->height * 4];
      [b endEncoding];
   }
   return 0;
}

static void complete_frame(void *user, uint32_t slot, uint64_t k, int timed, id<MTLCommandBuffer> cb)
{
   (void)k; (void)cb;
   mscene *s = user;
   if (!timed || !s->ts_ok) return;
   NSData *d = [s->sb resolveCounterRange:NSMakeRange(slot * NSAMP, NSAMP)];
   const MTLCounterResultTimestamp *ts = d.bytes;
   if (!ts) return;
   int dyn = s->sc.dyn_patches > 0, tdyn = s->p->tlas_mode != RT_AS_STATIC;
   if (s->frame_mode) {
      double a = (double)ts[dyn ? S_DEF0 : tdyn ? S_TLAS0 : S_GB0].timestamp, e = (double)ts[S_COMP1].timestamp;
      if (a != 0 && e != 0 && a != (double)MTLCounterErrorValue && e != (double)MTLCounterErrorValue)
         rt_vec_push(&s->frame_us, (e - a) * s->tick / 1e3);
      return;
   }
   /* pass boundaries, as in the Vulkan executor: B0 = start, B[i+1] = end of pass i (or B[i] if empty) */
   double b[NPASS + 1];
   uint64_t first = dyn ? ts[S_DEF0].timestamp : tdyn ? ts[S_TLAS0].timestamp : ts[S_GB0].timestamp;
   b[0] = (double)first;
   b[1] = dyn ? (double)ts[S_DEF1].timestamp : b[0];
   b[2] = tdyn ? (double)ts[S_TLAS1].timestamp : dyn ? (double)ts[S_BLAS1].timestamp : b[1];
   b[3] = (double)ts[S_GB1].timestamp;
   b[4] = (double)ts[S_RT1].timestamp;
   b[5] = (double)ts[S_COMP1].timestamp;
   for (int i = 0; i <= NPASS; i++)
      if (b[i] == (double)MTLCounterErrorValue || b[i] == 0) return;
   for (int i = 0; i < NPASS; i++) rt_vec_push(&s->pass_us[i], fmax(0, b[i + 1] - b[i]) * s->tick / 1e3);
   rt_vec_push(&s->frame_us, (b[NPASS] - b[0]) * s->tick / 1e3);
}

static int scene_depth(mscene *s, uint32_t depth)
{
   const rt_run *r = s->r;
   const rt_params *p = s->p;
   uint64_t t_start = rt_now_ns();
   for (int i = 0; i < NPASS; i++) rt_vec_free(&s->pass_us[i]);
   rt_vec_free(&s->frame_us);
   s->frame = 0;
   s->census = 0;
   const char *therm0 = thermal_state();
   if (init_build(s)) {
      rt_result_fail(RT_TEST_SCENE, p, r->mode, depth, "initial acceleration-structure build failed");
      return -1;
   }
   s->tick = tick_ns();
   mtl_stream st = {.depth = depth, .next = 1, .unretained = 1, .user = s, .record = record_frame, .complete = complete_frame};
   mtl_stream_stats un = {0}, tm = {0};
   rt_vec win_low = {0};
   int rc = stream_phase(&st, 0, r->warmup_ms, 0, 0, NULL);
   size_t mark = 0;
   for (uint32_t w = 0; !rc && w < r->runs; w++) {
      rc = stream_phase(&st, 1, r->run_ms, r->frames, 0, &un);
      rt_vec_push(&win_low, low1(un.interval_us.v + mark, un.interval_us.n - mark));
      mark = un.interval_us.n;
   }
   if (!rc && r->timestamps) {
      s->tick = tick_ns();
      rc = stream_phase(&st, 3, r->run_ms, r->frames, 1, &tm);
   }
   if (!rc) rc = stream_drain(&st);
   uint32_t cnt[4] = {0};
   double refl_frac = -1;
   uint32_t crc = 0;
   if (!rc) {
      memset(s->counters.contents, 0, 16);
      s->census = 1;
      s->frame = 0; /* the census always renders frame 0: same image and rays on every device */
      rc = stream_submit(&st, 0);
      if (!rc) rc = stream_drain(&st);
      s->census = 0;
      if (!rc) {
         memcpy(cnt, s->counters.contents, 16);
         refl_frac = rt_alpha_fraction(s->readback.contents, p->width, p->height, p->width * 4);
         crc = rt_crc32(0, s->readback.contents, (size_t)p->width * p->height * 4);
         if (r->png[0]) {
            char path[600];
            if (r->ndepths > 1) snprintf(path, sizeof(path), "%.*s-d%u.png", (int)strcspn(r->png, "."), r->png, depth);
            else snprintf(path, sizeof(path), "%s", r->png);
            if (rt_write_png(path, s->readback.contents, p->width, p->height, p->width * 4))
               fprintf(stderr, "cannot write %s\n", path);
            else fprintf(stderr, "wrote %s\n", path);
         }
      }
   }
   if (rc) {
      rt_result_fail(RT_TEST_SCENE, p, r->mode, depth, "Metal command buffer error during the run");
      stats_free(&un); stats_free(&tm); rt_vec_free(&win_low);
      return -1;
   }
   rt_json j;
   rt_result_begin(&j, RT_TEST_SCENE, p, r->mode, depth);
   rt_json_str(&j, "status", "ok");
   rt_result_windows(&j, r);
   rt_json_str(&j, "primary", "fps");
   rt_json_obj(&j, "metrics");
   rt_metric(&j, "fps", "fps", "higher", un.rate.v, un.rate.n, 1);
   rt_metric(&j, "fps_1pct_low", "fps", "higher", win_low.v, win_low.n, 1);
   rt_vec fm = {0};
   for (size_t i = 0; i < un.interval_us.n; i++) rt_vec_push(&fm, un.interval_us.v[i] / 1e3);
   rt_metric(&j, "frame_ms", "ms", "lower", fm.v, fm.n, 0);
   rt_vec_free(&fm);
   emit_us(&j, "latency_us", &un.latency_us);
   emit_us(&j, "record_cpu_us", &un.record_us);
   emit_us(&j, "submit_cpu_us", &un.submit_us);
   rt_metric(&j, "fps_timed", "fps", "higher", tm.rate.v, tm.rate.n, 1);
   if (s->frame_us.n) emit_us(&j, "gpu_us", &s->frame_us);
   else emit_us(&j, "gpu_us", &tm.gpu_us);
   emit_us(&j, "gpu_cb_us", &tm.gpu_us);
   for (int i = 0; i < NPASS; i++) emit_us(&j, pass_metric[i], &s->pass_us[i]);
   uint64_t rays = (uint64_t)cnt[0] + cnt[1] + cnt[2];
   if (rays && s->pass_us[P_RT].n) {
      rt_vec mr = {0};
      for (size_t i = 0; i < s->pass_us[P_RT].n; i++)
         if (s->pass_us[P_RT].v[i] > 0) rt_vec_push(&mr, (double)rays / s->pass_us[P_RT].v[i]);
      rt_metric(&j, "mrays_s", "Mray/s", "higher", mr.v, mr.n, 0);
      rt_vec_free(&mr);
   }
   rt_json_end(&j);
   rt_json_obj(&j, "counts");
   rt_json_int(&j, "frames", (int64_t)un.completions);
   rt_json_int(&j, "frames_timed", (int64_t)tm.completions);
   rt_json_end(&j);
   rt_json_obj(&j, "info");
   char os[128];
   os_version(os, sizeof(os));
   rt_json_str(&j, "api", "metal");
   rt_json_str(&j, "device", g_dev.name.UTF8String);
   rt_json_str(&j, "driver", os);
   rt_json_obj(&j, "scene");
   rt_json_int(&j, "unique_tris", (int64_t)s->sc.unique_tris);
   rt_json_int(&j, "instanced_tris", (int64_t)s->sc.instanced_tris);
   rt_json_int(&j, "vertices", s->sc.nverts);
   rt_json_int(&j, "blas", s->sc.nmeshes);
   rt_json_int(&j, "tlas_instances", s->sc.ninst);
   rt_json_int(&j, "moving_instances", s->sc.nmoving);
   rt_json_int(&j, "dyn_patches", s->sc.dyn_patches);
   rt_json_int(&j, "draws", s->sc.ndraws);
   rt_json_end(&j);
   rt_json_obj(&j, "as");
   uint64_t blas_bytes = 0, blas_alloc = 0;
   for (uint32_t m = 0; m < s->sc.nmeshes; m++) {
      blas_bytes += s->bsizes[m].accelerationStructureSize;
      blas_alloc += s->blas[m].allocatedSize;
   }
   rt_json_int(&j, "blas_bytes", (int64_t)blas_bytes);
   rt_json_int(&j, "blas_allocated_bytes", (int64_t)blas_alloc);
   rt_json_int(&j, "tlas_bytes", (int64_t)s->tsizes.accelerationStructureSize);
   rt_json_int(&j, "init_scratch_bytes", (int64_t)s->init_scratch);
   rt_json_int(&j, "frame_scratch_bytes", (int64_t)s->frame_scratch);
   rt_json_int(&j, "tlas_build_scratch_bytes", (int64_t)s->tsizes.buildScratchBufferSize);
   rt_json_num(&j, "init_deform_gpu_ms", s->init_ms[0]);
   rt_json_num(&j, "init_blas_gpu_ms", s->init_ms[1]);
   rt_json_num(&j, "init_tlas_gpu_ms", s->init_ms[2]);
   rt_json_num(&j, "init_wall_ms", s->init_wall_ms);
   rt_json_str(&j, "blas_flags", rt_as_mode_name(p->blas_mode));
   rt_json_str(&j, "tlas_flags", rt_as_mode_name(p->tlas_mode));
   rt_json_end(&j);
   rt_json_obj(&j, "rays");
   rt_json_int(&j, "primary_shadow", cnt[0]);
   rt_json_int(&j, "reflection", cnt[1]);
   rt_json_int(&j, "secondary_shadow", cnt[2]);
   rt_json_int(&j, "per_frame", (int64_t)rays);
   rt_json_num(&j, "reflective_pixel_frac", refl_frac);
   rt_json_end(&j);
   char crcs[16];
   snprintf(crcs, sizeof(crcs), "%08x", crc);
   rt_json_str(&j, "image_crc32", crcs);
   rt_json_str(&j, "timestamps", s->ts_ok ? (r->timestamps ? "on" : "off") : "cb-only");
   rt_json_num(&j, "gpu_tick_ns", s->tick);
   rt_json_end(&j);
   char th[128];
   snprintf(th, sizeof(th), "{\"thermal_state\":\"%s\"}", therm0);
   rt_json_raw(&j, "thermal_before", th);
   snprintf(th, sizeof(th), "{\"thermal_state\":\"%s\"}", thermal_state());
   rt_json_raw(&j, "thermal_after", th);
   rt_json_num(&j, "elapsed_s", (rt_now_ns() - t_start) / 1e9);
   rt_result_print(&j);
   stats_free(&un); stats_free(&tm); rt_vec_free(&win_low);
   return 0;
}

static int run_scene(const rt_params *p, const rt_run *r)
{
   if (strcmp(r->mode, "queue")) {
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "rt.scene runs only in mode=queue");
      return 0;
   }
   if (!g_dev.supportsRaytracing) {
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "device lacks ray tracing");
      return 0;
   }
   if (p->trace_mode == RT_TRACE_PIPELINE) {
      for (uint32_t d = 0; d < r->ndepths; d++)
         rt_result_skip(RT_TEST_SCENE, p, r->mode, r->depths[d], "Metal has no ray-tracing pipeline (trace=pipeline is Vulkan only)");
      return 0;
   }
   mscene *s = calloc(1, sizeof(mscene));
   s->p = p;
   s->r = r;
   s->frame_mode = r->stamps_frame;
   uint32_t maxd = 1;
   for (uint32_t d = 0; d < r->ndepths; d++) if (r->depths[d] > maxd) maxd = r->depths[d];
   if (maxd > MAXD) maxd = MAXD;
   rt_scene_build(&s->sc, p);
   fprintf(stderr, "rt.scene: %u unique tris, %llu instanced tris, %u instances, %u BLAS\n", (unsigned)s->sc.unique_tris,
           (unsigned long long)s->sc.instanced_tris, s->sc.ninst, s->sc.nmeshes);
   uint64_t t0 = rt_now_ns();
   if (create_scene(s, maxd)) {
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_fail(RT_TEST_SCENE, p, r->mode, r->depths[d], "setup failed");
   } else {
      fprintf(stderr, "setup %.1f ms\n", (rt_now_ns() - t0) / 1e6);
      for (uint32_t d = 0; d < r->ndepths; d++)
         if (scene_depth(s, r->depths[d] > maxd ? maxd : r->depths[d])) break;
   }
   free(s->bsizes);
   free(s->bscratch_off);
   for (uint32_t m = 0; m < s->sc.nmeshes; m++) s->bdesc[m] = nil;
   free((void *)s->bdesc);
   rt_scene_free(&s->sc);
   free(s);
   return 0;
}

/* ---------------- micro-benchmarks ---------------- */

typedef struct mmicro {
   const rt_params *p;
   int test, refit, tlas;
   uint32_t prims, nverts;
   id<MTLBuffer> vbuf, ibuf, instb, out[MAXD], scratch[MAXD];
   id<MTLAccelerationStructure> blas, as[MAXD];
   MTLAccelerationStructureDescriptor *desc;
   MTLAccelerationStructureSizes sizes;
   /* rays */
   uint64_t compacted;
   mscene *scene;
   id<MTLComputePipelineState> pipe;
   struct { float eye[4], fwd[4], right[4], up[4]; uint32_t mode, nrays, seed, width; } pc;
} mmicro;

static int record_build(void *user, uint32_t slot, uint64_t k, id<MTLCommandBuffer> cb, int timed)
{
   (void)k; (void)timed;
   mmicro *m = user;
   id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
   if (m->tlas) [e useResource:m->blas usage:MTLResourceUsageRead];
   if (m->refit)
      [e refitAccelerationStructure:m->as[slot] descriptor:m->desc destination:nil scratchBuffer:m->scratch[slot] scratchBufferOffset:0];
   else
      [e buildAccelerationStructure:m->as[slot] descriptor:m->desc scratchBuffer:m->scratch[slot] scratchBufferOffset:0];
   [e endEncoding];
   return 0;
}

static int record_rays(void *user, uint32_t slot, uint64_t k, id<MTLCommandBuffer> cb, int timed)
{
   (void)timed;
   mmicro *m = user;
   m->pc.seed = (uint32_t)k;
   id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
   [e setComputePipelineState:m->pipe];
   [e setAccelerationStructure:m->scene->tlas atBufferIndex:0];
   for (id<MTLAccelerationStructure> b in m->scene->blas) [e useResource:b usage:MTLResourceUsageRead];
   [e setBuffer:m->out[slot] offset:0 atIndex:1];
   [e setBytes:&m->pc length:sizeof(m->pc) atIndex:2];
   [e dispatchThreadgroups:MTLSizeMake((m->p->rays + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
   [e endEncoding];
   return 0;
}

static int setup_asbuild(mmicro *m, uint32_t depth)
{
   const rt_params *p = m->p;
   m->tlas = !strcmp(p->kind, "tlas");
   m->refit = !strcmp(p->kind, "refit");
   int fastbuild = !strcmp(p->build, "fastbuild");
   rt_vertex *v;
   uint32_t *idx, nv, ni;
   rt_gen_mesh(RT_KIND_ROCK, m->tlas ? 1024 : p->tris, &v, &nv, &idx, &ni);
   m->vbuf = buf((size_t)nv * sizeof(rt_vertex), 1, 0);
   memcpy(m->vbuf.contents, v, (size_t)nv * sizeof(rt_vertex));
   m->ibuf = buf((size_t)ni * 4, 1, 0);
   memcpy(m->ibuf.contents, idx, (size_t)ni * 4);
   free(v);
   free(idx);
   m->nverts = nv;
   MTLAccelerationStructureTriangleGeometryDescriptor *g = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
   g.vertexBuffer = m->vbuf;
   g.vertexStride = sizeof(rt_vertex);
   g.vertexFormat = MTLAttributeFormatFloat3;
   g.indexBuffer = m->ibuf;
   g.indexType = MTLIndexTypeUInt32;
   g.triangleCount = ni / 3;
   g.opaque = YES;
   MTLPrimitiveAccelerationStructureDescriptor *pd = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
   pd.geometryDescriptors = @[ g ];
   if (m->tlas) {
      MTLAccelerationStructureSizes bs = [g_dev accelerationStructureSizesWithDescriptor:pd];
      m->blas = [g_dev newAccelerationStructureWithSize:bs.accelerationStructureSize];
      id<MTLBuffer> scr = buf(bs.buildScratchBufferSize, 0, 0);
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
      [e buildAccelerationStructure:m->blas descriptor:pd scratchBuffer:scr scratchBufferOffset:0];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      uint32_t n = p->tris;
      m->instb = buf((size_t)n * sizeof(MTLAccelerationStructureInstanceDescriptor), 1, 0);
      MTLAccelerationStructureInstanceDescriptor *in = m->instb.contents;
      uint32_t side = (uint32_t)ceil(sqrt((double)n));
      for (uint32_t i = 0; i < n; i++) {
         memset(&in[i], 0, sizeof(in[i]));
         float x = (float)(i % side) * 3.0f, z = (float)(i / side) * 3.0f;
         float a = (float)((i * 2654435761u) >> 8) * (6.2831853f / 16777216.0f);
         /* same rows as the Vulkan instances: [cos 0 sin x; 0 1 0 y; -sin 0 cos z] */
         float r[12] = {cosf(a), 0, sinf(a), x, 0, 1, 0, (float)(i % 7) * 0.5f, -sinf(a), 0, cosf(a), z};
         for (int c = 0; c < 4; c++)
            for (int rr = 0; rr < 3; rr++) in[i].transformationMatrix.columns[c].elements[rr] = r[rr * 4 + c];
         in[i].mask = 0xff;
         in[i].accelerationStructureIndex = 0;
      }
      MTLInstanceAccelerationStructureDescriptor *td = [MTLInstanceAccelerationStructureDescriptor descriptor];
      td.instancedAccelerationStructures = @[ m->blas ];
      td.instanceCount = n;
      td.instanceDescriptorBuffer = m->instb;
      td.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
      td.usage = fastbuild ? MTLAccelerationStructureUsagePreferFastBuild : MTLAccelerationStructureUsageNone;
      /* Metal has no compaction-allowed flag: any structure can be compacted */
      m->desc = td;
      m->prims = n;
   } else {
      pd.usage = (fastbuild ? MTLAccelerationStructureUsagePreferFastBuild : MTLAccelerationStructureUsageNone) |
                 (m->refit ? MTLAccelerationStructureUsageRefit : 0);
      m->desc = pd;
      m->prims = ni / 3;
   }
   m->sizes = [g_dev accelerationStructureSizesWithDescriptor:m->desc];
   size_t scr = m->sizes.buildScratchBufferSize;
   if (m->refit && m->sizes.refitScratchBufferSize > scr) scr = m->sizes.refitScratchBufferSize;
   for (uint32_t i = 0; i < depth; i++) {
      m->as[i] = [g_dev newAccelerationStructureWithSize:m->sizes.accelerationStructureSize];
      m->scratch[i] = buf(scr, 0, 0);
      if (!m->as[i] || !m->scratch[i]) return -1;
   }
   if (m->refit) {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
      for (uint32_t i = 0; i < depth; i++)
         [e buildAccelerationStructure:m->as[i] descriptor:m->desc scratchBuffer:m->scratch[i] scratchBufferOffset:0];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
   }
   return 0;
}

static int setup_rays(mmicro *m, uint32_t depth)
{
   const rt_params *p = m->p;
   mscene *s = calloc(1, sizeof(mscene));
   static rt_params pc;
   pc = *p;
   pc.blas_mode = pc.tlas_mode = RT_AS_STATIC;
   pc.dyn_patches = 0;
   pc.moving = 0;
   pc.width = pc.height = 64;
   s->p = &pc;
   rt_scene_build(&s->sc, s->p);
   if (create_scene(s, 1) || init_build(s)) return -1;
   m->scene = s;
   NSError *err = nil;
   m->pipe = [g_dev newComputePipelineStateWithFunction:[g_lib newFunctionWithName:@"rays"] error:&err];
   if (!m->pipe) return -1;
   for (uint32_t i = 0; i < depth; i++) m->out[i] = buf((size_t)p->rays * 4, 0, 0);
   uint32_t width = 1024, h = (p->rays + width - 1) / width;
   rt_scene_camera(&s->sc, 0, (float)width / (float)h, m->pc.eye, m->pc.fwd, m->pc.right, m->pc.up);
   m->pc.eye[3] = p->tmax;
   m->pc.mode = !strcmp(p->kind, "incoherent");
   m->pc.nrays = p->rays;
   m->pc.width = width;
   return 0;
}

static int micro_depth(const rt_params *p, const rt_run *r, int test, uint32_t depth)
{
   mmicro m = {0};
   m.p = p;
   m.test = test;
   if (depth > MAXD) depth = MAXD;
   uint64_t t0 = rt_now_ns();
   if ((test == RT_TEST_ASBUILD ? setup_asbuild(&m, depth) : setup_rays(&m, depth))) {
      rt_result_fail(test, p, r->mode, depth, "setup failed");
      return -1;
   }
   mtl_stream st = {.depth = depth, .next = 1, .unretained = 1, .user = &m,
                    .record = test == RT_TEST_ASBUILD ? record_build : record_rays};
   mtl_stream_stats un = {0}, tm = {0};
   int rc = stream_phase(&st, 0, r->warmup_ms, 0, 0, NULL);
   if (!rc) rc = stream_phase(&st, r->runs, r->run_ms, r->frames, 0, &un);
   if (!rc && r->timestamps) rc = stream_phase(&st, 3, r->run_ms, r->frames, 1, &tm);
   if (!rc) rc = stream_drain(&st);
   if (!rc && test == RT_TEST_ASBUILD && p->compact) {
      id<MTLBuffer> sz = buf(8, 1, 1);
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> e = [cb accelerationStructureCommandEncoder];
      [e writeCompactedAccelerationStructureSize:m.as[0] toBuffer:sz offset:0 sizeDataType:MTLDataTypeULong];
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      memcpy(&m.compacted, sz.contents, 8);
   }
   if (rc) {
      rt_result_fail(test, p, r->mode, depth, "Metal command buffer error during the run");
      stats_free(&un); stats_free(&tm);
      return -1;
   }
   rt_json j;
   rt_result_begin(&j, test, p, r->mode, depth);
   rt_json_str(&j, "status", "ok");
   rt_result_windows(&j, r);
   int tlas = test == RT_TEST_ASBUILD && m.tlas;
   const char *rate_name = test == RT_TEST_RAYS ? "mrays_s" : tlas ? "minst_s" : "mtri_s";
   rt_json_str(&j, "primary", test == RT_TEST_RAYS ? "mrays_s" : r->timestamps ? rate_name : "us_per_submit");
   rt_json_obj(&j, "metrics");
   rt_vec ups = {0}, ups_t = {0}, rate = {0};
   double work = test == RT_TEST_RAYS ? (double)p->rays : (double)m.prims;
   for (size_t i = 0; i < un.rate.n; i++) rt_vec_push(&ups, 1e6 / un.rate.v[i]);
   for (size_t i = 0; i < tm.rate.n; i++) rt_vec_push(&ups_t, 1e6 / tm.rate.v[i]);
   if (test == RT_TEST_RAYS)
      for (size_t i = 0; i < un.rate.n; i++) rt_vec_push(&rate, un.rate.v[i] * work / 1e6);
   else
      for (size_t i = 0; i < tm.gpu_us.n; i++) if (tm.gpu_us.v[i] > 0) rt_vec_push(&rate, work / tm.gpu_us.v[i]);
   rt_metric(&j, "us_per_submit", "us", "lower", ups.v, ups.n, 1);
   rt_metric(&j, rate_name, test == RT_TEST_RAYS ? "Mray/s" : tlas ? "Minst/s" : "Mtri/s", "higher", rate.v, rate.n,
             test == RT_TEST_RAYS);
   rt_metric(&j, "latency_us", "us", "lower", un.latency_us.v, un.latency_us.n, 0);
   rt_metric(&j, "record_cpu_us", "us", "lower", un.record_us.v, un.record_us.n, 0);
   rt_metric(&j, "submit_cpu_us", "us", "lower", un.submit_us.v, un.submit_us.n, 0);
   rt_metric(&j, "gpu_us", "us", "lower", tm.gpu_us.v, tm.gpu_us.n, 0);
   rt_metric(&j, "gpu_gap_us", "us", "lower", tm.gap_us.v, tm.gap_us.n, 0);
   rt_metric(&j, "us_per_submit_timed", "us", "lower", ups_t.v, ups_t.n, 1);
   if (test == RT_TEST_RAYS && tm.gpu_us.n) {
      rt_vec gr = {0};
      for (size_t i = 0; i < tm.gpu_us.n; i++) if (tm.gpu_us.v[i] > 0) rt_vec_push(&gr, work / tm.gpu_us.v[i]);
      rt_metric(&j, "mrays_s_gpu", "Mray/s", "higher", gr.v, gr.n, 0);
      rt_vec_free(&gr);
   }
   rt_json_end(&j);
   rt_vec_free(&ups); rt_vec_free(&ups_t); rt_vec_free(&rate);
   rt_json_obj(&j, "counts");
   rt_json_int(&j, "submits", (int64_t)un.completions);
   rt_json_int(&j, "submits_timed", (int64_t)tm.completions);
   rt_json_end(&j);
   rt_json_obj(&j, "info");
   char os[128];
   os_version(os, sizeof(os));
   rt_json_str(&j, "api", "metal");
   rt_json_str(&j, "device", g_dev.name.UTF8String);
   rt_json_str(&j, "driver", os);
   if (test == RT_TEST_ASBUILD) {
      rt_json_int(&j, tlas ? "instances" : "triangles", m.prims);
      if (!tlas) rt_json_int(&j, "vertices", m.nverts);
      rt_json_int(&j, "as_bytes", (int64_t)m.sizes.accelerationStructureSize);
      rt_json_int(&j, "build_scratch_bytes", (int64_t)m.sizes.buildScratchBufferSize);
      rt_json_int(&j, "update_scratch_bytes", (int64_t)m.sizes.refitScratchBufferSize);
      if (p->compact) rt_json_int(&j, "compacted_bytes", (int64_t)m.compacted);
   } else {
      rt_json_int(&j, "scene_unique_tris", (int64_t)m.scene->sc.unique_tris);
      rt_json_int(&j, "scene_instanced_tris", (int64_t)m.scene->sc.instanced_tris);
      rt_json_int(&j, "scene_instances", m.scene->sc.ninst);
      rt_json_num(&j, "scene_build_gpu_ms", m.scene->init_ms[1] + m.scene->init_ms[2]);
   }
   rt_json_end(&j);
   rt_json_num(&j, "elapsed_s", (rt_now_ns() - t0) / 1e9);
   rt_result_print(&j);
   stats_free(&un); stats_free(&tm);
   if (m.scene) {
      rt_scene_free(&m.scene->sc);
      free(m.scene->bsizes);
      free(m.scene->bscratch_off);
      for (uint32_t i = 0; m.scene->bdesc && i < m.scene->blas.count; i++) m.scene->bdesc[i] = nil;
      free((void *)m.scene->bdesc);
      free(m.scene);
   }
   return 0;
}

static int run_micro(const rt_params *p, const rt_run *r, int test)
{
   if (strcmp(r->mode, "queue")) {
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_skip(test, p, r->mode, r->depths[d], "rt micro-tests run only in mode=queue");
      return 0;
   }
   if (!g_dev.supportsRaytracing) {
      for (uint32_t d = 0; d < r->ndepths; d++) rt_result_skip(test, p, r->mode, r->depths[d], "device lacks ray tracing");
      return 0;
   }
   if (test == RT_TEST_ASBUILD && strcmp(p->kind, "blas") && strcmp(p->kind, "refit") && strcmp(p->kind, "tlas")) {
      rt_result_fail(test, p, r->mode, r->depths[0], "kind must be blas, refit or tlas");
      return 0;
   }
   if (test == RT_TEST_RAYS && strcmp(p->kind, "coherent") && strcmp(p->kind, "incoherent")) {
      rt_result_fail(test, p, r->mode, r->depths[0], "kind must be coherent or incoherent");
      return 0;
   }
   for (uint32_t d = 0; d < r->ndepths; d++)
      @autoreleasepool {
         if (micro_depth(p, r, test, r->depths[d])) break;
      }
   return 0;
}

/* --dump-archive PREFIX: compile every pipeline of the benchmark (as the scene and the
 * micro-tests create them) into its own MTLBinaryArchive and serialize it to PREFIX-<name>.bin.
 * On the target device this is the OS compiler's native code for these exact shaders
 * (rt/FAIRNESS.md, decoded with tools/decode-g17p.py). */
static int rt_serialize_archive(id<MTLBinaryArchive> ar, const char *prefix, const char *name)
{
   NSError *err = nil;
   char path[700];
   snprintf(path, sizeof(path), "%s-%s.bin", prefix, name);
   if (![ar serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err]) {
      fprintf(stderr, "serialize %s: %s\n", path, err.localizedDescription.UTF8String);
      return 1;
   }
   fprintf(stderr, "wrote %s\n", path);
   return 0;
}

static int dump_archive(const rt_params *p, const char *prefix)
{
   (void)p;
   int rc = 0;
   NSError *err = nil;
   fprintf(stderr, "device %s\n", g_dev.name.UTF8String);
   const char *kernels[] = {"deform", "rt_reflect", "composite", "rays"};
   for (int i = 0; i < 4; i++) {
      id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
      MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
      cd.computeFunction = [g_lib newFunctionWithName:[NSString stringWithUTF8String:kernels[i]]];
      if (!ar || ![ar addComputePipelineFunctionsWithDescriptor:cd error:&err]) {
         fprintf(stderr, "archive %s: %s\n", kernels[i], err.localizedDescription.UTF8String);
         rc = 1;
         continue;
      }
      rc |= rt_serialize_archive(ar, prefix, kernels[i]);
   }
   MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
   vd.attributes[0].format = MTLVertexFormatFloat3; vd.attributes[0].offset = 0; vd.attributes[0].bufferIndex = 0;
   vd.attributes[1].format = MTLVertexFormatFloat3; vd.attributes[1].offset = 12; vd.attributes[1].bufferIndex = 0;
   vd.attributes[2].format = MTLVertexFormatFloat2; vd.attributes[2].offset = 24; vd.attributes[2].bufferIndex = 0;
   vd.layouts[0].stride = sizeof(rt_vertex);
   for (int alpha = 0; alpha < 2; alpha++) {
      MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
      bool a = alpha;
      [fc setConstantValue:&a type:MTLDataTypeBool atIndex:0];
      MTLRenderPipelineDescriptor *rd = [MTLRenderPipelineDescriptor new];
      rd.vertexFunction = [g_lib newFunctionWithName:@"gbuffer_vs"];
      rd.fragmentFunction = [g_lib newFunctionWithName:@"gbuffer_fs" constantValues:fc error:&err];
      rd.vertexDescriptor = vd;
      rd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      rd.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA16Float;
      rd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
      id<MTLBinaryArchive> ar = [g_dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
      if (!ar || ![ar addRenderPipelineFunctionsWithDescriptor:rd error:&err]) {
         fprintf(stderr, "archive gbuffer: %s\n", err.localizedDescription.UTF8String);
         rc = 1;
         continue;
      }
      rc |= rt_serialize_archive(ar, prefix, alpha ? "gbuffer_alpha" : "gbuffer_opaque");
   }
   return rc;
}

int main(int argc, char **argv)
{
   @autoreleasepool {
      rt_params p;
      rt_run r;
      int pr = rt_parse(&p, &r, argc, argv, "gpubench-rt-mtl");
      if (pr) return pr > 0 ? 0 : 2;
      if (r.list) { rt_print_list(); return 0; }
      if (r.info) { print_info(); return 0; }
      rt_run_defaults(&r);
      if (r.window) fprintf(stderr, "note: --window is not implemented in the Metal executor; running headless\n");
      if (metal_init(&r)) {
         fprintf(stderr, "Metal setup failed\n");
         return 1;
      }
      fprintf(stderr, "device: %s, stage-boundary counters: %s\n", g_dev.name.UTF8String, g_stage_sampling ? "yes" : "no");
      if (r.dump_archive[0]) return dump_archive(&p, r.dump_archive);
      if (r.test == RT_TEST_SCENE) return run_scene(&p, &r);
      return run_micro(&p, &r, r.test);
   }
}
