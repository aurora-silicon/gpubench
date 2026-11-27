// SPDX-License-Identifier: GPL-2.0-only
/* Real Direct3D 11 benchmark executor for gpubench.
 * Runs natively on Linux against real upstream DXVK (dxvk-native: libdxvk_d3d11.so
 * + libdxvk_dxgi.so) and builds from the exact same source as a Windows PE binary
 * (gpubench-d3d11.exe) for Wine/Proton + DXVK or macOS D3DMetal/GPTK. */

#define INITGUID
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sched.h>
#endif
#include <d3d11_1.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gb_d3d_common.h"
#include "shaders_dxbc.h"

#ifndef GB_SUITE_SHA
#define GB_SUITE_SHA "unknown"
#endif

template <typename T>
struct ComPtr {
   T *p = nullptr;
   ComPtr() = default;
   ~ComPtr() { reset(); }
   ComPtr(const ComPtr &) = delete;
   ComPtr &operator=(const ComPtr &) = delete;
   ComPtr(ComPtr &&o) noexcept : p(o.p) { o.p = nullptr; }
   ComPtr &operator=(ComPtr &&o) noexcept {
      if (this != &o) { reset(); p = o.p; o.p = nullptr; }
      return *this;
   }
   void reset() { if (p) { p->Release(); p = nullptr; } }
   T *get() const { return p; }
   T *operator->() const { return p; }
   T **put() { reset(); return &p; }
   explicit operator bool() const { return p != nullptr; }
};

static void
hr_check(HRESULT hr, const char *what)
{
   if (FAILED(hr))
      gb_die("D3D11 %s failed: HRESULT 0x%08x", what, (unsigned)hr);
}

struct d3d11_ctx {
   ComPtr<IDXGIFactory1> factory;
   ComPtr<IDXGIAdapter1> adapter;
   ComPtr<ID3D11Device> dev;
   ComPtr<ID3D11DeviceContext> ctx;
   ComPtr<ID3D11DeviceContext1> ctx1;
   D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
   char adapter_name[256] = {};
   uint32_t vendor_id = 0, device_id = 0;
   uint64_t vram_bytes = 0;
};

static void
wide_to_utf8(const WCHAR *src, char *dst, size_t dst_sz)
{
   size_t j = 0;
   for (size_t i = 0; src && src[i] && j + 1 < dst_sz; i++) {
      unsigned ch = (unsigned)src[i];
      dst[j++] = (ch >= 0x20 && ch < 0x7f) ? (char)ch : '?';
   }
   dst[j] = 0;
}

static void
d3d11_init(d3d11_ctx &c, int dev_idx)
{
#ifndef _WIN32
   if (!getenv("DXVK_WSI_DRIVER"))
      setenv("DXVK_WSI_DRIVER", "Wayland", 0);
#endif
   hr_check(CreateDXGIFactory1(IID_IDXGIFactory1, (void **)c.factory.put()), "CreateDXGIFactory1");
   HRESULT hr = c.factory->EnumAdapters1((UINT)dev_idx, c.adapter.put());
   if (FAILED(hr))
      gb_die("DXGI adapter %d not found (HRESULT 0x%08x)", dev_idx, (unsigned)hr);

   DXGI_ADAPTER_DESC1 desc = {};
   c.adapter->GetDesc1(&desc);
   wide_to_utf8(desc.Description, c.adapter_name, sizeof(c.adapter_name));
   c.vendor_id = desc.VendorId;
   c.device_id = desc.DeviceId;
   c.vram_bytes = (uint64_t)desc.DedicatedVideoMemory;

   const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
   hr = D3D11CreateDevice(c.adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
                          levels, 2, D3D11_SDK_VERSION, c.dev.put(), &c.fl, c.ctx.put());
   if (FAILED(hr)) {
      hr_check(D3D11CreateDevice(c.adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
                                 levels + 1, 1, D3D11_SDK_VERSION, c.dev.put(), &c.fl, c.ctx.put()),
               "D3D11CreateDevice");
   }
   c.ctx->QueryInterface(IID_ID3D11DeviceContext1, (void **)c.ctx1.put());
}

static void
emit_identity(struct gb_json *j, const d3d11_ctx &c)
{
   gb_json_str(j, "api", "d3d11");
#ifdef _WIN32
   gb_json_str(j, "backend", "d3d11-win32");
#else
   gb_json_str(j, "backend", "dxvk-native");
#endif
   gb_json_str(j, "adapter", c.adapter_name);
   gb_json_int(j, "vendor_id", c.vendor_id);
   gb_json_int(j, "device_id", c.device_id);
   gb_json_int(j, "vram_mib", (int64_t)(c.vram_bytes >> 20));
   gb_json_str(j, "feature_level", c.fl == D3D_FEATURE_LEVEL_11_1 ? "11_1" : "11_0");
   gb_json_str(j, "suite_sha", GB_SUITE_SHA);
   const char *vk_drv = getenv("VK_DRIVER_FILES");
   gb_json_str(j, "vk_driver_files", vk_drv ? vk_drv : "");
}

static bool
wait_event_query(ID3D11DeviceContext *ctx, ID3D11Query *q)
{
   uint64_t deadline = gb_now_ns() + 10000000000ull;
   BOOL done = FALSE;
   for (unsigned iter = 0;; iter++) {
      UINT flags = (iter == 0 || (iter & 63u) == 0) ? 0u : (UINT)D3D11_ASYNC_GETDATA_DONOTFLUSH;
      HRESULT hr = ctx->GetData(q, &done, sizeof(done), flags);
      if (hr == S_OK && done) return true;
      if (FAILED(hr)) return false;
      if (gb_now_ns() > deadline) return false;
#ifdef _WIN32
      SwitchToThread();
#else
      sched_yield();
#endif
   }
}

static bool
wait_query_data(ID3D11DeviceContext *ctx, ID3D11Query *q, void *out, UINT sz)
{
   uint64_t deadline = gb_now_ns() + 10000000000ull;
   for (unsigned iter = 0;; iter++) {
      UINT flags = (iter == 0 || (iter & 63u) == 0) ? 0u : (UINT)D3D11_ASYNC_GETDATA_DONOTFLUSH;
      HRESULT hr = ctx->GetData(q, out, sz, flags);
      if (hr == S_OK) return true;
      if (FAILED(hr)) return false;
      if (gb_now_ns() > deadline) return false;
#ifdef _WIN32
      SwitchToThread();
#else
      sched_yield();
#endif
   }
}

static void
gpu_idle(d3d11_ctx &c)
{
   D3D11_QUERY_DESC qd = {};
   qd.Query = D3D11_QUERY_EVENT;
   ComPtr<ID3D11Query> q;
   hr_check(c.dev->CreateQuery(&qd, q.put()), "CreateQuery(EVENT)");
   c.ctx->End(q.get());
   c.ctx->Flush();
   if (!wait_event_query(c.ctx.get(), q.get()))
      gb_die("D3D11 GPU idle wait timed out");
}

typedef void (*d11_rec_fn)(d3d11_ctx &c, void *user, uint64_t k);

static void
run_stream(d3d11_ctx &c, int depth, bool timed, int warm_ms, int runs, int run_ms,
           d11_rec_fn rec, void *user, struct gb_pass *p)
{
   struct slot {
      ComPtr<ID3D11Query> ev, ts0, ts1, disj;
      uint64_t submit_ns = 0;
      bool busy = false;
   };
   std::vector<slot> slots((size_t)depth);
   memset(p, 0, sizeof(*p));
   p->nwin = runs;
   p->win_s = run_ms / 1e3;
   p->timed = timed;

   D3D11_QUERY_DESC qev = {}; qev.Query = D3D11_QUERY_EVENT;
   D3D11_QUERY_DESC qts = {}; qts.Query = D3D11_QUERY_TIMESTAMP;
   D3D11_QUERY_DESC qdj = {}; qdj.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
   for (int i = 0; i < depth; i++) {
      hr_check(c.dev->CreateQuery(&qev, slots[i].ev.put()), "CreateQuery(EVENT)");
      if (timed) {
         hr_check(c.dev->CreateQuery(&qts, slots[i].ts0.put()), "CreateQuery(TS0)");
         hr_check(c.dev->CreateQuery(&qts, slots[i].ts1.put()), "CreateQuery(TS1)");
         hr_check(c.dev->CreateQuery(&qdj, slots[i].disj.put()), "CreateQuery(DISJ)");
      }
   }

   const uint64_t start = gb_now_ns() + (uint64_t)warm_ms * 1000000ull;
   const uint64_t win = (uint64_t)run_ms * 1000000ull;
   const uint64_t end = start + (uint64_t)runs * win;
   uint64_t sent = 0, done = 0;
   double last_end_us = 0.0;
   bool have_last_end = false;

   while (true) {
      uint64_t now = gb_now_ns();
      bool full = (sent - done) >= (uint64_t)depth;
      if (!full && now >= end) break;
      if (full) {
         slot &s = slots[done % (uint64_t)depth];
         if (!wait_event_query(c.ctx.get(), s.ev.get())) { p->failed = true; break; }
         uint64_t t = gb_now_ns();
         bool measured = (t >= start && t < end);
         if (measured) {
            p->win_count[(t - start) / win]++;
            gb_samples_push(&p->latency, (double)(t - s.submit_ns) / 1e3);
         }
         if (timed) {
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
            uint64_t t0 = 0, t1 = 0;
            if (!wait_query_data(c.ctx.get(), s.disj.get(), &dj, sizeof(dj)) ||
                !wait_query_data(c.ctx.get(), s.ts0.get(), &t0, sizeof(t0)) ||
                !wait_query_data(c.ctx.get(), s.ts1.get(), &t1, sizeof(t1))) {
               p->failed = true;
               break;
            }
            if (!dj.Disjoint && dj.Frequency > 0) {
               double scale = 1e6 / (double)dj.Frequency;
               double b_us = (double)t0 * scale;
               double e_us = (double)t1 * scale;
               double eff = (have_last_end && last_end_us > b_us) ? last_end_us : b_us;
               if (measured) {
                  gb_samples_push(&p->gpu, e_us > eff ? (e_us - eff) : 0.0);
                  if (have_last_end)
                     gb_samples_push(&p->gap, b_us - last_end_us);
               }
               last_end_us = e_us;
               have_last_end = true;
            }
         }
         s.busy = false;
         done++;
         continue;
      }
      slot &s = slots[sent % (uint64_t)depth];
      uint64_t r0 = gb_now_ns();
      if (timed) {
         c.ctx->Begin(s.disj.get());
         c.ctx->End(s.ts0.get());
      }
      rec(c, user, sent + 1);
      if (timed) {
         c.ctx->End(s.ts1.get());
         c.ctx->End(s.disj.get());
      }
      uint64_t r1 = gb_now_ns();
      s.submit_ns = r1;
      c.ctx->End(s.ev.get());
      c.ctx->Flush();
      uint64_t r2 = gb_now_ns();
      s.busy = true;
      if (r0 >= start && r0 < end) {
         gb_samples_push(&p->record_cpu, (double)(r1 - r0) / 1e3);
         gb_samples_push(&p->submit_cpu, (double)(r2 - r1) / 1e3);
      }
      sent++;
   }

   while (done < sent) {
      slot &s = slots[done % (uint64_t)depth];
      if (s.busy && !p->failed) {
         if (!wait_event_query(c.ctx.get(), s.ev.get())) p->failed = true;
      }
      done++;
   }
   for (int i = 0; i < runs; i++)
      p->submits += p->win_count[i];
}

static void
emit_stream_result(const d3d11_ctx &c, const struct gb_opts *o, int depth,
                   const struct gb_pass *u, const struct gb_pass *t, const struct gb_check *chk,
                   const char *m1, const char *u1, double w1,
                   const char *m2, const char *u2, double w2, const char *primary)
{
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, depth, nullptr);
   gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (u->failed || (t && t->failed) || !u->submits || (t && t->timed && !t->submits)) {
      gb_result_finish(&r, "fail", nullptr, "D3D11 GPU wait/error or no measured completions");
      return;
   }
   gb_result_stream_metrics(&r, u, (t && t->timed) ? t : nullptr);
   if (m1 || m2) {
      struct gb_samples us = {};
      gb_pass_rate(u, &us);
      if (m1) gb_derive_rate(&r, m1, u1, &us, w1);
      if (m2) gb_derive_rate(&r, m2, u2, &us, w2);
      gb_samples_free(&us);
   }
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts");
   gb_json_int(&r.j, "submits", (int64_t)u->submits);
   gb_json_obj_end(&r.j);
   if (chk && chk->done)
      gb_result_check(&r, chk);
   gb_result_finish(&r, (chk && chk->done && !chk->valid) ? "fail" : "ok",
                    primary ? primary : "us_per_submit",
                    (chk && chk->done && !chk->valid) ? "output check failed" : nullptr);
}

/* ---- Resource creation helpers ---- */

static ComPtr<ID3D11Buffer>
make_cb(d3d11_ctx &c, UINT bytes, bool dynamic = true, const void *init = nullptr)
{
   D3D11_BUFFER_DESC bd = {};
   bd.ByteWidth = (bytes + 15u) & ~15u;
   bd.Usage = dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT;
   bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
   bd.CPUAccessFlags = dynamic ? D3D11_CPU_ACCESS_WRITE : 0;
   D3D11_SUBRESOURCE_DATA sd = {}; sd.pSysMem = init;
   ComPtr<ID3D11Buffer> b;
   hr_check(c.dev->CreateBuffer(&bd, init ? &sd : nullptr, b.put()), "CreateBuffer(CB)");
   return b;
}

static void
update_cb_discard(ID3D11DeviceContext *ctx, ID3D11Buffer *cb, const void *data, size_t sz)
{
   D3D11_MAPPED_SUBRESOURCE ms = {};
   hr_check(ctx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms), "Map(CB)");
   memcpy(ms.pData, data, sz);
   ctx->Unmap(cb, 0);
}

static ComPtr<ID3D11Buffer>
make_raw_buf(d3d11_ctx &c, UINT bytes, UINT bind_flags, const void *init = nullptr)
{
   D3D11_BUFFER_DESC bd = {};
   bd.ByteWidth = bytes;
   bd.Usage = D3D11_USAGE_DEFAULT;
   bd.BindFlags = bind_flags;
   bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
   D3D11_SUBRESOURCE_DATA sd = {}; sd.pSysMem = init;
   ComPtr<ID3D11Buffer> b;
   hr_check(c.dev->CreateBuffer(&bd, init ? &sd : nullptr, b.put()), "CreateBuffer(RAW)");
   return b;
}

static ComPtr<ID3D11UnorderedAccessView>
make_raw_uav(d3d11_ctx &c, ID3D11Buffer *buf, UINT bytes)
{
   D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
   ud.Format = DXGI_FORMAT_R32_TYPELESS;
   ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
   ud.Buffer.FirstElement = 0;
   ud.Buffer.NumElements = bytes / 4u;
   ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
   ComPtr<ID3D11UnorderedAccessView> u;
   hr_check(c.dev->CreateUnorderedAccessView(buf, &ud, u.put()), "CreateUAV(RAW)");
   return u;
}

static ComPtr<ID3D11ShaderResourceView>
make_raw_srv(d3d11_ctx &c, ID3D11Buffer *buf, UINT bytes)
{
   D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
   sd.Format = DXGI_FORMAT_R32_TYPELESS;
   sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
   sd.BufferEx.FirstElement = 0;
   sd.BufferEx.NumElements = bytes / 4u;
   sd.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
   ComPtr<ID3D11ShaderResourceView> s;
   hr_check(c.dev->CreateShaderResourceView(buf, &sd, s.put()), "CreateSRV(RAW)");
   return s;
}

static void
readback_buf(d3d11_ctx &c, ID3D11Buffer *src, void *dst, UINT bytes)
{
   D3D11_BUFFER_DESC bd = {};
   bd.ByteWidth = bytes;
   bd.Usage = D3D11_USAGE_STAGING;
   bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   ComPtr<ID3D11Buffer> st;
   hr_check(c.dev->CreateBuffer(&bd, nullptr, st.put()), "CreateBuffer(STAGING)");
   D3D11_BOX box = {0, 0, 0, bytes, 1, 1};
   c.ctx->CopySubresourceRegion(st.get(), 0, 0, 0, 0, src, 0, &box);
   D3D11_MAPPED_SUBRESOURCE ms = {};
   hr_check(c.ctx->Map(st.get(), 0, D3D11_MAP_READ, 0, &ms), "Map(STAGING)");
   memcpy(dst, ms.pData, bytes);
   c.ctx->Unmap(st.get(), 0);
}

static void
readback_tex2d(d3d11_ctx &c, ID3D11Texture2D *src, UINT mip, uint32_t w, uint32_t h, uint8_t *dst)
{
   D3D11_TEXTURE2D_DESC td = {};
   src->GetDesc(&td);
   td.Width = w;
   td.Height = h;
   td.MipLevels = 1;
   td.ArraySize = 1;
   td.SampleDesc.Count = 1;
   td.SampleDesc.Quality = 0;
   td.Usage = D3D11_USAGE_STAGING;
   td.BindFlags = 0;
   td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
   td.MiscFlags = 0;
   ComPtr<ID3D11Texture2D> st;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, st.put()), "CreateTexture2D(STAGING)");
   c.ctx->CopySubresourceRegion(st.get(), 0, 0, 0, 0, src, mip, nullptr);
   D3D11_MAPPED_SUBRESOURCE ms = {};
   hr_check(c.ctx->Map(st.get(), 0, D3D11_MAP_READ, 0, &ms), "Map(TEX_STAGING)");
   for (uint32_t y = 0; y < h; y++)
      memcpy(dst + (size_t)y * w * 4u, (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch, (size_t)w * 4u);
   c.ctx->Unmap(st.get(), 0);
}

static ComPtr<ID3D11RasterizerState>
make_rs_nocull(d3d11_ctx &c)
{
   D3D11_RASTERIZER_DESC rd = {};
   rd.FillMode = D3D11_FILL_SOLID;
   rd.CullMode = D3D11_CULL_NONE;
   rd.DepthClipEnable = TRUE;
   ComPtr<ID3D11RasterizerState> rs;
   hr_check(c.dev->CreateRasterizerState(&rd, rs.put()), "CreateRasterizerState");
   return rs;
}

static ComPtr<ID3D11SamplerState>
make_sampler(d3d11_ctx &c, D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE addr,
             D3D11_COMPARISON_FUNC cmp = D3D11_COMPARISON_NEVER)
{
   D3D11_SAMPLER_DESC sd = {};
   sd.Filter = filter;
   sd.AddressU = sd.AddressV = sd.AddressW = addr;
   sd.ComparisonFunc = cmp;
   sd.MaxLOD = D3D11_FLOAT32_MAX;
   ComPtr<ID3D11SamplerState> s;
   hr_check(c.dev->CreateSamplerState(&sd, s.put()), "CreateSamplerState");
   return s;
}

static void
unbind_all(ID3D11DeviceContext *ctx)
{
   ID3D11RenderTargetView *rtvs[4] = {};
   ctx->OMSetRenderTargets(4, rtvs, nullptr);
   ID3D11ShaderResourceView *srvs[4] = {};
   ctx->PSSetShaderResources(0, 4, srvs);
   ctx->CSSetShaderResources(0, 4, srvs);
   ID3D11UnorderedAccessView *uavs[2] = {};
   ctx->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
}

static int
run_depths_helper(d3d11_ctx &c, const struct gb_opts *o, const int *defd, int ndef,
                  d11_rec_fn rec, void *user, const struct gb_check *chk,
                  const char *m1, const char *u1, double w1,
                  const char *m2, const char *u2, double w2, const char *primary)
{
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : ndef;
   int rc = 0;
   for (int i = 0; i < nd; i++) {
      int depth = depths[i];
      if (o->mode != GB_MODE_QUEUE || depth < 1 || depth > 64) {
         struct gb_result r;
         gb_result_begin(&r, o, o->mode, depth, nullptr);
         gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
         gb_result_finish(&r, "skip", nullptr, o->mode != GB_MODE_QUEUE ? "D3D11 supports mode=queue only" : "depth must be 1..64");
         continue;
      }
      bool timed = o->timestamps && o->timed_runs > 0;
      struct gb_pass u = {}, t = {};
      run_stream(c, depth, false, o->warmup_ms, o->runs, o->run_ms, rec, user, &u);
      if (timed && !u.failed)
         run_stream(c, depth, true, o->warmup_ms < 100 ? o->warmup_ms : 100, o->timed_runs, o->run_ms, rec, user, &t);
      if (u.failed || t.failed || !u.submits || (timed && !t.submits)) rc = 1;
      emit_stream_result(c, o, depth, &u, timed ? &t : nullptr, chk, m1, u1, w1, m2, u2, w2, primary);
      gb_pass_free(&u);
      gb_pass_free(&t);
   }
   if (chk && chk->done && !chk->valid) rc = 1;
   return rc;
}

/* ======================= 1. compute.light ======================= */

struct d11_cl {
   ComPtr<ID3D11ComputeShader> cs;
   ComPtr<ID3D11Buffer> out, cb;
   ComPtr<ID3D11UnorderedAccessView> uav;
};

static void
d11_cl_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   d11_cl *s = (d11_cl *)u;
   uint32_t data[4] = {(uint32_t)k, 0, 0, 0};
   update_cb_discard(c.ctx.get(), s->cb.get(), data, sizeof(data));
   ID3D11Buffer *cbs[1] = {s->cb.get()};
   ID3D11UnorderedAccessView *uavs[1] = {s->uav.get()};
   c.ctx->CSSetShader(s->cs.get(), nullptr, 0);
   c.ctx->CSSetConstantBuffers(0, 1, cbs);
   c.ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
   c.ctx->Dispatch(1, 1, 1);
}

static int
run_compute_light(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   d11_cl s;
   hr_check(c.dev->CreateComputeShader(gb_dxbc_light_cs, gb_dxbc_light_cs_size, nullptr, s.cs.put()), "CreateCS(light)");
   s.out = make_raw_buf(c, 4096, D3D11_BIND_UNORDERED_ACCESS);
   s.uav = make_raw_uav(c, s.out.get(), 4096);
   s.cb = make_cb(c, 16);

   struct gb_check chk = {};
   if (do_verify) {
      d11_cl_rec(c, &s, 255);
      gpu_idle(c);
      uint32_t got[GB_LIGHT_LOCAL] = {};
      readback_buf(c, s.out.get(), got, sizeof(got));
      gb_d3d_check_compute_light(got, 255, &chk);
   }
   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   int rc = run_depths_helper(c, o, defd, 7, d11_cl_rec, &s, &chk, nullptr, nullptr, 0, nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 2. compute.heavy ======================= */

struct d11_ch {
   bool bw = false;
   ComPtr<ID3D11ComputeShader> cs;
   ComPtr<ID3D11Buffer> a, b, cb;
   ComPtr<ID3D11ShaderResourceView> srv;
   ComPtr<ID3D11UnorderedAccessView> uav;
};

static void
d11_ch_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_ch *s = (d11_ch *)u;
   c.ctx->CSSetShader(s->cs.get(), nullptr, 0);
   ID3D11UnorderedAccessView *uavs[1] = {s->uav.get()};
   c.ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
   if (s->bw) {
      ID3D11ShaderResourceView *srvs[1] = {s->srv.get()};
      c.ctx->CSSetShaderResources(0, 1, srvs);
      c.ctx->Dispatch((UINT)(GB_BW_BYTES / 16u / GB_BW_LOCAL), 1, 1);
   } else {
      ID3D11Buffer *cbs[1] = {s->cb.get()};
      c.ctx->CSSetConstantBuffers(0, 1, cbs);
      c.ctx->Dispatch(GB_ALU_GROUPS, 1, 1);
   }
}

static int
run_compute_heavy(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "alu");
   d11_ch s;
   if (!strcmp(kind, "bw")) s.bw = true;
   else if (strcmp(kind, "alu")) gb_die("kind must be alu or bw");

   if (s.bw) {
      hr_check(c.dev->CreateComputeShader(gb_dxbc_bw_cs, gb_dxbc_bw_cs_size, nullptr, s.cs.put()), "CreateCS(bw)");
      std::vector<float> init(GB_BW_BYTES / sizeof(float));
      for (size_t i = 0; i < init.size(); i++) init[i] = (float)(i & 255u);
      s.a = make_raw_buf(c, (UINT)GB_BW_BYTES, D3D11_BIND_SHADER_RESOURCE, init.data());
      s.b = make_raw_buf(c, (UINT)GB_BW_BYTES, D3D11_BIND_UNORDERED_ACCESS);
      s.srv = make_raw_srv(c, s.a.get(), (UINT)GB_BW_BYTES);
      s.uav = make_raw_uav(c, s.b.get(), (UINT)GB_BW_BYTES);
   } else {
      hr_check(c.dev->CreateComputeShader(gb_dxbc_alu_cs, gb_dxbc_alu_cs_size, nullptr, s.cs.put()), "CreateCS(alu)");
      UINT bytes = GB_ALU_GROUPS * GB_ALU_LOCAL * 4u;
      s.a = make_raw_buf(c, bytes, D3D11_BIND_UNORDERED_ACCESS);
      s.uav = make_raw_uav(c, s.a.get(), bytes);
      struct { uint32_t iters; float a, b; uint32_t pad; } cb_data = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B, 0};
      s.cb = make_cb(c, 16, false, &cb_data);
   }

   struct gb_check chk = {};
   if (do_verify) {
      d11_ch_rec(c, &s, 1);
      gpu_idle(c);
      if (s.bw) {
         std::vector<float> got(GB_BW_BYTES / sizeof(float));
         readback_buf(c, s.b.get(), got.data(), (UINT)GB_BW_BYTES);
         gb_d3d_check_compute_bw(got.data(), &chk);
      } else {
         std::vector<float> got(GB_ALU_GROUPS * GB_ALU_LOCAL);
         readback_buf(c, s.a.get(), got.data(), (UINT)(got.size() * sizeof(float)));
         gb_d3d_check_compute_alu(got.data(), &chk);
      }
   }
   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   int rc = run_depths_helper(c, o, defd, 7, d11_ch_rec, &s, &chk,
                              s.bw ? "gbps" : "gflops", s.bw ? "GB/s" : "GFLOP/s",
                              s.bw ? (2.0 * GB_BW_BYTES / 1e9 * 1e6) : (GB_ALU_FLOP / 1e9 * 1e6),
                              nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 3. compute.conv2d ======================= */

struct d11_conv {
   uint32_t w = 0, h = 0;
   ComPtr<ID3D11ComputeShader> csh, csv;
   ComPtr<ID3D11Buffer> in, mid, out, cb;
   ComPtr<ID3D11ShaderResourceView> srv_in, srv_mid;
   ComPtr<ID3D11UnorderedAccessView> uav_mid, uav_out;
};

static void
d11_conv_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_conv *s = (d11_conv *)u;
   UINT gx = (s->w + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;
   UINT gy = (s->h + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;
   ID3D11Buffer *cbs[1] = {s->cb.get()};
   ID3D11ShaderResourceView *null_srv[1] = {nullptr};
   ID3D11UnorderedAccessView *null_uav[1] = {nullptr};

   ID3D11ShaderResourceView *s0[1] = {s->srv_in.get()};
   ID3D11UnorderedAccessView *u0[1] = {s->uav_mid.get()};
   c.ctx->CSSetShader(s->csh.get(), nullptr, 0);
   c.ctx->CSSetConstantBuffers(0, 1, cbs);
   c.ctx->CSSetShaderResources(0, 1, s0);
   c.ctx->CSSetUnorderedAccessViews(0, 1, u0, nullptr);
   c.ctx->Dispatch(gx, gy, 1);

   c.ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
   ID3D11ShaderResourceView *s1[1] = {s->srv_mid.get()};
   ID3D11UnorderedAccessView *u1[1] = {s->uav_out.get()};
   c.ctx->CSSetShader(s->csv.get(), nullptr, 0);
   c.ctx->CSSetShaderResources(0, 1, s1);
   c.ctx->CSSetUnorderedAccessViews(0, 1, u1, nullptr);
   c.ctx->Dispatch(gx, gy, 1);
   c.ctx->CSSetShaderResources(0, 1, null_srv);
   c.ctx->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);
}

static int
run_compute_conv2d(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", "res", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "gaussian");
   bool bilateral = !strcmp(kind, "bilateral");
   if (!bilateral && strcmp(kind, "gaussian")) gb_die("kind must be gaussian or bilateral");
   uint32_t w = 0, h = 0;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &w, &h)) gb_die("bad res");

   d11_conv s; s.w = w; s.h = h;
   size_t count = (size_t)w * h;
   size_t bytes = count * sizeof(uint32_t);
   std::vector<uint32_t> in_cpu(count), ref(count);
   gb_conv2d_fill_input(in_cpu.data(), w, h);
   uint32_t ref_crc = do_verify ? gb_conv2d_ref(bilateral, in_cpu.data(), ref.data(), w, h) : 0;

   const unsigned char *hb = bilateral ? gb_dxbc_conv2d_h_bilat_cs : gb_dxbc_conv2d_h_gauss_cs;
   size_t hs = bilateral ? gb_dxbc_conv2d_h_bilat_cs_size : gb_dxbc_conv2d_h_gauss_cs_size;
   const unsigned char *vb = bilateral ? gb_dxbc_conv2d_v_bilat_cs : gb_dxbc_conv2d_v_gauss_cs;
   size_t vs = bilateral ? gb_dxbc_conv2d_v_bilat_cs_size : gb_dxbc_conv2d_v_gauss_cs_size;
   hr_check(c.dev->CreateComputeShader(hb, hs, nullptr, s.csh.put()), "CreateCS(conv_h)");
   hr_check(c.dev->CreateComputeShader(vb, vs, nullptr, s.csv.put()), "CreateCS(conv_v)");

   s.in = make_raw_buf(c, (UINT)bytes, D3D11_BIND_SHADER_RESOURCE, in_cpu.data());
   s.mid = make_raw_buf(c, (UINT)bytes, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS);
   s.out = make_raw_buf(c, (UINT)bytes, D3D11_BIND_UNORDERED_ACCESS);
   s.srv_in = make_raw_srv(c, s.in.get(), (UINT)bytes);
   s.srv_mid = make_raw_srv(c, s.mid.get(), (UINT)bytes);
   s.uav_mid = make_raw_uav(c, s.mid.get(), (UINT)bytes);
   s.uav_out = make_raw_uav(c, s.out.get(), (UINT)bytes);
   int32_t cb_data[4] = {(int32_t)w, (int32_t)h, 0, 0};
   s.cb = make_cb(c, 16, false, cb_data);

   struct gb_check chk = {};
   if (do_verify) {
      d11_conv_rec(c, &s, 1);
      gpu_idle(c);
      std::vector<uint32_t> got(count);
      readback_buf(c, s.out.get(), got.data(), (UINT)bytes);
      gb_check_words(&chk, got.data(), ref.data(), ref_crc, count);
   }
   static const int defd[] = {1, 4};
   int rc = run_depths_helper(c, o, defd, 2, d11_conv_rec, &s, &chk,
                              "mpix_s", "Mpix/s", (double)w * h, nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 4 & 5. render.light & render.heavy ======================= */

struct d11_render {
   enum { TRI, CLEAR, FILL, OPAQUE, TEXTURE, GEOM } kind = TRI;
   uint32_t w = 64, h = 64, msaa = 1;
   ComPtr<ID3D11VertexShader> vs;
   ComPtr<ID3D11PixelShader> ps;
   ComPtr<ID3D11Texture2D> rt, rt_ms, tex;
   ComPtr<ID3D11RenderTargetView> rtv;
   ComPtr<ID3D11ShaderResourceView> tex_srv;
   ComPtr<ID3D11SamplerState> smp;
   ComPtr<ID3D11RasterizerState> rs;
   ComPtr<ID3D11BlendState> bs;
   ComPtr<ID3D11Buffer> cb;
};

static void
d11_render_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   d11_render *s = (d11_render *)u;
   const float clr[4] = {0.1f, 0.2f, 0.3f, 1.0f};
   ID3D11RenderTargetView *rtvs[1] = {s->rtv.get()};
   c.ctx->OMSetRenderTargets(1, rtvs, nullptr);
   c.ctx->ClearRenderTargetView(s->rtv.get(), clr);
   if (s->kind != d11_render::CLEAR) {
      D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)s->w, (float)s->h, 0.0f, 1.0f};
      c.ctx->RSSetViewports(1, &vp);
      c.ctx->RSSetState(s->rs.get());
      c.ctx->OMSetBlendState(s->bs.get(), nullptr, 0xffffffffu);
      c.ctx->IASetInputLayout(nullptr);
      c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      c.ctx->VSSetShader(s->vs.get(), nullptr, 0);
      c.ctx->PSSetShader(s->ps.get(), nullptr, 0);
      float col[4] = {(float)(k & 255u) / 255.0f, 0.5f, 0.25f, 0.5f};
      update_cb_discard(c.ctx.get(), s->cb.get(), col, sizeof(col));
      ID3D11Buffer *cbs[1] = {s->cb.get()};
      c.ctx->PSSetConstantBuffers(0, 1, cbs);
      if (s->kind == d11_render::TEXTURE) {
         ID3D11ShaderResourceView *srvs[1] = {s->tex_srv.get()};
         ID3D11SamplerState *smps[1] = {s->smp.get()};
         c.ctx->PSSetShaderResources(0, 1, srvs);
         c.ctx->PSSetSamplers(0, 1, smps);
      }
      if (s->kind == d11_render::TRI)
         c.ctx->Draw(3, 0);
      else if (s->kind == d11_render::GEOM)
         c.ctx->Draw(3u * GB_GEOM_TRIS, 0);
      else
         c.ctx->DrawInstanced(3, GB_FILL_LAYERS, 0, 0);
   }
   if (s->msaa == 4u)
      c.ctx->ResolveSubresource(s->rt.get(), 0, s->rt_ms.get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
}

static int
run_render_basic(d3d11_ctx &c, const struct gb_opts *o, bool heavy, bool do_verify)
{
   static const char *const k_light[] = {"kind", nullptr};
   static const char *const k_heavy[] = {"kind", "res", "msaa", nullptr};
   if (!gb_check_params(o, heavy ? k_heavy : k_light)) gb_die("unknown parameter");

   const char *kind = gb_param(o, "kind", heavy ? "fill" : "tri");
   d11_render s;
   if (!heavy) {
      s.w = GB_LIGHT_DIM; s.h = GB_LIGHT_DIM; s.msaa = 1;
      if (!strcmp(kind, "tri")) s.kind = d11_render::TRI;
      else if (!strcmp(kind, "clear")) s.kind = d11_render::CLEAR;
      else gb_die("render.light kind must be tri or clear");
   } else {
      if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &s.w, &s.h)) gb_die("bad res");
      long ms = gb_param_int(o, "msaa", 1);
      if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
      s.msaa = (uint32_t)ms;
      if (!strcmp(kind, "fill")) s.kind = d11_render::FILL;
      else if (!strcmp(kind, "opaque")) s.kind = d11_render::OPAQUE;
      else if (!strcmp(kind, "texture")) s.kind = d11_render::TEXTURE;
      else if (!strcmp(kind, "geom")) s.kind = d11_render::GEOM;
      else gb_die("render.heavy kind must be fill, opaque, texture or geom");
   }

   D3D11_TEXTURE2D_DESC td = {};
   td.Width = s.w; td.Height = s.h; td.MipLevels = 1; td.ArraySize = 1;
   td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
   td.SampleDesc.Count = 1;
   td.Usage = D3D11_USAGE_DEFAULT;
   td.BindFlags = D3D11_BIND_RENDER_TARGET;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, s.rt.put()), "CreateTexture2D(RT)");
   if (s.msaa == 4u) {
      td.SampleDesc.Count = 4;
      hr_check(c.dev->CreateTexture2D(&td, nullptr, s.rt_ms.put()), "CreateTexture2D(RT_MS)");
      hr_check(c.dev->CreateRenderTargetView(s.rt_ms.get(), nullptr, s.rtv.put()), "CreateRTV(MS)");
   } else {
      hr_check(c.dev->CreateRenderTargetView(s.rt.get(), nullptr, s.rtv.put()), "CreateRTV");
   }
   s.rs = make_rs_nocull(c);
   s.cb = make_cb(c, 16);

   if (s.kind != d11_render::CLEAR) {
      const unsigned char *vb = (s.kind == d11_render::GEOM) ? gb_dxbc_geom_vs : gb_dxbc_fs_vs;
      size_t vs = (s.kind == d11_render::GEOM) ? gb_dxbc_geom_vs_size : gb_dxbc_fs_vs_size;
      const unsigned char *pb = (s.kind == d11_render::TEXTURE) ? gb_dxbc_texture_ps : gb_dxbc_color_ps;
      size_t ps = (s.kind == d11_render::TEXTURE) ? gb_dxbc_texture_ps_size : gb_dxbc_color_ps_size;
      hr_check(c.dev->CreateVertexShader(vb, vs, nullptr, s.vs.put()), "CreateVS");
      hr_check(c.dev->CreatePixelShader(pb, ps, nullptr, s.ps.put()), "CreatePS");

      D3D11_BLEND_DESC bd = {};
      bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
      if (s.kind == d11_render::FILL || s.kind == d11_render::TEXTURE) {
         bd.RenderTarget[0].BlendEnable = TRUE;
         bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
         bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
         bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
         bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
         bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
         bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
      }
      hr_check(c.dev->CreateBlendState(&bd, s.bs.put()), "CreateBlendState");

      if (s.kind == d11_render::TEXTURE) {
         std::vector<uint8_t> data(GB_TEX_DIM * GB_TEX_DIM * 4u);
         for (uint32_t y = 0; y < GB_TEX_DIM; y++)
            for (uint32_t x = 0; x < GB_TEX_DIM; x++) {
               size_t idx = ((size_t)y * GB_TEX_DIM + x) * 4u;
               data[idx] = (uint8_t)x; data[idx + 1] = (uint8_t)y;
               data[idx + 2] = (uint8_t)(x ^ y); data[idx + 3] = 255;
            }
         D3D11_TEXTURE2D_DESC ttd = {};
         ttd.Width = GB_TEX_DIM; ttd.Height = GB_TEX_DIM; ttd.MipLevels = 1; ttd.ArraySize = 1;
         ttd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; ttd.SampleDesc.Count = 1;
         ttd.Usage = D3D11_USAGE_IMMUTABLE; ttd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
         D3D11_SUBRESOURCE_DATA sd = {data.data(), GB_TEX_DIM * 4u, 0};
         hr_check(c.dev->CreateTexture2D(&ttd, &sd, s.tex.put()), "CreateTexture2D(TEX)");
         hr_check(c.dev->CreateShaderResourceView(s.tex.get(), nullptr, s.tex_srv.put()), "CreateSRV(TEX)");
         s.smp = make_sampler(c, D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_WRAP);
      }
   }

   struct gb_check chk = {};
   if (do_verify) {
      d11_render_rec(c, &s, 255);
      gpu_idle(c);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.rt.get(), 0, s.w, s.h, got.data());
      gb_d3d_check_render_basic(got.data(), s.w, s.h, kind, 255, &chk);
   }

   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   const char *m1 = nullptr, *u1 = nullptr;
   double w1 = 0.0;
   if (heavy) {
      if (s.kind == d11_render::GEOM) {
         m1 = "mtri_s"; u1 = "Mtri/s"; w1 = (double)GB_GEOM_TRIS;
      } else {
         m1 = "gpix_s"; u1 = "Gpix/s"; w1 = (double)s.w * s.h * GB_FILL_LAYERS / 1e9 * 1e6;
      }
   }
   int rc = run_depths_helper(c, o, defd, 7, d11_render_rec, &s, &chk, m1, u1, w1, nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 6. render.draws (direct / instanced / indirect) ======================= */

struct d11_draws {
   enum { DIRECT, INSTANCED, INDIRECT } mode = DIRECT;
   uint32_t draws = 2048;
   ComPtr<ID3D11VertexShader> vs;
   ComPtr<ID3D11PixelShader> ps;
   ComPtr<ID3D11InputLayout> il;
   ComPtr<ID3D11Texture2D> rt;
   ComPtr<ID3D11RenderTargetView> rtv;
   ComPtr<ID3D11Buffer> vb, inst_vb, ib, cb, indirect_buf;
   ComPtr<ID3D11RasterizerState> rs;
   std::vector<float> cb_ring; /* 4 floats * draws */
};

static void
d11_draws_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_draws *s = (d11_draws *)u;
   const float clr[4] = {0.0f, 0.0f, 0.0f, 1.0f};
   ID3D11RenderTargetView *rtvs[1] = {s->rtv.get()};
   c.ctx->OMSetRenderTargets(1, rtvs, nullptr);
   c.ctx->ClearRenderTargetView(s->rtv.get(), clr);
   D3D11_VIEWPORT vp = {0.0f, 0.0f, 1024.0f, 1024.0f, 0.0f, 1.0f};
   c.ctx->RSSetViewports(1, &vp);
   c.ctx->RSSetState(s->rs.get());
   c.ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
   c.ctx->IASetInputLayout(s->il.get());
   if (s->mode == d11_draws::DIRECT) {
      UINT stride = 8, offset = 0;
      ID3D11Buffer *vbs[1] = {s->vb.get()};
      c.ctx->IASetVertexBuffers(0, 1, vbs, &stride, &offset);
   } else {
      UINT strides[2] = {8, 4}, offsets[2] = {0, 0};
      ID3D11Buffer *vbs[2] = {s->vb.get(), s->inst_vb.get()};
      c.ctx->IASetVertexBuffers(0, 2, vbs, strides, offsets);
   }
   c.ctx->IASetIndexBuffer(s->ib.get(), DXGI_FORMAT_R16_UINT, 0);
   c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   c.ctx->VSSetShader(s->vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->ps.get(), nullptr, 0);

   if (s->mode == d11_draws::INSTANCED) {
      c.ctx->DrawIndexedInstanced(6, s->draws, 0, 0, 0);
   } else if (s->mode == d11_draws::INDIRECT) {
      for (uint32_t d = 0; d < s->draws; d++)
         c.ctx->DrawIndexedInstancedIndirect(s->indirect_buf.get(), d * 20u);
   } else {
      ID3D11Buffer *cbs[1] = {s->cb.get()};
      c.ctx->VSSetConstantBuffers(0, 1, cbs);
      for (uint32_t d = 0; d < s->draws; d++) {
         update_cb_discard(c.ctx.get(), s->cb.get(), &s->cb_ring[d * 4u], 16);
         c.ctx->DrawIndexed(6, 0, 0);
      }
   }
}

static int
run_render_draws(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"draws", "mode", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   long draws = gb_param_int(o, "draws", 2048);
   if (draws < 1 || draws > 10000) gb_die("draws must be 1..10000");
   const char *mstr = gb_param(o, "mode", "direct");

   d11_draws s;
   s.draws = (uint32_t)draws;
   if (!strcmp(mstr, "direct")) s.mode = d11_draws::DIRECT;
   else if (!strcmp(mstr, "instanced")) s.mode = d11_draws::INSTANCED;
   else if (!strcmp(mstr, "indirect") || !strcmp(mstr, "mdi")) s.mode = d11_draws::INDIRECT;
   else gb_die("mode must be direct, instanced or indirect");

   const unsigned char *vb_code = (s.mode == d11_draws::DIRECT) ? gb_dxbc_draws_direct_vs : gb_dxbc_draws_inst_vs;
   size_t vb_sz = (s.mode == d11_draws::DIRECT) ? gb_dxbc_draws_direct_vs_size : gb_dxbc_draws_inst_vs_size;
   hr_check(c.dev->CreateVertexShader(vb_code, vb_sz, nullptr, s.vs.put()), "CreateVS(draws)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_draws_ps, gb_dxbc_draws_ps_size, nullptr, s.ps.put()), "CreatePS(draws)");

   D3D11_INPUT_ELEMENT_DESC ied[2] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32_UINT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1},
   };
   hr_check(c.dev->CreateInputLayout(ied, s.mode == d11_draws::DIRECT ? 1u : 2u, vb_code, vb_sz, s.il.put()), "CreateInputLayout");

   D3D11_TEXTURE2D_DESC td = {};
   td.Width = 1024; td.Height = 1024; td.MipLevels = 1; td.ArraySize = 1;
   td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
   td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, s.rt.put()), "CreateTexture2D(draws)");
   hr_check(c.dev->CreateRenderTargetView(s.rt.get(), nullptr, s.rtv.put()), "CreateRTV(draws)");

   const float verts[8] = {0, 0, 1, 0, 1, 1, 0, 1};
   const uint16_t idxs[6] = {0, 1, 2, 0, 2, 3};
   D3D11_BUFFER_DESC vbd = {sizeof(verts), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER, 0, 0, 0};
   D3D11_SUBRESOURCE_DATA vsd = {verts, 0, 0};
   hr_check(c.dev->CreateBuffer(&vbd, &vsd, s.vb.put()), "CreateBuffer(VB)");
   if (s.mode != d11_draws::DIRECT) {
      std::vector<uint32_t> inst_ids(s.draws);
      for (uint32_t d = 0; d < s.draws; d++) inst_ids[d] = d;
      D3D11_BUFFER_DESC ivbd = {(UINT)(inst_ids.size() * 4u), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER, 0, 0, 0};
      D3D11_SUBRESOURCE_DATA ivsd = {inst_ids.data(), 0, 0};
      hr_check(c.dev->CreateBuffer(&ivbd, &ivsd, s.inst_vb.put()), "CreateBuffer(INST_VB)");
   }
   D3D11_BUFFER_DESC ibd = {sizeof(idxs), D3D11_USAGE_IMMUTABLE, D3D11_BIND_INDEX_BUFFER, 0, 0, 0};
   D3D11_SUBRESOURCE_DATA isd = {idxs, 0, 0};
   hr_check(c.dev->CreateBuffer(&ibd, &isd, s.ib.put()), "CreateBuffer(IB)");
   s.rs = make_rs_nocull(c);

   if (s.mode == d11_draws::DIRECT) {
      s.cb = make_cb(c, 16, true);
      s.cb_ring.resize((size_t)s.draws * 4u);
      for (uint32_t d = 0; d < s.draws; d++) {
         uint32_t cell = d % 10000u;
         s.cb_ring[d * 4u + 0u] = (float)((cell % 100u) * 10u);
         s.cb_ring[d * 4u + 1u] = (float)((cell / 100u) * 10u);
         s.cb_ring[d * 4u + 2u] = (float)((d * 37u) & 255u) / 255.0f;
         s.cb_ring[d * 4u + 3u] = (float)((d * 73u) & 255u) / 255.0f;
      }
   } else if (s.mode == d11_draws::INDIRECT) {
      struct arg { uint32_t idx_cnt, inst_cnt, start_idx; int32_t base_vtx; uint32_t start_inst; };
      std::vector<arg> args(s.draws);
      for (uint32_t d = 0; d < s.draws; d++) args[d] = {6u, 1u, 0u, 0, d};
      D3D11_BUFFER_DESC id = {};
      id.ByteWidth = (UINT)(args.size() * sizeof(arg));
      id.Usage = D3D11_USAGE_IMMUTABLE;
      id.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
      D3D11_SUBRESOURCE_DATA sd = {args.data(), 0, 0};
      hr_check(c.dev->CreateBuffer(&id, &sd, s.indirect_buf.put()), "CreateBuffer(INDIRECT)");
   }

   struct gb_check chk = {};
   if (do_verify) {
      d11_draws_rec(c, &s, 1);
      gpu_idle(c);
      std::vector<uint8_t> got(1024u * 1024u * 4u);
      readback_tex2d(c, s.rt.get(), 0, 1024, 1024, got.data());
      gb_d3d_check_draws_fb(got.data(), s.draws, &chk);
   }

   static const int defd[] = {1, 4};
   int rc = run_depths_helper(c, o, defd, 2, d11_draws_rec, &s, &chk,
                              "mdraw_s", "Mdraw/s", (double)s.draws, nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 7. render.deferred (4 MRT G-buffer + lighting, 1x/4x MSAA) ======================= */

struct d11_deferred {
   uint32_t w = 1920, h = 1080, msaa = 1;
   ComPtr<ID3D11VertexShader> g_vs, l_vs;
   ComPtr<ID3D11PixelShader> g_ps, l_ps;
   ComPtr<ID3D11Texture2D> g[4], depth, final_rt;
   ComPtr<ID3D11RenderTargetView> g_rtv[4], final_rtv;
   ComPtr<ID3D11ShaderResourceView> g_srv[4];
   ComPtr<ID3D11DepthStencilView> dsv;
   ComPtr<ID3D11DepthStencilState> dss_less, dss_off;
   ComPtr<ID3D11RasterizerState> rs;
   ComPtr<ID3D11Buffer> cb;
};

static void
d11_deferred_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_deferred *s = (d11_deferred *)u;
   D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)s->w, (float)s->h, 0.0f, 1.0f};
   c.ctx->RSSetViewports(1, &vp);
   c.ctx->RSSetState(s->rs.get());
   c.ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
   c.ctx->IASetInputLayout(nullptr);
   c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

   /* Unbind SRVs before binding G-buffer as RTVs */
   ID3D11ShaderResourceView *null_srvs[4] = {};
   c.ctx->PSSetShaderResources(0, 4, null_srvs);

   /* Pass 1: Geometry -> 4 MRTs + Depth */
   ID3D11RenderTargetView *rtvs[4] = {s->g_rtv[0].get(), s->g_rtv[1].get(), s->g_rtv[2].get(), s->g_rtv[3].get()};
   c.ctx->OMSetRenderTargets(4, rtvs, s->dsv.get());
   const float z4[4] = {0, 0, 0, 0};
   for (int i = 0; i < 4; i++) c.ctx->ClearRenderTargetView(rtvs[i], z4);
   c.ctx->ClearDepthStencilView(s->dsv.get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
   c.ctx->OMSetDepthStencilState(s->dss_less.get(), 0);
   c.ctx->VSSetShader(s->g_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->g_ps.get(), nullptr, 0);
   ID3D11Buffer *cbs[1] = {s->cb.get()};
   c.ctx->VSSetConstantBuffers(0, 1, cbs);
   c.ctx->Draw(GB_DEFERRED_QUADS * 6u, 0);

   /* Pass 2: Lighting reads 4 MRTs -> Final RGBA8 */
   ID3D11RenderTargetView *frtv[4] = {s->final_rtv.get(), nullptr, nullptr, nullptr};
   c.ctx->OMSetRenderTargets(4, frtv, nullptr);
   c.ctx->OMSetDepthStencilState(s->dss_off.get(), 0);
   c.ctx->VSSetShader(s->l_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->l_ps.get(), nullptr, 0);
   ID3D11ShaderResourceView *srvs[4] = {s->g_srv[0].get(), s->g_srv[1].get(), s->g_srv[2].get(), s->g_srv[3].get()};
   c.ctx->PSSetShaderResources(0, 4, srvs);
   c.ctx->Draw(3, 0);
}

static int
run_render_deferred(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"mode", "res", "msaa", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *mode = gb_param(o, "mode", "dram");
   if (strcmp(mode, "dram") && strcmp(mode, "tile")) gb_die("mode must be dram or tile");
   d11_deferred s;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &s.w, &s.h)) gb_die("bad res");
   long ms = gb_param_int(o, "msaa", 1);
   if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
   s.msaa = (uint32_t)ms;

   hr_check(c.dev->CreateVertexShader(gb_dxbc_deferred_geom_vs, gb_dxbc_deferred_geom_vs_size, nullptr, s.g_vs.put()), "CreateVS(def_g)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_deferred_geom_ps, gb_dxbc_deferred_geom_ps_size, nullptr, s.g_ps.put()), "CreatePS(def_g)");
   hr_check(c.dev->CreateVertexShader(gb_dxbc_fs_vs, gb_dxbc_fs_vs_size, nullptr, s.l_vs.put()), "CreateVS(def_l)");
   const unsigned char *lp = (s.msaa == 4u) ? gb_dxbc_deferred_light_msaa_ps : gb_dxbc_deferred_light_ps;
   size_t ls = (s.msaa == 4u) ? gb_dxbc_deferred_light_msaa_ps_size : gb_dxbc_deferred_light_ps_size;
   hr_check(c.dev->CreatePixelShader(lp, ls, nullptr, s.l_ps.put()), "CreatePS(def_l)");

   static const DXGI_FORMAT gfmts[4] = {
      DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
      DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT
   };
   for (int i = 0; i < 4; i++) {
      D3D11_TEXTURE2D_DESC td = {};
      td.Width = s.w; td.Height = s.h; td.MipLevels = 1; td.ArraySize = 1;
      td.Format = gfmts[i]; td.SampleDesc.Count = s.msaa;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      hr_check(c.dev->CreateTexture2D(&td, nullptr, s.g[i].put()), "CreateTexture2D(G)");
      hr_check(c.dev->CreateRenderTargetView(s.g[i].get(), nullptr, s.g_rtv[i].put()), "CreateRTV(G)");
      hr_check(c.dev->CreateShaderResourceView(s.g[i].get(), nullptr, s.g_srv[i].put()), "CreateSRV(G)");
   }
   {
      D3D11_TEXTURE2D_DESC dd = {};
      dd.Width = s.w; dd.Height = s.h; dd.MipLevels = 1; dd.ArraySize = 1;
      dd.Format = DXGI_FORMAT_D32_FLOAT; dd.SampleDesc.Count = s.msaa;
      dd.Usage = D3D11_USAGE_DEFAULT; dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
      hr_check(c.dev->CreateTexture2D(&dd, nullptr, s.depth.put()), "CreateTexture2D(Depth)");
      hr_check(c.dev->CreateDepthStencilView(s.depth.get(), nullptr, s.dsv.put()), "CreateDSV");
   }
   {
      D3D11_TEXTURE2D_DESC fd = {};
      fd.Width = s.w; fd.Height = s.h; fd.MipLevels = 1; fd.ArraySize = 1;
      fd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; fd.SampleDesc.Count = 1;
      fd.Usage = D3D11_USAGE_DEFAULT; fd.BindFlags = D3D11_BIND_RENDER_TARGET;
      hr_check(c.dev->CreateTexture2D(&fd, nullptr, s.final_rt.put()), "CreateTexture2D(Final)");
      hr_check(c.dev->CreateRenderTargetView(s.final_rt.get(), nullptr, s.final_rtv.put()), "CreateRTV(Final)");
   }
   D3D11_DEPTH_STENCIL_DESC dsd = {};
   dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dsd.DepthFunc = D3D11_COMPARISON_LESS;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_less.put()), "CreateDSS(less)");
   dsd.DepthEnable = FALSE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_off.put()), "CreateDSS(off)");
   s.rs = make_rs_nocull(c);
   uint32_t zero_cb[4] = {0, 0, 0, 0};
   s.cb = make_cb(c, 16, false, zero_cb);

   struct gb_check chk = {};
   if (do_verify) {
      d11_deferred_rec(c, &s, 1);
      gpu_idle(c);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.final_rt.get(), 0, s.w, s.h, got.data());
      gb_deferred_check_fb(got.data(), s.w, s.h, &chk);
   }
   static const int defd[] = {1, 2};
   int rc = run_depths_helper(c, o, defd, 2, d11_deferred_rec, &s, &chk,
                              "gpix_s", "Gpix/s", (double)s.w * s.h / 1e9 * 1e6,
                              "fps", "fps", 1e6, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 8. render.shadow ======================= */

struct d11_shadow {
   bool alpha = false;
   uint32_t res = 2048, inst = GB_SHADOW_DEFAULT_INST;
   ComPtr<ID3D11VertexShader> d_vs, p_vs;
   ComPtr<ID3D11PixelShader> a_ps, p_ps;
   ComPtr<ID3D11Texture2D> depth_arr, recv_rt;
   ComPtr<ID3D11DepthStencilView> dsv[GB_SHADOW_CASCADES];
   ComPtr<ID3D11ShaderResourceView> depth_srv;
   ComPtr<ID3D11RenderTargetView> recv_rtv;
   ComPtr<ID3D11SamplerState> cmp_smp;
   ComPtr<ID3D11DepthStencilState> dss_less, dss_off;
   ComPtr<ID3D11RasterizerState> rs;
   ComPtr<ID3D11Buffer> cb_casc[GB_SHADOW_CASCADES], cb_pcf;
};

static void
d11_shadow_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_shadow *s = (d11_shadow *)u;
   ID3D11ShaderResourceView *null_srv[1] = {nullptr};
   c.ctx->PSSetShaderResources(0, 1, null_srv);
   c.ctx->RSSetState(s->rs.get());
   c.ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
   c.ctx->IASetInputLayout(nullptr);
   c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

   D3D11_VIEWPORT svp = {0.0f, 0.0f, (float)s->res, (float)s->res, 0.0f, 1.0f};
   c.ctx->RSSetViewports(1, &svp);
   c.ctx->OMSetDepthStencilState(s->dss_less.get(), 0);
   c.ctx->VSSetShader(s->d_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->alpha ? s->a_ps.get() : nullptr, nullptr, 0);
   for (uint32_t ci = 0; ci < GB_SHADOW_CASCADES; ci++) {
      c.ctx->OMSetRenderTargets(0, nullptr, s->dsv[ci].get());
      c.ctx->ClearDepthStencilView(s->dsv[ci].get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
      ID3D11Buffer *cbs[1] = {s->cb_casc[ci].get()};
      c.ctx->VSSetConstantBuffers(0, 1, cbs);
      c.ctx->DrawInstanced(6, s->inst, 0, 0);
   }

   ID3D11RenderTargetView *rtvs[1] = {s->recv_rtv.get()};
   c.ctx->OMSetRenderTargets(1, rtvs, nullptr);
   c.ctx->OMSetDepthStencilState(s->dss_off.get(), 0);
   const float blk[4] = {0, 0, 0, 1};
   c.ctx->ClearRenderTargetView(s->recv_rtv.get(), blk);
   D3D11_VIEWPORT rvp = {0.0f, 0.0f, (float)GB_SHADOW_RECV_DIM, (float)GB_SHADOW_RECV_DIM, 0.0f, 1.0f};
   c.ctx->RSSetViewports(1, &rvp);
   c.ctx->VSSetShader(s->p_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->p_ps.get(), nullptr, 0);
   ID3D11ShaderResourceView *srvs[1] = {s->depth_srv.get()};
   ID3D11SamplerState *smps[1] = {s->cmp_smp.get()};
   ID3D11Buffer *pcbs[1] = {s->cb_pcf.get()};
   c.ctx->PSSetShaderResources(0, 1, srvs);
   c.ctx->PSSetSamplers(0, 1, smps);
   c.ctx->PSSetConstantBuffers(0, 1, pcbs);
   c.ctx->Draw(3, 0);
}

static int
run_render_shadow(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"mode", "fmt", "res", "inst", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *mode = gb_param(o, "mode", "depth_only");
   bool alpha = !strcmp(mode, "alpha_test");
   if (!alpha && strcmp(mode, "depth_only")) gb_die("mode must be depth_only or alpha_test");
   const char *fmt_s = gb_param(o, "fmt", "d32f");
   bool d16 = !strcmp(fmt_s, "d16");
   if (!d16 && strcmp(fmt_s, "d32f")) gb_die("fmt must be d32f or d16");
   long res = gb_param_int(o, "res", 2048);
   long inst = gb_param_int(o, "inst", GB_SHADOW_DEFAULT_INST);
   if (res < 256 || res > 4096 || inst < 1024 || inst > 16384) gb_die("res/inst out of range");

   d11_shadow s; s.alpha = alpha; s.res = (uint32_t)res; s.inst = (uint32_t)inst;
   hr_check(c.dev->CreateVertexShader(gb_dxbc_shadow_depth_vs, gb_dxbc_shadow_depth_vs_size, nullptr, s.d_vs.put()), "CreateVS(shad)");
   if (alpha)
      hr_check(c.dev->CreatePixelShader(gb_dxbc_shadow_alpha_ps, gb_dxbc_shadow_alpha_ps_size, nullptr, s.a_ps.put()), "CreatePS(shad_a)");
   hr_check(c.dev->CreateVertexShader(gb_dxbc_fs_vs, gb_dxbc_fs_vs_size, nullptr, s.p_vs.put()), "CreateVS(pcf)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_shadow_pcf_ps, gb_dxbc_shadow_pcf_ps_size, nullptr, s.p_ps.put()), "CreatePS(pcf)");

   D3D11_TEXTURE2D_DESC td = {};
   td.Width = s.res; td.Height = s.res; td.MipLevels = 1; td.ArraySize = GB_SHADOW_CASCADES;
   td.Format = d16 ? DXGI_FORMAT_R16_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
   td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
   td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, s.depth_arr.put()), "CreateTexture2D(shad_arr)");

   for (uint32_t ci = 0; ci < GB_SHADOW_CASCADES; ci++) {
      D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {};
      dsvd.Format = d16 ? DXGI_FORMAT_D16_UNORM : DXGI_FORMAT_D32_FLOAT;
      dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
      dsvd.Texture2DArray.FirstArraySlice = ci;
      dsvd.Texture2DArray.ArraySize = 1;
      hr_check(c.dev->CreateDepthStencilView(s.depth_arr.get(), &dsvd, s.dsv[ci].put()), "CreateDSV(slice)");
      uint32_t cbd[4] = {ci, s.res, 0, 0};
      s.cb_casc[ci] = make_cb(c, 16, false, cbd);
   }
   D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
   srvd.Format = d16 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT;
   srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
   srvd.Texture2DArray.MipLevels = 1;
   srvd.Texture2DArray.ArraySize = GB_SHADOW_CASCADES;
   hr_check(c.dev->CreateShaderResourceView(s.depth_arr.get(), &srvd, s.depth_srv.put()), "CreateSRV(shad)");

   D3D11_TEXTURE2D_DESC rtd = {};
   rtd.Width = GB_SHADOW_RECV_DIM; rtd.Height = GB_SHADOW_RECV_DIM; rtd.MipLevels = 1; rtd.ArraySize = 1;
   rtd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rtd.SampleDesc.Count = 1;
   rtd.Usage = D3D11_USAGE_DEFAULT; rtd.BindFlags = D3D11_BIND_RENDER_TARGET;
   hr_check(c.dev->CreateTexture2D(&rtd, nullptr, s.recv_rt.put()), "CreateTexture2D(recv)");
   hr_check(c.dev->CreateRenderTargetView(s.recv_rt.get(), nullptr, s.recv_rtv.put()), "CreateRTV(recv)");

   s.cmp_smp = make_sampler(c, D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
                            D3D11_TEXTURE_ADDRESS_CLAMP, D3D11_COMPARISON_LESS);
   D3D11_DEPTH_STENCIL_DESC dsd = {};
   dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dsd.DepthFunc = D3D11_COMPARISON_LESS;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_less.put()), "CreateDSS(less)");
   dsd.DepthEnable = FALSE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_off.put()), "CreateDSS(off)");
   s.rs = make_rs_nocull(c);
   float pcf_cb[4] = {1.0f / (float)s.res, 0, 0, 0};
   s.cb_pcf = make_cb(c, 16, false, pcf_cb);

   struct gb_check chk = {};
   if (do_verify) {
      d11_shadow_rec(c, &s, 1);
      gpu_idle(c);
      std::vector<uint8_t> got((size_t)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM * 4u);
      readback_tex2d(c, s.recv_rt.get(), 0, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, got.data());
      gb_shadow_check_fb(got.data(), GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, s.alpha, &chk);
   }
   static const int defd[] = {1, 2};
   double tris = (double)GB_SHADOW_CASCADES * (double)s.inst * 2.0;
   double pix = (double)GB_SHADOW_CASCADES * (double)s.res * s.res +
                (double)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM;
   int rc = run_depths_helper(c, o, defd, 2, d11_shadow_rec, &s, &chk,
                              "mtri_s", "Mtri/s", tris, "gpix_s", "Gpix/s", pix / 1e3, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 9. render.postfx ======================= */

struct d11_postfx {
   uint32_t w = 2560, h = 1440, msaa = 4;
   uint32_t bw[GB_POSTFX_BLOOM_MIPS] = {}, bh[GB_POSTFX_BLOOM_MIPS] = {};
   ComPtr<ID3D11VertexShader> sc_vs, fs_vs;
   ComPtr<ID3D11PixelShader> sc_ps, dn_ps, up_ps, tn_ps;
   ComPtr<ID3D11Texture2D> hdr_ms, hdr_res, depth_ms, bloom[GB_POSTFX_BLOOM_MIPS], final_rt;
   ComPtr<ID3D11RenderTargetView> hdr_rtv, bloom_rtv[GB_POSTFX_BLOOM_MIPS], final_rtv;
   ComPtr<ID3D11ShaderResourceView> hdr_srv, bloom_srv[GB_POSTFX_BLOOM_MIPS];
   ComPtr<ID3D11DepthStencilView> dsv;
   ComPtr<ID3D11SamplerState> lin_smp;
   ComPtr<ID3D11BlendState> bs_add;
   ComPtr<ID3D11DepthStencilState> dss_less, dss_off;
   ComPtr<ID3D11RasterizerState> rs;
   ComPtr<ID3D11Buffer> cb_dn[GB_POSTFX_BLOOM_MIPS], cb_up[GB_POSTFX_BLOOM_MIPS - 1], cb_tn;
};

static void
d11_postfx_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_postfx *s = (d11_postfx *)u;
   ID3D11ShaderResourceView *null_srvs[2] = {};
   ID3D11SamplerState *smps[1] = {s->lin_smp.get()};
   c.ctx->RSSetState(s->rs.get());
   c.ctx->IASetInputLayout(nullptr);
   c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

   /* 1. HDR scene pass (+ ResolveSubresource when msaa=4) */
   c.ctx->PSSetShaderResources(0, 2, null_srvs);
   ID3D11RenderTargetView *rtv0[1] = {s->hdr_rtv.get()};
   c.ctx->OMSetRenderTargets(1, rtv0, s->dsv.get());
   const float bg[4] = {0.04f, 0.05f, 0.08f, 1.0f};
   c.ctx->ClearRenderTargetView(s->hdr_rtv.get(), bg);
   c.ctx->ClearDepthStencilView(s->dsv.get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
   c.ctx->OMSetDepthStencilState(s->dss_less.get(), 0);
   c.ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
   D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)s->w, (float)s->h, 0.0f, 1.0f};
   c.ctx->RSSetViewports(1, &vp);
   c.ctx->VSSetShader(s->sc_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->sc_ps.get(), nullptr, 0);
   c.ctx->Draw(GB_POSTFX_QUADS * 6u, 0);
   if (s->msaa == 4u)
      c.ctx->ResolveSubresource(s->hdr_res.get(), 0, s->hdr_ms.get(), 0, DXGI_FORMAT_R16G16B16A16_FLOAT);

   /* 2. 5-level bloom downsample */
   c.ctx->OMSetDepthStencilState(s->dss_off.get(), 0);
   c.ctx->VSSetShader(s->fs_vs.get(), nullptr, 0);
   c.ctx->PSSetShader(s->dn_ps.get(), nullptr, 0);
   c.ctx->PSSetSamplers(0, 1, smps);
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      c.ctx->PSSetShaderResources(0, 2, null_srvs);
      ID3D11RenderTargetView *br[1] = {s->bloom_rtv[i].get()};
      c.ctx->OMSetRenderTargets(1, br, nullptr);
      D3D11_VIEWPORT bvp = {0.0f, 0.0f, (float)s->bw[i], (float)s->bh[i], 0.0f, 1.0f};
      c.ctx->RSSetViewports(1, &bvp);
      ID3D11ShaderResourceView *sr[1] = {i == 0 ? s->hdr_srv.get() : s->bloom_srv[i - 1u].get()};
      ID3D11Buffer *cb[1] = {s->cb_dn[i].get()};
      c.ctx->PSSetShaderResources(0, 1, sr);
      c.ctx->PSSetConstantBuffers(0, 1, cb);
      c.ctx->Draw(3, 0);
   }

   /* 3. 4-level additive tent upsample */
   c.ctx->OMSetBlendState(s->bs_add.get(), nullptr, 0xffffffffu);
   c.ctx->PSSetShader(s->up_ps.get(), nullptr, 0);
   for (int i = (int)GB_POSTFX_BLOOM_MIPS - 2; i >= 0; i--) {
      c.ctx->PSSetShaderResources(0, 2, null_srvs);
      ID3D11RenderTargetView *br[1] = {s->bloom_rtv[i].get()};
      c.ctx->OMSetRenderTargets(1, br, nullptr);
      D3D11_VIEWPORT bvp = {0.0f, 0.0f, (float)s->bw[i], (float)s->bh[i], 0.0f, 1.0f};
      c.ctx->RSSetViewports(1, &bvp);
      ID3D11ShaderResourceView *sr[1] = {s->bloom_srv[i + 1].get()};
      ID3D11Buffer *cb[1] = {s->cb_up[i].get()};
      c.ctx->PSSetShaderResources(0, 1, sr);
      c.ctx->PSSetConstantBuffers(0, 1, cb);
      c.ctx->Draw(3, 0);
   }

   /* 4. ACES tonemap + vignette */
   c.ctx->OMSetBlendState(nullptr, nullptr, 0xffffffffu);
   c.ctx->PSSetShaderResources(0, 2, null_srvs);
   ID3D11RenderTargetView *fr[1] = {s->final_rtv.get()};
   c.ctx->OMSetRenderTargets(1, fr, nullptr);
   c.ctx->RSSetViewports(1, &vp);
   c.ctx->PSSetShader(s->tn_ps.get(), nullptr, 0);
   ID3D11ShaderResourceView *ts[2] = {s->hdr_srv.get(), s->bloom_srv[0].get()};
   ID3D11Buffer *tcb[1] = {s->cb_tn.get()};
   c.ctx->PSSetShaderResources(0, 2, ts);
   c.ctx->PSSetConstantBuffers(0, 1, tcb);
   c.ctx->Draw(3, 0);
}

static int
run_render_postfx(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"res", "msaa", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   d11_postfx s;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1440p"), &s.w, &s.h)) gb_die("bad res");
   long ms = gb_param_int(o, "msaa", 4);
   if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
   s.msaa = (uint32_t)ms;

   hr_check(c.dev->CreateVertexShader(gb_dxbc_postfx_scene_vs, gb_dxbc_postfx_scene_vs_size, nullptr, s.sc_vs.put()), "CreateVS(pfx_sc)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_postfx_scene_ps, gb_dxbc_postfx_scene_ps_size, nullptr, s.sc_ps.put()), "CreatePS(pfx_sc)");
   hr_check(c.dev->CreateVertexShader(gb_dxbc_fs_vs, gb_dxbc_fs_vs_size, nullptr, s.fs_vs.put()), "CreateVS(pfx_fs)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_postfx_down_ps, gb_dxbc_postfx_down_ps_size, nullptr, s.dn_ps.put()), "CreatePS(pfx_dn)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_postfx_up_ps, gb_dxbc_postfx_up_ps_size, nullptr, s.up_ps.put()), "CreatePS(pfx_up)");
   hr_check(c.dev->CreatePixelShader(gb_dxbc_postfx_tonemap_ps, gb_dxbc_postfx_tonemap_ps_size, nullptr, s.tn_ps.put()), "CreatePS(pfx_tn)");

   D3D11_TEXTURE2D_DESC td = {};
   td.Width = s.w; td.Height = s.h; td.MipLevels = 1; td.ArraySize = 1;
   td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.SampleDesc.Count = 1;
   td.Usage = D3D11_USAGE_DEFAULT;
   td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, s.hdr_res.put()), "CreateTexture2D(hdr_res)");
   hr_check(c.dev->CreateShaderResourceView(s.hdr_res.get(), nullptr, s.hdr_srv.put()), "CreateSRV(hdr_res)");
   if (s.msaa == 4u) {
      td.SampleDesc.Count = 4; td.BindFlags = D3D11_BIND_RENDER_TARGET;
      hr_check(c.dev->CreateTexture2D(&td, nullptr, s.hdr_ms.put()), "CreateTexture2D(hdr_ms)");
      hr_check(c.dev->CreateRenderTargetView(s.hdr_ms.get(), nullptr, s.hdr_rtv.put()), "CreateRTV(hdr_ms)");
   } else {
      hr_check(c.dev->CreateRenderTargetView(s.hdr_res.get(), nullptr, s.hdr_rtv.put()), "CreateRTV(hdr_res)");
   }

   D3D11_TEXTURE2D_DESC dd = {};
   dd.Width = s.w; dd.Height = s.h; dd.MipLevels = 1; dd.ArraySize = 1;
   dd.Format = DXGI_FORMAT_D32_FLOAT; dd.SampleDesc.Count = s.msaa;
   dd.Usage = D3D11_USAGE_DEFAULT; dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
   hr_check(c.dev->CreateTexture2D(&dd, nullptr, s.depth_ms.put()), "CreateTexture2D(pfx_depth)");
   hr_check(c.dev->CreateDepthStencilView(s.depth_ms.get(), nullptr, s.dsv.put()), "CreateDSV(pfx)");

   uint32_t mw = s.w, mh = s.h;
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      mw = mw > 1u ? mw / 2u : 1u;
      mh = mh > 1u ? mh / 2u : 1u;
      s.bw[i] = mw; s.bh[i] = mh;
      D3D11_TEXTURE2D_DESC bd = {};
      bd.Width = mw; bd.Height = mh; bd.MipLevels = 1; bd.ArraySize = 1;
      bd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; bd.SampleDesc.Count = 1;
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      hr_check(c.dev->CreateTexture2D(&bd, nullptr, s.bloom[i].put()), "CreateTexture2D(bloom)");
      hr_check(c.dev->CreateRenderTargetView(s.bloom[i].get(), nullptr, s.bloom_rtv[i].put()), "CreateRTV(bloom)");
      hr_check(c.dev->CreateShaderResourceView(s.bloom[i].get(), nullptr, s.bloom_srv[i].put()), "CreateSRV(bloom)");
      uint32_t sw = (i == 0) ? s.w : s.bw[i - 1u];
      uint32_t sh = (i == 0) ? s.h : s.bh[i - 1u];
      struct { float iw, ih, siw, sih; uint32_t bp, p0, p1, p2; } dcb = {
         1.0f / (float)mw, 1.0f / (float)mh, 1.0f / (float)sw, 1.0f / (float)sh, i == 0 ? 1u : 0u, 0, 0, 0
      };
      s.cb_dn[i] = make_cb(c, 32, false, &dcb);
   }
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS - 1u; i++) {
      float ucb[4] = {1.0f / (float)s.bw[i], 1.0f / (float)s.bh[i],
                      1.0f / (float)s.bw[i + 1u], 1.0f / (float)s.bh[i + 1u]};
      s.cb_up[i] = make_cb(c, 16, false, ucb);
   }
   float tcb[4] = {1.0f / (float)s.w, 1.0f / (float)s.h, 0, 0};
   s.cb_tn = make_cb(c, 16, false, tcb);

   D3D11_TEXTURE2D_DESC ftd = {};
   ftd.Width = s.w; ftd.Height = s.h; ftd.MipLevels = 1; ftd.ArraySize = 1;
   ftd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; ftd.SampleDesc.Count = 1;
   ftd.Usage = D3D11_USAGE_DEFAULT; ftd.BindFlags = D3D11_BIND_RENDER_TARGET;
   hr_check(c.dev->CreateTexture2D(&ftd, nullptr, s.final_rt.put()), "CreateTexture2D(pfx_final)");
   hr_check(c.dev->CreateRenderTargetView(s.final_rt.get(), nullptr, s.final_rtv.put()), "CreateRTV(pfx_final)");

   s.lin_smp = make_sampler(c, D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP);
   D3D11_BLEND_DESC abd = {};
   abd.RenderTarget[0].BlendEnable = TRUE;
   abd.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
   abd.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
   abd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
   abd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
   abd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
   abd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
   abd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
   hr_check(c.dev->CreateBlendState(&abd, s.bs_add.put()), "CreateBlendState(add)");

   D3D11_DEPTH_STENCIL_DESC dsd = {};
   dsd.DepthEnable = TRUE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dsd.DepthFunc = D3D11_COMPARISON_LESS;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_less.put()), "CreateDSS(less)");
   dsd.DepthEnable = FALSE; dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
   hr_check(c.dev->CreateDepthStencilState(&dsd, s.dss_off.put()), "CreateDSS(off)");
   s.rs = make_rs_nocull(c);

   struct gb_check chk = {};
   if (do_verify) {
      d11_postfx_rec(c, &s, 1);
      gpu_idle(c);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.final_rt.get(), 0, s.w, s.h, got.data());
      gb_postfx_check_fb(got.data(), s.w, s.h, &chk);
   }
   static const int defd[] = {1, 2};
   int rc = run_depths_helper(c, o, defd, 2, d11_postfx_rec, &s, &chk,
                              "fps", "fps", 1e6, "gpix_s", "Gpix/s", (double)s.w * s.h / 1e3, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 10. transfer.ops ======================= */

struct d11_xfer {
   enum { CLEAR_4M, CLEAR_16M, COPY_B2I, MIPCHAIN } kind = CLEAR_4M;
   uint32_t w = GB_XFER_4M_DIM, h = GB_XFER_4M_DIM, mips = 1, expected_crc = 0;
   ComPtr<ID3D11Texture2D> tex, staging_src;
   ComPtr<ID3D11RenderTargetView> rtv;
   ComPtr<ID3D11ShaderResourceView> srv;
};

static void
d11_xfer_rec(d3d11_ctx &c, void *u, uint64_t k)
{
   (void)k;
   d11_xfer *s = (d11_xfer *)u;
   if (s->kind == d11_xfer::CLEAR_4M || s->kind == d11_xfer::CLEAR_16M) {
      const float cv[4] = {192.0f / 255.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f};
      c.ctx->ClearRenderTargetView(s->rtv.get(), cv);
   } else if (s->kind == d11_xfer::COPY_B2I) {
      c.ctx->CopyResource(s->tex.get(), s->staging_src.get());
   } else {
      /* Real DXVK GenerateMips path (dxvk_meta_mipgen) across all 11 mip levels. */
      c.ctx->GenerateMips(s->srv.get());
   }
}

static int
run_transfer_ops(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *k = gb_param(o, "kind", "clear_4m");
   d11_xfer s;
   if (!strcmp(k, "clear_4m")) {
      s.kind = d11_xfer::CLEAR_4M; s.w = GB_XFER_4M_DIM; s.h = GB_XFER_4M_DIM; s.mips = 1;
      s.expected_crc = gb_transfer_clear_ref_crc(s.w, s.h);
   } else if (!strcmp(k, "clear_16m")) {
      s.kind = d11_xfer::CLEAR_16M; s.w = GB_XFER_16M_DIM; s.h = GB_XFER_16M_DIM; s.mips = 1;
      s.expected_crc = gb_transfer_clear_ref_crc(s.w, s.h);
   } else if (!strcmp(k, "copy_b2i")) {
      s.kind = d11_xfer::COPY_B2I; s.w = GB_XFER_16M_DIM; s.h = GB_XFER_16M_DIM; s.mips = 1;
   } else if (!strcmp(k, "mipchain")) {
      s.kind = d11_xfer::MIPCHAIN; s.w = GB_XFER_MIP_W; s.h = GB_XFER_MIP_H; s.mips = GB_XFER_MIP_LEVELS;
   } else {
      gb_die("kind must be clear_4m, clear_16m, copy_b2i or mipchain");
   }

   D3D11_TEXTURE2D_DESC td = {};
   td.Width = s.w; td.Height = s.h; td.MipLevels = s.mips; td.ArraySize = 1;
   td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
   td.Usage = D3D11_USAGE_DEFAULT;
   td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
   if (s.kind == d11_xfer::MIPCHAIN) td.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
   hr_check(c.dev->CreateTexture2D(&td, nullptr, s.tex.put()), "CreateTexture2D(xfer)");

   if (s.kind == d11_xfer::CLEAR_4M || s.kind == d11_xfer::CLEAR_16M) {
      hr_check(c.dev->CreateRenderTargetView(s.tex.get(), nullptr, s.rtv.put()), "CreateRTV(xfer)");
   } else if (s.kind == d11_xfer::COPY_B2I) {
      std::vector<uint32_t> src((size_t)s.w * s.h);
      gb_transfer_fill_rgba(s.w, s.h, src.data());
      s.expected_crc = gb_crc32(src.data(), src.size() * 4u);
      D3D11_TEXTURE2D_DESC std = td;
      std.Usage = D3D11_USAGE_STAGING; std.BindFlags = 0; std.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
      D3D11_SUBRESOURCE_DATA sd = {src.data(), s.w * 4u, 0};
      hr_check(c.dev->CreateTexture2D(&std, &sd, s.staging_src.put()), "CreateTexture2D(xfer_src)");
   } else {
      std::vector<uint32_t> mip0((size_t)s.w * s.h);
      gb_transfer_fill_mip0(s.w, s.h, mip0.data());
      c.ctx->UpdateSubresource(s.tex.get(), 0, nullptr, mip0.data(), s.w * 4u, 0);
      hr_check(c.dev->CreateShaderResourceView(s.tex.get(), nullptr, s.srv.put()), "CreateSRV(mip)");
   }

   struct gb_check chk = {};
   if (do_verify) {
      d11_xfer_rec(c, &s, 1);
      gpu_idle(c);
      if (s.kind != d11_xfer::MIPCHAIN) {
         std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
         readback_tex2d(c, s.tex.get(), 0, s.w, s.h, got.data());
         chk.done = true;
         chk.crc = gb_crc32(got.data(), got.size());
         chk.expected = s.expected_crc;
         chk.valid = (chk.crc == chk.expected);
      } else {
         uint32_t m4w = s.w >> 4u, m4h = s.h >> 4u;
         std::vector<uint8_t> m4((size_t)m4w * m4h * 4u);
         uint8_t mlast[4] = {};
         readback_tex2d(c, s.tex.get(), 4, m4w, m4h, m4.data());
         readback_tex2d(c, s.tex.get(), s.mips - 1u, 1, 1, mlast);
         gb_transfer_check_mipchain(m4.data(), m4w, m4h, mlast, &chk);
      }
   }

   double bytes = (s.kind == d11_xfer::MIPCHAIN) ? gb_d3d_mipchain_bytes() : ((double)s.w * s.h * 4.0);
   static const int defd[] = {1, 4};
   int rc = run_depths_helper(c, o, defd, 2, d11_xfer_rec, &s, &chk,
                              "gbps", "GB/s", bytes / 1e3, nullptr, nullptr, 0, "us_per_submit");
   unbind_all(c.ctx.get());
   return rc;
}

/* ======================= 11. compile.pso (DXBC -> SPIR-V -> GPU ISA) ======================= */

static int
run_compile_pso(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", "count", "salt", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "compute");
   bool is_gfx = !strcmp(kind, "graphics");
   if (!is_gfx && strcmp(kind, "compute") && strcmp(kind, "uber"))
      gb_die("kind must be compute, uber or graphics");
   long count = gb_param_int(o, "count", (long)GB_DXBC_PSO_VARIANTS);
   if (count < 1 || count > (long)GB_DXBC_PSO_VARIANTS)
      gb_die("count must be 1..%u", GB_DXBC_PSO_VARIANTS);

   ComPtr<ID3D11Buffer> out_buf;
   ComPtr<ID3D11UnorderedAccessView> out_uav;
   ComPtr<ID3D11Texture2D> out_rt;
   ComPtr<ID3D11RenderTargetView> out_rtv;
   ComPtr<ID3D11RasterizerState> rs = make_rs_nocull(c);
   if (!is_gfx) {
      out_buf = make_raw_buf(c, 4096, D3D11_BIND_UNORDERED_ACCESS);
      out_uav = make_raw_uav(c, out_buf.get(), 4096);
   } else {
      D3D11_TEXTURE2D_DESC td = {};
      td.Width = 64; td.Height = 64; td.MipLevels = 1; td.ArraySize = 1;
      td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
      hr_check(c.dev->CreateTexture2D(&td, nullptr, out_rt.put()), "CreateTexture2D(pso)");
      hr_check(c.dev->CreateRenderTargetView(out_rt.get(), nullptr, out_rtv.put()), "CreateRTV(pso)");
   }

   struct gb_samples cold_s = {}, warm_s = {};
   double cold_total = 0.0, warm_total = 0.0;
   D3D11_QUERY_DESC qd = {}; qd.Query = D3D11_QUERY_EVENT;
   ComPtr<ID3D11Query> sync_q;
   hr_check(c.dev->CreateQuery(&qd, sync_q.put()), "CreateQuery(pso_sync)");

   for (int phase = 0; phase < 2; phase++) {
      std::vector<ComPtr<ID3D11ComputeShader>> cs_list((size_t)count);
      std::vector<ComPtr<ID3D11VertexShader>> vs_list((size_t)count);
      std::vector<ComPtr<ID3D11PixelShader>> ps_list((size_t)count);

      uint64_t b0 = gb_now_ns();
      for (long i = 0; i < count; i++) {
         uint64_t t0 = gb_now_ns();
         if (!is_gfx) {
            const auto &b = gb_dxbc_pso_comp_variants[i];
            hr_check(c.dev->CreateComputeShader(b.data, b.size, nullptr, cs_list[i].put()), "CreateCS(pso)");
            ID3D11UnorderedAccessView *uavs[1] = {out_uav.get()};
            c.ctx->CSSetShader(cs_list[i].get(), nullptr, 0);
            c.ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
            c.ctx->Dispatch(1, 1, 1);
         } else {
            const auto &vb = gb_dxbc_pso_gfx_vs_variants[i];
            const auto &pb = gb_dxbc_pso_gfx_ps_variants[i];
            hr_check(c.dev->CreateVertexShader(vb.data, vb.size, nullptr, vs_list[i].put()), "CreateVS(pso)");
            hr_check(c.dev->CreatePixelShader(pb.data, pb.size, nullptr, ps_list[i].put()), "CreatePS(pso)");
            ID3D11RenderTargetView *rtvs[1] = {out_rtv.get()};
            D3D11_VIEWPORT vp = {0, 0, 64, 64, 0, 1};
            c.ctx->OMSetRenderTargets(1, rtvs, nullptr);
            c.ctx->RSSetViewports(1, &vp);
            c.ctx->RSSetState(rs.get());
            c.ctx->IASetInputLayout(nullptr);
            c.ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            c.ctx->VSSetShader(vs_list[i].get(), nullptr, 0);
            c.ctx->PSSetShader(ps_list[i].get(), nullptr, 0);
            c.ctx->Draw(3, 0);
         }
         /* Wait for the event query so DXVK completes lazy DXBC->SPIR-V->Vulkan PSO compilation. */
         c.ctx->End(sync_q.get());
         c.ctx->Flush();
         if (!wait_event_query(c.ctx.get(), sync_q.get()))
            gb_die("D3D11 compile.pso sync wait timed out");
         uint64_t t1 = gb_now_ns();
         gb_samples_push(phase == 0 ? &cold_s : &warm_s, (double)(t1 - t0) / 1e6);
      }
      uint64_t b1 = gb_now_ns();
      if (phase == 0) cold_total = (double)(b1 - b0) / 1e6;
      else warm_total = (double)(b1 - b0) / 1e6;
      unbind_all(c.ctx.get());
   }

   (void)do_verify;

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, nullptr);
   gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
   gb_result_metric(&r, "cold_per_pso_ms", "ms", true, &cold_s, true);
   gb_result_metric1(&r, "cold_total_ms", "ms", true, cold_total);
   gb_result_metric(&r, "warm_per_pso_ms", "ms", true, &warm_s, true);
   gb_result_metric1(&r, "warm_total_ms", "ms", true, warm_total);
   gb_result_metric1(&r, "cache_speedup", "x", false, warm_total > 0.0 ? cold_total / warm_total : 0.0);
   gb_result_end_metrics(&r);
   gb_json_obj_begin(&r.j, "counts"); gb_json_int(&r.j, "pipelines", count); gb_json_obj_end(&r.j);
   gb_result_finish(&r, "ok", "cold_per_pso_ms", nullptr);
   gb_samples_free(&cold_s);
   gb_samples_free(&warm_s);
   return 0;
}

/* ======================= 12. sync.signal (D3D11 event query turnaround) ======================= */

static int
run_sync_signal(d3d11_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "host");
   if (strcmp(kind, "host") && strcmp(kind, "query")) gb_die("kind must be host or query");

   d11_cl s;
   hr_check(c.dev->CreateComputeShader(gb_dxbc_light_cs, gb_dxbc_light_cs_size, nullptr, s.cs.put()), "CreateCS(sync)");
   s.out = make_raw_buf(c, 4096, D3D11_BIND_UNORDERED_ACCESS);
   s.uav = make_raw_uav(c, s.out.get(), 4096);
   s.cb = make_cb(c, 16);

   D3D11_QUERY_DESC qd = {}; qd.Query = D3D11_QUERY_EVENT;
   ComPtr<ID3D11Query> q;
   hr_check(c.dev->CreateQuery(&qd, q.put()), "CreateQuery(EVENT)");

   /* Prime the pipeline once before starting the timed window. */
   d11_cl_rec(c, &s, 0);
   c.ctx->End(q.get());
   c.ctx->Flush();
   bool bad = !wait_event_query(c.ctx.get(), q.get());

   struct gb_samples lat = {};
   const uint64_t start = gb_now_ns() + (uint64_t)o->warmup_ms * 1000000ull;
   const uint64_t end = start + (uint64_t)o->runs * (uint64_t)o->run_ms * 1000000ull;
   uint64_t v = 0;
   while (!bad && gb_now_ns() < end) {
      v++;
      d11_cl_rec(c, &s, v);
      uint64_t a = gb_now_ns();
      c.ctx->End(q.get());
      c.ctx->Flush();
      if (!wait_event_query(c.ctx.get(), q.get())) { bad = true; break; }
      uint64_t b = gb_now_ns();
      if (a >= start) gb_samples_push(&lat, (double)(b - a) / 1e3);
   }

   struct gb_check chk = {};
   if (do_verify && !bad) {
      uint32_t got[GB_LIGHT_LOCAL] = {};
      readback_buf(c, s.out.get(), got, sizeof(got));
      gb_d3d_check_compute_light(got, (uint32_t)v, &chk);
   }

   bool ok = !bad && lat.n > 0 && (!chk.done || chk.valid);
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", nullptr});
   gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (bad || !lat.n) {
      gb_result_finish(&r, "fail", nullptr, "D3D11 event query wait timed out");
   } else {
      gb_result_metric(&r, "latency_us", "us", true, &lat, false);
      gb_result_end_metrics(&r);
      gb_json_obj_begin(&r.j, "counts"); gb_json_int(&r.j, "submits", (int64_t)lat.n); gb_json_obj_end(&r.j);
      if (chk.done) gb_result_check(&r, &chk);
      gb_result_finish(&r, ok ? "ok" : "fail", "latency_us", nullptr);
   }
   gb_samples_free(&lat);
   unbind_all(c.ctx.get());
   return ok ? 0 : 1;
}

int
main(int argc, char **argv)
{
   bool do_verify = false;
   gb_d3d_strip_verify_flag(&argc, argv, &do_verify);
   struct gb_opts o;
   gb_parse_opts(&o, argc, argv);

   if (o.list) {
      puts("compute.light\ncompute.heavy\ncompute.conv2d\n"
           "render.light\nrender.heavy\nrender.draws\n"
           "render.deferred\nrender.shadow\nrender.postfx\n"
           "transfer.ops\ncompile.pso\nsync.signal");
      return 0;
   }
   if (o.warmup_ms < 0 || o.run_ms <= 0)
      gb_die("warmup-ms must be nonnegative and run-ms positive");

   d3d11_ctx c;
   d3d11_init(c, o.device);

   if (o.info) {
      struct gb_json j;
      gb_json_init(&j);
      gb_json_obj_begin(&j, nullptr);
      emit_identity(&j, c);
      gb_json_obj_end(&j);
      puts(j.buf);
      gb_json_free(&j);
      return 0;
   }
   if (!o.test) gb_die("need --test, --info or --list");

   if (!strcmp(o.test, "compute.light")) return run_compute_light(c, &o, do_verify);
   if (!strcmp(o.test, "compute.heavy")) return run_compute_heavy(c, &o, do_verify);
   if (!strcmp(o.test, "compute.conv2d")) return run_compute_conv2d(c, &o, do_verify);
   if (!strcmp(o.test, "render.light")) return run_render_basic(c, &o, false, do_verify);
   if (!strcmp(o.test, "render.heavy")) return run_render_basic(c, &o, true, do_verify);
   if (!strcmp(o.test, "render.draws")) return run_render_draws(c, &o, do_verify);
   if (!strcmp(o.test, "render.deferred")) return run_render_deferred(c, &o, do_verify);
   if (!strcmp(o.test, "render.shadow")) return run_render_shadow(c, &o, do_verify);
   if (!strcmp(o.test, "render.postfx")) return run_render_postfx(c, &o, do_verify);
   if (!strcmp(o.test, "transfer.ops")) return run_transfer_ops(c, &o, do_verify);
   if (!strcmp(o.test, "compile.pso")) return run_compile_pso(c, &o, do_verify);
   if (!strcmp(o.test, "sync.signal")) return run_sync_signal(c, &o, do_verify);
   gb_die("unknown test '%s' (use --list)", o.test);
   return 2;
}
