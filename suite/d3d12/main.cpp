// SPDX-License-Identifier: GPL-2.0-only
/* Real Direct3D 12 benchmark executor for gpubench.
 * Runs natively on Linux against real upstream vkd3d-proton
 * (libvkd3d-proton-d3d12.so + libvkd3d-proton-d3d12core.so) and builds from the
 * exact same source as a Windows PE binary (gpubench-d3d12.exe) for Wine/Proton
 * + vkd3d-proton or macOS D3DMetal/GPTK. */

#define INITGUID
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#else
#ifndef WIDL_EXPLICIT_AGGREGATE_RETURNS
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <vkd3d_windows.h>
#include <vkd3d_d3d12.h>
#undef min
#undef max
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gb_d3d_common.h"
#include "shaders_dxil.h"

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
      gb_die("D3D12 %s failed: HRESULT 0x%08x", what, (unsigned)hr);
}

struct d3d12_ctx {
   ComPtr<ID3D12Device> dev;
   ComPtr<ID3D12CommandQueue> queue;
   ComPtr<ID3D12Fence> fence;
   uint64_t fence_val = 0;
   uint64_t ts_freq = 0;
   UINT rtv_inc = 0, dsv_inc = 0, cbv_srv_uav_inc = 0;
   char adapter_name[256] = "vkd3d-proton";
   uint32_t vendor_id = 0, device_id = 0;
   uint32_t shader_model = 0x60;
#ifdef _WIN32
   HANDLE fence_ev = nullptr;
#endif
};

static bool
wait_fence_val(d3d12_ctx &c, ID3D12Fence *f, uint64_t val)
{
   if (f->GetCompletedValue() >= val)
      return true;
#ifdef _WIN32
   if (FAILED(f->SetEventOnCompletion(val, c.fence_ev)))
      return false;
   return WaitForSingleObject(c.fence_ev, 10000) == WAIT_OBJECT_0;
#else
   (void)c;
   return SUCCEEDED(f->SetEventOnCompletion(val, nullptr));
#endif
}

static void
gpu_idle(d3d12_ctx &c)
{
   uint64_t v = ++c.fence_val;
   hr_check(c.queue->Signal(c.fence.get(), v), "Queue::Signal(idle)");
   if (!wait_fence_val(c, c.fence.get(), v))
      gb_die("D3D12 GPU idle wait failed");
}

static void
d3d12_init(d3d12_ctx &c, int dev_idx)
{
#ifdef _WIN32
   ComPtr<IDXGIFactory4> factory;
   ComPtr<IDXGIAdapter1> adapter;
   hr_check(CreateDXGIFactory1(IID_IDXGIFactory4, (void **)factory.put()), "CreateDXGIFactory1");
   hr_check(factory->EnumAdapters1((UINT)dev_idx, adapter.put()), "EnumAdapters1");
   DXGI_ADAPTER_DESC1 desc = {};
   adapter->GetDesc1(&desc);
   size_t j = 0;
   for (size_t i = 0; desc.Description[i] && j + 1 < sizeof(c.adapter_name); i++)
      c.adapter_name[j++] = (desc.Description[i] >= 0x20 && desc.Description[i] < 0x7f) ? (char)desc.Description[i] : '?';
   c.adapter_name[j] = 0;
   c.vendor_id = desc.VendorId;
   c.device_id = desc.DeviceId;
   hr_check(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0, IID_ID3D12Device, (void **)c.dev.put()),
            "D3D12CreateDevice");
   c.fence_ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
   if (!c.fence_ev) gb_die("CreateEventA failed");
#else
   (void)dev_idx;
   hr_check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_ID3D12Device, (void **)c.dev.put()),
            "D3D12CreateDevice");
   const char *flt = getenv("VKD3D_FILTER_DEVICE_NAME");
   if (flt && *flt) snprintf(c.adapter_name, sizeof(c.adapter_name), "%s", flt);
#endif

   D3D12_COMMAND_QUEUE_DESC qd = {};
   qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
   hr_check(c.dev->CreateCommandQueue(&qd, IID_ID3D12CommandQueue, (void **)c.queue.put()), "CreateCommandQueue");
   hr_check(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)c.fence.put()), "CreateFence");
   c.queue->GetTimestampFrequency(&c.ts_freq);
   c.rtv_inc = c.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   c.dsv_inc = c.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
   c.cbv_srv_uav_inc = c.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

   D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
   if (SUCCEEDED(c.dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
      c.shader_model = (uint32_t)sm.HighestShaderModel;
}

static void
emit_identity(struct gb_json *j, const d3d12_ctx &c)
{
   gb_json_str(j, "api", "d3d12");
#ifdef _WIN32
   gb_json_str(j, "backend", "d3d12-win32");
#else
   gb_json_str(j, "backend", "vkd3d-proton-native");
#endif
   gb_json_str(j, "adapter", c.adapter_name);
   gb_json_int(j, "vendor_id", c.vendor_id);
   gb_json_int(j, "device_id", c.device_id);
   gb_json_int(j, "timestamp_freq_hz", (int64_t)c.ts_freq);
   char smbuf[16];
   snprintf(smbuf, sizeof(smbuf), "%u_%u", (c.shader_model >> 4) & 0xf, c.shader_model & 0xf);
   gb_json_str(j, "shader_model", smbuf);
   gb_json_str(j, "suite_sha", GB_SUITE_SHA);
   const char *vk_drv = getenv("VK_DRIVER_FILES");
   gb_json_str(j, "vk_driver_files", vk_drv ? vk_drv : "");
}

/* ---- D3D12 resource & helper utilities ---- */

static D3D12_HEAP_PROPERTIES
heap_props(D3D12_HEAP_TYPE t)
{
   D3D12_HEAP_PROPERTIES p = {};
   p.Type = t;
   return p;
}

static D3D12_RESOURCE_DESC
buf_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
   D3D12_RESOURCE_DESC d = {};
   d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
   d.Width = bytes;
   d.Height = 1;
   d.DepthOrArraySize = 1;
   d.MipLevels = 1;
   d.Format = DXGI_FORMAT_UNKNOWN;
   d.SampleDesc.Count = 1;
   d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
   d.Flags = flags;
   return d;
}

static D3D12_RESOURCE_DESC
tex2d_desc(uint32_t w, uint32_t h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
           uint16_t mips = 1, uint16_t arr = 1, uint32_t samples = 1)
{
   D3D12_RESOURCE_DESC d = {};
   d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
   d.Width = w;
   d.Height = h;
   d.DepthOrArraySize = arr;
   d.MipLevels = mips;
   d.Format = fmt;
   d.SampleDesc.Count = samples;
   d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
   d.Flags = flags;
   return d;
}

static ComPtr<ID3D12Resource>
make_buf(d3d12_ctx &c, UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES init_state,
         D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
   D3D12_HEAP_PROPERTIES hp = heap_props(heap);
   D3D12_RESOURCE_DESC rd = buf_desc(bytes, flags);
   ComPtr<ID3D12Resource> r;
   hr_check(c.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, nullptr,
                                           IID_ID3D12Resource, (void **)r.put()),
            "CreateCommittedResource(buf)");
   return r;
}

static ComPtr<ID3D12Resource>
make_tex2d(d3d12_ctx &c, uint32_t w, uint32_t h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
           D3D12_RESOURCE_STATES init_state, const D3D12_CLEAR_VALUE *cv = nullptr,
           uint16_t mips = 1, uint16_t arr = 1, uint32_t samples = 1)
{
   D3D12_HEAP_PROPERTIES hp = heap_props(D3D12_HEAP_TYPE_DEFAULT);
   D3D12_RESOURCE_DESC rd = tex2d_desc(w, h, fmt, flags, mips, arr, samples);
   ComPtr<ID3D12Resource> r;
   hr_check(c.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init_state, cv,
                                           IID_ID3D12Resource, (void **)r.put()),
            "CreateCommittedResource(tex2d)");
   return r;
}

static ComPtr<ID3D12DescriptorHeap>
make_heap(d3d12_ctx &c, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT num, bool shader_vis = false)
{
   D3D12_DESCRIPTOR_HEAP_DESC hd = {};
   hd.Type = type;
   hd.NumDescriptors = num;
   hd.Flags = shader_vis ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
   ComPtr<ID3D12DescriptorHeap> h;
   hr_check(c.dev->CreateDescriptorHeap(&hd, IID_ID3D12DescriptorHeap, (void **)h.put()), "CreateDescriptorHeap");
   return h;
}

static D3D12_CPU_DESCRIPTOR_HANDLE
cpu_h(ID3D12DescriptorHeap *h, UINT inc, UINT idx)
{
   D3D12_CPU_DESCRIPTOR_HANDLE r = h->GetCPUDescriptorHandleForHeapStart();
   r.ptr += (SIZE_T)inc * idx;
   return r;
}

static D3D12_GPU_DESCRIPTOR_HANDLE
gpu_h(ID3D12DescriptorHeap *h, UINT inc, UINT idx)
{
   D3D12_GPU_DESCRIPTOR_HANDLE r = h->GetGPUDescriptorHandleForHeapStart();
   r.ptr += (UINT64)inc * idx;
   return r;
}

static void
barrier_tr(ID3D12GraphicsCommandList *cl, ID3D12Resource *res,
           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after,
           UINT sub = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
{
   if (before == after) return;
   D3D12_RESOURCE_BARRIER b = {};
   b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
   b.Transition.pResource = res;
   b.Transition.Subresource = sub;
   b.Transition.StateBefore = before;
   b.Transition.StateAfter = after;
   cl->ResourceBarrier(1, &b);
}

static ComPtr<ID3D12RootSignature>
make_rs(d3d12_ctx &c, const D3D12_ROOT_SIGNATURE_DESC &desc)
{
   ComPtr<ID3DBlob> blob, err;
   HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, blob.put(), err.put());
   if (FAILED(hr))
      gb_die("D3D12SerializeRootSignature failed: %s", err ? (const char *)err->GetBufferPointer() : "");
   ComPtr<ID3D12RootSignature> rs;
   hr_check(c.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                       IID_ID3D12RootSignature, (void **)rs.put()),
            "CreateRootSignature");
   return rs;
}

static ComPtr<ID3D12PipelineState>
make_cs_pso(d3d12_ctx &c, ID3D12RootSignature *rs, const unsigned char *dxil, size_t sz)
{
   D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
   pd.pRootSignature = rs;
   pd.CS = {dxil, sz};
   ComPtr<ID3D12PipelineState> pso;
   hr_check(c.dev->CreateComputePipelineState(&pd, IID_ID3D12PipelineState, (void **)pso.put()), "CreateComputePSO");
   return pso;
}

static D3D12_GRAPHICS_PIPELINE_STATE_DESC
base_gfx_pso(ID3D12RootSignature *rs, const unsigned char *vs, size_t vsz,
             const unsigned char *ps, size_t psz, UINT nrtv, const DXGI_FORMAT *rtvf,
             DXGI_FORMAT dsvf = DXGI_FORMAT_UNKNOWN, UINT samples = 1)
{
   D3D12_GRAPHICS_PIPELINE_STATE_DESC d = {};
   d.pRootSignature = rs;
   d.VS = {vs, vsz};
   if (ps && psz) d.PS = {ps, psz};
   d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   for (UINT i = 1; i < 8; i++)
      d.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
   d.SampleMask = 0xffffffffu;
   d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
   d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
   d.RasterizerState.DepthClipEnable = TRUE;
   d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
   d.NumRenderTargets = nrtv;
   for (UINT i = 0; i < nrtv; i++) d.RTVFormats[i] = rtvf[i];
   d.DSVFormat = dsvf;
   d.SampleDesc.Count = samples;
   return d;
}

static void
oneshot_run(d3d12_ctx &c, void (*fn)(ID3D12GraphicsCommandList *, void *), void *u)
{
   ComPtr<ID3D12CommandAllocator> al;
   ComPtr<ID3D12GraphicsCommandList> cl;
   hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void **)al.put()), "CreateCA");
   hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.get(), nullptr, IID_ID3D12GraphicsCommandList, (void **)cl.put()), "CreateCL");
   fn(cl.get(), u);
   hr_check(cl->Close(), "CL::Close");
   ID3D12CommandList *lists[1] = {cl.get()};
   c.queue->ExecuteCommandLists(1, lists);
   gpu_idle(c);
}

static void
upload_buf(d3d12_ctx &c, ID3D12Resource *dst, const void *src, UINT64 bytes, D3D12_RESOURCE_STATES after_state)
{
   ComPtr<ID3D12Resource> up = make_buf(c, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
   void *p = nullptr;
   D3D12_RANGE rr = {0, 0};
   hr_check(up->Map(0, &rr, &p), "Map(upload)");
   memcpy(p, src, (size_t)bytes);
   up->Unmap(0, nullptr);
   struct arg { ID3D12Resource *dst, *up; UINT64 bytes; D3D12_RESOURCE_STATES after; } a = {dst, up.get(), bytes, after_state};
   oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *ua) {
      auto *x = (arg *)ua;
      cl->CopyBufferRegion(x->dst, 0, x->up, 0, x->bytes);
      barrier_tr(cl, x->dst, D3D12_RESOURCE_STATE_COPY_DEST, x->after);
   }, &a);
}

static void
readback_buf(d3d12_ctx &c, ID3D12Resource *src, D3D12_RESOURCE_STATES cur_state, void *dst, UINT64 bytes)
{
   ComPtr<ID3D12Resource> rb = make_buf(c, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
   struct arg { ID3D12Resource *src, *rb; D3D12_RESOURCE_STATES st; UINT64 bytes; } a = {src, rb.get(), cur_state, bytes};
   oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *ua) {
      auto *x = (arg *)ua;
      barrier_tr(cl, x->src, x->st, D3D12_RESOURCE_STATE_COPY_SOURCE);
      cl->CopyBufferRegion(x->rb, 0, x->src, 0, x->bytes);
      barrier_tr(cl, x->src, D3D12_RESOURCE_STATE_COPY_SOURCE, x->st);
   }, &a);
   void *p = nullptr;
   D3D12_RANGE rr = {0, (SIZE_T)bytes};
   hr_check(rb->Map(0, &rr, &p), "Map(readback)");
   memcpy(dst, p, (size_t)bytes);
   D3D12_RANGE wr = {0, 0};
   rb->Unmap(0, &wr);
}

static UINT
align256(UINT v) { return (v + 255u) & ~255u; }

static void
upload_tex2d(d3d12_ctx &c, ID3D12Resource *dst, UINT mip, uint32_t w, uint32_t h,
             const void *src, D3D12_RESOURCE_STATES cur_state, D3D12_RESOURCE_STATES after_state)
{
   UINT pitch = align256(w * 4u);
   UINT64 bytes = (UINT64)pitch * h;
   ComPtr<ID3D12Resource> up = make_buf(c, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
   void *p = nullptr;
   D3D12_RANGE rr = {0, 0};
   hr_check(up->Map(0, &rr, &p), "Map(tex_up)");
   for (uint32_t y = 0; y < h; y++)
      memcpy((uint8_t *)p + (size_t)y * pitch, (const uint8_t *)src + (size_t)y * w * 4u, (size_t)w * 4u);
   up->Unmap(0, nullptr);

   struct arg {
      ID3D12Resource *dst, *up; UINT mip, w, h, pitch;
      D3D12_RESOURCE_STATES cur, after;
   } a = {dst, up.get(), mip, w, h, pitch, cur_state, after_state};
   oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *ua) {
      auto *x = (arg *)ua;
      barrier_tr(cl, x->dst, x->cur, D3D12_RESOURCE_STATE_COPY_DEST, x->mip);
      D3D12_TEXTURE_COPY_LOCATION dloc = {}, sloc = {};
      dloc.pResource = x->dst; dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dloc.SubresourceIndex = x->mip;
      sloc.pResource = x->up; sloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      sloc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      sloc.PlacedFootprint.Footprint.Width = x->w;
      sloc.PlacedFootprint.Footprint.Height = x->h;
      sloc.PlacedFootprint.Footprint.Depth = 1;
      sloc.PlacedFootprint.Footprint.RowPitch = x->pitch;
      cl->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
      barrier_tr(cl, x->dst, D3D12_RESOURCE_STATE_COPY_DEST, x->after, x->mip);
   }, &a);
}

static void
readback_tex2d(d3d12_ctx &c, ID3D12Resource *src, UINT mip, uint32_t w, uint32_t h,
               D3D12_RESOURCE_STATES cur_state, uint8_t *dst)
{
   UINT pitch = align256(w * 4u);
   UINT64 bytes = (UINT64)pitch * h;
   ComPtr<ID3D12Resource> rb = make_buf(c, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
   struct arg {
      ID3D12Resource *src, *rb; UINT mip, w, h, pitch; D3D12_RESOURCE_STATES st;
   } a = {src, rb.get(), mip, w, h, pitch, cur_state};
   oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *ua) {
      auto *x = (arg *)ua;
      barrier_tr(cl, x->src, x->st, D3D12_RESOURCE_STATE_COPY_SOURCE, x->mip);
      D3D12_TEXTURE_COPY_LOCATION dloc = {}, sloc = {};
      sloc.pResource = x->src; sloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sloc.SubresourceIndex = x->mip;
      dloc.pResource = x->rb; dloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      dloc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      dloc.PlacedFootprint.Footprint.Width = x->w;
      dloc.PlacedFootprint.Footprint.Height = x->h;
      dloc.PlacedFootprint.Footprint.Depth = 1;
      dloc.PlacedFootprint.Footprint.RowPitch = x->pitch;
      cl->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
      barrier_tr(cl, x->src, D3D12_RESOURCE_STATE_COPY_SOURCE, x->st, x->mip);
   }, &a);
   void *p = nullptr;
   D3D12_RANGE rr = {0, (SIZE_T)bytes};
   hr_check(rb->Map(0, &rr, &p), "Map(tex_rb)");
   for (uint32_t y = 0; y < h; y++)
      memcpy(dst + (size_t)y * w * 4u, (const uint8_t *)p + (size_t)y * pitch, (size_t)w * 4u);
   D3D12_RANGE wr = {0, 0};
   rb->Unmap(0, &wr);
}

/* ---- Submission stream engine (single or multi-command-list per slot) ---- */

typedef void (*d12_rec_fn)(ID3D12GraphicsCommandList *const *cls, int ncl, void *user, uint64_t k);

static void
run_stream(d3d12_ctx &c, int depth, int lists_per_slot, bool timed, int warm_ms, int runs, int run_ms,
           d12_rec_fn rec, void *user, struct gb_pass *p)
{
   struct slot {
      std::vector<ComPtr<ID3D12CommandAllocator>> als;
      std::vector<ComPtr<ID3D12GraphicsCommandList>> cls;
      std::vector<ID3D12CommandList *> raw_cls;
      std::vector<ID3D12GraphicsCommandList *> raw_gcls;
      uint64_t fence_val = 0;
      uint64_t submit_ns = 0;
   };
   std::vector<slot> slots((size_t)depth);
   for (int i = 0; i < depth; i++) {
      slots[i].als.resize((size_t)lists_per_slot);
      slots[i].cls.resize((size_t)lists_per_slot);
      slots[i].raw_cls.resize((size_t)lists_per_slot);
      slots[i].raw_gcls.resize((size_t)lists_per_slot);
      for (int l = 0; l < lists_per_slot; l++) {
         hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_ID3D12CommandAllocator, (void **)slots[i].als[l].put()), "CreateCA");
         hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slots[i].als[l].get(), nullptr,
                                           IID_ID3D12GraphicsCommandList, (void **)slots[i].cls[l].put()), "CreateCL");
         hr_check(slots[i].cls[l]->Close(), "CL::Close(init)");
         slots[i].raw_cls[l] = slots[i].cls[l].get();
         slots[i].raw_gcls[l] = slots[i].cls[l].get();
      }
   }

   ComPtr<ID3D12QueryHeap> qh;
   ComPtr<ID3D12Resource> qrb;
   const uint64_t *ts_ptr = nullptr;
   if (timed && c.ts_freq > 0) {
      D3D12_QUERY_HEAP_DESC qhd = {};
      qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      qhd.Count = (UINT)(2 * depth);
      hr_check(c.dev->CreateQueryHeap(&qhd, IID_ID3D12QueryHeap, (void **)qh.put()), "CreateQueryHeap(TS)");
      qrb = make_buf(c, (UINT64)(2 * depth) * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_RANGE rr = {0, (SIZE_T)(2 * depth) * sizeof(uint64_t)};
      void *mp = nullptr;
      hr_check(qrb->Map(0, &rr, &mp), "Map(ts_rb)");
      ts_ptr = (const uint64_t *)mp;
   } else {
      timed = false;
   }

   memset(p, 0, sizeof(*p));
   p->nwin = runs;
   p->win_s = run_ms / 1e3;
   p->timed = timed;

   const uint64_t start = gb_now_ns() + (uint64_t)warm_ms * 1000000ull;
   const uint64_t win = (uint64_t)run_ms * 1000000ull;
   const uint64_t end = start + (uint64_t)runs * win;
   uint64_t sent = 0, done = 0;
   double last_end_us = 0.0;
   bool have_last_end = false;
   const double ts_scale = timed ? (1e6 / (double)c.ts_freq) : 0.0;

   while (true) {
      uint64_t now = gb_now_ns();
      bool full = (sent - done) >= (uint64_t)depth;
      if (!full && now >= end) break;
      if (full) {
         int idx = (int)(done % (uint64_t)depth);
         slot &s = slots[idx];
         if (!wait_fence_val(c, c.fence.get(), s.fence_val)) { p->failed = true; break; }
         uint64_t t = gb_now_ns();
         bool measured = (t >= start && t < end);
         if (measured) {
            p->win_count[(t - start) / win]++;
            gb_samples_push(&p->latency, (double)(t - s.submit_ns) / 1e3);
         }
         if (timed) {
            uint64_t t0 = ts_ptr[2 * idx];
            uint64_t t1 = ts_ptr[2 * idx + 1];
            double b_us = (double)t0 * ts_scale;
            double e_us = (double)t1 * ts_scale;
            double eff = (have_last_end && last_end_us > b_us) ? last_end_us : b_us;
            if (measured) {
               gb_samples_push(&p->gpu, e_us > eff ? (e_us - eff) : 0.0);
               if (have_last_end)
                  gb_samples_push(&p->gap, b_us - last_end_us);
            }
            last_end_us = e_us;
            have_last_end = true;
         }
         done++;
         continue;
      }

      int idx = (int)(sent % (uint64_t)depth);
      slot &s = slots[idx];
      uint64_t r0 = gb_now_ns();
      for (int l = 0; l < lists_per_slot; l++) {
         hr_check(s.als[l]->Reset(), "CA::Reset");
         hr_check(s.cls[l]->Reset(s.als[l].get(), nullptr), "CL::Reset");
      }
      if (timed)
         s.cls[0]->EndQuery(qh.get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * idx));
      rec(s.raw_gcls.data(), lists_per_slot, user, sent + 1);
      if (timed) {
         auto *last_cl = s.cls[lists_per_slot - 1].get();
         last_cl->EndQuery(qh.get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * idx + 1));
         last_cl->ResolveQueryData(qh.get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * idx), 2,
                                   qrb.get(), (UINT64)(2 * idx) * sizeof(uint64_t));
      }
      for (int l = 0; l < lists_per_slot; l++)
         hr_check(s.cls[l]->Close(), "CL::Close");
      uint64_t r1 = gb_now_ns();
      s.submit_ns = r1;
      s.fence_val = ++c.fence_val;
      c.queue->ExecuteCommandLists((UINT)lists_per_slot, s.raw_cls.data());
      hr_check(c.queue->Signal(c.fence.get(), s.fence_val), "Queue::Signal");
      uint64_t r2 = gb_now_ns();
      if (r0 >= start && r0 < end) {
         gb_samples_push(&p->record_cpu, (double)(r1 - r0) / 1e3);
         gb_samples_push(&p->submit_cpu, (double)(r2 - r1) / 1e3);
      }
      sent++;
   }

   while (done < sent) {
      int idx = (int)(done % (uint64_t)depth);
      if (!p->failed && !wait_fence_val(c, c.fence.get(), slots[idx].fence_val))
         p->failed = true;
      done++;
   }
   if (qrb) {
      D3D12_RANGE wr = {0, 0};
      qrb->Unmap(0, &wr);
   }
   for (int i = 0; i < runs; i++)
      p->submits += p->win_count[i];
}

static int
run_depths_helper(d3d12_ctx &c, const struct gb_opts *o, const int *defd, int ndef, int lists_per_slot,
                  d12_rec_fn rec, void *user, const struct gb_check *chk,
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
         gb_result_finish(&r, "skip", nullptr, o->mode != GB_MODE_QUEUE ? "D3D12 supports mode=queue only" : "depth must be 1..64");
         continue;
      }
      bool timed = o->timestamps && o->timed_runs > 0 && c.ts_freq > 0;
      struct gb_pass u = {}, t = {};
      run_stream(c, depth, lists_per_slot, false, o->warmup_ms, o->runs, o->run_ms, rec, user, &u);
      if (timed && !u.failed)
         run_stream(c, depth, lists_per_slot, true, o->warmup_ms < 100 ? o->warmup_ms : 100, o->timed_runs, o->run_ms, rec, user, &t);

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depth, nullptr);
      gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (u.failed || t.failed || !u.submits || (timed && !t.submits)) {
         gb_result_finish(&r, "fail", nullptr, "D3D12 GPU wait/error or no measured completions");
         rc = 1;
      } else {
         gb_result_stream_metrics(&r, &u, timed ? &t : nullptr);
         if (m1 || m2) {
            struct gb_samples us = {};
            gb_pass_rate(&u, &us);
            if (m1) gb_derive_rate(&r, m1, u1, &us, w1);
            if (m2) gb_derive_rate(&r, m2, u2, &us, w2);
            gb_samples_free(&us);
         }
         gb_result_end_metrics(&r);
         gb_json_obj_begin(&r.j, "counts"); gb_json_int(&r.j, "submits", (int64_t)u.submits); gb_json_obj_end(&r.j);
         if (chk && chk->done) gb_result_check(&r, chk);
         gb_result_finish(&r, (chk && chk->done && !chk->valid) ? "fail" : "ok",
                          primary ? primary : "us_per_submit",
                          (chk && chk->done && !chk->valid) ? "output check failed" : nullptr);
      }
      gb_pass_free(&u);
      gb_pass_free(&t);
   }
   if (chk && chk->done && !chk->valid) rc = 1;
   return rc;
}

/* ======================= 1. compute.light ======================= */

struct d12_cl {
   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12PipelineState> pso;
   ComPtr<ID3D12Resource> out;
};

static void
d12_cl_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl;
   d12_cl *s = (d12_cl *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   uint32_t rc[4] = {(uint32_t)k, 0, 0, 0};
   cl->SetPipelineState(s->pso.get());
   cl->SetComputeRootSignature(s->rs.get());
   cl->SetComputeRoot32BitConstants(0, 4, rc, 0);
   cl->SetComputeRootUnorderedAccessView(1, s->out->GetGPUVirtualAddress());
   cl->Dispatch(1, 1, 1);
}

static int
run_compute_light(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   d12_cl s;
   D3D12_ROOT_PARAMETER rp[2] = {};
   rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
   rp[0].Constants = {0, 0, 4};
   rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
   rp[1].Descriptor = {0, 0};
   D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
   s.rs = make_rs(c, rsd);
   s.pso = make_cs_pso(c, s.rs.get(), gb_dxil_light_cs, gb_dxil_light_cs_size);
   s.out = make_buf(c, 4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_cl_rec(&cl, 1, u, 255); }, &s);
      uint32_t got[GB_LIGHT_LOCAL] = {};
      readback_buf(c, s.out.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got, sizeof(got));
      gb_d3d_check_compute_light(got, 255, &chk);
   }
   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   return run_depths_helper(c, o, defd, 7, 1, d12_cl_rec, &s, &chk, nullptr, nullptr, 0, nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 2. compute.heavy ======================= */

struct d12_ch {
   bool bw = false;
   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12PipelineState> pso;
   ComPtr<ID3D12Resource> a, b;
};

static void
d12_ch_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_ch *s = (d12_ch *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   cl->SetPipelineState(s->pso.get());
   cl->SetComputeRootSignature(s->rs.get());
   if (s->bw) {
      cl->SetComputeRootShaderResourceView(0, s->a->GetGPUVirtualAddress());
      cl->SetComputeRootUnorderedAccessView(1, s->b->GetGPUVirtualAddress());
      cl->Dispatch((UINT)(GB_BW_BYTES / 16u / GB_BW_LOCAL), 1, 1);
   } else {
      struct { uint32_t iters; float a, b; uint32_t pad; } rc = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B, 0};
      cl->SetComputeRoot32BitConstants(0, 4, &rc, 0);
      cl->SetComputeRootUnorderedAccessView(1, s->a->GetGPUVirtualAddress());
      cl->Dispatch(GB_ALU_GROUPS, 1, 1);
   }
}

static int
run_compute_heavy(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "alu");
   d12_ch s;
   if (!strcmp(kind, "bw")) s.bw = true;
   else if (strcmp(kind, "alu")) gb_die("kind must be alu or bw");

   if (s.bw) {
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; rp[0].Descriptor = {0, 0};
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor = {0, 0};
      D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.rs = make_rs(c, rsd);
      s.pso = make_cs_pso(c, s.rs.get(), gb_dxil_bw_cs, gb_dxil_bw_cs_size);
      s.a = make_buf(c, GB_BW_BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
      s.b = make_buf(c, GB_BW_BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      std::vector<float> init(GB_BW_BYTES / sizeof(float));
      for (size_t i = 0; i < init.size(); i++) init[i] = (float)(i & 255u);
      upload_buf(c, s.a.get(), init.data(), GB_BW_BYTES, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
   } else {
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[0].Constants = {0, 0, 4};
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor = {0, 0};
      D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.rs = make_rs(c, rsd);
      s.pso = make_cs_pso(c, s.rs.get(), gb_dxil_alu_cs, gb_dxil_alu_cs_size);
      s.a = make_buf(c, (UINT64)GB_ALU_GROUPS * GB_ALU_LOCAL * 4u, D3D12_HEAP_TYPE_DEFAULT,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   }

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_ch_rec(&cl, 1, u, 1); }, &s);
      if (s.bw) {
         std::vector<float> got(GB_BW_BYTES / sizeof(float));
         readback_buf(c, s.b.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got.data(), GB_BW_BYTES);
         gb_d3d_check_compute_bw(got.data(), &chk);
      } else {
         std::vector<float> got(GB_ALU_GROUPS * GB_ALU_LOCAL);
         readback_buf(c, s.a.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got.data(), got.size() * sizeof(float));
         gb_d3d_check_compute_alu(got.data(), &chk);
      }
   }
   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   return run_depths_helper(c, o, defd, 7, 1, d12_ch_rec, &s, &chk,
                            s.bw ? "gbps" : "gflops", s.bw ? "GB/s" : "GFLOP/s",
                            s.bw ? (2.0 * GB_BW_BYTES / 1e9 * 1e6) : (GB_ALU_FLOP / 1e9 * 1e6),
                            nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 3. compute.conv2d ======================= */

struct d12_conv {
   uint32_t w = 0, h = 0;
   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12PipelineState> pso_h, pso_v;
   ComPtr<ID3D12Resource> in, mid, out;
};

static void
d12_conv_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_conv *s = (d12_conv *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   UINT gx = (s->w + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;
   UINT gy = (s->h + GB_CONV_LOCAL - 1u) / GB_CONV_LOCAL;
   int32_t rc[4] = {(int32_t)s->w, (int32_t)s->h, 0, 0};

   cl->SetComputeRootSignature(s->rs.get());
   cl->SetPipelineState(s->pso_h.get());
   cl->SetComputeRoot32BitConstants(0, 4, rc, 0);
   cl->SetComputeRootShaderResourceView(1, s->in->GetGPUVirtualAddress());
   cl->SetComputeRootUnorderedAccessView(2, s->mid->GetGPUVirtualAddress());
   cl->Dispatch(gx, gy, 1);

   barrier_tr(cl, s->mid.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

   cl->SetPipelineState(s->pso_v.get());
   cl->SetComputeRoot32BitConstants(0, 4, rc, 0);
   cl->SetComputeRootShaderResourceView(1, s->mid->GetGPUVirtualAddress());
   cl->SetComputeRootUnorderedAccessView(2, s->out->GetGPUVirtualAddress());
   cl->Dispatch(gx, gy, 1);

   barrier_tr(cl, s->mid.get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

static int
run_compute_conv2d(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", "res", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "gaussian");
   bool bilateral = !strcmp(kind, "bilateral");
   if (!bilateral && strcmp(kind, "gaussian")) gb_die("kind must be gaussian or bilateral");
   uint32_t w = 0, h = 0;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &w, &h)) gb_die("bad res");

   d12_conv s; s.w = w; s.h = h;
   size_t count = (size_t)w * h, bytes = count * sizeof(uint32_t);
   std::vector<uint32_t> in_cpu(count), ref(count);
   gb_conv2d_fill_input(in_cpu.data(), w, h);
   uint32_t ref_crc = do_verify ? gb_conv2d_ref(bilateral, in_cpu.data(), ref.data(), w, h) : 0;

   D3D12_ROOT_PARAMETER rp[3] = {};
   rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[0].Constants = {0, 0, 4};
   rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; rp[1].Descriptor = {0, 0};
   rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[2].Descriptor = {0, 0};
   D3D12_ROOT_SIGNATURE_DESC rsd = {3, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
   s.rs = make_rs(c, rsd);
   s.pso_h = make_cs_pso(c, s.rs.get(),
                         bilateral ? gb_dxil_conv2d_h_bilat_cs : gb_dxil_conv2d_h_gauss_cs,
                         bilateral ? gb_dxil_conv2d_h_bilat_cs_size : gb_dxil_conv2d_h_gauss_cs_size);
   s.pso_v = make_cs_pso(c, s.rs.get(),
                         bilateral ? gb_dxil_conv2d_v_bilat_cs : gb_dxil_conv2d_v_gauss_cs,
                         bilateral ? gb_dxil_conv2d_v_bilat_cs_size : gb_dxil_conv2d_v_gauss_cs_size);
   s.in = make_buf(c, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
   s.mid = make_buf(c, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   s.out = make_buf(c, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   upload_buf(c, s.in.get(), in_cpu.data(), bytes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_conv_rec(&cl, 1, u, 1); }, &s);
      std::vector<uint32_t> got(count);
      readback_buf(c, s.out.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got.data(), bytes);
      gb_check_words(&chk, got.data(), ref.data(), ref_crc, count);
   }
   static const int defd[] = {1, 4};
   return run_depths_helper(c, o, defd, 2, 1, d12_conv_rec, &s, &chk,
                            "mpix_s", "Mpix/s", (double)w * h, nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 4 & 5. render.light & render.heavy ======================= */

struct d12_render {
   enum { TRI, CLEAR, FILL, OPAQUE, TEXTURE, GEOM } kind = TRI;
   uint32_t w = 64, h = 64, msaa = 1;
   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12PipelineState> pso;
   ComPtr<ID3D12Resource> rt, rt_ms, tex;
   ComPtr<ID3D12DescriptorHeap> rtv_heap, srv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv_h = {};
};

static void
d12_render_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl;
   d12_render *s = (d12_render *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   const float clr[4] = {0.1f, 0.2f, 0.3f, 1.0f};
   cl->OMSetRenderTargets(1, &s->rtv_h, FALSE, nullptr);
   cl->ClearRenderTargetView(s->rtv_h, clr, 0, nullptr);
   if (s->kind != d12_render::CLEAR) {
      D3D12_VIEWPORT vp = {0, 0, (float)s->w, (float)s->h, 0, 1};
      D3D12_RECT sc = {0, 0, (LONG)s->w, (LONG)s->h};
      cl->RSSetViewports(1, &vp);
      cl->RSSetScissorRects(1, &sc);
      cl->SetGraphicsRootSignature(s->rs.get());
      cl->SetPipelineState(s->pso.get());
      cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      float col[4] = {(float)(k & 255u) / 255.0f, 0.5f, 0.25f, 0.5f};
      cl->SetGraphicsRoot32BitConstants(0, 4, col, 0);
      if (s->kind == d12_render::TEXTURE) {
         ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
         cl->SetDescriptorHeaps(1, hs);
         cl->SetGraphicsRootDescriptorTable(1, s->srv_heap->GetGPUDescriptorHandleForHeapStart());
      }
      if (s->kind == d12_render::TRI) cl->DrawInstanced(3, 1, 0, 0);
      else if (s->kind == d12_render::GEOM) cl->DrawInstanced(3u * GB_GEOM_TRIS, 1, 0, 0);
      else cl->DrawInstanced(3, GB_FILL_LAYERS, 0, 0);
   }
   if (s->msaa == 4u) {
      barrier_tr(cl, s->rt_ms.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
      barrier_tr(cl, s->rt.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);
      cl->ResolveSubresource(s->rt.get(), 0, s->rt_ms.get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
      barrier_tr(cl, s->rt_ms.get(), D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
      barrier_tr(cl, s->rt.get(), D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
   }
}

static int
run_render_basic(d3d12_ctx &c, const struct gb_opts *o, bool heavy, bool do_verify)
{
   static const char *const k_light[] = {"kind", nullptr};
   static const char *const k_heavy[] = {"kind", "res", "msaa", nullptr};
   if (!gb_check_params(o, heavy ? k_heavy : k_light)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", heavy ? "fill" : "tri");

   d12_render s;
   if (!heavy) {
      s.w = GB_LIGHT_DIM; s.h = GB_LIGHT_DIM; s.msaa = 1;
      if (!strcmp(kind, "tri")) s.kind = d12_render::TRI;
      else if (!strcmp(kind, "clear")) s.kind = d12_render::CLEAR;
      else gb_die("render.light kind must be tri or clear");
   } else {
      if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &s.w, &s.h)) gb_die("bad res");
      long ms = gb_param_int(o, "msaa", 1);
      if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
      s.msaa = (uint32_t)ms;
      if (!strcmp(kind, "fill")) s.kind = d12_render::FILL;
      else if (!strcmp(kind, "opaque")) s.kind = d12_render::OPAQUE;
      else if (!strcmp(kind, "texture")) s.kind = d12_render::TEXTURE;
      else if (!strcmp(kind, "geom")) s.kind = d12_render::GEOM;
      else gb_die("render.heavy kind must be fill, opaque, texture or geom");
   }

   D3D12_CLEAR_VALUE cv = {}; cv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
   cv.Color[0] = 0.1f; cv.Color[1] = 0.2f; cv.Color[2] = 0.3f; cv.Color[3] = 1.0f;
   s.rt = make_tex2d(c, s.w, s.h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                     D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, 1, 1, 1);
   s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
   s.rtv_h = s.rtv_heap->GetCPUDescriptorHandleForHeapStart();
   if (s.msaa == 4u) {
      s.rt_ms = make_tex2d(c, s.w, s.h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, 1, 1, 4);
      c.dev->CreateRenderTargetView(s.rt_ms.get(), nullptr, s.rtv_h);
   } else {
      c.dev->CreateRenderTargetView(s.rt.get(), nullptr, s.rtv_h);
   }

   if (s.kind != d12_render::CLEAR) {
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp[0].Constants = {0, 0, 4};
      rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_DESCRIPTOR_RANGE rng = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      rp[1].DescriptorTable = {1, &rng};
      rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_STATIC_SAMPLER_DESC ss = {};
      ss.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
      ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
      ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC rsd = {
         s.kind == d12_render::TEXTURE ? 2u : 1u, rp,
         s.kind == d12_render::TEXTURE ? 1u : 0u, &ss,
         D3D12_ROOT_SIGNATURE_FLAG_NONE
      };
      s.rs = make_rs(c, rsd);

      DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
      const unsigned char *vb = (s.kind == d12_render::GEOM) ? gb_dxil_geom_vs : gb_dxil_fs_vs;
      size_t vs = (s.kind == d12_render::GEOM) ? gb_dxil_geom_vs_size : gb_dxil_fs_vs_size;
      const unsigned char *pb = (s.kind == d12_render::TEXTURE) ? gb_dxil_texture_ps : gb_dxil_color_ps;
      size_t ps = (s.kind == d12_render::TEXTURE) ? gb_dxil_texture_ps_size : gb_dxil_color_ps_size;
      auto pd = base_gfx_pso(s.rs.get(), vb, vs, pb, ps, 1, &fmt, DXGI_FORMAT_UNKNOWN, s.msaa);
      if (s.kind == d12_render::FILL || s.kind == d12_render::TEXTURE) {
         auto &rtb = pd.BlendState.RenderTarget[0];
         rtb.BlendEnable = TRUE;
         rtb.SrcBlend = D3D12_BLEND_SRC_ALPHA;
         rtb.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
         rtb.BlendOp = D3D12_BLEND_OP_ADD;
         rtb.SrcBlendAlpha = D3D12_BLEND_ONE;
         rtb.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
         rtb.BlendOpAlpha = D3D12_BLEND_OP_ADD;
      }
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.pso.put()), "CreateGfxPSO");

      if (s.kind == d12_render::TEXTURE) {
         std::vector<uint8_t> data(GB_TEX_DIM * GB_TEX_DIM * 4u);
         for (uint32_t y = 0; y < GB_TEX_DIM; y++)
            for (uint32_t x = 0; x < GB_TEX_DIM; x++) {
               size_t idx = ((size_t)y * GB_TEX_DIM + x) * 4u;
               data[idx] = (uint8_t)x; data[idx + 1] = (uint8_t)y;
               data[idx + 2] = (uint8_t)(x ^ y); data[idx + 3] = 255;
            }
         s.tex = make_tex2d(c, GB_TEX_DIM, GB_TEX_DIM, DXGI_FORMAT_R8G8B8A8_UNORM,
                            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
         upload_tex2d(c, s.tex.get(), 0, GB_TEX_DIM, GB_TEX_DIM, data.data(),
                      D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         s.srv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true);
         c.dev->CreateShaderResourceView(s.tex.get(), nullptr, s.srv_heap->GetCPUDescriptorHandleForHeapStart());
      }
   }

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_render_rec(&cl, 1, u, 255); }, &s);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.rt.get(), 0, s.w, s.h, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
      gb_d3d_check_render_basic(got.data(), s.w, s.h, kind, 255, &chk);
   }
   static const int defd[] = {1, 2, 4, 8, 16, 32, 64};
   const char *m1 = nullptr, *u1 = nullptr;
   double w1 = 0.0;
   if (heavy) {
      if (s.kind == d12_render::GEOM) { m1 = "mtri_s"; u1 = "Mtri/s"; w1 = (double)GB_GEOM_TRIS; }
      else { m1 = "gpix_s"; u1 = "Gpix/s"; w1 = (double)s.w * s.h * GB_FILL_LAYERS / 1e9 * 1e6; }
   }
   return run_depths_helper(c, o, defd, 7, 1, d12_render_rec, &s, &chk, m1, u1, w1, nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 6. render.draws (direct CBV ring / instanced / ExecuteIndirect) ======================= */

struct d12_draws {
   enum { DIRECT, INSTANCED, INDIRECT } mode = DIRECT;
   uint32_t draws = 2048;
   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12PipelineState> pso;
   ComPtr<ID3D12CommandSignature> cmd_sig;
   ComPtr<ID3D12Resource> rt, vb, inst_vb, ib, cb_ring, ind_buf;
   ComPtr<ID3D12DescriptorHeap> rtv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv_h = {};
   D3D12_VERTEX_BUFFER_VIEW vbv[2] = {};
   D3D12_INDEX_BUFFER_VIEW ibv = {};
   D3D12_GPU_VIRTUAL_ADDRESS cb_gpu = 0;
};

static void
d12_draws_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_draws *s = (d12_draws *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   const float blk[4] = {0, 0, 0, 1};
   cl->OMSetRenderTargets(1, &s->rtv_h, FALSE, nullptr);
   cl->ClearRenderTargetView(s->rtv_h, blk, 0, nullptr);
   D3D12_VIEWPORT vp = {0, 0, 1024, 1024, 0, 1};
   D3D12_RECT sc = {0, 0, 1024, 1024};
   cl->RSSetViewports(1, &vp);
   cl->RSSetScissorRects(1, &sc);
   cl->SetGraphicsRootSignature(s->rs.get());
   cl->SetPipelineState(s->pso.get());
   cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   cl->IASetVertexBuffers(0, s->mode == d12_draws::DIRECT ? 1u : 2u, s->vbv);
   cl->IASetIndexBuffer(&s->ibv);

   if (s->mode == d12_draws::INSTANCED) {
      cl->DrawIndexedInstanced(6, s->draws, 0, 0, 0);
   } else if (s->mode == d12_draws::INDIRECT) {
      cl->ExecuteIndirect(s->cmd_sig.get(), s->draws, s->ind_buf.get(), 0, nullptr, 0);
   } else {
      for (uint32_t d = 0; d < s->draws; d++) {
         cl->SetGraphicsRootConstantBufferView(0, s->cb_gpu + (UINT64)d * 256ull);
         cl->DrawIndexedInstanced(6, 1, 0, 0, 0);
      }
   }
}

static int
run_render_draws(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"draws", "mode", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   long draws = gb_param_int(o, "draws", 2048);
   if (draws < 1 || draws > 10000) gb_die("draws must be 1..10000");
   const char *mstr = gb_param(o, "mode", "direct");

   d12_draws s; s.draws = (uint32_t)draws;
   if (!strcmp(mstr, "direct")) s.mode = d12_draws::DIRECT;
   else if (!strcmp(mstr, "instanced")) s.mode = d12_draws::INSTANCED;
   else if (!strcmp(mstr, "indirect") || !strcmp(mstr, "mdi")) s.mode = d12_draws::INDIRECT;
   else gb_die("mode must be direct, instanced or indirect");

   D3D12_ROOT_PARAMETER rp[1] = {};
   rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
   rp[0].Descriptor = {0, 0};
   rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
   D3D12_ROOT_SIGNATURE_DESC rsd = {
      s.mode == d12_draws::DIRECT ? 1u : 0u, rp, 0, nullptr,
      D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
   };
   s.rs = make_rs(c, rsd);

   D3D12_INPUT_ELEMENT_DESC ied[2] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32_UINT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
   };
   DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
   const unsigned char *vb = (s.mode == d12_draws::DIRECT) ? gb_dxil_draws_direct_vs : gb_dxil_draws_inst_vs;
   size_t vs = (s.mode == d12_draws::DIRECT) ? gb_dxil_draws_direct_vs_size : gb_dxil_draws_inst_vs_size;
   auto pd = base_gfx_pso(s.rs.get(), vb, vs, gb_dxil_draws_ps, gb_dxil_draws_ps_size, 1, &fmt);
   pd.InputLayout = {ied, s.mode == d12_draws::DIRECT ? 1u : 2u};
   hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.pso.put()), "CreateGfxPSO(draws)");

   D3D12_CLEAR_VALUE cv = {}; cv.Format = fmt; cv.Color[3] = 1.0f;
   s.rt = make_tex2d(c, 1024, 1024, fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
   s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
   s.rtv_h = s.rtv_heap->GetCPUDescriptorHandleForHeapStart();
   c.dev->CreateRenderTargetView(s.rt.get(), nullptr, s.rtv_h);

   const float verts[8] = {0, 0, 1, 0, 1, 1, 0, 1};
   const uint16_t idxs[6] = {0, 1, 2, 0, 2, 3};
   s.vb = make_buf(c, sizeof(verts), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
   s.ib = make_buf(c, sizeof(idxs), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
   upload_buf(c, s.vb.get(), verts, sizeof(verts), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
   upload_buf(c, s.ib.get(), idxs, sizeof(idxs), D3D12_RESOURCE_STATE_INDEX_BUFFER);
   s.vbv[0] = {s.vb->GetGPUVirtualAddress(), sizeof(verts), 8};
   if (s.mode != d12_draws::DIRECT) {
      std::vector<uint32_t> inst_ids(s.draws);
      for (uint32_t d = 0; d < s.draws; d++) inst_ids[d] = d;
      UINT64 ibytes = inst_ids.size() * 4ull;
      s.inst_vb = make_buf(c, ibytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
      upload_buf(c, s.inst_vb.get(), inst_ids.data(), ibytes, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
      s.vbv[1] = {s.inst_vb->GetGPUVirtualAddress(), (UINT)ibytes, 4};
   }
   s.ibv = {s.ib->GetGPUVirtualAddress(), sizeof(idxs), DXGI_FORMAT_R16_UINT};

   if (s.mode == d12_draws::DIRECT) {
      UINT64 bytes = (UINT64)s.draws * 256ull;
      s.cb_ring = make_buf(c, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
      void *p = nullptr; D3D12_RANGE rr = {0, 0};
      hr_check(s.cb_ring->Map(0, &rr, &p), "Map(cb_ring)");
      memset(p, 0, (size_t)bytes);
      for (uint32_t d = 0; d < s.draws; d++) {
         float *f = (float *)((uint8_t *)p + (size_t)d * 256u);
         uint32_t cell = d % 10000u;
         f[0] = (float)((cell % 100u) * 10u);
         f[1] = (float)((cell / 100u) * 10u);
         f[2] = (float)((d * 37u) & 255u) / 255.0f;
         f[3] = (float)((d * 73u) & 255u) / 255.0f;
      }
      s.cb_ring->Unmap(0, nullptr);
      s.cb_gpu = s.cb_ring->GetGPUVirtualAddress();
   } else if (s.mode == d12_draws::INDIRECT) {
      D3D12_INDIRECT_ARGUMENT_DESC iad = {};
      iad.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
      D3D12_COMMAND_SIGNATURE_DESC csd = {};
      csd.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
      csd.NumArgumentDescs = 1;
      csd.pArgumentDescs = &iad;
      hr_check(c.dev->CreateCommandSignature(&csd, nullptr, IID_ID3D12CommandSignature, (void **)s.cmd_sig.put()), "CreateCmdSig");
      std::vector<D3D12_DRAW_INDEXED_ARGUMENTS> args(s.draws);
      for (uint32_t d = 0; d < s.draws; d++) args[d] = {6u, 1u, 0u, 0, d};
      UINT64 bytes = args.size() * sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
      s.ind_buf = make_buf(c, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
      upload_buf(c, s.ind_buf.get(), args.data(), bytes, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
   }

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_draws_rec(&cl, 1, u, 1); }, &s);
      std::vector<uint8_t> got(1024u * 1024u * 4u);
      readback_tex2d(c, s.rt.get(), 0, 1024, 1024, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
      gb_d3d_check_draws_fb(got.data(), s.draws, &chk);
   }
   static const int defd[] = {1, 4};
   return run_depths_helper(c, o, defd, 2, 1, d12_draws_rec, &s, &chk,
                            "mdraw_s", "Mdraw/s", (double)s.draws, nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 7. render.deferred (+ split multi-command-list G-buffer) ======================= */

struct d12_deferred {
   uint32_t w = 1920, h = 1080, msaa = 1;
   bool split_cl = false;
   ComPtr<ID3D12RootSignature> g_rs, l_rs;
   ComPtr<ID3D12PipelineState> g_pso, l_pso;
   ComPtr<ID3D12Resource> g[4], depth, final_rt;
   ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap, srv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE g_rtv[4] = {}, final_rtv = {}, dsv_h = {};
};

static void
d12_deferred_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)k;
   d12_deferred *s = (d12_deferred *)u;
   D3D12_VIEWPORT vp = {0, 0, (float)s->w, (float)s->h, 0, 1};
   D3D12_RECT sc = {0, 0, (LONG)s->w, (LONG)s->h};
   const float z4[4] = {0, 0, 0, 0};

   if (s->split_cl && ncl == 5) {
      /* 4 split G-buffer command lists (1024 quads each) + 1 lighting command list. */
      for (int c_i = 0; c_i < 4; c_i++) {
         ID3D12GraphicsCommandList *cl = cls[c_i];
         cl->OMSetRenderTargets(4, s->g_rtv, FALSE, &s->dsv_h);
         if (c_i == 0) {
            for (int i = 0; i < 4; i++) cl->ClearRenderTargetView(s->g_rtv[i], z4, 0, nullptr);
            cl->ClearDepthStencilView(s->dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
         }
         cl->RSSetViewports(1, &vp);
         cl->RSSetScissorRects(1, &sc);
         cl->SetGraphicsRootSignature(s->g_rs.get());
         cl->SetPipelineState(s->g_pso.get());
         cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
         uint32_t rc[4] = {(uint32_t)c_i * 1024u, 0, 0, 0};
         cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
         cl->DrawInstanced(1024u * 6u, 1, 0, 0);
      }
      ID3D12GraphicsCommandList *cl = cls[4];
      for (int i = 0; i < 4; i++)
         barrier_tr(cl, s->g[i].get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      cl->OMSetRenderTargets(1, &s->final_rtv, FALSE, nullptr);
      cl->RSSetViewports(1, &vp);
      cl->RSSetScissorRects(1, &sc);
      cl->SetGraphicsRootSignature(s->l_rs.get());
      cl->SetPipelineState(s->l_pso.get());
      ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
      cl->SetDescriptorHeaps(1, hs);
      cl->SetGraphicsRootDescriptorTable(0, s->srv_heap->GetGPUDescriptorHandleForHeapStart());
      cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      cl->DrawInstanced(3, 1, 0, 0);
      for (int i = 0; i < 4; i++)
         barrier_tr(cl, s->g[i].get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
   } else {
      ID3D12GraphicsCommandList *cl = cls[0];
      cl->OMSetRenderTargets(4, s->g_rtv, FALSE, &s->dsv_h);
      for (int i = 0; i < 4; i++) cl->ClearRenderTargetView(s->g_rtv[i], z4, 0, nullptr);
      cl->ClearDepthStencilView(s->dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
      cl->RSSetViewports(1, &vp);
      cl->RSSetScissorRects(1, &sc);
      cl->SetGraphicsRootSignature(s->g_rs.get());
      cl->SetPipelineState(s->g_pso.get());
      cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      uint32_t rc[4] = {0, 0, 0, 0};
      cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
      cl->DrawInstanced(GB_DEFERRED_QUADS * 6u, 1, 0, 0);

      for (int i = 0; i < 4; i++)
         barrier_tr(cl, s->g[i].get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

      cl->OMSetRenderTargets(1, &s->final_rtv, FALSE, nullptr);
      cl->SetGraphicsRootSignature(s->l_rs.get());
      cl->SetPipelineState(s->l_pso.get());
      ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
      cl->SetDescriptorHeaps(1, hs);
      cl->SetGraphicsRootDescriptorTable(0, s->srv_heap->GetGPUDescriptorHandleForHeapStart());
      cl->DrawInstanced(3, 1, 0, 0);

      for (int i = 0; i < 4; i++)
         barrier_tr(cl, s->g[i].get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
   }
}

static int
run_render_deferred(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"mode", "res", "msaa", "split", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *mode = gb_param(o, "mode", "dram");
   bool split_cl = !strcmp(mode, "split") || gb_param_int(o, "split", 0) != 0;
   if (strcmp(mode, "dram") && strcmp(mode, "tile") && strcmp(mode, "split"))
      gb_die("mode must be dram, tile or split");
   d12_deferred s; s.split_cl = split_cl;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1080p"), &s.w, &s.h)) gb_die("bad res");
   long ms = gb_param_int(o, "msaa", 1);
   if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
   s.msaa = (uint32_t)ms;

   static const DXGI_FORMAT gfmts[4] = {
      DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
      DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT
   };
   {
      D3D12_ROOT_PARAMETER rp = {};
      rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp.Constants = {0, 0, 4};
      rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
      D3D12_ROOT_SIGNATURE_DESC rsd = {1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.g_rs = make_rs(c, rsd);
      auto pd = base_gfx_pso(s.g_rs.get(), gb_dxil_deferred_geom_vs, gb_dxil_deferred_geom_vs_size,
                             gb_dxil_deferred_geom_ps, gb_dxil_deferred_geom_ps_size, 4, gfmts, DXGI_FORMAT_D32_FLOAT, s.msaa);
      pd.DepthStencilState.DepthEnable = TRUE;
      pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.g_pso.put()), "CreateGfxPSO(def_g)");
   }
   {
      D3D12_DESCRIPTOR_RANGE rng = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, 0};
      D3D12_ROOT_PARAMETER rp = {};
      rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      rp.DescriptorTable = {1, &rng};
      rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC rsd = {1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.l_rs = make_rs(c, rsd);
      DXGI_FORMAT ffmt = DXGI_FORMAT_R8G8B8A8_UNORM;
      const unsigned char *lp = (s.msaa == 4u) ? gb_dxil_deferred_light_msaa_ps : gb_dxil_deferred_light_ps;
      size_t ls = (s.msaa == 4u) ? gb_dxil_deferred_light_msaa_ps_size : gb_dxil_deferred_light_ps_size;
      auto pd = base_gfx_pso(s.l_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size, lp, ls, 1, &ffmt);
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.l_pso.put()), "CreateGfxPSO(def_l)");
   }

   s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 5);
   s.dsv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1);
   s.srv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, true);

   for (int i = 0; i < 4; i++) {
      D3D12_CLEAR_VALUE cv = {}; cv.Format = gfmts[i];
      s.g[i] = make_tex2d(c, s.w, s.h, gfmts[i], D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_RENDER_TARGET, &cv, 1, 1, s.msaa);
      s.g_rtv[i] = cpu_h(s.rtv_heap.get(), c.rtv_inc, (UINT)i);
      c.dev->CreateRenderTargetView(s.g[i].get(), nullptr, s.g_rtv[i]);
      D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
      sd.Format = gfmts[i];
      sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      if (s.msaa == 4u) {
         sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
      } else {
         sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
         sd.Texture2D.MipLevels = 1;
      }
      c.dev->CreateShaderResourceView(s.g[i].get(), &sd, cpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, (UINT)i));
   }
   {
      D3D12_CLEAR_VALUE dcv = {}; dcv.Format = DXGI_FORMAT_D32_FLOAT; dcv.DepthStencil.Depth = 1.0f;
      s.depth = make_tex2d(c, s.w, s.h, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, &dcv, 1, 1, s.msaa);
      s.dsv_h = s.dsv_heap->GetCPUDescriptorHandleForHeapStart();
      c.dev->CreateDepthStencilView(s.depth.get(), nullptr, s.dsv_h);
   }
   {
      D3D12_CLEAR_VALUE fcv = {}; fcv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      s.final_rt = make_tex2d(c, s.w, s.h, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_RENDER_TARGET, &fcv);
      s.final_rtv = cpu_h(s.rtv_heap.get(), c.rtv_inc, 4);
      c.dev->CreateRenderTargetView(s.final_rt.get(), nullptr, s.final_rtv);
   }

   int ncl = s.split_cl ? 5 : 1;
   struct gb_check chk = {};
   if (do_verify) {
      std::vector<ComPtr<ID3D12CommandAllocator>> als((size_t)ncl);
      std::vector<ComPtr<ID3D12GraphicsCommandList>> cls((size_t)ncl);
      std::vector<ID3D12GraphicsCommandList *> raw_g((size_t)ncl);
      std::vector<ID3D12CommandList *> raw_c((size_t)ncl);
      for (int i = 0; i < ncl; i++) {
         hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void **)als[i].put()), "CA");
         hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, als[i].get(), nullptr, IID_ID3D12GraphicsCommandList, (void **)cls[i].put()), "CL");
         raw_g[i] = cls[i].get(); raw_c[i] = cls[i].get();
      }
      d12_deferred_rec(raw_g.data(), ncl, &s, 1);
      for (int i = 0; i < ncl; i++) hr_check(cls[i]->Close(), "Close");
      c.queue->ExecuteCommandLists((UINT)ncl, raw_c.data());
      gpu_idle(c);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.final_rt.get(), 0, s.w, s.h, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
      gb_deferred_check_fb(got.data(), s.w, s.h, &chk);
   }
   static const int defd[] = {1, 2};
   return run_depths_helper(c, o, defd, 2, ncl, d12_deferred_rec, &s, &chk,
                            "gpix_s", "Gpix/s", (double)s.w * s.h / 1e9 * 1e6,
                            "fps", "fps", 1e6, "us_per_submit");
}

/* ======================= 8. render.shadow ======================= */

struct d12_shadow {
   bool alpha = false;
   uint32_t res = 2048, inst = GB_SHADOW_DEFAULT_INST;
   ComPtr<ID3D12RootSignature> d_rs, p_rs;
   ComPtr<ID3D12PipelineState> d_pso, p_pso;
   ComPtr<ID3D12Resource> depth_arr, recv_rt;
   ComPtr<ID3D12DescriptorHeap> dsv_heap, rtv_heap, srv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE dsv[GB_SHADOW_CASCADES] = {}, recv_rtv = {};
};

static void
d12_shadow_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_shadow *s = (d12_shadow *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   D3D12_VIEWPORT svp = {0, 0, (float)s->res, (float)s->res, 0, 1};
   D3D12_RECT ssc = {0, 0, (LONG)s->res, (LONG)s->res};
   cl->RSSetViewports(1, &svp);
   cl->RSSetScissorRects(1, &ssc);
   cl->SetGraphicsRootSignature(s->d_rs.get());
   cl->SetPipelineState(s->d_pso.get());
   cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   for (uint32_t ci = 0; ci < GB_SHADOW_CASCADES; ci++) {
      cl->OMSetRenderTargets(0, nullptr, FALSE, &s->dsv[ci]);
      cl->ClearDepthStencilView(s->dsv[ci], D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
      uint32_t rc[4] = {ci, s->res, 0, 0};
      cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
      cl->DrawInstanced(6, s->inst, 0, 0);
   }

   barrier_tr(cl, s->depth_arr.get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

   const float blk[4] = {0, 0, 0, 1};
   cl->OMSetRenderTargets(1, &s->recv_rtv, FALSE, nullptr);
   cl->ClearRenderTargetView(s->recv_rtv, blk, 0, nullptr);
   D3D12_VIEWPORT rvp = {0, 0, (float)GB_SHADOW_RECV_DIM, (float)GB_SHADOW_RECV_DIM, 0, 1};
   D3D12_RECT rsc = {0, 0, (LONG)GB_SHADOW_RECV_DIM, (LONG)GB_SHADOW_RECV_DIM};
   cl->RSSetViewports(1, &rvp);
   cl->RSSetScissorRects(1, &rsc);
   cl->SetGraphicsRootSignature(s->p_rs.get());
   cl->SetPipelineState(s->p_pso.get());
   ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
   cl->SetDescriptorHeaps(1, hs);
   float rc[4] = {1.0f / (float)s->res, 0, 0, 0};
   cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
   cl->SetGraphicsRootDescriptorTable(1, s->srv_heap->GetGPUDescriptorHandleForHeapStart());
   cl->DrawInstanced(3, 1, 0, 0);

   barrier_tr(cl, s->depth_arr.get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
}

static int
run_render_shadow(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
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

   d12_shadow s; s.alpha = alpha; s.res = (uint32_t)res; s.inst = (uint32_t)inst;
   DXGI_FORMAT typeless = d16 ? DXGI_FORMAT_R16_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
   DXGI_FORMAT dsv_fmt = d16 ? DXGI_FORMAT_D16_UNORM : DXGI_FORMAT_D32_FLOAT;
   DXGI_FORMAT srv_fmt = d16 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT;

   {
      D3D12_ROOT_PARAMETER rp = {};
      rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp.Constants = {0, 0, 4}; rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
      D3D12_ROOT_SIGNATURE_DESC rsd = {1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.d_rs = make_rs(c, rsd);
      auto pd = base_gfx_pso(s.d_rs.get(), gb_dxil_shadow_depth_vs, gb_dxil_shadow_depth_vs_size,
                             alpha ? gb_dxil_shadow_alpha_ps : nullptr, alpha ? gb_dxil_shadow_alpha_ps_size : 0,
                             0, nullptr, dsv_fmt);
      pd.DepthStencilState.DepthEnable = TRUE;
      pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.d_pso.put()), "CreateGfxPSO(shad_d)");
   }
   {
      D3D12_DESCRIPTOR_RANGE rng = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp[0].Constants = {0, 0, 4}; rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      rp[1].DescriptorTable = {1, &rng}; rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_STATIC_SAMPLER_DESC ss = {};
      ss.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
      ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
      ss.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS;
      ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 1, &ss, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.p_rs = make_rs(c, rsd);
      DXGI_FORMAT cfmt = DXGI_FORMAT_R8G8B8A8_UNORM;
      auto pd = base_gfx_pso(s.p_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size,
                             gb_dxil_shadow_pcf_ps, gb_dxil_shadow_pcf_ps_size, 1, &cfmt);
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.p_pso.put()), "CreateGfxPSO(shad_p)");
   }

   D3D12_CLEAR_VALUE dcv = {}; dcv.Format = dsv_fmt; dcv.DepthStencil.Depth = 1.0f;
   s.depth_arr = make_tex2d(c, s.res, s.res, typeless, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                            D3D12_RESOURCE_STATE_DEPTH_WRITE, &dcv, 1, GB_SHADOW_CASCADES, 1);
   s.dsv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, GB_SHADOW_CASCADES);
   for (uint32_t ci = 0; ci < GB_SHADOW_CASCADES; ci++) {
      D3D12_DEPTH_STENCIL_VIEW_DESC dsvd = {};
      dsvd.Format = dsv_fmt; dsvd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
      dsvd.Texture2DArray.FirstArraySlice = ci; dsvd.Texture2DArray.ArraySize = 1;
      s.dsv[ci] = cpu_h(s.dsv_heap.get(), c.dsv_inc, ci);
      c.dev->CreateDepthStencilView(s.depth_arr.get(), &dsvd, s.dsv[ci]);
   }
   s.srv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true);
   D3D12_SHADER_RESOURCE_VIEW_DESC srvd = {};
   srvd.Format = srv_fmt; srvd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
   srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
   srvd.Texture2DArray.MipLevels = 1; srvd.Texture2DArray.ArraySize = GB_SHADOW_CASCADES;
   c.dev->CreateShaderResourceView(s.depth_arr.get(), &srvd, s.srv_heap->GetCPUDescriptorHandleForHeapStart());

   D3D12_CLEAR_VALUE ccv = {}; ccv.Format = DXGI_FORMAT_R8G8B8A8_UNORM; ccv.Color[3] = 1.0f;
   s.recv_rt = make_tex2d(c, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, &ccv);
   s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
   s.recv_rtv = s.rtv_heap->GetCPUDescriptorHandleForHeapStart();
   c.dev->CreateRenderTargetView(s.recv_rt.get(), nullptr, s.recv_rtv);

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_shadow_rec(&cl, 1, u, 1); }, &s);
      std::vector<uint8_t> got((size_t)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM * 4u);
      readback_tex2d(c, s.recv_rt.get(), 0, GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
      gb_shadow_check_fb(got.data(), GB_SHADOW_RECV_DIM, GB_SHADOW_RECV_DIM, s.alpha, &chk);
   }
   static const int defd[] = {1, 2};
   double tris = (double)GB_SHADOW_CASCADES * (double)s.inst * 2.0;
   double pix = (double)GB_SHADOW_CASCADES * (double)s.res * s.res + (double)GB_SHADOW_RECV_DIM * GB_SHADOW_RECV_DIM;
   return run_depths_helper(c, o, defd, 2, 1, d12_shadow_rec, &s, &chk,
                            "mtri_s", "Mtri/s", tris, "gpix_s", "Gpix/s", pix / 1e3, "us_per_submit");
}

/* ======================= 9. render.postfx ======================= */

struct d12_postfx {
   uint32_t w = 2560, h = 1440, msaa = 4;
   uint32_t bw[GB_POSTFX_BLOOM_MIPS] = {}, bh[GB_POSTFX_BLOOM_MIPS] = {};
   ComPtr<ID3D12RootSignature> sc_rs, dn_rs, up_rs, tn_rs;
   ComPtr<ID3D12PipelineState> sc_pso, dn_pso, up_pso, tn_pso;
   ComPtr<ID3D12Resource> hdr_ms, hdr_res, depth_ms, bloom[GB_POSTFX_BLOOM_MIPS], final_rt;
   ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap, srv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE hdr_rtv = {}, bloom_rtv[GB_POSTFX_BLOOM_MIPS] = {}, final_rtv = {}, dsv_h = {};
   /* srv_heap layout: [0]=hdr_res, [1..5]=bloom[0..4], [6]=hdr_res, [7]=bloom[0] (contiguous pair for tonemap!) */
   D3D12_GPU_DESCRIPTOR_HANDLE srv_hdr = {}, srv_bloom[GB_POSTFX_BLOOM_MIPS] = {}, srv_tone = {};
};

static void
d12_postfx_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_postfx *s = (d12_postfx *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
   cl->SetDescriptorHeaps(1, hs);
   cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

   /* 1. HDR scene pass (+ ResolveSubresource when msaa == 4) */
   const float bg[4] = {0.04f, 0.05f, 0.08f, 1.0f};
   cl->OMSetRenderTargets(1, &s->hdr_rtv, FALSE, &s->dsv_h);
   cl->ClearRenderTargetView(s->hdr_rtv, bg, 0, nullptr);
   cl->ClearDepthStencilView(s->dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
   D3D12_VIEWPORT vp = {0, 0, (float)s->w, (float)s->h, 0, 1};
   D3D12_RECT sc = {0, 0, (LONG)s->w, (LONG)s->h};
   cl->RSSetViewports(1, &vp);
   cl->RSSetScissorRects(1, &sc);
   cl->SetGraphicsRootSignature(s->sc_rs.get());
   cl->SetPipelineState(s->sc_pso.get());
   cl->DrawInstanced(GB_POSTFX_QUADS * 6u, 1, 0, 0);

   if (s->msaa == 4u) {
      barrier_tr(cl, s->hdr_ms.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
      barrier_tr(cl, s->hdr_res.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);
      cl->ResolveSubresource(s->hdr_res.get(), 0, s->hdr_ms.get(), 0, DXGI_FORMAT_R16G16B16A16_FLOAT);
      barrier_tr(cl, s->hdr_ms.get(), D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
      barrier_tr(cl, s->hdr_res.get(), D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
   } else {
      barrier_tr(cl, s->hdr_res.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
   }

   /* 2. 5-level bloom downsample */
   cl->SetGraphicsRootSignature(s->dn_rs.get());
   cl->SetPipelineState(s->dn_pso.get());
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      uint32_t sw = (i == 0) ? s->w : s->bw[i - 1u];
      uint32_t sh = (i == 0) ? s->h : s->bh[i - 1u];
      cl->OMSetRenderTargets(1, &s->bloom_rtv[i], FALSE, nullptr);
      D3D12_VIEWPORT bvp = {0, 0, (float)s->bw[i], (float)s->bh[i], 0, 1};
      D3D12_RECT bsc = {0, 0, (LONG)s->bw[i], (LONG)s->bh[i]};
      cl->RSSetViewports(1, &bvp);
      cl->RSSetScissorRects(1, &bsc);
      struct { float iw, ih, siw, sih; uint32_t bp; } rc = {
         1.0f / (float)s->bw[i], 1.0f / (float)s->bh[i], 1.0f / (float)sw, 1.0f / (float)sh, i == 0 ? 1u : 0u
      };
      cl->SetGraphicsRoot32BitConstants(0, 5, &rc, 0);
      cl->SetGraphicsRootDescriptorTable(1, i == 0 ? s->srv_hdr : s->srv_bloom[i - 1u]);
      cl->DrawInstanced(3, 1, 0, 0);
      barrier_tr(cl, s->bloom[i].get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
   }

   /* 3. 4-level additive tent upsample */
   cl->SetGraphicsRootSignature(s->up_rs.get());
   cl->SetPipelineState(s->up_pso.get());
   for (int i = (int)GB_POSTFX_BLOOM_MIPS - 2; i >= 0; i--) {
      barrier_tr(cl, s->bloom[i].get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
      cl->OMSetRenderTargets(1, &s->bloom_rtv[i], FALSE, nullptr);
      D3D12_VIEWPORT bvp = {0, 0, (float)s->bw[i], (float)s->bh[i], 0, 1};
      D3D12_RECT bsc = {0, 0, (LONG)s->bw[i], (LONG)s->bh[i]};
      cl->RSSetViewports(1, &bvp);
      cl->RSSetScissorRects(1, &bsc);
      float rc[4] = {1.0f / (float)s->bw[i], 1.0f / (float)s->bh[i],
                     1.0f / (float)s->bw[i + 1], 1.0f / (float)s->bh[i + 1]};
      cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
      cl->SetGraphicsRootDescriptorTable(1, s->srv_bloom[i + 1]);
      cl->DrawInstanced(3, 1, 0, 0);
      barrier_tr(cl, s->bloom[i].get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
   }

   /* 4. ACES tonemap + vignette */
   cl->OMSetRenderTargets(1, &s->final_rtv, FALSE, nullptr);
   cl->RSSetViewports(1, &vp);
   cl->RSSetScissorRects(1, &sc);
   cl->SetGraphicsRootSignature(s->tn_rs.get());
   cl->SetPipelineState(s->tn_pso.get());
   float trc[2] = {1.0f / (float)s->w, 1.0f / (float)s->h};
   cl->SetGraphicsRoot32BitConstants(0, 2, trc, 0);
   cl->SetGraphicsRootDescriptorTable(1, s->srv_tone);
   cl->DrawInstanced(3, 1, 0, 0);

   /* Restore states for next iteration */
   barrier_tr(cl, s->hdr_res.get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++)
      barrier_tr(cl, s->bloom[i].get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
}

static int
run_render_postfx(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"res", "msaa", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   d12_postfx s;
   if (!gb_d3d_parse_res(gb_param(o, "res", "1440p"), &s.w, &s.h)) gb_die("bad res");
   long ms = gb_param_int(o, "msaa", 4);
   if (ms != 1 && ms != 4) gb_die("msaa must be 1 or 4");
   s.msaa = (uint32_t)ms;

   D3D12_STATIC_SAMPLER_DESC ss = {};
   ss.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
   ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
   ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

   DXGI_FORMAT hdr_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
   DXGI_FORMAT ldr_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;

   {
      D3D12_ROOT_SIGNATURE_DESC rsd = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.sc_rs = make_rs(c, rsd);
      auto pd = base_gfx_pso(s.sc_rs.get(), gb_dxil_postfx_scene_vs, gb_dxil_postfx_scene_vs_size,
                             gb_dxil_postfx_scene_ps, gb_dxil_postfx_scene_ps_size, 1, &hdr_fmt, DXGI_FORMAT_D32_FLOAT, s.msaa);
      pd.DepthStencilState.DepthEnable = TRUE;
      pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.sc_pso.put()), "PSO(pfx_sc)");
   }
   auto make_pfx_rs = [&](UINT nconst, UINT nsrv) {
      D3D12_DESCRIPTOR_RANGE rng = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, nsrv, 0, 0, 0};
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp[0].Constants = {0, 0, nconst}; rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      rp[1].DescriptorTable = {1, &rng}; rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 1, &ss, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      return make_rs(c, rsd);
   };
   s.dn_rs = make_pfx_rs(5, 1);
   s.up_rs = make_pfx_rs(4, 1);
   s.tn_rs = make_pfx_rs(2, 2);
   {
      auto pd = base_gfx_pso(s.dn_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size, gb_dxil_postfx_down_ps, gb_dxil_postfx_down_ps_size, 1, &hdr_fmt);
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.dn_pso.put()), "PSO(pfx_dn)");
   }
   {
      auto pd = base_gfx_pso(s.up_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size, gb_dxil_postfx_up_ps, gb_dxil_postfx_up_ps_size, 1, &hdr_fmt);
      auto &b = pd.BlendState.RenderTarget[0];
      b.BlendEnable = TRUE;
      b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_ONE; b.BlendOp = D3D12_BLEND_OP_ADD;
      b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_ZERO; b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.up_pso.put()), "PSO(pfx_up)");
   }
   {
      auto pd = base_gfx_pso(s.tn_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size, gb_dxil_postfx_tonemap_ps, gb_dxil_postfx_tonemap_ps_size, 1, &ldr_fmt);
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.tn_pso.put()), "PSO(pfx_tn)");
   }

   s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2 + GB_POSTFX_BLOOM_MIPS);
   s.dsv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1);
   s.srv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, true);

   D3D12_CLEAR_VALUE hcv = {}; hcv.Format = hdr_fmt;
   hcv.Color[0] = 0.04f; hcv.Color[1] = 0.05f; hcv.Color[2] = 0.08f; hcv.Color[3] = 1.0f;
   s.hdr_res = make_tex2d(c, s.w, s.h, hdr_fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_RENDER_TARGET, &hcv);
   s.hdr_rtv = cpu_h(s.rtv_heap.get(), c.rtv_inc, 0);
   if (s.msaa == 4u) {
      s.hdr_ms = make_tex2d(c, s.w, s.h, hdr_fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_RENDER_TARGET, &hcv, 1, 1, 4);
      c.dev->CreateRenderTargetView(s.hdr_ms.get(), nullptr, s.hdr_rtv);
   } else {
      c.dev->CreateRenderTargetView(s.hdr_res.get(), nullptr, s.hdr_rtv);
   }
   D3D12_CLEAR_VALUE dcv = {}; dcv.Format = DXGI_FORMAT_D32_FLOAT; dcv.DepthStencil.Depth = 1.0f;
   s.depth_ms = make_tex2d(c, s.w, s.h, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                           D3D12_RESOURCE_STATE_DEPTH_WRITE, &dcv, 1, 1, s.msaa);
   s.dsv_h = s.dsv_heap->GetCPUDescriptorHandleForHeapStart();
   c.dev->CreateDepthStencilView(s.depth_ms.get(), nullptr, s.dsv_h);

   c.dev->CreateShaderResourceView(s.hdr_res.get(), nullptr, cpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 0));
   s.srv_hdr = gpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 0);

   uint32_t mw = s.w, mh = s.h;
   for (uint32_t i = 0; i < GB_POSTFX_BLOOM_MIPS; i++) {
      mw = mw > 1u ? mw / 2u : 1u; mh = mh > 1u ? mh / 2u : 1u;
      s.bw[i] = mw; s.bh[i] = mh;
      D3D12_CLEAR_VALUE bcv = {}; bcv.Format = hdr_fmt;
      s.bloom[i] = make_tex2d(c, mw, mh, hdr_fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_RENDER_TARGET, &bcv);
      s.bloom_rtv[i] = cpu_h(s.rtv_heap.get(), c.rtv_inc, 1u + i);
      c.dev->CreateRenderTargetView(s.bloom[i].get(), nullptr, s.bloom_rtv[i]);
      c.dev->CreateShaderResourceView(s.bloom[i].get(), nullptr, cpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 1u + i));
      s.srv_bloom[i] = gpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 1u + i);
   }
   c.dev->CreateShaderResourceView(s.hdr_res.get(), nullptr, cpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 6));
   c.dev->CreateShaderResourceView(s.bloom[0].get(), nullptr, cpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 7));
   s.srv_tone = gpu_h(s.srv_heap.get(), c.cbv_srv_uav_inc, 6);

   D3D12_CLEAR_VALUE fcv = {}; fcv.Format = ldr_fmt;
   s.final_rt = make_tex2d(c, s.w, s.h, ldr_fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_RENDER_TARGET, &fcv);
   s.final_rtv = cpu_h(s.rtv_heap.get(), c.rtv_inc, 1u + GB_POSTFX_BLOOM_MIPS);
   c.dev->CreateRenderTargetView(s.final_rt.get(), nullptr, s.final_rtv);

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_postfx_rec(&cl, 1, u, 1); }, &s);
      std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
      readback_tex2d(c, s.final_rt.get(), 0, s.w, s.h, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
      gb_postfx_check_fb(got.data(), s.w, s.h, &chk);
   }
   static const int defd[] = {1, 2};
   return run_depths_helper(c, o, defd, 2, 1, d12_postfx_rec, &s, &chk,
                            "fps", "fps", 1e6, "gpix_s", "Gpix/s", (double)s.w * s.h / 1e3, "us_per_submit");
}

/* ======================= 10. transfer.ops ======================= */

struct d12_xfer {
   enum { CLEAR_4M, CLEAR_16M, COPY_B2I, MIPCHAIN } kind = CLEAR_4M;
   uint32_t w = GB_XFER_4M_DIM, h = GB_XFER_4M_DIM, mips = 1, expected_crc = 0, pitch = 0;
   ComPtr<ID3D12Resource> tex, up_buf;
   ComPtr<ID3D12DescriptorHeap> rtv_heap, srv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtvs[GB_XFER_MIP_LEVELS] = {};
   ComPtr<ID3D12RootSignature> mip_rs;
   ComPtr<ID3D12PipelineState> mip_pso;
   uint32_t mw[GB_XFER_MIP_LEVELS] = {}, mh[GB_XFER_MIP_LEVELS] = {};
};

static void
d12_xfer_rec(ID3D12GraphicsCommandList *const *cls, int ncl, void *u, uint64_t k)
{
   (void)ncl; (void)k;
   d12_xfer *s = (d12_xfer *)u;
   ID3D12GraphicsCommandList *cl = cls[0];
   if (s->kind == d12_xfer::CLEAR_4M || s->kind == d12_xfer::CLEAR_16M) {
      const float cv[4] = {192.0f / 255.0f, 128.0f / 255.0f, 64.0f / 255.0f, 1.0f};
      cl->ClearRenderTargetView(s->rtvs[0], cv, 0, nullptr);
   } else if (s->kind == d12_xfer::COPY_B2I) {
      D3D12_TEXTURE_COPY_LOCATION dloc = {}, sloc = {};
      dloc.pResource = s->tex.get(); dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dloc.SubresourceIndex = 0;
      sloc.pResource = s->up_buf.get(); sloc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      sloc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      sloc.PlacedFootprint.Footprint.Width = s->w;
      sloc.PlacedFootprint.Footprint.Height = s->h;
      sloc.PlacedFootprint.Footprint.Depth = 1;
      sloc.PlacedFootprint.Footprint.RowPitch = s->pitch;
      cl->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
   } else {
      ID3D12DescriptorHeap *hs[1] = {s->srv_heap.get()};
      cl->SetDescriptorHeaps(1, hs);
      cl->SetGraphicsRootSignature(s->mip_rs.get());
      cl->SetPipelineState(s->mip_pso.get());
      cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      for (uint32_t i = 1; i < s->mips; i++) {
         barrier_tr(cl, s->tex.get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET, i);
         cl->OMSetRenderTargets(1, &s->rtvs[i], FALSE, nullptr);
         D3D12_VIEWPORT vp = {0, 0, (float)s->mw[i], (float)s->mh[i], 0, 1};
         D3D12_RECT sc = {0, 0, (LONG)s->mw[i], (LONG)s->mh[i]};
         cl->RSSetViewports(1, &vp);
         cl->RSSetScissorRects(1, &sc);
         float rc[4] = {1.0f / (float)s->mw[i], 1.0f / (float)s->mh[i], (float)(i - 1u), 0.0f};
         cl->SetGraphicsRoot32BitConstants(0, 4, rc, 0);
         cl->SetGraphicsRootDescriptorTable(1, s->srv_heap->GetGPUDescriptorHandleForHeapStart());
         cl->DrawInstanced(3, 1, 0, 0);
         barrier_tr(cl, s->tex.get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, i);
      }
   }
}

static int
run_transfer_ops(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *k = gb_param(o, "kind", "clear_4m");
   d12_xfer s;
   if (!strcmp(k, "clear_4m")) {
      s.kind = d12_xfer::CLEAR_4M; s.w = GB_XFER_4M_DIM; s.h = GB_XFER_4M_DIM; s.mips = 1;
      s.expected_crc = gb_transfer_clear_ref_crc(s.w, s.h);
   } else if (!strcmp(k, "clear_16m")) {
      s.kind = d12_xfer::CLEAR_16M; s.w = GB_XFER_16M_DIM; s.h = GB_XFER_16M_DIM; s.mips = 1;
      s.expected_crc = gb_transfer_clear_ref_crc(s.w, s.h);
   } else if (!strcmp(k, "copy_b2i")) {
      s.kind = d12_xfer::COPY_B2I; s.w = GB_XFER_16M_DIM; s.h = GB_XFER_16M_DIM; s.mips = 1;
   } else if (!strcmp(k, "mipchain")) {
      s.kind = d12_xfer::MIPCHAIN; s.w = GB_XFER_MIP_W; s.h = GB_XFER_MIP_H; s.mips = GB_XFER_MIP_LEVELS;
   } else {
      gb_die("kind must be clear_4m, clear_16m, copy_b2i or mipchain");
   }

   DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
   D3D12_RESOURCE_STATES st_init = (s.kind == d12_xfer::COPY_B2I) ? D3D12_RESOURCE_STATE_COPY_DEST :
                                   (s.kind == d12_xfer::MIPCHAIN) ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE :
                                                                    D3D12_RESOURCE_STATE_RENDER_TARGET;
   D3D12_RESOURCE_FLAGS fl = (s.kind == d12_xfer::COPY_B2I) ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
   D3D12_CLEAR_VALUE cv = {}; cv.Format = fmt;
   cv.Color[0] = 192.0f / 255.0f; cv.Color[1] = 128.0f / 255.0f; cv.Color[2] = 64.0f / 255.0f; cv.Color[3] = 1.0f;
   s.tex = make_tex2d(c, s.w, s.h, fmt, fl, st_init, (fl ? &cv : nullptr), (uint16_t)s.mips);

   if (s.kind == d12_xfer::CLEAR_4M || s.kind == d12_xfer::CLEAR_16M) {
      s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
      s.rtvs[0] = s.rtv_heap->GetCPUDescriptorHandleForHeapStart();
      c.dev->CreateRenderTargetView(s.tex.get(), nullptr, s.rtvs[0]);
   } else if (s.kind == d12_xfer::COPY_B2I) {
      std::vector<uint32_t> src((size_t)s.w * s.h);
      gb_transfer_fill_rgba(s.w, s.h, src.data());
      s.expected_crc = gb_crc32(src.data(), src.size() * 4u);
      s.pitch = align256(s.w * 4u);
      UINT64 bytes = (UINT64)s.pitch * s.h;
      s.up_buf = make_buf(c, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
      void *p = nullptr; D3D12_RANGE rr = {0, 0};
      hr_check(s.up_buf->Map(0, &rr, &p), "Map(xfer_up)");
      for (uint32_t y = 0; y < s.h; y++)
         memcpy((uint8_t *)p + (size_t)y * s.pitch, (const uint8_t *)src.data() + (size_t)y * s.w * 4u, (size_t)s.w * 4u);
      s.up_buf->Unmap(0, nullptr);
   } else {
      std::vector<uint32_t> mip0((size_t)s.w * s.h);
      gb_transfer_fill_mip0(s.w, s.h, mip0.data());
      upload_tex2d(c, s.tex.get(), 0, s.w, s.h, mip0.data(),
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      s.rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, s.mips);
      uint32_t mw = s.w, mh = s.h;
      for (uint32_t i = 0; i < s.mips; i++) {
         s.mw[i] = mw; s.mh[i] = mh;
         D3D12_RENDER_TARGET_VIEW_DESC rvd = {};
         rvd.Format = fmt; rvd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D; rvd.Texture2D.MipSlice = i;
         s.rtvs[i] = cpu_h(s.rtv_heap.get(), c.rtv_inc, i);
         c.dev->CreateRenderTargetView(s.tex.get(), &rvd, s.rtvs[i]);
         mw = mw > 1u ? mw / 2u : 1u; mh = mh > 1u ? mh / 2u : 1u;
      }
      s.srv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true);
      D3D12_SHADER_RESOURCE_VIEW_DESC srvd = {};
      srvd.Format = fmt; srvd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      srvd.Texture2D.MipLevels = s.mips;
      c.dev->CreateShaderResourceView(s.tex.get(), &srvd, s.srv_heap->GetCPUDescriptorHandleForHeapStart());

      D3D12_DESCRIPTOR_RANGE rng = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
      D3D12_ROOT_PARAMETER rp[2] = {};
      rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      rp[0].Constants = {0, 0, 4}; rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      rp[1].DescriptorTable = {1, &rng}; rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_STATIC_SAMPLER_DESC ss = {};
      ss.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
      ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
      ss.MaxLOD = D3D12_FLOAT32_MAX;
      ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 1, &ss, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      s.mip_rs = make_rs(c, rsd);
      auto pd = base_gfx_pso(s.mip_rs.get(), gb_dxil_fs_vs, gb_dxil_fs_vs_size, gb_dxil_mipgen_ps, gb_dxil_mipgen_ps_size, 1, &fmt);
      hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)s.mip_pso.put()), "PSO(mipgen)");
   }

   struct gb_check chk = {};
   if (do_verify) {
      oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) { d12_xfer_rec(&cl, 1, u, 1); }, &s);
      if (s.kind != d12_xfer::MIPCHAIN) {
         std::vector<uint8_t> got((size_t)s.w * s.h * 4u);
         readback_tex2d(c, s.tex.get(), 0, s.w, s.h, st_init, got.data());
         chk.done = true; chk.crc = gb_crc32(got.data(), got.size());
         chk.expected = s.expected_crc; chk.valid = (chk.crc == chk.expected);
      } else {
         uint32_t m4w = s.w >> 4u, m4h = s.h >> 4u;
         std::vector<uint8_t> m4((size_t)m4w * m4h * 4u);
         uint8_t mlast[4] = {};
         readback_tex2d(c, s.tex.get(), 4, m4w, m4h, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, m4.data());
         readback_tex2d(c, s.tex.get(), s.mips - 1u, 1, 1, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, mlast);
         gb_transfer_check_mipchain(m4.data(), m4w, m4h, mlast, &chk);
      }
   }
   double bytes = (s.kind == d12_xfer::MIPCHAIN) ? gb_d3d_mipchain_bytes() : ((double)s.w * s.h * 4.0);
   static const int defd[] = {1, 4};
   return run_depths_helper(c, o, defd, 2, 1, d12_xfer_rec, &s, &chk,
                            "gbps", "GB/s", bytes / 1e3, nullptr, nullptr, 0, "us_per_submit");
}

/* ======================= 11. compile.pso (DXIL -> SPIR-V -> GPU ISA) ======================= */

static int
run_compile_pso(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", "count", "salt", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "compute");
   bool is_gfx = !strcmp(kind, "graphics");
   if (!is_gfx && strcmp(kind, "compute") && strcmp(kind, "uber"))
      gb_die("kind must be compute, uber or graphics");
   long count = gb_param_int(o, "count", (long)GB_DXIL_PSO_VARIANTS);
   if (count < 1 || count > (long)GB_DXIL_PSO_VARIANTS)
      gb_die("count must be 1..%u", GB_DXIL_PSO_VARIANTS);

   ComPtr<ID3D12RootSignature> rs;
   ComPtr<ID3D12Resource> out_buf, out_rt;
   ComPtr<ID3D12DescriptorHeap> rtv_heap;
   D3D12_CPU_DESCRIPTOR_HANDLE rtv_h = {};
   DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;

   if (!is_gfx) {
      D3D12_ROOT_PARAMETER rp = {};
      rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp.Descriptor = {0, 0};
      D3D12_ROOT_SIGNATURE_DESC rsd = {1, &rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      rs = make_rs(c, rsd);
      out_buf = make_buf(c, 4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
   } else {
      D3D12_ROOT_SIGNATURE_DESC rsd = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
      rs = make_rs(c, rsd);
      D3D12_CLEAR_VALUE cv = {}; cv.Format = fmt;
      out_rt = make_tex2d(c, 64, 64, fmt, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv);
      rtv_heap = make_heap(c, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
      rtv_h = rtv_heap->GetCPUDescriptorHandleForHeapStart();
      c.dev->CreateRenderTargetView(out_rt.get(), nullptr, rtv_h);
   }

   struct gb_samples cold_s = {}, warm_s = {};
   double cold_total = 0.0, warm_total = 0.0;
   ComPtr<ID3D12PipelineState> last_pso;

   for (int phase = 0; phase < 2; phase++) {
      std::vector<ComPtr<ID3D12PipelineState>> psos((size_t)count);
      uint64_t b0 = gb_now_ns();
      for (long i = 0; i < count; i++) {
         uint64_t t0 = gb_now_ns();
         if (!is_gfx) {
            const auto &b = gb_dxil_pso_comp_variants[i];
            psos[i] = make_cs_pso(c, rs.get(), b.data, b.size);
         } else {
            const auto &vb = gb_dxil_pso_gfx_vs_variants[i];
            const auto &pb = gb_dxil_pso_gfx_ps_variants[i];
            auto pd = base_gfx_pso(rs.get(), vb.data, vb.size, pb.data, pb.size, 1, &fmt);
            hr_check(c.dev->CreateGraphicsPipelineState(&pd, IID_ID3D12PipelineState, (void **)psos[i].put()), "CreateGfxPSO(pso)");
         }
         uint64_t t1 = gb_now_ns();
         gb_samples_push(phase == 0 ? &cold_s : &warm_s, (double)(t1 - t0) / 1e6);
      }
      uint64_t b1 = gb_now_ns();
      if (phase == 0) cold_total = (double)(b1 - b0) / 1e6;
      else { warm_total = (double)(b1 - b0) / 1e6; last_pso = std::move(psos[count - 1]); }
   }

   /* Smoke-execute the last compiled pipeline. */
   struct smoke { bool is_gfx; ID3D12RootSignature *rs; ID3D12PipelineState *pso; ID3D12Resource *buf; D3D12_CPU_DESCRIPTOR_HANDLE rtv; } sm = {
      is_gfx, rs.get(), last_pso.get(), out_buf.get(), rtv_h
   };
   oneshot_run(c, [](ID3D12GraphicsCommandList *cl, void *u) {
      auto *x = (smoke *)u;
      if (!x->is_gfx) {
         cl->SetComputeRootSignature(x->rs);
         cl->SetPipelineState(x->pso);
         cl->SetComputeRootUnorderedAccessView(0, x->buf->GetGPUVirtualAddress());
         cl->Dispatch(1, 1, 1);
      } else {
         cl->OMSetRenderTargets(1, &x->rtv, FALSE, nullptr);
         D3D12_VIEWPORT vp = {0, 0, 64, 64, 0, 1}; D3D12_RECT sc = {0, 0, 64, 64};
         cl->RSSetViewports(1, &vp); cl->RSSetScissorRects(1, &sc);
         cl->SetGraphicsRootSignature(x->rs); cl->SetPipelineState(x->pso);
         cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
         cl->DrawInstanced(3, 1, 0, 0);
      }
   }, &sm);

   struct gb_check chk = {};
   if (do_verify) {
      chk.done = true;
      if (!is_gfx) {
         uint32_t got[64] = {};
         readback_buf(c, out_buf.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got, sizeof(got));
         chk.crc = gb_crc32(got, sizeof(got)); chk.expected = chk.crc;
         chk.valid = (got[0] != 0 || got[1] != 0);
      } else {
         std::vector<uint8_t> got(64u * 64u * 4u);
         readback_tex2d(c, out_rt.get(), 0, 64, 64, D3D12_RESOURCE_STATE_RENDER_TARGET, got.data());
         chk.crc = gb_crc32(got.data(), got.size()); chk.expected = chk.crc;
         chk.valid = (got[3] == 255);
      }
   }

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
   if (chk.done) gb_result_check(&r, &chk);
   gb_result_finish(&r, (chk.done && !chk.valid) ? "fail" : "ok", "cold_per_pso_ms", nullptr);
   gb_samples_free(&cold_s);
   gb_samples_free(&warm_s);
   return (chk.done && !chk.valid) ? 1 : 0;
}

/* ======================= 12. sync.signal (D3D12 fence host & cross-queue ping-pong) ======================= */

static int
run_sync_signal(d3d12_ctx &c, const struct gb_opts *o, bool do_verify)
{
   static const char *const known[] = {"kind", nullptr};
   if (!gb_check_params(o, known)) gb_die("unknown parameter");
   const char *kind = gb_param(o, "kind", "host");
   if (strcmp(kind, "host") && strcmp(kind, "queue")) gb_die("kind must be host or queue");

   d12_cl s;
   D3D12_ROOT_PARAMETER rp[2] = {};
   rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[0].Constants = {0, 0, 4};
   rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor = {0, 0};
   D3D12_ROOT_SIGNATURE_DESC rsd = {2, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
   s.rs = make_rs(c, rsd);
   s.pso = make_cs_pso(c, s.rs.get(), gb_dxil_light_cs, gb_dxil_light_cs_size);
   s.out = make_buf(c, 4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull;
   const uint64_t R = (uint64_t)o->run_ms * 1000000ull;

   if (!strcmp(kind, "host")) {
      ComPtr<ID3D12Fence> fT, fS;
      hr_check(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)fT.put()), "CreateFence(T)");
      hr_check(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)fS.put()), "CreateFence(S)");
      ComPtr<ID3D12CommandAllocator> al;
      ComPtr<ID3D12GraphicsCommandList> cl;
      hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void **)al.put()), "CA");
      hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.get(), nullptr, IID_ID3D12GraphicsCommandList, (void **)cl.put()), "CL");
      hr_check(cl->Close(), "Close");

      struct gb_samples lat = {}, base = {};
      uint64_t v = 0;
      bool bad = false;
      for (int phase = 0; phase < 2 && !bad; phase++) {
         uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
         while (gb_now_ns() < t_end) {
            v++;
            hr_check(al->Reset(), "CA::Reset");
            hr_check(cl->Reset(al.get(), nullptr), "CL::Reset");
            ID3D12GraphicsCommandList *gcl = cl.get();
            d12_cl_rec(&gcl, 1, &s, v);
            hr_check(cl->Close(), "CL::Close");
            ID3D12CommandList *lists[1] = {cl.get()};
            uint64_t a;
            if (phase == 0) {
               a = gb_now_ns();
               c.queue->ExecuteCommandLists(1, lists);
               hr_check(c.queue->Signal(fS.get(), v), "Signal(S)");
            } else {
               hr_check(c.queue->Wait(fT.get(), v), "Wait(T)");
               c.queue->ExecuteCommandLists(1, lists);
               hr_check(c.queue->Signal(fS.get(), v), "Signal(S)");
               gb_spin_ns(GB_HOST_SIGNAL_DELAY_NS);
               a = gb_now_ns();
               hr_check(fT->Signal(v), "Fence::Signal(T)");
            }
            if (!wait_fence_val(c, fS.get(), v)) { bad = true; break; }
            uint64_t b = gb_now_ns();
            if (a >= tm0) gb_samples_push(phase ? &lat : &base, (double)(b - a) / 1e3);
         }
      }
      struct gb_check chk = {};
      if (do_verify && !bad) {
         uint32_t got[GB_LIGHT_LOCAL] = {};
         readback_buf(c, s.out.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, got, sizeof(got));
         gb_d3d_check_compute_light(got, (uint32_t)v, &chk);
      }
      bool ok = !bad && lat.n > 0 && (!chk.done || chk.valid);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", nullptr});
      gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad || !lat.n) {
         gb_result_finish(&r, "fail", nullptr, "D3D12 fence wait failed");
      } else {
         gb_result_metric(&r, "latency_us", "us", true, &lat, false);
         gb_result_metric(&r, "base_latency_us", "us", true, &base, false);
         gb_result_metric1(&r, "added_us", "us", true, gb_median(&lat) - gb_median(&base));
         gb_result_end_metrics(&r);
         if (chk.done) gb_result_check(&r, &chk);
         gb_result_finish(&r, ok ? "ok" : "fail", "latency_us", nullptr);
      }
      gb_samples_free(&lat); gb_samples_free(&base);
      return ok ? 0 : 1;
   } else {
      /* Two D3D12 queues ping-ponging fences */
      D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
      ComPtr<ID3D12CommandQueue> q2;
      hr_check(c.dev->CreateCommandQueue(&qd, IID_ID3D12CommandQueue, (void **)q2.put()), "CreateQueue(COMPUTE)");
      ComPtr<ID3D12Fence> fS, fT;
      hr_check(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)fS.put()), "CreateFence(S)");
      hr_check(c.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence, (void **)fT.put()), "CreateFence(T)");

      enum { RING = GB_PING_RING };
      ComPtr<ID3D12CommandAllocator> ala[RING], alb[RING];
      ComPtr<ID3D12GraphicsCommandList> cla[RING], clb[RING];
      for (int i = 0; i < RING; i++) {
         hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_ID3D12CommandAllocator, (void **)ala[i].put()), "CAa");
         hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ala[i].get(), nullptr, IID_ID3D12GraphicsCommandList, (void **)cla[i].put()), "CLa");
         hr_check(cla[i]->Close(), "Close");
         hr_check(c.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_ID3D12CommandAllocator, (void **)alb[i].put()), "CAb");
         hr_check(c.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alb[i].get(), nullptr, IID_ID3D12GraphicsCommandList, (void **)clb[i].put()), "CLb");
         hr_check(clb[i]->Close(), "Close");
      }

      uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
      uint64_t k = 0, seen = 0, win[GB_MAX_WIN] = {};
      bool bad = false;
      for (;;) {
         uint64_t now = gb_now_ns();
         if ((k - seen >= RING) || (now >= t_end && seen < k)) {
            if (!wait_fence_val(c, fT.get(), seen + 1)) { bad = true; break; }
            uint64_t t = gb_now_ns();
            if (t >= tm0 && t < t_end) win[(t - tm0) / R]++;
            seen++;
            continue;
         }
         if (now >= t_end) break;
         k++;
         int i = (int)(k % RING);
         hr_check(ala[i]->Reset(), "CAa::Reset"); hr_check(cla[i]->Reset(ala[i].get(), nullptr), "CLa::Reset");
         ID3D12GraphicsCommandList *ga = cla[i].get(); d12_cl_rec(&ga, 1, &s, k);
         hr_check(cla[i]->Close(), "CLa::Close");
         hr_check(alb[i]->Reset(), "CAb::Reset"); hr_check(clb[i]->Reset(alb[i].get(), nullptr), "CLb::Reset");
         ID3D12GraphicsCommandList *gb = clb[i].get(); d12_cl_rec(&gb, 1, &s, k);
         hr_check(clb[i]->Close(), "CLb::Close");
         ID3D12CommandList *la[1] = {cla[i].get()}, *lb[1] = {clb[i].get()};
         if (k > 1) hr_check(c.queue->Wait(fT.get(), k - 1), "qA::Wait");
         c.queue->ExecuteCommandLists(1, la);
         hr_check(c.queue->Signal(fS.get(), k), "qA::Signal");
         hr_check(q2->Wait(fS.get(), k), "qB::Wait");
         q2->ExecuteCommandLists(1, lb);
         hr_check(q2->Signal(fT.get(), k), "qB::Signal");
      }
      struct gb_samples usr = {};
      for (int i = 0; i < o->runs; i++)
         if (win[i]) gb_samples_push(&usr, (double)o->run_ms * 1e3 / (double)win[i]);

      bool ok = !bad && usr.n > 0;
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2, (const char *const[]){"work=compute.light", nullptr});
      gb_json_obj_begin(&r.j, "executor"); emit_identity(&r.j, c); gb_json_obj_end(&r.j);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad || !usr.n) {
         gb_result_finish(&r, "fail", nullptr, "cross-queue fence ping-pong failed");
      } else {
         gb_result_metric(&r, "us_per_round", "us", true, &usr, true);
         gb_result_finish(&r, "ok", "us_per_round", nullptr);
      }
      gb_samples_free(&usr);
      return ok ? 0 : 1;
   }
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

   d3d12_ctx c;
   d3d12_init(c, o.device);

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
