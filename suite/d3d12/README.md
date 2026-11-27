# Direct3D 12 (vkd3d-proton) executor

`gpubench-d3d12` runs the gpubench render, compute, transfer, pipeline-compile,
and synchronization workloads through real Direct3D 12 (`ID3D12Device` /
`ID3D12CommandQueue` / `ID3D12GraphicsCommandList`) using real HLSL Shader
Model 6.0 DXIL bytecode (`suite/shaders/hlsl/` -> `shaders_dxil.h`).

It builds from the same C++17 source for:
- **Linux native (`vkd3d-proton`):** links directly against upstream
  vkd3d-proton's `libvkd3d-proton-d3d12.so` and `libvkd3d-proton-d3d12core.so`
  (zero Wine/FEX overhead, full real `dxil-spirv` shader compiler, descriptor
  heaps, `ExecuteIndirect`, fences, timestamps, and Vulkan submission path).
- **Windows PE (`gpubench-d3d12.exe`):** builds with MinGW-w64 or MSVC against
  standard `d3d12.lib` and `dxgi.lib` so the exact same executable runs under
  Wine/Proton + vkd3d-proton on Linux, macOS GPTK (D3DMetal), or Windows.

## Building

### Linux with upstream vkd3d-proton

Build and install upstream vkd3d-proton
(`https://github.com/HansKristian-Work/vkd3d-proton`) into a prefix:

```sh
meson setup vkd3d-build /path/to/vkd3d-proton \
  --prefix=/path/to/vkd3d-prefix --buildtype=release
ninja -C vkd3d-build && meson install -C vkd3d-build
```

Then configure and build `gpubench-d3d12`:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGB_BUILD_D3D12=ON -DGB_VKD3D_PROTON_PREFIX=/path/to/vkd3d-prefix
ninja -C build gpubench-d3d12
```

If `libvkd3d-proton-d3d12.pc` is already on `PKG_CONFIG_PATH`, omit
`-DGB_VKD3D_PROTON_PREFIX`.

### Windows PE (MinGW-w64 or MSVC)

```sh
cmake -S . -B build-win64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
  -DGB_BUILD_VK=OFF -DGB_BUILD_D3D12=ON
ninja -C build-win64 gpubench-d3d12
```

## Regenerating DXIL shaders

Precompiled SM 6.0 DXIL bytecode is checked into `shaders_dxil.h` so builds do
not require `dxc` at build time. To regenerate after editing
`suite/shaders/hlsl/*.hlsl`:

```sh
python3 tools/compile-hlsl.py --only dxil --dxc /path/to/dxc
```

## Running

Pass `LD_LIBRARY_PATH` (or `GB_D3D_LIB_DIR`) pointing at
`libvkd3d-proton-d3d12.so` and `libvkd3d-proton-d3d12core.so`, and optionally
set `VK_DRIVER_FILES` to select an isolated Mesa/Vulkan ICD:

```sh
LD_LIBRARY_PATH=/path/to/vkd3d-prefix/lib \
VK_DRIVER_FILES=/path/to/mesa/share/vulkan/icd.d/asahi_icd.aarch64.json \
  build/bin/gpubench-d3d12 --info

LD_LIBRARY_PATH=/path/to/vkd3d-prefix/lib \
  build/bin/gpubench-d3d12 --test render.deferred --param mode=split \
  --param res=1080p --param msaa=1 --depths 1,2 --verify
```

Use `tools/run-d3d.sh <out-dir> d3d12` (or `all`) and `tools/compare.py` to
compare vkd3d-proton directly against `gpubench-vk` (the runner automatically
sets `MESA_SHADER_CACHE_DISABLE=true VKD3D_SHADER_CACHE_PATH=0` for
`compile.pso`):

```sh
GB_D3D_LIB_DIR=/path/to/vkd3d-prefix/lib tools/run-d3d.sh run-out all
python3 tools/compare.py run-out/d3d12 run-out/vk --labels vkd3d-proton,VK
```

## Supported tests

`compute.light`, `compute.heavy` (`alu`, `bw`), `compute.conv2d` (`gaussian`,
`bilateral`), `render.light` (`tri`, `clear`), `render.heavy` (`fill`, `opaque`,
`texture`, `geom`, `msaa=1|4`), `render.draws` (`direct` CBV ring churn,
`instanced`, `indirect` via `ExecuteIndirect`), `render.deferred` (`1080p`,
`msaa=1|4`, plus `mode=split` / `split=1` multi-command-list split G-buffer
pass), `render.shadow` (`depth_only`, `alpha_test`), `render.postfx` (`1440p`,
`msaa=1|4`), `transfer.ops` (`clear_4m`, `clear_16m`, `copy_b2i`, `mipchain`),
`compile.pso` (`compute`, `graphics`), and `sync.signal` (`host`, `queue`).
Every test supports `--verify` for deterministic readback CRC checking and
emits `gpubench-result/1` JSONL with untimed and GPU-timestamp passes.
