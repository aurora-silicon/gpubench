# Direct3D 11 (DXVK) executor

`gpubench-d3d11` runs the gpubench render, compute, transfer, pipeline-compile,
and synchronization workloads through real Direct3D 11 (`ID3D11Device` /
`ID3D11DeviceContext`) using real HLSL Shader Model 5.0 DXBC bytecode
(`suite/shaders/hlsl/` -> `shaders_dxbc.h`).

It builds from the same C++17 source for:
- **Linux native (`dxvk-native`):** links directly against upstream DXVK's
  `libdxvk_d3d11.so` and `libdxvk_dxgi.so` (zero Wine/FEX overhead, full real
  DXVK DXBC->SPIR-V compiler, state cache, GPL, and Vulkan submission path).
- **Windows PE (`gpubench-d3d11.exe`):** builds with MinGW-w64 or MSVC against
  standard `d3d11.lib` and `dxgi.lib` so the exact same executable runs under
  Wine/Proton + DXVK on Linux, macOS GPTK (D3DMetal), or Windows.

## Building

### Linux with upstream DXVK (`dxvk-native`)

Build and install upstream DXVK (`https://github.com/doitsujin/dxvk`) into a
prefix (keeping only Wayland WSI enabled; `gpubench-d3d11` defaults
`DXVK_WSI_DRIVER=Wayland` and runs completely headless without a display server):

```sh
meson setup dxvk-build /path/to/dxvk --prefix=/path/to/dxvk-prefix \
  --buildtype=release -Denable_d3d8=false -Denable_d3d9=false \
  -Denable_d3d10=false -Dnative_sdl2=disabled -Dnative_sdl3=disabled \
  -Dnative_glfw=disabled -Dnative_xcb=disabled -Dnative_wayland=enabled
ninja -C dxvk-build && meson install -C dxvk-build
```

Then configure and build `gpubench-d3d11`:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGB_BUILD_D3D11=ON -DGB_DXVK_PREFIX=/path/to/dxvk-prefix
ninja -C build gpubench-d3d11
```

If `dxvk-d3d11.pc` and `dxvk-dxgi.pc` are already on `PKG_CONFIG_PATH`, omit
`-DGB_DXVK_PREFIX`.

### Windows PE (MinGW-w64 or MSVC)

```sh
cmake -S . -B build-win64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
  -DGB_BUILD_VK=OFF -DGB_BUILD_D3D11=ON
ninja -C build-win64 gpubench-d3d11
```

## Regenerating DXBC shaders

Precompiled SM 5.0 DXBC bytecode is checked into `shaders_dxbc.h` so builds do
not require an HLSL compiler at build time. To regenerate after editing
`suite/shaders/hlsl/*.hlsl`:

```sh
python3 tools/compile-hlsl.py --only dxbc --vkd3d-compiler /path/to/vkd3d-compiler
```

## Running

Pass `LD_LIBRARY_PATH` (or `GB_D3D_LIB_DIR`) pointing at `libdxvk_d3d11.so` and
`libdxvk_dxgi.so`, and optionally set `VK_DRIVER_FILES` to select an isolated
Mesa/Vulkan ICD:

```sh
LD_LIBRARY_PATH=/path/to/dxvk-prefix/lib \
VK_DRIVER_FILES=/path/to/mesa/share/vulkan/icd.d/asahi_icd.aarch64.json \
  build/bin/gpubench-d3d11 --info

LD_LIBRARY_PATH=/path/to/dxvk-prefix/lib \
  build/bin/gpubench-d3d11 --test render.deferred --param mode=dram \
  --param res=1080p --param msaa=4 --depths 1,2 --verify
```

Use `tools/run-d3d.sh <out-dir> d3d11` (or `all`) and `tools/compare.py` to
compare DXVK directly against `gpubench-vk` (the runner automatically sets
`MESA_SHADER_CACHE_DISABLE=true DXVK_STATE_CACHE=0` for `compile.pso`):

```sh
GB_D3D_LIB_DIR=/path/to/dxvk-prefix/lib tools/run-d3d.sh run-out all
python3 tools/compare.py run-out/d3d11 run-out/vk --labels DXVK,VK
```

## Supported tests

`compute.light`, `compute.heavy` (`alu`, `bw`), `compute.conv2d` (`gaussian`,
`bilateral`), `render.light` (`tri`, `clear`), `render.heavy` (`fill`, `opaque`,
`texture`, `geom`, `msaa=1|4`), `render.draws` (`direct` CB discard churn,
`instanced`, `indirect`), `render.deferred` (`1080p`, `msaa=1|4`),
`render.shadow` (`depth_only`, `alpha_test`), `render.postfx` (`1440p`,
`msaa=1|4`), `transfer.ops` (`clear_4m`, `clear_16m`, `copy_b2i`, `mipchain`),
`compile.pso` (`compute`, `graphics`), and `sync.signal` (`host`/`query`).
Every test supports `--verify` for deterministic readback CRC checking and
emits `gpubench-result/1` JSONL with untimed and GPU-timestamp passes.
