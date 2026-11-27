# SPDX-License-Identifier: GPL-2.0-only
enable_language(CXX)

if(WIN32)
  set(GB_D3D12_LIBS d3d12 dxgi)
  set(GB_D3D12_INCS)
else()
  set(GB_VKD3D_PROTON_PREFIX "" CACHE PATH "Install prefix of vkd3d-proton (containing include/vkd3d-proton and lib/libvkd3d-proton-d3d12.so)")
  if(GB_VKD3D_PROTON_PREFIX)
    set(GB_D3D12_INCS "${GB_VKD3D_PROTON_PREFIX}/include/vkd3d-proton")
    find_library(GB_VKD3D_D3D12_LIB NAMES vkd3d-proton-d3d12 PATHS "${GB_VKD3D_PROTON_PREFIX}/lib" "${GB_VKD3D_PROTON_PREFIX}/lib64" NO_DEFAULT_PATH REQUIRED)
    set(GB_D3D12_LIBS "${GB_VKD3D_D3D12_LIB}")
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(VKD3D_PROTON REQUIRED IMPORTED_TARGET libvkd3d-proton-d3d12)
    set(GB_D3D12_LIBS PkgConfig::VKD3D_PROTON)
    set(GB_D3D12_INCS)
  endif()
endif()

add_executable(gpubench-d3d12
  main.cpp
  ../vk/src/gb_wave1_ref.c
  ../vk/src/gb_wave2_ref.c)
set_target_properties(gpubench-d3d12 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(gpubench-d3d12 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/suite/d3d_common
  ${CMAKE_SOURCE_DIR}/suite/vk/src
  ${GB_D3D12_INCS})
target_compile_definitions(gpubench-d3d12 PRIVATE GB_SUITE_SHA="${GB_SUITE_SHA}")
if(NOT MSVC)
  target_compile_options(gpubench-d3d12 PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas)
endif()
target_link_libraries(gpubench-d3d12 PRIVATE gb_common ${GB_D3D12_LIBS})
