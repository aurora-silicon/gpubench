# SPDX-License-Identifier: GPL-2.0-only
enable_language(CXX)

if(WIN32)
  set(GB_D3D11_LIBS d3d11 dxgi)
  set(GB_D3D11_INCS)
else()
  set(GB_DXVK_PREFIX "" CACHE PATH "Install prefix of dxvk-native (containing include/dxvk and lib/libdxvk_d3d11.so)")
  if(GB_DXVK_PREFIX)
    set(GB_D3D11_INCS "${GB_DXVK_PREFIX}/include/dxvk")
    find_library(GB_DXVK_D3D11_LIB NAMES dxvk_d3d11 PATHS "${GB_DXVK_PREFIX}/lib" "${GB_DXVK_PREFIX}/lib64" NO_DEFAULT_PATH REQUIRED)
    find_library(GB_DXVK_DXGI_LIB NAMES dxvk_dxgi PATHS "${GB_DXVK_PREFIX}/lib" "${GB_DXVK_PREFIX}/lib64" NO_DEFAULT_PATH REQUIRED)
    set(GB_D3D11_LIBS "${GB_DXVK_D3D11_LIB}" "${GB_DXVK_DXGI_LIB}")
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DXVK_D3D11 REQUIRED IMPORTED_TARGET dxvk-d3d11 dxvk-dxgi)
    set(GB_D3D11_LIBS PkgConfig::DXVK_D3D11)
    set(GB_D3D11_INCS)
  endif()
endif()

add_executable(gpubench-d3d11
  main.cpp
  ../vk/src/gb_wave1_ref.c
  ../vk/src/gb_wave2_ref.c)
set_target_properties(gpubench-d3d11 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(gpubench-d3d11 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/suite/d3d_common
  ${CMAKE_SOURCE_DIR}/suite/vk/src
  ${GB_D3D11_INCS})
target_compile_definitions(gpubench-d3d11 PRIVATE GB_SUITE_SHA="${GB_SUITE_SHA}")
if(NOT MSVC)
  target_compile_options(gpubench-d3d11 PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unknown-pragmas)
endif()
target_link_libraries(gpubench-d3d11 PRIVATE gb_common ${GB_D3D11_LIBS})
