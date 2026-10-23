# SPDX-License-Identifier: GPL-2.0-only
# Ray-tracing lane: gpubench-rt-vk (Linux, Vulkan). The Metal executor is built on a Mac by
# rt/metal/build.sh. Works standalone (cmake -S rt -B build-rt) or from the top-level CMake.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(gpubench_rt C)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

find_program(GLSLANG glslangValidator REQUIRED)
find_library(VULKAN_LIB vulkan REQUIRED)
if(GB_VULKAN_INCLUDE)
  set(RT_VK_INCLUDE ${GB_VULKAN_INCLUDE})
else()
  set(RT_VK_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

set(RT_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(RT_GEN ${CMAKE_CURRENT_BINARY_DIR}/rtgen)
file(MAKE_DIRECTORY ${RT_GEN})
set(RT_SHADERS deform.comp gbuffer.vert gbuffer.frag rtreflect.comp composite.comp rays.comp
    rtpipe.rgen rtpipe.rchit rtpipe.rahit rtpipe.rmiss rtpipe_shadow.rmiss rtstub.comp)
file(GLOB RT_SHADER_INCLUDES ${RT_SHADER_DIR}/*.glsl)
set(RT_SPV "")
set(RT_EMBED_INPUTS "")
foreach(s ${RT_SHADERS})
  set(out ${RT_GEN}/${s}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} --quiet --target-env vulkan1.3 -V ${RT_SHADER_DIR}/${s} -o ${out}
    DEPENDS ${RT_SHADER_DIR}/${s} ${RT_SHADER_INCLUDES}
    COMMENT "glslang ${s}" VERBATIM)
  list(APPEND RT_SPV ${out})
  list(APPEND RT_EMBED_INPUTS "${s}=${out}")
endforeach()
string(REPLACE ";" "\;" RT_EMBED_ARG "${RT_EMBED_INPUTS}")
add_custom_command(OUTPUT ${RT_GEN}/spv_embed.c
  COMMAND ${CMAKE_COMMAND} -DOUT=${RT_GEN}/spv_embed.c -DMODE=spv "-DINPUTS=${RT_EMBED_ARG}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS ${RT_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
  COMMENT "embedding SPIR-V")
add_custom_command(OUTPUT ${RT_GEN}/presets_ini.c
  COMMAND ${CMAKE_COMMAND} -DOUT=${RT_GEN}/presets_ini.c -DMODE=text -DSYMBOL=rt_presets_ini
          "-DINPUTS=presets=${CMAKE_CURRENT_SOURCE_DIR}/presets/presets.ini"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/presets/presets.ini ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
  COMMENT "embedding presets")

# optional Wayland window (--window)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(WAYLAND QUIET wayland-client)
  pkg_get_variable(WL_PROTOCOLS wayland-protocols pkgdatadir)
endif()
find_program(WAYLAND_SCANNER wayland-scanner)
set(RT_WIN_SRC "")
if(WAYLAND_FOUND AND WAYLAND_SCANNER AND WL_PROTOCOLS)
  set(xdg ${WL_PROTOCOLS}/stable/xdg-shell/xdg-shell.xml)
  add_custom_command(OUTPUT ${RT_GEN}/xdg-shell-client-protocol.h ${RT_GEN}/xdg-shell-protocol.c
    COMMAND ${WAYLAND_SCANNER} client-header ${xdg} ${RT_GEN}/xdg-shell-client-protocol.h
    COMMAND ${WAYLAND_SCANNER} private-code ${xdg} ${RT_GEN}/xdg-shell-protocol.c
    DEPENDS ${xdg} COMMENT "wayland-scanner xdg-shell")
  set(RT_WIN_SRC vulkan/window_vk.c ${RT_GEN}/xdg-shell-protocol.c ${RT_GEN}/xdg-shell-client-protocol.h)
endif()

add_executable(gpubench-rt-vk
  common/params.c common/scene.c common/json.c common/util.c common/result.c
  vulkan/main_vk.c vulkan/vkutil.c vulkan/vkstream.c vulkan/scene_vk.c vulkan/micro_vk.c vulkan/vtrace_vk.c
  ${RT_GEN}/spv_embed.c ${RT_GEN}/presets_ini.c ${RT_WIN_SRC})
set_target_properties(gpubench-rt-vk PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_compile_definitions(gpubench-rt-vk PRIVATE _GNU_SOURCE)
target_include_directories(gpubench-rt-vk PRIVATE ${RT_VK_INCLUDE})
target_compile_options(gpubench-rt-vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
target_link_libraries(gpubench-rt-vk PRIVATE ${VULKAN_LIB} m)
if(RT_WIN_SRC)
  target_compile_definitions(gpubench-rt-vk PRIVATE RT_HAVE_WAYLAND VK_USE_PLATFORM_WAYLAND_KHR)
  target_include_directories(gpubench-rt-vk PRIVATE ${RT_GEN} ${WAYLAND_INCLUDE_DIRS})
  target_link_libraries(gpubench-rt-vk PRIVATE ${WAYLAND_LIBRARIES})
endif()
