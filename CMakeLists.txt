# SPDX-License-Identifier: GPL-2.0-only
cmake_minimum_required(VERSION 3.20)
project(gpubench C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Vulkan headers: a Vulkan SDK / distro package, or GB_VULKAN_INCLUDE pointing
# at a Vulkan-Headers include directory. Only the loader is linked.
set(GB_VULKAN_INCLUDE "" CACHE PATH "Directory containing vulkan/vulkan.h")
find_library(GB_VULKAN_LIB NAMES vulkan libvulkan.so.1 REQUIRED)
if(GB_VULKAN_INCLUDE)
  set(GB_VK_INC ${GB_VULKAN_INCLUDE})
else()
  find_path(GB_VK_INC vulkan/vulkan.h)
  if(NOT GB_VK_INC AND EXISTS "${CMAKE_SOURCE_DIR}/rt/third_party/vulkan/vulkan.h")
    set(GB_VK_INC "${CMAKE_SOURCE_DIR}/rt/third_party")
  endif()
  if(NOT GB_VK_INC)
    message(FATAL_ERROR "vulkan/vulkan.h not found; set -DGB_VULKAN_INCLUDE=<Vulkan-Headers>/include")
  endif()
endif()
find_program(GLSLANG glslangValidator REQUIRED)

# Suite version stamped into results.
execute_process(COMMAND git -C ${CMAKE_SOURCE_DIR} describe --always --dirty=+dirty --abbrev=7
                OUTPUT_VARIABLE GB_SUITE_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GB_SUITE_SHA)
  set(GB_SUITE_SHA "unknown")
endif()

include(CTest)
option(GB_BUILD_GL "Build the headless EGL GL/GLES executor" OFF)
add_subdirectory(suite/vk)
if(GB_BUILD_GL)
  add_subdirectory(suite/gl)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/rt/CMakeLists.txt)
  add_subdirectory(rt)
endif()
# Stress lane (Linux only: it uses fork, process groups and the kernel log).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_subdirectory(stress)
endif()

# Stamp the suite revision next to the binaries (read by tools/run-linux.sh).
add_custom_target(suite_sha ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
  COMMAND sh -c "printf '%s\\n' '${GB_SUITE_SHA}' > '${CMAKE_BINARY_DIR}/bin/SUITE_SHA'"
  VERBATIM)
