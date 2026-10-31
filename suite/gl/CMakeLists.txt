# SPDX-License-Identifier: GPL-2.0-only
find_package(PkgConfig REQUIRED)
pkg_check_modules(EPOXY REQUIRED IMPORTED_TARGET epoxy)
set(GB_GL_SHADER_NAMES light.comp alu.comp bw.comp fs.vert color.frag geom.vert texture.frag)
set(inputs)
foreach(name IN LISTS GB_GL_SHADER_NAMES)
  list(APPEND inputs ${CMAKE_SOURCE_DIR}/suite/shaders/${name})
endforeach()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders.h
  COMMAND ${CMAKE_COMMAND} -DSHADERS=${CMAKE_SOURCE_DIR}/suite/shaders
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/shaders.h -P ${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake
  DEPENDS ${inputs} embed.cmake VERBATIM)
add_library(gb_gl_work STATIC gb_gl_work.c)
target_link_libraries(gb_gl_work PUBLIC gb_common)
target_include_directories(gb_gl_work PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(gpubench-gl main.c ${CMAKE_CURRENT_BINARY_DIR}/shaders.h)
target_include_directories(gpubench-gl PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(gpubench-gl PRIVATE GB_SUITE_SHA="${GB_SUITE_SHA}")
target_compile_options(gpubench-gl PRIVATE -Wall -Wextra -Werror)
target_link_libraries(gpubench-gl PRIVATE gb_gl_work PkgConfig::EPOXY)
if(BUILD_TESTING)
  add_executable(test-gl-work tests/work.c)
  target_link_libraries(test-gl-work PRIVATE gb_gl_work)
  add_test(NAME gl-work COMMAND test-gl-work)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME gl-llvmpipe COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py $<TARGET_FILE:gpubench-gl> $<TARGET_FILE:gpubench-vk>)
  set_tests_properties(gl-llvmpipe PROPERTIES TIMEOUT 180 LABELS smoke)
endif()
