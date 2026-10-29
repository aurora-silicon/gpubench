# SPDX-License-Identifier: GPL-2.0-only
# Stress lane: gpubench-stress-vk (Linux, Vulkan). Built from the top-level CMake; reuses the
# suite's gb_common (timing, JSON) and its SPIR-V embedding script.
set(ST_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(ST_SHADERS chain.comp order.comp lcg.comp link.comp fault.comp rq.comp
    fill.vert rows.vert grid.vert color.frag xdata.frag)
set(st_spv)
foreach(s IN LISTS ST_SHADERS)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/${s}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
    COMMAND ${GLSLANG} -V --target-env vulkan1.3 -o ${out} ${ST_SHADER_DIR}/${s}
    DEPENDS ${ST_SHADER_DIR}/${s} VERBATIM)
  list(APPEND st_spv ${out})
endforeach()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/st_shaders.c
  COMMAND ${CMAKE_COMMAND} "-DSPV_FILES=${st_spv}" -DOUT=${CMAKE_CURRENT_BINARY_DIR}/st_shaders.c
          -P ${CMAKE_SOURCE_DIR}/suite/shaders/embed.cmake
  DEPENDS ${st_spv} ${CMAKE_SOURCE_DIR}/suite/shaders/embed.cmake VERBATIM)

add_executable(gpubench-stress-vk
  src/main.c src/klog.c src/st_vk.c src/st_work.c src/st_client.c src/st_ring.c
  src/t_xqueue.c src/t_render.c src/t_compute.c src/t_life.c src/t_mem.c src/t_sync.c
  src/t_queue.c src/t_sparse.c src/t_rq.c src/t_mix.c
  ${CMAKE_CURRENT_BINARY_DIR}/st_shaders.c)
target_include_directories(gpubench-stress-vk PRIVATE src ${GB_VK_INC})
target_compile_definitions(gpubench-stress-vk PRIVATE _GNU_SOURCE GB_SUITE_SHA="${GB_SUITE_SHA}")
target_compile_options(gpubench-stress-vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(gpubench-stress-vk PRIVATE gb_common ${GB_VULKAN_LIB})
