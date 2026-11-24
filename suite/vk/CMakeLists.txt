# SPDX-License-Identifier: GPL-2.0-only
# Shared libraries for every gpubench Vulkan executor.
add_library(gb_common STATIC src/gb_common.c)
target_include_directories(gb_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(gb_common PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(gb_common PUBLIC m pthread)

add_library(gb_vk STATIC src/gb_vk.c)
target_include_directories(gb_vk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${GB_VK_INC})
target_compile_options(gb_vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_link_libraries(gb_vk PUBLIC gb_common ${GB_VULKAN_LIB})

# SPIR-V for the suite's shaders, embedded into the executor.
set(GB_SHADER_DIR ${CMAKE_SOURCE_DIR}/suite/shaders)
set(GB_SHADERS
  light.comp alu.comp bw.comp fs.vert color.frag texture.frag geom.vert tvb.vert
  uber_div.comp uber_uni.comp uber_ilp.comp uber_sram.comp
  conv2d_h.comp conv2d_v.comp stereo.comp
  edge_sobel.comp edge_nms.comp edge_harris_resp.comp edge_harris_nms.comp
  deferred_geom.vert deferred_geom.frag
  deferred_tile.frag deferred_tile_msaa.frag
  deferred_dram.frag deferred_dram_msaa.frag
  mdi.vert mdi.frag
  pso_comp.comp pso_uber.comp pso_gfx.vert pso_gfx.frag
  compress_mrt4.frag compress_write1.frag compress_sample4.frag
  compress_sample1.frag compress_array.frag compress_store.comp
  shadow_depth.vert shadow_alpha.frag shadow_pcf.frag
  cull_mdi.comp cull_mdi.vert cull_mdi.frag
  bubo.vert bubo.frag
  fp16_gemv.comp fp16_alumix.comp
  postfx_scene.vert postfx_scene.frag postfx_down.frag postfx_up.frag postfx_tonemap.frag
  blur_fp32_h.comp blur_fp32_v.comp corner_ring.comp desc_search.comp chain_step.comp msaa_merge.frag)
set(spv_files)
foreach(s IN LISTS GB_SHADERS)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/spv/${s}.spv)
  add_custom_command(OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/spv
    COMMAND ${GLSLANG} -V --target-env vulkan1.3 -o ${out} ${GB_SHADER_DIR}/${s}
    DEPENDS ${GB_SHADER_DIR}/${s} VERBATIM)
  list(APPEND spv_files ${out})
endforeach()
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders.c
  COMMAND ${CMAKE_COMMAND} "-DSPV_FILES=${spv_files}" -DOUT=${CMAKE_CURRENT_BINARY_DIR}/shaders.c
          -P ${GB_SHADER_DIR}/embed.cmake
  DEPENDS ${spv_files} ${GB_SHADER_DIR}/embed.cmake VERBATIM)

add_executable(gpubench-vk src/main.c src/tests.c src/tests_wave1.c src/tests_wave2.c src/tests_wave3.c src/gb_wave1_ref.c src/gb_wave2_ref.c src/gb_wave3_ref.c ${CMAKE_CURRENT_BINARY_DIR}/shaders.c)
target_include_directories(gpubench-vk PRIVATE src)
target_compile_options(gpubench-vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
target_compile_definitions(gpubench-vk PRIVATE GB_SUITE_SHA="${GB_SUITE_SHA}")
target_link_libraries(gpubench-vk PRIVATE gb_vk)
