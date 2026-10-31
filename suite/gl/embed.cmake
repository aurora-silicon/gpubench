# SPDX-License-Identifier: GPL-2.0-only
# Adapt only API spelling; algorithms remain in the suite's shared GLSL files.
file(WRITE "${OUT}" "/* SPDX-License-Identifier: GPL-2.0-only */\n")
foreach(name light.comp alu.comp bw.comp fs.vert color.frag geom.vert texture.frag)
  file(READ "${SHADERS}/${name}" code)
  string(REPLACE "#version 450" "" code "${code}")
  string(REPLACE "gl_VertexIndex" "gl_VertexID" code "${code}")
  string(REPLACE "layout(set = 0, binding = " "layout(binding = " code "${code}")
  string(REPLACE "layout(push_constant) uniform PC" "struct PC" code "${code}")
  string(REPLACE "} pc;" "}; uniform PC pc;" code "${code}")
  # Explicit storage layout is required by GLES.
  string(REPLACE "layout(binding = " "layout(std430, binding = " code "${code}")
  string(REPLACE "layout(std430, binding = 0) uniform sampler2D" "uniform sampler2D" code "${code}")
  string(REPLACE "\\" "\\\\" code "${code}")
  string(REPLACE "\"" "\\\"" code "${code}")
  string(REPLACE "\n" "\\n\"\n\"" code "${code}")
  string(REPLACE "." "_" symbol "${name}")
  file(APPEND "${OUT}" "static const char ${symbol}[] = \"${code}\";\n")
endforeach()
