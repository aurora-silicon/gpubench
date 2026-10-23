// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <stddef.h>
#include <stdint.h>

#define GB_SHADER(n) extern const uint32_t n[]; extern const size_t n##_size;
GB_SHADER(light_comp)
GB_SHADER(alu_comp)
GB_SHADER(bw_comp)
GB_SHADER(fs_vert)
GB_SHADER(color_frag)
GB_SHADER(geom_vert)
GB_SHADER(tvb_vert)
#undef GB_SHADER

#define GB_SPV(n) n, n##_size
