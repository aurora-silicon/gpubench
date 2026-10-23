// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include "gb_common.h"

struct gb_test {
   const char *name;
   int (*run)(const struct gb_opts *o);
};
extern const struct gb_test gb_tests[];
/* Builds every suite pipeline and prints VK_KHR_pipeline_executable_properties
 * statistics as JSON lines (used for suite/FAIRNESS.md). */
int gb_shader_stats(const struct gb_opts *o);
