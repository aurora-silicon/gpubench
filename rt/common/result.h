// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-result/1 lines (records/SCHEMA.md), shared by both executors. */
#pragma once
#include "json.h"
#include "params.h"

/* Opens the result object: schema, lane, test, params, mode, depth. */
void rt_result_begin(rt_json *j, int test, const rt_params *p, const char *mode, uint32_t depth);
/* Adds the run controls. */
void rt_result_windows(rt_json *j, const rt_run *r);
/* Closes and prints the object as one line on stdout. */
void rt_result_print(rt_json *j);
/* Prints a complete skip / fail line. */
void rt_result_skip(int test, const rt_params *p, const char *mode, uint32_t depth, const char *reason);
void rt_result_fail(int test, const rt_params *p, const char *mode, uint32_t depth, const char *error);
/* Test names for --list. */
void rt_print_list(void);
