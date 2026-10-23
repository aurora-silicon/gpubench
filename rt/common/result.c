// SPDX-License-Identifier: GPL-2.0-only
#include "result.h"
#include <stdio.h>

void rt_result_begin(rt_json *j, int test, const rt_params *p, const char *mode, uint32_t depth)
{
   rt_json_init(j);
   rt_json_obj(j, NULL);
   rt_json_str(j, "schema", "gpubench-result/1");
   rt_json_str(j, "lane", "rt");
   rt_json_str(j, "test", rt_test_name(test));
   rt_params_json(j, p, test);
   rt_json_str(j, "mode", mode);
   rt_json_int(j, "depth", depth);
}

void rt_result_windows(rt_json *j, const rt_run *r)
{
   rt_json_int(j, "warmup_ms", r->warmup_ms);
   rt_json_int(j, "runs", r->runs);
   rt_json_int(j, "run_ms", r->frames ? 0 : r->run_ms);
   if (r->frames) rt_json_int(j, "run_submits", r->frames);
   if (r->tag[0]) rt_json_str(j, "tag", r->tag);
   if (r->stamps_frame) rt_json_str(j, "stamps", "frame");
}

void rt_result_print(rt_json *j)
{
   rt_json_end(j);
   printf("%s\n", j->buf);
   fflush(stdout);
   rt_json_free(j);
}

void rt_result_skip(int test, const rt_params *p, const char *mode, uint32_t depth, const char *reason)
{
   rt_json j;
   rt_result_begin(&j, test, p, mode, depth);
   rt_json_str(&j, "status", "skip");
   rt_json_str(&j, "reason", reason);
   rt_result_print(&j);
}

void rt_result_fail(int test, const rt_params *p, const char *mode, uint32_t depth, const char *error)
{
   rt_json j;
   rt_result_begin(&j, test, p, mode, depth);
   rt_json_str(&j, "status", "fail");
   rt_json_str(&j, "error", error);
   rt_result_print(&j);
}

void rt_print_list(void)
{
   for (int t = 0; t < RT_TEST_COUNT; t++) printf("%s\n", rt_test_name(t));
}
