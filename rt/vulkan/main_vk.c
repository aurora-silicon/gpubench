// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-rt-vk: the Vulkan executor of the ray-tracing lane (see rt/README.md). */
#include "../common/result.h"
#include "rtvk.h"
#include <stdlib.h>
#include <string.h>

PFN_vkCmdTraceRaysKHR rt_vk_trace_rays;

int main(int argc, char **argv)
{
   rt_params p;
   rt_run r;
   int pr = rt_parse(&p, &r, argc, argv, "gpubench-rt-vk");
   if (pr) return pr > 0 ? 0 : 2;
   if (r.list) { rt_print_list(); return 0; }
   if (r.info) { vk_print_info(r.device); return 0; }
   rt_run_defaults(&r);
   vk_ctx c;
   int want_rtp = r.test == RT_TEST_SCENE && p.trace_mode == RT_TRACE_PIPELINE;
   if (vk_ctx_init(&c, r.device, want_rtp, r.shader_stats, r.window && r.test == RT_TEST_SCENE) != VK_SUCCESS) {
      fprintf(stderr, "Vulkan device setup failed\n");
      return 1;
   }
   rt_vk_trace_rays = c.CmdTraceRays;
   vk_pipeline_markers = r.shader_stats;
   fprintf(stderr, "device: %s (%s)\n", c.props.deviceName, c.driver.driverInfo);
   int rc;
   switch (r.test) {
   case RT_TEST_ASBUILD: rc = rt_vk_asbuild(&c, &p, &r); break;
   case RT_TEST_RAYS: rc = rt_vk_rays(&c, &p, &r); break;
   case RT_TEST_TRAV: rc = rt_vk_trav(&c, &p, &r); break;
   default: rc = rt_vk_scene(&c, &p, &r); break;
   }
   vk_ctx_destroy(&c);
   return rc;
}
