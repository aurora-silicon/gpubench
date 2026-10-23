// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-vk: the standard suite's Vulkan executor. See suite/README.md. */
#include <stdlib.h>
#include <string.h>

#include "gb_vk.h"
#include "tests.h"

int
main(int argc, char **argv)
{
   struct gb_opts o;
   bool stats = false;
   for (int i = 1; i < argc; i++)
      if (!strcmp(argv[i], "--shader-stats")) {
         stats = true;
         for (int j = i; j < argc - 1; j++)
            argv[j] = argv[j + 1];
         argc--;
         break;
      }
   gb_parse_opts(&o, argc, argv);
   const char *dev = getenv("GB_DEVICE");
   if (dev && !o.device)
      o.device = atoi(dev);
   if (stats)
      return gb_shader_stats(&o);
   if (o.info)
      return gb_vk_print_info(o.device);
   if (o.list) {
      for (const struct gb_test *t = gb_tests; t->name; t++)
         puts(t->name);
      return 0;
   }
   if (!o.test)
      gb_die("need --test NAME, --list or --info (see --help)");
   for (const struct gb_test *t = gb_tests; t->name; t++)
      if (!strcmp(t->name, o.test))
         return t->run(&o);
   gb_die("unknown test %s", o.test);
}
