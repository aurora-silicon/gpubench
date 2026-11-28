// SPDX-License-Identifier: GPL-2.0-only
#include "gb_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   const char *state = getenv("GB_CHILD_TEST_STATE");
   if (argc > 1 && !strcmp(argv[argc - 1], "--child")) {
      int count = 0;
      FILE *file = fopen(state, "r+");
      if (!file || fscanf(file, "%d", &count) != 1)
         return 3;
      rewind(file);
      fprintf(file, "%d\n", count + 1);
      fclose(file);
      if (count == 2)
         return 2;
      gb_child_ready();
      return gb_child_recv_u64() != 0;
   }

   FILE *file = fopen(state, "w");
   if (!file)
      return 3;
   fputs("0\n", file);
   fclose(file);
   struct gb_opts options = {.argc = argc, .argv = argv, .argv0 = argv[0]};
   struct gb_child children[3] = {0};
   if (gb_spawn_children(&options, 3, children) == 0)
      return 4;
   int ready_exits = 0;
   for (size_t i = 0; i < 3; ++i) {
      if (children[i].pid && gb_child_reap(&children[i]) == 0)
         ++ready_exits;
   }
   return ready_exits == 2 ? 0 : 5;
}
