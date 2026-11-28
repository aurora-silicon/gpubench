// SPDX-License-Identifier: GPL-2.0-only
#include "gb_common.h"
#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int action_fails(const char *action)
{
   const char *failure = getenv("GB_CHILD_TEST_FAIL_ACTION");
   return failure && !strcmp(failure, action);
}

int fixture_actions_init(posix_spawn_file_actions_t *actions)
{
   if (action_fails("init"))
      return ENOMEM;
   return posix_spawn_file_actions_init(actions);
}

int fixture_actions_dup2(posix_spawn_file_actions_t *actions, int from, int to)
{
   if (action_fails(to == STDIN_FILENO ? "stdin" : "stdout"))
      return ENOMEM;
   return posix_spawn_file_actions_adddup2(actions, from, to);
}

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
   if (getenv("GB_CHILD_TEST_CLOSE_STDIO")) {
      close(STDIN_FILENO);
      close(STDOUT_FILENO);
   }
   struct gb_opts options = {.argc = argc, .argv = argv, .argv0 = argv[0]};
   struct gb_child children[3] = {0};
   if (gb_spawn_children(&options, 3, children) == 0)
      return 4;
   int ready_exits = 0;
   for (size_t i = 0; i < 3; ++i) {
      if (children[i].pid && gb_child_reap(&children[i]) == 0)
         ++ready_exits;
   }
   const int expected_exits = getenv("GB_CHILD_TEST_FAIL_ACTION") ? 0 : 2;
   return ready_exits == expected_exits ? 0 : 5;
}
