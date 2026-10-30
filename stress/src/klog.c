// SPDX-License-Identifier: GPL-2.0-only
/* Kernel-log check (--kernel-log, Linux only): collects the kernel messages logged while each
 * test runs and fails the test on lines that look like GPU errors.
 *
 * Source: /dev/kmsg when readable (read from its end), otherwise `journalctl -k` with a
 * cursor. Patterns are POSIX extended regular expressions, matched case-insensitively. */
#include "klog.h"

#include <errno.h>
#include <fcntl.h>
#include <regex.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Generic Linux DRM / GPU error lines. Driver names are the upstream kernel module names. */
static const char *const default_patterns[] = {
   "\\b(gpu|drm)\\b.*(hang|hung|time ?out|timed out|reset|recover|lost|crash|fault|wedged)",
   "\\[drm(:[^]]*)?\\] \\*error\\*",
   "\\b(amdgpu|radeon|i915|xe|nouveau|msm|freedreno|panfrost|panthor|lima|v3d|vc4|etnaviv|powervr|asahi)\\b.*"
   "(error|fail|fault|time ?out|timed out|hang|reset|lost|crash)",
   "firmware.*(crash|hang|time ?out|timed out)",
   "\\b(etimedout|device lost)\\b",
   "\\b(oops|kernel bug|bug:|unable to handle kernel|general protection fault|use-after-free|kasan)",
   "warning: .*(drm|gpu)",
   NULL,
};

struct klog {
   int fd;            /* /dev/kmsg, or -1 */
   char cursor[512];  /* journalctl cursor */
   regex_t *pat, *ign;
   int npat, nign;
};

static int
compile(regex_t **arr, int *n, const char *re, char *err, size_t errlen)
{
   *arr = realloc(*arr, sizeof(regex_t) * (*n + 1));
   int rc = regcomp(&(*arr)[*n], re, REG_EXTENDED | REG_ICASE | REG_NOSUB);
   if (rc) {
      char b[128];
      regerror(rc, &(*arr)[*n], b, sizeof(b));
      snprintf(err, errlen, "bad kernel-log pattern '%s': %s", re, b);
      return -1;
   }
   (*n)++;
   return 0;
}

static int
journal_cursor(char *cur, size_t n)
{
   FILE *p = popen("journalctl -k -n 0 --show-cursor --no-pager -q 2>/dev/null", "r");
   if (!p)
      return -1;
   char line[1024];
   int found = -1;
   while (fgets(line, sizeof(line), p)) {
      char *c = strstr(line, "-- cursor: ");
      if (c) {
         c += strlen("-- cursor: ");
         c[strcspn(c, "\r\n")] = 0;
         snprintf(cur, n, "%s", c);
         found = 0;
      }
   }
   if (pclose(p) != 0)
      return -1;
   return found;
}

struct klog *
klog_open(const struct klog_opts *o, char *err, size_t errlen)
{
   struct klog *k = calloc(1, sizeof(*k));
   k->fd = -1;
   if (!o->no_defaults)
      for (int i = 0; default_patterns[i]; i++)
         if (compile(&k->pat, &k->npat, default_patterns[i], err, errlen))
            goto fail;
   for (int i = 0; i < o->npatterns; i++)
      if (compile(&k->pat, &k->npat, o->patterns[i], err, errlen))
         goto fail;
   for (int i = 0; i < o->nignore; i++)
      if (compile(&k->ign, &k->nign, o->ignore[i], err, errlen))
         goto fail;
   if (!k->npat) {
      snprintf(err, errlen, "no kernel-log patterns");
      goto fail;
   }
   k->fd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK | O_CLOEXEC);
   if (k->fd >= 0) {
      char probe[8192];
      lseek(k->fd, 0, SEEK_END);
      /* dmesg_restrict can allow open but refuse reads. */
      if (read(k->fd, probe, sizeof(probe)) < 0 && errno != EAGAIN) {
         close(k->fd);
         k->fd = -1;
      }
   }
   if (k->fd < 0 && journal_cursor(k->cursor, sizeof(k->cursor))) {
      snprintf(err, errlen,
               "kernel log unavailable: /dev/kmsg is not readable and `journalctl -k` failed "
               "(run as a user who may read the kernel log, e.g. in the adm/wheel/systemd-journal group)");
      goto fail;
   }
   return k;
fail:
   klog_close(k);
   return NULL;
}

const char *
klog_source(const struct klog *k)
{
   return k->fd >= 0 ? "/dev/kmsg" : "journalctl";
}

static void
push(struct klog_lines *l, const char *s, bool expected)
{
   size_t n = strlen(s);
   while (n && (s[n - 1] == '\n' || s[n - 1] == '\r'))
      n--;
   char *c = strndup(s, n);
   if (expected) {
      l->expected++;
      free(c);
      return;
   }
   l->bad = realloc(l->bad, sizeof(char *) * (l->nbad + 1));
   l->bad[l->nbad++] = c;
}

/* Lines a test may never expect, even when they also mention its deliberate faults. */
static regex_t never_expected;
static bool never_compiled;

static bool
is_never_expected(const char *line)
{
   if (!never_compiled) {
      regcomp(&never_expected, "(time ?out|timed out|recover|reset|crash|lost|hang|hung|wedged|oops|bug:)",
              REG_EXTENDED | REG_ICASE | REG_NOSUB);
      never_compiled = true;
   }
   return !regexec(&never_expected, line, 0, NULL, 0);
}

static void
classify(const struct klog *k, const regex_t *expect, const char *line, struct klog_lines *out)
{
   out->total++;
   bool hit = false;
   for (int i = 0; i < k->npat && !hit; i++)
      hit = !regexec(&k->pat[i], line, 0, NULL, 0);
   if (!hit)
      return;
   for (int i = 0; i < k->nign; i++)
      if (!regexec(&k->ign[i], line, 0, NULL, 0)) {
         out->ignored++;
         return;
      }
   push(out, line, expect && !regexec(expect, line, 0, NULL, 0) && !is_never_expected(line));
}

int
klog_collect(struct klog *k, const char *expect_re, struct klog_lines *out)
{
   memset(out, 0, sizeof(*out));
   regex_t expect;
   bool have_expect = expect_re && !regcomp(&expect, expect_re, REG_EXTENDED | REG_ICASE | REG_NOSUB);
   int rc = 0;
   if (k->fd >= 0) {
      char rec[8192];
      for (;;) {
         ssize_t n = read(k->fd, rec, sizeof(rec) - 1);
         if (n < 0) {
            if (errno == EAGAIN)
               break;
            if (errno == EPIPE) /* records overwritten before we read them */
               continue;
            rc = -1;
            break;
         }
         rec[n] = 0;
         /* "prio,seq,usec,flags;message\n continuation..." */
         char *msg = strchr(rec, ';');
         msg = msg ? msg + 1 : rec;
         char *nl = strchr(msg, '\n');
         if (nl)
            *nl = 0;
         classify(k, have_expect ? &expect : NULL, msg, out);
      }
   } else {
      char cmd[700];
      snprintf(cmd, sizeof(cmd), "journalctl -k --after-cursor='%s' --show-cursor --no-pager -q -o cat 2>/dev/null",
               k->cursor);
      FILE *p = popen(cmd, "r");
      if (!p) {
         rc = -1;
      } else {
         char line[8192];
         while (fgets(line, sizeof(line), p)) {
            if (!strncmp(line, "-- cursor: ", 11)) {
               char *c = line + 11;
               c[strcspn(c, "\r\n")] = 0;
               if (strlen(c) < sizeof(k->cursor))
                  memcpy(k->cursor, c, strlen(c) + 1);
               continue;
            }
            classify(k, have_expect ? &expect : NULL, line, out);
         }
         if (pclose(p) != 0)
            rc = -1;
      }
   }
   if (have_expect)
      regfree(&expect);
   return rc;
}

void
klog_lines_free(struct klog_lines *l)
{
   for (int i = 0; i < l->nbad; i++)
      free(l->bad[i]);
   free(l->bad);
   memset(l, 0, sizeof(*l));
}

void
klog_close(struct klog *k)
{
   if (!k)
      return;
   if (k->fd >= 0)
      close(k->fd);
   for (int i = 0; i < k->npat; i++)
      regfree(&k->pat[i]);
   for (int i = 0; i < k->nign; i++)
      regfree(&k->ign[i]);
   free(k->pat);
   free(k->ign);
   free(k);
}

void
klog_print_defaults(FILE *f)
{
   for (int i = 0; default_patterns[i]; i++)
      fprintf(f, "    %s\n", default_patterns[i]);
}
