// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <stdbool.h>
#include <stdio.h>

#define KLOG_MAX_PATTERNS 32

struct klog_opts {
   const char *patterns[KLOG_MAX_PATTERNS];
   int npatterns;
   const char *ignore[KLOG_MAX_PATTERNS];
   int nignore;
   bool no_defaults;
};

struct klog_lines {
   int total;     /* kernel lines in the window */
   int expected;  /* error-like lines the test causes on purpose */
   int ignored;   /* error-like lines matching an ignore pattern */
   char **bad;    /* unexpected error lines */
   int nbad;
};

struct klog;
struct klog *klog_open(const struct klog_opts *o, char *err, size_t errlen);
const char *klog_source(const struct klog *k);
/* Lines logged since the previous call (or klog_open). Returns -1 if the log could not be read. */
int klog_collect(struct klog *k, const char *expect_re, struct klog_lines *out);
void klog_lines_free(struct klog_lines *l);
void klog_close(struct klog *k);
void klog_print_defaults(FILE *f);
