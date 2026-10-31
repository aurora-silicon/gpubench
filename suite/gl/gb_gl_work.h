// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include "gb_common.h"
#include "gb_params.h"

enum gb_gl_kind { GB_GL_TRI, GB_GL_CLEAR, GB_GL_FILL, GB_GL_OPAQUE, GB_GL_TEXTURE, GB_GL_GEOM,
                  GB_GL_LIGHT, GB_GL_ALU, GB_GL_BW };
struct gb_gl_work {
   enum gb_gl_kind kind;
   unsigned w, h;
   bool compute;
   const char *metric, *unit;
   double work_e6;
};
bool gb_gl_work(const struct gb_opts *o, struct gb_gl_work *w, const char **error);
