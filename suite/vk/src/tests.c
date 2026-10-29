// SPDX-License-Identifier: GPL-2.0-only
/* The standard suite's tests (Vulkan). The Metal executor mirrors each one;
 * see suite/README.md for what they measure. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "gb_vk.h"
#include "shaders.h"
#include "tests.h"
#include "gb_params.h"

static const VkFormat RT_FMT = VK_FORMAT_R8G8B8A8_UNORM;
static const float CLEAR[4] = {0.1f, 0.2f, 0.3f, 1.0f};

#define DEPTHS_ALL_INIT {1, 2, 4, 8, 16, 32, 64}, 7

static bool
parse_res(const char *s, uint32_t *w, uint32_t *h)
{
   if (!strcmp(s, "720p")) *w = 1280, *h = 720;
   else if (!strcmp(s, "1080p")) *w = 1920, *h = 1080;
   else if (!strcmp(s, "1440p")) *w = 2560, *h = 1440;
   else if (!strcmp(s, "2160p")) *w = 3840, *h = 2160;
   else return false;
   return true;
}

static void
set_err(char *err, size_t n, const char *msg)
{
   snprintf(err, n, "%s", msg);
}

static void
make_target(struct gb_vk *vk, uint32_t w, uint32_t h, struct gb_img *im)
{
   gb_img_create(vk, w, h, RT_FMT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, im);
   VkCommandBuffer cb = gb_oneshot_begin(vk);
   gb_img_barrier(cb, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
   gb_oneshot_end(vk, cb);
}

/* ======================= submit.empty ======================= */

static void *
empty_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   (void)vk; (void)o; (void)err; (void)n;
   return calloc(1, 1);
}

static void
generic_teardown(struct gb_vk *vk, void *st)
{
   (void)vk;
   free(st);
}

static void
empty_record(void *st, VkCommandBuffer cb, uint64_t k)
{
   (void)st; (void)cb; (void)k;
}

static const char *const no_params[] = {NULL};

static const struct gb_stream_test t_submit_empty = {
   .name = "submit.empty",
   .known_params = no_params,
   .default_depths = DEPTHS_ALL_INIT,
   .setup = empty_setup,
   .teardown = generic_teardown,
   .record = empty_record,
   .no_timed = true,
};

/* ======================= compute.light ======================= */

struct cl_state {
   struct gb_pipe p;
   struct gb_buf out;
};

static void *
cl_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   (void)o; (void)err; (void)n;
   struct cl_state *st = calloc(1, sizeof(*st));
   gb_buf_create(vk, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, &st->out);
   gb_compute_pipe(vk, GB_SPV(light_comp), 1, 4, &st->p);
   struct gb_buf *b[1] = {&st->out};
   gb_pipe_set_buffers(vk, &st->p, b, 1);
   return st;
}

static void
cl_teardown(struct gb_vk *vk, void *p)
{
   struct cl_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->out);
   free(st);
}

static void
cl_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct cl_state *st = p;
   uint32_t v = (uint32_t)k;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &v);
   vkCmdDispatch(cb, 1, 1, 1);
}

static const struct gb_stream_test t_compute_light = {
   .name = "compute.light",
   .known_params = no_params,
   .default_depths = DEPTHS_ALL_INIT,
   .setup = cl_setup,
   .teardown = cl_teardown,
   .record = cl_record,
};

/* ======================= compute.heavy ======================= */

struct ch_state {
   bool bw;
   struct gb_pipe p;
   struct gb_buf a, b;
};

static void *
ch_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "alu");
   struct ch_state *st = calloc(1, sizeof(*st));
   if (!strcmp(kind, "alu")) {
      gb_buf_create(vk, GB_ALU_GROUPS * GB_ALU_LOCAL * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->a);
      gb_compute_pipe(vk, GB_SPV(alu_comp), 1, 12, &st->p);
      struct gb_buf *b[1] = {&st->a};
      gb_pipe_set_buffers(vk, &st->p, b, 1);
   } else if (!strcmp(kind, "bw")) {
      st->bw = true;
      gb_buf_create(vk, GB_BW_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->a);
      gb_buf_create(vk, GB_BW_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, &st->b);
      gb_compute_pipe(vk, GB_SPV(bw_comp), 2, 0, &st->p);
      struct gb_buf *b[2] = {&st->a, &st->b};
      gb_pipe_set_buffers(vk, &st->p, b, 2);
   } else {
      free(st);
      set_err(err, n, "kind must be alu or bw");
      return NULL;
   }
   return st;
}

static void
ch_teardown(struct gb_vk *vk, void *p)
{
   struct ch_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_buf_destroy(vk, &st->a);
   if (st->bw)
      gb_buf_destroy(vk, &st->b);
   free(st);
}

static void
ch_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct ch_state *st = p;
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st->p.layout, 0, 1, &st->p.set, 0, NULL);
   if (st->bw) {
      vkCmdDispatch(cb, (uint32_t)(GB_BW_BYTES / 16 / GB_BW_LOCAL), 1, 1);
   } else {
      struct { uint32_t iters; float a, b; } pc = {GB_ALU_ITERS, GB_ALU_A, GB_ALU_B};
      vkCmdPushConstants(cb, st->p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, GB_ALU_GROUPS, 1, 1);
   }
}

static void
ch_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   if (!strcmp(gb_param(o, "kind", "alu"), "bw"))
      gb_derive_rate(r, "gbps", "GB/s", us, 2.0 * GB_BW_BYTES / 1e9 * 1e6);
   else
      gb_derive_rate(r, "gflops", "GFLOP/s", us, GB_ALU_FLOP / 1e9 * 1e6);
}

static const char *const kind_params[] = {"kind", NULL};

static const struct gb_stream_test t_compute_heavy = {
   .name = "compute.heavy",
   .known_params = kind_params,
   .default_depths = DEPTHS_ALL_INIT,
   .setup = ch_setup,
   .teardown = ch_teardown,
   .record = ch_record,
   .derive = ch_derive,
};

/* ======================= render.light / render.heavy / render.tvb ======================= */

enum r_kind { R_TRI, R_CLEAR, R_FILL, R_GEOM, R_TVB };

struct r_state {
   enum r_kind kind;
   struct gb_pipe p;
   struct gb_img img;
   uint32_t tris;
};

static void *
r_setup_kind(struct gb_vk *vk, enum r_kind kind, uint32_t w, uint32_t h, uint32_t tris)
{
   struct r_state *st = calloc(1, sizeof(*st));
   st->kind = kind;
   st->tris = tris;
   make_target(vk, w, h, &st->img);
   switch (kind) {
   case R_TRI:
      gb_graphics_pipe(vk, GB_SPV(fs_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &st->p);
      break;
   case R_FILL:
      gb_graphics_pipe(vk, GB_SPV(fs_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_ALPHA, 16, &st->p);
      break;
   case R_GEOM:
      gb_graphics_pipe(vk, GB_SPV(geom_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &st->p);
      break;
   case R_TVB:
      gb_graphics_pipe(vk, GB_SPV(tvb_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &st->p);
      break;
   case R_CLEAR:
      break;
   }
   return st;
}

static void
r_teardown(struct gb_vk *vk, void *p)
{
   struct r_state *st = p;
   gb_pipe_destroy(vk, &st->p);
   gb_img_destroy(vk, &st->img);
   free(st);
}

static void
r_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct r_state *st = p;
   float color[4] = {(float)(k & 255) / 255.0f, 0.5f, 0.25f, 0.5f};
   gb_begin_render(cb, &st->img, CLEAR);
   if (st->kind != R_CLEAR) {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, st->p.pipe);
      vkCmdPushConstants(cb, st->p.layout, st->p.push_stages, 0, 16, color);
      switch (st->kind) {
      case R_TRI: vkCmdDraw(cb, 3, 1, 0, 0); break;
      case R_FILL: vkCmdDraw(cb, 3, GB_FILL_LAYERS, 0, 0); break;
      case R_GEOM: vkCmdDraw(cb, 3 * GB_GEOM_TRIS, 1, 0, 0); break;
      case R_TVB: vkCmdDraw(cb, 3 * st->tris, 1, 0, 0); break;
      default: break;
      }
   }
   vkCmdEndRendering(cb);
}

static void *
rl_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "tri");
   if (!strcmp(kind, "tri"))
      return r_setup_kind(vk, R_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
   if (!strcmp(kind, "clear"))
      return r_setup_kind(vk, R_CLEAR, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
   set_err(err, n, "kind must be tri or clear");
   return NULL;
}

static const struct gb_stream_test t_render_light = {
   .name = "render.light",
   .known_params = kind_params,
   .default_depths = DEPTHS_ALL_INIT,
   .setup = rl_setup,
   .teardown = r_teardown,
   .record = r_record,
};

static void *
rh_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "fill");
   uint32_t w, h;
   if (!parse_res(gb_param(o, "res", "1080p"), &w, &h)) {
      set_err(err, n, "res must be 720p, 1080p, 1440p or 2160p");
      return NULL;
   }
   if (!strcmp(kind, "fill"))
      return r_setup_kind(vk, R_FILL, w, h, 0);
   if (!strcmp(kind, "geom"))
      return r_setup_kind(vk, R_GEOM, w, h, 0);
   set_err(err, n, "kind must be fill or geom");
   return NULL;
}

static void
rh_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   uint32_t w = 1920, h = 1080;
   parse_res(gb_param(o, "res", "1080p"), &w, &h);
   if (!strcmp(gb_param(o, "kind", "fill"), "geom"))
      gb_derive_rate(r, "mtri_s", "Mtri/s", us, GB_GEOM_TRIS / 1e6 * 1e6);
   else
      gb_derive_rate(r, "gpix_s", "Gpix/s", us, (double)w * h * GB_FILL_LAYERS / 1e9 * 1e6);
}

static const char *const rh_params[] = {"kind", "res", NULL};

static const struct gb_stream_test t_render_heavy = {
   .name = "render.heavy",
   .known_params = rh_params,
   .default_depths = DEPTHS_ALL_INIT,
   .setup = rh_setup,
   .teardown = r_teardown,
   .record = r_record,
   .derive = rh_derive,
};

static void *
tvb_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long tris = gb_param_int(o, "tris", 262144);
   if (tris < 1 || tris > (16 << 20)) {
      set_err(err, n, "tris out of range");
      return NULL;
   }
   /* dim=N renders into an N x N target instead (tiling load scales with tris, fragment work stays small). */
   long dim = gb_param_int(o, "dim", 0);
   if (dim < 0 || dim > 4096) {
      set_err(err, n, "dim out of range");
      return NULL;
   }
   if (dim)
      return r_setup_kind(vk, R_TVB, (uint32_t)dim, (uint32_t)dim, (uint32_t)tris);
   return r_setup_kind(vk, R_TVB, GB_TVB_W, GB_TVB_H, (uint32_t)tris);
}

static void
tvb_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   gb_derive_rate(r, "mtri_s", "Mtri/s", us, gb_param_int(o, "tris", 262144) / 1e6 * 1e6);
}

static const char *const tvb_params[] = {"tris", "dim", NULL};

static const struct gb_stream_test t_render_tvb = {
   .name = "render.tvb",
   .known_params = tvb_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .report_first = true,
   .setup = tvb_setup,
   .teardown = r_teardown,
   .record = r_record,
   .derive = tvb_derive,
};

/* ======================= copy.bw ======================= */

enum copy_kind { C_BUF2BUF, C_BUF2IMG, C_FILL };

struct copy_state {
   enum copy_kind kind;
   struct gb_buf src, dst;
   struct gb_img img;
};

static void *
copy_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   const char *kind = gb_param(o, "kind", "buf2buf");
   struct copy_state *st = calloc(1, sizeof(*st));
   if (!strcmp(kind, "buf2buf")) {
      st->kind = C_BUF2BUF;
      gb_buf_create(vk, GB_COPY_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->src);
      gb_buf_create(vk, GB_COPY_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->dst);
   } else if (!strcmp(kind, "buf2img")) {
      st->kind = C_BUF2IMG;
      gb_buf_create(vk, GB_COPY_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, &st->src);
      gb_img_create(vk, GB_COPY_DIM, GB_COPY_DIM, RT_FMT, VK_IMAGE_USAGE_TRANSFER_DST_BIT, &st->img);
      VkCommandBuffer cb = gb_oneshot_begin(vk);
      gb_img_barrier(cb, st->img.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      gb_oneshot_end(vk, cb);
   } else if (!strcmp(kind, "fill")) {
      st->kind = C_FILL;
      gb_buf_create(vk, GB_COPY_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &st->dst);
   } else {
      free(st);
      set_err(err, n, "kind must be buf2buf, buf2img or fill");
      return NULL;
   }
   return st;
}

static void
copy_teardown(struct gb_vk *vk, void *p)
{
   struct copy_state *st = p;
   if (st->src.buf) gb_buf_destroy(vk, &st->src);
   if (st->dst.buf) gb_buf_destroy(vk, &st->dst);
   if (st->img.img) gb_img_destroy(vk, &st->img);
   free(st);
}

static void
copy_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct copy_state *st = p;
   switch (st->kind) {
   case C_BUF2BUF: {
      VkBufferCopy r = {0, 0, GB_COPY_BYTES};
      vkCmdCopyBuffer(cb, st->src.buf, st->dst.buf, 1, &r);
      break;
   }
   case C_BUF2IMG: {
      VkBufferImageCopy r = {
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {GB_COPY_DIM, GB_COPY_DIM, 1},
      };
      vkCmdCopyBufferToImage(cb, st->src.buf, st->img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      break;
   }
   case C_FILL:
      vkCmdFillBuffer(cb, st->dst.buf, 0, GB_COPY_BYTES, 0x3f3f3f3fu);
      break;
   }
}

static void
copy_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   (void)o;
   gb_derive_rate(r, "gbps", "GB/s", us, GB_COPY_BYTES / 1e9 * 1e6);
}

static const struct gb_stream_test t_copy_bw = {
   .name = "copy.bw",
   .known_params = kind_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = copy_setup,
   .teardown = copy_teardown,
   .record = copy_record,
   .derive = copy_derive,
};

/* ======================= diag.fill ======================= */
/* copy.bw kind=fill with a variable size and an optional pre-touch, matching the Metal executor. */

static const char *const diag_fill_params[] = {"size_mib", "init", "storage", NULL};

struct diag_fill_state {
   struct gb_buf dst;
   VkDeviceSize bytes;
};

static void *
diag_fill_setup(struct gb_vk *vk, const struct gb_opts *o, char *err, size_t n)
{
   long mib = gb_param_int(o, "size_mib", 64), init = gb_param_int(o, "init", 0);
   const char *storage = gb_param(o, "storage", "private");
   bool shared = !strcmp(storage, "shared");
   if (mib < 1 || mib > 1024 || (init != 0 && init != 1) || (!shared && strcmp(storage, "private"))) {
      set_err(err, n, "size_mib 1..1024, init 0|1, storage private|shared");
      return NULL;
   }
   struct diag_fill_state *st = calloc(1, sizeof(*st));
   st->bytes = (VkDeviceSize)mib << 20;
   gb_buf_create(vk, st->bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, shared, &st->dst);
   if (init) {
      /* One fill before timing: later fills overwrite pages the GPU has already written. */
      VkCommandBuffer cb = gb_oneshot_begin(vk);
      vkCmdFillBuffer(cb, st->dst.buf, 0, st->bytes, 0);
      gb_oneshot_end(vk, cb);
   }
   return st;
}

static void
diag_fill_teardown(struct gb_vk *vk, void *p)
{
   struct diag_fill_state *st = p;
   gb_buf_destroy(vk, &st->dst);
   free(st);
}

static void
diag_fill_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   (void)k;
   struct diag_fill_state *st = p;
   vkCmdFillBuffer(cb, st->dst.buf, 0, st->bytes, 0x3f3f3f3fu);
}

static void
diag_fill_derive(const struct gb_opts *o, struct gb_result *r, const struct gb_samples *us)
{
   gb_derive_rate(r, "gbps", "GB/s", us, ((double)gb_param_int(o, "size_mib", 64) * (1 << 20)) / 1e9 * 1e6);
}

static const struct gb_stream_test t_diag_fill = {
   .name = "diag.fill",
   .known_params = diag_fill_params,
   .default_depths = {1, 4}, .ndefault = 2,
   .setup = diag_fill_setup,
   .teardown = diag_fill_teardown,
   .record = diag_fill_record,
   .derive = diag_fill_derive,
};

/* ======================= query ======================= */


struct q_state {
   bool occl;
   struct cl_state *cl;
   struct r_state *rl;
   VkDevice dev;
   VkQueryPool qp;
   struct gb_samples readback;
   bool collect;
};

static void
q_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct q_state *st = p;
   uint32_t i = (uint32_t)(k % GB_QRING);
   if (!st->occl) {
      vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, st->qp, 2 * i);
      cl_record(st->cl, cb, k);
      vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, st->qp, 2 * i + 1);
   } else {
      struct r_state *rl = st->rl;
      float color[4] = {1, 1, 1, 1};
      gb_begin_render(cb, &rl->img, CLEAR);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rl->p.pipe);
      vkCmdPushConstants(cb, rl->p.layout, rl->p.push_stages, 0, 16, color);
      vkCmdBeginQuery(cb, st->qp, i, 0);
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndQuery(cb, st->qp, i);
      vkCmdEndRendering(cb);
   }
}

static void
q_done(void *p, uint64_t k, int slot)
{
   (void)slot;
   struct q_state *st = p;
   uint32_t i = (uint32_t)(k % GB_QRING);
   uint64_t v[2];
   uint64_t t0 = gb_now_ns();
   if (!st->occl)
      vkGetQueryPoolResults(st->dev, st->qp, 2 * i, 2, sizeof(v), v, 8,
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   else
      vkGetQueryPoolResults(st->dev, st->qp, i, 1, 8, v, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   uint64_t t1 = gb_now_ns();
   if (st->collect)
      gb_samples_push(&st->readback, (t1 - t0) / 1e3);
   vkResetQueryPool(st->dev, st->qp, st->occl ? i : 2 * i, st->occl ? 1 : 2);
}

static int
run_query(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "timestamp");
   bool occl = !strcmp(kind, "occlusion");
   if (!occl && strcmp(kind, "timestamp")) {
      gb_result_simple(o, o->mode, 1, NULL, "fail", "kind must be timestamp or occlusion");
      return 1;
   }
   static const int defd[] = {1, 16};
   const int *depths = o->ndepths ? o->depths : defd;
   int nd = o->ndepths ? o->ndepths : 2;
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   char err[64];
   struct q_state st = {.occl = occl, .dev = vk.dev};
   if (occl)
      st.rl = r_setup_kind(&vk, R_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0);
   else
      st.cl = cl_setup(&vk, o, err, sizeof(err));
   VkQueryPoolCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = occl ? VK_QUERY_TYPE_OCCLUSION : VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = occl ? GB_QRING : 2 * GB_QRING,
   };
   GB_VK_CHECK(vkCreateQueryPool(vk.dev, &qci, NULL, &st.qp));
   vkResetQueryPool(vk.dev, st.qp, 0, qci.queryCount);
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;

   for (int di = 0; di < nd; di++) {
      int d = depths[di];
      struct gb_pass base = {0}, q = {0};
      struct gb_stream s;
      gb_stream_init(&s, &vk, 0, d, occl ? r_record : cl_record, NULL, occl ? (void *)st.rl : (void *)st.cl);
      gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &base);
      gb_stream_finish(&s);
      gb_stream_init(&s, &vk, 0, d, q_record, q_done, &st);
      gb_samples_clear(&st.readback);
      st.collect = true;
      gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &q);
      st.collect = false;
      gb_stream_finish(&s);

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, d, NULL);
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (base.failed || q.failed || !q.submits || !base.submits) {
         gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
      } else {
         gb_result_stream_metrics(&r, &q, NULL);
         struct gb_samples bus = {0}, qus = {0};
         gb_pass_rate(&base, &bus);
         gb_pass_rate(&q, &qus);
         gb_result_metric(&r, "base_us_per_submit", "us", true, &bus, true);
         gb_result_metric1(&r, "overhead_us", "us", true, gb_median(&qus) - gb_median(&bus));
         gb_result_metric(&r, "readback_us", "us", true, &st.readback, false);
         gb_samples_free(&bus);
         gb_samples_free(&qus);
         gb_result_finish(&r, "ok", "overhead_us", NULL);
      }
      gb_pass_free(&base);
      gb_pass_free(&q);
   }
   gb_samples_free(&st.readback);
   vkDestroyQueryPool(vk.dev, st.qp, NULL);
   if (occl)
      r_teardown(&vk, st.rl);
   else
      cl_teardown(&vk, st.cl);
   gb_vk_destroy(&vk);
   return 0;
}

/* ======================= concurrent.cr ======================= */

struct ileave {
   struct cl_state *cl;
   struct r_state *rl;
};

static void
ileave_record(void *p, VkCommandBuffer cb, uint64_t k)
{
   struct ileave *st = p;
   if (k & 1)
      cl_record(st->cl, cb, k);
   else
      r_record(st->rl, cb, k);
}

struct cr_thr {
   struct gb_stream *s;
   uint64_t t0, warm, win;
   int nwin;
   struct gb_pass pass;
   pthread_t th;
};

static void *
cr_thr_main(void *p)
{
   struct cr_thr *t = p;
   gb_stream_run(t->s, false, t->t0, t->warm, t->nwin, t->win, &t->pass);
   return NULL;
}

static int
run_concurrent(const struct gb_opts *o)
{
   static const char *const known[] = {"mode", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *mode = gb_param(o, "mode", "queues");
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   char err[64];
   if (!strcmp(mode, "queues")) {
      const int depth = GB_CR_DEPTH;
      if (gb_vk_probe_queue_count(o->device) < 2) {
         gb_result_simple(o, GB_MODE_QUEUES, 2 * depth, NULL, "skip", "device has 1 queue");
         return 0;
      }
      struct gb_vk vk;
      gb_vk_create(&vk, o->device, 2);
      struct gb_opts oc = *o;
      oc.nparams = 0;
      struct ch_state *ch = ch_setup(&vk, &oc, err, sizeof(err));
      struct r_state *rh = r_setup_kind(&vk, R_FILL, 1920, 1080, 0);
      struct gb_stream sc, sr;
      gb_stream_init(&sc, &vk, 0, depth, ch_record, NULL, ch);
      gb_stream_init(&sr, &vk, 1, depth, r_record, NULL, rh);
      struct gb_pass ca, ra;
      gb_stream_run(&sc, false, gb_now_ns(), W, o->runs, R, &ca);
      gb_stream_run(&sr, false, gb_now_ns(), W, o->runs, R, &ra);
      uint64_t t0 = gb_now_ns() + 20000000ull;
      struct cr_thr th[2] = {{.s = &sc, .t0 = t0, .warm = W, .win = R, .nwin = o->runs},
                             {.s = &sr, .t0 = t0, .warm = W, .win = R, .nwin = o->runs}};
      for (int i = 0; i < 2; i++)
         pthread_create(&th[i].th, NULL, cr_thr_main, &th[i]);
      for (int i = 0; i < 2; i++)
         pthread_join(th[i].th, NULL);
      gb_stream_finish(&sc);
      gb_stream_finish(&sr);

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2 * depth, (const char *const[]){"compute=compute.heavy:alu",
                                                                             "render=render.heavy:fill:1080p", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples s[4] = {0};
      gb_pass_rate(&ca, &s[0]);
      gb_pass_rate(&ra, &s[1]);
      gb_pass_rate(&th[0].pass, &s[2]);
      gb_pass_rate(&th[1].pass, &s[3]);
      bool bad = ca.failed || ra.failed || th[0].pass.failed || th[1].pass.failed;
      for (int i = 0; i < 4; i++)
         bad |= !s[i].n;
      if (bad) {
         gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
      } else {
         gb_result_metric(&r, "compute_alone_us", "us", true, &s[0], true);
         gb_result_metric(&r, "render_alone_us", "us", true, &s[1], true);
         gb_result_metric(&r, "compute_conc_us", "us", true, &s[2], true);
         gb_result_metric(&r, "render_conc_us", "us", true, &s[3], true);
         double ov = gb_median(&s[0]) / gb_median(&s[2]) + gb_median(&s[1]) / gb_median(&s[3]);
         gb_result_metric1(&r, "overlap", "x", false, ov);
         gb_result_finish(&r, "ok", "overlap", NULL);
      }
      for (int i = 0; i < 4; i++)
         gb_samples_free(&s[i]);
      gb_pass_free(&ca);
      gb_pass_free(&ra);
      gb_pass_free(&th[0].pass);
      gb_pass_free(&th[1].pass);
      ch_teardown(&vk, ch);
      r_teardown(&vk, rh);
      gb_vk_destroy(&vk);
      return 0;
   }
   if (!strcmp(mode, "interleave")) {
      const int depth = GB_IL_DEPTH;
      struct gb_vk vk;
      gb_vk_create(&vk, o->device, 1);
      struct gb_opts oc = *o;
      oc.nparams = 0;
      struct ileave il = {.cl = cl_setup(&vk, &oc, err, sizeof(err)), .rl = r_setup_kind(&vk, R_TRI, GB_LIGHT_DIM, GB_LIGHT_DIM, 0)};
      struct gb_pass pc, pr, pi;
      struct gb_stream s;
      gb_stream_init(&s, &vk, 0, depth, cl_record, NULL, il.cl);
      gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pc);
      gb_stream_finish(&s);
      gb_stream_init(&s, &vk, 0, depth, r_record, NULL, il.rl);
      gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pr);
      gb_stream_finish(&s);
      gb_stream_init(&s, &vk, 0, depth, ileave_record, NULL, &il);
      gb_stream_run(&s, false, gb_now_ns(), W, o->runs, R, &pi);
      gb_stream_finish(&s);

      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, depth, (const char *const[]){"compute=compute.light",
                                                                         "render=render.light:tri", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples s3[3] = {0};
      gb_pass_rate(&pc, &s3[0]);
      gb_pass_rate(&pr, &s3[1]);
      gb_pass_rate(&pi, &s3[2]);
      if (pc.failed || pr.failed || pi.failed || !s3[0].n || !s3[1].n || !s3[2].n) {
         gb_result_finish(&r, "fail", NULL, "GPU wait timed out or no completions");
      } else {
         gb_result_metric(&r, "us_per_submit", "us", true, &s3[2], true);
         gb_result_metric(&r, "latency_us", "us", true, &pi.latency, false);
         gb_result_metric(&r, "compute_only_us", "us", true, &s3[0], true);
         gb_result_metric(&r, "render_only_us", "us", true, &s3[1], true);
         gb_result_metric1(&r, "switch_us", "us", true,
                           gb_median(&s3[2]) - 0.5 * (gb_median(&s3[0]) + gb_median(&s3[1])));
         gb_result_finish(&r, "ok", "us_per_submit", NULL);
      }
      for (int i = 0; i < 3; i++)
         gb_samples_free(&s3[i]);
      gb_pass_free(&pc);
      gb_pass_free(&pr);
      gb_pass_free(&pi);
      cl_teardown(&vk, il.cl);
      r_teardown(&vk, il.rl);
      gb_vk_destroy(&vk);
      return 0;
   }
   gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "mode must be queues or interleave");
   return 1;
}

/* ======================= sync.signal ======================= */

static VkSemaphore
make_sem(struct gb_vk *vk, bool timeline)
{
   VkSemaphoreTypeCreateInfo tci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY,
   };
   VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
   VkSemaphore s;
   GB_VK_CHECK(vkCreateSemaphore(vk->dev, &sci, NULL, &s));
   return s;
}

static int
wait_tl(struct gb_vk *vk, VkSemaphore s, uint64_t v)
{
   VkSemaphoreWaitInfo wi = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &s,
      .pValues = &v,
   };
   return vkWaitSemaphores(vk->dev, &wi, 10000000000ull) == VK_SUCCESS ? 0 : -1;
}

struct slotcb {
   VkCommandPool pool;
   VkCommandBuffer cb;
};

static void
slotcb_init(struct gb_vk *vk, struct slotcb *s)
{
   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = vk->qfam,
   };
   GB_VK_CHECK(vkCreateCommandPool(vk->dev, &pci, NULL, &s->pool));
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = s->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   GB_VK_CHECK(vkAllocateCommandBuffers(vk->dev, &ai, &s->cb));
}

static void
slotcb_record(struct gb_vk *vk, struct slotcb *s, struct cl_state *cl, uint64_t k)
{
   vkResetCommandPool(vk->dev, s->pool, 0);
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   vkBeginCommandBuffer(s->cb, &bi);
   cl_record(cl, s->cb, k);
   vkEndCommandBuffer(s->cb);
}

static VkResult
submit_ws(VkQueue q, VkCommandBuffer cb, VkSemaphore wsem, uint64_t wval, VkSemaphore s1, uint64_t v1,
          VkSemaphore s2, uint64_t v2)
{
   VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
   VkSemaphoreSubmitInfo w = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = wsem,
      .value = wval,
      .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
   };
   VkSemaphoreSubmitInfo sig[2] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = s1, .value = v1,
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = s2, .value = v2,
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
   };
   VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = wsem ? 1 : 0,
      .pWaitSemaphoreInfos = &w,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cbi,
      .signalSemaphoreInfoCount = s2 ? 2 : (s1 ? 1 : 0),
      .pSignalSemaphoreInfos = sig,
   };
   return vkQueueSubmit2(q, 1, &si, VK_NULL_HANDLE);
}

static int
run_sync(const struct gb_opts *o)
{
   static const char *const known[] = {"kind", "sema", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   const char *kind = gb_param(o, "kind", "host");
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   char err[64];

   if (!strcmp(kind, "host")) {
      struct gb_vk vk;
      gb_vk_create(&vk, o->device, 1);
      struct cl_state *cl = cl_setup(&vk, o, err, sizeof(err));
      VkSemaphore T = make_sem(&vk, true), S = make_sem(&vk, true);
      struct slotcb sc;
      slotcb_init(&vk, &sc);
      struct gb_samples lat = {0}, base = {0};
      uint64_t v = 0;
      bool bad = false;
      for (int phase = 0; phase < 2 && !bad; phase++) {
         /* phase 0: plain round trip; phase 1: the submission waits on a host signal */
         uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
         for (;;) {
            uint64_t now = gb_now_ns();
            if (now >= t_end)
               break;
            v++;
            slotcb_record(&vk, &sc, cl, v);
            uint64_t a, b;
            if (phase == 0) {
               a = gb_now_ns();
               submit_ws(vk.queues[0], sc.cb, VK_NULL_HANDLE, 0, S, v, VK_NULL_HANDLE, 0);
            } else {
               submit_ws(vk.queues[0], sc.cb, T, v, S, v, VK_NULL_HANDLE, 0);
               gb_spin_ns(GB_HOST_SIGNAL_DELAY_NS);
               a = gb_now_ns();
               vkSignalSemaphore(vk.dev, &(VkSemaphoreSignalInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                                                                  .semaphore = T, .value = v});
            }
            if (wait_tl(&vk, S, v)) {
               bad = true;
               break;
            }
            b = gb_now_ns();
            if (a >= tm0)
               gb_samples_push(phase ? &lat : &base, (b - a) / 1e3);
         }
      }
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      if (bad || !lat.n) {
         gb_result_finish(&r, "fail", NULL, "semaphore wait timed out");
      } else {
         gb_result_metric(&r, "latency_us", "us", true, &lat, false);
         gb_result_metric(&r, "base_latency_us", "us", true, &base, false);
         gb_result_metric1(&r, "added_us", "us", true, gb_median(&lat) - gb_median(&base));
         gb_result_finish(&r, "ok", "latency_us", NULL);
      }
      gb_samples_free(&lat);
      gb_samples_free(&base);
      vkDestroyCommandPool(vk.dev, sc.pool, NULL);
      vkDestroySemaphore(vk.dev, T, NULL);
      vkDestroySemaphore(vk.dev, S, NULL);
      cl_teardown(&vk, cl);
      gb_vk_destroy(&vk);
      return 0;
   }

   if (!strcmp(kind, "queue")) {
      const char *sema = gb_param(o, "sema", "timeline");
      bool binary = !strcmp(sema, "binary");
      if (!binary && strcmp(sema, "timeline")) {
         gb_result_simple(o, GB_MODE_QUEUES, 2, NULL, "fail", "sema must be timeline or binary");
         return 1;
      }
      if (gb_vk_probe_queue_count(o->device) < 2) {
         gb_result_simple(o, GB_MODE_QUEUES, 2, NULL, "skip", "device has 1 queue");
         return 0;
      }
      enum { RING = GB_PING_RING, BRING = 4 * GB_PING_RING };
      struct gb_vk vk;
      gb_vk_create(&vk, o->device, 2);
      struct cl_state *cl = cl_setup(&vk, o, err, sizeof(err));
      VkSemaphore S = make_sem(&vk, true), T = make_sem(&vk, true);
      VkSemaphore bs[BRING], bt[BRING];
      for (int i = 0; i < BRING; i++) {
         bs[i] = binary ? make_sem(&vk, false) : VK_NULL_HANDLE;
         bt[i] = binary ? make_sem(&vk, false) : VK_NULL_HANDLE;
      }
      struct slotcb ca[RING], cbq[RING];
      for (int i = 0; i < RING; i++) {
         slotcb_init(&vk, &ca[i]);
         slotcb_init(&vk, &cbq[i]);
      }
      uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
      uint64_t k = 0, seen = 0, win[GB_MAX_WIN] = {0};
      bool bad = false;
      for (;;) {
         uint64_t now = gb_now_ns();
         if (k - seen >= RING || (now >= t_end && seen < k)) {
            if (wait_tl(&vk, T, seen + 1)) {
               bad = true;
               break;
            }
            uint64_t t = gb_now_ns();
            if (t >= tm0 && t < t_end)
               win[(t - tm0) / R]++;
            seen++;
            continue;
         }
         if (now >= t_end)
            break;
         k++;
         int i = k % RING;
         slotcb_record(&vk, &ca[i], cl, k);
         slotcb_record(&vk, &cbq[i], cl, k);
         VkResult r1, r2;
         if (!binary) {
            r1 = submit_ws(vk.queues[0], ca[i].cb, T, k - 1, S, k, VK_NULL_HANDLE, 0);
            r2 = submit_ws(vk.queues[1], cbq[i].cb, S, k, T, k, VK_NULL_HANDLE, 0);
         } else {
            r1 = submit_ws(vk.queues[0], ca[i].cb, k > 1 ? bt[(k - 1) % BRING] : VK_NULL_HANDLE, 0,
                           bs[k % BRING], 0, VK_NULL_HANDLE, 0);
            r2 = submit_ws(vk.queues[1], cbq[i].cb, bs[k % BRING], 0, bt[k % BRING], 0, T, k);
         }
         if (r1 != VK_SUCCESS || r2 != VK_SUCCESS) {
            bad = true;
            break;
         }
      }
      vkDeviceWaitIdle(vk.dev);
      struct gb_result r;
      gb_result_begin(&r, o, GB_MODE_QUEUES, 2, (const char *const[]){"work=compute.light", NULL});
      gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
      struct gb_samples hop = {0};
      for (int w = 0; w < o->runs; w++)
         if (win[w])
            gb_samples_push(&hop, R / 1e3 / (2.0 * win[w]));
      if (bad || !hop.n) {
         gb_result_finish(&r, "fail", NULL, "semaphore wait timed out or submit failed");
      } else {
         gb_result_metric(&r, "us_per_hop", "us", true, &hop, true);
         gb_result_finish(&r, "ok", "us_per_hop", NULL);
      }
      gb_samples_free(&hop);
      for (int i = 0; i < RING; i++) {
         vkDestroyCommandPool(vk.dev, ca[i].pool, NULL);
         vkDestroyCommandPool(vk.dev, cbq[i].pool, NULL);
      }
      for (int i = 0; i < BRING; i++)
         if (binary) {
            vkDestroySemaphore(vk.dev, bs[i], NULL);
            vkDestroySemaphore(vk.dev, bt[i], NULL);
         }
      vkDestroySemaphore(vk.dev, S, NULL);
      vkDestroySemaphore(vk.dev, T, NULL);
      cl_teardown(&vk, cl);
      gb_vk_destroy(&vk);
      return 0;
   }
   gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "kind must be host or queue");
   return 1;
}

/* ======================= idle.wake ======================= */

/* Round trip of one compute.light submission after the GPU has been idle for
 * gap_ms: the power-down / wake-up path in firmware and kernel. */
static int
run_idle(const struct gb_opts *o)
{
   static const char *const known[] = {"gap_ms", NULL};
   if (!gb_check_params(o, known)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   long gap = gb_param_int(o, "gap_ms", 20);
   if (gap < 0 || gap > 5000) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "gap_ms out of range");
      return 1;
   }
   int n = (int)(GB_IDLE_BUDGET_MS / (gap ? gap : 1));
   n = n < GB_IDLE_MIN_SAMPLES ? GB_IDLE_MIN_SAMPLES : n > GB_IDLE_MAX_SAMPLES ? GB_IDLE_MAX_SAMPLES : n;
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   char err[64];
   struct cl_state *cl = cl_setup(&vk, o, err, sizeof(err));
   VkSemaphore tl = make_sem(&vk, true);
   struct slotcb sc;
   slotcb_init(&vk, &sc);
   struct gb_samples lat = {0}, hot = {0};
   bool bad = false;
   uint64_t v = 0;
   for (int i = -2; i < n && !bad; i++) { /* two warm-up rounds */
      slotcb_record(&vk, &sc, cl, ++v);
      gb_sleep_until_ns(gb_now_ns() + (uint64_t)gap * 1000000ull);
      uint64_t a = gb_now_ns();
      submit_ws(vk.queues[0], sc.cb, VK_NULL_HANDLE, 0, tl, v, VK_NULL_HANDLE, 0);
      if (wait_tl(&vk, tl, v)) {
         bad = true;
         break;
      }
      uint64_t b = gb_now_ns();
      /* an immediate second submission: the same path with the GPU awake */
      slotcb_record(&vk, &sc, cl, ++v);
      uint64_t c = gb_now_ns();
      submit_ws(vk.queues[0], sc.cb, VK_NULL_HANDLE, 0, tl, v, VK_NULL_HANDLE, 0);
      if (wait_tl(&vk, tl, v)) {
         bad = true;
         break;
      }
      uint64_t d = gb_now_ns();
      if (i >= 0) {
         gb_samples_push(&lat, (b - a) / 1e3);
         gb_samples_push(&hot, (d - c) / 1e3);
      }
   }
   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, (const char *const[]){"work=compute.light", NULL});
   if (bad || !lat.n) {
      gb_result_finish(&r, "fail", NULL, "GPU wait timed out");
   } else {
      gb_result_metric(&r, "wake_latency_us", "us", true, &lat, false);
      gb_result_metric(&r, "awake_latency_us", "us", true, &hot, false);
      gb_result_metric1(&r, "wake_cost_us", "us", true, gb_median(&lat) - gb_median(&hot));
      gb_result_finish(&r, "ok", "wake_latency_us", NULL);
   }
   gb_samples_free(&lat);
   gb_samples_free(&hot);
   vkDestroyCommandPool(vk.dev, sc.pool, NULL);
   vkDestroySemaphore(vk.dev, tl, NULL);
   cl_teardown(&vk, cl);
   gb_vk_destroy(&vk);
   return 0;
}

/* ======================= mem.buffer / mem.image ======================= */

static int
run_mem(const struct gb_opts *o, bool image)
{
   static const char *const kb[] = {"size", NULL}, *const ki[] = {"dim", NULL};
   if (!gb_check_params(o, image ? ki : kb)) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "unknown parameter");
      return 1;
   }
   long size = gb_param_int(o, "size", 1 << 20);
   long dim = gb_param_int(o, "dim", 1024);
   if (size < 4096 || size > (1l << 30) || dim < 16 || dim > 16384) {
      gb_result_simple(o, GB_MODE_QUEUE, 1, NULL, "fail", "size/dim out of range");
      return 1;
   }
   const uint64_t W = (uint64_t)o->warmup_ms * 1000000ull, R = (uint64_t)o->run_ms * 1000000ull;
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   struct gb_samples s_alloc = {0}, s_map = {0}, s_touch = {0}, s_unmap = {0}, s_free = {0}, s_cycle = {0};
   const VkDevice dev = vk.dev;
   VkSemaphore tl = make_sem(&vk, true);
   uint64_t tlv = 0;
   struct slotcb sc;
   slotcb_init(&vk, &sc);
   bool bad = false;

   /* phase 1: CPU-side object lifetime */
   uint64_t t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
   while (gb_now_ns() < t_end) {
      uint64_t a = gb_now_ns();
      bool m = a >= tm0;
      if (!image) {
         struct gb_buf b;
         VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = (VkDeviceSize)size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
         GB_VK_CHECK(vkCreateBuffer(dev, &ci, NULL, &b.buf));
         VkMemoryRequirements req;
         vkGetBufferMemoryRequirements(dev, b.buf, &req);
         VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .allocationSize = req.size,
                                    .memoryTypeIndex = vk.mem_host};
         GB_VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
         GB_VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
         uint64_t t1 = gb_now_ns();
         void *p;
         GB_VK_CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p));
         uint64_t t2 = gb_now_ns();
         for (long off = 0; off < size; off += GB_TOUCH_STRIDE)
            ((volatile uint32_t *)p)[off / 4] = (uint32_t)off;
         uint64_t t3 = gb_now_ns();
         vkUnmapMemory(dev, b.mem);
         uint64_t t4 = gb_now_ns();
         vkDestroyBuffer(dev, b.buf, NULL);
         vkFreeMemory(dev, b.mem, NULL);
         uint64_t t5 = gb_now_ns();
         if (m) {
            gb_samples_push(&s_alloc, (t1 - a) / 1e3);
            gb_samples_push(&s_map, (t2 - t1) / 1e3);
            gb_samples_push(&s_touch, (t3 - t2) / 1e3);
            gb_samples_push(&s_unmap, (t4 - t3) / 1e3);
            gb_samples_push(&s_free, (t5 - t4) / 1e3);
         }
      } else {
         struct gb_img im;
         gb_img_create(&vk, (uint32_t)dim, (uint32_t)dim, RT_FMT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &im);
         uint64_t t1 = gb_now_ns();
         gb_img_destroy(&vk, &im);
         uint64_t t2 = gb_now_ns();
         if (m) {
            gb_samples_push(&s_alloc, (t1 - a) / 1e3);
            gb_samples_push(&s_free, (t2 - t1) / 1e3);
         }
      }
   }

   /* phase 2: allocate -> first GPU use -> wait -> free */
   t0 = gb_now_ns(), tm0 = t0 + W, t_end = tm0 + (uint64_t)o->runs * R;
   while (!bad && gb_now_ns() < t_end) {
      uint64_t a = gb_now_ns();
      vkResetCommandPool(dev, sc.pool, 0);
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      struct gb_buf b = {0};
      struct gb_img im = {0};
      if (!image) {
         gb_buf_create(&vk, (VkDeviceSize)size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, &b);
         vkBeginCommandBuffer(sc.cb, &bi);
         vkCmdFillBuffer(sc.cb, b.buf, 0, VK_WHOLE_SIZE, 0);
      } else {
         gb_img_create(&vk, (uint32_t)dim, (uint32_t)dim, RT_FMT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &im);
         vkBeginCommandBuffer(sc.cb, &bi);
         gb_img_barrier(sc.cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
         gb_begin_render(sc.cb, &im, CLEAR);
         vkCmdEndRendering(sc.cb);
      }
      vkEndCommandBuffer(sc.cb);
      tlv++;
      submit_ws(vk.queues[0], sc.cb, VK_NULL_HANDLE, 0, tl, tlv, VK_NULL_HANDLE, 0);
      if (wait_tl(&vk, tl, tlv))
         bad = true;
      if (!image)
         gb_buf_destroy(&vk, &b);
      else
         gb_img_destroy(&vk, &im);
      uint64_t z = gb_now_ns();
      if (a >= tm0)
         gb_samples_push(&s_cycle, (z - a) / 1e3);
   }

   struct gb_result r;
   gb_result_begin(&r, o, GB_MODE_QUEUE, 1, NULL);
   gb_result_run_info(&r, o->warmup_ms, o->runs, o->run_ms);
   if (bad) {
      gb_result_finish(&r, "fail", NULL, "GPU wait timed out");
   } else {
      gb_result_metric(&r, "alloc_us", "us", true, &s_alloc, false);
      gb_result_metric(&r, "map_us", "us", true, &s_map, false);
      gb_result_metric(&r, "touch_us", "us", true, &s_touch, false);
      gb_result_metric(&r, "unmap_us", "us", true, &s_unmap, false);
      gb_result_metric(&r, "free_us", "us", true, &s_free, false);
      gb_result_metric(&r, "cycle_us", "us", true, &s_cycle, false);
      gb_result_finish(&r, "ok", "cycle_us", NULL);
   }
   gb_samples_free(&s_alloc);
   gb_samples_free(&s_map);
   gb_samples_free(&s_touch);
   gb_samples_free(&s_unmap);
   gb_samples_free(&s_free);
   gb_samples_free(&s_cycle);
   vkDestroyCommandPool(dev, sc.pool, NULL);
   vkDestroySemaphore(dev, tl, NULL);
   gb_vk_destroy(&vk);
   return 0;
}

static int run_mem_buffer(const struct gb_opts *o) { return run_mem(o, false); }
static int run_mem_image(const struct gb_opts *o) { return run_mem(o, true); }

/* ======================= table ======================= */

static int st_submit_empty(const struct gb_opts *o) { return gb_run_stream_test(o, &t_submit_empty); }
static int st_compute_light(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_light); }
static int st_compute_heavy(const struct gb_opts *o) { return gb_run_stream_test(o, &t_compute_heavy); }
static int st_render_light(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_light); }
static int st_render_heavy(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_heavy); }
static int st_render_tvb(const struct gb_opts *o) { return gb_run_stream_test(o, &t_render_tvb); }
static int st_copy_bw(const struct gb_opts *o) { return gb_run_stream_test(o, &t_copy_bw); }
static int st_diag_fill(const struct gb_opts *o) { return gb_run_stream_test(o, &t_diag_fill); }

const struct gb_test gb_tests[] = {
   {"submit.empty", st_submit_empty},
   {"compute.light", st_compute_light},
   {"compute.heavy", st_compute_heavy},
   {"render.light", st_render_light},
   {"render.heavy", st_render_heavy},
   {"render.tvb", st_render_tvb},
   {"sync.signal", run_sync},
   {"mem.buffer", run_mem_buffer},
   {"mem.image", run_mem_image},
   {"copy.bw", st_copy_bw},
   {"diag.fill", st_diag_fill},
   {"concurrent.cr", run_concurrent},
   {"query", run_query},
   {"idle.wake", run_idle},
   {NULL, NULL},
};

/* ======================= shader statistics ======================= */

static void
print_stats(struct gb_vk *vk, const char *name, VkPipeline pipe)
{
   PFN_vkGetPipelineExecutablePropertiesKHR gp =
      (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(vk->dev, "vkGetPipelineExecutablePropertiesKHR");
   PFN_vkGetPipelineExecutableStatisticsKHR gs =
      (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(vk->dev, "vkGetPipelineExecutableStatisticsKHR");
   if (!gp || !gs) {
      printf("{\"pipeline\":\"%s\",\"error\":\"no pipeline executable properties\"}\n", name);
      return;
   }
   VkPipelineInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe};
   uint32_t n = 0;
   gp(vk->dev, &pi, &n, NULL);
   VkPipelineExecutablePropertiesKHR props[8];
   for (uint32_t i = 0; i < n && i < 8; i++)
      props[i] = (VkPipelineExecutablePropertiesKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
   if (n > 8)
      n = 8;
   gp(vk->dev, &pi, &n, props);
   for (uint32_t e = 0; e < n; e++) {
      VkPipelineExecutableInfoKHR ei = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                        .pipeline = pipe,
                                        .executableIndex = e};
      uint32_t ns = 0;
      gs(vk->dev, &ei, &ns, NULL);
      VkPipelineExecutableStatisticKHR *st = calloc(ns, sizeof(*st));
      for (uint32_t i = 0; i < ns; i++)
         st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      gs(vk->dev, &ei, &ns, st);
      struct gb_json j;
      gb_json_init(&j);
      gb_json_obj_begin(&j, NULL);
      gb_json_str(&j, "pipeline", name);
      gb_json_str(&j, "stage", props[e].name);
      gb_json_obj_begin(&j, "stats");
      for (uint32_t i = 0; i < ns; i++) {
         switch (st[i].format) {
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: gb_json_bool(&j, st[i].name, st[i].value.b32); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: gb_json_int(&j, st[i].name, st[i].value.i64); break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
            gb_json_int(&j, st[i].name, (int64_t)st[i].value.u64);
            break;
         case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: gb_json_num(&j, st[i].name, st[i].value.f64); break;
         default: break;
         }
      }
      gb_json_obj_end(&j);
      gb_json_obj_end(&j);
      puts(j.buf);
      fflush(stdout);
      gb_json_free(&j);
      free(st);
   }
}

/* The pipelines whose code FAIRNESS.md compares, in a fixed order. Each is
 * announced on stderr ("GB_PIPELINE <name>") before it is built, so compiler
 * dumps (AGX_MESA_DEBUG=shaders) can be attributed. */
int
gb_shader_stats(const struct gb_opts *o)
{
   gb_vk_want_pipeline_stats = true;
   struct gb_vk vk;
   gb_vk_create(&vk, o->device, 1);
   struct gb_pipe p;
#define MARK(n) (fprintf(stderr, "GB_PIPELINE %s\n", n), fflush(stderr))
   MARK("light");
   gb_compute_pipe(&vk, GB_SPV(light_comp), 1, 4, &p);
   print_stats(&vk, "light", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("alu");
   gb_compute_pipe(&vk, GB_SPV(alu_comp), 1, 12, &p);
   print_stats(&vk, "alu", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("bw");
   gb_compute_pipe(&vk, GB_SPV(bw_comp), 2, 0, &p);
   print_stats(&vk, "bw", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("fs_color");
   gb_graphics_pipe(&vk, GB_SPV(fs_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &p);
   print_stats(&vk, "fs_color", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("fs_color_blend");
   gb_graphics_pipe(&vk, GB_SPV(fs_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_ALPHA, 16, &p);
   print_stats(&vk, "fs_color_blend", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("geom_color");
   gb_graphics_pipe(&vk, GB_SPV(geom_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &p);
   print_stats(&vk, "geom_color", p.pipe);
   gb_pipe_destroy(&vk, &p);
   MARK("tvb_color");
   gb_graphics_pipe(&vk, GB_SPV(tvb_vert), GB_SPV(color_frag), RT_FMT, GB_BLEND_NONE, 16, &p);
   print_stats(&vk, "tvb_color", p.pipe);
   gb_pipe_destroy(&vk, &p);
#undef MARK
   gb_vk_destroy(&vk);
   return 0;
}
