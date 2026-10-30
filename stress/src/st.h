// SPDX-License-Identifier: GPL-2.0-only
/* gpubench-stress-vk: driver stress tests with pass/fail verdicts. See stress/README.md.
 *
 * Every test runs in its own process (forked by the runner), creates its own Vulkan instances
 * and devices, checks every result it produces against a CPU model, and reports one verdict. */
#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

#include <vulkan/vulkan.h>

#include "gb_common.h"

#define ST_MAX_QUEUES 16
#define ST_MAX_METRICS 32
#define ST_NS 1000000000ull
#define ST_MS 1000000ull

enum st_status { ST_PASS = 0, ST_FAIL = 1, ST_SKIP = 2 };

struct st_opts {
   int device;
   double duration_s;   /* total budget of the selected tests */
   bool dangerous;
   bool verbose;
   uint64_t seed;
};

struct st_run;

struct st_test {
   const char *name;
   const char *summary;   /* what it does */
   const char *protects;  /* the failure class it guards against */
   bool dangerous;
   double weight;          /* share of --duration */
   /* Kernel-log lines this test causes on purpose (extended regex, case-insensitive), or NULL. */
   const char *kernel_expect;
   int (*run)(struct st_run *r);
};

extern const struct st_test st_tests[];
const struct st_test *st_find_test(const char *name);

struct st_metric {
   char name[40];
   double v;
};

/* Per-test state in the test process. */
struct st_run {
   const struct st_opts *o;
   const struct st_test *t;
   uint64_t t0, budget_ns;
   uint64_t rng;
   uint64_t iterations, checks, errors;
   char reason[512];
   bool skipped;
   struct st_metric m[ST_MAX_METRICS];
   int nm;
};

/* ---- run helpers ---- */
uint64_t st_now(void);
void st_sleep_ns(uint64_t ns);
static inline uint64_t st_elapsed(const struct st_run *r) { return st_now() - r->t0; }
static inline bool st_time_left(const struct st_run *r) { return st_elapsed(r) < r->budget_ns; }
/* Fraction f of the budget, at least min_ns. */
uint64_t st_part(const struct st_run *r, double f, uint64_t min_ns);
uint32_t st_rand(struct st_run *r);
uint32_t st_rand_range(struct st_run *r, uint32_t lo, uint32_t hi); /* inclusive */
/* Records a failure (the first reason is kept) and returns ST_FAIL. */
int st_fail(struct st_run *r, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
int st_skip(struct st_run *r, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void st_log(struct st_run *r, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void st_metric(struct st_run *r, const char *name, double v);
void st_metric_max(struct st_run *r, const char *name, double v);
void st_metric_add(struct st_run *r, const char *name, double v);
static inline int st_verdict(const struct st_run *r) { return r->errors ? ST_FAIL : r->skipped ? ST_SKIP : ST_PASS; }
const char *st_vkr(VkResult r);

/* ---- Vulkan device ---- */
struct st_caps {
   char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   char driver[VK_MAX_DRIVER_NAME_SIZE];
   char driver_info[VK_MAX_DRIVER_INFO_SIZE];
   uint32_t api;
   uint32_t queues;       /* graphics+compute family */
   bool sparse_queue;     /* that family can bind sparse memory */
   bool bda, rq, sparse, gprio, frag_stores;
   VkDeviceSize heap;     /* device-local heap bytes */
   VkDeviceSize max_alloc;
};

struct st_req {
   int nqueues;           /* 0 = 1 */
   bool bda, rq, sparse;
   bool mixed;            /* queue priorities 0, 0.5, 1 in turn instead of all 1 */
   int gprio;             /* VK_QUEUE_GLOBAL_PRIORITY_*_KHR for every queue, 0 = default */
};

struct st_rq_fns;

struct st_dev {
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   uint32_t qf;
   int nq;
   VkQueue q[ST_MAX_QUEUES];
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties mp;
   struct st_caps caps;
   uint32_t mem_host, mem_dev;
   bool bda, rq, sparse, gprio;
   VkCommandPool pool[ST_MAX_QUEUES]; /* one per queue index, for one-off command buffers */
   struct st_rq_fns *rqf;
};

/* Capabilities without creating a device (creates and destroys an instance). */
VkResult st_probe(int device_index, struct st_caps *c);
VkResult st_dev_create(struct st_dev *d, int device_index, const struct st_req *req);
/* Waits for idle (bounded) and destroys. Returns false if the device did not go idle. */
bool st_dev_destroy(struct st_dev *d);

/* ---- resources ---- */
struct st_buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   VkDeviceSize size;
   void *map;
   VkDeviceAddress addr;
};
VkResult st_buf_create(struct st_dev *d, VkDeviceSize size, VkBufferUsageFlags usage, bool host, struct st_buf *b);
void st_buf_destroy(struct st_dev *d, struct st_buf *b);

struct st_img {
   VkImage img;
   VkDeviceMemory mem;
   VkImageView view;
   uint32_t w, h;
   VkFormat fmt;
};
VkResult st_img_create(struct st_dev *d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                       struct st_img *im);
void st_img_destroy(struct st_dev *d, struct st_img *im);

/* ---- pipelines ---- */
struct st_pipe {
   VkPipeline pipe;
   VkPipelineLayout layout;
   VkDescriptorSetLayout dsl;
   VkDescriptorPool dpool;
   VkDescriptorSet set;
   VkShaderStageFlags stages;
   VkPipelineBindPoint bind;
};
/* Compute pipeline: `nbuf` storage buffers at set 0, push constants of push_size bytes. */
VkResult st_compute_pipe(struct st_dev *d, const uint32_t *spv, size_t bytes, int nbuf, uint32_t push_size,
                         struct st_pipe *p);
enum st_blend { ST_BLEND_NONE = 0, ST_BLEND_ADD };
/* Graphics pipeline for dynamic rendering into one color attachment: no vertex input, triangle
 * list, dynamic viewport/scissor, push constants to VS+FS, `nbuf` storage buffers (VS+FS). */
VkResult st_graphics_pipe(struct st_dev *d, const uint32_t *vs, size_t vs_bytes, const uint32_t *fs,
                          size_t fs_bytes, VkFormat fmt, enum st_blend blend, int nbuf, uint32_t push_size,
                          struct st_pipe *p);
void st_pipe_buffers(struct st_dev *d, struct st_pipe *p, struct st_buf *const *bufs, int n);
void st_pipe_destroy(struct st_dev *d, struct st_pipe *p);

/* ---- command helpers ---- */
VkResult st_cb_alloc(struct st_dev *d, int qi, VkCommandBuffer *cb);
VkResult st_cb_begin(VkCommandBuffer cb);
void st_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                VkAccessFlags2 da);
void st_img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                    VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst, VkAccessFlags2 da);
void st_begin_render(VkCommandBuffer cb, const struct st_img *im, VkAttachmentLoadOp load,
                     const VkClearColorValue *clear);
void st_viewport(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t w, uint32_t h);
/* Submission with timeline (value != 0) or binary (value == 0) waits/signals. */
struct st_sem_op {
   VkSemaphore sem;
   uint64_t value;
};
VkResult st_submit(VkQueue q, VkCommandBuffer cb, const struct st_sem_op *waits, int nwait,
                   const struct st_sem_op *sigs, int nsig, VkFence fence);
VkResult st_timeline(struct st_dev *d, uint64_t initial, VkSemaphore *s);
VkResult st_binary(struct st_dev *d, VkSemaphore *s);
VkResult st_wait_value(struct st_dev *d, VkSemaphore s, uint64_t v, uint64_t timeout_ns);
/* Records nothing, submits `cb` on queue qi with a fence and waits up to timeout_ns. */
VkResult st_run_cb(struct st_dev *d, int qi, VkCommandBuffer cb, uint64_t timeout_ns);

/* ---- shared verified workloads (st_work.c) ---- */

/* Chain compute: v[base + i] = v[base + i] * 3 + k + i over n elements (n multiple of 64). */
struct st_chain {
   struct st_pipe pipe;
};
VkResult st_chain_init(struct st_dev *d, struct st_buf *data, struct st_chain *c);
void st_chain_record(VkCommandBuffer cb, struct st_chain *c, uint32_t base, uint32_t n, uint32_t k);
static inline uint32_t st_chain_model(uint32_t v, uint32_t k, uint32_t i) { return v * 3u + k + i; }
void st_chain_destroy(struct st_dev *d, struct st_chain *c);

/* Render check: clear a w x h RGBA8 target to colour A, draw a scissored full-screen triangle in
 * colour B over the left half and copy the target to a host buffer. Every pixel is checked. */
struct st_rcheck {
   struct st_pipe pipe;
   struct st_img img;
   struct st_buf rb;
   uint32_t w, h;
};
VkResult st_rcheck_init(struct st_dev *d, uint32_t w, uint32_t h, struct st_rcheck *rc);
void st_rcheck_record(VkCommandBuffer cb, struct st_rcheck *rc, uint32_t k);
/* Number of wrong pixels (0 = correct); describes the first one in `why`. */
uint64_t st_rcheck_verify(const struct st_rcheck *rc, uint32_t k, char *why, size_t n);
void st_rcheck_destroy(struct st_dev *d, struct st_rcheck *rc);

/* Long ALU: out[gid] = lcg^iters(seed ^ gid). groups x 64 invocations. */
struct st_lcg {
   struct st_pipe pipe;
   struct st_buf out;
   uint32_t groups;
};
VkResult st_lcg_init(struct st_dev *d, uint32_t groups, struct st_lcg *l);
void st_lcg_record(VkCommandBuffer cb, struct st_lcg *l, uint32_t iters, uint32_t seed);
uint32_t st_lcg_model(uint32_t seed, uint32_t gid, uint32_t iters);
/* Checks `samples` invocations against the model; returns the number of mismatches. */
uint64_t st_lcg_verify(const struct st_lcg *l, uint32_t iters, uint32_t seed, int samples, char *why, size_t n);
/* Iterations for a dispatch of about target_ms alone on queue qi (0 on failure). */
uint32_t st_lcg_calibrate(struct st_dev *d, int qi, struct st_lcg *l, double target_ms, double *measured_ms);
void st_lcg_destroy(struct st_dev *d, struct st_lcg *l);

/* ---- processes ---- */
/* Forks a child that runs fn(arg) and _exits with its return value. The child is in the
 * caller's process group. Returns the pid or -1. */
pid_t st_fork(int (*fn)(void *arg), void *arg);
/* Waits for pid up to timeout_ns. Returns the exit status (>= 0), -1 if it was killed by a
 * signal (*sig set), or -2 on timeout (the child is then SIGKILLed and reaped if possible). */
int st_wait_pid(pid_t pid, uint64_t timeout_ns, int *sig);
/* A one-shot start gate for several children: children block in st_gate_wait until
 * st_gate_open. Created before forking. */
struct st_gate {
   int fd[2];
};
void st_gate_init(struct st_gate *g);
void st_gate_wait(struct st_gate *g); /* child side */
void st_gate_open(struct st_gate *g); /* parent side, releases every waiter at once */

/* ---- shaders ---- */
#define ST_SHADER(n) extern const uint32_t n[]; extern const size_t n##_size;
ST_SHADER(chain_comp)
ST_SHADER(order_comp)
ST_SHADER(lcg_comp)
ST_SHADER(link_comp)
ST_SHADER(fault_comp)
ST_SHADER(rq_comp)
ST_SHADER(fill_vert)
ST_SHADER(rows_vert)
ST_SHADER(grid_vert)
ST_SHADER(color_frag)
ST_SHADER(xdata_frag)
#undef ST_SHADER
#define ST_SPV(n) n, n##_size

/* ---- tests ---- */
int t_xq_order(struct st_run *r);
int t_xq_chain(struct st_run *r);
int t_xq_exit(struct st_run *r);
int t_cr_concurrent(struct st_run *r);
int t_cr_pairs(struct st_run *r);
int t_render_first(struct st_run *r);
int t_render_tiler(struct st_run *r);
int t_compute_long(struct st_run *r);
int t_compute_starve(struct st_run *r);
int t_life_churn(struct st_run *r);
int t_life_fault(struct st_run *r);
int t_queue_exhaust(struct st_run *r);
int t_mem_limit(struct st_run *r);
int t_mem_churn(struct st_run *r);
int t_mem_exhaust(struct st_run *r);
int t_sync_dense(struct st_run *r);
int t_idle_wake(struct st_run *r);
int t_queue_depth(struct st_run *r);
int t_sparse_churn(struct st_run *r);
int t_rq_stress(struct st_run *r);
int t_mix_wedge(struct st_run *r);

/* Lanes of mix.wedge / compute.starve that other tests also use. Each runs until the budget of
 * `r` ends and records its own failures in `r`. */
int st_lane_render_heavy(struct st_run *r);
int st_lane_longalu(struct st_run *r, double target_ms);
int st_lane_queues(struct st_run *r, int nqueues, bool render);
/* Runs `n` lanes as concurrent child processes sharing r's remaining budget. Each lane's
 * verdict is folded into r. */
struct st_lane {
   const char *name;
   int (*fn)(struct st_run *r, void *arg);
   void *arg;
};
int st_run_lanes(struct st_run *r, const struct st_lane *lanes, int n);
