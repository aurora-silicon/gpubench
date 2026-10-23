// SPDX-License-Identifier: GPL-2.0-only
/* Optional Wayland window for gpubench-rt-vk (--window). */
#pragma once
#include "vkutil.h"

#define RT_WIN_FRAMES 2
typedef struct rt_window rt_window;
/* Records one benchmark frame into cb (already begun); slot < RT_WIN_FRAMES. */
typedef void (*rt_window_record_fn)(void *user, uint32_t slot, VkCommandBuffer cb);

rt_window *rt_window_open(vk_ctx *c, uint32_t width, uint32_t height, const char *title);
/* Renders and presents until the window closes or `seconds` pass (0 = until closed); src is the
 * composite output (GENERAL layout, same size as the window). Returns the frame count. */
int rt_window_run(rt_window *w, uint32_t seconds, rt_window_record_fn record, void *user, VkImage src);
void rt_window_close(rt_window *w);
