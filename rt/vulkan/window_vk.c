// SPDX-License-Identifier: GPL-2.0-only
/* Optional windowed mode (--window): a Wayland xdg-shell toplevel with a Vulkan swapchain. Each
 * frame is the benchmark frame followed by a blit of the composite output into the swapchain image.
 * For viewing and tweaking only; benchmark records come from the headless mode. */
#include "rtwin.h"
#include "../common/util.h"
#include "xdg-shell-client-protocol.h"
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_wayland.h>
#include <wayland-client.h>

struct rt_window {
   struct wl_display *dpy;
   struct wl_registry *reg;
   struct wl_compositor *comp;
   struct xdg_wm_base *wm;
   struct wl_surface *surf;
   struct xdg_surface *xs;
   struct xdg_toplevel *top;
   int configured, closed;
   vk_ctx *c;
   VkSurfaceKHR vs;
   VkSwapchainKHR sc;
   uint32_t nimg, w, h;
   VkImage img[16];
   VkSemaphore acq[RT_WIN_FRAMES], done[16];
   VkFence fence[RT_WIN_FRAMES];
   VkCommandPool pool[RT_WIN_FRAMES];
   VkCommandBuffer cb[RT_WIN_FRAMES];
};

static void wm_ping(void *d, struct xdg_wm_base *wm, uint32_t serial) { (void)d; xdg_wm_base_pong(wm, serial); }
static const struct xdg_wm_base_listener wm_listener = {wm_ping};
static void xs_configure(void *d, struct xdg_surface *xs, uint32_t serial)
{
   struct rt_window *w = d;
   xdg_surface_ack_configure(xs, serial);
   w->configured = 1;
}
static const struct xdg_surface_listener xs_listener = {xs_configure};
static void top_configure(void *d, struct xdg_toplevel *t, int32_t w, int32_t h, struct wl_array *s)
{
   (void)d; (void)t; (void)w; (void)h; (void)s;
}
static void top_close(void *d, struct xdg_toplevel *t) { (void)t; ((struct rt_window *)d)->closed = 1; }
static void top_bounds(void *d, struct xdg_toplevel *t, int32_t w, int32_t h) { (void)d; (void)t; (void)w; (void)h; }
static void top_caps(void *d, struct xdg_toplevel *t, struct wl_array *a) { (void)d; (void)t; (void)a; }
static const struct xdg_toplevel_listener top_listener = {top_configure, top_close, top_bounds, top_caps};

static void reg_global(void *d, struct wl_registry *r, uint32_t name, const char *iface, uint32_t ver)
{
   struct rt_window *w = d;
   if (!strcmp(iface, wl_compositor_interface.name))
      w->comp = wl_registry_bind(r, name, &wl_compositor_interface, ver < 4 ? ver : 4);
   else if (!strcmp(iface, xdg_wm_base_interface.name)) {
      w->wm = wl_registry_bind(r, name, &xdg_wm_base_interface, 1);
      xdg_wm_base_add_listener(w->wm, &wm_listener, w);
   }
}
static void reg_remove(void *d, struct wl_registry *r, uint32_t name) { (void)d; (void)r; (void)name; }
static const struct wl_registry_listener reg_listener = {reg_global, reg_remove};

rt_window *rt_window_open(vk_ctx *c, uint32_t width, uint32_t height, const char *title)
{
   rt_window *w = calloc(1, sizeof(*w));
   w->c = c;
   w->w = width;
   w->h = height;
   w->dpy = wl_display_connect(NULL);
   if (!w->dpy) { fprintf(stderr, "window: no Wayland display (WAYLAND_DISPLAY)\n"); free(w); return NULL; }
   w->reg = wl_display_get_registry(w->dpy);
   wl_registry_add_listener(w->reg, &reg_listener, w);
   wl_display_roundtrip(w->dpy);
   if (!w->comp || !w->wm) { fprintf(stderr, "window: compositor lacks xdg-shell\n"); rt_window_close(w); return NULL; }
   w->surf = wl_compositor_create_surface(w->comp);
   w->xs = xdg_wm_base_get_xdg_surface(w->wm, w->surf);
   xdg_surface_add_listener(w->xs, &xs_listener, w);
   w->top = xdg_surface_get_toplevel(w->xs);
   xdg_toplevel_add_listener(w->top, &top_listener, w);
   xdg_toplevel_set_title(w->top, title);
   xdg_toplevel_set_app_id(w->top, "gpubench-rt");
   wl_surface_commit(w->surf);
   while (!w->configured && wl_display_dispatch(w->dpy) >= 0) {
   }
   VkWaylandSurfaceCreateInfoKHR si = {.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                                       .display = w->dpy, .surface = w->surf};
   PFN_vkCreateWaylandSurfaceKHR cws = (void *)vkGetInstanceProcAddr(c->inst, "vkCreateWaylandSurfaceKHR");
   if (!cws || cws(c->inst, &si, NULL, &w->vs) != VK_SUCCESS) { rt_window_close(w); return NULL; }
   VkBool32 ok = VK_FALSE;
   vkGetPhysicalDeviceSurfaceSupportKHR(c->phys, c->qfam, w->vs, &ok);
   if (!ok) { fprintf(stderr, "window: queue cannot present\n"); rt_window_close(w); return NULL; }
   VkSurfaceCapabilitiesKHR caps;
   vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, w->vs, &caps);
   uint32_t nf = 0;
   vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, w->vs, &nf, NULL);
   VkSurfaceFormatKHR *fm = calloc(nf ? nf : 1, sizeof(*fm));
   vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, w->vs, &nf, fm);
   VkSurfaceFormatKHR f = fm[0];
   for (uint32_t i = 0; i < nf; i++)
      if (fm[i].format == VK_FORMAT_B8G8R8A8_UNORM || fm[i].format == VK_FORMAT_R8G8B8A8_UNORM) { f = fm[i]; break; }
   free(fm);
   uint32_t np = 0;
   vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, w->vs, &np, NULL);
   VkPresentModeKHR pm[8], mode = VK_PRESENT_MODE_FIFO_KHR;
   if (np > 8) np = 8;
   vkGetPhysicalDeviceSurfacePresentModesKHR(c->phys, w->vs, &np, pm);
   for (uint32_t i = 0; i < np; i++) if (pm[i] == VK_PRESENT_MODE_MAILBOX_KHR) mode = pm[i];
   uint32_t n = caps.minImageCount + 1;
   if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;
   VkSwapchainCreateInfoKHR sci = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                                   .surface = w->vs,
                                   .minImageCount = n,
                                   .imageFormat = f.format,
                                   .imageColorSpace = f.colorSpace,
                                   .imageExtent = {width, height},
                                   .imageArrayLayers = 1,
                                   .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
                                   .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                                   .presentMode = mode,
                                   .clipped = VK_TRUE};
   if (vkCreateSwapchainKHR(c->dev, &sci, NULL, &w->sc) != VK_SUCCESS) { rt_window_close(w); return NULL; }
   vkGetSwapchainImagesKHR(c->dev, w->sc, &w->nimg, NULL);
   if (w->nimg > 16) w->nimg = 16;
   vkGetSwapchainImagesKHR(c->dev, w->sc, &w->nimg, w->img);
   VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
   VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
   for (uint32_t i = 0; i < w->nimg; i++) vkCreateSemaphore(c->dev, &semi, NULL, &w->done[i]);
   for (int i = 0; i < RT_WIN_FRAMES; i++) {
      vkCreateSemaphore(c->dev, &semi, NULL, &w->acq[i]);
      vkCreateFence(c->dev, &fi, NULL, &w->fence[i]);
      VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = c->qfam};
      vkCreateCommandPool(c->dev, &pci, NULL, &w->pool[i]);
      VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = w->pool[i],
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
      vkAllocateCommandBuffers(c->dev, &ai, &w->cb[i]);
   }
   fprintf(stderr, "window: %ux%u, %u swapchain images, %s\n", width, height, w->nimg,
           mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "fifo");
   return w;
}

int rt_window_run(rt_window *w, uint32_t seconds, rt_window_record_fn record, void *user, VkImage src)
{
   vk_ctx *c = w->c;
   uint64_t t0 = rt_now_ns(), last = t0, end = t0 + (uint64_t)seconds * 1000000000ull;
   uint32_t frames = 0, total = 0;
   for (uint32_t i = 0; !w->closed && (!seconds || rt_now_ns() < end); i++) {
      uint32_t f = i % RT_WIN_FRAMES;
      vkWaitForFences(c->dev, 1, &w->fence[f], VK_TRUE, UINT64_MAX);
      uint32_t idx;
      VkResult r = vkAcquireNextImageKHR(c->dev, w->sc, UINT64_MAX, w->acq[f], VK_NULL_HANDLE, &idx);
      if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { fprintf(stderr, "window: acquire %d\n", r); break; }
      vkResetFences(c->dev, 1, &w->fence[f]);
      vkResetCommandPool(c->dev, w->pool[f], 0);
      VkCommandBuffer cb = w->cb[f];
      VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
      vkBeginCommandBuffer(cb, &bi);
      record(user, f, cb);
      vk_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      vk_image_barrier(cb, w->img[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, 0,
                       VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkImageBlit rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {(int32_t)w->w, (int32_t)w->h, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{0, 0, 0}, {(int32_t)w->w, (int32_t)w->h, 1}}};
      vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, w->img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg,
                     VK_FILTER_NEAREST);
      vk_image_barrier(cb, w->img[idx], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_NONE, 0);
      vkEndCommandBuffer(cb);
      VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &w->acq[f],
                         .pWaitDstStageMask = &ws, .commandBufferCount = 1, .pCommandBuffers = &cb,
                         .signalSemaphoreCount = 1, .pSignalSemaphores = &w->done[idx]};
      if (vkQueueSubmit(c->queue, 1, &si, w->fence[f]) != VK_SUCCESS) break;
      VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                             .pWaitSemaphores = &w->done[idx], .swapchainCount = 1, .pSwapchains = &w->sc,
                             .pImageIndices = &idx};
      vkQueuePresentKHR(c->queue, &pi);
      wl_display_dispatch_pending(w->dpy);
      wl_display_flush(w->dpy);
      frames++;
      total++;
      uint64_t now = rt_now_ns();
      if (now - last >= 1000000000ull) {
         fprintf(stderr, "window: %.1f fps\n", frames / ((now - last) / 1e9));
         frames = 0;
         last = now;
      }
   }
   vkDeviceWaitIdle(c->dev);
   return (int)total;
}

void rt_window_close(rt_window *w)
{
   if (!w) return;
   vk_ctx *c = w->c;
   if (c && c->dev) {
      vkDeviceWaitIdle(c->dev);
      for (int i = 0; i < RT_WIN_FRAMES; i++) {
         if (w->acq[i]) vkDestroySemaphore(c->dev, w->acq[i], NULL);
         if (w->fence[i]) vkDestroyFence(c->dev, w->fence[i], NULL);
         if (w->pool[i]) vkDestroyCommandPool(c->dev, w->pool[i], NULL);
      }
      for (uint32_t i = 0; i < w->nimg; i++) if (w->done[i]) vkDestroySemaphore(c->dev, w->done[i], NULL);
      if (w->sc) vkDestroySwapchainKHR(c->dev, w->sc, NULL);
   }
   if (w->vs) vkDestroySurfaceKHR(c->inst, w->vs, NULL);
   if (w->top) xdg_toplevel_destroy(w->top);
   if (w->xs) xdg_surface_destroy(w->xs);
   if (w->surf) wl_surface_destroy(w->surf);
   if (w->wm) xdg_wm_base_destroy(w->wm);
   if (w->comp) wl_compositor_destroy(w->comp);
   if (w->reg) wl_registry_destroy(w->reg);
   if (w->dpy) wl_display_disconnect(w->dpy);
   free(w);
}
