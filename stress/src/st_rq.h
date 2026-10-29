// SPDX-License-Identifier: GPL-2.0-only
/* Acceleration-structure entry points, loaded per device when ray query is enabled. */
#pragma once
#include "st.h"

#include <stdlib.h>

struct st_rq_fns {
   PFN_vkCreateAccelerationStructureKHR create;
   PFN_vkDestroyAccelerationStructureKHR destroy;
   PFN_vkGetAccelerationStructureBuildSizesKHR sizes;
   PFN_vkCmdBuildAccelerationStructuresKHR build;
   PFN_vkGetAccelerationStructureDeviceAddressKHR addr;
};

static inline VkResult
st_rq_load(struct st_dev *d)
{
   struct st_rq_fns *f = calloc(1, sizeof(*f));
   if (!f)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   f->create = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(d->dev, "vkCreateAccelerationStructureKHR");
   f->destroy = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(d->dev, "vkDestroyAccelerationStructureKHR");
   f->sizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
      d->dev, "vkGetAccelerationStructureBuildSizesKHR");
   f->build = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(d->dev, "vkCmdBuildAccelerationStructuresKHR");
   f->addr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(
      d->dev, "vkGetAccelerationStructureDeviceAddressKHR");
   d->rqf = f;
   if (!f->create || !f->destroy || !f->sizes || !f->build || !f->addr)
      return VK_ERROR_EXTENSION_NOT_PRESENT;
   return VK_SUCCESS;
}
