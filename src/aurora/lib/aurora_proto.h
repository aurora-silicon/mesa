/*
 * Copyright 2024 Sergio Lopez
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef AURORA_PROTO_H_
#define AURORA_PROTO_H_

#include "drm-uapi/aurora_drm.h"

/**
 * Defines the layout of shmem buffer used for host->guest communication.
 */
struct aurora_shmem {
   struct vdrm_shmem base;

   /**
    * Counter that is incremented on asynchronous errors, like SUBMIT
    * or GEM_NEW failures.  The guest should treat errors as context-
    * lost.
    */
   uint32_t async_error;

   /**
    * Counter that is incremented on global fault (see MSM_PARAM_FAULTS)
    */
   uint32_t global_faults;
};
DEFINE_CAST(vdrm_shmem, aurora_shmem)

/*
 * Possible cmd types for "command stream", ie. payload of EXECBUF ioctl:
 */
enum aurora_ccmd {
   AURORA_CCMD_NOP = 1, /* No payload, can be used to sync with host */
   AURORA_CCMD_IOCTL_SIMPLE,
   AURORA_CCMD_GET_PARAMS,
   AURORA_CCMD_GEM_NEW,
   AURORA_CCMD_VM_BIND,
   AURORA_CCMD_SUBMIT,
   AURORA_CCMD_GEM_BIND_OBJECT,
};

#define AURORA_CCMD(_cmd, _len)                                                 \
   (struct vdrm_ccmd_req)                                                      \
   {                                                                           \
      .cmd = AURORA_CCMD_##_cmd, .len = (_len),                                 \
   }

/*
 * AURORA_CCMD_NOP
 */
struct aurora_ccmd_nop_req {
   struct vdrm_ccmd_req hdr;
};

/*
 * AURORA_CCMD_IOCTL_SIMPLE
 *
 * Forward simple/flat IOC_RW or IOC_W ioctls.  Limited ioctls are supported.
 */
struct aurora_ccmd_ioctl_simple_req {
   struct vdrm_ccmd_req hdr;

   uint32_t cmd;
   uint8_t payload[];
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_ioctl_simple_req)

struct aurora_ccmd_ioctl_simple_rsp {
   struct vdrm_ccmd_rsp hdr;

   /* ioctl return value, interrupted syscalls are handled on the host without
    * returning to the guest.
    */
   int32_t ret;

   /* The output payload for IOC_RW ioctls, the payload is the same size as
    * aurora_context_cmd_ioctl_simple_req.
    *
    * For IOC_W ioctls (userspace writes, kernel reads) this is zero length.
    */
   uint8_t payload[];
};

struct aurora_ccmd_get_params_req {
   struct vdrm_ccmd_req hdr;
   struct drm_aurora_get_params params;
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_get_params_req)

struct aurora_ccmd_get_params_rsp {
   struct vdrm_ccmd_rsp hdr;
   int32_t ret;
   uint8_t payload[];
};

struct aurora_ccmd_gem_new_req {
   struct vdrm_ccmd_req hdr;
   uint32_t flags;
   uint32_t bind_flags;
   uint32_t vm_id;
   uint32_t blob_id;
   uint64_t size;
   uint64_t addr;
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_gem_new_req)

struct aurora_ccmd_vm_bind_req {
   struct vdrm_ccmd_req hdr;
   uint32_t vm_id;
   uint32_t stride;
   uint32_t count;
   uint8_t payload[];
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_vm_bind_req)

struct aurora_ccmd_gem_bind_object_req {
   struct vdrm_ccmd_req hdr;
   struct drm_aurora_gem_bind_object bind;
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_gem_bind_object_req)

struct aurora_ccmd_gem_bind_object_rsp {
   struct vdrm_ccmd_rsp hdr;
   int32_t ret;
   uint32_t object_handle;
};

#define AURORA_EXTRES_READ  0x01
#define AURORA_EXTRES_WRITE 0x02

struct aurora_ccmd_submit_res {
   uint32_t res_id;
   uint32_t flags;
};

struct aurora_ccmd_submit_req {
   struct vdrm_ccmd_req hdr;
   uint32_t flags;
   uint32_t queue_id;
   uint32_t cmdbuf_size;
   uint32_t extres_count;

   uint8_t payload[];
};
DEFINE_CAST(vdrm_ccmd_req, aurora_ccmd_submit_req)

#endif // AURORA_PROTO_H_
