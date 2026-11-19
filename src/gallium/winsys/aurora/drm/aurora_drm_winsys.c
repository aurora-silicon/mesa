/*
 * Copyright 2014 Broadcom
 * Copyright 2018 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "util/format/u_format.h"
#include "util/os_file.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_screen.h"

#include "aurora/agx_public.h"
#include "drm-uapi/drm.h"
#include "renderonly/renderonly.h"
#include "aurora_drm_public.h"

static struct pipe_screen *
aurora_screen_create(int fd, const struct pipe_screen_config *config,
                    struct renderonly *ro)
{
   return agx_screen_create(fd, ro, config);
}

struct pipe_screen *
aurora_drm_screen_create(int fd, const struct pipe_screen_config *config)
{
   return u_pipe_screen_lookup_or_create(os_dupfd_cloexec(fd), config, NULL,
                                         aurora_screen_create);
}

struct pipe_screen *
aurora_drm_screen_create_renderonly(int fd, struct renderonly *ro,
                                   const struct pipe_screen_config *config)
{
   return u_pipe_screen_lookup_or_create(os_dupfd_cloexec(fd), config, ro,
                                         aurora_screen_create);
}
