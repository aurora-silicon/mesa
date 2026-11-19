# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

if get_option('precomp-compiler') == 'system'
  prog_aurora_clc = find_program('aurora_clc', native : true)
else
  prog_aurora_clc = executable(
    'aurora_clc',
    ['aurora_clc.c'],
    link_with : [libaurora_compiler],
    include_directories : [inc_include, inc_src],
    c_args : [pre_args, no_override_init_args],
    link_args : [ld_args_build_id],
    dependencies : [idep_vtn, idep_nir, idep_mesautil],
    # If we can run host binaries directly, just build aurora_clc for the host.
    # Most commonly this happens when doing a cross compile from an x86_64 build
    # machine to an x86 host
    native : not meson.can_run_host_binaries(),
    install : get_option('install-precomp-compiler'),
  )
endif
