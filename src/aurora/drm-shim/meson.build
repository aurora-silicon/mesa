# Copyright 2019 Broadcom
# SPDX-License-Identifier: MIT

libaurora_noop_drm_shim = shared_library(
  'aurora_noop_drm_shim',
  ['aurora_noop.c', 'aurora_g15s_stream_gate.c',
   'aurora_g17p_stream_validator.c', 'aurora_m5_submit_gate.c'],
  objects : libaurora_decode.extract_objects(
    'agx3_cdm_decode_core.c',
    'agx_ioto_core.c',
    'agx_ppp_hal200_graphics_core.c',
    'agx_ppp_viewport_payload_core.c',
    'agx_ppp_w_clamp_cull2_core.c',
    'agx_vdm_barrier_core.c',
    'agx_vdm_decode_core.c',
    'agx_vdm_hal200_bind_tail_core.c',
    'agx_vdm_index_list_core.c',
    'agx_vdm_mesh_core.c',
    'agx_vdm_ppp_state_core.c',
    'agx_vdm_state_core.c',
    'agx_vdm_state_update_core.c',
    'agx_vdm_stream_link_core.c',
    'agx_vdm_stream_terminate_core.c',
    'agx_vdm_tessellate_core.c',
  ),
  include_directories: [inc_include, inc_src, inc_aurora],
  dependencies: [dep_drm_shim, dep_valgrind],
  link_with : [libagx3b_disasm],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_tests
  aurora_vm_limits_test = executable(
    'aurora-vm-limits', 'test-vm-limits.c',
    include_directories : [inc_include],
    dependencies : [dep_libdrm],
  )
  foreach hal : ['legacy', 'hal200']
    test(
      'aurora-vm-limits-' + hal, prog_python,
      args : [files('../../gallium/drivers/aurora/tests/run-shim.py'),
              libaurora_noop_drm_shim, aurora_vm_limits_test],
      env : ['AURORA_SHIM_CHIP=g15g',
             'AURORA_SHIM_G15G_HAL200=' + (hal == 'hal200' ? '1' : '0')],
      suite : ['aurora'],
    )
  endforeach
endif
