# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libaurora_lib_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_decode_core.h',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx3_cdm_direct_launch_semantic_core.h',
  'agx3_compute_program_core.c',
  'agx3_compute_program_core.h',
  'agx3_g17p_grid_helper_core.c',
  'agx3_g17p_grid_helper_core.h',
  'agx3_code_container_core.c',
  'agx3_code_container_core.h',
  'agx_ioto_core.c',
  'agx_ioto_core.h',
  'agx_ioto_load_core.c',
  'agx_ioto_load_core.h',
  'agx_ioto_preshader_core.c',
  'agx_ioto_preshader_core.h',
  'agx_ioto_profile_core.c',
  'agx_ioto_profile_core.h',
  'agx_ioto_program_core.c',
  'agx_ioto_program_core.h',
  'agx_ioto_state_core.c',
  'agx_ioto_state_core.h',
  'agx_g17p_draw_state_upload.h',
  'agx_rt_result.h',
  'agx_rt_ift.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_bo.c',
  'agx_border.c',
  'agx_chip.c',
  'agx_device.c',
  'agx_bytes.h',
  'agx_linker.c',
  'agx_bg_eot.c',
  'agx_bg_eot_key.c',
  'agx_tilebuffer.c',
  'agx_nir_lower_msaa.c',
  'agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_tilebuffer.c',
  'agx_nir_lower_uvs.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_nir_legacy_epilog.c',
  'agx_ppp.h',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal200_graphics_core.h',
  'agx_ppp_hal200_raster_core.c',
  'agx_ppp_hal200_raster_core.h',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.h',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_hal300_depth_stencil_core.h',
  'agx_ppp_decode_core.c',
  'agx_ppp_decode_core.h',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_viewport_payload_core.h',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_ppp_w_clamp_cull2_core.h',
  'agx_pbe_agx3_core.c',
  'agx_pbe_agx3_core.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_scratch.c',
  'agx_texture_side_entry.c',
  'agx_texture_side_entry.h',
  'agx_vdm_barrier_core.c',
  'agx_vdm_barrier_core.h',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_bind_tail_core.h',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_hal200_direct_draw_core.h',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_decode_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_index_list_core.h',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_ppp_state_core.h',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_link_core.h',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_stream_terminate_core.h',
  'agx_vdm_tessellate_core.c',
  'agx_vdm_tessellate_core.h',
  'agx_vdm_state_core.c',
  'agx_vdm_state_core.h',
  'agx_vdm_state_update_core.c',
  'agx_vdm_state_update_core.h',
  'agx_va.c',
  'agx_va_runtime_hole.c',
  'agx_va_runtime_hole.h',
  'pool.c',
)

aurora_lib_has_virtio = with_vdrm
if aurora_lib_has_virtio
  libaurora_lib_files += files('agx_device_virtio.c')
endif

libaurora_decode_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx_ioto_core.c',
  'agx_ppp_decode_core.c',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_vdm_barrier_core.c',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_state_core.c',
  'agx_vdm_state_update_core.c',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_tessellate_core.c',
  'decode.c',
)

libaurora_decode = static_library(
  'aurora_decode',
  [libaurora_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with : [libagx2_disasm, libagx3d_disasm],
  build_by_default : false,
)

libaurora_decode_shared = shared_library(
  'aurora_decode',
  [libaurora_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  build_by_default : with_tools.contains('aurora'),
  link_with : [libagx2_disasm, libagx3d_disasm],
)

if aurora_lib_has_virtio
  libaurora_lib_includes = [inc_aurora, inc_virtio_gpu, inc_virtio_vdrm]
  libaurora_lib_links = [libaurora_decode, libvdrm, libpoly_nir]
  libaurora_lib_deps = [
    dep_libdrm, dep_valgrind, idep_nir, idep_mesautil, idep_libagx,
    idep_libpoly,
  ]
else
  libaurora_lib_includes = [inc_aurora]
  libaurora_lib_links = [libaurora_decode, libpoly_nir]
  libaurora_lib_deps = [
    dep_valgrind, idep_nir, idep_mesautil, idep_libagx, idep_libpoly,
  ]
endif

libagx_shaders = custom_target(
  input : libagx_spv,
  output : ['libagx_shaders.h', 'libagx_shaders.c'],
  command : [prog_aurora_clc, libagx_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_libagx_shaders_h = declare_dependency(
  sources : [libagx_shaders],
  include_directories : include_directories('.'),
)

libaurora_lib = static_library(
  'aurora_lib',
  [libaurora_lib_files, libagx_shaders, agx_pack, sha1_h],
  include_directories : libaurora_lib_includes,
  c_args : [
    no_override_init_args,
    '-Wno-c2x-extensions',
  ],
  gnu_symbol_visibility : 'hidden',
  link_with: libaurora_lib_links,
  dependencies: libaurora_lib_deps,
  build_by_default : false,
)

if with_tests
  foreach rt_test : ['rt-bvh', 'rt-ift', 'scs-layout']
    test('aurora-' + rt_test,
      executable('aurora-' + rt_test,
        ['tests/test-' + rt_test + '.c', 'agx_scs_layout.c'],
        include_directories : include_directories('.'),
        c_args : ['-UNDEBUG'],
      ),
      suite : ['aurora'],
    )
  endforeach
  test(
    'aurora-chip-admission',
    executable(
      'aurora-chip-admission',
      files('tests/test-chip-admission.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libaurora_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['aurora'],
  )
  test(
    'aurora-i386-layout',
    executable(
      'aurora-i386-layout',
      files('tests/test-i386-layout.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['aurora'],
  )
  test(
    'aurora-usc-base-page',
    executable(
      'aurora-usc-base-page',
      files('tests/test-usc-base-page.c'),
      include_directories : [inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['aurora'],
  )
  test(
    'aurora-agx-chip',
    executable(
      'aurora-agx-chip',
      files(
        'agx_chip.c',
        'tests/test-chip.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-g17g-tile-format-gate',
    executable(
      'aurora-agx-g17g-tile-format-gate',
      files('tests/test-agx-g17g-tile-format-gate.c'),
      include_directories : [
        inc_include, inc_src, inc_aurora, libaurora_lib_includes,
        include_directories('../libagx'),
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx_shaders_h, idep_agx_pack,
      ],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-linker-container',
    executable(
      'aurora-agx-linker-container',
      files(
        'agx3_code_container_core.c',
        'tests/test-agx-linker-container.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-g17p-srgb-epilog',
    executable(
      'aurora-g17p-srgb-epilog',
      files('tests/test-agx-g17p-srgb-epilog.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-g17p-mrt-epilog',
    executable(
      'aurora-g17p-mrt-epilog',
      files('tests/test-agx-g17p-mrt-epilog.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-g17p-mrt-quad',
    executable(
      'aurora-g17p-mrt-quad',
      files('tests/test-agx-g17p-mrt-quad.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-g17p-mrt-mixed',
    executable(
      'aurora-g17p-mrt-mixed',
      files('tests/test-agx-g17p-mrt-mixed.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-va-runtime-hole',
    executable(
      'aurora-agx-va-runtime-hole',
      files(
        'agx_va_runtime_hole.c',
        'tests/test-agx-va-runtime-hole.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_mesautil],
      c_args : ['-UNDEBUG'],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-m5-va',
    executable(
      'aurora-agx-m5-va',
      files('tests/test-agx-m5-va.c'),
      include_directories : [inc_include, inc_src, inc_aurora],
      c_args : ['-UNDEBUG'],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-ioto-load-core',
    executable(
      'aurora-agx-ioto-load-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'tests/test-agx-ioto-load-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-ioto-preshader-core',
    executable(
      'aurora-agx-ioto-preshader-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'tests/test-agx-ioto-preshader-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx-ioto-program-core',
    executable(
      'aurora-agx-ioto-program-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_state_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_program_core.c',
        'tests/test-agx-ioto-program-core.c',
      ),
      dependencies : [idep_mesautil],
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-vdm-decode-core',
    executable(
      'aurora-vdm-decode-core',
      files(
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-vdm-decode-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-g18g-capture-vectors',
    executable(
      'aurora-g18g-capture-vectors',
      files(
        'agx3_cdm_decode_core.c',
        'agx_pbe_agx3_core.c',
        'agx_ppp_decode_core.c',
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'agx_ppp_hal300_depth_stencil_core.c',
        'agx_ppp_viewport_payload_core.c',
        'agx_ppp_w_clamp_cull2_core.c',
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-g18g-capture-vectors.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-ppp-hal300-g18g',
    executable(
      'aurora-ppp-hal300-g18g',
      files(
        'agx3_cdm_decode_core.c',
        'agx_pbe_agx3_core.c',
        'agx_ppp_decode_core.c',
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'agx_ppp_hal300_depth_stencil_core.c',
        'agx_ppp_hal300_graphics_core.c',
        'agx_ppp_viewport_payload_core.c',
        'agx_ppp_w_clamp_cull2_core.c',
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_mesh_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-ppp-hal300-g18g.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-agx3-compute-program-core',
    executable(
      'aurora-agx3-compute-program-core',
      files(
        'agx3_cdm_direct_launch_semantic_core.c',
        'agx3_compute_program_core.c',
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_program_core.c',
        'agx_ioto_state_core.c',
        'tests/test-agx3-compute-program-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-vdm-index-list-core',
    executable(
      'aurora-vdm-index-list-core',
      files(
        'agx_vdm_index_list_core.c',
        'tests/test-vdm-index-list-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-pbe-agx3-core',
    executable(
      'aurora-pbe-agx3-core',
      files(
        'agx_pbe_agx3_core.c',
        'tests/test-pbe-agx3-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-ppp-hal300-graphics-core',
    executable(
      'aurora-ppp-hal300-graphics-core',
      files(
        'agx_ppp_hal300_graphics_core.c',
        'agx_vdm_state_core.c',
        'tests/test-ppp-hal300-graphics-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'aurora-vdm-state-update-core',
    executable(
      'aurora-vdm-state-update-core',
      files(
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'tests/test-vdm-state-update-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_aurora],
    ),
    suite : ['aurora'],
  )

  test(
    'libaurora_tests',
    executable(
      'libaurora_tests',
      files(
        'tests/test-chip.cpp',
        'tests/test-packing.cpp',
        'tests/test-tilebuffer.cpp',
        'tests/test-g17p-msaa-epilog.cpp',
        'tests/test-m5-depth-export.cpp',
        'tests/test-primitive-varyings.cpp',
        '../vulkan/hk_nir_widen_16bit_varyings.c',
        'tests/test-dual-source-epilog.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [
        idep_gtest, idep_agx_pack, idep_mesautil, idep_nir,
      ],
      include_directories: [inc_aurora],
      link_with : [libaurora_lib, libaurora_compiler, libaurora_layout,
                   libagx3c_disasm],
    ),
    suite : ['aurora'],
    protocol : 'gtest',
  )
endif

if with_tests
  test('vbo_robustness', executable('vbo_robustness_test',
    files('tests/test-vbo-robustness.c'),
    include_directories : [inc_aurora, inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx_shaders_h],
    link_with : [libaurora_lib, libaurora_compiler, libaurora_layout]),
    suite : ['aurora'])
endif

if dep_iokit.found()
  libaurora_wrap = shared_library(
    'wrap',
    'wrap.c',
    include_directories : [inc_src, inc_aurora],
    dependencies : [idep_mesautil, dep_iokit],
    c_args : [no_override_init_args, '-Wno-missing-prototypes'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : with_tools.contains('aurora'),
    link_with: libaurora_decode,
  )
endif

if with_tests
  test('aurora-rt-payload-gen1', executable('aurora-rt-payload-gen1',
    'tests/test-rt-payload-gen1.c',
    c_args : ['-ffp-contract=off'],
    include_directories : include_directories('.'),
    dependencies : [dep_m]), suite : ['aurora'])
  test('agx3_m5_programs', executable('agx3_m5_programs',
    '../compiler/test/test-agx3-m5-program.c',
    include_directories: [inc_include, inc_src, inc_aurora],
    dependencies: [idep_mesautil, idep_nir],
    link_with: [libaurora_compiler, libaurora_lib]), suite: ['aurora'])
endif

if with_tests
  test('agx_rt_envelope', executable('agx_rt_envelope',
    'tests/test-rt-envelope.c',
    c_args: ['-ffp-contract=off'],
    include_directories: include_directories('.')), suite: ['aurora'])
  test('agx_rt_blas_gen3', executable('agx_rt_blas_gen3',
    'tests/test-rt-blas-gen3.c',
    include_directories: include_directories('.')), suite: ['aurora'])
  test('agx_rt_bvh', executable('agx_rt_bvh',
    'tests/test-rt-bvh.c',
    include_directories: include_directories('.')), suite: ['aurora'])
  test('agx_m5_layout', executable('agx_m5_layout',
    ['tests/test-agx-m5-layout.c', 'agx3_code_container_core.c'],
    include_directories: include_directories('.')), suite: ['aurora'])
endif

if with_tests
  test('agx3_m5_instructions', executable('agx3_m5_instructions',
    '../compiler/test/test-agx3-m5-instructions.c',
    include_directories: [inc_include, inc_src, inc_aurora],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libaurora_compiler, libaurora_lib]), suite: ['aurora'])
endif

if with_tests
  test('agx_scs_nir', executable('agx_scs_nir',
    '../compiler/test/test-scs-nir.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_aurora],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libaurora_compiler, libaurora_lib]), suite: ['aurora'])
endif

if with_tests
  test('agx3-m5-mesh-program', executable('agx3-m5-mesh-program',
    '../compiler/test/test-native-mesh-program.c',
    include_directories: [inc_include, inc_src, inc_aurora],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libaurora_compiler, libaurora_lib]), suite: ['aurora'])
endif

if with_tests
  test('agx_scs_query', executable('agx_scs_query',
    '../compiler/test/test-scs-query.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_aurora],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libaurora_compiler, libaurora_lib]), suite: ['aurora'])
endif

if with_tests
  test('agx_scs_layout', executable('agx_scs_layout',
    ['tests/test-scs-layout.c', 'agx_scs_layout.c'],
    c_args: ['-UNDEBUG'],
    include_directories: include_directories('.')), suite: ['aurora'])
endif

if with_tests
  test('agx_rt_ift', executable('agx_rt_ift', 'tests/test-rt-ift.c',
    c_args: ['-UNDEBUG'],
    include_directories: include_directories('.')), suite: ['aurora'])
endif
