# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT
brl_files = files(
  'brl_barrier.c',
  'brl_barrier.h',
  'brl_acceleration_structure.c',
  'brl_m5_acceleration_structure.c',
  'brl_host_as.c',
  'brl_acceleration_structure.h',
  'brl_buffer.c',
  'brl_buffer.h',
  'brl_buffer_view.c',
  'brl_bvh_depth_core.c',
  'brl_bvh_depth_core.h',
  'brl_2d_stencil.h',
  'brl_byte_stencil.h',
  'brl_buffer_view.h',
  'brl_bvh.h',
  'brl_cmd_buffer.c',
  'brl_cmd_upload.c',
  'brl_cmd_buffer.h',
  'brl_conditional_render_core.h',
  'brl_cmd_lifetime.c',
  'brl_cs_root_size.h',
  'brl_cmd_clear.c',
  'brl_cmd_dispatch.c',
  'brl_gen1_query_state.c',
  'brl_cmd_draw.c',
  'brl_cmd_meta.c',
  'brl_cmd_trace_rays.c',
  'brl_cmd_pool.c',
  'brl_cmd_pool.h',
  'brl_descriptor_set.h',
  'brl_descriptor_set.c',
  'brl_descriptor_set_layout.c',
  'brl_descriptor_set_layout_hash.c',
  'brl_descriptor_set_layout.h',
  'brl_descriptor_table.c',
  'brl_descriptor_table.h',
  'brl_compute_kick.c',
  'brl_compute_kick.h',
  'brl_device.c',
  'brl_device.h',
  'brl_device_memory.c',
  'brl_device_memory.h',
  'brl_memory_budget.c',
  'brl_memory_budget.h',
  'brl_host_alias.c',
  'brl_host_alias.h',
  'brl_event.c',
  'brl_event.h',
  'brl_format.c',
  'brl_g17p_meta_copy_core.c',
  'brl_ioto_arena.c',
  'brl_ioto_arena_core.c',
  'brl_ioto_arena_core.h',
  'brl_g17p_meta_copy_core.h',
  'brl_agx3_ioto_program_core.c',
  'brl_agx3_ioto_program_core.h',
  'brl_agx3_vdm_decode.c',
  'brl_agx3_vdm_decode.h',
  'brl_ppp_state.h',
  'brl_image.c',
  'brl_image.h',
  'brl_image_layout_core.c',
  'brl_image_layout_core.h',
  'brl_identity_copy4.h',
  'brl_nir_wgid.h',
  'brl_image_view.c',
  'brl_nir_copy_identity4.c',
  'brl_nir_tensor_ops.c',
  'brl_nir_tensor_ops.h',
  'brl_nir_match_pointwise_clamp.c',
  'brl_image_view.h',
  'brl_instance.c',
  'brl_instance.h',
  'brl_nir_forward_image_stores.c',
  'brl_nir_forward_image_stores.h',
  'brl_nir_lower_descriptors.c',
  'brl_nir_lower_gs.c',
  'brl_nir_lower_tes_rq.c',
  'brl_nir_lower_gs.h',
  'brl_nir_vertex_preamble.c',
  'brl_nir_vertex_preamble.h',
  'brl_nir_root_size.c',
  'brl_nir_tile_2d_stencil.c',
  'brl_nir_tile_byte_stencil.c',
  'brl_nir_opt_compute_kernels.c',
  'brl_nir_lower_ray_queries.c',
  'brl_nir_lower_robust_image.c',
  'brl_nir_lower_robust_image.h',
  'brl_nir_ray_tracing.c',
  'brl_ray_tracing_core.c',
  'brl_rt_inlining.c',
  'brl_ray_tracing_core.h',
  'brl_ray_tracing_pipeline.c',
  'brl_ray_tracing_pipeline.h',
  'brl_nir_ray_tracing.h',
  'brl_nir_widen_16bit_varyings.c',
  'brl_nir_widen_16bit_varyings.h',
  'brl_native_descriptor.h',
  'brl_native_vertex_template.h',
  'brl_native_descriptor_pack.c',
  'brl_native_descriptor_pack.h',
  'brl_native_sampler.c',
  'brl_native_sampler.h',
  'brl_physical_device.c',
  'brl_physical_device.h',
  'brl_pipeline_statistics_core.c',
  'brl_pipeline_statistics_core.h',
  'brl_private.h',
  'brl_query_oq.c',
  'brl_query_pool.c',
  'brl_query_reset.c',
  'brl_query_pool.h',
  'brl_drm_render_core.c',
  'brl_drm_render_core.h',
  'brl_execution_allowance.h',
  'brl_pack.c',
  'brl_pack.h',
  'brl_inline_blend_layout.h',
  'brl_partial_mrt_hsr.h',
  'brl_queue.c',
  'brl_queue.h',
  'brl_queue_flush.h',
  'brl_frame_anatomy.c',
  'brl_frame_anatomy.h',
  'brl_sampler.c',
  'brl_sampler.h',
  'brl_shader.c',
  'brl_nir_lower_image_view_min_lod.c',
  'brl_shader.h',
  'brl_sync.c',
  'brl_sync.h',
  'brl_tess_batch.c',
  'brl_tess_batch.h',
  'brl_wsi.c',
  'brl_wsi.h'
)

brl_drirc = custom_target(
  'brl_drirc',
  input  : [files('brl_drirc_gen.py'), drirc_gen, '00-borealis-defaults.conf'],
  output : ['brl_drirc.c', 'brl_drirc.h'],
  command : [
    prog_python, '@INPUT0@',
    '--import-path', join_paths(dir_source_root, 'src/util'),
    '--drirc-src', '@OUTPUT0@',
    '--drirc-hdr', '@OUTPUT1@',
    '--validate', '@INPUT2@',
  ],
)

brl_entrypoints = custom_target(
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['brl_entrypoints.h', 'brl_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'brl',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

if with_tests
  test('brl_execution_allowance',
    executable('brl_execution_allowance_test',
      files('tests/test-brl-execution-allowance.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ]),
    suite : ['asahi'])

  test('brl_inline_blend_layout',
    executable('brl_inline_blend_layout_test',
      files('tests/test-brl-inline-blend-layout.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi,
      ],
      dependencies : [idep_nir, idep_mesautil]),
    suite : ['asahi'])

  test('brl_partial_mrt_hsr',
    executable('brl_partial_mrt_hsr_test',
      files('tests/test-brl-partial-mrt-hsr.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout]),
    suite : ['asahi'])

  test('brl_memory_budget', executable('brl_memory_budget_test',
    files('tests/test-brl-memory-budget.c', 'brl_memory_budget.c'),
    include_directories : include_directories('.')),
    suite : ['asahi'])
  test('brl_host_alias', executable('brl_host_alias_test',
    files('tests/test-brl-host-alias.c', 'brl_host_alias.c'),
    include_directories : include_directories('.')),
    suite : ['asahi'])
  test(
    'brl_barrier',
    executable(
      'brl_barrier_test',
      files('tests/test-brl-barrier.c', 'brl_barrier.c'),
      include_directories : [include_directories('.'), inc_include],
    ),
    suite : ['asahi'],
  )
  test(
    'brl_g17p_storage_image',
    executable(
      'brl_g17p_storage_image_test',
      files('tests/test-brl-g17p-storage-image.c'),
      include_directories : include_directories('.'),
    ),
    suite : ['asahi'],
  )
  test(
    'brl_pass_merge',
    executable(
      'brl_pass_merge_test',
      files('tests/test-brl-pass-merge.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_gen1_query_state',
    executable(
      'brl_gen1_query_state_test',
      files('tests/test-brl-gen1-query-state.c', 'brl_gen1_query_state.c',
            '../lib/agx_scs_layout.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test('brl_image_store_forward', executable('brl_image_store_forward_test',
    files('tests/test-brl-image-store-forward.c', 'brl_nir_forward_image_stores.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil]), suite : ['asahi'])

  test('brl_root_size', executable('brl_root_size_test',
    files('tests/test-brl-root-size.c', 'brl_nir_root_size.c'),
    include_directories : [include_directories('.'), inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil]), suite : ['asahi'])

  test('brl_vdm_indirect_count',
    executable('brl_vdm_indirect_count_test',
      files('tests/test-brl-vdm-indirect-count.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil, idep_agx_pack],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden'),
    suite : ['asahi'])
  # Unused recording functions require section garbage collection so the
  # test can link the production resolver without the rest of the driver.
  if ld_args_gc_sections.length() > 0
    test(
      'brl_tess_batch',
      executable(
        'brl_tess_batch_test',
        files('tests/test-brl-tess-batch.c', 'brl_tess_batch.c',
              'brl_drm_render_core.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )
    test(
      'brl_query_reset',
      executable(
        'brl_query_reset_test',
        files('tests/test-brl-query-reset.c', 'brl_query_reset.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # Exercise the real resolver; unused command-recording entry points in
    # brl_sync.c need no device implementation in this CPU-only executable.
    test(
      'brl_sync_prior',
      executable(
        'brl_sync_prior_test',
        files('tests/test-brl-sync-prior.c', 'brl_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # Timestamp carriers, copies and their launches through the real
    # brl_sync recorder, on CPU-allocated streams; no device or GPU.
    test(
      'brl_sync_stamps',
      executable(
        'brl_sync_stamps_test',
        files('tests/test-brl-sync-stamps.c', 'brl_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # Host availability deferred to the end of the submission, resolved
    # through the real brl_sync resolver; no device or GPU.
    test(
      'brl_host_flush',
      executable(
        'brl_host_flush_test',
        files('tests/test-brl-host-flush.c', 'brl_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # Submit packing: the member check, the planner and the dependency fold,
    # resolved through the real brl_sync resolver; no device or GPU.
    test(
      'brl_pack',
      executable(
        'brl_pack_test',
        files('tests/test-brl-pack.c', 'brl_pack.c', 'brl_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

    # The deferred occlusion-availability resolver against mock control
    # streams, with the real brl_sync stream bookkeeping it drops streams from.
    test(
      'brl_query_oq',
      executable(
        'brl_query_oq_test',
        files('tests/test-brl-query-oq.c', 'brl_query_oq.c', 'brl_sync.c',
              'brl_event.c'),
        brl_entrypoints[0],
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

  endif

  test(
    'brl_root_fragment',
    executable(
      'brl_root_fragment_test',
      files('tests/test-brl-root-fragment.c'),
      include_directories : [
        include_directories('.'),
        inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
        inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir, idep_mesautil, idep_libagx,
        idep_libagx_shaders_h, idep_agx_pack,
        idep_vulkan_runtime_headers, idep_vulkan_util_headers,
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_render_resource_span',
    executable(
      'brl_render_resource_span_test',
      files('tests/test-brl-render-resource-span.c', 'brl_drm_render_core.c'),
      include_directories : [include_directories('.'), inc_include],
      c_args : [c_msvc_compat_args],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_query_program',
    executable(
      'brl_query_program_test',
      files('tests/test-brl-query-program.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [
        idep_nir_headers, idep_mesautil, idep_libagx, idep_libagx_shaders_h,
      ],
      link_with : [libasahi_lib],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_gen1_as_precomp',
    executable(
      'brl_gen1_as_precomp_test',
      files('tests/test-brl-gen1-as-precomp.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi,
      ],
      dependencies : [
        idep_nir_headers, idep_mesautil, idep_libagx, idep_libagx_shaders_h,
      ],
      link_with : [libasahi_lib],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_query_copy',
    executable(
      'brl_query_copy_test',
      files('tests/test-brl-query-copy.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_drm_render_core',
    executable(
      'brl_drm_render_core_test',
      files('tests/test-brl-drm-render-core.c',
            'brl_drm_render_core.c'),
      include_directories : [inc_include, inc_src],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_agx3_ioto_program_core',
    executable(
      'brl_agx3_ioto_program_core_test',
      files('tests/test-brl-agx3-ioto-program-core.c',
            'brl_agx3_ioto_program_core.c',
            '../compiler/agx3_compute_program.c',
            '../compiler/agx3_graphics_program.c',
            '../compiler/agx3_m5_graphics_program.c',
            '../compiler/agx3_copy_core.c',
            '../compiler/agx3_ioto_uniform_program.c',
            '../compiler/agx3_ioto_load_packer.c',
            '../compiler/agx3_cprog_input.c',
            '../compiler/agx3_static_spill_core.c',
            '../compiler/agx3_mem_core.c',
            '../compiler/agx3_binder.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib, libasahi_layout, libagx3_compose],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_agx3_vdm_decode',
    executable(
      'brl_agx3_vdm_decode_test',
      files('tests/test-brl-agx3-vdm-decode.c',
            'brl_agx3_vdm_decode.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_zls_stride',
    executable(
      'brl_zls_stride_test',
      files('tests/test-brl-zls-stride.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_sampler_index',
    executable(
      'brl_sampler_index_test',
      files('tests/test-brl-sampler-index.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

    executable(
      'g17p-render-only',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )
  test(
    'brl_g17p_compression',
    executable(
      'brl_g17p_compression_test',
      files('tests/test-brl-g17p-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g17p_decompress',
    executable(
      'brl_g17p_decompress_test',
      files('tests/test-brl-g17p-decompress.c', 'brl_native_descriptor_pack.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux, inc_include,
        inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir_headers, idep_mesautil, idep_libagx, idep_libagx_shaders_h,
        idep_agx_pack, idep_vulkan_util, idep_vulkan_runtime_headers,
      ],
      link_with : [libasahi_lib, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g16s_compression',
    executable(
      'brl_g16s_compression_test',
      files('tests/test-brl-g16s-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g17p_zls',
    executable(
      'brl_g17p_zls_test',
      files('tests/test-brl-g17p-zls.c', 'brl_drm_render_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_nir, idep_libagx,
                      idep_libagx_shaders_h, idep_agx_pack],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g17p_restart_state',
    executable(
      'brl_g17p_restart_state_test',
      files('tests/test-brl-g17p-restart-state.c'),
      include_directories : [include_directories('.')],
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g17p_metadata_grid',
    executable(
      'brl_g17p_metadata_grid_test',
      files('tests/test-brl-g17p-metadata-grid.c',
            'brl_g17p_meta_copy_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_vulkan_util,
                      idep_vulkan_runtime_headers],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_agx3_compressed_buf2img',
    executable(
      'brl_agx3_compressed_buf2img_test',
      files('tests/test-brl-agx3-compressed-buf2img.c',
            'brl_g17p_meta_copy_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_vulkan_util,
                      idep_vulkan_runtime_headers],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_g17p_linear_slice_copy',
    executable(
      'brl_g17p_linear_slice_copy_test',
      files('tests/test-brl-g17p-linear-slice-copy.c',
            'brl_g17p_meta_copy_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src,
                             inc_asahi],
      dependencies : [idep_mesautil, idep_vulkan_util,
                      idep_vulkan_runtime_headers],
      link_with : [libasahi_layout],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_image_layout_core',
    executable(
      'brl_image_layout_core_test',
      files('tests/test-brl-image-layout-core.c',
            'brl_image_layout_core.c'),
      include_directories : [include_directories('.')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_native_descriptor',
    executable(
      'brl_native_descriptor_test',
      files('tests/test-brl-native-descriptor.c',
            'brl_native_descriptor_pack.c',
            '../lib/agx_pbe_agx3_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_native_sampler',
    executable(
      'brl_native_sampler_test',
      files('tests/test-brl-native-sampler.c', 'brl_native_sampler.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'brl_gs_ray_query',
    executable(
      'brl_gs_ray_query_test',
      files('tests/test-brl-gs-ray-query.c', 'brl_nir_lower_gs.c'),
      include_directories : [include_directories('.'), inc_asahi, inc_poly],
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
      link_with : [libpoly_nir, libasahi_compiler],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_vertex_preamble_shift',
    executable(
      'brl_vertex_preamble_shift_test',
      files('tests/test-brl-vertex-preamble-shift.c',
            'brl_nir_vertex_preamble.c'),
      include_directories : [include_directories('.'), inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'brl_sampler_bias',
    executable(
      'brl_sampler_bias_test',
      files('tests/test-brl-sampler-bias.c'),
      include_directories : [include_directories('.'), inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

endif

subdir('bvh')

brl_deps = [
  dep_libdrm,
  idep_nir,
  idep_vulkan_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi,
  idep_vulkan_wsi_headers,
  idep_agx_pack,
  idep_libagx,
  idep_libagx_shaders_h,
]

libbrl = static_library(
  'brl',
  [
    brl_entrypoints,
    brl_drirc,
    brl_files,
    brl_bvh_spv,
    sha1_h,
  ],
  include_directories : [
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
    inc_asahi,
    inc_virtio_gpu,
    inc_virtio_vdrm,
    include_directories('.'),
  ],
  link_with : [libasahi_lib, libasahi_layout, libasahi_compiler],
  c_args : ['-Wno-c2x-extensions'],
  dependencies : [brl_deps],
  gnu_symbol_visibility : 'hidden',
)

# The tests below replace driver functions with -Wl,--wrap. Link-time
# optimisation resolves calls inside the driver libraries before the linker
# applies the wrap, so link them from the regular objects of fat LTO builds.
brl_wrap_test_link_args = cc.get_supported_link_arguments('-fno-lto')

if with_tests
  test('brl_sparse_alias', executable('brl_sparse_alias_test',
    files('tests/test-brl-sparse-alias.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
  test('brl_rt_inlining', executable('brl_rt_inlining_test',
    files('tests/test-brl-rt-inlining.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_memory_oom', executable('brl_memory_oom_test',
    files('tests/test-brl-memory-oom.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=agx_bo_cache_evict_all',
      '-Wl,--wrap=agx_bo_export', '-Wl,--wrap=agx_bo_import',
      '-Wl,--wrap=brl_get_cgroup_memory_available', '-Wl,--wrap=__vk_errorf',
      '-Wl,--wrap=os_get_available_system_memory',
      '-Wl,--wrap=os_get_total_physical_memory']),
    suite : ['asahi'])
  test('brl_cache_identity', executable('brl_cache_identity_test',
    files('tests/test-brl-cache-identity.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args]),
    suite : ['asahi'])
  test('brl_push_descriptor_size', executable('brl_push_descriptor_size_test',
    files('tests/test-brl-push-descriptor-size.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_shader_loop_bound_cache', executable('brl_shader_loop_bound_cache_test',
    files('tests/test-brl-shader-loop-bound-cache.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
  test('brl_compute_pair_cache', executable('brl_compute_pair_cache_test',
    files('tests/test-brl-compute-pair-cache.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_host_image_copy', executable('brl_host_image_copy_test',
    files('tests/test-brl-host-image-copy.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps, dep_thread], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=__vk_errorf']),
    suite : ['asahi'])
  test('brl_tes_query', executable('brl_tes_query_test', files('tests/test-brl-tes-query.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_g17p_task_kernel', executable('brl_g17p_task_kernel_test',
    files('tests/test-brl-g17p-task-kernel.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_indirect_vdm', executable('brl_indirect_vdm_test',
    files('tests/test-brl-indirect-vdm.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_coherent_image', executable('brl_coherent_image_test',
    files('tests/test-brl-coherent-image.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_binding_base', executable('brl_binding_base_test',
    files('tests/test-brl-binding-base.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_m5_store_window', executable('brl_m5_store_window_test',
    files('tests/test-brl-m5-store-window.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_aligned_uadd_sat', executable('brl_aligned_uadd_sat_test',
    files('tests/test-brl-aligned-uadd-sat.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_m5_vs_cprog', executable('brl_m5_vs_cprog_test',
    files('tests/test-brl-m5-vs-cprog.c'),
    include_directories : [inc_gallium, inc_gallium_aux, inc_include, inc_src,
      inc_asahi, inc_virtio_gpu, inc_virtio_vdrm, include_directories('.')],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=brl_code_alloc', '-Wl,--wrap=agx_bo_create',
      '-Wl,--wrap=brl_device_ensure_geometry_heap']), suite : ['asahi'])
  test(
    'brl_rt_pipeline_executable',
    executable(
      'brl_rt_pipeline_executable_test',
      files('tests/test-brl-rt-pipeline-executable.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      link_with : [libbrl],
      dependencies : [brl_deps],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
endif

libvulkan_aurora = shared_library(
  'vulkan_aurora',
  link_whole : [libbrl],
  link_args: [ld_args_build_id],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

# Kept for one release so tools that look for the library under its old
# name still find it. Only the library is aliased: a second ICD manifest
# would make the loader list every GPU twice.
install_symlink(
  libname_prefix + 'vulkan_asahi.' + libname_suffix,
  pointing_to : libname_prefix + 'vulkan_aurora.' + libname_suffix,
  install_dir : get_option('libdir'),
  install_tag : 'runtime',
)

icd_file_name = libname_prefix + 'vulkan_aurora.' + libname_suffix

aurora_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : 'aurora_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

_dev_icdname = 'aurora_devenv_icd.@0@.json'.format(host_machine.cpu())
_dev_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : _dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', _dev_icd.full_path())

if with_tests and ld_args_gc_sections.length() > 0
  test(
    'brl_compression_contract', executable('brl_compression_contract_test',
      files('tests/test-brl-compression-contract.c'),
      include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
      dependencies : [brl_deps], link_with : [libbrl, libasahi_layout],
      link_args : [ld_args_gc_sections],
      gnu_symbol_visibility : 'hidden'), suite : ['asahi'])
  test(
    'brl_sparse_formats', executable('brl_sparse_formats_test',
      files('tests/test-brl-sparse-formats.c'),
      include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
      dependencies : [brl_deps], link_with : [libbrl, libasahi_layout],
      link_args : [ld_args_gc_sections],
      gnu_symbol_visibility : 'hidden'), suite : ['asahi'])
  test(
    'brl_host_as', executable('brl_host_as_test',
      files('tests/test-brl-host-as.c', 'brl_host_as.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [brl_deps],
      link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
        '-Wl,--wrap=util_flush_range']), suite : ['asahi'])
  test('brl_agx3_as_format', executable('brl_agx3_as_format_test',
    files('tests/test-brl-agx3-as-format.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args]),
    suite : ['asahi'])
  test('brl_as_registry', executable('brl_as_registry_test',
    files('tests/test-brl-as-registry.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args, '-Wl,--wrap=agx_bo_create',
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=_mesa_hash_table_insert',
      '-Wl,--wrap=agx_va_alloc', '-Wl,--wrap=agx_va_free']),
    timeout : 60, suite : ['asahi'])
  test('brl_host_as_lifetime', executable('brl_host_as_lifetime_test',
    files('tests/test-brl-host-as-lifetime.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args, '-Wl,--wrap=brl_meta_begin',
      '-Wl,--wrap=brl_meta_end']), suite : ['asahi'])
  test('brl_native_feature_defaults', executable('brl_native_feature_defaults_test',
    files('tests/test-native-feature-defaults.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_index_robustness', executable('brl_index_robustness_test',
    files('tests/test-brl-index-robustness.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]), suite : ['asahi'])
  test('brl_rt_on_demand', executable('brl_rt_on_demand_test',
    files('tests/test-brl-rt-on-demand.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=agx_va_alloc', '-Wl,--wrap=agx_va_free',
      '-Wl,--wrap=agx_bo_create', '-Wl,--wrap=agx_bo_unreference',
      '-Wl,--wrap=agx_bo_bind', '-Wl,--wrap=__vk_errorf']), suite : ['asahi'])
  test(
    'brl_rt_cache_control',
    executable(
      'brl_rt_cache_control_test',
      files('tests/test-brl-rt-cache-control.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [brl_deps],
      link_with : [libbrl],
      link_args : [
        ld_args_gc_sections,
        brl_wrap_test_link_args,
        '-Wl,--wrap=vk_pipeline_shader_stage_to_nir',
        '-Wl,--wrap=brl_compile_shader',
        '-Wl,--wrap=brl_api_shader_destroy',
        '-Wl,--wrap=ralloc_free',
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'brl_rt_pipeline_binary',
    executable(
      'brl_rt_pipeline_binary_test',
      files('tests/test-brl-rt-pipeline-binary.c'),
      include_directories : [
        include_directories('.'), inc_gallium, inc_gallium_aux,
        inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm,
      ],
      dependencies : [brl_deps],
      link_with : [libbrl],
      link_args : [
        ld_args_gc_sections,
        brl_wrap_test_link_args,
        '-Wl,--wrap=vk_pipeline_shader_stage_to_nir',
        '-Wl,--wrap=vk_common_CreatePipelineBinariesKHR',
        '-Wl,--wrap=vk_common_GetPipelineKeyKHR',
        '-Wl,--wrap=brl_compile_shader',
        '-Wl,--wrap=brl_api_shader_destroy',
        '-Wl,--wrap=ralloc_free',
      ],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
endif

if with_tests
  brl_dual_source_gpl_test = executable(
    'brl-dual-source-gpl',
    files('tests/brl_dual_source_gpl.c'),
    include_directories : [
      inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm, include_directories('.'),
    ],
    dependencies : [brl_deps, cc.find_library('vulkan')],
    # brl_cmd_buffer.h uses GNU void-pointer arithmetic in inline helpers.
    c_args : [c_msvc_compat_args, '-Wno-pointer-arith'],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-xfb-resume',
    files('tests/vulkan_xfb_resume.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )
  executable(
    'vulkan-mipmap-layers',
    files('tests/vulkan_mipmap_layers.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-render-area',
    files('tests/vulkan_render_area.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-tess-batch',
    files('tests/vulkan_tess_batch.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  vulkan_tess_batch_order = executable(
    'vulkan-tess-batch-order',
    files('tests/vulkan_tess_batch_order.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-ubo-record-loop',
    files('tests/vulkan_ubo_record_loop.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), cc.find_library('m')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-sample-variant-cache',
    files('tests/vulkan_sample_variant_cache.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-border-emulation-cache',
    files('tests/vulkan_border_emulation_cache.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-msaa-xfer',
    files('tests/vulkan_msaa_xfer.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-r11g11b10f',
    files('tests/aurora_r11g11b10f.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-tile-visit',
    files('tests/aurora_tile_visit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-restart-switch',
    files('tests/aurora_restart_switch.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-msaa-mrt',
    files('tests/vulkan_msaa_mrt.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'g17p-vulkan-multiview-clear',
    files('tests/g17p_vulkan_multiview_clear.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'aurora-tilesubmit',
    files('tests/aurora_tilesubmit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Submit packing on hardware: run with HK_PACK_CDM_SUBMIT=0 and =1.
  executable(
    'aurora-packtest',
    files('tests/aurora_packtest.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_thread],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  # Also no-op shim tests, registered with the shim in ../meson.build.
  g17p_vulkan_bind_tail_state_change = executable(
    'g17p-vulkan-bind-tail-state-change',
    files('tests/g17p_vulkan_bind_tail_state_change.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_indirect_multi = executable(
    'g17p-vulkan-indirect-multi',
    files('tests/g17p_vulkan_indirect_multi.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_compressed_mip_blit = executable(
    'g17p-vulkan-compressed-mip-blit',
    files('tests/g17p_vulkan_compressed_mip_blit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_garbage_metadata = executable(
    'g17p-vulkan-garbage-metadata',
    files('tests/g17p_vulkan_garbage_metadata.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_global_priority = executable(
    'brl-vulkan-global-priority',
    files('tests/brl_vulkan_global_priority.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_sparse_bind_first_submit = executable(
    'brl-vulkan-sparse-bind-first-submit',
    files('tests/brl_vulkan_sparse_bind_first_submit.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_descriptor_buffer_sizes = executable(
    'brl-vulkan-descriptor-buffer-sizes',
    files('tests/brl_vulkan_descriptor_buffer_sizes.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_descriptor_buffer_texel = executable(
    'brl-vulkan-descriptor-buffer-texel',
    files('tests/brl_vulkan_descriptor_buffer_texel.c'),
    include_directories : [inc_include, inc_src],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_cross_queue = executable(
    'g17p-vulkan-cross-queue',
    files('tests/g17p_vulkan_cross_queue.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_bg_eot_threads = executable(
    'brl-vulkan-bg-eot-threads',
    files('tests/brl_vulkan_bg_eot_threads.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_thread],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  brl_vulkan_explicit_plane_layout = executable(
    'brl-vulkan-explicit-plane-layout',
    files('tests/brl_vulkan_explicit_plane_layout.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g16s_vulkan_buffer_upload = executable(
    'g16s-vulkan-buffer-upload',
    files('tests/g16s_vulkan_buffer_upload.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_compressed_layer_copy = executable(
    'g17p-vulkan-compressed-layer-copy',
    files('tests/g17p_vulkan_compressed_layer_copy.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  hal200_vulkan_msaa_depth_compression = executable(
    'hal200-vulkan-msaa-depth-compression',
    files('tests/hal200_vulkan_msaa_depth_compression.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  g17p_vulkan_layered_depth = executable(
    'g17p-vulkan-layered-depth',
    files('tests/g17p_vulkan_layered_depth.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_m],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-storage-compression',
    files('tests/vulkan_storage_compression.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan'), dep_m],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'vulkan-dynamic-offset-draws',
    files('tests/vulkan_dynamic_offset_draws.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

endif

if with_tests
  brl_cpu_upload_bench = executable(
    'brl-cpu-upload-bench',
    files('tests/brl_cpu_upload_bench.c'),
    include_directories : [
      inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
      inc_virtio_gpu, inc_virtio_vdrm, include_directories('.'),
    ],
    dependencies : [brl_deps],
    link_with : [libbrl, libasahi_lib, libasahi_layout, libasahi_compiler],
    gnu_symbol_visibility : 'hidden',
  )
  test('brl_cpu_upload', brl_cpu_upload_bench, args : ['1'], suite : ['asahi'])
endif

if with_tests
  test('brl_submit_admission', executable('brl_submit_admission_test',
    files('tests/test-brl-submit-admission.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
  test('brl_fused_cprog', executable('brl_fused_cprog_test',
    files('tests/test-brl-fused-cprog.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  # The queue's own-timeline wait of a kernel submission: one entry at the
  # latest point any reason asks for. Header only; no device.
  test('brl_own_wait', executable('brl_own_wait_test',
    files('tests/test-brl-own-wait.c'),
    include_directories : [include_directories('.'), inc_include, inc_src]),
    suite : ['asahi'])
  test('brl_errored_submit', executable('brl_errored_submit_test',
    files('tests/test-brl-errored-submit.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [brl_wrap_test_link_args]),
    timeout : 30, suite : ['asahi'])
  test('brl_lost_object_submit', executable('brl_lost_object_submit_test',
    files('tests/test-brl-lost-object-submit.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [brl_wrap_test_link_args,
      '-Wl,--wrap=vk_queue_wait_lost_submissions']),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_sparse_submit_wait', executable('brl_sparse_submit_wait_test',
    files('tests/test-brl-sparse-submit-wait.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_queue_sharing', executable('brl_queue_sharing_test',
    files('tests/test-brl-queue-sharing.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
  test('brl_pack_atomics', executable('brl_pack_atomics_test',
    files('tests/test-brl-pack-atomics.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_cmd_pending', executable('brl_cmd_pending_test',
    files('tests/test-brl-cmd-pending.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl, libasahi_lib],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_query_stamps', executable('brl_query_stamps_test',
    files('tests/test-brl-query-stamps.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl]),
    timeout : 30, suite : ['asahi'])
  test('brl_query_no_access', executable('brl_query_no_access_test',
    files('tests/test-brl-query-no-access.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
  test('brl_byte_stencil', executable('brl_byte_stencil_test',
    files('tests/test-brl-byte-stencil.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
  test('brl_2d_stencil', executable('brl_2d_stencil_test',
    files('tests/test-brl-2d-stencil.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
  test('brl_compute_opt', executable('brl_compute_opt_test',
    files('tests/test-brl-compute-opt.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl, libasahi_compiler]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_attachmentless', executable('brl_attachmentless_test',
    files('tests/test-brl-attachmentless.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_ray_tracing_core', executable('brl_ray_tracing_core_test',
    files('tests/test-brl-ray-tracing-core.c', 'brl_ray_tracing_core.c'),
    include_directories : [include_directories('.'), inc_include, inc_src]),
    suite : ['asahi'])
  test('brl_root_only_flush', executable('brl_root_only_flush_test',
    files('tests/test-brl-root-only-flush.c'),
    c_args : ['-DNDEBUG'],
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps], link_with : [libbrl],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_uma_heap_sizing', executable('brl_uma_heap_sizing_test',
    files('tests/test-brl-uma-heap-sizing.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
    dependencies : [brl_deps, dep_m], link_with : [libbrl, libasahi_lib],
    link_args : [ld_args_gc_sections, brl_wrap_test_link_args,
      '-Wl,--wrap=drmIoctl',
      '-Wl,--wrap=os_get_available_system_memory',
      '-Wl,--wrap=os_get_total_physical_memory']),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_identity_copy4', executable('brl_identity_copy4_test',
    files('tests/test-brl-identity-copy4.c', 'brl_nir_copy_identity4.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx, dep_m],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif

if with_tests
  test('brl_compute_pair', executable('brl_compute_pair_test',
    files('tests/test-brl-compute-pair.c', 'brl_nir_match_pointwise_clamp.c',
      'brl_nir_tensor_ops.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])

  test('brl_tensor_ops', executable('brl_tensor_ops_test',
    files('tests/test-brl-tensor-ops.c', 'brl_nir_tensor_ops.c'),
    include_directories : [include_directories('.'), inc_gallium, inc_gallium_aux,
      inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil, idep_libagx, dep_m],
    link_with : [libasahi_compiler],
    link_args : [ld_args_gc_sections]),
    timeout : 30, suite : ['asahi'])
endif
