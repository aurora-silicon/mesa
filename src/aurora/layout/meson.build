# Copyright 2022 Alyssa Rosenzweig
# SPDX-License-Identifier: MIT

libaurora_layout_files = files(
  'formats.c',
  'layout.c',
  'layout_core.c',
  'tiling.cc',
)

libaurora_layout = static_library(
  'aurora_layout',
  [libaurora_layout_files],
  include_directories : [inc_include, inc_src],
  dependencies: [idep_mesautil, idep_agx_pack],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'aurora_layout_tests',
    executable(
      'aurora_layout_tests',
      files(
        'tests/test-layout.cpp',
        'tests/test-generated.cpp',
        'tests/test-twiddling.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_mesautil],
      link_with : [libaurora_layout],
    ),
    suite : ['aurora'],
    protocol : 'gtest',
  )
endif
