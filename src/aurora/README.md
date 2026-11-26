<!--
Copyright 2026 Aurora Silicon
SPDX-License-Identifier: MIT
-->

# Aurora: Mesa drivers for Apple GPUs

Aurora Silicon's Mesa drivers for the Apple GPUs of the M1 to M5 families and the A18 Pro (G13 to
G17). The G18 (M6) code paths are built in but have not run on hardware yet. They started as the
Asahi Linux drivers (`src/asahi`, `src/gallium/drivers/asahi`) and keep those authors' copyright
notices. Nothing here uses `src/asahi`.

| Driver   | API                | Meson option                | Code                                  |
|----------|--------------------|-----------------------------|---------------------------------------|
| Borealis | Vulkan 1.4         | `-Dvulkan-drivers=aurora`   | `src/aurora/vulkan` (`brl_*`)         |
| Aurora   | OpenGL, OpenGL ES  | `-Dgallium-drivers=aurora`  | `src/gallium/drivers/aurora`, `src/gallium/winsys/aurora` |

`asahi` is still accepted for both options, with a deprecation warning. Inside the build the
Gallium driver keeps its internal name (`asahi_dri.so`, `HAVE_ASAHI`), which the loader matches to
the kernel driver.

Borealis reports `VK_DRIVER_ID_AURORA_BOREALIS` (35), allocated by Khronos in Vulkan-Docs
70304594e7 ("Add a driver ID for Borealis"). It installs `libvulkan_aurora.so`, the loader manifest
`aurora_icd.<arch>.json` and the driconf defaults `00-borealis-defaults.conf` (driver name
`borealis`; sections written for the old name `hk` still apply). For one release
`libvulkan_asahi.so` is a symlink to it. Its environment variables keep their `HK_` names and its
driconf options their `hk_` names.

## Layout

| Directory   | What it is                                                     |
|-------------|----------------------------------------------------------------|
| `vulkan/`   | Borealis                                                       |
| `compiler/` | the AGX NIR backend and AGX3 packer, shared by both drivers    |
| `isa/`      | ISA descriptions and disassemblers                             |
| `genxml/`   | hardware structure descriptions (`agx_pack.h`)                 |
| `layout/`   | image layout                                                   |
| `lib/`      | device, BO, command stream and shader runtime helpers          |
| `libagx/`   | OpenCL C helper kernels, precompiled by `clc/`                 |
| `clc/`      | the precompiler                                                |
| `drm-shim/` | a no-op DRM device for the tests                               |
| `tools/`    | developer tools                                                |
| `ci/`       | CI expectations and the upstream divergence check              |

The Gallium driver and its winsys live under `src/gallium`, where Mesa builds every Gallium driver.
Outside these trees the drivers use only hardware-independent Mesa: `src/vulkan`, `src/compiler`,
`src/util`, `src/poly`, Gallium's auxiliary code and interfaces, the loader and WSI.

## Building and testing

    meson setup build -Dvulkan-drivers=aurora -Dgallium-drivers=aurora \
        -Dtools=drm-shim -Dbuild-tests=true
    meson test -C build --suite asahi

The drm-shim tests record and submit command streams against a fake device of a chosen generation
(`ASAHI_SHIM_CHIP=g13g`, `g17p`, ...); they do not execute anything on a GPU.

## Upstream

`src/asahi` in this tree is upstream Mesa's asahi driver, exactly as upstream has it at the commit
named in `UPSTREAM` (the last upstream merge), and is not built. To see how far our trees have
diverged from it:

    python3 src/aurora/ci/aurora-divergence.py [--files]

It compares `src/aurora` with `src/asahi` after undoing the mechanical renames of the move
(`aurora/` includes, the `brl_` prefix and file names), and the Gallium driver and winsys with the
upstream commit through git when it is in the repository. With `--upstream FETCH_HEAD` after
`git fetch https://gitlab.freedesktop.org/mesa/mesa.git main` it also lists the upstream commits
since then that touch the asahi driver: the candidates to cross-port. It runs as the
`aurora-divergence` meson test (informational).

## Conformance baselines

`ci/deqp-borealis-<gen>-{fails,flakes,skips}.txt` (Vulkan) and `ci/deqp-aurora-gl-<gen>-{fails,flakes}.txt`
(OpenGL ES and OpenGL) are deqp-runner expectation files per GPU generation, from runs on real machines.
Each file's header names the machine, kernel, Mesa commit, CTS version and which cases ran. Entries
that passed with the installed package on the same Mac are listed last, under a comment: they are
regressions to fix, not accepted results.

| Generation | Mac | Vulkan | OpenGL ES / OpenGL |
|------------|-----|--------|--------------------|
| G13S (M1 Pro) | J316s | 636,000-case sample | all 124,965 |
| G14 (M2) | J413, J473 | not run yet (Macs held) | not run yet |
| G15G (M3) | J615 | 636,000-case sample | GLES3 + 28,220 GL |
| G16G (M4) | J713 | 190,114 cases, plus skips | GLES3 |
| G16S (M4 Pro) | J616s | 564,000 cases | not run yet |
| G17P (A18 Pro) | J700 | 16,541-case smoke | not run yet |
| G17 (M5) | none in our lab | asked of other boards' owners | not run yet |
| G18 (M6) | none (no hardware yet) | not run | not run |

The samples are every 5th 6,000-case chunk of the VK-CTS 1.4.7.1 vk-default mustpass (3,215,124
cases), so cases outside them have no expectation yet.

## Borealis compared with upstream asahi

What the data shows today (aurora-wip `3f1c2bc3` against upstream Mesa `ab0ab2209c4e`, the copy in
`src/asahi`):

- **GPUs.** Upstream knows G13 and G14 (M1, M2: `AGX_CHIP_G13G/G13X/G14G/G14X`). These trees add
  G15 (M3), G16 (M4), G17 (M5, and the A18 Pro's G17P) and G18 (M6), 23 chip entries in total.
  Upstream does not run on M3 and later at all.
- **Vulkan extensions.** Borealis's extension table lists 192 extensions, upstream Honeykrisp's 168;
  none is upstream-only. The 24 Borealis-only ones (some only on the chips that support them):
  EXT_attachment_feedback_loop_dynamic_state, EXT_descriptor_buffer, EXT_external_memory_host,
  EXT_fragment_shader_interlock, EXT_memory_priority, EXT_mesh_shader, EXT_nested_command_buffer,
  EXT_pageable_device_local_memory, EXT_pipeline_library_group_handles,
  EXT_ray_tracing_invocation_reorder, EXT_shader_atomic_float, EXT_shader_atomic_float2,
  EXT_texture_compression_astc_hdr, KHR_acceleration_structure, KHR_compute_shader_derivatives,
  KHR_cooperative_matrix, KHR_deferred_host_operations, KHR_ray_query, KHR_ray_tracing_maintenance1,
  KHR_ray_tracing_pipeline, KHR_ray_tracing_position_fetch, KHR_shader_bfloat16,
  KHR_unified_image_layouts, NV_cooperative_matrix2.
- **Conformance on our GPUs.** See the baselines table above (Borealis only).

Not measured yet, and needed before an upstream series argues for M1 and M2: upstream Honeykrisp and
the upstream Gallium driver on the same M1 Pro and M2 Macs, same CTS lists and same kernel, for
CTS pass deltas, and gpubench/game-trace performance for both drivers. The rows kit in
`kits/test/20261126-borealis-split` takes any overlay pair, so this is one more overlay built from
upstream main.

## Sharing code with upstream asahi

`src/aurora` carries its own copy of everything upstream keeps in `src/asahi`. How far each part
has moved since `ab0ab2209c4e` (commits on aurora-wip touching it, upstream files changed):

| Part | Commits | Lines added | Upstream files changed | Sharing outlook |
|------|---------|-------------|------------------------|-----------------|
| `vulkan` | 1,066 | 212k | most | separate driver (Borealis) |
| `compiler` | 770 | 362k | most | AGX3 backend is new; the AGX2 core could be shared if upstream takes our fixes |
| `lib` | 310 | 54k | many | device/BO layer shareable, AGX3 command stream ours |
| Gallium driver + winsys | 234 | | most | separate driver |
| `libagx` | 47 | 3.6k | few | good candidate to share |
| `drm-shim` | 39 | 4.9k | few | good candidate to share |
| `genxml` | 27 | 2.5k | few | AGX2 XML shareable; AGX3 XML added beside it |
| `clc` | 26 | 0.5k | few | shareable |
| `layout` | 20 | 1.4k | few | shareable |
| `isa` | 7 | 18k | few | AGX2 description shareable; AGX3 added beside it |

`ci/aurora-divergence.py` gives the per-file picture. For an upstream series the plan is: send the
fixes that apply to upstream's AGX2 code to upstream asahi first; once a part (layout, clc, libagx,
drm-shim, the AGX2 halves of genxml and isa) matches upstream again, build Borealis against the
upstream copy instead of ours, so that only the AGX3-specific code and the Borealis driver are new.
A human submits and talks to upstream; no tool interacts with upstream merge requests.
