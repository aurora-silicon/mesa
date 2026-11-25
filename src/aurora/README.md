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

`ci/deqp-borealis-{fails,flakes,skips}.txt` (Vulkan) and `ci/deqp-aurora-gl-*.txt` (OpenGL ES and
OpenGL) list the expected non-passing cases per GPU generation, from runs on real machines; each
section names the machine, kernel, Mesa commit and CTS version it came from.
