# T8140 colour MSAA transient spill backing

This ports the transient eMRT work from draft Mesa PR #72 onto `aurora-wip`.
The three original commits were `6e3cd8d48a6`, `96b08ee73ba` and `075a4102f9b`.
The base already has HAL200 array/MSAA descriptor packing, including the
T6040 changes in `33f2bf125e5`. It does not enable compressed T8140 MSAA
allocations or supply transient spill storage.

The new allocation helper admits only physical T8140, 2x/4x, one-mip colour
images with ordinary 2D/array storage. It normalizes sample/layer count before
calling the existing conservative policy, preserving its format, flag, usage,
sharing and extension-chain exclusions. Existing depth/stencil, cube, 3D,
single-sample and mip-chain policy remains. Eight samples remains disabled.
Colour plus input-attachment usage stays plain because application descriptors
would otherwise bypass the backing. Other generic compression checks still
apply, including format/store compatibility and the compression debug opt-out.

T6040/M4 Pro uses `AGX_CHIP_G17P` internally. Both admission and render redirection
therefore require `chip_id == 0x8140` in addition to the architecture enum.
T6040 keeps its HAL200 policy and direct eMRT path. Other products retain theirs.

Every ordinary private spilled T8140 target receives a command-buffer-owned
plain allocation using the existing AIL tiled sample layout. The allocation
covers the view's mip extent and pass layers, uses one level, and has no
compression metadata. Each pixel's samples remain adjacent. The eMRT texture
fetch and raw PBE store use the same existing descriptors for this layout.
The pool's 32-bit allocation limit is checked before allocation.

LOAD imports the render rectangle through the graphics meta copy path. Stored
targets export through the ordinary compressed PBE before attachment resolves
read the original image. CLEAR uses the existing clear draw. Partial areas,
view base layers and sparse multiview masks copy only the relevant samples.
Suspending exports backing and resuming imports it; secondary buffers and
physical stream splits use the same backing during an instance. Graphics
dependency tracking orders these copies. There is no in-place decompression
kernel or metadata-state transition. Resident compressed targets keep their
normal background/EOT descriptors.

Private CPU image/view handles and their descriptors live until command-buffer
reset/destruction. Their GPU allocations belong to the command-buffer pool.
Meta-created views are reset before the private image handles are destroyed.
Failed allocations remain on the cleanup list. Aliased, volume-slice and
input/feedback images retain direct plain storage; unsupported compressed
feedback fails closed.

Boundary copies and full-view scratch allocations add cost. No performance or
bandwidth improvement is established by this implementation. Before merge,
run a paired device benchmark with raw values and CTS with zero new bad cases.
The original draft must not be closed as already subsumed: the feature is absent
from its current target's consolidated base.

## Host checks

With tests enabled, run:

```sh
meson test -C build hk_g17p_color_msaa_compression hk_g17p_msaa_spill \
  hk_g17p_compression hk_g16s_compression hk_native_descriptor \
  hk_g17p_metadata_grid --print-errorlogs
```

The new compression test exercises 46 admitted colour formats, all sample
counts through 64, zero through eight layers, product/plane identity, flags,
usage and runtime extension chains. Its checks survive `NDEBUG`. The existing
T8140 conservative-policy and T6040 policy tests remain separate controls.

The spill test calls the production tilebuffer allocator and backing-mask
helper. It discovers every AIL colour-renderable tile size/alignment class,
then enumerates stable-sorted footprint sequences of zero through eight
attachments at 1x/2x/4x, layered and non-layered. Equal-alignment orders are
preserved; permutations across alignments and holes rename allocator outputs.
Witnesses move a small target through every slot and cover a compressed
resident target beside spills. Both normal and eight-byte LOAD/multiview
limits are covered. Region checks verify mip/layer/rectangle translation.
T6040, AGX2 and 8x are explicit backing-mask negative controls.

The same executable can inspect an individual layout without a GPU:

```sh
build/src/asahi/vulkan/hk_g17p_msaa_spill_test --probe 4 reload layered \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R8G8B8A8_UNORM
```

Use `normal`, `reload` or `multiview` and `plain` or `layered`. `PIPE_FORMAT_NONE`
represents an unused slot. Output includes spill and transient masks. These
CPU checks prove policy/layout behavior; device sample preservation needs CTS.
