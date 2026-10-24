# G17P colour MSAA transient spill backing

G17P chooses lossless compression per image. The existing colour format and
sharing restrictions still apply, and 8x remains disabled. An eligible 2x/4x
colour attachment stays compressed when other targets force its pass to spill.
Resident attachments retain their normal compressed background/EOT descriptors.

Each ordinary private 2D spilled target uses a separate command-buffer-owned
allocation. The driver
chooses the existing uncompressed AIL tiled layout for this private storage.
Within a layer, tiles are row-major and pixels within a tile are Morton ordered.
Samples of a pixel are adjacent. For byte size `B` and sample count `S`, the
sample address is `base + layer * layer_stride + (pixel_index * S + sample) * B`.
AIL chooses tile width/height, padded width and layer stride from the format and
extent. These are the same values consumed by the existing native eMRT texture
fetch and linear PBE store. Each allocation is page aligned, has one mip level,
and covers the view's mip extent and the pass's layer range. It has no metadata
and is never exposed as application storage. Allocations exceeding the pool's
32-bit size limit fail with an allocation error rather than truncating.

The initial implementation reuses graphics meta copies at pass boundaries. LOAD
imports the render rectangle from the original compressed texture into private
storage, preserving every sample. CLEAR still uses the existing clear draw.
Stored targets are exported through the normal PBE path before an attachment
resolve reads them. DONT_CARE/NONE stores need no export. Copies touch only the
render rectangle and active layers of a multiview mask. A suspended instance
exports its backing, and a resumed instance imports from the original image.
Physical stream splits and secondary command buffers keep the same backing.
The ordinary graphics dependency tracking orders these operations. There is no
in-place decompression, metadata transition or compression-specific barrier.

The backing image/view handles and their heap descriptors live until command
buffer reset/destruction. GPU storage belongs to the command-buffer pool. Failed
allocations remain on the cleanup list. M1/M2 keep their existing eMRT path.
Volume slices, aliased images and input/feedback images retain direct plain
eMRT storage because they can be accessed through application descriptors during a pass. A compressed
input-feedback configuration is refused until that protocol is implemented.
The allocation policy therefore keeps MSAA images with both colour and input
attachment usage plain. Other input images keep their existing admission.

Boundary copies add work and full-view scratch storage. This is a correctness
implementation, with no bandwidth improvement claimed before measurement.
Fusing imports/exports into BG/EOT is a possible subsequent optimization.

Run `meson test -C build hk_g17p_msaa_spill hk_g17p_compression` with tests enabled.
The spill test calls the production allocator and transient admission helper.
It discovers every colour-renderable AIL tile size/alignment class, a superset of
Vulkan's padded/scaled formats, then enumerates stable-sorted footprint sequences
of zero through eight attachments at 1x/2x/4x, layered and non-layered. Orders
within equal alignments are preserved. Permutations across alignments and holes
rename allocator outputs. Explicit witnesses move a small target through every
slot and cover a compressed resident target beside spills.

Both the normal imageblock limit and the eight-byte stored-LOAD/multiview limit
are covered. All 10,290,450 layouts must select transient backing for every
spilled target, with compression still admitted for the eligible formats.
The helper is disabled on M1/M2 and at 8x. Region tests check mip, layer and
rectangle translation. These CPU checks do not prove GPU execution of eMRT.

The same executable can inspect individual layouts without a GPU.

```sh
build/src/asahi/vulkan/hk_g17p_msaa_spill_test --probe 2 normal plain \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R8G8B8A8_UNORM
```

Use `normal`, `reload` or `multiview` and `plain` or `layered`. `reload` means a
LOAD attachment that is also stored. `PIPE_FORMAT_NONE` represents an unused
slot. Output includes the spill and transient masks. Invalid arguments return 2.
