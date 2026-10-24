# G17P colour MSAA spill coverage

Colour MSAA render attachments remain plain. An image can be attached beside
other formats in any later render pass, so an allocation rule based on that
image's footprint alone cannot establish that its passes never spill.

The current eMRT preparation path converts every colour attachment of a spilling
pass to raw storage. G17P has no in-place decompression kernel. Enabling compressed
PBE writes does not supply that conversion. Compression admission must remain
closed until either that conversion or compression-preserving eMRT access is
implemented, including all samples, layers, metadata and ordering.

Run `meson test -C build hk_g17p_msaa_spill hk_g17p_compression` with tests enabled.
The spill test calls the production allocator. It discovers every colour-renderable
AIL format's tilebuffer size and alignment, including a superset of Vulkan's
padded/scaled formats. It enumerates all stable-sorted sequences of those
footprints for zero through eight attachments at 1x, 2x and 4x, both layered and
non-layered. Orders within equal alignments are kept. Permutations across different
alignments and holes only rename slots after the allocator's stable sort, so they
have the same spill outcome. Explicit witnesses also move a small target through
every slot.

Both allocator limits used by the driver are covered: the normal imageblock and,
for MSAA, the eight-byte limit for stored LOAD attachments or multiview. A footprint
is marked compressed if any admitted Vulkan colour format uses it. Every spilling
layout must contain no compressed colour attachment, including resident targets,
because render-pass preparation considers all its attachments. These are CPU
policy/allocator checks. They do not prove GPU execution of eMRT.

The same executable can inspect individual layouts without a GPU. For example:

```sh
build/src/asahi/vulkan/hk_g17p_msaa_spill_test --probe 2 normal plain \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R32G32B32A32_FLOAT \
  PIPE_FORMAT_R32G32B32A32_FLOAT PIPE_FORMAT_R8G8B8A8_UNORM
```

Use `normal`, `reload` or `multiview` for the pass mode and `plain` or `layered`
for its layer shape. `reload` means a LOAD attachment that is also stored.
`PIPE_FORMAT_NONE` represents an unused attachment slot. The output lists tile
geometry and a spill bit for each logical slot. Invalid arguments return status 2.
