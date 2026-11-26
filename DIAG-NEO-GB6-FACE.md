# Diagnostic branch: GB6 Face Detection traffic on the MacBook Neo (J700, A18 Pro G17P)

Not for merge. These commits are measurement tools, controlled by environment variables, on top of
`aurora-wip` 25f15c435b6. Measured on 2026-11-26, J700, stock kernel 7.3.0-41, GB6 6.7.1 Vulkan, 20
iterations, GPU-busy per iteration from firmware command intervals. Every arm had 0 validation errors and
0 GPU faults, recoveries or AVC denials.

## Result

GB6 Vulkan Face Detection (about 11.72 ms) is at the DRAM floor set by the writes the application
requires. A Mesa change can't take it to the 10.24 ms that beating Metal needs.

### Bytes per iteration

From the `PROF_FLOW` dump, as unions of the footprint intervals.

| traffic | MB | note |
|---|---:|---|
| writes | 472.8 | every write targets an app-visible VkBuffer; the sibling SSH im2col outputs are distinct buffers |
| reads: weights and constants | 16.9 | |
| reads: activations, read by the next dispatch | 91.6 | |
| reads: activations, farther back | 31.5 | |
| **total** | **612.8** | 53.0 GB/s over 11.56 ms (split profile) |

### DRAM roofs on the Neo

gpubench-vk GPU time, measured on aurora-wip 36623c0a4a5.

| test | GB/s |
|---|---:|
| `diag.fill size_mib=512` (write only) | 57.7 |
| `compute.heavy kind=bw` (64 MiB read + 64 MiB write) | 52.8 |
| `copy.bw kind=buf2buf` | 53.3 |

### Where the time goes

Least-squares fit over all 84 split-profiled dispatches, R² 0.988.

| term | cost | time per iteration |
|---|---|---:|
| writes | 17.4 µs/MB (= the 57.7 GB/s fill roof) | 8.24 ms |
| re-reads with under 8 MB written since their producer | 17.6 µs/MB (DRAM rate) | 1.62 ms |
| far reads | 27 µs/MB | 0.85 ms |
| ALU | | about 0.7 ms |

Forcing the near-read cost to zero makes the fit 2.8x worse.

Near re-reads of 4–6 MB tensors are cheap (1–7 µs/MB). The cost sits in the 13–15 MB tensors read by
ord 1, 5, 25, 26 and 29, about 1.0 ms in total.

### Ruled out

- **GPR occupancy tier.** The G17P compute launch ignores the GPR count; Face was unchanged with the
  reported count padded to 136 or 176.
- **Smaller tiles.** They lose on every small-grid Face GEMM.
- **Compute resource-policy word.** 0x60 against 0x132/0x200 was null on GB6 (dcv2 lane, 2026-11-21).
- **GPU memory attribute.** macOS maps GPU user buffers with memattr index 2, the same as drm/asahi
  `PROT_GPU_SHARED_*`.
- **Sharing the sibling im2col inputs.** Saves about 6 MB of reads.
- **Implicit GEMM.** Already in place: the fused im2col_a GEMMs read the input directly. What remains is
  GB6's own im2col output writes.

### Metal

No Metal capture exists. GB6 runs Face through MPSGraph on Metal (7 graph intervals, 10.72 ms on the
Neo), which has no app-visible intermediates. There is nothing literal to copy.

### Workgroup traversal order

`HK_PROBE_REV` reverses the workgroup order of selected fused cmat kernel classes. Same boot, Face busy
per iteration:

| arm | ms |
|---|---:|
| probe off | 11.711 / 11.721 |
| `dc` (dw_a + dcopy reversed) | 11.611 / 11.624 (−0.84 %) |
| `ig` (im2col_a + gemm_dw reversed) | 11.580 (−1.16 %) |

- Background Blur moved −0.24 % (`dc`) and −0.68 % (`ig`).
- Split profile, per dispatch: ord 1 −71 µs, ord 3 −50 µs, ord 5 −41 µs. The small 80x80 dw_a layers lose
  10–20 µs each.
- Below the 5 % bar, so not turned into a PR.

## Tools (environment variables)

- `HK_PACK_PROFILE_SPLIT=1`: one firmware command per fused dispatch, so anatomy times each dispatch.
- `HK_PACK_PROFILE_DUMP=1`: `PROF_DISP` shapes and `PROF_FLOW` data flow for the first packed submit of
  200 or more members.
- `HK_PROBE_PAD_GPRS=<n>`: pad the reported compute GPR count.
- `HK_PROBE_IM2A_TILE` / `HK_PROBE_DWA_TILE`: tile overrides for the small-grid im2col_a and dw_a GEMMs.
- `HK_PROBE_REV=<d|c|i|g|f…>`: workgroup reversal, as above.
