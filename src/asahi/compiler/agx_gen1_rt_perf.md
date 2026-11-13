# G17P traversal experiments

`ASAHI_G17P_RT_PERF_ARM` is an integer bitmask, defaulting to zero. The Vulkan
selection is restricted to A18 Pro. Set it before creating the device; changing
it during a process is unsupported. Each bit changes one part of the native
candidate-query path. These are experiments requiring hardware qualification.

| Bit | Value | Change |
| --- | ---: | --- |
| 0 | 1 | Allow ForceOpaque rays through the existing hardware triangle-commit predicate. NoOpaque, facing/opacity culls, and SkipTriangles retain software handling. |
| 1 | 2 | Reduce the effective queue budget from ten to eight records, by changing the complete-burst reserve in both generated leaf paths. The physical ten-record storage and record stride stay unchanged. |
| 2 | 4 | Cache the immutable completed queue control word until the next refill. |
| 3 | 8 | Publish tmax only if the software committed bound changed since its last publication. |
| 4 | 16 | Issue refills using the polling continuation's staging/publication form. Keep the completion wait. |
| 5 | 32 | Pad each resident query slot from 304 to 320 bytes per lane. Update the packed SCS profile, runtime stride, allocation, and compiler address calculation together. |
| 6 | 64 | Read the runtime status byte for the initial pending test, sharing the polling loop's address calculation. |
| 7 | 128 | Clear the hardware committed tag/ID packet before refilling. Retain the software committed result. |
| 8 | 256 | Publish tmax to both RayCore banks. Can be combined with bit 3 to make both stores conditional. |

The selector participates in Vulkan's shader-cache identity. The zero-mask
compiler path preserves the existing query control, slot layout, refill form,
and wait sequence. Generated-runtime queue changes are applied after its
body/table source checks. External diagnostic runtime captures are outside the
queue-budget experiment's contract.

Measure single bits first, using the same driver build for baseline and arm
processes. Use separate cold-cache warmups and interleave A/B/B/A. Record each
process's selector. Then test interactions, particularly conditional tmax plus
bank mirroring (264), hardware commit plus mirroring (257), and eight-record
queues plus conditional mirrored tmax (266).

Qualification must cover coherent and incoherent closest-hit rays, scene and
build rows, and the same ray-query CTS subset for both sides. Host compilation
and tests establish address/allocation consistency and code generation; they
do not establish traversal correctness or a performance gain on the GPU.
