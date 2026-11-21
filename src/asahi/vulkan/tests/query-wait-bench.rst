HK query wait checks
====================

The CPU-only regression targets are ``hk_query_wait``,
``hk_query_wait_wiring`` and ``hk_query_wait_bench_profile``. The first links the production wait/tracking helper
with controlled clock, completion and loss dependencies. The second calls the
real timestamp/reset recording, secondary replay, host read/reset and
QueueSubmit2 registration entrypoints, replacing GPU emission and the final
submission callback. The profile fixture uses the real pure ``agx_chip`` helpers with observed raw
G16S parameters and rejects wrong chip, USC, HAL, topology, clock, architecture
and required-feature cases. None of these targets opens a GPU device.

Compatibility boundary
----------------------

The new path is enabled only when ``hk_check_status`` has a usable loss source:
a registered status-mirror word, an async-error counter, or native VM_STATUS.
A mirror feature bit without successful registration does not qualify; a
native feature bit on a virtio device does not qualify by itself. Otherwise
query tracking allocation/recording/registration is skipped and the original
finite wait is retained. Busy spinning and the two-second timeout remain
unresolved on those legacy configurations. This guard avoids a new indefinite
wait on kernels that cannot report the loss of an unrelated/unpublished fence.
It adds no kernel interface or asynchronous loss-notification contract.

Host CPU measurements
---------------------

The helper test also accepts ``--measure-delayed 150`` and
``--measure-delayed 2200``. A CPU thread publishes readiness and mock completion
at the requested monotonic-clock deadline. The actual production helper waits;
this is not the earlier condition-variable strategy model. JSON output reports
wall, process and reader CPU time, availability/status checks, completion waits
and the actual publication time. For example, from the source root::

   timeout 6s build/src/asahi/vulkan/hk_query_wait_test --measure-delayed 150
   timeout 6s build/src/asahi/vulkan/hk_query_wait_test --measure-delayed 2200

The local candidate does not spin. Both completion waits and condition waits
request at most 1 ms while the queried slot has any live registered, queued,
reserved or accepted writer. Availability can precede the submit callback's
publication of an accepted timeline point, so shortening only accepted-producer
waits misses that case. With no matching writer, condition waits retain the
100 ms fallback: registration broadcasts under the tracking mutex, atomically
with the predicate change. A writer for another query does not shorten an idle
slot's wait. None of these intervals is a deadline for producing the result.

The initial 50-microsecond spin from the earlier candidate was removed: it can
be exhausted before a delayed producer runs, and the approved short hardware
comparison did not establish a latency benefit. These parameters remain tuning
candidates; shorter condition/completion slices increase availability/status
checks and may increase CPU/DRM work. Any revised build needs fresh physical
measurements, not the earlier candidate's results.

Fake-time tests place availability just after a check for registered, queued,
reserved and accepted writers and observe 999,999 ns of extra blocking without
waiting for the accepted point or submission completion. They check loss with
availability on the same wake, long valid work, a future producer registering
after 2500 ms of idle, unrelated query slots and no busy spin. These are
production-helper models, not GPU/kernel performance evidence. Actual latency
also includes scheduling, locking and status-query costs. A timeline point is
a blocking opportunity, never proof that the queried value is available. No
independent query-publication kernel event or universal idle/loss rule is added.
Live measurements and CTS still require separate authorization before
publication or merge.

Retired writers and reset generations
------------------------------------

For a slot without a GPU reset since its last host reset, observed completion
of its latest registered writer now retains missing-write evidence. Once no
writer remains, the first reader observation starts a shared two-second grace.
Availability and sticky loss are rechecked before a generation-checked final
decision. A newer registration or host reset clears the evidence and deadline;
unrelated queue/query activity does not restart it. Queued, reserved and
accepted work has no overall timeout. Cancellation, teardown and host-reset
unlinking are not completion evidence. The existing M5 idle/victim rule and
the finite legacy path remain separate.

This is a conservative partial diagnostic. GPU resets can make a successfully
written query unavailable again, including after the old producer completed.
Record reset ranges so submission can suppress the diagnostic for that slot
until its next host reset. Suppression starts before execution, also for
failed/deferred submissions, and survives writer/reset recording replay.
Ordered reset records do not prove execution order across queues or identify
an executed GPU generation. Missing-write timeouts after GPU resets remain
open; a sticky historical retired flag would be unsafe. No GPU memory layout,
reset emission, cache flush, timestamp write or synchronization is changed.

Reset-only recordings retain range metadata and pool references again, but
create no producer or batch, allocate no producer-pointer array and issue no
tracker completion query on submission. On aarch64 the slot grows from 16 to
48 bytes and the per-writer link from 32 to 40 bytes; a recording op remains
24 bytes. A fresh reset-only recording first allocates the dynarray's 64-byte
minimum capacity; subsequent growth is amortized. Submission scans the ops
and locks the tracker to invalidate reset ranges at registration and again
before actual submission. This is additional host memory/locking work, not a
measured performance claim. Repeated invalidation while suppressed does not
bump the device serial or broadcast. Only availability is read under the
tracking mutex at the final decision; status/error callbacks, which can log
through application debug callbacks, remain outside it.

The CPU-only regression and bounded fault-model probes are::

   timeout -k 1s 8s build/src/asahi/vulkan/hk_query_wait_test --case retired-reset-future
   timeout -k 1s 8s build/src/asahi/vulkan/hk_query_wait_test --probe-retired-missing completion
   timeout -k 1s 8s build/src/asahi/vulkan/hk_query_wait_test --probe-retired-missing prune

The probes use a three-second fake-time watchdog returning VK_ERROR_UNKNOWN;
exit one means no independent missing-write exit was observed. They return
device loss at two seconds on the corrected host-reset path. These probes
inject absent availability after a successful mock completion; they are not
valid Vulkan GPU fault reproductions. The reset/reuse control models a
completed reset before the reader, then registers a future writer at 2500 ms.
It must still succeed, including when the old writer is pruned before reading.
The default suite also checks deadline status/registration/reset races, sticky
loss, unrelated churn and generation recovery by host reset.

Private physical workload (separate authorization required)
----------------------------------------------------------

``hk-query-wait-bench.c`` is a standalone, manually invoked Vulkan workload,
not a Meson test. Compilation and ``--help`` do not enumerate or submit GPU
work. Build against the headers in this source tree and the installed loader
and libdrm::

   cc -std=c11 -O2 -Wall -Wextra -Werror -Iinclude -Isrc -Isrc/asahi/lib \
      $(pkg-config --cflags libdrm) \
      src/asahi/vulkan/tests/hk-query-wait-bench.c src/asahi/lib/agx_chip.c \
      -lvulkan -ldl -pthread -ldrm -o /PRIVATE/hk-query-wait-bench
   /PRIVATE/hk-query-wait-bench --help

Both ``--probe`` and ``--run`` access real hardware and require authorization.
Use a fresh process with one absolute private ICD manifest in
``VK_DRIVER_FILES``. ``--library`` must name the exact private library; the
program verifies that it was actually mapped, requires exactly Mesa's default
Vulkan vendor ID and the Borealis driver ID, and prints observed IDs and library
mapping before identity rejection. Native Apple hardware identity is verified
separately through the kernel parameters. Workload version four reports both
the raw GPU generation/variant and the normalized driver profile.

The preflight reads the native kernel's global parameters using the enumerated
render node. This first benchmark is deliberately limited to the M4 Pro's
source-defined profile: chip 0x6040, normalized G17P, one die, two clusters,
USC 3, HAL200, nonzero timestamp frequency, VM_STATUS, PER_COMMAND_INPUT_SYNCS
and EXACT_PRIOR_BARRIERS. The benchmark links the actual pure
``agx_chip_from_params``, ``agx_params_driver_supported`` and
``agx_params_native_kernel_abi_supported`` helpers. Raw G16S T6040 with USC 3
and HAL200 maps to G17P in that source; raw and normalized identities must not
be confused. The exact chip/topology/USC/HAL/clock/feature guards remain. A mismatch stops before creating the logical device.
These checks do not certify kernel/ICD compatibility or recovery safety; review
the raw parameters and the kernel build provenance before authorizing a run.
The VM status mirror is optional; the sticky VM status query is required here.

After authorization, a probe command has this form::

   timeout -k 2s 12s env VK_DRIVER_FILES=/PRIVATE/asahi_icd.aarch64.json \
      /PRIVATE/hk-query-wait-bench --probe \
      --library /PRIVATE/libvulkan_asahi.so

``--run`` enables only timelineSemaphore, hostQueryReset and synchronization2,
selects a timestamp-capable graphics/compute queue, host-resets an unused pool,
and records one BOTTOM_OF_PIPE timestamp. One submit contains both a timeline
wait (zero to one, ALL_COMMANDS) and the timestamp command. A CPU release thread
signals value one relative to the reader-entry handshake. There is no shader,
draw or dispatch. All resources stay alive through the release thread and
retirement. A retirement timeout stops the process without explicitly
destroying pending resources; it is a stop condition, not a recovery technique.

Offer the following cases independently:

* ``--delay-ms 150``: short successful-read comparison, with no intended timeout
  or device loss.
* ``--delay-ms 2200``: the baseline may deliberately report device loss near
  two seconds before host release. This requires a separate decision and a
  reviewed recovery plan, preferably on an isolated test machine.

A run command, only after approval of its case, has this form::

   timeout -k 2s 12s env VK_DRIVER_FILES=/PRIVATE/asahi_icd.aarch64.json \
      /PRIVATE/hk-query-wait-bench --run --delay-ms 150 \
      --library /PRIVATE/libvulkan_asahi.so

The delayed signal is still attempted if the query read returns early. Output
contains configuration and measurement JSON Lines, including VkResult,
availability, wall/process/reader CPU, actual host signal time/result and
retirement result. Nonzero exit is expected for the long failing baseline;
preserve its output and exit status rather than dropping that sample.

Local candidate measurement modes (not yet run on hardware)
----------------------------------------------------------

Workload version four additionally reports CPU and wall time around the single
``vkQueueSubmit2`` call. ``--timestamps N`` records between 1 and 1024 ordered
BOTTOM_OF_PIPE timestamps in that command buffer, with separate query slots.
With normal readback, only query zero is read with WAIT. The subsequent fence
status is sampled after the timed reader region: ``VK_NOT_READY`` proves a
still-pending tail at that sample; ``VK_SUCCESS`` does not locate the fence
signal relative to availability or reader return. A 1024-timestamp tail is
bounded real work, not a synthetic host wait inside a command buffer, and may
be too short to demonstrate early publication on a particular driver. Report
that case as inconclusive if no pending tail is observed.

``--submit-only --delay-ms 0`` skips host query readback entirely and signals
the timeline gate immediately after the reader handshake. Use timestamp counts
1, 128 and 1024 to measure submit CPU cost even without WAIT readers.
``--reset-only`` requires that mode and replaces the timestamp writes with one
GPU reset of the specified number of slots. This isolates reset-only recording
from timestamp writer registration. It does not measure all recording costs or
steady-state frame throughput. Each invocation still makes just one submit,
keeps resources alive through retirement and retains the exact-device guards.
For example, after separate approval::

   timeout -k 2s 12s env VK_DRIVER_FILES=/PRIVATE/asahi_icd.aarch64.json \
      /PRIVATE/hk-query-wait-bench --run --delay-ms 0 --timestamps 1024 \
      --submit-only --library /PRIVATE/libvulkan_asahi.so

The identical workload binary must be used for target baseline, current public
PR and local candidate. Compare submit CPU and wall distributions separately
from reader CPU, wall and signal-to-return latency. More frequent completion
checks can cost more CPU/DRM operations even with only one initial spin; no
physical benefit is established by compilation or CPU-only models.

Comparison and path verification
-------------------------------

Use frozen baseline and proposed libraries built with matching flags and
identical dependencies. Never rebuild the baseline from a patched worktree.
Keep the library, manifest and benchmark SHA-256 hashes and both full Mesa
commit IDs with the results. Use the same physical machine, kernel, workload
binary, power/thermal conditions and process-local driver settings.

For each separately approved case, run six fresh-device processes per build,
alternating baseline/proposed order (AB, BA, AB, BA, AB, BA). Keep every raw
stdout, stderr, exit status and timeout. Record the loader, kernel, compiler,
libdrm, dependency and benchmark versions, the full commands/settings and
physical device identity. Report min/median/max and standard deviation, plus
side-by-side measurements with units and percentage changes. Repeat only if
variability or an anomaly needs investigation.

Separately trace one representative short run per build with
``strace -f -T -e trace=ioctl,futex -o TRACE`` around the exact command above.
Confirm the expected private library and recorded driver revision, successful
submission, and baseline polling versus proposed bounded registration/DRM
completion waits. Do not include traced runs in the timing samples. If the
proposed read happens to find the result already available, that sample does
not verify the affected blocking path and must be identified as such.

Acceptance checks for the short case are success, nonzero availability and
successful retirement for both builds, plus substantially lower reader CPU
(target at least 90% median reduction). For the long case, the proposed read
must succeed after host release, with availability and retirement successful;
retain the baseline's observed result instead of assuming it lost the device.
Investigate signal-to-return delay above 150 ms (100 ms blocking interval plus
a 50 ms scheduling budget) or proposed reader CPU above 5 ms on this otherwise
idle workload. These are measurement gates, not guaranteed real-time bounds.
The long baseline's early return and the proposed successful later return are
different outcomes; report the approximately 10% longer wall duration as the
necessary wait for readiness, not as a throughput regression.

This timestamp workload does not establish all-query GPU correctness,
real recovery-victim behavior, live fault cancellation, stage-specific timestamp
progress before a later host-dependent tail, or Vulkan conformance. Host tests
cover the blocking logic with controlled dependencies. Physical query-type and
recovery validation remains required before merge. Repository policy also
requires accessible same-hardware raw before/after results before any PR.
