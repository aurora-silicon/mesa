# Borealis synchronization regressions

A suite run on hardware in addition to CTS. It looks for synchronization and
visibility bugs where every fence and semaphore completes but the pixels or
memory are wrong: work on several queues, partial render areas and tile
preservation, feedback loops, transfers into compressed images, submission
packing and command-buffer retirement. Every case checks exact contents
after its waits complete. A signalled fence is never treated as a pass.

The suite does not replace CTS or the Khronos validation layers and does not
claim to be complete. CTS remains the authority on conformance.

## Pieces

- `borealis-sync` (`../borealis_sync.c`): one executable holding the cases
  written for this suite. `borealis-sync --list` prints them with their tier;
  `borealis-sync <case>` runs one. Options: `--iterations N`, `--seed N`
  (the upload pattern), and `--shim`, which records, submits and waits but
  does not check contents, for the no-op DRM shim.
- `runner.py`: runs the cases in `cases.json`, each in its own process
  session, with a timeout, and writes JSON Lines. It uses only the Python
  standard library, builds nothing and selects no driver.
- `cases.json`: the suite. It runs the `borealis-sync` cases and a curated
  set of existing standalone tests from `src/asahi/vulkan/tests`.
- `test_runner.py`: unit tests of the runner with fake executables (no
  Vulkan). Meson runs them as `borealis-sync-runner`; the
  `borealis-sync-<case>` Meson tests run each case under the no-op DRM shim
  with the `t6041` profile, which has independent queues.

## Tiers

- **strict**: the case follows the Vulkan synchronization rules, so a wrong
  result is a driver bug. Each strict `borealis-sync` case states in its
  source comment why no further dependency is needed. A new strict case
  must also run clean under the Khronos validation layer with
  synchronization validation (`VK_LAYER_VALIDATE_SYNC=1`) on a driver that
  can run it. Strict cases run by default unless marked `opt_in` (such as
  `submit-pack-dup`, which tracks a known pre-existing kernel device-loss
  case and runs only when named).
- **compat**: the case does something applications do that Vulkan does not
  guarantee, such as unsynchronized writes from two queues to different
  texels of one optimally tiled subresource, or storage writes across draws
  without a barrier. A failure is still worth knowing about, but it is not a
  conformance bug. Compatibility cases run only with `--include-compat` or
  when named.

## The `borealis-sync` cases

| Case | Tier | What must hold |
| --- | --- | --- |
| `layer-disjoint-queues` | strict | Two queues of one family render 128 partial render areas each into different array layers of one image, submitted at the same time from two threads. Both layers must hold exactly their own colours. |
| `feedback-loop-outside-area` | strict | Queue 0 uploads a new pattern each iteration; a graphics queue waits for it and renders the left half of every 32x32 cell in the feedback-loop layout while sampling the right half of the same image, outside the render area. The left halves must hold the inverted right halves and the right halves the upload. |
| `same-image-disjoint-tiles` | compat | Two queues render the left and right halves of every cell of one image with no dependency between them. Both halves must survive. |

Each queue orders its own render pass instances with a barrier, because a
store operation may touch the whole attachment, not only the render area.
In the strict cases nothing else is left unordered: the two queues write
different subresources, or wait for each other with timeline semaphores.
In the compatibility case the two queues are the only unordered writers.

The colours alternate between iterations and the upload pattern changes
every iteration, so contents left over from the previous iteration fail.
The readback buffer is filled with a value no case expects before every
iteration, so a copy that never ran fails too.
Submissions to two queues are made from two host threads released together.
That makes the submissions concurrent; it does not prove that the GPU ran
them at the same time, and the cases do not claim it.

A case that needs a feature the device lacks exits 77 (`SKIP`).
`layer-disjoint-queues` and `same-image-disjoint-tiles` need two queues in
one graphics family. `feedback-loop-outside-area` uses a second queue of
another family when the graphics family has only one, with a concurrently
shared image.

## Running

```sh
python3 src/asahi/vulkan/tests/sync/runner.py \
  --bindir build/src/asahi/vulkan --repetitions 5 --seed 716 \
  --jsonl sync-results.jsonl
```

`--bindir` is the directory holding the test executables, normally the
build's `src/asahi/vulkan`. By default every strict case runs. Add
`--include-compat` for the compatibility cases too, or name cases to run
exactly those, compatibility cases included. `--list` prints the cases.
`--timeout` overrides every case's timeout; a timeout kills the test's
process group and is a failure. If a process that left the group keeps the
test's output open, the runner stops collecting output after 5 seconds and
says so in the record. `--seed` fixes the order cases run in.

## Results

Each record holds the case, its command, exit status, duration, captured
output and status; the last record is a summary. An exit status of 0 is
`PASS`, 77 is `SKIP`, anything else or a timeout is `FAIL`. A required
executable that is missing is `HARNESS_ERROR`; a case marked `optional` is
`SKIP` instead, with the reason in its record. `clear-pass-load-compat`
is optional: its executable comes with a separate change.

Failures are not hidden by default. To track known failures, pass
`--xfail` a JSON object mapping case names to reasons (or a list of names).
A listed case that fails in any way (any failing exit status or a timeout)
is `XFAIL`; one that passes is `XPASS`, which fails the suite so the entry
gets removed. The record keeps the exit status, so a change in how a case
fails stays visible. Names that are not in the suite are rejected.

Runner exit status: 0 when everything passed, skipped or failed as
expected; 1 on a failure or `XPASS`; 2 on a harness error or bad input; 77
when everything was skipped.

## Cross-process pressure

`cross-process-pressure` starts several strict cases at once in separate
processes. It shares no Vulkan object between them; it adds mixed render,
transfer and fragment load from other processes while each case checks its
own results. A strict parallel case may contain only strict cases. Its
members are cases that pass on their own, so a failure there points at the
added load.

## Case file

`cases.json` is versioned. A leaf case names an `executable` relative to
`--bindir` (no `..`) with optional `args`, `env`, `timeout`, `tier`,
`opt_in` and `optional`. A parallel case lists `members` (leaf cases)
instead of an executable; parallel cases do not nest.

## Not covered yet

Linear or DRM-modifier images aliased between render and vertex input,
external memory and DMA-BUF visibility, a shared read-only
`SIMULTANEOUS_USE` secondary command buffer executed into separate
attachments on several queues, and unit tests of the driver's submit-time
dependency tracking.
