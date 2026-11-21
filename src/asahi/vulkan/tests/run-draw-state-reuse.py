#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Check what a native AGX3 chip does with its prepared bind tail when one
state changes between two draws.

brl-draw-state-bench records, for each state in its list, one command buffer
of a first draw and DRAWS more that each change that state. Under the DRM
shim and with ASAHI_MESA_DEBUG=perf the driver prints, per command buffer,
how many bind tails it built in full, rebuilt for a new fragment program
only, and published again unchanged. Nothing is submitted.

The list is recorded twice. In one rendering instance the tail is verified:
the driver takes the dynamic dirty bits for changes. With every draw in a
rendering instance of its own it is not, and the driver compares the state
the tail was built from with the live one.

usage: run-draw-state-reuse.py <shim.so> <brl-draw-state-bench>
"""
import os
import re
import subprocess
import sys

DRAWS = 64

# A draw that changes nothing publishes no render state at all.
NOTHING = "nothing"
# State the tail is not built from: the prepared tail is published again.
REUSE = "reuse"
# State that reaches the tail only through the root: a new fragment program
# in a copy of the last state block.
PROGRAM = "program"
# State the tail is built from.
BUILD = "build"

EXPECTED = {
    "plain": NOTHING,
    "base-instance": REUSE,
    "restart-enable": REUSE,
    "blend-constants": PROGRAM,
    "push-constants": PROGRAM,
    "stencil-ref": BUILD,
    "viewport": BUILD,
    "indexed": NOTHING,
    # Setting a state to the value it has dirties nothing.
    "same-viewport": NOTHING,
    # Away and back leaves a dirty bit, which a verified tail takes for a
    # change: it is built again, from the same state.
    "stencil-ref-back": BUILD,
    "scissor": BUILD,
    "cull-mode": BUILD,
    "front-face": BUILD,
    "depth-compare": BUILD,
    "depth-test": BUILD,
    "depth-write": BUILD,
    "depth-bias": BUILD,
    "stencil-op": BUILD,
    "stencil-write-mask": BUILD,
    "stencil-compare-mask": BUILD,
    "topology": BUILD,
    "stencil-test": BUILD,
    "depth-bias-enable": BUILD,
    "line-width": BUILD,
    "polygon-mode": BUILD,
    "depth-clamp": BUILD,
    "depth-clip": BUILD,
    "blend-equation": BUILD,
}

# Every draw in a rendering instance of its own: each republishes its render
# state, so a tail whose state is unchanged is published again, and a state
# set to another value and back compares equal.
EXPECTED_NEW_PASS = dict(EXPECTED, **{
    "plain": REUSE,
    "indexed": REUSE,
    "same-viewport": REUSE,
    "stencil-ref-back": REUSE,
    # A new rendering instance drops the state block a new fragment program
    # would be copied into (hk_cmd_buffer_dirty_all()).
    "blend-constants": BUILD,
    # A new root in a new rendering instance is not about the dynamic state.
    "push-constants": None,
})


def verdict(builds, programs, reuses, new_pass):
    if (builds, programs, reuses) == (1, 0, 0):
        return NOTHING
    # The first draw of a recording builds the tail. A scenario whose first
    # change sets the value the state already has changes it DRAWS - 1 times;
    # that draw publishes a tail only in a rendering instance of its own.
    for changed in (DRAWS - 1, DRAWS):
        idle = DRAWS - changed if new_pass else 0
        if (builds - 1, programs, reuses) == (0, 0, changed + idle):
            return REUSE
        if (builds - 1, programs, reuses) == (0, changed, idle):
            return PROGRAM
        if (builds - 1, programs, reuses) == (changed, 0, idle):
            return BUILD
    return "none of them"


def record(shim, bench, new_pass, expected):
    env = dict(os.environ, ASAHI_MESA_DEBUG="perf")
    preload = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = shim + (":" + preload if preload else "")
    env.setdefault("ASAHI_SHIM_CHIP", "g17p")
    run = subprocess.run([bench, str(DRAWS), "1", "all"]
                         + (["new-pass"] if new_pass else []), env=env,
                         capture_output=True, text=True)
    if run.returncode:
        sys.stderr.write(run.stdout + run.stderr)
        return 1

    names = re.findall(r"^(\S+)\s+min ", run.stdout, re.M)
    counts = re.findall(r"draws=(\d+) .*bind_tail=(\d+) bind_tail_program=(\d+)"
                        r" bind_tail_reuse=(\d+)", run.stderr)
    if sorted(names) != sorted(expected) or len(counts) != len(names):
        print(f"{len(names)} scenarios, {len(counts)} recordings, "
              f"{len(expected)} expected")
        return 1

    print("every draw in a rendering instance of its own:" if new_pass
          else "one rendering instance:")
    failed = 0
    for name, (draws, *tail) in zip(names, counts):
        if expected[name] is None:
            continue
        builds, programs, reuses = map(int, tail)
        got = verdict(builds, programs, reuses, new_pass)
        if int(draws) != DRAWS + 1:
            got = f"{draws} draws"
        ok = got == expected[name]
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:22s} full builds {builds:3d}"
              f"  program only {programs:3d}  reuses {reuses:3d}  -> {got}"
              + ("" if ok else f", expected {expected[name]}"))
    return failed


def main():
    shim, bench = sys.argv[1:3]
    failed = record(shim, bench, False, EXPECTED)
    failed += record(shim, bench, True, EXPECTED_NEW_PASS)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
