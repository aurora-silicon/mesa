#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""List how the Aurora driver trees have diverged from upstream Mesa's asahi driver.

src/aurora (Borealis, the shared compiler/ISA/layout/runtime, tools, CI) and the Gallium driver in
src/gallium/{drivers,winsys}/aurora started as upstream Mesa's src/asahi and
src/gallium/{drivers,winsys}/asahi. src/aurora/UPSTREAM names the upstream commit and the path
pairs; src/asahi in this tree is upstream's, unchanged and not built.

For each pair this prints how many upstream files are unchanged, changed or gone in our tree and
how many files are ours only, after undoing the mechanical renames of the move (aurora/ includes,
the brl_ prefix and file names, Borealis in prose). src/aurora is compared with the in-tree
src/asahi, so no git is needed. The Gallium pairs are compared with the upstream commit through
git when it is in the repository, and skipped otherwise.

  aurora-divergence.py [--root DIR] [--files] [--upstream REF]

--files lists every changed, ours-only and gone file. --upstream REF (for example FETCH_HEAD after
'git fetch https://gitlab.freedesktop.org/mesa/mesa.git main') also lists the upstream commits after
the recorded one that touch the asahi trees: the candidates to cross-port.
"""
import argparse
import os
import re
import subprocess
import sys

NAME = re.compile(r"\b(Borealis|Honeykrisp|hk|brl)\b")


def normalise(text):
    text = re.sub(r'(#\s*include\s*[<"])aurora/', r"\1asahi/", text)
    text = re.sub(r"(?<![A-Za-z0-9])brl(?=_)", "hk", text)
    text = re.sub(r"(?<![A-Za-z0-9])BRL(?=_)", "HK", text)
    text = re.sub(r"(?<=[-/])brl(?=[-_.])|(?<![A-Za-z0-9_\-])brl(?=-)", "hk", text)
    text = text.replace("src/aurora/", "src/asahi/").replace("drivers/aurora", "drivers/asahi")
    text = text.replace("winsys/aurora", "winsys/asahi")
    return NAME.sub("@", text)


def their_name(rel):
    return re.sub(r"(^|/)brl_", r"\1hk_", re.sub(r"(?<=[-_])brl(?=[-_.])", "hk", rel))


def walk(top):
    out = {}
    for d, _, files in os.walk(top):
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                out[os.path.relpath(p, top)] = p
    return out


def git(root, *a):
    try:
        return subprocess.run(["git", "-C", root, *a], capture_output=True)
    except FileNotFoundError:  # no git installed: behave like a repository without the commit
        return subprocess.CompletedProcess(a, 127, b"", b"git not found")


def compare(ours, theirs, read_theirs, files):
    seen, same, changed, mine = set(), 0, [], []
    for rel, path in sorted(ours.items()):
        t = their_name(rel)
        if t not in theirs:
            mine.append(rel)
            continue
        seen.add(t)
        a = open(path, "rb").read()
        b = read_theirs(t)
        try:
            eq = normalise(a.decode()) == normalise(b.decode())
        except UnicodeDecodeError:
            eq = a == b
        if eq:
            same += 1
        else:
            changed.append(rel)
    gone = sorted(set(theirs) - seen)
    return same, changed, mine, gone


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.normpath(os.path.join(here, "..", "..", "..")))
    ap.add_argument("--files", action="store_true")
    ap.add_argument("--upstream")
    a = ap.parse_args()
    base, maps = None, []
    for line in open(os.path.join(a.root, "src/aurora/UPSTREAM")):
        f = line.split()
        if f and f[0] == "commit":
            base = f[1]
        elif f and f[0] == "map":
            maps.append((f[1], f[2]))
    have_git = git(a.root, "cat-file", "-e", f"{base}^{{commit}}").returncode == 0 if base else False
    print(f"Aurora trees vs upstream Mesa {base[:12]}")
    for ours_dir, theirs_dir in maps:
        ours = walk(os.path.join(a.root, ours_dir))
        local = os.path.join(a.root, theirs_dir)
        if os.path.isdir(local):
            theirs = walk(local)
            read = lambda t, local=local: open(os.path.join(local, t), "rb").read()
            src = "in-tree"
        elif have_git:
            names = git(a.root, "ls-tree", "-r", "--name-only", f"{base}:{theirs_dir}").stdout.decode().split("\n")
            theirs = {n: n for n in names if n}
            read = lambda t, d=theirs_dir: git(a.root, "show", f"{base}:{d}/{t}").stdout
            src = "git"
        else:
            print(f"  {ours_dir:28} <- {theirs_dir}: upstream copy not available (skipped)")
            continue
        same, changed, mine, gone = compare(ours, theirs, read, a.files)
        print(f"  {ours_dir:28} <- {theirs_dir:26} ({src}): upstream files {len(theirs)}: unchanged {same}, "
              f"changed {len(changed)}, gone {len(gone)}; ours only {len(mine)}")
        if a.files:
            for kind, lst in (("changed", changed), ("ours", mine), ("gone", gone)):
                for f in lst:
                    print(f"    {kind:8} {f}")
    if a.upstream:
        r = git(a.root, "log", "--oneline", "--no-merges", f"{base}..{a.upstream}", "--", *[t for _, t in maps])
        lines = r.stdout.decode().splitlines()
        print(f"upstream commits after {base[:12]} touching the asahi trees: {len(lines)}")
        for l in lines:
            print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
