#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
extract_slices.py SNAPSHOT.csv [OUTDIR] - the three central planes of a director_field snapshot.

Writes x_SNAPSHOT.csv, y_SNAPSHOT.csv and z_SNAPSHOT.csv (in OUTDIR, default: the current folder) with the lines of
the planes x = Nx/2, y = Ny/2 and z = Nz/2 (integer division, with N = largest index + 1, the center used by the
sphere geometry), copied unchanged. Python 3 standard library only.
"""
import os
import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    with open(path) as f:
        header = f.readline()
        cols = [c.strip() for c in header.split(",")]
        idx = [cols.index(c) for c in ("x", "y", "z")]
        lines = f.readlines()
    rows = [[int(l.split(",")[i]) for i in idx] for l in lines]
    mid = [(max(r[a] for r in rows) + 1) // 2 for a in range(3)]
    name = os.path.basename(path)
    for a, axis in enumerate("xyz"):
        with open(os.path.join(out, axis + "_" + name), "w") as f:
            f.write(header)
            f.writelines(l for l, r in zip(lines, rows) if r[a] == mid[a])


if __name__ == "__main__":
    main()
