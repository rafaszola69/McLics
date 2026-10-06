#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
make_cylinder.py N L OUTDIR — geometry and initial condition of the cylinder case of the regression test.

Writes OUTDIR/cyl_N<N>_L<L>.csv (custom geometry: a cylinder along z in an N x N x L box, bulk inside the radius
(N-1)/2 - 1, a homeotropic surface shell up to (N-1)/2 + 1, empty outside, radial normals) and
OUTDIR/helix_N<N>_L<L>.csv (two +1/2 disclination lines making one full turn over L). Same rules as
paper/cylinder/prepare.py, with the Python standard library only.
"""
import math
import os
import sys


def main():
    N, L, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    c = (N - 1) / 2.0
    d = (c - 1) / 2.0
    with open(os.path.join(out, "cyl_N%d_L%d.csv" % (N, L)), "w") as g, \
            open(os.path.join(out, "helix_N%d_L%d.csv" % (N, L)), "w") as h:
        g.write("x,y,z,nx,ny,nz,pt\n")
        h.write("x,y,z,nx,ny,nz\n")
        for k in range(L):
            a = -2 * math.pi * k / L
            ca, sa = math.cos(a), math.sin(a)
            for j in range(N):
                for i in range(N):
                    x, y = i - c, j - c
                    r = math.hypot(x, y)
                    pt = 1 if r <= c - 1 else (2 if r <= c + 1 else 0)
                    rr = r if r > 0 else 1.0
                    nx, ny, nz = (x / rr, y / rr, 0.0) if pt else (0.0, 0.0, 1.0)
                    g.write("%d,%d,%d,%.6g,%.6g,%.6g,%d\n" % (i, j, k, nx, ny, nz, pt))
                    xr, yr = ca * x + sa * y, -sa * x + ca * y
                    th = 0.5 * (math.atan2(yr, xr - d) + math.atan2(yr, xr + d)) + a
                    h.write("%d,%d,%d,%.6g,%.6g,0\n" % (i, j, k, math.cos(th), math.sin(th)))


if __name__ == "__main__":
    main()
