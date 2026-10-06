#!/usr/bin/env python3
"""
prepare.py — geometry and initial-condition files for the chiral cylinder test (Wand & Bates, PRE 100, 052702).

  python3 prepare.py geometry N L out.csv
      Solid cylinder of a N x N x L box (axis along z), same rules as c_cilinder.cpp: bulk (pt=1) inside the circle of
      radius (N-1)/2 - 1, homeotropic surface shell (pt=2) up to (N-1)/2 + 1, empty space (pt=0) outside; radial normals.

  python3 prepare.py helix N L xi sign out.csv
      Idealized double helix of two s=+1/2 disclination lines: in each layer the director is the field of two +1/2
      defects at distance R/2 from the axis (radial far from them, i.e. compatible with homeotropic anchoring), and the
      whole pattern rotates by 2*pi*xi*sign*z/L along the cylinder. xi is the number of full turns of the pair over the
      length L (xi = n/2, half turns allowed: the pair is symmetric under a rotation by pi).

  python3 prepare.py stretch snapshot.csv[.gz] N Lsrc Ldst out.csv
      Initial condition made from a relaxed snapshot of length Lsrc stretched to Ldst along z (layer k of the new box
      copies layer floor(k*Lsrc/Ldst)). The twist of the double helix (turns over the periodic length) is kept, so the
      repeat distance becomes lambda*Ldst/Lsrc with the relaxed texture of the cooled runs.

  python3 prepare.py pick RUNS p0 R
      Prints the directory of the lowest-energy scan run (RUNS/scan/p<p0>/R<R>/L*/seed*) whose double helix (two +1/2 lines,
      no -1/2) makes one full turn over its length (the source for the stretch stage); prints nothing if there is none.
"""
import sys

import numpy as np


def geometry(N, L, out):
    c = (N - 1) / 2.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rx, ry = i - c, j - c
    r = np.hypot(rx, ry)
    pt2d = np.where(r <= c - 1, 1, np.where(r <= c + 1, 2, 0))
    rr = np.where(r == 0, 1, r)
    nx, ny = np.where(pt2d > 0, rx / rr, 0.0), np.where(pt2d > 0, ry / rr, 0.0)
    nz = np.where(pt2d > 0, 0.0, 1.0)
    with open(out, "w") as f:
        f.write("x,y,z,nx,ny,nz,pt\n")
        for k in range(L):
            for jj in range(N):
                for ii in range(N):
                    f.write(f"{ii},{jj},{k},{nx[ii, jj]:.6g},{ny[ii, jj]:.6g},{nz[ii, jj]:.6g},{pt2d[ii, jj]}\n")


def helix(N, L, xi, sign, out):
    c = (N - 1) / 2.0
    R = c - 1
    d = R / 2.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    x, y = i - c, j - c
    with open(out, "w") as f:
        f.write("x,y,z,nx,ny,nz\n")
        for k in range(L):
            a = 2 * np.pi * xi * sign * k / L
            ca, sa = np.cos(a), np.sin(a)
            xr, yr = ca * x + sa * y, -sa * x + ca * y          # coordinates in the rotating frame
            th = 0.5 * (np.arctan2(yr, xr - d) + np.arctan2(yr, xr + d)) + a
            nx, ny = np.cos(th), np.sin(th)
            for jj in range(N):
                for ii in range(N):
                    f.write(f"{ii},{jj},{k},{nx[ii, jj]:.6g},{ny[ii, jj]:.6g},0\n")


def stretch(snap, N, Ls, Ld, out):
    import pandas as pd
    df = pd.read_csv(snap)
    n = np.zeros((N, N, Ls, 3))
    n[df["x"], df["y"], df["z"]] = df[["nx", "ny", "nz"]].to_numpy()
    with open(out, "w") as f:
        f.write("x,y,z,nx,ny,nz\n")
        for k in range(Ld):
            ks = (k * Ls) // Ld
            for jj in range(N):
                for ii in range(N):
                    v = n[ii, jj, ks]
                    f.write(f"{ii},{jj},{k},{v[0]:.6g},{v[1]:.6g},{v[2]:.6g}\n")


def pick(runs, p0, R):
    import glob
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_figures as mf
    best = None
    for d in glob.glob(os.path.join(runs, "scan", f"p{p0}", f"R{R}", "L*")):
        for r in mf.runs_in(d):
            prm = mf.read_input(r.attrs["dir"])
            shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
            snap = mf.last_snapshot(r.attrs["dir"])
            if snap is None:
                continue
            nv, pt, S = mf.load(snap, shape)
            xi = mf.SIGN * mf.pair_angle(nv, pt, S, int(R))[2]
            E = float(r["E"].iloc[-1])
            st = mf.structure(nv, pt)
            clean = abs(st["nplus"] - 2) < 0.2 and st["nminus"] < 0.1          # just the two +1/2 lines
            if clean and np.isfinite(xi) and abs(xi - 1) < 0.15 and (best is None or E < best[0]):
                best = (E, r.attrs["dir"])
    if best:
        print(best[1])


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "geometry":
        geometry(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) >= 7 and sys.argv[1] == "helix":
        helix(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif len(sys.argv) >= 7 and sys.argv[1] == "stretch":
        stretch(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif len(sys.argv) >= 5 and sys.argv[1] == "pick":
        pick(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(1)
