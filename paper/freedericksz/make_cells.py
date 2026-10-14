#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
make_cells.py — cross-sections of the three Freedericksz cells (panels (a)-(c) of the Freedericksz figure of the paper),
drawn in the same style as the droplet cross-sections of paper/examples/make_figures.py (headless director segments,
blue-grey-red colour map, horizontal colour bar).

For each cell (splay, twist, bend) the snapshot of one run at the field E closest to RATIO * E_c is read, with E_c the
critical field of figs/numbers.txt (written by make_figures.py). The plane is the central x-z plane (y = Ny/2): the
segments are the projections of the directors onto that plane, and their colour is the deformation angle
theta = acos(|n . n0|), with n0 the initial easy axis (x for the planar cells, z for the homeotropic one). In the twist
cell the director rotates out of the plane, so the segments become shorter where theta is large. The grey bars are the
two plates (z = 0 and z = Nz - 1 are the anchored layers); the arrow (or the dot, for a field normal to the plane)
shows the field direction.

  python3 make_cells.py                   -> figs/fig_freedericksz_cells.pdf (and .png)
  python3 make_cells.py --ratio 1.5 --seed 2
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from make_figures import HERE, open_snapshot, read_params, snapshot_scale

MM = 1 / 25.4
INK, INK2 = "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("ndr", ["#2a78d6", "#c9c8c3", "#e34948"])   # as in the droplet sections
CELLS = [("splay", "x", "Splay"), ("twist", "x", "Twist"), ("bend", "z", "Bend")]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "savefig.dpi": 300, "figure.facecolor": "white",
})


def critical_fields(path):
    ec = {}
    for line in open(path):
        m = re.match(r"(\w+)\s+seeds\s+\d+\s+E_c = ([0-9.]+)", line)
        if m:
            ec[m.group(1)] = float(m.group(2))
    return ec


def load(run, E):
    """Director field (nx, ny, nz, 3) and point types of the snapshot at the available field closest to E."""
    p = read_params(run)
    scale = snapshot_scale(float(p["de"]))
    with open(os.path.join(run, "po.dat")) as f:
        next(f)
        fields = np.array([float(l.split()[0]) for l in f if l.strip()])
    Eu = float(fields[np.argmin(abs(fields - E))])
    shape = tuple(int(p[k]) for k in ("nx", "ny", "nz"))
    nv = np.zeros(shape + (3,))
    pt = np.zeros(shape, int)
    with open_snapshot(run, int(scale * (Eu + 1e-3 / scale))) as f:
        h = next(f).strip().split(",")
        ix = [h.index(k) for k in ("x", "y", "z", "nx", "ny", "nz", "pt")]
        for line in f:
            w = line.split(",")
            i, j, k = int(w[ix[0]]), int(w[ix[1]]), int(w[ix[2]])
            nv[i, j, k] = [float(w[c]) for c in ix[3:6]]
            pt[i, j, k] = int(w[ix[6]])
    return nv, pt, Eu, p


def section(ax, nv, pt, axis):
    """Headless segments on the central x-z plane, coloured by theta = acos(|n . n0|) in degrees."""
    nx, ny, nz = pt.shape
    j = ny // 2
    a = "xyz".index(axis)
    X, Z, U, V, C = [], [], [], [], []
    for i in range(nx):
        for k in range(nz):
            if pt[i, j, k] == 0:
                continue
            n = nv[i, j, k]
            s = 1 if n[0] >= 0 else -1          # n and -n are the same state: draw with a fixed sign
            X.append(i); Z.append(k); U.append(s * n[0]); V.append(s * n[2])
            C.append(np.degrees(np.arccos(min(1.0, abs(n[a])))))
    q = ax.quiver(X, Z, U, V, C, cmap=CMAP, clim=(0, 90), pivot="mid", angles="xy", scale_units="xy",
                  scale=1 / 0.85, headwidth=1, headlength=0, headaxislength=0, width=0.022)
    for z in (-0.75, nz - 0.25):                 # the two plates
        ax.add_patch(plt.Rectangle((-0.6, z), nx + 0.2, 0.5, color="#c9c8c3", lw=0))
    ax.set_xlim(-0.8, nx - 0.2)
    ax.set_ylim(-1.0, nz + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return q


def field_marker(ax, p, nx, nz):
    e = np.array([float(p.get(k, 0)) for k in ("elecx", "elecy", "elecz")])
    x0, z0 = nx + 0.6, nz / 2
    if abs(e[1]) > 0.5:                         # field along y, normal to the plane: dot in a circle
        ax.plot([x0], [z0], marker="o", ms=6, mfc="white", mec=INK, mew=0.8, clip_on=False)
        ax.plot([x0], [z0], marker="o", ms=1.5, color=INK, clip_on=False)
    else:
        ax.annotate("", xy=(x0, z0 + 3), xytext=(x0, z0 - 3), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7))
    ax.text(x0 + 0.9, z0, r"$\mathbf{E}$", ha="left", va="center", fontsize=8, color=INK, clip_on=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--ratio", type=float, default=1.3, help="field of the sections in units of E_c (default 1.3)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ec = critical_fields(os.path.join(args.out, "numbers.txt"))
    fig, axes = plt.subplots(1, 3, figsize=(88 * MM, 62 * MM), gridspec_kw={"wspace": 0.55})
    lines = []
    for ax, (cell, axis, title), letter in zip(axes, CELLS, "abc"):
        nv, pt, Eu, p = load(os.path.join(args.runs, cell, "seed%d" % args.seed), args.ratio * ec[cell])
        q = section(ax, nv, pt, axis)
        field_marker(ax, p, pt.shape[0], pt.shape[2])
        ax.set_title(title, fontsize=8, color=INK, pad=2)
        ax.text(-0.05, 1.02, "(%s)" % letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
        lines.append("%-6s seed %d  E = %.3f (E/E_c = %.2f)" % (cell, args.seed, Eu, Eu / ec[cell]))
    cb = fig.colorbar(q, ax=list(axes), orientation="horizontal", fraction=0.05, pad=0.04, aspect=30,
                      ticks=[0, 30, 60, 90])
    cb.set_label(r"deformation angle $\theta=\arccos|\hat{\mathbf{n}}\cdot\hat{\mathbf{n}}_0|$ (degrees)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, "fig_freedericksz_cells.%s" % ext), bbox_inches="tight")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
