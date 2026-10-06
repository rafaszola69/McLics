#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
make_figures.py — Freedericksz test problem (see run.sh): deformation angle versus field and critical fields.

For each cell (splay, twist, bend) and field value, the deformation angle is the mean over the two middle layers of
the cell of acos(|n . n0|), with n0 the initial easy axis (x for the planar cells, z for the homeotropic one),
averaged over the seeds. Below the threshold the angle stays at a small thermal baseline (about 0.5 degrees at
T = 0.0008).

Critical field E_c: above the threshold, theta_m^2 grows linearly with E - E_c. E_c is where the straight line fitted
to theta^2 versus E, for theta between max(3 x baseline, 2) and 40 degrees, reaches the mean theta^2 of the points
well below the threshold (baseline = median angle of the first five field values).
The field where the angle crosses 20 degrees (interpolated) is also given. From E_c^2 ~ K_i:
K1/K2 = (E_c splay / E_c twist)^2 and K3/K2 = (E_c bend / E_c twist)^2 (input: 1.48 and 2.28).

  python3 make_figures.py [--runs runs] [--out figs]
Writes figs/fig_freedericksz.pdf/.png, figs/freedericksz_angles.csv and figs/numbers.txt.
"""
import argparse
import glob
import gzip
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("splay", "x", "Splay"), ("twist", "x", "Twist"), ("bend", "z", "Bend")]
K_INPUT = {"K1/K2": 1.48, "K3/K2": 2.28}


def read_params(run):
    p = {}
    for line in open(os.path.join(run, "input_parameters.txt")):
        w = line.split("#")[0].split()
        if len(w) >= 2:
            p.setdefault(w[0].lower(), w[1])
    return p


def snapshot_scale(step):
    """Same rule as include/sweep.h."""
    s = 100
    while s * abs(step) < 1 - 1e-9 and s < 100000000:
        s *= 10
    return s


def open_snapshot(run, idx):
    f = os.path.join(run, "director_field_%d.csv" % idx)
    if os.path.exists(f + ".gz"):
        return gzip.open(f + ".gz", "rt")
    return open(f)


def angle_curve(run, axis):
    """Field values and mid-cell deformation angle (degrees) of one run."""
    p = read_params(run)
    scale = snapshot_scale(float(p["de"]))
    with open(os.path.join(run, "po.dat")) as f:
        next(f)
        fields = [float(l.split()[0]) for l in f if l.strip()]
    a = "xyz".index(axis)
    out = []
    for E in fields:
        idx = int(scale * (E + 1e-3 / scale))
        with open_snapshot(run, idx) as f:
            header = next(f).strip().split(",")
            cz, cpt, cn = header.index("z"), header.index("pt"), header.index("n" + axis)
            rows = [l.split(",") for l in f]
        nz = max(int(r[cz]) for r in rows) + 1
        mid = {nz // 2 - 1, nz // 2} if nz % 2 == 0 else {nz // 2}
        ang = [math.degrees(math.acos(min(1.0, abs(float(r[cn]))))) for r in rows
               if int(r[cz]) in mid and int(r[cpt]) > 0]
        out.append(sum(ang) / len(ang))
    return np.array(fields), np.array(out)


def critical_field(E, th):
    """(E_c from the theta^2 fit, E at 20 degrees); None where not determined."""
    lo = max(3 * np.median(th[:5]), 2.0)
    above = np.where(th > lo)[0]
    if len(above) == 0:
        return None, None
    ion = above[0]
    e20 = None
    for i in range(1, len(E)):
        if th[i - 1] < 20 <= th[i]:
            e20 = E[i - 1] + (20 - th[i - 1]) * (E[i] - E[i - 1]) / (th[i] - th[i - 1])
            break
    base = th[E < E[ion] - 0.05]
    sel = (E >= E[ion]) & (th > lo) & (th < 40)
    if len(base) < 3 or sel.sum() < 3:
        return None, e20
    b = np.mean(base ** 2)
    slope, icpt = np.polyfit(E[sel], th[sel] ** 2, 1)
    return (b - icpt) / slope, e20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, lines, table = {}, [], {}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    colors = {"splay": "#2a78d6", "twist": "#eb6834", "bend": "#3a9a4a"}
    for cell, axis, label in CELLS:
        runs = sorted(d for d in glob.glob(os.path.join(args.runs, cell, "seed*"))
                      if os.path.exists(os.path.join(d, "DONE")))
        if not runs:
            print("no finished runs for %s" % cell)
            continue
        curves = [angle_curve(d, axis) for d in runs]
        n = min(len(c[0]) for c in curves)
        E = curves[0][0][:n]
        th = np.array([c[1][:n] for c in curves])
        mean = th.mean(axis=0)
        table[cell] = (E, mean)
        ec, e20 = critical_field(E, mean)
        per_seed = [critical_field(E, t)[0] for t in th]
        per_seed = [x for x in per_seed if x is not None]
        err = np.std(per_seed, ddof=1) / math.sqrt(len(per_seed)) if len(per_seed) > 1 else float("nan")
        res[cell] = (ec, err, e20)
        lines.append("%-6s seeds %d  E_c = %s +- %s   E(20 deg) = %s   per seed: %s" % (
            cell, len(runs), "%.4f" % ec if ec else "n/a", "%.4f" % err, "%.4f" % e20 if e20 else "n/a",
            " ".join("%.4f" % x for x in per_seed)))
        ax.plot(E, mean, color=colors[cell], lw=1.6, label=label)
        if len(runs) > 1:
            ax.fill_between(E, th.min(axis=0), th.max(axis=0), color=colors[cell], alpha=0.18, lw=0)
        if ec:
            ax.axvline(ec, color=colors[cell], ls="--", lw=0.8)
    ax.set_xlabel("Field $E$")
    ax.set_ylabel(r"Deformation angle at mid-cell (deg)")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, "fig_freedericksz.%s" % ext), dpi=300)

    if all(c in res and res[c][0] for c in ("splay", "twist", "bend")):
        es, et, eb = (res[c][0] for c in ("splay", "twist", "bend"))
        lines.append("K1/K2 = (E_c splay / E_c twist)^2 = %.3f   (input %.2f)" % ((es / et) ** 2, K_INPUT["K1/K2"]))
        lines.append("K3/K2 = (E_c bend  / E_c twist)^2 = %.3f   (input %.2f)" % ((eb / et) ** 2, K_INPUT["K3/K2"]))
        s20 = [res[c][2] for c in ("splay", "twist", "bend")]
        if all(s20):
            lines.append("from the 20-degree fields: K1/K2 = %.3f, K3/K2 = %.3f" % (
                (s20[0] / s20[1]) ** 2, (s20[2] / s20[1]) ** 2))
    with open(os.path.join(args.out, "numbers.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "freedericksz_angles.csv"), "w") as f:
        cells = [c for c, _, _ in CELLS if c in table]
        f.write("E," + ",".join("angle_" + c for c in cells) + "\n")
        n = min(len(table[c][0]) for c in cells)
        for i in range(n):
            f.write("%.4f," % table[cells[0]][0][i] + ",".join("%.3f" % table[c][1][i] for c in cells) + "\n")
    print("\n".join(lines))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
