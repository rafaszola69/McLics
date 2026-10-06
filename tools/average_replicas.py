#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
average_replicas.py — combine independent replicas of po.dat
=============================================================
In systems with a degenerate energy landscape (e.g. a nematic droplet with homeotropic anchoring), a single Monte
Carlo run can freeze into any of several defect textures with similar energies but quite different global order
parameters S. A single run is then a poor estimate of the typical behaviour of the system.

This script combines N independent replicas (same input file, different `seed`) into one curve with the correct
statistics.

Usage:
    python3 tools/average_replicas.py run_seed1/po.dat run_seed2/po.dat ... -o po_avg.dat
    python3 tools/average_replicas.py replicas/seed*/po.dat -o po_avg.dat --txt

Making the replicas:
    copy the input file to N folders, changing only the `seed` line (1, 2, 3, ...), run mclics in each of them,
    then combine their po.dat files here.

Statistics:
    At each T, the replicas are combined with the law of total variance:
        S_mean      = mean(S_r)                       over the replicas r
        Var(S) total = mean(varS_r)  +  var(S_r)
                       [variance within each replica] + [variance between replicas]
    The second term is what is lost when looking at a single run: it is the variability of "which defect texture
    formed". The extra columns (P, dtheta, radial order, ...) are plain means over the replicas.
"""

import sys
import argparse
import numpy as np


def read_po_dat(path):
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line[0].replace('.', '').replace('-', '').isdigit():
                header = line
                continue
            parts = line.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        print(f"  [ERROR] No valid data in {path}")
        sys.exit(1)
    return header, rows


def combine(paths):
    headers = []
    per_file = []
    for p in paths:
        header, rows = read_po_dat(p)
        headers.append(header)
        per_file.append(rows)

    ncols = len(per_file[0][0])
    if not all(len(r) == ncols for rows in per_file for r in rows):
        print("  [ERROR] The replicas have different numbers of columns (e.g. one uses the pear potential and another does not).")
        sys.exit(1)

    lengths = [len(rows) for rows in per_file]
    if len(set(lengths)) != 1:
        print(f"  [WARNING] The replicas have different numbers of points: {lengths}")
        print("  [WARNING] Using only the first min(N) points of each replica.")
    n_points = min(lengths)

    combined = []
    for i in range(n_points):
        T_vals = [rows[i][0] for rows in per_file]
        if max(T_vals) - min(T_vals) > 1e-3:
            print(f"  [WARNING] Point {i}: the temperatures differ between replicas ({T_vals}); skipped.")
            continue
        T = np.mean(T_vals)

        S_r = np.array([rows[i][1] for rows in per_file])
        varS_r = np.array([rows[i][2] for rows in per_file])
        E_r = np.array([rows[i][3] for rows in per_file])
        varE_r = np.array([rows[i][4] for rows in per_file])

        S_mean = S_r.mean()
        # Law of total variance: Var[S] = E[Var(S|replica)] + Var[E(S|replica)]
        S_var_total = varS_r.mean() + S_r.var(ddof=0)

        E_mean = E_r.mean()
        E_var_total = varE_r.mean() + E_r.var(ddof=0)

        row = [T, S_mean, S_var_total, E_mean, E_var_total]

        # Extra columns (P, dtheta, radial order, ...): plain mean over the replicas.
        for c in range(5, ncols):
            row.append(np.mean([rows[i][c] for rows in per_file]))

        combined.append(row)

    return headers[0] or "T S varS E varE", combined


def main():
    parser = argparse.ArgumentParser(
        description="Combines independent replicas of po.dat (same parameters, different seeds).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="po.dat file of each replica")
    parser.add_argument("-o", "--output", default="po_avg.dat", help="output file (default: po_avg.dat)")
    parser.add_argument("--txt", action="store_true", help="print the combined table on the terminal")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("  [WARNING] Only one replica given: without independent replicas the variability between")
        print("  textures cannot be estimated. The result is identical to the input po.dat.")

    header, combined = combine(args.inputs)

    with open(args.output, "w") as f:
        f.write(header + "\n")
        for row in combined:
            f.write(" ".join(f"{v:g}" for v in row) + "\n")

    print(f"\n  {len(args.inputs)} replicas combined -> {args.output}")
    print(f"  {len(combined)} points\n")

    if args.txt:
        print(f"  {'T':>8} {'S':>10} {'varS':>12} {'E':>10} {'varE':>12}")
        for row in combined:
            print(f"  {row[0]:>8.4f} {row[1]:>10.4f} {row[2]:>12.4g} {row[3]:>10.4f} {row[4]:>12.4g}")
        print()


if __name__ == "__main__":
    main()
