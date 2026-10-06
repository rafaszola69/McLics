#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compares the po.dat files (and snapshot names) of the old and new runs made by run_regression.sh.

Rows are matched by the value of the first column (temperature, field or file index), so a sweep in which one
version has an extra last point is still compared on the common points.
"""
import glob
import os
import sys

import numpy as np

# Known, intended differences between two versions can be listed here (case -> note), e.g. when a release
# renames a column. Empty: any difference is reported as such.
EXPECTED = {}


def read_po(path):
    lines = open(path).read().split("\n")
    header = lines[0].split()
    rows = [l.split() for l in lines[1:] if l.strip()]
    return header, rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    print("\n%-28s %-10s %s" % ("case", "result", "details"))
    n_bad = 0
    for case_dir in sorted(glob.glob(os.path.join(out, "*"))):
        if not os.path.isdir(case_dir):
            continue
        case = os.path.basename(case_dir)
        st = {v: open(os.path.join(case_dir, v, "status")).read().split() for v in ("old", "new")}
        if st["old"][0] != "0" or st["new"][0] != "0":
            print("%-28s %-10s exit codes old=%s new=%s (see run.log)" % (case, "FAILED", st["old"][0], st["new"][0]))
            n_bad += 1
            continue
        h_old, r_old = read_po(os.path.join(case_dir, "old", "po.dat"))
        h_new, r_new = read_po(os.path.join(case_dir, "new", "po.dat"))
        notes = []
        if h_old != h_new:
            notes.append("header %s -> %s" % (" ".join(h_old), " ".join(h_new)))
        old_by_x = {round(float(r[0]), 6): r for r in r_old}
        new_by_x = {round(float(r[0]), 6): r for r in r_new}
        common = sorted(set(old_by_x) & set(new_by_x))
        only_old = sorted(set(old_by_x) - set(new_by_x))
        only_new = sorted(set(new_by_x) - set(old_by_x))
        if only_old:
            notes.append("rows only in old: %s" % only_old)
        if only_new:
            notes.append("rows only in new: %s" % only_new)
        identical = all(old_by_x[x][1:] == new_by_x[x][1:] for x in common)
        if identical:
            result = "IDENTICAL"
        else:
            a = np.array([[float(v) for v in old_by_x[x][1:]] for x in common])
            b = np.array([[float(v) for v in new_by_x[x][1:]] for x in common])
            rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))
            result = "DIFFERENT"
            notes.append("max relative difference %.2e over %d common rows" % (rel, len(common)))
        snaps = {v: sorted(os.path.basename(f) for f in glob.glob(os.path.join(case_dir, v, "director_field_*.csv")))
                 for v in ("old", "new")}
        if snaps["old"] != snaps["new"]:
            notes.append("snapshot names differ: old %s, new %s" % (snaps["old"], snaps["new"]))
        t = (float(st["old"][1]), float(st["new"][1]))
        notes.append("time old %.1f s, new %.1f s" % t)
        if case in EXPECTED and (h_old != h_new or snaps["old"] != snaps["new"]):
            notes.append("(" + EXPECTED[case] + ")")
        if result != "IDENTICAL":
            n_bad += 1
        print("%-28s %-10s %s" % (case, result, "; ".join(notes)))
    print("\n%s" % ("all po.dat values identical on the common rows" if n_bad == 0
                    else "%d case(s) with differences or failures: see out/summary.txt and the run.log files" % n_bad))


if __name__ == "__main__":
    main()
