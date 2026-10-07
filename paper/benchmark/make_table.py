#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
make_table.py — performance table from results.csv (see run.sh).

For each case and mode, the cost of the Monte Carlo steps is the wall time of the run with N steps minus that of
the run with 0 steps (fastest of the repetitions of each), given as the time per site update,
t = cost / (N L^3), and as the speedup relative to one CPU thread. Writes table.tex (a table* for the appendix)
and prints a summary.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("ll32", r"LL, $32^3$"), ("ll64", r"LL, $64^3$"), ("ll128", r"LL, $128^3$"), ("ll256", r"LL, $256^3$"),
         ("ghrl64", r"GHRL, $64^3$")]


def main():
    best = defaultdict(lambda: float("inf"))
    alltimes = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "results.csv"))):
        key = (r["case"], r["mode"], int(r["threads"]), int(r["L"]), int(r["steps"]))
        best[key] = min(best[key], float(r["real_s"]))
        alltimes[key].append(float(r["real_s"]))
    cost = {}  # (case, mode, threads) -> (seconds of MC, ns per site update, N, L)
    for (c, m, th, L, n), t in best.items():
        if n == 0 or (c, m, th, L, 0) not in best:
            continue
        dt = t - best[(c, m, th, L, 0)]
        if dt <= 0.1 * t:  # the steps must be a sizable part of the run, otherwise the difference is noise
            print("warning: %s %s %d threads: the %d steps take %.2f s of %.2f s; increase SCALE" % (c, m, th, n, dt, t))
            continue
        cost[(c, m, th)] = (dt, 1e9 * dt / (n * L ** 3), n, L)
    # spread of the repetitions: each run with N steps minus the fastest run with 0 steps
    spread = {}
    for (c, m, th), (dt, ns, n, L) in cost.items():
        ts = [t - best[(c, m, th, L, 0)] for t in alltimes[(c, m, th, L, n)]]
        if len(ts) > 1:
            spread[(c, m, th)] = (max(ts) - min(ts)) / min(ts)
    if spread:
        worst = max(spread, key=spread.get)
        print("repetitions: %d; largest spread %.1f%% (%s %s %d threads), median %.1f%%" % (
            max(len(v) for v in alltimes.values()), 100 * spread[worst], worst[0], worst[1], worst[2],
            100 * sorted(spread.values())[len(spread) // 2]))
    modes = sorted({(m, th) for (c, m, th) in cost}, key=lambda x: (x[0] == "gpu", x[1]))
    cases = [(c, lab) for c, lab in CASES if any(k[0] == c for k in cost)]

    def label(m, th):
        return "GPU (CUDA)" if m == "gpu" else ("CPU, 1 thread" if th == 1 else "CPU, %d threads" % th)

    print("%-16s" % "" + "".join("%-24s" % c for c, _ in cases))
    rows = []
    for m, th in modes:
        cells, line = [], "%-16s" % label(m, th)
        for c, _ in cases:
            if (c, m, th) not in cost:
                cells.append("-- & --"); line += "%-24s" % "--"; continue
            dt, ns, n, L = cost[(c, m, th)]
            base = cost.get((c, "cpu", 1))
            sp = base[1] / ns if base else float("nan")  # no single-thread value: speedup unknown
            cells.append(r"%s & %.1f$\times$" % ("%.0f" % ns if ns >= 100 else "%.3g" % ns, sp))
            line += "%-24s" % ("%.3g ns (x%.3g) %.0fs" % (ns, sp, dt))
        rows.append(label(m, th) + " & " + " & ".join(cells) + r" \\")
        print(line)
    ncol = len(cases)
    head = " & ".join(r"\multicolumn{2}{c}{%s}" % lab for _, lab in cases)
    sub = " & ".join(r"$t$ (ns) & speedup" for _ in cases)
    rules = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(ncol))
    tex = r"""\begin{table*}[htbp]
\centering
\caption{Time per site update $t$ of the Monte Carlo step and speedup relative to one CPU thread, for the
Lebwohl--Lasher potential in periodic boxes of three sizes and for the GHRL potential (E7 constants). %s}
\label{tab:performance_benchmark}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l%s@{}}
\toprule
 & %s \\
%s
Execution mode & %s \\
\midrule
%s
\bottomrule
\end{tabular*}
\end{table*}
""" % ("MACHINE", "rr" * ncol, head, rules, sub, "\n".join(rows))
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(tex)
    print("-> table.tex (replace MACHINE in the caption by the processor and GPU used)")


if __name__ == "__main__":
    main()
