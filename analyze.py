# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
analyze.py — plots the results of McLics (po.dat)
=================================================
Usage:
    python3 analyze.py                        # reads po.dat in the current folder
    python3 analyze.py results/po.dat         # reads a given file
    python3 analyze.py po.dat -o plot.png     # saves to a given file
    python3 analyze.py po.dat --show          # also shows the interactive plot
    python3 analyze.py po.dat --txt           # prints a table of statistics
    python3 analyze.py po.dat --dark          # dark theme (presentations)
    python3 analyze.py po.dat --mode electric # sets the x-axis label explicitly

Requires numpy, pandas and matplotlib (e.g. in a virtual environment:
python3 -m venv venv && source venv/bin/activate && pip install numpy pandas matplotlib).

Output:
    - Plot with three stacked panels: S(x), E(x) and Var(S)
    - Error bands (±1 standard deviation) and an estimate of the phase transition
    - Table of statistics on the terminal (with --txt)

The evolution mode is detected from the first column of the po.dat header: T (thermal), field (electric) or ii
(step or quench, labelled as step). --mode overrides it.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False


# ── Reading the file ──────────────────────────────────────────────────────────

# x-axis label and short name of each evolution mode
MODE_INFO = {
    "thermal":  ("Temperature (T)", "T"),
    "electric": ("Electric field (E)", "E"),
    "step":     ("Block (index)", "block"),
    "quench":   ("Quench block (index)", "block"),
}
# first column of the po.dat header -> mode
HEADER_MODE = {"T": "thermal", "field": "electric", "ii": "step"}


def read_po_dat(file_path, forced_mode=None):
    """
    Reads po.dat and returns a DataFrame with the columns x, S, varS, E, varE, where x is the temperature, the
    field or the block number. The mode is forced_mode if given, otherwise detected from the header.
    """
    if not os.path.isfile(file_path):
        print(f"  [ERROR] File not found: {file_path}")
        sys.exit(1)

    rows = []
    detected = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # header line ("T S varS E varE ..."): its first column gives the mode
            if not line[0].replace('.', '').replace('-', '').isdigit():
                detected = HEADER_MODE.get(line.split()[0], detected)
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(p) for p in parts[:5]])
            except ValueError:
                continue

    if not rows:
        print(f"  [ERROR] No valid data in {file_path}.")
        sys.exit(1)

    mode = forced_mode or detected or "thermal"
    xlabel, xcol = MODE_INFO[mode]

    df = pd.DataFrame(rows, columns=["x", "S", "varS", "E", "varE"])
    df["errS"] = np.sqrt(df["varS"].clip(lower=0))
    df["errE"] = np.sqrt(df["varE"].clip(lower=0))

    if mode == "thermal":
        df = df.sort_values("x").reset_index(drop=True)

    return df, mode, xlabel, xcol


# ── Phase transition ──────────────────────────────────────────────────────────

def find_transition(df):
    """
    Estimates the transition temperature (or field) as the point of largest |dS/dx|. Returns None if the change
    is too small.
    """
    if len(df) < 4:
        return None
    dS = np.abs(np.gradient(df["S"].values, df["x"].values))
    idx = np.argmax(dS)
    # report it only if the change is significant
    if dS[idx] < 0.01:
        return None
    return df["x"].iloc[idx]


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(df, xlabel):
    """Prints a table of descriptive statistics on the terminal."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Statistics  —  {len(df)} points")
    print(sep)
    print(f"  {'':12s}  {'S':>10s}  {'E':>10s}")
    print(f"  {'Minimum':12s}  {df['S'].min():10.4f}  {df['E'].min():10.4f}")
    print(f"  {'Maximum':12s}  {df['S'].max():10.4f}  {df['E'].max():10.4f}")
    print(f"  {'Mean':12s}  {df['S'].mean():10.4f}  {df['E'].mean():10.4f}")
    print(f"  {'Std. dev.':12s}  {df['S'].std():10.4f}  {df['E'].std():10.4f}")

    Tx = find_transition(df)
    if Tx is not None:
        print(f"\n  Estimated phase transition at x ≈ {Tx:.4f}")
    print(sep + "\n")


# ── Palette ───────────────────────────────────────────────────────────────────

def get_palette(dark=False):
    """
    Colour-blind safe palette, the same hues for the light and dark themes: blue for S, orange for E.
    """
    if dark:
        return dict(
            page="#0d0d0d", surface="#1a1a19", grid="#2c2c2a", axis="#383835",
            ink="#ffffff", ink2="#c3c2b7", muted="#898781",
            blue="#3987e5", orange="#d95926", ref="#c3c2b7",
        )
    return dict(
        page="#f9f9f7", surface="#fcfcfb", grid="#e1e0d9", axis="#c3c2b7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        blue="#2a78d6", orange="#eb6834", ref="#52514e",
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def _style_axis(ax, pal):
    """Light solid grid, no top/right spines."""
    ax.set_facecolor(pal["surface"])
    ax.grid(True, linestyle="-", linewidth=0.8, alpha=1.0, color=pal["grid"], zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(pal["axis"])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=pal["muted"], labelsize=9.5)


def plot(df, mode, xlabel, output_path, show, dark=False):
    """Plots S(x), E(x) and Var(S) in stacked panels."""

    pal = get_palette(dark)
    x = df["x"].values
    n = len(x)

    fig = plt.figure(figsize=(10, 8.5), constrained_layout=True)
    fig.patch.set_facecolor(pal["page"])

    gs = fig.add_gridspec(3, 1, hspace=0.06, height_ratios=[3, 3, 1.6])
    ax_s   = fig.add_subplot(gs[0])
    ax_e   = fig.add_subplot(gs[1], sharex=ax_s)
    ax_var = fig.add_subplot(gs[2], sharex=ax_s)

    Tx = find_transition(df)

    # With many close points the markers make the line look dashed: above 30 points, draw the line only.
    show_markers = n <= 30
    marker_kw = dict(marker="o", markersize=6, markeredgewidth=1.2,
                      markeredgecolor=pal["surface"]) if show_markers else dict(marker="")

    # ── Panel 1: S(x) ────────────────────────────────────────────────────────
    _style_axis(ax_s, pal)
    ax_s.plot(x, df["S"], color=pal["blue"], linewidth=2,
              markerfacecolor=pal["blue"], zorder=3, **marker_kw)
    if n > 1:
        ax_s.fill_between(x, df["S"] - df["errS"], df["S"] + df["errS"],
                           color=pal["blue"], alpha=0.12, linewidth=0, zorder=2)
    ax_s.set_ylabel("Order parameter  S", color=pal["ink2"], fontsize=11)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.annotate(f"{df['S'].iloc[-1]:.3f}",
                  (x[-1], df["S"].iloc[-1]), xytext=(6, 0),
                  textcoords="offset points", va="center", fontsize=9,
                  color=pal["ink2"])
    ax_s.set_title("Monte Carlo simulation results — McLics",
                    fontsize=13.5, color=pal["ink"], pad=14, loc="left",
                    fontweight="semibold")
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # ── Panel 2: E(x) ────────────────────────────────────────────────────────
    _style_axis(ax_e, pal)
    ax_e.plot(x, df["E"], color=pal["orange"], linewidth=2,
              markerfacecolor=pal["orange"], zorder=3, **marker_kw)
    if n > 1:
        ax_e.fill_between(x, df["E"] - df["errE"], df["E"] + df["errE"],
                           color=pal["orange"], alpha=0.14, linewidth=0, zorder=2)
    ax_e.set_ylabel("Mean energy  E", color=pal["ink2"], fontsize=11)
    ax_e.annotate(f"{df['E'].iloc[-1]:.3f}",
                  (x[-1], df["E"].iloc[-1]), xytext=(6, 0),
                  textcoords="offset points", va="center", fontsize=9,
                  color=pal["ink2"])
    plt.setp(ax_e.get_xticklabels(), visible=False)

    # ── Panel 3: Var(S) ──────────────────────────────────────────────────────
    _style_axis(ax_var, pal)
    bar_w = (x[-1] - x[0]) / (n + 1) * 0.85 if n > 1 else 0.01
    ax_var.bar(x, df["varS"], width=bar_w, color=pal["blue"], alpha=0.55,
               linewidth=0, zorder=3)
    ax_var.set_ylabel("Var(S)", fontsize=10, color=pal["ink2"])
    ax_var.set_xlabel(xlabel, fontsize=11.5, color=pal["ink2"])
    ax_var.xaxis.set_minor_locator(AutoMinorLocator())
    ax_var.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))

    # ── Transition line (annotation) ─────────────────────────────────────────
    if Tx is not None:
        for ax in (ax_s, ax_e, ax_var):
            ax.axvline(Tx, color=pal["ref"], linewidth=1.1,
                       linestyle=(0, (3, 3)), zorder=1)
        ax_s.annotate(f"transition ≈ {Tx:.3f}", (Tx, 1.0),
                      xytext=(6, -4), textcoords="offset points",
                      fontsize=9, color=pal["ink2"], rotation=0, va="top")

    for ax in (ax_s, ax_e, ax_var):
        ax.set_facecolor(pal["surface"])

    fig.align_ylabels([ax_s, ax_e, ax_var])

    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plots the results of McLics (po.dat).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", nargs="?", default="po.dat",
        help="po.dat file (default: ./po.dat)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="output image (default: mclics_<mode>_<date>_<time>.png)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="also show the interactive plot"
    )
    parser.add_argument(
        "--txt", action="store_true",
        help="print a table of statistics on the terminal"
    )
    parser.add_argument(
        "--dark", action="store_true",
        help="dark theme (for presentations)"
    )
    parser.add_argument(
        "--mode", choices=sorted(MODE_INFO.keys()), default=None,
        help=("evolution mode of the run (thermal | electric | step | quench), which sets the x-axis label. "
              "Default: detected from the po.dat header.")
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df, mode, xlabel, xcol = read_po_dat(args.input, forced_mode=args.mode)

    # descriptive output name unless one is given
    if args.output:
        output = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"mclics_{mode}_{timestamp}.png"

    print(f"\n  McLics Analyzer")
    print(f"  {'─'*40}")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {output}")
    print(f"  Mode    : {mode}")
    print(f"  Points  : {len(df)}")
    print(f"  {xcol} from {df['x'].min():.4f} to {df['x'].max():.4f}\n")

    if args.txt:
        print_stats(df, xlabel)

    plot(df, mode, xlabel, output, args.show, dark=args.dark)

    Tx = find_transition(df)
    if Tx is not None:
        print(f"  Estimated phase transition at {xcol} ≈ {Tx:.4f}")

    print()


if __name__ == "__main__":
    main()