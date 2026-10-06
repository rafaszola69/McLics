#!/usr/bin/env python3
"""
make_paper_figure.py — figure of the chiral cylinder example for the article (E7 constants, runs/).

  (a) disclination lines of a cooled double helix (render of runs/scan/p25/R10/L72/seed2, the lowest-energy run at L = 72, xi = 3)
  (b) the same branch stretched to L = 110 with one turn (render; both renders at the same scale)
  (c) twist selected on cooling vs L (all runs, lowest energy per L filled)
  (d) energy per site along the double-helix branch vs lambda/p, measured from the untwisted pair (xi = 0)
  (e) the same stretched branch with K1 = K2 = K3 (runs_eqK): energy minimum at a finite repeat distance

  python3 make_paper_figure.py        -> figs/fig_cylinder_paper.pdf/.png, figs/paper_numbers.txt
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402

RD = os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figs")
RENDERS = [("render_helix2/iso_10.png", 72), ("render_streched/iso_10.png", 110)]


def trim(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    nonwhite = np.where(im.min(axis=2) < 245)
    y0, y1, x0, x1 = nonwhite[0].min(), nonwhite[0].max(), nonwhite[1].min(), nonwhite[1].max()
    return im[y0:y1 + 1, x0:x1 + 1]


def main():
    sc = mf.scan_data(RD, 0.15)
    st = mf.scan_data(RD, 0.15, "stretch")
    p0s = sorted(set(sc["p0"]))
    cols = {p0: mf.PALETTE[i] for i, p0 in enumerate(p0s)}
    # pitch of the lattice ground state, Eq. lattice_pitch (the 6x6x96 bulkp runs were not fully relaxed at T = 0.1)
    pb = {p0: 2 * np.pi / np.arcsin(2 * np.pi / p0) for p0 in p0s}
    E0 = {}
    for p0 in p0s:      # untwisted pair: idealized xi = 0 configuration relaxed at the same T (runs/bates/.../xi0)
        E0[p0] = float(np.mean([r["E"].iloc[-1] for r in mf.runs_in(os.path.join(RD, "bates", f"p{p0:g}", "R10", "xi0"))]))
    nums = {"p_bulk": pb, "E_untwisted": E0}

    fig = plt.figure(figsize=(190 * mf.MM, 82 * mf.MM))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.42, 0.42, 1.25, 1.25], height_ratios=[1, 1], wspace=0.55, hspace=0.6)
    # (a), (b) renders at the same scale: heights proportional to L
    ims = [trim(os.path.join(FIGS, f)) for f, _ in RENDERS]
    hmax = max(im.shape[0] for im in ims)
    wmax = max(im.shape[1] for im in ims)
    for k, ((f, L), im) in enumerate(zip(RENDERS, ims)):
        ax = fig.add_subplot(gs[:, k])
        x0 = (wmax - im.shape[1]) / 2
        ax.imshow(im, extent=(x0, x0 + im.shape[1], 0, im.shape[0]))
        ax.set_xlim(0, wmax)                                 # identical boxes: same scale, labels aligned
        ax.set_ylim(-0.04 * hmax, 1.04 * hmax)
        ax.set_axis_off()
        ax.text(0.0, 1.0, f"({'ab'[k]})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
        ax.text(0.5, 0.0, rf"$L={L}$", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    # (c) twist selected on cooling
    ax = fig.add_subplot(gs[0, 2:])
    stair = {}
    for p0 in p0s:
        g = sc[sc["p0"] == p0]
        best = g.loc[g.groupby("L")["E"].idxmin()].sort_values("L")
        ax.plot(g["L"], g["xi"], ls="none", marker="o", ms=2.5, mfc="none", mew=0.6, color=cols[p0], alpha=0.6)
        ax.plot(best["L"], best["xi"], ls="none", marker="o", ms=3.5, color=cols[p0], label=rf"$p_0^*={p0:g}$")
        lam = mf.fit_lambda(best["L"], best["xi_q"])
        LL = np.linspace(0, g["L"].max() * 1.03, 500)
        ax.plot(LL, np.round(2 * LL / lam[1]) / 2, color=cols[p0], lw=0.8)
        stair[p0] = dict(lam=lam[1], lo=lam[2], hi=lam[3], miss=lam[4], n=len(best), lam_over_p=lam[1] / pb[p0])
    nums["staircase"] = stair
    ax.set_xlabel(r"cylinder length $L$")
    ax.set_ylabel(r"twist $\xi$ (turns)")
    ax.legend(fontsize=6, loc="upper left")
    mf.panel(ax, "c")
    # (d) energy along the branch
    ax = fig.add_subplot(gs[1, 2])
    branch = {}
    for p0 in p0s:
        g = pd.concat([sc[sc["p0"] == p0].assign(src="cooled"),
                       st[(st["p0"] == p0) & ((st["xi"] - 1).abs() < 0.15)].assign(src="stretched")], ignore_index=True)
        g = g[g["xi"].abs() > 0.25]
        g["lam"] = (g["L"] / g["xi"]).round(2)
        best = g.loc[g.groupby("lam")["E"].idxmin()].sort_values("lam")
        for src, mk, fill in (("cooled", "o", cols[p0]), ("stretched", "s", "none")):
            b = best[best["src"] == src]
            ax.plot(b["lam"] / pb[p0], b["E"] - E0[p0], ls="none", marker=mk, ms=3, mfc=fill, mew=0.8, color=cols[p0],
                    label=rf"$p_0^*={p0:g}$, {src}")
        branch[p0] = best[["lam", "E", "src"]].round(4).values.tolist()
    nums["branch"] = branch
    ax.axhline(0, color=mf.INK2, ls=":", lw=0.9)
    ax.text(4.5, -0.0012, r"untwisted pair ($\xi=0$)", ha="right", va="top", fontsize=6, color=mf.INK2)
    ax.set_ylim(bottom=-0.006)
    ax.set_xlabel(r"repeat distance $\lambda/p$")
    ax.set_ylabel(r"$E-E_{\xi=0}$ per site")
    ax.legend(fontsize=5, loc="upper right")
    mf.panel(ax, "d")
    # (e) the same stretched branch with equal elastic constants (runs_eqK): a finite repeat distance appears
    ax = fig.add_subplot(gs[1, 3], sharex=ax)
    stq = mf.scan_data(os.path.join(HERE, "runs_eqK"), 0.15, "stretch")
    eq = {}
    for p0 in p0s:
        g = stq[(stq["p0"] == p0) & ((stq["xi"] - 1).abs() < 0.15)].copy()
        if g.empty:
            continue
        g["lam"] = (g["L"] / g["xi"]).round(2)
        best = g.loc[g.groupby("lam")["E"].idxmin()].sort_values("lam")
        x, y = best["lam"].values / pb[p0], best["E"].values - best["E"].min()
        ax.plot(x, y, ls="none", marker="s", ms=3, mfc="none", mew=0.8, color=cols[p0], label=rf"$p_0^*={p0:g}$")
        em = mf.energy_min(best["lam"], best["E"], pb[p0])
        if em:
            ax.axvline(em[0] / pb[p0], color=cols[p0], ls="--", lw=0.8)
            eq[p0] = dict(lam_star=round(em[0], 2), lo=round(em[1], 2), hi=round(em[2], 2),
                          over_p=round(em[0] / pb[p0], 3))
    nums["equalK_min"] = eq
    ax.set_xlabel(r"repeat distance $\lambda/p$")
    ax.set_ylabel(r"$E-E_{\min}$ per site")
    ax.set_title(r"$K_1=K_2=K_3$", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper right")
    mf.panel(ax, "e")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_cylinder_paper.{ext}"), dpi=300, bbox_inches="tight")
    with open(os.path.join(FIGS, "paper_numbers.txt"), "w") as f:
        for k, v in nums.items():
            f.write(f"{k}: {v}\n")
    for k, v in nums.items():
        print(k, v)


if __name__ == "__main__":
    main()
