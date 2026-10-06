#!/usr/bin/env python3
"""
energy_terms.py — Frank decomposition of the double-helix states along the stretched branch.

For each snapshot the gradient of the director is taken by central differences on bulk sites whose four in-plane and
two axial neighbours are bulk sites too (z periodic), aligning each neighbour with the central director first (n and -n
are the same state). From it, per site:
    splay s = div n,   twist t = n . curl n,   bend b = |n x curl n|^2,
and the Frank energy density f = K11 s^2/2 + K22 (t - t0)^2/2 + K33 b/2 with t0 = +-2*pi/p (sign taken from the data:
+ for p0 > 0, checked on the bulk cholesteric of runs/bulkp: t = 0.252 vs 2*pi/p = 0.254). Sites within 1.5 lattice spacings of a disclination are reported apart
(the continuum density is not meaningful at the core). Also <nz^2>.

  python3 energy_terms.py                 -> figs_terms/fig_terms.png, figs_terms/terms.csv
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402

SETS = {"runs": ("E7", (1.48, 1.0, 2.28)), "runs_eqK": ("equal K", (1.0, 1.0, 1.0)),
        "runs_wbK": ("Wand-Bates K", (1.054, 1.0, 1.015))}


def align(n0, n1):
    s = np.sign(np.sum(n0 * n1, axis=-1, keepdims=True))
    s[s == 0] = 1
    return n1 * s


def gradients(nv, pt):
    """d_a n_b on the valid sites (central differences, neighbours aligned to the centre)."""
    b = pt == 1
    g = np.zeros(nv.shape[:3] + (3, 3))
    ok = b.copy()
    for a in range(3):
        if a == 2:
            fwd, bwd = np.roll(nv, -1, axis=2), np.roll(nv, 1, axis=2)
            bf, bb = np.roll(b, -1, axis=2), np.roll(b, 1, axis=2)
        else:
            fwd, bwd = np.zeros_like(nv), np.zeros_like(nv)
            bf, bb = np.zeros_like(b), np.zeros_like(b)
            sl_c, sl_f, sl_b = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            sl_c[a], sl_f[a] = slice(0, -1), slice(1, None)
            fwd[tuple(sl_c)], bf[tuple(sl_c)] = nv[tuple(sl_f)], b[tuple(sl_f)]
            sl_c[a], sl_b[a] = slice(1, None), slice(0, -1)
            bwd[tuple(sl_c)], bb[tuple(sl_c)] = nv[tuple(sl_b)], b[tuple(sl_b)]
        g[..., a, :] = (align(nv, fwd) - align(nv, bwd)) / 2
        ok &= bf & bb
    return g, ok


def frank(nv, pt):
    g, ok = gradients(nv, pt)
    div = g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
    curl = np.stack([g[..., 1, 2] - g[..., 2, 1], g[..., 2, 0] - g[..., 0, 2], g[..., 0, 1] - g[..., 1, 0]], axis=-1)
    tw = np.sum(nv * curl, axis=-1)
    bend = np.sum(np.cross(nv, curl) ** 2, axis=-1)
    return div, tw, bend, ok


def core_mask(nv, pt, rad=1.5):
    N, _, L = pt.shape
    c = (N - 1) / 2.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    m = np.zeros(pt.shape, bool)
    for k in range(L):
        dx, dy, q = mf.defects(nv, pt, k)
        for x0, y0 in zip(dx, dy):
            m[:, :, k] |= np.hypot(i - c - x0, j - c - y0) <= rad
    return m


def analyse(d):
    prm = mf.read_input(d)
    shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
    nv, pt, S = mf.load(mf.last_snapshot(d), shape)
    div, tw, bend, ok = frank(nv, pt)
    core = core_mask(nv, pt)
    far = ok & ~core
    return dict(L=shape[2], n_far=far.sum() / shape[2], n_bulk=(pt == 1).sum() / shape[2], s2=(div[far] ** 2).mean(), t=tw[far].mean(), t2=tw[far].var() + tw[far].mean() ** 2,
                b2=bend[far].mean(), nz2=(nv[..., 2] ** 2)[pt == 1].mean(), frac_core=(ok & core).sum() / ok.sum(),
                s2_core=(div[ok & core] ** 2).mean() if (ok & core).any() else np.nan,
                b2_core=bend[ok & core].mean() if (ok & core).any() else np.nan,
                t2_core=(tw[ok & core] ** 2).mean() if (ok & core).any() else np.nan)


def main():
    out = os.path.join(HERE, "figs_terms")
    os.makedirs(out, exist_ok=True)
    rows = []
    for rd, (name, K) in SETS.items():
        for d in sorted(glob.glob(os.path.join(HERE, rd, "stretch", "p*", "R*", "L*", "seed*"))):
            if not os.path.exists(os.path.join(d, "DONE")):
                continue
            p0 = float(d.split(os.sep)[-4][1:])
            r = analyse(d)
            E = float(pd.read_csv(os.path.join(d, "po.dat"), sep=r"\s+")["E"].iloc[-1])
            rows.append(dict(set=name, p0=p0, seed=os.path.basename(d), E=E, **r))
            print(name, p0, r["L"], os.path.basename(d), f"E={E:.4f}", f"t={r['t']:+.4f}", f"nz2={r['nz2']:.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "terms.csv"), index=False, float_format="%.6g")
    present = [v for v in SETS.values() if v[0] in set(df["set"])]
    fig, axes = plt.subplots(2, len(present), figsize=(190 * mf.MM * len(present) / 3, 110 * mf.MM), sharex=True,
                             squeeze=False)
    for col, (name, K) in enumerate(present):
        for p0, c in zip(sorted(set(df["p0"])), mf.PALETTE):
            g = df[(df["set"] == name) & (df["p0"] == p0)].groupby("L").mean(numeric_only=True).reset_index()
            if g.empty:
                continue
            pb = 2 * np.pi / np.arcsin(2 * np.pi / p0)
            t0 = 2 * np.pi / pb                                 # natural twist (+ for p0 > 0, checked on runs/bulkp)
            fs = K[0] * g["s2"] / 2
            ft = K[1] * (g["t2"] - 2 * t0 * g["t"] + t0 ** 2) / 2
            fb = K[2] * g["b2"] / 2
            x = g["L"] / pb
            ax = axes[0, col]
            ffar = (fs + ft + fb) * g["n_far"] / g["n_bulk"]          # per bulk site, like E
            ax.plot(x, g["E"] - g["E"].iloc[0], "o-", ms=2.5, color=c, label=rf"$E$ (MC), $p_0={p0:g}$")
            ax.plot(x, ffar - ffar.iloc[0], "--", color=c, lw=0.9, label="Frank, far from cores")
            ax.plot(x, (g["E"] - ffar) - (g["E"] - ffar).iloc[0], ":", color=c, lw=0.9, label="rest (cores, surface)")
            ax.set_title(name, fontsize=8)
            ax = axes[1, col]
            ax.plot(x, fs - fs.iloc[0], ":", color=c, label="splay" if p0 == 25 else None)
            ax.plot(x, ft - ft.iloc[0], "-", color=c, label="twist" if p0 == 25 else None)
            ax.plot(x, fb - fb.iloc[0], "--", color=c, label="bend" if p0 == 25 else None)
        axes[0, col].legend(fontsize=5)
        axes[1, col].legend(fontsize=5)
        axes[1, col].set_xlabel(r"$\lambda/p$")
    axes[0, 0].set_ylabel(r"$\Delta E$ per site (from shortest $\lambda$)")
    axes[1, 0].set_ylabel(r"$\Delta f$ per site (from shortest $\lambda$)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_terms.png"), dpi=200)
    print("->", out)


if __name__ == "__main__":
    main()
