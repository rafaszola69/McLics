#!/usr/bin/env python3
"""
make_figures.py — chiral nematic in a homeotropic cylinder (custom geometry), after Wand & Bates (PRE 100, 052702).

Reads runs/ (from run.sh) and writes to figs/:
  fig_cylinder.pdf/.png  (a) orientation of the disclination pair along the cylinder for the idealized double helices;
                         (b) energy per site vs imposed twist xi for each p0, with parabolic fits; (c) inverse repeat
                         distance of the double helix vs inverse bulk pitch, with the fit lambda = kappa * p
  figs/numbers.txt       numbers used in the caption;   cylinder_figure.tex   figure environment with English caption

Notes
  - xi is the number of full turns of the disclination pair over the length L; repeat distance lambda = L / xi*,
    with xi* the minimum of a parabola fitted to E(xi) around the lowest-energy point.
  - The twist actually present at the end of each run is measured from the snapshot (low-S sites = disclination
    cores; orientation of the pair in each layer) and reported; runs whose twist changed are flagged.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
DOUBLE = 190 * MM
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BATES = 1.45
SIGN = -1   # run.sh imposes the pair rotation 2*pi*SIGN*xi*z/L (handedness of the bulk helix for p0 > 0)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 3, "legend.frameon": False,
    "savefig.dpi": 300, "figure.facecolor": "white",
})


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)


def runs_in(path):
    out = []
    for d in sorted(glob.glob(os.path.join(path, "seed*"))):
        f = os.path.join(d, "po.dat")
        if os.path.exists(os.path.join(d, "DONE")) and os.path.exists(f):
            df = pd.read_csv(f, sep=r"\s+")
            df.attrs["dir"] = d
            out.append(df)
    return out


def read_input(d):
    prm = {}
    for line in open(os.path.join(d, "input_parameters.txt")):
        t = line.split("#")[0].split()
        if len(t) >= 2:
            prm[t[0].lower()] = t[1:]
    return prm


def last_snapshot(d):
    files = {}
    for f in glob.glob(os.path.join(d, "director_field_*.csv*")):
        m = re.search(r"director_field_(\d+)\.csv(\.gz)?$", f)
        if m:
            files[int(m.group(1))] = f
    return files[min(files)] if files else None      # lowest T = last snapshot on cooling / the only one at fixed T


def load(csv, shape):
    df = pd.read_csv(csv)
    nv = np.zeros(shape + (3,))
    pt = np.zeros(shape, dtype=int)
    S = np.ones(shape)
    x, y, z = (df[c].to_numpy() for c in "xyz")
    nv[x, y, z] = df[["nx", "ny", "nz"]].to_numpy()
    pt[x, y, z] = df["pt"].to_numpy()
    S[x, y, z] = np.nan_to_num(df["S"].to_numpy(), nan=1.0)
    return nv, pt, S


def defects(nv, pt, k):
    """Disclinations in layer k: winding of the in-plane director (mod pi) around each plaquette whose four corners are
    bulk sites. Returns plaquette centres (x, y relative to the box centre) and charges (+-1/2)."""
    N = pt.shape[0]
    c = (N - 1) / 2.0
    ph = 2 * np.arctan2(nv[:, :, k, 1], nv[:, :, k, 0])        # doubled angle: n and -n equivalent
    wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    a, b_, c_, d = ph[:-1, :-1], ph[1:, :-1], ph[1:, 1:], ph[:-1, 1:]
    q = (wrap(b_ - a) + wrap(c_ - b_) + wrap(d - c_) + wrap(a - d)) / (4 * np.pi)
    inside = (pt[:-1, :-1, k] == 1) & (pt[1:, :-1, k] == 1) & (pt[1:, 1:, k] == 1) & (pt[:-1, 1:, k] == 1)
    ii, jj = np.nonzero(inside & (np.abs(q) > 0.25))
    return ii + 0.5 - c, jj + 0.5 - c, np.round(2 * q[ii, jj]) / 2


def structure(nv, pt):
    """Diagnostics of a snapshot: <nz^2> in the bulk, mean number of +1/2 and -1/2 lines per layer, and the total
    rotation of the in-plane director over the periodic length (bulk average, full turns)."""
    L = pt.shape[2]
    b = pt == 1
    npl, nmi = [], []
    for k in range(L):
        q = defects(nv, pt, k)[2]
        npl.append((q > 0).sum())
        nmi.append((q < 0).sum())
    ph = 2 * np.arctan2(nv[..., 1], nv[..., 0])
    d = (np.roll(ph, -1, axis=2) - ph + np.pi) % (2 * np.pi) - np.pi      # includes the periodic closure
    m = b & np.roll(b, -1, axis=2)
    rate = [d[:, :, k][m[:, :, k]].mean() / 2 for k in range(L)]
    return dict(nz2=float((nv[..., 2] ** 2)[b].mean()), nplus=float(np.mean(npl)), nminus=float(np.mean(nmi)),
                dir_turns=float(SIGN * np.sum(rate) / (2 * np.pi)))


def pair_angle(nv, pt, S, R):
    """Orientation (mod pi) of the disclination pair in each z layer and the total rotation over the periodic length
    in full turns. Defects are found by the winding of the director around each plaquette (+1/2 lines only); if a
    layer has none, the low-order sites are used instead."""
    N, _, L = pt.shape
    c = (N - 1) / 2.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    x, y = i - c, j - c
    r = np.hypot(x, y)
    beta = np.full(L, np.nan)
    for k in range(L):
        dx, dy, q = defects(nv, pt, k)
        p = q > 0
        if p.sum() >= 2:
            w, xx, yy = np.ones(p.sum()), dx[p], dy[p]
        else:
            m = (pt[:, :, k] == 1) & (r < R - 1.5)
            w = (np.clip(0.8 - S[:, :, k], 0, None) * m).ravel()
            xx, yy = x.ravel(), y.ravel()
        if w.sum() <= 0:
            continue
        sxx, syy, sxy = (w * xx * xx).sum(), (w * yy * yy).sum(), (w * xx * yy).sum()
        beta[k] = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    ok = ~np.isnan(beta)
    if ok.sum() < 2:
        return np.arange(L)[ok], np.zeros(ok.sum()), np.nan
    b = np.unwrap(2 * beta[ok]) / 2
    closure = ((2 * (beta[ok][0] - b[-1]) + np.pi) % (2 * np.pi) - np.pi) / 2   # last layer -> first (periodic)
    turns = (b[-1] - b[0] + closure) / (2 * np.pi)
    return np.arange(L)[ok], b - b[0], turns


def parabola_min(x, y):
    i = int(np.argmin(y))
    lo, hi = max(0, i - 2), min(len(x), i + 3)
    if hi - lo < 3:
        return None
    a, b, c = np.polyfit(x[lo:hi], y[lo:hi], 2)
    if a <= 0:
        return None
    return float(-b / (2 * a)), (a, b, c), (x[lo], x[hi - 1])


def bulk_pitch(rd, p0):
    vals = []
    for r in runs_in(os.path.join(rd, "bulkp", f"p{p0:g}")):
        prm = read_input(r.attrs["dir"])
        shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
        nv, _, _ = load(last_snapshot(r.attrs["dir"]), shape)
        c2 = np.mean(nv[..., 0] ** 2 - nv[..., 1] ** 2, axis=(0, 1))
        s2 = np.mean(2 * nv[..., 0] * nv[..., 1], axis=(0, 1))
        phi = np.unwrap(np.arctan2(s2, c2)) / 2
        z = np.arange(len(phi))
        m = (z >= 8) & (z < len(phi) - 8)
        vals.append(2 * np.pi / abs(np.polyfit(z[m], phi[m], 1)[0]))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def scan_data(rd, tol, stage="scan"):
    rows = []
    for d in glob.glob(os.path.join(rd, stage, "p*", "R*", "L*")):
        p0 = float(d.split(os.sep)[-3][1:])
        R = int(d.split(os.sep)[-2][1:])
        L = int(d.split(os.sep)[-1][1:])
        for r in runs_in(d):
            prm = read_input(r.attrs["dir"])
            shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
            snap = last_snapshot(r.attrs["dir"])
            if snap is None:
                continue
            nv, pt, S = load(snap, shape)
            turns = pair_angle(nv, pt, S, R)[2]
            if not np.isfinite(turns):                          # no disclination pair found (e.g. planar cholesteric)
                continue
            xi = SIGN * turns                                   # positive = handedness of the bulk helix
            rows.append(dict(p0=p0, R=R, L=L, xi=xi, xi_q=np.round(2 * xi) / 2, E=float(r["E"].iloc[-1]),
                             seed=os.path.basename(r.attrs["dir"]), **structure(nv, pt)))
    return pd.DataFrame(rows)


def fit_lambda(L, xi):
    """lambda from the selected twists: least squares of xi = L / lambda through the origin, and the staircase fit
    (every lambda for which the fewest points fall off the staircase xi = round(2L/lambda)/2; the centre of that
    interval is the estimate and its ends the uncertainty)."""
    L, xi = np.asarray(L, float), np.asarray(xi, float)
    slope = float(np.sum(L * xi) / np.sum(L * L))
    lam_lin = 1 / slope if slope > 0 else np.inf
    if not np.isfinite(lam_lin):
        return lam_lin, np.nan, np.nan, np.nan, 0
    grid = np.linspace(0.5 * lam_lin, 2 * lam_lin, 3001)
    miss = np.array([np.sum(np.abs(np.round(2 * L / lam) / 2 - xi) > 0.25) for lam in grid])
    good = grid[miss == miss.min()]
    return lam_lin, float(np.median(good)), float(good.min()), float(good.max()), int(miss.min())


def energy_min(lam, E, p):
    """Repeat distance of minimum energy: parabola in q = 1/lambda through the k points nearest (in q) to the lowest
    one, for k = 5, 7, 9; returns the k = 7 value and the spread. None if the lowest point is at the largest lambda
    (minimum not bracketed)."""
    q = 1 / np.asarray(lam, float)
    E = np.asarray(E, float)
    i0 = int(np.argmin(E))
    if q[i0] == q.min() or q[i0] == q.max():
        return None
    out = []
    for k in (5, 7, 9):
        idx = np.argsort(np.abs(q - q[i0]))[:min(k, len(q))]
        c = np.polyfit(q[idx], E[idx], 2)
        if c[0] > 0:
            out.append(1 / (-c[1] / (2 * c[0])))
    if not out:
        return None
    return out[len(out) // 2] if len(out) == 3 else float(np.median(out)), min(out), max(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--tol", type=float, default=0.15, help="max |measured - imposed| turns to accept a bates/stretch run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rd = args.runs
    nums = {}
    sc = scan_data(rd, args.tol)
    st = scan_data(rd, args.tol, "stretch")
    rows = []
    for d in glob.glob(os.path.join(rd, "bates", "p*", "R*", "xi*")):
        p0 = float(d.split(os.sep)[-3][1:])
        R = int(d.split(os.sep)[-2][1:])
        xi = float(d.split(os.sep)[-1][2:])
        for r in runs_in(d):
            prm = read_input(r.attrs["dir"])
            shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
            snap = last_snapshot(r.attrs["dir"])
            if snap is None:
                continue
            nv, pt, S = load(snap, shape)
            turns = pair_angle(nv, pt, S, R)[2]
            rows.append(dict(p0=p0, R=R, L=shape[2], xi=xi, E=float(r["E"].iloc[-1]), turns=SIGN * turns,
                             seed=os.path.basename(r.attrs["dir"]), **structure(nv, pt)))
    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["kept"] = (bt["turns"] - bt["xi"]).abs() <= args.tol
        nums["bates_kept"] = f"{int(bt['kept'].sum())} of {len(bt)}"
        nums["bates_changed"] = bt.loc[~bt["kept"], ["p0", "xi", "turns"]].round(2).values.tolist()
    if not st.empty:
        nums["stretch_twist_changed"] = st.loc[(st["xi"] - 1).abs() > args.tol, ["p0", "L", "xi"]].round(2).values.tolist()
    p0s = sorted((set(sc["p0"]) if not sc.empty else set()) | (set(st["p0"]) if not st.empty else set()))
    cols = {p0: PALETTE[i % len(PALETTE)] for i, p0 in enumerate(p0s)}
    pbulk = {}
    for p0 in p0s:
        pb, pbs = bulk_pitch(rd, p0)
        pbulk[p0] = pb if pb is not None else 2 * np.pi / np.arcsin(2 * np.pi / p0)   # lattice pitch at T -> 0

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM))
    # (a) twist selected on cooling vs L
    ax = axes[0]
    lam_rows = []
    for p0 in p0s:
        g = sc[sc["p0"] == p0] if not sc.empty else sc
        if g.empty:
            lam_rows.append(dict(p0=p0, p_bulk=pbulk[p0], lam_st=np.nan, lam_lo=np.nan, lam_hi=np.nan, miss=0))
            continue
        ax.plot(g["L"], g["xi"], ls="none", marker="o", ms=2.5, mfc="none", mew=0.6, color=cols[p0], alpha=0.6)
        if "nminus" in g:
            g = g[g["nminus"] < 0.1]                            # staircase only from double helices (no -1/2 lines)
        if g.empty:
            lam_rows.append(dict(p0=p0, p_bulk=pbulk[p0], lam_st=np.nan, lam_lo=np.nan, lam_hi=np.nan, miss=0))
            continue
        best = g.loc[g.groupby("L")["E"].idxmin()].sort_values("L")
        ax.plot(best["L"], best["xi"], ls="none", marker="o", ms=4, color=cols[p0], label=rf"$p_0={p0:g}$")
        lam_lin, lam_st, lam_lo, lam_hi, miss = fit_lambda(best["L"], best["xi_q"])
        if np.isfinite(lam_st):
            LL = np.linspace(0, g["L"].max() * 1.05, 400)
            ax.plot(LL, np.round(2 * LL / lam_st) / 2, color=cols[p0], lw=0.9)
        lam_rows.append(dict(p0=p0, p_bulk=pbulk[p0], lam_st=lam_st, lam_lo=lam_lo, lam_hi=lam_hi, miss=miss))
        nums[f"scan_p{p0:g}"] = best[["L", "xi", "E"]].round(4).values.tolist()
    if p0s:
        ax.set_xlabel(r"cylinder length $L$")
        ax.set_ylabel(r"twist selected on cooling $\xi$ (turns)")
        ax.legend(fontsize=6, loc="upper left")
    panel(ax, "a")
    # (b) energy vs repeat distance: cooled states (scan), stretched states, idealized helices (bates)
    ax = axes[1]
    for p0 in p0s:
        pb, c = pbulk[p0], cols[p0]
        g = sc[sc["p0"] == p0] if not sc.empty else pd.DataFrame()
        if "nminus" in g:
            g = g[g["nminus"] < 0.1]                            # only the double helix (no -1/2 lines)
        fam = [g.assign(src="scan")] if not g.empty else []
        if not st.empty:
            fam.append(st[(st["p0"] == p0) & ((st["xi"] - 1).abs() <= args.tol)].assign(src="stretch"))
        fam = pd.concat(fam, ignore_index=True)
        fam = fam[fam["xi"].abs() > 0.25]
        fam["lam"] = fam["L"] / fam["xi"]
        fam["lam_r"] = fam["lam"].round(2)
        best = fam.loc[fam.groupby("lam_r")["E"].idxmin()].sort_values("lam")
        E0 = best["E"].min()
        if not bt.empty:
            b = bt[(bt["p0"] == p0) & bt["kept"]]
            bz = b[b["xi"] == 0]["E"]
            E0 = min(E0, bz.min()) if len(bz) else E0
        s_ = best["src"] == "scan"
        ax.plot(best["lam"][s_] / pb, best["E"][s_] - E0, ls="none", marker="o", ms=3, color=c,
                label=rf"$p_0={p0:g}$, cooled")
        ax.plot(best["lam"][~s_] / pb, best["E"][~s_] - E0, ls="none", marker="s", ms=3.5, color=c, mfc="none",
                mew=0.9, label=rf"$p_0={p0:g}$, stretched")
        if not bt.empty:
            b = b[b["xi"] > 0].groupby("xi")["E"].mean()
            ax.plot(b.index.map(lambda x: 48 / x / pb), b.values - E0, ls="none", marker="^", ms=3, color=c,
                    alpha=0.5)
            if len(bz):
                ax.axhline(bz.min() - E0, color=c, ls=":", lw=0.9)
        sb = best[best["src"] == "stretch"]                     # controlled branch: one turn, only L varies
        em = energy_min(sb["lam"], sb["E"], pb) if len(sb) >= 5 else energy_min(best["lam"], best["E"], pb)
        r = next(x for x in lam_rows if x["p0"] == p0)
        if em:
            r.update(lam_E=em[0], lam_E_lo=em[1], lam_E_hi=em[2])
            ax.axvline(em[0] / pb, color=c, ls="--", lw=0.7)
        else:
            r.update(lam_E=np.nan, lam_E_lo=np.nan, lam_E_hi=np.nan, lam_E_bound=float(best.loc[best["E"].idxmin(), "lam"]))
        nums[f"E_vs_lambda_p{p0:g}"] = best[["lam", "E", "src"]].round(4).values.tolist()
    if p0s:
        ax.axvline(BATES, color=INK2, ls="-.", lw=0.8)
        ax.set_xlabel(r"repeat distance $\lambda/p$")
        ax.set_ylabel(r"$E-E_{\rm ref}$ per site")
        ax.legend(fontsize=5.5, loc="upper right")
    panel(ax, "b")
    # (c) lambda vs bulk pitch
    ax = axes[2]
    lr = pd.DataFrame(lam_rows)
    if not lr.empty:
        x = lr["p_bulk"].values
        ax.errorbar(x, lr["lam_st"], yerr=np.vstack([lr["lam_st"] - lr["lam_lo"], lr["lam_hi"] - lr["lam_st"]]),
                    ls="none", marker="o", ms=4, mfc="none", capsize=2, color=PALETTE[0], label="selected on cooling")
        if "lam_E" in lr and lr["lam_E"].notna().any():
            ok = lr["lam_E"].notna()
            ax.errorbar(x[ok], lr["lam_E"][ok], yerr=np.vstack([(lr["lam_E"] - lr["lam_E_lo"])[ok],
                                                                 (lr["lam_E_hi"] - lr["lam_E"])[ok]]),
                        ls="none", marker="s", ms=4, capsize=2, color=PALETTE[1], label="energy minimum")
        xx = np.linspace(0, x.max() * 1.2, 20)
        ax.plot(xx, xx, color=INK2, ls=":", lw=1, label=r"$\lambda=p$")
        ax.plot(xx, BATES * xx, color=INK2, ls="-.", lw=1, label=rf"$\lambda={BATES}\,p$ (Wand and Bates)")
        ax.set_xlabel(r"bulk pitch $p$ at the same $T$")
        ax.set_ylabel(r"repeat distance $\lambda$")
        ax.legend(fontsize=6, loc="upper left")
        nums["lambda"] = lr.round(3).to_dict("records")
    panel(ax, "c")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig_cylinder.{ext}"))
    plt.close(fig)

    tab = []
    if not sc.empty:
        tab.append(sc.assign(stage="scan", xi_imposed=np.nan, xi_meas=sc["xi"]))
    if not st.empty:
        tab.append(st.assign(stage="stretch", xi_imposed=1.0, xi_meas=st["xi"]))
    if not bt.empty:
        tab.append(bt.assign(stage="bates", xi_imposed=bt["xi"], xi_meas=bt["turns"]))
    if tab:
        t = pd.concat(tab, ignore_index=True)
        t["lam_eff"] = t["L"] / t["xi_meas"].where(t["xi_meas"].abs() > 0.05)
        cols_ = ["stage", "p0", "R", "L", "seed", "xi_imposed", "xi_meas", "dir_turns", "lam_eff", "E", "nz2", "nplus",
                 "nminus"]
        t.reindex(columns=cols_).sort_values(["stage", "p0", "L", "xi_imposed", "seed"]).to_csv(
            os.path.join(args.out, "runs_table.csv"), index=False, float_format="%.5g")
    with open(os.path.join(args.out, "numbers.txt"), "w") as f:
        for k_, v in nums.items():
            f.write(f"{k_}: {v}\n")
    for k_, v in nums.items():
        print(f"{k_}: {v}")


if __name__ == "__main__":
    main()
