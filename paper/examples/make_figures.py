#!/usr/bin/env python3
"""
make_figures.py — figures for three further test problems of the McLics paper (see run.sh).

  figs/fig_chol.pdf      (a) equilibrium pitch vs p0 (lattice prediction 2*pi/arcsin(2*pi/p0)); (b) pitch/prediction vs T;
                         (c) twist across a planar cell (Grandjean steps)
  figs/fig_homeo.pdf     homeotropic droplets with weak anchoring: S_r vs W and vs W*R (collapse); two cross-sections
  figs/fig_chiral.pdf    chiral droplets with planar anchoring: cross-sections for +/- handedness; twist vs R/p0
  figs/fig_llbulk.pdf    (a) S vs T; (b) heat capacity vs T (cooling and heating); (c) T_NI(L) vs 1/L^3
  figures.tex            LaTeX figure environments with English captions (numbers filled in from the data)
  figs/numbers.txt       numbers quoted in the captions

Usage: python3 make_figures.py
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
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
DOUBLE = 190 * MM
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = LinearSegmentedColormap.from_list("ramp", ["#86b6ef", "#184f95"])
T_NI_LIT = 1.1232
TWIST_TMAX = 0.3   # p0 = 8 at T = 0.5 shows a slip of the helix; the panel is restricted to T <= 0.3

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


def save(fig, out):
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ─── reading ────────────────────────────────────────────────────────────────

def runs_in(path):
    out = []
    for d in sorted(glob.glob(os.path.join(path, "seed*"))):
        f = os.path.join(d, "po.dat")
        if os.path.exists(os.path.join(d, "DONE")) and os.path.exists(f):
            df = pd.read_csv(f, sep=r"\s+")
            df["T"] = df["T"].round(4)
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


def snapshot(run, T=None):
    """Snapshot at temperature T (default: last one). One snapshot per temperature, in po.dat order."""
    files = {}
    for f in glob.glob(os.path.join(run.attrs["dir"], "director_field_*.csv*")):
        m = re.search(r"director_field_(\d+)\.csv(\.gz)?$", f)
        if m:
            files[int(m.group(1))] = f
    Ts = list(run["T"])
    if not files or len(files) != len(Ts):
        return None
    idx = sorted(files, reverse=len(Ts) < 2 or Ts[0] > Ts[-1])
    i = len(Ts) - 1 if T is None else (Ts.index(T) if T in Ts else None)
    return None if i is None else files[idx[i]]


def load_field(csv, shape):
    df = pd.read_csv(csv)
    nv = np.zeros(shape + (3,))
    pt = np.zeros(shape, dtype=int)
    S = np.full(shape, np.nan)
    x, y, z = (df[c].to_numpy() for c in "xyz")
    nv[x, y, z] = df[["nx", "ny", "nz"]].to_numpy()
    pt[x, y, z] = df["pt"].to_numpy()
    S[x, y, z] = df["S"].to_numpy()
    return nv, pt, S


def layer_angles(nv):
    """In-plane director angle of each z layer (nematic average, period pi), unwrapped along z."""
    c2 = np.mean(nv[..., 0] ** 2 - nv[..., 1] ** 2, axis=(0, 1))
    s2 = np.mean(2 * nv[..., 0] * nv[..., 1], axis=(0, 1))
    phi = 0.5 * np.arctan2(s2, c2)
    return np.unwrap(2 * phi) / 2


def p_lattice(p0):
    return 2 * np.pi / np.arcsin(2 * np.pi / np.asarray(p0, float))


# ─── example 1: cholesteric ────────────────────────────────────────────────

def measure_pitch(r):
    prm = read_input(r.attrs["dir"])
    shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
    csv = snapshot(r)
    if csv is None:
        return None
    nv, _, _ = load_field(csv, shape)
    phi = layer_angles(nv)
    z = np.arange(len(phi))
    m = (z >= 8) & (z < len(phi) - 8)
    return 2 * np.pi / abs(np.polyfit(z[m], phi[m], 1)[0])


def fig_chol(rd, out, nums):
    rows = []
    for d in glob.glob(os.path.join(rd, "twist", "p*", "start*")):
        p0 = float(os.path.basename(os.path.dirname(d))[1:])
        start = float(os.path.basename(d)[5:])
        for r in runs_in(d):
            p = measure_pitch(r)
            if p is not None:
                rows.append((p0, start, p))
    rowsT = []
    for d in glob.glob(os.path.join(rd, "twistT", "p*", "T*")):
        p0 = float(os.path.basename(os.path.dirname(d))[1:])
        T = float(os.path.basename(d)[1:])
        for r in runs_in(d):
            p = measure_pitch(r)
            if p is not None:
                rowsT.append((p0, T, p))
    cano = []
    for d in glob.glob(os.path.join(rd, "cano", "Nz*")):
        nz = int(os.path.basename(d)[2:])
        for r in runs_in(d):
            prm = read_input(r.attrs["dir"])
            shape = tuple(int(prm[k][0]) for k in ("nx", "ny", "nz"))
            csv = snapshot(r)
            if csv is None:
                continue
            nv, _, _ = load_field(csv, shape)
            phi = layer_angles(nv)
            cano.append((nz - 1, abs(phi[-1] - phi[0]) / np.pi, float(prm["p0"][0])))
    if not rows and not cano and not rowsT:
        return
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 60 * MM))
    ax = axes[0]
    if rows:
        df = pd.DataFrame(rows, columns=["p0", "start", "p"])
        g = df.groupby(["p0", "start"])["p"].agg(["mean", "std"]).reset_index()
        pp = np.linspace(df["p0"].min() * 0.9, df["p0"].max() * 1.05, 200)
        ax.plot(pp, pp, color=INK2, ls=":", lw=1, label=r"$p=p_0$")
        ax.plot(pp, p_lattice(pp), color=INK2, ls="--", lw=1, label=r"$2\pi/\arcsin(2\pi/p_0)$")
        for st, col, mk in ((0.8, AQUA, "^"), (1.0, BLUE, "o"), (1.25, ORANGE, "s")):
            sel = g[g["start"] == st]
            ax.errorbar(sel["p0"], sel["mean"], yerr=sel["std"].fillna(0), color=col, marker=mk, ms=3.5, lw=0,
                        elinewidth=1, capsize=2, label=rf"$p_i={st:g}\,p_0$")
        nums["twist_maxdev_lattice"] = float((df["p"] / p_lattice(df["p0"]) - 1).abs().max())
        nums["twist_maxdev_nominal"] = float((df["p"] / df["p0"] - 1).abs().max())
        nums["twist_spread_starts"] = float(g.groupby("p0")["mean"].agg(lambda v: v.max() / v.min() - 1).max())
        # per p0: spread between the three initial conditions, and the pitch of each start relative to the lattice
        # prediction (p / p_lattice - 1)
        nums["twist_spread_by_p0"] = {float(k): round(float(v), 4) for k, v in
                                      g.groupby("p0")["mean"].agg(lambda v: v.max() / v.min() - 1).items()}
        nums["twist_dev_lattice_by_p0_start"] = {
            f"{p0:g}/{st:g}": round(float(m / p_lattice(p0) - 1), 4) for p0, st, m in g[["p0", "start", "mean"]].values}
        ax.set_xlabel(r"nominal pitch $p_0$")
        ax.set_ylabel(r"measured pitch $p$")
        ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axes[1]
    if rowsT:
        df = pd.DataFrame(rowsT, columns=["p0", "T", "p"])
        df["ratio"] = df["p"] / p_lattice(df["p0"])
        df = df[df["T"] <= TWIST_TMAX]
        ax.axhline(1, color=INK2, ls="--", lw=1)
        for p0, col, mk in ((8, BLUE, "o"), (10, ORANGE, "s")):
            g = df[df["p0"] == p0].groupby("T")["ratio"].agg(["mean", "std"])
            if g.empty:
                continue
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=col, marker=mk, ms=3.5, capsize=2,
                        label=rf"$p_0={p0}$")
            nums[f"twistT_p{p0}"] = {float(T): round(float(v), 3) for T, v in g["mean"].items()}
        ax.set_xlabel(r"$T$")
        ax.set_ylabel(r"$p\,/\,[2\pi/\arcsin(2\pi/p_0)]$")
        ax.legend(loc="lower left")
    panel(ax, "b")
    ax = axes[2]
    if cano:
        df = pd.DataFrame(cano, columns=["d", "m", "p0"])
        p_eff = float(p_lattice(df["p0"].iloc[0]))
        x = df["d"] / p_eff
        xx = np.linspace(0, x.max() * 1.05, 400)
        ax.plot(xx, np.round(2 * xx), color=INK2, lw=1, label="minimum-energy staircase")
        ax.plot(xx, 2 * xx, color=INK2, ls=":", lw=1, label=r"free twist $2d/p$")
        ax.plot(x, df["m"], ls="none", marker="o", ms=4, color=BLUE, mfc="none", mew=1, label="simulation")
        mi = np.rint(df["m"]) - np.round(2 * x)
        nums["cano_p_eff"] = p_eff
        nums["cano_match"] = int((mi == 0).sum())
        nums["cano_extra"] = int((mi == 1).sum())
        nums["cano_n"] = len(df)
        # runs that are neither on the staircase nor one half turn above it: (d, d/p, measured half turns, staircase)
        oth = (mi != 0) & (mi != 1)
        nums["cano_other"] = [(int(d), round(float(xx), 3), round(float(m), 2), int(np.round(2 * xx)))
                              for d, xx, m in zip(df["d"][oth], x[oth], df["m"][oth])]
        ax.set_xlabel(r"$d/p$")
        ax.set_ylabel(r"total twist $\Delta\varphi/\pi$")
        ax.legend(loc="upper left")
    panel(ax, "c")
    fig.tight_layout()
    save(fig, out)


# ─── droplets: helpers ──────────────────────────────────────────────────────

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def droplet_axis(nv, pt):
    v = nv[pt > 0]
    w, e = np.linalg.eigh(v.T @ v / len(v))
    return e[:, -1]


def droplet_twist(nv, pt, d):
    """chi = <(n.phi)(n.d)> over the droplet (excluding the axis): sign = handedness of the twist about d."""
    n = nv.shape[0]
    c = n // 2
    g = np.indices(nv.shape[:3]).transpose(1, 2, 3, 0) - c
    inside = pt > 0
    v = nv[inside]
    r = g[inside].astype(float)
    rperp = r - np.outer(r @ d, d)
    rho = np.linalg.norm(rperp, axis=1)
    ok = rho > 1.5
    phi_hat = np.cross(d, rperp[ok] / rho[ok, None])
    a = np.sum(v[ok] * phi_hat, axis=1)
    b = v[ok] @ d
    chi = float(np.mean(a * b))
    # tilt of the director from d towards phi, averaged as a nematic (headless) quantity; monotonic up to 90 deg
    theta = 0.5 * np.degrees(np.arctan2(2 * chi, float(np.mean(b * b - a * a))))
    return chi, theta


def plane_slice(ax, nv, pt, u, v, colour, clim, cmap):
    """Headless director segments on the central plane spanned by u, v (projected), coloured by colour(n, r)."""
    n = nv.shape[0]
    c = n // 2
    step = max(1, round(n / 18))
    P, A, C = [], [], []
    for a in range(-c, c + 1, step):
        for b in range(-c, c + 1, step):
            q = np.rint(c + a * u + b * v).astype(int)
            if np.any(q < 0) or np.any(q >= n) or pt[tuple(q)] == 0:
                continue
            nn = nv[tuple(q)]
            P.append((a, b))
            A.append((nn @ u, nn @ v))
            C.append(colour(nn, a * u + b * v))
    P, A = np.array(P), np.array(A)
    sgn = np.where(A[:, 0] >= 0, 1, -1)
    qv = ax.quiver(P[:, 0], P[:, 1], A[:, 0] * sgn, A[:, 1] * sgn, C, cmap=cmap, clim=clim, pivot="mid",
                   angles="xy", scale_units="xy", scale=1 / (0.9 * step), headwidth=1, headlength=0,
                   headaxislength=0, width=0.012)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return qv


def ndotr2(nn, r):
    rr = np.linalg.norm(r)
    return 1.0 if rr == 0 else float((nn @ r / rr) ** 2)


# ─── example 2: homeotropic droplets, weak anchoring ────────────────────────

def fig_homeo(rd, out, nums):
    rows, snaps = [], {}
    for d in glob.glob(os.path.join(rd, "homeo", "N*", "W*")):
        N = int(os.path.basename(os.path.dirname(d))[1:])
        W = float(os.path.basename(d)[1:])
        for r in runs_in(d):
            if "Sr_sph" not in r:
                continue
            rows.append((N, W, float(r["Sr_sph"].iloc[-1]), float(r["T"].iloc[-1])))
            snaps.setdefault((N, W), r)
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["N", "W", "Sr", "T"])
    df["R"] = df["N"] / 2 - 1
    g = df.groupby(["N", "W"]).agg(Sr=("Sr", "mean"), sd=("Sr", "std"), R=("R", "first")).reset_index()
    Ns = sorted(g["N"].unique())
    cols = {N: PALETTE[i] for i, N in enumerate(Ns)}
    fig = plt.figure(figsize=(DOUBLE, 60 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 0.75, 0.75], wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cross = {}
    for N in Ns:
        h = g[g["N"] == N].sort_values("W")
        R = h["R"].iloc[0]
        runs_N = df[df["N"] == N]
        for ax, sc in ((ax1, 1.0), (ax2, R)):
            ax.plot(h["W"] * sc, h["Sr"], color=cols[N], lw=1.2, label=rf"$R\approx{R:g}$")
            ax.plot(runs_N["W"] * sc, runs_N["Sr"], ls="none", marker="o", ms=3, color=cols[N], alpha=0.8)
        y = h["Sr"].values
        if y.min() < 0.5 < y.max():
            i = int(np.argmax(y >= 0.5))
            if i > 0:
                lw = np.interp(0.5, [y[i - 1], y[i]], np.log([h["W"].values[i - 1], h["W"].values[i]]))
                cross[N] = (float(np.exp(lw)), float(np.exp(lw) * R))
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_ylabel(r"radial order $S_r$")
        ax.axhline(0.5, color=INK2, ls=":", lw=0.8)
    ax1.set_xlabel(r"anchoring strength $W_s$")
    ax2.set_xlabel(r"$W_s R$ (lattice units)")
    ax1.legend(loc="lower right")
    panel(ax1, "a")
    panel(ax2, "b")
    nums["homeo_cross"] = cross
    if cross:
        wr = np.array([v[1] for v in cross.values()])
        wv = np.array([v[0] for v in cross.values()])
        nums["homeo_WR_mean"] = float(wr.mean())
        nums["homeo_WR_spread"] = float(wr.max() / wr.min() - 1)
        nums["homeo_W_spread"] = float(wv.max() / wv.min() - 1)
    Nbig = Ns[-1]
    Ws = sorted(g[g["N"] == Nbig]["W"])
    cmap = LinearSegmentedColormap.from_list("ndr", ["#2a78d6", "#c9c8c3", "#e34948"])
    for j, (W, letter) in enumerate(((Ws[0], "c"), (Ws[-1], "d"))):
        ax = fig.add_subplot(gs[0, 2 + j])
        r = snaps[(Nbig, W)]
        nv, pt, _ = load_field(snapshot(r), (Nbig,) * 3)
        d = droplet_axis(nv, pt)
        u = d
        w_ = np.cross(u, [1, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1, 0])
        w_ /= np.linalg.norm(w_)
        q = plane_slice(ax, nv, pt, u, np.cross(w_, u), ndotr2, (0, 1), cmap)
        ax.set_title(rf"$W_s={W:g}$", fontsize=8, color=INK)
        panel(ax, letter)
    cb = fig.colorbar(q, ax=fig.axes[-2:], orientation="horizontal", fraction=0.06, pad=0.04)
    cb.set_label(r"$(\hat{\mathbf{n}}\cdot\hat{\mathbf{r}})^2$", fontsize=8)
    nums["homeo_T"] = float(df["T"].iloc[0])
    nums["homeo_slice_N"] = int(Nbig)
    save(fig, out)


# ─── example 3: chiral bipolar droplets ─────────────────────────────────────

def fig_chiral(rd, out, nums):
    rows, snaps = [], {}
    for d in glob.glob(os.path.join(rd, "chiral", "N*", "x*")):
        N = int(os.path.basename(os.path.dirname(d))[1:])
        x = float(os.path.basename(d)[1:])
        for r in runs_in(d):
            nv, pt, _ = load_field(snapshot(r), (N,) * 3)
            dvec = droplet_axis(nv, pt)
            chi_, th_ = droplet_twist(nv, pt, dvec)
            rows.append((N, x, chi_, th_, float(r["Sr_sph"].iloc[-1]) if "Sr_sph" in r else np.nan))
            snaps.setdefault(x, (r, N))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["N", "x", "chi", "theta", "Sr"])
    g = df.groupby("x").agg(chi=("chi", "mean"), theta=("theta", "mean"), sd=("theta", "std"),
                            Sr=("Sr", "mean")).reset_index().sort_values("x")
    fig = plt.figure(figsize=(DOUBLE, 60 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.8, 1.4], wspace=0.3)
    cmap = LinearSegmentedColormap.from_list("chi", ["#2a78d6", "#c9c8c3", "#e34948"])
    xs = [x for x in (g["x"].max(), g["x"].min()) if x != 0]
    for j, (x, letter) in enumerate(zip(xs[:2], "ab")):
        ax = fig.add_subplot(gs[0, j])
        r, N = snaps[x]
        nv, pt, _ = load_field(snapshot(r), (N,) * 3)
        d = droplet_axis(nv, pt)
        u = np.cross(d, [1, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)

        def tw(nn, rr, d=d):
            rp = rr - (rr @ d) * d
            if np.linalg.norm(rp) < 1e-9:
                return 0.0
            ph = np.cross(d, rp / np.linalg.norm(rp))
            return float((nn @ ph) * (nn @ d))
        q = plane_slice(ax, nv, pt, u, v, tw, (-0.5, 0.5), cmap)
        ax.set_title(rf"$R/p_0={x:g}$", fontsize=8, color=INK)
        panel(ax, letter)
    cb = fig.colorbar(q, ax=fig.axes[:2], orientation="horizontal", fraction=0.06, pad=0.04)
    cb.set_label(r"$(\hat{\mathbf{n}}\cdot\hat{\varphi})(\hat{\mathbf{n}}\cdot\mathbf{d})$", fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.errorbar(g["x"], g["theta"], yerr=g["sd"].fillna(0), color=BLUE, marker="o", ms=4, lw=0, elinewidth=1,
                capsize=2)
    small = g[(g["x"].abs() <= 0.031) & (g["x"] != 0)]
    if len(small) >= 2:
        k = float(np.sum(small["x"] * small["theta"]) / np.sum(small["x"] ** 2))
        xx = np.linspace(-0.08, 0.08, 20)
        ax.plot(xx, k * xx, color=INK2, ls="--", lw=1, label=rf"initial slope ${k:.0f}^\circ$ per unit $R/p_0$")
        ax.legend(loc="upper left")
        nums["chiral_slope_deg"] = k
    ax.set_xlabel(r"$R/p_0$")
    ax.set_ylabel(r"twist angle $\theta_t$ (degrees)")
    panel(ax, "c")
    pos = g[g["x"] > 0]
    if not pos.empty:
        nums["chiral_theta_max"] = float(pos["theta"].max())
    pairs = [(x, c) for x, c in zip(g["x"], g["theta"]) if x > 0 and -x in set(g["x"])]
    if pairs:
        gm = dict(zip(g["x"], g["theta"]))
        nums["chiral_antisym_deg"] = float(max(abs(c + gm[-x]) for x, c in pairs))
    nums["chiral_theta"] = {float(x): round(float(c), 1) for x, c in zip(g["x"], g["theta"])}
    nums["chiral_chi"] = {float(x): round(float(c), 4) for x, c in zip(g["x"], g["chi"])}
    nums["chiral_sd"] = {float(x): round(float(c), 4) for x, c in zip(g["x"], g["sd"].fillna(0))}
    nums["chiral_Sr"] = {float(x): round(float(c), 3) for x, c in zip(g["x"], g["Sr"])}
    nums["chiral_N"] = int(df["N"].iloc[0])
    save(fig, out)


# ─── example 3: Lebwohl-Lasher bulk ─────────────────────────────────────────

def cv_peak(df, L):
    cv = L ** 3 * df["varE"] / (4 * df.index.values ** 2)   # E in po.dat counts every bond twice
    i = int(np.argmax(cv.values))
    if 0 < i < len(cv) - 1:
        y0, y1, y2 = cv.values[i - 1:i + 2]
        h = df.index.values[1] - df.index.values[0]
        return cv, float(df.index.values[i] + 0.5 * h * (y0 - y2) / (y0 - 2 * y1 + y2))
    return cv, float(df.index.values[i])


def fig_llbulk(rd, out, nums):
    proto = {}
    for name in ("llbulk", "llheat"):
        sizes = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(rd, name, "L*")))
        data = {L: runs_in(os.path.join(rd, name, f"L{L}")) for L in sizes}
        data = {L: r for L, r in data.items() if r}
        if data:
            proto[name] = data
    if "llbulk" not in proto:
        return
    allL = sorted(set().union(*[set(d) for d in proto.values()]))
    cols = {L: PALETTE[i % len(PALETTE)] for i, L in enumerate(allL)}
    style = {"llbulk": ("-", "o", "cooling"), "llheat": ("--", "s", "heating")}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 58 * MM))
    peaks, perr = {}, {}
    for name, data in proto.items():
        ls, mk, lab = style[name]
        peaks[name] = {}
        perr.setdefault(name, {})
        for L, runs in sorted(data.items()):
            df = pd.concat([r[["T", "S", "varE"]] for r in runs]).groupby("T").mean().sort_index()
            cv, pk = cv_peak(df, L)
            peaks[name][L] = pk
            # uncertainty of the peak position from the scatter of the individual runs
            pr = [cv_peak(r[["T", "varE"]].groupby("T").mean().sort_index(), L)[1] for r in runs]
            perr[name][L] = float(np.std(pr, ddof=1) / np.sqrt(len(pr))) if len(pr) > 1 else np.nan
            axes[0].plot(df.index, df["S"], color=cols[L], ls=ls, lw=1, label=rf"$L={L}$" if name == "llbulk" else None)
            axes[1].plot(df.index, cv, color=cols[L], ls=ls, lw=1)
    for ax in axes[:2]:
        ax.axvline(T_NI_LIT, color=INK2, ls=":", lw=1)
        ax.set_xlabel(r"$T$")
    axes[0].set_ylabel(r"$S$")
    axes[1].set_ylabel(r"$C_V$ per site")
    h1 = axes[0].legend(loc="upper right")
    if "llheat" in proto:
        from matplotlib.lines import Line2D
        axes[1].legend(handles=[Line2D([], [], color=INK2, ls="-", label="cooling"),
                                Line2D([], [], color=INK2, ls="--", label="heating")], loc="upper left")
    ax = axes[2]
    fits = {}
    if len(perr) == 2:         # pool the run-to-run scatter of the two protocols (3 runs alone give unreliable errors)
        a_, b_ = perr.values()
        for L in set(a_) & set(b_):
            a_[L] = b_[L] = float(np.sqrt((a_[L] ** 2 + b_[L] ** 2) / 2))
    for name, pk in peaks.items():
        ls, mk, lab = style[name]
        Ls = np.array(sorted(pk))
        x, y = 1.0 / Ls ** 3, np.array([pk[L] for L in Ls])
        for L_, x_, y_ in zip(Ls, x, y):
            ax.plot(x_, y_, ls="none", marker=mk, ms=4.5, color=cols[L_], mfc=cols[L_] if name == "llbulk" else "none")
        big = Ls >= 16
        if big.sum() >= 2:
            b_, a_ = np.polyfit(x[big], y[big], 1)
            fits[name] = float(a_)
            e = np.array([perr[name][L] for L in Ls])
            ok = big & np.isfinite(e) & (e > 0)
            if ok.sum() >= 3:      # weighted fit: intercept and its statistical error from the run-to-run scatter
                (b_, a_), cov = np.polyfit(x[ok], y[ok], 1, w=1 / e[ok], cov="unscaled")
                chi2 = float(np.sum(((y[ok] - a_ - b_ * x[ok]) / e[ok]) ** 2))
                birge = max(1.0, np.sqrt(chi2 / (ok.sum() - 2)))     # inflate the error if the fit is poor
                nums[f"ll_fitw_{name}"] = (float(a_), float(np.sqrt(cov[1, 1]) * birge))
                nums[f"ll_fitw_chi2_{name}"] = (chi2, int(ok.sum() - 2))
            xx = np.linspace(0, x.max() * 1.05, 50)
            ax.plot(xx, a_ + b_ * xx, color=INK2, ls=ls, lw=1)
            ax.errorbar(x, y, yerr=e, ls="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=0)
    ax.axhline(T_NI_LIT, color=INK2, ls=":", lw=1)
    from matplotlib.lines import Line2D
    hl = []
    for name, (ls, mk, lab) in style.items():
        if name in peaks:
            fw = nums.get(f"ll_fitw_{name}")
            txt = f"{lab}" + (rf", $T_{{NI}}(\infty)={fw[0]:.4f}\pm{fw[1]:.4f}$" if fw else
                              (rf", fit $T_{{NI}}(\infty)={fits[name]:.4f}$" if name in fits else ""))
            hl.append(Line2D([], [], color=INK2, ls=ls, lw=1, marker=mk, ms=4.5,
                             mfc=INK2 if name == "llbulk" else "none", label=txt))
    hl.append(Line2D([], [], color=INK2, ls=":", lw=1, label=rf"literature $T_{{NI}}={T_NI_LIT}$"))
    ax.set_xlabel(r"$1/L^3$")
    ax.set_ylabel(r"$T$ of the $C_V$ maximum")
    ax.legend(handles=hl, loc="upper left", fontsize=6.5, title="colours: $L$ as in (a)", title_fontsize=6.5)
    nums["ll_peaks"] = peaks.get("llbulk", {})
    nums["ll_peaks_heat"] = peaks.get("llheat", {})
    if "llbulk" in fits:
        nums["ll_TNI_inf"] = fits["llbulk"]
    if "llheat" in fits:
        nums["ll_TNI_inf_heat"] = fits["llheat"]
    if len(fits) == 2:
        nums["ll_TNI_mid"] = 0.5 * (fits["llbulk"] + fits["llheat"])
        nums["ll_TNI_halfwidth"] = 0.5 * abs(fits["llheat"] - fits["llbulk"])
    nums["ll_peak_err"] = perr
    w = [nums[k] for k in ("ll_fitw_llbulk", "ll_fitw_llheat") if k in nums]
    if w:                          # weighted mean of the two protocols
        wi = np.array([1 / v[1] ** 2 for v in w])
        m = float(np.sum(wi * np.array([v[0] for v in w])) / wi.sum())
        nums["ll_TNI_comb"] = (m, float(1 / np.sqrt(wi.sum())))
        nums["ll_TNI_comb_nsigma"] = float((T_NI_LIT - m) * np.sqrt(wi.sum()))
    for a_, l_ in zip(axes, "abc"):
        panel(a_, l_)
    fig.tight_layout()
    save(fig, out)


# ─── captions ───────────────────────────────────────────────────────────────

def captions(nums, args):
    f = lambda k, d=3: "?" if nums.get(k) is None else f"{nums[k]:.{d}f}"
    pct = lambda k: "?" if nums.get(k) is None else f"{100 * nums[k]:.0f}"
    tT = {k: v for k, v in nums.items() if k.startswith("twistT_p")}
    tT_txt = "; ".join(f"$p_0={k[8:]}$: " + ", ".join(f"{r:.3f} ($T={T:g}$)" for T, r in sorted(v.items()))
                       for k, v in sorted(tT.items()))
    peaks = nums.get("ll_peaks", {})
    heat = nums.get("ll_peaks_heat", {})
    cross = nums.get("homeo_cross", {})
    cross_txt = ", ".join(f"{v[1]:.2f} ($R\\approx{N / 2 - 1:g}$)" for N, v in sorted(cross.items()))
    th = nums.get("chiral_theta", {})
    th_txt = ", ".join(f"${c:+.1f}^\\circ$ ($R/p_0={x:g}$)" for x, c in sorted(th.items()))
    slope_txt = ("" if nums.get("chiral_slope_deg") is None else
                 f" (initial slope ${nums['chiral_slope_deg']:.0f}^\\circ$ per unit $R/p_0$, dashed)")
    if nums.get("ll_TNI_comb") is not None:
        fc, fh = nums.get("ll_fitw_llbulk"), nums.get("ll_fitw_llheat")
        m, e = nums["ll_TNI_comb"]
        prot = "; ".join(f"{lab} ${v[0]:.4f}\\pm{v[1]:.4f}$" for lab, v in (("cooling", fc), ("heating", fh)) if v)
        # combined error with two significant digits (0.00045 rounded to 0.0004 would not match the n-sigma below)
        ll_txt = (f"Weighted extrapolations: {prot}; combined $T_{{NI}}={m:.5f}\\pm{e:.5f}$, "
                  f"{100 * abs(m / 1.1232 - 1):.2f}\\% ({nums['ll_TNI_comb_nsigma']:.1f} standard deviations) below the "
                  f"literature value $1.1232$.")
        pc, ph = nums.get("ll_peaks", {}), nums.get("ll_peaks_heat", {})
        big = sorted(set(pc) & set(ph))[-2:]     # cooling below heating for the largest lattices: metastable isotropic
        if len(big) == 2 and all(ph[L] > pc[L] for L in big):
            ll_txt += (" For the two largest lattices the maximum on cooling lies "
                       + " and ".join(f"${ph[L] - pc[L]:.3f}$ ($L={L}$)" for L in big)
                       + " below that on heating, as expected from a metastable isotropic phase on cooling.")
    elif nums.get("ll_TNI_mid") is not None:
        ll_txt = (f"The extrapolations give {f('ll_TNI_inf', 4)} (cooling) and {f('ll_TNI_inf_heat', 4)} (heating), "
                  f"i.e. $T_{{NI}}={f('ll_TNI_mid', 4)}\\pm{f('ll_TNI_halfwidth', 4)}$, within "
                  f"{100 * abs(nums['ll_TNI_mid'] / 1.1232 - 1):.1f}\\% of the literature value $1.1232$; the difference "
                  f"between the two protocols is of the order of the statistical uncertainty of the peak positions.")
    else:
        ll_txt = f"A linear extrapolation of the sizes $L\\geq16$ gives $T_{{NI}}(\\infty)={f('ll_TNI_inf', 4)}$."
    oth = nums.get("cano_other", [])
    def step(x):                     # nearest Grandjean step, at d/p = (2k+1)/4
        return (2 * round((4 * x - 1) / 2) + 1) / 4
    other_txt = "".join(f"; one run just {'above' if x > step(x) else 'below'} the step at $d/p={step(x):g}$ "
                        f"($d/p={x:.2f}$) has {abs(m - round(mm))} half turn{'s' if abs(m - round(mm)) > 1 else ''} "
                        f"{'less' if round(mm) < m else 'more'}" for d, x, mm, m in oth)
    return rf"""% Generated by make_figures.py — numbers come from the simulation data.
\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_chol.pdf}}
\caption{{Cholesteric pitch with the chiral GHRL potential (E7 elastic constants). (a) Equilibrium pitch at $T=0.3$,
measured from the twist rate in the centre of a $6\times6\times48$ box (periodic in $x$ and $y$, free along the helix
axis), for initial helices with $p_i=0.8\,p_0$, $p_0$ and $1.25\,p_0$, compared with the ground state of the lattice
pair potential, $p=2\pi/\arcsin(2\pi/p_0)$ (dashed), and with the nominal $p_0$ (dotted). At this temperature the two
cannot be distinguished: for $p_0\geq16$ they differ by less than the spread between initial conditions (
{", ".join(f"{100 * v:.0f}\\% ($p_0={k:g}$)" for k, v in sorted(nums.get("twist_spread_by_p0", {}).items()))}),
and for $p_0\leq12$ the thermal shortening of panel (b) is comparable to the lattice correction. (b) Ratio between the
measured pitch and the lattice prediction versus temperature ({tT_txt}): as $T\to0$ the ratio tends to 1, whereas the
nominal pitch would give {", ".join(f"{p0 / (2 * np.pi / np.arcsin(2 * np.pi / p0)):.2f} ($p_0={p0:g}$)" for p0 in (10, 8))};
thermal fluctuations shorten the pitch by a few percent.
(c) Total twist across a cell with planar anchoring along $x$ on both plates ($p_0=20$, $p={f('cano_p_eff', 2)}$) versus
thickness $d$, after cooling from the isotropic phase: the twist is quantized in half turns and follows the
minimum-energy staircase (Grandjean steps) in {nums.get('cano_match', '?')} of {nums.get('cano_n', '?')} runs; in
{nums.get('cano_extra', '?')} runs, all just below a step, one extra half turn formed at higher temperature, where the
pitch is shorter (panel b), remains trapped on cooling, as in Grandjean--Cano experiments{other_txt}.}}
\label{{fig:chol}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_homeo.pdf}}
\caption{{Lebwohl--Lasher droplets with homeotropic anchoring of strength $W_s$, at $T={nums.get('homeo_T', '?')}$.
(a) Radial order $S_r=\langle(3(\hat{{\mathbf n}}\cdot\hat{{\mathbf r}})^2-1)/2\rangle$ versus $W_s$ for four radii
(points: individual runs; lines: mean): weak
anchoring gives a nearly uniform (axial) director field ($S_r\approx0$), strong anchoring the radial hedgehog
($S_r\to1$); $S_r$ grows continuously in the axial regime and jumps discontinuously to the radial
structure (first-order transition; at the transition of the largest droplet the two runs ended in different
structures), and the transition moves to weaker anchoring for larger droplets. (b) The same data versus $W_sR$ collapse
onto a single curve, as expected from the competition between the surface energy, $\sim W R^2$, and the elastic
energy of the hedgehog, $\sim K R$ (Lavrentovich): $S_r=0.5$ at $W_sR\approx{f('homeo_WR_mean', 2)}$ (values
{cross_txt}; spread {pct('homeo_WR_spread')}\% in $W_sR$ against {pct('homeo_W_spread')}\% in $W_s$). (c,d) Cross-sections
of the largest droplet containing the symmetry axis, for the weakest and strongest anchoring; segments are coloured by
$(\hat{{\mathbf n}}\cdot\hat{{\mathbf r}})^2$.}}
\label{{fig:homeo}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_chiral.pdf}}
\caption{{Chiral droplets (GHRL, E7 constants, $N={nums.get('chiral_N', '?')}$) with degenerate planar anchoring
($W_s=1$) at $T=0.2$. (a,b) Cross-sections through the centre perpendicular to the bipolar axis $\mathbf d$ for the
largest positive and negative $R/p_0$, coloured by $(\hat{{\mathbf n}}\cdot\hat{{\boldsymbol\varphi}})(\hat{{\mathbf n}}\cdot\mathbf d)$:
the director twists about the bipolar axis with opposite handedness. (c) Twist angle $\theta_t=\tfrac12\arctan\left[2\langle(\hat{{\mathbf n}}\cdot\hat{{\boldsymbol\varphi}})(\hat{{\mathbf n}}\cdot\mathbf d)\rangle/\langle(\hat{{\mathbf n}}\cdot\mathbf d)^2-(\hat{{\mathbf n}}\cdot\hat{{\boldsymbol\varphi}})^2\rangle\right]$,
the average tilt of the director from the bipolar axis towards the azimuthal direction, versus $R/p_0$ (mean and spread
over 4 runs): the achiral droplet is untwisted, and the handedness of the twist follows the sign of $p_0$
[$\theta_t(-R/p_0)=-\theta_t(R/p_0)$ within ${f('chiral_antisym_deg', 1)}^\circ$]. At weak chirality $\theta_t$ is linear in
$R/p_0${slope_txt}, close to the natural twist over the radius, $q_0R=2\pi R/p_0$; at stronger chirality the response is
sublinear ($\theta_t={f('chiral_theta_max', 0)}^\circ$ at the largest $R/p_0$).}}
\label{{fig:chiral}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_llbulk.pdf}}
\caption{{Isotropic--nematic transition of the Lebwohl--Lasher model in periodic $L^3$ lattices. (a) Nematic order
parameter and (b) heat capacity per site on cooling (solid) and on heating from the aligned state (dashed); the dotted
line marks the literature value $T_{{NI}}=1.1232$. (c) Temperature of the heat-capacity maximum versus $1/L^3$ (filled:
cooling, open: heating; peaks on cooling {", ".join(f"{v:.4f}" for L, v in sorted(peaks.items()))}, on heating
{", ".join(f"{v:.4f}" for L, v in sorted(heat.items()))} for $L={", ".join(str(L) for L in sorted(peaks))}$). The peak
position is the vertex of a parabola through the three highest points of $C_V(T)$ (temperature step $0.004$), and its
error bar is the standard error over the three runs of each protocol, with the run-to-run scatter pooled over the
two protocols. Lines: linear fits of the sizes $L\geq16$, weighted by these
errors; the error of each extrapolation is multiplied by $\sqrt{{\chi^2/\nu}}$ when this ratio exceeds 1. {ll_txt}}}
\label{{fig:llbulk}}
\end{{figure*}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nums = {}
    fig_chol(args.runs, os.path.join(args.out, "fig_chol"), nums)
    fig_homeo(args.runs, os.path.join(args.out, "fig_homeo"), nums)
    fig_chiral(args.runs, os.path.join(args.out, "fig_chiral"), nums)
    fig_llbulk(args.runs, os.path.join(args.out, "fig_llbulk"), nums)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "figures.tex"), "w") as fh:  # next to figs/
        fh.write(captions(nums, args))
    with open(os.path.join(args.out, "numbers.txt"), "w") as fh:
        for k, v in nums.items():
            fh.write(f"{k}: {v}\n")
    for k, v in nums.items():
        print(f"{k}: {v}")
    print(f"\nFigures in {args.out}, captions in figures.tex")


if __name__ == "__main__":
    main()
