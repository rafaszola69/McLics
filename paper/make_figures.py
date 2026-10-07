#!/usr/bin/env python3
"""
make_figures.py — figures for the pear-potential test case of the McLics paper.

Reads runs/ (produced by run.sh) and writes to figs/:
  fig_pear_bulk.pdf/.png      (a) S and P vs T, bulk 20^3 with free x,y (Dhakal & Selinger conditions);
                              (b) transition temperatures vs B1 (I-N-polar vs direct I-polar)
  fig_pear_droplet.pdf/.png   (a,b) hedgehog cross-sections at low T for C=+0.3 and C=-0.3;
                              (c) radial polarization vs T for C=+0.3, -0.3, 0; (d) P_r/C for C=0.1, 0.2, 0.3
  fig_pear_profile.pdf/.png   (a) polarization profile P(r) for three droplet sizes; (b) r P(r)
  figures.tex                 LaTeX figure environments with English captions (numbers filled in from the data)
  numbers.txt                 the numbers quoted in the captions, for checking

Usage: python3 make_figures.py [--profile-T 0.75] [--slice-T 0.1] [--slice-N 41] [--curve-N 41] [--linear-N 31]
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
C_POS, C_NEG, C_ZERO = "#2a78d6", "#eb6834", "#1baf7a"
C_EXTRA = {0.1: "#008300", 0.2: "#e87ba4", 0.3: C_POS}
INOUT = LinearSegmentedColormap.from_list("inout", ["#2a78d6", "#c9c8c3", "#e34948"])
SIZE_RAMP = LinearSegmentedColormap.from_list("sizes", ["#86b6ef", "#184f95"])

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


# ─── reading ────────────────────────────────────────────────────────────────

def load_group(path):
    """List of po.dat DataFrames (one per finished seed) in path/seed*/."""
    runs = []
    for d in sorted(glob.glob(os.path.join(path, "seed*"))):
        f = os.path.join(d, "po.dat")
        if not os.path.exists(os.path.join(d, "DONE")) or not os.path.exists(f):
            continue
        df = pd.read_csv(f, sep=r"\s+")
        df["T"] = df["T"].round(4)
        df.attrs["dir"] = d
        runs.append(df)
    return runs


def mean_std(runs, col):
    if not runs or col not in runs[0]:
        return None
    g = pd.concat([r[["T", col]] for r in runs]).groupby("T")[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std().fillna(0)}).sort_index()


def snapshot(run, T):
    """Snapshot of a run at temperature T; the code writes one per temperature, in po.dat order."""
    files = {}
    for f in glob.glob(os.path.join(run.attrs["dir"], "director_field_*.csv*")):
        m = re.search(r"director_field_(\d+)\.csv(\.gz)?$", f)
        if m:
            files[int(m.group(1))] = f
    Ts = list(run["T"])
    if len(files) != len(Ts) or T not in Ts:
        return None
    idx = sorted(files, reverse=Ts[0] > Ts[-1])
    return files[idx[Ts.index(T)]]


def site_ndotr(csv, n):
    df = pd.read_csv(csv)
    df = df[df["pt"] > 0]
    c = n // 2
    r = df[["x", "y", "z"]].to_numpy(float) - c
    rr = np.linalg.norm(r, axis=1)
    ok = rr > 0
    return df[ok], rr[ok], np.sum(df[["nx", "ny", "nz"]].to_numpy()[ok] * r[ok], axis=1) / rr[ok]


def steepest(s, min_rise=0.3):
    if s is None or len(s) < 3 or s["mean"].max() < min_rise:
        return None
    T = s.index.values
    return float(T[np.argmax(np.abs(np.gradient(s["mean"].values, T)))])


def polar_onset(s, weak=0.1):
    """Polar onset: steepest rise of P when P becomes large; otherwise (strongly splayed polar phase at weak B1,
    where the global P stays small) the highest temperature at which P first exceeds `weak`. Returns (T, weak_used)."""
    t = steepest(s)
    if t is not None:
        return t, False
    if s is None or s["mean"].max() < weak:
        return None, False
    above = s.index.values[s["mean"].values > weak]
    return float(above.max()), True


def radius(n):
    return n / 2 - 1


# ─── figure 1: bulk ─────────────────────────────────────────────────────────

def fig_bulk(runs_dir, out, nums):
    sel = load_group(os.path.join(runs_dir, "selinger", "C0.3"))
    phase = {}
    group = "phasefine" if glob.glob(os.path.join(runs_dir, "phasefine", "B*", "seed*", "DONE")) else "phase"
    nums["phase_group"] = group
    for d in glob.glob(os.path.join(runs_dir, group, "B*")):
        b = float(os.path.basename(d)[1:])
        r = load_group(d)
        if r:
            phase[b] = r
    if not sel and not phase:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM))
    ax = axes[0]
    if sel:
        S, P = mean_std(sel, "S"), mean_std(sel, "P")
        t_ni, t_p = steepest(S), steepest(P)
        nums["bulk_TNI"], nums["bulk_TP"] = t_ni, t_p
        nums["bulk_S_low"], nums["bulk_P_low"] = float(S["mean"].iloc[0]), float(P["mean"].iloc[0])
        nums["bulk_nseeds"] = len(sel)
        if t_ni and t_p:
            ax.axvspan(t_p, t_ni, color=SHADE, lw=0, zorder=0)
        for s, lab, c, m in ((S, r"$S$", C_POS, "o"), (P, r"$P$", C_NEG, "s")):
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.2, lw=0)
            ax.plot(s.index, s["mean"], color=c, marker=m, ms=2.5, label=lab)
        ax.set_xlabel(r"$T$")
        ax.set_ylabel("order parameter")
        ax.set_ylim(-0.02, 1.0)
        ax.legend(loc="upper right")
    panel(ax, "a")
    ax = axes[1]
    if phase:
        rows, weak = [], []
        for b, r in sorted(phase.items()):
            t_pp, w = polar_onset(mean_std(r, "P"))
            rows.append((b, steepest(mean_std(r, "S")), t_pp))
            if w:
                weak.append(b)
        nums["phase_weak"] = weak
        nums["phase"] = rows
        Tgrid = np.sort(next(iter(phase.values()))[0]["T"].unique())
        nums["phase_dT"] = float(np.median(np.diff(Tgrid)))
        B = np.array([b for b, _, _ in rows])
        ts = np.array([np.nan if t is None else t for _, t, _ in rows])
        tp = np.array([np.nan if t is None else t for _, _, t in rows])
        ax.plot(B, ts, color=C_POS, marker="o", ms=3.5, label="nematic order onset (this work)")
        ax.plot(B, tp, color=C_NEG, marker="s", ms=3.5, ls="--", label="polar order onset (this work)")
        ref = os.path.join(HERE, "selinger_phase_diagram_digitized.csv")
        top = np.nanmax(np.concatenate([ts, tp])) * 1.12
        if os.path.exists(ref):
            dsd = pd.read_csv(ref, comment="#")
            for i, (line, grp) in enumerate(dsd.groupby("line")):
                ax.plot(grp["B"], grp["T"], ls="none", marker="o", ms=4.5, mfc="none", mec=INK2, mew=0.8,
                        label="Dhakal and Selinger" if i == 0 else None, zorder=1)
            dev = []
            for b, t_s, t_p in rows:
                for t, lines in ((t_s, ("NI", "IP")), (t_p, ("NP", "IP"))):
                    g = dsd[dsd["line"].isin(lines)].sort_values("B")
                    if t is not None and g["B"].min() <= b <= g["B"].max():
                        dev.append(abs(t - float(np.interp(b, g["B"], g["T"]))))
            nums["selinger_maxdev"] = max(dev) if dev else None
            nums["selinger_npts"] = len(dev)
            top = max(top, dsd["T"].max() * 1.12)
            if B.max() <= 0.8:
                ax.text(0.05, 1.55, "isotropic", color=INK2, style="italic")
                ax.text(0.03, 0.78, "nematic", color=INK2, style="italic")
                ax.text(0.30, 0.45, "polar", color=INK2, style="italic")
        ax.set_xlabel(r"$B$")
        ax.set_ylabel(r"transition temperature")
        ax.set_ylim(0, top)
        ax.legend(loc="lower right")
    panel(ax, "b")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)
    return True


# ─── figure 2: droplets ─────────────────────────────────────────────────────

def draw_slice(ax, csv, n):
    df, rr, ndotr = site_ndotr(csv, n)
    c = n // 2
    stride = max(1, round(n / 20))
    m = ((df["y"] == c) & (df["x"] % stride == 0) & (df["z"] % stride == 0)).to_numpy()
    sl = df[m]
    ax.quiver(sl["x"], sl["z"], sl["nx"], sl["nz"], ndotr[m], cmap=INOUT, clim=(-1, 1),
              pivot="mid", angles="xy", scale_units="xy", scale=1 / (0.85 * stride),
              width=0.008, headwidth=3.5, headlength=3.5, headaxislength=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def sym_pr(dr, n, cpos=0.3):
    p = mean_std(load_group(os.path.join(dr, f"N{n}", f"C{cpos}")), "Pr_sph")
    q = mean_std(load_group(os.path.join(dr, f"N{n}", f"C-{cpos}")), "Pr_sph")
    if p is None or q is None:
        return p
    d = p.join(q, lsuffix="_p", rsuffix="_n", how="inner")
    return pd.DataFrame({"mean": (d["mean_p"] - d["mean_n"]) / 2})


def fig_droplet(runs_dir, args, window, out, nums):
    dr = os.path.join(runs_dir, "droplets")
    fig = plt.figure(figsize=(DOUBLE, 118 * MM))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.35], height_ratios=[1, 0.05, 0.75], hspace=0.45, wspace=0.3)
    n = args.slice_N
    for k, (c, letter, lab) in enumerate(((0.3, "a", r"$C=+0.3$"), (-0.3, "b", r"$C=-0.3$"))):
        ax = fig.add_subplot(gs[0, k])
        runs = load_group(os.path.join(dr, f"N{n}", f"C{c}"))
        csv = snapshot(runs[0], args.slice_T) if runs else None
        if csv:
            draw_slice(ax, csv, n)
        ax.set_title(f"{lab},  $T={args.slice_T:g}$", fontsize=8, color=INK)
        panel(ax, letter)
    holder = fig.add_subplot(gs[1, 0:2])
    holder.axis("off")
    cax = holder.inset_axes([0.2, 0.0, 0.6, 1.0])
    sm = plt.cm.ScalarMappable(cmap=INOUT, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(r"$\hat{\mathbf{n}}\cdot\hat{\mathbf{r}}$", fontsize=8, labelpad=1)
    ax = fig.add_subplot(gs[2, 0:2])
    lin = {}
    for cval in (0.1, 0.2, 0.3):
        if cval == 0.3:
            s = sym_pr(dr, args.linear_N)
        else:
            s = mean_std(load_group(os.path.join(dr, f"N{args.linear_N}", f"C{cval}")), "Pr_sph")
        if s is not None:
            lin[cval] = s["mean"] / cval
            ax.plot(s.index, s["mean"] / cval, color=C_EXTRA[cval], marker="o", ms=2,
                    label=rf"$C={cval:g}$" + (r" (sym.)" if cval == 0.3 else ""))
    if window:
        ax.axvspan(*window, color=SHADE, lw=0, zorder=0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$P_r\,/\,C$")
    ax.legend(loc="upper right", ncol=3)
    panel(ax, "d")
    nums["linear"] = {c: {T: float(v.loc[T]) for T in (0.75, 0.95) if T in v.index} for c, v in lin.items()}
    ax = fig.add_subplot(gs[:, 2])
    if window:
        ax.axvspan(*window, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6)
    m = args.curve_N
    for c, col, mk, lab in ((0.3, C_POS, "o", r"$C=+0.3$"), (-0.3, C_NEG, "s", r"$C=-0.3$")):
        s = mean_std(load_group(os.path.join(dr, f"N{m}", f"C{c}")), "Pr_sph")
        if s is not None:
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.2, lw=0)
            ax.plot(s.index, s["mean"], color=col, marker=mk, ms=2.5, label=lab)
    zero = load_group(os.path.join(dr, f"N{m}", "C0.0"))
    for i, r in enumerate(zero):
        ax.plot(r["T"], r["Pr_sph"], color=C_ZERO, lw=0.9, label=r"$C=0$ (each replica)" if i == 0 else None)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$P_r=\langle\hat{\mathbf{n}}\cdot\hat{\mathbf{r}}\rangle$")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    panel(ax, "c")
    sp = sym_pr(dr, m)
    if sp is not None:
        nums["Pr_curve"] = {T: float(sp.loc[T, "mean"]) for T in (0.5, 0.75, 0.95) if T in sp.index}
    nums["zero_final"] = [float(r["Pr_sph"].iloc[-1]) for r in zero]
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ─── figure 3: radial profile ───────────────────────────────────────────────

def radial_profile(runs, n, T, sign=1):
    rs, ps = [], []
    for r in runs:
        csv = snapshot(r, T)
        if csv is None:
            continue
        _, rr, nd = site_ndotr(csv, n)
        rs.append(rr)
        ps.append(sign * nd)
    if not rs:
        return None
    rr, p = np.concatenate(rs), np.concatenate(ps)
    g = pd.DataFrame({"s": np.floor(rr).astype(int), "p": p}).groupby("s")["p"]
    out = pd.DataFrame({"p": g.mean(), "sem": g.std() / np.sqrt(g.count())})
    out.index = out.index + 0.5
    return out


def fig_profile(runs_dir, T, out, nums):
    dr = os.path.join(runs_dir, "droplets")
    sizes = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(os.path.join(dr, "N*")))
    sizes = [n for n in sizes if load_group(os.path.join(dr, f"N{n}", "C0.3"))]
    if not sizes:
        return
    cols = {n: SIZE_RAMP(i / max(1, len(sizes) - 1)) for i, n in enumerate(sizes)}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 58 * MM))
    nums["K"] = {}
    for n in sizes:
        a = radial_profile(load_group(os.path.join(dr, f"N{n}", "C0.3")), n, T)
        b = radial_profile(load_group(os.path.join(dr, f"N{n}", "C-0.3")), n, T, sign=-1)
        if a is None or b is None:
            continue
        prof = pd.DataFrame({"p": (a["p"] + b["p"]) / 2, "sem": np.hypot(a["sem"], b["sem"]) / 2}).dropna()
        r = prof.index.values
        lab = rf"$R\approx{radius(n):g}$"
        for ax, y, e in ((axes[0], prof["p"], prof["sem"]), (axes[1], r * prof["p"], r * prof["sem"])):
            ax.fill_between(r, y - e, y + e, color=cols[n], alpha=0.2, lw=0)
            ax.plot(r, y, color=cols[n], lw=1.2, label=lab)
        R = radius(n)
        m = (r >= max(3, R / 3)) & (r <= R - 2)
        if m.sum() >= 2:
            nums["K"][n] = float(np.median(r[m] * prof["p"].values[m]))
    if nums["K"]:
        K = np.mean(list(nums["K"].values()))
        rr = np.linspace(2, max(radius(n) for n in sizes), 100)
        axes[0].plot(rr, K / rr, color=INK2, ls=":", lw=1, label=rf"$\kappa/r$, $\kappa={K:.2f}$")
        axes[1].axhline(K, color=INK2, ls=":", lw=1)
        nums["K_mean"] = float(K)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel(r"$r$ (lattice units)")
    axes[0].set_ylabel(r"$P(r)=\langle\hat{\mathbf{n}}\cdot\hat{\mathbf{r}}\rangle_{\mathrm{shell}}$")
    axes[1].set_ylabel(r"$r\,P(r)$")
    axes[0].legend(loc="upper right")
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}")
    plt.close(fig)


# ─── captions ───────────────────────────────────────────────────────────────

def fmt(x, d=2):
    return "?" if x is None else f"{x:.{d}f}"


def captions(nums, args):
    tni, tp = nums.get("bulk_TNI"), nums.get("bulk_TP")
    phase = nums.get("phase", [])
    tol = 1.5 * nums.get("phase_dT", 0.05)
    direct = [b for b, ts, tpp in phase if ts is not None and tpp is not None and abs(ts - tpp) < tol]
    nematic = [b for b, ts, tpp in phase if b not in direct]
    below = [b for b in nematic if not direct or b < min(direct)]
    weak = nums.get("phase_weak", [])
    weak_txt = ("" if not weak else
                f" For $B\\le{max(weak):g}$ the polar phase is strongly splayed and the global polarization stays small"
                f" (it rises and then decreases on cooling while $\\langle\\nabla\\cdot\\hat{{\\mathbf n}}\\rangle$ grows),"
                f" so the polar onset is taken as the temperature at which $P$ first exceeds 0.1.")
    dev = nums.get("selinger_maxdev")
    cmp_txt = ("" if dev is None else
               f" Open circles: phase diagram of Dhakal and Selinger; the transition temperatures agree within {dev:.2f}"
               f" (largest deviation over {nums.get('selinger_npts')} comparisons, of the order of the temperature step).")
    if direct and below:
        thr = f"for $B\\geq{min(direct):g}$ both orders appear together (direct isotropic--polar transition); the change of\nsequence happens between $B={max(below):g}$ and ${min(direct):g}$"
    elif direct:
        thr = f"for $B\\geq{min(direct):g}$ both orders appear together (direct isotropic--polar transition)"
    else:
        thr = "no direct isotropic--polar transition is found in the range studied"
    lin = nums.get("linear", {})
    lin_txt = ""
    if all(c in lin and 0.75 in lin[c] for c in (0.1, 0.2, 0.3)):
        lin_txt = (f" At $T=0.75$, $P_r/C = {lin[0.1][0.75]:.3f}$, ${lin[0.2][0.75]:.3f}$ and ${lin[0.3][0.75]:.3f}$ "
                   f"for $C=0.1$, $0.2$ and $0.3$.")
    K = nums.get("K_mean")
    ks = ", ".join(f"{v:.2f} ($R\\approx{radius(n):g}$)" for n, v in sorted(nums.get("K", {}).items()))
    tex = rf"""% Generated by make_figures.py — numbers come from the simulation data.
\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_pear_bulk.pdf}}
\caption{{Flexoelectric (pear-shaped) potential, Eq.~(12), in a $20^3$ lattice with free boundaries along $x$ and $y$
and periodic along $z$, as in Dhakal and Selinger. (a) Nematic ($S$) and polar ($P$) order parameters on cooling for
$A=1.5$, $B=0.09$ and $C=0.3$ (mean and $\\pm1$ standard deviation over {nums.get('bulk_nseeds', '?')} independent runs). The system
goes from isotropic to nematic at $T\approx{fmt(tni)}$ and from nematic to polar at $T\approx{fmt(tp)}$; the shaded
region is the non-polar nematic window. Below the polar transition $S$ drops to $\approx{fmt(nums.get('bulk_S_low'))}$
because the polar phase develops spontaneous splay, which the free lateral boundaries accommodate.
(b) Onset temperatures of nematic and polar order as a function of the polar coupling $B$ (same $A$ and $C$):
for weak $B$ the nematic phase appears between the isotropic and polar phases, while {thr}.{weak_txt}{cmp_txt}}}
\label{{fig:pear_bulk}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_pear_droplet.pdf}}
\caption{{Nematic droplets of the pear-shaped model with homeotropic anchoring ($W=1.5$). (a,b) Cross-sections
through the droplet centre ($N={args.slice_N}$, radius $R\approx{radius(args.slice_N):g}$) at $T={args.slice_T:g}$:
the homeotropic anchoring imposes a radial hedgehog, and the sign of $C$ selects outward ($C>0$) or inward ($C<0$)
polarization; arrows are coloured by $\hat{{\mathbf n}}\cdot\hat{{\mathbf r}}$. (c) Radial polarization
$P_r=\langle\hat{{\mathbf n}}\cdot\hat{{\mathbf r}}\rangle$ versus temperature for $R\approx{radius(args.curve_N):g}$.
Within the non-polar nematic window of the bulk (shaded), the splay imposed by the hedgehog induces a polarization
with the sign of $C$ (e.g. $P_r={fmt(nums.get('Pr_curve', {}).get(0.75), 3)}$ at $T=0.75$), and $P_r(C)=-P_r(-C)$
as required by the $\hat{{\mathbf n}}\to-\hat{{\mathbf n}}$, $C\to-C$ symmetry of the potential. For $C=0$ there is no
polarization in the window, and below the polar transition each run breaks the symmetry spontaneously
(final values {", ".join(f"{v:+.2f}" for v in nums.get('zero_final', []))}). (d) $P_r/C$ for $C=0.1$, $0.2$ and $0.3$
($R\approx{radius(args.linear_N):g}$; $C=0.3$ symmetrized over $\pm C$): the curves collapse in the window, i.e.
the induced polarization is linear in $C$.{lin_txt}}}
\label{{fig:pear_droplet}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_pear_profile.pdf}}
\caption{{Radial profile of the induced polarization at $T={args.profile_T:g}$, averaged over spherical shells and over
runs with $C=\pm0.3$ (symmetrized), for three droplet radii. (a) $P(r)$ collapses when plotted against the distance
$r$ from the centre, showing that the polarization responds locally to the splay of the hedgehog,
$\nabla\cdot\hat{{\mathbf n}}=2/r$. (b) $r\,P(r)$ reaches a common plateau, i.e. $P(r)\simeq\kappa/r$ away from the defect
core and from the anchoring surface, with $\kappa\approx{fmt(K)}$ (plateau values {ks}). Averaging $\kappa/r$ over the droplet
volume gives $P_r\simeq 3\kappa/2R$. Shaded bands: statistical error of each shell average (standard deviation over the sites
of the shell divided by the square root of their number).}}
\label{{fig:pear_profile}}
\end{{figure*}}
"""
    return tex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--profile-T", type=float, default=0.75)
    ap.add_argument("--slice-T", type=float, default=0.1)
    ap.add_argument("--slice-N", type=int, default=41)
    ap.add_argument("--curve-N", type=int, default=41)
    ap.add_argument("--linear-N", type=int, default=31)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nums = {}
    fig_bulk(args.runs, os.path.join(args.out, "fig_pear_bulk"), nums)
    window = (nums["bulk_TP"], nums["bulk_TNI"]) if nums.get("bulk_TP") and nums.get("bulk_TNI") else None
    fig_droplet(args.runs, args, window, os.path.join(args.out, "fig_pear_droplet"), nums)
    fig_profile(args.runs, args.profile_T, os.path.join(args.out, "fig_pear_profile"), nums)
    with open(os.path.join(HERE, "figures.tex"), "w") as f:
        f.write(captions(nums, args))
    with open(os.path.join(args.out, "numbers.txt"), "w") as f:
        for k, v in nums.items():
            f.write(f"{k}: {v}\n")
    for k, v in nums.items():
        print(f"{k}: {v}")
    print(f"\nFigures in {args.out}, captions in figures.tex")


if __name__ == "__main__":
    main()
