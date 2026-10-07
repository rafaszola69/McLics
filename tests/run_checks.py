#!/usr/bin/env python3
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""
run_checks.py — quick checks of McLics against known answers.

Each check writes a small input file, runs mclics in a temporary folder and compares the output with a value that
is known exactly (ground-state energies at T = 0, sweep points, file names, column headers, error handling) or
within a small tolerance (radial order of a hedgehog). The whole set takes a few seconds and needs only the Python
standard library.

Usage:
    python3 tests/run_checks.py                    # uses ./mclics, CPU modes
    python3 tests/run_checks.py path/to/mclics     # another program
    python3 tests/run_checks.py --gpu              # the same checks with the GPU modes (thermalGPU, ...)
    python3 tests/run_checks.py -k ground          # only the checks whose name contains "ground"
    python3 tests/run_checks.py --keep             # keep the run folders (printed at the end)

Exit status: 0 if every check passed, 1 otherwise.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


class Fail(Exception):
    pass


class Runner:
    def __init__(self, program, gpu, workdir):
        self.program = program
        self.gpu = gpu
        self.workdir = workdir
        self.count = 0

    def run(self, params, files=None, setup=None, env=None):
        """Runs mclics with the given parameters (dict or list of lines) in a new folder; returns that folder."""
        self.count += 1
        d = os.path.join(self.workdir, "run%03d" % self.count)
        os.makedirs(d)
        lines = []
        for k, v in (params.items() if isinstance(params, dict) else params):
            if k.lower() in ("evol", "mode") and self.gpu:
                v = v + "GPU"
            lines.append("%s %s" % (k, v))
        with open(os.path.join(d, "input.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        for name, text in (files or {}).items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        if setup:
            setup(d)
        e = dict(os.environ)
        e.setdefault("OMP_NUM_THREADS", "2")
        e.update(env or {})
        p = subprocess.run([self.program, "input.txt"], cwd=d, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, errors="replace", timeout=600)
        with open(os.path.join(d, "run.log"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        self.last = p
        return d


def read_po(d):
    if not os.path.exists(os.path.join(d, "po.dat")):
        raise Fail("no po.dat; see %s/run.log" % d)
    with open(os.path.join(d, "po.dat")) as f:
        lines = [l.split() for l in f if l.strip()]
    header = lines[0]
    rows = [dict(zip(header, map(float, l))) for l in lines[1:]]
    return header, rows


def ok_run(r, d):
    if r.last.returncode != 0:
        raise Fail("mclics exited with status %d; see %s/run.log" % (r.last.returncode, d))


def close(name, value, expected, tol):
    if not (abs(value - expected) <= tol):
        raise Fail("%s = %.7g, expected %.7g (tolerance %g)" % (name, value, expected, tol))


def base(**kw):
    """Small periodic bulk at T = 0 starting from the uniform state along z: one measurement sweep."""
    p = dict(Nx=8, Ny=8, Nz=8, geometry="bulk", xbound="periodic", ybound="periodic", zbound="periodic",
             potential="ll", A=1, evol="thermal", Ti=0, Tf=0, dT=-0.1, MCT=0, MCS=1, ic="homogeneous",
             theta_0=0, phi_0=0, seed=1)
    p.update(kw)
    return p


def plus(params, *lines):
    """params (dict) followed by extra "name value" lines, for parameters that appear more than once."""
    return list(params.items()) + [tuple(l.split(None, 1)) for l in lines]


# ── Exact energies of ground states (T = 0 keeps the state: every rotation raises the energy) ──────────────────

@check
def ground_state_ll(r):
    """LL, uniform state: E = 6 bonds x (-1.5 A) = -9 per site, S = 1."""
    d = r.run(base())
    ok_run(r, d)
    _, rows = read_po(d)
    close("E", rows[0]["E"], -9.0, 1e-4)
    close("S", rows[0]["S"], 1.0, 1e-5)
    close("varE", rows[0]["varE"], 0.0, 1e-8)


@check
def ground_state_ghrl_equal_k(r):
    """GHRL with K1 = K2 = K3: U = -P2(n_i.n_j), so the uniform state has E = -6 per site."""
    d = r.run(base(potential="ghrl", k11=1, k22=1, k33=1))
    ok_run(r, d)
    _, rows = read_po(d)
    close("E", rows[0]["E"], -6.0, 1e-4)


@check
def ground_state_pear(r):
    """Pear, uniform polar state: U = -A - B1 per bond (no splay), E = 6 (-A - B1); the n -> -n trial is rejected."""
    d = r.run(base(potential="pear", A=1.5, B1=0.09, C=0.3))
    ok_run(r, d)
    _, rows = read_po(d)
    close("E", rows[0]["E"], 6 * (-1.5 - 0.09), 1e-4)
    close("P", rows[0]["P"], 1.0, 1e-5)


@check
def ground_state_slab_homeotropic(r):
    """Slab 8x8x6 (x, y periodic) along z with homeotropic W = 1 on both plates: each surface site has 5 bonds and
    anchoring -W, so E = (-1.5 (256*6 + 128*5) - 128) / 384."""
    d = r.run(plus(base(Nz=6, geometry="slab"), "anchoring_type 0 homeotropic", "W 0 1",
                   "anchoring_type 1 homeotropic", "W 1 1"))
    ok_run(r, d)
    _, rows = read_po(d)
    close("E", rows[0]["E"], (-1.5 * (256 * 6 + 128 * 5) - 128) / 384, 1e-4)


@check
def ground_state_field(r):
    """Uniform state along z with a dielectric coupling along z: E = -9 - elecA. The direction (0, 0, 2) must be
    normalized, so the result does not depend on its length."""
    d = r.run(base(elecA=0.5, elecX=0, elecY=0, elecZ=2))
    ok_run(r, d)
    _, rows = read_po(d)
    close("E", rows[0]["E"], -9.5, 1e-4)


# ── Observables ────────────────────────────────────────────────────────────────────────────────────────────────

def hedgehog_csv(n, sign=1):
    c = n // 2
    out = ["x,y,z,nx,ny,nz"]
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v = (i - c, j - c, k - c)
                m = math.sqrt(sum(x * x for x in v))
                u = (0, 0, 1) if m == 0 else tuple(sign * x / m for x in v)
                out.append("%d,%d,%d,%.6f,%.6f,%.6f" % ((i, j, k) + u))
    return "\n".join(out) + "\n"


@check
def hedgehog_radial_order(r):
    """Pear droplet 21^3 starting from an outward hedgehog (read with ic_file), one sweep at T = 0:
    Sr_sph and Pr_sph close to 1 and divn close to +0.275 (README reference values)."""
    p = plus(base(Nx=21, Ny=21, Nz=21, geometry="sphere", potential="pear", A=1.5, B1=0.09, C=0.3, ic="ic_file",
                  ic_file="hedgehog.csv"), "anchoring_type 0 homeotropic", "W 0 1.5")
    d = r.run(p, files={"hedgehog.csv": hedgehog_csv(21)})
    ok_run(r, d)
    header, rows = read_po(d)
    for col in ("P", "dtheta", "divn", "Sr_sph", "Pr_sph"):
        if col not in header:
            raise Fail("column %s missing in po.dat header %s" % (col, header))
    close("Sr_sph", rows[0]["Sr_sph"], 1.0, 0.03)
    close("Pr_sph", rows[0]["Pr_sph"], 1.0, 0.03)
    close("divn", rows[0]["divn"], 0.275, 0.02)


@check
def columns_ll_sphere(r):
    """LL in a sphere: po.dat has the radial nematic order Sr_sph but no polar columns."""
    d = r.run(plus(base(Nx=11, Ny=11, Nz=11, geometry="sphere", ic="random"), "anchoring_type 0 homeotropic",
                   "W 0 1"))
    ok_run(r, d)
    header, _ = read_po(d)
    if header != ["T", "S", "varS", "E", "varE", "Sr_sph"]:
        raise Fail("header %s" % header)


# ── Sweeps, file names and modes ───────────────────────────────────────────────────────────────────────────────

def snapshots(d):
    return sorted(f for f in os.listdir(d) if f.startswith("director_field_"))


@check
def thermal_sweep_points(r):
    """Ti = 1, Tf = 0.5, dT = -0.1: six temperatures ending exactly at 0.5, snapshots director_field_<100 T>.csv."""
    d = r.run(base(Nx=4, Ny=4, Nz=4, Ti=1.0, Tf=0.5, dT=-0.1, ic="random"))
    ok_run(r, d)
    _, rows = read_po(d)
    T = [row["T"] for row in rows]
    if len(T) != 6 or any(abs(a - b) > 1e-6 for a, b in zip(T, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])):
        raise Fail("temperatures %s" % T)
    want = sorted("director_field_%d.csv" % v for v in (100, 90, 80, 70, 60, 50))
    if snapshots(d) != want:
        raise Fail("snapshots %s" % snapshots(d))


@check
def thermal_fine_step_names(r):
    """dT = -0.004 (finer than 0.01): snapshot names use 1000 T, so the five temperatures get five files."""
    d = r.run(base(Nx=4, Ny=4, Nz=4, Ti=1.24, Tf=1.224, dT=-0.004, ic="random"))
    ok_run(r, d)
    _, rows = read_po(d)
    if len(rows) != 5:
        raise Fail("%d temperatures instead of 5" % len(rows))
    want = sorted("director_field_%d.csv" % v for v in (1240, 1236, 1232, 1228, 1224))
    if snapshots(d) != want:
        raise Fail("snapshots %s" % snapshots(d))


@check
def thermal_step_001_names(r):
    """dT = -0.01 (read as the float -0.0099999998) still uses 100 T: director_field_120.csv ... _110.csv."""
    d = r.run(base(Nx=4, Ny=4, Nz=4, Ti=1.2, Tf=1.1, dT=-0.01, ic="random"))
    ok_run(r, d)
    want = sorted("director_field_%d.csv" % v for v in range(110, 121))
    if snapshots(d) != want:
        raise Fail("snapshots %s" % snapshots(d))


@check
def electric_sweep(r):
    """Electric mode, E = 0 ... 1.2 in steps of 0.1: 13 points (the last one included), first column `field`."""
    d = r.run(base(Nx=4, Ny=4, Nz=4, evol="electric", Ti=0.5, elecA=0.1, elecZ=1, Ei=0, Ef=1.2, dE=0.1,
                   ic="random"))
    ok_run(r, d)
    header, rows = read_po(d)
    if header[0] != "field":
        raise Fail("first column '%s'" % header[0])
    E = [row["field"] for row in rows]
    if len(E) != 13 or abs(E[-1] - 1.2) > 1e-6:
        raise Fail("field values %s" % E)


@check
def step_blocks(r):
    """Step mode with fn = 3 and first_file_number = 5: blocks 5, 6, 7 in po.dat and in the snapshot names."""
    d = r.run(base(Nx=4, Ny=4, Nz=4, evol="step", Ti=0.5, fn=3, first_file_number=5, MCT=2, ic="random"))
    ok_run(r, d)
    header, rows = read_po(d)
    if header[0] != "ii" or [row["ii"] for row in rows] != [5, 6, 7]:
        raise Fail("header %s, blocks %s" % (header, [row["ii"] for row in rows]))
    if snapshots(d) != ["director_field_5.csv", "director_field_6.csv", "director_field_7.csv"]:
        raise Fail("snapshots %s" % snapshots(d))


@check
def quench_relaxes(r):
    """Quench of LL 8^3 from Ti = 2 to Tf = 0.5: with MCQ sweeps at Tf the system orders (S > 0.8)."""
    d = r.run(base(evol="quench", Ti=2.0, Tf=0.5, MCT=50, MCQ=300, MCS=20, fn=1, ic="random"))
    ok_run(r, d)
    _, rows = read_po(d)
    if not rows[0]["S"] > 0.8:
        raise Fail("S = %g after the quench" % rows[0]["S"])


@check
def snapshot_as_ic_file(r):
    """A snapshot can be read back with ic ic_file: the initial condition of the second run (ic.csv) has the same
    directors as the snapshot of the first one."""
    d1 = r.run(base(Nx=6, Ny=6, Nz=6, Ti=0.3, Tf=0.3, MCT=5, ic="random"))
    ok_run(r, d1)
    snap = os.path.join(d1, "director_field_30.csv")

    def copy(d):
        shutil.copy(snap, os.path.join(d, "start.csv"))
    d2 = r.run(base(Nx=6, Ny=6, Nz=6, ic="ic_file", ic_file="start.csv"), setup=copy)
    ok_run(r, d2)

    def directors(path):
        with open(path) as f:
            next(f)
            return [tuple(round(float(x), 5) for x in l.split(",")[3:6]) for l in f]
    if directors(snap) != directors(os.path.join(d2, "ic.csv")):
        raise Fail("ic.csv differs from the snapshot that was read")


@check
def same_seed_same_run(r):
    """Same input and seed give identical po.dat files (CPU with a fixed number of threads); another seed gives a
    different trajectory."""
    p = base(Nx=6, Ny=6, Nz=6, Ti=0.8, Tf=0.8, MCT=20, MCS=20, ic="random")
    env = {"OMP_NUM_THREADS": "2"}
    a = r.run(p, env=env)
    b = r.run(p, env=env)
    p["seed"] = 2
    c = r.run(p, env=env)
    for x in (a, b, c):
        read_po(x)
    po = [open(os.path.join(x, "po.dat")).read() for x in (a, b, c)]
    if po[0] != po[1]:
        raise Fail("two runs with the same seed differ")
    if po[0] == po[2]:
        raise Fail("runs with seeds 1 and 2 are identical")


# ── Input errors: the run stops, reports the problem and leaves the old output alone ──────────────────────────

def expect_error(r, params, text):
    d = r.run(params, files={"po.dat": "old results\n"})
    if r.last.returncode == 0:
        raise Fail("the run did not stop")
    if text not in r.last.stderr:
        raise Fail("message '%s' not found in: %s" % (text, r.last.stderr.strip()[:300]))
    f = os.path.join(d, "po.dat")
    if not os.path.exists(f) or open(f).read() != "old results\n":
        raise Fail("po.dat of the previous run was changed")


@check
def error_odd_periodic(r):
    """A periodic direction with an odd number of sites is rejected (checkerboard update)."""
    expect_error(r, base(Nx=5), "requires an even number of sites")


@check
def error_wrong_dT(r):
    """dT pointing away from Tf is rejected."""
    expect_error(r, base(Ti=0.5, Tf=1.0, dT=-0.1), "does not lead from Ti")


@check
def error_unknown_parameter(r):
    """A misspelled parameter name is rejected."""
    expect_error(r, plus(base(), "Tff 0.1"), "Unknown parameter")


@check
def error_unknown_anchoring(r):
    """An anchoring type that does not exist is rejected, with the list of valid ones."""
    expect_error(r, plus(base(Nz=6, geometry="slab"), "anchoring_type 0 strong", "W 0 1", "anchoring_type 1 rp",
                         "W 1 1", "theta_s 1 90", "phi_s 1 0"), "not implemented")


@check
def error_missing_anchoring(r):
    """A surface without anchoring_type stops the run with an error status (run scripts must not mark it as done)."""
    expect_error(r, plus(base(Nz=6, geometry="slab"), "anchoring_type 0 homeotropic", "W 0 1"),
                 "anchoring_type 1 is not defined")


@check
def error_ghrl_constants(r):
    """GHRL with k11 > 3 k22 + k33 is rejected: the energy scale nu would be positive (not nematic)."""
    expect_error(r, base(potential="ghrl", k11=5, k22=1, k33=1), "needs k11 < 3 k22 + k33")


@check
def error_negative_temperature(r):
    """Negative temperatures are rejected (T = 0 is allowed)."""
    expect_error(r, base(Ti=-0.1, Tf=-0.1), "Ti must be >= 0")


def main():
    ap = argparse.ArgumentParser(description="Quick checks of McLics against known answers.")
    ap.add_argument("program", nargs="?", default="./mclics", help="mclics program to test (default: ./mclics)")
    ap.add_argument("--gpu", action="store_true", help="run the GPU evolution modes (thermalGPU, ...)")
    ap.add_argument("-k", default="", help="run only the checks whose name contains this text")
    ap.add_argument("--keep", action="store_true", help="keep the run folders")
    args = ap.parse_args()

    program = os.path.abspath(args.program)
    if not os.access(program, os.X_OK):
        sys.exit("not an executable: %s (build it with make or make CPU)" % program)
    work = tempfile.mkdtemp(prefix="mclics_checks_")
    r = Runner(program, args.gpu, work)
    selected = [c for c in CHECKS if args.k in c.__name__]
    print("McLics checks: %s (%s modes), %d checks\n" % (program, "GPU" if args.gpu else "CPU", len(selected)))
    failed = 0
    for c in selected:
        try:
            c(r)
            print("  PASS  %s" % c.__name__)
        except Fail as e:
            failed += 1
            print("  FAIL  %s: %s" % (c.__name__, e))
        except Exception as e:  # a crash of the check itself (missing file, ...)
            failed += 1
            print("  FAIL  %s: %s: %s" % (c.__name__, type(e).__name__, e))
    print("\n%d of %d checks passed." % (len(selected) - failed, len(selected)))
    if args.keep or failed:
        print("Run folders: %s" % work)
    else:
        shutil.rmtree(work)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
