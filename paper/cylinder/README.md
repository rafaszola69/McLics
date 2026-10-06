# Chiral nematic in a homeotropic cylinder (custom geometry)

Test problem after Wand & Bates, PRE 100, 052702 (2019): two s = +1/2 disclination lines form a double helix whose
repeat distance λ they found to be ≈ 1.45 p. Chiral GHRL with E7 constants (k11 1.48, k22 1, k33 2.28), T = 0.1, `homeotropic_ghrl` anchoring with W = 5.
With plain `homeotropic` W = 1, cooling gives a planar cholesteric across the cylinder and no disclinations.
From W = 3 on, both anchorings give the double helix.

## Files
- `prepare.py` — geometry file (same rules as `c_cilinder.cpp`) and the double-helix initial condition.
- `run.sh` — stages:
  - `scan`: spontaneous selection. Cylinders R = 10, L = 24…144 (z periodic), cooled from T = 1.4 to 0.1, 3 seeds.
  - `bates`: idealized double helices with ξ = −0.5…3 turns over L = 48, held at T = 0.1.
  - `bulkp`: bulk pitch at T = 0.1 (6×6×96 box).
- `make_figures.py` → `figs/fig_cylinder.pdf/png` and `figs/numbers.txt`.

## Running (GPU machine)
```bash
JOBS=4 ./run.sh scan
JOBS=4 ./run.sh bates
./run.sh bulkp
python3 make_figures.py
```
Override with environment variables: `P0`, `RADII`, `LS`, `SEEDS`, `T`, `MCT_SCAN`, `MCS_SCAN`, `ANCH`, `W`, `EVOL`.
Only the final (lowest-T) snapshot is kept, gzipped. Finished runs are marked `DONE` and skipped on rerun.

## Analysis
- The +1/2 lines in each layer are found from the winding of the director around each plaquette.
  The pair orientation is the second moment of their positions; layers without defects fall back to the low-S sites. Its total winding over L gives
  the twist ξ (half turns are allowed).
- (a) ξ vs L: all seeds (open) and the lowest-energy seed (filled). A staircase ξ = round(2L/λ)/2 is fitted → λ.
- (b) E(ξ) at L = 48, using only runs whose twist stayed within 0.15 turns of the imposed one.
- (c) λ vs bulk pitch p, compared with λ = p and λ = 1.45 p.

## Later stages and paper figure
- `./run.sh stretch`: a relaxed one-turn helix from the scan, stretched along z (`prepare.py stretch`) to λ up to
  4.5 p and relaxed at T = 0.1. `KSET=eq` / `KSET=wb` run the same stages with equal K or the K ratios of Wand & Bates
  (folders `runs_eqK/`, `runs_wbK/`); the paper uses E7 (`runs/`) and the equal-K stretched branch (`runs_eqK/`).
- `energy_terms.py` → `figs_terms/`: Frank decomposition (splay, twist, bend) of the stretched states.
- `make_paper_figure.py` → `figs/fig_cylinder_paper.pdf`. Panels (a) and (b) are 3D renders of the disclination
  lines made outside McLics from the snapshots `runs/scan/p25/R10/L72/seed2` and `runs/stretch/p25/R10/L110/seed1`;
  the script reads them from `figs/render_helix2/` and `figs/render_streched/`.
