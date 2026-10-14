# Freedericksz transition (elastic constants from critical fields)

Test problem "Elastic Behavior in Nematic Liquid Crystals" of the McLics paper, after Le Masurier et al.: the GHRL
potential with the E7 elastic constants (K1/K2 = 1.48, K3/K2 = 2.28) in a 10 × 10 × 16 slab, with the field swept
in the electric mode. Three cells select one deformation each at the threshold:

| Cell | Anchoring | Field | Dielectric anisotropy | Deformation |
|---|---|---|---|---|
| `splay` | planar along x (`rp`) | z | +0.25 | splay |
| `twist` | planar along x (`rp`) | y | +0.25 | twist |
| `bend` | homeotropic | z | −0.25 | bend |

Since E_c² ∝ K_i, the critical fields give K1/K2 = (E_c splay / E_c twist)² and K3/K2 = (E_c bend / E_c twist)².

## Running

```bash
cd paper/freedericksz
./run.sh            # the three cells, 2 seeds each, on the GPU (EVOL=electric for the CPU)
python3 make_figures.py
python3 make_cells.py   # cross-sections of the three cells (needs figs/numbers.txt from make_figures.py)
```

Defaults: T = 0.0008 (as in Le Masurier et al.), W = 10 on both plates, field 0.1 → 1.2 in steps of 0.005, 40000 sweeps at each field value
before 1000 measurement sweeps. All can be changed through variables (see the header of `run.sh`).

**Do not shorten the sweep.** Near the threshold the deformation grows more and more slowly, and the slower the
smaller the elastic constant. With fewer sweeps per field value or larger field steps the deformation appears late,
the thresholds are overestimated (the twist one most) and the ratios of elastic constants come out wrong.

## Output

- `figs/fig_freedericksz.pdf` — mid-cell deformation angle versus field for the three cells (band: range over the
  seeds; dashed lines: critical fields).
- `figs/fig_freedericksz_cells.pdf` — central x-z cross-sections of the three cells at E = 1.3 E_c (`--ratio`),
  director segments coloured by the deformation angle θ = arccos|n·n0|, in the style of the droplet sections of
  `paper/examples`.
- `figs/numbers.txt` — critical fields (with the error of the mean over the seeds), the ratios K1/K2 and K3/K2, and
  the critical fields of the continuum theory for the same cell, E_c = π/(d + 2b)·√(K*/2Δε) with K* = Λ*K_i/K_2,
  b = K*/(2W) and d = Nz − 1 (no free parameter).
- `figs/freedericksz_angles.csv` — the curves of the figure.

The deformation angle is the mean of acos(|n · n0|) over the two middle layers, with n0 the initial easy axis. The
critical field is where the straight line fitted to θ² versus E (from 3 times the thermal baseline angle, at least
2°, up to 40°) reaches the baseline of θ² below the threshold; the field where θ crosses 20° is also reported.
