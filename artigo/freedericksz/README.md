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
cd artigo/freedericksz
./run.sh            # the three cells, 2 seeds each, on the GPU (EVOL=electric for the CPU)
python3 make_figures.py
```

Defaults: T = 0.1, W = 10 on both plates, field 0.1 → 1.2 in steps of 0.005, 40000 sweeps at each field value
before 1000 measurement sweeps. All can be changed through variables (see the header of `run.sh`).

**Do not shorten the sweep.** Near the threshold the deformation grows more and more slowly, and the slower the
smaller the elastic constant. With fewer sweeps per field value or larger field steps the deformation appears late,
the thresholds are overestimated (the twist one most) and the ratios of elastic constants come out wrong.

## Output

- `figs/fig_freedericksz.pdf` — mid-cell deformation angle versus field for the three cells (band: range over the
  seeds; dashed lines: critical fields).
- `figs/numbers.txt` — critical fields (with the error of the mean over the seeds) and the ratios K1/K2 and K3/K2.
- `figs/freedericksz_angles.csv` — the curves of the figure.

The deformation angle is the mean of acos(|n · n0|) over the two middle layers, with n0 the initial easy axis. The
critical field is where the straight line fitted to θ² versus E (15° < θ < 40°) reaches the thermal baseline of θ²
below the threshold; the field where θ crosses 20° is also reported.
