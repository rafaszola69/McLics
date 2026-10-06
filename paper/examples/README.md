# Further test problems of the McLics paper

```bash
cd paper/examples
./run.sh            # or by stage: twist | twistT | cano | llbulk | llheat | homeo | chiral
python3 make_figures.py
```

Uses the `mclics` found in the PATH. Runs on the GPU (`evol thermalGPU`); `EVOL=thermal ./run.sh` runs on the CPU.
Runs that have a `DONE` file are skipped. Variables: `JOBS` (default 2, simultaneous runs on the same GPU),
`SEEDS`, `TWIST_P0`, `CANO_NZ`, `BIP_ANCH`, `LL_SIZES`.

| Stage | System | What it shows | Runs |
|---|---|---|---|
| `twist` | chiral GHRL (E7), 6×6×48, x, y periodic, z free; initial helix p_i = 0.8, 1.0, 1.25 p0; p0 = 8 … 40; T = 0.3 | the pitch is 2π/arcsin(2π/p0) on the lattice, not p0 | 42 |
| `twistT` | same, p0 = 8 and 10, T = 0.05 … 0.5 | pitch versus T: approach to the T = 0 prediction | 20 |
| `cano` | chiral GHRL, p0 = 20, slab 12×12×Nz, `rp` along x on both plates, Nz = 6 … 46 | Grandjean steps | 42 |
| `llbulk` | LL, periodic bulk L = 10 … 40, fine cooling around T_NI | T_NI(L) from the maximum of C_V | 15 |
| `llheat` | same, heating from the aligned state | T_NI from both sides together with the cooling runs | 15 |
| `homeo` | LL, droplet N = 15, 21, 31, 41, homeotropic W = 0.03 … 1 (fine grid in 0.1–0.5) | uniform/axial → radial; collapse of S_r as a function of W·R | 104 |
| `chiral` | chiral GHRL (E7), droplet N = 41, FG W = 1, 15 values of R/p0 between −0.3 and 0.3 (`CHIRAL_X`); 4 seeds | bipolar → twisted bipolar, handedness given by the sign of p0 | 60 |

Suggested `JOBS` on an RTX 4070: `twist`, `twistT`, `cano`: 8; `homeo`, `chiral`: 4; `llbulk`, `llheat`: 2.

## Output

- `figs/fig_chol.pdf` — (a) measured pitch versus p0; (b) total twist versus d/p (Grandjean steps).
- `figs/fig_homeo.pdf` — S_r versus W and versus W·R; slices for weak and strong W.
- `figs/fig_chiral.pdf` — slices perpendicular to the bipolar axis for both handednesses; twist angle versus R/p0.
- `figs/fig_llbulk.pdf` — (a) S versus T; (b) C_V versus T (cooling and heating); (c) T_NI(L) versus 1/L³.
- `figures.tex` — captions with the numbers taken from the data; `figs/numbers.txt` for checking.

## Notes

- The pitch of the chiral GHRL helix on the lattice is 2π/arcsin(2π/p0) (e.g. p0 = 10 → 9.25); the difference from
  p0 decreases as 1/p0².
- The chiral droplets use only |R/p0| ≤ 0.3: with R/p0 = 0.75 (p0 ≈ 13) the droplet forms square cholesteric layers
  (a lattice artefact).
- C_V per site = L³·varE/(4T²), because the E of `po.dat` counts each bond twice.
