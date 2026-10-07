# Test problems of the McLics paper

Inputs and scripts of the simulations in the "Test Problems" section of the McLics paper, and the scripts that make
its figures. Each campaign has a `run.sh` that writes the input files and runs them, and a `make_figures.py`.

| Folder | Paper section | What it shows |
|---|---|---|
| `freedericksz/` | Elastic Behavior in Nematic Liquid Crystals | Fréedericksz transitions (splay, twist, bend) of the GHRL model with the E7 constants; K1/K2 and K3/K2 from the critical fields |
| this folder (`run.sh`, `make_figures.py`, `sr_table.py`) | Splay-Induced Polarization in Nematic Droplets of Pear-Shaped Molecules | pear potential of Dhakal & Selinger: bulk phase sequence and phase diagram, polar order induced in homeotropic droplets |
| `examples/` | Lebwohl–Lasher Isotropic–Nematic Transition; Cholesteric Pitch and Grandjean Steps; Radial, Axial and Twisted Droplets | T_NI of the LL model, pitch of the chiral GHRL helix on the lattice, Grandjean steps, droplets with homeotropic and chiral planar anchoring |
| `input_example.txt` | Appendix (example of input file) | complete, commented input file: chiral nematic in a hybrid cell with a field |
| `benchmark/` | Appendix (performance) | time per site update of the Monte Carlo step on one CPU thread, on all cores and on the GPU |
| `cylinder/` | Double-Helix Disclinations in a Chiral Cylinder (Custom Geometry) | custom geometry: chiral nematic in a homeotropic cylinder, double helix of +1/2 lines |

**Common rules.** The scripts use the `mclics` found in the PATH and the GPU modes by default (`EVOL=thermal` or
`EVOL=electric` runs on the CPU). Each run goes to its own folder under `runs/`; finished runs get a `DONE` file
and are skipped when the script is run again, so a campaign can be interrupted and restarted. `JOBS` sets how many
runs share the GPU. The number of sweeps is the one used in the paper: shorter runs do not reproduce the results
(see the notes in each folder). The run data are not part of the repository.

## Running the pear campaign

```bash
cd paper
./run.sh               # all stages; or ./run.sh selinger | phase | phasefine | droplets
python3 make_figures.py
```

`run.sh` uses the `mclics` found in the PATH. Runs that have a `DONE` file are skipped, so a campaign can be
interrupted and restarted. Variables: `JOBS` (simultaneous runs), `SEEDS` (default `"1 2 3"`), `DROP_SIZES`,
`LINEAR_N`, `B1_VALUES`.

| Stage | System | What it shows |
|---|---|---|
| `selinger` | bulk 20³, x, y free, z periodic; A = 1.5, B1 = 0.09, C = 0.3; T 1.6 → 0.1, ΔT = 0.025 | validation: I → N → polar sequence (conditions of Fig. 2a of Dhakal & Selinger) |
| `phase` | same bulk, B1 = 0, 0.09, 0.3, 0.6, 1.0, 1.5; T 3.0 → 0.1 | intermediate nematic for weak B1, direct I → polar transition for strong B1 |
| `phasefine` | same bulk on the B grid of the phase diagram of Dhakal & Selinger (B1 = 0.02 … 0.76), T 2.0 → 0.04, ΔT = 0.02 | direct comparison with their phase diagram (overlaid on panel b of the bulk figure) |
| `droplets` | homeotropic droplets (W = 1.5) N = 21, 31, 41 with C = +0.3, −0.3, 0; N = 31 with C = 0.1, 0.2; T 1.4 → 0.1 | radial polarization induced by the splay of the hedgehog, symmetry in C, linearity, p(r) = κ/r |

With 3 seeds: 3 (selinger) + 18 (phase) + 48 (phasefine) + 33 (droplets) = 102 runs, all of size ≤ 41³. If
`runs/phasefine` has finished runs, the figure uses that stage instead of `phase`.

## Output of the pear campaign

- `figs/fig_pear_bulk.pdf` — (a) S and P versus T; (b) transition temperatures versus B.
- `figs/fig_pear_droplet.pdf` — (a, b) slices of the hedgehog for C = ±0.3; (c) P_r versus T; (d) P_r / C.
- `figs/fig_pear_profile.pdf` — (a) p(r) for three radii; (b) r·p(r).
- `figures.tex` — `figure*` environments with the captions; the numbers in the captions come from the data.
- `figs/numbers.txt` — the numbers used in the captions, for checking.

Options of `make_figures.py`: `--profile-T` (0.75), `--slice-T` (0.1), `--slice-N` (41), `--curve-N` (41),
`--linear-N` (31).

`sr_table.py` writes the table of radial order S_r of the droplets (`figs/sr_table.txt` and `.tex`); the
Lebwohl–Lasher droplets used for comparison are given with `--ll`.

## Notes on the pear campaign

- Panel (b) of the bulk figure: the open circles come from `selinger_phase_diagram_digitized.csv`, digitized by
  eye from the phase diagram of Dhakal & Selinger (accuracy ≈ 0.02 in T). Redo it with WebPlotDigitizer on the original PDF
  before publication. The caption gives the largest deviation between the two diagrams.
- The B of Dhakal & Selinger is the `B1` of the code.
- For small B1 (0.02, 0.04) the polar phase is strongly deformed by splay and the global P stays small, so
  `make_figures.py` takes as the onset of the polar phase the first T with P > 0.1, and says so in the caption.
