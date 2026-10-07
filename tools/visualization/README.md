# Visualization of director fields (ParaView)

Scripts that turn the `director_field_*.csv` snapshots written by McLics into images. They need
[ParaView](https://www.paraview.org) (its batch interpreter `pvbatch`) and Python 3; they were tested with
ParaView 5.6 and 5.10. `pvbatch` is taken from the variable `PVBATCH` or from the `PATH`:

```bash
export PVBATCH=/path/to/ParaView/bin/pvbatch     # if pvbatch is not in the PATH
```

Both scripts are run in the folder of the snapshots, with the number of the first snapshot and, optionally, a step
and the last number (`render_slices.sh 10 10 120` renders `director_field_10.csv`, `_20`, ..., `_120`).

## Director field in the central planes — `render_slices.sh`

```bash
render_slices.sh 50          # director_field_50.csv -> x_, y_, z_director_field_50.png
render_slices.sh -C 50       # colored by the director component normal to each plane
```

Each image shows one of the planes x = Nx/2, y = Ny/2 and z = Nz/2, seen along its normal, with one cylinder per
site of the sample along the director (for large planes, a uniform subset of about 4000 sites). By default the
cylinders are green, and red where the local order parameter is S < 0.438, which marks the defects. With `-C` they
are colored by |n·e|, where e is the normal of the plane: red for directors in the plane and blue for directors
along the normal, which shows the twist of cholesterics.

## Defect lines in 3D — `render_defects.sh`

```bash
render_defects.sh free 50    # director_field_50.csv -> iso_50.png, iso_x_50.png, iso_y_50.png, iso_z_50.png
render_defects.sh sphere 50
```

The defects are drawn as the volume where the local order parameter is below 0.78 of its largest value, inside a
translucent shell that shows the sample: `free` (the sites of the sample themselves, for any geometry), `sphere`,
`hemisphere`, `slab` (the box) or `cylinder` (axis along z). `iso_<i>.png` is an oblique view, and the other three
images are views along x, y and z.

## Files

- `render_slices.sh`, `render_defects.sh` — the commands above.
- `slice.py`, `defects.py` — the ParaView scripts they run (`pvbatch slice.py ...`; see the header of each file).
- `extract_slices.py` — writes the three central planes of a snapshot as separate CSV files (Python 3 standard
  library), used by `render_slices.sh`.
