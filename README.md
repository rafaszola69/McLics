<div align="center">

# McLics
### Monte Carlo Liquid Crystal Simulator

*Monte Carlo simulator for nematic and polar liquid crystals on a cubic lattice, written in C++/CUDA.*

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11%2B-76B900?logo=nvidia&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-parallel-blue)
![GSL](https://img.shields.io/badge/GSL-2.x-orange)
![License: GPL v3+](https://img.shields.io/badge/license-GPL%20v3%2B-blue)

**Author:** Eric Kamakawa &nbsp;·&nbsp; **Advisors:** Prof. Dr. Rafael Zola, Prof. Dr. Rodolfo Teixeira

**Computer Engineering — UTFPR** &nbsp;·&nbsp; full list of contributors in [AUTHORS](AUTHORS)

</div>

---

## Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Building](#building)
- [Usage](#usage)
- [Input file](#input-file)
- [Parameter reference](#parameter-reference)
- [Geometries](#geometries)
- [Evolution modes](#evolution-modes)
- [Potentials](#potentials)
- [Boundary conditions](#boundary-conditions)
- [Surface anchoring](#surface-anchoring)
- [Electric field](#electric-field)
- [Output](#output)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Notes](#notes)
- [License](#license)

---

## Overview

**McLics** simulates nematic (and polar) liquid crystals on a three-dimensional cubic lattice with the Metropolis
Monte Carlo method. Each lattice site carries a unit vector (the molecular director), which interacts with its six
nearest neighbours through a pair potential, with surfaces through an anchoring energy and, optionally, with an
applied field. Each Monte Carlo sweep proposes one random rotation per site and accepts or rejects it with the
Metropolis rule.

At each temperature (or field value, or block, depending on the mode) the program writes the **scalar order
parameter `S`**, the **mean energy `E`**, their variances and optional extra observables to `po.dat`, and a full
snapshot of the director field.

The computational core is written in **C++/CUDA** and can be built in two flavours:

| Build | Command | Needs nvcc? | Runs the `*GPU` modes? |
|---|---|:---:|:---:|
| Full (CPU + GPU) | `make` | Yes | Yes |
| CPU only | `make CPU` | No | No |

In both cases the program is `./mclics`; the CPU code is parallelized with OpenMP.

---

## Dependencies

| Dependency | Minimum version | Needed for |
|---|---|---|
| GCC | 9+ | Building (CPU and GPU) |
| NVCC (CUDA Toolkit) | 11+ | GPU support (`make`) |
| OpenMP | — | CPU parallelism |
| GSL (GNU Scientific Library) | 2.x | Random numbers and linear algebra (eigenvalues of the Q tensor) |

On Ubuntu/Debian:

```bash
sudo apt install g++ libgsl-dev libomp-dev
# For GPU support, install the CUDA Toolkit that matches your driver:
# https://developer.nvidia.com/cuda-downloads
```

---

## Building

```bash
make            # full program (CPU + GPU), needs nvcc; without nvcc, builds the CPU version
make CPU        # CPU-only program, no CUDA Toolkit needed
make clean      # remove the program and the build files
make clean-data # remove simulation output in this folder (ic.csv, po.dat, director_field_*.csv)
```

Each version is built in `build/gpu/` or `build/cpu/` and copied to `./mclics`, so `./mclics` is always the
version built last.

> The program built with `make` **also runs CPU simulations**: the device used by a run is chosen by the `evol`
> parameter of the input file (see [Evolution modes](#evolution-modes)), not by a command-line option.

---

## Usage

```bash
./mclics <input_file>
```

The output files are written to the current directory. If no input file is given, the program runs with the
built-in defaults (see the tables below).

All parameters are checked before the run starts: impossible values (negative sizes or temperatures, a temperature
step pointing away from `Tf`, invalid elastic constants, an odd number of sites along a periodic direction, ...)
are reported together and the program stops without touching the output of a previous run.

---

## Input file

The input file is a text file with one parameter per line, in the form `name value`. Everything after `#` is a
comment. Parameter names are **not case-sensitive**. Parameters that belong to a surface take the surface number
first (`W 0 1.5`).

Complete example:

```
# Lattice
Nx          30
Ny          30
Nz          30

# Geometry: bulk | slab | sphere | custom
geometry    bulk

# Boundary conditions: free | periodic
xbound      periodic
ybound      periodic
zbound      periodic

# Potential: ll | ghrl | pear
potential   ll
A           1.0

# Evolution mode: thermal | step | quench | electric (or with the GPU suffix)
evol        thermal

# Initial temperature, final temperature and step
Ti          1.2
Tf          0.1
dT         -0.02

# Monte Carlo sweeps for thermalization and measurement at each temperature
MCT         500
MCS         200

# Initial condition: random | homogeneous | ic_file | cholesteric
ic          random
seed        1
```

---

## Parameter reference

#### Lattice

| Parameter | Values | Default | Description |
|---|---|---|---|
| `Nx`, `Ny`, `Nz` | integer | 16 | Lattice size along each axis |
| `Nk` (`neighbour_kind`) | `1` | 1 | Neighbour shell. **Only `1` (nearest neighbours) is supported**; any other value stops the program |
| `geometry` | `bulk` / `slab` / `sphere` / `custom` | `bulk` | Geometry of the sample |
| `boundary_file` | path | — | Geometry file (`geometry custom`) |
| `xbound`, `ybound`, `zbound` | `free` / `periodic` | `free` | Boundary condition along each axis |

#### Potential and elastic constants

| Parameter | Values | Default | Description |
|---|---|---|---|
| `potential` | `ll` / `ghrl` / `pear` | `ll` | Pair potential |
| `A` | float | 1.0 | Nematic coupling (LL and pear) |
| `B1` | float | 0.04 | Polar coupling `B` of the pear potential |
| `C` | float | 0.3 | Splay–polarization coupling of the pear potential |
| `k11`, `k22`, `k33` | float | 1.0 | Frank elastic constants (splay, twist, bend) of the GHRL potential; only their ratios matter |
| `p0` | float | 0 | Natural cholesteric pitch in lattice spacings (GHRL; `0` = nematic) |

#### Evolution and Monte Carlo

| Parameter | Aliases | Default | Description |
|---|---|---|---|
| `evol` | `mode` | `thermal` | Evolution mode: `thermal`, `step`, `quench`, `electric`, or the same with the `GPU` suffix (e.g. `thermalGPU`) to run on the graphics card |
| `Ti`, `Tf`, `dT` | — | 1.2 / 0.0 / -0.02 | Initial temperature, final temperature and step. `T = 0` is allowed (zero-temperature Metropolis) |
| `MCT` | — | 100 | Thermalization sweeps at each temperature (or field value, or block) |
| `MCS` | — | 100 | Measurement sweeps at each temperature (one sample per sweep) |
| `MCQ` | `quench_steps` | `MCT` | `quench` mode only: sweeps at `Tf` before measuring |
| `fn` | — | 1 | Number of blocks (`step` and `quench` modes) |
| `first_file_number` | `initial_output` | 0 | Number of the first block (`step` and `quench` modes) |
| `seed` | — | 1 | Seed of the random number generators |
| `radial` | — | `sphere` if `geometry sphere`, otherwise `none` | Radial columns in `po.dat`: `sphere`, `cylinder`, `both` or `none`. See [`po.dat`](#podat) |

#### Initial condition

| Parameter | Values | Default | Description |
|---|---|---|---|
| `ic` | `random` / `homogeneous` / `ic_file` / `cholesteric` | `random` | Type of initial condition |
| `ic_file` | path | — | CSV file with the initial director field (`ic ic_file`); a snapshot `director_field_*.csv` can be used directly |
| `theta_0`, `phi_0` | float (degrees) | 0 | Polar and azimuthal angles of the director (`homogeneous`), or polar angle and initial azimuth of the helix (`cholesteric`) |
| `p0_i` | float | 0 | Pitch of the `cholesteric` initial condition (helix along z); `0` uses `p0` |

#### Surface anchoring (for each surface number `n`)

| Parameter | Description |
|---|---|
| `anchoring_type n <type>` | Anchoring type of surface `n` |
| `W n <value>` | Anchoring strength of surface `n` |
| `theta_s n <degrees>` | Polar angle of the easy axis of surface `n` (lab frame; `0` = z axis) |
| `phi_s n <degrees>` | Azimuthal angle of the easy axis of surface `n` |

Types: `homeotropic`, `rp` (Rapini–Papoular), `rp_tangent` (planar Rapini–Papoular for curved surfaces), `fg`
(Fournier–Galatola-like, degenerate planar), and `homeotropic_ghrl`, `rp_ghrl`, `fg_ghrl` for the GHRL potential.
See [Surface anchoring](#surface-anchoring).

#### Electric field

| Parameter | Aliases | Description |
|---|---|---|
| `elecX`, `elecY`, `elecZ` | `Electric_field_x/y/z`, `EF_x/y/z` | Field direction (any length; normalized by the program) |
| `elecA` | `dielectric_anisotropy`, `Aniso_E` | Dielectric anisotropy (quadratic coupling) |
| `elecP` | `dipolar_coupling`, `Polar_E` | Dipolar coupling (linear, polar) |
| `Ei`, `Ef`, `dE` | `initial_E`, `final_E`, `E_variation` | Initial and final field magnitude and step (`electric` mode) |

---

## Geometries

Every lattice site has a point type `pt`: `0` empty (outside the sample), `1` bulk, `n ≥ 2` a site of surface
`n − 2`, whose anchoring is set by `anchoring_type n−2`. Surface sites also have an outward normal.

**`bulk`** — no surfaces; boundaries along x, y and z from the input.

**`slab`** — two flat surfaces: surface 0 at `z = 0` (normal −z) and surface 1 at `z = Nz − 1` (normal +z). A film
confined between two substrates; z is free, x and y follow the input.

**`sphere`** — droplet centred at `(Nx/2, Ny/2, Nz/2)`: bulk for `r < Nz/2 − 1`, one layer of surface sites
(surface 0, radial normals) for `r < Nz/2 + 1`, empty outside. All boundaries are free.

**`custom`** — geometry read from a CSV file:

```
geometry       custom
boundary_file  my_geometry.csv
```

The file has a header line naming the columns and one line per grid point. The columns `x`, `y`, `z` (integer grid
indices), `nx`, `ny`, `nz` (outward unit normal, used on surface points) and `pt` are required, in any order; other
columns are ignored. Points that are not listed are bulk. Errors (missing columns, indices outside the grid,
repeated points, zero normals on surface points) are reported with the line number.

---

## Evolution modes

| Mode | Behaviour |
|---|---|
| `thermal` | Sweeps the temperature from `Ti` to `Tf` in steps of `dT`. At each temperature: `MCT` thermalization sweeps, then `MCS` measurement sweeps; one line of `po.dat` and one snapshot per temperature. Each temperature starts from the final state of the previous one. |
| `step` | Evolution at the fixed temperature `Ti` in `fn` blocks of `MCT` + `MCS` sweeps; one line of `po.dat` and one snapshot per block. Useful to follow the relaxation and to check convergence. |
| `quench` | `fn` blocks, each with `MCT` sweeps at `Ti`, a jump to `Tf`, `MCQ` sweeps at `Tf` and then `MCS` measurement sweeps. `dT` is not used. |
| `electric` | Sweeps the field magnitude from `Ei` to `Ef` in steps of `dE` at the fixed temperature `Ti`. |

Each of the four modes accepts the `GPU` suffix (e.g. `thermalGPU`) to run on a CUDA device, if the program was
built with `make` (not `make CPU`).

Sweep points are computed as `Ti + n·dT` (or `Ei + n·dE`), and the last point is exactly `Tf` (`Ef`) when `Tf` lies
on that grid, so rounding never skips the last temperature.

---

## Potentials

**Lebwohl–Lasher (`ll`)** — the classic nematic pair potential, `U_ij = −1.5·A·(n_i·n_j)²`, i.e. `−A·P2(n_i·n_j)`
up to a constant. With `A = 1` the nematic–isotropic transition of the bulk is at `T_NI ≈ 1.1232`.

**Gruhn–Hess / Romano (`ghrl`)** — pair potential whose continuum limit has the Frank elastic constants `k11`,
`k22`, `k33` and, with `p0 ≠ 0`, a cholesteric pitch. It allows real elastic anisotropy (e.g. E7:
`k11 1.48`, `k22 1`, `k33 2.28`). Its coefficients are scaled so that the temperature unit is the same as in the LL
model. On the lattice, the twist between neighbouring layers is `arcsin(2π/p0)`, so the pitch of the relaxed helix is
`2π/arcsin(2π/p0)` (e.g. `p0 = 10` gives 9.25).

**Pear (`pear`)** — pear-shaped molecules with splay flexoelectricity (Dhakal & Selinger, PRE 81, 031704, 2010,
Eq. 6):
`U_ij = −A(n_i·n_j)² − B1(n_i·n_j) − C((1+n_i·n_j)/2)² r̂_ij·(n_j−n_i)`, with `r̂_ij` pointing from i to j.
Here `n` is a polar vector. With this potential `po.dat` gets the columns `P` (Eq. 8: `|⟨d·n_i⟩|`, with `d` the
director), `dtheta` (magnitude of the splay vector of Eq. 3) and `divn` (signed splay), plus `Pr_sph`/`Pr_cyl`
according to `radial` (see [`po.dat`](#podat)).

With `pear` (or with `elecP ≠ 0`) each Monte Carlo step automatically includes, besides the small rotation, a
Metropolis trial of the inversion `n → −n`. Without it the nematic→polar transition (Fig. 2a of the paper) is
frozen: inverting a molecule by small rotations requires passing through the perpendicular orientation, with a
barrier ≈ 6A. The term `−A(n_i·n_j)²` is invariant under the inversion, so its acceptance depends only on `B1`,
`C` and `elecP`. The flips are not counted in the acceptance rate used to adapt the rotation amplitude.

To reproduce the paper, use `xbound free`, `ybound free`, `zbound periodic`: with periodic boundaries along the
splay direction a uniform splay is not possible and `dtheta` tends to zero.

---

## Boundary conditions

| Value | Description |
|---|---|
| `free` | No neighbours beyond the edge of the lattice |
| `periodic` | The lattice repeats along that axis |

Each axis has its own condition. **A periodic axis needs an even number of sites**: the lattice is updated in 8
sublattices by the parity of `(i, j, k)`, and with an odd size the first and last sites, which are neighbours across
the periodic boundary, would have the same parity and be updated at the same time. The program stops with an error
in that case. The `sphere` geometry makes all axes free, and `slab` makes z free.

---

## Surface anchoring

Sets how the directors at a surface align. Each surface has a number `n`, and its anchoring energy is added once
to the energy of each of its sites (`s` is the outward normal of the site):

| Type | Energy | Easy axis |
|---|---|---|
| `homeotropic` | `−W (n·s)²` | local normal |
| `rp` | `−W (n·e)²` | fixed lab axis `e(theta_s, phi_s)` |
| `rp_tangent` | `−W (n·e)²` | lab axis projected on the local tangent plane (below) |
| `fg` | `+W (n·s)²` | any direction in the tangent plane (degenerate planar) |
| `homeotropic_ghrl`, `rp_ghrl`, `fg_ghrl` | `W·U_GHRL(n, e, s)` | the surface acts as a phantom GHRL neighbour along `s` with director `e` = normal, lab axis, or projection of `n` on the tangent plane |

Example — slab with homeotropic anchoring on both faces:

```
anchoring_type  0  homeotropic
W               0  0.5
anchoring_type  1  homeotropic
W               1  0.5
```

Example — tilted easy axis on surface 0:

```
anchoring_type  0  rp
W               0  0.3
theta_s         0  60.0
phi_s           0  45.0
```

### `rp_tangent` — planar Rapini–Papoular on curved surfaces

`rp` uses an easy axis fixed in the lab frame, which only makes sense on flat surfaces. `rp_tangent` projects the
reference direction `d(theta_s, phi_s)` (same angle convention as `rp`) on the local tangent plane of each site:

```
e = (d − (d·s) s) / |d − (d·s) s|        U = −W (n·e)²
```

- On a flat surface with `d` in the plane (e.g. `theta_s 90`) it is identical to `rp`, and the two faces of a slab
  agree.
- On a sphere, `d = ẑ` (`theta_s 0`) gives an easy axis along the meridians (bipolar texture, poles at ±z).
- Where the normal is parallel to `d` the projection vanishes (unavoidable on a closed surface, by the hairy-ball
  theorem); at these few sites the anchoring energy is zero. The GPU version reports how many sites are affected.

```
geometry        sphere
anchoring_type  0  rp_tangent
W               0  4
theta_s         0  0
phi_s           0  0
```

---

## Electric field

The field adds a term to the energy of each site:

```
U_field = −elecP · (n·E)  −  elecA · (n·E)²,     E = elecE · (elecX, elecY, elecZ)
```

- **`elecA` — quadratic (dielectric):** invariant under `n → −n`; aligns the director with the field without
  choosing a sense. This is the usual coupling for nematics.
- **`elecP` — linear (dipolar):** distinguishes `n` from `−n` and induces polar order. It is the `−E·n_i` term of
  the pear potential (Dhakal & Selinger, Eq. 6): the paper's `E` is `elecP · elecE`.

Both can be used together; the term is computed when either is nonzero. The direction `(elecX, elecY, elecZ)` is
normalized by the program (a zero direction with a nonzero coupling is an error). The magnitude `elecE` is 1, except
in the `electric` mode, where it is swept from `Ei` to `Ef` in steps of `dE`.

Example (Fig. 2d of the paper, `E = 0.06` along z):

```
potential  pear
elecZ      1
elecP      0.06
```

---

## Output

Each run **removes the output of a previous run** in the same folder (`ic.csv`, `po.dat`, `director_field_*.csv`)
before starting, so old and new results are never mixed. This happens only after the whole setup has succeeded (an
input error leaves the old files untouched), and after the initial condition has been read, so a snapshot of the
previous run can be used as `ic_file`.

### `po.dat`

```
T       S        varS      E        varE
0.80    0.712    0.00031   -1.234   0.00018
```

The first column is the temperature (`T`), the field magnitude (`field`, in the `electric` mode) or the block number
(`ii`, in the `step` and `quench` modes). `S` is the global order parameter (largest eigenvalue of the Q tensor),
`E` the mean energy per non-empty site, and `varS`, `varE` their variances over the `MCS` samples. In `E`, the energy
of a site includes its bonds with its neighbours, its anchoring and the field, so each bond is counted twice while
anchoring and field are counted once.

Extra columns are appended in this order, depending on the potential and on `radial`:

```
T S varS E varE  [P dtheta divn]  [Sr_sph Pr_sph]  [Sr_cyl Pr_cyl]
                  pear only        radial sphere    radial cylinder
                                   or both          or both
```

The `Pr_*` columns only appear with `pear`; the `Sr_*` columns appear with any potential. Examples: `ll` in a
sphere → `T S varS E varE Sr_sph`; `pear` in a sphere → `... P dtheta divn Sr_sph Pr_sph`; `pear` in bulk without
`radial` → `... P dtheta divn`. All are time averages over the `MCS` measurement sweeps.

| Column | What it measures | When to use it |
|---|---|---|
| `P` | Global polarization `\|⟨d·n_i⟩\|` (Eq. 8 of the paper), with `d` the global director (eigenvector of Q). From 0 to 1. | **Uniform** polarization (bulk, slab, HAN cell). |
| `dtheta` | Magnitude of the mean splay vector ⟨Δθ⟩ (Eq. 3). A **vector** average, so splays in opposite directions cancel. | **Uniform** splay, as in the paper. |
| `divn` | Signed mean splay `⟨∇·n⟩`, in units of 1/(lattice spacing). Central difference in the interior, one-sided at surfaces. Reference: an outward hedgehog in a sphere of radius R gives ≈ +3/R, an inward one ≈ −3/R. Exactly 0 in a fully periodic box. Does not depend on a centre. | Any geometry; shows the sign of the splay. |
| `Sr_sph` | Radial nematic order `⟨(3c²−1)/2⟩`, `c = n_i·r̂_i`, with `r̂` the direction from the grid centre `(Nx/2, Ny/2, Nz/2)`, the same centre as the `sphere` geometry. Independent of the sign of n: 1 = radial, 0 = unrelated to the radial direction, −1/2 = tangential. | Compare the **structure** of `ll` and `pear` in droplets and spherical shells. |
| `Pr_sph` | Radial polarization `⟨n_i·r̂_i⟩`, same `r̂`. **Signed**: +1 = outward polar hedgehog, −1 = inward, ≈0 = no radial polar order. | Tell whether the radial structure is **polar** (`pear` only). |
| `Sr_cyl`, `Pr_cyl` | Same as above, with `r̂` in the xy plane from the z axis through the grid centre (z component ignored). | Cylinders and cylindrical shells along z. |

On comparing `ll` and `pear`: in `ll`, n and −n are the same state, so polarization and signed splay are
meaningless there. That is why `P`, `dtheta`, `divn` and `Pr_*` exist only for `pear`, while `Sr_*` serves both.
`Sr` answers "is the structure radial?"; `Pr` answers "besides radial, is it polar?".

`P` and `dtheta` **do not see** radial configurations: in a perfect polar hedgehog both give ≈0, because the
projections and the splay vectors cancel around the centre.

The radial quantities only make sense when the simulated region is centred on the grid (for `custom`, generate a
centred geometry file). Using the wrong one underestimates the order: a perfect cylindrical radial field gives
`Sr_cyl = 1`, `Pr_cyl = 1`, but `Sr_sph = 0.50`, `Pr_sph = 0.79`.

Reference values (sphere 21³, R ≈ 10, `pear`): outward hedgehog → `Sr_sph = 1`, `Pr_sph = 1`, `divn = 0.275`;
inward → `Pr_sph = −0.99`, `divn = −0.275`; uniform along z → `Sr_sph ≈ 0`, `Pr_sph ≈ 0`, `Sr_cyl = −0.5`.

### `ic.csv`

Snapshot of the initial condition, before the evolution starts (same format as the snapshots below).

### `director_field_<id>.csv`

```
x,y,z,nx,ny,nz,S,pt
0,0,0,0.123,-0.456,0.781,0.712,1
```

`(x,y,z)` grid indices · `(nx,ny,nz)` director · `S` local order parameter (largest eigenvalue of Q averaged over the
site and its non-empty neighbours) · `pt` point type (`1` bulk, `≥2` surface, `0` empty).

The name is `director_field_<int(100·x)>.csv`, with `x` the temperature (`thermal`) or the field (`electric`), e.g.
`director_field_50.csv` for `T = 0.5`. For steps finer than 0.01 the factor becomes 1000, 10000, ..., so that
consecutive points never share a name. In the `step` and `quench` modes `<id>` is the block number.

> At points with `pt = 0` (outside the sample) `S` is written as `NaN`: there is no material there. Tools such as
> pandas, Excel and Origin recognize `NaN` and leave it out of plots and statistics.

---

## Testing

```bash
make test        # quick checks of ./mclics against known answers (CPU modes, a few seconds)
make test-gpu    # the same checks with the GPU modes
```

`tests/regression/run_regression.sh OLD NEW` compares two versions of the program on seven longer cases and is
meant for changes that should not alter the results. See [tests/README.md](tests/README.md).

---

## Project layout

```
mclics/
├── include/
│   ├── parameters.h               # All simulation parameters and their defaults
│   ├── define.h                   # Lattice layout (indices, point types, local environment of a site)
│   ├── evolve.h / evolve.cuh      # Evolution modes: CPU and GPU classes
│   ├── geometry.h                 # Geometry classes and the site energy
│   ├── potential.h / potential.cuh
│   ├── anchoring.h                # Anchoring types
│   ├── sweep.h                    # Temperature/field sweep points and snapshot names
│   ├── io.h                       # Input reading and validation, output
│   ├── ic.h / simulator.h
│   ├── parameter_order.h          # Q tensor and the extra po.dat columns
│   └── reduce.cuh / stringGPU.cuh # GPU helpers
│
├── src/
│   ├── main.cpp                   # Entry point, banner, removal of old output
│   ├── io.cpp                     # Input reading and validation, GHRL coefficients, snapshots
│   ├── simulator.cpp              # Builds the simulation from the parameters
│   ├── evolve.cpp / evolve.cu     # Monte Carlo sweep and energy (CPU / GPU)
│   ├── evolve_thermal.cpp/.cu     # Temperature sweep
│   ├── evolve_step.cpp/.cu        # Blocks at fixed temperature
│   ├── evolve_quench.cpp/.cu      # Quench
│   ├── evolve_electric.cpp/.cu    # Field sweep
│   ├── geometry*.cpp, geometry.cu # bulk, slab, sphere, custom
│   ├── potential.cpp/.cu          # LL, GHRL and pear potentials, field coupling
│   ├── anchoring*.cpp             # homeotropic, rp, rp_tangent, fg (and the _ghrl variants)
│   ├── parameter_order.cpp        # Order parameters (Q tensor diagonalization)
│   ├── ic.cpp                     # Initial conditions
│   └── reduce.cu, stringGPU.cu    # GPU sums and string comparison
│
├── analyze.py                     # Plots of po.dat
├── average_replicas.py            # Average of po.dat over runs with different seeds
├── tests/                         # Quick checks and regression test (see tests/README.md)
├── artigo/                        # Scripts for the test problems and figures of the paper
└── Makefile
```

---

## Notes

- Only nearest neighbours are used (**`Nk = 1`**); any other value stops the program with an error message.
- The choice between CPU and GPU is made by the **suffix of `evol`** (e.g. `thermal` vs. `thermalGPU`), not by a
  command-line option. The program built by `make` runs both; the one built by `make CPU` only accepts the modes
  without the `GPU` suffix.
- On the CPU, OpenMP uses all available cores, with one random number generator per thread. Which thread updates
  which site depends on the number of threads, so the trajectory of a run changes with `OMP_NUM_THREADS` (its
  averages do not). For bit-identical CPU runs, fix `OMP_NUM_THREADS`. On the GPU, a run is reproducible for a given
  `seed`.
- The rotation amplitude of the trial moves is adapted after every sweep to keep the acceptance near 50%, also
  during the measurement sweeps.
- The order parameter `S` is the largest eigenvalue of the Landau–de Gennes Q tensor (GSL `gsl_eigen_symmv`).

---

## License

McLics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
later version. See [LICENSE](LICENSE) for the full text and [AUTHORS](AUTHORS) for the list of contributors.

McLics uses the GNU Scientific Library (GSL), which is also distributed under the GNU GPL.
