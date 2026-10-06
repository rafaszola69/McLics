#!/usr/bin/env bash
# Simulations for further test problems of the McLics paper.
#
# Usage:  ./run.sh [twist|twistT|cano|llbulk|llheat|homeo|chiral|all]     (default: all)
#   twist    chiral GHRL (E7 constants) in a 6x6x48 box, periodic x,y and free z (no anchoring), starting from a
#            helix along z with p0_i = 0.8, 1.0 and 1.25 p0; one temperature, long run (small layers let the
#            twist relax quickly through the free ends).
#            Measures the equilibrium pitch vs p0 (lattice prediction: 2*pi/arcsin(2*pi/p0)).
#   cano     chiral GHRL, p0 = 20, slab with planar anchoring along x on both plates, thickness Nz = 6..46,
#            cooled from the isotropic phase: number of half-turns vs d/p (Grandjean steps).
#   twistT   as twist, start p0_i = p0, for p0 = 8 and 10 at T = 0.05 ... 0.5: pitch vs temperature
#            (tests the approach to the T = 0 lattice prediction).
#   llbulk   Lebwohl-Lasher, periodic bulk L = 10, 16, 24, 32, 40, fine cooling across T_NI (literature: 1.1232).
#   llheat   same sizes, heating from the aligned state: brackets T_NI together with llbulk (first-order hysteresis).
#   homeo    Lebwohl-Lasher droplets N = 15, 21, 31, 41 with homeotropic anchoring of strength W = 0.01 ... 1:
#            uniform/axial -> radial structure; the radial order should collapse as a function of W*R.
#   chiral   chiral GHRL (E7) droplets with degenerate planar (FG) anchoring, N = 41, |R/p0| <= 0.3 (p0 >= 65 sites),
#            4 seeds: bipolar -> twisted bipolar, twist sign set by the handedness (sign of p0).
#
#   JOBS=4 ./run.sh homeo        simultaneous runs (default: 2; several runs share the GPU)
#   EVOL=thermal ./run.sh         run on the CPU instead of the GPU (default: thermalGPU)
#   SEEDS="1" ./run.sh            fewer replicas (default: 1 2 for twist/twistT/cano/homeo, 1 2 3 otherwise)
#
# Each run goes to runs/<stage>/<variant>/seed<s>/; runs with DONE are skipped. Snapshots are gzipped.
# Afterwards: python3 make_figures.py
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-all}
JOBS=${JOBS:-2}
EVOL=${EVOL:-thermalGPU}
SEEDS_FAST=${SEEDS:-"1 2"}
SEEDS_SLOW=${SEEDS:-"1 2 3"}
TWIST_P0=${TWIST_P0:-"8 10 12 16 20 30 40"}
CANO_NZ=${CANO_NZ:-"6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46"}
TWIST_T=${TWIST_T:-"0.05 0.1 0.2 0.3 0.5"}
LL_SIZES=${LL_SIZES:-"40 32 24 16 10"}
HOMEO_N=${HOMEO_N:-"41 31 21 15"}
HOMEO_W=${HOMEO_W:-"0.03 0.06 0.1 0.13 0.16 0.2 0.24 0.28 0.33 0.4 0.5 0.7 1.0"}
CHIRAL_N=${CHIRAL_N:-41}
CHIRAL_X=${CHIRAL_X:-"0 0.01 0.02 0.03 0.05 0.1 0.15 0.2 0.3 -0.01 -0.02 -0.03 -0.1 -0.2 -0.3"}  # R/p0 (negative = opposite handedness);
# keep |R/p0| <= 0.3 so that p0 >= 65 sites: shorter pitches give square cholesteric layers (lattice artefact)
SEEDS_CHIRAL=${SEEDS:-"1 2 3 4"}

case $STAGE in
  twist|twistT|cano|llbulk|llheat|homeo|chiral|all) ;;
  *) echo "unknown stage: $STAGE (use twist, twistT, cano, llbulk, llheat, homeo, chiral or all)" >&2; exit 1 ;;
esac

JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT

new_run() {  # dir seed "key value" ...
  local dir=$1 seed=$2; shift 2
  [[ -f $dir/DONE ]] && return
  mkdir -p "$dir"
  {
    echo "seed $seed"
    echo "MCS 2000"
    local kv
    for kv in "$@"; do echo "$kv"; done
  } > "$dir/input_parameters.txt"
  echo "$dir" >> "$JOBLIST"
}

GHRL_E7=("potential ghrl" "k11 1.48" "k22 1" "k33 2.28")

if [[ $STAGE == twist || $STAGE == all ]]; then
  for p in $TWIST_P0; do
    for f in 0.80 1.00 1.25; do
      pi=$(python3 -c "print(round($p*$f, 3))")
      for s in $SEEDS_FAST; do
        new_run "runs/twist/p$p/start$f/seed$s" "$s" "${GHRL_E7[@]}" "p0 $p" \
          "Nx 6" "Ny 6" "Nz 48" "geometry bulk" "xbound periodic" "ybound periodic" "zbound free" \
          "ic cholesteric" "p0_i $pi" "theta_0 90" "phi_0 0" \
          "evol $EVOL" "Ti 0.3" "Tf 0.3" "dT -0.1" "MCT 100000" "radial none"
      done
    done
  done
fi

if [[ $STAGE == cano || $STAGE == all ]]; then
  for nz in $CANO_NZ; do
    for s in $SEEDS_FAST; do
      new_run "runs/cano/Nz$nz/seed$s" "$s" "${GHRL_E7[@]}" "p0 20" \
        "Nx 12" "Ny 12" "Nz $nz" "geometry slab" "xbound periodic" "ybound periodic" \
        "anchoring_type 0 rp" "W 0 1" "theta_s 0 90" "phi_s 0 0" \
        "anchoring_type 1 rp" "W 1 1" "theta_s 1 90" "phi_s 1 0" \
        "ic random" "evol $EVOL" "Ti 1.4" "Tf 0.3" "dT -0.05" "MCT 10000" "radial none"
    done
  done
fi

if [[ $STAGE == llbulk || $STAGE == all ]]; then
  for L in $LL_SIZES; do
    for s in $SEEDS_SLOW; do
      new_run "runs/llbulk/L$L/seed$s" "$s" "potential ll" "A 1" \
        "Nx $L" "Ny $L" "Nz $L" "geometry bulk" "xbound periodic" "ybound periodic" "zbound periodic" \
        "ic random" "evol $EVOL" "Ti 1.24" "Tf 1.02" "dT -0.004" "MCT 4000" "MCS 16000" "radial none"
    done
  done
fi

if [[ $STAGE == twistT || $STAGE == all ]]; then
  for p in 8 10; do
    for T in $TWIST_T; do
      for s in $SEEDS_FAST; do
        new_run "runs/twistT/p$p/T$T/seed$s" "$s" "${GHRL_E7[@]}" "p0 $p" \
          "Nx 6" "Ny 6" "Nz 48" "geometry bulk" "xbound periodic" "ybound periodic" "zbound free" \
          "ic cholesteric" "p0_i $p" "theta_0 90" "phi_0 0" \
          "evol $EVOL" "Ti $T" "Tf $T" "dT -0.1" "MCT 100000" "radial none"
      done
    done
  done
fi

if [[ $STAGE == llheat || $STAGE == all ]]; then
  for L in $LL_SIZES; do
    for s in $SEEDS_SLOW; do
      new_run "runs/llheat/L$L/seed$s" "$s" "potential ll" "A 1" \
        "Nx $L" "Ny $L" "Nz $L" "geometry bulk" "xbound periodic" "ybound periodic" "zbound periodic" \
        "ic homogeneous" "theta_0 0" "phi_0 0" \
        "evol $EVOL" "Ti 1.02" "Tf 1.24" "dT 0.004" "MCT 4000" "MCS 16000" "radial none"
    done
  done
fi

if [[ $STAGE == homeo || $STAGE == all ]]; then
  for N in $HOMEO_N; do
    for w in $HOMEO_W; do
      for s in $SEEDS_FAST; do
        new_run "runs/homeo/N$N/W$w/seed$s" "$s" "potential ll" "A 1" \
          "Nx $N" "Ny $N" "Nz $N" "geometry sphere" "anchoring_type 0 homeotropic" "W 0 $w" "radial sphere" \
          "ic random" "evol $EVOL" "Ti 1.4" "Tf 0.3" "dT -0.05" "MCT 10000"
      done
    done
  done
fi

if [[ $STAGE == chiral || $STAGE == all ]]; then
  R=$(python3 -c "print($CHIRAL_N/2-1)")
  for x in $CHIRAL_X; do
    p=$(python3 -c "x=$x; print(0 if x==0 else round($R/x, 3))")
    for s in $SEEDS_CHIRAL; do
      new_run "runs/chiral/N$CHIRAL_N/x$x/seed$s" "$s" "${GHRL_E7[@]}" "p0 $p" \
        "Nx $CHIRAL_N" "Ny $CHIRAL_N" "Nz $CHIRAL_N" "geometry sphere" "anchoring_type 0 fg" "W 0 1" "radial sphere" \
        "ic random" "evol $EVOL" "Ti 1.4" "Tf 0.2" "dT -0.05" "MCT 10000"
    done
  done
fi

echo "$(wc -l < "$JOBLIST") runs to execute, $JOBS at a time."

xargs -a "$JOBLIST" -P "$JOBS" -I{} bash -c '
  cd "{}" && echo "[start] {}" &&
  if OMP_NUM_THREADS=1 mclics input_parameters.txt > run.log 2>&1; then
    gzip -f director_field_*.csv 2>/dev/null; touch DONE; echo "[ok]    {}"
  else
    echo "[FAIL]  {} (see run.log)"
  fi'
