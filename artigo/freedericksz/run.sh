#!/usr/bin/env bash
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Freedericksz transition in three cells (test problem "Elastic Behavior in Nematic Liquid Crystals" of the paper,
# after Le Masurier et al.): GHRL potential (no chirality) with the E7 elastic constants (K1/K2 = 1.48, K3/K2 = 2.28) in a
# 10 x 10 x 16 slab (x, y periodic), electric mode with the field swept from Ei to Ef.
#
#   splay  planar anchoring along x, field along z, dielectric anisotropy +DEPS
#   twist  planar anchoring along x, field along y, +DEPS
#   bend   homeotropic anchoring,    field along z, -DEPS
#
# The critical fields satisfy E_c^2 ~ K_i, so (E_c splay / E_c twist)^2 = K1/K2 and (E_c bend / E_c twist)^2 = K3/K2.
# Near the threshold the deformation grows slowly (the relaxation time diverges), and more slowly the smaller the
# elastic constant, so the field must be swept slowly (many sweeps per field value, small dE): faster sweeps
# overestimate the thresholds, the twist one most.
#
#   ./run.sh                      all three cells (default: one at a time on the GPU)
#   ./run.sh splay                one cell
#   JOBS=3 ./run.sh               simultaneous runs
#   EVOL=electric ./run.sh        run on the CPU instead of the GPU (default: electricGPU)
#   SEEDS="1 2 3" ./run.sh        replicas (default: 1 2)
#   T=0.1 W=10 MCT=40000 MCS=1000 EI=0.1 EF=1.2 DE=0.005 DEPS=0.25 ./run.sh    (defaults shown)
#
# Each run goes to runs/<cell>/seed<s>/; runs with DONE are skipped. Snapshots are gzipped.
# Afterwards: python3 make_figures.py
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-all}
JOBS=${JOBS:-1}
EVOL=${EVOL:-electricGPU}
SEEDS=${SEEDS:-"1 2"}
T=${T:-0.1}
W=${W:-10}
MCT=${MCT:-40000}
MCS=${MCS:-1000}
EI=${EI:-0.1}
EF=${EF:-1.2}
DE=${DE:-0.005}
DEPS=${DEPS:-0.25}

case $STAGE in
  splay|twist|bend|all) ;;
  *) echo "unknown cell: $STAGE (use splay, twist, bend or all)" >&2; exit 1 ;;
esac

JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT

new_run() {  # dir seed "key value" ...
  local dir=$1 seed=$2; shift 2
  [[ -f $dir/DONE ]] && return
  mkdir -p "$dir"
  {
    echo "seed $seed"
    local kv
    for kv in "$@"; do echo "$kv"; done
  } > "$dir/input_parameters.txt"
  echo "$dir" >> "$JOBLIST"
}

COMMON=("potential ghrl" "k11 1.48" "k22 1" "k33 2.28"
        "Nx 10" "Ny 10" "Nz 16" "geometry slab" "xbound periodic" "ybound periodic"
        "evol $EVOL" "Ti $T" "Ei $EI" "Ef $EF" "dE $DE" "MCT $MCT" "MCS $MCS")
PLANAR=("ic homogeneous" "theta_0 90" "phi_0 0"
        "anchoring_type 0 rp" "W 0 $W" "theta_s 0 90" "phi_s 0 0"
        "anchoring_type 1 rp" "W 1 $W" "theta_s 1 90" "phi_s 1 0")
HOMEO=("ic homogeneous" "theta_0 0" "phi_0 0"
       "anchoring_type 0 homeotropic" "W 0 $W" "anchoring_type 1 homeotropic" "W 1 $W")

for s in $SEEDS; do
  if [[ $STAGE == splay || $STAGE == all ]]; then
    new_run "runs/splay/seed$s" "$s" "${COMMON[@]}" "${PLANAR[@]}" "elecA $DEPS" "elecX 0" "elecY 0" "elecZ 1"
  fi
  if [[ $STAGE == twist || $STAGE == all ]]; then
    new_run "runs/twist/seed$s" "$s" "${COMMON[@]}" "${PLANAR[@]}" "elecA $DEPS" "elecX 0" "elecY 1" "elecZ 0"
  fi
  if [[ $STAGE == bend || $STAGE == all ]]; then
    new_run "runs/bend/seed$s" "$s" "${COMMON[@]}" "${HOMEO[@]}" "elecA -$DEPS" "elecX 0" "elecY 0" "elecZ 1"
  fi
done

echo "$(wc -l < "$JOBLIST") runs to execute, $JOBS at a time."
xargs -a "$JOBLIST" -P "$JOBS" -I{} bash -c '
  cd "{}" && echo "[start] {}" &&
  if OMP_NUM_THREADS=1 mclics input_parameters.txt > run.log 2>&1; then
    gzip -f director_field_*.csv 2>/dev/null; touch DONE; echo "[ok]    {}"
  else
    echo "[FAIL]  {} (see run.log)"
  fi'
