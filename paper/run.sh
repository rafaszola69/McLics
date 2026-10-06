#!/usr/bin/env bash
# Simulations for the pear-potential test case of the McLics paper.
#
# Usage:  ./run.sh [selinger|phase|droplets|all]    (default: all)
#   selinger  bulk 20^3, x,y free and z periodic, A=1.5 B1=0.09 C=0.3 (Dhakal & Selinger, PRE 81, 031704,
#             Fig. 2a conditions): S and P vs T, fine temperature grid
#   phase     same bulk for several B1: I-N-polar sequence vs direct I-polar transition (coarse, quick look)
#   phasefine same bulk on the B grid of Dhakal & Selinger's phase diagram (B1 0.02-0.76), dT=0.02;
#             make_figures.py uses it instead of phase when present
#   droplets  homeotropic droplets N=21,31,41 with C=+0.3, -0.3, 0; and N=31 with C=0.1, 0.2 (linear response)
#
#   JOBS=4 ./run.sh droplets      simultaneous runs (default: nproc; for one GPU, try JOBS=1..4)
#   SEEDS="1" ./run.sh            fewer replicas (default: 3)
#   DROP_SIZES="41 31 21" LINEAR_N=31 B1_VALUES="..."   override the defaults
#
# Each run goes to runs/<group>/<variant>/seed<s>/ with its own input, po.dat, snapshots (gzip) and run.log.
# Runs with a DONE file are skipped, so the script can be interrupted and relaunched.
# Afterwards: python3 make_figures.py
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-all}
JOBS=${JOBS:-$(nproc)}
SEEDS=${SEEDS:-"1 2 3"}
BASE=base.in
DROP_SIZES=${DROP_SIZES:-"41 31 21"}
B1_VALUES=${B1_VALUES:-"0.0 0.09 0.3 0.6 1.0 1.5"}
LINEAR_N=${LINEAR_N:-31}
B1_FINE=${B1_FINE:-"0.02 0.04 0.08 0.12 0.16 0.20 0.24 0.28 0.32 0.36 0.40 0.48 0.56 0.64 0.72 0.76"}

case $STAGE in
  selinger|phase|phasefine|droplets|all) ;;
  *) echo "unknown stage: $STAGE (use selinger, phase, phasefine, droplets or all)" >&2; exit 1 ;;
esac

set_param() {  # file key value...
  local f=$1 k=$2; shift 2
  if grep -q "^$k " "$f"; then sed -i "s/^$k .*/$k $*/" "$f"; else echo "$k $*" >> "$f"; fi
}

JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT

new_run() {  # dir seed key=value...
  local dir=$1 seed=$2; shift 2
  [[ -f $dir/DONE ]] && return
  mkdir -p "$dir"
  cp "$BASE" "$dir/input_parameters.txt"
  set_param "$dir/input_parameters.txt" seed "$seed"
  local kv
  for kv in "$@"; do set_param "$dir/input_parameters.txt" "${kv%%=*}" "${kv#*=}"; done
  echo "$dir" >> "$JOBLIST"
}

BULK=(Nx=20 Ny=20 Nz=20 geometry=bulk xbound=free ybound=free zbound=periodic radial=none)

if [[ $STAGE == selinger || $STAGE == all ]]; then
  for s in $SEEDS; do
    new_run "runs/selinger/C0.3/seed$s" "$s" "${BULK[@]}" C=0.3 Ti=1.6 Tf=0.1 dT=-0.025
  done
fi

if [[ $STAGE == phase || $STAGE == all ]]; then
  for b in $B1_VALUES; do
    for s in $SEEDS; do
      new_run "runs/phase/B$b/seed$s" "$s" "${BULK[@]}" C=0.3 B1="$b" Ti=3.0 Tf=0.1 dT=-0.05
    done
  done
fi

if [[ $STAGE == phasefine || $STAGE == all ]]; then
  for b in $B1_FINE; do
    for s in $SEEDS; do
      new_run "runs/phasefine/B$b/seed$s" "$s" "${BULK[@]}" C=0.3 B1="$b" Ti=2.0 Tf=0.04 dT=-0.02
    done
  done
fi

if [[ $STAGE == droplets || $STAGE == all ]]; then
  for N in $DROP_SIZES; do              # largest first: better load balance
    for c in 0.3 -0.3 0.0; do
      for s in $SEEDS; do
        new_run "runs/droplets/N$N/C$c/seed$s" "$s" Nx="$N" Ny="$N" Nz="$N" C="$c"
      done
    done
  done
  for c in 0.1 0.2; do
    for s in $SEEDS; do
      new_run "runs/droplets/N$LINEAR_N/C$c/seed$s" "$s" Nx="$LINEAR_N" Ny="$LINEAR_N" Nz="$LINEAR_N" C="$c"
    done
  done
fi

echo "$(wc -l < "$JOBLIST") runs to execute, $JOBS at a time."

xargs -a "$JOBLIST" -P "$JOBS" -I{} bash -c '
  cd "{}" && echo "[start] {}" &&
  if OMP_NUM_THREADS=1 mclics input_parameters.txt > run.log 2>&1; then
    gzip -f director_field_*.csv; touch DONE; echo "[ok]    {}"
  else
    echo "[FAIL]  {} (see run.log)"
  fi'
