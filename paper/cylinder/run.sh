#!/usr/bin/env bash
# Chiral nematic in a cylindrical cavity with homeotropic anchoring (custom geometry), after Wand & Bates,
# PRE 100, 052702 (2019): two s=+1/2 disclination lines form a double helix whose repeat distance lambda is compared
# with the bulk cholesteric pitch p (they find lambda = 1.45 p in an off-lattice model).
#
# Usage:  ./run.sh [scan|bates|bulkp|all]     (default: all)
#   scan    spontaneous selection: cylinders of length L = 24 ... 144 (periodic along z) cooled from the isotropic phase
#           to T = 0.1 with several seeds; the twist of the double helix is quantized (xi = n/2 full turns over L),
#           so the twist chosen vs L forms steps whose average slope gives 1/lambda (like Grandjean steps)
#   bates   Wand & Bates procedure as a check: idealized double helices with xi = -0.5 ... 3 turns over L = 48, kept at
#           T = 0.1; energy vs xi (runs whose twist changed are discarded by the analysis)
#   bulkp   bulk pitch of the same chiral GHRL at T = 0.1 (6x6x96 box, free along the helix)
#   stretch relaxed double helix of the scan (one turn over L) stretched along z to longer L and relaxed again at
#           T = 0.1: energy vs repeat distance lambda = L on the branch of the cooled states, beyond the lambdas that
#           cooling reaches (the twist trapped on cooling is too large); needs the scan results
#
#   JOBS=4 ./run.sh scan          simultaneous runs (default: 2; they share the GPU)
#   EVOL=thermal ./run.sh         run on the CPU (default: thermalGPU)
#   P0="25 40" RADII="10" LS="..." SEEDS="1 2 3" ANCH=homeotropic_ghrl W=5 ./run.sh       override the defaults
#   KSET=eq LS="24 30 36 42 48" ./run.sh scan; KSET=eq ./run.sh stretch     equal elastic constants (runs_eqK/)
set -euo pipefail
cd "$(dirname "$0")"

STAGE=${1:-all}
JOBS=${JOBS:-2}
EVOL=${EVOL:-thermalGPU}
P0=${P0:-"25 40"}
RADII=${RADII:-"10"}
LS=${LS:-"24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 120 132 144"}
SEEDS=${SEEDS:-"1 2 3"}
T=${T:-0.1}
ANCH=${ANCH:-homeotropic_ghrl}   # W=1 is too weak: cooling then gives a planar cholesteric across the cylinder, no defects
W=${W:-5}
MCT_SCAN=${MCT_SCAN:-10000}
MCS_SCAN=${MCS_SCAN:-5000}
L_BATES=${L_BATES:-48}
XI=${XI:-"-0.5 0 0.5 1 1.5 2 2.5 3"}
SIGN=-1          # handedness of the bulk helix for p0 > 0 in this code (see the cholesteric example)

case $STAGE in
  scan|bates|bulkp|stretch|all) ;;
  *) echo "unknown stage: $STAGE (use scan, bates, bulkp, stretch or all)" >&2; exit 1 ;;
esac

JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT
mkdir -p geom ics

new_run() {  # dir seed "key value" ...
  local dir=$1 seed=$2; shift 2
  [[ -f $dir/DONE ]] && return
  mkdir -p "$dir"
  { echo "seed $seed"; local kv; for kv in "$@"; do echo "$kv"; done; } > "$dir/input_parameters.txt"
  echo "$dir" >> "$JOBLIST"
}

KSET=${KSET:-e7}  # e7: E7 constants (runs/); eq: K11=K22=K33 (runs_eqK/); wb: ratios of Wand & Bates' model (runs_wbK/)
case $KSET in
  e7) GHRL=("potential ghrl" "k11 1.48" "k22 1" "k33 2.28"); RUNS=runs ;;
  eq) GHRL=("potential ghrl" "k11 1" "k22 1" "k33 1"); RUNS=runs_eqK ;;
  wb) GHRL=("potential ghrl" "k11 1.054" "k22 1" "k33 1.015"); RUNS=runs_wbK ;;   # K1/K3=1.038, K2/K3=0.985 (Wand & Bates)
  *) echo "unknown KSET: $KSET (use e7, eq or wb)" >&2; exit 1 ;;
esac
CYL=("xbound free" "ybound free" "zbound periodic" "anchoring_type 0 $ANCH" "W 0 $W" "radial cylinder")

geometry_file() {  # N L -> geom/cyl_N<N>_L<L>.csv (created if needed)
  local f=geom/cyl_N$1_L$2.csv
  [[ -f $f ]] || python3 prepare.py geometry "$1" "$2" "$f"
  echo "$f"
}

if [[ $STAGE == scan || $STAGE == all ]]; then
  for R in $RADII; do
    N=$(( 2 * R + 3 ))
    for L in $LS; do
      g=$(geometry_file "$N" "$L")
      for p in $P0; do
        for s in $SEEDS; do
          new_run "$RUNS/scan/p$p/R$R/L$L/seed$s" "$s" "${GHRL[@]}" "p0 $p" \
            "Nx $N" "Ny $N" "Nz $L" "geometry custom" "boundary_file ../../../../../../$g" "${CYL[@]}" \
            "ic random" "evol $EVOL" "Ti 1.4" "Tf $T" "dT -0.05" "MCT $MCT_SCAN" "MCS $MCS_SCAN"
        done
      done
    done
  done
fi

if [[ $STAGE == bates || $STAGE == all ]]; then
  for R in $RADII; do
    N=$(( 2 * R + 3 ))
    g=$(geometry_file "$N" "$L_BATES")
    for xi in $XI; do
      ic=ics/helix_N${N}_L${L_BATES}_xi${xi}.csv
      [[ -f $ic ]] || python3 prepare.py helix "$N" "$L_BATES" "$xi" "$SIGN" "$ic"
      for p in $P0; do
        for s in 1 2; do
          new_run "$RUNS/bates/p$p/R$R/xi$xi/seed$s" "$s" "${GHRL[@]}" "p0 $p" \
            "Nx $N" "Ny $N" "Nz $L_BATES" "geometry custom" "boundary_file ../../../../../../$g" "${CYL[@]}" \
            "ic ic_file" "ic_file ../../../../../../$ic" \
            "evol $EVOL" "Ti $T" "Tf $T" "dT -0.1" "MCT 40000" "MCS 10000"
        done
      done
    done
  done
fi

if [[ $STAGE == bulkp || $STAGE == all ]]; then
  for p in $P0; do
    for s in 1 2; do
      new_run "$RUNS/bulkp/p$p/seed$s" "$s" "${GHRL[@]}" "p0 $p" \
        "Nx 6" "Ny 6" "Nz 96" "geometry bulk" "xbound periodic" "ybound periodic" "zbound free" \
        "ic cholesteric" "p0_i $p" "theta_0 90" "phi_0 0" "radial none" \
        "evol $EVOL" "Ti $T" "Tf $T" "dT -0.1" "MCT 100000" "MCS 2000"
    done
  done
fi

# stretch: source = the lowest-energy scan run whose double helix makes exactly one turn (chosen by prepare.py pick;
# STRETCH_SRC_25 / STRETCH_SRC_40 override it)
STRETCH_L_25=${STRETCH_L_25:-"24 28 32 36 40 44 50 56 64 76 90 110"}   # even L: z is periodic (checkerboard)
STRETCH_L_40=${STRETCH_L_40:-"48 52 58 64 72 80 90 104 120 140 160 180"}
if [[ $STAGE == stretch ]]; then    # not part of "all": it needs the scan results
  R=10; N=$(( 2 * R + 3 ))
  for p in 25 40; do
    src=STRETCH_SRC_$p; src=${!src:-}; lst=STRETCH_L_$p
    [[ -n $src ]] || src=$(python3 prepare.py pick "$RUNS" "$p" "$R")
    [[ -n $src ]] || { echo "no clean scan run with one turn for p0=$p in $RUNS: skipped (set STRETCH_SRC_$p)" >&2; continue; }
    echo "stretch p0=$p: source $src"
    snap=$(ls "$src"/director_field_*.csv* 2>/dev/null | head -1)
    [[ -n $snap ]] || { echo "no snapshot in $src (run the scan first)" >&2; exit 1; }
    Ls=$(awk '$1=="Nz"{print $2}' "$src/input_parameters.txt")
    for L in ${!lst}; do
      g=$(geometry_file "$N" "$L")
      ic=ics/stretch_p${p}_$(echo "$snap" | md5sum | cut -c1-8)_L${Ls}to${L}.csv   # name tied to the source snapshot
      [[ -f $ic ]] || python3 prepare.py stretch "$snap" "$N" "$Ls" "$L" "$ic"
      for s in 1 2; do
        new_run "$RUNS/stretch/p$p/R$R/L$L/seed$s" "$s" "${GHRL[@]}" "p0 $p" \
          "Nx $N" "Ny $N" "Nz $L" "geometry custom" "boundary_file ../../../../../../$g" "${CYL[@]}" \
          "ic ic_file" "ic_file ../../../../../../$ic" \
          "evol $EVOL" "Ti $T" "Tf $T" "dT -0.1" "MCT 40000" "MCS 10000"
      done
    done
  done
fi

echo "$(wc -l < "$JOBLIST") runs to execute, $JOBS at a time."

xargs -a "$JOBLIST" -P "$JOBS" -I{} bash -c '
  cd "{}" || exit 0
  [ -f DONE ] && { echo "[skip]  {} (done elsewhere)"; exit 0; }
  mkdir .running 2>/dev/null || { echo "[skip]  {} (running elsewhere)"; exit 0; }   # lock shared between machines
  trap "rmdir .running 2>/dev/null" EXIT
  echo "[start] {}"
  if OMP_NUM_THREADS=1 mclics input_parameters.txt > run.log 2>&1; then
    f=$(ls director_field_*.csv 2>/dev/null | sort -t_ -k3 -n | head -1)
    for g in director_field_*.csv; do [ "$g" = "$f" ] || rm -f "$g"; done   # keep only the final (lowest T) snapshot
    gzip -f director_field_*.csv 2>/dev/null; touch DONE; echo "[ok]    {}"
  else
    echo "[FAIL]  {} (see run.log)"
  fi'
