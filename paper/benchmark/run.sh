#!/usr/bin/env bash
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Performance of the Monte Carlo step on one CPU thread, on all cores (OpenMP) and on the GPU (appendix of the paper).
#
# Cases: Lebwohl-Lasher periodic bulk with L = 32, 64, 128, 256 and GHRL (E7 constants) with L = 64, at T = 1.0 (LL,
# nematic) or 0.1 (GHRL), from a random initial condition. The number of Monte Carlo steps N is chosen so that a
# single-thread run does about 1e8 site updates; the multithread runs do MT_FACTOR (default 4) times and the GPU
# runs GPU_FACTOR (default 200) times more, so that the steps dominate their run time as well. The output files of
# the runs are deleted after each measurement (they are large for L = 256). Each case runs twice, with N steps and with 0 steps (same setup and output),
# and the difference of the two times is the cost of the N steps alone: the setup and the output files (whose
# local order parameter is computed on one thread) are left out.
#
#   ./run.sh                     all modes: 1 thread, THREADS threads (default "8 16"), GPU
#   SKIP_GPU=1 ./run.sh          CPU modes only
#   THREADS="8" ./run.sh         other thread counts for the multithread runs
#   SCALE=0.01 ./run.sh          fewer steps (quick check of the script)
#   REPEAT=3 ./run.sh            repeat each measurement (the table uses the fastest and reports the spread)
#   CASES_ONLY="ll256" ./run.sh  only some of the cases
#   Close other programs: the multithread runs use all cores.
#
# Uses the mclics found in the PATH (built with make, so that the GPU modes exist). Results are appended to
# results.csv; then: python3 make_table.py
set -euo pipefail
cd "$(dirname "$0")"

THREADS=${THREADS:-"8 16"}
SKIP_GPU=${SKIP_GPU:-0}
SCALE=${SCALE:-1}
REPEAT=${REPEAT:-1}
MT_FACTOR=${MT_FACTOR:-4}
GPU_FACTOR=${GPU_FACTOR:-200}
# case name, potential lines, size, temperature
CASES=("ll32|potential ll;A 1|32|1.0"
       "ll64|potential ll;A 1|64|1.0"
       "ll128|potential ll;A 1|128|1.0"
       "ll256|potential ll;A 1|256|1.0"
       "ghrl64|potential ghrl;k11 1.48;k22 1;k33 2.28|64|0.1")

{
  echo "# $(date '+%Y-%m-%d %H:%M')"
  echo "# CPU: $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}') ($(nproc) logical CPUs)"
  command -v nvidia-smi > /dev/null && echo "# GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "# mclics: $(command -v mclics)  $(cd "$(dirname "$(readlink -f "$(command -v mclics)")")" && git describe --always --dirty 2> /dev/null || echo 'no git')"
  echo "# gcc: $(gcc -dumpversion)  nvcc: $(nvcc --version 2> /dev/null | awk '/release/{print $6}')"
} >> machine.txt
[[ -f results.csv ]] || echo "case,mode,threads,L,steps,real_s,user_s,sys_s" > results.csv

run_one() {  # case mode threads L steps evol potential_lines T
  local c=$1 mode=$2 th=$3 L=$4 n=$5 evol=$6 pot=$7 T=$8
  local d=runs/$c/$mode$th/steps$n
  rm -rf "$d"; mkdir -p "$d"
  {
    echo "Nx $L"; echo "Ny $L"; echo "Nz $L"
    echo "geometry bulk"; echo "xbound periodic"; echo "ybound periodic"; echo "zbound periodic"
    tr ';' '\n' <<< "$pot"
    echo "evol $evol"; echo "Ti $T"; echo "Tf $T"; echo "dT -0.1"
    echo "MCT $n"; echo "MCS 1"; echo "ic random"; echo "seed 1"
  } > "$d/input_parameters.txt"
  local t
  t=$( { cd "$d" && OMP_NUM_THREADS=$th /usr/bin/time -f "%e,%U,%S" mclics input_parameters.txt > run.log 2> time.log; tail -1 time.log; } )
  echo "$c,$mode,$th,$L,$n,$t" >> results.csv
  rm -f "$d"/ic.csv "$d"/director_field_*.csv  # not needed here (about 1.4 GB per run for L = 256)
  printf "%-7s %-6s %3s threads  L=%-4s steps=%-5s %s s\n" "$c" "$mode" "$th" "$L" "$n" "${t%%,*}"
}

CASES_ONLY=${CASES_ONLY:-}
for r in $(seq "$REPEAT"); do
  for spec in "${CASES[@]}"; do
    IFS='|' read -r c pot L T <<< "$spec"
    [[ -z $CASES_ONLY || " $CASES_ONLY " == *" $c "* ]] || continue
    n=$(python3 -c "print(max(1, round($SCALE * 1e8 / $L**3)))")
    for steps in "$n" 0; do
      run_one "$c" cpu 1 "$L" "$steps" thermal "$pot" "$T"
    done
    for th in $THREADS; do
      for steps in $(( n * MT_FACTOR )) 0; do run_one "$c" cpu "$th" "$L" "$steps" thermal "$pot" "$T"; done
    done
    if [[ $SKIP_GPU != 1 ]]; then
      for steps in $(( n * GPU_FACTOR )) 0; do run_one "$c" gpu 1 "$L" "$steps" thermalGPU "$pot" "$T"; done
    fi
  done
done
echo "done: results.csv, machine.txt"
