#!/usr/bin/env bash
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Regression test: runs every input in inputs/*.txt with an OLD and a NEW mclics program and compares the results.
# Use it after changing the code, with OLD = the program built before the change: a change that should not affect
# the results must give identical po.dat files.
#
#   ./run_regression.sh OLD NEW          GPU modes (as written in the inputs)
#   ./run_regression.sh --cpu OLD NEW    CPU modes (GPU suffix removed, OMP_NUM_THREADS=1 so runs are reproducible)
#
# Results go to out/<case>/{old,new}/; the comparison is printed and saved to out/summary.txt.
set -uo pipefail
# Runs started in parallel use one thread each: a GOMP_CPU_AFFINITY list in the environment would bind all of
# them to the same CPU (the first one in the list).
unset GOMP_CPU_AFFINITY
export LC_NUMERIC=C  # decimal point in printf/bc whatever the system locale
cd "$(dirname "$0")"

CPU=0
if [[ ${1:-} == --cpu ]]; then CPU=1; shift; fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 [--cpu] OLD_PROGRAM NEW_PROGRAM" >&2
  exit 1
fi
OLD=$(readlink -f "$1")
NEW=$(readlink -f "$2")
for b in "$OLD" "$NEW"; do
  [[ -x $b ]] || { echo "not an executable: $b" >&2; exit 1; }
done
[[ $OLD != "$NEW" ]] || { echo "OLD and NEW are the same file" >&2; exit 1; }

rm -rf out
python3 make_cylinder.py 23 24 out || exit 1
for inp in inputs/*.txt; do
  case=$(basename "$inp" .txt)
  for v in old new; do
    d=out/$case/$v
    mkdir -p "$d"
    if [[ $CPU == 1 ]]; then
      sed -E 's/^(evol|mode)([[:space:]]+[A-Za-z]+)GPU/\1\2/' "$inp" > "$d/input_parameters.txt"
    else
      cp "$inp" "$d/input_parameters.txt"
    fi
    bin=$OLD; [[ $v == new ]] && bin=$NEW
    start=$(date +%s.%N)
    if [[ $CPU == 1 ]]; then
      (cd "$d" && OMP_NUM_THREADS=1 "$bin" input_parameters.txt > run.log 2>&1)
    else
      (cd "$d" && "$bin" input_parameters.txt > run.log 2>&1)
    fi
    status=$?
    end=$(date +%s.%N)
    t=$(echo "$end - $start" | bc)
    echo "$status $t" > "$d/status"
    printf "%-28s %-4s exit=%d  %6.1f s\n" "$case" "$v" "$status" "$t"
  done
done

python3 compare.py out | tee out/summary.txt
