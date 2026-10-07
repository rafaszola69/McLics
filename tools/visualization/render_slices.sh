#!/usr/bin/env bash
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Images of the director field in the three central planes of director_field snapshots (needs ParaView).
#
#   render_slices.sh [-C] FIRST [STEP LAST]
#
# Run it in the folder of the snapshots. For each snapshot director_field_<i>.csv, i = FIRST, FIRST+STEP, ..., LAST
# (default: only FIRST), writes x_director_field_<i>.png, y_... and z_..., the planes x = Nx/2, y = Ny/2 and
# z = Nz/2 seen along their normal, with one cylinder per site along the director.
#   default  colored by the local order parameter: green, and red where S < 0.438 (defects)
#   -C       colored by the director component normal to the plane: red in the plane, blue along the normal
#            (useful for cholesterics)
# pvbatch is taken from the variable PVBATCH or from the PATH (tested with ParaView 5.6 and 5.10).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PVBATCH=${PVBATCH:-$(command -v pvbatch || true)}
[[ -n $PVBATCH ]] || { echo "pvbatch not found: install ParaView or set PVBATCH=/path/to/pvbatch" >&2; exit 1; }

color=S
args=()
for a in "$@"; do
  case $a in
    -C|--chol|--abs) color=abs ;;
    -h|--help) sed -n '6,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) args+=("$a") ;;
  esac
done
[[ ${#args[@]} -ge 1 ]] || { sed -n '6,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 1; }
first=${args[0]}; step=${args[1]:-1}; last=${args[2]:-$first}

AXES=(x y z)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for i in $(seq "$first" "$step" "$last"); do
  f=director_field_$i.csv
  [[ -e $f ]] || { echo "skipped: $f not found" >&2; continue; }
  python3 "$HERE/extract_slices.py" "$f" "$tmp"
  pids=()
  for axis in x y z; do  # the three planes in parallel
    "$PVBATCH" "$HERE/slice.py" "$tmp/${axis}_$f" "${axis}_director_field_$i.png" "$axis" "$color" > "$tmp/$axis.log" 2>&1 &
    pids+=($!)
  done
  for k in 0 1 2; do
    axis=${AXES[$k]}
    if ! wait "${pids[$k]}" || [[ ! -s ${axis}_director_field_$i.png ]]; then
      echo "pvbatch failed for the $axis plane of $f:" >&2; cat "$tmp/$axis.log" >&2; exit 1
    fi
  done
  echo "$f -> x_, y_, z_director_field_$i.png"
done
