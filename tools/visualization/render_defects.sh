#!/usr/bin/env bash
# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# 3D images of the defects (regions of low local order parameter) of director_field snapshots (needs ParaView).
#
#   render_defects.sh SHELL FIRST [STEP LAST]
#
# Run it in the folder of the snapshots. For each snapshot director_field_<i>.csv, i = FIRST, FIRST+STEP, ..., LAST
# (default: only FIRST), writes iso_<i>.png (oblique view) and iso_x_<i>.png, iso_y_<i>.png, iso_z_<i>.png (views
# along x, y and z). The defects are the volume where S < 0.78 S_max, drawn in red inside a translucent shell:
#   free        the sites of the sample (pt >= 1), for any geometry
#   sphere, hemisphere, slab, cylinder (axis along z)   an analytic surface fitted to the grid
# pvbatch is taken from the variable PVBATCH or from the PATH (tested with ParaView 5.6 and 5.10).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PVBATCH=${PVBATCH:-$(command -v pvbatch || true)}
[[ -n $PVBATCH ]] || { echo "pvbatch not found: install ParaView or set PVBATCH=/path/to/pvbatch" >&2; exit 1; }
usage() { sed -n '6,/^set /p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
[[ ${1:-} == -h || ${1:-} == --help ]] && { usage; exit 0; }
[[ $# -ge 2 ]] || { usage; exit 1; }
shell=$1; first=$2; step=${3:-1}; last=${4:-$first}
case $shell in free|sphere|hemisphere|slab|cylinder) ;; *) echo "unknown shell: $shell" >&2; usage; exit 1 ;; esac

for i in $(seq "$first" "$step" "$last"); do
  f=director_field_$i.csv
  [[ -e $f ]] || { echo "skipped: $f not found" >&2; continue; }
  # largest grid indices (the grid has N = index + 1 sites along each direction)
  read -r nx ny nz < <(awk -F, 'NR > 1 { if ($1 > x) x = $1; if ($2 > y) y = $2; if ($3 > z) z = $3 } END { print x, y, z }' "$f")
  "$PVBATCH" "$HERE/defects.py" "$f" "$i.png" "$nx" "$ny" "$nz" "$shell" 2>&1 | grep -v '^$' || true
  [[ -s iso_$i.png ]] || { echo "pvbatch failed for $f" >&2; exit 1; }
done
