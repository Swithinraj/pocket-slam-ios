#!/usr/bin/env bash
# Replays one recording through the VLIO core with each ablation setting (results in <rec>/_var/<variant>/).
# Usage (from the repo root, after building core/ with CMake into ~/build/slamcore):
#   bash tools/run_ablation.sh <recording_dir>
set -euo pipefail
REC="$1"
BIN="${SLAMCORE_BUILD:-$HOME/build/slamcore}/run_recording"
run() { local name="$1"; shift; mkdir -p "$REC/_var/$name"; "$BIN" "$REC" "$REC/_var/$name" "$@" > "$REC/_var/$name/log.txt" 2>&1; echo "done $name"; }
run base &
run nodepth --no-depth &
run hostonly --no-target-depth &
run targetonly --no-host-depth &
wait
run alldepth --flatness 10 &
run loosedepth --depth-sigma-scale 3 &
run tightpx --pixel-sigma 2 &
run age10 --max-age 10 &
wait
