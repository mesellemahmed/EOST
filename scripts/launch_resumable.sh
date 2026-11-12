#!/usr/bin/env bash
# Resumable orchestration (host side: Docker CLI only).
# One container per (phase, bank). A bank whose artifacts/<phase>/<bank>/config.json
# exists is complete and is skipped, so an interrupted campaign is resumed by
# simply re-running this script. Every container first re-checks PREREG_GATE.
# Runs are deterministic (all seeds derive from the master seed by hashing), so a
# bank re-run from scratch reproduces exactly what an uninterrupted run would give.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=artifacts/logs/resume.log
mkdir -p artifacts/logs

run_bank() {  # run_bank <phase> <bank> [run_benchmark args...]
  local phase=$1 ds=$2; shift 2
  if [ -f "artifacts/$phase/$ds/config.json" ]; then
    echo "$(date -u +%FT%TZ) skip $phase/$ds (complete)" >> "$LOG"; return 0
  fi
  echo "$(date -u +%FT%TZ) start $phase/$ds $*" >> "$LOG"
  docker run --rm --network none --cpus 2 -v "$PWD":/work eost-runtime:cpu bash -c \
    "python scripts/verify_artifacts.py --prereg-only >/dev/null && python scripts/run_benchmark_resumable.py --phase $phase --datasets $ds $*" \
    >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) done $phase/$ds" >> "$LOG"
}

CONF=(ollb_gsm8k ollb_arc ollb_hellaswag ollb_winogrande ollb_mmlu swe_verified livecodebench cruxeval_o)
SYN=(synth_2pl synth_guess synth_2dim)

stage="${1:-all}"
if [[ "$stage" == all || "$stage" == confirm ]]; then
  for ds in "${CONF[@]}" "${SYN[@]}"; do run_bank confirm "$ds" --targets 60 --reps 5; done
  echo "$(date -u +%FT%TZ) CONFIRM_COMPLETE" >> "$LOG"
fi
if [[ "$stage" == all || "$stage" == sensitivity ]]; then
  # --evr-guard: numerical guard for the EVR criterion (no-op unless an expected response
  # probability leaves [0,1] by rounding; see install_evr_guard and configs/deviations.txt)
  for ds in "${CONF[@]}"; do run_bank sens_1pl "$ds" --irt-model 1pl --targets 30 --reps 3 --panel-sizes 8 64 --evr-guard; done
  for ds in "${CONF[@]}"; do run_bank sens_fixedprior "$ds" --fixed-prior --targets 30 --reps 3 --panel-sizes 8 64 --evr-guard; done
  for ds in ollb_gsm8k ollb_mmlu ollb_arc ollb_hellaswag ollb_winogrande; do
    run_bank sens_dedup "$ds" --max-kappa 0.8 --targets 30 --reps 3 --panel-sizes 8 64 --evr-guard
  done
  echo "$(date -u +%FT%TZ) SENSITIVITY_COMPLETE" >> "$LOG"
fi
if [[ "$stage" == all || "$stage" == explore ]]; then
  if [ ! -f artifacts/explore_ranking/ranking.csv ]; then
    docker run --rm --network none --cpus 2 -v "$PWD":/work eost-runtime:cpu \
      python scripts/explore_ranking.py --datasets "${CONF[@]}" --N 8 --reps 5 >> "$LOG" 2>&1
  fi
  echo "$(date -u +%FT%TZ) EXPLORE_COMPLETE" >> "$LOG"
fi
if [[ "$stage" == posthoc ]]; then
  # Post hoc (not pre-registered): the pre-registered one-parameter joint MAP is degenerate
  # (the common discrimination diverges); a strongly informative prior on log a (sd 0.05)
  # pins it to a finite value. Reported separately and labelled exploratory.
  for ds in "${CONF[@]}"; do
    run_bank sens_1pl_reg "$ds" --irt-model 1pl --sd-log-a 0.05 --targets 30 --reps 3 --panel-sizes 8 64 --evr-guard
  done
  echo "$(date -u +%FT%TZ) POSTHOC_COMPLETE" >> "$LOG"
fi
echo "$(date -u +%FT%TZ) ALL_COMPLETE ($stage)" >> "$LOG"
