#!/usr/bin/env bash
# Host-side orchestration only: run the post hoc corrections of study 2 (decided before any
# study-2 result was inspected; docs/RESEARCH_LOG.md) in one detached container. The container
# waits until the frozen main component has written runs.csv.gz for all eight confirmatory
# banks, then runs: reproduction check of the frozen fully Bayesian rows -> tinyBenchmarks anchor
# audit (N = 8) -> fully Bayesian CAT with a 60-sweep burn-in (N = 8, then N = 4) -> analysis.
# Resumable: completed banks/targets are skipped. Log: docker logs eost-s2fix.
set -euo pipefail
cd "$(dirname "$0")/.."
docker rm -f eost-s2fix >/dev/null 2>&1 || true
docker run -d --name eost-s2fix --network none --cpus 2 -v "$PWD":/work -w /work -e PYTHONPATH=/work/src \
  eost-runtime:cpu bash -c '
  set -e
  for b in ollb_gsm8k ollb_arc ollb_hellaswag ollb_winogrande ollb_mmlu swe_verified livecodebench cruxeval_o; do
    until [ -f "artifacts/study2/main/$b/runs.csv.gz" ]; do sleep 60; done
  done
  echo "main component complete; starting post hoc corrections"
  python scripts/posthoc_s2_corrections.py --task fbfix --verify-frozen
  python scripts/posthoc_s2_corrections.py --task tiny
  python scripts/posthoc_s2_corrections.py --task fbfix --N 8
  python scripts/posthoc_s2_corrections.py --task fbfix --N 4
  python scripts/analyze_s2_corrections.py
  echo S2_POSTHOC_ALL_COMPLETE' >/dev/null
echo "started eost-s2fix (docker logs -f eost-s2fix)"
