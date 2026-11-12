#!/usr/bin/env bash
# Host-side orchestration only: run the frozen study 2 in one detached container
# (main -> temporal -> synthetic). Resumable: completed banks/targets are skipped,
# so the same command resumes after an interruption. Log: artifacts/logs/study2.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
docker rm -f eost-study2 >/dev/null 2>&1 || true
docker run -d --name eost-study2 --network none --cpus 2 -v "$PWD":/work -w /work eost-runtime:cpu bash -c \
  "python scripts/run_study2.py --component main && python scripts/run_study2.py --component temporal && \
   python scripts/run_study2.py --component synthetic && echo STUDY2_ALL_COMPLETE" \
  >/dev/null
echo "started eost-study2 (docker logs -f eost-study2)"
