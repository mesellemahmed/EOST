#!/usr/bin/env bash
# Host-side orchestration only: sparse git clones of the pinned public sources.
# No Python is executed on the host. Parsing happens in the container.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

fetch() {  # fetch <dir> <repo-url> <commit> <sparse-pattern>...
  local dir="$1" url="$2" commit="$3"; shift 3
  local dest="$RAW/$dir"
  if [ ! -d "$dest/.git" ]; then
    git clone --quiet --filter=blob:none --no-checkout "$url" "$dest"
  fi
  git -C "$dest" sparse-checkout init --no-cone >/dev/null
  printf '%s\n' "$@" > "$dest/.git/info/sparse-checkout"
  git -C "$dest" fetch --quiet --depth 1 origin "$commit" || true
  git -C "$dest" checkout --quiet "$commit"
  local got; got="$(git -C "$dest" rev-parse HEAD)"
  if [ "$got" != "$commit" ]; then
    echo "FETCH_GATE=FAIL $dir expected $commit got $got" >&2; exit 1
  fi
  echo "fetched $dir @ $commit"
}

fetch swe_exp https://github.com/SWE-bench/experiments \
  40f164d5b8f1d249bf95a6df8b74b577fd8e519d \
  '/evaluation/*/*/results/results.json' '/evaluation/*/*/metadata.yaml' '/evaluation/*/*/per_instance_details.json'
fetch felipemaiapolo_tinyBenchmarks https://github.com/felipemaiapolo/tinyBenchmarks \
  e9a8b1031b0340571beb6c9ca3a27891be09a8fd '/tutorials/data/lb.pickle' '/LICENSE'
fetch LiveCodeBench_livecodebench.github.io https://github.com/LiveCodeBench/livecodebench.github.io \
  f186ab5041d6b768733b8ae8145151c867942134 '/build/performances_generation.json'
fetch facebookresearch_cruxeval https://github.com/facebookresearch/cruxeval \
  190faf16d175b5847b0af05d937872b1fb395942 '/samples/evaluation_results.zip' '/LICENSE' '/data/cruxeval.jsonl'
# item texts for the live (prospective) protocol only (scripts/live_eost.py)
fetch openai_grade-school-math https://github.com/openai/grade-school-math \
  3101c7d5072418e28b9008a6636bde82a006892c '/grade_school_math/data/test.jsonl' \
  '/grade_school_math/data/train.jsonl' '/LICENSE'
echo "FETCH_GATE=PASS"
