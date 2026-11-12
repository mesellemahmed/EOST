"""Integrity gates.

--freeze       write configs/PREREGISTRATION.lock.json (hashes of the plan and of
               analysis-relevant code) -- run once, before confirmatory runs.
--prereg-only  check that the frozen plan and code are unchanged (PREREG_GATE).
(default)      PREREG_GATE + data manifest + per-phase run gates + artifact
               hash manifest written to artifacts/ARTIFACT_MANIFEST.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.safeio import sha256_file  # noqa: E402

LOCK = "configs/PREREGISTRATION.lock.json"
FROZEN = ["configs/preregistration.md", "configs/hypotheses.json", "src/eost/irt.py", "src/eost/replay.py",
          "src/eost/experiment.py", "src/eost/data.py", "src/eost/stats.py", "scripts/run_benchmark.py",
          "scripts/analyze.py", "scripts/build_datasets.py", "data/processed/MANIFEST.json",
          "data/processed/SYNTH_MANIFEST.json"]


LOCK2 = "configs/STUDY2.lock.json"
FROZEN2 = ["configs/study2_plan.md", "configs/study2_hypotheses.json", "configs/study2_clusters.json",
           "src/eost/study2.py", "src/eost/replay2.py", "src/eost/irt2.py", "src/eost/fullbayes.py",
           "src/eost/reduction.py", "src/eost/families.py", "scripts/run_study2.py", "scripts/analyze_study2.py",
           "src/eost/irt.py", "src/eost/replay.py", "src/eost/experiment.py", "src/eost/data.py",
           "src/eost/stats.py", "data/processed/MANIFEST.json", "data/processed/SYNTH_MANIFEST.json"]


def freeze_study2():
    if os.path.exists(LOCK2):
        raise SystemExit("STUDY2 lock already exists; refusing to overwrite")
    hyp = json.load(open("configs/study2_hypotheses.json"))
    if "TO_BE_FIXED" in hyp.get("recalibration_rule", ""):
        raise SystemExit("recalibration rule not fixed; refusing to freeze")
    lock = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "files": {f: sha256_file(f) for f in FROZEN2}}
    with open(LOCK2, "w") as fh:
        json.dump(lock, fh, indent=1)
    print("STUDY2_FROZEN", lock["frozen_at_utc"])


def check_study2() -> bool:
    if not os.path.exists(LOCK2):
        print("STUDY2_GATE=FAIL (no lock)")
        return False
    lock = json.load(open(LOCK2))
    bad = [f for f, h in lock["files"].items() if not os.path.exists(f) or sha256_file(f) != h]
    print(f"STUDY2_GATE={'PASS' if not bad else 'FAIL'} frozen_at={lock['frozen_at_utc']} changed={bad}")
    return not bad


def freeze():
    if os.path.exists(LOCK):
        raise SystemExit("PREREG lock already exists; refusing to overwrite")
    lock = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "files": {f: sha256_file(f) for f in FROZEN}}
    with open(LOCK, "w") as fh:
        json.dump(lock, fh, indent=1)
    print("PREREG_FROZEN", lock["frozen_at_utc"])


def check_prereg() -> bool:
    if not os.path.exists(LOCK):
        print("PREREG_GATE=FAIL (no lock)")
        return False
    lock = json.load(open(LOCK))
    bad = [f for f, h in lock["files"].items() if not os.path.exists(f) or sha256_file(f) != h]
    print(f"PREREG_GATE={'PASS' if not bad else 'FAIL'} frozen_at={lock['frozen_at_utc']} changed={bad}")
    return not bad


def check_all() -> bool:
    ok = check_prereg()
    if os.path.exists(LOCK2):
        ok &= check_study2()
    man = json.load(open("data/processed/MANIFEST.json"))
    bad = [k for k, v in man["datasets"].items() if sha256_file(v["file"]) != v["sha256"]]
    print(f"DATA_GATE={'PASS' if not bad else 'FAIL'} datasets={len(man['datasets'])} mismatched={bad}")
    ok &= not bad
    files = {}
    for f in sorted(glob.glob("artifacts/**/*", recursive=True)):
        if os.path.isfile(f) and not f.endswith("ARTIFACT_MANIFEST.json"):
            files[f] = sha256_file(f)
    for cfgf in sorted(glob.glob("artifacts/*/*/config.json")) + sorted(glob.glob("artifacts/study2/*/*/config.json")):
        c = json.load(open(cfgf))
        if c.get("n_rows", 0) <= 0:
            print("RUN_GATE=FAIL", cfgf)
            ok = False
    with open("artifacts/ARTIFACT_MANIFEST.json", "w") as fh:
        json.dump({"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "files": files}, fh, indent=1)
    print(f"ARTIFACT_MANIFEST files={len(files)}")
    print(f"VERIFY_GATE={'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--prereg-only", action="store_true")
    ap.add_argument("--freeze-study2", action="store_true")
    ap.add_argument("--study2-only", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        sys.exit(0)
    if a.freeze_study2:
        freeze_study2()
        sys.exit(0)
    if a.study2_only:
        sys.exit(0 if check_study2() else 1)
    sys.exit(0 if (check_prereg() if a.prereg_only else check_all()) else 1)
