"""Runner for study 2 (resumable, per-target checkpoints).

Components:
  main      -- the frozen confirmatory comparison on the eight banks of study 1;
  temporal  -- SWE-bench Verified, targets submitted in the latest quarter of dates,
               calibration panels from earlier submissions only;
  synthetic -- the three synthetic banks (secondary: known ground truth);
  dev       -- development banks only (used before the freeze to choose the
               recalibration rule and to time the runs).
Every container checks STUDY2_GATE (hashes of the frozen plan and code) before running,
except the dev component, which runs before the freeze.
Outputs: artifacts/study2/<component>/<bank>/{parts/target_XXXX.json.gz, runs.csv.gz,
diag.csv.gz, config.json}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.study2 import S2Config, choose_targets_s2, run_target_s2  # noqa: E402

CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
SYN = ["synth_2pl", "synth_guess", "synth_2dim"]
DEV = ["swe_lite", "cruxeval_i", "ollb_truthfulqa", "synth_2pl"]
_S = {}


def study1_targets(name):
    f = f"artifacts/confirm/{name}/config.json"
    return json.load(open(f))["targets"] if os.path.exists(f) else []


def temporal_split(bank):
    """Latest quarter of SWE-bench submissions by date prefix (ties broken by name)."""
    names = [str(s) for s in bank.systems]
    order = sorted(range(len(names)), key=lambda j: (names[j][:8], names[j]))
    n_late = int(np.ceil(0.25 * len(names)))
    late = np.array(sorted(order[-n_late:]))
    early = np.array(sorted(order[:-n_late]))
    return early, late


def _init(name, comp, cfg):
    B = load_bank(f"data/processed/{name}.npz")
    cl = np.array(json.load(open("configs/study2_clusters.json"))["banks"][name]["cluster"])
    allowed = temporal_split(B)[0] if comp == "temporal" else None
    _S.update(bank=B, clusters=cl, cfg=cfg, comp=comp, allowed=allowed)


def _work(t):
    t0 = time.time()
    rows, diags = run_target_s2(_S["bank"], _S["clusters"], t, _S["cfg"], _S["allowed"], _S["comp"])
    return t, rows, diags, time.time() - t0


def run_bank(name, comp, cfg, workers):
    B = load_bank(f"data/processed/{name}.npz")
    cl = np.array(json.load(open("configs/study2_clusters.json"))["banks"][name]["cluster"])
    if comp == "temporal":
        early, late = temporal_split(B)
        targets = choose_targets_s2(B, cl, cfg, avoid=(), eligible_systems=late)
        targets = np.array([t for t in targets if len(set(early) - set(np.flatnonzero(cl == cl[t])))
                            >= max(cfg.panel_sizes)], dtype=int)
    else:
        targets = choose_targets_s2(B, cl, cfg, avoid=study1_targets(name))
    od = f"artifacts/study2/{comp}/{name}"
    if os.path.exists(f"{od}/config.json"):
        print(f"[{comp}/{name}] already complete", flush=True)
        return
    os.makedirs(f"{od}/parts", exist_ok=True)
    todo = [int(t) for t in targets if not os.path.exists(f"{od}/parts/target_{int(t):04d}.json.gz")]
    print(f"[{comp}/{name}] {len(targets) - len(todo)}/{len(targets)} targets checkpointed", flush=True)
    t0 = time.time()
    if todo:
        with get_context("fork").Pool(workers, initializer=_init, initargs=(name, comp, cfg)) as pool:
            for k, (t, rows, diags, secs) in enumerate(pool.imap_unordered(_work, todo)):
                tmp = f"{od}/parts/target_{t:04d}.json.gz.tmp"
                with gzip.open(tmp, "wt") as fh:
                    json.dump({"target": t, "seconds": secs, "rows": rows, "diag": diags}, fh)
                os.replace(tmp, f"{od}/parts/target_{t:04d}.json.gz")
                print(f"[{comp}/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)}  {time.time() - t0:.0f}s",
                      flush=True)
    rows, diags, secs = [], [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{int(t):04d}.json.gz", "rt"))
        rows += p["rows"]; diags += p["diag"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/runs.csv.gz", index=False)
    pd.DataFrame(diags).to_csv(f"{od}/diag.csv.gz", index=False)
    json.dump({"dataset": name, "component": comp, "config": cfg.as_dict(), "targets": [int(t) for t in targets],
               "n_rows": len(rows), "cpu_seconds": round(secs, 1)}, open(f"{od}/config.json", "w"), indent=1)
    print(f"S2_RUN_GATE=PASS {comp}/{name} targets={len(targets)} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["main", "temporal", "synthetic", "dev"], required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--targets", type=int, default=None, help="dev only: override the number of targets")
    ap.add_argument("--replicates", type=int, default=None, help="dev only")
    ap.add_argument("--panel-sizes", type=int, nargs="*", default=None, help="dev only")
    a = ap.parse_args()
    if a.component != "dev":
        import subprocess
        if subprocess.run([sys.executable, "scripts/verify_artifacts.py", "--study2-only"]).returncode != 0:
            raise SystemExit("STUDY2_GATE failed: the frozen plan or code changed")
    cfg = S2Config()
    if a.component == "dev":
        cfg = replace(cfg, **{k: v for k, v in dict(n_targets=a.targets, replicates=a.replicates,
                                                       panel_sizes=tuple(a.panel_sizes) if a.panel_sizes else None).items()
                                if v is not None})
    elif a.targets or a.replicates or a.panel_sizes:
        raise SystemExit("overrides are allowed in the dev component only")
    if a.component == "synthetic":
        cfg = replace(cfg, n_targets=20)
    if a.component == "temporal":
        cfg = replace(cfg, panel_sizes=(8,), loco_N=(8,), fb_N=(8,), eb_arms_N=())
    names = a.datasets or {"main": CONF, "temporal": ["swe_verified"], "synthetic": SYN, "dev": DEV}[a.component]
    for name in names:
        run_bank(name, a.component, cfg, a.workers)


if __name__ == "__main__":
    main()
