"""Run the leave-family-out adaptive-testing experiment on one or more banks.

Usage (inside the container):
    python scripts/run_benchmark.py --phase dev --datasets swe_lite --targets 20 --reps 2
Outputs go to artifacts/<phase>/<dataset>/ : runs.csv.gz, calib_diag.csv.gz,
config.json, and a RUN_GATE line on stdout. Any exception aborts with exit 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import ExpConfig, choose_targets, run_target  # noqa: E402
from eost.irt import Priors  # noqa: E402

_BANK = None
_CFG = None


def _init(path, cfg):
    global _BANK, _CFG
    _BANK, _CFG = load_bank(path), cfg


def _work(t):
    return run_target(_BANK, int(t), _CFG, {})


def run_one(name, cfg: ExpConfig, phase: str, workers: int):
    path = f"data/processed/{name}.npz"
    bank = load_bank(path)
    targets = choose_targets(bank, cfg)
    outdir = f"artifacts/{phase}/{name}"
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    rows, diag = [], []
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(path, cfg)) as pool:
        for k, (r, d) in enumerate(pool.imap_unordered(_work, targets)):
            rows += r; diag += d
            if (k + 1) % 5 == 0 or k + 1 == len(targets):
                print(f"[{name}] {k + 1}/{len(targets)} targets  {time.time() - t0:.0f}s", flush=True)
    runs = pd.DataFrame(rows)
    dg = pd.DataFrame(diag)
    runs.to_csv(f"{outdir}/runs.csv.gz", index=False)
    dg.to_csv(f"{outdir}/calib_diag.csv.gz", index=False)
    meta = {"dataset": name, "phase": phase, "config": cfg.as_dict(), "n_targets": int(len(targets)),
            "targets": [int(t) for t in targets], "n_rows": int(len(runs)),
            "seconds": round(time.time() - t0, 1), "bank_meta_source": bank.meta.get("source")}
    with open(f"{outdir}/config.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    expected = len(cfg.policies)
    per = runs.groupby(["target", "rep", "N"]).policy.nunique()
    ok = bool((per == expected).all()) and runs[["iae", "coverage"]].notna().all().all()
    print(f"RUN_GATE={'PASS' if ok else 'FAIL'} {name} rows={len(runs)} targets={len(targets)} "
          f"seconds={meta['seconds']}")
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--targets", type=int, default=60)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--panel-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--irt-model", default="2pl", choices=["2pl", "1pl"])
    ap.add_argument("--sd-log-a", type=float, default=1.0)
    ap.add_argument("--sd-b", type=float, default=2.0)
    ap.add_argument("--max-kappa", type=float, default=None)
    ap.add_argument("--fixed-prior", action="store_true", help="disable empirical-Bayes item hyperprior")
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = ExpConfig(panel_sizes=tuple(a.panel_sizes), replicates=a.reps, n_targets=a.targets,
                    budget=a.budget, irt_model=a.irt_model,
                    priors=Priors(sd_log_a=a.sd_log_a, sd_b=a.sd_b), max_kappa=a.max_kappa,
                    empirical_bayes=not a.fixed_prior)
    if a.policies:
        cfg.policies = tuple(a.policies)
    for name in a.datasets:
        run_one(name, cfg, a.phase, a.workers)


if __name__ == "__main__":
    main()
