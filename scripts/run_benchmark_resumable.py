"""Checkpointed runner: same computation as scripts/run_benchmark.py, resumable per target.

It calls the frozen functions (eost.experiment.choose_targets / run_target) with
the same configuration and seeds, but writes every finished target to
artifacts/<phase>/<bank>/parts/target_<t>.json.gz, so an infrastructure restart
only loses the targets in flight. When every target of a bank is present, the
parts are assembled into runs.csv.gz, calib_diag.csv.gz and config.json with
the same content as the non-resumable runner (row order aside), and the same
RUN_GATE check is applied.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import get_context

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import ExpConfig, choose_targets, run_target  # noqa: E402
from eost.irt import Priors  # noqa: E402

_BANK = None
_CFG = None
EVR_GUARD = False


def _init(path, cfg):
    global _BANK, _CFG
    _BANK, _CFG = load_bank(path), cfg


def _work(t):
    t0 = time.time()
    rows, diag = run_target(_BANK, int(t), _CFG, {})
    return int(t), rows, diag, time.time() - t0


def install_evr_guard():
    """Numerical guard for the score-targeted (EVR) criterion -- post-freeze, documented deviation.

    Expected response probabilities are computed as W @ S and can exceed 1 (or fall below 0)
    by floating-point rounding when an item's response curve is saturated. The frozen EVR
    computation (eost.replay._expected_posterior_var) then forms a negative outcome weight,
    divides it by a floored normaliser and overflows, which stops the run (observed only in the
    one-parameter sensitivity analysis on HellaSwag). The guard clips the candidate
    probabilities to [0, 1] before calling the frozen function; it is an exact no-op whenever
    they already lie in [0, 1]. The frozen module itself is not modified.
    """
    import numpy as np
    import eost.replay as R
    frozen = R._expected_posterior_var

    def guarded(Pc, PQc, pi, S_un, V_un):
        return frozen(np.clip(Pc, 0.0, 1.0), PQc, pi, S_un, V_un)

    R._expected_posterior_var = guarded


def _part(outdir, t):
    return os.path.join(outdir, "parts", f"target_{t:04d}.json.gz")


def run_one(name, cfg: ExpConfig, phase: str, workers: int):
    path = f"data/processed/{name}.npz"
    bank = load_bank(path)
    targets = [int(t) for t in choose_targets(bank, cfg)]
    outdir = f"artifacts/{phase}/{name}"
    if os.path.exists(f"{outdir}/config.json"):
        print(f"[{name}] already complete", flush=True)
        return
    os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)
    todo = [t for t in targets if not os.path.exists(_part(outdir, t))]
    print(f"[{name}] {len(targets) - len(todo)}/{len(targets)} targets already checkpointed", flush=True)
    t0 = time.time()
    if todo:
        with get_context("fork").Pool(workers, initializer=_init, initargs=(path, cfg)) as pool:
            for k, (t, rows, diag, secs) in enumerate(pool.imap_unordered(_work, todo)):
                tmp = _part(outdir, t) + ".tmp"
                with gzip.open(tmp, "wt") as fh:
                    json.dump({"target": t, "seconds": secs, "rows": rows, "diag": diag}, fh)
                os.replace(tmp, _part(outdir, t))          # atomic: a part is either complete or absent
                print(f"[{name}] {len(targets) - len(todo) + k + 1}/{len(targets)} targets  {time.time() - t0:.0f}s",
                      flush=True)
    rows, diag, secs = [], [], 0.0
    for t in targets:
        with gzip.open(_part(outdir, t), "rt") as fh:
            p = json.load(fh)
        rows += p["rows"]; diag += p["diag"]; secs += p["seconds"]
    runs, dg = pd.DataFrame(rows), pd.DataFrame(diag)
    runs.to_csv(f"{outdir}/runs.csv.gz", index=False)
    dg.to_csv(f"{outdir}/calib_diag.csv.gz", index=False)
    meta = {"dataset": name, "phase": phase, "config": cfg.as_dict(), "n_targets": len(targets),
            "targets": targets, "n_rows": int(len(runs)), "seconds": round(secs / max(workers, 1), 1),
            "bank_meta_source": bank.meta.get("source"), "runner": "run_benchmark_resumable (per-target checkpoints)",
            "evr_guard": bool(EVR_GUARD)}
    per = runs.groupby(["target", "rep", "N"]).policy.nunique()
    ok = bool((per == len(cfg.policies)).all()) and runs[["iae", "coverage"]].notna().all().all()
    if ok:
        with open(f"{outdir}/config.json", "w") as fh:
            json.dump(meta, fh, indent=1)
    print(f"RUN_GATE={'PASS' if ok else 'FAIL'} {name} rows={len(runs)} targets={len(targets)} "
          f"seconds={meta['seconds']}", flush=True)
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
    ap.add_argument("--fixed-prior", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--evr-guard", action="store_true",
                    help="clip expected response probabilities to [0,1] inside the EVR criterion (see install_evr_guard)")
    a = ap.parse_args()
    if a.evr_guard:
        global EVR_GUARD
        EVR_GUARD = True
        install_evr_guard()
    cfg = ExpConfig(panel_sizes=tuple(a.panel_sizes), replicates=a.reps, n_targets=a.targets,
                    budget=a.budget, irt_model=a.irt_model,
                    priors=Priors(sd_log_a=a.sd_log_a, sd_b=a.sd_b), max_kappa=a.max_kappa,
                    empirical_bayes=not a.fixed_prior)
    for name in a.datasets:
        run_one(name, cfg, a.phase, a.workers)


if __name__ == "__main__":
    main()
