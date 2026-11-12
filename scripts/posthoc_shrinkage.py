"""Exploratory experiment PH2 (plan in docs/RESEARCH_LOG.md): regularization path.

Same banks, targets, replicates, holdouts and calibration panels as the confirmatory
study, N = 8. The calibration reproduces the main analysis (joint MAP abilities and the
empirical-Bayes hyperprior, built from the frozen functions of eost.irt) except that the
prior standard deviation of log a is fixed to tau. For each tau the script replays
EIG under the marginal model and MFI under plug-in parameters. The main-analysis
EIG:mg is recomputed as a reference (it must equal the confirmatory rows).
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
from eost.experiment import ExpConfig, _seed, choose_targets  # noqa: E402
from eost.irt import (B_RANGE, LOG_A_RANGE, _softmax_rows, default_theta_grid,  # noqa: E402
                      empirical_bayes, fit_items_given_theta, fit_map, grid_loglik, grid_logprior,
                      make_item_grid, marginal_model, plugin_model)
from eost.replay import Precomputed, replay, summarize  # noqa: E402

OUT = "artifacts/posthoc_shrinkage"
TAUS = (0.05, 0.15, 0.30, 0.50)
N0 = 8
_B = _C = None


def _init(path, cfg):
    global _B, _C
    _B, _C = load_bank(path), cfg


def calibrate_shrunk(Y, priors, taus):
    """Main-analysis calibration with the EB prior, then one re-fit per fixed sd(log a) = tau."""
    cal0 = fit_map(Y, "2pl", priors)
    th = cal0.theta_cal
    grid = make_item_grid("2pl")
    coarse = make_item_grid("2pl", b_range=(B_RANGE[0], B_RANGE[1], 65),
                            log_a_range=(LOG_A_RANGE[0], LOG_A_RANGE[1], 14))
    pr, _ = empirical_bayes(grid_loglik(Y, th, coarse), coarse, priors, "2pl", iters=60, tol=1e-7)
    LL = grid_loglik(Y, th, grid)
    W_main = _softmax_rows(LL + grid_logprior(grid, pr, "2pl")[None, :])   # = eost.irt.calibrate's W
    out = {}
    for tau in taus:
        p = replace(pr, sd_log_a=float(tau))
        a, b = fit_items_given_theta(Y, th, p, "2pl", u0=np.log(cal0.a), b0=cal0.b)
        W = _softmax_rows(LL + grid_logprior(grid, p, "2pl")[None, :])
        out[tau] = (type("Cal", (), {"a": a, "b": b})(), W, grid, p)
    return out, pr, W_main, grid


def run_one_target(t):
    bank, cfg = _B, _C
    t0 = time.time()
    Y, I = bank.Y, bank.I
    y = Y[t].astype(np.int64)
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    elig = bank.eligible_calibrators(t, cfg.max_kappa)
    rows = []
    for r in range(cfg.replicates):
        rng_h = np.random.default_rng(_seed(cfg.master_seed, bank.name, "holdout", t, r))
        perm = rng_h.permutation(I)
        nh = int(round(cfg.holdout_frac * I))
        holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
        if N0 > len(elig):
            continue
        rng_p = np.random.default_rng(_seed(cfg.master_seed, bank.name, "panel", t, r, N0))
        panel = np.sort(rng_p.choice(elig, size=N0, replace=False))
        Yp = Y[panel].astype(float)
        shr, pr_eb, W, grid = calibrate_shrunk(Yp, cfg.priors, TAUS)
        runs = [("eig:mg", "eig", Precomputed.build(marginal_model(W, grid, theta), cfg.priors.sd_theta), None)]
        for tau, (c, Wt, g, p) in shr.items():
            runs.append((f"eig:mg@{tau:.2f}", "eig", Precomputed.build(marginal_model(Wt, g, theta), p.sd_theta), tau))
            runs.append((f"mfi:pi@{tau:.2f}", "mfi", Precomputed.build(plugin_model(c, theta), p.sd_theta), tau))
        for pol, crit, pre, tau in runs:
            rng_s = np.random.default_rng(_seed(cfg.master_seed, bank.name, "policy", t, r, N0, pol))
            out = replay(y, pre, pool, holdout, crit, cfg.budget, rng_s)
            s = summarize(out)
            s.update({"dataset": bank.name, "target": int(t), "rep": r, "N": N0, "policy": pol,
                      "tau": tau if tau is not None else float(pr_eb.sd_log_a),
                      "eb_sd_log_a": float(pr_eb.sd_log_a)})
            rows.append(s)
    return int(t), rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = ExpConfig()
    for name in a.datasets:
        path = f"data/processed/{name}.npz"
        bank = load_bank(path)
        targets = [int(t) for t in choose_targets(bank, cfg)]
        od = f"{OUT}/{name}"
        os.makedirs(f"{od}/parts", exist_ok=True)
        todo = [t for t in targets if not os.path.exists(f"{od}/parts/target_{t:04d}.json.gz")]
        print(f"[{name}] {len(targets) - len(todo)}/{len(targets)} targets already checkpointed", flush=True)
        if todo:
            with get_context("fork").Pool(a.workers, initializer=_init, initargs=(path, cfg)) as pool:
                for k, (t, rows, secs) in enumerate(pool.imap_unordered(run_one_target, todo)):
                    tmp = f"{od}/parts/target_{t:04d}.json.gz.tmp"
                    with gzip.open(tmp, "wt") as fh:
                        json.dump({"target": t, "seconds": secs, "rows": rows}, fh)
                    os.replace(tmp, f"{od}/parts/target_{t:04d}.json.gz")
                    print(f"[{name}] {len(targets) - len(todo) + k + 1}/{len(targets)}", flush=True)
        rows = []
        for t in targets:
            rows += json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))["rows"]
        pd.DataFrame(rows).to_csv(f"{od}/runs.csv.gz", index=False)
        json.dump({"dataset": name, "N": N0, "taus": list(TAUS), "n_targets": len(targets), "n_rows": len(rows)},
                  open(f"{od}/config.json", "w"), indent=1)
        print(f"PH2_RUN_GATE=PASS {name} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
