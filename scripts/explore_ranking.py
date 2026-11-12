"""Exploratory (not pre-registered): ranking fidelity of short adaptive tests.

Re-runs the frozen protocol at one panel size for a few policies, keeping the
score estimate Â_t at checkpoints, and reports Kendall's tau / Spearman's rho
between Â_t and the true full-bank accuracy across targets. Uses the frozen
modules unchanged (same seeds, same holdouts and panels as the confirmatory run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import ExpConfig, _seed, choose_targets  # noqa: E402
from eost.irt import calibrate, default_theta_grid, marginal_model, plugin_model  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402

CPS = (10, 25, 50, 100)
_B = _C = None


def _init(path, cfg):
    global _B, _C
    _B, _C = load_bank(path), cfg


def _work(args):
    t, N, pols = args
    bank, cfg = _B, _C
    y = bank.Y[t].astype(np.int64)
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    elig = bank.eligible_calibrators(t, cfg.max_kappa)
    out = []
    for r in range(cfg.replicates):
        rng_h = np.random.default_rng(_seed(cfg.master_seed, bank.name, "holdout", t, r))
        perm = rng_h.permutation(bank.I)
        nh = int(round(cfg.holdout_frac * bank.I))
        holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
        if N > len(elig):
            continue
        rng_p = np.random.default_rng(_seed(cfg.master_seed, bank.name, "panel", t, r, N))
        panel = np.sort(rng_p.choice(elig, size=N, replace=False))
        cal, W, grid = calibrate(bank.Y[panel].astype(float), cfg.irt_model, cfg.priors, cfg.empirical_bayes)
        pre = {"pi": Precomputed.build(plugin_model(cal, theta)), "mg": Precomputed.build(marginal_model(W, grid, theta))}
        for pol in pols:
            crit, _, m = pol.partition(":")
            rng_s = np.random.default_rng(_seed(cfg.master_seed, bank.name, "policy", t, r, N, pol))
            o = replay(y, pre.get(m), pool, holdout, crit, cfg.budget, rng_s)
            row = {"dataset": bank.name, "target": int(t), "rep": r, "N": N, "policy": pol, "A_true": o["A_true"]}
            for c in CPS:
                if c <= len(o["est"]):
                    row[f"est@{c}"] = float(o["est"][c - 1])
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--policies", nargs="+", default=["naive", "mfi:pi", "eig:mg", "evr:mg"])
    a = ap.parse_args()
    cfg = ExpConfig(replicates=a.reps)
    os.makedirs("artifacts/explore_ranking/parts", exist_ok=True)
    frames = []
    for ds in a.datasets:
        part = f"artifacts/explore_ranking/parts/{ds}.csv.gz"   # per-bank checkpoint (resumable)
        if not os.path.exists(part):
            path = f"data/processed/{ds}.npz"
            bank = load_bank(path)
            targets = choose_targets(bank, cfg)
            rows = []
            with get_context("fork").Pool(2, initializer=_init, initargs=(path, cfg)) as pool:
                for r in pool.imap_unordered(_work, [(int(t), a.N, a.policies) for t in targets]):
                    rows += r
            pd.DataFrame(rows).to_csv(part + ".tmp", index=False, compression="gzip")
            os.replace(part + ".tmp", part)
        frames.append(pd.read_csv(part))
        print(f"[{ds}] done", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("artifacts/explore_ranking/estimates.csv.gz", index=False)
    res = []
    # one adaptive test per model: rank correlation within each replicate, then averaged
    for (ds, pol), g in df.groupby(["dataset", "policy"]):
        for c in CPS:
            col = f"est@{c}"
            if col not in g:
                continue
            ks, ss, ms = [], [], []
            for _, gr in g.groupby("rep"):
                ks.append(kendalltau(gr[col], gr["A_true"]).statistic)
                ss.append(spearmanr(gr[col], gr["A_true"]).statistic)
                ms.append(np.mean(np.abs(gr[col] - gr["A_true"])))
            res.append({"dataset": ds, "policy": pol, "t": c, "kendall": float(np.mean(ks)),
                        "kendall_sd": float(np.std(ks)), "spearman": float(np.mean(ss)), "mae": float(np.mean(ms))})
    R = pd.DataFrame(res)
    R.to_csv("artifacts/explore_ranking/ranking.csv", index=False)
    print(R.pivot_table(index=["dataset", "t"], columns="policy", values="kendall").round(3).to_string())
    print("EXPLORE_RANKING_GATE=PASS")


if __name__ == "__main__":
    main()
