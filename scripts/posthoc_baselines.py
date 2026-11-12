"""Post hoc experiment PH1 (not pre-registered; plan in docs/RESEARCH_LOG.md): strong baselines.

Same banks, targets, replicates, holdouts and calibration panels as the confirmatory
study (identical seeds), panel sizes N in {4, 8, 16}. Frozen modules are used unchanged:

* eig:mg, mfi:pi    -- recomputed exactly as in the confirmatory run (reproduction check);
* efi:mg            -- posterior-expected Fisher information E_psi[I_psi(theta)] (averaged
                       over the item posterior, then posterior-weighted over theta), ERF updates;
                       implemented by replacing the Fisher matrix of the marginal model and
                       calling the frozen 'pwfi' criterion;
* mfi:eap, eig:eap  -- plug-in 2PL with posterior-mean item parameters from the same posterior;
* mfi:1pr, eig:1prmg -- one-parameter model with log a ~ N(0, 0.05^2) and EB prior on b:
                       plug-in MFI (Rasch-type CAT) and calibration-marginal EIG.

Per-target checkpoints: artifacts/posthoc_baselines/<bank>/parts/target_XXXX.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import H0_STD_NORMAL, ExpConfig, _seed, choose_targets  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model, plugin_model  # noqa: E402
from eost.replay import Precomputed, reference_theta, replay, summarize  # noqa: E402

OUT = "artifacts/posthoc_baselines"
PANELS = (4, 8, 16)
PRIOR_1PR = Priors(sd_log_a=0.05, sd_b=2.0)
_B = _C = None


def _init(path, cfg):
    global _B, _C
    _B, _C = load_bank(path), cfg


def expected_fisher(W, grid, theta):
    """E_psi[I_psi(theta)] under the item grid posterior W (I x G): one GEMM."""
    a = np.exp(grid.log_a)
    S = expit(a[:, None] * (theta[None, :] - grid.b[:, None]))       # (G, K)
    return W @ (a[:, None] ** 2 * S * (1.0 - S))                       # (I, K)


def eap_calibration(W, grid):
    """Posterior-mean item parameters (a, b) from the grid posterior."""
    return SimpleNamespace(a=W @ np.exp(grid.log_a), b=W @ grid.b)


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
        for N in PANELS:
            if N > len(elig):
                continue
            rng_p = np.random.default_rng(_seed(cfg.master_seed, bank.name, "panel", t, r, N))
            panel = np.sort(rng_p.choice(elig, size=N, replace=False))
            Yp = Y[panel].astype(float)
            # 2PL calibration, exactly as in the confirmatory run
            cal, W, grid = calibrate(Yp, cfg.irt_model, cfg.priors, cfg.empirical_bayes)
            m_pi, m_mg = plugin_model(cal, theta), marginal_model(W, grid, theta)
            pre = {"pi": Precomputed.build(m_pi, cfg.priors.sd_theta),
                   "mg": Precomputed.build(m_mg, cfg.priors.sd_theta),
                   "eap": Precomputed.build(plugin_model(eap_calibration(W, grid), theta), cfg.priors.sd_theta)}
            efi = Precomputed.build(m_mg, cfg.priors.sd_theta)
            efi.FI = expected_fisher(W, grid, theta)
            pre["efimg"] = efi
            # regularized one-parameter calibration
            cal1, W1, grid1 = calibrate(Yp, "1pl", PRIOR_1PR, True)
            pre["1pr"] = Precomputed.build(plugin_model(cal1, theta), PRIOR_1PR.sd_theta)
            pre["1prmg"] = Precomputed.build(marginal_model(W1, grid1, theta), PRIOR_1PR.sd_theta)
            policies = [("eig:mg", "eig", "mg"), ("mfi:pi", "mfi", "pi"), ("efi:mg", "pwfi", "efimg"),
                        ("mfi:eap", "mfi", "eap"), ("eig:eap", "eig", "eap"), ("mfi:1pr", "mfi", "1pr"),
                        ("eig:1prmg", "eig", "1prmg")]
            for pol, crit, key in policies:
                rng_s = np.random.default_rng(_seed(cfg.master_seed, bank.name, "policy", t, r, N, pol))
                out = replay(y, pre[key], pool, holdout, crit, cfg.budget, rng_s)
                s = summarize(out)
                ref_m, _ = reference_theta(y, pre[key], pool)
                th = out["theta_mean"]
                s["iref_err"] = float(np.mean(np.abs(th - ref_m)))
                s["final_ref_err"] = float(abs(th[-1] - ref_m))
                s["iinfo"] = float(np.mean(H0_STD_NORMAL - out["post_H"]))
                s["ref_theta"] = ref_m
                s.update({"dataset": bank.name, "target": int(t), "system": str(bank.systems[t]),
                          "rep": r, "N": N, "policy": pol, "common_a_1pr": float(cal1.a[0])})
                rows.append(s)
    return int(t), rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = ExpConfig()          # the confirmatory configuration (targets, replicates, seeds)
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
        json.dump({"dataset": name, "panel_sizes": list(PANELS), "n_targets": len(targets), "n_rows": len(rows),
                   "prior_1pr": {"sd_log_a": PRIOR_1PR.sd_log_a, "sd_b": PRIOR_1PR.sd_b}},
                  open(f"{od}/config.json", "w"), indent=1)
        print(f"PH1_RUN_GATE=PASS {name} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
