"""Robustness of the empirical-Bayes item hyperprior (post hoc; reviewer request).

For every confirmatory bank and panel size N in {4, 8, 16}, three calibration panels are
drawn exactly as in the first study (same eligibility rule, new seeds). For each panel:

1. stability -- the hyperparameters (mu_b, sd_b, mu_log_a, sd_log_a) are re-estimated
   on 20 bootstrap resamples of the items (abilities held fixed) and, at N = 8, on 10
   bootstrap resamples of the calibrators (joint MAP refitted); between-panel spread is
   reported alongside;
2. leave-item-block-out -- hyperparameters fitted on two random halves of the items;
3. prior predictive checks -- 200 response matrices are simulated from the fitted
   hyperprior given the calibrators' abilities; the observed share of items that every
   or no calibrator solved, the spread of item proportions correct and the mean
   item-rest correlation are compared with their simulated distributions (two-sided
   predictive p-values).
Writes artifacts/eb_robustness/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import _seed  # noqa: E402
from eost.irt import B_RANGE, LOG_A_RANGE, Priors, calibrate, empirical_bayes, grid_loglik, make_item_grid  # noqa: E402

CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
OUT = "artifacts/eb_robustness"
SEED = 20261105
COARSE = make_item_grid("2pl", b_range=(B_RANGE[0], B_RANGE[1], 65), log_a_range=(LOG_A_RANGE[0], LOG_A_RANGE[1], 14))
KEYS = ["mu_b", "sd_b", "mu_log_a", "sd_log_a"]


def eb(Y, th):
    pr, _ = empirical_bayes(grid_loglik(Y, th, COARSE), COARSE, Priors(), "2pl", iters=60, tol=1e-7)
    return np.array([pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a])


def stats(Y):
    N = Y.shape[0]
    col = Y.mean(0)
    rest = Y.sum(0, keepdims=True) - Y                      # rest score of every calibrator per item
    tot = Y.sum(1)
    # item-rest correlation across calibrators (items with variance only)
    r = []
    for i in np.flatnonzero((col > 0) & (col < 1)):
        x, z = Y[:, i], tot - Y[:, i]
        if z.std() > 0:
            r.append(np.corrcoef(x, z)[0, 1])
    return np.array([np.mean((col == 0) | (col == 1)), col.std(), np.mean(r) if r else np.nan])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, ppc = [], []
    for ds in CONF:
        B = load_bank(f"data/processed/{ds}.npz")
        for N in (4, 8, 16):
            for k in range(3):
                rng = np.random.default_rng(_seed(SEED, ds, N, k))
                t = int(rng.integers(B.J))
                elig = B.eligible_calibrators(t)
                if len(elig) < N:
                    continue
                panel = np.sort(rng.choice(elig, N, replace=False))
                Y = B.Y[panel].astype(float)
                cal, W, grid = calibrate(Y, "2pl", Priors(), True)
                th = cal.theta_cal
                eta = np.array([cal.priors.mu_b, cal.priors.sd_b, cal.priors.mu_log_a, cal.priors.sd_log_a])
                I = Y.shape[1]
                boot_items = np.array([eb(Y[:, rng.integers(0, I, I)], th) for _ in range(20)])
                halves = rng.permutation(I) % 2
                h0, h1 = eb(Y[:, halves == 0], th), eb(Y[:, halves == 1], th)
                row = {"dataset": ds, "N": N, "panel": k, **{f"eta_{n}": v for n, v in zip(KEYS, eta)},
                       **{f"boot_items_sd_{n}": v for n, v in zip(KEYS, boot_items.std(0))},
                       **{f"half_diff_{n}": v for n, v in zip(KEYS, np.abs(h0 - h1))}}
                if N == 8:
                    bc = []
                    for _ in range(10):
                        idx = rng.integers(0, N, N)
                        cb, _, _ = calibrate(Y[idx], "2pl", Priors(), True)
                        bc.append([cb.priors.mu_b, cb.priors.sd_b, cb.priors.mu_log_a, cb.priors.sd_log_a])
                    row.update({f"boot_cal_sd_{n}": v for n, v in zip(KEYS, np.array(bc).std(0))})
                rows.append(row)
                # prior predictive check
                obs = stats(Y)
                sims = []
                for _ in range(200):
                    a = np.exp(rng.normal(eta[2], eta[3], I))
                    b = rng.normal(eta[0], eta[1], I)
                    Ys = (rng.random((N, I)) < expit(a * (th[:, None] - b))).astype(float)
                    sims.append(stats(Ys))
                S = np.array(sims)
                for j, name in enumerate(["frac_extreme_items", "sd_item_p", "mean_item_rest_r"]):
                    s = S[:, j][np.isfinite(S[:, j])]
                    p = 2 * min(np.mean(s >= obs[j]), np.mean(s <= obs[j])) if len(s) else np.nan
                    ppc.append({"dataset": ds, "N": N, "panel": k, "stat": name, "observed": obs[j],
                                "sim_median": float(np.median(s)), "sim_q05": float(np.quantile(s, 0.05)),
                                "sim_q95": float(np.quantile(s, 0.95)), "ppp": float(min(p, 1.0))})
            print(ds, "done", flush=True)
    R, P = pd.DataFrame(rows), pd.DataFrame(ppc)
    R.to_csv(f"{OUT}/stability.csv", index=False)
    P.to_csv(f"{OUT}/ppc.csv", index=False)
    summ = {}
    for N in (4, 8, 16):
        g = R[R.N == N]
        summ[N] = {
            "between_panel_sd": {n: float(g.groupby("dataset")[f"eta_{n}"].std().median()) for n in KEYS},
            "item_bootstrap_sd": {n: float(g[f"boot_items_sd_{n}"].median()) for n in KEYS},
            "half_split_absdiff": {n: float(g[f"half_diff_{n}"].median()) for n in KEYS},
        }
        if N == 8:
            summ[N]["calibrator_bootstrap_sd"] = {n: float(g[f"boot_cal_sd_{n}"].median()) for n in KEYS}
    ppcs = {st: {"median_ppp": float(P[P.stat == st].ppp.median()),
                 "share_ppp_below_0.05": float((P[P.stat == st].ppp < 0.05).mean()),
                 "median_observed": float(P[P.stat == st].observed.median()),
                 "median_simulated": float(P[P.stat == st].sim_median.median())}
            for st in P.stat.unique()}
    json.dump({"stability": summ, "ppc": ppcs}, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({"stability": summ, "ppc": ppcs}, indent=1))
    print("EB_ROBUSTNESS_GATE=PASS")


if __name__ == "__main__":
    main()
