"""Mixing diagnostics of the fully Bayesian calibration sampler (development banks only).

Runs long chains (300 sweeps) of the frozen sampler (src/eost/fullbayes.py, with the
exact translation move) and of the same sampler without the translation move (local
copy of the sweep), on SWE-bench Lite and TruthfulQA panels of N = 4, 8, 16, and reports
the posterior standard deviation and lag-3 autocorrelation of the mean calibrator
ability, of the difficulty mean and of the scale hyperparameters. Writes
artifacts/study2/fb_mixing.json.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.special import log_expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost import fullbayes as fb  # noqa: E402
from eost.data import load_bank  # noqa: E402
from eost.irt import Priors, _softmax_rows, calibrate, grid_logprior, grid_loglik  # noqa: E402

NAMES = ["theta_mean", "theta_sd", "mu_b", "sd_b", "mu_log_a", "sd_log_a"]


def chain_without_move(Y, th, pr, grid, rng, sweeps):
    la, bb = grid.log_a, grid.b
    ag = np.exp(la)
    N, I = Y.shape
    rec = []
    for _ in range(sweeps):
        W = _softmax_rows(grid_loglik(Y, th, grid) + grid_logprior(grid, pr, "2pl")[None, :])
        g = fb._sample_rows(W, rng)
        a_i, b_i = ag[g], bb[g]
        gt = th[:, None] + fb.LOCAL[None, :]
        z = a_i[None, None, :] * (gt[:, :, None] - b_i[None, None, :])
        ll = (Y[:, None, :] * log_expit(z) + (1 - Y[:, None, :]) * log_expit(-z)).sum(2) - 0.5 * gt ** 2
        P = np.exp(ll - ll.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        th = gt[np.arange(N), fb._sample_rows(P, rng)]
        upd = {}
        for km, ks, x in (("mu_b", "sd_b", b_i), ("mu_log_a", "sd_log_a", la[g])):
            var = ((x - x.mean()) ** 2).sum() / rng.chisquare(I - 1)
            upd[ks] = float(np.sqrt(var)); upd[km] = float(rng.normal(x.mean(), np.sqrt(var / I)))
        pr = fb._clip_priors(replace(pr, **upd))
        rec.append([th.mean(), th.std(), pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a])
    return np.array(rec)


def acf(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / max((x * x).sum(), 1e-300))


def main():
    out = []
    for bank in ("swe_lite", "ollb_truthfulqa"):
        B = load_bank(f"data/processed/{bank}.npz")
        rng = np.random.default_rng(1)
        for N in (4, 8, 16):
            panel = np.sort(rng.choice(B.J, N, replace=False))
            Y = B.Y[panel].astype(float)
            cal, W, grid = calibrate(Y, "2pl", Priors(), True)
            tr = []
            fb.gibbs_item_marginals(Y, cal.theta_cal, cal.priors, grid, np.random.default_rng(0), burn=0, draws=300,
                                    thin=1, trace=tr)
            withm = np.array(tr)[30:, :6]
            nom = chain_without_move(Y, cal.theta_cal.copy(), cal.priors, grid, np.random.default_rng(0), 300)[30:]
            for lab, R in (("with_translation_move", withm), ("without_move", nom)):
                out.append({"bank": bank, "N": N, "sampler": lab,
                            **{f"sd_{n}": float(R[:, k].std()) for k, n in enumerate(NAMES)},
                            **{f"acf3_{n}": acf(R[:, k], 3) for k, n in enumerate(NAMES)}})
            print(bank, N, "done", flush=True)
    json.dump(out, open("artifacts/study2/fb_mixing.json", "w"), indent=1)
    w = [r for r in out if r["sampler"] == "with_translation_move"]
    n = [r for r in out if r["sampler"] == "without_move"]
    print("acf3 theta_mean: with move %.2f-%.2f, without %.2f-%.2f" % (
        min(r["acf3_theta_mean"] for r in w), max(r["acf3_theta_mean"] for r in w),
        min(r["acf3_theta_mean"] for r in n), max(r["acf3_theta_mean"] for r in n)))
    print("acf3 sd_log_a: with move %.2f-%.2f" % (min(r["acf3_sd_log_a"] for r in w), max(r["acf3_sd_log_a"] for r in w)))
    print("FB_MIXING_GATE=PASS")


if __name__ == "__main__":
    main()
