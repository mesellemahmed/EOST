"""Robustness follow-up RF2 (plan: docs/RESEARCH_LOG.md, iteration 4, written before running).

MCMC quality of the Bayesian CAT of study 2. For 6 frozen study-2 targets per bank (fixed
seed), replicate 0, N = 8: four Gibbs chains (the post hoc copy of the frozen sampler, which
reproduces it exactly) -- chain 1 from EOST-M's estimates as in the frozen arm, chains 2-4 from
dispersed starting points -- with a burn-in of 250 sweeps and 100 retained draws per chain at
thinning 3 (400 retained draws). Records rank-normalized split R-hat, bulk ESS and MCSE for the
location (mean calibrator ability), mu_b, sigma_b, mu_a and sigma_a, the between-chain spread of
the held-out log loss of the posterior-draw marginal model, and replays

  pdm:bmi, pdm:eig   posterior-draw marginal Bayesian CAT (item marginals averaged over the 400
                     pooled draws; Bayesian maximum information / EIG)
  jfb:eig, jfb:bmi   joint fully Bayesian CAT (mixture over 40 draws, every 10th pooled draw)

Outputs: artifacts/robust/fb_convergence/<bank>/{parts,rows.csv.gz,traces.npz,done.json}.
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
from scipy.stats import norm, rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eost.data import load_bank  # noqa: E402
from eost.experiment import _seed  # noqa: E402
from eost.fullbayes import _clip_priors, expected_fisher  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402
from eost.replay2 import summarize2  # noqa: E402
from eost.study2 import S2Config, eligible_s2  # noqa: E402
from posthoc_s2_corrections import gibbs_with_states, joint_mixture_model  # noqa: E402

OUT = "artifacts/robust/fb_convergence"
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
CHAINS, BURN, DRAWS, THIN, K_TARGETS, N0, REP, JOINT_EVERY = 4, 250, 100, 3, 6, 8, 0, 10
PARAMS = {"loc": 0, "mu_b": 2, "sd_b": 3, "mu_a": 4, "sd_a": 5}   # columns of the sampler trace


# ----------------------------------------------------------------------------- diagnostics
def _rhat(y):
    M, N = y.shape
    B = N * y.mean(1).var(ddof=1)
    W = y.var(1, ddof=1).mean()
    return float(np.sqrt(((N - 1) / N * W + B / N) / W)) if W > 0 else float("nan")


def _split(x):
    h = x.shape[1] // 2
    return np.concatenate([x[:, :h], x[:, h:2 * h]], 0)


def _ranknorm(s):
    r = rankdata(s.ravel()).reshape(s.shape)
    return norm.ppf((r - 3 / 8) / (r.size + 1 / 4))


def rhat_rank(x):
    """Rank-normalized split R-hat (max of bulk and folded), Vehtari et al. (2021)."""
    s = _split(np.asarray(x, float))
    return max(_rhat(_ranknorm(s)), _rhat(_ranknorm(np.abs(s - np.median(s)))))


def _autocov(v):
    n = len(v)
    f = np.fft.rfft(v - v.mean(), n=2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


def ess(x, rank=True):
    """Bulk ESS (rank-normalized split chains) or ESS of the raw draws (rank=False), with
    Geyer's initial monotone sequence, as in Stan."""
    s = _split(np.asarray(x, float))
    z = _ranknorm(s) if rank else s
    M, N = z.shape
    ac = np.array([_autocov(z[m]) for m in range(M)])
    chain_var = ac[:, 0] * N / (N - 1)
    mean_var = chain_var.mean()
    var_plus = mean_var * (N - 1) / N + z.mean(1).var(ddof=1)
    if var_plus <= 0:
        return float("nan")
    rho = 1 - (mean_var - ac.mean(0)) / var_plus
    rho[0] = 1.0
    P, t = [], 0
    while t + 1 < N:
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        P.append(min(p, P[-1]) if P else p)
        t += 2
    tau = -1 + 2 * sum(P) if P else 1.0
    return float(M * N / max(tau, 1e-12))


def diagnostics(T):
    """T: chains x sweeps x columns (post burn-in)."""
    out = {}
    for k, col in PARAMS.items():
        x = T[:, :, col]
        e_bulk, e_mean = ess(x, True), ess(x, False)
        out[f"rhat_{k}"] = rhat_rank(x)
        out[f"ess_{k}"] = e_bulk
        out[f"mcse_{k}"] = float(x.std(ddof=1) / np.sqrt(max(e_mean, 1.0)))
        out[f"sd_{k}"] = float(x.std(ddof=1))
    return out


# ----------------------------------------------------------------------------- one panel
_S = {}


def _init(name):
    _S.update(bank=load_bank(f"data/processed/{name}.npz"),
              cl=np.array(json.load(open("configs/study2_clusters.json"))["banks"][name]["cluster"]),
              cfg=S2Config())


def _start(cal, rng, c):
    if c == 0:
        return cal.theta_cal.copy(), cal.priors
    pr = cal.priors
    th = cal.theta_cal + rng.normal(0.0, 0.5, size=len(cal.theta_cal))
    pr = replace(pr, mu_b=pr.mu_b + rng.choice([-0.5, 0.5]), sd_b=pr.sd_b * rng.choice([0.7, 1.4]),
                 mu_log_a=pr.mu_log_a + rng.choice([-0.3, 0.3]), sd_log_a=pr.sd_log_a * rng.choice([0.7, 1.4]))
    return th, _clip_priors(pr)


def _pre_bmi(W, grid, theta):
    p = Precomputed.build(marginal_model(W, grid, theta), 1.0)
    p.FI = expected_fisher(W, grid, theta)
    return p


def _work(t):
    B, cl, cfg = _S["bank"], _S["cl"], _S["cfg"]
    y = B.Y[t].astype(np.int64)
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    t0 = time.time()
    r = REP
    rng_h = np.random.default_rng(_seed(cfg.master_seed, B.name, "main", "holdout", t, r))
    perm = rng_h.permutation(B.I)
    nh = int(round(cfg.holdout_frac * B.I))
    holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
    elig = eligible_s2(B, cl, t, None)
    rng_p = np.random.default_rng(_seed(cfg.master_seed, B.name, "main", "panel", t, r, N0))
    panel = np.sort(rng_p.choice(elig, size=N0, replace=False))
    Yp = B.Y[panel].astype(float)
    sd = lambda *k: _seed(cfg.master_seed, B.name, "main", *k, t, r, N0)
    cal, W, grid = calibrate(Yp, "2pl", Priors(), True)
    Ws, states, traces = [], [], []
    for c in range(CHAINS):
        rng_c = np.random.default_rng(sd("rf2_chain", c))
        th0, pr0 = _start(cal, rng_c, c)
        tr = []
        Wc, st = gibbs_with_states(Yp, th0, pr0, grid, rng_c, BURN, DRAWS, THIN, trace=tr)
        Ws.append(Wc); states += st; traces.append(np.array(tr)[BURN:])
    T = np.stack(traces)                                   # chains x sweeps x 7
    diag = diagnostics(T)
    W_pool = np.mean(Ws, 0)
    base = {"dataset": B.name, "target": int(t), "rep": r, "N": N0}
    rows = []
    # between-chain spread of the held-out log loss (posterior-draw marginal model, BMI)
    ll_chain = []
    for c in range(CHAINS):
        out = replay(y, _pre_bmi(Ws[c], grid, theta), pool, holdout, "pwfi", cfg.budget,
                     np.random.default_rng(sd("policy", "rf2:pdm:bmi")))
        ll_chain.append(summarize2(out, None)["ilogloss"])
    diag["ll_chain_spread"] = float(max(ll_chain) - min(ll_chain))
    diag["ll_chain_spread_rel"] = float((max(ll_chain) - min(ll_chain)) / np.mean(ll_chain))
    m_pool = marginal_model(W_pool, grid, theta)
    sub = states[::JOINT_EVERY]
    mj, M = joint_mixture_model(Yp, sub, grid, theta)
    pre_j = Precomputed.build(mj, 1.0)
    pre_j.log_prior = pre_j.log_prior - np.log(M)
    arms = [("pdm:bmi", "pwfi", _pre_bmi(W_pool, grid, theta)), ("pdm:eig", "eig", Precomputed.build(m_pool, 1.0)),
            ("jfb:eig", "eig", pre_j), ("jfb:bmi", "pwfi", pre_j)]
    for pol, crit, pre in arms:
        out = replay(y, pre, pool, holdout, crit, cfg.budget, np.random.default_rng(sd("policy", "rf2:" + pol)))
        rows.append(base | {"policy": pol} | summarize2(out, None) | diag)
    return int(t), rows, time.time() - t0, T.astype(np.float32)


def pick_targets(name):
    cfg = S2Config()
    targets = json.load(open(f"artifacts/study2/main/{name}/config.json"))["targets"]
    rng = np.random.default_rng(_seed(cfg.master_seed, name, "rf2_targets"))
    return sorted(int(t) for t in rng.choice(targets, size=min(K_TARGETS, len(targets)), replace=False))


def run(name, workers):
    od = f"{OUT}/{name}"
    if os.path.exists(f"{od}/done.json"):
        print(f"[rf2/{name}] complete", flush=True)
        return
    targets = pick_targets(name)
    os.makedirs(f"{od}/parts", exist_ok=True)
    todo = [t for t in targets if not os.path.exists(f"{od}/parts/target_{t:04d}.json.gz")]
    if todo:
        with get_context("fork").Pool(workers, initializer=_init, initargs=(name,)) as pool:
            for k, (t, rows, secs, T) in enumerate(pool.imap_unordered(_work, todo)):
                np.savez_compressed(f"{od}/parts/traces_{t:04d}.npz", T=T)
                tmp = f"{od}/parts/target_{t:04d}.json.gz.tmp"
                with gzip.open(tmp, "wt") as fh:
                    json.dump({"target": t, "seconds": secs, "rows": rows}, fh)
                os.replace(tmp, f"{od}/parts/target_{t:04d}.json.gz")
                print(f"[rf2/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)} ({secs:.0f}s)", flush=True)
    rows, secs = [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))
        rows += p["rows"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/rows.csv.gz", index=False)
    json.dump({"dataset": name, "targets": targets, "chains": CHAINS, "burn": BURN, "draws": DRAWS, "thin": THIN,
               "n_rows": len(rows), "cpu_seconds": round(secs, 1)}, open(f"{od}/done.json", "w"), indent=1)
    print(f"RF2_GATE=PASS {name} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CONF)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for n in a.datasets:
        run(n, a.workers)
    print("RF2_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
