"""Exploratory follow-up RF2b (plan: docs/RESEARCH_LOG.md, iteration 4; written after RF2's
convergence diagnostics had been seen and before this script was run).

RF2 found that the Gibbs sampler of the Bayesian CATs converges for the overall location
(it has an exact translation move) but not for sigma_b, mu_a, sigma_a and the ability scale,
which move together along a ridge of the posterior: multiplying every ability and difficulty
by s and dividing every discrimination by s leaves the likelihood unchanged, and with eight
calibrators only their N(0, 1) prior pins down the scale. RF2b adds the missing move.

At the start of every sweep, a Metropolis-Hastings move along the scale ridge acts on the
posterior of (abilities, hyperparameters) with the item parameters summed out on the grid:

    theta -> s theta,  mu_b -> s mu_b,  sigma_b -> s sigma_b,  mu_a -> mu_a - log s,

with log s = +-0.2 or +-0.4 (multiples of the log-discrimination grid step, so the move maps
that grid onto itself). The collapsed target is the joint posterior implied by the frozen
sampler's conditionals -- abilities N(0, 1), flat priors on the means, Jeffreys priors on the
standard deviations, item parameters with Gaussian densities evaluated on the grid -- and the
acceptance ratio contains the Jacobian s^(N + 2). The item parameters are then drawn from
their exact conditional given the (possibly moved) state, so the scheme is a valid partially
collapsed Gibbs sampler. Everything else is the RF2 sampler: same targets, panels, holdouts,
starting points (chain 1 at EOST-M's estimates, chains 2-4 dispersed) and replay arms.

Outputs: artifacts/robust/fb_scale/<bank>/{parts,rows.csv.gz,done.json}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import log_expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eost.experiment import _seed  # noqa: E402
from eost.fullbayes import B_STEP, LOCAL, NB, SHIFT_STEPS, _clip_priors, _sample_rows  # noqa: E402
from eost.irt import Priors, _softmax_rows, calibrate, default_theta_grid, grid_logprior, grid_loglik  # noqa: E402
from eost.irt import marginal_model  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402
from eost.replay2 import summarize2  # noqa: E402
from eost.study2 import eligible_s2  # noqa: E402
from fb_convergence import CONF, _S, _init, _pre_bmi, _start, ess, pick_targets, rhat_rank  # noqa: E402
from posthoc_s2_corrections import joint_mixture_model  # noqa: E402

OUT = "artifacts/robust/fb_scale"
CHAINS, BURN, DRAWS, THIN, N0, REP, JOINT_EVERY = 4, 200, 100, 2, 8, 0, 10
SCALE_STEP = 0.2                     # = step of the log-discrimination grid (irt.LOG_A_RANGE)
SCALE_K = np.array([-2, -1, 1, 2])
PARAMS = {"loc": 0, "scale": 1, "mu_b": 2, "sd_b": 3, "mu_a": 4, "sd_a": 5}
LOG2PI = float(np.log(2 * np.pi))


# ----------------------------------------------------------------------------- collapsed target
def collapsed_terms(A, grid, pr):
    """Item conditionals for the item log likelihoods A (I x G, overwritten) under Gaussian
    item densities evaluated on the grid: unnormalized rows X, row sums S and per-item log
    normalizers (X / S is the exact conditional; sampling does not need the division)."""
    lp = (-0.5 * ((grid.b - pr.mu_b) / pr.sd_b) ** 2 - np.log(pr.sd_b)
          - 0.5 * ((grid.log_a - pr.mu_log_a) / pr.sd_log_a) ** 2 - np.log(pr.sd_log_a) - LOG2PI)
    E = A
    E += lp[None, :]
    m = E.max(1, keepdims=True)
    E -= m
    np.exp(E, out=E)
    S = E.sum(1, keepdims=True)
    return E, S, (m + np.log(S)).ravel()


def log_target(th, pr, lse, sd_theta=1.0):
    """Log collapsed posterior of (abilities, hyperparameters), up to a constant."""
    return float(lse.sum() - 0.5 * np.sum((th / sd_theta) ** 2) - np.log(pr.sd_b) - np.log(pr.sd_log_a))


def in_box(pr):
    return (-6 <= pr.mu_b <= 6 and 0.3 <= pr.sd_b <= 4.0 and -1.5 <= pr.mu_log_a <= 1.5
            and 0.1 <= pr.sd_log_a <= 1.5)


def scale_proposal(th, pr, k):
    ls = float(k) * SCALE_STEP
    s = np.exp(ls)
    return s * th, replace(pr, mu_b=s * pr.mu_b, sd_b=s * pr.sd_b, mu_log_a=pr.mu_log_a - ls), ls


def scale_move(Y, th, pr, grid, X, S, lse, rng, sd_theta=1.0):
    k = int(rng.choice(SCALE_K))
    th2, pr2, ls = scale_proposal(th, pr, k)
    u = np.log(rng.random())
    if not in_box(pr2):
        return th, pr, X, S, lse, False
    X2, S2, lse2 = collapsed_terms(grid_loglik(Y, th2, grid), grid, pr2)
    log_acc = log_target(th2, pr2, lse2, sd_theta) - log_target(th, pr, lse, sd_theta) + (len(th) + 2) * ls
    if u < log_acc:
        return th2, pr2, X2, S2, lse2, True
    return th, pr, X, S, lse, False


# ----------------------------------------------------------------------------- sampler
def gibbs_scale(Y, theta0, pr0, grid, rng, burn, draws, thin, sd_theta=1.0, trace=None, scale=True):
    """RF2's sampler (posthoc_s2_corrections.gibbs_with_states) plus the collapsed scale move;
    with scale=False it is that sampler exactly (tests/test_rf2b.py)."""
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    th = np.array(theta0, dtype=float)
    pr = pr0
    la, bb = grid.log_a, grid.b
    a_grid = np.exp(la)
    W_bar = np.zeros((I, len(bb)))
    states, n_acc, n_prop = [], 0, 0
    for s_ in range(burn + draws * thin):
        A = grid_loglik(Y, th, grid)
        keep = s_ >= burn and (s_ - burn) % thin == thin - 1
        if scale:
            X, S, lse = collapsed_terms(A, grid, pr)
            th, pr, W, S, lse, acc = scale_move(Y, th, pr, grid, X, S, lse, rng, sd_theta)
            n_acc += int(acc); n_prop += 1
            if keep:
                W_bar += W / S
        else:
            W = _softmax_rows(A + grid_logprior(grid, pr, "2pl")[None, :])
            if keep:
                W_bar += W
        if keep:
            states.append((th.copy(), pr))
        g = _sample_rows(W, rng)
        a_i, b_i = a_grid[g], bb[g]
        grid_th = th[:, None] + LOCAL[None, :]
        z = a_i[None, None, :] * (grid_th[:, :, None] - b_i[None, None, :])
        ll = (Y[:, None, :] * log_expit(z) + (1.0 - Y[:, None, :]) * log_expit(-z)).sum(2)
        ll += -0.5 * (grid_th / sd_theta) ** 2
        P = np.exp(ll - ll.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        k = _sample_rows(P, rng)
        th = grid_th[np.arange(N), k]
        sh = np.arange(-SHIFT_STEPS, SHIFT_STEPS + 1)
        lw = -0.5 * (((th[None, :] + sh[:, None] * B_STEP) / sd_theta) ** 2).sum(1)
        gb = g % NB
        valid = np.array([((gb + k_) >= 0).all() and ((gb + k_) < NB).all() for k_ in sh])
        lw = np.where(valid, lw, -np.inf)
        pw = np.exp(lw - lw.max()); pw /= pw.sum()
        k_sh = int(sh[rng.choice(len(sh), p=pw)])
        if k_sh:
            th = th + k_sh * B_STEP
            g = g + k_sh
            a_i, b_i = a_grid[g], bb[g]
            pr = replace(pr, mu_b=pr.mu_b + k_sh * B_STEP)
        upd = {}
        for key_mu, key_sd, x in (("mu_b", "sd_b", b_i), ("mu_log_a", "sd_log_a", la[g])):
            xb, ss = x.mean(), ((x - x.mean()) ** 2).sum()
            var = ss / max(rng.chisquare(I - 1), 1e-12)
            upd[key_sd] = float(np.sqrt(var))
            upd[key_mu] = float(rng.normal(xb, np.sqrt(var / I)))
        pr = _clip_priors(replace(pr, **upd))
        if trace is not None:
            trace.append([th.mean(), th.std(), pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a, th[0]])
    return W_bar / max(len(states), 1), states, (n_acc / n_prop if n_prop else float("nan"))


# ----------------------------------------------------------------------------- diagnostics
def diagnostics(T):
    out = {}
    for k, col in PARAMS.items():
        x = T[:, :, col]
        e_mean = ess(x, False)
        out[f"rhat_{k}"] = rhat_rank(x)
        out[f"ess_{k}"] = ess(x, True)
        out[f"mcse_{k}"] = float(x.std(ddof=1) / np.sqrt(max(e_mean, 1.0)))
        out[f"sd_{k}"] = float(x.std(ddof=1))
        out[f"mean_{k}"] = float(x.mean())
    return out


# ----------------------------------------------------------------------------- one panel
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
    Ws, states, traces, accs = [], [], [], []
    for c in range(CHAINS):
        rng_c = np.random.default_rng(sd("rf2b_chain", c))
        th0, pr0 = _start(cal, rng_c, c)
        tr = []
        Wc, st, acc = gibbs_scale(Yp, th0, pr0, grid, rng_c, BURN, DRAWS, THIN, trace=tr)
        Ws.append(Wc); states += st; traces.append(np.array(tr)[BURN:]); accs.append(acc)
    T = np.stack(traces)
    diag = diagnostics(T)
    diag["scale_acceptance"] = float(np.mean(accs))
    W_pool = np.mean(Ws, 0)
    base = {"dataset": B.name, "target": int(t), "rep": r, "N": N0}
    rows = []
    ll_chain = []
    for c in range(CHAINS):
        out = replay(y, _pre_bmi(Ws[c], grid, theta), pool, holdout, "pwfi", cfg.budget,
                     np.random.default_rng(sd("policy", "rf2b:pdm:bmi")))
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
        out = replay(y, pre, pool, holdout, crit, cfg.budget, np.random.default_rng(sd("policy", "rf2b:" + pol)))
        rows.append(base | {"policy": pol} | summarize2(out, None) | diag)
    return int(t), rows, time.time() - t0, T.astype(np.float32)


def run(name, workers):
    from multiprocessing import get_context
    od = f"{OUT}/{name}"
    if os.path.exists(f"{od}/done.json"):
        print(f"[rf2b/{name}] complete", flush=True)
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
                print(f"[rf2b/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)} ({secs:.0f}s)", flush=True)
    rows, secs = [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))
        rows += p["rows"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/rows.csv.gz", index=False)
    json.dump({"dataset": name, "targets": targets, "chains": CHAINS, "burn": BURN, "draws": DRAWS, "thin": THIN,
               "scale_step": SCALE_STEP, "n_rows": len(rows), "cpu_seconds": round(secs, 1)},
              open(f"{od}/done.json", "w"), indent=1)
    print(f"RF2B_GATE=PASS {name} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CONF)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for n in a.datasets:
        run(n, a.workers)
    print("RF2B_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
