"""Exploratory follow-up RF2c (plan: docs/RESEARCH_LOG.md, iteration 4; written after the
diagnostics of RF2 and of the first benchmark of RF2b had been seen, before this script was run).

RF2b's scale move fixed the ridge along which abilities, sigma_b and mu_a move together, but
sigma_a (the spread of the log discriminations) still mixed slowly, and a few chains stuck at
its lower bound. Two causes: (1) the frozen sampler updates the hyperparameters by conjugate
draws given the sampled grid points, which with eight calibrators carry little information about
each item -- the classic slow mixing of a variance parameter under data augmentation; (2) those
draws treat the grid prior as a continuous density, whereas EOST-M (and Lemma 1) normalize it
over the grid; when sigma_a approaches the grid step the unnormalized version piles up at the
lower bound.

RF2c samples the fully Bayesian model of Lemma 1 exactly -- EOST-M's model, with the calibrators'
abilities N(0, 1), the item prior normalized over the grid, flat priors on the means and Jeffreys
priors on the standard deviations (on the same box as the frozen sampler) -- with the item
parameters summed out wherever the hyperparameters move:

  1. random-walk Metropolis moves on mu_a, log sigma_a, mu_b and log sigma_b (two rounds per
     sweep; step sizes adapted during burn-in only), on the collapsed posterior, which is cheap
     because the grid is a product of a discrimination and a difficulty axis;
  2. the scale move of RF2b (abilities, mu_b, sigma_b times s; mu_a minus log s);
  3. every item's grid point from its exact conditional;
  4. every calibrator's ability by griddy Gibbs given the items (as frozen);
  5. the frozen translation move, with the factor (Z(eta) / Z(eta'))^I that the normalized
     grid prior requires.

Everything else is RF2/RF2b: same targets, panels, holdouts, starting points and replay arms.
Outputs: artifacts/robust/fb_exact/<bank>/{parts,rows.csv.gz,done.json}.
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
from scipy.special import log_expit, logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eost.experiment import _seed  # noqa: E402
from eost.fullbayes import B_STEP, LOCAL, NB, SHIFT_STEPS, _sample_rows  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, grid_loglik, marginal_model  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402
from eost.replay2 import summarize2  # noqa: E402
from eost.study2 import eligible_s2  # noqa: E402
from fb_convergence import CONF, _S, _init, _pre_bmi, _start, pick_targets  # noqa: E402
from fb_scale import SCALE_K, SCALE_STEP, diagnostics, in_box  # noqa: E402
from posthoc_s2_corrections import joint_mixture_model  # noqa: E402

OUT = "artifacts/robust/fb_exact"
CHAINS, BURN, DRAWS, THIN, N0, REP, JOINT_EVERY = 4, 200, 100, 2, 8, 0, 10
ROUNDS = 2
CHAIN_TAG, POLICY_TAG, CTRANS, LOG_TAG = "rf2c_chain", "rf2c:", False, "rf2c"
SCALE_LS = None          # RF2c: log s in +-0.2, +-0.4 (fb_scale); RF2d sets finer steps as well
STEP0 = {"mu_a": 0.05, "sd_a": 0.05, "mu_b": 0.05, "sd_b": 0.05}
TARGET_ACC = 0.4


# ----------------------------------------------------------------------------- the exact model
class Axes:
    """The item grid is the product of NA discrimination levels and NB difficulty levels,
    ordered with the difficulty index running fastest (irt.make_item_grid)."""

    def __init__(self, grid):
        self.na = len(grid.b) // NB
        self.la = grid.log_a.reshape(self.na, NB)[:, 0].copy()
        self.b = grid.b.reshape(self.na, NB)[0, :].copy()
        assert np.allclose(grid.log_a.reshape(self.na, NB), self.la[:, None])
        assert np.allclose(grid.b.reshape(self.na, NB), self.b[None, :])

    def u(self, pr):                       # discrimination factor, max 1
        h = -0.5 * ((self.la - pr.mu_log_a) / pr.sd_log_a) ** 2
        return np.exp(h - h.max())

    def v(self, pr):                       # difficulty factor, max 1
        f = -0.5 * ((self.b - pr.mu_b) / pr.sd_b) ** 2
        return np.exp(f - f.max())


def item_block(A, ax):
    """Row maxima and exp(A - max), shaped (I, NA, NB)."""
    m = A.max(1)
    E = np.exp(A - m[:, None]).reshape(len(m), ax.na, NB)
    return m, E


def item_loglik(m, E, u, v):
    """Sum over items of log sum_g p(g | eta) L_i(g), with p(g | eta) = u_l v_k / (sum u sum v)."""
    s = np.einsum("ilk,l,k->i", E, u, v, optimize=True)
    return float(np.sum(m + np.log(s)) - len(m) * (np.log(u.sum()) + np.log(v.sum())))


def log_prior(th, pr):
    return float(-0.5 * np.sum(th ** 2) - np.log(pr.sd_b) - np.log(pr.sd_log_a))


def log_target(th, pr, m, E, ax):
    return log_prior(th, pr) + item_loglik(m, E, ax.u(pr), ax.v(pr))


# ----------------------------------------------------------------------------- moves
def hyper_moves(th, pr, m, E, ax, rng, steps, adapt, t):
    """Random-walk Metropolis on mu_a, log sigma_a (items summed out given the difficulty
    factor) and on mu_b, log sigma_b (given the discrimination factor)."""
    I = len(m)
    acc = {}
    for _ in range(ROUNDS):
        F = np.einsum("ilk,k->il", E, ax.v(pr), optimize=True)          # (I, NA)
        logv = np.log(ax.v(pr).sum())

        def ll_a(p):
            u = ax.u(p)
            return float(np.sum(np.log(F @ u)) - I * (np.log(u.sum()) + logv))
        cur = ll_a(pr)
        for key, attr in (("mu_a", "mu_log_a"), ("sd_a", "sd_log_a")):
            z = rng.normal() * steps[key]
            if key == "mu_a":
                p2, jac = replace(pr, mu_log_a=pr.mu_log_a + z), 0.0
            else:
                p2, jac = replace(pr, sd_log_a=pr.sd_log_a * np.exp(z)), z
            ok = in_box(p2)
            new = ll_a(p2) if ok else -np.inf
            la = new - cur + (np.log(pr.sd_log_a) - np.log(p2.sd_log_a)) + jac if ok else -np.inf
            a = np.log(rng.random()) < la
            if a:
                pr, cur = p2, new
            acc[key] = acc.get(key, 0) + int(a)
        G = np.einsum("ilk,l->ik", E, ax.u(pr), optimize=True)          # (I, NB)
        logu = np.log(ax.u(pr).sum())

        def ll_b(p):
            v = ax.v(p)
            return float(np.sum(np.log(G @ v)) - I * (np.log(v.sum()) + logu))
        cur = ll_b(pr)
        for key in ("mu_b", "sd_b"):
            z = rng.normal() * steps[key]
            if key == "mu_b":
                p2, jac = replace(pr, mu_b=pr.mu_b + z), 0.0
            else:
                p2, jac = replace(pr, sd_b=pr.sd_b * np.exp(z)), z
            ok = in_box(p2)
            new = ll_b(p2) if ok else -np.inf
            la = new - cur + (np.log(pr.sd_b) - np.log(p2.sd_b)) + jac if ok else -np.inf
            a = np.log(rng.random()) < la
            if a:
                pr, cur = p2, new
            acc[key] = acc.get(key, 0) + int(a)
    if adapt:
        for key in steps:
            steps[key] *= float(np.exp((acc[key] / ROUNDS - TARGET_ACC) / np.sqrt(t + 1)))
    return pr, acc


def scale_proposal(th, pr, k):
    ls = float(k) * SCALE_STEP
    s = np.exp(ls)
    return s * th, replace(pr, mu_b=s * pr.mu_b, sd_b=s * pr.sd_b, mu_log_a=pr.mu_log_a - ls), ls


def scale_move(Y, th, pr, grid, ax, m, E, rng):
    if SCALE_LS is None:
        th2, pr2, ls = scale_proposal(th, pr, int(rng.choice(SCALE_K)))
    else:
        ls = float(rng.choice(SCALE_LS))
        th2, pr2, ls = scale_proposal(th, pr, ls / SCALE_STEP)
    u = np.log(rng.random())
    if not in_box(pr2):
        return th, pr, m, E, False
    A2 = grid_loglik(Y, th2, grid)
    m2, E2 = item_block(A2, ax)
    la = log_target(th2, pr2, m2, E2, ax) - log_target(th, pr, m, E, ax) + (len(th) + 2) * ls
    if u < la:
        return th2, pr2, m2, E2, True
    return th, pr, m, E, False


TRANS_K = np.array([-4, -2, -1, 1, 2, 4])


def translation_move(Y, th, pr, grid, ax, m, E, rng):
    """Collapsed translation (RF2d): abilities and mu_b shifted by k difficulty-grid steps with the
    item parameters summed out, so the move is not blocked by items at the edge of the grid or by
    the normalization of a truncated difficulty prior (Jacobian 1)."""
    k = int(rng.choice(TRANS_K))
    d = k * B_STEP
    th2, pr2 = th + d, replace(pr, mu_b=pr.mu_b + d)
    u = np.log(rng.random())
    if not in_box(pr2):
        return th, pr, m, E, False
    m2, E2 = item_block(grid_loglik(Y, th2, grid), ax)
    la = log_target(th2, pr2, m2, E2, ax) - log_target(th, pr, m, E, ax)
    if u < la:
        return th2, pr2, m2, E2, True
    return th, pr, m, E, False


def translation_weights(th, pr, g, ax, I):
    """Exact conditional over the shift group (abilities, difficulties and mu_b moved by k grid
    steps) under the normalized grid prior."""
    sh = np.arange(-SHIFT_STEPS, SHIFT_STEPS + 1)
    lw = -0.5 * ((th[None, :] + sh[:, None] * B_STEP) ** 2).sum(1)
    gb = g % NB
    lo, hi = gb.min(), gb.max()
    valid = (lo + sh >= 0) & (hi + sh < NB) & (np.abs(pr.mu_b + sh * B_STEP) <= 6)
    logz = np.array([logsumexp(-0.5 * ((ax.b - (pr.mu_b + k * B_STEP)) / pr.sd_b) ** 2) for k in sh])
    lw = lw - I * (logz - logz[SHIFT_STEPS])
    lw = np.where(valid, lw, -np.inf)
    return sh, lw


# ----------------------------------------------------------------------------- sampler
def gibbs_exact(Y, theta0, pr0, grid, rng, burn, draws, thin, trace=None, stats=None, ctrans=False):
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    ax = Axes(grid)
    th = np.array(theta0, dtype=float)
    pr = pr0
    if not in_box(pr):
        raise ValueError("start outside the prior box")
    la, bb = grid.log_a, grid.b
    a_grid = np.exp(la)
    W_bar = np.zeros((I, len(bb)))
    states = []
    steps = dict(STEP0)
    n_acc = {k: 0 for k in list(STEP0) + ["scale"] + (["trans"] if ctrans else [])}
    for s_ in range(burn + draws * thin):
        A = grid_loglik(Y, th, grid)
        m, E = item_block(A, ax)
        pr, acc = hyper_moves(th, pr, m, E, ax, rng, steps, s_ < burn, s_)
        th, pr, m, E, a = scale_move(Y, th, pr, grid, ax, m, E, rng)
        if ctrans:
            th, pr, m, E, a2 = translation_move(Y, th, pr, grid, ax, m, E, rng)
        if s_ >= burn:
            for k_, v_ in acc.items():
                n_acc[k_] += v_
            n_acc["scale"] += int(a)
            if ctrans:
                n_acc["trans"] += int(a2)
        Wu = (E * ax.u(pr)[None, :, None] * ax.v(pr)[None, None, :]).reshape(I, -1)
        if s_ >= burn and (s_ - burn) % thin == thin - 1:
            W_bar += Wu / Wu.sum(1, keepdims=True)
            states.append((th.copy(), pr))
        g = _sample_rows(Wu, rng)
        a_i, b_i = a_grid[g], bb[g]
        grid_th = th[:, None] + LOCAL[None, :]
        z = a_i[None, None, :] * (grid_th[:, :, None] - b_i[None, None, :])
        ll = (Y[:, None, :] * log_expit(z) + (1.0 - Y[:, None, :]) * log_expit(-z)).sum(2)
        ll += -0.5 * grid_th ** 2
        P = np.exp(ll - ll.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        k = _sample_rows(P, rng)
        th = grid_th[np.arange(N), k]
        sh, lw = translation_weights(th, pr, g, ax, I)
        pw = np.exp(lw - lw.max()); pw /= pw.sum()
        k_sh = int(sh[rng.choice(len(sh), p=pw)])
        if k_sh:
            th = th + k_sh * B_STEP
            pr = replace(pr, mu_b=pr.mu_b + k_sh * B_STEP)
        if trace is not None:
            trace.append([th.mean(), th.std(), pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a, th[0]])
    if stats is not None:
        n_kept = draws * thin
        stats.update({f"acc_{k}": v / (n_kept * (ROUNDS if k not in ("scale", "trans") else 1)) for k, v in n_acc.items()})
        stats.update({f"step_{k}": v for k, v in steps.items()})
    return W_bar / max(len(states), 1), states


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
    Ws, states, traces, st_all = [], [], [], []
    for c in range(CHAINS):
        rng_c = np.random.default_rng(sd(CHAIN_TAG, c))
        th0, pr0 = _start(cal, rng_c, c)
        tr, st = [], {}
        Wc, s_ = gibbs_exact(Yp, th0, pr0, grid, rng_c, BURN, DRAWS, THIN, trace=tr, stats=st, ctrans=CTRANS)
        Ws.append(Wc); states += s_; traces.append(np.array(tr)[BURN:]); st_all.append(st)
    T = np.stack(traces)
    diag = diagnostics(T)
    for k in st_all[0]:
        diag[k] = float(np.mean([s[k] for s in st_all]))
    W_pool = np.mean(Ws, 0)
    base = {"dataset": B.name, "target": int(t), "rep": r, "N": N0}
    rows, ll_chain = [], []
    for c in range(CHAINS):
        out = replay(y, _pre_bmi(Ws[c], grid, theta), pool, holdout, "pwfi", cfg.budget,
                     np.random.default_rng(sd("policy", POLICY_TAG + "pdm:bmi")))
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
        out = replay(y, pre, pool, holdout, crit, cfg.budget, np.random.default_rng(sd("policy", POLICY_TAG + pol)))
        rows.append(base | {"policy": pol} | summarize2(out, None) | diag)
    return int(t), rows, time.time() - t0, T.astype(np.float32)


def run(name, workers):
    from multiprocessing import get_context
    od = f"{OUT}/{name}"
    if os.path.exists(f"{od}/done.json"):
        print(f"[{LOG_TAG}/{name}] complete", flush=True)
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
                print(f"[{LOG_TAG}/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)} ({secs:.0f}s)", flush=True)
    rows, secs = [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))
        rows += p["rows"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/rows.csv.gz", index=False)
    json.dump({"dataset": name, "targets": targets, "chains": CHAINS, "burn": BURN, "draws": DRAWS, "thin": THIN,
               "rounds": ROUNDS, "collapsed_translation": CTRANS, "n_rows": len(rows), "cpu_seconds": round(secs, 1)},
              open(f"{od}/done.json", "w"), indent=1)
    print(f"{LOG_TAG.upper()}_GATE=PASS {name} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CONF)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for n in a.datasets:
        run(n, a.workers)
    print("RF2C_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
