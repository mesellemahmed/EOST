"""Benchmark-reduction baselines for study 2: tinyBenchmarks and MINCE.

tinyBenchmarks (Polo et al., 2024; reference code github.com/felipemaiapolo/efficbench):
IRT anchor points -- K-Means (K = budget, best of 5 restarts) on the item embeddings
(a_i, a_i b_i) of the calibration panel's 2PL fit (their (alpha_i, beta_i) with
P = sigma(alpha theta - beta)); the anchor is the item closest to each centroid and its
weight the fraction of items in its cluster. Estimators: weighted anchor mean ("irt"),
p-IRT (observed anchors + IRT predictions for every other item at the MLE ability) and
gp-IRT = lambda * anchor + (1 - lambda) * p-IRT with lambda = b^2 / (v / (4 n) + b^2),
b the validation error of the IRT prediction and v the mean Bernoulli variance of the
training systems (their get_lambda). With a calibration panel instead of hundreds of
training models, validation uses leave-one-calibrator-out refits of the item parameters.

MINCE (Das et al., 2026): for candidate sizes n (step 25), R random subsets of the pool;
worst-case drift over the calibration systems max_c |mean_subset(y_c) - A_c|; P95 over
subsets; n* = first size whose marginal reduction of P95 is below tau = 1 point. The
score of a new system is the plain mean on one frozen random subset; P95(n) is reported
as its drift bound.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit

from .irt import Priors, fit_items_given_theta


def kmeans(X: np.ndarray, K: int, rng: np.random.Generator, restarts: int = 5, iters: int = 100):
    """Lloyd's algorithm with k-means++ seeding; best of `restarts` by inertia."""
    n = X.shape[0]
    K = min(K, n)
    best = None
    for _ in range(restarts):
        C = np.empty((K, X.shape[1]))
        C[0] = X[rng.integers(n)]
        d2 = ((X - C[0]) ** 2).sum(1)
        for k in range(1, K):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            C[k] = X[rng.choice(n, p=p)]
            d2 = np.minimum(d2, ((X - C[k]) ** 2).sum(1))
        for _it in range(iters):
            D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
            lab = D.argmin(1)
            newC = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C.copy(), lab.copy())
    return best[1], best[2]


def anchor_points(a, b, pool, K, rng):
    """IRT anchor points among `pool` (indices) and their weights."""
    X = np.column_stack([a[pool], a[pool] * b[pool]])
    C, lab = kmeans(X, K, rng)
    D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    anchors, weights = [], []
    for k in range(C.shape[0]):
        members = np.flatnonzero(lab == k)
        if len(members) == 0:
            continue
        j = members[np.argmin(D[members, k])]
        anchors.append(pool[j]); weights.append(len(members) / len(pool))
    return np.array(anchors), np.array(weights)


def mle_theta(y_obs, a, b, theta_grid):
    """Maximum-likelihood ability on a bounded grid (no prior), as the tinyBenchmarks fit."""
    z = a[:, None] * (theta_grid[None, :] - b[:, None])
    ll = (y_obs[:, None] * log_expit(z) + (1 - y_obs[:, None]) * log_expit(-z)).sum(0)
    return float(theta_grid[np.argmax(ll)])


def pirt_estimate(y_anchor, anchors, a, b, I, theta_grid):
    th = mle_theta(y_anchor, a[anchors], b[anchors], theta_grid)
    p = expit(a * (th - b))
    seen = np.zeros(I, dtype=bool); seen[anchors] = True
    return (y_anchor.sum() + p[~seen].sum()) / I


def tiny_estimates(y, Yp, theta_cal, pr: Priors, a, b, pool, budgets, rng, theta_grid):
    """tinyBenchmarks estimates of the full-bank score of target responses y at each budget."""
    I = y.shape[0]
    N = Yp.shape[0]
    # leave-one-calibrator-out item refits for the validation error b of the IRT prediction
    loo = []
    for c in range(N):
        keep = np.arange(N) != c
        ac, bc = fit_items_given_theta(Yp[keep], theta_cal[keep], pr, "2pl", u0=np.log(a), b0=b)
        loo.append((ac, bc))
    v = float(np.mean(Yp.mean(1) * (1 - Yp.mean(1))))
    out = {}
    for T in budgets:
        anchors, w = anchor_points(a, b, pool, T, rng)
        ya = y[anchors].astype(float)
        z_anchor = float((w * ya).sum())
        z_pirt = pirt_estimate(ya, anchors, a, b, I, theta_grid)
        seen = np.zeros(I, dtype=bool); seen[anchors] = True
        errs = []
        for c, (ac, bc) in enumerate(loo):
            yc = Yp[c]
            th = mle_theta(yc[anchors], ac[anchors], bc[anchors], theta_grid)
            pred = expit(ac * (th - bc))[~seen].mean()
            errs.append(abs(pred - yc[~seen].mean()))
        bb = float(np.mean(errs))
        lam = bb ** 2 / (v / (4 * len(anchors)) + bb ** 2) if (v > 0 or bb > 0) else 0.5
        out[T] = {"anchor": z_anchor, "pirt": z_pirt, "gpirt": lam * z_anchor + (1 - lam) * z_pirt,
                  "lambda": lam, "n_items": int(len(anchors))}
    return out


def mince(y, Yp, pool, budgets, rng, R=2000, step=25, tau=0.01, chunk=200):
    """MINCE sizing on the calibration panel + subset-mean estimates for target y."""
    P = len(pool)
    sizes = np.arange(step, P + 1, step)
    A_c = Yp.mean(1)                                   # calibrators' full-bank accuracies
    Yc = Yp[:, pool].astype(np.float32)
    drifts = np.empty((R, len(sizes)), dtype=np.float32)
    done = 0
    while done < R:
        m = min(chunk, R - done)
        perms = np.argsort(rng.random((m, P)), axis=1)
        cs = np.cumsum(Yc[:, perms], axis=2)            # (N, m, P)
        means = cs[:, :, sizes - 1] / sizes[None, None, :]
        drifts[done:done + m] = np.abs(means - A_c[:, None, None]).max(0)
        done += m
    p95 = np.quantile(drifts, 0.95, axis=0)
    gain = np.r_[np.inf, p95[:-1] - p95[1:]]
    ok = np.flatnonzero(gain < tau)
    n_star = int(sizes[ok[0]]) if len(ok) else int(sizes[-1])
    order = pool[rng.permutation(P)]                    # the frozen subset (nested prefixes)
    A = float(y.mean())
    res = {"n_star": n_star, "p95_nstar": float(p95[sizes == n_star][0]) if n_star in sizes else np.nan}
    est_star = float(y[order[:n_star]].mean())
    res["err_nstar"] = abs(est_star - A)
    for T in budgets:
        est = float(y[order[:T]].mean())
        bound = float(np.interp(T, sizes, p95)) if T >= sizes[0] else float(p95[0])
        res[T] = {"est": est, "bound": bound, "covered": float(abs(est - A) <= bound)}
    return res
