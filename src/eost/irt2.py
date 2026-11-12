"""Calibration variants for study 2 (the frozen ``irt.py`` of study 1 is not modified).

* 3PL with a fixed lower asymptote c (chance rate of the item format), item parameters
  by grid MAP given the calibrators' abilities, used by the ATLAS-style CAT;
* empirical-Bayes hyperprior with Student-t (nu = 4) instead of Gaussian families;
* cross-fitted empirical Bayes: the hyperprior applied to an item is estimated on the
  other half of the items only (leave-item-block-out).
All share the calibrators' joint-MAP abilities and the item grid of ``irt.py``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_expit, logsumexp

from .irt import (B_RANGE, LOG_A_RANGE, ItemGrid, MeasurementModel, Priors, _softmax_rows,
                  empirical_bayes, grid_logprior, grid_loglik, make_item_grid)

NU_T = 4.0


# ------------------------------------------------------------------ 3PL (fixed c)
def grid_loglik_3pl(Y, theta_cal, grid: ItemGrid, c: float) -> np.ndarray:
    """(I, G) log-likelihood under P = c + (1 - c) * sigma(a (theta - b)), computed stably."""
    Y = np.asarray(Y, dtype=float)
    z = np.exp(grid.log_a)[None, :] * (theta_cal[:, None] - grid.b[None, :])      # (N, G)
    if c <= 0:
        logP, log1mP = log_expit(z), log_expit(-z)
    else:
        logP = np.logaddexp(np.log(c), np.log1p(-c) + log_expit(z))
        log1mP = np.log1p(-c) + log_expit(-z)
    return Y.T @ logP + (1.0 - Y).T @ log1mP


def fit_3pl_fixed_c(Y, theta_cal, pr: Priors, c: float, grid: ItemGrid | None = None):
    """Grid MAP of (a_i, b_i) given calibrator abilities, lower asymptote fixed at c."""
    grid = make_item_grid("2pl") if grid is None else grid
    L = grid_loglik_3pl(Y, theta_cal, grid, c) + grid_logprior(grid, pr, "2pl")[None, :]
    g = L.argmax(1)
    return np.exp(grid.log_a[g]), grid.b[g]


def plugin_model_3pl(a, b, c, theta) -> MeasurementModel:
    S = expit(a[:, None] * (theta[None, :] - b[:, None]))
    P = c + (1.0 - c) * S
    dP = (1.0 - c) * a[:, None] * S * (1 - S)
    return MeasurementModel("3pl", theta, P, dP, item_a_eff=a * (1.0 - c))


# ------------------------------------------------------------ Student-t hyperprior
def _logt(x, mu, s, nu=NU_T):
    u = (x - mu) / s
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s) \
        - (nu + 1) / 2 * np.log1p(u ** 2 / nu)


def grid_logprior_t(grid: ItemGrid, pr: Priors, nu=NU_T) -> np.ndarray:
    lp = _logt(grid.b, pr.mu_b, pr.sd_b, nu) + _logt(grid.log_a, pr.mu_log_a, pr.sd_log_a, nu)
    return lp - logsumexp(lp)


def empirical_bayes_t(LL: np.ndarray, grid: ItemGrid, init: Priors, nu=NU_T):
    """Type-II ML of a Student-t hyperprior (location/scale for b and log a) on the grid."""
    def nll(x):
        pr = replace(init, mu_b=x[0], sd_b=np.exp(x[1]), mu_log_a=x[2], sd_log_a=np.exp(x[3]))
        return -float(logsumexp(LL + grid_logprior_t(grid, pr, nu)[None, :], axis=1).sum())
    x0 = np.array([init.mu_b, np.log(init.sd_b), init.mu_log_a, np.log(init.sd_log_a)])
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
    x = res.x
    pr = replace(init, mu_b=float(np.clip(x[0], -6, 6)), sd_b=float(np.clip(np.exp(x[1]), 0.3, 4.0)),
                 mu_log_a=float(np.clip(x[2], -1.5, 1.5)), sd_log_a=float(np.clip(np.exp(x[3]), 0.1, 1.5)))
    return pr, {"marginal_loglik": -float(res.fun), "converged": bool(res.success)}


def t_prior_posterior(Y, theta_cal, init: Priors, grid: ItemGrid | None = None):
    """Item grid posteriors under an empirical-Bayes Student-t hyperprior."""
    grid = make_item_grid("2pl") if grid is None else grid
    coarse = make_item_grid("2pl", b_range=(B_RANGE[0], B_RANGE[1], 65),
                            log_a_range=(LOG_A_RANGE[0], LOG_A_RANGE[1], 14))
    pr, info = empirical_bayes_t(grid_loglik(Y, theta_cal, coarse), coarse, init)
    W = _softmax_rows(grid_loglik(Y, theta_cal, grid) + grid_logprior_t(grid, pr)[None, :])
    return W, grid, pr, info


# ------------------------------------------------------------ cross-fitted EB
def crossfit_posterior(Y, theta_cal, init: Priors, rng: np.random.Generator, grid: ItemGrid | None = None):
    """Item grid posteriors with a hyperprior fitted on the other half of the items."""
    grid = make_item_grid("2pl") if grid is None else grid
    I = Y.shape[1]
    fold = rng.permutation(I) % 2
    coarse = make_item_grid("2pl", b_range=(B_RANGE[0], B_RANGE[1], 65),
                            log_a_range=(LOG_A_RANGE[0], LOG_A_RANGE[1], 14))
    LLc = grid_loglik(Y, theta_cal, coarse)
    LL = grid_loglik(Y, theta_cal, grid)
    W = np.empty((I, len(grid.b)))
    prs = []
    for k in (0, 1):
        pr, _ = empirical_bayes(LLc[fold != k], coarse, init, "2pl", iters=60, tol=1e-7)
        W[fold == k] = _softmax_rows(LL[fold == k] + grid_logprior(grid, pr, "2pl")[None, :])
        prs.append(pr)
    return W, grid, prs
