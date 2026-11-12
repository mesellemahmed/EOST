"""Fully Bayesian calibration posterior for the fully Bayesian CAT baseline (study 2).

Hierarchical 2PL on the calibration panel: calibrator abilities theta_n ~ N(0, 1);
item parameters (log a_i, b_i) on the item grid of ``irt.py`` with a Gaussian
hyperprior N(mu_a, s_a^2) x N(mu_b, s_b^2); flat priors on the means and Jeffreys
priors on the standard deviations. A Gibbs sampler alternates
  1. every item's grid point given (theta, hyperparameters)   -- exact, independent;
  2. every calibrator's ability given the items                -- griddy Gibbs on a
     local window (65 points, step 0.025) around the current value;
  2b. an exact Gibbs move along the translation group (all abilities and difficulties
     and the difficulty mean shifted by a common multiple of the grid step), which
     removes the slow mixing of the overall location;
  3. the hyperparameters given the items                       -- conjugate draws.
It starts at the joint-MAP abilities and the empirical-Bayes hyperparameters used by
EOST-M, so EOST-M is the same model with abilities and hyperparameters fixed at their
point estimates.

The output is the item *marginal* posterior averaged over the retained draws (each
draw contributes the exact conditional grid posterior, i.e. Rao-Blackwellized). Using
the resulting expected response functions in the target's ability posterior is the
target of the fully Bayesian adaptive-testing algorithm of van der Linden & Ren (2020),
which resamples stored item-parameter draws independently per item at every MCMC
iteration (as used by Fink, Koenig & Frey, 2025); the Bayesian maximum-information
criterion averages Fisher information over the item draws and the ability posterior.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import expit, log_expit

from .irt import ItemGrid, Priors, _softmax_rows, grid_logprior, grid_loglik

LOCAL = np.linspace(-0.8, 0.8, 65)
NB = 129                       # difficulty grid points per discrimination level (irt.B_RANGE)
B_STEP = 16.0 / (NB - 1)       # 0.125
SHIFT_STEPS = 16               # translation moves of up to +/- 2 in steps of 0.125


def _sample_rows(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(W, axis=1)
    u = rng.random(W.shape[0]) * c[:, -1]
    return np.minimum((c < u[:, None]).sum(1), W.shape[1] - 1)


def _clip_priors(pr: Priors) -> Priors:
    return replace(pr, mu_b=float(np.clip(pr.mu_b, -6, 6)), sd_b=float(np.clip(pr.sd_b, 0.3, 4.0)),
                   mu_log_a=float(np.clip(pr.mu_log_a, -1.5, 1.5)),
                   sd_log_a=float(np.clip(pr.sd_log_a, 0.1, 1.5)))


def gibbs_item_marginals(Y: np.ndarray, theta0: np.ndarray, pr0: Priors, grid: ItemGrid,
                         rng: np.random.Generator, burn: int = 10, draws: int = 8, thin: int = 3,
                         sd_theta: float = 1.0, trace: list | None = None):
    """Return (W_bar, diagnostics) for the fully Bayesian item marginal posterior."""
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    th = np.array(theta0, dtype=float)
    pr = pr0
    la, bb = grid.log_a, grid.b
    a_grid = np.exp(la)
    W_bar = np.zeros((I, len(bb)))
    th_draws, eta_draws = [], []
    n_sweeps = burn + draws * thin
    for s in range(n_sweeps):
        # 1. items | abilities, hyperparameters (exact conditional grid posterior)
        W = _softmax_rows(grid_loglik(Y, th, grid) + grid_logprior(grid, pr, "2pl")[None, :])
        keep = s >= burn and (s - burn) % thin == thin - 1
        if keep:
            W_bar += W
            th_draws.append(th.copy())
            eta_draws.append([pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a])
        g = _sample_rows(W, rng)
        a_i, b_i = a_grid[g], bb[g]
        # 2. abilities | items (griddy Gibbs on a local window)
        grid_th = th[:, None] + LOCAL[None, :]                              # (N, K)
        z = a_i[None, None, :] * (grid_th[:, :, None] - b_i[None, None, :])  # (N, K, I)
        ll = (Y[:, None, :] * log_expit(z) + (1.0 - Y[:, None, :]) * log_expit(-z)).sum(2)
        ll += -0.5 * (grid_th / sd_theta) ** 2
        P = np.exp(ll - ll.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        k = _sample_rows(P, rng)
        th = grid_th[np.arange(N), k]
        # 2b. exact Gibbs move along the translation group: (theta, b, mu_b) -> (+d, +d, +d) with
        #     d a multiple of the difficulty grid step leaves the likelihood and the difficulty
        #     hyperprior unchanged, so p(d | rest) is proportional to the ability prior only.
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
        # 3. hyperparameters | items (flat mean, Jeffreys sd; conjugate)
        upd = {}
        for key_mu, key_sd, x in (("mu_b", "sd_b", b_i), ("mu_log_a", "sd_log_a", la[g])):
            xb, ss = x.mean(), ((x - x.mean()) ** 2).sum()
            var = ss / max(rng.chisquare(I - 1), 1e-12)
            upd[key_sd] = float(np.sqrt(var))
            upd[key_mu] = float(rng.normal(xb, np.sqrt(var / I)))
        pr = _clip_priors(replace(pr, **upd))
        if trace is not None:
            trace.append([th.mean(), th.std(), pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a, th[0]])
    W_bar /= max(len(th_draws), 1)
    TH = np.array(th_draws)
    E = np.array(eta_draws)
    diag = {"fb_theta_sd_mean": float(TH.std(0).mean()) if len(TH) > 1 else 0.0,
            "fb_theta_shift_mean": float(np.abs(TH.mean(0) - np.asarray(theta0)).mean()),
            "fb_mu_b_sd": float(E[:, 0].std()), "fb_sd_b_sd": float(E[:, 1].std()),
            "fb_mu_log_a_sd": float(E[:, 2].std()), "fb_sd_log_a_sd": float(E[:, 3].std()),
            "fb_draws": int(len(TH)), "fb_sweeps": int(n_sweeps)}
    return W_bar, diag


def expected_fisher(W: np.ndarray, grid: ItemGrid, theta: np.ndarray) -> np.ndarray:
    """E_xi[I_xi(theta)] under an item grid posterior W (I x G), on the ability grid."""
    a = np.exp(grid.log_a)
    S = expit(a[:, None] * (theta[None, :] - grid.b[:, None]))
    return W @ (a[:, None] ** 2 * S * (1.0 - S))
