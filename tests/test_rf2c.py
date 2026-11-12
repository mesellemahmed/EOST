"""Checks of the exact sampler of follow-up RF2c (scripts/fb_exact.py): the collapsed likelihood
equals the direct computation with the normalized grid prior, and every Metropolis move leaves
the collapsed posterior invariant."""
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eost.irt import Priors, grid_logprior, grid_loglik, make_item_grid  # noqa: E402
from eost.fullbayes import B_STEP, NB  # noqa: E402
import fb_exact as X  # noqa: E402


def _problem(N=4, I=15, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 1, N)
    a = np.exp(rng.normal(0.2, 0.3, I))
    b = rng.normal(0.0, 1.0, I)
    P = 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    return (rng.random((N, I)) < P).astype(float), th


PR = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=0.2, sd_b=1.0)


def test_collapsed_loglik_equals_normalized_grid_prior():
    Y, th = _problem()
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    A = grid_loglik(Y, th, grid)
    m, E = X.item_block(A, ax)
    for pr in (PR, Priors(mu_log_a=-0.4, sd_log_a=0.12, mu_b=1.3, sd_b=0.5)):
        direct = float(logsumexp(A + grid_logprior(grid, pr, "2pl")[None, :], axis=1).sum())
        assert abs(X.item_loglik(m, E, ax.u(pr), ax.v(pr)) - direct) < 1e-8


def _ks(sample, grid_x, dens):
    cdf = np.cumsum(dens) / dens.sum()
    emp = np.searchsorted(np.sort(sample), grid_x, side="right") / len(sample)
    return float(np.max(np.abs(emp - cdf)))


def test_discrimination_moves_stationary():
    Y, th = _problem(seed=3)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    m, E = X.item_block(grid_loglik(Y, th, grid), ax)
    F = np.einsum("ilk,k->il", E, ax.v(PR))
    mu = np.linspace(-1.5, 1.5, 241)
    ls = np.linspace(np.log(0.1), np.log(1.5), 241)
    L = np.empty((len(mu), len(ls)))
    for i, a in enumerate(mu):
        for j, s in enumerate(ls):
            u = ax.u(replace(PR, mu_log_a=a, sd_log_a=np.exp(s)))
            L[i, j] = np.sum(np.log(F @ u)) - len(m) * np.log(u.sum())
    P = np.exp(L - L.max())                     # density in (mu_a, log sigma_a) coordinates
    rng = np.random.default_rng(7)
    pr, mus, lss = PR, [], []
    steps = {"mu_a": 0.4, "sd_a": 0.4, "mu_b": 0.0, "sd_b": 0.0}
    for t in range(15000):
        pr, _ = X.hyper_moves(th, pr, m, E, ax, rng, steps, False, t)
        mus.append(pr.mu_log_a); lss.append(np.log(pr.sd_log_a))
    assert pr.mu_b == PR.mu_b and pr.sd_b == PR.sd_b
    assert _ks(np.array(mus), mu, P.sum(1)) < 0.06
    assert _ks(np.array(lss), ls, P.sum(0)) < 0.06


def test_difficulty_moves_stationary():
    Y, th = _problem(seed=4)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    m, E = X.item_block(grid_loglik(Y, th, grid), ax)
    G = np.einsum("ilk,l->ik", E, ax.u(PR))
    mu = np.linspace(-6, 6, 241)
    ls = np.linspace(np.log(0.3), np.log(4.0), 241)
    L = np.empty((len(mu), len(ls)))
    for i, a in enumerate(mu):
        for j, s in enumerate(ls):
            v = ax.v(replace(PR, mu_b=a, sd_b=np.exp(s)))
            L[i, j] = np.sum(np.log(G @ v)) - len(m) * np.log(v.sum())
    P = np.exp(L - L.max())
    rng = np.random.default_rng(8)
    pr, mus, lss = PR, [], []
    steps = {"mu_a": 0.0, "sd_a": 0.0, "mu_b": 0.6, "sd_b": 0.4}
    for t in range(15000):
        pr, _ = X.hyper_moves(th, pr, m, E, ax, rng, steps, False, t)
        mus.append(pr.mu_b); lss.append(np.log(pr.sd_b))
    assert _ks(np.array(mus), mu, P.sum(1)) < 0.06
    assert _ks(np.array(lss), ls, P.sum(0)) < 0.06


def test_scale_move_stationary_along_orbit():
    Y, th0 = _problem(N=4, I=15, seed=1)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    N = len(th0)
    ks = [k for k in range(-12, 13) if X.in_box(X.scale_proposal(th0, PR, k)[1])]
    logp = []
    for k in ks:
        t, p, ls = X.scale_proposal(th0, PR, k)
        m, E = X.item_block(grid_loglik(Y, t, grid), ax)
        logp.append(X.log_target(t, p, m, E, ax) + (N + 2) * ls)
    logp = np.array(logp)
    p_exact = np.exp(logp - logp.max()); p_exact /= p_exact.sum()
    rng = np.random.default_rng(11)
    th, pr = th0.copy(), PR
    m, E = X.item_block(grid_loglik(Y, th, grid), ax)
    counts = dict.fromkeys(ks, 0)
    for _ in range(12000):
        th, pr, m, E, _a = X.scale_move(Y, th, pr, grid, ax, m, E, rng)
        counts[int(round((PR.mu_log_a - pr.mu_log_a) / X.SCALE_STEP))] += 1
    p_emp = np.array([counts[k] for k in ks], float); p_emp /= p_emp.sum()
    assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.05, (p_emp.round(3), p_exact.round(3))


def test_translation_weights_match_joint_density():
    """The translation weights equal the joint density of (abilities, grid points, eta) along
    the shift orbit, with the normalized grid prior."""
    Y, th = _problem(seed=5)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    I = Y.shape[1]
    rng = np.random.default_rng(2)
    pr = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=4.9, sd_b=2.5)      # near the box and grid edge
    W = np.exp(grid_logprior(grid, pr, "2pl"))
    g = rng.choice(len(W), size=I, p=W / W.sum())
    sh, lw = X.translation_weights(th, pr, g, ax, I)

    def joint(k):
        p2 = replace(pr, mu_b=pr.mu_b + k * B_STEP)
        lp = grid_logprior(grid, p2, "2pl")
        ll = grid_loglik(Y, th + k * B_STEP, grid)
        gg = g + k
        return (-0.5 * np.sum((th + k * B_STEP) ** 2) + lp[gg].sum() + ll[np.arange(I), gg].sum())
    ok = np.isfinite(lw)
    assert ok.sum() >= 3
    ref = np.array([joint(k) if f else -np.inf for k, f in zip(sh, ok)])
    d = lw[ok] - ref[ok]
    assert np.allclose(d, d[0], atol=1e-6)
    gb = g % NB
    assert all(((gb + k) >= 0).all() and ((gb + k) < NB).all() and abs(pr.mu_b + k * B_STEP) <= 6
               for k in sh[ok])


def test_collapsed_translation_stationary_along_orbit():
    """RF2d's collapsed translation: along the orbit x_k = (theta + k d, mu_b + k d) the stationary
    law is p_k ~ pi(x_k) (Jacobian 1), including the edge of the grid (truncated prior)."""
    Y, th0 = _problem(N=4, I=15, seed=6)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    pr0 = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=-4.0, sd_b=3.0)    # prior mass beyond the grid edge
    ks = [k for k in range(-20, 21) if X.in_box(replace(pr0, mu_b=pr0.mu_b + k * B_STEP))]
    logp = []
    for k in ks:
        t, p = th0 + k * B_STEP, replace(pr0, mu_b=pr0.mu_b + k * B_STEP)
        m, E = X.item_block(grid_loglik(Y, t, grid), ax)
        logp.append(X.log_target(t, p, m, E, ax))
    logp = np.array(logp)
    p_exact = np.exp(logp - logp.max()); p_exact /= p_exact.sum()
    rng = np.random.default_rng(13)
    th, pr = th0.copy(), pr0
    m, E = X.item_block(grid_loglik(Y, th, grid), ax)
    counts = dict.fromkeys(ks, 0)
    for _ in range(12000):
        th, pr, m, E, _a = X.translation_move(Y, th, pr, grid, ax, m, E, rng)
        counts[int(round((pr.mu_b - pr0.mu_b) / B_STEP))] += 1
    p_emp = np.array([counts[k] for k in ks], float); p_emp /= p_emp.sum()
    assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.05, (p_emp.round(3), p_exact.round(3))


def test_rf2c_defaults_unchanged_and_rf2d_overrides():
    import importlib
    import fb_exact2  # noqa: F401
    importlib.reload(X)
    assert X.CTRANS is False and X.CHAIN_TAG == "rf2c_chain" and X.BURN == 200


def test_rf2d_fine_scale_steps_stationary():
    """RF2d's scale move with log s in +-0.05, 0.1, 0.2, 0.4: stationary law on the 0.05 lattice."""
    Y, th0 = _problem(N=4, I=15, seed=7)
    grid = make_item_grid("2pl")
    ax = X.Axes(grid)
    old = X.SCALE_LS
    X.SCALE_LS = np.array([-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4])
    try:
        N = len(th0)
        js = [j for j in range(-60, 61) if X.in_box(X.scale_proposal(th0, PR, j * 0.05 / X.SCALE_STEP)[1])]
        logp = []
        for j in js:
            t, p, ls = X.scale_proposal(th0, PR, j * 0.05 / X.SCALE_STEP)
            m, E = X.item_block(grid_loglik(Y, t, grid), ax)
            logp.append(X.log_target(t, p, m, E, ax) + (N + 2) * ls)
        logp = np.array(logp)
        p_exact = np.exp(logp - logp.max()); p_exact /= p_exact.sum()
        rng = np.random.default_rng(17)
        th, pr = th0.copy(), PR
        m, E = X.item_block(grid_loglik(Y, th, grid), ax)
        counts = dict.fromkeys(js, 0)
        for _ in range(15000):
            th, pr, m, E, _a = X.scale_move(Y, th, pr, grid, ax, m, E, rng)
            counts[int(round((PR.mu_log_a - pr.mu_log_a) / 0.05))] += 1
        p_emp = np.array([counts[j] for j in js], float); p_emp /= p_emp.sum()
        assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.06, (p_emp.round(3), p_exact.round(3))
    finally:
        X.SCALE_LS = old
