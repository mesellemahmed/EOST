"""Checks of the scale move of follow-up RF2b (scripts/fb_scale.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eost.irt import Priors, make_item_grid, grid_loglik  # noqa: E402
from fb_scale import (SCALE_STEP, collapsed_terms, gibbs_scale, in_box, log_target, scale_move,  # noqa: E402
                      scale_proposal)
from posthoc_s2_corrections import gibbs_with_states  # noqa: E402


def _problem(N=4, I=15, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 1, N)
    a = np.exp(rng.normal(0.2, 0.3, I))
    b = rng.normal(0.0, 1.0, I)
    P = 1 / (1 + np.exp(-a[None, :] * (th[:, None] - b[None, :])))
    return (rng.random((N, I)) < P).astype(float), th


def test_scale_off_reproduces_rf2_sampler():
    Y, th = _problem()
    grid = make_item_grid("2pl")
    pr = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=0.2, sd_b=1.0)
    W1, st1 = gibbs_with_states(Y, th, pr, grid, np.random.default_rng(5), 5, 4, 2)
    W2, st2, _ = gibbs_scale(Y, th, pr, grid, np.random.default_rng(5), 5, 4, 2, scale=False)
    assert np.array_equal(W1, W2)
    assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(st1, st2))


def test_collapsed_conditionals_match_frozen_item_conditionals():
    Y, th = _problem()
    grid = make_item_grid("2pl")
    pr = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=0.2, sd_b=1.0)
    from eost.irt import _softmax_rows, grid_logprior
    A = grid_loglik(Y, th, grid)
    ref = _softmax_rows(A + grid_logprior(grid, pr, "2pl")[None, :])
    X, S, lse = collapsed_terms(A.copy(), grid, pr)
    assert np.allclose(X / S, ref, atol=1e-12)
    assert np.all(np.isfinite(lse))


def test_scale_move_stationary_along_orbit():
    """Only the scale move: the chain lives on the orbit x_k = T_{exp(0.2 k)} x_0, where its
    stationary law is p_k ~ pi(x_k) exp(0.2 k (N + 2)) (deterministic-transformation MH)."""
    Y, th0 = _problem(N=4, I=15, seed=1)
    grid = make_item_grid("2pl")
    pr0 = Priors(mu_log_a=0.1, sd_log_a=0.4, mu_b=0.2, sd_b=1.0)
    N = len(th0)
    ks = [k for k in range(-12, 13) if in_box(scale_proposal(th0, pr0, k)[1])]
    logp = []
    for k in ks:
        t, p, ls = scale_proposal(th0, pr0, k)
        _, _, lse = collapsed_terms(grid_loglik(Y, t, grid), grid, p)
        logp.append(log_target(t, p, lse) + (N + 2) * ls)
    logp = np.array(logp)
    p_exact = np.exp(logp - logp.max()); p_exact /= p_exact.sum()
    rng = np.random.default_rng(11)
    th, pr = th0.copy(), pr0
    X, S, lse = collapsed_terms(grid_loglik(Y, th, grid), grid, pr)
    counts = dict.fromkeys(ks, 0)
    for _ in range(12000):
        th, pr, X, S, lse, _acc = scale_move(Y, th, pr, grid, X, S, lse, rng)
        counts[int(round((pr0.mu_log_a - pr.mu_log_a) / SCALE_STEP))] += 1
    p_emp = np.array([counts[k] for k in ks], float); p_emp /= p_emp.sum()
    assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.05, (p_emp.round(3), p_exact.round(3))


def test_scale_move_leaves_posterior_unchanged_small_problem():
    """With and without the scale move the sampler targets the same posterior: on a small
    problem, where the frozen sampler mixes, long runs agree within Monte Carlo error."""
    Y, th0 = _problem(N=5, I=12, seed=2)
    grid = make_item_grid("2pl")
    pr0 = Priors(mu_log_a=0.0, sd_log_a=0.4, mu_b=0.0, sd_b=1.0)

    def run(scale, seed):
        tr = []
        gibbs_scale(Y, th0, pr0, grid, np.random.default_rng(seed), 300, 3000, 1, trace=tr, scale=scale)
        return np.array(tr)[300:]

    def mean_mcse(x, nb=30):
        b = x[: len(x) // nb * nb].reshape(nb, -1).mean(1)
        return x.mean(), b.std(ddof=1) / np.sqrt(nb)

    A, B = run(False, 3), run(True, 4)
    for col in (1, 3, 4):           # ability scale, sigma_b, mu_a
        (ma, ea), (mb, eb) = mean_mcse(A[:, col]), mean_mcse(B[:, col])
        assert abs(ma - mb) < 4 * np.hypot(ea, eb) + 0.02, (col, ma, mb, ea, eb)
