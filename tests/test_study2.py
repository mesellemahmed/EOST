"""Unit tests for the study-2 modules (new code only; study-1 modules are frozen)."""
import numpy as np
from scipy.special import expit
from scipy.stats import norm

from eost.fullbayes import expected_fisher, gibbs_item_marginals
from eost.irt import Priors, calibrate, default_theta_grid, make_item_grid, marginal_model
from eost.irt2 import fit_3pl_fixed_c, grid_loglik_3pl
from eost.reduction import kmeans, mince, tiny_estimates
from eost.replay import Precomputed, replay
from eost.replay2 import (chooser_eig, interval_score, recalibration_factors, replay_chooser,
                          summarize2)


def _bank(J=40, I=120, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=J)
    a = np.exp(rng.normal(0, 0.4, I))
    b = rng.normal(0, 1, I)
    Y = (rng.random((J, I)) < expit(a * (th[:, None] - b))).astype(float)
    return Y, th, a, b


def test_replay_chooser_reproduces_frozen_replay():
    Y, *_ = _bank()
    cal, W, grid = calibrate(Y[:8], "2pl", Priors(), True)
    theta = default_theta_grid()
    pre = Precomputed.build(marginal_model(W, grid, theta), 1.0)
    y = Y[20].astype(np.int64)
    pool, hold = np.arange(20, 120), np.arange(20)
    o1 = replay(y, pre, pool, hold, "eig", 30, np.random.default_rng(1))
    o2 = replay_chooser(y, pre, pool, hold, chooser_eig, 30, np.random.default_rng(1))
    for k in ["est", "lo", "hi", "logloss", "theta_mean", "items"]:
        assert np.array_equal(o1[k], o2[k]), k


def test_information_cap_holds_exactly_for_the_logistic_link():
    """Proposition 4 (exact form): with b ~ N(mu, s^2), the Fisher information of the expected
    logistic response function never exceeds 2 / (pi s^2), for any discrimination.

    The expected response is computed through the latent representation used in the proof,
    E_b[sigma(a(theta - b))] = E_L[Phi((theta - L/a)/s)] with L standard logistic, which is a
    smooth integrand and can be integrated accurately on a fine grid.
    """
    L = np.linspace(-45, 45, 90001)
    wL = expit(L) * (1 - expit(L))
    wL = wL / wL.sum()
    th = np.linspace(-3, 3, 241)
    for s in (0.3, 0.7, 1.5):
        cap = 2 / (np.pi * s ** 2)
        for a in (0.5, 2.0, 8.0, 40.0, 400.0):
            u = (th[:, None] - L[None, :] / a) / s
            P, Q = norm.cdf(u) @ wL, norm.sf(u) @ wL      # Q = 1 - P computed without cancellation
            dP = (norm.pdf(u) / s) @ wL
            ok = (P > 1e-280) & (Q > 1e-280)
            info = dP[ok] ** 2 / (P[ok] * Q[ok])
            assert info.max() <= cap * (1 + 1e-9), (s, a, info.max(), cap)
            if a == 400.0:                          # probit limit: the cap is approached
                assert info.max() > 0.99 * cap


def test_recalibration_factor_is_one_for_standard_normal_errors():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 100))
    kq = recalibration_factors(Z, "quantile")
    kr = recalibration_factors(Z, "rms")
    assert abs(np.median(kq) - 1) < 0.05 and abs(np.median(kr) - 1) < 0.05
    assert abs(np.median(recalibration_factors(2 * Z, "quantile")) - 2) < 0.1


def test_interval_score_known_values():
    assert np.isclose(interval_score(0.4, 0.6, 0.5), 0.2)
    assert np.isclose(interval_score(0.4, 0.6, 0.7), 0.2 + 20 * 0.1)
    assert np.isclose(interval_score(0.4, 0.6, 0.3), 0.2 + 20 * 0.1)


def test_summarize2_recalibrated_interval_widens():
    Y, *_ = _bank()
    cal, W, grid = calibrate(Y[:8], "2pl", Priors(), True)
    pre = Precomputed.build(marginal_model(W, grid, default_theta_grid()), 1.0)
    out = replay(Y[30].astype(np.int64), pre, np.arange(10, 120), np.arange(10), "eig", 20,
                 np.random.default_rng(0))
    s = summarize2(out, {"R": np.full(20, 2.0)})
    assert np.isclose(s["mean_width_R"], 2 * s["mean_width"])
    assert s["coverage_R"] >= s["coverage"]


def test_3pl_with_zero_guessing_matches_2pl_grid_likelihood():
    Y, th, *_ = _bank()
    grid = make_item_grid("2pl")
    from eost.irt import grid_loglik
    assert np.allclose(grid_loglik_3pl(Y[:6], th[:6], grid, 0.0), grid_loglik(Y[:6], th[:6], grid))
    a, b = fit_3pl_fixed_c(Y[:6], th[:6], Priors(), 0.25, grid)
    assert a.shape == b.shape == (Y.shape[1],)


def test_gibbs_item_marginals_are_distributions_and_close_to_eb_with_large_panel():
    Y, *_ = _bank(J=60, I=80, seed=3)
    cal, W, grid = calibrate(Y, "2pl", Priors(), True)
    Wfb, d = gibbs_item_marginals(Y, cal.theta_cal, cal.priors, grid, np.random.default_rng(0),
                                  burn=4, draws=4, thin=2)
    assert np.allclose(Wfb.sum(1), 1.0)
    th = default_theta_grid()
    P_eb = marginal_model(W, grid, th).P
    P_fb = marginal_model(Wfb, grid, th).P
    assert np.abs(P_eb - P_fb).mean() < 0.05
    assert expected_fisher(Wfb, grid, th).shape == (Y.shape[1], len(th))


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))])
    C, lab = kmeans(X, 2, rng)
    assert len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[-1]


def test_mince_and_tiny_run_and_are_exact_with_the_whole_pool():
    Y, *_ = _bank(J=30, I=100, seed=5)
    cal, W, grid = calibrate(Y[:8], "2pl", Priors(), True)
    pool = np.arange(100)
    res = mince(Y[20], Y[:8], pool, (25, 100), np.random.default_rng(0), R=200)
    assert res[100]["est"] == Y[20].mean()                      # whole pool = exact score
    assert 25 <= res["n_star"] <= 100
    tb = tiny_estimates(Y[20], Y[:8], cal.theta_cal, cal.priors, cal.a, cal.b, pool, (25,),
                        np.random.default_rng(0), default_theta_grid())
    assert 0 <= tb[25]["gpirt"] <= 1 and 0 <= tb[25]["lambda"] <= 1


def test_z90_constant():
    assert np.isclose(norm.ppf(0.95), 1.6448536, atol=1e-6)
