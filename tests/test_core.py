import io
import pickle

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import expit

from eost import irt
from eost.data import Bank, effective_panel_size
from eost.replay import Precomputed, h2, replay, summarize
from eost.safeio import RestrictedUnpickler
from eost.stats import holm, paired_contrast, rank_biserial


def simulate(N=40, I=150, seed=0, sd_log_a=0.4):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=N)
    a = np.exp(rng.normal(0, sd_log_a, size=I))
    b = rng.normal(0, 1.2, size=I)
    P = expit(a[None] * (th[:, None] - b[None]))
    return (rng.random((N, I)) < P).astype(float), th, a, b


@pytest.mark.parametrize("model", ["2pl", "1pl"])
def test_gradient_matches_finite_differences(model):
    Y, *_ = simulate(N=6, I=9, seed=1)
    pr = irt.Priors()
    N, I = Y.shape
    M = np.ones_like(Y)
    fun = irt._negpost_2pl if model == "2pl" else irt._negpost_1pl
    nx = N + 2 * I if model == "2pl" else N + I + 1
    x = np.random.default_rng(2).normal(scale=0.3, size=nx)
    pr = irt.Priors(mu_b=0.4, mu_log_a=0.2)
    f, g = fun(x, Y, M, N, I, pr)
    g_num = approx_fprime(x, lambda v: fun(v, Y, M, N, I, pr)[0], 1e-6)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-4)


def test_map_recovers_parameters_on_large_panel():
    Y, th, a, b = simulate(N=400, I=120, seed=3)
    cal = irt.fit_map(Y, "2pl")
    assert cal.converged
    assert np.corrcoef(cal.b, b)[0, 1] > 0.95
    assert np.corrcoef(cal.theta_cal, th)[0, 1] > 0.95
    assert np.corrcoef(np.log(cal.a), np.log(a))[0, 1] > 0.6


def test_grid_posterior_normalised_and_concentrates():
    Y, th, a, b = simulate(N=300, I=150, seed=4)
    cal = irt.fit_map(Y, "2pl")
    grid = irt.make_item_grid()
    W = irt.item_grid_posterior(Y, cal.theta_cal, grid, irt.Priors())
    assert np.allclose(W.sum(1), 1)
    b_mean = W @ grid.b
    assert np.corrcoef(b_mean, b)[0, 1] > 0.95
    # small panel -> much wider posteriors
    W4 = irt.item_grid_posterior(Y[:4], cal.theta_cal[:4], grid, irt.Priors())
    sd = lambda W_: np.sqrt(W_ @ grid.b ** 2 - (W_ @ grid.b) ** 2)
    assert np.median(sd(W4)) > 3 * np.median(sd(W))


def test_erf_shrinks_discrimination_like_probit_approximation():
    # single item, b uncertain with sd s: slope of ERF at b ≈ a / sqrt(1 + pi a^2 s^2 / 8)
    a, s = 3.0, 0.8
    bs = np.linspace(-6, 6, 4001)
    w = np.exp(-0.5 * (bs / s) ** 2); w /= w.sum()
    theta = np.linspace(-0.01, 0.01, 3)
    erf = (w[:, None] * expit(a * (theta[None] - bs[:, None]))).sum(0)
    slope = (erf[2] - erf[0]) / 0.02
    a_eff_numeric = 4 * slope
    a_eff_formula = a / np.sqrt(1 + np.pi * a ** 2 * s ** 2 / 8)
    assert abs(a_eff_numeric - a_eff_formula) / a_eff_formula < 0.03
    assert a_eff_numeric < a


def test_eig_small_variance_limit_is_half_var_times_fisher():
    theta = np.linspace(-1, 1, 2001)
    sig = 0.05
    pi = np.exp(-0.5 * (theta / sig) ** 2); pi /= pi.sum()
    for a, b in [(0.5, 0.3), (2.0, -0.2), (4.0, 0.1)]:
        P = expit(a * (theta - b))
        eig = h2(P @ pi) - h2(P) @ pi
        p0 = expit(a * (0 - b))
        fisher = a ** 2 * p0 * (1 - p0)
        assert abs(eig - 0.5 * sig ** 2 * fisher) / (0.5 * sig ** 2 * fisher) < 0.05


def test_entropy_ignores_discrimination_but_eig_does_not():
    theta = np.linspace(-4, 4, 801)
    pi = np.exp(-0.5 * (theta / 0.5) ** 2); pi /= pi.sum()
    lo = expit(0.3 * (theta - 0.0))   # weak item, predictive p = 0.5
    hi = expit(3.0 * (theta - 0.0))   # sharp item, predictive p = 0.5
    assert abs(h2(lo @ pi) - h2(hi @ pi)) < 1e-6       # identical entropy scores
    eig = lambda P: h2(P @ pi) - h2(P) @ pi
    assert eig(hi) > 5 * eig(lo)                          # very different information


def _toy_pre(I=60, seed=0):
    rng = np.random.default_rng(seed)
    theta = irt.default_theta_grid(K=161)
    a, b = np.exp(rng.normal(0, 0.3, I)), rng.normal(0, 1, I)
    cal = irt.Calibration("2pl", np.zeros(3), a, b, irt.Priors(), True, 0)
    return Precomputed.build(irt.plugin_model(cal, theta)), a, b


@pytest.mark.parametrize("crit", ["random", "entropy", "mfi", "pwfi", "eig", "evr", "naive"])
def test_replay_runs_and_respects_pool(crit):
    pre, a, b = _toy_pre()
    rng = np.random.default_rng(0)
    y = (rng.random(60) < expit(a * (0.7 - b))).astype(int)
    pool, hold = np.arange(48), np.arange(48, 60)
    out = replay(y, pre, pool, hold, crit, 30, np.random.default_rng(1))
    assert len(set(out["items"])) == 30 and set(out["items"]) <= set(pool)
    s = summarize(out)
    assert 0 <= s["coverage"] <= 1 and s["iae"] >= 0


def test_full_pool_estimate_uses_observed_answers():
    pre, a, b = _toy_pre(I=40)
    y = (np.random.default_rng(3).random(40) < 0.5).astype(int)
    out = replay(y, pre, np.arange(40), np.array([], dtype=int), "eig", 40, np.random.default_rng(0))
    assert abs(out["est"][-1] - y.mean()) < 1e-9


def test_sustained_tolerance_is_censored_when_last_step_fails():
    out = {"est": np.array([0.5, 0.31, 0.29, 0.5]), "A_true": 0.3, "lo": np.zeros(4), "hi": np.ones(4),
           "brier": np.zeros(4), "logloss": np.zeros(4), "theta_sd": np.zeros(4), "item_eig": np.zeros(4)}
    s = summarize(out)
    assert s["ttol_02"] == 5  # T + 1: never sustained
    out["est"] = np.array([0.5, 0.31, 0.29, 0.30])
    assert summarize(out)["ttol_02"] == 2


def test_restricted_unpickler_blocks_arbitrary_globals():
    payload = pickle.dumps(print)  # references builtins.print
    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO(payload)).load()
    ok = pickle.dumps(np.arange(3.0))
    assert np.allclose(RestrictedUnpickler(io.BytesIO(ok)).load(), [0, 1, 2])


def test_effective_panel_size_extremes():
    rng = np.random.default_rng(0)
    row = (rng.random(500) < 0.5).astype(float)
    assert effective_panel_size(np.vstack([row] * 8)) < 1.01
    indep = (rng.random((8, 5000)) < 0.5).astype(float)
    assert effective_panel_size(indep) > 7.5


def test_family_exclusion():
    Y = np.zeros((4, 5), dtype=np.uint8)
    bank = Bank("x", Y, np.array(list("abcd")), np.array(["f1", "f1", "f2", "f3"]), np.arange(5), {})
    assert list(bank.eligible_calibrators(0)) == [2, 3]
    bank.family_sets = [{"gpt"}, {"claude"}, {"gpt", "claude"}, {"qwen"}]
    assert list(bank.eligible_calibrators(0)) == [1, 3]


def test_holm_and_rank_biserial():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    res = paired_contrast(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.6, 0.7, 0.4]))
    assert res["wins"] == 4 and res["mean_diff"] > 0


def test_expected_posterior_variance_matches_brute_force():
    from eost.replay import _expected_posterior_var
    rng = np.random.default_rng(5)
    K, I = 41, 12
    theta = np.linspace(-3, 3, K)
    pi = np.exp(-0.5 * theta ** 2); pi /= pi.sum()
    P = expit(rng.normal(1, 0.3, (I, 1)) * (theta[None] - rng.normal(0, 1, (I, 1))))
    PQ = P * (1 - P)
    S_un, V_un = P.sum(0), PQ.sum(0)
    fast = _expected_posterior_var(P, PQ, pi, S_un, V_un)
    for i in range(I):
        brute = 0.0
        others = np.delete(np.arange(I), i)
        for yv in (0, 1):
            lik = P[i] if yv else 1 - P[i]
            post = pi * lik; Z = post.sum(); post /= Z
            m_k = P[others].sum(0); v_k = PQ[others].sum(0)
            mu = m_k @ post
            brute += Z * (v_k @ post + ((m_k - mu) ** 2) @ post)
        assert abs(brute - fast[i]) < 1e-9


def test_empirical_bayes_recovers_shifted_difficulty_prior():
    rng = np.random.default_rng(7)
    N, I = 6, 800
    th = rng.normal(size=N)
    a = np.exp(rng.normal(0.3, 0.3, I)); b = rng.normal(1.5, 1.0, I)   # hard bank
    Y = (rng.random((N, I)) < expit(a[None] * (th[:, None] - b[None]))).astype(float)
    cal, W, grid = irt.calibrate(Y, "2pl", irt.Priors(), empirical_bayes_prior=True)
    assert cal.priors.mu_b > 0.8          # moved towards the true centre 1.5
    assert np.allclose(W.sum(1), 1)
    cal1, W1, g1 = irt.calibrate(Y, "1pl", irt.Priors())
    assert W1.shape == (I, len(g1.b)) and np.allclose(W1.sum(1), 1)


@pytest.mark.parametrize("model", ["2pl", "1pl"])
def test_conditional_item_map_matches_joint_map(model):
    Y, *_ = simulate(N=30, I=60, seed=9)
    pr = irt.Priors(mu_b=0.2, sd_b=1.5)
    cal = irt.fit_map(Y, model, pr)
    a, b = irt.fit_items_given_theta(Y, cal.theta_cal, pr, model,
                                     a_common=float(cal.a[0]) if model == "1pl" else None)
    assert np.max(np.abs(b - cal.b)) < 5e-3
    assert np.max(np.abs(np.log(a) - np.log(cal.a))) < 5e-3


@pytest.mark.parametrize("a,s", [(0.5, 0.3), (2.0, 0.5), (6.0, 0.4), (12.0, 1.0)])
def test_information_cap_of_expected_response_function_logistic(a, s):
    # Proposition 4: sup_theta Fisher information of the ERF < 2 / (pi s^2) (logistic link, b ~ N(0, s^2))
    bs = np.linspace(-8 * s, 8 * s, 6001)
    w = np.exp(-0.5 * (bs / s) ** 2); w /= w.sum()
    theta = np.linspace(-6, 6, 2401)
    S = expit(a * (theta[None, :] - bs[:, None]))
    P = w @ S
    dP = w @ (a * S * (1 - S))
    fisher = dP ** 2 / (P * (1 - P))
    assert fisher.max() < 2 / (np.pi * s ** 2)
    assert fisher.max() < a ** 2 / 4 + 1e-12          # never more than the plug-in maximum


def test_score_targeted_reduction_closed_form():
    # Proposition 5: R_i = Var(S_U) - E[Var(S_{U\i} | y_i)] = (v_i + c_i)^2 / v_i
    from eost.replay import _expected_posterior_var
    rng = np.random.default_rng(11)
    K, I = 61, 9
    theta = np.linspace(-3, 3, K)
    pi = np.exp(-0.5 * (theta / 0.8) ** 2); pi /= pi.sum()
    P = expit(rng.uniform(0.5, 3, (I, 1)) * (theta[None] - rng.normal(0, 1, (I, 1))))
    PQ = P * (1 - P)
    S_un, V_un = P.sum(0), PQ.sum(0)
    mean = S_un @ pi
    var_total = V_un @ pi + ((S_un - mean) ** 2) @ pi
    ev = _expected_posterior_var(P, PQ, pi, S_un, V_un)
    for i in range(I):
        pbar = P[i] @ pi
        v = pbar * (1 - pbar)
        M = S_un - P[i]
        c = (P[i] * M) @ pi - pbar * (M @ pi)
        assert abs((var_total - ev[i]) - (v + c) ** 2 / v) < 1e-9


def test_erf_information_below_posterior_expected_information():
    # Corollary (Jensen): Fisher information of the ERF <= posterior-expected Fisher information
    rng = np.random.default_rng(3)
    theta = np.linspace(-4, 4, 401)
    a = np.exp(rng.normal(0.3, 0.6, 300)); b = rng.normal(0.2, 0.9, 300)
    w = rng.dirichlet(np.ones(300))
    S = expit(a[:, None] * (theta[None] - b[:, None]))
    dS = a[:, None] * S * (1 - S)
    P, dP = w @ S, w @ dS
    fi_erf = dP ** 2 / (P * (1 - P))
    fi_avg = w @ (dS ** 2 / (S * (1 - S)))
    assert np.all(fi_erf <= fi_avg + 1e-12)


def test_evr_guard_is_noop_on_valid_probabilities_and_prevents_overflow():
    # Post-freeze numerical guard used by the checkpointed runner (configs/deviations.txt, item iii)
    import importlib.util
    import os
    import eost.replay as R
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_benchmark_resumable.py")
    spec = importlib.util.spec_from_file_location("rbr", path)
    rbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbr)
    frozen = R._expected_posterior_var
    rng = np.random.default_rng(5)
    K, I = 61, 12
    theta = np.linspace(-3, 3, K)
    pi = np.exp(-0.5 * (theta / 0.4) ** 2); pi /= pi.sum()
    P = expit(rng.uniform(0.5, 3, (I, 1)) * (theta[None] - rng.normal(0, 1, (I, 1))))
    S_un, V_un = 400 * P.sum(0), 400 * (P * (1 - P)).sum(0)
    try:
        rbr.install_evr_guard()
        guarded = R._expected_posterior_var
        # exact no-op when probabilities are valid
        assert np.array_equal(guarded(P, P * (1 - P), pi, S_un, V_un), frozen(P, P * (1 - P), pi, S_un, V_un))
        # a saturated item whose expected response probability exceeds 1 by rounding
        Pbad = P.copy(); Pbad[0] = 1.0 + 2.3e-16
        PQ = np.clip(Pbad, 1e-12, 1 - 1e-12) * (1 - np.clip(Pbad, 1e-12, 1 - 1e-12))
        with np.errstate(all="ignore"):
            bad = frozen(Pbad, PQ, pi, S_un, V_un)
        good = guarded(Pbad, PQ, pi, S_un, V_un)
        assert not np.isfinite(bad[0]) or abs(bad[0]) > 1e100   # the frozen computation blows up
        assert np.all(np.isfinite(good)) and np.allclose(good[1:], frozen(P, P * (1 - P), pi, S_un, V_un)[1:])
    finally:
        R._expected_posterior_var = frozen
