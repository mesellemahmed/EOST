"""Study 2: frozen confirmatory comparison against strong competitors.

Design (see configs/study2_plan.md):
* strict family clusters (heuristic families joined with near-duplicates whose IRT
  residual correlation is >= 0.5; joint over the six Open-LLM-Leaderboard banks);
* at most one target per cluster (new seeds; systems that were not targets of study 1
  are preferred as cluster representatives), so targets are independent units;
* calibration panels drawn outside the target's cluster;
* policies: EOST-M (EIG, calibration-marginal 2PL, EB prior), regularized Rasch-type
  plug-in CAT, regularized one-parameter marginal model, fully Bayesian CAT (Bayesian
  maximum information and EIG), Random:mg, naive subsampling, plug-in 2PL CAT,
  ATLAS-style 3PL CAT, tinyBenchmarks, MINCE, two empirical-Bayes variants and an
  exploratory mixed policy;
* leave-one-calibrator(-cluster)-out recalibration of the score interval for EOST-M and
  the Rasch-type test.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

from .data import Bank, stratified_targets
from .experiment import H0_STD_NORMAL, _seed
from .fullbayes import expected_fisher, gibbs_item_marginals
from .irt import (Priors, _softmax_rows, calibrate, default_theta_grid, fit_items_given_theta, fit_map,
                  grid_logprior, grid_loglik, marginal_model, plugin_model)
from .irt2 import crossfit_posterior, fit_3pl_fixed_c, plugin_model_3pl, t_prior_posterior
from .reduction import mince, tiny_estimates
from .replay import Precomputed, reference_theta, replay
from .replay2 import (Z90, chooser_atlas, chooser_mix, recalibration_factors, replay_chooser,
                      standardized_errors, summarize2)

PRIOR_1PR = Priors(sd_log_a=0.05, sd_b=2.0)
OLLB = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "ollb_truthfulqa"]
CHANCE = {"ollb_arc": 0.25, "ollb_hellaswag": 0.25, "ollb_mmlu": 0.25, "ollb_winogrande": 0.5}


@dataclass
class S2Config:
    master_seed: int = 20261104
    n_targets: int = 40
    replicates: int = 3
    panel_sizes: tuple = (4, 8, 16)
    primary_N: int = 8
    budget: int = 100
    holdout_frac: float = 0.2
    theta_lo: float = -5.0
    theta_hi: float = 5.0
    theta_K: int = 201
    resid_thr: float = 0.5
    loco_N: tuple = (4, 8, 16)
    fb_N: tuple = (4, 8)
    eb_arms_N: tuple = (8,)
    fb_burn: int = 10
    fb_draws: int = 8
    fb_thin: int = 4
    reduction_T: tuple = (25, 50, 100)
    mince_R: int = 2000
    mince_step: int = 25
    mince_tau: float = 0.01
    mix_every: int = 4
    recal_window: float = 1.5

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ clusters
def residual_matrix(Y: np.ndarray) -> np.ndarray:
    """Standardized IRT residuals of a 2PL joint-MAP fit on all systems (rows normalized)."""
    Y = np.asarray(Y, dtype=float)
    cal = fit_map(Y, "2pl", Priors())
    P = 1.0 / (1.0 + np.exp(-cal.a[None, :] * (cal.theta_cal[:, None] - cal.b[None, :])))
    R = (Y - P) / np.sqrt(np.clip(P * (1 - P), 1e-4, None))
    return R - R.mean(1, keepdims=True)


def components(A: np.ndarray) -> np.ndarray:
    J = A.shape[0]
    lab = -np.ones(J, dtype=int)
    c = 0
    for s in range(J):
        if lab[s] >= 0:
            continue
        stack = [s]; lab[s] = c
        while stack:
            x = stack.pop()
            for y in np.flatnonzero(A[x]):
                if lab[y] < 0:
                    lab[y] = c; stack.append(y)
        c += 1
    return lab


def strict_clusters_from(banks: list[Bank], thr: float) -> np.ndarray:
    """Clusters over the systems shared by `banks` (same system order required)."""
    sys0 = banks[0].systems
    H = np.zeros((banks[0].J, banks[0].J), dtype=bool)
    Rs = []
    for B in banks:
        assert (B.systems == sys0).all()
        for j in range(B.J):
            H[j] |= B.related(j)
        Rs.append(residual_matrix(B.Y))
    R = np.concatenate(Rs, 1)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    C = R @ R.T
    return components(H | H.T | (C >= thr))


# ------------------------------------------------------------------ targets / panels
def choose_targets_s2(bank: Bank, clusters: np.ndarray, cfg: S2Config, avoid=(), eligible_systems=None):
    """One representative per cluster (preferring systems outside `avoid`), then at most
    n_targets representatives stratified by accuracy; each needs >= max(N) calibrators."""
    rng = np.random.default_rng(_seed(cfg.master_seed, bank.name, "targets"))
    avoid = set(int(a) for a in avoid)
    pool_sys = np.arange(bank.J) if eligible_systems is None else np.asarray(eligible_systems)
    reps = []
    for c in np.unique(clusters[pool_sys]):
        mem = pool_sys[clusters[pool_sys] == c]
        pref = np.array([m for m in mem if int(m) not in avoid])
        cand = pref if len(pref) else mem
        reps.append(int(rng.choice(cand)))
    reps = np.array(sorted(reps))
    ok = [t for t in reps if (clusters != clusters[t]).sum() >= max(cfg.panel_sizes)]
    reps = np.array(ok, dtype=int)
    if len(reps) > cfg.n_targets:
        idx = stratified_targets(bank.Y[reps], cfg.n_targets, rng)
        reps = reps[idx]
    return np.sort(reps)


def eligible_s2(bank: Bank, clusters: np.ndarray, t: int, allowed=None) -> np.ndarray:
    ok = clusters != clusters[t]
    if allowed is not None:
        m = np.zeros(bank.J, dtype=bool); m[np.asarray(allowed)] = True
        ok &= m
    return np.flatnonzero(ok)


# ------------------------------------------------------------------ LOCO recalibration
def loco_marginal(Yp, theta_cal, pr, grid, clusters_p, Y_all_p, pool, budget, theta, sd_theta, seed_base):
    """Standardized errors of EOST-M replays on each calibrator, item posteriors refitted
    without that calibrator's cluster (abilities and hyperprior held at the panel fit)."""
    N = Yp.shape[0]
    Z, E = [], []
    lp = grid_logprior(grid, pr, "2pl")[None, :]
    for c in range(N):
        keep = clusters_p != clusters_p[c]
        if keep.sum() < 2:
            continue
        W = _softmax_rows(grid_loglik(Yp[keep], theta_cal[keep], grid) + lp)
        pre = Precomputed.build(marginal_model(W, grid, theta), sd_theta)
        yc = Y_all_p[c].astype(np.int64)
        out = replay(yc, pre, pool, np.array([], dtype=int), "eig", budget, np.random.default_rng(seed_base + c))
        Z.append(standardized_errors(out, float(yc.mean())))
        E.append(out["est"] - float(yc.mean()))
    return np.array(Z), np.array(E)


def loco_rasch(Yp, cal1, clusters_p, Y_all_p, pool, budget, theta, sd_theta, seed_base):
    N = Yp.shape[0]
    Z, E = [], []
    for c in range(N):
        keep = clusters_p != clusters_p[c]
        if keep.sum() < 2:
            continue
        a, b = fit_items_given_theta(Yp[keep], cal1.theta_cal[keep], cal1.priors, "1pl",
                                     a_common=float(cal1.a[0]), b0=cal1.b)
        pre = Precomputed.build(plugin_model(SimpleNamespace(a=a, b=b), theta), sd_theta)
        yc = Y_all_p[c].astype(np.int64)
        out = replay(yc, pre, pool, np.array([], dtype=int), "mfi", budget, np.random.default_rng(seed_base + c))
        Z.append(standardized_errors(out, float(yc.mean())))
        E.append(out["est"] - float(yc.mean()))
    return np.array(Z), np.array(E)


# ------------------------------------------------------------------ one target
def _model_row(out, kappas, extra):
    s = summarize2(out, kappas)
    s.update(extra)
    return s


def run_target_s2(bank: Bank, clusters: np.ndarray, t: int, cfg: S2Config, allowed=None,
                  component: str = "main"):
    Y = bank.Y
    y = Y[t].astype(np.int64)
    I = bank.I
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    elig = eligible_s2(bank, clusters, t, allowed)
    c_chance = CHANCE.get(bank.name, 0.0)
    rows, diags = [], []
    for r in range(cfg.replicates):
        rng_h = np.random.default_rng(_seed(cfg.master_seed, bank.name, component, "holdout", t, r))
        perm = rng_h.permutation(I)
        nh = int(round(cfg.holdout_frac * I))
        holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
        for N in cfg.panel_sizes:
            if N > len(elig):
                continue
            rng_p = np.random.default_rng(_seed(cfg.master_seed, bank.name, component, "panel", t, r, N))
            panel = np.sort(rng_p.choice(elig, size=N, replace=False))
            Yp = Y[panel].astype(float)
            cl_p = clusters[panel]
            base = dict(dataset=bank.name, component=component, target=int(t), system=str(bank.systems[t]),
                        cluster=int(clusters[t]), rep=r, N=N)
            sd = lambda *k: _seed(cfg.master_seed, bank.name, component, *k, t, r, N)
            timing = {}
            t0 = time.time()
            cal, W, grid = calibrate(Yp, "2pl", Priors(), True)
            m_mg, m_pi = marginal_model(W, grid, theta), plugin_model(cal, theta)
            pre = {"mg": Precomputed.build(m_mg, 1.0), "pi": Precomputed.build(m_pi, 1.0)}
            timing["cal2pl"] = time.time() - t0
            t0 = time.time()
            cal1, W1, grid1 = calibrate(Yp, "1pl", PRIOR_1PR, True)
            pre["1pr"] = Precomputed.build(plugin_model(cal1, theta), 1.0)
            pre["1prmg"] = Precomputed.build(marginal_model(W1, grid1, theta), 1.0)
            timing["cal1pl"] = time.time() - t0
            fbd = {}
            if N in cfg.fb_N:
                t0 = time.time()
                W_fb, fbd = gibbs_item_marginals(Yp, cal.theta_cal, cal.priors, grid,
                                                 np.random.default_rng(sd("fb")), burn=cfg.fb_burn,
                                                 draws=cfg.fb_draws, thin=cfg.fb_thin)
                m_fb = marginal_model(W_fb, grid, theta)
                pre["fb"] = Precomputed.build(m_fb, 1.0)
                p_bmi = Precomputed.build(m_fb, 1.0)
                p_bmi.FI = expected_fisher(W_fb, grid, theta)
                pre["fbbmi"] = p_bmi
                timing["fb"] = time.time() - t0
            t0 = time.time()
            a3, b3 = fit_3pl_fixed_c(Yp, cal.theta_cal, cal.priors, c_chance, grid)
            pre["3pl"] = Precomputed.build(plugin_model_3pl(a3, b3, c_chance, theta), 1.0)
            timing["cal3pl"] = time.time() - t0
            if N in cfg.eb_arms_N:
                t0 = time.time()
                W_xf, _, prs_xf = crossfit_posterior(Yp, cal.theta_cal, Priors(), np.random.default_rng(sd("xf")), grid)
                pre["xf"] = Precomputed.build(marginal_model(W_xf, grid, theta), 1.0)
                W_t, _, pr_t, info_t = t_prior_posterior(Yp, cal.theta_cal, Priors(), grid)
                pre["tp"] = Precomputed.build(marginal_model(W_t, grid, theta), 1.0)
                timing["eb_arms"] = time.time() - t0
            # LOCO recalibration factors
            kap_mg = kap_1pr = None
            if N in cfg.loco_N:
                t0 = time.time()
                Zmg, Emg = loco_marginal(Yp, cal.theta_cal, cal.priors, grid, cl_p, Y[panel], pool, cfg.budget,
                                         theta, 1.0, sd("loco_mg") % (2 ** 31))
                Z1, E1 = loco_rasch(Yp, cal1, cl_p, Y[panel], pool, cfg.budget, theta, 1.0,
                                    sd("loco_1pr") % (2 ** 31))
                if len(Zmg):
                    kap_mg = {"Rq": recalibration_factors(Zmg, "quantile", cfg.recal_window),
                              "Rr": recalibration_factors(Zmg, "rms", cfg.recal_window),
                              "Ra": ("abs", recalibration_factors(Emg * Z90, "quantile", cfg.recal_window))}
                if len(Z1):
                    kap_1pr = {"Rq": recalibration_factors(Z1, "quantile", cfg.recal_window),
                               "Rr": recalibration_factors(Z1, "rms", cfg.recal_window),
                               "Ra": ("abs", recalibration_factors(E1 * Z90, "quantile", cfg.recal_window))}
                timing["loco"] = time.time() - t0
            # policies
            t0 = time.time()
            plan = [("eig:mg", "eig", "mg", kap_mg), ("mfi:1pr", "mfi", "1pr", kap_1pr),
                    ("eig:1prmg", "eig", "1prmg", None), ("random:mg", "random", "mg", None),
                    ("naive", "naive", None, None), ("mfi:pi", "mfi", "pi", None)]
            if N in cfg.fb_N:
                plan += [("fb:bmi", "pwfi", "fbbmi", None), ("fb:eig", "eig", "fb", None)]
            if N in cfg.eb_arms_N:
                plan += [("eig:xf", "eig", "xf", None), ("eig:tp", "eig", "tp", None)]
            for pol, crit, key, kap in plan:
                rng_s = np.random.default_rng(sd("policy", pol))
                out = replay(y, pre.get(key), pool, holdout, crit, cfg.budget, rng_s)
                if kap is not None:
                    hw = np.maximum(0.5 * (out["hi"] - out["lo"]), 1e-9)
                    kap = {k: (v[1] / hw if isinstance(v, tuple) else v) for k, v in kap.items()}
                s = _model_row(out, kap, base | {"policy": pol})
                if key is not None:
                    th = out["theta_mean"]
                    ref_m, _ = reference_theta(y, pre[key], pool)
                    s["iref_err"] = float(np.mean(np.abs(th - ref_m)))
                    s["iinfo"] = float(np.mean(H0_STD_NORMAL - out["post_H"]))
                rows.append(s)
            for pol, chooser, key, state in [("atlas", chooser_atlas, "3pl", {"b": b3}),
                                             ("mix:mg", chooser_mix, "mg", {})]:
                rng_s = np.random.default_rng(sd("policy", pol))
                out = replay_chooser(y, pre[key], pool, holdout, chooser, cfg.budget, rng_s, state)
                rows.append(_model_row(out, None, base | {"policy": pol}))
            timing["replays"] = time.time() - t0
            # benchmark-reduction baselines (non-adaptive)
            t0 = time.time()
            A = float(y.mean())
            tb = tiny_estimates(y, Yp, cal.theta_cal, cal.priors, cal.a, cal.b, pool, cfg.reduction_T,
                                np.random.default_rng(sd("tiny")), theta)
            mc = mince(y, Yp, pool, cfg.reduction_T, np.random.default_rng(sd("mince")), R=cfg.mince_R,
                       step=cfg.mince_step, tau=cfg.mince_tau)
            for T in cfg.reduction_T:
                for est in ("gpirt", "pirt", "anchor"):
                    rows.append(base | {"policy": f"tiny:{est}", "T": T, "A_true": A,
                                        f"err@{T}": abs(tb[T][est] - A), "lambda": tb[T]["lambda"]})
                rows.append(base | {"policy": "mince", "T": T, "A_true": A, f"err@{T}": abs(mc[T]["est"] - A),
                                    f"cov@{T}": mc[T]["covered"], f"width@{T}": 2 * mc[T]["bound"]})
            rows.append(base | {"policy": "mince:nstar", "A_true": A, "n_star": mc["n_star"],
                                "err_nstar": mc["err_nstar"], "p95_nstar": mc["p95_nstar"]})
            timing["reduction"] = time.time() - t0
            d = {**base, **fbd, "common_a_1pr": float(cal1.a[0]), "eb_sd_log_a": cal.priors.sd_log_a,
                 "eb_sd_b": cal.priors.sd_b, "eb_mu_b": cal.priors.mu_b, "eb_mu_log_a": cal.priors.mu_log_a,
                 "chance_c": c_chance}
            if N in cfg.eb_arms_N:
                d.update({"t_mu_b": pr_t.mu_b, "t_sd_b": pr_t.sd_b, "t_mu_log_a": pr_t.mu_log_a,
                          "t_sd_log_a": pr_t.sd_log_a, "xf_sd_log_a_0": prs_xf[0].sd_log_a,
                          "xf_sd_log_a_1": prs_xf[1].sd_log_a})
            if kap_mg is not None:
                d.update({"kappa_Rq_mg_med": float(np.median(kap_mg["Rq"])),
                          "kappa_Rr_mg_med": float(np.median(kap_mg["Rr"])),
                          "halfwidth_Ra_mg_med": float(np.median(kap_mg["Ra"][1]))})
            d.update({f"sec_{k}": v for k, v in timing.items()})
            diags.append(d)
    return rows, diags
