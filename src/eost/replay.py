"""Adaptive-testing replay on recorded responses (post-hoc CAT simulation).

A target system has answered every item of a bank. An adaptive policy chooses
items one at a time from a candidate *pool*; the recorded verdict is revealed;
the ability posterior is updated on a grid. A fixed *holdout* set, shared by all
policies (the corrected-S4 common-holdout design), is never administered and is
used for predictive evaluation.

Selection criteria (all computed from the current grid posterior π over θ):

* ``random``  – uniform over the remaining pool;
* ``entropy`` – maximise predictive entropy H[y_i | D]  (uncertainty sampling);
* ``mfi``     – maximise Fisher information at the posterior mean (classical CAT);
* ``pwfi``    – maximise posterior-weighted Fisher information (van der Linden, 1998);
* ``eig``     – maximise the mutual information I(θ; y_i | D)
                = H[y_i | D] − E_{θ|D} H[y_i | θ]  (EOST criterion).

Each criterion is evaluated under a measurement model: plug-in (``pi``) or
calibration-marginal ERF (``mg``). ``naive`` is random selection with the plain
sample mean as score estimate (no model), the usual subsampling baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .irt import MeasurementModel

EPS = 1e-12
CHECKPOINTS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)


def h2(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


@dataclass
class Precomputed:
    """Per-model quantities reused by every replay of the same calibration."""
    model: MeasurementModel
    logP: np.ndarray
    log1mP: np.ndarray
    HP: np.ndarray
    FI: np.ndarray
    PQ: np.ndarray
    log_prior: np.ndarray

    @classmethod
    def build(cls, model: MeasurementModel, prior_sd: float = 1.0) -> "Precomputed":
        P = np.clip(model.P, EPS, 1 - EPS)
        return cls(model=model, logP=np.log(P), log1mP=np.log1p(-P), HP=h2(P), FI=model.fisher,
                   PQ=P * (1 - P), log_prior=norm.logpdf(model.theta, 0.0, prior_sd))


def _posterior(logpost):
    w = np.exp(logpost - logpost.max())
    return w / w.sum()


def _expected_posterior_var(Pc, PQc, pi, S_un, V_un):
    """Expected posterior variance of the unobserved bank total after asking each candidate.

    Goal-oriented (score-targeted) design: the quantity of interest is the bank
    total S = Σ_i y_i, not θ. Asking item i removes its own Bernoulli variance
    and updates θ. For y ∈ {0, 1} the updated posterior is π·P_i^y(1-P_i)^{1-y}.
    """
    out = np.zeros(Pc.shape[0])
    for w in (Pc * pi[None, :], (1.0 - Pc) * pi[None, :]):
        Z = np.maximum(w.sum(1), 1e-300)
        ev = (w @ V_un - (w * PQc).sum(1)) / Z
        em = (w @ S_un - (w * Pc).sum(1)) / Z
        em2 = (w @ (S_un ** 2) - 2.0 * (w * Pc) @ S_un + (w * Pc ** 2).sum(1)) / Z
        out += Z * (ev + em2 - em ** 2)
    return out


def _argmax_tiebreak(scores, rng):
    m = scores.max()
    idx = np.flatnonzero(scores >= m - 1e-12 * max(1.0, abs(m)))
    return idx[0] if len(idx) == 1 else rng.choice(idx)


def replay(y: np.ndarray, pre: Precomputed | None, pool: np.ndarray, holdout: np.ndarray,
           criterion: str, budget: int, rng: np.random.Generator, ci_level: float = 0.90) -> dict:
    """Run one adaptive test. Returns per-step arrays (length T) of metrics."""
    I = y.shape[0]
    A_true = float(y.mean())
    T = int(min(budget, len(pool)))
    z = norm.ppf(0.5 + ci_level / 2)
    out = {k: np.empty(T) for k in ["est", "lo", "hi", "brier", "logloss", "theta_mean", "theta_sd", "post_H", "item_eig"]}
    out["items"] = np.empty(T, dtype=np.int64)
    remaining = np.ones(I, dtype=bool)
    remaining[np.setdiff1d(np.arange(I), pool)] = False
    observed_sum, n_obs = 0.0, 0

    if criterion == "naive":  # model-free subsampling baseline
        order = rng.permutation(pool)[:T]
        for t, i in enumerate(order):
            observed_sum += y[i]; n_obs += 1
            p_hat = observed_sum / n_obs
            # Wilson interval with finite-population correction
            n_eff = n_obs * (I - 1) / max(I - n_obs, 1) if n_obs < I else 1e9
            den = 1 + z ** 2 / n_eff
            c = (p_hat + z ** 2 / (2 * n_eff)) / den
            hw = z * np.sqrt(p_hat * (1 - p_hat) / n_eff + z ** 2 / (4 * n_eff ** 2)) / den
            # unobserved items predicted by the running mean (for the full-bank score)
            est = (observed_sum + p_hat * (I - n_obs)) / I
            out["est"][t], out["lo"][t], out["hi"][t] = est, c - hw, c + hw
            ph = np.full(len(holdout), np.clip(p_hat, 1e-3, 1 - 1e-3))
            out["brier"][t] = np.mean((ph - y[holdout]) ** 2)
            out["logloss"][t] = -np.mean(y[holdout] * np.log(ph) + (1 - y[holdout]) * np.log(1 - ph))
            out["theta_mean"][t] = out["theta_sd"][t] = out["post_H"][t] = out["item_eig"][t] = np.nan
            out["items"][t] = i
        out["A_true"] = A_true
        return out

    m = pre.model
    theta = m.theta
    dth = theta[1] - theta[0]
    logpost = pre.log_prior.copy()
    S_un = m.P.sum(0)            # Σ_{i unobserved} P[i, k]
    V_un = pre.PQ.sum(0)         # Σ_{i unobserved} P(1-P)[i, k]
    for t in range(T):
        pi = _posterior(logpost)
        p_all = m.P @ pi
        cand = np.flatnonzero(remaining)
        if criterion == "random":
            i = rng.choice(cand)
        else:
            if criterion == "entropy":
                sc = h2(p_all[cand])
            elif criterion == "eig":
                sc = h2(p_all[cand]) - pre.HP[cand] @ pi
            elif criterion == "mfi":
                th_hat = float(theta @ pi)
                x = np.clip((th_hat - theta[0]) / dth, 0, len(theta) - 1 - 1e-9)
                k0 = int(np.floor(x)); w = x - k0
                sc = (1 - w) * pre.FI[cand, k0] + w * pre.FI[cand, k0 + 1]
            elif criterion == "pwfi":
                sc = pre.FI[cand] @ pi
            elif criterion == "evr":
                sc = -_expected_posterior_var(m.P[cand], pre.PQ[cand], pi, S_un, V_un)
            else:
                raise ValueError(criterion)
            i = cand[_argmax_tiebreak(sc, rng)]
        out["item_eig"][t] = float(h2(p_all[i]) - pre.HP[i] @ pi)
        # observe and update
        yi = y[i]
        logpost = logpost + (pre.logP[i] if yi == 1 else pre.log1mP[i])
        remaining[i] = False
        observed_sum += yi; n_obs += 1
        S_un = S_un - m.P[i]; V_un = V_un - pre.PQ[i]
        pi = _posterior(logpost)
        # full-bank score: observed responses + posterior predictive for the rest
        mean_k = observed_sum + S_un
        mu = float(mean_k @ pi)
        var = float(V_un @ pi + ((mean_k - mu) ** 2) @ pi)
        sd = np.sqrt(max(var, 0.0))
        out["est"][t] = mu / I
        out["lo"][t], out["hi"][t] = (mu - z * sd) / I, (mu + z * sd) / I
        if len(holdout):
            ph = np.clip(m.P[holdout] @ pi, 1e-6, 1 - 1e-6)
            yh = y[holdout]
            out["brier"][t] = np.mean((ph - yh) ** 2)
            out["logloss"][t] = -np.mean(yh * np.log(ph) + (1 - yh) * np.log(1 - ph))
        else:
            out["brier"][t] = out["logloss"][t] = np.nan
        tm = float(theta @ pi)
        out["theta_mean"][t] = tm
        out["theta_sd"][t] = np.sqrt(max(float(((theta - tm) ** 2) @ pi), 0.0))
        out["post_H"][t] = float(-(pi * np.log(np.maximum(pi, 1e-300))).sum() + np.log(dth))
        out["items"][t] = i
    out["A_true"] = A_true
    return out


def reference_theta(y: np.ndarray, pre: Precomputed, items: np.ndarray) -> tuple[float, float]:
    """Full-pool reference posterior mean/sd of θ under a measurement model."""
    lp = pre.log_prior + np.where(y[items, None] == 1, pre.logP[items], pre.log1mP[items]).sum(0)
    pi = _posterior(lp)
    m = float(pre.model.theta @ pi)
    return m, float(np.sqrt(((pre.model.theta - m) ** 2) @ pi))


def summarize(out: dict, eps_list=(0.02, 0.03, 0.05)) -> dict:
    """Scalar endpoints for one replay."""
    est, A = out["est"], out["A_true"]
    err = np.abs(est - A)
    T = len(est)
    cover = (out["lo"] <= A) & (A <= out["hi"])
    s = {
        "T": T, "A_true": A,
        "iae": float(err.mean()),                 # integrated absolute score error (primary)
        "final_err": float(err[-1]),
        "ibrier": float(np.mean(out["brier"])),
        "ilogloss": float(np.mean(out["logloss"])),
        "final_brier": float(out["brier"][-1]),
        "coverage": float(cover.mean()),
        "mean_width": float(np.mean(out["hi"] - out["lo"])),
        "mean_item_eig": float(np.nanmean(out["item_eig"])) if np.isfinite(out["item_eig"]).any() else np.nan,
    }
    for e in eps_list:
        ok = err <= e
        # sustained crossing: first t such that all later errors are within tolerance
        bad = np.flatnonzero(~ok)
        s[f"ttol_{int(e*100):02d}"] = int(bad[-1] + 2) if len(bad) else 1
        if len(bad) and bad[-1] == T - 1:
            s[f"ttol_{int(e*100):02d}"] = T + 1          # censored: never sustained
    for c in CHECKPOINTS:
        if c <= T:
            s[f"err@{c}"] = float(err[c - 1])
            s[f"cov@{c}"] = float(cover[c - 1])
            s[f"brier@{c}"] = float(out["brier"][c - 1])
            s[f"thsd@{c}"] = float(out["theta_sd"][c - 1])
    return s
