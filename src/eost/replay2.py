"""Replay extensions for study 2 (the frozen ``replay.py`` of study 1 is not modified).

* :func:`replay_chooser` is a line-by-line copy of the frozen ``replay`` loop (same
  posterior update, same full-bank score estimate and interval, same holdout metrics)
  with a pluggable item chooser. With the EIG chooser it reproduces ``replay`` exactly
  (unit-tested), and it adds the ATLAS-style chooser (first item closest to the prior
  mean, then a uniform draw among the five most informative items at the EAP) and a
  mixed chooser (every k-th item uniform at random, otherwise EIG).
* :func:`summarize2` adds the integrated interval score and, given step-wise
  recalibration factors, the coverage, width and interval score of the recalibrated
  interval ``est +/- kappa_t * half-width``.
* :func:`recalibration_factors` turns leave-one-calibrator-out standardized errors
  into step-wise factors (window-pooled 90% quantile, or root mean square).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .replay import CHECKPOINTS, Precomputed, _argmax_tiebreak, _posterior, h2, summarize

ALPHA = 0.10
Z90 = float(norm.ppf(0.95))


# ------------------------------------------------------------------ choosers
def chooser_eig(t, cand, pi, p_all, pre, theta, rng, state):
    sc = h2(p_all[cand]) - pre.HP[cand] @ pi
    return cand[_argmax_tiebreak(sc, rng)]


def chooser_pwfi(t, cand, pi, p_all, pre, theta, rng, state):
    return cand[_argmax_tiebreak(pre.FI[cand] @ pi, rng)]


def chooser_atlas(t, cand, pi, p_all, pre, theta, rng, state, top=5):
    """ATLAS (Li et al., 2026): first item with difficulty closest to theta_0 = 0, then a
    uniform draw among the `top` items with the largest Fisher information at the EAP."""
    if t == 0:
        d = np.abs(state["b"][cand] - 0.0)
        return cand[_argmax_tiebreak(-d, rng)]
    th_hat = float(theta @ pi)
    dth = theta[1] - theta[0]
    x = np.clip((th_hat - theta[0]) / dth, 0, len(theta) - 1 - 1e-9)
    k0 = int(np.floor(x)); w = x - k0
    sc = (1 - w) * pre.FI[cand, k0] + w * pre.FI[cand, k0 + 1]
    k = min(top, len(cand))
    best = np.argpartition(-sc, k - 1)[:k]
    return cand[rng.choice(best)]


def chooser_mix(t, cand, pi, p_all, pre, theta, rng, state, every=4):
    """Every `every`-th item (t = every-1, 2*every-1, ...) uniform at random, otherwise EIG."""
    if (t + 1) % every == 0:
        return rng.choice(cand)
    return chooser_eig(t, cand, pi, p_all, pre, theta, rng, state)


CHOOSERS = {"eig": chooser_eig, "pwfi": chooser_pwfi, "atlas": chooser_atlas, "mix": chooser_mix}


def replay_chooser(y, pre: Precomputed, pool, holdout, chooser, budget, rng, state=None, ci_level=0.90):
    """Copy of the frozen ``replay`` loop with a pluggable chooser (model-based policies only)."""
    I = y.shape[0]
    A_true = float(y.mean())
    T = int(min(budget, len(pool)))
    z = norm.ppf(0.5 + ci_level / 2)
    out = {k: np.empty(T) for k in ["est", "lo", "hi", "brier", "logloss", "theta_mean", "theta_sd", "post_H", "item_eig"]}
    out["items"] = np.empty(T, dtype=np.int64)
    remaining = np.ones(I, dtype=bool)
    remaining[np.setdiff1d(np.arange(I), pool)] = False
    observed_sum, n_obs = 0.0, 0
    m = pre.model
    theta = m.theta
    dth = theta[1] - theta[0]
    logpost = pre.log_prior.copy()
    S_un = m.P.sum(0)
    V_un = pre.PQ.sum(0)
    state = {} if state is None else state
    for t in range(T):
        pi = _posterior(logpost)
        p_all = m.P @ pi
        cand = np.flatnonzero(remaining)
        i = chooser(t, cand, pi, p_all, pre, theta, rng, state)
        out["item_eig"][t] = float(h2(p_all[i]) - pre.HP[i] @ pi)
        yi = y[i]
        logpost = logpost + (pre.logP[i] if yi == 1 else pre.log1mP[i])
        remaining[i] = False
        observed_sum += yi; n_obs += 1
        S_un = S_un - m.P[i]; V_un = V_un - pre.PQ[i]
        pi = _posterior(logpost)
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


# ------------------------------------------------------------- endpoints
def interval_score(lo, hi, A, alpha=ALPHA):
    """Winkler/Gneiting-Raftery interval score of central (1-alpha) intervals (lower is better)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - A, 0.0) + (2.0 / alpha) * np.maximum(A - hi, 0.0)


def recalibrated(out, kappa):
    est, lo, hi = out["est"], out["lo"], out["hi"]
    hw = 0.5 * (hi - lo)
    return est - kappa * hw, est + kappa * hw


def summarize2(out, kappas: dict | None = None):
    """Frozen endpoints + interval score (+ recalibrated intervals for each kappa vector)."""
    s = summarize(out)
    A = out["A_true"]
    s["iis"] = float(np.mean(interval_score(out["lo"], out["hi"], A)))
    T = len(out["est"])
    for c in CHECKPOINTS:
        if c <= T:
            s[f"width@{c}"] = float(out["hi"][c - 1] - out["lo"][c - 1])
    for name, kap in (kappas or {}).items():
        lo, hi = recalibrated(out, kap)
        cov = (lo <= A) & (A <= hi)
        s[f"coverage_{name}"] = float(cov.mean())
        s[f"mean_width_{name}"] = float(np.mean(hi - lo))
        s[f"iis_{name}"] = float(np.mean(interval_score(lo, hi, A)))
        s[f"kappa_med_{name}"] = float(np.median(kap))
        for c in CHECKPOINTS:
            if c <= T:
                s[f"cov@{c}_{name}"] = float(cov[c - 1])
    return s


def standardized_errors(out, A):
    """z_t = (est_t - A) / sd_t with sd_t the model sd of the score (half-width / z_0.95)."""
    hw = 0.5 * (out["hi"] - out["lo"])
    sd = np.maximum(hw / Z90, 1e-9)
    return (out["est"] - A) / sd


def recalibration_factors(Z: np.ndarray, how: str = "quantile", window: float = 1.5, level: float = 0.90):
    """Step-wise factors from pseudo-target standardized errors Z (n_pseudo x T).

    The factor at step t pools |z| over all pseudo-targets and the steps t' with
    t / window <= t' <= t * window (1-based), and is either the `level` quantile of
    |z| divided by z_0.95 ('quantile') or the root mean square of z ('rms').
    """
    Z = np.asarray(Z, dtype=float)
    n, T = Z.shape
    steps = np.arange(1, T + 1)
    kap = np.empty(T)
    for t in range(1, T + 1):
        sel = (steps >= t / window) & (steps <= t * window)
        v = np.abs(Z[:, sel]).ravel()
        v = v[np.isfinite(v)]
        if len(v) == 0:
            kap[t - 1] = 1.0
        elif how == "quantile":
            kap[t - 1] = float(np.quantile(v, level) / Z90)
        elif how == "rms":
            kap[t - 1] = float(np.sqrt(np.mean(v ** 2)))
        else:
            raise ValueError(how)
    return np.maximum(kap, 1e-3)
