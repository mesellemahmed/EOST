"""Leave-family-out adaptive-testing experiment on one response matrix.

For every target system t, replicate r and panel size N:
  1. a fixed holdout H_{t,r} (shared by all policies and panel sizes) is drawn;
  2. a calibration panel of N systems is drawn among systems unrelated to t;
  3. the panel is calibrated (joint MAP 2PL or 1PL) and two measurement
     models are built: plug-in and calibration-marginal (ERF);
  4. every policy is replayed on t's recorded responses from the pool P = I \\ H.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import Bank, effective_panel_size, stratified_targets
from .irt import Priors, calibrate, default_theta_grid, marginal_model, plugin_model
from .replay import Precomputed, reference_theta, replay, summarize

H0_STD_NORMAL = 0.5 * np.log(2 * np.pi * np.e)


@dataclass
class ExpConfig:
    panel_sizes: tuple = (4, 8, 16, 32, 64)
    replicates: int = 5
    n_targets: int = 60
    budget: int = 100
    holdout_frac: float = 0.2
    irt_model: str = "2pl"
    priors: Priors = field(default_factory=Priors)
    policies: tuple = ("naive", "random:pi", "random:mg", "entropy:pi", "mfi:pi", "eig:pi", "evr:pi",
                       "entropy:mg", "mfi:mg", "pwfi:mg", "eig:mg", "evr:mg")
    max_kappa: float | None = None
    theta_lo: float = -5.0
    theta_hi: float = 5.0
    theta_K: int = 201
    empirical_bayes: bool = True
    master_seed: int = 20261029

    def as_dict(self):
        d = asdict(self)
        d["priors"] = asdict(self.priors)
        return d


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def calibration_diagnostics(Yp: np.ndarray, cal, mg, pi) -> dict:
    col = Yp.sum(0)
    N = Yp.shape[0]
    patterns = len({row.tobytes() for row in Yp.T})
    return {
        "n_eff": effective_panel_size(Yp),
        "frac_items_extreme": float(np.mean((col == 0) | (col == N))),
        "n_patterns": int(patterns),
        "frac_a_gt_4": float(np.mean(cal.a > 4)),
        "median_a_map": float(np.median(cal.a)),
        "median_a_eff_mg": float(np.median(mg.item_a_eff)),
        "panel_acc_min": float(Yp.mean(1).min()),
        "panel_acc_max": float(Yp.mean(1).max()),
        "converged": bool(cal.converged),
        "eb_mu_b": cal.priors.mu_b, "eb_sd_b": cal.priors.sd_b,
        "eb_mu_log_a": cal.priors.mu_log_a, "eb_sd_log_a": cal.priors.sd_log_a,
    }


def run_target(bank: Bank, t: int, cfg: ExpConfig, grid_cache: dict) -> tuple[list, list]:
    rows, diag = [], []
    Y = bank.Y
    y = Y[t].astype(np.int64)
    I = bank.I
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    elig = bank.eligible_calibrators(t, cfg.max_kappa)
    for r in range(cfg.replicates):
        rng_h = np.random.default_rng(_seed(cfg.master_seed, bank.name, "holdout", t, r))
        perm = rng_h.permutation(I)
        nh = int(round(cfg.holdout_frac * I))
        holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
        for N in cfg.panel_sizes:
            if N > len(elig):
                continue
            rng_p = np.random.default_rng(_seed(cfg.master_seed, bank.name, "panel", t, r, N))
            panel = np.sort(rng_p.choice(elig, size=N, replace=False))
            Yp = Y[panel].astype(float)
            t0 = time.time()
            cal, W, grid = calibrate(Yp, cfg.irt_model, cfg.priors, cfg.empirical_bayes)
            models = {"pi": plugin_model(cal, theta), "mg": marginal_model(W, grid, theta)}
            pre = {k: Precomputed.build(m, cfg.priors.sd_theta) for k, m in models.items()}
            refs = {k: reference_theta(y, p, pool) for k, p in pre.items()}
            d = calibration_diagnostics(Y[panel], cal, models["mg"], models["pi"])
            d.update({"dataset": bank.name, "target": int(t), "rep": r, "N": N,
                      "calib_seconds": time.time() - t0})
            diag.append(d)
            for pol in cfg.policies:
                crit, _, mname = pol.partition(":")
                rng_s = np.random.default_rng(_seed(cfg.master_seed, bank.name, "policy", t, r, N, pol))
                out = replay(y, pre.get(mname), pool, holdout, crit, cfg.budget, rng_s)
                s = summarize(out)
                if mname:
                    ref_m, ref_sd = refs[mname]
                    th = out["theta_mean"]
                    s["iref_err"] = float(np.mean(np.abs(th - ref_m)))
                    s["final_ref_err"] = float(abs(th[-1] - ref_m))
                    s["iinfo"] = float(np.mean(H0_STD_NORMAL - out["post_H"]))
                    s["ref_theta"] = ref_m
                    if bank.theta_true is not None:
                        tt = float(bank.theta_true[t])
                        sd = out["theta_sd"]
                        s["itrue_err"] = float(np.mean(np.abs(th - tt)))
                        s["final_true_err"] = float(abs(th[-1] - tt))
                        s["true_theta_cov"] = float(np.mean(np.abs(th - tt) <= 1.6448536 * sd))
                s.update({"dataset": bank.name, "target": int(t), "system": str(bank.systems[t]),
                          "rep": r, "N": N, "policy": pol})
                rows.append(s)
    return rows, diag


def choose_targets(bank: Bank, cfg: ExpConfig) -> np.ndarray:
    rng = np.random.default_rng(_seed(cfg.master_seed, bank.name, "targets"))
    cand = stratified_targets(bank.Y, cfg.n_targets, rng)
    # a target needs at least min(panel_sizes) eligible calibrators
    keep = [t for t in cand if len(bank.eligible_calibrators(t, cfg.max_kappa)) >= min(cfg.panel_sizes)]
    return np.array(keep, dtype=int)
