"""Synthetic response banks with known ground truth (the S0 analogue of EOST v2)."""
from __future__ import annotations

import json

import numpy as np
from scipy.special import expit


def make_bank(path: str, J: int = 300, I: int = 600, sd_log_a: float = 0.5, sd_b: float = 1.5,
              guessing: float = 0.0, frac_guess: float = 0.0, two_dim_rho: float | None = None,
              seed: int = 0, name: str = "synth") -> dict:
    """Simulate a 2PL bank (optionally with guessing or a 2-D ability) and save it.

    ``guessing``/``frac_guess`` add a lower asymptote to a fraction of items
    (multiple-choice-like misspecification); ``two_dim_rho`` draws a second
    correlated ability dimension used by half of the items (multidimensional
    misspecification). The fitted models remain unidimensional 2PL.
    """
    rng = np.random.default_rng(seed)
    th = rng.normal(size=J)
    a = np.exp(rng.normal(0.0, sd_log_a, size=I))
    b = rng.normal(0.0, sd_b, size=I)
    th_item = np.tile(th[:, None], (1, I))
    if two_dim_rho is not None:
        th2 = two_dim_rho * th + np.sqrt(1 - two_dim_rho ** 2) * rng.normal(size=J)
        dim2 = rng.random(I) < 0.5
        th_item[:, dim2] = th2[:, None]
    P = expit(a[None, :] * (th_item - b[None, :]))
    if frac_guess > 0:
        g = (rng.random(I) < frac_guess) * guessing
        P = g[None, :] + (1 - g[None, :]) * P
    Y = (rng.random((J, I)) < P).astype(np.uint8)
    systems = np.array([f"sys{j:03d}" for j in range(J)])
    meta = {"source": "synthetic", "seed": seed, "sd_log_a": sd_log_a, "sd_b": sd_b,
            "guessing": guessing, "frac_guess": frac_guess, "two_dim_rho": two_dim_rho,
            "domain": "synthetic 2PL" if two_dim_rho is None and frac_guess == 0 else "synthetic misspecified"}
    np.savez_compressed(path, Y=Y, systems=systems, families=systems.copy(),
                        items=np.array([f"item{i:03d}" for i in range(I)]), meta=json.dumps(meta),
                        theta_true=th, a_true=a, b_true=b)
    return {"name": name, "J": J, "I": I, "mean_acc": float(Y.mean())}
