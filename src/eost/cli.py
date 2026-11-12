"""Minimal command-line demo of EOST-M on one bank and one target system.

    docker compose run --rm -T demo --bank data/processed/cruxeval_o.npz --target 0 --panel-size 8 --budget 30

Calibrates a random leave-family-out panel, runs an adaptive test with the
chosen criterion and prints each administered item, the running score estimate
with its 90% interval, and the true full-bank accuracy for comparison.
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import load_bank
from .irt import Priors, calibrate, default_theta_grid, marginal_model, plugin_model
from .replay import Precomputed, replay


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--panel-size", type=int, default=8)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--policy", default="eig:mg", help="criterion:model, e.g. eig:mg, mfi:pi, entropy:mg, evr:mg")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    bank = load_bank(a.bank)
    rng = np.random.default_rng(a.seed)
    elig = bank.eligible_calibrators(a.target)
    panel = np.sort(rng.choice(elig, size=min(a.panel_size, len(elig)), replace=False))
    cal, W, grid = calibrate(bank.Y[panel].astype(float), "2pl", Priors(), True)
    theta = default_theta_grid()
    crit, _, model = a.policy.partition(":")
    mm = marginal_model(W, grid, theta) if model == "mg" else plugin_model(cal, theta)
    out = replay(bank.Y[a.target].astype(int), Precomputed.build(mm), np.arange(bank.I), np.array([], dtype=int),
                 crit, a.budget, rng)
    print(f"bank={bank.name} target={bank.systems[a.target]} panel={len(panel)} policy={a.policy}")
    print(f"empirical-Bayes item prior: {cal.priors}")
    for t, (i, est, lo, hi) in enumerate(zip(out["items"], out["est"], out["lo"], out["hi"]), 1):
        print(f"t={t:3d} item={bank.items[i]!s:>28}  verdict={int(bank.Y[a.target, i])}  "
              f"score={est:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"true full-bank accuracy = {out['A_true']:.3f}")


if __name__ == "__main__":
    main()
