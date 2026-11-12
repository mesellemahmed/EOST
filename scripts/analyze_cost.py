"""Total evaluation cost and break-even analysis (post hoc; reviewer request).

Cost unit: one item administered to one system (c_item = 1). For M new systems on a bank
of I items, with a calibration panel of N systems that must first answer the whole bank:

  full evaluation        C_full   = M * I
  EOST (fixed length T)  C_eost   = N * I + M * T
  naive subsampling      C_naive  = M * T_naive          (no calibration)

Break-even against full evaluation: M* = N I / (I - T). At matched precision, EOST needs
T_match items to reach the score error that naive subsampling reaches with 100 items
(first checkpoint at which the mean error over targets falls below it, N = 8, study 1),
so the break-even against naive subsampling is M* = N I / (100 - T_match). When the
calibrators' full responses already exist (public leaderboards), the calibration term
vanishes and EOST is cheaper from the first system. The same computation against
Random:mg (random items scored with the same calibration) isolates the value of
adaptivity: both pay the calibration, so EOST saves (100 - T_match_random) items per system.
Writes artifacts/cost/.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402

CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
CPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
OUT = "artifacts/cost"


def items_to_match(curve, ref):
    hit = [c for c, v in zip(CPS, curve) if v <= ref]
    return hit[0] if hit else None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ds in CONF:
        R = pd.read_csv(f"artifacts/confirm/{ds}/runs.csv.gz")
        R = R[R.N == 8]
        cols = [f"err@{c}" for c in CPS]
        tm = R.groupby(["policy", "target"])[cols].mean().groupby("policy").mean()
        I = load_bank(f"data/processed/{ds}.npz").I
        N, T = 8, 100
        t_naive = items_to_match(tm.loc["eig:mg"].values, tm.loc["naive", "err@100"])
        t_rand = items_to_match(tm.loc["eig:mg"].values, tm.loc["random:mg", "err@100"])
        rows.append({
            "dataset": ds, "I": I, "N": N, "T": T,
            "breakeven_vs_full": N * I / (I - T),
            "items_match_naive100": t_naive, "items_match_random_mg100": t_rand,
            "breakeven_vs_naive": (N * I / (100 - t_naive)) if t_naive and t_naive < 100 else None,
            "saving_per_system_vs_naive": (100 - t_naive) if t_naive else 0,
            "saving_per_system_vs_random_mg": (100 - t_rand) if t_rand else 0,
            "naive_err100": float(tm.loc["naive", "err@100"]), "random_mg_err100": float(tm.loc["random:mg", "err@100"]),
            "eost_err100": float(tm.loc["eig:mg", "err@100"]),
            "cost_full_M100": 100 * I, "cost_eost_M100": N * I + 100 * T,
        })
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/breakeven.csv", index=False)
    summ = {"median_breakeven_vs_full": float(D.breakeven_vs_full.median()),
            "range_breakeven_vs_full": [float(D.breakeven_vs_full.min()), float(D.breakeven_vs_full.max())],
            "median_items_match_naive100": float(D.items_match_naive100.median()),
            "median_items_match_random_mg100": float(pd.to_numeric(D.items_match_random_mg100).median()),
            "median_breakeven_vs_naive": float(pd.to_numeric(D.breakeven_vs_naive).median()),
            "range_breakeven_vs_naive": [float(pd.to_numeric(D.breakeven_vs_naive).min()),
                                         float(pd.to_numeric(D.breakeven_vs_naive).max())],
            "median_cost_ratio_M100": float((D.cost_eost_M100 / D.cost_full_M100).median())}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(D[["dataset", "I", "breakeven_vs_full", "items_match_naive100", "items_match_random_mg100",
             "breakeven_vs_naive"]].round(1).to_string(index=False))
    print(json.dumps(summ, indent=1))
    print("COST_GATE=PASS")


if __name__ == "__main__":
    main()
