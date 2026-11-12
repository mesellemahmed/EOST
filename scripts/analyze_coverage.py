"""Exploratory analysis PH3 (plan: docs/RESEARCH_LOG.md, written before computing).

Where does the better coverage of the regularized Rasch-type plug-in test come from?
Decomposes coverage of the 90% score intervals into interval width and score error,
using the existing PH1 artifacts only (no new runs). N = 8 calibrators.

Per bank and policy: target-level means of coverage, mean interval width and score
IAE; the width-to-error ratio mean_width / iae (a calibrated Gaussian 90% interval has
ratio 2 * 1.645 / sqrt(2 / pi) = 4.12); paired Wilcoxon of mean_width (mfi:1pr vs
eig:mg) per bank with Holm over the 8 banks; Spearman correlation between the ratio
and coverage over all (bank, policy) cells. Applies the interpretation rule stated in
advance. Writes artifacts/posthoc_coverage/ and prints PH3_ANALYSIS_GATE.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.stats import holm, paired_contrast  # noqa: E402

SRC = "artifacts/posthoc_baselines"
OUT = "artifacts/posthoc_coverage"
N_MAIN = 8
POLICIES = ["eig:mg", "mfi:1pr", "eig:1prmg", "mfi:pi"]
GAUSS_RATIO = float(2 * norm.ppf(0.95) / np.sqrt(2 / np.pi))


def main():
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{SRC}/*/runs.csv.gz"))], ignore_index=True)
    R = R[(R.N == N_MAIN) & R.policy.isin(POLICIES)]
    os.makedirs(OUT, exist_ok=True)
    tm = R.groupby(["dataset", "policy", "target"], as_index=False)[["coverage", "mean_width", "iae"]].mean()
    tm.to_csv(f"{OUT}/target_means.csv.gz", index=False)
    cells = tm.groupby(["dataset", "policy"], as_index=False)[["coverage", "mean_width", "iae"]].mean()
    cells["ratio"] = cells.mean_width / cells.iae
    cells.to_csv(f"{OUT}/cells.csv", index=False)

    banks, tests = [], []
    for ds, g in tm.groupby("dataset"):
        w = g.pivot(index="target", columns="policy")
        x = w["mean_width"][["eig:mg", "mfi:1pr"]].dropna()
        # lower_is_better=False: d = new - base > 0 means the Rasch-type interval is wider
        t = paired_contrast(x["eig:mg"].values, x["mfi:1pr"].values, lower_is_better=False)
        t["dataset"] = ds
        tests.append(t)
        c = cells[cells.dataset == ds].set_index("policy")
        banks.append({
            "dataset": ds,
            "cov_eost": float(c.loc["eig:mg", "coverage"]), "cov_rasch": float(c.loc["mfi:1pr", "coverage"]),
            "width_eost": float(c.loc["eig:mg", "mean_width"]),
            "width_rasch": float(c.loc["mfi:1pr", "mean_width"]),
            "median_width_ratio_rasch_over_eost": float(np.median(x["mfi:1pr"].values / x["eig:mg"].values)),
            "iae_eost": float(c.loc["eig:mg", "iae"]), "iae_rasch": float(c.loc["mfi:1pr", "iae"]),
            "ratio_eost": float(c.loc["eig:mg", "ratio"]), "ratio_rasch": float(c.loc["mfi:1pr", "ratio"]),
            "ratio_1prmg": float(c.loc["eig:1prmg", "ratio"]), "ratio_pi": float(c.loc["mfi:pi", "ratio"]),
            "width_wins_rasch": t["wins"], "width_losses_rasch": t["losses"], "p_width": t["p"],
        })
    B = pd.DataFrame(banks)
    B["p_width_holm"] = holm(B.p_width.values)
    B["rasch_wider"] = B.median_width_ratio_rasch_over_eost > 1
    B["rasch_cov_higher"] = B.cov_rasch > B.cov_eost
    B["rasch_iae_not_lower"] = B.iae_rasch >= B.iae_eost
    B.to_csv(f"{OUT}/banks.csv", index=False)

    hi = B[B.rasch_cov_higher]
    rho, p_rho = spearmanr(cells.ratio, cells.coverage)
    wider_all = bool(hi.rasch_wider.all()) and len(hi) > 0
    iae_half = bool(hi.rasch_iae_not_lower.sum() >= len(hi) / 2) and len(hi) > 0
    if wider_all and iae_half:
        verdict = "wider_intervals"
    elif len(hi) > 0 and not hi.rasch_wider.any():
        verdict = "centres_open"
    else:
        verdict = "mixed"
    pol_mean = cells.groupby("policy")[["coverage", "mean_width", "iae", "ratio"]].mean()
    out = {
        "N": N_MAIN, "gaussian_ratio": GAUSS_RATIO,
        "n_banks_rasch_cov_higher": int(len(hi)),
        "n_banks_rasch_wider": int(B.rasch_wider.sum()),
        "n_banks_rasch_wider_sig_holm": int(((B.p_width_holm < 0.05) & B.rasch_wider).sum()),
        "n_banks_rasch_narrower_sig_holm": int(((B.p_width_holm < 0.05) & ~B.rasch_wider).sum()),
        "rasch_wider_on_all_banks_with_higher_coverage": wider_all,
        "n_higher_cov_banks_rasch_iae_not_lower": int(hi.rasch_iae_not_lower.sum()),
        "median_width_ratio_rasch_over_eost": float(B.median_width_ratio_rasch_over_eost.median()),
        "median_bank_ratio": {p: float(cells[cells.policy == p].ratio.median()) for p in POLICIES},
        "mean_over_banks": {p: {k: float(v) for k, v in pol_mean.loc[p].items()} for p in POLICIES},
        "spearman_ratio_coverage": {"rho": float(rho), "p": float(p_rho), "n_cells": int(len(cells))},
        "rule": ("wider_intervals if mfi:1pr is wider on every bank where its coverage is higher and its IAE "
                 "is not lower on at least half of those banks; centres_open if it is wider on none of them; "
                 "mixed otherwise"),
        "verdict": verdict,
    }
    json.dump(out, open(f"{OUT}/decisions.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(B[["dataset", "cov_eost", "cov_rasch", "median_width_ratio_rasch_over_eost", "iae_eost", "iae_rasch",
             "ratio_eost", "ratio_rasch", "ratio_1prmg", "ratio_pi", "p_width_holm"]].round(3).to_string(index=False))
    print(pol_mean.round(3).to_string())
    print(f"Spearman(ratio, coverage) = {rho:.3f} (p = {p_rho:.2g}, {len(cells)} cells); verdict = {verdict}")
    ok = len(B) == 8 and np.isfinite(rho)
    print(f"PH3_ANALYSIS_GATE={'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
