"""Analysis of the exploratory regularization-path experiment PH2 (plan: docs/RESEARCH_LOG.md).

For each prior sd tau of log a, and for the marginal (EIG) and plug-in (MFI) models:
median over banks of the relative log-loss and score-IAE difference to EOST-M (main
analysis, EIG:mg with the empirical-Bayes sd), mean coverage, and Wilcoxon tests
against EOST-M (Holm over this family). Applies the "promising" criterion stated in
advance. Writes artifacts/posthoc_shrinkage/_analysis/.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.stats import holm, paired_contrast  # noqa: E402

D = "artifacts/posthoc_shrinkage"


def main():
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{D}/*/runs.csv.gz"))], ignore_index=True)
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    # reproduction of the confirmatory EIG:mg rows (N = 8)
    rep = {}
    for ds in sorted(R.dataset.unique()):
        old = pd.read_csv(f"artifacts/confirm/{ds}/runs.csv.gz")
        m = R[(R.dataset == ds) & (R.policy == "eig:mg")].merge(old, on=["target", "rep", "N", "policy"],
                                                                    suffixes=("_n", "_o"))
        rep[ds] = float(max((m[c + "_n"] - m[c + "_o"]).abs().max() for c in ["ilogloss", "iae", "coverage"]))
    tm = R.groupby(["dataset", "policy", "target"], as_index=False)[["ilogloss", "iae", "coverage"]].mean()
    tm["covgap"] = (tm.coverage - 0.9).abs()
    tm.to_csv(f"{D}/_analysis/target_means.csv.gz", index=False)
    rows, tests = [], []
    for pol in sorted(p for p in tm.policy.unique() if p != "eig:mg"):
        per_bank = []
        for ds, g in tm.groupby("dataset"):
            w = g.pivot(index="target", columns="policy")
            for met in ["ilogloss", "iae", "covgap"]:
                x = w[met][["eig:mg", pol]].dropna()
                r = paired_contrast(x["eig:mg"].values, x[pol].values, lower_is_better=True)  # positive: pol better
                r.update({"dataset": ds, "policy": pol, "metric": met})
                tests.append(r)
            per_bank.append({"dataset": ds, "coverage": float(g[g.policy == pol].coverage.mean())})
        pb = pd.DataFrame(per_bank)
        T = pd.DataFrame([t for t in tests if t["policy"] == pol])
        crit, _, tau = pol.partition("@")
        rows.append({"policy": pol, "criterion": crit, "tau": float(tau),
                     "ll_rel_vs_eost_%": float(-100 * T[T.metric == "ilogloss"].rel_improvement.median()),
                     "iae_rel_vs_eost_%": float(-100 * T[T.metric == "iae"].rel_improvement.median()),
                     "coverage": float(pb.coverage.mean())})
    S = pd.DataFrame(rows).sort_values(["criterion", "tau"])
    eost_cov = float(tm[tm.policy == "eig:mg"].groupby("dataset").coverage.mean().mean())
    T = pd.DataFrame(tests)
    T["p_holm"] = holm(T.p.values)
    T.to_csv(f"{D}/_analysis/contrasts.csv", index=False)
    S.to_csv(f"{D}/_analysis/path.csv", index=False)
    promising = S[(S.coverage >= 0.80) & (S["ll_rel_vs_eost_%"] <= 0.5)]
    out = {"reproduction_max_abs_diff": rep, "reproduction_exact": all(v == 0.0 for v in rep.values()),
           "eost_coverage": eost_cov, "path": S.to_dict("records"),
           "promising_criterion": "coverage >= 0.80 and median log loss at most 0.5% worse than EOST-M",
           "promising": promising.policy.tolist()}
    json.dump(out, open(f"{D}/_analysis/decisions.json", "w"), indent=1)
    print(S.round(3).to_string(index=False))
    print("EOST-M coverage", round(eost_cov, 3), "| promising:", out["promising"])
    print(f"PH2_ANALYSIS_GATE={'PASS' if out['reproduction_exact'] else 'FAIL'}")
    if not out["reproduction_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
