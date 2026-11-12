"""Analysis of the post hoc strong-baseline experiment PH1 (plan: docs/RESEARCH_LOG.md).

1. Reproduction check: eig:mg and mfi:pi rows must equal the confirmatory rows exactly.
2. Paired per-target contrasts of EOST-M (eig:mg) against every other PH1 policy on
   integrated holdout log loss, score IAE and |coverage - 0.9|, at N in {4, 8, 16};
   two-sided Wilcoxon, Holm over this post hoc family.
3. The decision rules fixed before running.
Writes artifacts/posthoc_baselines/_analysis/ and prints PH1_ANALYSIS_GATE.
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

D = "artifacts/posthoc_baselines"
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
METRICS = [("ilogloss", True), ("iae", True), ("covgap", True)]
BASES = ["mfi:pi", "efi:mg", "mfi:eap", "eig:eap", "mfi:1pr", "eig:1prmg"]


def main():
    fs = sorted(glob.glob(f"{D}/*/runs.csv.gz"))
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    # 1. reproduction of the confirmatory rows
    rep = {}
    for ds in sorted(R.dataset.unique()):
        old = pd.read_csv(f"artifacts/confirm/{ds}/runs.csv.gz")
        new = R[(R.dataset == ds) & R.policy.isin(["eig:mg", "mfi:pi"])]
        m = new.merge(old, on=["target", "rep", "N", "policy"], suffixes=("_n", "_o"))
        cols = ["ilogloss", "iae", "coverage", "final_err", "ibrier", "iinfo"]
        d = max(float((m[c + "_n"] - m[c + "_o"]).abs().max()) for c in cols)
        rep[ds] = {"rows_compared": int(len(m)), "rows_new": int(len(new)), "max_abs_diff": d}
    repro_ok = all(v["max_abs_diff"] == 0.0 and v["rows_compared"] == v["rows_new"] for v in rep.values())
    # 2. target means and contrasts
    tm = R.groupby(["dataset", "N", "policy", "target"], as_index=False)[["ilogloss", "iae", "coverage"]].mean()
    tm["covgap"] = (tm.coverage - 0.9).abs()
    tm.to_csv(f"{D}/_analysis/target_means.csv.gz", index=False)
    rows = []
    for ds in sorted(tm.dataset.unique()):
        for N in sorted(tm.N.unique()):
            g = tm[(tm.dataset == ds) & (tm.N == N)]
            for met, lower in METRICS:
                wide = g.pivot(index="target", columns="policy", values=met)
                for base in BASES:
                    w = wide[["eig:mg", base]].dropna()
                    if len(w) < 6:
                        continue
                    r = paired_contrast(w[base].values, w["eig:mg"].values, lower_is_better=lower)
                    r.update({"dataset": ds, "N": int(N), "base": base, "metric": met})
                    rows.append(r)
    C = pd.DataFrame(rows)
    C["p_holm"] = holm(C.p.values)
    C["eost_better"] = (C.p_holm < 0.05) & (C.mean_diff > 0)
    C["base_better"] = (C.p_holm < 0.05) & (C.mean_diff < 0)
    C.to_csv(f"{D}/_analysis/contrasts.csv", index=False)
    # 3. decision rules (fixed before running)
    c8 = C[(C.N == 8) & (C.metric == "ilogloss")].set_index(["base", "dataset"])
    dec = {}
    for base in ["mfi:eap", "mfi:1pr", "efi:mg"]:
        sub = c8.loc[base]
        dec[base] = {"eost_sig_better": int(sub.eost_better.sum()), "base_sig_better": int(sub.base_better.sum()),
                     "n_banks": int(len(sub)), "median_rel_improvement_pct": float(100 * sub.rel_improvement.median())}
    out = {
        "reproduction": rep, "reproduction_exact": repro_ok,
        "rule1_strong_plugin": {b: dec[b] for b in ["mfi:eap", "mfi:1pr"]},
        "rule1_triggered": any(dec[b]["eost_sig_better"] < 5 for b in ["mfi:eap", "mfi:1pr"]),
        "rule2_efi": dec["efi:mg"], "rule2_supports_corollary": dec["efi:mg"]["eost_sig_better"] >= 5,
    }
    json.dump(out, open(f"{D}/_analysis/decisions.json", "w"), indent=1)
    summ = C.groupby(["N", "metric", "base"]).agg(
        median_rel_impr_pct=("rel_improvement", lambda v: 100 * v.median()),
        eost_better=("eost_better", "sum"), base_better=("base_better", "sum"), banks=("dataset", "nunique"))
    print(summ.round(2).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "reproduction"}, indent=1))
    print(f"PH1_ANALYSIS_GATE={'PASS' if repro_ok else 'FAIL'} contrasts={len(C)}")
    if not repro_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
