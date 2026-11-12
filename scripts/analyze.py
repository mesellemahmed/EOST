"""Target-level analysis of replay results (confirmatory + secondary).

Rows are first averaged over replicates within (dataset, target, N, policy):
the target system is the unit of analysis (no pseudo-replication). The
confirmatory family and decision rule are read from configs/hypotheses.json,
which is frozen by configs/PREREGISTRATION.lock.json before confirmatory runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.stats import holm, paired_contrast  # noqa: E402

METRICS = ["ilogloss", "ibrier", "iae", "coverage", "final_err", "final_brier", "ttol_02", "ttol_03", "ttol_05",
           "iref_err", "iinfo", "mean_item_eig", "itrue_err", "true_theta_cov", "final_true_err", "mean_width",
           "err@5", "err@10", "err@25", "err@50", "err@100", "brier@10", "brier@25", "brier@50", "brier@100",
           "cov@10", "cov@25", "cov@50", "cov@100"]


def load_phase(phase: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"artifacts/{phase}/*/runs.csv.gz"))]
    if not frames:
        raise SystemExit(f"no runs for phase {phase}")
    return pd.concat(frames, ignore_index=True)


def target_means(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS if c in runs.columns]
    tm = runs.groupby(["dataset", "N", "policy", "target"], as_index=False)[cols].mean()
    tm["covgap"] = (tm["coverage"] - 0.90).abs()
    return tm


def one_contrast(tm, ds, N, new, base, metric, lower):
    g = tm[(tm.dataset == ds) & (tm.N == N)]
    wide = g.pivot(index="target", columns="policy", values=metric)
    if new not in wide or base not in wide:
        return None
    w = wide[[new, base]].dropna()
    if len(w) < 6:
        return None
    r = paired_contrast(w[base].values, w[new].values, lower_is_better=lower)
    r.update({"dataset": ds, "N": int(N), "new": new, "base": base, "metric": metric})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--spec", default="configs/hypotheses.json")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    runs = load_phase(a.phase)
    tm = target_means(runs)
    od = f"artifacts/{a.phase}/_analysis"
    os.makedirs(od, exist_ok=True)
    tm.to_csv(f"{od}/target_means.csv.gz", index=False)
    summ = tm.groupby(["dataset", "N", "policy"]).mean(numeric_only=True).drop(columns="target")
    summ.to_csv(f"{od}/policy_means.csv")

    datasets = sorted(tm.dataset.unique())
    Ns = sorted(tm.N.unique())
    # ---------------------------------------------------------- confirmatory family
    conf = []
    for h in spec["hypotheses"]:
        for ds in datasets:
            for N in Ns:
                r = one_contrast(tm, ds, N, h["new"], h["base"], h["metric"], h["lower_is_better"])
                if r:
                    r["hypothesis"] = h["id"]
                    conf.append(r)
    C = pd.DataFrame(conf)
    C["confirmatory_dataset"] = C.dataset.isin(spec["confirmatory_datasets"])
    C["p_holm"] = np.nan
    fam = C.confirmatory_dataset
    C.loc[fam, "p_holm"] = holm(C.loc[fam, "p"].values)
    if (~fam).any():  # synthetic / dev banks: own family, reported separately
        C.loc[~fam, "p_holm"] = holm(C.loc[~fam, "p"].values)
    C["favours_new"] = (C.p_holm < spec["alpha"]) & (C.mean_diff > 0)
    C["favours_base"] = (C.p_holm < spec["alpha"]) & (C.mean_diff < 0)
    C.to_csv(f"{od}/confirmatory_tests.csv", index=False)

    decisions = {}
    N0 = spec["primary_panel_size"]
    for h in spec["hypotheses"]:
        sub = C[(C.hypothesis == h["id"]) & (C.N == N0) & C.confirmatory_dataset]
        n_new, n_base = int(sub.favours_new.sum()), int(sub.favours_base.sum())
        decisions[h["id"]] = {
            "name": h["name"], "n_benchmarks": int(len(sub)), "holm_sig_favouring_new": n_new,
            "holm_sig_favouring_base": n_base,
            "supported": bool(len(sub) > 0 and n_new >= 5 and n_base == 0),
            "per_N": {int(N): {"favour_new": int(C[(C.hypothesis == h["id"]) & (C.N == N) & C.confirmatory_dataset].favours_new.sum()),
                               "favour_base": int(C[(C.hypothesis == h["id"]) & (C.N == N) & C.confirmatory_dataset].favours_base.sum()),
                               "n": int(len(C[(C.hypothesis == h["id"]) & (C.N == N) & C.confirmatory_dataset]))}
                      for N in Ns},
        }
    # ---------------------------------------------------------- secondary contrasts
    sec = []
    for new, base, metric in spec["secondary_contrasts"]:
        lower = metric not in ("iinfo", "coverage")
        for ds in datasets:
            for N in Ns:
                r = one_contrast(tm, ds, N, new, base, metric, lower)
                if r:
                    sec.append(r)
    S = pd.DataFrame(sec)
    if len(S):
        S["p_holm"] = holm(S["p"].values)
    S.to_csv(f"{od}/secondary_tests.csv", index=False)
    out = {"phase": a.phase, "n_rows": int(len(runs)), "datasets": datasets, "panel_sizes": [int(n) for n in Ns],
           "decisions": decisions}
    with open(f"{od}/decisions.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (v["supported"], v["holm_sig_favouring_new"], v["holm_sig_favouring_base"])
                      for k, v in decisions.items()}))
    print(f"ANALYSIS_GATE=PASS phase={a.phase} rows={len(runs)} confirmatory_tests={len(C)} secondary={len(S)}")


if __name__ == "__main__":
    main()
