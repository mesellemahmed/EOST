"""Cluster-aware re-analysis of the first study (post hoc; reviewer request).

The first study used the target system as the unit of analysis. Targets of the same
family are not independent, so every confirmatory contrast is recomputed with the
family cluster as the unit:

* cluster definitions -- (a) heuristic families (connected components of the family
  rule used for leave-family-out panels) and (b) strict clusters (heuristic families
  joined with near-duplicates of IRT-residual correlation >= 0.5, configs/study2_clusters.json);
* per bank, target-level values (already averaged over replicates) are averaged within
  cluster, then the paired two-sided Wilcoxon test of the frozen plan is applied to the
  cluster means, with Holm over the same 144-test confirmatory family;
* the frozen decision rule is re-applied (>= 5 of 8 banks Holm-significant for EOST-M at
  N = 8 and none against);
* a cluster bootstrap (resampling clusters within every bank, 4000 replicates) gives a
  95% interval for the median over banks of the relative improvement at N = 8.
Writes artifacts/confirm/_clusters/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.stats import holm, paired_contrast  # noqa: E402
from eost.study2 import components  # noqa: E402

OUT = "artifacts/confirm/_clusters"
B_BOOT = 4000


def heuristic_labels(bank):
    J = bank.J
    A = np.zeros((J, J), dtype=bool)
    for j in range(J):
        A[j] = bank.related(j)
    return components(A | A.T)


def main():
    spec = json.load(open("configs/hypotheses.json"))
    tm = pd.read_csv("artifacts/confirm/_analysis/target_means.csv.gz")
    conf = spec["confirmatory_datasets"]
    tm = tm[tm.dataset.isin(conf)]
    strict = json.load(open("configs/study2_clusters.json"))["banks"]
    labels = {}
    for ds in conf:
        B = load_bank(f"data/processed/{ds}.npz")
        labels[ds] = {"heuristic": heuristic_labels(B), "strict": np.array(strict[ds]["cluster"])}
    os.makedirs(OUT, exist_ok=True)
    out = {}
    rows = []
    for kind in ("target", "heuristic", "strict"):
        res = []
        for h in spec["hypotheses"]:
            for ds in conf:
                for N in sorted(tm.N.unique()):
                    g = tm[(tm.dataset == ds) & (tm.N == N)]
                    w = g.pivot(index="target", columns="policy", values=h["metric"] if h["metric"] != "covgap" else "coverage")
                    if h["metric"] == "covgap":
                        w = (w - 0.90).abs()
                    if h["new"] not in w or h["base"] not in w:
                        continue
                    w = w[[h["new"], h["base"]]].dropna()
                    if kind != "target":
                        lab = labels[ds][kind][w.index.values]
                        w = w.groupby(lab).mean()
                    if len(w) < 6:
                        continue
                    r = paired_contrast(w[h["base"]].values, w[h["new"]].values, lower_is_better=h["lower_is_better"])
                    r.update({"unit": kind, "hypothesis": h["id"], "dataset": ds, "N": int(N), "n_units": int(len(w))})
                    res.append(r)
        R = pd.DataFrame(res)
        R["p_holm"] = holm(R.p.values)
        R["favours_new"] = (R.p_holm < spec["alpha"]) & (R.mean_diff > 0)
        R["favours_base"] = (R.p_holm < spec["alpha"]) & (R.mean_diff < 0)
        rows.append(R)
        dec = {}
        for h in spec["hypotheses"]:
            sub = R[(R.hypothesis == h["id"]) & (R.N == spec["primary_panel_size"])]
            dec[h["id"]] = {"favour_new": int(sub.favours_new.sum()), "favour_base": int(sub.favours_base.sum()),
                            "n_banks": int(len(sub)), "supported": bool(sub.favours_new.sum() >= 5 and sub.favours_base.sum() == 0),
                            "median_rel_improvement_pct": float(100 * sub.rel_improvement.median()),
                            "units_per_bank": {d: int(n) for d, n in zip(sub.dataset, sub.n_units)},
                            "tests": int(len(R))}
        out[kind] = dec
    ALL = pd.concat(rows, ignore_index=True)
    ALL.to_csv(f"{OUT}/contrasts.csv", index=False)
    # cluster bootstrap of the median over banks of the relative improvement (strict clusters, N = 8)
    rng = np.random.default_rng(20261104)
    boot = {}
    for h in spec["hypotheses"]:
        per_bank = []
        for ds in conf:
            g = tm[(tm.dataset == ds) & (tm.N == spec["primary_panel_size"])]
            col = h["metric"] if h["metric"] != "covgap" else "coverage"
            w = g.pivot(index="target", columns="policy", values=col)[[h["new"], h["base"]]].dropna()
            if h["metric"] == "covgap":
                w = (w - 0.90).abs()
            lab = labels[ds]["strict"][w.index.values]
            cm = w.groupby(lab).agg(["sum", "count"])
            per_bank.append((cm[(h["base"], "sum")].values, cm[(h["new"], "sum")].values))
        meds = np.empty(B_BOOT)
        for b in range(B_BOOT):
            rel = []
            for base_s, new_s in per_bank:
                k = rng.integers(0, len(base_s), len(base_s))
                rel.append((base_s[k].sum() - new_s[k].sum()) / base_s[k].sum())
            meds[b] = np.median(rel)
        boot[h["id"]] = {"median_rel_improvement_ci95_pct": [float(100 * np.quantile(meds, 0.025)),
                                                             float(100 * np.quantile(meds, 0.975))]}
    out["strict_cluster_bootstrap"] = boot
    json.dump(out, open(f"{OUT}/decisions.json", "w"), indent=1)
    for kind in ("target", "heuristic", "strict"):
        print(kind, {k: (v["favour_new"], v["favour_base"], round(v["median_rel_improvement_pct"], 1), v["supported"])
                     for k, v in out[kind].items()})
    print("bootstrap", boot)
    tgt_ok = all(out["target"][h]["supported"] == json.load(open("artifacts/confirm/_analysis/decisions.json"))["decisions"][h]["supported"]
                 for h in out["target"])
    print(f"CLUSTER_ANALYSIS_GATE={'PASS' if tgt_ok else 'FAIL'} (target-level re-run reproduces the frozen decisions: {tgt_ok})")
    if not tgt_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
