"""Analysis of study 2 with the decision rules of configs/study2_hypotheses.json (frozen).

1. Target-level means over replicates (targets are one per strict family cluster, hence
   independent units); recalibrated intervals of EOST-M and of the Rasch-type test are
   analysed as the derived policies 'eig:mg+R' and 'mfi:1pr+R'.
2. Paired two-sided Wilcoxon signed-rank tests per bank at the primary panel size; Holm
   over the whole confirmatory family (every Holm-tested hypothesis x bank).
3. Decision rules: superiority hypotheses need >= 5 of 8 banks Holm-significant for
   EOST-M and none against; the equivalence hypothesis needs the 90% bootstrap interval of
   the relative difference inside the margin on >= 6 of 8 banks; the coverage criterion
   needs the mean coverage inside [0.85, 0.95] on >= 6 of 8 banks.
4. Secondary analyses (own Holm families): other panel sizes, temporal component,
   synthetic banks, empirical-Bayes variants, fully Bayesian criteria, mixed policy.
Writes artifacts/study2/_analysis/ and prints STUDY2_ANALYSIS_GATE.
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

SPEC = "configs/study2_hypotheses.json"
OUT = "artifacts/study2/_analysis"
METRIC_COLS = ["ilogloss", "ibrier", "iae", "final_err", "coverage", "mean_width", "iis", "iref_err", "iinfo",
               "err@25", "err@50", "err@100", "cov@25", "cov@50", "cov@100", "width@25", "width@50", "width@100",
               "n_star", "err_nstar", "p95_nstar", "lambda"]


def load(component):
    fs = sorted(glob.glob(f"artifacts/study2/{component}/*/runs.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def derive_recalibrated(R, rule):
    """Rows of the recalibrated policies built from the stored recalibrated endpoints."""
    out = []
    for pol in ("eig:mg", "mfi:1pr"):
        g = R[R.policy == pol].copy()
        if g.empty or f"coverage_{rule}" not in g:
            continue
        g = g[g[f"coverage_{rule}"].notna()]
        g["policy"] = pol + "+R"
        g["coverage"] = g[f"coverage_{rule}"]
        g["mean_width"] = g[f"mean_width_{rule}"]
        g["iis"] = g[f"iis_{rule}"]
        for c in (25, 50, 100):
            if f"cov@{c}_{rule}" in g:
                g[f"cov@{c}"] = g[f"cov@{c}_{rule}"]
        out.append(g)
    return pd.concat([R] + out, ignore_index=True)


def target_means(R):
    cols = [c for c in METRIC_COLS if c in R.columns]
    tm = R.groupby(["dataset", "component", "N", "policy", "target", "cluster"], as_index=False)[cols].mean()
    tm["covgap"] = (tm["coverage"] - 0.90).abs()
    return tm


def contrast(tm, ds, N, new, base, metric, lower=True):
    g = tm[(tm.dataset == ds) & (tm.N == N)]
    w = g.pivot_table(index="target", columns="policy", values=metric)
    if new not in w or base not in w:
        return None
    w = w[[new, base]].dropna()
    if len(w) < 6:
        return None
    r = paired_contrast(w[base].values, w[new].values, lower_is_better=lower)
    r.update({"dataset": ds, "N": int(N), "new": new, "base": base, "metric": metric})
    return r


def boot_rel_ci(tm, ds, N, new, base, metric, B=4000, level=0.90, seed=0):
    """Bootstrap CI (over targets) of the relative difference (new - base) / base of means."""
    g = tm[(tm.dataset == ds) & (tm.N == N)]
    w = g.pivot_table(index="target", columns="policy", values=metric)[[new, base]].dropna()
    rng = np.random.default_rng(seed)
    x, y = w[new].values, w[base].values
    idx = rng.integers(0, len(w), (B, len(w)))
    rel = (x[idx].mean(1) - y[idx].mean(1)) / y[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(rel, a)), float(np.quantile(rel, 1 - a)), float((x.mean() - y.mean()) / y.mean()), int(len(w))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="main", help="testing only: analyse another component")
    ap.add_argument("--banks", nargs="*", default=None, help="testing only: override the bank list")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    spec = json.load(open(SPEC))
    if a.banks:
        spec["banks"] = a.banks
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    rule = spec["recalibration_rule"]
    main_runs = load(a.component)
    if main_runs is None:
        raise SystemExit("no study-2 runs")
    R = derive_recalibrated(main_runs, rule)
    tm = target_means(R)
    tm.to_csv(f"{out_dir}/target_means.csv.gz", index=False)
    banks = spec["banks"]
    N0 = spec["primary_panel_size"]
    # ---- confirmatory family
    tests = []
    for h in spec["superiority"]:
        for ds in banks:
            r = contrast(tm, ds, N0, h["new"], h["base"], h["metric"])
            if r:
                r["hypothesis"] = h["id"]
                tests.append(r)
    C = pd.DataFrame(tests)
    C["p_holm"] = holm(C.p.values)
    C["favours_new"] = (C.p_holm < spec["alpha"]) & (C.mean_diff > 0)
    C["favours_base"] = (C.p_holm < spec["alpha"]) & (C.mean_diff < 0)
    C.to_csv(f"{out_dir}/confirmatory_tests.csv", index=False)
    dec = {}
    for h in spec["superiority"]:
        sub = C[C.hypothesis == h["id"]]
        dec[h["id"]] = {"name": h["name"], "n_banks": int(len(sub)), "favour_new": int(sub.favours_new.sum()),
                        "favour_base": int(sub.favours_base.sum()),
                        "positive": int((sub.rel_improvement > 0).sum()),
                        "median_rel_improvement_pct": float(100 * sub.rel_improvement.median()),
                        "supported": bool(sub.favours_new.sum() >= spec["min_banks"] and sub.favours_base.sum() == 0)}
    # ---- equivalence hypothesis
    for h in spec["equivalence"]:
        eq = []
        for ds in banks:
            lo, hi, rel, n = boot_rel_ci(tm, ds, N0, h["new"], h["base"], h["metric"],
                                         level=h["ci_level"], seed=spec["seed"])
            eq.append({"dataset": ds, "rel_diff": rel, "ci_lo": lo, "ci_hi": hi, "n": n,
                       "equivalent": bool(-h["margin"] < lo and hi < h["margin"])})
        E = pd.DataFrame(eq)
        E.to_csv(f"{out_dir}/equivalence_{h['id']}.csv", index=False)
        dec[h["id"]] = {"name": h["name"], "n_equivalent": int(E.equivalent.sum()), "n_banks": int(len(E)),
                        "median_rel_diff_pct": float(100 * E.rel_diff.median()),
                        "supported": bool(E.equivalent.sum() >= h["min_banks"])}
    # ---- coverage criterion
    for h in spec["coverage"]:
        cov = tm[(tm.N == N0) & (tm.policy == h["policy"])].groupby("dataset").coverage.mean()
        cov = cov.reindex(banks)
        inside = ((cov >= h["lo"]) & (cov <= h["hi"]))
        dec[h["id"]] = {"name": h["name"], "coverage": {d: float(v) for d, v in cov.items()},
                        "n_inside": int(inside.sum()), "supported": bool(inside.sum() >= h["min_banks"])}
    # ---- secondary: other panel sizes (own Holm family)
    sec = []
    for N in [n for n in spec["secondary_panel_sizes"]]:
        for h in spec["superiority"]:
            for ds in banks:
                r = contrast(tm, ds, N, h["new"], h["base"], h["metric"])
                if r:
                    r["hypothesis"] = h["id"]
                    sec.append(r)
    for new, base, metric in spec["secondary_contrasts"]:
        for N in [N0] + list(spec["secondary_panel_sizes"]):
            for ds in banks:
                r = contrast(tm, ds, N, new, base, metric)
                if r:
                    r["hypothesis"] = f"S:{new}|{base}|{metric}"
                    sec.append(r)
    S = pd.DataFrame(sec)
    if len(S):
        S["p_holm"] = holm(S.p.values)
        S.to_csv(f"{out_dir}/secondary_tests.csv", index=False)
    # ---- policy means at N0 (for the paper's tables)
    pm = tm[tm.N == N0].groupby(["dataset", "policy"]).mean(numeric_only=True).drop(columns=["target", "cluster", "N"])
    pm.to_csv(f"{out_dir}/policy_means.csv")
    # ---- temporal and synthetic components (secondary)
    extra = {}
    for comp in ("temporal", "synthetic"):
        Rc = load(comp)
        if Rc is None:
            continue
        tmc = target_means(derive_recalibrated(Rc, rule))
        tmc.to_csv(f"{out_dir}/target_means_{comp}.csv.gz", index=False)
        res = []
        for h in spec["superiority"]:
            for ds in sorted(tmc.dataset.unique()):
                for N in sorted(tmc.N.unique()):
                    r = contrast(tmc, ds, N, h["new"], h["base"], h["metric"])
                    if r:
                        r["hypothesis"] = h["id"]
                        res.append(r)
        X = pd.DataFrame(res)
        if len(X):
            X["p_holm"] = holm(X.p.values)
            X.to_csv(f"{out_dir}/tests_{comp}.csv", index=False)
        cov = tmc[tmc.policy.isin(["eig:mg", "eig:mg+R", "mfi:1pr", "mfi:1pr+R", "naive", "random:mg"])] \
            .groupby(["dataset", "N", "policy"]).coverage.mean()
        extra[comp] = {"coverage": {f"{d}|{n}|{p}": float(v) for (d, n, p), v in cov.items()},
                       "n_targets": {d: int(g.target.nunique()) for d, g in tmc.groupby("dataset")}}
    out = {"recalibration_rule": rule, "decisions": dec, "secondary_components": extra,
           "n_targets": {d: int(g.target.nunique()) for d, g in tm[tm.N == N0].groupby("dataset")},
           "n_tests_confirmatory": int(len(C))}
    json.dump(out, open(f"{out_dir}/decisions.json", "w"), indent=1)
    for k, v in dec.items():
        print(k, {kk: vv for kk, vv in v.items() if kk in ("favour_new", "favour_base", "median_rel_improvement_pct",
                                                           "n_equivalent", "median_rel_diff_pct", "n_inside", "supported")})
    print("STUDY2_ANALYSIS_GATE=PASS")


if __name__ == "__main__":
    main()
