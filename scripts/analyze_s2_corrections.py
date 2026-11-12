"""Analysis of the post hoc corrections of study 2 (plans in docs/RESEARCH_LOG.md, written
before any study-2 result was inspected).

tiny  -- reproduction of the frozen tinyBenchmarks errors; share of N = 8 cells in which
         tinyBenchmarks used fewer anchors than its budget; hypothesis C8 restricted to
         full-budget cells; items-matched comparison (EOST-M after the largest checkpoint not
         exceeding the number of anchors actually used).
fbfix -- location drift of the frozen chains (sweeps 10-41) versus the corrected chains;
         equivalence test of C3 for the corrected fully Bayesian CAT and for the joint model;
         paired contrasts of every fully Bayesian variant against EOST-M (held-out log loss,
         score error, coverage); N = 8 (the C3 panel size) and, secondary, N = 4.
signs -- review finding (3): the direction of a significant contrast is taken from the sign of
         the mean difference although the test is rank-based; count the Holm-significant
         contrasts (both studies) whose mean difference and rank-biserial correlation disagree.
Writes artifacts/study2_posthoc/_analysis/.
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

D = "artifacts/study2_posthoc"
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
CPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
SPEC = json.load(open("configs/study2_hypotheses.json"))


def load_main():
    return pd.concat([pd.read_csv(f"artifacts/study2/main/{b}/runs.csv.gz") for b in CONF], ignore_index=True)


def boot_rel_ci(x, y, B=4000, level=0.90, seed=SPEC["seed"]):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    rel = (x[idx].mean(1) - y[idx].mean(1)) / y[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(rel, a)), float(np.quantile(rel, 1 - a)), float((x.mean() - y.mean()) / y.mean())


def tiny(R):
    T = pd.concat([pd.read_csv(f"{D}/tiny/{b}/rows.csv.gz") for b in CONF if os.path.exists(f"{D}/tiny/{b}/rows.csv.gz")],
                  ignore_index=True)
    out = {}
    g = R[(R.policy == "tiny:gpirt") & (R.N == 8)]
    rep = []
    for Tb in (25, 50, 100):
        m = T[T["T"] == Tb].merge(g[g["T"] == Tb][["dataset", "target", "rep", f"err@{Tb}"]], on=["dataset", "target", "rep"])
        rep.append(float((m.err_gpirt - m[f"err@{Tb}"]).abs().max()))
        out[f"share_short_T{Tb}"] = {b: float((T[(T["T"] == Tb) & (T.dataset == b)].n_items < Tb).mean()) for b in CONF}
        out[f"median_items_T{Tb}"] = {b: float(T[(T["T"] == Tb) & (T.dataset == b)].n_items.median()) for b in CONF}
    out["reproduction_max_abs_diff"] = max(rep)
    # C8 restricted to full-budget cells, and items-matched comparison at budget 50
    e = R[(R.policy == "eig:mg") & (R.N == 8)][["dataset", "target", "rep"] + [f"err@{c}" for c in CPS]]
    T50 = T[T["T"] == 50].merge(e, on=["dataset", "target", "rep"])
    T50["eost_err_matched"] = [r[f"err@{max(c for c in CPS if c <= r.n_items)}"] for _, r in T50.iterrows()]
    tests = []
    for kind in ("full_budget", "items_matched"):
        for b in CONF:
            s = T50[T50.dataset == b]
            if kind == "full_budget":
                s = s[s.n_items == 50]
                pt = s.groupby("target").agg(base=("err_gpirt", "mean"), new=("err@50", "mean"))
            else:
                pt = s.groupby("target").agg(base=("err_gpirt", "mean"), new=("eost_err_matched", "mean"))
            if len(pt) < 6:
                continue
            r = paired_contrast(pt.base.values, pt.new.values, lower_is_better=True)
            r.update({"kind": kind, "dataset": b})
            tests.append(r)
    X = pd.DataFrame(tests)
    # Multiplicity exactly as in the frozen decision: the corrected C8 p-values replace the frozen ones
    # inside the frozen 88-test Holm family (a first version of this script applied Holm within the
    # eight corrected tests only, which is less conservative than the frozen rule; see the research log).
    CF = pd.read_csv("artifacts/study2/_analysis/confirmatory_tests.csv")
    for kind in ("full_budget", "items_matched"):
        sel = X.kind == kind
        F = CF.copy()
        for _, r in X[sel].iterrows():
            j = F.index[(F.hypothesis == "C8") & (F.dataset == r.dataset)]
            F.loc[j, ["p", "mean_diff", "rel_improvement"]] = [r.p, r.mean_diff, r.rel_improvement]
        F["p_holm_new"] = holm(F.p.values)
        X.loc[sel, "p_holm"] = [float(F.loc[(F.hypothesis == "C8") & (F.dataset == d), "p_holm_new"].iloc[0])
                                for d in X.loc[sel, "dataset"]]
        X.loc[sel, "p_holm_within8"] = holm(X.loc[sel, "p"].values)
    X.to_csv(f"{D}/_analysis/tiny_tests.csv", index=False)
    fc = CF[CF.hypothesis == "C8"]
    out["C8_frozen"] = {"favour_eost": int(((fc.p_holm < 0.05) & (fc.mean_diff > 0)).sum()),
                        "favour_tiny": int(((fc.p_holm < 0.05) & (fc.mean_diff < 0)).sum()),
                        "median_rel_improvement_pct": float(100 * fc.rel_improvement.median()), "n_banks": int(len(fc))}
    for kind in ("full_budget", "items_matched"):
        s = X[X.kind == kind]
        out[f"C8_{kind}"] = {"favour_eost": int(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum()),
                             "favour_tiny": int(((s.p_holm < 0.05) & (s.mean_diff < 0)).sum()),
                             "favour_eost_within8_holm": int(((s.p_holm_within8 < 0.05) & (s.mean_diff > 0)).sum()),
                             "median_rel_improvement_pct": float(100 * s.rel_improvement.median()), "n_banks": int(len(s)),
                             "supported_by_frozen_rule": bool(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum() >= 5
                                                              and ((s.p_holm < 0.05) & (s.mean_diff < 0)).sum() == 0)}
    return out


def fbfix(R, N=8):
    sub = "fbfix" if N == 8 else f"fbfix_N{N}"
    banks = [b for b in CONF if os.path.exists(f"{D}/{sub}/{b}/rows.csv.gz")]
    F = pd.concat([pd.read_csv(f"{D}/{sub}/{b}/rows.csv.gz") for b in banks], ignore_index=True)
    out = {"N": N, "banks": banks, "location_slope_frozen_window": {}, "location_slope_corrected": {},
           "location_drift_frozen_window": {}, "location_drift_corrected": {}}
    for b in banks:
        s = F[(F.dataset == b) & (F.policy == "fb:bmi")]
        out["location_slope_frozen_window"][b] = float(s.loc_frozen_window_slope.abs().median())
        out["location_slope_corrected"][b] = float(s.loc_kept_slope.abs().median())
        # drift of the mean calibrator ability over the 32 retained sweeps (slope x 31)
        out["location_drift_frozen_window"][b] = float(31 * s.loc_frozen_window_slope.abs().median())
        out["location_drift_corrected"][b] = float(31 * s.loc_kept_slope.abs().median())
    eost = R[(R.policy == "eig:mg") & (R.N == N)].groupby(["dataset", "target"])[["ilogloss", "iae", "coverage"]].mean()
    frozen = R[(R.policy == "fb:bmi") & (R.N == N)].groupby(["dataset", "target"])[["ilogloss", "iae", "coverage"]].mean()
    frozen_e = R[(R.policy == "fb:eig") & (R.N == N)].groupby(["dataset", "target"])[["ilogloss", "iae", "coverage"]].mean()
    fm = F.groupby(["dataset", "policy", "target"])[["ilogloss", "iae", "coverage"]].mean()
    eq, tests = [], []
    for b in banks:
        for pol, src in [("fb:bmi (frozen)", frozen), ("fb:eig (frozen)", frozen_e), ("fb:bmi", None), ("fb:eig", None),
                         ("fbj:bmi", None), ("fbj:eig", None)]:
            x = src.loc[b] if src is not None else fm.loc[(b, pol)]
            y = eost.loc[b]
            j = x.index.intersection(y.index)
            lo, hi, rel = boot_rel_ci(x.loc[j, "ilogloss"].values, y.loc[j, "ilogloss"].values)
            eq.append({"dataset": b, "policy": pol, "rel_diff": rel, "ci_lo": lo, "ci_hi": hi,
                       "equivalent": bool(-0.01 < lo and hi < 0.01), "coverage": float(x.loc[j, "coverage"].mean()),
                       "eost_coverage": float(y.loc[j, "coverage"].mean())})
            for met in ("ilogloss", "iae"):
                r = paired_contrast(x.loc[j, met].values, y.loc[j, met].values, lower_is_better=True)  # >0: EOST-M better
                r.update({"dataset": b, "policy": pol, "metric": met})
                tests.append(r)
    E, X = pd.DataFrame(eq), pd.DataFrame(tests)
    X["p_holm"] = holm(X.p.values)
    sfx = "" if N == 8 else f"_N{N}"
    E.to_csv(f"{D}/_analysis/fb_equivalence{sfx}.csv", index=False)
    X.to_csv(f"{D}/_analysis/fb_tests{sfx}.csv", index=False)
    for pol in E.policy.unique():
        s = E[E.policy == pol]
        t = X[(X.policy == pol) & (X.metric == "ilogloss")]
        u = X[(X.policy == pol) & (X.metric == "iae")]
        out[pol] = {"n_equivalent": int(s.equivalent.sum()), "n_banks": int(len(s)),
                    "median_rel_diff_pct": float(100 * s.rel_diff.median()),
                    "eost_better_sig": int(((t.p_holm < 0.05) & (t.mean_diff > 0)).sum()),
                    "fb_better_sig": int(((t.p_holm < 0.05) & (t.mean_diff < 0)).sum()),
                    "iae_median_rel_improvement_eost_pct": float(100 * u.rel_improvement.median()),
                    "iae_eost_better_sig": int(((u.p_holm < 0.05) & (u.mean_diff > 0)).sum()),
                    "iae_fb_better_sig": int(((u.p_holm < 0.05) & (u.mean_diff < 0)).sum()),
                    "mean_coverage": float(s.coverage.mean()), "eost_mean_coverage": float(s.eost_coverage.mean()),
                    "C3_rule_supported": bool(s.equivalent.sum() >= 6)}
    return out


def signs():
    """Holm-significant contrasts whose mean difference and rank-biserial correlation disagree."""
    files = {"study1_confirmatory": "artifacts/confirm/_analysis/confirmatory_tests.csv",
             "study1_clusters": "artifacts/confirm/_clusters/contrasts.csv",
             "study1_posthoc_baselines": "artifacts/posthoc_baselines/_analysis/contrasts.csv",
             "study2_confirmatory": "artifacts/study2/_analysis/confirmatory_tests.csv",
             "study2_secondary": "artifacts/study2/_analysis/secondary_tests.csv"}
    out = {}
    for k, f in files.items():
        if not os.path.exists(f):
            continue
        X = pd.read_csv(f)
        sig = X[X.p_holm < 0.05]
        disc = sig[np.sign(sig.mean_diff) * np.sign(sig.rank_biserial) < 0]
        out[k] = {"n_significant": int(len(sig)), "n_discordant": int(len(disc)),
                  "discordant": disc.drop(columns=[c for c in disc.columns if c.startswith("ci_")]).to_dict("records")}
    out["total_significant"] = int(sum(v["n_significant"] for v in out.values()))
    out["total_discordant"] = int(sum(v["n_discordant"] for v in out.values() if isinstance(v, dict)))
    return out


def main():
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    R = load_main()
    res = {}
    if glob.glob(f"{D}/tiny/*/rows.csv.gz"):
        res["tiny"] = tiny(R)
    if glob.glob(f"{D}/fbfix/*/rows.csv.gz"):
        res["fbfix"] = fbfix(R, 8)
    if glob.glob(f"{D}/fbfix_N4/*/rows.csv.gz"):
        res["fbfix_N4"] = fbfix(R, 4)
    res["signs"] = signs()
    json.dump(res, open(f"{D}/_analysis/decisions.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
    ok = res.get("tiny", {}).get("reproduction_max_abs_diff", 0.0) == 0.0
    print(f"S2_CORRECTIONS_GATE={'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
