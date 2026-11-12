"""Analysis of the robustness follow-ups RF1 (clustering) and RF2 (MCMC quality) of study 2.
Plans and decision rules: docs/RESEARCH_LOG.md, iteration 4 (written before running).
Outputs: artifacts/robust/clusters/_analysis/ and artifacts/robust/fb_convergence/_analysis/.
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

CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
VARIANTS = ["metadata", "lbo", "joint_0.4", "joint_0.6", "joint_0.7"]
HYP = {"R2": ("eig:mg", "mfi:pi", "ilogloss"), "C1": ("eig:mg", "mfi:1pr", "ilogloss"),
       "C4": ("eig:mg", "random:mg", "iae"), "R3": ("eig:mg", "naive", "iae")}
RULE_H = ["R2", "C1"]


# ----------------------------------------------------------------------------- RF1
def rf1():
    D = "artifacts/robust/clusters"
    fs = sorted(glob.glob(f"{D}/*/*/rows.csv.gz"))
    if not fs:
        return None
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    V = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = {"variants": {}}
    # reproduction check: joint_0.5 equals the frozen rows
    fro = pd.concat([pd.read_csv(f"artifacts/study2/main/{b}/runs.csv.gz") for b in CONF], ignore_index=True)
    fro = fro[fro.N == 8]
    j5 = V[(V.variant == "joint_0.5") & (~V.skipped.astype(bool))]
    if len(j5):
        m = j5.merge(fro, on=["dataset", "target", "rep", "policy"], suffixes=("_v", "_f"))
        out["repro_rows"] = int(len(m))
        out["repro_max_abs_diff"] = float(max((m[c + "_v"] - m[c + "_f"]).abs().max() for c in ("ilogloss", "iae", "coverage")))
    CF = pd.read_csv("artifacts/study2/_analysis/confirmatory_tests.csv")
    frozen_dec = json.load(open("artifacts/study2/_analysis/decisions.json"))["decisions"]
    tests = []
    for v in VARIANTS:
        S = V[(V.variant == v)]
        if S.empty:
            continue
        skipped = S[S.skipped.astype(bool)].groupby("dataset").target.nunique().to_dict()
        S = S[~S.skipped.astype(bool)]
        tm = S.groupby(["dataset", "target", "policy"])[["ilogloss", "iae", "coverage"]].mean().reset_index()
        neli = S.groupby("dataset").n_eligible.median().to_dict()
        rows = []
        for h, (new, base, met) in HYP.items():
            for b in CONF:
                w = tm[tm.dataset == b].pivot_table(index="target", columns="policy", values=met)
                if new not in w or base not in w:
                    continue
                w = w[[new, base]].dropna()
                if len(w) < 6:
                    continue
                r = paired_contrast(w[base].values, w[new].values, lower_is_better=True)
                r.update({"variant": v, "hypothesis": h, "dataset": b})
                rows.append(r)
        X = pd.DataFrame(rows)
        # Holm exactly as in the frozen decision: the variant's p-values replace the frozen ones
        F = CF.copy()
        for _, r in X.iterrows():
            j = F.index[(F.hypothesis == r.hypothesis) & (F.dataset == r.dataset)]
            F.loc[j, ["p", "mean_diff", "rel_improvement"]] = [r.p, r.mean_diff, r.rel_improvement]
        F["ph"] = holm(F.p.values)
        X["p_holm"] = [float(F.loc[(F.hypothesis == r.hypothesis) & (F.dataset == r.dataset), "ph"].iloc[0])
                       for _, r in X.iterrows()]
        tests.append(X)
        dec = {}
        for h in HYP:
            s = X[X.hypothesis == h]
            fav = int(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum())
            against = int(((s.p_holm < 0.05) & (s.mean_diff < 0)).sum())
            dec[h] = {"median_rel_improvement_pct": float(100 * s.rel_improvement.median()),
                      "positive": int((s.rel_improvement > 0).sum()), "n_banks": int(len(s)),
                      "favour_new": fav, "favour_base": against,
                      "frozen_favour_new": int(frozen_dec[h]["favour_new"]),
                      "frozen_median_pct": float(frozen_dec[h]["median_rel_improvement_pct"])}
            if h in RULE_H:
                dec[h]["robust_by_rule"] = bool(dec[h]["median_rel_improvement_pct"] > 0 and dec[h]["positive"] >= 7
                                                and fav >= frozen_dec[h]["favour_new"] - 1 and against == 0)
        out["variants"][v] = {"decisions": dec, "targets_dropped": skipped, "median_eligible": neli,
                              "n_targets": int(S.groupby("dataset").target.nunique().sum())}
    if tests:
        T = pd.concat(tests, ignore_index=True)
        T.to_csv(f"{D}/_analysis/tests.csv", index=False)
    out["complete_variants"] = sorted(out["variants"])
    out["rule_met_all_variants"] = bool(out["variants"]) and bool(all(
        out["variants"][v]["decisions"][h]["robust_by_rule"] for v in out["variants"] for h in RULE_H))
    cl = json.load(open(f"{D}/_clusters.json"))["summary"]
    out["cluster_summary"] = cl
    json.dump(out, open(f"{D}/_analysis/decisions.json", "w"), indent=1)
    return out


# ----------------------------------------------------------------------------- RF2
PARAMS = ["loc", "mu_b", "sd_b", "mu_a", "sd_a"]


def _scale_diag(D, panels):
    """Explanatory only (not part of the rule): R-hat and bulk ESS of the ability scale (column 1
    of the trace, standard deviation of the calibrators' abilities) and its correlation with the
    item hyperparameters over the pooled post-burn-in sweeps."""
    sys.path.insert(0, os.path.dirname(__file__))
    from fb_convergence import ess, rhat_rank
    out = []
    for _, r in panels.iterrows():
        f = f"{D}/{r.dataset}/parts/traces_{int(r.target):04d}.npz"
        if not os.path.exists(f):
            continue
        T = np.load(f)["T"].astype(float)
        x = T[:, :, 1]
        flat = T.reshape(-1, T.shape[2])
        cor = lambda j: float(np.corrcoef(flat[:, 1], flat[:, j])[0, 1])
        out.append({"dataset": r.dataset, "target": int(r.target), "rhat_scale": rhat_rank(x),
                    "ess_scale": ess(x, True), "mcse_scale": float(x.std(ddof=1) / np.sqrt(max(ess(x, False), 1.0))),
                    "sd_scale": float(x.std(ddof=1)), "scale_min_chain_mean": float(x.mean(1).min()),
                    "scale_max_chain_mean": float(x.mean(1).max()), "cor_scale_sd_b": cor(3),
                    "cor_scale_mu_a": cor(4), "cor_scale_mu_b": cor(2), "cor_scale_sd_a": cor(5)})
    return pd.DataFrame(out)


def _floor_chains(D):
    """Chains that spent more than half of their post-burn-in sweeps at the lower bound of
    sigma_a (0.1)."""
    n = tot = 0
    for f in glob.glob(f"{D}/*/parts/traces_*.npz"):
        T = np.load(f)["T"]
        tot += T.shape[0]
        n += int(sum((T[c, :, 5] <= 0.1001).mean() > 0.5 for c in range(T.shape[0])))
    return n, tot


def _sign_test(k, n):
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue) if n else float("nan")


def rf2():
    D = "artifacts/robust/fb_convergence"
    fs = sorted(glob.glob(f"{D}/*/rows.csv.gz"))
    if not fs:
        return None
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    L = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    panels = L.drop_duplicates(["dataset", "target"])
    SC = _scale_diag(D, panels)
    SC.to_csv(f"{D}/_analysis/scale.csv", index=False)
    panels = panels.merge(SC, on=["dataset", "target"], how="left")
    conv = []
    for b, g in panels.groupby("dataset", sort=False):
        row = {"dataset": b, "panels": int(len(g))}
        for p in PARAMS + ["scale"]:
            row[f"rhat_{p}_max"] = float(g[f"rhat_{p}"].max())
            row[f"rhat_{p}_median"] = float(g[f"rhat_{p}"].median())
            row[f"ess_{p}_min"] = float(g[f"ess_{p}"].min())
            row[f"ess_{p}_median"] = float(g[f"ess_{p}"].median())
            row[f"n_rhat_{p}_lt_1.01"] = int((g[f"rhat_{p}"] < 1.01).sum())
            row[f"n_rhat_{p}_lt_1.05"] = int((g[f"rhat_{p}"] < 1.05).sum())
            row[f"n_ess_{p}_ge_100"] = int((g[f"ess_{p}"] >= 100).sum())
            row[f"mcse_{p}_median"] = float(g[f"mcse_{p}"].median())
            row[f"mcse_rel_{p}_median"] = float((g[f"mcse_{p}"] / g[f"sd_{p}"]).median())
        row["ll_spread_rel_median"] = float(g.ll_chain_spread_rel.median())
        row["ll_spread_rel_max"] = float(g.ll_chain_spread_rel.max())
        conv.append(row)
    C = pd.DataFrame(conv)
    C.to_csv(f"{D}/_analysis/convergence.csv", index=False)
    summ = {"panels": int(len(panels)), "banks": int(panels.dataset.nunique()), "chains": 4, "retained_draws": 400}
    for p in PARAMS + ["scale"]:
        summ[f"share_rhat_{p}_lt_1.01"] = float((panels[f"rhat_{p}"] < 1.01).mean())
        summ[f"share_rhat_{p}_lt_1.05"] = float((panels[f"rhat_{p}"] < 1.05).mean())
        summ[f"share_ess_{p}_ge_100"] = float((panels[f"ess_{p}"] >= 100).mean())
        summ[f"rhat_{p}_max"] = float(panels[f"rhat_{p}"].max())
        summ[f"rhat_{p}_median"] = float(panels[f"rhat_{p}"].median())
        summ[f"ess_{p}_median"] = float(panels[f"ess_{p}"].median())
        summ[f"ess_{p}_min"] = float(panels[f"ess_{p}"].min())
        summ[f"mcse_{p}_median"] = float(panels[f"mcse_{p}"].median())
        summ[f"mcse_rel_{p}_median"] = float((panels[f"mcse_{p}"] / panels[f"sd_{p}"]).median())
    for k in ["cor_scale_sd_b", "cor_scale_mu_a", "cor_scale_mu_b", "cor_scale_sd_a"]:
        summ[f"{k}_median"] = float(panels[k].median())
    summ["scale_chain_ratio_median"] = float((panels.scale_max_chain_mean / panels.scale_min_chain_mean).median())
    summ["scale_chain_ratio_max"] = float((panels.scale_max_chain_mean / panels.scale_min_chain_mean).max())
    summ["ll_spread_rel_median"] = float(panels.ll_chain_spread_rel.median())
    summ["ll_spread_rel_max"] = float(panels.ll_chain_spread_rel.max())
    summ["share_ll_spread_rel_gt_5pct"] = float((panels.ll_chain_spread_rel > 0.05).mean())
    summ["chains_at_sd_a_floor"], summ["chains_total"] = _floor_chains(D)
    # same cells: EOST-M, frozen PDM, burn-in-60 PDM and joint, long-chain arms
    keys = panels[["dataset", "target"]]
    banks = list(panels.dataset.unique())
    fro = pd.concat([pd.read_csv(f"artifacts/study2/main/{b}/runs.csv.gz") for b in banks], ignore_index=True)
    fro = fro[(fro.N == 8) & (fro.rep == 0)].merge(keys, on=["dataset", "target"])
    fix = pd.concat([pd.read_csv(f"artifacts/study2_posthoc/fbfix/{b}/rows.csv.gz") for b in banks], ignore_index=True)
    fix = fix[(fix.rep == 0)].merge(keys, on=["dataset", "target"])
    if "N" in fix.columns:
        fix = fix[fix.N == 8]
    arms = {"eost": fro[fro.policy == "eig:mg"], "pdm_bmi_frozen": fro[fro.policy == "fb:bmi"],
            "pdm_eig_frozen": fro[fro.policy == "fb:eig"], "pdm_bmi_60": fix[fix.policy == "fb:bmi"],
            "pdm_eig_60": fix[fix.policy == "fb:eig"], "jfb_eig_60": fix[fix.policy == "fbj:eig"],
            "jfb_bmi_60": fix[fix.policy == "fbj:bmi"], "pdm_bmi_long": L[L.policy == "pdm:bmi"],
            "pdm_eig_long": L[L.policy == "pdm:eig"], "jfb_eig_long": L[L.policy == "jfb:eig"],
            "jfb_bmi_long": L[L.policy == "jfb:bmi"]}
    W = None
    for k, a in arms.items():
        assert not a.duplicated(["dataset", "target"]).any(), k
        a = a[["dataset", "target", "ilogloss", "iae", "coverage"]].rename(
            columns={"ilogloss": f"ll_{k}", "iae": f"iae_{k}", "coverage": f"cov_{k}"})
        W = a if W is None else W.merge(a, on=["dataset", "target"])
    W.to_csv(f"{D}/_analysis/cells.csv", index=False)
    comp = {}
    for k in arms:
        if k == "eost":
            continue
        rel = (W[f"ll_{k}"] - W["ll_eost"]) / W["ll_eost"]
        per_bank = rel.groupby(W.dataset).median()
        n_pos, n_neg = int((per_bank > 0).sum()), int((per_bank < 0).sum())
        comp[k] = {"median_rel_ll_vs_eost_pct": float(100 * rel.median()),
                   "mean_rel_ll_vs_eost_pct": float(100 * rel.mean()),
                   "bank_median_rel_ll_vs_eost_pct": float(100 * per_bank.median()),
                   "banks_within_1pct": int((per_bank.abs() < 0.01).sum()),
                   "banks_worse_than_eost": n_pos, "banks_better_than_eost": n_neg,
                   "bank_sign_test_p": _sign_test(n_pos, n_pos + n_neg),
                   "cells_worse_than_eost": int((rel > 0).sum()), "cells_better_than_eost": int((rel < 0).sum()),
                   "mean_coverage": float(W[f"cov_{k}"].mean()),
                   "median_rel_iae_vs_eost_pct": float(100 * ((W[f"iae_{k}"] - W["iae_eost"]) / W["iae_eost"]).median())}
    comp["eost"] = {"mean_coverage": float(W["cov_eost"].mean())}
    rule_arms = {}
    for lng, short in [("pdm_bmi_long", "pdm_bmi_60"), ("pdm_eig_long", "pdm_eig_60"), ("jfb_eig_long", "jfb_eig_60"),
                       ("jfb_bmi_long", "jfb_bmi_60")]:
        d = (W[f"ll_{lng}"] - W[f"ll_{short}"]) / W[f"ll_{short}"]
        comp[lng]["median_rel_ll_vs_burn60_pct"] = float(100 * d.median())
        comp[lng]["median_abs_rel_ll_vs_burn60_pct"] = float(100 * d.abs().median())
        comp[lng]["share_cells_abs_vs_burn60_gt_1pct"] = float((d.abs() > 0.01).mean())
        c = comp[lng]
        rule_arms[lng] = {"signed_vs_burn60_lt_0.5": abs(c["median_rel_ll_vs_burn60_pct"]) < 0.5,
                          "abs_vs_burn60_lt_0.5": c["median_abs_rel_ll_vs_burn60_pct"] < 0.5,
                          "vs_eost_within_1": abs(c["median_rel_ll_vs_eost_pct"]) < 1.0}
    rule = all(all(v.values()) for v in rule_arms.values())
    rule_narrow = all(rule_arms[k]["signed_vs_burn60_lt_0.5"] and rule_arms[k]["vs_eost_within_1"]
                      for k in ("pdm_bmi_long", "pdm_eig_long"))
    out = {"summary": summ, "comparison": comp, "rule_by_arm": rule_arms, "rule_confirmed": bool(rule),
           "rule_confirmed_narrow_reading": bool(rule_narrow), "n_cells": int(len(W)),
           "complete": bool(len(banks) == len(CONF))}
    json.dump(out, open(f"{D}/_analysis/summary.json", "w"), indent=1)
    return out


def _eb_scale(panels):
    """Explanatory: EOST-M's point estimates of the ability scale and hyperparameters on the
    RF2/RF2b panels (the empirical-Bayes calibration, recomputed deterministically)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from eost.data import load_bank
    from eost.experiment import _seed
    from eost.irt import Priors, calibrate
    from eost.study2 import S2Config, eligible_s2
    cfg, out, banks = S2Config(), [], {}
    clus = json.load(open("configs/study2_clusters.json"))["banks"]
    for _, r in panels.iterrows():
        b, t = r.dataset, int(r.target)
        if b not in banks:
            banks[b] = load_bank(f"data/processed/{b}.npz")
        B = banks[b]
        elig = eligible_s2(B, np.array(clus[b]["cluster"]), t, None)
        rng_p = np.random.default_rng(_seed(cfg.master_seed, B.name, "main", "panel", t, 0, 8))
        panel = np.sort(rng_p.choice(elig, size=8, replace=False))
        cal, _, _ = calibrate(B.Y[panel].astype(float), "2pl", Priors(), True)
        out.append({"dataset": b, "target": t, "eb_scale": float(cal.theta_cal.std()), "eb_mu_b": cal.priors.mu_b,
                    "eb_sd_b": cal.priors.sd_b, "eb_mu_a": cal.priors.mu_log_a, "eb_sd_a": cal.priors.sd_log_a})
    return pd.DataFrame(out)


PARAMS_B = ["loc", "scale", "mu_b", "sd_b", "mu_a", "sd_a"]


def rf2x(sub="fb_scale", tag="scale"):
    """RF2b (sub fb_scale, tag scale) and RF2d (sub fb_exact2, tag exact)."""
    D = f"artifacts/robust/{sub}"
    fs = sorted(glob.glob(f"{D}/*/rows.csv.gz"))
    if not fs:
        return None
    os.makedirs(f"{D}/_analysis", exist_ok=True)
    L = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    panels = L.drop_duplicates(["dataset", "target"]).copy()
    if "scale_acceptance" not in panels:
        panels["scale_acceptance"] = panels["acc_scale"]
    EB = _eb_scale(panels)
    panels = panels.merge(EB, on=["dataset", "target"], how="left")
    banks = list(panels.dataset.unique())
    conv = []
    for b, g in panels.groupby("dataset", sort=False):
        row = {"dataset": b, "panels": int(len(g)), "acc_median": float(g.scale_acceptance.median())}
        for p in PARAMS_B:
            row[f"rhat_{p}_max"] = float(g[f"rhat_{p}"].max())
            row[f"ess_{p}_min"] = float(g[f"ess_{p}"].min())
            row[f"n_rhat_{p}_lt_1.01"] = int((g[f"rhat_{p}"] < 1.01).sum())
            row[f"n_rhat_{p}_lt_1.05"] = int((g[f"rhat_{p}"] < 1.05).sum())
            row[f"n_ess_{p}_ge_100"] = int((g[f"ess_{p}"] >= 100).sum())
        row["ll_spread_rel_median"] = float(g.ll_chain_spread_rel.median())
        row["ll_spread_rel_max"] = float(g.ll_chain_spread_rel.max())
        conv.append(row)
    C = pd.DataFrame(conv)
    C.to_csv(f"{D}/_analysis/convergence.csv", index=False)
    summ = {"panels": int(len(panels)), "banks": len(banks), "chains": 4, "retained_draws": 400,
            "acc_median": float(panels.scale_acceptance.median()), "acc_min": float(panels.scale_acceptance.min()),
            "acc_max": float(panels.scale_acceptance.max())}
    for k in ("acc_mu_a", "acc_sd_a", "acc_mu_b", "acc_sd_b", "acc_trans"):
        if k in panels:
            summ[f"{k}_median"] = float(panels[k].median())
    for p in PARAMS_B:
        summ[f"share_rhat_{p}_lt_1.01"] = float((panels[f"rhat_{p}"] < 1.01).mean())
        summ[f"share_rhat_{p}_lt_1.05"] = float((panels[f"rhat_{p}"] < 1.05).mean())
        summ[f"share_ess_{p}_ge_100"] = float((panels[f"ess_{p}"] >= 100).mean())
        summ[f"rhat_{p}_max"] = float(panels[f"rhat_{p}"].max())
        summ[f"rhat_{p}_median"] = float(panels[f"rhat_{p}"].median())
        summ[f"ess_{p}_median"] = float(panels[f"ess_{p}"].median())
        summ[f"ess_{p}_min"] = float(panels[f"ess_{p}"].min())
        summ[f"mcse_rel_{p}_median"] = float((panels[f"mcse_{p}"] / panels[f"sd_{p}"]).median())
    summ["ll_spread_rel_median"] = float(panels.ll_chain_spread_rel.median())
    summ["ll_spread_rel_max"] = float(panels.ll_chain_spread_rel.max())
    summ["share_ll_spread_rel_gt_5pct"] = float((panels.ll_chain_spread_rel > 0.05).mean())
    summ["chains_at_sd_a_floor"], summ["chains_total"] = _floor_chains(D)
    # panels whose sigma_b / sigma_a spend more than a quarter of the post-burn-in sweeps within 5% of the
    # upper bound of the prior box (4 and 1.5)
    nb = na = 0
    for f in glob.glob(f"{D}/*/parts/traces_*.npz"):
        T = np.load(f)["T"]
        nb += int((T[:, :, 3] >= 0.95 * 4.0).mean() > 0.25)
        na += int((T[:, :, 5] >= 0.95 * 1.5).mean() > 0.25)
    summ["panels_sd_b_near_upper"], summ["panels_sd_a_near_upper"] = nb, na
    # explanatory: where EOST-M's empirical-Bayes fit sits on the scale ridge
    z = (panels.mean_scale - panels.eb_scale) / panels.sd_scale
    summ["eb_scale_median"] = float(panels.eb_scale.median())
    summ["post_scale_median"] = float(panels.mean_scale.median())
    summ["post_over_eb_scale_median"] = float((panels.mean_scale / panels.eb_scale).median())
    summ["post_over_eb_scale_min"] = float((panels.mean_scale / panels.eb_scale).min())
    summ["post_over_eb_scale_max"] = float((panels.mean_scale / panels.eb_scale).max())
    summ["share_post_scale_above_eb"] = float((panels.mean_scale > panels.eb_scale).mean())
    summ["z_scale_median"] = float(z.median())
    summ["post_sd_scale_median"] = float(panels.sd_scale.median())
    converged = all(summ[f"share_rhat_{p}_lt_1.05"] >= 0.9 and summ[f"share_ess_{p}_ge_100"] >= 0.9 for p in PARAMS_B)
    # same cells as RF2: EOST-M, burn-in-60 arms, RF2 long chains, RF2b
    keys = panels[["dataset", "target"]]
    fro = pd.concat([pd.read_csv(f"artifacts/study2/main/{b}/runs.csv.gz") for b in banks], ignore_index=True)
    fro = fro[(fro.N == 8) & (fro.rep == 0)].merge(keys, on=["dataset", "target"])
    fix = pd.concat([pd.read_csv(f"artifacts/study2_posthoc/fbfix/{b}/rows.csv.gz") for b in banks], ignore_index=True)
    fix = fix[(fix.rep == 0) & (fix.N == 8)].merge(keys, on=["dataset", "target"])
    R2 = pd.concat([pd.read_csv(f"artifacts/robust/fb_convergence/{b}/rows.csv.gz") for b in banks
                    if os.path.exists(f"artifacts/robust/fb_convergence/{b}/rows.csv.gz")], ignore_index=True)
    R2 = R2.merge(keys, on=["dataset", "target"])
    arms = {"eost": fro[fro.policy == "eig:mg"], "pdm_bmi_frozen": fro[fro.policy == "fb:bmi"],
            "pdm_bmi_60": fix[fix.policy == "fb:bmi"], "pdm_eig_60": fix[fix.policy == "fb:eig"],
            "jfb_eig_60": fix[fix.policy == "fbj:eig"], "jfb_bmi_60": fix[fix.policy == "fbj:bmi"],
            "pdm_bmi_long": R2[R2.policy == "pdm:bmi"], "pdm_eig_long": R2[R2.policy == "pdm:eig"],
            "jfb_eig_long": R2[R2.policy == "jfb:eig"], "jfb_bmi_long": R2[R2.policy == "jfb:bmi"],
            f"pdm_bmi_{tag}": L[L.policy == "pdm:bmi"], f"pdm_eig_{tag}": L[L.policy == "pdm:eig"],
            f"jfb_eig_{tag}": L[L.policy == "jfb:eig"], f"jfb_bmi_{tag}": L[L.policy == "jfb:bmi"]}
    refs = ["long", "60"]
    if tag != "scale" and all(os.path.exists(f"artifacts/robust/fb_scale/{b}/rows.csv.gz") for b in banks):
        R3 = pd.concat([pd.read_csv(f"artifacts/robust/fb_scale/{b}/rows.csv.gz") for b in banks], ignore_index=True)
        R3 = R3.merge(keys, on=["dataset", "target"])
        for pol in ("pdm:bmi", "pdm:eig", "jfb:eig", "jfb:bmi"):
            arms[pol.replace(":", "_") + "_scale"] = R3[R3.policy == pol]
        refs.append("scale")
    W = None
    for k, a in arms.items():
        assert not a.duplicated(["dataset", "target"]).any(), k
        a = a[["dataset", "target", "ilogloss", "iae", "coverage"]].rename(
            columns={"ilogloss": f"ll_{k}", "iae": f"iae_{k}", "coverage": f"cov_{k}"})
        W = a if W is None else W.merge(a, on=["dataset", "target"])
    W.to_csv(f"{D}/_analysis/cells.csv", index=False)
    comp = {}
    for k in arms:
        if k == "eost":
            continue
        rel = (W[f"ll_{k}"] - W["ll_eost"]) / W["ll_eost"]
        per_bank = rel.groupby(W.dataset).median()
        n_pos, n_neg = int((per_bank > 0).sum()), int((per_bank < 0).sum())
        reli = (W[f"iae_{k}"] - W["iae_eost"]) / W["iae_eost"]
        comp[k] = {"median_rel_ll_vs_eost_pct": float(100 * rel.median()),
                   "mean_rel_ll_vs_eost_pct": float(100 * rel.mean()),
                   "banks_within_1pct": int((per_bank.abs() < 0.01).sum()),
                   "banks_worse_than_eost": n_pos, "banks_better_than_eost": n_neg,
                   "bank_sign_test_p": _sign_test(n_pos, n_pos + n_neg),
                   "cells_worse_than_eost": int((rel > 0).sum()), "cells_better_than_eost": int((rel < 0).sum()),
                   "mean_coverage": float(W[f"cov_{k}"].mean()),
                   "median_rel_iae_vs_eost_pct": float(100 * reli.median()),
                   "banks_iae_worse_than_eost": int((reli.groupby(W.dataset).median() > 0).sum())}
    comp["eost"] = {"mean_coverage": float(W["cov_eost"].mean())}
    for k in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi"):
        for ref in refs:
            d = (W[f"ll_{k}_{tag}"] - W[f"ll_{k}_{ref}"]) / W[f"ll_{k}_{ref}"]
            comp[f"{k}_{tag}"][f"median_rel_ll_vs_{ref}_pct"] = float(100 * d.median())
            comp[f"{k}_{tag}"][f"median_abs_rel_ll_vs_{ref}_pct"] = float(100 * d.abs().median())
    keep = {k: bool(abs(comp[k]["median_rel_ll_vs_eost_pct"]) < 1.0 and comp[k]["bank_sign_test_p"] >= 0.05)
            for k in (f"pdm_bmi_{tag}", f"pdm_eig_{tag}", f"jfb_eig_{tag}", f"jfb_bmi_{tag}")}
    out = {"summary": summ, "comparison": comp, "converged": bool(converged), "keep_not_different_by_arm": keep,
           "keep_not_different": bool(converged and all(keep.values())), "n_cells": int(len(W)),
           "complete": bool(len(banks) == len(CONF))}
    panels[["dataset", "target", "eb_scale", "mean_scale", "sd_scale", "eb_mu_a", "mean_mu_a", "eb_sd_b",
            "mean_sd_b"]].to_csv(f"{D}/_analysis/scale.csv", index=False)
    json.dump(out, open(f"{D}/_analysis/summary.json", "w"), indent=1)
    return out


def rf2c_partial():
    """RF2c was stopped after 14 panels (docs/RESEARCH_LOG.md); its convergence diagnostics are
    summarized from the saved parts, for the record."""
    fs = sorted(glob.glob("artifacts/robust/fb_exact/*/parts/target_*.json.gz"))
    if not fs:
        return None
    import gzip
    rows = [json.load(gzip.open(f, "rt"))["rows"][0] for f in fs]
    P = pd.DataFrame(rows)
    out = {"panels": int(len(P)), "per_bank": {}}
    for b, g in P.groupby("dataset", sort=False):
        out["per_bank"][b] = {"panels": int(len(g)),
                              "max_rhat_all": float(max(g[f"rhat_{p}"].max() for p in PARAMS_B)),
                              "max_rhat_loc": float(g.rhat_loc.max()), "min_rhat_loc": float(g.rhat_loc.min()),
                              "max_rhat_excl_loc_mu_b": float(max(g[f"rhat_{p}"].max() for p in PARAMS_B
                                                                  if p not in ("loc", "mu_b")))}
    os.makedirs("artifacts/robust/fb_exact/_analysis", exist_ok=True)
    json.dump(out, open("artifacts/robust/fb_exact/_analysis/partial.json", "w"), indent=1)
    return out


def main():
    a, b, c, d = rf1(), rf2(), rf2x("fb_scale", "scale"), rf2x("fb_exact2", "exact")
    e = rf2c_partial()
    if e is not None:
        print(json.dumps(e, indent=1))
    if a is not None:
        print(json.dumps({k: v for k, v in a.items() if k != "cluster_summary"}, indent=1)[:5000])
        ok = a.get("repro_max_abs_diff", 0.0) == 0.0
        print(f"RF1_ANALYSIS_GATE={'PASS' if ok else 'FAIL'}")
    if b is not None:
        print(json.dumps(b, indent=1)[:5000])
        print("RF2_ANALYSIS_GATE=PASS")
    if c is not None:
        print(json.dumps({k: v for k, v in c.items() if k != "comparison"}, indent=1)[:5000])
        print("RF2B_ANALYSIS_GATE=PASS")
    if d is not None:
        print(json.dumps({k: v for k, v in d.items() if k != "comparison"}, indent=1)[:5000])
        print("RF2D_ANALYSIS_GATE=PASS")


if __name__ == "__main__":
    main()
