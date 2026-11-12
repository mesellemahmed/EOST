"""Checks every qualitative statement of the manuscript against the artifacts.

The numbers quoted in the paper are LaTeX macros generated from the artifacts
(scripts/make_report.py), so they cannot drift. Qualitative wording ("never",
"monotonically", "the worst adaptive rule", "behaves like") can, so each such
statement is encoded here as a predicate on the analysis outputs. The script
writes artifacts/CLAIMS.json and prints CLAIMS_GATE=PASS only if every claim
holds. A claim whose data are missing is reported as SKIP (and fails the gate).

Runs in the container after `./run analyze <phase>` for confirm and the
sensitivity phases, and after the exploratory ranking analysis.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
SYN = ["synth_2pl", "synth_guess", "synth_2dim"]
ADAPTIVE = ["entropy:pi", "entropy:mg", "mfi:pi", "mfi:mg", "pwfi:mg", "eig:pi", "eig:mg", "evr:pi", "evr:mg"]
N0 = 8


def load(phase, name):
    f = f"artifacts/{phase}/_analysis/{name}"
    if not os.path.exists(f):
        return None
    return pd.read_csv(f) if f.endswith((".csv", ".csv.gz")) else json.load(open(f))


def runs(phase):
    fs = sorted(glob.glob(f"artifacts/{phase}/*/runs.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def diag(phase):
    fs = sorted(glob.glob(f"artifacts/{phase}/*/calib_diag.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


CLAIMS = []


def claim(cid, where, text):
    def deco(fn):
        CLAIMS.append((cid, where, text, fn))
        return fn
    return deco


C = load("confirm", "confirmatory_tests.csv")
S = load("confirm", "secondary_tests.csv")
TM = load("confirm", "target_means.csv.gz")
D = diag("confirm")


def H(hid, N=N0):
    return C[(C.hypothesis == hid) & (C.N == N) & C.dataset.isin(CONF)].set_index("dataset")


def bank_means(tm, N, metric, banks=CONF):
    """Mean over targets per (bank, policy); rows = banks."""
    s = tm[(tm.N == N) & tm.dataset.isin(banks)]
    return s.groupby(["dataset", "policy"])[metric].mean().unstack()


# ------------------------------------------------------------------ Section 6.1 (hypotheses)
@claim("H2-never-base", "Results 6.1", "H2 was never Holm-significant in favour of plug-in Fisher CAT at N=8")
def _():
    h = H("H2")
    return bool(h.favours_base.sum() == 0), {"favours_base": int(h.favours_base.sum())}


@claim("H2-smaller-at-64", "Results 6.1", "the H2 gain at N=64 is smaller than at N=8 and N=4")
def _():
    m4, m8, m64 = (100 * H("H2", N).rel_improvement.median() for N in (4, 8, 64))
    return bool(m64 < m8 < m4 or (m64 < m8 and m64 < m4)), {"N4": m4, "N8": m8, "N64": m64}


@claim("GSM8K-only-nonsig", "Results 6.1",
       "GSM8K is the only benchmark where H2-H4 were not Holm-significant at N=8, and its contrasts point the same way")
def _():
    out, ok = {}, True
    for hid in ["H2", "H3", "H4"]:
        h = H(hid)
        nonsig = sorted(h.index[~h.favours_new.astype(bool)])
        out[hid] = {"non_significant": nonsig, "gsm8k_rel_impr": float(h.loc["ollb_gsm8k", "rel_improvement"])}
        ok &= set(nonsig) <= {"ollb_gsm8k"} and h.loc["ollb_gsm8k", "rel_improvement"] > 0
    ok &= any(out[h]["non_significant"] == ["ollb_gsm8k"] for h in out)
    return bool(ok), out


@claim("GSM8K-fewest-unlocated", "Results 6.1", "GSM8K has the fewest items with identical panel verdicts at N=8")
def _():
    g8 = D[(D.N == 8) & D.dataset.isin(CONF)].groupby("dataset").frac_items_extreme.mean()
    return bool(g8.idxmin() == "ollb_gsm8k"), {k: round(float(v), 3) for k, v in g8.items()}


@claim("H1-sign-everywhere", "Results 6.1", "EIG:mg had lower held-out log loss than Entropy:mg on every benchmark at N=8")
def _():
    h = H("H1")
    return bool((h.rel_improvement > 0).all()), {k: round(100 * v, 2) for k, v in h.rel_improvement.items()}


@claim("H1-never-entropy", "Results 6.1", "H1 was never Holm-significant in favour of entropy at any panel size")
def _():
    c = C[(C.hypothesis == "H1") & C.dataset.isin(CONF)]
    return bool(c.favours_base.sum() == 0), {"favours_base": int(c.favours_base.sum())}


@claim("H1-more-sig-other-N", "Results 6.1",
       "H1 was significant in more benchmarks at N=4 and at N=64 than at N=8 (as a share of benchmarks tested)")
def _():
    sh = {int(N): float(H("H1", N).favours_new.mean()) for N in (4, 8, 64)}
    return bool(sh[4] > sh[8] and sh[64] > sh[8]), sh


@claim("H1-small", "Results 6.1", "the H1 effect is small at every panel size (median < 3%)")
def _():
    med = {int(N): 100 * C[(C.hypothesis == "H1") & (C.N == N) & C.dataset.isin(CONF)].rel_improvement.median()
           for N in sorted(C.N.unique())}
    return bool(max(med.values()) < 3), med


@claim("score-gap-larger", "Results 6.1/6.2",
       "on score estimation the EIG-vs-entropy gap (marginal model) is larger than on held-out log loss")
def _():
    ll, ia = bank_means(TM, N0, "ilogloss"), bank_means(TM, N0, "iae")
    g_ll = (100 * (1 - ll["eig:mg"] / ll["entropy:mg"])).median()
    g_ia = (100 * (1 - ia["eig:mg"] / ia["entropy:mg"])).median()
    return bool(g_ia > g_ll), {"logloss_gap_%": g_ll, "iae_gap_%": g_ia}


# ------------------------------------------------------------------ Section 6.2 (factorial)
@claim("ERF-helps-even-with-entropy", "Results 6.2", "Entropy:mg improves log loss over MFI:pi (median > 0)")
def _():
    ll = bank_means(TM, N0, "ilogloss")
    v = (100 * (1 - ll["entropy:mg"] / ll["mfi:pi"])).median()
    return bool(v > 0), {"median_%": v}


@claim("criterion-under-plugin-small", "Results 6.2",
       "under plug-in parameters, switching the criterion changes log loss little (|median| < half the ERF gain)")
def _():
    ll = bank_means(TM, N0, "ilogloss")
    e_pi = (100 * (1 - ll["entropy:pi"] / ll["mfi:pi"])).median()
    g_pi = (100 * (1 - ll["eig:pi"] / ll["mfi:pi"])).median()
    e_mg = (100 * (1 - ll["entropy:mg"] / ll["mfi:pi"])).median()
    return bool(max(abs(e_pi), abs(g_pi)) < 0.5 * e_mg), {"Ent:pi": e_pi, "EIG:pi": g_pi, "Ent:mg": e_mg}


@claim("entropy-worst-score", "Results 6.2 / Discussion",
       "entropy selection has the largest score IAE among adaptive rules under each measurement model (N=8)")
def _():
    ia = bank_means(TM, N0, "iae").mean(axis=0)
    pi = ia[[p for p in ADAPTIVE if p.endswith(":pi")]]
    mg = ia[[p for p in ADAPTIVE if p.endswith(":mg")]]
    return bool(pi.idxmax() == "entropy:pi" and mg.idxmax() == "entropy:mg"), \
        {"pi": pi.round(5).to_dict(), "mg": mg.round(5).to_dict()}


@claim("info-criteria-equivalent-mg", "Results 6.2",
       "under the marginal model MFI, PWFI, EIG and EVR have mean log losses within 0.5% of each other (N=8)")
def _():
    ll = bank_means(TM, N0, "ilogloss").mean(axis=0)
    v = ll[["mfi:mg", "pwfi:mg", "eig:mg", "evr:mg"]]
    spread = float(v.max() / v.min() - 1)
    return bool(spread < 0.005), {"spread": spread, **v.round(5).to_dict()}


@claim("EIGmg-improves-score", "Results 6.2", "EIG:mg improves the score IAE over MFI:pi (median > 0)")
def _():
    ia = bank_means(TM, N0, "iae")
    v = (100 * (1 - ia["eig:mg"] / ia["mfi:pi"])).median()
    return bool(v > 0), {"median_%": v}


# ------------------------------------------------------------------ Section 6.3 (panel size)
@claim("H2-monotone", "Results 6.3", "the H2 gain decreases monotonically with N on every benchmark")
def _():
    out, ok = {}, True
    for d in CONF:
        s = C[(C.hypothesis == "H2") & (C.dataset == d)].sort_values("N")
        v = (100 * s.rel_improvement).round(2).tolist()
        out[d] = dict(zip(s.N.astype(int).tolist(), v))
        ok &= bool(np.all(np.diff(v) < 0))
    return bool(ok), out


@claim("H2-vanishes-64", "Results 6.3", "the H2 gain nearly vanishes at N=64 (median < 1.5%)")
def _():
    v = 100 * H("H2", 64).rel_improvement.median()
    return bool(v < 1.5), {"median_%": v}


@claim("IAE-larger-noisier", "Results 6.3",
       "score-error gains (EIG:mg vs MFI:pi) are larger than log-loss gains but less regular across N")
def _():
    s = S[(S.new == "eig:mg") & (S.base == "mfi:pi") & (S.metric == "iae") & S.dataset.isin(CONF)]
    c = C[(C.hypothesis == "H2") & C.dataset.isin(CONF)]
    mono_ll = sum(bool(np.all(np.diff(c[c.dataset == d].sort_values("N").rel_improvement) < 0)) for d in CONF)
    mono_ia = sum(bool(np.all(np.diff(s[s.dataset == d].sort_values("N").rel_improvement) < 0)) for d in CONF)
    larger = s.rel_improvement.median() > c.rel_improvement.median()
    return bool(larger and mono_ia < mono_ll), {"median_iae_%": 100 * s.rel_improvement.median(),
                                                 "median_ll_%": 100 * c.rel_improvement.median(),
                                                 "monotone_banks_iae": mono_ia, "monotone_banks_ll": mono_ll}


@claim("H1-no-trend", "Results 6.3", "the EIG-vs-entropy advantage shows no monotone trend with N")
def _():
    c = C[(C.hypothesis == "H1") & C.dataset.isin(CONF)]
    med = c.groupby("N").rel_improvement.median()
    rho = spearmanr(c.N, c.rel_improvement).statistic
    mono = bool(np.all(np.diff(med.values) < 0) or np.all(np.diff(med.values) > 0))
    return bool(not mono and abs(rho) < 0.5), {"median_by_N_%": (100 * med).round(2).to_dict(), "spearman": rho}


# ------------------------------------------------------------------ Section 6.4 (curves)
R = None


def _runs():
    global R
    if R is None:
        R = runs("confirm")
    return R


def _curve_means():
    r = _runs()
    r = r[(r.N == N0) & r.dataset.isin(CONF)]
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if f"err@{c}" in r]
    return {d: g.groupby(["policy", "target"])[[f"err@{c}" for c in cps]].mean().groupby("policy").mean()
            for d, g in r.groupby("dataset")}


@claim("model-starts-below-naive", "Results 6.4",
       "after one item every model-based policy is below the naive error, and the head start is smallest on GSM8K")
def _():
    out = {d: float(tm.loc[["entropy:mg", "mfi:pi", "eig:mg"], "err@1"].max() / tm.loc["naive", "err@1"])
           for d, tm in _curve_means().items()}
    ok = all(v < 1 for v in out.values()) and max(out, key=out.get) == "ollb_gsm8k"
    return bool(ok), out


@claim("plugin-fast-early", "Results 6.4", "at t=5 a plug-in policy is ahead of EIG:mg on most benchmarks")
def _():
    cm = _curve_means()
    k = sum(int(min(tm.loc["mfi:pi", "err@5"], tm.loc["eig:pi", "err@5"]) < tm.loc["eig:mg", "err@5"])
            for tm in cm.values())
    return bool(k > len(cm) / 2), {"ahead": k, "B": len(cm)}


@claim("plugin-levels-off", "Results 6.4",
       "between t=25 and t=100 plug-in Fisher CAT's error falls less than EIG:mg's (median), and EIG:mg ends lower on most banks")
def _():
    cm = _curve_means()
    pi = np.median([tm.loc["mfi:pi", "err@100"] / tm.loc["mfi:pi", "err@25"] for tm in cm.values()])
    mg = np.median([tm.loc["eig:mg", "err@100"] / tm.loc["eig:mg", "err@25"] for tm in cm.values()])
    k = sum(int(tm.loc["eig:mg", "err@100"] < tm.loc["mfi:pi", "err@100"]) for tm in cm.values())
    return bool(pi > mg and k > len(cm) / 2), {"ratio100/25_mfi:pi": pi, "ratio100/25_eig:mg": mg, "eig_below": k}


@claim("entropy-catches-up", "Results 6.4",
       "after 100 items Entropy:mg has the lowest error of all policies on most benchmarks")
def _():
    best = {d: tm["err@100"].drop("naive").idxmin() for d, tm in _curve_means().items()}
    k = sum(v == "entropy:mg" for v in best.values())
    return bool(k > len(best) / 2), best


@claim("crossover-small-vs-large", "Results 6.4",
       "the crossover occurs within 100 items far more often on banks of at most 1,172 items than on larger banks")
def _():
    man = json.load(open("data/processed/MANIFEST.json"))["datasets"]
    X = pd.read_csv("paper/generated/crossover.csv") if os.path.exists("paper/generated/crossover.csv") else None
    if X is None:
        return None, "crossover.csv missing (run make_report first)"
    small = X[X.n_items <= 1172].crossover.notna().mean()
    large = X[X.n_items > 1172].crossover.notna().mean()
    return bool(small > 0.8 and large < 0.3), {"share_small": small, "share_large": large}


@claim("crossover-grows-with-size", "Results 6.4",
       "the entropy-vs-EIG crossover item increases with bank size across panel sizes (Spearman rho > 0.5)")
def _():
    man = json.load(open("data/processed/MANIFEST.json"))["datasets"]
    r = _runs()
    r = r[r.dataset.isin(CONF) & r.policy.isin(["entropy:mg", "eig:mg"])]
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if f"err@{c}" in r]
    xs, ys = [], []
    for (d, N), g in r.groupby(["dataset", "N"]):
        tm = g.groupby(["policy", "target"])[[f"err@{c}" for c in cps]].mean().groupby("policy").mean()
        below = (tm.loc["entropy:mg"] <= tm.loc["eig:mg"]).values
        cross = next((cps[k] for k in range(len(cps)) if below[k:].all()), 101)
        xs.append(man[d]["n_items"]); ys.append(cross)
    rho = spearmanr(xs, ys).statistic
    return bool(rho > 0.5), {"rho": rho, "points": len(xs)}


@claim("evr-moves-partway", "Results 6.4 / Discussion",
       "EVR:mg ends below EIG:mg more often on the entropy-best (small) banks than on the others, but not below entropy")
def _():
    best, notb, ent_beats_evr = [], [], True
    for d, tm in _curve_means().items():
        lt = bool(tm.loc["evr:mg", "err@100"] < tm.loc["eig:mg", "err@100"])
        if tm["err@100"].drop("naive").idxmin() == "entropy:mg":
            best.append(lt)
            ent_beats_evr &= bool(tm.loc["entropy:mg", "err@100"] <= tm.loc["evr:mg", "err@100"])
        else:
            notb.append(lt)
    ok = bool(best) and bool(notb) and np.mean(best) > np.mean(notb) and ent_beats_evr
    return bool(ok), {"evr<eig on entropy-best banks": best, "evr<eig on others": notb}


@claim("adaptivity-helps", "Results 6.4",
       "under the marginal model, EIG:mg has a lower mean score IAE than random selection (N=8)")
def _():
    ia = bank_means(TM, N0, "iae").mean(axis=0)
    return bool(ia["eig:mg"] < ia["random:mg"]), {"eig:mg": float(ia["eig:mg"]), "random:mg": float(ia["random:mg"])}


# ------------------------------------------------------------------ Section 6.5 (coverage)
@claim("plugin-overconfident", "Results 6.5", "plug-in Fisher CAT intervals are overconfident at N=8 (coverage < 0.8)")
def _():
    v = bank_means(TM, N0, "coverage")["mfi:pi"].mean()
    return bool(v < 0.8), {"coverage": v}


@claim("marg-helps-most-small-N", "Results 6.5",
       "the coverage gain of EIG:mg over MFI:pi is larger at N=4 than at N=64")
def _():
    g = {N: float((bank_means(TM, N, "coverage")["eig:mg"] - bank_means(TM, N, "coverage")["mfi:pi"]).mean())
         for N in (4, 64)}
    return bool(g[4] > g[64]), g


@claim("no-adaptive-nominal", "Results 6.5",
       "no model-based adaptive policy reaches nominal coverage on real data (all < 0.85 at every N)")
def _():
    mx = {}
    for N in sorted(TM.N.unique()):
        cov = bank_means(TM, N, "coverage")
        mx[int(N)] = float(cov[[p for p in ADAPTIVE if p in cov]].mean(axis=0).max())
    return bool(max(mx.values()) < 0.85), mx


@claim("coverage-opposite-trends", "Results 6.5",
       "as N grows from 4 to 64, EIG:mg and MFI:pi coverage rises on the synthetic banks (mean) and falls on real data (mean)")
def _():
    if TM is None or not set(SYN) <= set(TM.dataset):
        return None, "synthetic results missing"
    out, ok = {}, True
    for p in ["eig:mg", "mfi:pi"]:
        real = [float(bank_means(TM, N, "coverage")[p].mean()) for N in (4, 64)]
        syn = [float(bank_means(TM, N, "coverage", SYN)[p].mean()) for N in (4, 64)]
        out[p] = {"real": real, "synthetic": syn}
        ok &= real[1] < real[0] and syn[1] > syn[0]
    return bool(ok), out


@claim("both-overconfident-64", "Discussion", "at N=64 both EIG:mg and MFI:pi are overconfident (coverage < 0.8)")
def _():
    cov = bank_means(TM, 64, "coverage")
    v = {p: float(cov[p].mean()) for p in ["eig:mg", "mfi:pi"]}
    return bool(max(v.values()) < 0.8), v


# ------------------------------------------------------------------ Section 6.6 (diagnostics)
@claim("unlocated-shrinks", "Results 6.6", "fewer items have identical verdicts with 64 calibrators than with 4")
def _():
    m = D[D.dataset.isin(CONF)].groupby("N").frac_items_extreme.mean()
    return bool(m.loc[64] < m.loc[4]), m.round(3).to_dict()


@claim("a-gt4-absent-small-N", "Results 6.6",
       "under the EB prior, plug-in discriminations above 4 are practically absent at N=4 (< 1% of items) and < 15% at N=64")
def _():
    m = D[D.dataset.isin(CONF)].groupby("N").frac_a_gt_4.mean()
    return bool(m.loc[4] < 0.01 and m.loc[64] < 0.15), m.round(4).to_dict()


@claim("ERF-shrinks-a", "Results 6.6", "the ERF effective discrimination is below the plug-in MAP discrimination at N=4")
def _():
    m = D[D.dataset.isin(CONF)].groupby("N")[["median_a_map", "median_a_eff_mg"]].mean()
    return bool(m.loc[4, "median_a_eff_mg"] < m.loc[4, "median_a_map"]), m.round(3).to_dict()


@claim("mechanism-rho", "Results 6.6", "H2 gain increases with the fraction of unlocated items (Spearman rho > 0.5)")
def _():
    c = C[(C.hypothesis == "H2") & C.dataset.isin(CONF)]
    dm = D.groupby(["dataset", "N"]).frac_items_extreme.mean().reset_index()
    M = c.merge(dm, on=["dataset", "N"])
    rho = spearmanr(M.frac_items_extreme, M.rel_improvement).statistic
    return bool(rho > 0.5), {"rho": rho, "points": int(len(M))}


# ------------------------------------------------------------------ Section 6.7 (synthetic)
def _syn(metric, N, banks=SYN):
    s = TM[(TM.N == N) & TM.dataset.isin(banks)]
    return s.groupby(["dataset", "policy"])[metric].mean().unstack()


def _has_syn():
    return TM is not None and set(SYN) <= set(TM.dataset)


@claim("syn-EIGmg-best-theta-N8", "Results 6.7",
       "at N=8, EIG:mg has the smallest ability error of the four plotted policies on every synthetic bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    e = _syn("itrue_err", 8)[["entropy:mg", "mfi:pi", "eig:pi", "eig:mg"]]
    return bool((e.idxmin(axis=1) == "eig:mg").all()), e.round(4).to_dict("index")


@claim("syn-theta-coverage", "Results 6.7",
       "EIG:mg's ability interval covers the true ability more often than MFI:pi's at N=8 on every synthetic bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    c = _syn("true_theta_cov", 8)
    return bool((c["eig:mg"] > c["mfi:pi"]).all()), c[["eig:mg", "mfi:pi"]].round(3).to_dict("index")


@claim("syn-ranking-varies-N4", "Results 6.7",
       "with four calibrators the best of the four plotted policies (ability error) is not the same on every synthetic bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    e = _syn("itrue_err", 4)[["entropy:mg", "mfi:pi", "eig:pi", "eig:mg"]]
    best = e.idxmin(axis=1)
    return bool(best.nunique() > 1), best.to_dict()


@claim("syn-mg-ahead-large-N", "Results 6.7",
       "from N=16 on, EIG:mg has a smaller ability error than both EIG:pi and MFI:pi on every synthetic bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    ok, out = True, {}
    for N in sorted(n for n in TM.N.unique() if n >= 16):
        e = _syn("itrue_err", int(N))
        d = e[["eig:pi", "mfi:pi"]].min(axis=1) - e["eig:mg"]
        out[int(N)] = d.round(4).to_dict()
        ok &= bool((d > 0).all())
    return bool(ok), out


@claim("syn-theta-cov-falls", "Results 6.7",
       "coverage of the true ability is lower at N=64 than at N=8 for every adaptive policy on every synthetic bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    c8, c64 = _syn("true_theta_cov", 8), _syn("true_theta_cov", 64)
    pols = [p for p in ADAPTIVE if p in c8.columns]
    diff = c64[pols] - c8[pols]
    return bool((diff < 0).all().all()), diff.round(3).to_dict("index")


@claim("syn-coverage-degrades", "Results 6.5",
       "EIG:mg's score coverage at N=8 decreases from the well-specified bank to guessing to two dimensions to real data,"
       " and is near nominal (within 0.03) on the well-specified bank")
def _():
    if not _has_syn():
        return None, "synthetic results missing"
    c = _syn("coverage", 8)["eig:mg"]
    real = float(bank_means(TM, 8, "coverage")["eig:mg"].mean())
    seq = [float(c["synth_2pl"]), float(c["synth_guess"]), float(c["synth_2dim"]), real]
    ok = all(a > b for a, b in zip(seq, seq[1:])) and abs(seq[0] - 0.9) < 0.03
    return bool(ok), {"2pl": seq[0], "guess": seq[1], "2dim": seq[2], "real": seq[3]}


# ------------------------------------------------------------------ Section 6.8 (sensitivity)
def _sens(phase, N, met, new, base):
    tm = load(phase, "target_means.csv.gz")
    if tm is None:
        return None
    per = tm[(tm.N == N) & tm.dataset.isin(CONF)].groupby(["dataset", "policy"])[met].mean().unstack()
    return 100 * (1 - per[new] / per[base])


@claim("sens-dedup-unchanged", "Results 6.8",
       "removing near-duplicate calibrators leaves the H2 gain at N=8 positive and within 2 points of the main analysis on the same banks")
def _():
    tm = load("sens_dedup", "target_means.csv.gz")
    if tm is None:
        return None, "sens_dedup missing"
    banks = sorted(set(tm.dataset) & set(CONF))
    v = _sens("sens_dedup", 8, "ilogloss", "eig:mg", "mfi:pi")
    m = 100 * (1 - bank_means(TM, 8, "ilogloss", banks)["eig:mg"] / bank_means(TM, 8, "ilogloss", banks)["mfi:pi"])
    return bool(v.median() > 0 and abs(v.median() - m.median()) < 2), {"dedup": float(v.median()), "main_same_banks": float(m.median())}


@claim("sens-fixedprior-essential", "Results 6.8",
       "with the fixed item prior the H2 gain at N=8 is smaller than in the main analysis and reverses on some benchmarks")
def _():
    v = _sens("sens_fixedprior", 8, "ilogloss", "eig:mg", "mfi:pi")
    if v is None:
        return None, "sens_fixedprior missing"
    m = _sens("confirm", 8, "ilogloss", "eig:mg", "mfi:pi")
    return bool(v.median() < m.median() and (v < 0).any()), {"fixed": v.round(2).to_dict(), "main_median": float(m.median())}


@claim("sens-fixedprior-halves", "Introduction",
       "without the empirical-Bayes prior the median H2 gain at N=8 roughly halves (ratio between 0.35 and 0.65)")
def _():
    v = _sens("sens_fixedprior", 8, "ilogloss", "eig:mg", "mfi:pi")
    if v is None:
        return None, "sens_fixedprior missing"
    m = _sens("confirm", 8, "ilogloss", "eig:mg", "mfi:pi")
    r = float(v.median() / m.median())
    return bool(0.35 <= r <= 0.65), {"ratio": r}


@claim("sens-1plreg-H1-vanishes", "Results 6.8",
       "under the regularized 1PL the EIG-vs-entropy log-loss difference at N=8 is below 1% (both models) and smaller than under the 2PL")
def _():
    out = {}
    for ph in ["sens_1pl_reg", "confirm"]:
        tm = load(ph, "target_means.csv.gz")
        if tm is None:
            return None, f"{ph} missing"
        per = tm[(tm.N == 8) & tm.dataset.isin(CONF)].groupby(["dataset", "policy"]).ilogloss.mean().unstack()
        out[ph] = {"mg": float((100 * (1 - per["eig:mg"] / per["entropy:mg"])).median()),
                   "pi": float((100 * (1 - per["eig:pi"] / per["entropy:pi"])).median())}
    r, m = out["sens_1pl_reg"], out["confirm"]
    ok = abs(r["mg"]) < 1 and abs(r["pi"]) < 1 and abs(r["mg"]) < abs(m["mg"]) and abs(r["pi"]) < abs(m["pi"])
    return bool(ok), out


@claim("sens-1plreg-H2-small", "Results 6.8",
       "under the regularized 1PL the H2 gain at N=8 is small (median below 2%)")
def _():
    v = _sens("sens_1pl_reg", 8, "ilogloss", "eig:mg", "mfi:pi")
    if v is None:
        return None, "sens_1pl_reg missing"
    return bool(v.median() < 2), {"median": float(v.median())}


def _ph1():
    f = "artifacts/posthoc_baselines/_analysis/contrasts.csv"
    if not os.path.exists(f):
        return None, None, None
    C_ = pd.read_csv(f)
    dec = json.load(open("artifacts/posthoc_baselines/_analysis/decisions.json"))
    T_ = pd.read_csv("artifacts/posthoc_baselines/_analysis/target_means.csv.gz")
    return C_, dec, T_


@claim("ph1-reproduction-exact", "Results 6.9", "PH1 reproduced every recomputed confirmatory row exactly")
def _():
    C_, dec, _ = _ph1()
    if C_ is None:
        return None, "PH1 missing"
    return bool(dec["reproduction_exact"]), {k: v["max_abs_diff"] for k, v in dec["reproduction"].items()}


@claim("ph1-eap-worse", "Results 6.9",
       "posterior-mean plug-in parameters are significantly worse than EOST-M on log loss at N=8 in >= 5 banks")
def _():
    C_, _, _ = _ph1()
    if C_ is None:
        return None, "PH1 missing"
    s = C_[(C_.base == "mfi:eap") & (C_.metric == "ilogloss") & (C_.N == 8)]
    return bool(s.eost_better.sum() >= 5 and s.base_better.sum() == 0), {"eost_better": int(s.eost_better.sum())}


@claim("ph1-rasch-close", "Results 6.9 / Abstract",
       "vs the regularized Rasch plug-in: EOST-M log loss lower on all banks but by < 3% (median, N=8); "
       "the Rasch plug-in has better coverage; rule 1 was triggered")
def _():
    C_, dec, T_ = _ph1()
    if C_ is None:
        return None, "PH1 missing"
    s = C_[(C_.base == "mfi:1pr") & (C_.metric == "ilogloss") & (C_.N == 8)]
    cov = T_[T_.N == 8].groupby(["dataset", "policy"]).coverage.mean().unstack().mean(axis=0)
    med = float(100 * s.rel_improvement.median())
    ok = (s.rel_improvement > 0).all() and 0 < med < 3 and cov["mfi:1pr"] > cov["eig:mg"] and dec["rule1_triggered"]
    return bool(ok), {"median_ll_%": med, "cov_rasch": float(cov["mfi:1pr"]), "cov_eost": float(cov["eig:mg"]),
                      "rule1_triggered": dec["rule1_triggered"]}


@claim("ph1-rasch-score", "Results 6.9",
       "vs the regularized Rasch plug-in, EOST-M's score error is lower on most banks at N=8 and the gap widens at N=16")
def _():
    C_, _, _ = _ph1()
    if C_ is None:
        return None, "PH1 missing"
    s8 = C_[(C_.base == "mfi:1pr") & (C_.metric == "iae") & (C_.N == 8)]
    s16 = C_[(C_.base == "mfi:1pr") & (C_.metric == "iae") & (C_.N == 16)]
    ok = (s8.rel_improvement > 0).sum() > len(s8) / 2 and s16.rel_improvement.median() > s8.rel_improvement.median()
    return bool(ok), {"pos_N8": int((s8.rel_improvement > 0).sum()), "med_N8": float(100 * s8.rel_improvement.median()),
                      "med_N16": float(100 * s16.rel_improvement.median())}


@claim("ph1-efi-equivalent", "Results 6.9 / Theory / Related work",
       "posterior-expected Fisher information selects equally good items: |median log-loss difference| < 0.5% at N=8, no bank significant")
def _():
    C_, _, _ = _ph1()
    if C_ is None:
        return None, "PH1 missing"
    s = C_[(C_.base == "efi:mg") & (C_.metric == "ilogloss") & (C_.N == 8)]
    ok = abs(100 * s.rel_improvement.median()) < 0.5 and s.eost_better.sum() == 0 and s.base_better.sum() == 0
    return bool(ok), {"median_%": float(100 * s.rel_improvement.median())}


@claim("ph2-no-promising", "Results 6.9",
       "PH2: no shrinkage level meets the pre-stated criterion; coverage of EIG:mg@tau stays within 0.05 of EOST-M and log loss rises for every tau")
def _():
    f = "artifacts/posthoc_shrinkage/_analysis/decisions.json"
    if not os.path.exists(f):
        return None, "PH2 missing"
    d = json.load(open(f))
    S_ = pd.DataFrame(d["path"])
    mg = S_[S_.criterion == "eig:mg"]
    ok = (not d["promising"]) and d["reproduction_exact"] and (abs(mg.coverage - d["eost_coverage"]) < 0.05).all() \
        and (mg["ll_rel_vs_eost_%"] > 0).all()
    return bool(ok), {"coverage": mg.coverage.round(3).tolist(), "ll": mg["ll_rel_vs_eost_%"].round(2).tolist(),
                      "eost_cov": d["eost_coverage"]}


def _ph3():
    f = "artifacts/posthoc_coverage/decisions.json"
    if not os.path.exists(f):
        return None, None, None
    return (json.load(open(f)), pd.read_csv("artifacts/posthoc_coverage/banks.csv"),
            pd.read_csv("artifacts/posthoc_coverage/cells.csv"))


@claim("ph3-wider-not-more-accurate", "Results 6.9 / Discussion / Conclusion",
       "PH3: the Rasch-type intervals are wider than EOST-M's on all 8 benchmarks (Holm-significant on all), its score "
       "error is not lower on at least half of the benchmarks where its coverage is higher, and the pre-stated verdict "
       "is 'wider_intervals'")
def _():
    d, B, _c = _ph3()
    if d is None:
        return None, "PH3 missing"
    hi = B[B.cov_rasch > B.cov_eost]
    ok = d["verdict"] == "wider_intervals" and int(B.rasch_wider.sum()) == 8 \
        and d["n_banks_rasch_wider_sig_holm"] == 8 and hi.rasch_iae_not_lower.sum() >= len(hi) / 2
    return bool(ok), {"verdict": d["verdict"], "wider": int(B.rasch_wider.sum()),
                      "wider_sig": d["n_banks_rasch_wider_sig_holm"], "higher_cov": int(len(hi)),
                      "iae_not_lower": int(hi.rasch_iae_not_lower.sum())}


@claim("ph3-coverage-follows-ratio", "Results 6.9 / Discussion",
       "PH3: coverage is almost a monotone function of the width-to-IAE ratio (Spearman rho > 0.9 over all bank-model "
       "cells), and the ratio is below the calibrated-Gaussian value in every cell")
def _():
    d, _B, cells = _ph3()
    if d is None:
        return None, "PH3 missing"
    rho = d["spearman_ratio_coverage"]["rho"]
    ok = rho > 0.9 and bool((cells.ratio < d["gaussian_ratio"]).all())
    return bool(ok), {"rho": rho, "max_ratio": float(cells.ratio.max()), "gauss": d["gaussian_ratio"]}


@claim("adaptive-shortfall", "Results 6.9 / Discussion / Conclusion",
       "no model-based adaptive policy (confirmatory and post hoc, every N) reaches 0.9 mean coverage on the real banks, "
       "while Random:mg is close to nominal (>= 0.85 at N=8) and EOST-M covers >= 0.85 on the well-specified synthetic bank")
def _():
    worst = {}
    for ph in ("confirm", "posthoc_baselines"):
        f = f"artifacts/{ph}/_analysis/target_means.csv.gz"
        if not os.path.exists(f):
            return None, f"{ph} missing"
        t = pd.read_csv(f)
        t = t[t.dataset.isin(CONF) & ~t.policy.isin(["naive", "random:mg", "random:pi"])]
        m = t.groupby(["N", "policy", "dataset"]).coverage.mean().groupby(["N", "policy"]).mean()
        worst[ph] = float(m.max())
    f2 = "artifacts/posthoc_shrinkage/_analysis/target_means.csv.gz"
    if os.path.exists(f2):  # PH2 (N = 8 only)
        t = pd.read_csv(f2)
        t = t[t.dataset.isin(CONF)]
        worst["posthoc_shrinkage"] = float(t.groupby(["policy", "dataset"]).coverage.mean().groupby("policy").mean().max())
    rnd = float(bank_means(TM, 8, "coverage")["random:mg"].mean())
    syn = float(TM[(TM.dataset == "synth_2pl") & (TM.N == 8) & (TM.policy == "eig:mg")].coverage.mean())
    ok = max(worst.values()) < 0.9 and rnd >= 0.85 and syn >= 0.85
    return bool(ok), {"max_adaptive_cov": worst, "random_mg_N8": rnd, "synth_2pl_eig_mg_N8": syn}


@claim("sens-1pl-degenerate", "Results 6.8",
       "the pre-registered one-parameter calibration is degenerate: median common discrimination > 100, EOST-M coverage < 0.5")
def _():
    d = diag("sens_1pl")
    tm = load("sens_1pl", "target_means.csv.gz")
    if d is None or tm is None:
        return None, "sens_1pl missing"
    a = float(d[d.dataset.isin(CONF)].median_a_map.median())
    cov = float(bank_means(tm, 8, "coverage")["eig:mg"].mean())
    return bool(a > 100 and cov < 0.5), {"median_a": a, "coverage_eig_mg_N8": cov}


@claim("sens-1pl-reg-well-behaved", "Results 6.8",
       "the post hoc regularized one-parameter calibration is well behaved: median common discrimination < 10, coverage > 0.5")
def _():
    d = diag("sens_1pl_reg")
    tm = load("sens_1pl_reg", "target_means.csv.gz")
    if d is None or tm is None:
        return None, "sens_1pl_reg missing"
    a = float(d[d.dataset.isin(CONF)].median_a_map.median())
    cov = float(bank_means(tm, 8, "coverage")["eig:mg"].mean())
    return bool(a < 10 and cov > 0.5), {"median_a": a, "coverage_eig_mg_N8": cov}


# ------------------------------------------------------------------ Discussion
@claim("EVR-like-EIG", "Discussion", "EVR and EIG give score IAEs within 5% of each other (N=8, both models)")
def _():
    ia = bank_means(TM, N0, "iae").mean(axis=0)
    r_mg = abs(ia["evr:mg"] / ia["eig:mg"] - 1)
    r_pi = abs(ia["evr:pi"] / ia["eig:pi"] - 1)
    return bool(max(r_mg, r_pi) < 0.05), {"mg": r_mg, "pi": r_pi}


@claim("small-panel-reliable", "Discussion",
       "at N=4 and N=8, EIG:mg beats MFI:pi on log loss (H2) and score IAE in most benchmarks, never significantly worse")
def _():
    out, ok = {}, True
    for N in (4, 8):
        h2 = H("H2", N)
        s = S[(S.new == "eig:mg") & (S.base == "mfi:pi") & (S.metric == "iae") & (S.N == N) & S.dataset.isin(CONF)]
        sig_new = int(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum())
        sig_base = int(((s.p_holm < 0.05) & (s.mean_diff < 0)).sum())
        out[N] = {"H2_fav": int(h2.favours_new.sum()), "H2_base": int(h2.favours_base.sum()), "iae_fav": sig_new,
                  "iae_base": sig_base, "B": int(len(h2))}
        ok &= h2.favours_new.sum() > len(h2) / 2 and h2.favours_base.sum() == 0 and sig_base == 0 \
            and (s.rel_improvement > 0).sum() > len(s) / 2
    return bool(ok), out


@claim("fades-32-64", "Discussion", "the H2 gain fades with 32-64 calibrators (median < 2.5%)")
def _():
    v = {N: 100 * H("H2", N).rel_improvement.median() for N in (32, 64)}
    return bool(max(v.values()) < 2.5), v


@claim("ranking-adaptive-beats-random", "Results 6.4",
       "after 25 items, median Kendall tau of EIG:mg and MFI:pi exceeds that of naive subsampling by > 0.1, and EIG:mg and MFI:pi differ by < 0.05")
def _():
    f = "artifacts/explore_ranking/ranking.csv"
    if not os.path.exists(f):
        return None, "ranking missing"
    R_ = pd.read_csv(f)
    piv = R_[R_.t == 25].pivot_table(index="dataset", columns="policy", values="kendall")
    med = piv.median()
    ok = (med["eig:mg"] - med["naive"] > 0.1) and (med["mfi:pi"] - med["naive"] > 0.1) \
        and abs(med["eig:mg"] - med["mfi:pi"]) < 0.05
    return bool(ok), med.round(3).to_dict()



# ------------------------------------------------------------------ revision (iteration 3)
@claim("clusters-h2-strict", "Study 1 / cluster-aware re-analysis",
       "with strict clusters as units H2 is Holm-significant on fewer than 5 banks (not supported) while its median "
       "effect stays above 4% with a cluster-bootstrap interval excluding 0; H3 and H4 stay supported with every unit; "
       "with heuristic families all four frozen decisions are unchanged")
def _():
    f = "artifacts/confirm/_clusters/decisions.json"
    if not os.path.exists(f):
        return None, "cluster analysis missing"
    d = json.load(open(f))
    frozen = load("confirm", "decisions.json")["decisions"]
    ok = d["strict"]["H2"]["favour_new"] < 5 and not d["strict"]["H2"]["supported"] \
        and d["strict"]["H2"]["median_rel_improvement_pct"] > 4 \
        and d["strict_cluster_bootstrap"]["H2"]["median_rel_improvement_ci95_pct"][0] > 0 \
        and all(d[u][h]["supported"] for u in ("target", "heuristic", "strict") for h in ("H3", "H4")) \
        and all(d["heuristic"][h]["supported"] == frozen[h]["supported"] for h in ("H1", "H2", "H3", "H4"))
    return bool(ok), {u: {h: (d[u][h]["favour_new"], d[u][h]["supported"]) for h in d[u]} for u in ("target", "heuristic", "strict")}


@claim("kappa-chains", "Design / strict clusters",
       "Cohen's kappa >= 0.8 chains more than 300 of the 395 HellaSwag systems into one component, whereas the "
       "residual-correlation clusters of the leaderboard have at most 40 members")
def _():
    from eost.data import load_bank
    from eost.families import strict_clusters
    B = load_bank("data/processed/ollb_hellaswag.npz")
    big = int(np.bincount(strict_clusters(B, 0.8)).max())
    c = json.load(open("configs/study2_clusters.json"))["banks"]["ollb_gsm8k"]
    return bool(big > 300 and c["max_size"] <= 40), {"kappa_component": big, "residual_max": c["max_size"]}


@claim("cost-breakeven", "Cost",
       "with 8 fresh calibrators and 100-item tests, the break-even against full evaluation lies between 8 and 10 new "
       "systems on every bank; EOST-M matches 100 random items scored by the same model with fewer items on 7 of 8 banks")
def _():
    f = "artifacts/cost/breakeven.csv"
    if not os.path.exists(f):
        return None, "cost analysis missing"
    D = pd.read_csv(f)
    n_save = int((pd.to_numeric(D.items_match_random_mg100) < 100).sum())
    ok = D.breakeven_vs_full.between(8, 10).all() and n_save == 7
    return bool(ok), {"breakeven_full": D.breakeven_vs_full.round(2).tolist(), "random_saves": n_save}


@claim("cap-logistic", "Theory, Proposition 4",
       "the information of the expected logistic response under N(0, s^2) difficulty uncertainty never exceeds "
       "2/(pi s^2) on a grid of discriminations and abilities")
def _():
    from scipy.special import expit
    from scipy.stats import norm
    L = np.linspace(-45, 45, 60001)
    w = expit(L) * (1 - expit(L)); w /= w.sum()
    th = np.linspace(-3, 3, 121)
    worst = 0.0
    for s_ in (0.3, 1.0):
        for a in (0.5, 4.0, 40.0):
            u = (th[:, None] - L[None, :] / a) / s_
            P, Q, dP = norm.cdf(u) @ w, norm.sf(u) @ w, (norm.pdf(u) / s_) @ w
            ok = (P > 1e-280) & (Q > 1e-280)
            worst = max(worst, float((dP[ok] ** 2 / (P[ok] * Q[ok])).max() * np.pi * s_ ** 2 / 2))
    return bool(worst <= 1 + 1e-9), {"max_info_over_cap": worst}


def _ppc():
    f = "artifacts/eb_robustness/ppc.csv"
    return pd.read_csv(f) if os.path.exists(f) else None


@claim("ppc-extreme", "Study 2 / robustness of the empirical-Bayes hyperprior",
       "the share of extreme items is reproduced on average (medians within 0.02) but flagged (p<0.05) on more than "
       "5% of panels, most of them on HellaSwag and MMLU; the spread of proportions correct is flagged on at most 5%")
def _():
    P = _ppc()
    if P is None:
        return None, "ppc missing"
    s = P[P.stat == "frac_extreme_items"]
    f = s.ppp < 0.05
    sp = P[P.stat == "sd_item_p"]
    ok = abs(s.observed.median() - s.sim_median.median()) < 0.02 and f.mean() > 0.05 \
        and s[f].dataset.isin(["ollb_hellaswag", "ollb_mmlu"]).mean() > 0.5 and (sp.ppp < 0.05).mean() <= 0.05
    return bool(ok), {"flagged": float(f.mean()), "hs_mmlu_share": float(s[f].dataset.isin(["ollb_hellaswag", "ollb_mmlu"]).mean()),
                      "sd_flagged": float((sp.ppp < 0.05).mean())}


@claim("ppc-rest-lower", "Study 2 / robustness of the empirical-Bayes hyperprior",
       "the item-rest correlation is mostly lower than simulated (flagged below more often than above, median observed "
       "below median simulated), mainly on the multiple-choice Open LLM Leaderboard banks")
def _():
    P = _ppc()
    if P is None:
        return None, "ppc missing"
    s = P[P.stat == "mean_item_rest_r"]
    f = s.ppp < 0.05
    below, above = (f & (s.observed < s.sim_median)).mean(), (f & (s.observed > s.sim_median)).mean()
    mc = s[f].dataset.isin(["ollb_arc", "ollb_hellaswag", "ollb_mmlu", "ollb_winogrande"]).mean()
    ok = below > above and s.observed.median() < s.sim_median.median() and mc > 0.5
    return bool(ok), {"below": float(below), "above": float(above), "mc_share": float(mc)}


# ----------------------------------------------------------------------------- study 2
S2A = "artifacts/study2/_analysis"


def _s2(name):
    f = f"{S2A}/{name}"
    if not os.path.exists(f):
        return None
    return pd.read_csv(f) if f.endswith((".csv", ".csv.gz")) else json.load(open(f))


def _s2pm(N=8):
    tm = _s2("target_means.csv.gz")
    return tm[tm.N == N].groupby(["dataset", "policy"]).mean(numeric_only=True)


def _sec(h, N=8):
    S = _s2("secondary_tests.csv")
    return S[(S.hypothesis == h) & (S.N == N)]


@claim("s2-direction", "Study 2 / summary",
       "apart from C5b every superiority contrast is positive on at least 7 of 8 banks and no competitor is "
       "Holm-significantly better anywhere; the unsupported superiority hypotheses are C2, C4, C5b, C7, C8, and C2, C4, "
       "C7, C8 have exactly 4 significant banks and none against; CRUXEval-O has the fewest targets and no significant contrast")
def _():
    C, d = _s2("confirmatory_tests.csv"), _s2("decisions.json")
    if C is None:
        return None, "study-2 analysis missing"
    dec = d["decisions"]
    sup = [h for h in ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"] if not dec[h]["supported"]]
    pos = {h: int((g.rel_improvement > 0).sum()) for h, g in C.groupby("hypothesis")}
    against = C[(C.p_holm < 0.05) & (C.mean_diff < 0) & (C.hypothesis != "C5b")]
    nt = d["n_targets"]
    ok = all(v >= 7 for h, v in pos.items() if h != "C5b") and against.empty \
        and sup == ["C2", "C4", "C5b", "C7", "C8"] \
        and all(dec[h]["favour_new"] == 4 and dec[h]["favour_base"] == 0 for h in ["C2", "C4", "C7", "C8"]) \
        and min(nt, key=nt.get) == "cruxeval_o" and C[(C.dataset == "cruxeval_o") & (C.p_holm < 0.05)].empty
    return bool(ok), {"positive": pos, "unsupported": sup}


@claim("s2-r2-panel-verdicts", "Study 2 / replication",
       "the R2 gain is larger at N=4 than at N=8 and N=16, no bank is significant at N=16, and deterministic and "
       "aggregate-verdict banks differ by less than one point")
def _():
    C = _s2("confirmatory_tests.csv")
    if C is None:
        return None, "missing"
    m4, m16 = _sec("R2", 4), _sec("R2", 16)
    r2 = C[C.hypothesis == "R2"]
    agg = r2.dataset.isin(["livecodebench", "cruxeval_o"])
    det_, agg_ = 100 * r2[~agg].rel_improvement.median(), 100 * r2[agg].rel_improvement.median()
    ok = m4.rel_improvement.median() > r2.rel_improvement.median() > m16.rel_improvement.median() \
        and int((m16.p_holm < 0.05).sum()) == 0 and abs(det_ - agg_) < 1
    return bool(ok), {"N4": m4.rel_improvement.median(), "N8": r2.rel_improvement.median(),
                      "N16": m16.rel_improvement.median(), "det": det_, "agg": agg_}


@claim("s2-rasch", "Study 2 / integration vs constraint",
       "C1 is larger at N=16 than at N=4; on the score EOST-M and the Rasch-type test are close (positive median, no "
       "bank significant); the Rasch-type raw coverage exceeds EOST-M's")
def _():
    pm = _s2pm()
    if pm is None:
        return None, "missing"
    c4, c16 = _sec("C1", 4), _sec("C1", 16)
    iae = _sec("S:eig:mg|mfi:1pr|iae")
    cov = pm.groupby("policy").coverage.mean()
    ok = c16.rel_improvement.median() > c4.rel_improvement.median() and iae.rel_improvement.median() > 0 \
        and int((iae.p_holm < 0.05).sum()) == 0 and cov["mfi:1pr"] > cov["eig:mg"]
    return bool(ok), {"C1_N4": c4.rel_improvement.median(), "C1_N16": c16.rel_improvement.median(),
                      "iae_med": iae.rel_improvement.median(), "cov": cov[["mfi:1pr", "eig:mg"]].to_dict()}


@claim("s2-reduction", "Study 2 / adaptivity and reduction",
       "the C4 gain is larger at N=16 than at N=4 and similar (within 5 points) for deterministic and aggregate banks; "
       "the ATLAS-style test has the largest score IAE of the model-based policies; MINCE's error at its own n* exceeds "
       "EOST-M's error after 100 items; the tinyBenchmarks anchor shortfall at budget 50 is at most 10% of cells and "
       "the corrected C8 counts equal the frozen one")
def _():
    pm, C = _s2pm(), _s2("confirmatory_tests.csv")
    if pm is None:
        return None, "missing"
    c4, c16 = _sec("C4", 4), _sec("C4", 16)
    r = C[C.hypothesis == "C4"]
    agg = r.dataset.isin(["livecodebench", "cruxeval_o"])
    iae = pm.groupby("policy").iae.mean().drop(["naive"], errors="ignore").dropna()
    tm = _s2("target_means.csv.gz")
    ns = tm[(tm.N == 8) & (tm.policy == "mince:nstar")]
    t = json.load(open("artifacts/study2_posthoc/_analysis/decisions.json"))["tiny"] \
        if os.path.exists("artifacts/study2_posthoc/_analysis/decisions.json") else None
    if t is None:
        return None, "post hoc audit missing"
    ok = c16.rel_improvement.median() > c4.rel_improvement.median() \
        and abs(r[~agg].rel_improvement.median() - r[agg].rel_improvement.median()) < 0.05 \
        and iae.idxmax() == "atlas" and ns.err_nstar.mean() > pm.groupby("policy")["err@100"].mean()["eig:mg"] \
        and max(t["share_short_T50"].values()) <= 0.10 \
        and t["C8_full_budget"]["favour_eost"] == t["C8_frozen"]["favour_eost"] == t["C8_items_matched"]["favour_eost"]
    return bool(ok), {"worst_iae": iae.idxmax(), "nstar_err": float(ns.err_nstar.mean())}


@claim("s2-coverage", "Study 2 / calibrated uncertainty",
       "model-based raw intervals (EOST-M, fully Bayesian, Rasch-type, plug-in) cover < 0.85 while Random:mg and Naive "
       "cover >= 0.88; C5a misses by one bank; C5b is significantly against recalibration exactly on HellaSwag and "
       "WinoGrande; recalibrated EOST-M intervals are much narrower than Naive's (ratio < 0.7); the recalibrated "
       "Rasch-type test is inside [0.85, 0.95] on all banks with intervals 1.8-2.2 times as wide as recalibrated "
       "EOST-M's and is never better on interval score; mixing keeps coverage below 0.80")
def _():
    pm, C, d = _s2pm(), _s2("confirmatory_tests.csv"), _s2("decisions.json")
    if pm is None:
        return None, "missing"
    cov = pm.groupby("policy").coverage.mean()
    w = pm.groupby("policy").mean_width.mean()
    c5b = C[(C.hypothesis == "C5b") & (C.p_holm < 0.05) & (C.mean_diff < 0)].dataset.tolist()
    rr = pm.xs("mfi:1pr+R", level="policy").coverage
    isr = _sec("S:eig:mg+R|mfi:1pr+R|iis")
    ok = all(cov[p] < 0.85 for p in ["eig:mg", "fb:eig", "fb:bmi", "mfi:1pr", "mfi:pi"]) \
        and cov["random:mg"] >= 0.88 and cov["naive"] >= 0.88 and d["decisions"]["C5a"]["n_inside"] == 5 \
        and sorted(c5b) == ["ollb_hellaswag", "ollb_winogrande"] and w["eig:mg+R"] / w["naive"] < 0.7 \
        and rr.between(0.85, 0.95).all() and 1.8 <= w["mfi:1pr+R"] / w["eig:mg+R"] <= 2.2 \
        and int(((isr.p_holm < 0.05) & (isr.mean_diff < 0)).sum()) == 0 and cov["mix:mg"] < 0.80
    return bool(ok), {"cov": cov.round(3).to_dict(), "c5b_against": c5b}


@claim("s2-recal-panel", "Study 2 / calibrated uncertainty",
       "recalibrated EOST-M coverage increases from N=4 to 8 to 16; its interval score is worse than the raw "
       "interval's at N=4 and better at N=16")
def _():
    out = {}
    for N in (4, 8, 16):
        pm = _s2pm(N).groupby("policy")[["coverage", "iis"]].mean()
        out[N] = (pm.loc["eig:mg+R", "coverage"], pm.loc["eig:mg+R", "iis"], pm.loc["eig:mg", "iis"])
    ok = out[4][0] < out[8][0] < out[16][0] and out[4][1] > out[4][2] and out[16][1] < out[16][2]
    return bool(ok), {str(k): v for k, v in out.items()}


@claim("s2-temporal-eb-synth", "Study 2 / temporal validation, synthetic banks, empirical-Bayes variants",
       "temporal: 10 of 11 contrasts significant for EOST-M, none against, C5b not significant; cross-fitted and "
       "Student-t hyperpriors change log loss by less than 1% with no bank significant (cross-fitted: median 0.0%; "
       "Student-t: worse on every bank); recalibration moves coverage "
       "away from 0.9 on the 2PL and guessing banks and towards it on the two-dimensional bank (N=8); no significant "
       "contrast of either study has discordant signs")
def _():
    T, d = _s2("tests_temporal.csv"), _s2("decisions.json")
    if T is None:
        return None, "missing"
    t8 = T[T.N == 8]
    sig = (t8.p_holm < 0.05) & (t8.mean_diff > 0)
    xf, tp = _sec("S:eig:xf|eig:mg|ilogloss"), _sec("S:eig:tp|eig:mg|ilogloss")
    cv = d["secondary_components"]["synthetic"]["coverage"]
    gap = lambda ds, p: abs(cv[f"{ds}|8|{p}"] - 0.9)
    ph = json.load(open("artifacts/study2_posthoc/_analysis/decisions.json"))
    ok = int(sig.sum()) == 10 and int(((t8.p_holm < 0.05) & (t8.mean_diff < 0)).sum()) == 0 \
        and not bool(t8[t8.hypothesis == "C5b"].p_holm.iloc[0] < 0.05) \
        and all(abs(s.rel_improvement.median()) < 0.01 and int((s.p_holm < 0.05).sum()) == 0 for s in (xf, tp)) \
        and bool((tp.rel_improvement < 0).all()) and abs(xf.rel_improvement.median()) < 0.0005 \
        and gap("synth_2pl", "eig:mg+R") > gap("synth_2pl", "eig:mg") \
        and gap("synth_guess", "eig:mg+R") > gap("synth_guess", "eig:mg") \
        and gap("synth_2dim", "eig:mg+R") < gap("synth_2dim", "eig:mg") and ph["signs"]["total_discordant"] == 0
    return bool(ok), {"temporal_sig": int(sig.sum()), "signs": ph["signs"]["total_discordant"]}


@claim("s2-fb", "Study 2 / fully Bayesian CAT (post hoc correction)",
       "the frozen drift is largest on HellaSwag (more than twice the median of the other banks) and the corrected "
       "burn-in at least halves it; at N=8 no fully Bayesian variant (frozen or corrected, BMI or EIG, per-item or joint) "
       "differs significantly from EOST-M in log loss or score error; corrected per-item variants have a positive median "
       "log-loss difference below 1% and are equivalent on fewer than 6 banks; the joint EIG variant has the smallest "
       "absolute median difference and at least as many equivalent banks as any variant; every fully Bayesian coverage is "
       "below 0.80 and within 0.05 of EOST-M's; the re-analysed frozen arm reproduces the frozen C3")
def _():
    f = "artifacts/study2_posthoc/_analysis/decisions.json"
    if not os.path.exists(f):
        return None, "post hoc analysis missing"
    x = json.load(open(f)).get("fbfix")
    d = _s2("decisions.json")
    if x is None or d is None:
        return None, "missing"
    dr = x["location_drift_frozen_window"]
    others = [v for b, v in dr.items() if b != "ollb_hellaswag"]
    pols = ["fb:bmi (frozen)", "fb:eig (frozen)", "fb:bmi", "fb:eig", "fbj:bmi", "fbj:eig"]
    nosig = all(x[p][k] == 0 for p in pols for k in ("eost_better_sig", "fb_better_sig", "iae_eost_better_sig", "iae_fb_better_sig"))
    corr = all(0 < x[p]["median_rel_diff_pct"] < 1 and x[p]["n_equivalent"] < 6 for p in ("fb:bmi", "fb:eig"))
    joint = abs(x["fbj:eig"]["median_rel_diff_pct"]) == min(abs(x[p]["median_rel_diff_pct"]) for p in pols) \
        and x["fbj:eig"]["n_equivalent"] == max(x[p]["n_equivalent"] for p in pols)
    cov = all(x[p]["mean_coverage"] < 0.80 and abs(x[p]["mean_coverage"] - x[p]["eost_mean_coverage"]) < 0.05 for p in pols)
    repro = x["fb:bmi (frozen)"]["n_equivalent"] == d["decisions"]["C3"]["n_equivalent"] \
        and abs(x["fb:bmi (frozen)"]["median_rel_diff_pct"] - d["decisions"]["C3"]["median_rel_diff_pct"]) < 1e-9
    ok = max(dr, key=dr.get) == "ollb_hellaswag" and dr["ollb_hellaswag"] > 2 * float(np.median(others)) \
        and x["location_drift_corrected"]["ollb_hellaswag"] <= 0.5 * dr["ollb_hellaswag"] \
        and nosig and corr and joint and cov and repro
    return bool(ok), {"nosig": nosig, "corr": corr, "joint": joint, "cov": cov, "repro": repro,
                      "drift_hs": (dr["ollb_hellaswag"], x["location_drift_corrected"]["ollb_hellaswag"])}


@claim("s2-fb-n4", "Study 2 / Bayesian CATs at N = 4",
       "at N=4 the location drift is larger than at N=8 (median over banks, frozen window), and the corrected fully "
       "Bayesian coverage exceeds EOST-M's by less than 0.05")
def _():
    f = "artifacts/study2_posthoc/_analysis/decisions.json"
    if not os.path.exists(f):
        return None, "missing"
    d = json.load(open(f))
    if "fbfix_N4" not in d:
        return None, "N = 4 corrections missing"
    x4, x8 = d["fbfix_N4"], d["fbfix"]
    m4 = float(np.median(list(x4["location_drift_frozen_window"].values())))
    m8 = float(np.median(list(x8["location_drift_frozen_window"].values())))
    cov = x4["fb:bmi"]["mean_coverage"] - x4["fb:bmi"]["eost_mean_coverage"]
    return bool(m4 > m8 and 0 < cov < 0.05), {"drift4": m4, "drift8": m8, "cov_diff": cov}


@claim("rf1-clusters", "Robustness follow-ups / RF1",
       "the recomputed frozen clustering reproduces the frozen rows exactly; under every alternative panel-exclusion "
       "rule R2 and C1 keep their frozen medians (within 0.1 point), R2 is positive on 8/8 and C1 on 7/8 banks, none "
       "significantly against, and the rule fixed in advance is met; response-derived links change the clusters "
       "substantially only on the Open LLM Leaderboard banks")
def _():
    f = "artifacts/robust/clusters/_analysis/decisions.json"
    if not os.path.exists(f):
        return None, "RF1 analysis missing"
    d = json.load(open(f))
    ok = d.get("repro_max_abs_diff", 1.0) == 0.0 and d["rule_met_all_variants"] and len(d["variants"]) == 5
    for v, x in d["variants"].items():
        r2, c1 = x["decisions"]["R2"], x["decisions"]["C1"]
        ok &= abs(r2["median_rel_improvement_pct"] - r2["frozen_median_pct"]) < 0.1
        ok &= abs(c1["median_rel_improvement_pct"] - c1["frozen_median_pct"]) < 0.1
        ok &= r2["positive"] == 8 and c1["positive"] == 7 and r2["favour_base"] == 0 and c1["favour_base"] == 0
    cs = d["cluster_summary"]
    ok &= all(abs(cs["joint_0.5"][b]["n_clusters"] - cs["metadata"][b]["n_clusters"]) <= 2
              for b in ("swe_verified", "livecodebench", "cruxeval_o"))
    ok &= cs["metadata"]["ollb_gsm8k"]["n_clusters"] - cs["joint_0.5"]["ollb_gsm8k"]["n_clusters"] >= 20
    return bool(ok), {"rule": d["rule_met_all_variants"], "repro": d.get("repro_max_abs_diff")}


@claim("rf3-meta", "Robustness follow-ups / RF3",
       "an effect can favour EOST-M on all eight benchmarks and still be declared not supported; the hypotheses whose "
       "pooled effect excludes zero although not supported are exactly those that missed the threshold by one bank "
       "(4 significant, none against)")
def _():
    f = "artifacts/robust/meta/meta.csv"
    if not os.path.exists(f):
        return None, "RF3 missing"
    M = pd.read_csv(f).set_index("hypothesis")
    dec = _s2("decisions.json")["decisions"]
    unsup_excl = sorted(h for h in M.index if M.loc[h, "ci_lo"] > 0 and not M.loc[h, "frozen_supported"])
    missed = sorted(h for h, x in dec.items() if "favour_new" in x and not x["supported"] and x["favour_new"] == 4
                    and x["favour_base"] == 0)
    ok = any(int(M.loc[h, "n_positive"]) == 8 and not bool(M.loc[h, "frozen_supported"]) for h in M.index) \
        and unsup_excl == missed
    return bool(ok), {"unsupported_but_pooled_positive": unsup_excl, "missed_by_one": missed}


@claim("rf2-convergence", "Robustness follow-ups / RF2",
       "with the frozen sampler and four long chains the overall location converged (R-hat < 1.05 on at least 90% "
       "of panels) but sigma_b, mu_a and sigma_a did not (R-hat < 1.05 on at most 10% of panels), and the ability "
       "scale moved with sigma_b and mu_a (median |r| > 0.7, opposite signs)")
def _():
    f = "artifacts/robust/fb_convergence/_analysis/summary.json"
    if not os.path.exists(f):
        return None, "RF2 analysis missing"
    d = json.load(open(f))
    S = d["summary"]
    ok = d.get("complete", False) and S["panels"] == 48 and S["share_rhat_loc_lt_1.05"] >= 0.9
    ok &= all(S[f"share_rhat_{p}_lt_1.05"] <= 0.10 for p in ("sd_b", "mu_a", "sd_a"))
    ok &= S["cor_scale_sd_b_median"] > 0.7 and S["cor_scale_mu_a_median"] < -0.7
    return bool(ok), {k: round(S[k], 3) for k in ("share_rhat_loc_lt_1.05", "share_rhat_sd_b_lt_1.05",
                                                  "share_rhat_mu_a_lt_1.05", "share_rhat_sd_a_lt_1.05")}


@claim("rf2-rule", "Robustness follow-ups / RF2",
       "the pre-specified rule is not confirmed under the reading recorded before the comparison (median absolute "
       "change versus burn-in 60 above 0.5% for every long-chain arm) while the signed medians are within 0.5%; the "
       "long-chain arms stay within 1% of EOST-M in median held-out log loss with no benchmark sign test below 0.05")
def _():
    f = "artifacts/robust/fb_convergence/_analysis/summary.json"
    if not os.path.exists(f):
        return None, "RF2 analysis missing"
    d = json.load(open(f))
    c = d["comparison"]
    longs = ("pdm_bmi_long", "pdm_eig_long", "jfb_eig_long", "jfb_bmi_long")
    ok = (not d["rule_confirmed"]) and d["rule_confirmed_narrow_reading"]
    ok &= all(c[k]["median_abs_rel_ll_vs_burn60_pct"] > 0.5 and abs(c[k]["median_rel_ll_vs_burn60_pct"]) < 0.5
              and abs(c[k]["median_rel_ll_vs_eost_pct"]) < 1.0 and c[k]["bank_sign_test_p"] >= 0.05 for k in longs)
    return bool(ok), {k: round(c[k]["median_rel_ll_vs_eost_pct"], 2) for k in longs}


@claim("rf2b-partial", "Robustness follow-ups / RF2b",
       "the scale move brings R-hat below 1.05 for the ability scale, sigma_b and mu_a on most panels (at least 70%, "
       "against at most 2% in RF2) but not for sigma_a (at most 20%), so the convergence criterion is not met; the "
       "between-chain spread of held-out log loss falls to at most 10% (RF2: above 40%)")
def _():
    f, g = "artifacts/robust/fb_scale/_analysis/summary.json", "artifacts/robust/fb_convergence/_analysis/summary.json"
    if not (os.path.exists(f) and os.path.exists(g)):
        return None, "RF2b analysis missing"
    d, r2 = json.load(open(f)), json.load(open(g))
    S, S2 = d["summary"], r2["summary"]
    ok = d["complete"] and not d["converged"] and S["panels"] == 48
    ok &= all(S[f"share_rhat_{p}_lt_1.05"] >= 0.70 and S2[f"share_rhat_{p}_lt_1.05"] <= 0.021
              for p in ("scale", "sd_b", "mu_a"))
    ok &= S["share_rhat_sd_a_lt_1.05"] <= 0.20
    ok &= S["ll_spread_rel_max"] <= 0.10 and S2["ll_spread_rel_max"] > 0.40
    return bool(ok), {p: round(S[f"share_rhat_{p}_lt_1.05"], 3) for p in ("scale", "sd_b", "mu_a", "sd_a")}


@claim("rf2d-near", "Robustness follow-ups / RF2c-RF2d",
       "RF2c was stopped after 14 panels with the location stuck on HellaSwag (R-hat > 2) while GSM8K converged; "
       "RF2d's median R-hat is at most 1.06 for every quantity and its largest R-hat is well below RF2's, no chain "
       "stays at the lower bound of sigma_a, yet the 90% convergence criterion is not met")
def _():
    f, g = "artifacts/robust/fb_exact2/_analysis/summary.json", "artifacts/robust/fb_convergence/_analysis/summary.json"
    h = "artifacts/robust/fb_exact/_analysis/partial.json"
    if not all(os.path.exists(x) for x in (f, g, h)):
        return None, "RF2c/RF2d analysis missing"
    d, r2, pc = json.load(open(f)), json.load(open(g)), json.load(open(h))
    S = d["summary"]
    P = ("loc", "scale", "mu_b", "sd_b", "mu_a", "sd_a")
    ok = d["complete"] and not d["converged"] and S["panels"] == 48
    ok &= max(S[f"rhat_{p}_median"] for p in P) <= 1.06
    ok &= max(S[f"rhat_{p}_max"] for p in P) < 0.5 * r2["summary"]["rhat_sd_a_max"]
    ok &= S["chains_at_sd_a_floor"] == 0 and min(S[f"share_rhat_{p}_lt_1.05"] for p in P) < 0.9
    ok &= pc["panels"] == 14 and pc["per_bank"]["ollb_hellaswag"]["min_rhat_loc"] > 2.0
    ok &= pc["per_bank"]["ollb_gsm8k"]["max_rhat_all"] < 1.06
    return bool(ok), {"median_rhat_max": round(max(S[f"rhat_{p}_median"] for p in P), 3),
                      "rhat_max": round(max(S[f"rhat_{p}_max"] for p in P), 3)}


@claim("rf2-all-samplers", "Robustness follow-ups / RF2-RF2d; Introduction; Conclusion",
       "across every Bayesian arm and sampler on the RF2 cells (frozen, burn-in 60, long chains, scale move, exact "
       "sampler) the median difference from EOST-M in held-out log loss stays below 0.5% and no benchmark-level sign "
       "test reaches 0.05; the RF2d arms cover as often as EOST-M (within 0.03) while RF2's long chains covered more; "
       "in score error the RF2d arms are ahead on more than half of the cells but on only 4-5 of 8 benchmarks")
def _():
    f, g, h = ("artifacts/robust/fb_exact2/_analysis/summary.json", "artifacts/robust/fb_convergence/_analysis/summary.json",
               "artifacts/robust/fb_scale/_analysis/summary.json")
    if not all(os.path.exists(x) for x in (f, g, h)):
        return None, "RF2 analyses missing"
    d, r2, rb = json.load(open(f)), json.load(open(g)), json.load(open(h))
    arms = [v for k, v in r2["comparison"].items() if k != "eost"]
    arms += [rb["comparison"][f"{a}_scale"] for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")]
    ex = [d["comparison"][f"{a}_exact"] for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")]
    arms += ex
    ok = len(arms) == 18 and all(abs(a["median_rel_ll_vs_eost_pct"]) < 0.5 and a["bank_sign_test_p"] >= 0.05 for a in arms)
    eost_cov = d["comparison"]["eost"]["mean_coverage"]
    ok &= all(abs(a["mean_coverage"] - eost_cov) <= 0.03 for a in ex)
    ok &= all(r2["comparison"][k]["mean_coverage"] >= eost_cov + 0.05 for k in ("pdm_bmi_long", "pdm_eig_long",
                                                                               "jfb_eig_long", "jfb_bmi_long"))
    ok &= all(a["median_rel_iae_vs_eost_pct"] < 0 and 4 <= 8 - a["banks_iae_worse_than_eost"] <= 5 for a in ex)
    return bool(ok), {"max_abs_median_pct": round(max(abs(a["median_rel_ll_vs_eost_pct"]) for a in arms), 3)}


@claim("eb-stability", "Study 2 / robustness of the empirical-Bayes hyperprior",
       "at N = 8 resampling items moves every hyperparameter little (item-bootstrap SD at most 0.10); resampling "
       "calibrators moves sigma_a and sigma_b more than resampling items, by about as much as the variation between "
       "panels (within a factor of 1.5)")
def _():
    f = "artifacts/eb_robustness/summary.json"
    if not os.path.exists(f):
        return None, "EB robustness missing"
    st = json.load(open(f))["stability"]["8"]
    it, cal, bp = st["item_bootstrap_sd"], st["calibrator_bootstrap_sd"], st["between_panel_sd"]
    ok = max(it.values()) <= 0.10
    ok &= all(cal[k] > it[k] and 1 / 1.5 <= cal[k] / bp[k] <= 1.5 for k in ("sd_log_a", "sd_b"))
    return bool(ok), {k: (round(it[k], 3), round(cal[k], 3), round(bp[k], 3)) for k in ("sd_log_a", "sd_b")}


@claim("rf2-mub-scale", "Robustness follow-ups / RF2 and RF2b",
       "with the frozen sampler mu_b converged on most panels (R-hat < 1.05 on at least 75%) and the ability scale on "
       "almost none (at most 5%); at least one chain of RF2 and one of RF2b spent most of their sweeps at the lower "
       "bound of sigma_a")
def _():
    f, g = "artifacts/robust/fb_convergence/_analysis/summary.json", "artifacts/robust/fb_scale/_analysis/summary.json"
    if not (os.path.exists(f) and os.path.exists(g)):
        return None, "RF2/RF2b analysis missing"
    S, Sb = json.load(open(f))["summary"], json.load(open(g))["summary"]
    ok = S["share_rhat_mu_b_lt_1.05"] >= 0.75 and S["share_rhat_scale_lt_1.05"] <= 0.05
    ok &= S["chains_at_sd_a_floor"] >= 1 and Sb["chains_at_sd_a_floor"] >= 1
    return bool(ok), {"mu_b": round(S["share_rhat_mu_b_lt_1.05"], 3), "scale": round(S["share_rhat_scale_lt_1.05"], 3),
                      "floor": [S["chains_at_sd_a_floor"], Sb["chains_at_sd_a_floor"]]}


@claim("rf2d-scale-spread", "Robustness follow-ups / RF2d",
       "under the exact sampler the posterior mean of the ability scale exceeds EOST-M's point estimate on more than "
       "half of the panels, with a median ratio above 1")
def _():
    f = "artifacts/robust/fb_exact2/_analysis/summary.json"
    if not os.path.exists(f):
        return None, "RF2d analysis missing"
    S = json.load(open(f))["summary"]
    ok = S["share_post_scale_above_eb"] > 0.5 and S["post_over_eb_scale_median"] > 1
    return bool(ok), {"share": round(S["share_post_scale_above_eb"], 3),
                      "median_ratio": round(S["post_over_eb_scale_median"], 3)}


@claim("rf3b-sources", "Robustness follow-ups / RF3b; Introduction; Discussion (xi); cover letter; README",
       "with the five Open LLM Leaderboard benchmarks counted as one source, fewer pooled intervals exclude zero than "
       "in RF3; the hypotheses that lose it stay positive in at least 3 of 4 sources; of the four hypotheses that "
       "missed the frozen threshold by one benchmark only C2 keeps an interval above zero; the sampling dependence "
       "of the OLLB estimates through shared systems is small (|median r| < 0.1)")
def _():
    f, g = "artifacts/robust/meta/sources.csv", "artifacts/robust/meta/meta.csv"
    if not (os.path.exists(f) and os.path.exists(g)):
        return None, "RF3b missing"
    S, M = pd.read_csv(f).set_index("hypothesis"), pd.read_csv(g).set_index("hypothesis")
    excl_src = {h for h in S.index if S.loc[h, "ci_lo"] > 0}
    excl_rf3 = {h for h in M.index if M.loc[h, "ci_lo"] > 0}
    lost = excl_rf3 - excl_src
    missed = {"C2", "C4", "C7", "C8"}
    ok = len(excl_src) < len(excl_rf3) and excl_src <= excl_rf3 and len(lost) > 0
    ok &= all(int(S.loc[h, "n_positive_sources"]) >= 3 for h in lost)
    ok &= (excl_src & missed) == {"C2"}
    ok &= abs(float(S.ollb_median_pairwise_r.median())) < 0.1 and (S.k_sources == 4).all()
    return bool(ok), {"excl_sources": sorted(excl_src), "lost": sorted(lost)}


@claim("s2-abstract", "Abstract / Introduction",
       "EOST-M is ahead of ATLAS-style CAT, tinyBenchmarks and MINCE on every benchmark, significantly on four or five "
       "of eight; under the fully Bayesian model, BMI and EIG selection differ by less than 1% in log loss with no bank "
       "significant (N = 4 and 8): the criterion matters little")
def _():
    d = _s2("decisions.json")
    if d is None:
        return None, "missing"
    dec = d["decisions"]
    red = all(dec[h]["positive"] == 8 and dec[h]["favour_new"] in (4, 5) and dec[h]["favour_base"] == 0
              for h in ("C7", "C8", "C9"))
    crit = all(abs(_sec("S:fb:bmi|fb:eig|ilogloss", N).rel_improvement.median()) < 0.01
               and int((_sec("S:fb:bmi|fb:eig|ilogloss", N).p_holm < 0.05).sum()) == 0 for N in (4, 8))
    return bool(red and crit), {"reduction": red, "criterion": crit}


def main():
    res, n_fail = [], 0
    for cid, where, text, fn in CLAIMS:
        try:
            ok, ev = fn()
        except Exception as e:  # noqa: BLE001 -- a claim that cannot be evaluated fails the gate
            ok, ev = None, f"error: {type(e).__name__}: {e}"
        status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        n_fail += status != "PASS"
        res.append({"id": cid, "where": where, "claim": text, "status": status, "evidence": ev})
        print(f"[{status}] {cid}: {text}")
        if status != "PASS":
            print("        ", json.dumps(ev, default=float)[:600])
    with open("artifacts/CLAIMS.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"CLAIMS_GATE={'PASS' if n_fail == 0 else 'FAIL'} ({len(res) - n_fail}/{len(res)} claims hold)")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
