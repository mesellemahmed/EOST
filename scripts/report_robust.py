"""Tables, figure and macros of the robustness follow-ups RF1-RF3 (after scripts/analyze_robust.py
and scripts/meta_analysis.py). Writes paper/tables/robust_*.tex, paper/figures/rf2_traces.pdf and
paper/generated/numbers3.tex. Every block is skipped if its artifacts are missing.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG, TAB, GEN = "paper/figures", "paper/tables", "paper/generated"
NICE = {"ollb_gsm8k": "GSM8K", "ollb_arc": "ARC-C", "ollb_hellaswag": "HellaSwag", "ollb_winogrande": "WinoGrande",
        "ollb_mmlu": "MMLU", "swe_verified": "SWE-bench V.", "livecodebench": "LiveCodeBench", "cruxeval_o": "CRUXEval-O"}
CONF = list(NICE)
VNAME = {"metadata": "metadata only (no responses)", "lbo": "leave-bank-out residuals",
         "joint_0.4": "joint residuals, $r\\ge0.4$", "joint_0.6": "joint residuals, $r\\ge0.6$",
         "joint_0.7": "joint residuals, $r\\ge0.7$"}
VSHORT = {"joint_0.5": "frozen, $r\\ge0.5$", "metadata": "metadata only", "lbo": "leave-bank-out",
          "joint_0.4": "joint, $r\\ge0.4$", "joint_0.6": "joint, $r\\ge0.6$", "joint_0.7": "joint, $r\\ge0.7$"}
ABBR = {"ollb_gsm8k": "GSM8K", "ollb_arc": "ARC", "ollb_hellaswag": "HS", "ollb_winogrande": "WG", "ollb_mmlu": "MMLU",
        "swe_verified": "SWE-V", "livecodebench": "LCB", "cruxeval_o": "CRUX-O"}
VTAG = {"metadata": "Meta", "lbo": "Lbo", "joint_0.4": "Four", "joint_0.6": "Six", "joint_0.7": "Seven"}
NUM = {}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def macro(name, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    if isinstance(value, str):
        NUM[name] = value
    else:
        t = fmt.format(value)
        NUM[name] = ("$-$" + t[1:]) if t.startswith("-") else t


def signed(v, fmt="{:+.1f}"):
    t = fmt.format(v)
    if float(t) == 0:
        t = t.lstrip("+-")
    return "$" + t + "$"


# ----------------------------------------------------------------------------- RF1
def rf1():
    f = "artifacts/robust/clusters/_analysis/decisions.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    if "repro_max_abs_diff" in d:
        macro("RobReproRows", d["repro_rows"], "{:,d}")
        macro("RobReproDiff", d["repro_max_abs_diff"], "{:.0f}" if d["repro_max_abs_diff"] == 0 else "{:.1e}")
    if not d["variants"]:
        return
    rows = []
    fr = None
    for v in ["metadata", "lbo", "joint_0.4", "joint_0.6", "joint_0.7"]:
        if v not in d["variants"]:
            continue
        x = d["variants"][v]
        r2, c1, c4 = x["decisions"]["R2"], x["decisions"]["C1"], x["decisions"]["C4"]
        fr = (r2, c1, c4)
        drop = sum(x["targets_dropped"].values()) if x["targets_dropped"] else 0
        rows.append(f"{VNAME[v]} & {x['n_targets']} & {signed(r2['median_rel_improvement_pct'])} & {r2['positive']}/{r2['n_banks']} & "
                    f"{r2['favour_new']}/{r2['favour_base']} & {signed(c1['median_rel_improvement_pct'])} & {c1['positive']}/{c1['n_banks']} & "
                    f"{c1['favour_new']}/{c1['favour_base']} & {signed(c4['median_rel_improvement_pct'])} & {c4['favour_new']}/{c4['favour_base']} \\\\")
        t = VTAG[v]
        for h, s in [("RTwo", r2), ("COne", c1), ("CFour", c4)]:
            macro(f"Rob{t}{h}Med", s["median_rel_improvement_pct"], "{:.1f}")
            macro(f"Rob{t}{h}Fav", s["favour_new"], "{:d}")
            macro(f"Rob{t}{h}Pos", s["positive"], "{:d}")
        macro(f"Rob{t}Targets", x["n_targets"], "{:d}")
        macro(f"Rob{t}Dropped", drop, "{:d}")
    if fr is not None:
        r2, c1, c4 = fr
        fd = json.load(open("artifacts/study2/_analysis/decisions.json"))["decisions"]
        frozen_row = (f"frozen (joint residuals, $r\\ge0.5$) & {sum(len(json.load(open(f"artifacts/study2/main/{b}/config.json"))["targets"]) for b in CONF)} & "
                      f"{signed(fd['R2']['median_rel_improvement_pct'])} & {fd['R2']['positive']}/{fd['R2']['n_banks']} & "
                      f"{fd['R2']['favour_new']}/{fd['R2']['favour_base']} & {signed(fd['C1']['median_rel_improvement_pct'])} & "
                      f"{fd['C1']['positive']}/{fd['C1']['n_banks']} & {fd['C1']['favour_new']}/{fd['C1']['favour_base']} & "
                      f"{signed(fd['C4']['median_rel_improvement_pct'])} & {fd['C4']['favour_new']}/{fd['C4']['favour_base']} \\\\")
        rows = [frozen_row, r"\midrule"] + rows
    allv = [x for x in d["variants"].values()]
    macro("RobRTwoMedMin", min(x["decisions"]["R2"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobRTwoMedMax", max(x["decisions"]["R2"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobCOneMedMin", min(x["decisions"]["C1"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobCOneMedMax", max(x["decisions"]["C1"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobRTwoFavMin", min(x["decisions"]["R2"]["favour_new"] for x in allv), "{:d}")
    macro("RobRTwoFavMax", max(x["decisions"]["R2"]["favour_new"] for x in allv), "{:d}")
    macro("RobCOneFavMin", min(x["decisions"]["C1"]["favour_new"] for x in allv), "{:d}")
    macro("RobCOneFavMax", max(x["decisions"]["C1"]["favour_new"] for x in allv), "{:d}")
    macro("RobCFourMedMin", min(x["decisions"]["C4"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobCFourMedMax", max(x["decisions"]["C4"]["median_rel_improvement_pct"] for x in allv), "{:.1f}")
    macro("RobCFourFavMin", min(x["decisions"]["C4"]["favour_new"] for x in allv), "{:d}")
    macro("RobCFourFavMax", max(x["decisions"]["C4"]["favour_new"] for x in allv), "{:d}")
    for h, nm in [("R2", "RTwo"), ("C1", "COne"), ("C4", "CFour")]:
        lo = min(x["decisions"][h]["favour_new"] for x in allv)
        hi = max(x["decisions"][h]["favour_new"] for x in allv)
        macro(f"Rob{nm}FavRange", f"{lo}" if lo == hi else f"{lo}--{hi}")
    macro("RobRuleMet", "met" if d["rule_met_all_variants"] else "not met")
    macro("RobNVariants", len(allv), "{:d}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Robustness of study 2 to the clustering (follow-up RF1, planned before running). Frozen targets, holdouts and panel seeds; only the systems excluded from each target's panel change. For each exclusion rule: targets used, and for R2 (EOST-M vs plug-in Fisher CAT, log loss), C1 (vs the Rasch-type plug-in test, log loss) and C4 (vs Random:mg, score IAE): median relative improvement of EOST-M over benchmarks (\%), benchmarks with a positive difference, and benchmarks Holm-significant for / against EOST-M (frozen 88-test family with the rule's $p$-values substituted).}
\label{tab:robclust}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{C1} & \multicolumn{2}{c}{C4} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}
Panel exclusion rule & targets & median & pos. & sig. & median & pos. & sig. & median & sig. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/robust_clusters.tex", "w").write(tex)
    # cluster counts per rule
    cs = d["cluster_summary"]
    crow = []
    for v in ["joint_0.5", "metadata", "lbo", "joint_0.4", "joint_0.6", "joint_0.7"]:
        crow.append(VSHORT[v] + " & " +
                    " & ".join(f"{cs[v][b]['n_clusters']}/{cs[v][b]['max_size']}" for b in ["ollb_gsm8k", "swe_verified",
                                                                                           "livecodebench", "cruxeval_o"]) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Clusters under each panel-exclusion rule of follow-up RF1: number of clusters / size of the largest cluster. The five Open LLM Leaderboard banks share their systems and their joint clusters; leave-bank-out clusters differ per leaderboard bank (GSM8K shown).}
\label{tab:robclustcount}
\begin{tabular}{@{}lcccc@{}}
\toprule
Rule & Open LLM LB (GSM8K) & SWE-bench V. & LCB & CRUX-O \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/robust_cluster_counts.tex", "w").write(tex)
    T = pd.read_csv("artifacts/robust/clusters/_analysis/tests.csv")
    det = []
    for v in ["metadata", "lbo", "joint_0.4", "joint_0.6", "joint_0.7"]:
        for h in ["R2", "C1"]:
            s = T[(T.variant == v) & (T.hypothesis == h)].set_index("dataset")
            if s.empty:
                continue
            cells = []
            for b in CONF:
                if b not in s.index:
                    cells.append("--"); continue
                r = s.loc[b]
                cells.append(f"{signed(100 * r.rel_improvement)}{'*' if r.p_holm < 0.05 else ''}")
            det.append(f"{VSHORT[v]} & {h} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{Follow-up RF1 per benchmark: relative improvement (\%) of EOST-M in held-out log loss over plug-in Fisher CAT (R2) and over the Rasch-type plug-in test (C1) under each panel-exclusion rule; * Holm-significant (frozen 88-test family with the rule's $p$-values substituted). HS: HellaSwag; WG: WinoGrande; LCB: LiveCodeBench.}
\label{tab:robclustbanks}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
Rule & & """ + " & ".join(ABBR[b] for b in CONF) + r""" \\
\midrule
""" + "\n".join(det) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/robust_clusters_banks.tex", "w").write(tex)


# ----------------------------------------------------------------------------- RF2
PNAME = [("loc", "Loc", "location"), ("mu_b", "MuB", r"$\mu_b$"), ("sd_b", "SdB", r"$\sigma_b$"),
         ("mu_a", "MuA", r"$\mu_a$"), ("sd_a", "SdA", r"$\sigma_a$"), ("scale", "Scale", "scale")]
ARMS = [("pdm_bmi_frozen", "PdmBmiFrozen", "PDM, BMI", "frozen: 1 chain, burn-in 10, 8 draws"),
        ("pdm_eig_frozen", "PdmEigFrozen", "PDM, EIG", "frozen: 1 chain, burn-in 10, 8 draws"),
        ("pdm_bmi_60", "PdmBmiSixty", "PDM, BMI", "corrected: 1 chain, burn-in 60, 8 draws"),
        ("pdm_eig_60", "PdmEigSixty", "PDM, EIG", "corrected: 1 chain, burn-in 60, 8 draws"),
        ("jfb_eig_60", "JfbEigSixty", "joint, EIG", "corrected: 1 chain, burn-in 60, 8 draws"),
        ("jfb_bmi_60", "JfbBmiSixty", "joint, BMI", "corrected: 1 chain, burn-in 60, 8 draws"),
        ("pdm_bmi_long", "PdmBmiLong", "PDM, BMI", "long: 4 chains, burn-in 250, 400 draws"),
        ("pdm_eig_long", "PdmEigLong", "PDM, EIG", "long: 4 chains, burn-in 250, 400 draws"),
        ("jfb_eig_long", "JfbEigLong", "joint, EIG", "long: 4 chains, burn-in 250, 40 of 400 draws"),
        ("jfb_bmi_long", "JfbBmiLong", "joint, BMI", "long: 4 chains, burn-in 250, 40 of 400 draws")]
LONG = ["pdm_bmi_long", "pdm_eig_long", "jfb_eig_long", "jfb_bmi_long"]


def rf2():
    f = "artifacts/robust/fb_convergence/_analysis/summary.json"
    if not os.path.exists(f):
        return
    s = json.load(open(f))
    S, comp = s["summary"], s["comparison"]
    macro("ConvPanels", S["panels"], "{:d}")
    macro("ConvBanks", S["banks"], "{:d}")
    macro("ConvCells", s["n_cells"], "{:d}")
    for p, nm, _ in PNAME:
        macro(f"Conv{nm}RhatOne", 100 * S[f"share_rhat_{p}_lt_1.01"], "{:.0f}")
        macro(f"Conv{nm}RhatFive", 100 * S[f"share_rhat_{p}_lt_1.05"], "{:.0f}")
        macro(f"Conv{nm}EssHundred", 100 * S[f"share_ess_{p}_ge_100"], "{:.0f}")
        macro(f"Conv{nm}RhatMax", S[f"rhat_{p}_max"], "{:.2f}")
        macro(f"Conv{nm}RhatMed", S[f"rhat_{p}_median"], "{:.2f}")
        macro(f"Conv{nm}EssMed", S[f"ess_{p}_median"], "{:.0f}")
        macro(f"Conv{nm}EssMin", S[f"ess_{p}_min"], "{:.0f}")
        macro(f"Conv{nm}McseRel", 100 * S[f"mcse_rel_{p}_median"], "{:.0f}")
    hyp = ["sd_b", "mu_a", "sd_a"]
    macro("ConvHypRhatFiveMax", 100 * max(S[f"share_rhat_{p}_lt_1.05"] for p in hyp), "{:.0f}")
    macro("ConvHypEssHundredMax", 100 * max(S[f"share_ess_{p}_ge_100"] for p in hyp), "{:.0f}")
    macro("ConvHypRhatMedMin", min(S[f"rhat_{p}_median"] for p in hyp), "{:.2f}")
    macro("ConvHypRhatMedMax", max(S[f"rhat_{p}_median"] for p in hyp), "{:.2f}")
    macro("ConvHypEssMedMin", min(S[f"ess_{p}_median"] for p in hyp), "{:.0f}")
    macro("ConvHypEssMedMax", max(S[f"ess_{p}_median"] for p in hyp), "{:.0f}")
    macro("ConvHypRhatMax", max(S[f"rhat_{p}_max"] for p in hyp), "{:.2f}")
    macro("ConvHypMcseMin", min(100 * S[f"mcse_rel_{p}_median"] for p in hyp), "{:.0f}")
    macro("ConvHypMcseMax", max(100 * S[f"mcse_rel_{p}_median"] for p in hyp), "{:.0f}")
    macro("ConvCorScaleSdB", S["cor_scale_sd_b_median"], "{:.2f}")
    macro("ConvCorScaleMuA", S["cor_scale_mu_a_median"], "{:.2f}")
    macro("ConvScaleRatioMed", S["scale_chain_ratio_median"], "{:.2f}")
    macro("ConvScaleRatioMax", S["scale_chain_ratio_max"], "{:.2f}")
    macro("ConvLLSpreadMed", 100 * S["ll_spread_rel_median"], "{:.1f}")
    macro("ConvLLSpreadMax", 100 * S["ll_spread_rel_max"], "{:.0f}")
    macro("ConvLLSpreadGtFive", 100 * S["share_ll_spread_rel_gt_5pct"], "{:.0f}")
    macro("ConvFloorChains", S["chains_at_sd_a_floor"], "{:d}")
    macro("ConvChains", S["chains_total"], "{:d}")
    for k, nm, _, _ in ARMS:
        c = comp[k]
        macro(f"Conv{nm}VsEost", c["median_rel_ll_vs_eost_pct"], "{:.2f}")
        macro(f"Conv{nm}Cov", c["mean_coverage"])
        macro(f"Conv{nm}BanksBetter", c["banks_better_than_eost"], "{:d}")
        macro(f"Conv{nm}BanksWorse", c["banks_worse_than_eost"], "{:d}")
        macro(f"Conv{nm}WithinOne", c["banks_within_1pct"], "{:d}")
        if "median_rel_ll_vs_burn60_pct" in c:
            macro(f"Conv{nm}VsSixty", c["median_rel_ll_vs_burn60_pct"], "{:.2f}")
            macro(f"Conv{nm}AbsVsSixty", c["median_abs_rel_ll_vs_burn60_pct"], "{:.2f}")
            macro(f"Conv{nm}ShareGtOne", 100 * c["share_cells_abs_vs_burn60_gt_1pct"], "{:.0f}")
    lv = [comp[k]["median_rel_ll_vs_eost_pct"] for k in LONG]
    macro("ConvLongVsEostMin", min(lv), "{:.2f}")
    macro("ConvLongVsEostMax", max(lv), "{:.2f}")
    la = [comp[k]["median_abs_rel_ll_vs_burn60_pct"] for k in LONG]
    macro("ConvLongAbsVsSixtyMin", min(la), "{:.2f}")
    macro("ConvLongAbsVsSixtyMax", max(la), "{:.2f}")
    ls = [comp[k]["median_rel_ll_vs_burn60_pct"] for k in LONG]
    macro("ConvLongVsSixtyMin", min(ls), "{:.2f}")
    macro("ConvLongVsSixtyMax", max(ls), "{:.2f}")
    lc = [comp[k]["mean_coverage"] for k in LONG]
    macro("ConvLongCovMin", min(lc))
    macro("ConvLongCovMax", max(lc))
    lo, hi = f"{min(lc):.2f}", f"{max(lc):.2f}"
    macro("ConvLongCovRange", lo if lo == hi else f"{lo}--{hi}")   # one value when both ends print alike
    macro("ConvEostCov", comp["eost"]["mean_coverage"])
    macro("ConvRule", "confirmed" if s["rule_confirmed"] else "not confirmed")
    macro("ConvRuleNarrow", "confirmed" if s["rule_confirmed_narrow_reading"] else "not confirmed")
    # ---- tables and figures, with the RF2b block when available
    fb = "artifacts/robust/fb_scale/_analysis/summary.json"
    sb = json.load(open(fb)) if os.path.exists(fb) else None
    fc = "artifacts/robust/fb_exact2/_analysis/summary.json"
    sc = json.load(open(fc)) if os.path.exists(fc) else None
    conv_table(s, sb, sc)
    arms_table(s, sb, sc)
    trace_figure("fb_convergence", "rf2_traces.pdf")
    if sb is not None:
        trace_figure("fb_scale", "rf2b_traces.pdf")
        rf2x_macros(sb, "Scl", "scale")
    if sc is not None:
        trace_figure("fb_exact2", "rf2d_traces.pdf")
        rf2x_macros(sc, "Ext", "exact")
    # all Bayesian-CAT variants on the RF2 cells: largest |median difference| to EOST-M in held-out log loss
    allv = [comp[k]["median_rel_ll_vs_eost_pct"] for k, _, _, _ in ARMS]
    allc = [comp[k]["mean_coverage"] for k, _, _, _ in ARMS]
    for extra, tag in ((sb, "scale"), (sc, "exact")):
        if extra is not None:
            allv += [extra["comparison"][f"{a}_{tag}"]["median_rel_ll_vs_eost_pct"] for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")]
            allc += [extra["comparison"][f"{a}_{tag}"]["mean_coverage"] for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")]
    macro("AllBayesVsEostMaxAbs", max(abs(v) for v in allv), "{:.2f}")
    macro("AllBayesNArms", len(allv), "{:d}")
    macro("AllBayesCovMin", min(allc))
    macro("AllBayesCovMax", max(allc))
    if sb is not None:
        Sb = sb["summary"]
        rid = [100 * Sb[f"share_rhat_{p}_lt_1.05"] for p in ("scale", "sd_b", "mu_a")]
        macro("SclRidgeRhatFiveMin", min(rid), "{:.0f}")
        macro("SclRidgeRhatFiveMax", max(rid), "{:.0f}")
    fp = "artifacts/robust/fb_exact/_analysis/partial.json"
    if os.path.exists(fp):
        pp = json.load(open(fp))
        macro("RfcPanels", pp["panels"], "{:d}")
        pb = pp["per_bank"]
        if "ollb_gsm8k" in pb:
            macro("RfcGsmRhatMax", pb["ollb_gsm8k"]["max_rhat_all"], "{:.2f}")
        macro("RfcLocRhatMax", max(v["max_rhat_loc"] for v in pb.values()), "{:.1f}")
        if "ollb_arc" in pb:
            macro("RfcArcLocMin", pb["ollb_arc"]["min_rhat_loc"], "{:.2f}")
            macro("RfcArcLocMax", pb["ollb_arc"]["max_rhat_loc"], "{:.2f}")


def _conv_block(S, C, pnames):
    rows = []
    for b in CONF:
        if b not in C.index:
            continue
        r = C.loc[b]
        cells = " & ".join(f"{int(r[f'n_rhat_{p}_lt_1.01'])}/{int(r[f'n_rhat_{p}_lt_1.05'])}/{int(r[f'n_ess_{p}_ge_100'])}"
                           for p in pnames)
        rows.append(f"{NICE[b]} & {cells} & {100 * r.ll_spread_rel_median:.1f} & {100 * r.ll_spread_rel_max:.1f} \\\\")
    share = " & ".join(f"{100 * S[f'share_rhat_{p}_lt_1.01']:.0f}/{100 * S[f'share_rhat_{p}_lt_1.05']:.0f}/"
                       f"{100 * S[f'share_ess_{p}_ge_100']:.0f}" for p in pnames)
    med_r = " & ".join(f"{S[f'rhat_{p}_median']:.2f} ({S[f'rhat_{p}_max']:.2f})" for p in pnames)
    med_e = " & ".join(f"{S[f'ess_{p}_median']:.0f} ({S[f'ess_{p}_min']:.0f})" for p in pnames)
    mcse = " & ".join((f"{100 * S[f'mcse_rel_{p}_median']:.0f}" if f"mcse_rel_{p}_median" in S else "--") for p in pnames)
    return "\n".join(rows) + r"""
\addlinespace
all panels (\%) & """ + share + r""" & """ + f"{100 * S['ll_spread_rel_median']:.1f} & {100 * S['ll_spread_rel_max']:.1f}" + r""" \\
$\hat R$, median (max) & """ + med_r + r""" & & \\
bulk ESS, median (min) & """ + med_e + r""" & & \\
MCSE / sd (\%) & """ + mcse + r""" & & \\"""


def conv_table(s, sb, sc=None):
    """RF2 convergence table, and a second table with the exploratory samplers RF2b and RF2d (one table with
    all three blocks does not fit on a page)."""
    pnames = [p for p, _, _ in PNAME]
    head = " & ".join(lab for _, _, lab in PNAME)
    C = pd.read_csv("artifacts/robust/fb_convergence/_analysis/convergence.csv").set_index("dataset")
    colhead = r"""\toprule
 & \multicolumn{6}{c}{panels with $\hat R<1.01$ / $\hat R<1.05$ / ESS $\ge100$} & \multicolumn{2}{c}{chain spread (\%)} \\
\cmidrule(lr){2-7}\cmidrule(l){8-9}
Benchmark & """ + head + r""" & median & max \\
\midrule
"""
    body = _conv_block(s["summary"], C, pnames)
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{MCMC convergence of the Bayesian CATs' sampler (follow-up RF2: the frozen sampler with long chains; 6 targets per benchmark, replicate 0, $N=8$; four chains, one from EOST-M's estimates and three from dispersed starts; burn-in 250, then 300 sweeps per chain with every third retained, 400 retained draws). Per benchmark and parameter: number of the six panels with rank-normalized split $\hat R<1.01$ / $\hat R<1.05$ / bulk ESS $\ge100$. Location: mean calibrator ability; scale: standard deviation of the calibrators' abilities (explanatory, not in the plan of RF2); $\mu_a,\sigma_a$: mean and standard deviation of the log discriminations, $\mu_b,\sigma_b$: of the difficulties. Last two columns: median and largest relative spread (\%) of the held-out log loss of the PDM Bayesian CAT (BMI) across the four chains. Summary rows over all panels: percentages; median (maximum) $\hat R$; median (minimum) bulk ESS; median Monte Carlo standard error as a percentage of the posterior standard deviation.}
\label{tab:rf2}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
""" + colhead + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    blocks, caps = [], []
    if sb is not None:
        Cb = pd.read_csv("artifacts/robust/fb_scale/_analysis/convergence.csv").set_index("dataset")
        blocks.append(r"\multicolumn{9}{@{}l}{\emph{RF2b: with the scale move, 4 chains, burn-in 200, 200 further sweeps per chain}} \\"
                      + "\n" + _conv_block(sb["summary"], Cb, pnames))
        caps.append(r"Upper block (RF2b): a Metropolis--Hastings move along the scale ridge added to every sweep of the frozen sampler.")
    if sc is not None:
        Cc = pd.read_csv("artifacts/robust/fb_exact2/_analysis/convergence.csv").set_index("dataset")
        blocks.append(r"\multicolumn{9}{@{}l}{\emph{RF2d: exact sampler of Lemma~\ref{lem:exact}'s model, 4 chains, burn-in 150, 200 further sweeps per chain}} \\"
                      + "\n" + _conv_block(sc["summary"], Cc, pnames))
        caps.append(r"Lower block (RF2d): the model of Lemma~\ref{lem:exact} (item prior normalized over the grid) sampled with the hyperparameters, the scale and the location moved by Metropolis--Hastings steps with the item parameters summed out.")
    if blocks:
        tex += r"""
\begin{table}[tbp]
\centering\scriptsize
\caption{MCMC convergence of the exploratory samplers (follow-ups RF2b and RF2d; same panels, starting points and columns as Table~\ref{tab:rf2}; 400 retained draws). """ + " ".join(caps) + r"""}
\label{tab:rf2x}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
""" + colhead + ("\n" + r"\midrule" + "\n").join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/robust_mcmc.tex", "w").write(tex)


def arms_table(s, sb, sc=None):
    comp = s["comparison"]
    rows = [r"EOST-M (empirical Bayes, no chain) & -- & -- & -- & -- & -- & " + f"{comp['eost']['mean_coverage']:.2f} \\\\"]

    def line(c, lab, ref):
        v = (f"{signed(c[f'median_rel_ll_vs_{ref}_pct'], '{:+.2f}')} & {c[f'median_abs_rel_ll_vs_{ref}_pct']:.2f}"
             if ref and f"median_rel_ll_vs_{ref}_pct" in c else "-- & --")
        return (f"\\quad {lab} & {signed(c['median_rel_ll_vs_eost_pct'], '{:+.2f}')} & "
                f"{c['banks_better_than_eost']}/{c['banks_worse_than_eost']} & {c['banks_within_1pct']} & {v} & "
                f"{c['mean_coverage']:.2f} \\\\")

    def head(text):
        return r"\addlinespace" + "\n" + r"\multicolumn{7}{@{}l}{\emph{" + text + r"}} \\"
    groups = [("frozen arm of study 2: 1 chain, burn-in 10, 8 draws", ["pdm_bmi_frozen", "pdm_eig_frozen"], None),
              ("post hoc correction: 1 chain, burn-in 60, 8 draws", ["pdm_bmi_60", "pdm_eig_60", "jfb_eig_60", "jfb_bmi_60"], None),
              ("RF2, long chains: 4 chains, burn-in 250, 400 draws (joint: 40)",
               ["pdm_bmi_long", "pdm_eig_long", "jfb_eig_long", "jfb_bmi_long"], "burn60")]
    lab = {"pdm_bmi": "PDM, BMI", "pdm_eig": "PDM, EIG", "jfb_eig": "joint, EIG", "jfb_bmi": "joint, BMI"}
    for text, keys, ref in groups:
        rows.append(head(text))
        for k in keys:
            rows.append(line(comp[k], lab[k.rsplit("_", 1)[0]], ref))
    cap_b = ""
    if sb is not None:
        rows.append(head("RF2b, scale move: 4 chains, burn-in 200, 400 draws (joint: 40)"))
        for k in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi"):
            rows.append(line(sb["comparison"][f"{k}_scale"], lab[k], "60"))
        cap_b = (r" Exploratory follow-ups RF2b (sampler with the scale move) and RF2d (exact sampler of the model of"
                 r" Lemma~\ref{lem:exact}) are compared with the same arm with burn-in 60.")
    if sc is not None:
        rows.append(head(r"RF2d, exact sampler: 4 chains, burn-in 150, 400 draws (joint: 40)"))
        for k in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi"):
            rows.append(line(sc["comparison"][f"{k}_exact"], lab[k], "60"))
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{Bayesian CATs with frozen, corrected and long chains against EOST-M on the same """ + f"{s['n_cells']}" + r""" cells (follow-up RF2; 6 targets per benchmark, replicate 0, $N=8$). Median over cells of the relative difference in held-out log loss to EOST-M (\%; positive: worse than EOST-M); benchmarks whose median favours the arm / EOST-M; benchmarks with a median within $\pm1\%$; for the long chains, median signed and median absolute relative difference (\%) to the same arm with burn-in 60; mean coverage of the 90\% intervals.""" + cap_b + r""" PDM: posterior-draw marginal; joint: joint fully Bayesian (mixture over 40 of the retained draws); BMI: Bayesian maximum information; EIG: expected information gain.}
\label{tab:rf2arms}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{log loss vs EOST-M} & \multicolumn{2}{c}{vs burn-in 60 (\%)} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Arm & median (\%) & banks & $\pm1\%$ & signed & abs. & coverage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/robust_mcmc_arms.tex", "w").write(tex)


def trace_figure(sub, fname):
    banks = [b for b in CONF if glob.glob(f"artifacts/robust/{sub}/{b}/parts/traces_*.npz")]
    if not banks:
        return
    fig, ax = plt.subplots(len(banks), 4, figsize=(6.9, 0.95 * len(banks) + 0.3), sharex=True)
    ax = np.atleast_2d(ax)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6f6e69"]
    for i, b in enumerate(banks):
        T = np.load(sorted(glob.glob(f"artifacts/robust/{sub}/{b}/parts/traces_*.npz"))[0])["T"]
        for j, (col, lab) in enumerate([(0, "location"), (1, "ability scale"), (5, r"$\sigma_a$"), (3, r"$\sigma_b$")]):
            for c in range(T.shape[0]):
                ax[i, j].plot(T[c, :, col], lw=0.5, color=cols[c % 4], label=f"chain {c + 1}")
            if i == 0:
                ax[i, j].set_title(lab, fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel(NICE[b], fontsize=7)
            ax[i, j].tick_params(labelsize=6)
    for j in range(4):
        ax[-1, j].set_xlabel("sweep after burn-in", fontsize=7)
    h, l_ = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(h_pad=0.3, w_pad=0.6, rect=(0, 0.025, 1, 1))
    fig.savefig(f"{FIG}/{fname}")
    plt.close(fig)


def rf2x_macros(sb, P="Scl", tag="scale"):
    S, comp = sb["summary"], sb["comparison"]
    macro(P + "Panels", S["panels"], "{:d}")
    macro(P + "Banks", S["banks"], "{:d}")
    macro(P + "AccMed", 100 * S["acc_median"], "{:.0f}")
    macro(P + "AccMin", 100 * S["acc_min"], "{:.0f}")
    macro(P + "AccMax", 100 * S["acc_max"], "{:.0f}")
    for k, nm in (("acc_mu_a", "AccMuA"), ("acc_sd_a", "AccSdA"), ("acc_mu_b", "AccMuB"), ("acc_sd_b", "AccSdB"),
                  ("acc_trans", "AccTrans")):
        if f"{k}_median" in S:
            macro(P + nm, 100 * S[f"{k}_median"], "{:.0f}")
    for p, nm, _ in PNAME:
        macro(P + f"{nm}RhatOne", 100 * S[f"share_rhat_{p}_lt_1.01"], "{:.0f}")
        macro(P + f"{nm}RhatFive", 100 * S[f"share_rhat_{p}_lt_1.05"], "{:.0f}")
        macro(P + f"{nm}EssHundred", 100 * S[f"share_ess_{p}_ge_100"], "{:.0f}")
        macro(P + f"{nm}RhatMax", S[f"rhat_{p}_max"], "{:.2f}")
        macro(P + f"{nm}RhatMed", S[f"rhat_{p}_median"], "{:.3f}")
        macro(P + f"{nm}EssMed", S[f"ess_{p}_median"], "{:.0f}")
        macro(P + f"{nm}EssMin", S[f"ess_{p}_min"], "{:.0f}")
    pn = [p for p, _, _ in PNAME]
    macro(P + "RhatFiveMin", 100 * min(S[f"share_rhat_{p}_lt_1.05"] for p in pn), "{:.0f}")
    macro(P + "RhatOneMin", 100 * min(S[f"share_rhat_{p}_lt_1.01"] for p in pn), "{:.0f}")
    macro(P + "EssHundredMin", 100 * min(S[f"share_ess_{p}_ge_100"] for p in pn), "{:.0f}")
    macro(P + "RhatMaxAll", max(S[f"rhat_{p}_max"] for p in pn), "{:.2f}")
    macro(P + "RhatFiveMax", 100 * max(S[f"share_rhat_{p}_lt_1.05"] for p in pn), "{:.0f}")
    macro(P + "EssHundredMax", 100 * max(S[f"share_ess_{p}_ge_100"] for p in pn), "{:.0f}")
    macro(P + "RhatMedMaxAll", max(S[f"rhat_{p}_median"] for p in pn), "{:.3f}")
    macro(P + "EssMedMin", min(S[f"ess_{p}_median"] for p in pn), "{:.0f}")
    macro(P + "LLSpreadMed", 100 * S["ll_spread_rel_median"], "{:.1f}")
    macro(P + "FloorChains", S["chains_at_sd_a_floor"], "{:d}")
    macro(P + "BoxSdB", S["panels_sd_b_near_upper"], "{:d}")
    macro(P + "BoxSdA", S["panels_sd_a_near_upper"], "{:d}")
    macro(P + "Chains", S["chains_total"], "{:d}")
    macro(P + "LLSpreadMax", 100 * S["ll_spread_rel_max"], "{:.1f}")
    macro(P + "EbScaleMed", S["eb_scale_median"], "{:.2f}")
    macro(P + "PostScaleMed", S["post_scale_median"], "{:.2f}")
    macro(P + "PostOverEbMed", S["post_over_eb_scale_median"], "{:.2f}")
    macro(P + "PostOverEbMin", S["post_over_eb_scale_min"], "{:.2f}")
    macro(P + "PostOverEbMax", S["post_over_eb_scale_max"], "{:.2f}")
    macro(P + "PostAboveEb", 100 * S["share_post_scale_above_eb"], "{:.0f}")
    macro(P + "PostSdScaleMed", S["post_sd_scale_median"], "{:.2f}")
    macro(P + "Converged", "met" if sb["converged"] else "not met")
    macro(P + "Keep", "kept" if sb["keep_not_different"] else "not kept")
    for k, nm in [(f"pdm_bmi_{tag}", "PdmBmi"), (f"pdm_eig_{tag}", "PdmEig"), (f"jfb_eig_{tag}", "JfbEig"),
                  (f"jfb_bmi_{tag}", "JfbBmi")]:
        c = comp[k]
        macro(P + f"{nm}VsEost", c["median_rel_ll_vs_eost_pct"], "{:.2f}")
        macro(P + f"{nm}BanksBetter", c["banks_better_than_eost"], "{:d}")
        macro(P + f"{nm}BanksWorse", c["banks_worse_than_eost"], "{:d}")
        macro(P + f"{nm}WithinOne", c["banks_within_1pct"], "{:d}")
        macro(P + f"{nm}SignP", c["bank_sign_test_p"], "{:.2f}")
        macro(P + f"{nm}Cov", c["mean_coverage"])
        macro(P + f"{nm}IaeVsEost", c["median_rel_iae_vs_eost_pct"], "{:.1f}")
        macro(P + f"{nm}VsSixty", c["median_rel_ll_vs_60_pct"], "{:.2f}")
        macro(P + f"{nm}AbsVsSixty", c["median_abs_rel_ll_vs_60_pct"], "{:.2f}")
        macro(P + f"{nm}VsLong", c["median_rel_ll_vs_long_pct"], "{:.2f}")
        macro(P + f"{nm}AbsVsLong", c["median_abs_rel_ll_vs_long_pct"], "{:.2f}")
        if "median_rel_ll_vs_scale_pct" in c:
            macro(P + f"{nm}VsScale", c["median_rel_ll_vs_scale_pct"], "{:.2f}")
            macro(P + f"{nm}AbsVsScale", c["median_abs_rel_ll_vs_scale_pct"], "{:.2f}")
    keys = [f"pdm_bmi_{tag}", f"pdm_eig_{tag}", f"jfb_eig_{tag}", f"jfb_bmi_{tag}"]
    gain = [-comp[k]["median_rel_iae_vs_eost_pct"] for k in keys]
    macro(P + "IaeGainMin", min(gain), "{:.0f}")
    macro(P + "IaeGainMax", max(gain), "{:.0f}")
    nb = S["banks"]
    better = [nb - comp[k]["banks_iae_worse_than_eost"] for k in keys]
    macro(P + "IaeBanksMin", min(better), "{:d}")
    macro(P + "IaeBanksMax", max(better), "{:d}")
    v = [comp[k]["median_rel_ll_vs_eost_pct"] for k in keys]
    macro(P + "VsEostMin", min(v), "{:.2f}")
    macro(P + "VsEostMax", max(v), "{:.2f}")
    cv = [comp[k]["mean_coverage"] for k in keys]
    macro(P + "CovMin", min(cv))
    macro(P + "CovMax", max(cv))
    macro(P + "EostCov", comp["eost"]["mean_coverage"])


# ----------------------------------------------------------------------------- RF3
def rf3():
    f = "artifacts/robust/meta/meta.csv"
    if not os.path.exists(f):
        return
    M = pd.read_csv(f).set_index("hypothesis")
    lab = {"R2": "R2 plug-in 2PL, log loss", "R3": "R3 Naive, score IAE", "R4": "R4 plug-in 2PL, $|\\mathrm{cov}-0.9|$",
           "C1": "C1 Rasch-type plug-in, log loss", "C2": "C2 Rasch-type marginal, log loss",
           "C4": "C4 Random:mg, score IAE", "C5b": "C5b raw interval$^\\ast$, interval score",
           "C6": "C6 Naive (Wilson)$^\\ast$, interval score", "C7": "C7 ATLAS-style, score IAE",
           "C8": "C8 tinyBenchmarks, error @50", "C9": "C9 MINCE, error @50"}
    fs = "artifacts/robust/meta/sources.csv"
    Ssrc = pd.read_csv(fs).set_index("hypothesis") if os.path.exists(fs) else None
    rows = []
    for h in ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"]:
        r = M.loc[h]
        n = lambda v: (f"$-${-v:.1f}" if v < -0.05 else f"{abs(v) if abs(v) < 0.05 else v:.1f}")
        src = (f" & [{n(100 * Ssrc.loc[h, 'ci_lo'])}, {n(100 * Ssrc.loc[h, 'ci_hi'])}]" if Ssrc is not None else "")
        rows.append(f"{lab[h]} & {signed(100 * r.pooled)} & [{n(100 * r.ci_lo)}, {n(100 * r.ci_hi)}] & "
                    f"[{n(100 * r.pi_lo)}, {n(100 * r.pi_hi)}] & {100 * r.I2:.0f} & {int(r.n_positive)}/{int(r.n_banks)} & "
                    f"{r.p_sign:.3f}{src} & {'yes' if r.frozen_supported else 'no'} \\\\")
        t = h.replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four").replace("5", "Five") \
            .replace("6", "Six").replace("7", "Seven").replace("8", "Eight").replace("9", "Nine")
        macro(f"Meta{t}Pooled", 100 * r.pooled, "{:.1f}")
        macro(f"Meta{t}Lo", 100 * r.ci_lo, "{:.1f}")
        macro(f"Meta{t}Hi", 100 * r.ci_hi, "{:.1f}")
        macro(f"Meta{t}PiLo", 100 * r.pi_lo, "{:.1f}")
        macro(f"Meta{t}PiHi", 100 * r.pi_hi, "{:.1f}")
    exc = [h for h in M.index if M.loc[h, "ci_lo"] > 0]
    nsup = [h for h in exc if not M.loc[h, "frozen_supported"]]
    macro("MetaNExclZero", len(exc), "{:d}")
    macro("MetaNExclZeroUnsup", len(nsup), "{:d}")
    macro("MetaUnsupList", ", ".join(nsup))
    pix = [h for h in ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"] if M.loc[h, "pi_lo"] > 0]
    macro("MetaPiExclList", ", ".join(pix))
    macro("MetaNPiExcl", len(pix), "{:d}")
    hi = M.loc[nsup, "I2"] if nsup else M.I2
    macro("MetaUnsupIMin", 100 * float(hi.min()), "{:.0f}")
    macro("MetaUnsupIMax", 100 * float(hi.max()), "{:.0f}")
    if Ssrc is not None:
        order = ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"]
        ex = [h for h in order if Ssrc.loc[h, "ci_lo"] > 0]
        lost = [h for h in order if M.loc[h, "ci_lo"] > 0 and not Ssrc.loc[h, "ci_lo"] > 0]
        kept_unsup = [h for h in nsup if Ssrc.loc[h, "ci_lo"] > 0]
        macro("MetaSrcK", int(Ssrc.k_sources.max()), "{:d}")
        macro("MetaSrcNExclZero", len(ex), "{:d}")
        macro("MetaSrcExclList", ", ".join(ex))
        macro("MetaSrcLostList", ", ".join(lost))
        macro("MetaSrcNLost", len(lost), "{:d}")
        macro("MetaSrcUnsupKeptList", ", ".join(kept_unsup) if kept_unsup else "none")
        macro("MetaSrcNUnsupKept", len(kept_unsup), "{:d}")
        pos = [int(Ssrc.loc[h, "n_positive_sources"]) for h in lost]
        macro("MetaSrcLostPosMin", min(pos) if pos else 0, "{:d}")
        macro("MetaSrcLostPosMax", max(pos) if pos else 0, "{:d}")
        macro("MetaSrcR", float(Ssrc.ollb_median_pairwise_r.median()), "{:.2f}")
        macro("MetaSrcSeRatio", float((Ssrc.ollb_se / Ssrc.ollb_se_independent).median()), "{:.2f}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Cross-benchmark effect of each study-2 superiority hypothesis (follow-up RF3, secondary; no frozen decision changes): random-effects pooled relative improvement of EOST-M (\%) over the eight benchmarks treated as independent, Hartung--Knapp--Sidik--Jonkman 95\% interval, 95\% prediction interval for a new benchmark, $I^2$ (\%), benchmarks with a positive effect and exact two-sided sign-test $p$; ``sources'': 95\% interval when the five Open LLM Leaderboard benchmarks, which share their 395 systems, count as one source (RF3b; four sources, 3 degrees of freedom); frozen decision. $^\ast$Recalibrated EOST-M interval.}
\label{tab:meta}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrccccccl@{}}
\toprule
EOST-M versus & pooled & 95\% CI & 95\% PI & $I^2$ & pos. & sign $p$ & sources & frozen \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/meta.tex", "w").write(tex)


def results_md():
    """Append the robustness follow-ups to RESULTS.md (after make_report.py and report_study2.py)."""
    if not os.path.exists("RESULTS.md"):
        return
    L = ["", "## Robustness follow-ups after the second review (docs/RESEARCH_LOG.md, iteration 4)", ""]
    f = "artifacts/robust/clusters/_analysis/decisions.json"
    if os.path.exists(f):
        d = json.load(open(f))
        L += ["RF1 -- clusters that never see the target's responses (planned before running); reproduction of the "
              f"frozen rows: max |diff| = {d.get('repro_max_abs_diff')}; rule met under every variant: "
              f"{d.get('rule_met_all_variants')}.", "",
              "| Panel-exclusion rule | R2 median / pos. / sig. | C1 median / pos. / sig. |", "|---|---|---|"]
        for v, x in d["variants"].items():
            r2, c1 = x["decisions"]["R2"], x["decisions"]["C1"]
            L.append(f"| {v} | {r2['median_rel_improvement_pct']:+.1f}% / {r2['positive']} / {r2['favour_new']}-{r2['favour_base']} | "
                     f"{c1['median_rel_improvement_pct']:+.1f}% / {c1['positive']} / {c1['favour_new']}-{c1['favour_base']} |")
    for tag, sub, name in (("RF2", "fb_convergence", "frozen sampler, 4 long chains (planned before running)"),
                           ("RF2b", "fb_scale", "scale move (exploratory)"),
                           ("RF2d", "fb_exact2", "exact sampler of Lemma 1's model (exploratory)")):
        f = f"artifacts/robust/{sub}/_analysis/summary.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        S = d["summary"]
        ps = [p for p in ("loc", "scale", "mu_b", "sd_b", "mu_a", "sd_a") if f"share_rhat_{p}_lt_1.05" in S]
        L += ["", f"{tag} -- {name}: share of panels with R-hat < 1.05 / bulk ESS >= 100: " +
              ", ".join(f"{p} {100 * S[f'share_rhat_{p}_lt_1.05']:.0f}%/{100 * S[f'share_ess_{p}_ge_100']:.0f}%" for p in ps)]
        comp = d["comparison"]
        suf = {"RF2": "long", "RF2b": "scale", "RF2d": "exact"}[tag]
        L.append("  median relative difference in held-out log loss to EOST-M (positive: worse than EOST-M): " +
                 ", ".join(f"{a} {comp[f'{a}_{suf}']['median_rel_ll_vs_eost_pct']:+.2f}%"
                           for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")) +
                 f"; coverage EOST-M {comp['eost']['mean_coverage']:.2f}, arms " +
                 ", ".join(f"{comp[f'{a}_{suf}']['mean_coverage']:.2f}" for a in ("pdm_bmi", "pdm_eig", "jfb_eig", "jfb_bmi")))
        if tag == "RF2":
            L.append(f"  pre-specified rule: {'confirmed' if d['rule_confirmed'] else 'NOT confirmed'} (reading recorded "
                     f"before the comparison); narrow reading: {'confirmed' if d['rule_confirmed_narrow_reading'] else 'not confirmed'}")
        else:
            L.append(f"  convergence criterion (R-hat < 1.05 and ESS >= 100 on >= 90% of panels for every quantity): "
                     f"{'met' if d['converged'] else 'NOT met'}")
    f = "artifacts/robust/meta/meta.csv"
    if os.path.exists(f):
        M = pd.read_csv(f)
        L += ["", "RF3 -- random-effects meta-analysis over benchmarks (secondary):", "",
              "| Hypothesis | pooled | 95% CI | 95% PI | I2 | frozen decision |", "|---|---|---|---|---|---|"]
        for _, r in M.iterrows():
            L.append(f"| {r.hypothesis} | {100 * r.pooled:+.1f}% | {100 * r.ci_lo:+.1f} to {100 * r.ci_hi:+.1f} | "
                     f"{100 * r.pi_lo:+.1f} to {100 * r.pi_hi:+.1f} | {100 * r.I2:.0f}% | "
                     f"{'supported' if r.frozen_supported else 'not supported'} |")
    txt = open("RESULTS.md").read()
    head, marker = "\n## Robustness follow-ups after the second review", "Every qualitative statement of the paper"
    if head in txt:                                   # idempotent: drop the block of an earlier run
        j = txt.index(head)
        k = txt.find(marker, j)
        txt = txt[:j] + ("\n\n" + txt[k:] if k >= 0 else "\n")
    i = txt.find(marker)
    block = "\n".join(L) + "\n\n"
    txt = (txt[:i].rstrip("\n") + "\n" + block + txt[i:]) if i >= 0 else txt + block
    open("RESULTS.md", "w").write(txt)


def write():
    with open(f"{GEN}/numbers3.tex", "w") as fh:
        for k, v in sorted(NUM.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"REPORT3_GATE=PASS macros={len(NUM)}")


if __name__ == "__main__":
    rf1()
    rf2()
    rf3()
    write()
    results_md()
