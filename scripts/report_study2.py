"""Tables, figures and number macros for the revision (study-1 re-analyses and study 2).

Runs in the container after make_report.py. Reads the artifacts and writes
paper/tables/*.tex, paper/figures/*.pdf and paper/generated/numbers2.tex (macros).
Every block is skipped if its artifacts are missing, so the paper compiles at any stage;
the final build must contain no '??' (checked by the release gate).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIG, TAB, GEN = "paper/figures", "paper/tables", "paper/generated"
CONF = os.environ.get("S2_BANKS", "ollb_gsm8k,ollb_arc,ollb_hellaswag,ollb_winogrande,ollb_mmlu,swe_verified,"
                                  "livecodebench,cruxeval_o").split(",")    # override for testing only
CONF1 = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
         "livecodebench", "cruxeval_o"]                                    # study 1 (never overridden)
NICE = {"ollb_gsm8k": "GSM8K", "ollb_arc": "ARC-C", "ollb_hellaswag": "HellaSwag", "ollb_winogrande": "WinoGrande",
        "ollb_mmlu": "MMLU", "swe_verified": "SWE-bench V.", "livecodebench": "LiveCodeBench",
        "cruxeval_o": "CRUXEval-O", "synth_2pl": "Synth. 2PL", "synth_guess": "Synth. guessing",
        "synth_2dim": "Synth. 2-D", "swe_lite": "SWE-bench Lite", "cruxeval_i": "CRUXEval-I",
        "ollb_truthfulqa": "TruthfulQA"}
AGG = {"livecodebench", "cruxeval_o"}          # recorded aggregate verdicts
INK, INK2 = "#0b0b0b", "#52514e"
NUM = {}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.fontsize": 7})


def macro(name, value, fmt="{:.2f}"):
    assert name.isalpha(), name
    if isinstance(value, str):
        NUM[name] = value
    else:
        t = fmt.format(value)
        NUM[name] = ("$-$" + t[1:]) if t.startswith("-") else t


def pct(v, fmt="{:.1f}"):
    return fmt.format(v)


def signed(v, fmt="{:+.1f}"):
    t = fmt.format(v)
    if float(t) == 0:
        t = t.lstrip("+-")
    return "$" + t + "$"


# ----------------------------------------------------------------------------- environment, clusters
def env_cluster_macros():
    if os.path.exists("configs/ENVIRONMENT.json"):
        e = json.load(open("configs/ENVIRONMENT.json"))
        macro("ImageIdShort", e["release_runtime"]["image_id"].split(":")[1][:12])
        macro("ImageIdFull", e["release_runtime"]["image_id"].split(":")[1])
        macro("BaseDigestShort", e["release_runtime"]["base_image"].split("sha256:")[1][:12])
        macro("PaperImageIdShort", e["release_paper"]["image_id"].split(":")[1][:12])
        macro("PilotImageIdShort", e["pilot_runtime_as_recorded_in_project_log"]["image_id"].split(":")[1][:12])
    if os.path.exists("configs/study2_clusters.json"):
        c = json.load(open("configs/study2_clusters.json"))["banks"]
        macro("ClusOLLB", c["ollb_gsm8k"]["n_clusters"], "{:d}")
        macro("ClusOLLBMax", c["ollb_gsm8k"]["max_size"], "{:d}")
        macro("ClusSWE", c["swe_verified"]["n_clusters"], "{:d}")
        macro("ClusSWEMax", c["swe_verified"]["max_size"], "{:d}")
        macro("ClusLCB", c["livecodebench"]["n_clusters"], "{:d}")
        macro("ClusCRUX", c["cruxeval_o"]["n_clusters"], "{:d}")


# ----------------------------------------------------------------------------- study 1, cluster-aware
def study1_clusters():
    f = "artifacts/confirm/_clusters/decisions.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    names = {"H1": "EIG:mg vs Entropy:mg (log loss)", "H2": "EIG:mg vs MFI:pi (log loss)",
             "H3": "EIG:mg vs Naive (score IAE)", "H4": "EIG:mg vs MFI:pi ($|\\mathrm{cov}-0.9|$)"}
    rows = []
    for h in ["H1", "H2", "H3", "H4"]:
        cells = []
        for unit in ["target", "heuristic", "strict"]:
            u = d[unit][h]
            cells.append(f"{u['favour_new']}/{u['favour_base']}")
            tag = {"target": "Tgt", "heuristic": "Heu", "strict": "Str"}[unit]
            macro(f"Clu{h.replace('H1','HOne').replace('H2','HTwo').replace('H3','HThree').replace('H4','HFour')}{tag}Fav",
                  u["favour_new"], "{:d}")
            macro(f"Clu{h.replace('H1','HOne').replace('H2','HTwo').replace('H3','HThree').replace('H4','HFour')}{tag}Med",
                  u["median_rel_improvement_pct"], "{:.1f}")
        ci = d["strict_cluster_bootstrap"][h]["median_rel_improvement_ci95_pct"]
        hh = h.replace('H1', 'HOne').replace('H2', 'HTwo').replace('H3', 'HThree').replace('H4', 'HFour')
        macro(f"Clu{hh}CILo", ci[0], "{:.1f}")
        macro(f"Clu{hh}CIHi", ci[1], "{:.1f}")
        sup = ["yes" if d[u][h]["supported"] else "no" for u in ["target", "heuristic", "strict"]]
        rows.append(f"{h} & {names[h]} & " + " & ".join(f"{c} ({s})" for c, s in zip(cells, sup))
                    + f" & {d['strict'][h]['median_rel_improvement_pct']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] \\\\")
    units = d["strict"]["H2"]["units_per_bank"]
    macro("CluStrictUnitsMin", min(units.values()), "{:d}")
    macro("CluStrictUnitsMax", max(units.values()), "{:d}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 1 with dependent targets taken into account (post hoc). Benchmarks with a Holm-significant difference favouring EOST-M / favouring the baseline at $N=8$, with the frozen decision in parentheses, when the unit of analysis is the target (frozen plan), the heuristic family or the strict cluster (targets averaged within unit; same 144-test Holm family). Last column: median relative improvement over benchmarks with strict clusters as units, and its 95\% cluster-bootstrap interval.}
\label{tab:clusters}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
 & Contrast & target & heuristic family & strict cluster & median \% [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/clusters.tex", "w").write(tex)


# ----------------------------------------------------------------------------- cost
def cost():
    f = "artifacts/cost/breakeven.csv"
    if not os.path.exists(f):
        return
    D = pd.read_csv(f).set_index("dataset")
    s = json.load(open("artifacts/cost/summary.json"))
    rows = []
    for ds in CONF1:
        r = D.loc[ds]
        bn = "--" if pd.isna(r.breakeven_vs_naive) else f"{r.breakeven_vs_naive:.0f}"
        tr = "$>$100" if pd.isna(r.items_match_random_mg100) else f"{int(r.items_match_random_mg100)}"
        rows.append(f"{NICE[ds]} & {int(r.I)} & {r.breakeven_vs_full:.1f} & {int(r.items_match_naive100)} & {bn} & {tr} \\\\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Cost of adaptive evaluation with a fresh calibration panel of $N=8$ systems that must first answer all $I$ items (study 1). Break-even: number of new systems from which EOST-M with 100 items costs less than full evaluation; items for which EOST-M matches the score error of 100 uniformly sampled items scored by the sample mean (Naive) or by the same calibrated model (Random:mg), and the resulting break-even against Naive. When the calibrators' full results already exist (public leaderboards), the calibration cost vanishes and EOST-M is cheaper from the first system.}
\label{tab:cost}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & & break-even vs & items to match & break-even vs & items to match \\
Benchmark & $I$ & full evaluation & Naive@100 & Naive & Random:mg@100 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/cost.tex", "w").write(tex)
    macro("CostBreakFullMed", s["median_breakeven_vs_full"], "{:.1f}")
    macro("CostBreakFullMin", s["range_breakeven_vs_full"][0], "{:.1f}")
    macro("CostBreakFullMax", s["range_breakeven_vs_full"][1], "{:.0f}")
    macro("CostItemsNaive", s["median_items_match_naive100"], "{:.1f}")
    macro("CostSaveNaive", 100 - s["median_items_match_naive100"], "{:.0f}")
    macro("CostRandomSaveBanks", int((pd.to_numeric(D.items_match_random_mg100) < 100).sum()), "{:d}")
    macro("CostItemsRandom", s["median_items_match_random_mg100"], "{:.0f}")
    macro("CostBreakNaiveMed", s["median_breakeven_vs_naive"], "{:.0f}")
    macro("CostBreakNaiveMin", s["range_breakeven_vs_naive"][0], "{:.0f}")
    macro("CostBreakNaiveMax", s["range_breakeven_vs_naive"][1], "{:.0f}")
    macro("CostRatioHundred", 100 * s["median_cost_ratio_M100"], "{:.0f}")


# ----------------------------------------------------------------------------- EB robustness
def eb_robustness():
    f = "artifacts/eb_robustness/summary.json"
    if not os.path.exists(f):
        return
    s = json.load(open(f))
    st, ppc = s["stability"], s["ppc"]
    keys = [("mu_b", r"$\mu_b$"), ("sd_b", r"$\sigma_b$"), ("mu_log_a", r"$\mu_a$"), ("sd_log_a", r"$\sigma_a$")]
    rows = []
    for k, lab in keys:
        cells = []
        for N in ("4", "8", "16"):
            g = st[N]
            cells += [f"{g['between_panel_sd'][k]:.2f}", f"{g['item_bootstrap_sd'][k]:.2f}", f"{g['half_split_absdiff'][k]:.2f}"]
        cells.append(f"{st['8']['calibrator_bootstrap_sd'][k]:.2f}")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Stability of the empirical-Bayes hyperparameters (post hoc; three panels per benchmark and panel size). Medians over benchmarks of: the standard deviation across panels (between), the bootstrap standard deviation over items with abilities fixed (items), the absolute difference between fits on two random halves of the items (halves), and at $N=8$ the bootstrap standard deviation over calibrators (calibrators).}
\label{tab:ebstab}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccccc c@{}}
\toprule
 & \multicolumn{3}{c}{$N=4$} & \multicolumn{3}{c}{$N=8$} & \multicolumn{3}{c}{$N=16$} & $N=8$ \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-11}
 & between & items & halves & between & items & halves & between & items & halves & calibrators \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/eb_stability.tex", "w").write(tex)
    macro("EBItemsSdMaxEight", max(st["8"]["item_bootstrap_sd"].values()), "{:.2f}")
    macro("EBCalSdSigmaA", st["8"]["calibrator_bootstrap_sd"]["sd_log_a"], "{:.2f}")
    macro("EBCalSdSigmaB", st["8"]["calibrator_bootstrap_sd"]["sd_b"], "{:.2f}")
    macro("EBBetweenSigmaA", st["8"]["between_panel_sd"]["sd_log_a"], "{:.2f}")
    P = pd.read_csv("artifacts/eb_robustness/ppc.csv") if os.path.exists("artifacts/eb_robustness/ppc.csv") else None
    for k, nm in [("frac_extreme_items", "Extreme"), ("sd_item_p", "SdP"), ("mean_item_rest_r", "Rest")]:
        if k in ppc:
            macro(f"PPC{nm}Ppp", ppc[k]["median_ppp"], "{:.2f}")
            macro(f"PPC{nm}Low", 100 * ppc[k]["share_ppp_below_0.05"], "{:.0f}")
            macro(f"PPC{nm}Obs", ppc[k]["median_observed"], "{:.2f}")
            macro(f"PPC{nm}Sim", ppc[k]["median_simulated"], "{:.2f}")
        if P is not None:
            s_ = P[P.stat == k]
            flag = s_.ppp < 0.05
            # direction of the misfit among flagged panels (observed above / below the simulated median)
            macro(f"PPC{nm}Above", 100 * float((flag & (s_.observed > s_.sim_median)).mean()), "{:.0f}")
            macro(f"PPC{nm}Below", 100 * float((flag & (s_.observed < s_.sim_median)).mean()), "{:.0f}")
            macro("PPCPanels", int(len(s_)), "{:d}")


# ----------------------------------------------------------------------------- sampler mixing
def fb_mixing():
    f = "artifacts/study2/fb_mixing.json"
    if not os.path.exists(f):
        return
    M = pd.DataFrame(json.load(open(f)))
    w, n = M[M.sampler == "with_translation_move"], M[M.sampler == "without_move"]
    macro("MixLocWithMax", w.acf3_theta_mean.max())
    macro("MixLocWithoutMin", n.acf3_theta_mean.min())
    macro("MixLocWithoutMax", n.acf3_theta_mean.max())
    macro("MixScaleMin", w.acf3_sd_log_a.min())
    macro("MixScaleMax", w.acf3_sd_log_a.max())
    for N, nm in [(4, "Four"), (8, "Eight"), (16, "Sixteen")]:
        macro(f"MixLocSd{nm}", float(w[w.N == N].sd_theta_mean.median()))


# ----------------------------------------------------------------------------- development (rule choice)
def dev_rule():
    fs = sorted(glob.glob("artifacts/study2/dev/*/runs.csv.gz"))
    if not fs:
        return
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = R[R.policy == "eig:mg"].groupby(["dataset", "N", "target"])[
        ["coverage", "coverage_Rq", "coverage_Rr", "coverage_Ra"]].mean().groupby(["dataset", "N"]).mean()
    rows = []
    for (ds, N), r in g.iterrows():
        rows.append(f"{NICE.get(ds, ds)} & {N} & {r.coverage:.2f} & {r.coverage_Rq:.2f} & {r.coverage_Rr:.2f} & {r.coverage_Ra:.2f} \\\\")
    m = g.mean()
    rows.append(r"\midrule" + "\n" + f"mean & & {m.coverage:.2f} & {m.coverage_Rq:.2f} & {m.coverage_Rr:.2f} & {m.coverage_Ra:.2f} \\\\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Choice of the interval recalibration on the development banks (before study 2; 20 targets or fewer, two replicates). Mean coverage of EOST-M's 90\% score interval without recalibration and with the three leave-one-calibrator-out rules: quantile of standardized errors (Rq), their root mean square (Rr), quantile of absolute errors (Ra). The rule written in advance selects the rule with mean coverage closest to 0.90: Ra.}
\label{tab:devrule}
\begin{tabular}{@{}lrcccc@{}}
\toprule
Bank & $N$ & raw & Rq & Rr & Ra \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/devrule.tex", "w").write(tex)
    macro("DevRaCov", m.coverage_Ra)
    macro("DevRqCov", m.coverage_Rq)
    macro("DevRrCov", m.coverage_Rr)


# ----------------------------------------------------------------------------- study 2
S2 = os.environ.get("S2_ANALYSIS", "artifacts/study2/_analysis")   # override for testing only
S2_COMP = os.environ.get("S2_COMPONENT", "main")
POLNAME = {"eig:mg": "EOST-M (EIG:mg)", "eig:mg+R": "EOST-M, recalibrated", "mfi:1pr": "Rasch-type plug-in (MFI:1pr)",
           "mfi:1pr+R": "Rasch-type, recalibrated", "eig:1prmg": "Rasch-type marginal (EIG:1prmg)",
           "fb:bmi": "PDM Bayesian CAT, BMI (frozen burn-in)", "fb:eig": "PDM Bayesian CAT, EIG (frozen burn-in)", "random:mg": "Random:mg",
           "naive": "Naive (Wilson)", "mfi:pi": "plug-in 2PL (MFI:pi)", "atlas": "ATLAS-style 3PL CAT",
           "eig:xf": "EOST-M, cross-fitted EB", "eig:tp": "EOST-M, Student-$t$ EB", "mix:mg": "mixed (every 4th random)",
           "tiny:gpirt": "tinyBenchmarks gp-IRT", "tiny:pirt": "tinyBenchmarks p-IRT", "tiny:anchor": "tinyBenchmarks anchors",
           "mince": "MINCE subset mean"}
HNAME = {"R2": ("plug-in 2PL (MFI:pi)", "log loss"), "R3": ("Naive", "score IAE"),
         "R4": ("plug-in 2PL (MFI:pi)", "$|\\mathrm{cov}-0.9|$"),
         "C1": ("Rasch-type plug-in", "log loss"), "C2": ("Rasch-type marginal", "log loss"),
         "C4": ("Random:mg", "score IAE"), "C5b": ("raw EOST-M interval$^\\ast$", "interval score"),
         "C6": ("Naive (Wilson)$^\\ast$", "interval score"), "C7": ("ATLAS-style 3PL CAT", "score IAE"),
         "C8": ("tinyBenchmarks", "error @50"), "C9": ("MINCE", "error @50")}


def study2():
    f = f"{S2}/decisions.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    dec = d["decisions"]
    C = pd.read_csv(f"{S2}/confirmatory_tests.csv")
    tm = pd.read_csv(f"{S2}/target_means.csv.gz")
    rows = []
    tag = lambda h: h.replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four").replace("5", "Five") \
        .replace("6", "Six").replace("7", "Seven").replace("8", "Eight").replace("9", "Nine")
    for h in ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"]:
        x = dec[h]
        comp, met = HNAME[h]
        sub = C[C.hypothesis == h]
        rows.append(f"{h} & {comp} & {met} & {x['favour_new']}/{x['favour_base']} & {x['positive']}/{x['n_banks']} & "
                    f"{signed(x['median_rel_improvement_pct'])} & {'yes' if x['supported'] else 'no'} \\\\")
        t = tag(h)
        macro(f"STwo{t}Fav", x["favour_new"], "{:d}")
        macro(f"STwo{t}Base", x["favour_base"], "{:d}")
        macro(f"STwo{t}Pos", x["positive"], "{:d}")
        macro(f"STwo{t}Med", x["median_rel_improvement_pct"], "{:.1f}")
        macro(f"STwo{t}MedAbs", abs(x["median_rel_improvement_pct"]), "{:.1f}")
        macro(f"STwo{t}Verdict", "supported" if x["supported"] else "not supported")
    x = dec["C3"]
    rows.append(r"\midrule" + "\n" + f"C3 & PDM Bayesian CAT (equivalence) & log loss & \\multicolumn{{2}}{{c}}{{{x['n_equivalent']}/{x['n_banks']} within $\\pm$1\\%}} & "
                f"{signed(x['median_rel_diff_pct'], '{:+.2f}')} & {'yes' if x['supported'] else 'no'} \\\\")
    macro("STwoCThreeEq", x["n_equivalent"], "{:d}")
    macro("STwoCThreeMed", x["median_rel_diff_pct"], "{:.2f}")
    macro("STwoCThreeVerdict", "supported" if x["supported"] else "not supported")
    x = dec["C5a"]
    covs = [v for v in x["coverage"].values() if v == v]
    rows.append(f"C5a & (coverage in $[0.85,0.95]$)$^\\ast$ & coverage & \\multicolumn{{2}}{{c}}{{{x['n_inside']}/{len(covs)} banks inside}} & "
                f"{min(covs):.2f}--{max(covs):.2f} & {'yes' if x['supported'] else 'no'} \\\\")
    macro("STwoCFiveaIn", x["n_inside"], "{:d}")
    macro("STwoCFiveaMin", min(covs))
    macro("STwoCFiveaMax", max(covs))
    macro("STwoCFiveaVerdict", "supported" if x["supported"] else "not supported")
    n_sup = sum(1 for k, v in dec.items() if v.get("supported"))
    macro("STwoNSupported", n_sup, "{:d}")
    macro("STwoNHyp", len(dec), "{:d}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2: decisions under the frozen plan (primary panel size $N=8$; one target per strict family cluster). Holm sig.: benchmarks with a Holm-significant difference favouring EOST-M / favouring the competitor (Holm over the 88 superiority tests); positive: benchmarks on which EOST-M is better; median: median over benchmarks of the relative improvement of EOST-M (\%; for C3 the relative difference of the posterior-draw marginal (PDM) Bayesian CAT, for C5a the range of coverage). Supported: at least five benchmarks for EOST-M and none against (C3: equivalence on at least six; C5a: inside on at least six). $^\ast$C5a, C5b and C6 concern EOST-M's recalibrated interval.}
\label{tab:study2}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllcccl@{}}
\toprule
 & EOST-M versus & Endpoint & Holm sig. & positive & median & Supported \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/study2.tex", "w").write(tex)

    # per-bank detail of the main contrasts
    det = []
    for ds in CONF:
        cells = []
        for h in ["R2", "C1", "C2", "C4", "C7"]:
            r = C[(C.hypothesis == h) & (C.dataset == ds)]
            if r.empty:
                cells.append("--"); continue
            r = r.iloc[0]
            star = "*" if r.p_holm < 0.05 else ""
            cells.append(f"{signed(100 * r.rel_improvement)}{star}")
        n = int(tm[(tm.dataset == ds) & (tm.N == 8)].target.nunique())
        det.append(f"{NICE[ds]}{'$^\\dagger$' if ds in AGG else ''} & {n} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2 per benchmark ($N=8$): number of targets (independent clusters) and relative improvement (\%) of EOST-M over each competitor; * Holm-significant. $^\dagger$Recorded aggregate verdicts.}
\label{tab:study2banks}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{held-out log loss} & \multicolumn{2}{c}{score IAE} \\
\cmidrule(lr){3-5}\cmidrule(l){6-7}
Benchmark & targets & MFI:pi & Rasch plug-in & Rasch marg. & Random:mg & ATLAS \\
\midrule
""" + "\n".join(det) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/study2_banks.tex", "w").write(tex)
    macro("STwoTargetsTotal", int(tm[tm.N == 8].groupby("dataset").target.nunique().sum()), "{:d}")
    macro("STwoTargetsMin", int(tm[tm.N == 8].groupby("dataset").target.nunique().min()), "{:d}")
    macro("STwoTargetsMax", int(tm[tm.N == 8].groupby("dataset").target.nunique().max()), "{:d}")

    # deterministic vs aggregate verdicts: median relative improvement for the main contrasts
    for h in ["R2", "C1", "C4"]:
        sub = C[C.hypothesis == h]
        det_ = sub[~sub.dataset.isin(AGG)].rel_improvement.median() * 100
        agg_ = sub[sub.dataset.isin(AGG)].rel_improvement.median() * 100
        macro(f"STwo{tag(h)}Det", det_, "{:.1f}")
        macro(f"STwo{tag(h)}Agg", agg_, "{:.1f}")

    # policy means at N = 8 (averaged over banks)
    pm = tm[tm.N == 8].groupby(["dataset", "policy"]).mean(numeric_only=True)
    avg = pm.groupby("policy").mean()
    order = ["eig:mg", "eig:mg+R", "mfi:1pr", "mfi:1pr+R", "eig:1prmg", "fb:bmi", "fb:eig", "mfi:pi", "atlas",
             "random:mg", "naive", "eig:xf", "eig:tp", "mix:mg"]
    rows = []
    for p in order:
        if p not in avg.index:
            continue
        r = avg.loc[p]
        ll = "--" if p in ("naive",) or pd.isna(r.get("ilogloss")) else f"{r.ilogloss:.4f}"
        if p.endswith("+R"):
            ll = "(same)"
        rows.append(f"{POLNAME[p]} & {ll} & {r.iae:.4f} & {r.coverage:.2f} & {r.mean_width:.3f} & {r.iis:.3f} \\\\")
        t = p.replace(":", "").replace("+", "R").replace("1", "One")
        t = "".join(ch for ch in t if ch.isalpha())
        macro(f"STwoCov{t}", r.coverage)
        macro(f"STwoWidth{t}", r.mean_width, "{:.3f}")
        macro(f"STwoIS{t}", r.iis, "{:.3f}")
        macro(f"STwoIAE{t}", 100 * r.iae, "{:.2f}")
        if not pd.isna(r.get("ilogloss")):
            macro(f"STwoLL{t}", r.ilogloss, "{:.4f}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2: policy means at $N=8$ averaged over the eight benchmarks: integrated held-out log loss, integrated absolute score error, mean coverage and width of the 90\% score interval, and integrated interval score ($\alpha=0.1$; lower is better). Recalibrated intervals share the estimates of the raw policy.}
\label{tab:study2policies}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Policy & log loss & score IAE & coverage & width & interval score \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/study2_policies.tex", "w").write(tex)

    # coverage by panel size (EOST-M raw and recalibrated) and the recalibrated Rasch-type test per bank
    pmN = tm.groupby(["N", "dataset", "policy"]).mean(numeric_only=True)
    for N, nm in [(4, "Four"), (8, "Eight"), (16, "Sixteen")]:
        for p, t in [("eig:mg", "Eost"), ("eig:mg+R", "EostR"), ("mfi:1pr+R", "RaschR")]:
            v = pmN.loc[N].xs(p, level="policy") if p in pmN.loc[N].index.get_level_values("policy") else None
            if v is not None:
                macro(f"STwoCov{t}N{nm}", float(v.coverage.mean()))
                macro(f"STwoIS{t}N{nm}", float(v.iis.mean()), "{:.3f}")
    rr = pm.xs("mfi:1pr+R", level="policy").coverage if "mfi:1pr+R" in pm.index.get_level_values(1) else None
    if rr is not None:
        macro("STwoRaschRCovMin", float(rr.min()))
        macro("STwoRaschRCovMax", float(rr.max()))
        macro("STwoRaschRCovIn", int(rr.between(0.85, 0.95).sum()), "{:d}")
    er = pm.xs("eig:mg+R", level="policy").coverage
    macro("STwoEostRCovMinBank", NICE[er.idxmin()])
    # benchmark-reduction comparison at fixed budgets
    rows = []
    for p in ["eig:mg", "atlas", "random:mg", "tiny:gpirt", "tiny:pirt", "tiny:anchor", "mince", "naive"]:
        if p not in avg.index:
            continue
        r = avg.loc[p]
        rows.append(f"{POLNAME[p]} & " + " & ".join(f"{100 * r[f'err@{T}']:.2f}" for T in (25, 50, 100)) + r" \\")
        t = "".join(ch for ch in p.replace("1", "One") if ch.isalpha())
        for T, nm in [(25, "TwentyFive"), (50, "Fifty"), (100, "Hundred")]:
            macro(f"STwoErr{t}{nm}", 100 * r[f"err@{T}"], "{:.2f}")
    ns = tm[(tm.N == 8) & (tm.policy == "mince:nstar")]
    if len(ns):
        macro("STwoMinceNstarMed", float(ns.n_star.median()), "{:.0f}")
        macro("STwoMinceErrNstar", 100 * float(ns.err_nstar.mean()), "{:.2f}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2: absolute error of the full-bank score (percentage points) after 25, 50 and 100 items at $N=8$, averaged over targets and benchmarks. tinyBenchmarks and MINCE select their items without adapting to the target.}
\label{tab:reduction}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & 25 items & 50 items & 100 items \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/reduction.tex", "w").write(tex)

    # coverage figure per benchmark
    pols = [("naive", "#6f6e69", "Naive (Wilson)"), ("mfi:pi", "#eb6834", "MFI:pi"), ("eig:mg", "#2a78d6", "EOST-M"),
            ("eig:mg+R", "#123f86", "EOST-M recalibrated"), ("mfi:1pr", "#1baf7a", "Rasch-type"),
            ("mfi:1pr+R", "#0a6b47", "Rasch-type recalibrated")]
    fig, ax = plt.subplots(figsize=(6.9, 2.5))
    xs = np.arange(len(CONF))
    for k, (p, col, lab) in enumerate(pols):
        if p not in pm.index.get_level_values(1):
            continue
        v = [pm.loc[(ds, p), "coverage"] if (ds, p) in pm.index else np.nan for ds in CONF]
        ax.scatter(xs + (k - 2.5) * 0.12, v, color=col, s=16, label=lab, zorder=3)
    ax.axhspan(0.85, 0.95, color="#e4e3df", alpha=0.6, zorder=1)
    ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (1, 1)))
    ax.set_xticks(xs); ax.set_xticklabels([NICE[d] for d in CONF], rotation=20, fontsize=7)
    ax.set_ylabel("coverage of 90% interval"); ax.set_ylim(0.4, 1.0)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/study2_coverage.pdf"); plt.close(fig)

    # equivalence table
    for ef in glob.glob(f"{S2}/equivalence_*.csv"):
        E = pd.read_csv(ef).set_index("dataset")
        rows = [f"{NICE[ds]} & {100 * E.loc[ds, 'rel_diff']:+.2f} & [{100 * E.loc[ds, 'ci_lo']:+.2f}, {100 * E.loc[ds, 'ci_hi']:+.2f}] & "
                f"{'yes' if E.loc[ds, 'equivalent'] else 'no'} \\\\" for ds in CONF if ds in E.index]
        tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2, hypothesis C3: relative difference (\%) of held-out log loss of the posterior-draw marginal (PDM) Bayesian CAT (Bayesian maximum information) with respect to EOST-M at $N=8$, its 90\% bootstrap interval over targets, and equivalence within $\pm1\%$.}
\label{tab:equiv}
\begin{tabular}{@{}lrcc@{}}
\toprule
Benchmark & difference & 90\% interval & equivalent \\
\midrule
""" + "\n".join(rows).replace("+-", "-") + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        open(f"{TAB}/equivalence.tex", "w").write(tex)

    # secondary contrasts of interest
    S = pd.read_csv(f"{S2}/secondary_tests.csv") if os.path.exists(f"{S2}/secondary_tests.csv") else None
    if S is not None:
        def sec(h, N=8):
            s = S[(S.hypothesis == h) & (S.N == N)]
            return s
        for h, nm in [("S:fb:eig|eig:mg|ilogloss", "FbeigVsEost"), ("S:fb:bmi|fb:eig|ilogloss", "BmiVsFbeig"),
                      ("S:eig:xf|eig:mg|ilogloss", "XfVsEost"), ("S:eig:tp|eig:mg|ilogloss", "TpVsEost"),
                      ("S:eig:mg|mfi:1pr|iae", "EostVsRaschIAE"), ("S:eig:mg|mfi:1pr|covgap", "EostVsRaschCov"),
                      ("S:eig:mg+R|mfi:1pr+R|iis", "EostRVsRaschRIS"), ("S:mix:mg|eig:mg|covgap", "MixVsEostCov"),
                      ("S:mix:mg|eig:mg|iae", "MixVsEostIAE"), ("S:eig:mg|atlas|ilogloss", "EostVsAtlasLL"),
                      ("S:eig:mg|random:mg|ilogloss", "EostVsRandomLL")]:
            s = sec(h)
            if s.empty:
                continue
            macro(f"Sec{nm}Med", 100 * s.rel_improvement.median(), "{:.1f}")
            macro(f"Sec{nm}MedAbs", abs(100 * s.rel_improvement.median()), "{:.1f}")
            macro(f"Sec{nm}Fav", int(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum()), "{:d}")
            macro(f"Sec{nm}Base", int(((s.p_holm < 0.05) & (s.mean_diff < 0)).sum()), "{:d}")
        for N, nm in [(4, "Four"), (16, "Sixteen")]:
            for h in ["R2", "C1", "C4"]:
                s = S[(S.hypothesis == h) & (S.N == N)]
                if s.empty:
                    continue
                macro(f"Sec{tag(h)}N{nm}Med", 100 * s.rel_improvement.median(), "{:.1f}")
                macro(f"Sec{tag(h)}N{nm}Fav", int(((s.p_holm < 0.05) & (s.mean_diff > 0)).sum()), "{:d}")

    # secondary components
    comp = d.get("secondary_components", {})
    if "temporal" in comp and os.path.exists(f"{S2}/tests_temporal.csv"):
        T = pd.read_csv(f"{S2}/tests_temporal.csv")
        t8 = T[T.N == 8]
        ph_ = t8.p_holm if "p_holm" in t8 else t8.p
        macro("TempNSig", int(((ph_ < 0.05) & (t8.mean_diff > 0)).sum()), "{:d}")
        macro("TempNAgainst", int(((ph_ < 0.05) & (t8.mean_diff < 0)).sum()), "{:d}")
        macro("TempNTests", int(len(t8)), "{:d}")
        for h in ["R2", "R3", "R4", "C1", "C2", "C4", "C5b", "C6", "C7", "C8", "C9"]:
            s = T[(T.hypothesis == h) & (T.N == 8)]
            if not s.empty:
                macro(f"Temp{tag(h)}Med", 100 * float(s.rel_improvement.iloc[0]), "{:.1f}")
                ph = float(s.p_holm.iloc[0]) if "p_holm" in s else float(s.p.iloc[0])
                macro(f"Temp{tag(h)}P", "<0.001" if ph < 0.001 else f"{ph:.3f}")   # Holm within the temporal tests; use in math mode
                macro(f"Temp{tag(h)}Sig", "yes" if ph < 0.05 else "no")
                macro(f"Temp{tag(h)}Wins", int(s.wins.iloc[0]), "{:d}")
                macro(f"Temp{tag(h)}N", int(s.n.iloc[0]), "{:d}")
        cv = comp["temporal"]["coverage"]
        for p, nm in [("eig:mg", "Eost"), ("eig:mg+R", "EostR"), ("naive", "Naive"), ("mfi:1pr", "Rasch"), ("mfi:1pr+R", "RaschR"), ("random:mg", "Randommg")]:
            k = f"swe_verified|8|{p}"
            if k in cv:
                macro(f"TempCov{nm}", cv[k])
        macro("TempTargets", comp["temporal"]["n_targets"].get("swe_verified", 0), "{:d}")
    if "synthetic" in comp:
        cv = comp["synthetic"]["coverage"]
        for ds, nm in [("synth_2pl", "Twopl"), ("synth_guess", "Guess"), ("synth_2dim", "Twodim")]:
            for p, pn in [("eig:mg", "Eost"), ("eig:mg+R", "EostR")]:
                k = f"{ds}|8|{p}"
                if k in cv:
                    macro(f"Syn{nm}Cov{pn}", cv[k])

    # fully Bayesian diagnostics and cost of study 2
    dg = pd.concat([pd.read_csv(f) for f in glob.glob(f"artifacts/study2/{S2_COMP}/*/diag.csv.gz")], ignore_index=True)
    g = dg[dg.N == 8]
    if "fb_theta_shift_mean" in g:
        macro("FbThetaShift", float(g.fb_theta_shift_mean.median()))
        macro("FbThetaSd", float(g.fb_theta_sd_mean.median()))
    if "sec_fb" in g:
        macro("STwoSecCal", float(g.sec_cal2pl.median()), "{:.1f}")
        macro("STwoSecFb", float(g.sec_fb.median()), "{:.1f}")
        macro("STwoSecLoco", float(g.sec_loco.median()), "{:.1f}")
    secs = sum(json.load(open(f))["cpu_seconds"] for f in glob.glob("artifacts/study2/*/*/config.json")
               if "/dev" not in f)
    macro("StudyTwoHours", secs / 3600 / 2, "{:.0f}")
    macro("StudyTwoCPUHours", secs / 3600, "{:.0f}")
    fx = [json.load(open(f)).get("cpu_seconds", 0.0) for f in glob.glob("artifacts/study2_posthoc/*/*/done.json")]
    if fx:
        macro("StudyTwoFixHours", sum(fx) / 3600 / 2, "{:.1f}")


PH2 = os.environ.get("S2_POSTHOC", "artifacts/study2_posthoc/_analysis")   # override for testing only
FBPOL = [("fb:bmi (frozen)", "FbFrozenBmi", "PDM Bayesian, BMI (frozen, burn-in 10)"),
         ("fb:eig (frozen)", "FbFrozenEig", "PDM Bayesian, EIG (frozen, burn-in 10)"),
         ("fb:bmi", "FbBmi", "PDM Bayesian, BMI (burn-in 60)"),
         ("fb:eig", "FbEig", "PDM Bayesian, EIG (burn-in 60)"),
         ("fbj:bmi", "FbjBmi", "joint fully Bayesian, BMI (exploratory)"),
         ("fbj:eig", "FbjEig", "joint fully Bayesian, EIG (exploratory)")]


def posthoc2():
    """Macros and tables of the post hoc corrections of study 2 (artifacts/study2_posthoc)."""
    f = f"{PH2}/decisions.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    if "tiny" in d:
        t = d["tiny"]
        macro("TinyReproDiff", t["reproduction_max_abs_diff"], "{:.0f}" if t["reproduction_max_abs_diff"] == 0 else "{:.1e}")
        for T, nm in [(25, "TwentyFive"), (50, "Fifty"), (100, "Hundred")]:
            sh = t[f"share_short_T{T}"]
            macro(f"TinyShort{nm}Max", 100 * max(sh.values()), "{:.0f}")
            macro(f"TinyShort{nm}Min", 100 * min(sh.values()), "{:.0f}")
            macro(f"TinyShort{nm}Banks", sum(1 for v in sh.values() if v > 0), "{:d}")
        rows = []
        for ds in CONF1:
            if ds not in t["share_short_T50"]:
                continue
            rows.append(f"{NICE[ds]} & " + " & ".join(f"{100 * t[f'share_short_T{T}'][ds]:.0f}" for T in (25, 50, 100))
                        + f" & {t['median_items_T100'][ds]:.0f} \\\\")
        tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2, post hoc audit of tinyBenchmarks at $N=8$: percentage of cells (target $\times$ replicate) in which fewer anchor items than the budget were available: K-Means returns at most one anchor per distinct item embedding, and with eight calibrators the embeddings take only as many distinct values as there are distinct response patterns among the items, often fewer than 100; last column: median number of anchors actually used at a budget of 100.}
\label{tab:tinyaudit}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{3}{c}{cells with fewer anchors than budget (\%)} & anchors used \\
\cmidrule(lr){2-4}
Benchmark & 25 & 50 & 100 & at budget 100 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        open(f"{TAB}/tiny_audit.tex", "w").write(tex)
        for kind, nm in [("C8_full_budget", "Full"), ("C8_items_matched", "Match")]:
            x = t[kind]
            macro(f"Tiny{nm}Fav", x["favour_eost"], "{:d}")
            macro(f"Tiny{nm}Base", x["favour_tiny"], "{:d}")
            macro(f"Tiny{nm}Med", x["median_rel_improvement_pct"], "{:.1f}")
            macro(f"Tiny{nm}N", x["n_banks"], "{:d}")
    for key, sfx in [("fbfix", ""), ("fbfix_N4", "NFour")]:
        if key not in d:
            continue
        x = d[key]
        rows = []
        for pol, nm, lab in FBPOL:
            if pol not in x:
                continue
            v = x[pol]
            macro(f"{nm}Eq{sfx}", v["n_equivalent"], "{:d}")
            macro(f"{nm}NB{sfx}", v["n_banks"], "{:d}")
            macro(f"{nm}Med{sfx}", v["median_rel_diff_pct"], "{:.2f}")
            macro(f"{nm}MedAbs{sfx}", abs(v["median_rel_diff_pct"]), "{:.2f}")
            macro(f"{nm}EostSig{sfx}", v["eost_better_sig"], "{:d}")
            macro(f"{nm}FbSig{sfx}", v["fb_better_sig"], "{:d}")
            macro(f"{nm}Cov{sfx}", v["mean_coverage"])
            macro(f"{nm}IAEMed{sfx}", v["iae_median_rel_improvement_eost_pct"], "{:.1f}")
            macro(f"{nm}Verdict{sfx}", "met" if v["C3_rule_supported"] else "not met")
            rows.append(f"{lab} & {v['n_equivalent']}/{v['n_banks']} & {signed(v['median_rel_diff_pct'], '{:+.2f}')} & "
                        f"{v['eost_better_sig']}/{v['fb_better_sig']} & {signed(v['iae_median_rel_improvement_eost_pct'])} & "
                        f"{v['iae_eost_better_sig']}/{v['iae_fb_better_sig']} & {v['mean_coverage']:.2f} \\\\")
        if key == "fbfix":
            dr_f, dr_c = x["location_drift_frozen_window"], x["location_drift_corrected"]
            bmax = max(dr_f, key=dr_f.get)
            macro("FbDriftFrozenMax", dr_f[bmax])
            macro("FbDriftFrozenBank", NICE[bmax])
            macro("FbDriftFrozenMedian", float(np.median(list(dr_f.values()))))
            macro("FbDriftCorrMax", max(dr_c.values()))
            macro("FbDriftCorrMedian", float(np.median(list(dr_c.values()))))
            macro("FbEostCov", x["fb:bmi"]["eost_mean_coverage"] if "fb:bmi" in x else float("nan"))
        Nn = x["N"]
        tex = (r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 2, Bayesian CATs against EOST-M at $N=""" + str(Nn) + r"""$ (post hoc correction of the burn-in and exploratory joint model; same targets, panels, holdouts and chains). Equivalence: 90\% bootstrap interval of the relative difference in held-out log loss within $\pm1\%$ (rule of C3: at least six benchmarks); median relative difference of log loss (positive: Bayesian CAT worse); benchmarks with a Holm-significant difference favouring EOST-M / the Bayesian CAT; the same for the score error (median relative improvement of EOST-M); mean coverage of the 90\% score interval (EOST-M: """
               + f"{x['fb:bmi']['eost_mean_coverage']:.2f}" + r""").}
\label{tab:fbfix""" + ("" if Nn == 8 else "four") + r"""}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{held-out log loss} & \multicolumn{2}{c}{score IAE} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Variant & equiv. & median \% & sig. & median \% & sig. & coverage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        open(f"{TAB}/fbfix{'' if Nn == 8 else '_N4'}.tex", "w").write(tex)
    if "signs" in d:
        macro("SignsSig", d["signs"]["total_significant"], "{:d}")
        macro("SignsDisc", d["signs"]["total_discordant"], "{:d}")


def results_md():
    """Append study-2 and post hoc sections to RESULTS.md (written first by make_report.py)."""
    if not os.path.exists("RESULTS.md"):
        return
    L = []
    f = f"{S2}/decisions.json"
    if os.path.exists(f):
        d = json.load(open(f))
        L += ["", "## Study 2 (frozen plan, one target per strict family cluster, N = 8)", "",
              "| Hypothesis | Holm-significant for EOST-M / competitor | median improvement | decision |", "|---|---|---|---|"]
        for h, x in d["decisions"].items():
            if "favour_new" in x:
                L.append(f"| {h}: {x['name']} | {x['favour_new']}/{x['favour_base']} | {x['median_rel_improvement_pct']:+.1f}% | "
                         f"{'supported' if x['supported'] else 'not supported'} |")
            elif "n_equivalent" in x:
                L.append(f"| {h}: {x['name']} | equivalent on {x['n_equivalent']}/{x['n_banks']} | {x['median_rel_diff_pct']:+.2f}% | "
                         f"{'supported' if x['supported'] else 'not supported'} |")
            else:
                L.append(f"| {h}: {x['name']} | inside on {x['n_inside']}/8 | -- | {'supported' if x['supported'] else 'not supported'} |")
    f = "artifacts/study2_posthoc/_analysis/decisions.json"
    if os.path.exists(f):
        d = json.load(open(f))
        L += ["", "## Study 2, post hoc corrections (planned before results; see docs/RESEARCH_LOG.md)", ""]
        if "tiny" in d:
            t = d["tiny"]
            L.append(f"* tinyBenchmarks anchors: reproduction max |diff| = {t['reproduction_max_abs_diff']}; C8 on full-budget cells: "
                     f"{t['C8_full_budget']['favour_eost']}/{t['C8_full_budget']['favour_tiny']} banks "
                     f"(median {t['C8_full_budget']['median_rel_improvement_pct']:+.1f}%); items-matched: "
                     f"{t['C8_items_matched']['favour_eost']}/{t['C8_items_matched']['favour_tiny']} "
                     f"(median {t['C8_items_matched']['median_rel_improvement_pct']:+.1f}%).")
        for key in ("fbfix", "fbfix_N4"):
            if key not in d:
                continue
            for pol, _, _ in FBPOL:
                if pol in d[key]:
                    x = d[key][pol]
                    L.append(f"* N = {d[key]['N']}, {pol} vs EOST-M (log loss): equivalent within 1% on "
                             f"{x['n_equivalent']}/{x['n_banks']}, median {x['median_rel_diff_pct']:+.2f}%, EOST-M better "
                             f"(Holm) on {x['eost_better_sig']}, Bayesian CAT better on {x['fb_better_sig']}; mean coverage "
                             f"{x['mean_coverage']:.2f}.")
        if "signs" in d:
            L.append(f"* Sign check: {d['signs']['total_discordant']} of {d['signs']['total_significant']} Holm-significant "
                     "contrasts (both studies) have a mean difference and a rank-biserial correlation of opposite signs.")
    if L:
        txt = open("RESULTS.md").read()
        marker = "Every qualitative statement of the paper"
        j = txt.find("\n## Study 2 (frozen plan")       # idempotent: drop a block from an earlier run
        if j < 0:
            j = txt.find("\n## Study 2, post hoc corrections")
        if j >= 0:
            k = txt.find(marker)
            txt = txt[:j] + "\n\n" + (txt[k:] if k > j else "")
        i = txt.find(marker)
        txt = (txt[:i] + "\n".join(L) + "\n\n" + txt[i:]) if i >= 0 else txt + "\n".join(L) + "\n"
        open("RESULTS.md", "w").write(txt)


def write():
    with open(f"{GEN}/numbers2.tex", "w") as fh:
        for k, v in sorted(NUM.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"REPORT2_GATE=PASS macros={len(NUM)}")


if __name__ == "__main__":
    env_cluster_macros()
    study1_clusters()
    cost()
    eb_robustness()
    dev_rule()
    fb_mixing()
    study2()
    posthoc2()
    results_md()
    write()
