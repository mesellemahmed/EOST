"""Figures (PDF) and LaTeX tables for the paper, generated from frozen artifacts.

Runs in the container. Reads artifacts/<phase>/... and writes paper/figures/*.pdf,
paper/tables/*.tex and paper/generated/numbers.tex (macros used in the text so
that every number quoted in the manuscript comes from an artifact).
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
from scipy.special import expit  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIG, TAB, GEN = "paper/figures", "paper/tables", "paper/generated"
for d in (FIG, TAB, GEN):
    os.makedirs(d, exist_ok=True)

# Fixed policy -> colour mapping (categorical slots in fixed order; neutral for the model-free baseline)
COL = {"eig:mg": "#2a78d6", "mfi:pi": "#eb6834", "entropy:mg": "#1baf7a", "eig:pi": "#eda100",
       "evr:mg": "#6250d6", "random:mg": "#e87ba4", "naive": "#6f6e69", "pwfi:mg": "#008300",
       "entropy:pi": "#1baf7a", "mfi:mg": "#eb6834", "random:pi": "#e87ba4", "evr:pi": "#6250d6"}
LS = {k: ("-" if k.endswith(":mg") else "--") for k in COL}
LS["naive"] = ":"
LABEL = {"eig:mg": "EIG:mg (EOST-M)", "mfi:pi": "MFI:pi (plug-in CAT)", "entropy:mg": "Entropy:mg",
         "eig:pi": "EIG:pi", "evr:mg": "EVR:mg", "random:mg": "Random:mg", "naive": "Naive",
         "pwfi:mg": "PWFI:mg", "entropy:pi": "Entropy:pi", "mfi:mg": "MFI:mg", "random:pi": "Random:pi",
         "evr:pi": "EVR:pi"}
NICE = {"ollb_gsm8k": "GSM8K", "ollb_arc": "ARC-C", "ollb_hellaswag": "HellaSwag", "ollb_winogrande": "WinoGrande",
        "ollb_mmlu": "MMLU", "swe_verified": "SWE-bench V.", "livecodebench": "LiveCodeBench",
        "cruxeval_o": "CRUXEval-O", "synth_2pl": "Synth. 2PL", "synth_guess": "Synth. guessing",
        "synth_2dim": "Synth. 2-D", "swe_lite": "SWE-bench Lite", "cruxeval_i": "CRUXEval-I",
        "ollb_truthfulqa": "TruthfulQA"}
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
SYN = ["synth_2pl", "synth_guess", "synth_2dim"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NUM = {}  # LaTeX macros


def macro(name, value, fmt="{:.3f}"):
    assert name.isalpha(), f"LaTeX macro name must be letters only: {name}"
    if isinstance(value, str):
        NUM[name] = value
    else:
        txt = fmt.format(value)
        NUM[name] = ("$-$" + txt[1:]) if txt.startswith("-") else txt   # typographic minus


def signed(v, fmt="{:+.1f}"):
    """Signed number for LaTeX tables (math-mode minus; zero printed unsigned)."""
    txt = fmt.format(v)
    if float(txt) == 0.0:
        txt = txt.lstrip("+-")
    return "$" + txt + "$"


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


def fmt_p(p):
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# ------------------------------------------------------------------ theory figure
def fig_theory():
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15))
    h2 = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
    th = np.linspace(-6, 6, 4001)
    # (a) entropy vs EIG for items with predictive p = 0.5 but different a
    for sd, c in [(1.0, "#2a78d6"), (0.5, "#eb6834"), (0.25, "#1baf7a")]:
        pi = norm.pdf(th, 0, sd); pi /= pi.sum()
        a_s = np.linspace(0.05, 6, 120)
        eig = [h2(np.clip(expit(a * th) @ pi, 1e-12, 1 - 1e-12)) - h2(np.clip(expit(a * th), 1e-12, 1 - 1e-12)) @ pi
               for a in a_s]
        ax[0].plot(a_s, eig, color=c, label=rf"EIG, $\sigma_t$={sd}")
    ax[0].axhline(np.log(2), color=INK2, ls=":", lw=1.2, label="entropy (all $a$)")
    ax[0].set(xlabel="discrimination $a$ (item at $b=\\mu_t$)", ylabel="nats", title="(a) Entropy is blind to $a$")
    ax[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45))
    # (b) ERF effective discrimination and information cap
    a_s = np.linspace(0.01, 12, 300)
    for s, c in [(0.25, "#2a78d6"), (0.5, "#eb6834"), (1.0, "#1baf7a")]:
        ax[1].plot(a_s, a_s / np.sqrt(1 + np.pi * a_s ** 2 * s ** 2 / 8), color=c, label=f"$s$={s}")
        ax[1].axhline(np.sqrt(8 / np.pi) / s, color=c, ls=":", lw=0.9)
    ax[1].plot(a_s, a_s, color=INK2, ls="--", lw=1.0, label="plug-in ($s$=0)")
    ax[1].set(xlabel="point-estimated discrimination $\\hat a$", ylabel="$a^{\\rm eff}$ of the ERF",
              title="(b) Calibration uncertainty caps $a$", ylim=(0, 7))
    ax[1].legend(frameon=False, loc="upper left")
    # (c) score-targeted EVR: share of the information term vs posterior sd
    sig = np.linspace(0.02, 1.2, 200)
    for Mp, c in [(10, "#2a78d6"), (50, "#eb6834"), (250, "#1baf7a")]:
        a, P = 1.5, 0.5
        v = P * (1 - P); Pp = a * P * (1 - P)
        cterm = sig ** 2 * Pp * Mp
        info = cterm ** 2 / v
        ax[2].plot(sig, info / (v + 2 * cterm + info), color=c, label=f"$M'$={Mp}")
    ax[2].set(xlabel="posterior sd $\\sigma_t$", ylabel="information share of $R_i$",
              title="(c) EVR: entropy $\\to$ information", ylim=(0, 1))
    ax[2].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{FIG}/theory.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ dataset table
def tab_datasets():
    man = json.load(open("data/processed/MANIFEST.json"))["datasets"]
    rows = []
    oracle = {"ollb_gsm8k": "numeric match", "ollb": "log-lik.\\ choice", "swe": "repository tests",
              "livecodebench": "hidden tests", "cruxeval": "execution"}
    role = {d: "1, 2" for d in CONF}
    role.update({"swe_lite": "dev.", "cruxeval_i": "dev.", "ollb_truthfulqa": "dev."})
    order = CONF + ["swe_lite", "cruxeval_i", "ollb_truthfulqa"]
    agg = {"livecodebench", "cruxeval_o", "cruxeval_i", "ollb_truthfulqa"}
    for d in order:
        m = man[d]
        key = d if d in oracle else next(k for k in oracle if d.startswith(k))
        dom = m['meta']['domain'].split(' (')[0].replace("agentic software engineering", "agentic SE") \
            .replace("math word problems", "math").replace("multiple choice", "multiple choice")
        verdict = "aggregate" if d in agg else "single"
        rows.append(f"{NICE[d]} & {role[d]} & {dom} & {oracle[key]} & {verdict} & "
                    f"{m['n_systems']} & {m['n_families']} & {m['n_items']} & "
                    f"{m['accuracy_range'][0]:.2f}--{m['accuracy_range'][1]:.2f} & "
                    f"{m['items_never_correct']} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{Real response matrices. Verdict: one deterministic evaluation per system and item (single) or a recorded pass rate over several samples binarized at one half (aggregate; TruthfulQA: continuous MC2 score binarized). $J$ systems, $F$ heuristic families, $I$ items; accuracy range across systems; items solved by no system. Role: confirmatory in studies 1 and 2, or development bank used only to design the protocols (Sections~\ref{sec:dev} and~\ref{sec:design2}). SE: software engineering.}
\label{tab:data}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lllllrrrcr@{}}
\toprule
Bank & Role & Domain & Oracle & Verdict & $J$ & $F$ & $I$ & Accuracy & Unsolved \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/datasets.tex", "w").write(tex)
    # distinct systems in the confirmatory study (the 395 leaderboard models are shared by its five banks)
    macro("NSystemsTotal", man["ollb_gsm8k"]["n_systems"] + sum(man[d]["n_systems"] for d in CONF if not d.startswith("ollb")), "{:d}")
    macro("NItemsConf", sum(man[d]["n_items"] for d in CONF), "{:,d}")


# ------------------------------------------------------------------ confirmatory table
def tab_confirmatory(phase="confirm"):
    C = load(phase, "confirmatory_tests.csv")
    dec = load(phase, "decisions.json")
    if C is None:
        return
    N0 = 8
    hy = [("H1", r"H1: EIG:mg vs Entropy:mg (log loss)"), ("H2", r"H2: EIG:mg vs MFI:pi (log loss)"),
          ("H3", r"H3: EIG:mg vs Naive (score IAE)"), ("H4", r"H4: EIG:mg vs MFI:pi ($|\mathrm{cov}-0.9|$)")]
    lines = []
    for hid, title in hy:
        lines.append(r"\multicolumn{8}{@{}l}{\textit{" + title + r"}}\\")
        sub = C[(C.hypothesis == hid) & (C.N == N0)].set_index("dataset")
        for d in CONF:
            if d not in sub.index:
                continue
            r = sub.loc[d]
            mark = r"$\uparrow$" if r.favours_new else (r"$\downarrow$" if r.favours_base else "")
            lines.append(f"\\quad {NICE[d]} & {r.n:.0f} & {r.mean_base:.4f} & {r.mean_new:.4f} & "
                         f"{signed(100 * r.rel_improvement)}\\% & {int(r.wins)}/{int(r.losses)} & "
                         f"{signed(r.rank_biserial, '{:+.2f}')} & {fmt_p(r.p_holm)} {mark}\\\\")
        d_ = dec["decisions"][hid]
        verdict = "supported" if d_["supported"] else "not supported"
        lines.append(r"\multicolumn{8}{@{}r}{\footnotesize decision rule: " +
                     f"{d_['holm_sig_favouring_new']}/{d_['n_benchmarks']} favour EOST-M, "
                     f"{d_['holm_sig_favouring_base']} favour baseline $\\Rightarrow$ \\textbf{{{verdict}}}" + r"}\\[2pt]")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 1 in full at the primary panel size $N=8$. Per-target means over 5 replicates; $n$ targets; improvement is relative to the baseline (positive favours EOST-M); W/L wins/losses over targets; $r_{rb}$ matched-pairs rank-biserial; $p$ two-sided Wilcoxon, Holm-adjusted over the whole family; $\uparrow$/$\downarrow$ significant in favour of EOST-M/baseline.}
\label{tab:confirmfull}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrl@{}}
\toprule
Benchmark & $n$ & Baseline & EOST-M & Improv. & W/L & $r_{rb}$ & $p_{\rm Holm}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/confirmatory_N8_full.tex", "w").write(tex)
    # compact version for the main text: relative improvement per benchmark and hypothesis
    rows = []
    for d in CONF:
        cells = []
        for hid, _ in hy:
            sub = C[(C.hypothesis == hid) & (C.N == N0) & (C.dataset == d)]
            if sub.empty:
                cells.append("--"); continue
            r = sub.iloc[0]
            mark = "*" if r.favours_new else (r"\textsuperscript{$\dagger$}" if r.favours_base else "")
            cells.append(f"{signed(100 * r.rel_improvement)}{mark}")
        n = int(C[(C.hypothesis == "H2") & (C.N == N0) & (C.dataset == d)].n.iloc[0])
        rows.append(f"{NICE[d]} & {n} & " + " & ".join(cells) + r" \\")
    dl = []
    for hid, _ in hy:
        d_ = dec["decisions"][hid]
        dl.append(f"{d_['holm_sig_favouring_new']}/{d_['holm_sig_favouring_base']} " + ("supported" if d_["supported"] else "not supported"))
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Study 1 at the primary panel size $N=8$: relative improvement (\%) of EOST-M over the baseline of each hypothesis (log loss for H1--H2, score error for H3, $|\mathrm{coverage}-0.9|$ for H4); * Holm-significant in favour of EOST-M (none in favour of a baseline). Last row: benchmarks significant for EOST-M / for the baseline, and the frozen decision. Full statistics in Table~\ref{tab:confirmfull}.}
\label{tab:confirm}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrcccc@{}}
\toprule
 & & H1 & H2 & H3 & H4 \\
Benchmark & targets & vs Entropy:mg & vs MFI:pi & vs Naive & vs MFI:pi \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
decision & & """ + " & ".join(dl) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/confirmatory_N8.tex", "w").write(tex)
    words = {"H1": "HOne", "H2": "HTwo", "H3": "HThree", "H4": "HFour"}  # LaTeX macro names cannot hold digits
    for hid in ["H1", "H2", "H3", "H4"]:
        d_ = dec["decisions"][hid]
        w = words[hid]
        macro(f"Dec{w}", "supported" if d_["supported"] else "not supported")
        macro(f"Fav{w}", d_["holm_sig_favouring_new"], "{:d}")
        macro(f"Base{w}", d_["holm_sig_favouring_base"], "{:d}")
        sub = C[(C.hypothesis == hid) & (C.N == N0) & C.dataset.isin(CONF)]
        macro(f"MedImp{w}", 100 * sub.rel_improvement.median(), "{:.1f}")
        macro(f"MinImp{w}", 100 * sub.rel_improvement.min(), "{:.1f}")
        macro(f"MaxImp{w}", 100 * sub.rel_improvement.max(), "{:.1f}")
        macro(f"Pos{w}", int((sub.rel_improvement > 0).sum()), "{:d}")
        macro(f"NB{w}", int(len(sub)), "{:d}")
        for N, nm in [(4, "Nfour"), (64, "Nsixtyfour")]:
            sN = C[(C.hypothesis == hid) & (C.N == N) & C.dataset.isin(CONF)]
            if len(sN):
                macro(f"MedImp{w}{nm}", 100 * sN.rel_improvement.median(), "{:.1f}")
                macro(f"Fav{w}{nm}", int(sN.favours_new.sum()), "{:d}")
                macro(f"NB{w}{nm}", int(len(sN)), "{:d}")
    # all-N summary table
    Ns = sorted(C.N.unique())
    rows = []
    for hid, title in hy:
        cells = []
        for N in Ns:
            s = C[(C.hypothesis == hid) & (C.N == N) & C.dataset.isin(CONF)]
            if len(s) == 0:
                cells.append("--")
                continue
            cells.append(f"{int(s.favours_new.sum())}/{int(s.favours_base.sum())}/{len(s)} "
                         f"({signed(100 * s.rel_improvement.median())}\\%)")
        rows.append(title.split(":")[0] + " & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Confirmatory tests at every panel size: benchmarks with a Holm-significant difference favouring EOST-M / favouring the baseline / tested, and (in parentheses) the median relative improvement across benchmarks.}
\label{tab:allN}
\begin{tabular}{@{}l""" + "c" * len(Ns) + r"""@{}}
\toprule
 & """ + " & ".join(f"$N={n}$" for n in Ns) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/confirmatory_allN.tex", "w").write(tex)


# ------------------------------------------------------------------ curves
def fig_curves(phase="confirm", metric="brier", N0=8, fname="curves_brier_N8.pdf", ylabel="holdout Brier"):
    R = runs(phase)
    if R is None:
        return
    pols = ["naive", "entropy:mg", "mfi:pi", "eig:mg"]
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
           if f"{metric}@{c}" in R.columns]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4), sharex=True)
    for ax, d in zip(axes.ravel(), CONF):
        sub = R[(R.dataset == d) & (R.N == N0)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for p in pols:
            s = sub[sub.policy == p]
            tm = s.groupby("target")[[f"{metric}@{c}" for c in cps]].mean()
            y = tm.mean(axis=0).values
            ax.plot(cps[:len(y)], y, color=COL[p], ls=LS[p], label=LABEL[p])
        ax.set_xscale("log")
        ax.set_title(NICE[d])
        ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    for ax in axes[1]:
        ax.set_xlabel("items administered $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}")
    plt.close(fig)


def fig_heatmap(phase="confirm"):
    C = load(phase, "confirmatory_tests.csv")
    S = load(phase, "secondary_tests.csv")
    if C is None:
        return
    Ns = sorted(C.N.unique())
    panels = [("H2", C[C.hypothesis == "H2"], "(a) log loss: EIG:mg vs MFI:pi"),
              ("IAE", S[(S.new == "eig:mg") & (S.base == "mfi:pi") & (S.metric == "iae")],
               "(b) score IAE: EIG:mg vs MFI:pi"),
              ("H1", C[C.hypothesis == "H1"], "(c) log loss: EIG:mg vs Entropy:mg")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f4f3ef", "#1c5cab"])
    for ax, (key, df, title) in zip(axes, panels):
        M = np.full((len(CONF), len(Ns)), np.nan)
        sig = np.zeros_like(M, dtype=bool)
        for i, d in enumerate(CONF):
            for j, N in enumerate(Ns):
                r = df[(df.dataset == d) & (df.N == N)]
                if len(r):
                    M[i, j] = 100 * r.rel_improvement.values[0]
                    sig[i, j] = r.p_holm.values[0] < 0.05
        lim = max(1.0, np.nanpercentile(np.abs(M), 95))
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.1f}".replace("-", "\u2212") + ("*" if sig[i, j] else ""),
                            ha="center", va="center",
                            fontsize=6.3, color=INK)
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_yticks(range(len(CONF)))
        ax.set_yticklabels([NICE[d] for d in CONF] if ax is axes[0] else [])
        ax.set_xlabel("panel size $N$"); ax.set_title(title, fontsize=8)
        ax.grid(False)
    fig.text(0.5, -0.04, "cell = relative improvement of EIG:mg in %, * = Holm-adjusted $p<0.05$; "
             "blue favours EIG:mg, orange the baseline", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/heatmap_improvement.pdf")
    plt.close(fig)


def fig_coverage(phase="confirm"):
    tm = load(phase, "target_means.csv.gz")
    if tm is None:
        return
    pols = ["naive", "random:mg", "mfi:pi", "eig:pi", "eig:mg"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4))
    for ax, group, title in [(axes[0], CONF, "(a) real benchmarks (mean over 8)"),
                             (axes[1], SYN, "(b) synthetic banks (mean over 3)")]:
        sub = tm[tm.dataset.isin(group)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for p in pols:
            s = sub[sub.policy == p].groupby(["dataset", "N"]).coverage.mean().reset_index()
            m = s.groupby("N").coverage.mean()
            ax.plot(m.index, m.values, color=COL[p], ls=LS[p], marker="o", ms=3.5, label=LABEL[p])
        ax.axhline(0.9, color=INK, lw=0.9, ls=(0, (1, 1)))
        ax.annotate("nominal 0.90", xy=(64, 0.9), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=6.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="panel size $N$", ylabel="coverage of 90% interval for $A$", title=title)
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{FIG}/coverage.pdf")
    plt.close(fig)


def fig_diagnostics(phase="confirm"):
    D = diag(phase)
    if D is None:
        return
    D = D[D.dataset.isin(CONF)]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))
    for d in CONF:
        s = D[D.dataset == d].groupby("N").mean(numeric_only=True)
        kw = dict(color="#9a9893", lw=0.9)
        axes[0].plot(s.index, s.frac_items_extreme, **kw)
        axes[1].plot(s.index, s.median_a_map, color="#eb6834", lw=0.9, alpha=0.7)
        axes[1].plot(s.index, s.median_a_eff_mg, color="#2a78d6", lw=0.9, alpha=0.7)
        axes[2].plot(s.index, 100 * s.frac_a_gt_4, **kw)
    m = D.groupby("N").mean(numeric_only=True)
    axes[0].plot(m.index, m.frac_items_extreme, color=INK, lw=2)
    axes[1].plot(m.index, m.median_a_map, color="#eb6834", lw=2.2, label="plug-in MAP $\\hat a$")
    axes[1].plot(m.index, m.median_a_eff_mg, color="#2a78d6", lw=2.2, label="ERF $a^{\\rm eff}$")
    axes[2].plot(m.index, 100 * m.frac_a_gt_4, color=INK, lw=2)
    axes[0].set(title="(a) items with identical panel verdicts", ylabel="fraction of items")
    axes[1].set(title="(b) median discrimination", ylabel="$a$")
    axes[2].set(title="(c) inflated plug-in discriminations", ylabel="items with $\\hat a>4$ (%)")
    axes[1].legend(frameon=False, fontsize=6.8)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("panel size $N$")
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    fig.tight_layout()
    fig.savefig(f"{FIG}/calibration_diagnostics.pdf")
    plt.close(fig)
    macro("FracExtremeNfour", 100 * m.loc[4, "frac_items_extreme"], "{:.0f}")
    macro("FracExtremeNsixtyfour", 100 * m.loc[64, "frac_items_extreme"], "{:.0f}")
    macro("AmapNfour", m.loc[4, "median_a_map"], "{:.2f}")
    macro("AeffNfour", m.loc[4, "median_a_eff_mg"], "{:.2f}")
    macro("NeffRatioNeight", m.loc[8, "n_eff"] / 8, "{:.2f}")
    big = D[D.dataset.isin(["ollb_mmlu", "ollb_hellaswag"]) & (D.N == 64)]
    if len(big):
        macro("CalibSecNsixtyfour", float(big.calib_seconds.median()), "{:.1f}")
    macro("FracAgtFourNfour", 100 * m.loc[4, "frac_a_gt_4"], "{:.0f}")
    g8 = D[D.N == 8].groupby("dataset").frac_items_extreme.mean()
    if "ollb_gsm8k" in g8:
        macro("GsmFracExtremeNeight", 100 * g8["ollb_gsm8k"], "{:.0f}")
        macro("OtherFracExtremeNeight", 100 * g8.drop("ollb_gsm8k").median(), "{:.0f}")
        z = np.load("data/processed/ollb_gsm8k.npz")
        macro("GsmFloorFrac", 100 * float((z["Y"].mean(1) < 0.05).mean()), "{:.0f}")
    macro("FracAgtFourNsixtyfour", 100 * m.loc[64, "frac_a_gt_4"], "{:.0f}")


def fig_synthetic(phase="confirm"):
    tm = load(phase, "target_means.csv.gz")
    if tm is None or "itrue_err" not in tm:
        return
    pols = ["entropy:mg", "mfi:pi", "eig:pi", "eig:mg"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), sharex=True)
    for ax, d in zip(axes, SYN):
        sub = tm[tm.dataset == d]
        for p in pols:
            m = sub[sub.policy == p].groupby("N").itrue_err.mean()
            ax.plot(m.index, m.values, color=COL[p], ls=LS[p], marker="o", ms=3, label=LABEL[p])
        ax.set_xscale("log", base=2); ax.set_title(NICE[d]); ax.set_xlabel("panel size $N$")
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    axes[0].set_ylabel(r"mean $|\hat\theta_t-\theta^\star|$")
    g = tm[tm.dataset.isin(SYN)].groupby(["N", "policy"])[["itrue_err", "true_theta_cov", "coverage", "ilogloss", "iae"]].mean()
    for N, nm in [(4, "Nfour"), (8, "Neight"), (64, "Nsixtyfour")]:
        for p, pn in [("eig:mg", "EigMg"), ("mfi:pi", "MfiPi"), ("eig:pi", "EigPi"), ("entropy:mg", "EntMg")]:
            if (N, p) in g.index:
                macro(f"Syn{pn}Err{nm}", g.loc[(N, p), "itrue_err"], "{:.3f}")
                macro(f"Syn{pn}ThCov{nm}", g.loc[(N, p), "true_theta_cov"], "{:.2f}")
                macro(f"Syn{pn}Cov{nm}", g.loc[(N, p), "coverage"], "{:.2f}")
    # per synthetic bank (the well-specified bank is reported separately in the text)
    gb = tm[tm.dataset.isin(SYN)].groupby(["dataset", "N", "policy"])[["itrue_err", "true_theta_cov", "coverage"]].mean()
    tags = {"synth_2pl": "Twopl", "synth_guess": "Guess", "synth_2dim": "Twodim"}
    for d, tg in tags.items():
        for N, nm in [(4, "Nfour"), (8, "Neight"), (64, "Nsixtyfour")]:
            for p, pn in [("eig:mg", "EigMg"), ("mfi:pi", "MfiPi"), ("eig:pi", "EigPi"), ("entropy:mg", "EntMg"),
                          ("naive", "Naive")]:
                if (d, N, p) in gb.index:
                    r = gb.loc[(d, N, p)]
                    macro(f"Syn{tg}{pn}Cov{nm}", r["coverage"], "{:.2f}")
                    if p != "naive":
                        macro(f"Syn{tg}{pn}Err{nm}", r["itrue_err"], "{:.3f}")
                        macro(f"Syn{tg}{pn}ThCov{nm}", r["true_theta_cov"], "{:.2f}")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/synthetic_theta.pdf")
    plt.close(fig)


def tab_policies(phase="confirm", N0=8):
    tm = load(phase, "target_means.csv.gz")
    if tm is None:
        return
    pols = ["naive", "random:pi", "random:mg", "entropy:pi", "entropy:mg", "mfi:pi", "mfi:mg", "pwfi:mg",
            "eig:pi", "eig:mg", "evr:pi", "evr:mg"]
    sub = tm[(tm.N == N0) & tm.dataset.isin(CONF)]
    # rank policies within each benchmark, then average ranks (lower = better)
    out = []
    for met, lower in [("ilogloss", True), ("iae", True), ("covgap", True), ("ttol_03", True)]:
        per = sub.groupby(["dataset", "policy"])[met].mean().unstack()
        per = per[[p for p in pols if p in per]]
        if met == "ilogloss":
            per = per.drop(columns=["naive"], errors="ignore")
        ranks = per.rank(axis=1, ascending=lower)
        out.append((met, per.mean(axis=0), ranks.mean(axis=0)))
    rows = []
    for p in pols:
        cells = []
        for met, mean, rk in out:
            if p in mean.index:
                cells.append(f"{mean[p]:.4f} ({rk[p]:.1f})" if met != "ttol_03" else f"{mean[p]:.1f} ({rk[p]:.1f})")
            else:
                cells.append("--")
        name = LABEL[p].replace(" (EOST-M)", r" \textbf{(EOST-M)}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{All twelve policies at $N=8$, averaged over the eight confirmatory benchmarks (mean of per-benchmark means; in parentheses the mean rank across benchmarks, 1 = best). IAE: integrated absolute error of the full-bank score; $|\mathrm{cov}-0.9|$: miscalibration of the 90\% interval; $t_{0.03}$: sustained items-to-tolerance at $\varepsilon=0.03$ (101 = never).}
\label{tab:policies}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lllll@{}}
\toprule
Policy & holdout log loss & score IAE & $|\mathrm{cov}-0.9|$ & $t_{0.03}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/policies_N8.tex", "w").write(tex)
    m_iae = out[1][1]
    macro("IAEeigmgNeight", m_iae["eig:mg"], "{:.4f}")
    macro("IAEnaiveNeight", m_iae["naive"], "{:.4f}")
    macro("IAEmfipiNeight", m_iae["mfi:pi"], "{:.4f}")
    macro("IAEredNaive", 100 * (1 - m_iae["eig:mg"] / m_iae["naive"]), "{:.0f}")
    macro("IAErandmgNeight", m_iae["random:mg"], "{:.4f}")
    info_mg = ["mfi:mg", "pwfi:mg", "eig:mg", "evr:mg"]
    m_ll = out[0][1]
    macro("InfoSpreadLL", 100 * (m_ll[info_mg].max() / m_ll[info_mg].min() - 1), "{:.1f}")
    macro("InfoSpreadIAE", 100 * (m_iae[info_mg].max() / m_iae[info_mg].min() - 1), "{:.0f}")
    macro("EntVsInfoLL", 100 * (m_ll["entropy:mg"] / m_ll[info_mg].mean() - 1), "{:.1f}")
    macro("IAEredRandmg", 100 * (1 - m_iae["eig:mg"] / m_iae["random:mg"]), "{:.0f}")
    S = load(phase, "secondary_tests.csv")
    if S is not None:
        s8 = S[(S.new == "eig:mg") & (S.base == "random:mg") & (S.metric == "iae") & (S.N == N0) & S.dataset.isin(CONF)]
        macro("RandIaeFav", int(((s8.p_holm < 0.05) & (s8.mean_diff > 0)).sum()), "{:d}")
        macro("RandIaeNB", int(len(s8)), "{:d}")
        macro("RandIaeMed", 100 * float(s8.rel_improvement.median()), "{:.1f}")   # median over banks, like MedImpHThree
    t03 = out[3][1]
    macro("TtolEigmg", t03["eig:mg"], "{:.0f}")
    macro("TtolNaive", t03["naive"], "{:.0f}")
    cg = out[2][1]
    covm = sub.groupby(["dataset", "policy"]).coverage.mean().unstack().mean(axis=0)
    macro("CovEigmg", covm["eig:mg"], "{:.2f}")
    macro("CovMfipi", covm["mfi:pi"], "{:.2f}")
    macro("CovNaive", covm["naive"], "{:.2f}")
    macro("CovRandmg", covm["random:mg"], "{:.2f}")
    macro("CovEntmg", covm["entropy:mg"], "{:.2f}")


def tab_sensitivity():
    rows = []
    variants = [("confirm", "main (EB prior, 2PL)"), ("sens_fixedprior", "fixed item prior"),
                ("sens_dedup", "near-duplicates removed"),
                ("sens_1pl", r"one-parameter$^{\dagger}$"),
                ("sens_1pl_reg", r"one-parameter, regularized$^{\ddagger}$")]
    vnames = {"confirm": "Main", "sens_1pl": "Onepl", "sens_fixedprior": "Fixedprior", "sens_dedup": "Dedup",
              "sens_1pl_reg": "Oneplreg"}
    for phase, name in variants:
        tm = load(phase, "target_means.csv.gz")
        if tm is None:
            continue
        tm = tm[tm.dataset.isin(CONF)]
        if phase in ("sens_1pl", "sens_1pl_reg") and rows and not rows[-1].startswith(r"\midrule"):
            rows.append(r"\midrule")
        for N in [8, 64]:
            s = tm[tm.N == N]
            if s.empty:
                continue
            per = s.groupby(["dataset", "policy"])[["ilogloss", "iae", "covgap", "coverage"]].mean().unstack()

            def rel(met, new, base, _phase=phase, _N=N):
                v = 100 * (1 - per[met][new] / per[met][base])
                hname = {("ilogloss", "entropy:mg"): "HOne", ("ilogloss", "mfi:pi"): "HTwo",
                         ("iae", "naive"): "HThree", ("covgap", "mfi:pi"): "HFour"}[(met, base)]
                nname = {8: "Neight", 64: "Nsixtyfour"}[_N]
                macro(f"Sens{vnames[_phase]}{hname}{nname}", float(v.median()), "{:+.1f}")
                macro(f"Sens{vnames[_phase]}{hname}{nname}Pos", int((v > 0).sum()), "{:d}")
                macro(f"Sens{vnames[_phase]}{hname}{nname}B", int(len(v)), "{:d}")
                return f"{signed(v.median())} [{(v > 0).sum()}/{len(v)}]"
            if N == 8 and "entropy:pi" in per["ilogloss"]:
                vpi = 100 * (1 - per["ilogloss"]["eig:pi"] / per["ilogloss"]["entropy:pi"])
                macro(f"Sens{vnames[phase]}HOnePiNeight", float(vpi.median()), "{:+.1f}")
            cov = float(per["coverage"]["eig:mg"].mean())
            macro(f"SensCov{vnames[phase]}{ {8: 'Neight', 64: 'Nsixtyfour'}[N] }", cov, "{:.2f}")
            rows.append(f"{name} & {N} & {rel('ilogloss', 'eig:mg', 'entropy:mg')} & {rel('ilogloss', 'eig:mg', 'mfi:pi')} & "
                        f"{rel('iae', 'eig:mg', 'naive')} & {rel('covgap', 'eig:mg', 'mfi:pi')} & {cov:.2f} \\\\")
        if phase == "sens_dedup":
            # the main analysis restricted to the same benchmarks, for a like-for-like comparison
            tmm = load("confirm", "target_means.csv.gz")
            tmm = tmm[tmm.dataset.isin(set(tm.dataset)) & (tmm.N == 8)]
            pm = tmm.groupby(["dataset", "policy"]).ilogloss.mean().unstack()
            macro("SensMainDedupBanksHTwoNeight", float((100 * (1 - pm["eig:mg"] / pm["mfi:pi"])).median()), "{:+.1f}")
            macro("SensDedupNBanks", int(tm.dataset.nunique()), "{:d}")
        d = diag(phase)
        if d is not None and phase in ("sens_1pl", "sens_1pl_reg"):
            d = d[d.dataset.isin(CONF)]
            def sci(v):
                e = int(np.floor(np.log10(v)))
                return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$" if e >= 3 else f"{v:.1f}"
            macro(f"{vnames[phase]}MedianA", sci(float(d.median_a_map.median())))
            macro(f"{vnames[phase]}MaxA", sci(float(d.median_a_map.max())))
    if not rows:
        return
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Sensitivity analyses (30 targets, 3 replicates; main analysis for reference). Median relative improvement of EIG:mg across confirmatory benchmarks, in \%, [benchmarks improved / tested], and mean coverage of EOST-M's 90\% score interval. $^{\dagger}$Specified in the frozen plan, but its joint MAP is degenerate (the common discrimination diverges; Section~\ref{sec:sensitivity}), so its contrasts are not interpreted. $^{\ddagger}$Post hoc: common discrimination with prior $\log a\sim\mathcal N(0,0.05^2)$.}
\label{tab:sens}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrlllll@{}}
\toprule
Variant & $N$ & H1 log loss & H2 log loss & H3 IAE & H4 cov.\ gap & cov. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/sensitivity.tex", "w").write(tex)


def tab_development():
    rows = []
    for phase, prior in [("dev_fixedprior", "fixed"), ("dev_eb", "EB")]:
        tm = load(phase, "target_means.csv.gz")
        if tm is None:
            continue
        for N in [4, 8, 64]:
            s = tm[tm.N == N].groupby(["dataset", "policy"])[["ilogloss", "iae", "coverage"]].mean().unstack()
            for d in ["synth_2pl", "swe_lite", "cruxeval_i", "ollb_truthfulqa"]:
                if d not in s.index:
                    continue
                r = s.loc[d]
                rows.append(f"{NICE[d]} & {prior} & {N} & {r['ilogloss']['eig:mg']:.4f} & {r['ilogloss']['mfi:pi']:.4f} & "
                            f"{r['iae']['eig:mg']:.4f} & {r['iae']['mfi:pi']:.4f} & {r['coverage']['eig:mg']:.2f} & "
                            f"{r['coverage']['mfi:pi']:.2f} \\\\")
    if not rows:
        return
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{Development phase (exploratory; 30 targets, 3 replicates). The switch from fixed to empirical-Bayes (EB) item hyperpriors was decided here, before freezing the confirmatory plan.}
\label{tab:dev}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{holdout log loss} & \multicolumn{2}{c}{score IAE} & \multicolumn{2}{c}{coverage} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Bank & prior & $N$ & EIG:mg & MFI:pi & EIG:mg & MFI:pi & EIG:mg & MFI:pi \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/development.tex", "w").write(tex)


def tab_full_results(phase="confirm"):
    tm = load(phase, "target_means.csv.gz")
    if tm is None:
        return
    pols = ["naive", "entropy:mg", "mfi:pi", "eig:pi", "eig:mg", "evr:mg"]
    lines = []
    for d in CONF + SYN:
        s = tm[tm.dataset == d]
        if s.empty:
            continue
        for N in sorted(s.N.unique()):
            g = s[s.N == N].groupby("policy")[["ilogloss", "iae", "coverage"]].mean()
            cells = []
            for p in pols:
                if p in g.index:
                    ll = "--" if p == "naive" else f"{g.loc[p, 'ilogloss']:.3f}".lstrip("0")
                    cells.append(f"{ll}/{g.loc[p, 'iae']:.3f}".replace("/0.", "/.") + f"/{g.loc[p, 'coverage']:.2f}".replace("/0.", "/."))
                else:
                    cells.append("--")
            lines.append(f"{NICE[d]} & {N} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1pt]")
    tex = r"""\begin{table}[p]
\centering\scriptsize
\caption{Full results: holdout log loss / score IAE / coverage of the 90\% interval (per-target means, then averaged over targets; leading zeros omitted).}
\label{tab:full}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lr""" + "l" * len(pols) + r"""@{}}
\toprule
Bank & $N$ & """ + " & ".join(LABEL[p].replace(" (EOST-M)", "") for p in pols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/full_results.tex", "w").write(tex)


def tab_efficiency(phase="confirm"):
    """Items needed by each policy to match the score error of 100 random items (naive),
    at the primary panel size N=8 and at the largest panel available for the bank."""
    R = runs(phase)
    if R is None:
        return
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if f"err@{c}" in R]
    pols = ["entropy:mg", "mfi:pi", "eig:mg", "evr:mg"]
    rows, ratios, items_needed = [], {8: [], "max": []}, {8: [], "max": []}
    for d in CONF:
        sub_d = R[R.dataset == d]
        if sub_d.empty:
            continue
        nmax = int(sub_d.N.max())
        for N, tag in [(8, 8), (nmax, "max")]:
            sub = sub_d[sub_d.N == N]
            tm = sub.groupby(["policy", "target"])[[f"err@{c}" for c in cps]].mean().groupby("policy").mean()
            ref = tm.loc["naive", "err@100"]
            cells = []
            for p in pols:
                hit = [c for c, v in zip(cps, tm.loc[p].values) if v <= ref]
                k = hit[0] if hit else None
                cells.append(f"{k}" if k else "$>$100")
                if p == "eig:mg":
                    ratios[tag].append(100 / k if k else 1.0)
                    items_needed[tag].append(k if k else 101)
            name = NICE[d] if tag == 8 else ""
            rows.append(f"{name} & {N} & {ref:.4f} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Items needed to match the full-bank score error of 100 uniformly sampled items (Naive): first checkpoint $t$ at which the mean $|\hat A_t-A|$ over targets falls below the Naive error at $t=100$, at the primary panel size $N=8$ and at the largest panel available for the benchmark.}
\label{tab:efficiency}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Benchmark & $N$ & Naive error @100 & Entropy:mg & MFI:pi & EIG:mg & EVR:mg \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/efficiency.tex", "w").write(tex)
    for tag, nm in [(8, "Neight"), ("max", "Nmax")]:
        if ratios[tag]:
            macro(f"EffRatioMedian{nm}", float(np.median(ratios[tag])), "{:.1f}")
            macro(f"EffRatioMin{nm}", float(np.min(ratios[tag])), "{:.1f}")
            macro(f"EffRatioMax{nm}", float(np.max(ratios[tag])), "{:.1f}")
            macro(f"EffItemsMedian{nm}", float(np.median(items_needed[tag])), "{:g}")   # e.g. 22.5, as in the cost section
            macro(f"EffItemsMin{nm}", float(np.min(items_needed[tag])), "{:.0f}")
            mx = int(np.max(items_needed[tag]))
            macro(f"EffItemsMax{nm}", "more than 100" if mx > 100 else str(mx))


def curves_macros(phase="confirm", N0=8):
    """Facts about the score-error curves quoted in Section 6.4 (all from runs at N=8)."""
    R = runs(phase)
    if R is None:
        return
    man = json.load(open("data/processed/MANIFEST.json"))["datasets"]
    R = R[(R.N == N0) & R.dataset.isin(CONF)]
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if f"err@{c}" in R]
    head, early_plugin, plateau_pi, plateau_mg, final_better, ent_best, ent_not = {}, 0, [], [], 0, [], []
    for d, g in R.groupby("dataset"):
        tm = g.groupby(["policy", "target"])[[f"err@{c}" for c in cps]].mean().groupby("policy").mean()
        head[d] = float(tm.loc[["entropy:mg", "mfi:pi", "eig:mg"], "err@1"].max() / tm.loc["naive", "err@1"])
        early_plugin += int(min(tm.loc["mfi:pi", "err@5"], tm.loc["eig:pi", "err@5"]) < tm.loc["eig:mg", "err@5"])
        plateau_pi.append(tm.loc["mfi:pi", "err@100"] / tm.loc["mfi:pi", "err@25"])
        plateau_mg.append(tm.loc["eig:mg", "err@100"] / tm.loc["eig:mg", "err@25"])
        final_better += int(tm.loc["eig:mg", "err@100"] < tm.loc["mfi:pi", "err@100"])
        best = tm["err@100"].drop("naive").idxmin()
        evr_lt_eig = bool(tm.loc["evr:mg", "err@100"] < tm.loc["eig:mg", "err@100"])
        (ent_best if best == "entropy:mg" else ent_not).append((man[d]["n_items"], d, evr_lt_eig))
    macro("HeadStartGsm", 100 * head.get("ollb_gsm8k", np.nan), "{:.0f}")
    others = [v for k, v in head.items() if k != "ollb_gsm8k"]
    macro("HeadStartOtherMax", 100 * max(others), "{:.0f}")
    macro("HeadStartOtherMin", 100 * min(others), "{:.0f}")
    macro("EarlyPluginBetter", early_plugin, "{:d}")
    macro("PlateauPi", 100 * (1 - np.median(plateau_pi)), "{:.0f}")
    macro("PlateauMg", 100 * (1 - np.median(plateau_mg)), "{:.0f}")
    macro("FinalEigBelowMfi", final_better, "{:d}")
    macro("NCurveBanks", len(head), "{:d}")
    macro("EntLateBest", len(ent_best), "{:d}")
    names = [NICE[d] for _, d, _ in sorted(ent_best)]
    macro("EvrBelowEigEntBest", sum(e for _, _, e in ent_best), "{:d}")
    macro("EvrBelowEigEntNot", sum(e for _, _, e in ent_not), "{:d}")
    macro("EntLateNot", len(ent_not), "{:d}")
    macro("EntLateBestList", ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else "none")
    if ent_best:
        macro("EntLateBestMaxItems", max(i for i, _, _ in ent_best), "{:,d}")
        macro("EntLateBestMinItems", min(i for i, _, _ in ent_best), "{:,d}")
    if ent_not:
        macro("EntLateNotMinItems", min(i for i, _, _ in ent_not), "{:,d}")
        macro("EntLateNotMaxItems", max(i for i, _, _ in ent_not), "{:,d}")
    json.dump({"head_start_ratio": head, "ent_best_final": ent_best, "ent_not_best_final": ent_not},
              open(f"{GEN}/curves_facts.json", "w"), indent=1)


def fig_crossover(phase="confirm"):
    """Exploratory test of Proposition 4's regime change: the checkpoint after which entropy
    selection (marginal model) has a lower score error than EIG:mg at every later checkpoint,
    as a function of the bank size, for every panel size."""
    R = runs(phase)
    if R is None:
        return
    from scipy.stats import spearmanr
    man = json.load(open("data/processed/MANIFEST.json"))["datasets"]
    R = R[R.dataset.isin(CONF) & R.policy.isin(["entropy:mg", "eig:mg"])]
    cps = [c for c in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] if f"err@{c}" in R]
    rows = []
    for (d, N), g in R.groupby(["dataset", "N"]):
        tm = g.groupby(["policy", "target"])[[f"err@{c}" for c in cps]].mean().groupby("policy").mean()
        below = (tm.loc["entropy:mg"] <= tm.loc["eig:mg"]).values
        cross = None
        for k in range(len(cps)):
            if below[k:].all():
                cross = cps[k]
                break
        rows.append({"dataset": d, "N": int(N), "n_items": man[d]["n_items"], "crossover": cross,
                     "ratio_final": float(tm.loc["entropy:mg", "err@100"] / tm.loc["eig:mg", "err@100"])})
    X = pd.DataFrame(rows)
    X.to_csv(f"{GEN}/crossover.csv", index=False)
    X["cross_plot"] = X.crossover.fillna(125)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    shades = {4: "#0d366b", 8: "#1c5cab", 16: "#3987e5", 32: "#86b6ef", 64: "#b7d3f6"}
    rng = np.random.default_rng(0)
    for N, g in X.groupby("N"):
        jitter = np.exp(rng.uniform(-0.035, 0.035, len(g)))
        ax.scatter(g.n_items * jitter, g.cross_plot, s=22, color=shades.get(int(N), "#2a78d6"), edgecolor="white",
                   linewidth=0.6, label=f"$N$={int(N)}", zorder=3)
    ax.axhspan(110, 140, color="#f4f3ef", zorder=1)
    ax.text(462, 125, "none within\n100 items", fontsize=6.0, color=INK2, va="center", linespacing=0.9)
    ax.set_xscale("log")
    ax.set_xticks([500, 800, 1000, 1300, 2000]); ax.set_xticklabels(["500", "800", "1000", "1300", "2000"])
    ax.minorticks_off()
    ax.set(xlabel="bank size $I$ (items)", ylabel="crossover item $t$", ylim=(0, 140))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=6.2, ncol=1, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/crossover.pdf")
    plt.close(fig)
    rho = spearmanr(X.n_items, X.crossover.fillna(101)).statistic
    macro("CrossRho", rho, "{:.2f}")
    macro("CrossPoints", len(X), "{:d}")
    x8 = X[X.N == 8].sort_values("n_items")
    parts = [f"{NICE[r.dataset]} ({r.n_items:,d} items) after {int(r.crossover)}" for r in x8.itertuples()
             if pd.notna(r.crossover)]
    none = [NICE[r.dataset] for r in x8.itertuples() if pd.isna(r.crossover)]
    macro("CrossListNeight", "; ".join(parts) if parts else "on no benchmark")
    macro("CrossNoneNeight", ", ".join(none[:-1]) + (" and " if len(none) > 1 else "") + none[-1] if none else "none")
    macro("CrossNNoneNeight", len(none), "{:d}")
    never = sorted({(man[d]["n_items"], d) for d in X.dataset.unique() if X[X.dataset == d].crossover.isna().all()})
    nm = [f"{NICE[d]} ({i:,d} items)" for i, d in never]
    macro("CrossNeverList", ", ".join(nm[:-1]) + (" and " if len(nm) > 1 else "") + nm[-1] if nm else "no benchmark")
    small = X[X.n_items <= 1172]
    large = X[X.n_items > 1172]
    macro("CrossShareSmall", 100 * small.crossover.notna().mean(), "{:.0f}")
    macro("CrossShareLarge", 100 * large.crossover.notna().mean(), "{:.0f}")


def tab_posthoc():
    """Post hoc experiment PH1 (strong baselines; not in the frozen plan)."""
    f = "artifacts/posthoc_baselines/_analysis/contrasts.csv"
    if not os.path.exists(f):
        return
    C = pd.read_csv(f)
    dec = json.load(open("artifacts/posthoc_baselines/_analysis/decisions.json"))
    T = pd.read_csv("artifacts/posthoc_baselines/_analysis/target_means.csv.gz")
    cov8 = T[T.N == 8].groupby(["dataset", "policy"]).coverage.mean().unstack().mean(axis=0)
    names = {"mfi:pi": "plug-in 2PL (MFI:pi)", "efi:mg": "expected Fisher information (EFI:mg)",
             "mfi:eap": "plug-in posterior-mean 2PL, MFI", "eig:eap": "plug-in posterior-mean 2PL, EIG",
             "mfi:1pr": "plug-in regularized 1PL, MFI", "eig:1prmg": "marginal regularized 1PL, EIG"}
    tags = {"mfi:pi": "Mfipi", "efi:mg": "Efi", "mfi:eap": "Mfieap", "eig:eap": "Eigeap", "mfi:1pr": "Mfionepr",
            "eig:1prmg": "Eigoneprmg"}
    rows = [r"EOST-M (EIG:mg), reference & -- & -- & -- & -- & -- & " + f"{cov8['eig:mg']:.2f}" + r" \\",
            r"\midrule"]
    macro("PhCovEigmg", float(cov8["eig:mg"]), "{:.2f}")
    for base in ["mfi:pi", "mfi:eap", "eig:eap", "mfi:1pr", "eig:1prmg", "efi:mg"]:
        for met in ["ilogloss", "iae", "covgap"]:          # macros for every metric and panel size
            for N in sorted(C.N.unique()):
                sub = C[(C.base == base) & (C.metric == met) & (C.N == N)]
                if sub.empty:
                    continue
                nm = {"ilogloss": "LL", "iae": "IAE", "covgap": "Cov"}[met] + \
                    {4: "Nfour", 8: "Neight", 16: "Nsixteen"}[int(N)]
                macro(f"Ph{tags[base]}{nm}", 100 * sub.rel_improvement.median(), "{:+.1f}")
                macro(f"Ph{tags[base]}{nm}Abs", abs(100 * sub.rel_improvement.median()), "{:.1f}")
                macro(f"Ph{tags[base]}{nm}Fav", int(sub.eost_better.sum()), "{:d}")
                macro(f"Ph{tags[base]}{nm}Base", int(sub.base_better.sum()), "{:d}")
                macro(f"Ph{tags[base]}{nm}Pos", int((sub.rel_improvement > 0).sum()), "{:d}")
                macro(f"Ph{tags[base]}{nm}B", int(len(sub)), "{:d}")
        cells = []
        for met, N in [("ilogloss", 4), ("ilogloss", 8), ("ilogloss", 16), ("iae", 8), ("covgap", 8)]:
            sub = C[(C.base == base) & (C.metric == met) & (C.N == N)]
            if sub.empty:
                cells.append("--")
                continue
            med = 100 * sub.rel_improvement.median()
            cells.append(f"{signed(med)} [{int(sub.eost_better.sum())}/{int(sub.base_better.sum())}]")
        macro(f"PhCov{tags[base]}", float(cov8[base]), "{:.2f}")
        rows.append(f"{names[base]} & " + " & ".join(cells) + f" & {cov8[base]:.2f}" + r" \\")
    macro("PhReproRows", sum(v["rows_compared"] for v in dec["reproduction"].values()), "{:,d}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Post hoc comparison with strong baselines (not in the frozen plan; same targets, replicates, holdouts and calibration panels as the confirmatory study; decision rules recorded before the runs). Median relative improvement of EOST-M (EIG:mg) over each baseline across the eight benchmarks, in \%, with [benchmarks with a Holm-significant difference favouring EOST-M / favouring the baseline], Holm over this post hoc family, so that the counts for MFI:pi differ from Table~\ref{tab:confirm}; last column: mean coverage of the baseline's 90\% score interval at $N=8$.}
\label{tab:posthoc}
\setlength{\tabcolsep}{2.3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{holdout log loss} & score IAE & $|\mathrm{cov}-0.9|$ & cov. \\
\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}
Policy & $N=4$ & $N=8$ & $N=16$ & $N=8$ & $N=8$ & $N=8$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/posthoc.tex", "w").write(tex)


def fig_shrinkage():
    """Exploratory PH2: regularization path of the discrimination prior (coverage vs log loss)."""
    f = "artifacts/posthoc_shrinkage/_analysis/path.csv"
    if not os.path.exists(f):
        return
    S = pd.read_csv(f)
    dec = json.load(open("artifacts/posthoc_shrinkage/_analysis/decisions.json"))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for crit, col, lab in [("eig:mg", "#2a78d6", "marginal, EIG"), ("mfi:pi", "#eb6834", "plug-in, MFI")]:
        g = S[S.criterion == crit].sort_values("tau")
        ax.plot(g["ll_rel_vs_eost_%"], g.coverage, color=col, marker="o", ms=4, lw=1.4, label=lab)
        for k, (_, r) in enumerate(g.iterrows()):
            off = (3, 3) if crit == "eig:mg" else ((-20, -2) if k % 2 else (4, -2))
            ax.annotate(f"{r.tau:.2f}", (r["ll_rel_vs_eost_%"], r.coverage), xytext=off, textcoords="offset points",
                        fontsize=6, color=col)
    ax.scatter([0], [dec["eost_coverage"]], marker="*", s=90, color=INK, zorder=4, label="EOST-M (EB sd)")
    ax.axhline(0.9, color=INK2, lw=0.8, ls=(0, (1, 1)))
    ax.set(xlabel="median log-loss difference to EOST-M (%)", ylabel="coverage of 90% interval", ylim=(0.6, 0.93))
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", bbox_to_anchor=(1.0, 0.94))
    fig.tight_layout()
    fig.savefig(f"{FIG}/shrinkage_path.pdf")
    plt.close(fig)
    best = S[S.criterion == "eig:mg"].sort_values("coverage").iloc[-1]
    macro("ShrEostCov", dec["eost_coverage"], "{:.2f}")
    for _, r in S.iterrows():
        tg = {"eig:mg": "Mg", "mfi:pi": "Pi"}[r.criterion] + {0.05: "A", 0.15: "B", 0.30: "C", 0.50: "D"}[round(r.tau, 2)]
        macro(f"Shr{tg}LL", r["ll_rel_vs_eost_%"], "{:+.1f}")
        macro(f"Shr{tg}IAE", r["iae_rel_vs_eost_%"], "{:+.1f}")
        macro(f"Shr{tg}Cov", r.coverage, "{:.2f}")
    macro("ShrPromisingN", len(dec["promising"]), "{:d}")
    mg = S[S.criterion == "eig:mg"]
    macro("ShrMgCovMin", float(mg.coverage.min()), "{:.2f}")
    macro("ShrMgCovMax", float(mg.coverage.max()), "{:.2f}")
    macro("ShrMgLLMin", float(mg["ll_rel_vs_eost_%"].min()), "{:.1f}")
    macro("ShrMgLLMax", float(mg["ll_rel_vs_eost_%"].max()), "{:.1f}")
    macro("ShrMgIAEMax", float(mg["iae_rel_vs_eost_%"].max()), "{:.0f}")


def tab_coverage_decomp():
    """Exploratory PH3: coverage of the 90% score interval decomposed into width and score error."""
    f = "artifacts/posthoc_coverage/decisions.json"
    if not os.path.exists(f):
        return
    dec = json.load(open(f))
    B = pd.read_csv("artifacts/posthoc_coverage/banks.csv").set_index("dataset")
    cells = pd.read_csv("artifacts/posthoc_coverage/cells.csv")
    mob = dec["mean_over_banks"]
    macro("PhThreeWiderBanks", dec["n_banks_rasch_wider"], "{:d}")
    macro("PhThreeWiderSig", dec["n_banks_rasch_wider_sig_holm"], "{:d}")
    macro("PhThreeWidthPct", 100 * (dec["median_width_ratio_rasch_over_eost"] - 1), "{:.0f}")
    macro("PhThreeCovHigher", dec["n_banks_rasch_cov_higher"], "{:d}")
    macro("PhThreeIaeNotLower", dec["n_higher_cov_banks_rasch_iae_not_lower"], "{:d}")
    macro("PhThreeRho", dec["spearman_ratio_coverage"]["rho"], "{:.2f}")
    macro("PhThreeCells", dec["spearman_ratio_coverage"]["n_cells"], "{:d}")
    macro("PhThreeGauss", dec["gaussian_ratio"], "{:.2f}")
    macro("PhThreeMaxRatio", float(cells.ratio.max()), "{:.2f}")
    for pol, tg in [("eig:mg", "Eost"), ("mfi:1pr", "Rasch"), ("eig:1prmg", "OnePrMg"), ("mfi:pi", "Pi")]:
        macro(f"PhThreeRatio{tg}", mob[pol]["ratio"], "{:.2f}")
        macro(f"PhThreeWidth{tg}", 100 * mob[pol]["mean_width"], "{:.1f}")
        macro(f"PhThreeIae{tg}", 100 * mob[pol]["iae"], "{:.1f}")
    rows = []
    for ds in CONF:
        r = B.loc[ds]
        rows.append(f"{NICE[ds]} & {r.cov_eost:.2f} & {r.cov_rasch:.2f} & {r.median_width_ratio_rasch_over_eost:.2f} & "
                    f"{100 * r.iae_eost:.1f} & {100 * r.iae_rasch:.1f} & {r.ratio_pi:.2f} & {r.ratio_eost:.2f} & "
                    f"{r.ratio_rasch:.2f} & {r.ratio_1prmg:.2f}" + r" \\")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Where the coverage of the 90\% score interval comes from (exploratory analysis of the post hoc runs, planned before computing; $N=8$). Coverage and score IAE (in percentage points) of EOST-M (EIG:mg) and of the regularized Rasch-type plug-in test (MFI:1pr), median over targets of the ratio of their mean interval widths (Rasch-type over EOST-M; Holm-significant on """ + \
        f"{dec['n_banks_rasch_wider_sig_holm']}" + r""" of 8 benchmarks), and ratio of mean interval width to score IAE for plug-in 2PL (MFI:pi), EOST-M, the Rasch-type test and the marginal regularized one-parameter model (EIG:1prmg, column 1PL:mg). A calibrated 90\% interval for unbiased Gaussian errors has ratio $2\cdot1.645/\sqrt{2/\pi}=""" + \
        f"{dec['gaussian_ratio']:.2f}" + r"""$.}
\label{tab:covdecomp}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & \multicolumn{2}{c}{coverage} & width & \multicolumn{2}{c}{score IAE} & \multicolumn{4}{c}{width / IAE} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(l){7-10}
Benchmark & EOST-M & Rasch & ratio & EOST-M & Rasch & MFI:pi & EOST-M & Rasch & 1PL:mg \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/coverage_decomp.tex", "w").write(tex)


def fig_mechanism(phase="confirm"):
    """Exploratory: H2 improvement vs fraction of items with identical panel verdicts."""
    C = load(phase, "confirmatory_tests.csv")
    D = diag(phase)
    if C is None or D is None:
        return
    C = C[(C.hypothesis == "H2") & C.dataset.isin(CONF)]
    dm = D.groupby(["dataset", "N"]).frac_items_extreme.mean().reset_index()
    M = C.merge(dm, on=["dataset", "N"])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    shades = {4: "#0d366b", 8: "#1c5cab", 16: "#3987e5", 32: "#86b6ef", 64: "#b7d3f6"}
    for N, g in M.groupby("N"):
        ax.scatter(100 * g.frac_items_extreme, 100 * g.rel_improvement, s=22, color=shades.get(int(N), "#2a78d6"),
                   edgecolor="white", linewidth=0.6, label=f"$N$={int(N)}", zorder=3)
    x, y = 100 * M.frac_items_extreme.values, 100 * M.rel_improvement.values
    if len(x) > 2:
        b1, b0 = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, b0 + b1 * xx, color=INK2, lw=1.0, ls="--", zorder=2)
        from scipy.stats import spearmanr
        rho = spearmanr(x, y).statistic
        macro("MechRho", rho, "{:.2f}")
        macro("MechSlope", b1, "{:.2f}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xlabel="items with identical panel verdicts (%)", ylabel="log-loss gain of EIG:mg\nover MFI:pi (%)")
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/mechanism.pdf")
    plt.close(fig)


def tab_ranking():
    """Exploratory: Kendall tau between the score estimate after t items and true accuracy."""
    f = "artifacts/explore_ranking/ranking.csv"
    if not os.path.exists(f):
        return
    R = pd.read_csv(f)
    pols = ["naive", "mfi:pi", "eig:mg", "evr:mg"]
    ts = [10, 25, 50]
    rows = []
    for d in CONF:
        g = R[R.dataset == d]
        if g.empty:
            continue
        cells = []
        for t in ts:
            for p in pols:
                v = g[(g.policy == p) & (g.t == t)].kendall
                cells.append(f"{v.values[0]:.2f}" if len(v) else "--")
        rows.append(f"{NICE[d]} & " + " & ".join(cells) + r" \\")
    hdr = " & ".join([r"\multicolumn{4}{c}{$t=" + str(t) + r"$}" for t in ts])
    sub = " & ".join(["Naive & MFI:pi & EIG:mg & EVR:mg"] * len(ts))
    cm = "".join([r"\cmidrule(lr){" + f"{2 + 4 * k}-{5 + 4 * k}" + "}" for k in range(len(ts))])
    tex = r"""\begin{table}[tbp]
\centering\scriptsize
\caption{Exploratory (not in the frozen plan): Kendall's $\tau$ between the full-bank score estimated after $t$ items and the true full-bank accuracy, across targets, at $N=8$ (one adaptive test per target; mean over 5 replicates).}
\label{tab:ranking}
\setlength{\tabcolsep}{2.0pt}
\begin{tabular}{@{}l""" + "c" * (4 * len(ts)) + r"""@{}}
\toprule
 & """ + hdr + r""" \\
""" + cm + r"""
Benchmark & """ + sub + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/ranking.tex", "w").write(tex)
    piv = R[R.t == 25].pivot_table(index="dataset", columns="policy", values="kendall")
    if {"eig:mg", "naive"} <= set(piv.columns):
        macro("TauEigTwentyfive", float(piv["eig:mg"].median()), "{:.2f}")
        macro("TauNaiveTwentyfive", float(piv["naive"].median()), "{:.2f}")
        macro("TauMfiTwentyfive", float(piv["mfi:pi"].median()), "{:.2f}")


def tab_factorial(phase="confirm"):
    """2x2 decomposition: criterion {entropy, EIG} x measurement model {plug-in, marginal}."""
    tm = load(phase, "target_means.csv.gz")
    if tm is None:
        return
    rows = []
    for N in sorted(tm.N.unique()):
        sub = tm[(tm.N == N) & tm.dataset.isin(CONF)]
        per = sub.groupby(["dataset", "policy"])[["ilogloss", "iae", "covgap"]].mean().unstack()
        if per.empty:
            continue
        cells = []
        for met in ["ilogloss", "iae"]:
            base = per[met]["mfi:pi"]
            for p in ["entropy:pi", "eig:pi", "entropy:mg", "eig:mg"]:
                v = 100 * (1 - per[met][p] / base)
                cells.append(signed(v.median()))
        rows.append(f"{int(N)} & {per.shape[0]} & " + " & ".join(cells) + r" \\")
        if N == 8:
            ll = per["ilogloss"]
            for p, nm in [("entropy:pi", "EntPi"), ("eig:pi", "EigPi"), ("entropy:mg", "EntMg"), ("eig:mg", "EigMg")]:
                macro(f"Fact{nm}LL", float((100 * (1 - ll[p] / ll["mfi:pi"])).median()), "{:+.1f}")
            ia = per["iae"]
            for p, nm in [("entropy:mg", "EntMg"), ("eig:mg", "EigMg")]:
                macro(f"Fact{nm}IAE", float((100 * (1 - ia[p] / ia["mfi:pi"])).median()), "{:+.1f}")
            d_pi = 100 * (1 - ll["eig:pi"] / ll["entropy:pi"])
            d_mg = 100 * (1 - ll["eig:mg"] / ll["entropy:mg"])
            macro("EigVsEntPiPos", int((d_pi > 0).sum()), "{:d}")
            macro("EigVsEntMgPos", int((d_mg > 0).sum()), "{:d}")
            macro("NConfBanks", int(len(d_mg)), "{:d}")
    tex = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{Decomposing the gain over plug-in Fisher CAT (MFI:pi): selection criterion (entropy vs.\ EIG) $\times$ measurement model (plug-in vs.\ calibration-marginal). Median across confirmatory benchmarks of the relative improvement over MFI:pi, in \%; $B$ = number of benchmarks.}
\label{tab:factorial}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}rrrrrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{holdout log loss} & \multicolumn{4}{c}{score IAE} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
$N$ & $B$ & Ent:pi & EIG:pi & Ent:mg & EIG:mg & Ent:pi & EIG:pi & Ent:mg & EIG:mg \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(f"{TAB}/factorial.tex", "w").write(tex)


def write_results_md(phase="confirm"):
    """Markdown summary for GitHub readers (generated from artifacts)."""
    C = load(phase, "confirmatory_tests.csv")
    dec = load(phase, "decisions.json")
    tm = load(phase, "target_means.csv.gz")
    if C is None or tm is None:
        return
    L = ["# EOST v2 -- results at a glance", "",
         "Generated by `scripts/make_report.py` from `artifacts/confirm/_analysis`. "
         "Unit of analysis: target system (mean over 5 replicates). Two-sided Wilcoxon, Holm over the whole confirmatory family.", "",
         "## Decisions under the frozen plan (primary panel size N = 8)", "",
         "| Hypothesis | Holm-significant for EOST-M | for baseline | verdict |", "|---|---|---|---|"]
    for hid, d in dec["decisions"].items():
        L.append(f"| {hid}: {d['name'].replace('|', chr(92) + '|')} | {d['holm_sig_favouring_new']}/{d['n_benchmarks']} | "
                 f"{d['holm_sig_favouring_base']} | **{'supported' if d['supported'] else 'not supported'}** |")
    L += ["", "## Confirmatory contrasts at N = 8", "",
          "| Hyp. | Benchmark | n | baseline | EOST-M | improvement | W/L | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in C[(C.N == 8) & C.dataset.isin(CONF)].sort_values(["hypothesis", "dataset"]).iterrows():
        L.append(f"| {r.hypothesis} | {NICE[r.dataset]} | {int(r.n)} | {r.mean_base:.4f} | {r.mean_new:.4f} | "
                 f"{100 * r.rel_improvement:+.1f}% | {int(r.wins)}/{int(r.losses)} | {r.p_holm:.2g} |")
    L += ["", "## Policy means at N = 8 (average over the 8 confirmatory benchmarks)", "",
          "| Policy | holdout log loss | score IAE | coverage of 90% interval |", "|---|---|---|---|"]
    sub = tm[(tm.N == 8) & tm.dataset.isin(CONF)].groupby(["dataset", "policy"])[["ilogloss", "iae", "coverage"]].mean()
    agg = sub.groupby("policy").mean()
    for p in ["naive", "random:mg", "entropy:pi", "entropy:mg", "mfi:pi", "mfi:mg", "pwfi:mg", "eig:pi", "eig:mg", "evr:pi", "evr:mg"]:
        if p in agg.index:
            r = agg.loc[p]
            ll = "--" if p == "naive" else f"{r.ilogloss:.4f}"
            L.append(f"| {LABEL[p]} | {ll} | {r.iae:.4f} | {r.coverage:.3f} |")
    # sensitivity analyses (N = 8): median relative improvement of EIG:mg and mean coverage
    L += ["", "## Sensitivity analyses at N = 8 (median relative improvement of EIG:mg across benchmarks)", "",
          "| Variant | H1 log loss vs Entropy:mg | H2 log loss vs MFI:pi | H3 IAE vs Naive | H4 cov. gap vs MFI:pi | EOST-M coverage |",
          "|---|---|---|---|---|---|"]
    names = [("confirm", "main analysis (frozen plan)"), ("sens_fixedprior", "fixed item prior (frozen plan)"),
             ("sens_dedup", "near-duplicates removed, 5 leaderboard banks (frozen plan)"),
             ("sens_1pl", "one-parameter (frozen plan; degenerate calibration, not interpreted)"),
             ("sens_1pl_reg", "one-parameter, log a ~ N(0, 0.05^2) (post hoc)")]
    for ph, nm in names:
        t = load(ph, "target_means.csv.gz")
        if t is None:
            continue
        t = t[t.dataset.isin(CONF) & (t.N == 8)]
        per = t.groupby(["dataset", "policy"])[["ilogloss", "iae", "covgap", "coverage"]].mean().unstack()
        f = lambda met, base: f"{(100 * (1 - per[met]['eig:mg'] / per[met][base])).median():+.1f}%"
        L.append(f"| {nm} | {f('ilogloss', 'entropy:mg')} | {f('ilogloss', 'mfi:pi')} | {f('iae', 'naive')} | "
                 f"{f('covgap', 'mfi:pi')} | {per['coverage']['eig:mg'].mean():.2f} |")
    # post hoc experiments (not in the frozen plan; plans recorded in docs/RESEARCH_LOG.md before running)
    ph1 = "artifacts/posthoc_baselines/_analysis/contrasts.csv"
    if os.path.exists(ph1):
        C1 = pd.read_csv(ph1)
        T1 = pd.read_csv("artifacts/posthoc_baselines/_analysis/target_means.csv.gz")
        cov1 = T1[T1.N == 8].groupby(["dataset", "policy"]).coverage.mean().unstack().mean(axis=0)
        L += ["", "## Post hoc strong baselines (PH1, N = 8; not in the frozen plan)", "",
              "Median relative improvement of EOST-M (EIG:mg) over each baseline across the 8 benchmarks "
              "[benchmarks Holm-significant for EOST-M / for the baseline].", "",
              "| Baseline | holdout log loss | score IAE | baseline coverage |", "|---|---|---|---|"]
        for base, nm in [("mfi:pi", "plug-in 2PL, MFI"), ("mfi:eap", "plug-in posterior-mean 2PL, MFI"),
                         ("eig:eap", "plug-in posterior-mean 2PL, EIG"), ("mfi:1pr", "plug-in regularized 1PL, MFI"),
                         ("eig:1prmg", "marginal regularized 1PL, EIG"), ("efi:mg", "expected Fisher information")]:
            cell = []
            for met in ["ilogloss", "iae"]:
                s = C1[(C1.base == base) & (C1.metric == met) & (C1.N == 8)]
                cell.append(f"{100 * s.rel_improvement.median():+.1f}% [{int(s.eost_better.sum())}/{int(s.base_better.sum())}]")
            L.append(f"| {nm} | {cell[0]} | {cell[1]} | {cov1[base]:.2f} |")
        L.append(f"| *EOST-M (reference)* | -- | -- | {cov1['eig:mg']:.2f} |")
    ph2 = "artifacts/posthoc_shrinkage/_analysis/decisions.json"
    if os.path.exists(ph2):
        d2 = json.load(open(ph2))
        L += ["", "## Exploratory regularization path (PH2, N = 8)", "",
              "| criterion | sd(log a) | log loss vs EOST-M | score IAE vs EOST-M | coverage |", "|---|---|---|---|---|"]
        for r in d2["path"]:
            L.append(f"| {r['criterion']} | {r['tau']:.2f} | {r['ll_rel_vs_eost_%']:+.1f}% | "
                     f"{r['iae_rel_vs_eost_%']:+.1f}% | {r['coverage']:.2f} |")
        L.append(f"\nPre-stated criterion: {d2['promising_criterion']}. Promising settings: "
                 f"{', '.join(d2['promising']) if d2['promising'] else 'none'}.")
    ph3 = "artifacts/posthoc_coverage/decisions.json"
    if os.path.exists(ph3):
        d3 = json.load(open(ph3))
        L += ["", "## Exploratory coverage decomposition (PH3, N = 8)", "",
              "| model | coverage | mean interval width | score IAE | width / IAE |", "|---|---|---|---|---|"]
        for p, nm in [("mfi:pi", "plug-in 2PL (MFI:pi)"), ("eig:mg", "EOST-M (EIG:mg)"),
                      ("mfi:1pr", "Rasch-type plug-in (MFI:1pr)"), ("eig:1prmg", "marginal regularized 1PL (EIG:1prmg)")]:
            m = d3["mean_over_banks"][p]
            L.append(f"| {nm} | {m['coverage']:.3f} | {m['mean_width']:.3f} | {m['iae']:.4f} | {m['ratio']:.2f} |")
        L.append(f"\nRasch-type intervals wider than EOST-M's on {d3['n_banks_rasch_wider']}/8 benchmarks; "
                 f"Spearman(width/IAE, coverage) = {d3['spearman_ratio_coverage']['rho']:.2f} over "
                 f"{d3['spearman_ratio_coverage']['n_cells']} bank-model cells; a calibrated Gaussian 90% interval "
                 f"has width/IAE = {d3['gaussian_ratio']:.2f}. Verdict (rule fixed before computing): {d3['verdict']}.")
    L += ["", "Every qualitative statement of the paper is checked against these artifacts by "
          "`scripts/check_claims.py` (`artifacts/CLAIMS.json`)."]
    open("RESULTS.md", "w").write("\n".join(L) + "\n")


def runtime_macros(phase="confirm"):
    secs, rows = 0.0, 0
    for f in glob.glob(f"artifacts/{phase}/*/config.json"):
        c = json.load(open(f))
        secs += c.get("seconds", 0.0)
        rows += c.get("n_rows", 0)
    if rows:
        macro("ConfirmHours", secs / 3600, "{:.1f}")
        macro("ConfirmRuns", rows, "{:,d}")


def write_numbers():
    dev = "configs/deviations.txt"
    macro("DeviationText", open(dev).read().strip() if os.path.exists(dev) else "None.")
    with open(f"{GEN}/numbers.tex", "w") as fh:
        for k, v in sorted(NUM.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    fig_theory()
    tab_datasets()
    tab_confirmatory()
    tab_policies()
    tab_sensitivity()
    tab_development()
    tab_full_results()
    tab_efficiency()
    curves_macros()
    fig_crossover()
    tab_posthoc()
    fig_shrinkage()
    tab_coverage_decomp()
    tab_ranking()
    tab_factorial()
    fig_curves()
    fig_curves(metric="err", fname="curves_err_N8.pdf", ylabel="$|\\hat A_t-A|$")
    fig_heatmap()
    fig_coverage()
    fig_diagnostics()
    fig_synthetic()
    fig_mechanism()
    write_results_md()
    runtime_macros()
    write_numbers()
    print("REPORT_GATE=PASS figures:", sorted(os.listdir(FIG)), "tables:", sorted(os.listdir(TAB)))
