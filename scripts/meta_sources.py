"""Sensitivity RF3b of the secondary meta-analysis RF3 (plan: docs/RESEARCH_LOG.md, iteration 5,
written before running).

RF3 pools the eight benchmarks as if they were independent studies, but GSM8K, ARC-C, HellaSwag,
WinoGrande and MMLU come from the same 395 Open LLM Leaderboard (OLLB) systems. RF3b pools over
four sources instead -- OLLB, SWE-bench Verified, LiveCodeBench, CRUXEval-O -- with the OLLB effect
the equal-weight mean of RF3's five per-benchmark effects and its standard error from a bootstrap
over the 395 shared systems (a resampled system brings its target-level results in every OLLB
benchmark where it is a target). Other sources keep RF3's effects and standard errors.
DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman 95% interval with 3 degrees of freedom.
No frozen decision changes. Output: artifacts/robust/meta/{sources.csv,sources_summary.json}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from meta_analysis import B_BOOT, OUT, SPEC, random_effects, rel  # noqa: E402

OLLB = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu"]
N_SYSTEMS = 395


def ollb_bootstrap(arrays, rng):
    """arrays: per bank (targets, base, new). Returns (B_BOOT x 5) replicates of the per-bank effects."""
    out = np.full((B_BOOT, len(arrays)), np.nan)
    for r in range(B_BOOT):
        counts = np.bincount(rng.integers(0, N_SYSTEMS, N_SYSTEMS), minlength=N_SYSTEMS)
        for j, (t, base, new) in enumerate(arrays):
            w = counts[t].astype(float)
            if w.sum() == 0:
                continue
            b, n = (w * base).sum() / w.sum(), (w * new).sum() / w.sum()
            out[r, j] = (b - n) / b
    return out


def main():
    tm = pd.read_csv("artifacts/study2/_analysis/target_means.csv.gz")
    tm = tm[tm.N == SPEC["primary_panel_size"]]
    pb = pd.read_csv(f"{OUT}/per_bank.csv")
    rows, cors = [], []
    for i, h in enumerate(SPEC["superiority"]):
        arrays, ys = [], []
        for ds in OLLB:
            g = tm[tm.dataset == ds].pivot_table(index="target", columns="policy", values=h["metric"])
            g = g[[h["new"], h["base"]]].dropna()
            t = g.index.values.astype(int)
            assert t.max() < N_SYSTEMS
            base, new = g[h["base"]].values, g[h["new"]].values
            arrays.append((t, base, new))
            ys.append(rel(base, new))
            ref = pb[(pb.hypothesis == h["id"]) & (pb.dataset == ds)].rel.values
            assert len(ref) == 1 and abs(ref[0] - ys[-1]) < 1e-12, (h["id"], ds)    # same statistic as RF3
        R = ollb_bootstrap(arrays, np.random.default_rng([SPEC["seed"], 3, 2, i]))
        R = R[~np.isnan(R).any(1)]
        y_ollb, se_ollb = float(np.mean(ys)), float(R.mean(1).std(ddof=1))
        C = np.corrcoef(R.T)
        cors.append(float(np.median(C[np.triu_indices(len(OLLB), 1)])))
        y, se, src = [y_ollb], [se_ollb], ["OLLB"]
        for ds in ("swe_verified", "livecodebench", "cruxeval_o"):
            x = pb[(pb.hypothesis == h["id"]) & (pb.dataset == ds)]
            if len(x) == 1:
                y.append(float(x.rel.iloc[0])); se.append(float(x.se.iloc[0])); src.append(ds)
        re = random_effects(y, se)
        rows.append({"hypothesis": h["id"], "name": h["name"], **re, "k_sources": len(y),
                     "n_positive_sources": int(sum(v > 0 for v in y)), "excludes_zero": bool(re["ci_lo"] > 0),
                     "ollb_effect": y_ollb, "ollb_se": se_ollb, "ollb_median_pairwise_r": cors[-1],
                     "ollb_se_independent": float(np.sqrt(np.sum(pb[(pb.hypothesis == h["id"]) &
                                                                   (pb.dataset.isin(OLLB))].se.values ** 2)) / len(OLLB)),
                     "boot_replicates": int(len(R))})
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUT}/sources.csv", index=False)
    M = pd.read_csv(f"{OUT}/meta.csv").set_index("hypothesis")
    summ = {"k_sources": 4, "n_hypotheses": int(len(S)),
            "n_excl_zero_sources": int(S.excludes_zero.sum()),
            "excl_zero_sources": sorted(S[S.excludes_zero].hypothesis.tolist()),
            "n_excl_zero_banks": int((M.ci_lo > 0).sum()),
            "lost": sorted(h for h in M.index if M.loc[h, "ci_lo"] > 0 and not bool(S.set_index("hypothesis").loc[h, "excludes_zero"])),
            "median_ollb_pairwise_r": float(np.median(cors)),
            "ollb_se_ratio_median": float((S.ollb_se / S.ollb_se_independent).median())}
    json.dump(summ, open(f"{OUT}/sources_summary.json", "w"), indent=1)
    with pd.option_context("display.width", 200):
        print(S[["hypothesis", "pooled", "ci_lo", "ci_hi", "p_hksj", "I2", "n_positive_sources", "excludes_zero",
                 "ollb_median_pairwise_r"]].round(4).to_string(index=False))
    print(json.dumps(summ, indent=1))
    print("RF3B_GATE=PASS")


if __name__ == "__main__":
    main()
