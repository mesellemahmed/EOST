"""Robustness follow-up RF3 (plan: docs/RESEARCH_LOG.md, iteration 4, written before running).

Secondary question that the frozen per-benchmark decision rule does not answer: does an effect
generalize across benchmarks on average? For every study-2 superiority hypothesis (N = 8):
random-effects meta-analysis over the eight benchmarks of the per-benchmark relative
improvement of EOST-M (DerSimonian-Laird tau^2; Hartung-Knapp-Sidik-Jonkman 95% interval with
k - 1 degrees of freedom; 95% prediction interval with k - 2 degrees of freedom; I^2), with
per-benchmark standard errors from 2,000 paired bootstrap resamples of targets, and an exact
two-sided sign test over benchmarks. No frozen decision changes.
Output: artifacts/robust/meta/{meta.csv,summary.json}.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t as tdist

SPEC = json.load(open("configs/study2_hypotheses.json"))
OUT = "artifacts/robust/meta"
B_BOOT = 2000


def rel(base, new):
    return (base.mean() - new.mean()) / base.mean()


def boot_se(base, new, rng):
    n = len(base)
    idx = rng.integers(0, n, (B_BOOT, n))
    b, w = base[idx].mean(1), new[idx].mean(1)
    return float(((b - w) / b).std(ddof=1))


def random_effects(y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    mu_fe = (w * y).sum() / w.sum()
    Q = float((w * (y - mu_fe) ** 2).sum())
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c)
    ws = 1 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    q = float((ws * (y - mu) ** 2).sum() / (k - 1))
    se_hk = float(np.sqrt(max(q, 1e-300) / ws.sum()))
    tq = tdist.ppf(0.975, k - 1)
    se_re = float(np.sqrt(1 / ws.sum()))
    tp = tdist.ppf(0.975, k - 2)
    pi = (mu - tp * np.sqrt(tau2 + se_re ** 2), mu + tp * np.sqrt(tau2 + se_re ** 2))
    p_hk = float(2 * tdist.sf(abs(mu / se_hk), k - 1))
    return {"pooled": mu, "ci_lo": mu - tq * se_hk, "ci_hi": mu + tq * se_hk, "p_hksj": p_hk, "tau": float(np.sqrt(tau2)),
            "pi_lo": float(pi[0]), "pi_hi": float(pi[1]), "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0,
            "Q": Q, "k": k}


def main():
    os.makedirs(OUT, exist_ok=True)
    tm = pd.read_csv("artifacts/study2/_analysis/target_means.csv.gz")
    tm = tm[tm.N == SPEC["primary_panel_size"]]
    frozen = json.load(open("artifacts/study2/_analysis/decisions.json"))["decisions"]
    rows, per_bank = [], []
    for h in SPEC["superiority"]:
        ys, ses = [], []
        for ds in SPEC["banks"]:
            g = tm[tm.dataset == ds].pivot_table(index="target", columns="policy", values=h["metric"])
            if h["new"] not in g or h["base"] not in g:
                continue
            g = g[[h["new"], h["base"]]].dropna()
            rng = np.random.default_rng([SPEC["seed"], len(per_bank)])
            base, new = g[h["base"]].values, g[h["new"]].values
            y, se = rel(base, new), boot_se(base, new, rng)
            ys.append(y); ses.append(se)
            per_bank.append({"hypothesis": h["id"], "dataset": ds, "rel": y, "se": se, "n": len(g)})
        re = random_effects(ys, ses)
        npos = int(sum(v > 0 for v in ys))
        rows.append({"hypothesis": h["id"], "name": h["name"], **re, "n_positive": npos, "n_banks": len(ys),
                     "p_sign": float(binomtest(npos, len(ys), 0.5).pvalue),
                     "frozen_supported": bool(frozen[h["id"]]["supported"]),
                     "frozen_favour_new": int(frozen[h["id"]]["favour_new"])})
    M = pd.DataFrame(rows)
    M.to_csv(f"{OUT}/meta.csv", index=False)
    pd.DataFrame(per_bank).to_csv(f"{OUT}/per_bank.csv", index=False)
    json.dump({"rows": rows}, open(f"{OUT}/summary.json", "w"), indent=1)
    with pd.option_context("display.width", 200):
        print(M[["hypothesis", "pooled", "ci_lo", "ci_hi", "p_hksj", "tau", "I2", "n_positive", "p_sign",
                 "frozen_supported"]].round(4).to_string(index=False))
    print("RF3_GATE=PASS")


if __name__ == "__main__":
    main()
