"""Robustness follow-up RF1 (plan: docs/RESEARCH_LOG.md, iteration 4, written before running).

Does the response-derived clustering of study 2 favour EOST-M? The frozen strict clusters join
two systems when they share a heuristic family or when the correlation of their standardized
2PL residuals is >= 0.5, computed from every system's full response vector, the target's
included. Here the frozen targets, holdouts, replicates and panel seeds are kept, and only the
set of systems excluded from the target's panel changes:

  metadata      heuristic families only (no response data)
  lbo           leave-bank-out residual clusters: heuristic family OR residual correlation
                >= 0.5 computed on the *other* banks containing the same systems (other Open LLM
                Leaderboard banks; CRUXEval-I for CRUXEval-O; SWE-bench Lite for the SWE-bench
                Verified systems it contains); metadata only for LiveCodeBench
  joint_0.4/0.6/0.7   frozen construction at other thresholds
  joint_0.5     frozen construction recomputed (reproduction check: must equal the frozen rows)

Policies (N = 8): EOST-M (eig:mg), plug-in Fisher CAT (mfi:pi), Rasch-type plug-in (mfi:1pr),
Random:mg and Naive. Frozen modules are imported unchanged; seeds are the frozen ones.
Outputs: artifacts/robust/clusters/<variant>/<bank>/{parts,rows.csv.gz,done.json} and
artifacts/robust/clusters/_clusters.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import _seed  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model, plugin_model  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402
from eost.replay2 import summarize2  # noqa: E402
from eost.study2 import OLLB, PRIOR_1PR, S2Config, components, eligible_s2, residual_matrix  # noqa: E402

OUT = "artifacts/robust/clusters"
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
VARIANTS = ["joint_0.5", "metadata", "lbo", "joint_0.4", "joint_0.6", "joint_0.7"]
N0 = 8
POLICIES = [("eig:mg", "eig", "mg"), ("mfi:pi", "mfi", "pi"), ("mfi:1pr", "mfi", "1pr"),
            ("random:mg", "random", "mg"), ("naive", "naive", None)]


# ----------------------------------------------------------------------------- clusters
def _family(B):
    H = np.zeros((B.J, B.J), dtype=bool)
    for j in range(B.J):
        H[j] |= B.related(j)
    return H | H.T


def _corr(Rs):
    R = np.concatenate(Rs, 1)
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    return R @ R.T


def compute_clusters():
    f = f"{OUT}/_clusters.json"
    if os.path.exists(f):
        return json.load(open(f))
    os.makedirs(OUT, exist_ok=True)
    res = {v: {} for v in VARIANTS}
    groups = {"ollb": OLLB, "crux": ["cruxeval_o", "cruxeval_i"]}
    banks = {n: load_bank(f"data/processed/{n}.npz") for n in OLLB + ["cruxeval_o", "cruxeval_i", "swe_verified",
                                                                          "swe_lite", "livecodebench"]}
    resid = {}
    for n, B in banks.items():
        t0 = time.time()
        resid[n] = residual_matrix(B.Y)
        print(f"[residuals] {n} {time.time() - t0:.0f}s", flush=True)
    # joint (frozen construction) at several thresholds
    for gname, g in list(groups.items()) + [("swe", ["swe_verified"]), ("lcb", ["livecodebench"])]:
        H = _family(banks[g[0]])
        C = _corr([resid[n] for n in g])
        for v in VARIANTS:
            if v.startswith("joint_"):
                lab = components(H | (C >= float(v.split("_")[1])))
                for n in g:
                    res[v][n] = lab.tolist()
    # metadata only
    for n in CONF:
        res["metadata"][n] = components(_family(banks[n])).tolist()
    # leave-bank-out residual clusters
    for n in CONF:
        B = banks[n]
        H = _family(B)
        if n in OLLB:
            C = _corr([resid[m] for m in OLLB if m != n])
            lab = components(H | (C >= 0.5))
        elif n == "cruxeval_o":
            lab = components(H | (_corr([resid["cruxeval_i"]]) >= 0.5))
        elif n == "swe_verified":
            L = banks["swe_lite"]
            pos = {str(s): i for i, s in enumerate(L.systems)}
            idx = [(j, pos[str(s)]) for j, s in enumerate(B.systems) if str(s) in pos]
            CL = _corr([resid["swe_lite"]])
            A = np.zeros((B.J, B.J), dtype=bool)
            for j1, l1 in idx:
                for j2, l2 in idx:
                    A[j1, j2] = CL[l1, l2] >= 0.5
            lab = components(H | A)
        else:                                   # livecodebench: no other bank shares its systems
            lab = components(H)
        res["lbo"][n] = lab.tolist()
    frozen = json.load(open("configs/study2_clusters.json"))["banks"]
    for n in CONF:                              # the recomputed frozen partition must equal the frozen one
        a, b = np.array(res["joint_0.5"][n]), np.array(frozen[n]["cluster"])
        same = all(len(set(b[a == c])) == 1 for c in np.unique(a)) and len(np.unique(a)) == len(np.unique(b))
        assert same, f"joint_0.5 does not reproduce the frozen clusters on {n}"
    summ = {v: {n: {"n_clusters": int(len(np.unique(res[v][n]))), "max_size": int(np.bincount(res[v][n]).max())}
                for n in CONF} for v in VARIANTS}
    json.dump({"clusters": res, "summary": summ}, open(f, "w"))
    print("RF1_CLUSTERS=PASS (joint_0.5 reproduces the frozen partition)", flush=True)
    return json.load(open(f))


# ----------------------------------------------------------------------------- replays
_S = {}


def _init(name, variant, labels):
    _S.update(bank=load_bank(f"data/processed/{name}.npz"), variant=variant, cl=np.array(labels), cfg=S2Config())


def _work(t):
    B, cl, cfg, variant = _S["bank"], _S["cl"], _S["cfg"], _S["variant"]
    y = B.Y[t].astype(np.int64)
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    elig = eligible_s2(B, cl, t, None)
    rows, t0 = [], time.time()
    if len(elig) < N0:
        return int(t), [{"dataset": B.name, "variant": variant, "target": int(t), "skipped": True,
                         "n_eligible": int(len(elig))}], 0.0
    for r in range(cfg.replicates):
        rng_h = np.random.default_rng(_seed(cfg.master_seed, B.name, "main", "holdout", t, r))
        perm = rng_h.permutation(B.I)
        nh = int(round(cfg.holdout_frac * B.I))
        holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
        rng_p = np.random.default_rng(_seed(cfg.master_seed, B.name, "main", "panel", t, r, N0))
        panel = np.sort(rng_p.choice(elig, size=N0, replace=False))
        Yp = B.Y[panel].astype(float)
        sd = lambda *k: _seed(cfg.master_seed, B.name, "main", *k, t, r, N0)
        cal, W, grid = calibrate(Yp, "2pl", Priors(), True)
        pre = {"mg": Precomputed.build(marginal_model(W, grid, theta), 1.0),
               "pi": Precomputed.build(plugin_model(cal, theta), 1.0)}
        cal1, _, _ = calibrate(Yp, "1pl", PRIOR_1PR, True)
        pre["1pr"] = Precomputed.build(plugin_model(cal1, theta), 1.0)
        for pol, crit, key in POLICIES:
            out = replay(y, pre.get(key), pool, holdout, crit, cfg.budget, np.random.default_rng(sd("policy", pol)))
            s = summarize2(out, None)
            s.update({"dataset": B.name, "variant": variant, "target": int(t), "rep": r, "N": N0, "policy": pol,
                      "skipped": False, "n_eligible": int(len(elig)), "panel": " ".join(map(str, panel))})
            rows.append(s)
    return int(t), rows, time.time() - t0


def run(variant, name, labels, workers):
    od = f"{OUT}/{variant}/{name}"
    if os.path.exists(f"{od}/done.json"):
        print(f"[{variant}/{name}] complete", flush=True)
        return
    targets = json.load(open(f"artifacts/study2/main/{name}/config.json"))["targets"]
    os.makedirs(f"{od}/parts", exist_ok=True)
    todo = [t for t in targets if not os.path.exists(f"{od}/parts/target_{t:04d}.json.gz")]
    if todo:
        with get_context("fork").Pool(workers, initializer=_init, initargs=(name, variant, labels)) as pool:
            for k, (t, rows, secs) in enumerate(pool.imap_unordered(_work, todo)):
                tmp = f"{od}/parts/target_{t:04d}.json.gz.tmp"
                with gzip.open(tmp, "wt") as fh:
                    json.dump({"target": t, "seconds": secs, "rows": rows}, fh)
                os.replace(tmp, f"{od}/parts/target_{t:04d}.json.gz")
                print(f"[{variant}/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)}", flush=True)
    rows, secs = [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))
        rows += p["rows"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/rows.csv.gz", index=False)
    json.dump({"variant": variant, "dataset": name, "n_rows": len(rows), "cpu_seconds": round(secs, 1)},
              open(f"{od}/done.json", "w"), indent=1)
    print(f"RF1_GATE=PASS {variant}/{name} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--datasets", nargs="*", default=CONF)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cl = compute_clusters()["clusters"]
    for v in a.variants:
        for n in a.datasets:
            run(v, n, cl[v][n], a.workers)
    print("RF1_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
