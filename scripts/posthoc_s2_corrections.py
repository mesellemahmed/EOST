"""Post hoc corrections of study 2 decided before any study-2 result was inspected
(docs/RESEARCH_LOG.md, "Independent code review of the frozen study-2 code").

tiny   -- recompute, for every N = 8 cell of the main component, the number of anchors that
          tinyBenchmarks actually used at budgets 25/50/100 (same seeds, same calls as the
          frozen run), so that the comparison can be restricted to full-budget cells and
          matched on the number of items.
fbfix  -- re-run the two fully Bayesian arms at N = 8 and N = 4 (--N; the plan in the research
          log names both) with a burn-in of 60 sweeps instead of
          10 (draws and thinning unchanged; same targets, panels, holdouts and seeds), and
          record the drift of the overall location within the retained sweeps; also run,
          as an exploratory addition, the joint fully Bayesian model (a mixture over draws of
          products of expected response functions, 'fbj:eig' and 'fbj:bmi').
Frozen modules are imported unchanged. Outputs: artifacts/study2_posthoc/{tiny,fbfix,fbfix_N4}/<bank>/.
--verify-frozen re-runs the first target of each bank with the frozen burn-in (10 sweeps) and
checks that the fully Bayesian rows reproduce the frozen artifacts exactly (same seeds, panels,
holdouts and chains; only the burn-in differs in the correction).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.experiment import H0_STD_NORMAL, _seed  # noqa: E402
from eost.fullbayes import expected_fisher, gibbs_item_marginals  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model  # noqa: E402
from eost.reduction import tiny_estimates  # noqa: E402
from eost.replay import Precomputed, reference_theta, replay  # noqa: E402
from eost.replay2 import summarize2  # noqa: E402
from eost.study2 import S2Config, eligible_s2  # noqa: E402

OUT = "artifacts/study2_posthoc"


def gibbs_with_states(Y, theta0, pr0, grid, rng, burn, draws, thin, sd_theta=1.0, trace=None):
    """Verbatim copy of eost.fullbayes.gibbs_item_marginals that also returns the retained
    (abilities, hyperparameters) states; tests/test_posthoc.py checks that its W_bar equals the
    frozen function's for the same seed."""
    from dataclasses import replace as _replace
    from scipy.special import log_expit
    from eost.fullbayes import B_STEP, LOCAL, NB, SHIFT_STEPS, _clip_priors, _sample_rows
    from eost.irt import _softmax_rows, grid_logprior, grid_loglik
    Y = np.asarray(Y, dtype=float)
    N, I = Y.shape
    th = np.array(theta0, dtype=float)
    pr = pr0
    la, bb = grid.log_a, grid.b
    a_grid = np.exp(la)
    W_bar = np.zeros((I, len(bb)))
    states = []
    for s_ in range(burn + draws * thin):
        W = _softmax_rows(grid_loglik(Y, th, grid) + grid_logprior(grid, pr, "2pl")[None, :])
        if s_ >= burn and (s_ - burn) % thin == thin - 1:
            W_bar += W
            states.append((th.copy(), pr))
        g = _sample_rows(W, rng)
        a_i, b_i = a_grid[g], bb[g]
        grid_th = th[:, None] + LOCAL[None, :]
        z = a_i[None, None, :] * (grid_th[:, :, None] - b_i[None, None, :])
        ll = (Y[:, None, :] * log_expit(z) + (1.0 - Y[:, None, :]) * log_expit(-z)).sum(2)
        ll += -0.5 * (grid_th / sd_theta) ** 2
        P = np.exp(ll - ll.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        k = _sample_rows(P, rng)
        th = grid_th[np.arange(N), k]
        sh = np.arange(-SHIFT_STEPS, SHIFT_STEPS + 1)
        lw = -0.5 * (((th[None, :] + sh[:, None] * B_STEP) / sd_theta) ** 2).sum(1)
        gb = g % NB
        valid = np.array([((gb + k_) >= 0).all() and ((gb + k_) < NB).all() for k_ in sh])
        lw = np.where(valid, lw, -np.inf)
        pw = np.exp(lw - lw.max()); pw /= pw.sum()
        k_sh = int(sh[rng.choice(len(sh), p=pw)])
        if k_sh:
            th = th + k_sh * B_STEP
            g = g + k_sh
            a_i, b_i = a_grid[g], bb[g]
            pr = _replace(pr, mu_b=pr.mu_b + k_sh * B_STEP)
        upd = {}
        for key_mu, key_sd, x in (("mu_b", "sd_b", b_i), ("mu_log_a", "sd_log_a", la[g])):
            xb, ss = x.mean(), ((x - x.mean()) ** 2).sum()
            var = ss / max(rng.chisquare(I - 1), 1e-12)
            upd[key_sd] = float(np.sqrt(var))
            upd[key_mu] = float(rng.normal(xb, np.sqrt(var / I)))
        pr = _clip_priors(_replace(pr, **upd))
        if trace is not None:
            trace.append([th.mean(), th.std(), pr.mu_b, pr.sd_b, pr.mu_log_a, pr.sd_log_a, th[0]])
    return W_bar / max(len(states), 1), states


def joint_mixture_model(Y, states, grid, theta):
    """Flattened (draw, ability) grid: the target's likelihood is a mixture over retained draws
    of products of each draw's expected response functions (the joint fully Bayesian model),
    instead of a product of per-item averages."""
    from eost.irt import MeasurementModel, _softmax_rows, grid_logprior, grid_loglik
    Ps, dPs = [], []
    for th, pr in states:
        W = _softmax_rows(grid_loglik(Y, th, grid) + grid_logprior(grid, pr, "2pl")[None, :])
        m = marginal_model(W, grid, theta)
        Ps.append(m.P); dPs.append(m.dP)
    P, dP = np.concatenate(Ps, 1), np.concatenate(dPs, 1)
    flat = np.tile(theta, len(states))
    return MeasurementModel("fbj", flat, P, dP), len(states)
CONF = ["ollb_gsm8k", "ollb_arc", "ollb_hellaswag", "ollb_winogrande", "ollb_mmlu", "swe_verified",
        "livecodebench", "cruxeval_o"]
BURN_FIX = 60
_S = {}


def _init(name, task, N=8, burn=BURN_FIX):
    B = load_bank(f"data/processed/{name}.npz")
    cl = np.array(json.load(open("configs/study2_clusters.json"))["banks"][name]["cluster"])
    _S.update(bank=B, clusters=cl, cfg=S2Config(), task=task, N=N, burn=burn)


def _cell_setup(B, cl, cfg, t, r, N0, comp="main"):
    I = B.I
    rng_h = np.random.default_rng(_seed(cfg.master_seed, B.name, comp, "holdout", t, r))
    perm = rng_h.permutation(I)
    nh = int(round(cfg.holdout_frac * I))
    holdout, pool = np.sort(perm[:nh]), np.sort(perm[nh:])
    elig = eligible_s2(B, cl, t, None)
    rng_p = np.random.default_rng(_seed(cfg.master_seed, B.name, comp, "panel", t, r, N0))
    panel = np.sort(rng_p.choice(elig, size=N0, replace=False))
    return holdout, pool, panel


def _work(t):
    B, cl, cfg, task = _S["bank"], _S["clusters"], _S["cfg"], _S["task"]
    N0, BURN = _S["N"], _S["burn"]
    y = B.Y[t].astype(np.int64)
    theta = default_theta_grid(cfg.theta_lo, cfg.theta_hi, cfg.theta_K)
    rows = []
    t0 = time.time()
    for r in range(cfg.replicates):
        holdout, pool, panel = _cell_setup(B, cl, cfg, t, r, N0)
        Yp = B.Y[panel].astype(float)
        sd = lambda *k: _seed(cfg.master_seed, B.name, "main", *k, t, r, N0)
        cal, W, grid = calibrate(Yp, "2pl", Priors(), True)
        base = dict(dataset=B.name, target=int(t), rep=r, N=N0)
        if task == "tiny":
            tb = tiny_estimates(y, Yp, cal.theta_cal, cal.priors, cal.a, cal.b, pool, cfg.reduction_T,
                                np.random.default_rng(sd("tiny")), theta)
            A = float(y.mean())
            for T in cfg.reduction_T:
                rows.append(base | {"T": T, "n_items": tb[T]["n_items"], "err_gpirt": abs(tb[T]["gpirt"] - A),
                                    "err_pirt": abs(tb[T]["pirt"] - A)})
        else:
            tr = []
            W_fb, states = gibbs_with_states(Yp, cal.theta_cal, cal.priors, grid, np.random.default_rng(sd("fb")),
                                             BURN, cfg.fb_draws, cfg.fb_thin, trace=tr)
            T_ = np.array(tr)
            kept = T_[BURN:, 0]
            loc = {"loc_start": float(T_[0, 0]), "loc_burn_end": float(T_[BURN - 1, 0]),
                   "loc_kept_first": float(kept[0]), "loc_kept_last": float(kept[-1]),
                   "loc_kept_slope": float(np.polyfit(np.arange(len(kept)), kept, 1)[0]),
                   "loc_frozen_window_slope": float(np.polyfit(np.arange(10, 42), T_[10:42, 0], 1)[0])}
            m_fb = marginal_model(W_fb, grid, theta)
            pre_eig = Precomputed.build(m_fb, 1.0)
            pre_bmi = Precomputed.build(m_fb, 1.0)
            pre_bmi.FI = expected_fisher(W_fb, grid, theta)
            mj, M = joint_mixture_model(Yp, states, grid, theta)
            pre_j = Precomputed.build(mj, 1.0)
            pre_j.log_prior = pre_j.log_prior - np.log(M)
            for pol, crit, pre in (("fb:bmi", "pwfi", pre_bmi), ("fb:eig", "eig", pre_eig),
                                   ("fbj:eig", "eig", pre_j), ("fbj:bmi", "pwfi", pre_j)):
                out = replay(y, pre, pool, holdout, crit, cfg.budget, np.random.default_rng(sd("policy", pol)))
                s = summarize2(out, None)
                rows.append(base | {"policy": pol, "cluster": int(cl[t]), "component": "main"} | s | loc)
    return int(t), rows, time.time() - t0


def run(task, name, workers, max_targets=None, out=OUT, N=8):
    assert task == "fbfix" or N == 8, "the anchor audit concerns N = 8 only"
    od = f"{out}/{task}/{name}" if N == 8 else f"{out}/{task}_N{N}/{name}"
    if os.path.exists(f"{od}/done.json"):
        print(f"[{task}/{name}] complete", flush=True)
        return
    targets = json.load(open(f"artifacts/study2/main/{name}/config.json"))["targets"][:max_targets]
    os.makedirs(f"{od}/parts", exist_ok=True)
    todo = [t for t in targets if not os.path.exists(f"{od}/parts/target_{t:04d}.json.gz")]
    if todo:
        with get_context("fork").Pool(workers, initializer=_init, initargs=(name, task, N)) as pool:
            for k, (t, rows, secs) in enumerate(pool.imap_unordered(_work, todo)):
                tmp = f"{od}/parts/target_{t:04d}.json.gz.tmp"
                with gzip.open(tmp, "wt") as fh:
                    json.dump({"target": t, "seconds": secs, "rows": rows}, fh)
                os.replace(tmp, f"{od}/parts/target_{t:04d}.json.gz")
                print(f"[{task}/{name}] {len(targets) - len(todo) + k + 1}/{len(targets)}", flush=True)
    rows, secs = [], 0.0
    for t in targets:
        p = json.load(gzip.open(f"{od}/parts/target_{t:04d}.json.gz", "rt"))
        rows += p["rows"]; secs += p["seconds"]
    pd.DataFrame(rows).to_csv(f"{od}/rows.csv.gz", index=False)
    json.dump({"task": task, "dataset": name, "N": N, "n_rows": len(rows), "cpu_seconds": round(secs, 1),
               "burn_fix": BURN_FIX if task == "fbfix" else None}, open(f"{od}/done.json", "w"), indent=1)
    print(f"S2_POSTHOC_GATE=PASS {task}/{name} rows={len(rows)}", flush=True)


def verify_frozen(name, N=8):
    """First target, frozen burn-in: the fully Bayesian rows must equal the frozen artifacts."""
    targets = json.load(open(f"artifacts/study2/main/{name}/config.json"))["targets"]
    _init(name, "fbfix", N, burn=S2Config().fb_burn)
    t, rows, _ = _work(targets[0])
    R = pd.read_csv(f"artifacts/study2/main/{name}/runs.csv.gz")
    worst = 0.0
    for row in rows:
        if row["policy"] not in ("fb:bmi", "fb:eig"):
            continue
        f = R[(R.target == t) & (R.rep == row["rep"]) & (R.N == N) & (R.policy == row["policy"])]
        assert len(f) == 1, (name, t, row["rep"], row["policy"])
        for m in ("ilogloss", "iae", "coverage"):
            worst = max(worst, abs(float(f.iloc[0][m]) - float(row[m])))
    print(f"[verify {name} N={N}] target {t}: max |diff| vs frozen = {worst:.3g}", flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["tiny", "fbfix"], required=True)
    ap.add_argument("--datasets", nargs="*", default=CONF)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--N", type=int, default=8, help="panel size (fbfix: 8, then 4)")
    ap.add_argument("--verify-frozen", action="store_true",
                    help="fbfix: reproduce the frozen fully Bayesian rows of the first target of each bank")
    ap.add_argument("--max-targets", type=int, default=None, help="smoke tests only")
    ap.add_argument("--out", default=OUT, help="smoke tests only")
    a = ap.parse_args()
    if a.verify_frozen:
        worst = max(verify_frozen(name, a.N) for name in a.datasets)
        print(f"S2_POSTHOC_REPRO={'PASS' if worst < 1e-9 else 'FAIL'} max_abs_diff={worst:.3g}", flush=True)
        if worst >= 1e-9:
            raise SystemExit(1)
        return
    for name in a.datasets:
        run(a.task, name, a.workers, a.max_targets, a.out, a.N)


if __name__ == "__main__":
    main()
