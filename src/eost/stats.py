"""Paired, target-level statistics used for every confirmatory contrast."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014); >0 favours positive d."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ranks for ties in |d|
    ad = np.abs(d)
    for v in np.unique(ad):
        sel = ad == v
        r[sel] = r[sel].mean()
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def paired_contrast(base: np.ndarray, new: np.ndarray, lower_is_better: bool = True,
                    seed: int = 0) -> dict:
    """Compare ``new`` with ``base`` on paired per-target values.

    The improvement is ``d = base - new`` when lower is better (so d > 0 means the
    new method is better). Two-sided Wilcoxon signed-rank test.
    """
    base, new = np.asarray(base, float), np.asarray(new, float)
    d = (base - new) if lower_is_better else (new - base)
    n = len(d)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bootstrap_ci(d, np.mean, seed=seed)
    rel = float(np.mean(d) / np.mean(base)) if lower_is_better and np.mean(base) != 0 else np.nan
    return {"n": n, "mean_base": float(base.mean()), "mean_new": float(new.mean()),
            "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "median_diff": float(np.median(d)),
            "rel_improvement": rel, "wins": wins, "losses": losses, "p": p,
            "rank_biserial": rank_biserial(d)}
