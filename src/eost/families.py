"""Strict family clusters of systems (study 2).

Heuristic families (same model family, same organisation; for SWE-bench the set of
underlying LLMs) do not capture fine-tunes across organisations, renamed base models,
merges or distillation. A *strict cluster* is a connected component of the graph whose
edges join two systems if they share a heuristic family OR their verdict vectors agree
almost perfectly beyond chance (Cohen's kappa over items >= ``kappa_thr``, the
near-duplicate threshold of the first study). Connected components are deliberately
conservative: anything linked by a chain of near-duplicates is one cluster.

Strict clusters are used (i) to exclude the target's whole cluster from its calibration
panel, (ii) to pick at most one target per cluster, and (iii) as the resampling unit of
cluster-aware inference.
"""
from __future__ import annotations

import numpy as np

from .data import Bank


def kappa_matrix(Y: np.ndarray) -> np.ndarray:
    """Cohen's kappa between every pair of systems (rows of a binary J x I matrix)."""
    X = np.asarray(Y, dtype=float)
    I = X.shape[1]
    agree11 = X @ X.T
    agree00 = (1 - X) @ (1 - X).T
    po = (agree11 + agree00) / I
    p = X.mean(1)
    pe = np.outer(p, p) + np.outer(1 - p, 1 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(pe < 1, (po - pe) / np.maximum(1 - pe, 1e-12), 1.0)
    np.fill_diagonal(k, 1.0)
    return k


def heuristic_adjacency(bank: Bank) -> np.ndarray:
    J = bank.J
    A = np.zeros((J, J), dtype=bool)
    for j in range(J):
        A[j] = bank.related(j)
    return A | A.T


def strict_clusters(bank: Bank, kappa_thr: float = 0.8) -> np.ndarray:
    """Cluster label (0..C-1) of every system; labels ordered by first appearance."""
    A = heuristic_adjacency(bank) | (kappa_matrix(bank.Y) >= kappa_thr)
    J = bank.J
    parent = np.arange(J)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.nonzero(np.triu(A, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(j) for j in range(J)])
    _, labels = np.unique(roots, return_inverse=True)
    # relabel by first appearance so that labels are deterministic and readable
    order = {}
    out = np.empty(J, dtype=int)
    for j, l in enumerate(labels):
        out[j] = order.setdefault(int(l), len(order))
    return out
