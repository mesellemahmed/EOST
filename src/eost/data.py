"""Loading processed response matrices and building leakage-free panels."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Bank:
    name: str
    Y: np.ndarray            # (J, I) uint8
    systems: np.ndarray
    families: np.ndarray
    items: np.ndarray
    meta: dict
    family_sets: list | None = None   # SWE-bench: set of underlying LLMs per system
    theta_true: np.ndarray | None = None  # synthetic banks only

    @property
    def J(self):
        return self.Y.shape[0]

    @property
    def I(self):
        return self.Y.shape[1]

    def related(self, j: int) -> np.ndarray:
        """Boolean mask of systems related to system j (same family), including j."""
        if self.family_sets is not None:
            fj = self.family_sets[j]
            return np.array([len(fj & fk) > 0 for fk in self.family_sets])
        return self.families == self.families[j]

    def eligible_calibrators(self, j: int, max_kappa: float | None = None) -> np.ndarray:
        """Systems allowed in the calibration panel of target j.

        Excludes the target, its family, and (optionally) near-duplicates whose
        chance-corrected agreement with the target (Cohen's kappa over items) is
        >= ``max_kappa``. The near-duplicate rule only removes calibrators, so it
        can only make calibration harder, never leak target information into
        item parameters.
        """
        ok = ~self.related(j)
        if max_kappa is not None:
            Y = self.Y.astype(float)
            po = (Y == Y[j]).mean(1)
            p, q = Y.mean(1), Y[j].mean()
            pe = p * q + (1 - p) * (1 - q)
            kappa = np.where(pe < 1, (po - pe) / np.maximum(1 - pe, 1e-12), 1.0)
            ok &= kappa < max_kappa
        return np.flatnonzero(ok)


def load_bank(path: str) -> Bank:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    name = path.rsplit("/", 1)[-1].replace(".npz", "")
    fs = None
    if "family_models" in meta:
        fm = meta["family_models"]
        fs = [set(fm[s]) for s in z["systems"]]
    tt = z["theta_true"] if "theta_true" in z.files else None
    return Bank(name=name, Y=z["Y"].astype(np.uint8), systems=z["systems"], families=z["families"],
                items=z["items"], meta=meta, family_sets=fs, theta_true=tt)


def effective_panel_size(Ypanel: np.ndarray) -> float:
    """Participation ratio of the system-by-system correlation spectrum.

    Equals N for mutually uncorrelated systems and tends to 1 when all systems
    produce the same response pattern (e.g. prompt variants of one checkpoint).
    """
    X = Ypanel.astype(float)
    X = X - X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    keep = sd[:, 0] > 0
    X = X[keep] / sd[keep]
    if X.shape[0] < 2:
        return float(X.shape[0])
    C = X @ X.T / X.shape[1]
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def stratified_targets(Y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n target systems spread over accuracy deciles (all if J <= n)."""
    J = Y.shape[0]
    if J <= n:
        return np.arange(J)
    acc = Y.mean(1)
    order = np.argsort(acc, kind="stable")
    bins = np.array_split(order, 10)
    per = [n // 10 + (1 if k < n % 10 else 0) for k in range(10)]
    pick = [rng.choice(b, size=min(p, len(b)), replace=False) for b, p in zip(bins, per)]
    return np.sort(np.concatenate(pick))
