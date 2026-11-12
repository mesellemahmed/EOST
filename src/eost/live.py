"""Live (prospective) administration of an EOST-M test to a model that is actually queried.

:class:`LiveTest` performs exactly the computations of the replay engine (same
posterior update, same full-bank score estimate and 90% interval), but it asks for one
verdict at a time instead of reading a recorded response vector, so the same frozen
calibration and policy can be used on a model that has never been evaluated. A unit test
checks that feeding it recorded verdicts reproduces ``replay`` exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .replay import Precomputed, _argmax_tiebreak, _posterior, h2


class LiveTest:
    def __init__(self, pre: Precomputed, pool: np.ndarray, I: int, criterion: str = "eig",
                 rng: np.random.Generator | None = None, ci_level: float = 0.90):
        self.pre, self.m, self.I = pre, pre.model, I
        self.criterion = criterion
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.z = norm.ppf(0.5 + ci_level / 2)
        self.remaining = np.zeros(I, dtype=bool)
        self.remaining[np.asarray(pool)] = True
        self.logpost = pre.log_prior.copy()
        self.S_un = self.m.P.sum(0)
        self.V_un = pre.PQ.sum(0)
        self.observed_sum, self.n_obs = 0.0, 0
        self.history = []

    def next_item(self) -> int:
        pi = _posterior(self.logpost)
        cand = np.flatnonzero(self.remaining)
        if self.criterion == "eig":
            p_all = self.m.P[cand] @ pi
            sc = h2(p_all) - self.pre.HP[cand] @ pi
        elif self.criterion == "random":
            return int(self.rng.choice(cand))
        else:
            raise ValueError(self.criterion)
        return int(cand[_argmax_tiebreak(sc, self.rng)])

    def record(self, i: int, y: int) -> dict:
        pre, m = self.pre, self.m
        self.logpost = self.logpost + (pre.logP[i] if y == 1 else pre.log1mP[i])
        self.remaining[i] = False
        self.observed_sum += y; self.n_obs += 1
        self.S_un = self.S_un - m.P[i]; self.V_un = self.V_un - pre.PQ[i]
        pi = _posterior(self.logpost)
        mean_k = self.observed_sum + self.S_un
        mu = float(mean_k @ pi)
        var = float(self.V_un @ pi + ((mean_k - mu) ** 2) @ pi)
        sd = np.sqrt(max(var, 0.0))
        th = m.theta
        tm = float(th @ pi)
        state = {"t": self.n_obs, "item": int(i), "y": int(y), "est": mu / self.I,
                 "lo": (mu - self.z * sd) / self.I, "hi": (mu + self.z * sd) / self.I,
                 "theta_mean": tm, "theta_sd": float(np.sqrt(max(((th - tm) ** 2) @ pi, 0.0)))}
        self.history.append(state)
        return state
