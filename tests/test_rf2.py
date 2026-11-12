"""Convergence diagnostics used by the robustness follow-up RF2 (scripts/fb_convergence.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import fb_convergence as fc  # noqa: E402


def _ar1(rng, M, N, rho):
    x = np.zeros((M, N))
    x[:, 0] = rng.normal(size=M)
    for t in range(1, N):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=M)
    return x


def test_rhat_and_ess_on_known_chains():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 2000))
    assert abs(fc.rhat_rank(iid) - 1) < 0.01
    assert 0.8 * 8000 < fc.ess(iid) < 1.2 * 8000
    ar = _ar1(rng, 4, 4000, 0.9)
    expected = 16000 * (1 - 0.9) / (1 + 0.9)
    assert 0.7 * expected < fc.ess(ar, rank=False) < 1.3 * expected
    shifted = iid.copy()
    shifted[0] += 1.0                         # one chain stuck elsewhere
    assert fc.rhat_rank(shifted) > 1.05


def test_diagnostics_keys():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(4, 300, 7))
    d = fc.diagnostics(T)
    for k in fc.PARAMS:
        assert {f"rhat_{k}", f"ess_{k}", f"mcse_{k}", f"sd_{k}"} <= set(d)
