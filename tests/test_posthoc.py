"""The post hoc copy of the Gibbs sampler must reproduce the frozen sampler exactly."""
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eost.fullbayes import gibbs_item_marginals  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid  # noqa: E402
import posthoc_s2_corrections as ph  # noqa: E402


def test_sampler_copy_reproduces_frozen_sampler():
    rng = np.random.default_rng(2)
    th = rng.normal(size=12); a = np.exp(rng.normal(0, .4, 90)); b = rng.normal(0, 1, 90)
    Y = (rng.random((12, 90)) < expit(a * (th[:, None] - b))).astype(float)
    cal, W, grid = calibrate(Y[:8], "2pl", Priors(), True)
    W1, _ = gibbs_item_marginals(Y[:8], cal.theta_cal, cal.priors, grid, np.random.default_rng(7), burn=5, draws=3, thin=2)
    W2, states = ph.gibbs_with_states(Y[:8], cal.theta_cal, cal.priors, grid, np.random.default_rng(7), 5, 3, 2)
    assert np.array_equal(W1, W2) and len(states) == 3
    m, M = ph.joint_mixture_model(Y[:8], states, grid, default_theta_grid())
    assert M == 3 and m.P.shape == (90, 3 * 201)
