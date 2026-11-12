"""Exploratory follow-up RF2d (plan: docs/RESEARCH_LOG.md, iteration 4; written after RF2c was
stopped and before this script was run).

RF2c (scripts/fb_exact.py) sampled the model of Lemma 1 exactly, but on benchmarks whose
difficulty prior extends beyond the edge of the grid (ARC-C, HellaSwag) its translation move --
a Gibbs move over shifts of abilities, sampled difficulties and mu_b, corrected by the factor
(Z(eta)/Z(eta'))^I of the normalized prior -- almost never moved, and the overall location
did not converge (R-hat up to 2.8 on the first 14 panels). RF2d is RF2c plus, in every sweep,
a collapsed translation move (abilities and mu_b shifted by +-1, 2 or 4 difficulty-grid steps
with the item parameters summed out; Metropolis-Hastings, Jacobian 1), finer steps for the scale
move (log s in +-0.05, 0.1, 0.2, 0.4; RF2c's acceptance was 0-10% on HellaSwag), a burn-in of 150
sweeps and 100 retained draws per chain at thinning 2. Same targets, panels, holdouts,
starting points and replay arms as RF2-RF2c.

Outputs: artifacts/robust/fb_exact2/<bank>/{parts,rows.csv.gz,done.json}.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fb_exact as X  # noqa: E402

X.OUT = "artifacts/robust/fb_exact2"
X.BURN, X.DRAWS, X.THIN = 150, 100, 2
X.CHAIN_TAG = "rf2d_chain"
X.POLICY_TAG = "rf2d:"
X.CTRANS = True
X.LOG_TAG = "rf2d"
X.SCALE_LS = X.np.array([-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=X.CONF)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for n in a.datasets:
        X.run(n, a.workers)
    print("RF2D_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
