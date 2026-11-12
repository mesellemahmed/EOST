"""Strict family clusters for study 2 -> configs/study2_clusters.json (run once, then frozen).

Joint residual correlations over banks that share the same systems (the six Open LLM
Leaderboard banks; CRUXEval-O and -I), per bank otherwise. Threshold 0.5 (S2Config).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.study2 import OLLB, S2Config, strict_clusters_from  # noqa: E402

GROUPS = [OLLB, ["cruxeval_o", "cruxeval_i"], ["swe_verified"], ["swe_lite"], ["livecodebench"],
          ["synth_2pl"], ["synth_guess"], ["synth_2dim"]]


def main():
    cfg = S2Config()
    out = {"threshold": cfg.resid_thr, "rule": "connected components of heuristic family links OR "
           "IRT-residual correlation >= threshold (joint over banks sharing systems)", "banks": {}}
    for g in GROUPS:
        banks = [load_bank(f"data/processed/{b}.npz") for b in g]
        lab = strict_clusters_from(banks, cfg.resid_thr)
        for B in banks:
            sz = np.bincount(lab)
            out["banks"][B.name] = {"systems": [str(s) for s in B.systems], "cluster": lab.tolist(),
                                    "n_clusters": int(len(sz)), "max_size": int(sz.max())}
            print(f"{B.name:16s} J={B.J:3d} clusters={len(sz):3d} max={sz.max():3d}")
    json.dump(out, open("configs/study2_clusters.json", "w"))
    print("CLUSTERS_WRITTEN")


if __name__ == "__main__":
    main()
