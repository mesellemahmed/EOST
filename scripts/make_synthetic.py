"""Generate the synthetic banks used by the ground-truth study (runs in the container)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.safeio import sha256_file  # noqa: E402
from eost.synthetic import make_bank  # noqa: E402

SPECS = {
    "synth_2pl": dict(seed=11),
    "synth_guess": dict(seed=12, guessing=0.25, frac_guess=0.6),
    "synth_2dim": dict(seed=13, two_dim_rho=0.6),
}

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    man = {}
    for name, kw in SPECS.items():
        path = f"data/processed/{name}.npz"
        info = make_bank(path, name=name, **kw)
        info["sha256"] = sha256_file(path)
        man[name] = info
        print(name, info)
    with open("data/processed/SYNTH_MANIFEST.json", "w") as fh:
        json.dump(man, fh, indent=1)
    print("SYNTH_GATE=PASS")
