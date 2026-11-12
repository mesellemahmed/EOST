"""Container doctor: environment facts + data-manifest integrity (prints DOCTOR=PASS/FAIL)."""
from __future__ import annotations

import json
import os
import platform
import sys


def main() -> int:
    import numpy as np
    import pandas as pd
    import scipy

    ok = True
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "cpu_count": os.cpu_count(), "in_container": os.path.exists("/.dockerenv"),
    }
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_devices"] = [d.platform for d in jax.devices()]
        ok &= all(p == "cpu" for p in info["jax_devices"])
    except Exception as e:  # pragma: no cover
        info["jax"] = f"unavailable: {e}"
    man = "data/processed/MANIFEST.json"
    if os.path.exists(man):
        from eost.safeio import sha256_file
        m = json.load(open(man))
        bad = [k for k, v in m["datasets"].items()
               if not os.path.exists(v["file"]) or sha256_file(v["file"]) != v["sha256"]]
        info["datasets_checked"] = len(m["datasets"])
        info["datasets_hash_mismatch"] = bad
        ok &= not bad
    print(json.dumps(info, indent=1))
    print(f"DOCTOR={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
