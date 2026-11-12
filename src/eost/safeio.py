"""Safe I/O helpers for third-party raw data.

Third-party pickles can execute arbitrary code when loaded. The only pickle used
by EOST (tinyBenchmarks ``lb.pickle``) was audited with ``pickletools`` and
references exclusively ``numpy.dtype`` and ``numpy.core.numeric._frombuffer``.
:class:`RestrictedUnpickler` enforces that allow-list at load time, so any other
global reference raises instead of executing.
"""
from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path

_ALLOWED_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only resolves an explicit allow-list of numpy globals."""

    def find_class(self, module: str, name: str):  # noqa: D401
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed")
        if module.startswith("numpy.core."):
            module = "numpy._core." + module[len("numpy.core."):]
        return super().find_class(module, name)


def safe_load_pickle(path: str | Path):
    data = Path(path).read_bytes()
    return RestrictedUnpickler(io.BytesIO(data)).load()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
