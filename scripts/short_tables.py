"""Tables of the short article (paper/short.tex): copies of generated tables with one-line captions.

The tabular bodies are copied unchanged from paper/tables/ (generated from the artifacts by the report
scripts); only the caption is replaced, and what the long caption explained is said in the text of the
short article. Run after the report scripts: `python scripts/short_tables.py` (also run by
`paper/build.sh`). Presentation only; no number is computed here.
"""
from __future__ import annotations

import os
import re

SRC, DST = "paper/tables", "paper/short/tables"

CAPTIONS = {
    "datasets": "Response matrices of the eight confirmatory and three development banks.",
    "confirmatory_N8": r"Study 1: relative improvement (\%) of EOST-M for H1--H4 at $N=8$.",
    "factorial": r"Study 1: gain over MFI:pi by criterion and measurement model (median, \%).",
    "study2": r"Study 2: frozen hypotheses at $N=8$ and their decisions.",
    "reduction": r"Study 2: absolute score error (points) after 25, 50 and 100 items at $N=8$.",
    "cost": r"Cost of adaptive evaluation with a fresh panel of eight calibrators.",
}


def caption_span(s: str) -> tuple[int, int]:
    """Start and end (exclusive) of the argument of the first \\caption{...}."""
    i = s.index("\\caption{") + len("\\caption{")
    depth, j = 1, i
    while depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return i, j - 1


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    for name, cap in CAPTIONS.items():
        s = open(f"{SRC}/{name}.tex", encoding="utf-8").read()
        i, j = caption_span(s)
        s = s[:i] + cap + s[j:]
        s = re.sub(r"\\begin\{table\}\[[a-z!]*\]", r"\\begin{table}[htbp]", s, count=1)
        open(f"{DST}/{name}.tex", "w", encoding="utf-8").write(s)
    print(f"SHORT_TABLES={len(CAPTIONS)}")


if __name__ == "__main__":
    main()
