"""Self-contained LaTeX sources of the short article: one main.tex plus the files it needs.

Writes OUT/main.tex, in which the section files and tables of paper/short.tex are inlined and only
the number macros that the article uses are defined (copied verbatim from paper/generated/), and
copies the figures it includes, refs.bib and jmlr2e.sty. Usage (in the container):
`python scripts/flatten_short.py dist/eost-v5.0-latex`. Presentation only; no number is computed.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys

PAPER = "paper"


def definitions() -> dict[str, str]:
    """name -> full \\newcommand line(s) from the generated number files."""
    defs = {}
    for f in sorted(glob.glob(f"{PAPER}/generated/numbers*.tex")):
        s = open(f, encoding="utf-8").read()
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", s):
            j, depth = m.end(), 0
            while True:                      # argument in braces, possibly nested
                c = s[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            defs.setdefault(m.group(1), s[m.start():j + 1])
    return defs


def inline(path: str) -> str:
    s = open(f"{PAPER}/{path}", encoding="utf-8").read()
    return re.sub(r"\\input\{(short/[^}]*)\}",
                  lambda m: inline(m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex")).rstrip("\n"), s)


def main(out: str) -> None:
    body = inline("short.tex")
    defs = definitions()
    used, frontier = set(), set(re.findall(r"\\([A-Za-z]+)", body))
    while frontier:                          # macros used by the text, and by those macros
        new = {n for n in frontier if n in defs and n not in used}
        used |= new
        frontier = set().union(*(set(re.findall(r"\\([A-Za-z]+)", defs[n])) for n in new)) if new else set()
    block = ("% Numbers generated from the artifacts (paper/generated/numbers*.tex of the repository)\n"
             + "\n".join(defs[n] for n in sorted(used)) + "\n")
    body = re.sub(r"(\\input\{generated/numbers[0-9]*\}\n)+", lambda m: block, body, count=1)
    assert "\\input{" not in body, "unresolved input"
    os.makedirs(f"{out}/figures", exist_ok=True)
    open(f"{out}/main.tex", "w", encoding="utf-8").write(body)
    for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", body))):
        shutil.copy(f"{PAPER}/{fig}", f"{out}/{fig}")
    for f in ("refs.bib", "jmlr2e.sty"):
        shutil.copy(f"{PAPER}/{f}", f"{out}/{f}")
    print(f"FLAT_MACROS={len(used)} FLAT_FIGURES={len(os.listdir(out + '/figures'))}")


if __name__ == "__main__":
    main(sys.argv[1])
