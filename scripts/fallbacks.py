"""Add \\providecommand fallbacks ('??') for every macro used in the paper but not yet generated.

Development aid only: the release gate requires a final PDF without any '??'.
"""
import glob
import re

defined = set()
for f in ["paper/generated/numbers.tex", "paper/generated/numbers2.tex", "paper/generated/numbers3.tex",
          "paper/main.tex"] + sorted(glob.glob("paper/sections/*.tex")):
    try:
        defined |= set(re.findall(r"\\(?:newcommand|providecommand)\{\\([A-Za-z]+)\}", open(f).read()))
    except FileNotFoundError:
        pass
used = set()
for f in glob.glob("paper/sections/*.tex") + glob.glob("paper/tables/*.tex") + ["configs/deviations.txt"]:
    used |= set(re.findall(r"\\([A-Z][A-Za-z]*[a-z][A-Za-z]*)", open(f).read()))
latex = {"BlackBox", "Require", "State", "For", "EndFor", "Return", "Comment", "If", "EndIf", "Else", "IfFileExists",
         "ShortHeadings", "Sigma", "Lambda", "Phi", "Theta", "Pr", "Delta", "Gamma", "Omega", "Psi", "Pi", "Upsilon",
         "Xi", "DeviationText", "LaTeX", "TeX", "AND", "Rightarrow", "Leftarrow", "Longrightarrow", "FloatBarrier"}
missing = sorted(u for u in used if u not in defined and u not in latex)
with open("paper/sections/defaults.tex", "a") as fh:
    for n in missing:
        fh.write("\\providecommand{\\%s}{\\textbf{??}}\n" % n)
print("FALLBACKS_ADDED", len(missing))
