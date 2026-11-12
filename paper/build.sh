#!/usr/bin/env bash
# Builds the article (paper/short.pdf, about 25 pages) and its extended version (paper/main.pdf,
# one continuous article that contains every secondary table and figure).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error short.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
