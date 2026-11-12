# TeX Live image used only to compile the manuscript (./run paper).
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
      texlive-science texlive-bibtex-extra lmodern make \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /work/paper
CMD ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
