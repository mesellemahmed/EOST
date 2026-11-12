# Calibration Uncertainty in Adaptive Large Language Model Evaluation: Theory and Empirical Evidence

Ahmed Mesellem, Lotfi Bouhedja, Ahmed Taalbi, Mokhtar Berrahal and Mohammed Mana
(STIC Laboratory, Abou Bekr Belkaid University, Tlemcen; LESAIL Laboratory, Ecole Supérieure
Normale d'Oran). Manuscript submitted to the *Journal of Machine Learning Research*.

This repository contains the code, the data-building scripts, both frozen analysis plans,
every experimental artifact with its SHA-256 hash, and the manuscript sources of the article
(`paper/short.pdf`) and of its extended version (`paper/main.pdf`). The project's working name
is **EOST** (entropy-oracled information-optimal self-testing); the method studied is
**EOST-M**, a calibration-marginal adaptive test. Everything runs in Docker, and every number
in the article is generated from the released artifacts by the released code.

**For reviewers.** `./run build && ./run doctor && ./run claims && ./run verify` (a few minutes)
rebuilds the images, checks the data hashes, re-evaluates every qualitative statement of the
article on the artifacts, and re-checks the frozen plans and the artifact manifest; `./run paper`
rebuilds both PDFs from the artifacts. [`docs/REVIEWER_FAQ.md`](docs/REVIEWER_FAQ.md) answers
anticipated questions with the location of each answer, and [`docs/AUDIT.md`](docs/AUDIT.md)
records the adversarial checks of the claims.

Adaptive tests built with item response theory (IRT) can evaluate a language model with a
fraction of a benchmark, but they need item parameters calibrated on other systems. With
the small calibration panels typical of new, private or expensive benchmarks (4–16
systems), those parameters—especially discriminations—are poorly identified. EOST-M
lets every item enter the test through its **expected response function** under an
empirical-Bayes item prior, selects items by **expected information gain**, and leaves
verdicts to an **independent deterministic oracle**.

## Results at a glance

Two studies with analysis plans frozen before execution: eight benchmarks from five
domains (629 systems), calibration panels of 4–64 systems, primary panel size N = 8.
Study 1 (leave-family-out replay, 172,620 adaptive tests); study 2 (one target per family
cluster, 267 targets, strong competitors, temporal split). “sig.” = benchmarks with a
Holm-significant difference for EOST-M / against it; a hypothesis is supported with ≥ 5/8
and none against.

| EOST-M versus … (N = 8) | Study 1 | Study 2 |
|---|---|---|
| plug-in Fisher CAT (MFI:pi), held-out log loss | 7/0 sig., median +5.2% (strict clusters as units: 4/0, not supported) | 5/0, +5.9% — **supported** |
| uniform subsampling (Naive), score error | 7/0, +54.4% | 6/0, +53.9% — **supported** |
| random items, same calibrated model (Random:mg), score error | 7/0, median +25.5% (secondary) | 4/0, +21.1% — not supported (positive on 8/8) |
| regularized Rasch-type plug-in, log loss | post hoc: 4/0, +2.1% | 5/0, +2.4% — **supported** |
| Rasch-type marginal, log loss | post hoc | 4/0, +1.8% — not supported |
| posterior-draw marginal (PDM) Bayesian CAT, log loss (equivalence ±1%) | — | 4/8 banks within ±1% (needs 6) — not supported; no bank significant either way; after a post hoc burn-in correction 3/8, median +0.4% (Bayesian CAT worse) |
| ATLAS-style 3PL CAT / tinyBenchmarks / MINCE, score error | — | 4/0, 4/0, 5/0 sig.; +26%, +30%, +45%; positive on 8/8 each |
| coverage of the 90% score interval | 0.75 (plug-in 0.60, Naive 0.90) | 0.74 raw, 0.86 recalibrated (5/8 banks in [0.85, 0.95]; not supported) |

Also: information vs entropy (H1) is consistent in sign but small and not supported;
interval recalibration does not improve the interval score; on the 31 latest SWE-bench
Verified submissions (calibrated on earlier ones) 10 of 11 study-2 contrasts favour EOST-M
significantly; with eight fresh calibrators adaptive testing pays off after 8.4–10 new
systems (immediately when leaderboard results already exist).

Robustness follow-ups (each planned before running; no frozen decision
changes): R2 and C1 are unchanged when the clusters never see the target's responses
(metadata-only, leave-bank-out residuals) and at residual thresholds 0.4–0.7; the frozen Bayesian-CAT sampler does not converge for the scale hyperparameters even with four long chains, and exploratory samplers with collapsed moves (RF2b–RF2d) come close without meeting the convergence criterion, but across all 18 arm/sampler combinations the median difference from EOST-M in held-out log loss stays within ±0.45%;
the effects that missed the frozen threshold by one benchmark point the same way across
benchmarks in a secondary random-effects meta-analysis, but with large heterogeneity, and when
the five Open LLM Leaderboard benchmarks (which share their 395 systems) count as one source
only C2 keeps an interval above zero (6 of 11 hypotheses overall).

Full tables: [`RESULTS.md`](RESULTS.md) (generated from the artifacts). The article is
[`paper/short.pdf`](paper/short.pdf) (25 pages; sources `paper/short.tex` and `paper/short/`,
or one self-contained `main.tex` with `./run flatten`); its extended version
[`paper/main.pdf`](paper/main.pdf) is one continuous article that contains every secondary
table and figure (no appendices: each proof follows its statement and each additional analysis
sits in the section that discusses it). Adversarial audit: [`docs/AUDIT.md`](docs/AUDIT.md);
research log with every plan written before its run:
[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Everything runs in Docker

The host only orchestrates Git and Docker. All science (data parsing, calibration,
adaptive replay, statistics, figures, LaTeX) runs in containers, and every scientific
container runs **without network access** (the only exception is `./run live`, which
must reach the model you serve).

```bash
./run build        # eost-runtime:cpu (Ubuntu 24.04, Python 3.12.3, pinned deps) + eost-paper:texlive
./run doctor       # environment facts + SHA-256 check of processed data  -> DOCTOR=PASS
./run test         # unit tests (exact information cap for the logistic link, replay/live equivalence, Gibbs sampler, ...)
./run fetch        # sparse git checkouts of the pinned public sources (host git only)
./run data         # build response matrices -> data/processed/*.npz + MANIFEST.json
./run synthetic    # synthetic ground-truth banks
# study 1 (frozen plan: configs/preregistration.md + PREREGISTRATION.lock.json)
./run confirm      # refuses to start if the frozen plan changed
./run sensitivity  # 1PL, fixed-prior and near-duplicate sensitivity analyses (complete banks are skipped)
./run regularized  # post hoc 1PL variant with log a ~ N(0, 0.05^2)
./run analyze confirm      # (and sens_1pl, sens_fixedprior, sens_dedup, sens_1pl_reg)
./run explore      # exploratory ranking-fidelity analysis
./run strong       # post hoc strong baselines (PH1)
./run shrinkage    # exploratory regularization path (PH2)
./run coverage     # exploratory coverage decomposition (PH3)
./run clusters     # study 1 re-analysed with family clusters as units (post hoc)
./run cost         # cost and break-even analysis
./run eb           # robustness of the empirical-Bayes hyperprior
# study 2 (frozen plan: configs/study2_plan.md + STUDY2.lock.json)
./run study2       # main, temporal and synthetic components; checks STUDY2_GATE; resumable
./run analyze2     # decisions under the frozen study-2 rules
./run s2fix        # post hoc corrections decided before the study-2 results (tinyBenchmarks anchor audit,
                   # PDM Bayesian CAT with a longer burn-in + exploratory joint model, sign check)
./run robust       # robustness follow-ups: RF1 target-blind clustering and
                   # thresholds, RF2 MCMC convergence of the Bayesian CATs (+ exploratory RF2b scale move,
                   # RF2c exact sampler of Lemma 1's model), RF3 cross-benchmark meta-analysis (several hours on 2 CPUs)
# outputs
./run report       # figures + LaTeX tables + number macros from the artifacts
./run claims       # every qualitative statement of the paper checked on the artifacts -> CLAIMS_GATE=PASS
./run verify       # PREREG_GATE + STUDY2_GATE + DATA_GATE + artifact hash manifest
./run paper        # paper/short.pdf (the article) and paper/main.pdf (extended version)
./run flatten      # dist/eost-latex/: the article as one self-contained main.tex + figures
./run all          # the whole pipeline end to end
# live, prospective evaluation of a model you serve (OpenAI-compatible endpoint; not run in the paper)
./run live --bank gsm8k --out artifacts/live/mymodel --endpoint http://localhost:8000/v1 --model mymodel --full
```

Every step is fail-fast: a failing gate returns a non-zero exit code. Building behind a
TLS-intercepting proxy: pass the proxy CA as a BuildKit secret
(`docker build --secret id=extra_ca,src=ca.crt ...`); it is used only while installing
Python wheels and never stored in the image. Image IDs, the base-image digest and the full
package list are in [`configs/ENVIRONMENT.json`](configs/ENVIRONMENT.json).

Compute: study 1 (172,620 adaptive tests of 100 items) needs about 4 h on two CPU cores,
study 2 about 3.3 h (main, temporal and synthetic components) plus about 2.5 h for its post hoc corrections. No GPU and no LLM inference are needed: all verdicts are
recorded public item-level results.

## How the text is tied to the numbers

* Every number in the manuscript is a LaTeX macro written from the artifacts
  (`scripts/make_report.py`, `scripts/report_study2.py` → `paper/generated/`).
* Every qualitative statement is a predicate in `scripts/check_claims.py`;
  `./run claims` re-evaluates all of them and writes `artifacts/CLAIMS.json`.
* Both analysis plans were hashed and committed **before** their runs
  (`configs/PREREGISTRATION.lock.json`, `configs/STUDY2.lock.json`; `./run verify`
  re-checks them; the git history shows each freeze commit before the corresponding
  artifacts). No external registry or time stamp was reachable from the execution
  environment, so the paper calls them *frozen analysis plans*. The tagged release (v5.0) and
  its Zenodo deposit (`.zenodo.json` holds the metadata) make the final package immutable; they
  do not turn the studies into preregistrations.

## Repository layout

| Path | Content |
|---|---|
| `src/eost/irt.py`, `replay.py`, `experiment.py`, `data.py`, `stats.py` | study-1 modules (frozen): calibration (joint MAP, empirical Bayes, item grid posteriors, ERF), adaptive replay, protocol, statistics |
| `src/eost/study2.py`, `replay2.py`, `irt2.py`, `fullbayes.py`, `reduction.py`, `families.py` | study-2 modules (frozen): strict clusters, LOCO interval recalibration, ATLAS-style 3PL, Student-t / cross-fitted EB, Gibbs sampler for the fully Bayesian CAT, tinyBenchmarks and MINCE |
| `src/eost/live.py`, `scripts/live_eost.py` | live (prospective) protocol |
| `scripts/run_benchmark*.py`, `scripts/run_study2.py` | runners (checkpointed, deterministic seeds) |
| `scripts/analyze*.py`, `scripts/make_report.py`, `scripts/report_study2.py`, `scripts/check_claims.py` | statistics, figures/tables/macros, claim checks |
| `configs/` | frozen plans and locks, hypotheses, cluster file, environment, deviations |
| `artifacts/` | every run output + `ARTIFACT_MANIFEST.json` (SHA-256 of every file) |
| `paper/` | LaTeX sources (JMLR style) of the article (`short.tex`, `short/`) and of the extended version (`main.tex`, `sections/`), generated figures/tables/macros, both PDFs |
| `docs/` | reviewer FAQ, adversarial audit, research log, data format |

## Data and licences

| Bank | Source (pinned commit) | Licence of source | Redistributed here |
|---|---|---|---|
| Open LLM Leaderboard v1 (GSM8K, ARC, HellaSwag, WinoGrande, MMLU, TruthfulQA; 395 models) | `felipemaiapolo/tinyBenchmarks@e9a8b10` | MIT | yes (processed matrices) |
| CRUXEval-O / -I (31 models) | `facebookresearch/cruxeval@190faf1` | MIT | yes (processed matrices) |
| SWE-bench Verified / Lite (175 / 84 systems) | `SWE-bench/experiments@40f164d` | no licence file | **no** — rebuild with `./run fetch && ./run data`; expected SHA-256 in `data/processed/MANIFEST.json` |
| LiveCodeBench (28 models) | `LiveCodeBench/livecodebench.github.io@f186ab5` | no licence file | **no** — rebuild as above |
| GSM8K item texts (live protocol only) | `openai/grade-school-math@3101c7d` | MIT | fetched by `./run fetch` |
| Synthetic banks (3) | `scripts/make_synthetic.py` | MIT | yes |

The pickle shipped by tinyBenchmarks is loaded with an allow-list unpickler
(`src/eost/safeio.py`) that only resolves `numpy.dtype` and
`numpy._core.numeric._frombuffer`; it was audited with `pickletools` first.

## Provenance and honesty rules

* Development banks (SWE-bench Lite, CRUXEval-I, TruthfulQA, synthetic 2PL) were used to
  design both protocols and are reported separately.
* Infrastructure restarts interrupted study 1 twice; completed banks were kept and the rest
  were run with a checkpointing wrapper that reproduces the frozen runner exactly
  (`configs/deviations.txt`).
* Study 2 ran without interruption under its frozen plan. During the run, before any result
  was inspected, a code review found that tinyBenchmarks can use fewer anchors than its
  budget with eight calibrators and that the fully Bayesian sampler's burn-in was too short
  on some banks. The frozen run was completed unchanged and every frozen decision is reported
  as computed; the post hoc corrections (`./run s2fix`) were specified in
  `docs/RESEARCH_LOG.md` before the results were looked at, reproduce the frozen rows exactly
  when given the frozen settings, and are labelled post hoc in the paper. One analysis choice
  of the anchor audit (the Holm family) was corrected after the first look, towards the more
  conservative frozen family; both versions are kept and logged.
* The pilot program S0–S4B (math word problems, `Qwen3-8B` target) predates this pipeline.
  Its numbers are quoted **as recorded** in the project log, including failed gates (S1-R2,
  S2, original S4); S4B is incomplete; its raw artifacts are not part of this release and
  were not re-executed. It ran in an earlier runtime image of the project (see
  `configs/ENVIRONMENT.json`), not in the image of this release.
* No LLM inference is performed by this release; a live protocol is provided but was not
  run (no access to model weights in the execution environment).

## Citation

See `CITATION.cff`. Code is released under the MIT licence (`LICENSE`).
