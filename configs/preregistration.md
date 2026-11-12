# EOST v2 — Pre-registered confirmatory analysis plan

Frozen before any confirmatory run. The SHA-256 of this file, of
`configs/hypotheses.json` and of every analysis-relevant source file is stored
in `configs/PREREGISTRATION.lock.json`; `./run confirm` refuses to start if any
of them changed (gate `PREREG_GATE`).

## 1. What was decided during development (exploratory, reported as such)

Development used only `synth_2pl`, `swe_lite`, `cruxeval_i` and
`ollb_truthfulqa` (30 targets, 3 replicates). Decisions taken there:

1. Item hyperprior: fixed priors (log a ~ N(0,1), b ~ N(0,2²)) made predictions
   for items that no calibrator solved prior-driven (SWE-bench Lite: ~20% of
   items never resolved). We therefore adopt an **empirical-Bayes item
   hyperprior** (type-II ML by EM on the item grid, given calibrator abilities),
   shared by the plug-in and the marginal model, for every policy.
2. Plug-in and marginal models share calibrator abilities (joint MAP) and the
   hyperprior; plug-in = conditional MAP, marginal = exact grid posterior
   (26 × 129 grid over log a ∈ [-2.5, 2.5], b ∈ [-8, 8]).
3. Ability grid: 201 points on [-5, 5], prior N(0, 1).
4. Policy set fixed to the 12 policies listed in §3.
5. Near-duplicate sensitivity rule uses Cohen's kappa (raw agreement is
   misleading on high-accuracy banks such as HellaSwag).

Only descriptive statistics of the confirmatory banks (sizes, accuracy ranges,
eligible panel sizes, near-duplicate rates) were inspected before freezing. No
policy outcome on a confirmatory bank was computed before freezing.

## 2. Confirmatory banks

`ollb_gsm8k`, `ollb_arc`, `ollb_hellaswag`, `ollb_winogrande`, `ollb_mmlu`
(Open LLM Leaderboard v1 item-level verdicts, 395 models; HellaSwag and MMLU
are uniform random subsets of 2,000 items, seed 20261029), `swe_verified`
(175 systems, 500 instances), `livecodebench` (28 models, 1,055 problems),
`cruxeval_o` (31 models, 800 functions). Synthetic banks `synth_2pl`,
`synth_guess`, `synth_2dim` are run with the same protocol and analysed as a
separate family.

## 3. Protocol

* Targets: all systems if J ≤ 60, otherwise 60 systems stratified by accuracy
  decile (seeded). A target needs ≥ 4 eligible calibrators.
* Eligible calibrators: all systems except the target's family (organisation
  namespace for Open LLM Leaderboard; manual model-line map for LiveCodeBench;
  model-line prefix for CRUXEval; any shared underlying LLM for SWE-bench).
* Replicates r = 0..4: a holdout H (20% of items) drawn per (target, r), shared
  by all policies and panel sizes; panel of N calibrators drawn per (target, r, N).
* Panel sizes N ∈ {4, 8, 16, 32, 64} (those exceeding the eligible pool are skipped).
* Budget T = 100 items (or the pool size if smaller).
* Policies: naive; random:pi; random:mg; entropy:pi; entropy:mg; mfi:pi;
  mfi:mg; pwfi:mg; eig:pi; eig:mg (EOST-M); evr:pi; evr:mg.
* Master seed 20261029; every random stream is derived by SHA-256 of
  (seed, bank, purpose, target, replicate, N, policy).

## 4. Endpoints

* `ilogloss`: holdout log loss averaged over steps t = 1..T (primary for H1, H2).
* `iae`: |Â_t − A| averaged over t (primary for H3), A = full-bank accuracy.
* `covgap`: |coverage − 0.90| of the 90% predictive interval for A, coverage
  averaged over steps and replicates within target (primary for H4).

## 5. Hypotheses (all at the primary panel size N = 8; other N reported)

* H1 information vs entropy: eig:mg < entropy:mg on `ilogloss`.
* H2 calibration-marginal EIG vs plug-in Fisher CAT: eig:mg < mfi:pi on `ilogloss`.
* H3 adaptive EOST-M vs naive subsampling: eig:mg < naive on `iae`.
* H4 interval calibration: eig:mg < mfi:pi on `covgap`.

## 6. Statistics

Unit = target (mean over replicates). Two-sided Wilcoxon signed-rank test on
paired per-target values; Holm correction over the whole confirmatory family
(8 banks × available N × 4 hypotheses); α = 0.05. Effect sizes: mean paired
difference with 10,000-sample bootstrap CI over targets, relative improvement,
wins/losses, matched-pairs rank-biserial correlation.

**Decision rule.** A hypothesis is *supported* if, at N = 8, the Holm-adjusted
test favours the new policy in ≥ 5 of the 8 confirmatory banks and favours the
baseline in none. Results are reported in full whatever the outcome.

## 7. Secondary and exploratory analyses (not confirmatory)

Secondary contrasts listed in `configs/hypotheses.json` (own Holm family);
items-to-tolerance; Brier; final-step errors; synthetic ground-truth metrics
(true-θ error and coverage); calibration diagnostics (effective panel size,
fraction of perfectly separated items, discrimination inflation); sensitivity
analyses with the one-parameter model, fixed (non-EB) priors, and
near-duplicate exclusion (Cohen's kappa ≥ 0.8), at N ∈ {8, 64}, 30 targets,
3 replicates.

## 8. Deviations

Any deviation from this plan will be listed in the paper under "Deviations
from the pre-registered plan".
