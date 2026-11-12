# Study 2 — frozen analysis plan (written before any confirmatory study-2 run)

Status: this plan, `configs/study2_hypotheses.json`, the study-2 code and the cluster
file are hashed in `configs/STUDY2.lock.json` and committed **before** the first
study-2 run on a confirmatory bank. The plan and its SHA-256 are also sent to the
first author outside the execution environment before execution. No public registry
or external time-stamping service was reachable from the execution environment, so
the paper calls this a *frozen analysis plan*, not a preregistration.

## Why a second study

An external review of the first study raised five issues that only new runs can
settle: (i) the strongest competitor (a regularized Rasch-type plug-in test) was found
post hoc; (ii) a fully Bayesian adaptive test (van der Linden & Ren, 2020; Fink,
König & Frey, 2025) was not compared; (iii) targets of the same model family may
not be independent; (iv) 90% score intervals covered about 75% of the time; (v) recent
benchmark-reduction methods (ATLAS, tinyBenchmarks, MINCE) were cited but not run.

## Design

* **Banks**: the eight confirmatory banks of study 1 (GSM8K, ARC-C, HellaSwag,
  WinoGrande, MMLU, SWE-bench Verified, LiveCodeBench, CRUXEval-O), unchanged data
  (`data/processed/MANIFEST.json`).
* **Strict family clusters** (`configs/study2_clusters.json`, computed before this
  plan from responses only): connected components of the graph joining two systems if
  they share a heuristic family or if the correlation of their standardized 2PL
  residuals is at least 0.5 (joint over the six Open LLM Leaderboard banks and over
  CRUXEval-O/-I, which share systems; per bank otherwise).
* **Targets**: one representative per cluster (systems that were not targets in study
  1 are preferred), at most 40 per bank, stratified by accuracy deciles; targets are
  therefore independent at the cluster level. Master seed 20261104 (study 1: 20261029).
* **Panels**: N ∈ {4, 8, 16} calibrators drawn outside the target's cluster; primary
  N = 8. Holdout: 20% of the items, shared by all policies. Three replicates (panel and
  holdout redrawn). Test length 100 items.
* **Policies** (all from the same panel and holdout):
  * `eig:mg` — EOST-M (EIG; calibration-marginal 2PL; empirical-Bayes hyperprior);
  * `mfi:1pr` — regularized Rasch-type plug-in CAT (common discrimination,
    log a ~ N(0, 0.05²), EB prior on b; maximum Fisher information);
  * `eig:1prmg` — the same one-parameter model, calibration-marginal, EIG;
  * `fb:bmi`, `fb:eig` — fully Bayesian CAT: Gibbs sampler for the hierarchical 2PL of
    the panel (abilities, item parameters on the grid, hyperparameters; exact
    translation move), 8 retained draws (burn-in 10, thinning 4); the target's ability
    posterior uses the item marginals averaged over draws (the target of van der Linden
    & Ren's algorithm); selection by Bayesian maximum information (posterior-weighted
    expected Fisher information; Fink et al.'s best strategy) or by EIG; N ∈ {4, 8};
  * `random:mg` — uniformly random items scored with the EOST-M model;
  * `naive` — uniformly random items, sample mean, Wilson interval (finite-population
    corrected);
  * `mfi:pi` — plug-in 2PL Fisher CAT (study 1 baseline);
  * `atlas` — ATLAS-style CAT (Li et al., 2026): 3PL with the lower asymptote fixed at
    the chance rate of the item format (ARC, HellaSwag, MMLU 0.25; WinoGrande 0.5;
    others 0), item parameters by grid MAP given the panel's abilities and the EB prior,
    first item with difficulty closest to 0, then a uniform draw among the five most
    informative items at the EAP, EAP scoring; fixed length;
  * `tiny:gpirt` (and `tiny:pirt`, `tiny:anchor`) — tinyBenchmarks: IRT anchor points
    by K-Means (best of 5) on the panel's item embeddings, gp-IRT estimator with
    lambda = b²/(v/(4n)+b²) estimated by leave-one-calibrator-out; budgets 25/50/100;
  * `mince` — MINCE: Monte Carlo sizing on the panel (R = 2000 subsets, step 25,
    tau = 1 point), plain mean on one frozen random subset, P95 drift bound; budgets
    25/50/100 and its own n*;
  * `eig:xf` (cross-fitted EB hyperprior) and `eig:tp` (Student-t hyperprior family),
    N = 8 only — empirical-Bayes robustness;
  * `mix:mg` — exploratory: every fourth item uniform at random, otherwise EIG.
* **Interval recalibration** (`eig:mg+R`, `mfi:1pr+R`): leave-one-calibrator-out
  replays. For every calibrator c, the item posteriors are recomputed without c's
  cluster (abilities and hyperprior held at the panel fit), the same policy is replayed
  on c, and the standardized errors (or absolute errors) of its score estimate are
  pooled over calibrators and over a ±50% window of test steps. Three rules were
  implemented: Rq (90% quantile of |standardized error| / z_0.95, multiplying the model
  half-width), Rr (root mean square of the standardized error, multiplying it) and Ra
  (90% quantile of the absolute error, replacing it). The primary rule was chosen on the
  development banks only (SWE-bench Lite, CRUXEval-I, TruthfulQA, synthetic 2PL) by a
  selection rule written in the research log before that development run: the rule whose
  EOST-M coverage at N in {4, 8}, averaged over development banks, is closest to 0.90.
  Result: **Ra** (mean coverage 0.83; Rq 0.80, Rr 0.80). The development runs also showed
  that no rule helps at N = 4 (too few pseudo-targets), so the recalibration hypotheses
  are evaluated at N = 8 only, as planned.
* **Secondary components**: (a) *temporal validation* on SWE-bench Verified — targets
  are the latest quarter of submissions by date (one per cluster), calibrators are
  earlier submissions only, N = 8; (b) the three synthetic banks (20 targets each).

## Endpoints

Integrated held-out log loss (`ilogloss`), integrated absolute score error (`iae`),
|coverage − 0.90| of the 90% score interval (`covgap`), integrated interval score
(Winkler, α = 0.1; `iis`), absolute score error after 25/50/100 items (`err@T`).
All are averaged over the 100 steps (or taken at step T) and over replicates.

## Hypotheses and decision rules (primary panel size N = 8)

Superiority (paired two-sided Wilcoxon per bank on the targets; Holm over all 11 × 8
tests; supported if ≥ 5 of 8 banks are Holm-significant for EOST-M and none against):
R2, R3, R4 (replications of H2–H4 with independent targets), C1 (vs Rasch-type plug-in,
log loss), C2 (vs one-parameter marginal, log loss), C4 (vs Random:mg, score error),
C5b (recalibrated vs raw EOST-M intervals, interval score), C6 (recalibrated EOST-M vs
naive Wilson intervals, interval score), C7 (vs ATLAS-style CAT, score error), C8 (vs
tinyBenchmarks gp-IRT, error at 50 items), C9 (vs MINCE, error at 50 items).

Equivalence C3: the fully Bayesian CAT (Bayesian maximum information) and EOST-M are
practically equivalent on held-out log loss if the 90% bootstrap interval (over
targets) of the relative difference of means lies within ±1% on ≥ 6 of 8 banks.

Coverage C5a: the recalibrated EOST-M interval has mean coverage in [0.85, 0.95] on
≥ 6 of 8 banks.

Everything else (other panel sizes, the temporal and synthetic components, EB
variants, fully Bayesian selection criteria, the mixed policy, recalibrated Rasch-type
intervals, MINCE's own n*) is secondary and reported with its own Holm family.

## What will be reported whatever the outcome

Every decision above, all secondary contrasts, the coverage and width of every
interval, the computational cost, and any deviation from this plan (with its reason).
