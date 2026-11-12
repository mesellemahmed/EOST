# Adversarial audit of the EOST study (iterations 2 and 3)

Purpose: try to break the paper's claims before a reviewer does. Each threat is
stated as an attack, followed by the test that was run and its outcome. Numbers
come from the committed artifacts; qualitative statements of the manuscript are
machine-checked by `scripts/check_claims.py` (`artifacts/CLAIMS.json`).

## A. Integrity of the confirmatory evidence

| # | Attack | Test | Outcome |
|---|---|---|---|
| A1 | The plan was written after seeing the results. | Freeze commit `901afa2` (2026-10-29 12:27:10 −04:00) hashes the plan, hypotheses, analysis code and data manifests; the first confirmatory bank finished at 12:46 local time (`artifacts/confirm/cruxeval_o/config.json`). `PREREG_GATE` re-checked after all runs. | Holds. |
| A2 | Infrastructure restarts or the checkpointing wrapper changed results. | Wrapper vs frozen runner on a test bank; guard vs frozen code on stored targets; **independent recomputation one week later (experiment PH1) of all EIG:mg and MFI:pi rows at N ∈ {4,8,16}**. | Maximum absolute difference 0 in every check (PH1: see `artifacts/posthoc_baselines/_analysis/decisions.json`). |
| A3 | Multiplicity: with 144 tests something is always significant. | Holm over the whole confirmatory family; decision rule needs ≥ 5/8 banks and none against. | H2–H4 pass with 7/8; H1 fails with 4/8 and is reported as not supported. |
| A4 | Pseudo-replication inflates significance. | Unit = target system; replicates averaged within target. | Holds. |
| A5 | Leakage: the target's own verdicts inform calibration. | Panels exclude the target and its family; the holdout is never administered; near-duplicate calibrators (κ ≥ 0.8) removed in a sensitivity analysis. | H2 unchanged without near-duplicates (6.1% vs 5.0% on the same banks). |
| A6 | The study-2 plan was written after seeing its results. | Freeze commit `bd46f79` (2026-11-03 23:02:39 −05:00 = 2026-11-04T04:02:39Z; lock `frozen_at` 04:02:37Z) hashes plan, hypotheses, cluster file, code and manifests; the first study-2 target file was written at 23:04:22 −05:00; `STUDY2_GATE=PASS` before every component and after the run. The post hoc corrections were committed (`1735cce`, `a83b730`) before the frozen analysis was run. | Holds. |

## B. Competing explanations for the main effects

| # | Attack | Test | Outcome |
|---|---|---|---|
| B1 | H2 is driven by the empirical-Bayes prior, not by marginalization. | Factorial (same EB prior for plug-in and marginal models); fixed-prior sensitivity. | Under the same prior, marginalization alone gives +4.4% (entropy) / +5.2% (EIG). Without the EB prior the gain halves and reverses on 3/8 banks: EB and marginalization work together; the paper says so. |
| B2 | H3 is just model-assisted estimation, not adaptivity. | Random items scored by the same marginal model. | EOST-M's score IAE is 28% lower (7/8 significant). |
| B3 | EIG is not special; any information criterion works. | MFI, PWFI, EVR under the marginal model. | True: within 0.1% (log loss) and 3% (IAE). The paper presents EIG as the principled member of an equivalence class. |
| B4 | H1 is small because of a bug, not because of the theory. | Corollary 3 predicts no EIG–entropy difference under a common discrimination; regularized 1PL (post hoc). | Difference vanishes (+0.4% mg, −0.4% pi vs +1.9%, +1.6% under the 2PL). |
| B5 | EOST-M only beats an *unregularized* plug-in 2PL (strawman baseline). | Experiment PH1 (post hoc, rules fixed in `docs/RESEARCH_LOG.md` before running): posterior-mean plug-in 2PL; regularized Rasch-type plug-in CAT; marginal regularized 1PL. | See section D. |
| B6 | Averaging Fisher information over the item posterior (Laplace-PSN-IRT style) is as good as integrating it. | PH1 policy EFI:mg. | See section D. |
| B7 | The Rasch-type competitor's better coverage means it is the better method for uncertainty. | PH3: coverage decomposed into interval width and score error on the stored PH1 runs (rule fixed before computing). | Its intervals are wider on 8/8 banks; its error is not lower; coverage tracks width/error (ρ = 0.98). See section E. |

## C. Numerical and estimation failures found during the audit

| # | Finding | Action |
|---|---|---|
| C1 | EVR overflow when an expected response probability exceeds 1 by rounding (1PL, HellaSwag). | Guard in the wrapper (exact no-op on valid input, unit-tested); deviation (iii). Confirmatory runs unaffected (they completed without it). |
| C2 | The planned (frozen) one-parameter joint MAP is degenerate (common discrimination 1e4–1e7; abilities collapse). | Reported, not interpreted; post hoc regularized variant; limitation (ix) recommends marginal maximum likelihood. |
| C3 | Narrative claims drifted from the numbers during drafting (e.g. "lowest error throughout", "the four smallest banks"). | Every qualitative claim encoded in `check_claims.py`; false claims were rewritten (the HellaSwag result broke the "smallest banks" statement, now reported as a tendency, ρ = 0.73). |
| C4 | `./run sensitivity` called the frozen runner without the EVR guard: a from-scratch rerun would stop on the 1PL HellaSwag bank, and `./run all` did not regenerate the post hoc 1PL variant. | `./run sensitivity` now uses the checkpointing runner with `--evr-guard` (how the committed artifacts were produced; exact no-op on valid input); new `./run regularized` and `./run coverage`, both in `./run all`. No artifact changed. |

## D. Strong baselines (experiment PH1) — outcome

Source: `artifacts/posthoc_baselines/_analysis/` (contrasts, decisions, reproduction check); paper Section 6.9, Table 8.

* **Reproduction:** all 12,570 recomputed EIG:mg and MFI:pi rows equal the confirmatory rows (max |diff| = 0), six days later and after several machine restarts.
* **Posterior-mean plug-in 2PL** (better point estimates, no integration): EOST-M's held-out log loss is 15.5% lower at N = 8 (7/8 Holm-significant). Point estimation is not the fix.
* **Regularized Rasch-type plug-in CAT** (common discrimination, log a ~ N(0, 0.05²)): EOST-M's log loss is lower on 8/8 banks but by a median of only 2.1% (4/8 significant); score error lower on 5/8 (median 2.9%; 15.3% at N = 16); the Rasch-type test has better coverage (0.81 vs 0.75). **Decision rule 1 triggered** → the paper states that the advantage over plug-in CAT is largely an advantage over unregularized item-specific discriminations (abstract, introduction, Section 6.9, discussion, recommendations, conclusion, limitation x).
* **Expected Fisher information** (Laplace-PSN-IRT-style averaging): identical to EIG:mg in practice (median 0.0%, no bank significant). **Rule 2 not supported** → the paper states that Corollary 5's optimism did not matter in these banks (theory section, related work, Section 6.9).

## E. Verdict

The confirmatory claims (H2–H4 supported, H1 not supported) survive every attack
on integrity (section A). Their *interpretation* did not survive unchanged: the
benefit over current practice is real but is mostly the benefit of not trusting
item-specific discriminations from small panels, which a strongly regularized
plug-in model also obtains; the paper has been reframed accordingly, with
integration ahead on prediction and score error and the constrained model
ahead on interval calibration. The exploratory regularization path (PH2,
`docs/RESEARCH_LOG.md`, planned before running) asked whether the two remedies
can be combined by shrinking EOST-M's discriminations towards a common value:
they cannot in that form (coverage 0.74–0.76 for every shrinkage level vs 0.75,
log loss 0.2–2.4% worse; no level met the pre-stated criterion). A further
decomposition (PH3, analysis of the stored PH1 runs, rule fixed before
computing) answers where the Rasch-type calibration comes from: its intervals
are wider on 8/8 banks (median +29%, all Holm-significant) while its score
error is not lower on 4 of the 7 banks where its coverage is higher, so it is
better calibrated because it is less confident, not because it is more
accurate. Across all bank × model cells, coverage follows the ratio of interval
width to score error almost exactly (Spearman 0.98), every model is below the
calibrated-Gaussian ratio, and the same marginal model is near nominal (0.88)
with uniformly sampled items: the shortfall arises when adaptive selection
meets a misspecified model. Recalibrating the interval itself is the next
iteration (planned in `docs/RESEARCH_LOG.md`, not run). Audit status:
**favourable for publication with the reframed claims**; all 61 qualitative
claims of the manuscript hold on the artifacts.

## F. Iteration 3: external review, study 2 and its audit

After the 26-point external review (2026-11-03), this section records the attacks that
required new evidence, the tests run, and their outcomes.

| # | Attack | Test | Outcome |
|---|---|---|---|
| F1 | Targets of one model family are not independent; per-bank significance is inflated. | Study 1 re-analysed with heuristic families and strict clusters (heuristic family OR IRT-residual correlation ≥ 0.5) as units (post hoc); study 2 with one target per strict cluster (frozen). | Study 1: H2 drops to 4/8 significant banks with strict clusters (not supported), effect unchanged (5.4%, 95% cluster-bootstrap CI 4.3–6.4%); H3, H4 survive. Study 2: R2 5/0 (+5.9%), R3 6/0, R4 5/0 — all supported. |
| F2 | The post hoc Rasch-type competitor is as good as EOST-M. | Frozen C1 (plug-in) and C2 (marginal Rasch-type). | C1 supported (5/0, +2.4%, positive 7/8); C2 not supported (4/0, +1.8%, positive 8/8). Integration beats constraint on prediction by a small margin; scores are close; Rasch-type raw intervals better calibrated (wider). |
| F3 | A fully Bayesian CAT (integrating calibrator abilities and hyperprior) is the proper method. | Frozen C3 (equivalence ±1% on ≥ 6 banks). | Frozen version: equivalent on 4/8 (not supported). Code review during the run (before results): burn-in too short on some banks → post hoc correction (60 sweeps) and exploratory joint model. Corrected: 3/8 (BMI and EIG), median +0.4–0.5% (FB worse), no bank significant either way; joint model 5/8, median +0.05% (EIG) / +0.33% (BMI); coverage 0.76 vs 0.74. The frozen decision (not supported) stands; integrating θ_C made no significant difference, equivalence not shown. |
| F4 | Naive subsampling flatters the headline. | Frozen C4 vs Random:mg (same calibration). | 4/0 (+21.1%, positive 8/8): not supported by the ≥ 5 rule (study 1, secondary: median +25.5%, 7/8). Reported in the abstract. |
| F5 | Modern reduction methods (ATLAS, tinyBenchmarks, MINCE) are only cited. | Frozen C7, C8, C9 (implemented from the papers). | C7 4/0 (+26.0%), C8 4/0 (+29.6%), C9 5/0 (+44.8%); positive on 8/8 each. Code review found that tinyBenchmarks can use fewer anchors than its budget with 8 calibrators (≤ 2^8 patterns); post hoc audit: at budget 50 at most 7% of cells affected, C8 unchanged (4/0) on full-budget cells and items-matched under the frozen Holm family. A first version of the audit applied Holm within the 8 corrected tests (5/0) — less conservative than the frozen rule, corrected and logged. |
| F6 | Intervals under-cover (0.75 for a nominal 0.90). | Frozen C5a/C5b/C6 (leave-one-calibrator-out recalibration, rule chosen on development banks), FB CAT, mixed policy. | Recalibration 0.74 → 0.86; inside [0.85, 0.95] on 5/8 (C5a not supported); interval score not improved (C5b not supported; worse on HellaSwag, WinoGrande); much better than Naive's Wilson interval (C6 supported). Recalibrated Rasch-type: 0.91, all 8 banks inside, but twice as wide. Mixed policy 0.77. Not solved; stated as the open problem. |
| F7 | The empirical-Bayes hyperprior is opportunistic tuning. | Item/half-split/calibrator bootstrap, prior predictive checks (post hoc); cross-fitted and Student-t hyperpriors (study 2). | Hyperparameters stable over items (bootstrap sd ≤ 0.08 at N = 8), vary with the panel; PPC reproduces extreme-item share on average (misfit on HellaSwag/MMLU panels) and shows lower item–rest correlation than the model (dependence misfit); cross-fitted hyperprior: log loss unchanged (median 0.0%); Student-t: slightly worse on every bank (median +0.4%); no bank significant. |
| F8 | Replay ≠ prospective deployment. | Live protocol (not runnable here: no model weights); frozen temporal component (later SWE-bench submissions). | Temporal: 10/11 contrasts significant for EOST-M, none against (C5b n.s.). Live protocol released and mock-validated, not run. |
| F9 | Direction of significance read from mean differences in a rank test. | Sign check of every Holm-significant contrast (both studies). | 0 of 815 discordant. |
| F10 | "Pre-registration" not externally verifiable. | No registry/time stamp reachable. | Renamed "frozen analysis plan"; locks and git history; study-2 plan sent to the author before execution. |
| F11 | Environment provenance inconsistent. | `configs/ENVIRONMENT.json`; container image IDs of the study-2 runs. | Every result of the paper (studies 1, 2, corrections) ran in image `sha256:5b21ccfe…`; the pilot's different image is documented. |

Verdict of iteration 3: the core claim (small panels break plug-in adaptive testing;
integrating calibration uncertainty fixes much of it) survives independent targets, a
stronger competitor set and a temporal split. Several comparisons fall one benchmark short
of the frozen threshold and are reported as not supported; the interval problem is not
solved. All 77 qualitative claims of the manuscript hold on the artifacts (`artifacts/CLAIMS.json`, `CLAIMS_GATE=PASS`).

## G. Iteration 4: second external review

| # | Attack | Test | Outcome |
|---|---|---|---|
| G1 | The strict clusters use the target's own responses; excluding residually similar calibrators may make panels harder and favour EOST-M. | RF1 (planned before running): frozen targets, holdouts, seeds; panels drawn under metadata-only, leave-bank-out residual and threshold 0.4/0.6/0.7 exclusion rules; frozen 0.5 recomputed. | Frozen rows reproduced exactly (4,005). R2 8/8 positive, 4–6 Holm-significant (frozen 5); C1 7/8 positive, 5 significant (frozen 5); medians unchanged; rule met under all five rules. Response-derived links only matter on the Open LLM Leaderboard. |
| G2 | The Bayesian comparator is under-converged (8 draws, slow scale mixing), so "no difference from EOST-M" may be an MCMC artefact. | RF2 (planned before running): 4 chains, burn-in 250, 400 retained draws, R̂/ESS/MCSE, trace plots, long-chain arms vs EOST-M and short-chain arms on 48 panels. Exploratory RF2b (scale move), RF2c (exact model of Lemma 1; stopped after 14 panels, location move blocked at the grid edge) and RF2d (exact model, collapsed moves), each planned before running and validated against exact stationary laws. | Attack partly confirmed. Frozen sampler: location converged (92% of panels R̂ < 1.05) but σ_b, μ_a, σ_a did not (≤ 8%; R̂ up to 3.04) — a scale ridge without a move; pre-specified rule not confirmed (cell-level changes 0.6–1.0% vs burn-in 60). RF2d came close (median R̂ ≤ 1.06, max 1.25) but not to the 90% criterion: no fully converged comparator. Across all 18 arm/sampler combinations the median log-loss difference from EOST-M stays within ±0.45%, benchmarks split; RF2d coverage 0.76–0.79 vs EOST-M 0.78. Paper: artefact unlikely, statements limited to the implementations studied; RF2d's score-error lead (5–11% over cells, 4–5/8 banks) reported as open. |
| G3 | The frozen rule (≥ 5 of 8 after Holm over 88 tests) hides effects that are consistent across benchmarks. | RF3 (secondary): random-effects meta-analysis + sign tests. | 10/11 pooled effects exclude zero, including C2, C4, C7, C8 (missed by one bank); I² 75–93% for these; prediction intervals for a new benchmark exclude zero only for R2, R3, R4, C9. Third review: five benchmarks share their 395 systems, which RF3 ignores. RF3b (planned before running) counts them as one source (k = 4; bootstrap over shared systems): 6/11 intervals exclude zero (R2, R3, R4, C2, C6, C9); C1, C4, C7, C8 stay positive in 3–4 of 4 sources but their intervals include zero. Sampling dependence small (median r = 0.02). RF3 kept as secondary ("point the same way", not "generalize"). Frozen decisions unchanged. |
| G4 | "Fully Bayesian" mislabels the per-item averaged arm. | Renamed (posterior-draw marginal vs joint fully Bayesian CAT), consistent with Lemma 1. | Done throughout. |
| G5 | JMLR desk-reject risks (abstract, running title, keywords, length). | Abstract 200 words (by word counter / PDF text); five keywords; the article (v5.0) is one 25-page PDF with every frozen decision, and its extended version (66 pages) contains every secondary table and figure. | Holds. |
