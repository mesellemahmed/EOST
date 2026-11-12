# Anticipated reviewer questions (with where the answer lives)

Section, proposition and table numbers refer to the extended version, `paper/main.pdf`, which contains every secondary table and figure; the article is `paper/short.pdf`. Every number quoted in
the manuscript is a macro generated from the artifacts, and every qualitative statement is
re-checked by `scripts/check_claims.py` (`./run claims` → `CLAIMS_GATE=PASS`).

**What exactly is new, given expected response functions (1994) and Bayesian CAT (2020)?**
Not the ingredients. The paper does not claim to be first to integrate calibration
uncertainty into adaptive testing (Section 2, "What is new"). It contributes (i) a
systematic study of the regime LLM evaluation actually faces—calibration panels of 4–64
systems, eight real benchmarks from five domains, independent targets; (ii) theory for
that regime: the calibration-marginal model is the exact Bayesian model given the
calibrators' abilities and the hyperprior (Lemma 1), and a difficulty with a Gaussian posterior of standard
deviation s caps an item's Fisher information at 2/(πs²) exactly, for the logistic and any
other link (Proposition 5; a general bound covers other posteriors); (iii) head-to-head comparisons under frozen analysis plans with the
competitors that matter—a regularized Rasch-type test, a fully Bayesian CAT, ATLAS-style
3PL CAT, tinyBenchmarks, MINCE, uniform subsampling with and without the same calibration;
(iv) an account of uncertainty (coverage, recalibration) and cost (break-even).

**Is Proposition 5 only a probit approximation for the logistic 2PL?** No. The bound is
proved exactly for any continuous link through the latent representation
y = 1{L/a + b ≤ θ} and the data-processing inequality for Fisher information
(proof right after Proposition 5 in Section 4); it is sharp as a → ∞ and unit-tested numerically for the logistic link. It
assumes a Gaussian difficulty posterior; the same proof gives sup f²/{F(1−F)} for any other
difficulty posterior with a density (π²/(12s²) for a logistic one), and the paper says so.

**Do entropy and EIG agree "only when" discriminations are equal?** The paper no longer
says so. Corollary 4 gives the sufficient condition (a common discrimination) and the
constructive source of disagreement (ties in success probability with unequal
discriminations); it does not claim necessity.

**Is EIG "optimal"?** Only myopically: every optimality statement in Section 4 concerns
the best next item given the current posterior (EIG for information about competence,
Proposition 7 for the variance of the bank score), not the best sequential policy.

**Are targets of the same model family independent?** No, and the paper handles this in
two ways. Study 1 is re-analysed with heuristic families and with strict clusters
(connected components of shared family or IRT-residual correlation ≥ 0.5) as units
(Table 3): with strict clusters H2 is significant on 4/8 benchmarks only and is no longer
supported by the frozen rule, although its median effect is unchanged (5.4%, 95%
cluster-bootstrap interval 4.3–6.4%); H3 and H4 survive. Study 2 uses one target per
strict cluster, and its panels exclude the target's whole cluster.

**Why not Cohen's κ for near-duplicates?** On HellaSwag, κ ≥ 0.8 chains 353 of 395
systems into one component, because all strong models agree on most items. Residual
correlation removes what ability explains (Section 5.1).

**Is the comparison with Naive subsampling flattering?** Partly—Naive uses no
calibration. The abstract and Section 7.3 therefore report EOST-M against Random:mg
(random items scored by the same calibrated model), which isolates adaptivity; it is
hypothesis C4 of study 2, and the cost table gives the items needed to match Random:mg.

**What does adaptive testing cost once the calibration panel is counted?** With eight
fresh calibrators who answer the whole bank and 100-item tests, adaptive testing beats
full evaluation after 8.4–10 new systems on every benchmark; against uniform subsampling at
matched precision the break-even is a median of 122 systems; when the calibrators'
results already exist (public leaderboards) the panel is free (Section 8, Table 5).

**Is the empirical-Bayes prior a tuning knob?** It is load-bearing (the study-1 gain
halves without it), so Section 7.6 checks it: item-bootstrap and half-split stability,
calibrator bootstrap, prior predictive checks; study 2 runs EOST-M with a cross-fitted
hyperprior and with Student-t hyperprior families, and the fully Bayesian CAT integrates
the hyperparameters instead of plugging them in.

**"Deterministic oracle" is too clean for LiveCodeBench and CRUXEval.** Agreed: these
banks carry recorded aggregate verdicts (pass rates over samples, binarized at one half).
Section 3.1 distinguishes the two verdict types, Table 1 has a verdict column, and study-2
contrasts are also reported separately for the two groups.

**Replay is not a prospective deployment.** Correct, and it could not be run here: the
execution environment cannot download model weights, and the pinned leaderboards have no
newer submissions. Instead, study 2 has a frozen temporal component (latest quarter of
SWE-bench Verified submissions as targets, only earlier submissions as calibrators), and a
complete live protocol (`scripts/live_eost.py`, `./run live`) is released and validated
end to end with a mock model (Section 10, "A protocol for live evaluation").

**Was "pre-registration" externally verifiable?** No registry or time-stamping service was
reachable from the execution environment, so the paper says "frozen analysis plan" for
both studies. Plans, hypotheses, code and data manifests were hashed and committed before
execution (`configs/PREREGISTRATION.lock.json`, `configs/STUDY2.lock.json`; `./run verify`),
and the study-2 plan was also sent to the first author before its first run. Registering
the study-2 lock hash on OSF and making a GitHub/Zenodo release are recommended before
publication.

**Which software environment produced the results?** One pinned image for every result
of this paper (Ubuntu 24.04 by digest, Python 3.12.3, no PyTorch; image ID in Section 5.6
and `configs/ENVIRONMENT.json`). The historical pilot used an earlier image
(python:3.12-slim-bookworm, Python 3.12.15, torch 2.7.0+cpu) that shares the tag; no result
of the paper was computed in it.

**Pilot program numbers cannot be checked.** They are labelled historical, quoted as
recorded (including failed gates: S1-R2, S2, original S4; S4B incomplete), and no claim of
the paper depends on them (Section 5.4).

**Is EOST-M just beating a weak, unregularized plug-in baseline?** Partly, and the paper
says so: after study 1 a regularized Rasch-type plug-in test came within about 2% post hoc.
Study 2 therefore tested it under a frozen plan with one target per cluster: EOST-M's
held-out log loss was lower on 7/8 benchmarks (median 2.4%, Holm-significant on 5, none
against; C1 supported). Against the Rasch-type model integrated over its item posterior the
margin is 1.8% (4/0; C2 not supported). Score errors of EOST-M and the Rasch-type test are
close; the Rasch-type raw intervals are better calibrated because they are wider.

**Why not a fully Bayesian CAT?** Study 2 ran one (Gibbs sampler on the panel's hierarchical
2PL, Bayesian maximum information as recommended by Fink et al. 2025, also with EIG). It was
never significantly better or worse than EOST-M on any benchmark, but the frozen
equivalence criterion (±1% on ≥ 6 banks) was met on 4/8 only (C3 not supported); a post hoc
correction of its burn-in, decided before the results, gave 3/8 (median +0.4%, fully
Bayesian worse), and an exploratory joint version was closest (median +0.05%, 5/8). With
eight calibrators, integrating the calibrators' abilities made no significant difference once
item uncertainty is integrated (equivalence was not shown either); EOST-M is exact given the
calibrators (Lemma 1) and simpler.

**Several study-2 hypotheses are "not supported" although they are positive everywhere.**
Yes: C2, C4, C7 and C8 have four of eight benchmarks Holm-significant and none against,
one short of the frozen threshold of five, and are reported as not supported. The paper
does not reinterpret them; it reports the direction (positive on 8/8) next to the decision.
CRUXEval-O (12 clusters) and LiveCodeBench (15) have little power.

**Did the interval recalibration work?** Partly. It raises EOST-M's coverage from 0.74 to
0.86 (0.81–0.91 across banks), inside the frozen band [0.85, 0.95] on 5/8 (C5a needs 6: not
supported), and it does not improve the interval score (C5b not supported; worse on
HellaSwag and WinoGrande). The recalibrated interval beats Naive's Wilson interval on
interval score (C6 supported). The recalibrated Rasch-type test is nominal on every bank but
twice as wide. The paper states calibrated uncertainty as the open problem.

**Is tinyBenchmarks treated fairly with eight calibrators?** A code review during the run
found that K-Means can return fewer anchors than the budget (eight calibrators produce at
most 2^8 response patterns). A post hoc audit, decided before the results, recomputed the
number of anchors with the same seeds: at the 50-item budget of C8 at most 7% of a bank's
cells are affected, and C8 restricted to full-budget cells or matched on the number of
items gives the same decision under the frozen Holm family (4/0, median +29.6%).

**Does the result hold for systems released after the calibrators?** On the 31 latest
SWE-bench Verified submissions, calibrated only on earlier ones, 10 of the 11 frozen
contrasts favour EOST-M significantly and none favours a competitor.

**Do the strict clusters leak the target's responses into the design?** They use them: the
residual correlations are computed from every system's full response vector. A follow-up
(RF1, planned before running) kept the frozen targets, holdouts and seeds and drew panels
under exclusion rules that never see the target's responses (heuristic families only;
leave-bank-out residual clusters) and under thresholds 0.4–0.7. R2 and C1 were unchanged in
every case (R2 positive on 8/8 with 4–6 significant benchmarks, C1 positive on 7/8 with 5),
and recomputing the frozen clusters reproduced every frozen row exactly.

**Was the Bayesian comparator run long enough?** RF2 re-ran it with 4 chains, a burn-in of
250 sweeps and 400 retained draws on 48 panels, with R̂, bulk ESS, MCSE and trace plots, and
compared the long-chain arms with EOST-M and with the short-chain arms on the same cells
(Section 8.2). Not long enough to converge: the location converged, but σ_b, μ_a and σ_a did not (R̂ up to 3.0), because with eight calibrators the posterior has a scale ridge—abilities and difficulties multiplied by a constant, discriminations divided by it—along which the frozen Gibbs sampler has no move. Under the rule fixed in advance the study-2 statement was therefore not confirmed. Exploratory samplers then added a scale move (RF2b) and sampled the model of Lemma 1 exactly with collapsed moves (RF2d; a first version, RF2c, was stopped when its location move proved blocked at the edge of the grid). RF2d came much closer (median R̂ ≤ 1.06, largest 1.25) but did not meet our criterion of R̂ < 1.05 on 90% of panels for every quantity. The comparison itself never moved: across all 18 combinations of arm and sampler the median difference from EOST-M in held-out log loss stayed within ±0.45%, with benchmarks split, and RF2d's intervals covered as often as EOST-M's. An artefact of the frozen chain is therefore unlikely, but the paper's statements concern the implementations studied; a fully converged comparator (for instance Hamiltonian Monte Carlo on a continuous parametrization) remains to be run.

**The frozen rule is so strict that consistent effects are "not supported".** Yes, by design;
the decisions are not changed. As a secondary analysis, a random-effects meta-analysis over
benchmarks (RF3) shows that the pooled effect excludes zero for 10 of 11 superiority
hypotheses, including the four that missed the threshold by one benchmark, with large
heterogeneity: the 95% prediction interval for a new benchmark excludes zero only for R2, R3,
R4 and C9. RF3 treats the benchmarks as independent, but five share the same 395 Open LLM
Leaderboard systems; counted as one source (RF3b), 6 of 11 intervals exclude zero, and among
the four that missed by one benchmark only C2 does. So the effects point the same way across
benchmarks; the meta-analysis does not show that they generalize.

**What exactly is the "fully Bayesian" arm?** The frozen arm averages each item's response
function over the posterior draws separately (posterior-draw marginal Bayesian CAT, the van
der Linden–Ren form); the exploratory joint fully Bayesian CAT mixes over draws of the whole
calibration, which keeps the shared uncertainty about the calibration's location (Lemma 1).
