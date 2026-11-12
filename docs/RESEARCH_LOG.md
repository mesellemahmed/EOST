# Research log and resumable state

This file records hypotheses, experiments, outcomes, failures and the next
actions, so that the work can be resumed exactly. Newest entries last.
Every number cited here is in `artifacts/` (see `RESULTS.md`, `artifacts/CLAIMS.json`).

## State at 2026-11-04 (UTC)

Resume after a machine restart: `rm -f /var/run/docker.pid /var/run/docker.sock; dockerd --iptables=false --ip6tables=false --bridge=none &`, then re-run the interrupted step; every runner is resumable from per-target checkpoints (`scripts/launch_resumable.sh <stage>`, `scripts/posthoc_baselines.py`, `scripts/posthoc_shrinkage.py`).

### Pre-registered study (frozen 2026-10-29T16:27:10Z, `configs/PREREGISTRATION.lock.json`)

| Item | Status | Outcome |
|---|---|---|
| Confirmatory runs, 8 real banks + 3 synthetic | done (`artifacts/confirm`) | H2, H3, H4 supported (7/8 Holm-significant, 0 for baseline); H1 not supported (4/8) |
| Sensitivity: fixed item prior | done (`artifacts/sens_fixedprior`) | H2 median gain 2.8% (main 5.2%), reverses on 3/8 banks -> EB prior essential |
| Sensitivity: near-duplicates removed | done (`artifacts/sens_dedup`) | H2 gain unchanged (6.1% vs 5.0% on same 5 banks) |
| Sensitivity: one-parameter model | done (`artifacts/sens_1pl`) | **failure of the estimator**: joint MAP degenerate (common a ~1e4-1e7); not interpreted |

### Post hoc / exploratory work (labelled as such in the paper)

| Item | Status | Outcome |
|---|---|---|
| EVR numerical guard (rounding overflow) | done, documented deviation (iii) | exact no-op on valid inputs; unit-tested |
| Regularized 1PL (log a ~ N(0, 0.05^2)) | done (`artifacts/sens_1pl_reg`) | well behaved (a ~ 2-4); EIG vs entropy difference vanishes (median +0.4% mg, -0.4% pi at N=8; 2PL: +1.9%, +1.6%) -> supports Corollary 1 account; H2 gain small (0.75%); coverage 0.84 |
| Late-test entropy crossover vs bank size | done | crossover within 100 items in 94% of (bank, N) on banks <= 1,172 items, 15% on larger ones; Spearman 0.73 |
| Ranking fidelity (Kendall tau) | done (`artifacts/explore_ranking`) | t=25: EIG:mg 0.77, MFI:pi 0.74, naive 0.54 |
| PH1 strong baselines | done (`artifacts/posthoc_baselines`) | see "Experiment PH1 -- outcome" below; rule 1 triggered, paper reframed |
| PH2 regularization path | done (`artifacts/posthoc_shrinkage`) | negative: no shrinkage level meets the criterion |
| Literature refresh (2026-11-04) | done | new: MINCE, Growing Pains (COLM 2026), Can We Trust IRT (2026), continuous-score CAT (2026), LAAT (2025); none propagates item-parameter uncertainty into adaptive selection |
| Manuscript template | done | official JMLR style `jmlr2e.sty` (JmlrOrg/jmlr-style-file f413f63), `[preprint]` option |

### Open threat raised by the regularized 1PL (falsification, 2026-11-04)

On the targets shared by the main and the regularized-1PL runs, a *plug-in*
CAT with a common, regularized discrimination is much better calibrated than
plug-in 2PL CAT (coverage 0.81 vs 0.54) and close to EOST-M on log loss and
score error. Competing explanation for H2: **EOST-M beats an unregularized
plug-in 2PL, not plug-in CAT as such.** This must be tested with strong
plug-in baselines on the confirmatory design before any claim is written.

## Experiment PH1 -- strong baselines (plan written before running)

Design: the 8 confirmatory banks, the same targets, replicates, holdouts and
panels as the confirmatory study (same seeds), panel sizes N in {4, 8, 16}
(the small-panel regime of the claims). Policies:

* `eig:mg`, `mfi:pi` -- recomputed; must reproduce the stored confirmatory rows
  exactly (reproducibility check, max |diff| = 0).
* `efi:mg` -- posterior-expected Fisher information E_psi[I_psi(theta)],
  posterior-weighted over theta, with ERF updates (the "averaging information"
  criterion of Corollary 2; cf. Laplace-PSN-IRT).
* `mfi:eap`, `eig:eap` -- plug-in 2PL with posterior-mean (EAP) item parameters
  from the same item posterior (better point estimates, no integration).
* `mfi:1pr`, `eig:1prmg` -- regularized one-parameter model (log a ~ N(0, 0.05^2),
  EB prior on b): plug-in Rasch-type CAT, and its calibration-marginal EIG.

Endpoints and tests: integrated holdout log loss, score IAE and |coverage-0.9|;
per-target means over replicates; two-sided Wilcoxon per bank; Holm over this
post hoc family; median relative differences across banks.

Decision rules fixed in advance:
1. If a strong plug-in baseline (`mfi:eap` or `mfi:1pr`) is not significantly
   worse than `eig:mg` on log loss at N = 8 in at least 5 of 8 banks, the paper
   must state that the advantage of EOST-M over plug-in CAT is largely an
   advantage over unregularized plug-in parameters, and reframe accordingly.
2. If `efi:mg` is significantly worse than `eig:mg` in at least 5 of 8 banks,
   this supports the practical relevance of Corollary 2; otherwise the paper
   states that averaging Fisher information performed comparably in practice.
3. Reproduction rows must match the confirmatory artifacts exactly.

Resume: `docker compose run --rm -T bench ...` is not used here; run
`python scripts/posthoc_baselines.py --datasets <banks>` in the runtime
container (per-target checkpoints in `artifacts/posthoc_baselines/<bank>/parts`).

## Experiment PH1 -- outcome (2026-11-04)

Reproduction exact (12,570 rows). At N = 8: posterior-mean plug-in 2PL is far
worse than EOST-M (log loss +15.5%, 7/8 significant). Regularized Rasch-type
plug-in (`mfi:1pr`): EOST-M lower log loss on 8/8 banks but median only 2.1%
(4/8 significant, rule 1 **triggered**), score error lower on 5/8 (median 2.9%;
15.3% at N = 16), but `mfi:1pr` has better coverage (0.81 vs 0.75).
`efi:mg` = `eig:mg` in practice (rule 2 **not supported**). The paper was
reframed: the decisive ingredient is not trusting item-specific
discriminations from small panels; integration and constraint are two remedies
with complementary strengths.

## Experiment PH2 -- regularization path (plan written before running)

Question raised by PH1: can a calibration-marginal model whose discriminations
are shrunk towards a common value keep EOST-M's predictive accuracy while
gaining the constrained model's calibration? Exploratory; no confirmatory claim.

Design: confirmatory banks, targets, replicates, holdouts and panels; N = 8.
Calibration as in the main analysis (same joint MAP abilities, same
empirical-Bayes hyperprior for b and for the mean of log a), except that the
prior standard deviation of log a is fixed to tau in {0.05, 0.15, 0.30, 0.50}
(the main analysis uses its EB estimate, median 0.68). For each tau: EIG under
the marginal model (`eig:mg@tau`) and MFI under plug-in parameters
(`mfi:pi@tau`).

Reported: median over banks of the relative log-loss and score-IAE difference
to EOST-M (main), and mean coverage, for each tau (a trade-off curve).
"Promising" (pre-stated): some tau reaches mean coverage >= 0.80 while its
median log loss is at most 0.5% worse than EOST-M's. If so, it is proposed as
the method of a future pre-registered study, not as a confirmed result.

## Experiment PH2 -- outcome (2026-11-04)

Reproduction of the confirmatory EIG:mg rows exact. Fixing sd(log a) to
tau in {0.05, 0.15, 0.30, 0.50}: coverage of EIG:mg 0.74-0.76 (EOST-M 0.75),
median log loss +0.2% to +2.4%, score IAE +1% to +14%; plug-in MFI coverage
0.65-0.73. **No tau is promising** (criterion fixed before running). Decision:
abandon "shrink the 2PL discriminations" as the way to combine the remedies.
Open question for a next iteration: the Rasch-type model also re-estimates the
calibrators' abilities on its own scale; test whether its calibration
advantage comes from that scale (e.g. 2PL marginal with the 1PL ability scale),
with a new pre-registration before any confirmatory claim.

## Analysis PH3 -- where does the Rasch-type coverage come from? (plan written before computing)

Question left open by PH1/PH2: is the better coverage of the regularized
Rasch-type plug-in test (`mfi:1pr`, 0.81 vs 0.75 for EOST-M at N = 8) due to
wider intervals or to more accurate interval centres? Exploratory; analysis of
the existing PH1 artifacts only (no new runs, no new seeds).

Quantities, N = 8, policies `eig:mg` (EOST-M), `mfi:1pr`, `eig:1prmg` (marginal
regularized 1PL) and `mfi:pi` (reference): per bank, target-level means of
coverage, mean width of the 90% score interval (`mean_width`) and score IAE;
the width-to-error ratio `mean_width / iae`, read against 4.11, the ratio of a
calibrated Gaussian 90% interval (2 x 1.645 / sqrt(2/pi)); paired Wilcoxon of
`mean_width` (`mfi:1pr` vs `eig:mg`) per bank, Holm over the 8 banks; Spearman
correlation between the width-to-error ratio and coverage over all
(bank, policy) cells.

Interpretation rule (fixed now): if `mfi:1pr` intervals are wider (median over
targets) on every bank where its coverage exceeds EOST-M's, and its IAE is not
lower than EOST-M's on at least half of those banks, the paper will say that
its better coverage is bought with wider intervals rather than with more
accurate centres. If its intervals are not wider on those banks, the gain comes
from the centres and the question stays open. Mixed outcomes are reported as
mixed.

Erratum to the plan above (arithmetic slip, not part of the decision rule): the
Gaussian reference ratio is 2 x 1.645 / sqrt(2/pi) = 4.12, not 4.11.

## Analysis PH3 -- outcome (2026-11-04)

`scripts/analyze_coverage.py` -> `artifacts/posthoc_coverage/`. Rasch-type
intervals wider than EOST-M's on 8/8 banks (median width ratio 1.29,
Holm-significant on 8/8); its coverage is higher on 7/8 banks (WinoGrande
equal), and its score IAE is not lower on 4 of those 7. Verdict by the rule
fixed in advance: **wider_intervals** -- its better coverage is bought with
wider intervals, not with more accurate centres. Coverage follows the
width-to-IAE ratio almost exactly (Spearman 0.98 over 32 bank x model cells);
every cell is below the Gaussian reference 4.12 (max 3.97); mean ratios:
plug-in 2PL 2.19, EOST-M 3.02, Rasch-type 3.48, marginal 1PL 3.59. Context
from the confirmatory artifacts: Random:mg covers 0.88 at N = 8 on the same
banks and EOST-M covers 0.89 on the well-specified synthetic bank, so the
shortfall arises when adaptive selection meets a misspecified model.

Also found and fixed during this iteration (reproducibility, not science):
`./run sensitivity` called the frozen runner without the EVR guard, so a
from-scratch rerun would have stopped on the 1PL HellaSwag bank exactly as the
original run did; it now uses the checkpointing runner with `--evr-guard`
(how the committed artifacts were produced). `./run all` now also regenerates
the post hoc 1PL variant (`./run regularized`) and PH3 (`./run coverage`).

Next iteration (not started; needs its own plan before any run): recalibrate
the interval width, e.g. a conformal factor estimated by leave-one-calibrator-out
replay of the panel, and/or mix a small fraction of uniformly sampled items into
the adaptive test; success criterion to be fixed in advance (coverage within
0.03 of nominal on the real banks without losing EOST-M's score-IAE advantage
over uniform subsampling).

## Final state of this iteration

Manuscript (JMLR style) reframed on the evidence; adversarial audit in
`docs/AUDIT.md`; `CLAIMS_GATE=PASS` (61/61, including the PH3 claims). The
open question of PH2 is answered by PH3 (wider intervals). Next actions if
resumed: the interval-recalibration iteration planned above; author names,
affiliations, funding and repository URL in `paper/main.tex` and
`paper/sections/acks.tex` (red placeholders).

Resume from a clean state: `./run verify` (hash gates), `./run claims`,
`./run paper`; every experiment directory under `artifacts/` is complete
(`config.json` or `_analysis/decisions.json` present), so `./run campaign`,
`./run sensitivity` and `./run regularized` skip finished banks.

## Iteration 3 (2026-11-04): external review -> study 2 (plan written before running)

An external review (26 points) asked for: repositioning around calibration
uncertainty; related work (Mislevy et al. 1994; van der Linden & Ren 2020; Fink,
Koenig & Frey 2025) and a fully Bayesian CAT comparison; a new frozen confirmatory
study with the Rasch-type, fully Bayesian and Random:mg competitors; family-cluster
inference; an exact (not probit-approximate) information cap; myopic-optimality
wording; interval recalibration; EB robustness; cost/break-even; ATLAS,
tinyBenchmarks and MINCE run, not only cited; a prospective deployment; verifiable
freezing; environment provenance; a shorter, refocused manuscript.

Environment facts checked first: Hugging Face, public time-stamping services
(RFC 3161, OpenTimestamps), Zenodo and OSF are blocked by the sandbox policy; the
GitHub CLI is not authenticated; the pinned leaderboards have no newer submissions
(remote HEAD = pinned commit for SWE-bench, LiveCodeBench and CRUXEval). Consequences:
no live model inference here (a live protocol is released instead, scripts/live_eost.py),
no external time stamp (the paper says "frozen analysis plan"), and a temporal split on
SWE-bench Verified replaces a prospective out-of-time sample.

Done before any study-2 run on a confirmatory bank:
* strict family clusters (configs/study2_clusters.json): kappa-based components chain
  into giant clusters on OLLB (353/395 systems on HellaSwag), so near-duplicates are
  defined by the correlation of standardized 2PL residuals (>= 0.5), joint over banks
  that share systems: 176 OLLB clusters (largest 39), 92 SWE-bench Verified, 15
  LiveCodeBench, 12 CRUXEval;
* cluster-aware re-analysis of study 1 (scripts/analyze_clusters.py, post hoc):
  H2 is significant on 7/8 banks with targets as units, 5/8 with heuristic families,
  4/8 with strict clusters (median effect unchanged, 5.4%; cluster bootstrap 95% CI
  of the median 4.3-6.4%); H3 7/8 and H4 5/8 survive; H1 4/8 throughout;
* cost analysis (scripts/analyze_cost.py): break-even against full evaluation ~9 new
  systems with 8 fresh calibrators; against naive subsampling at matched precision
  ~122 systems (EOST needs 22.5 items to match 100 random ones); against Random:mg
  65 items to match 100.
* development runs on the development banks only (artifacts/study2/dev_v1): leave-one-
  calibrator-out recalibration raises EOST-M coverage at N = 8 (CRUXEval-I 0.84 -> 0.88,
  SWE-bench Lite 0.76 -> 0.82) but not at N = 4.

Selection rule for the primary interval recalibration, fixed before the second
development run (artifacts/study2/dev): among the standardized-quantile (Rq),
standardized-rms (Rr) and absolute-error-quantile (Ra) rules, the primary rule is the
one whose EOST-M coverage at N in {4, 8}, averaged over the development banks, is
closest to 0.90; ties within 0.01 are broken by the lower mean interval score.

Second development run (artifacts/study2/dev): EOST-M coverage at N in {4, 8}
averaged over development banks -- raw 0.80, Rq 0.80, Rr 0.80, Ra 0.83 -> **Ra**
chosen by the rule above (written before the run). No rule helps at N = 4.

Study 2 frozen at 2026-11-04T04:02:37Z (configs/STUDY2.lock.json, commit bd46f79);
the plan, hypotheses and lock were sent to the first author before the first run.
Launched immediately after (scripts/launch_study2.sh; one container, main ->
temporal -> synthetic). While it runs: EB robustness analysis
(scripts/analyze_eb.py), sampler mixing diagnostics on development banks
(scripts/fb_mixing.py: the translation move brings the lag-3 autocorrelation of
the mean calibrator ability from 0.86-0.94 down to at most 0.16; scale
hyperparameters 0.65-0.89), and an independent code review of the study-2 modules.

### Independent code review of the frozen study-2 code (during the run; before any study-2 result was looked at)

Findings (reviewer had no access to results; the main component had finished GSM8K only):
1. **tinyBenchmarks can use fewer anchors than its budget.** With N calibrators, items have at
   most 2^N distinct response patterns, hence at most 2^N distinct (a, ab) embeddings; K-Means
   then yields fewer anchors than the budget, and the number used was not recorded. At N = 8,
   fewer than 100 distinct embeddings occur in 100% of SWE-bench Verified panels, 98%
   LiveCodeBench, 76% GSM8K, 29% HellaSwag; at budget 50 it affects ~7% of GSM8K and ~3% of
   SWE-bench cells; at N = 4 at most 16 anchors exist. This is a property of the method with a
   small panel, but the comparison "error after T items" is then biased against tinyBenchmarks.
2. **The fully Bayesian sampler has not converged on HellaSwag.** The translation move is only
   allowed when every sampled difficulty stays on the grid; on HellaSwag many difficulties sit
   near the grid edge, so the move rarely acts, and the overall location drifts (mean calibrator
   ability from ~1.7 to ~0.4) throughout the retained sweeps (it levels off after 20-30 sweeps).
   GSM8K, ARC and SWE-bench are unaffected; LiveCodeBench possibly.
3. Minor: significance direction is taken from the sign of the mean difference although the
   test is rank-based (same convention as study 1); the recalibration window is [t/1.5, 1.5t]
   (the plan says "+-50%"); the translation move is exact only up to its window truncation;
   K-Means can return centroids of the last update with labels of the previous one.

Decided now, before looking at any study-2 result:
* The frozen run is completed unchanged, and every frozen decision is reported as computed.
* Post hoc correction for (1): recompute deterministically, for every N = 8 cell, the number of
  anchors tinyBenchmarks actually used; report the share of affected cells, the C8 contrast
  restricted to cells with the full budget, and an items-matched comparison (EOST-M's error
  after the number of items tinyBenchmarks actually used).
* Post hoc correction for (2): re-run the two fully Bayesian arms on every bank (N = 4, 8, same
  targets, panels, holdouts and seeds) with a burn-in of 60 sweeps instead of 10 (draws and
  thinning unchanged), check the location trace, and report C3 for both versions; the
  substantive statement about fully Bayesian CAT will rest on the corrected version, and the
  frozen-version decision is reported as invalid on the affected banks.
* For (3): report whether any Holm-significant contrast has opposite signs of mean difference
  and rank-biserial correlation; correct the wording of the window and of the move.

Addition decided before any study-2 result (exploratory, labelled as such): while implementing
the sampler correction it became clear that the frozen fully Bayesian arm targets the posterior
of van der Linden & Ren's algorithm, which averages each item's response function over the
draws *separately*. With a small panel, the overall location (and scale) of the calibration is
uncertain (posterior sd of the mean calibrator ability ~ 1/sqrt(N)) and shifts all items
together; averaging item by item turns this common shift into independent item-level noise and
flattens every expected response function. The joint fully Bayesian model keeps the shift
common: the target's likelihood is a mixture over draws of products of each draw's expected
response functions. The correction run therefore also evaluates the joint model ('fbj:eig',
'fbj:bmi'; same chains, burn-in 60). Expectation stated now: the joint model behaves like
EOST-M (the common shift only perturbs the target's prior), while per-item averaging can be
worse. This is exploratory and will be reported as such.

Implementation note (2026-11-04 ~05:10 UTC, still before any study-2 result was inspected): the
correction script first covered N = 8 only; it was extended to N = 4 (`--N`) so that it matches
the plan above. A reproduction check (`--verify-frozen`: first target of a bank, frozen burn-in of
10 sweeps) reproduced the frozen fully Bayesian rows exactly (max |diff| = 0 on GSM8K at N = 8 and
N = 4 and on ARC at N = 4), so the corrected arms share targets, panels, holdouts and chains with
the frozen ones and differ only by the burn-in. The corrections run in a separate container
(`scripts/launch_s2_posthoc.sh`) that starts once the frozen main component has finished; the
frozen run itself is not touched. A sign-consistency check for finding (3) was added to
`scripts/analyze_s2_corrections.py` (Holm-significant contrasts of both studies whose mean
difference and rank-biserial correlation disagree).

## Study 2 -- outcome (2026-11-04, ~07:30 UTC)

Frozen run complete (all S2_RUN_GATE=PASS; STUDY2_GATE=PASS before every component); 267
main targets (40 per bank, 15 LiveCodeBench, 12 CRUXEval-O), 31 temporal targets, 20 per
synthetic bank. Frozen analysis (`./run analyze2`, STUDY2_ANALYSIS_GATE=PASS), N = 8,
supported = >= 5 banks Holm-significant for EOST-M and none against:
R2 5/0 (median +5.9%) supported; R3 6/0 (+53.9%) supported; R4 5/0 (+41.6%) supported;
C1 5/0 (+2.4%) supported; C2 4/0 (+1.8%) not supported; C4 4/0 (+21.1%) not supported;
C5b 0/2 (+1.0%) not supported (recalibration significantly worsens the interval score on
HellaSwag and WinoGrande); C6 6/0 (+35.0%) supported; C7 4/0 (+26.0%) not supported;
C8 4/0 (+29.6%) not supported; C9 5/0 (+44.8%) supported; C3 equivalence 4/8 (needs 6) not
supported; C5a recalibrated coverage inside [0.85, 0.95] on 5/8 (needs 6) not supported.
Temporal component (31 later SWE-bench Verified submissions): every contrast except C5b
favours EOST-M at p_holm < 0.05.

Post hoc corrections so far: the reproduction check of the frozen fully Bayesian rows
passes on all eight banks (max |diff| = 0); the tinyBenchmarks anchor audit is complete.
Note on the anchor audit: the first version of `analyze_s2_corrections.py` applied Holm
within the eight corrected C8 tests only. Because the frozen C8 decision uses the 88-test
Holm family, that version was less conservative than the frozen rule (it made ARC
significant: within-8 p_holm 0.017 vs 0.13 in the frozen family, with an identical raw p).
The reported version substitutes the corrected p-values into the frozen 88-test family:
C8 restricted to full-budget cells 4/0 and items-matched 4/0 (median +29.6%), i.e. the
frozen decision is unchanged. Both versions are kept in
`artifacts/study2_posthoc/_analysis/tiny_tests.csv` (columns p_holm, p_holm_within8).
Sign check: 0 of 815 Holm-significant contrasts (both studies) have a mean difference and
a rank-biserial correlation of opposite signs.

Fully Bayesian correction, N = 8 (complete 2026-11-04 ~08:26 UTC; N = 4 running): the
post hoc re-analysis of the frozen arm reproduces the frozen C3 exactly (4/8 equivalent,
median +0.32%). Location drift of the mean calibrator ability over the 32 retained sweeps
(median over cells): frozen window 0.51 on HellaSwag, 0.12-0.24 elsewhere; burn-in 60:
0.24 on HellaSwag, 0.13-0.19 elsewhere (posterior sd of the location ~0.32 at N = 8). With
the corrected burn-in: BMI 3/8 equivalent, median +0.39%; EIG 3/8, +0.50%; no benchmark
Holm-significant in either direction for log loss or score error (96-test family); coverage
0.76 (EOST-M 0.74). Exploratory joint model: EIG 5/8 equivalent, median +0.05%; BMI 5/8,
+0.33%; no significant difference. As expected before running, the joint model is closest to
EOST-M and per-item averaging is slightly worse; the substantive conclusion (integrating
calibrator abilities and the hyperprior does not improve on EOST-M with eight calibrators)
rests on the corrected arms, and the frozen C3 decision (not supported) is reported as
computed.

Fully Bayesian correction, N = 4 (complete 2026-11-04 09:30 UTC; all corrections:
S2_CORRECTIONS_GATE=PASS, S2_POSTHOC_ALL_COMPLETE): frozen arms 4/8 equivalent (BMI +0.33%,
EIG +0.45%); burn-in 60: BMI 4/8 (+0.21%), EIG 2/8 (+0.45%; EOST-M significantly better on one
bank for log loss and one for score error); joint model 4/8 (BMI -0.03%, EIG +0.17%), no
significant difference; coverage 0.79-0.81 against EOST-M's 0.78. Location drift is larger at
N = 4 (posterior sd of the location ~0.44) and is only partly reduced by the longer burn-in.
Same conclusion as at N = 8.

## Final state of iteration 3 (2026-11-04)

* Study 2 (frozen plan, 267 targets, one per strict cluster): 6 of 13 frozen decisions
  supported (R2, R3, R4, C1, C6, C9); not supported: C2, C4, C7, C8 (each 4/8 significant,
  none against), C5b (recalibration does not improve the interval score), C5a
  (recalibrated coverage inside [0.85, 0.95] on 5/8), C3 (fully Bayesian equivalence 4/8).
  Temporal component: 10/11 contrasts significant for EOST-M.
* Post hoc corrections (planned before results): tinyBenchmarks anchor audit leaves C8
  unchanged; fully Bayesian CAT with a 60-sweep burn-in is never significantly different
  from EOST-M at N = 8 (3/8 equivalent, +0.4-0.5%), joint model closest (+0.05%); same
  picture at N = 4. Sign check: 0/815 discordant.
* Manuscript: rewritten around the three messages with every study-2 number as a macro;
  77/77 claims hold (CLAIMS_GATE=PASS); PREREG_GATE, STUDY2_GATE, DATA_GATE and VERIFY_GATE
  pass; 56 pages (JMLR format), main text 29 pages.
* Not done, with reasons: live prospective deployment (no model access here; protocol
  released and mock-validated), external time stamp (no registry reachable), an ICLR 9-page
  version (target venue is JMLR).

Independent manuscript review (2026-11-04 ~10:00 UTC, after all results): about
60 numbers checked against the artifacts and found correct; findings fixed: a sign error in the
Student-t hyperprior sentence (it is slightly worse, not better), a study-1 Random:mg figure
given as a ratio of means instead of the median used elsewhere (now 25.5%), an incorrect
explanation of the ATLAS-style result (fixed lower asymptote: no extra parameters), the
Gaussian assumption of the information cap now stated wherever the cap is used (with the
general bound for other posteriors), several overclaims softened (abstract and conclusion now
say that the frozen criterion was met only against MINCE among reduction methods and report
6/13 supported decisions; fully Bayesian "no significant difference, equivalence not shown";
recalibration "not yet validated"; criterion-insensitivity labelled descriptive and limited to
prediction), the break-even against uniform subsampling added next to the one against full
evaluation, and consistency fixes (burn-in "some benchmarks", joint EIG vs BMI, frozen FB rows
labelled, Rasch-type "inside [0.85, 0.95]"). Claims re-checked: 77/77.

## Iteration 4 (2026-11-04): second external review -> robustness follow-ups (plan written before running)

A second review of v3 raised two methodological points and several JMLR formatting
points. The frozen decisions of both studies are not touched; everything below is a
robustness follow-up, labelled as such in the paper.

RF1 -- target-blind clustering and threshold sensitivity. Concern: the strict clusters use
the target's own responses (2PL residual correlations over all systems of a bank); excluding
residually similar calibrators could make panels harder to calibrate and favour a method built
for calibration uncertainty. Design: frozen study-2 targets, holdouts, replicates and panel
seeds, N = 8; only the set of systems excluded from the target's panel changes:
(a) metadata only (heuristic families, no response data); (b) leave-bank-out residual
clusters: heuristic family OR residual correlation >= 0.5 computed on the other banks that
contain the same systems (the other five Open LLM Leaderboard banks; CRUXEval-I for CRUXEval-O;
SWE-bench Lite for the 25 SWE-bench Verified systems it contains), metadata only for
LiveCodeBench; (c) joint residual clusters as frozen but at thresholds 0.4, 0.6 and 0.7; the
frozen threshold 0.5 is recomputed as a reproduction check (must reproduce the frozen rows
exactly). Policies: EOST-M, plug-in Fisher CAT, Rasch-type plug-in (R2, C1), plus Random:mg
and Naive (C4, R3; secondary). Targets whose eligible pool has fewer than 8 systems under a
variant are dropped from that variant and counted. Analysis per variant: per-bank paired
Wilcoxon on target means, median relative improvement, positives, Holm-significant counts in
the frozen 88-test family with the variant's R2 and C1 p-values substituted. Rule: R2 and C1
are robust to the clustering if under every variant both keep a positive median, are positive
on >= 7 of 8 banks and have at least 4 Holm-significant banks (frozen count minus one) and none
against; otherwise the paper states which variant breaks which conclusion.

RF2 -- MCMC quality of the Bayesian CAT. Design: per bank, 6 frozen study-2 targets drawn with
a fixed seed, replicate 0, N = 8; 4 chains (chain 1 starts at EOST-M's estimates like the frozen
arm; chains 2-4 from dispersed points: abilities + N(0, 0.5^2), mu_b +- 0.5, sigma_b x 0.7 or
1.4, mu_a +- 0.3, sigma_a x 0.7 or 1.4), burn-in 250 sweeps, then 100 retained draws per chain
at thinning 3 (400 retained draws). Diagnostics per panel: rank-normalized split R-hat, bulk
ESS and MCSE for the location (mean calibrator ability), mu_b, sigma_b, mu_a, sigma_a (300
post-burn-in sweeps per chain); between-chain spread of the held-out log loss of the
posterior-draw marginal model; trace plots of one panel per bank. Arms on the same cells:
posterior-draw marginal Bayesian CAT (BMI, EIG; 400 pooled draws) and joint fully Bayesian CAT
(EIG, BMI; 40 draws, every 10th pooled draw), compared with EOST-M and with the frozen
(burn-in 10) and corrected (burn-in 60) arms. Rule: report the share of panels with R-hat
< 1.01 and < 1.05 and bulk ESS >= 100 per parameter and bank; the study-2 statement about the
Bayesian CAT (no significant difference from EOST-M, equivalence within +-1% not shown) is
confirmed if the long-chain arms differ from the burn-in-60 arms by a median of less than 0.5%
in held-out log loss and their median difference to EOST-M stays within +-1%; otherwise the
paper reports the long-chain result as the better estimate.

RF3 -- cross-benchmark effect (secondary; no frozen decision changes). For every study-2
superiority hypothesis: random-effects meta-analysis over the 8 benchmarks of the per-bank
relative improvement (DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-Jonkman 95% interval with
k - 1 degrees of freedom, 95% prediction interval, I^2), per-bank standard errors from 2,000
bootstrap resamples of targets (fixed seed), and an exact two-sided sign test over benchmarks.

Editorial (no new data): JMLR format (abstract <= 200 words, running title <= 50 characters,
five keywords, main PDF of about 40 pages with an online appendix, length note in the cover
letter); "fully Bayesian CAT" (per-item averages over posterior draws, van der Linden-Ren)
renamed "posterior-draw marginal Bayesian CAT", the exploratory joint version "joint fully
Bayesian CAT"; comparisons with ATLAS, tinyBenchmarks and MINCE always qualified as the
eight-calibrator regime. Still not done: a live prospective study (no model access); the
Zenodo deposit is made with the tagged release.

RF1 outcome (2026-11-04 10:00 local): the recomputed frozen clustering (threshold 0.5)
reproduces all 4,005 frozen rows exactly. No target was dropped under any rule. Under every
alternative panel-exclusion rule (metadata only; leave-bank-out residuals; joint residuals at
0.4, 0.6, 0.7), R2 is positive on 8/8 banks with 4-6 Holm-significant (frozen 5) and C1 on 7/8
with 5 Holm-significant (frozen 5), none against; median improvements unchanged (R2 5.9%, C1
2.4-2.4%). Rule met for all variants. Secondary: C4 3/0 under every alternative rule (frozen
4/0; median 18-21%), R3 6/0. Response-derived links matter only for the Open LLM Leaderboard
banks; on SWE-bench Verified, LiveCodeBench and CRUXEval-O the clusters are essentially the
heuristic families.

RF3 outcome: pooled random-effects relative improvements exclude zero for 10 of 11 study-2
superiority hypotheses (all but C5b), including C2 (2.0%), C4 (23.4%), C7 (27.1%) and C8
(34.5%), which missed the frozen threshold by one bank; heterogeneity is large (I^2 75-93% for
these four) and the 95% prediction interval for a new benchmark excludes zero only for R2, R3,
R4 and C9. No frozen decision changes.

RF2 rule, operational reading (written 2026-11-04 10:22 local, while RF2 was running; two of
eight banks finished, no comparison with EOST-M or with the burn-in-60 arms computed yet).
"The long-chain arms" is read as all four long-chain arms (posterior-draw marginal BMI and EIG,
joint EIG and BMI), because the study-2 sentence it tests ("Bayesian tests that also integrate
the calibrators' abilities were not significantly different from EOST-M") covers both
versions. "Differ ... by a median of less than 0.5%" is evaluated both as the median of the
signed relative differences and as the median of their absolute values; the statement is
confirmed only if both readings pass for every long-chain arm and the signed median difference
to EOST-M of every long-chain arm is within +-1%. The narrower reading (posterior-draw marginal
arms only, signed medians only) is also reported. Added for explanation only (not part of the
rule): R-hat and ESS of the ability scale (standard deviation of the calibrators' abilities),
computed from the saved traces, because the sampler has a location move but no scale move.

RF2b -- exploratory sampler fix (plan written 2026-11-04 10:30 local, before any RF2b run; NOT
part of the plan of iteration 4, and designed after RF2's convergence diagnostics had been seen
on GSM8K, ARC-C and HellaSwag: overall location converged, sigma_b, mu_a, sigma_a and the
ability scale not, with the ability scale correlated with sigma_b (r ~ 0.9) and mu_a
(r ~ -0.9); no comparison of RF2's arms with EOST-M had been inspected beyond the two-bank
pipeline test). Diagnosis: multiplying abilities and difficulties by s and dividing
discriminations by s leaves the likelihood unchanged; with eight calibrators only their N(0, 1)
prior fixes the scale, and the frozen sampler has a translation move but no scale move.
Design: the RF2 sampler plus, at the start of every sweep, a Metropolis-Hastings move along the
scale ridge on the posterior of (abilities, hyperparameters) with the item parameters summed
out on the grid (abilities, mu_b, sigma_b multiplied by s; mu_a decreased by log s;
log s = +-0.2 or +-0.4, multiples of the log-discrimination grid step; acceptance ratio with
the Jacobian s^(N+2); the collapsed target is the joint posterior implied by the frozen
sampler's conditionals); the items are then drawn from their exact conditional (partially
collapsed Gibbs). Same 6 targets per bank, panels, holdouts, chain starting points and replay
arms as RF2; burn-in 200 sweeps, then 100 retained draws per chain at thinning 2 (400 draws).
Validation before running (tests/test_rf2b.py): with the move off the sampler is RF2's sampler
exactly; the move's stationary law along an orbit matches the exact one; on a small problem
long runs with and without the move agree. Report: RF2's diagnostics plus the ability scale and
the acceptance rate, and RF2's comparisons (EOST-M, burn-in-60 arms, RF2 long chains).
Interpretation rule: if rank-normalized split R-hat < 1.05 for every recorded parameter on at
least 90% of panels and bulk ESS >= 100 on at least 90%, the RF2b arms are treated as the
converged comparator and the paper's statements about the Bayesian CATs rest on them:
"not significantly different from EOST-M" is kept only if, for every RF2b arm, the median
relative difference in held-out log loss to EOST-M is within +-1% and the bank medians are not
all on one side (exact two-sided sign test over banks, p >= 0.05); otherwise the paper reports
the direction and size of the difference. If the convergence criterion is not met, the paper
says that no converged fully Bayesian comparator was obtained and limits its statements to the
implementations studied.

Correction (2026-11-04 10:45): the RF2b entry first gave its time as 11:00; it was written at
10:30 (commit 54f01f5, 10:31). Only the stated time changed.

RF2 outcome (2026-11-04 11:20 local; 48 panels, 8 banks, 4 chains x 550 sweeps, 400 retained
draws). Convergence, share of panels with R-hat < 1.05 / bulk ESS >= 100: location 92% / 88%
(median R-hat 1.01, median ESS 778), mu_b 90% / 85%, sigma_b 0% / 0% (median R-hat 1.47, max
2.46, median ESS 8), mu_a 2% / 0% (median R-hat 1.50, max 2.77), sigma_a 8% / 4% (median R-hat
1.27, max 3.04); ability scale (explanatory) 0% / 0% (median R-hat 1.40), correlated with
sigma_b (median r = 0.88) and mu_a (r = -0.86); MCSE 30-38% of the posterior sd for these four
quantities. Between-chain spread of the PDM-BMI held-out log loss: median 1.0%, max 43%, > 5% on
10% of panels. Comparison on the same 48 cells: the four long-chain arms differ from EOST-M by a
median of +0.17% (PDM BMI), +0.04% (PDM EIG), +0.20% (joint EIG) and -0.45% (joint BMI) in
held-out log loss, banks split (3/5, 4/4, 4/4, 6/2 better/worse; sign tests p >= 0.29), within
+-1% on 7-8 of 8 banks; their signed median change relative to the burn-in-60 arms is +0.05% to
+0.13% (joint BMI -0.25%) but the median absolute change is 0.61-1.02%. Rule: NOT confirmed under
the reading recorded before the comparison (the absolute changes exceed 0.5%); confirmed under
the narrow reading (posterior-draw marginal arms, signed medians). Per the rule the paper reports
the long-chain results as the better estimate. Mean coverage on these cells: EOST-M 0.78,
long-chain arms 0.86 (burn-in 60: 0.80-0.84; frozen: 0.82-0.85). RF2b (scale move) started
11:17.

RF2c -- exact sampler for the fully Bayesian model of Lemma 1 (plan written 2026-11-04 11:31
local, before any RF2c run; exploratory, NOT part of the plan of iteration 4). Designed after
RF2's outcome and after the first 10 panels of RF2b, whose diagnostics show the scale move
working (R-hat < 1.05 for the ability scale on 8/10 panels, sigma_b 9/10, mu_a 7/10) but
sigma_a converging on 0/10 (one chain stuck at its lower bound 0.1). No comparison of RF2b's arms
with EOST-M has been looked at. Diagnosis: (1) the conjugate hyperparameter draws given the
sampled grid points mix slowly for a variance parameter when each item carries little
information (eight calibrators); (2) those draws treat the grid prior as a continuous density,
while EOST-M (Lemma 1) normalizes it over the grid; when sigma_a approaches the grid step
(0.2 in log a) the two differ and the unnormalized version piles up at the lower bound. Design:
sample Lemma 1's model exactly (calibrators' abilities N(0, 1); item prior normalized over the
grid; flat priors on the means, Jeffreys priors on the standard deviations on the frozen
sampler's box) with, per sweep: two rounds of random-walk Metropolis moves on mu_a,
log sigma_a, mu_b, log sigma_b with the item parameters summed out (step sizes adapted during
burn-in only, target acceptance 0.4); RF2b's scale move; the items from their exact conditional;
the frozen griddy-Gibbs ability update; the frozen translation move corrected by the factor
(Z(eta)/Z(eta'))^I of the normalized prior. Same targets, panels, holdouts, starting points and
replay arms as RF2/RF2b; burn-in 200, 100 retained draws per chain at thinning 2 (400 draws).
Validation before running (tests/test_rf2c.py): the collapsed likelihood equals the direct
computation with the normalized grid prior; the stationary laws of the discrimination moves,
the difficulty moves and the scale move match exact computations; the translation weights match
the joint density along the shift orbit. Report and interpretation rule: as for RF2b (convergence
criterion R-hat < 1.05 for every recorded parameter on >= 90% of panels and bulk ESS >= 100 on
>= 90%; if met, "not significantly different from EOST-M" is kept only if every RF2c arm's median
relative difference in held-out log loss to EOST-M is within +-1% and its bank medians are not all
on one side, exact sign test p >= 0.05; otherwise the paper reports direction and size; if not
met, the paper says that no converged fully Bayesian comparator was obtained). RF2b is completed
and reported as run.

RF2b outcome (2026-11-04 12:25 local; 48 panels). Scale-move acceptance median 21% (1-54%).
Share of panels with R-hat < 1.05 / bulk ESS >= 100: location 92% / 92%, ability scale 75% /
67%, mu_b 100% / 98%, sigma_b 73% / 71%, mu_a 79% / 83%, sigma_a 10% / 0% (median R-hat 1.32,
median ESS 11); 3 of 192 chains mostly at the sigma_a lower bound. Convergence criterion NOT met
(sigma_a; also scale, sigma_b, mu_a below 90%). Between-chain spread of the PDM-BMI held-out log
loss: median 1.0%, max 6% (RF2: max 43%). As the criterion was not met, the comparisons are
descriptive: the four arms differ from EOST-M by a median of +0.10% (PDM BMI), +0.01% (PDM EIG),
+0.07% (joint EIG; 7 of 8 banks worse, sign test p = 0.07) and +0.29% (joint BMI) in held-out
log loss; coverage 0.85, 0.86, 0.76, 0.82 (EOST-M 0.78 on these cells). Explanatory: the
posterior mean of the ability scale was below EOST-M's estimate on 79% of panels (median ratio
0.84). RF2c started 12:22.

RF2c stopped (2026-11-04 12:52 local) after 14 of 48 panels (GSM8K and ARC-C complete,
HellaSwag 2 of 6); its partial artifacts are kept in artifacts/robust/fb_exact and reported as
such. Reason, from the convergence diagnostics only (no comparison with EOST-M looked at): on
GSM8K every recorded quantity converged (max R-hat <= 1.054), but on ARC-C the overall location
did not (R-hat 1.03-1.17) and on HellaSwag it was stuck (R-hat 2.4 and 2.8). RF2c's translation
move is the frozen Gibbs move over shifts of abilities, sampled difficulties and mu_b, corrected
by the factor (Z(eta)/Z(eta'))^I of the normalized prior; when the difficulty prior extends beyond
the edge of the grid (HellaSwag: mu_b ~ -3 to -5, sigma_b ~ 3.5) that factor, raised to the power
of ~2,000 items, blocks almost every shift. This is a flaw of RF2c's design, not of the model.

RF2d -- RF2c with a collapsed translation move (plan written 2026-11-04 12:55 local, before any
RF2d run; exploratory). Design: RF2c's sampler (same model and moves) plus, in every sweep, a
Metropolis-Hastings translation of the abilities and mu_b by +-1, 2 or 4 difficulty-grid steps
with the item parameters summed out (Jacobian 1), which is not blocked by items at the grid edge;
scale steps log s in {+-0.05, 0.1, 0.2, 0.4} (RF2c's +-0.2/0.4 steps were accepted 0-10% of the
time on HellaSwag); burn-in 150 sweeps, then 100 retained draws per chain at thinning 2 (400
draws); same targets, panels, holdouts, starting points (new chain seeds) and replay arms.
Validation before running (tests/test_rf2c.py): the stationary laws of the collapsed translation
(including a difficulty prior truncated by the grid edge) and of the finer scale move match exact
computations. Report and interpretation rule: as for RF2b/RF2c (R-hat < 1.05 for every recorded
parameter on >= 90% of panels and bulk ESS >= 100 on >= 90%; if met, "not significantly different
from EOST-M" is kept only if every RF2d arm's median relative difference in held-out log loss to
EOST-M is within +-1% and its bank medians are not all on one side, exact sign test p >= 0.05;
otherwise direction and size; if not met, no converged comparator was obtained).

RF2d outcome (2026-11-04 14:40 local; 48 panels, 192 chains). Acceptance: translation 68%, scale
47%, hyperparameter moves ~40%. Share of panels with R-hat < 1.05 / bulk ESS >= 100: location
88% / 77%, ability scale 58% / 35%, mu_b 67% / 58%, sigma_b 48% / 33%, mu_a 42% / 27%, sigma_a
73% / 63%; median R-hat 1.02-1.06, largest 1.25 (RF2: 3.04); no chain at the sigma_a lower bound.
Convergence criterion NOT met (slowest quantity mu_a, 42%): no converged fully Bayesian comparator
was obtained, and the paper limits its statements to the implementations studied. Between-chain
spread of held-out log loss: median 0.7%, max 26%. Descriptive comparisons on the 48 cells: the
four RF2d arms differ from EOST-M by a median of -0.31% (PDM BMI), +0.15% (PDM EIG), +0.20%
(joint EIG), -0.13% (joint BMI) in held-out log loss (banks better/worse 5/3, 3/5, 2/6, 4/4;
sign tests p >= 0.29; within +-1% on 6-7 of 8 banks); coverage 0.76-0.79 (EOST-M 0.78); score
error (IAE) lower than EOST-M by a median of 5-11% over cells, on 4-5 of 8 banks. Explanatory:
under Lemma 1's model the posterior mean of the ability scale exceeded EOST-M's estimate on 65% of
panels (median ratio 1.15). Across all 18 Bayesian arm/sampler combinations on these cells the
median log-loss difference to EOST-M stayed within +-0.45%.

Iteration 4 closed (2026-11-04 15:00 local). Paper: Section 7.6 reports RF1-RF3 and the
exploratory RF2b-RF2d (main text: RF2's diagnostics and rule outcome, the scale ridge, RF2b and
RF2d in brief, the comparison across all 18 arm/sampler combinations, the score-error lead);
Online Appendix S4 gives the designs, validation, convergence tables per benchmark (Table S17),
all arms on the same cells (Table S18) and trace plots (Figures S9-S10). Statements about the
Bayesian CATs changed accordingly: introduction (follow-ups kept the median difference below 0.5%),
discussion ("Why not always use a fully Bayesian adaptive test?": the scale ridge, near-converged
exact sampler, open leads; the PDM-vs-joint explanation no longer offered as a reading), limitation
(vi) (no fully converged comparator; HMC on a continuous parametrization suggested), conclusion.
Abstract unchanged (199 words). Main PDF 43 pages, online appendix 26. Claims 84/84, tests pass,
PREREG/STUDY2/DATA gates pass.

## Iteration 5 (2026-11-04): third review of v4 -> RF3 sensitivity and finalization (plan written before running)

The third review (v4) recommends Conditional Accept on the science and asks for finalization,
not new research: (1) a limitation for RF3: five of the eight benchmarks share the same 395
Open LLM Leaderboard systems, a dependence the random-effects model ignores, so RF3 must stay
clearly secondary; (2) never claim that a fully Bayesian optimum was beaten or shown
equivalent. It also asks to keep, in the discussion, the
possible score-estimation advantage of the near-converged Bayesian sampler as an open lead.

RF3b -- source-level sensitivity of RF3 (secondary to a secondary analysis; no frozen decision
changes). Sources: Open LLM Leaderboard (GSM8K, ARC-C, HellaSwag, WinoGrande, MMLU),
SWE-bench Verified, LiveCodeBench, CRUXEval-O (k = 4). Effect per source: RF3's per-benchmark
relative improvement, averaged with equal weights over the five OLLB benchmarks. Standard errors:
for OLLB, 2,000 bootstrap resamples of the 395 shared systems (fixed seed; a resampled system
brings its target-level results in every OLLB benchmark where it is a target); for the other
sources, RF3's bootstrap over targets. Pooling: DerSimonian-Laird tau^2, Hartung-Knapp-Sidik-
Jonkman 95% interval with k - 1 = 3 degrees of freedom; sign count over the four sources.
Explanatory: median pairwise correlation of the bootstrap replicates of the five OLLB effects.
Reporting rule: for each superiority hypothesis the paper states whether the source-level 95%
interval excludes zero, and every sentence based on RF3 is qualified by that count, whatever it
is; RF3's own numbers are not changed.

RF3b outcome (2026-11-04 15:50 local). With the five OLLB benchmarks counted as one source
(k = 4), the pooled 95% interval excludes zero for 6 of 11 superiority hypotheses (R2 +6.6%,
R3 +50.5%, R4 +39.7%, C2 +2.1%, C6 +34.9%, C9 +41.7%); it no longer excludes zero for C1 (+2.3%,
-0.4 to 5.0; positive in 3 of 4 sources), C4 (+23.8%, -5.0 to 52.6), C7 (+20.6%, -1.1 to 42.4)
and C8 (+37.4%, -1.8 to 76.7), all positive in 4 of 4 sources. Of the four hypotheses that missed
the frozen threshold by one benchmark, only C2 keeps an interval excluding zero. The sampling
dependence of the OLLB estimates through shared systems is small (median pairwise bootstrap
correlation 0.02; the OLLB standard error is 1.05 times its value under independence): the loss
comes from having four sources (t with 3 df) and from heterogeneity between them. Per the
reporting rule, every RF3 sentence is qualified accordingly. No frozen decision changes.

Iteration 5 closed (2026-11-04 15:55 local; v4.1). Paper: RF3 paragraph rewritten with the
dependence limitation and RF3b ("point the same way across benchmarks; does not show that they
generalize"); Table 10 gains the source-level interval; limitation (xi) added; the introduction's
RF3 sentence qualified; the score-estimation lead of RF2d kept in the review's wording; the
discussion summary and recommendation (3) limited to "the Bayesian tests we ran". Claims 85/85,
tests and PREREG/STUDY2/DATA gates pass.

## Iterations 6-9 (2026-11-04 to 2026-11-05): form of the manuscript (v4.2 to v5.0)

Presentation only: no analysis was run, and no number, frozen decision or claim changed. The
online appendix was first merged into one continuous article in which each proof follows its
statement and each additional analysis sits in the section that discusses it (v4.4, 66 pages,
kept as the extended version, `paper/main.pdf`); the article itself was then condensed to 25
pages (v5.0, `paper/short.pdf`): all seven theoretical results with condensed proofs, every
frozen decision reported as computed (the unsupported ones included) with the deviations from
the frozen plans, six tables and four figures with one-line captions, all referenced from the
text. The short article reuses the generated number macros and table bodies
(`scripts/short_tables.py`), so every number still comes from the artifacts; each of its
qualitative statements is a condensation of a statement of the extended version. Claims 88/88;
PREREG, STUDY2 and DATA gates pass. `scripts/flatten_short.py` (`./run flatten`) writes the
article as one self-contained `main.tex` with the macros it uses, its figures, `refs.bib` and
`jmlr2e.sty`; compiled alone it gives the same 25 pages.
