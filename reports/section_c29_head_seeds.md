# Section C29 — head-seed replication at n = 8, and the effective-λ control

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-01. Pre-registration `outputs/c29_prereg/C29.0_preregistration.md`, frozen with its
SHA-256 in `outputs/c29_prereg/prereg_lock.json` before any C29 measurement existed. Every
number below is re-derived from JSON by `tests/test_head_seeds.py`.

---

## The verdict, up front

C29 replicated four families of guided runs at **eight head-training seeds** each — the
three C23 anchor arms and the deployed probe-point-12 arm at every anchor — and filled in
the deployed λ envelope at 1.25 / 1.5 / 2.5 so C23's Rule A could be re-priced at matched
steering strength. Every validity gate is an identity at residual **0.0**. Four results
follow, and two of them go against the reviewer objections that motivated C29.

1. **The "25×" is wrong, and not by a little.** C25's headline — head-seed span 0.0916
   against a generation-seed span of 0.0040, "a factor of 23" — compared a head-seed span
   against **the smallest of three per-cell generation-seed spans**. The per-cell spans on
   that same arm are 0.0672 / 0.0413 / 0.0040 at seeds 1234 / 2345 / 3456, and at n = 8 they
   run from 0.0040 to 0.0672. Estimated properly, with the generation-seed sd pooled over
   16 degrees of freedom, the ratio on the decisive arm is **1.71**, 95% F interval
   **[0.96, 3.65]**. C25's 24.76 is not inside that interval, and the interval does not
   exclude 1. **Head seed and generation seed are the same order of magnitude here.**
2. **The qualitative claim survives in a much weaker form.** The head-seed sd on the Rule B
   arm is **0.0366**, 95% chi-square interval **[0.0242, 0.0744]**, against a pooled
   generation-seed sd of **0.0213**. R1 fires; R2 does not. The defensible statement is
   *"head-seed variance is at least as large as generation-seed variance and is never
   reported"*, not *"it is 25× larger"*.
3. **It is about probes in general, not about mid-network probes.** `q =
   sd_head(mid)/sd_head(deployed)` is **1.52** (hbd_count), **1.61** (aromatic_rings) and
   **0.81** (qed) — all three inside the pre-registered [0.5, 2] band. The deployed λ=1
   configuration, which the report treats as *the* result, has a head-seed sd of
   0.0241 / 0.0176 / 0.0142. **Every guidance number in this project inherits this.**
4. **The effective-λ confound is real, is roughly the size the reviewer estimated, and
   flips five of C23's fifteen arms.** Priced against the *coarse* envelope the reviewer
   used, the three named λ=1 arms move +0.0964 → +0.0460, +0.1096 → +0.0592 and
   +0.0701 → +0.0245 — reproducing the reviewer's +0.0459 / +0.0592 / +0.0245 to the fourth
   decimal. Against the newly measured *fine* envelope they are +0.0375, +0.0507, +0.0217,
   i.e. **54% to 69% of the λ=1 headline is steering strength, not depth**. Five of the
   fifteen C23 arms change sign under the correction. **Rule A's "15/15 positive" becomes
   10/15.**

Two further results that C29 did not go looking for:

5. **Rule A survives head-seed variation, comfortably, and P5 is falsified.** Paired by head
   seed at matched λ, mid minus deployed is **+0.0941** [+0.0534, +0.1349] (hbd_count),
   **+0.1270** [+0.0957, +0.1584] (aromatic_rings) and **+0.0266** [+0.0113, +0.0419] (qed).
   All three intervals exclude zero. R4 fires 3 of 3. Corrected to matched effective λ it is
   **+0.1072**, **+0.0632**, **+0.0409** — still 3 of 3. I predicted at most one.
6. **Rule B comes back at n = 8 against C23's own comparator, and dies against every harder
   one.** The head-seed mean advantage over compute-matched best-of-N on the Rule B arm is
   **+0.0490** [+0.0135, +0.0845], 6 of 8 head seeds positive. Against the
   token-conservative comparator it is **+0.0246** [-0.0060, +0.0552]; against C26's
   corrected estimator **+0.0156** [-0.0173, +0.0485]. **C25 retired Rule B on three seeds,
   one of which (3456) is the worst of eight. That retirement was under-powered. The
   correct verdict is not "Rule B holds" but "Rule B is unresolved, and the comparator
   choice decides it."**

**What C29 does not show.** It does not show guided decoding is worth running: the
token-conservative and C26-corrected comparators both contain zero, and C27 already showed
the deployed configuration loses to head-selected best-of-N. It does not rehabilitate C25 —
it corrects C25's central number. And the effective-λ control is a **scalar-moment control,
not the λ-rescale identity**; §C29.9.

---

## C29.0 The pre-registration, verbatim

The block below is `outputs/c29_prereg/C29.0_preregistration.md` from `## C29.0.1 Why`
onward, reproduced in full and unedited.
`tests/test_head_seeds.py::test_the_report_copies_the_prereg_verbatim` asserts the copy is a
byte-identical substring of this section;
`test_the_prereg_was_written_before_every_measurement` asserts its mtime strictly precedes
every C29 artefact; `test_the_prereg_lock_records_the_prereg_hash` asserts the lock file's
SHA-256 matches.

Three things in it are **wrong and are scored as failures**: decision rule R4 is not
scoreable as written (§C29.4), and predictions 2, 3, 5 and 7 are falsified (§C29.7).

<!-- BEGIN VERBATIM PREREG COPY -->
## C29.0.1 Why

Two reviewer objections, both aimed at the only positive end-to-end result the project has
left (C23's Rule A: a mid-network head improves guided generation).

**(a) The variance claim rests on n = 3.** C25 measured a head-seed span of **0.09160960684359803**
on the `hbd_count` probe-point-4 lambda=2 arm (hit rates 0.5603420596566593 / 0.5600814321286762 /
0.46873245281306125 at head seeds 1234 / 2345 / 3456) against a generation-seed span of 0.0037
— a factor of about 25 — and that is what retired C23's Rule B. It is the project's most
transferable contribution: *the standard protocol for evaluating inference-time steering —
one trained probe, error bars over generation seeds — cannot distinguish the method from its
own noise.* But three head seeds give two degrees of freedom, and the sampling distribution
of a span at n = 3 is enormous. The claim needs a real interval.

**(b) Rule A is confounded with an effective-lambda increase that was already measured and
never controlled.** `outputs/c17_layer_steering/layer_steering_metrics.json` records
`mean_head_q_spread_across_candidates`:

| property | probe point 12 (deployed) | probe point 3 | probe point 4 | probe point 6 |
|---|---|---|---|---|
| aromatic_rings | 0.19924567639827728 | 0.30125945806503296 | 0.29877007007598877 | 0.3013074994087219 |
| hbd_count | 0.13209988176822662 | 0.17264041304588318 | 0.16804006695747375 | 0.16643817722797394 |
| qed | 0.10286020487546921 | 0.1233861893415451 | 0.12792399525642395 | 0.12277279049158096 |

By the project's own lambda-rescale identity (§20.3, `lam*log(c*q^a) = (lam*a)*log q + lam*log c`,
the softmax annihilating the additive constant), **a multiplicative rescale of `log q` IS a
lambda rescale**. So "the mid-layer head beats the deployed head at matched lambda" is not
matched *steering strength*. C29 re-prices Rule A against the deployed lambda envelope
evaluated at the spread-implied effective lambda.

## C29.0.2 Design

Dataset `pilot_50k_p2`. Generation seeds **101 / 202 / 303**, unchanged. Windows and target
intervals inherited frozen from `outputs/pilot_50k_p2/windows.json` and `target_intervals.json`;
never re-derived. Base generator GP-MoLFormer-Uniq stays **frozen** — C29 trains *heads* only.
`actual` token accounting. Processed tokens are reported, never wall-clock. The one permitted
DAgger round remains spent and is not used.

**Head seeds, in this order, fixed here:** `1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901`
(n = 8). The order is the truncation order: if the 24-hour budget binds, arms are reported at
whatever prefix of this list completed, with n stated per arm. **The pre-registered minimum is
n = 6**; an arm that reaches only n < 6 is reported as *not replicated at the pre-registered
depth* and no interval is claimed from it.

**Head training.** `scripts/20_pooled_sweep.py`, unmodified, with `--variants last1
--depths mid final --properties aromatic_rings hbd_count qed --out c29_heads`. §C25.0.1
establishes that the `last1` pooling variant is not merely equivalent to the deployed
single-position readout but *the identical feature array through the identical trainer*
(`probe_layers.train_one_probe`), which is script 03's `frozen_state` branch. One route
trains every C29 head, at every depth, so depth is the only thing that varies.

**Arms.** Four families, at every head seed:

| family | key | property | probe point | lambda | seed-1234 reference |
|---|---|---|---|---|---|
| mid | A1 | hbd_count | 4 | 2.0 | `c23_guided_L4_lam2_hbd_count` |
| mid | A2 | aromatic_rings | 3 | 1.0 | `c23_guided_L3_lam1_aromatic_rings` |
| mid | A3 | qed | 4 | 1.0 | `c23_guided_L4_lam1_qed` |
| deployed | D1/D2/D3 | each anchor | 12 | 1.0 | `pilot_50k_p2_guided_{prop}` |

A1 is **priority 1**: it is the arm that carried C23's Rule B and the arm C25 measured the
0.0916 span on. The deployed family is not decoration — **if head-seed variance is just as
large at probe point 12, the finding is about learned probes in general, not about
mid-network probes**, and the section must say so.

Head seeds 2345 and 3456 on the mid arms already exist as `outputs/c25_hs{seed}_*` and are
**reused, not regenerated** (subject to G1). Head seed 1234 on all six arms already exists as
the published C23 / deployed runs and is reused subject to G1.

**Compute-matched best-of-N** (`scripts/06_best_of_n.py`, `--accounting actual`, N re-solved
from each arm's own realised tokens) and **quality** (`scripts/10_quality_analysis.py`) are run
for every mid-arm cell. They are **not** run for the deployed family or for the lambda envelope:
those two exist to measure hit-rate variance and to price Rule A against the deployed curve,
and neither uses a best-of-N comparator. This is a budget decision, taken here, before any
result.

**The lambda envelope (priority 3).** The **deployed** head (probe point 12, head seed 1234,
default code path — no `--layer`, no `--head-file`, exactly as the published deployed lambda runs
were produced) at lambda in **{1.25, 1.5, 2.5}**, on all three anchors. The existing deployed grid
is lambda in {0.25, 0.5, 1, 2, 4, 8} with throughout hit rates:

| property | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| aromatic_rings | 0.2359 | 0.3011 | 0.4735 | 0.5579 | 0.4989 | 0.3993 |
| hbd_count | 0.1324 | 0.1813 | 0.2988 | 0.4303 | 0.3730 | 0.3104 |
| qed | 0.1332 | 0.1692 | 0.1908 | 0.2361 | 0.2607 | 0.2438 |

The three new points are chosen so that every C23 arm's effective lambda is bracketed by
*adjacent* measured points rather than across an octave. They are fixed here and are not
re-chosen after seeing a correction.

## C29.0.3 The effective-lambda correction, defined before it is computed

For a C23 arm at probe point L with guidance strength lambda, let

    r(prop, L) = spread(prop, L) / spread(prop, 12)

using `mean_head_q_spread_across_candidates` from
`outputs/c17_layer_steering/layer_steering_metrics.json` verbatim (the table in §C29.0.1;
these are C17's numbers, transcribed, not re-derived). The **effective lambda** of the arm is

    lambda_eff = lambda * r(prop, L)

The deployed comparator is the deployed throughout hit rate interpolated **linearly in
log2(lambda)** between the two bracketing measured deployed points — the same axis
`scripts/21_summarise_c26.py` uses. The **effective-lambda-corrected advantage** is

    adv_eff = (mid arm hit rate) - (deployed hit rate interpolated at lambda_eff)

against the **raw advantage** `adv_raw = (mid arm hit rate) - (deployed hit rate at lambda)`.
Both are reported for every arm, at head seed 1234 and averaged over head seeds where the
deployed family has matching seeds.

Two versions are reported and labelled:

* **envelope-coarse** — interpolation on the pre-existing lambda grid {0.25, 0.5, 1, 2, 4, 8} only.
  Computable from artefacts that already exist, so it survives a kill.
* **envelope-fine** — interpolation on {0.25, 0.5, 1, 1.25, 1.5, 2, 2.5, 4, 8} once the new points land.

**What this correction is and is not, stated now.** The lambda-rescale identity is exact only
if `log q_mid = r * log q_deployed + const` pointwise. That is false: they are different
functions of different hidden states. `r` is a *scalar moment ratio* — a ratio of mean
across-candidate spreads. The correction is therefore a **first-order scalar-moment control,
not an identity**, and any claim that survives it is scoped accordingly. This limitation is
written here so it cannot be discovered later and presented as a caveat.

## C29.0.4 Uncertainty policy

The unit of replication for C29 is the **head seed**. Each arm cell is one guided run over
three generation seeds; its hit rate is the mean over those three, exactly as every previous
section computed it.

1. **Head-seed sd.** `sd` with `ddof=1` over the n head-seed hit rates. Its two-sided 95% CI is
   the chi-square interval `[s*sqrt((n-1)/chi2_{0.975,n-1}), s*sqrt((n-1)/chi2_{0.025,n-1})]`.
   This assumes normality of the per-head-seed hit rate; the assumption is stated, not tested
   at n = 8, and the per-seed values are published so a reader can judge it.
2. **Generation-seed sd.** Within each head seed, the sd (ddof=1) of that cell's three
   generation-seed hit rates. The arm-level estimate **pools** these across head seeds
   (`sqrt(mean of the within-cell variances)`), giving `n*(3-1)` degrees of freedom instead of
   2 — this is the main reason the C29 ratio is more trustworthy than C25's, and it is
   registered before the numbers exist.
3. **The variance ratio, with its own uncertainty.** The point estimate is
   `sd_head / sd_gen_pooled`. Its 95% CI comes from the F distribution on
   `(n-1, 2n)` degrees of freedom: `[ (s1/s2)/sqrt(F_{0.975,n-1,2n}), (s1/s2)/sqrt(F_{0.025,n-1,2n}) ]`.
   **The ratio-of-sds interval is the headline uncertainty C25 did not have.**
4. **Span comparability.** C25 compared a span of 3 against a span of 3. A span of 8 is not
   comparable to a span of 3: for normal data `E[span_n] = d2(n)*sigma` with
   d2(3) = 1.69257, d2(8) = 2.84720. Raw spans are reported, **and** each span is divided by
   its own `d2(n)` to give a sigma estimate on a common scale, and the C25 25x figure is
   restated on that scale. Any comparison of raw spans at different n is flagged as invalid.
5. **Seed-level t intervals.** For any mean over head seeds, a Student t interval on `n-1`
   df is reported (t_{0.975,7} = 2.3646242515927844 at n = 8; t_{0.975,5} = 2.5705818366147395 at
   n = 6). For means over the three generation seeds the existing 2-df interval
   (t_{0.975,2} = 4.302653) is used, imported from `scripts/18_summarise_c23.py`.
6. **Bootstrap.** **No three-seed percentile bootstrap is computed anywhere in C29.** At n = 3
   the percentile bootstrap of a mean is identically [min, max] — P(all three resamples hit the
   minimum) = 1/27 = 0.037 > 0.025 — so it conveys only a sign test at null probability 0.25.
   At n = 8 the same event has probability 8^-8 = 5.96e-8, three standard analyses away from
   binding, so a **percentile bootstrap over head seeds (10000 resamples, rng seed 20260801) is
   computed for head-seed means at n >= 6 and reported *alongside* the seed-level t interval,
   never instead of it**. Where the two disagree the section reports both and prefers neither.
   The molecule-level seed-stratified bootstrap of `scripts/18_summarise_c23.py` is imported
   unchanged where a C29 number is compared against a C23 number, so the estimator is shared
   rather than re-implemented.
7. **Multiplicity.** Six arm families are measured. The pre-registered rules below are scored
   at nominal alpha = 0.05 and **every** arm is reported, so there is no selection to correct;
   where a rule aggregates over anchors the required count is fixed here.

## C29.0.5 Validity gates, checked and reported as numbers before any result is read

**G1 — head-seed-1234 checkpoint identity (tensor, not bytes).** For every (property, probe
point) in the design, the C29-trained head at seed 1234 is compared **parameter tensor by
parameter tensor** against the published checkpoint it must reproduce
(`outputs/c17_probe_layers/head_{prop}_frozen_state_L{L}.pt` for the mid points, and
`outputs/c25_pooled_heads/head_{prop}_last1_L{L}_seed1234.pt` for every point). The residual
is `max |a - b|` over all parameters, published as a number. Expected `0.0`.
**File SHA-256 is deliberately not used** — see the header. If the residual is non-zero, seed
1234 must be regenerated end to end rather than reused, and §C29.6 says so.

**G2 — end-to-end pipeline identity at head seed 1234.** Two guided runs are executed with the
C29 checkpoints and compared molecule-by-molecule and hit-rate-by-hit-rate against the
published runs, using `scripts/18_summarise_c23.py::score_gate`'s comparison logic:

* `qed` probe point 4, lambda = 1, against `c23_guided_L4_lam1_qed` (a C23 mid arm);
* `qed` probe point 12, lambda = 1, against `pilot_50k_p2_guided_qed` (the deployed arm).

`qed` is chosen because it is the cheapest anchor (about 110 s per guided run in C25's log)
and the gate must land early. Residual = max absolute hit-rate difference over
{unguided, throughout} x {101, 202, 303}, plus a boolean molecule-identity check and the count
of molecules compared. Expected residual `0.0` and identical SMILES.
**If G2 fails the C29 arms are not comparable to C23's and the section reports that as its
headline**, rather than proceeding.

**G3 — head-seed replication of C25.** The C29-trained heads at seeds 2345 and 3456 must equal
C25's, by the same tensor comparison as G1, since C25's `c25_hs*` guided runs are reused.
Residual published. Expected `0.0`.

**G4 — the seed-1234 cross-route AUROC identity.** The C29 `last1` cells' per-head-seed test
AUROC and NLL must reproduce `outputs/c25_pooled_heads/cell_*_last1.json`'s `per_seed` entries
for the three shared seeds, to a residual reported as a number. Expected `0.0`. This is the
`--head-seeds` list-independence check: training seed 1234 in a list of 8 must give what
training it in a list of 3 gave.

**G5 — lambda envelope provenance.** Each new deployed lambda run must record
`head_checkpoint: head_{prop}_frozen_state.pt`, `layer: -1`, `layer_source: default (-1)` and
`lambda_source: cli --lam` in its `guidance_metrics.json`, i.e. the published deployed code
path with only lambda changed. Reported as the recorded values, not as a boolean.

**G6 — unguided invariance.** `unguided` cannot depend on the head seed, the probe point or
lambda. The spread of the `unguided` throughout-condition hit rate across every C29 run of a
given anchor is published as a number. It is a bug alarm, not a finding; a non-zero value
means seeding is not what the project claims.

## C29.0.6 Decision rules

**R1 — the head-seed sd is real.** On A1 (`hbd_count` pp4 lambda=2), the lower bound of the 95%
chi-square CI on `sd_head` exceeds the pooled generation-seed sd `sd_gen`. Fires / does not
fire, reported with both numbers.

**R2 — the variance ratio excludes 1.** On A1, the 95% F-based CI on `sd_head / sd_gen` excludes
1. Additionally reported, and pre-specified as *descriptive rather than a test*: whether C25's
25x figure lies inside that interval. **A ratio CI that contains 25 and a ratio CI that
excludes 25 are both publishable**; what is not acceptable is quoting 25 without an interval.

**R3 — is this about mid-network probes, or about probes?** Let `q = sd_head(mid arm) /
sd_head(deployed arm at the same anchor)`. If `q` lies in **[0.5, 2]** for at least 2 of the 3
anchors, C29 concludes the head-seed variance finding is about **learned probes in general**,
and every guidance result in the project — including the deployed headline — inherits it. If
`q > 2` on at least 2 anchors, the finding is **specific to mid-network probes** and the
deployed results are comparatively stable. The interval [0.5, 2] is fixed here.

**R4 — does Rule A survive head-seed variation?** For each anchor, the head-seed-**paired**
difference `d_h = (mid arm at head seed h) - (deployed arm at head seed h)` is averaged over
the head seeds present in both families, with a t interval on `n-1` df and the head-seed
bootstrap of §C29.0.4.6. Rule A survives on an anchor iff that interval is strictly positive.
Rule A survives overall iff it survives on **at least 2 of 3** anchors — the same 2-of-3
threshold C23's own Rule A used, transcribed rather than chosen. Pairing by head seed is a
strictly stronger design than C23 had and is registered here as the primary estimator; the
unpaired difference of means is reported alongside.

**R5 — does Rule A survive the effective-lambda correction?** As R4, but with the deployed side
replaced by the deployed envelope interpolated at `lambda_eff` (§C29.0.3), envelope-fine where
available and envelope-coarse otherwise. Rule A survives the correction on an anchor iff the
head-seed mean of `adv_eff` is positive **and** its t interval excludes 0. Overall threshold:
at least 2 of 3 anchors. **Reported for every arm whatever the sign, raw and corrected, with
the corrected-minus-raw difference stated as the size of the confound.**

**R6 — Rule B at n >= 6.** On A1, the head-seed mean of `(arm hit rate - its own compute-matched
best-of-N hit rate)` with a t interval on `n-1` df, and the count of head seeds on which the
advantage is positive. C23's Rule B fires iff that interval is strictly positive. C25 already
demoted Rule B on 3 seeds (advantages +0.0760 / +0.0366 / -0.0156); R6 either confirms the
demotion at n >= 6 or overturns it.

**R7 — null.** If none of R1, R2, R4, R5, R6 fires, C29 reports that plainly. A null on R1/R2
would mean head-seed variance is *not* demonstrably larger than generation-seed variance at
n = 8, which would **restore** C23's Rule B and is a result C29 must be willing to report.

## C29.0.7 Predictions, to be scored verbatim including failures

1. `sd_head` on A1 lands in **[0.030, 0.080]**. Reasoning: C25's 3-seed span 0.09160960684359803
   divided by d2(3) = 1.69257 gives sigma-hat = 0.0541.
2. `sd_gen` pooled on A1 lands **below 0.010**. Reasoning: C25 reported a generation-seed span of
   0.0037 on this arm; 0.0037/1.69257 = 0.0022.
3. The point estimate of `sd_head / sd_gen` on A1 is **above 5** but its 95% CI **contains
   values below 25**, i.e. the headline 25x shrinks. This is the prediction the whole of
   priority 1 exists to test and I expect the ratio to remain large but the "25" to lose its
   third significant figure.
4. **R3 concludes "probes in general"**: the deployed arm's head-seed sd is within a factor of
   2 of the mid arm's on at least 2 anchors. Reasoning: nothing about probe point 12 makes a
   randomly initialised 768->256->256->n_bins MLP more reproducible, and
   `head_metrics.json`'s across-seed AUROC sd at probe point 12 (0.0041 on hbd_count) is *larger*
   than C25's mid-layer AUROC sd (0.0016 at pp4). If this prediction holds, the project's
   transferable finding gets stronger, not weaker.
5. **Rule A survives in sign but not in significance.** I predict the head-seed-paired mean
   advantage is positive on all 3 anchors but its t interval excludes 0 on **at most 1**, so R4
   does not fire at the 2-of-3 threshold.
6. **The effective-lambda correction removes roughly half of the lambda=1 headline.** Reviewer
   arithmetic against the existing envelope gives aromatic_rings L3 +0.0963 -> +0.0459,
   aromatic_rings L6 +0.1096 -> +0.0592, hbd_count L4 lambda=1 +0.0701 -> +0.0245. I predict
   envelope-fine lands within **+-0.02** of each of those three coarse figures, and that the
   sign survives on all three while the magnitude does not.
7. **R6 does not fire**: the head-seed mean advantage over compute-matched best-of-N on A1 is
   positive but its t interval contains 0. Reasoning: C25's three values (+0.0760 / +0.0366 /
   -0.0156) have mean 0.0323 and sd 0.0460; five more seeds from the same distribution would
   give a t half-width of about 0.038.
8. `qed` shows the **smallest** head-seed sd of the three mid anchors (C25: 0.0137 against
   0.0387 and 0.0528) and `hbd_count` the largest.

Prediction 5 is the one I expect to be least reliable: at n = 8 a paired design can be much
more powerful than the unpaired n = 3 comparison C23 ran, and the pairing removes the deployed
arm's own head-seed noise from the difference.

## C29.0.8 The attack this design invites, stated before the result

1. *You changed the head-training route.* The mid-layer C23 checkpoints came from
   `scripts/16_probe_layer_sweep.py`; C29's come from `scripts/20_pooled_sweep.py --variants last1`.
   G1 and G4 exist precisely to make that a measured identity rather than an assumption, and if
   the residual is non-zero C29 says so and regenerates rather than reusing.
2. *Your effective-lambda correction is not the identity you cite.* Correct, and §C29.0.3 says
   so before any number is produced. `r` is a scalar moment ratio, the identity is pointwise,
   and the correction is first-order.
3. *You are estimating a variance from 8 points.* Yes. That is why the sd is published with a
   chi-square interval and the ratio with an F interval, and why no raw span at n = 8 is
   compared to a raw span at n = 3 without the d2(n) rescaling.
4. *Head seeds and generation seeds are not exchangeable units.* Also true: a head seed
   re-randomises an MLP initialisation and its minibatch order, a generation seed re-randomises
   sampling. C29 does not claim they are the same kind of noise. It claims — and this is the
   whole point — that **the protocol the literature uses reports only the second**.
5. *Priority 4 (C27 across head seeds) may not be run.* If it is not, the section says so and
   claims nothing about it. C27's equal-information result rests on one head seed and two of its
   three effects (-0.0439 / -0.0292 / -0.0522) are smaller than C25's 0.0916 head-seed span; that
   remains an open weakness whether or not C29 gets to it.

## C29.0.9 What is not done

No fine-tuning, LoRA, RL or activation edit of the generator. No second generator. No second
DAgger round. No re-derivation of windows or target intervals. No modification of any existing
`outputs/` directory, any existing script, `reports/pilot_report.md` or `README.md`. No
`git add` and no `git commit`. Conflicts with existing sections are **listed for the owner to
merge**, not edited in place.
<!-- END VERBATIM PREREG COPY -->

---

## C29.1 What was run

Eight head seeds — 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901 — on seven families, three
generation seeds each, 512 molecules per condition per seed. Every arm reached the
pre-registered n = 8; nothing was truncated.

| family | property | probe point | λ | n | new C29 runs |
|---|---|---|---|---|---|
| `A1` | hbd_count | 4 | 2 | 8 | 5 (1234 is C23's, 2345/3456 are C25's) |
| `A2` | aromatic_rings | 3 | 1 | 8 | 5 |
| `A3` | qed | 4 | 1 | 8 | 5 |
| `D_aromatic_rings` | aromatic_rings | 12 | 1 | 8 | 7 (1234 is the published deployed run) |
| `D_hbd_count` | hbd_count | 12 | 1 | 8 | 7 |
| `D_qed` | qed | 12 | 1 | 8 | 7 |
| `D2_hbd_count` (post-hoc) | hbd_count | 12 | 2 | 8 | 7 |

Plus the deployed λ envelope at 1.25 / 1.5 / 2.5 on all three anchors (9 runs), two
end-to-end gate runs, and 48 `last1` head checkpoints trained by the unmodified
`scripts/20_pooled_sweep.py`. 85 cells completed. One cell (`c29_deplam1p25_guided_qed`) hit
a CUDA OOM against the concurrent C28 experiment, which at that moment had five
generation processes on one 24 GB card; the envelope stage stopped, a retry-with-backoff
policy was added to the C29 driver, and the stage was relaunched and completed. A CUDA OOM
is a scheduling accident, not a result — every cell is deterministic in its seeds, so the
relaunch changes no measured quantity — but it is recorded here and in
`outputs/c29_progress.jsonl` rather than swept up. Progress is logged cell by cell in that
file.

`D2_hbd_count` is **post-hoc** and labelled as such everywhere it appears; §C29.4 explains
why it had to exist.

---

## C29.2 Validity gates, checked before any result was read

### C29.2.1 G1/G3 — checkpoint identity, tensor by tensor

**33 comparisons, maximum absolute parameter residual `0.0`, every binner identical.**

The C29 head at head seed 1234 is compared against `c17_probe_layers`'s mid-layer
checkpoints, `c25_pooled_heads`'s `last1` checkpoints, and — the strongest available
reference — `pilot_50k_heads_p2/head_{prop}_frozen_state_seed{s}.pt`, the checkpoint the
published deployed λ=1 runs actually steered with. Seeds 2345 and 3456 are compared the same
way, because C25's `c25_hs*` runs are reused rather than regenerated.

This is deliberately a **tensor** comparison. C27's gate 4 failed as pre-registered because
it compared checkpoints by file SHA-256, and `torch.save` names the zip archive after the
output path, so identical tensors give different bytes. C29 took that lesson into its own
pre-registration (§C29.0.5) rather than rediscovering it.

Consequence: head seed 1234 on all six λ-matched families and head seeds 2345/3456 on the
three mid families **are** the published runs, not replications of them, so the reuse in
§C29.1 is an identity.

### C29.2.2 G2 — end-to-end identity at head seed 1234

**Two runs, 6144 molecules compared, maximum absolute hit-rate residual `0.0`, every SMILES
string identical.**

| replay | reference | what it checks |
|---|---|---|
| `c29_gate_L4_lam1_qed_hs1234` | `c23_guided_L4_lam1_qed` | the `--layer` / `--head-file` mid-layer path |
| `c29_gate_L12_lam1_qed_hs1234` | `pilot_50k_p2_guided_qed` | the deployed path, which C29 drives through `--layer 12` |

qed was chosen in advance as the cheapest anchor so the gate would land early; it did, before
any priority-1 arm finished.

### C29.2.3 G4 — training does not depend on the seed list

**18 comparisons, maximum residual `0.0`.** Training seed 1234 inside a list of eight gives
bit-identical per-seed test AUROC and NLL to training it inside C25's list of three.

One honest detail: `scripts/20_pooled_sweep.py`'s **own** internal gate reports a maximum
residual of 0.002423 for the C29 run. That gate compares an *8-seed mean* AUROC against a
*3-seed mean* reference and therefore cannot pass at n = 8 by construction. It is not C29's
gate; G4 is the per-seed version and it is exact.

### C29.2.4 G5 — the envelope runs are the deployed code path

**9 runs, maximum λ residual `0.0`.** Every new envelope run records `layer: -1`,
`layer_source: "default (-1)"`, `head_file_source: "default"`,
`head_checkpoint: "head_{prop}_frozen_state.pt"` and `lambda_source: "cli --lam"` — i.e. the
published deployed configuration with only λ changed.

### C29.2.5 G6 — the unguided condition is invariant

**Maximum span `0.0` across every C29 run of each anchor.** Over 24 hbd_count runs, 16
aromatic_rings runs and 16 qed runs the unguided hit rate is exactly 0.0837, 0.1785 and
0.0896 respectively. A non-zero value here would mean the base policy is being disturbed by
something it must not see.

---

## C29.3 The head-seed distribution

### C29.3.1 Hit rate by head seed

Each cell is one guided run over generation seeds 101 / 202 / 303, 512 molecules per
condition per seed; the entry is that run's mean `throughout` hit rate.

| arm | 1234 | 2345 | 3456 | 4567 | 5678 | 6789 | 7890 | 8901 | mean |
|---|---|---|---|---|---|---|---|---|---|
| `A1` hbd_count pp 4, lam 2  (the Rule B arm) | 0.5603 | 0.5601 | 0.4687 | 0.5185 | 0.5747 | 0.5709 | 0.5605 | 0.5705 | 0.5480 |
| `A2` aromatic_rings pp 3, lam 1 | 0.5698 | 0.6101 | 0.6472 | 0.5812 | 0.5949 | 0.5543 | 0.6089 | 0.5983 | 0.5956 |
| `A3` qed pp 4, lam 1 | 0.2288 | 0.2422 | 0.2147 | 0.2164 | 0.2175 | 0.2327 | 0.2253 | 0.2445 | 0.2278 |
| `D_aromatic_rings` deployed pp 12, lam 1, aromatic_rings | 0.4735 | 0.4814 | 0.4589 | 0.4350 | 0.4741 | 0.4944 | 0.4696 | 0.4615 | 0.4685 |
| `D_hbd_count` deployed pp 12, lam 1, hbd_count | 0.2988 | 0.3462 | 0.3686 | 0.3252 | 0.3207 | 0.3091 | 0.3501 | 0.3100 | 0.3286 |
| `D_qed` deployed pp 12, lam 1, qed | 0.1908 | 0.1998 | 0.1837 | 0.2009 | 0.2128 | 0.2277 | 0.2050 | 0.1889 | 0.2012 |
| `D2_hbd_count` deployed pp 12, lam 2, hbd_count (post-hoc) | 0.4303 | 0.4665 | 0.4629 | 0.4826 | 0.4253 | 0.4600 | 0.4521 | 0.4513 | 0.4539 |

### C29.3.2 Variance decomposition — the correction to C25

| arm | n | sd_head | 95% chi-sq CI | span | span/d2(8) | sd_gen pooled (16 df) | sd_head/sd_gen | 95% F CI |
|---|---|---|---|---|---|---|---|---|
| `A1` | 8 | 0.0366 | [0.0242, 0.0744] | 0.1059 | 0.0372 | 0.0213 | 1.71 | [0.96, 3.65] |
| `A2` | 8 | 0.0284 | [0.0188, 0.0577] | 0.0929 | 0.0326 | 0.0205 | 1.38 | [0.77, 2.95] |
| `A3` | 8 | 0.0115 | [0.0076, 0.0234] | 0.0298 | 0.0105 | 0.0180 | 0.64 | [0.36, 1.36] |
| `D_aromatic_rings` | 8 | 0.0176 | [0.0116, 0.0358] | 0.0595 | 0.0209 | 0.0277 | 0.63 | [0.35, 1.35] |
| `D_hbd_count` | 8 | 0.0241 | [0.0159, 0.0490] | 0.0699 | 0.0245 | 0.0204 | 1.18 | [0.66, 2.51] |
| `D_qed` | 8 | 0.0142 | [0.0094, 0.0289] | 0.0439 | 0.0154 | 0.0174 | 0.82 | [0.46, 1.75] |
| `D2_hbd_count` | 8 | 0.0189 | [0.0125, 0.0384] | 0.0573 | 0.0201 | 0.0166 | 1.13 | [0.63, 2.42] |

**Read the `sd_gen pooled` column against the `sd_head` column.** They are the same order of
magnitude everywhere, and on three of the seven families the *generation* seed is the larger
source. C25's factor of 23–25 does not survive contact with more than one head seed's worth
of generation-seed data.

Where the 25× came from, exactly: C25 reported the head-seed span (0.0916) against **head
seed 3456's** generation-seed span (0.0040). It is the smallest of the three, by a factor of
17 against head seed 1234's 0.0672, and at n = 8 it is still the smallest of eight. The
per-cell generation-seed sds are:

| arm | 1234 | 2345 | 3456 | 4567 | 5678 | 6789 | 7890 | 8901 |
|---|---|---|---|---|---|---|---|---|
| `A1` | 0.0342 | 0.0210 | 0.0022 | 0.0075 | 0.0120 | 0.0361 | 0.0217 | 0.0069 |
| `A2` | 0.0253 | 0.0098 | 0.0089 | 0.0390 | 0.0141 | 0.0214 | 0.0035 | 0.0187 |
| `A3` | 0.0098 | 0.0220 | 0.0151 | 0.0147 | 0.0255 | 0.0152 | 0.0067 | 0.0254 |
| `D_aromatic_rings` | 0.0195 | 0.0088 | 0.0368 | 0.0288 | 0.0373 | 0.0235 | 0.0042 | 0.0392 |
| `D_hbd_count` | 0.0076 | 0.0190 | 0.0151 | 0.0382 | 0.0194 | 0.0149 | 0.0223 | 0.0116 |
| `D_qed` | 0.0077 | 0.0148 | 0.0149 | 0.0205 | 0.0316 | 0.0045 | 0.0211 | 0.0049 |
| `D2_hbd_count` | 0.0321 | 0.0118 | 0.0135 | 0.0089 | 0.0132 | 0.0129 | 0.0196 | 0.0078 |

A span is not comparable across different n either: `E[span_n] = d2(n)·σ`, with
d2(3) = 1.69257 and d2(8) = 2.84720. On a common σ scale C25's head-seed span of 0.0916
implies σ = 0.0541 and C29's span of 0.1059 implies σ = 0.0372 — so the *span* grew, as it
must with more seeds, while the *sd* it implies fell. Any comparison of raw spans at
different n in this project is invalid, and C29 reports both columns so that cannot be
repeated.

---

## C29.4 R4 — does Rule A survive head-seed variation?

| arm | comparator | n | lambda matched | mean | 95% t CI | 95% head-seed bootstrap |
|---|---|---|---|---|---|---|
| `A1` | `D2_hbd_count` | 8 | True | 0.0941 | [0.0534, 0.1349] | [0.0610, 0.1230] |
| `A2` | `D_aromatic_rings` | 8 | True | 0.1270 | [0.0957, 0.1584] | [0.1021, 0.1504] |
| `A3` | `D_qed` | 8 | True | 0.0266 | [0.0113, 0.0419] | [0.0147, 0.0385] |
| `A1_as_preregistered` | `D_hbd_count` | 8 | False | 0.2194 | [0.1731, 0.2658] | [0.1804, 0.2509] |

**The pre-registration defect, reported rather than amended.** §C29.0.6 R4 asks for the
paired difference *at matched λ*, but §C29.0.2 fixed the deployed family at λ = 1 while A1
sits at λ = 2. R4 as written is therefore **not scoreable on A1**. This is a defect in the
pre-registration, found while scoring it, and it is handled the way C27 handled its gate-4
failure: the defect is stated, the mis-specified comparison is still reported
(`A1_as_preregistered`, +0.2194, against a deployed arm at half the steering strength, which
is meaningless as a depth comparison), and the *measurement* is repaired by adding
`D2_hbd_count` — the deployed head at λ = 2 across the same eight head seeds. That family is
post-hoc and is labelled post-hoc in the artefact, in the driver and here.

**R4 fires, 3 of 3.** Pairing by head seed is a materially stronger design than C23 had: it
removes the deployed arm's own head-seed noise from the difference, which is why the
intervals are tight despite sd_head ≈ 0.02–0.04 on both sides.

---

## C29.5 R5 — the effective-λ control

The rescale factor is C17's `mean_head_q_spread_across_candidates`, transcribed from
`outputs/c17_layer_steering/layer_steering_metrics.json` and never re-derived:
r = 1.5120 (aromatic_rings pp 3), 1.5122 (pp 6), 1.2721 (hbd_count pp 4), 1.2599 (pp 6),
1.2437 (qed pp 4). λ_eff = λ·r, and the deployed comparator is interpolated linearly in
log₂λ between bracketing measured points.

### C29.5.1 The deployed λ envelope, with C29's three new points

| property | 0.25 | 0.5 | 1 | 1.25 | 1.5 | 2 | 2.5 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| aromatic_rings | 0.2359 | 0.3011 | 0.4735 | 0.5134 | 0.5316 | 0.5579 | 0.5601 | 0.4989 | 0.3993 |
| hbd_count | 0.1324 | 0.1813 | 0.2988 | 0.3437 | 0.3802 | 0.4303 | 0.4435 | 0.3730 | 0.3104 |
| qed | 0.1332 | 0.1692 | 0.1908 | 0.1868 | 0.2039 | 0.2361 | 0.2843 | 0.2607 | 0.2438 |

The three new points are the middle of the table. They matter: on hbd_count the λ=1→2 octave
that the coarse envelope had to interpolate across turns out to be strongly convex
(0.2988 → 0.3437 → 0.3802 → 0.4303), so the coarse interpolation *understates* the deployed
comparator and flatters the mid-layer arm. On qed the envelope is **not monotone** — λ=1.25
gives 0.1868 against λ=1's 0.1908 — which is why qed's correction moves the advantage the
"wrong" way in §C29.5.3.

### C29.5.2 All fifteen C23 arms, at head seed 1234

| property | pp | lambda | r | lambda_eff | mid | deployed@lambda | raw | corrected (coarse) | corrected (fine) | confound share (fine) |
|---|---|---|---|---|---|---|---|---|---|---|
| aromatic_rings | 3 | 0.5 | 1.5120 | 0.7560 | 0.3616 | 0.3011 | 0.0605 | -0.0423 | -0.0423 | 1.70 |
| aromatic_rings | 3 | 1 | 1.5120 | 1.5120 | 0.5698 | 0.4735 | 0.0964 | 0.0460 | 0.0375 | 0.61 |
| aromatic_rings | 3 | 2 | 1.5120 | 3.0240 | 0.8159 | 0.5579 | 0.2579 | 0.2931 | 0.2805 | -0.09 |
| aromatic_rings | 6 | 0.5 | 1.5122 | 0.7561 | 0.3881 | 0.3011 | 0.0870 | -0.0158 | -0.0158 | 1.18 |
| aromatic_rings | 6 | 1 | 1.5122 | 1.5122 | 0.5831 | 0.4735 | 0.1096 | 0.0592 | 0.0507 | 0.54 |
| aromatic_rings | 6 | 2 | 1.5122 | 3.0245 | 0.6842 | 0.5579 | 0.1262 | 0.1614 | 0.1488 | -0.18 |
| hbd_count | 4 | 0.5 | 1.2721 | 0.6360 | 0.2319 | 0.1813 | 0.0506 | 0.0098 | 0.0098 | 0.81 |
| hbd_count | 4 | 1 | 1.2721 | 1.2721 | 0.3689 | 0.2988 | 0.0701 | 0.0245 | 0.0217 | 0.69 |
| hbd_count | 4 | 2 | 1.2721 | 2.5441 | 0.5603 | 0.4303 | 0.1300 | 0.1499 | 0.1195 | 0.08 |
| hbd_count | 6 | 0.5 | 1.2599 | 0.6300 | 0.2027 | 0.1813 | 0.0214 | -0.0177 | -0.0177 | 1.83 |
| hbd_count | 6 | 1 | 1.2599 | 1.2599 | 0.3395 | 0.2988 | 0.0407 | -0.0032 | -0.0058 | 1.14 |
| hbd_count | 6 | 2 | 1.2599 | 2.5199 | 0.4684 | 0.4303 | 0.0381 | 0.0572 | 0.0261 | 0.31 |
| qed | 4 | 1 | 1.2437 | 1.2437 | 0.2288 | 0.1908 | 0.0380 | 0.0237 | 0.0419 | -0.10 |
| qed | 4 | 2 | 1.2437 | 2.4873 | 0.2796 | 0.2361 | 0.0435 | 0.0357 | -0.0036 | 1.08 |
| qed | 4 | 4 | 1.2437 | 4.9747 | 0.3092 | 0.2607 | 0.0485 | 0.0538 | 0.0538 | -0.11 |

**The reviewer's arithmetic is confirmed exactly on the coarse envelope**: +0.0460 against
+0.0459, +0.0592 against +0.0592, +0.0245 against +0.0245. On the fine envelope the same
three arms are +0.0375, +0.0507 and +0.0217 — the correction is slightly *larger* than the
reviewer estimated, not smaller.

**Five arms change sign** under the fine correction: aromatic_rings pp 3 λ=0.5
(+0.0605 → -0.0423), aromatic_rings pp 6 λ=0.5 (+0.0870 → -0.0158), hbd_count pp 6 λ=0.5
(+0.0214 → -0.0177), hbd_count pp 6 λ=1 (+0.0407 → -0.0058) and qed pp 4 λ=2
(+0.0435 → -0.0036). C23's Rule A was reported as 15 of 15 arms positive; at matched
effective λ it is **10 of 15**.

Four arms have a *negative* confound share, i.e. the correction helps them: aromatic_rings
pp 3 λ=2, aromatic_rings pp 6 λ=2, qed pp 4 λ=1 and qed pp 4 λ=4. That is not a bug — those arms
sit at an effective λ past the deployed envelope's peak (aromatic_rings peaks near λ=2.5 at
0.5601, then falls), so a stronger-λ comparator is a *weaker* comparator there. This is a
real limitation of the control and is stated as one in §C29.9.

### C29.5.3 The correction at n = 8 head seeds

| arm | r | lambda_eff | deployed@lambda | deployed@lambda_eff | raw mean | raw 95% t CI | corrected mean | corrected 95% t CI |
|---|---|---|---|---|---|---|---|---|
| `A1` | 1.2721 | 2.5441 | 0.4303 | 0.4408 | 0.1177 | [0.0872, 0.1483] | 0.1072 | [0.0766, 0.1378] |
| `A2` | 1.5120 | 1.5120 | 0.4735 | 0.5323 | 0.1221 | [0.0984, 0.1458] | 0.0632 | [0.0395, 0.0869] |
| `A3` | 1.2437 | 1.2437 | 0.1908 | 0.1869 | 0.0369 | [0.0273, 0.0466] | 0.0409 | [0.0313, 0.0505] |

**R5 fires, 3 of 3.** Note what varies here: the mid arm has eight head seeds, but the
deployed envelope exists only at head seed 1234, so these intervals carry the mid arm's
head-seed variance and not the deployed arm's. That is stated in the artefact and is not
hidden. The paired, matched-λ, both-sides-replicated comparison is §C29.4's, and it agrees.

Sizes of the confound at n = 8: aromatic_rings loses **48%** of its raw advantage
(+0.1221 → +0.0632), hbd_count loses **9%** (+0.1177 → +0.1072, because r is only 1.2721 and
the λ=2→2.5 step is small), and qed *gains* (+0.0369 → +0.0409) because its envelope is
non-monotone at λ=1.25.

---

## C29.6 R6 — Rule B at n = 8

Three prices for the same eight guided runs. The first is C23's own comparator, which is
what R6 was pre-registered against; the other two are harder and are attached here so the
verdict cannot be quoted without them.

| comparator | mean | 95% t CI | 95% head-seed bootstrap | head seeds positive | fires |
|---|---|---|---|---|---|
| C23's own compute-matched best-of-N | 0.0490 | [0.0135, 0.0845] | [0.0199, 0.0750] | 6/8 | **yes** |
| token-conservative (best-of-N never underfunded) | 0.0246 | [-0.0060, 0.0552] | [-0.0009, 0.0443] | 6/8 | no |
| C26's corrected estimator at the same budget | 0.0156 | [-0.0173, 0.0485] | [-0.0116, 0.0380] | 6/8 | no |

Cell by cell:

| head seed | guided | matched N | best-of-N | advantage | token-conservative adv | vs C26 curve |
|---|---|---|---|---|---|---|
| 1234 | 0.5603 | 8 | 0.4844 | 0.0760 | 0.0369 | 0.0267 |
| 2345 | 0.5601 | 9 | 0.5234 | 0.0366 | 0.0366 | 0.0210 |
| 3456 | 0.4687 | 8 | 0.4844 | -0.0156 | -0.0547 | -0.0649 |
| 4567 | 0.5185 | 9 | 0.5234 | -0.0049 | -0.0049 | -0.0192 |
| 5678 | 0.5747 | 8 | 0.4844 | 0.0903 | 0.0512 | 0.0491 |
| 6789 | 0.5709 | 8 | 0.4844 | 0.0866 | 0.0475 | 0.0379 |
| 7890 | 0.5605 | 9 | 0.5234 | 0.0371 | 0.0371 | 0.0218 |
| 8901 | 0.5705 | 8 | 0.4844 | 0.0861 | 0.0471 | 0.0525 |

**This is the result that most changes the reading of C25.** C25 retired C23's Rule B on
three head seeds whose advantages were +0.0760 / +0.0366 / -0.0156. At eight seeds the mean
is +0.0490 with a t interval that excludes zero, and head seed 3456 — the one that produced
C25's sign flip — is the worst of the eight (-0.0156, and -0.0649 against C26's curve). A
three-seed retirement of a positive result was under-powered in exactly the way C29 was
commissioned to check.

**It does not follow that Rule B holds.** Both harder comparators contain zero:

- **token-conservative**: `scripts/06_best_of_n.py` floors the matched N, which for this arm
  takes N from 9 to 8 and leaves best-of-N spending fewer tokens per returned molecule than
  guidance. Re-priced against `c18_bestofn_N9_hbd_count`, which spends at least as many
  tokens, the advantage is +0.0246 [-0.0060, +0.0552].
- **C26's corrected estimator**: the 3.6× larger best-of-N estimator C26 adopted precisely
  because the slot estimator is optimistic in guidance's favour. +0.0156 [-0.0173, +0.0485].

The honest verdict is that Rule B is **unresolved at n = 8** and that the comparator choice,
not the head seed, now decides it.

---

## C29.7 The pre-registered rules and predictions, scored verbatim

### C29.7.1 Decision rules

| rule | verdict | numbers |
|---|---|---|
| **R1** — sd_head's chi-square lower bound exceeds sd_gen | **FIRES** | 0.0242 > 0.0213, sd_head 0.0366 |
| **R2** — the sd ratio's F interval excludes 1 | **does not fire** | 1.71, [0.96, 3.65]; C25's 24.76 is outside |
| **R3** — q in [0.5, 2] on ≥ 2 anchors → "probes in general" | **FIRES**, 3 of 3 | q = 1.52 / 1.61 / 0.81, table below |
| **R4** — paired mid minus deployed positive on ≥ 2 anchors | **FIRES**, 3 of 3 | but **not scoreable as written on A1**; §C29.4 |
| **R5** — effective-λ-corrected advantage positive on ≥ 2 anchors | **FIRES**, 3 of 3 | +0.1072 / +0.0632 / +0.0409 |
| **R6** — Rule B's advantage strictly above 0 | **FIRES** against C23's comparator only | +0.0490 [+0.0135, +0.0845]; fails both harder comparators |
| **R7** — null | **does not fire** | R1, R4, R5, R6 fired |

R3 in full — the comparison that says whether this is a fact about mid-network probes or
about learned probes:

| anchor | sd_head(mid) | sd_head(deployed) | q | 95% F CI | in [0.5, 2] |
|---|---|---|---|---|---|
| hbd_count | 0.0366 | 0.0241 | 1.52 | [0.68, 3.39] | True |
| aromatic_rings | 0.0284 | 0.0176 | 1.61 | [0.72, 3.61] | True |
| qed | 0.0115 | 0.0142 | 0.81 | [0.36, 1.81] | True |

R1 fires and R2 does not, and the difference between them is the whole point: R1 treats the
generation-seed sd as known, R2 propagates the uncertainty in both estimates. **R2 is the
honest one.** A section that quoted only R1 would be repeating C25's error in a more
sophisticated form.

### C29.7.2 Predictions

| # | prediction | verdict | measured |
|---|---|---|---|
| **P1** | sd_head on A1 in [0.030, 0.080] | **HOLDS** | 0.0366 |
| **P2** | pooled sd_gen on A1 below 0.010 | **FALSIFIED** | 0.0213, more than twice the bound. I inherited C25's 0.0037 without checking it was one cell's span. |
| **P3** | ratio above 5, CI containing values below 25 | **FALSIFIED** | ratio 1.71, CI [0.96, 3.65]. The CI does contain values below 25, but the ratio is nowhere near 5. Falsified on the first clause. |
| **P4** | R3 concludes "probes in general" | **HOLDS** | 3 of 3 anchors in band |
| **P5** | positive on all 3 anchors, significant on at most 1 | **FALSIFIED** | positive 3 of 3 **and** significant 3 of 3. I said this was the least reliable prediction; pairing by head seed was more powerful than I allowed for. |
| **P6** | fine correction within ±0.02 of the reviewer's three figures, sign survives | **HOLDS** | 3 of 3 within, 3 of 3 sign |
| **P7** | R6's interval contains 0 | **FALSIFIED** against C23's comparator | +0.0490 [+0.0135, +0.0845]. It contains 0 against both harder comparators, but the prediction named C23's. |
| **P8** | qed smallest head-seed sd, hbd_count largest | **HOLDS** | 0.0115 / 0.0284 / 0.0366 |

**Four of eight predictions falsified.** P2 and P3 are the interesting ones: I predicted the
25× would shrink but survive as "large". It did not survive at all. P5 and P7 are falsified
in the direction *favourable* to the project, which is the direction a pre-registration is
least able to protect against motivated reading — hence both are reported with their harder
comparators attached.

---

## C29.8 What C29 changes in the rest of the report

These are conflicts for the owner to merge, **not** edits C29 has made.
`reports/pilot_report.md`, `README.md`, `reports/section_c23_layer_end_to_end.md`,
`reports/section_c25_pooling.md`, `reports/section_c27_head_selected_bestofn.md`,
`docs/TODO.md` and every existing `outputs/` directory are untouched.

1. **§C25.4 and `docs/TODO.md`'s C25 entry must drop the "factor of 25 / factor of 23".**
   The correct statement is: head-seed sd 0.0366 [0.0242, 0.0744] against pooled
   generation-seed sd 0.0213, ratio 1.71 [0.96, 3.65]. C25's sentence is accurate as
   written — it does say "at head seed 3456" — but the number that travelled into the
   abstract, the TODO and the C29 brief is the cherry-picked one.
2. **The transferable finding needs restating, not retracting.** *"The standard protocol —
   one trained probe, error bars over generation seeds — cannot distinguish the method from
   its own noise"* is still supported: head-seed sd is comparable to or larger than
   generation-seed sd on 4 of 7 families, it is never reported anywhere in the literature,
   and at the deployed configuration it is 0.0142 to 0.0241 on a hit rate of 0.2012 to 0.4685.
   What is **not** supported is a multiplier.
3. **C25's retirement of C23's Rule B should be marked under-powered.** §C25.4.1 and the
   TODO's "C23's three seeds replicated the wrong thing" rest on n = 3 including the worst
   of eight seeds. C29 does not restore Rule B — two of three comparators contain zero — but
   the reason it is unresolved is the comparator, not the head seed.
4. **C23's Rule A needs an effective-λ column.** "15 of 15 arms positive" becomes 10 of 15
   at matched effective λ, and the three headline λ=1 arms lose 54% to 69% of their margin.
   The depth effect is real and survives at n = 8 head seeds (§C29.4), but it is roughly
   half the size §C23 reports.
5. **§19's λ grid should absorb C29's three new deployed points** (1.25 / 1.5 / 2.5, all
   three anchors, `outputs/c29_deplam*`). They show the aromatic_rings envelope peaks near
   λ = 2.5 at 0.5601 rather than at λ = 2, and that the qed envelope is **non-monotone**
   between λ = 1 and λ = 1.25 — neither is visible on the octave grid.
6. **C27's E4 must be reported as replicating on two anchors of three.** Re-run at head
   seeds 2345 and 3456, the deployed-vs-head-selected margin is -0.0439 / -0.0301 / -0.0408
   on aromatic_rings and -0.0522 / -0.0430 / -0.0602 on qed, but -0.0292 / +0.0238 / +0.0400
   on hbd_count — a sign flip, with a head-seed span of 0.0692 against a published effect of
   0.0292. C27's limitation 2 predicted exactly this. §C29.10.
7. **Nothing here touches C17, C18, C24 or C26's structural finding.** C29 re-measures
   variance and re-prices a comparator; it re-derives no window, no interval and no head
   recipe.

---

## C29.9 Limitations

1. **The effective-λ control is a scalar-moment control, not the λ-rescale identity.** The
   identity `λ·log(c·q^α) = (λα)·log q + λ·log c` is pointwise; `r` is a ratio of *mean*
   across-candidate spreads of two different functions of two different hidden states. The
   correction is first order. This was stated in §C29.0.3 before any number existed.
2. **The correction is not monotone in λ, and past the envelope peak it is anti-conservative.**
   Four arms get *better* under it because their effective λ lands beyond the deployed
   envelope's maximum. Pricing at a higher effective λ is only a penalty while the envelope
   is rising; C29 reports those arms rather than excluding them, but they should not be read
   as evidence that depth helps.
3. **`r` comes from C17's 400-prefix per-position probe, not from the end-to-end runs.** It
   was measured at λ = 1 on 267–400 sampled prefixes and is applied to arms at λ = 0.5, 1, 2
   and 4. Nothing checks that the spread ratio is λ-invariant.
4. **Eight head seeds is eight.** The chi-square interval on a sd at 7 df spans a factor of
   three ([0.0242, 0.0744] on A1), and the F interval on a ratio of sds spans a factor of
   nearly four. C29 buys a real interval; it does not buy a precise one.
5. **The R5 intervals are one-sided in their replication.** The mid arm is replicated eight
   times, the deployed envelope once (head seed 1234). §C29.4's paired comparison is the one
   with both sides replicated, and it is the one to quote.
6. **Head seeds and generation seeds are not exchangeable units.** A head seed re-randomises
   an MLP initialisation and its minibatch order; a generation seed re-randomises sampling.
   C29 does not claim they are the same kind of noise. It claims the protocol reports only
   the second.
7. **One dataset, one frozen generator, three anchors, `actual` accounting.** C24 is the
   external-validity check; C29 is not.
8. **The GPU was shared with a concurrent experiment.** One cell OOMed and was retried; the
   retry is deterministic in its seeds and changes no measured quantity, but wall-clock
   figures in `outputs/c29_progress.jsonl` are not comparable to a quiet machine. Compute is
   reported as processed tokens, never as wall-clock, throughout.

---

## C29.10 Priority 4 — C27 across head seeds

C27's equal-information result (deployed guidance loses to head-selected best-of-N by
-0.0439 / -0.0292 / -0.0522) rests on one head seed, and two of those three effects are
smaller than the head-seed spread C29 measures at the deployed configuration
(sd 0.0176 to 0.0241). Re-running `scripts/22_head_selected_bestofn.py` at head seeds 2345
and 3456 was pre-registered as priority 4, to be attempted only if time allowed.

**Priority 4 was run**, on head seeds 1234 / 2345 / 3456, all three anchors. Both sides
move with the head seed: the guided arm is C29's own deployed lambda = 1 cell at that head
seed, and the head-selected curve is rebuilt with the same checkpoint. Interpolation is
`scripts/21_summarise_c26.py::interp`, imported, exactly as C27 used it.

| anchor | head seed | guided | tokens/mol | head-selected @ budget | advantage | vs oracle |
|---|---|---|---|---|---|---|
| aromatic_rings | 1234 | 0.4735 | 419.3 | 0.5174 | -0.0439 | -0.3532 |
| aromatic_rings | 2345 | 0.4814 | 408.1 | 0.5115 | -0.0301 | -0.3386 |
| aromatic_rings | 3456 | 0.4589 | 417.5 | 0.4997 | -0.0408 | -0.3668 |
| hbd_count | 1234 | 0.2988 | 401.6 | 0.3279 | -0.0292 | -0.2472 |
| hbd_count | 2345 | 0.3462 | 401.2 | 0.3224 | 0.0238 | -0.1995 |
| hbd_count | 3456 | 0.3686 | 392.1 | 0.3286 | 0.0400 | -0.1689 |
| qed | 1234 | 0.1908 | 367.3 | 0.2431 | -0.0522 | -0.3715 |
| qed | 2345 | 0.1998 | 373.4 | 0.2428 | -0.0430 | -0.3689 |
| qed | 3456 | 0.1837 | 369.9 | 0.2439 | -0.0602 | -0.3813 |

| anchor | C27's published E4 | C29 mean over 3 head seeds | span | negative on | sign stable |
|---|---|---|---|---|---|
| aromatic_rings | -0.0439 | -0.0383 | 0.0139 | 3/3 | True |
| hbd_count | -0.0292 | 0.0115 | 0.0692 | 1/3 | False |
| qed | -0.0522 | -0.0518 | 0.0172 | 3/3 | True |

**C27's E4 flips sign on hbd_count.** At head seed 1234 the deployed arm sits 0.0292 below
the head-selected curve; at 2345 and 3456 it sits 0.0238 and 0.0400 **above** it. The span
across three head seeds is 0.0692 — larger than the published effect itself. aromatic_rings
(0.0439 / 0.0301 / 0.0408) and qed (0.0522 / 0.0430 / 0.0602) are sign-stable and their
spans (0.0139, 0.0172) are well inside their effects.

So C27's headline — *"the deployed configuration still loses to head-selected best-of-N on
all three anchors"* — holds on two anchors and **does not replicate on the third**. This is
precisely the weakness C27's own limitation 2 named, and it is the third result in this
section where a conclusion drawn at one head seed does not survive replication. No interval
is quoted here: n = 3 head seeds, and §C29.0.4.6 forbids a three-point bootstrap. The
per-head-seed values and the span are the whole evidence.

C27's *oracle* comparison is untouched: the deployed arm is 0.1689 to 0.3813 below the
oracle curve at every head seed on every anchor. It is only the equal-information margin
that is inside head-seed noise, which is the same lesson as §C29.3.2 — the effects this
project is now arguing about are the size of a variance component nobody measures.

---

## REPRODUCE

```bash
# pre-registration must already exist and be older than every output
# stage 0 (CPU): 48 `last1` head checkpoints at 8 head seeds, 3 anchors, 2 depths
.venv/bin/python scripts/23_head_seed_variance.py --stages heads --threads 8

# stages 1-5 (GPU): gates, the three mid arms, the deployed family, the lambda
# envelope, and the post-hoc deployed lambda=2 family.  Every cell is its own
# directory and a completed cell is never regenerated, so any stage can be
# interrupted and resumed.
.venv/bin/python scripts/23_head_seed_variance.py --stages gate
.venv/bin/python scripts/23_head_seed_variance.py --stages p1
.venv/bin/python scripts/23_head_seed_variance.py --stages p2mid
.venv/bin/python scripts/23_head_seed_variance.py --stages p2dep
.venv/bin/python scripts/23_head_seed_variance.py --stages p3
.venv/bin/python scripts/23_head_seed_variance.py --stages p4dep2

# assembly: reads existing artefacts only, generates nothing, trains nothing
.venv/bin/python scripts/23_summarise_c29.py

# priority 4: C27's equal-information comparison at head seeds 2345 and 3456.
# `scripts/22_head_selected_bestofn.py` is UNMODIFIED; it resolves the head from
# `<heads>/head_<prop>_frozen_state.pt`, so each head seed gets a staging directory
# holding a copy of `pilot_50k_heads_p2/head_<prop>_frozen_state_seed<hs>.pt` under
# both the plain and the seed-suffixed name.
for hs in 2345 3456; do
  mkdir -p "outputs/c29_c27heads_hs${hs}"
  for p in aromatic_rings hbd_count qed; do
    cp "outputs/pilot_50k_heads_p2/head_${p}_frozen_state_seed${hs}.pt" \
       "outputs/c29_c27heads_hs${hs}/head_${p}_frozen_state.pt"
    cp "outputs/pilot_50k_heads_p2/head_${p}_frozen_state_seed${hs}.pt" \
       "outputs/c29_c27heads_hs${hs}/head_${p}_frozen_state_seed${hs}.pt"
  done
  for p in aromatic_rings hbd_count qed; do
    .venv/bin/python scripts/22_head_selected_bestofn.py --dataset pilot_50k_p2 \
        --heads "c29_c27heads_hs${hs}" --property "$p" --n-max 32 --n-molecules 512 \
        --out "c29_c27_hs${hs}_${p}"
  done
done

# binding tests
.venv/bin/python -m pytest tests/test_head_seeds.py -q
```

Artefacts: `outputs/c29_prereg/` (pre-registration and SHA-256 lock), `outputs/c29_heads/`
(48 checkpoints and their cells), `outputs/c29_hs*_L*_{guided,bestofn,quality}/`,
`outputs/c29_dep_hs*_lam{1,2}_*_guided/`, `outputs/c29_deplam{1p25,1p5,2p5}_guided_*/`,
`outputs/c29_gate_L{4,12}_lam1_qed_hs1234/`, all with run contexts;
`outputs/c29_c27heads_hs{2345,3456}/` and `outputs/c29_c27_hs{2345,3456}_*/` (priority 4),
`outputs/c29_progress.jsonl` (per-cell completion log), `outputs/c29_*.log`, and
`outputs/c29_summary/c29_metrics.json`.
