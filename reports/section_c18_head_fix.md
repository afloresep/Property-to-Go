# Section C18 — can the head be fixed?

*Draft for merge into `reports/pilot_report.md` as section 20. Written 2026-07-30 on the
RTX 4090, phase-2 dataset `outputs/pilot_50k_p2/`, phase-2 heads
`outputs/pilot_50k_heads_p2/` (seed 1234, the checkpoint script 05 steers with).
Artefacts under `outputs/c18_*`; every asserted number is bound by
`tests/test_head_calibration.py`. No pre-existing directory under `outputs/` was
modified and no file outside the C18 allocation was touched.*

Section 15.6 decomposed the per-position capture loss: at λ=1 the deployed rule permits
32.6–53.2% of the head-free ceiling and our head collects 11.9–21.5% of that, so **the
head is the larger per-position loss**. C18 asks whether it can be fixed. The answer is
that the two cheap routes both fail, and the reason the first one fails is not empirical
— it is algebraic, and it was written down before the runs.

---

## 20.0 What was run

| stage | artefact | processed tokens |
| --- | --- | ---: |
| the written prediction, committed first | `c18_prediction/prediction.json` | 0 |
| off-policy re-measurement + post-hoc calibration, 6 properties × 2,000 guided molecules | `c18_offpolicy_calibration/` | 5,349,199 |
| three retrained readouts × 6 properties | `c18_heads_{wide,focused,wide_focused}/`, `c18_head_variants/` | 0 (no generation) |
| per-position capture for 7 arms × 6 properties | `c18_per_position/` | 85,127 |
| the λ-equivalence identity, 4 full guided runs | `c18_identity*/` | 2,511,378 |
| end to end: 4 calibration arms × 3 anchors, plus the sharpened arm of §20.5.1 | `c18_guided_{uncalibrated,platt,isotonic,bin_temperature,binT0p4}_*` | 8,975,924 |
| end to end: 2 retrained readouts × 3 anchors | `c18_guided_head_*` | 4,139,058 |
| compute-matched best-of-N, three shared runs | `c18_bestofn_N9_*`, `c18_matched_best_of_n/` | 1,842,813 |
| **total** | | **22,903,499** |

Cost is reported in processed tokens throughout, per §11.7. No wall-clock claim is made.

**Nothing here is a second DAgger round.** The one permitted data-aggregation round is
spent (§9.2.1). Every calibrator is fitted on the head's *outputs*; every retrained head
is trained on the phase-2 dataset's **base-policy** prefixes and on nothing else. Guided
prefixes are used for measurement and for post-hoc calibration only, which
`docs/HANDOFF.md` §6 E2 explicitly distinguishes from retraining.

---

## 20.1 The prediction, committed before any measurement

`outputs/c18_prediction/prediction.json`, written and timestamped before the first
measurement script ran; `test_the_prediction_was_written_before_the_measurements`
enforces the ordering by file mtime rather than by trust.

The decoder samples the next token from `softmax_a( log p_base(a) + λ·log(q(a) + ε) )`
over eight candidates. The question C18 has to answer before spending GPU time is which
calibration families are, and are not, algebraically distinct from a rescale of λ —
because §19 has already swept λ over {0.25, 0.5, 1, 2, 4, 8}.

| family | map | verdict committed in advance |
| --- | --- | --- |
| power | `g(q) = c·q^α` | **exactly** a λ rescale: `λ·log(c·q^α) = (λα)·log q + λ·log c`, and the softmax annihilates the second term |
| Platt | `g(q) = σ(a·logit q + b)` | that family **to first order**: `σ(x) → e^x` and `logit q → log q` as `q → 0`, and our candidate probabilities sit near base rates of 0.08–0.17 |
| isotonic | any monotone step function of `q` | **not** a λ rescale — it makes the effective exponent `d log g / d log q` depend on `q` — but it is monotone, so it cannot change *which* candidate the head prefers |
| bin-logit temperature | `q(T) = Σ_{i∈M} e^{z_i/T} / Σ_j e^{z_j/T}` | **not even a function of `q` alone**; two candidates with equal `q` can move differently, so this is the only post-hoc family that can reorder candidates |

And the directional prediction, which is the one that mattered:

> The stated defect is *under*-confidence. Any map correcting that at small `q` must
> satisfy `g(q) > q`, which for a power map means `α < 1`, which means
> `λ_eff = λα < 1`. §19 measures the lift falling steeply below λ=1. **So "fix the
> calibration by raising the head's probabilities" is a λ *decrease* and should make
> end-to-end guidance worse, not better.**

Three falsifiable claims were recorded with it: that the fitted Platt slope would be
`α < 1`; that a power-calibrated head at λ=1 would produce the *same molecules* as the
raw head at λ=α under the same seed; and that no arm would beat compute-matched
best-of-N.

**How it did.** The three claims held. The λ-equivalence held exactly. Two subsidiary
predictions failed and are marked below: that the off-policy factor would be smaller for
the count properties (it is largest for HBD count), and that the bin-logit temperature
would be "genuinely distinct" in a way that mattered (it is formally distinct and
behaves in practice like a reparametrised λ).

---

## 20.2 Trap 1 — the off-policy gap, re-measured on the fixed heads

`outputs/c18_offpolicy_calibration/offpolicy_calibration.json`,
`scripts/17_offpolicy_calibration.py`. Per property: 2,000 guided molecules generated at
λ=1 with the phase-2 head (seed 91234), one prefix drawn per position quartile from each
kept molecule exactly as scripts 02 and 08 draw them, giving ~7,970 guided prefixes;
against the phase-2 dataset's own held-out base-policy prefixes for the on-policy column.

| property | on-policy predicted | on-policy observed | ratio | on-policy ECE | **off-policy predicted** | **off-policy observed** | **ratio** | off-policy ECE | off-policy AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.1506 | 0.1587 | 1.05 | 0.0143 | 0.2513 | 0.4428 | **1.76** | 0.1922 | 0.6261 |
| HBD count | 0.0746 | 0.0825 | 1.11 | 0.0087 | 0.1387 | 0.2903 | **2.09** | 0.1516 | 0.6742 |
| rotatable bonds | 0.0940 | 0.0803 | 0.85 | 0.0137 | 0.1662 | 0.2246 | **1.35** | 0.0685 | 0.6305 |
| TPSA | 0.0955 | 0.0977 | 1.02 | 0.0074 | 0.1472 | 0.2489 | **1.69** | 0.1047 | 0.6358 |
| cLogP | 0.0943 | 0.0929 | 0.99 | 0.0045 | 0.1668 | 0.2825 | **1.69** | 0.1157 | 0.6598 |
| QED | 0.0907 | 0.0990 | 1.09 | 0.0085 | 0.1310 | 0.1846 | **1.41** | 0.0536 | 0.6120 |

**The pilot's 3.5x is now 1.69x for cLogP.** §9.2.1 reported the head predicting 0.076
against an observed 0.267 on guided prefixes; §11.6 argued about half of that factor was
the interval-mask defect. Measured on the fixed head, on an independent sample, the
factor is **1.69** — within 0.01 of what §11.5's arithmetic implies (3.5 / 2.08 = 1.68).
That agreement is close enough to be worth a caveat: the two quantities are measured on
different samples and one of them is a ratio of two noisy means, so it should be read as
"the same order and the same direction", not as a three-significant-figure match.
§11.6's correction to §8.2 is confirmed rather than merely argued.

Three further things this table says that the pilot could not.

**On-policy, every head is essentially calibrated.** Ratios 0.85–1.11, ECE 0.0045–0.0143.
The cLogP row (0.0943 against 0.0929) reproduces §11.6's fixed-mask measurement exactly,
which is a useful check that this pipeline and script 14 agree.

**The off-policy gap is real but modest, and it does not order by locality.** 1.35–2.09
across six properties. The prediction that the count properties — which never had the
mask defect — would show the *smallest* off-policy factors is **wrong**: HBD count has the
largest (2.09) and rotatable bonds the smallest (1.35), and the three counts straddle the
three continuous properties. Whatever drives the residual shift, it is not the defect and
it is not the count/continuous split.

**The head still ranks.** Off-policy AUROC is 0.6120–0.6742, well above chance and only
moderately below the on-policy target AUROCs of 0.735–0.790 (§13). This matters more than
the calibration does, and §20.3 says why.

---

## 20.3 Route (a) — post-hoc calibration works as calibration and does nothing for decoding

Calibrators fitted on half the guided molecules and scored on the other half, split by
canonical molecule so no molecule's four prefixes straddle the halves (the §3.2 grouping
rule). Platt by Newton–Raphson on two parameters; isotonic by PAVA; the bin-logit
temperature selected by lowest ECE on the fit half over a fixed grid.

| property | ECE uncalibrated | ECE Platt | ECE isotonic | AUROC uncalibrated | AUROC Platt | AUROC isotonic | Platt slope `a` | Platt intercept `b` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.2110 | **0.0464** | **0.0424** | 0.6368 | 0.6368 | 0.6363 | **0.405** | +0.236 |
| HBD count | 0.1624 | **0.0279** | **0.0213** | 0.6844 | 0.6844 | 0.6827 | **0.538** | +0.183 |
| rotatable bonds | 0.0785 | **0.0277** | **0.0273** | 0.6352 | 0.6352 | 0.6339 | **0.443** | −0.514 |
| TPSA | 0.1098 | **0.0205** | **0.0236** | 0.6256 | 0.6256 | 0.6243 | **0.583** | −0.023 |
| cLogP | 0.1181 | **0.0205** | **0.0121** | 0.6590 | 0.6590 | 0.6581 | **0.618** | +0.161 |
| QED | 0.0595 | **0.0138** | **0.0134** | 0.6216 | 0.6216 | 0.6204 | **0.545** | −0.434 |

**Calibration works, by a factor of 3 to 6 on ECE.** This is not a failed fit. If the
downstream result is negative it is not because the calibrator is bad.

**And AUROC does not move at all.** Platt's AUROC column is *identical* to the
uncalibrated column at four decimal places for all six properties, because Platt is
strictly monotone and AUROC is a rank statistic. Isotonic moves it by at most 0.0017, and
downward. **The decoder consumes a softmax over eight candidates, which is a function of
the *differences* of `log q` — a rank-and-spacing quantity, not a level. Post-hoc
calibration moves the level and leaves the ranking untouched.**

**Every fitted slope is below 1, as predicted, and by a lot: 0.405–0.618.** The
calibration that fixes the head's under-confidence is `λ_eff ≈ 0.4–0.6`, and §19 measures
guidance at λ=0.5 to be about **40%** as effective as at λ=1 (aromatic rings +0.1225
against +0.2949). The intercepts, which carry most of the *level* correction, are the part
the softmax annihilates — and where an intercept is large enough to push the calibrated
probability toward 1 the sigmoid saturates, which flattens the map further and lowers the
effective exponent again. Both mechanisms point the same way.

### 20.3.1 The identity, demonstrated end to end rather than argued

`outputs/c18_identity/identity_check.json`, `scripts/17_check_identity.py`. Four full
guided runs on aromatic rings, 512 molecules × 3 seeds, differing only in whether the
power calibration `g(q) = c·q^α` (α = 0.4047, from the fitted Platt slope) is applied at
λ=1 or the raw head is run at λ = α.

| ε | identical molecules | hit rate, calibrated arm | hit rate, λ = α arm | difference |
| --- | ---: | ---: | ---: | ---: |
| 0 (exact arithmetic) | **1536 / 1536 = 1.000** | 0.2869 | 0.2869 | **0.0** |
| 1e-6 (deployed) | 1535 / 1536 = 0.9993 | 0.2869 | 0.2869 | **0.0** |

At ε = 0 the two runs return **the same molecules, all 1,536 of them**. The `docs/HANDOFF.md`
§6 E2 recipe — "temperature-scale or isotonic-calibrate the head's interval probability,
then re-run the guided evaluation" — is, in its temperature-scaling half, not a new
experiment. It is a point on the λ sweep of §19, and the fitted slope says which point.

And the point it lands on is a bad one. Aromatic rings at λ=1 has `throughout` = 0.4735
(§16.1); the calibrated head has **0.2869**. The λ term is the whole of the difference.

### 20.3.2 Per-position: every monotone calibrator makes capture worse

`outputs/c18_per_position/per_position_capture.json`,
`scripts/17_per_position_capture.py`. The expensive half of §15.6 is on disk — 51,200
rollouts gave `p_hit[i, j]` and the permutation nulls — so only the head's `q[i, j]` has
to be recomputed, at 85,127 processed tokens for all seven arms and six properties.

The script rebuilds script 11's 400-prefix sample from seed 7777 and **refuses to run
unless it reproduces it**: candidate ids identical, base log-probabilities equal to
0.0 absolute difference. The baseline arm then reproduces §15.6's published
`our_head_gain` to six decimal places, which is the gate on everything below.

Per-position gain in hit-rate units at one decoding position, on the 267 prefixes §15.1
uses:

| property | **baseline** (§15.6) | Platt | isotonic | bin temperature (ECE-selected) | `wide` | `focused` | `wide_focused` | head-free ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | **+0.0147** | +0.0037 | +0.0087 | +0.0084 | +0.0127 | +0.0124 | **+0.0166** | 0.1349 |
| HBD count | **+0.0045** | +0.0020 | +0.0027 | +0.0010 | **+0.0080** | +0.0048 | +0.0055 | 0.0812 |
| rotatable bonds | **+0.0031** | +0.0014 | +0.0014 | +0.0023 | **+0.0039** | +0.0026 | +0.0031 | 0.0430 |
| TPSA | **+0.0065** | +0.0036 | +0.0051 | **+0.0085** | +0.0071 | +0.0071 | +0.0078 | 0.0744 |
| cLogP | **+0.0071** | +0.0039 | +0.0058 | **+0.0105** | +0.0072 | +0.0065 | +0.0060 | 0.0770 |
| QED | **+0.0028** | +0.0014 | +0.0017 | +0.0026 | +0.0026 | +0.0020 | +0.0013 | 0.0583 |

As a share of the λ=1 optimum (the noise-corrected oracle head of §15.6):

| property | baseline | Platt | isotonic | bin temperature | `wide` | `focused` | `wide_focused` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 20.8% | 5.2% | 12.4% | 11.9% | 18.0% | 17.5% | **23.6%** |
| HBD count | 11.9% | 5.4% | 7.1% | 2.6% | **21.2%** | 12.8% | 14.4% |
| rotatable bonds | 13.5% | 6.3% | 6.1% | 9.9% | **17.0%** | 11.2% | 13.6% |
| TPSA | 21.5% | 12.0% | 16.9% | **28.2%** | 23.8% | 23.6% | 25.8% |
| cLogP | 17.9% | 9.7% | 14.6% | **26.5%** | 18.1% | 16.3% | 15.0% |
| QED | 14.8% | 7.5% | 9.0% | 13.5% | 13.4% | 10.7% | 6.8% |

**Platt and isotonic are below the uncalibrated head at every one of the six properties**,
and Platt lands within 0.0013 of the λ = α rescale it is predicted to be
(`lambda_rescale_at_the_platt_equivalent` in the artefact). Isotonic sits above Platt
because it is less aggressive, not because it is doing something different in kind.

**Neither can change which candidate is chosen, and that is measured, not assumed.**
`picks_the_best_candidate_rate` — the share of prefixes at which the head assigns its
maximum score to the candidate the rollouts say is best — is *bit-identical* under Platt
for five of six properties, and rises under isotonic only because isotonic's flat steps
create ties. Ties, not discoveries.

### 20.3.3 The bin-logit temperature: formally distinct, practically another λ

This is the one post-hoc family that *can* reorder candidates, and it does: its
`picks_the_best_candidate_rate` differs from the baseline's for five of six properties.
It is also the only post-hoc arm that ever improves per-position capture — TPSA
+0.0065 → +0.0085, cLogP +0.0071 → +0.0105.

Sweeping the temperature explains both facts at once. Per-position gain against `T`:

| property | T=0.4 | T=0.5 | T=0.75 | T=1.0 | T=1.6 | T=3.0 | T=4.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.0385 | 0.0314 | 0.0202 | 0.0147 | 0.0084 | 0.0041 | 0.0030 |
| HBD count | 0.0114 | 0.0089 | 0.0060 | 0.0045 | 0.0027 | 0.0014 | 0.0010 |
| cLogP | 0.0157 | 0.0128 | 0.0090 | 0.0071 | 0.0047 | 0.0026 | 0.0020 |

**The response is monotone decreasing in T for all six properties**, with the best value
at the grid edge. Sharpening the bin logits sharpens `q`, which widens the spread of
`log q` across candidates, which is what raising λ does. Aromatic rings at T=0.4 gives
+0.0385; the same head at λ=2 gives +0.0324. The family is formally outside the λ
parametrisation and behaves inside it.

**And the temperature that calibrates is not the temperature that decodes.** The
ECE-selected temperature is above 1 for aromatic rings (1.6), HBD count (4.0), rotatable
bonds (1.4) and QED (1.1) — the calibration objective wants the head *flattened*, and
flattening costs the decoder. Where the ECE-selected temperature happened to fall below 1
(cLogP 0.625, TPSA 0.675) the apparent "calibration improvement" in the per-position table
is not a calibration benefit at all; it is a λ increase wearing a calibration label.

**This is the sharpest statement C18 can make.** Calibration error and decoding quality
are different objectives, they point in opposite directions for four of six properties,
and improving the first is not a route to the second.

---

## 20.4 Route (b) — a larger and a differently-targeted readout

`scripts/17_train_head_variants.py`, `outputs/c18_head_variants/head_variants_summary.json`.
Three variants, all trained on the phase-2 base-policy prefixes with the same recipe,
same seed (1234), and all deliberately loadable by the **unmodified**
`scripts/05_guided_generation.py`, so the end-to-end stage inherits every test that
already covers guided decoding.

* **`wide`** — the same two-layer MLP with `hidden_dim` 1024 instead of 256: 1.84M
  parameters against 268k, a 6.9x increase. §8.3 records the single architecture as an
  explicit limit on the negative result; this is the cheapest test of whether it binds.
* **`focused`** — a three-bin readout whose middle bin *is* the target interval, so the
  head optimises the event guidance scores instead of a 20-way distribution of which the
  target is a marginal. Built as a `QuantileBinner` with `edges = [−∞, lo, hi, ∞)`, so the
  §11.5 union-of-bins invariant holds by construction and no new binner class is needed.
* **`wide_focused`** — both.

Held-out target-interval AUROC, phase-2 test split:

| property | baseline (256, full) | `wide` (1024, full) | `focused` (256, 3-bin) | `wide_focused` | parameters, `wide` |
| --- | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.7904 | 0.7886 | 0.7814 | 0.7794 | 1,843,206 |
| HBD count | 0.7781 | 0.7780 | 0.7564 | 0.7570 | 1,844,231 |
| rotatable bonds | 0.7806 | 0.7794 | 0.7654 | 0.7566 | 1,850,381 |
| TPSA | 0.7391 | 0.7253 | 0.7149 | 0.7147 | 1,859,606 |
| cLogP | 0.7901 | 0.7920 | 0.7831 | 0.7773 | 1,859,606 |
| QED | 0.7355 | 0.7367 | 0.7271 | 0.7283 | 1,859,606 |

**Seven times the parameters buys nothing.** `wide` is within ±0.014 of the baseline
everywhere and its median change is −0.0007 — inside the ±0.0041 head-seed standard
deviation §13.2 measured. `focused` is *worse* on **all six**, most on TPSA (−0.0242) and
HBD count (−0.0217): coarsening to three bins removes the auxiliary signal a 7-to-22-way
head gets from having to place the whole distribution, and buys nothing in exchange
because the target marginal was already exact after the §11.5 fix.

**This eliminates one of §8.3's three suspects and adds a fourth that also fails.** §8.3
says the aromatic-ring negative result "shows *this* readout fails, not that the
information is absent from the state", and names three: an earlier layer, a larger head,
a different pooling. **The larger head is now tested and does not help.** The
target-focused readout was not on §8.3's list and does not help either. **Pooling is
still untested** — every head here, baseline and variant, reads the single hidden state
at one position — and the layer is C17's. Two of the four remain open and should be
described that way.

NLL and expected-value MAE are **not** comparable across these variants, because the
`focused` readout partitions the outcome space differently — the same trap §11.6 finding 4
identified. The comparable columns are the ones on the same event: target AUROC, target
Brier, target ECE. The artefact carries a note to that effect on every row.

---

## 20.5 End to end — the measurement, not the extrapolation

**This is the part that matters, and `docs/TODO.md` C22.1 is why.** Everything in §20.3.2
and §20.4 is a gain at *one decoding position* with the rest of the sequence left to the
base policy. End-to-end lift is 20–48x the per-step gain, and transferring the ratios
linearly implies hit rates above 1 for four of six properties. **A per-position
improvement is not an end-to-end improvement and none of the numbers above may be
converted into one.** So the arms were run.

`outputs/c18_summary/c18_summary.json`, assembled by `scripts/17_summarise_c18.py` from
the individual runs. Three anchors — aromatic rings (most steerable at λ=1), HBD count
(the pre-registered discriminating case), cLogP (the property the whole calibration story
was about). `unguided` + `throughout`, 3 seeds, 512 molecules per condition per seed,
frozen windows and intervals, compute-matched best-of-N solved from each run's own token
count under `actual` accounting exactly as §16.2 does.

The `uncalibrated` arm is the within-script control: `guided_sample` reseeds at every
call, so it must reproduce the central test's own numbers, and it does exactly —
aromatic rings `unguided` 0.1785, `throughout` 0.4735, against §16.1's 0.1785 / 0.4735.

Best-of-N is **not** run once per arm. It is deterministic in (property, N, seeds), and
every one of the nineteen arms solves to the same **N = 9**, with realised token ratios
0.954–1.018 (`c18_matched_best_of_n/matched_best_of_n.json`). So it is run once per
distinct N and shared, and the shared run reproduces §16.2's published value exactly
(aromatic rings 0.8294, HBD count 0.5234, cLogP 0.6107). Its token cost is also identical
across the three properties (614,271 each), which is the expected signature of a baseline
that samples unconditionally and only *selects* on the property. Running it per arm would
have added **9,828,336 processed tokens** — 75% of the entire guided budget — to recompute
three numbers. The sharing is recorded per arm and asserted by
`test_best_of_n_is_shared_because_every_arm_solves_to_the_same_n` rather than done
silently; if any arm ever solves to a different N it gets its own run automatically.

| property | arm | per-position gain | ×baseline | **`throughout`** | **lift** | **×baseline** | validity | best-of-9 | **advantage** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **aromatic rings** | uncalibrated (control) | +0.0147 | 1.00x | 0.4735 | +0.2949 | 1.00x | 0.9954 | 0.8294 | **−0.3560** |
| | Platt | +0.0037 | 0.25x | 0.2465 | +0.0680 | **0.23x** | 0.9954 | 0.8294 | **−0.5829** |
| | isotonic | +0.0087 | 0.59x | 0.3009 | +0.1223 | **0.41x** | 0.9935 | 0.8294 | **−0.5286** |
| | bin temperature | +0.0084 | 0.57x | 0.3436 | +0.1650 | **0.56x** | 0.9967 | 0.8294 | **−0.4859** |
| | `wide` readout | +0.0127 | 0.86x | 0.4532 | +0.2747 | 0.93x | 0.9954 | 0.8294 | **−0.3762** |
| | `wide_focused` readout | +0.0166 | 1.14x | **0.5007** | **+0.3221** | **1.09x** | 0.9961 | 0.8294 | **−0.3288** |
| **HBD count** | uncalibrated (control) | +0.0045 | 1.00x | 0.2988 | +0.2150 | 1.00x | 0.9980 | 0.5234 | **−0.2247** |
| | Platt | +0.0020 | 0.45x | 0.1567 | +0.0730 | **0.34x** | 0.9974 | 0.5234 | **−0.3668** |
| | isotonic | +0.0027 | 0.60x | 0.2128 | +0.1291 | **0.60x** | 0.9941 | 0.5234 | **−0.3106** |
| | bin temperature | +0.0010 | 0.22x | 0.1280 | +0.0443 | **0.21x** | 0.9967 | 0.5234 | **−0.3954** |
| | `wide` readout | +0.0080 | **1.78x** | 0.2997 | +0.2160 | **1.00x** | 0.9928 | 0.5234 | **−0.2238** |
| | `wide_focused` readout | +0.0055 | **1.22x** | 0.2847 | +0.2010 | **0.93x** | 0.9948 | 0.5234 | **−0.2388** |
| **cLogP** | uncalibrated (control) | +0.0071 | 1.00x | 0.3030 | +0.1997 | 1.00x | 0.9948 | 0.6107 | **−0.3077** |
| | Platt | +0.0039 | 0.54x | 0.2040 | +0.1007 | **0.50x** | 0.9954 | 0.6107 | **−0.4066** |
| | isotonic | +0.0058 | 0.81x | 0.2435 | +0.1402 | **0.70x** | 0.9863 | 0.6107 | **−0.3671** |
| | bin temperature | +0.0105 | 1.48x | **0.3593** | **+0.2559** | **1.28x** | 0.9967 | 0.6107 | **−0.2514** |
| | `wide` readout | +0.0072 | 1.01x | 0.3022 | +0.1989 | 1.00x | 0.9974 | 0.6107 | **−0.3085** |
| | `wide_focused` readout | +0.0060 | 0.84x | 0.2555 | +0.1522 | **0.76x** | 0.9935 | 0.6107 | **−0.3552** |

Five things, in descending order of how much they matter.

**1. No arm beats compute-matched best-of-N, anywhere.** The best advantage across all
nineteen arms — the eighteen above plus the sharpened arm of §20.5.1 — is **−0.2238**
(HBD count, `wide` readout), against the deployed head's −0.2247. R4 continues to fire. This is now established across six properties, two token
accountings, six values of λ (§19) and six head-and-calibrator arms.
`any_arm_anywhere_beats_compute_matched_best_of_n` is `false` in the artefact.

**2. Every arm that actually corrects the calibration is worse end to end, at every
property, without exception.** Platt costs 0.23–0.54x of the deployed lift and isotonic
0.41–0.70x — six of six cells, no exceptions. The ECE-selected bin temperature is worse at
two of three anchors (0.56x, 0.21x) and better at one (cLogP, 1.28x), and point 3 explains
why that one is not a counterexample. So the prediction of §20.1 — that correcting the
head's under-confidence is a λ decrease and should *hurt* — holds at **eight of the nine
calibration cells**, and the ninth is a sharpening rather than a level correction.
`docs/HANDOFF.md` §6 E2's recommendation to "fix the calibration
first, *then* re-test guidance, so 'guidance fails' is not confounded with 'the guidance
signal is broken'" is therefore not merely unnecessary; **following it would have made the
negative result worse and would have been read as evidence for it.**

**3. cLogP's bin-temperature arm is the only calibration cell above 1.00x, and it is a λ
result wearing a calibration label.** Its ECE-selected temperature is 0.625 — *sharpening*,
not the flattening calibration usually means — and §20.3.3's sweep shows the response to
temperature is monotone in the sharpening direction for all six properties. The +0.2559
lift is what a λ increase buys; it is available more directly and more controllably from
§19's λ knob, and §19 also records what that knob costs above its optimum.

**4. The best retrained readout buys 1.09x at one property and nothing at the other two.**
`wide_focused` takes aromatic rings from +0.2949 to **+0.3221**, a real 1.09x with validity
unchanged at 0.9961, and narrows the gap to best-of-N from −0.3560 to −0.3288. On HBD count
and cLogP the same readout is *worse* (0.93x, 0.76x) and `wide` is exactly neutral (1.00x,
1.00x). Averaged over the three anchors the retrained readouts buy nothing; the one gain is
property-specific and is smaller than §19's λ tuning (1.29–1.69x).

**5. And the clearest empirical demonstration of `docs/TODO.md` C22.1 this project has
produced.** Compare the two ratio columns:

| case | per-position | end to end |
| --- | ---: | ---: |
| HBD count, `wide` | **1.78x** | **1.00x** |
| HBD count, `wide_focused` | **1.22x** | **0.93x** — *sign reversal* |
| cLogP, `wide_focused` | 0.84x | 0.76x |
| aromatic rings, `wide_focused` | 1.14x | 1.09x |

A **78% per-position improvement produced a 0.5% end-to-end improvement**, and a 22%
per-position improvement produced a 7% end-to-end *loss*. The per-step decomposition of
§15.6 is sound as a decomposition and is not a predictor of end-to-end effect, exactly as
the audit said. Had C18 stopped at the per-position table it would have reported "the
readout can be improved by up to 1.78x" and been wrong.

### 20.5.1 The decoder-optimal temperature is a worse λ than λ

`outputs/c18_guided_binT0p4_aromatic_rings/`. §20.3.3 showed the per-position gain rises
monotonically as the bin logits are sharpened, with the best value at the grid edge
`T = 0.4`. Run end to end on aromatic rings, against §19's own λ=2 run on the same
dataset, seeds and windows:

| arm | `throughout` | lift | validity | uniqueness | content length | tokens/molecule | advantage vs best-of-9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bin temperature `T = 0.4`, λ=1 | 0.5495 | +0.3710 | 0.9857 | 1.0000 | 46.09 | 426.3 | **−0.2799** |
| raw head, λ=2 (§19) | **0.5579** | **+0.3794** | **0.9909** | 1.0000 | 45.57 | 420.3 | **−0.2715** |

**Sharpening the head is a slightly worse version of raising λ**: lower hit rate, lower
validity, and more tokens per molecule. It confirms the §20.3.3 reading — the family is
formally outside the λ parametrisation and inside it in every way that matters — and it
closes the one route by which a post-hoc method might have escaped the §19 envelope.
It also still loses to best-of-9.

---

## 20.6 What C18 changes

**1. §8.2's diagnosis is retired completely, and its causal story was inverted, not merely
exaggerated.** §8.2 argued that "a head that never emits a confident probability produces a
compressed score range and a weaker effective lambda", i.e. under-confidence ⟹ weak λ.
§11.6 corrected that for the mask defect ("that is not what a dropped bin does"). C18
establishes the general case, and the arrow points the other way: under-confidence is
mostly a *level* error, the softmax over candidates is invariant to the level, and
**correcting it with a monotone map necessarily flattens `log q`, which is a λ decrease.**
Measured: correcting it costs 0.23–0.70x of the deployed lift. So the direction of §8.2's
mechanism is wrong, not just its magnitude. **This is a claim in the existing report that
C18 contradicts and it should be flagged where §8.2 is quoted.**

**2. The re-measured off-policy factor is 1.35–2.09, not 3.5, and §11.6's arithmetic
predicted it.** cLogP is 1.69 against §11.6's implied 3.5/2.08 = 1.68. On base-policy
prefixes every head is calibrated (ratios 0.85–1.11). §11.6's correction to §8.2 is now
confirmed on an independent sample rather than inferred from a checkpoint.

**3. `docs/HANDOFF.md` §6 E2 can be closed, and it should be closed as a negative rather
than as unattempted.** Its "cheap version" is, in the temperature/Platt half, *exactly* a
rescale of λ — proved algebraically, checked on the real candidate array, and demonstrated
by two full guided runs returning the same 1,536 molecules. Its isotonic half is not a λ
rescale but is monotone in `q` and so cannot change which candidate is preferred; measured,
it costs 0.41–0.70x. **Following E2's advice would have made the negative result worse and
would then have been read as evidence for it**, which is the specific failure mode E2 was
written to prevent.

**4. One of §8.3's three named suspects is eliminated and a fourth is added and also
eliminated.** The larger head is tested and does not help (±0.014 AUROC, median −0.0007,
at 6.9x the parameters); a target-focused readout — not on §8.3's list — is tested and is
worse on all six properties. **Pooling and the probe layer remain untested here** and the
layer is C17's, so §8.3 should be rewritten to name the two that survive rather than the
three it currently lists.

**5. §19.4's closing sentence — "that makes fixing the head the better bet" — does not
survive C18.** §19 measured the λ term end to end at 1.29–1.69x and left the head term
measured only per position. C18 measures the head term end to end by the two routes the
brief permits, and the best retrained-readout result across three anchors is **1.09x**
(aromatic rings, `wide_focused`), with the other two anchors at 1.00x and 0.76x for the
same readout. The
honest position after C18 is that **neither term is cheaply available**: λ is worth
1.3–1.7x and is capped by base-policy destruction (§19.1), and the head is not improvable
by calibration at all and only marginally and inconsistently by capacity or readout shape.

**6. R4 is now established across six head-and-calibrator arms per anchor — nineteen
runs in all — as well as across six properties, two accountings and six λ.** The best advantage anywhere is −0.2238. Nothing in C18 moves
the negative result, and the space of cheap explanations for it has narrowed rather than
widened.

**7. A methodological result the paper should carry.** C18 produced the cleanest available
demonstration that a per-position improvement is not an end-to-end improvement: 1.78x
per-position became 1.00x end to end, and 1.22x per-position became 0.93x — a sign
reversal. `docs/TODO.md` C22.1 established this as an arithmetic argument; it is now an
observation.

**8. And a result about probe calibration that is not specific to this project.** The
head's calibration error and its usefulness to the decoder are different objectives that
point in opposite directions. Post-hoc calibration reduced ECE by a factor of 3–6 and left
AUROC bit-identical, because it is monotone and AUROC is a rank statistic — and a softmax
over `k` candidates consumes ranks and spacings, never levels. **A miscalibrated probe used
inside a softmax over candidates does not need calibrating; it needs sharpening or it needs
to be a better ranker.** The ECE-selected temperature was above 1 — the wrong direction for
decoding — for four of the six properties.

---

## 20.7 Limitations, stated rather than defended

**One head-training seed per variant.** The baseline heads have three (§13.2, seed sd
≤0.0041 AUROC); the three C18 variants have one each, seed 1234. The `wide` differences
are inside that band, so "capacity does not help" is safe; `focused`'s −0.0217 and
−0.0242 are outside it and are safe in the other direction. A margin between two variants
under ~0.008 should not be read from this table.

**The variants inherit the baseline's training hyperparameters.** Learning rate, weight
decay, batch size, patience and epoch budget come from `configs/pilot_50k.yaml` unchanged,
which is the recipe tuned around a 20-way head. A three-bin head might prefer different
settings. Tuning per variant was deliberately not done, because a per-variant hyperparameter
search is a degree of freedom that would make "the retrained head is better" unfalsifiable;
the cost is that `focused` may be under-served rather than genuinely worse.

**The end-to-end bin-temperature arm uses the calibration-optimal `T`, not the
decoder-optimal one.** §20.3.3 shows those are different, and that the decoder-optimal `T`
sits at the sharpening edge of the grid. Running the decoder-optimal `T` end to end is
λ-tuning under another name and §19 has already established what λ-tuning is worth and
what it costs, so it was not run for every property. Where it *was* run — aromatic rings,
§20.5.1 — it is reported as a λ result rather than as a calibration result.

**Best-of-N is one run per distinct `N` rather than one per arm.** Justified by the
determinism of `bestofn.best_of_n` in (property, `N`, seeds) and checked: every arm
solves to the same `N` and the shared run reproduces the central test's published value.
`c18_matched_best_of_n/matched_best_of_n.json` records each arm's solved `N` and realised
token ratio so the match is auditable.

**Pooling is untested.** Every head here reads `h_t` at a single position. A readout over
a pooled window of hidden states is the one item on §8.3's list that neither C17 nor C18
touches.

**The off-policy prefixes come from one guided configuration.** λ=1, `throughout`, one
generation seed, 2,000 molecules per property. A head calibrated for λ=1 prefixes is not
calibrated for λ=2 prefixes, and §19's optimum is λ=2 for two of three anchors. This
matters less than it sounds, because the finding is that calibration does not transfer to
decoding at all — but it does mean the *size* of the off-policy gap is quoted at one λ.

---

## Commands to add to `docs/REPRODUCE.md`

```bash
# C18 -- fixing the head. Stages are resumable and idempotent; run in this order.
bash scripts/17_run_c18.sh prediction      # writes the pre-committed prediction FIRST
bash scripts/17_run_c18.sh offpolicy       # re-measure the off-policy gap, fit calibrators
bash scripts/17_run_c18.sh heads           # train the wide / focused / wide_focused readouts
bash scripts/17_run_c18.sh perposition     # per-position capture for every arm
bash scripts/17_run_c18.sh identity        # the lambda-equivalence identity, end to end
bash scripts/17_run_c18.sh e2e_calibrated  # 4 calibration arms x 3 anchors
bash scripts/17_run_c18.sh e2e_heads       # 2 retrained readouts x 3 anchors
bash scripts/17_run_c18.sh bestofn         # matched best-of-N, once per distinct N
bash scripts/17_run_c18.sh summary         # assemble the end-to-end table
.venv/bin/python -m pytest tests/test_head_calibration.py -p no:cacheprovider   # 34 passed

# section 20.5.1 only: the decoder-optimal bin temperature, run as a lambda result
.venv/bin/python scripts/17_guided_calibrated.py --property aromatic_rings \
    --arm bin_temperature --bin-temperature 0.4 --out c18_guided_binT0p4_aromatic_rings
bash scripts/17_run_c18.sh bestofn && bash scripts/17_run_c18.sh summary
```

`ANCHORS` and `HEAD_VARIANTS` are environment overrides on the driver; the defaults are
the three anchors and the two readouts that ever improved per-position capture on an
anchor (`focused` improved none and is reported at the per-position stage only).
`scripts/17_run_c18.sh all` runs the whole chain. A completed guided run is never redone,
so the chain can be resumed after an interruption without discarding work.

**Order matters in one place only, and it is load-bearing**: `prediction` must run before
`offpolicy`, because `test_the_prediction_was_written_before_the_measurements` compares
file mtimes.
