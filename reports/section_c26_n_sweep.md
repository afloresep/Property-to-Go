# Section C26 — the N sweep: best-of-N's compute–accuracy frontier

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-01. Pre-registration `outputs/c26_prereg/C26.0_preregistration.md`, written
2026-07-31 11:42:58, before any C26 output directory existed. Every number below is
re-derived from JSON by `tests/test_n_sweep.py`.

---

## The verdict, up front

**Best-of-N dominates guided decoding at every measured budget, on all three anchors, once
the comparison is made against a curve instead of a point.** Forty-six guidance arms were
priced against the best-of-N frontier at their own exact token cost. Forty-five sit below
it, by −0.0277 to −0.5825. One sits above it by +0.0267 — C23's hbd_count L4 λ=2 arm, the
sole basis of C23's Rule B — and **that one does not survive head-seed replication**:
priced on the same curve, C25's three head seeds give **+0.0267, +0.0210, −0.0649**, mean
−0.0057, sd 0.0513, sign flipped.

**C23's Rule B is dead, not marginal.** This is C26's main result and it is a negative one.

A second finding is structural rather than statistical, and may matter more than the first.
**Guidance has no compute knob.** Across all 46 arms — six λ values, four calibrations, two
readout widths, four probe layers — the realised cost stays inside 5.1% (hbd_count), 14.3%
(aromatic rings) and 17.0% (QED) of its own minimum, while best-of-N spans **32×** over the
same grid. Guidance is a near-vertical line at ~360–460 processed tokens per molecule, not a
frontier. §4's conclusion is therefore stronger than it was stated: guided decoding does not
merely lose at the budget we matched at, it **cannot be offered more compute** within the
method as specified.

What C26 does **not** show: that best-of-N is a good method, that guidance could not win at
some budget outside 44–1422 tokens per molecule, or that guidance is useless. §C26.7.

---

## C26.0 The pre-registration, verbatim

The file below is `outputs/c26_prereg/C26.0_preregistration.md`. It is reproduced here in
full and unedited; `tests/test_n_sweep.py::test_the_report_copies_the_prereg_verbatim`
asserts the copy is a byte-identical substring of this section, and
`test_the_prereg_was_written_before_every_measurement` asserts its mtime strictly precedes
every C26 artefact.

Two things in it are wrong and are scored as wrong in §C26.5: the nesting design in
C26.0.2 was replaced after it failed its own gate (§C26.2.2), and prediction 3 is falsified
(§C26.5.3).

<!-- BEGIN VERBATIM PREREG COPY -->

## C26.0.1 Why

`docs/HANDOFF.md` E1 asked for a λ sweep **and** an N sweep; only the λ half was run
(`pilot_report.md` §19). The simulated reviewer's actual request was a **compute–accuracy
frontier for both methods**, not one point each. At present guidance has six points per
anchor (§19's λ grid) plus C23's mid-layer arms, and best-of-N has exactly one point per
guided run — the compute-matched N. Two curves and one point cannot distinguish
"reranking is fundamentally uncompetitive at matched compute" from "reranking loses at the
budget we happened to match at".

C23 makes this urgent rather than tidy. Its one arm that beat best-of-N did so with a
realised guided/best-of-N token ratio of **1.0897**, because integer flooring took the
matched N from 9 to 8. With a continuous best-of-N frontier the comparison no longer
depends on where the floor lands.

## C26.0.2 Design

Three anchors — **aromatic_rings, hbd_count, qed** — the same three as §19 and C23, so every
number is directly comparable. Not chosen after seeing anything.

N grid, fixed here: **1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32**. N=1 is the unguided base policy
by construction and is a bug alarm, not a data point. N=9 is in the grid because it is the
matched N published in §16.2 and §19.2 for two of the three anchors.

Seeds 101 / 202 / 303, 512 returned molecules per seed, the frozen intervals and windows,
`actual` token accounting. `full_recompute` is not run: §19 records that best-of-N saturates
to 1.0000 under it for every property, so it cannot discriminate.

**One pool, nested subsets.** For each (property, seed) a single pool of 32 × 512
unconditional molecules is drawn, split into 512 slots of 32, and best-of-N for each N in the
grid selects over the **first N** of each slot. Each slot therefore still selects among N
i.i.d. base-policy draws, so every point is an unbiased best-of-N estimate; the points are
*paired* across N, which is the right structure for reading a single curve and is stated
rather than hidden. It costs one generation pass instead of eleven.

## C26.0.3 Validity gates, checked before any curve is read

1. **Exact gate.** A separate run drawing the pool with the *published* call signature
   (total = 9 × 512) must reproduce `outputs/pilot_50k_p2_bestofn_aromatic_rings`'s
   published best-of-9 hit rate (0.8294) and its tokens per molecule (401.09 at seed 101).
   Residual reported as a number, not as the word "matches".
2. **Nesting gate.** The N=9 point read off the nested 32-pool must agree with gate 1's
   exact value within seed noise. The nested pool is drawn with a different total, so the RNG
   stream differs and bit-identity is **not** expected; what is claimed is that the nesting
   approximation costs nothing that matters. The discrepancy is reported as a number and, if
   it exceeds the between-seed sd, the nested estimates are reported as approximate and the
   exact call signature is used for every grid point instead.
3. **N=1 gate.** The N=1 point must reproduce the unguided hit rate of the corresponding
   §16.1 run (aromatic rings 0.1785) within seed noise. If it does not, the pool is not the
   base policy and nothing below is valid.

## C26.0.4 What the frontier is for, and the decision rules

Guidance points come from artefacts that already exist and are **not** regenerated: §19's λ
grid at the deployed layer, plus C23's mid-layer arms. Each supplies (tokens per returned
molecule, hit rate, validity).

- **D1 — does best-of-N dominate?** Best-of-N dominates at a budget if its interpolated hit
  rate at that budget exceeds guidance's. The claim "guidance loses at matched compute"
  is upheld iff best-of-N dominates at **every** budget where both are measured, for all
  three anchors. Any budget where guidance is above the best-of-N curve is reported, with
  its margin, whether or not it is significant.
- **D2 — the C23 arm, priced properly.** The hbd_count L4 λ=2 arm spends 387.79 tokens per
  molecule. Its advantage is recomputed against the best-of-N curve **interpolated at that
  exact budget**, which removes the flooring artefact in both directions. This is the number
  C23's headline should have been stated in. It is reported whatever its sign.
- **D3 — does best-of-N saturate?** If the best-of-N curve flattens inside the measured
  range, then "loses to best-of-N" is a statement with a ceiling and the interesting budget
  is where the curves would cross if extrapolated. If it does not flatten, say so; do not
  extrapolate a curve that is still rising.

## C26.0.5 Multiplicity and uncertainty

11 grid points × 3 anchors. The curve is descriptive and no per-point test is claimed, so no
correction is applied to the curve itself. **D2 is a single pre-specified comparison** and is
reported with a seed-stratified bootstrap CI **and** a seed-level interval on n=3, because
the C23 review established that a molecule-level bootstrap understates the variance that
matters when only three seeds exist. Both intervals are published. Per-seed values for every
grid point are published so the noise floor is legible.

## C26.0.6 Predictions

1. Best-of-N rises with N and is **concave** — each doubling of N buys less. Falsified if any
   anchor's curve is convex over the measured range.
2. Best-of-N dominates guidance at every measured budget for **aromatic rings and QED**,
   where the deployed gaps are −0.27 and −0.28.
3. **For HBD count it does not**, and this is the sub-prediction that can fail independently:
   C23's L4 λ=2 arm should sit above the interpolated curve, because the flooring that
   produced the 1.0897 ratio moved the comparator by less than the arm's +0.0369 margin over
   the published N=9 value. If it sits below, C23's Rule B is dead rather than marginal, and
   that is the result.
4. The curves do **not** cross for aromatic rings anywhere in 1 ≤ N ≤ 32.

Prediction 3 is the one I expect to be least reliable, because it rests on a single arm with
one seed near a tie (per-seed advantages 0.0547 / 0.0031 / 0.0529 against the published N=9).

<!-- END VERBATIM PREREG COPY -->

---

## C26.1 What was run

| stage | script | output | cost |
|---|---|---|---|
| exact gate | `scripts/21_n_sweep.py --exact-n 9` | `outputs/c26_gate_exact_N9_aromatic_rings/` | 1 pool of 9×512 per seed |
| N sweep, v1 (**superseded**) | `scripts/21_n_sweep.py` | `outputs/c26_nsweep_v1_nested_{aromatic_rings,hbd_count,qed}/`, log `outputs/c26_nsweep_v1_nested.log` | 3 anchors × 3 seeds × 32×512 |
| N sweep, v2 (**reported**) | `scripts/21_n_sweep.py` | `outputs/c26_nsweep_{aromatic_rings,hbd_count,qed}/`, log `outputs/c26_nsweep.log` | 3 anchors × 3 seeds × 32×512 |
| frontier assembly | `scripts/21_summarise_c26.py` | `outputs/c26_summary/c26_metrics.json` | reads only; generates nothing |

`scripts/21_summarise_c26.py` **generates no molecules**. Every guidance point on the
frontier comes from an artefact that already existed before C26 began — §19's λ grid, C18's
calibration and readout arms, C23's mid-layer arms, and the deployed λ=1 runs. C26 adds the
best-of-N curve and prices the existing points against it. This is why the frontier can
carry 46 arms at no generation cost, and it is also why C26 cannot add a guidance arm at a
budget nobody has already measured.

The v1 outputs are kept rather than deleted, because §C26.2.2 is a result about the
estimator and the discarded numbers are its evidence.

---

## C26.2 Validity gates, checked before any curve was read

### C26.2.1 Gate 1 — the exact call signature

A separate run drawing the pool with `scripts/06_best_of_n.py`'s own call signature
(9 × 512, `seed * 1000`) must reproduce the published best-of-9. Residuals as numbers:

| seed | gate hit rate | published | residual | gate tokens/mol | published | residual |
|---|---|---|---|---|---|---|
| 101 | 0.80859375 | 0.80859375 | **0.0** | 401.09375 | 401.09375 | **0.0** |
| 202 | 0.830078125 | 0.830078125 | **0.0** | 398.837890625 | 398.837890625 | **0.0** |
| 303 | 0.849609375 | 0.849609375 | **0.0** | 399.81640625 | 399.81640625 | **0.0** |

Bit-identical on hit rate *and* on token cost, all three seeds. **PASS.**

Reaching this took one correction that is recorded rather than quietly fixed. The first
version of the sweep drew its pool with the raw seed; `scripts/06_best_of_n.py` draws with
`seed * 1000`. The resulting mismatch was 0.008 and initially looked like a reproducibility
failure in the *published* artefact. It was a bug in the checker. Three independently
generated best-of-N runs (`c23_`, `c18_`, `pilot_50k_p2_`) all give 0.808594 / 401.0938 at
seed 101; the published artefact was never in doubt.

### C26.2.2 Gate 2 — the estimator, and why the pre-registered one was replaced

C26.0.2 pre-registered a **nested** estimator: draw 512 slots of 32, and at each N select
over the first N of each slot. C26.0.3 gate 2 required its N=9 point to agree with gate 1
within the between-seed sd, and required the estimator to be **replaced rather than
caveated** if it did not.

It did not:

| anchor | nested N=9 − published | between-seed sd | exceeds sd? |
|---|---|---|---|
| aromatic_rings | −0.0195 | 0.0163 | **yes** |
| hbd_count | +0.0209 | 0.0288 | no |
| qed | +0.0384 | 0.0147 | **yes** |

Two of three exceeded it, and the discrepancies are comparable in size to the effect C26
exists to measure. The estimator discarded 70% of the pool at N=9.

The replacement evaluates **all disjoint consecutive groups of N over the whole 16,384-molecule
pool** — 1820 groups at N=9 rather than 512. This makes the first 512 groups *exactly*
`scripts/06_best_of_n.py`'s, so gate 2 stops being a noise comparison and becomes an
**identity**. `first_512_groups_hit_rate` is written into every sweep artefact for this
purpose.

| anchor | published N | seed 101 | seed 202 | seed 303 | max abs residual |
|---|---|---|---|---|---|
| aromatic_rings | 9 | 0.80859375 | 0.830078125 | 0.849609375 | **0.0** |
| hbd_count | 9 | 0.54296875 | 0.52734375 | 0.5 | **0.0** |
| qed | **8** | 0.572265625 | 0.515625 | 0.54296875 | **0.0** |

All nine cells reproduce the published per-seed hit rate exactly. **PASS.**

One trap, recorded because the reviewer fell into it: **the compute-matched N is not 9 for
every anchor.** It is 9 for aromatic rings and HBD count and **8** for QED, because QED's
guided run is cheaper. Checking QED at N=9 makes the gate appear to fail by ~0.05. The
summariser now reads the published N out of each artefact rather than assuming it.

The estimator change is not cosmetic. On 1820 groups, best-of-9 for aromatic rings is
**0.8150**, against the published 512-group **0.8294** — the published comparator was
optimistic by 0.0144, inside its own noise but in the direction that *favours* guidance.
Every comparison below uses the more precise estimate, which is the conservative choice
against C26's own headline.

### C26.2.3 Gate 3 — N=1 is the base policy

N=1 must reproduce the published unguided hit rate.

| anchor | C26 N=1 (16,384 mol/seed) | published unguided (§16.1) | difference |
|---|---|---|---|
| aromatic_rings | 0.1704 (sd 0.0054) | 0.17855 (sd 0.0031) | −0.0081 |
| hbd_count | 0.0847 (sd 0.0013) | 0.08371 (sd 0.0078) | +0.0010 |
| qed | 0.0955 (sd 0.0015) | 0.08960 (sd 0.0112) | +0.0059 |

Two of three sit well inside the published run's own between-seed spread. Aromatic rings
differs by −0.0081, about 1.5× C26's between-seed sd, and is **reported as a discrepancy
rather than waved through**: the two estimates use different denominators — `summarise`
divides hits by molecules that RDKit parsed *and* that have the property, while the §16.1
figure is a valid-molecule rate over a 1536-molecule run — and C26's N=1 value (0.1704) is
closer to the dataset base rate recorded in the frozen intervals (0.171387) than the
published figure is. This gate passes on the criterion as written ("within seed noise") but
it is the weakest of the three and nothing below turns on it.

---

## C26.3 The best-of-N curves

Hit rate is the mean over three seeds; tokens per molecule is `actual` accounting.

### aromatic_rings

| N | hit rate | sd | tokens/mol | validity | uniqueness |
|---|---|---|---|---|---|
| 1 | 0.1704 | 0.0054 | 44.4 | 0.9963 | 0.9999 |
| 2 | 0.3105 | 0.0097 | 88.9 | 1.0000 | 1.0000 |
| 3 | 0.4302 | 0.0122 | 133.3 | 1.0000 | 0.9999 |
| 4 | 0.5258 | 0.0122 | 177.7 | 1.0000 | 1.0000 |
| 6 | 0.6761 | 0.0104 | 266.6 | 1.0000 | 1.0000 |
| 8 | 0.7751 | 0.0133 | 355.5 | 1.0000 | 1.0000 |
| 9 | 0.8150 | 0.0145 | 399.9 | 1.0000 | 1.0000 |
| 12 | 0.8952 | 0.0191 | 533.2 | 1.0000 | 1.0000 |
| 16 | 0.9535 | 0.0135 | 711.0 | 1.0000 | 1.0000 |
| 24 | 0.9888 | 0.0081 | 1066.5 | 1.0000 | 1.0000 |
| 32 | 0.9961 | 0.0020 | 1422.0 | 1.0000 | 1.0000 |

### hbd_count

| N | hit rate | sd | tokens/mol | validity | uniqueness |
|---|---|---|---|---|---|
| 1 | 0.0847 | 0.0013 | 44.4 | 0.9963 | 0.9999 |
| 2 | 0.1617 | 0.0022 | 88.9 | 1.0000 | 1.0000 |
| 3 | 0.2327 | 0.0025 | 133.3 | 1.0000 | 0.9999 |
| 4 | 0.2975 | 0.0029 | 177.7 | 1.0000 | 1.0000 |
| 6 | 0.4105 | 0.0048 | 266.6 | 1.0000 | 0.9999 |
| 8 | 0.5042 | 0.0054 | 355.5 | 1.0000 | 1.0000 |
| 9 | 0.5447 | 0.0030 | 399.9 | 1.0000 | 1.0000 |
| 12 | 0.6474 | 0.0038 | 533.2 | 1.0000 | 1.0000 |
| 16 | 0.7520 | 0.0049 | 711.0 | 1.0000 | 1.0000 |
| 24 | 0.8744 | 0.0061 | 1066.5 | 1.0000 | 1.0000 |
| 32 | 0.9271 | 0.0081 | 1422.0 | 1.0000 | 1.0000 |

### qed

| N | hit rate | sd | tokens/mol | validity | uniqueness |
|---|---|---|---|---|---|
| 1 | 0.0955 | 0.0015 | 44.4 | 0.9963 | 0.9999 |
| 2 | 0.1810 | 0.0030 | 88.9 | 1.0000 | 0.9999 |
| 3 | 0.2588 | 0.0056 | 133.3 | 1.0000 | 0.9999 |
| 4 | 0.3280 | 0.0054 | 177.7 | 1.0000 | 1.0000 |
| 6 | 0.4521 | 0.0096 | 266.6 | 1.0000 | 1.0000 |
| 8 | 0.5500 | 0.0079 | 355.5 | 1.0000 | 1.0000 |
| 9 | 0.5965 | 0.0097 | 399.9 | 1.0000 | 1.0000 |
| 12 | 0.6994 | 0.0096 | 533.2 | 1.0000 | 1.0000 |
| 16 | 0.7923 | 0.0108 | 711.0 | 1.0000 | 1.0000 |
| 24 | 0.9032 | 0.0051 | 1066.5 | 1.0000 | 1.0000 |
| 32 | 0.9603 | 0.0030 | 1422.0 | 1.0000 | 1.0000 |

### C26.3.1 Concavity, and a statistic that had to be thrown away

The curves are strictly concave in N. The secant slopes per unit N decrease monotonically at
every step, on all three anchors:

| anchor | secant slopes, N = 1→2→3→4→6→8→9→12→16→24→32 |
|---|---|
| aromatic_rings | 0.1401, 0.1197, 0.0956, 0.0751, 0.0495, 0.0400, 0.0267, 0.0146, 0.0044, 0.0009 |
| hbd_count | 0.0770, 0.0710, 0.0648, 0.0565, 0.0469, 0.0405, 0.0342, 0.0261, 0.0153, 0.0066 |
| qed | 0.0856, 0.0778, 0.0692, 0.0621, 0.0489, 0.0466, 0.0343, 0.0232, 0.0139, 0.0071 |

The first version of the summariser tested concavity with the textbook second difference
`h[i+1] − 2h[i] + h[i−1]` and reported all three curves **non-concave**. That statistic is
invalid here: the grid 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32 is not uniform, and its spacings
run from 1 to 8. The correct discrete test on an unequal grid is that the divided
differences are non-increasing, and by that test every curve is concave without exception.
Both statistics are written to `c26_metrics.json` — the invalid one under the key
`curve_second_differences_uniform_grid_invalid` — so the discarded number is visible rather
than silently replaced.

---

## C26.4 The frontier: 46 guidance arms priced against the curve

Each arm's `advantage` is its hit rate minus the best-of-N curve interpolated **linearly in
tokens** at that arm's own exact realised budget, with the bracketing grid points published
so the interpolation can be checked. Positive means guidance is above the curve.

### C26.4.1 aromatic_rings — 19 arms, 0 above the curve

| run | family | λ | layer | guided | tokens/mol | best-of-N @ budget | bracket | advantage |
|---|---|---|---|---|---|---|---|---|
| `c23_guided_L3_lam2_aromatic_rings` | c23_mid_layer | 2.0 | 3 | 0.8159 | 447.5 | 0.8436 | 9–12 | -0.0277 |
| `c23_guided_L6_lam2_aromatic_rings` | c23_mid_layer | 2.0 | 6 | 0.6842 | 462.8 | 0.8528 | 9–12 | -0.1687 |
| `c23_guided_L6_lam1_aromatic_rings` | c23_mid_layer | 1.0 | 6 | 0.5831 | 422.8 | 0.8288 | 9–12 | -0.2457 |
| `c23_guided_L3_lam1_aromatic_rings` | c23_mid_layer | 1.0 | 3 | 0.5698 | 412.5 | 0.8226 | 9–12 | -0.2528 |
| `pilot_50k_p2_lam2_guided_aromatic_rings` | section19_lambda_sweep | 2.0 | -1 | 0.5579 | 420.3 | 0.8273 | 9–12 | -0.2694 |
| `c18_guided_binT0p4_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.5495 | 426.3 | 0.8309 | 9–12 | -0.2813 |
| `c18_guided_head_wide_focused_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.5007 | 413.3 | 0.8230 | 9–12 | -0.3224 |
| `pilot_50k_p2_lam4_guided_aromatic_rings` | section19_lambda_sweep | 4.0 | -1 | 0.4989 | 415.3 | 0.8243 | 9–12 | -0.3254 |
| `c18_guided_uncalibrated_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.4735 | 419.3 | 0.8267 | 9–12 | -0.3532 |
| `pilot_50k_p2_guided_aromatic_rings` | deployed_lambda1 | 1.0 | -1 | 0.4735 | 419.3 | 0.8267 | 9–12 | -0.3532 |
| `c18_guided_head_wide_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.4532 | 409.0 | 0.8205 | 9–12 | -0.3673 |
| `c23_guided_L6_lam0p5_aromatic_rings` | c23_mid_layer | 0.5 | 6 | 0.3881 | 409.3 | 0.8207 | 9–12 | -0.4326 |
| `pilot_50k_p2_lam8_guided_aromatic_rings` | section19_lambda_sweep | 8.0 | -1 | 0.3993 | 455.8 | 0.8486 | 9–12 | -0.4493 |
| `c23_guided_L3_lam0p5_aromatic_rings` | c23_mid_layer | 0.5 | 3 | 0.3616 | 405.8 | 0.8186 | 9–12 | -0.4570 |
| `c18_guided_bin_temperature_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.3436 | 412.4 | 0.8225 | 9–12 | -0.4790 |
| `pilot_50k_p2_lam0p5_guided_aromatic_rings` | section19_lambda_sweep | 0.5 | -1 | 0.3011 | 408.3 | 0.8201 | 9–12 | -0.5190 |
| `c18_guided_isotonic_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.3009 | 409.4 | 0.8207 | 9–12 | -0.5199 |
| `c18_guided_platt_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.2465 | 404.8 | 0.8179 | 9–12 | -0.5714 |
| `pilot_50k_p2_lam0p25_guided_aromatic_rings` | section19_lambda_sweep | 0.25 | -1 | 0.2359 | 405.6 | 0.8184 | 9–12 | -0.5825 |

### C26.4.2 hbd_count — 18 arms, 1 above the curve

| run | family | λ | layer | guided | tokens/mol | best-of-N @ budget | bracket | advantage |
|---|---|---|---|---|---|---|---|---|
| `c23_guided_L4_lam2_hbd_count` | c23_mid_layer | 2.0 | 4 | 0.5603 | 387.8 | 0.5336 | 8–9 | **+0.0267** |
| `c23_guided_L6_lam2_hbd_count` | c23_mid_layer | 2.0 | 6 | 0.4684 | 399.1 | 0.5440 | 8–9 | -0.0756 |
| `pilot_50k_p2_lam2_guided_hbd_count` | section19_lambda_sweep | 2.0 | -1 | 0.4303 | 393.2 | 0.5385 | 8–9 | -0.1082 |
| `pilot_50k_p2_lam4_guided_hbd_count` | section19_lambda_sweep | 4.0 | -1 | 0.3730 | 393.7 | 0.5390 | 8–9 | -0.1660 |
| `c23_guided_L4_lam1_hbd_count` | c23_mid_layer | 1.0 | 4 | 0.3689 | 394.3 | 0.5395 | 8–9 | -0.1706 |
| `c23_guided_L6_lam1_hbd_count` | c23_mid_layer | 1.0 | 6 | 0.3395 | 398.6 | 0.5435 | 8–9 | -0.2040 |
| `pilot_50k_p2_lam8_guided_hbd_count` | section19_lambda_sweep | 8.0 | -1 | 0.3104 | 406.1 | 0.5495 | 9–12 | -0.2390 |
| `c18_guided_head_wide_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2997 | 401.2 | 0.5457 | 9–12 | -0.2460 |
| `c18_guided_uncalibrated_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2988 | 401.6 | 0.5460 | 9–12 | -0.2472 |
| `pilot_50k_p2_guided_hbd_count` | deployed_lambda1 | 1.0 | -1 | 0.2988 | 401.6 | 0.5460 | 9–12 | -0.2472 |
| `c18_guided_head_wide_focused_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2847 | 400.4 | 0.5451 | 9–12 | -0.2604 |
| `c23_guided_L4_lam0p5_hbd_count` | c23_mid_layer | 0.5 | 4 | 0.2319 | 397.0 | 0.5420 | 8–9 | -0.3101 |
| `c23_guided_L6_lam0p5_hbd_count` | c23_mid_layer | 0.5 | 6 | 0.2027 | 394.4 | 0.5397 | 8–9 | -0.3370 |
| `c18_guided_isotonic_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2128 | 407.5 | 0.5505 | 9–12 | -0.3376 |
| `pilot_50k_p2_lam0p5_guided_hbd_count` | section19_lambda_sweep | 0.5 | -1 | 0.1813 | 395.9 | 0.5410 | 8–9 | -0.3597 |
| `c18_guided_platt_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.1567 | 395.3 | 0.5405 | 8–9 | -0.3838 |
| `pilot_50k_p2_lam0p25_guided_hbd_count` | section19_lambda_sweep | 0.25 | -1 | 0.1324 | 393.5 | 0.5388 | 8–9 | -0.4064 |
| `c18_guided_bin_temperature_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.1280 | 393.0 | 0.5383 | 8–9 | -0.4103 |

### C26.4.3 qed — 9 arms, 0 above the curve

| run | family | λ | layer | guided | tokens/mol | best-of-N @ budget | bracket | advantage |
|---|---|---|---|---|---|---|---|---|
| `c23_guided_L4_lam4_qed` | c23_mid_layer | 4.0 | 4 | 0.3092 | 375.3 | 0.5707 | 8–9 | -0.2616 |
| `c23_guided_L4_lam2_qed` | c23_mid_layer | 2.0 | 4 | 0.2796 | 366.1 | 0.5611 | 8–9 | -0.2815 |
| `pilot_50k_p2_lam4_guided_qed` | section19_lambda_sweep | 4.0 | -1 | 0.2607 | 371.7 | 0.5670 | 8–9 | -0.3063 |
| `pilot_50k_p2_lam2_guided_qed` | section19_lambda_sweep | 2.0 | -1 | 0.2361 | 361.9 | 0.5567 | 8–9 | -0.3206 |
| `c23_guided_L4_lam1_qed` | c23_mid_layer | 1.0 | 4 | 0.2288 | 367.8 | 0.5629 | 8–9 | -0.3341 |
| `pilot_50k_p2_lam8_guided_qed` | section19_lambda_sweep | 8.0 | -1 | 0.2438 | 423.5 | 0.6147 | 9–12 | -0.3709 |
| `pilot_50k_p2_guided_qed` | deployed_lambda1 | 1.0 | -1 | 0.1908 | 367.3 | 0.5623 | 8–9 | -0.3715 |
| `pilot_50k_p2_lam0p5_guided_qed` | section19_lambda_sweep | 0.5 | -1 | 0.1692 | 376.2 | 0.5716 | 8–9 | -0.4025 |
| `pilot_50k_p2_lam0p25_guided_qed` | section19_lambda_sweep | 0.25 | -1 | 0.1332 | 381.6 | 0.5773 | 8–9 | -0.4441 |

### C26.4.4 Guidance is a vertical line, not a frontier

| anchor | arms | guidance tokens/mol, min → max | spread | best-of-N span over the grid |
|---|---|---|---|---|
| aromatic_rings | 19 | 404.8 → 462.8 | 58.0 (14.3%) | 44.4 → 1422.0 (**32.0×**) |
| hbd_count | 18 | 387.8 → 407.5 | 19.7 (5.1%) | 44.4 → 1422.0 (**32.0×**) |
| qed | 9 | 361.9 → 423.5 | 61.6 (17.0%) | 44.4 → 1422.0 (**32.0×**) |

Every knob the project has turned — λ over 0.25 to 8, Platt, isotonic, binned, bin
temperature, two head widths, probe layers 3/4/6/12 — moves the *hit rate* over a range of
0.24 to 0.82 while moving the *cost* by at most 17%. λ is not a compute control; it is an
accuracy control at fixed cost. The one thing that does move cost is decoding longer
molecules, and that is a confound, not a knob.

This has a consequence the earlier sections did not state. §4 concluded that guidance loses
at matched compute. The frontier shows something stronger: at 400 tokens per molecule the
best guidance arm on HBD count reaches 0.5603 and best-of-N reaches 0.5447, but at 1422
tokens best-of-N reaches 0.9271 and **guidance cannot be run at 1422 tokens at all** within
the method as specified. The comparison is not close at any budget where a practitioner
would actually operate.

---

## C26.5 The pre-registered decision rules and predictions, scored

### C26.5.1 D1 — does best-of-N dominate?

Pre-registered: upheld iff no measured guidance arm sits above the interpolated best-of-N
curve at its own budget, for all three anchors.

**D1 is NOT upheld as literally written.** Exactly one of 46 arms is above the curve:
`c23_guided_L4_lam2_hbd_count`, advantage **+0.0267**, tokens/mol 387.79, bracketed by
N = 8 and 9, validity 0.9935.

Per-seed advantages are **+0.0607, -0.0006, +0.0198** — one seed is on the wrong side of
zero. The seed-level t interval on 2 df (mean **+0.0266**, sd 0.0313,
`t₀.₉₇₅,₂ = 4.302653`) is **[-0.0510, +0.1043]** and does not exclude zero.

**The second pre-registered interval has been withdrawn as vacuous, and this is a
correction to C26's own method, not a caveat on it.** C26.0.5 promised a seed-resampled
bootstrap alongside the t interval, on the reasoning — correct as far as it goes — that
C23's molecule-level resample treated the three seed means as fixed and so ignored the
variance that matters. Resampling seeds fixes *which* variance is described. It does not
survive the sample size:

> At n = 3 the percentile bootstrap of a mean is **identically [min, max]** of the three
> values. The smallest attainable bootstrap mean is the minimum, attained when all three
> resampled indices land on it, with probability 1/27 = **0.0370 > 0.025** — so the 2.5th
> percentile *is* the minimum, and symmetrically the 97.5th *is* the maximum, for any
> three numbers whatsoever.

The computed endpoints are `−0.0006335669255155274` and `+0.06074870995714299`, which are
exactly `min` and `max` of the per-seed advantages. So "the bootstrap CI excludes zero"
carries precisely the information of "all three seeds share a sign" — a three-way sign
test with two-sided null probability **0.25**, which cannot reject anything at any
conventional level. The key `advantage_seed_bootstrap_ci` no longer exists in
`c26_metrics.json`; the quantity is kept under `advantage_seed_sign_test`, carrying
`degenerate_equals_min_max: true`, `sign_test_p_two_sided: 0.25` and the two endpoints
relabelled `degenerate_bootstrap_lo` / `_hi`, because two C26 decision rules were once read
off it and deleting it outright would hide that. It is not used to support any claim in
this section, and **no reader should treat a three-seed percentile bootstrap anywhere in
this project as a confidence interval.** The same correction was applied to C24 on the same
day, where it flipped two of three "excludes zero" verdicts (§C24.8).

This matters beyond C26. C23's Rule B fired on a corrected interval of [+0.0234, +0.1298]
whose half-width implies a standard error of about 0.018 ≈ √(2·0.25/1536) — a
molecule-level binomial interval with the generation seeds held fixed and head seed absent
altogether. The head-seed sd on the same quantity is **0.0513**, roughly three times what
that interval assumed. The interval was not merely optimistic; it was estimating the wrong
variance component, which is why §C26.5.2 below could overturn the arm at all.

### C26.5.2 D2 — the C23 arm, priced properly, and then replicated

Pre-registered: recompute C23's Rule B advantage against the curve interpolated at its exact
budget, reported whatever its sign. That number is **+0.0267**, down from the +0.0369 C23
reported against the published N=9 point. The flooring artefact was worth about 0.010 in
C23's favour, and removing it does not by itself kill the arm.

**What kills it is C25.** The C26 pre-registration prices D2 on one head seed because the
replicates did not exist when it was written. C25 produced two more. Priced on the same
curve at each run's own budget:

| head seed | guided | tokens/mol | best-of-N @ budget | advantage |
|---|---|---|---|---|
| 1234 | 0.5603 | 387.79 | 0.5336 | **+0.0267** |
| 2345 | 0.5601 | 393.80 | 0.5391 | **+0.0210** |
| 3456 | 0.4687 | 387.79 | 0.5336 | **-0.0649** |

Across head seeds: mean **-0.0057**, sd **0.0513**, min -0.0649, max +0.0267, **sign
flips**. The guided hit rate itself spans **0.0916** across head seeds, against a spread of
0.0037 across generation seeds at head seed 3456 — a factor of **25**. The three generation
seeds C23 reported are replicates of the wrong thing.

**Conclusion: C23's Rule B does not survive. The single arm that beat best-of-N was a
head-seed artefact.** §C26.6 records what this obliges the rest of the report to change.

### C26.5.3 D3 — does best-of-N saturate?

| anchor | hit rate at N=32 | gain over the last doubling (16→32) |
|---|---|---|
| aromatic_rings | 0.9961 | +0.0426 |
| hbd_count | 0.9271 | +0.1751 |
| qed | 0.9603 | +0.1680 |

Aromatic rings has effectively saturated — it is at 0.9961 and the last doubling bought
0.0426, most of which was headroom to 1.0. HBD count and QED have **not**: each is still
buying 0.17 per doubling at the top of the grid. Per the pre-registration, no extrapolation
is offered for the two curves that are still rising.

### C26.5.4 The four predictions

| # | prediction | outcome |
|---|---|---|
| 1 | best-of-N rises with N and is concave; falsified if any anchor is convex | **HOLDS**, on all three anchors, by the divided-difference test (§C26.3.1) |
| 2 | best-of-N dominates guidance at every measured budget for aromatic rings and QED | **HOLDS**; 19/19 and 9/9 arms below the curve |
| 3 | for HBD count it does not — C23's L4 λ=2 arm should sit above the curve | **HOLDS as written, then FALSIFIED on replication.** The arm is above the curve at head seed 1234 (+0.0267) and 2345 (+0.0210) and below it at 3456 (−0.0649) |
| 4 | the curves do not cross for aromatic rings anywhere in 1 ≤ N ≤ 32 | **HOLDS**; the closest approach is −0.0277 |

The pre-registration said of prediction 3: *"Prediction 3 is the one I expect to be least
reliable, because it rests on a single arm with one seed near a tie."* That was the right
worry and it was still too optimistic — the instability was not in the generation seeds it
named but in the head seed it did not think to name.

---

## C26.6 What C26 changes in the rest of the report

These are conflicts for the owner to merge, not edits C26 has made. `reports/pilot_report.md`
is untouched.

1. **`reports/section_c23_layer_end_to_end.md` — Rule B must be retired, not annotated.**
   C23's summary row reads `Rule B | fires, one arm, +0.0760`. Priced on the frontier and
   replicated across head seeds it is mean −0.0057 with a sign flip. C25 reaches the same
   verdict by its own pre-committed rule (§C25.0.7, span ≥ 0.05). Rule A — that a
   mid-network head improves guided generation over unguided — is untouched and survives
   both C25's head seeds and C26's pricing.
2. **§19.4 and §4 can be stated more strongly, and should be.** "Guidance loses at matched
   compute" understates it. Guidance has no compute knob: 46 arms inside a 5–17% cost band
   against best-of-N's 32×.
3. **The published best-of-9 comparators are slightly optimistic.** Aromatic rings 0.8294 →
   0.8150 on the 3.6× larger estimator. Wherever the report quotes a best-of-N hit rate as
   the comparator, the more precise value is the one to use, and it moves the comparison in
   guidance's favour, which is why it is adopted.
4. **Nothing here touches C17's or C18's conclusions.** C26 prices guidance arms; it does
   not re-measure probes.

---

## C26.7 Limitations

1. **Three generation seeds and, where it matters most, three head seeds.** The D2b sign
   flip is established from three head seeds. That is enough to demote a single-arm
   positive; it is not enough to bound the head-seed variance. Six to ten head seeds on this
   arm is the cheap follow-up, and it is the follow-up C25 also recommends.
2. **The frontier prices arms that already existed.** C26 generated no guidance runs. Every
   guidance point is at a budget somebody previously chose, so the frontier is dense in
   360–463 tokens per molecule and empty everywhere else. That is a fact about guidance
   (§C26.4.4), but it also means C26 cannot rule out a differently-specified guidance method
   with a real compute knob — for example, resampling or beam search over guided
   continuations. Such a method is not what this project measured.
3. **Linear interpolation in tokens.** The curve is concave, so linear interpolation between
   bracketing grid points *underestimates* best-of-N slightly, biasing every advantage
   upward — again against C26's own headline. The bracketing N values are published so the
   size of this is checkable; brackets are 8–9 or 9–12 for every arm, where the curve is
   close to linear.
4. **One dataset, one generator, six properties, three anchors.** C24 is the external-validity
   check; C26 is not.
5. **`actual` accounting only.** §19 records that `full_recompute` saturates best-of-N to
   1.0000 everywhere, so it cannot discriminate; the choice is stated, not hidden. Under
   `full_recompute` guidance looks far worse, not better.
6. **N=1 gate 3 for aromatic rings is 1.5 sd off** (§C26.2.3) and rests on a
   denominator difference the section states but does not fully reconcile.

---

## REPRODUCE

```bash
# validity gate 1 -- the published call signature
.venv/bin/python scripts/21_n_sweep.py --dataset pilot_50k_p2 --property aromatic_rings \
    --exact-n 9 --out c26_gate_exact_N9_aromatic_rings

# the sweep itself, three anchors
for prop in aromatic_rings hbd_count qed; do
  .venv/bin/python scripts/21_n_sweep.py --dataset pilot_50k_p2 --property "$prop" \
      --n-max 32 --out "c26_nsweep_${prop}"
done

# the frontier: reads existing artefacts only, generates nothing
.venv/bin/python scripts/21_summarise_c26.py

# binding tests
.venv/bin/python -m pytest tests/test_n_sweep.py -q
```

Artefacts: `outputs/c26_prereg/`, `outputs/c26_gate_exact_N9_aromatic_rings/`,
`outputs/c26_nsweep_{aromatic_rings,hbd_count,qed}/`, `outputs/c26_summary/`,
and the superseded `outputs/c26_nsweep_v1_nested_*` with `outputs/c26_nsweep_v1_nested.log`.
