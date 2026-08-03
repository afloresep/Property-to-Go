# Section C27 — the head-selected best-of-N control: is the comparison fair?

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-01. Pre-registration `outputs/c27_prereg/C27.0_preregistration.md`, frozen with its
SHA-256 in `outputs/c27_prereg/prereg_lock.json` before any C27 measurement existed. Every
number below is re-derived from JSON by `tests/test_head_selected_bestofn.py`.

---

## The verdict, up front

**Most of "best-of-N dominates guided decoding" was the oracle, not the selection.**

Best-of-N in this project selects with the true RDKit property of the completed molecule
(`bestofn.selection_key`); guidance only ever sees a learned probe. C27 removes that
asymmetry by making best-of-N select with **the same head the deployed λ=1 guided run steers
with**, evaluated by the same true RDKit hit rate. Three results follow.

1. **The price of ground truth is enormous.** At the compute-matched N=9, oracle selection
   beats head selection by **0.3024** (aromatic rings), **0.2172** (HBD count) and
   **0.3534** (QED). At N=32 the gaps are **0.3724**, **0.4415** and **0.6836**. Head
   selection at N=32 — 32× the base budget, 1422.0 tokens per molecule — **does not reach
   oracle selection at N=9** on any anchor.

2. **C26's decision rule D1 does not survive the correction.** C26 priced 46 guidance arms
   against the oracle-selected curve: 1 sat above it. Priced against the **head-selected**
   curve at the identical budgets, **15 of 46 sit above it** — 6 of 19 on aromatic rings,
   6 of 18 on HBD count, 3 of 9 on QED. The largest margin is
   `c23_guided_L3_lam2_aromatic_rings` at **+0.2915**. E1 is **NOT upheld**, and unlike
   C26's single violation this is not one arm near a tie.

3. **The deployed configuration still loses, but by an order of magnitude less.** The
   pre-specified headline comparison E4 — the deployed λ=1 arm at its own budget — is
   **-0.0439**, **-0.0292**, **-0.0522** against the head-selected curve, where against the
   oracle-selected curve it was -0.3532, -0.2472, -0.3715. Equalising the information
   removes **0.876**, **0.882** and **0.859** of the measured gap. Only QED's seed-level t
   interval excludes zero; with three seeds the other two cannot be resolved.

**What this changes.** The project's headline is not "steering loses to selection". It is
**"steering loses to an oracle, and is competitive with selection at equal information —
but only in configurations that were not the deployed one."** That is a materially better-
scoped claim and a materially weaker negative result.

**What C27 does not show.** It does not show that guidance beats best-of-N in deployment: at
the deployed λ=1 setting it still loses on all three anchors. It does not show that a
practitioner should prefer guidance — a practitioner who has RDKit *has* the oracle, and the
oracle curve is the one that governs. C27 is a statement about what the project measured,
not about what a chemist should run. §C27.8.

---

## C27.0 The pre-registration, verbatim

The block below is `outputs/c27_prereg/C27.0_preregistration.md` from `## C27.0.1 Why`
onward, reproduced in full and unedited.
`tests/test_head_selected_bestofn.py::test_the_report_copies_the_prereg_verbatim` asserts
the copy is a byte-identical substring of this section;
`test_the_prereg_was_written_before_every_measurement` asserts its mtime strictly precedes
every C27 artefact; `test_the_prereg_lock_records_the_prereg_hash` asserts the lock file's
SHA-256 matches.

One thing in it is **wrong and fails**: validity gate 4's file-hash criterion, scored in
§C27.2.4. Two of the six predictions are falsified, scored in §C27.6.

<!-- BEGIN VERBATIM PREREG COPY -->

## C27.0.1 Why

Every headline in this project compares guided decoding against best-of-N at matched
processed tokens, and guided decoding loses. That comparison is **fatally scoped**, and the
scoping defect is in `src/property_to_go/bestofn.py`:

> `selection_key(value, lo, hi)` is evaluated on the **true RDKit property of the completed
> molecule**. Best-of-N therefore selects with the ground-truth oracle. Guidance only ever
> sees a learned probe — a 768→256→256→`n_bins` MLP on a frozen hidden state.

So "best-of-N dominates guidance at matched compute" is, in part, a restatement of "ground
truth beats an estimate of it at equal token cost". That is not news, and no reviewer should
accept a negative result about steering that rests on it.

C27 removes the asymmetry in the only direction that is cheap and honest: it makes
**best-of-N select with the same learned head that guidance steers with**, and evaluates
both arms with the same true RDKit oracle. Selection differs; evaluation does not. The
question becomes *at equal information, does selection still beat steering?*

Both outcomes are publishable and neither is to be forced:

- head-selected best-of-N still beats guidance → the negative result becomes **stronger and
  properly scoped**: the advantage was never the oracle, it was selection-over-commitment;
- guidance beats head-selected best-of-N → C27 is a **positive result** and a better paper.

## C27.0.2 Design

Anchors **aromatic_rings, hbd_count, qed** — the same three as §19, C23 and C26, so every
number is directly comparable. Seeds **101 / 202 / 303**. Dataset `pilot_50k_p2`. Frozen
windows and target intervals, inherited, never re-derived. `actual` token accounting. The
base generator stays frozen; **no head is trained** for C27.

**The pool.** `scripts/22_head_selected_bestofn.py` draws the pool with exactly C26's call,
`sample_unconditional(gen, policy, 32 * 512, seed=seed * 1000)` — note `seed * 1000`, which
is `scripts/06_best_of_n.py`'s convention and the reason C26's gate 1 was an identity. Best-
of-N is evaluated over **all disjoint consecutive groups of N over the whole 16,384-molecule
pool**, not the first N of a slot, which is C26's corrected estimator. The pool depends only
on (seed, policy), so it is byte-identical across the three anchors and identical to C26's.

**N grid, fixed here:** 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32 — C26's grid unchanged.

**The arms.** Over the same groups:

1. `oracle_selected` — `bestofn.selection_key` on the true RDKit property, invalid and
   property-unavailable candidates ranked worst. This is C26's arm verbatim; it exists to be
   a validity gate, not a new measurement.
2. `head_selected` — the group member with the **highest head-predicted probability that the
   finished molecule lands in the target interval**, i.e. `argmax_j TargetScorer(h_j)`, ties
   broken by lowest index (the same first-wins rule `min` gives the oracle arm).
3. `head_selected_at_75pct` — **secondary/diagnostic**, defined here so it cannot be added
   later to rescue a verdict. Identical to arm 2 except the head reads the state at content
   position `max(1, floor(3n/4))` instead of the terminal one. It bounds how much of arm 2's
   performance comes from the head seeing a *complete* molecule, which is the first thing a
   sceptical reviewer will attack (§C27.0.8).

**Arm 2 gets no oracle information whatsoever.** In particular, invalid candidates are
**not** down-ranked in the head arms — RDKit validity is oracle information. A head arm can
therefore return an unparseable molecule that the oracle arm would have rejected. That is a
real cost of not having ground truth and it is reported, not repaired: validity and
`hit_rate_over_all_returned` are published for every arm at every grid point.

**Evaluation is identical for all three arms**: `bestofn.summarise` on the selected
molecule, true RDKit property, frozen interval, `hit_rate` over molecules RDKit parsed and
scored — the same denominator C26 and every guidance run use.

## C27.0.3 Which head, and how a completed molecule is scored

The head must be the one the **deployed λ=1 guided run** used, or the information is not
matched. It is identified from the artefacts, not chosen:
`outputs/pilot_50k_p2_guided_{prop}/guidance_metrics.json` records
`head_input: "frozen_state"` and `head_checkpoint: "head_{prop}_frozen_state.pt"`, and
carries **no** `layer` / `head_file` / `layer_source` key, because it predates the C23 edit
that added them — so the probe point is `05_guided_generation.py`'s default, `layer = -1`,
the final hidden layer. `run_phase2.sh` passes `--heads pilot_50k_heads_p2`. The checkpoint
is therefore `outputs/pilot_50k_heads_p2/head_{prop}_frozen_state.pt`, head seed 1234, and
the binner and interval mask travel inside the checkpoint, so binning is matched by
construction. If any part of this fails to resolve unambiguously, C27 **stops and reports
that** rather than guessing.

**Scoring a completed molecule with a head trained on prefixes.** The head predicts, from
the state at a *partial* molecule, whether the *finished* molecule lands in the interval. A
completed sequence is the limiting case of that: `prefixes.select_quartile_prefixes` draws
one prefix per quartile of `[1, n]`, and quartile 4's upper bound is exactly `n` — the whole
content. Reading the state at the **last content token** (full-sequence index `n`, since
index 0 is `<bos>`) is therefore the `relative_position = 1.0` endpoint of the head's own
training distribution, not an extrapolation off it. The `<eos>` position is deliberately
**not** used: no training prefix ever ended at `<eos>`, so scoring there would be
out-of-distribution. Empty-content sequences fall back to index 0.

This choice has a consequence stated here rather than discovered later: at the terminal
position the head is being asked its easiest possible question, so `head_selected` may come
close to `oracle_selected`. If it does, the honest reading is **not** "the head is secretly
the oracle" but "the head has the property information once the molecule exists; what
guidance lacks is that information *early enough to act on*". §C27.0.8.

## C27.0.4 Token accounting

`actual`, consistent with C26. Generation cost is identical for every arm because every arm
selects from the same pool: `oracle_selected` and `head_selected` must report **the same**
tokens per returned molecule, matching C26's 44.4 at N=1 up to 1422.0 at N=32.

**Head scoring is charged zero generator tokens, and this is a claim, not an oversight.**
The head's input is `hidden_states[-1]` at a position the generator has *already* computed:
the final-layer state at position `t` is exactly what the LM head reads to emit the logits
for position `t+1`, so it is materialised by the forward pass that generated the molecule.
Reading it costs one ~0.4-MFLOP MLP against a 47M-parameter transformer step. A deployed
head-selecting sampler pays nothing extra.

Our *implementation* does recompute those states, with one extra forward pass over the pool,
because `transformers`' `generate()` does not expose hidden states. That is an engineering
artefact, not a property of the method. It is measured anyway and published as
`head_scoring_recompute_tokens_per_molecule`, and §C27.0.5 pre-registers the pessimistic
sensitivity that charges it.

## C27.0.5 Validity gates, checked before any curve is read

1. **Oracle identity.** `oracle_selected` must reproduce
   `outputs/c26_nsweep_{prop}/n_sweep_metrics.json` **exactly** — every grid point, every
   seed, hit rate and tokens per molecule. Residual reported as a number, never as the word
   "matches". If it is not `0.0`, the pool differs from C26's, C27 stops and diagnoses.
2. **Head AUROC on the pool.** Per anchor and per seed, the AUROC of the head's terminal-
   position `P(y_final ∈ I)` for discriminating true hits among the 16,384 pool molecules,
   reported as a number. **If it is near 0.5 the head arm is measuring nothing and the
   section must say so**, and no comparison against guidance may be drawn from it. Reported
   for arm 3's 75%-position score as well.
3. **N=1 identity.** At N=1 there is one candidate and nothing to select, so all three arms
   must be bit-identical to each other and to C26. A difference is a bug alarm.
4. **Head provenance.** The SHA-256 of the loaded checkpoint must equal that of
   `head_{prop}_frozen_state_seed1234.pt`, and the checkpoint's recorded `input`,
   `head_seed`, `n_bins` and binner must match `head_metrics.json`. Published as hashes.
5. **Token identity between arms.** Reported as a residual, expected `0.0` by construction.

**Sensitivity S1, pre-registered.** If head scoring is charged a full re-read of every
candidate — the pessimistic accounting, which we argue is wrong — the head arm's budget
rises by the measured recompute cost. The head-selected curve is re-priced under that
accounting and reported as a secondary curve. Any C27 verdict that survives only under the
free accounting is labelled as such.

## C27.0.6 Decision rules

- **E1 — does selection still beat steering at equal information?** Every one of the 46
  existing guidance arms (19 aromatic_rings, 18 hbd_count, 9 qed, glob-derived by
  `scripts/21_summarise_c26.py::guidance_points`) is priced against the **head-selected**
  curve interpolated linearly in tokens at its own realised budget, exactly as C26 priced
  them against the oracle-selected curve. E1 is upheld iff **no** arm sits above the head-
  selected curve, on all three anchors. Every arm above the curve is reported with its
  margin, whether or not it is significant. **No new guidance run is generated.**
- **E2 — the price of not having ground truth.** The gap `oracle_selected − head_selected`
  at every grid point, per anchor: reported at N=9 and N=32, as the maximum over the grid,
  and as an *effective N* — the N at which head selection reaches the hit rate oracle
  selection reaches at N=9. This is the quantity that says how much of "best-of-N dominates"
  was ever about the oracle.
- **E3 — degeneracy check, which can void E1.** If `head_selected` does not rise
  meaningfully above its own N=1 point — pre-specified as a gain of **< 0.02** absolute hit
  rate from N=1 to N=32 on an anchor — then the head carries no usable ranking signal there,
  the arm measures nothing, and E1 **must not** be read as evidence for guidance on that
  anchor. It is reported as degenerate instead.
- **E4 — the headline single comparison, one per anchor.** The **deployed λ=1** guided arm
  (`pilot_50k_p2_guided_{prop}`) priced against the head-selected curve at its own exact
  budget. Per-seed advantages against each seed's own curve, plus a seed-level t interval on
  2 df. Reported whatever its sign, for all three anchors, including where it is negative.

## C27.0.7 Multiplicity and uncertainty

11 grid points × 3 anchors × 3 arms is descriptive; no per-point test is claimed and no
correction is applied to the curves. E4 is **three** pre-specified comparisons, one per
anchor; all three are reported, so there is no selection to correct for, and no anchor's
result may be quoted without the other two.

**Uncertainty is reported as per-seed values and a seed-level t interval on 2 df**
(`t₀.₉₇₅,₂ = 4.302653`). **No bootstrap is computed anywhere in C27.** At n = 3 the
percentile bootstrap of a mean is identically `[min, max]`: the smallest attainable
bootstrap mean is the minimum, attained when all three resampled indices hit it, with
probability 1/27 = 0.0370 > 0.025. Such an interval carries exactly the information of a
three-way sign test at null probability 0.25 and cannot reject anything. C24 and C26 are
having theirs removed concurrently; C27 does not introduce a new one. Where n = 3 cannot
support an inference, the section **says so plainly** instead of decorating the number.

## C27.0.8 The attack this design invites, stated before the result

A sceptical reviewer will say: *scoring the head at the terminal state gives best-of-N a
near-oracle, so you have not matched information at all — guidance only ever sees partial
prefixes.* The pre-registered answer, which the design must be judged against:

1. Guidance's head also sees the near-complete prefix, at the last guided step. Nothing is
   withheld from guidance that arm 2 is given.
2. The residual asymmetry is therefore **not** about the property oracle. It is that
   selection may act *after* the outcome exists while steering must commit *before* it. That
   is the real thing being measured, and C27's whole point is to isolate it.
3. Arm 3 (`head_selected_at_75pct`) quantifies how much of arm 2 depends on the molecule
   being finished, using the same head and no extra compute. It is registered here, before
   any number is seen, precisely so it cannot be produced afterwards as a rescue.

C27 does **not** claim to match the *timing* of information, only its source. That
limitation is stated in the section, not in a footnote.

## C27.0.9 Predictions

1. `head_selected` lies strictly between the N=1 base rate and `oracle_selected` at every
   N ≥ 2 on all three anchors. Falsified below N=1 (head useless) or above the oracle (a
   bug: the oracle maximises the evaluated quantity by construction, up to ties).
2. **`head_selected` still beats the deployed λ=1 guidance arm at its own budget on all
   three anchors** (E4 negative for guidance everywhere). Reasoning: the deployed arms sit
   0.25 / 0.25 / 0.37 below the *oracle* curve, and a selector with pooled held-out target
   AUROC of 0.79 / 0.78 / 0.74 — higher still at the terminal position — should not give
   back a quarter of a hit rate. This is the prediction that decides whether C27 strengthens
   the negative result or overturns it, and I expect it to hold.
3. **The single sub-prediction that can fail independently:** the best guidance arm on
   hbd_count, `c23_guided_L4_lam2_hbd_count` (0.5603 at 387.79 tokens/molecule, the only arm
   of 46 above C26's oracle curve), sits **above** the head-selected curve too. If it does,
   E1 fails on one arm again and the properly-scoped negative result has exactly the same
   single exception as the badly-scoped one — which would be worth stating plainly, given
   C26 already showed that arm does not survive head-seed replication.
4. The oracle-minus-head gap (E2) is **largest for qed and smallest for aromatic_rings**,
   because qed's head has the lowest held-out target AUROC (0.7355 vs 0.7904) and must
   spread mass over 22 bins rather than 6, three of which form the target.
5. The head's terminal-position pool AUROC (gate 2) **exceeds** the pooled held-out AUROC in
   `head_metrics.json` (0.7904 / 0.7781 / 0.7355) on all three anchors, because the pooled
   figure averages over four position quartiles and the terminal position is the easiest.
6. Arm 3 (75% position) lands strictly between arm 2 and the N=1 base rate on all three
   anchors — i.e. some but not all of the head's selection power requires the finished
   molecule.

Prediction 3 is the one I expect to be least reliable, for the same reason C26's prediction 3
was: it rests on a single arm whose per-seed advantages against C26's oracle curve were
+0.0607 / −0.0006 / +0.0198, one of them on the wrong side of zero, and whose sign flips
across head-training seeds.

<!-- END VERBATIM PREREG COPY -->

---

## C27.1 What was run

| stage | script | output | cost |
|---|---|---|---|
| pre-registration | — | `outputs/c27_prereg/C27.0_preregistration.md`, `prereg_lock.json` | — |
| three-arm sweep | `scripts/22_head_selected_bestofn.py` | `outputs/c27_headsel_{aromatic_rings,hbd_count,qed}/`, log `outputs/c27_headsel.log` | 3 anchors x 3 seeds x 32x512 draws + 1 state pass |
| frontier assembly | `scripts/22_summarise_c27.py` | `outputs/c27_summary/c27_metrics.json` | reads only; generates nothing |

`scripts/22_summarise_c27.py` **generates no molecules**, and C27 **trains no head**. Every
guidance point priced below comes from an artefact that existed before C27 began; the
frontier machinery (`guidance_points`, `interp`, `t_interval`) is *imported* from
`scripts/21_summarise_c26.py` rather than copied, so the C26 and C27 frontiers differ only
in which best-of-N curve the arms are priced against. `scripts/21_n_sweep.py::score_pool` is
likewise imported rather than re-implemented, which is what makes gate 1 an identity instead
of an agreement.

No existing output directory, script, `reports/pilot_report.md` or `README.md` was modified.

---

## C27.2 Validity gates, checked before any curve was read

### C27.2.1 Gate 1 — the oracle arm is C26's, exactly

`oracle_selected` must reproduce `outputs/c26_nsweep_{prop}/n_sweep_metrics.json` at every
grid point and every seed. The residual as a number, over all **99** cells (11 grid points x
3 seeds x 3 anchors):

| anchor | cells | max abs hit-rate residual | max abs tokens/molecule residual |
|---|---|---|---|
| aromatic_rings | 33 | **0.0** | **0.0** |
| hbd_count | 33 | **0.0** | **0.0** |
| qed | 33 | **0.0** | **0.0** |

Bit-identical on hit rate *and* token cost in all 99 cells. **PASS.** The pool is C26's;
nothing below is comparing across different draws.

### C27.2.2 Gate 2 — does the head discriminate anything?

AUROC of the head's `P(y_final ∈ I)` for separating true hits among the 16,384 pool
molecules, per seed. If this were near 0.5 the head arm would be measuring nothing and every
comparison below would be void.

| anchor | seed 101 | seed 202 | seed 303 | mean | 75%-position mean | pooled held-out (`head_metrics.json`) |
|---|---|---|---|---|---|---|
| aromatic_rings | 0.8705 | 0.8646 | 0.8722 | 0.8691 | 0.8615 | 0.7904 |
| hbd_count | 0.8789 | 0.8709 | 0.8742 | 0.8747 | 0.8346 | 0.7781 |
| qed | 0.8103 | 0.7993 | 0.8115 | 0.8070 | 0.7796 | 0.7355 |

Nowhere near chance. **PASS.** The head is a genuinely informative ranker of finished
molecules — which is what makes the size of the E2 gap in §C27.4 the interesting number
rather than a trivial one.

### C27.2.3 Gate 3 — N=1 is an identity across arms

At N=1 each group has one candidate and there is nothing to select, so all three arms must
agree exactly. Max absolute residual across all anchors, seeds and arms: **0.0**. **PASS.**
This is a bug alarm on the selection code, and it fired clean.

### C27.2.4 Gate 4 — the head is the deployed one, and the pre-registered criterion FAILS

The head resolved from the deployed artefacts, with no ambiguity anywhere:

| anchor | checkpoint | head seed | probe layer | n_bins | target bins | binner | parameter SHA-256 (first 16) |
|---|---|---|---|---|---|---|---|
| aromatic_rings | `head_aromatic_rings_frozen_state.pt` | 1234 | -1 | 6 | 1 | categorical | `bc1509dd11665d93` |
| hbd_count | `head_hbd_count_frozen_state.pt` | 1234 | -1 | 7 | 1 | categorical | `20f687debaf80ff0` |
| qed | `head_qed_frozen_state.pt` | 1234 | -1 | 22 | 3 | quantile | `fc0be3df33438a5c` |

How it was verified it is the deployed one, in order of strength:

1. `outputs/pilot_50k_p2_guided_{prop}/guidance_metrics.json` records
   `head_input: "frozen_state"` and `head_checkpoint: "head_{prop}_frozen_state.pt"`, and
   contains **no** `layer`, `layer_source`, `head_file` or `head_file_source` key — those
   keys were added to `scripts/05_guided_generation.py` by C23, after the deployed run.
   Their absence pins the probe layer to the pre-C23 default, `-1`.
2. `scripts/run_phase2.sh` runs script 05 with `--heads pilot_50k_heads_p2` and no
   `--head-seed`, and script 05's default is `heads_dir/head_<prop>_frozen_state.pt`.
3. `scripts/03_train_heads.py` writes the unsuffixed name only for `rank == 0` of
   `--head-seeds 1234 2345 3456`, so the unsuffixed checkpoint *is* head seed 1234; the
   loaded checkpoint's own `head_seed` field reads 1234, and `head_metrics.json` agrees.
4. The binner and the interval mask travel inside the checkpoint, so C27's binning is the
   deployed binning by construction, not by re-derivation.

**The pre-registered form of gate 4 fails, on all three anchors, and it is scored as a
failure rather than waived.** C27.0.5 required the loaded checkpoint to be **byte-identical
(SHA-256)** to `head_{prop}_frozen_state_seed1234.pt`. It is not:

| anchor | file bytes, unsuffixed | file bytes, seed-1234 twin | file SHA-256 equal? |
|---|---|---|---|
| aromatic_rings | 1066796 | 1066904 | **no** |
| hbd_count | 1067696 | 1067868 | **no** |
| qed | 1083624 | 1083796 | **no** |

The cause is not a content difference. `torch.save` writes a zip whose internal archive
directory is named after the output file, so saving one `ckpt` dict twice under two
different names produces two files that differ in bytes and in length while holding
identical tensors. **The criterion tested serialisation, not content.** It was replaced, in
the direction that makes the gate stricter about what it was meant to check: every parameter
tensor, every metadata field and the binner dict are compared element-wise against the
seed-1234 twin.

| criterion | result |
|---|---|
| parameter tensors identical to seed-1234 twin | **yes**, all three anchors |
| metadata (`in_dim`, `hidden_dim`, `n_bins`, `dropout`, `property`, `input`, `head_seed`) identical | **yes**, all three anchors |
| binner dict identical | **yes**, all three anchors |
| probe layer is -1 | **yes**, all three anchors |
| *pre-registered file-SHA-256 criterion* | ***no***, all three anchors |

The failed criterion is retained in `c27_metrics.json` under
`prereg_criterion_file_sha256_equal` so the discarded test stays visible. Gate 4 **passes on
content and fails as written**, and a reader who thinks a file hash was the right test should
know it was tried.

### C27.2.5 Gate 5 — both arms cost the same, and the cost is C26's

All three arms select from one pool, so tokens per returned molecule must be identical.
Max absolute residual between arms, all anchors, all grid points: **0.0**. The cost runs
from **44.4** tokens per returned molecule at N=1 to **1422.0** at N=32, matching C26
exactly. **PASS.**

Head scoring adds **zero** generator tokens to the headline accounting, for the reason given
in C27.0.4: the state the head reads is the one the LM head already consumed to emit the
next token. §C27.7 prices the pessimistic alternative.

---

## C27.3 The two curves

Hit rate is the mean over three seeds, `actual` accounting, evaluated by the true RDKit
property on the selected molecule in every arm. "agreement with oracle pick" is the fraction
of groups where the head arm selected the same pool index the oracle arm did.

### aromatic_rings

| N | tokens/mol | oracle-selected | sd | head-selected | sd | head @75% | oracle-head gap | head validity | agreement with oracle pick |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 44.4 | 0.1704 | 0.0054 | 0.1704 | 0.0054 | 0.1704 | 0.0000 | 0.9963 | 1.0000 |
| 2 | 88.9 | 0.3105 | 0.0097 | 0.2759 | 0.0090 | 0.2743 | 0.0347 | 0.9961 | 0.7544 |
| 3 | 133.3 | 0.4302 | 0.0122 | 0.3474 | 0.0097 | 0.3456 | 0.0828 | 0.9963 | 0.6457 |
| 4 | 177.7 | 0.5258 | 0.0122 | 0.3980 | 0.0057 | 0.3956 | 0.1278 | 0.9963 | 0.5799 |
| 6 | 266.6 | 0.6761 | 0.0104 | 0.4627 | 0.0033 | 0.4679 | 0.2134 | 0.9969 | 0.4741 |
| 8 | 355.5 | 0.7751 | 0.0133 | 0.5063 | 0.0102 | 0.5119 | 0.2688 | 0.9976 | 0.4106 |
| 9 | 399.9 | 0.8150 | 0.0145 | 0.5126 | 0.0135 | 0.5275 | 0.3024 | 0.9976 | 0.3667 |
| 12 | 533.2 | 0.8952 | 0.0191 | 0.5457 | 0.0134 | 0.5617 | 0.3496 | 0.9976 | 0.2955 |
| 16 | 711.0 | 0.9535 | 0.0135 | 0.5743 | 0.0176 | 0.5980 | 0.3792 | 0.9987 | 0.2357 |
| 24 | 1066.5 | 0.9888 | 0.0081 | 0.6049 | 0.0201 | 0.6383 | 0.3839 | 0.9995 | 0.1579 |
| 32 | 1422.0 | 0.9961 | 0.0020 | 0.6237 | 0.0141 | 0.6741 | 0.3724 | 1.0000 | 0.1159 |

### hbd_count

| N | tokens/mol | oracle-selected | sd | head-selected | sd | head @75% | oracle-head gap | head validity | agreement with oracle pick |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 44.4 | 0.0847 | 0.0013 | 0.0847 | 0.0013 | 0.0847 | 0.0000 | 0.9963 | 1.0000 |
| 2 | 88.9 | 0.1617 | 0.0022 | 0.1440 | 0.0028 | 0.1364 | 0.0177 | 0.9960 | 0.7325 |
| 3 | 133.3 | 0.2327 | 0.0025 | 0.1872 | 0.0048 | 0.1718 | 0.0456 | 0.9957 | 0.6300 |
| 4 | 177.7 | 0.2975 | 0.0029 | 0.2240 | 0.0057 | 0.2040 | 0.0736 | 0.9953 | 0.5716 |
| 6 | 266.6 | 0.4105 | 0.0048 | 0.2753 | 0.0101 | 0.2475 | 0.1352 | 0.9958 | 0.4907 |
| 8 | 355.5 | 0.5042 | 0.0054 | 0.3173 | 0.0151 | 0.2784 | 0.1870 | 0.9958 | 0.4378 |
| 9 | 399.9 | 0.5447 | 0.0030 | 0.3274 | 0.0141 | 0.2915 | 0.2172 | 0.9962 | 0.4132 |
| 12 | 533.2 | 0.6474 | 0.0038 | 0.3672 | 0.0111 | 0.3244 | 0.2801 | 0.9961 | 0.3548 |
| 16 | 711.0 | 0.7520 | 0.0049 | 0.4028 | 0.0093 | 0.3495 | 0.3491 | 0.9964 | 0.2965 |
| 24 | 1066.5 | 0.8744 | 0.0061 | 0.4507 | 0.0265 | 0.4054 | 0.4237 | 0.9966 | 0.2419 |
| 32 | 1422.0 | 0.9271 | 0.0081 | 0.4855 | 0.0479 | 0.4268 | 0.4415 | 0.9974 | 0.1927 |

### qed

| N | tokens/mol | oracle-selected | sd | head-selected | sd | head @75% | oracle-head gap | head validity | agreement with oracle pick |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 44.4 | 0.0955 | 0.0015 | 0.0955 | 0.0015 | 0.0955 | 0.0000 | 0.9963 | 1.0000 |
| 2 | 88.9 | 0.1810 | 0.0030 | 0.1479 | 0.0046 | 0.1428 | 0.0331 | 0.9988 | 0.7539 |
| 3 | 133.3 | 0.2588 | 0.0056 | 0.1797 | 0.0069 | 0.1740 | 0.0791 | 0.9992 | 0.6345 |
| 4 | 177.7 | 0.3280 | 0.0054 | 0.2023 | 0.0048 | 0.1879 | 0.1257 | 0.9994 | 0.5482 |
| 6 | 266.6 | 0.4521 | 0.0096 | 0.2255 | 0.0101 | 0.2145 | 0.2267 | 0.9998 | 0.4310 |
| 8 | 355.5 | 0.5500 | 0.0079 | 0.2431 | 0.0084 | 0.2284 | 0.3069 | 0.9997 | 0.3561 |
| 9 | 399.9 | 0.5965 | 0.0097 | 0.2431 | 0.0057 | 0.2335 | 0.3534 | 0.9998 | 0.3203 |
| 12 | 533.2 | 0.6994 | 0.0096 | 0.2606 | 0.0151 | 0.2529 | 0.4388 | 0.9998 | 0.2601 |
| 16 | 711.0 | 0.7923 | 0.0108 | 0.2579 | 0.0095 | 0.2588 | 0.5344 | 0.9997 | 0.1979 |
| 24 | 1066.5 | 0.9032 | 0.0051 | 0.2698 | 0.0116 | 0.2670 | 0.6334 | 1.0000 | 0.1246 |
| 32 | 1422.0 | 0.9603 | 0.0030 | 0.2767 | 0.0108 | 0.2786 | 0.6836 | 1.0000 | 0.0944 |

Three things in these tables are worth reading slowly.

**The head-selected curve saturates and the oracle curve does not.** QED is the extreme
case: head selection reaches 0.2431 at N=8 and 0.2767 at N=32, so quadrupling the candidate
pool buys 0.0336, while oracle selection goes 0.5500 → 0.9603 over the same range. Argmax
over a noisy score has a ceiling set by the score's ranking quality; argmax over the truth
does not.

**Selection agreement collapses with N.** At N=2 the head picks the oracle's molecule 73-75%
of the time; at N=32 it is 0.1159 / 0.1927 / 0.0944 against a chance rate of 1/32 = 0.03.
The head is far better than chance at every N and still disagrees with the oracle about
nine times in ten at N=32. That is the mechanism behind the E2 gap.

**The head arm's validity is not damaged by withholding the oracle.** Arm 2 does not
down-rank invalid candidates, so it could have returned unparseable molecules the oracle arm
would have rejected; head validity stays at 0.9953-1.0000 throughout, against the oracle
arm's 0.9963 at N=1 and 1.0000 elsewhere. The cost of not having ground truth shows up
entirely in hit rate, not in validity.

---

## C27.4 E2 — the price of not having ground truth

| anchor | gap at N=9 | gap at N=32 | max gap over the grid | N of max | effective N: head selection matching oracle-at-N=9 |
|---|---|---|---|---|---|
| aromatic_rings | 0.3024 | 0.3724 | 0.3839 | 24 | **never inside 1 ≤ N ≤ 32** (oracle@9 = 0.8150) |
| hbd_count | 0.2172 | 0.4415 | 0.4415 | 32 | **never inside 1 ≤ N ≤ 32** (oracle@9 = 0.5447) |
| qed | 0.3534 | 0.6836 | 0.6836 | 32 | **never inside 1 ≤ N ≤ 32** (oracle@9 = 0.5965) |

No extrapolation is offered, per C27.0.6: the head-selected curves are still rising on two
anchors and no effective N exists inside the measured range on any of them.

This is the single most important number C27 produces. **Head selection at 32 candidates —
1422.0 tokens per returned molecule, 32× the base cost — is worse than oracle selection at 9
candidates on every anchor.** Whatever "best-of-N dominates guidance at matched compute"
means in the rest of the report, at least this much of it is a statement about having RDKit
in the loop rather than about reranking as a strategy.

---

## C27.5 E1 — the 46 guidance arms priced against the head-selected curve

Each arm's advantage is its hit rate minus the head-selected curve interpolated **linearly
in tokens** at that arm's own exact realised budget, with the bracketing grid points
published so the interpolation is checkable by hand. The final column is C26's number — the
same arm priced against the *oracle*-selected curve — so the two frontiers can be read
against each other.

**E1 is NOT upheld.** **15** of **46** arms sit above the head-selected curve, against 1 of
46 above the oracle-selected curve.

### aromatic_rings — 19 arms, 6 above the head-selected curve

| run | family | λ | layer | guided | tokens/mol | head-selected @ budget | bracket | advantage vs head | advantage vs oracle (C26) |
|---|---|---|---|---|---|---|---|---|---|
| `c23_guided_L3_lam2_aromatic_rings` | c23_mid_layer | 2.0 | 3 | 0.8159 | 447.5 | 0.5244 | 9-12 | **+0.2915** | -0.0277 |
| `c23_guided_L6_lam2_aromatic_rings` | c23_mid_layer | 2.0 | 6 | 0.6842 | 462.8 | 0.5282 | 9-12 | **+0.1560** | -0.1687 |
| `c23_guided_L6_lam1_aromatic_rings` | c23_mid_layer | 1.0 | 6 | 0.5831 | 422.8 | 0.5182 | 9-12 | **+0.0648** | -0.2457 |
| `c23_guided_L3_lam1_aromatic_rings` | c23_mid_layer | 1.0 | 3 | 0.5698 | 412.5 | 0.5157 | 9-12 | **+0.0541** | -0.2528 |
| `pilot_50k_p2_lam2_guided_aromatic_rings` | section19_lambda_sweep | 2.0 | -1 | 0.5579 | 420.3 | 0.5177 | 9-12 | **+0.0403** | -0.2694 |
| `c18_guided_binT0p4_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.5495 | 426.3 | 0.5191 | 9-12 | **+0.0304** | -0.2813 |
| `c18_guided_head_wide_focused_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.5007 | 413.3 | 0.5159 | 9-12 | -0.0152 | -0.3224 |
| `pilot_50k_p2_lam4_guided_aromatic_rings` | section19_lambda_sweep | 4.0 | -1 | 0.4989 | 415.3 | 0.5164 | 9-12 | -0.0175 | -0.3254 |
| `c18_guided_uncalibrated_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.4735 | 419.3 | 0.5174 | 9-12 | -0.0439 | -0.3532 |
| `pilot_50k_p2_guided_aromatic_rings` | deployed_lambda1 | 1.0 | -1 | 0.4735 | 419.3 | 0.5174 | 9-12 | -0.0439 | -0.3532 |
| `c18_guided_head_wide_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.4532 | 409.0 | 0.5148 | 9-12 | -0.0616 | -0.3673 |
| `c23_guided_L6_lam0p5_aromatic_rings` | c23_mid_layer | 0.5 | 6 | 0.3881 | 409.3 | 0.5149 | 9-12 | -0.1268 | -0.4326 |
| `pilot_50k_p2_lam8_guided_aromatic_rings` | section19_lambda_sweep | 8.0 | -1 | 0.3993 | 455.8 | 0.5264 | 9-12 | -0.1272 | -0.4493 |
| `c23_guided_L3_lam0p5_aromatic_rings` | c23_mid_layer | 0.5 | 3 | 0.3616 | 405.8 | 0.5141 | 9-12 | -0.1524 | -0.4570 |
| `c18_guided_bin_temperature_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.3436 | 412.4 | 0.5157 | 9-12 | -0.1721 | -0.4790 |
| `pilot_50k_p2_lam0p5_guided_aromatic_rings` | section19_lambda_sweep | 0.5 | -1 | 0.3011 | 408.3 | 0.5147 | 9-12 | -0.2136 | -0.5190 |
| `c18_guided_isotonic_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.3009 | 409.4 | 0.5149 | 9-12 | -0.2141 | -0.5199 |
| `c18_guided_platt_aromatic_rings` | c18_calibration_or_readout | 1.0 | -1 | 0.2465 | 404.8 | 0.5138 | 9-12 | -0.2673 | -0.5714 |
| `pilot_50k_p2_lam0p25_guided_aromatic_rings` | section19_lambda_sweep | 0.25 | -1 | 0.2359 | 405.6 | 0.5140 | 9-12 | -0.2781 | -0.5825 |

### hbd_count — 18 arms, 6 above the head-selected curve

| run | family | λ | layer | guided | tokens/mol | head-selected @ budget | bracket | advantage vs head | advantage vs oracle (C26) |
|---|---|---|---|---|---|---|---|---|---|
| `c23_guided_L4_lam2_hbd_count` | c23_mid_layer | 2.0 | 4 | 0.5603 | 387.8 | 0.3247 | 8-9 | **+0.2357** | 0.0267 |
| `c23_guided_L6_lam2_hbd_count` | c23_mid_layer | 2.0 | 6 | 0.4684 | 399.1 | 0.3273 | 8-9 | **+0.1411** | -0.0756 |
| `pilot_50k_p2_lam2_guided_hbd_count` | section19_lambda_sweep | 2.0 | -1 | 0.4303 | 393.2 | 0.3259 | 8-9 | **+0.1044** | -0.1082 |
| `pilot_50k_p2_lam4_guided_hbd_count` | section19_lambda_sweep | 4.0 | -1 | 0.3730 | 393.7 | 0.3260 | 8-9 | **+0.0470** | -0.1660 |
| `c23_guided_L4_lam1_hbd_count` | c23_mid_layer | 1.0 | 4 | 0.3689 | 394.3 | 0.3261 | 8-9 | **+0.0427** | -0.1706 |
| `c23_guided_L6_lam1_hbd_count` | c23_mid_layer | 1.0 | 6 | 0.3395 | 398.6 | 0.3271 | 8-9 | **+0.0123** | -0.2040 |
| `pilot_50k_p2_lam8_guided_hbd_count` | section19_lambda_sweep | 8.0 | -1 | 0.3104 | 406.1 | 0.3293 | 9-12 | -0.0189 | -0.2390 |
| `c18_guided_head_wide_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2997 | 401.2 | 0.3278 | 9-12 | -0.0281 | -0.2460 |
| `c18_guided_uncalibrated_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2988 | 401.6 | 0.3279 | 9-12 | -0.0292 | -0.2472 |
| `pilot_50k_p2_guided_hbd_count` | deployed_lambda1 | 1.0 | -1 | 0.2988 | 401.6 | 0.3279 | 9-12 | -0.0292 | -0.2472 |
| `c18_guided_head_wide_focused_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2847 | 400.4 | 0.3276 | 9-12 | -0.0429 | -0.2604 |
| `c23_guided_L4_lam0p5_hbd_count` | c23_mid_layer | 0.5 | 4 | 0.2319 | 397.0 | 0.3268 | 8-9 | -0.0949 | -0.3101 |
| `c18_guided_isotonic_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.2128 | 407.5 | 0.3297 | 9-12 | -0.1168 | -0.3376 |
| `c23_guided_L6_lam0p5_hbd_count` | c23_mid_layer | 0.5 | 6 | 0.2027 | 394.4 | 0.3262 | 8-9 | -0.1235 | -0.3370 |
| `pilot_50k_p2_lam0p5_guided_hbd_count` | section19_lambda_sweep | 0.5 | -1 | 0.1813 | 395.9 | 0.3265 | 8-9 | -0.1452 | -0.3597 |
| `c18_guided_platt_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.1567 | 395.3 | 0.3264 | 8-9 | -0.1697 | -0.3838 |
| `pilot_50k_p2_lam0p25_guided_hbd_count` | section19_lambda_sweep | 0.25 | -1 | 0.1324 | 393.5 | 0.3260 | 8-9 | -0.1935 | -0.4064 |
| `c18_guided_bin_temperature_hbd_count` | c18_calibration_or_readout | 1.0 | -1 | 0.1280 | 393.0 | 0.3258 | 8-9 | -0.1978 | -0.4103 |

### qed — 9 arms, 3 above the head-selected curve

| run | family | λ | layer | guided | tokens/mol | head-selected @ budget | bracket | advantage vs head | advantage vs oracle (C26) |
|---|---|---|---|---|---|---|---|---|---|
| `c23_guided_L4_lam4_qed` | c23_mid_layer | 4.0 | 4 | 0.3092 | 375.3 | 0.2431 | 8-9 | **+0.0661** | -0.2616 |
| `c23_guided_L4_lam2_qed` | c23_mid_layer | 2.0 | 4 | 0.2796 | 366.1 | 0.2431 | 8-9 | **+0.0365** | -0.2815 |
| `pilot_50k_p2_lam4_guided_qed` | section19_lambda_sweep | 4.0 | -1 | 0.2607 | 371.7 | 0.2431 | 8-9 | **+0.0176** | -0.3063 |
| `pilot_50k_p2_lam8_guided_qed` | section19_lambda_sweep | 8.0 | -1 | 0.2438 | 423.5 | 0.2462 | 9-12 | -0.0024 | -0.3709 |
| `pilot_50k_p2_lam2_guided_qed` | section19_lambda_sweep | 2.0 | -1 | 0.2361 | 361.9 | 0.2431 | 8-9 | -0.0070 | -0.3206 |
| `c23_guided_L4_lam1_qed` | c23_mid_layer | 1.0 | 4 | 0.2288 | 367.8 | 0.2431 | 8-9 | -0.0143 | -0.3341 |
| `pilot_50k_p2_guided_qed` | deployed_lambda1 | 1.0 | -1 | 0.1908 | 367.3 | 0.2431 | 8-9 | -0.0522 | -0.3715 |
| `pilot_50k_p2_lam0p5_guided_qed` | section19_lambda_sweep | 0.5 | -1 | 0.1692 | 376.2 | 0.2431 | 8-9 | -0.0739 | -0.4025 |
| `pilot_50k_p2_lam0p25_guided_qed` | section19_lambda_sweep | 0.25 | -1 | 0.1332 | 381.6 | 0.2431 | 8-9 | -0.1099 | -0.4441 |

### C27.5.1 What separates the arms that win from the arms that lose

The 15 arms above the head-selected curve are not a random subset. **14** of them are either
**high λ** (2 or 4) or read a **mid-network probe layer** (3, 4, 6), and often both: 10 are
`c23_mid_layer`, 4 are λ ≥ 2 from the §19 sweep, and 1 is a C18 calibration variant. Not one
of the 15 is a deployed-configuration arm (λ=1, layer -1, uncalibrated).

The single exception to the "high λ or mid layer" rule is
`c18_guided_binT0p4_aromatic_rings` — λ=1 at the final layer, but with the head's bin
distribution sharpened by a temperature of 0.4. Sharpening the head's output is
functionally the same lever as raising λ, so it is an exception to the rule as stated and
not to the reading below; it is called out here rather than folded in.

The reading is that guidance and head selection use the *same* head but are limited by
different things. Head selection is capped by how well the head ranks *finished* molecules,
which is fixed. Guidance is capped by how hard it pushes and where it reads, and those are
knobs — turning λ up from 1 to 2 moves `pilot_50k_p2_lam2_guided_aromatic_rings` from
-0.0439 to +0.0403 at essentially the same cost. C26's finding that guidance has no *compute*
knob stands; C27 adds that it does have an *accuracy* knob and that the knob was not turned
to its best setting in the deployed configuration.

**Two cautions on the 15.** First, `c23_guided_L3_lam2_aromatic_rings` and
`c23_guided_L6_lam2_aromatic_rings` carry validity 0.9681 and 0.9688 against the head arm's
0.9976 at the bracketing N — high λ buys hit rate partly by making molecules the base policy
would not have made. Second, C25 showed that `c23_guided_L4_lam2_hbd_count` does not survive
head-seed replication (mean -0.0057, sign flipped, on the *oracle* curve). C27 re-prices only
the comparator, not that instability; a +0.2357 margin is far larger than the 0.0916
head-seed span, but the same span applies to every C23 arm here and none of them has been
replicated across head seeds.

---

## C27.6 The pre-registered decision rules and predictions, scored

### C27.6.1 E1 — does selection still beat steering at equal information?

Pre-registered: upheld iff no measured guidance arm sits above the head-selected curve at its
own budget, on all three anchors.

**NOT upheld.** 15 of 46 arms are above, on all three anchors (6, 6, 3). The largest margin
is +0.2915. This is the opposite of C26's D1 result on the same 46 arms and the same budgets;
only the comparator changed.

### C27.6.2 E2 — the price of ground truth

Reported in §C27.4. Gaps of 0.3024 / 0.2172 / 0.3534 at N=9 and 0.3724 / 0.4415 / 0.6836 at
N=32, with no effective N inside the grid on any anchor.

### C27.6.3 E3 — degeneracy check

Pre-registered: if the head arm gains < 0.02 from N=1 to N=32 on an anchor, it measures
nothing and E1 must not be read as evidence for guidance there.

| anchor | head-selected at N=1 | at N=32 | gain | threshold | degenerate? |
|---|---|---|---|---|---|
| aromatic_rings | 0.1704 | 0.6237 | 0.4533 | 0.02 | no |
| hbd_count | 0.0847 | 0.4855 | 0.4008 | 0.02 | no |
| qed | 0.0955 | 0.2767 | 0.1812 | 0.02 | no |

No anchor is degenerate; E1 stands as measured. QED's head arm is nonetheless the weakest and
nearly flat above N=8 (0.2431 → 0.2767 from N=8 to N=32), which is why QED's three E1
violations should be read as "the head-selected comparator is weak here", not as "guidance is
strong here".

### C27.6.4 E4 — the deployed λ=1 arm, the pre-specified headline comparison

| anchor | guided | tokens/mol | head-selected @ budget | bracket | advantage vs head | per-seed (101/202/303) | seed t interval, 2 df | vs oracle (C26) |
|---|---|---|---|---|---|---|---|---|
| aromatic_rings | 0.4735 | 419.3 | 0.5174 | 9-12 | **-0.0439** | -0.0121, -0.0425, -0.0766 | [-0.1239, 0.0365] | -0.3532 |
| hbd_count | 0.2988 | 401.6 | 0.3279 | 9-12 | **-0.0292** | -0.0429, -0.0326, -0.0118 | [-0.0685, 0.0103] | -0.2472 |
| qed | 0.1908 | 367.3 | 0.2431 | 8-9 | **-0.0522** | -0.0619, -0.0521, -0.0431 | [-0.0757, -0.0290] | -0.3715 |

**The deployed configuration still loses on all three anchors**, but the measured gap shrinks
by **0.876**, **0.882** and **0.859** of its former size once the comparator stops using the
oracle.

On inference: all three anchors have all three seeds on the same side of zero, which is a
three-way sign test at null probability 0.25 and rejects nothing. The seed-level t interval
on 2 df excludes zero for **QED only** ([-0.0757, -0.0290]); for aromatic rings and HBD count
it does not. **With three generation seeds, and no replication across head-training seeds at
all, E4's sign is suggestive on two anchors and supported on one. That is what n = 3 can
carry, and it is stated rather than dressed up.**

### C27.6.5 The six predictions

| # | prediction | outcome |
|---|---|---|
| prediction 1 | head-selected lies strictly between the N=1 base rate and oracle-selected at every N ≥ 2, all anchors | **HOLDS**, 30 of 30 grid-point comparisons |
| prediction 2 | head-selected still beats the deployed λ=1 arm at its own budget on all three anchors | **HOLDS**; E4 is -0.0439 / -0.0292 / -0.0522, negative everywhere, though only QED's interval excludes zero |
| prediction 3 | `c23_guided_L4_lam2_hbd_count` sits above the head-selected curve too | **HOLDS**, +0.2357 — but the framing around it was wrong: the prediction expected it to be the *only* exception again, and it is one of 15 |
| prediction 4 | the oracle-head gap is largest for qed and smallest for aromatic_rings | **FALSIFIED as written.** "Largest for QED" holds for N ≥ 9 and fails for N ≤ 4, where aromatic rings is largest. "Smallest for aromatic rings" fails at every N except 32; the smallest gap is **hbd_count** at N = 2, 4, 9, 16 and 24 |
| prediction 5 | terminal-position pool AUROC exceeds the pooled held-out AUROC (0.7904 / 0.7781 / 0.7355) on all three anchors | **HOLDS**, on the worst seed of each anchor: 0.8646 / 0.8709 / 0.7993 |
| prediction 6 | the 75%-position arm lands strictly between the terminal arm and the N=1 base rate on all three anchors | **FALSIFIED.** It holds only on hbd_count. On aromatic rings the 75% arm is **above** the terminal arm for every N ≥ 6 (0.6741 vs 0.6237 at N=32), and on QED it is above at N=16 and N=32 |

**Prediction 6's failure is the most interesting thing C27 did not expect**, and it is a
result rather than a nuisance. On aromatic rings the head is a *better selector* when it
reads a 75%-complete prefix than when it reads the finished molecule, even though its pool
AUROC at that position is *lower* (0.8615 vs 0.8691). AUROC scores the ordering of a random
pair; best-of-N takes an argmax over 32 draws, which is dominated by the extreme upper tail of
the score. A head that is sharper but more overconfident at the terminal position loses more
to the winner's curse than it gains in average ranking. This is a concrete, measured warning
that **selection quality is not AUROC**, and it applies to any reranker in this literature.

It also, incidentally, weakens the sceptical attack pre-registered in C27.0.8: if reading the
*finished* molecule were what made arm 2 a near-oracle, arm 3 would be far worse than arm 2.
On two of three anchors it is not worse at all.

---

## C27.7 Sensitivity S1 — the pessimistic accounting

C27's headline charges head scoring zero generator tokens (C27.0.4). If a reader rejects that
and charges a full re-read of every candidate, the measured recompute cost is **44.4** tokens
per pool molecule on all three anchors — one extra pass over the sequence, so the head arm's
budget exactly doubles at every N: 88.9 at N=1 through 2844.0 at N=32.

| anchor | recompute tokens per pool molecule | arms above head-selected curve, free accounting | arms above, pessimistic accounting |
|---|---|---|---|
| aromatic_rings | 44.4 | 6 | 11 |
| hbd_count | 44.4 | 6 | 11 |
| qed | 44.4 | 3 | 6 |

Under the pessimistic accounting **28** of 46 arms sit above the head-selected curve instead
of 15. Every C27 verdict therefore moves *further* against selection, not towards it: no
conclusion in this section depends on the free-scoring argument, and the free accounting is
the conservative choice against C27's own headline.

---

## C27.8 What C27 changes in the rest of the report

These are conflicts for the owner to merge, **not** edits C27 has made.
`reports/pilot_report.md`, `README.md` and every existing `outputs/` directory are untouched.

1. **§4, §16.2, §19.4 and C26's headline must be re-scoped.** "Guided decoding loses to
   best-of-N at matched compute" is true only against an oracle-selecting best-of-N. The
   accurate statement is: *guided decoding loses to oracle-selected best-of-N at matched
   compute, and at the deployed setting also loses to head-selected best-of-N, by 0.0292 to
   0.0522 rather than 0.2472 to 0.3715.* Every headline gap in the report is inflated by
   roughly 8x by the oracle.
2. **C26's D1 needs a companion line, not a retraction.** D1 was scored "NOT upheld, 1 of 46
   arms above". Against the equal-information comparator it is 15 of 46. C26's conclusion
   that best-of-N dominates *as a compute-frontier* is untouched; its implicit claim that
   this settles steering-vs-selection is not.
3. **C23's Rule B gains a second life and keeps its problem.** `c23_guided_L4_lam2_hbd_count`
   is +0.2357 above the head-selected curve, ten times its +0.0267 margin over the oracle
   curve. C25's head-seed sign flip is still unaddressed and C27 does not address it: the
   arm is not rehabilitated, its comparator is.
4. **§19's λ sweep should be re-read as an accuracy knob.** The arms that beat head selection
   are the high-λ and mid-layer ones. The deployed λ=1 final-layer configuration is not the
   best guidance configuration measured, and the report currently reports it as *the* result.
5. **Nothing here touches C17, C18's probe conclusions, or C24's generality result.** C27
   re-prices a comparator; it re-measures nothing.

---

## C27.9 Limitations

1. **The information is matched at the source but not in time.** Arm 2 scores a *finished*
   molecule; guidance must commit token by token. C27 isolates selection-after-the-fact from
   steering-before-the-fact deliberately (C27.0.8), and arm 3 shows this matters less than
   expected, but "equal information" here means "same head, same probe point, same interval,
   same binning" — not "same decision timing".
2. **Three generation seeds, one head seed.** E4's t intervals on 2 df exclude zero on one
   anchor of three, and no C27 arm is replicated across head-training seeds. C25 measured a
   0.0916 head-seed span on a single guided arm — larger than two of the three E4 effects.
   **The right follow-up is C27 re-run on head seeds 2345 and 3456**, which costs one more
   sweep and would settle whether the E4 signs are stable.
3. **The head-selected arm is one selection rule among many.** Argmax of `P(y ∈ I)` is the
   obvious rule and the one matched to guidance's own scorer, but §C27.6.5 shows argmax is
   exactly where a miscalibrated score is weakest. A rank-average over positions, or a
   calibrated score, could raise the head-selected curve and reduce the count of 15.
   C27 does not explore that; C18's calibration variants exist for guidance, not for
   selection.
4. **A practitioner has the oracle.** RDKit is free and fast. Nobody deploying best-of-N on
   these properties would use a learned head when the true property is computable, so the
   head-selected curve is a *scientific* control, not a practical baseline. It says what the
   comparison was measuring; it does not say what to run.
5. **Linear interpolation in tokens.** The head-selected curves are concave, so linear
   interpolation between bracketing grid points slightly underestimates the comparator and
   biases every advantage upward — against C27's own E4 result and in favour of the 15
   violations. Brackets are 8-9 or 9-12 for every arm; the grid is dense there.
6. **`actual` accounting only**, consistent with C26 and for the same reason: under
   `full_recompute` best-of-N saturates to 1.0000 and cannot discriminate.
7. **One dataset, one generator, three anchors.** C24 is the external-validity check; C27 is
   not.

---

## REPRODUCE

```bash
# the three-arm sweep (pre-registration must already exist and be older than every output)
for prop in aromatic_rings hbd_count qed; do
  .venv/bin/python scripts/22_head_selected_bestofn.py --dataset pilot_50k_p2 \
      --property "$prop" --n-max 32 --n-molecules 512 --out "c27_headsel_${prop}"
done

# the frontier: reads existing artefacts only, generates nothing, trains nothing
.venv/bin/python scripts/22_summarise_c27.py

# binding tests
.venv/bin/python -m pytest tests/test_head_selected_bestofn.py -q
```

Artefacts: `outputs/c27_prereg/` (pre-registration and SHA-256 lock),
`outputs/c27_headsel_{aromatic_rings,hbd_count,qed}/` with run contexts,
`outputs/c27_headsel.log`, and `outputs/c27_summary/c27_metrics.json`.
