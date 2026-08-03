# Section C28 — the top-k sweep: does guided decoding have a compute knob?

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-01. Pre-registration `outputs/c28_prereg/C28.0_preregistration.md`, frozen with its
SHA-256 in `outputs/c28_prereg/prereg_lock.json` before any C28 measurement existed. Every
number below is re-derived from JSON by `tests/test_k_sweep.py`.

---

## The verdict, up front

**The knob exists. C26 was wrong about that. Turning it buys nothing, which is what C26 was
right about.**

C26's structural headline — *"guided decoding has no compute knob; all 46 arms sit inside a
5.1-17.0% token band while best-of-N spans 32x"* — was a claim about `top_k_candidates: 8`,
a hyperparameter frozen in `configs/guidance.yaml` and never swept anywhere in this project.
C28 sweeps it, k in {2, 4, 8, 16, 32}, on hbd_count at the deployed probe point and at the
best mid-network one, seeds 101/202/303, 512 molecules per cell, `actual` accounting.

1. **The cost band is an artefact of the frozen hyperparameter.** Within a single strand —
   same frozen generator, same frozen head, same frozen lambda, one hyperparameter moved —
   cost spans **10.28x** (140.83 to 1447.42 processed tokens per molecule) against C26's
   1.170. Decision rule D1: **REFUTED AS STATED**. Guidance can be offered more compute
   inside the method exactly as specified.

2. **And it does nothing.** Over that 10.28x, hbd_count's hit rate at the deployed layer
   moves from 0.3283 to 0.3065 — a **span of 0.0320**, and the *cheapest* cell is the best
   one. `hit_rate(k=32) - hit_rate(k=8)` is **0.0078** (A1), **-0.0185** (A2), **0.0044**
   (A3), **-0.0271** (C1), **0.0020** (C2) and **0.0077** (C3). Decision rule D2 is **NULL on
   five strands of six** and **HARMFUL on one** — C1, where all three seeds fall. Guidance is
   flat in its own compute knob while both best-of-N curves rise steeply through the same
   budgets.

3. **So the frontier verdict is unchanged and now properly earned.** At k = 32 the deployed
   hbd_count arm sits **-0.6206** below the oracle-selected curve and **-0.1790** below the
   head-selected curve at its own budget; the mid-layer arm **-0.5767** and **-0.1352**.
   Every one of the six strands is below the oracle-selected curve at k = 32, with every
   seed-level t interval excluding zero. C26's negative result survives the attack that
   would have destroyed it.

4. **The composition does not rescue it either.** N guided drafts reranked by the deployed
   head — the experiment nobody in this project had run — reaches **0.4716** at **3179.77**
   tokens per molecule, which is **-0.0139** against the head-selected best-of-N curve and
   **-0.4555** against the oracle-selected one under the pre-registered pricing, and
   **-0.0457** and **-0.5249** once both curves are measured out to that budget (§C28.6).
   It is below the head-selected curve at every N. Reranking guided drafts by the *oracle*
   reaches **0.9453**, but that arm has the oracle in it and is not a guidance result — and
   it too loses, by **-0.0512**, to plain oracle-selected best-of-N at the same budget,
   which reaches **0.9965** there.

**What this changes in C26.** §C26.4.4's mechanism is wrong and its conclusion is right. The
sentence "it cannot be offered more compute within the method as specified" must go; the
replacement is stronger: *guidance can be offered 10.28x more compute inside the method as
specified and gains nothing from it (-0.0218 in hit rate), while oracle-selected best-of-N
gains 0.6834 and head-selected best-of-N gains 0.2922 over the identical budgets.*
Listed as a conflict for the owner in §C28.8 — that section is not edited here.

**What C28 does not show.** It does not show that k is useless everywhere: on aromatic_rings
the deployed arm collapses to 0.1996 at k = 2 and recovers to 0.4517 by k = 4, so k below 4
starves guidance on that anchor. It does not show that the k x lambda interaction is flat —
lambda was held at three frozen values. And it does not overturn C27: two strands still sit
**above** the head-selected curve at k = 32 — A3 at +0.0807 (interval [-0.0710, 0.2313],
which n = 3 cannot resolve) and C2 at +0.1942 (interval [0.1552, 0.2332], which excludes
zero). Both are mid-layer, high-lambda arms that C27 had already placed above that curve at
k = 8, and both are *worse* relative to it at k = 32 than at k = 8 — C2 falls from +0.2915 to
+0.1942 — because the comparator used the extra budget and guidance did not.

---

## C28.0 The pre-registration, verbatim

The block below is `outputs/c28_prereg/C28.0_preregistration.md` from `## C28.0.1 Why`
onward, reproduced in full and unedited.
`tests/test_k_sweep.py::test_the_report_copies_the_prereg_verbatim` asserts the copy is a
byte-identical substring of this section;
`test_the_prereg_was_written_before_every_measurement` asserts its mtime strictly precedes
every C28 artefact; `test_the_prereg_lock_records_the_prereg_hash` asserts the lock file's
SHA-256 matches.

<!-- BEGIN VERBATIM PREREG COPY -->

## C28.0.1 Why

C26's structural headline is:

> **Guided decoding has no compute knob.** All 46 measured guidance arms sit inside a
> 5.1–17.0% token band while best-of-N spans 32x over the same grid. Guided decoding does
> not merely lose at the budget we matched at — it cannot be offered more compute within
> the method as specified.

That claim is currently false as an argument, for one reason:

> `top_k_candidates: 8` is fixed in `configs/guidance.yaml`, is read by
> `scripts/05_guided_generation.py`, is the default of `guidance.guided_sample`, is hard-set
> in `pooling.py` and `generality.py`, and **has never been swept in this project** — not in
> the molecular runs, not in the text runs. Guidance's cost under the `cached` backend is
> exactly `(k+1)` x the base cost per generated position: 43.448568 base tokens per molecule
> become 401.619141 at k = 8, a ratio of 9.2434. **k is the compute knob**, it lives inside
> the method exactly as specified, it changes no weight and trains nothing, and one guided
> run at k = 32 costs roughly 1420 tokens per molecule — precisely the top of C26's own
> best-of-N grid, where oracle-selected best-of-N reaches 0.9271 (hbd_count), and where
> head-selected best-of-N reaches only 0.4855.

So "guidance cannot be offered more compute" is, today, a claim about a hyperparameter we
froze rather than a finding about the method. C28 measures it.

A partial defence exists and will be stated either way: the **molecular truncation control
is null**. Restricting to the top 8 with the property term switched off gives aromatic rings
0.1829 against unguided 0.1785, hbd_count 0.0849 against 0.0837, qed 0.0829 against 0.0896.
Little property-relevant probability mass sits outside the top 8 in a ~2.4k-token SMILES
vocabulary, so k may buy little accuracy *here*. But that is a fact about this vocabulary,
not about FUDGE-style guidance: on GPT-2 the same control destroys 47.5–85.7% of the base
hit rate. The defence is reported; it does not substitute for the measurement.

Second, nobody in this project has run the obvious composition: **generate N guided drafts
and rerank them**. That composition gives guidance a continuous compute axis immediately and
it is the experiment a hostile reviewer will demand. C28 runs it, reranked both by the RDKit
oracle and by the deployed head.

Both outcomes are results and neither is to be forced:

- guidance still tracks below both best-of-N curves at ~1400 tokens per molecule → C26's
  negative result becomes **genuinely strong and properly scoped**: the knob exists, it was
  turned, and it does not close the gap;
- guidance crosses either curve → C28 is a **positive result** and a better paper.

## C28.0.2 Design

Dataset `pilot_50k_p2`. Seeds **101 / 202 / 303**. 512 molecules per seed per cell. `actual`
token accounting. Frozen windows and frozen target intervals, inherited, never re-derived.
The base generator stays FROZEN; **no head is trained**; only existing deployed and C17
mid-layer checkpoints are loaded.

**k grid, fixed here: k ∈ {2, 4, 8, 16, 32}.** k = 8 is the published value and is therefore
a validity gate rather than a new measurement. k = 1 is excluded because at k = 1 there is
one candidate, the softmax over it is degenerate and the property term cannot act; it would
measure greedy-restricted decoding, not guidance.

**Condition: `throughout` only.** `guidance.guided_sample` calls `torch.manual_seed(seed)` on
entry and each condition is a separate call, so a run restricted to `throughout` is
bit-identical to the same condition inside a six-condition run. This is asserted, not
assumed: validity gate G1 requires exact reproduction of the published six-condition
artefact. The unguided reference (43.448568 tokens per molecule, hit rate 0.083707642 for
hbd_count) is taken from the published artefact and is k-independent by construction, because
`window_fn` returns False at every step and the candidate branch is never entered.

**Strand A — the k sweep, priority 1, on hbd_count.** hbd_count is the anchor where guidance
has always been strongest relative to the head-selected curve (6 of C27's 15 violations) and
is the pre-specified anchor. Three probe-point/λ strands:

| strand | layer | head checkpoint | λ | why this strand |
|---|---|---|---|---|
| A1 | 12 (`-1`) | `pilot_50k_heads_p2/head_hbd_count_frozen_state.pt` | 1.0 | the **deployed** configuration; k = 8 is the published run |
| A2 | 4 | `c17_probe_layers/head_hbd_count_frozen_state_L4.pt` | 1.0 | the **best mid-network layer** |
| A3 | 4 | `c17_probe_layers/head_hbd_count_frozen_state_L4.pt` | 2.0 | the strongest hbd_count guidance arm on record |

**Why layer 4 is "the best mid-network layer", fixed by frozen prior numbers and not chosen
here.** C23 ran hbd_count at exactly two mid-network probe points, 4 and 6, at three λ. Layer
4 beats layer 6 end to end at every common λ: 0.2319 vs 0.2027 at λ = 0.5, 0.3689 vs 0.3395
at λ = 1, 0.5603 vs 0.4684 at λ = 2. There is no tie and no discretion. λ = 2 is §19's
optimal λ for hbd_count, also frozen before C28.

**Strand B — the guided-drafts composition, priority 2, on hbd_count.** Draw a pool of
8 x 512 = 4096 **guided** drafts per seed at the deployed configuration (layer 12, λ = 1,
k = 8), then select over **all disjoint consecutive groups of N** for N ∈ {1, 2, 4, 8}, under
two rerank rules:

- `oracle_reranked` — `bestofn.selection_key` on the true RDKit property, C26's rule verbatim;
- `head_reranked` — argmax of the deployed head's P(y_final ∈ I) read at the **terminal
  content token**, C27's rule verbatim, with no oracle information (invalid drafts are not
  down-ranked, because RDKit validity is oracle information).

N = 1 is an identity gate. `guided_sample` batches at 64 and `n_molecules` only changes the
loop count, so the first 512 drafts of the pool are bit-identical to the published 512; that
is validity gate G6.

**Strand C — priority 3, if and only if strands A and B are complete and on disk.** Extend
the k sweep to `aromatic_rings` (layer 12 λ = 1; mid-layer 3 at λ = 2, C23's best aromatic
arm, 0.8159) and `qed` (layer 12 λ = 1). Named here so that running it is not a choice made
after seeing strand A.

**Run order, fixed here** (the GPU is shared and a kill must cost at most one cell): A1 k=8
(gate G1) first and alone; then A1 k ∈ {2, 4, 16, 32} in that order; then A2 k=8 (gate G2),
A2 {2, 4, 16, 32}; then A3 k=8 (gate G3), A3 {2, 4, 16, 32}; then strand B; then strand C.
Every cell is its own output directory and a completed cell is never regenerated.

## C28.0.3 Token accounting

`actual`, C26's and C27's. Under the `cached` candidate backend a guided step charges
`active` base tokens plus `active * k` candidate tokens, so for a `throughout` run

    processed_tokens_actual = (k + 1) * S,   S = sum over steps of the active-molecule count

exactly, with S itself a function of k because guidance changes sequence lengths. Per-draft
attribution in strand B uses the same identity: a draft of full sequence length L (`<bos>`,
n content tokens, `<eos>`; L = n + 2) is active for L - 1 steps and therefore costs
`(k + 1) * (L - 1)` tokens. Gates G4 and G7 check both statements exactly.

**Head scoring in strand B is charged zero generator tokens**, C27's rule verbatim and for
C27's reason: the head reads `hidden_states[-1]` at a position the generator already
computed. The recompute this implementation actually pays is measured and published so a
reader who rejects the argument can price it (sensitivity S1).

Processed tokens are reported, never wall-clock.

## C28.0.4 What the k sweep is priced against

Both curves, on one x-axis, at the same budgets:

- **oracle-selected** best-of-N: `outputs/c26_nsweep_hbd_count/n_sweep_metrics.json::curve`;
- **head-selected** best-of-N: `outputs/c27_headsel_hbd_count/head_selected_metrics.json::curves.head_selected`.

Pricing uses `interp` and `t_interval` **imported** from `scripts/21_summarise_c26.py`, not
reimplemented, so C28 prices arms with the code that produced C26's and C27's numbers. Points
whose budget exceeds the grid maximum (1421.98 tokens per molecule) are flagged
`extrapolated_beyond_grid` and are compared against the curve's terminal value, which is
**conservative against guidance is false here** — it is conservative *for* guidance, since
both curves are still rising at N = 32. Every such comparison says so explicitly.

## C28.0.5 Validity gates, checked before any curve is read

| gate | rule | criterion |
|---|---|---|
| **G1** | strand A1 at k = 8 must reproduce `outputs/pilot_50k_p2_guided_hbd_count` (`throughout`) | hit-rate residual **0.0** and tokens-per-molecule residual **0.0**, aggregate and every seed |
| **G2** | strand A2 at k = 8 must reproduce `outputs/c23_guided_L4_lam1_hbd_count` | same, residual 0.0 |
| **G3** | strand A3 at k = 8 must reproduce `outputs/c23_guided_L4_lam2_hbd_count` | same, residual 0.0 |
| **G4** | the cost identity | `processed_tokens_actual % (k + 1) == 0` for every cell and seed |
| **G5** | strand B at N = 1 | both rerank arms identical; residual 0.0 |
| **G6** | strand B pool provenance | the first 512 drafts of each seed must be the published run's 512, compared as canonical SMILES strings; residual 0 mismatches |
| **G7** | strand B per-draft token attribution | `sum over drafts of (k+1)*(L-1)` equals the meter's `processed_tokens_actual`; residual 0 |

**G1 is blocking**: if its residual is not exactly 0.0, C28 stops and diagnoses before running
any other cell. G2 and G3 are blocking for their own strand only. G4–G7 are reported as
numbers and, if any fails, the affected quantity is withdrawn rather than caveated.

## C28.0.6 Decision rules

**D1 — the cost band.** C26 states that all 46 guidance arms sit inside a **5.1–17.0%** token
band, i.e. a max/min token ratio of **1.170**. D1 is scored on the measured ratio
`max_k tokens_per_molecule / min_k tokens_per_molecule` within a single strand. C26's "no
compute knob" claim is **REFUTED AS STATED** iff that ratio exceeds 2.0 in at least one
strand while every cell in that strand is the same method with the same frozen generator,
the same frozen head and the same frozen λ. Otherwise it is **UPHELD**.

**D2 — does the knob buy accuracy?** For each strand, `hit_rate(k=32) - hit_rate(k=8)`. The
knob is **PRODUCTIVE** on a strand iff that difference exceeds **+0.02** (C27's E3 threshold,
reused rather than re-chosen) **and** all three seeds share its sign. It is **NULL** iff
|difference| ≤ 0.02. It is **HARMFUL** iff the difference is below −0.02 with a consistent
sign.

**D3 — the frontier verdict, the one the paper turns on.** For each strand and each k,
the advantage against both curves interpolated at that cell's own budget. Reported per cell
with per-seed values and a seed-level t interval on 2 df. Summarised at the k = 32 cell:

- **D3a**: guidance at k = 32 sits **above / below** the oracle-selected curve at its budget;
- **D3b**: guidance at k = 32 sits **above / below** the head-selected curve at its budget.

C26's negative result **survives in its scoped form** iff D3a is "below" for every strand.
The stronger claim ("guidance loses at equal information") requires D3b "below"; C27 already
showed it fails for some non-deployed arms, so a D3b crossing at k = 32 is expected for A3
and would not by itself be news. A D3b crossing for **A1, the deployed configuration**, would
be news.

**D4 — the composition.** For strand B, the advantage of `oracle_reranked` and of
`head_reranked` against both best-of-N curves at their own budgets. The composition is a
**usable compute axis for guidance** iff at some N it sits above the head-selected curve at
its own budget with all three seeds sharing the sign; it **dominates** iff it also sits above
the oracle-selected curve. Explicitly pre-registered as the likely failure mode: the
composition costs `N * 401.6` tokens per returned molecule, so at N = 8 it costs ~3213 tokens
per molecule, which is 2.26x beyond the grid maximum and 74x the base cost; plain best-of-N
at that budget would be roughly N = 73 and its oracle curve is already at 0.9271 by N = 32.

**D5 — the defence.** The truncation control is null at k = 8. D5 asks whether it stays null:
report `hit_rate(k)` for strand A1 against the unguided 0.083708 and against C26's published
truncation control 0.084915 at k = 8. If hit rate is flat in k, the null truncation control
is the *explanation* and must be reported as such; the section then states that this is a
statement about a 2.4k-token SMILES vocabulary, and quotes the GPT-2 47.5–85.7% figure to
scope it.

## C28.0.7 Multiplicity and uncertainty

**No bootstrap anywhere in C28.** At n = 3 the percentile bootstrap of a mean is identically
[min, max]: P(all three resampled indices hit the minimum) = 1/27 = 0.0370 > 0.025, so the
2.5th percentile *is* the minimum for any three numbers whatsoever, and "the CI excludes
zero" carries exactly the information of a three-way sign test at null probability 0.25.
Reported instead: **every per-seed value**, and a **seed-level Student t interval on 2 df**
with t(0.975, 2) = 4.302653. Where n = 3 cannot resolve a comparison, the section says so in
those words rather than reporting a decorated interval.

Multiplicity: strand A is 15 cells (3 strands x 5 k), of which 3 are gates, leaving 12 new
measurements; strand B is 8 (2 arms x 4 N), of which 2 are the N = 1 identity. No per-cell
significance test is performed and no cell is declared significant on its own; D2 and D3 are
scored on the k = 32 cell named in advance, and the whole k profile is shown so a reader can
see the shape rather than a selected point.

## C28.0.8 The attack this design invites, stated before the result

1. **"k = 32 is not the only knob; you could also raise λ, or both."** True. λ was swept in
   §19 and is crossed with layer in C23; C28 holds λ at three pre-specified frozen values and
   sweeps k, because k is the one that changes *cost*. The k x λ interaction is not measured
   and the section will say so.
2. **"Your budget axis charges guidance for candidate evaluation but charges head-selected
   best-of-N nothing for head scoring."** Correct, and deliberate: it is C27's rule, and it is
   the accounting that is *hardest* on guidance. Sensitivity S1 re-prices strand B under the
   pessimistic rule that charges the recompute in full.
3. **"At k = 32 the softmax is over 32 candidates and validity collapses, so the hit rate is
   not comparable."** Validity is reported at every cell. If validity falls below 0.95 at any
   k the cell is flagged and the §19 fragmentation finding is cited; hit rate is *not*
   silently revalidated.
4. **"The first 512 drafts of strand B's pool cannot be the published 512."** G6 tests exactly
   this. If it fails, the pool is not the deployed sampler's and strand B is reported as a
   *different* sampler rather than as the deployed one at depth.

## C28.0.9 Predictions

Scored verbatim in §C28.7, including failures.

- **P1 (cost).** Tokens per molecule at k is within **±10%** of `401.619141 * (k+1)/9` for
  every cell of strand A1 — i.e. cost is (k+1) x base and sequence length moves little.
- **P2 (monotonicity).** Hit rate is **non-decreasing in k** on strand A1 across the grid,
  within seed noise: `hit_rate(32) >= hit_rate(2)`.
- **P3 (the knob is weak here).** `hit_rate(k=32) - hit_rate(k=8)` on strand A1 is **less
  than +0.10**, because the truncation control is null. This is the prediction the defence in
  §C28.0.1 makes and it is the one most likely to be wrong.
- **P4 (D3a).** Guidance at k = 32 sits **below** the oracle-selected curve at its own budget
  on **all three** strands. The oracle curve is at 0.9271 at 1421.98 tokens.
- **P5 (D3b, the deployed arm).** Strand A1 at k = 32 sits **below** the head-selected curve
  at its own budget (which is 0.4855 at 1421.98 tokens).
- **P6 (D3b, the strong arm).** Strand A3 at k = 32 sits **above** the head-selected curve at
  its own budget, because it already sits above it at k = 8 (0.5603 at 387.79 tokens against
  an interpolated ~0.325).
- **P7 (validity).** Validity at k = 32 is **lower** than at k = 2 on every strand, because a
  wider candidate set lets the property term overrule the base policy more often — the same
  mechanism §19 measured in λ.
- **P8 (composition, oracle).** `oracle_reranked` at N = 8 exceeds **0.90** hit rate, and
  nevertheless sits **below** the oracle-selected best-of-N curve extrapolated to its own
  ~3213-token budget.
- **P9 (composition, head).** `head_reranked` at N = 8 sits **above** the head-selected
  best-of-N curve's terminal value 0.4855.
- **P10 (the cost band).** D1 is REFUTED AS STATED: the measured max/min token ratio within
  strand A1 exceeds 3.0, against C26's 1.170.
<!-- END VERBATIM PREREG COPY -->

---

## C28.1 What was run

Nothing was forked. Every molecule in strand A comes from `guidance.guided_sample` called
with the arguments `scripts/05_guided_generation.py` passes it, and the per-condition summary
is that script's own `summarise` **imported** rather than copied, so the k = 8 identity below
cannot be passed by a re-implementation that merely agrees today. The only thing that varies
is `top_k`. No weight changed, no head was trained, no second generator was used, and no
existing script or output directory was modified — `scripts/23_k_sweep.py`,
`scripts/23_guided_drafts.py` and `scripts/23_summarise_c28.py` are new files. `guidance.py`
needed **no edit at all**: `guided_sample` already took `top_k` as an argument; only
`scripts/05_guided_generation.py` hard-read it from `configs/guidance.yaml`.

| strand | property | probe point | head checkpoint | lambda | k grid |
|---|---|---|---|---|---|
| A1 | hbd_count | 12 (`-1`, deployed) | `pilot_50k_heads_p2/head_hbd_count_frozen_state.pt` | 1.0 | 2, 4, 8, 16, 32 |
| A2 | hbd_count | 4 | `c17_probe_layers/head_hbd_count_frozen_state_L4.pt` | 1.0 | 2, 4, 8, 16, 32 |
| A3 | hbd_count | 4 | `c17_probe_layers/head_hbd_count_frozen_state_L4.pt` | 2.0 | 2, 4, 8, 16, 32 |
| B | hbd_count | 12 (`-1`, deployed) | `pilot_50k_heads_p2/head_hbd_count_frozen_state.pt` | 1.0 | k = 8, N drafts in 1, 2, 4, 8 |
| C1 | aromatic_rings | 12 (`-1`, deployed) | `pilot_50k_heads_p2/head_aromatic_rings_frozen_state.pt` | 1.0 | 2, 4, 8, 16, 32 |
| C2 | aromatic_rings | 3 | `c17_probe_layers/head_aromatic_rings_frozen_state_L3.pt` | 2.0 | 2, 4, 8, 16, 32 |
| C3 | qed | 12 (`-1`, deployed) | `pilot_50k_heads_p2/head_qed_frozen_state.pt` | 1.0 | 2, 4, 8, 16, 32 |

Seeds 101 / 202 / 303 everywhere, 512 molecules per seed per cell, condition `throughout`
only, `actual` token accounting, frozen windows (t33 = 15, t67 = 29) and frozen target
interval [3.0, 4.0) for hbd_count with base rate 0.0823. Both comparator curves are read
from disk and never regenerated: `outputs/c26_nsweep_hbd_count/n_sweep_metrics.json` for
oracle-selected best-of-N and `outputs/c27_headsel_hbd_count/head_selected_metrics.json` for
head-selected, and the pricing uses `interp` and `t_interval` imported from
`scripts/21_summarise_c26.py`.

---

## C28.2 Validity gates, as numeric residuals

### C28.2.1 G1-G3 — the k = 8 cell reproduces its frozen artefact exactly

The gate the task named: at k = 8 the sweep must reproduce the published deployed run.

| gate | strand | reference artefact | hit-rate residual | tokens/molecule residual | worst per-seed residual |
|---|---|---|---|---|---|
| G1 | A1 | `pilot_50k_p2_guided_hbd_count` | 0.0 | 0.0 | 0.0 / 0.0 |
| G2 | A2 | `c23_guided_L4_lam1_hbd_count` | 0.0 | 0.0 | 0.0 / 0.0 |
| G3 | A3 | `c23_guided_L4_lam2_hbd_count` | 0.0 | 0.0 | 0.0 / 0.0 |

The measured A1 k = 8 aggregate is hit rate **0.29875114714835704** at
**401.619140625** processed tokens per returned molecule — the published values, bit for
bit, aggregate and on every one of the three seeds. This also confirms the design assumption
that a run restricted to the `throughout` condition is identical to the same condition inside
a six-condition run, because `guided_sample` reseeds on entry.

### C28.2.2 G4 — the cost identity

Under the `cached` backend a guided step charges `active` base tokens plus `active * k`
candidate tokens, so `processed_tokens_actual` must be divisible by `k + 1` exactly. Maximum
residual over every strand, every k and every seed: **0**. Cost is exactly (k+1)x base steps,
which is why k is a compute knob at all.

### C28.2.3 G5-G7 — the composition

| gate | rule | residual |
|---|---|---|
| G5 | at N = 1 both rerank arms must agree | 0.0 |
| G6 | the first 512 drafts of each seed must be the published deployed run's 512, as SMILES | 0 / 0 / 0 mismatches |
| G7 | per-draft charge `(k+1)*(len(seq)-1)` must sum to the meter's own total | 0 / 0 / 0 |

G6 is the strong form of the provenance check: the guided-draft pool is not merely *like*
the deployed sampler's output, its first 512 molecules **are** the published 512, string for
string, on all three seeds. The composition is therefore the deployed configuration run
deeper, not a new sampler.

---

## C28.3 The k sweep

Figure: `outputs/c28_figures/c28_k_sweep_frontier.png` — guidance and **both** best-of-N
curves on one processed-token x-axis. That axis has never existed in this project: guidance
had one budget and best-of-N had a curve, so they were only ever compared at a single matched
point.

Advantages are against each curve linearly interpolated in tokens at the cell's own budget.
A `*` marks a budget beyond C26's grid maximum of 1421.98 tokens per molecule, where the
comparison is against the curve's terminal value and is therefore **conservative in
guidance's favour** — both curves are still rising at N = 32. §C28.6 removes that
extrapolation by measurement.

### C28.3.1 A1 — hbd_count at the deployed probe point, lambda = 1

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 140.83 | **0.3283** | 0.3242 / 0.3418 / 0.3190 | 0.9993 | 0.0846 | 0.1349 |
| 4 | 226.44 | **0.3270** | 0.3092 / 0.3386 / 0.3333 | 0.9974 | -0.0324 | 0.0750 |
| 8 | 401.62 | **0.2988** | 0.3014 / 0.2902 / 0.3047 | 0.9980 | -0.2472 | -0.0292 |
| 16 | 749.23 | **0.2963** | 0.3189 / 0.2784 / 0.2916 | 0.9954 | -0.4688 | -0.1116 |
| 32 | 1447.42 | **0.3065** | 0.3366 / 0.3000 / 0.2829 | 0.9941 | -0.6206 \* | -0.1790 \* |

Cost spans **10.28x**. Hit rate spans **0.0320** and is *highest at the cheapest cell*. The
advantage columns fall monotonically not because guidance gets worse but because the
comparators get better: guidance is a horizontal line drawn across two rising curves.

### C28.3.2 A2 — hbd_count at the best mid-network probe point (layer 4), lambda = 1

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 139.96 | **0.3749** | 0.3843 / 0.3699 / 0.3706 | 0.9967 | 0.1325 | 0.1822 |
| 4 | 224.06 | **0.3782** | 0.3686 / 0.3679 / 0.3980 | 0.9967 | 0.0218 | 0.1275 |
| 8 | 394.27 | **0.3689** | 0.3608 / 0.3640 / 0.3819 | 0.9954 | -0.1706 | 0.0427 |
| 16 | 743.77 | **0.3771** | 0.3777 / 0.3699 / 0.3839 | 0.9961 | -0.3861 | -0.0301 |
| 32 | 1443.36 | **0.3504** | 0.3870 / 0.3143 / 0.3497 | 0.9941 | -0.5767 \* | -0.1352 \* |

Cost spans **10.31x**; hit rate spans **0.0278**.

### C28.3.3 A3 — hbd_count at layer 4, lambda = 2 (the strongest arm on record)

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 136.42 | **0.4872** | 0.4882 / 0.4637 / 0.5098 | 0.9941 | 0.2499 | 0.2975 |
| 4 | 216.67 | **0.5563** | 0.5604 / 0.5591 / 0.5494 | 0.9889 | 0.2093 | 0.3099 |
| 8 | 387.79 | **0.5603** | 0.5976 / 0.5305 / 0.5529 | 0.9935 | 0.0267 | 0.2357 |
| 16 | 723.06 | **0.5361** | 0.5531 / 0.5306 / 0.5247 | 0.9909 | -0.2200 | 0.1317 |
| 32 | 1405.36 | **0.5647** | 0.5580 / 0.5564 / 0.5796 | 0.9915 | -0.3600 | 0.0807 |

Cost spans **10.30x**; hit rate spans **0.0774**, all of it between k = 2 and k = 4. From
k = 4 to k = 32 — a 6.49x cost increase — the hit rate moves from 0.5563 to 0.5647, i.e.
**0.0084**. This is the strand where guidance is strongest and where a compute knob would
have mattered most.

### C28.3.4 Strand C — the other two anchors

#### C1 — aromatic_rings at the deployed probe point, lambda = 1

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 149.84 | **0.1996** | 0.2055 / 0.1922 / 0.2012 | 0.9980 | -0.2662 | -0.1666 |
| 4 | 236.80 | **0.4517** | 0.4403 / 0.4766 / 0.4384 | 0.9987 | -0.1739 | 0.0107 |
| 8 | 419.33 | **0.4735** | 0.4892 / 0.4795 / 0.4517 | 0.9954 | -0.3532 | -0.0439 |
| 16 | 784.38 | **0.4624** | 0.4784 / 0.4648 / 0.4440 | 0.9967 | -0.4983 | -0.1182 |
| 32 | 1533.88 | **0.4463** | 0.4667 / 0.4754 / 0.3969 | 0.9948 | -0.5498 \* | -0.1774 \* |

Cost spans **10.24x**; hit rate spans **0.2739**.

#### C2 — aromatic_rings at layer 3, lambda = 2 (the strongest aromatic arm on record)

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 150.18 | **0.4672** | 0.4451 / 0.4752 / 0.4813 | 0.9909 | 0.0007 | 0.1006 |
| 4 | 253.53 | **0.8230** | 0.8515 / 0.8287 / 0.7887 | 0.9792 | 0.1690 | 0.3698 |
| 8 | 447.46 | **0.8159** | 0.8233 / 0.8452 / 0.7791 | 0.9681 | -0.0277 | 0.2915 |
| 16 | 832.39 | **0.8239** | 0.8164 / 0.8219 / 0.8333 | 0.9720 | -0.1417 | 0.2391 |
| 32 | 1660.46 | **0.8179** | 0.7988 / 0.8439 / 0.8110 | 0.9544 | -0.1782 \* | 0.1942 \* |

Cost spans **11.06x**; hit rate spans **0.3567**.

#### C3 — qed at the deployed probe point, lambda = 1

| k | tokens/molecule | hit rate | per-seed hit rate | validity | vs oracle-selected | vs head-selected |
|---|---|---|---|---|---|---|
| 2 | 131.70 | **0.1940** | 0.1973 / 0.1895 / 0.1953 | 1.0000 | -0.0620 | 0.0155 |
| 4 | 209.20 | **0.2025** | 0.2148 / 0.1953 / 0.1973 | 1.0000 | -0.1694 | -0.0080 |
| 8 | 367.27 | **0.1908** | 0.1879 / 0.1850 / 0.1996 | 0.9961 | -0.3715 | -0.0522 |
| 16 | 695.65 | **0.1948** | 0.1811 / 0.1855 / 0.2176 | 0.9961 | -0.5895 | -0.0634 |
| 32 | 1352.72 | **0.1985** | 0.1834 / 0.1965 / 0.2157 | 0.9935 | -0.7506 | -0.0768 |

Cost spans **10.27x**; hit rate spans **0.0116**.

Strand C adds one qualification and one confirmation. The qualification: on aromatic_rings
the k profile is **not** flat at the cheap end — C1 falls to 0.1996 at k = 2 and C2 to 0.4672,
against 0.4517 and 0.8230 at k = 4. Below four candidates the property term stops working on
this anchor. The confirmation: above k = 4 both aromatic strands are as flat as hbd_count,
and qed is flatter still (span 0.0116 across a 10.27x cost span, the flattest strand in C28).

The pre-registration's attack 3 asked for a validity flag below 0.95. The lowest cell in C28
is **C2 at k = 32, validity 0.9544**, which does not trip it; but C2's validity does fall
monotonically-ish with k (0.9909, 0.9792, 0.9681, 0.9720, 0.9544) and it is the only strand
where the degradation is visible at all. Its hit rate is reported unrevalidated, as
pre-registered.

### C28.3.5 D1 — the cost band

C26 §C26.4.4 states that all 46 guidance arms sit inside a 5.1-17.0% token band, a max/min
ratio of 1.170. Within a single strand, holding the generator, the head and lambda frozen and
moving one hyperparameter:

| strand | min tokens/molecule | max tokens/molecule | ratio |
|---|---|---|---|
| A1 | 140.83 | 1447.42 | **10.28** |
| A2 | 139.96 | 1443.36 | **10.31** |
| A3 | 136.42 | 1405.36 | **10.30** |
| C1 | 149.84 | 1533.88 | **10.24** |
| C2 | 150.18 | 1660.46 | **11.06** |
| C3 | 131.70 | 1352.72 | **10.27** |

D1 verdict: **REFUTED AS STATED**. The 5.1-17.0% band is a property of the 46 arms that were
run, all of which shared `top_k_candidates: 8`; it is not a property of the method. Cost is
exactly `(k+1)` x base (gate G4, residual 0), so k is a compute knob with the same
mechanical status as N in best-of-N.

### C28.3.6 D2 — does the knob buy accuracy?

Pre-registered as `hit_rate(k=32) - hit_rate(k=8)`, PRODUCTIVE above +0.02 with all three
seeds sharing the sign, NULL within +/-0.02.

| strand | hit rate at k=8 | hit rate at k=32 | difference | per-seed differences | seeds share sign | seed t interval (2 df) | verdict |
|---|---|---|---|---|---|---|---|
| A1 | 0.2988 | 0.3065 | **0.0078** | 0.0352 / 0.0098 / -0.0218 | no | [-0.0632, 0.0787] | **NULL** |
| A2 | 0.3689 | 0.3504 | **-0.0185** | 0.0262 / -0.0497 / -0.0322 | no | [-0.1173, 0.0802] | **NULL** |
| A3 | 0.5603 | 0.5647 | **0.0044** | -0.0397 / 0.0260 / 0.0266 | no | [-0.0903, 0.0989] | **NULL** |
| C1 | 0.4735 | 0.4463 | **-0.0271** | -0.0226 / -0.0040 / -0.0548 | yes | [-0.0910, 0.0367] | **HARMFUL** |
| C2 | 0.8159 | 0.8179 | **0.0020** | -0.0245 / -0.0013 / 0.0319 | no | [-0.0684, 0.0724] | **NULL** |
| C3 | 0.1908 | 0.1985 | **0.0077** | -0.0044 / 0.0114 / 0.0161 | no | [-0.0190, 0.0344] | **NULL** |

Every hbd_count strand is NULL and its three seeds do not share a sign, so the sign of the
change is not resolved at n = 3 — which is itself the finding. **PRODUCTIVE is not returned
anywhere**; the only strand whose seeds agree is C1, and they agree that k = 32 is *worse*
than k = 8 (its t interval [-0.0910, 0.0367] still includes zero, so the magnitude is not
resolved either). Over the identical budget span
(140.83 to 1447.42 tokens per molecule) the oracle-selected curve gains **0.6834**
(0.2437 to 0.9271) and the head-selected curve gains **0.2922** (0.1934 to 0.4855), while
A1's guidance moves **-0.0218**. A difference this small does not need three seeds to be
uninteresting.

The best k on the grid is **k = 2** (A1), **k = 4** (A2), **k = 32** (A3), **k = 8** (C1),
**k = 16** (C2) and **k = 4** (C3) — the argmax wanders across the whole grid, which is what
noise around a flat line looks like rather than a curve with an optimum. On the three
strands whose argmax is at k = 2 or k = 4, the best cell is also the **cheapest or second
cheapest**, so the practical reading is that the deployed k = 8 is already past the point
of return.

### C28.3.7 D3 — the frontier verdict at k = 32

| strand | tokens/molecule | guidance | oracle-selected at that budget | advantage | seed t interval | head-selected at that budget | advantage | seed t interval |
|---|---|---|---|---|---|---|---|---|
| A1 | 1447.42 | 0.3065 | 0.9271 \* | **-0.6206** | [-0.6718, -0.5694] | 0.4855 \* | **-0.1790** | [-0.2645, -0.0936] |
| A2 | 1443.36 | 0.3504 | 0.9271 \* | **-0.5767** | [-0.6641, -0.4894] | 0.4855 \* | **-0.1352** | [-0.2872, 0.0168] |
| A3 | 1405.36 | 0.5647 | 0.9246 | **-0.3600** | [-0.4162, -0.3038] | 0.4839 | **0.0807** | [-0.0710, 0.2313] |
| C1 | 1533.88 | 0.4463 | 0.9961 \* | **-0.5498** | [-0.6607, -0.4388] | 0.6237 \* | **-0.1774** | [-0.3009, -0.0538] |
| C2 | 1660.46 | 0.8179 | 0.9961 \* | **-0.1782** | [-0.2352, -0.1212] | 0.6237 \* | **0.1942** | [0.1552, 0.2332] |
| C3 | 1352.72 | 0.1985 | 0.9492 | **-0.7506** | [-0.7939, -0.7073] | 0.2753 | **-0.0768** | [-0.1438, -0.0100] |

**D3a: below the oracle-selected curve on every strand**, with every seed-level t interval
excluding zero. C26's negative result survives the strongest attack available to it.

**D3b: below the head-selected curve on four strands of six** — the deployed hbd_count arm
(-0.1790), the mid-layer hbd_count arm at lambda = 1 (-0.1352), the deployed aromatic_rings
arm (-0.1774) and the deployed qed arm (-0.0768). It is **above** on the two
mid-layer/high-lambda arms: A3 at +0.0807, interval [-0.0710, 0.2313], which does not exclude
zero and is not resolved at n = 3; and C2 at **+0.1942**, interval [0.1552, 0.2332], which
does exclude zero.

Neither of those two is new and neither is a compute result. C27 already placed both arms
above the head-selected curve at k = 8, and `c23_guided_L3_lam2_aromatic_rings` was C27's
largest violation at +0.2915. At k = 32 that same arm's advantage has **shrunk to +0.1942**,
because the comparator gained from the extra budget and guidance did not. Turning the knob
moved guidance's relative position in the wrong direction on the one arm where it was
winning.

---

## C28.4 The composition: N guided drafts, then rerank

Strand B draws 8 x 512 = 4096 guided drafts per seed at the deployed configuration
(layer 12, lambda = 1, k = 8) and reranks all disjoint consecutive groups of N. Cost is
exactly linear in N because every draft is a full guided molecule.

| N | tokens/molecule | oracle-reranked | per-seed | head-reranked | per-seed |
|---|---|---|---|---|---|
| 1 | 397.47 | **0.3103** | 0.3144 / 0.3055 / 0.3110 | **0.3103** | 0.3144 / 0.3055 / 0.3110 |
| 2 | 794.94 | **0.5173** | 0.5220 / 0.5093 / 0.5205 | **0.4018** | 0.3978 / 0.4009 / 0.4067 |
| 4 | 1589.88 | **0.7633** | 0.7734 / 0.7510 / 0.7656 | **0.4521** | 0.4382 / 0.4730 / 0.4451 |
| 8 | 3179.77 | **0.9453** | 0.9336 / 0.9492 / 0.9531 | **0.4716** | 0.4392 / 0.5108 / 0.4648 |

| N | tokens/molecule | oracle-reranked vs oracle curve | vs head curve | head-reranked vs oracle curve | vs head curve |
|---|---|---|---|---|---|
| 1 | 397.47 | -0.2321 | -0.0166 | -0.2321 | -0.0166 |
| 2 | 794.94 | -0.2636 | 0.1031 | -0.3791 | -0.0123 |
| 4 | 1589.88 | -0.1637 \* | 0.2778 \* | -0.4750 \* | -0.0334 \* |
| 8 | 3179.77 | 0.0182 \* | 0.4598 \* | -0.4555 \* | -0.0139 \* |

**D4: the composition is not a usable compute axis for guidance.** `head_reranked` — the
only arm with no oracle information anywhere in it — sits **below** the head-selected
best-of-N curve at every N: -0.0166, -0.0123, -0.0334, -0.0139. It never sits above it, so
the pre-registered condition for "usable compute axis" is not met and "dominates" is not
reached either. Its own scaling is poor for the same reason C27 found: from N = 1 to N = 8
it gains 0.1613 while the *oracle* rerank of the same drafts gains 0.6350.

`oracle_reranked` reaches 0.9453 at 3179.77 tokens and shows **+0.0182** against the
oracle-selected curve — but that number is the pre-registered comparison against the
curve's **terminal value** at 1421.98 tokens, 2.24x below the composition's own budget. It
is an artefact of the extrapolation rule, not a win; §C28.6 measures the curve out to that
budget and the sign flips. `oracle_reranked` also is not a guidance result: it selects with
RDKit, so it belongs on the oracle side of C27's information axis.

The head's discrimination on its own steered pool is **0.7066 / 0.7302 / 0.7157** AUROC
(terminal content position, seeds 101/202/303) against a pool true hit rate of
**0.3137 / 0.3044 / 0.3096**, so the head-rerank arm is measuring something real; it is not
a degenerate-scorer artefact.

---

## C28.5 D5 — the defence, and where it does not hold

The pre-registration named a partial defence in advance: the molecular truncation control is
null, so little property-relevant probability mass sits outside the top 8 in SMILES and k may
buy little accuracy *here*.

| anchor | unguided | truncation control (top-8, lambda = 0) | difference | hit-rate span across the k grid |
|---|---|---|---|---|
| hbd_count | 0.0837 | 0.0849 | 0.0012 | 0.0320 (A1), 0.0278 (A2), 0.0774 (A3) |
| aromatic_rings | 0.1785 | 0.1829 | 0.0043 | 0.2739 (C1), 0.3567 (C2) |
| qed | 0.0896 | 0.0829 | -0.0067 | 0.0116 (C3) |

On hbd_count and qed the defence holds exactly: the truncation control is null and the k
profile is flat (span 0.0320 / 0.0278 / 0.0774 and 0.0116). **On aromatic_rings it does
not.** The deployed arm collapses to **0.1996** at k = 2 — barely above the unguided 0.1785 —
and recovers to **0.4517** by k = 4; the layer-3 lambda = 2 arm goes **0.4672** at k = 2 to
**0.8230** at k = 4. So k *can* matter: below about four candidates the property term has
too little to choose between and guidance stops working on this anchor. What it cannot do on
any anchor measured here is convert *more* candidates into more accuracy above k = 4 — from
k = 4 to k = 32 the six strands move by -0.0205, -0.0278, +0.0084, -0.0054, -0.0051 and
-0.0039, for cost increases of 6.39x to 6.55x. Five of six go down.

This is a statement about a ~2.4k-token chemistry vocabulary, not about FUDGE-style guidance
in general. On GPT-2 the same truncation control destroys **47.5-85.7%** of the base hit
rate, so a text-domain k sweep is a genuinely open question that C28 does not answer.

---

## C28.6 POST HOC, NOT PRE-REGISTERED — the extended best-of-N curve

The pre-registration (C28.0.4) said that points beyond C26's grid maximum of 1421.98 tokens
per molecule would be flagged and priced against the curve's **terminal value**. Two C28
results land there: strand A1/A2 at k = 32, and the whole composition. That rule is
conservative in guidance's favour, and in one place it produced a spurious win —
`oracle_reranked` at N = 8 scored **+0.0182** against a curve that stops 2.24x short of its
budget.

Rather than leave that standing, the **unmodified** `scripts/22_head_selected_bestofn.py` was
rerun with `--n-max 80 --grid 1 2 4 8 9 16 24 32 48 64 72 80`, which *measures* both curves
across the composition's budget. **This is post hoc and is not pre-registered.** The
pre-registered comparison above is scored as written in §C28.7 and is not replaced by it.

Consistency at the overlap: the extended pool re-estimates N = 32 over 2.5x more disjoint
groups, so this is a check, not an identity. Oracle-selected **0.9372** here against C26's
0.9271 (difference **0.0102**); head-selected **0.4662** here against C27's 0.4855
(difference **-0.0194**).

The measured curve, hbd_count, tokens per returned molecule:

| N | tokens/molecule | oracle-selected | head-selected |
|---|---|---|---|
| 32 | 1421.30 | 0.9372 | 0.4662 |
| 48 | 2131.97 | 0.9840 | 0.4837 |
| 64 | 2842.61 | 0.9964 | 0.5235 |
| 72 | 3197.98 | 0.9965 | 0.5170 |
| 80 | 3553.26 | 0.9980 | 0.5374 |

Re-priced against the measured curves, with **no extrapolation anywhere**:

| point | tokens/molecule | hit rate | vs oracle-selected | vs head-selected |
|---|---|---|---|---|
| A1, k = 32 | 1447.42 | 0.3065 | **-0.6325** | **-0.1603** |
| A2, k = 32 | 1443.36 | 0.3504 | **-0.5883** | **-0.1164** |
| A3, k = 32 | 1405.36 | 0.5647 | **-0.3699** | **0.1000** |
| composition, oracle-reranked, N = 8 | 3179.77 | 0.9453 | **-0.0512** | 0.4280 |
| composition, head-reranked, N = 2 | 794.94 | 0.4018 | -0.3794 | **0.0018** |
| composition, head-reranked, N = 4 | 1589.88 | 0.4521 | -0.4962 | **-0.0182** |
| composition, head-reranked, N = 8 | 3179.77 | 0.4716 | -0.5249 | **-0.0457** |

**The one apparent composition win disappears.** `oracle_reranked` at N = 8 goes from
+0.0182 against an extrapolated terminal value to **-0.0512** against the measured curve:
plain oracle-selected best-of-N reaches **0.9965** at 3197.98 tokens per molecule, where
reranking 8 guided drafts reaches 0.9453 for the same money. Spending the budget on guidance
first and selection second is *worse* than spending all of it on selection.

`head_reranked` is below the measured head-selected curve at N = 4 (-0.0182) and N = 8
(-0.0457), and indistinguishable from it at N = 2 (+0.0018). D4's verdict is unchanged.

Every strand-A conclusion is also unchanged and now carries no extrapolation: A1 and A2 sit
further below both curves than the pre-registered pricing said, and A3's advantage over the
head-selected curve grows slightly, to +0.1000.

---

## C28.7 The pre-registration scored, verbatim, including failures

### C28.7.1 Validity gates

| gate | criterion | result |
|---|---|---|
| G1 | A1 k=8 reproduces `pilot_50k_p2_guided_hbd_count` with residual 0.0 | **PASS**, residual 0.0 on hit rate and tokens, aggregate and all three seeds |
| G2 | A2 k=8 reproduces `c23_guided_L4_lam1_hbd_count` | **PASS**, residual 0.0 |
| G3 | A3 k=8 reproduces `c23_guided_L4_lam2_hbd_count` | **PASS**, residual 0.0 |
| G4 | `processed_tokens_actual mod (k+1) == 0` everywhere | **PASS**, max residual 0 |
| G5 | N=1 identical across rerank arms | **PASS**, residual 0.0 |
| G6 | first 512 drafts are the published 512 | **PASS**, 0 / 0 / 0 mismatches |
| G7 | per-draft token attribution sums to the meter | **PASS**, residual 0 / 0 / 0 |

All seven pass. No gate was replaced, waived or reinterpreted.

### C28.7.2 Decision rules

| rule | pre-registered criterion | result |
|---|---|---|
| D1 | REFUTED AS STATED iff max/min tokens within one strand exceeds 2.0 | **REFUTED AS STATED** — 10.28x, 10.31x, 10.30x, 10.24x, 11.06x, 10.27x on all six strands |
| D2 | PRODUCTIVE / NULL / HARMFUL on `hit(32) - hit(8)` | **NULL on five strands** (0.0078, -0.0185, 0.0044, 0.0020, 0.0077) and **HARMFUL on C1** (-0.0271, all three seeds negative). PRODUCTIVE nowhere |
| D3a | below the oracle-selected curve at k=32 on every strand | **below on all six**: -0.6206, -0.5767, -0.3600, -0.5498, -0.1782, -0.7506 |
| D3b | above/below the head-selected curve at k=32 | **below** on A1 (-0.1790), A2 (-0.1352), C1 (-0.1774), C3 (-0.0768); **above** on A3 (+0.0807, interval includes zero) and C2 (+0.1942, interval excludes zero) |
| D4 | usable compute axis iff above the head-selected curve at some N with consistent seeds | **not met** — `head_reranked` is below at every N (-0.0166, -0.0123, -0.0334, -0.0139) |
| D5 | if the k profile is flat, the null truncation control is the explanation | **holds on hbd_count and qed** (spans 0.0320 / 0.0278 / 0.0774 and 0.0116), **fails on aromatic_rings** (spans 0.2739 and 0.3567), where k = 2 collapses the deployed arm to 0.1996 |

### C28.7.3 The ten predictions

| prediction | statement | result |
|---|---|---|
| **P1** | cost within +/-10% of 401.619141*(k+1)/9 on A1 | **HOLDS** — max absolute relative error 0.0519 (at k=2) |
| **P2** | hit rate non-decreasing in k on A1: `hit(32) >= hit(2)` | **FAILS** — 0.3065 against 0.3283. The knob runs slightly *backwards* on the deployed arm |
| **P3** | `hit(32) - hit(8)` on A1 below +0.10 | **HOLDS** — 0.0078 |
| **P4** | below the oracle-selected curve at k=32 on all strands | **HOLDS** — -0.6206, -0.5767, -0.3600 on the pre-registered anchor, and -0.5498, -0.1782, -0.7506 on strand C |
| **P5** | A1 at k=32 below the head-selected curve | **HOLDS** — -0.1790 |
| **P6** | A3 at k=32 above the head-selected curve | **HOLDS** — +0.0807, though the seed interval [-0.0710, 0.2313] includes zero |
| **P7** | validity at k=32 below validity at k=2 on every strand | **HOLDS** — 0.9941 vs 0.9993 (A1), 0.9941 vs 0.9967 (A2), 0.9915 vs 0.9941 (A3), 0.9948 vs 0.9980 (C1), 0.9544 vs 0.9909 (C2), 0.9935 vs 1.0000 (C3) |
| **P8** | `oracle_reranked` at N=8 above 0.90 **and** below the oracle-selected curve | **FAILS as scored** — 0.9453 is above 0.90, but the pre-registered comparison returns **+0.0182**, i.e. above the curve. The failure is in the comparison rule, not the arm: it prices a 3179.77-token point against a curve measured only to 1421.98. §C28.6 measures the curve out to that budget and the advantage becomes **-0.0512**, so the prediction's substance was right and its pre-registered scoring rule was wrong |
| **P9** | `head_reranked` at N=8 above the head-selected curve's terminal value 0.4855 | **FAILS** — 0.4716, i.e. -0.0139 |
| **P10** | A1's max/min token ratio exceeds 3.0 | **HOLDS** — 10.28 |

Seven of ten hold; **P2, P8 and P9 fail** and are reported as failures.

P2 is the interesting one. It was written expecting a wider candidate set to give the
property term more to work with; instead the deployed arm's best cell is the cheapest.
P8's failure is a failure of the pre-registered *pricing rule* under extrapolation and is
the reason §C28.6 exists. P9 failed in the direction that makes guidance look worse, and is
reported unchanged.

---

## C28.8 What this changes in C26 §C26.4.4 — a conflict for the owner to merge

**§C26.4.4 is not edited here.** The following is the conflict, stated for whoever merges it.

C26 §C26.4.4 currently claims, and `docs/TODO.md`'s C26 entry repeats:

> **The structural finding may matter more: guidance has no compute knob.** All 46 arms sit
> inside a 5.1-17.0% token band while best-of-N spans 32x. Guided decoding does not merely
> lose at the budget we matched at — **it cannot be offered more compute within the method as
> specified.**

1. **The last sentence is false and must be withdrawn.** Guidance can be offered 10.28x more
   compute inside the method exactly as specified, by the hyperparameter `top_k_candidates`
   that the method already has, changing no weight and training nothing. Cost is exactly
   (k+1) x base (gate G4, residual 0). C26 measured a band across 46 arms that all shared
   k = 8; that band is a fact about the arms, not about the method.

2. **The observation about the band is still true as an observation** and should be
   re-scoped: *all 46 arms sit in a 5.1-17.0% band because every one of them was run at the
   frozen default `top_k_candidates: 8`.*

3. **The conclusion C26 drew from it is strengthened, not weakened.** The replacement
   sentence, which C28 has measured rather than assumed:

   > Guided decoding *can* be offered more compute within the method as specified — the
   > candidate-set size k is a compute knob and spans 10.28x — and it converts none of it
   > into accuracy. Over that span (140.83 to 1447.42 processed tokens per molecule) the
   > deployed hbd_count arm moves by **-0.0218** in hit rate, best cell the cheapest, while
   > oracle-selected best-of-N gains **0.6834** and head-selected best-of-N gains **0.2922**
   > over the identical budgets.

4. **C26's D1/E1 arm counts are unaffected.** C28 adds new arms; it does not move any
   existing one. Anyone re-running C26's pricing over the C28 cells should note that at
   k = 2 and k = 4 several cells sit *above* both curves (A1 k=2: +0.0846 vs oracle,
   +0.1349 vs head; A3 k=2: +0.2499 / +0.2975) — because at 140 tokens per molecule the
   comparators have barely started. Those are cheap-end wins, not compute-scaling wins, and
   they say guidance is efficient at small budgets, which C26 never disputed.

---

## C28.9 Limitations

1. **k and lambda are not crossed.** Three frozen lambda values, one k grid. §19 showed the
   lambda response is an inverted U with an optimum at 2-4; whether the k profile changes
   shape at other lambda is unmeasured. A3 (lambda = 2) does have a larger k = 2 to k = 4
   step than A1 (lambda = 1), which is weak evidence that the interaction is not nil.
2. **Three generation seeds, one head seed.** Every D2 difference has three seeds that do
   not share a sign, so the *sign* of the k effect is unresolved on every strand. That is
   reported rather than decorated: n = 3 cannot resolve a 0.0078 difference. No C28 arm is
   replicated across head-training seeds, and C25 measured a 0.0916 head-seed span on a
   single guided arm — larger than every D2 difference here.
3. **One anchor carries the pre-registered argument.** hbd_count has three strands;
   aromatic_rings has two and qed one, and all three strand-C arms were run last, as
   priority 3. The aromatic_rings result (k = 2 collapses both arms) is a real qualification
   of D5 and shows the k profile is property-dependent; qed has no mid-layer strand at all,
   so nothing here says whether a mid-layer qed arm behaves like A2/A3 or like C3.
4. **The composition was run at k = 8 only.** Drafts could be made cheaper (k = 2 costs 140.83
   tokens, so 8 drafts would cost ~1127 rather than 3179.77) and the composition re-priced.
   That is the obvious follow-up and C28 did not run it.
5. **Linear interpolation in tokens**, inherited from C26 and C27. Both curves are concave in
   tokens, so linear interpolation between bracketing grid points slightly understates the
   comparator and biases every advantage **upward, in guidance's favour**. The C28 conclusions
   are all negative for guidance, so this bias works against them.
6. **`actual` accounting only**, consistent with C26 and C27 and for the same reason: under
   `full_recompute` best-of-N saturates to 1.0000 and cannot discriminate.
7. **A statement about SMILES, not about FUDGE.** The flat k profile is explained by the null
   molecular truncation control. On GPT-2 the same control destroys 47.5-85.7% of the base
   hit rate, so nothing here licenses "k is a dead knob for guided decoding in general".
8. **`pooling.py` and `generality.py` still hard-code k = 8** and were not touched. C28
   swept k only for the molecular `guidance.py` path.

---

## REPRODUCE

```bash
# the k sweep: every (strand, k) is its own directory and a completed cell is never redone.
# The k=8 cell of each strand is a blocking validity gate against a frozen artefact.
.venv/bin/python scripts/23_k_sweep.py --strand A1 --k 8      # gate G1 alone, first
.venv/bin/python scripts/23_k_sweep.py --all                  # A1, A2, A3, C1, C2, C3

# the composition: N guided drafts, reranked by the oracle and by the deployed head
.venv/bin/python scripts/23_guided_drafts.py --property hbd_count

# POST HOC (C28.6): best-of-N measured out to N=80 so the composition's budget is not
# extrapolated.  scripts/22_head_selected_bestofn.py is unmodified.
.venv/bin/python scripts/22_head_selected_bestofn.py --dataset pilot_50k_p2 \
    --property hbd_count --n-max 80 --grid 1 2 4 8 9 16 24 32 48 64 72 80 \
    --out c28_bon_extended_hbd_count

# the frontier and the pre-registration scoring: reads existing artefacts only,
# generates nothing, trains nothing
.venv/bin/python scripts/23_summarise_c28.py

# binding tests
.venv/bin/python -m pytest tests/test_k_sweep.py -q
```

Artefacts: `outputs/c28_prereg/` (pre-registration and SHA-256 lock),
`outputs/c28_ksweep_<property>_L<layer>_lam<x>_k<k>/` (one per cell, with run contexts and
molecules), `outputs/c28_guided_drafts_hbd_count/`, `outputs/c28_bon_extended_hbd_count/`,
`outputs/c28_summary/c28_metrics.json`, `outputs/c28_figures/c28_k_sweep_frontier.png` and
`outputs/c28_logs/`.
