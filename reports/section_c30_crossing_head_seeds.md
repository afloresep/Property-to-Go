# C30 — does the crossing survive the probe-training seed?

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-03. Conflicts with existing sections are **listed for the owner to merge** in
§C30.8, not edited in place.

**Headline, stated before the detail so it cannot be buried.** C28's crossing — guided
decoding above the *oracle-selected* best-of-N frontier at small budgets — **replicates
across eight probe-training seeds**, and the deployed configuration's margin comes out
**larger** than C28 reported, not smaller. The pre-registered verdict is nevertheless
**UNINTERPRETABLE**, because a validity screen written into C30.0.8 fired on one of 56
(cell, head seed) points. Both facts are reported, in that order, in §C30.5.

Two of the eight C28 cells do not survive, one of them reversing sign with an interval
that excludes zero on the negative side. Those are in §C30.4.3.

---

## C30.0.1 Why this experiment exists

C28 found the only positive result in this project. Priced against the **oracle-selected**
best-of-N frontier at their own budgets, 8 of 30 guided cells sit above it:

| strand | configuration | k | advantage vs oracle best-of-N | t interval (2 df) excludes 0 |
| --- | --- | ---: | ---: | :---: |
| A3 | hbd_count, probe point 4, λ=2 | 2 | +0.2499 | yes |
| A3 | hbd_count, probe point 4, λ=2 | 4 | +0.2093 | yes |
| C2 | aromatic_rings, probe point 3, λ=2 | 4 | +0.1690 | yes |
| A2 | hbd_count, probe point 4, λ=1 | 2 | +0.1325 | yes |
| A1 | hbd_count, probe point 12, λ=1 (**deployed**) | 2 | +0.0846 | yes |
| A3 | hbd_count, probe point 4, λ=2 | 8 | +0.0267 | no |
| A2 | hbd_count, probe point 4, λ=1 | 4 | +0.0218 | no |
| C2 | aromatic_rings, probe point 3, λ=2 | 2 | +0.0007 | no |

**Every one of those cells was generated with a single probe-training seed, 1234.**

C29 then measured what a probe-training seed is worth, at eight seeds across seven arms:
the between-head-seed sd of end-to-end hit rate is **0.0142 to 0.0366**. Three of the eight
margins above are smaller than 0.0366, and one (C2 k=2, +0.0007) is smaller than every
head-seed sd C29 measured. C29's own transferable finding is that *"one trained probe with
error bars over generation seeds cannot distinguish the method from its own noise"*.

So the project's headline positive result is produced by exactly the protocol the project's
own methodological finding says is inadequate. **C30 runs the missing replication.** This is
not a robustness appendix; it is the experiment that decides whether there is a result.

This project has retracted the 25× head-seed multiplier, halved C23's Rule A after finding
an effective-λ confound, and reversed C23's Rule B twice. **The prior that the crossing
survives unchanged should be treated as weak**, and this pre-registration is written to make
a negative outcome as reportable as a positive one.

## C30.0.2 What is run — fixed here, in full

Four strands, transcribed from `scripts/23_k_sweep.py::STRANDS` and **not** re-derived:

| strand | property | probe point | λ | head file family |
| --- | --- | ---: | ---: | --- |
| A1 | hbd_count | 12 | 1.0 | `c29_heads/head_hbd_count_last1_L12_seed<S>.pt` |
| A2 | hbd_count | 4 | 1.0 | `c29_heads/head_hbd_count_last1_L4_seed<S>.pt` |
| A3 | hbd_count | 4 | 2.0 | `c29_heads/head_hbd_count_last1_L4_seed<S>.pt` |
| C2 | aromatic_rings | 3 | 2.0 | `c29_heads/head_aromatic_rings_last1_L3_seed<S>.pt` |

These are the four strands that produce a winning cell. C1 (aromatic_rings, probe point 12,
λ=1) and C3 (qed, probe point 12, λ=1) produce none and are **not** run: no cell of theirs is
above the oracle curve at any k, so there is nothing for a replication to overturn. This is
stated now so that their absence is not later read as a selective omission.

`k ∈ {2, 4}`. k = 8, 16 and 32 are not re-run: the only winning cell above k = 4 is A3 at
k = 8, which is C26's D2 arm and was **already** replicated across head seeds by C26 §D2b
and by C29 §R6, both of which are cited rather than repeated.

Head seeds: **1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901** — the eight C29 trained, in
full, with no seed dropped for any reason.

Generation seeds: **101, 202, 303**, unchanged from C28, 512 molecules each, so every cell is
1,536 molecules and is directly comparable to its C28 counterpart.

Total: 4 strands × 2 k × 8 head seeds = **64 cells**, 98,304 molecules.

**Nothing else varies.** Same frozen generator at the same pinned revision, same frozen
target intervals and windows, same `throughout` condition, same ε, same batch size, same
`cached` backend, same `guidance.guided_sample` entry point. The generator is not fine-tuned,
no LoRA, no RL, no activation edit, no second generator, no alternative serialization.

## C30.0.3 Validity gates, run and reported BEFORE any decision rule

**G1 — checkpoint identity.** For each of the three (property, probe point) families, the
C29 seed-1234 checkpoint must be **tensor-by-tensor identical** to the checkpoint C28 used
(`pilot_50k_heads_p2/head_hbd_count_frozen_state.pt`,
`c17_probe_layers/head_hbd_count_frozen_state_L4.pt`,
`c17_probe_layers/head_aromatic_rings_frozen_state_L3.pt`). Maximum absolute difference must
be exactly 0.0 on every tensor, and the binner edges must match exactly. C29 established
this for six families; C30 re-establishes it for the three it uses rather than citing it.

**G2 — end-to-end reproduction.** At head seed 1234, all eight cells must reproduce their
C28 counterpart **exactly**: hit rate residual 0.0 and processed-token residual 0.0 on every
one of the three generation seeds, and the returned molecule strings identical.

**G3 — cost identity.** `processed_tokens_actual mod (k+1)` must be 0 in every cell, as C28's
G4 required.

**If G1 or G2 fails on any cell, C30 stops and reports the failure.** No decision rule below
is scored on a run whose gate failed.

## C30.0.4 The statistic, fixed now

For each of the 8 C28 winning cells, and for each head seed S, the advantage is

```
adv(cell, S) = hit_rate(cell, S) - oracle_curve_interpolated_at(tokens(cell, S))
```

where `oracle_curve_interpolated_at` is **C26's measured oracle-selected best-of-N curve**,
linearly interpolated in processed tokens per molecule, using the *unmodified*
`scripts/21_summarise_c26.py::interp` and `scripts/23_summarise_c28.py::price`. The
best-of-N curve does **not** depend on the head — it selects with RDKit — so it is held fixed
across head seeds, and that is the point: only guidance moves.

Across the 8 head seeds we report the mean, the sd, and a **Student t interval on 7 df**
(`t₀.₉₇₅,₇ = 2.364624`). We do **not** report a percentile bootstrap over head seeds as the
primary statistic; at n = 8 it is no longer degenerate, so it is published as a secondary
figure, but the t interval is the pre-committed one. Per-head-seed values are printed in
full for all 64 cells so the reader can see the entire sample.

Advantage against the **head-selected** curve is also computed and reported, but is
**secondary**: that curve is itself a function of the head, so it moves with the head seed
and the comparison is not held fixed. No decision rule below depends on it.

## C30.0.5 Decision rules — scored as written

| # | rule | fires iff |
| --- | --- | --- |
| **D1** | **the crossing survives** | the head-seed mean advantage over the oracle curve is > 0 on **≥ 5 of the 8** C28 winning cells |
| **D2** | **the crossing is confirmed** | ≥ 3 of the 8 cells additionally have a t interval on 7 df strictly above 0 |
| **D3** | **the deployed cell survives** | A1 at k = 2 — the only deployed configuration among the eight — has a head-seed mean > 0 with a t interval strictly above 0 |
| **D4** | **sign stability** | ≥ 75% of the 64 (cell, head seed) points that correspond to a C28 winning cell have a positive advantage |
| **D5** | **the crossing is refuted** | fewer than 3 of the 8 cells keep a positive head-seed mean |

**The verdict rule, fixed now:**

- **CONFIRMED** iff D1 and D2 both fire.
- **SURVIVES, UNDERPOWERED** iff D1 fires and D2 does not. The wording committed to in
  advance is: *"the crossing's sign replicates across probe seeds and its size is not
  resolved at eight."* It may **not** be reported as "the crossing is confirmed".
- **REFUTED** iff D5 fires.
- **AMBIGUOUS** in any remaining case, reported as such and not as a win.

**D6 — the honesty rule, which fires regardless of the others.** For every cell whose
head-seed sd exceeds its own C28 single-seed margin, C30 reports that cell as **"not
resolvable at one probe seed"** and states that C28's published margin for it was a draw from
a distribution wider than the margin. This fires on evidence about *C28's protocol* and is
scored even if D1 and D2 both fire.

## C30.0.6 Predictions, committed before the run

Scored in the section. Falsified predictions are reported as falsified.

1. **P1** — ≥ 6 of the 8 cells keep a positive head-seed mean. *(Rationale: five of the eight
   C28 margins exceed 0.08, and C29's largest measured head-seed sd is 0.0366.)*
2. **P2** — A3 at k = 2, the largest margin (+0.2499), keeps a t interval excluding zero.
3. **P3** — A1 at k = 2, the deployed cell (+0.0846), keeps a t interval excluding zero.
   *(0.0846 against a deployed head-seed sd of 0.0241 gives a standard error of ≈0.0085 at
   n = 8, so this should clear comfortably if C29's sd transfers.)*
4. **P4** — C2 at k = 2 (+0.0007) does **not** keep a t interval excluding zero, and its
   head-seed mean may be of either sign. Predicted to fail D6 as well.
5. **P5** — at least one of the eight cells changes the *sign* of its mean advantage.
6. **P6** — the head-seed sd of the advantage is **larger** than the head-seed sd of the hit
   rate alone for at least one cell, because the budget also moves with the head seed and the
   interpolated comparator moves with it.
7. **P7** — the four strands' head-seed sds fall inside C29's measured 0.0142–0.0366 band on
   at least 3 of 4 strands.

## C30.0.7 What C30 will NOT do

Stated so its absence is not read as an omission.

- No new head is trained. C30 uses C29's eight checkpoints exactly as they are on disk.
- No k outside {2, 4}; no λ sweep; no new probe point; no pooling variant.
- No new best-of-N run. C26's oracle curve and C27's head-selected curve are read, not
  regenerated.
- No quality panel beyond validity and uniqueness, both of which are recorded per cell.
- No second generator and no text replication.
- The one permitted DAgger round remains spent and is not repeated.

## C30.0.8 What would make C30 uninterpretable rather than negative

| condition | why it would void the experiment |
| --- | --- |
| G1 or G2 fails | the cells are not comparable to C28's and nothing can be concluded |
| any cell's validity falls below 0.90 | a hit rate over molecules that do not parse is not a controller |
| fewer than 8 head seeds available for any strand | the t interval's df would differ between cells and the comparison across cells would not be like for like |

Any of these is reported as **UNINTERPRETABLE** and the decision rules are left unscored.

## C30.0.9 The reporting rule

Whatever comes out, the section states it in this order: gates, then per-head-seed tables in
full, then the decision rules scored as written, then the predictions scored, then what it
changes elsewhere in the report. A negative outcome is written up at the same length as a
positive one, and `reports/pilot_report.md` §22.2 and §23 are updated to match **whichever
way it goes** — including deleting the crossing from the abstract if D5 fires.

---

## C30.1 What was run

64 cells: 4 strands × 2 values of k × 8 head seeds, 3 generation seeds of 512
molecules each, **98,304 molecules**. Nothing else varied: same frozen generator at the
same pinned revision, same frozen intervals and windows, same `throughout` condition,
same ε, same batch size, same `cached` backend.

`scripts/24_c30_crossing_head_seeds.py` does **not** contain a copy of C28's generation
path. It imports `scripts/23_k_sweep.py::run_cell` and points the strand at a different
head checkpoint through a scoped, restored mutation of C28's own `STRANDS` table. A copy
could agree with C28 today and drift tomorrow, and gate G2 would then pass for the wrong
reason.

Wall time 3,903 s on the RTX 4090. Cost is reported as processed tokens throughout; the
wall figure is given only because it bounds what a replication costs.

## C30.2 Validity gates — all three pass, before any decision rule

**G1 — checkpoint identity.** The C29 seed-1234 checkpoints must be tensor-by-tensor
identical to the ones C28 used. C29 established this for six families; C30 re-establishes
it for the three it uses, because a gate cited is a gate not run.

| family | C28 checkpoint | C29 checkpoint | max abs diff | binner edges identical |
| --- | --- | --- | ---: | :---: |
| `hbd_count_L12` | `pilot_50k_heads_p2/head_hbd_count_frozen_state.pt` | `c29_heads/head_hbd_count_last1_L12_seed1234.pt` | **0.0** | True |
| `hbd_count_L4` | `c17_probe_layers/head_hbd_count_frozen_state_L4.pt` | `c29_heads/head_hbd_count_last1_L4_seed1234.pt` | **0.0** | True |
| `aromatic_rings_L3` | `c17_probe_layers/head_aromatic_rings_frozen_state_L3.pt` | `c29_heads/head_aromatic_rings_last1_L3_seed1234.pt` | **0.0** | True |

Maximum over all families: **0.0**. Gate passes.

**G2 — end-to-end reproduction.** At head seed 1234 every cell must reproduce its C28
counterpart exactly. A summary statistic can agree by accident; 1,536 SMILES strings
cannot, so the molecules are compared one by one.

| cell | hit-rate residual | token residual | molecules identical |
| --- | ---: | ---: | ---: |
| A1_k2 | 0.0 | 0.0 | **1536 / 1536** |
| A1_k4 | 0.0 | 0.0 | **1536 / 1536** |
| A2_k2 | 0.0 | 0.0 | **1536 / 1536** |
| A2_k4 | 0.0 | 0.0 | **1536 / 1536** |
| A3_k2 | 0.0 | 0.0 | **1536 / 1536** |
| A3_k4 | 0.0 | 0.0 | **1536 / 1536** |
| C2_k2 | 0.0 | 0.0 | **1536 / 1536** |
| C2_k4 | 0.0 | 0.0 | **1536 / 1536** |

**12288 of 12288 molecules identical**, hit-rate and token residuals exactly
0.0 on every one of the three generation seeds in all 8 cells.
Head seed 1234 is C28, bit for bit, so the other seven seeds are measuring the probe and
nothing else. Gate passes.

**G3 — cost identity.** `processed_tokens_actual mod (k+1)` is **0**
in all 64 cells. Gate passes.

## C30.3 The result, cell by cell

Advantage over C26's oracle-selected best-of-N curve, interpolated in processed tokens at
each cell's own budget. The curve selects with RDKit and does not depend on the head, so
it is **held fixed across head seeds** — the whole design is that only guidance moves.

| cell | property | probe point | λ | k | C28, one seed | mean over 8 | sd | t interval, 7 df | seeds positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A3_k2 | hbd_count | 4 | 2 | 2 | +0.2499 | **+0.2691** | 0.0411 | [+0.2348, +0.3035] | 8/8 |
| A3_k4 | hbd_count | 4 | 2 | 4 | +0.2093 | **+0.2000** | 0.0366 | [+0.1694, +0.2306] | 8/8 |
| C2_k4 | aromatic_rings | 3 | 2 | 4 | +0.1690 | **+0.1052** | 0.0557 | [+0.0586, +0.1517] | 8/8 |
| A2_k2 | hbd_count | 4 | 1 | 2 | +0.1325 | **+0.1326** | 0.0265 | [+0.1104, +0.1547] | 8/8 |
| A1_k2 | hbd_count | 12 | 1 | 2 | +0.0846 | **+0.1044** | 0.0179 | [+0.0895, +0.1194] | 8/8 |
| A2_k4 | hbd_count | 4 | 1 | 4 | +0.0218 | +0.0207 | 0.0334 | [-0.0073, +0.0486] | 6/8 |
| C2_k2 | aromatic_rings | 3 | 2 | 2 | +0.0007 | -0.0341 | 0.0358 | [-0.0640, -0.0042] | 2/8 |

`t₀.₉₇₅,₇ = 2.364624`. Five of the seven re-run cells have intervals strictly above zero.

### C30.3.1 Every per-head-seed value, in full

C30.0.4 promised the whole sample rather than a summary of it.

| cell | hs1234 | hs2345 | hs3456 | hs4567 | hs5678 | hs6789 | hs7890 | hs8901 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3_k2 | +0.2499 | +0.3004 | +0.2290 | +0.2063 | +0.2892 | +0.3345 | +0.2825 | +0.2614 |
| A3_k4 | +0.2093 | +0.1968 | +0.1308 | +0.1673 | +0.2463 | +0.2216 | +0.1996 | +0.2283 |
| C2_k4 | +0.1690 | +0.1380 | +0.0923 | +0.1426 | +0.1599 | +0.0668 | +0.0596 | +0.0133 |
| A2_k2 | +0.1325 | +0.1320 | +0.1024 | +0.1041 | +0.1247 | +0.1874 | +0.1334 | +0.1440 |
| A1_k2 | +0.0846 | +0.1077 | +0.1306 | +0.1235 | +0.0870 | +0.0931 | +0.1178 | +0.0911 |
| A2_k4 | +0.0218 | +0.0276 | -0.0328 | -0.0272 | +0.0336 | +0.0462 | +0.0359 | +0.0603 |
| C2_k2 | +0.0007 | -0.0277 | +0.0133 | -0.0753 | -0.0117 | -0.0640 | -0.0260 | -0.0823 |

Head seed 1234 is C28's own value in every row, by G2.

## C30.4 What the numbers say

### C30.4.1 The deployed configuration is the strongest replication in the table

A1 at k = 2 is `hbd_count` at probe point 12 with λ = 1 — **the configuration §16 deploys**,
not a mid-network arm and not a re-tuned λ. It is the one cell in C28's table that cannot
be dismissed as a post-hoc pick, and it is the cell that replicates best:

- head-seed mean **+0.1044**, interval [+0.0895, +0.1194]
- **8 of 8** head seeds positive
- head-seed sd **0.0179**, the smallest of the seven cells
- C28 reported **+0.0846** at one seed; the eight-seed mean is
  **+0.0198** *larger*

C28's single-seed number was, if anything, **pessimistic** about the deployed arm. That is
the opposite of the direction a selection effect would produce, and it is worth saying
plainly because this project has three times found a headline shrink under scrutiny.

### C30.4.2 The large mid-network margins hold

- **A3_k2** (hbd_count, probe point 4, λ=2, k=2): C28 +0.2499 → **+0.2691** [+0.2348, +0.3035], 8/8 positive
- **A3_k4** (hbd_count, probe point 4, λ=2, k=4): C28 +0.2093 → **+0.2000** [+0.1694, +0.2306], 8/8 positive
- **A2_k2** (hbd_count, probe point 4, λ=1, k=2): C28 +0.1325 → **+0.1326** [+0.1104, +0.1547], 8/8 positive

All three keep every seed on the same side of zero.

### C30.4.3 Two cells do not survive, and one of them reverses

**C2 at k = 2 reverses sign.** C28 reported **+0.0007**. Over eight
probe seeds the mean is **-0.0341** with an interval
[-0.0640, -0.0042] that **excludes zero on the negative side**, and only
**2 of 8** seeds are positive. C28's `+0.0007` was not a small win; it was
the largest of eight draws from a distribution centred below zero. This cell is deleted
from the crossing.

**A2 at k = 4 is not resolved.** Mean **+0.0207**, interval
[-0.0073, +0.0486] containing zero, 6 of 8 positive. It is
reported as unresolved rather than as a win or a loss.

### C30.4.4 D6 — what C28's protocol could and could not have seen

D6 was written to fire on evidence about C28's *protocol*, independently of whether C30
came out positive. It fires on two cells:

| cell | C28's single-seed margin | head-seed sd of the advantage | margin ÷ sd |
| --- | ---: | ---: | ---: |
| A2_k4 | +0.0218 | 0.0334 | 0.65 |
| C2_k2 | +0.0007 | 0.0358 | 0.02 |

For both, **the spread across probe seeds is larger than the margin C28 published**. Those
two rows of C28's table were never resolvable at one probe seed, whatever their sign, and
they are now labelled **not resolvable at one probe seed** wherever they appear. The other
five cells have margins between 2.5 and 6.5 times their own head-seed sd and were
resolvable — which is why they replicated.

### C30.4.5 The probe seed moves validity, not only accuracy

Not pre-registered as a finding and reported because the screen surfaced it. On C2 at
k = 4 — `aromatic_rings`, probe point 3, λ = 2 — validity falls monotonically with the
advantage across head seeds:

| head seed | validity | tokens/molecule | advantage |
| ---: | ---: | ---: | ---: |
| 1234 | 0.9792 | 253.53 | +0.1690 |
| 2345 | 0.9603 | 269.17 | +0.1380 |
| 3456 | 0.9036 | 311.04 | +0.0923 |
| 4567 | 0.9616 | 254.32 | +0.1426 |
| 5678 | 0.9811 | 239.62 | +0.1599 |
| 6789 | 0.9212 | 286.94 | +0.0668 |
| 7890 | 0.9030 | 322.68 | +0.0596 |
| 8901 | 0.8301 | 372.56 | +0.0133 |

The mechanism is legible: a probe that steers into invalid strings produces **longer**
molecules, which cost **more tokens**, which prices the arm against a further-along point
of the best-of-N curve. Validity, cost and advantage are one phenomenon here, not three.
§19 already established that λ = 2 is near the edge where validity starts to go; C30 adds
that **which side of that edge you land on depends on the probe seed.**

This is the strongest single argument in the project for reporting the probe seed: at
`hs8901` this arm returns molecules that are 17% unparseable, and nothing in a
single-seed protocol would have shown it.

## C30.5 The pre-registration, scored as written

### C30.5.1 The verdict is UNINTERPRETABLE, and here is exactly why

C30.0.8 named three conditions that void the experiment. One fired:

> | any cell's validity falls below 0.90 | a hit rate over molecules that do not parse is not a controller |

Under C30's own naming a cell is one (strand, k, head seed) directory, so the screen is
scored per head seed. It fires on **1 of 56**
points: **C2_k4 at head seed 8901**, validity **0.8301**.

C30.0.8 says such a run is *"reported as UNINTERPRETABLE and the decision rules are left
unscored"*. **That is the score, and it is recorded as the score.** One point in 56 voiding
the other 55 is the rule as written; it is also the least informative possible response to
the data, which is a defect in how I wrote the rule and not a fact about the experiment.
The rule is **not** rewritten after the fact. Instead the conservative repair is computed
below and labelled as post hoc.

### C30.5.2 The decision rules, computed and reported anyway

Reported so that a reader can see what the screen voided rather than take it on trust.

| # | rule | measured | fires |
| --- | --- | --- | :---: |
| **D1** | mean advantage > 0 on ≥ 5 of 8 cells | 6 of 8 | **True** |
| **D2** | ≥ 3 of 8 additionally have a t interval above 0 | 5 of 8 | **True** |
| **D3** | the deployed cell A1 k=2 survives | +0.1044, CI [+0.0895, +0.1194] | **True** |
| **D4** | ≥ 75% of (cell, head seed) points positive | 48 of 56 = 0.8571 | **True** |
| **D5** | fewer than 3 cells positive → refuted | 6 of 8 | False |
| **D6** | any cell's head-seed sd exceeds its C28 margin | 2 cells | **True** |

D1 is scored **conservatively against C30's own claim**: A3 at k = 8 was not re-run
(C30.0.2) and is counted as *not* positive, so the denominator stays at the
pre-registered 8 while the numerator can only reach 7.

### C30.5.3 S1 — the conservative repair, POST HOC and labelled as such

The screen fired on one head seed of one cell. Dropping **only that head seed** would be
selecting on the outcome, so S1 drops **the entire cell, all eight head seeds**, and
re-scores D1–D4 against their **original thresholds of 5 and 3 out of 8** — not rescaled to
the smaller denominator, because rescaling would make the rule easier to pass after
dropping a cell.

The cell dropped is **C2_k4**, whose head-seed mean is
**+0.1052** with an interval excluding zero. **S1 therefore
removes a cell that supports C30's own headline.** If the verdict survives that, it
survives the screen.

| | pre-registered | S1 (post hoc) |
| --- | --- | --- |
| cells scored | 8 | 6 |
| positive means | 6 | 5 |
| intervals above zero | 5 | 4 |
| D1 (≥ 5) | True | True |
| D2 (≥ 3) | True | True |
| D3 (deployed cell) | True | True |
| **verdict** | **UNINTERPRETABLE** | **CONFIRMED** |

**S1 returns CONFIRMED.** The honest one-line summary of C30 is therefore: *the
crossing replicates across probe seeds; the pre-registered screen voided the run on one
point in 56, and the crossing still replicates when the whole offending cell is thrown
away.*

## C30.6 The predictions, scored

| # | prediction | measured | holds |
| --- | --- | --- | :---: |
| P1 | ≥ 6 of 8 cells keep a positive mean | 6 of 8 | **True** |
| P2 | A3 k=2 keeps an interval excluding zero | +0.2691 | **True** |
| P3 | the deployed A1 k=2 keeps an interval excluding zero | +0.1044 | **True** |
| P4 | C2 k=2 does NOT survive, and fires D6 | -0.0341, D6=True | **True** |
| P5 | at least one cell changes sign | 1 cell | **True** |
| P6 | advantage sd exceeds hit-rate sd somewhere | yes, all 7 cells | **True** |
| P7 | hit-rate sd inside C29's 0.0142–0.0366 band on ≥ 3 of 4 strands | 3 of 4 | **True** |

**7 of 7 hold; none is falsified.** That is an unusual
outcome for this project — C24 falsified five of its own, C29 four of eight — and it should
be read as evidence that C29 had already characterised the head-seed effect well enough to
predict C30, not as evidence that C30 was easy.

P7's one miss is instructive rather than adverse: strand A3's mean hit-rate head-seed sd is
**0.0368**, just
above C29's measured band, so the strongest arm is also the noisiest across probe seeds.

P6 holds on **all seven** cells, not just one: the sd of the *advantage* exceeds the sd of
the hit rate everywhere, because the budget moves with the head seed too and the
comparator is read at a different point of the curve. Most extreme is C2 k=4, where the
hit-rate sd is 0.0186
and the advantage sd is 0.0557
— a three-fold amplification. **A replication that varied the probe seed but held the
budget fixed would understate the variance by that factor.**

## C30.7 The cell that was not re-run

**A3 at k = 8** is one of C28's eight winning cells and C30 did not re-run it. C30.0.2 said
so before the run, and the reason is that it is already replicated: it is C26's D2 arm,
`c23_guided_L4_lam2_hbd_count`, replicated across three head seeds by C26 §D2b and across
eight by **C29 §R6**. C29's numbers are cited rather than regenerated:

- head-seed mean advantage over C23's own comparator **+0.0490**, interval [+0.0135, +0.0845]
- against C26's corrected frontier the interval contains zero (fires = False)

So that cell is **comparator-dependent**, exactly as §22.5 records. It is not counted as a
surviving cell anywhere in C30, and D1's denominator keeps it at 8 rather than dropping to
7, which scores C30 against itself.

### C30.7.1 A1 at k = 4 — run, and not scored

C30.0.2 specifies *"4 strands × 2 k"*, which is eight combinations. Seven of them are C28
winning cells. The eighth, **A1 at k = 4**, is not — C28 placed it *below* the oracle curve —
so no pre-registered rule applies to it. It was nevertheless generated at all eight head
seeds, and it is reported here because **a cell that was generated and then never mentioned
is indistinguishable, to a reader, from a cell that was dropped.**

- head-seed mean **-0.0081**, interval [-0.0277, +0.0114]
- **3 of 8** head seeds positive; minimum validity 0.9967

| head seed | 1234 | 2345 | 3456 | 4567 | 5678 | 6789 | 7890 | 8901 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| advantage | -0.0324 | +0.0052 | +0.0256 | -0.0047 | -0.0385 | -0.0119 | +0.0173 | -0.0258 |

A clean null, straddling zero, exactly where C28 put it. It neither supports nor
undermines the crossing, and it is the one place in C30 where a *losing* C28 cell was
re-run across probe seeds — it did not become a winner. That is one data point against the
worry raised in C30.9 limitation 3, and one data point is not an answer to it.
## C30.8 What C30 changes elsewhere

Conflicts for the owner to merge. `reports/pilot_report.md` §22 and §23 are the merge
point; C30 does not edit §§1–21.

1. **§22.2's table must carry eight-seed numbers, not one-seed numbers.** Five of its eight
   rows replicate, one is unresolved, one reverses, one is cited from C29. The row that
   most needs changing is **C2 k = 2**: `+0.0007` becomes
   **-0.0341** with an interval excluding zero on the
   negative side. It should be struck from the crossing, not annotated.
2. **§22.2's caveat sentence — *"These cells were run at one probe-training seed"* — is
   discharged** and should be replaced by the eight-seed result, with C30 cited.
3. **The deployed cell's margin goes up, not down.** §22.2 and `ABSTRACT.md` v5 quote
   `+0.0846` for A1 k=2; the eight-seed mean is **+0.1044**.
   Any statement of the crossing's size should use the eight-seed figure.
4. **§19's validity story gains a probe-seed axis.** §19 established that validity collapses
   above λ ≈ 2. C30.4.5 shows that *at* λ = 2 the collapse is probe-seed-dependent: one seed
   in eight returns 17% unparseable molecules on an arm the other seven keep above 90%.
   §19.1's validity figures are single-seed and should say so.
5. **C28's limitation 2 — *"Three generation seeds, one head seed... No C28 arm is
   replicated across head-training seeds"* — is discharged for the winning cells** and
   remains open for the 22 losing ones, which C30 did not re-run and does not claim about.
6. **Nothing here touches C17, C18, C24, C26's frontier or C27's oracle result.** C30
   re-runs guidance against a fixed comparator; it re-measures no probe and no baseline.

## C30.9 Limitations

1. **The pre-registered verdict is UNINTERPRETABLE.** Everything positive in this section
   rests on a post-hoc repair (S1) that is labelled as one. A reader who declines the
   repair is entitled to, and gets no result from C30.
2. **Two properties, not three.** `qed` has no winning cell and was not run, so the
   crossing is replicated on `hbd_count` (5 cells) and `aromatic_rings` (2 cells) only.
   Six of the seven re-run cells are `hbd_count`.
3. **The losing cells were not re-run.** C30 asks whether the winners survive, not whether
   a loser becomes a winner at some other probe seed. The second question is live and
   unanswered, and it runs in C30's favour if anything, which is a reason to distrust the
   asymmetry rather than enjoy it.
4. **Eight probe seeds is eight.** The t interval on 7 df is honest but not tight, and the
   sd of an sd at n = 8 is still large.
5. **k and λ are still not crossed**, and C30 inherits C28's grid unchanged.
6. **One generator, one domain.** Unchanged from every other section.
7. **The comparator is C26's measured curve, interpolated.** Linear interpolation on a
   concave curve *underestimates* best-of-N, biasing every advantage upward — the same
   conservative-in-the-wrong-direction caveat C26 §7.3 records, inherited here.

## C30.10 Reproduce

```bash
# C30.0  the pre-registration must already be on disk and must not be edited.
sha256sum outputs/c30_prereg/C30.0_preregistration.md   # must match prereg_lock.json

# C30.1  gate G1 alone -- no GPU needed, and it is the gate that makes G2 interpretable.
.venv/bin/python scripts/24_c30_crossing_head_seeds.py --gates

# C30.2  the 64 cells. Head seed 1234 runs first so G2 is answered before the other
#        seven seeds are spent; a G2 failure stops the run. ~65 min on an RTX 4090.
#        Idempotent per cell: a kill costs at most one cell.
setsid nohup .venv/bin/python scripts/24_c30_crossing_head_seeds.py --all \
    > outputs/c30_run.log 2>&1 &

# C30.3  score the pre-registration. Reads only.
.venv/bin/python scripts/24_summarise_c30.py

.venv/bin/python -m pytest tests/test_crossing_head_seeds.py -q -p no:cacheprovider
```

**Artefact sizes.** `outputs/c30_hs*/` is 64 directories totalling **43 MB** — ~684 kB each,
because a cell is 3 x 512 = 1,536 molecules rather than one of the larger phase-2 runs. Only
`k_cell_metrics.json` is needed to re-derive any number here.
The `.npy`/`.pt` exclusions added to `.gitignore` on 2026-08-03 do not apply — C30 writes
neither. `outputs/c30_summary/c30_metrics.json` must stay: every number above is read from
it by `tests/test_crossing_head_seeds.py`.

**Pins.** Recorded by `write_run_context` beside every cell: the frozen GP-MoLFormer-Uniq
revision, torch, transformers, numpy, rdkit, float32, device `cuda`, generator in eval mode
and never modified. Cost is `processed_tokens_actual`; never wall-clock.
