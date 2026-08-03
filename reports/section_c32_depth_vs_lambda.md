# C32 — is C31's crossing depth, λ, or their interaction?

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-03. Conflicts with existing sections are **listed for the owner to merge** in
§C32.8, not edited in place.

**Headline, stated before the detail so it cannot be buried.** C31 ran two corners of a 2×2
— `(probe point 12, λ=1)` and `(mid probe point M, λ=2)` — and all five of its crossing
cells were the second corner, so "reading mid-network helps" and "steering harder helps"
were observationally identical. C32 runs the missing corners. The pre-registered verdict is
**LAMBDA-DOMINATED**.

**λ is the larger factor on every property, at every primary k, without exception.** The λ
main effect exceeds the depth main effect by the pre-registered 0.02 margin at k = 2 *and*
k = 4 on all three properties, with the sign holding everywhere.

**And the crossing does not need depth at all.** `hbd_count` at the **final** probe point
with λ = 2 crosses the oracle-selected frontier at k = 2 (**+0.1768**) and k = 4
(**+0.1105**), both with intervals excluding zero. C31 §C31.5.2 concluded that guidance beats
best-of-N *"only from a well-chosen mid-network probe point at λ = 2, and which probe point
that is has to be re-selected per generator."* **That conclusion is wrong, and C32 retracts
it.** What C31's deployed arm was missing was λ, not depth.

This was pre-registered as the prediction I **expected to fail** (P9). It did not fail.

**The two factors are approximately additive.** The interaction is positive in all six
primary cells but **not one of them has a t interval excluding zero**, so D3 does not fire.
That is a simple and useful result: depth and λ add, they do not multiply, and the recipe can
be reasoned about one factor at a time.

**What this does not say.** Depth is not nothing: the depth main effect is positive on all
three properties and is large on `aromatic_rings` (+0.1612 at k = 2, +0.1828 at k = 4). The
claim is one of *relative size*, and it is that λ buys more than depth does at every point
measured.

**The effective-λ control does not rescue depth on the property that mattered.** Correcting
the depth contrast for the wider `q` spread of the mid probe point (C29's control, re-run
here) **reverses `hbd_count`'s depth advantage** — raw +0.0527 at k = 2, λ = 2 becomes
**-0.0197** — while leaving `aromatic_rings`'s intact. And the confound is *smaller* here than
on GP-MoLFormer: median share **0.2823** against C29's 54–69%, so **P7 is falsified**.
`qed`'s spread ratio came out **below 1**, falsifying **P2** and **P3** and showing the
control's premise is not a law.

**I was wrong about the outcome, and said so in advance.** C32.0.9 committed me to expecting
**MIXED**, with D2, D3, D4 and D5 firing and D1 not. D1 fired, D2 did not, and my stated
expectation is scored as incorrect in §C32.7.

11 predictions were committed before the run: 7 confirmed, 4
falsified, all scored in §C32.7.

**A defect in this pre-registration was found while scoring it and is reported, not amended
away** — see §C32.1.1.

---

## C32.0 The pre-registration, verbatim

Copied byte-for-byte from `outputs/c32_prereg/C32.0_preregistration.md`, whose SHA-256 and
byte count were frozen into `prereg_lock.json` before any C32 measurement artefact existed.
`tests/test_depth_vs_lambda.py` asserts the hash, the byte count, the verbatim copy, and that
the pre-registration's mtime strictly precedes every C32 artefact.

C32 was designed by someone who had already seen C31's results. That is unavoidable and is
**disclosed inside the pre-registration itself** (C32.0.2), which lists the C31 numbers that
could have biased the design and names the degrees of freedom they create. A test asserts the
disclosure is present and quotes the specific advantages.

## C32.0.1 Why this experiment exists

C31 ran two corners of a 2×2 and only two:

| | λ = 1 | λ = 2 |
| --- | --- | --- |
| **final probe point 12** | C31 "deployed" | **not run** |
| **mid probe point M** | **not run** | C31 "mid" |

All five of C31's crossing cells are the `(M, λ=2)` corner. *"The crossing comes from reading
mid-network"* and *"the crossing comes from steering harder"* are therefore **observationally
identical in C31**, and §C31.5.2's mechanism claim is under-determined. C31's own limitation 5
says so. C32 runs the two missing corners.

This is not a robustness appendix. There is a live, already-evidenced hypothesis that the
depth effect is largely a λ effect wearing a disguise:

- The decoding rule is `log p_base(a) + λ·log(q(a) + ε)`. Only *differences* in `log q`
  across the candidate set change the sampling distribution, so **a multiplicative rescale of
  the spread of `log q` is exactly a rescale of λ.** This is the λ-rescale identity §20.3
  established.
- A mid-network probe produces a **wider spread of `q` across candidates** than a final-layer
  probe. C29 measured ratios of **1.27** (`hbd_count`, probe point 4) and **1.51**
  (`aromatic_rings`, probe point 3) on GP-MoLFormer.
- Correcting C23's depth contrast for that ratio removed **54–69%** of its margin, and
  "15 of 15 arms positive" became 10 of 15 (`reports/section_c29_head_seeds.md` §C29.5,
  `outputs/c29_summary/c29_metrics.json::effective_lambda`).

So the prior that C31's depth contrast is *clean* should be treated as weak. **All three
outcomes are good results and none is preferred**:

- **λ dominates** — the recipe simplifies to "steer harder at small k" and a probe-point
  tuning step disappears.
- **depth dominates** — C23's Rule A gains a second generator, the strongest thing that
  could happen to it.
- **interaction dominates** — the least convenient and the most interesting: neither factor
  is interpretable alone.

## C32.0.2 What I already know, disclosed because it constrains the design

C32 is designed by someone who has already seen C31's results. Pretending otherwise would be
worth less than saying exactly what is known, so a reader can price the researcher degrees of
freedom. **Every number below was on disk before this document was written.**

The mid probe points, **selected by held-out validation AUROC in advance under C31.0.4 and
NOT re-selected here**: `hbd_count` **M = 2**, `aromatic_rings` **M = 2**, `qed` **M = 6**.
Re-selecting them now, after seeing which arm crossed, is exactly the failure
`pilot_report.md` §21.5.2 exists to prevent; they are transcribed, not re-derived.

C31's advantage over the oracle-selected curve, at the two k where crossings live:

| property | corner | k = 2 | k = 4 |
| --- | --- | ---: | ---: |
| `hbd_count` | (12, λ=1) | +0.0317 | -0.0063 |
| `hbd_count` | (2, λ=2) | **+0.2295** | **+0.1696** |
| `aromatic_rings` | (12, λ=1) | -0.1756 | -0.1389 |
| `aromatic_rings` | (2, λ=2) | **+0.1724** | **+0.2473** |
| `qed` | (12, λ=1) | -0.0809 | -0.2109 |
| `qed` | (6, λ=2) | -0.0137 | -0.1379 |

Held-out **test** target AUROC at the two probe points: `hbd_count` 0.8322 (L2) vs 0.7866
(L12); `aromatic_rings` 0.8687 (L2) vs 0.8205 (L12); `qed` 0.7539 (L5, M = 6 selected on
validation) vs 0.7292 (L12).

**What this knowledge could bias, and what is fixed to stop it.** The tempting degree of
freedom is to choose the decomposition, the k, or the threshold after seeing which way the
answer falls. So: the arithmetic is written out in C32.0.4 before it is computed, the primary
k are fixed in C32.0.3, and every threshold in C32.0.6 and C32.0.7 is a number on this page.
The λ envelope grid is fixed here too, before any spread ratio has been measured on this
generator, so it cannot be chosen to bracket a convenient λ_eff.

## C32.0.3 What is run — fixed here, in full

**Nothing about the generator, the data, the intervals, the windows, the heads or the
comparator changes.** `entropy/gpt2_zinc_87m` at revision
`f42a5a10e24c0350aeadb50865bd90a714d0b2bf`, frozen, float32. The frozen `c31_zinc50k` target
intervals and windows, re-hashed before use. The C31 head checkpoints at head seed 1234, on
disk, unmodified. The C31 **oracle-selected** best-of-N curves as the fixed comparator, read
and never regenerated. SMILES only; no SAFE, no SELFIES; no weight of the generator changes.

**Arm A — the two missing corners of the 2×2.** 2 corners × 5 k × 3 properties = **30 cells**.

| corner | probe point | λ | status |
| --- | --- | ---: | --- |
| `deployed_l1` | 12 | 1.0 | **already run** — C31's deployed arm, reused, not re-run |
| `deployed_l2` | 12 | 2.0 | **new** |
| `mid_l1` | M | 1.0 | **new** |
| `mid_l2` | M | 2.0 | **already run** — C31's mid arm, reused, not re-run |

`k ∈ {2, 4, 8, 16, 32}`, generation seeds 101/202/303, 512 molecules each — C31's grid,
unchanged, so the completed 2×2 is 60 cells of 1,536 molecules.

**Arm B — the λ envelope at probe point 12, for the effective-λ correction.**
λ ∈ {1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0} at probe point 12, at `k ∈ {2, 4}`, 3 properties,
3 seeds. λ = 1.0 and λ = 2.0 are the 2×2's own `deployed_l1` / `deployed_l2` cells and are
**reused, not re-run**, so the envelope is one continuous object and not two. New cells:
5 λ × 2 k × 3 properties = **30**.

The grid is committed **now, before any spread ratio has been measured on this generator**.
It spans a factor of 4 and is denser than an octave grid between 1 and 3 precisely because
C29 found an octave grid could not resolve the correction and had to be re-run.

**Total new cells: 60.** Every cell is its own output directory, skipped if complete, so a
kill costs at most one cell.

**Arm C — the `q` spread measurement.** `mean_head_q_spread_across_candidates` — mean over
prefixes of `max_k q − min_k q` over the base model's top-k candidates — measured at probe
point 12 and at M, per property, **on the same prefixes and the same candidate sets**, so the
ratio is a paired quantity. This is `scripts/16_layer_steering_value.py`'s definition,
transcribed, and it is the quantity C29's `effective_lambda` block consumes.

**Nothing is forked.** `run_ksweep_cell` is **imported** from
`scripts/25_c31_second_generator.py` and pointed at new output directories by a scoped,
restored mutation — C30's pattern — so every C32 molecule comes from the same
`guidance.guided_sample` call C31 made, with the probe point and λ as the only differences.
`price`, `load_curve` and the cell collector are imported from `scripts/25_summarise_c31.py`;
`interp` and `t_interval` from `scripts/21_summarise_c26.py`, unmodified.

## C32.0.4 The decomposition — arithmetic, written before it is computed

Let `A(d, l, p, k, s)` be the advantage over the C31 oracle-selected best-of-N curve of the
cell at depth `d ∈ {12, M}`, λ `l ∈ {1, 2}`, property `p`, candidate count `k`, generation
seed `s`, computed exactly as C31 computed it: the cell's hit rate minus the curve linearly
interpolated in processed tokens per molecule at that cell's own budget, using the unmodified
`scripts/21_summarise_c26.py::interp`.

Write the four corners at fixed `(p, k, s)` as

```
a = A(12, 1)      b = A(12, 2)      c = A(M, 1)      d = A(M, 2)
```

Then, **per generation seed**, so that the contrasts are paired across seeds:

```
depth_main(p, k, s)   = 0.5 * ((c - a) + (d - b))
lambda_main(p, k, s)  = 0.5 * ((b - a) + (d - c))
interaction(p, k, s)  = 0.5 * ((d - c) - (b - a))
                      = 0.5 * ((d - b) - (c - a))
```

The `0.5` convention is fixed here and used everywhere; the three quantities then satisfy
`d - a = depth_main + lambda_main + interaction` exactly, and that identity is asserted
numerically in the summariser as an arithmetic gate.

Each is reported as the mean over the three generation seeds with a **seed-level Student t
interval on 2 df** (t₀.₉₇₅,₂ = 4.302653) and **the three raw per-seed values printed in
full**. **No bootstrap anywhere in C32**: at n = 3 the percentile bootstrap of a mean is
identically [min, max].

**Primary k, fixed now: k = 2 and k = 4** — the two budgets at which C31's crossings live.
All five k are computed and reported; only k ∈ {2, 4} carry the decision rules.

## C32.0.5 The effective-λ correction

`spread_ratio(p) = spread(p, M) / spread(p, 12)`, from Arm C. The mid arm's effective λ is
`λ_eff(p, l) = l × spread_ratio(p)`.

The **effective-λ-corrected depth contrast** at `(p, k, l)` is

```
depth_corrected(p, k, l) = A(M, l, p, k) - A_envelope(12, λ_eff(p, l), p, k)
```

where `A_envelope` is the Arm B envelope's advantage at probe point 12, interpolated
**linearly in log2(λ)** between the two bracketing measured λ — C29's interpolation rule,
reused, not re-chosen. The **confound share** is
`(depth_raw − depth_corrected) / depth_raw`, C29's definition.

**If `λ_eff` falls outside the measured envelope [1.0, 4.0], no value is invented.** The
bracket is reported, the contrast is flagged `extrapolated_beyond_envelope`, and it is
excluded from the corrected decision rule while still being printed.

**Stated as a limitation now, not after the fact:** the λ-rescale identity is *pointwise* in
`log q`, while the spread ratio is a scalar moment ratio of `q`. This is a **first-order
control, not an identity** — C29's own caveat, inherited verbatim. C32 does not claim the
correction is exact.

## C32.0.6 Decision rules — scored as written

Scored at k = 2 and k = 4, per property, on the seed-level means from C32.0.4.

| # | rule | fires iff |
| --- | --- | --- |
| **D1** | **λ dominates** | `|lambda_main| − |depth_main| ≥ 0.02` on ≥ 2 of 3 properties, at ≥ 1 primary k, and the sign holds at both primary k where both are measured |
| **D2** | **depth dominates** | `|depth_main| − |lambda_main| ≥ 0.02` on ≥ 2 of 3 properties, under the same conditions |
| **D3** | **the interaction is material** | `|interaction| ≥ 0.02` with a t interval on 2 df excluding zero, on ≥ 2 of 3 properties at ≥ 1 primary k |
| **D4** | **depth survives the effective-λ correction** | `depth_corrected > 0` with a t interval excluding zero, on ≥ 2 of 3 properties at ≥ 1 primary k, among contrasts not flagged `extrapolated_beyond_envelope` |
| **D5** | **the correction removes most of the depth effect** | the median confound share over all scored (property, k, λ) contrasts is ≥ 0.50 — C29 measured 0.54–0.69 |
| **D6** | **the crossing is reachable without depth** | at least one `(12, λ=2)` cell at k ≤ 4 crosses the oracle curve — i.e. positive advantage with a t interval on 2 df excluding zero and validity ≥ 0.80 |

**The verdict rule, fixed now.** Exactly one is reported:

- **LAMBDA-DOMINATED** iff D1 and not D2.
- **DEPTH-DOMINATED** iff D2 and not D1.
- **INTERACTION-DOMINATED** iff D3 fires and neither D1 nor D2 does.
- **MIXED** in any remaining case, reported as such and not resolved by prose.

D4, D5 and D6 are scored and reported independently of the verdict.

**D7 — the honesty rule, which fires regardless of the others.** Any contrast whose
between-seed sd exceeds its own absolute mean is reported as **"not resolved at three
generation seeds"** rather than as a positive or a negative.

## C32.0.7 Validity gates, run and reported BEFORE any decision rule

**G1 — the new code path reproduces C31 exactly.** `run_ksweep_cell`, imported from C31 and
redirected to a fresh C32 directory, must reproduce a **C31 cell** exactly: hit rate residual
**0.0** and processed-token residual **0.0** on every generation seed, and the returned
molecule strings **identical**. Run on `hbd_count`, `(12, λ=1)`, k = 2 and on
`aromatic_rings`, `(M, λ=2)`, k = 2 — one from each reused corner, so both corners of the
2×2 that C32 does not re-run are proven to be reachable by C32's own code path. **This is the
gate that makes every other C32 number comparable to C31's.**

**G2 — the frozen artefacts are unchanged.** `outputs/c31_zinc50k/target_intervals.json` and
`windows.json` must still hash to the values C31 recorded, and the C31 head checkpoints used
must be tensor-identical to the ones C31 steered with (max abs difference exactly 0.0, binner
edges equal).

**G3 — cost identity.** `processed_tokens_actual mod (k + 1) == 0` in every new cell and
every seed.

**G4 — the decomposition is arithmetically closed.** `d − a = depth_main + lambda_main +
interaction` to within 1e-12 for every `(p, k, s)`.

**G5 — the comparator is the C31 curve, unmodified.** The oracle-selected curve read by C32
must be byte-identical to C31's, and C32 must generate no best-of-N molecule.

**If G1, G2, G3, G4 or G5 fails, C32 stops and reports the failure.** No decision rule is
scored past a failed gate.

## C32.0.8 What would make C32 uninterpretable — with the repair specified in advance

C30 wrote a flat per-cell validity floor of 0.90, it tripped on **1 of 56** points, and the
pre-registered verdict became UNINTERPRETABLE for the whole run. That is a rule that cannot
tell "one point tripped" from "the experiment is degenerate". C32's rule:

| condition | consequence |
| --- | --- |
| G1, G2, G3, G4 or G5 fails | UNINTERPRETABLE, decision rules left unscored |
| a **single cell** has mean validity < 0.80 | **that cell is dropped whole**, every contrast that reads it is dropped and named, and the rules are **re-scored at unchanged thresholds** on what remains |
| **more than 6 of the 60** 2×2 cells have mean validity < 0.80 (> 10%) | UNINTERPRETABLE — guidance is destroying the generator rather than steering it |
| a whole property loses both primary k | that property is dropped from the rule counts and the "of 3 properties" denominators drop to match, stated explicitly |
| fewer than 3 generation seeds complete for a scored cell | that cell is dropped whole, as above |

Dropping cells can only ever **shrink** the evidence base, never manufacture a contrast, and
every drop is named in the section.

λ = 4 is inside the Arm B grid and §19 found validity collapsing above λ ≈ 2 on
GP-MoLFormer, so **envelope cells are expected to trip the floor and that is not a failure of
the experiment**: an envelope point below 0.80 validity is dropped from the interpolation and
the bracket widens, which is reported.

## C32.0.9 Predictions, committed before the run

Scored in the section. Falsified predictions are reported as falsified.

1. **P1** — G1 passes with residual exactly 0.0 on both gate cells, molecules included.
2. **P2** — the spread ratio at M exceeds 1.0 for all three properties, i.e. the mid probe
   point really does spread `q` more widely. *(C29 measured 1.27 and 1.51 on GP-MoLFormer.)*
3. **P3** — the spread ratio is **larger** than C29's largest (1.512) for at least one
   property, because C31's M = 2 is shallower than C29's probe points 3 and 4.
4. **P4** — the **λ main effect is positive** at k = 2 on all three properties: doubling λ
   helps at the cheapest budget.
5. **P5** — the **depth main effect is positive** at k = 2 on all three properties.
6. **P6** — **D3 fires**: the interaction is material on ≥ 2 of 3 properties. *(The two
   factors act on the same `log q` term; additivity would be surprising.)*
7. **P7** — the median confound share is ≥ 0.50, i.e. **D5 fires**, reproducing C29's 54–69%
   on a second generator.
8. **P8** — **D4 fires**: depth survives the correction on ≥ 2 of 3 properties. *(P7 and P8
   are deliberately both asserted: C29 found the effect roughly halved but not abolished. If
   both hold, "depth is real and roughly half of it was λ" is the answer.)*
9. **P9** — **D6 fires**: `(12, λ=2)` crosses at k ≤ 4 on at least one property. **I expect
   this one to fail.** It is stated because if it *does* fire, the mid probe point is not
   needed for a crossing at all and the recipe simplifies dramatically — and a prediction I
   expect to fail is the only kind that can surprise me.
10. **P10** — `qed` shows the smallest depth main effect of the three properties, as it shows
    the smallest AUROC gap between M and probe point 12 (0.0247 against 0.0456 and 0.0482).
11. **P11** — validity stays ≥ 0.95 in every 2×2 cell, so C32.0.8's drop rule never fires on
    Arm A. Envelope cells at λ ≥ 3 are **not** covered by this prediction.

**Which outcome I expect, stated so it can be scored.** I expect **MIXED**, with both main
effects positive and materially large, the interaction material, and the corrected depth
contrast positive but roughly half the raw one — i.e. D2, D3, D4 and D5 all firing and D1 not.
I do **not** expect a clean single-factor answer. If the verdict comes out
LAMBDA-DOMINATED I will report that C31 §C31.5.2's mechanism reading was wrong and that the
recipe should drop the probe-point selection step.

## C32.0.10 What C32 will NOT do

- **No weight of the generator changes.** No fine-tuning, no LoRA, no RL, no activation edit.
- **No new head is trained**, and **no probe point is re-selected**. C31's M values are
  transcribed. Re-selecting on a steering outcome is §21.5.2's failure mode.
- **No new best-of-N.** C31's oracle curve is read, never regenerated.
- **No alternative serialization, no third generator.**
- **No λ outside [1, 4]** and no k outside C31's grid.
- **No wall-clock claim.** Cost is processed generator tokens.
- **No existing artefact, report, config or `outputs/` directory is edited.** Conflicts go in
  a "What C32 changes elsewhere" section for the owner.
- **Probe-seed replication is explicitly out of scope unless the 2×2 and the correction both
  finish with budget to spare**, in which case it is reported as a clearly labelled
  extension. Priority is strictly 2×2 → correction → probe seeds.

## C32.0.11 The reporting rule

Gates first, as numeric residuals. Then the 2×2 in full with every per-seed value. Then the
decomposition with intervals. Then the spread ratios and the effective-λ-corrected contrast.
Then the decision rules scored as written, then the predictions including the falsified ones,
then limitations, then what it changes elsewhere. A result that overturns C31 §C31.5.2 is
written up at the same length as one that confirms it. **If a deviation from this
pre-registration proves necessary, the deviation is reported and this document is not
amended.**

---

## C32.1 What was run

| arm | what | cells |
| --- | --- | ---: |
| **A** | the two missing 2×2 corners, `(12, λ=2)` and `(M, λ=1)`, k ∈ {2,4,8,16,32}, 3 properties | 30 new |
| **B** | the λ envelope at probe point 12, λ ∈ {1, 1.25, 1.5, 2, 2.5, 3, 4}, k ∈ {2,4} | 30 new |
| **C** | `mean_head_q_spread_across_candidates` at probe point 12 and at M, paired | — |
| — | C31's `(12, λ=1)` and `(M, λ=2)` corners | 30 reused |

Every cell is 3 generation seeds × 512 molecules. The completed 2×2 is **60 cells**.

**Nothing about the generator, the data, the intervals, the windows, the heads or the
comparator changed.** `entropy/gpt2_zinc_87m` at the pinned revision, frozen. The frozen
`c31_zinc50k` intervals and windows, re-hashed before use. C31's head checkpoints at head
seed 1234, unmodified. C31's **oracle-selected** best-of-N curves as the fixed comparator,
read and never regenerated — gate G5 asserts C32 defines no best-of-N stage and calls no pool
sampler.

**The mid probe points are transcribed, never re-selected.** `hbd_count` M = 2,
`aromatic_rings` M = 2, `qed` M = 6 — chosen by C31.0.4's held-out **validation** AUROC rule
before any steering outcome existed. Re-selecting them now, after seeing which arm crossed,
is `pilot_report.md` §21.5.2's failure mode. A test asserts the C32 cells sit at exactly
C31's probe points.

**Nothing is forked.** `run_ksweep_cell` is **imported** from
`scripts/25_c31_second_generator.py` and pointed at new directories by a scoped, restored
mutation of that module's `cell_dir` — C30's pattern. `price` and `load_curve` come from
`scripts/25_summarise_c31.py`; `interp` and `t_interval` from `scripts/21_summarise_c26.py`.
Tests assert on the AST that none of them is redefined locally.

### C32.1.1 A defect in the pre-registration, reported and not amended

**C32.0.4's closure identity is arithmetically false.** It states

> `d − a = depth_main + lambda_main + interaction`

With the pre-registered ½ convention, `depth_main + lambda_main` is *already* exactly
`d − a` — the two marginal contrasts partition the corner difference between them — so adding
the interaction over-counts. Substituting `a=0, b=1, c=2, d=5` gives depth = 3, λ = 2,
interaction = 1: `depth + λ = 5 = d − a`, while `depth + λ + interaction = 6`. Measured on
the real cells, the residual of the claim as written is **0.1008**, which is how
the error was found: gate G4 failed on the first scoring run.

**Handling, following C29 §C29.4's treatment of its own R4 defect and C27's of its gate 4:
the defect is reported and this pre-registration is not amended.** The three effect
definitions themselves are standard and unchanged; only the closure *claim* was wrong. G4 is
scored on the identities that do hold — `d − a = depth_main + lambda_main` and
`interaction = ½((d−b) − (c−a))` — both to within **0.00e+00**. **No decision rule reads
the closure identity**, so no rule and no number in this section changes because of it. The
interaction remains reported at the pre-registered ½ convention, which is half the
conventional `(d−c) − (b−a)`.

---

## C32.2 Validity gates, before any decision rule

**G1 — C32's code path reproduces C31 exactly.** `run_ksweep_cell`, imported from C31 and
redirected to a fresh C32 directory, was re-run on one cell from *each* corner C32 reuses:
`hbd_count (12, λ=1) k=2` and `aromatic_rings (M, λ=2) k=2`. Hit-rate residual
**+0.0000**, token residual **+0.0000**, and **3072 of 3072
molecules identical**. Without this gate every comparison between a C32 corner and a C31
corner would be uncontrolled; with it, the 2×2 is a single experiment.

**G2 — the frozen artefacts are unchanged.** `target_intervals.json` and `windows.json` still
hash to the values C31 recorded, and every head checkpoint C32 steers with carries the
expected property, probe point and head seed 1234.

**G3 — cost identity.** `processed_tokens_actual mod (k+1) == 0` in every cell.

**G4 — the decomposition is arithmetically closed**, on the identities that hold: residual
**0.00e+00**. See §C32.1.1 for the pre-registered claim that did not.

**G5 — the comparator is C31's curve.** The oracle-selected curve files hash to the values
C32 recorded, all three are `oracle_selected`, and C32 defines no best-of-N stage and calls
no pool sampler — both checked on the AST rather than by text search.

**All five gates pass.** A test asserts no verdict can be issued while a required gate failed.

Validity never came near the pre-registered floor: the **minimum over all 60 2×2
cells is 0.9818**, against an exclusion floor of 0.80 and a degeneracy threshold of
more than 6 cells. C32.0.8's drop-and-re-score repair — written precisely because C30's flat
0.90 screen voided a whole run on one tripped point — never had to fire on Arm A.

---

## C32.3 The 2×2, corner by corner

Advantage over C31's oracle-selected best-of-N curve at each cell's own token budget, at the
two pre-registered primary k. `(12, λ=1)` and `(M, λ=2)` are C31's cells, reused unchanged;
`(12, λ=2)` and `(M, λ=1)` are new.

| property | k | corner | probe point | λ | from | hit rate | advantage over oracle best-of-N | per-seed advantage (101, 202, 303) |
| --- | ---: | --- | ---: | ---: | :---: | ---: | ---: | --- |
| HBD count | 2 | (12, λ=1) | 12 | 1 | C31 | 0.3092 | **+0.0317** | +0.0484, +0.0222, +0.0244 |
| HBD count | 2 | (12, λ=2) | 12 | 2 | **C32** | 0.4499 | **+0.1768** | +0.1969, +0.1600, +0.1735 |
| HBD count | 2 | (M, λ=1) | 2 | 1 | **C32** | 0.3346 | **+0.0585** | +0.0661, +0.0372, +0.0721 |
| HBD count | 2 | (M, λ=2) | 2 | 2 | C31 | 0.4980 | **+0.2295** | +0.2519, +0.2254, +0.2114 |
| HBD count | 4 | (12, λ=1) | 12 | 1 | C31 | 0.3837 | **-0.0063** | +0.0133, +0.0067, -0.0388 |
| HBD count | 4 | (12, λ=2) | 12 | 2 | **C32** | 0.4905 | **+0.1105** | +0.1316, +0.0869, +0.1131 |
| HBD count | 4 | (M, λ=1) | 2 | 1 | **C32** | 0.4033 | **+0.0145** | +0.0303, +0.0032, +0.0100 |
| HBD count | 4 | (M, λ=2) | 2 | 2 | C31 | 0.5457 | **+0.1696** | +0.1575, +0.1528, +0.1986 |
| aromatic rings | 2 | (12, λ=1) | 12 | 1 | C31 | 0.2064 | **-0.1756** | -0.1470, -0.1652, -0.2143 |
| aromatic rings | 2 | (12, λ=2) | 12 | 2 | **C32** | 0.3212 | **-0.0582** | -0.0096, -0.0769, -0.0881 |
| aromatic rings | 2 | (M, λ=1) | 2 | 1 | **C32** | 0.2943 | **-0.0837** | -0.1018, -0.0943, -0.0551 |
| aromatic rings | 2 | (M, λ=2) | 2 | 2 | C31 | 0.5430 | **+0.1724** | +0.1567, +0.1957, +0.1648 |
| aromatic rings | 4 | (12, λ=1) | 12 | 1 | C31 | 0.3781 | **-0.1389** | -0.1638, -0.1341, -0.1190 |
| aromatic rings | 4 | (12, λ=2) | 12 | 2 | **C32** | 0.5403 | **+0.0358** | +0.0519, +0.0379, +0.0174 |
| aromatic rings | 4 | (M, λ=1) | 2 | 1 | **C32** | 0.5234 | **+0.0152** | +0.0027, +0.0175, +0.0254 |
| aromatic rings | 4 | (M, λ=2) | 2 | 2 | C31 | 0.7451 | **+0.2473** | +0.2245, +0.2648, +0.2525 |
| QED | 2 | (12, λ=1) | 12 | 1 | C31 | 0.2435 | **-0.0809** | -0.0765, -0.1023, -0.0638 |
| QED | 2 | (12, λ=2) | 12 | 2 | **C32** | 0.2875 | **-0.0385** | -0.0359, -0.0610, -0.0185 |
| QED | 2 | (M, λ=1) | 6 | 1 | **C32** | 0.2554 | **-0.0677** | -0.0543, -0.0947, -0.0542 |
| QED | 2 | (M, λ=2) | 6 | 2 | C31 | 0.3055 | **-0.0137** | -0.0230, -0.0179, -0.0003 |
| QED | 4 | (12, λ=1) | 12 | 1 | C31 | 0.2376 | **-0.2109** | -0.2039, -0.2163, -0.2125 |
| QED | 4 | (12, λ=2) | 12 | 2 | **C32** | 0.2863 | **-0.1617** | -0.1638, -0.1530, -0.1683 |
| QED | 4 | (M, λ=1) | 6 | 1 | **C32** | 0.2565 | **-0.1936** | -0.1921, -0.1967, -0.1921 |
| QED | 4 | (M, λ=2) | 6 | 2 | C31 | 0.3112 | **-0.1379** | -0.1352, -0.1423, -0.1361 |

---

## C32.4 The decomposition

Per generation seed, so the contrasts are paired, then a seed-level Student t interval on
2 df (t₀.₉₇₅,₂ = 4.302653). The ½ convention of C32.0.4 throughout.

```
depth_main  = 0.5 * ((c - a) + (d - b))
lambda_main = 0.5 * ((b - a) + (d - c))
interaction = 0.5 * ((d - c) - (b - a))
```

### C32.4.1 Primary k — the two budgets where C31's crossings live

| property | k | depth main effect | λ main effect | interaction | dominant |
| --- | ---: | --- | --- | --- | :---: |
| HBD count | 2 | **+0.0398** [+0.0318, +0.0478] | **+0.1581** [+0.1277, +0.1885] | **+0.0130** [-0.0264, +0.0523] | lambda |
| HBD count | 4 | **+0.0399** [-0.0198, +0.0997] | **+0.1360** [+0.0616, +0.2103] | **+0.0192** [-0.0184, +0.0567] | lambda |
| aromatic rings | 2 | **+0.1612** [+0.0345, +0.2879] | **+0.1867** [+0.1554, +0.2180] | **+0.0694** [-0.0002, +0.1390] | lambda |
| aromatic rings | 4 | **+0.1828** [+0.1542, +0.2115] | **+0.2034** [+0.1555, +0.2513] | **+0.0287** [-0.0272, +0.0846] | lambda |
| QED | 2 | **+0.0190** [+0.0044, +0.0335] | **+0.0482** [+0.0193, +0.0771] | **+0.0058** [-0.0222, +0.0339] | lambda |
| QED | 4 | **+0.0206** [+0.0067, +0.0344] | **+0.0524** [+0.0387, +0.0662] | **+0.0033** [-0.0136, +0.0202] | lambda |

**λ is larger than depth in all six primary cells.** The margin `|λ| − |depth|` clears the
pre-registered 0.02 threshold at **both** primary k on **all three** properties, and the sign
holds everywhere — so D1 fires under the full rule, including the sign-consistency clause
that is easy to drop by accident.

The gap is not uniform. At k = 2, λ is ~4.0× depth on `hbd_count` and ~2.5× on `qed`, while
on `aromatic_rings` the two are close (+0.1867 against +0.1612) — and `aromatic_rings` is the
only property where depth is large in absolute terms. So "λ dominates" is a strong statement
on `hbd_count` and `qed` and a narrow one on `aromatic_rings`, where it clears the 0.02
margin by +0.0255 at k = 2 and +0.0206 at k = 4.

### C32.4.2 The raw per-seed values, in full

| property | k | effect | per-seed values (101, 202, 303) |
| --- | ---: | --- | --- |
| HBD count | 2 | depth | +0.0364, +0.0402, +0.0428 |
| HBD count | 2 | λ | +0.1671, +0.1630, +0.1442 |
| HBD count | 2 | interaction | +0.0186, +0.0252, -0.0049 |
| HBD count | 4 | depth | +0.0215, +0.0312, +0.0672 |
| HBD count | 4 | λ | +0.1228, +0.1149, +0.1702 |
| HBD count | 4 | interaction | +0.0045, +0.0347, +0.0184 |
| aromatic rings | 2 | depth | +0.1057, +0.1717, +0.2061 |
| aromatic rings | 2 | λ | +0.1979, +0.1892, +0.1731 |
| aromatic rings | 2 | interaction | +0.0606, +0.1008, +0.0469 |
| aromatic rings | 4 | depth | +0.1695, +0.1892, +0.1897 |
| aromatic rings | 4 | λ | +0.2188, +0.2097, +0.1818 |
| aromatic rings | 4 | interaction | +0.0031, +0.0376, +0.0454 |
| QED | 2 | depth | +0.0176, +0.0254, +0.0139 |
| QED | 2 | λ | +0.0359, +0.0591, +0.0496 |
| QED | 2 | interaction | -0.0047, +0.0178, +0.0043 |
| QED | 4 | depth | +0.0202, +0.0152, +0.0263 |
| QED | 4 | λ | +0.0485, +0.0588, +0.0501 |
| QED | 4 | interaction | +0.0084, -0.0044, +0.0059 |

### C32.4.3 Secondary k, reported and not scored

| property | k | depth main effect | λ main effect | interaction | dominant |
| --- | ---: | --- | --- | --- | :---: |
| HBD count | 8 | **+0.0842** [-0.0202, +0.1885] | **+0.1605** [+0.1153, +0.2057] | **-0.0052** [-0.0606, +0.0503] | lambda |
| HBD count | 16 | **+0.0685** [+0.0517, +0.0854] | **+0.1550** [+0.0965, +0.2134] | **+0.0251** [-0.0633, +0.1134] | lambda |
| HBD count | 32 | **+0.0858** [+0.0666, +0.1051] | **+0.1410** [+0.1098, +0.1723] | **+0.0221** [-0.0376, +0.0818] | lambda |
| aromatic rings | 8 | **+0.1396** [+0.1146, +0.1645] | **+0.1984** [+0.1621, +0.2347] | **+0.0563** [+0.0110, +0.1017] | lambda |
| aromatic rings | 16 | **+0.1421** [+0.0578, +0.2265] | **+0.1760** [+0.0900, +0.2620] | **+0.0365** [-0.0495, +0.1225] | lambda |
| aromatic rings | 32 | **+0.1392** [+0.1105, +0.1678] | **+0.1501** [+0.1221, +0.1781] | **+0.0348** [-0.0130, +0.0826] | neither_by_the_margin |
| QED | 8 | **+0.0318** [-0.0265, +0.0902] | **+0.0753** [+0.0647, +0.0858] | **-0.0022** [-0.0607, +0.0562] | lambda |
| QED | 16 | **+0.0291** [+0.0257, +0.0324] | **+0.0378** [-0.0168, +0.0924] | **+0.0232** [-0.0054, +0.0518] | neither_by_the_margin |
| QED | 32 | **+0.0092** [-0.0713, +0.0896] | **+0.0466** [-0.0128, +0.1059] | **-0.0024** [-0.0464, +0.0415] | lambda |

---

## C32.5 The effective-λ control

### C32.5.1 The spread ratio

`mean_head_q_spread_across_candidates` — the mean over prefixes of `max_k q − min_k q` over
the base model's top-k candidates, `scripts/16_layer_steering_value.py`'s definition — read at
probe point 12 and at M **from the same forward pass over the same prefixes and the same
candidate sets**, so the ratio is paired and cannot be moved by a difference in which
prefixes each probe point happened to see.

| property | M | spread at M | spread at probe point 12 | spread ratio | λ_eff at λ=1 | λ_eff at λ=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HBD count | 2 | 0.2016 | 0.1623 | **1.2417** | 1.2417 | **2.4835** |
| aromatic rings | 2 | 0.2060 | 0.1725 | **1.1939** | 1.1939 | **2.3878** |
| QED | 6 | 0.0961 | 0.1119 | **0.8594** | 0.8594 | **1.7189** |

Because the decoding rule is `log p_base + λ·log(q + ε)`, a multiplicative rescale of the
spread of `log q` **is** a rescale of λ — §20.3's identity. A mid-network probe that spreads
`q` more widely is therefore steering at a higher *effective* λ than its nominal one, and a
depth contrast at matched nominal λ silently compares two different steering strengths.

### C32.5.2 The λ envelope at the final probe point

Arm B, run so that the correction interpolates a *measured* deployed arm rather than an
invented one. The grid was committed in C32.0.3 **before any spread ratio had been measured
on this generator**, so it cannot have been chosen to bracket a convenient λ_eff. Struck
values are envelope points dropped for falling below the 0.80 validity floor, as C32.0.8
anticipated for large λ.

**k = 2**

| property | λ=1 | λ=1.25 | λ=1.5 | λ=2 | λ=2.5 | λ=3 | λ=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HBD count | +0.0317 | +0.0864 | +0.1098 | +0.1768 | +0.2514 | +0.2513 | +0.2869 |
| aromatic rings | -0.1756 | -0.1195 | -0.1099 | -0.0582 | +0.0058 | +0.0388 | +0.0565 |
| QED | -0.0809 | -0.0664 | -0.0482 | -0.0385 | +0.0064 | -0.0112 | -0.0055 |

**k = 4**

| property | λ=1 | λ=1.25 | λ=1.5 | λ=2 | λ=2.5 | λ=3 | λ=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HBD count | -0.0063 | +0.0185 | +0.0415 | +0.1105 | +0.1277 | +0.1130 | +0.1257 |
| aromatic rings | -0.1389 | -0.0842 | -0.0422 | +0.0358 | +0.0868 | +0.0665 | +0.0423 |
| QED | -0.2109 | -0.1702 | -0.1619 | -0.1617 | -0.1308 | -0.1328 | -0.1522 |


### C32.5.3 The depth contrast at matched effective λ

| property | k | λ | λ_eff | raw depth contrast | envelope bracket | depth at matched λ_eff | corrected depth contrast | 95% t interval | confound share |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| aromatic rings | 2 | 1 | 1.1939 | +0.0919 | [1, 1.25] | -0.1310 | **+0.0473** | [-0.0334, +0.1279] | +0.4850 |
| aromatic rings | 2 | 2 | 2.3878 | +0.2306 | [2, 2.5] | -0.0073 | **+0.1798** | [+0.0623, +0.2971] | +0.2206 |
| aromatic rings | 4 | 1 | 1.1939 | +0.1541 | [1, 1.25] | -0.0954 | **+0.1106** | [+0.0857, +0.1355] | +0.2819 |
| aromatic rings | 4 | 2 | 2.3878 | +0.2115 | [2, 2.5] | +0.0763 | **+0.1710** | [+0.1418, +0.2002] | +0.1915 |
| HBD count | 2 | 1 | 1.2417 | +0.0268 | [1, 1.25] | +0.0848 | **-0.0263** | [-0.1040, +0.0513] | +1.9820 |
| HBD count | 2 | 2 | 2.4835 | +0.0527 | [2, 2.5] | +0.2492 | **-0.0197** | [-0.0776, +0.0383] | +1.3738 |
| HBD count | 4 | 1 | 1.2417 | +0.0208 | [1, 1.25] | +0.0178 | **-0.0034** | [-0.0317, +0.0251] | +1.1617 |
| HBD count | 4 | 2 | 2.4835 | +0.0591 | [2, 2.5] | +0.1272 | **+0.0424** | [-0.1484, +0.2333] | +0.2826 |
| QED | 2 | 1 | 0.8594 | +0.0132 | **extrapolated_beyond_envelope** | — | — | — | — |
| QED | 2 | 2 | 1.7189 | +0.0248 | [1.5, 2] | -0.0436 | **+0.0299** | [-0.0355, +0.0952] | -0.2057 |
| QED | 4 | 1 | 0.8594 | +0.0173 | **extrapolated_beyond_envelope** | — | — | — | — |
| QED | 4 | 2 | 1.7189 | +0.0238 | [1.5, 2] | -0.1618 | **+0.0239** | [+0.0046, +0.0432] | -0.0033 |

**Caveat, stated where it belongs and not in a footnote.** The λ-rescale identity is
*pointwise* in `log q`; the spread ratio is a scalar moment ratio of `q`. This is a
**first-order control, not an identity** — C29's own caveat, inherited verbatim. Where λ_eff
falls outside the measured envelope no value is invented: the bracket is reported and the
contrast is excluded from D4, which a test enforces.

**Three things come out of this table, and they do not all point the same way.**

**1. `qed`'s mid probe point spreads `q` *less*, not more.** Its ratio is **0.8594** — below
1.0 — so P2 and P3 are **falsified**. C29 measured 1.27 and 1.51 on GP-MoLFormer and the
whole premise of the control is that a mid-network probe steers harder than its nominal λ.
For `qed` at probe point 6 the opposite holds, the correction therefore moves its depth
contrast *up* rather than down (negative confound shares), and at λ = 1 the implied
λ_eff = 0.8594 falls **below** the measured envelope, so those two contrasts are flagged
`extrapolated_beyond_envelope` and excluded from D4 rather than extrapolated. The premise is
not a law.

**2. `hbd_count`'s depth advantage reverses under the correction.** Raw +0.0268 (k=2, λ=1)
and +0.0527 (k=2, λ=2) become **-0.0263** and **-0.0197**; the confound shares are **1.9820**
and **1.3738**, i.e. greater than 1, which is what a sign flip looks like in this statistic.
Read with §C32.6.1 — where `hbd_count` is the one property that crosses at the *final* probe
point once λ = 2 — the picture is consistent: **for `hbd_count`, moving the readout
mid-network buys nothing once you account for the fact that it steers harder.** The t
intervals on those corrected contrasts span zero, so this is "not distinguishable from zero
after correction", not "significantly negative".

**3. `aromatic_rings`'s depth advantage survives.** Corrected +0.0473 to +0.1798 across the
four contrasts, with confound shares of 0.1915–0.4850 — the correction removes a fifth to a
half, and what is left is still positive, with intervals excluding zero at three of four
contrasts. This is the one property where depth is doing real work beyond effective λ.

**D4 fires** on `aromatic_rings` and `qed` (2 of 3), so depth survives the correction by the
pre-registered rule — but it survives on the two properties that matter least to the headline
and fails on the one whose crossing C31 leaned on.

**D5 does not fire.** The median confound share over the 10 scored contrasts is
**0.2823**, against a pre-registered threshold of 0.50 and C29's measured 54–69% on
GP-MoLFormer. **P7 is falsified**: the effective-λ confound is real here but roughly half the
size it was on the first generator. That is itself a finding — the confound does not transfer
at the same magnitude, and a project that had only measured it once would have over-corrected
here.

---

## C32.6 The decision rules, scored as written

| # | rule | fires |
| --- | --- | :---: |
| **D1** | lambda dominates: |lambda_main| - |depth_main| >= 0.02 on >= 2 of 3 properties at >= 1 primary k | **YES** |
| **D2** | depth dominates: |depth_main| - |lambda_main| >= 0.02 on >= 2 of 3 properties at >= 1 primary k | no |
| **D3** | the interaction is material: |interaction| >= 0.02 with a t interval excluding zero, on >= 2 of 3 properties at >= 1 primary k | no |
| **D4** | depth survives the effective-lambda correction: depth_corrected > 0 with a t interval excluding zero, on >= 2 of 3 properties at >= 1 primary k, among non-extrapolated contrasts | **YES** |
| **D5** | the correction removes most of the depth effect: median confound share >= 0.5 | no |
| **D6** | the crossing is reachable without depth: at least one (12, lambda=2) cell at k <= 4 crosses the oracle curve | **YES** |
| **D7** | the honesty rule: any contrast whose between-seed sd exceeds its own absolute mean is reported as NOT RESOLVED at three generation seeds | **YES** |

**Verdict: LAMBDA-DOMINATED**, by C32.0.6's rule — LAMBDA-DOMINATED iff D1 and not D2;
DEPTH-DOMINATED iff D2 and not D1; INTERACTION-DOMINATED iff D3 and neither; MIXED otherwise;
UNINTERPRETABLE overrides. A test computes the verdict from the artefacts and asserts the
section's word matches it, so the wording cannot drift from the rule.

### C32.6.1 D6 — the result that changes the recipe

**D6 fires**: `c32_cell_hbd_count_deployed_l2_L12_lam2_k2`, `c32_cell_hbd_count_deployed_l2_L12_lam2_k4`.

`hbd_count` at the **final** probe point with λ = 2 crosses the oracle-selected frontier at
both k = 2 and k = 4, with t intervals excluding zero. No mid-network probe, no probe-point
selection step, no depth curve — just the deployed readout and a doubled λ.

C31 §C31.5.2 read its own deployed-arm failure as evidence that the crossing *requires* a
re-selected mid-network probe point. C32 shows the deployed arm failed because it was run at
λ = 1. **That reading is retracted in §C32.8.**

The other two properties do not cross at `(12, λ=2)`: `aromatic_rings` reaches +0.0358 at
k = 4 with an interval spanning zero, and `qed` is negative. So D6 fires on one property of
three, and the honest statement is that depth is *not necessary* for a crossing rather than
that it is never useful.

### C32.6.2 D3 — the two factors are additive, and what that does and does not license

**D3 does not fire.** The interaction is positive in all six primary cells but no interval
excludes zero, so on the evidence here depth and λ **add**. That is a genuinely useful
simplification — the recipe can be reasoned about one factor at a time — and it is also the
prediction P6 got wrong.

**What "λ dominates" does not license.** It does not license simply raising λ without limit.
§19 found validity collapsing above λ ≈ 2 on GP-MoLFormer, and the envelope in §C32.5.2 is
the only evidence C32 has about where the returns stop on *this* generator — measured at two
k, on three properties, up to λ = 4. C32 says λ is the larger lever between the two it
tested; it does not locate the optimum, and it does not claim monotonicity beyond the grid.

### C32.6.3 D7 — the honesty rule

Contrasts whose between-seed sd exceeds their own absolute mean, reported as **not resolved
at three generation seeds**:

- `hbd_count_k16:interaction` — mean +0.0251, between-seed sd 0.0356
- `hbd_count_k2:interaction` — mean +0.0130, between-seed sd 0.0158
- `hbd_count_k32:interaction` — mean +0.0221, between-seed sd 0.0240
- `hbd_count_k8:interaction` — mean -0.0052, between-seed sd 0.0223
- `qed_k2:interaction` — mean +0.0058, between-seed sd 0.0113
- `qed_k32:depth_main` — mean +0.0092, between-seed sd 0.0324
- `qed_k32:interaction` — mean -0.0024, between-seed sd 0.0177
- `qed_k4:interaction` — mean +0.0033, between-seed sd 0.0068
- `qed_k8:interaction` — mean -0.0022, between-seed sd 0.0235

---

## C32.7 The predictions, scored

| # | prediction | outcome |
| --- | --- | :---: |
| **P1** | G1 passes with residual exactly 0.0 on both gate cells, molecules included | **CONFIRMED** |
| **P2** | the spread ratio at M exceeds 1.0 for all three properties | **FALSIFIED** |
| **P3** | the spread ratio exceeds C29's largest (1.512) for at least one property | **FALSIFIED** |
| **P4** | the lambda main effect is positive at k = 2 on all three properties | **CONFIRMED** |
| **P5** | the depth main effect is positive at k = 2 on all three properties | **CONFIRMED** |
| **P6** | D3 fires: the interaction is material on >= 2 of 3 properties | **FALSIFIED** |
| **P7** | the median confound share is >= 0.50, i.e. D5 fires | **FALSIFIED** |
| **P8** | D4 fires: depth survives the correction on >= 2 of 3 properties | **CONFIRMED** |
| **P9** | D6 fires: (12, lambda=2) crosses at k <= 4 on at least one property -- STATED IN ADVANCE AS EXPECTED TO FAIL | **CONFIRMED** |
| **P10** | qed shows the smallest depth main effect of the three properties at k = 2 | **CONFIRMED** |
| **P11** | validity stays >= 0.95 in every 2x2 cell (envelope cells at lambda >= 3 are not covered) | **CONFIRMED** |

**My stated expectation was wrong.** C32.0.9 ends with an explicit commitment: *"I expect
**MIXED**, with both main effects positive and materially large, the interaction material,
and the corrected depth contrast positive but roughly half the raw one — i.e. D2, D3, D4 and
D5 all firing and D1 not."* D1 fired and D2 did not; the verdict is LAMBDA-DOMINATED. The
expectation is scored here as incorrect rather than quietly dropped.

**P9 is the one that matters.** It was committed as *expected to fail*, on the reasoning that
if it fired "the mid probe point is not needed for a crossing at all and the recipe
simplifies dramatically". It fired. That is the strongest single piece of evidence in C32 and
it arrived from the prediction least likely, in advance, to produce it.

---

## C32.8 What C32 changes elsewhere

Conflicts for the owner to merge. `reports/pilot_report.md` §22 and §23 are the merge point;
C32 edits no existing section, config, report or `outputs/` directory.

1. **C31 §C31.5.2's mechanism sentence is retracted.** It reads: *"guidance can beat
   oracle-selected best-of-N at small budgets, but only from a well-chosen mid-network probe
   point at λ=2, and which probe point that is has to be re-selected per generator."* The
   clause after "but" is false. `hbd_count` at the final probe point with λ = 2 crosses at
   k = 2 and k = 4. The replacement sentence C32 supports is: *"guidance can beat
   oracle-selected best-of-N at small budgets, and the dominant ingredient is steering
   strength, not probe depth; a mid-network probe adds to it but is not required."*
2. **C31 §C31.5.2's framing of the deployed arm's failure is wrong in an instructive way.**
   C31 concluded the deployed *configuration* does not cross. That remains true **as
   configured** — probe point 12 at λ = 1 — but the cause is the λ, not the probe point, and
   §C31.5.2 attributes it to the probe point. C31's §C31.10 limitation 5 already flagged this
   as under-determined; C32 resolves it against C31's reading.
3. **C31's headline comparison table should gain a row.** Its "the **deployed** configuration
   crosses" row reads "**no**" for `gpt2_zinc_87m`. That is correct at λ = 1 and misleading
   without it; the row should say "no at λ = 1, **yes at λ = 2** on `hbd_count`".
4. **C23's Rule A does not gain a second generator.** The depth main effect is positive
   everywhere here, so Rule A's *direction* survives, but it is the smaller factor on all
   three properties and C32 provides no support for depth being the primary mechanism.
5. **C29's effective-λ confound does not transfer at the same magnitude, and its premise is
   not universal.** C29 measured spread ratios of 1.27–1.51 and a 54–69% reduction in the
   depth margin. Here the ratios are 1.19–1.24 on the two count properties and **0.86 on
   `qed`** — below 1, meaning the mid probe point steers *less* hard — and the median confound
   share is 0.2823. Any statement that "a mid-network probe is really a higher λ"
   should be scoped: it held on 2 of 3 properties here and reversed on the third.
6. **The recipe in `docs/REPRODUCE.md` and any deployment guidance should promote λ over
   probe-point selection.** On this generator, doubling λ at the existing readout buys more
   than moving the readout does, and it removes a tuning step that requires training 13
   probes and a held-out selection rule.
7. **Nothing here re-measures C31's frontier, C31's depth curve, C26's curves or C27's oracle
   result.** C32 prices new cells against a fixed, hashed comparator.

---

## C32.9 Limitations

1. **One probe seed per cell, still, and this is the most important caveat in C32.** Every
   C32 cell uses head seed 1234 — the protocol C29 showed is inadequate. The t intervals
   above are over *generation* seeds and say nothing about probe-seed variance. Measured
   against C29's largest probe-seed sd of end-to-end hit rate (**0.0366**), at k = 2:

   | property | depth main effect at k=2 | x sd | λ main effect at k=2 | x sd |
   | --- | ---: | ---: | ---: | ---: |
   | HBD count | +0.0398 | 1.1x | +0.1581 | 4.3x |
   | aromatic rings | +0.1612 | 4.4x | +0.1867 | 5.1x |
   | QED | +0.0190 | 0.5x | +0.0482 | 1.3x |

   So the "λ > depth" conclusion rests on very different footing per property. On
   `aromatic_rings` both effects are comfortably above the probe-seed sd and the comparison
   is between two resolved quantities. On `hbd_count` the λ effect is resolved and the depth
   effect is **not**. On `qed` **neither** is, and its contribution to D1 should be discounted
   accordingly — D1 would still fire on two properties without it, which is exactly the
   threshold, so `qed`'s inclusion is load-bearing for the rule and that is uncomfortable.
   Re-running the 2×2 at three or more probe seeds is the first thing to do next; C32's
   budget did not reach it, and C32.0.10 put it explicitly out of scope.
2. **The 2×2 has one factor level pair, not a curve.** Depth is `{12, M}` and λ is `{1, 2}`.
   A main effect measured at two levels is a slope between two points; neither factor's
   effect is claimed to be linear, and a different M or a different λ pair could reorder them.
3. **M is one probe point per property, chosen by C31's rule.** C32 did not sweep depth, so
   "depth" here means "probe point 12 versus this particular M", not "depth in general".
4. **The effective-λ correction is first-order** (§C32.5.3), and where λ_eff leaves the
   measured envelope the contrast is dropped rather than extrapolated — so D4 and D5 are
   scored on fewer contrasts than the full 2×2 provides.
5. **The envelope is measured at k ∈ {2, 4} only**, so the correction is available only at
   the primary k. That is where the decision rules live, but it means the secondary k have a
   raw depth contrast and no corrected one.
6. **The comparator is C31's curve, linearly interpolated**, which underestimates a concave
   best-of-N and biases every advantage **upward** — inherited from C26 §7.3 and C31, and
   running in the direction that flatters guidance. It affects all four corners roughly
   equally, so it should largely cancel in the contrasts, but "largely" is not "exactly".
7. **λ = 2 is not optimal and is not claimed to be.** §19 found validity collapsing above
   λ ≈ 2 on GP-MoLFormer; here the envelope shows what happens at λ up to 4 on one generator
   at two k, and nothing about where the optimum sits.
8. **No wall-clock claim.** Cost is processed generator tokens throughout.

---

## REPRODUCE

```bash
# C32.0  the pre-registration must already be on disk and must not be edited.
sha256sum outputs/c32_prereg/C32.0_preregistration.md   # must match prereg_lock.json

# Gates first.  G1 re-runs two C31 cells through C32's redirected code path and requires
# molecule-for-molecule identity; a failure stops the experiment.  ~2 min.
.venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage gate

# Arm A -- the two missing 2x2 corners, 30 cells, cheap k first, idempotent per cell.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup .venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage grid \
  > outputs/c32_run.log 2>&1 &

# Arm B -- the lambda envelope at probe point 12.  lambda = 1 and 2 are the 2x2's own
# deployed cells and are reused, not re-run.  30 new cells.
.venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage envelope

# Arm C -- the paired q-spread measurement.  ~2 min.
.venv/bin/python scripts/26_c32_depth_vs_lambda.py --stage spread

# score the pre-registration.  Reads existing artefacts only; generates nothing.
.venv/bin/python scripts/26_summarise_c32.py

# binding tests
.venv/bin/python -m pytest tests/test_depth_vs_lambda.py -q -p no:cacheprovider
```

**Artefact sizes.** C32 writes 60 new cell directories plus two gate directories; each cell
is 3 × 512 = 1,536 molecules, so `molecules.json` dominates at a few hundred kB. C32 writes
no `.npy` and no `.pt`, so the `.gitignore` exclusions do not apply and no `!` negation was
added. Everything this section prints is re-derivable from
`outputs/c32_summary/c32_metrics.json`, which is tracked.

**Pins.** Recorded by `write_run_context` beside every output directory: the
`entropy/gpt2_zinc_87m` revision SHA, torch, transformers, numpy, rdkit, float32, device
`cuda`, generator in eval mode with `requires_grad = False` and never modified. The head
checkpoints are C31's at head seed 1234, hashed by gate G2. Cost is
`processed_tokens_actual`; never wall-clock.
