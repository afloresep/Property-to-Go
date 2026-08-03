# C31 — does the crossing replicate on a second, independent molecular generator?

Draft section, written to be merged into `reports/pilot_report.md`. Author: reviewer,
2026-08-03. Conflicts with existing sections are **listed for the owner to merge** in
§C31.9, not edited in place.

**Headline, stated before the detail so it cannot be buried.** The pre-registered verdict is
**REPLICATES**: C28's crossing — guided decoding above the *oracle-selected* best-of-N
frontier at its own budget — is reproduced on `entropy/gpt2_zinc_87m` (GPT-2 architecture,
full softmax attention, 87M parameters, byte-level BPE, ~480M ZINC SMILES). 5 of
30 cells sit above the frontier with a seed-level t interval excluding zero.

**It is one of only two pre-registered rules that came out the way GP-MoLFormer predicted.
Of the six substantive decision rules, three fire and three do not, and four of 11
predictions are falsified.** The honest summary is that *the phenomenon* transfers and
*most of the specifics about it* do not.

| what | on GP-MoLFormer (C26–C30) | on `gpt2_zinc_87m` (C31) |
| --- | --- | --- |
| the crossing exists at small budgets | yes | **yes** — D1 fires |
| every property predicted best mid-network | yes | **yes** — D4 fires, third architecture |
| the **deployed** configuration crosses | yes, +0.0846 → +0.1044 | **no** — D6 does not fire, and on aromatic rings it is far *below* |
| crossing confined to k ≤ 4 | yes | **no** — D2 does not fire, one cell crosses at k = 8 |
| k is not a compute knob | yes, -0.0218 over the span | **no** — D3 does not fire; up to **+0.2347** over an 8.9× token span |
| frozen state beats `trivial` on aromatic rings | +0.0205 | **no** — +0.0003, a tie |
| best mid-network probe point | 3–4 | **2** — P4 falsified |

**The two negatives that matter most.**

1. **The deployed configuration does not cross.** The cell §22.2 singles out as *"the one
   cell in this table that cannot be dismissed as a post-hoc pick"* fails on all three
   properties here. Everything positive in C31 comes from the **mid-network λ=2** arm — the
   family §22.2 was hedging against.
2. **"Guided decoding has no compute knob" does not replicate.** On four of six arms,
   raising k from 2 to 32 raises the hit rate by more than the pre-registered +0.02, by as
   much as +0.2347. k is still a *bad* knob — best-of-N buys far more at the same tokens —
   but the C26/C28 claim as stated is refuted on this generator.

All rules and predictions are scored as written in §C31.7 and §C31.8, including the failures.

---

## C31.0 The pre-registration, verbatim

Copied byte-for-byte from `outputs/c31_prereg/C31.0_preregistration.md`, whose SHA-256 and
byte count were frozen into `outputs/c31_prereg/prereg_lock.json` **before the Stage 0
feasibility run** — not merely before the decision stage, so the validity floor that decides
whether this experiment is interpretable at all was committed before the validity was known.
`tests/test_second_generator.py` asserts the hash, the byte count, the verbatim copy, and
that the pre-registration's mtime strictly precedes every C31 artefact.

## C31.0.1 Why this experiment exists

C28 found the only positive result in this project. Priced against the **oracle-selected**
best-of-N frontier at their own budgets, 8 of 30 guided cells sit above it, all at k = 2 or
k = 4. C30 then re-ran the winning cells at eight probe-training seeds and 5 of 8 survived
with 95% t intervals excluding zero, the deployed configuration among them with a margin
that *grew* from +0.0846 to +0.1044.

**Every one of those numbers is a fact about GP-MoLFormer-Uniq.** One generator, 46.8M
parameters, linear attention, an atom-level SMILES vocabulary, trained on one corpus. The
project has three sources of variance under control — generation seed (C23–C28), probe seed
(C29, C30), and candidate-set size (C28) — and one it has never touched. The single largest
remaining threat to the crossing is that it is a property of that particular model.

C31 runs the pipeline end to end on **`entropy/gpt2_zinc_87m`**: GPT-2 architecture with
full softmax attention, 87M parameters, byte-level BPE over SMILES, trained on ~480M ZINC
strings. Different architecture, different attention, ~1.9× the parameters, different
corpus, different tokenisation. It emits **SMILES**, so no alternative serialization is
involved.

**On the "no second generator" rule.** The original project brief forbade a second
generator, so that a *negative* result could not be blamed on generator choice. The owner
has now explicitly instructed that this experiment be run, and the constraint is lifted by
owner instruction. The use here is the legitimate inverse of what the rule was written to
prevent: C31 does not shop for a generator that makes a negative go away, it tests whether a
*positive* generalises. Exactly one alternative generator is run, it is named in advance, and
it is not swapped if the answer is unwelcome.

This project has retracted its own numbers three times. **The prior that the crossing
replicates should be treated as weak**, and this pre-registration is written so that a
failure to replicate is as reportable, and is written up at the same length, as a success. A
failure would be a real finding: it would mean the crossing is a fact about GP-MoLFormer and
not about the method.

## C31.0.2 What is run — fixed here, in full

Every setting is in `configs/c31_second_generator.yaml`, which is additive and touches no
existing config. Values with a molecular counterpart are transcribed from it and the source
is named in that file.

**Generator.** `entropy/gpt2_zinc_87m`, revision `f42a5a10e24c0350aeadb50865bd90a714d0b2bf`,
float32, on one RTX 4090. **Frozen**: `eval()`, `requires_grad_(False)` on every parameter,
no fine-tuning, no LoRA, no RL, no activation edit, no weight change of any kind. Only the
probe head is trained. A parameter-sum fingerprint and the revision SHA are recorded in
`provenance.json` beside every output directory.

**Base policy.** Temperature 1.0, full-vocabulary sampling, `max_length = 256`, batch 256,
seed 20260729 — `configs/base_policy.yaml` verbatim except `max_length`, which moves from
202 to 256 because 202 is GP-MoLFormer's `max_position_embeddings` and 256 is this model's.

**Stage 0 — feasibility.** 2,048 unconditional molecules at the base policy; RDKit validity
and uniqueness measured and reported *before* any further compute is spent.

**Stage 1 — dataset.** 50,000 unconditional trajectories, minimum 8 content tokens, one
prefix per length quartile (4 per trajectory), grouped 80/10/10 split by canonical completed
SMILES with `splits.split_by_group`, split seed 11, prefix seed 12. All transcribed from
`configs/pilot_50k.yaml`.

**Properties.** `hbd_count` and `aromatic_rings` are **required** — they are the two anchors
that carry the crossing on GP-MoLFormer. `qed` is added if the budget allows; it never
crosses on the molecular model, and including it is the honest test of whether *that*
transfers too.

**Target intervals.** The uniform quantile rule of `configs/guidance.yaml`, applied by the
identical code path `binning.resolve_target_interval`: counts get `quantile_value q = 0.90`,
i.e. `[v, v+1)` for `v = round(quantile(0.90))`; continuous properties get
`quantile_band [0.85, 0.95)`, base rate 0.10 by construction. For `aromatic_rings` this
replaces the pilot's hand-picked `exact_value 3`, which is a value read off GP-MoLFormer's
distribution and is therefore not portable; `configs/guidance.yaml` itself records that the
quantile rule "would also have returned 3" on GP-MoLFormer, so the uniform rule agrees with
the pilot's frozen value on the molecular model. **Intervals and windows are derived from the
base sample once, written to disk, and SHA-256'd before any guided molecule exists**, and are
never re-derived.

**Probe points.** 13 (0 = embedding output, 1–12 = the twelve blocks), structurally matched
to GP-MoLFormer's 13. Every probe point is trained, at **three head seeds** (1234, 2345,
3456), so the depth curve is measured rather than assumed. One forward pass returns all 13,
so the sweep costs the processed tokens of one.

**Stage 3 — best-of-N.** Oracle-selected (true RDKit property of the finished molecule),
N ∈ {1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32}, 3 generation seeds (101, 202, 303), one pool of
32 × 512 molecules per seed evaluated over **all disjoint consecutive groups of N** — C26's
corrected estimator, not the nested one C26 retracted.

**Stage 4 — the guidance k sweep.** Two arms per property:

| arm | probe point | λ | GP-MoLFormer counterpart |
| --- | --- | ---: | --- |
| **deployed** | 12 | 1.0 | C28 strands A1 / C1 / C3 |
| **mid** | selected by C31.0.4 | 2.0 | C28 strands A3 / C2 |

`k ∈ {2, 4, 8, 16, 32}` — C28's grid; k = 1 is excluded for C28's reason, that one candidate
makes the softmax degenerate and the property term cannot act. Condition `throughout`,
ε = 1e-6, `cached` backend, batch 64, 512 molecules × 3 generation seeds per cell.

**What is shared with the molecular pipeline, deliberately.** The decoding rule is
`guidance.combine_scores` and the decoder is `guidance.guided_sample` — the molecular
functions, **imported, not re-implemented** — so a difference between the two generators
cannot be an implementation difference in the rule. Likewise `properties.*`, `binning.*`,
`bestofn.selection_key`, `bestofn.target_error`, `compute.ComputeMeter`, `splits.*`,
`metrics.*`, `heads.*`, `prefixes.*`, `probe_layers.train_one_probe`,
`scripts/21_n_sweep.py::score_pool`, `scripts/05_guided_generation.py::summarise`, and
`scripts/21_summarise_c26.py::interp` / `::t_interval`. What is new is the generator adapter
and the KV-cache repeat for standard attention, both of which are gated in C31.0.3.

## C31.0.3 Validity gates, run and reported BEFORE any decision rule

**G0 — the generator is usable at all.** Unconditional RDKit validity at the base policy
must be **≥ 0.80** on the Stage 0 sample. Below that, a hit rate computed over molecules that
do not parse is not a controller and C31 is **UNINTERPRETABLE**, not negative. Reported
before any further compute is spent.

**G1 — the cached decode equals full-prefix recomputation, state by state.** C24's gate,
re-run on this generator: candidate hidden states obtained from a shared KV cache versus
obtained by re-running the whole extended prefix, at **all 13 probe points**, on real C31
dataset sequences at real prefix positions. Tolerance **2e-3** absolute — C24's tolerance,
reused and not re-chosen. This is not a bit-identity claim: standard attention reduces in a
different order on the two paths, exactly as C24 found. The residual is reported, not
asserted away. **This is the gate that makes the token accounting meaningful.**

**G2 — the cached decode makes the same *decision*.** State equality within 2e-3 is not
automatically decision equality. On ≥ 512 real prefixes, for each arm's probe point and each
k in the grid, C31 computes the full guided sampling distribution
`softmax(combine_scores(log p_base, q, λ, ε))` from cached states and from full-recomputed
states. Required: **max absolute difference in sampling probability ≤ 1e-3**, and the
**argmax candidate identical on ≥ 99.5%** of prefixes. Deterministic, so this is a real
equality check and not a noise comparison.

**G3 — end-to-end backend equivalence, reported with a tolerance.** One cell (`hbd_count`,
probe point 12, λ = 1, k = 8, generation seed 101, 512 molecules) is decoded under both the
`cached` and the `full` backends. Because the two paths differ at ~1e-3 in `q`, sampled
trajectories may diverge, so exact equality is *not* required and claiming it would be
dishonest. Required: **|Δ hit rate| ≤ 0.05** (binomial noise alone at n = 512 is
±0.044 at two standard errors) and **|Δ tokens per molecule under full-recompute accounting|
/ value ≤ 0.02**. Reported as a residual either way.

**G4 — cost identity.** `processed_tokens_actual mod (k + 1) == 0` in every guidance cell and
every generation seed, as C28's G4 required. The `cached` backend charges `active` +
`active × k` at every guided step, so this is an exact arithmetic identity and any nonzero
residual means the accounting does not describe what ran.

**G5 — the target interval is a union of binner bins.** For every property,
`binning.interval_mask_coverage(binner, lo, hi, base_values)["is_exact"]` must be **True**,
and the number of selected bins must be ≥ 1. This is the `pilot_report.md` §11.5 bug, which
presented as a miscalibrated head rather than as an error: if an interval edge falls inside a
bin, the head silently predicts a strict subset of the target. Additionally the binner's
top category must sit **strictly above** the target value returned by the rule, so the
target is not absorbed into the "or more" bin.

**G6 — the freeze is a freeze.** `target_intervals.json` and `windows.json` are SHA-256'd
when written and re-checked at the start of every later stage; any change is a gate failure.
`splits.check_no_group_leakage` must pass, so no molecule's prefixes straddle train and test.

**If G0, G1, G2, G4, G5 or G6 fails, C31 stops and reports the failure. No decision rule
below is scored on a run whose gate failed.** G3 is reported as a residual and does not
block.

## C31.0.4 The mid-network probe point — selected by prediction, in advance

C17 §21.5.2 records why this matters: selecting a probe point post hoc on its *steering*
outcome manufactures a positive. The rule, fixed now:

> For each property, the mid-network probe point **M** is the probe point in
> `{1, …, 11}` that maximises **held-out validation** target AUROC — `P(y_final ∈ I)` scored
> against the true membership indicator — averaged over the three head seeds. Ties are broken
> by the **lower** index. Probe point 0 is excluded because it is the embedding output and
> carries no contextual computation; probe point 12 is excluded because it is the deployed
> point and is already the other arm.

Selection is on **validation**, and the depth curve is reported on **test**, so the probe
point is not chosen on the split it is then scored on. M is written to disk before any guided
molecule is generated at M.

## C31.0.5 The statistic, fixed now

For each guidance cell, the advantage against the oracle-selected best-of-N frontier at its
own budget is

```
adv(cell) = hit_rate(cell) - oracle_curve_interpolated_at(tokens_per_molecule(cell))
```

interpolated **linearly in processed tokens per molecule** using the *unmodified*
`scripts/21_summarise_c26.py::interp`, with a `price`-shaped loader analogous to
`scripts/23_summarise_c28.py::price`. Cost is **processed generator tokens**
(`compute.ComputeMeter`), and both `processed_tokens_actual` and
`processed_tokens_full_recompute` are reported. **No wall-clock claim is made anywhere in
C31.**

Uncertainty is a **seed-level Student t interval on 2 df** (t₀.₉₇₅,₂ = 4.302653) over the
three generation seeds, from `scripts/21_summarise_c26.py::t_interval`, with the **raw
per-seed values published alongside every interval**. **No percentile bootstrap is computed
anywhere in C31**: at n = 3 the percentile bootstrap of a mean is identically [min, max],
because P(all three resample the minimum) = 1/27 = 0.037 > 0.025, so it is a sign test with
null probability 0.25 wearing the costume of a confidence interval.

Points whose budget falls outside the measured best-of-N grid are flagged
`extrapolated_beyond_grid` and compared against the curve's terminal value, exactly as C28
pre-registered. Per-cell validity and uniqueness are reported for every cell.

## C31.0.6 Decision rules — scored as written

A cell **crosses** iff (i) its mean advantage against the oracle-selected curve is > 0,
(ii) its seed-level t interval on 2 df is strictly above 0, and (iii) its mean validity is
≥ 0.80.

| # | rule | fires iff |
| --- | --- | --- |
| **D1** | **the crossing replicates** | ≥ 1 cell crosses at k ∈ {2, 4} |
| **D2** | **the crossing is at the cheap end** | every crossing cell, at any k, has k ≤ 4 |
| **D3** | **no compute knob** | for every arm, hit rate at k = 32 minus hit rate at k = 2 is ≤ +0.02 |
| **D4** | **the depth curve replicates** | for both required properties, the probe point maximising held-out **test** target AUROC (averaged over head seeds) is strictly less than 12 |
| **D5** | **the crossing does not replicate** | no cell crosses at any k |
| **D6** | **the deployed configuration replicates** | the deployed arm (probe point 12, λ = 1) crosses at k = 2 |

**The verdict rule, fixed now:**

- **REPLICATES** iff D1 fires.
- **DOES NOT REPLICATE** iff D5 fires. The wording committed to in advance is: *"the crossing
  is a fact about GP-MoLFormer, not about the method"*, and it must be written up at the same
  length and with the same care as a positive.
- **PARTIAL** in any remaining case — i.e. some cell has a positive mean advantage but no cell
  clears the interval at k ≤ 4. Reported as **PARTIAL** and explicitly **not** as a
  replication.

D2, D3, D4 and D6 are scored independently of the verdict and are reported whichever way
D1/D5 go.

**D7 — the honesty rule, which fires regardless of the others.** For every cell whose
between-generation-seed sd exceeds its own mean advantage, C31 reports that cell as **"not
resolved at three generation seeds"** rather than as a win or a loss.

## C31.0.7 What would make C31 uninterpretable rather than negative

| condition | consequence |
| --- | --- |
| Stage 0 unconditional validity < 0.80 (**G0**) | UNINTERPRETABLE, stop before Stage 1 |
| G1, G2, G4, G5 or G6 fails | UNINTERPRETABLE, decision rules left unscored |
| more than 25% of guidance cells have mean validity < 0.80 | UNINTERPRETABLE — guidance is destroying the generator rather than steering it |
| both deployed-arm cells at k ∈ {2, 4} have mean validity < 0.80, for both required properties | UNINTERPRETABLE for the same reason |
| fewer than 3 generation seeds complete for any scored cell | the t interval's df would differ between cells; that cell is dropped and named |

**The repair is specified in advance, and it is deliberately asymmetric against this
experiment's own headline.** C30 wrote a flat per-cell validity floor of 0.90, it fired on 1
of 56 points, and it voided the whole run — a rule that cannot tell "one point tripped" from
"the experiment is degenerate". C31's rule instead is:

- A cell with mean validity **< 0.80** is **reported in full but excluded from crossing**,
  and named explicitly in the section. It cannot count as a win, because a hit rate over
  molecules that do not parse is not control. It is *not* excluded from D3 (the compute-knob
  rule), because a knob that destroys validity is still evidence about the knob.
- A cell with mean validity in **[0.80, 0.90)** is **validity-flagged**, reported with the
  flag, and **still scored**. Flagging without excluding is the conservative choice here only
  because exclusion at this level would remove cells from D3 and D5 as well.
- The **experiment** is declared degenerate only under the two population-level conditions in
  the table above.

Exclusion can only ever *remove* a crossing, never create one, which is the direction this
pre-registration wants its own errors to point.

## C31.0.8 Predictions, committed before the run

Scored in the section. Falsified predictions are reported as falsified, in the same table.

1. **P1** — Stage 0 unconditional RDKit validity ≥ 0.95. *(The model card reports 0.999 at
   temperature 1.0 over 1M compounds; our sampling is full-vocabulary at temperature 1.0, so
   this should hold with room to spare. If it does not, the card is wrong or our decode is.)*
2. **P2** — Stage 0 uniqueness ≥ 0.95 over the sample.
3. **P3** — the depth curve peaks strictly before probe point 12 for **both** required
   properties, i.e. D4 fires. *(C17/C23/C24 found a mid-network peak on GP-MoLFormer and on
   GPT-2 for text; this is a third architecture and the third independent test.)*
4. **P4** — the selected mid probe point **M ∈ [3, 8]** for both required properties.
   *(GP-MoLFormer selected 4 for `hbd_count` and 3 for `aromatic_rings`.)*
5. **P5** — the frozen-state head beats the `trivial` prefix-statistics baseline on held-out
   test target AUROC by ≥ 0.02 at its best probe point, for both required properties.
6. **P6** — D1 fires: at least one cell crosses at k ∈ {2, 4}.
7. **P7** — D6 fires: the deployed arm crosses at k = 2 for at least one required property.
   *(Stated separately from P6 and predicted to be the weaker of the two: on GP-MoLFormer the
   deployed cell's margin, +0.0846 in C28 and +0.1044 in C30, is the fifth largest of eight,
   and the mid-network λ = 2 arms carry the largest margins.)*
8. **P8** — D3 fires: hit rate at k = 32 minus k = 2 is ≤ +0.02 on **every** arm.
9. **P9** — `qed`, if run, does **not** cross at any k, on either arm. *(It never crosses on
   GP-MoLFormer.)*
10. **P10** — guidance mean validity at k = 2 stays ≥ 0.95 on every arm, because a two-way
    restricted softmax is close to the base policy.
11. **P11** — the base best-of-N curve is monotone non-decreasing in N on the measured grid
    for every property, up to seed noise. *(A violation would mean the comparator itself is
    unreliable and would be reported as such.)*

## C31.0.9 What C31 will NOT do

Stated so its absence is not read as an omission.

- **No weight of the generator changes.** No fine-tuning, no LoRA, no RL, no activation edit,
  no prompt tuning. Only the probe head is trained.
- **No alternative serialization.** SMILES only. No SAFE (`datamol-io/safe-gpt`,
  `nvidia/NV-GenMol-89M-v1`), no SELFIES. If `entropy/gpt2_zinc_87m` turned out unusable, C31
  would stop and report rather than substitute a SAFE or SELFIES model.
- **No third generator.** Exactly one alternative generator, named in advance, not swapped.
- **No λ sweep.** λ ∈ {1, 2} only, the two values that define the two arms.
- **No pooling variant, no calibration variant, no DAgger round, no head-selected best-of-N**
  unless Stage 6 budget remains after Stages 0–5 are complete, in which case it is reported
  as a clearly labelled extension.
- **No wall-clock claim anywhere.** Cost is processed generator tokens.
- **No existing artefact, report, config or `outputs/` directory is edited.** Conflicts with
  the merged report go in a "What C31 changes elsewhere" section for the owner to merge,
  exactly as C23–C30 did.

## C31.0.10 The reporting rule

Whatever comes out, the section states it in this order: what was run, then the gates as
numeric residuals, then the probe depth curve, then the per-cell tables with **every per-seed
value in full**, then the decision rules scored as written, then the predictions scored
including the falsified ones, then limitations, then what it changes elsewhere. A negative
outcome is written up at the same length as a positive one. If a deviation from this
pre-registration proves necessary, **the deviation is reported and this document is not
amended**.

---

## C31.1 What was run

| stage | what | output |
| --- | --- | --- |
| 0 | 2048 unconditional molecules; G0 and G1 | `outputs/c31_feasibility/` |
| 1 | 50,000 trajectories, 199,584 prefix rows, states at all 13 probe points, frozen intervals and windows; G5 and G6 | `outputs/c31_zinc50k/`, `outputs/c31_layer_states/` |
| 2 | 13 probe points × 3 head seeds × 3 properties = 117 heads, plus 9 `trivial` heads | `outputs/c31_heads/` |
| — | G2, the decision-equality gate | `outputs/c31_gates/` |
| 3 | oracle-selected best-of-N, N ∈ {1…32}, 3 seeds, 16,384-molecule pool per seed | `outputs/c31_bestofn_*/` |
| 4 | the k sweep, 30 cells × 1,536 molecules; G4 | `outputs/c31_ksweep_*/` |
| — | G3, end-to-end backend equivalence | `outputs/c31_gates/` |
| post hoc | the length-matched control (§C31.6) | `outputs/c31_length_control/` |

**The generator.** `entropy/gpt2_zinc_87m`, revision `f42a5a10e24c0350aeadb50865bd90a714d0b2bf`,
87,331,584 parameters, 12 blocks, 768 wide, vocabulary 2707, `n_positions` 256. The
parameter-sum checksum is `19996.594887717303` and is asserted identical across the feasibility,
dataset and best-of-N stages by `tests/test_second_generator.py`. `requires_grad` is `False`
on every parameter and `model.training` is `False` in every recorded fingerprint. **No
weight changed at any point.**

**On the "no second generator" rule.** The original project brief forbade a second
generator, so that a *negative* result could not be blamed on generator choice. **The owner
explicitly instructed this experiment**, and the constraint is lifted by owner instruction.
The use here is the legitimate inverse of what the rule was written to prevent: C31 does not
shop for a generator that makes a negative go away, it tests whether a *positive*
generalises. One alternative generator, named in advance, not swapped. It emits **SMILES**;
no SAFE and no SELFIES model was used, considered as a substitute, or loaded — a test
asserts this on the import graph.

**Nothing about the method is forked.** The decoding rule is `guidance.combine_scores` and
the decoder is `guidance.guided_sample` — the molecular functions, *imported*. So are
`properties.compute_all_properties`, `binning.*`, `bestofn.selection_key`,
`bestofn.target_error`, `compute.ComputeMeter`, `splits.split_by_group`, `metrics.*`,
`heads.MLPHead`, `prefixes.select_quartile_prefixes`, `probe_layers.train_one_probe`,
`scripts/05_guided_generation.py::summarise`, `scripts/21_n_sweep.py::score_pool` and
`scripts/21_summarise_c26.py::interp` / `::t_interval`. `tests/test_second_generator.py`
asserts on the AST that none of these is redefined locally, so a future copy-paste fork
fails a test rather than drifting quietly. **If the crossing had failed to replicate here,
the failure could not have been an implementation difference in the method, because there
is no second implementation of the method.**

Exactly two lines of the molecular library are new, and both default to the existing
behaviour:

- `guidance._candidate_states_cached` now reads `getattr(gen, "repeat_cache_fn", None) or
  repeat_cache`. GPT-2 carries a standard `(key, value)` cache; `generation.repeat_cache` is
  written for GP-MoLFormer's linear-attention running-sum cache and is not applicable.
  `FrozenGenerator` has no `repeat_cache_fn`, so every molecular call site takes the same
  path it was measured on — asserted by a test.
- `probe_layers.train_one_probe` gained `trainer=None`, defaulting to `heads.train_head`.
  C31 passes `generality.train_head_on_device`, which `tests/test_generality.py` already
  asserts is **bit-identical** to `heads.train_head` on CPU.

### C31.1.1 Deviations from the pre-registration, reported not amended

**One deviation.** C31.0.2 transcribed `configs/guidance.yaml`'s guidance `batch_size = 64`.
At **k = 32** that exhausts 24 GB: GPT-2 keeps a full `(key, value)` KV cache, so repeating
it 32-fold across 64 concurrent sequences is ~9 GB *before* the copy the repeat itself makes.
GP-MoLFormer's linear-attention cache is a running sum and does not grow this way, which is
why the molecular config could fix one batch size for every k and C31 cannot.

The sweep was re-run at **batch 16 for every cell**, so k remains the only thing that varies
across the sweep. The pre-registration is **not amended** (C31.0.10). The deviation is
accounting-neutral by construction — `ComputeMeter` charges `active` (not-done) sequences
only, so processed tokens do not depend on batch size — and it changes the RNG draw, not the
per-molecule sampling distribution. `outputs/c31_gates/batch_size_deviation.json` keeps the
batch-64 numbers for the 24 cells that completed before the OOM so a reader can check: the
deployed HBD-count cell at k = 2 gave hit rate 0.3151 at 130.93 tokens/molecule at batch 64
and 0.3092 at 130.78 at batch 16 — a shift well inside the between-seed spread of that cell.

**One post-hoc addition**, labelled as such and kept out of the pre-registered result: the
length-matched control in §C31.6.

---

## C31.2 Validity gates, as numeric residuals, before any decision rule

### C31.2.1 G0 — the generator is usable

2048 unconditional molecules at the base policy:

| quantity | measured | pre-registered floor |
| --- | ---: | ---: |
| RDKit validity | **0.9980** | ≥ 0.80 |
| uniqueness | **1.0000** | — |
| mean content tokens | 34.08 | — |
| sequences hitting `max_length` | 0 | — |

G0 **passes** with a factor of 1.25 to spare on the floor and clears prediction P1's 0.95 as
well. Zero sequences reach `max_length`, so this generator terminates on its own `</s>` and
the truncation lever §21 warns about is not available to the decoder. The full 50,000-molecule
Stage 1 draw agrees: validity 0.9980, uniqueness 1.0000.

### C31.2.2 G1 — the cached decode equals full-prefix recomputation

C24's gate, re-run on this generator at **all 13 probe points**, on real C31 sequences
at real prefix positions. This is the gate that makes the token accounting mean anything:
`processed_tokens_actual` charges **one** token per candidate precisely because evaluating a
candidate from a shared cache is the same computation as re-running its whole prefix.

| quantity | value |
| --- | ---: |
| max absolute difference over 13 probe points | **5.722e-06** |
| tolerance (C24's, reused not re-chosen) | 2e-03 |
| worst probe point | 11 |
| hidden-state scale | 26.83 |
| relative to state scale | 2.13e-07 |
| bit-identical | **no** |

G1 **passes** with ~350× margin. It is *not* a bit-identity claim and the section does not
make one: standard attention reduces in a different order on the two paths, exactly as C24
found on GPT-2 for text. The molecular linear-attention backends *are* bit-identical; this
one is not, and the residual is published rather than asserted away.

### C31.2.3 G2 — the cached decode makes the same *decision*

State equality to 2e-3 is not automatically decision equality: `q` feeds a logarithm and then
a softmax over k candidates. G2 measures the thing that actually picks a token, on 512 real
prefixes, for every arm × every k — 30 cells in all. It is deterministic, so this is an
equality check and not a noise comparison.

| quantity | measured | required |
| --- | ---: | ---: |
| max absolute difference in sampling probability | **2.563e-06** | ≤ 1e-03 |
| minimum argmax agreement | **1.000000** | ≥ 0.995 |

G2 **passes**: the two paths pick the same candidate on **every one** of the prefixes tested,
in every cell. The 1e-3 state residual of G1 never reaches a token.

### C31.2.4 G4 — the cost identity

The `cached` backend charges `active` + `active × k` at every guided step, so
`processed_tokens_actual mod (k + 1)` is an exact arithmetic identity. Residual is **0** in
all 30 cells and all three generation seeds of each — checked from the cells themselves,
not only from the gate's summary. G4 **passes**.

### C31.2.5 G5 — the target interval is an exact union of binner bins

`pilot_report.md` §11.5's bug presents as a *miscalibrated head*, not as an error, so it is
checked numerically rather than reasoned about.

| property | target interval | base rate | binner | bins selected | exact union of bins |
| --- | --- | ---: | --- | ---: | :---: |
| HBD count | `[3, 4)` | 0.0901 | categorical (7) | 1 | yes |
| aromatic rings | `[3, 4)` | 0.1254 | categorical (6) | 1 | yes |
| QED | `[0.8503, 0.8968)` | 0.1000 | quantile (20) | 2 | yes |

G5 **passes** on all three. Worth recording: the uniform quantile rule (`quantile_value` at
q = 0.90) returned **3** for aromatic rings on this generator — the same value the pilot
hand-picked from GP-MoLFormer's distribution, and the same value `configs/guidance.yaml`
records the quantile rule would have returned there. The two generators agree on the target
without being made to.

### C31.2.6 G6 — the freeze is a freeze

`splits.check_no_group_leakage` passes: no molecule's prefixes straddle train and test.
`target_intervals.json` and `windows.json` were SHA-256'd at write time and both still hash
to the recorded value — asserted by a test that re-reads the files rather than trusting the
gate. Splits: 159,700 / 19,808 / 20,076 prefix rows over
49,896 kept trajectories.

### C31.2.7 G3 — end-to-end backend equivalence, reported as a residual

Non-blocking by pre-registration, because G1 shows the two paths differ at ~1e-6 in the
candidate state and a sampled trajectory *may* therefore diverge. Claiming exact equality
would be dishonest.

| quantity | measured | tolerance |
| --- | ---: | ---: |
| Δ hit rate (`cached` − `full`) | +0.0000 | ≤ 0.05 |
| Δ tokens/molecule under full-recompute accounting, relative | 0.000000 | ≤ 0.02 |
| binomial 2 s.e. at n = 512, for scale | 0.0442 | — |

The `actual` token counts differ **by design** — that is precisely what the two accounting
rules measure — and the section does not present them as equal; a test asserts they differ.

**All six blocking gates pass. No decision rule below is scored past a failed gate**, and a
test enforces that a verdict cannot be issued while a required gate failed.

---

## C31.3 The probe depth curve — measured at every probe point, on a third architecture

13 probe points × 3 head seeds × 3 properties, all from one forward pass per batch, so the
sweep cost the processed tokens of a single-layer extraction. `trivial` is
`tokens.prefix_features` — the same 21 cheap prefix statistics the molecular baseline uses —
computed on the **atom-level re-split** of the decoded prefix, because this generator's
vocabulary is byte-level BPE with multi-character merges (`Cc`, `ccc`, `(=`). Feeding BPE
tokens straight to `tokens.classify` would classify almost everything as an unrecognised
structural token and would make the baseline artificially weak, which is the exact failure
`tokens.py`'s own docstring says the baseline exists to prevent.

Held-out target AUROC, mean over three head seeds. Selection is on **validation**; the curve
is reported on **test**, so the probe point is never chosen on the split it is scored on.

| probe point | HBD count val | HBD count test | aromatic rings val | aromatic rings test | QED val | QED test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5859 | 0.5844 | 0.6563 | 0.6466 | 0.5598 | 0.5538 |
| 1 | 0.7856 | 0.7892 | 0.8426 | 0.8444 | 0.6993 | 0.6920 |
| 2 **(M)** | 0.8288 | 0.8322 | 0.8688 | 0.8687 | 0.7505 | 0.7436 |
| 3 | 0.8226 | 0.8315 | 0.8634 | 0.8661 | 0.7515 | 0.7472 |
| 4 | 0.8240 | 0.8307 | 0.8556 | 0.8588 | 0.7526 | 0.7530 |
| 5 | 0.8207 | 0.8246 | 0.8501 | 0.8535 | 0.7535 | 0.7539 |
| 6 **(M)** | 0.8157 | 0.8200 | 0.8428 | 0.8454 | 0.7538 | 0.7502 |
| 7 | 0.8105 | 0.8120 | 0.8396 | 0.8412 | 0.7517 | 0.7482 |
| 8 | 0.8065 | 0.8092 | 0.8367 | 0.8375 | 0.7513 | 0.7457 |
| 9 | 0.8022 | 0.8074 | 0.8328 | 0.8311 | 0.7487 | 0.7432 |
| 10 | 0.7987 | 0.8020 | 0.8293 | 0.8292 | 0.7431 | 0.7415 |
| 11 | 0.7894 | 0.7922 | 0.8232 | 0.8248 | 0.7354 | 0.7356 |
| 12 | 0.7805 | 0.7866 | 0.8187 | 0.8205 | 0.7301 | 0.7292 |
| `trivial` baseline | — | 0.7346 | — | 0.8683 | — | 0.7041 |

**(M)** marks a probe point selected as the mid-network arm for *at least one* property; the
per-property selections are in the next table.

| property | selected M (validation) | test peak | test AUROC at peak | test AUROC at probe point 12 | `trivial` | margin over `trivial` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HBD count | 2 | 2 | 0.8322 | 0.7866 | 0.7346 | +0.0976 |
| aromatic rings | 2 | 2 | 0.8687 | 0.8205 | 0.8683 | +0.0003 |
| QED | 6 | 5 | 0.7539 | 0.7292 | 0.7041 | +0.0497 |

**The shape replicates, and it replicates cleanly.** Every property rises steeply from the
embedding output, peaks early, and then declines monotonically to the final probe point —
without a single reversal larger than the head-seed sd, which is at most 0.0041 anywhere in
the table. Probe point 12 is the minimum over the range peak–12 for all three properties.
This is the third architecture on which the finding holds: GP-MoLFormer (C17, peaks at 3–5),
GPT-2 on text attributes (C24), and now GPT-2 on SMILES. **D4 fires; P3 is confirmed.**

**Two things do not replicate.**

**The peak moves earlier.** C17 selected probe point 3 for aromatic rings and 4 for HBD
count; C31 selects **2 for both**. QED selects 6 (test peak 5) where C17 selected 4. So
"mid-network" survives as a *direction* and not as a *depth*: **P4, which committed to
M ∈ [3, 8], is falsified for both required properties.** A plausible reading is that a
byte-level BPE vocabulary front-loads the work — a single token already carries `Cc` or
`ccc`, so the composition a mid-network layer has to do on an atom-level vocabulary is partly
done by the embedding — but C31 ran no experiment to test that, and it is offered as a
conjecture, not a result.

**The frozen state stops beating surface counting on aromatic rings.** Read the last two
columns of the summary table: the margin over `trivial` is +0.0003, two orders of magnitude
below the +0.0205 C17 measured on GP-MoLFormer and far below the 0.02 P5 committed to.
**P5 is falsified.** HBD count is the opposite case
(+0.0976, comparable to C17's +0.1129) and QED sits between (+0.0497 against C17's +0.0374).

This is worth stating plainly because it is uncomfortable for the aromatic-rings result
below: on this generator, **a 21-feature count of the prefix predicts the final aromatic-ring
class as well as 768 dimensions of frozen transformer state**. The aromatic-rings crossing in
§C31.4 is therefore a demonstration that *a cheap predictor steered through the decoder beats
best-of-N at matched tokens* — which is still the claim under test — but it is **not**
evidence that the generator's internal representation is doing the work. On GP-MoLFormer the
two questions were separable; here, for aromatic rings, they are not.

---

## C31.4 The frontier and the k sweep

### C31.4.1 The comparator

Oracle-selected best-of-N: `bestofn.selection_key` on the **true RDKit property of the
finished molecule**, which guidance never sees. C26's grid and C26's corrected estimator (all
disjoint consecutive groups of N over one 16,384-molecule pool per seed), 3 generation seeds.

**HBD count**, target `[3, 4)`, base rate 0.0901

| N | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| processed tokens / molecule | 36.04 | 72.09 | 108.13 | 144.17 | 216.26 | 288.35 | 324.39 | 432.52 | 576.70 | 865.07 | 1153.40 |
| hit rate | 0.0865 | 0.1654 | 0.2366 | 0.3018 | 0.4151 | 0.5151 | 0.5551 | 0.6623 | 0.7585 | 0.8939 | 0.9453 |

**aromatic rings**, target `[3, 4)`, base rate 0.1254

| N | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| processed tokens / molecule | 36.04 | 72.09 | 108.13 | 144.17 | 216.26 | 288.35 | 324.39 | 432.52 | 576.70 | 865.07 | 1153.40 |
| hit rate | 0.1244 | 0.2325 | 0.3275 | 0.4119 | 0.5458 | 0.6514 | 0.6881 | 0.7958 | 0.8809 | 0.9536 | 0.9811 |

**QED**, target `[0.8503, 0.8968)`, base rate 0.1000

| N | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| processed tokens / molecule | 36.04 | 72.09 | 108.13 | 144.17 | 216.26 | 288.35 | 324.39 | 432.52 | 576.70 | 865.07 | 1153.40 |
| hit rate | 0.1021 | 0.1934 | 0.2749 | 0.3503 | 0.4752 | 0.5776 | 0.6229 | 0.7294 | 0.8167 | 0.9247 | 0.9648 |


The curve is monotone in N on every property (**P11 confirmed**) and is a strong comparator:
best-of-32 reaches 0.9453 / 0.9811 / 0.9648 against base rates near 0.10. It is a *stronger*
comparator than GP-MoLFormer's, because this generator's per-molecule cost is lower (36.04
processed tokens against GP-MoLFormer's 44.4), so a fixed token budget buys more draws.

**Four cells are flagged `extrapolated_beyond_grid`** and are named here rather than left in
the JSON: the four deployed-arm k = 32 cells sit at 1161.34–1171.61 processed tokens per
molecule against a measured grid maximum of 1153.40, so their comparator is the curve's
terminal value rather than an interpolation. All four are far *below* the frontier, so the
flag cannot be helping any crossing; C31 ran no extended best-of-N grid, and C28 §C28.6's
precedent for measuring one is noted but not followed.

### C31.4.2 HBD count

| arm | probe point | λ | k | processed tokens / molecule | hit rate | per-seed hit rates (101, 202, 303) | oracle best-of-N at that budget | advantage | 95% t interval (2 df) | validity | crosses |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | :---: |
| deployed | 12 | 1 | 2 | 130.78 | **0.3092** | 0.3125, 0.3047, 0.3105 | 0.2776 | **+0.0317** | [-0.0045, +0.0678] | 1.0000 | no |
| deployed | 12 | 1 | 4 | 200.28 | **0.3837** | 0.3894, 0.4004, 0.3613 | 0.3900 | **-0.0063** | [-0.0767, +0.0641] | 0.9993 | no |
| deployed | 12 | 1 | 8 | 340.49 | **0.3424** | 0.3340, 0.3398, 0.3535 | 0.5711 | **-0.2286** | [-0.2341, -0.2232] | 1.0000 | no |
| deployed | 12 | 1 | 16 | 617.87 | **0.3496** | 0.3965, 0.3379, 0.3145 | 0.7778 | **-0.4282** | [-0.5667, -0.2898] | 1.0000 | no |
| deployed | 12 | 1 | 32 | 1163.57 | **0.3418** | 0.3340, 0.3516, 0.3398 | 0.9453 | **-0.6035** | [-0.6548, -0.5522] | 1.0000 | no |
| mid | 2 | 2 | 2 | 125.78 | **0.4980** | 0.5098, 0.4961, 0.4883 | 0.2685 | **+0.2295** | [+0.1785, +0.2807] | 1.0000 | **yes** |
| mid | 2 | 2 | 4 | 191.42 | **0.5457** | 0.5225, 0.5303, 0.5843 | 0.3761 | **+0.1696** | [+0.1071, +0.2322] | 0.9974 | **yes** |
| mid | 2 | 2 | 8 | 325.21 | **0.5719** | 0.6020, 0.5645, 0.5492 | 0.5559 | **+0.0159** | [-0.0778, +0.1098] | 0.9961 | no |
| mid | 2 | 2 | 16 | 593.05 | **0.5614** | 0.5812, 0.5421, 0.5610 | 0.7661 | **-0.2047** | [-0.2769, -0.1327] | 0.9961 | no |
| mid | 2 | 2 | 32 | 1099.48 | **0.5592** | 0.5714, 0.5639, 0.5422 | 0.9357 | **-0.3765** | [-0.4445, -0.3088] | 0.9954 | no |

### C31.4.3 Aromatic rings

| arm | probe point | λ | k | processed tokens / molecule | hit rate | per-seed hit rates (101, 202, 303) | oracle best-of-N at that budget | advantage | 95% t interval (2 df) | validity | crosses |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | :---: |
| deployed | 12 | 1 | 2 | 131.37 | **0.2064** | 0.2344, 0.2129, 0.1719 | 0.3820 | **-0.1756** | [-0.2621, -0.0890] | 1.0000 | no |
| deployed | 12 | 1 | 4 | 200.74 | **0.3781** | 0.3672, 0.3738, 0.3933 | 0.5170 | **-0.1389** | [-0.1956, -0.0823] | 0.9987 | no |
| deployed | 12 | 1 | 8 | 338.05 | **0.4234** | 0.4373, 0.4062, 0.4266 | 0.7017 | **-0.2783** | [-0.3024, -0.2543] | 0.9980 | no |
| deployed | 12 | 1 | 16 | 613.15 | **0.4326** | 0.4023, 0.4707, 0.4247 | 0.8901 | **-0.4575** | [-0.5731, -0.3416] | 0.9993 | no |
| deployed | 12 | 1 | 32 | 1162.86 | **0.4410** | 0.4414, 0.4375, 0.4442 | 0.9811 | **-0.5401** | [-0.5552, -0.5249] | 0.9993 | no |
| mid | 2 | 2 | 2 | 126.49 | **0.5430** | 0.5352, 0.5586, 0.5352 | 0.3705 | **+0.1724** | [+0.1213, +0.2235] | 1.0000 | **yes** |
| mid | 2 | 2 | 4 | 190.41 | **0.7451** | 0.7308, 0.7539, 0.7505 | 0.4978 | **+0.2473** | [+0.1960, +0.2986] | 0.9961 | **yes** |
| mid | 2 | 2 | 8 | 321.98 | **0.7451** | 0.7632, 0.7266, 0.7456 | 0.6856 | **+0.0595** | [+0.0236, +0.0957] | 0.9987 | **yes** |
| mid | 2 | 2 | 16 | 585.48 | **0.7438** | 0.7202, 0.7721, 0.7392 | 0.8831 | **-0.1392** | [-0.2318, -0.0467] | 0.9961 | no |
| mid | 2 | 2 | 32 | 1118.78 | **0.7271** | 0.7216, 0.7358, 0.7239 | 0.9778 | **-0.2507** | [-0.2912, -0.2104] | 0.9948 | no |

### C31.4.4 QED

| arm | probe point | λ | k | processed tokens / molecule | hit rate | per-seed hit rates (101, 202, 303) | oracle best-of-N at that budget | advantage | 95% t interval (2 df) | validity | crosses |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | :---: |
| deployed | 12 | 1 | 2 | 131.79 | **0.2435** | 0.2422, 0.2285, 0.2598 | 0.3244 | **-0.0809** | [-0.1297, -0.0321] | 1.0000 | no |
| deployed | 12 | 1 | 4 | 200.86 | **0.2376** | 0.2422, 0.2422, 0.2285 | 0.4485 | **-0.2109** | [-0.2267, -0.1951] | 1.0000 | no |
| deployed | 12 | 1 | 8 | 338.26 | **0.2129** | 0.2168, 0.2148, 0.2070 | 0.6366 | **-0.4237** | [-0.4583, -0.3894] | 1.0000 | no |
| deployed | 12 | 1 | 16 | 612.61 | **0.2332** | 0.2422, 0.2246, 0.2329 | 0.8302 | **-0.5970** | [-0.6614, -0.5325] | 0.9993 | no |
| deployed | 12 | 1 | 32 | 1161.34 | **0.2249** | 0.2246, 0.2266, 0.2235 | 0.9648 | **-0.7399** | [-0.7637, -0.7160] | 0.9987 | no |
| mid | 6 | 2 | 2 | 129.32 | **0.3055** | 0.2935, 0.3105, 0.3125 | 0.3192 | **-0.0137** | [-0.0433, +0.0159] | 0.9993 | no |
| mid | 6 | 2 | 4 | 201.19 | **0.3112** | 0.3105, 0.3145, 0.3086 | 0.4491 | **-0.1379** | [-0.1475, -0.1283] | 1.0000 | no |
| mid | 6 | 2 | 8 | 339.68 | **0.3212** | 0.3464, 0.3340, 0.2832 | 0.6380 | **-0.3168** | [-0.4036, -0.2298] | 0.9974 | no |
| mid | 6 | 2 | 16 | 620.87 | **0.3032** | 0.2933, 0.3190, 0.2975 | 0.8333 | **-0.5300** | [-0.5468, -0.5133] | 0.9961 | no |
| mid | 6 | 2 | 32 | 1171.61 | **0.2808** | 0.3006, 0.2868, 0.2549 | 0.9648 | **-0.6841** | [-0.7246, -0.6435] | 0.9948 | no |

---

## C31.5 What the numbers say

### C31.5.1 The crossing replicates, on the mid-network arm

5 cells sit above the oracle-selected frontier at their own budget with a t interval
on 2 df excluding zero:

- `c31_ksweep_aromatic_rings_mid_L2_lam2_k4` — advantage **+0.2473**, 95% t interval [+0.1960, +0.2986], at 190.41 processed tokens/molecule
- `c31_ksweep_hbd_count_mid_L2_lam2_k2` — advantage **+0.2295**, 95% t interval [+0.1785, +0.2807], at 125.78 processed tokens/molecule
- `c31_ksweep_aromatic_rings_mid_L2_lam2_k2` — advantage **+0.1724**, 95% t interval [+0.1213, +0.2235], at 126.49 processed tokens/molecule
- `c31_ksweep_hbd_count_mid_L2_lam2_k4` — advantage **+0.1696**, 95% t interval [+0.1071, +0.2322], at 191.42 processed tokens/molecule
- `c31_ksweep_aromatic_rings_mid_L2_lam2_k8` — advantage **+0.0595**, 95% t interval [+0.0236, +0.0957], at 321.98 processed tokens/molecule

**Every crossing cell is the mid-network λ=2 arm.** The margins are large — the biggest on
GP-MoLFormer was +0.2499 — and the intervals are comfortably clear of zero rather than
marginal.

**But one of them is at k = 8, so D2 does not fire.** On GP-MoLFormer every cell whose
interval excluded zero was at k = 2 or k = 4; the one k = 8 cell in C28's table (A3, +0.0267)
did **not** clear its interval, which is why "every crossing cell has k ≤ 4" was worth
pre-registering. Here it fails, and it is scored as written. `aromatic_rings`, mid, k = 8 crosses at +0.0595 with interval `[+0.0236, +0.0957]`.
The claim that survives is the weaker one: **the crossing is a small-budget phenomenon, not a
k ≤ 4 phenomenon.**

**D1 fires. The pre-registered verdict is REPLICATES.**

### C31.5.2 The deployed configuration does not replicate

This is the most important negative in the section and it is not a near miss.

On GP-MoLFormer the deployed configuration — final probe point, λ=1 — crossed at k=2 with
+0.0846 (C28), which C30 raised to +0.1044 across eight probe seeds. It was the cell the
project's abstract quoted. On `gpt2_zinc_87m`:

- **HBD count, deployed, k=2**: +0.0317, interval `[-0.0045, +0.0678]` — the right sign,
  and it does **not** clear zero. Under the pre-registered rule this is not a crossing.
- **Aromatic rings, deployed, k=2**: **-0.1756**, interval `[-0.2621, -0.0890]`. Not a near
  miss; the deployed arm is decisively *below* the frontier, with an interval excluding zero
  on the negative side.
- **QED, deployed, k=2**: -0.0809, also excluding zero on the negative side.

**D6 does not fire.** The configuration this project actually deployed does not beat
best-of-N on a second generator at any k, on any of the three properties.

The read that survives both experiments is narrower than the one C28 and C30 support on their
own: **guidance can beat oracle-selected best-of-N at small budgets, but only from a
well-chosen mid-network probe point at λ=2, and which probe point that is has to be
re-selected per generator.** The specific configuration in the deployed pipeline is not the
one that carries the result, and on this generator it is not a winner at all.

### C31.5.3 The advantage decays with budget, and the deployed arm goes badly negative

Both arms' advantage falls as k rises, because the frontier climbs steeply with tokens while
guidance's hit rate saturates. At k = 8 and beyond the deployed arm is strongly negative on
every property. This is the same shape C28 found and is the reason the crossing is a
*small-budget* phenomenon rather than a general claim.

### C31.5.4 D3 — "guided decoding has no compute knob" does **not** replicate

| property | arm | hit rate at k=2 | hit rate at k=32 | Δ | tokens at k=2 | tokens at k=32 | token ratio | ≤ +0.02 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| aromatic rings | deployed | 0.2064 | 0.4410 | +0.2347 | 131.37 | 1162.86 | 8.85x | **no** |
| aromatic rings | mid | 0.5430 | 0.7271 | +0.1841 | 126.49 | 1118.78 | 8.84x | **no** |
| HBD count | deployed | 0.3092 | 0.3418 | +0.0326 | 130.78 | 1163.57 | 8.90x | **no** |
| HBD count | mid | 0.4980 | 0.5592 | +0.0611 | 125.78 | 1099.48 | 8.74x | **no** |
| QED | deployed | 0.2435 | 0.2249 | -0.0186 | 131.79 | 1161.34 | 8.81x | yes |
| QED | mid | 0.3055 | 0.2808 | -0.0248 | 129.32 | 1171.61 | 9.06x | yes |

**D3 does not fire.** C26's headline — *guided decoding has no compute knob* — and C28's
confirmation of it (−0.0218 over the whole k span) do **not** transfer. On four of six arms
raising k from 2 to 32 raises the hit rate by more than the pre-registered +0.02 threshold,
and on `aromatic_rings` at the deployed probe point by **+0.2347**. Only the two `qed` arms
behave the way GP-MoLFormer did, and they do so by getting slightly *worse*.

Two qualifications, both of which cut against reading this as good news for guidance.

**First, k is still a bad knob.** It costs ~8.8× the tokens across that span and the
frontier climbs far faster: every arm's *advantage* against best-of-N falls monotonically
with k, and at k = 32 every single cell is below the frontier by between −0.25 and −0.74.
Buying accuracy with k is strictly worse than buying it with N.

**Second, a large part of the k effect is length, not control.** §C31.6's post-hoc control
shows that guided molecules at k = 2 are ~25% longer than base-policy ones and shrink back
to base length as k rises. On the `aromatic_rings` deployed arm — the one with the +0.2347
span — length matching *raises* the k = 2 hit rate from 0.2064 to 0.2677 and leaves k = 32
essentially unchanged, so roughly a quarter of the apparent knob is the low-k cells being
penalised for over-long molecules rather than the high-k cells being better controlled.
C31 did not design this out, and says so in §C31.10.

### C31.5.5 D7 — the honesty rule

Cells whose between-generation-seed sd exceeds their own mean advantage, reported as **not
resolved at three generation seeds** rather than as a win or a loss:

- `c31_ksweep_hbd_count_deployed_L12_lam1_k4` — mean advantage -0.0063, between-seed sd 0.0283
- `c31_ksweep_hbd_count_mid_L2_lam2_k8` — mean advantage +0.0159, between-seed sd 0.0378

### C31.5.6 Validity is not the story here

Every guidance cell holds validity above 0.99, and the base policy is at 0.9980. No cell
comes near the 0.80 exclusion floor or even the 0.90 flag level, so the repair machinery
C31.0.7 specified in advance never fired and no cell was excluded from anything. This is a
difference from GP-MoLFormer, where §19 found validity collapsing above λ ≈ 2 and where C30's
flat 0.90 screen fired on 1 of 56 points and voided a whole run. λ = 2 is simply not a
destructive setting on this generator.

---

## C31.6 POST HOC, NOT PRE-REGISTERED — the length-matched control

Guided molecules come out longer than base-policy ones, and both HBD count and aromatic-ring
count grow with molecule size, so a decoder that only made molecules bigger would look like a
controller. `pilot_report.md` §19 ran this control on GP-MoLFormer. **C31.0 did not
pre-register it.** It is run here and reported as a labelled post-hoc addition; it does not
replace any pre-registered number.

`scripts/05_guided_generation.py::length_matched_hit_rate` is imported: each cell's
per-length-bin hit rate is reweighted by the **unguided** length distribution, bin width 5
content tokens. The reference is 512 × 3 base-policy molecules at the same generation seeds.

| cell | raw hit rate | length-matched hit rate | Δ | mean content tokens | × unguided |
| --- | ---: | ---: | ---: | ---: | ---: |
| _unguided reference (HBD count)_ | 0.0893 | — | — | 34.15 | 1.00x |
| `c31_ksweep_hbd_count_deployed_L12_lam1_k2` | 0.3092 | 0.2322 | -0.0771 | 42.59 | 1.25x |
| `c31_ksweep_hbd_count_deployed_L12_lam1_k4` | 0.3837 | 0.3686 | -0.0151 | 39.06 | 1.14x |
| `c31_ksweep_hbd_count_deployed_L12_lam1_k8` | 0.3424 | 0.3489 | +0.0065 | 36.83 | 1.08x |
| `c31_ksweep_hbd_count_deployed_L12_lam1_k16` | 0.3496 | 0.3436 | -0.0060 | 35.35 | 1.03x |
| `c31_ksweep_hbd_count_deployed_L12_lam1_k32` | 0.3418 | 0.3380 | -0.0038 | 34.26 | 1.00x |
| `c31_ksweep_hbd_count_mid_L2_lam2_k2` | 0.4980 | 0.4794 | -0.0186 | 40.93 | 1.20x |
| `c31_ksweep_hbd_count_mid_L2_lam2_k4` | 0.5457 | 0.5380 | -0.0077 | 37.23 | 1.09x |
| `c31_ksweep_hbd_count_mid_L2_lam2_k8` | 0.5719 | 0.5649 | -0.0070 | 35.14 | 1.03x |
| `c31_ksweep_hbd_count_mid_L2_lam2_k16` | 0.5614 | 0.5503 | -0.0111 | 33.86 | 0.99x |
| `c31_ksweep_hbd_count_mid_L2_lam2_k32` | 0.5592 | 0.5465 | -0.0127 | 32.33 | 0.95x |
| _unguided reference (aromatic rings)_ | 0.1296 | — | — | 34.15 | 1.00x |
| `c31_ksweep_aromatic_rings_deployed_L12_lam1_k2` | 0.2064 | 0.2677 | +0.0613 | 42.79 | 1.25x |
| `c31_ksweep_aromatic_rings_deployed_L12_lam1_k4` | 0.3781 | 0.3997 | +0.0216 | 39.14 | 1.15x |
| `c31_ksweep_aromatic_rings_deployed_L12_lam1_k8` | 0.4234 | 0.4175 | -0.0058 | 36.55 | 1.07x |
| `c31_ksweep_aromatic_rings_deployed_L12_lam1_k16` | 0.4326 | 0.4331 | +0.0005 | 35.06 | 1.03x |
| `c31_ksweep_aromatic_rings_deployed_L12_lam1_k32` | 0.4410 | 0.4431 | +0.0021 | 34.24 | 1.00x |
| `c31_ksweep_aromatic_rings_mid_L2_lam2_k2` | 0.5430 | 0.5402 | -0.0028 | 41.16 | 1.21x |
| `c31_ksweep_aromatic_rings_mid_L2_lam2_k4` | 0.7451 | 0.7454 | +0.0003 | 37.08 | 1.09x |
| `c31_ksweep_aromatic_rings_mid_L2_lam2_k8` | 0.7451 | 0.7437 | -0.0014 | 34.77 | 1.02x |
| `c31_ksweep_aromatic_rings_mid_L2_lam2_k16` | 0.7438 | 0.7342 | -0.0096 | 33.42 | 0.98x |
| `c31_ksweep_aromatic_rings_mid_L2_lam2_k32` | 0.7271 | 0.7257 | -0.0014 | 32.87 | 0.96x |
| _unguided reference (QED)_ | 0.1186 | — | — | 34.15 | 1.00x |
| `c31_ksweep_qed_deployed_L12_lam1_k2` | 0.2435 | 0.2556 | +0.0121 | 42.93 | 1.26x |
| `c31_ksweep_qed_deployed_L12_lam1_k4` | 0.2376 | 0.2468 | +0.0091 | 39.17 | 1.15x |
| `c31_ksweep_qed_deployed_L12_lam1_k8` | 0.2129 | 0.2194 | +0.0065 | 36.58 | 1.07x |
| `c31_ksweep_qed_deployed_L12_lam1_k16` | 0.2332 | 0.2397 | +0.0065 | 35.03 | 1.03x |
| `c31_ksweep_qed_deployed_L12_lam1_k32` | 0.2249 | 0.2241 | -0.0008 | 34.19 | 1.00x |
| `c31_ksweep_qed_mid_L6_lam2_k2` | 0.3055 | 0.3013 | -0.0043 | 42.10 | 1.23x |
| `c31_ksweep_qed_mid_L6_lam2_k4` | 0.3112 | 0.3050 | -0.0062 | 39.24 | 1.15x |
| `c31_ksweep_qed_mid_L6_lam2_k8` | 0.3211 | 0.3111 | -0.0101 | 36.76 | 1.08x |
| `c31_ksweep_qed_mid_L6_lam2_k16` | 0.3033 | 0.3016 | -0.0016 | 35.50 | 1.04x |
| `c31_ksweep_qed_mid_L6_lam2_k32` | 0.2808 | 0.2810 | +0.0003 | 34.48 | 1.01x |

**The crossing survives length matching almost untouched.** The four largest crossing cells
move by at most 0.019: `hbd_count` mid k=2 goes 0.4980 → 0.4794, k=4 goes 0.5457 → 0.5380,
`aromatic_rings` mid k=2 goes 0.5430 → 0.5402 and k=4 goes 0.7451 → 0.7454. Their advantages
over the frontier are +0.1696 to +0.2473, so no crossing in this section is a length artefact.

**The one cell length matching does move is the deployed one, and it moves the wrong way for
the deployed story.** `hbd_count` deployed at k = 2 — the only deployed cell with a positive
raw advantage (+0.0317, interval already spanning zero) — falls from 0.3092 to **0.2322**
once reweighted to the unguided length distribution, a drop of 0.077, more than twice its
whole advantage. On the length-matched number that cell is clearly *below* the frontier.
This is a post-hoc control and does not change any pre-registered score, but it means the
deployed configuration's failure to replicate (§C31.5.2) is if anything understated.

**λ = 1 at the final probe point lengthens molecules; λ = 2 mid-network does not, as much.**
Content length runs 42.6–42.9 tokens at k = 2 on the deployed arm against 40.9–42.1 on the
mid arm and 34.2 unguided, and falls monotonically toward the unguided value as k rises on
every arm. That the *weakest* arm is the one that most inflates length is consistent with a
final-layer probe having little usable signal and the decoder responding by simply not
stopping.

---

## C31.7 The pre-registration's decision rules, scored as written

| # | rule | fires |
| --- | --- | :---: |
| **D1** | the crossing replicates: >= 1 cell crosses at k in {2,4} | **YES** |
| **D2** | the crossing is at the cheap end: every crossing cell has k <= 4 | no |
| **D3** | no compute knob: hit rate at k=32 minus k=2 is <= +0.02 on every arm | no |
| **D4** | the depth curve replicates: for both required properties the probe point maximising held-out TEST target AUROC is strictly less than 12 | **YES** |
| **D5** | the crossing does not replicate: no cell crosses at any k | no |
| **D6** | the deployed configuration replicates: the deployed arm (final probe point, lambda = 1) crosses at k = 2 | no |
| **D7** | the honesty rule: every cell whose between-generation-seed sd exceeds its own mean advantage is reported as NOT RESOLVED at three generation seeds | **YES** |

**Verdict: REPLICATES**, by C31.0.6's rule — REPLICATES iff D1; DOES NOT REPLICATE iff D5;
PARTIAL otherwise; UNINTERPRETABLE overrides. A test computes the verdict from the artefacts
and asserts the section's word matches, so the wording cannot drift from the rule.

**The verdict word is narrower than it sounds and the pre-registration is responsible for
that.** C31.0.6 defined REPLICATES on D1 alone — "≥ 1 cell crosses at k ∈ {2, 4}" — and that
is what is scored. It says nothing about D2, D3 or D6, all of which fail. A reader who takes
"REPLICATES" to mean "C28 and C30 transfer" would be wrong: what transfers is the existence
of a small-budget crossing, from a re-selected mid-network probe point at λ = 2. The verdict
rule was written before the result and is not rewritten now; the qualification is stated here
instead, where C31.0.10's reporting rule puts it.

**Uninterpretability was checked and did not fire.** All six blocking gates pass; 0 of
30 cells fall below the 0.80 crossing-validity floor, against a degeneracy threshold
of 25%; all three generation seeds completed in every cell.

---

## C31.8 The predictions, scored

| # | prediction | outcome |
| --- | --- | :---: |
| **P1** | Stage 0 unconditional RDKit validity >= 0.95 | **CONFIRMED** |
| **P2** | Stage 0 uniqueness >= 0.95 | **CONFIRMED** |
| **P3** | the depth curve peaks strictly before probe point 12 for both required properties (D4 fires) | **CONFIRMED** |
| **P4** | the selected mid probe point M is in [3, 8] for both required properties | **FALSIFIED** |
| **P5** | the frozen-state head beats the trivial prefix-statistics baseline on held-out test target AUROC by >= 0.02 at its best probe point, for both required properties | **FALSIFIED** |
| **P6** | D1 fires: at least one cell crosses at k in {2,4} | **CONFIRMED** |
| **P7** | D6 fires: the deployed arm crosses at k = 2 | **FALSIFIED** |
| **P8** | D3 fires: hit rate at k=32 minus k=2 is <= +0.02 on every arm | **FALSIFIED** |
| **P9** | qed does not cross at any k, on either arm | **CONFIRMED** |
| **P10** | guidance mean validity at k = 2 stays >= 0.95 on every arm | **CONFIRMED** |
| **P11** | the oracle-selected best-of-N curve is monotone non-decreasing in N on the measured grid for every property | **CONFIRMED** |

**Four of 11 falsified.** They are the interesting ones:

- **P4** and **P5** are the two that say something about *representations*. The depth peak
  moved earlier than any C17 value, and the frozen state's advantage over surface counting
  on aromatic rings vanished. Neither was expected; both are reported as failures rather
  than re-described.
- **P7** is the deployed-configuration prediction, and it was flagged in advance as the
  weaker of the two crossing predictions. It failed, and P6 — the one the pre-registration
  said was stronger — held.
- **P8** is D3 — *"guided decoding has no compute knob"* — and it is the failure with the
  widest consequences, because that claim is a C26 headline that C28 confirmed and that
  §C31.5.4 now shows does not hold on a second generator. It is falsified by a wide margin
  (+0.2347 against a +0.02 threshold), not marginally.

**Seven held**, including both of the two that the pre-registration flagged as load-bearing:
P6 (the crossing itself) and P11 (the comparator is monotone and therefore trustworthy).
P9 held too — `qed` crosses nowhere, on either arm, at any k, exactly as on GP-MoLFormer.

---

## C31.9 What C31 changes elsewhere

Conflicts for the owner to merge. `reports/pilot_report.md` §22 and §23 are the merge point;
C31 edits no existing section, config, report or `outputs/` directory.

1. **§22.2's strongest sentence does not survive a second generator, and it is the sentence
   the abstract leans on.** §22.2 says of the deployed cell: *"A1 at k = 2 is the deployed
   configuration — not a mid-network arm, not a re-tuned λ, the exact setting §16 reports —
   which is the one cell in this table that cannot be dismissed as a post-hoc pick."* On
   `gpt2_zinc_87m` that configuration does not cross on any of the three properties, and on
   aromatic rings it is decisively below the frontier. The claim that survives both
   generators is the **mid-network, λ = 2, small-k** one — which is precisely the family
   §22.2 was hedging against. §22.2 and `ABSTRACT.md` should be rewritten around it, with
   the deployed cell's crossing scoped to GP-MoLFormer.
2. **§23.7's "Settles" paragraph should gain a scope line.** "The crossing is not a
   probe-seed artefact" stands. "Including the deployed configuration, whose margin grows"
   is a GP-MoLFormer statement and should say so.
3. **C26's "guided decoding has no compute knob" must be scoped to GP-MoLFormer.** This is
   the largest conflict in the list. C26 states it as a property of the method and C28
   confirmed it by sweeping the only compute knob the method has (−0.0218 over the full k
   span). On `gpt2_zinc_87m` the same sweep gives up to **+0.2347** on four of six arms.
   What survives on both generators is the *useful* half of the claim — **k is a bad knob,
   because best-of-N converts the same tokens into far more accuracy** — and that is what
   §22 and the abstract should say instead.
4. **The "one generator" limitation, which every section from C17 onward carries, is
   partially discharged and should be reworded rather than deleted.** The crossing and the
   depth-curve shape now hold on two generators; the deployed configuration's crossing, the
   no-compute-knob result, and the k ≤ 4 confinement hold on one.
5. **C17's "every property is predicted best mid-network" gains a third architecture and
   loses a number.** The *shape* replicates on `gpt2_zinc_87m`; the *depth* does not — the
   peak is at probe point 2, earlier than C17's 3–5 range. §21's statement of the finding
   should be about the shape.
6. **C17's aromatic-rings margin over `trivial` (+0.0205) does not transfer.** On this
   generator it is +0.0003. Any statement that the frozen state carries information cheap
   prefix statistics do not should be scoped to the generator and the property.
7. **C24's `generality.py` is no longer text-only.** C31 reuses `repeat_cache_gpt2` and
   `train_head_on_device` from it. `tests/test_generality.py::
   test_no_molecular_module_imports_the_generality_module` was rewritten to check the
   **import graph** rather than raw text, with an explicit C31 allowlist, and a companion
   test `test_c31_never_loads_the_molecular_generator` was added so the separation guarantee
   that test protects is re-established from the other side. This is the only existing test
   C31 touched.
8. **`configs/c31_second_generator.yaml` is new and additive.** No existing config changed.
9. **Nothing here re-measures C26's frontier, C27's oracle result, C29's head-seed variance
   or C30's replication.** C31 builds its own frontier on its own generator and compares
   nothing across generators except qualitatively.

---

## C31.10 Limitations

1. **Three generation seeds, one probe seed per cell.** C29 measured a probe-seed sd of
   0.0142–0.0366 on GP-MoLFormer and C30 showed it matters. C31's cells were each generated
   from **one** head seed (1234), which is exactly the protocol C29's finding says is
   inadequate. Against C29's largest measured probe-seed sd (0.0366), the four largest
   crossing margins are 4.6× to 6.8× it, so their *sign* is unlikely to be seed noise — but
   their *sizes* are not resolved at one probe seed and C31 does not claim them to be. **The
   fifth crossing cell, `aromatic_rings` mid at k = 8 (+0.0595), is only 1.6× that sd and
   should be treated as unresolved**, which matters because it is the single cell that makes
   D2 fail. Three head seeds were trained per probe point and their AUROC spread is reported
   (sd ≤ 0.0041), but their **end-to-end** spread was not measured. Re-running the five
   crossing cells at three or more probe seeds — the C30 analogue — is the first thing to do
   next, and C31's budget did not reach it.
2. **The deployed-arm negative is a single-configuration result too.** "The deployed
   configuration does not cross" rests on the same one-probe-seed protocol as the positives,
   and deserves the same discount.
3. **The aromatic-rings crossing is not evidence about representations** (§C31.3): the
   `trivial` baseline ties the best probe point there. It is evidence about *steering a
   predictor through a decoder*, which is a weaker and different claim.
4. **The length confound is measured post hoc, not designed out.** §C31.6 is not
   pre-registered, and C31 ran no truncation control and no `unguided` / windowed conditions
   — the six-condition panel `configs/guidance.yaml` defines was reduced to `throughout`
   alone, following C28.
5. **λ ∈ {1, 2} only, and λ is confounded with probe point.** The deployed arm is (12, λ=1)
   and the mid arm is (M, λ=2); C31 never ran (12, λ=2) or (M, λ=1), so it cannot separate
   "mid-network helps" from "λ=2 helps". C23 separated these on GP-MoLFormer and found both
   matter; that separation is **not** replicated here, and every statement in §C31.5.2 about
   *why* the deployed arm fails is therefore under-determined. This is the largest design
   weakness in C31.
6. **The comparator is C31's own measured curve, linearly interpolated.** Linear
   interpolation on a concave curve *underestimates* best-of-N, which biases every advantage
   **upward** — the same conservative-in-the-wrong-direction caveat C26 §7.3 records,
   inherited here and running in the headline's favour.
7. **Two generators is two.** They share an architecture family (decoder-only transformer),
   a serialization (SMILES), a corpus family (ZINC-like drug-like space) and a training
   objective. C31 shows the crossing is not unique to GP-MoLFormer; it does not show it is
   general.
8. **The batch-size deviation** (§C31.1.1) means no C31 cell was run at the pre-registered
   batch size, though the evidence that this is neutral is published.
9. **No wall-clock claim is made anywhere.** Cost is processed generator tokens throughout,
   under both accounting rules.

---

## REPRODUCE

```bash
# C31.0  the pre-registration must already be on disk and must not be edited.
sha256sum outputs/c31_prereg/C31.0_preregistration.md   # must match prereg_lock.json

# Stage 0  feasibility: G0 (validity floor) and G1 (cached vs full states, 13 probe points).
#          Stops the experiment if validity < 0.80.  ~1 min.
.venv/bin/python scripts/25_c31_second_generator.py --stage feasibility

# Stage 1  50k trajectories, prefixes, states at all 13 probe points, frozen intervals and
#          windows, G5 and G6.  ~4 min generation + ~1 min states.  Writes ~8 GB of .npy.
.venv/bin/python scripts/25_c31_second_generator.py --stage dataset

# Stage 2  117 frozen-state heads + 9 trivial heads; selects the mid probe point by the
#          C31.0.4 VALIDATION-AUROC rule.  ~8 min.
.venv/bin/python scripts/25_c31_second_generator.py --stage heads

# G2       the decision-equality gate.  Needs the heads.  ~3 min.
.venv/bin/python scripts/25_c31_second_generator.py --stage decision-gate

# Stage 3  the oracle-selected frontier.  ~10 min.
.venv/bin/python scripts/25_c31_second_generator.py --stage bestofn

# Stage 4  the k sweep, 30 cells, idempotent per cell so a kill costs at most one cell.
#          batch 16 -- see C31.1.1.  ~50 min on an RTX 4090.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup .venv/bin/python scripts/25_c31_second_generator.py --stage ksweep \
  > outputs/c31_run.log 2>&1 &

# G3       end-to-end backend equivalence, and the POST HOC length control.
.venv/bin/python scripts/25_c31_second_generator.py --stage backend-gate
.venv/bin/python scripts/25_c31_second_generator.py --stage length-control

# score the pre-registration.  Reads existing artefacts only; generates nothing.
.venv/bin/python scripts/25_summarise_c31.py

# binding tests
.venv/bin/python -m pytest tests/test_second_generator.py -q -p no:cacheprovider
```

**Artefact sizes.** `outputs/c31_layer_states/` is 13 × 585 MB of frozen hidden states and is
`.gitignore`d (`outputs/**/*.npy`), as are the 126 head checkpoints
(`outputs/**/*.pt`). **No test loads a `.pt` or `.npy` file**, so no `!` negation was added to
`.gitignore`. Everything the section prints is re-derivable from
`outputs/c31_summary/c31_metrics.json`, which is tracked and which
`tests/test_second_generator.py` reads.

**Pins.** Recorded by `write_run_context` beside every output directory: the
`entropy/gpt2_zinc_87m` revision SHA, the parameter-sum fingerprint, torch, transformers,
numpy, rdkit, float32, device `cuda`, generator in eval mode with `requires_grad = False` and
never modified. Cost is `processed_tokens_actual`, with `processed_tokens_full_recompute`
reported alongside; never wall-clock.
