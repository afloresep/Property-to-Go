# Section C24 — the generality (external-validity) experiment

> ## ⚠ Read this box before any number below
>
> **C24 uses a non-molecular generator (GPT-2 small, 124M) and is an external-validity
> check only. It is never part of the main molecular result.** No number in this section
> belongs in `reports/pilot_report.md`'s results chain, and no number in that chain moves
> because of C24. GP-MoLFormer is not loaded anywhere in C24; no molecular `outputs/`
> directory is written; the molecular pipeline is read only for the published numbers
> quoted here for comparison.
>
> The specification's **"no second generator" rule is about the molecular experiment** —
> it forbids comparing or ensembling molecular generators so that the negative result
> cannot be blamed on a generator choice. A different-*domain* generator used solely to
> ask whether a stated claim travels is outside that rule, and the owner explicitly asked
> for this experiment. The boundary is drawn here so that it stays visible.

Draft section, written to be merged into `reports/pilot_report.md` as a clearly separated
external-validity appendix. Nothing in this file edits an existing report claim; where the
data bears on one it is flagged in §C24.12 and left for the owner to merge.

C24 asks whether two of the project's most portable findings are facts about FUDGE-style
guided decoding **in general** or facts about **GP-MoLFormer and molecules**:

* **Claim 1** — post-hoc calibration of the probe is algebraically a rescale of λ, so
  calibrating it does not help (`reports/pilot_report.md` §20.3).
* **Claim 2** — the probe's best layer is mid-network, and the layer that predicts best is
  not the layer that steers best; the per-position proxy and the end-to-end measurement
  diverge (§21.5 and `reports/section_c23_layer_end_to_end.md`).

**Headline, stated before the detail and not softened.** Claim 1's *algebraic core*
replicates **exactly** — at ε = 0 a power-calibrated head at λ=1 and the raw head at λ=α
return the **same 1536 sequences**, for all three attributes — but the pre-registered
verdict rule for Claim 1 returns **FAILS TO REPLICATE**, because prediction 1a (every
Platt slope below 1) is false on `mean_word_length` (slope **1.6154**). Claim 2's
*prediction* half replicates strongly (the depth peak is probe point 2 for all three
attributes, before the final layer, with a monotone decline after it), and its
*divergence* half does **not**: the cell is (**NOT MATERIAL**, **NOT POSITIVE**), i.e.
**NO DIVERGENCE** — here the cheap per-position proxy and the expensive end-to-end
measurement agree, which is the opposite of the molecular finding.

---

## C24.0 Pre-registration

## C24.0.0 Why this experiment exists, and what is out of scope

The project's two most portable findings are currently stated from **one model, one
guidance rule, one domain**:

* **Claim 1 — post-hoc calibration of the probe is algebraically a rescale of λ, so
  calibrating it hurts.** (`reports/pilot_report.md` §20.)
* **Claim 2 — the probe's best layer is mid-network, and the layer that predicts best
  is not the layer that steers best; the per-position proxy and the end-to-end
  measurement diverge.** (§21 and `reports/section_c23_layer_end_to_end.md`.)

C24 asks whether these are facts about FUDGE-style guided decoding in general or facts
about GP-MoLFormer and molecules. It builds a **second, non-molecular** instance of the
same pipeline and re-runs the two measurements on it.

**Scope, stated explicitly because it looks like a rule violation and is not.** The
specification's "no second generator" rule is about the *molecular* experiment: it
forbids comparing or ensembling molecular generators, so that the negative result cannot
be attributed to a generator choice. A different-*domain* generator used only for an
external-validity check is outside that rule. C24 is therefore reported as a clearly
separated **external-validity check**, never as part of the main molecular result.

**C24 does not touch the molecular pipeline.** GP-MoLFormer is not loaded anywhere in
C24. No existing `outputs/` directory is read for anything except the published numbers
quoted for comparison, and none is written. No number in `reports/pilot_report.md` can
move because of C24.

---

## C24.0.1 The decoding rule, identical in both domains

At a guided step, over the base model's top-`k` candidate next tokens `a`:

    score(a) = log p_base(a | prefix) + λ · log( q(a) + ε )
    next token ~ softmax over the k candidates of score(a)

where `q(a) = P(final attribute of the completed sequence ∈ target interval | prefix + a)`
is a probe's estimate, read from a frozen hidden state at one probe point.

The combination is computed by `property_to_go.guidance.combine_scores` — **the molecular
function, imported, not re-derived** — so a difference between the two domains cannot be
a difference in the rule. `TargetScorer`, `CalibratedTargetScorer`, the Platt/isotonic/
power fitters, `binning`, `metrics`, `headroom.candidate_weights` /
`headroom.guided_weights`, `bestofn.selection_key` / `target_error`, `ComputeMeter` and
`solve_best_of_n` are all shared with the molecular library for the same reason. What is
re-implemented is the generator loader, the sampler and the KV-cache plumbing (GPT-2's
cache is `(key, value)` per layer; `generation.repeat_cache` is written for Molformer's
linear-attention running-sum cache and does not apply), and a device-aware copy of
`heads.train_head` which is asserted bit-identical to the original on CPU.

---

## C24.0.2 Substrate, and why this one

**Preference 1 of the brief: a real pretrained text LM with a cheap, deterministic,
computable attribute.** Network access was checked first and works, and **GPT-2 small is
already in the local HF cache**, so the download risk that would have forced preference 2
does not arise.

    generator   gpt2 (124M), revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
    tokenizer   gpt2, same revision
    dtype       float32, device cuda, frozen, eval mode, no weight is ever modified

Chosen over a synthetic formal language because it is a *second real domain* rather than
a synthetic replication, and over larger LMs because 12 transformer blocks / 13 probe
points is a **structural match to GP-MoLFormer's 12 layers / 13 probe points**, which
makes the depth curves directly comparable rather than merely analogous.

**Sequences are fixed-length: `[<|endoftext|>] + 40 sampled tokens`, temperature 1.0,
full-vocabulary multinomial sampling.** GPT-2 does not emit a natural terminator often
enough for an end-of-sequence rule to give a clean "completed sequence", and a truncation
boundary would hand the guided decoder a degenerate lever (steer towards being cut off).
The cost is that the sequence-length confound the molecular pipeline standardises for
(§9) does not exist here. That is a **design difference and is recorded as one**, not
hidden: it makes C24 a cleaner test of the two claims and a weaker test of everything the
molecular pipeline says about length.

### Attributes

Exact functions of the completed decoded text, computable in microseconds — which is what
makes compute-matched best-of-N available, since the baseline gets ground truth exactly as
RDKit gives it ground truth in the molecular pipeline.

| attribute | definition | kind |
| --- | --- | --- |
| `digit_count` | number of digit characters | count |
| `upper_count` | number of upper-case characters | count |
| `mean_word_length` | mean length of whitespace-delimited words | continuous |

`digit_count` and `upper_count` are declared integer-valued, so `target_error` measures to
the nearest *attainable* in-target value (the `docs/HANDOFF.md` §4 boundary bug).

A `trivial` head over eight cheap prefix statistics (length, character count, digits so
far, upper-case so far, spaces, words, mean word length, punctuation) is trained once per
attribute as the surface-statistics baseline, mirroring `tokens.FEATURE_NAMES`.

---

## C24.0.3 The target-interval rule, fixed here

For each attribute, over all bands `[q_a, q_b)` with `a < b` drawn from the fixed grid
`{0.00, 0.05, …, 1.00}` of the **frozen base sample**, take the band whose realised base
rate is **closest to 0.10**; ties broken by smallest `a`, then smallest `b`. Integer
attributes have both edges rounded to the nearest integer first, so the band is a union
of `CategoricalBinner` bins by construction (the `docs/HANDOFF.md` §3.6 invariant, which
`binning.interval_mask_coverage` re-checks).

This is the molecular pipeline's own continuous rule (`quantile_band [0.85, 0.95)`)
generalised so that it can also serve heavy-tailed counts. The molecular *count* rule
(`[v, v+1)` at `v = round(q_0.90)`) is **not** used, because on this substrate the count
distributions are heavy-tailed and it returns base rates below 0.02 — outside the
0.05–0.20 regime the brief requires. That substitution is made here, before any interval
is resolved, and not after seeing one.

**Admission gate.** An attribute whose realised base rate falls outside **[0.05, 0.20]**
is dropped from the battery and the drop is recorded. If fewer than two attributes
survive, C24 is reported as not executable on this substrate.

Binning: `CategoricalBinner` for counts, capped at `ceil(q_0.995)` of the base sample and
never below `hi`; `QuantileBinner` with 20 bins for `mean_word_length`, fitted with
`extra_edges=(lo, hi)`.

---

## C24.0.4 The dataset, splits, and everything numeric that is fixed now

| | |
| --- | --- |
| base sequences | 20,000, seed 20240001 |
| content tokens per sequence | 40 |
| prefixes per sequence | 4, one drawn uniformly from each quartile of positions [1,39], prefix seed 12 |
| splits | 0.8 / 0.1 / 0.1, **grouped by completed text** via `splits.split_by_group`, split seed 11 |
| head | 2-layer MLP, hidden 256, dropout 0.1, AdamW lr 1e-3, wd 0.01, batch 512, ≤60 epochs, patience 8 — `configs/pilot_50k.yaml` unchanged |
| head seeds | 1234 / 2345 / 3456 |
| probe points | all 13 (`hidden_states[0..12]`; 0 is the embedding output, a control, counted in the multiplicity correction rather than excluded) |
| guided decoding | top-k 8, λ = 1.0, ε = 1e-6, `throughout` |
| generation seeds | 101 / 202 / 303 |
| sequences per condition per seed | 512 |
| per-position rollouts | 300 test-split prefixes × 8 candidates × 32 base-policy rollouts, rollout seed 7777 |
| best-of-N | N solved from each arm's own `processed_tokens_actual` per returned sequence; the **realised** token ratio is written to the artefact |

Cost is reported in **processed tokens**. No wall-clock claim is made
(`pilot_report.md` §11.7).

---

## C24.0.5 The feasibility probe that was run *before* this file, declared in full

Two scratch scripts were run before this pre-registration and neither wrote anything
under `outputs/`. They are declared because they are measurements and because concealing
them would be the exact degree of freedom a pre-registration exists to remove.

1. **A distribution probe.** 1,024 (then 2,048) base samples, to confirm the substrate
   loads offline at the pinned revision and that *some* attribute admits a band with base
   rate in 0.05–0.20. It looked at attribute histograms only. No probe, no head, no
   guided decoding.
2. **A narrow predictability check.** 3,000 sequences, **one** attribute
   (`digit_count`), **one** probe point (**12, the final layer — the molecular
   default**), **one** head seed. Verbatim output:

   ```
   band {'lo': 4.0, 'hi': 6.0, 'q_lo': 0.75, 'q_hi': 0.85, 'base_rate': 0.09933}
   FEASIBILITY: n_test=1132 base_rate=0.0777 target_AUROC=0.6531 ECE=0.0215
                mean_q=0.0992 epochs=2 wall=19s
   ```

   Its purpose was to establish that the probe is not at chance, because a substrate on
   which nothing is predictable makes both claims untestable rather than false.
   **It touched no other layer, no calibrator, and no end-to-end or per-position
   steering quantity, so nothing bearing on Claim 1 or Claim 2 was seen.** The C24
   dataset is a fresh 20,000-sequence sample at a different seed; the numbers above are
   not reused.

---

## C24.0.6 Claim 1 — what is predicted, and what would falsify it

Calibrators are fitted on **held-out off-policy prefixes** — prefixes of guided (λ=1,
`throughout`, probe point 12) sequences, split in half by completed text so no sequence's
prefixes straddle the fit and score halves.

| # | prediction | falsified if |
| --- | --- | --- |
| **1a** | every fitted Platt slope `α < 1` | any `α ≥ 1` |
| **1b** | target AUROC is **unchanged** by Platt to 4 dp, for every attribute (a strictly monotone map cannot move a rank statistic) | any \|ΔAUROC\| > 1e-4 |
| **1c** | ECE falls by a factor ≥ 2 under Platt and under isotonic, for every attribute — i.e. the calibrators genuinely work | any factor < 2 |
| **1d** | **the identity.** At **ε = 0**, a power-calibrated head `g(q)=q^α` at λ=1 and the raw head at λ=α return the **same sequences**: identical fraction **1.000**, hit-rate difference **0.0**, at every attribute and every seed | any identical fraction < 1.000, or any hit-rate difference ≠ 0 |
| **1e** | **end to end**, Platt and isotonic arms have lift **< 1.00×** the uncalibrated arm's, at every attribute | any calibrated arm ≥ 1.00× |

**Verdict rule for Claim 1.** *REPLICATES* iff 1a, 1b and 1d hold exactly and 1e holds at
**≥ 2 of 3** attributes for **both** Platt and isotonic. *PARTIALLY REPLICATES* iff 1a,
1b, 1d hold but 1e holds at fewer than 2 of 3. *FAILS TO REPLICATE* iff 1a, 1b or 1d
fails. **1d failing is the most informative outcome available and is to be reported as
the headline if it happens**, because the identity is algebraic: if two implementations
of the same rule disagree about it, one of them is not implementing the rule.

---

## C24.0.7 Claim 2 — what is predicted, and what would falsify it

Let `A(L)` be the mean held-out target-interval AUROC at probe point `L` over head seeds
1234/2345/3456, and `L* = argmax_L A(L)` **chosen by prediction alone**, before any
steering quantity is computed.

| # | prediction | falsified if |
| --- | --- | --- |
| **2a** | the depth curve peaks **strictly before** probe point 12, in the first half of the stack (`L* ∈ 1..6`), for ≥ 2 of 3 attributes, and no attribute peaks at 12 | fewer than 2 of 3 peak in 1..6, or any attribute peaks at 12 |
| **2b** | probe point 12 is the **minimum over probe points `L*`..12** for every attribute (monotone decline after the peak) | any attribute has an interior minimum below probe point 12 |
| **2c** | **per position**, swapping to `L*` is **NOT MATERIAL** — it improves `our_head_gain` for fewer than 2 of 3 attributes, **or** the median relative improvement is < +0.25 | it improves ≥ 2 of 3 **and** median relative ≥ +0.25 |
| **2d** | **end to end**, guided generation at `L*` beats guided generation at probe point 12 — mean lift ratio > 1.00 and the seed-paired bootstrap CI on the hit-rate difference excludes 0 — for ≥ 2 of 3 attributes | fewer than 2 of 3 |

**2e — the divergence, which is the actual claim under test.** The molecular result is
the *pair* (2c NOT MATERIAL, 2d POSITIVE). Four cells are named in advance:

| per position (2c) | end to end (2d) | verdict |
| --- | --- | --- |
| NOT MATERIAL | POSITIVE | **DIVERGENCE REPLICATES** — the molecular pattern |
| MATERIAL | POSITIVE | *no divergence*: the proxy was informative after all. Claim 2's methodological half **fails to replicate** |
| NOT MATERIAL | NOT POSITIVE | *no divergence*: both say the layer does not help. Claim 2's methodological half **fails to replicate**, and in the direction that would have made the cheap proxy adequate |
| MATERIAL | NOT POSITIVE | **divergence with the opposite sign** — reportable, and a *different* result from the molecular one |

A per-position number is never to be quoted as if it were end to end, in either
direction. Both are reported for every attribute with their signs compared explicitly.

**Multiplicity.** 13 probe points × 3 attributes = 39 comparisons. Within one attribute
the 13 probe points are one family: a "probe point `L` beats probe point 12" claim must
clear a Bonferroni-corrected paired bootstrap at `1 − 0.05/13`. **No isolated spikes:**
`A(L) − A(12) ≥ 0.010` and both neighbours `≥ 0.005` (one neighbour at the ends). Per-seed
values are published for every cell so the noise floor is legible rather than asserted.

---

## C24.0.8 Compute matching

Best-of-N is matched on **processed tokens**, never on forward calls or returned
sequences. `N` is solved from each guided arm's own `processed_tokens_actual` per returned
sequence against the base policy's, and the **realised** ratio
`(N × base tokens) / (guided tokens)` is written to the artefact rather than assumed. Both
accountings (`actual` and `full_recompute`) are recorded. Because every C24 sequence is
exactly 41 tokens, the base cost per sequence is a constant and the realised ratio is
expected to be exactly `N × 41 / guided_tokens_per_sequence`; that expectation is checked
against the measured value, not substituted for it.

---

## C24.0.9 What C24 will NOT do

Stated now so that its absence later is not read as a silent omission:

* no λ sweep (§19 has the molecular one; C24 runs λ = 1 only, plus λ = α for the identity);
* no bin-logit-temperature arm and no retrained-readout arm (§20 routes (a)-temperature
  and (b)); C24 tests the *calibration* half of Claim 1 and the *depth* half of Claim 2;
* no chemical-quality analogue beyond uniqueness and descriptive text statistics;
* no DAgger, no fine-tuning, no activation steering — the generator is frozen throughout;
* no window sweep: `throughout` and `unguided` only, as in §20.5 and C23.

---

## C24.0.10 What would make C24 uninterpretable rather than negative

Recorded so that a null is not passed off as a finding:

* fewer than two attributes surviving the C24.0.3 base-rate gate;
* a probe at every layer at chance (max `A(L) < 0.55` for every attribute), which makes
  both claims untestable on this substrate;
* an uncalibrated guided arm whose end-to-end lift over `unguided` is not positive at any
  attribute — with no steering effect at all there is nothing for calibration or depth to
  modulate, and Claims 1e and 2d become vacuous. This would be reported as "the substrate
  does not steer", not as "the claims fail".

---


---

*(Everything above this line was on disk before the first C24 measurement ran; it is
`outputs/c24_prereg/prereg.md`, copied byte for byte and checked by
`tests/test_generality.py::test_the_report_copies_the_prereg_verbatim`. Everything below
was written after the run.)*

---

## C24.1 What was run

| stage | script | writes | cost (processed tokens) |
| --- | --- | --- | ---: |
| base sample, splits, target intervals, binners | `scripts/19_c24_dataset.py` | `outputs/c24_dataset/` | 1,620,000 |
| heads at all 13 probe points × 3 head seeds, plus `trivial` | `scripts/19_c24_heads.py` | `outputs/c24_probe_layers/` | no generation (states reused from the dataset stage) |
| off-policy calibrator fit, ECE/AUROC, the numeric identity | `scripts/19_c24_calibrate.py` | `outputs/c24_calibration/` | 2,420,748 |
| per-position steering proxy at all 13 probe points | `scripts/19_c24_steering.py` | `outputs/c24_layer_steering/` | 3,082,172 |
| end-to-end guided arms + compute-matched best-of-N | `scripts/19_c24_endtoend.py` | `outputs/c24_endtoend/` | 13,639,680 |
| cached-decode validity gate at all 13 probe points | `scripts/19_c24_gate.py` | `outputs/c24_gate/` | (128 sequences, 5 positions) |
| score every pre-registered decision rule | `scripts/19_c24_summarise.py` | `outputs/c24_summary/c24_summary.json` | reads artefacts only |

**Total 20,762,600 processed tokens.** Cost is reported in processed tokens only; no
wall-clock claim is made anywhere in this section (§11.7).

The library is `src/property_to_go/generality.py`. The decoding rule is
`guidance.combine_scores` — the molecular function, **imported, not re-derived**
(`tests/test_generality.py::test_the_decoding_rule_is_the_molecular_function`), and so are
`TargetScorer`, `CalibratedTargetScorer`, the Platt/isotonic/power fitters, `binning`,
`bestofn.selection_key`/`target_error`, `ComputeMeter` and `solve_best_of_n`. Re-implemented
are only the generator loader, the sampler, GPT-2's `(key, value)` KV-cache plumbing and a
device-aware copy of `heads.train_head`, which is asserted **bit-identical** to the original
on CPU by `test_the_device_trainer_reproduces_the_molecular_trainer_on_cpu`.

Two directional separation tests are enforced: `test_c24_never_loads_the_molecular_generator`
(no C24 file may mention `load_generator` or `GP-MoLFormer-Uniq`) and
`test_no_molecular_module_imports_the_generality_module`.

### C24.1.1 The arm that was missing, and the reading it invalidated

The first C24 run had no λ=0 arm. Every guided arm samples from GPT-2's **top-8** truncated
candidate set; `unguided` samples the full vocabulary. So `guided − unguided` confounds the
property term with the truncation, and the first draft's reading of the digit-count and
mean-word-length arms was wrong because of it. **`truncation_control`** — λ=0, top-k=8,
identical seeds (101/202/303), identical 512 sequences per condition, identical code path,
the property term switched off — was added to `scripts/19_c24_endtoend.py` **additively**
and is the primary reference for every guided contrast below. The `guided − unguided`
contrast is reported alongside, and §C24.7 says plainly which one licenses which claim.

Additivity was proved rather than asserted: the 54 pre-existing `arm_*.json` and
`texts_*.json` files in `outputs/c24_endtoend/` are **byte-identical** (SHA-256) before and
after the re-run; only the aggregate `endtoend_metrics.json` changed, and it changed only by
gaining the `truncation_control` arm and its best-of-1 entry. `run_arm` is idempotent, so no
existing arm was recomputed.

## C24.2 Validity gate — the cached decode path against full-prefix recomputation

The two-way token accounting is only meaningful if evaluating a candidate from a shared
cache is the *same computation* as re-running its whole prefix. Checked at **all 13 probe
points**, on 128 real C24 sequences, at positions 5/12/20/28/36.

| quantity | value |
| --- | ---: |
| max abs difference over all 13 probe points | `1.526e-03` |
| hidden-state max abs value (the scale) | 315.95 |
| relative to state scale | `4.830e-06` |
| bit-identical | **no** |
| within the pre-set tolerance 2e-3 | **yes** |

Unlike the molecular linear-attention backends (`docs/HANDOFF.md` §1.2), the two GPT-2 paths
are **not** bit-identical: standard attention reduces over the key dimension in a different
order when the prefix comes from a cache. The residual is reported, not asserted away. It is
concentrated at the top of the stack (0.0 at probe point 0, `1.526e-03` at probe point 12).

`outputs/c24_gate/gate.json` also fingerprints the generator: 124,439,808 parameters, 12
layers, hidden size 768, `gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` —
the same fingerprint recorded by the dataset stage, so all stages used one frozen model.

## C24.3 Substrate and targets

20,000 base sequences (seed 20240001), 40 content tokens each after `<|endoftext|>`,
temperature 1.0, full-vocabulary multinomial sampling; **19,991 of 20,000 texts unique**.
Grouped splits by completed text: 16,013 / 1,960 / 2,027 sequences → 64,052 / 7,840 / 8,108
prefix rows.

Target intervals were resolved by the C24.0.3 rule from the frozen base sample, before any
head was trained. All three attributes cleared the [0.05, 0.20] admission gate, so none was
dropped and C24 is executable on this substrate (C24.0.10, first bullet).

| attribute | interval `[lo, hi)` | quantile band | realised base rate |
| --- | --- | --- | ---: |
| `digit_count` | [7, 13) | [0.85, 0.95) | 0.0958 |
| `upper_count` | [14, 18) | [0.80, 0.90) | 0.0973 |
| `mean_word_length` | [5.5926, 5.9286) | [0.75, 0.85) | 0.1000 |

Every interval is an exact union of `CategoricalBinner`/`QuantileBinner` bins
(`interval_mask_coverage.is_exact` true for all three —
`test_the_target_interval_is_a_union_of_bins`), so the `docs/HANDOFF.md` §3.6 invariant
holds and the §11 interval-mask defect cannot recur here.

## C24.4 Claim 2, prediction half — the depth curve

Mean held-out target-interval AUROC over head seeds 1234/2345/3456. Bold = the argmax,
chosen by prediction alone before any steering quantity was computed.

| probe point | `digit_count` | `upper_count` | `mean_word_length` |
| ---: | ---: | ---: | ---: |
| 0 | 0.6294 | 0.6358 | 0.5708 |
| 1 | 0.7708 | 0.7546 | 0.6787 |
| 2 | **0.7979** | **0.7686** | **0.6857** |
| 3 | 0.7933 | 0.7549 | 0.6714 |
| 4 | 0.7939 | 0.7557 | 0.6656 |
| 5 | 0.7938 | 0.7484 | 0.6609 |
| 6 | 0.7940 | 0.7463 | 0.6614 |
| 7 | 0.7943 | 0.7443 | 0.6569 |
| 8 | 0.7866 | 0.7412 | 0.6589 |
| 9 | 0.7829 | 0.7419 | 0.6594 |
| 10 | 0.7791 | 0.7405 | 0.6528 |
| 11 | 0.7748 | 0.7362 | 0.6508 |
| 12 | 0.7693 | 0.7336 | 0.6478 |
| `trivial` | 0.8730 | 0.8747 | 0.7941 |

| attribute | `L*` | AUROC at `L*` | AUROC at 12 | gain | Bonferroni CI (1 − 0.05/13) | excludes 0 | neighbours `L*−1` / `L*+1` | no isolated spike | `trivial` | margin over `trivial` |
| --- | ---: | ---: | ---: | ---: | --- | :---: | ---: | :---: | ---: | ---: |
| `digit_count` | 2 | 0.7979 | 0.7693 | +0.0286 | +0.0315 [+0.0108, +0.0517] | true | +0.0015 / +0.0240 | false | 0.8730 | -0.0750 |
| `upper_count` | 2 | 0.7686 | 0.7336 | +0.0351 | +0.0361 [+0.0176, +0.0567] | true | +0.0210 / +0.0214 | true | 0.8747 | -0.1061 |
| `mean_word_length` | 2 | 0.6857 | 0.6478 | +0.0380 | +0.0415 [+0.0161, +0.0666] | true | +0.0309 / +0.0236 | true | 0.7941 | -0.1084 |

**This is a clean replication of the molecular depth curve's shape.** For all three
attributes the best probe point is **2** — strictly before the final layer, in the first
half of the stack — the decline from `L*` to 12 is monotone (probe point 12 is the minimum
over `L*..12` for every attribute), and every best-minus-final gain clears the
Bonferroni-corrected paired bootstrap. The molecular curve peaked at probe points 3/4/4/5/5/4
of 13 (§21.3); the text curve peaks one to three points shallower, on a structurally matched
12-block / 13-probe-point stack.

**Two things that must be said against it.**

1. **`no_isolated_spike` is `false` for `digit_count`.** The pre-registered anti-spike guard
   requires both neighbours to be at least +0.005 above probe point 12; probe point 1 is only
   **+0.0015** above it. The peak at 2 for `digit_count` is therefore a *shoulder on a step*,
   not a broad plateau, and the pre-registration's own multiplicity guard declines to certify
   it. The other two attributes pass the guard.
2. **The `trivial` surface-statistics baseline beats the probe at every probe point, for
   every attribute** — margins **-0.0750**, **-0.1061**, **-0.1084** against the best layer.
   Eight cheap prefix statistics predict the finished text's attribute better than a 768-dim
   GPT-2 hidden state does. This is a stronger version of the molecular §21.3 finding
   (`trivial` 0.8269 against probe point 3's 0.8474 and probe point 12's 0.7878 for aromatic
   rings — there the probe beat `trivial` at the peak and lost to it at the final layer;
   here it loses everywhere). It does not affect Claim 2, which is a statement about the *shape* of the depth
   curve and about proxy-vs-end-to-end agreement, but it does mean the C24 probe is a weak
   probe in absolute terms, and §C24.11 treats that as a limit on how far the replication
   travels.

## C24.5 Claim 1 — calibration, fitted off-policy

Calibrators fitted on held-out off-policy prefixes (prefixes of λ=1 `throughout` probe-point-12
sequences), split in half by completed text so no sequence's prefixes straddle the fit and
score halves.

| attribute | ECE uncal. | ECE Platt | ECE isotonic | ECE factor Platt | ECE factor isotonic | Platt slope α | ΔAUROC Platt | ΔAUROC isotonic | off-policy factor | on-policy factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `digit_count` | 0.0191 | 0.0129 | 0.0100 | 1.48x | 1.90x | 0.8880 | 0.000000 | +0.000749 | 0.9768 | 0.9743 |
| `upper_count` | 0.0423 | 0.0156 | 0.0151 | 2.71x | 2.80x | 0.7791 | 0.000000 | -0.004243 | 1.3090 | 1.0304 |
| `mean_word_length` | 0.0843 | 0.0193 | 0.0188 | 4.37x | 4.50x | **1.6154** | 0.000000 | -0.000251 | 0.3537 | 1.0107 |

**1b replicates exactly.** Platt moves AUROC by `0.000000` for all three attributes — not
"to four decimal places", but to the last bit the bootstrap prints, because Platt is
strictly monotone and AUROC is a rank statistic. Isotonic moves it by at most 0.004243, and
mostly downward. This is §20.3's finding, reproduced in a second domain on a different
architecture.

**1a fails, and this is the one place where C24 contradicts the molecular *empirics* rather
than confirming them.** The molecular slopes were 0.405–0.618, *all* below 1, and §20.3
built its story on that: the probe is **under**-confident off-policy — it predicts 0.076
against an observed 0.267, `pilot_report.md` §8.2 and §9.2.1 — correcting it *raises* small
`q`, which for a power map is an exponent below 1, i.e. it flattens
`log q`, and flattening `log q` is a λ decrease. On text, two attributes behave the same way
(0.8880, 0.7791, both below 1 but much nearer 1 than any molecular slope) and
`mean_word_length` goes the **other way**: α = **1.6154**, well above 1. Its off-policy
`under_confidence_factor` is **0.3537** — and that factor is `observed / mean_predicted`
(`src/property_to_go/calibration.py:312`), so a value *below* 1 means the head predicts
**more** than it observes: 0.1159 predicted against 0.0410 observed. The head is
**over-predicting** the guided policy's hit rate by a factor of 2.83, which is the opposite
of the molecular heads, where the off-policy failure is *under*-prediction (0.076 predicted
against 0.267 observed, C6). Correcting an over-prediction pushes small `q` **down**, which
for a power map is an exponent **above** 1, so the correcting map **sharpens** `log q`
rather than flattening it, and the equivalent λ is 1.6154, not 0.89. Under the
pre-registered verdict rule this is enough on its own: **Claim 1 FAILS TO REPLICATE.**

> **Correction, 2026-08-03.** The two sentences above and the parallel one in §C24.11 both
> said "*under*-predicting" / "under-confident" for a factor of 0.3537. That is inverted
> relative to `calibration.py:312`. The direction word was wrong in both places; the
> **mechanism was right in both**, and every downstream number (α = 1.6154, the sharpening,
> the 2.0000 lift ratio, the FAILS TO REPLICATE verdict) follows from α, not from the label,
> so nothing computed changes. What changes is that the sentence now names the same
> direction the artefact does.

**1c also fails**, on `digit_count`: Platt reduces its ECE by only **1.48x**, below the
pre-registered factor of 2. Isotonic manages **1.90x**, also below 2. Both other attributes
clear the bar comfortably (2.71x–4.50x). `digit_count`'s uncalibrated ECE is already small
(0.0191) and its off-policy factor is 0.9768, i.e. it is essentially calibrated to begin
with, so there is little for a calibrator to fix. 1c is an auxiliary check that the
calibrators genuinely work; it is not part of the verdict rule, and its failure here is a
statement about the head being already-calibrated on one attribute, not about the fitters.

## C24.6 Claim 1's algebraic core — the identity, sequence by sequence

`λ·log(c·q^α) = (λα)·log q + λ·log c`, and a softmax over candidates annihilates the
constant. At ε = 0 a power-calibrated head `g(q)=q^α` at λ=1 must therefore be the *same
sampler* as the raw head at λ=α. Run as two full guided generations, 1536 sequences each
(3 seeds × 512), compared text by text.

| attribute | α | identical at ε=0 | fraction | Δ hit rate | identical at ε=1e-6 | fraction | Δ hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `digit_count` | 0.8880 | 1536 / 1536 | 1.0000 | +0.000000 | 1535 / 1536 | 0.9993 | +0.000000 |
| `upper_count` | 0.7791 | 1536 / 1536 | 1.0000 | +0.000000 | 1536 / 1536 | 1.0000 | +0.000000 |
| `mean_word_length` | 1.6154 | 1536 / 1536 | 1.0000 | +0.000000 | 1536 / 1536 | 1.0000 | +0.000000 |

**Exact, in every cell, including the attribute whose slope is above 1.** The identity does
not care about the sign of `log α`; it is algebra, and it holds on GPT-2 exactly as it holds
on GP-MoLFormer. This is the single most portable result in C24 and the one that most
deserves to be quoted outside this project.

The one non-identical sequence is at the **deployed** ε = 1e-6, for `digit_count`: 1535 of
1536. That is the expected failure mode and it is the right size. With ε > 0 the two
expressions differ by `λ·log((q+ε)^α·c)` versus `(λα)·log(q+ε)`, which is not a constant, so
the samplers can diverge — and they diverge on **one sequence in 1536**, once, without moving
the hit rate. The numeric check in `outputs/c24_calibration/` is stronger still: max absolute
difference between the two candidate-weight arrays at ε=0 is `2.220446e-16` (one ULP) with
argmax agreement 1.0.

## C24.7 End to end — and the reference that decides the sign

**The truncation control is not a formality on this substrate.** Restricting GPT-2 to its
top 8 tokens, with the property term switched off, moves every attribute hard and in the
*wrong* direction:

| attribute | `unguided` | `truncation_control` (λ=0, top-8) | truncation effect |
| --- | ---: | ---: | ---: |
| `digit_count` | 0.0944 | 0.0182 | -0.0762 |
| `upper_count` | 0.0781 | 0.0410 | -0.0371 |
| `mean_word_length` | 0.1003 | 0.0143 | -0.0859 |

Top-8 truncation destroys 47.5%–85.7% of the base hit rate before any guidance happens, because
each of these attributes lives in the tail of GPT-2's next-token distribution (digits, capital
letters and long words are exactly what a top-8 restriction removes). The molecular §7.9
control is a formality; this one is not. **Every lift below is therefore measured against
`truncation_control`, and the `guided − unguided` column is printed beside it so the reader
can see how large the difference is.**

Hit rate, mean over generation seeds 101/202/303, 512 sequences per seed:

**`digit_count`**, target [7, 13)

| arm | hit | sd | per seed (101 / 202 / 303) | − trunc | − unguided | tok/seq | N | best-of-N hit | advantage | realised ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unguided` | 0.0944 | 0.0127 | 0.0820 / 0.1074 / 0.0938 | +0.0762 | +0.0000 | 40.0 | — | — | — | — |
| `truncation_control` | 0.0182 | 0.0063 | 0.0137 / 0.0254 / 0.0156 | +0.0000 | -0.0762 | 40.0 | 1 | 0.1074 | -0.0892 | 1.0000 |
| `throughout_L12` | 0.0957 | 0.0135 | 0.0801 / 0.1035 / 0.1035 | **+0.0775** | +0.0013 | 360.0 | 9 | 0.5931 | -0.4974 | 1.0000 |
| `throughout_L2` | 0.0651 | 0.0127 | 0.0645 / 0.0781 / 0.0527 | +0.0469 | -0.0293 | 360.0 | 9 | 0.5931 | -0.5280 | 1.0000 |
| `platt_L12` | 0.0853 | 0.0164 | 0.0664 / 0.0938 / 0.0957 | +0.0671 | -0.0091 | 360.0 | 9 | 0.5931 | -0.5078 | 1.0000 |
| `isotonic_L12` | 0.0814 | 0.0214 | 0.0566 / 0.0938 / 0.0938 | +0.0632 | -0.0130 | 360.0 | 9 | 0.5931 | -0.5117 | 1.0000 |
| `identity_power_eps0` | 0.0866 | 0.0158 | 0.0684 / 0.0957 / 0.0957 | +0.0684 | -0.0078 | 360.0 | 9 | 0.5931 | -0.5065 | 1.0000 |
| `identity_lamalpha_eps0` | 0.0866 | 0.0158 | 0.0684 / 0.0957 / 0.0957 | +0.0684 | -0.0078 | 360.0 | 9 | 0.5931 | -0.5065 | 1.0000 |
| `identity_power_epsdep` | 0.0866 | 0.0158 | 0.0684 / 0.0957 / 0.0957 | +0.0684 | -0.0078 | 360.0 | 9 | 0.5931 | -0.5065 | 1.0000 |
| `identity_lamalpha_epsdep` | 0.0866 | 0.0158 | 0.0684 / 0.0957 / 0.0957 | +0.0684 | -0.0078 | 360.0 | 9 | 0.5931 | -0.5065 | 1.0000 |

**`upper_count`**, target [14, 18)

| arm | hit | sd | per seed (101 / 202 / 303) | − trunc | − unguided | tok/seq | N | best-of-N hit | advantage | realised ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unguided` | 0.0781 | 0.0249 | 0.0508 / 0.0996 / 0.0840 | +0.0371 | +0.0000 | 40.0 | — | — | — | — |
| `truncation_control` | 0.0410 | 0.0155 | 0.0352 / 0.0293 / 0.0586 | +0.0000 | -0.0371 | 40.0 | 1 | 0.0983 | -0.0573 | 1.0000 |
| `throughout_L12` | 0.1354 | 0.0171 | 0.1309 / 0.1543 / 0.1211 | **+0.0944** | +0.0573 | 360.0 | 9 | 0.6224 | -0.4870 | 1.0000 |
| `throughout_L2` | 0.1224 | 0.0126 | 0.1172 / 0.1367 / 0.1133 | +0.0814 | +0.0443 | 360.0 | 9 | 0.6224 | -0.5000 | 1.0000 |
| `platt_L12` | 0.1139 | 0.0049 | 0.1133 / 0.1094 / 0.1191 | +0.0729 | +0.0358 | 360.0 | 9 | 0.6224 | -0.5085 | 1.0000 |
| `isotonic_L12` | 0.1257 | 0.0088 | 0.1250 / 0.1172 / 0.1348 | +0.0846 | +0.0475 | 360.0 | 9 | 0.6224 | -0.4967 | 1.0000 |
| `identity_power_eps0` | 0.1126 | 0.0069 | 0.1055 / 0.1191 / 0.1133 | +0.0716 | +0.0345 | 360.0 | 9 | 0.6224 | -0.5098 | 1.0000 |
| `identity_lamalpha_eps0` | 0.1126 | 0.0069 | 0.1055 / 0.1191 / 0.1133 | +0.0716 | +0.0345 | 360.0 | 9 | 0.6224 | -0.5098 | 1.0000 |
| `identity_power_epsdep` | 0.1126 | 0.0069 | 0.1055 / 0.1191 / 0.1133 | +0.0716 | +0.0345 | 360.0 | 9 | 0.6224 | -0.5098 | 1.0000 |
| `identity_lamalpha_epsdep` | 0.1126 | 0.0069 | 0.1055 / 0.1191 / 0.1133 | +0.0716 | +0.0345 | 360.0 | 9 | 0.6224 | -0.5098 | 1.0000 |

**`mean_word_length`**, target [5.5926, 5.9286)

| arm | hit | sd | per seed (101 / 202 / 303) | − trunc | − unguided | tok/seq | N | best-of-N hit | advantage | realised ratio |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unguided` | 0.1003 | 0.0092 | 0.1074 / 0.1035 / 0.0898 | +0.0859 | +0.0000 | 40.0 | — | — | — | — |
| `truncation_control` | 0.0143 | 0.0011 | 0.0156 / 0.0137 / 0.0137 | +0.0000 | -0.0859 | 40.0 | 1 | 0.0996 | -0.0853 | 1.0000 |
| `throughout_L12` | 0.0345 | 0.0092 | 0.0449 / 0.0273 / 0.0312 | **+0.0202** | -0.0658 | 360.0 | 9 | 0.6159 | -0.5814 | 1.0000 |
| `throughout_L2` | 0.0521 | 0.0023 | 0.0508 / 0.0508 / 0.0547 | +0.0378 | -0.0482 | 360.0 | 9 | 0.6159 | -0.5638 | 1.0000 |
| `platt_L12` | 0.0547 | 0.0117 | 0.0664 / 0.0547 / 0.0430 | +0.0404 | -0.0456 | 360.0 | 9 | 0.6159 | -0.5612 | 1.0000 |
| `isotonic_L12` | 0.0534 | 0.0092 | 0.0566 / 0.0605 / 0.0430 | +0.0391 | -0.0469 | 360.0 | 9 | 0.6159 | -0.5625 | 1.0000 |
| `identity_power_eps0` | 0.0501 | 0.0069 | 0.0566 / 0.0508 / 0.0430 | +0.0358 | -0.0501 | 360.0 | 9 | 0.6159 | -0.5658 | 1.0000 |
| `identity_lamalpha_eps0` | 0.0501 | 0.0069 | 0.0566 / 0.0508 / 0.0430 | +0.0358 | -0.0501 | 360.0 | 9 | 0.6159 | -0.5658 | 1.0000 |
| `identity_power_epsdep` | 0.0501 | 0.0069 | 0.0566 / 0.0508 / 0.0430 | +0.0358 | -0.0501 | 360.0 | 9 | 0.6159 | -0.5658 | 1.0000 |
| `identity_lamalpha_epsdep` | 0.0501 | 0.0069 | 0.0566 / 0.0508 / 0.0430 | +0.0358 | -0.0501 | 360.0 | 9 | 0.6159 | -0.5658 | 1.0000 |

**Which reference licenses which claim.**

* Against the **truncation control**, the deployed arm's lift is positive for all three
  attributes: **+0.0775**, **+0.0944**, **+0.0202**. The property term does steer, on every
  attribute, and C24.0.10's third uninterpretability condition is not triggered. Every
  ratio in §C24.10 is computed against this reference.
* Against **`unguided`**, the same arm is +0.0013, +0.0573 and **-0.0658**. Read this way
  `mean_word_length` looks like guidance *hurting*, and the first draft read it that way.
  It is not: -0.0859 of that is the top-8 truncation and +0.0202 is the property term
  pushing back against it. **The sign of the mean-word-length result depends entirely on
  which control is used, which is why the control had to be run.**
* What the `unguided` column *does* license, and the truncation column does not, is the
  practical statement that **on this substrate the whole top-8 FUDGE apparatus, guidance
  included, is worse than plain sampling for two of three attributes.** Both readings are
  true and they answer different questions; neither is quoted as the other.

## C24.8 Claim 2, divergence half — per position against end to end

Per-position `our_head_gain` from 300 test-split prefixes × 8 candidates × 32 base-policy
rollouts (rollout seed 7777), against the end-to-end hit rate of a full guided run at the
same probe point. A per-position number is never quoted as if it were end to end.

| attribute | per position, `L`=12 | per position, `L*`=2 | relative | improves? | end to end, `L`=12 | end to end, `L*`=2 | lift ratio | Δhit mean (sd) | t interval, 2 df | excludes 0 | improves? |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | --- | --- | :---: | :---: |
| `digit_count` | +0.00327 | +0.00180 | -0.4491 | no | 0.0957 | 0.0651 | 0.6050 | -0.0306 (0.0181) | [-0.0757, +0.0145] | **false** | no |
| `upper_count` | +0.00113 | +0.00140 | +0.2346 | yes | 0.1354 | 0.1224 | 0.8621 | -0.0130 (0.0049) | [-0.0252, -0.0008] | true | no |
| `mean_word_length` | +0.00058 | +0.00129 | +1.2256 | yes | 0.0345 | 0.0521 | 1.8710 | +0.0176 (0.0101) | [-0.0076, +0.0428] | **false** | yes |

Per-seed Δhit (L2 − L12): `digit_count` -0.015625 / -0.025390625 / -0.05078125;
`upper_count` -0.013671875 / -0.017578125 / -0.0078125; `mean_word_length` +0.005859375 /
+0.0234375 / +0.0234375. All three seeds agree in sign for all three attributes.

> **Correction, 2026-08-01 — the interval in this table was replaced, and two of the three
> "excludes 0" verdicts flipped as a result.** The original table reported a *seed-paired
> percentile bootstrap*, and at n = 3 that statistic is vacuous: the percentile bootstrap
> of a mean over three values is **identically [min, max]** of those three values, because
> P(all three resampled indices land on the minimum) = 1/27 = 0.0370 > 0.025, so the 2.5th
> percentile *is* the minimum for any three numbers whatsoever. Its endpoints above were
> literally the smallest and largest per-seed differences. "The bootstrap excludes zero"
> was therefore not an interval statement at all — it was exactly the statement "all three
> seeds share a sign", a three-way sign test with two-sided null probability
> **2 × (1/2)³ = 0.25**, which cannot reject at any conventional level. It was reported as
> a confidence interval and it overstated the evidence in every row.
>
> The replacement is a Student t interval on 2 df (`t₀.₉₇₅,₂ = 4.302653`), which needs
> |mean| / sd > 2.48 to exclude zero. Under it, **only `upper_count` survives**;
> `digit_count` (-0.0306, sd 0.0181) and `mean_word_length` (+0.0176, sd 0.0101) do not.
> The per-seed values are unchanged and are printed above so the reader can see the whole
> sample. The verdicts 2c, 2d and 2e are computed from lift *ratios* and sign counts, not
> from this interval, so **none of them moves**; what moves is how strongly the supporting
> sentences may be phrased. See §C24.12 item 3.

**2c: NOT MATERIAL.** Swapping to `L*` improves the per-position proxy for 2 of 3
attributes, but the median relative improvement is **+0.2346**, below the pre-registered
+0.25. The prediction was NOT MATERIAL and NOT MATERIAL is what came out — **by 0.0154 of
relative gain, on a median of three numbers.** That is not a robust verdict and it is
labelled as fragile everywhere it is used below.

**2d: NOT POSITIVE.** End to end, `L*`=2 beats probe point 12 for **1 of 3** attributes,
not the required 2. And the failure is not a null: for `digit_count` and `upper_count` the
best-*predicting* layer steers **worse** (ratios 0.6050 and 0.8621), with all three seeds
agreeing in sign in both cases. Only `upper_count`'s t interval excludes zero; the word
"significantly" has been withdrawn from this sentence for `digit_count`, whose interval
[-0.0757, +0.0145] does not.

**2e: the cell is (NOT MATERIAL, NOT POSITIVE) → NO DIVERGENCE.** The pre-registration named
this cell in advance and named what it means: *both say the layer does not help; Claim 2's
methodological half fails to replicate, and in the direction that would have made the cheap
proxy adequate.* On text, the cheap per-position proxy and the expensive end-to-end
measurement **agree**. Had the analyst run only the proxy and concluded "do not switch
layer", the end-to-end result would have confirmed it.

**This is a genuine divergence from the molecular C23 result and is reported as one.** In
molecules the proxy said NOT MATERIAL and end to end said POSITIVE — probe point 4 beat
probe point 12 for HBD count (**0.3689 against 0.2988, a margin of +0.0701**) and probe
point 3 beat 12 for aromatic
rings, which the per-position column had ranked the other way. On text the proxy said
NOT MATERIAL and end to end says NOT POSITIVE. **The molecular claim "the layer that
predicts best is not the layer that steers best" survives in a weaker form** — the
best-predicting layer is still not the best-steering layer here either: probe point 2
predicts best at every attribute, yet of the two probe points measured end to end it is
probe point 12 that steers better at two of the three — but the specific
*methodological* moral drawn from C23, that the cheap proxy misleads and must be replaced by
an end-to-end run, does **not** travel to this substrate.

> **Correction, 2026-08-03 — the molecular comparison above quoted the wrong arm.** The
> original sentence read "probe point 4 beat probe point 12 for HBD count (0.3689 against
> 0.3395)". **0.3395 is probe point *6*, not probe point 12.** Read off
> `outputs/c23_summary/c23_metrics.json`: `hbd_count_L4_lam1/throughout_mean` = 0.3689,
> `hbd_count_L6_lam1/throughout_mean` = **0.3395**, and the deployed probe point 12 value
> that both are compared against is `deployed_throughout_mean` = **0.2988**. C23's own
> §C23.3 table publishes both rows correctly (probe point 4: +0.0701; probe point 6:
> +0.0407); the transcription into this section collapsed them.
>
> **The margin was understated by a factor of 2.4** — +0.0294 as quoted, +0.0701 as
> measured. The error runs *against* C24's own conclusion, since a larger molecular effect
> makes the failure to replicate it on text a sharper divergence, not a softer one. No
> verdict in this section moves: 2c, 2d and 2e are scored from text artefacts and never
> touched this number.



## C24.9 Compute-matched best-of-N

`N` solved from each arm's own `processed_tokens_actual` per returned sequence against the
base policy's. Every guided arm costs **360.0** tokens per sequence against the base
policy's **40.0**, so `N` = 9 for every guided arm and the realised token ratio is
**1.0000** everywhere — the check the pre-registration asked for (C24.0.8: 9 × 40.0 = 360.0
tokens per returned best-of-N sequence against the guided arm's 360.0, *measured* and
written to the artefact) rather than an assumption. `truncation_control` costs 40.0
tokens per sequence, because the λ=0 path never runs the probe, so its match is `N` = 1.

| attribute | best-of-9 hit | best guided arm hit | best advantage |
| --- | ---: | ---: | ---: |
| `digit_count` | 0.5931 | 0.0957 | -0.4974 |
| `upper_count` | 0.6224 | 0.1354 | -0.4870 |
| `mean_word_length` | 0.6159 | 0.0547 | -0.5612 |

**No arm, at any attribute, beats compute-matched best-of-N** — the closest is -0.4870, an
enormous margin. C24 replicates the molecular pipeline's central negative result (§16.2,
§19.2) on a second generator in a second domain, and does so more emphatically, because the
text attributes are exactly computable so best-of-N gets perfect ground truth for free. This
was not one of the two claims under test and is reported as a by-product.

`truncation_control` at N=1 is the degenerate case and is included for completeness: it
loses to a single unguided sample (advantages -0.0892, -0.0573, -0.0853), which is just the
truncation penalty restated.

## C24.10 Every pre-registered decision rule, scored

Read off `outputs/c24_summary/c24_summary.json`, not argued in prose. **Failures carry the
same weight as successes.**

> **Integrity disclosure, added 2026-08-03 — the reference arm below was chosen after the
> first run, not pre-registered.** The pre-registration (§C24.0) named `unguided` as the
> reference for every guided contrast. `truncation_control` (λ=0, top-k=8) did not exist
> when the rules were written; it was added after the first run showed that
> `guided − unguided` confounds the property term with top-k truncation, and §C24.1.1
> documents that addition. **What §C24.1.1 does not say, and this note does, is that the
> post-hoc arm then became the denominator of the lift ratios in 1e and of the
> not-merely-null check in §C24.0.10.** A reference chosen after seeing that the
> pre-registered one gave the wrong answer is a researcher degree of freedom, and it must be
> visible at the point where it is used rather than 350 lines earlier.
>
> Three things bound how much it can have bought:
>
> 1. **Both denominators are published side by side** and disagree in one place, which is
>    reported rather than resolved: the guided lift is positive at **3 of 3** attributes
>    against `truncation_control` and at **2 of 3** against `unguided` (§C24.0.10). The
>    weaker of the two is the pre-registered one.
> 2. **The verdict rule did not move.** Claim 1 scores FAILS TO REPLICATE on 1a, which is a
>    fitted Platt slope and involves no reference arm at all; 1b and 1d are identities. The
>    denominator affects 1e, which the rule text explicitly does not weight ("*1e would have
>    given REPLICATES on its own*").
> 3. **The addition was proved additive rather than asserted**: the 54 pre-existing
>    `arm_*.json` and `texts_*.json` files are byte-identical (SHA-256) before and after the
>    re-run, so no existing arm was recomputed to fit the new reference.
>
> The honest statement is that `truncation_control` is the **scientifically correct**
> reference — a guided arm that samples from a top-8 set must be compared against a λ=0 arm
> that also samples from a top-8 set — and that it is nevertheless **not the pre-registered**
> one. Both facts are true and both belong here.

### Claim 1

| # | prediction | result | holds? |
| --- | --- | --- | :---: |
| **1a** | every fitted Platt slope α < 1 | 0.8880, 0.7791, **1.6154** | **NO** |
| **1b** | Platt leaves target AUROC unchanged to 4 dp | ΔAUROC = 0.000000 for all three | **YES** |
| **1c** | ECE falls by a factor ≥ 2 under Platt *and* isotonic, every attribute | Platt 1.48x / 2.71x / 4.37x; isotonic 1.90x / 2.80x / 4.50x | **NO** (`digit_count`) |
| **1d** | the identity: at ε=0, `q^α` at λ=1 ≡ raw at λ=α, identical fraction 1.000 and Δhit 0.0 | 1.0000 / 1.0000 / 1.0000, Δhit +0.000000 everywhere | **YES** |
| **1e** | calibrated arms have lift < 1.00× the uncalibrated arm's | Platt 0.8655 / 0.7724 / **2.0000**; isotonic 0.8151 / 0.8966 / **1.9355** — below 1 at 2 of 3 for both | **YES** (2 of 3, both fitters) |

**Verdict rule, applied literally: `FAILS TO REPLICATE`**, because 1a fails. (The rule is
*"FAILS TO REPLICATE iff 1a, 1b or 1d fails"*; 1b and 1d hold, 1e would have given
REPLICATES on its own.) The pre-registration is scored as written, including where the
verdict rule is arguably harsher than the claim it encodes — see §C24.11.

**1e is worth reading with 1a in hand.** The lift ratio tracks α with an accuracy that is
hard to dismiss: `upper_count` α = 0.7791 and Platt lift ratio 0.7724; `digit_count`
α = 0.8880 and ratio 0.8655; `mean_word_length` α = 1.6154 and ratio 2.0000, i.e. above 1 on
the attribute whose slope is above 1. **Calibration behaved exactly as a λ rescale of size α
in all three cases, including the one where that made it help.** The mechanism replicates
even where the sign flips; what does not replicate is the empirical regularity that α is
always below 1.

### Claim 2

| # | prediction | result | holds? |
| --- | --- | --- | :---: |
| **2a** | peak strictly before 12, in `L* ∈ 1..6`, for ≥ 2 of 3, and no attribute peaks at 12 | `L*` = 2 / 2 / 2; 3 of 3 in the first half; none at 12 | **YES** |
| **2b** | probe point 12 is the minimum over `L*..12`, every attribute | true / true / true | **YES** |
| **2c** | swapping to `L*` is NOT MATERIAL per position | improves 2 of 3, median relative +0.2346 < +0.25 → **NOT MATERIAL** | **YES** (marginally) |
| **2d** | end to end, `L*` beats 12 for ≥ 2 of 3, bootstrap excluding 0 | 1 of 3 → **NOT POSITIVE** | **NO** |
| **2e** | the cell (NOT MATERIAL, POSITIVE) — the molecular pattern | cell is (NOT MATERIAL, NOT POSITIVE) → **NO DIVERGENCE** | **NO** |

**Multiplicity.** 13 probe points × 3 attributes = 39 comparisons; each "probe point `L`
beats 12" claim was required to clear a paired bootstrap at 1 − 0.05/13. All three cleared
it. The anti-spike guard (`A(L) − A(12) ≥ 0.010` with both neighbours ≥ 0.005) is satisfied
for `upper_count` and `mean_word_length` and **not** for `digit_count`. Per-seed AUROC values
for all 39 cells are published in `outputs/c24_probe_layers/probe_layer_metrics.json`.

### C24.0.10 — is the result interpretable, or merely null?

| condition for uninterpretability | measured | triggered? |
| --- | --- | :---: |
| fewer than two attributes surviving the base-rate gate | 3 of 3 survived (0.0958, 0.0973, 0.1000) | no |
| probe at chance at every layer (max `A(L)` < 0.55 for every attribute) | max `A(L)` = 0.7979 / 0.7686 / 0.6857 | no |
| uncalibrated guided lift not positive at any attribute | positive at 3 of 3 against `truncation_control`; at 2 of 3 against `unguided` | no |

C24 is a **negative-to-mixed result, not an uninterpretable one**.

## C24.11 What C24 says about Claim 1 and Claim 2

**Claim 1 — the algebra travels; the empirical regularity does not.** The identity is exact
on a second architecture, in a second domain, at every attribute, with 1536 sequences
matching sequence for sequence. Platt's non-effect on AUROC is exact. The lift ratio tracks
α. Every mechanical component of §20.3's argument reproduces. What does not reproduce is the
*premise* that the off-policy probe is always *under*-confident in the molecular direction —
predicting less than it observes, which is what the molecular heads do (0.076 against 0.267)
and what makes their fitted α fall below 1. On `mean_word_length` the probe fails the other
way: it **over**-predicts, 0.1159 against 0.0410, an `under_confidence_factor` of 0.3537
(that field is `observed / mean_predicted`, so below 1 means over-prediction). Hence
α = 1.6154 and calibration *doubles* the
lift (ratio 2.0000) instead of shrinking it. The pre-registered verdict rule bundled that
premise into 1a and returns **FAILS TO REPLICATE**; that is the honest score of the
pre-registration as written. The honest score of the *claim* is narrower and should be
stated that way when §20 is revised: **"post-hoc calibration is exactly a λ rescale by α" is
general; "and therefore it hurts" is contingent on α < 1, which is a property of the
particular head and the particular off-policy gap, not of the method.** C24 is the first
measurement in this project that separates those two statements, and it separates them
because a substrate was found where α > 1.

**Claim 2 — the depth curve travels; the proxy-versus-end-to-end divergence does not.** The
shape replicates cleanly and with a Bonferroni-corrected interval: mid-network beats final
layer for prediction at every attribute, monotonically. The methodological moral does not:
here the proxy and the end-to-end measurement agree, and following the proxy would have been
correct. Two things follow. First, C23's insistence on measuring end to end rather than
extrapolating from the proxy is **vindicated as a procedure and not as a prediction** — the
proxy happens to be right on this substrate, but nothing in the proxy told us that, and it
took an end-to-end run to find out. Second, and against C23's own suggested repair, the
**mid-network layer is not a free win**: on text it is the *worse* steering layer for two of
three attributes, by ratios 0.6050 and 0.8621. Any statement of the form "probe mid-network
and guidance improves" must be scoped to molecules until measured again.

## C24.12 What a sceptical reviewer should attack, and my answers

Listed because they are the weakest joints in this section, not because they are resolved.

1. **2c is decided by 0.0154 on a median of three numbers.** Median relative improvement
   +0.2346 against a threshold of +0.25. Had `upper_count`'s per-position gain been 2%
   larger, 2c would read MATERIAL and the 2e cell would be (MATERIAL, NOT POSITIVE) —
   "DIVERGENCE WITH THE OPPOSITE SIGN", a *different reported result*. The threshold was
   pre-registered and is not moved here, but the verdict is one attribute away from
   flipping and no one should quote 2e without this sentence. The underlying per-position
   gains are tiny in absolute terms (+0.00058 to +0.00327) and three attributes is a very
   small family for a median.
2. **Three generation seeds — and the interval that was reported for them has been
   withdrawn.** §C24.8 originally resampled three paired values and reported the result as
   a bootstrap CI. That statistic is degenerate: at n = 3 the percentile bootstrap of a
   mean is identically [min, max], so "the interval excludes zero" was precisely "all
   three seeds agree in sign" — a sign test at p_null = 0.25 — and not an interval
   statement at all. The earlier text said the intervals "exclude zero because all three
   seeds agree in sign, not because three points pin down a distribution", which was the
   right intuition attached to the wrong statistic; the honest move was to stop reporting
   it as a CI, and that has now been done throughout (Student t on 2 df instead, under
   which two of the three rows no longer exclude zero). Per-seed values are printed for
   every arm so the spread is legible rather than asserted. The seed-to-seed spread is
   comparable to some of the effects: `unguided` `upper_count` ranges 0.0508 to 0.0996
   across seeds.
   **This is a defect the section shared with C23 and C26 and it was corrected in all
   three on 2026-08-01.** It is recorded here rather than silently fixed because two rows
   of §C24.8 read as significant before the correction and do not after.
3. **The `trivial` baseline beats the probe everywhere.** Margins -0.0750, -0.1061, -0.1084.
   A reviewer is entitled to say the C24 probe is simply bad, and that a depth curve over
   a bad probe says little about depth curves over good ones. My answer is only partial:
   the probe is well above chance (0.6857–0.7979), the depth *ordering* is what Claim 2 is
   about, and the same relation holds more weakly in molecules (`trivial` 0.8269 against
   0.8474). But it is a real limit and it is why §C24.11 scopes the Claim 2 result to shape
   rather than to magnitude.
4. **The substrate is easy for best-of-N and hard for guidance.** Attributes computable in
   microseconds hand best-of-N perfect ground truth, and top-8 truncation costs 47.5%–85.7% of
   the base hit rate before guidance starts. Both make the guided arms look worse than they
   would on a substrate where the attribute is expensive and lives inside the top 8. §C24.9's
   -0.4870 margin should not be read as a stronger version of the molecular result; it is a
   result about a substrate chosen for the two claims, not for the best-of-N comparison.
5. **The λ=0 control was added after the first run.** It was, and the omission produced a
   wrong first reading of `mean_word_length` (recorded in §C24.1.1 rather than quietly
   fixed). The mitigation is mechanical: the new arm was added additively, `run_arm` is
   idempotent, and all 54 pre-existing arm and text artefacts are byte-identical before and
   after. Nothing else was re-run and nothing else could have moved. But the control was
   not pre-registered as an arm, and a reviewer should treat it as what it is — a control
   added on discovering a design flaw, whose *result* was not seen before it was specified.
6. **No λ sweep.** C24 runs λ = 1 only (plus λ = α for the identity), as pre-registered in
   C24.0.9. `mean_word_length`'s calibrated arm being 2.0000× the uncalibrated one is
   consistent with the inverted-U of §19.1 not having peaked yet at λ=1 on that attribute;
   C24 cannot distinguish "calibration helps" from "λ=1.6 helps and calibration is how we
   got there". The identity in §C24.6 says these are the *same statement*, which is the
   point, but it means no independent claim about calibration-as-calibration is available.
7. **One head seed for every generation arm.** Depth AUROCs use head seeds 1234/2345/3456;
   every end-to-end arm uses head seed 1234 only. A head-seed replication of the end-to-end
   arms was not run and is not claimed.
8. **Fixed 41-token sequences remove the length confound the molecular pipeline standardises
   for (§9), by design (C24.0.2).** That makes C24 a cleaner test of these two claims and no
   test at all of anything the molecular report says about length.

## C24.13 Contradictions and tensions with the existing report

**Flagged for the owner to merge. Nothing in `reports/pilot_report.md` was edited by C24,
and nothing in it should be edited on C24's authority alone — C24 is a different generator
in a different domain.**

1. **§20.3's "every fitted slope is below 1, as predicted" is a molecular fact, not a
   general one.** On text one of three slopes is 1.6154. Suggested repair: keep §20.3's
   numbers, and separate the algebraic statement (a λ rescale by α — general, and now
   confirmed on a second architecture) from the empirical one (α < 1 — measured, and now
   known to fail elsewhere). §20.6's item 1, which states the mechanism as
   "correcting under-confidence necessarily flattens `log q`, which is a λ decrease", is the
   sentence that needs the qualifier: it is necessary only when the correction flattens,
   i.e. when α < 1.
2. **`reports/section_c23_layer_end_to_end.md`'s item 4 — "the per-position column is
   contradicted as a guide" — does not generalise.** On text the per-position column is a
   *correct* guide. C23's finding stands as measured on molecules; the broader reading, that
   per-position proxies mislead in general, is not supported.
3. **§16.2 / §19.2's "no arm beats compute-matched best-of-N" gains an independent
   replication** on a second generator and domain (advantages -0.4870 to -0.5814). This
   strengthens the existing claim rather than contradicting it, and C23's molecular
   exception (a mid-network arm that does beat best-of-N) has **no analogue here**: probe
   point 2 loses to best-of-9 by -0.5280, -0.5000 and -0.5638.
4. **§21.3's depth curve replicates on a non-molecular stack.** Best probe point strictly
   before the final layer, monotone decline after the peak, all three attributes, structurally
   matched 12-block / 13-probe-point architecture. This is the most transferable positive
   result in C24 after the identity.

## C24.14 What C24 did not do

Restated so its absence is not read as an omission: no λ sweep; no bin-logit-temperature and
no retrained-readout arm (§20.4's route (b)); no quality analogue beyond uniqueness and
descriptive statistics; no DAgger, no fine-tuning, no activation steering — GPT-2 is frozen
throughout and no weight was modified; no window sweep (`throughout` and `unguided` only);
no head-seed replication of the end-to-end arms.

---

## Commands to add to `docs/REPRODUCE.md`

A section "C24 — the generality (external-validity) experiment", to run **after** the
molecular chain and independently of it. C24 reads no molecular artefact and writes none, so
it can be run at any point; it needs `gpt2` in the local HF cache (revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`).

```bash
# C24.0  the pre-registration must already be on disk and must not be edited.
#        outputs/c24_prereg/prereg.{md,json} — every measurement artefact below is
#        asserted newer than it by tests/test_generality.py.
sha256sum outputs/c24_prereg/prereg.md   # must match prereg.json's "sha256"

# C24.1  base sample, grouped splits, target intervals, binners. ~1.6M processed tokens.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_dataset.py

# C24.2  heads at all 13 probe points x 3 head seeds, plus the `trivial` baseline.
#        No generation; head training only.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_heads.py

# C24.3  off-policy calibrators, ECE/AUROC, and the numeric form of the identity.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_calibrate.py

# C24.4  per-position steering proxy at all 13 probe points (300 prefixes x 8 x 32).
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_steering.py

# C24.5  validity gate: cached decode vs full-prefix recomputation at all 13 probe
#        points. Run it before trusting the token accounting. Exits non-zero on failure.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_gate.py

# C24.6  the end-to-end arms and their compute-matched best-of-N.
#        Idempotent per arm: a completed arm_<attr>_<name>.json is NOT regenerated, so a
#        kill costs at most one arm — and so adding an arm (as `truncation_control` was
#        added) recomputes only the new one.  --batch-size 32 for a shared GPU.
HF_HUB_OFFLINE=1 .venv/bin/python scripts/19_c24_endtoend.py --batch-size 32

# C24.7  score every pre-registered decision rule. Reads artefacts only.
.venv/bin/python scripts/19_c24_summarise.py

.venv/bin/python -m pytest tests/test_generality.py -q -p no:cacheprovider
```

**Pins recorded with every result.** `write_run_context` writes `provenance.json` and
`configs_used.json` beside every C24 output directory: Python 3.12.3, torch 2.4.1+cu121,
transformers 4.44.2, numpy 1.26.4, `gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, float32, device `cuda`, generator frozen and in
eval mode. Deterministic seeds throughout: base sample 20240001, prefix 12, split 11, head
seeds 1234/2345/3456, generation seeds 101/202/303, rollout seed 7777, best-of-N seed 5150,
calibration guided seed 91234. Cost is `processed_tokens_actual` and
`processed_tokens_full_recompute`; never wall-clock.

**Artefact sizes.** `outputs/c24_endtoend/` is ~9 MB and holds **60** arm and text JSONs
(30 `arm_*.json` + 30 `texts_*.json`) plus the aggregate `endtoend_metrics.json`,
`configs_used.json` and `provenance.json` — 63 files;
`outputs/c24_calibration/calibration_metrics.json` is ~800 kB;
`outputs/c24_probe_layers/heads/` holds **39** head checkpoints, 13 probe points × 3
attributes at **one** head seed (1234) — the three head seeds named in the pins are used in
`probe_layer_metrics.json`'s per-seed AUROCs, but only the seed-1234 checkpoints are written
to disk. Only
`outputs/c24_summary/c24_summary.json` and the four stage metrics JSONs are needed to
re-derive every number in this section; `tests/test_generality.py` reads them and requires
each printed number to appear here, formatted exactly as printed.
