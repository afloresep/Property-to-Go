# C24.0 — Pre-registration: the generality experiment

**Written and saved to disk before any C24 measurement script was run.** This file is
immutable. It is copied verbatim into `reports/section_c24_generality.md` §C24.0 and
scored there, including where it fails. `tests/test_generality.py` asserts (a) that the
copy is byte-identical to this file and (b) that this file's mtime strictly precedes the
mtime of every C24 measurement artefact.

---

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

*(Everything above was on disk before the first C24 measurement ran.)*
