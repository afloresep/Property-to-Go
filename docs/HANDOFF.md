# Handoff: continuing this project on other hardware

Written 2026-07-30, after the CPU pilot completed. Read this **and**
[`../reports/PLAIN_SUMMARY.md`](../reports/PLAIN_SUMMARY.md) before touching anything.
If you only read one other file, read
[`../reports/ABSTRACT.md`](../reports/ABSTRACT.md) — it lists every claim we make and
the strongest objection to each, which is what the follow-up work has to address.

This document exists so an agent or person with no prior context can run the next
experiments without re-deriving the design or re-making the mistakes.

---

## 0. Thirty-second orientation

We test whether a **frozen** autoregressive SMILES model's internal state at a partial
molecule predicts the *finished* molecule's properties, and whether that prediction can
steer generation. Prediction works. Steering works but loses badly to
compute-matched best-of-N sampling. The pilot is **complete and reported**; what remains
is a small set of follow-ups that would strengthen or overturn specific claims.

> **UPDATE, 2026-08-01. C23, C24, C25 and C26 are all run and reported, and together they
> settle the follow-up programme.** Sections `reports/section_c23_layer_end_to_end.md`,
> `section_c24_generality.md`, `section_c25_pooling.md`, `section_c26_n_sweep.md`; tests
> `tests/test_layer_end_to_end.py`, `test_generality.py`, `test_pooled_readout.py`,
> `test_n_sweep.py`. Full suite 477 passed, 0 skipped.
>
> Four things changed in what this project claims:
>
> 1. **A mid-network head does improve guided generation** (C23 Rule A, 15/15 arms
>    positive). This contradicts §21.5 in *sign*: per position the same swap looked
>    neutral-to-harmful. Per position is not end to end, and here they disagree.
> 2. **Nothing beats compute-matched best-of-N.** C23's one apparent exception is retired:
>    priced on C26's continuous frontier it is +0.0267, and across C25's three head seeds
>    it is +0.0267 / +0.0210 / **−0.0649**. Head-seed spread on that arm is 25× the
>    generation-seed spread, so C23's three seeds replicated the wrong thing. **Any future
>    end-to-end claim in this project must be replicated across head seeds, not just
>    generation seeds.**
> 3. **Guidance has no compute knob** (C26). 46 arms inside a 5–17% token band against
>    best-of-N's 32×. The right statement is not "guidance loses at matched compute" but
>    "guidance cannot be given more compute".
> 4. **"Calibration hurts" is contingent, not general** (C24). The λ-rescale identity is
>    exact on a second, non-molecular substrate, but where the Platt slope exceeds 1,
>    calibration *helps*. The algebra travels; the sign does not.
>
> Still genuinely open: pooling has **no** steering result (C25's end-to-end arms were
> never run), and the head-seed variance that killed Rule B is estimated from three seeds.

> **UPDATE, 2026-08-03. C27–C33 are all run and reported.** The merges live in
> `reports/pilot_report.md` §22 (C23–C29), §23 (C30), §24 (C31, C32) and §25 (C33). Full
> suite **799 passed, 2 skipped**.
>
> Three things a newcomer should know before reading anything above:
>
> 1. **"Guidance has no compute knob" is withdrawn** (§22.1.2) and **the crossing is
>    real** (§22.2, §23, §24): guidance beats even oracle-selected best-of-N at small
>    budgets, on two generators, and the ingredient is **λ, not probe depth** (§24.2).
> 2. **C27's oracle asymmetry — the "~8×" this project promoted hardest — failed its
>    pre-registered replication on the second generator** (§25). What survives is the
>    matched-N *curve*, which agrees across generators to 0.008–0.037; the single-number
>    share does not travel, because it is normalised at whatever budget the guided arm
>    occupies. Any text quoting 0.876/0.882/0.859, "~8×", or "0.03–0.05" without naming
>    generator 1 is out of date.
> 3. **C-numbers were two colliding namespaces and are now one each.** Claims from
>    `reports/ABSTRACT.md` are **`CL<n>`**; experiments from `docs/TODO.md` stay bare
>    **`C<n>`**. `C33` used to mean both "every property predicted best mid-network" and
>    "does the oracle asymmetry replicate?"; it now means only the second. Experiment IDs
>    could not move — they name `outputs/c33_*/`, `scripts/`, `tests/` and frozen
>    pre-registrations.
>
> Paper drafts: `reports/PAPER_WORKSHOP_DRAFT.md` (v2, workshop format) and
> `reports/PAPER_DRAFT_PLAIN.md` (plain English). Figures:
> `.venv/bin/python scripts/28_paper_figures.py` → `outputs/paper_figures/`.

---

## 1. Environment — the part that will waste your time if you skip it

```bash
cd property-to-go
uv venv && uv pip install -e .          # or: python -m venv .venv && pip install -e .
.venv/bin/python -m pytest              # expect 799 passed, 2 skipped (2026-08-03, after C33)
```

**Always invoke `.venv/bin/python` explicitly.** On the original machine the ambient
`python` was a miniforge install with `transformers` 5.2.0, which **cannot load the
pinned GP-MoLFormer revision** — `AutoConfig.from_pretrained` fails inside the remote
code module. The failure looks like a model bug and is not one. The pin is
`transformers==4.44.2` in `pyproject.toml`.

Non-negotiable pins, all in `configs/model.yaml`:

| Field | Value |
|---|---|
| `model_repo` | `ibm-research/GP-MoLFormer-Uniq` |
| `model_revision` | `6eca879581e2302b4e1ab07bb02908636bddb4a2` |
| `tokenizer_repo` | `ibm-research/MoLFormer-XL-both-10pct` |
| `tokenizer_revision` | `361063d0ad524ef77cf39b08469f6be770dc550f` |
| `deterministic_eval` | **`true`** — see below |
| `dtype` | `float32` |

### 1.1 `deterministic_eval` — read this before changing `dtype` or `device`

GP-MoLFormer uses linear attention with a ReLU feature map over **random orthogonal
projections**. The released config sets `deterministic_eval: false`, which makes
`MolformerFeatureMap` **redraw those projections on every un-cached forward pass**. With
the released default, the frozen model's forward pass is stochastic: the same prefix
yields different hidden states on repeat calls, hidden-state datasets are not
reproducible, and a stored prefix cannot be continued reproducibly.

The projections are stored in the checkpoint
(`molformer.encoder.layer.*.attention.self.feature_map.weight`), so setting the flag to
`true` pins the released projections. This is a flag the model authors provide. **No
weight is modified anywhere in this project.** This is the only config field we changed
from the release, and it is recorded as deviation D1 in `reports/pilot_report.md` §7.

`tests/test_model_contracts.py::test_forward_pass_is_deterministic` guards this. If you
ever see it fail, stop — something has un-pinned the projections.

### 1.2 Moving to GPU

`device: cpu` in `configs/model.yaml` was chosen because it benchmarked *faster than
mps* for this 46.8M linear-attention model on the original laptop. On an RTX card set
`device: cuda`. Before running anything long:

1. Run the full test suite on the GPU. `test_forward_pass_is_deterministic` and
   `test_candidate_backends_agree` are the two that matter — the second asserts that
   the cached candidate backend and full-prefix recomputation produce numerically equal
   results, and it is the load-bearing assumption behind the compute accounting.
2. **Do not switch to float16/bf16 without re-running those two tests.** The cached-vs-full
   agreement is a numerical-equality claim; reduced precision may break it. If it does,
   either keep float32 or loosen the tolerance *and say so in the report*, because the
   compute accounting depends on the two backends being interchangeable.
3. Re-run one small guided condition and confirm the hit rate matches the CPU result
   before trusting any new number. GPU/CPU sampling divergence is a real risk with a
   pinned seed.

### 1.3 What does *not* reproduce

Wall-clock time. Two bit-identical runs on the original machine differed 20–25% while
token counts matched to the digit. **Every wall-clock margin in `pilot_report.md` is
below the noise floor and should be ignored.** Report `processed_tokens_actual` and
`processed_tokens_full_recompute`. If you want a timing claim, you need many repeats on
a quiet machine, and it still will not be the interesting number.

### 1.4 The sampled dataset does not reproduce across devices — measured, phase 2

This section was written before phase 2 and was **wrong by omission**; the correction is
recorded here because it is the single most expensive trap in this repository.

`.gitignore` excludes the large arrays on the grounds that they are "reproduced
deterministically from the pinned model revision and recorded seeds". That is true **on
the same class of device and false across devices.** Regenerating `pilot_50k` on an RTX
4090 with the recorded seeds produces a *different* 50,000-molecule sample.

Measured, in `outputs/device_equivalence/device_equivalence.json` and
`pilot_report.md` §11.2:

| | |
|---|---|
| weight checksums, CPU vs CUDA | identical |
| max abs. logit difference | 1.3e-05 |
| **top-8 candidate set** | **identical on every test prefix** |
| same seed, same molecules? | **no — 0 of 64** |
| same device, same seed, twice | **bit-identical, arrays included** |

The cause is not numerics. `torch.manual_seed` seeds the CPU Mersenne Twister and the
CUDA Philox generator both, but `torch.multinomial` draws from whichever generator owns
the tensor, so CUDA makes different choices at the same seed. The base policy is
unchanged; the sample is a fresh draw from it.

**Consequences you must plan for.**

1. **Target intervals are sample quantiles, so they move.** Phase 2's regenerated
   intervals differed from the frozen ones by ~2 standard errors — enough to matter, since
   "frozen before any guided result was inspected" is what makes sections 5 and 6
   interpretable.
2. **Use `scripts/02_generate_trajectories.py --inherit-intervals <path>`.** Any property
   present in the referenced file is copied verbatim; only new ones are derived. What the
   current run *would* have derived is written to `target_intervals_provenance.json`, so
   the divergence is on disk.
3. **Never mix two samples.** `prefix_meta.csv` row indices, `rollout_bank.json`
   `prefix_row` values and head checkpoints all refer to a particular dataset. Phase 2
   therefore lives in its own `outputs/pilot_50k_p2/` and overwrites nothing from phase 1.
4. **The pilot's own 49,823 molecules are gone.** `trajectories.json` was never tracked.
   Any question that needs the pilot's exact molecules cannot be answered on other
   hardware — only re-measured on a fresh sample.

### 1.5 The GPU port needed one real code fix

`guidance.TargetScorer` held a head loaded with `map_location="cpu"` while the frozen
generator produced CUDA candidate states — every guided run would have crashed. It now
migrates the head and the interval mask to the device the states arrive on. Two tests
also called `.numpy()` on CUDA tensors; those were test-only defects.

Speedups measured, for scale: 50k generation 2,744 s → 239 s (11.5x); hidden-state
extraction 1,301 s → 26 s (51x); a 3-seed guided `unguided`+`throughout` run ~846 s → 72 s.

---

## 2. Repository map

```
configs/          model, base_policy, guidance, and per-dataset configs (YAML)
src/property_to_go/
  model_io.py     loads the frozen generator; the only place the pins are applied
  generation.py   batched sampling, prefix extraction, hidden states for positions
  guidance.py     TargetScorer, Windows, guided_sample  <- the method under test
  heads.py        MLPHead + train_head
  binning.py      QuantileBinner / CategoricalBinner, interval_probability
  bestofn.py      best-of-N, and the interval/selection semantics (see §5)
  compute.py      ComputeMeter -- two-way token accounting
  confound.py     direct standardisation with explicit coverage
  quality.py      chemical-quality descriptors and degeneracy flags
  splits.py       grouped splitting by canonical molecule (blake2b)
  metrics.py      brier, auroc, ECE, reliability, bootstrap
scripts/01..10    the pipeline, in order; each writes provenance.json + configs_used.json
tests/            177 tests; see §6
reports/          pilot_report.md (full), PLAIN_SUMMARY.md, ABSTRACT.md
docs/             REPRODUCE.md (8 named reproductions), LITERATURE.md, this file
outputs/          all results; pilot_50k* is the main run
```

Pipeline order and what each stage consumes:

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `01_compatibility.py` | — | `compatibility/` | 10 enumerated ops; the go/no-go gate |
| `02_generate_trajectories.py` | configs | `<dataset>/` | also freezes `windows.json` and `target_intervals.json` **before** any guided run |
| `03_train_heads.py` | `<dataset>/` | `<dataset>_heads/` | trains `frozen_state`, `trivial`, `combined` per property |
| `04_rollout_bank.py` | heads + dataset | `<dataset>_rollouts/` | 800 prefixes x 32 continuations; one bank serves both properties |
| `05_guided_generation.py` | heads + windows | `<dataset>_guided_<prop>/` | 6 conditions x 3 seeds; saves every molecule |
| `06_best_of_n.py` | guided metrics | `<dataset>_bestofn_<prop>/` | solves N from the guided run's token count |
| `07_figures.py` | all of the above | `<dataset>_figures/` | |
| `08_data_aggregation.py` | heads + windows | `<dataset>_dagger_<prop>/` | the one permitted DAgger round |
| `09_confound_analysis.py` | guided molecules | `<dataset>_confound_<prop>/` | length / size / joint standardisation |
| `10_quality_analysis.py` | guided molecules | `<dataset>_quality_<prop>/` | chemical quality; needs no generation |

---

## 3. Design decisions you must not silently undo

These are places where the obvious implementation is wrong. Each cost real debugging
time.

### 3.1 Windows are quantiles of *positions*, not of *lengths*

`early`/`middle`/`late` are defined from the pooled distribution of generated token
**positions** (`t = 1…n` for every trajectory of length `n`), not from quantiles of the
final lengths. Quantiles over final lengths put the 33rd percentile near the median
molecule's *end*, so "early" would cover almost the entire trajectory. See
`Windows.from_lengths` in `src/property_to_go/guidance.py`. This is the one
implementation note added to `README.md`.

### 3.2 Splits are grouped by canonical completed molecule

Every prefix of one molecule goes to the same split, via a stable blake2b hash of the
canonical SMILES. Ungrouped splitting leaks prefixes of the same molecule across
train/test and inflates every predictability number.

### 3.3 Guidance can only use the `frozen_state` head

The decoder has to score a *candidate hidden state* that does not correspond to any
finished string, so the `trivial` and `combined` heads are unusable at decode time. That
the `trivial` head is the better *predictor* for aromatic rings is a finding, not a knob
you are free to turn to make guidance look better.

### 3.4 Two-way compute accounting is mandatory

`ComputeMeter` tracks `processed_tokens_actual` (what the cached backend really did) and
`processed_tokens_full_recompute` (what a naive full-prefix implementation would do).
Best-of-N is matched on tokens, never on forward calls or returned molecules — matching
on forward calls understates guidance's cost by roughly the candidate count. Report
both. The gap between them (−0.35 vs −0.74) is itself part of the argument.

### 3.5 Standardised estimates always carry `coverage`

A length-matched hit rate computed over 40% of the reference distribution is not a
matched hit rate. `confound.py` returns `coverage` with every estimate and the report
quotes it. Do not drop it.

---

## 3.6 The target interval must be a union of bins — enforce it, do not assume it

Added phase 2, because the pilot violated its own invariant here. `binning.py` states
that "target intervals are always defined on bin boundaries, which makes that sum exact
rather than an approximation". Nothing enforced it, and the two quantities came from
different samples: the interval is a quantile of *all* kept trajectories while the binner
is fitted on the *train split's* prefix rows.

`QuantileBinner.interval_mask` keeps only bins lying **wholly** inside `[lo, hi)`, so a
target edge landing mid-bin drops that bin and the head silently predicts a strict subset
of the target. In the pilot's cLogP head this cost one of two bins: the head learned a
0.050-mass event for a 0.100-base-rate target, which the report recorded as the head being
"under-confident by a factor of ~2". See `pilot_report.md` §11.5.

**Pass the target interval to the binner** (`QuantileBinner.fit(..., extra_edges=(lo, hi))`,
which `scripts/03_train_heads.py` now does), and rely on the check rather than on care:
`binning.interval_mask_coverage` compares the masked bin sum against the empirical rate
and script 03 exits non-zero if they disagree. Integer properties are immune —
`CategoricalBinner` represents `[v, v+1)` exactly — which is why aromatic rings escaped.

## 4. The bug that already happened — do not reintroduce it

`target_distance(value, lo, hi)` returns the set distance to the half-open interval
`[lo, hi)`, which is **0 for `value == hi`** as well as inside. Ranking best-of-N
candidates by that distance alone therefore treated a 4-ring molecule as **tied** with a
correct 3-ring molecule for the target `[3, 4)`.

Symptom: best-of-9 hit rate 0.6914 where the binomial prediction from the base rate was
0.8147 — a 7σ discrepancy, for aromatic rings only, while cLogP reconciled fine. The
first hypothesis (correlated candidate pools) was tested and rejected before the real
cause was found.

Fix: rank with `selection_key(value, lo, hi) -> (0 if hit else 1, target_distance(...))`
so a genuine hit always outranks a boundary miss, and score error with
`target_error(value, lo, hi, integer_valued)` which knows that for integer properties
the nearest in-target value below `hi` is `hi - 1`. `INTEGER_PROPERTIES` holds the set.

Guarded by `tests/test_bestofn.py::test_best_of_n_prefers_a_real_hit_over_a_boundary_miss`,
which deliberately places 4-ring quaterphenyl **first** in the candidate pool. An earlier
version of that test used biphenyl/terphenyl, never exercised the boundary, and passed
against the buggy code. If you write a regression test for an interval bug, make sure it
fails against the old code.

**Any new property you add must be declared in `INTEGER_PROPERTIES` if it is
integer-valued.** This is the single easiest way to reintroduce the bug.

---

## 5. What was verified, so you know what to trust

- The head, the `P(y in I)` computation, and the guidance score were all checked and are
  **unaffected** by the §4 bug; it touched only best-of-N selection and error reporting.
  Rings best-of-N was re-run after the fix (0.6914 → 0.8021, reconciling with theory).
- The cached and full-recompute candidate backends are asserted numerically equal.
- 12 tests re-read each JSON artifact, format it exactly as `pilot_report.md` formats
  it, and require it to appear in the report text. Hand-transcription is the one error
  mode that no reasoning about the pipeline can rule out.
- Two claim-level assertions run against the artifacts, not the prose:
  `test_guidance_always_loses_to_compute_matched_best_of_n` and
  `test_the_aromatic_ring_crossover_is_real`.

If you change a number in `reports/pilot_report.md` by hand, the artifact tests will
fail. That is the intended behaviour: change the artifact, not the prose.

---

## 6. The follow-up experiments, in priority order

Each entry states the claim it targets and how the result should be interpreted **before
you see it**, so the analysis is not chosen after the fact.

> **PHASE-2 REPRIORITISATION, 2026-07-30.** The ranking below was written before phase 2
> and it is now partly wrong, on the strength of a measurement rather than an opinion.
>
> Phase 2 measured the head-free ceiling on one-step steering and decomposed the loss
> (`pilot_report.md` §15.1, §15.6). At λ=1 the deployed rule captures 4.8–10.9% of the
> ceiling, and that splits into **λ=1 permitting only 32–53% of the ceiling** and **our head
> collecting 12–22% of what λ=1 permits**.
>
> **Both of those are per *decoding position*.** An earlier version of this banner read them
> as end-to-end and concluded "E1 is worth at most a factor of 2, fixing the head is worth
> 5–8". That inference was not available — end-to-end lift is 20–48x per-step gain, and
> transferring the ratios linearly implies lifts above the arithmetic maximum for four of six
> properties (`pilot_report.md` §15.6, `docs/TODO.md` C22.1).
>
> **E1 has since been run, so its term is now measured rather than extrapolated**
> (`pilot_report.md` §19): tuning λ is worth **1.29–1.69x** end to end, the response is an
> inverted U peaking at λ = 2–4 rather than rising monotonically, **no λ beats
> compute-matched best-of-N** (best gap −0.0931, HBD count at λ=2), and the degenerate
> molecules predicted below **do appear**, at λ ≥ 4.
>
> So E1 was not "the single highest-leverage addition" as ranked here and by the simulated
> reviewer in `reports/ABSTRACT.md` — but that is now known from the sweep rather than
> guessed from a per-position bound. The head term remains measured only per position, so
> **E2 (on-policy calibration)** and **E3 (probe-layer sweep)** are promoted above E1 on the
> grounds that the λ term is now known to be small and known to be capped by base-policy
> destruction, a mechanism a better head does not obviously share.
>
> Phase 2 also retires **E4** (six properties, done) and **E5** (head-seed replication:
> variance is ≤0.004 AUROC, an order of magnitude below the effects compared).
>
> **UPDATE, 2026-07-30 (later the same day). E2 and E3 have now both been run, and both
> come back negative** (`pilot_report.md` §20 and §21). The promotion above was correct as a
> ranking and wrong as a forecast: neither term is cheaply available.
>
> * **E2 is closed as a NEGATIVE, not as unattempted, and this matters.** Its "cheap
>   version" — temperature-scale or isotonic-calibrate, then re-run — is, in its
>   temperature/Platt half, *exactly* a rescale of λ, proved algebraically and demonstrated
>   by two full guided runs returning the same 1,536 molecules at ε=0. Every fitted slope is
>   below 1, so **following E2's advice would have made the negative result worse and would
>   then have been read as evidence for it** — the specific failure mode E2 was written to
>   prevent. Measured cost: 0.23–0.70x of the deployed lift, six of six properties.
> * **E3 fires its own "we probed the wrong layer" branch, and R1 must be re-read** — but
>   the better layer does not steer better. Question 1 returns ARTEFACT for all six
>   properties; question 2 returns NOT MATERIAL (2 of 6, median relative −0.077, against a
>   pre-registered ≥4/6 and ≥+0.25).
>
> **The one thing E3 leaves open is end-to-end.** C17's steering measurement is per position,
> and per position is not end to end (C22.1). Guided generation at a mid-network probe point
> has not been run; it is the obvious next experiment and is filed as **C23** in
> `docs/TODO.md`.

### E1 — λ sweep ~~**and** N sweep~~: **the λ half is DONE** (`pilot_report.md` §19)

**Targets CL4 and every "no effect" claim.** ~~Currently λ=1 only, so every negative
statement is really "no effect *at λ=1*".~~ **Run 2026-07-30** on three anchor properties
(most steerable, least steerable, the pre-registered discriminating case) at λ ∈ {0.25, 0.5,
1, 2, 4, 8}, with compute-matched best-of-N and `scripts/10_quality_analysis.py` at every λ.
`--lam` was added to script 05 as specified below. **The N sweep is still outstanding.**

Scored against the pre-committed interpretation stated further down this entry: the third
branch fires (saturation below best-of-N at all λ — in fact a *decline* past the optimum,
which none of the three branches anticipated), with the second layered on top (quality
degrades, at λ ≥ 4). CL4 is strengthened; P5 is not falsified. See §19.4 for the full list of
what it changes.

A simulated NeurIPS-workshop review (recorded in `../reports/ABSTRACT.md`) independently
identified this as the single highest-leverage addition, ranking it above adding a third
property, and asked for the **N sweep too**: the deliverable is a compute–accuracy
*frontier* for both methods, not one point each. Two points cannot distinguish
"reranking is fundamentally uncompetitive at matched compute" from "reranking was not
tuned".

```bash
for lam in 0.5 1 2 4 8; do
  .venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k \
      --property clogp --out pilot_50k_guided_clogp_lam$lam
  # set lam via a config override or --lam if you add the flag
done
```

`configs/guidance.yaml` has `lam: 1.0`; add a `--lam` argument to script 05 rather than
editing the config in place, so `configs_used.json` records the value actually used.

Pre-committed interpretation:
- If hit rate rises monotonically with λ and beats compute-matched best-of-N at some λ,
  **CL4 is overturned** and the paper changes completely. Report it.
- If hit rate rises but chemical quality degrades (run script 10 at every λ), that is
  the classic reward-hacking trade-off and it is a *result*, not a failure.
- If hit rate saturates below best-of-N at all λ, CL4 is strengthened substantially.

**Run `scripts/10_quality_analysis.py` at every λ.** The CPU pilot found no quality cost
at λ=1 except under late intervention, but λ=1 keeps `log p_base` at full weight. High
λ is exactly where the literature's garbage molecules should appear. This is a
prediction; test it.

### E2 — On-policy head calibration: **DONE, and it is a NEGATIVE** (`pilot_report.md` §20)

~~**Targets CL6, which currently weakens CL4.** The head predicts 0.076 where the truth is
0.267 on guided prefixes (ECE 0.190). Fix the calibration first, *then* re-test
guidance, so "guidance fails" is not confounded with "the guidance signal is broken".~~

~~Cheap version: temperature-scale or isotonic-calibrate the head's interval probability
on held-out guided prefixes (script 08 already generates and saves them), then re-run
the guided evaluation.~~ **Run 2026-07-30, as C18.** It is a post-hoc calibration and not
retraining, so it did not consume the one permitted DAgger round — that is still spent, once,
on §9.2.1.

**Result.** The 3.5x figure re-measures to **1.69x** on the fixed head (1.35–2.09 across the
battery), and on base-policy prefixes every head is essentially calibrated (ratios 0.85–1.11,
ECE 0.0045–0.0143). Calibration itself works — ECE falls by 3–6x — and **buys nothing for
decoding**, because a monotone map preserves rank and the decoder's softmax over eight
candidates consumes ranks and spacings, never levels. AUROC is bit-identical under Platt at
four decimal places for all six properties. Worse, the correction is a **λ decrease**: fitted
Platt slopes 0.405–0.618, and end to end Platt costs 0.23–0.54x and isotonic 0.41–0.70x of
the deployed lift, at every anchor. Bin-logit temperature — the only family that can reorder
candidates — behaves like a reparametrised λ, and the temperature that *calibrates* is above
1 (the wrong direction) for four of six properties. **This entry's recommendation, followed
literally, would have made the negative result worse.**

### E3 — Probe-layer sweep, all 12 layers: **DONE** (`pilot_report.md` §21)

**Both branches below fired, in opposite directions.** *"Ring count is cleanly decodable at
some other layer"* — **yes**: probe point 3 reaches 0.8474 against trivial 0.8269, the curve
is unimodal with a mid-network peak for all six properties, and the pre-registered rule
scores ARTEFACT. So "we probed the wrong layer" is the honest conclusion for the
**predictability** half, and R1 must be re-evaluated. *But the better layer does not steer
better*: per-position steering improves on 2 of 6 properties, median relative −0.077, against
a pre-registered bar of ≥4/6 and ≥+0.25 — **NOT MATERIAL**. The AUROC-best layer is not the
steering-best layer for any of the six. End-to-end guided generation at a mid-network layer
is **not** covered by this and is filed as C23.

The original entry, kept for the record:

**Targets CL3, specifically the aromatic-ring half.** The current claim is that
token-counting beats the frozen state late. That may be a fact about **one layer**, not
about the model: the model has 12 layers of hidden size 768, and
`configs/pilot_50k.yaml` sets `hidden_layer: -1`, so every number in the report comes
from the **final** layer. Property information is often more linearly accessible in
middle layers, so this is a live possibility rather than a formality.

Requires only 12 head trainings and **no generation** — hidden states for other layers
can be re-extracted from the saved trajectories with
`generation.hidden_states_for_positions(..., layer=L)`.

If ring count is cleanly decodable at some other layer, the honest conclusion becomes
"we probed the wrong layer", and rejection criterion R1 should be re-evaluated.

### E4 — More properties (the fix for CL3's biggest weakness)

**CL3 is a double dissociation resting on two properties.** That is a pattern, not a law,
and it is the objection we cannot currently answer. Minimum credible version: 4–6
properties so each cell of the predictability x steerability 2x2 has more than one
occupant. Cheap RDKit candidates: TPSA, H-bond donor count, fraction of sp3 carbons,
rotatable-bond count, largest ring size.

Add them to `ALL_PROPERTIES` in `src/property_to_go/properties.py`, declare
integer-valued ones in `INTEGER_PROPERTIES` (see §4), and add a
`target_interval_rule` entry in `configs/guidance.yaml`. Everything downstream is
property-generic.

### E5 — Head-seed replication, 5 seeds

Tells you which small differences are real. Cheap. Do this before making any claim about
a margin under ~0.03.

### E5b — Compositional confound (small, closes a real hole)

**Targets CL2.** We standardised on sequence length and heavy-atom count. A reviewer
asked whether a *compositional* confound could survive that undetected — for example
halogen or heteroatom fraction correlating with both token frequency and cLogP. It is a
fair question and we do not currently know the answer.

`src/property_to_go/confound.py` is estimator-generic: adding heteroatom fraction (or
halogen count) as a third stratifying covariate is a small change to
`scripts/09_confound_analysis.py`. Watch `coverage` — a third covariate thins the strata
fast, and a matched estimate at 0.4 coverage is not a matched estimate.

### E6 — The expensive-oracle regime (a reframing, not a tweak)

**The strongest remaining case for the method.** Best-of-N wins partly because RDKit
oracles are free, so it can afford to evaluate every candidate. Guidance's only
structural advantage is needing a cheap neural estimate instead of a real measurement.
We chose the most unfavourable possible test case for our own method and say so in the
report.

Sharpen this before designing the experiment: the comparison is **oracle versus proxy**,
not proxy versus proxy. Best-of-N selects with `compute_properties` — the true RDKit
value on the completed molecule (`bestofn.py:104`) — while guidance only ever sees a
learned head on incomplete prefixes. `pilot_report.md` §5 scopes the claim accordingly.
Any expensive-oracle experiment has to decide deliberately whether the baseline still
gets ground truth, and match compute on **oracle calls** rather than tokens if it does
not.

To test the favourable case, match compute on **oracle calls** instead of tokens, or use
a property whose evaluation is genuinely expensive. This changes what the paper is about,
so decide deliberately rather than drifting into it.

---

## 7. Rules that carried over from the original specification

Still in force unless the project owner says otherwise:

- The base generator stays **frozen**. No fine-tuning, no LoRA, no RL, no activation
  edits, no weight changes of any kind.
- Two primary properties (cLogP, aromatic rings); molecular weight is a **diagnostic
  control only**. Adding properties for E4 extends this, but the two primaries remain
  the ones the headline claims rest on.
- No reinforcement learning. The single DAgger round in script 08 has already been used;
  it is not to be iterated.
- No explicit partial-graph models, no activation steering, no multiple generators, no
  alternative molecular serializations, no elaborate uncertainty estimation.
- Windows and target intervals are frozen **before** guided results are inspected. If you
  define new ones for new properties, write them to disk first and commit to them.
- Save configurations alongside every result. Every script calls `write_run_context`.
- **Do not force a positive conclusion**, and keep executed results clearly separated
  from unexecuted plans.
- If a follow-up overturns a reported claim, change the report. The artifact tests exist
  to make that the path of least resistance.

---

## 8. Current status, precisely

Updated 2026-07-30 after phase 2.

| | |
|---|---|
| Pilot phases 1–5 | complete, executed, reported (sections 1–10) |
| Optional DAgger round | complete (used; not to be repeated) |
| Chemical-quality analysis | complete |
| **Phase 2, the lexical-locality test** | **executed; sections 11 onward** |
| Phase-2 dataset | `outputs/pilot_50k_p2/`, a *different* 50k sample — see §1.4 |
| Phase-1 artefacts | untouched by phase 2, by construction |
| Report | `reports/pilot_report.md`, artifact-bound |
| **E1, λ half** (λ sweep + matched best-of-N + quality at every λ, three anchors) | **done; `pilot_report.md` §19** |
| **E2** (on-policy head calibration, as C18) | **done, NEGATIVE; §20** |
| **E3** (probe-layer sweep, all 13 probe points, as C17) | **done; §21 — overturns §13.1's scope, does not help steering** |
| **C23** (end-to-end guided generation at a mid-network probe point) | **in progress, 2026-07-31** |
| Follow-ups E1 (N sweep), E5b, E6 (compositional confound, expensive oracle), pooled readout | **not started** |
| Audit of the phase-2 write-up | done; six defects, none in a measured value — `docs/TODO.md` C22 |

**Two defects found in phase 2, both in phase-1 code, both now fixed and regression-tested:**
the cross-device sampling divergence (§1.4) and the target-interval-versus-binner
misalignment (§3.6). Neither changes a single reported hit rate; the second means the
pilot's cLogP target-interval AUROCs are underestimates and its cLogP "calibration
failure" was largely a misspecified target. `pilot_report.md` §11.5 has the full account.

Phase 2 also *fixed* a GPU-blocking bug in `guidance.TargetScorer` (§1.5).
