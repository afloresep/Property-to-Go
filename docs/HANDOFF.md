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

---

## 1. Environment — the part that will waste your time if you skip it

```bash
cd property-to-go
uv venv && uv pip install -e .          # or: python -m venv .venv && pip install -e .
.venv/bin/python -m pytest              # expect 177 passed
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

### E1 — λ sweep **and** N sweep: build the frontier, not two points (highest value)

**Targets C4 and every "no effect" claim.** Currently λ=1 only, so every negative
statement is really "no effect *at λ=1*".

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
  **C4 is overturned** and the paper changes completely. Report it.
- If hit rate rises but chemical quality degrades (run script 10 at every λ), that is
  the classic reward-hacking trade-off and it is a *result*, not a failure.
- If hit rate saturates below best-of-N at all λ, C4 is strengthened substantially.

**Run `scripts/10_quality_analysis.py` at every λ.** The CPU pilot found no quality cost
at λ=1 except under late intervention, but λ=1 keeps `log p_base` at full weight. High
λ is exactly where the literature's garbage molecules should appear. This is a
prediction; test it.

### E2 — On-policy head calibration

**Targets C6, which currently weakens C4.** The head predicts 0.076 where the truth is
0.267 on guided prefixes (ECE 0.190). Fix the calibration first, *then* re-test
guidance, so "guidance fails" is not confounded with "the guidance signal is broken".

Cheap version: temperature-scale or isotonic-calibrate the head's interval probability
on held-out guided prefixes (script 08 already generates and saves them), then re-run
the guided evaluation. Note that this is a *post-hoc* calibration, not retraining, so it
does not count as the one permitted DAgger round.

### E3 — Probe-layer sweep, all 12 layers

**Targets C3, specifically the aromatic-ring half.** The current claim is that
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

### E4 — More properties (the fix for C3's biggest weakness)

**C3 is a double dissociation resting on two properties.** That is a pattern, not a law,
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

**Targets C2.** We standardised on sequence length and heavy-atom count. A reviewer
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

| | |
|---|---|
| Pilot phases 1–5 | complete, executed, reported |
| Optional DAgger round | complete (used; not to be repeated) |
| Chemical-quality analysis | complete (added 2026-07-30) |
| Tests | 177 passing |
| Follow-ups E1–E6 | **not started**; new work, not part of the pilot |
| Report | `reports/pilot_report.md`, artifact-bound |

Nothing is running. Nothing is half-finished.
