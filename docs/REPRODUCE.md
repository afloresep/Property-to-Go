# Reproducing the Property-to-Go pilot

Every command below is the exact command that produced the artefacts in `outputs/`.

**Provenance.** Every script writes the configuration it used next to its results
(`config_used.json` or `configs_used.json`), so no result is separable from its
settings. A full software-stack record (Python, torch, transformers, RDKit, platform,
thread count, git SHA) is written as `provenance.json` by the scripts that use
`RunDir` — steps 1 and 2 — and a session-level `outputs/provenance.json` records the
environment shared by **all** runs in this pilot, which was a single machine and a
single virtual environment throughout. Steps 3-9 do not currently write their own
`provenance.json`; that is a known gap, noted in the report, not a claim that they do.

## 0. Environment

The released GP-MoLFormer custom modelling code targets `transformers==4.32.1`. It
imports `transformers.onnx` (removed in transformers v5) and relies on
`PreTrainedModel` still carrying `GenerationMixin` (untrue from transformers 4.50).
`transformers==4.44.2` is the newest release that runs the checkpoint's own code
**unmodified**, so the pilot uses a dedicated virtual environment.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # or: .venv/bin/pip install -r requirements.lock.txt
```

Pinned model and tokenizer revisions live in `configs/model.yaml`:

| item | value |
| --- | --- |
| model | `ibm-research/GP-MoLFormer-Uniq` @ `6eca879581e2302b4e1ab07bb02908636bddb4a2` |
| tokenizer | `ibm-research/MoLFormer-XL-both-10pct` @ `361063d0ad524ef77cf39b08469f6be770dc550f` |
| RDKit | 2024.3.5 |
| torch | 2.4.1 (CPU; benchmarked faster than MPS for this 46.8M linear-attention model) |

`deterministic_eval: true` is set in the config. The released config has it `false`,
which makes `MolformerFeatureMap` redraw its orthogonal random projections on every
un-cached forward pass — that would make hidden states and prefix continuations
irreproducible. The projections are stored in the checkpoint
(`...attention.self.feature_map.weight`), so setting the flag pins the *released*
projections rather than resampling them. No weight is modified.

## 1. Compatibility spike

```bash
.venv/bin/python scripts/01_compatibility.py --n 100
```

Writes `outputs/compatibility/compatibility_report.json` with a pass/fail status for
each of the ten required operations, plus `prefix_hidden_states.npy` and
`candidate_hidden_states.npy`. Exits non-zero if any check fails.

## 2. Trajectory datasets

```bash
.venv/bin/python scripts/02_generate_trajectories.py --config pilot_10k
.venv/bin/python scripts/02_generate_trajectories.py --config pilot_50k
```

Each writes to `outputs/<name>/`:

| file | contents |
| --- | --- |
| `hidden.npy` | `(n_prefixes, 768)` frozen final-layer states |
| `features.npy` | `(n_prefixes, 24)` trivial prefix statistics |
| `prefix_meta.csv` | quartile, prefix length, terminal properties, split |
| `prefix_token_ids.json` | prefix token ids, replayed verbatim by the rollout bank |
| `trajectories.json` | completed molecules and their properties |
| `target_intervals.json` | target intervals **and base rates**, frozen here |
| `windows.json` | early / middle / late token windows, frozen here |
| `dataset_summary.json` | validity, uniqueness, base distributions, compute |

Target intervals and windows are derived from the base generator's own empirical
distributions and written before any guided molecule exists.

## 3. Value-head training

```bash
.venv/bin/python scripts/03_train_heads.py --dataset pilot_10k
.venv/bin/python scripts/03_train_heads.py --dataset pilot_50k
```

Trains `frozen_state`, `trivial` and `combined` heads under one identical recipe for
each of cLogP, aromatic ring count and molecular weight, and reports a `marginal`
floor. Writes `head_metrics.json` (overall, per prefix-position quartile, and a
paired bootstrap of frozen-vs-trivial), `predictions_<prop>.npz` and
`head_<prop>_<input>.pt`.

The scaling gate is evaluated here and printed as `SCALING GATE: PASS/FAIL`.

## 4. Repeated-continuation rollout bank

```bash
.venv/bin/python scripts/04_rollout_bank.py --dataset pilot_50k --n-prefixes 800 --n-rollouts 32
```

800 held-out prefixes balanced across the four position quartiles, 32 base-policy
continuations each, one bank shared by both properties. Writes `rollout_bank.json`
and `rollout_metrics.json` (predictability curve, rank correlation, interval Brier,
reliability).

## 5. Guided generation

```bash
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k --property clogp
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k --property aromatic_rings
```

Runs `unguided`, `throughout`, `early`, `middle`, `late` and `truncation_control`
(top-8 restriction with lambda = 0) at three seeds each, and writes
`guidance_metrics.json` including per-condition validity, uniqueness, sequence
length, heavy-atom count, processed-token compute, wall time, and a length-matched
hit rate.

## 6. Compute-matched best-of-N

```bash
.venv/bin/python scripts/06_best_of_n.py --dataset pilot_50k --property clogp --full-recompute-n 64
.venv/bin/python scripts/06_best_of_n.py --dataset pilot_50k --property aromatic_rings --full-recompute-n 64
```

Solves N from the guided run's measured tokens per returned molecule, under both
accountings (`actual` and `full_recompute`), and reports wall time separately.

`--full-recompute-n` is smaller than the 512 molecules used for the `actual` match
because the full-recompute budget implies an N roughly twenty times larger, so the
same number of returned molecules would cost twenty times the sampling. The reduced
sample is a **precision** trade, not a budget trade: each returned molecule still
receives its full matched N candidates, so the comparison stays exactly
compute-matched per molecule and only its standard error grows. Both are reported.

## 7. Figures

```bash
.venv/bin/python scripts/07_figures.py --dataset pilot_50k
```

## 7b. Length / size confound analysis

```bash
.venv/bin/python scripts/09_confound_analysis.py --dataset pilot_50k --property clogp
.venv/bin/python scripts/09_confound_analysis.py --dataset pilot_50k --property aromatic_rings
```

Analysis only — reads `molecules.json` from the guided run, generates nothing, and so
can be re-run freely. Step 5 already reports a hit rate matched on **sequence length**;
this recomputes it under four estimators (`raw`, `length`, `size` on heavy-atom count,
and `joint`), because a guided run could hold token count fixed and still win by
producing denser molecules. Each reports `coverage`, the share of unguided mass in
strata the condition actually visited; low coverage is itself evidence of distribution
shift and is reported rather than hidden.

## 7c. Chemical-quality analysis

```bash
.venv/bin/python scripts/10_quality_analysis.py --dataset pilot_50k --property clogp
.venv/bin/python scripts/10_quality_analysis.py --dataset pilot_50k --property aromatic_rings
```

Analysis only — re-scores the molecules already saved by step 5, generates nothing, and
can be re-run freely. Answers whether the property was bought with degenerate molecules:
synthetic accessibility, QED, longest acyclic carbon path, carbon fraction, ring sizes,
fragment count and formal charge, compared **within the set of molecules that hit the
target** so quality is not confounded with the property shift itself. Writes
`quality_metrics.json` plus `examples.json`, which holds random hits, worst-SA hits and
longest-chain hits per condition for eyeballing. Reported in `pilot_report.md` §10.

Re-run this at every guidance strength in any λ sweep. The pilot's null result on quality
is specific to λ = 1 into a bounded interval.

## 8. Optional single data-aggregation round

```bash
.venv/bin/python scripts/08_data_aggregation.py --dataset pilot_50k --property clogp
```

Adds guided prefixes and their terminal outcomes to the head's training data **once**
and re-runs the held-out guidance test. Run only after the main pilot; it is not part
of it.

---

# Phase 2 — the lexical-locality test

Added 2026-07-30. Hypothesis, operationalisation and pre-registered predictions P1–P6
are in [`LEXICAL_LOCALITY.md`](LEXICAL_LOCALITY.md); the checklist is
[`TODO.md`](TODO.md) section C.

**Everything below ran on an NVIDIA RTX 4090** with `device: cuda` in
`configs/model.yaml` (`dtype: float32` unchanged — see §P0.2 for why). Phase-1 artefacts
were produced on a laptop CPU and are **not** overwritten by any command here: phase-2
runs use distinct output directory names (`*_p2`, `*_p2guided_*`, `*_p2bestofn_*`) so
both sets are on disk and comparable.

## P0. Setup gates, in order

### P0.1 Environment and the phase-1 test suite

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest                      # 183 passed, before any phase-2 edit
```

### P0.2 Switch to the GPU

`device: cuda` in `configs/model.yaml`. `dtype` stays `float32`: the cached-vs-full
candidate-backend agreement is a *numerical equality* claim and the entire compute
accounting rests on it, so reduced precision is not a free change.

### P0.3 Model contracts on the GPU

```bash
.venv/bin/python -m pytest tests/test_model_contracts.py
```

`test_forward_pass_is_deterministic` underwrites reproducibility and
`test_candidate_backends_agree` underwrites the compute accounting; both pass on CUDA.
Two other tests in that file called `.numpy()` on what is now a CUDA tensor and were
fixed to `.cpu().numpy()` — device-portability defects in the tests, not in the model.

One real porting defect was in the library: `guidance.TargetScorer` held a head loaded
with `map_location="cpu"` and was handed CUDA candidate states. It now migrates the head
and the interval mask to the device the states arrive on.

### P0.3b Characterise the CPU-versus-GPU difference

```bash
.venv/bin/python scripts/13_device_equivalence.py --n-molecules 64 --timing-repeats 3
```

Loads the same pinned revision on both devices and compares weight checksums,
last-position logits, **the top-8 candidate set guided decoding actually consumes**, and
whether the same seed draws the same molecules. Also measures whether wall time reproduces
on this machine before any timing number is quoted. Writes
`outputs/device_equivalence/device_equivalence.json`; reported in `pilot_report.md`
§11.2 and §11.7.

Run it with nothing else on the GPU, or the timing measurement is meaningless.

### P0.4 Regenerate the arrays git excludes, and check the frozen artefacts

```bash
.venv/bin/python scripts/02_generate_trajectories.py --config pilot_50k --out pilot_50k_gate_d
```

Written to a **separate** directory on purpose, so the tracked `windows.json` and
`target_intervals.json` are still on disk to diff against. Those two were frozen before
any guided result was inspected, and re-deriving them on different hardware could in
principle move them.

The check is now permanent rather than a one-off diff:
`tests/test_report_matches_artifacts.py::test_the_frozen_target_intervals_did_not_move`
and `::test_the_frozen_windows_did_not_move` pin the committed values as literals.

### P0.5 Reproduce a known number

```bash
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k \
    --property aromatic_rings --seeds 101 202 303 \
    --conditions unguided throughout --out pilot_50k_gpucheck_guided_aromatic_rings
```

Bit-identical reproduction is not available and was never possible: `torch.manual_seed`
seeds the CUDA generator, which draws a different stream from the CPU generator, so
`torch.multinomial` makes different choices at identical seeds. The comparison is
therefore distributional, over all three seeds. Reported in `pilot_report.md` §11.1.

## P1. Design lock — before any new guided run

No command. `properties.PREDICTED_LOCALITY_ORDER` is the pre-registration and is pinned
as a literal by
`tests/test_properties.py::test_the_pre_registration_is_pinned_literally`, so editing it
to match observed data fails a test instead of passing quietly.

The four staged properties are wired in: `properties.ALL_PROPERTIES`,
`bestofn.INTEGER_PROPERTIES` (the boundary bug of `HANDOFF.md` §4 — now derived from
`properties.DISCRETE_PROPERTIES` by a test rather than trusted to memory), and
`target_interval_rule` entries in `configs/guidance.yaml`.

Target intervals are set by a **rule**, not by hand, so they could be committed before
anyone had seen the four new base distributions: `quantile_band [0.85, 0.95)` for
continuous properties (base rate 0.10 by construction, so all four are base-rate matched
to each other) and `quantile_value q = 0.90` for counts (interval `[v, v+1)`, one count
unit wide). Applying the count rule to aromatic rings returns 3, the value the pilot
picked by hand.

## P2. Build the phase-2 dataset, inheriting the frozen intervals

```bash
.venv/bin/python scripts/02_generate_trajectories.py --config pilot_50k \
    --out pilot_50k_p2 --inherit-intervals outputs/pilot_50k/target_intervals.json
```

A **separate dataset**, `pilot_50k_p2`, not a rebuild of `pilot_50k`. Two reasons, both
consequences of P0.4: the regenerated sample is a different draw, so mixing it with
phase-1 artefacts would silently pair row indices across two different molecule sets; and
`outputs/pilot_50k/` must keep the frozen JSONs the report is bound to.

`--inherit-intervals` copies the pilot's three intervals **verbatim** and derives only the
four new ones from this run's base distribution. What this run *would* have derived is
written to `target_intervals_provenance.json` alongside, together with each inherited
interval's empirical base rate on *this* sample — which differs from the recorded one and
is needed for any later comparison to be correct.

Two independent runs of this command produce bit-identical arrays (`pilot_report.md` §11.4).

## P3. Heads for all seven properties, three head seeds each

```bash
.venv/bin/python scripts/03_train_heads.py --dataset pilot_50k_p2 --config pilot_50k \
    --head-seeds 1234 2345 3456 --out pilot_50k_heads_p2
```

`--config pilot_50k` because the dataset directory name and the config name now differ.

`--head-seeds` also seeds head *initialisation*, which the pilot did not: `MLPHead` draws
its Linear init from the ambient torch RNG and `train_head`'s own `manual_seed` runs
afterwards, so the pilot controlled batch shuffling but not initialisation. Omitting the
flag keeps the pilot's single-seed path bit-identical, so `outputs/pilot_50k_heads/` is
untouched and every phase-1 number stands.

This run also **exits non-zero if any target interval is not a union of bins**, which is
the check that would have caught the §11.5 defect.

### P3b. Quantify what the interval-mask defect cost

```bash
.venv/bin/python scripts/03_train_heads.py --dataset pilot_50k_p2 --config pilot_50k \
    --head-seeds 1234 --legacy-interval-mask --out pilot_50k_heads_p2_legacymask
.venv/bin/python scripts/14_interval_mask_impact.py
```

`--legacy-interval-mask` reproduces the pilot's un-aligned binner. The comparison holds
dataset, initialisation seed and recipe fixed, so the only difference is the mask.
Reported in `pilot_report.md` §11.6. These heads are **not** used downstream.

## P4. Rollout bank with the phase-2 battery

```bash
.venv/bin/python scripts/04_rollout_bank.py --dataset pilot_50k_p2 \
    --heads pilot_50k_heads_p2 --n-prefixes 800 --n-rollouts 32
```

A new bank rather than an extension of the old one, because `rollout_bank.json` stores
property *values* and not the rollout SMILES, so the four new properties cannot be
computed from it after the fact. The pilot's CPU bank stays in place; this one doubles as
a GPU replication of the aromatic-ring crossover. 25,600 continuations in 84 s.

## P5. Steering headroom — the head-free, λ-free ceiling

```bash
.venv/bin/python scripts/11_steering_headroom.py --dataset pilot_50k_p2 \
    --heads pilot_50k_heads_p2 --n-prefixes 400 --n-rollouts 16
```

400 held-out prefixes balanced across position quartiles × top-8 candidates × 16
base-policy rollouts. No new inference machinery: the extended prefixes `x_{<=t} + a_i`
are passed to `generation.continue_from_prefixes` as ordinary prefixes, which means
`test_candidate_backends_agree` still covers the numerics underneath the measurement.

Prefixes are drawn with seed 7777, different from the Phase 4 bank's 4242, so headroom and
the predictability curve are independent samples rather than two views of the same prefixes.

Writes `headroom_metrics.json` and `headroom_arrays.npz` (per-prefix, per-candidate `mu`,
`p_hit`, raw and permutation-null spreads in both units, and the head's own interval
probabilities).

## P6. Guidance and compute-matched best-of-N, all six properties

```bash
for prop in aromatic_rings hbd_count rotatable_bonds tpsa clogp qed; do
  .venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k_p2 \
      --heads pilot_50k_heads_p2 --property "$prop"
  .venv/bin/python scripts/06_best_of_n.py --dataset pilot_50k_p2 --property "$prop" \
      --full-recompute-n 64
  .venv/bin/python scripts/09_confound_analysis.py --dataset pilot_50k_p2 --property "$prop"
  .venv/bin/python scripts/10_quality_analysis.py --dataset pilot_50k_p2 --property "$prop"
done
```

Six conditions × three seeds × 512 molecules per property, exactly the pilot's protocol.
Output directories are the scripts' own defaults for this dataset
(`pilot_50k_p2_guided_<prop>`, `pilot_50k_p2_bestofn_<prop>`, …), which is why no phase-1
directory can be touched: the dataset name differs.

`scripts/run_phase2.sh` replays this whole chain unattended, by stage.

### P6b. Does the interval-mask defect change guided decoding?

```bash
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k_p2 \
    --heads pilot_50k_heads_p2_legacymask --property clogp \
    --conditions unguided throughout --out pilot_50k_p2_guided_clogp_legacymask
```

Same dataset, same seeds, same conditions — only the head's interval mask differs. A
softmax over candidates is invariant to an additive constant, so a uniform shrink of every
candidate's target probability should cancel almost exactly; this measures whether it does.
`pilot_report.md` §15.4.

## P7. The scatter and the pre-registered predictions

```bash
.venv/bin/python scripts/12_locality_scatter.py --dataset pilot_50k_p2 \
    --headroom pilot_50k_p2_headroom --heads pilot_50k_heads_p2 \
    --guided-suffix guided --bestofn-suffix bestofn
```

Reads only; generates nothing. Writes `locality_metrics.json`: per-property locality
score and steerability, the P1/P2/P3/P6 rank correlations, and the pre-registered
ordering against the measured one.

The interval on the P1 correlation propagates *measurement* noise in each property's two
coordinates (resampling guidance seeds, and headroom prefixes) with the six properties
held fixed. It does **not** cover the uncertainty from having chosen six hand-picked
properties, which is larger and is not estimable from these data.

## P9. The λ sweep, and chemical quality at every λ

`docs/TODO.md` C10 and C12. Three anchor properties, five new λ values; λ = 1 is the
central test's own run and is **not** regenerated.

```bash
bash scripts/run_phase2.sh lambda
```

which is, per property in `aromatic_rings hbd_count qed` and per λ in `0.25 0.5 2 4 8`:

```bash
tag=lam$(echo "$lam" | tr '.' 'p')
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k_p2 \
    --heads pilot_50k_heads_p2 --property "$prop" --lam "$lam" \
    --conditions unguided throughout --out pilot_50k_p2_${tag}_guided_${prop}
.venv/bin/python scripts/06_best_of_n.py --dataset pilot_50k_p2 --property "$prop" \
    --guided pilot_50k_p2_${tag}_guided_${prop} --accounting actual \
    --out pilot_50k_p2_${tag}_bestofn_${prop}
.venv/bin/python scripts/10_quality_analysis.py --dataset pilot_50k_p2 --property "$prop" \
    --guided pilot_50k_p2_${tag}_guided_${prop} --out pilot_50k_p2_${tag}_quality_${prop}
```

then the read-only assembly:

```bash
.venv/bin/python scripts/15_lambda_sweep.py --dataset pilot_50k_p2
```

Four things about the invocation are deliberate.

- **`--lam` rather than editing `configs/guidance.yaml`.** The override is folded into the
  config dict before the run, so `configs_used.json` and `guidance_metrics.json["lambda"]`
  both record the value actually used and cannot disagree with it. Script 15 refuses to
  assemble a run whose recorded λ differs from the directory it was expected in.
- **Only `unguided` and `throughout`.** `truncation_control` is λ = 0 by definition and the
  three window conditions are a separate axis. `unguided` *is* regenerated at every λ even
  though it cannot depend on λ: it is a bug alarm, and it reproduces its central-test value
  of 0.1785 (aromatic rings) exactly at every λ.
- **`--accounting actual` only.** Under `full_recompute` the budget is N ≈ 200 independent
  base-policy draws against a base rate of 0.08–0.17, so best-of-N misses with probability
  under 1e-7. It measured exactly 1.0000 for all six properties at λ = 1 and cannot do
  anything else at any λ. `n_candidates_solved` and `base_rate` are still written to every
  `bestofn_metrics.json` so the saturation argument can be checked rather than assumed.
- **The stage is idempotent.** A combination whose metrics JSON already exists is skipped,
  so the sweep can be resumed after an interruption without discarding finished work.

## P10. C18 — can the head be fixed? (`pilot_report.md` §20)

Stages are resumable and idempotent; run in this order. `scripts/17_run_c18.sh all` runs the
whole chain, and a completed guided run is never redone, so it can be resumed after an
interruption without discarding work.

```bash
bash scripts/17_run_c18.sh prediction      # writes the pre-committed prediction FIRST
bash scripts/17_run_c18.sh offpolicy       # re-measure the off-policy gap, fit calibrators
bash scripts/17_run_c18.sh heads           # train the wide / focused / wide_focused readouts
bash scripts/17_run_c18.sh perposition     # per-position capture for every arm
bash scripts/17_run_c18.sh identity        # the lambda-equivalence identity, end to end
bash scripts/17_run_c18.sh e2e_calibrated  # 4 calibration arms x 3 anchors
bash scripts/17_run_c18.sh e2e_heads       # 2 retrained readouts x 3 anchors
bash scripts/17_run_c18.sh bestofn         # matched best-of-N, once per distinct N
bash scripts/17_run_c18.sh summary         # assemble the end-to-end table
.venv/bin/python -m pytest tests/test_head_calibration.py -p no:cacheprovider   # 34 passed

# section 20.5.1 only: the decoder-optimal bin temperature, run as a lambda result
.venv/bin/python scripts/17_guided_calibrated.py --property aromatic_rings \
    --arm bin_temperature --bin-temperature 0.4 --out c18_guided_binT0p4_aromatic_rings
bash scripts/17_run_c18.sh bestofn && bash scripts/17_run_c18.sh summary
```

`ANCHORS` and `HEAD_VARIANTS` are environment overrides on the driver; the defaults are
the three anchors and the two readouts that ever improved per-position capture on an
anchor (`focused` improved none and is reported at the per-position stage only).

**Order matters in one place only, and it is load-bearing**: `prediction` must run before
`offpolicy`, because `test_the_prediction_was_written_before_the_measurements` compares
file mtimes. Total cost 22,903,499 processed tokens.

## P11. C17 — the probe-layer sweep (`pilot_report.md` §21)

Run after the phase-2 chain (P0–P7). It needs `outputs/pilot_50k_p2/`,
`outputs/pilot_50k_heads_p2/`, `outputs/pilot_50k_p2_headroom/` and
`outputs/pilot_50k_p2_locality/` to exist. It generates **no molecules** and does not touch
any frozen artefact.

```bash
# 21.1  every probe point for the phase-2 prefixes, in one forward pass per batch.
#       ~8 GB of float32 under outputs/c17_layer_states_pilot_50k_p2/.
#       Exits non-zero unless probe point 12 is bit-identical to the dataset's hidden.npy.
.venv/bin/python scripts/16_extract_layer_states.py --dataset pilot_50k_p2

# 21.2  13 probe points x 6 properties x 3 head seeds. CPU-bound, ~2 h on the 4090 box
#       while sharing it. Writes partial_L<L>.json after each probe point; re-running
#       the same command resumes from those. --no-resume forces a full retrain.
#       Run it detached: a shell timeout will otherwise kill a multi-hour job.
setsid nohup .venv/bin/python scripts/16_probe_layer_sweep.py \
    --dataset pilot_50k_p2 --out c17_probe_layers \
    > outputs/c17_probe_layers_sweep.log 2>&1 &

# 21.3  what each layer is worth for steering. One forward pass over 3,200 extended
#       prefixes; no generation. Reuses phase 2's rollouts, nulls and capture set.
.venv/bin/python scripts/16_layer_steering_value.py --dataset pilot_50k_p2

# 21.4  depth curves.
.venv/bin/python scripts/16_layer_figures.py

.venv/bin/python -m pytest tests/test_probe_layers.py -p no:cacheprovider
```

Extraction cost 2,205,784 processed tokens for all 13 probe points — thirteen separate
forward passes would have cost 28,675,192 for the same 13 arrays.

**Artefact sizes, and one open decision.** `outputs/c17_layer_states_*/` is **7.5 GB** of
`layer<L>/hidden.npy` and is already excluded by `.gitignore`'s `outputs/**/hidden.npy` rule;
only its small JSON files are visible to git, which is the right behaviour and needs no
change. `outputs/c17_probe_layers/` is **89 MB**, of which **81 MB is the 78 seed-1234 head
checkpoints** (`head_<prop>_frozen_state_L<L>.pt`) plus a 6 MB `partial_trivial_probs.npz`
resume file. Existing `.pt` files under `outputs/` are tracked deliberately, so these would
be committed by default. They are needed only to re-run `scripts/16_layer_steering_value.py`
without retraining, and the sweep regenerates them in ~2 h. **`.gitignore` has not been
edited** — to avoid carrying 81 MB, add `outputs/c17_probe_layers/*.pt` and
`outputs/c17_probe_layers/partial_*` to it; the metrics JSON, which is what the tests and
§21 read, is 1.5 MB and must stay.

### Two additive flags on existing scripts

Both default to the existing behaviour, so no executed artefact changes:

```bash
# script 02: store extra probe points alongside hidden.npy, at zero extra token cost.
#   Omitted -> byte-identical output to before (verified on a 64-trajectory run).
#   -1 is normalised against the layer count, so `--layers 0 6 -1` writes
#   hidden_layer0.npy and hidden_layer6.npy and does NOT rewrite hidden.npy.
.venv/bin/python scripts/02_generate_trajectories.py --config <cfg> --layers 0 6 -1

# script 03: train the heads on an alternative frozen-state array.
#   Omitted -> reads <dataset>/hidden.npy exactly as before.
#   The path used is recorded in head_metrics.json as `hidden_file`.
.venv/bin/python scripts/03_train_heads.py --dataset <ds> --hidden-file hidden_layer6.npy
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_model_contracts.py` runs against the real checkpoint and asserts the
assumptions the pilot depends on: forward determinism, right-padding exactness,
causal prefix-state equivalence, agreement of the two candidate backends, prefix
preservation and seed reproducibility of continuations, and batch-row independence.
Deselect them with `-m "not model"` if the checkpoint is unavailable.

## P8. The scatter figure set

```bash
.venv/bin/python scripts/07_figures.py --dataset pilot_50k_p2 --heads pilot_50k_heads_p2 \
    --rollouts pilot_50k_p2_rollouts --guided-suffix guided --bestofn-suffix bestofn
```

110 PNGs in `outputs/pilot_50k_p2_figures/`.

## Phase-2 runtime, for planning

RTX 4090, sharing the GPU with an unrelated job for most of the run, so these are upper
bounds rather than clean benchmarks.

| stage | wall time |
| --- | --- |
| 50k trajectory generation | 239 s |
| 50k hidden-state extraction (2.21 M tokens) | 26 s |
| heads, 7 properties x 3 inputs x 3 seeds (63 trainings) | ~13 min |
| heads, legacy-mask control, 1 seed | ~6 min |
| rollout bank, 25,600 continuations | 84 s |
| steering headroom, 51,200 continuations (2.43 M tokens) | ~11 min |
| guided generation, per property (6 conditions x 3 seeds) | 5-8 min |
| compute-matched best-of-N, per property (both accountings) | 6-8 min |
| confound + quality analysis, per property | < 1 min |
| full test suite, 366 tests incl. real-checkpoint contracts | 10 s |

Phase 2 total is roughly **2.5 hours** of compute against the pilot's ~6.5 hours on CPU,
while doing three times as many properties and adding the headroom measurement.
