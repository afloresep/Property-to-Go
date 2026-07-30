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

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_model_contracts.py` runs against the real checkpoint and asserts the
assumptions the pilot depends on: forward determinism, right-padding exactness,
causal prefix-state equivalence, agreement of the two candidate backends, prefix
preservation and seed reproducibility of continuations, and batch-row independence.
Deselect them with `-m "not model"` if the checkpoint is unavailable.
