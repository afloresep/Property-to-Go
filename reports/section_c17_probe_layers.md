# Section C17 — the probe-layer sweep

Draft section, written to be merged into `reports/pilot_report.md`. Structured as report
subsections so it can be renumbered and inserted without rewriting. Nothing in this file
edits an existing report claim; where my data contradicts one, it is flagged under
"Contradictions with the existing report" and left for the owner to merge.

Experiment C17 (`docs/TODO.md`, promoted second in the Week-3 list; `docs/HANDOFF.md` §6
E3). It targets §8.3 — *"Only one probe point was tested"* — and the aromatic-ring half of
C3, §13.1.

---

## C17.0 Pre-registration

**Written and committed to disk before any layer other than the final one was extracted,
trained on, or scored.** It is not revised below; §C17.5 scores the executed result
against it verbatim, including where it fails.

### C17.0.1 What is swept

`res.hidden_states` of GP-MoLFormer has **13 entries**: index 0 is the embedding output
and indices 1–12 are the 12 transformer layers. `configs/model.yaml`'s `hidden_layer: -1`
selects index 12, which is every number in the report to date.

The sweep is over **all 13 probe points**. Index 0 is a control, not a candidate
explanation — it is the token embedding of the last prefix token with no contextual
computation at all — but it is *counted in the multiplicity correction* rather than
excluded from it, because excluding a probe point after seeing it is exactly the
degree of freedom this pre-registration exists to remove.

Battery: the six properties of `properties.PREDICTED_LOCALITY_ORDER`. Input:
`frozen_state` only. `trivial` does not depend on the layer, so it is trained once per
property and reused; its values must reproduce `outputs/pilot_50k_heads_p2/`. Head seeds:
**1234 / 2345 / 3456**, the same three as phase 2, seeding initialisation as well as batch
shuffling. Dataset: `outputs/pilot_50k_p2/`, unchanged, intervals unchanged, splits
unchanged.

### C17.0.2 Validity gate, checked before any comparison is read

Two identity checks. If either fails the sweep is reported as invalid rather than
interpreted:

1. The re-extracted states at probe index 12 must equal `outputs/pilot_50k_p2/hidden.npy`
   **exactly** (bit-identical; §11.4 says within-device determinism holds).
2. The probe-index-12 heads must reproduce §13's table — mean target AUROC and mean NLL
   per property to **4 decimal places**. That is a check on the training recipe, not on
   the model: if my per-layer trainer is not the same trainer script 03 ran, every
   cross-layer comparison below is measuring my refactor.

### C17.0.3 Primary metric, and what happens if the metrics disagree

**Primary: held-out target-interval AUROC**, mean over the three head seeds, on the
phase-2 test split. It is the metric §13.1's claim is stated in, and it is the quantity
guided decoding consumes (`P(y ∈ I | prefix + a)`).

**Secondary: held-out NLL**, same rows, same seeds.

**If they disagree** — i.e. the AUROC criterion below fires at some layer but that layer's
NLL is worse than the `trivial` head's — the verdict is recorded as **metric-dependent**
and *neither* branch is claimed. I will not choose the metric that produces the more
interesting conclusion. Both columns are printed for every layer regardless.

### C17.0.4 Question 1 — is the aromatic-ring crossover a final-layer artefact?

Phase 2 (§13.1): frozen state **0.7878 ± 0.0023**, trivial **0.8269 ± 0.0019**, a deficit
of **−0.0391**. Head-seed sd is ≤ 0.0041 across the whole battery (§13.2), so the sd of a
3-seed mean is ≈ 0.0024 and §13.2 licenses margins down to ~0.01.

Let `A(L)` be the mean frozen-state target AUROC for aromatic rings at probe point `L`,
and `A_triv = 0.8269` the trivial head's.

| verdict | fires when |
| --- | --- |
| **ARTEFACT** — the crossover is a fact about layer 12, not about the representation | some `L` with `A(L) ≥ A_triv + 0.010` **and** that layer's paired-bootstrap CI for (frozen − trivial) AUROC excludes 0 at the multiplicity-corrected level of C17.0.6 **and** its NLL is no worse than trivial's |
| **TIE AT THE BEST LAYER** — inconclusive | `max_L A(L)` lands in `[A_triv − 0.004, A_triv + 0.010)` |
| **REPRESENTATION** — the crossover survives the whole depth of the model; §13.1 is strengthened | `max_L A(L) < A_triv − 0.004` |

0.004 is one head-seed sd (§13.2's maximum, 0.0041, rounded down); 0.010 is §13.2's own
stated safe-margin floor and ≈ 4 sd of a 3-seed mean. The asymmetry is deliberate: the
claim currently in the report is the negative one, so overturning it is required to clear
a higher bar than confirming it.

### C17.0.5 Question 2 — does some layer close part of the head gap?

§15.6: at λ=1 our head collects **11.9–21.5%** of what λ=1 permits. The layer is evaluated
for steering **without re-selecting on steering**, to keep the multiplicity honest:

1. For each property, `L*(prop) = argmax_L A(L)` — chosen by *prediction*, question 1's
   metric, before any steering quantity is computed.
2. Recompute `our_head_gain` and `our_head_share_of_the_lambda1_optimum` exactly as
   `scripts/12_locality_scatter.py` computes them, substituting the layer-`L*` head's
   `head_q`. Everything else — `candidate_base_logprobs`, `p_hit_<prop>`, `n_valid`,
   `null_available_<prop>` from `outputs/pilot_50k_p2_headroom/headroom_arrays.npz`, λ, ε,
   the 267-prefix capture set — is held fixed. **No new generation.**
3. **The layer sweep counts as a material attack on the head gap** iff `our_head_gain` at
   `L*` exceeds its value at probe point 12 for **at least 4 of the 6** properties **and**
   the median relative improvement across the six is **≥ 25%**. Otherwise it is reported
   as *not* an effective attack on §15.6's head term, whatever the AUROC table says.

The steering recomputation is run for **all 13 layers**, not only `L*`, because it costs
one forward pass over 3,200 extended prefixes; but only the `L*` column scores the
criterion above, and the full table is descriptive.

### C17.0.6 Multiplicity, committed before the numbers exist

13 probe points × 6 properties = **78 comparisons**. Three rules, all fixed here:

1. **The aromatic-ring question is one pre-specified family of 13.** A "layer L beats
   trivial" claim inside it must clear a Bonferroni-corrected paired bootstrap: the
   `1 − 0.05/13 = 99.615%` interval for the AUROC gain must exclude zero. (Implemented as
   the 0.1923%/99.808% quantiles of the paired bootstrap difference, two-sided.)
2. **No isolated spikes.** For any property, "probe point L is genuinely better than the
   final layer" requires (a) `A(L) − A(12) ≥ 0.010` and (b) both neighbours `L−1` and
   `L+1` to satisfy `A(·) − A(12) ≥ 0.005`. A one-layer spike with non-improving
   neighbours is reported as noise, not as a finding. Depth curves are smooth; a genuine
   representational effect should be visible at more than one depth. (Probe point 12 has
   only one neighbour, and is the reference, so the rule is vacuous there; probe point 0
   has one neighbour and the rule is applied to that one.)
3. **The per-seed values are published for every cell** so the noise floor is legible
   rather than asserted.

### C17.0.7 What would falsify what

- If **REPRESENTATION** fires, §13.1's crossover is strengthened: the claim becomes "no
  linear-plus-MLP readout of *any* layer of this frozen model recovers aromatic ring count
  as well as counting `c` and ring-closure tokens does", and §8.3's limitation is closed
  for this property.
- If **ARTEFACT** fires, §13.1 must be rewritten to "the final layer does not linearly
  expose ring count", which is much weaker, and rejection criterion R1 has to be
  re-evaluated (`docs/HANDOFF.md` §6 E3 says so explicitly).
- If question 2's criterion fails, the cheapest attack on §15.6's head term is spent, and
  C18 (calibration / readout capacity) carries it alone. That is a *reportable* negative
  and it is not to be softened by quoting the AUROC table instead.

---

*(Results below this line were written after the run. Everything above was on disk
before it.)*

---

## C17.1 What was run

Three steps, none of which generates a molecule.

| step | script | writes | cost |
| --- | --- | --- | --- |
| extract every probe point for the phase-2 prefixes | `scripts/16_extract_layer_states.py` | `outputs/c17_layer_states_pilot_50k_p2/layer{0..12}/hidden.npy` | one forward pass per batch |
| train a `frozen_state` head at each probe point, 6 properties × 3 seeds | `scripts/16_probe_layer_sweep.py` | `outputs/c17_probe_layers/` | CPU head training only |
| recompute the λ=1 steering quantities with each layer's head | `scripts/16_layer_steering_value.py` | `outputs/c17_layer_steering/` | one forward pass over 3,200 extended prefixes |
| depth curves | `scripts/16_layer_figures.py` | `outputs/c17_figures/` | reads artefacts only |

The dataset is `outputs/pilot_50k_p2/` exactly as phase 2 left it. `windows.json` and
`target_intervals.json` were **not** regenerated and not read for anything but their
frozen values; no property's binner, split or interval differs from §13's by construction,
because the prefix rows are the same rows.

**The extraction is a replay, not a regeneration.** `prefix_meta.csv` is written
trajectory-major and its `prefix_len` column *is* the position index script 02 extracted
at, so the `(sequence, positions)` pairs are regrouped from `trajectories.json` rather
than re-derived. The script asserts the trajectory-major ordering rather than assuming it.

### C17.1.1 Thirteen probe points cost the tokens of one

`output_hidden_states=True` already returns all 13 entries from the single forward pass
the pipeline runs anyway, so the sweep needs **no additional processed tokens at all**
relative to the extraction script 02 performs. This is the reason C17 was cheap, and it is
stated in tokens rather than in seconds for the reason §11.7 gives.

### C17.1.2 Provenance of the run, including a mid-sweep restart

Stated in full because a reader who sees `scripts/16_probe_layer_sweep.py` edited between
the start and the end of a sweep should assume the worst unless the timeline is shown.

1. The sweep was launched, ran for about an hour, completed roughly three of thirteen
   probe points, and was then **killed by a shell timeout in the harness running it** —
   not by a failure in the script. It had written **no** `probe_layer_metrics.json` and no
   partial results, only head checkpoints.
2. Between the kill and the relaunch the only commands run against it were
   `ls outputs/c17_probe_layers/*.pt | wc -l`, `ps` and `date`. **No metric, no AUROC, no
   NLL and no verdict from any layer was read, because none had been written.** The
   printed per-layer lines went to a pipe that the kill discarded.
3. The edit made in response was to add per-layer checkpointing (`partial_L<L>.json`,
   `partial_trivial*.json`) and a `--no-resume` switch, so an interrupted run is
   inspectable and restartable. It touches persistence only: no threshold, no metric, no
   training call and no decision rule was changed. `git diff` shows the change.
4. The relaunch **deleted the orphaned checkpoints and retrained from scratch**, detached
   from the shell so a harness timeout could not repeat the outcome.

Which layers were resumed from a partial rather than trained in one pass, and the
validity-gate outcome under the edited code, are reported in §C17.2 — the gate is not
inherited from the pre-restart run, it is re-run.

The pre-registration in §C17.0 was written, saved and staged in git **before the first
launch** and has not been edited since; its hash is in the git index.

**Which probe points were resumed from a partial: none.** The relaunch found no
`partial_*.json` in the output directory, so all 13 probe points and the trivial baseline
were trained in a single uninterrupted pass under the edited code. The run log contains
zero `resumed probe point` lines. Total 7691.6 s of CPU head training, sharing the
machine with C18.

## C17.2 The two validity gates

Both are checked **under the edited code**, in the relaunched run, and neither is
inherited from the interrupted one.

**Gate 1 — the replay reproduces the dataset's own hidden states.** Probe point 12 of
`scripts/16_extract_layer_states.py` against `outputs/pilot_50k_p2/hidden.npy`, all
199300 rows: **bit-identical, maximum absolute difference exactly 0**. So the sweep's
reference column is literally the array every number in §13 was computed from, and no
cross-layer difference below can be an artefact of re-extraction.

The extraction took **2205784 processed tokens** for all 13 probe points. One forward pass
per batch returns every layer, so 13 separate passes would have cost 28675192 —
**13× more tokens for the same 13 arrays**. Wall time is recorded (56.6 s) but is not the
cost unit, for the reason §11.7 gives.

**Gate 2 — the per-layer trainer is script 03's trainer.** The probe-point-12 heads,
trained by `scripts/16_probe_layer_sweep.py`, against
`outputs/pilot_50k_heads_p2/head_metrics.json`:

| property | §13 AUROC | probe point 12 AUROC | Δ | §13 NLL | probe point 12 NLL | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.7878 | 0.7878 | 0.0 | 1.0445 | 1.0445 | 0.0 |
| HBD count | 0.7799 | 0.7799 | 0.0 | 1.1071 | 1.1071 | 0.0 |
| rotatable bonds | 0.7820 | 0.7820 | 0.0 | 1.8084 | 1.8084 | 0.0 |
| TPSA | 0.7369 | 0.7369 | 0.0 | 2.7527 | 2.7527 | 0.0 |
| cLogP | 0.7913 | 0.7913 | 0.0 | 2.7154 | 2.7154 | 0.0 |
| QED | 0.7348 | 0.7348 | 0.0 | 2.7715 | 2.7715 | 0.0 |

Not "to four decimals" — **identical to every printed digit, on all six properties, for
both metrics, and for the layer-independent `trivial` head as well**. The pre-registration
asked for 4 dp; the run cleared it by a wide margin. The two gates together mean the only
thing that varies across the columns below is the probe point.

## C17.3 Result — the depth curve

Held-out target-interval AUROC, mean ± sd over head seeds 1234 / 2345 / 3456. Probe point
0 is the embedding output; 1–12 are the twelve transformer layers; **12 is the layer every
number in `pilot_report.md` uses**. `trivial` does not depend on the layer.

| probe point | aromatic rings | HBD count | rotatable bonds | TPSA | cLogP | QED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (embedding) | 0.6159 | 0.5565 | 0.5327 | 0.5406 | 0.5783 | 0.5228 |
| 1 | 0.8320 | 0.7550 | 0.7578 | 0.7355 | 0.7590 | 0.7145 |
| 2 | 0.8422 | 0.7873 | 0.7930 | 0.7512 | 0.7788 | 0.7375 |
| 3 | **0.8474** | 0.8123 | 0.8073 | 0.7568 | 0.7889 | 0.7467 |
| 4 | 0.8458 | **0.8226** | **0.8158** | 0.7586 | 0.7979 | **0.7532** |
| 5 | 0.8389 | 0.8206 | 0.8134 | **0.7595** | **0.7998** | 0.7526 |
| 6 | 0.8280 | 0.8125 | 0.8117 | 0.7565 | 0.7988 | 0.7494 |
| 7 | 0.8216 | 0.8035 | 0.8125 | 0.7509 | 0.7966 | 0.7478 |
| 8 | 0.8112 | 0.7991 | 0.8086 | 0.7467 | 0.7947 | 0.7461 |
| 9 | 0.8045 | 0.8003 | 0.8042 | 0.7462 | 0.7978 | 0.7414 |
| 10 | 0.8026 | 0.7895 | 0.7958 | 0.7431 | 0.7945 | 0.7399 |
| 11 | 0.7931 | 0.7841 | 0.7876 | 0.7422 | 0.7937 | 0.7393 |
| **12 (the probed layer)** | 0.7878 | 0.7799 | 0.7820 | 0.7369 | 0.7913 | 0.7348 |
| `trivial` prefix stats | 0.8269 | 0.7098 | 0.7431 | 0.7389 | 0.7516 | 0.7158 |

Seed spread is ≤ 0.0057 in every cell (largest: HBD count at probe point 8), consistent
with §13.2's ≤ 0.0041 and an order of magnitude below the depth effect.

The same shape in NLL (nats, lower is better; means over the three seeds):

| probe point | aromatic rings | HBD count | rotatable bonds | TPSA | cLogP | QED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.3333 | 1.3486 | 2.1708 | 2.9920 | 3.0119 | 3.0069 |
| 3 | **0.8658** | 1.0295 | 1.7360 | 2.6698 | 2.7092 | 2.7483 |
| 4 | 0.8743 | **0.9837** | 1.6975 | **2.6631** | **2.6863** | **2.7348** |
| 5 | 0.8946 | 0.9990 | **1.6954** | 2.6812 | 2.6878 | 2.7468 |
| 12 | 1.0445 | 1.1071 | 1.8084 | 2.7527 | 2.7154 | 2.7715 |
| `trivial` | 0.9001 | 1.2571 | 1.9154 | 2.7014 | 2.7863 | 2.8064 |

**AUROC and NLL do not disagree anywhere that matters**, so C17.0.3's tie-break never
fires. The AUROC argmax and the NLL argmin coincide for aromatic rings (both 3), HBD count
(both 4) and QED (both 4), and differ by one layer for rotatable bonds (4 vs 5), TPSA
(5 vs 4) and cLogP (5 vs 4) — within a region where the curve is flat to ~0.001 AUROC.
The full per-layer NLL table is in `probe_layer_metrics.json`; the rows above are the ones
the argument uses.

### C17.3.1 The shape is the result, not the argmax

Two facts hold for **all six properties simultaneously**, which is what makes this
something other than six independent lucky maxima:

1. **The curve is unimodal with a mid-network peak.** It rises steeply from the embedding,
   peaks at probe point **3, 4 or 5** — never later than 5, never earlier than 3 — and
   then declines to probe point 12. Across the six properties and the eight probe points
   from the peak to the end there is not a single reversal larger than seed noise.
2. **The final layer is the worst probe point on the whole descending stretch**, for every
   property: probe point 12 is the minimum over probe points *peak*–12 in all six columns.
   Once the decline starts it does not stop before the output.

Two stronger statements are tempting and **false**. Both are recorded because earlier
drafts of this section made them and the artefact-binding tests caught them, which is what
those tests are for.

- *"The final layer is the worst of all twelve contextual layers, and one transformer
  block already beats twelve."* True for **aromatic rings alone** — 0.8320 at probe point 1
  against 0.7878 at 12. For the other five, probe point 1 is the *worst* contextual layer
  and the final layer is comfortably above it (HBD count 0.7550 vs 0.7799; rotatable bonds
  0.7578 vs 0.7820; TPSA 0.7355 vs 0.7369; cLogP 0.7590 vs 0.7913; QED 0.7145 vs 0.7348).
  The claim that the network ends below where it started is specific to the one property
  whose information is most nearly lexical.
- *"The final layer is the minimum over probe points 3–12."* True for five of six and
  false for **cLogP**, whose probe point 3 (0.7889) sits just *below* probe point 12
  (0.7913). cLogP rises more slowly and falls less far than the others — the same fact
  that makes it the one property failing C17.0.6's material margin in §C17.4.2.

Figures: `outputs/c17_figures/auroc_by_probe_point.png`,
`nll_by_probe_point.png`, `steering_gain_by_probe_point.png`.

The natural reading is that the last layers specialise towards next-token prediction and
discard property-relevant information that is present mid-stack — a well-documented
pattern in language models, here measured rather than assumed. This report does not test
that mechanism and does not claim it; what is measured is the curve.

## C17.4 Result — question 1, scored

### C17.4.1 The aromatic-ring crossover is a final-layer artefact

The pre-registered rule fires **ARTEFACT**, and it fires on every one of its three
required conditions rather than on the AUROC arm alone:

| condition (C17.0.4) | required | measured |
| --- | --- | --- |
| best layer's AUROC over `trivial` | ≥ +0.010 | best layer **3**, **0.8474** vs **0.8269**, margin **+0.0205** |
| paired-bootstrap CI, Bonferroni-corrected to α = 0.05/13 | excludes 0 | **[+0.0142, +0.0284]**, mean +0.0210 |
| NLL at the best layer not worse than `trivial` | ≤ 0.9001 | **0.8658** |
| no-isolated-spike (C17.0.6 rule 2), vs probe point 12 | both neighbours ≥ +0.005 | probe point 2 **+0.0544**, probe point 4 **+0.0580** |

**So §13.1's claim as currently written does not survive.** "The frozen state loses to
token counting for aromatic rings" is true of layer 12 and false of layers 1–5. The
correct statement is the weaker one `docs/HANDOFF.md` §6 E3 anticipated:

> **the model's final layer does not linearly expose aromatic ring count as well as
> counting `c` and ring-closure tokens does, while its middle layers do.**

This is the outcome that is inconvenient for the project, and it is reported as the
pre-registration required. Three qualifications, none of which rescues the original claim:

- The margin is **+0.0205 AUROC**, roughly five head-seed standard deviations. It is real
  but it is not large; the honest summary is "the middle of the network is modestly better
  than counting, the end of it is clearly worse", not "the representation obviously
  contains ring count".
- The crossover **still exists at the layer the pilot probed** and it still replicated
  across two independent 50k samples (§13.1). Nothing about the phase-1 or phase-2
  measurement was wrong; the *scope* of the conclusion drawn from it was.
- **R1 should be re-evaluated**, as `docs/HANDOFF.md` §6 E3 says it should if this
  happened.

### C17.4.2 The same thing happens to TPSA, and to the other four

TPSA is the other property where §13 reports the frozen state losing to surface statistics
(on NLL, with AUROC indistinguishable). It behaves identically: best layer **5**,
**0.7595** against `trivial` **0.7389**, margin **+0.0206**, Bonferroni CI
[+0.0082, +0.0341], NLL 2.6812 against 2.7014, neighbours +0.0217 and +0.0196. So **both**
of §13.3's "the frozen state loses" cases are final-layer facts.

For the four properties where the final layer already beat `trivial`, the rule's ARTEFACT
label is vacuous — it was already satisfied at probe point 12 — so what matters there is
the size of the improvement over the probed layer:

| property | best layer | best AUROC | `trivial` | margin over `trivial` | gain over probe point 12 | neighbours support it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| aromatic rings | 3 | 0.8474 | 0.8269 | +0.0205 | +0.0596 | yes |
| HBD count | 4 | 0.8226 | 0.7098 | +0.1129 | +0.0427 | yes |
| rotatable bonds | 4 | 0.8158 | 0.7431 | +0.0727 | +0.0338 | yes |
| TPSA | 5 | 0.7595 | 0.7389 | +0.0206 | +0.0226 | yes |
| cLogP | 5 | 0.7998 | 0.7516 | +0.0482 | +0.0085 | **no — fails the material margin** |
| QED | 4 | 0.7532 | 0.7158 | +0.0374 | +0.0184 | yes |

**cLogP is the exception and it is reported as one.** Its gain over the probed layer is
+0.0085, below C17.0.6's 0.010 material margin, so by the pre-registered rule cLogP shows
**no genuine improvement** over the final layer even though its curve has the same shape.
Its neighbours do support it (+0.0066, +0.0075); it is the margin it fails, not the
smoothness. cLogP is the property whose information is most nearly preserved to the top of
the stack, which is consistent with §5.5's finding that it is the one property the frozen
state dominates at every position — but that is an observation, not a test.

### C17.4.3 What this closes, and what it does not

§8.3 says: *"Whether an earlier layer, a larger head, or a different pooling would close
the aromatic-ring gap is untested."* Two of those three are now tested, by two experiments
that answer in opposite directions:

- **an earlier layer: yes, it closes the gap** (this section);
- **a larger head: no, it does not** — C18 measures a 4×-wider readout (hidden dim 1024) at
  the final layer moving seed-matched target AUROC by at most +0.0019 on any property, with
  aromatic rings still at 0.7886 against `trivial` 0.8267.

Together these are stronger than either alone: the gap at layer 12 is not a capacity
limit, and the information is present five layers earlier. **The caveat that keeps this
honest: C18's width test is at the final layer only, so capacity is excluded *there*. A
capacity × depth interaction — whether a wider head at probe point 3 would do better still
— is untested by either experiment.** Pooling remains untested by both.

## C17.5 Result — question 2, scored, and it fails

### C17.5.1 The pre-registered criterion

C17.0.5 fixes the protocol: choose each property's layer by **prediction**
(`L* = argmax_L AUROC`, from §C17.3, with no reference to any steering quantity), then
recompute the λ=1 steering quantities with that layer's head. Everything else is phase 2's:
the same 400 prefixes and their stored candidate token ids, the same `p_hit`, `n_valid`
and permutation nulls from `headroom_arrays.npz`, the same 267-prefix capture set, the same
λ = 1 and ε. **No generation.** The recomputation cost 76024 processed tokens for all 13
probe points, and `molecules_returned` is 0.

Consistency gates, before any comparison: at probe point 12 the recomputed `head_q`
matches the stored `head_q_<prop>` to **maximum absolute difference 0.0** on all six
properties, and the recomputed `our_head_gain` matches
`outputs/pilot_50k_p2_locality/locality_metrics.json` to **0.0**. So the layer columns
differ only by the layer.

| property | L* (by AUROC) | `our_head_gain` at probe point 12 | at L* | relative | share of the λ=1 optimum, 12 → L* |
| --- | ---: | ---: | ---: | ---: | --- |
| aromatic rings | 3 | +0.01466 | +0.01303 | -0.111 | 20.8% → 18.5% |
| HBD count | 4 | +0.00449 | +0.00470 | +0.045 | 11.9% → 12.4% |
| rotatable bonds | 4 | +0.00308 | +0.00536 | +0.737 | 13.5% → 23.4% |
| TPSA | 5 | +0.00646 | +0.00618 | -0.044 | 21.5% → 20.6% |
| cLogP | 5 | +0.00709 | +0.00547 | -0.228 | 17.9% → 13.8% |
| QED | 4 | +0.00281 | -0.00003 | -1.011 | 14.8% -> -0.2% |

**Improved on 2 of 6 properties; median relative improvement -0.077. The criterion
required ≥ 4 of 6 and a median ≥ +0.25. It is NOT MATERIAL, and C17.0.7 binds this section
to report that as the negative it is.**

So: **the probe-layer sweep is not an effective attack on the head term of §15.6.** The
cheapest available attack has been spent and it did not work. A better-predicting head is
not, on this evidence, a better-steering head.

### C17.5.2 Why the AUROC table must not be quoted instead

The temptation here is obvious and C17.0.7 forecloses it: the AUROC table is a clean
positive and the steering table is a flat negative, and quoting the first in place of the
second would be exactly the substitution the pre-registration exists to prevent.

It is worth being precise about *why* they diverge, because the reason is itself the
finding. If the layer is chosen **after** seeing the steering numbers, the picture looks
good — aromatic rings 20.8% → 29.9% at probe point 6, HBD count 11.9% → 20.0% at 6,
rotatable bonds 13.5% → 30.4% at 3. But those layers are **not** the ones that predict
best, and picking them is selection on the outcome across 13 candidates per property. The
pre-registered protocol exists precisely to price that in, and when it is priced in the
effect disappears. The per-layer steering gains are reported in full in
`layer_steering_metrics.json` and plotted in `steering_gain_by_probe_point.png` so a reader
can see both readings.

The mechanism is visible in the numbers: target-interval AUROC is a **ranking** statistic,
while guidance consumes `log P(y ∈ I | prefix + a)` inside a softmax against `log p_base`.
A head can rank held-out prefixes better and still produce a *flatter* set of eight
candidate probabilities at a decoding position, which is what moves the sampling
distribution. QED is the extreme case: its best-predicting layer produces a per-position
gain of -0.00003, i.e. nothing at all. Improving discrimination and improving the
magnitude of the score spread are different objectives, and C17 shows they can come apart.

### C17.5.3 What this does and does not say about end-to-end steering

Everything in §C17.5 is a **per-position** quantity, with the rest of the sequence left to
the base policy. §15.6 and `docs/TODO.md` C22.1 record why per-position ratios must not be
transferred linearly to end-to-end lift.

**Had the criterion passed, the correct statement would still have been "per position
only".** It failed, so the point is moot in the favourable direction — but the converse
matters and should be stated: **this section does not prove that a mid-network head would
fail end to end.** That would need fresh guided generation at probe point 3–5, which
nobody has run and which C17 explicitly did not need. It is the obvious follow-up and it
is the honest limit of this result.

What it does establish is that the *cheap* version of the fix — swap the layer, keep
everything else — does not buy per-position steering, and that the property that
distinguishes a good probe layer (discrimination) is not the property guidance consumes
(score spread at a decoding position).

Adjacent evidence, from C18 rather than from C17, points the same way: C18's best
retrained readout bought 1.09× at one anchor and 1.00× and 0.76× at the other two end to
end, and **no arm anywhere beat compute-matched best-of-N** (best advantage −0.2238). Two
independent attacks on the head term — depth here, capacity and calibration there — both
came back approximately empty.

## C17.6 Contradictions with the existing report

Flagged for the owner to merge. Nothing in `reports/pilot_report.md` was edited by C17.

1. **§13.1 is too strongly scoped.** "That is a replication to three decimal places of the
   pilot's most-defended claim" remains true as a statement about layer 12, but the
   surrounding claim — that the frozen state does not carry aromatic ring count as well as
   token counting does — is **contradicted** at probe points 1–5. Suggested repair: keep
   the measurement, restate the conclusion as a fact about the final layer, and cite this
   section. §5.5's "a learned readout of a 768-dimensional state does not" and §8.1's "the
   specification's kill-test is met for this property" carry the same over-scope.
2. **§13.3 generalises from two losers.** Both of its "the frozen state loses" cases
   (aromatic rings, TPSA) reverse at a middle layer, so the inference that a fixed
   token-statistics vector "can nearly count" those properties better than the model
   represents them does not hold of the model, only of its output layer.
3. **§8.3 is partly closed and should be updated rather than deleted.** "An earlier layer"
   is now tested and the answer is yes; "a larger head" is tested by C18 and the answer is
   no; "a different pooling" remains untested.
4. **Nothing in §15, §16 or §19 is contradicted.** Every steering, best-of-N, λ-sweep and
   quality result stands exactly as reported, and C17.5 adds a negative that is consistent
   with all of them.

One thing C17 explicitly does **not** contradict: the pre-registered
`PREDICTED_LOCALITY_ORDER` and the P1–P6 verdicts are untouched. C17 changes which layer
the predictability half is measured at; it does not re-run the locality scatter, and the
steering coordinate of that scatter is unchanged because C17.0.5's criterion failed.

## C17.6.1 A note on what is and is not test-bound

Every number in §C17.1–§C17.5 that comes from C17 is re-read from its JSON artefact and
required to appear in this text by `tests/test_probe_layers.py`, in the style of
`tests/test_report_matches_artifacts.py`. Two of those tests are deliberately written as
*tripwires on the data* rather than on the prose — the no-isolated-spike table and the
question-2 failure — so a re-run that changes the finding fails a test instead of leaving
this section standing unchallenged. Both caught real over-claims during drafting: an
earlier version of §C17.3.1 asserted that the final layer is the worst of all twelve
contextual layers and that probe point 1 beats probe point 12 for every property; both are
false for five of six, and are now recorded as false.

**The C18 figures quoted in §C17.4.3 and §C17.5.3 (+0.0019, 0.7886 vs 0.8267, 1.09×/1.00×/
0.76×, −0.2238) are not bound by `tests/test_probe_layers.py`.** They are C18's artefacts,
quoted with attribution, and are bound by C18's own tests. C17 does not read C18's outputs.

## C17.7 Limitations

- **One head architecture, one pooling.** Every probe point uses the same two-layer MLP on
  the last-position state. Pooling over positions is untested here and in C18.
- **Capacity × depth is untested**, as §C17.4.3 says.
- **The steering result is per position.** End-to-end guided generation at a middle layer
  has not been run.
- **The layer sweep is over probe points, not over probe *methods*.** A linear probe, a
  different regulariser or a different readout depth could move the curve; only depth was
  varied.
- **The 13 probe points are one family, corrected as one.** The Bonferroni correction in
  C17.0.6 covers the 13 layers within a property. Across the six properties the results are
  not independent tests of independent hypotheses — the same six-property battery is used
  throughout the project — and no further correction is applied, deliberately, because the
  claim being made is about the *shape shared by all six* rather than about any one of them.

---

## Commands to add to `docs/REPRODUCE.md`

A section "C17 — the probe-layer sweep", to run after the phase-2 chain (P0–P7). It needs
`outputs/pilot_50k_p2/`, `outputs/pilot_50k_heads_p2/`, `outputs/pilot_50k_p2_headroom/`
and `outputs/pilot_50k_p2_locality/` to exist. It generates **no molecules** and does not
touch any frozen artefact.

```bash
# C17.1  every probe point for the phase-2 prefixes, in one forward pass per batch.
#        ~8 GB of float32 under outputs/c17_layer_states_pilot_50k_p2/.
#        Exits non-zero unless probe point 12 is bit-identical to the dataset's hidden.npy.
.venv/bin/python scripts/16_extract_layer_states.py --dataset pilot_50k_p2

# C17.2  13 probe points x 6 properties x 3 head seeds. CPU-bound, ~2 h on the 4090 box
#        while sharing it. Writes partial_L<L>.json after each probe point; re-running
#        the same command resumes from those. --no-resume forces a full retrain.
#        Run it detached: a shell timeout will otherwise kill a multi-hour job.
setsid nohup .venv/bin/python scripts/16_probe_layer_sweep.py \
    --dataset pilot_50k_p2 --out c17_probe_layers \
    > outputs/c17_probe_layers_sweep.log 2>&1 &

# C17.3  what each layer is worth for steering. One forward pass over 3,200 extended
#        prefixes; no generation. Reuses phase 2's rollouts, nulls and capture set.
.venv/bin/python scripts/16_layer_steering_value.py --dataset pilot_50k_p2

# C17.4  depth curves.
.venv/bin/python scripts/16_layer_figures.py

.venv/bin/python -m pytest tests/test_probe_layers.py -p no:cacheprovider
```

**Artefact sizes, and one decision for the owner.** `outputs/c17_layer_states_*/` is
**7.5 GB** of `layer<L>/hidden.npy` and is already excluded by `.gitignore`'s
`outputs/**/hidden.npy` rule; only its small JSON files are visible to git, which is the
right behaviour and needs no change. `outputs/c17_probe_layers/` is **89 MB**, of which
**81 MB is the 78 seed-1234 head checkpoints** (`head_<prop>_frozen_state_L<L>.pt`) plus a
6 MB `partial_trivial_probs.npz` resume file. Existing `.pt` files under `outputs/` are
tracked deliberately, so these would be committed by default. They are needed only to
re-run `scripts/16_layer_steering_value.py` without retraining, and the sweep regenerates
them in ~2 h. **I have not edited `.gitignore`** — if you would rather not carry 81 MB,
add `outputs/c17_probe_layers/*.pt` and `outputs/c17_probe_layers/partial_*` to it; the
metrics JSON, which is what the tests and this section read, is 1.5 MB and must stay.

Two flags were added to existing scripts, both additive and both defaulting to the
existing behaviour, so no executed artefact changes:

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
