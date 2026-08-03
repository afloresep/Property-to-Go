# Section C23 — end-to-end guided decoding with a mid-network head

Draft section, written to be merged into `reports/pilot_report.md`. Structured as report
subsections so it can be renumbered and inserted without rewriting. Nothing in this file
edits an existing report claim; where the data contradicts one it is flagged under
"Contradictions with the existing report" and left for the owner to merge.

Experiment C23. It targets the limit C17 states about itself in §C17.5.3:

> **this section does not prove that a mid-network head would fail end to end.** That
> would need fresh guided generation at probe point 3–5, which nobody has run.

Nobody has run it. That is C23, and only that.

---

## C23.0 Pre-registration

**Written and saved to disk before any C23 output directory existed.** It is not revised
below; §C23.5 scores the executed result against it verbatim, including where it fails.

A verbatim copy of this subsection is frozen at
`outputs/c23_prereg/C23.0_preregistration.md`, with the SHA-256 of this whole file at
the moment of freezing recorded in `outputs/c23_prereg/prereg_lock.json`, so that the
"prereg precedes the run" claim is checkable after later subsections are appended and
this file's mtime has moved. `tests/test_layer_end_to_end.py` asserts the frozen copy is
still a verbatim substring of this file.

### C23.0.0 What C23 is, and the inference it exists to block

C17 swept all 13 probe points and found, for all six properties, that a linear-plus-MLP
head on a **mid-network** hidden state predicts the finished molecule's property better
than one on the final layer (best probe points 3/4/4/5/5/4; aromatic rings AUROC 0.8474 at
probe point 3 against 0.7878 at 12, `trivial` 0.8269). C17's second question — does the
better layer *steer* better — came back **NOT MATERIAL**: 2 of 6 properties improved,
median relative change −0.0774, and the AUROC-best layer was the steering-best layer for
**no** property.

That measurement is **per decoding position**. `docs/TODO.md` C22.1 records what happens
when a per-position quantity is read as an end-to-end one; this project has already made
that error once and withdrawn the conclusion. C23 therefore measures the end-to-end
quantity directly and claims nothing that is not measured end to end.

**C23 changes exactly one thing** relative to §16/§19: which entry of
`res.hidden_states` the guidance head reads, and which checkpoint it is. The generator is
frozen, the windows and target intervals are `outputs/pilot_50k_p2/{windows,
target_intervals}.json` verbatim and are **not** re-derived, the serialization is SMILES,
the seeds and molecule counts are §19's, and no DAgger round is run.

### C23.0.1 Validity gate — checked before any experimental arm is generated

`scripts/05_guided_generation.py` did not expose `layer`. C23 adds exactly two arguments,
`--layer` (default `None` → `-1`, the pre-edit value) and `--head-file` (default `None` →
`heads_dir/head_<prop>_frozen_state[_seed<s>].pt`, the pre-edit path), records both in
`guidance_metrics.json` and `configs_used.json`, and changes nothing else.
`tests/test_layer_end_to_end.py::test_script_05_defaults_reproduce_the_pre_edit_code_path`
asserts the defaults reproduce the pre-edit code path.

**The gate.** Re-run the `unguided` and `throughout` conditions of
`outputs/pilot_50k_p2_guided_aromatic_rings/` — the λ=1 central-test run whose
`throughout` hit rate is **0.4735** and whose `unguided` hit rate is **0.1785** — at
λ=1, seeds 101/202/303, 512 molecules per condition per seed, adding
`--layer 12 --head-file c17_probe_layers/head_aromatic_rings_frozen_state_L12.pt`.
Output: `outputs/c23_gate_L12_lam1_aromatic_rings/`.

**"Matches" means, and this is the whole gate:**

1. every one of the six per-(condition, seed) hit rates equals the reference to **all
   printed digits**, i.e. maximum absolute residual exactly **0.0**; and
2. the returned molecules are **identical string for string**, all 3072 of them, in
   order.

Criterion 2 is stronger than criterion 1 and is the one that actually rules out a
compensating pair of errors. The measured residual is reported in §C23.2, as a number;
"matches" is not an acceptable report.

**Which head the gate uses, and which head the arms use.** The C17 probe-point-12
checkpoint is **not byte-identical** to the deployed
`outputs/pilot_50k_heads_p2/head_aromatic_rings_frozen_state.pt` — it carries an extra
`probe_point: 12` key, so the files differ in size (1066844 vs 1066796 bytes) and in
SHA-256. It **is** numerically identical where it matters: every tensor of the state dict
agrees to maximum absolute difference **0.0**, the binner dictionary compares equal, and
`head_seed` is 1234 in both. The same holds for HBD count and QED. Checked before the
gate was designed, on the checkpoints only, with no guided run inspected.

The gate therefore uses **the C17 probe-point-12 checkpoint**, as the reviewer specified,
because that is the checkpoint family the experimental arms come from and the gate should
exercise the same loading path. The experimental arms use the C17 checkpoints
`outputs/c17_probe_layers/head_<prop>_frozen_state_L<L>.pt` for the same reason: they are
the only mid-layer heads that exist, they were trained by C17's trainer, and C17's own
gate 2 showed that trainer reproduces script 03's numbers to every printed digit.

**If the gate fails, C23 stops and reports the failure.** No experimental arm is
generated, and no arm already generated (there will be none, because the gate runs first)
is interpreted.

### C23.0.2 Anchors

**aromatic_rings, hbd_count, qed** — the three properties that have both a full §19 λ
envelope and a compute-matched best-of-N at every λ. Fixed here. None is dropped later
for not fitting, and none is added later.

Their deployed-layer reference values, verified against the artefacts before this
pre-registration was written (`pilot_50k_p2_{,lam*_}{guided,bestofn}_<prop>`):

| property | unguided | λ=1 | §19-optimal λ | lift at optimum | compute-matched best-of-N | best advantage, any λ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic_rings | 0.1785 | 0.4735 | 2 | 0.5579 | 0.8294 (N=9) | −0.2715 |
| hbd_count | 0.0837 | 0.2988 | 2 | 0.4303 | 0.5234 (N=9) | −0.0931 |
| qed | 0.0896 | 0.1908 | 4 | 0.2607 | 0.5436 (N=8) | −0.2829 |

### C23.0.3 Layers — chosen by C17's frozen numbers, never by a C23 outcome

Two layers per property: the **AUROC-best** probe point
(`outputs/c17_probe_layers/probe_layer_metrics.json`) and the **per-position
steering-best** probe point (`outputs/c17_layer_steering/layer_steering_metrics.json`,
`our_head_gain`). Both tables were written by C17 and are read here without
recomputation.

| property | AUROC-best | per-position steering-best | deployed | experimental layers |
| --- | ---: | ---: | ---: | --- |
| aromatic_rings | 3 | 6 | 12 | **3, 6** |
| hbd_count | 4 | 6 | 12 | **4, 6** |
| qed | 4 | **12** | 12 | **4** only |

QED's per-position steering-best probe point *is* the deployed layer 12, so QED
contributes **one** experimental arm, not two. That is a consequence of C17's frozen
numbers, not a choice made here, and it is not revised if QED turns out to be
inconvenient. **Five (property, layer) combinations.**

### C23.0.4 λ

Three λ per combination, run in this order:

1. **λ = 1** for every arm, first — direct comparability with every other number in the
   report.
2. **the §19-optimal λ** for that property: aromatic_rings 2, hbd_count 2, qed 4.
3. **one extra λ per property, named here before any arm is run: aromatic_rings 0.5,
   hbd_count 0.5, qed 2** — i.e. **one grid step below the deployed layer's own
   optimum**, on §19's grid {0.25, 0.5, 1, 2, 4, 8}.

The reason for choosing *below* rather than above is a prediction, not a hedge, and it is
falsifiable (see C23.0.8, P-C). C17's `layer_steering_metrics.json` records
`mean_head_q_spread_across_candidates`: at the deployed layer it is 0.1992
(aromatic_rings), 0.1321 (hbd_count) and 0.1029 (qed), and at the C23 layers it is
0.3013 / 0.3013 (probe points 3 / 6), 0.1680 / 0.1664 (4 / 6) and 0.1279 (4) — **1.26× to
1.51× larger**. A larger spread in `q` is a larger spread in `λ log(q + ε)`, so at matched
λ a mid-layer head pushes the softmax *harder*. §19 showed the λ response is an inverted U
whose downslope is base-policy destruction. If the mid-layer head is effectively a larger
λ, its optimum must sit at or below the deployed layer's, and the interesting extra grid
point is the one below, not above.

**λ may not be selected after seeing the lift.** These nine (property, λ) pairs and five
(property, layer) combinations give **15 experimental arms**, fixed here:

| property | layers | λ |
| --- | --- | --- |
| aromatic_rings | 3, 6 | 0.5, 1, 2 |
| hbd_count | 4, 6 | 0.5, 1, 2 |
| qed | 4 | 1, 2, 4 |

If compute runs out, arms are dropped **from the end of the priority order** given by the
reviewer (λ=1 at the AUROC-best layer, then λ=1 at the steering-best layer, then the
§19-optimal λ, then quality, then matched best-of-N, then the extra λ), and §C23.5 says
which were not run. Nothing is dropped because of what it showed.

### C23.0.5 Protocol

Identical to §19's λ-sweep arms and to §16's central test, so the numbers are directly
comparable:

- conditions `unguided` and `throughout` only, as `scripts/run_phase2.sh lambda` used;
- seeds **101 / 202 / 303**; **512** molecules per condition per seed;
- dataset `outputs/pilot_50k_p2/`; `windows.json` and `target_intervals.json` **read
  verbatim, never re-derived**;
- `top_k_candidates: 8`, `eps: 1e-6`, `candidate_backend: cached`, `batch_size: 64`;
- model revision `6eca879581e2302b4e1ab07bb02908636bddb4a2`, tokenizer revision
  `361063d0ad524ef77cf39b08469f6be770dc550f`, `deterministic_eval: true`, `dtype:
  float32`, `device: cuda`, transformers 4.44.2, `.venv/bin/python`;
- `write_run_context` + `configs_used.json` beside every result; the `--layer` and
  `--head-file` values recorded in both;
- compute reported as `processed_tokens_actual` and `processed_tokens_full_recompute`,
  never as wall-clock (§11.7).

**`unguided` is regenerated at every arm as a bug alarm.** It cannot depend on the layer
or on λ, so it must reproduce its central-test value **exactly**: aromatic_rings
**0.1785**, hbd_count **0.0837**, qed **0.0896**. Any deviation is a bug and the arm is
reported as invalid rather than interpreted.

### C23.0.6 Decision rules, both directions

Three verdicts. They are evaluated in the order given; the third fires iff neither of the
first two does.

**Multiplicity.** 15 experimental arms. Every interval below is a **two-sided
Bonferroni-corrected 95% interval at α = 0.05/15 = 0.003333**, i.e. the 0.1667% and
99.8333% quantiles of the bootstrap distribution, with **n_boot = 10000** and a fixed
bootstrap seed recorded in the artefact.

**The bootstrap.** Two arms generate *different* molecules, so there is no molecule-level
pairing to exploit and none is claimed. The resampling is **seed-stratified**: within
each of the three seeds, 512 molecules are drawn with replacement from each arm
independently; each arm's three per-seed hit rates are averaged with equal weight; the
statistic is the difference of those two seed-matched means. Every comparison is
seed-matched by construction — same three seeds, same 512 per seed, never a comparison
against a differently-seeded mean.

**Seed noise.** The three per-seed differences give a standard error
`sem = sd/sqrt(3)`. "By more than the seed noise" means `|mean difference| > 2 × sem`.

---

**Rule A — the layer improves guidance.** Fires iff, for **≥ 2 of the 3 properties**,
that property's **best mid-layer arm** exceeds the **deployed-layer arm at the matched
λ** (`pilot_50k_p2_{,lam*_}guided_<prop>`, same seeds, same n), **and** for each such
property the difference exceeds the seed noise **and** its Bonferroni-corrected paired
bootstrap CI excludes 0 **and** the arm is not disqualified by C23.0.7.

**Rule B — the headline falsification.** Fires iff **some** (property, layer, λ) arm's
`throughout` hit rate exceeds its **own compute-matched best-of-N** hit rate, where N is
re-solved by `scripts/06_best_of_n.py` from *that arm's own* measured
`tokens_per_molecule_actual` and `unguided` base tokens/molecule — not inherited from the
deployed-layer run — with the corrected CI on (guided − best-of-N) excluding 0, and with
the arm not disqualified by C23.0.7.

**Token matching is part of the rule, not an assumption.** For every arm the realised
`processed_tokens_actual` per returned molecule is reported, together with the ratio to
the seed-matched deployed-layer arm at the same λ, and the solved N. An arm whose
realised tokens per molecule exceeds the matched deployed-layer arm's by **more than 5%**
does not count as a win for Rule A or Rule B, because a lift bought with more tokens is
not a lift; it is reported separately with its ratio.

Beating compute-matched best-of-N would falsify C4 and P5 and is the only outcome that
changes the paper's verdict. The deployed layer's best advantage at any λ is −0.2715
(aromatic_rings), −0.0931 (hbd_count) and −0.2829 (qed); those are the gaps that have to
be closed.

**Rule C — null.** If neither A nor B fires, the result is a **null**, and that is the
expected outcome given C17 question 2. It is reported as a null, in those words, and as
**the third and final cheap fix being spent** — after C17 (depth, per position) and C18
(capacity and calibration). It is not softened by quoting C17's AUROC table, by quoting a
per-position number, or by reporting the best arm as if the selection were free.

### C23.0.7 Quality is part of the result

`scripts/10_quality_analysis.py` is run on **every** arm, guided hits against base-policy
hits, exactly as §16.5 and §19.3 do. Validity, uniqueness and mean content length are
reported for every arm.

**Stated before any number is seen: a lift that comes with a validity drop is not a
win.** Operationally, an arm is **disqualified** from firing Rule A or Rule B if its mean
validity is more than **0.01** below the seed-matched deployed-layer arm at the same λ,
or if its mean uniqueness is more than **0.01** below it. A disqualified arm is still
reported, with its hit rate and its validity, and labelled as disqualified — it is not
deleted.

§19.3 found the degeneracy cost appears at λ ≥ 4 on the deployed layer. If C23.0.8's P-C
is right, mid-layer arms should show it at lower λ.

### C23.0.8 Predictions

**Headline prediction: Rule C (null) fires.** Neither A nor B. Reasons, in order of
weight: (i) C17 measured the per-position steering value of exactly these layers and
found 2/6 improving with median relative −0.0774; (ii) C18 attacked the same head term
from capacity and calibration and came back with 1.09× / 1.00× / 0.76× end to end and no
arm beating best-of-N; (iii) the gaps Rule B must close are 0.09–0.28 while the largest
end-to-end factor any head or λ intervention has bought in this project is 1.7×, which on
hbd_count at λ=2 (0.4303) would need a further 1.22× just to reach 0.5234 — conceivable
for hbd_count alone, and not for the other two.

Four sub-predictions, each falsifiable **independently of the headline**:

- **P-A (size).** At λ=1, every one of the five mid-layer arms lands within **±0.05**
  absolute hit rate of its deployed-layer arm. Falsified by any arm outside that band,
  in either direction.
- **P-B (does C17's per-position steering ranking transfer?).** For aromatic_rings and
  hbd_count, the **steering-best** layer (6 and 6) beats the **AUROC-best** layer (3 and
  4) end to end at λ=1 — for both properties. This is the prediction C17 could not make,
  and it is falsified if the AUROC-best layer wins for either property. It says nothing
  about whether either beats the deployed layer, so it survives the headline null.
- **P-C (the λ curve shifts left).** Because the mid-layer heads' `q` spread is 1.26–1.51×
  larger, at matched λ they destroy more of the base policy. Concretely: for at least two
  of the five combinations, mean **validity** at the §19-optimal λ is lower than the
  deployed layer's validity at that λ by more than 0.005; and for at least one of the four
  count arms, the mid-layer arm's advantage over the deployed layer is larger at λ=0.5
  than at λ=2. Falsified if validity is not lower anywhere and the advantage does not
  shrink with λ.
- **P-D (the gap).** Across all 15 arms the best advantage over compute-matched
  best-of-N stays **below −0.05** — still a clear loss — and hbd_count at λ=2 is the
  closest arm, as it is on the deployed layer. Falsified by any arm reaching −0.05 or
  better, which is weaker than Rule B and would be worth reporting on its own.

If the headline null fires and P-B also fails, the honest summary is that C17's
per-position steering column does not even *rank* layers correctly for end-to-end
steering, which is a stronger negative than C23 was designed to find and would have to be
reported as such.

---

*(Results below this line were written after the run. Everything above was on disk
before it, frozen at `outputs/c23_prereg/`.)*

## C23.1 What was run

Everything below is **end to end**: molecules generated by the frozen generator under
guided decoding, scored by RDKit on the finished string. No per-position quantity appears
in this section, and none of C17's per-position numbers is transferred to it.

| step | script | writes | cost |
| --- | --- | --- | --- |
| validity gate: replay the deployed λ=1 aromatic-ring run at `--layer 12` | `scripts/05_guided_generation.py` (`--layer`, `--head-file`) | `outputs/c23_gate_L12_lam1_aromatic_rings/` | 3072 molecules |
| 15 experimental arms | `scripts/18_layer_end_to_end.py` → script 05 | `outputs/c23_guided_L<L>_lam<λ>_<prop>/` | 15 × 3072 molecules |
| compute-matched best-of-N, N re-solved per arm | script 06 | `outputs/c23_bestofn_*/` | 15 matched baselines |
| chemical quality on every arm | script 10 | `outputs/c23_quality_*/` | reads molecules only |
| assemble and score the decision rules | `scripts/18_summarise_c23.py` | `outputs/c23_summary/c23_metrics.json` | reads artefacts only |

`scripts/18_layer_end_to_end.py` is a driver: it forks no logic, and every molecule comes
from the unmodified `guided_sample`. The one permitted edit to script 05 is additive and
is guarded by three tests
(`tests/test_layer_end_to_end.py::test_script_05_defaults_reproduce_the_pre_edit_code_path`
and its two siblings). Each arm is its own directory, so the harness kill that cost C17 an
hour would have cost C23 one arm.

Windows and target intervals are `outputs/pilot_50k_p2/{windows,target_intervals}.json`
read verbatim; nothing was re-derived. All arms use the C17 checkpoints, head seed
**1234** — the same head seed as the deployed `head_<prop>_frozen_state.pt`, so the layer
is the only difference and not the head seed. C17 saved only the seed-1234 checkpoints, so
head-seed replication of a mid-layer arm is **not available** and was not attempted.

## C23.2 The validity gate — the residual, not the word "matches"

`outputs/c23_gate_L12_lam1_aromatic_rings/` against
`outputs/pilot_50k_p2_guided_aromatic_rings/`, λ=1, seeds 101/202/303, 512 molecules per
condition per seed, `--layer 12 --head-file
c17_probe_layers/head_aromatic_rings_frozen_state_L12.pt`.

| condition | seed | reference | replay | residual |
| --- | ---: | ---: | ---: | ---: |
| unguided | 101 | 0.1749 | 0.1749 | 0.0 |
| unguided | 202 | 0.1824 | 0.1824 | 0.0 |
| unguided | 303 | 0.1784 | 0.1784 | 0.0 |
| throughout | 101 | 0.4892 | 0.4892 | 0.0 |
| throughout | 202 | 0.4795 | 0.4795 | 0.0 |
| throughout | 303 | 0.4517 | 0.4517 | 0.0 |

**Maximum absolute residual over all six cells: exactly 0.0**, and all **3072** returned
molecules are identical string for string and in order. Means: `throughout` **0.4735**
against **0.4735**, `unguided` **0.1785** against **0.1785**.

Two things this establishes and one it does not. It establishes that `--layer 12` is the
pre-edit `-1` code path — the tuple `res.hidden_states` has 13 entries and the two indices
are the same tensor, asserted separately against the real checkpoint — and that the C17
probe-point-12 checkpoint is interchangeable with the deployed head. It does **not**
establish that the C17 files are byte-identical to the deployed ones: they are not. As
C23.0.1 stated in advance, the C17 checkpoint carries an extra `probe_point` key, so it
differs in size and SHA-256, while every tensor of the state dict agrees to maximum
absolute difference **0.0** and the binner dictionary compares equal. The gate is run with
the C17 checkpoint, which is the family the arms come from.

**The gate passes.** The arms below are therefore interpretable.

## C23.3 Result — the arms

15 arms, 3 seeds, 512 molecules per condition per seed. `deployed` is the seed-matched
probe-point-12 run at the **same λ** (`pilot_50k_p2_{,lam*_}guided_<prop>`), never a
differently-seeded mean. `Δ` is the seed-matched difference and the interval is the
two-sided Bonferroni-corrected seed-stratified bootstrap at α = 0.05/15 = 0.003333,
n_boot = 10000.

| property | layer | why that layer | λ | **hit rate** | deployed | **Δ** | corrected CI | validity | tok/mol | tok ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| aromatic rings | 3 | AUROC-best | 0.5 | 0.3616 | 0.3011 | +0.0605 | [+0.0109, +0.1123] | 0.9974 | 405.8 | 0.9939 |
| aromatic rings | 3 | AUROC-best | 1 | **0.5698** | 0.4735 | **+0.0964** | [+0.0444, +0.1486] | 0.9941 | 412.5 | 0.9837 |
| aromatic rings | 3 | AUROC-best | 2 | 0.8159 | 0.5579 | +0.2579 | [+0.2100, +0.3063] | **0.9681** | 447.5 | **1.0645** |
| aromatic rings | 6 | steering-best | 0.5 | 0.3881 | 0.3011 | +0.0870 | [+0.0365, +0.1373] | 0.9980 | 409.3 | 1.0025 |
| aromatic rings | 6 | steering-best | 1 | **0.5831** | 0.4735 | **+0.1096** | [+0.0578, +0.1635] | 0.9915 | 422.8 | 1.0082 |
| aromatic rings | 6 | steering-best | 2 | 0.6842 | 0.5579 | +0.1262 | [+0.0750, +0.1774] | **0.9688** | 462.8 | **1.1010** |
| HBD count | 4 | AUROC-best | 0.5 | 0.2319 | 0.1813 | +0.0506 | [+0.0075, +0.0935] | 0.9967 | 397.0 | 1.0028 |
| HBD count | 4 | AUROC-best | 1 | **0.3689** | 0.2988 | **+0.0701** | [+0.0222, +0.1205] | 0.9954 | 394.3 | 0.9817 |
| HBD count | 4 | AUROC-best | 2 | **0.5603** | 0.4303 | **+0.1300** | [+0.0773, +0.1831] | 0.9935 | 387.8 | 0.9863 |
| HBD count | 6 | steering-best | 0.5 | 0.2027 | 0.1813 | +0.0214 | [−0.0204, +0.0627] | 0.9954 | 394.4 | 0.9964 |
| HBD count | 6 | steering-best | 1 | 0.3395 | 0.2988 | +0.0407 | [−0.0083, +0.0884] | 0.9954 | 398.6 | 0.9925 |
| HBD count | 6 | steering-best | 2 | 0.4684 | 0.4303 | +0.0381 | [−0.0157, +0.0919] | 0.9883 | 399.1 | 1.0151 |
| QED | 4 | AUROC-best | 1 | 0.2288 | 0.1908 | +0.0380 | [−0.0051, +0.0817] | 0.9987 | 367.8 | 1.0015 |
| QED | 4 | AUROC-best | 2 | 0.2796 | 0.2361 | +0.0435 | [−0.0011, +0.0894] | 0.9896 | 366.1 | 1.0117 |
| QED | 4 | AUROC-best | 4 | 0.3092 | 0.2607 | +0.0485 | [+0.0007, +0.0966] | **0.9518** | 375.3 | 1.0097 |

**`unguided` reproduces its central-test value in all 15 arms**: 0.1785 (aromatic rings),
0.0837 (HBD count), 0.0896 (QED), to every printed digit. The bug alarm does not fire.

**Every one of the fifteen differences is positive.** Twelve have a corrected CI excluding
zero; the three that do not are all HBD count at probe point 6.

**The effect is not a length effect.** `scripts/05_guided_generation.py`'s length-matched
estimator, at coverage 0.991–0.998: HBD count L4 λ=2 raw +0.4766, length-matched
**+0.4791** against the deployed layer's +0.3464; aromatic rings L3 λ=1 raw +0.3912,
length-matched **+0.3549** against the deployed layer's +0.2668. The mid-layer arms are if
anything *shorter* than their deployed counterparts (41.9 against 42.4 content tokens for
HBD count L4 λ=2; 44.5 against 45.4 for aromatic rings L3 λ=1), so the standardisation
moves the estimate slightly the wrong way for a length explanation. The full §16.4-style
joint standardisation on heavy-atom count was **not run** (see §C23.6).

### C23.3.1 Three arms are disqualified, by the rule stated before the numbers existed

C23.0.7 fixed the rule: an arm loses more than 0.01 validity or uniqueness against the
seed-matched deployed arm, or spends more than 5% more tokens, and it does not count as a
win. Three arms trip it, and all three are the ones with the largest raw gains:

| arm | validity | deployed validity | Δ validity | tok ratio | degeneracy, guided hits | base-policy hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings L3 λ=2 | 0.9681 | 0.9909 | −0.0228 | 1.0645 | **0.1731** | 0.0330 |
| aromatic rings L6 λ=2 | 0.9688 | 0.9909 | −0.0221 | 1.1010 | **0.1110** | 0.0330 |
| QED L4 λ=4 | 0.9518 | 0.9714 | −0.0195 | 1.0097 | 0.0199 | 0.0146 |

Aromatic rings at probe point 3, λ=2 reaches **0.8159** — within 0.014 of the deployed
layer's compute-matched best-of-N — but it does so with a **5.2× degeneracy rate** against
base-policy hits, 6.5% more tokens per molecule, and 2.3 points of validity. **That is not
a win, and it is not reported as one.** §19.3 found this failure mode on the deployed
layer only at λ ≥ 4; on a mid-network head it arrives at **λ = 2**.

Uniqueness is **1.0000** in every one of the fifteen arms, so no arm is disqualified on
that limb.

## C23.4 Result — the two decision rules

### C23.4.1 Rule A fires: the layer improves guidance

Required: the best mid-layer arm beats the deployed-layer arm at matched λ for **≥ 2 of 3**
properties, each beyond seed noise, with a corrected CI excluding 0, not disqualified.

| property | qualifying arms | best qualifying Δ | fires |
| --- | --- | ---: | --- |
| aromatic rings | L3 λ=1, L3 λ=0.5, L6 λ=1, L6 λ=0.5 | +0.1096 (L6, λ=1) | **yes** |
| HBD count | L4 λ=1, L4 λ=2, L4 λ=0.5 | +0.1300 (L4, λ=2) | **yes** |
| QED | none | — (best +0.0485, disqualified; best qualifying CI includes 0) | no |

**2 of 3. Rule A fires.** A mid-network head steers better end to end than the final-layer
head the whole report deploys, on two of the three anchors, at matched λ, matched seeds and
matched or lower token cost.

**This is the opposite of what C17's question 2 predicted, and it is why C23 exists.**
C17 measured these same layers per decoding position and found 2 of 6 properties improving
with median relative −0.0774. The two properties C17 found *improving* per position were
HBD count and rotatable bonds; aromatic rings was one of the four that got *worse* per
position, at **−0.111 relative** at probe point 3. **End to end, that same probe point 3
head is +0.0964 better than the deployed one at λ=1.** So the per-position measurement did
not merely understate the effect for aromatic rings — **it had the wrong sign**. That is a
stronger version of `docs/TODO.md` C22.1's warning than the audit had evidence for: a
per-position quantity is not a scaled-down end-to-end quantity, and here it is not even a
reliable indicator of direction.

It is not uniformly wrong either, and saying so matters. For HBD count the per-position
column ranked probe point 6 above probe point 4 (+0.00757 against +0.00470) and end to end
the ranking reverses (0.3395 against 0.3689 at λ=1). For QED the per-position column
ranked the deployed layer best, and end to end QED is the one anchor where no arm
qualifies. The per-position column is right about QED, wrong about aromatic rings, and
wrong about the ordering within HBD count.

### C23.4.2 Rule B fires — on exactly one arm, and it is marginal

Required: some arm's `throughout` hit rate exceeds its **own** compute-matched best-of-N,
with the corrected CI excluding 0 and the arm not disqualified.

**One arm does: HBD count, probe point 4, λ = 2.**

| | |
| --- | ---: |
| guided hit rate | **0.5603** (seeds 0.5976 / 0.5305 / 0.5529) |
| its own compute-matched best-of-N, N = 8 | **0.4844** (seeds 0.5000 / 0.4805 / 0.4727) |
| **advantage** | **+0.0760** |
| corrected CI (α = 0.05/15) | **[+0.0234, +0.1298]**, excludes 0 |
| per-seed advantage | +0.0976 / +0.0500 / +0.0803 — positive on all three |
| guided tokens per returned molecule | 387.8 |
| best-of-N tokens per returned molecule | 355.9 |
| **realised token ratio, guided / best-of-N** | **1.0897** |
| validity | 0.9935 against deployed 0.9941 |
| uniqueness | 1.0000 |
| degeneracy among guided hits | 0.0199 against base-policy hits 0.0703 |

By the pre-registered criterion, **Rule B fires: C4 and P5 are falsified for this arm.**

**But the realised token ratio is 1.0897, and that has to be dealt with before anything is
claimed.** `solve_best_of_n` floors, so a guided arm that got *cheaper* — this one spends
387.8 tokens per molecule against the deployed λ=2 arm's 393.2 — can push the solved N from
9 down to 8 and end up compared against a baseline that spends 8% fewer
tokens per returned molecule than it does. §16.2 already notes
that flooring always favours guidance slightly; here the margin and the slack are the same
size, so "reported, not assumed" is not enough.

**Token-conservative check, added after the arms were run and labelled as such.** Compare
the same arm against the cheapest already-executed best-of-N run for HBD count that spends
**at least as many** tokens per returned molecule as the guided arm — N = 9, 399.9 tokens
per molecule, realised ratio **0.9697** in best-of-N's favour:

| | |
| --- | ---: |
| guided | 0.5603 |
| best-of-N, N = 9 | 0.5234 |
| **advantage** | **+0.0369** |
| per-seed | +0.0547 / +0.0031 / +0.0529 — positive on all three, but seed 202 is nearly a tie |
| uncorrected 95% CI | [+0.0010, +0.0722], excludes 0 |
| **corrected CI (α = 0.05/15)** | **[−0.0173, +0.0885], includes 0** |

**So the honest statement is:** guidance with a mid-network head beats compute-matched
best-of-N on this arm under the pre-registered matching rule, by +0.0760 with a corrected
interval excluding zero; the point estimate stays positive (+0.0369) and seed-consistent
when best-of-N is instead given *more* tokens than guidance, but at that funding level the
margin no longer clears the multiplicity-corrected bar. **The falsification is real and it
is marginal.** It should not be reported as "guidance beats best-of-N" without the second
row of that table, and it should not be dismissed either: the point estimate is positive
under both accountings and on all three seeds under both.

No other arm comes close. The next best advantage is aromatic rings L3 λ=2 at **−0.0409**,
and that arm is disqualified on validity and tokens; among qualifying arms the next best is
HBD count L6 λ=2 at **−0.0551**. Only the `actual` accounting was run, as in §19.2; under
`full_recompute` best-of-N measured exactly 1.0000 at every λ tested and no guided hit rate
can exceed 1.


### C23.4.3 Rule C does not fire

The null was the pre-registered expectation and it is **not** the outcome.

### C23.4.4 Every arm against its own compute-matched best-of-N

N is re-solved by `scripts/06_best_of_n.py` from each arm's own measured
`tokens_per_molecule_actual` and its own `unguided` base tokens per molecule; it is not
inherited from the deployed run. `ratio` is the **realised** guided-over-best-of-N token
ratio, measured rather than assumed: above 1 means guidance was the better-funded side.
The last column is the token-conservative comparator of §C23.4.2 — the cheapest executed
best-of-N for that property that spends **at least** as many tokens per returned molecule
as the arm. **DQ** marks an arm disqualified by C23.0.7.

| property | layer | λ | guided | best-of-N | N | ratio | **advantage** | conservative best-of-N \| advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aromatic rings | 3 | 0.5 | 0.3616 | 0.8294 | 9 | 1.0148 | **-0.4678** | 0.8568 (N=10, ratio 0.9141) | -0.4952 |
| aromatic rings | 3 | 1 | 0.5698 | 0.8294 | 9 | 1.0315 | **-0.2596** | 0.8568 (N=10, ratio 0.9291) | -0.2870 |
| aromatic rings | 3 | 2 | 0.8159 **DQ** | 0.8568 | 10 | 1.0078 | **-0.0409** | none funded above this arm | — |
| aromatic rings | 6 | 0.5 | 0.3881 | 0.8294 | 9 | 1.0236 | **-0.4413** | 0.8568 (N=10, ratio 0.9219) | -0.4687 |
| aromatic rings | 6 | 1 | 0.5831 | 0.8294 | 9 | 1.0571 | **-0.2463** | 0.8568 (N=10, ratio 0.9521) | -0.2737 |
| aromatic rings | 6 | 2 | 0.6842 **DQ** | 0.8568 | 10 | 1.0423 | **-0.1726** | none funded above this arm | — |
| HBD count | 4 | 0.5 | 0.2319 | 0.5234 | 9 | 0.9926 | **-0.2916** | 0.5234 (N=9, ratio 0.9926) | -0.2916 |
| HBD count | 4 | 1 | 0.3689 | 0.5234 | 9 | 0.9859 | **-0.1545** | 0.5234 (N=9, ratio 0.9859) | -0.1545 |
| HBD count | 4 | 2 | 0.5603 | 0.4844 | 8 | 1.0897 | **+0.0760** | 0.5234 (N=9, ratio 0.9697) | +0.0369 |
| HBD count | 6 | 0.5 | 0.2027 | 0.5234 | 9 | 0.9863 | **-0.3207** | 0.5234 (N=9, ratio 0.9863) | -0.3207 |
| HBD count | 6 | 1 | 0.3395 | 0.5234 | 9 | 0.9967 | **-0.1840** | 0.5234 (N=9, ratio 0.9967) | -0.1840 |
| HBD count | 6 | 2 | 0.4684 | 0.5234 | 9 | 0.9980 | **-0.0551** | 0.5234 (N=9, ratio 0.9980) | -0.0551 |
| QED | 4 | 1 | 0.2288 | 0.5436 | 8 | 1.0336 | **-0.3148** | 0.5964 (N=9, ratio 0.9197) | -0.3675 |
| QED | 4 | 2 | 0.2796 | 0.5436 | 8 | 1.0289 | **-0.2640** | 0.5964 (N=9, ratio 0.9155) | -0.3168 |
| QED | 4 | 4 | 0.3092 **DQ** | 0.5436 | 8 | 1.0546 | **-0.2345** | 0.5964 (N=9, ratio 0.9385) | -0.2872 |

**Fourteen of fifteen arms lose to their own compute-matched best-of-N**, by −0.0409 to
−0.4678. One wins, by +0.0760.

Three arms have a realised guided-over-best-of-N ratio above 1.05, i.e. the flooring left
best-of-N materially under-funded: aromatic rings L6 λ=1 (1.0571), **HBD count L4 λ=2
(1.0897)** and QED L4 λ=4 (1.0546). Of those, QED L4 λ=4 is disqualified on quality anyway
and aromatic rings L6 λ=1 loses by 0.2463, so no amount of flooring slack changes its sign.
**Only the HBD count arm has a margin small enough for the slack to matter**, which is why
it gets the token-conservative comparator in §C23.4.2 rather than a footnote. Under that
comparator the sign does not change, but the corrected interval no longer excludes zero.


## C23.5 §C23.0 scored, verbatim, including where it fails

| item | committed | outcome |
| --- | --- | --- |
| C23.0.1 validity gate | residual 0.0, molecules identical | **passed**; residual exactly 0.0, 3072/3072 identical |
| C23.0.2 anchors | aromatic_rings, hbd_count, qed | held; none dropped, none added |
| C23.0.3 layers | {3,6}, {4,6}, {4} | held; taken from C17's frozen tables |
| C23.0.4 λ | 1, then §19-optimum, then 0.5/0.5/2 | held; the extra λ was named before any arm ran |
| C23.0.5 protocol | 3 seeds, 512/condition/seed, frozen windows and intervals | held; `unguided` reproduced exactly in all 15 arms |
| C23.0.6 Rule A | ≥2 of 3 properties | **fires**, 2 of 3 |
| C23.0.6 Rule B | some arm beats its own matched best-of-N | **fires**, one arm, +0.0760 |
| C23.0.6 Rule C | null | does not fire |
| C23.0.7 quality | disqualify on >0.01 validity or uniqueness drop | applied; **3 arms disqualified**, all of them large-gain arms |
| C23.0.8 headline prediction | **null fires** | **wrong** |
| C23.0.8 P-A | every λ=1 arm within ±0.05 of deployed | **falsified**; largest absolute difference **0.1096** (aromatic rings, probe point 6) |
| C23.0.8 P-B | probe point 6 beats the AUROC-best point at λ=1 for both counts | **falsified**; true for aromatic rings (0.5831 > 0.5698), false for HBD count (0.3395 < 0.3689) |
| C23.0.8 P-C | validity drop at the optimum **and** advantage larger at λ=0.5 than λ=2 | **falsified as a conjunction**; first clause holds (4 of 5 combinations lose >0.005 validity), second clause fails for all four count arms — the advantage *grows* with λ |
| C23.0.8 P-D | best advantage stays below −0.05 | **falsified**; best advantage **+0.0760** |

**Five of five predictions failed, including the headline one.** That is the correct
summary and it is written here rather than softened. The pre-registration's reasoning —
that C17's per-position null and C18's capacity/calibration null bounded what a layer swap
could buy end to end — was the same category of inference `docs/TODO.md` C22.1 warns
against, applied in the pessimistic direction this time.

Two things the pre-registration got right and that mattered:

- **C23.0.7's quality rule.** Written before any number was seen, it disqualifies the three
  largest gains, including the aromatic-ring arm that would otherwise have been quoted at
  0.8159 against a best-of-N of 0.8294. Had that rule been written afterwards it would be
  worthless.
- **C23.0.6's insistence that N be re-solved per arm.** It is what surfaced the flooring
  slack on the one arm that fires Rule B, rather than letting the deployed run's N=9 stand
  unexamined in either direction.

One thing it got wrong in *design*, not in prediction: the 5% token-ratio ceiling was
written against the **deployed arm**, not against the arm's own best-of-N. The arm that
fires Rule B has a deployed ratio of 0.9863 and a best-of-N ratio of 1.0897, so it passes
the rule as written while failing the rule's evident intent. §C23.4.2's conservative check
is the repair, and it is labelled post-hoc because it is.

## C23.6 What was not run

Stated explicitly, because a partial grid honestly scored is the deliverable and an
implied-complete one is not.

- **HBD count at λ=4 on probe point 4** — the obvious next arm, since λ=2 is the winner and
  §19's λ envelope peaks at 2 only for the *deployed* head. It was **deliberately not run**:
  C23.0.4 forbids selecting λ after seeing the lift, and running the neighbour of the
  winning arm after seeing it win is exactly that. It is the first thing a follow-up should
  do, pre-registered.
- **The other three properties** (rotatable bonds, TPSA, cLogP) at any mid-network layer.
  C23 has three anchors because those are the three with a full §19 λ envelope.
- **Any layer other than the two per property fixed by C17.** In particular probe points
  4 and 5, which C17's AUROC table puts at or near the peak for five of six properties, are
  untested end to end for aromatic rings.
- **The `early` / `middle` / `late` / `truncation_control` conditions** at any mid-network
  layer, so C23 says nothing about *where* in the trajectory the mid-layer advantage arises.
- **The `full_recompute` accounting**, as in §19.2.
- **`scripts/09_confound_analysis.py`** — the §16.4 joint standardisation on heavy-atom
  count. Only script 05's native length-matched estimator was computed (§C23.3).
- **Head-seed replication.** C17 saved only seed-1234 mid-layer checkpoints, so every arm
  rests on one head seed. §13.2 puts head-seed sd at ≤0.0041 AUROC, but that is a
  *prediction* quantity; the end-to-end head-seed variance of a guided run has never been
  measured in this project.
- **A best-of-N comparator for aromatic rings L3 λ=2 at N ≥ 11.** No executed best-of-N run
  spends as many tokens per molecule as that arm (447.5), so it has no token-conservative
  comparator. It is disqualified on quality anyway.
- **Wall-clock claims.** None are made; §11.7 says why.

## C23.7 Limitations

> **Correction, 2026-08-01 — what C23's intervals were actually measuring.**
>
> Every corrected CI in this section comes from a bootstrap that resamples **molecules
> within each seed** and then averages the three per-seed hit rates. That construction
> estimates the between-seed component as **exactly zero**: the width is driven by the
> ~1536 molecules alone, giving a standard error of about
> 0.018 ≈ √(2·0.25/1536). It is a correct interval for the question *"would these three
> runs, re-rolled molecule by molecule, still differ?"* and the wrong interval for the
> question the section actually asks, which is about runs.
>
> `scripts/18_summarise_c23.py` now reports a **seed-level Student t interval on 2 df**
> (`t₀.₉₇₅,₂ = 4.302653`) alongside every bootstrap, under the keys `diff_seed_level_t`,
> `advantage_seed_level_t` and `seed_level_t_interval`. Comparing the two across all 43
> interval verdicts in `c23_metrics.json`: **41 agree and 2 change** — `hbd_count_L4_lam0p5`
> and `qed_L4_lam4` no longer exclude zero against the deployed layer once between-seed
> variance is admitted.
>
> **The seed-level interval is not uniformly wider, and reading it as a strictly stricter
> test would be wrong.** The bootstrap is Bonferroni-corrected at α = 0.05/15 — roughly a
> 99.7% interval — while the t is a plain 95% on 2 df whose width is set by how much the
> three seeds happen to disagree. For four arms (aromatic L3 λ=1, aromatic L3 λ=0.5,
> aromatic L6 λ=0.5, HBD L4 λ=1) the t interval is the *narrower* of the two. That is a
> warning, not a reassurance: at n = 3 an sd can be small by luck, and a narrow t interval
> on three agreeing seeds is precisely the failure mode that produced Rule B. Both
> intervals are published for every arm and neither is claimed to dominate.
>
> **The headline rule is unaffected.** `layer_improves_guidance` still fires for 2 of 3
> properties: aromatic rings keeps all four qualifying arms (L3 λ=1, L3 λ=0.5, L6 λ=1,
> L6 λ=0.5), HBD count keeps L4 λ=1 and L4 λ=2 and loses only L4 λ=0.5, and QED did not
> fire under either interval. So **Rule A survives the correction** — which is worth
> stating positively, because it is now the only positive end-to-end result left standing.
>
> **Rule B did not, and this is why.** Its falsification fired on an interval of the first
> kind, [+0.0234, +0.1298]. C25 subsequently measured the variance that actually governs
> these arms — head seed, sd **0.0513** on the advantage, roughly **3×** what the
> molecule-level bootstrap assumed — and the arm reversed sign across head seeds. The
> interval was not merely optimistic; it was estimating the wrong variance component.
> No three-seed *percentile* bootstrap is reported anywhere in this project any more: at
> n = 3 it is identically [min, max] and conveys only a sign test at p_null = 0.25. The
> same correction was applied to C24 and C26 on the same day.

- **One arm carries Rule B, and it is marginal.** The falsification rests on HBD count at
  probe point 4, λ=2, and shrinks from +0.0760 to +0.0369 with a corrected interval
  spanning zero when best-of-N is funded above rather than below the guided arm. A
  replication — more seeds, more molecules, or the pre-registered λ=4 neighbour — is what
  would settle it, and none was run. **Superseded: C25 ran the head-seed replication and
  C26 re-priced the arm on a continuous frontier. Rule B is retired, not marginal.**
- **HBD count is the property where best-of-N was always weakest.** §16.2 measured Spearman
  ρ = −0.771 between base rate and the best-of-N advantage; HBD count has the lowest base
  rate (0.0823) and had the smallest gap (−0.2247 at λ=1, −0.0931 at λ=2) before C23
  started. The arm that crosses the line is the arm that was closest to it, which is what
  should have happened if the effect is real, and is also what would happen if the effect
  were noise on the most favourable cell of the grid.
- **The mid-layer advantage grows with λ and so does its quality cost.** The largest gains
  in §C23.3 are all at the largest λ tested for that property, and the three arms with the
  largest gains are the three that fail the quality gate. The λ envelope for a mid-network
  head has not been mapped; only three points per property were run.
- **Nothing here identifies the mechanism.** C17 observed that mid-layer heads have a
  1.26–1.51× larger candidate-probability spread; C23 does not test whether that is what
  produces the end-to-end gain, and does not claim it.
- **The base generator, the serialization, the windows and the intervals are unchanged**,
  and no DAgger round was run. C23's only degree of freedom is the probe point and the
  checkpoint that reads it.

## C23.8 Contradictions with the existing report

Flagged for the owner to merge. Nothing in `reports/pilot_report.md` was edited by C23.

1. **§19.2's "no λ beats compute-matched best-of-N" is now false as a general statement,
   and remains true as written.** As written it is a statement about the **deployed
   final-layer head**, and every number in it stands. What C23 contradicts is the broader
   reading — that guidance in this project never beats compute-matched best-of-N — which
   §19.4 and `reports/ABSTRACT.md` both lean on. Suggested repair: scope C4 and P5 to the
   final-layer head and cite this section for the mid-network exception.
2. **C4 is falsified on one (property, layer, λ) cell and survives on the other fourteen.**
   The honest headline is not "guidance wins" and not "guidance loses"; it is that the
   negative result was **layer-dependent**, and that the layer the whole project probed was
   the worst of the thirteen for this purpose — which is exactly what C17 found for
   *prediction* and could not establish for *steering*.
3. **§19.4's "fixing the head is the better bet" is supported, and it is now measured end
   to end rather than extrapolated.** The best qualifying arm is worth **+0.1096** for
   aromatic rings (probe point 6, λ=1) and **+0.1300** for HBD count (probe point 4, λ=2)
   in hit rate at matched λ. For scale, the entire λ sweep moved HBD count by +0.1316 in
   hit rate (0.2988 at λ=1 to 0.4303 at λ=2). The two terms are **comparable in size and
   they compose**: HBD count's best arm is the layer swap *and* the λ move together, and
   only the pair crosses the best-of-N line. §19.4's ranking of the head term above the λ
   term is not contradicted, but "the head term is 5–8×" — the withdrawn per-position
   figure — is not what C23 measures either.
4. **§C17.5's NOT MATERIAL verdict is not contradicted as a measurement and is
   contradicted as a guide.** C17's per-position numbers are unchanged and C23 does not
   re-run them. But the per-position column ranked probe point 6 above probe point 4 for
   HBD count, and end to end probe point 4 wins (0.3689 against 0.3395) — and it ranked
   aromatic rings at probe point 3 *below* probe point 12, where end to end it is well
   above. §C17.5.3's own caveat — "this section does not prove that a mid-network head
   would fail end to end" — was the right caveat, and the answer is that it does not fail.
5. **Rejection criterion R1 and §8.3.** C17 already reopened R1 for prediction. C23 reopens
   it for steering: the frozen state does carry steerable property information, it is
   simply not at the output layer.

---

## Commands to add to `docs/REPRODUCE.md`

A section "C23 — end-to-end guided decoding with a mid-network head", to run after the
phase-2 chain (P0–P7) and after C17. It needs `outputs/pilot_50k_p2/`,
`outputs/pilot_50k_heads_p2/`, `outputs/c17_probe_layers/` (for the mid-layer head
checkpoints) and the deployed λ-sweep runs `outputs/pilot_50k_p2_lam*_guided_*`.

```bash
# C23.1  validity gate FIRST. Reproduces outputs/pilot_50k_p2_guided_aromatic_rings at
#        --layer 12 with the C17 probe-point-12 checkpoint. Must come out at residual 0.0
#        with all 3072 molecules identical; if it does not, stop.
setsid nohup .venv/bin/python scripts/18_layer_end_to_end.py --stage gate \
    > outputs/c23_gate.log 2>&1 &

# C23.2  the 15 experimental arms, their compute-matched best-of-N and their quality
#        analysis, in the pre-registered priority order. ~45 min on an RTX 4090.
#        Idempotent: a completed arm is not regenerated, so a kill loses at most one arm.
setsid nohup .venv/bin/python scripts/18_layer_end_to_end.py --stage all \
    > outputs/c23_run.log 2>&1 &

# C23.3  assemble, bootstrap and score the decision rules. Reads only.
.venv/bin/python scripts/18_summarise_c23.py

.venv/bin/python -m pytest tests/test_layer_end_to_end.py -p no:cacheprovider
```

The two arguments added to `scripts/05_guided_generation.py` are additive and default to
the pre-edit behaviour:

```bash
# --layer      probe point of res.hidden_states for the head. Omitted -> -1, the value
#              guided_sample already defaulted to. --layer 12 is identical to omitting it.
# --head-file  head checkpoint (absolute, or relative to outputs/). Omitted ->
#              <heads>/head_<prop>_frozen_state[_seed<s>].pt, exactly as before.
.venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k_p2 \
    --heads pilot_50k_heads_p2 --property hbd_count --lam 2 \
    --layer 4 --head-file c17_probe_layers/head_hbd_count_frozen_state_L4.pt \
    --conditions unguided throughout --out c23_guided_L4_lam2_hbd_count
```

**Artefact sizes.** `outputs/c23_*/` is 15 guided directories of ~13 MB `molecules.json`
each plus small JSON, about 200 MB in total, of which only the metrics JSON is needed to
re-derive any number in this section. `outputs/c23_summary/c23_metrics.json` is 72 kB and
must stay: every number above is read from it by `tests/test_layer_end_to_end.py`.
