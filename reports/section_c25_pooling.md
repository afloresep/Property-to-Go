# Section C25 — the pooled readout

Draft section, written to be merged into `reports/pilot_report.md`. Structured as report
subsections so it can be renumbered and inserted without rewriting. Nothing in this file
edits an existing report claim; where the data contradicts one it is flagged under
"Contradictions with the existing report" and left for the owner to merge.

Experiment C25. It targets the one suspect `pilot_report.md` §8.3 names and nobody has
tested:

> Whether an earlier layer, a larger head, or **a different pooling** would close the
> aromatic-ring gap is untested.

Two of the three are now closed. §20.4 trained a 6.9× wider readout and moved seed-matched
target AUROC by at most +0.0019 on any property — capacity does not bind **at the final
layer**. §21 swept all 13 probe points and C23 took the best ones end to end: every one of
15 arms improved, +0.02 to +0.26 in hit rate. **Pooling is untouched.** Every head in this
project, baseline and variant, reads the single hidden state `h_t` at one position.

C25 also closes two gaps C17 and C23 explicitly leave open:

- **capacity × depth** — §20.4 excluded capacity at probe point 12, §21 measured depth with
  the 256-unit head, and a wider head at a mid-network probe point has never been trained;
- **head-seed replication at a mid layer** — C17 saved only seed-1234 checkpoints at
  non-final probe points, so every C23 arm, including the single arm that fires C23's Rule
  B, rests on one head seed.

### The verdict, up front

Three findings, in descending order of how much they should change what the report says.
Each is scored against a rule written before the run and frozen at `outputs/c25_prereg/`.

1. **Pooling improves prediction, and the pre-registration predicted it would not.**
   **Rule P1 fires** (5 of 6 properties at probe point 12) and **Rule P2 fires** (5 of 6 at
   the best mid-network probe point). §C25.0.8's headline prediction was that Rule N — the
   null — would fire; it **does not fire**. The best margin is +0.0382 and the median over
   the 48 pooled comparisons is +0.0116, against a material margin of 0.010: real, and
   small. The winning operator is a **parameter-free 16-position mean**; the
   parameter-heavy `concat4` is the worst pooled variant, so this is not capacity
   (**Rule P3 does not fire**: `wide1` helps at 0 of 6). Pooling still does not reach the
   trivial token-counting head for aromatic rings (**Rule P4 does not fire**).
2. **No pooled arm was run end to end, so C25 has no steering result.** Trigger T fired on
   15 comparisons and the six capped arms plus the pre-committed insurance arm were all
   **not run** (§C25.6). Rules E1, E2 and E3 are therefore vacuous and must not be quoted
   as an end-to-end null. Given C18 (calibration and capacity improved nothing end to end)
   and C23.4.1 (per-position quantities mis-ranked the end-to-end arms), a +0.038 AUROC
   gain is **not** evidence that guidance improves. §8.3's suspect is narrowed, not closed.
3. **C23's Rule B does not replicate across head seeds.** The arm carrying it — HBD count,
   probe point 4, λ = 2 — has hit rates 0.5603 / 0.5601 / 0.4687 at head seeds
   1234 / 2345 / 3456, a span of **0.0916**, and its advantage over its own
   compute-matched best-of-N goes **+0.0760 → +0.0366 → −0.0156**. §C25.0.7 committed in
   advance that a span ≥ 0.05 demotes the result, so **C23's Rule B is reported as not
   replicated at the head-seed level** (§C25.4, §C25.7). C23's Rule A — guided beats
   unguided with a mid-network head — survives on all three arms and all nine
   (head seed × generation seed) cells.

---

## C25.0 Pre-registration

**Written and saved to disk before any C25 output directory existed.** It is not revised
below; §C25.5 scores the executed result against it verbatim, including where it fails.

A verbatim copy of this subsection is frozen at
`outputs/c25_prereg/C25.0_preregistration.md`, with the SHA-256 of this whole file at the
moment of freezing recorded in `outputs/c25_prereg/prereg_lock.json`, so that the "prereg
precedes the run" claim is checkable after later subsections are appended and this file's
mtime has moved. `tests/test_pooled_readout.py` asserts the frozen copy is still a verbatim
substring of this file.

### C25.0.0 What C25 changes, and what it does not

**C25 changes exactly one thing** relative to §13 / §21 / C23: how many hidden-state
positions the readout reads. The generator is frozen — no fine-tuning, no LoRA, no RL, no
activation edits, no weight change of any kind. The serialization is SMILES. The windows
and target intervals are `outputs/pilot_50k_p2/{windows,target_intervals}.json` read
**verbatim and never re-derived**. Seeds, molecule counts and conditions are §16/§19/C23's.
No DAgger round is run; the one permitted round is spent.

`src/property_to_go/guidance.py` is **not modified**. Pooled decoding lives in a new module
`src/property_to_go/pooling.py` and new `scripts/20_*.py`, because `guidance.py` is
load-bearing for every existing number in the report.

### C25.0.1 Validity gate — checked before any experimental cell is trained

**A pooled readout with window size 1 must be _exactly_ the deployed single-position
head.** If it is not, C25 is measuring a refactor and everything downstream is void.

The gate has two limbs, both reported as numbers rather than as the word "matches".

**Limb A — the features.** `scripts/20_extract_pooled_states.py` re-extracts the window of
states around every prefix position, using `probe_layers.hidden_states_all_layers`'
batching unchanged (length-sorted, right padded, explicit attention mask,
`use_cache=False`). The last slot of the window stack is the state at the prefix position
itself, so it must come back **bit-identical** to `outputs/c17_layer_states_pilot_50k_p2/
layer<L>/hidden.npy` at every probe point extracted — and hence, at probe point 12, to the
dataset's own `hidden.npy`. Reported: `bit_identical` and `max_abs_difference` per probe
point.

**Limb B — the trained head.** The `last1` variant must reproduce, to a residual of
**exactly 0.0**:

* `outputs/pilot_50k_heads_p2/head_metrics.json` — §13's three-head-seed mean target AUROC
  and mean NLL for all six properties — at probe point 12; and
* `outputs/c17_probe_layers/probe_layer_metrics.json` — C17's three-head-seed means — at
  every probe point used.

Both residuals are reported per property as numbers. **If either limb fails, C25 stops and
reports the failure**; no experimental cell is interpreted.

For the learned pooling operator the same identity is asserted in code rather than by
argument: `AttnPoolHead` constructs its `MLPHead` **before** its attention parameters, so
it consumes the ambient RNG in the same order, and at window 1 its softmax is identically
1 and its extra parameters receive zero gradient.
`tests/test_pooled_readout.py::test_attention_pool_at_window_one_is_the_deployed_head`
asserts bit-identical logits rather than the reasoning.

A third identity, for the decoder rather than the trainer:
`pooling.pooled_guided_sample` at window 1 must return the **same molecules** and the
**same token counts** as `guidance.guided_sample`, asserted in
`tests/test_pooled_readout.py`.

### C25.0.2 What "pooling" means here, chosen before any result

Five variants. **The family is fixed here and no variant is added later.** It lives in
`property_to_go.pooling.POOL_VARIANTS` and the tests count it.

For a prefix at position `p` — the state after consuming tokens `0..p`, which is
`generation.hidden_states_for_positions`' contract — the window of size `w` is the
positions `max(0, p-w+1) .. p`. It contains `c = min(w, p+1)` **distinct** positions;
averaging operators average over those `c` and never over padding.

| name | `w` | operator | input dim | role |
| --- | ---: | --- | ---: | --- |
| `last1` | 1 | `h_p` | 768 | the deployed readout — baseline and validity gate |
| `mean4` | 4 | uniform mean | 768 | the short uniform window |
| `mean16` | 16 | uniform mean | 768 | the long uniform window |
| `concat4` | 4 | concatenation | 3072 | order-preserving; strictly more information than `last1` |
| `attn4` | 4 | learned single-query attention, `α = softmax(v·tanh(W h_j))` | 768 | the learned/gated pool |
| `wide1` | 1 | `h_p`, hidden dim 1024 | 768 | **not a pooling variant** — the capacity × depth control (§C25.0.3) |

`concat4` is in the family because it is the variant that cannot lose information: it
contains `h_p` as a sub-vector. If `concat4` does not beat `last1`, no window of four
positions helps at all, whatever the operator. It costs 4× the first-layer parameters, so
it is read together with §20.4's capacity result and with `wide1`.

**Decode-time computability, checked before the family was chosen, because it is a compute
question and not a taste question.** Every operator above needs only the last `w−1`
*committed* hidden states plus the candidate state. The committed states are returned by
the KV-cached forward pass the decoder already runs — `output_hidden_states=True` on that
call processes exactly the same tokens — and the candidate state is the one
`guidance._candidate_states_cached` already produces. So **a pooled readout costs the same
`processed_tokens_actual` and the same `processed_tokens_full_recompute` as the deployed
single-position readout**; the extra cost is a rolling buffer of at most 15 × 768 floats
per sequence. No variant in this family is affordable only under full-recompute accounting.
This is asserted, not assumed: the window-1 decoder identity test compares token counts as
well as molecules. A pooling operator that had to *re-read* the prefix — for example
attention whose keys are recomputed under each candidate — would **not** have this property
and is deliberately excluded from the family for that reason.

### C25.0.3 Where — pooling crossed with depth

Two depths per property, both fixed here from frozen numbers and never chosen from a C25
outcome:

* **probe point 12**, the final layer, where §8.3's claim lives and where §20.4 excluded
  capacity;
* **that property's AUROC-best mid-network probe point from §21's frozen table**:
  aromatic_rings 3, hbd_count 4, rotatable_bonds 4, tpsa 5, clogp 5, qed 4.

Crossing pooling with depth is what makes this more than a repeat of §20.4. `wide1`
(hidden dim 1024, window 1) is trained at the **mid** probe point only, because §20.4
already ran it at probe point 12; that cell is the capacity × depth test.

Six properties × two depths × five variants, plus six `wide1` cells at the mid depth =
**66 cells**, each at **three head seeds** (1234 / 2345 / 3456), so every comparison in
this section is seed-matched by construction and no one-seed variant is ever compared
against a three-seed mean.

### C25.0.4 Decision rules, both directions

**Primary metric:** held-out target-interval AUROC on the phase-2 test split, mean over the
three head seeds. **Secondary:** held-out NLL. Both are the metrics §13 and §21 use.

**Noise floor.** §13.2 measures head-seed sd ≤ 0.0041 AUROC and §21 ≤ 0.0057. C25 uses
C17's own pre-registered constants without changing them:
`probe_layers.MATERIAL_MARGIN = 0.010` as the material margin and
`probe_layers.SEED_SD = 0.004`.

**Multiplicity.** 54 comparisons (4 pooled variants × 6 properties × 2 depths, plus 6
`wide1` cells). Bootstrap intervals are two-sided at
**α = 0.05 / 54 = 0.000926**, paired on the identical held-out rows, `n_boot = 1000`, the
estimator being `probe_layers.paired_bootstrap_diff` unchanged — the same estimator §13 and
§21 use, with `alpha` exposed. The nominal-level interval is reported beside it.

**A variant HELPS at a (property, depth) cell** iff

1. its three-seed mean AUROC exceeds the **seed-matched** `last1` mean at the same
   (property, depth) by **≥ 0.010**, and
2. its Bonferroni-corrected paired-bootstrap CI on the AUROC difference **excludes 0**.

**A variant HURTS** iff the mean is **≤ −0.010** with the corrected CI excluding 0.
Anything else is **INDISTINGUISHABLE**.

---

**Rule P1 — pooling helps at the final layer.** Fires iff, at probe point 12, **≥ 4 of the
6 properties** have at least one pooled variant that HELPS.

**Rule P2 — pooling helps at the best mid-network probe point.** The same, at the mid
depth.

**Rule P3 — capacity binds at depth.** Fires iff `wide1` HELPS at **≥ 4 of the 6**
properties at the mid depth. This is §8.3's "larger head" suspect re-asked where §20.4 did
not ask it. It is scored independently of P1 and P2.

**Rule P4 — the aromatic-ring crossover.** §8.3's negative result is specifically that the
frozen state loses to the trivial token-counting head for aromatic rings. Fires iff a
pooled variant at probe point **12** reaches aromatic-ring AUROC ≥ `trivial` + 0.010
(trivial = **0.8269**, §13), i.e. iff pooling alone, at the layer the report deploys, does
what an earlier layer did.

**Rule N — null.** If none of P1–P4 fires, the result is a **null**, reported in those
words, as **the fourth cheap fix being spent** after C17 (depth), C18 (capacity and
calibration) and C23 (depth end to end). It is not softened by quoting the best cell as if
the selection were free, and it is not softened by a per-position number.

### C25.0.5 Prediction is not steering — the end-to-end trigger, fixed in advance

§21 and C23 together show a probe can predict better and steer worse per position while
steering better end to end. **An AUROC improvement is therefore not the result.** If
pooling improves AUROC that is a licence to run end-to-end guided generation, never a
substitute for it, and no per-position or prediction quantity is reported in this section
as if it were an end-to-end one.

**Trigger T.** An arm is run end to end iff its property is one of the three C23 anchors —
**aromatic_rings, hbd_count, qed**, fixed here because those are the three with a full §19 λ
envelope and a compute-matched best-of-N at every λ — and its variant **HELPS** at that
(property, depth) cell by the C25.0.4 rule, i.e. margin ≥ +0.010 over the seed-matched
`last1` with the corrected CI excluding 0.

**What a triggered arm runs.** Conditions `unguided` and `throughout`; seeds 101/202/303;
512 molecules per condition per seed; λ = 1 **and** that property's §19-optimal λ
(aromatic_rings 2, hbd_count 2, qed 4); compute-matched best-of-N with **N re-solved from
that arm's own** measured `tokens_per_molecule_actual`; and `scripts/10_quality_analysis.py`
on every arm. `unguided` must reproduce its central-test value exactly — 0.1785
(aromatic_rings), 0.0837 (hbd_count), 0.0896 (qed) — or the arm is reported as invalid
rather than interpreted.

**Cap.** At most **6** end-to-end arms, taken in descending order of AUROC margin. If more
qualify, the remainder are reported as not run, with their margins.

**Insurance arm I, fixed here and run regardless of the trigger.** Exactly one: the
**highest-margin pooled variant for hbd_count at probe point 4**, at **λ = 2**. Reason,
stated before any number: C23's Rule B is carried by a single marginal arm (hbd_count,
probe point 4, λ = 2, +0.0760 falling to +0.0369 under a token-conservative comparator),
and C23 itself showed that per-position and prediction quantities do not reliably rank
end-to-end steering in either direction. One arm is budgeted against that dissociation. It
is one arm, named in advance, and it is reported whatever it shows.

**End-to-end decision rules.** For a triggered arm, seed-stratified bootstrap exactly as
C23.0.6: within each of the three seeds, 512 molecules drawn with replacement from each
arm; the statistic is the difference of the seed-matched three-seed means; `n_boot = 10000`
with a recorded seed; two-sided Bonferroni correction at α = 0.05 / (number of end-to-end
arms actually run).

* **Rule E1 — pooling improves guidance end to end.** Fires iff, for ≥ 2 of the anchors
  run, the pooled arm exceeds the seed-matched `last1` arm at the same probe point and the
  same λ, beyond seed noise (`|Δ| > 2 · sd/√3`), with the corrected CI excluding 0, and not
  disqualified by C25.0.6.
* **Rule E2 — the headline falsification.** Fires iff some pooled arm's `throughout` hit
  rate exceeds **its own** compute-matched best-of-N, corrected CI excluding 0, not
  disqualified. The **realised** token ratio (guided per returned molecule ÷ best-of-N per
  returned molecule) is reported for every arm, never assumed; an arm whose realised ratio
  exceeds **1.05** also gets C23.4.2's token-conservative comparator — the cheapest executed
  best-of-N for that property that spends **at least** as many tokens per returned molecule
  — before the win is described as one.
* **Rule E3 — null end to end.** Neither fires.

### C25.0.6 Quality — a lift bought with a validity drop is not a win

Stated before any number is seen. `scripts/10_quality_analysis.py` runs on **every**
end-to-end arm, guided hits against base-policy hits, exactly as §16.5, §19.3 and C23.0.7
do. Validity, uniqueness and mean content length are reported for every arm.

An arm is **disqualified** from firing Rule E1 or E2 if its mean validity is more than
**0.01** below the seed-matched comparator arm at the same λ and probe point, or if its mean
uniqueness is more than **0.01** below it, or if its realised `processed_tokens_actual` per
returned molecule exceeds that comparator's by more than **5%**. A disqualified arm is still
reported, with its hit rate and its validity, and labelled disqualified — it is not deleted.
C23 learned that this rule has to be written first: it disqualified the three largest gains
in that section.

### C25.0.7 Head-seed replication at a mid layer (priority 4, and it is not about pooling)

C23's end-to-end result rests on head seed 1234 alone, because C17 saved no other mid-layer
checkpoint. C25's sweep trains `last1` at every property's mid probe point at all three head
seeds and **saves every checkpoint**, so the replication becomes a generation job.

Fixed here: three C23 arms are replicated at head seeds **2345** and **3456** —
**aromatic_rings probe point 3 at λ = 1**, **hbd_count probe point 4 at λ = 2** (the arm
that carries C23's Rule B), and **qed probe point 4 at λ = 1** — with the same generation
seeds 101/202/303, 512 molecules per condition per seed, through the **unmodified**
`scripts/05_guided_generation.py` with its C23 `--layer` / `--head-file` flags. Reported: the
per-head-seed hit rates, their spread, and — for the hbd_count arm — whether Rule B still
fires under each head seed against its own compute-matched best-of-N.

**Committed before the numbers exist:** if the three head seeds' end-to-end hit rates for
the hbd_count arm span more than **0.05**, C23's Rule B is reported as **not replicated at
the head-seed level**, whatever the mean does. §13.2's ≤0.0041 is a *prediction* sd and is
not evidence about this quantity.

### C25.0.8 Predictions

**Headline prediction: Rule N (null) fires — no pooled variant HELPS at ≥ 4 of 6 properties
at either depth.** Reasons, in order of weight: (i) adjacent hidden states of an
autoregressive model are highly correlated, so a uniform mean mostly smooths and a learned
pool mostly learns to select `h_p`; (ii) §20.4 showed the readout is not capacity-bound at
probe point 12, and `concat4` is largely a capacity increase in disguise; (iii) the depth
effect §21 measured (+0.02 to +0.06 AUROC) is the only readout intervention in this project
that has ever moved the number, and it moved it by changing *which* representation is read,
not *how much of it*.

Five sub-predictions, each falsifiable **independently of the headline**:

- **P-A (concat dominates).** If any variant HELPS anywhere, `concat4` HELPS at strictly
  more (property, depth) cells than `mean4` and `mean16` combined. Falsified if a mean pool
  HELPS at more cells than `concat4`.
- **P-B (the long window is the worst).** `mean16` is the worst of the four pooled variants
  by mean AUROC margin, averaged over the twelve cells. Falsified if any other pooled
  variant has a lower mean margin.
- **P-C (pooling does not rescue aromatic rings at layer 12).** Rule P4 does not fire: no
  pooled variant reaches aromatic-ring AUROC ≥ 0.8369 at probe point 12. Falsified by any
  variant that does.
- **P-D (capacity still does not bind, now at depth).** Rule P3 does not fire: `wide1` is
  INDISTINGUISHABLE from `last1` at ≥ 4 of the 6 mid-depth cells, replicating §20.4 one
  probe point at a time. Falsified if `wide1` HELPS at ≥ 4 of 6.
- **P-E (head seeds do not overturn C23).** The three head seeds' end-to-end hit rates for
  hbd_count probe point 4 λ = 2 span **less than 0.05**, and the seed-1234 value (0.5603)
  lies inside that span. Falsified by a span ≥ 0.05, which by C25.0.7 would demote C23's
  Rule B.

### C25.0.9 Priority order, and what is dropped first

Work is done in this order and dropped **from the end**, never because of what a stage
showed. §C25.6 states exactly what was not run.

1. Validity gate — `last1` reproduces the deployed head (features and metrics).
2. `last1` at the mid depth — the seed-matched mid baseline, and the checkpoints priority 4
   needs. (Placed second rather than third so the GPU stage in step 4 can run at the same
   time as the CPU stages; this changes which results arrive first and changes no measured
   quantity.)
3. The four pooled variants at probe point 12, six properties.
4. Head-seed replication of three C23 arms (§C25.0.7) — a GPU job, run concurrently.
5. The four pooled variants at the mid probe point, six properties.
6. `wide1` at the mid probe point — capacity × depth.
7. End-to-end guided generation for arms clearing Trigger T, plus insurance arm I, with
   matched best-of-N and quality analysis.

### C25.0.10 Pins recorded with every result

Model revision `6eca879581e2302b4e1ab07bb02908636bddb4a2`, tokenizer revision
`361063d0ad524ef77cf39b08469f6be770dc550f`, `deterministic_eval: true`, `dtype: float32`,
transformers 4.44.2, `.venv/bin/python`, head training on CPU (as in §13, §21 and C18),
generation on `device: cuda`. `write_run_context` and `configs_used.json` beside every
result. Compute reported as `processed_tokens_actual` and
`processed_tokens_full_recompute`, never as wall-clock (§11.7). Deterministic seeds
throughout: head seeds 1234/2345/3456, generation seeds 101/202/303.

---

*(Results below this line were written after the run. Everything above was on disk before
it, frozen at `outputs/c25_prereg/`.)*
## C25.1 What was run

| step | script | writes | status and cost |
| --- | --- | --- | --- |
| re-extract the window of hidden states at probe points 3, 4, 5 and 12 | `scripts/20_extract_pooled_states.py` | `outputs/c25_window_states_pilot_50k_p2/` | **run.** 2,205,784 processed tokens, one forward pass for all four probe points and both window sizes |
| train every (property, depth, variant) at three head seeds | `scripts/20_pooled_sweep.py` | `outputs/c25_pooled_heads/` | **run.** 66 cells × 3 head seeds; no generation, CPU head training, no tokens |
| head-seed replication of three C23 arms | `scripts/20_head_seed_replication.py` → scripts 05 / 06 / 10 | `outputs/c25_hs2345_*`, `outputs/c25_hs3456_*` (`_guided` / `_bestofn` / `_quality`) | **run.** 6 arms × 3072 molecules; 7,618,055 processed tokens including best-of-N |
| end-to-end guided decoding with a pooled readout | `scripts/20_pooled_guided_generation.py` | `outputs/c25_guided_*/` | **NOT RUN.** The script exists and its window-1 identity to `guidance.guided_sample` is tested, but no arm was executed: `outputs/c25_guided_*` is empty. See §C25.5.2 and §C25.6 |
| assemble and score the decision rules | `scripts/20_summarise_c25.py` | `outputs/c25_summary/c25_metrics.json` | **run.** Reads artefacts only |

**The fourth row is the one to read first.** C25 delivers a prediction result and no steering
result; §C25.5.2 says what that costs and §C25.6 lists the seven arms that were not run.

`src/property_to_go/guidance.py` was **not modified**, and neither were
`scripts/05_guided_generation.py`, `scripts/06_best_of_n.py` or
`scripts/10_quality_analysis.py`. The head-seed replication runs entirely through script 05
using the `--layer` / `--head-file` arguments C23 added. The pooled decoder is
`pooling.pooled_guided_sample`, a transcription of `guidance.guided_sample`; its
summarisation, its length-confound estimator and its artefact layout are script 05's own
functions, imported rather than reimplemented, which is why scripts 06 and 10 consume its
output unchanged.

Four smoke-test directories exist and are **not results**: `outputs/c25_smoke_states/`
(a 200-trajectory extraction used to exercise the extractor), `outputs/c25_smoke_guided/`,
`outputs/c25_smoke_quality/` and `outputs/c25_smoke_bestofn/` (a 16-molecule single-seed run
used to check that scripts 06 and 10 accept the pooled runner's artefacts; its best-of-N
came out at N = 9 and hit rate 0.8294, which is §16.2's aromatic-ring value, so the
compatibility check doubles as a sanity check).

The **assemble** row was the last thing run, after the sweep and the replication had both
finished; `outputs/c25_summary/c25_metrics.json` is the file every number below is read
from, and `tests/test_pooled_readout.py` re-reads it and requires the numbers to appear in
this text.

**Compute.** Extraction: **2,205,784** processed tokens actual, identical under full
recompute because it is one forward pass (an implementation that took one pass per probe
point would have cost 8,823,136). Head training: CPU only, no generation, no tokens.
Head-seed replication: **4,067,679** processed tokens actual across the six guided
directories (`unguided` + `throughout`, three generation seeds each) and **3,550,376**
across their six compute-matched best-of-N runs, **7,618,055** in total. No wall-clock claim
is made anywhere in this section (§11.7).

---

## C25.2 The validity gate — both limbs, as numbers

§C25.0.1 said C25 stops if either limb fails. Neither failed, and the residuals are exactly
zero rather than small.

### C25.2.1 Limb A — the features are bit-identical

`scripts/20_extract_pooled_states.py` re-extracted, for each of the **199,300** prefix rows
over 49,825 phase-2 trajectories, a stack of 4 states and a stack of 16 states ending at the
prefix position, at probe points 3, 4, 5 and 12, in one forward pass at batch size 96.

| probe point | reference | `bit_identical` | `max_abs_difference` |
| ---: | --- | :---: | ---: |
| 3 | `c17_layer_states_pilot_50k_p2/layer3/hidden.npy` | true | 0.0 |
| 4 | `c17_layer_states_pilot_50k_p2/layer4/hidden.npy` | true | 0.0 |
| 5 | `c17_layer_states_pilot_50k_p2/layer5/hidden.npy` | true | 0.0 |
| 12 | `c17_layer_states_pilot_50k_p2/layer12/hidden.npy` | true | 0.0 |

`all_bit_identical: true`. The last slot of the window stack is the single-position state
C17 extracted, to the last bit, at every probe point used — so the window stacks are the
deployed features plus context, not a re-derivation of them.

### C25.2.2 Limb B — the trained `last1` head is the deployed head

18 gate cells: six properties against §13's `head_metrics.json` at probe point 12, the same
six against C17's `probe_layer_metrics.json` at probe point 12, and six against C17 at each
property's mid probe point. Every one of the 18 came back with

* `auroc_residual` = **0.0**, and
* `nll_residual` = **0.0**,

so `max_residual` is **0.0** and `passes_at_exactly_zero` is **true**. That is the number
§C25.0.1 demanded; it is not "matches to 4 decimal places".

### C25.2.3 A third identity the pre-registration did not ask for

Because the sweep trains three head seeds and C17 recorded three head seeds, the gate can be
run **per head seed** rather than on the three-seed mean. All **36** rows (12 `last1` cells ×
3 head seeds) reproduce C17's per-seed target AUROC with a maximum residual of **0.0**. This
is stronger than what §C25.0.1 asked for and is reported because it was free, not because it
was needed: it rules out the possibility that a mean matched while its constituents did not.

The two code-level identities also hold:
`test_attention_pool_at_window_one_is_the_deployed_head` (bit-identical logits and the same
draw from the ambient RNG), `test_training_the_attention_pool_at_window_one_reproduces_train_head`
(bit-identical trained weights), and the decoder identity
`test_pooled_guided_sample_at_window_one_matches_guided_sample` (same molecules **and** the
same `processed_tokens_actual` and `processed_tokens_full_recompute`).

---

## C25.3 The pooling sweep — all 54 comparisons, printed

66 cells, each at head seeds 1234 / 2345 / 3456, 54 seed-matched comparisons against the
`last1` baseline at the same (property, depth). Every comparison is below; none is omitted,
and the selection-free reading is the whole table rather than its maximum.

Bonferroni α = 0.05 / 54 = 0.000926, `n_boot` = 1000, paired on the identical held-out rows,
estimator `probe_layers.paired_bootstrap_diff` unchanged. "**HELPS**" means margin ≥ +0.010
**and** the corrected CI excludes 0; "**HURTS**" means margin ≤ −0.010 with the corrected CI
excluding 0; everything else is "indist." (INDISTINGUISHABLE). The nominal-level interval is
printed beside the corrected one, as §C25.0.4 required, and is **not** what any verdict is
based on.

### C25.3.1 Probe point 12 — the final layer, where §8.3's claim lives

| property | probe point | variant | AUROC | `last1` AUROC | **margin** | corrected CI | nominal CI | NLL | verdict |
| --- | ---: | --- | ---: | ---: | ---: | :---: | :---: | ---: | --- |
| aromatic_rings | 12 | `last1` (baseline) | 0.7878 | — | — | — | — | 1.0445 | baseline |
| aromatic_rings | 12 | `mean4` | 0.8061 | 0.7878 | **+0.0183** | [+0.0085, +0.0237] | [+0.0119, +0.0215] | 0.9957 | **HELPS** |
| aromatic_rings | 12 | `mean16` | 0.8257 | 0.7878 | **+0.0380** | [+0.0244, +0.0466] | [+0.0301, +0.0427] | 0.9302 | **HELPS** |
| aromatic_rings | 12 | `concat4` | 0.8016 | 0.7878 | **+0.0139** | [+0.0033, +0.0189] | [+0.0075, +0.0172] | 1.0093 | **HELPS** |
| aromatic_rings | 12 | `attn4` | 0.8082 | 0.7878 | **+0.0205** | [+0.0098, +0.0244] | [+0.0136, +0.0226] | 0.9884 | **HELPS** |
| hbd_count | 12 | `last1` (baseline) | 0.7799 | — | — | — | — | 1.1071 | baseline |
| hbd_count | 12 | `mean4` | 0.7965 | 0.7799 | **+0.0166** | [+0.0088, +0.0296] | [+0.0116, +0.0259] | 1.0585 | **HELPS** |
| hbd_count | 12 | `mean16` | 0.8181 | 0.7799 | **+0.0382** | [+0.0239, +0.0540] | [+0.0302, +0.0480] | 0.9917 | **HELPS** |
| hbd_count | 12 | `concat4` | 0.7955 | 0.7799 | **+0.0156** | [+0.0060, +0.0273] | [+0.0100, +0.0240] | 1.0742 | **HELPS** |
| hbd_count | 12 | `attn4` | 0.8009 | 0.7799 | **+0.0210** | [+0.0122, +0.0314] | [+0.0148, +0.0288] | 1.0508 | **HELPS** |
| rotatable_bonds | 12 | `last1` (baseline) | 0.7820 | — | — | — | — | 1.8084 | baseline |
| rotatable_bonds | 12 | `mean4` | 0.7985 | 0.7820 | **+0.0165** | [+0.0081, +0.0263] | [+0.0109, +0.0227] | 1.7553 | **HELPS** |
| rotatable_bonds | 12 | `mean16` | 0.8154 | 0.7820 | **+0.0335** | [+0.0205, +0.0479] | [+0.0250, +0.0416] | 1.6763 | **HELPS** |
| rotatable_bonds | 12 | `concat4` | 0.7952 | 0.7820 | **+0.0133** | [+0.0084, +0.0252] | [+0.0110, +0.0226] | 1.7761 | **HELPS** |
| rotatable_bonds | 12 | `attn4` | 0.8000 | 0.7820 | **+0.0180** | [+0.0093, +0.0272] | [+0.0117, +0.0236] | 1.7461 | **HELPS** |
| tpsa | 12 | `last1` (baseline) | 0.7369 | — | — | — | — | 2.7527 | baseline |
| tpsa | 12 | `mean4` | 0.7482 | 0.7369 | **+0.0113** | [-0.0042, +0.0173] | [+0.0015, +0.0138] | 2.7239 | indist. |
| tpsa | 12 | `mean16` | 0.7590 | 0.7369 | **+0.0221** | [+0.0060, +0.0334] | [+0.0125, +0.0291] | 2.6942 | **HELPS** |
| tpsa | 12 | `concat4` | 0.7451 | 0.7369 | **+0.0081** | [-0.0030, +0.0172] | [+0.0001, +0.0132] | 2.7341 | indist. |
| tpsa | 12 | `attn4` | 0.7492 | 0.7369 | **+0.0123** | [-0.0015, +0.0182] | [+0.0034, +0.0155] | 2.7191 | indist. |
| clogp | 12 | `last1` (baseline) | 0.7913 | — | — | — | — | 2.7154 | baseline |
| clogp | 12 | `mean4` | 0.7988 | 0.7913 | **+0.0075** | [+0.0036, +0.0169] | [+0.0059, +0.0151] | 2.6947 | indist. |
| clogp | 12 | `mean16` | 0.8001 | 0.7913 | **+0.0088** | [+0.0019, +0.0240] | [+0.0080, +0.0213] | 2.6976 | indist. |
| clogp | 12 | `concat4` | 0.7970 | 0.7913 | **+0.0057** | [+0.0028, +0.0154] | [+0.0042, +0.0133] | 2.7008 | indist. |
| clogp | 12 | `attn4` | 0.8004 | 0.7913 | **+0.0091** | [+0.0051, +0.0176] | [+0.0065, +0.0152] | 2.6899 | indist. |
| qed | 12 | `last1` (baseline) | 0.7348 | — | — | — | — | 2.7715 | baseline |
| qed | 12 | `mean4` | 0.7452 | 0.7348 | **+0.0104** | [+0.0015, +0.0203] | [+0.0063, +0.0176] | 2.7466 | **HELPS** |
| qed | 12 | `mean16` | 0.7618 | 0.7348 | **+0.0269** | [+0.0102, +0.0344] | [+0.0152, +0.0306] | 2.7186 | **HELPS** |
| qed | 12 | `concat4` | 0.7386 | 0.7348 | **+0.0038** | [-0.0035, +0.0145] | [-0.0012, +0.0105] | 2.7608 | indist. |
| qed | 12 | `attn4` | 0.7465 | 0.7348 | **+0.0117** | [+0.0060, +0.0235] | [+0.0096, +0.0204] | 2.7418 | **HELPS** |

Properties with at least one pooled variant HELPING at probe point 12: **aromatic_rings,
hbd_count, rotatable_bonds, tpsa, qed — five of six.** cLogP is the exception: all four
pooled variants are positive (+0.0057 to +0.0091) and all four corrected intervals exclude
zero, but none clears the +0.010 material margin, so all four are INDISTINGUISHABLE by the
pre-registered rule. That is the rule working as intended — a real but immaterial effect is
not a HELP.

### C25.3.2 The best mid-network probe point, plus the capacity control

| property | probe point | variant | AUROC | `last1` AUROC | **margin** | corrected CI | nominal CI | NLL | verdict |
| --- | ---: | --- | ---: | ---: | ---: | :---: | :---: | ---: | --- |
| aromatic_rings | 3 | `last1` (baseline) | 0.8474 | — | — | — | — | 0.8658 | baseline |
| aromatic_rings | 3 | `mean4` | 0.8517 | 0.8474 | **+0.0043** | [-0.0028, +0.0081] | [-0.0004, +0.0060] | 0.8514 | indist. |
| aromatic_rings | 3 | `mean16` | 0.8522 | 0.8474 | **+0.0048** | [-0.0031, +0.0104] | [-0.0002, +0.0083] | 0.8549 | indist. |
| aromatic_rings | 3 | `concat4` | 0.8458 | 0.8474 | **-0.0016** | [-0.0072, +0.0028] | [-0.0051, +0.0009] | 0.8631 | indist. |
| aromatic_rings | 3 | `attn4` | 0.8533 | 0.8474 | **+0.0059** | [+0.0003, +0.0102] | [+0.0023, +0.0085] | 0.8456 | indist. |
| aromatic_rings | 3 | `wide1` | 0.8454 | 0.8474 | **-0.0020** | [-0.0041, +0.0056] | [-0.0024, +0.0036] | 0.8697 | indist. |
| hbd_count | 4 | `last1` (baseline) | 0.8226 | — | — | — | — | 0.9837 | baseline |
| hbd_count | 4 | `mean4` | 0.8356 | 0.8226 | **+0.0130** | [+0.0027, +0.0208] | [+0.0071, +0.0170] | 0.9439 | **HELPS** |
| hbd_count | 4 | `mean16` | 0.8381 | 0.8226 | **+0.0155** | [-0.0005, +0.0227] | [+0.0047, +0.0187] | 0.9045 | indist. |
| hbd_count | 4 | `concat4` | 0.8261 | 0.8226 | **+0.0035** | [-0.0081, +0.0103] | [-0.0036, +0.0072] | 0.9671 | indist. |
| hbd_count | 4 | `attn4` | 0.8351 | 0.8226 | **+0.0125** | [-0.0021, +0.0185] | [+0.0037, +0.0156] | 0.9377 | indist. |
| hbd_count | 4 | `wide1` | 0.8183 | 0.8226 | **-0.0043** | [-0.0098, +0.0010] | [-0.0087, -0.0019] | 0.9917 | indist. |
| rotatable_bonds | 4 | `last1` (baseline) | 0.8158 | — | — | — | — | 1.6975 | baseline |
| rotatable_bonds | 4 | `mean4` | 0.8228 | 0.8158 | **+0.0070** | [-0.0028, +0.0132] | [+0.0004, +0.0097] | 1.6452 | indist. |
| rotatable_bonds | 4 | `mean16` | 0.8302 | 0.8158 | **+0.0144** | [+0.0040, +0.0235] | [+0.0090, +0.0208] | 1.6021 | **HELPS** |
| rotatable_bonds | 4 | `concat4` | 0.8171 | 0.8158 | **+0.0013** | [-0.0065, +0.0101] | [-0.0038, +0.0073] | 1.6818 | indist. |
| rotatable_bonds | 4 | `attn4` | 0.8233 | 0.8158 | **+0.0075** | [-0.0025, +0.0132] | [+0.0008, +0.0106] | 1.6383 | indist. |
| rotatable_bonds | 4 | `wide1` | 0.8140 | 0.8158 | **-0.0018** | [-0.0095, +0.0031] | [-0.0083, +0.0005] | 1.7134 | indist. |
| tpsa | 5 | `last1` (baseline) | 0.7595 | — | — | — | — | 2.6812 | baseline |
| tpsa | 5 | `mean4` | 0.7673 | 0.7595 | **+0.0079** | [-0.0001, +0.0164] | [+0.0025, +0.0125] | 2.6460 | indist. |
| tpsa | 5 | `mean16` | 0.7743 | 0.7595 | **+0.0148** | [+0.0020, +0.0246] | [+0.0060, +0.0196] | 2.6226 | **HELPS** |
| tpsa | 5 | `concat4` | 0.7615 | 0.7595 | **+0.0021** | [-0.0049, +0.0120] | [-0.0018, +0.0077] | 2.6739 | indist. |
| tpsa | 5 | `attn4` | 0.7693 | 0.7595 | **+0.0099** | [+0.0020, +0.0179] | [+0.0043, +0.0141] | 2.6401 | indist. |
| tpsa | 5 | `wide1` | 0.7593 | 0.7595 | **-0.0001** | [-0.0038, +0.0066] | [-0.0021, +0.0045] | 2.6927 | indist. |
| clogp | 5 | `last1` (baseline) | 0.7998 | — | — | — | — | 2.6878 | baseline |
| clogp | 5 | `mean4` | 0.8045 | 0.7998 | **+0.0047** | [-0.0032, +0.0103] | [-0.0001, +0.0086] | 2.6664 | indist. |
| clogp | 5 | `mean16` | 0.8114 | 0.7998 | **+0.0116** | [+0.0050, +0.0216] | [+0.0086, +0.0191] | 2.6609 | **HELPS** |
| clogp | 5 | `concat4` | 0.8020 | 0.7998 | **+0.0022** | [-0.0057, +0.0082] | [-0.0029, +0.0062] | 2.6822 | indist. |
| clogp | 5 | `attn4` | 0.8081 | 0.7998 | **+0.0083** | [+0.0029, +0.0146] | [+0.0057, +0.0128] | 2.6624 | indist. |
| clogp | 5 | `wide1` | 0.7978 | 0.7998 | **-0.0020** | [-0.0054, +0.0035] | [-0.0035, +0.0018] | 2.6980 | indist. |
| qed | 4 | `last1` (baseline) | 0.7532 | — | — | — | — | 2.7348 | baseline |
| qed | 4 | `mean4` | 0.7671 | 0.7532 | **+0.0139** | [+0.0029, +0.0237] | [+0.0072, +0.0191] | 2.7052 | **HELPS** |
| qed | 4 | `mean16` | 0.7808 | 0.7532 | **+0.0275** | [+0.0170, +0.0431] | [+0.0228, +0.0379] | 2.6789 | **HELPS** |
| qed | 4 | `concat4` | 0.7539 | 0.7532 | **+0.0006** | [-0.0103, +0.0075] | [-0.0075, +0.0046] | 2.7344 | indist. |
| qed | 4 | `attn4` | 0.7696 | 0.7532 | **+0.0164** | [+0.0058, +0.0229] | [+0.0090, +0.0202] | 2.6990 | **HELPS** |
| qed | 4 | `wide1` | 0.7480 | 0.7532 | **-0.0053** | [-0.0059, +0.0087] | [-0.0042, +0.0044] | 2.7447 | indist. |

Properties with at least one pooled variant HELPING at the mid probe point: **hbd_count,
rotatable_bonds, tpsa, clogp, qed — five of six.** Aromatic rings is the exception at depth,
and it is the property where depth already did the most: at probe point 3 the single-position
head is already at 0.8474 and the best pooled variant adds +0.0059, which does not clear the
margin.

**No cell HURTS.** The most negative margin in the whole table is **-0.0053** (`wide1`, qed,
probe point 4) and it is nowhere near −0.010, so `wide1` is INDISTINGUISHABLE rather than
harmful. That matters for how §C25.5 scores Rule P3: capacity at depth does nothing, in
either direction.

### C25.3.3 The structure of the table, which is more informative than its maximum

Averaged over cells, so no maximum is being quoted as if the selection were free:

| variant | mean margin, probe point 12 (6 cells) | mean margin, mid (6 cells) | cells HELPED (of 12) |
| --- | ---: | ---: | ---: |
| `mean16` | +0.0279 | +0.0148 | 9 |
| `attn4` | +0.0154 | +0.0101 | 5 |
| `mean4` | +0.0134 | +0.0085 | 6 |
| `concat4` | +0.0101 | +0.0014 | 3 |
| `wide1` | not run (§20.4 ran it) | −0.0026 | 0 |

Three things fall out, and the second is the one that matters most.

**(1) The effect is real but small, and it is monotone in window length, not in capacity.**
`mean16` — a plain unweighted average over sixteen positions, with **zero** extra parameters
— is the best variant at both depths and at almost every cell. `concat4`, which has 4× the
first-layer parameters and provably contains `h_p` as a sub-vector, is the **worst** pooled
variant. A pooled readout is therefore not buying capacity; it is buying smoothing. This is
consistent with §20.4 and with `wide1`'s −0.0026 mean here.

**(2) Pooling at the final layer buys back most, and for two properties all, of what depth
buys.** Comparing the best pooled cell at probe point 12 against the *single-position* head
at that property's best mid probe point — the C17/C23 intervention:

| property | best pooled at pp 12 | `last1` at mid pp | mid `last1` − pooled pp 12 |
| --- | ---: | ---: | ---: |
| aromatic_rings | 0.8257 (`mean16`) | 0.8474 (pp 3) | +0.0216 |
| hbd_count | 0.8181 (`mean16`) | 0.8226 (pp 4) | +0.0045 |
| rotatable_bonds | 0.8154 (`mean16`) | 0.8158 (pp 4) | +0.0004 |
| tpsa | 0.7590 (`mean16`) | 0.7595 (pp 5) | +0.0005 |
| clogp | 0.8004 (`attn4`) | 0.7998 (pp 5) | −0.0006 |
| qed | 0.7618 (`mean16`) | 0.7532 (pp 4) | −0.0085 |

For four of six properties the difference is inside ±0.010, and for two (cLogP, QED) pooling
at the deployed layer is *better* than reading a single position at the best mid-network
layer. Only aromatic rings keeps a material depth advantage. **Depth and window appear to be
largely substitutable ways of reading the same information**, which is also why they do not
add: the mean pooled margin at the mid depth (+0.0087 over 24 comparisons) is roughly half
the mean margin at probe point 12 (+0.0167 over 24). Where the single-position readout is
already good, pooling adds little.

**(3) It does not close the aromatic-ring gap.** The best pooled cell at probe point 12 is
0.8257, against the trivial token-counting head's 0.8269. Pooling at the deployed layer moves
aromatic rings from 0.7878 to 0.8257 and **still does not reach the trivial baseline**, let
alone beat it by the material margin. §8.3's specific negative result stands.

---

## C25.4 Head-seed replication of C23's three anchors (§C25.0.7)

**This is not a pooling result and is not scored by any P- or E-rule.** It is the replication
C23 could not run because C17 saved no mid-layer checkpoint outside head seed 1234. C25's
sweep saved every checkpoint, so the same three C23 arms were re-run at head seeds **2345**
and **3456** through the **unmodified** `scripts/05_guided_generation.py`, same generation
seeds 101/202/303, 512 molecules per condition per seed, followed by
`scripts/06_best_of_n.py` (N re-solved from each arm's own measured tokens per molecule) and
`scripts/10_quality_analysis.py`.

All six new runs reproduce their `unguided` reference exactly — **0.1785** (aromatic rings),
**0.0837** (HBD count), **0.0896** (QED) — because `unguided` does not touch the head, so
none is invalid under §C25.0.5.

| arm | head seed 1234 (C23) | head seed 2345 | head seed 3456 | mean | sd | **span** | span < 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| aromatic_rings, pp 3, λ=1 | 0.5698 | 0.6101 | 0.6472 | 0.6090 | 0.0387 | **0.0773** | **no** |
| hbd_count, pp 4, λ=2 (C23's Rule B arm) | 0.5603 | 0.5601 | 0.4687 | 0.5297 | 0.0528 | **0.0916** | **no** |
| qed, pp 4, λ=1 | 0.2288 | 0.2422 | 0.2147 | 0.2286 | 0.0137 | **0.0275** | yes |

Per generation seed, so the head-seed spread can be read against the generation-seed spread
it has to beat:

| arm | head seed | seeds 101 / 202 / 303 | validity | uniqueness | tokens / molecule (actual) |
| --- | ---: | --- | ---: | ---: | ---: |
| aromatic_rings pp 3 λ=1 | 1234 | 0.5917 / 0.5756 / 0.5421 | 0.9941 | 1.0000 | 412.50 |
| | 2345 | 0.6004 / 0.6099 / 0.6201 | 0.9902 | 1.0000 | 427.54 |
| | 3456 | 0.6373 / 0.6547 / 0.6495 | 0.9798 | 1.0000 | 434.85 |
| hbd_count pp 4 λ=2 | 1234 | 0.5976 / 0.5305 / 0.5529 | 0.9935 | 1.0000 | 387.79 |
| | 2345 | 0.5787 / 0.5641 / 0.5374 | 0.9915 | 1.0000 | 393.80 |
| | 3456 | 0.4676 / 0.4673 / 0.4713 | 0.9889 | 1.0000 | 387.79 |
| qed pp 4 λ=1 | 1234 | 0.2387 / 0.2285 / 0.2192 | 0.9987 | 1.0000 | 367.82 |
| | 2345 | 0.2505 / 0.2589 / 0.2172 | 0.9922 | 1.0000 | 370.28 |
| | 3456 | 0.2188 / 0.1980 / 0.2275 | 0.9974 | 1.0000 | 373.28 |

The head-seed spread is **larger than the generation-seed spread within a head seed** for
two of the three arms — for HBD count at head seed 3456 the three generation seeds span
0.0040 while the three head seeds span 0.0916, a factor of 23. Head seed is the dominant
source of variance in these end-to-end numbers, and no previous section in this project
measured it.

### C25.4.1 Does C23 survive? Three answers, and they differ

**(a) "Guided beats unguided" — survives, comfortably, on all three arms and all nine
(head seed × generation seed) combinations.** The weakest guided cell anywhere in the table
is QED at head seed 3456, generation seed 202, at 0.1980 against an unguided 0.0896; the
HBD-count arm's worst cell is 0.4673 against 0.0837. C23's Rule A result — that a mid-network
head steers better end to end than the deployed one — is not threatened by head-seed
variation in any arm.

**(b) "Guided beats its own compute-matched best-of-N" — does not survive.** This is C23's
Rule B, and it is the falsification claim, so it is the one that matters.

| arm | head seed | guided | its own best-of-N | N | **advantage** |
| --- | ---: | ---: | ---: | ---: | ---: |
| hbd_count pp 4 λ=2 | 1234 | 0.5603 | 0.4844 | 8 | **+0.0760** |
| | 2345 | 0.5601 | 0.5234 | 9 | **+0.0366** |
| | 3456 | 0.4687 | 0.4844 | 8 | **−0.0156** |
| aromatic_rings pp 3 λ=1 | 1234 | 0.5698 | 0.8294 | 9 | −0.2596 |
| | 2345 | 0.6101 | 0.8294 | 9 | −0.2193 |
| | 3456 | 0.6472 | 0.8568 | 10 | −0.2096 |
| qed pp 4 λ=1 | 1234 | 0.2288 | 0.5436 | 8 | −0.3148 |
| | 2345 | 0.2422 | 0.5436 | 8 | −0.3014 |
| | 3456 | 0.2147 | 0.5436 | 8 | −0.3289 |

`beats_best_of_n_on_all_head_seeds` is **false** for all three arms. The one arm that fires
C23's Rule B **changes sign** under head seed 3456: +0.0760 → +0.0366 → −0.0156. The
head seed also moves the comparator, because N is re-solved from the arm's own tokens per
molecule and head seed 2345's guided run is slightly more expensive (393.80 vs 387.79), which
pushes its N from 8 to 9 — so head seed 2345's +0.0366 is already measured against the
token-conservative comparator and is nearly identical to C23's own post-hoc conservative
figure of +0.0369.

Applying C23.4.2's token-conservative comparator to head seed 3456 as well (its realised
ratio against N=8 is 387.79 / 355.87 = 1.0897, over the 1.05 ceiling, so the check applies):
against N = 9 at 399.92 tokens per molecule and hit rate 0.5234, the advantage is
**−0.0547**. Under the comparator C23 itself said was the honest one, the arm loses at head
seed 3456.

**(c) §C25.0.7's pre-committed verdict.** The commitment, written before any of these numbers
existed, was: *if the three head seeds' end-to-end hit rates for the HBD-count arm span more
than 0.05, C23's Rule B is reported as not replicated at the head-seed level, whatever the
mean does.* The span is **0.0916**. Therefore:

> **C23's Rule B is NOT REPLICATED at the head-seed level.**

The mean over the three head seeds (0.5297) still exceeds the mean of the three matched
best-of-N runs (0.4974), and one could construct a pooled estimate that looks positive. The
pre-registration forbids doing that, and this section does not do it. The aromatic-ring arm's
span (0.0773) also exceeds the limit, though nothing turns on it because that arm never came
close to its best-of-N under any head seed.

### C25.4.2 Quality, for completeness

Uniqueness is 1.0000 on all nine runs. Validity is above 0.979 everywhere, but head seed 3456
on aromatic rings sits at 0.9798 against head seed 1234's 0.9941 — a drop of 0.0143, which
exceeds §C25.0.6's 0.01 disqualification threshold, and its tokens per molecule (434.85
against 412.50, ratio 1.0542) exceed the 5% ceiling. Had that arm been a pooling arm rather
than a replication arm it would have been disqualified. It is reported here rather than
scored, because §C25.0.6's disqualification rule is written for the pooled end-to-end arms.
The synthetic-accessibility comparison on guided hits against base-policy hits reproduces
C23's sign in every arm: SA score differences of −0.1111 / −0.2042 / −0.1935 for the three
HBD-count head seeds (negative is better), all three with intervals excluding zero.

---

## C25.5 Scoring the pre-registration, rule by rule, including where it fails

Every rule in §C25.0.4, §C25.0.5 and §C25.0.8 is scored below, in the order it was written,
with no rule dropped. The verdicts are read from `outputs/c25_summary/c25_metrics.json` and
`tests/test_pooled_readout.py` binds them to this text.

### C25.5.1 The prediction rules

| rule | criterion | measured | verdict |
| --- | --- | --- | --- |
| **Rule P1** | ≥ 4 of 6 properties have a pooled variant that HELPS at probe point 12 | **5 of 6** (all but cLogP) | **FIRES** |
| **Rule P2** | ≥ 4 of 6 at the mid probe point | **5 of 6** (all but aromatic_rings) | **FIRES** |
| **Rule P3** | `wide1` HELPS at ≥ 4 of 6 mid cells | **0 of 6**; mean margin −0.0026 | **does not fire** |
| **Rule P4** | a pooled variant at pp 12 reaches aromatic-ring AUROC ≥ 0.8269 + 0.010 = 0.8369 | best is **0.8257** (`mean16`), i.e. **−0.0011** *below* the trivial head, not +0.010 above it | **does not fire** |
| **Rule N** | none of P1–P4 fires | P1 and P2 both fire | **does not fire** |

**The pre-registration's headline prediction is falsified.** §C25.0.8 predicted Rule N would
fire and that no pooled variant would HELP at ≥ 4 of 6 properties at either depth. Both
depths came in at 5 of 6. Pooling is the first readout intervention since depth to move the
prediction metric materially, and — unlike depth — it does so with an operator that has no
parameters at all.

Two of the three reasons given for the prediction were wrong in an instructive way. Reason
(i), that adjacent states are correlated so a mean "mostly smooths", was right about the
mechanism and wrong about the sign of its usefulness: smoothing is exactly what helps, which
is why the 16-position unweighted mean beats the 4-position learned attention. Reason (ii),
that `concat4` is "a capacity increase in disguise", was right — and `concat4` is the worst
pooled variant, for that reason.

### C25.5.2 The end-to-end rules — **not scored, because no pooled arm was run**

This is the most important limitation of the section and it is stated before the rules rather
than after them.

**Trigger T fired, and it fired widely.** 15 of the 54 comparisons are on an anchor property
(aromatic_rings, hbd_count, qed) with verdict HELPS. §C25.0.5 caps end-to-end arms at 6,
taken in descending order of AUROC margin; the six that qualified are:

| # | property | depth | probe point | variant | AUROC margin | status |
| ---: | --- | --- | ---: | --- | ---: | --- |
| 1 | hbd_count | final | 12 | `mean16` | +0.0382 | **not run** |
| 2 | aromatic_rings | final | 12 | `mean16` | +0.0380 | **not run** |
| 3 | qed | mid | 4 | `mean16` | +0.0275 | **not run** |
| 4 | qed | final | 12 | `mean16` | +0.0269 | **not run** |
| 5 | hbd_count | final | 12 | `attn4` | +0.0210 | **not run** |
| 6 | aromatic_rings | final | 12 | `attn4` | +0.0205 | **not run** |

The nine further qualifying cells, reported with their margins as §C25.0.5 requires, are
hbd_count/final/`mean4` +0.0166, qed/mid/`attn4` +0.0164, hbd_count/final/`concat4` +0.0156,
qed/mid/`mean4` +0.0139, aromatic_rings/final/`concat4` +0.0139, hbd_count/mid/`mean4`
+0.0130, qed/final/`attn4` +0.0117, qed/final/`mean4` +0.0104, and
aromatic_rings/final/`mean4` +0.0183 (which ranks 7th by margin and is over the cap).
**Insurance arm I** — the highest-margin pooled variant for hbd_count at probe point 4,
which is `mean16` at +0.0155, at λ = 2, fixed in advance and committed to run regardless of
the trigger — was **also not run**.

Consequently:

* **Rule E1** (pooling improves guidance end to end) **does not fire** — vacuously: zero arms
  were run, so the criterion "≥ 2 of the anchors run" is satisfied by nothing.
* **Rule E2** (a pooled arm beats its own compute-matched best-of-N) **does not fire** —
  vacuously, for the same reason.
* **Rule E3** (null end to end) **does not fire** either, because `outputs/c25_guided_*` is
  empty and E3 is defined only over arms that exist.

**None of these three is evidence about pooling.** They are the artefact of an unexecuted
stage and must not be quoted as an end-to-end null. §C25.0.5 said in advance that "an AUROC
improvement is therefore not the result" and that improved AUROC is "a licence to run
end-to-end guided generation, never a substitute for it". The licence was granted by the
data and not exercised. **C25 therefore reports a prediction result and no steering result.**

The project's own record is the reason this matters rather than being a formality. C17
improved AUROC with depth and C18 then showed that calibration and capacity did not convert
into guidance; C23 showed a mid-network head *did* convert, but only into hit rate over
unguided, and its one falsification of best-of-N is the arm §C25.4 has just failed to
replicate. Two of the three precedents say a prediction gain does not transfer. The honest
statement of C25's status is: **pooling is the first cheap fix in this project to pass the
prediction test at both depths, and the only one whose steering test has not been run at
all.**

### C25.5.3 The five sub-predictions

| prediction | claim | outcome |
| --- | --- | --- |
| **P-A** | `concat4` HELPS at strictly more cells than `mean4` + `mean16` combined | **FALSIFIED.** `concat4` 3 cells; `mean4` 6 + `mean16` 9 = 15 |
| **P-B** | `mean16` is the worst pooled variant by mean margin over the 12 cells | **FALSIFIED, and in the opposite direction.** `mean16` is the **best** (+0.0213 mean); `concat4` is the worst (+0.0057) |
| **P-C** | Rule P4 does not fire; no pooled variant reaches 0.8369 for aromatic rings at pp 12 | **CONFIRMED.** Best is 0.8257 |
| **P-D** | Rule P3 does not fire; `wide1` is INDISTINGUISHABLE at ≥ 4 of 6 mid cells | **CONFIRMED**, at 6 of 6 |
| **P-E** | the three head seeds' hit rates for hbd_count pp 4 λ=2 span < 0.05, with 0.5603 inside | **FALSIFIED.** Span **0.0916**. 0.5603 does lie inside the span, but the span is 1.8× the limit |

Three of five falsified, including the headline. The two confirmed are the two negative
predictions — capacity does not bind, and pooling does not rescue aromatic rings — which are
also the two the previous sections had already made likely.

---

## C25.6 What was not run

Stated explicitly, as §C25.0.9 requires, and dropped from the end of the priority order
rather than because of what any stage showed.

- **Every pooled end-to-end arm — priority 7, the whole of it.** Six triggered arms plus
  insurance arm I, listed with their margins in §C25.5.2. This is the single largest gap in
  the section and the reason it answers a prediction question and not a steering one. Cost
  estimate for a follow-up, at C23's realised rate: 7 arms × (2 conditions × 3 seeds × 512
  molecules) plus matched best-of-N, roughly 9–10 M processed tokens.
- **Everything downstream of those arms**: the seed-stratified bootstraps at
  `n_boot` = 10000, the realised token ratios against best-of-N, the C25.0.6 quality
  disqualification pass, and Rules E1/E2/E3 as substantive rather than vacuous verdicts.
- **`concat16`, `attn16`, weighted or positional pools, and any window between 4 and 16.**
  §C25.0.2 fixed the family of five before results existed and forbids adding to it now that
  `mean16` is known to be the best; the window-length trend is a pre-registered question for
  a successor, not a variant to append here.
- **Pooling at any probe point other than 12 and the six frozen mid points.** In particular
  the interaction "does the pooling gain keep shrinking as depth increases?" is measured at
  exactly two depths.
- **Head-seed replication of any C23 arm other than the three in §C25.0.7**, and of any arm
  at a λ other than the one C23 ran.
- **`scripts/09_confound_analysis.py`** on the replication arms; only script 05's native
  length-matched estimator was computed.
- **Any full-recompute end-to-end accounting** (§19.2), and any wall-clock claim (§11.7).

---

## C25.7 Contradictions with the existing report

Flagged here and **not** edited into `reports/pilot_report.md` or
`reports/section_c23_layer_end_to_end.md` by this section. The owner should merge them.

1. **C23's Rule B — `reports/section_c23_layer_end_to_end.md` §C23.4.2 and its summary row
   "C23.0.6 Rule B | some arm beats its own matched best-of-N | **fires**, one arm, +0.0760".**
   That row is now known to rest on a single head seed whose end-to-end hit rate varies by
   0.0916 across head seeds and whose advantage over its own compute-matched best-of-N is
   +0.0760 / +0.0366 / **−0.0156**. Under §C25.0.7's pre-committed rule, **Rule B is not
   replicated at the head-seed level.** C23's own §C23.7 anticipated this ("one arm carries
   Rule B, and it is marginal... a replication is what would settle it, and none was run") —
   the replication has now been run and it does not settle it in C23's favour. The
   recommended edit is to keep the row and annotate it "not replicated across head seeds
   (C25.4)", not to delete it.
2. **C23's §C23.6 "What was not run" entry "Head-seed replication".** It is now run for three
   arms and should point at `outputs/c25_hs2345_*` / `outputs/c25_hs3456_*`.
3. **§8.3's open suspect list.** "Whether an earlier layer, a larger head, or a different
   pooling would close the aromatic-ring gap is untested." All three are now tested at the
   prediction level and **none closes the aromatic-ring gap**: earlier layer (C17, 0.8474 at
   pp 3 — which does clear the trivial 0.8269, so depth alone does close *that* gap), larger
   head (§20.4 and C25's `wide1`, −0.0026), pooling (0.8257 at pp 12, still below trivial).
   §8.3's sentence should be replaced by the measured outcome for each of the three, and the
   distinction drawn between "closes the gap at pp 12" (nothing does) and "closes the gap by
   moving layers" (depth does).
4. **Any claim anywhere that the final-layer readout is capacity-bound or
   pooling-invariant.** Pooling at probe point 12 moves target AUROC by +0.0221 to +0.0382 on
   five of six properties, which is the same order as the depth effect §21 measured. No
   existing sentence asserts pooling-invariance, but §20.4's "the readout is not
   capacity-bound" should be narrowed explicitly to *capacity*, since a zero-parameter
   16-position mean is not a capacity increase and does move the number.

---

## C25.8 Limitations

- **No steering result.** See §C25.5.2. Every P-rule verdict in this section is a statement
  about a held-out AUROC on 199,300 prefix rows and about nothing else. This project has
  already produced one clean dissociation between prediction and steering (C18) and one
  partial one (C23.4.1, where the per-position column mis-ranked the end-to-end arms), so the
  prior that a +0.038 AUROC gain converts into guidance should be treated as weak.
- **The margins are small in absolute terms.** The largest is +0.0382 and the median over the
  48 pooled comparisons is **+0.0116**, against a pre-registered material
  margin of 0.010 and a head-seed sd of ~0.004. Many cells clear the bar by less than one
  material margin, which is why the verdicts are given per cell with intervals and why "5 of
  6" is reported rather than a headline number.
- **`mean16` wins almost everywhere, and the family stops at 16.** The trend in window length
  is monotone across the three lengths tested (1, 4, 16), so the family's upper end is very
  likely not the optimum and the reported best-case is a lower bound on what a longer window
  would give. That is an argument for a successor experiment, not for reading these numbers
  as the ceiling.
- **The two depths are not independent of each other.** The mid probe points were frozen from
  §21's AUROC-best table, which is itself an argmax over 13 probe points on the same test
  split. `last1`'s mid-depth AUROC is therefore mildly optimistic as a baseline, which makes
  the mid-depth pooling margins mildly *conservative*, not the reverse — but the two depths
  should not be treated as two independent replications of the same question.
- **The head-seed replication is three arms and three seeds.** A span of 0.0916 estimated
  from three points is itself noisy. What it establishes is that head-seed variance in an
  end-to-end guided run is of the same order as the effects C23 measured — which is enough to
  demote a single-arm falsification, and not enough to put a confidence interval on it. Six
  or ten head seeds on the HBD-count arm is the cheap follow-up.
- **Aromatic rings behaves unlike the other five properties at both depths** — it is the one
  property where pooling does not help at the mid layer, and the one where the depth effect
  is largest. It is also the property §8.3's negative result is about. Any general claim about
  pooling should be checked against it separately.

---

## Commands to add to `docs/REPRODUCE.md`

A section "C25 — the pooled readout", to run after the phase-2 chain (P0–P7), after C17
(`outputs/c17_layer_states_pilot_50k_p2/`, `outputs/c17_probe_layers/`) and after C23
(`outputs/c23_guided_*`, `outputs/c23_bestofn_*`, `outputs/c23_quality_*`, needed as the
seed-1234 leg of the head-seed replication).

**Step 0, and it is not a script.** §C25.0 was written into `reports/section_c25_pooling.md`
and copied by hand to `outputs/c25_prereg/C25.0_preregistration.md` **before any C25 output
directory existed**, with `outputs/c25_prereg/prereg_lock.json` recording
`source_sha256`, `prereg_sha256`, `prereg_bytes`, `frozen_at_utc`
(`2026-07-31T09:59:06Z`) and `git_head` (`fd8abb0`). There is deliberately no
`--freeze-prereg` flag: a freeze that a results script can re-run is not a freeze. What
enforces it afterwards is
`tests/test_pooled_readout.py::test_frozen_prereg_is_a_verbatim_substring_of_the_section`,
which fails if §C25.0 in the prose is ever edited away from the frozen copy. To re-check the
lock by hand:

```bash
sha256sum outputs/c25_prereg/C25.0_preregistration.md   # must equal prereg_sha256
```

```bash
# R1  re-extract the window of hidden states at probe points 3, 4, 5, 12.
#        2,205,784 processed tokens, one forward pass for every probe point and both
#        window sizes. The gate must come back bit_identical=true /
#        max_abs_difference=0.0 at every probe point; if it does not, stop --
#        everything downstream is void (§C25.0.1 limb A). Writes ~12 GB.
setsid nohup .venv/bin/python scripts/20_extract_pooled_states.py \
    --dataset pilot_50k_p2 --layers 3 4 5 12 --stack-window 4 --wide-window 16 \
    --batch-size 96 --out c25_window_states_pilot_50k_p2 \
    > outputs/c25_extract.log 2>&1 &

# R2  the 66-cell sweep at three head seeds. CPU only, no generation, no tokens.
#        The last1 validity gate (§C25.0.1 limb B) runs FIRST over all 18 gate cells
#        and the log ends with "validity gate max residual: 0.0 (PASS)"; if it ends
#        any other way, stop. Saves all 198 head checkpoints, which is what R3
#        and R4 consume. Resumable: --no-resume to force a full retrain.
setsid nohup .venv/bin/python scripts/20_pooled_sweep.py \
    --dataset pilot_50k_p2 --states c25_window_states_pilot_50k_p2 \
    --reference-heads pilot_50k_heads_p2 --reference-c17 c17_probe_layers \
    --head-seeds 1234 2345 3456 --n-boot 1000 --out c25_pooled_heads \
    > outputs/c25_sweep.log 2>&1 &

# R3  head-seed replication of three C23 arms at head seeds 2345 and 3456
#        (§C25.0.7). GPU. It is a driver: every generation, best-of-N and quality
#        call is a subprocess into the UNMODIFIED scripts 05 / 06 / 10, using
#        R2's head_<prop>_last1_L<pp>_seed<hs>.pt checkpoints. 7,618,055
#        processed tokens in total. Idempotent -- a completed arm is skipped, so a
#        kill loses at most one arm. --stage guided | analysis | all.
setsid nohup .venv/bin/python scripts/20_head_seed_replication.py --stage all \
    > outputs/c25_headseed.log 2>&1 &

# R4  end-to-end guided decoding with a pooled readout. NOT RUN in this section
#        (§C25.6); the commands are recorded so the six triggered arms of §C25.5.2
#        plus insurance arm I can be executed unchanged. The variant and the window
#        are read from the checkpoint, not passed -- there is no --variant flag,
#        which is why the checkpoint name carries the variant. Seeds default to
#        configs/guidance.yaml's 101/202/303 and n to 512 per condition.
#        Shown: arm 1, hbd_count / probe point 12 / mean16 / lambda 2.
.venv/bin/python scripts/20_pooled_guided_generation.py --dataset pilot_50k_p2 \
    --property hbd_count --layer 12 --lam 2 \
    --head-file c25_pooled_heads/head_hbd_count_mean16_L12_seed1234.pt \
    --conditions unguided throughout --out c25_guided_L12_mean16_lam2_hbd_count
.venv/bin/python scripts/06_best_of_n.py --dataset pilot_50k_p2 \
    --property hbd_count --guided c25_guided_L12_mean16_lam2_hbd_count \
    --accounting actual --out c25_bestofn_L12_mean16_lam2_hbd_count
.venv/bin/python scripts/10_quality_analysis.py --dataset pilot_50k_p2 \
    --property hbd_count --guided c25_guided_L12_mean16_lam2_hbd_count \
    --out c25_quality_L12_mean16_lam2_hbd_count

# R5  assemble, bootstrap and score every decision rule. Reads only; generates
#        nothing. Must be re-run after any new arm, because the end-to-end
#        Bonferroni alpha is 0.05 / (number of c25_guided_* directories).
.venv/bin/python scripts/20_summarise_c25.py

.venv/bin/python -m pytest tests/test_pooled_readout.py -p no:cacheprovider
```

`src/property_to_go/guidance.py`, `scripts/05_guided_generation.py`,
`scripts/06_best_of_n.py` and `scripts/10_quality_analysis.py` are **unmodified** by C25. The
pooled decoder is a separate transcription in `src/property_to_go/pooling.py`, and the
decoder identity test at window 1 is what stops the copy drifting from the original. The
`--layer` / `--head-file` arguments used in step R3 are the ones C23 added to script 05; C25
added none.

**Artefact sizes, measured.** `outputs/c25_window_states_pilot_50k_p2/` is **12 GB**: per
probe point a `stack.npy` of shape (199300, 4, 768) float32 — the 4-position window, which
`last1`, `mean4`, `concat4` and `attn4` are all computed from — and a `mean16.npy` of shape
(199300, 768) float32, the 16-position window **already reduced to its mean at extraction
time**, since a uniform mean is precomputable and storing 16 × 768 floats per row is not.
`counts.npy` and `counts16.npy` (int16, one entry per row) record how many of the 4 and 16
slots are distinct, so no averaging operator ever averages over an index-clamped repeat. The
whole directory is **regenerable in one forward pass**, so it can be deleted once
`outputs/c25_pooled_heads/` exists. `outputs/c25_pooled_heads/` is **489 MB**: 66
cell JSONs, `pooled_metrics.json`, and **198** head checkpoints
(`head_<property>_<variant>_L<probe point>_seed<head seed>.pt`). It **must stay** — the
checkpoints are what a pooled end-to-end run in step R4 consumes, and the `last1` mid-layer ones
are the only reason the head-seed replication was a generation job rather than a retraining
job. `outputs/c25_hs2345_*` and `outputs/c25_hs3456_*` are six guided directories of ~13 MB
`molecules.json` each plus their best-of-N and quality JSON.
`outputs/c25_summary/c25_metrics.json` must stay: every number in §C25.2 to §C25.5 is read
from it by `tests/test_pooled_readout.py`. The `outputs/c25_smoke_*` directories are not
results and can be deleted.
