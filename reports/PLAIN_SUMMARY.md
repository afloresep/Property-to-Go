# Property-to-Go: what we asked, what we found

A plain-language summary followed by the technical detail. Written 2026-07-30.

For the artifact-bound version of every number below, see [`pilot_report.md`](pilot_report.md).
To re-run anything, see [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).
To continue the work on other hardware, see [`../docs/HANDOFF.md`](../docs/HANDOFF.md).

---

## Part 1 — In plain English

### The idea

There is a kind of AI that writes molecules the way your phone writes text messages:
one piece at a time, guessing what comes next. Molecules are written as strings of
characters in a format called SMILES, so `CCO` is ethanol.

We asked two questions about a half-written molecule:

1. **Can you see the future?** If the AI has written half a molecule, can you already
   tell what the *finished* molecule will be like? Not what it is now — what it will
   become.
2. **Can you steer it?** Can you lean on the AI while it is writing so the finished
   molecule comes out the way you want? And does it matter *when* you lean — at the
   start, the middle, or the end?

We measured two properties of the finished molecules: how oily versus water-loving
they are (cLogP, a standard number in drug design) and how many benzene-like rings
they contain. Molecular weight was tracked only as a sanity check.

The rule throughout was that the molecule-writing AI is never retrained or modified.
It stays frozen. The only thing we add is a small, cheap side-predictor that reads
the AI's internal state.

### What we did

1. Took a real published molecule-writing AI (GP-MoLFormer) and locked it so it
   cannot change.
2. Had it write 50,000 molecules, recording at many points inside each one both the
   AI's internal state and the properties of whatever molecule it ended up with.
3. Trained a small predictor: given the internal state at a half-written molecule,
   guess the final property.
4. Compared it against a deliberately stupid baseline — just count how many characters
   have been written so far. If the fancy predictor cannot beat character counting, it
   has not learned anything real.
5. Used the predictor *during* writing. At each step the AI proposes its 8 best next
   characters; we re-rank those 8 to favour ones the predictor thinks lead to the
   property we want.
6. Ran the controls that keep us honest:
   - Did it only "work" because the molecules got bigger and longer? (Bigger molecules
     are oilier by default, so this could fake a result.)
   - Is steering better than the lazy alternative — generate many molecules normally
     and keep the best — at the same computational cost?
   - Are the steered molecules still sensible chemistry, or did we win the metric by
     producing junk?

### What worked

**Seeing the future is real.** From a half-written molecule you genuinely can predict
the final oiliness, and the prediction improves as the molecule nears completion. On a
scale where 0.5 is coin-flipping and 1.0 is perfect, it rose from 0.66 early to 0.88
near the end.

**Steering is real, and not an illusion.** The molecules did shift toward the target.
When we forced the steered molecules to have the same length and size distribution as
the unsteered ones, 97% (oiliness) and 92% (rings) of the effect remained. It is not a
size trick.

**The steered molecules are not junk** — which genuinely surprised us. Steered
molecules that hit the target are no harder to synthesise than base-model molecules
that hit the target, and are slightly *more* drug-like. The reason is structural: we
steer toward a bounded target *interval* that the model already reaches about 11% of
the time, rather than maximising a property without limit. Unbounded maximisation is
what produces the notorious garbage in this literature. One exception, described
below: steering *late* does measurably damage the molecules.

**The most interesting result was one we did not expect: being predictable and being
steerable are different things.**

- Oiliness is *easiest to predict* late but *hardest to steer* late. In hindsight this
  makes sense — by the time it is obvious, it is already decided.
- Ring count is *harder to predict* than plain character counting, yet it is the
  *easiest* thing to steer.

So "the model knows X" and "you can control X" are separate claims. Work in this area
often treats them as one.

### What failed

**The big one: steering loses to the lazy method.** Give the same compute budget to
"generate a pile of molecules and keep the best" and it wins by a wide margin.
Steering was 0.33–0.35 worse in hit rate under generous accounting and 0.53–0.74 worse
under strict accounting. This was a pre-registered kill criterion and it fired for both
properties under both accountings. As a practical tool, the method fails.

**The ring predictor was beaten by character counting.** Our small predictor could not
read ring count out of the AI's internal state as well as a one-line trivial feature
could. Honest reading: this is a fact about our readout, not necessarily about the AI.

**Steering breaks the predictor.** The moment you steer, you push the AI into writing
molecules it normally would not, and the predictor has never seen those. It became
badly miscalibrated: it predicted a 7.6% success rate where the true rate was 26.7%.
We were permitted one round of retraining on steered data; it helped by +0.03, which
changes nothing.

**Late steering for oiliness did essentially nothing** (+0.015, smaller than run-to-run
noise), and **late steering for rings is where quality degrades** (synthetic
accessibility worsened by +0.14, and the longest carbon chain grew) — the one place we
found the expected cheating behaviour.

Two process failures worth recording. We found and fixed a real bug in the scoring
code where a 4-ring molecule was treated as tied with a correct 3-ring one; it was
caught because a number disagreed with theory by 7 standard deviations. And wall-clock
timing on this machine swings 20–25% between bit-identical runs, so every timing
comparison in the report is below the noise floor and should not be trusted.

### The honest summary

The pilot **succeeds as a measurement and fails as an optimisation method.** We learned
something real about what these models know and when they know it. We did not produce a
useful molecule-design tool, and we have not dressed up the negative result.

---

## Part 2 — Technical detail

### 2.1 Setup

| Component | Value |
|---|---|
| Generator | `ibm-research/GP-MoLFormer-Uniq`, decoder-only, 46.8M params |
| Attention | Linear attention, ReLU feature map, random orthogonal projections |
| Positional encoding | Rotary; `max_position_embeddings = 202` |
| Vocabulary | 2362 tokens |
| Weights | **Frozen.** No fine-tuning, no LoRA, no RL, no activation edits |
| Config change | `deterministic_eval: false → True` (one field, no weights) |
| `transformers` | pinned 4.44.2 (5.x cannot load this revision) |
| RDKit | 2024.03.5 |
| Hardware | laptop CPU only; no GPU used anywhere |

The `deterministic_eval` flag matters more than it looks. Left at the released default
of `false`, the linear-attention layer redraws its random feature projections on *every
forward pass*, so the same prefix gives different logits on repeat calls and nothing is
reproducible. Set to `True`, the projections stored in the checkpoint are used. This is
the only configuration field we changed, and it changes no weights.

### 2.2 The head, and what "predicting the future" means

For a prefix `x_{<=t}` drawn from a completed trajectory whose final molecule has
property `y`, the head sees the generator's hidden state `h_t` at layer `L` and outputs
a categorical distribution over discretized bins of `y`:

- continuous properties (cLogP, MW) → `QuantileBinner`, equal-mass bins from the
  training distribution;
- count properties (aromatic rings) → `CategoricalBinner`, one bin per observed count.

`P(y in I | prefix)` is then an exact sum of bin probabilities, not a Gaussian tail
approximation. The head is a small MLP; only the head is trained.

Three input variants are trained for every property so the comparison is fair:

| Head input | Contents |
|---|---|
| `frozen_state` | the generator's hidden state only — the method under test |
| `trivial` | hand-made features from the prefix string (token count, atom counts, ring-open count, …) — the "is the model needed at all?" control |
| `combined` | both, concatenated |

Only `frozen_state` can be used *during* decoding, because guidance has to score a
candidate hidden state that does not correspond to a finished string. That the
`trivial` head sometimes wins is a finding, not a knob we were free to turn.

Splits are grouped by canonical completed molecule via a stable blake2b hash, so no
molecule contributes prefixes to more than one of train/val/test. Without grouping,
prefixes of the same molecule leak across the split and every number inflates.

### 2.3 Predictability results (Phase 4, rollout bank)

The honest measurement of "can you predict the future" is not "does the head recover
the one completion this prefix happened to come from". It is "does the head match the
distribution of completions the frozen generator *actually produces* from this prefix".
So we built a rollout bank: 800 held-out prefixes balanced across generation position,
32 independent base-policy continuations each, one bank reused for both properties.

**cLogP** — target-interval AUROC against individual rollout outcomes, by prefix
quartile:

| Quartile (prefix completeness) | `frozen_state` AUROC |
|---|---|
| Q1 (earliest) | 0.663 |
| Q4 (latest) | 0.883 |

The curve is monotone. Predictability rises with completeness, which is exactly what a
"future property" head should do — and note that the remaining uncertainty at Q4 is
real conditional spread, not head error.

**Aromatic rings** — the crossover. Ranked by Spearman correlation against the
empirical conditional mean over 32 rollouts, the `frozen_state` head leads early and
the `trivial` (token-counting) head leads late, and the frozen-state curve *declines*
late. This is asserted directly against the artifact in
`tests/test_report_matches_artifacts.py::test_the_aromatic_ring_crossover_is_real`, and
it replicated at both 10k and 50k on three estimators.

Reading: counting ring-opening characters in the prefix is a nearly sufficient statistic
for the final ring count, and our MLP does not extract it as cleanly from the hidden
state as a one-line feature does. This is a statement about this readout at this layer,
not a proof that the representation lacks the information.

### 2.4 Guided decoding

At each generated position, take the base model's top-8 candidate tokens and score

```
score(a) = log p_base(a | prefix) + lambda * log( P(y_final in I | prefix + a) + eps )
```

with `lambda = 1`, then sample from the re-normalised scores. This is FUDGE-style
future-discriminator guidance. Getting `P(y_final in I | prefix + a)` requires the
hidden state of `prefix + a` for all 8 candidates, which is the expensive part.

Two candidate backends were implemented and proven numerically equal
(`test_candidate_backends_agree`):

- `_candidate_states_full` — re-runs `prefix + a` from scratch, batched;
- `_candidate_states_cached` — uses the model's own released `use_cache=True` path.

Six conditions, all pre-registered, windows frozen in `windows.json` before any guided
result was inspected:

`unguided`, `throughout`, `early`, `middle`, `late`, `truncation_control`
(top-8 restriction with `lambda = 0`, which separates "the property term moved the
molecule" from "restricting to 8 candidates moved the molecule").

Windows are quantiles of the **pooled distribution of generated token positions**
(position `t = 1…n` for every trajectory of length `n`), not quantiles of final
lengths. Quantiles over final lengths would put the 33rd percentile near the median
molecule's *end*, making "early" cover almost the whole trajectory. This is the one
implementation note added to `README.md`.

### 2.5 Intervention-response results, 3 seeds

Hit-rate gain over `unguided`, mean ± sd across seeds 101/202/303:

| Condition | cLogP | Aromatic rings |
|---|---|---|
| early | +0.053 | +0.115 |
| middle | +0.057 | +0.131 |
| late | +0.015 ± 0.022 | +0.065 |

The dissociation is the point:

- **cLogP** is most predictable late (AUROC 0.883) and least steerable late (+0.015,
  inside seed noise). It retains ~10% of the full effect late.
- **Aromatic rings** are less predictable than token counting, yet most steerable, and
  retain ~22% of the full effect late.

Predictability and controllability move in opposite directions across these two
properties. That is a double dissociation, and it is our most novel claim.

### 2.6 Confound analysis (Phase 3 requirement)

A controller that only makes molecules longer or heavier would look like a property
controller, because both target properties correlate with size. So each condition was
re-standardised onto the `unguided` stratum distribution under four estimators:

| Estimator | Strata |
|---|---|
| `raw` | none |
| `length` | content-token count, bin width 5 |
| `size` | heavy-atom count, bin width 3 |
| `joint` | both |

Every standardised estimate is reported with explicit `coverage` (the fraction of the
reference distribution that had any matching condition molecules), because a matched
estimate over 40% of the reference distribution is not a matched estimate.

Result: **97%** (cLogP) and **92%** (aromatic rings) of the `throughout` effect survives
joint length-and-size matching, at coverage 0.98–1.00. Pre-registered rejection
criterion R2 does not fire. The effect is not a size artefact.

### 2.7 Compute-matched best-of-N (Phase 5) — the decisive negative result

Guidance is compared against best-of-N sampling from the frozen base policy, matched on
**processed generator tokens**, including the full-prefix recomputation that candidate
scoring requires. Matching only Python forward-call counts or returned molecule counts
would understate guidance's cost by roughly the candidate count.

Two accountings are reported because the honest number depends on whether you credit
guidance with the KV-cache optimisation:

| Accounting | What it counts |
|---|---|
| `actual` | tokens actually processed by the cached candidate backend |
| `full_recompute` | tokens a naive full-prefix implementation would process |

Guidance advantage (negative = best-of-N wins):

| Property | `actual` | `full_recompute` |
|---|---|---|
| cLogP | **−0.3488** | **−0.7355** |
| Aromatic rings | **−0.3312** | **−0.5291** |

Pre-registered rejection criterion R4 fires for both properties under both accountings.
This is not a close call: at matched token budget, best-of-9 sampling reaches hit rates
around 0.59–0.63 (cLogP) where guided decoding reaches around 0.26.

Wall time is reported separately and should be ignored: two bit-identical runs on this
machine differed by 20–25% while token counts matched to the digit, so every wall-clock
margin in this project is below the noise floor.

### 2.8 Are the steered molecules junk? (Phase 6, added 2026-07-30)

Hit rate is blind to *how* the target was reached. The classic failure in property
optimisation is a controller that satisfies the objective degenerately — a long alkane
tail to raise cLogP, fused rings to raise a ring count. So every saved molecule was
scored on descriptors the molecular-optimisation literature uses to catch this:
synthetic accessibility (SA, Ertl & Schuffenhauer), drug-likeness (QED), longest
acyclic carbon path, carbon fraction, maximum ring size, fragment count, formal charge.

The comparison that matters is **within the set of molecules that hit the target**.
Comparing all guided molecules against all unguided molecules would confound quality
with the property shift itself — oilier molecules are legitimately greasier and more
flexible. Restricting to hits holds the achieved property roughly fixed and asks only
how the molecule got there.

The `unguided (hit)` row is also **the baseline's actual output**: compute-matched
best-of-N returns base-policy samples selected for target proximity, so base-policy
hits are exactly the molecules best-of-N hands you. No extra generation was needed.

**cLogP**, molecules hitting [4.173, 5.038):

| Condition | n | SA | QED | longest chain | max ring | any degeneracy |
|---|---|---|---|---|---|---|
| unguided (= best-of-N output) | 163 | 2.793 | 0.610 | 2.276 | 6.147 | 0.031 |
| throughout | 405 | 2.706 | 0.644 | 2.388 | 6.030 | 0.037 |
| early | 245 | 2.756 | 0.624 | 2.355 | 6.033 | 0.029 |
| middle | 251 | 2.782 | 0.642 | 2.359 | 6.076 | 0.044 |
| late | 187 | 2.790 | 0.647 | 2.417 | 6.005 | 0.032 |
| truncation_control | 138 | 2.722 | 0.643 | 2.406 | 6.014 | 0.022 |

No SA difference excludes zero. QED is *higher* for guided hits
(throughout +0.033 [+0.007, +0.062]). Degeneracy rates are ~3% everywhere and
indistinguishable from base.

**Aromatic rings**, molecules hitting [3, 4):

| Condition | n | SA | QED | longest chain | max ring | any degeneracy |
|---|---|---|---|---|---|---|
| unguided (= best-of-N output) | 261 | 2.739 | 0.597 | 2.015 | 6.008 | 0.015 |
| throughout | 718 | 2.711 | 0.625 | 1.943 | 6.007 | 0.019 |
| early | 437 | 2.738 | 0.608 | 1.973 | 6.000 | 0.011 |
| middle | 463 | 2.767 | 0.606 | 2.050 | 6.037 | 0.024 |
| **late** | 360 | **2.882** | 0.589 | **2.214** | 6.058 | 0.022 |
| truncation_control | 280 | 2.784 | 0.609 | 2.121 | 6.014 | 0.007 |

Here **late guidance does pay a quality cost**, and it is the only condition that does:
SA **+0.143 [+0.055, +0.236]** and longest chain **+0.199 [+0.011, +0.412]** versus
base-policy hits, both excluding zero. Chemically this is what you would expect — to
add an aromatic ring when the scaffold is already committed, the model has to tack
something awkward onto a nearly finished molecule.

So the quality cost of steering is **localised to late intervention**, not a property
of steering in general. Early and throughout steering is free.

**Why steering does not produce the notorious garbage here.** Two structural reasons,
both worth stating because they bound the claim:

1. The objective is a **bounded target interval** that the base model already reaches
   ~11% (cLogP) and ~17% (rings) of the time — not unbounded maximisation. There is no
   pressure to run off the end of chemical space, because overshooting the interval is
   penalised exactly like undershooting.
2. `lambda = 1` keeps `log p_base` in the score at full weight, so the base likelihood
   is never overridden, only tilted.

A larger `lambda`, or an unbounded objective, would very plausibly reproduce the
degeneracies the literature reports. **We have not tested that**, and the λ sweep below
is where it would show up.

The one genuinely broken molecule we found is worth quoting, because it shows what
"valid" means here:

```
C.C.C.C=CC=c1[n-]c(CC(CC)CCC)c(CC(C)C)c1=CC.[CH2-]C[CH2-]
```

RDKit parses it, so it counts as valid; it contains three aromatic rings, so it counts
as a target hit. It is four loose methanes next to a charged ring and two carbanions.
It came from the `late` ring condition. It is now caught by the `multi_fragment` and
`net_charged` flags, and by a regression test.

### 2.9 Distribution shift, and the one permitted correction

The head is trained on base-policy prefixes, but guided decoding visits prefixes the
base policy rarely produces. Measured on guided prefixes, the original head predicted a
mean target probability of **0.076** where the observed rate was **0.267** — ECE 0.190.
It is under-confident by a factor of ~3.5 exactly where it is being used to make
decisions.

One round of data aggregation was run (guided prefixes and terminal outcomes added to
the head's training data, head retrained once, held-out guidance test repeated), as the
specification permits. It gave **+0.0305** hit rate (~2.3 SE). No conclusion changes.
No RL, no iteration.

This is the single largest known weakness in the negative result: "guidance does not
work" currently means "guidance does not work *at λ=1 with a head that is
under-confident by 3.5× on the states it is steering from*."

### 2.10 Pre-registered rejection criteria: what fired

| Criterion | Fired? |
|---|---|
| R1 (head no better than trivial features) | **Yes, aromatic rings only** |
| R2 (effect explained by length/size) | No — 92–97% survives joint matching |
| R3 (validity/uniqueness collapse) | No |
| R4 (loses to compute-matched best-of-N) | **Yes, both properties, both accountings** |
| Kill test criterion 2 | **Failed** |
| Kill test criterion 3 | Passed |

### 2.11 Verification

- 177 tests pass (`.venv/bin/python -m pytest`), including 16 that re-read each JSON
  artifact, format it the way the report formats it, and require it to appear in the
  report text — hand-transcription is the one error mode no amount of reasoning about
  the pipeline can rule out.
- Two claim-level assertions are asserted rather than narrated:
  `test_guidance_always_loses_to_compute_matched_best_of_n` and
  `test_the_aromatic_ring_crossover_is_real`.
- Every result directory carries `provenance.json` and `configs_used.json`.

### 2.12 Known limitations, stated plainly

1. **One λ value.** The specification fixed λ=1. Every "no effect" claim is really "no
   effect at λ=1".
2. **One probe layer.** One layer of twelve was used. The aromatic-ring negative result
   may be about the readout, not the representation.
3. **One head seed.** No replication over head initialisations.
4. **One model, two properties.** No evidence about generality.
5. **The comparison is maximally unfavourable to guidance.** RDKit properties are free
   to evaluate, so best-of-N can afford to score every candidate. Guidance's only
   structural advantage — needing a cheap neural estimate instead of a real measurement
   — is worth nothing here. We chose the hardest possible test case for our own method,
   and say so.
6. **Wall time does not reproduce** on this machine (20–25% between identical runs).

---

## Part 3 — What should be done next

Ranked by how much each would change the conclusions.

| # | Experiment | Why it matters | Cost |
|---|---|---|---|
| 1 | **λ sweep**, 0.5 → 10, both properties, all conditions | The single most exploitable weakness. Every negative claim is currently λ=1-specific. Also the experiment most likely to *reproduce the garbage-molecule failure mode* — quality should be re-scored at every λ. | ~6 guided runs per λ |
| 2 | **Calibrate the head on-policy** before concluding | The head is under-confident by 3.5× on guided prefixes. Concluding "steering fails" from a miscalibrated signal is premature. | cheap |
| 3 | **Probe-layer sweep**, all 12 layers | Decides whether the aromatic-ring crossover is about the readout or the representation. | 12 head trainings, no generation |
| 4 | **Head-seed replication**, 5 seeds | Tells us which small differences are real. | cheap |
| 5 | **Expensive-oracle regime** | The honest place to look for a win. Best-of-N wins because RDKit is free. If the property needed a docking run or an assay, the comparison could flip. This is a *reframing*, not a tweak, and it is the strongest remaining case for the method. | design work |
| 6 | Second model and/or third property | Defuses the single-model objection reviewers will raise. | moderate |

Items 1–4 are slow on this laptop (CPU only) and quick on an NVIDIA GPU. **None are
started.** They are new work, not part of the completed pilot.

A simulated NeurIPS-workshop review (recorded in [`ABSTRACT.md`](ABSTRACT.md)) reached the
same conclusion independently and sharpened it: run the λ sweep **and** an N sweep, so the
deliverable is a compute–accuracy *frontier* for both methods rather than one point each.
It ranked this above adding a third property.

---

## Part 4 — Is steering even the interesting question?

Worth asking directly, because the honest answer is "not on its own".

**The case against.** If the property is cheap to evaluate — and RDKit cLogP costs
microseconds — there is no reason to steer during generation. Generate molecules, measure
them, keep the ones you want. That is best-of-N, it is trivial to implement, and our own
experiment says it wins by a wide margin. Worse, the molecular-optimisation literature has
a standing objection to logP as a target at all: the PMO benchmark (Gao et al., NeurIPS
2022) **excludes logP from its task suite** because "adding carbons monotonically
increases the estimated LogP value... simply maximizing LogP is not a meaningful goal in
drug design". A reviewer who knows that paper will ask why we used cLogP.

**What survives the objection.** Three things, in decreasing strength.

1. **The measurement is the contribution, not the method.** "How much of a molecule's
   final properties are already determined halfway through writing it?" is a question
   about what the model represents and when. It does not depend on steering being useful.
   The rollout-bank predictability curve stands whether or not anyone ever steers.
2. **The dissociation is a claim about representations.** That one property is predictable
   late but unsteerable late, while another is steerable despite being poorly predicted,
   says something about the geometry of what the model encodes. This is the part a
   reviewer called "the single most theoretically interesting result", and it is also the
   part currently resting on only two properties.
3. **The negative result has a specific, checkable target.** Neither literature scan found
   *any* molecular steering paper that runs a compute-matched best-of-N baseline — while
   two adjacent fields (controlled text decoding, inference-time scaling for diffusion)
   treat that baseline as mandatory. Pointing at a missing control is a modest but real
   contribution, and it is cheap for anyone to verify.

**Where steering would actually matter, and we did not test it.** Best-of-N has to
evaluate the property on every candidate. Guidance needs only a cheap neural estimate. So
guidance's advantage grows exactly as the oracle gets expensive — docking, ADMET
prediction, DFT, an assay. We chose the two properties where that advantage is worth
nothing, which makes our negative result clean but narrow. The comparison is also
**oracle versus proxy**: best-of-N selects on the true RDKit value while guidance sees
only a learned head. `pilot_report.md` §5 scopes the claim accordingly, and this is the
strongest remaining case for the method.

**Bottom line.** Steering is the weaker half of this project. The predictability
measurement and the missing-baseline observation are the parts worth publishing; steering
is what makes them testable. If the work were rebuilt from scratch, the framing would lead
with "what does a half-written molecule already determine?" and treat guided decoding as
the instrument rather than the subject.
