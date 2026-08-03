# Property-to-Go: what we asked, what we found

A plain-language summary followed by the technical detail. Written 2026-07-30.

> **Phase 2 has since been executed** (lexical-locality test, six properties, RTX 4090).
> It changes three things in this document, all flagged inline below:
> the cLogP "predictor is under-confident" finding was largely a **bug in our own code**;
> the claim that cLogP is hardest to steer *late* is **withdrawn** because it did not
> replicate; and the pre-registered hypothesis phase 2 set out to test was **falsified** in
> the form it committed to. Everything else here stands, and the aromatic-ring results
> replicated on fresh data. Part 5 at the end summarises phase 2 in the same plain language.
> Full detail in [`pilot_report.md`](pilot_report.md) sections 11–18.
>
> **Two follow-ups have since been run** (sections 19–21), and they change one more thing in
> this document: the aromatic-ring result. We reported that the model does not represent how
> many rings a molecule will end up with, because counting characters beat our predictor. That
> is true of the model's **final** layer only — the middle of the model predicts it better
> than counting does. See the end of Part 5. The steering conclusions are unchanged.

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

> **Phase-2 correction — the first bullet is withdrawn, not just re-explained.**
>
> Two things went wrong with it. First the explanation: "by the time it is obvious, it is
> already decided" is a good story and it is not what happens. Measured directly — asking at
> each point what the *best possible* next character would do to the expected finished
> molecule — one character late in the string can move expected final oiliness by **more than
> the entire target range**. The lever is *bigger* late, not smaller.
>
> Second, and worse, **the measurement itself did not replicate.** On a fresh set of 50,000
> molecules, steering oiliness late buys +0.048 rather than +0.015, which is 24% of the full
> effect rather than 10% — and slightly *more* than ring count retains late (19%), so the
> comparison that made this interesting has reversed. We ran a control to check whether our
> own bug caused the change: it did not; the change is the fresh sample. The original number
> came from three runs whose spread (±0.022) was already as large as the effect.
>
> So the specific claim "oiliness is hardest to steer late" is **withdrawn**. The broader
> dissociation in the second bullet — ring count is predicted worse than character counting
> yet is the most steerable property of six — stands and replicated. See `pilot_report.md`
> §16.3 and §17.4.

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

> **Phase-2 correction.** About half of that miscalibration was **our bug, not the
> predictor's fault**. We had asked the predictor to estimate the chance of landing in a
> target range, but through a boundary error it was actually estimating the chance of
> landing in *half* that range — so of course it looked pessimistic by a factor of two.
> Fixed, the predictor is essentially perfectly calibrated on ordinary molecules. The
> remaining gap on steered molecules is real but smaller than reported here. Details in
> `pilot_report.md` §11.5–11.6.

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
| ~~1~~ | ~~**λ sweep**, 0.5 → 10~~ **DONE** — see "What the λ sweep found" below | It was the single most exploitable weakness, and it has been closed. | done |
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

---

## Part 5 — Phase 2, in plain English

Executed 2026-07-30 on a desktop graphics card, after everything above. Two things came out
of it: an answer to the question the pilot could not answer, and two bugs in our own code.

### The question the pilot could not answer

The pilot found that steering shifts molecules toward the target, but far less than simply
generating a pile of molecules and keeping the best. It could not say **why**. There were two
possibilities and no way to tell them apart:

1. **There is nothing to steer.** At most points in writing a molecule, no available choice
   of next character meaningfully changes how the finished molecule turns out. If so,
   steering is hopeless and no amount of tuning helps.
2. **Our steering is bad.** The lever exists and our little predictor is not pulling it.

So we measured the lever directly, without using the predictor at all. At 400 points inside
partly-written molecules, we took the eight characters the AI itself considered most likely
next, and for each one we finished the molecule 16 different ways and measured what came
out. 51,200 finished molecules in total. The spread between the best and worst of those
eight choices is, by construction, the most that *any* steering method could achieve at that
point.

**The lever is large, and it is large everywhere.** For all six properties, picking the best
of the eight characters the AI already proposes would roughly **double or triple** the chance
of landing in the target range at that step. And our steering method captures between
**5% and 11%** of that.

So possibility 1 is dead: there is plenty to steer. The pilot's negative result is about our
method, not about the task.

### But there is a third possibility the pilot did not consider

The eight candidate characters are not equally likely. The AI typically puts about **90%** of
its confidence on just one of them. The big levers tend to sit on the characters the AI
thinks are unlikely. Our method was deliberately set up to respect the AI's own preferences
(the setting called λ=1 keeps the AI's opinion at full weight), so it *structurally cannot*
reach for those levers however good its predictor is.

That is a different diagnosis from "bad predictor", and it points at a specific, cheap
follow-up: turn the steering strength up. That experiment is not done, and it is now the
obvious next one.

### The hypothesis we set out to test, and how it did

We were testing whether how *easy a property is to steer* is explained by how *directly it is
written into the molecule's text*. Ring count is written as specific characters, so it should
be easy; oiliness is spread thinly across the whole molecule, so it should be hard. We wrote
the predicted ranking of six properties down in code, before measuring anything, so we could
not quietly adjust it afterwards.

**In the units we committed to in advance, the prediction came out almost exactly backwards.**
The reason is mundane and we had flagged the risk before looking: the measure divides by the
width of each property's target range, and one property's range is 400 times narrower than
another's, so the division mostly measured range width rather than anything about text. In a
version of the measure that does not divide by range width, ring count does come out top as
predicted, but the rest of the ranking does not follow, and with only six properties nothing
here is statistically meaningful either way.

Also predicted, and also wrong: we expected the lever to shrink toward the end of a molecule
for the diffuse properties. It **grows**, for every property, by a factor of three to ten.
That directly contradicts the story Part 2 tells about why oiliness is hard to steer late.

### Two bugs in our own code, both found and both fixed

**One made the predictor look worse than it was.** We asked it for the probability of
landing in a target range. Through a boundary error it was computing the probability of
landing in *half* that range. So the pilot's finding that "the predictor is systematically
under-confident by a factor of two" was arithmetic, not a property of the predictor. Fixed,
it is essentially perfectly calibrated. This affected all four of our continuous properties
and none of the counting ones, and it was structural rather than bad luck — it would have hit
any continuous property. The pilot's discrimination numbers are unaffected, which we checked
rather than assumed.

**One is about reproducibility, and it is a caveat we had asserted the opposite of.** We had
documented that the large data files, which are too big to store, can always be regenerated
exactly from the recorded settings. On a graphics card they cannot: the random number
generator is a different one, so the same seed produces a *different* set of 50,000
molecules. The AI itself is unchanged — its numerical outputs match to five decimal places
and it proposes exactly the same eight candidate characters — but the sample is a fresh draw.
Consequence: the target ranges the pilot froze before looking at any result **cannot be
recreated on new hardware**, so phase 2 copies them across verbatim rather than
re-deriving them.

### What replicated

The pilot's most-defended finding — that plain character counting predicts ring count better
than reading the AI's internal state — **replicated to three decimal places on a completely
fresh set of 50,000 molecules**, with a different random initialisation, and it is immune to
the bug above by construction. So is the size of the steering effect for ring count
(+0.295 in phase 2 against +0.300 in phase 1).

We also finally checked whether the single random seed used for training the predictor
mattered. It does not: across three seeds the variation is about ten times smaller than the
differences being compared.

### How the hypothesis did, in the end

Six properties, ranked in advance by chemistry, then measured.

**The advance ranking was largely right.** The order we wrote down before measuring —
ring count, H-bond donors, rotatable bonds, TPSA, oiliness, drug-likeness, from
"most directly written into the text" to "least" — matches the measured order of how
steerable each property turned out to be, with one swap (oiliness steers better than
predicted, rotatable bonds worse). On a six-item ranking that is a correlation of 0.77,
which is suggestive rather than proven.

**Our attempt to measure the underlying quantity was wrong.** The number we designed to
capture "how directly is this written into the text" ranks the six properties almost exactly
backwards. The reason is arithmetic: it divides by the width of each property's target range,
and those widths differ by a factor of 400, so the division mostly measures range width. We
had flagged that risk in writing before looking at the data, which is the only reason we can
be confident that is what happened rather than guessing.

So the idea survives and the instrument does not. We report both, and we do not pretend the
version that worked was the version we committed to.

**One prediction is worth singling out because it was sharp and it held.** A different
research group's method finds H-bond donor count the *hardest* property to steer. Our
reasoning said it should be one of the *easiest* for us, because the two methods pull on
different things. It came out second-easiest of six.

**And one of our own earlier findings did not survive.** Part 2 above reports that oiliness
is hardest to steer near the end of a molecule, and builds an argument on it. Re-measured on
fresh data it is no longer true — oiliness is now steerable at the end about as well as
anywhere else, and slightly better than ring count is. We ran a control to check whether our
own bug caused the change; it did not. The original number came from three runs whose spread
was already as large as the effect, so it should not have been leaned on. We have withdrawn
that specific claim.

### What the λ sweep found

The obvious objection to everything above was that we only ever tried one setting of the
steering strength λ. So we tried six, from a quarter of the original to eight times it, on
three properties. Three things came out of it.

**Turning the steering up helps, then hurts.** The best setting is about twice what we used,
and it improves the hit rate by roughly a third to two thirds. Past that it gets *worse*, and
the reason is visible: at the highest setting, between 10% and 20% of what the model writes
is no longer a valid molecule at all. We are not overriding the model's chemistry knowledge
at that point, we are destroying it.

**It still loses to the simple baseline, at every setting.** The closest it ever comes is a
gap of 0.09 on H-bond donor count, against 0.22 at the original setting. That is real
improvement, and it is still a loss. So the negative result was not an artefact of a badly
chosen dial.

**The "garbage molecules" the literature warns about do show up — just not where we expected.**
At the highest settings, the molecules that hit the target are markedly worse: harder to
synthesise, and increasingly split into disconnected fragments. Interestingly, they are not the
long greasy tails that steering papers usually report. That is a consequence of aiming at a
*band* rather than a maximum: if you are told to hit a range, breaking the molecule into pieces
is a cheaper way to land in it than growing it. For one of the three properties (QED) the
setting that maximises the hit rate is *already* in the damaging regime, so the trade-off is not
hypothetical.

### Then we tried to fix the predictor, three ways, and none of them worked

The λ sweep measured the *steering strength* half of the diagnosis. The other half was our
predictor. There were three obvious cheap things to try, and we tried all three.

**1. Make the predictor's probabilities honest.** Our predictor was systematically
under-confident: it would say "10% chance of hitting the target" when the real answer was
closer to 27%. Fixing that is standard practice, it took about ten lines of code, and it made
the predictor demonstrably better — the calibration error dropped by a factor of three to six.
**It also made the steering worse, every single time, on every property.** Once we worked out
why, the result stopped being surprising and started being interesting.

The steering rule picks a token by weighing the model's own preference against the
predictor's. What it actually responds to is the *gaps* between the predictor's scores for
the eight candidates — not their absolute size. Under-confidence is an error in absolute
size. Squashing all the numbers toward the truth also squashes the gaps between them, and
squashing the gaps is mathematically the same thing as turning the steering strength *down* —
which the λ sweep had already shown makes things worse. We proved this rather than argued it:
a calibrated predictor at strength 1 and the raw predictor at strength 0.40 produce **the
identical 1,536 molecules**, not similar ones, the same ones. So the most obvious fix in the
book is not a fix at all; it is the dial we had already swept, wearing a different label.

**2. Make the predictor bigger.** Seven times the parameters. It changed the predictor's
accuracy by less than the noise between two random initialisations.

**3. Read from a different place inside the model.** The model has twelve layers, and every
number in this report until now came from the last one. We tried all thirteen possible
reading points. **This one found something real, and it went against us.** Every property is
predicted *best* from the middle of the model — around layer 3 to 5 — and the accuracy then
declines steadily to the final layer. Which means one of our own headline claims was too
strong: we had reported that the model does not really represent how many benzene-like rings
a molecule will have, because simply counting characters in the half-written string beat our
predictor. That is true of the model's **last** layer and false of its middle. The
information is in there; the top of the model discards it. We have rewritten that claim.

And yet: **the layer that predicts best is not the layer that steers best — for any of the
six properties.** Reading from the better layer did not improve steering. We had committed in
writing, before running it, to choosing the layer by prediction accuracy and only then
measuring steering, precisely because choosing the other way round would have let us report a
win that was really just picking the best of thirteen numbers after the fact.

### The one thing to take away

Not the hypothesis. Two things, and they fit together.

The ceiling measurement: **for every property, at every point in a molecule, there is a much
better choice available than the one our steering makes**. That turns "our method loses to the
simple baseline" from a dead end into a specific diagnosis.

And then the diagnosis narrows: **a better predictor is not a better steerer.** We made the
predictor better-calibrated, bigger, and read it from a better place in the model. All three
made it a better predictor by the usual measures. None of the three made the generation
better, and the first actively made it worse. The reason is that the number a predictor is
scored on — can it tell good prefixes from bad ones — is not the number the generator
consumes. That is a lesson about this whole style of method, not just about molecules.

One caution about how we first stated it. That measurement is made *one token at a time*,
and steering runs over a whole molecule — roughly forty decisions. An earlier version of this
write-up converted the one-token accounting ("about half the gap is the strength setting,
most of the rest is our predictor") into advice about which whole-molecule experiment to run
next. That conversion does not hold: the whole-molecule effect is twenty to fifty times the
one-token effect, and doing the arithmetic naively predicts hit rates above 100%. We have
withdrawn the inference and **measured** the strength-setting half instead, which is what the
λ sweep above is. The predictor half is still only measured one token at a time, and saying
so is more honest than the confident version we wrote first.
