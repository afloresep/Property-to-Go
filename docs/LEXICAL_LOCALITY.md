# The lexical-locality hypothesis, and the experiment that tests it head-free

Written 2026-07-30. This document states a hypothesis, an operationalisation, and a
pre-registered prediction, so that the analysis is fixed before the data exists.

> **STATUS: TESTED 2026-07-30.** Results are in `reports/pilot_report.md` §15–17. This
> document is **left as written** apart from §3.1, which was added before any headroom
> number was computed, and this banner. Nothing below has been adjusted to match the
> outcome — that is the whole point of it existing.
>
> Summary of how it did, so a reader is not misled by the confident tone below:
> **P1 fails in the units this document chose** (Spearman rho = −0.886 against the
> predicted ordering, because normalising headroom by target-interval width largely
> measures 1/width — rho = +0.698 against inverse width). **P2 fails, in the opposite
> direction from the prediction**: the diffuse properties' headroom *rises* with position
> rather than declining, and §2's account of why cLogP is unsteerable late is contradicted
> by direct measurement. **§6's discriminating case succeeds**: HBD count, SLIM's hardest
> property under additive latent steering, is among the most steerable by token choice.
> **P5 was not testable** without the λ sweep.
>
> The measurement this document proposed turned out to be worth more than the hypothesis
> it was designed to test. Headroom shows there is a large one-step lever at every position
> for every property, and that the deployed rule captures 5–11% of it — which answers the
> question the pilot could not answer, independently of whether locality explains anything.

Origin: a reviewer proposed this as the *alternative* explanation for our
predictability/controllability dissociation. It is a better hypothesis than ours, so we
are adopting it as the thing to test rather than waiting for it in review.

---

## 1. The idea in one paragraph

A property of the finished molecule has to be written into the SMILES string somehow, and
different properties are written in structurally different ways. Some properties are
created by **specific, discrete token events**: emitting a lowercase aromatic atom plus a
matching ring-closure digit *is* an aromatic ring. Other properties are **diffuse sums
over the whole string**: Crippen cLogP adds up a contribution from every atom, and no
single token moves it much. Our claim is that **steerability tracks this structural
property of how the target is lexically instantiated, not how predictable it is from the
model's hidden state.**

## 2. Why this explains our two odd results

**Aromatic rings: unpredictable by our head, yet the most steerable thing we tested.**

- In SMILES, an aromatic ring is written as lowercase atoms closed by a matching digit
  pair (`c1ccccc1`). Ring count in the final molecule is essentially a count of those
  events.
- So there is, at most steps, **a specific token whose selection increments the property
  by exactly one**.
- And the target interval `[3, 4)` is **exactly one count unit wide**. A single token
  choice therefore moves the molecule a full target-width. The lever is as large as the
  thing it has to move.
- This also explains the crossover: counting ring-open tokens in the prefix nearly
  determines the answer, which is why the *trivial* head beat the frozen-state head. The
  same locality that makes it trivially predictable from surface statistics makes it
  trivially steerable by token choice. **Predictable-by-counting and steerable-by-choosing
  are two faces of one structural fact.**

**cLogP: highly predictable late, barely steerable late.**

- Crippen cLogP is a **sum of atom-type contributions over every atom**. Each token
  contributes a little; none contributes much.
- The target band is 0.87 log units wide. No single token choice moves cLogP by 0.87.
- Late in the sequence, two effects compound: the sum is already mostly determined
  (which is exactly *why* it is predictable late), and the remaining tokens each have a
  small lever. **Predictability late and unsteerability late are the same fact viewed from
  two sides:** the property is predictable precisely because it is already decided, and
  it is already decided precisely because no remaining choice can move it.

That last sentence is the paper, if it survives testing. The dissociation is not a
mystery about representations; it is a near-tautology about *when* a quantity becomes
determined, plus a fact about *lever size*.

## 3. The quantity that matters: steering headroom

Define, at a prefix `x_{<=t}` with the base model's top-k candidate tokens
`a_1 … a_k`:

```
mu(a_i) = E[ y_final | x_{<=t}, a_i ]     estimated by K base-policy rollouts per candidate
headroom(x_{<=t}) = max_i mu(a_i) - min_i mu(a_i)
```

`headroom` is the **largest one-step effect on the expected final property that any
decoding rule could achieve at this position.** Three properties make it the right
measurement:

1. **It is head-free.** It involves no trained predictor, so it is not confounded by our
   head's miscalibration — which is currently the biggest hole in the report (§8.2:
   the head is under-confident by 3.5x on guided prefixes).
2. **It is λ-free.** It does not depend on guidance strength, so it survives the
   reviewer's main objection to the negative result. No λ sweep can exceed the ceiling.
3. **It is an upper bound.** Any guidance method, however good its head, is bounded by it.
   So `achieved_effect / headroom` cleanly separates two explanations that our current
   data cannot distinguish:
   - **"no lever"** — headroom is small, guidance cannot work here, and no amount of λ
     tuning or calibration will change that;
   - **"bad head"** — headroom is large and we captured little of it, in which case the
     negative result is about our head, not about the method.

Normalise by the target interval width to compare across properties:

```
relative_headroom = headroom / (hi - lo)
```

Prediction: `relative_headroom` for aromatic rings should be near or above 1 at most
positions (one token = one ring = one interval width), and well below 1 for cLogP,
declining with position.

### 3.1 Estimator details, fixed before any headroom number was computed

Added 2026-07-30, **before running `scripts/11_steering_headroom.py`**. These are
estimator choices, not changes to P1–P6. They are recorded here rather than decided
during analysis because two of them could otherwise be tuned into the predicted answer.

**The finite-sample bias, and the correction.** `mu(a_i)` is a mean over `K` rollouts,
so `max_i mu - min_i mu` over `k` noisy means is **biased upward**: `k` candidates with
identical true means still show a positive spread. Worse, the bias grows with rollout
variance, and rollout variance is larger for exactly the diffuse properties the
hypothesis says should score *low*. Uncorrected headroom would therefore manufacture
part of the predicted ordering out of noise — in the wrong direction, but by an amount
we could not bound.

So a **permutation null** is computed alongside every headroom estimate: pool a
prefix's `k × K` rollout values, repartition them at random into groups of the observed
per-candidate sizes, and recompute the spread. That is the spread expected if all `k`
candidates had the same true mean. Define

```
headroom_excess = headroom_raw - headroom_null
```

Both are reported. **`relative_headroom_excess` is the primary locality score.**
`tests/test_headroom.py::test_the_null_removes_the_finite_sample_bias` constructs two
synthetic properties with identical (zero) true headroom and very different rollout
variance, and asserts that the raw statistic ranks them 10× apart while the corrected
one does not.

**The primary steerability score** is the pilot's own effect size, so phase 2 is
measured on the axis phase 1 already reported: `throughout` hit rate minus `unguided`
hit rate, averaged over the three guidance seeds. Reported alongside, because the count
properties' base rates are set by the `q = 0.90` rule rather than matched to each other:

* `fraction_of_room_captured = lift / (1 - unguided)`, the share of the available room
  above the base rate that guidance closed;
* the lift measured against `truncation_control` rather than `unguided`, which is the
  control that isolates the property term from the top-8 restriction.

**Capture of headroom** (C7) is computed in probability units at a single position, so
that "achieved" and "ceiling" are the same kind of quantity:

```
base      = sum_i w_base(a_i)   * p(a_i)      base policy restricted to the top-k
guided    = sum_i w_guided(a_i) * p(a_i)      the rule the pilot actually ran
ceiling   = max_i p(a_i)
captured  = (guided - base) / (ceiling - base)
```

`w_base` is the renormalised base policy over the top-`k` — **not** the unrestricted
base policy, because that would fold the top-8 truncation effect into the answer, which
is the confound `truncation_control` exists to remove. `w_guided` is
`softmax(log p_base + lambda log(q + eps))` over the same candidates, i.e. literally
what `guidance.guided_sample` samples from, with `q` from the trained head. Aggregated
as `sum(achieved) / sum(available)` rather than as a mean of per-prefix ratios, because
`available` is near zero at many prefixes where the ratio is numerically meaningless
but would carry equal weight.

Note the asymmetry, which is the point: **`ceiling` is head-free and lambda-free;
only `achieved` involves the head.** So a small `captured` with a large `available` is
evidence against our head, while a small `available` is evidence against the method at
that position, whatever head it used.

**Sample.** The headroom prefixes are drawn with a different seed (7777) from the Phase
4 rollout bank's (4242), so headroom and the predictability curve are independent
samples of held-out prefixes rather than two views of the same 800.

## 4. Pre-registered predictions

Fix these before running anything.

| # | Prediction | Falsified if |
|---|---|---|
| P1 | `relative_headroom` ranks properties in the same order as steerability (hit-rate gain) | The ranks disagree, i.e. a property with large headroom steers poorly or vice versa |
| P2 | `relative_headroom` declines with position for diffuse properties (cLogP, QED, TPSA) and stays roughly flat for local count properties (rings, HBD, largest ring) | Both classes behave the same way |
| P3 | The trivial-feature head's predictive performance correlates with steerability **across properties** | No correlation, which would kill the surface-statistics account |
| P4 | Guidance captures a **larger fraction of available headroom** for local properties than diffuse ones | The captured fraction is constant across properties, which would mean the head, not the lexical structure, is the binding constraint |
| P5 | Where headroom is small, no λ recovers the loss to best-of-N | Some λ beats best-of-N in a low-headroom regime, which would falsify the whole account |
| P6 | **The size of the guidance-vs-best-of-N gap itself tracks locality** — small or competitive for local properties, large for diffuse ones | The gap is constant across properties, or ordered against locality |

P1 and P2 are the thesis. P5 is the link back to the negative result: if it holds, "guidance
loses" stops being a fact about λ=1 and becomes a fact about the property.

**P6 is the one that makes this a single paper rather than two.** The pilot currently has
two disconnected findings — a dissociation, and a loss to best-of-N. If the *size of the
loss* is itself predicted by locality, both become consequences of one variable. This
prediction came from the reviewer, and it is the cheapest available way to unify the paper:
it needs no new experiment beyond running the λ/N sweep on properties that span the
locality axis, which is already planned.

## 5. Two operationalisations of locality, and the circularity to avoid

**Primary (causal): headroom**, as above. Measures the actual effect of a token choice on
the final property. Not circular — it is an interventional quantity.

**Secondary (correlational): the trivial-feature head's performance.** We already train it
(token counts, atom counts, ring-open counts from the prefix string, no model internals).
Its accuracy is a ready-made measure of how much of the property is carried by surface
token statistics.

**There are two distinct circularity risks. One does not apply to us; the other does.**

**Risk A — feature-selection circularity: does not apply, and this is checkable.** If the
trivial head used features hand-picked per property (ring-open counts chosen *because* the
target is ring count), "surface features predict ring count" would be near-tautological.
We do not do that. `tokens.prefix_features` computes **one fixed, property-agnostic
vector** — token count, atom count, aromatic-atom count, ring-marker count, currently-open
rings, branch open/close/depth, bond count, dot count, bracket-atom count, per-element
counts for a fixed element list, and three ratios — and the identical vector is stored once
in `features.npy` and reused for every property. There is no per-property branch anywhere
in the featurizer. Any new property inherits the same set unchanged.

The residual objection, which should be answered in the paper rather than waited for: the
generic set *does* contain aromatic-atom and ring-marker counts, which are close to
sufficient statistics for aromatic ring count. That is not cherry-picking — no one building
generic SMILES token statistics would omit them — and it is **precisely the locality claim**.
Ring count is local exactly because generic surface statistics nearly determine it. The
right response is to state this openly, not to weaken the feature set.

**Risk B — retrodiction: applies, and pre-registration is the only fix.** The hypothesis
currently exists to explain two results we already have. That is a story, not evidence. It
becomes evidence only when locality scores are computed for *new* properties and a predicted
steerability ranking is committed **before** those properties' guidance runs. The predicted
ordering is recorded in §5 above and in
`properties.PREDICTED_LOCALITY_ORDER`, in code, so it cannot be quietly revised. If the
data disagrees, P1 fails and that is the finding.

Priority between the two measures:

- headroom is primary and the claims rest on it — it is interventional, so neither
  circularity applies to it at all;
- the trivial head is a secondary, much cheaper proxy;
- **the paper must report whether they agree**, and if they do not, headroom wins.

Additionally, **pre-register the predicted locality ordering from chemistry alone**, before
measuring either. Getting the ordering right in advance is much stronger evidence than
fitting it afterwards. Our prediction, from how SMILES writes each property:

| Property | Type | Predicted locality | Reasoning |
|---|---|---|---|
| aromatic ring count | count | **high** | one lowercase atom + ring digit pair = one ring |
| H-bond donor count | count | **high** | per-atom N-H / O-H, additive, locally visible |
| largest ring size | count | **high** | determined by ring-digit placement |
| rotatable bonds | count | medium | bond-local, but ring membership is a global veto |
| TPSA | continuous | medium | additive over polar atoms, but many small contributions |
| cLogP | continuous | **low** | sum of atom contributions over the entire molecule |
| QED | continuous | **low** | nonlinear function of eight descriptors |

## 6. The hypothesis makes a prediction about SLIM, and SLIM's own data supports it

SLIM (arXiv:2605.10831, verified) steers by **adding one vector to the residual stream at
all token positions**, in a sparse-autoencoder basis, to **edit an input molecule under a
Tanimoto-similarity constraint**. It reports that counting properties (HBA, HBD) resist
this — weaker baseline vectors actively *hurt* them (CAA: HBA −15.6, HBD −17.6 Acc@0.15),
and SLIM needs a dedicated gradient-alignment loss before counts are steerable at all.

Lexical locality predicts exactly that, and predicts the opposite for us:

- **Latent-vector steering asks whether the property varies along a linear direction
  applied uniformly across the sequence.** An integer count is a step function. Changing
  it while preserving similarity to an input molecule needs coordinated, multi-position
  structural change. Badly suited to one additive direction.
- **Token-choice steering asks whether some single next-token decision moves the
  property.** For a count, one does: emit a lowercase aromatic atom and a ring digit.
  Perfectly suited.

So the same property is hard for SLIM and should be easy for us, **because the two methods
exploit different degrees of freedom**, not because either is wrong. SLIM does not test
aromatic ring count at all, so it makes no claim in either direction about our property.

This is a much stronger position than a citation dispute: a hypothesis that predicts
*both* results is more credible than one that has to explain a competitor's away.

**Including H-bond donor count is therefore the key discriminating experiment.** It is
SLIM's hardest case. Locality says it should be easy for us — it is per-atom, locally
visible (N-H / O-H), and integer-valued with an interval one unit wide. If HBD count turns
out hard for token-choice steering too, the hypothesis is in trouble and P1 fails. Design
the experiment so that outcome is reportable.

## 7. Cost

Headroom needs, per property-free rollout pass: `n_prefixes x k_candidates x K_rollouts`
continuations. The property values are computed from completed molecules, so **one rollout
bank serves every property** — the same trick the existing Phase 4 bank uses.

At `n_prefixes = 300`, `k = 8`, `K = 16`: 38,400 continuations, versus 25,600 for the
existing bank. Roughly 1.5x a run that was already tractable on CPU, so comfortable on a
GPU. Scale `n_prefixes` up if the GPU is fast; keep the position balance across quartiles.

Implementation note: `generation.continue_from_prefixes` already does the work. Build the
extended prefixes `x_{<=t} + a_i` for each of the 8 candidates and pass them in as
ordinary prefixes. No new inference machinery is required — which also means
`test_candidate_backends_agree` still covers the numerics.

---

## 8. The two limitations to state prominently, not defend

Both are unfixable in four weeks, and naming them is worth more than hedging them.

**One property list, chosen by us.** Six properties selected specifically because they were
expected to span the locality axis is a small, non-random sample for a claim about a general
mechanism. Pre-registration and the fixed feature set answer "did you cherry-pick after the
fact"; they do **not** answer "did you cherry-pick a list that would confirm the story."
The partial mitigation is including **HBD count** as an adversarial case — a property whose
behaviour a different framework (SLIM) reports as the hard one. Say all of this.

**One tokenization scheme, and the thesis is about tokenization.** Lexical locality is a
claim about how GP-MoLFormer's SMILES vocabulary writes each property. It is therefore
*a claim about SMILES*, not about molecules or about the model. The sharpest test would be
**SELFIES**, where ring closures and branches are encoded completely differently — locality
would rank the same properties differently, so the hypothesis makes a concrete,
falsifiable, and currently untested prediction there.

Do **not** spend the four weeks on a second generator or a second serialization. Both are
out of scope for this round, and the original specification excludes alternative
serializations anyway. State the SELFIES prediction as the explicit next step; a reviewer
will think of it regardless, and naming it first reads as understanding the claim's scope
rather than as an omission.
