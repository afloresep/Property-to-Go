# Draft abstract and claim inventory

Status: **v7, written 2026-08-03 after C33.** Not submitted anywhere.

## Title, v7 (recommended)

**Your Baseline Has the Answer Key: Two Measurement Practices Decide Whether Probe-Guided
Decoding Beats Best-of-N**

This is `reports/PAPER_WORKSHOP_DRAFT.md`'s title, and v7 adopts it. v6 led with the
positive result (*Steer Harder, Not Deeper*); v7 leads with the measurement result, because
C33 turned the measurement half of the paper into **two** findings rather than one — the
field's baseline is oracle-advantaged, *and* our own summary of that fact failed its
pre-registered replication. A paper that demonstrates its own point twice, once against the
literature's habit and once against its own statistic, is a stronger ICBINB paper than one
that reports a narrow crossing.

Alternatives, both keeping the positive result in front: *Steer Harder, Not Deeper:
Probe-Guided Decoding Beats Best-of-N at Small Compute Budgets on Two Frozen Molecular
Language Models* (v6's); *A Knob That Loses a Race: Compute Frontiers for Probe-Guided
Molecular Generation*.

## Abstract v7 (≈350 words) — recommended

Inference-time steering of a frozen autoregressive generator is usually evaluated against
best-of-N sampling at a single matched compute budget, and usually loses. We report compute
**frontiers** instead of points, on **two independent frozen molecular language models** —
GP-MoLFormer-Uniq (46.8M parameters, linear attention) and a GPT-2-architecture model trained
on ZINC (87.3M, full attention) — and find the received result is largely an artefact of two
measurement practices.

**The baseline holds the answer key.** Best-of-N ranks candidates by the true RDKit property
of the finished molecule; guided decoding only ever sees a learned probe. Giving best-of-N
the same probe — an equal-information control — separates the two curves monotonically in N:
the oracle is worth **0.37 to 0.68 hit-rate points by N = 32**, and the two generators agree
on that curve to a mean absolute difference of 0.0081–0.0369. We first summarised this as a
single "oracle share" of the reported gap (0.86–0.88 on one generator), pre-registered its
replication on the second, and **it failed** — 0.42, undefined, and 1.44. The reason is
diagnosable and is not the generator: the share is normalised at whatever budget the steered
arm occupies, and the two generators' deployed arms sit at ~131 against 367–419 processed
tokens per molecule. **We report the curve and retire the ratio we published.**

**The probe's training seed is a first-class factor**, never reported anywhere. It moves
end-to-end hit rate more than the generation seed does (sd 0.0366 against 0.0213) and moves
chemical *validity* enough that one seed in eight ships a generator returning 17%
unparseable strings — which propagates through the token budget into the comparison itself.

With both corrections applied, a narrow positive result survives: sweeping best-of-N over
eleven values of N and guidance over its candidate-set size k, the two **cross** at small
budgets on both generators, against the oracle-selected baseline, and the crossing replicates
across **eight probe seeds** including the deployed configuration. A completed depth × λ
factorial identifies the ingredient as **steering strength, not probe depth**. Guidance
cannot follow best-of-N upward: its ~10× knob converts into −0.36 to −0.67 of *advantage*.

We report eleven falsified pre-registered predictions, four retracted numbers of our own,
and a statistic we used that is vacuous at n = 3.

## What changed between v6 and v7

| v6 said | v7 says | why |
| --- | --- | --- |
| "86–88% of the gap was the comparator's information advantage" | that is a **generator-1, single-budget** number; on generator 2 the same cell gives 0.42 / undefined / 1.44 | C33, `pilot_report.md` §25.3 |
| the oracle asymmetry is the project's most portable claim | the *curve* is portable to 0.008–0.037 across generators; the *ratio* is not portable at all | C33, §25.2 and §25.4 |
| lead with the crossing | lead with the measurement, and report the failed share as a second instance of the same point | the §3 restructure in `PAPER_WORKSHOP_DRAFT.md` |
| deployed gaps of −0.0292 / −0.0439 / −0.0522 against the equal-information curve | those are generator 1's; generator 2 gives −0.1025 / +0.0980 / +0.0355 at the matching cell | C33 headline table |
| nine falsified predictions, three retracted numbers | eleven falsified (Q4, Q5), four retracted (the share) | C33.9 |

**The framing this supports, in one sentence:** *probe-guided decoding is compute-efficient
and compute-inelastic, the efficient regime is reached by raising λ rather than by searching
for a probe layer, and most of the gap previously reported for this method class is a
baseline holding an oracle the method is denied — a fact that must be reported as a curve in
N, because every single-number summary of it we tried, including our own, failed to
replicate.*

---

Status of what follows: **v6, written 2026-08-03 after C30, C31 and C32. Superseded by v7
above; its fourth paragraph is the part C33 falsified.**

## Title, v6 (superseded)

**Steer Harder, Not Deeper: Probe-Guided Decoding Beats Best-of-N at Small Compute Budgets
on Two Frozen Molecular Language Models**

Alternatives: *A Knob That Loses a Race: Compute Frontiers for Probe-Guided Molecular
Generation*; or, leading with methodology, *Your Baseline Has an Oracle and Your Probe Has a
Seed*.

## Abstract v6 (≈350 words) — superseded by v7

Inference-time steering of a frozen autoregressive generator is usually evaluated against
best-of-N sampling at a single matched compute budget, and usually loses. We report compute
**frontiers** instead of points, on **two independent frozen molecular language models** —
GP-MoLFormer-Uniq (46.8M parameters, linear attention) and a GPT-2-architecture model trained
on ZINC (87.3M, full attention) — and find the received result is an artefact of where the
single point was placed.

Sweeping best-of-N over eleven values of N and guidance over its own candidate-set size k,
the two methods **cross**. At small budgets guidance is ahead on both generators — 8 of 30
and 5 of 30 measured cells sit above the best-of-N curve at their own budget — and this holds
against a best-of-N that selects with the **true RDKit oracle**, the maximally unfavourable
comparator. On the first generator we re-ran every crossing cell at **eight probe-training
seeds**: five replicate with intervals excluding zero and all eight seeds on the same side,
including the deployed configuration, whose margin *grows* from +0.0846 to +0.1044. One cell
reverses and is withdrawn.

A completed depth × λ factorial on the second generator identifies the ingredient:
**steering strength, not probe depth.** The λ main effect exceeds the depth main effect in
all six primary cells, the two are additive (no interaction interval excludes zero), and the
*deployed* readout — no mid-network probe, no per-generator re-selection — crosses once λ = 2
(+0.1768 [+0.1304, +0.2232]). Guidance nevertheless cannot follow best-of-N upward on either
generator: its ~10× compute knob converts into **−0.36 to −0.67** of advantage on all six
arms of the second generator, because best-of-N converts the same tokens faster. It is not
that k buys nothing — on the second generator it raises raw hit rate by up to +0.2347 — but
that k loses a race.

Two measurement practices explain most of the gap this literature reports. **The baseline's
oracle**: restricting best-of-N to the same learned probe shrinks the deployed gap from
−0.2472/−0.3532/−0.3715 to −0.0292/−0.0439/−0.0522, so 86–88% was the comparator's
information advantage. *(**Falsified as a general claim by C33** — those are generator-1
numbers at a generator-1 budget; see v7 and `pilot_report.md` §25.)* **The probe's training
seed**: never reported anywhere, it moves
end-to-end hit rate as much as the generation seed does, and moves *validity* enough that one
seed in eight ships a generator returning 17% unparseable strings.

We report nine falsified pre-registered predictions, three retracted numbers of our own, and
a statistic we used that is vacuous at n = 3.

## What changed between v5 and v6

| v5 said | v6 says | why |
| --- | --- | --- |
| one generator | **two**, independent architectures | C31 |
| the crossing needs a mid-network probe re-selected per generator | it needs **λ ≈ 2**; depth adds but is not required | C32's 2×2 |
| guidance converts none of its compute into accuracy | it converts compute into *accuracy* on one generator and into **advantage** on neither | C31 D3 |
| the deployed arm crosses | on GP-MoLFormer at λ=1; on the second generator at λ=2, on 1 of 3 properties | C31, C32 |
| depth is roughly half of C23's claim | depth is the **minority** factor on all three properties of the second generator | C32 |
| "a mid-network probe is really a higher λ" | true on 2 of 3 properties; **reverses** on the third (spread ratio 0.86) | C32 |

**The framing this supports, in one sentence:** *probe-guided decoding is compute-efficient
and compute-inelastic, the efficient regime is reached by raising λ rather than by searching
for a probe layer, and most of the gap previously reported for this method class is a
baseline holding an oracle the method is denied.*

---

Status of what follows: **v5, written 2026-08-03 after C23–C29.**

> ## v4's title is falsified by this project's own data. Do not use it.
>
> **"Better Probes Do Not Decode Better"** was written on 2026-07-31, when the only
> evidence about probe depth was C17's *per-position* proxy. C23 then ran the end-to-end
> experiment C17 said it could not substitute for, and a mid-network probe **does** decode
> better: paired by head seed at matched λ over eight seeds, +0.0941, +0.1270 and +0.0266
> on the three anchors, all three intervals excluding zero, and 3 of 3 again after C29's
> effective-λ correction. A better probe decoded better. The title says the opposite.
>
> The rest of v4's abstract has three further problems, all found after it was written:
>
> 1. **"loses to compute-matched best-of-N for every property"** is true only against a
>    comparator holding a free RDKit oracle. C27 re-ran the frontier with best-of-N
>    restricted to the same head guidance uses, and the deployed gap goes -0.2472/-0.3532/
>    -0.3715 → **-0.0292/-0.0439/-0.0522**. Roughly 87% of the headline was the oracle.
> 2. **The λ-rescale identity is not novel.** Dhariwal & Nichol, NeurIPS 2021
>    (arXiv:2105.05233 §4.4) state `s·∇ₓ log p(y|x) = ∇ₓ log (1/Z) p(y|x)^s` — the same
>    algebra, for classifier guidance in diffusion. Our contribution is the *measured
>    consequence* for a discrete reranker, not the identity.
> 3. **"6.9x-wider readout"** describes parameters; the hidden layer is 4× wider. Both are
>    true, they are different quantities, and the report used them interchangeably. Fixed
>    in `pilot_report.md` §20.4 and §21.5.3.
>
> v4 is kept below unmodified because it is the version the simulated NeurIPS panel
> reviewed (R1 4/10, R2 3/10, R3 5/10, all reject, venue consensus TMLR), and the review is
> part of the record.

---

## Title, v5 (recommended)

**Guided Decoding Beats Best-of-N Where Compute Is Cheap and Nowhere Else: A Compute
Frontier for Probe-Guided Molecular Generation**

Alternatives: *A Crossing, Not a Loss: Probe-Guided Decoding Against a Best-of-N Frontier
in a Frozen Chemical Language Model*; or, if the methodological finding is to lead,
*Your Baseline Has an Oracle and Your Probe Has a Seed: Two Ways Steering Comparisons Are
Reported Wrong*.

## Abstract v5 (≈350 words) — recommended, post-C23–C29

Inference-time steering of a frozen autoregressive generator is usually evaluated against
best-of-N sampling at matched compute, and usually loses. We report a compute **frontier**
for that comparison rather than the customary two points, on a single frozen released
chemical language model (GP-MoLFormer-Uniq, 46.8M parameters, no weight updates, FUDGE-style
reranking of the base model's top-k candidates), and find the received result is an artefact
of where the two points were placed.

Sweeping best-of-N over eleven values of N and guidance over its own candidate-set size k
(five values, a 10.2–11.1× span in processed tokens), the two methods **cross**. At small
budgets guidance is ahead: 8 of 30 measured guidance cells sit above the best-of-N curve at
their own budget — and this holds against a best-of-N that selects with the *true RDKit
oracle*, the maximally unfavourable comparator. Because a probe's training seed is itself a
source of variance we measure (below), we **re-ran every one of those cells at eight
probe-training seeds** against the same fixed oracle curve: five replicate with 95%
intervals excluding zero and all eight seeds on the same side, including the **deployed**
configuration, whose margin *grows* from +0.0846 to **+0.1044** [+0.0895, +0.1194]. One cell
reverses (+0.0007 → **-0.0341**, interval excluding zero on the negative side) and is
withdrawn. At larger budgets guidance cannot follow: over its full 10× compute span the
deployed arm moves **-0.0218** in hit rate while oracle best-of-N gains +0.6834 over the
identical budgets.

Two measurement practices explain most of the gap previously reported for this method
class. First, **the baseline's oracle**: best-of-N normally selects with ground truth while
guidance sees only a learned probe. Re-running the frontier with best-of-N restricted to the
same probe, probe point, interval and binning, the deployed gap falls from -0.2472/-0.3532/
-0.3715 to **-0.0292/-0.0439/-0.0522** — 86–88% of the reported gap was the comparator's
information advantage, not the method's weakness. Second, **the probe's training seed**: at
eight probe-training seeds the between-seed sd of end-to-end hit rate is 0.0142–0.0366,
comparable to the pooled generation-seed sd (ratios 0.63–1.71, every 95% F interval
containing 1) and to several published effect sizes. It is never reported. It also moves
**validity**, not only accuracy: on one arm at λ=2 the eight seeds span 0.8301–0.9792 in
fraction-parseable, so one seed in eight ships a generator returning 17% unparseable
strings — invisible to any single-seed protocol. We also show that
post-hoc calibration of such a probe is exactly a rescale of the guidance strength λ by the
fitted exponent — an identity known from diffusion guidance — and that it therefore helps or
hurts purely according to the sign of `log α`, which we demonstrate in both directions across
two architectures and two domains.

We report five falsified pre-registered predictions, one retracted number of our own, and a
statistic (the three-seed percentile bootstrap) that we used, that is vacuous, and that we
have withdrawn throughout.

## What changed between v4 and v5, in one table

| v4 said | v5 says | why |
| --- | --- | --- |
| better probes do not decode better | a mid-network probe decodes better, by about half of C23's raw margin | C23 end to end; C29's effective-λ correction |
| guidance loses to best-of-N, full stop | guidance and best-of-N **cross**; guidance wins below ~150 tokens/molecule | C26 + C28 frontiers |
| the gap is -0.22 to -0.37 | that is the gap to an **oracle-holding** baseline; the equal-information gap is -0.03 to -0.05 | C27 |
| guidance has no compute knob | it has one (k), spanning 10×, and converts none of it | C28 |
| head-seed variance is 25× generation-seed variance | it is comparable — ratios 0.63–1.71, all intervals containing 1 | C29; C25's figure retracted |
| calibration is a λ rescale, and therefore hurts | the rescale is an identity and general; "therefore hurts" needs α < 1 | C24, where α = 1.6154 |

**The framing this supports, stated plainly.** This is a *crossing-point* paper with a
positive result at the cheap end and a hard ceiling above it, plus two measurement-practice
findings that apply to the whole comparison literature. It is not the negative-result paper
v3 and v4 described. The simulated NeurIPS panel could not have recommended this framing —
it reviewed on 2026-08-01, and the k sweep that produces the crossing did not exist.

---

Status of what follows: **v2, revised after simulated NeurIPS-workshop review.**

The review scored v1 **4/10 (borderline)** at confidence 4/5, and the reason was framing,
not evidence: *"not whether the work was done carefully (it was), but whether the
rhetorical framing matches the evidence."* Two of four claims were stated more generally
than one λ and two properties can support. v2 below narrows every claim to what was
actually tested. The reviewer's projection was that scoping plus a λ sweep moves this to
a clear 5–6. Full review verdict in §Review below.

---

> **v3 is below, written after phase 2.** The phase-2 data does **not** support the
> lexical-locality thesis in the form it was pre-registered, so v3 is not the mechanistic
> paper `docs/TODO.md` D1 hoped for. It is something else that phase 2 did establish and
> that neither v1 nor v2 could claim: the negative result is now **located**. There is a
> large one-step lever at every position for every property tested, λ=1 throttles about
> half of it, and the head collects only 12–22% of what remains. v2 is kept below because
> the review it responds to is part of the record.

> **v4 is below, written 2026-07-31 after C18 (§20) and C17 (§21).** Those two experiments
> do not change the negative result; they change what can be said about *why*, by closing
> three of the cheapest explanations for it and overturning one of the report's own claims.
> v3 rested the paper on "the negative result is located". v4 can rest it on something
> stronger and more portable: **the objective a future-property probe is trained and scored
> on is not the objective the decoder consumes, and the two come apart measurably in three
> independent ways.** v3 is kept below because it is the version the λ sweep supports on
> its own.

## Title

**v4 (recommended, post-C17/C18):** Better Probes Do Not Decode Better: Calibration, Capacity
and Depth All Improve a Future-Property Head Without Improving Guided Molecular Generation

Alternatives: *Rank, Level, Spread: Why a Well-Calibrated Future-Property Probe Is the Wrong
Probe for Guided Decoding*; or, keeping the ceiling result in the title, *A Lever Everywhere
and No Way to Pull It*.

**v3 (post-phase-2):** Located, Not Just Observed: A Head-Free Ceiling Shows
Why Future-Property Reranking Loses to Compute-Matched Sampling in a Frozen Molecular
Language Model

Alternatives: *The Lever Exists: Measuring the Ceiling on Token-Choice Property Control in
a Frozen Chemical Language Model*; or, if the negative result is to lead, *Guidance Is Not
Signal-Limited: Reranking Captures a Tenth of the Available One-Step Headroom*.

**v2 (scoped):** Predictable Is Not Steerable: Future-Property Reranking in a Frozen
Molecular Language Model Is Dominated by Compute-Matched Sampling for Cheaply-Evaluable
Properties

**v1 (rejected as overclaiming):** ~~Predictable Is Not Steerable: Future-Property
Guidance in a Frozen Molecular Language Model Loses to Compute-Matched Sampling~~ —
"Guidance" reads as a claim about steering methods as a class; we tested next-token
reranking over 8 candidates. "Loses to compute-matched sampling", unqualified, is not
supportable from one λ on two cheap-oracle properties.

Alternatives if the scoped title is too long: *Two Properties That Dissociate: Knowing
vs. Controlling in Guided Molecular Decoding*.

---

## Abstract v4 (≈340 words) — recommended, post-C17/C18

Inference-time steering of an autoregressive generator is usually built the same way: train
a probe to predict a future property from a prefix, then rerank the base model's candidate
tokens by it. We ask, on a single frozen released chemical language model (GP-MoLFormer-Uniq,
46.8M parameters, no weight updates), whether improving that probe improves the generation —
and find that it does not, in three independent ways, for a reason that is structural rather
than incidental.

We first measure what any decoding rule could achieve. At each of 400 held-out prefixes we
extend by all eight candidate tokens and complete each extension 16 times under the base
policy (51,200 continuations, no probe involved). A lever exists everywhere: choosing the
best available candidate would double to triple the probability of landing in the target
band for all six properties. FUDGE-style reranking at λ=1 captures 4.8–10.9% of it, and
loses to compute-matched best-of-N sampling for every property under both token accountings.

We then try to close that gap by making the probe better, three ways, each pre-registered
with its decision rule fixed before the run. **Calibration:** post-hoc calibration cuts
expected calibration error 3–6x and leaves target AUROC bit-identical, because monotone maps
preserve rank; and it makes generation *worse* — 0.23–0.70x of the deployed lift at every
property — because a power calibration is algebraically a rescale of the guidance strength
λ, with fitted exponents 0.40–0.62. At ε=0 a calibrated probe at λ=1 and the raw probe at
λ=α return the identical 1,536 molecules. **Capacity:** a 6.9x-wider readout moves AUROC by
at most +0.002. **Depth:** sweeping all 13 probe points shows every property is predicted
best in the middle of the network, not at the final layer — overturning our own earlier
claim that this model does not represent aromatic ring count — yet the best-predicting layer
is not the best-steering layer for any of the six properties.

The unifying account is that the probe is scored on discrimination, a rank statistic, while
the decoder consumes `log q` inside a softmax over candidates, which is a function of
*spacings*. Calibration moves levels, which the softmax discards; depth moves ranks, which do
not determine spacings. We also report that a per-position improvement of 1.78x became 1.00x
end to end, and 1.22x became a 7% loss.

## Abstract v3 (≈330 words) — post-phase-2, superseded by v4

Autoregressive chemical language models write molecules token by token, and inference-time
steering work usually conflates two questions: how much does an unfinished SMILES prefix
already determine the finished molecule's properties, and can that information be used to
control generation? We separate them on a single frozen released generator
(GP-MoLFormer-Uniq, 46.8M parameters, no weight updates), with a future-property head over
discretized property bins and FUDGE-style reranking of the base model's top-8 candidates.

Reranking shifts hit rate — by +0.295 for aromatic ring count and +0.215 for H-bond donor
count over six properties — and survives direct standardisation on sequence length and
heavy-atom count. It nevertheless loses to best-of-N sampling at compute matched on
processed generator tokens, for every property and under both accountings. That much
replicates a known pattern.

The contribution is that we can now say **why**, because we measure the ceiling. At each of
400 held-out prefixes we extend by all eight candidate tokens and complete each extension 16
times under the base policy: 51,200 continuations, no head and no guidance strength involved.
The spread across candidates bounds what any decoding rule could achieve at that position.
**A lever exists everywhere**: choosing the best available candidate would double to triple
the probability of landing in the target band (0.09–0.15 → 0.22–0.40) for all six
properties. Reranking at λ=1 captures **4.8–10.9%** of it. Substituting an oracle head into
the same λ=1 softmax, with a permutation null for the oracle's self-exploitation, splits the
loss: **λ=1 permits only 32–53% of the ceiling, and our head collects 12–22% of that.** Those
are per-position figures, so we test the λ term end to end rather than extrapolating it: over
six values of λ on three anchor properties, tuning λ is worth **1.29–1.69x**, the response is
an inverted U with an optimum at λ = 2–4 beyond which validity collapses to 0.81–0.90, and
**no λ beats compute-matched best-of-N** — the closest is a gap of 0.093. The failure is
therefore neither a shortage of signal nor the guidance strength.

We pre-registered a lexical-locality hypothesis to explain which properties are steerable.
It fails in the units it committed to (rank correlation −0.886 against the predicted
ordering, because normalising by target-band width largely measures inverse width), though
its sharpest discriminating case succeeds: H-bond donor count, reported hardest to steer by
additive latent vectors, is among the easiest by token choice. We report the falsification.

Chemical quality is unharmed up to λ=2 and breaks above it: at λ=8 the degeneracy rate among
target-hitting molecules rises up to eightfold and synthetic accessibility worsens by 0.39,
with the failure mode being fragmentation rather than the long greasy tails the
molecular-optimisation literature reports — a consequence of steering into a bounded band
rather than maximising.

We also report two defects found in our own previously published pilot code, one of which
means its cLogP head was scoring a strict subset of the target interval, and both of which
are now enforced by tests.

## Abstract v2 (≈270 words) — superseded by v3, kept because the review below responds to it

Autoregressive chemical language models write molecules token by token, which raises two
questions that inference-time steering work usually conflates: how much does an
unfinished SMILES prefix already determine the completed molecule's properties, and can
that information be used to control generation? We separate them on a single frozen,
released generator (GP-MoLFormer-Uniq, 46.8M parameters, no weight updates), using a
future-property head that predicts a discretized distribution over the *completed*
molecule's property from a partial prefix, and FUDGE-style decoding that re-ranks the
base model's top-8 candidates by that head at guidance strength λ=1.

Prediction works and improves with prefix completeness: for cLogP, target-interval AUROC
against a bank of 32 independent continuations per prefix rises from 0.663 to 0.883
across position quartiles. Reranking also shifts hit rate, by up to +0.13, and survives
direct standardisation on both sequence length and heavy-atom count (92–97% of the effect
retained at 0.98–1.00 coverage), so it is not a size artefact. Chemical quality is
unharmed — guided hits match base-policy hits on synthetic accessibility and exceed them
on drug-likeness — except under late intervention, the one condition where quality
measurably degrades.

Two findings undercut the method. First, our two properties dissociate in *opposite*
directions: cLogP is most predictable late but least steerable late (+0.015, within seed
noise), while aromatic ring count is predicted *worse* than a token-counting baseline yet
is the more steerable of the two. Second, matching compute on processed generator tokens
rather than sample or forward-call counts, reranking loses to best-of-N by 0.33–0.35 hit
rate (0.53–0.74 under full-recomputation accounting) — with the important caveat that
best-of-N selects using the true RDKit oracle while reranking sees only a learned head,
so this bounds the claim to cheaply-evaluable properties.

---

## Abstract v1 (superseded, kept for the record)

Autoregressive chemical language models write molecules token by token, which raises two
questions that are usually conflated: how much does an unfinished SMILES prefix already
determine the completed molecule's properties, and can that information be used to
control generation? We separate them on a single frozen, released generator
(GP-MoLFormer-Uniq, 46.8M parameters, no weight updates), using a small future-property
head that predicts a discretized distribution over the *completed* molecule's property
from a partial prefix, and FUDGE-style decoding that re-ranks the base model's top-8
candidates by that head.

Prediction works and improves with prefix completeness: for cLogP, target-interval AUROC
against a bank of 32 independent continuations per prefix rises from 0.663 to 0.883
across position quartiles. Guidance also works, shifting hit rate by up to +0.13, and
survives direct standardisation on both sequence length and heavy-atom count (92–97% of
the effect retained at 0.98–1.00 coverage), so it is not a size artefact.

Two findings undercut the method. First, predictability and controllability dissociate in
*both* directions: cLogP is most predictable late but least steerable late (+0.015,
within seed noise), while aromatic ring count is predicted *worse* than a token-counting
baseline yet is the most steerable property we tested. Second, when compute is matched on
processed generator tokens rather than forward calls or returned molecules, guidance
loses to best-of-N sampling by 0.33–0.35 hit rate (0.53–0.74 under full-recomputation
accounting). Guidance does not degrade chemical quality except under late intervention.
We release all artifacts, pre-registered rejection criteria, and both compute accountings.

---

## Claim inventory, with the evidence and the strongest objection to each

> **Numbering, renamed 2026-08-03.** These claims were `C1`–`C37` until now, which
> collided head-on with the **experiment register** in `docs/TODO.md`, also `C1`–`C33`,
> and the collision had become unreadable: `C33` meant both *"every property is predicted
> best mid-network"* (claim) and *"does the oracle asymmetry replicate on a second
> generator?"* (experiment), and `C30`/`C31`/`C32` each meant two different things inside
> a single paragraph of `pilot_report.md` §22.11.
>
> The claims moved because the experiment IDs cannot: they name `outputs/c33_*/`,
> `scripts/27_c33_*.py`, `reports/section_c33_*.md`, `tests/test_*.py` and the frozen
> pre-registrations, none of which may be rewritten after the fact.
>
> **From here on: `CL<n>` is a claim in this inventory; a bare `C<n>` is an experiment in
> `docs/TODO.md`.** Nothing else changed — `CL4` is the claim that was `C4`.

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL1 | A future-property head predicts completed-molecule cLogP from a partial prefix, improving with completeness | AUROC 0.663→0.883 by quartile, scored against 32-rollout empirical conditional distributions, grouped splits | Unsurprising; length alone predicts a lot. Answered by the `trivial` baseline head, but the *margin* over trivial is the number that matters and is modest |
| CL2 | Guided decoding shifts the property distribution and it is not a length/size artefact | +0.115/+0.131 (rings early/middle); 92–97% survives joint standardisation at 0.98–1.00 coverage | Only tested at λ=1; a critic can say the effect size is small relative to what tuning would give |
| CL3 | ~~**Double dissociation**: cLogP predictable-late/unsteerable-late; rings unpredictable/steerable~~ **REVISED after phase 2 — the timing half is withdrawn.** What survives: rings are predicted *worse* than token counting yet are the most steerable of six properties, while cLogP is predicted well and is not. That is still a predictability/controllability dissociation; it is no longer a claim about *when* | Predictability half replicated on an independent 50k sample (§13.1). Timing half **did not replicate**: cLogP's late lift went from +0.0153 (10% of throughout) to +0.0481 (24%), overtaking aromatic rings' 19%, and a same-sample control attributes 0.026 of the 0.033 change to the sample rather than to the interval-mask defect (§16.3, §17.4) | The withdrawn claim was ours and was prominent. Its own reported seed spread (±0.022) was already as large as the effect, which should have been read as a warning at the time |
| CL4 | Guidance loses to best-of-N at compute matched on **processed tokens** | −0.3488/−0.3312 (`actual`), −0.7355/−0.5291 (`full_recompute`) | RDKit oracles are free, which is maximally unfavourable to guidance. We say so, but it limits the claim's reach |
| CL5 | Token-matched accounting is the right accounting, and sample/forward-call matching flatters guidance | Both accountings computed and reported; the gap between them (0.35 vs 0.74) is itself the argument | Methodological, not empirical — reviewers may see it as a framing point rather than a result |
| CL6 | The head is badly miscalibrated on the off-policy prefixes guidance visits | predicted 0.076 vs observed 0.267, ECE 0.190; one DAgger round gives +0.0305 | This weakens CL4: "guidance fails" may be "guidance fails *with a broken signal*" |
| CL7 | Steering to a bounded interval does not degrade chemical quality; only late intervention does | SA/QED/chain/fragment panel, guided hits vs base-policy hits, bootstrap CIs; late rings SA +0.143 [+0.055, +0.236] | Bounded intervals are the easy case. Says nothing about maximisation, which is what practitioners actually do |

## Phase-2 claims (added 2026-07-30)

Phase 2 tested the lexical-locality hypothesis on six properties and, in the course of
setting it up, found two defects in phase-1 code. The methodological claims are listed
here separately from the scientific ones because they are independently checkable and do
not depend on how the locality thesis came out.

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL8 | The pilot's cLogP head was trained to predict a **0.050-mass event for a 0.100-base-rate target**, because `interval_mask` keeps only bins wholly inside `[lo, hi)` and the interval (a full-sample quantile) and the binner (fitted on the train split) never exactly agreed | Read directly off the committed checkpoints: 1 of 20 bins selected. Reproduced and quantified by re-training with and without the fix: mask covers 0.0465 of a 0.0929 target; mean predicted target probability 0.0494 against base rate 0.0929, versus 0.0943 when fixed; ECE 0.0437 → 0.0045 | It is our own bug. The mitigation is that it is now enforced by a numerical check that exits non-zero, plus 11 regression tests, and every affected number is restated |
| CL9 | **The pilot's reported cLogP "calibration failure" was that defect, not a miscalibrated head.** Fixed, the head is essentially perfectly calibrated on-policy (predicted/base-rate = 1.014) | §11.6 | Weakens our own §8.2 and §9.2.1. The off-policy shift is real but smaller than reported |
| CL10 | Discrimination was **not** materially affected: cLogP target AUROC moves by +0.0004 | §11.6 | None; this is the reason CL8 does not invalidate section 5 |
| CL11 | The git-excluded arrays are reproducible from the pinned revision and seeds **only on hardware whose RNG stream matches**. CUDA draws a different 50k sample at the same seed; the numerics agree to 1.4e-05 and the top-8 candidate set is identical | `outputs/device_equivalence/`: 0 of 64 molecules identical across devices, bit-identical within device; all base-distribution means agree at \|z\| ≤ 1.0 | A reproducibility caveat rather than a result. But it is a caveat this repository asserted the opposite of, and it makes a *frozen* interval unrecoverable on new hardware |
| CL12 | The aromatic-ring crossover **replicates on an independent 50k sample**, with seeded head initialisation and three seeds: frozen state 0.7878 ± 0.0023 against trivial 0.8269 ± 0.0019 target AUROC | §13.1, and again on the new rollout bank (0.8407 against 0.8889 Spearman) | None known. This is the pilot's most robust claim getting more robust |
| CL13 | Head-initialisation variance is **an order of magnitude smaller** than the effects being compared (AUROC sd ≤ 0.0041 across three seeds) | §13.2 | Three seeds is few; the sd of an sd at n = 3 is large. It bounds the effect loosely, which is all that is needed |

CL8–CL10 mean two numbers in the phase-1 report were **wrong in our favour** in one respect
and against us in another: the head looked worse-calibrated than it was, and the guidance
signal was aimed at a narrower band than intended. Both are now stated in the report.

### The scientific phase-2 claims

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL14 | **There is a large one-step lever at every position for every one of six properties.** Choosing the best of the eight candidates the base model already proposes roughly doubles to triples the probability of landing in the target band (0.09–0.15 → 0.22–0.40) | 400 prefixes x 8 candidates x 16 base-policy rollouts, 51,200 continuations; head-free and λ-free, with a permutation null for the finite-sample bias in `max − min`. §15.1 | The ceiling is an *unweighted* max over eight candidates, and the base policy puts a mean 0.738 of its top-8 mass on one token (on the prefix set the capture figures use), so attaining it means overriding `log p_base` hard. It is a ceiling for any rule allowed to choose among the eight, which is the honest reading, but it is not a ceiling reachable at λ=1 |
| CL15 | **Reranking at λ=1 captures only 4.8–10.9% of that lever**, and the range across six properties is narrow — a factor of 2.3 | §15.1, on the 267 of 400 prefixes where all eight candidates cleared the rollout threshold | Confounded between "bad head" and "λ=1 too weak"; CL16 separates them. Two disclosures the first draft omitted: the denominator is noise-corrected, and under the **literal pre-registered formula** capture is **2.1–5.8%** (the correction moves it *up*, so it is conservative for CL20); and this is a **per-position** figure, not a share of what is achievable end to end — see C21 |
| CL16 | The pilot's negative result is therefore **not** explained by "there is nothing to steer". The "no lever" branch of the pilot's own dichotomy is refuted for every property | §15.1–15.3, including a check that the ceiling is not a survivorship artefact (the ceiling-setting candidate has validity 0.983–0.990, and counting invalid completions as misses moves the ceiling by 0.0014–0.0046, under 1.5% of it) | The pilot posed a dichotomy that was missing a third option. This claim is about refuting one branch, not about establishing the second. Headroom is also measured at *base-policy* prefixes while guidance visits its own, where CL6 says the head is worse — so the reported capture is if anything optimistic |
| CL17 | **P1 is falsified in its pre-registered units** (Spearman rho = −0.886 against the predicted ordering) and the mechanism is that normalising headroom by target-interval width largely measures 1/width (rho = +0.698). In band-width-free probability units rho is +0.371, right sign and far from significance | §15.4 | The pre-registration chose the normalisation, and we are reporting that the normalisation was a poor choice. That is a legitimate finding, but it means the *headline* pre-registered test failed and the alternative unit was not the pre-committed one. Both are reported; neither is quietly dropped |
| CL18 | **P2 is falsified, and in the opposite direction from the prediction.** The local counts are flat with position, as predicted; the diffuse properties **rise** rather than decline, cLogP and QED most of all. In probability units all six rise by a factor of 3–10 from Q1 to Q4 | §15.5 | None known |
| CL19 | **The phase-1 explanation for cLogP's null late-window result was wrong.** Phase 1 argued no remaining token choice could move cLogP by an interval width; measured, cLogP's relative headroom exceeds one full interval width at every quartile from Q2 onward and is larger at Q4 than at Q1 | §15.5 | The phase-1 *measurement* stands (+0.015 late); only its explanation falls. The reconciliation is that the large levers sit on low-probability candidates λ=1 will not select |

CL14–CL16 are the phase-2 contribution that does not depend on the locality thesis at all,
and they are stronger than it. They convert the pilot's "guidance loses to best-of-N" from
an unexplained negative into a *located* one: the signal is there, at every position, for
every property, and the deployed rule reaches under a ninth of it.

### The final phase-2 claims

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL20 | **λ=1 is not the binding constraint; the head is.** An oracle head at λ=1 reaches 32.6–53.2% of the head-free ceiling, and ours reaches 11.9–21.5% of *that* | §15.6, with a permutation null for the in-sample oracle's self-exploitation (the null is 26–57% of the raw gain; skipping it gives a nonsensical >100% for one property) | The oracle is in-sample even after correction, so 33–53% is an upper bound on what λ=1 permits. That direction *strengthens* "λ is a real constraint" and weakens "the head is the whole story" — stated in the report |
| CL21 | **This reverses the follow-up ranking — but the λ half is now measured rather than extrapolated.** Per position, λ=1 permits 32–53% of the ceiling and the head collects 12–22% of that. End to end, **tuning λ is worth 1.29–1.69x** and no λ beats best-of-N. `docs/HANDOFF.md` E1 and the simulated reviewer both named the λ sweep the highest-leverage experiment; it is not | §15.6 consequence 3, **superseded for the λ term by §19** | The first draft converted the per-position ratios into end-to-end multipliers, which is invalid: end-to-end lift is 20–48x per-step gain, and linear transfer implies lifts above the arithmetic maximum for four of six properties. Withdrawn and replaced by the sweep. The **head** term is still measured only per position, so the head-versus-λ ranking rests on the λ term being small and capped by base-policy destruction — a mechanism a better head need not share — rather than on a like-for-like comparison |
| CL22 | **Reranking loses to compute-matched best-of-N for all six properties under both accountings** (−0.22 to −0.36 `actual`, −0.53 to −0.81 `full_recompute`), extending the pilot's two-property result | §16.2 | Unchanged from CL4: RDKit oracles are free, so this bounds the claim to cheaply-evaluable properties |
| CL23 | **The size of that gap is set by the target's base rate, not by locality** (rho = −0.771 against base rate; −0.086 against the locality score) | §16.2, §17.2 | This kills P6, which was the prediction that would have unified the pilot's two findings into one variable. Reported as a failure with the actual predictor named |
| CL24 | **P1, P2, P3, P4 and P6 are all falsified as pre-registered.** The hypothesis they encode is nevertheless supported by the quantity with the *fewest* post-hoc degrees of freedom: the chemistry-derived ordering pinned in code before measurement correlates with steerability at **rho = +0.771** (p = 0.072, n = 6), and at **rho = +0.800** (n = 4) when the two properties phase 1 had already measured are dropped | §17.1–17.3 | The width-normalised locality score we invented correlates at **−0.771**, and the band-width-free version at **+0.714** — but the unit was chosen after seeing the data, so P1b is not evidence of the same grade. And "fewest" is not "none": aromatic rings at predicted rank 1 and cLogP at rank 5 restate phase-1 results, which is why the n = 4 leave-phase-1-out check is quoted alongside. Six hand-picked properties; p = 0.07; this is suggestive, not established, and the report says so |

### The λ sweep, which closes the largest outstanding objection

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL25 | **The response to λ is an inverted U, not a saturation.** Every anchor property has an interior optimum — λ=2 for both counts, λ=4 for QED — and every one is worse at λ=8 than at λ=2. Tuning λ is worth **1.29x / 1.61x / 1.69x** on the lift | §19.1. Three anchors x six λ x three seeds x 512 molecules; λ=1 is the central test's own run, `unguided` regenerated at every λ and identical to 10 decimal places | Three properties, not six. The anchor rule (most steerable, least steerable, the pre-registered discriminating case) was fixed in code before the sweep ran, but three points cannot rule out a property whose optimum lies elsewhere |
| CL26 | **No λ beats compute-matched best-of-N**, so **P5 is not falsified** and CL4 now holds across six properties, two token accountings and six values of λ. The closest guidance gets is HBD count at λ=2, gap **−0.0931** against −0.2247 at λ=1 | §19.2 | The "reranking was simply not tuned" objection (limitation 1 below) is answered rather than deflected: it was not tuned, tuning is worth 1.3–1.7x, and it still loses |
| CL27 | **The mechanism capping λ is base-policy destruction, not diminishing returns.** Validity falls 0.995–0.998 → 0.807–0.902 from λ=1 to λ=8, molecules shorten by 3–6 tokens, and HBD count's uniqueness drops from 1.000 to 0.900 | §19.1 | A hit rate computed over molecules a tenth of which no longer parse is not a better controller, which is the point — but it also means the λ=8 hit rates are not directly comparable to the others |
| CL28 | **C12's prediction fires: the degenerate molecules appear, above the optimum.** R3 does not fire at λ ≤ 2 and does at λ ≥ 4. QED's degeneracy rate among hits goes 0.0103 → 0.1217 and its SA score worsens by **+0.389**, five times the largest degradation anywhere in phase 1 | §19.3, guided hits against base-policy hits with 95% bootstrap CIs | This bounds CL7 rather than overturning it: "steering to a bounded interval does not degrade quality" is now known to be a statement about λ ≤ 2 |
| CL29 | **The failure mode is fragmentation, not long greasy tails** — the opposite of the literature's canonical reward-hacked molecule. Longest chain *falls* significantly at λ=8 for all three anchors while fragment count rises | §19.3 | Follows from steering into a **bounded** interval rather than maximising: a bounded objective rewards whatever cheap edit lands inside the band, and fragmenting is cheaper than growing. Untested against a maximisation objective, which is what practitioners actually use |

The one property whose hit-rate optimum is *already* in the degenerate regime is **QED**
(best λ=4, degeneracy difference +0.0420 with the CI excluding zero). For the two counts the
optimum is still clean and the damage starts one step above it. So the trade-off is real and
binding for at least one property, not merely theoretical.

**The honest one-line summary of phase 2:** the measurement built to test the hypothesis
turned out to be worth more than the hypothesis, and it says the method fails for a
locatable reason that is neither "no signal" nor "wrong λ" — the second half of which the
λ sweep has now confirmed by measurement rather than by extrapolation.

### C18 and C17, which close the three cheapest explanations for the negative result

Added 2026-07-31. These are the claims v4's abstract rests on. All are pre-registered with
the decision rule fixed before the run; `pilot_report.md` §20 and §21.

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| CL30 | **Post-hoc calibration of the head is, for the two commonest families, algebraically a rescale of λ.** A power map `g(q)=c·q^α` gives `λ·log(c·q^α) = (λα)·log q + λ·log c`, and the softmax over candidates annihilates the constant. Platt is that family to first order for small `q` | §20.1, §20.3.1. Not just argued: at ε=0 the calibrated head at λ=1 and the raw head at λ=α return the **same 1,536 molecules**, hit rates equal to 0.0 | It is an identity, so the only objection is scope: it holds for maps that are functions of `q` alone. Bin-logit temperature is not, and is tested separately (`pilot_report.md` §20.3.3) |
| CL31 | **Calibration works and is useless — and worse than useless.** ECE falls 3–6x; target AUROC is **bit-identical** under Platt for all six properties, because monotone maps preserve rank. Every fitted Platt slope is < 1 (0.405–0.618), so the correction is a λ *decrease*, and end to end Platt costs 0.23–0.54x and isotonic 0.41–0.70x of the deployed lift at every anchor | §20.3, §20.5 | The calibrators are fitted on prefixes generated by the *uncalibrated* head — a one-step fixed point, not iterated. Both biases run in favour of calibration and it still loses, so the direction of the conclusion is robust |
| CL32 | **A 6.9x-wider readout buys nothing (≤ +0.0019 AUROC, median −0.0007), and a readout focused on exactly the three-bin event guidance consumes is worse on all six properties.** The best end-to-end gain from any retrained readout is 1.09x at one anchor, against 1.00x and 0.76x at the other two | §20.4, §20.5 | One head-training seed per variant against the baseline's three; margins under ~0.008 should not be read from that table. The `wide` differences are inside the seed band and `focused`'s are outside it, so both readings are safe |
| CL33 | **Every property in this model is predicted best in the middle of the network, and the curve is unimodal.** Peak at probe point 3–5 for all six, monotone decline to the final layer, embedding control near chance. Aromatic rings 0.8474 at probe point 3 against 0.7878 at probe point 12 | §21.3. All 13 probe points, 6 properties, 3 head seeds; extraction cost 2,205,784 processed tokens because one forward pass returns every layer | The shape is well documented in language models generally. The contribution is not the phenomenon but its consequence for guidance (CL34) and for our own earlier claim (CL35) |
| CL34 | **The best-predicting layer is not the best-steering layer for any of the six properties**, and swapping to it is NOT MATERIAL under a pre-registered rule: 2 of 6 improve, median relative −0.077, against a bar of ≥4/6 and ≥+0.25. Selecting the layer *post hoc on steering* would have manufactured a positive (20.8% → 29.9% for aromatic rings) — which is why the protocol fixed selection-by-prediction in advance | §21.5, §21.5.2 | It is a **per-position** measurement. It does not prove a mid-network head fails end to end; that needs fresh guided generation, filed as C23 and not yet run |
| CL35 | **We overturn our own most-defended claim, in the direction unfavourable to the paper's original thesis.** §13.1's aromatic-ring crossover is a fact about layer 12, not about the representation: probe point 3 beats the trivial counter by +0.0205 with a Bonferroni-corrected CI of [+0.0142, +0.0284] at α = 0.05/13, better NLL, and both neighbours supporting it. TPSA, the other loser, reverses too | §21.4.1, §21.4.2 | The margin is modest — about five head-seed sd. The honest summary is "the middle of the network is *modestly* better than counting, the end of it is clearly worse", not "the representation obviously contains ring count" |
| CL36 | **A per-position improvement is not an end-to-end improvement, and this is now an observation rather than an arithmetic argument.** 1.78x per position became 1.00x end to end; 1.22x per position became a 7% end-to-end *loss* — a sign reversal | §20.5, point 5 | None known. It is the cleanest demonstration in the project of the defect the audit (`docs/TODO.md` C22.1) removed from an earlier draft |
| CL37 | **The general lesson, which is not specific to molecules.** A probe used inside a softmax over `k` candidates is consumed as ranks and spacings, never levels. Calibration moves levels; depth moves ranks. Neither moves spacings, which is what the decoder responds to. The ECE-selected temperature is above 1 — the wrong direction for decoding — for four of six properties | §20.3.3, §20.6 point 8 | Stated from one model, one guidance rule and six properties. It follows from the form of the score, so it should generalise to any FUDGE-style reranker, but that is an argument rather than a measurement |

**What this does not change.** The negative result is untouched: no arm anywhere beats
compute-matched best-of-N (best advantage −0.2238 against the deployed −0.2247), across six
properties, two accountings, six values of λ, six head-and-calibrator arms and 13 probe
points. What has changed is that three cheap explanations for it are now spent rather than
one, and one of the report's own claims has been narrowed as a result.

## Known prior art that constrains the framing

From the literature scan (see `docs/LITERATURE.md`):

- **Mudgal et al., ICML 2024 (arXiv:2310.17022)** already report that blockwise controlled
  decoding fails to match best-of-K on the harder attributes (helpfulness/harmlessness,
  summarization) while winning on the easiest one (response length). Our CL4 is therefore
  **consistent with, and partly anticipated by, existing evidence** — matched on samples
  rather than processed tokens. We must position against this explicitly rather than
  presenting CL4 as a surprise.
- The general "predictable but not steerable" phenomenon is **not novel**; there is an
  active 2026 cluster on detection-vs-steering dissociation in LLMs. What appears to be
  unclaimed is the *bidirectional* version in a single study (CL3) and its appearance in
  reranking-style rather than activation-steering-style control.

## What CL3 needs before it can carry a paper

A double dissociation resting on two properties is the weakest link. Minimum credible
version: 4–6 properties spanning the predictability and steerability axes, so the
dissociation is a 2×2 with more than one property per cell. Candidates that are cheap to
compute and plausibly span the space: TPSA, H-bond donor count, fraction of sp3 carbons,
rotatable-bond count, largest-ring size, formal charge.

The reviewer went further and named the *alternative explanation* we have not excluded:
ring count is flagged by ring-closure digits — discrete, local, early — while cLogP is a
diffuse compositional function of nearly the whole string. That difference in **lexical
instantiation** could produce the entire dissociation with no general principle involved.
It is a better explanation than ours until more properties are tested, and it should be
stated in the paper as the leading alternative rather than waited for in review.

---

## Review (simulated NeurIPS workshop reviewer, 2026-07-30)

**Score 4/10 — borderline. Confidence 4/5.** Reviewed against ICBINB-BIO / ML4Molecules.

Strengths credited: the predictability-vs-controllability separation is "the correct
question to ask, and it asks it cleanly"; the size-artefact control is "the check that
most papers skip"; two-way compute accounting is "the paper's most legitimate claim to
novelty"; disclosing non-reproducible wall time and the head's own miscalibration is
"exactly the transparency ICBINB-BIO exists to reward."

Weaknesses, ranked by the reviewer:

| # | Objection | Our response |
|---|---|---|
| 1 | CL4 rests on **one λ**; cannot distinguish "fundamentally uncompetitive" from "not tuned" | ~~Accepted. This is experiment E1 and the reviewer's named highest-leverage fix~~ **ANSWERED, `pilot_report.md` §19.** Six λ from 0.25 to 8 on three anchor properties. It was not tuned; tuning is worth 1.29–1.69x; the response is an inverted U with an optimum at λ = 2–4; and it still loses to compute-matched best-of-N by 0.09–0.28 at *every* λ. CL4 is strengthened. The **N** half of the reviewer's request is still outstanding |
| 2 | "Double dissociation" from n=2 is "not epistemically supportable"; the lexical-instantiation alternative is at least as plausible | Accepted. Reframed in v2; alternative now stated explicitly |
| 3 | Cheap RDKit oracles make best-of-N artificially strong, and **this is not visible in the abstract** | Accepted. Report §5 scoped it correctly; the abstract did not. Fixed in v2 |
| 4 | One model, one probe layer, one head seed | Accepted as a scope limitation; E3/E5 |
| 5 | Top-8 local reranking is "one specific, fairly weak instantiation of guidance" — scope the title | Accepted. Title changed to "reranking" |
| 6 | Token-count-as-cost-proxy is asserted, given wall time is unstable | Partly accepted. Tokens are the defensible unit; the honest version is that we cannot make a wall-clock claim at all, and we say so |

Two questions the reviewer asked that we can answer immediately:

- *"Is best-of-N's selection criterion the true RDKit property or the same learned
  proxy?"* — **The true oracle** (`compute_properties` on the completed molecule, see
  `bestofn.py:104`). Disclosed and scoped in `pilot_report.md` §5 but missing from
  abstract v1. Now in v2.
- *"Is +0.015 distinguishable from 0, or is this 'no significant effect' rather than
  'steerability decreases'?"* — It is **not** distinguishable (+0.015 ± 0.022 over three
  seeds). The paper must say "no detectable effect", not "steerability decreases". The
  *decrease relative to early/middle* (+0.053/+0.057) is the defensible statement.

One question we cannot yet answer, and should run: *could a compositional confound
(halogen/heteroatom fraction correlating with both token frequency and cLogP) survive
length-and-size standardisation undetected?* `confound.py` is estimator-generic — adding
heteroatom fraction as a third stratifying covariate is a small change and closes a real
hole.

**Reviewer's single highest-leverage addition, 4 weeks and one RTX GPU:** the λ sweep,
plus an N-sweep for best-of-N, producing an actual compute–accuracy **frontier** rather
than two points. Either it holds across the frontier — converting CL4 from "true at one
arbitrary setting" into a robust negative result — or it reveals a regime where reranking
is competitive, which is "still a valuable, honest correction." The reviewer ranked this
above adding a third property.
