# Draft abstract and claim inventory

Status: **v2, revised after simulated NeurIPS-workshop review.** Not submitted anywhere.

The review scored v1 **4/10 (borderline)** at confidence 4/5, and the reason was framing,
not evidence: *"not whether the work was done carefully (it was), but whether the
rhetorical framing matches the evidence."* Two of four claims were stated more generally
than one λ and two properties can support. v2 below narrows every claim to what was
actually tested. The reviewer's projection was that scoping plus a λ sweep moves this to
a clear 5–6. Full review verdict in §Review below.

---

## Title

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

## Abstract v2 (≈270 words) — recommended

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

| # | Claim | Evidence | Strongest objection |
|---|---|---|---|
| C1 | A future-property head predicts completed-molecule cLogP from a partial prefix, improving with completeness | AUROC 0.663→0.883 by quartile, scored against 32-rollout empirical conditional distributions, grouped splits | Unsurprising; length alone predicts a lot. Answered by the `trivial` baseline head, but the *margin* over trivial is the number that matters and is modest |
| C2 | Guided decoding shifts the property distribution and it is not a length/size artefact | +0.115/+0.131 (rings early/middle); 92–97% survives joint standardisation at 0.98–1.00 coverage | Only tested at λ=1; a critic can say the effect size is small relative to what tuning would give |
| C3 | **Double dissociation**: cLogP predictable-late/unsteerable-late; rings unpredictable/steerable | Rollout-bank AUROC and Spearman curves + intervention response over 3 seeds; crossover replicated at 10k and 50k on three estimators | n = 2 properties. A double dissociation on two data points is a pattern, not a law. This is the objection I cannot currently answer |
| C4 | Guidance loses to best-of-N at compute matched on **processed tokens** | −0.3488/−0.3312 (`actual`), −0.7355/−0.5291 (`full_recompute`) | RDKit oracles are free, which is maximally unfavourable to guidance. We say so, but it limits the claim's reach |
| C5 | Token-matched accounting is the right accounting, and sample/forward-call matching flatters guidance | Both accountings computed and reported; the gap between them (0.35 vs 0.74) is itself the argument | Methodological, not empirical — reviewers may see it as a framing point rather than a result |
| C6 | The head is badly miscalibrated on the off-policy prefixes guidance visits | predicted 0.076 vs observed 0.267, ECE 0.190; one DAgger round gives +0.0305 | This weakens C4: "guidance fails" may be "guidance fails *with a broken signal*" |
| C7 | Steering to a bounded interval does not degrade chemical quality; only late intervention does | SA/QED/chain/fragment panel, guided hits vs base-policy hits, bootstrap CIs; late rings SA +0.143 [+0.055, +0.236] | Bounded intervals are the easy case. Says nothing about maximisation, which is what practitioners actually do |

## Known prior art that constrains the framing

From the literature scan (see `docs/LITERATURE.md`):

- **Mudgal et al., ICML 2024 (arXiv:2310.17022)** already report that blockwise controlled
  decoding fails to match best-of-K on the harder attributes (helpfulness/harmlessness,
  summarization) while winning on the easiest one (response length). Our C4 is therefore
  **consistent with, and partly anticipated by, existing evidence** — matched on samples
  rather than processed tokens. We must position against this explicitly rather than
  presenting C4 as a surprise.
- The general "predictable but not steerable" phenomenon is **not novel**; there is an
  active 2026 cluster on detection-vs-steering dissociation in LLMs. What appears to be
  unclaimed is the *bidirectional* version in a single study (C3) and its appearance in
  reranking-style rather than activation-steering-style control.

## What C3 needs before it can carry a paper

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
| 1 | C4 rests on **one λ**; cannot distinguish "fundamentally uncompetitive" from "not tuned" | Accepted. This is experiment E1 and the reviewer's named highest-leverage fix |
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
than two points. Either it holds across the frontier — converting C4 from "true at one
arbitrary setting" into a robust negative result — or it reveals a regime where reranking
is competitive, which is "still a valuable, honest correction." The reviewer ranked this
above adding a third property.
