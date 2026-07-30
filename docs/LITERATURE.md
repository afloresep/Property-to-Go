# Literature position

Compiled 2026-07-30 from two independent scans. **Verification status matters here** —
see §0 before citing anything from this file.

---

## 0. How much to trust this file

Everything below came from automated web searches, not from reading primary PDFs. The
scans flagged their own extraction quality, and those flags are preserved. Before any
number or quote from this file goes into a submission:

1. **Pull the actual PDF.** Several numbers came through lossy PDF-to-text extraction.
2. **Verify the 2026 arXiv IDs exist.** A cluster of 2026 preprints is cited below
   (IDs of the form `26xx.xxxxx`). These were reported by the scans and have **not been
   independently confirmed by opening them**. Treat them as leads, not citations.
3. **Re-run a fresh arXiv search immediately before submitting.** One paper found in
   this scan (Q-Steer) was posted the day before the scan ran. This subfield is moving
   in weeks, not months.

Items marked **[VERIFIED-CLASSIC]** are well-established papers whose existence and
content are not in doubt. Items marked **[LEAD]** need a manual check.

---

## 1. The papers that must be cited

| Paper | ID | Why |
|---|---|---|
| Yang & Klein, *FUDGE: Controlled Text Generation With Future Discriminators*, NAACL 2021 | arXiv:2104.05218 | **[VERIFIED-CLASSIC]** The method we implement |
| Ross et al., *GP-MoLFormer: A Foundation Model For Molecular Generation*, Digital Discovery 2025 | arXiv:2405.04912 | **[VERIFIED-CLASSIC]** The model we freeze |
| Mudgal et al., *Controlled Decoding from Language Models*, ICML 2024 | arXiv:2310.17022 | **[VERIFIED-CLASSIC]** The closest prior best-of-N comparison. See §3 |
| Gao, Fu, Sun & Coley, *Sample Efficiency Matters (PMO)*, NeurIPS 2022 D&B | arXiv:2206.12411 | **[VERIFIED-CLASSIC]** Excludes logP as a task. See §2 — this is the most important single citation for us |
| Zhou et al., *Optimization of Molecules via Deep RL (MolDQN)*, Sci. Rep. 2019 | arXiv:1810.08678 | **[VERIFIED-CLASSIC]** The canonical "logP optimisation yields non-drug-like molecules" evidence |
| Renz et al., *On Failure Modes of Molecule Generators and Optimizers*, 2020 | J. Cheminformatics | **[VERIFIED-CLASSIC]** The "AddCarbon" dummy generator that beats benchmarks — directly relevant to our trivial-features result |
| Bagal et al., *MolGPT*, JCIM 2021 | 10.1021/acs.jcim.1c00600 | **[VERIFIED-CLASSIC]** The training-time-conditioning contrast |
| Navratil et al., *GP-MoLFormer-Sim*, AAAI 2026 | arXiv:2506.05628 | **[LEAD]** Same base model, training-free test-time steering. The paper we most need to differentiate from |

---

## 2. The citation that most affects our design — and our defence

**PMO (Gao et al., NeurIPS 2022) deliberately excludes logP from its benchmark**, with
this reasoning (as quoted by the scan, to be verified against the PDF):

> "LogP is unbounded and the relationship between LogP values and molecular structures
> is fairly simple: adding carbons monotonically increases the estimated LogP value...
> simply maximizing LogP is not a meaningful goal in drug design. Therefore, we exclude
> LogP in this benchmark."

A reviewer who knows this paper will ask why we use cLogP at all. **The answer is
structural and it is a good one**, but it has to be stated explicitly and early:

- PMO's objection is to **unbounded maximisation** of logP. We do not maximise. We target
  a **bounded interval** ([4.173, 5.038), the 85th–95th percentile band of the base
  model's own distribution) that the model already reaches ~11% of the time.
- Overshooting is penalised exactly like undershooting, so the "add carbons monotonically"
  degeneracy is not a winning strategy under our objective — it is a losing one.
- Our own quality analysis confirms this empirically: guided hits are no harder to
  synthesise than base-policy hits, and are slightly *more* drug-like
  (QED +0.027 to +0.033). See `reports/PLAIN_SUMMARY.md` §2.8.

So PMO is simultaneously the strongest objection to our property choice and, once the
interval framing is made explicit, evidence that our design avoids the failure mode it
warns about. **This should be a paragraph in the paper, not a footnote.**

The corollary is a genuine limitation: because we chose the *interpolative* regime, we
have **no evidence about the extrapolative regime** practitioners actually care about.
Raising λ, or switching to an unbounded objective, is where the PMO/MolDQN degeneracy
should appear. That is experiment E1 in `HANDOFF.md`, and it is a prediction we have not
tested.

---

## 3. Is our headline negative result novel?

**Partly. It is novel *in this domain* and anticipated *in general*.**

Mudgal et al. (ICML 2024) compare blockwise controlled decoding against best-of-K and
find the answer depends on the attribute:

- On a response-**length** task, blockwise CD-Q matches best-of-K(K=50) with only K=6 —
  guidance wins on efficiency.
- On Anthropic helpfulness/harmlessness and on summarization, the paper states plainly
  that **"neither method was able to match best-of-K"** — best-of-K wins outright.

Two things follow. First, our result is **consistent with and partly anticipated by**
existing evidence, and we must present it that way rather than as a surprise. Guidance
losing to best-of-N on a non-trivial attribute is a known outcome. Second, Mudgal et al.
match on **sample count**, not on processed tokens, which is precisely the accounting
choice we argue is too generous to guidance. Our contribution on this axis is the
accounting, not the direction of the result.

**Neither scan found a molecular-generation steering paper that runs a compute-matched
best-of-N or rejection-sampling baseline at all.** Absent in: GP-MoLFormer-Sim, Q-Steer,
Steering Vector Fields, SLIM, PILOT, and the discrete-diffusion classifier-guidance work.
Meanwhile two adjacent literatures treat the baseline as mandatory — general controlled
decoding (Mudgal et al.) and inference-time scaling for diffusion (Feynman–Kac steering,
arXiv:2501.06848, which calls best-of-N "a widely-used test-time selection baseline").

That asymmetry is the defensible claim: **the control is standard elsewhere and missing
here.** It is a claim about the field's methodology, and it is cheap for a reviewer to
check.

Also unfound: a standalone methodological critique arguing that guided decoding is
routinely under-costed by counting forward calls or samples rather than processed tokens.
The cost formulas exist scattered in efficiency papers; the critique as a thesis does not
appear to. **[LEAD — worth one more targeted search before claiming priority.]**

---

## 4. The predictability/controllability dissociation — reframe required

This was our claimed most-novel contribution. **The general phenomenon is not novel.**

The first scan surfaced an active 2026 cluster on detection-versus-steering dissociation
in LLMs — probes that decode an attribute cleanly while steering along the probe
direction fails. Reported titles include *Perfect Detection, Failed Control*
(arXiv:2606.24952), *Detection Without Correction* (arXiv:2604.13068), *Decodable but Not
Corrected by Fixed Residual-Stream Linear Steering* (arXiv:2605.05715), and
*Representation Without Control* (arXiv:2605.25151); the reverse direction is reported in
*Steerable but Not Decodable* (arXiv:2604.02608). **All [LEAD] — none independently
verified.** Given how neatly this cluster fits what we wanted to find, it deserves
particular scepticism until the IDs are opened by hand.

In molecule-land specifically, *Molecules Meet Language* (arXiv:2605.06303) **[LEAD]**
reportedly makes the explicit claim for QED, HBD count and rotatable-bond count: high MLP
R² (decodable) but low linear-probe R² (not linearly steerable), attributed to curved
property manifolds.

What appears to remain unclaimed:

1. **The bidirectional version in one study.** Every dissociation above is
   unidirectional — predictable-but-unsteerable, or steerable-but-unpredictable, not both
   in the same setup. *Perfect Detection, Failed Control* is reported to say explicitly
   that its own result is not a double dissociation.
2. **The mechanism.** All of the above use activation steering or linear probes. We use
   discriminator reranking of top-k candidates, which is a different intervention.
3. **Position-resolved control.** We measure predictability and steerability *as a
   function of position within the sequence*, which is what makes the cLogP
   predictable-late/unsteerable-late statement possible at all.

**But** our double dissociation rests on **two properties**. That is a pattern, not a
law, and it is the objection we cannot currently answer. Fixing it is experiment E4
(4–6 properties, so each cell of the 2×2 has more than one occupant).

---

## 5. The apparent SLIM conflict is a false conflict — resolved 2026-07-30

**Verification status: SLIM exists and is correctly cited.** arXiv:2605.10831, *SLIM:
Sparse Latent Steering for Interpretable and Property-Directed LLM-Based Molecular
Editing*, Zhang, Li, Li, Shen, Xiong & Sun (HKUST-GZ / NUAA / NUDT), submitted
2026-05-11. Author list, affiliations and date all match. **[VERIFIED]**

**The conflict was a misreading, and the misreading was ours.** An earlier draft of this
file reported that SLIM finds continuous properties easier to steer than discrete counts,
citing Spearman ρ = 0.93 for molecular weight against ρ ≈ 0.32 for H-bond donor count,
and treated that as contradicting our result. Reading the paper shows those ρ values are
**not steerability numbers at all**. They come from SLIM's Table 7 feature-interpretability
case study and are defined as "Spearman rank correlation between feature activation and
property value" — how well a *single sparse autoencoder feature's activation tracks a
property across ~50K static ZINC molecules.* That is a monosemanticity/probe statistic.
SLIM's actual steering-success metric is a different quantity entirely, Acc@τ: "the
percentage of test molecules for which at least one generated candidate improves the
target property and has Tanimoto similarity ≥ τ to the input."

Two constructs, both called "Spearman correlation", got merged. Worth remembering as a
failure mode of automated literature scanning: the number was real, the paper was real,
and the claim built on it was still wrong.

**Beyond the metric confusion, SLIM is a different experiment in almost every dimension:**

| | SLIM | Ours |
|---|---|---|
| Intervention | adds one steering vector to the residual stream **at all token positions**, in a sparse-autoencoder basis | chooses among 8 discrete candidate next tokens |
| Task | **editing** an input molecule under a Tanimoto-similarity constraint | free generation from scratch, no similarity constraint |
| Base models | DrugAssist (LLaMA-2-7B), GeLLM3O (LLaMA-3.1-8B / Mistral-7B), MolGen (BART) | GP-MoLFormer-Uniq, 46.8M |
| Properties | QED, DRD2, logP, MW, RotBond, SA, HBA, HBD — **aromatic ring count is not tested** | cLogP, aromatic ring count |
| Best-of-N baseline | **none** (compares against REINVENT4, MolEditRL, and internal ablations) | yes, token-matched |

**There is a weaker, real version of the "counts are harder" claim in SLIM**, and it
should be cited accurately. From an ablation (§4.3): "Random and CAA inflate Acc@0.15 for
continuous properties (logP +30.8, MW +35.4) but decrease it for counting properties
(HBA −15.6, HBD −17.6 with CAA)" — a statement about *weaker baseline steering vectors*.
And SLIM needs a dedicated gradient-alignment loss for counts to work at all: "ℒgrad
specifically aligns the SAE basis with the gradient structure needed for discrete property
control." That is a claim about their feature-extraction pipeline, not a general law about
discrete properties in language models.

**This is better than a contradiction — it is evidence for the lexical-locality
hypothesis.** SLIM's own data supplies the mechanism: a property requiring coordinated,
multi-position structural change is genuinely hard to express as a single additive
direction applied uniformly across a sequence, *especially* under a constraint to stay
similar to an input molecule. The very same property can be trivial to hit by emitting one
ring-closure digit during free generation. The two results are not in tension; they are
two different degrees of freedom, and our account predicts exactly this pattern. See
`LEXICAL_LOCALITY.md` §6.

**Consequence for the property set:** include **HBD count**. It is SLIM's hardest case
under latent-vector editing, and lexical locality predicts it should be *easy* under token
choice. That is a genuine discriminating test rather than a citation skirmish.

---

## 6. Closest prior art, and honest scoop risk

| Paper | Mechanism overlap | Model overlap | Runs best-of-N? | Verdict |
|---|---|---|---|---|
| GP-MoLFormer-Sim (arXiv:2506.05628) **[LEAD]** | Medium — frozen, training-free test-time logit steering, but signal is cosine similarity to a target set plus a GA outer loop | **High — same base model** | No | **Must differentiate explicitly.** Any reviewer who knows GP-MoLFormer will ask |
| Q-Steer (arXiv:2607.26391) **[LEAD]** | **High** — prefix → hidden state → MLP → logit bonus | None (LSTM/GPT-2 ACEGEN) | No | Posted 2026-07-29. Base policy is **not** frozen (co-trained with PPO/REINVENT), reward is a regression target not a Bayesian classifier term. Cite to preempt the question |
| Margin-calibrated Classifier Guidance (arXiv:2605.13101) **[LEAD]** | **High** — literal classifier reranking of a frozen autoregressive chemistry model's beam | None (retrosynthesis) | Not found | Nearest methodological cousin; different task (route feasibility, not final-molecule property) |
| Steering Vector Fields (bioRxiv 2025.09.24.678080) **[LEAD]** | Low-medium — activation steering | Unclear which CLM | No | Cite for the validity-versus-effect-size trade-off |
| SLIM (arXiv:2605.10831) **[LEAD]** | Low — latent steering for editing | None | No | Cite for §5's contradiction |

Neither scan found our exact combination: the literal FUDGE decomposition with a
**discretized property-interval head**, on the frozen released **GP-MoLFormer-Uniq**,
evaluated against a **token-matched best-of-N** baseline, reporting that guidance loses.
The ingredients all exist separately. The combination and the negative headline do not
appear to.

---

## 7. What no one appears to have shown, and we can

From the second scan's synthesis: every molecular steering paper that reports a quality
metric reports *aggregate* degradation (validity down, uniqueness down, SA unstable), but
**none show the SMILES-level picture of what guided decoding actually produces when it
succeeds.** The vivid "long alkane chains and absurd fused rings" evidence comes from the
RL and genetic-optimisation literature, not from the decoding-steering literature.

We have that picture, and it is not the expected one:

- At λ=1 into a bounded interval, guided hits are chemically indistinguishable from
  base-policy hits, and slightly more drug-like.
- The exception is sharp and interpretable: **late** ring guidance costs SA
  +0.143 [+0.055, +0.236] and longest-chain +0.199 [+0.011, +0.412], because adding an
  aromatic ring to an already-committed scaffold requires tacking something awkward on.
- One genuinely broken molecule, quotable:
  `C.C.C.C=CC=c1[n-]c(CC(CC)CCC)c(CC(C)C)c1=CC.[CH2-]C[CH2-]` — RDKit parses it, it has
  three aromatic rings, so it scores as a valid target hit. It is four loose methanes
  beside a charged ring and two carbanions. From the late ring condition.

The last item doubles as a methodological point about the field's validity metric:
RDKit-parseability is the standard validity check, and it admits this.

---

## 8. Actions before any submission

1. Open every **[LEAD]** arXiv ID by hand. Do not cite one that cannot be opened.
2. Read the GP-MoLFormer-Sim camera-ready properly and write the differentiation
   paragraph.
3. Confirm the PMO logP quote against the published PDF; it is load-bearing for §2.
4. Fresh arXiv sweep of cs.LG and q-bio.BM in the week before submitting.
5. Decide whether the SLIM contradiction (§5) goes in the paper. It should, if it is real.
