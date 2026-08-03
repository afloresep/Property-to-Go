# Your Baseline Has the Answer Key

### Two measurement practices decide whether probe-guided decoding beats best-of-N

**Draft v2 — NeurIPS workshop format (ICBINB track), target 4 pages + appendix.**
Status: every number below is machine-derived from a tracked artifact under `outputs/*_summary/`
and bound by a test in `tests/`. C33 has completed and §3 is written from its artifacts;
both figures are drawn by `scripts/28_paper_figures.py` into `outputs/paper_figures/` and
bound by `tests/test_paper_figures.py`.

---

## Abstract

We tried to show that a learned probe reading a frozen molecular language model's hidden
states can steer generation better than simply sampling many molecules and picking the best.
It could not: guided decoding lost to compute-matched best-of-N by 0.25-0.37 in hit rate. We
then audited the comparison itself and found two measurement practices that between them
account for most of that gap and most of its apparent stability. First, the best-of-N
baseline selects its winner using the ground-truth property oracle, which the guided method
never sees; on two independent generators, **the oracle is worth 0.37 to 0.68 hit-rate
points at N = 32** — 0.37 to 0.71 of the baseline's own hit rate there — and it grows with
N. Second,
the *training seed of the probe* — not the generation seed — is the dominant source of
variance, and it moves chemical validity, which propagates through the token budget into the
comparison itself. With both practices corrected we find a real but narrow result: on two
architecturally distinct frozen generators, guided decoding **does** beat oracle-selected
best-of-N, but only in a small-compute regime (2-4 candidates), and the advantage collapses
monotonically as compute grows. A 2×2 factorial further shows the popular "steer from a
mid-network layer" recommendation is largely a guidance-strength effect in disguise. We
report the protocol, not a win.

---

## 1. Introduction

A frozen language model plus a small learned probe is an attractive way to control
generation: you train nothing large, you change no weights, and at each step you reweight the
model's own candidates by how likely the probe thinks each one is to end up satisfying your
constraint. This is FUDGE-style guided decoding, and in molecular design it promises
property-targeted generation from a general-purpose chemical LM.

The obvious baseline is best-of-N: sample N molecules unconditionally, keep the one that best
satisfies the constraint. Both methods spend generator compute; a fair comparison matches
that compute and asks who wins.

**We ran that comparison and lost.** Across three properties and a full 11-point compute
frontier, guided decoding sat below the best-of-N curve essentially everywhere — 1 of 46
measured guided configurations sat above it. The honest thing to do with that result was to
write it up as a negative one.

Before doing so we audited the comparison, and the audit is what this paper is about. Two
things that no one reports turned out to determine the answer:

1. **The baseline had the answer key.** Best-of-N ranks its candidates using the true RDKit
   property of the *finished* molecule. The guided method only ever sees a learned probe
   reading a hidden state mid-generation. The comparison was partly a restatement of *ground
   truth beats an estimate of it*.
2. **The probe's training seed is a first-class experimental factor.** Retraining the probe
   under a different random seed — same architecture, same data, same everything else —
   moves the measured advantage more than changing the generation seed does, and it moves
   chemical validity, which feeds back into the token accounting.

**Contributions.**

- An **equal-information control** for steering-vs-search comparisons: give the baseline the
  same learned selector the steered method uses, and evaluate both with the same oracle. The
  two curves separate in N on both generators, reaching 0.37-0.68 hit-rate points by N = 32
  (§3.2, Figure 1) — and we show that summarising this as a single "oracle share" **fails to
  replicate**, because that ratio depends on where the steered arm sits on the compute axis
  rather than on the model (§3.3).
- Evidence that the **probe seed dominates the generation seed** as a variance source, with a
  concrete mechanism by which it corrupts compute-matched comparisons (§4).
- With both corrections applied, a **narrow positive result** replicated across two frozen
  generators with different attention mechanisms: guidance beats *oracle-selected* best-of-N
  at 2-4 candidates, and loses monotonically thereafter (§5).
- A **2×2 factorial** separating probe depth from guidance strength λ, showing λ dominates
  and the two are additive — the "use a mid-network layer" recommendation is largely a
  λ effect (§6).

Everything we report is measured with the generator **frozen**: no fine-tuning, no adapters,
no reinforcement learning, no activation edits.

---

## 2. Setup

**Generators (both frozen, both pinned by revision).**

| | generator 1 | generator 2 |
|---|---|---|
| model | GP-MoLFormer-Uniq | `entropy/gpt2_zinc_87m` |
| parameters | 46.8M | 87,331,584 |
| attention | linear | full softmax |
| probe points | 13 | 13 (12 blocks) |
| tokenizer / data | own, PubChem-scale | own, ZINC |

The two share no weights, no tokenizer and no training corpus, and differ in attention
mechanism. Replication across them is the strongest generality claim we make.

**Properties.** `aromatic_rings`, `hbd_count`, `qed`. Target intervals and prefix windows were
frozen from held-out data **before any guided result was inspected**, and are never
re-derived.

**Guided decoding.** At each step, over the base model's top-`k` candidate tokens, sample from

  softmax( log p_base(a) + λ · log( q(a) + ε ) )

where `q` is the probe's predicted probability that the finished molecule lands in the target
interval. The probe is a 768→256→256→bins MLP on a frozen hidden state. **`k` is the compute
knob**: guidance costs exactly `(k+1)×` the base model's forward passes, an identity we check
per run.

**Compute matching.** All comparisons are at matched *processed generator tokens*, never wall
clock. Best-of-N is evaluated over all disjoint consecutive groups of N across the whole pool,
on the grid N ∈ {1,2,3,4,6,8,9,12,16,24,32}.

**Statistics.** Generation seeds 101/202/303; probe seeds where stated. We report seed-level
Student *t* intervals with the correct degrees of freedom (t₀.₉₇₅,₂ = 4.302653 at three
seeds; t₀.₉₇₅,₇ = 2.364624 at eight). **We use no bootstrap.** At n = 3 the percentile
bootstrap of a mean is identically [min, max], and P(all three resamples draw the minimum) =
1/27 = 0.037 > 0.025, so the nominal 95% interval does not exist. We withdrew one such
statistic from an earlier version of this work.

---

## 3. Finding 1 — the baseline selects with ground truth

### 3.1 The control, and what it removes

Best-of-N's selection rule evaluates the **true RDKit property of the completed molecule**.
Guided decoding never has that. So "best-of-N dominates at matched compute" is, in part, "an
oracle beats an estimate of it at equal cost" — which is not a finding about steering.

We built an **equal-information comparator**: best-of-N that selects its winner with the
*same probe, same probe point, same target interval, same binning* that guidance steers with,
reading the state at the last content token. Selection differs between arms; **evaluation
does not** — both are scored with the same true RDKit oracle. The probe arm receives no
oracle information at all, including no validity filtering, since RDKit parseability is
itself oracle information.

### 3.2 What replicates: the oracle's value grows with the baseline's budget

The two curves are identical at N = 1 by construction — with one candidate there is nothing
to select — and separate as N grows, because the oracle gets to exploit every extra draw and
the probe does not. **That separation is the quantity that replicates.** Measured
independently on both generators, at matched N:

| N | arom. g1 | arom. g2 | HBD g1 | HBD g2 | QED g1 | QED g2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0347 | 0.0200 | 0.0177 | 0.0186 | 0.0331 | 0.0344 |
| 4 | 0.1278 | 0.0855 | 0.0736 | 0.0775 | 0.1257 | 0.1329 |
| 9 | 0.3024 | 0.2282 | 0.2172 | 0.2289 | 0.3534 | 0.3630 |
| 16 | 0.3792 | 0.3442 | 0.3491 | 0.3591 | 0.5344 | 0.5327 |
| 32 | 0.3724 | 0.3764 | 0.4415 | 0.4750 | 0.6836 | 0.6597 |

Two generators sharing no weights, no tokenizer, no training corpus and no attention
mechanism agree on this to a mean absolute difference of **0.0111** (HBD), **0.0081** (QED)
and **0.0369** (aromatic rings). **By N = 32 the oracle is worth between 0.37 and 0.68
hit-rate points**, which is 0.37 to 0.71 of the baseline's own hit rate at that budget. So
the headline is not a single number: *the more compute you give best-of-N, the more of its
advantage is the answer key rather than the search.* Any comparison that reports one N is
reporting one point on this curve.

> **Figure 1** (`outputs/paper_figures/fig1_oracle_gap_vs_n.png`). The gap above, plotted
> against N: generator 1 solid, generator 2 dashed, one colour per property. The visual
> claim is that **the pairs lie on top of each other while the deployed-arm budget markers
> do not** — the shaded bands mark where each generator's deployed guided arm actually sits
> on this axis (generator 2 at N ≈ 3.6, generator 1 at N ≈ 8-9), which is the whole of §3.3.
> The curve is monotone in N for HBD count and QED on both generators; on aromatic rings,
> generator 1 peaks at N = 24 (0.3839) and dips to 0.3724 at N = 32, so the caption says
> **grows**, not *grows monotonically*.

### 3.3 What does not replicate: the single-number "oracle share"

C27 summarised the above as a share — the fraction of the deployed arm's gap that the oracle
accounted for — and got 0.876 / 0.882 / 0.859 on generator 1. We pre-registered a replication
on generator 2 and it **failed**, on the rule as written:

| anchor | vs oracle-selected | vs equal-information | share, g2 | share, g1 |
|---|---:|---:|---:|---:|
| aromatic rings | -0.1756 | -0.1025 | **0.4162** | 0.8756 |
| HBD count | +0.0317 | +0.0980 | **undefined** | 0.8819 |
| QED | -0.0809 | +0.0355 | **1.4386** | 0.8594 |

The share is undefined for HBD count because that arm's gap is positive — it is already above
the baseline — a case the pre-registration fixed in advance rather than filling in after the
fact. The two computable shares miss C27's band **from opposite sides**.

**Why, and it is not the generator.** The share is `gap(b) / |advantage(b)|`, both evaluated
at whatever budget `b` the guided arm happens to occupy. Generator 2's deployed arms cost
about **131 tokens/molecule**, landing between N = 3 and N = 4, where §3.2's curves have
barely separated. Generator 1's cost **367-419**, landing at N = 8-12, where they have
separated by 0.22-0.35. The two generators were never compared at the same point on the
compute axis. The numerator replicates (§3.2); the denominator is a property of where the arm
sits, not of the model.

**So we report the curve, not the share**, and we state the pre-registered failure rather than
retiring the statistic quietly: a summary that moves by a factor of three when the budget
moves is the wrong summary, and we published it first.

Counting configurations: on generator 1, **1 of 46** guided arms sits above the oracle-selected
curve and **15 of 46** above the equal-information curve. On generator 2, **7 of 30** and
**18 of 30**. The direction — restoring information symmetry moves many arms above the
baseline — replicates cleanly on both.

**Caveat, stated plainly.** For these three properties RDKit *is* free, and no practitioner
would deploy a learned selector in its place. The equal-information curve is a **scientific
control that says what the comparison was measuring** — not a baseline anyone should ship.
The result generalises to constraints where scoring a candidate is expensive (assays, docking,
human judgement), which is where steering is actually proposed.

---

## 4. Finding 2 — the probe's training seed is the dominant variance source

Everything above holds the probe fixed. We retrained it under **eight seeds**, changing
nothing else — same architecture, same data, same probe point, same target interval — and
re-ran the full guided pipeline for each.

**The probe seed moves the result more than the generation seed does.** On the deployed
configuration, the standard deviation across probe seeds is **0.0366** [0.0242, 0.0744],
against a pooled within-probe-seed generation-seed sd of **0.0213** — the factor the field
routinely reports is the smaller one.

**And it moves chemical validity, which is worse.** For one configuration, validity across the
eight probe seeds ran 0.9811, 0.9792, 0.9616, 0.9603, 0.9212, 0.9036, 0.9030, **0.8301**.
That last seed is not merely a worse probe; the failure **chains**. Lower validity means
longer sequences, longer sequences mean more processed tokens, more tokens mean the
configuration is priced against a *stronger* point on the best-of-N curve, and its measured
advantage collapses from +0.1690 to +0.0133. A compute-matched comparison converts a quality
regression into a budget penalty, and the two are not separable after the fact.

This is why our pre-registered verdict on the replication returned **UNINTERPRETABLE**: we
had committed in advance to voiding the analysis if any point fell below 0.90 validity, and
1 of 56 points did. We report that, and report the post-hoc sensitivity analysis that drops
the affected configuration entirely and returns CONFIRMED at unchanged thresholds, as
explicitly post-hoc.

**Recommendation.** Report the probe seed as a factor, with at least three seeds, and report
validity per seed rather than pooled. A single-seed probe result in this literature is a
point estimate with an unreported standard deviation larger than most claimed effects.

---

## 5. Finding 3 — guidance does cross, at small compute, on both generators

With the measurement corrected, the result that survives is narrow and real. Note that
everything in this section is priced against the **oracle-selected** curve — the *harder*
baseline, the one §3 says is unfairly advantaged. The crossing is not an artifact of the
§3 control.

> **Figure 2** (`outputs/paper_figures/fig2_frontiers.png`). Both frontiers in processed
> tokens per molecule, one panel per generator, with every guided k-sweep cell plotted at
> its own measured budget. The crossings are the cells sitting above the solid
> oracle-selected line, and they are all at the cheap end of the axis.

**Generator 1, eight probe seeds** (advantage over compute-matched oracle-selected best-of-N;
*t* interval, 7 df; final column = probe seeds with a positive advantage):

| configuration | property | probe point | λ | k | mean of 8 | 95% CI | +/8 |
|---|---|---:|---:|---:|---:|---|---:|
| A3 | hbd_count | 4 (mid) | 2 | 2 | **+0.2691** | [+0.2348, +0.3035] | 8/8 |
| A3 | hbd_count | 4 (mid) | 2 | 4 | +0.2000 | [+0.1694, +0.2306] | 8/8 |
| A2 | hbd_count | 4 (mid) | 1 | 2 | +0.1326 | [+0.1104, +0.1547] | 8/8 |
| C2 | aromatic_rings | 3 (mid) | 2 | 4 | +0.1052 | [+0.0586, +0.1517] | 8/8 |
| **A1** | **hbd_count** | **12 (deployed)** | **1** | **2** | **+0.1044** | [+0.0895, +0.1194] | **8/8** |
| A2 | hbd_count | 4 (mid) | 1 | 4 | +0.0207 | [-0.0073, +0.0486] | 6/8 |
| C2 | aromatic_rings | 3 (mid) | 2 | 2 | -0.0341 | [-0.0640, -0.0042] | 2/8 |
| A1 | hbd_count | 12 (deployed) | 1 | 4 | -0.0081 | [-0.0277, +0.0114] | 3/8 |

The bolded row is the **deployed** configuration — not a mid-network variant selected after
the fact — and it is positive on 8 of 8 probe seeds. The last row is reported because it was
generated: a configuration that is run and then never mentioned is indistinguishable, to a
reader, from one that was dropped.

**Generator 2** reproduces the shape on a different architecture: 5 of 30 configurations cross
the oracle-selected curve with intervals excluding zero, all of them mid-network at λ = 2 —
e.g. `aromatic_rings` +0.2473 [+0.1960, +0.2986] at k = 4, and `hbd_count`
+0.2295 [+0.1785, +0.2807] at k = 2. Probe depth peaks before the final layer here too
(selected probe points 2, 2 and 6 of 12).

**Two counting rules, stated once so the paper is consistent.** "Crosses" above means the
seed-level interval excludes zero (5 of 30). Counting *point estimates* above the curve — the
convention §3.3 inherits from C27's arm counts — gives **7 of 30**. The two extra cells are
`hbd_count` deployed at k = 2 (+0.0317, [-0.0045, +0.0678]) and `hbd_count` mid at k = 8
(+0.0159, [-0.0778, +0.1098]); both intervals span zero. The first is the same cell whose
share is undefined in §3.3, which is why that table shows a positive advantage for a
configuration this section does not count as a crossing. **We report intervals for claims and
point estimates only for arm counts, and never mix them in one sentence.**

**The knob loses the race it is in.** `k` raises the guided hit rate — up to +0.2347 in raw
terms on generator 2 — but on **all six** generator-2 configurations the *advantage over
best-of-N* falls monotonically from k = 2 to k = 32, by -0.3645, -0.4232, -0.6352, -0.6061,
-0.6591 and -0.6704. Guidance has a compute knob; best-of-N's is simply better. The honest
summary is that **guidance wins where best-of-N has drawn two to four samples, and nowhere
else** — a regime that matters when each sample is expensive, and does not when it is not.

We also flag one weakness against ourselves: on generator 2 the `aromatic_rings` probe barely
beats a trivial predictor on held-out AUROC (0.8687 vs 0.8683, a margin of +0.0003), yet it
produces the largest crossings. Whatever guidance is exploiting there, probe accuracy on the
target is not a sufficient explanation.

---

## 6. Finding 4 — it is λ, not depth

Both generators show larger crossings from mid-network probes, which invites the conclusion
that mid-network representations steer better. That conclusion is confounded.

The probe's output spread `log q` differs systematically by probe point, and **a
multiplicative rescale of `log q` is exactly a rescale of λ**: λ·log(c·qᵅ) = (λα)·log q +
λ·log c, and the softmax annihilates the constant. This is the same identity Dhariwal &
Nichol (2021, §4.4) use for classifier guidance scale. Comparing a mid-network probe to a
final-layer probe at nominal λ = 1 is therefore not a depth comparison.

We ran the **2×2 factorial** — {final, mid} × {λ=1, λ=2} — on generator 2 across five values
of k:

- **λ's main effect exceeds depth's in all 15 cells.**
- **The two are additive.** The interaction interval excludes zero in 1 of 15 cells, and in
  **0 of the 6 primary cells** (k ∈ {2,4}).
- After correcting for the effective-λ confound (measured spread ratios 1.2417 for
  `hbd_count`, 1.1939 for `aromatic_rings`, **0.8594** for `qed` — note the third is *below*
  1), depth survives on `aromatic_rings`, marginally on `qed`, and **goes negative** on
  `hbd_count`.
- The residual confound share is a median 0.2823 here, against 54-69% for the uncorrected
  single-generator comparison.

One consequence contradicts an earlier version of this work, and we state it rather than
quietly dropping it: `hbd_count` at the **deployed final probe point** crosses the baseline
once λ = 2 — +0.1768 [+0.1304, +0.2232] at k = 2 — so mid-network depth is not required for
a crossing. This holds on one property of three (`aromatic_rings` +0.0358 spans zero; `qed`
is negative), which is why we scope it rather than lead with it.

**Practical reading: turn λ up before you move the probe.** λ is a scalar with no
re-training cost; changing probe depth requires training and validating a new probe, and buys
less.

---

## 7. Discussion

**What we would tell someone comparing steering to search.** Three checks, in order of how
much they moved our numbers:

1. **Does your baseline see something your method does not?** If best-of-N ranks with a
   ground-truth scorer, report the equal-information control alongside — and report it as a
   curve in N, not a single number. Ours reached 0.37-0.68 hit-rate points by N = 32 on
   both generators.
2. **How many probe seeds?** If one, your error bar is missing its largest term. Report
   validity per seed, because validity leaks into the token budget.
3. **Where on the compute axis are you?** A crossing at k = 2 that vanishes by k = 32 is a
   different claim from a crossing. Report the whole frontier. This is also what broke our
   own summary statistic (§3.3): a ratio normalised at one budget is not portable to a model
   whose arms sit at a different one.

**Limitations.** Three properties, one domain, two generators. The compute grid stops at
N = 32 and k = 32. Two λ levels and two depth levels locate no optimum — we can say λ
dominates, not what λ should be. The generator-2 replication gate closes to 5.722e-06 rather
than bit-identically (2.13e-07 of state scale). The 2×2 of §6 uses a **single probe seed**,
which §4 argues is exactly the wrong number; on `qed` neither main effect resolves against
the measured probe-seed noise floor, and `qed` is load-bearing for the "2 of 3 properties"
threshold. Repeating the 2×2 at 3-8 probe seeds is the single most valuable remaining
experiment, and we do not claim §6 at the strength §4 and §5 are claimed. The two generators'
guided arms occupy different regions of the token axis (about 131 tokens/molecule against
367-419), which is what breaks the §3.3 share; comparing them at a matched budget would need
a generator-1 arm roughly three times cheaper than any we ran, so §3.2's matched-N curve
comparison is the closest available control and not a substitute for one.

**What this is.** We did not get the paper we set out to write. Guided decoding is not a
general replacement for sampling and filtering, and we do not report it as one. What the
work produced instead is a measurement protocol for a comparison the field runs constantly
and, as far as we can tell, runs with an oracle-advantaged baseline and a one-seed probe. The
narrow positive result in §5 is real, replicates across two architectures, and is small.
Both of those facts belong in the abstract.

---

## Appendix pointers (not counted in the page limit)

| topic | where |
|---|---|
| full pre-registrations, scored verbatim including failures | `outputs/c2*_prereg/`, `outputs/c3*_prereg/` |
| all replication gates and residuals | `outputs/*_summary/*.json`, key `validity_gates` |
| complete 46-arm and 30-cell tables | §22, §23, §24 of `reports/pilot_report.md` |
| the failed share replication, scored rule by rule | §25 of `reports/pilot_report.md`; `reports/section_c33_oracle_asymmetry_gen2.md` |
| figure sources, and the tests that bind them | `scripts/28_paper_figures.py`, `tests/test_paper_figures.py` |
| the withdrawn bootstrap statistic and why | §22.9 |
| the withdrawn 25× probe-seed multiplier | §22.3 |
| probe calibration, capacity, pooling, text-domain generality | §20, §21, C24, C25 |
| the falsified lexical-locality hypothesis | `docs/LEXICAL_LOCALITY.md` |

---

## Notes for the next revision

- **§3 has been restructured after C33 and the title still stands.** The pre-registered share
  replication failed, but the underlying separation replicated, so Finding 1 is now a curve
  (§3.2) with the failed summary reported as its own result (§3.3). That is a better paper
  than the one where the share had replicated: it demonstrates the measurement point twice —
  once about the field's baseline, once about our own statistic.
- ~~**The §3.2 figure is now the paper's most important exhibit**~~ **Drawn.** Figure 1,
  `outputs/paper_figures/fig1_oracle_gap_vs_n.png`, two lines per property, generator 1
  solid and generator 2 dashed, with the deployed-arm budget bands marked. Figure 2,
  `outputs/paper_figures/fig2_frontiers.png`, is the two frontiers with the crossings, one
  panel per generator. Both are produced by `scripts/28_paper_figures.py` from committed
  `outputs/*_summary/*.json` only, and every value they plot is asserted against its
  artifact by `tests/test_paper_figures.py`.
- **A units error was corrected in v2.** v1 wrote "0.37 to 0.68 of the baseline's hit
  rate", but 0.37 and 0.68 are the **absolute** gaps in hit-rate points; as a fraction of
  the baseline's own hit rate at N = 32 the range is 0.37 to 0.71. Both units are now
  stated, and `tests/test_paper_figures.py` pins the second one.
- **Cut list if over length:** §6's effective-λ arithmetic → appendix, keeping the one-line
  conclusion; the generator-2 architecture table → one sentence; §5's last paragraph
  (trivial-predictor flag) → footnote. Figure 1 is not on the cut list.
- **Still to draw, if space allows:** nothing required. A third panel showing the *share*
  against budget — the quantity §3.3 retires — was considered and rejected: plotting a
  statistic we are withdrawing gives it more space than it has earned.
- **Venue check:** confirm the ICBINB deadline and page limit before the cut list is applied.
