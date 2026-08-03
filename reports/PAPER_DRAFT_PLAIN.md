# A paper draft, in the simplest English I can write

Written 2026-08-03. This is a **draft of the paper**, not the lab report. The lab report is
`reports/pilot_report.md` and it is 4,000 lines long. This is the story that report tells,
written so that someone who has never worked on language models can follow it.

Every number here comes from the report. Where I say "we measured", it was measured. Where
I say "we did not measure", it was not.

---

## 1. Introduction

### 1.1 The setting, in everyday terms

A **language model** writes text one piece at a time. Give it "The cat sat on the", and it
produces a list of guesses for the next word — "mat", "floor", "chair" — each with a
probability. You pick one, add it, and ask again.

A **chemical language model** does the same thing, but instead of English it writes
molecules. Chemists have a way of writing a molecule as a line of text, called SMILES.
Aspirin is `CC(=O)Oc1ccccc1C(=O)O`. So a model that writes text can write molecules, and
the model we use — GP-MoLFormer-Uniq, a released model with 46.8 million parameters — does
exactly that. It writes valid, sensible molecules.

Now the practical problem. Suppose you don't just want *a* molecule. You want a molecule
with a particular property — say, a greasiness score (chemists call it cLogP) between 4.17
and 5.04, or exactly three aromatic rings. The model doesn't know you want that. It just
writes whatever it writes. Roughly 10% of what it writes happens to land in your target
range, because that is how the target ranges were chosen.

**How do you get more?**

### 1.2 The two obvious answers

There are two cheap answers, and they are the two this paper compares.

**Answer 1: sample a lot and keep the best.** Ask the model for 9 molecules. Check all 9
with a chemistry program. Keep whichever one lands in your target. This is called
**best-of-N**. It is embarrassingly simple and it works well.

**Answer 2: steer while writing.** At every step, the model offers you a handful of
candidate next characters. Train a small helper — a **probe** — that looks at the
half-finished molecule and guesses "if I add this character and let the model finish, how
likely is the finished molecule to land in the target?" Then tilt the model's choice toward
the candidates the probe likes. This is called **guided decoding**, and the specific recipe
we use is **FUDGE**.

Answer 2 is more interesting, more publishable, and the subject of a large literature.
Answer 1 is what most people would actually do.

### 1.3 What people usually report, and why it is not quite right

The usual comparison is: run guided decoding, run best-of-N, spend the same compute on
both, see which wins. Usually best-of-N wins, and papers say so.

We think that comparison, as usually run, has **three problems**. This paper is mostly
about those three problems.

**Problem 1 — it is two points, not a curve.** You pick one compute budget, you run both
methods at it, and you report which won. But methods have *shapes*. One might be better
when compute is cheap and worse when compute is plentiful. Two points cannot see that.

**Problem 2 — the baseline is allowed to cheat.** Best-of-N picks its winner by *actually
measuring* the property of each finished molecule, using a chemistry program that gives the
exact right answer for free. Guided decoding never sees that answer — it only sees the
probe's guess. So the comparison is not "steering versus selecting". It is "steering with a
noisy guess versus selecting with perfect knowledge". Those are very different things and
the literature usually reports the second while claiming the first.

**Problem 3 — nobody reports the probe's seed.** When you train the probe, you initialise
it randomly. Train it twice with different random starts and you get two slightly different
probes. Everyone reports error bars over *generation* randomness — the dice the model rolls
while writing. Almost nobody reports error bars over *probe-training* randomness. We
measured it and it is the same size.

### 1.4 What we found

Five things.

**(a) The two methods cross.** When we measured the whole compute curve instead of one
point, guided decoding is **ahead** of best-of-N at small budgets and **behind** at large
ones. At around 140 processed tokens per molecule, 8 of our 30 measured guidance settings
sit above the best-of-N curve — and that is against a best-of-N that *is* allowed to cheat.
Five of the eight have error bars that exclude zero.

**(b) It happens on a second, different model too.** We repeated the whole thing on a
GPT-2-style generator trained on a different chemical database: 5 of 30 cells cross there as
well. So this is not a quirk of one model.

**(c) What produces it is steering *harder*, not reading *deeper*.** We ran the full 2×2 —
final layer vs mid-network layer, crossed with λ=1 vs λ=2 — and λ beat depth in all six
comparisons, additively. The ordinary final-layer readout, the one everybody already uses,
starts beating best-of-N as soon as you set λ=2. **You do not need to go hunting for the
right layer.** (On one of three properties, at least; see §4.7 for the honest limit.)

**(d) Guidance still cannot chase best-of-N upward.** Guided decoding has a knob that costs
more — how many candidate characters you score at each step — and turning it up spends about
**10 times** more compute. On the first model it buys nothing at all (**-0.0218**). On the
second it *does* buy raw accuracy (up to **+0.2347**) — so our original claim there was
wrong and we retract it — but it still loses ground to best-of-N on every single arm,
because best-of-N improves faster over the same budget. **k is not a knob that buys nothing;
it is a knob that loses a race.**

**(e) Most of the reported gap was the cheating — but "87%" was the wrong way to say it.**
When we re-ran best-of-N with the same noisy probe guidance uses instead of the chemistry
program, the gap on the first model collapsed from about **-0.30** to about **-0.04**, and
we summarised that as *roughly 87% of the reported gap is the baseline's free access to the
right answer*. We then pre-registered that 87% as a prediction for the second model, and
**it failed** — 42% on one property, uncomputable on another, and over 100% on the third.

The reason turned out to be arithmetic rather than chemistry, and it is the most useful
thing we learned. That percentage is a fraction whose bottom half depends on **how much
compute the guided method happened to be using**, and the two models' guided runs sat at
very different points on that axis (about 131 tokens per molecule against 367–419). What
*does* transfer, almost exactly, is the raw quantity underneath: **how far apart the
cheating and non-cheating baselines are at the same number of samples.** The two models
agree on that to within 0.008–0.037 across the whole curve, and by 32 samples the cheating
is worth **0.37 to 0.68** of hit rate. So: the finding stands, our way of summarising it
does not, and we report both. See §4.6.

---

## 2. Background

You need five ideas to read the rest. None is hard.

### 2.1 A generator that writes one token at a time

Our model writes a molecule character by character (technically "token by token"). At each
step it produces a probability for every possible next token. We only ever look at its top
few — by default the top 8. Call the model's own opinion `log p_base`.

**The model is frozen.** We never change a single weight. No fine-tuning, no reinforcement
learning, no editing its internal activations. This is a hard constraint of the project and
it matters for interpreting everything below: we are asking what you can do to a model you
are not allowed to modify.

### 2.2 A probe that predicts the future

The probe is a small neural network — two layers, about 268,000 parameters — that reads the
model's internal state at the current position and outputs a guess about the **finished**
molecule's property. Not the current partial molecule: the finished one. That is why we
call it a *future-property* probe.

Concretely it outputs a probability `q` that the finished molecule will land in your target
range.

### 2.3 The steering rule

At each step, for each of the 8 candidate tokens, we compute:

```
score(token) = log p_base(token)  +  λ · log( q(token) + ε )
```

and sample from the softmax of those scores. `λ` (lambda) is the **guidance strength**. At
λ = 0 you get the plain model. Turn λ up and the probe's opinion matters more.

Two things follow from this formula that we will use later, and both are simple algebra:

- Multiplying `q` by any constant changes nothing, because the softmax subtracts a common
  constant. **Only the differences between candidates matter, not the levels.**
- Raising `q` to a power `α` is *exactly* the same as multiplying `λ` by `α`, because
  `λ · log(q^α) = (λα) · log q`. **A power-shaped correction to the probe is a λ knob in
  disguise.** (This identity is not ours — Dhariwal and Nichol state it for diffusion
  guidance in 2021.)

### 2.4 Counting compute honestly

You cannot compare two methods without agreeing what "same compute" means. We count
**processed generator tokens** — every token the big model actually had to look at. Not
wall-clock time, which is unstable and depends on the machine. Not the number of molecules
returned, which flatters guidance.

The arithmetic that matters: guided decoding must evaluate all `k` candidates at each step,
so it costs exactly `(k + 1) ×` the plain model. At the default `k = 8`, that is 9×. So
guidance at k=8 is compute-matched against **best-of-9**.

### 2.5 Pre-registration

Before every experiment, we wrote down what we predicted and what result would count as a
failure, saved it to a file, and hashed it. The tests check that every measurement file is
*newer* than the prediction file. This is not decoration — several predictions failed, and
because they were written down first, we had to report the failures.

---

## 3. Methods

### 3.1 What was frozen, and when

- The generator: never modified, pinned to one exact released revision.
- The target ranges and the intervention windows: fixed **before** any steering result was
  looked at, and never re-derived.
- Every random seed: recorded and reused.
- Every software version: recorded next to every output file.

### 3.2 The six properties

We use six molecular properties, chosen to span "easy to see in the text" (like counting
ring-closure digits) to "diffuse over the whole molecule" (like greasiness): aromatic ring
count, hydrogen-bond donor count, rotatable bond count, TPSA, cLogP, and QED. Three of them
— aromatic rings, HBD count, QED — are the "anchor" properties used for the expensive
experiments.

### 3.3 The experiments, in the order they were run

| # | Question | Short answer |
| --- | --- | --- |
| Phase 1 | Can a probe predict the finished property from a prefix? | Yes for cLogP, no for ring count |
| Phase 2 | Same, on six properties and an independent 50,000-molecule sample | Confirmed |
| §15 | Is there anything to steer? (measured with **no probe at all**) | Yes, a big lever at every position |
| §19 | Does tuning λ fix it? | Worth 1.3–1.7×, still loses |
| §20 (C18) | Does a better-calibrated or bigger probe fix it? | No |
| §21 (C17) | Does reading a different layer of the network fix it? | Better *prediction*, unclear steering |
| C23 | Same question, run end to end | Yes, it does help |
| C24 | Does any of this hold on GPT-2 and text? | The algebra yes, the morals no |
| C25 | Does pooling help? Does the probe's seed matter? | No; yes |
| C26 | The full best-of-N curve, 11 values of N | Guidance below it at 45 of 46 points |
| C27 | The same curve, **without** the chemistry oracle | Gap shrinks ~8× *(on this model, at this budget — see C33)* |
| C28 | The full guidance curve, 5 values of k | The crossing |
| C29 | Eight probe seeds, and a confound in C23 | Retracts a number, halves another |
| C30 | The crossing, re-run at eight probe seeds | Holds; one cell reverses and is withdrawn |
| C31 | The whole thing again on a second generator | The crossing holds |
| C32 | Depth vs λ, as a proper 2×2 | It is λ, not depth |
| C33 | C27's "~87%" again, on the second generator | **Fails**; the curve transfers, the percentage does not |

### 3.4 The one thing we insist on

**Validity gates before results.** Before trusting any new measurement, we re-ran a
*previously published* configuration through the new code and required it to reproduce
exactly. Examples:

- C26 reproduced its published call signature **bit-identically** on both hit rate and
  token count — residual 0.0, all seeds.
- C27 reproduced C26 on all 33 (property, N) means.
- C29 compared probe checkpoints tensor by tensor — difference 0.0 across 33 comparisons —
  and replayed generation end to end: **6,144 of 6,144 molecules identical**.

If a gate fails, the experiment stops. None has failed since the phase-2 fixes.

---

## 4. Results

### 4.1 There is something to steer (and this rules out the boring explanation)

Before blaming the probe, we checked whether the problem is simply that no choice of next
character can change the outcome. It is not.

At 400 held-out positions we took **all eight** candidate characters, and for each one let
the frozen model finish the molecule 16 times. That is 51,200 finished molecules, with
**no probe and no λ involved at all**. Then we asked: how much does the chance of hitting
the target differ between the best candidate and the worst?

A lot. Choosing the best available candidate would roughly **double to triple** the hit
rate, for all six properties (0.09–0.15 → 0.22–0.40).

So the lever exists everywhere. Our steering rule at λ = 1 pulls **4.8–10.9%** of it.

### 4.2 The three cheap fixes, all of which fail

**Make the probe better calibrated.** Calibration means making the probe's stated
probabilities match reality. We did it two ways. It works as calibration — the calibration
error falls 3–6×. It leaves the probe's *ranking* ability exactly unchanged, to the last
decimal, because calibration is a monotone map and ranking doesn't care about monotone maps.
And it makes generation **worse**.

Why? Because of the identity in §2.3. A power-shaped calibration is a λ rescale, the fitted
exponents are 0.405–0.618, all below 1, so calibrating the probe is *secretly turning λ
down*. We verified this is not just an argument: with `ε = 0`, the calibrated probe at λ=1
and the raw probe at λ=α return **the identical 1,536 molecules**.

**Make the probe bigger.** A hidden layer 4× wider — 6.9× the parameters — moves the
probe's accuracy by at most +0.0019, with a median change of **-0.0007**. Nothing.

**Read a different layer.** Every property is predicted best in the *middle* of the network,
not at the end. This overturned one of our own claims: we had said this model doesn't
represent aromatic ring count, because a probe at the final layer scores 0.7878 against a
dumb token-counting baseline's 0.8269. At probe point 3 it scores **0.8474**. The
information was there; we were reading the wrong place.

Does reading the better place help *steering*? Per-position, it looked like no. Run end to
end (C23), it is **yes** — and that gap between the cheap proxy and the real measurement is
one of the project's more useful findings.

But C29 then found a confound. A mid-network probe produces a wider spread of `q` across
candidates, and by §2.3's identity, a wider spread of `log q` **is a larger λ**. Correcting
for that, the three headline gains fall from +0.0964/+0.1096/+0.0701 to
**+0.0375/+0.0507/+0.0217** — so **54% to 69% of the effect was steering harder, not
reading deeper.** The effect is real (3 of 3 anchors survive at 8 probe seeds, intervals
excluding zero) and about half the size first reported.

### 4.3 The main result: a crossing, not a loss

This is the part that is new.

**First, the full best-of-N curve.** We measured best-of-N at N = 1, 2, 3, 4, 6, 8, 9, 12,
16, 24, 32, giving a proper compute-versus-accuracy curve. It is concave — each extra sample
buys less than the last — and it spans **32×** in compute. Then we priced all 46 guidance
settings we had ever run against it, each at its own exact budget. **45 sit below it.**

**Second, the honest comparator.** We re-ran the entire curve with best-of-N forbidden from
using the chemistry program, and forced to select using the *same probe, same probe point,
same target range, same binning* that guidance uses. The deployed gap:

| property | vs. best-of-N **with** the oracle | vs. best-of-N **without** it | share of gap that was the oracle |
| --- | ---: | ---: | ---: |
| aromatic rings | -0.3532 | **-0.0439** | 0.876 |
| HBD count | -0.2472 | **-0.0292** | 0.882 |
| QED | -0.3715 | **-0.0522** | 0.859 |

**These are first-model numbers, and the last column does not survive the second model.**
We pre-registered that column as a replication target and it failed — §4.6 is the whole
story. The first two columns are what travels.

Counting settings rather than properties: **1 of 46** sits above the oracle curve; **15 of
46** sit above the honest one. None of the 15 is the configuration we actually deployed,
which is why this re-scopes the result rather than reversing it. That count *does*
replicate: 7 of 30 and 18 of 30 on the second model.

**Third, the guidance curve.** Guidance also has a compute knob — `k`, how many candidates
you score. We swept k ∈ {2, 4, 8, 16, 32}. Cost spans **10.2× to 11.1×** within a strand,
and cost is exactly `(k+1) ×` base, verified to residual 0.

Now put the two curves on the same axes. At k = 2 and k = 4, guidance costs 136–150 tokens
per molecule; best-of-N at that budget has barely started, drawing about three samples.
**8 of our 30 guidance cells sit above the oracle-holding best-of-N curve:**

| setting | k | advantage over oracle best-of-N | error bar excludes 0? | validity |
| --- | ---: | ---: | :---: | ---: |
| HBD count, mid-layer (probe point 4), λ=2 | 2 | +0.2499 | yes | 0.9941 |
| HBD count, mid-layer, λ=2 | 4 | +0.2093 | yes | 0.9889 |
| aromatic rings, mid-layer (probe point 3), λ=2 | 4 | +0.1690 | yes | 0.9792 |
| HBD count, mid-layer, λ=1 | 2 | +0.1325 | yes | 0.9967 |
| HBD count, **the deployed setting** (final layer, λ=1) | 2 | +0.0846 | yes | 0.9993 |
| HBD count, mid-layer, λ=2 | 8 | +0.0267 | no | 0.9935 |
| HBD count, mid-layer, λ=1 | 4 | +0.0218 | no | 0.9967 |
| aromatic rings, mid-layer, λ=2 | 2 | +0.0007 | no | 0.9909 |

None is extrapolated past the measured grid. None is degenerate — the molecules are 98–100%
valid. These are **real wins at cheap budgets**.

Two honest caveats belong right here. **Six of the eight are the same property** (HBD count);
the other two are aromatic rings; QED never crosses. And **all eight were run with one
trained probe** — which is why we then re-ran them with eight, in §4.6.

### 4.3.1 We re-ran the crossing with eight different probes, and it held

This is the experiment that decides whether the table above is real, so it is worth being
precise about what was done. We trained the probe eight times from eight different random
starts, re-generated every winning cell with each one, and priced all of them against the
**same** best-of-N curve — the same curve, not a re-measured one, because best-of-N selects
with the chemistry program and does not care which probe we trained.

Before trusting any of it we checked that probe number one reproduced the original run
exactly. It did: **12,288 of 12,288 molecules identical**, hit rates and token counts equal
to the last decimal. So the other seven probes were measuring the probe and nothing else.

| cell | k | one probe (before) | eight probes (mean) | 95% interval | probes positive |
| --- | ---: | ---: | ---: | --- | ---: |
| HBD, mid-layer, λ=2 | 2 | +0.2499 | **+0.2691** | [+0.2348, +0.3035] | 8/8 |
| HBD, mid-layer, λ=2 | 4 | +0.2093 | **+0.2000** | [+0.1694, +0.2306] | 8/8 |
| aromatic rings, mid-layer, λ=2 | 4 | +0.1690 | **+0.1052** | [+0.0586, +0.1517] | 8/8 |
| HBD, mid-layer, λ=1 | 2 | +0.1325 | **+0.1326** | [+0.1104, +0.1547] | 8/8 |
| HBD, **deployed setting** | 2 | +0.0846 | **+0.1044** | [+0.0895, +0.1194] | 8/8 |
| HBD, mid-layer, λ=1 | 4 | +0.0218 | +0.0207 | [-0.0073, +0.0486] | 6/8 |
| aromatic rings, mid-layer, λ=2 | 2 | +0.0007 | **-0.0341** | [-0.0640, -0.0042] | 2/8 |

Three things to take from it.

**The setting we actually deploy replicated best**, and its margin got *bigger*: +0.0846 with
one probe, +0.1044 averaged over eight, every one of the eight positive, and the smallest
spread of any row. A result that grows under replication is the opposite of what a
cherry-picked result does.

**One row reversed.** The bottom row was +0.0007 with one probe — a win by a hair. With eight
probes it is **-0.0341**, and the interval excludes zero *on the losing side*. That +0.0007
was not a small win; it was the luckiest of eight draws from a distribution centred below
zero. We delete that row rather than annotate it.

**Two rows were never answerable with one probe.** For those two, the spread across probes is
*larger than the margin being reported*. That is a fact about the old protocol, not about the
result: whatever their sign, one probe could not have told you.

### 4.3.2 The honest problem with §4.3.1

We wrote down, before running, three conditions that would make the experiment void. One of
them was "any cell where more than 10% of molecules fail to parse". **It fired** — on one of
56 (cell, probe) combinations. So the *pre-registered verdict is that the experiment is
uninterpretable*, and that is what we record.

One point in 56 voiding the other 55 is a badly written rule, but rewriting a rule after
seeing the data is exactly the thing pre-registration exists to prevent. So instead we threw
away the **entire offending cell** — all eight of its probes, including the seven that were
fine — and re-scored against the same thresholds. That cell was one of the *winners*, so
throwing it out makes the result harder to get, not easier.

Re-scored that way, the crossing is confirmed: five cells positive against a threshold of
five, four intervals excluding zero against a threshold of three.

A reader who declines that repair gets no result from this experiment, and is entitled to.

### 4.3.3 The probe's seed changes whether the molecules are even valid

This came out of the failed check and is arguably more useful than the check itself. On one
arm — aromatic rings, mid-layer, λ=2, k=4 — validity across the eight probes runs 0.9792,
0.9603, 0.9036, 0.9616, 0.9811, 0.9212, 0.9030, **0.8301**.

The last probe returns molecules that are **17% unparseable**. Nothing about how it was
trained was different; only the random seed. And the effect chains: invalid molecules are
longer, longer molecules cost more tokens, and more tokens means being compared against a
stronger point of the best-of-N curve — so that probe's advantage collapses from +0.1690 to
+0.0133.

We already knew (from the λ sweep) that validity collapses above λ≈2. What is new is that
**at** λ=2, which side of the cliff you land on depends on the probe's random seed. A
practitioner who trains one probe and ships it has a one-in-eight chance, on this arm, of
shipping a generator that breaks.

This is a stronger argument for reporting the probe seed than the accuracy argument is.

**Fourth, and this is what stops it being a triumph.** Guidance cannot chase best-of-N
upward. Over the full 10× range of its own knob (140.83 → 1447.42 tokens per molecule) the
deployed arm moves **-0.0218**. The cheapest setting is the best one. Over the identical
budgets, oracle best-of-N gains **+0.6834** and honest best-of-N gains **+0.2922**. Validity
gets *worse* as k rises. Five of six strands are worse at k=32 than at k=4.

**So the shape of the result is: guidance is efficient and it does not scale.**

### 4.4 The probe's random seed is as noisy as everything else

We trained the probe with **eight** different random initialisations and re-ran generation
with each.

| setting | probe-seed sd | ratio to generation-seed sd | 95% interval |
| --- | ---: | ---: | --- |
| HBD, probe point 4, λ=2 | 0.0366 | 1.71 | [0.96, 3.65] |
| aromatic rings, probe point 3, λ=1 | 0.0284 | 1.38 | [0.77, 2.95] |
| QED, probe point 4, λ=1 | 0.0115 | 0.64 | [0.36, 1.36] |
| the deployed setting, three properties | 0.0142–0.0241 | 0.63–1.18 | all contain 1 |

**Every interval contains 1.** Probe-seed noise is roughly the same size as generation
noise. It is never reported anywhere.

Why that matters: at the deployed setting the hit rates being compared are 0.20–0.47, and
the probe-seed sd is 0.014–0.024. Several effects this literature treats as findings are
that size. A paper that trains one probe and puts error bars over generation seeds is
reporting an error bar that omits a source of variation as large as the one it includes.

### 4.5 Things that did not replicate, and one number we got wrong

We are reporting these because leaving them out would make the paper look better than the
work was.

**Five pre-registered predictions failed.** The lexical-locality hypothesis this project
was built to test failed in the units it committed to (rank correlation -0.886 against its
own prediction). We report the falsification.

**The 25× claim is retracted.** We reported that probe-seed variance was ~25× generation-seed
variance. It is not. That figure divided a probe-seed spread by the **smallest** of three
generation-seed spreads (0.0672, 0.0413, 0.0040 — the 0.0040 was used). Done properly, at
eight seeds, it is 0.63–1.71 and every interval contains 1. The finding survives without
the multiplier; the multiplier does not survive.

**A statistic we used is vacuous.** We reported "seed-paired bootstrap confidence intervals"
over three seeds. At n = 3 the percentile bootstrap of a mean is **exactly [smallest,
largest]** of the three numbers — because the chance all three resamples land on the
smallest is 1/27 = 0.037, which is bigger than 0.025. So "the interval excludes zero" was
never an interval statement; it was "all three seeds have the same sign", which has a null
probability of 0.25 and cannot reject anything. We withdrew it everywhere and replaced it
with a t-interval on 2 degrees of freedom, publishing the raw three values alongside.

**On text, the moral does not travel.** We repeated the two central claims on GPT-2 with
three text attributes. The algebra reproduced exactly — the calibrated-probe-equals-rescaled-λ
identity returns the identical 1,536 sequences. But one fitted exponent came out at **1.6154**,
above 1, because that probe *over*-predicts rather than under-predicts, so calibration
**helped** there. "Calibration is a λ rescale" is general; "and therefore it hurts" is not.
And the mid-network layer, which helps steering on molecules, *hurts* it on two of three text
attributes.

---

### 4.6 We tested our own headline number, and it broke

This is the finding we were most attached to, so we tested it hardest, and it did not
survive in the form we published it.

**What we had claimed.** Best-of-N picks its winner by running the chemistry program on the
finished molecule. Guided decoding never gets to do that — it only ever sees the probe's
guess. So we re-ran best-of-N with the probe in place of the chemistry program, an
**equal-information** comparison, and the guided method's deficit shrank from about -0.30 to
about -0.04. We summarised that as: *the chemistry program accounted for about 87% of the
gap.* We called it the most portable thing the project had produced.

**What we did about it.** We wrote down, in advance and in a file we then froze, exactly
what would count as that replicating on the second model, exactly which single run per
property we would look at, and exactly what we would conclude if the number came out
elsewhere. Then we ran it.

**What happened.** It failed. On the second model the chemistry program accounted for
**42%** of the gap on aromatic rings, **could not be computed at all** for HBD count — that
run was already *above* the baseline, so the fraction had no sensible denominator — and
**144%** on QED, meaning removing the chemistry program flipped the sign rather than
shrinking the gap. Our pre-registered rule required at least 50% on every property where the
number exists. It does not.

**Why, and this is the part worth keeping.** The percentage is a ratio, and its denominator
is *how far apart the two baselines are at whatever compute budget the guided run happened
to use*. The two models' guided runs did not sit at the same budget, and nobody chose that:
about **131** processed tokens per molecule on the second model against **367–419** on the
first. In sample terms that is "after about 3½ draws" against "after about 9 draws". The two
baselines have barely separated at 3½ draws and have separated a lot by 9. We were dividing
by two different things and calling the answers comparable.

**What survives, and it survives cleanly.** Strip the ratio away and look at the raw
separation between the cheating and non-cheating baselines *at the same number of draws*,
and the two models agree to a mean absolute difference of **0.0111** (HBD count),
**0.0081** (QED) and **0.0369** (aromatic rings) across all eleven points of the curve. It
is 0 at one draw, because with one candidate there is nothing to select, and it grows to
**0.37–0.68** by thirty-two. The direction holds on all three properties of both models:
taking the answer key away from the baseline always makes guided decoding look better, never
worse. And the count of guided settings that beat the baseline once you level the
information more than doubles on both models — 15 of 46 against 1 of 46 on the first, 18 of
30 against 7 of 30 on the second.

**So the honest form of the claim is a curve, not a number:** *the more compute you give
best-of-N, the more of its advantage is the answer key rather than the search.* That
sentence is about the comparison and holds regardless of where anyone's method sits on the
axis. The percentage was about our own experiment's budget, which is not a fact about
anything.

We are leaving this in the paper at full length rather than quietly swapping the curve in
for the ratio, because it is the same lesson as §4.5 pointed at the authors instead of at
the field: **a summary statistic that moves by a factor of three when the budget moves is
the wrong summary, and we published it first.**

---


### 4.7 We did the whole thing again on a different model, and then found out why it works

Everything up to here is one model. That is the standard complaint about a result like
this, and it is a fair one, so we ran it again from scratch on a second molecular
generator: a **GPT-2**-style model with 87 million parameters trained on the ZINC database.
It is genuinely different — different attention mechanism, different training data,
different tokenizer, roughly twice the size. Same SMILES though, so nothing about how
molecules are written changed.

**The crossing happened again.** 5 of 30 cells beat the oracle-holding best-of-N at their
own budget, with error bars excluding zero.

**But the setting we deploy did *not* cross this time** — on any of the three properties.
Only the mid-network probe at λ=2 crossed. Which raised an awkward question: does the
crossing need a carefully chosen layer, re-picked for every new model? If so, the method
is a lot less useful than it looked.

#### The 2×2 that answered it

The first run only tested two of the four combinations: (final layer, λ=1) and
(mid layer, λ=2). So "reading deeper helps" and "steering harder helps" made *exactly the
same prediction*. There was no way to tell them apart. We ran the two missing corners.

| property | k | reading mid-network is worth | doubling λ is worth |
| --- | ---: | ---: | ---: |
| hbd_count | 2 | +0.0398 | **+0.1581** |
| hbd_count | 4 | +0.0399 | **+0.1360** |
| aromatic_rings | 2 | +0.1612 | **+0.1867** |
| aromatic_rings | 4 | +0.1828 | **+0.2034** |
| qed | 2 | +0.0190 | **+0.0482** |
| qed | 4 | +0.0206 | **+0.0524** |

**λ wins all six.** And the two effects simply add up — there is no interaction we can
detect, meaning depth and λ do separate jobs and you can reason about them one at a time.

#### The payoff

Once we set λ=2, **the ordinary final-layer readout crosses too** — no mid-network probe,
no per-model layer search:

- HBD count, k=2: **+0.1768** [+0.1304, +0.2232]
- HBD count, k=4: **+0.1105** [+0.0547, +0.1663]

So the first run's conclusion — *you must re-select the layer per model* — was **wrong**,
and we retract it. What the deployed setting was missing was λ, not depth.

**The caveat, and it is a real one:** that only worked for HBD count. For aromatic rings
and QED the final-layer readout still loses even at λ=2. One property of three.

#### One claim of ours died here

We had said guided decoding *"converts none of its extra compute into accuracy"*. On the
second model that is false — turning k up raises the raw hit rate by as much as **+0.2347**.

But it does not help you *win*, because best-of-N improves faster over the same budget. On
all six arms the gap to best-of-N gets worse as k rises:

| arm | gap at k=2 | gap at k=32 |
| --- | ---: | ---: |
| hbd count deployed | +0.0317 | -0.6035 |
| hbd count mid | +0.2295 | -0.3765 |
| aromatic rings deployed | -0.1756 | -0.5401 |
| aromatic rings mid | +0.1724 | -0.2507 |
| qed deployed | -0.0809 | -0.7399 |
| qed mid | -0.0137 | -0.6841 |

So the corrected sentence is: **k is not a knob that buys nothing — it is a knob that
loses a race.** That version is true on both models; the old one was true on one.

#### One more thing, which is a little uncomfortable

On the second model, the probe reading 768 numbers from inside the network scores **0.8687**
at predicting aromatic rings. A dumb baseline that just counts 21 things in the text
scores **0.8683**. A tie.

That same tied-with-counting probe produces the **largest** crossing in the entire
experiment, +0.2473.

This is the oldest finding in the project — being able to *predict* a property and being
able to *steer* it are different skills — showing up again on a different architecture, a
different corpus and a different tokenizer. It is also a warning: if we had picked which
probe to use by "how much does it beat the dumb baseline", we would have thrown away the
best steerer we have.

---

## 5. Discussion

### 5.1 What we think the finding actually is

Not "guided decoding fails". Not "guided decoding works". It is:

> **Probe-guided decoding is compute-efficient and compute-inelastic.** It converts a small
> budget into accuracy better than best-of-N does, and it cannot convert a large one at all.
> Best-of-N is the opposite: wasteful when cheap, and it scales.

That is a genuinely useful thing to know if you are choosing a method, and it is invisible
if you compare at one budget — which is what the literature does.

And since §4.7 we can say **how** to reach the efficient regime, which is the practical half:

> **Turn λ up to about 2 and keep k small.** Do not go looking for the right layer to read.

That matters because the two are not equally cheap. Finding a good mid-network layer means
training probes at every layer, holding out data to choose between them, and redoing it for
every new generator. Doubling λ is one number in a config file. We measured both on the
second generator and **λ won every comparison** — and the readout everyone already uses, the
final layer, starts beating best-of-N as soon as λ = 2.

The honest limit on that: it held on one of the three properties for the *deployed* readout
(HBD count). On the other two you still need the mid-network probe. So "λ first, depth if you
must" is the recipe, not "λ instead of depth".

### 5.2 Two recommendations for how these comparisons should be reported

These are, honestly, the parts most likely to be worth something to other people.

**Report the frontier, not the point.** A single matched-compute comparison is a sample from
a curve at an arbitrary location. If the two curves have different slopes — and they do —
the winner depends entirely on where you sampled.

**Match the information, not just the compute.** If your baseline selects with ground truth
and your method sees only a learned score, you are measuring the value of ground truth, not
the value of your method. Report both: the oracle comparison, because that is what a
practitioner with a cheap oracle would actually get, and the equal-information comparison,
because that is what tells you about the method.

**And report that difference as a curve, not as a percentage.** This is the recommendation
we learned the hard way, in §4.6. We published "the oracle was worth about 87% of the gap",
pre-registered it on a second model, and watched it come out at 42%, undefined and 144% —
not because the models differ, but because a percentage of *your* gap is normalised at
*your* compute budget and does not travel to anyone else's. The separation between the two
baselines at matched sample count does travel: to within 0.008–0.037 across two models that
share no weights, no tokenizer, no training data and no attention mechanism.

And a third, smaller one: **report the probe's training seed**. It costs one more training
run to find out whether your effect is bigger than the noise in your own probe.

### 5.3 Limitations, stated rather than defended

1. **Two generators, one domain.** GP-MoLFormer-Uniq and a GPT-2-style ZINC model, both
   SMILES molecules, both decoder-only transformers. The crossing replicates on both
   (§4.7); the separate GPT-2/text run (C24) covers the algebra only. Nothing here speaks
   to diffusion models, graph generators, or other chemical serialisations.
   **1b. The two generators' guided runs never sat at the same compute budget** — about 131
   tokens per molecule against 367–419 — which is what broke our percentage in §4.6.
   Matching them would need a run on the first model about three times cheaper than any we
   did. Comparing at matched sample count is the closest control available and is not a
   substitute for one.
2. **The cheap-end wins are cheap-end wins.** They happen where best-of-N has drawn about
   three samples. They do not become large-budget wins and we do not claim they do.
3. **A practitioner has the oracle.** RDKit is free and fast. Nobody deploying best-of-N on
   these properties would use a learned selector. The equal-information comparison is a
   *scientific control that says what the comparison was measuring* — it is not a
   recommendation about what to run.
4. **k and λ were not crossed.** We swept k at three fixed λ values. Whether the shape of
   the k response changes at other λ is unmeasured.
5. **Three generation seeds on most arms.** Enough to see a large effect, not enough to
   resolve a 0.008 difference, and we say so rather than decorating it.
6. **Bounded targets, not maximisation.** We steer *into a band*. Practitioners often
   maximise. The failure mode we see (molecules fragment) is probably specific to bounded
   targets, and the literature's canonical failure mode (long greasy tails) is probably
   specific to maximisation.
7. **The probe architecture is fixed.** One two-layer MLP on one position's hidden state.
   Pooling was tried and did not help; other probe families were not tried.

### 5.4 What we would do next

- **Run the crossing on a second generator.** This is the single thing that would most
  strengthen the paper, and it is the thing a reviewer will ask for first.
- **Find out where the crossing point is as a function of the target's base rate.** We have
  three anchors; the crossing budget differs between them and we do not know the rule.
- **Cross k with λ.** There is weak evidence the two interact.

---

## 6. One-paragraph version, for someone in a hurry

To make a frozen molecule-writing model produce molecules with a property you want, you can
either sample many and keep the best (best-of-N) or steer it as it writes using a small
learned predictor (guided decoding). The literature compares these at one compute budget and
reports that steering loses. We measured the whole compute curve for both and found they
**cross**: steering wins below about 150 processed tokens per molecule and loses above it,
and it cannot chase — its own compute knob spans 10× and buys nothing. We also found that
most of the reported gap is not about steering at all: the baseline is normally allowed to
select using the true property, which the steering method never sees. How much that is
worth grows with the baseline's budget — by 32 draws it is 0.37 to 0.68 of hit rate, on
both models — and we report it as that curve rather than as the single "about 8×" we first
published and then failed to replicate. Finally, the random seed used to train the
predictor moves the
final result about as much as the generation randomness everyone does report, and nobody
reports it.

---

*Companion documents:* `reports/pilot_report.md` (the full lab record; §22, §23, §24 and
§25 are the merges of the eleven follow-up experiments), `reports/ABSTRACT.md` (title and
abstract versions, with the claim inventory), `reports/PAPER_WORKSHOP_DRAFT.md` (the same
paper in workshop format), and `reports/section_c*.md` (one document per follow-up
experiment, each with its pre-registration reproduced verbatim). Figures:
`outputs/paper_figures/`, drawn by `scripts/28_paper_figures.py`.
