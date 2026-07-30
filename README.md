# Property-to-Go

> **Working paper title:** *When Does a Molecular Property Become Uncontrollable During
> Autoregressive Generation?*

## Decision

**Accepted.**

This is an inexpensive, automated, architecture-oriented chemistry project with a credible
scientific contribution and a small initial implementation.

The general decoding rule is not new: FUDGE already trained predictors on partial sequences and
used their scores to adjust a frozen generator's next-token probabilities. The molecular question
remains open enough to support a paper:

> Can a molecular language model predict the final property distribution from an unfinished
> molecule, and how does the effectiveness of guidance change over the course of generation?

The project does not need a more complicated method. It needs experiments that distinguish
predicting a final property from practically controlling it.

The targeted literature assessment in this README was last updated **2026-07-29**.

## Scientific thesis

Use a frozen autoregressive molecular generator \(\pi_0\). After a prefix \(x_{\leq t}\), estimate
the distribution of a completed molecule's property:

\[
q_\phi(y\mid h_t)
\approx
p_{\pi_0}(y_{\mathrm{final}}=y\mid x_{\leq t}),
\]

where \(h_t\) is a frozen-model representation of the prefix.

For a requested property interval \(I\), estimate

\[
q_I(h_t)
=
\Pr_{\pi_0}\!\left(y_{\mathrm{final}}\in I\mid x_{\leq t}\right).
\]

This is **generator-relative** or **practical reachability**. It is not a claim that a continuation
is chemically or mathematically impossible. A molecule may be theoretically possible while having
negligible probability under the frozen generator and sampling policy.

The paper needs only two scientific claims:

1. A small head over frozen molecular-LM states can predict the final property distribution from
   unfinished molecular sequences.
2. Different properties have different timing profiles for inference-time control.

## Two central curves

### 1. Predictability curve

At each generation position \(t\), measure how well the frozen state predicts the final property
distribution.

This asks:

> When can the model foresee the eventual molecular property?

Use held-out continuations to plot prediction quality against generation position. Suitable pilot
metrics are:

- distribution or interval prediction error;
- rank correlation;
- Brier score for target intervals;
- coarse probability calibration;
- discrimination between prefixes with high and low future target probability.

The pilot only needs to establish useful ranking and discrimination. Strong calibration is required
for a final paper that uses words such as “probability” or “reachability,” but it does not need to
block the first guided-decoding experiment.

### 2. Intervention-response curve

Apply guidance:

- throughout generation;
- only early;
- only in the middle;
- only late.

Measure the change in terminal property values and target hit rate.

This asks:

> When is the property still practically controllable?

The relationship between the two curves is the main result. A property might:

- become predictable before it becomes difficult to alter;
- be influenceable before its final value is accurately predictable;
- remain adjustable until late generation;
- become fixed after one early structural decision.

No more elaborate horizon taxonomy is needed for the pilot.

## Guided decoding

For each candidate next token \(a\), score

\[
\operatorname{score}(a)
=
\log \pi_0(a\mid x_{\leq t})
+
\lambda\log\left(q_I(h_{t+1}^{(a)})+\varepsilon\right),
\]

where \(h_{t+1}^{(a)}\) is the hidden state after appending candidate \(a\).

At every guided step:

1. obtain the eight most likely next tokens;
2. append all eight candidates in one batch;
3. calculate their candidate hidden states;
4. estimate their final target probabilities;
5. combine target probability with the original token log-probability;
6. sample from the reweighted candidates.

The molecular generator remains frozen.

For the pilot, recompute candidate prefixes in batches. Efficient caching in GP-MoLFormer's custom
linear-attention implementation is an optimization, not a prerequisite for determining whether the
idea works.

## Novelty and related work

The prior work narrows the novelty claim but does not reject the project.

| Work | What it establishes | Why Property-to-Go remains distinct |
| --- | --- | --- |
| [FUDGE (Yang & Klein, 2021)](https://aclanthology.org/2021.naacl-main.276/) | A predictor on partial sequences can be used to reweight a frozen generator's token probabilities. | Prevents claiming a new generic decoding principle. It does not study molecular property distributions or property-specific timing. |
| [Janz et al. (ICLR 2018)](https://openreview.net/forum?id=rkrC3GbRW) | A model can predict whether a partial sequence can eventually become valid, including for SMILES generation. | Eventual syntactic validity is materially different from the distribution of a completed molecule's physicochemical property. |
| [GAMS (Sha, 2025)](https://francis-press.com/papers/19920) | Candidate SMILES extensions can be scored with a fragment-property discriminator and added to decoder logits. | This is mechanically close, but it scores the current extended fragment rather than estimating the final-property distribution over future continuations. It does not study predictability and intervention-response curves. |
| [GP-MoLFormer-Sim (Navratil et al., 2025)](https://arxiv.org/abs/2506.05628) | Frozen GP-MoLFormer states can guide test-time decoding through contextual similarity. | It guides toward molecular exemplars rather than a calibrated final-property interval. |
| [InstGAN (Tang et al., IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0694.pdf) | Token-level molecular property critics can provide actor-critic rewards. | It updates the generator through GAN/RL training rather than controlling a frozen model using future target probability. |

Age and venue do not make prior work irrelevant. They determine what this paper should claim.

### Claim to make

> We estimate the distribution of a completed molecule's property from unfinished generations and
> compare when that property becomes predictable with when inference-time guidance remains
> effective.

### Claims not to make

- a new generic future-discriminator factorization;
- the first property-guided molecular decoder;
- universal or hard chemical reachability;
- impossibility outside a target interval;
- that a final-layer vector is a formally sufficient state for all future Transformer dynamics.

## Lean pilot

### Base generator

Start with [GP-MoLFormer](https://arxiv.org/abs/2405.04912), an open 46.8M-parameter
autoregressive SMILES model with linear attention and rotary positional encodings. The full model
was trained on roughly 1.1B SMILES. Code and weights are available from the
[official repository](https://github.com/IBM/gp-molformer).

Before collecting the full pilot:

1. load the released model and tokenizer;
2. reproduce unconditional generation;
3. verify validity, uniqueness, sequence-length, and property distributions;
4. expose hidden states for arbitrary prefixes;
5. verify that a stored prefix can be continued reproducibly;
6. pin the model revision, tokenizer, sampling policy, and RDKit version.

### Data

Generate approximately **50,000 molecules** with the frozen base model.

For each trajectory:

- randomly select one prefix from each sequence-length quartile;
- store the prefix token IDs and frozen hidden state;
- store prefix length and simple token/atom counts;
- calculate the completed molecule's cLogP and aromatic ring count;
- calculate molecular weight only as a diagnostic control.

This produces approximately 200,000 prefix examples without performing rollouts from every
training prefix.

Use trajectory-level train/validation/test splits. If the same canonical molecule appears multiple
times, keep all its trajectories in one split.

### Properties

Primary properties:

- **cLogP:** a continuous property distributed across many molecular fragments;
- **aromatic ring count:** a discrete structural property that may commit at localized ring
  construction decisions.

Diagnostic only:

- **molecular weight:** useful for checking that the pipeline learns an easy signal, but too
  confounded by atom count and sequence length for the main claim.

Choose target intervals from the frozen generator's empirical property distributions before
evaluating guidance.

### Predictor

Train a small two-layer MLP over the frozen hidden state.

Use a discretized distribution head:

- quantile-based bins for cLogP;
- categorical values or bins for aromatic ring count;
- cross-entropy training;
- target interval probability obtained by summing the appropriate bin probabilities.

This supports multiple target intervals without training a separate classifier for each one.

Required trivial baseline:

- prefix length;
- number of atom tokens;
- simple element-token counts;
- aromatic-token and ring-marker counts.

The first kill gate is whether the LM-state head provides substantially better held-out prediction
than these cheap prefix statistics.

### Repeated-continuation evaluation

Select approximately **500–1,000 held-out prefixes**, balanced across generation positions.

Generate **32 continuations per prefix** under the frozen base policy. Reuse the same continuations
to evaluate both properties.

The rollout bank estimates the actual conditional final-property distribution and supports:

- prediction error and rank correlation;
- Brier scores for target intervals;
- coarse reliability plots;
- prediction quality by prefix position.

Thirty-two continuations are sufficient for the pilot. Increase the number only for final
calibration figures if the project passes.

### Guidance experiments

Run:

1. guidance throughout generation;
2. early-only guidance;
3. middle-only guidance;
4. late-only guidance.

Define and freeze the position windows from the base model's sequence-length distribution before
looking at guided results.

*Implementation note.* The windows are quantiles of the pooled distribution of generated token
*positions* implied by that sequence-length distribution (position \(t = 1 \dots n\) for every
trajectory of length \(n\)), not quantiles of the final lengths themselves. Quantiles taken
directly over final lengths put the 33rd percentile near the median molecule's *end*, so
"early" would cover almost the entire trajectory. See `Windows.from_lengths` in
`src/property_to_go/guidance.py`.

Measure:

- target hit rate;
- absolute target error;
- property-distribution shift;
- validity;
- uniqueness;
- sequence and atom counts;
- generator compute;
- wall-clock time.

Sequence and atom counts are essential for detecting a controller that changes cLogP or ring count
only by changing molecule length.

### Pilot baselines

Use only:

1. ordinary frozen-model sampling;
2. compute-matched best-of-\(N\);
3. top-eight Property-to-Go guidance;
4. the simple prefix-statistics predictor.

For compute matching, count the tokens processed in candidate-prefix recomputation, not only the
number of Python forward calls. Report wall time separately because batching changes hardware
efficiency.

## Kill test

The project passes if any of the following holds:

1. the frozen-state head predicts final properties substantially better than simple prefix
   statistics;
2. guided decoding improves target hit rate at matched compute without a large validity collapse;
3. cLogP and aromatic ring count show clearly different, reproducible predictability or
   intervention-response timing.

The third outcome is sufficient even if the optimization gain is modest.

Reject or radically reframe if:

- the value head adds no information beyond length and token/atom counts;
- guidance changes the property only by changing sequence or molecule size;
- the timing curves are flat and indistinguishable across the two properties;
- all gains disappear against compute-matched best-of-\(N\).

## Implementation estimate

This is a moderate engineering task with a simple model.

| Milestone | Focused time |
| --- | ---: |
| Load generator, reproduce sampling, expose hidden states | 1–2 days |
| Create 10,000-molecule dataset and trivial baselines | 2–3 days |
| Train and inspect the first distribution head | 1 day |
| Scale to 50,000 molecules and implement top-eight reranking | 2–4 days |
| Repeated continuations, best-of-\(N\), and timing windows | 3–5 days |
| Debugging, reruns, and pilot figures | 2–4 days |

Expected total:

- **rough proof of concept:** about one focused working week;
- **credible pilot:** approximately two to three weeks.

The runtime should be measured in GPU-hours rather than a large training campaign. Engineering
around prefix continuation and batched candidate evaluation is likely to take longer than training
the small value head.

## Implementation sequence

### Days 1–2

- load GP-MoLFormer;
- reproduce unconditional sampling;
- expose and save prefix hidden states;
- confirm batched prefix recomputation.

### Days 3–5

- generate 10,000 trajectories;
- calculate terminal properties;
- train the value head;
- compare it with length and token/atom-count baselines;
- stop early if the LM state adds no useful signal.

### Days 6–10

- scale to approximately 50,000 trajectories;
- implement top-eight candidate scoring;
- run throughout-generation guidance;
- verify that observed shifts are not merely length shifts.

### Second week

- construct the repeated-continuation evaluation set;
- run compute-matched best-of-\(N\);
- run early-, middle-, and late-only guidance;
- produce the two central curves with repeated seeds.

## Distribution shift

The predictor is trained on ordinary base-model trajectories, while guided decoding may create
unfamiliar prefixes. This does not need to be solved before the pilot.

If it is the principal observed failure mode, try exactly one simple correction:

1. generate guided trajectories;
2. collect their prefixes and terminal properties;
3. mix them into the training data;
4. retrain the value head once;
5. repeat the held-out guidance test.

Do not turn the project into reinforcement learning unless this inexpensive correction produces a
clear benefit.

## Deferred until the signal is established

- explicit partial-graph prediction models;
- activation steering;
- multiple molecular generators;
- randomized SMILES or alternative serializations;
- iterative on-policy retraining;
- elaborate uncertainty estimation;
- large descriptor panels;
- a 500,000-trajectory dataset;
- the discrete molecular-diffusion variant.

These are possible paper-strengthening experiments, not requirements for the first implementation.

## Immediate next step

Build a small compatibility script that:

1. loads the pinned GP-MoLFormer checkpoint;
2. generates 100 unconditional molecules;
3. reports RDKit validity and the three diagnostic properties;
4. selects four prefixes from each sequence;
5. extracts their hidden states;
6. continues generation from a stored prefix;
7. batch-evaluates eight candidate prefix extensions.

If this works without model surgery, proceed directly to the 10,000-trajectory dataset.
