# Property-to-Go pilot report

*When does a molecular property become uncontrollable during autoregressive generation?*

All five specified phases were executed. This report separates **what was executed**
from **what was only implemented**; every number is backed by a JSON artefact under
`outputs/` and reproduced by a command in `docs/REPRODUCE.md`.

**Summary.** The pilot succeeds as a measurement and fails as an optimisation method.
A frozen GP-MoLFormer's hidden state predicts a completed molecule's cLogP better than
prefix token statistics at every generation position, but predicts aromatic ring count
*worse* than simply counting tokens — and the two properties also differ in *when* they
can still be steered. Against compute-matched best-of-N, however, guided decoding loses
decisively for both properties under both compute accountings. Section 9 gives the
kill-test verdict; section 8.7 documents a selection bug found mid-run, what it
corrupted, and what was re-run.

---

## 1. What was actually run

| Phase | Status | Artefact |
| --- | --- | --- |
| 1. Compatibility spike (100 molecules, 10 required operations) | **executed**, all checks pass | `outputs/compatibility/compatibility_report.json` |
| 2. 10,000-trajectory dataset | **executed** | `outputs/pilot_10k/` |
| 2. Frozen-state head + trivial baseline + combined + marginal | **executed** | `outputs/pilot_10k_heads/head_metrics.json` |
| 4. Rollout bank, 400 held-out prefixes x 32 continuations | **executed** | `outputs/pilot_10k_rollouts/` |
| 3. 50,000-trajectory dataset | **executed** | `outputs/pilot_50k/` |
| 3. Head training at 50k (gate **PASS** on cLogP) | **executed** | `outputs/pilot_50k_heads/head_metrics.json` |
| 4. Rollout bank at 50k, 800 prefixes x 32 (25,600 continuations) | **executed** | `outputs/pilot_50k_rollouts/` |
| 3. Guided decoding, cLogP, 6 conditions x 3 seeds | **executed** | `outputs/pilot_50k_guided_clogp/` |
| 5. Compute-matched best-of-N, cLogP, both accountings | **executed** | `outputs/pilot_50k_bestofn_clogp/` |
| 3. Guided decoding, aromatic rings, 6 conditions x 3 seeds | **executed** | `outputs/pilot_50k_guided_aromatic_rings/` |
| 5. Compute-matched best-of-N, aromatic rings (re-run after the section 8.7 fix) | **executed** | `outputs/pilot_50k_bestofn_aromatic_rings/` |
| 6. Length / size confound analysis, both properties | **executed** | `outputs/pilot_50k_confound_*/` |
| optional. One data-aggregation round (cLogP) | **executed once** | `outputs/pilot_50k_dagger_clogp/` |

## 2. Hardware and runtime

Apple M-series (arm64, 12 cores, 24 GB RAM), macOS 26.5.2 (Darwin 25.5.0), Python
3.12.8, numpy 1.26.4. The full software stack is recorded in
`outputs/provenance.json`. **No GPU was used.** CUDA
is unavailable on this machine, and MPS was benchmarked *slower* than CPU for this
46.8 M-parameter linear-attention model (18 mol/s on MPS at batch 256 vs 29 mol/s on
CPU at the same batch), so all runs are CPU float32 with `torch.set_num_threads(10)`.

Measured throughput, batch 256: unconditional sampling 24-29 molecules/s; un-cached
forward pass ~1.9 k tokens/s; cached incremental stepping ~2.9 k tokens/s. The
un-cached path is slow because the released fallback implementation of causal linear
attention materialises a `(batch, heads, length, features, dim)` tensor before the
cumulative sum (`fast_transformers` is not installed, so the C++ `causal_dot_product`
kernel is unavailable).

Runtimes of completed stages:

| stage | wall time |
| --- | --- |
| compatibility spike | 8.6 s |
| 10k generation | 470 s |
| 10k hidden-state extraction (441 k tokens) | 258 s |
| 10k head training, 3 inputs x 3 properties | 176 s |
| 10k rollout bank, 12,800 continuations | 902 s |
| 50k generation (2.22 M tokens) | 2,744 s |
| 50k hidden-state extraction (2.20 M tokens) | 1,301 s |
| 50k head training, 3 inputs x 3 properties | 187 s |
| 50k rollout bank, 25,600 continuations | 1,601 s |
| 50k guided decoding, cLogP, 6 conditions x 3 seeds | 1,671 s |
| 50k guided decoding, aromatic rings (run 2, committed) | 1,986 s |
| 50k compute-matched best-of-N, cLogP, both accountings | 2,847 s |
| 50k compute-matched best-of-N, aromatic rings, both accountings | 2,784 s |
| optional data-aggregation round | 2,546 s |
| full test suite, 153 tests incl. real-checkpoint contracts | 44 s |

Total wall time for the executed pilot is roughly 6.5 hours on this machine, dominated
by generation. Subject to the ~25% wall-clock reproducibility band established in
section 6.2.

## 3. Model revision and sampling policy

| item | value |
| --- | --- |
| model | `ibm-research/GP-MoLFormer-Uniq` @ `6eca879581e2302b4e1ab07bb02908636bddb4a2` |
| tokenizer | `ibm-research/MoLFormer-XL-both-10pct` @ `361063d0ad524ef77cf39b08469f6be770dc550f` |
| parameters | 46,781,184, frozen (`requires_grad=False` on every parameter) |
| python / torch / transformers / RDKit | 3.12.8 / 2.4.1 / 4.44.2 / 2024.3.5 |
| base policy pi_0 | `do_sample=True`, `temperature=1.0`, `top_k=None` (full vocabulary), `top_p=None`, `max_length=202` |
| guidance | top-8 candidates, `lambda=1`, `eps=1e-6` |
| seeds | dataset 20260729; prefix selection 12; split 11; head 1234; guidance {101, 202, 303}; rollouts 4242 |

The base policy matches the one in the official model card. Validity and uniqueness
reproduce the published numbers: 100/100 valid and unique in the compatibility spike;
0.9972 validity and 1.0000 uniqueness over 10,000 molecules (model card reports
validity 1.000, Uniq@10k 0.977).

## 4. Datasets

### 10,000 trajectories (`outputs/pilot_10k`)

10,000 generated; 28 rejected by RDKit; 1 shorter than the 8-token minimum; **9,971
kept**, all with distinct canonical SMILES. Four prefixes per trajectory, one drawn
uniformly from each sequence-length quartile, gives **39,884 prefix examples**.

Grouped splits by canonical completed molecule: 31,700 / 4,076 / 4,108 rows over
7,925 / 1,019 / 1,027 molecules. `check_no_group_leakage` asserts no molecule spans
two splits.

Base generator distributions (over the 9,971 kept molecules):

| quantity | mean | sd | 5% | 50% | 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| cLogP | 2.74 | 1.74 | 0.26 | 2.69 | 5.06 |
| aromatic rings | 1.82 | 1.10 | 0 | 2 | 4 |
| molecular weight | 366.2 | 85.4 | 257.4 | 352.5 | 488.1 |
| content tokens | 42.2 | 10.8 | 28 | 41 | 59 |

### Target intervals and windows, frozen before any guided run

Written by `scripts/02_generate_trajectories.py` into `target_intervals.json` and
`windows.json` from the base generator's own empirical distributions:

| property | interval | base rate |
| --- | --- | ---: |
| cLogP | [4.178, 5.063) — the 85th to 95th percentile band | 0.100 |
| aromatic rings | exactly 3, i.e. [3, 4) | 0.172 |
| molecular weight (diagnostic) | [434.6, 488.1) | 0.100 |

Position windows: **early** t < 15, **middle** 15 <= t < 29, **late** t >= 29. These
are the 33rd and 67th percentiles of the pooled distribution of *generated token
positions* (position t = 1..n for every trajectory of length n), which splits the
positions the base model actually visits into three equal-mass thirds.

## 5. Result 1 — predictability

### 5.1 Held-out prediction of the completed molecule's property (10k)

Three heads share one training recipe (two-layer MLP, 256 hidden units, GELU,
dropout 0.1, AdamW lr 1e-3, best epoch by validation NLL); only the input differs.
`marginal` is the input-independent training frequency, i.e. the floor.
Test split, 4,108 prefixes:

| property | head | NLL | E[y] MAE | target AUROC |
| --- | --- | ---: | ---: | ---: |
| cLogP | marginal | 3.00 | 1.40 | 0.500 |
| | trivial prefix stats | 2.790 | 0.978 | 0.723 |
| | **frozen LM state** | **2.748** | **0.940** | **0.764** |
| | both | 2.714 | 0.911 | 0.775 |
| aromatic rings | marginal | 1.44 | 0.83 | 0.500 |
| | **trivial prefix stats** | **0.953** | **0.533** | **0.799** |
| | frozen LM state | 1.151 | 0.629 | 0.758 |
| | both | 0.954 | 0.515 | 0.819 |
| molecular weight (diagnostic) | trivial prefix stats | 2.793 | 45.96 | 0.701 |
| | **frozen LM state** | **2.675** | **40.82** | **0.815** |
| | both | 2.639 | 39.50 | 0.822 |

Paired bootstrap (1,000 resamples) of frozen-state minus trivial, on the test split:

| property | NLL gain (nats) | target AUROC gain |
| --- | --- | --- |
| cLogP | **+0.042** [+0.022, +0.061] | **+0.042** [+0.024, +0.063] |
| aromatic rings | **-0.198** [-0.223, -0.174] | **-0.041** [-0.059, -0.023] |
| molecular weight | +0.118 [+0.095, +0.143] | +0.114 [+0.089, +0.139] |

Both cLogP intervals exclude zero, so the frozen state carries information about
final cLogP that prefix statistics do not. For aromatic rings the sign is reversed
and also excludes zero: **explicit token counting beats the learned readout.**

### 5.2 Predictability curve against the empirical conditional distribution

400 held-out prefixes balanced 100 per quartile, 32 base-policy continuations each
(12,800 continuations, 0.9975 of them valid). Each head is scored against the
distribution the frozen generator actually produces from that prefix, not against
the single completion the prefix came from. Spearman rho between the head's
predicted mean and the empirical conditional mean:

| property | head | Q1 | Q2 | Q3 | Q4 |
| --- | --- | ---: | ---: | ---: | ---: |
| cLogP | frozen LM state | **0.661** | **0.777** | **0.853** | 0.862 |
| | trivial prefix stats | 0.535 | 0.562 | 0.761 | 0.820 |
| | both | 0.613 | 0.758 | 0.867 | **0.899** |
| aromatic rings | frozen LM state | 0.805 | **0.848** | 0.783 | 0.676 |
| | trivial prefix stats | 0.804 | 0.782 | **0.880** | **0.913** |
| | both | 0.794 | 0.846 | 0.918 | 0.920 |
| molecular weight | frozen LM state | **0.522** | **0.763** | **0.888** | **0.902** |
| | trivial prefix stats | 0.327 | 0.416 | 0.501 | 0.792 |

**The two primary properties have qualitatively different predictability curves.**
For cLogP the frozen-state curve rises monotonically and dominates the prefix-statistics
curve at every position, with the largest advantage early (Q1 +0.13, Q2 +0.22). For
aromatic ring count the two curves **cross**: they are indistinguishable at Q1, and
from Q3 onward explicit counting pulls ahead while the frozen-state curve *falls*
(0.848 at Q2 to 0.676 at Q4).

How much of the property is still undetermined, measured as the standard deviation
across the 32 continuations of a prefix:

| property | Q1 | Q2 | Q3 | Q4 | marginal |
| --- | ---: | ---: | ---: | ---: | ---: |
| cLogP | 1.517 | 1.103 | 0.780 | 0.337 | 1.753 |
| aromatic rings | 0.948 | 0.700 | 0.461 | 0.110 | 1.072 |
| molecular weight | 76.7 | 52.1 | 36.2 | 19.4 | 91.2 |

Aromatic ring count collapses further: by Q4 its conditional sd is 0.11 of a count
(10% of the marginal sd), while cLogP still retains 19% of its marginal sd. The
decline in the frozen state's rank correlation for rings coincides with the regime
where the remaining variation is a small integer difference that an explicit counter
resolves and a 768-dimensional learned readout does not.

Figures: `outputs/pilot_10k_figures/rollout_*_spearman_vs_empirical_mean.png`,
`conditional_spread_*.png`, `reliability_*.png`.

### 5.3 The preregistered scaling gate

Threshold fixed before any result was inspected (`scripts/03_train_heads.py`,
`GATE_MIN_NLL_GAIN = 0.05`, `GATE_MIN_AUROC_GAIN = 0.02`): the frozen-state head must
beat the trivial baseline on at least one **primary** property by >= 0.05 nats **and**
>= 0.02 target AUROC, with bootstrap 95% CIs excluding zero.

**The gate as written FAILED.** cLogP met the AUROC criterion (+0.042 >= 0.02, CI
excludes 0) and the CI criterion on NLL, but its NLL gain of +0.042 nats fell short of
the 0.05-nat threshold. Aromatic rings failed with the opposite sign.

The specification's own wording for this gate is "a real held-out advantage over
trivial prefix statistics on at least one primary property". cLogP meets that reading:
both bootstrap intervals exclude zero, and the advantage is confirmed independently by
the rollout bank. The numeric threshold was mine and was stricter. **Both readings are
reported; the decision to scale to 50k is recorded as a deviation in section 7** and
was taken because (a) the cLogP advantage is statistically real though small, (b) the
divergence between the two primary properties is itself the third kill-test criterion
and needs the larger sample to be shown reproducible, and (c) 31,700 training rows for
a 768-input head leaves the underfitting hypothesis untested.

### 5.4 The same experiment at 50,000 trajectories

49,823 trajectories kept (171 invalid, 6 too short), 199,292 prefix examples,
grouped splits 158,588 / 20,752 / 19,952. Target interval for cLogP is
[4.173, 5.038) (base rate 0.100); aromatic rings exactly 3 (base rate 0.171).
Windows are unchanged at early < 15, middle [15, 29), late >= 29.

Test split, 19,952 prefixes:

| property | head | NLL | E[y] MAE | target AUROC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cLogP | marginal | 2.996 | 1.229 | 0.500 | 0.0945 | 0.052 |
| | trivial prefix stats | 2.748 | 0.986 | 0.754 | 0.0893 | 0.053 |
| | **frozen LM state** | **2.676** | **0.926** | **0.797** | **0.0877** | 0.053 |
| | both | 2.620 | 0.881 | 0.813 | 0.0885 | 0.061 |
| aromatic rings | marginal | 1.403 | 0.810 | 0.500 | 0.1374 | 0.008 |
| | **trivial prefix stats** | **0.918** | **0.527** | **0.825** | **0.1001** | 0.018 |
| | frozen LM state | 1.059 | 0.593 | 0.788 | 0.1156 | 0.010 |
| | both | 0.840 | 0.478 | 0.857 | 0.0932 | 0.009 |
| molecular weight (diagnostic) | trivial prefix stats | 2.698 | 47.53 | 0.737 | 0.0736 | 0.007 |
| | **frozen LM state** | **2.575** | **41.14** | **0.832** | **0.0675** | 0.006 |
| | both | 2.517 | 39.88 | 0.844 | 0.0659 | 0.009 |

Paired bootstrap (1,000 resamples), frozen-state minus trivial:

| property | NLL gain (nats) | target AUROC gain | E[y] MAE gain |
| --- | --- | --- | --- |
| cLogP | **+0.073** [+0.065, +0.081] | **+0.043** [+0.034, +0.053] | +0.060 [+0.053, +0.067] |
| aromatic rings | **-0.141** [-0.152, -0.131] | **-0.037** [-0.045, -0.030] | -0.066 [-0.072, -0.061] |
| molecular weight | +0.123 [+0.112, +0.133] | +0.095 [+0.083, +0.107] | +6.38 [+6.04, +6.74] |

**The pre-registered gate, unchanged, now passes on cLogP** (`SCALING GATE: PASS
(primary properties passing: ['clogp'])`): +0.073 >= 0.05 nats and +0.043 >= 0.02
AUROC, both CIs excluding zero. The 10k failure was a sample-size effect, which is
what section 5.3 predicted and why the run was scaled. Every 10k number in section 5
stands as reported; nothing was re-run to obtain a better result.

Aromatic rings fails at 50k for the same reason and with the same sign as at 10k, and
the gap is not shrinking with data. This is a **stable negative result for that
property**, not an underfitting artefact.

### 5.5 The predictability curve, measured directly on the head (50k)

Per prefix-position quartile, on the same test split — the specification's required
"performance by prefix-position quartile" for the head comparison:

| property | head | Q1 | Q2 | Q3 | Q4 |
| --- | --- | ---: | ---: | ---: | ---: |
| cLogP NLL | frozen LM state | **2.922** | **2.783** | **2.629** | **2.369** |
| | trivial prefix stats | 2.949 | 2.858 | 2.728 | 2.458 |
| cLogP AUROC | frozen LM state | **0.663** | **0.766** | **0.819** | **0.883** |
| | trivial prefix stats | 0.628 | 0.712 | 0.773 | 0.860 |
| rings NLL | frozen LM state | **1.301** | **1.139** | 0.970 | 0.827 |
| | trivial prefix stats | 1.304 | 1.156 | **0.890** | **0.322** |
| rings AUROC | frozen LM state | 0.646 | **0.750** | 0.827 | 0.872 |
| | trivial prefix stats | 0.645 | 0.728 | **0.833** | **0.978** |
| mol. weight AUROC | frozen LM state | **0.626** | **0.778** | **0.882** | **0.941** |
| | trivial prefix stats | 0.603 | 0.637 | 0.772 | 0.917 |

This reproduces the 10k rollout-bank finding at four times the data, on a different
estimator (head loss against the realised completion rather than rank correlation
against an empirical conditional mean), and sharpens it:

- **cLogP: the frozen state dominates at every position**, and its margin *grows*
  from Q1 (+0.035 AUROC) to Q3 (+0.046), i.e. the LM state keeps adding information
  the token counts do not have, right through the trajectory.
- **Aromatic rings: the curves cross between Q2 and Q3.** By Q4 the trivial counter
  reaches 0.978 AUROC and 0.322 nats while the frozen state stalls at 0.872 and
  0.827 nats — a 0.50-nat deficit. Late in a SMILES string the number of aromatic
  rings is almost literally written in the prefix; counting `c` tokens and ring-closure
  digits recovers it, and a learned readout of a 768-dimensional state does not.

**Calibration caveat.** For cLogP both heads are systematically **under-confident**:
mean predicted target probability 0.049 against a 0.102 base rate (ECE 0.053), and
the reliability curve has no support above predicted 0.3. Guidance uses
`log P(y in I | prefix + a)`, so a monotone squashing of these probabilities
attenuates the effective guidance strength at lambda = 1. This is a reason any null
intervention result below must be read as "no effect *at this lambda*" rather than
"no effect in principle". Rings and molecular weight are well calibrated
(ECE 0.010 and 0.006).

### 5.6 Repeated-continuation evaluation at 50k (Phase 4, spec scale)

800 held-out prefixes balanced 200 per position quartile, 32 base-policy continuations
each: **25,600 continuations**, 0.9965 valid, one rollout bank reused for all three
properties. Each head is scored against the distribution the frozen generator actually
produces from that prefix. Spearman rho between predicted mean and empirical
conditional mean:

| property | head | Q1 | Q2 | Q3 | Q4 | overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cLogP | **frozen LM state** | **0.776** | **0.841** | **0.819** | **0.845** | **0.820** |
| | trivial prefix stats | 0.601 | 0.742 | 0.744 | 0.795 | 0.721 |
| | both | 0.772 | 0.866 | 0.876 | 0.896 | 0.859 |
| aromatic rings | frozen LM state | **0.851** | **0.866** | 0.848 | 0.799 | 0.836 |
| | trivial prefix stats | 0.820 | 0.846 | **0.915** | **0.921** | **0.888** |
| | both | 0.855 | 0.908 | 0.952 | 0.946 | 0.928 |
| molecular weight | frozen LM state | **0.572** | **0.786** | **0.786** | **0.884** | **0.779** |
| | trivial prefix stats | 0.399 | 0.432 | 0.583 | 0.817 | 0.524 |

**Both findings replicate at spec scale on the independent estimator.** cLogP: the
frozen state leads at every quartile, by most at Q1 (+0.175). Aromatic rings: the
curves **cross between Q2 and Q3**, and the frozen-state curve *declines* from 0.866 to
0.799 while token counting climbs to 0.921 — the same qualitative crossover seen at 10k
(section 5.2) and in the head metrics (section 5.5), now on 800 prefixes.

How much of the property is still undetermined, as the standard deviation across the 32
continuations of a prefix:

| property | Q1 | Q2 | Q3 | Q4 | marginal |
| --- | ---: | ---: | ---: | ---: | ---: |
| cLogP | 1.509 | 1.160 | 0.818 | 0.324 | 1.878 |
| aromatic rings | 0.924 | 0.701 | 0.451 | 0.131 | 1.131 |
| molecular weight | 78.1 | 57.1 | 37.3 | 17.6 | 95.5 |

By Q4 aromatic ring count retains 12% of its marginal spread against cLogP's 17%. The
regime where the frozen state loses to counting is exactly the regime where the
remaining variation is a single integer step.

## 6. Result 2 — intervention response

Guided decoding with `score(a) = log p_base(a | prefix) + lambda * log(P(y in I | prefix + a) + eps)`,
lambda = 1, top-8 candidates, frozen-state head, 512 molecules per condition per seed,
three seeds (101/202/303). Target interval and windows were frozen in section 4 before
any guided molecule existed.

### 6.1 cLogP, target [4.173, 5.038), base rate 0.100

| condition | hit rate | abs. target error | validity | uniqueness | tokens/mol | wall s/seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unguided | 0.1067 +- 0.0018 | 1.701 | 0.9948 | 1.000 | 44 | 36 |
| truncation_control | 0.0899 +- 0.0175 | 1.671 | 0.9993 | 1.000 | 43 | 31 |
| early | 0.1601 +- 0.0131 | 1.259 | 0.9961 | 1.000 | 164 | 59 |
| middle | 0.1641 +- 0.0134 | 1.230 | 0.9961 | 1.000 | 155 | 59 |
| late | 0.1220 +- 0.0216 | 1.383 | 0.9980 | 1.000 | 170 | 160 |
| **throughout** | **0.2645 +- 0.0123** | **0.716** | 0.9967 | 1.000 | 406 | 212 |

**The intervention-response curve is not the predictability curve.** Guidance applied
early or middle each buys roughly +0.055 hit rate — about seven times the seed
standard error — while guidance applied **late buys +0.015, within seed spread**,
despite the head being at its *most* accurate late (Q4 AUROC 0.883 against 0.663 at
Q1, section 5.5). Predictability rises monotonically along the trajectory while
controllability falls away. By the time the frozen model can tell you what the
molecule's cLogP will be, there is very little left to steer: the remaining tokens are
ring closures, terminal groups and stereochemistry that no longer move the property
by a full interval width.

Late guidance is also the *worst* value for money: it costs 170 tokens and 160 s per
seed — more than early or middle, because scoring candidates at long prefixes is
expensive — and returns almost nothing.

**The gain is not an artefact of the top-8 restriction.** `truncation_control`
restricts sampling to the same eight candidates but never consults the head
(lambda = 0), and it lands **below** unguided at 0.0899. Restricting the candidate set
on its own slightly *hurts* the target rate, so measured against the correct control
the `throughout` effect is +0.175, not +0.158.

**No validity collapse.** Validity stays in 0.9948-0.9993 and uniqueness is 1.000 in
every condition, including `throughout`. Guidance at lambda = 1 does not push the
generator off-manifold.

**The gain is not a length effect.** Mean content length moves from 42.4 (unguided) to
43.9 (throughout) and heavy atoms from 25.6 to 26.6 — about 3% — and standardising
the guided molecules onto the unguided length distribution barely changes anything:

| condition | raw hit rate | length-matched | delta raw | delta length-matched |
| --- | ---: | ---: | ---: | ---: |
| throughout | 0.2645 | 0.2600 | +0.1579 | **+0.1534** |
| middle | 0.1641 | 0.1636 | +0.0574 | +0.0569 |
| early | 0.1601 | 0.1604 | +0.0535 | +0.0538 |
| late | 0.1220 | 0.1181 | +0.0153 | +0.0114 |
| truncation_control | 0.0899 | 0.0909 | -0.0168 | -0.0158 |

97% of the `throughout` effect survives length matching. Section 6.3 repeats this
matched on heavy-atom count and on the joint (length, size) cell, because token count
and molecule size can come apart.

### 6.2 Aromatic ring count, target exactly 3, base rate 0.171

| condition | hit rate | abs. target error | validity | tokens/mol | wall s/seed (run 1 / run 2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| unguided | 0.1708 +- 0.0209 | 1.337 | 0.9948 | 44 | 36 / 43 |
| truncation_control | 0.1824 +- 0.0099 | 1.285 | 0.9993 | 43 | 32 / 37 |
| early | 0.2856 +- 0.0147 | 0.964 | 0.9961 | 164 | 59 / 74 |
| middle | 0.3020 +- 0.0148 | 0.915 | 0.9980 | 156 | 55 / 69 |
| late | 0.2359 +- 0.0127 | 1.129 | 0.9935 | 166 | 161 / 200 |
| **throughout** | **0.4709 +- 0.0241** | **0.600** | 0.9928 | 409 | 188 / 239 |

Uniqueness is 1.000 in every condition.

This condition set was **executed twice** — once before the section 8.7 fix and once
after — and the two runs agree to four decimal places on every property-level
quantity: every hit rate, every length-matched hit rate, every validity, every token
count. Only the error metric changed. That is the evidence that replacing the committed
artefact was a metric correction rather than a different experiment. (The post-hoc
recompute in `corrected_target_error.json`, written before the re-run, also agrees to
four decimals — two independent routes to the same corrected numbers.)

**Wall time, however, did not reproduce.** The two wall columns are the two runs, and
the second is 20-25% slower for bit-identical work on an otherwise idle machine, most
plausibly thermal after several hours at full load. **Token counts were identical to
the digit.** This is a concrete demonstration of why the specification asks for wall
time to be reported *separately* from processed tokens: on this hardware, wall-clock
differences below roughly 25% carry no information. Every wall-time comparison in this
report should be read with that band in mind — including the guided-versus-best-of-N
timings in section 6.4, whose margins are smaller than 25% and therefore should be
treated as "comparable", not as a ranking.

**Guidance overshoots on a discrete count.** 14.5% of `throughout` molecules land on
*four* aromatic rings against 4.0% unguided -- the head pushes towards "more rings" and
routinely goes one past the target. That is why the corrected error for `throughout`
(0.600) is so much worse than the buggy figure (0.435): the overshoot was being scored
as a perfect hit. The effect is specific to guidance, scaling with guidance strength
across conditions (4.0% unguided, 3.6% truncation_control, 4.3% late, 6.3% middle,
8.2% early, 14.5% throughout).

Guidance moves aromatic ring count roughly twice as far as it moves cLogP
(+0.300 against +0.158), at the same token cost and with no validity collapse
(0.993 at `throughout`). Again `truncation_control` accounts for almost none of it
(+0.012), and length matching removes almost none of it:

| condition | raw | length-matched | delta raw | delta length-matched |
| --- | ---: | ---: | ---: | ---: |
| throughout | 0.4708 | 0.4599 | +0.3000 | **+0.2891** |
| middle | 0.3020 | 0.2920 | +0.1312 | +0.1212 |
| early | 0.2856 | 0.2823 | +0.1148 | +0.1115 |
| late | 0.2359 | 0.2316 | +0.0651 | +0.0607 |
| truncation_control | 0.1824 | 0.1815 | +0.0116 | +0.0107 |

96% of the `throughout` effect survives length matching.

### 6.2.1 The two properties have different intervention-response timing

Lift over unguided, and the same lift as a share of the `throughout` effect:

| window | cLogP lift | share of throughout | rings lift | share of throughout |
| --- | ---: | ---: | ---: | ---: |
| early | +0.053 | 34% | +0.115 | 38% |
| middle | +0.057 | 36% | +0.131 | 44% |
| **late** | **+0.015** | **10%** | **+0.065** | **22%** |
| throughout | +0.158 | 100% | +0.300 | 100% |

The shapes differ in a way that is stable across seeds. **cLogP is flat early-to-middle
and then collapses**: its late lift of +0.015 is smaller than the seed standard
deviation (0.022) and is the only window in either property that is not clearly
separated from zero. **Aromatic ring count peaks in the middle and stays steerable
late**, retaining 22% of the full effect against cLogP's 10%, with a late lift
(+0.065) five times its seed standard deviation (0.013).

This is chemically legible rather than mysterious. Near the end of a SMILES string the
remaining tokens are ring closures, terminal substituents and stereo markers. A ring
closure digit still changes the aromatic ring count by a whole unit -- the entire
target interval -- whereas bulk lipophilicity is already fixed to within less than one
interval width by the scaffold chosen much earlier (section 5.2: cLogP's conditional
standard deviation at Q4 is 0.337, against an interval width of 0.87).

**Predictability and controllability dissociate in both directions.** cLogP is the
property whose completed value the frozen state predicts *best* and steers *worst*
late; aromatic ring count is the property the frozen state predicts *worse than token
counting* (section 5.4) and yet steers *best*, including late. Whatever the LM state
encodes about a future property is not the same thing as leverage over it.

### 6.3 Length and size confound analysis

Section 6.1 matched on sequence length only. Token count and molecule size can come
apart -- branch and ring-closure tokens cost length without adding heavy atoms -- so
`scripts/09_confound_analysis.py` re-standardises each condition onto the **unguided**
stratum distribution under four estimators: raw, length (content tokens, bin 5), size
(heavy atoms, bin 3), and the joint cell. `coverage` is the share of unguided mass in
strata the condition actually visited; it is 0.98-1.00 everywhere below, so these are
not extrapolations.

Lift over unguided:

| property | condition | raw | length-matched | size-matched | joint-matched |
| --- | --- | ---: | ---: | ---: | ---: |
| cLogP | **throughout** | +0.1579 | +0.1534 | +0.1485 | **+0.1531** |
| | middle | +0.0574 | +0.0569 | +0.0537 | +0.0565 |
| | early | +0.0535 | +0.0538 | +0.0489 | +0.0471 |
| | late | +0.0153 | +0.0114 | +0.0086 | +0.0096 |
| | truncation_control | -0.0168 | -0.0158 | -0.0148 | -0.0133 |
| rings | **throughout** | +0.3000 | +0.2891 | +0.2812 | **+0.2751** |
| | middle | +0.1312 | +0.1212 | +0.1102 | +0.1093 |
| | early | +0.1148 | +0.1115 | +0.1060 | +0.1093 |
| | late | +0.0651 | +0.0607 | +0.0515 | +0.0537 |
| | truncation_control | +0.0116 | +0.0107 | +0.0127 | +0.0161 |

**Rejection criterion R2 does not fire.** Matching jointly on sequence length *and*
molecule size leaves **97%** of the cLogP `throughout` effect (+0.1531 of +0.1579) and
**92%** of the aromatic-ring effect (+0.2751 of +0.3000). Guidance is moving the
property itself, not smuggling it in through molecule size.

Size matching removes slightly more than length matching in both properties, and more
for rings than for cLogP, which is the expected direction: adding an aromatic ring adds
six heavy atoms, so a little of the ring effect genuinely *is* a size effect. It is a
small fraction of the total, not the mechanism.

### 6.4 Compute-matched best-of-N

N is solved from the guided run's **measured** tokens per returned molecule, so the
match is empirical rather than assumed. Best-of-N draws N molecules from the frozen
base policy and returns the one closest to the target, so it costs
`N x base_tokens_per_molecule` per returned molecule. `solve_best_of_n` **floors**,
so best-of-N is always given slightly *less* than the guided budget — the rounding
favours guidance.

Two accountings, as the specification requires:

| accounting | guided tokens/mol | base tokens/mol | N | budget used |
| --- | ---: | ---: | ---: | ---: |
| `actual` (cached candidate backend) | 406 | 44 | 9 | 399 / 406 = 98% |
| `full_recompute` (the README's reference implementation) | 9,300 | 44 | 212 | -- |

#### cLogP, `actual` accounting

| | guided `throughout` | best-of-9 |
| --- | ---: | ---: |
| hit rate | 0.2645 +- 0.0123 | **0.6133** |
| abs. target error | 0.716 | **0.134** |
| validity | 0.9967 | 1.0000 |
| uniqueness | 1.0000 | 1.0000 |
| tokens / molecule | 406 | 400 |
| wall seconds / seed | 212 | 278 (382 / 236 / 216) |

Best-of-N's first seed carries model warm-up; the two steady-state seeds average 226 s
against guidance's 212 s, so the two methods are within about 7% on wall time as well
as being matched on tokens. Guidance is not buying its deficit back in speed.

**Guidance advantage: -0.3488.** At equal token cost, plain rejection sampling with an
oracle scorer is more than twice as good at hitting the target and produces five times
smaller target error.

Best-of-9's hit rate is almost exactly what independent draws predict:
`1 - (1 - 0.1067)^9 = 0.6378` against 0.6133 measured. Nothing subtle is happening --
the baseline is simply strong, because a 10%-base-rate target is easy to hit within
nine attempts.

A sensitivity run at N = 10 (108% of the guided budget, i.e. rounding the other way)
was planned and **deliberately not run**: it would move the baseline from 0.6133 to
about 0.676, while the measured gap is 0.349. The verdict cannot depend on the
rounding direction, so spending an hour of compute to confirm it would have been
theatre.

**The one asymmetry that matters, stated plainly.** Best-of-N selects using RDKit on
*completed* molecules — ground truth — while guidance only ever sees a learned head on
*incomplete* prefixes. That is the baseline the specification mandates, and for a
property as cheap to evaluate as cLogP it is also the honest practical baseline: if
scoring a finished molecule costs microseconds, there is no reason to steer during
generation. The correct scope for this result is therefore **"for cheaply-evaluable
properties, top-8 guidance at lambda = 1 is dominated by compute-matched rejection
sampling"**. It is *not* evidence about regimes where property evaluation is expensive
relative to generation (docking, DFT, assays), which this pilot did not test and which
would require a different compute accounting to test properly.

#### Aromatic rings, both accountings

Re-run after the section 8.7 fix. Guided cost 409 tokens/molecule (`actual`) and
9,368 (`full_recompute`), against a base cost of 43.7.

| | guided `throughout` | best-of-9 | best-of-214 |
| --- | ---: | ---: | ---: |
| hit rate | 0.4709 +- 0.0241 | **0.8021 +- 0.0040** | **1.0000 +- 0.0000** |
| abs. target error | 0.600 | **0.201** | **0.000** |
| validity | 0.9928 | 1.0000 | 1.0000 |
| uniqueness | 1.0000 | 1.0000 | 1.0000 |
| tokens / molecule | 409 / 9,368 | 400 | 9,497 |
| wall seconds / seed | 188 | 226 | 702 |
| **guidance advantage** | -- | **-0.3312** | **-0.5291** |

Best-of-9 lands at 0.8021 against the binomial prediction of 0.8147, so the corrected
baseline now reconciles with theory for both properties (cLogP 0.6133 against 0.6378).
That agreement is what confirms the section 8.7 fix was correcting a real defect rather
than tuning a number.

**The two properties lose by almost the same margin** (-0.3312 for rings, -0.3488 for
cLogP under `actual`), even though rings are nearly twice as steerable in absolute
terms (+0.300 against +0.158 over unguided). The reason is that best-of-N's advantage
also scales with the base rate: rings' higher base rate (0.171 against 0.100) hands
the baseline the same extra leverage that guidance gained. Being easier to steer does
not make a property a better case for steering.

#### cLogP, `full_recompute` accounting

This is the accounting the specification actually asks for -- "processed generator
tokens **including full-prefix candidate recomputation**". Under it the guided run
costs 9,300 tokens per molecule, so the matched baseline is best-of-**212**
(64 molecules per seed; see `docs/REPRODUCE.md` for why the sample is smaller).

| | guided `throughout` | best-of-212 |
| --- | ---: | ---: |
| hit rate | 0.2645 +- 0.0123 | **1.0000 +- 0.0000** |
| abs. target error | 0.716 | **0.000** |
| validity | 0.9967 | 1.0000 |
| uniqueness | 1.0000 | 1.0000 |
| tokens / molecule | 9,300 | 9,409 |
| wall seconds / seed | 212 | 671 |

**Guidance advantage: -0.7355.** Every one of 192 returned molecules landed in the
target interval, which is what 212 independent draws at a 0.107 base rate predict
(`1 - (1 - 0.107)^212 > 0.9999`).

The two accountings bracket the honest answer rather than disagreeing: guidance loses
by 0.35 under the accounting most favourable to it (cached backend, floored N) and by
0.74 under the accounting the specification prescribes. **Rejection criterion R4 fires
under both.**

## 7. Deviations from the specification

Every deviation below was a deliberate choice; none was forced by a result.

**7.1 The scaling gate was applied twice, and the numeric threshold was mine.**
The specification says to scale "if the frozen-state head shows a real held-out
advantage on at least one primary property". I turned that into a numeric
pre-registered rule (>= 0.05 nats *and* >= 0.02 AUROC, CIs excluding zero) before
looking at any result. At 10k that rule failed on the NLL leg (+0.042 nats). I
scaled anyway, on the reading that the specification's own criterion was met.
Section 5.3 reports the failure; section 5.4 reports that the same rule **passes at
50k**. The honest summary is that my threshold was calibrated for a dataset size
that had not yet been run, not that the criterion was moved after seeing data.

**7.2 `deterministic_eval` is set to `True` on the frozen model.**
This is the only field of the released config this project changes, and it changes
no weight. GP-MoLFormer's linear attention uses random orthogonal feature
projections; with the released default the projections are **redrawn on every
forward pass**, so two forwards over the same tokens give different logits. Under
that default no result in this report would be reproducible, and the candidate
scoring in `score(a)` would be comparing candidates evaluated under different
random features. The projections are stored in the checkpoint, so setting this flag
pins them to the released values rather than inventing new ones. Verified by
`tests/test_model_contracts.py::test_forward_is_deterministic`.

**7.3 A cached candidate backend is used by default, alongside the full
recomputation the specification asks for.**
The specification prefers "simple batched full-prefix recomputation over modifying
GP-MoLFormer's linear-attention implementation". **No attention code was modified.**
Both backends are implemented: `_candidate_states_full` re-runs each `prefix + a`
from scratch, and `_candidate_states_cached` advances one step using the model's own
released `use_cache=True` path. `test_candidate_backends_agree` asserts the two give
the same hidden states to floating-point tolerance on the real checkpoint, so the
cached path is an optimisation, not a different method. Compute accounting reports
**both** counts for every run (`processed_tokens_actual` and
`processed_tokens_full_recompute`), and compute-matched best-of-N is run twice, once
matched against each.

**7.4 CPU only.** No CUDA device is available on this machine and MPS benchmarked
slower than CPU for this model (section 2). This lengthens wall-clock but does not
change any token count; wall time is reported separately from token accounting
throughout, as the specification requires.

**7.5 `transformers` is pinned to 4.44.2.** The released GP-MoLFormer modelling code
imports `transformers.onnx` (removed in v5) and assumes `PreTrainedModel` still
carries `GenerationMixin` (untrue from 4.50). 4.44.2 is the newest release that runs
the checkpoint's own code **unmodified**. The alternative — patching the released
model file — was rejected because it would make the frozen generator no longer the
released artefact.

**7.6 Phase 4 was run twice, at two scales.** The specification asks for 500-1,000
held-out prefixes. The 10k rollout bank used **400** prefixes (100 per quartile) and
is reported as a first pass; the spec-compliant run is the 50k bank in section 5.5.
Both are reported, not only the larger one.

**7.7 The 10k guided run in `outputs/pilot_10k_guided_clogp/` is a smoke test**, not
a result: 24 molecules per condition, one seed, five of the six conditions. It exists
to prove the guided loop runs end to end and is excluded from every conclusion.

**7.8 Free parameters the specification left open**, fixed before any guided run and
recorded in `configs/guidance.yaml`: 512 molecules per condition, 3 seeds, batch 64,
window quantiles at 1/3 and 2/3 of the pooled generated-position distribution.

**7.9 A sixth condition, `truncation_control`, was added** beyond the five the
specification names. It restricts sampling to the same top-8 candidate set as guided
decoding but never evaluates the head (`lambda = 0`). Without it, any difference
between guided and unguided decoding is confounded with the top-8 restriction itself
rather than with the property head. It adds no head compute and is reported
alongside the required conditions.

## 8. Failures and unresolved issues

**8.1 Aromatic ring count is a negative result and stays negative.** The frozen state
loses to counting `c` tokens and ring-closure digits, at both dataset sizes, with the
gap widening towards the end of the trajectory (Q4: 0.50 nats worse at 50k). More data
did not help. The specification's kill-test "hidden states add no information beyond
prefix statistics" is **met for this property**, and it is reported as such rather
than being dropped in favour of the property that worked.

**8.2 The head is under-confident for cLogP, and much more so where it is used.** On
held-out *base-policy* prefixes it predicts a mean target probability of 0.049 against
a base rate of 0.102 (ECE 0.053). On the *guided* prefixes the decoder actually visits,
section 9.2.1 measures 0.076 predicted against 0.267 observed — under-confident by a
factor of 3.5, ECE 0.190. The quantity guidance actually consumes is miscalibrated on
the distribution guidance actually induces. and no reliability support above predicted
0.3 (section 5.5). Because guidance scores with `log P(y in I | prefix + a)`, a head
that never emits a confident probability produces a compressed score range and a
weaker effective lambda. Nothing was recalibrated, because recalibration was not in
the specification and would have been a post-hoc change made after seeing guided
results. The consequence is that intervention results are conditional on lambda = 1
with an uncalibrated head, and a null would not be evidence against the method.

**8.3 Only one probe point was tested.** All results use the final layer
(`hidden_layer: -1`) and one head architecture (two-layer MLP, 256 units). Whether an
earlier layer, a larger head, or a different pooling would close the aromatic-ring gap
is untested. This is a genuine limit on the negative result in 8.1: it shows *this*
readout fails, not that the information is absent from the state.

**8.4 No GPU.** All numbers are CPU float32. The un-cached forward path is
additionally slowed because `fast_transformers` is not installed, so the released
fallback materialises a `(batch, heads, length, features, dim)` tensor before its
cumulative sum instead of using the C++ `causal_dot_product` kernel. This inflates
every wall-clock figure but changes no token count. The guided-vs-best-of-N wall-time
comparison is therefore valid *relative to itself on this hardware* and should not be
read as a general statement about the two methods' speed.

**8.5 A padding warning is emitted during guided decoding.** The released code calls
`warn_if_padding_and_no_attention_mask` on batches that contain pad tokens. Rather
than assume it benign, two tests were added against the real checkpoint
(`test_batch_rows_are_independent`, `test_cached_stepping_is_row_independent`) proving
that one row's padding cannot influence another row's logits under this linear
attention. Both pass, so the warning is cosmetic. It is recorded here because
"the warning looked harmless" is not an argument.

**8.6 Single-machine, single-run.** Head training uses one seed (1234). Guidance and
best-of-N use three seeds each and report spread, but the head itself was not retrained
under multiple seeds, so head-to-head NLL differences carry training-seed variance that
the paired bootstrap does not capture. The bootstrap CIs quantify sampling variance of
the test set, not initialisation variance.

**8.7 A selection bug was found and fixed mid-run; aromatic-ring best-of-N was
re-run.** Best-of-N ranked candidates by `target_distance`, the mathematical distance
to the half-open target interval. That distance is 0 at `value == hi`, so with the
aromatic-ring target [3, 4) a **four**-ring molecule scored 0.0 — tied with a genuine
three-ring hit — and best-of-N returned whichever appeared first.

It was caught by a consistency check rather than by a test: best-of-9 measured 0.6914
against a binomial prediction of 0.8147 from the observed base rate, a 7-sigma
shortfall, while cLogP reconciled to three decimal places (0.6133 against 0.6378).
Inverting the observed rates gave an implied per-candidate hit rate of 0.1225 against
a true 0.1708 — the signature of hits being diluted by ties, and continuous cLogP was
immune because `P(value == hi)` is zero for a float. A hypothesis that the candidate
pool was correlated was tested first and **rejected** (consecutive groups of nine
scored 0.7929 against 0.8136 +- 0.0247 for randomly permuted groups).

What it affected, and what it did not:

- **Hit rates were never affected**, anywhere. Every hit rate is computed as
  `lo <= v < hi`, which is correct. All hit-rate results in sections 5 and 6 stand.
- **Aromatic-ring best-of-N** was corrupted and was re-run after the fix. The
  discarded pre-fix figures were best-of-9 = 0.6914 / 0.7090 (seeds 101 / 303).
- **Absolute target error for aromatic rings** was understated everywhere, in guided
  runs too, because a four-ring molecule was scored as zero error. Corrected values
  are recomputed from the saved (seeded, therefore identical) molecules and stored in
  `corrected_target_error.json`; section 6.2 quotes the corrected column.
- **cLogP and molecular weight were unaffected** in every metric, since their interval
  edges are irrational-valued quantiles.
- **Head training and the guidance score were unaffected**, verified rather than
  assumed: the target label (`in_interval`), the categorical interval mask used for
  aromatic rings (`CategoricalBinner.interval_mask`) and the quantile mask all use the
  correct half-open test `lo <= v < hi`. So `P(y in I | prefix + a)`, every head
  metric in section 5, and the guided decoding score itself never saw the defect. The
  bug lived only in best-of-N's ranking and in the reported error.

Fixed by `selection_key`, which ranks membership ahead of distance, and `target_error`,
which measures a count property's miss to the nearest *attainable* target value. Three
regression tests were added; `test_best_of_n_prefers_a_real_hit_over_a_boundary_miss`
deliberately places the four-ring molecule first in the pool, so it fails against the
old code and passes against the new.

**8.8 Scope items deliberately not attempted**, per the specification: no
reinforcement learning, no partial-graph model, no activation steering, no second
generator, no alternative serialisation, no uncertainty estimation beyond the
discretised head and bootstrap intervals.

## 9. Interpretation

### 9.1 How the kill test is scored

The mapping below was written **before** the aromatic-ring guided runs and the
compute-matched best-of-N had produced any number, so that each verdict is a lookup
rather than a judgement made after seeing the result. README lines 299-316 define the
criteria; the evidence each one will be scored against is fixed here.

**Passes if any of:**

| # | Criterion | Scored against | Verdict |
| --- | --- | --- | --- |
| 1 | frozen-state head predicts final properties substantially better than prefix statistics | section 5.4 bootstrap CIs, both dataset sizes | see 9.2 |
| 2 | guided decoding improves hit rate at matched compute without large validity collapse | section 6 vs best-of-N, plus per-condition validity | see 9.2 |
| 3 | the two primary properties show clearly different, reproducible predictability or intervention-response timing | sections 5.2 / 5.5 / 5.6 (predictability) and 6 (timing) | see 9.2 |

**Rejects or radically reframes if any of:**

| # | Rejection condition | Scored against | Verdict |
| --- | --- | --- | --- |
| R1 | the head adds no information beyond length and token/atom counts | frozen vs trivial, per property | see 9.2 |
| R2 | guidance changes the property only by changing sequence or molecule size | `09_confound_analysis.py`, all four estimators | see 9.2 |
| R3 | the timing curves are flat and indistinguishable across the two properties | section 6, both properties | see 9.2 |
| R4 | all gains disappear against compute-matched best-of-N | section 6, both accountings | see 9.2 |

Note that R1 and criterion 1 are **not** complements here: the pilot has two primary
properties and they answer oppositely, so both a pass on cLogP and a rejection-style
finding on aromatic rings can be true at once. That is the substance of criterion 3,
which README says "is sufficient even if the optimization gain is modest".

### 9.2 Verdict

**Passes if any of:**

| # | Criterion | Verdict |
| --- | --- | --- |
| 1 | head beats prefix statistics substantially | **partial pass** -- clears the pre-registered gate on cLogP at 50k (+0.073 nats [+0.065, +0.081], +0.043 AUROC [+0.034, +0.053]) and on the diagnostic control; **fails on aromatic rings with the opposite sign** (-0.141 nats). "Substantially" is generous for a 0.073-nat gain; it is robust, not large. |
| 2 | guidance improves hit rate at matched compute without validity collapse | **fails.** Validity never collapsed (0.993-0.999), but guidance loses to compute-matched best-of-N by -0.3488 (cLogP) and -0.3312 (rings) under `actual`, and by -0.7355 / -0.5291 under the accounting the specification prescribes. |
| 3 | the two primary properties show clearly different, reproducible predictability **or** intervention-response timing | **passes, on both axes.** Predictability: the curves cross for rings and never for cLogP, replicated at 10k and 50k across three independent estimators (head NLL/AUROC by quartile, rollout rank correlation, and the head-vs-trivial bootstrap). Timing: cLogP retains 10% of its full effect in the late window against rings' 22%, with cLogP's late lift the only one in either property inside its own seed spread. |

**Rejects or radically reframes if any of:**

| # | Rejection condition | Verdict |
| --- | --- | --- |
| R1 | head adds nothing beyond length and token/atom counts | **fires for aromatic rings, not for cLogP.** For rings the trivial counter wins at every scale and pulls further ahead late (Q4: 0.978 against 0.872 AUROC). For cLogP the frozen state leads at every quartile on every estimator. |
| R2 | guidance changes the property only through sequence or molecule size | **does not fire.** 97% (cLogP) and 92% (rings) of the effect survives joint matching on length *and* heavy-atom count at 0.98-1.00 coverage. |
| R3 | timing curves flat and indistinguishable across the two properties | **does not fire.** The curves differ in shape (cLogP flat-then-collapse, rings peaked-then-partial) and in late-window retention (10% against 22%). |
| R4 | all gains disappear against compute-matched best-of-N | **fires, for both properties, under both accountings.** |

### 9.2.1 Optional extension — one data-aggregation round

Run **once**, after the main pilot, as the specification permits, and reported as an
extension rather than as part of the pilot. No reinforcement learning, no iteration.
2,000 guided molecules generated with the original head, 7,976 prefixes extracted with
their terminal cLogP, given their own grouped split, mixed into the head's training
data, head retrained once, guided decoding re-tested. Total 2,546 s.

**The distribution shift it was meant to expose is large.** On the guided prefixes the
decoder actually visits, the original head predicts a mean target probability of
**0.076 against an observed rate of 0.267** — under-confident by a factor of 3.5,
ECE 0.190, AUROC 0.651. The head was trained on base-policy prefixes and is being asked
to score prefixes the base policy rarely produces.

| | mean predicted | observed | ECE | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| original head on guided prefixes | 0.076 | 0.267 | 0.190 | 0.651 | 0.225 |
| retrained head on guided prefixes | 0.091 | 0.267 | 0.176 | 0.680 | 0.214 |

**One round barely fixes the calibration** (ECE 0.190 to 0.176; predicted 0.076 to
0.091 against a target of 0.267) but does modestly improve decoding:

| arm | hit rate | seeds | abs. target error | validity |
| --- | ---: | --- | ---: | ---: |
| original head | 0.2477 +- 0.0175 | 0.2682 / 0.2255 / 0.2495 | 0.734 | 0.9935 |
| retrained head | **0.2782 +- 0.0068** | 0.2824 / 0.2686 / 0.2838 | **0.638** | 0.9967 |

**+0.0305 hit rate, about 2.3 standard errors** — suggestive, not established, on three
seeds. Both arms must be compared *within* this script: the original-head arm here
(0.2477) differs from the main run's `throughout` (0.2645) despite identical seeds,
because 2,000 aggregation molecules are generated first in the same process and shift
global RNG state. The within-script contrast is the valid one.

**It changes no conclusion.** Even taking +0.0305 at face value, guided decoding moves
from 0.2645 to roughly 0.295, against compute-matched best-of-9 at **0.6133** — and the
aggregation round itself cost 814,617 additional processed tokens, which is not in that
budget. The gap to the baseline is an order of magnitude larger than the gain, and
closing it by this route would require a different method, not more rounds. Consistent
with the specification's instruction, it was run once and is reported as-is.

The specification says outcome 3 "is sufficient even if the optimization gain is
modest". Outcome 3 holds and outcome 2 fails, so the honest summary is:

**The pilot succeeds as a measurement and fails as an optimisation method.**

1. **A frozen autoregressive SMILES model does carry information about a future
   property in its hidden state, for some properties.** For cLogP the frozen state
   beats explicit prefix statistics at every generation position, on three independent
   estimators, at two dataset sizes. For aromatic ring count it does not, and more data
   does not help — counting `c` tokens is simply better, decisively so once the count
   is nearly determined. Any claim that "LM hidden states encode future molecular
   properties" needs the property named.

2. **Predictability and controllability are different quantities, and they dissociate
   in both directions.** cLogP is predicted best late and steered worst late. Aromatic
   rings are predicted worse than a token counter yet steered best, including late.
   Knowing what a molecule will become is not the same as being able to change it. This
   is the pilot's most defensible contribution and the one that would survive
   replication most easily, because it appears in the base model's own conditional
   distributions (section 5.6) and not only in a learned head.

3. **As a way to hit a property target, top-8 guidance at lambda = 1 is dominated by
   rejection sampling.** Best-of-N wins by roughly 0.33-0.35 at equal tokens and equal
   wall time, and by 0.53-0.74 under the specification's own accounting. This is not
   close, and it does not depend on the rounding of N.

### 9.4 What would and would not follow from this

**Not shown:** that guidance is useless in general. The comparison that defeats it
assumes property evaluation on a *completed* molecule is nearly free, which is true of
RDKit descriptors and false of docking, DFT or an assay. In an expensive-oracle regime
the accounting changes completely and best-of-N's advantage is not transferable. This
pilot did not test that regime and cannot speak to it.

**Not shown:** that the frozen state lacks aromatic-ring information. Only one probe
was tried — a two-layer MLP on the final layer. The negative result is about this
readout, not about the representation (section 8.3).

**Weakened by:** the cLogP head being under-confident (mean predicted 0.049 against a
0.102 base rate). Guidance scores with `log P(y in I | prefix + a)`, so a compressed
probability range means a weaker effective lambda, and the intervention results are
properly read as "at lambda = 1 with an uncalibrated head". A lambda sweep and a
calibrated head are the two cheapest experiments that could change the section 6
numbers; neither could plausibly close a 0.33 gap to best-of-N.

**Most robust single claim:** the crossover in section 5.6 — the frozen state's rank
correlation with the *base model's own* conditional mean for aromatic rings falls from
0.866 at Q2 to 0.799 at Q4 while a token counter rises from 0.846 to 0.921. That
statement involves no head at generation time, no guidance, no compute matching, and no
choice of lambda.

---

## 10. Result 4 — is the property bought with degenerate molecules?

Added 2026-07-30, after the five specified phases. `scripts/10_quality_analysis.py`,
`outputs/pilot_50k_quality_{clogp,aromatic_rings}/`. No new generation: this re-scores
the molecules already saved by script 05.

Hit rate is blind to *how* the target was reached. The documented failure mode in
property optimisation is a controller that satisfies the objective degenerately, and the
literature is explicit about this property in particular: the PMO benchmark
(Gao et al., NeurIPS 2022) excludes logP as a task because "adding carbons monotonically
increases the estimated LogP value", and MolDQN (Zhou et al., 2019) reports that
penalised-logP optimisation yields molecules that are "obviously not drug-like".

So every saved molecule was scored on synthetic accessibility (SA), drug-likeness (QED),
longest acyclic carbon path, carbon fraction, maximum ring size, fragment count and
formal charge.

**The comparison is restricted to molecules that hit the target.** Comparing all guided
molecules against all unguided molecules would confound quality with the property shift
itself, since oilier molecules are legitimately greasier and more flexible. Restricting
to hits holds the achieved property roughly fixed and asks only how the molecule got
there. The `unguided` hits row is also **the baseline's own output**: compute-matched
best-of-N returns base-policy samples selected for target proximity, so base-policy hits
are exactly the molecules section 6.4's baseline hands you.

### 10.1 cLogP, molecules hitting [4.173, 5.038)

| condition | n | SA | QED | longest chain | max ring | any degeneracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unguided (= best-of-N output) | 163 | 2.793 | 0.610 | 2.276 | 6.147 | 0.031 |
| throughout | 405 | 2.706 | 0.644 | 2.388 | 6.030 | 0.037 |
| early | 245 | 2.756 | 0.624 | 2.355 | 6.033 | 0.029 |
| middle | 251 | 2.782 | 0.642 | 2.359 | 6.076 | 0.044 |
| late | 187 | 2.790 | 0.647 | 2.417 | 6.005 | 0.032 |
| truncation_control | 138 | 2.722 | 0.643 | 2.406 | 6.014 | 0.022 |

No SA difference against base-policy hits excludes zero. QED is *higher* under guidance
(`throughout` +0.033, 95% bootstrap CI [+0.007, +0.062]). Degeneracy rates sit near 3%
in every condition and none differs from base.

### 10.2 Aromatic rings, molecules hitting [3, 4)

| condition | n | SA | QED | longest chain | max ring | any degeneracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unguided (= best-of-N output) | 261 | 2.739 | 0.597 | 2.015 | 6.008 | 0.015 |
| throughout | 718 | 2.711 | 0.625 | 1.943 | 6.007 | 0.019 |
| early | 437 | 2.738 | 0.608 | 1.973 | 6.000 | 0.011 |
| middle | 463 | 2.767 | 0.606 | 2.050 | 6.037 | 0.024 |
| late | 360 | 2.882 | 0.589 | 2.214 | 6.058 | 0.022 |
| truncation_control | 280 | 2.784 | 0.609 | 2.121 | 6.014 | 0.007 |

**Late guidance is the one condition that pays a quality cost**, and both signals point
the same way: SA +0.143 [+0.055, +0.236] and longest acyclic chain +0.199 [+0.011,
+0.412] against base-policy hits, each excluding zero. Chemically this is what the
position story predicts — adding an aromatic ring to an already-committed scaffold
requires tacking something awkward onto a nearly finished molecule. Early and
`throughout` guidance costs nothing measurable.

### 10.3 Why the expected degeneracy does not appear, and what that does not license

Two structural reasons, both of which bound the claim:

1. The objective is a **bounded interval** the base model already reaches ~11% (cLogP)
   and ~17% (rings) of the time, not unbounded maximisation. Overshooting is penalised
   exactly like undershooting, so "add carbons" is a losing strategy under this
   objective rather than a winning one. PMO's objection is to maximisation; this is
   interpolation.
2. At lambda = 1, `log p_base` keeps full weight in the score, so the base likelihood is
   tilted rather than overridden.

**This is therefore evidence about the interpolative, lambda = 1 regime only.** A larger
lambda or an unbounded objective is exactly where the PMO and MolDQN degeneracies should
appear, and we have not tested it. Any lambda sweep must re-run this analysis at every
lambda; that is a prediction, not a formality.

### 10.4 What "valid" admits

One molecule from the `late` aromatic-ring condition, quoted because it shows the limit
of the validity metric used throughout this report:

```
C.C.C.C=CC=c1[n-]c(CC(CC)CCC)c(CC(C)C)c1=CC.[CH2-]C[CH2-]
```

RDKit parses it, so it counts as valid. It contains three aromatic rings, so it counts as
a target hit. It is four loose methanes beside a charged ring and two carbanions. It is
now caught by the `multi_fragment` and `net_charged` flags and by a regression test, but
it was counted as a success everywhere in sections 6 and 6.4, and RDKit-parseability is
the standard validity check in this literature. Aggregate validity of 0.99 does not mean
what it appears to mean.

### 10.5 Effect on the report's conclusions

None of section 6's or 6.4's numbers change. Rejection criterion R3 (validity or
uniqueness collapse) still does not fire, and now does not fire on a considerably wider
panel. The negative result in section 6.4 stands and is not rescued: guidance does not
buy its 0.33 deficit back in molecule quality, because at lambda = 1 there is no quality
difference to buy it back with.
