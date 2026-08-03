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
corrupted, and what was re-run. **The pre-registered scaling gate that authorised the
move from 10k to 50k trajectories FAILED as written** — cLogP cleared the AUROC criterion
(+0.042 ≥ 0.02) but its NLL gain of +0.042 nats fell short of the 0.05-nat threshold, and
aromatic rings failed with the opposite sign. Scaling proceeded anyway, on the
specification's looser wording, and is recorded as a deviation in §7. §5.3 has the full
reasoning; it is stated here because a gate that fired against the project should not be
reachable only by reading to §5.3.

**Phase 2 and the follow-ups.** Sections 11–19 extend the battery to six properties on an
independent 50,000-molecule sample, measure the steering headroom directly, and sweep λ.
Sections 20 and 21 close the three cheapest explanations for the negative result, one at a
time: the head is not fixable by post-hoc calibration (§20.3 — correcting it is
algebraically a λ *decrease*), not by 6.9x more readout capacity or a differently-shaped
readout (§20.4), and not by reading a better probe point (§21). Section 21 does overturn one
of the report's own claims: the aromatic-ring crossover of §13.1 is a fact about the final
layer, not about the representation, and the middle of the network predicts ring count
better than counting tokens does. It does not change the steering result.

**Sections 22 and beyond, added 2026-08-03.** Eleven further experiments (C23–C33) were run
after §21 and were written up as standalone sections under `reports/section_c*.md`. Each
filed its disagreements with this report rather than editing it. **The merges are §22
(C23–C29), §23 (C30), §24 (C31, C32) and §25 (C33)**: each lists every claim above that the
new work changed, states the replacement, and says which of this report's own headlines no
longer stand. **Read all four before quoting any number from §16, §19 or §20** — several of
them supersede each other, and §25 supersedes a headline §22 introduced.

Two headlines above do not stand. Guidance **does** have a compute knob (C28). And
guidance's loss to best-of-N shrinks sharply once the comparator is denied a free oracle
(C27) — but the "roughly 8×" that §22.1.1 attached to that shrinkage is a **GP-MoLFormer,
single-budget number**, and it **failed its pre-registered replication on a second
generator** (C33, §25). The portable form is the matched-N curve of §25.2, not any ratio.

**Two numbering systems, separated 2026-08-03.** Claims from `reports/ABSTRACT.md`'s
inventory are **`CL<n>`**; experiments from `docs/TODO.md`'s register are bare **`C<n>`**.
They both used to be `C<n>` and collided on C30–C33 — see the note at §22.11.

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

> **Scope correction, 2026-08-03.** This section describes **phase 1 only**. It said "No
> GPU was used" without qualification, and that has been false since phase 2 moved to an
> RTX 4090 — §11.1 records the move, the device-equivalence experiment it forced, and
> claim CL11. Every phase-1 number below is CPU; every phase-2 number, and every number in
> §20, §21 and §22, is CUDA. The sentence is corrected in place rather than deleted,
> because the phase-1 throughput figures that follow it are CPU figures and are still the
> figures those runs were produced at.

Apple M-series (arm64, 12 cores, 24 GB RAM), macOS 26.5.2 (Darwin 25.5.0), Python
3.12.8, numpy 1.26.4. The full software stack is recorded in
`outputs/provenance.json`. **No GPU was used *in phase 1*.** CUDA
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
  **Corrected by §21 (C17): "a learned readout of a 768-dimensional state" here means a
  readout of the *final* layer's state. Section 21.3 measures all 13 probe points and finds
  the same readout at probe points 1–5 beats the trivial counter (0.8474 at probe point 3
  against trivial 0.8269). The sentence is true of layer 12 and false of the middle of the
  network.**

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

**Re-scoped by §21 (C17), and this is a correction to the claim, not a caveat on it.** The
kill-test is met *at the final layer*. Section 21.4.1 sweeps all 13 probe points under a
pre-registered rule and returns **ARTEFACT**: probe point 3 reaches 0.8474 target AUROC
against the trivial counter's 0.8269, a +0.0205 margin whose Bonferroni-corrected bootstrap
CI is [+0.0142, +0.0284], with NLL also better and both neighbouring layers supporting it.
The correct statement is the weaker one: *the model's final layer does not linearly expose
aromatic ring count as well as counting `c` and ring-closure tokens does, while its middle
layers do.* The measurement was right; the scope of the conclusion drawn from it was not.
What is **not** rescued is the steering result — §21.5 finds the better-predicting layer
does not steer better (2 of 6 properties, median relative −0.077, against a pre-registered
bar of 4 of 6 and +0.25).

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

**Retired by §20 (C18). This diagnosis is wrong in direction, not merely in magnitude,
and the "a null would not be evidence against the method" escape clause does not stand.**
Three corrections, all measured:

1. **The 3.5x factor is 1.69x on the fixed head** (§20.2), re-measured on an independent
   sample of guided prefixes. §11.6 argued about half of the original factor was the
   interval-mask defect and its arithmetic implies 3.5 / 2.08 = 1.68. On *base-policy*
   prefixes every head in the six-property battery is essentially calibrated (predicted /
   observed ratios 0.85–1.11, ECE 0.0045–0.0143).
2. **Under-confidence is a level error, and the decoder is invariant to levels.** Guidance
   samples from `softmax(log p_base + λ·log q)` over eight candidates, which depends on the
   *differences* of `log q`. Post-hoc calibration cut ECE by a factor of 3–6 and left AUROC
   **bit-identical to four decimal places** for all six properties (§20.3), because a
   monotone map preserves rank.
3. **Correcting the under-confidence is a λ *decrease*, so it makes guidance worse.** A
   power map `g(q) = c·q^α` is exactly a rescale `λ_eff = λα`, and the fitted Platt slopes
   are 0.405–0.618 — all below 1. Measured end to end, Platt costs 0.23–0.54x and isotonic
   0.41–0.70x of the deployed lift at every anchor (§20.5). At ε = 0 a power-calibrated head
   at λ=1 and the raw head at λ=α return **the same 1,536 molecules** (§20.3.1). So the
   arrow in the sentence above — under-confidence ⟹ weak effective λ ⟹ fixing it helps —
   points the wrong way: fixing it *is* a weak λ.

**8.3 Only one probe point was tested.** All results use the final layer
(`hidden_layer: -1`) and one head architecture (two-layer MLP, 256 units). Whether an
earlier layer, a larger head, or a different pooling would close the aromatic-ring gap
is untested. This is a genuine limit on the negative result in 8.1: it shows *this*
readout fails, not that the information is absent from the state.

**Two of the three suspects have since been tested, and they answer in opposite
directions.** *An earlier layer:* **yes, it closes the gap** — §21.3 sweeps all 13 probe
points and the aromatic-ring crossover reverses at probe points 1–5. *A larger head:* **no,
it does not** — §20.4 trains a readout with a 4× wider hidden layer (256 → 1024), which is
6.9x the parameters (1.84M against 268k), at the
final layer and moves seed-matched target AUROC by at most +0.0019 on any property, with
aromatic rings still at 0.7886 against trivial 0.8267 (the single-seed value; §13.1's
0.8269 is the three-seed mean of the same quantity — see §21.5.3). A fourth suspect not on the original
list, a readout focused on exactly the three-bin event guidance consumes, is **worse on all
six properties**. **A different pooling remains untested** — every head in this report,
baseline and variant, reads the single hidden state at one position. So does a **capacity ×
depth interaction**: capacity is excluded at layer 12, and depth is measured with the
256-unit head, but a wider head at probe point 3 has not been tried.

The two results together are stronger than either alone: the gap at layer 12 is not a
capacity limit, and the information is present five layers earlier. What neither result does
is rescue the *steering* claim — see §21.5.

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

---

# Phase 2 — the lexical-locality test

Everything from section 11 onward was executed after the pilot, on different hardware,
to test a **different hypothesis**: that steerability tracks the *lexical locality* of a
property — how directly the property is written into the SMILES string — rather than its
predictability from the frozen model's hidden state. The hypothesis, its
operationalisation and its pre-registered predictions P1–P6 are in
`docs/LEXICAL_LOCALITY.md`, committed before any phase-2 measurement.

Sections 1–10 are unchanged. Where phase 2 revises something, it says so explicitly.

## 11. Phase-2 setup, and two reproducibility findings

### 11.1 Hardware, and what moving to a GPU does and does not change

| item | phase 1 | phase 2 |
| --- | --- | --- |
| device | Apple M-series CPU, `torch.set_num_threads(10)` | **NVIDIA RTX 4090**, CUDA 13.2, driver 595.58.03 |
| torch | 2.4.1 (CPU) | 2.4.1+cu121 |
| python / transformers / RDKit | 3.12.8 / 4.44.2 / 2024.3.5 | 3.12.3 / 4.44.2 / 2024.03.5 |
| `dtype` | float32 | float32, **deliberately unchanged** |
| model revision | `6eca8795…` | `6eca8795…`, identical |

`dtype` stays float32 because the cached-versus-full candidate-backend agreement is a
*numerical equality* claim and the whole compute accounting of section 6.4 rests on it
(`test_candidate_backends_agree`). Reduced precision is therefore not a free change, and
was not made.

**All 14 model contracts pass on CUDA**, including the two that matter:
`test_forward_pass_is_deterministic`, which underwrites reproducibility, and
`test_candidate_backends_agree`, which underwrites the compute accounting.

Three defects surfaced in the port, and it is worth separating them because only one was
in the library:

1. Two tests called `.numpy()` on what is now a CUDA tensor. Device-portability bugs in
   the *tests*, fixed to `.cpu().numpy()`.
2. `guidance.TargetScorer` held a head loaded with `map_location="cpu"` and was handed
   CUDA candidate hidden states — a genuine library bug that would have crashed every
   guided run on a GPU. It now migrates the head and the interval mask to the device the
   states arrive on.
3. The one in section 11.2, which is not a bug.

### 11.2 The frozen target intervals could not be re-derived — setup gate (d) failed

The phase-2 kickoff required regenerating `pilot_50k` and diffing the regenerated
`windows.json` and `target_intervals.json` against the tracked ones, with the
instruction to stop and write it up if they moved. **They moved.** This section is that
writeup.

`outputs/pilot_50k_gate_d/`, `outputs/device_equivalence/device_equivalence.json`.

**Windows did not move.** `t33 = 15`, `t67 = 29`, identical to the digit. They are
quantiles of the pooled distribution of generated token *positions*, which is a
statistic over ~2.1 M positions and is correspondingly stable.

**Target intervals did move**, all three:

| property | frozen (phase 1) | re-derived (phase 2 GPU) |
| --- | --- | --- |
| cLogP | [4.172676, 5.038300) | [4.138940, 4.997184) |
| aromatic rings | [3, 4), base rate 0.171387 | [3, 4), base rate 0.168650 |
| mol. weight (diagnostic) | [435.4494, 486.9970) | [435.5024, 486.9832) |

#### What caused it

Not numerics. `scripts/13_device_equivalence.py` loads the same pinned revision on both
devices and compares them directly:

| check | result |
| --- | --- |
| weight checksum identical | **yes** (46,781,184 parameters, identical `parameter_sum`) |
| max abs. logit difference, CPU vs CUDA | **1.359e-05** |
| **top-8 candidate set identical on every test prefix** | **yes** |
| same seed draws the same molecules | **no — 0 of 64 identical** |
| same device, same seed, reproduces itself | **yes** |

The forward pass agrees to float32 tolerance, and — the check that actually matters —
**the top-8 candidate set guided decoding consumes is identical on both devices.** So the
numerical risk to the method under test is bounded at essentially zero.

The cause is the random-number stream. `torch.manual_seed` seeds the CPU Mersenne
Twister and the CUDA Philox generator both, but `torch.multinomial` draws from whichever
generator owns the tensor. On CUDA it therefore makes different choices at the same seed.
**The base policy pi_0 is unchanged; the 50,000-molecule *sample* is a different draw
from it.**

Every distributional statistic confirms that reading. Over 49,823 (CPU) against 49,825
(GPU) kept trajectories:

| quantity | CPU | GPU | z |
| --- | ---: | ---: | ---: |
| cLogP mean | 2.7256 | 2.7145 | −0.99 |
| aromatic rings mean | 1.8149 | 1.8093 | −0.83 |
| mol. weight mean | 366.451 | 366.650 | +0.33 |
| content length mean | 42.252 | 42.271 | +0.26 |
| validity | 0.99658 | 0.99656 | — |
| uniqueness | 0.99996 | 0.99998 | — |
| P(exactly 3 aromatic rings) | 0.17139 | 0.16865 | −1.14 |

No mean differs by more than one standard error of the difference. The intervals moved
because they are **empirical quantiles of a finite sample**: the 85th percentile of cLogP
over 50,000 draws has a standard error near 0.012, and the two independent estimates
differ by 0.034, about two standard errors of their difference.

One apparent anomaly, resolved: molecular-weight standard deviation moved from 87.45 to
102.92, which looks like far more than sampling noise. It is one molecule — a 12,964 Da
polyiodide chain. Excluding the ten heaviest molecules the GPU sample's sd is 84.12, and
the 95th percentiles agree to 0.014 Da. The sd is simply not a robust summary of this
distribution's tail. The molecule is also a second exhibit for section 10.4's point about
what RDKit-parseability admits:

```
IC(I)I.II(I)I(I)I.III.IIIIII.III(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)I(I)…
```

#### Why this is a finding and not an inconvenience

Because **the pilot's own base sample is not recoverable.** `trajectories.json` is
excluded from git by design (it is ~700 MB and was documented as deterministically
reproducible from the pinned revision and recorded seeds). That documented guarantee is
now known to be **device-conditional**: the arrays regenerate exactly on the *same* class
of device and not across devices. The pilot's 49,823 molecules exist only in the record
of statistics computed from them.

This bounds a claim made in `docs/REPRODUCE.md` and in `.gitignore`, and the honest
correction is: *the excluded arrays are reproducible from the pinned revision and
recorded seeds **on hardware whose RNG stream matches the original run's**.* For a
different device the regenerated dataset is a fresh draw from the same policy, which is
adequate for training a head and inadequate for reconstituting a frozen interval.

#### What phase 2 does about it

Two options were available: re-derive the intervals from the phase-2 sample, which
breaks comparability with every phase-1 number; or inherit them. Phase 2 **inherits**,
because "frozen before any guided result was inspected" is the property that makes
sections 5 and 6 interpretable at all, and re-deriving would silently unfreeze it.

`scripts/02_generate_trajectories.py` gained `--inherit-intervals`: any property present
in the referenced file is copied **verbatim**, and only new properties are derived from
the current run's base distribution. What the current run *would* have derived is written
alongside, in `target_intervals_provenance.json`, so the size of the divergence is on
disk rather than only in this prose.

The consequence, stated plainly: **the phase-2 dataset is a different 50,000-molecule
sample from the phase-1 dataset**, and phase-2 heads are trained on it. For cLogP and
aromatic rings this makes phase 2 an *independent replication* of the head training
rather than a continuation of it. It also means the two datasets must not be mixed —
row indices in `prefix_meta.csv` refer to different molecules — so the phase-2 sample
lives in its own directory, `outputs/pilot_50k_p2/`, and **no tracked phase-1 artefact is
overwritten anywhere in phase 2.**

### 11.3 Reproducing a known number across the device change — setup gate (e)

`outputs/pilot_50k_gpucheck_guided_aromatic_rings/`. Bit-identical reproduction is
impossible for the reason just established, so the comparison is distributional, over
all three seeds and all 512 molecules per seed.

| condition | CPU hits / n | GPU hits / n | CPU rate | GPU rate | z |
| --- | ---: | ---: | ---: | ---: | ---: |
| unguided | 261 / 1528 | 273 / 1529 | 0.1708 | 0.1785 | 0.56 |
| guided `throughout` | 718 / 1525 | 752 / 1532 | 0.4708 | 0.4909 | 1.11 |

And the quantity the pilot's claim is actually about:

| | guidance effect (`throughout` − `unguided`) | binomial SE |
| --- | --- | --- |
| phase 1, CPU | **+0.3001** | ± 0.0160 |
| phase 2, GPU | **+0.3123** | ± 0.0161 |

Neither condition differs significantly, and the effect sizes agree to well within one
standard error. Token accounting also agrees: 43.6 against 43.5 tokens per molecule
unguided, 405 against 415 guided. **Gate (e) passes**: the pilot's headline
intervention-response result reproduces across the device change.

One incidental observation, since it bears on how the pilot's seed spreads should be read:
the GPU run's *seed-to-seed* standard deviation is much smaller (0.0059 on `throughout`,
0.0031 on `unguided`) than the CPU run's (0.0241 and 0.0209). Both are three-seed
estimates, so the sample standard deviation is itself extremely noisy at n = 3, and the
binomial expectation over 512 molecules is ~0.022 for a rate near 0.47. The CPU spreads
match that expectation and the GPU spreads came in low by luck. Nothing should be read into
either, and in particular the pilot's practice of quoting ± seed sd is a *loose* uncertainty
estimate rather than a tight one.

### 11.4 What *does* reproduce: determinism within a device

Two independent phase-2 regenerations of the 50,000-trajectory dataset, run separately
on the same GPU with the same seeds, agree **bit-for-bit** — including the 768-dimensional
hidden-state array:

| artefact | run 1 vs run 2 |
| --- | --- |
| kept trajectories / invalid / too short | 49,825 / 172 / 3, identical |
| validity, uniqueness, base distribution | identical |
| `hidden.npy` sha256 | identical |
| `features.npy` sha256 | identical |

So the precise statement replacing the pilot's blanket reproducibility claim is: **the
git-excluded arrays regenerate exactly from the pinned revision and recorded seeds on the
same class of device, and constitute a fresh draw from the same policy on a different
one.**

### 11.5 A second defect, found by moving devices: the target interval was not a union of bins

This one is not about hardware. It was found while working out how an *inherited* interval
would interact with a binner fitted on a different sample, and it turned out to affect the
pilot as executed. It is the same species of bug as section 8.7 — an interval-boundary
error that produces a plausible number rather than an exception.

#### The mechanism

The head predicts a categorical distribution over bins, and `P(y in I)` is read off as a
sum of bin probabilities. `binning.py`'s own docstring states the invariant that makes
this exact: *"Target intervals are therefore always defined on bin boundaries, which makes
that sum exact rather than an approximation."* Nothing enforced it.

`QuantileBinner.interval_mask` keeps only bins lying **wholly** inside `[lo, hi)`. So if a
target edge falls in the middle of a bin, that bin is dropped and the head's `P(y in I)`
becomes the probability of a strict **subset** of the target.

And the two quantities were computed from different samples. The target interval is a
quantile of **all** kept trajectories (`scripts/02`), while the binner is fitted on the
**train split's** prefix rows (`scripts/03`). They agree closely and never exactly.

#### What it did to the pilot, read off the committed checkpoints

`outputs/pilot_50k_heads/head_*_frozen_state.pt` carry their binners, so this is
measurable now without re-running anything:

| property | binner | target | bins selected | mask covers | true base rate |
| --- | --- | --- | ---: | ---: | ---: |
| **cLogP** | 20 quantile bins | [4.172676, 5.038300) | **1 of 20** | **~0.050** | **0.100** |
| aromatic rings | 6 categories | [3, 4) | 1 of 6 | 0.171 | 0.171 |
| mol. weight | 20 quantile bins | [435.4494, 486.9970) | 2 of 20 | 0.100 | 0.100 |

The cLogP binner's edges nearest the interval are **4.174** and **5.04028**, both just
*above* the interval's own edges — by 1.324e-03 and 1.980e-03. Bin 17, `[4.174, 4.5153)`,
therefore sits wholly inside and is kept; bin 18, `[4.5153, 5.04028)`, extends 0.002 past
`hi` and is dropped. **The cLogP head was trained and used to predict a 0.050-mass event
while the target it was steering toward had base rate 0.100.**

Aromatic rings escaped entirely: `CategoricalBinner` represents an integer target `[v, v+1)`
as exactly one category, so the invariant holds by construction. Molecular weight escaped
by luck — both its nearby edges happen to fall *inside* its interval.

#### This explains a result the pilot recorded as a calibration failure

Section 5.5 reports, of cLogP: *"both heads are systematically under-confident: mean
predicted target probability 0.049 against a 0.102 base rate (ECE 0.053), and the
reliability curve has no support above predicted 0.3."* Section 8.2 calls this "the head is
under-confident for cLogP, and much more so where it is used."

0.049 against 0.102 is a factor of 2.08. The mask covers one of the two bins the target
spans. **That is not a miscalibrated head; it is a correctly calibrated head answering a
different question** — and the factor is not approximately 2 by coincidence.

The same arithmetic partly accounts for section 9.2.1's off-policy figure, 0.076 predicted
against 0.267 observed on guided prefixes. A factor of ~2 of that 3.5 is this defect. The
remainder is genuine distribution shift, so section 9.2.1's conclusion survives in reduced
form: the head *is* miscalibrated off-policy, by appreciably less than the report claims.

#### What it did and did not affect

Following the same discipline as section 8.7 — what was verified, not what was assumed:

- **Every hit rate is unaffected**, everywhere. Hit rate is `lo <= v < hi` computed by
  RDKit on the *completed* molecule and never touches the binner. Sections 6.1, 6.2, 6.3,
  6.4 and 10 stand exactly as reported.
- **Aromatic rings is unaffected in every metric**, by construction. That includes the
  crossover of section 5.6, the pilot's most-defended claim, and the whole of section 6.2.
- **Molecular weight is unaffected**, by luck rather than design.
- **cLogP's reported target-interval AUROC and Brier were computed from the wrong score.**
  What was ranked is `P(y in bin 17)`, not `P(y in [lo, hi))`. Which direction that biases
  a rank statistic is not obvious from the argument, so it was measured rather than
  asserted — section 11.6, where it turns out to be **negligible** (+0.0004 AUROC). The
  calibration damage, by contrast, is the whole of the pilot's reported under-confidence.
- **cLogP's NLL, expected-value MAE, and the frozen-versus-trivial bootstrap are
  unaffected.** They are computed over the full bin distribution and never use the mask.
  So the scaling gate, which is decided on NLL and AUROC gain, keeps its verdict: the NLL
  leg is untouched, and the AUROC leg was measured too low rather than too high.
- **cLogP guided decoding was steering toward the lower half of its target band.** The
  guidance score is `log p_base + lambda log P(y in I | prefix + a)`, and the quantity
  supplied was `P(y in [4.174, 4.5153))`. Guidance was therefore aimed at a narrower,
  lower sub-band than the one it was scored against — a *handicap*, and one that makes
  cLogP's measured lift of +0.158 a lower bound on what the intended objective would give.
- **The section 6.4 negative result is not rescued by this.** It cuts the right way for
  guidance and the margin is 0.35–0.74; the phase-2 cLogP run in section 16 uses a correct
  mask against a compute-matched baseline, so how much of the gap it closes is measured
  rather than argued.

#### The fix, and why it cannot recur

`QuantileBinner.fit` takes `extra_edges`, and `scripts/03_train_heads.py` passes the
target interval's `(lo, hi)`, so the interval is exactly a union of bins by construction.
The invariant is then **verified numerically rather than reasoned about**:
`binning.interval_mask_coverage` bins the held-out values, sums the mask, compares against
the empirical rate, and script 03 **exits non-zero** if they disagree. Every property's
coverage is recorded in `head_metrics.json`.

Eleven regression tests in `tests/test_binning.py::TestTargetIntervalMustBeAUnionOfBins`
reproduce the defect at the pilot's own offsets (−1.324e-03, −1.980e-03), confirm the
one-bin outcome, and assert the fix. As with section 8.7's regression test, they fail
against the old code.

### 11.6 What the interval-mask defect actually cost — measured

`outputs/interval_mask_impact/interval_mask_impact.json`,
`scripts/14_interval_mask_impact.py`. Two head-training runs on the **same** phase-2
sample with the **same** initialisation seed and the same recipe, differing only in whether
the target interval's edges are forced to be bin boundaries. The second reproduces the
pilot's behaviour.

| property | bins selected fixed / legacy | mask covers, legacy / true | target AUROC fixed / legacy | mean predicted fixed / legacy | base rate | ECE fixed / legacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **cLogP** | 3 / **1** | **0.0465 / 0.0929** | 0.7901 / 0.7896 | 0.0943 / **0.0494** | 0.0929 | **0.0045 / 0.0437** |
| **mol. weight** | 3 / **1** | 0.0469 / 0.1006 | 0.8299 / 0.8223 | 0.0957 / 0.0469 | 0.1006 | 0.0058 / 0.0537 |
| **TPSA** | 3 / **1** | 0.0488 / 0.0977 | 0.7391 / 0.7350 | 0.0955 / 0.0509 | 0.0977 | 0.0074 / 0.0468 |
| **QED** | 3 / **1** | 0.0500 / 0.0990 | 0.7355 / 0.7328 | 0.0907 / 0.0483 | 0.0990 | 0.0085 / 0.0507 |
| aromatic rings | 1 / 1 | 0.1587 / 0.1587 | 0.7904 / 0.7904 | 0.1506 / 0.1506 | 0.1587 | 0.0143 / 0.0143 |
| HBD count | 1 / 1 | 0.0825 / 0.0825 | 0.7781 / 0.7781 | 0.0746 / 0.0746 | 0.0825 | 0.0087 / 0.0087 |
| rotatable bonds | 1 / 1 | 0.0803 / 0.0803 | 0.7806 / 0.7806 | 0.0940 / 0.0940 | 0.0803 | 0.0137 / 0.0137 |

Four findings, and two of them correct what section 11.5 first said.

**1. It was not bad luck; it is structural for every continuous property.** All four
quantile-band targets lose exactly one of their two bins, and the legacy mask therefore
covers about half the target in each case: cLogP **0.500**, TPSA **0.500**, QED **0.505**,
molecular weight **0.466**. (Not exactly half everywhere, because the bins are equal-mass on
the *train* split the binner was fitted to and these rates are measured on the test split.)
With 20 equal-mass bins a `[0.85, 0.95)` band spans exactly two, so any misalignment at all
drops one. All three count properties are immune, as `CategoricalBinner` guarantees. Had the
pilot used TPSA or QED it would have hit the same defect; had it used only counts it would
never have seen it.

**2. The damage is to calibration, and it is almost the entire reported effect.** For
cLogP the fixed head predicts a mean target probability of **0.0943** against a base rate
of **0.0929** — a ratio of **1.014**, essentially perfectly calibrated on-policy — while
the legacy head predicts **0.0494**, a ratio of **0.531**. ECE falls from 0.0437 to
**0.0045**, a factor of 9.8.

Section 5.5 reported "mean predicted target probability 0.049 against a 0.102 base rate
(ECE 0.053)" and section 8.2 called it "the head is under-confident for cLogP". **That
diagnosis was wrong.** The head was not under-confident; it was correctly calibrated for
the event it had been given. The pilot's stated ECE of 0.053 and the phase-2 legacy
replication's 0.0437 are the same artefact measured twice.

**3. Discrimination barely moved, so the AUROCs stand.** cLogP's target AUROC changes by
**+0.0004** (0.7896 → 0.7901); the largest shift across the four affected properties is
mol. weight's +0.0076. A subset of the target event turns out to rank the target event
almost as well as the target event itself, which is unsurprising for a monotone
property but was not safe to assume. **Section 5.4's 0.797 and section 5.5's 0.663 → 0.883
quartile curve are therefore sound as reported**, and section 11.5's first draft of this
paragraph — which guessed they were underestimates — is corrected here.

**4. NLL is not a valid control, and the right control passes.** The two runs differ by
0.0307 nats on cLogP, but that is because the aligned binner has 22 bins against 20: a
finer partition has higher entropy, so a higher NLL is expected and means nothing. The
binning-invariant control is expected-value MAE, and it is **identical to five decimal
places** (0.91440 both) for cLogP, and within 0.006 for every property. So the two runs
really do differ only in the mask.

#### What this means for guided decoding, which is the part that matters

Less than it appears, and for a reason worth stating because it also corrects section 8.2's
reasoning. Section 8.2 argued that an under-confident head "produces a compressed score
range and a weaker effective lambda". That is not what a dropped bin does. The guidance
score is

```
score(a) = log p_base(a) + lambda * log( P(y in I | prefix + a) + eps )
```

and the next token is sampled from a **softmax over the eight candidates**. A softmax is
invariant to an additive constant. Scaling every candidate's probability by the same factor
is exactly an additive constant in log space, so a uniform 2x shrink would cancel
completely and change nothing at all.

The defect is therefore only felt through the part of the ratio that *varies* across
candidates — `P(bin 17 | a) / P(bin 17 or 18 | a)` is not constant in `a` — plus the `eps`
floor biting at smaller probabilities. Both are second-order. Section 16.3 measures the
guided hit rate under both heads on the same dataset and seeds, so this is settled by
measurement rather than by the argument just given.

### 11.7 Wall time on this machine

The pilot refused every timing claim because two bit-identical runs on its laptop differed
by 20–25% while token counts matched to the digit. Whether that band applies here is a
separate empirical question, and it was measured rather than assumed in either direction.

`scripts/13_device_equivalence.py` runs an identical 64-molecule sampling workload three
times: **0.79 / 0.81 / 0.84 s, a relative spread of 5.5%**, with processed token counts
identical across the three. So timing on this machine is roughly four times more
reproducible than on the pilot's laptop.

That licenses a *bounded* timing statement rather than a blanket refusal, and two caveats
keep it bounded. The measured workload is sub-second, so it constrains short-run jitter
and not thermal drift over hours — which is the mechanism the pilot blamed for its own
20–25% band. And a 5.5% band still cannot resolve the guided-versus-best-of-N margins of
section 6.4, which are smaller than that.

Where a timing ratio is quoted below it is because it is very large relative to 5.5%:

| stage | phase 1 (CPU) | phase 2 (GPU) | ratio |
| --- | ---: | ---: | ---: |
| 50k generation | 2,744 s | **239 s** | 11.5x |
| 50k hidden-state extraction | 1,301 s | **26 s** | 51x |
| guided `unguided` + `throughout`, 3 seeds | ~846 s | **72 s** | ~12x |

The hidden-state ratio is the largest because that stage runs the un-cached forward path,
whose released fallback implementation materialises a
`(batch, heads, length, features, dim)` tensor before its cumulative sum — arithmetic the
CPU was doing badly and the GPU does well.

**Processed tokens remain the reported cost unit throughout**, exactly as in phase 1. No
conclusion anywhere in this report rests on wall time.

## 12. The phase-2 design, and what was frozen before it ran

### 12.1 The hypothesis

Phase 1 found that predictability and steerability dissociate, and could not say why. The
reviewer recorded in `reports/ABSTRACT.md` proposed the alternative that phase 2 tests, and
it is a better hypothesis than ours, so it was adopted as the thing to test rather than
waited for in review: **steerability tracks the lexical locality of a property — how
directly the property is written into the SMILES string — not its predictability from the
frozen hidden state.**

Aromatic ring count is created by specific, discrete token events (a lowercase aromatic
atom plus a matching ring-closure digit), and its target interval is exactly one count unit
wide, so a single token choice moves the molecule a full target width. Crippen cLogP is a
sum of atom-type contributions over the whole molecule, and no single token moves it by the
0.87 log units its band spans. On that account the phase-1 dissociation is not a fact about
representations at all; it is a fact about *when a quantity becomes determined* plus a fact
about *lever size*.

### 12.2 What was pre-registered, and how it is held in place

`docs/LEXICAL_LOCALITY.md` was written before any phase-2 measurement and contains
predictions P1–P6. The retrodiction risk is obvious — the hypothesis exists to explain two
results we already had — so the ordering is pinned in code rather than prose:

- `properties.PREDICTED_LOCALITY_ORDER` is the predicted ranking, most local first:
  **aromatic rings, HBD count, rotatable bonds, TPSA, cLogP, QED**. It is asserted as a
  literal by `tests/test_properties.py::test_the_pre_registration_is_pinned_literally`, so
  editing it to match observed data fails a test rather than passing quietly.
- Target intervals come from a **rule**, not a hand-picked value, so they could be
  committed before anyone had seen the four new base distributions. Continuous properties
  get the `[0.85, 0.95)` quantile band; count properties get `[v, v+1)` for
  `v = round(quantile(0.90))`. Applying the count rule to aromatic rings returns 3 — the
  value the pilot chose by hand — so the uniform rule agrees with the pilot rather than
  overriding it.
- Estimator choices that could otherwise be tuned toward the predicted answer were fixed
  in `LEXICAL_LOCALITY.md` §3.1 before any headroom number existed: the permutation null
  for headroom's finite-sample bias, the primary locality and steerability scores, and the
  definition of headroom capture.

cLogP is **demoted, not dropped** (decision A7, `docs/TODO.md`): one property of six, out
of the title, and it does not carry the headline number. It is retained for continuity with
the executed pilot and because it is a clean stress test of the bounded-interval defence
against PMO's exclusion of logP.

### 12.3 The battery

Seven properties are computed; the six of `PREDICTED_LOCALITY_ORDER` are on the locality
scatter, and molecular weight remains a diagnostic control that is never a target.

| property | type | predicted locality | target interval | width | base rate |
| --- | --- | --- | --- | ---: | ---: |
| aromatic rings | count | **high** | [3, 4) | 1.0000 | 0.1714 |
| HBD count | count | **high** | [3, 4) | 1.0000 | 0.0823 |
| rotatable bonds | count | medium | [8, 9) | 1.0000 | 0.0827 |
| TPSA | continuous | medium | [100.5200, 119.3100) | 18.7900 | 0.1001 |
| cLogP | continuous | **low** | [4.172676, 5.038300) | 0.8656 | 0.1000 |
| QED | continuous | **low** | [0.8506, 0.8970) | 0.0463 | 0.1000 |

Two things about this table are worth stating rather than glossing.

**The four continuous properties are base-rate matched to each other by construction** —
all are `[0.85, 0.95)` bands, so all have base rate 0.100. The three counts are not,
because their rule positions a one-unit interval rather than fixing its mass, and a count
distribution is lumpy. HBD count and rotatable bonds land near 0.082 while aromatic rings
sits at 0.171. Base rate matters for both steerability (a rarer target leaves more room
above it) and for best-of-N (a commoner target is easier to hit in N draws), so section 15
reports a base-rate-adjusted effect size alongside the raw one, and the raw lift is not
the only number quoted.

**QED's band is only 0.0463 wide.** Normalising headroom by interval width, which is what
the hypothesis asks for, therefore gives QED a much smaller denominator than cLogP's 0.8656
or TPSA's 18.79. That is internally consistent — the question the normalisation asks is
"can one token move the property by a target width", and for QED a target width is a small
absolute change — but it means QED is the property most likely to behave unlike its
predicted rank for a reason that is about band geometry rather than lexical structure.
Flagged here, before the result, and revisited in section 15.4 — where it turns out to be
the mechanism that falsifies P1.

### 12.4 cLogP and aromatic rings are now measured on an independent sample

Because the phase-2 dataset is a different 50,000-molecule draw (section 11.2), phase 2 is
not a continuation of phase 1 for these two properties — it is an **independent
replication** of them, on a fresh sample, with head initialisation seeded (which the pilot
did not do) and with the section 11.5 defect fixed. That is a stronger position than a
continuation would have been, and section 13 uses it.

## 13. Result 5 — predictability across six properties

`outputs/pilot_50k_heads_p2/head_metrics.json`. Three heads per property share one training
recipe so only the input representation differs, exactly as in phase 1: `frozen_state` (the
768-dimensional hidden state, the method under test), `trivial` (a fixed,
property-agnostic vector of prefix token statistics), and `combined`. Only `frozen_state`
is usable at decode time, because guidance has to score a candidate hidden state that does
not correspond to any finished string.

**Three head seeds per cell** (1234 / 2345 / 3456), seeding initialisation as well as batch
shuffling. This closes the limitation recorded at §8.6, which the phase-1 paired bootstrap
could not address.

Held-out test split of the phase-2 sample; mean over three head seeds, ± standard deviation
across seeds:

| property | head | NLL | AUROC | E[y] MAE |
| --- | --- | ---: | ---: | ---: |
| aromatic rings | frozen state | 1.0445 ± 0.0041 | 0.7878 ± 0.0023 | 0.5844 |
| | **trivial prefix stats** | **0.9001** ± 0.0025 | **0.8269** ± 0.0019 | 0.5143 |
| HBD count | **frozen state** | **1.1071** ± 0.0016 | **0.7799** ± 0.0041 | 0.6258 |
| | trivial prefix stats | 1.2571 ± 0.0008 | 0.7098 ± 0.0014 | 0.7193 |
| rotatable bonds | **frozen state** | **1.8084** ± 0.0035 | **0.7820** ± 0.0012 | 1.3343 |
| | trivial prefix stats | 1.9154 ± 0.0010 | 0.7431 ± 0.0007 | 1.4527 |
| TPSA | frozen state | 2.7527 ± 0.0042 | 0.7369 ± 0.0023 | 17.1326 |
| | **trivial prefix stats** | **2.7014** ± 0.0002 | 0.7389 ± 0.0010 | 16.9993 |
| cLogP | **frozen state** | **2.7154** ± 0.0018 | **0.7913** ± 0.0012 | 0.9151 |
| | trivial prefix stats | 2.7863 ± 0.0016 | 0.7516 ± 0.0012 | 0.9718 |
| QED | **frozen state** | **2.7715** ± 0.0030 | **0.7348** ± 0.0016 | 0.0993 |
| | trivial prefix stats | 2.8064 ± 0.0008 | 0.7158 ± 0.0030 | 0.1044 |

Paired bootstrap of frozen-state minus trivial, 1,000 resamples, on the same held-out rows:

| property | NLL gain (nats) | target AUROC gain | pre-registered gate |
| --- | --- | --- | --- |
| aromatic rings | **−0.1409** [−0.1506, −0.1310] | **−0.0363** [−0.0440, −0.0288] | fails, opposite sign |
| HBD count | +0.1504 [+0.1417, +0.1588] | +0.0680 [+0.0554, +0.0811] | **PASS** |
| rotatable bonds | +0.1094 [+0.0978, +0.1209] | +0.0376 [+0.0268, +0.0492] | **PASS** |
| TPSA | **−0.0464** [−0.0559, −0.0361] | +0.0001 [−0.0100, +0.0121] | fails |
| cLogP | +0.0708 [+0.0622, +0.0796] | +0.0396 [+0.0299, +0.0500] | **PASS** |
| QED | +0.0327 [+0.0241, +0.0416] | +0.0168 [+0.0076, +0.0270] | fails, threshold |

### 13.1 The aromatic-ring crossover replicates on an independent sample

Phase 1 reported frozen-state 0.788 against trivial 0.825 target AUROC for aromatic rings.
Phase 2, on a **different 50,000-molecule draw**, with head initialisation seeded, and with
three seeds: frozen-state **0.7878 ± 0.0023** against trivial **0.8269 ± 0.0019**.

That is a replication to three decimal places of the pilot's most-defended claim, on
independent data, and it is immune to the section 11.5 defect by construction because
`CategoricalBinner` represents an integer target exactly. The NLL gap replicates too:
−0.141 nats in phase 1, **−0.1409** [−0.1506, −0.1310] in phase 2.

**Re-scoped by §21 (C17).** The replication stands and the numbers are unchanged — but they
are numbers about **probe point 12**. Section 21.3 measures the same quantity at all 13
probe points, and the crossover reverses in the middle of the network: aromatic rings reach
**0.8474** at probe point 3 against trivial **0.8269**. Read this subsection as "the final
layer loses to token counting for aromatic rings, and that replicates", not as "the frozen
representation loses". The pre-registered rule in §21.0.4 scores this **ARTEFACT**, and
rejection criterion R1 has to be read in that light.

### 13.2 Head-seed variance is small, which retires a limitation

Section 8.6 recorded that "head-to-head NLL differences carry training-seed variance that
the paired bootstrap does not capture". Now measured: across three seeds the standard
deviation of target AUROC is **at most 0.0041** (HBD count) and of NLL at most 0.0041
(aromatic rings), against between-head differences of 0.02–0.10 AUROC and 0.03–0.15 nats.

So initialisation variance is roughly an order of magnitude smaller than the effects being
compared, and every sign in the table above is safe from it. The pilot's single-seed
results were not fragile — but that is now a measurement rather than a hope, and it also
means the guidance advice in §8.6 ("do this before any claim about a margin under ~0.03")
can be relaxed to margins under ~0.01.

### 13.3 Predictability does not order the properties the way locality predicts

Stated here because it is the input to P3, and it is more interesting than a simple
correlation.

The frozen state beats fixed surface statistics on **four of six** properties, and loses on
**two**: aromatic rings decisively (−0.141 nats, −0.036 AUROC), and TPSA on NLL only
(−0.046 nats) with AUROC indistinguishable (+0.0001, CI spanning zero). Both losers are
properties a fixed token-statistics vector can nearly count: aromatic-atom and ring-marker
counts nearly determine ring count, and polar-atom counts nearly determine TPSA.

**Both losers reverse mid-network (§21.4.2).** Aromatic rings peak at probe point 3
(0.8474 vs trivial 0.8269, margin +0.0205) and TPSA at probe point 5 (0.7595 vs 0.7389,
+0.0206), each with a Bonferroni-corrected CI excluding zero and both neighbours supporting
it. So the inference that a fixed token-statistics vector "can nearly count" these
properties better than the model represents them holds of the model's **output layer**, not
of the model. The four-of-six count and the disagreement with the predicted locality
ordering below are unaffected — at probe point 4 or 5 the frozen state beats trivial on all
six — but the direction of the two losing cases is a fact about depth.

That is the surface-statistics side of the locality account behaving as it should. But note
what it does *not* do: it does not reproduce the predicted locality ordering. HBD count is
predicted second-most-local, yet the frozen state beats surface statistics on it by the
*largest* margin of any property (+0.150 nats). Per-atom N-H and O-H groups are locally
visible to a chemist and are *not* visible to a token-count vector that tracks element
identities but not hydrogen counts or bracket contents in detail.

So the secondary, correlational locality proxy and the predicted ordering already
disagree before headroom is measured. `LEXICAL_LOCALITY.md` §5 commits to how that is
resolved: **headroom is primary and wins**, because it is interventional and neither
circularity risk applies to it. Section 14 states how it will be scored, section 15 measures
it, and section 17.2 reports whether the two measures agree, as the pre-registration
requires. (They do not: locality score against trivial-head AUROC is rho = +0.086.)

## 14. How P1–P6 will be scored

Written **before** the headroom and central-test artefacts were inspected, in the same
spirit as section 9.1: each verdict below is a lookup against a named artefact rather than
a judgement formed after seeing the number. `docs/LEXICAL_LOCALITY.md` §4 states the
predictions; this section states what will count as passing them.

| # | Prediction | Scored against | Falsified if |
| --- | --- | --- | --- |
| P1 | relative headroom ranks properties as steerability does | Spearman rho of locality score against `throughout − unguided` hit-rate lift, six properties | rho <= 0, or the measurement-noise interval spans 0 |
| P2 | relative headroom declines with position for diffuse properties and stays flat for local counts | per-quartile slope, by P2's own named classes | both classes have the same sign of slope |
| P3 | the trivial head's performance correlates with steerability across properties | Spearman rho of trivial-head target AUROC against the same lift | rho <= 0 |
| P4 | guidance captures a larger fraction of available headroom for local than diffuse properties | `captured_fraction` by property | the fraction is flat across properties |
| P5 | where headroom is small, no lambda recovers the loss to best-of-N | **not testable in phase 2** — needs the lambda sweep (C10) | — |
| P6 | the size of the guidance-vs-best-of-N gap itself tracks locality | Spearman rho of locality score against `guidance_advantage` | rho <= 0, or the gap is constant |

Three commitments about how the numbers will be read, fixed here:

1. **Headroom is primary and beats the trivial head.** `LEXICAL_LOCALITY.md` §5 says so,
   because headroom is interventional and neither circularity risk applies to it. If P1
   and P3 disagree, P1 decides.
2. **n = 6, and the interval will not pretend otherwise.** The bootstrap propagates
   measurement noise in each property's two coordinates with the six properties held
   fixed. It does not cover the uncertainty from having chosen six hand-picked properties,
   which is larger and is not estimable from these data. A Spearman rho on six points has
   a two-sided p below 0.05 only at |rho| >= 0.886, i.e. essentially only for a perfect or
   near-perfect ordering. This is a weak test by construction and is reported as one.
3. **No property leaves the scatter.** If QED is an outlier it is reported as an outlier
   with the best available account of why, as `docs/TODO.md` §E requires. Section 12.3
   already flags QED's 0.0463-wide band as the most likely source of one.

**Two units, and which one carries the claim.** The hypothesis is stated in units of
target-interval width, so `relative_headroom` is the pre-registered primary score. But the
six bands differ in width by a factor of 400 (QED 0.0463, TPSA 18.79), and dividing by a
small number is a good way to manufacture a large ratio. So headroom is also reported in
**probability units** — the spread of `P(y_final in I | prefix + a)` across candidates —
which is band-width free and directly commensurable with a hit rate. Where the two units
disagree, both are reported and the disagreement is the finding rather than something to
resolve by picking one.

## 15. Result 6 — steering headroom

`outputs/pilot_50k_p2_headroom/`, `scripts/11_steering_headroom.py`. 400 held-out prefixes
balanced 100 per position quartile, each extended by all eight of the base model's top
candidate next tokens, each extended prefix continued 16 times under the base policy:
**51,200 continuations, 2,429,953 processed tokens**, one pass serving all six properties.

Rollout validity is **0.8371**, against 0.9965 for the Phase 4 bank. That gap is
informative rather than a problem, and section 15.3 uses it.

### 15.1 There is a lever at every position, for every property

The headline table. `base` is the base policy restricted to the top-8 (the
`truncation_control` reference); `best` is the single best candidate; `available` is the
noise-corrected difference; `captured` is the share of it the pilot's rule achieved.

| property | base policy | guided (λ=1) | best candidate | available (corrected) | **captured** |
| --- | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.1486 | 0.1633 | **0.3995** | +0.1349 | **0.1086** |
| HBD count | 0.0775 | 0.0820 | **0.2439** | +0.0812 | **0.0553** |
| rotatable bonds | 0.0968 | 0.0999 | **0.2262** | +0.0430 | **0.0717** |
| TPSA | 0.1038 | 0.1102 | **0.2789** | +0.0744 | **0.0868** |
| cLogP | 0.1075 | 0.1146 | **0.2840** | +0.0770 | **0.0922** |
| QED | 0.0899 | 0.0927 | **0.2217** | +0.0583 | **0.0482** |

All in hit-rate units at a single decoding position, on the **267 of 400** sampled prefixes
where all eight candidates cleared the four-usable-rollout threshold; the headroom columns of
section 15.4 use the wider **380 of 400** where at least two did. Both counts are in
`headroom_metrics.json` and neither table is computed on the other's set.

**This settles the question the pilot could not answer, and it settles it against us.**
For every one of the six properties there is a large one-step lever: choosing the best of
the eight candidates the base model already proposes would roughly **double to triple** the
probability of landing in the target band at that position (0.09–0.15 → 0.22–0.40). The
"there is no lever to pull" explanation of the pilot's negative result is **refuted for
every property**, not merely for some.

And the deployed rule captures between **4.8% and 10.9%** of it. That range is narrow —
a factor of 2.3 across six properties whose predicted locality spans the whole axis.

Two things about that number have to be said rather than left in the JSON.

**The denominator is noise-corrected, and the pre-registration's is not.**
`LEXICAL_LOCALITY.md` §3 writes the capture fraction as `(guided − base) / (ceiling − base)`.
`ceiling` is a max over eight noisy per-candidate estimates and is biased upward, so §3.1 —
written before any headroom number existed — added a permutation null, and the table above
subtracts it. Under the **literal pre-registered formula** capture is **2.1%–5.8%**
(`captured_fraction_of_headroom_preregistered_formula`). The correction moves capture *up*,
i.e. it makes our head look **better**, so it is conservative with respect to everything
section 15.6 concludes; but it is still a departure from the formula as written and both
values are carried.

**These are one-step quantities and they are not end-to-end quantities.** The ceiling is
"choose the best candidate here, then let the base policy finish", so it bounds a *single*
deviation. Guided decoding deviates at every position, and its end-to-end lift is 20–48x its
per-step gain (section 15.6, `per_step_versus_end_to_end`). A capture of 4.8–10.9% therefore
does **not** mean guidance achieves 5–11% of what is achievable end to end.

**And the prefixes are base-policy prefixes.** Headroom is measured at held-out prefixes the
*base* model generated, while guided decoding visits prefixes *guidance itself* produced,
where section 9.2.1 shows the head is worse calibrated. The `achieved` column is therefore
the head's performance in the easier of the two regimes, so the true capture is more likely
below this range than above it — conservative for section 15.6's conclusion, but it is an
extrapolation and not a measurement.

### 15.2 Why capture is low is a third explanation the pilot never posed

The pilot framed the ambiguity as "no lever" versus "bad head". The headroom data rules out
the first, which would leave the second — except that there is a third possibility the
pilot's framing missed, and it is measurable.

The ceiling is attained by picking whichever of the eight candidates has the highest
`P(y in I | prefix + a)`. But **the base policy is extremely concentrated**: it places a
mean of **0.738** of its top-8 mass on its own single favourite token, median **0.758**, on
the 267 prefixes every number in this section is computed on. (Over all 400 sampled prefixes
the concentration is higher still — mean 0.800, median 0.905 — and an earlier draft quoted
those, which overstated the argument by measuring it on a set none of the other figures use.)
At λ=1 the score keeps `log p_base` at full weight, so the property term has to overcome a
log-odds gap of roughly 2–3 nats to move the choice. It structurally cannot reach the
ceiling however good its head is.

Substituting an **oracle head** — the realised rollout hit rate itself — into the same λ=1
softmax the decoder uses separates the two, and section 15.6 reports the split. The oracle
is scored on the same rollouts that define it, so it is optimistic; that is the direction
that makes the conclusion conservative, because the argument is "even an optimistic oracle
only reaches this far at λ=1".

### 15.3 Is the ceiling inflated by survivorship? No

`P(y in I | prefix + a)` is computed over the rollouts that produced a parseable molecule,
and forcing a low-probability candidate lowers validity — which is exactly why overall
rollout validity is 0.8371 here against 0.9965 for un-forced prefixes. A candidate that
usually produces garbage but hits the target on the few molecules that survive would inflate
the ceiling.

It does not happen. The candidate that maximises the target probability has validity
**0.9831–0.9904**, barely below the base-policy-weighted 0.9972, and recomputing the
ceiling with an invalid completion counted as a **miss** moves it hardly at all
(`survivorship_check` in `outputs/pilot_50k_p2_locality/locality_metrics.json`):

| property | ceiling, valid-only | ceiling, invalid = miss | Δ | available (raw), valid-only | available (raw), invalid = miss | argmax validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.3995 | 0.3949 | −0.0046 | +0.2509 | +0.2468 | 0.9831 |
| HBD count | 0.2439 | 0.2418 | −0.0020 | +0.1664 | +0.1646 | 0.9874 |
| rotatable bonds | 0.2262 | 0.2238 | −0.0024 | +0.1294 | +0.1271 | 0.9860 |
| TPSA | 0.2789 | 0.2750 | −0.0039 | +0.1752 | +0.1715 | 0.9836 |
| cLogP | 0.2840 | 0.2802 | −0.0038 | +0.1766 | +0.1731 | 0.9846 |
| QED | 0.2217 | 0.2203 | −0.0014 | +0.1318 | +0.1304 | 0.9904 |

Two reading instructions, because an earlier draft of this table got both wrong and the
errors flattered the conclusion. First, it is computed on the **same 267 prefixes** as
section 15.1, so `ceiling, valid-only` reproduces that table's `best candidate` column
exactly; a version computed on the wider 380-prefix headroom set disagreed with it by up to
0.024 for no substantive reason. Second, the two `available` columns are **raw**, not
noise-corrected, so they are comparable to each other and **not** to section 15.1's
`available (corrected)`. Counting invalid completions as misses costs between 0.0014 and
0.0046 of ceiling — under 1.5% of it in every case.

The 0.837 figure is produced by the *low-probability* candidates — the 8th candidate carries
a mean base-policy weight of 0.0021 on this prefix set — and those are not the ones the
ceiling selects. So the
lever is real and it is reachable without abandoning valid chemistry.

### 15.4 Headroom in the pre-registered units falsifies P1, and shows why

The pre-registered primary score is headroom normalised by target-interval width. Measured,
noise-corrected, over 400 prefixes:

| property | interval width | relative headroom, raw | **relative headroom, excess** | probability-unit excess |
| --- | ---: | ---: | ---: | ---: |
| QED | 0.0463 | 5.677 | **3.509** | 0.085 |
| cLogP | 0.8656 | 2.721 | **1.553** | 0.102 |
| rotatable bonds | 1.0000 | 2.467 | **1.237** | 0.067 |
| TPSA | 18.7900 | 1.945 | **1.015** | 0.111 |
| aromatic rings | 1.0000 | 1.209 | **0.659** | 0.164 |
| HBD count | 1.0000 | 1.040 | **0.540** | 0.098 |

| ordering | |
| --- | --- |
| **pre-registered** (most local first) | aromatic rings, HBD count, rotatable bonds, TPSA, cLogP, QED |
| **measured, width-normalised** | QED, cLogP, rotatable bonds, TPSA, aromatic rings, HBD count |
| **measured, probability units** | aromatic rings, TPSA, cLogP, HBD count, QED, rotatable bonds |

The width-normalised ordering is **very nearly the exact reverse of the prediction**:
Spearman rho between predicted rank and measured relative headroom is **−0.886**, the most
negative value attainable on six points short of a perfect reversal.

And the mechanism is the one flagged in section 12.3 before the data existed. Spearman rho
between **1 / interval width** and relative headroom is **+0.698**. Normalising by band
width largely measures band width: QED's 0.0463-wide band gives it a denominator 20x
smaller than cLogP's and 400x smaller than TPSA's, and its relative headroom of 3.5 says
little more than that a small absolute change spans a narrow band several times over.

In band-width-free probability units the picture is different and much weaker: aromatic
rings is first, as predicted, but TPSA and cLogP come next and rotatable bonds comes last.
Spearman rho against the predicted ordering is **+0.371** — the right sign, nowhere near
significance on six points, and not the same conclusion as the primary score.

`LEXICAL_LOCALITY.md` §5 committed to headroom as primary. It did not anticipate that
headroom would give two different answers in two units. Both are reported; section 16 draws
the verdict.

### 15.5 P2 is falsified, and the direction is the interesting part

P2 predicted relative headroom would **decline with position** for the diffuse properties
and stay **flat** for the local counts. Q4 minus Q1, per property:

| property | P2 class | width-normalised Q1 → Q4 | Q4 − Q1 | probability-unit Q1 → Q4 | Q4 − Q1 |
| --- | --- | --- | ---: | --- | ---: |
| aromatic rings | local count | 0.610 → 0.631 | **+0.021** | 0.050 → 0.203 | **+0.153** |
| HBD count | local count | 0.486 → 0.415 | **−0.071** | 0.030 → 0.159 | **+0.129** |
| rotatable bonds | *unassigned by P2* | 1.270 → 0.975 | −0.295 | 0.026 → 0.090 | +0.064 |
| TPSA | diffuse | 0.899 → 0.947 | **+0.048** | 0.022 → 0.245 | **+0.223** |
| cLogP | diffuse | 1.204 → 1.695 | **+0.492** | 0.049 → 0.192 | **+0.143** |
| QED | diffuse | 3.065 → 3.751 | **+0.686** | 0.033 → 0.117 | **+0.084** |

Half of P2 holds: the two local counts are essentially **flat** in width-normalised units
(+0.021 and −0.071). The other half fails: the diffuse properties do not decline, they
**rise**, and cLogP and QED rise the most of any property.

In probability units **every property rises**, monotonically for four of six, and by a
factor of 3 to 10 from Q1 to Q4.

**That contradicts the mechanism `LEXICAL_LOCALITY.md` §2 proposed and the phase-1 report
adopted.** Section 6.2.1 explained cLogP's null late-window result by arguing that
"bulk lipophilicity is already fixed to within less than one interval width by the scaffold
chosen much earlier", so no remaining choice can move it. The ceiling says the opposite:
cLogP's relative headroom is **above one full interval width at every quartile from Q2
onward** (1.803, 1.486, 1.695), and it is **higher at Q4 than at Q1** (1.695 against 1.204).
**Late in the string a single token choice can move expected final cLogP by more than the
whole target band.**

This is not a contradiction of the phase-1 *measurement*, which stands — late guidance
really did buy only +0.015. It is a refutation of the phase-1 *explanation*. Guidance failed
late for cLogP not because the lever had vanished but because the rule did not pull it.

The reason the two are consistent is worth spelling out, because it is the same point as
section 15.2. Headroom is an unweighted max-minus-min over eight candidates. Late in a
sequence the conditional spread of the final property is small (phase 1 §5.6: cLogP's
conditional sd at Q4 is 0.324), *and* the base policy is very concentrated, so the
low-probability candidates that carry the large levers are exactly the ones λ=1 will not
select. A tight conditional distribution and a large unweighted candidate spread are not in
tension: the spread lives in the tail of the token distribution.

### 15.6 Splitting the loss: how much is λ=1 and how much is the head?

`scripts/12_locality_scatter.py`, `lambda1_ceiling_analysis` in
`outputs/pilot_50k_p2_locality/locality_metrics.json`.

Section 15.1 established that the deployed rule captures 4.8–10.9% of the ceiling. Two
things could be responsible and they call for completely different follow-ups:

- **λ=1 is too weak.** The ceiling is set by the *best* of eight candidates, and the base
  policy puts a mean 0.74 of its top-8 mass on one token. At λ=1 the property term must
  overcome that on its own. If this is the binding constraint, the λ sweep is the fix.
- **The head is too weak.** If λ=1 already permits most of the ceiling, no λ will help much
  and the fix is a better estimator.

Substituting an **oracle head** — the realised rollout hit rate — into the same λ=1 softmax
separates them. The oracle is scored on the rollouts that define it, so it banks its own
sampling noise; with about 13 usable rollouts per candidate that bias is large, and it is
removed by the same permutation-null construction used for headroom itself
(`headroom.permutation_null_oracle_gain`, six regression tests).

All figures are per-position gains in hit-rate units:

| property | our head | oracle at λ=1, raw | its noise floor | **oracle at λ=1, corrected** | head-free ceiling | **λ=1 permits** | **our head gets, of that** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | +0.0147 | +0.0956 | +0.0251 | **+0.0705** | +0.1349 | **52.2%** | **20.8%** |
| HBD count | +0.0045 | +0.0678 | +0.0300 | **+0.0378** | +0.0812 | **46.6%** | **11.9%** |
| rotatable bonds | +0.0031 | +0.0531 | +0.0302 | **+0.0229** | +0.0430 | **53.2%** | **13.5%** |
| TPSA | +0.0065 | +0.0625 | +0.0324 | **+0.0300** | +0.0744 | **40.4%** | **21.5%** |
| cLogP | +0.0071 | +0.0674 | +0.0278 | **+0.0397** | +0.0770 | **51.5%** | **17.9%** |
| QED | +0.0028 | +0.0435 | +0.0245 | **+0.0190** | +0.0583 | **32.6%** | **14.8%** |

**Both constraints bind, and the head binds about twice as hard.** λ=1 costs roughly half
the ceiling (it permits 32.6–53.2%), and of what it does permit our head collects only
**11.9–21.5%**. Multiplying gives the 5–11% of section 15.1.

Three consequences, and the third is the one that matters most for what to do next.

1. The pilot's dichotomy was incomplete but its second branch is the right one. There is a
   lever, λ=1 does throttle it, and the head is nevertheless the larger loss.
2. Note how large the *uncorrected* oracle gain is relative to its own noise floor. The
   noise floor is **26–57% of the raw gain** depending on the property (aromatic rings
   +0.0251 of +0.0956; rotatable bonds +0.0302 of +0.0531). Skipping the correction would
   put "λ=1 permits" at 71–124% instead of 33–53% — including a nonsensical value above
   100% for rotatable bonds, which is how the bias announces itself. That is why the
   correction lives in the library with regression tests rather than in a notebook.
3. **This is a per-position statement, and converting it into a ranking of experiments
   requires an assumption the data does not license.** Both ratios are gains in
   final-hit probability from **one** token choice with the rest of the sequence left to
   the base policy. Guided decoding intervenes at every position, and the amplification is
   large: our head's per-step gain is +0.0028 to +0.0147, while the same head's end-to-end
   lift (section 16.1) is +0.1012 to +0.2949 — a factor of **20 to 48**.

   Whether the per-step ratios survive that amplification is the whole question, and taking
   them at face value is provably wrong at these magnitudes. Multiplying each property's
   measured end-to-end lift by `1 / head-share` gives the lift a per-step-optimal head
   would have to produce if the ratio transferred linearly:

   | property | per-step head gain | end-to-end lift | amplification | implied lift at the λ=1 optimum | implied lift at the ceiling | largest lift arithmetically possible |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | aromatic rings | +0.0147 | +0.2949 | 20.1x | **1.418** | **2.715** | 0.821 |
   | HBD count | +0.0045 | +0.2150 | 47.9x | **1.811** | **3.888** | 0.916 |
   | rotatable bonds | +0.0031 | +0.1475 | 47.8x | **1.094** | **2.056** | 0.918 |
   | TPSA | +0.0065 | +0.1495 | 23.1x | 0.695 | **1.721** | 0.899 |
   | cLogP | +0.0071 | +0.1997 | 28.1x | **1.116** | **2.166** | 0.897 |
   | QED | +0.0028 | +0.1012 | 36.0x | 0.685 | **2.099** | 0.910 |

   A hit rate cannot exceed 1, so a lift cannot exceed `1 − unguided`. Four of six implied
   values at the λ=1 optimum, and **all six** at the ceiling, are above that bound. The
   per-step decomposition is sound; the linear transfer of it is not, and no factor read off
   this table — including "a factor of two" and "a factor of five to eight" — is an
   end-to-end factor. (`per_step_versus_end_to_end` in `locality_metrics.json`;
   `tests/test_report_matches_artifacts.py::TestPhase2PerStepIsNotEndToEnd`.)

   What the decomposition does support is a **per-position** claim: at a single position,
   λ=1 throttles about half the ceiling and our head collects a minority of the rest, so the
   head is the larger per-position loss. `reports/ABSTRACT.md` records a simulated reviewer
   naming the λ sweep as "the single highest-leverage addition", ranked explicitly above
   adding properties, and `docs/HANDOFF.md` §6 carries that as experiment E1. This
   measurement is **evidence against** that ranking but it does not settle it, because the
   quantity the ranking is about is end-to-end. **Section 19 settles it by measurement
   instead**, which is what the λ sweep is for.

## 16. Result 7 — the central test: guidance and best-of-N across six properties

`outputs/pilot_50k_p2_{guided,bestofn,confound,quality}_*/`. The pilot's protocol exactly:
six conditions, three seeds (101/202/303), 512 molecules per condition per seed, windows and
target intervals frozen before any of it ran. 18 guided runs, 36 best-of-N runs across two
accountings.

### 16.1 Intervention response

| property | unguided | truncation control | early | middle | late | **throughout** | **lift** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.1785 | 0.1829 | 0.2702 | 0.2822 | 0.2353 | **0.4735** | **+0.2949** |
| HBD count | 0.0837 | 0.0849 | 0.1450 | 0.1873 | 0.1189 | **0.2988** | **+0.2150** |
| rotatable bonds | 0.0824 | 0.0895 | 0.1527 | 0.1465 | 0.1061 | **0.2299** | **+0.1475** |
| TPSA | 0.1007 | 0.1097 | 0.1351 | 0.1538 | 0.1390 | **0.2502** | **+0.1495** |
| cLogP | 0.1033 | 0.1039 | 0.1475 | 0.1504 | 0.1514 | **0.3030** | **+0.1997** |
| QED | 0.0896 | 0.0829 | 0.1319 | 0.1373 | 0.1252 | **0.1908** | **+0.1012** |

Base rates differ across the counts, so the base-rate-adjusted view is reported alongside.
`vs truncation control` is the lift measured against the top-8 restriction rather than
against full-vocabulary sampling, which is the control that isolates the property term.

| property | base rate | lift | vs truncation control | fraction of room | late share of throughout |
| --- | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.1714 | +0.2949 | +0.2906 | 0.3590 | 19.2% |
| HBD count | 0.0823 | +0.2150 | +0.2138 | 0.2347 | 16.4% |
| rotatable bonds | 0.0827 | +0.1475 | +0.1404 | 0.1607 | 16.1% |
| TPSA | 0.1001 | +0.1495 | +0.1404 | 0.1662 | 25.6% |
| cLogP | 0.1000 | +0.1997 | +0.1991 | 0.2227 | 24.1% |
| QED | 0.1000 | +0.1012 | +0.1079 | 0.1112 | 35.2% |

**Reranking moves every property, and the top-8 restriction accounts for none of it.**
`truncation_control` sits within 0.007 of `unguided` for five of six properties and 0.0071
above for rotatable bonds, so the property term is doing essentially all of the work — the
same conclusion the pilot reached, now on six properties instead of two.

**The steerability ordering is** aromatic rings, HBD count, cLogP, TPSA, rotatable bonds,
QED, and it is identical whether ranked by raw lift or by fraction-of-room, so the base-rate
differences do not drive it.

### 16.2 Compute-matched best-of-N

N is solved from each guided run's measured tokens per returned molecule, under both
accountings, exactly as in section 6.4. `solve_best_of_n` floors, so best-of-N is always
given slightly less than the guided budget.

| property | guided tok/mol (actual / full) | N (actual / full) | best-of-N actual | **advantage** | best-of-N full | **advantage** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 419 / 9779 | 9 / 225 | 0.8294 | **-0.3560** | 1.0000 | **-0.5265** |
| HBD count | 402 / 9126 | 9 / 210 | 0.5234 | **-0.2247** | 1.0000 | **-0.7012** |
| rotatable bonds | 395 / 8699 | 9 / 200 | 0.5566 | **-0.3267** | 1.0000 | **-0.7701** |
| TPSA | 423 / 9906 | 9 / 227 | 0.6113 | **-0.3611** | 1.0000 | **-0.7498** |
| cLogP | 403 / 9201 | 9 / 211 | 0.6107 | **-0.3077** | 1.0000 | **-0.6970** |
| QED | 367 / 7541 | 8 / 173 | 0.5436 | **-0.3528** | 1.0000 | **-0.8092** |

**Rejection criterion R4 fires for all six properties under both accountings.** The pilot's
headline negative result generalises from two properties to six without exception, and the
`actual`-accounting margins (−0.22 to −0.36) sit in the same band the pilot reported
(−0.33 to −0.35). Under `full_recompute` best-of-N reaches 1.0000 for every property, so the
margin there is just the guided hit rate.

One thing the six-property view shows that two could not: **the size of the gap is driven by
the base rate, not by locality.** Spearman rho between base rate and the `actual`-accounting
advantage is **−0.771**, against **−0.086** for the locality score. A commoner target is
easier for best-of-N to hit in N independent draws, so the properties where guidance looks
least bad are the rare ones — HBD count, base rate 0.0823, has the smallest gap at −0.2247.
That is a fact about the baseline, not about the method, and it is the correct reading of
prediction P6 (section 17).

### 16.3 Does the interval-mask defect change guided decoding? No — as predicted

Section 11.6 predicted, from the shift-invariance of a softmax over candidates, that the
section 11.5 defect should barely affect *decoding* even though it halved every probability
the head reported. The control confirms it.

`outputs/pilot_50k_p2_guided_clogp_legacymask/` — same dataset, same seeds, all six
conditions, the head's interval mask the only difference:

| condition | fixed mask | legacy mask | difference |
| --- | ---: | ---: | ---: |
| unguided | 0.1033 | 0.1033 | 0.0000 |
| truncation_control | 0.1039 | 0.1039 | 0.0000 |
| early | 0.1475 | 0.1626 | **−0.0151** |
| middle | 0.1504 | 0.1909 | **−0.0404** |
| late | 0.1514 | 0.1443 | **+0.0071** |
| **throughout** | **0.3030** | **0.3068** | **−0.0039** |
| **guidance lift** | **+0.1997** | **+0.2035** | **−0.0039** |

**On the headline number the prediction is confirmed exactly: 0.004, well inside the seed
spread**, and in the direction that slightly *favours* the defective head. The mechanism is
the predicted one — `score(a) = log p_base(a) + λ log(P(y ∈ I | prefix + a) + ε)` goes into a
softmax over eight candidates, and a softmax is invariant to an additive constant, so scaling
every candidate's target probability by roughly the same factor cancels.

**But the windows tell a more interesting story than "no effect".** The defect *redistributes*
where the gain comes from: it makes early and middle guidance appreciably better (+0.0151 and
+0.0404) and late guidance slightly worse (−0.0071), netting to nothing over the whole
trajectory. So the ratio `P(bin 17 | a) / P(bins 17–18 | a)` is not constant in `a` after all —
it varies enough to matter by position — but the variation happens to be favourable early and
unfavourable late, and cancels in aggregate. That is luck, not a property of the defect, and it
is worth stating rather than presenting the −0.004 as if the mask were simply irrelevant.

Two consequences.

**The pilot's guided hit rates were not damaged.** Section 11.5 listed cLogP's guided lift
among the things the defect affected — "guidance was aimed at a narrower, lower sub-band" —
and on `throughout`, which is the number section 6.4 and the negative result rest on, that is
now measured at −0.004. What the defect damaged was the *reported calibration diagnostics*
(section 11.6). Section 8.2 inferred from the one to the other, and that inference was the
error.

**And cLogP's late-window change is a sample effect, not a mask effect.** Under the *legacy*
mask on the *phase-2* sample, cLogP's late lift is **+0.0410, 20% of `throughout`** — against
phase 1's +0.0153 and 10% under the same mask on a different sample. The mask moves it by
0.007; the sample moves it by 0.026. Section 17.4 takes that up, because it bears on the
pilot's most-quoted claim.

### 16.4 Length and size confound

Each condition re-standardised onto the `unguided` stratum distribution, four estimators, as
in section 6.3. `coverage` is the share of unguided mass in strata the condition visited.

| property | raw | length | size | **joint** | coverage | share surviving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | +0.2950 | +0.2668 | +0.2623 | **+0.2632** | 0.980 | 89% |
| HBD count | +0.2150 | +0.2168 | +0.2181 | **+0.2165** | 0.989 | 101% |
| rotatable bonds | +0.1475 | +0.1414 | +0.1370 | **+0.1315** | 0.979 | 89% |
| TPSA | +0.1494 | +0.1297 | +0.1384 | **+0.1264** | 0.981 | 85% |
| cLogP | +0.1997 | +0.1915 | +0.1875 | **+0.1899** | 0.986 | 95% |
| QED | +0.1012 | +0.0749 | +0.0675 | **+0.0724** | 0.965 | 72% |

**Rejection criterion R2 does not fire for any property.** 72–101% of each effect survives
joint matching on sequence length and heavy-atom count, at coverage 0.965–0.989.

Two entries deserve comment rather than a footnote. **HBD count's 101%** means matching very
slightly *increases* the estimate; H-bond donors are not a size proxy, so there is nothing
for the standardisation to remove. **QED's 72%** is the lowest of the six and the only one
below 80%: a quarter of the QED effect is attributable to length and size. QED is a
nonlinear function of eight descriptors, two of which (molecular weight, rotatable bonds)
are size-like, so this is expected — but it makes QED the property whose effect is least
clearly a property effect, and it is already the smallest effect of the six.

### 16.5 Chemical quality

`scripts/10_quality_analysis.py` on all six, comparing guided hits against base-policy hits
so quality is not confounded with the property shift itself. Descriptors whose bootstrap CI
excludes zero:

| property | condition | descriptor | difference | direction |
| --- | --- | --- | ---: | --- |
| aromatic rings | late | SA score | **+0.087** [+0.002, +0.175] | worse |
| HBD count | early | longest chain | **+0.572** [+0.002, +1.176] | worse |
| TPSA | late | fragment count | **+0.024** [+0.005, +0.047] | worse |
| rotatable bonds | throughout | QED | +0.030 | *better* |
| cLogP | throughout | QED | +0.035 | *better* |

Out of six properties × six conditions × seven descriptors, **three** degradations reach
significance and **two** improvements do. **Rejection criterion R3 does not fire anywhere.**

The pilot's specific finding replicates in reduced form: late aromatic-ring guidance is the
one condition that damages synthetic accessibility, **+0.087** [+0.002, +0.175] here against
+0.143 [+0.055, +0.236] in phase 1 — same sign, same condition, smaller and now only just
excluding zero. Two new small ones appear (HBD count early, longest chain; TPSA late,
fragment count) and the two QED improvements match the pilot's observation that bounded-interval
guidance makes molecules slightly *more* drug-like.

So the pilot's section 10 conclusion holds on a three-times-wider panel: **at λ=1 into a
bounded interval, steering does not buy the property with degenerate molecules**, and where
it costs anything the cost is small and localised to a single window.

## 17. Verdict on P1–P6

`outputs/pilot_50k_p2_locality/locality_metrics.json`. Scored against the rubric fixed in
section 14, before the artefacts were inspected.

**The interval on the P1 correlation propagates measurement noise in each property's two
coordinates — resampling guidance seeds and headroom prefixes — with the six properties held
fixed. It does not cover the uncertainty from having chosen six hand-picked properties, which
is larger and is not estimable from these data. On six points a two-sided Spearman p falls
below 0.05 only at |rho| ≥ 0.886.**

### 17.1 The three orderings

| ordering | |
| --- | --- |
| **pre-registered**, from chemistry alone, most local first | aromatic rings, HBD count, rotatable bonds, TPSA, cLogP, QED |
| **measured steerability** (identical by raw lift and by fraction-of-room) | aromatic rings, HBD count, **cLogP**, TPSA, **rotatable bonds**, QED |
| **measured locality score**, width-normalised headroom | QED, cLogP, rotatable bonds, TPSA, aromatic rings, HBD count |
| **measured locality**, probability units | aromatic rings, TPSA, cLogP, HBD count, QED, rotatable bonds |

### 17.2 The correlations

| test | rho | p | verdict |
| --- | ---: | ---: | --- |
| **P1** locality score (width-normalised) vs steerability | **−0.771** | 0.072 | **FALSIFIED** |
| P1, base-rate-adjusted steerability | −0.771 | 0.072 | FALSIFIED |
| P1b the same in probability units | **+0.714** | 0.111 | supported, not pre-registered |
| **pre-registered ordering vs steerability** | **+0.771** | 0.072 | **supported** |
| **the same, excluding the two properties phase 1 had already measured** (n = 4) | **+0.800** | 0.200 | **supported out of sample** |
| pre-registered ordering vs measured locality score | **−0.886** | **0.019** | the score contradicts its own pre-registration |
| **P3** trivial-head AUROC vs steerability | **+0.371** | 0.468 | not supported |
| **P4** locality score vs captured fraction | −0.257 | 0.623 | not supported |
| P4b the same in probability units | +0.771 | 0.072 | supported, not pre-registered |
| **P6** locality score vs guidance advantage (`actual`) | −0.086 | 0.872 | not supported |
| P6 the same under `full_recompute` | −0.543 | 0.266 | not supported |
| — base rate vs guidance advantage (`actual`) | **−0.771** | 0.072 | this is what predicts the gap |
| — base rate vs steerability | +0.314 | 0.544 | base rate does not drive steerability |
| — locality score vs trivial-head AUROC | +0.086 | 0.872 | the two locality measures do not agree |

The P1 measurement-noise interval is **[−0.771, −0.486]**, so the falsification is not a
noise artefact: every bootstrap draw puts the correlation on the wrong side of zero.

### 17.3 What this adds up to, stated carefully

**The hypothesis survives. Our measurement of it does not.**

The quantity in this study with **fewest post-hoc degrees of freedom** is
`properties.PREDICTED_LOCALITY_ORDER` — six properties ranked from chemistry alone, written
into code and pinned by a test before any phase-2 measurement existed. It correlates with
measured steerability at **rho = +0.771**. Only one substantial inversion occurs, and it is a
swap of adjacent-ish ranks: cLogP steers better than predicted (3rd, predicted 5th) and
rotatable bonds worse (5th, predicted 3rd).

**"Fewest" and not "none", which an earlier draft of this paragraph claimed.** Section 12.2
already concedes the retrodiction risk — the hypothesis exists to explain two results we
already had — and that concession has a specific consequence here: aromatic rings sits at
predicted rank 1 and cLogP at rank 5, which is exactly what phase 1 measured for those two
properties. Two of the six ranks restate a known result, so the ordering was not blind.

The check that fixes this is to drop them and correlate on the four properties whose ranks
could not have been informed by phase 1 — HBD count, rotatable bonds, TPSA, QED.
**The result survives: rho = +0.800 on n = 4** (p = 0.20, one adjacent inversion, TPSA and
rotatable bonds). Qualitatively that is a strengthening — the a-priori ordering holds on the
properties it was genuinely a-priori about — and statistically it is much weaker, because
four points cannot reach significance at all. Both readings are correct and neither should be
quoted without the other.

The quantity we *invented to operationalise* that ordering — headroom normalised by
target-interval width, designated primary in `LEXICAL_LOCALITY.md` §5 — correlates with
steerability at **rho = −0.771** and with its own pre-registration at **rho = −0.886**. It is
measuring inverse band width (rho = +0.698 against 1/width, section 15.4).

So P1 fails as written, and it fails because the operationalisation was bad rather than
because the idea was wrong. The band-width-free version of the same measurement recovers
+0.714. **We are obliged to weight this correctly: choosing the unit after seeing the data is
exactly the freedom pre-registration exists to remove, so P1b is not evidence of the same
grade as P1 would have been.** What carries weight is that the *fully a-priori* ordering, which
involved no measurement choice at all, agrees with steerability independently — and it agrees
with the probability-unit measure rather than the width-normalised one, which is at least
consistent.

Honest summary: **on six hand-picked properties, chemical reasoning about how a property is
written into SMILES predicts which properties a token-choice reranker can move, at rho ≈ 0.77
with p ≈ 0.07 and n = 6. That is a suggestive result, not an established one, and the
mechanism measurement designed to explain it pointed the wrong way.**

**P2 is falsified**, and in the opposite direction from the prediction (section 15.5): the
local counts are flat (slopes −0.001 and −0.028 per quartile) as predicted, but the diffuse
properties **rise** (+0.014, +0.116, +0.219) rather than declining.

**P3 is not supported** (+0.371, p = 0.47). Surface token statistics predict ring count and
TPSA well and HBD count poorly, and HBD count is the second most steerable property, so the
correlational proxy fails where the interventional one and the a-priori ordering succeed.

**P4 is not supported** in the pre-registered units. Note a drafting weakness in the original
pre-registration: P4's stated falsification condition is that the captured fraction is *flat*
across properties, and it is not flat (0.048–0.109, a factor of 2.3), so the literal
condition does not fire — but the substantive prediction was that it would be *ordered by
locality*, and rho is −0.257. Reported as not supported, with the discrepancy named rather
than exploited.

**P5 was not testable** in phase 2 as originally written; it requires the λ sweep, which
**section 19** now reports. Section 15.6 bears on it only per-position — λ=1 permits 32–53%
of the one-step ceiling — and an earlier draft of this paragraph turned that into "a λ sweep
has at most a factor of two to give and cannot close a 0.22–0.36 gap". That inference was not
available: the 32–53% is a single-position quantity and the 0.22–0.36 gap is end-to-end, and
section 15.6 consequence 3 shows the two are separated by a factor of 20–48. The sentence is
withdrawn and replaced by the measurement.

**P6 is not supported, and the reason is a better finding than P6 would have been.** The gap
to best-of-N does not track locality (−0.086 under `actual`). It tracks the **base rate**
(−0.771). Best-of-N's advantage is mechanical: a commoner target is easier to hit in N
independent draws. P6 was the prediction that would have unified the pilot's two findings
into one variable; it does not, and what actually predicts the gap is a property of the
baseline rather than of the model or the method.

### 17.4 A phase-1 claim that did not replicate

This belongs here rather than buried, because it touches the pilot's most-quoted result.

Phase 1 §6.2.1 reported cLogP's late-window lift as **+0.015 ± 0.022**, "the only window in
either property that is not clearly separated from zero", retaining 10% of the full effect
against aromatic rings' 22%. That asymmetry is one of the two legs of the CL3 double
dissociation.

Phase 2 measures cLogP's late lift at **+0.0481, 24.1% of `throughout`** — and aromatic
rings' at **19.2%**. The ordering has **reversed**, and cLogP's window profile is now
essentially flat (22% / 24% / 24%) rather than flat-then-collapse (34% / 36% / 10%).

The section 16.3 control attributes it. Running the **legacy** mask on the **phase-2** sample
gives a late lift of +0.0410 (20% of `throughout`), so of the +0.033 difference between phase 1
and phase 2 the mask accounts for **0.007** and the sample for **0.026**. **The phase-1
late-window number was a single-sample estimate that has not replicated.** Its own reported
seed spread (± 0.022) was already as large as the effect; the honest reading now is that the
pilot's three seeds understated its uncertainty, exactly as section 11.3 warned about
three-seed spreads.

**Consequence for CL3.** The predictability half of the dissociation is untouched and
replicated (section 13.1). The *steerability timing* half — "cLogP is least steerable late" —
is **withdrawn**. What survives is the position-independent version: aromatic ring count is
predicted worse than token counting yet is the most steerable property of six, and cLogP is
predicted well yet is not the most steerable. That is still a dissociation between
predictability and controllability. It is no longer a dissociation about *when*.

`reports/ABSTRACT.md` claim CL3 is updated accordingly, and abstract v3 does not make the
late-window claim.

## 18. What phase 2 changes, in total

**Unchanged and now better supported.** Guided reranking moves every property and the top-8
restriction accounts for none of it; the effect is not a length or size artefact anywhere
(72–101% survives joint matching); chemical quality is essentially unharmed at λ=1 into a
bounded interval; and reranking loses to compute-matched best-of-N for **all six** properties
under **both** accountings. The aromatic-ring predictability crossover replicated to three
decimal places on an independent 50,000-molecule sample.

**Newly established, and the strongest thing phase 2 produced.** The negative result is
**located**. A head-free, λ-free ceiling shows a large one-step lever at every position for
every property — the best available candidate roughly doubles to triples the target
probability — and reranking at λ=1 captures 4.8–10.9% of it. Decomposed: λ=1 permits 32–53%
of the ceiling, and the head collects 12–22% of what λ=1 permits. Per position, the failure is
neither a shortage of signal nor primarily the guidance strength; it is the estimator. All of
those figures are **per decoding position**, and end-to-end lift is 20–48x per-step gain
(§15.6), so they rank causes at a position and do not by themselves rank experiments;
section 19 measures the λ term end to end instead of extrapolating it.

**Withdrawn or corrected.** cLogP's late-window null result did not replicate (§17.4), so the
timing half of the double dissociation is withdrawn. The pilot's cLogP "under-confident head"
was a target-interval/binner misalignment in our own code, not a property of the head
(§11.5–11.6); its AUROCs stand. The claim that the git-excluded arrays are reproducible from
the pinned revision and seeds holds only within a device class (§11.2).

**Falsified.** The pre-registered lexical-locality predictions P1, P2, P3, P4 and P6, as
written. The hypothesis they encode is supported by the one fully a-priori quantity
(rho = +0.771 between the chemistry-derived ordering and steerability) but not by the
mechanism measurement designed to test it.

**What to do next, in the order the data implies** — which is not the order
`docs/HANDOFF.md` §6 or the recorded reviewer proposed. Item 2 was run immediately after this
section was first written, and section 19 reports it:

1. Fix the head (probe-layer sweep; on-policy calibration re-measured after §11.6). Worth up
   to 5–8x on captured headroom **per position**; how much of that survives to the end of a
   sequence is unmeasured, and section 19 shows the analogous per-position bound on λ
   overstated λ's end-to-end value by roughly 20–50%.
2. ~~The λ sweep.~~ **Done, section 19.** Tuning λ is worth **1.29–1.69x** end to end, the
   response is an inverted U with an optimum at λ = 2–4 rather than the monotone rise the
   pre-committed interpretation allowed for, no λ beats compute-matched best-of-N, and the
   degenerate molecules do appear — above the optimum, not at it.
3. Restrict the headroom ceiling to likelihood-plausible candidates, which bounds what a
   base-policy-respecting decoder could ever reach and is tighter than what is reported here.
   Section 19 makes this more interesting, not less: λ = 8 *is* a likelihood-disrespecting
   decoder, and it is worse than λ = 2 on every axis.
4. Explain why headroom in probability units *rises* toward the end of a sequence for every
   property. Nothing in this project predicted it and the arrays needed are already on disk.

## 19. Result 8 — the λ sweep (C10), and chemical quality at every λ (C12)

`outputs/pilot_50k_p2_lam*_{guided,bestofn,quality}_*/`, assembled by
`scripts/15_lambda_sweep.py` into `outputs/pilot_50k_p2_lambda_sweep/`.

Every negative statement in phase 1 and in sections 15–18 was really "no effect **at λ=1**",
and section 15.6's per-position decomposition could not fix that, because — as the audit
recorded in `docs/TODO.md` C22.1 established — a per-position bound is not an end-to-end
bound. This measures the end-to-end quantity directly.

**Design, and what was fixed before the sweep ran.** Three anchor properties, chosen by a
rule stated in `scripts/15_lambda_sweep.py` before any sweep result existed: the most
steerable at λ=1 (aromatic rings, +0.2949), the least steerable (QED, +0.1012), and the
pre-registered discriminating case (HBD count). Five new λ values, 0.25 / 0.5 / 2 / 4 / 8;
λ=1 is the central test's own run and was **not** re-generated. Three seeds, 512 molecules per
condition per seed, same head, same backend, same frozen intervals and windows. λ is set by
`--lam`, which folds the override into the config dict so `configs_used.json` records the
value used. `unguided` is regenerated at every λ as a bug alarm and reproduces its
central-test value exactly (aromatic rings 0.1785 at all six λ).

`docs/HANDOFF.md` §6 states the interpretation **before the result**: monotone rise plus a
win over best-of-N at some λ overturns CL4; rise with quality degradation is the classic
reward-hacking trade-off and is a result; saturation below best-of-N at all λ strengthens CL4.
The outcome is the third, with the second layered on top, and neither exactly as written.

### 19.1 The response to λ is an inverted U, not a saturation

| property | λ=0.25 | λ=0.5 | **λ=1** | **λ=2** | λ=4 | λ=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **aromatic rings** lift | +0.0573 | +0.1225 | +0.2949 | **+0.3794** | +0.3204 | +0.2207 |
| **HBD count** lift | +0.0487 | +0.0976 | +0.2150 | **+0.3466** | +0.2893 | +0.2267 |
| **QED** lift | +0.0436 | +0.0796 | +0.1012 | +0.1465 | **+0.1711** | +0.1542 |

| property | validity λ=1 → λ=8 | uniqueness λ=8 | content length λ=1 → λ=8 |
| --- | --- | ---: | --- |
| aromatic rings | 0.995 → **0.807** | 0.997 | 45.4 → 44.0 |
| HBD count | 0.998 → **0.902** | **0.900** | 43.4 → 39.8 |
| QED | 0.996 → **0.900** | 0.996 | 39.6 → 36.4 |

**The pre-committed "hit rate rises monotonically with λ" branch does not fire.** Every
property has an interior optimum — λ=2 for both counts, λ=4 for QED — and every property is
*worse at λ=8 than at λ=2*. Aromatic rings at λ=8 (+0.2207) is below its λ=1 value.

The mechanism is visible in the same table: pushing λ past the optimum destroys the base
policy rather than overriding it. Validity falls from 0.995–0.998 to 0.807–0.902, molecules
get shorter by 3–6 content tokens, and HBD count's uniqueness — 1.000 at every λ up to 4 —
drops to 0.900, which is the signature of a decoder collapsing onto a few high-scoring
strings. A hit rate computed over molecules 10–20% of which no longer parse is not a better
controller.

**Tuning λ is worth 1.29x (aromatic rings), 1.61x (HBD count) and 1.69x (QED)** on the lift,
end to end. Section 15.6's withdrawn per-position claim was "at most about a factor of two";
that number turns out to be roughly the right order — but it was not available from the
measurement that produced it, and it was optimistic by 20–50% for two of the three anchors.
The honest position is that the per-position bound and the end-to-end measurement agree here,
and that we could not have known they would.

### 19.2 P5 survives, and CL4 is strengthened

**P5** (`docs/LEXICAL_LOCALITY.md` §4): *where headroom is small, no λ recovers the loss to
best-of-N.* Falsified if some λ beats best-of-N in a low-headroom regime.

| property | best λ | guidance lift at best λ | compute-matched best-of-N | **best advantage, any λ** |
| --- | ---: | ---: | ---: | ---: |
| aromatic rings | 2 | 0.5579 | 0.8294 | **−0.2715** |
| HBD count | 2 | 0.4303 | 0.5234 | **−0.0931** |
| QED | 4 | 0.2607 | 0.5436 | **−0.2829** |

**No λ beats compute-matched best-of-N for any of the three.** P5 is not falsified, and the
pilot's headline negative result CL4 is now established across **six properties, two token
accountings and six values of λ** rather than at one λ. The closest guidance ever comes is HBD
count at λ=2, where the gap narrows from −0.2247 to **−0.0931** — a real improvement, on the
property with the smallest gap to begin with, and still a loss.

Best-of-N's own hit rate is held fixed across λ because N is solved from the guided run's
measured tokens per molecule and that barely moves; the two cases where it changes (λ=8,
where guided molecules are shorter) give best-of-N *more* budget, not less. Only the `actual`
accounting is run: under `full_recompute` the budget is N ≈ 200 independent draws against a
base rate of 0.08–0.17, so best-of-N misses with probability below 1e-7 and measured exactly
1.0000 at λ=1 for all six properties. `n_candidates_solved` and `base_rate` are written to
every `bestofn_metrics.json` so that argument is checkable rather than asserted.

### 19.3 C12: the degenerate molecules appear, above the optimum

`scripts/10_quality_analysis.py` at every λ, guided hits against base-policy hits, so quality
is not confounded with the property shift itself. Degeneracy rate among molecules that hit the
target:

| property | base-policy hits | λ=0.25 | λ=0.5 | λ=1 | λ=2 | λ=4 | λ=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.0330 | 0.0139 | 0.0130 | 0.0138 | 0.0141 | 0.0336 | **0.0606** |
| HBD count | 0.0703 | 0.0296 | 0.0253 | 0.0197 | 0.0457 | 0.0479 | **0.1070** |
| QED | 0.0146 | 0.0049 | 0.0154 | 0.0103 | 0.0166 | **0.0566** | **0.1217** |

**Rejection criterion R3 does not fire at λ ≤ 2 and fires at λ ≥ 4.** The differences whose
bootstrap CI excludes zero, guided hits minus base-policy hits:

| property | λ | any degeneracy | SA score | longest chain | carbon fraction | fragment count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QED | 4 | **+0.0420*** | **+0.227*** | −0.290* | +0.021* | **+0.044*** |
| QED | 8 | **+0.1071*** | **+0.389*** | −0.452* | +0.016* | **+0.136*** |
| aromatic rings | 8 | +0.0276 | **+0.110*** | −0.707* | −0.035* | **+0.037*** |
| HBD count | 8 | +0.0367 | −0.054 | −0.608* | −0.079* | +0.065 |

`*` = 95% bootstrap CI excludes zero. Higher SA score, higher fragment count and higher
degeneracy rate are all worse.

**This is the prediction `docs/HANDOFF.md` E1 made, and it holds.** The pilot's "steering does
not buy the property with degenerate molecules" result is confirmed to be **specific to low
λ**: at λ=8 every property's guided hits are more degenerate than base-policy hits, QED's by
a factor of **8.3**, and QED's synthetic accessibility is worse by +0.389 — five times the
largest degradation anywhere in phase 1 (late aromatic-ring guidance, +0.143). Multi-fragment
molecules are the specific failure: fragment count rises significantly for three of the four
rows above.

Two details are worth more than the headline.

**The one property whose optimum is already in the degenerate regime is QED.** Its best λ is 4,
and at λ=4 its degeneracy difference is +0.0420 with the CI excluding zero. So for QED there is
no λ that is simultaneously best for hit rate and clean — the trade-off is real and binding,
not merely theoretical. For the two counts the optimum (λ=2) is still clean, and the damage
starts one step above it.

**Longest chain falls rather than rises at high λ, for all three properties.** The literature's
canonical reward-hacked molecule is a long greasy tail; ours is the opposite — high λ produces
*shorter, more fragmented, harder-to-synthesise* molecules. That is a different failure mode
from the one section 10.3 anticipated, and it follows from steering into a **bounded interval**
rather than maximising: an unbounded objective rewards runaway growth, a bounded one rewards
whatever cheap edit lands inside the band, and fragmenting the molecule is cheaper than growing
it. The bounded-interval defence in section 10.3 survives, but its scope is now measured: it
holds up to about λ=2 and fails above it.

### 19.4 What section 19 changes

- **CL4 is strengthened, not overturned.** Six properties, two accountings, six λ. The
  "reranking was simply not tuned" objection recorded as limitation 1 in
  `reports/ABSTRACT.md` is answered: it was not tuned, tuning it is worth 1.3–1.7x, and it
  still loses by 0.09–0.28.
- **P5 is not falsified**, and it was testable after all — it was the one pre-registered
  prediction phase 2 had to leave open.
- **The quality null result is now bounded rather than general.** Section 10's conclusion and
  section 16.5's replication both stand *at λ ≤ 2*; R3 fires at λ ≥ 4.
- **λ=1 was a defensible but suboptimal choice.** Everything phase 1 and sections 15–18 report
  understates guided decoding by 1.3–1.7x. None of the conclusions change sign, because
  best-of-N's margin is larger than that factor for five of the six properties, and for the
  sixth (HBD count) it is 0.0931 against a 1.61x improvement already applied.
- **What it does not do is settle the head-versus-λ ranking.** It measures the λ term end to
  end (1.3–1.7x) and leaves the head term measured only per position (5–8x there). The
  comparison the audit invalidated is still not available; what is available is that the λ
  term is now known to be small and known to be capped by a mechanism — base-policy
  destruction — that a better head does not obviously share. That makes fixing the head the
  better bet, on weaker grounds than section 15.6 originally claimed.

  **Withdrawn by §20 and §21.** Sections 20 and 21 measure the head term end to end, by the
  three routes available without unfreezing anything: post-hoc calibration, readout capacity
  and shape, and probe depth. The best retrained readout is worth **1.09x** at one anchor
  and 1.00x and 0.76x at the other two (§20.5); every calibrator is worth **less** than 1.00x
  (§20.3, §20.5); and a better-predicting probe layer buys no per-position steering at all
  (§21.5). The honest position after §20 and §21 is that **neither term is cheaply
  available**: λ is worth 1.3–1.7x and is capped by base-policy destruction, and the head is
  not improvable by calibration, only marginally and inconsistently by capacity or readout
  shape, and not at all by moving the probe point. "Fixing the head is the better bet" does
  not survive.

---

## 20. Result 9 — can the head be fixed? (C18)

Run 2026-07-30 on the RTX 4090, phase-2 dataset `outputs/pilot_50k_p2/`, phase-2 heads
`outputs/pilot_50k_heads_p2/` (seed 1234, the checkpoint script 05 steers with).
Artefacts under `outputs/c18_*`; every asserted number is bound by
`tests/test_head_calibration.py`. No pre-existing directory under `outputs/` was
modified. The pre-registration in §20.1 was written and timestamped before the first
measurement script ran.

Section 15.6 decomposed the per-position capture loss: at λ=1 the deployed rule permits
32.6–53.2% of the head-free ceiling and our head collects 11.9–21.5% of that, so **the
head is the larger per-position loss**. C18 asks whether it can be fixed. The answer is
that the two cheap routes both fail, and the reason the first one fails is not empirical
— it is algebraic, and it was written down before the runs.

---

### 20.0 What was run

| stage | artefact | processed tokens |
| --- | --- | ---: |
| the written prediction, committed first | `c18_prediction/prediction.json` | 0 |
| off-policy re-measurement + post-hoc calibration, 6 properties × 2,000 guided molecules | `c18_offpolicy_calibration/` | 5,349,199 |
| three retrained readouts × 6 properties | `c18_heads_{wide,focused,wide_focused}/`, `c18_head_variants/` | 0 (no generation) |
| per-position capture for 7 arms × 6 properties | `c18_per_position/` | 85,127 |
| the λ-equivalence identity, 4 full guided runs | `c18_identity*/` | 2,511,378 |
| end to end: 4 calibration arms × 3 anchors, plus the sharpened arm of §20.5.1 | `c18_guided_{uncalibrated,platt,isotonic,bin_temperature,binT0p4}_*` | 8,975,924 |
| end to end: 2 retrained readouts × 3 anchors | `c18_guided_head_*` | 4,139,058 |
| compute-matched best-of-N, three shared runs | `c18_bestofn_N9_*`, `c18_matched_best_of_n/` | 1,842,813 |
| **total** | | **22,903,499** |

Cost is reported in processed tokens throughout, per §11.7. No wall-clock claim is made.

**Nothing here is a second DAgger round.** The one permitted data-aggregation round is
spent (§9.2.1). Every calibrator is fitted on the head's *outputs*; every retrained head
is trained on the phase-2 dataset's **base-policy** prefixes and on nothing else. Guided
prefixes are used for measurement and for post-hoc calibration only, which
`docs/HANDOFF.md` §6 E2 explicitly distinguishes from retraining.

---

### 20.1 The prediction, committed before any measurement

`outputs/c18_prediction/prediction.json`, written and timestamped before the first
measurement script ran; `test_the_prediction_was_written_before_the_measurements`
enforces the ordering by file mtime rather than by trust.

The decoder samples the next token from `softmax_a( log p_base(a) + λ·log(q(a) + ε) )`
over eight candidates. The question C18 has to answer before spending GPU time is which
calibration families are, and are not, algebraically distinct from a rescale of λ —
because §19 has already swept λ over {0.25, 0.5, 1, 2, 4, 8}.

| family | map | verdict committed in advance |
| --- | --- | --- |
| power | `g(q) = c·q^α` | **exactly** a λ rescale: `λ·log(c·q^α) = (λα)·log q + λ·log c`, and the softmax annihilates the second term |
| Platt | `g(q) = σ(a·logit q + b)` | that family **to first order**: `σ(x) → e^x` and `logit q → log q` as `q → 0`, and our candidate probabilities sit near base rates of 0.08–0.17 |
| isotonic | any monotone step function of `q` | **not** a λ rescale — it makes the effective exponent `d log g / d log q` depend on `q` — but it is monotone, so it cannot change *which* candidate the head prefers |
| bin-logit temperature | `q(T) = Σ_{i∈M} e^{z_i/T} / Σ_j e^{z_j/T}` | **not even a function of `q` alone**; two candidates with equal `q` can move differently, so this is the only post-hoc family that can reorder candidates |

And the directional prediction, which is the one that mattered:

> The stated defect is *under*-confidence. Any map correcting that at small `q` must
> satisfy `g(q) > q`, which for a power map means `α < 1`, which means
> `λ_eff = λα < 1`. §19 measures the lift falling steeply below λ=1. **So "fix the
> calibration by raising the head's probabilities" is a λ *decrease* and should make
> end-to-end guidance worse, not better.**

Three falsifiable claims were recorded with it: that the fitted Platt slope would be
`α < 1`; that a power-calibrated head at λ=1 would produce the *same molecules* as the
raw head at λ=α under the same seed; and that no arm would beat compute-matched
best-of-N.

**How it did.** The three claims held. The λ-equivalence held exactly. Two subsidiary
predictions failed and are marked below: that the off-policy factor would be smaller for
the count properties (it is largest for HBD count), and that the bin-logit temperature
would be "genuinely distinct" in a way that mattered (it is formally distinct and
behaves in practice like a reparametrised λ).

---

### 20.2 Trap 1 — the off-policy gap, re-measured on the fixed heads

`outputs/c18_offpolicy_calibration/offpolicy_calibration.json`,
`scripts/17_offpolicy_calibration.py`. Per property: 2,000 guided molecules generated at
λ=1 with the phase-2 head (seed 91234), one prefix drawn per position quartile from each
kept molecule exactly as scripts 02 and 08 draw them, giving ~7,970 guided prefixes;
against the phase-2 dataset's own held-out base-policy prefixes for the on-policy column.

| property | on-policy predicted | on-policy observed | ratio | on-policy ECE | **off-policy predicted** | **off-policy observed** | **ratio** | off-policy ECE | off-policy AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.1506 | 0.1587 | 1.05 | 0.0143 | 0.2513 | 0.4428 | **1.76** | 0.1922 | 0.6261 |
| HBD count | 0.0746 | 0.0825 | 1.11 | 0.0087 | 0.1387 | 0.2903 | **2.09** | 0.1516 | 0.6742 |
| rotatable bonds | 0.0940 | 0.0803 | 0.85 | 0.0137 | 0.1662 | 0.2246 | **1.35** | 0.0685 | 0.6305 |
| TPSA | 0.0955 | 0.0977 | 1.02 | 0.0074 | 0.1472 | 0.2489 | **1.69** | 0.1047 | 0.6358 |
| cLogP | 0.0943 | 0.0929 | 0.99 | 0.0045 | 0.1668 | 0.2825 | **1.69** | 0.1157 | 0.6598 |
| QED | 0.0907 | 0.0990 | 1.09 | 0.0085 | 0.1310 | 0.1846 | **1.41** | 0.0536 | 0.6120 |

**The pilot's 3.5x is now 1.69x for cLogP.** §9.2.1 reported the head predicting 0.076
against an observed 0.267 on guided prefixes; §11.6 argued about half of that factor was
the interval-mask defect. Measured on the fixed head, on an independent sample, the
factor is **1.69** — within 0.01 of what §11.5's arithmetic implies (3.5 / 2.08 = 1.68).
That agreement is close enough to be worth a caveat: the two quantities are measured on
different samples and one of them is a ratio of two noisy means, so it should be read as
"the same order and the same direction", not as a three-significant-figure match.
§11.6's correction to §8.2 is confirmed rather than merely argued.

Three further things this table says that the pilot could not.

**On-policy, every head is essentially calibrated.** Ratios 0.85–1.11, ECE 0.0045–0.0143.
The cLogP row (0.0943 against 0.0929) reproduces §11.6's fixed-mask measurement exactly,
which is a useful check that this pipeline and script 14 agree.

**The off-policy gap is real but modest, and it does not order by locality.** 1.35–2.09
across six properties. The prediction that the count properties — which never had the
mask defect — would show the *smallest* off-policy factors is **wrong**: HBD count has the
largest (2.09) and rotatable bonds the smallest (1.35), and the three counts straddle the
three continuous properties. Whatever drives the residual shift, it is not the defect and
it is not the count/continuous split.

**The head still ranks.** Off-policy AUROC is 0.6120–0.6742, well above chance and only
moderately below the on-policy target AUROCs of 0.735–0.790 (§13). This matters more than
the calibration does, and §20.3 says why.

---

### 20.3 Route (a) — post-hoc calibration works as calibration and does nothing for decoding

Calibrators fitted on half the guided molecules and scored on the other half, split by
canonical molecule so no molecule's four prefixes straddle the halves (the §3.2 grouping
rule). Platt by Newton–Raphson on two parameters; isotonic by PAVA; the bin-logit
temperature selected by lowest ECE on the fit half over a fixed grid.

| property | ECE uncalibrated | ECE Platt | ECE isotonic | AUROC uncalibrated | AUROC Platt | AUROC isotonic | Platt slope `a` | Platt intercept `b` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.2110 | **0.0464** | **0.0424** | 0.6368 | 0.6368 | 0.6363 | **0.405** | +0.236 |
| HBD count | 0.1624 | **0.0279** | **0.0213** | 0.6844 | 0.6844 | 0.6827 | **0.538** | +0.183 |
| rotatable bonds | 0.0785 | **0.0277** | **0.0273** | 0.6352 | 0.6352 | 0.6339 | **0.443** | −0.514 |
| TPSA | 0.1098 | **0.0205** | **0.0236** | 0.6256 | 0.6256 | 0.6243 | **0.583** | −0.023 |
| cLogP | 0.1181 | **0.0205** | **0.0121** | 0.6590 | 0.6590 | 0.6581 | **0.618** | +0.161 |
| QED | 0.0595 | **0.0138** | **0.0134** | 0.6216 | 0.6216 | 0.6204 | **0.545** | −0.434 |

**Calibration works, by a factor of 3 to 6 on ECE.** This is not a failed fit. If the
downstream result is negative it is not because the calibrator is bad.

**And AUROC does not move at all.** Platt's AUROC column is *identical* to the
uncalibrated column at four decimal places for all six properties, because Platt is
strictly monotone and AUROC is a rank statistic. Isotonic moves it by at most 0.0017, and
downward. **The decoder consumes a softmax over eight candidates, which is a function of
the *differences* of `log q` — a rank-and-spacing quantity, not a level. Post-hoc
calibration moves the level and leaves the ranking untouched.**

**Every fitted slope is below 1, as predicted, and by a lot: 0.405–0.618.** The
calibration that fixes the head's under-confidence is `λ_eff ≈ 0.4–0.6`, and §19 measures
guidance at λ=0.5 to be about **40%** as effective as at λ=1 (aromatic rings +0.1225
against +0.2949). The intercepts, which carry most of the *level* correction, are the part
the softmax annihilates — and where an intercept is large enough to push the calibrated
probability toward 1 the sigmoid saturates, which flattens the map further and lowers the
effective exponent again. Both mechanisms point the same way.

#### 20.3.1 The identity, demonstrated end to end rather than argued

`outputs/c18_identity/identity_check.json`, `scripts/17_check_identity.py`. Four full
guided runs on aromatic rings, 512 molecules × 3 seeds, differing only in whether the
power calibration `g(q) = c·q^α` (α = 0.4047, from the fitted Platt slope) is applied at
λ=1 or the raw head is run at λ = α.

| ε | identical molecules | hit rate, calibrated arm | hit rate, λ = α arm | difference |
| --- | ---: | ---: | ---: | ---: |
| 0 (exact arithmetic) | **1536 / 1536 = 1.000** | 0.2869 | 0.2869 | **0.0** |
| 1e-6 (deployed) | 1535 / 1536 = 0.9993 | 0.2869 | 0.2869 | **0.0** |

At ε = 0 the two runs return **the same molecules, all 1,536 of them**. The `docs/HANDOFF.md`
§6 E2 recipe — "temperature-scale or isotonic-calibrate the head's interval probability,
then re-run the guided evaluation" — is, in its temperature-scaling half, not a new
experiment. It is a point on the λ sweep of §19, and the fitted slope says which point.

And the point it lands on is a bad one. Aromatic rings at λ=1 has `throughout` = 0.4735
(§16.1); the calibrated head has **0.2869**. The λ term is the whole of the difference.

#### 20.3.2 Per-position: every monotone calibrator makes capture worse

`outputs/c18_per_position/per_position_capture.json`,
`scripts/17_per_position_capture.py`. The expensive half of §15.6 is on disk — 51,200
rollouts gave `p_hit[i, j]` and the permutation nulls — so only the head's `q[i, j]` has
to be recomputed, at 85,127 processed tokens for all seven arms and six properties.

The script rebuilds script 11's 400-prefix sample from seed 7777 and **refuses to run
unless it reproduces it**: candidate ids identical, base log-probabilities equal to
0.0 absolute difference. The baseline arm then reproduces §15.6's published
`our_head_gain` to six decimal places, which is the gate on everything below.

Per-position gain in hit-rate units at one decoding position, on the 267 prefixes §15.1
uses:

| property | **baseline** (§15.6) | Platt | isotonic | bin temperature (ECE-selected) | `wide` | `focused` | `wide_focused` | head-free ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | **+0.0147** | +0.0037 | +0.0087 | +0.0084 | +0.0127 | +0.0124 | **+0.0166** | 0.1349 |
| HBD count | **+0.0045** | +0.0020 | +0.0027 | +0.0010 | **+0.0080** | +0.0048 | +0.0055 | 0.0812 |
| rotatable bonds | **+0.0031** | +0.0014 | +0.0014 | +0.0023 | **+0.0039** | +0.0026 | +0.0031 | 0.0430 |
| TPSA | **+0.0065** | +0.0036 | +0.0051 | **+0.0085** | +0.0071 | +0.0071 | +0.0078 | 0.0744 |
| cLogP | **+0.0071** | +0.0039 | +0.0058 | **+0.0105** | +0.0072 | +0.0065 | +0.0060 | 0.0770 |
| QED | **+0.0028** | +0.0014 | +0.0017 | +0.0026 | +0.0026 | +0.0020 | +0.0013 | 0.0583 |

As a share of the λ=1 optimum (the noise-corrected oracle head of §15.6):

| property | baseline | Platt | isotonic | bin temperature | `wide` | `focused` | `wide_focused` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 20.8% | 5.2% | 12.4% | 11.9% | 18.0% | 17.5% | **23.6%** |
| HBD count | 11.9% | 5.4% | 7.1% | 2.6% | **21.2%** | 12.8% | 14.4% |
| rotatable bonds | 13.5% | 6.3% | 6.1% | 9.9% | **17.0%** | 11.2% | 13.6% |
| TPSA | 21.5% | 12.0% | 16.9% | **28.2%** | 23.8% | 23.6% | 25.8% |
| cLogP | 17.9% | 9.7% | 14.6% | **26.5%** | 18.1% | 16.3% | 15.0% |
| QED | 14.8% | 7.5% | 9.0% | 13.5% | 13.4% | 10.7% | 6.8% |

**Platt and isotonic are below the uncalibrated head at every one of the six properties**,
and Platt lands within 0.0013 of the λ = α rescale it is predicted to be
(`lambda_rescale_at_the_platt_equivalent` in the artefact). Isotonic sits above Platt
because it is less aggressive, not because it is doing something different in kind.

**Neither can change which candidate is chosen, and that is measured, not assumed.**
`picks_the_best_candidate_rate` — the share of prefixes at which the head assigns its
maximum score to the candidate the rollouts say is best — is *bit-identical* under Platt
for five of six properties, and rises under isotonic only because isotonic's flat steps
create ties. Ties, not discoveries.

#### 20.3.3 The bin-logit temperature: formally distinct, practically another λ

This is the one post-hoc family that *can* reorder candidates, and it does: its
`picks_the_best_candidate_rate` differs from the baseline's for five of six properties.
It is also the only post-hoc arm that ever improves per-position capture — TPSA
+0.0065 → +0.0085, cLogP +0.0071 → +0.0105.

Sweeping the temperature explains both facts at once. Per-position gain against `T`:

| property | T=0.4 | T=0.5 | T=0.75 | T=1.0 | T=1.6 | T=3.0 | T=4.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.0385 | 0.0314 | 0.0202 | 0.0147 | 0.0084 | 0.0041 | 0.0030 |
| HBD count | 0.0114 | 0.0089 | 0.0060 | 0.0045 | 0.0027 | 0.0014 | 0.0010 |
| cLogP | 0.0157 | 0.0128 | 0.0090 | 0.0071 | 0.0047 | 0.0026 | 0.0020 |

**The response is monotone decreasing in T for all six properties**, with the best value
at the grid edge. Sharpening the bin logits sharpens `q`, which widens the spread of
`log q` across candidates, which is what raising λ does. Aromatic rings at T=0.4 gives
+0.0385; the same head at λ=2 gives +0.0324. The family is formally outside the λ
parametrisation and behaves inside it.

**And the temperature that calibrates is not the temperature that decodes.** The
ECE-selected temperature is above 1 for aromatic rings (1.6), HBD count (4.0), rotatable
bonds (1.4) and QED (1.1) — the calibration objective wants the head *flattened*, and
flattening costs the decoder. Where the ECE-selected temperature happened to fall below 1
(cLogP 0.625, TPSA 0.675) the apparent "calibration improvement" in the per-position table
is not a calibration benefit at all; it is a λ increase wearing a calibration label.

**This is the sharpest statement C18 can make.** Calibration error and decoding quality
are different objectives, they point in opposite directions for four of six properties,
and improving the first is not a route to the second.

---

### 20.4 Route (b) — a larger and a differently-targeted readout

`scripts/17_train_head_variants.py`, `outputs/c18_head_variants/head_variants_summary.json`.
Three variants, all trained on the phase-2 base-policy prefixes with the same recipe,
same seed (1234), and all deliberately loadable by the **unmodified**
`scripts/05_guided_generation.py`, so the end-to-end stage inherits every test that
already covers guided decoding.

* **`wide`** — the same two-layer MLP with `hidden_dim` 1024 instead of 256: 1.84M
  parameters against 268k, a 6.9x increase. §8.3 records the single architecture as an
  explicit limit on the negative result; this is the cheapest test of whether it binds.
* **`focused`** — a three-bin readout whose middle bin *is* the target interval, so the
  head optimises the event guidance scores instead of a 20-way distribution of which the
  target is a marginal. Built as a `QuantileBinner` with `edges = [−∞, lo, hi, ∞)`, so the
  §11.5 union-of-bins invariant holds by construction and no new binner class is needed.
* **`wide_focused`** — both.

Held-out target-interval AUROC, phase-2 test split:

| property | baseline (256, full) | `wide` (1024, full) | `focused` (256, 3-bin) | `wide_focused` | parameters, `wide` |
| --- | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.7904 | 0.7886 | 0.7814 | 0.7794 | 1,843,206 |
| HBD count | 0.7781 | 0.7780 | 0.7564 | 0.7570 | 1,844,231 |
| rotatable bonds | 0.7806 | 0.7794 | 0.7654 | 0.7566 | 1,850,381 |
| TPSA | 0.7391 | 0.7253 | 0.7149 | 0.7147 | 1,859,606 |
| cLogP | 0.7901 | 0.7920 | 0.7831 | 0.7773 | 1,859,606 |
| QED | 0.7355 | 0.7367 | 0.7271 | 0.7283 | 1,859,606 |

**Seven times the parameters buys nothing.** `wide` is within ±0.014 of the baseline
everywhere and its median change is −0.0007 — inside the ±0.0041 head-seed standard
deviation §13.2 measured. `focused` is *worse* on **all six**, most on TPSA (−0.0242) and
HBD count (−0.0217): coarsening to three bins removes the auxiliary signal a 7-to-22-way
head gets from having to place the whole distribution, and buys nothing in exchange
because the target marginal was already exact after the §11.5 fix.

**This eliminates one of §8.3's three suspects and adds a fourth that also fails.** §8.3
says the aromatic-ring negative result "shows *this* readout fails, not that the
information is absent from the state", and names three: an earlier layer, a larger head,
a different pooling. **The larger head is now tested and does not help.** The
target-focused readout was not on §8.3's list and does not help either. **Pooling is
still untested** — every head here, baseline and variant, reads the single hidden state
at one position — and the layer is C17's. Two of the four remain open and should be
described that way.

NLL and expected-value MAE are **not** comparable across these variants, because the
`focused` readout partitions the outcome space differently — the same trap §11.6 finding 4
identified. The comparable columns are the ones on the same event: target AUROC, target
Brier, target ECE. The artefact carries a note to that effect on every row.

---

### 20.5 End to end — the measurement, not the extrapolation

**This is the part that matters, and `docs/TODO.md` C22.1 is why.** Everything in §20.3.2
and §20.4 is a gain at *one decoding position* with the rest of the sequence left to the
base policy. End-to-end lift is 20–48x the per-step gain, and transferring the ratios
linearly implies hit rates above 1 for four of six properties. **A per-position
improvement is not an end-to-end improvement and none of the numbers above may be
converted into one.** So the arms were run.

`outputs/c18_summary/c18_summary.json`, assembled by `scripts/17_summarise_c18.py` from
the individual runs. Three anchors — aromatic rings (most steerable at λ=1), HBD count
(the pre-registered discriminating case), cLogP (the property the whole calibration story
was about). `unguided` + `throughout`, 3 seeds, 512 molecules per condition per seed,
frozen windows and intervals, compute-matched best-of-N solved from each run's own token
count under `actual` accounting exactly as §16.2 does.

The `uncalibrated` arm is the within-script control: `guided_sample` reseeds at every
call, so it must reproduce the central test's own numbers, and it does exactly —
aromatic rings `unguided` 0.1785, `throughout` 0.4735, against §16.1's 0.1785 / 0.4735.

Best-of-N is **not** run once per arm. It is deterministic in (property, N, seeds), and
every one of the nineteen arms solves to the same **N = 9**, with realised token ratios
0.954–1.018 (`c18_matched_best_of_n/matched_best_of_n.json`). So it is run once per
distinct N and shared, and the shared run reproduces §16.2's published value exactly
(aromatic rings 0.8294, HBD count 0.5234, cLogP 0.6107). Its token cost is also identical
across the three properties (614,271 each), which is the expected signature of a baseline
that samples unconditionally and only *selects* on the property. Running it per arm would
have added **9,828,336 processed tokens** — 75% of the entire guided budget — to recompute
three numbers. The sharing is recorded per arm and asserted by
`test_best_of_n_is_shared_because_every_arm_solves_to_the_same_n` rather than done
silently; if any arm ever solves to a different N it gets its own run automatically.

| property | arm | per-position gain | ×baseline | **`throughout`** | **lift** | **×baseline** | validity | best-of-9 | **advantage** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **aromatic rings** | uncalibrated (control) | +0.0147 | 1.00x | 0.4735 | +0.2949 | 1.00x | 0.9954 | 0.8294 | **−0.3560** |
| | Platt | +0.0037 | 0.25x | 0.2465 | +0.0680 | **0.23x** | 0.9954 | 0.8294 | **−0.5829** |
| | isotonic | +0.0087 | 0.59x | 0.3009 | +0.1223 | **0.41x** | 0.9935 | 0.8294 | **−0.5286** |
| | bin temperature | +0.0084 | 0.57x | 0.3436 | +0.1650 | **0.56x** | 0.9967 | 0.8294 | **−0.4859** |
| | `wide` readout | +0.0127 | 0.86x | 0.4532 | +0.2747 | 0.93x | 0.9954 | 0.8294 | **−0.3762** |
| | `wide_focused` readout | +0.0166 | 1.14x | **0.5007** | **+0.3221** | **1.09x** | 0.9961 | 0.8294 | **−0.3288** |
| **HBD count** | uncalibrated (control) | +0.0045 | 1.00x | 0.2988 | +0.2150 | 1.00x | 0.9980 | 0.5234 | **−0.2247** |
| | Platt | +0.0020 | 0.45x | 0.1567 | +0.0730 | **0.34x** | 0.9974 | 0.5234 | **−0.3668** |
| | isotonic | +0.0027 | 0.60x | 0.2128 | +0.1291 | **0.60x** | 0.9941 | 0.5234 | **−0.3106** |
| | bin temperature | +0.0010 | 0.22x | 0.1280 | +0.0443 | **0.21x** | 0.9967 | 0.5234 | **−0.3954** |
| | `wide` readout | +0.0080 | **1.78x** | 0.2997 | +0.2160 | **1.00x** | 0.9928 | 0.5234 | **−0.2238** |
| | `wide_focused` readout | +0.0055 | **1.22x** | 0.2847 | +0.2010 | **0.93x** | 0.9948 | 0.5234 | **−0.2388** |
| **cLogP** | uncalibrated (control) | +0.0071 | 1.00x | 0.3030 | +0.1997 | 1.00x | 0.9948 | 0.6107 | **−0.3077** |
| | Platt | +0.0039 | 0.54x | 0.2040 | +0.1007 | **0.50x** | 0.9954 | 0.6107 | **−0.4066** |
| | isotonic | +0.0058 | 0.81x | 0.2435 | +0.1402 | **0.70x** | 0.9863 | 0.6107 | **−0.3671** |
| | bin temperature | +0.0105 | 1.48x | **0.3593** | **+0.2559** | **1.28x** | 0.9967 | 0.6107 | **−0.2514** |
| | `wide` readout | +0.0072 | 1.01x | 0.3022 | +0.1989 | 1.00x | 0.9974 | 0.6107 | **−0.3085** |
| | `wide_focused` readout | +0.0060 | 0.84x | 0.2555 | +0.1522 | **0.76x** | 0.9935 | 0.6107 | **−0.3552** |

Five things, in descending order of how much they matter.

**1. No arm beats compute-matched best-of-N, anywhere.** The best advantage across all
nineteen arms — the eighteen above plus the sharpened arm of §20.5.1 — is **−0.2238**
(HBD count, `wide` readout), against the deployed head's −0.2247. R4 continues to fire. This is now established across six properties, two token
accountings, six values of λ (§19) and six head-and-calibrator arms.
`any_arm_anywhere_beats_compute_matched_best_of_n` is `false` in the artefact.

**2. Every arm that actually corrects the calibration is worse end to end, at every
property, without exception.** Platt costs 0.23–0.54x of the deployed lift and isotonic
0.41–0.70x — six of six cells, no exceptions. The ECE-selected bin temperature is worse at
two of three anchors (0.56x, 0.21x) and better at one (cLogP, 1.28x), and point 3 explains
why that one is not a counterexample. So the prediction of §20.1 — that correcting the
head's under-confidence is a λ decrease and should *hurt* — holds at **eight of the nine
calibration cells**, and the ninth is a sharpening rather than a level correction.
`docs/HANDOFF.md` §6 E2's recommendation to "fix the calibration
first, *then* re-test guidance, so 'guidance fails' is not confounded with 'the guidance
signal is broken'" is therefore not merely unnecessary; **following it would have made the
negative result worse and would have been read as evidence for it.**

**3. cLogP's bin-temperature arm is the only calibration cell above 1.00x, and it is a λ
result wearing a calibration label.** Its ECE-selected temperature is 0.625 — *sharpening*,
not the flattening calibration usually means — and §20.3.3's sweep shows the response to
temperature is monotone in the sharpening direction for all six properties. The +0.2559
lift is what a λ increase buys; it is available more directly and more controllably from
§19's λ knob, and §19 also records what that knob costs above its optimum.

**4. The best retrained readout buys 1.09x at one property and nothing at the other two.**
`wide_focused` takes aromatic rings from +0.2949 to **+0.3221**, a real 1.09x with validity
unchanged at 0.9961, and narrows the gap to best-of-N from −0.3560 to −0.3288. On HBD count
and cLogP the same readout is *worse* (0.93x, 0.76x) and `wide` is exactly neutral (1.00x,
1.00x). Averaged over the three anchors the retrained readouts buy nothing; the one gain is
property-specific and is smaller than §19's λ tuning (1.29–1.69x).

**5. And the clearest empirical demonstration of `docs/TODO.md` C22.1 this project has
produced.** Compare the two ratio columns:

| case | per-position | end to end |
| --- | ---: | ---: |
| HBD count, `wide` | **1.78x** | **1.00x** |
| HBD count, `wide_focused` | **1.22x** | **0.93x** — *sign reversal* |
| cLogP, `wide_focused` | 0.84x | 0.76x |
| aromatic rings, `wide_focused` | 1.14x | 1.09x |

A **78% per-position improvement produced a 0.5% end-to-end improvement**, and a 22%
per-position improvement produced a 7% end-to-end *loss*. The per-step decomposition of
§15.6 is sound as a decomposition and is not a predictor of end-to-end effect, exactly as
the audit said. Had C18 stopped at the per-position table it would have reported "the
readout can be improved by up to 1.78x" and been wrong.

#### 20.5.1 The decoder-optimal temperature is a worse λ than λ

`outputs/c18_guided_binT0p4_aromatic_rings/`. §20.3.3 showed the per-position gain rises
monotonically as the bin logits are sharpened, with the best value at the grid edge
`T = 0.4`. Run end to end on aromatic rings, against §19's own λ=2 run on the same
dataset, seeds and windows:

| arm | `throughout` | lift | validity | uniqueness | content length | tokens/molecule | advantage vs best-of-9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bin temperature `T = 0.4`, λ=1 | 0.5495 | +0.3710 | 0.9857 | 1.0000 | 46.09 | 426.3 | **−0.2799** |
| raw head, λ=2 (§19) | **0.5579** | **+0.3794** | **0.9909** | 1.0000 | 45.57 | 420.3 | **−0.2715** |

**Sharpening the head is a slightly worse version of raising λ**: lower hit rate, lower
validity, and more tokens per molecule. It confirms the §20.3.3 reading — the family is
formally outside the λ parametrisation and inside it in every way that matters — and it
closes the one route by which a post-hoc method might have escaped the §19 envelope.
It also still loses to best-of-9.

---

### 20.6 What C18 changes

**1. §8.2's diagnosis is retired completely, and its causal story was inverted, not merely
exaggerated.** §8.2 argued that "a head that never emits a confident probability produces a
compressed score range and a weaker effective lambda", i.e. under-confidence ⟹ weak λ.
§11.6 corrected that for the mask defect ("that is not what a dropped bin does"). C18
establishes the general case, and the arrow points the other way: under-confidence is
mostly a *level* error, the softmax over candidates is invariant to the level, and
**correcting it with a monotone map necessarily flattens `log q`, which is a λ decrease.**
Measured: correcting it costs 0.23–0.70x of the deployed lift. So the direction of §8.2's
mechanism is wrong, not just its magnitude. **This is a claim in the existing report that
C18 contradicts and it should be flagged where §8.2 is quoted.**

**2. The re-measured off-policy factor is 1.35–2.09, not 3.5, and §11.6's arithmetic
predicted it.** cLogP is 1.69 against §11.6's implied 3.5/2.08 = 1.68. On base-policy
prefixes every head is calibrated (ratios 0.85–1.11). §11.6's correction to §8.2 is now
confirmed on an independent sample rather than inferred from a checkpoint.

**3. `docs/HANDOFF.md` §6 E2 can be closed, and it should be closed as a negative rather
than as unattempted.** Its "cheap version" is, in the temperature/Platt half, *exactly* a
rescale of λ — proved algebraically, checked on the real candidate array, and demonstrated
by two full guided runs returning the same 1,536 molecules. Its isotonic half is not a λ
rescale but is monotone in `q` and so cannot change which candidate is preferred; measured,
it costs 0.41–0.70x. **Following E2's advice would have made the negative result worse and
would then have been read as evidence for it**, which is the specific failure mode E2 was
written to prevent.

**4. One of §8.3's three named suspects is eliminated and a fourth is added and also
eliminated.** The larger head is tested and does not help (±0.014 AUROC, median −0.0007,
at 6.9x the parameters); a target-focused readout — not on §8.3's list — is tested and is
worse on all six properties. **Pooling and the probe layer remain untested here** and the
layer is C17's, so §8.3 should be rewritten to name the two that survive rather than the
three it currently lists.

**5. §19.4's closing sentence — "that makes fixing the head the better bet" — does not
survive C18.** §19 measured the λ term end to end at 1.29–1.69x and left the head term
measured only per position. C18 measures the head term end to end by the two routes the
brief permits, and the best retrained-readout result across three anchors is **1.09x**
(aromatic rings, `wide_focused`), with the other two anchors at 1.00x and 0.76x for the
same readout. The
honest position after C18 is that **neither term is cheaply available**: λ is worth
1.3–1.7x and is capped by base-policy destruction (§19.1), and the head is not improvable
by calibration at all and only marginally and inconsistently by capacity or readout shape.

**6. R4 is now established across six head-and-calibrator arms per anchor — nineteen
runs in all — as well as across six properties, two accountings and six λ.** The best advantage anywhere is −0.2238. Nothing in C18 moves
the negative result, and the space of cheap explanations for it has narrowed rather than
widened.

**7. A methodological result the paper should carry.** C18 produced the cleanest available
demonstration that a per-position improvement is not an end-to-end improvement: 1.78x
per-position became 1.00x end to end, and 1.22x per-position became 0.93x — a sign
reversal. `docs/TODO.md` C22.1 established this as an arithmetic argument; it is now an
observation.

**8. And a result about probe calibration that is not specific to this project.** The
head's calibration error and its usefulness to the decoder are different objectives that
point in opposite directions. Post-hoc calibration reduced ECE by a factor of 3–6 and left
AUROC bit-identical, because it is monotone and AUROC is a rank statistic — and a softmax
over `k` candidates consumes ranks and spacings, never levels. **A miscalibrated probe used
inside a softmax over candidates does not need calibrating; it needs sharpening or it needs
to be a better ranker.** The ECE-selected temperature was above 1 — the wrong direction for
decoding — for four of the six properties.

---

### 20.7 Limitations, stated rather than defended

**One head-training seed per variant.** The baseline heads have three (§13.2, seed sd
≤0.0041 AUROC); the three C18 variants have one each, seed 1234. The `wide` differences
are inside that band, so "capacity does not help" is safe; `focused`'s −0.0217 and
−0.0242 are outside it and are safe in the other direction. A margin between two variants
under ~0.008 should not be read from this table.

**The variants inherit the baseline's training hyperparameters.** Learning rate, weight
decay, batch size, patience and epoch budget come from `configs/pilot_50k.yaml` unchanged,
which is the recipe tuned around a 20-way head. A three-bin head might prefer different
settings. Tuning per variant was deliberately not done, because a per-variant hyperparameter
search is a degree of freedom that would make "the retrained head is better" unfalsifiable;
the cost is that `focused` may be under-served rather than genuinely worse.

**The end-to-end bin-temperature arm uses the calibration-optimal `T`, not the
decoder-optimal one.** §20.3.3 shows those are different, and that the decoder-optimal `T`
sits at the sharpening edge of the grid. Running the decoder-optimal `T` end to end is
λ-tuning under another name and §19 has already established what λ-tuning is worth and
what it costs, so it was not run for every property. Where it *was* run — aromatic rings,
§20.5.1 — it is reported as a λ result rather than as a calibration result.

**Best-of-N is one run per distinct `N` rather than one per arm.** Justified by the
determinism of `bestofn.best_of_n` in (property, `N`, seeds) and checked: every arm
solves to the same `N` and the shared run reproduces the central test's published value.
`c18_matched_best_of_n/matched_best_of_n.json` records each arm's solved `N` and realised
token ratio so the match is auditable.

**Pooling is untested.** Every head here reads `h_t` at a single position. A readout over
a pooled window of hidden states is the one item on §8.3's list that neither C17 nor C18
touches.

**The off-policy prefixes come from one guided configuration.** λ=1, `throughout`, one
generation seed, 2,000 molecules per property. A head calibrated for λ=1 prefixes is not
calibrated for λ=2 prefixes, and §19's optimum is λ=2 for two of three anchors. This
matters less than it sounds, because the finding is that calibration does not transfer to
decoding at all — but it does mean the *size* of the off-policy gap is quoted at one λ.

---

---

## 21. Result 10 — the probe-layer sweep (C17)

Experiment C17 (`docs/TODO.md`, promoted second in the Week-3 list; `docs/HANDOFF.md` §6
E3). It targets §8.3 — *"Only one probe point was tested"* — and the aromatic-ring half of
CL3, §13.1. Artefacts under `outputs/c17_*`; every asserted number is bound by
`tests/test_probe_layers.py`. No molecule is generated anywhere in this section.

### 21.0 Pre-registration

**Written and committed to disk before any layer other than the final one was extracted,
trained on, or scored.** It is not revised below; §21.5 scores the executed result
against it verbatim, including where it fails.

#### 21.0.1 What is swept

`res.hidden_states` of GP-MoLFormer has **13 entries**: index 0 is the embedding output
and indices 1–12 are the 12 transformer layers. `configs/model.yaml`'s `hidden_layer: -1`
selects index 12, which is every number in the report to date.

The sweep is over **all 13 probe points**. Index 0 is a control, not a candidate
explanation — it is the token embedding of the last prefix token with no contextual
computation at all — but it is *counted in the multiplicity correction* rather than
excluded from it, because excluding a probe point after seeing it is exactly the
degree of freedom this pre-registration exists to remove.

Battery: the six properties of `properties.PREDICTED_LOCALITY_ORDER`. Input:
`frozen_state` only. `trivial` does not depend on the layer, so it is trained once per
property and reused; its values must reproduce `outputs/pilot_50k_heads_p2/`. Head seeds:
**1234 / 2345 / 3456**, the same three as phase 2, seeding initialisation as well as batch
shuffling. Dataset: `outputs/pilot_50k_p2/`, unchanged, intervals unchanged, splits
unchanged.

#### 21.0.2 Validity gate, checked before any comparison is read

Two identity checks. If either fails the sweep is reported as invalid rather than
interpreted:

1. The re-extracted states at probe index 12 must equal `outputs/pilot_50k_p2/hidden.npy`
   **exactly** (bit-identical; §11.4 says within-device determinism holds).
2. The probe-index-12 heads must reproduce §13's table — mean target AUROC and mean NLL
   per property to **4 decimal places**. That is a check on the training recipe, not on
   the model: if my per-layer trainer is not the same trainer script 03 ran, every
   cross-layer comparison below is measuring my refactor.

#### 21.0.3 Primary metric, and what happens if the metrics disagree

**Primary: held-out target-interval AUROC**, mean over the three head seeds, on the
phase-2 test split. It is the metric §13.1's claim is stated in, and it is the quantity
guided decoding consumes (`P(y ∈ I | prefix + a)`).

**Secondary: held-out NLL**, same rows, same seeds.

**If they disagree** — i.e. the AUROC criterion below fires at some layer but that layer's
NLL is worse than the `trivial` head's — the verdict is recorded as **metric-dependent**
and *neither* branch is claimed. I will not choose the metric that produces the more
interesting conclusion. Both columns are printed for every layer regardless.

#### 21.0.4 Question 1 — is the aromatic-ring crossover a final-layer artefact?

Phase 2 (§13.1): frozen state **0.7878 ± 0.0023**, trivial **0.8269 ± 0.0019**, a deficit
of **−0.0391**. Head-seed sd is ≤ 0.0041 across the whole battery (§13.2), so the sd of a
3-seed mean is ≈ 0.0024 and §13.2 licenses margins down to ~0.01.

Let `A(L)` be the mean frozen-state target AUROC for aromatic rings at probe point `L`,
and `A_triv = 0.8269` the trivial head's.

| verdict | fires when |
| --- | --- |
| **ARTEFACT** — the crossover is a fact about layer 12, not about the representation | some `L` with `A(L) ≥ A_triv + 0.010` **and** that layer's paired-bootstrap CI for (frozen − trivial) AUROC excludes 0 at the multiplicity-corrected level of 21.0.6 **and** its NLL is no worse than trivial's |
| **TIE AT THE BEST LAYER** — inconclusive | `max_L A(L)` lands in `[A_triv − 0.004, A_triv + 0.010)` |
| **REPRESENTATION** — the crossover survives the whole depth of the model; §13.1 is strengthened | `max_L A(L) < A_triv − 0.004` |

0.004 is one head-seed sd (§13.2's maximum, 0.0041, rounded down); 0.010 is §13.2's own
stated safe-margin floor and ≈ 4 sd of a 3-seed mean. The asymmetry is deliberate: the
claim currently in the report is the negative one, so overturning it is required to clear
a higher bar than confirming it.

#### 21.0.5 Question 2 — does some layer close part of the head gap?

§15.6: at λ=1 our head collects **11.9–21.5%** of what λ=1 permits. The layer is evaluated
for steering **without re-selecting on steering**, to keep the multiplicity honest:

1. For each property, `L*(prop) = argmax_L A(L)` — chosen by *prediction*, question 1's
   metric, before any steering quantity is computed.
2. Recompute `our_head_gain` and `our_head_share_of_the_lambda1_optimum` exactly as
   `scripts/12_locality_scatter.py` computes them, substituting the layer-`L*` head's
   `head_q`. Everything else — `candidate_base_logprobs`, `p_hit_<prop>`, `n_valid`,
   `null_available_<prop>` from `outputs/pilot_50k_p2_headroom/headroom_arrays.npz`, λ, ε,
   the 267-prefix capture set — is held fixed. **No new generation.**
3. **The layer sweep counts as a material attack on the head gap** iff `our_head_gain` at
   `L*` exceeds its value at probe point 12 for **at least 4 of the 6** properties **and**
   the median relative improvement across the six is **≥ 25%**. Otherwise it is reported
   as *not* an effective attack on §15.6's head term, whatever the AUROC table says.

The steering recomputation is run for **all 13 layers**, not only `L*`, because it costs
one forward pass over 3,200 extended prefixes; but only the `L*` column scores the
criterion above, and the full table is descriptive.

#### 21.0.6 Multiplicity, committed before the numbers exist

13 probe points × 6 properties = **78 comparisons**. Three rules, all fixed here:

1. **The aromatic-ring question is one pre-specified family of 13.** A "layer L beats
   trivial" claim inside it must clear a Bonferroni-corrected paired bootstrap: the
   `1 − 0.05/13 = 99.615%` interval for the AUROC gain must exclude zero. (Implemented as
   the 0.1923%/99.808% quantiles of the paired bootstrap difference, two-sided.)
2. **No isolated spikes.** For any property, "probe point L is genuinely better than the
   final layer" requires (a) `A(L) − A(12) ≥ 0.010` and (b) both neighbours `L−1` and
   `L+1` to satisfy `A(·) − A(12) ≥ 0.005`. A one-layer spike with non-improving
   neighbours is reported as noise, not as a finding. Depth curves are smooth; a genuine
   representational effect should be visible at more than one depth. (Probe point 12 has
   only one neighbour, and is the reference, so the rule is vacuous there; probe point 0
   has one neighbour and the rule is applied to that one.)
3. **The per-seed values are published for every cell** so the noise floor is legible
   rather than asserted.

#### 21.0.7 What would falsify what

- If **REPRESENTATION** fires, §13.1's crossover is strengthened: the claim becomes "no
  linear-plus-MLP readout of *any* layer of this frozen model recovers aromatic ring count
  as well as counting `c` and ring-closure tokens does", and §8.3's limitation is closed
  for this property.
- If **ARTEFACT** fires, §13.1 must be rewritten to "the final layer does not linearly
  expose ring count", which is much weaker, and rejection criterion R1 has to be
  re-evaluated (`docs/HANDOFF.md` §6 E3 says so explicitly).
- If question 2's criterion fails, the cheapest attack on §15.6's head term is spent, and
  C18 (calibration / readout capacity) carries it alone. That is a *reportable* negative
  and it is not to be softened by quoting the AUROC table instead.

---

*(Results below this line were written after the run. Everything above was on disk
before it.)*

---

### 21.1 What was run

Three steps, none of which generates a molecule.

| step | script | writes | cost |
| --- | --- | --- | --- |
| extract every probe point for the phase-2 prefixes | `scripts/16_extract_layer_states.py` | `outputs/c17_layer_states_pilot_50k_p2/layer{0..12}/hidden.npy` | one forward pass per batch |
| train a `frozen_state` head at each probe point, 6 properties × 3 seeds | `scripts/16_probe_layer_sweep.py` | `outputs/c17_probe_layers/` | CPU head training only |
| recompute the λ=1 steering quantities with each layer's head | `scripts/16_layer_steering_value.py` | `outputs/c17_layer_steering/` | one forward pass over 3,200 extended prefixes |
| depth curves | `scripts/16_layer_figures.py` | `outputs/c17_figures/` | reads artefacts only |

The dataset is `outputs/pilot_50k_p2/` exactly as phase 2 left it. `windows.json` and
`target_intervals.json` were **not** regenerated and not read for anything but their
frozen values; no property's binner, split or interval differs from §13's by construction,
because the prefix rows are the same rows.

**The extraction is a replay, not a regeneration.** `prefix_meta.csv` is written
trajectory-major and its `prefix_len` column *is* the position index script 02 extracted
at, so the `(sequence, positions)` pairs are regrouped from `trajectories.json` rather
than re-derived. The script asserts the trajectory-major ordering rather than assuming it.

#### 21.1.1 Thirteen probe points cost the tokens of one

`output_hidden_states=True` already returns all 13 entries from the single forward pass
the pipeline runs anyway, so the sweep needs **no additional processed tokens at all**
relative to the extraction script 02 performs. This is the reason C17 was cheap, and it is
stated in tokens rather than in seconds for the reason §11.7 gives.

#### 21.1.2 Provenance of the run, including a mid-sweep restart

Stated in full because a reader who sees `scripts/16_probe_layer_sweep.py` edited between
the start and the end of a sweep should assume the worst unless the timeline is shown.

1. The sweep was launched, ran for about an hour, completed roughly three of thirteen
   probe points, and was then **killed by a shell timeout in the harness running it** —
   not by a failure in the script. It had written **no** `probe_layer_metrics.json` and no
   partial results, only head checkpoints.
2. Between the kill and the relaunch the only commands run against it were
   `ls outputs/c17_probe_layers/*.pt | wc -l`, `ps` and `date`. **No metric, no AUROC, no
   NLL and no verdict from any layer was read, because none had been written.** The
   printed per-layer lines went to a pipe that the kill discarded.
3. The edit made in response was to add per-layer checkpointing (`partial_L<L>.json`,
   `partial_trivial*.json`) and a `--no-resume` switch, so an interrupted run is
   inspectable and restartable. It touches persistence only: no threshold, no metric, no
   training call and no decision rule was changed. `git diff` shows the change.
4. The relaunch **deleted the orphaned checkpoints and retrained from scratch**, detached
   from the shell so a harness timeout could not repeat the outcome.

Which layers were resumed from a partial rather than trained in one pass, and the
validity-gate outcome under the edited code, are reported in §21.2 — the gate is not
inherited from the pre-restart run, it is re-run.

The pre-registration in §21.0 was written, saved and staged in git **before the first
launch** and has not been edited since; its hash is in the git index. The original
timestamped draft is preserved unmodified at `reports/section_c17_probe_layers.md`, which is
the file whose mtime carries the ordering evidence; the text was copied into this section on
2026-07-31 without alteration, and `tests/test_probe_layers.py` now binds *this* copy to the
artefacts. The same holds for §20 and `reports/section_c18_head_fix.md`.

**Which probe points were resumed from a partial: none.** The relaunch found no
`partial_*.json` in the output directory, so all 13 probe points and the trivial baseline
were trained in a single uninterrupted pass under the edited code. The run log contains
zero `resumed probe point` lines. Total 7691.6 s of CPU head training, sharing the
machine with C18.

### 21.2 The two validity gates

Both are checked **under the edited code**, in the relaunched run, and neither is
inherited from the interrupted one.

**Gate 1 — the replay reproduces the dataset's own hidden states.** Probe point 12 of
`scripts/16_extract_layer_states.py` against `outputs/pilot_50k_p2/hidden.npy`, all
199300 rows: **bit-identical, maximum absolute difference exactly 0**. So the sweep's
reference column is literally the array every number in §13 was computed from, and no
cross-layer difference below can be an artefact of re-extraction.

The extraction took **2205784 processed tokens** for all 13 probe points. One forward pass
per batch returns every layer, so 13 separate passes would have cost 28675192 —
**13× more tokens for the same 13 arrays**. Wall time is recorded (56.6 s) but is not the
cost unit, for the reason §11.7 gives.

**Gate 2 — the per-layer trainer is script 03's trainer.** The probe-point-12 heads,
trained by `scripts/16_probe_layer_sweep.py`, against
`outputs/pilot_50k_heads_p2/head_metrics.json`:

| property | §13 AUROC | probe point 12 AUROC | Δ | §13 NLL | probe point 12 NLL | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aromatic rings | 0.7878 | 0.7878 | 0.0 | 1.0445 | 1.0445 | 0.0 |
| HBD count | 0.7799 | 0.7799 | 0.0 | 1.1071 | 1.1071 | 0.0 |
| rotatable bonds | 0.7820 | 0.7820 | 0.0 | 1.8084 | 1.8084 | 0.0 |
| TPSA | 0.7369 | 0.7369 | 0.0 | 2.7527 | 2.7527 | 0.0 |
| cLogP | 0.7913 | 0.7913 | 0.0 | 2.7154 | 2.7154 | 0.0 |
| QED | 0.7348 | 0.7348 | 0.0 | 2.7715 | 2.7715 | 0.0 |

Not "to four decimals" — **identical to every printed digit, on all six properties, for
both metrics, and for the layer-independent `trivial` head as well**. The pre-registration
asked for 4 dp; the run cleared it by a wide margin. The two gates together mean the only
thing that varies across the columns below is the probe point.

### 21.3 Result — the depth curve

Held-out target-interval AUROC, mean ± sd over head seeds 1234 / 2345 / 3456. Probe point
0 is the embedding output; 1–12 are the twelve transformer layers; **12 is the layer every
number in `pilot_report.md` uses**. `trivial` does not depend on the layer.

| probe point | aromatic rings | HBD count | rotatable bonds | TPSA | cLogP | QED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (embedding) | 0.6159 | 0.5565 | 0.5327 | 0.5406 | 0.5783 | 0.5228 |
| 1 | 0.8320 | 0.7550 | 0.7578 | 0.7355 | 0.7590 | 0.7145 |
| 2 | 0.8422 | 0.7873 | 0.7930 | 0.7512 | 0.7788 | 0.7375 |
| 3 | **0.8474** | 0.8123 | 0.8073 | 0.7568 | 0.7889 | 0.7467 |
| 4 | 0.8458 | **0.8226** | **0.8158** | 0.7586 | 0.7979 | **0.7532** |
| 5 | 0.8389 | 0.8206 | 0.8134 | **0.7595** | **0.7998** | 0.7526 |
| 6 | 0.8280 | 0.8125 | 0.8117 | 0.7565 | 0.7988 | 0.7494 |
| 7 | 0.8216 | 0.8035 | 0.8125 | 0.7509 | 0.7966 | 0.7478 |
| 8 | 0.8112 | 0.7991 | 0.8086 | 0.7467 | 0.7947 | 0.7461 |
| 9 | 0.8045 | 0.8003 | 0.8042 | 0.7462 | 0.7978 | 0.7414 |
| 10 | 0.8026 | 0.7895 | 0.7958 | 0.7431 | 0.7945 | 0.7399 |
| 11 | 0.7931 | 0.7841 | 0.7876 | 0.7422 | 0.7937 | 0.7393 |
| **12 (the probed layer)** | 0.7878 | 0.7799 | 0.7820 | 0.7369 | 0.7913 | 0.7348 |
| `trivial` prefix stats | 0.8269 | 0.7098 | 0.7431 | 0.7389 | 0.7516 | 0.7158 |

Seed spread is ≤ 0.0057 in every cell (largest: HBD count at probe point 8), consistent
with §13.2's ≤ 0.0041 and an order of magnitude below the depth effect.

The same shape in NLL (nats, lower is better; means over the three seeds):

| probe point | aromatic rings | HBD count | rotatable bonds | TPSA | cLogP | QED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.3333 | 1.3486 | 2.1708 | 2.9920 | 3.0119 | 3.0069 |
| 3 | **0.8658** | 1.0295 | 1.7360 | 2.6698 | 2.7092 | 2.7483 |
| 4 | 0.8743 | **0.9837** | 1.6975 | **2.6631** | **2.6863** | **2.7348** |
| 5 | 0.8946 | 0.9990 | **1.6954** | 2.6812 | 2.6878 | 2.7468 |
| 12 | 1.0445 | 1.1071 | 1.8084 | 2.7527 | 2.7154 | 2.7715 |
| `trivial` | 0.9001 | 1.2571 | 1.9154 | 2.7014 | 2.7863 | 2.8064 |

**AUROC and NLL do not disagree anywhere that matters**, so 21.0.3's tie-break never
fires. The AUROC argmax and the NLL argmin coincide for aromatic rings (both 3), HBD count
(both 4) and QED (both 4), and differ by one layer for rotatable bonds (4 vs 5), TPSA
(5 vs 4) and cLogP (5 vs 4) — within a region where the curve is flat to ~0.001 AUROC.
The full per-layer NLL table is in `probe_layer_metrics.json`; the rows above are the ones
the argument uses.

#### 21.3.1 The shape is the result, not the argmax

Two facts hold for **all six properties simultaneously**, which is what makes this
something other than six independent lucky maxima:

1. **The curve is unimodal with a mid-network peak.** It rises steeply from the embedding,
   peaks at probe point **3, 4 or 5** — never later than 5, never earlier than 3 — and
   then declines to probe point 12. Across the six properties and the eight probe points
   from the peak to the end there is not a single reversal larger than seed noise.
2. **The final layer is the worst probe point on the whole descending stretch**, for every
   property: probe point 12 is the minimum over probe points *peak*–12 in all six columns.
   Once the decline starts it does not stop before the output.

Two stronger statements are tempting and **false**. Both are recorded because earlier
drafts of this section made them and the artefact-binding tests caught them, which is what
those tests are for.

- *"The final layer is the worst of all twelve contextual layers, and one transformer
  block already beats twelve."* True for **aromatic rings alone** — 0.8320 at probe point 1
  against 0.7878 at 12. For the other five, probe point 1 is the *worst* contextual layer
  and the final layer is comfortably above it (HBD count 0.7550 vs 0.7799; rotatable bonds
  0.7578 vs 0.7820; TPSA 0.7355 vs 0.7369; cLogP 0.7590 vs 0.7913; QED 0.7145 vs 0.7348).
  The claim that the network ends below where it started is specific to the one property
  whose information is most nearly lexical.
- *"The final layer is the minimum over probe points 3–12."* True for five of six and
  false for **cLogP**, whose probe point 3 (0.7889) sits just *below* probe point 12
  (0.7913). cLogP rises more slowly and falls less far than the others — the same fact
  that makes it the one property failing 21.0.6's material margin in §21.4.2.

Figures: `outputs/c17_figures/auroc_by_probe_point.png`,
`nll_by_probe_point.png`, `steering_gain_by_probe_point.png`.

The natural reading is that the last layers specialise towards next-token prediction and
discard property-relevant information that is present mid-stack — a well-documented
pattern in language models, here measured rather than assumed. This report does not test
that mechanism and does not claim it; what is measured is the curve.

### 21.4 Result — question 1, scored

#### 21.4.1 The aromatic-ring crossover is a final-layer artefact

The pre-registered rule fires **ARTEFACT**, and it fires on every one of its three
required conditions rather than on the AUROC arm alone:

| condition (21.0.4) | required | measured |
| --- | --- | --- |
| best layer's AUROC over `trivial` | ≥ +0.010 | best layer **3**, **0.8474** vs **0.8269**, margin **+0.0205** |
| paired-bootstrap CI, Bonferroni-corrected to α = 0.05/13 | excludes 0 | **[+0.0142, +0.0284]**, mean +0.0210 |
| NLL at the best layer not worse than `trivial` | ≤ 0.9001 | **0.8658** |
| no-isolated-spike (21.0.6 rule 2), vs probe point 12 | both neighbours ≥ +0.005 | probe point 2 **+0.0544**, probe point 4 **+0.0580** |

**So §13.1's claim as currently written does not survive.** "The frozen state loses to
token counting for aromatic rings" is true of layer 12 and false of layers 1–5. The
correct statement is the weaker one `docs/HANDOFF.md` §6 E3 anticipated:

> **the model's final layer does not linearly expose aromatic ring count as well as
> counting `c` and ring-closure tokens does, while its middle layers do.**

This is the outcome that is inconvenient for the project, and it is reported as the
pre-registration required. Three qualifications, none of which rescues the original claim:

- The margin is **+0.0205 AUROC**, roughly five head-seed standard deviations. It is real
  but it is not large; the honest summary is "the middle of the network is modestly better
  than counting, the end of it is clearly worse", not "the representation obviously
  contains ring count".
- The crossover **still exists at the layer the pilot probed** and it still replicated
  across two independent 50k samples (§13.1). Nothing about the phase-1 or phase-2
  measurement was wrong; the *scope* of the conclusion drawn from it was.
- **R1 should be re-evaluated**, as `docs/HANDOFF.md` §6 E3 says it should if this
  happened.

#### 21.4.2 The same thing happens to TPSA, and to the other four

TPSA is the other property where §13 reports the frozen state losing to surface statistics
(on NLL, with AUROC indistinguishable). It behaves identically: best layer **5**,
**0.7595** against `trivial` **0.7389**, margin **+0.0206**, Bonferroni CI
[+0.0082, +0.0341], NLL 2.6812 against 2.7014, neighbours +0.0217 and +0.0196. So **both**
of §13.3's "the frozen state loses" cases are final-layer facts.

For the four properties where the final layer already beat `trivial`, the rule's ARTEFACT
label is vacuous — it was already satisfied at probe point 12 — so what matters there is
the size of the improvement over the probed layer:

| property | best layer | best AUROC | `trivial` | margin over `trivial` | gain over probe point 12 | neighbours support it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| aromatic rings | 3 | 0.8474 | 0.8269 | +0.0205 | +0.0596 | yes |
| HBD count | 4 | 0.8226 | 0.7098 | +0.1129 | +0.0427 | yes |
| rotatable bonds | 4 | 0.8158 | 0.7431 | +0.0727 | +0.0338 | yes |
| TPSA | 5 | 0.7595 | 0.7389 | +0.0206 | +0.0226 | yes |
| cLogP | 5 | 0.7998 | 0.7516 | +0.0482 | +0.0085 | **no — fails the material margin** |
| QED | 4 | 0.7532 | 0.7158 | +0.0374 | +0.0184 | yes |

**cLogP is the exception and it is reported as one.** Its gain over the probed layer is
+0.0085, below 21.0.6's 0.010 material margin, so by the pre-registered rule cLogP shows
**no genuine improvement** over the final layer even though its curve has the same shape.
Its neighbours do support it (+0.0066, +0.0075); it is the margin it fails, not the
smoothness. cLogP is the property whose information is most nearly preserved to the top of
the stack, which is consistent with §5.5's finding that it is the one property the frozen
state dominates at every position — but that is an observation, not a test.

#### 21.4.3 What this closes, and what it does not

§8.3 says: *"Whether an earlier layer, a larger head, or a different pooling would close
the aromatic-ring gap is untested."* Two of those three are now tested, by two experiments
that answer in opposite directions:

- **an earlier layer: yes, it closes the gap** (this section);
- **a larger head: no, it does not** — C18 measures a readout whose hidden layer is 4×
  wider (256 → 1024), which is **6.9× the parameters** (1.84M against 268k), at the final
  layer, moving seed-matched target AUROC by at most +0.0019 on any property, with
  aromatic rings still at 0.7886 against `trivial` **0.8267** — the trivial head's
  **single-seed (1234)** value, which is the seed C18's variants were trained at. §13.1's
  **0.8269** is the *three-seed mean* of the same quantity. The two differ by 0.00015,
  a thirteenth of the ±0.0019 seed sd, and the report quotes whichever matches the
  comparison being made; they are not two measurements that disagree.

Together these are stronger than either alone: the gap at layer 12 is not a capacity
limit, and the information is present five layers earlier. **The caveat that keeps this
honest: C18's width test is at the final layer only, so capacity is excluded *there*. A
capacity × depth interaction — whether a wider head at probe point 3 would do better still
— is untested by either experiment.** Pooling remains untested by both.

### 21.5 Result — question 2, scored, and it fails

#### 21.5.1 The pre-registered criterion

21.0.5 fixes the protocol: choose each property's layer by **prediction**
(`L* = argmax_L AUROC`, from §21.3, with no reference to any steering quantity), then
recompute the λ=1 steering quantities with that layer's head. Everything else is phase 2's:
the same 400 prefixes and their stored candidate token ids, the same `p_hit`, `n_valid`
and permutation nulls from `headroom_arrays.npz`, the same 267-prefix capture set, the same
λ = 1 and ε. **No generation.** The recomputation cost 76024 processed tokens for all 13
probe points, and `molecules_returned` is 0.

Consistency gates, before any comparison: at probe point 12 the recomputed `head_q`
matches the stored `head_q_<prop>` to **maximum absolute difference 0.0** on all six
properties, and the recomputed `our_head_gain` matches
`outputs/pilot_50k_p2_locality/locality_metrics.json` to **0.0**. So the layer columns
differ only by the layer.

| property | L* (by AUROC) | `our_head_gain` at probe point 12 | at L* | relative | share of the λ=1 optimum, 12 → L* |
| --- | ---: | ---: | ---: | ---: | --- |
| aromatic rings | 3 | +0.01466 | +0.01303 | -0.111 | 20.8% → 18.5% |
| HBD count | 4 | +0.00449 | +0.00470 | +0.045 | 11.9% → 12.4% |
| rotatable bonds | 4 | +0.00308 | +0.00536 | +0.737 | 13.5% → 23.4% |
| TPSA | 5 | +0.00646 | +0.00618 | -0.044 | 21.5% → 20.6% |
| cLogP | 5 | +0.00709 | +0.00547 | -0.228 | 17.9% → 13.8% |
| QED | 4 | +0.00281 | -0.00003 | -1.011 | 14.8% -> -0.2% |

**Improved on 2 of 6 properties; median relative improvement -0.077. The criterion
required ≥ 4 of 6 and a median ≥ +0.25. It is NOT MATERIAL, and 21.0.7 binds this section
to report that as the negative it is.**

So: **the probe-layer sweep is not an effective attack on the head term of §15.6.** The
cheapest available attack has been spent and it did not work. A better-predicting head is
not, on this evidence, a better-steering head.

#### 21.5.2 Why the AUROC table must not be quoted instead

The temptation here is obvious and 21.0.7 forecloses it: the AUROC table is a clean
positive and the steering table is a flat negative, and quoting the first in place of the
second would be exactly the substitution the pre-registration exists to prevent.

It is worth being precise about *why* they diverge, because the reason is itself the
finding. If the layer is chosen **after** seeing the steering numbers, the picture looks
good — aromatic rings 20.8% → 29.9% at probe point 6, HBD count 11.9% → 20.0% at 6,
rotatable bonds 13.5% → 30.4% at 3. But those layers are **not** the ones that predict
best, and picking them is selection on the outcome across 13 candidates per property. The
pre-registered protocol exists precisely to price that in, and when it is priced in the
effect disappears. The per-layer steering gains are reported in full in
`layer_steering_metrics.json` and plotted in `steering_gain_by_probe_point.png` so a reader
can see both readings.

The mechanism is visible in the numbers: target-interval AUROC is a **ranking** statistic,
while guidance consumes `log P(y ∈ I | prefix + a)` inside a softmax against `log p_base`.
A head can rank held-out prefixes better and still produce a *flatter* set of eight
candidate probabilities at a decoding position, which is what moves the sampling
distribution. QED is the extreme case: its best-predicting layer produces a per-position
gain of -0.00003, i.e. nothing at all. Improving discrimination and improving the
magnitude of the score spread are different objectives, and C17 shows they can come apart.

#### 21.5.3 What this does and does not say about end-to-end steering

Everything in §21.5 is a **per-position** quantity, with the rest of the sequence left to
the base policy. §15.6 and `docs/TODO.md` C22.1 record why per-position ratios must not be
transferred linearly to end-to-end lift.

**Had the criterion passed, the correct statement would still have been "per position
only".** It failed, so the point is moot in the favourable direction — but the converse
matters and should be stated: **this section does not prove that a mid-network head would
fail end to end.** That would need fresh guided generation at probe point 3–5, which
nobody has run and which C17 explicitly did not need. It is the obvious follow-up and it
is the honest limit of this result.

What it does establish is that the *cheap* version of the fix — swap the layer, keep
everything else — does not buy per-position steering, and that the property that
distinguishes a good probe layer (discrimination) is not the property guidance consumes
(score spread at a decoding position).

Adjacent evidence, from C18 rather than from C17, points the same way: C18's best
retrained readout bought 1.09× at one anchor and 1.00× and 0.76× at the other two end to
end, and **no arm anywhere beat compute-matched best-of-N** (best advantage −0.2238). Two
independent attacks on the head term — depth here, capacity and calibration there — both
came back approximately empty.

### 21.6 What C17 changes in the rest of this report

These are contradictions with claims made earlier in this document. Under the standing rule
that a new result which contradicts the report changes the report, each has been actioned
where it occurs; this list records what moved and why, so the change is auditable rather
than silent.

1. **§13.1 was too strongly scoped, and is now scoped to the final layer.** "That is a
   replication to three decimal places of the pilot's most-defended claim" remains true as a
   statement about layer 12, but the surrounding claim — that the frozen state does not carry
   aromatic ring count as well as token counting does — is **contradicted** at probe points
   1–5. The measurement is kept; the conclusion is restated as a fact about the final layer.
   §5.5's "a learned readout of a 768-dimensional state does not" and §8.1's "the
   specification's kill-test is met for this property" carried the same over-scope and are
   corrected in the same way.
2. **§13.3 generalised from two losers.** Both of its "the frozen state loses" cases
   (aromatic rings, TPSA) reverse at a middle layer, so the inference that a fixed
   token-statistics vector "can nearly count" those properties better than the model
   represents them does not hold of the model, only of its output layer.
3. **§8.3 is partly closed and has been updated rather than deleted.** "An earlier layer"
   is now tested and the answer is yes; "a larger head" is tested by §20.4 and the answer is
   no; "a different pooling" remains untested.
4. **Nothing in §15, §16 or §19 is contradicted.** Every steering, best-of-N, λ-sweep and
   quality result stands exactly as reported, and 21.5 adds a negative that is consistent
   with all of them.

One thing C17 explicitly does **not** contradict: the pre-registered
`PREDICTED_LOCALITY_ORDER` and the P1–P6 verdicts are untouched. C17 changes which layer
the predictability half is measured at; it does not re-run the locality scatter, and the
steering coordinate of that scatter is unchanged because 21.0.5's criterion failed.

#### 21.6.1 A note on what is and is not test-bound

Every number in §21.1–§21.5 that comes from C17 is re-read from its JSON artefact and
required to appear in this text by `tests/test_probe_layers.py`, in the style of
`tests/test_report_matches_artifacts.py`. Two of those tests are deliberately written as
*tripwires on the data* rather than on the prose — the no-isolated-spike table and the
question-2 failure — so a re-run that changes the finding fails a test instead of leaving
this section standing unchallenged. Both caught real over-claims during drafting: an
earlier version of §21.3.1 asserted that the final layer is the worst of all twelve
contextual layers and that probe point 1 beats probe point 12 for every property; both are
false for five of six, and are now recorded as false.

**The C18 figures quoted in §21.4.3 and §21.5.3 (+0.0019, 0.7886 vs 0.8267, 1.09×/1.00×/
0.76×, −0.2238) are not bound by `tests/test_probe_layers.py`.** They are C18's artefacts,
quoted with attribution, and are bound by C18's own tests. C17 does not read C18's outputs.

### 21.7 Limitations

- **One head architecture, one pooling.** Every probe point uses the same two-layer MLP on
  the last-position state. Pooling over positions is untested here and in C18.
- **Capacity × depth is untested**, as §21.4.3 says.
- **The steering result is per position.** End-to-end guided generation at a middle layer
  has not been run.
- **The layer sweep is over probe points, not over probe *methods*.** A linear probe, a
  different regulariser or a different readout depth could move the curve; only depth was
  varied.
- **The 13 probe points are one family, corrected as one.** The Bonferroni correction in
  21.0.6 covers the 13 layers within a property. Across the six properties the results are
  not independent tests of independent hypotheses — the same six-property battery is used
  throughout the project — and no further correction is applied, deliberately, because the
  claim being made is about the *shape shared by all six* rather than about any one of them.

---

## 22. What C23–C29 change in this report — the merge

Added 2026-08-03, by the owner. Sections 1–21 above were written between 2026-07-25 and
2026-07-31. Seven further pre-registered experiments were run after them and were written
up as standalone documents, each of which deliberately **filed** its disagreements with
this report instead of editing it:

| section | file | what it measured |
| --- | --- | --- |
| C23 | `reports/section_c23_layer_end_to_end.md` | guided generation with a mid-network head, end to end |
| C24 | `reports/section_c24_generality.md` | the same two claims on GPT-2 and three text attributes |
| C25 | `reports/section_c25_pooling.md` | pooled readouts, and head-seed variance |
| C26 | `reports/section_c26_n_sweep.md` | the full best-of-N compute frontier, 11 values of N |
| C27 | `reports/section_c27_head_selected_bestofn.md` | best-of-N denied the RDKit oracle |
| C28 | `reports/section_c28_k_sweep.md` | the candidate-set size `k` as a compute knob |
| C29 | `reports/section_c29_head_seeds.md` | eight head seeds, and the effective-λ confound |

This section is the merge. It is organised by **what happens to a claim**, not by which
experiment produced it, because several claims are moved by more than one. Every number
below is read from a tracked `outputs/*_summary/*.json`; the section that derived it is
named so the derivation can be checked rather than taken on trust.

**Nothing in §§1–21 has been deleted.** Where a sentence there is now wrong, it is left in
place and contradicted here, because the sequence in which the project found things out is
part of what it has to report.

### 22.1 Two headline claims that do not stand as written

#### 22.1.1 "Guidance loses to compute-matched best-of-N" — true, and ~8× smaller than reported

CL4, CL22, §16.2, §19.2 and §19.4 all state some version of this, and the gaps quoted are
-0.22 to -0.36 on `actual` accounting. **C27 shows that most of that gap is the
comparator's free access to RDKit, not guidance's weakness.** Best-of-N in §16.2 selects
its winner with `compute_properties` on the finished molecule — the true oracle. Guidance
sees only the learned head. C27 re-ran the entire frontier with best-of-N restricted to the
*same head, same probe point, same interval, same binning* that guidance uses — an
equal-information comparator — and the deployed arm's gap collapses:

| anchor | vs **oracle**-selected best-of-N | vs **head**-selected best-of-N | fraction of the gap that was the oracle |
| --- | ---: | ---: | ---: |
| aromatic rings | -0.3532 | **-0.0439** | 0.876 |
| HBD count | -0.2472 | **-0.0292** | 0.882 |
| QED | -0.3715 | **-0.0522** | 0.859 |

Counting arms rather than anchors: against the oracle curve **1 of 46** guidance arms sits
above it (C26 D1); against the head-selected curve **15 of 46** do (C27 E1). None of the 15
is a deployed configuration, which is why this re-scopes the claim rather than reversing it.

**Replacement wording.** *Guided decoding loses to oracle-selected best-of-N at matched
compute by 0.25–0.37 in hit rate, and to an equal-information best-of-N by 0.03–0.05.* The
first number bounds the claim to properties with a cheap ground-truth oracle — which the
report already said in prose (§5, CL4's "strongest objection" column) but never quantified.
The second is the number that describes the method rather than the benchmark.

> **Withdrawn as a general statement, 2026-08-03, by §25.** The replacement wording above
> is **still a single-generator, single-budget sentence**, and it was written as though it
> were neither. C33 re-ran the equal-information comparison on `gpt2_zinc_87m` and the
> deployed arm's advantage against the equal-information curve is **-0.1025 / +0.0980 /
> +0.0355** on aromatic rings / HBD count / QED — one anchor three times worse than
> "0.03–0.05", two anchors on the *other side of zero*. Neither `0.03–0.05` nor its ratio
> form survives the generator change.
>
> The reason is not the generator. Both quantities are read at whatever token budget the
> guided arm happens to occupy, and the two generators' deployed arms occupy different
> budgets: **367–419** tokens/molecule on GP-MoLFormer against **131** on
> `gpt2_zinc_87m`. What *does* replicate is the thing underneath both numbers — the
> separation between the oracle-selected and equal-information curves at **matched N** —
> which agrees across the two generators to a mean absolute difference of 0.0111, 0.0081
> and 0.0369 (§25.2). **Report the curve, not the number.** §25.6 is the merged form.

The honest caveat, which C27 states as its own limitation 4: a practitioner *has* the RDKit
oracle, it is free, and nobody would deploy a learned selector for these properties. The
head-selected curve is a **scientific control that says what the comparison was measuring**,
not a practical baseline.

#### 22.1.2 "Guidance has no compute knob" — withdrawn

C26 §C26.4.4 observed that all 46 guidance arms sat inside a 5.1–17.0% token band while
best-of-N spanned 32×, and concluded that guidance "cannot be offered more compute within
the method as specified." **C28 refutes the conclusion and keeps the observation.**

All 46 arms shared the frozen default `top_k_candidates: 8`. That hyperparameter is a
compute knob the method already has: cost is exactly `(k+1) ×` base (C28 gate G4, residual
0.0), and sweeping k over {2, 4, 8, 16, 32} spans **10.24× to 11.06×** in processed tokens
per molecule within a single strand, changing no weight and training nothing.

The knob exists and buys nothing. Over that span — **140.83 to 1447.42** processed tokens
per molecule on the deployed HBD strand — the arm moves **-0.0218** in
hit rate, and the *cheapest* setting (k = 2) is the best one, while over the identical
budgets
oracle-selected best-of-N gains **+0.6834** and head-selected best-of-N gains **+0.2922**.
Validity *decreases* with k. Five of six strands are lower at k = 32 than at k = 4.

**Replacement wording, superseded once — read §24.1.** C28's own replacement was: *guided
decoding can be offered ~10× more compute through k, and converts none of it into accuracy.*
**The second half of that is a fact about GP-MoLFormer, not about the method.** On
`gpt2_zinc_87m` the identical sweep raises raw hit rate by up to **+0.2347** on four of six
arms (C31 D3). The claim that survives **both** generators is the one that mattered anyway:

> Guided decoding can be offered ~10× more compute through the candidate-set size k, and on
> no arm of either generator does that compute convert into **advantage over best-of-N** —
> because best-of-N converts the same tokens faster. On `gpt2_zinc_87m` every one of the six
> arms loses ground from k = 2 to k = 32, by **-0.3645 to -0.6704**.

`k` is not "not a knob" and it is not "a knob that buys nothing"; it is **a knob that loses a
race**. §24.1 has the two-generator table.

### 22.2 One finding that is new, positive, and was not visible before C28

Pricing the k sweep against both frontiers, **8 of 30 guidance cells sit above the
oracle-selected best-of-N curve at their own budget** — five with seed-level t intervals on
2 df excluding zero, none extrapolated beyond the measured grid, none degenerate (validity
0.979–0.999):

| strand | configuration | k | advantage over **oracle** best-of-N | t interval excludes 0 | validity |
| --- | --- | ---: | ---: | :---: | ---: |
| A3 | hbd_count, probe point 4, λ=2 | 2 | +0.2499 | yes | 0.9941 |
| A3 | hbd_count, probe point 4, λ=2 | 4 | +0.2093 | yes | 0.9889 |
| C2 | aromatic_rings, probe point 3, λ=2 | 4 | +0.1690 | yes | 0.9792 |
| A2 | hbd_count, probe point 4, λ=1 | 2 | +0.1325 | yes | 0.9967 |
| A1 | hbd_count, probe point 12, λ=1 — **the deployed configuration** | 2 | +0.0846 | yes | 0.9993 |
| A3 | hbd_count, probe point 4, λ=2 | 8 | +0.0267 | no | 0.9935 |
| A2 | hbd_count, probe point 4, λ=1 | 4 | +0.0218 | no | 0.9967 |
| C2 | aromatic_rings, probe point 3, λ=2 | 2 | +0.0007 | no | 0.9909 |

Every one is at k = 2 or k = 4 — except A3 at k = 8, which is C26's single D2 exception.
The mechanism is not mysterious: at 136–150 processed tokens per molecule the comparator has
barely started, since best-of-N at that budget is drawing about three samples. These are
**cheap-end wins, not compute-scaling wins**, and C28 says so.

**Two things about this table that must be said next to it, not below it.** First, **six of
the eight winning cells are `hbd_count`** and the other two are `aromatic_rings`; `qed`'s
strand was run only at the deployed probe point 12 and never crosses. So this is a result
about two of three anchors, and one of those two carries most of it. Second, and in the
other direction, **A1 at k = 2 is the deployed configuration** — not a mid-network arm, not
a re-tuned λ, the exact setting §16 reports — which is the one cell in this table that
cannot be dismissed as a post-hoc pick.

> **Scope correction, 2026-08-03, from §24.** That second sentence is the strongest in this
> subsection and it is **a GP-MoLFormer statement.** On the second generator the deployed
> configuration (final probe point, λ=1) crosses on **none** of three properties, and on
> aromatic rings it is decisively below the frontier at -0.1756. What survives on both
> generators is the **λ = 2, small-k** family — and §24.2 shows that the deployed *probe
> point* crosses on the second generator too, once λ = 2. The ingredient the deployed arm
> was missing is **λ, not depth.**

**These cells were run at one probe-training seed**, which C29 measured at a hit-rate sd of
0.0142–0.0366 — larger than three of the eight margins above. **§23 has now replicated this
table across eight probe seeds.** Five of the eight rows survive with intervals excluding
zero, the deployed row comes out *larger* than shown here (+0.0846 → +0.1044), one row is
unresolved, and **C2 at k = 2 reverses sign** (+0.0007 → -0.0341, interval excluding zero on
the negative side). Read this table together with §23; the eight-seed figures there
supersede the single-seed figures here.

Put beside §22.1.2, the shape of the result is a crossing: guidance is ahead of best-of-N at
small budgets and cannot follow it up. That is a statement §§16–21 could not have made,
because neither frontier had been measured.

### 22.3 One number retracted: the 25× head-seed multiplier

`reports/section_c25_pooling.md` §C25.4 reported that head-seed variance was ~25× the
generation-seed variance, and that figure travelled into `docs/TODO.md` and into a brief.
**It is wrong, and it is retracted.** C25 divided a head-seed span by the **minimum** of
three per-cell generation-seed spans. On the arm in question those three spans are 0.0672,
0.0413 and 0.0040; 0.0040 was quoted. Using the mean gives 2.4×, the maximum 1.4×.

C29 re-measured it properly at **eight** head seeds across seven arms:

| arm | head-seed sd | ratio to pooled generation-seed sd | 95% F interval |
| --- | ---: | ---: | --- |
| hbd_count, probe point 4, λ=2 | 0.0366 | 1.71 | [0.96, 3.65] |
| aromatic_rings, probe point 3, λ=1 | 0.0284 | 1.38 | [0.77, 2.95] |
| qed, probe point 4, λ=1 | 0.0115 | 0.64 | [0.36, 1.36] |
| deployed λ=1, three anchors | 0.0142–0.0241 | 0.63–1.18 | all contain 1 |

**Every interval contains 1.** C25's 24.76 lies outside all seven.

What survives, and is worth keeping: head-seed variance is **comparable to** generation-seed
variance, it is of the same order as several effects this report treats as findings
(0.0142–0.0241 on hit rates of 0.2012–0.4685 at the deployed setting), and **it is never
reported** — not here before C25, and not in the steering literature. The transferable
sentence is *"one trained probe with error bars over generation seeds cannot distinguish the
method from its own noise"*, with no multiplier attached. C29's pre-registered depth test
(q = 1.52 / 1.61 / 0.81, band [0.5, 2]) fires on all three anchors, so this is a statement
about **learned probes generally**, not about mid-network probes specifically.

This also narrows **CL13** (§13.2, "head-initialisation variance is an order of magnitude
smaller than the effects being compared"). CL13 is a statement about *prediction* AUROC and
is unchanged as measured. It does not transfer to end-to-end hit rate, where the head-seed
sd is the same order as the effects.

### 22.4 §21's depth result: real, and about half the size C23 reported

C23 found that a mid-network head improves guided generation end to end (its "Rule A"),
reporting 15 of 15 arms positive. **C29 found a confound and re-priced it.**

The confound: `mean_head_q_spread_across_candidates` differs by probe point, by factors of
1.24–1.51. A multiplicative rescale of `log q` inside the softmax **is** a rescale of λ —
the same identity §20.3 uses for calibration (`λ·log(c·q^α) = (λα)·log q + λ·log c`, and the
softmax annihilates the constant). So part of "the mid-network head steers better" was
simply "the mid-network head steers harder."

Against a newly measured fine λ envelope (λ ∈ {1.25, 1.5, 2.5}), the three headline λ=1 arms
give **+0.0375, +0.0507, +0.0217** against raw margins of +0.0964, +0.1096, +0.0701 —
**54% to 69% of the λ=1 headline was steering strength, not depth.** Five arms change sign.
**C23's 15 of 15 becomes 10 of 15.**

Rule A nevertheless survives. Paired by head seed at matched λ over eight seeds, mid minus
deployed is +0.0941 [+0.0534, +0.1349], +0.1270 [+0.0957, +0.1584] and +0.0266 [+0.0113,
+0.0419] — 3 of 3 anchors, all intervals excluding zero, and 3 of 3 again after the
effective-λ correction.

**Replacement wording for §21.6 and §8.3:** *reading a mid-network probe point improves
guided generation end to end, by roughly half of what C23's raw λ=1 comparison suggests,
with the other half attributable to the larger `log q` spread a mid-network head produces —
which is a λ increase in disguise.*

> **Narrowed again, 2026-08-03, by §24.2.** C32 completed the depth × λ 2×2 on the second
> generator. Depth is positive on all three properties there, so Rule A's **direction**
> survives a second architecture — but it is the **smaller** factor on all three, beaten by
> λ in all six primary cells with both intervals excluding zero, and the interaction is not
> resolved anywhere. After the effective-λ correction the depth effect survives on
> `aromatic_rings` only; on `hbd_count` it goes **negative**. **C23's Rule A does not gain a
> second generator as a primary mechanism.** It gains one as a real but minority effect.
>
> C29's premise is also not universal: the spread ratio that drives the correction is
> 1.19–1.24 on the two count properties but **0.86 on `qed`**, where the mid probe steers
> *less* hard. "A mid-network probe is really a higher λ" held on 2 of 3 properties there
> and reversed on the third.

### 22.5 C23's Rule B: not dead, unresolved, and decided by the comparator

C23 reported one arm beating its own compute-matched best-of-N by +0.0760. C25 retired it on
a three-head-seed sign flip; C26 priced it on the interpolated frontier and got mean -0.0057.
**Both verdicts were under-powered.** At eight head seeds (C29 R6) the advantage over C23's
own comparator is **+0.0490 [+0.0135, +0.0845]**, 6 of 8 positive, and head seed 3456 — the
one C25's n = 3 happened to draw — is the worst of the eight.

It does not survive the harder comparators. Against the token-conservative accounting it is
+0.0246 [-0.0060, +0.0552]; against C26's corrected frontier +0.0156 [-0.0173, +0.0485].
Both contain zero.

**Rule B is decided by comparator choice, not by head seed.** That is the honest status, and
it is less flattering than "it fires" and less dismissive than "it is dead."

### 22.6 §20.3's calibration result: the algebra is general, the sign is not

C24 reproduced the λ-rescale identity exactly on GPT-2 and three text attributes — at ε = 0,
the calibrated head at λ=1 and the raw head at λ=α return the identical 1,536 sequences, all
three attributes. **What did not reproduce is the empirical premise that α < 1.**

On `mean_word_length` the fitted Platt slope is **1.6154**, because that head *over*-predicts
its off-policy hit rate (0.1159 predicted against 0.0410 observed) rather than
under-predicting it as the molecular heads do (0.076 against 0.267, CL6). Correcting an
over-prediction **sharpens** `log q`, which is a λ *increase*, and calibration accordingly
*doubles* that attribute's lift.

**Replacement wording for §20.6 item 1.** Separate the two statements: *"post-hoc
calibration of a probe consumed inside a softmax is exactly a rescale of λ by the fitted
exponent α"* is an identity and is general. *"And therefore it hurts"* is contingent on
α < 1, which is a property of the particular head and its particular off-policy gap. The
molecular slopes are 0.405–0.618 and the claim holds here; it is not a law.

This is the first measurement in the project that separates the mechanism from its sign, and
it took a second substrate to find one where the sign flips.

### 22.7 §21.3's depth curve replicates; §C17.5's methodological moral does not travel

C24 confirms the depth curve on a structurally matched 12-block GPT-2: the best probe point
is strictly before the final layer for all three attributes, the decline after the peak is
monotone, and the embedding control is near chance. **This is the most transferable positive
result the project has**, after the identity.

C23's methodological moral — that the cheap per-position steering proxy misleads and must be
replaced by an end-to-end run — **does not** travel. On text the proxy and the end-to-end
measurement agree, and following the proxy would have been correct. C23's insistence on
measuring end to end is vindicated **as a procedure and not as a prediction**: the proxy
happened to be right there, nothing in the proxy said so, and it took the expensive run to
find out.

Also against C23's suggested repair: on text the mid-network layer is the *worse* steering
layer for two of three attributes (ratios 0.6050 and 0.8621). **"Probe mid-network and
guidance improves" must be scoped to molecules until measured again.**

### 22.8 §8.3's three open suspects are now all closed

§8.3 said: *"Whether an earlier layer, a larger head, or a different pooling would close the
aromatic-ring gap is untested."*

- **an earlier layer — yes, it closes it.** §21.3: 0.8474 at probe point 3 against the
  trivial counter's 0.8269.
- **a larger head — no.** §20.4: at most +0.0019 AUROC, median -0.0007.
- **a different pooling — no.** C25: a zero-parameter 16-position mean at probe point 12
  reaches 0.8257, still below trivial. It *does* move target AUROC by +0.0221 to +0.0382 on
  five of six properties — the same order as the depth effect — so §20.4's "the readout is
  not capacity-bound" should be narrowed explicitly to **capacity**: a parameterless mean is
  not a capacity increase and it moves the number.

The distinction that matters: **nothing closes the gap at probe point 12; moving layers
does.**

### 22.9 A statistical method this report used and should not have

Every three-seed **percentile bootstrap** reported in C23, C24 and C26 has been withdrawn and
recomputed, because at n = 3 that statistic is vacuous.

The percentile bootstrap of a mean over three values is **identically [min, max]** of those
three values: P(all three resampled indices land on the minimum) = 1/27 = 0.0370 > 0.025, so
the 2.5th percentile *is* the minimum, for any three numbers whatsoever. "The bootstrap
excludes zero" was therefore never an interval statement — it was exactly "all three seeds
share a sign", a three-way sign test whose two-sided null probability is **2 × (1/2)³ =
0.25**, which cannot reject at any conventional level. It was reported as a confidence
interval and it overstated the evidence in every row it appeared in.

The replacement is a seed-level Student **t interval on 2 df** (`t₀.₉₇₅,₂ = 4.302653`), which
requires |mean| / sd > 2.48 to exclude zero, published alongside the raw per-seed values so
the reader can see the entire sample. Two of three "excludes 0" verdicts in C24 §C24.8 flipped
as a result; no pre-registered verdict moved, because the verdicts were computed from ratios
and sign counts rather than from the interval.

**This is a defect in the analysis and not in the data**, and it was found by adversarial
review of the project's own output. Bootstrap resampling over generation seeds *within* an
arm (n = 1536 molecules) is unaffected and is still used.

### 22.10 What §§1–21 said that C23–C29 leave completely untouched

Stated so the merge is not read as a general retraction:

1. **The one-step ceiling (CL14–CL16, §15).** 51,200 base-policy continuations, head-free and
   λ-free. Nothing since re-measures it and nothing contradicts it.
2. **The λ-rescale identity itself (CL30, §20.3.1)**, now confirmed on a second architecture.
3. **The inverted-U response to λ and the quality collapse above the optimum (CL25–CL29 of the
   claim inventory, §19).** C29 adds three interior grid points (λ = 1.25, 1.5, 2.5) which
   show the aromatic-rings envelope peaks nearer λ = 2.5 at 0.5601 than at λ = 2, and that
   the QED envelope is **non-monotone** between λ = 1 and λ = 1.25 — neither visible on the
   octave grid. The shape claim stands; the located optimum shifts.
4. **The interval-mask defect and its accounting (CL8–CL10, §11.5–11.6).**
5. **The device-equivalence finding (CL11, §11.2–11.4)**, which is now the reason
   `outputs/**/*.npy` is git-ignored with a caveat rather than silently.
6. **The depth curve for prediction (CL33, §21.3)** and the overturning of §13.1 (CL35).
7. **Every validity gate.** C26's published-call-signature gate reproduces bit-identically on
   hit rate and tokens (residual 0.0, all seeds); C27's gate reproduces C26 on all 33
   (anchor, N) means; C29's checkpoint identity is 0.0 across 33 tensor-by-tensor comparisons
   and **6144 of 6144 molecules identical** on the end-to-end replay. The gates are the
   strongest part of the record and none has ever failed after the §11 fixes.

### 22.11 The claim ledger after the merge

> **Two numbering systems, separated 2026-08-03.** This table mixes rows from
> `reports/ABSTRACT.md`'s **claim inventory** with rows naming **experiments** from
> `docs/TODO.md`'s register, and until now both spelled themselves `C<n>`. They collided:
> `C33` was simultaneously the claim *"every property is predicted best mid-network"* and
> the experiment *"does the oracle asymmetry replicate on a second generator?"*, and
> `C30`–`C32` were each doubly booked. The claim inventory was renamed to **`CL<n>`**
> because the experiment IDs name directories, scripts, tests and frozen
> pre-registrations that cannot be rewritten. **In this table and everywhere below,
> `CL<n>` is a claim and a bare `C<n>` is an experiment.**
>
> **This ledger is superseded twice**: by §24.5 after a second generator, and by §25.6
> after C33. The `CL4`/`CL22` row below is left in place, wrong as written, and
> contradicted at §25.6 — see the note under it.

| claim | status after C23–C29 |
| --- | --- |
| CL4 / CL22 — guidance loses to compute-matched best-of-N | **re-scoped, and the re-scoping was over-stated. Superseded by §25.6.** True against an oracle-selected comparator. The "0.03–0.05 against an equal-information one" is a **GP-MoLFormer, single-budget** figure, not a general one: on `gpt2_zinc_87m` the same deployed comparison gives **-0.1025 / +0.0980 / +0.0355** (§25.3). What is general is §25.2's curve, not either number (§22.1.1, §25.6) |
| CL13 — head-seed variance is an order of magnitude smaller than the effects | **narrowed to prediction.** False for end-to-end hit rate, where it is the same order (§22.3) |
| CL30 — calibration is algebraically a λ rescale | **stands, and generalises** (§22.6) |
| CL31 — calibration therefore hurts | **narrowed.** Contingent on α < 1; a substrate with α > 1 exists (§22.6) |
| CL32 — a wider readout buys nothing | **stands**, with "capacity" now distinguished from "pooling" (§22.8) |
| CL33 — every property predicted best mid-network | **stands, and replicates on GPT-2** (§22.7) |
| CL34 — best-predicting layer ≠ best-steering layer, per position | **stands per position; the end-to-end sign is now measured** and is positive for molecules, negative for two of three text attributes (§22.4, §22.7) |
| CL35 — §13.1's crossover is a fact about layer 12 | **stands** |
| CL36 — a per-position improvement is not an end-to-end improvement | **stands, and is now demonstrated in both directions** |
| CL37 — a probe inside a softmax is consumed as ranks and spacings | **stands**, and is the mechanism behind the effective-λ confound C29 found (§22.4) |
| C26's "no compute knob" | **withdrawn** (§22.1.2) |
| C25's 25× head-seed multiplier | **retracted** (§22.3) |
| C23's Rule A (mid-network head helps end to end) | **survives at ~half the reported size** (§22.4) |
| C23's Rule B (one arm beats its own best-of-N) | **unresolved; decided by comparator choice** (§22.5) |

### 22.12 What the project now says, in three sentences

*Written after C29, superseded by §24.4 after C31 and C32 added a second generator. Kept
because it is the version C23–C30 support on their own.*

Guided decoding with a future-property probe **beats even oracle-selected best-of-N at small
compute budgets** — 8 of 30 measured cells, five with intervals excluding zero — and
**cannot follow it upward**: the method's own compute knob spans 10× and converts none of it
into accuracy. Most of the large gap this report previously attributed to guidance was the
comparator's free access to a ground-truth oracle that guidance was denied; against an
equal-information comparator the deployed gap is 0.03–0.05 rather than 0.25–0.37. And the
variance that decides which of these numbers you get — the choice of probe-training seed —
is of the same order as the effects, is measured here at eight seeds, and is reported nowhere
in the literature this work builds on.

## 23. Result 11 — the crossing across eight probe-training seeds (C30)

Added 2026-08-03. Full section: `reports/section_c30_crossing_head_seeds.md`.
Pre-registration `outputs/c30_prereg/`, frozen before any cell existed.

§22.2 reported the project's only positive result — guided decoding above the
oracle-selected best-of-N frontier at small budgets — from cells generated with **one**
probe-training seed. §22.3 reported, from C29, that a probe-training seed moves end-to-end
hit rate by 0.0142–0.0366, which is larger than three of those eight margins. The headline
was therefore produced by exactly the protocol this report's own methodological finding
says is inadequate. **§23 is the missing replication:** the same cells, at all eight of
C29's probe seeds, against a comparator that cannot move.

### 23.1 Why the comparator is the design

C26's oracle-selected best-of-N curve **selects with RDKit and does not depend on the
head.** Held fixed across probe seeds, it makes every unit of spread in the advantage
attributable to the probe. The head-selected curve of C27 is computed too and reported as
secondary precisely because it *does* move with the probe and so cannot isolate the same
thing. **No decision rule reads it**, and a test enforces that.

### 23.2 Gates

| gate | what it required | result |
| --- | --- | --- |
| G1 | C29's seed-1234 checkpoints tensor-by-tensor identical to the ones C28 used | max abs diff **0.0**, 3 of 3 families, binner edges identical |
| G2 | at seed 1234, every cell reproduces C28 exactly | hit-rate and token residuals **0.0** on all three generation seeds, and **12288 of 12288 molecules identical** |
| G3 | `processed_tokens_actual mod (k+1) == 0` | residual **0** in all 64 cells |

G2 is the one that matters: head seed 1234 *is* C28, bit for bit, so the other seven seeds
measure the probe and nothing else.

### 23.3 The result

| cell | configuration | k | C28, one seed | mean over 8 | sd | t interval, 7 df | seeds positive |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| A3_k2 | hbd_count, pp 4, λ=2 | 2 | +0.2499 | **+0.2691** | 0.0411 | [+0.2348, +0.3035] | 8/8 |
| A3_k4 | hbd_count, pp 4, λ=2 | 4 | +0.2093 | **+0.2000** | 0.0366 | [+0.1694, +0.2306] | 8/8 |
| C2_k4 | aromatic_rings, pp 3, λ=2 | 4 | +0.1690 | **+0.1052** | 0.0557 | [+0.0586, +0.1517] | 8/8 |
| A2_k2 | hbd_count, pp 4, λ=1 | 2 | +0.1325 | **+0.1326** | 0.0265 | [+0.1104, +0.1547] | 8/8 |
| A1_k2 | hbd_count, pp 12, λ=1 — **deployed** | 2 | +0.0846 | **+0.1044** | 0.0179 | [+0.0895, +0.1194] | 8/8 |
| A2_k4 | hbd_count, pp 4, λ=1 | 4 | +0.0218 | +0.0207 | 0.0334 | [-0.0073, +0.0486] | 6/8 |
| C2_k2 | aromatic_rings, pp 3, λ=2 | 2 | +0.0007 | -0.0341 | 0.0358 | [-0.0640, -0.0042] | 2/8 |

`t₀.₉₇₅,₇ = 2.364624`. Three things to read off it:

1. **The deployed configuration replicates best.** A1 at k = 2 is `hbd_count` at probe
   point 12 with λ = 1 — the setting §16 deploys, not a mid-network arm and not a
   re-tuned λ. Its eight-seed mean is **+0.1044**
   [+0.0895, +0.1194],
   **8 of 8** seeds positive, and the smallest head-seed sd of the seven
   (0.0179). C28's single-seed +0.0846 was
   **+0.0198 pessimistic**, which is the
   opposite of the direction a selection effect produces.
2. **One cell reverses.** C2 at k = 2 goes +0.0007 →
   **-0.0341**, interval [-0.0640, -0.0042] excluding
   zero **on the negative side**, 2 of 8 positive. C28's `+0.0007` was the
   largest of eight draws from a distribution centred below zero. It is struck from the
   crossing, not annotated.
3. **One cell is unresolved.** A2 at k = 4: +0.0207
   [-0.0073, +0.0486], 6 of 8. Reported as unresolved.

### 23.4 The pre-registered verdict is UNINTERPRETABLE, and why that is reported first

C30.0.8 named three conditions that void the experiment before any decision rule is read.
One fired: *"any cell's validity falls below 0.90"*. It fires on
**1 of 56** points — **C2_k4 at head
seed 8901, validity 0.8301**.

One point in 56 voiding the other 55 is the rule as written. **That is the score and it is
recorded as the score.** It is also the least informative possible response to the data,
which is a defect in how the rule was written and not a fact about the experiment. The
rule is not rewritten after the fact.

The decision rules were computed anyway, so a reader can see what the screen voided:

| # | rule | measured | fires |
| --- | --- | --- | :---: |
| D1 | mean > 0 on ≥ 5 of 8 cells | 6 of 8 | True |
| D2 | ≥ 3 of 8 with intervals above 0 | 5 of 8 | True |
| D3 | the deployed cell survives | +0.1044 | True |
| D4 | ≥ 75% of points positive | 48 of 56 = 0.8571 | True |
| D5 | fewer than 3 positive → refuted | 6 of 8 | False |
| D6 | head-seed sd exceeds C28's own margin | 2 cells | True |

**S1, the conservative repair — post hoc, and labelled as such.** Dropping only the failing
head seed would be selecting on the outcome, so S1 drops the **entire cell, all eight**
seeds, and re-scores against the **unchanged** thresholds of 5 and 3 out of 8. The cell
dropped is `C2_k4`, whose own mean is
**+0.1052** with an interval excluding zero — **so S1
removes a cell that supports the headline.**

S1 returns **CONFIRMED**: 5 positive means and
4 intervals above zero, against thresholds of 5 and 3.

### 23.5 D6 — what C28's protocol could not have seen

D6 was written to fire on evidence about C28's *protocol*, independently of C30's sign. It
fires on two cells:

| cell | C28's published margin | head-seed sd | margin ÷ sd |
| --- | ---: | ---: | ---: |
| A2_k4 | +0.0218 | 0.0334 | 0.65 |
| C2_k2 | +0.0007 | 0.0358 | 0.02 |

For both, the spread across probe seeds exceeds the margin C28 published: those rows were
**never resolvable at one probe seed**, whatever their sign. The five that replicated have
margins 2.5–6.5× their own head-seed sd. That ratio, not the margin, is what a single-seed
protocol cannot show you.

### 23.6 A finding the screen surfaced: the probe seed moves *validity*

Not pre-registered, and reported because the validity screen exposed it. On C2 at k = 4
(`aromatic_rings`, probe point 3, λ = 2), validity falls with the advantage across seeds:

| head seed | validity | tokens/molecule | advantage |
| ---: | ---: | ---: | ---: |
| 1234 | 0.9792 | 253.53 | +0.1690 |
| 2345 | 0.9603 | 269.17 | +0.1380 |
| 3456 | 0.9036 | 311.04 | +0.0923 |
| 4567 | 0.9616 | 254.32 | +0.1426 |
| 5678 | 0.9811 | 239.62 | +0.1599 |
| 6789 | 0.9212 | 286.94 | +0.0668 |
| 7890 | 0.9030 | 322.68 | +0.0596 |
| 8901 | 0.8301 | 372.56 | +0.0133 |

The mechanism is legible: a probe that steers into invalid strings makes **longer**
molecules, which cost **more tokens**, which prices the arm against a further-along point
of the best-of-N curve. Validity, cost and advantage are one phenomenon here.

§19 established that validity collapses above λ ≈ 2. §23 adds that **at** λ = 2, which side
of that edge you land on depends on the probe seed: at `hs8901` this arm returns molecules
that are 17% unparseable, and nothing in a single-seed protocol would have shown it.
§19.1's validity figures are single-seed and should say so.

This is the strongest argument in the whole report for reporting the probe-training seed —
stronger than the accuracy argument, because a practitioner who ships this arm at that seed
ships a broken generator.

### 23.7 What §23 settles and what it does not

**Settles.** The crossing is not a probe-seed artefact. Five of eight cells replicate with
intervals excluding zero, including the deployed configuration, whose margin grows.
C28's limitation 2 is discharged for the winning cells.

> **Scope line, added 2026-08-03 from §24.** *"The crossing is not a probe-seed artefact"*
> stands unqualified. *"Including the deployed configuration, whose margin grows"* is a
> **GP-MoLFormer** statement: on the second generator that configuration crosses on none of
> three properties at λ=1, and crosses on one of three once λ=2 (§24.2). Everything in §23
> is one generator and eight probe seeds; everything in §24 is two generators and one probe
> seed. The two experiments cover different axes and neither substitutes for the other.

**Does not settle.** The pre-registered verdict is UNINTERPRETABLE and everything positive
rests on a labelled post-hoc repair. Six of the seven re-run cells are `hbd_count`; `qed`
has no winning cell at all. The 22 *losing* C28 cells were not re-run, so the question
"does a loser become a winner at another probe seed?" is open — and it runs in the
crossing's favour if anything, which is a reason to distrust the asymmetry rather than
enjoy it. The one losing cell that was re-run (A1 at k = 4) stayed a null:
-0.0081, 3 of 8 positive.

**All seven pre-registered predictions held.** That is unusual for this project — C24
falsified five of its own, C29 four of eight — and it should be read as C29 having already
characterised the probe-seed effect well enough to predict C30, not as C30 being easy.

## 24. Result 12 — a second generator, and what actually produces the crossing (C31, C32)

Added 2026-08-03, by the owner. Full sections: `reports/section_c31_second_generator.md`
and `reports/section_c32_depth_vs_lambda.md`. Both pre-registered and frozen before any
measurement; both file their conflicts rather than editing §§1–23.

Everything in §§1–23 is **one generator**, GP-MoLFormer-Uniq. §24 adds a second:
**`entropy/gpt2_zinc_87m`** — GPT-2 architecture with full softmax attention against
GP-MoLFormer's linear attention, 87.3M parameters against 46.8M, trained on ZINC with its
own tokenizer, and **SMILES**, so no serialization changed.

> **A standing constraint was lifted to run this.** The original brief said *no second
> generator*, so that a negative result could not be blamed on generator choice. The owner
> lifted it explicitly on 2026-08-03. The use here is the legitimate inverse: testing
> whether a **positive** result generalises. The deviation is recorded rather than silent.

### 24.1 What replicates, what does not, and what turns out to be the wrong claim

| claim | GP-MoLFormer | `gpt2_zinc_87m` | status |
| --- | --- | --- | --- |
| a crossing exists at small k | yes, 8 of 30 cells | yes, **5 of 30** cells | **replicates** |
| depth curve peaks before the final layer | yes, probe point 3–5 | yes, probe point **2** (all 3 properties) | **replicates as shape** |
| the *deployed* arm crosses at λ=1 | yes, +0.1044 over 8 probe seeds | **no**, on none of 3 properties | **does not replicate** |
| every crossing cell is at k ≤ 4 | yes | **no** — one at k = 8 | **does not replicate** |
| k buys no raw accuracy | yes, -0.0218 | **no**, up to **+0.2347** | **does not replicate** |
| k buys no *advantage over best-of-N* | yes | yes, **-0.3645 to -0.6704** on all 6 arms | **replicates** |

**The compute-knob claim was stated at the wrong level.** §22.1.2 said guidance *converts
none of its extra compute into accuracy*. That is true on GP-MoLFormer and false here: raw
hit rate rises with k on four of six arms. What is true on **both** is that the extra
compute never becomes *advantage*, because best-of-N converts the same tokens faster:

| arm (`gpt2_zinc_87m`) | advantage at k = 2 | at k = 32 | change |
| --- | ---: | ---: | ---: |
| hbd count deployed | +0.0317 | -0.6035 | **-0.6352** |
| hbd count mid | +0.2295 | -0.3765 | **-0.6061** |
| aromatic rings deployed | -0.1756 | -0.5401 | **-0.3645** |
| aromatic rings mid | +0.1724 | -0.2507 | **-0.4232** |
| qed deployed | -0.0809 | -0.7399 | **-0.6591** |
| qed mid | -0.0137 | -0.6841 | **-0.6704** |

Six of six arms lose ground. `k` is not *"not a knob"* and not *"a knob that buys
nothing"*; it is **a knob that loses a race**.

### 24.2 The mechanism: it is λ, not depth

C31 concluded the crossing needs *"a well-chosen mid-network probe point at λ=2… re-selected
per generator"* — but C31 ran only two corners of a 2×2, **(final probe point, λ=1)** and
**(mid probe point, λ=2)**, so depth and λ were observationally identical. C32 ran the two
missing corners. Gate first: one cell from **each** reused corner re-run through the new
code path gives hit-rate and token residuals of **0.0** and **3072 of 3072 molecules
identical**, so the 2×2 is one experiment rather than two spliced together.

Effects are half-differences, the standard factorial convention, with seed-level t
intervals on 2 df:

| property | k | depth main effect | λ main effect | interaction |
| --- | ---: | --- | --- | --- |
| hbd_count | 2 | +0.0398 [+0.0318, +0.0478] | **+0.1581** [+0.1277, +0.1885] | +0.0130 [-0.0264, +0.0523] (spans 0) |
| hbd_count | 4 | +0.0399 [-0.0198, +0.0997] (spans 0) | **+0.1360** [+0.0616, +0.2103] | +0.0192 [-0.0184, +0.0567] (spans 0) |
| aromatic_rings | 2 | +0.1612 [+0.0345, +0.2879] | **+0.1867** [+0.1554, +0.2180] | +0.0694 [-0.0002, +0.1390] (spans 0) |
| aromatic_rings | 4 | +0.1828 [+0.1542, +0.2115] | **+0.2034** [+0.1555, +0.2513] | +0.0287 [-0.0272, +0.0846] (spans 0) |
| qed | 2 | +0.0190 [+0.0044, +0.0335] | **+0.0482** [+0.0193, +0.0771] | +0.0058 [-0.0222, +0.0339] (spans 0) |
| qed | 4 | +0.0206 [+0.0067, +0.0344] | **+0.0524** [+0.0387, +0.0662] | +0.0033 [-0.0136, +0.0202] (spans 0) |

**λ beats depth in all six primary cells**, with both main effects' intervals excluding
zero in five of six. **No interaction interval excludes zero** — the two factors are
additive, and under the half-difference convention the arithmetic closes **exactly, per
generation seed**: on `hbd_count` at k = 2 the corner-to-corner total `d - a` is +0.203492,
+0.203279 and +0.186993 on seeds 101/202/303, and `depth + λ` reproduces each to the last
bit. Averaged, depth + λ = +0.0398 + 0.1581 = **+0.1979**, against a measured `d - a` of
**+0.1979**. (The two agree to 4 dp but not exactly, because the main effects average
per-seed advantages while the corner figure is priced at the aggregate budget — a 6.0e-05
difference, and the per-seed identity is the one that is exact.)

#### 24.2.1 The retraction: the deployed arm was missing λ, not depth

`hbd_count` at the **final probe point** — the deployed readout, no mid-network probe, no
re-selection — **crosses once λ = 2**:

- k = 2: **+0.1768** [+0.1304, +0.2232], validity 1.0000
- k = 4: **+0.1105** [+0.0547, +0.1663], validity 0.9954

C31 §C31.5.2's "re-selected per generator" clause is **retracted**. The replacement:
*guidance beats oracle-selected best-of-N at small budgets, and the dominant ingredient is
steering strength, not probe depth; a mid-network probe adds to it but is not required.*

**Scope this honestly — it is one property of three.** At λ = 2 the deployed arm still
fails on the other two:

| property, deployed probe point, λ = 2 | k = 2 | k = 4 |
| --- | ---: | ---: |
| hbd_count | **+0.1768** | **+0.1105** |
| aromatic_rings | -0.0582 (spans 0) | +0.0358 (spans 0) |
| qed | -0.0385 (spans 0) | -0.1617 |

### 24.3 The effective-λ correction, applied to a second generator

C29 established that a mid-network probe produces a wider spread of `log q` across
candidates, and that by the λ-rescale identity a multiplicative rescale of `log q` **is** a
rescale of λ. C32 measures the same spread ratio here and re-prices the depth contrast at
matched *effective* λ.

| property | spread ratio (mid ÷ final) | depth, raw | depth, corrected | confound share |
| --- | ---: | ---: | ---: | ---: |
| hbd count k2 lam1 | 1.2417 | +0.0268 | -0.0263 | +1.982 |
| hbd count k2 lam2 | 1.2417 | +0.0527 | -0.0197 | +1.374 |
| hbd count k4 lam1 | 1.2417 | +0.0208 | -0.0034 | +1.162 |
| hbd count k4 lam2 | 1.2417 | +0.0591 | +0.0424 | +0.283 |
| aromatic rings k2 lam1 | 1.1939 | +0.0919 | +0.0473 | +0.485 |
| aromatic rings k2 lam2 | 1.1939 | +0.2306 | **+0.1798** | +0.221 |
| aromatic rings k4 lam1 | 1.1939 | +0.1541 | **+0.1106** | +0.282 |
| aromatic rings k4 lam2 | 1.1939 | +0.2115 | **+0.1710** | +0.192 |
| qed k2 lam2 | 0.8594 | +0.0248 | +0.0299 | -0.206 |
| qed k4 lam2 | 0.8594 | +0.0238 | **+0.0239** | -0.003 |

Two results, both narrowing earlier claims:

1. **The depth effect survives correction on `aromatic_rings`, and barely on `qed`.** On
   `aromatic_rings` it keeps +0.1106 to +0.1798 with intervals excluding zero on three of
   four rows — this is the one property where depth is a substantial, corrected effect. On
   `qed` one of the two usable rows survives at **+0.0239**, which is smaller than every
   probe-seed sd C29 measured and should not be leaned on. On `hbd_count` the corrected
   effect goes **negative** on three of four rows — the confound share exceeds 1, meaning
   the correction more than eats the effect — and no interval excludes zero.
2. **C29's premise is not universal.** C29 measured spread ratios of 1.27–1.51 and a
   54–69% reduction. Here they are 1.19–1.24 on the two count properties and **0.8594 on
   `qed`** — *below 1*, so the mid probe steers **less** hard and the correction runs the
   other way. The `qed` λ=1 rows fall outside the measured envelope and are flagged rather
   than extrapolated. Median confound share **0.2823** against C29's 54–69%.

So "a mid-network probe is really a higher λ" held on 2 of 3 properties here and reversed
on the third. It is a real mechanism, not a law.

### 24.4 What the project now says, in three sentences

*Supersedes §22.12, which is kept because it is the version C23–C30 support alone.
Superseded in turn by §25.7, which replaces the "worth roughly 8×" clause below — C33
measured it on the second generator and it is not a constant.*

**Probe-guided decoding beats even oracle-selected best-of-N at small compute budgets on
two independent generators, and the ingredient that produces it is steering strength, not
probe depth** — λ beats depth in all six primary cells of a completed 2×2, additively, and
the deployed readout crosses once λ = 2. **It cannot follow best-of-N upward on either
generator**: the method's own compute knob spans ~10× and on all six arms of the second
generator converts into **-0.36 to -0.67** of *advantage*, because best-of-N converts the
same tokens faster. And two measurement practices account for most of what this literature
reports: the baseline is normally given a **free ground-truth oracle** the method is denied
— worth roughly **8×** the gap — and the **probe's training seed**, never reported anywhere,
moves end-to-end hit rate as much as the generation seed does and moves *validity* enough
that one seed in eight can ship a generator returning 17% unparseable strings.

### 24.5 The claim ledger after two generators

| claim | status |
| --- | --- |
| the crossing at small k | **holds on both generators** |
| the mechanism is λ, not depth | **measured on the second generator**; unmeasured as a 2×2 on the first |
| the deployed arm crosses | GP-MoLFormer at λ=1 (8 probe seeds); `gpt2_zinc_87m` at λ=2 on 1 of 3 properties |
| every crossing cell at k ≤ 4 | GP-MoLFormer only; one k = 8 crossing on the second generator |
| k buys no raw accuracy | **GP-MoLFormer only** — withdrawn as a general claim (§24.1) |
| k buys no advantage over best-of-N | **holds on both**, all arms |
| the oracle asymmetry (~8× the gap) | **measured on the second generator by C33 and it does not replicate as a ratio** — shares 0.4162 / undefined / 1.4386 against 0.8756 / 0.8819 / 0.8594 (§25.3). The *direction* replicates on all three anchors of both generators, and the underlying matched-N curve replicates to 0.008–0.037 (§25.2) |
| probe-seed variance ≈ generation-seed variance | GP-MoLFormer only, 8 seeds; every C31/C32 cell is **one** probe seed |
| depth curve peaks mid-network | **three architectures** (GP-MoLFormer, GPT-2/text, gpt2-zinc) |
| C23's Rule A — depth improves generation end to end | direction survives on a second generator; **minority factor on all three properties there** |
| C17's aromatic-rings margin over `trivial` (+0.0205) | **does not transfer** — +0.0003 on the second generator |

### 24.6 A dissociation this project has reported since phase 1, now on a second generator

On `gpt2_zinc_87m` the 768-d frozen-state probe scores **0.8687** target AUROC on aromatic
rings against a 21-feature prefix-counting baseline's **0.8683** — a tie, +0.0003. That same
probe produces the **largest crossing in the whole experiment**, +0.2473. A probe no better
than counting tokens is the best steerer measured.

This is CL3 and CL12's *predictable ≠ steerable* dissociation, reproduced on a different
architecture, a different corpus and a different tokenizer. It is also a warning about
probe selection: had the mid probe point been chosen by *margin over trivial* rather than by
absolute held-out AUROC, aromatic rings would have been dropped from the battery.

### 24.7 What §24 does not settle

1. **Every C31 and C32 cell is one probe seed.** §23 spent eight probe seeds on one
   generator; §24 spends one probe seed on two. Against C29's largest measured probe-seed
   sd of 0.0366: on `aromatic_rings` both main effects are resolved; on `hbd_count` the λ
   effect is (4.3×) but depth is not (1.1×); on **`qed` neither is** (1.3×, 0.5×) — and
   `qed` is load-bearing for C32's *"≥ 2 of 3"* dominance threshold. **The single most
   valuable remaining experiment is the 2×2 at 3–8 probe seeds.**
2. **Two λ levels and two depth levels locate no optimum.** "λ = 2 beats λ = 1" is not
   "λ = 2 is right", and C32 explicitly declines to claim *just crank λ*.
3. ~~**The oracle asymmetry was not re-measured on the second generator.** C27's ~8× is
   still one generator, and it is the project's most portable methodological claim, so it
   is the second most valuable follow-up.~~ **Run as C33, 2026-08-03; see §25.** The
   pre-registered replication **failed**, and the reason is that the ~8× was a ratio
   normalised at one budget. The claim survives as a curve rather than as a number.
4. **The gates are weaker on the second generator.** C31's cached-vs-recompute residual is
   **5.722e-06**, not the exact 0.0 the linear-attention backends give, because standard
   attention reduces in a different order on the two paths. It sits at 2.13e-07 of the
   state scale and inside C24's reused 2e-3 tolerance, and it is **not claimed to be
   bit-identical**. C32's own gate against C31 *is* exact (0.0, 3072/3072 molecules).
5. **Two generators is two.** Both are decoder-only transformers trained on overlapping
   chemical space with SMILES. Nothing here speaks to diffusion, to graph generators, or
   to SAFE/SELFIES serializations — the last deliberately, since the brief forbids
   changing serialization and that constraint was **not** lifted.
6. **`qed` never crosses on either generator.** The result is about two of three anchors,
   on both.

### 24.8 One pre-registration defect, disclosed and not amended

C32.0.4 wrote the closure identity as `d − a = depth + λ + interaction`. Under the
half-difference convention the effects are actually defined with, that is **arithmetically
false**: the truth is `d − a = depth + λ`, and the measured residual against the written
form is 0.1008. C32's own gate G4 caught it on the first scoring run. **The
pre-registration was not edited**; the defect is reported, and no decision rule reads the
identity, so nothing moved. Verified independently on `hbd_count` at k = 2: depth + λ =
+0.0398 + 0.1581 = +0.1979 against a measured `d − a` of +0.1978.

This is the fourth time in this project that a pre-registration or an analysis has been
found defective after the fact and reported rather than quietly repaired — the others being
the n = 3 percentile bootstrap (§22.9), C30's validity screen (§23.4), and C25's retracted
multiplier (§22.3). The record of those repairs is, at this point, a larger contribution
than several of the results they touch.

## 25. Result 13 — the oracle asymmetry on a second generator (C33)

Added 2026-08-03. Full section: `reports/section_c33_oracle_asymmetry_gen2.md`.
Pre-registration `outputs/c33_prereg/`, frozen at `2026-08-03T13:41:02Z`, before the
earliest C33 artifact at `2026-08-03T15:52:18Z`; `tests/test_oracle_asymmetry_gen2.py`
asserts that ordering and binds every number below to
`outputs/c33_summary/c33_metrics.json`.

C33 answers §24.7 item 3, which named this the second most valuable follow-up in the
project: **C27's oracle asymmetry — the finding that most of the guidance-versus-best-of-N
gap was the baseline's free RDKit oracle — rested on one generator.** It is also the claim
this report has promoted hardest. §22.1.1 called it "the number that describes the method
rather than the benchmark"; §24.4 called the ~8× "the most portable methodological claim
this project has produced".

It does not replicate as stated, and this section is the merge.

### 25.0 What was run, and the gates

`entropy/gpt2_zinc_87m` @ `f42a5a10e24c0350aeadb50865bd90a714d0b2bf`, 87,331,584 parameters,
all frozen. C31's three anchors, C31's generation seeds 101/202/303, C31's heads read off
disk at probe point 12, C31's frozen intervals and windows, C26's N grid and disjoint-group
estimator, a 16,384-molecule pool per seed. **No head was trained, no weight changed, no
interval re-derived, and all 30 C31 k-sweep cells were re-priced rather than re-run.**

All three blocking gates pass, so the headline is stated rather than withheld:

| gate | criterion | residual |
| --- | --- | --- |
| **G1** (blocking) | the regenerated `oracle_selected` arm reproduces C31's own curve at every grid point and seed | max abs hit-rate residual **0.0**, max abs token residual **0.0**, 99 cells |
| **G3(a)** (blocking) | tokens per returned molecule identical across arms at every N | **0.0** |
| **G6** (blocking) | the frozen interval is byte-for-byte C31's | passes, hashes stable |
| G2 | head provenance, parameter-level | passes; 5/5 C31 deployed k cells agree per anchor |
| G4 | all arms and C31 agree at N = 1 | **0.0** |
| G5 | the head arm is not near chance | pool AUROC 0.9028 / 0.8733 / 0.8067 |

**Ten predictions were scored; 8 confirmed, 2 falsified.** The two falsified are Q4 and Q5
— the two that carry the replication claim. Q5 was flagged in the pre-registration as the
one held most weakly and failed for the reasons named there. **Q4 was not flagged as weak,
and it failed.**

### 25.1 The verdict

> **DOES NOT REPLICATE.** On generator 2 the oracle is not worth at least half of the
> deployed arm's reported gap on every anchor where the share is computable.

Decision rule F1 does not fire, because `aromatic_rings` returns a share of **0.4162**,
below the pre-registered 0.50. F2 does not fire either: neither computable share falls in
C27's [0.75, 1.00] band, and they miss it **from opposite sides**.

The verdict is accounting-dependent and C33 says so up front. Under the pessimistic
accounting S1 — which charges head scoring a full re-read of every candidate, an accounting
C33.0.3 argues is wrong — F1 *would* fire and the verdict would be REPLICATES IN
DIRECTION, NOT IN MAGNITUDE. **Under neither accounting does F2 fire, so no accounting
reproduces C27's magnitude on generator 2.**

### 25.2 What replicates, and it is the part worth keeping

The two best-of-N curves — oracle-selected and equal-information — are identical at N = 1
by construction and separate as N grows. **That separation, read at matched N, is what
travels across generators.** Generator 1 is C27's; generator 2 is C33's; the two were
measured independently, on models sharing no weights, no tokenizer, no training corpus and
no attention mechanism:

| N | arom. g1 | arom. g2 | HBD g1 | HBD g2 | QED g1 | QED g2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0347 | 0.0200 | 0.0177 | 0.0186 | 0.0331 | 0.0344 |
| 3 | 0.0828 | 0.0505 | 0.0456 | 0.0476 | 0.0791 | 0.0846 |
| 4 | 0.1278 | 0.0855 | 0.0736 | 0.0775 | 0.1257 | 0.1329 |
| 6 | 0.2134 | 0.1478 | 0.1352 | 0.1412 | 0.2267 | 0.2292 |
| 8 | 0.2688 | 0.2038 | 0.1870 | 0.1983 | 0.3069 | 0.3175 |
| 9 | 0.3024 | 0.2282 | 0.2172 | 0.2289 | 0.3534 | 0.3630 |
| 12 | 0.3496 | 0.2889 | 0.2801 | 0.2965 | 0.4388 | 0.4555 |
| 16 | 0.3792 | 0.3442 | 0.3491 | 0.3591 | 0.5344 | 0.5327 |
| 24 | 0.3839 | 0.3722 | 0.4237 | 0.4502 | 0.6334 | 0.6229 |
| 32 | 0.3724 | 0.3764 | 0.4415 | 0.4750 | 0.6836 | 0.6597 |

Mean absolute difference between the generators, over all eleven grid points: **0.0111**
(HBD count), **0.0081** (QED), **0.0369** (aromatic rings). By N = 32 the oracle is worth
**0.37 to 0.68 hit-rate points** on both generators, which is **0.37 to 0.71 of the
baseline's own hit rate at that budget** (0.3739 / 0.4763 / 0.7119 on generator 1;
0.3836 / 0.5024 / 0.6838 on generator 2). The two units are close here only because the
oracle arm is near saturation at N = 32; they are stated separately because they are not
the same quantity and an earlier draft of the paper conflated them.

The curve is monotone in N for `hbd_count` and `qed` on both generators. For
`aromatic_rings` on generator 1 it peaks at N = 24 (0.3839) and dips to 0.3724 at N = 32,
so the correct verb is **grows**, not *grows monotonically*.

This is the figure `outputs/paper_figures/fig1_oracle_gap_vs_n.png` draws, and it is the
statement §22.1.1 should have made in the first place: *the more compute you give
best-of-N, the more of its advantage is the answer key rather than the search.* It is a
claim about the comparison, it does not depend on where any guided arm sits, and it is the
form in which the finding survives.

### 25.3 What does not replicate: the single-number share

C27 summarised §25.2 as a **share** — the fraction of the deployed arm's gap that the
oracle accounted for — and reported 0.876 / 0.882 / 0.859. On generator 2, at the
pre-registered matching cell (the deployed arm at k = 2, one cell per anchor, fixed in
advance and **not** selected from the 30 afterwards):

| anchor | adv vs oracle curve | 95% t interval, 2 df | adv vs equal-info curve | 95% t interval, 2 df | share g2 | share g1 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| aromatic_rings | **-0.1756** | [-0.2621, -0.0890] | **-0.1025** | [-0.1800, -0.0248] | **0.4162** | 0.8756 |
| hbd_count | **+0.0317** | [-0.0045, +0.0678] | **+0.0980** | [+0.0657, +0.1303] | **undefined** | 0.8819 |
| qed | **-0.0809** | [-0.1297, -0.0321] | **+0.0355** | [-0.0128, +0.0837] | **1.4386** | 0.8594 |

Three things must be read together here.

1. **`aromatic_rings` kills F1 at 0.4162.** Removing the oracle from best-of-N moves the
   deployed arm from -0.1756 to -0.1025. It still loses, by a margin whose interval
   excludes zero. The oracle explains 42% of that gap, not 88%.
2. **`hbd_count`'s share does not exist, as pre-registered.** Its advantage against the
   oracle curve is **positive** — the arm is already above the baseline — so the
   denominator's sign is wrong and the cell is left undefined rather than imputed or
   sign-flipped. C33.0.6 rule 5 and prediction Q7 committed to exactly this in advance;
   the empty cell is a scored prediction, not a suppressed result. Its interval also spans
   zero, so the anchor is **unresolved at three generation seeds** on top of having no
   share.
3. **`qed`'s 1.4386 is a reversal and is not clipped** — equalising the information does
   not shrink this gap, it flips its sign. But the numerator's interval `[-0.0128,
   +0.0837]` spans zero, so the reversal is itself unresolved and the share inherits that.
   The honest quotation is not "the oracle was worth 144% of the gap" but "on `qed` the
   deployed arm is at or slightly above the equal-information curve, and three seeds
   cannot say which."

**The direction survives on all three anchors of both generators**: equalising the
information always makes the deployed guided arm look better, never worse. The magnitude
does not.

### 25.4 Why the share moved, and it is not the generator

The share is a ratio whose denominator is `gap(b)` — §25.2's curve read at whatever token
budget `b` the guided arm happens to occupy. The two generators' deployed arms occupy
different budgets, and nobody chose that:

| | generator 1 | generator 2 |
| --- | ---: | ---: |
| tokens/molecule at N = 1 | 44.44 | 36.04 |
| deployed-arm budget, arom. / HBD / QED | 419.33 / 401.62 / 367.27 | 131.37 / 130.78 / 131.79 |
| that budget in units of N | **9.44 / 9.04 / 8.26** | **3.64 / 3.63 / 3.66** |
| `gap(b)` there | 0.3093 / 0.2180 / 0.3193 | **0.0731 / 0.0664 / 0.1163** |

Generator 1's deployed arms sit at N ≈ 8–12, where the two curves have separated by
0.22–0.35. Generator 2's sit between N = 3 and N = 4, where they have barely separated at
all. **The denominators are 3 to 5 times smaller, and the numerators moved too.** The two
generators were never compared at the same point on the compute axis, and a ratio
normalised at one budget is not portable to a model whose arms sit at another.

C33.0.9 raised this before the run, which is why the pre-registration required the raw gap
to be reported alongside the share everywhere. That requirement is what makes the failure
diagnosable instead of merely negative.

**The failure is also local to the cheapest cell.** On `aromatic_rings`, k = 4 through
k = 32 give shares 0.9678, 0.8474, 0.7600, 0.6969, and `qed` gives 0.9895, 0.8848, 0.9112,
0.8916 — squarely in C27's band. The anchor that fails F1 fails it **only at k = 2**, which
is the cell the pre-registration fixed as the headline. C33 reports that rather than moving
the headline, because moving it after seeing the numbers is the thing a pre-registration
exists to prevent.

### 25.5 The arm count replicates cleanly, and it is the portable form

| curve the cells are priced against | generator 1 (C27, 46 cells) | generator 2 (C33, 30 cells) |
| --- | ---: | ---: |
| cells above `oracle_selected` | 1 | **7** |
| cells above `head_selected` (equal-information) | 15 | **18** |

F3 fires: `18 > 7`. Restoring information symmetry more than doubles the number of guided
configurations sitting above the comparator at their own budget, on both generators. Note
these are **point-estimate** counts, the convention C27 used; §24 counts *crossings*, which
additionally require the seed interval to exclude zero and the validity floor to be
cleared, and gives 5 of 30 on generator 2. The two counts are not interchangeable and this
report does not mix them in one sentence.

### 25.6 What C33 changes in this report — the merge

Unlike §22 and §24, this merge has a short list, because C33 touches exactly one claim
family. **Every item below is an edit that has been made, not a disagreement filed.**

1. **§22.1.1's "0.03–0.05 against an equal-information comparator" is withdrawn as a
   general statement** and marked as such in place. It is a GP-MoLFormer, single-budget
   figure. The generator-2 values at the matching cell are -0.1025 / +0.0980 / +0.0355.
2. **§22.11's `CL4`/`CL22` ledger row is superseded** and now points here. It stated the
   re-scoped gap as a general range; it is not one.
3. **§24.4's "worth roughly 8× the gap" is withdrawn**, superseded by §25.7.
4. **§24.5's oracle-asymmetry row is updated** from "not re-measured" to "measured and does
   not replicate as a ratio".
5. **§24.7 item 3 is closed**, and closed in the direction that costs the project its most
   promoted methodological claim.
6. **C27's headline must be scoped to generator 1 wherever it appears.** "The oracle is
   worth ~85–88% of the reported gap" is a generator-1 statement. Any text stating the
   share without naming the generator is now wrong.
7. **The portable form of C27's claim is the matched-N curve (§25.2) and the arm count
   (§25.5), not the share.** Both replicate. That is what should be promoted, and the
   paper draft's §3 was restructured around it.
8. **The gap should be reported wherever the share is.** `gap(b)` is defined for every cell
   including the crossing ones; the share is not.
9. **Nothing else moves.** C33 re-ran no C31 cell and changed no C31 number — prediction
   Q10, confirmed: the newest C31 artifact predates the C33 pre-registration freeze.
   §§23–24's crossing results, the λ-versus-depth 2×2, and every probe-seed result are
   untouched, because C33 measures the comparator and not the guidance.

### 25.7 What the project now says, in three sentences

*Supersedes §24.4, which is kept because it is the version C23–C32 support alone. The
change is confined to the third sentence.*

**Probe-guided decoding beats even oracle-selected best-of-N at small compute budgets on
two independent generators, and the ingredient that produces it is steering strength, not
probe depth** — λ beats depth in all six primary cells of a completed 2×2, additively, and
the deployed readout crosses once λ = 2. **It cannot follow best-of-N upward on either
generator**: the method's own compute knob spans ~10× and on all six arms of the second
generator converts into -0.36 to -0.67 of *advantage*, because best-of-N converts the same
tokens faster. And two measurement practices account for most of what this literature
reports: **the baseline is normally given a free ground-truth oracle the method is denied,
worth 0.37 to 0.68 hit-rate points by N = 32 on both generators and growing with N — a
curve, not the single ~8× ratio this report promoted and C33 falsified** — and the
**probe's training seed**, never reported anywhere, moves end-to-end
hit rate as much as the generation seed does and moves *validity* enough that one seed in
eight can ship a generator returning 17% unparseable strings.

### 25.8 What §25 does not settle

1. **Nothing here revisits generator 1.** C33 does not re-run C27 and cannot say whether
   C27's own shares would move under a larger seed count. The 0.8756 / 0.8819 / 0.8594
   figures are three generation seeds each, like everything else here.
2. **Two anchors, not three, have a computable share.** The verdict rests on
   `aromatic_rings` and `qed`.
3. **Three generation seeds.** Every interval is a Student *t* interval on 2 df with
   t = 4.302653. No bootstrap is computed anywhere, for §22.9's reason.
4. **The head is scored at the terminal content token**, the easiest question it can be
   asked. On generator 2 the resulting near-oracle worry does not arise — the head arm is
   far *below* the oracle arm at every N ≥ 2, flat from N = 8 on `qed` while the oracle arm
   climbs from 0.5776 to 0.9648. The head can rank three candidates and cannot find the
   best of thirty-two.
5. **The budgets were never matched across generators**, and matching them would need a
   generator-1 guided arm roughly three times cheaper than any this project has run.
   §25.2's matched-N comparison is the closest available control and is not a substitute
   for one.
6. **The pre-registration has four disclosed defects**, none of which moves a number:
   §C33.0.6 miscounts the crossing cells as 5 (the point-estimate count is 7, and one of
   them is a deployed arm); rule 1 guards the *sign* of the denominator but not its
   magnitude, so two cells with denominators a hair below zero produce shares of 20.1802
   and 8.2766 — both caught by rule 3's interval flag, so the pre-registration is
   self-repairing rather than broken; and two runner defects were found and fixed **before**
   the measurement run and are disclosed because "the code compiled" is not "the code was
   correct". **The pre-registration was not amended.** This is the fifth time in this
   project that a pre-registration or analysis has been found defective after the fact and
   reported rather than quietly repaired.
