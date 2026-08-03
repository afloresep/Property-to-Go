# Next steps

Written 2026-07-30, updated after decisions A1 and A7.

**Deadline status: confirmed, and deliberately not binding.** The 2026-08-29 workshop
deadline is real, but the project owner has decided not to optimise for it — the work can
wait for a later workshop cycle or go to a journal. TMLR is the natural target: rolling
submission, no page limit, and its two stated acceptance criteria are *"are the claims
supported by accurate, convincing and clear evidence"* and *"would at least some
individuals in TMLR's audience be interested"* — novelty and significance are explicitly
**not** acceptance criteria, which fits a careful negative-or-mechanistic result well.

What this changes: nothing about the priority order, which is dependency-driven and still
correct. What it does change is that items previously marked "cut if time is tight" are
now in scope, and **no result should be rushed to fit a date**. It does *not* license
open-ended scope growth — section E still applies, because unbounded time is its own
failure mode.

Ordering is by dependency, not importance. Items marked **[you]** need a human decision or
an account only you have; everything else an agent can do.

---

## A. Before any compute — verification and decisions

These are cheap and they gate everything downstream.

- [x] **A1 [you]** ~~Verify the submission deadline~~ **Done 2026-07-30.** Confirmed, and
      deliberately not treated as binding. Candidates remain **ICBINB-BIO** at NeurIPS 2026
      (8pp full / 4pp tiny, explicitly solicits negative results and failure analyses) and
      **ML4Molecules 2026** (5pp, explicitly solicits "negative results, careful ablations,
      and rigorous baselines"), with **TMLR** as the no-deadline archival target.
- [ ] **A2 [you]** Decide the venue once the phase-2 data exists, not before. The choice
      now depends on the result: if the locality thesis holds it is a mechanistic paper and
      TMLR or ML4Molecules fit; if it fails it is a careful negative result and ICBINB-BIO
      is the better home. Deferring this is correct rather than lazy.
- [ ] **A3** Verify the PMO logP-exclusion quote against the published PDF
      (Gao, Fu, Sun & Coley, NeurIPS 2022 D&B, arXiv:2206.12411). It is load-bearing for
      the cLogP defence in `LITERATURE.md` §2 and will appear in the paper verbatim.
- [ ] **A4** Read the GP-MoLFormer-Sim camera-ready properly (arXiv:2506.05628, AAAI 2026,
      **verified to exist**) and write the differentiation paragraph. Same base model,
      training-free test-time steering — any reviewer who knows GP-MoLFormer will ask how
      this differs. Also check whether it runs any sampling baseline we claimed it does not.
- [ ] **A5** Open the still-unverified arXiv IDs by hand and drop any that do not resolve.
      **Verified already:** 2605.10831 (SLIM), 2605.06303, 2606.24952, 2506.05628,
      2607.26391. **Still `[LEAD]`:** 2604.13068, 2605.05715, 2605.25151, 2604.02608,
      2604.15557 (the detection-vs-steering cluster), 2605.13101 (margin-calibrated
      classifier guidance), the Steering Vector Fields bioRxiv preprint, and the CAST
      OpenReview submission. Do not cite one that cannot be opened.
- [ ] **A6** Fresh arXiv sweep of cs.LG and q-bio.BM in the week before submitting.
      Q-Steer was posted one day before our literature scan ran. This subfield moves in
      weeks.
- [x] **A7 [you]** ~~Confirm the property decision~~ **Decided 2026-07-30: DEMOTE cLogP,
      do not drop it.** It stays as one of six properties, out of the title, and does not
      carry the headline number. It is retained for continuity with the executed pilot and
      because it is a clean stress test of the bounded-interval defence against PMO. QED
      (an actual PMO task) and TPSA / HBD / rotatable bonds (standard MPO components) carry
      credibility with a drug-discovery reader instead.

---

## B. Setup gates on the RTX machine — do not skip, do not reorder

Full detail in `KICKOFF_PROMPT.md` §2.

- [x] **B1** ~~Clone, `uv venv && uv pip install -e .`, confirm 183 tests pass.~~
      **Done.** 183 passed on CPU before any phase-2 edit. torch 2.4.1+cu121, RTX 4090.
- [x] **B2** ~~Set `device: cuda`.~~ **Done**, `dtype: float32` left alone.
- [x] **B3** ~~Run `tests/test_model_contracts.py` on GPU.~~ **Done, 14/14 pass**, including
      both load-bearing contracts. Two tests called `.numpy()` on CUDA tensors and were
      fixed; one real library bug (`guidance.TargetScorer` held a CPU head) was fixed.
- [x] **B4** ~~Regenerate and diff the frozen artefacts.~~ **Done — GATE FAILED, written up
      in `pilot_report.md` §11.2.** `windows.json` is identical; `target_intervals.json`
      **moved**, because CUDA's RNG stream draws a different 50k sample at the same seed.
      Not numerics: logits agree to 1.4e-05 and the top-8 candidate set is identical.
      Remedy: `--inherit-intervals`, plus a permanent test pinning the frozen values.
- [x] **B5** ~~Reproduce one known number end to end.~~ **Done, gate passes.** Bit-identical
      reproduction is impossible for the reason above, so the comparison is distributional
      over three seeds: guidance effect +0.3001 (CPU) against +0.3123 (GPU), z = 0.56 and
      1.11 on the two conditions. `pilot_report.md` §11.3.
- [x] **B6 (unplanned)** A second defect surfaced while working out B4's remedy and it
      affected the pilot as executed: the target interval was **not a union of bins**, so
      the cLogP head predicted a 0.050-mass event for a 0.100 target. Fixed, quantified,
      and written up in §11.5 and §11.6. It explains the pilot's reported cLogP
      "calibration failure" and leaves its AUROCs intact.

---

## C. Phase-2 experiments

Hypothesis, operationalisation and pre-registered predictions P1–P6 are in
`LEXICAL_LOCALITY.md`. Compute is not the bottleneck; engineering time and RDKit edge
cases are.

> **This list is the experiment register, and `C<n>` here means an experiment.** It is a
> different numbering system from the **claim inventory** in `reports/ABSTRACT.md`, which
> was renamed to **`CL<n>`** on 2026-08-03 because the two had collided outright: `C33`
> meant both *"every property is predicted best mid-network"* (a claim) and *"does the
> oracle asymmetry replicate on a second generator?"* (this register's last entry). The
> experiment IDs stayed put because they name `outputs/c33_*/`, `scripts/27_c33_*.py`,
> `reports/section_c33_*.md`, `tests/` and frozen pre-registrations.

> **Register status, 2026-08-03: C1–C33 are all complete.** C30–C33 were added after the
> Week-3 list below was written and are recorded here rather than interleaved, because
> each was proposed by the section that preceded it:
>
> - **C30** — the crossing at eight probe-training seeds. `reports/section_c30_crossing_head_seeds.md`,
>   `tests/test_crossing_head_seeds.py`, merged at `pilot_report.md` §23. Pre-registered
>   verdict UNINTERPRETABLE (1 of 56 points below the 0.90 validity floor); the crossing
>   holds on 8/8 seeds at the deployed configuration, one cell reverses and is withdrawn.
> - **C31** — the whole pipeline again on a second generator, `entropy/gpt2_zinc_87m`.
>   `section_c31_second_generator.md`, `tests/test_second_generator.py`, merged at §24.
>   The crossing replicates; "k buys no raw accuracy" is withdrawn as a general claim.
> - **C32** — depth × λ as a completed 2×2. `section_c32_depth_vs_lambda.md`,
>   `tests/test_depth_vs_lambda.py`, merged at §24.2. **It is λ, not depth**, additively,
>   and C31's "re-selected per generator" clause is retracted.
> - **C33** — does C27's oracle asymmetry replicate on generator 2?
>   `section_c33_oracle_asymmetry_gen2.md`, `tests/test_oracle_asymmetry_gen2.py`, merged
>   at §25. **DOES NOT REPLICATE** as a ratio (0.4162 / undefined / 1.4386 against
>   0.8756 / 0.8819 / 0.8594). The matched-N curve underneath it replicates to
>   0.008–0.037, and that is the form the claim now takes. This closes §24.7 item 3, the
>   last named follow-up in the project.
>
> Figures for the paper: `scripts/28_paper_figures.py` → `outputs/paper_figures/`, bound
> by `tests/test_paper_figures.py`.

### Days 1–2 — lock the design

- [x] **C1** ~~Confirm `PREDICTED_LOCALITY_ORDER` is committed before any new guidance run.~~
      **Done**, and now pinned as a literal by
      `tests/test_properties.py::test_the_pre_registration_is_pinned_literally`, so editing
      it to match data fails a test rather than passing quietly.
- [x] **C2** ~~Wire in the four staged properties.~~ **Done.** In `ALL_PROPERTIES`; both
      counts in `bestofn.INTEGER_PROPERTIES`, and that set is now *derived* from
      `properties.DISCRETE_PROPERTIES` by a test rather than trusted to memory;
      `target_interval_rule` entries added as **rules** so the targets could be committed
      before the four new base distributions had been seen. `target_intervals.json` was
      written before any guided result.

### Week 1 — cheap breadth

- [x] **C3** ~~Compute the four new properties on the molecules and rollout bank you already
      have.~~ **Done, with one correction to the plan**: the pilot's `rollout_bank.json`
      stores property *values* and not the rollout SMILES, so the new properties could not
      be added to it after the fact. A new bank was generated instead
      (`pilot_50k_p2_rollouts`, 25,600 continuations, 84 s on the GPU).
- [x] **C4** ~~Train frozen-state heads, 2–3 head seeds each, plus the trivial head for all
      six.~~ **Done, 3 seeds** (1234/2345/3456), seeding initialisation as well as
      shuffling — which the pilot did not. Seed spread turns out to be ≤0.004 AUROC, an
      order of magnitude below the effects being compared, so §8.6 is retired.
- [x] **C5** ~~Predictability-vs-position curves and one locality score per property.~~
      **Done**; `pilot_report.md` §13.

### Week 2 — the central test

- [x] **C6** ~~**Steering headroom**.~~ **Done**, `scripts/11_steering_headroom.py`, 400
      prefixes x 8 candidates x 16 rollouts. Implemented via
      `generation.continue_from_prefixes` as planned, so `test_candidate_backends_agree`
      still covers the numerics. One thing the plan did not anticipate: `max - min` over
      k noisy means is **biased upward**, and biased most where rollout variance is
      highest — which is aligned with the very axis under test. A permutation null was
      added, in both property and probability units, and fixed before any headroom number
      was computed (`LEXICAL_LOCALITY.md` §3.1).
- [x] **C7** ~~Fraction of available headroom captured, per property and per quartile.~~
      **Done**, in probability units so that "achieved" and "ceiling" are commensurable.
- [x] **C8** ~~Guidance + matched best-of-N for all six properties; the scatter.~~ **Done**;
      `pilot_report.md` §15–16.
- [x] **C9** ~~HBD count is the discriminating case.~~ **Done**; verdict in §16.

### Week 3 — depth, on anchors only. **Reordered by the phase-2 measurement.**

Phase 2 decomposed the capture loss (`pilot_report.md` §15.6): λ=1 permits 32–53% of the
head-free ceiling, and our head collects 12–22% of what λ=1 permits. So **the head is worth
5–8x and λ is worth at most ~2x**. C17 and C18 are now the high-value items and C10 is not,
which reverses this section's original ranking and the simulated reviewer's.

- [x] **C18 (promoted to first, DONE 2026-07-30)** ~~Fix the head. Two cheap routes, both
      untried: post-hoc calibration on held-out guided prefixes, and simply a larger /
      differently pooled readout.~~ **Done; `pilot_report.md` §20. Both routes fail, and the
      first fails algebraically rather than empirically.** A power calibration `g(q) = c·q^α`
      is *exactly* a rescale `λ_eff = λα` — verified end to end: at ε=0 the calibrated head
      at λ=1 and the raw head at λ=α return **the same 1,536 molecules**. Every fitted Platt
      slope is < 1 (0.405–0.618), so correcting the head's under-confidence **is a λ
      decrease**, and §19 already measured what that costs: Platt 0.23–0.54x, isotonic
      0.41–0.70x of the deployed lift, six of six properties. Post-hoc calibration cut ECE
      3–6x and left AUROC **bit-identical**, because the decoder's softmax over candidates
      consumes ranks and spacings, never levels. A 6.9x-wider readout moves seed-matched
      AUROC by ≤ +0.0019; a target-focused 3-bin readout is worse on all six. Best
      end-to-end gain anywhere: **1.09x** at one anchor, 1.00x and 0.76x at the other two.
      **No arm beats compute-matched best-of-N** (best advantage −0.2238 vs deployed
      −0.2247). Off-policy miscalibration re-measured at **1.35–2.09, not 3.5**, confirming
      §11.6's arithmetic. Pooling remains untested — it is the one item on §8.3's list that
      neither C17 nor C18 touches.
- [x] **C17 (promoted to second, DONE 2026-07-30)** ~~Probe-layer sweep, all 12 layers.~~
      **Done, all 13 probe points; `pilot_report.md` §21. It overturns a report claim and
      still does not help steering.** Question 1: **ARTEFACT for all six properties.** The
      depth curve is unimodal with a mid-network peak at probe point 3–5 for every property,
      declining monotonically to the final layer. Aromatic rings 0.8474 at probe point 3
      against trivial 0.8269 (+0.0205, Bonferroni CI [+0.0142, +0.0284] at α = 0.05/13);
      TPSA reverses too. So §13.1's crossover is a fact about **layer 12**, not about the
      representation, and §5.5, §8.1 and §13.3 were re-scoped accordingly. Question 2:
      **NOT MATERIAL** — the better-predicting layer improves per-position steering on 2 of
      6 properties, median relative −0.077, against a pre-registered bar of ≥4/6 and ≥+0.25.
      The AUROC-best layer is **not** the steering-best layer for **any** of the six.
      Cost: 2,205,784 processed tokens, no molecules generated, because
      `output_hidden_states=True` returns all 13 probe points from one forward pass.
      **Still outstanding: end-to-end guided generation at a mid-network layer** — C17
      measures per position only, and per position is not end to end (C22.1). See C23.
- [x] **C23 (new, from C17) — DONE 2026-07-31.** End-to-end guided generation with a
      mid-network head. Report `reports/section_c23_layer_end_to_end.md`; tests
      `tests/test_layer_end_to_end.py`. Validity gate is the strongest control in the
      project: the C17 layer-12 checkpoint through the new code path reproduced the
      published aromatic-rings run with residual **exactly 0.0** on all six (condition,
      seed) cells and 3072/3072 identical molecule strings, so the only difference between
      the experimental arms is the layer.
      **Rule A — a mid-network head improves guided generation over unguided — HOLDS.**
      All 15 arms positive, 11/15 with corrected CI excluding zero (aromatic L3 λ=1
      +0.0964, aromatic L6 λ=1 +0.1096, HBD L4 λ=2 +0.1300). It survives C25's head seeds.
      **This contradicts C17/§21.5 in sign**: aromatic at L3 is −0.111 per position and
      +0.0964 end to end. Per position is not end to end (C22.1), and here they disagree.
      **Rule B — a mid-layer arm beats compute-matched best-of-N — RETIRED by C25 and
      C26.** It rested on one arm (hbd_count L4 λ=2) whose realised token ratio was 1.0897
      from integer flooring. Priced on C26's continuous frontier at its own budget the
      advantage is +0.0267, and across C25's three head seeds it is +0.0267 / +0.0210 /
      **−0.0649** — mean −0.0057, sd 0.0513, sign flipped.
- [x] **C24 (generality / external validity) — DONE 2026-08-01.** Report
      `reports/section_c24_generality.md`; tests `tests/test_generality.py`. A second,
      **non-molecular** substrate (GPT-2 text; `digit_count`, `upper_count`,
      `mean_word_length`) running the *same* guidance rule, with `combine_scores` and the
      calibration/binning/best-of-N library imported from the molecular code rather than
      re-derived. Reported strictly as an external-validity check, never as part of the
      molecular result. The λ=0 top-8 **truncation control** was missing and was added:
      truncation alone destroys 47.5–85.7% of the base hit rate, and for
      `mean_word_length` it flips the sign of the guided-vs-reference contrast
      (−0.0658 against `unguided`, **+0.0202** against the correct control).
      **Claim 1 (calibration is a λ rescale): the algebra travels, the empirical
      regularity does not.** The identity is exact on GPT-2 (ΔAUROC 0.000000; 1536/1536
      identical sequences at ε=0), but one of three Platt slopes is **above** 1 (1.6154),
      and there calibration *helps*. "Calibration is a λ rescale" is general; "and
      therefore it hurts" is contingent on α<1 and is a molecular fact.
      **Claim 2 (best-predicting layer ≠ best-steering layer): does not travel.** The
      depth curve does — AUROC peaks at probe point 2 for all three attributes — but the
      per-position proxy and the end-to-end run **agree** here, the opposite of C23.
      Nothing beats compute-matched best-of-9 anywhere.
- [x] **C25 (pooled readout) — DONE 2026-08-01.** Report `reports/section_c25_pooling.md`;
      tests `tests/test_pooled_readout.py`. §8.3's last open suspect: does the head fail
      because it reads a single final-position state rather than a pooled window?
      **Pooling helps prediction** — rules P1 and P2 fire (5/6 properties each), max AUROC
      margin +0.0382, median +0.0116 over 48 comparisons, no cell hurts. But the winner is
      `mean16`, a **parameter-free** 16-position mean; the parameter-heavy `concat4` is the
      worst and `wide1` helps nowhere. So it is smoothing, not capacity — P3 does not fire.
      **C25 has no steering result**: its trigger fired on 15 comparisons and none of the
      capped end-to-end arms was run. E1/E2/E3 are vacuous and must not be read as an
      end-to-end null.
      **The head-seed replication is C25's most consequential output** and is what retires
      C23's Rule B. Head-seed span 0.0916 on the Rule B arm against 0.0037 across
      generation seeds — a factor of 25. C23's three seeds replicated the wrong thing.
- [x] **C29 (from the reviewer panel) — head-seed replication at n=8, plus the effective-λ
      control. DONE 2026-08-01.** Report `reports/section_c29_head_seeds.md`; tests
      `tests/test_head_seeds.py`. All four priorities ran; every validity gate is an
      identity at residual 0.0 (33 tensor comparisons, 6144 molecules).
      **C25's "factor of 25" is wrong.** It compared the head-seed span against the
      *smallest of three* per-cell generation-seed spans (0.0672 / 0.0413 / 0.0040).
      Pooled over 16 df the ratio on the Rule B arm is **1.71, 95% F interval
      [0.96, 3.65]** — 24.76 is outside it and the interval does not exclude 1. Head-seed
      sd 0.0366 [0.0242, 0.0744] against generation-seed sd 0.0213. The transferable
      finding survives as *"head-seed variance is at least as large as generation-seed
      variance and is never reported"*, with no multiplier.
      **It is about probes in general, not mid-network probes** (q = 1.52 / 1.61 / 0.81,
      all inside the pre-registered [0.5, 2] band), so every guidance number in the
      project inherits it.
      **The effective-λ confound is real and roughly the reviewer's size:** the coarse
      re-pricing reproduces +0.0459 / +0.0592 / +0.0245 exactly; on the newly measured
      fine envelope the three λ=1 arms are +0.0375 / +0.0507 / +0.0217, i.e. 54–69% of
      the headline. **Five of C23's fifteen arms change sign; Rule A's 15/15 becomes
      10/15.** But Rule A itself *survives* head-seed variation: paired at matched λ,
      +0.0941 / +0.1270 / +0.0266, all three intervals excluding zero.
      **C25's retirement of Rule B was under-powered.** At n=8 the advantage over C23's
      own compute-matched best-of-N is +0.0490 [+0.0135, +0.0845] (6/8 seeds positive),
      and head seed 3456 — C25's sign flip — is the worst of eight. It still contains zero
      against the token-conservative (+0.0246 [-0.0060, +0.0552]) and C26-corrected
      (+0.0156 [-0.0173, +0.0485]) comparators, so Rule B is **unresolved**, decided by
      comparator choice rather than by head seed.
      **C27's E4 does not replicate on hbd_count**: -0.0292 / +0.0238 / +0.0400 across
      head seeds, span 0.0692 against a published effect of 0.0292. aromatic_rings and qed
      are sign-stable. Four of eight pre-registered predictions were falsified, and R4 was
      **not scoreable as pre-registered** (A1 sits at λ=2, the deployed family was fixed at
      λ=1); the defect is reported, not amended.
      Original plan, for the record:
      Priority order: (1) 6-10 head seeds on hbd_count pp4 λ=2, the arm that carried Rule B;
      (2) the same set on aromatic L3 λ=1, qed L4 λ=1 and the **deployed** λ=1 arm — if
      head-seed variance is as large at the deployed layer, the finding is about probes in
      general, not mid-network probes; (3) the effective-λ control; (4) C27 across head
      seeds 2345/3456, time permitting.
      Two experiments, one directory, because they defend the same claim.
      **(a) Head seeds.** Six to ten head seeds on each of C23's three anchor arms
      (aromatic L3 λ=1, hbd L4 λ=2, qed L4 λ=1) **and** on the deployed λ=1 arm. C25
      established the head-seed span at 0.0916 against a generation-seed span of 0.0037
      — a factor of 25 — but from **three** head seeds, and the sd of a span at n=3 is
      enormous. This converts the project's most transferable finding from suggestive
      into the paper's spine and puts a real interval on the 25× ratio. It also tells us
      whether C23's Rule A (15/15 arms positive) survives head-seed variation, which
      currently rests on head seed 1234 alone for 12 of its 15 arms.
      **(b) The effective-λ control.** The mid-network head's `log q` is *wider* across
      candidates than the deployed head's: `outputs/c17_layer_steering/layer_steering_metrics.json`
      gives `mean_head_q_spread_across_candidates` of 0.1992 at layer 12 against 0.3013
      at layers 3 and 6 for aromatic rings (**1.51×**), 0.1321 vs 0.1680 for HBD
      (**1.27×**), 0.1029 vs 0.1279 for QED (**1.24×**). By the project's own λ-rescale
      identity (§20.3), a multiplicative rescale of `log q` **is** a λ rescale — so
      "the mid-layer head beats the deployed head at matched λ" is **not** matched
      steering strength. Run the deployed head at λ ∈ {1.25, 1.5, 2.5} on all three
      anchors and re-price Rule A against the interpolated deployed envelope.
      Reviewer arithmetic against the existing envelope suggests roughly **half** the
      λ=1 headline is the rescale: aromatic L3 +0.0963 → +0.0459, aromatic L6 +0.1096 →
      +0.0592, HBD L4 λ=1 +0.0701 → +0.0245. The sign survives, so this is scoping, not
      refutation — but it is unaddressed and it attacks the last positive result standing.
- [x] **C28 (from the reviewer panel) — the k sweep and the guided-drafts composition.
      DONE 2026-08-01.** Report `reports/section_c28_k_sweep.md`; pre-registration
      `outputs/c28_prereg/` (SHA-256 locked before any measurement); tests
      `tests/test_k_sweep.py`. All three priorities completed: the k sweep on all three
      anchors (six strands x five k, 30 cells), the composition at N ∈ {1,2,4,8} reranked
      by oracle and by head, plus a post-hoc extension of best-of-N to N=80 so the
      composition's 3179.77-token budget is measured rather than extrapolated.
      **The knob exists and it is empty.** D1 **REFUTED AS STATED**: within one strand,
      one hyperparameter moved, cost spans **10.24–11.06×** against C26's 1.170; cost is
      exactly (k+1)× base (gate G4 residual 0). D2 **NULL on five strands and HARMFUL on
      one**, PRODUCTIVE nowhere: hit(k=32) − hit(k=8) is +0.0078 / −0.0185 / +0.0044 /
      −0.0271 / +0.0020 / +0.0077. From k=4 to k=32 (a 6.4× cost increase) five of six
      strands go **down**. D3a: **below the oracle-selected curve on all six strands** at
      k=32 (−0.6206 to −0.1782), every seed t interval excluding zero.
      Validity gates G1/G2/G3 all reproduce their frozen artefacts with residual **0.0**
      on hit rate and tokens, aggregate and every seed; G6 is stronger than required —
      the first 512 guided drafts of each seed are the published 512 string for string.
      The composition is not a compute axis: `head_reranked` is **below** the
      head-selected curve at every N (−0.0166, −0.0123, −0.0334, −0.0139), and
      `oracle_reranked`'s apparent +0.0182 win at N=8 becomes **−0.0512** once the
      comparator is measured out to its budget (0.9965 at 3197.98 tok/mol).
      Seven of ten pre-registered predictions hold; **P2, P8 and P9 fail** and are
      reported as failures. **C26 §C26.4.4's mechanism must be withdrawn and its
      conclusion strengthened** — listed as a conflict in §C28.8, not edited there.
      Original brief:
      Priority order: (1) k ∈ {2,4,8,16,32} on hbd_count at layers 12 and mid, priced
      against **both** the C26 oracle-selected and the C27 head-selected curves; (2) the
      guided-drafts-then-rerank composition at N ∈ {2,4,8}, reranked by oracle and by head;
      (3) the other two anchors. Validity gate: k=8 must reproduce the published deployed
      run exactly (0.29875114714835704, 401.619140625 tok/mol).
      `top_k_candidates: 8` is fixed in every molecular run, every text run and every
      module (`guidance.py`, `pooling.py`, `generality.py`) and has **never been swept**.
      Guidance's cost is exactly (k+1)× base — 43.4 → 401.6 processed tokens per
      molecule, ratio 9.25 — so **k is the compute knob**, it sits inside the method as
      specified, and C26's headline "guidance has no compute knob" is currently a claim
      about a hyperparameter we froze rather than a finding. One guided run at k=32 costs
      ~1420 tokens/molecule, exactly the top of C26's own best-of-N grid, where best-of-N
      reaches 0.9271–0.9961.
      **(a)** Sweep k ∈ {2, 4, 8, 16, 32} on one anchor at the deployed layer and at the
      best mid-network layer, and plot guidance on the same x-axis as best-of-N.
      **(b)** Run the composition nobody has run: generate N **guided** drafts and rerank.
      This is close to a one-line change to `scripts/06_best_of_n.py` and it gives
      guidance a continuous compute axis immediately. It is the experiment a hostile
      reviewer will demand.
      There is a partial defence to state either way: the molecular truncation control is
      null (0.1829 vs unguided 0.1785 for aromatic rings, 0.0849 vs 0.0837 for HBD), so
      little property-relevant mass sits outside the top 8 in SMILES and k may buy little
      accuracy *here*. But that is a fact about a ~2.4k-token chemistry vocabulary, not
      about FUDGE: on GPT-2 the same control destroys 47.5–85.7% of the base hit rate.
      Outcome either way is a result. If guidance still tracks below the curve at ~1400
      tokens/molecule, the negative result becomes genuinely strong.
- [x] **C26 (N sweep — the other half of E1) — DONE 2026-08-01.** Report
      `reports/section_c26_n_sweep.md`; tests `tests/test_n_sweep.py`. Best-of-N's
      compute–accuracy frontier, N ∈ {1,2,3,4,6,8,9,12,16,24,32}, three anchors, 3 seeds.
      Gates: the published call signature reproduces bit-identically on hit rate **and**
      tokens (residual 0.0, all seeds), and the sweep's first 512 groups reproduce
      `scripts/06_best_of_n.py` **exactly** on all nine (anchor, seed) cells.
      The pre-registered nested estimator **failed its own gate** and was replaced rather
      than caveated (v1 kept at `outputs/c26_nsweep_v1_nested_*`).
      **46 guidance arms priced against the curve at their own budgets: 45 below it, 1
      above, and that one dies on head-seed replication.**
      **The structural finding may matter more: guidance has no compute knob.** All 46
      arms sit inside a 5.1–17.0% token band while best-of-N spans 32×. Guided decoding
      does not merely lose at the budget we matched at — it cannot be offered more compute
      within the method as specified.
      Also: the published best-of-9 comparators are slightly optimistic (aromatic rings
      0.8294 → **0.8150** on the 3.6× larger estimator), which moves comparisons in
      guidance's favour and is adopted for that reason.
- [x] **C10 (demoted, then DONE 2026-07-30)** ~~λ sweep, 5–6 values, crossed with
      matched-budget best-of-N, 3 seeds, on 2–3 anchor properties only.~~ **Done exactly as
      specified**: 0.25 / 0.5 / 1 / 2 / 4 / 8 on aromatic rings, HBD count and QED, three
      seeds, 512 molecules per condition per seed, matched best-of-N at each λ, `--lam` flag
      added to script 05 rather than editing the config. `pilot_report.md` §19.
      **Result:** the response is an **inverted U** with an optimum at λ = 2–4, not the
      monotone rise the pre-committed interpretation allowed for; tuning λ is worth
      **1.29–1.69x** end to end; **no λ beats compute-matched best-of-N** (best gap −0.0931,
      HBD count at λ=2), so **P5 is not falsified and CL4 is strengthened**; past the optimum
      the base policy is destroyed rather than overridden (validity 0.995–0.998 → 0.807–0.902).
      The demotion was right, but note *why* it was right is now a measurement rather than the
      per-position extrapolation this entry originally rested on — see C22.1.
      **Still outstanding: the N sweep**, which the simulated reviewer asked for alongside it.
- [x] **C11** ~~Test **P6**: does the guidance-vs-best-of-N gap track locality?~~ **Done at
      λ=1 for all six properties**; `pilot_report.md` §16–17. (The λ-crossed version still
      needs C10.)
- [x] **C12 (DONE 2026-07-30)** ~~Run `scripts/10_quality_analysis.py` at every λ.~~
      **The prediction holds.** R3 does not fire at λ ≤ 2 and does at λ ≥ 4. Degeneracy among
      target-hitting molecules rises from 0.010–0.020 at λ=1 to 0.061–0.122 at λ=8; QED's SA
      score worsens by **+0.389**, five times the largest degradation anywhere in phase 1.
      So the pilot's quality null result is confirmed **λ-specific** rather than general, and
      §10's conclusion is now bounded instead of unqualified.
      Two things the prediction did *not* anticipate, both in `pilot_report.md` §19.3:
      the failure mode is **fragmentation, not long greasy tails** (longest chain *falls*
      significantly at λ=8 for all three anchors while fragment count rises), which follows
      from a bounded target rather than a maximisation one; and for **QED the hit-rate optimum
      is already inside the degenerate regime**, so at least one property has no λ that is
      both best and clean.

### Week 4 — analysis and writing

- [x] **C13** ~~Locality↔steerability rank correlation with a bootstrap CI, n=6 stated
      plainly.~~ **Done**, `scripts/12_locality_scatter.py`. The interval propagates
      measurement noise in both coordinates with the six properties held fixed, and says so;
      it does **not** cover the uncertainty from having chosen six hand-picked properties.
- [x] **C14** ~~Artifact-binding tests for every new claim.~~ **Done**; the suite went from
      183 to 280+ tests, and several report claims were *corrected* by them during writing.
- [x] **C15** ~~Extend `docs/REPRODUCE.md`.~~ **Done**, a full phase-2 section P0–P7, plus
      `scripts/run_phase2.sh` to replay the chain by stage.
- [x] **C16** ~~Buffer for RDKit surprises.~~ **Spent, but not where expected.** RDKit gave
      no trouble at all: zero molecules failed QED across 49,825, and the HBD /
      rotatable-bond definitions were already pinned in `compute_candidate_properties`. The
      time went on two defects in our own phase-1 code instead (§11.2, §11.5).

### Also now in scope (previously "cut if tight")

- [ ] **C19** Compositional confound: add heteroatom or halogen fraction as a third
      stratifying covariate in `scripts/09_confound_analysis.py`. Watch `coverage` — a
      third covariate thins strata fast.
- [ ] **C20 (new, from phase 2)** The headroom ceiling is an *unweighted* max over eight
      candidates, while λ=1 keeps `log p_base` at full weight and the base policy puts a
      mean 0.738 of its top-8 mass on one token. That gap is the whole of the λ term in
      §15.6. Worth measuring directly: how does the ceiling change if candidates are
      restricted to those with base probability above some floor? It bounds what a
      *likelihood-respecting* decoder could ever do, which is the practically relevant
      ceiling and is tighter than the one reported.
- [ ] **C21 (new, from phase 2)** Headroom in probability units **rises** with position for
      every property, by a factor of 3–10 from Q1 to Q4 (§15.5). Nothing in the project
      predicted that and it is the most interesting unexplained pattern phase 2 produced.
      Cheap to investigate with the arrays already on disk.

### C22 — the audit of the phase-2 write-up, 2026-07-30

Run after phase 2 was written up, because the first pass was drafted in one long session
and the owner was right to want it re-checked. Six defects were found. **None changed a
measured value; five changed what the report was entitled to say, and every one of the five
was in the direction that flattered our own conclusion.** All are now fixed and bound by
tests in `tests/test_report_matches_artifacts.py`.

- [x] **C22.1 Per-step read as end-to-end.** §15.6 and §17.3 converted a *per-position*
      decomposition into end-to-end experiment ranking ("λ worth ~2x, the head worth 5–8x",
      "a λ sweep cannot close a 0.22–0.36 gap"). End-to-end lift is 20–48x per-step gain, and
      transferring the ratios linearly implies lifts above the arithmetic maximum for four of
      six properties. The inference is withdrawn and **replaced by the measurement** (§19).
- [x] **C22.2 Two tables, two prefix sets.** §15.3's survivorship check ran on the
      380-prefix headroom set while §15.1 ran on the 267-prefix capture set, so the same
      ceiling appeared with two values (up to 0.024 apart) and §15.3's `available` column was
      raw against §15.1's noise-corrected one. Recomputed on one set, in
      `scripts/12_locality_scatter.py` so it is an artefact rather than an ad-hoc number.
- [x] **C22.3 Base-policy concentration quoted off the wrong set.** "median 0.90 of its
      top-8 mass on one token" is the all-400 figure; on the 267 prefixes the argument is
      computed on it is mean 0.738 / median 0.758. The wider set overstated the case for
      "λ=1 structurally cannot reach the ceiling".
- [x] **C22.4 Capture's denominator is not the pre-registered one.** The reported 4.8–10.9%
      subtracts a permutation null; `LEXICAL_LOCALITY.md` §3's literal formula gives
      **2.1–5.8%**. Both are now reported. (This one is conservative for us — the correction
      makes our head look better.)
- [x] **C22.5 "No post-hoc degrees of freedom whatsoever."** §17.3 said that of
      `PREDICTED_LOCALITY_ORDER` while §12.2 concedes the hypothesis exists to explain two
      results already in hand — and those two properties sit at predicted ranks 1 and 5.
      Repaired with a leave-phase-1-out correlation, which the finding **survives**:
      rho = +0.800 on the four properties whose ranks were genuinely blind (n = 4, p = 0.20).
- [x] **C22.6 Off-policy prefixes not disclosed.** Headroom is measured at base-policy
      prefixes; guidance visits its own, where §9.2.1 says the head is worse. Stated.

Two further items were checked and found sound rather than fixed: the token match for
best-of-N (realised/target 0.95–1.02 across twelve runs, and where it errs it errs against
best-of-N), and the reconstruction of per-candidate hit counts inside the oracle null
(exact, now asserted rather than assumed).

---

## D. Paper

- [x] **D1** ~~Restructure around the locality thesis **only if the data supports it**; if
      P1/P2 fail, revert to the better-scoped negative result and say so.~~ **Decided by the
      data: P1 and P2 both fail** (§15.4, §15.5), so the locality restructure does **not**
      happen. But the fallback is better than "the pilot's negative result, scoped": phase 2
      produced a *third* option neither branch anticipated. The headroom measurement
      **locates** the negative result — the lever exists everywhere, λ=1 throttles half of
      it, the head loses most of the rest — and that is a mechanistic claim that does not
      depend on locality at all. Abstract v3 in `reports/ABSTRACT.md` is written around it.
      **A2 (venue) can now be decided**: this is a mechanistic-plus-negative result with a
      falsified pre-registration reported as such, which fits TMLR's stated criteria better
      than a workshop's novelty framing.
- [x] **D2** ~~Use abstract v2 as the base and update every number.~~ **Superseded**:
      abstract **v3** is written from the phase-2 result, with v2 kept because the review it
      answers is part of the record.
- [ ] **D3** State in text, with the citation, that cLogP is retained despite PMO's
      exclusion of it as a *maximisation* target, because ours is a bounded percentile band
      penalised symmetrically for over- and undershoot.
- [ ] **D4** State the oracle-vs-proxy asymmetry in the **abstract**, not only in §5:
      best-of-N selects on the true RDKit value; guidance sees only a learned head.
- [ ] **D5** Limitations, stated rather than defended: six hand-picked properties is a
      small non-random sample; and the thesis is a claim about **SMILES tokenization**, so
      the sharpest untested prediction is SELFIES, where locality ranks properties
      differently.
- [ ] **D6** Handle SLIM accurately — a mechanism hypothesis consistent with their published
      result, **not** a resolution of a conflict, and **not** a reimplementation of their
      method. `LITERATURE.md` §5 has the corrected account.

---

## E. Do not do these

Each would either weaken the paper or eat the four weeks.

- [ ] Second generator. Out of scope: new checkpoint sourcing, new tokenizer quirks, new
      confounds, and it does not serve the locality thesis.
- [ ] **SELFIES — ask the owner first, do not just start.** The original specification
      excludes alternative molecular serializations, so this needs an explicit decision to
      lift. But note the tension honestly: lexical locality is a claim *about SMILES
      tokenization*, so SELFIES — where ring closures and branches are encoded completely
      differently, and locality would therefore rank properties differently — is the
      sharpest available test of the thesis rather than a tangent. With no binding deadline
      it is a genuine candidate for a phase 3. Raise it; do not decide it unilaterally.
- [ ] Full λ×N grid on all six properties. Depth on 2–3 anchors, breadth at one λ elsewhere.
- [ ] New guidance algorithms (lookahead beam search, classifier-free-guidance analogues).
      Different research question.
- [ ] Extending the DAgger line to the new properties. Tangential, and the one permitted
      round is spent.
- [ ] Chasing wall-clock reproducibility on the GPU. Token accounting is settled.
- [ ] More than six properties because compute is cheap. Credibility comes from correct
      definitions per property, not more points.
- [ ] Dropping an outlier property from the scatter. If QED does not fit, report it as an
      outlier with your best account of why. The artifacts are released.
- [ ] Any change to the frozen base generator. No fine-tuning, no LoRA, no RL, no
      activation edits.
