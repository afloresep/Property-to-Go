# Next steps

Written 2026-07-30. Target: a NeurIPS 2026 workshop, deadline reported as **2026-08-29**
(unverified — item A1). That is about four weeks out.

Ordering is by dependency, not importance. Items marked **[you]** need a human decision or
an account only you have; everything else an agent can do.

---

## A. Before any compute — verification and decisions

These are cheap and they gate everything downstream.

- [ ] **A1 [you]** Verify the submission deadline and page limit directly at the venue.
      Both candidates came from an automated web scan: **ICBINB-BIO** at NeurIPS 2026
      (icbinb-bio.github.io, reported 8pp full / 4pp tiny, explicitly solicits negative
      results and failure analyses) and **ML4Molecules 2026** (moleculediscovery.github.io,
      reported 5pp, explicitly solicits "negative results, careful ablations, and rigorous
      baselines"). Do not plan four weeks around a date nobody confirmed.
- [ ] **A2 [you]** Decide: submit to one venue or both. They reportedly share a deadline,
      and both are non-archival, so both is plausible. Page limits differ (8 vs 5), which
      changes how much goes in an appendix.
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
- [ ] **A7 [you]** Confirm the property decision. The plan as written **demotes** cLogP
      rather than dropping it: keep it as one of six, out of the title, not carrying the
      headline number. Dropping it entirely discards continuity with the executed pilot,
      since every existing number is cLogP or aromatic rings.

---

## B. Setup gates on the RTX machine — do not skip, do not reorder

Full detail in `KICKOFF_PROMPT.md` §2.

- [ ] **B1** Clone, `uv venv && uv pip install -e .`, confirm **183 tests pass**.
      Always invoke `.venv/bin/python` — an ambient `transformers` 5.x cannot load the
      pinned model revision, and the failure looks like a model bug.
- [ ] **B2** Set `device: cuda` in `configs/model.yaml`. Leave `dtype: float32` alone.
- [ ] **B3** Run `tests/test_model_contracts.py` on GPU. `test_forward_pass_is_deterministic`
      and `test_candidate_backends_agree` are the two that matter — the first underwrites
      reproducibility, the second underwrites the entire compute accounting.
- [ ] **B4** Regenerate the large arrays excluded from git:
      `scripts/02_generate_trajectories.py` for `pilot_50k`. Then **diff the regenerated
      `windows.json` and `target_intervals.json` against the tracked ones.** If they moved,
      stop and write that up — the pilot's windows were frozen before any guided result was
      inspected, and silently re-deriving new ones would invalidate the comparison.
- [ ] **B5** Reproduce one known number end to end:
      `scripts/05_guided_generation.py --dataset pilot_50k --property aromatic_rings
      --seeds 101 --conditions unguided throughout`. Expect ~0.47 (throughout) and ~0.17
      (unguided). **If GPU diverges from CPU, that is a result to characterise and report,
      not an obstacle to route around.**

---

## C. Phase-2 experiments

Hypothesis, operationalisation and pre-registered predictions P1–P6 are in
`LEXICAL_LOCALITY.md`. Compute is not the bottleneck; engineering time and RDKit edge
cases are.

### Days 1–2 — lock the design

- [ ] **C1** Confirm `properties.PREDICTED_LOCALITY_ORDER` is committed **before** any new
      guidance run. It is the pre-registration. Do not edit it to match observed data.
- [ ] **C2** Wire in the four staged properties: add to `ALL_PROPERTIES`, add
      `hbd_count` and `rotatable_bonds` to `bestofn.INTEGER_PROPERTIES` (see `HANDOFF.md`
      §4 — this is the boundary bug), add `target_interval_rule` entries to
      `configs/guidance.yaml`, and write `target_intervals.json` **before** inspecting any
      guided result.

### Week 1 — cheap breadth

- [ ] **C3** Compute the four new properties on the molecules and rollout bank you already
      have. Minutes of compute, no regeneration.
- [ ] **C4** Train frozen-state heads for the new properties, **2–3 head seeds each**
      (single-seed was a flagged weakness), plus the fixed trivial-features head for all
      six properties.
- [ ] **C5** Produce predictability-vs-position curves and one locality score per property.

### Week 2 — the central test

- [ ] **C6** **Steering headroom** — the highest-value new measurement. At each prefix, for
      each of the 8 candidates, estimate `E[final property | prefix + candidate]` by
      rollouts; take the spread. Head-free and λ-free, so it is an upper bound on what any
      decoding rule could achieve, and it separates "no lever to pull" from "our head is
      bad" — the question the pilot could not answer. Implement with
      `generation.continue_from_prefixes`; no new inference machinery.
- [ ] **C7** Report what fraction of available headroom the pilot's guidance actually
      captured, per property and per position quartile.
- [ ] **C8** One guidance run per property (3 seeds, n≈512) plus one matched best-of-N
      point, all six properties. **The deliverable is the scatter: locality score against
      steerability effect size.**
- [ ] **C9** HBD count is the discriminating case — SLIM's hardest property under additive
      latent steering, which locality predicts should be *easy* under token choice. If it
      is hard for us too, **P1 fails and that is the finding.**

### Week 3 — depth, on anchors only

- [ ] **C10** λ sweep, 5–6 values (e.g. 0.25 / 0.5 / 1 / 2 / 4 / 8), crossed with
      matched-budget best-of-N, 3 seeds, on **2–3 anchor properties only** spanning the
      locality axis. Add a `--lam` flag rather than editing `configs/guidance.yaml`, so
      `configs_used.json` records the value used.
- [ ] **C11** Test **P6**: does the guidance-vs-best-of-N *gap* itself track locality? If
      yes, the pilot's two disconnected findings become consequences of one variable — this
      is what makes it one paper instead of two.
- [ ] **C12** Run `scripts/10_quality_analysis.py` **at every λ**. The pilot's finding that
      molecules stay chemically sound is specific to λ=1 into a bounded interval, and high
      λ is exactly where the literature's degenerate molecules should appear. That is a
      prediction; test it.

### Week 4 — analysis and writing

- [ ] **C13** Locality↔steerability rank correlation with a bootstrap CI, and state plainly
      that n=6.
- [ ] **C14** Add artifact-binding tests for every new claim entering the report. The
      existing 16 make prose-drift impossible; keep that property.
- [ ] **C15** Extend `docs/REPRODUCE.md` with exact commands for every new experiment.
- [ ] **C16** Real buffer for RDKit surprises (QED raising on odd intermediates, HBD and
      rotatable-bond definition mismatches). Budget days, not hours.

### Lower priority — do these only if weeks 1–3 finish early

- [ ] **C17** Probe-layer sweep, all 12 layers. The pilot used only the final layer
      (`hidden_layer: -1`), so the aromatic-ring negative result may be about the readout
      rather than the representation. No generation needed.
- [ ] **C18** Post-hoc calibration of the head on held-out guided prefixes (it is
      under-confident by 3.5× there: predicted 0.076 vs observed 0.267, ECE 0.190). Fix the
      signal, then re-test, so "guidance fails" is not confounded with "the signal is
      broken". Note headroom (C6) addresses the same question more cleanly.
- [ ] **C19** Compositional confound: add heteroatom or halogen fraction as a third
      stratifying covariate in `scripts/09_confound_analysis.py`. Watch `coverage` — a
      third covariate thins strata fast.

---

## D. Paper

- [ ] **D1** Restructure around the locality thesis, with the negative result as supporting
      evidence — **only if the data supports it.** If P1/P2 fail, the paper reverts to the
      better-scoped negative result and says so.
- [ ] **D2** Use abstract v2 in `reports/ABSTRACT.md` as the base and update every number.
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

- [ ] Second generator or second molecular serialization. Out of scope; the original
      specification excludes alternative serializations. SELFIES goes in future work.
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
