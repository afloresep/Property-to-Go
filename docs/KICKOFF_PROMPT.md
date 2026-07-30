# Continuing on the RTX machine: transfer, setup, and the prompt to paste

Written 2026-07-30. Everything below is instructions for a *fresh* session on different
hardware. No compute work is to continue on the original laptop.

---

## 1. Getting the project across

**Already done.** The repository is pushed to `git@github.com:afloresep/Property-to-Go.git`
(branch `main`). On the RTX machine:

```bash
git clone git@github.com:afloresep/Property-to-Go.git
cd Property-to-Go
```

That clone is ~31 MB and contains everything the pipeline cannot regenerate: all code,
configs, tests, docs, reports, every `*_metrics.json`, the frozen `windows.json` and
`target_intervals.json`, `molecules.json`, `rollout_bank.json`, and the trained head
checkpoints.

**What is deliberately not in the clone**, because it is regenerated deterministically from
the pinned model revision and recorded seeds (~700 MB): `hidden.npy`, `features.npy`,
`trajectories.json`, `prefix_token_ids.json`, `prefix_meta.csv`, and the prediction `.npz`
files. `.gitignore` explains this inline so nobody later assumes it was an accident.

Rebuild them with step 2 of `docs/REPRODUCE.md`, then **diff the regenerated
`windows.json` and `target_intervals.json` against the tracked ones**. They were frozen
before any guided result was inspected; if regenerating on different hardware moves them,
that is a finding to write up, not something to silently accept.

---

## 2. First 20 minutes on the new machine

Do these in order. Do not skip to the experiments.

```bash
cd Property-to-Go
uv venv && uv pip install -e .        # transformers 4.44.2 will be pinned for you
.venv/bin/python -m pytest            # expect 183 passed
```

Then, before any long run:

1. Set `device: cuda` in `configs/model.yaml`. Leave `dtype: float32` alone for now.
2. Re-run the model contract tests specifically:
   `.venv/bin/python -m pytest tests/test_model_contracts.py -v`
   The two that matter are `test_forward_pass_is_deterministic` and
   `test_candidate_backends_agree`. The second underwrites the entire compute accounting.
3. Reproduce **one** known number end to end. Cheapest meaningful check:
   ```bash
   .venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k \
       --property aromatic_rings --seeds 101 --conditions unguided throughout
   ```
   `throughout` should give a hit rate near 0.47 and `unguided` near 0.17. If it does not,
   stop and find out why before generating anything new.

Only after step 3 passes should new experiments begin.

---

## 3. The prompt to paste into a fresh session on the RTX machine

Copy everything between the lines.

---

```
Work autonomously in this repository. It contains a COMPLETE, reported pilot study;
your job is the next phase, not a rewrite.

READ FIRST, IN THIS ORDER, COMPLETELY, BEFORE TOUCHING ANYTHING:
  1. docs/HANDOFF.md            - environment traps, design decisions not to undo, the
                                  bug not to reintroduce, and the scoped experiment list
  2. reports/PLAIN_SUMMARY.md   - what was done and what was found
  3. reports/ABSTRACT.md        - every claim we make, the strongest objection to each,
                                  and a reviewer's verdict on the draft
  4. docs/LEXICAL_LOCALITY.md   - the hypothesis this phase is designed to test, with
                                  pre-registered predictions P1-P6
  5. docs/LITERATURE.md         - prior art, and which citations are still unverified
  6. docs/TODO.md               - the dependency-ordered checklist this prompt implements
  7. docs/REPRODUCE.md          - exact commands for everything already run

Treat reports/pilot_report.md as the authoritative record of what has been executed. Do
not restate it; extend it.

HARDWARE: this machine has an NVIDIA RTX GPU. The pilot ran entirely on a laptop CPU and
took about one day of compute, so your budget is large relative to what has been done.

BEFORE ANY NEW GENERATION, in this order:
  a. Set device: cuda in configs/model.yaml. Keep dtype: float32 until step (c) passes.
  b. Run the full test suite. Expect 183 passed. Then run tests/test_model_contracts.py
     specifically and confirm test_forward_pass_is_deterministic and
     test_candidate_backends_agree both pass on GPU. These two underwrite reproducibility
     and the compute accounting respectively.
  c. Reproduce one known number end to end:
       .venv/bin/python scripts/05_guided_generation.py --dataset pilot_50k \
           --property aromatic_rings --seeds 101 --conditions unguided throughout
     Expect hit rate ~0.47 (throughout) and ~0.17 (unguided). If GPU results diverge from
     the CPU pilot, STOP, characterise the divergence, and write it up before proceeding.
     A reproducibility failure is a result, not an obstacle to route around.

THE SCIENTIFIC GOAL OF THIS PHASE
Test whether steerability tracks the LEXICAL LOCALITY of a property rather than its
predictability from the frozen model's hidden state. docs/LEXICAL_LOCALITY.md states the
hypothesis, the operationalisation, and predictions P1-P6. Those predictions are
pre-registered: do not revise them after seeing data. If the data falsifies them, report
that.

The central new measurement is STEERING HEADROOM: at a prefix, for each of the base
model's top-8 candidate tokens, estimate E[final property | prefix + candidate] by
base-policy rollouts, and take the spread across candidates. It is head-free and
lambda-free, so it is an upper bound on what ANY decoding rule could achieve at that
position. It separates "there is no lever to pull" from "our head is bad" - the question
the pilot could not answer.

STOP AND REPORT after item 2 below (the locality-vs-steerability scatter). That result
determines what the paper IS: if the thesis holds, the paper becomes mechanistic and the
negative result becomes supporting evidence; if it fails, the paper reverts to the
better-scoped negative result. Do not start the lambda sweep before that decision is made.
Everything after item 2 is listed in docs/TODO.md sections C10 onward.

WORK TO DO, in priority order. Read docs/HANDOFF.md section 6 for the full list; this
phase is E-headroom (new), E1, E4, E3, E5, E5b.

  1. HEADROOM (new, highest value). Implement it as a new script following the existing
     conventions. One rollout bank serves all properties, exactly as Phase 4 does.
     generation.continue_from_prefixes already does the inference: build the extended
     prefixes prefix+candidate and pass them in as ordinary prefixes. Report headroom
     normalised by target-interval width, by position quartile, per property. Also report
     what fraction of available headroom the existing guidance actually captured.

  2. MORE PROPERTIES (E4). The pilot used two, which is the weakest point in the whole
     paper - a "double dissociation" on two properties is a pattern, not a law. Add
     H-bond donor count, rotatable bond count, TPSA and QED to the existing cLogP and
     aromatic ring count. The RDKit functions are already written and tested as
     properties.compute_candidate_properties, deliberately kept OUT of ALL_PROPERTIES so
     that wiring them in cannot silently change an executed result; the docstring lists
     the three steps to opt in, and the definitional choices (Lipinski vs Gasteiger HBD,
     strict rotatable bonds, QED raising on pathological structures) are recorded there.
     DEMOTE cLogP: keep it for continuity and because it is a clean stress test of the
     bounded-interval defence, but it should not be in the title and should not carry the
     headline number. QED and TPSA/HBD/rotatable bonds are standard MPO components and
     QED is an actual PMO task, so they carry credibility with a drug-discovery reader.
     Also state in text, with the citation, that cLogP is retained despite its exclusion
     from PMO (Gao et al. 2022) as a MAXIMIZATION target, because our target is a bounded
     percentile band of the base model's own output distribution and is penalised
     symmetrically for over- and undershoot. Rationale, which matters: QED is an actual PMO benchmark task,
     so it answers the objection that our targets are not real drug-design goals; HBD
     count is the case a competing preprint reports as HARD to steer, and our hypothesis
     predicts it should be EASY for us, so it is a direct discriminating test.
     Declare integer-valued properties in INTEGER_PROPERTIES - see docs/HANDOFF.md
     section 4 for the bug this prevents.
     PRE-REGISTER the predicted locality ordering (it is already written down in
     docs/LEXICAL_LOCALITY.md section 5) before you measure anything.

  3. LAMBDA SWEEP plus N SWEEP (E1). Produce a compute-accuracy FRONTIER for guidance and
     for best-of-N, not one point each. Two points cannot distinguish "guidance is
     uncompetitive at matched compute" from "guidance was not tuned". Add a --lam flag
     rather than editing configs/guidance.yaml in place, so configs_used.json records the
     value actually used. RUN scripts/10_quality_analysis.py AT EVERY LAMBDA: the pilot's
     finding that molecules stay chemically sound is specific to lambda=1 into a bounded
     interval, and high lambda is exactly where the literature's degenerate molecules
     should appear. That is a prediction; test it.

  4. PROBE-LAYER SWEEP (E3). All 12 layers. The pilot used only the final layer
     (hidden_layer: -1), so the aromatic-ring negative result may be about the readout
     rather than the representation. No generation needed.

  5. HEAD-SEED REPLICATION (E5), 5 seeds. Do this before claiming any margin under 0.03.

  6. COMPOSITIONAL CONFOUND (E5b). Add heteroatom or halogen fraction as a third
     stratifying covariate in scripts/09_confound_analysis.py. Watch coverage: a third
     covariate thins strata fast, and a matched estimate at 0.4 coverage is not a matched
     estimate.

SHAPE OF THE FOUR WEEKS. Compute is not the bottleneck - the CPU pilot did the whole
pipeline in about a day. Engineering time and RDKit edge cases are the bottleneck. Budget
accordingly.

  Days 1-2  Reproduce one pilot number on GPU (above). Lock the six-property battery.
            Confirm properties.PREDICTED_LOCALITY_ORDER is committed BEFORE any new
            guidance run. Do not skip this; it is what makes the thesis falsifiable
            rather than a story.
  Week 1    Cheap breadth. Compute the four new properties on the molecules and rollout
            bank you ALREADY have - minutes of compute, no regeneration. Train
            frozen-state heads for the new properties with 2-3 head seeds each, and the
            fixed trivial-features head for all six. Output: predictability-vs-position
            curves and a locality score per property.
  Week 2    The central test. Headroom for all six properties, plus one guidance run per
            property (3 seeds, n~512) and one matched best-of-N point. The deliverable is
            the scatter: locality score against steerability effect size. HBD count is
            the discriminating case. If the picture is messy you will know by end of week
            2, with two weeks left to build an honest qualified claim instead of forcing
            a clean one.
  Week 3    Depth, on 2-3 ANCHOR properties only, spanning the locality axis (aromatic
            rings, one mid property, and cLogP or QED). Lambda sweep of 5-6 values crossed
            with matched-budget best-of-N, 3 seeds. Test P6: does the guidance-vs-best-of-N
            GAP itself track locality? Extend the quality panel to the anchors.
  Week 4    Analysis, artifact-binding tests for every new claim, writing, and real buffer
            for RDKit surprises. Report the locality-steerability rank correlation with a
            bootstrap CI and state plainly that n=6.

DO NOT DO THESE, however tempting:
  - Do not add a second generator or a second molecular serialization. Out of scope, eats
    weeks, and the original specification excludes alternative serializations. State the
    SELFIES prediction as future work instead - see docs/LEXICAL_LOCALITY.md section 8.
  - Do not run the full lambda x N sweep on all six properties. Depth on 2-3 anchors,
    breadth at a single lambda on the rest. The marginal information from a full grid is
    low and the time cost is not.
  - Do not add new guidance algorithms (lookahead beam search, classifier-free-guidance
    analogues). Different research question.
  - Do not extend the DAgger/calibration line to the new properties. Tangential.
  - Do not chase wall-clock reproducibility on the GPU. Token accounting is settled;
    reopening it buys nothing.
  - Do not add properties past six because compute is cheap. Credibility comes from
    correct RDKit definitions and sound target intervals per property, not from more
    points.
  - Do not drop a property from the scatter because it does not fit. If QED is an outlier,
    report it as an outlier with your best account of why. The artifacts are released and
    a reviewer can check.
  - Do not claim the SLIM tension is "resolved". We have a mechanism hypothesis consistent
    with their published result, not a reimplementation of their method.

CONSTRAINTS, all still in force from the original specification:
  - The base generator stays FROZEN. No fine-tuning, no LoRA, no RL, no activation edits,
    no weight changes of any kind.
  - No reinforcement learning. The single permitted data-aggregation round has already
    been used and is not to be repeated.
  - No explicit partial-graph models, no activation steering, no multiple generators, no
    alternative molecular serializations, no elaborate uncertainty estimation.
  - Windows and target intervals are frozen BEFORE guided results are inspected. New
    properties need new intervals written to disk first, and then committed to.
  - Save configurations alongside every result; every script calls write_run_context.
  - Deterministic seeds throughout. Record every pin.
  - Do not force a positive conclusion. Keep executed results clearly separated from
    unexecuted plans.
  - Preserve README.md.
  - Wall-clock time does not reproduce on this project's original machine (20-25% between
    bit-identical runs). Report processed tokens. If you want to make a timing claim on
    this GPU, first measure whether timing reproduces here, and say what you find.

WHEN A NEW RESULT CONTRADICTS THE REPORT
Change the report. The artifact-binding tests in
tests/test_report_matches_artifacts.py exist to make that the path of least resistance:
they re-read each JSON and require the number to appear in the prose. Add binding tests
for every new claim you put in the report.

DELIVERABLES
  - New sections in reports/pilot_report.md for everything executed, artifact-bound.
  - reports/ABSTRACT.md updated: the claim inventory with the strongest objection to each,
    revised to reflect what the new data supports. If the lexical-locality hypothesis
    survives, it becomes the paper's thesis and the negative result becomes supporting
    evidence. If it dies, say so.
  - docs/REPRODUCE.md extended with exact commands for every new experiment.
  - A concise final response listing changed files, executed commands, test results, and
    the most important empirical result.

There is NO BINDING DEADLINE. A 2026-08-29 workshop deadline exists and has been
confirmed, but the project owner has decided not to optimise for it - the work can wait for
a later cycle or go to TMLR, which has rolling submission and does not list novelty or
significance as acceptance criteria. Do not rush a result to fit a date, and do not let the
absence of a date turn into open-ended scope growth either: the do-not list above still
applies.

Ask me nothing that you can decide reasonably yourself. Do ask before anything
destructive, before deleting or overwriting existing outputs, and before changing a claim
in reports/pilot_report.md that the current artifacts still support.
```

---

## 4. Adjusting the prompt

Three places you may want to edit before pasting:

- **cLogP: decided 2026-07-30 — demote, do not drop.** Item 2 already reflects this. No
  edit needed.
- **SLIM is verified to exist** (arXiv:2605.10831) and does *not* contradict the pilot —
  see `LITERATURE.md` §5 for why the apparent conflict was a metric confusion on our side.
  Keep the HBD-count item as written.
- **The deadline is confirmed but not binding** (decided 2026-07-30), so items 4, 5 and 6
  are in scope rather than optional. Items 1, 2 and 3 are still the paper.
