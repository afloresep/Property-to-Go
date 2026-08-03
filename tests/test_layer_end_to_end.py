"""C23 -- end-to-end guided decoding with a mid-network head, bound to its artefacts.

Three kinds of test live here, kept apart on purpose.

**The additive-edit contract.**  `scripts/05_guided_generation.py` gained exactly two
arguments, `--layer` and `--head-file`.  Omitting both must reproduce the pre-edit code
path *exactly*: the head read from `heads_dir/head_<prop>_frozen_state[_seed<s>].pt` and
`layer=-1` handed to `guided_sample`.  Every executed artefact in the repository was
produced by the pre-edit script, so if this drifts the new runs are not comparable with
the old ones and nothing in section C23 means what it says.

**The layer-index contract.**  `--layer 12` must be the same probe point as the default
`-1`, because C23's validity gate is exactly that identity.  Asserted against the real
checkpoint, not argued from the tuple length.

**Artefact binding.**  Every number asserted in
`reports/section_c23_layer_end_to_end.md` is re-read from the JSON it came from,
formatted the way the section formats it, and required to appear in the text -- the
pattern `tests/test_report_matches_artifacts.py` and `tests/test_probe_layers.py` use.
These skip when the artefacts are absent, so a fresh clone passes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "section_c23_layer_end_to_end.md"
SUMMARY = OUT / "c23_summary" / "c23_metrics.json"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def section_text() -> str:
    if not SECTION.exists():
        pytest.skip("section not written yet")
    # The prose uses a typographic minus (U+2212) where `f"{x:.4f}"` emits ASCII "-".
    # Normalising the sign is a typography allowance, not a numeric one: every digit
    # still has to match.
    return SECTION.read_text().replace("−", "-")


# --------------------------------------------------------------------------
# 1. the additive-edit contract
# --------------------------------------------------------------------------

class _FakeGen:
    bos_id, eos_id, pad_id = 0, 1, 2

    def decode(self, seqs):
        return ["CCO" for _ in seqs]


def _run_script_05(monkeypatch, tmp_path, argv_extra):
    """Run script 05's `main()` with the generator, head and sampler replaced.

    Everything the *arguments* touch is real -- the config load, the frozen windows
    and intervals, the resolution of the head path, the value of `layer` -- and only
    the expensive parts are faked, so this exercises the edit rather than a mock of it.
    """
    mod = _load_script("05_guided_generation.py")
    if not (mod.OUTPUT_DIR / "pilot_50k_p2" / "target_intervals.json").exists():
        pytest.skip("phase-2 dataset not present")

    seen: dict = {}

    def fake_load_head(path):
        seen["head_path"] = Path(path)
        return object(), object()

    def fake_guided_sample(gen, **kw):
        seen.setdefault("layers", []).append(kw["layer"])
        meter = kw["meter"]
        meter.add_forward(4 * kw["n_molecules"])
        meter.molecules_returned += kw["n_molecules"]
        return [[0, 5, 6, 1] for _ in range(kw["n_molecules"])]

    monkeypatch.setattr(mod, "load_generator", lambda cfg: _FakeGen())
    monkeypatch.setattr(mod, "load_head", fake_load_head)
    monkeypatch.setattr(mod, "TargetScorer", lambda *a, **k: object())
    monkeypatch.setattr(mod, "guided_sample", fake_guided_sample)

    out_name = f"c23_unittest_{tmp_path.name}"
    argv = ["05_guided_generation.py", "--dataset", "pilot_50k_p2",
            "--heads", "pilot_50k_heads_p2", "--property", "aromatic_rings",
            "--n", "4", "--seeds", "101", "--conditions", "unguided", "throughout",
            "--out", out_name] + argv_extra
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    out_dir = mod.OUTPUT_DIR / out_name
    # Read the artefacts, then remove the scratch directory: a unit test must not
    # leave a run directory under outputs/ that a later reader could mistake for one.
    seen["report"] = json.loads((out_dir / "guidance_metrics.json").read_text())
    seen["configs"] = json.loads((out_dir / "configs_used.json").read_text())
    shutil.rmtree(out_dir)
    return seen


def test_script_05_defaults_reproduce_the_pre_edit_code_path(monkeypatch, tmp_path):
    """No --layer and no --head-file must behave exactly as the script did before C23."""
    seen = _run_script_05(monkeypatch, tmp_path, [])
    assert seen["head_path"].name == "head_aromatic_rings_frozen_state.pt"
    assert seen["head_path"].parent.name == "pilot_50k_heads_p2"
    # -1 is the value `guided_sample` defaulted to before the edit existed.
    assert set(seen["layers"]) == {-1} and len(seen["layers"]) == 2
    report = seen["report"]
    assert report["layer"] == -1
    assert report["layer_source"] == "default (-1)"
    assert report["head_file_source"] == "default"
    assert report["head_checkpoint"] == "head_aromatic_rings_frozen_state.pt"


def test_script_05_head_seed_still_wins_over_the_default_name(monkeypatch, tmp_path):
    seen = _run_script_05(monkeypatch, tmp_path, ["--head-seed", "2345"])
    assert seen["head_path"].name == "head_aromatic_rings_frozen_state_seed2345.pt"
    assert set(seen["layers"]) == {-1}


def test_script_05_layer_and_head_file_are_honoured_and_recorded(monkeypatch, tmp_path):
    seen = _run_script_05(
        monkeypatch, tmp_path,
        ["--layer", "3", "--head-file", "c17_probe_layers/head_aromatic_rings_frozen_state_L3.pt"],
    )
    assert seen["head_path"].name == "head_aromatic_rings_frozen_state_L3.pt"
    assert seen["head_path"].parent.name == "c17_probe_layers"
    assert set(seen["layers"]) == {3}
    report = seen["report"]
    assert report["layer"] == 3
    assert report["layer_source"] == "cli --layer"
    assert report["head_file_source"] == "cli --head-file"
    cfg = seen["configs"]
    assert cfg["cli"]["layer"] == 3
    assert cfg["cli"]["head_file"].endswith("head_aromatic_rings_frozen_state_L3.pt")


# --------------------------------------------------------------------------
# 2. the layer-index contract, against the real checkpoint
# --------------------------------------------------------------------------

@pytest.mark.model
def test_probe_point_12_is_the_default_minus_one(generator):
    """C23's validity gate rests on --layer 12 being the deployed probe point."""
    ids = generator.tokenizer("CCOc1ccccc1", return_tensors="pt").input_ids[:, :-1]
    ids = ids.to(generator.device)
    with torch.no_grad():
        res = generator.model(ids, output_hidden_states=True, use_cache=False)
    assert len(res.hidden_states) == 13
    assert torch.equal(res.hidden_states[-1], res.hidden_states[12])


@pytest.mark.model
def test_guided_sample_layer_12_equals_layer_default(generator):
    """The identity has to hold through `guided_sample`, not only through the tuple."""
    from property_to_go.binning import binner_from_dict
    from property_to_go.config import OUTPUT_DIR, load_config, read_json
    from property_to_go.guidance import TargetScorer, Windows, guided_sample
    from property_to_go.heads import MLPHead

    heads = OUTPUT_DIR / "pilot_50k_heads_p2"
    data = OUTPUT_DIR / "pilot_50k_p2"
    if not (heads / "head_aromatic_rings_frozen_state.pt").exists():
        pytest.skip("phase-2 heads not present")
    ck = torch.load(heads / "head_aromatic_rings_frozen_state.pt",
                    map_location="cpu", weights_only=False)
    head = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.eval()
    iv = read_json(data / "target_intervals.json")["aromatic_rings"]
    win = read_json(data / "windows.json")
    windows = Windows(t33=win["t33"], t67=win["t67"], source=win["source"])
    policy = load_config("base_policy")

    def run(layer):
        return guided_sample(
            generator,
            scorer=TargetScorer(head, binner_from_dict(ck["binner"]),
                                float(iv["lo"]), float(iv["hi"])),
            window_fn=windows.fn("throughout"), policy=policy,
            n_molecules=8, seed=101, top_k=8, lam=1.0, eps=1e-6,
            backend="cached", batch_size=8, layer=layer,
        )

    assert run(-1) == run(12)


# --------------------------------------------------------------------------
# 3. the pre-registration is still the one that was frozen
# --------------------------------------------------------------------------

def test_the_frozen_preregistration_is_verbatim_in_the_section():
    """C23.0 was frozen before the first run; appending results must not edit it."""
    frozen = OUT / "c23_prereg" / "C23.0_preregistration.md"
    lock = OUT / "c23_prereg" / "prereg_lock.json"
    if not frozen.exists():
        pytest.skip("pre-registration not frozen yet")
    if not SECTION.exists():
        pytest.skip("section not written yet")
    import hashlib
    block = frozen.read_text()
    assert block in SECTION.read_text(), "C23.0 has been edited since it was frozen"
    meta = json.loads(lock.read_text())
    assert hashlib.sha256(block.encode()).hexdigest() == meta["prereg_block_sha256"]


def test_the_preregistration_predates_every_c23_output_directory():
    lock = OUT / "c23_prereg" / "prereg_lock.json"
    if not lock.exists():
        pytest.skip("pre-registration not frozen yet")
    t_lock = lock.stat().st_mtime
    produced = [d for d in OUT.glob("c23_*")
                if d.is_dir() and d.name != "c23_prereg"]
    if not produced:
        pytest.skip("no C23 run directories yet")
    assert min(d.stat().st_mtime for d in produced) > t_lock


# --------------------------------------------------------------------------
# 4. artefact binding for section C23
# --------------------------------------------------------------------------

def _fmt(x, n=4):
    return f"{x:.{n}f}"


@pytest.fixture(scope="module")
def summary():
    return _load(SUMMARY)


def test_validity_gate_reproduces_and_the_residual_is_quoted(summary, section_text):
    gate = summary["validity_gate"]
    assert gate["max_abs_hit_rate_residual"] == 0.0, gate
    assert gate["molecules_identical"] is True, gate
    for seed, v in gate["per_seed"].items():
        assert _fmt(v["reference_hit_rate"]) in section_text
    assert _fmt(gate["reference_throughout_mean"]) in section_text
    assert _fmt(gate["replay_throughout_mean"]) in section_text


def test_every_arm_hit_rate_appears_in_the_section(summary, section_text):
    for key, arm in summary["arms"].items():
        assert _fmt(arm["throughout_mean"]) in section_text, key


def test_the_unguided_bug_alarm_holds(summary, section_text):
    """Regenerated `unguided` must reproduce its central-test value at every arm."""
    for key, arm in summary["arms"].items():
        assert _fmt(arm["unguided_mean"]) == _fmt(arm["unguided_reference"]), key


def test_decision_rules_are_scored_from_the_artefact(summary, section_text):
    d = summary["decision_rules"]
    for name in ("layer_improves_guidance", "headline_falsification", "null"):
        assert name in d
    # exactly one of the three verdicts is the reported one
    fired = [k for k in ("layer_improves_guidance", "headline_falsification")
             if d[k]["fires"]]
    assert d["null"]["fires"] == (not fired)
    assert d["null"]["fires"] is not None


def test_best_advantage_over_matched_best_of_n_is_bound(summary, section_text):
    for key, arm in summary["arms"].items():
        if arm.get("best_of_n_hit_rate") is None:
            continue
        assert _fmt(arm["advantage_vs_best_of_n"]) in section_text, key


def test_every_arm_difference_and_interval_appear_in_the_section(summary, section_text):
    """The seed-matched difference against the deployed layer, and its corrected CI."""
    for key, arm in summary["arms"].items():
        assert _fmt(arm["diff_vs_deployed"]) in section_text, f"{key} difference"
        assert _fmt(arm["diff_ci_lo"]) in section_text, f"{key} ci_lo"
        assert _fmt(arm["diff_ci_hi"]) in section_text, f"{key} ci_hi"


def test_every_arm_quality_and_token_ratio_appear_in_the_section(summary, section_text):
    """C23.0.7: a lift with a validity drop is not a win, so validity is quoted too."""
    for key, arm in summary["arms"].items():
        assert _fmt(arm["validity_mean"]) in section_text, f"{key} validity"
        assert _fmt(arm["token_ratio_vs_deployed"]) in section_text, f"{key} token ratio"
        assert _fmt(arm["tokens_per_molecule_actual"], 1) in section_text, f"{key} tok/mol"


def test_the_realised_token_ratio_against_best_of_n_is_quoted(summary, section_text):
    """Token matching has to be reported as realised, never assumed."""
    for key, arm in summary["arms"].items():
        if arm.get("best_of_n_hit_rate") is None:
            continue
        assert _fmt(arm["realised_token_ratio_guided_over_best_of_n"]) in section_text, key
        assert _fmt(arm["best_of_n_hit_rate"]) in section_text, key


def test_the_disqualification_rule_is_applied_and_the_arms_named(summary, section_text):
    dq = {k for k, a in summary["arms"].items() if a["disqualified"]}
    # tripwire on the data, not on the prose: the rule was fixed in C23.0.7 before any
    # number existed, and these are the arms it removes.
    assert dq == {"aromatic_rings_L3_lam2", "aromatic_rings_L6_lam2", "qed_L4_lam4"}, dq
    for k in dq:
        a = summary["arms"][k]
        assert _fmt(a["validity_delta"]) in section_text or \
            _fmt(abs(a["validity_delta"])) in section_text, k


def test_rule_b_fires_on_exactly_one_arm_and_the_conservative_check_is_reported(
        summary, section_text):
    """Tripwire: a re-run that changes the falsification must fail a test."""
    b = summary["decision_rules"]["headline_falsification"]
    assert b["fires"] is True
    assert b["arms_beating_best_of_n"] == ["hbd_count_L4_lam2"], b
    arm = summary["arms"]["hbd_count_L4_lam2"]
    assert arm["advantage_excludes_zero"] is True
    for k in ("advantage_vs_best_of_n", "advantage_ci_lo", "advantage_ci_hi"):
        assert _fmt(arm[k]) in section_text, k
    c = arm["conservative_best_of_n"]
    # the conservative comparator must actually be the better-funded side
    assert c["realised_token_ratio_guided_over_best_of_n"] < 1.0
    assert c["excludes_zero"] is False, "corrected CI must be reported as spanning zero"
    assert c["excludes_zero_uncorrected"] is True
    for k in ("advantage", "ci_lo", "ci_hi", "hit_rate",
              "ci95_uncorrected_lo", "ci95_uncorrected_hi"):
        assert _fmt(c[k]) in section_text, f"conservative {k}"
    # and the survival list must be empty, i.e. the section may not claim it survives
    assert b["survives_token_conservative_check"] == []


def test_rule_a_fires_on_two_of_three_properties(summary, section_text):
    a = summary["decision_rules"]["layer_improves_guidance"]
    assert a["fires"] is True
    assert a["properties_firing"] == 2
    fired = {p for p, v in a["per_property"].items() if v["fires"]}
    assert fired == {"aromatic_rings", "hbd_count"}, fired


def test_every_prediction_is_scored_and_the_failures_are_reported(summary, section_text):
    """C23.0.8 was written to be wrong; §C23.5 has to say so."""
    p = summary["predictions"]
    assert set(p) == {"P_A_within_0.05_at_lambda1",
                      "P_B_steering_best_layer_transfers",
                      "P_C_lambda_curve_shifts_left",
                      "P_D_gap_stays_below_minus_0.05"}
    assert all(v["holds"] is False for v in p.values()), \
        "every sub-prediction failed; the section says so"
    assert _fmt(p["P_A_within_0.05_at_lambda1"]["max_abs_difference"]) in section_text
    assert _fmt(p["P_D_gap_stays_below_minus_0.05"]["best_advantage"]) in section_text


def test_the_multiplicity_correction_is_the_one_that_was_committed(summary):
    m = summary["multiplicity"]
    assert m["n_experimental_arms"] == 15
    assert m["alpha_family"] == 0.05
    assert abs(m["alpha_per_arm"] - 0.05 / 15) < 1e-12
    assert m["n_boot"] == 10000


def test_the_degeneracy_rates_of_the_disqualified_arms_are_quoted(summary, section_text):
    """C23.0.7 makes quality part of the result, so the numbers must be in the prose."""
    for k in ("aromatic_rings_L3_lam2", "aromatic_rings_L6_lam2"):
        a = summary["arms"][k]
        assert _fmt(a["degeneracy_any_guided_hits"]) in section_text, k
        assert _fmt(a["degeneracy_any_base_hits"]) in section_text, k


def test_the_gate_used_the_c17_checkpoint_at_probe_point_12(summary):
    gate = summary["validity_gate"]
    assert gate["replay_layer"] == 12
    assert gate["replay_head_file"].endswith(
        "c17_probe_layers/head_aromatic_rings_frozen_state_L12.pt")
    assert gate["n_molecules_compared"] == 3072
    assert gate["passes"] is True


def test_no_arm_was_dropped_after_the_fact(summary):
    """C23.0.2-0.4 fixed 15 arms; all 15 must be present or explicitly listed as not run."""
    assert summary["n_arms_planned"] == 15
    assert summary["n_arms_run"] + len(summary["arms_not_run"]) == 15


# ------------------------------------------------- the 2026-08-01 interval correction


def test_every_bootstrap_carries_a_seed_level_interval_beside_it():
    """C23's bootstrap resamples molecules WITHIN each seed, so it estimates the
    between-seed component as exactly zero.  That is the wrong variance for a claim about
    runs, and it is the interval Rule B fired on before C25 overturned the arm on
    head-seed variance.  Every bootstrap must now publish a seed-level t interval beside
    it, and the bootstrap must be labelled with the unit it actually resamples."""
    m = _load(OUT / "c23_summary" / "c23_metrics.json")
    pairs = [("diff_excludes_zero", "diff_seed_level_t"),
             ("advantage_excludes_zero", "advantage_seed_level_t"),
             ("excludes_zero", "seed_level_t_interval")]
    found = 0

    def walk(o):
        nonlocal found
        if isinstance(o, dict):
            for key, tkey in pairs:
                if key in o and ("ci_lo" in o or "diff_ci_lo" in o
                                 or "advantage_ci_lo" in o):
                    assert tkey in o, f"{key} present without {tkey}"
                    t = o[tkey]
                    assert t["interval"].startswith("Student t")
                    assert t["n_seeds"] == 3
                    found += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(m)
    assert found >= 30, f"only {found} interval pairs found; expected the full grid"


def test_the_two_intervals_measure_different_variance_components():
    """Neither interval dominates, and assuming one does is a trap.

    The bootstrap resamples ~1536 molecules within seed, so its half-width is set by
    molecule noise and is roughly constant across arms; it is also Bonferroni-corrected at
    alpha = 0.05/15, which makes it about a 99.7% interval and therefore WIDE.  The
    seed-level t is a plain 95% interval on 2 df whose width is set by how much the three
    seeds happen to disagree -- so when they agree closely it comes out NARROWER than the
    bootstrap, not wider.  That is not reassurance: at n = 3 an sd can be small by luck,
    and a narrow t interval is the failure mode that produced Rule B.

    What is asserted is the structural fact rather than an ordering: the bootstrap's
    implied standard error must sit near the molecule-level scale sqrt(2*0.25/1536) for
    every arm, which is what makes it blind to between-seed variance.
    """
    import math
    m = _load(OUT / "c23_summary" / "c23_metrics.json")
    molecule_scale = math.sqrt(2 * 0.25 / 1536)   # ~0.018
    ses = []
    for arm, a in m["arms"].items():
        # corrected interval -> undo the Bonferroni z to recover the implied SE
        half = ((a["advantage_ci95_uncorrected_hi"]
                 - a["advantage_ci95_uncorrected_lo"]) / 2
                if "advantage_ci95_uncorrected_hi" in a else None)
        if half is None:
            continue
        ses.append(half / 1.959964)
    assert ses, "no uncorrected intervals found to check"
    assert all(0.3 * molecule_scale < s < 3 * molecule_scale for s in ses), (
        f"bootstrap SEs {min(ses):.4f}-{max(ses):.4f} are not on the molecule scale "
        f"{molecule_scale:.4f}; the interval may no longer be molecule-level")


def test_rule_a_still_fires_under_the_seed_level_interval():
    """The correction's headline consequence, bound to the data.

    Re-score `layer_improves_guidance` using the seed-level t rather than the
    molecule-level bootstrap.  It must still fire for aromatic rings and HBD count, which
    is what leaves Rule A standing as the project's only positive end-to-end result.
    """
    m = _load(OUT / "c23_summary" / "c23_metrics.json")
    fires = {}
    for prop in ("aromatic_rings", "hbd_count", "qed"):
        qualifying = [name for name, a in m["arms"].items()
                      if a.get("property") == prop
                      and a["diff_vs_deployed"] > 0
                      and a["diff_seed_level_t"]["excludes_zero"]
                      and not a["disqualified"]]
        fires[prop] = bool(qualifying)
    assert fires["aromatic_rings"] is True
    assert fires["hbd_count"] is True
    assert sum(fires.values()) >= 2, fires
