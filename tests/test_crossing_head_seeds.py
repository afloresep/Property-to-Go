"""C30 -- the probe-seed replication of C28's crossing.

C28 is the only positive result in this project and it was produced at one probe-training
seed.  C29 then measured a probe-seed sd of 0.0142-0.0366 on end-to-end hit rate, which is
larger than three of C28's eight margins.  C30 re-runs the eight winning cells at all eight
of C29's head seeds.

These tests hold three things in place:

1. **The pre-registration was frozen before any measurement**, byte-for-byte, and the
   section copies it verbatim.  Enforced by mtime and SHA-256, the same machinery C23-C29
   use.
2. **The gates are real.**  G1 (checkpoint identity) and G2 (head seed 1234 reproduces C28
   exactly, molecules included) are the reason any of the other seven seeds mean anything.
   A test that only checked the *conclusion* would pass on a run whose gate silently failed.
3. **The verdict follows the pre-registered rule** rather than the prose.  In particular
   "SURVIVES, UNDERPOWERED" may not be reported as "CONFIRMED"; that distinction is scored
   from artefacts here, not left to the writing.

The t critical value is pinned: at n = 8 head seeds the interval is on 7 df, t = 2.364624.
Before 2026-08-03 the shared helper fell back to 1.96 for any unlisted df, which is 17% too
narrow at 7 df and would have manufactured significance.  `test_the_t_critical_value_is_the
_one_for_seven_degrees_of_freedom` pins the fix and re-pins the n = 3 value that C23-C28
depend on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREREG = OUT / "c30_prereg" / "C30.0_preregistration.md"
LOCK = OUT / "c30_prereg" / "prereg_lock.json"
METRICS = OUT / "c30_summary" / "c30_metrics.json"
SECTION = ROOT / "reports" / "section_c30_crossing_head_seeds.md"

HEAD_SEEDS = (1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901)
GATE_HEAD_SEED = 1234


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _json(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def metrics():
    return _json(METRICS)


@pytest.fixture(scope="module")
def section():
    if not SECTION.exists():
        pytest.skip("section_c30_crossing_head_seeds.md not present")
    return SECTION.read_text()


def assert_in(section: str, value: str, what: str) -> None:
    assert value in section, f"the C30 section does not print {value} ({what})"


# ------------------------------------------------------------------ the statistic itself

def test_the_t_critical_value_is_the_one_for_seven_degrees_of_freedom():
    """n = 8 head seeds -> 7 df -> 2.364624.  The old fallback was the normal 1.96.

    This is the single change most able to turn a null into a finding by accident, so it is
    pinned rather than trusted.  The n = 3 value C23-C28 rely on is re-pinned in the same
    test so that extending the table cannot have moved it.
    """
    c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_for_test")
    assert c26.T_CRIT_95[7] == 2.364624
    assert c26.T_CRIT_95[2] == 4.302653, "the n = 3 critical value moved; C23-C28 depend on it"

    ti = c26.t_interval([1.0] * 7 + [2.0])
    n, sd = 8, ti["sd"]
    expected_half = 2.364624 * sd / math.sqrt(n)
    assert ti["n_seeds"] == 8
    assert "7 df" in ti["note"]
    assert abs((ti["hi"] - ti["lo"]) / 2 - expected_half) < 1e-12, (
        "the 8-value interval is not built from t(7); a normal quantile would be 17% narrower"
    )


def test_a_three_seed_percentile_bootstrap_is_not_the_primary_statistic(metrics):
    """The rule this project adopted after finding the n = 3 bootstrap vacuous."""
    blob = json.dumps(metrics)
    assert "advantage_seed_bootstrap_ci" not in blob
    for cell in metrics["cells"].values():
        if not cell.get("n_head_seeds"):
            continue
        assert cell["advantage_t_interval"]["n_seeds"] == len(HEAD_SEEDS)
        assert "7 df" in cell["advantage_t_interval"]["note"]


# --------------------------------------------------------------------- the pre-registration

def test_the_prereg_hash_matches_the_recorded_one():
    lock = _json(LOCK)
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock["file_sha256"], (
        "the C30 pre-registration was edited after it was frozen"
    )
    assert len(raw) == lock["file_bytes"]


def test_the_prereg_was_written_before_every_measurement():
    """mtime ordering, the same enforcement C23-C29 use."""
    lock = _json(LOCK)
    t0 = PREREG.stat().st_mtime
    checked = 0
    for pattern in ("c30_hs*/k_cell_metrics.json", "c30_gates/*.json",
                    "c30_summary/*.json"):
        for f in OUT.glob(pattern):
            checked += 1
            assert f.stat().st_mtime > t0, (
                f"{f} is older than the pre-registration -- the ordering that makes C30's "
                f"decision rules pre-registered rather than post-hoc is broken"
            )
    if checked == 0:
        pytest.skip("no C30 artefacts yet")
    assert lock["file_mtime_utc"] <= lock["frozen_at_utc"]


def test_the_section_copies_the_prereg_verbatim(section):
    text = PREREG.read_text()
    block = text[text.index("## C30.0.1 Why this experiment exists"):]
    lock = _json(LOCK)
    assert hashlib.sha256(block.encode()).hexdigest() == lock["prereg_block_sha256"]
    # the section must contain the block, not a paraphrase of it
    assert block.strip() in section, (
        "the C30 section does not reproduce the pre-registration byte for byte"
    )


def test_the_cells_c30_tests_are_the_cells_c28_actually_found(metrics):
    """C30 was designed around eight named cells.  If C28's artefact now yields a different
    set, C30 is testing a stale list and must say so rather than score it silently."""
    agree = metrics["c28_cell_agreement"]
    if not agree.get("checked"):
        pytest.skip("c28_metrics.json absent")
    assert agree["same_cells"], (
        f"C28 now reports {agree['n_derived']} winning cells, not the "
        f"{agree['n_transcribed']} C30 was pre-registered against: {agree['derived']}"
    )
    assert agree["max_abs_margin_discrepancy"] < 5e-5, (
        "a transcribed C28 margin disagrees with the artefact by more than rounding"
    )


# --------------------------------------------------------------------------- the gates

def test_gate_g1_checkpoint_identity_passes_and_is_exact(metrics):
    g1 = metrics["validity_gates"]["G1"]
    assert g1.get("passes") is True, f"G1 failed: {g1}"
    assert g1["max_abs_diff_over_all_families"] == 0.0, (
        "the C29 seed-1234 checkpoints are not bit-identical to the ones C28 used, so "
        "nothing in C30 is comparable to C28"
    )
    assert g1["n_families_checked"] == 3
    for name, row in g1["families"].items():
        assert row["binner_edges_identical"], f"{name}: binner edges differ"
        assert row["same_keys"], f"{name}: state dict keys differ"


def test_gate_g2_reproduces_c28_exactly_including_the_molecules(metrics):
    """A summary statistic can agree by accident; 1536 SMILES strings cannot."""
    g2 = metrics["validity_gates"]["G2"]
    assert g2.get("passes") is True, f"G2 failed: {g2}"
    assert g2["max_abs_hit_rate_residual"] == 0.0
    assert g2["max_abs_token_residual"] == 0.0
    assert g2["n_cells_checked"] == g2["n_cells_passing"] > 0
    for name, cell in g2["cells"].items():
        if not cell.get("checked"):
            continue
        assert cell["molecules_all_identical"] is not False, (
            f"{name}: head seed 1234 returned different molecules from C28"
        )
        if cell.get("molecules_total"):
            assert cell["molecules_identical"] == cell["molecules_total"]


def test_gate_g3_cost_identity_holds_in_every_cell(metrics):
    """G3 scans every directory the run produced, which is more than the scored set.

    C30.0.2 specifies 4 strands x 2 values of k = 8 combinations x 8 head seeds = 64 cells.
    Only 7 of those 8 combinations are C28 winning cells; A1 at k = 4 is the eighth and is
    reported under `run_but_not_scored`.  So G3's denominator is 64, not 7 x 8 = 56.
    """
    g3 = metrics["validity_gates"]["G3"]
    assert g3.get("passes") is True, f"G3 failed: {g3}"
    assert g3["max_residual"] == 0
    n_scored = len(metrics["cells"])
    n_extra = 1 if metrics.get("run_but_not_scored") else 0
    assert g3["n_cells_checked"] == (n_scored + n_extra) * len(HEAD_SEEDS), (
        f"G3 checked {g3['n_cells_checked']} cells; expected every generated directory"
    )


def test_the_cell_that_was_run_but_not_scored_is_reported(metrics, section):
    """A generated cell that is never mentioned is indistinguishable from a dropped one."""
    extra = metrics.get("run_but_not_scored")
    if not extra:
        pytest.skip("no unscored cell in this run")
    assert extra["n_head_seeds"] == len(HEAD_SEEDS)
    assert extra["why_not_scored"]
    assert extra["cell"] in section, (
        f"{extra['cell']} was generated and is not mentioned in the section"
    )
    assert_in(section, f"{extra['advantage_mean']:+.4f}", "the unscored cell's mean")


def test_no_decision_rule_is_scored_past_a_failed_gate(metrics):
    """C30.0.3 forbids it, and C30.0.8 makes a failed gate UNINTERPRETABLE."""
    gates_ok = all(metrics["validity_gates"][g]["passes"] for g in ("G1", "G2", "G3"))
    assert metrics["verdict"]["gates_pass"] == gates_ok
    if not gates_ok:
        assert metrics["verdict"]["verdict"] == "UNINTERPRETABLE"


# ------------------------------------------------------------------- coverage of the run

def test_every_cell_has_all_eight_head_seeds(metrics):
    for name, cell in metrics["cells"].items():
        assert cell["n_head_seeds"] == len(HEAD_SEEDS), (
            f"{name} has {cell['n_head_seeds']} head seeds, not 8; C30.0.8 makes an "
            f"unequal-df comparison uninterpretable"
        )
        assert cell["head_seeds_missing"] == []


def test_the_validity_screen_agrees_with_the_per_cell_validities(metrics):
    """C30.0.8's screen fired.  The test is that it fired *correctly*, not that it didn't.

    Written before the run as `assert validity_min >= 0.90`, which is the wrong test: it
    asserts an outcome rather than checking that the pre-registered screen was applied
    faithfully.  The screen is the mechanism that makes a low-validity cell reportable
    instead of silently averaged in, so what must be pinned is that every failing point is
    enumerated and none is missed.
    """
    screen = metrics["validity_screen"]
    assert screen["threshold"] == 0.90
    derived = [(name, hs) for name, cell in metrics["cells"].items()
               for hs, r in cell["per_head_seed"].items() if r["validity_mean"] < 0.90]
    reported = [(f["cell"], str(f["head_seed"])) for f in screen["failures"]]
    assert sorted(derived) == sorted(reported), (
        f"the screen reports {reported} but the per-cell validities imply {derived}"
    )
    assert screen["n_points_failing"] == len(derived)
    assert screen["passes"] == (not derived)
    assert screen["n_points_checked"] == sum(
        len(c["per_head_seed"]) for c in metrics["cells"].values())


def test_a_low_validity_cell_is_never_silently_averaged_into_a_headline(metrics, section):
    """If the screen fired, the section must show the offending point, not just note it."""
    screen = metrics["validity_screen"]
    if screen["passes"]:
        pytest.skip("no validity failure in this run")
    for f in screen["failures"]:
        assert_in(section, f"{f['validity']:.4f}", f"{f['cell']} hs{f['head_seed']} validity")
        assert str(f["head_seed"]) in section
    # and the sensitivity analysis must drop the WHOLE cell, not the failing seed alone
    s1 = metrics["sensitivity_S1_drop_cells_failing_the_validity_screen"]
    assert set(s1["cells_dropped"]) == set(screen["cells_affected"])
    assert "POST HOC" in s1["status"]


def test_the_comparator_is_held_fixed_across_head_seeds(metrics):
    """The whole design.  If the oracle curve moved with the head seed, the spread in the
    advantage would not be attributable to the probe."""
    assert "oracle-selected" in metrics["comparator"]
    assert "FIXED" in metrics["comparator"] or "fixed" in metrics["comparator"]
    c26 = _load_module(ROOT / "scripts" / "21_summarise_c26.py", "c26_for_curve")
    c28 = _load_module(ROOT / "scripts" / "23_summarise_c28.py", "c28_for_curve")
    for name, cell in metrics["cells"].items():
        curves = c28.load_curves(cell["property"])
        if curves is None:
            pytest.skip(f"no C26 curve for {cell['property']}")
        # The interpolated comparator DOES move across head seeds -- because the budget
        # moves, not because the curve does.  The strong check is therefore not "it barely
        # moves" (it does; C2_k4's budget spans 239-373 tokens) but "it is exactly what
        # reading the ONE fixed curve at that budget gives".
        for hs, r in cell["per_head_seed"].items():
            expected, _, _, _ = c26.interp(curves["tokens_per_molecule_actual"],
                                           curves["oracle_selected"],
                                           r["tokens_per_molecule_actual"])
            assert abs(r["oracle_selected_interpolated_hit_rate"] - expected) < 1e-12, (
                f"{name} hs{hs}: the comparator is not C26's fixed oracle curve read at "
                f"this cell's own budget"
            )
            # and the advantage must be the difference, not something else
            assert abs(r["advantage_vs_oracle_selected"]
                       - (r["hit_rate_mean"] - expected)) < 1e-12


def test_the_secondary_head_selected_comparison_drives_no_decision_rule(metrics):
    """C30.0.4 makes it explicitly secondary because that curve moves with the head."""
    blob = json.dumps(metrics["decision_rules"])
    assert "head_selected" not in blob, (
        "a decision rule reads the head-selected curve, which is not held fixed across "
        "head seeds and therefore cannot isolate the probe-seed effect"
    )


# ----------------------------------------------------------------------- the verdict

def test_the_verdict_follows_the_pre_registered_rule_not_the_prose(metrics):
    r = metrics["decision_rules"]
    v = metrics["verdict"]["verdict"]
    d1, d2, d5 = (r["D1_crossing_survives"]["fires"],
                  r["D2_crossing_confirmed"]["fires"],
                  r["D5_crossing_refuted"]["fires"])
    # C30.0.8 lists THREE voiding conditions, not one: a failed gate, a cell below 0.90
    # validity, or unequal head-seed counts.  Any of them precedes the D-rules.
    voided = (not metrics["verdict"]["gates_pass"]
              or not metrics["verdict"]["all_validity_at_least_0.90"]
              or not metrics["verdict"]["all_cells_have_eight_head_seeds"])
    if voided:
        expected = "UNINTERPRETABLE"
    elif d5:
        expected = "REFUTED"
    elif d1 and d2:
        expected = "CONFIRMED"
    elif d1:
        expected = "SURVIVES, UNDERPOWERED"
    else:
        expected = "AMBIGUOUS"
    assert v == expected, f"verdict {v} does not follow the rule; expected {expected}"


def test_an_underpowered_result_is_not_written_up_as_a_confirmation(metrics, section):
    """The wording was committed to in C30.0.5 before the run."""
    if metrics["verdict"]["verdict"] != "SURVIVES, UNDERPOWERED":
        pytest.skip("verdict is not SURVIVES, UNDERPOWERED")
    committed = metrics["verdict"]["committed_wording"]
    assert committed
    assert committed in section, "the pre-committed wording is not in the section"
    lowered = section.lower()
    assert "the crossing is confirmed" not in lowered, (
        "the section claims confirmation on an underpowered verdict"
    )


def test_d6_the_honesty_rule_is_scored_whatever_the_others_did(metrics, section):
    """D6 is about C28's protocol and fires independently of whether C30 is positive."""
    d6 = metrics["decision_rules"]["D6_not_resolvable_at_one_seed"]
    assert "per_cell" in d6 and d6["per_cell"], "D6 was not scored"
    for name, row in d6["per_cell"].items():
        assert row["head_seed_sd"] is not None, f"{name}: D6 needs a head-seed sd"
    if d6["fires"]:
        assert "not resolvable at one probe seed" in section.lower(), (
            "D6 fires and the section does not use the wording committed to in C30.0.5"
        )


def test_every_prediction_is_scored_including_the_falsified_ones(metrics, section):
    preds = metrics["predictions"]
    assert len(preds) == 7
    for name, p in preds.items():
        assert p.get("holds") is not None, f"{name} was not scored"
    summary = metrics["predictions_summary"]
    assert summary["n_total"] == 7
    n_false = len(summary["falsified"])
    assert summary["n_holding"] + n_false == 7
    if n_false:
        assert "falsified" in section.lower() or "does not hold" in section.lower(), (
            f"{n_false} predictions were falsified and the section does not say so"
        )


# --------------------------------------------------------- the section prints what it scored

def test_the_section_prints_every_cell_mean_and_interval(metrics, section):
    for name, cell in metrics["cells"].items():
        assert_in(section, f"{cell['advantage_mean']:+.4f}", f"{name} head-seed mean")
        ti = cell["advantage_t_interval"]
        assert_in(section, f"{ti['lo']:+.4f}", f"{name} interval lower end")
        assert_in(section, f"{ti['hi']:+.4f}", f"{name} interval upper end")
        assert_in(section, f"{cell['c28_single_seed_margin']:+.4f}", f"{name} C28 margin")


def test_the_section_prints_the_per_head_seed_values_in_full(metrics, section):
    """C30.0.4 promised the whole sample, not a summary of it."""
    for name, cell in metrics["cells"].items():
        for hs, adv in cell["advantage_by_head_seed"].items():
            assert_in(section, f"{adv:+.4f}", f"{name} at head seed {hs}")


def test_the_section_names_the_cells_that_were_not_re_run(metrics, section):
    """A3 k=8 is excluded by design; an unexplained absence reads as a dropped result."""
    cited = metrics["cited_not_rerun"]
    assert cited["cell"] == "A3_k8"
    assert "A3" in section and "k = 8" in section.replace("k=8", "k = 8")
    assert "C29" in section, "the section must cite where A3 k=8 was replicated instead"
