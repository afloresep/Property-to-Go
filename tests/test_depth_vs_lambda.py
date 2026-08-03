"""C32 -- artefact-binding and implementation tests for the depth/lambda de-confound.

C31 ran two corners of a 2x2 and all five of its crossing cells were the same corner, so
"reading mid-network helps" and "steering harder helps" were observationally identical.
C32 runs the missing corners and adds C29's effective-lambda control.

These tests hold six things in place:

1. **The pre-registration was frozen before any measurement**, byte-for-byte, and the
   section copies it verbatim -- including C32.0.2's disclosure that C31's results were
   already known when C32 was designed.
2. **G1 is real.**  C32 reuses C31's `run_ksweep_cell` by import and redirection; G1 proves
   the redirected path reproduces a C31 cell *molecule for molecule*.  Without it, every
   comparison between a C32 corner and a C31 corner is uncontrolled.
3. **The mid probe points were transcribed, never re-selected.**  Re-selecting them after
   seeing which arm crossed is `pilot_report.md` section 21.5.2's failure mode.
4. **The decomposition is the pre-registered arithmetic**, re-derived here from the four
   corners rather than trusted, including the closure identity.
5. **The effective-lambda correction never invents a value** outside the measured envelope.
6. **Every number the section prints is re-derived from the artefacts.**  Machine-derived
   numbers use ASCII hyphens (`f"{x:+.4f}"`), never a Unicode minus.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREREG = OUT / "c32_prereg" / "C32.0_preregistration.md"
LOCK = OUT / "c32_prereg" / "prereg_lock.json"
METRICS = OUT / "c32_summary" / "c32_metrics.json"
SECTION = ROOT / "reports" / "section_c32_depth_vs_lambda.md"

SEEDS = ("101", "202", "303")
K_GRID = (2, 4, 8, 16, 32)
PRIMARY_K = (2, 4)
CORNERS = ("deployed_l1", "deployed_l2", "mid_l1", "mid_l2")
PROPERTIES = ("hbd_count", "aromatic_rings", "qed")
#: C31 selected these by held-out validation AUROC, in advance.  C32 must not move them.
C31_MID_PROBE_POINTS = {"hbd_count": 2, "aromatic_rings": 2, "qed": 6}
ENVELOPE_LAMBDAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


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
        pytest.skip(f"{SECTION} not present")
    return SECTION.read_text()


def f4(x):
    return f"{x:.4f}"


def s4(x):
    return f"{x:+.4f}"


# ==================================================== the pre-registration is a freeze


def test_the_prereg_hash_matches_the_recorded_one():
    lock = _json(LOCK)
    raw = PREREG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock["file_sha256"], (
        "the C32 pre-registration was edited after it was frozen"
    )
    assert len(raw) == lock["file_bytes"]


def test_the_prereg_was_written_before_every_measurement():
    lock = _json(LOCK)
    t0 = PREREG.stat().st_mtime
    checked = 0
    for pattern in ("c32_gates/*.json", "c32_cell_*/*.json", "c32_gate_*/*.json",
                    "c32_spread/*.json", "c32_summary/*.json"):
        for f in OUT.glob(pattern):
            checked += 1
            assert f.stat().st_mtime > t0, (
                f"{f} is older than the pre-registration -- the ordering that makes C32's "
                f"decision rules pre-registered rather than post-hoc is broken"
            )
    if checked == 0:
        pytest.skip("no C32 artefacts yet")
    assert lock["file_mtime_utc"] <= lock["frozen_at_utc"]


def test_the_section_copies_the_prereg_verbatim(section):
    text = PREREG.read_text()
    block = text[text.index("## C32.0.1 Why this experiment exists"):]
    lock = _json(LOCK)
    assert hashlib.sha256(block.encode()).hexdigest() == lock["prereg_block_sha256"]
    assert block.strip() in section, (
        "the C32 section does not reproduce the pre-registration byte for byte"
    )


def test_the_prereg_discloses_what_was_already_known():
    """C32 was designed knowing C31's results.  The disclosure is the mitigation."""
    text = PREREG.read_text()
    assert "## C32.0.2 What I already know" in text
    # the specific C31 advantages that could bias the design must be on the page
    for v in ("+0.2295", "+0.1724", "+0.2473", "+0.1696"):
        assert v in text, f"C32.0.2 does not disclose C31's {v}"


# =============================================================== the method is not forked


def test_c32_imports_c31s_cell_runner_rather_than_copying_it():
    src = (ROOT / "scripts" / "26_c32_depth_vs_lambda.py").read_text()
    assert '_load_module(ROOT / "scripts" / "25_c31_second_generator.py"' in src
    assert "run_ksweep_cell = _c31.run_ksweep_cell" in src
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "run_ksweep_cell" not in defined, "C32 forked C31's cell runner"
    assert "guided_sample" not in defined
    assert "price" not in defined, "C32 forked C31's pricing function"


def test_c32_imports_the_frontier_machinery_unmodified():
    src = (ROOT / "scripts" / "26_summarise_c32.py").read_text()
    for line in ("interp = _c26.interp", "t_interval = _c26.t_interval",
                 "price = _c31s.price", "load_curve = _c31s.load_curve"):
        assert line in src, line
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("interp", "t_interval", "price", "load_curve"):
        assert name not in defined, f"C32 re-defines {name}"


# ============================================================================== the gates


def test_gate_g1_reproduces_c31_molecule_for_molecule(metrics):
    """The gate that makes every C32-vs-C31 comparison controlled."""
    g1 = metrics["validity_gates"]["G1"]
    assert g1["passes"] is True, f"G1 failed: {g1}"
    assert g1["max_abs_hit_rate_residual"] == 0.0
    assert g1["max_abs_token_residual"] == 0.0
    assert g1["n_cells_checked"] >= 2, "G1 must cover both reused 2x2 corners"
    seen = set()
    for name, row in g1["cells"].items():
        if not row.get("checked"):
            continue
        assert row["molecules_all_identical"] is True, name
        assert row["molecules_identical"] == row["molecules_total"] > 0, name
        # identify the corner by what it actually is, not by parsing its name
        seen.add(("final" if row["probe_point"] == 12 else "mid", row["lam"]))
    assert ("final", 1.0) in seen and ("mid", 2.0) in seen, (
        f"G1 must prove BOTH reused corners -- (12, lambda=1) and (M, lambda=2) -- are "
        f"reachable by C32's code path; it covered {sorted(seen)}"
    )


def test_gate_g2_the_frozen_artefacts_are_unchanged(metrics):
    g2 = metrics["validity_gates"]["G2"]
    assert g2["passes"] is True, f"G2 failed: {g2}"
    for name, row in g2["files"].items():
        assert row["matches"] is True, name
    for name, row in g2["heads"].items():
        assert row["exists"] is True, name
        assert row["matches_requested_probe_point"] is True, name
        assert row["matches_requested_property"] is True, name
        assert row["head_seed"] == 1234, name


def test_gate_g3_cost_identity_holds_in_every_cell(metrics):
    g3 = metrics["validity_gates"]["G3"]
    assert g3["passes"] is True, f"G3 failed: {g3}"
    assert g3["max_residual"] == 0
    for name, c in metrics["cells"].items():
        assert c["cost_identity_max_residual"] == 0, name


def test_gate_g4_the_decomposition_is_arithmetically_closed(metrics):
    """The three effects are re-derived from the four corners, and the TRUE closure holds.

    C32.0.4 pre-registered `d - a == depth + lambda + interaction`, which is arithmetically
    false with the pre-registered 0.5 convention.  This test checks the identities that do
    hold; `test_the_preregistration_defect_is_reported` checks that the false one is
    disclosed rather than quietly dropped.
    """
    g4 = metrics["validity_gates"]["G4"]
    assert g4["passes"] is True, f"G4 failed: {g4}"
    assert g4["max_abs_residual"] < 1e-12
    n = 0
    for key, r in metrics["decomposition"].items():
        if not r.get("complete"):
            continue
        for i in range(len(SEEDS)):
            a = r["corners"]["a"]["advantage_by_seed"][SEEDS[i]]
            b = r["corners"]["b"]["advantage_by_seed"][SEEDS[i]]
            c = r["corners"]["c"]["advantage_by_seed"][SEEDS[i]]
            d = r["corners"]["d"]["advantage_by_seed"][SEEDS[i]]
            dep = r["depth_main"]["per_seed"][i]
            lam = r["lambda_main"]["per_seed"][i]
            inter = r["interaction"]["per_seed"][i]
            assert abs(dep - 0.5 * ((c - a) + (d - b))) < 1e-12, key
            assert abs(lam - 0.5 * ((b - a) + (d - c))) < 1e-12, key
            assert abs(inter - 0.5 * ((d - c) - (b - a))) < 1e-12, key
            # the closure that actually holds
            assert abs((dep + lam) - (d - a)) < 1e-12, key
            # the interaction's two equivalent forms must agree
            assert abs(inter - 0.5 * ((d - b) - (c - a))) < 1e-12, key
            n += 1
    assert n > 0


def test_the_preregistration_defect_is_reported_not_amended(metrics, section):
    """C32.0.4's closure claim is false.  The project's rule is to report, not amend."""
    raw = PREREG.read_text()
    assert "d - a = depth_main + lambda_main + interaction" in raw, (
        "the pre-registration was edited to remove the defective claim"
    )
    d = metrics["preregistration_defects"]["D1"]
    assert d["status"] == "FALSE as written"
    assert d["measured_residual_of_the_false_claim"] > 1e-6, (
        "the defect is claimed but the measured residual does not show it"
    )
    assert "NOT amended" in d["handling"] or "not amended" in d["handling"]
    assert "C32.1.1" in section, "the section does not report the pre-registration defect"


def test_gate_g5_the_comparator_is_c31s_curve_and_c32_draws_no_pool(metrics):
    g5 = metrics["validity_gates"]["G5"]
    assert g5["passes"] is True, f"G5 failed: {g5}"
    assert g5["no_best_of_n_stage_defined"] is True
    assert g5["no_pool_sampler_called"] is True
    for prop, row in g5["curves"].items():
        assert row["is_oracle_selected"] is True, prop
    # and the curve the summariser priced against really is the C31 file
    for prop, row in g5["curves"].items():
        f = OUT / f"c31_bestofn_{prop}" / "n_sweep_metrics.json"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"], prop


def test_no_decision_rule_is_scored_past_a_failed_gate(metrics):
    u = metrics["uninterpretability"]
    if metrics["verdict"]["verdict"] != "UNINTERPRETABLE":
        assert u["all_gates_pass"] is True
        assert u["experiment_degenerate"] is False


# ================================================================== design integrity


def test_the_mid_probe_points_were_transcribed_not_reselected(metrics):
    """section 21.5.2's failure mode: selecting a probe point on a steering outcome."""
    for prop, row in metrics["mid_probe_points"].items():
        assert row["M"] == C31_MID_PROBE_POINTS[prop], (
            f"{prop}: C32 uses probe point {row['M']} where C31 selected "
            f"{C31_MID_PROBE_POINTS[prop]} -- a probe point was re-selected"
        )
        assert "TRANSCRIBED" in row["source"]
        assert "NOT re-selected" in row["source"]
    # and the cells really do sit at those probe points
    for name, c in metrics["cells"].items():
        expect = 12 if c["depth"] == "final" else C31_MID_PROBE_POINTS[c["property"]]
        assert c["probe_point"] == expect, name


def test_the_two_reused_corners_point_at_c31_directories(metrics):
    """C32 must not silently re-run the corners it claims to reuse."""
    for name, c in metrics["cells"].items():
        if c["corner"] in ("deployed_l1", "mid_l2"):
            assert c["source"] == "c31", name
            assert c["dir"].startswith("c31_ksweep_"), (
                f"{name} claims to reuse C31 but sits in {c['dir']}"
            )
        else:
            assert c["source"] == "c32", name
            assert c["dir"].startswith("c32_cell_"), name


def test_the_two_by_two_is_complete_at_the_primary_k(metrics):
    for prop in PROPERTIES:
        for k in PRIMARY_K:
            r = metrics["decomposition"].get(f"{prop}_k{k}")
            assert r is not None, f"{prop} k={k} missing from the decomposition"
            assert r["complete"] is True, f"{prop} k={k}: {r.get('missing')}"
            assert set(r["corners"]) == {"a", "b", "c", "d"}


def test_every_cell_has_all_three_generation_seeds(metrics):
    for name, c in metrics["cells"].items():
        assert sorted(c["per_seed"]) == sorted(SEEDS), name
        assert len(c["hit_rate_values"]) == 3, name


def test_the_uncertainty_is_a_t_interval_on_two_df_and_not_a_bootstrap(metrics):
    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    assert not [k for k in keys(metrics) if "bootstrap" in k.lower()]
    n = 0
    for key, r in metrics["decomposition"].items():
        if not r.get("complete"):
            continue
        for name in ("depth_main", "lambda_main", "interaction"):
            ti = r[name]["t_interval"]
            vals = r[name]["per_seed"]
            assert ti["n_seeds"] == 3, (key, name)
            assert "2 df" in ti["note"], (key, name)
            mean = sum(vals) / 3
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2)
            half = 4.302653 * sd / math.sqrt(3)
            assert abs(ti["lo"] - (mean - half)) < 1e-9, (key, name)
            assert abs(ti["hi"] - (mean + half)) < 1e-9, (key, name)
            n += 1
    assert n > 0


def test_the_effective_lambda_correction_never_invents_a_value(metrics):
    """Outside the measured envelope the bracket is reported, not a number."""
    for key, r in metrics["effective_lambda"]["rows"].items():
        if r.get("extrapolated_beyond_envelope"):
            assert "depth_corrected" not in r, (
                f"{key} is flagged extrapolated yet still carries a corrected value"
            )
        if r.get("envelope", {}).get("ok"):
            lo, hi = r["envelope"]["bracket"]
            assert lo <= r["lambda_effective"] <= hi, key
            assert lo in ENVELOPE_LAMBDAS and hi in ENVELOPE_LAMBDAS, key


def test_the_effective_lambda_is_the_spread_ratio_times_lambda(metrics):
    sp = metrics["spread"]["properties"]
    for key, r in metrics["effective_lambda"]["rows"].items():
        assert abs(r["spread_ratio"] - sp[r["property"]]["spread_ratio"]) < 1e-12, key
        assert abs(r["lambda_effective"] - r["lam"] * r["spread_ratio"]) < 1e-12, key


def test_the_correction_is_labelled_first_order_not_an_identity(metrics, section):
    assert "FIRST-ORDER CONTROL" in metrics["effective_lambda"]["caveat"]
    assert "first-order" in section.lower()


def test_the_spread_ratio_is_paired_across_probe_points(metrics):
    sp = metrics["spread"]
    assert "same forward pass" in sp["pairing"]
    for prop, row in sp["properties"].items():
        assert str(row["final_probe_point"]) in row["spread"]
        assert str(row["mid_probe_point"]) in row["spread"]
        ref = row["spread"][str(row["final_probe_point"])]
        mid = row["spread"][str(row["mid_probe_point"])]
        assert abs(row["spread_ratio"] - mid / ref) < 1e-12, prop


def test_the_verdict_follows_the_pre_registered_rule_not_the_prose(metrics):
    dr = metrics["decision_rules"]
    v = metrics["verdict"]["verdict"]
    if metrics["uninterpretability"]["uninterpretable"]:
        assert v == "UNINTERPRETABLE"
    elif dr["D1"]["fires"] and not dr["D2"]["fires"]:
        assert v == "LAMBDA-DOMINATED"
    elif dr["D2"]["fires"] and not dr["D1"]["fires"]:
        assert v == "DEPTH-DOMINATED"
    elif dr["D3"]["fires"]:
        assert v == "INTERACTION-DOMINATED"
    else:
        assert v == "MIXED"


def test_d7_the_honesty_rule_is_scored_whatever_the_others_did(metrics, section):
    d7 = metrics["decision_rules"]["D7"]
    assert "D7" in section
    for name in d7["contrasts"]:
        key = name.split(":")[0]
        assert key in section, f"D7 fired on {name} but the section does not name {key}"


# =================================================== every printed number is re-derivable


def test_the_section_prints_the_gate_residuals(metrics, section):
    g1 = metrics["validity_gates"]["G1"]
    assert s4(g1["max_abs_hit_rate_residual"]) in section or "0.0" in section
    total = sum(r["molecules_total"] for r in g1["cells"].values() if r.get("checked"))
    same = sum(r["molecules_identical"] for r in g1["cells"].values() if r.get("checked"))
    assert str(total) in section and str(same) in section, (
        "the section does not state how many molecules G1 compared"
    )


def test_the_section_prints_every_two_by_two_corner(metrics, section):
    for key, r in metrics["decomposition"].items():
        if not r.get("complete") or not r["primary_k"]:
            continue
        for x in ("a", "b", "c", "d"):
            assert s4(r["corners"][x]["advantage"]) in section, f"{key} corner {x}"


def test_the_section_prints_the_decomposition_with_intervals(metrics, section):
    for key, r in metrics["decomposition"].items():
        if not r.get("complete") or not r["primary_k"]:
            continue
        for name in ("depth_main", "lambda_main", "interaction"):
            assert s4(r[name]["mean"]) in section, f"{key} {name}"
            ti = r[name]["t_interval"]
            assert s4(ti["lo"]) in section, f"{key} {name} lo"
            assert s4(ti["hi"]) in section, f"{key} {name} hi"


def test_the_section_prints_the_per_seed_values_in_full(metrics, section):
    for key, r in metrics["decomposition"].items():
        if not r.get("complete") or not r["primary_k"]:
            continue
        for name in ("depth_main", "lambda_main", "interaction"):
            for v in r[name]["per_seed"]:
                assert s4(v) in section, f"{key} {name} per-seed {v}"


def test_the_section_prints_the_spread_ratios(metrics, section):
    for prop, row in metrics["spread"]["properties"].items():
        assert f4(row["spread_ratio"]) in section, prop
        assert f4(row["lambda_effective_at_lam2"]) in section, prop


def test_the_section_prints_the_corrected_depth_contrast(metrics, section):
    for key, r in metrics["effective_lambda"]["rows"].items():
        if "depth_corrected" not in r:
            continue
        assert s4(r["depth_raw"]) in section, f"{key} raw"
        assert s4(r["depth_corrected"]) in section, f"{key} corrected"


def test_every_decision_rule_is_scored_in_the_section(metrics, section):
    for name in metrics["decision_rules"]:
        assert name in section, f"decision rule {name} is not scored in the section"
    assert metrics["verdict"]["verdict"] in section


def test_every_prediction_is_scored_including_the_falsified_ones(metrics, section):
    P = metrics["predictions"]
    assert P
    for name, p in P.items():
        assert name in section, f"prediction {name} is not scored in the section"
        assert p["outcome"] in ("CONFIRMED", "FALSIFIED")
    if metrics["prediction_summary"]["falsified"]:
        assert "FALSIFIED" in section


def test_the_prediction_expected_to_fail_is_identified_as_such(metrics, section):
    """C32.0.9 P9 was committed as expected-to-fail.  That must survive into the section."""
    assert "EXPECTED TO FAIL" in metrics["predictions"]["P9"]["statement"]
    assert "P9" in section


def test_the_section_has_the_required_structure(section):
    for heading in ("What C32 changes elsewhere", "Limitations", "REPRODUCE"):
        assert heading in section, f"the section is missing its {heading!r} part"
