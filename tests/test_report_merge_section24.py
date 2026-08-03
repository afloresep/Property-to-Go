"""§24 of `reports/pilot_report.md` merges C31 (a second generator) and C32 (depth vs λ).

§24 is the only place in the report where numbers from **two different generators** sit in
the same table, which is exactly where a claim gets attached to the wrong one.  So the tests
here do two things beyond the usual re-derivation:

* every cross-generator row is checked against the artefact of the generator it names, and
* the claims §24 *scopes* to one generator are asserted to be scoped -- `test_the_withdrawn_
  compute_knob_claim_is_scoped_not_deleted` fails if §24 states the GP-MoLFormer version of
  the compute-knob result as a general claim again.

The second kind matters more than the first.  C31's own conflict list opens by saying the
sentence §22.2 leans on does not survive a second generator; a test that only re-derived
numbers would pass while that sentence stood unqualified.

ASCII hyphens throughout: `f"{x:+.4f}"` emits U+002D, and a Unicode minus in a bound cell is
indistinguishable in a browser and fatal to the assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = ROOT / "reports" / "pilot_report.md"

ARMS = ("hbd_count_deployed", "hbd_count_mid", "aromatic_rings_deployed",
        "aromatic_rings_mid", "qed_deployed", "qed_mid")
PROPS = ("hbd_count", "aromatic_rings", "qed")


def _metrics(name: str) -> dict:
    p = OUT / f"{name}_summary" / f"{name}_metrics.json"
    if not p.exists():  # pragma: no cover
        pytest.skip(f"{p} not present")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def section() -> str:
    if not REPORT.exists():  # pragma: no cover
        pytest.skip("pilot_report.md not present")
    text = REPORT.read_text()
    marker = "## 24. Result 12"
    assert marker in text, "§24, the second-generator merge, is missing from the report"
    return text[text.index(marker):]


@pytest.fixture(scope="module")
def c31() -> dict:
    return _metrics("c31")


@pytest.fixture(scope="module")
def c32() -> dict:
    return _metrics("c32")


def assert_in(section: str, value: str, what: str) -> None:
    assert value in section, f"§24 does not print {value} ({what})"


def flat(section: str) -> str:
    """The section with markdown line-wrapping and blockquote markers collapsed.

    Prose assertions must run against this, not the raw text: a phrase that happens to
    straddle a line break -- or a `> ` blockquote continuation -- is present to a reader and
    absent to `in`.  Numeric assertions stay on the raw text, where a line break inside a
    formatted number would be a genuine defect.
    """
    import re
    return re.sub(r"\s*\n>?\s*", " ", section)


# ----------------------------------------------------- the deviation is on the record

def test_the_lifted_no_second_generator_constraint_is_disclosed(section):
    """The brief said 'no second generator'.  Running one is a deviation, not a default."""
    text = flat(section)
    assert "no second generator" in text
    assert "lifted" in text.lower()
    assert "recorded rather than silent" in text or "not silent" in text.lower()


def test_the_second_generator_is_named_with_its_architecture(section):
    assert "gpt2_zinc_87m" in section
    assert "SMILES" in section, "the section must say the serialization did not change"


# -------------------------------------------------- 24.1  what replicates and what does not

def test_the_compute_knob_advantage_table_is_re_derived_from_c31(section, c31):
    """The row that carries the surviving two-generator claim."""
    cells = c31["cells"]
    for arm in ARMS:
        a2 = cells[f"{arm}_k2"]["advantage_vs_oracle_selected"]
        a32 = cells[f"{arm}_k32"]["advantage_vs_oracle_selected"]
        assert_in(section, f"{a2:+.4f}", f"{arm} advantage at k=2")
        assert_in(section, f"{a32:+.4f}", f"{arm} advantage at k=32")
        assert_in(section, f"{a32 - a2:+.4f}", f"{arm} change across the k span")
        assert a32 < a2, (
            f"{arm} no longer loses ground across the k span; §24.1's 'six of six arms lose "
            f"ground' is stale"
        )


def test_the_withdrawn_compute_knob_claim_is_scoped_not_deleted(section, c31):
    """§22.1.2 said k 'converts none of it into accuracy'.  That is generator-specific.

    The failure mode this guards against is restating the GP-MoLFormer version as a general
    claim after C31 refuted it.  So the section must contain BOTH the refuting number and
    the surviving, narrower claim.
    """
    d3 = c31["decision_rules"]["D3"]["per_arm"]
    biggest = max(v["delta"] for v in d3.values())
    assert_in(section, f"{biggest:+.4f}".lstrip("+"), "the largest raw hit-rate gain from k")
    assert "advantage" in section.lower()
    lowered = section.lower()
    assert "loses a race" in lowered or "never becomes" in lowered or "loses ground" in lowered


# ------------------------------------------------------------- 24.2  the 2x2 decomposition

def test_the_2x2_main_effects_and_intervals_come_from_c32(section, c32):
    for name, v in c32["decomposition"].items():
        if not v.get("complete") or v.get("k") not in (2, 4):
            continue
        for key in ("depth_main", "lambda_main", "interaction"):
            t = v[key]["t_interval"]
            assert_in(section, f"{t['mean']:+.4f}", f"{name} {key} mean")
            assert_in(section, f"{t['lo']:+.4f}", f"{name} {key} interval low")
            assert_in(section, f"{t['hi']:+.4f}", f"{name} {key} interval high")


def test_lambda_beats_depth_in_every_primary_cell(section, c32):
    """§24.2's headline.  Re-derived, not quoted."""
    primary = [v for v in c32["decomposition"].values()
               if v.get("complete") and v.get("k") in (2, 4)]
    assert len(primary) == 6, f"expected 6 primary cells, found {len(primary)}"
    for v in primary:
        assert v["lambda_main"]["mean"] > v["depth_main"]["mean"], (
            f"{v['property']} k={v['k']}: depth now exceeds lambda; §24.2 is stale"
        )


def test_no_interaction_is_resolved_so_the_factors_are_additive(section, c32):
    """The additivity claim is what licenses reading the main effects separately."""
    for v in c32["decomposition"].values():
        if not v.get("complete") or v.get("k") not in (2, 4):
            continue
        assert not v["interaction"]["t_interval"]["excludes_zero"], (
            f"{v['property']} k={v['k']}: the interaction now excludes zero; §24.2's "
            f"'the two factors are additive' must be withdrawn"
        )
    assert "additive" in section


def test_the_closure_arithmetic_closes_exactly_per_seed(section, c32):
    """`d - a == depth + lambda` under the half-difference convention.

    Checked **per generation seed**, where it is an algebraic identity and holds to the last
    bit.  It does NOT close exactly on the aggregate figures, because the main effects
    average per-seed advantages while a corner's `advantage` is priced once at the aggregate
    budget -- a 6.0e-05 difference on this cell.  An earlier version of this test asserted
    aggregate closure to 1e-9 and failed for that reason; the section now states both, and
    which one is exact.
    """
    v = c32["decomposition"]["hbd_count_k2"]
    per_seed = {k: v["corners"][k]["advantage_by_seed"] for k in ("a", "b", "c", "d")}
    for seed in per_seed["a"]:
        a, b, c, d = (per_seed[x][seed] for x in ("a", "b", "c", "d"))
        depth = ((c - a) + (d - b)) / 2
        lam = ((b - a) + (d - c)) / 2
        assert abs((d - a) - (depth + lam)) < 1e-12, (
            f"seed {seed}: the half-difference convention no longer closes"
        )
        assert_in(section, f"{d - a:+.6f}", f"per-seed corner-to-corner total, seed {seed}")

    parts = v["depth_main"]["mean"] + v["lambda_main"]["mean"]
    total = v["corners"]["d"]["advantage"] - v["corners"]["a"]["advantage"]
    assert_in(section, f"{parts:+.4f}", "aggregate depth + lambda")
    assert_in(section, f"{total:+.4f}", "aggregate corner-to-corner total")
    assert abs(total - parts) < 1e-3, "the aggregate gap is larger than the section admits"


# ------------------------------------------------- 24.2.1  the retraction, correctly scoped

def test_the_deployed_probe_point_crossing_is_re_derived(section, c32):
    for k in (2, 4):
        c = c32["cells"][f"hbd_count_deployed_l2_k{k}"]
        t = c["advantage_vs_oracle_selected_seed_t_interval"]
        assert t["excludes_zero"] and c["advantage_vs_oracle_selected"] > 0, (
            "hbd_count no longer crosses at the deployed probe point with lambda=2; "
            "§24.2.1's retraction of C31 must be withdrawn"
        )
        assert_in(section, f"{c['advantage_vs_oracle_selected']:+.4f}", f"deployed l2 k{k}")
        assert_in(section, f"{t['lo']:+.4f}", f"deployed l2 k{k} interval low")
        assert_in(section, f"{c['validity_mean']:.4f}", f"deployed l2 k{k} validity")


def test_the_retraction_is_scoped_to_one_property_of_three(section, c32):
    """The claim is 'hbd_count crosses at the deployed probe point once lambda=2'.

    It is NOT 'the deployed probe point crosses'.  §24.2.1 prints the other two properties
    precisely so the scope cannot be lost, and this test fails if they stop being printed or
    if one of them silently starts crossing.
    """
    crossing = []
    for prop in PROPS:
        for k in (2, 4):
            c = c32["cells"][f"{prop}_deployed_l2_k{k}"]
            t = c["advantage_vs_oracle_selected_seed_t_interval"]
            assert_in(section, f"{c['advantage_vs_oracle_selected']:+.4f}",
                      f"{prop} deployed l2 k{k}")
            if c["advantage_vs_oracle_selected"] > 0 and t["excludes_zero"]:
                crossing.append(prop)
    assert set(crossing) == {"hbd_count"}, (
        f"the set of properties whose deployed arm crosses at lambda=2 is now {set(crossing)}; "
        f"§24.2.1 says hbd_count only"
    )
    assert "one property of three" in section


# ------------------------------------------------------- 24.3  the effective-lambda control

def test_the_spread_ratios_and_corrected_depth_are_re_derived(section, c32):
    rows = c32["effective_lambda"]["rows"]
    for name, r in rows.items():
        if not r["envelope"].get("ok"):
            continue
        assert_in(section, f"{r['spread_ratio']:.4f}", f"{name} spread ratio")
        assert_in(section, f"{r['depth_raw']:+.4f}", f"{name} raw depth")
        assert_in(section, f"{r['depth_corrected']:+.4f}", f"{name} corrected depth")


def test_c29s_premise_is_reported_as_non_universal(section, c32):
    """The qed spread ratio is below 1, which reverses the correction's direction."""
    rows = c32["effective_lambda"]["rows"]
    below_one = {r["spread_ratio"] for r in rows.values() if r["spread_ratio"] < 1.0}
    assert below_one, "no spread ratio is below 1; §24.3's 'not universal' claim is stale"
    for ratio in below_one:
        assert_in(section, f"{ratio:.4f}", "the sub-unity spread ratio")
    assert "not universal" in section


def test_depth_survives_correction_on_exactly_the_properties_the_section_names(section, c32):
    rows = c32["effective_lambda"]["rows"]
    survives = {r["property"] if "property" in r else name.split("_k")[0]
                for name, r in rows.items()
                if r["envelope"].get("ok")
                and (r.get("depth_corrected_t_interval") or {}).get("excludes_zero")
                and (r.get("depth_corrected") or 0) > 0}
    assert survives == {"aromatic_rings", "qed"}, (
        f"the set of properties on which depth survives the effective-lambda correction is "
        f"now {survives}; §24.3 names aromatic_rings (substantially) and qed (marginally)"
    )
    # and the section must not present qed's survival as comparable to aromatic_rings'
    qed_rows = [r for n, r in rows.items() if n.startswith("qed")
                and (r.get("depth_corrected_t_interval") or {}).get("excludes_zero")]
    assert qed_rows and all(abs(r["depth_corrected"]) < 0.0366 for r in qed_rows), (
        "qed's corrected depth effect is no longer below C29's largest probe-seed sd; "
        "§24.3's 'should not be leaned on' needs revisiting"
    )
    assert "barely on `qed`" in section or "marginally" in section


# ------------------------------------------------------------------ 24.4-24.8  the framing

def test_the_three_sentence_summary_carries_the_loss_and_the_scope(section):
    closing = section[section.index("24.4"):section.index("24.5")]
    assert "cannot follow best-of-N upward" in closing
    assert "two independent generators" in closing
    assert "steering strength, not" in closing, (
        "the summary must name the mechanism the 2x2 established"
    )


def test_the_ledger_marks_every_single_generator_claim_as_such(section):
    ledger = section[section.index("24.5"):section.index("24.6")]
    for claim in ("oracle asymmetry", "probe-seed variance", "k buys no raw accuracy"):
        assert claim in ledger, f"the ledger does not carry '{claim}'"
    assert ledger.count("GP-MoLFormer only") >= 2, (
        "claims resting on one generator must be labelled as such in the ledger"
    )


def test_the_open_questions_name_the_probe_seed_gap_as_the_top_priority(section):
    tail = section[section.index("24.7"):]
    assert "one probe seed" in tail
    assert "most" in tail and "valuable remaining experiment" in tail


def test_the_prereg_defect_is_disclosed_and_the_prereg_was_not_edited(section, c32):
    """C32 found its own closure identity to be false.  Reporting it is the contribution."""
    tail = section[section.index("24.8"):]
    assert "not edited" in tail or "was not edited" in tail
    defects = c32.get("preregistration_defects")
    assert defects, "C32 records no pre-registration defect but §24.8 describes one"
    blob = json.dumps(defects)
    assert "d - a" in blob or "d-a" in blob or "closure" in blob.lower()
