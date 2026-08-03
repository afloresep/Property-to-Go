"""C24 -- artefact-binding and implementation tests for the generality experiment.

Two kinds of test live here.

**Implementation tests** check that the text substrate really is running the same rule as
the molecular pipeline: that the device-aware head trainer is bit-identical to
`heads.train_head` on CPU, that the cached candidate backend agrees with full-prefix
recomputation (the assumption the token accounting rests on), and that the attribute
oracles are exact.

**Artefact-binding tests** re-read each C24 JSON, format it exactly as
`reports/section_c24_generality.md` formats it, and require it to appear in the report
text -- the style of `tests/test_report_matches_artifacts.py`.  Some are deliberately
written as *tripwires on the data* rather than on the prose, so a re-run that changes a
verdict fails a test instead of leaving the section standing unchallenged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
REPORT_PATH = ROOT / "reports" / "section_c24_generality.md"
PREREG_PATH = OUT / "c24_prereg" / "prereg.md"

C24_DIRS = (
    "c24_dataset", "c24_probe_layers", "c24_layer_steering", "c24_calibration",
    "c24_endtoend", "c24_summary",
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _have(*names) -> bool:
    return all((OUT / n).exists() for n in names)


requires_artifacts = pytest.mark.skipif(
    not _have(*C24_DIRS), reason="C24 artefacts not present"
)
requires_report = pytest.mark.skipif(
    not REPORT_PATH.exists(), reason="C24 report section not present"
)


@pytest.fixture(scope="module")
def summary():
    return _load(OUT / "c24_summary" / "c24_summary.json")


@pytest.fixture(scope="module")
def report():
    return REPORT_PATH.read_text()


def assert_in_report(text: str, value: str, label: str):
    assert value in text, f"{label}: {value!r} not found in the C24 report section"


# ---------------------------------------------------------------- pre-registration


@pytest.mark.skipif(not PREREG_PATH.exists(), reason="prereg not present")
def test_the_prereg_hash_matches_the_recorded_one():
    rec = _load(OUT / "c24_prereg" / "prereg.json")
    assert hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest() == rec["sha256"]


@requires_artifacts
def test_the_prereg_was_written_before_every_measurement():
    """Ordering by file mtime rather than by trust (the §20.1 pattern)."""
    t_prereg = PREREG_PATH.stat().st_mtime
    for name in C24_DIRS:
        for p in (OUT / name).rglob("*"):
            if p.is_file():
                assert t_prereg < p.stat().st_mtime, (
                    f"{p} is not newer than the pre-registration"
                )


@requires_report
def test_the_report_copies_the_prereg_verbatim():
    """The section's C24.0 must be the pre-registered text, byte for byte."""
    report = REPORT_PATH.read_text()
    prereg = PREREG_PATH.read_text()
    body = prereg.split("## C24.0.0", 1)[1]
    body = body.rsplit("*(Everything above was on disk", 1)[0].strip()
    assert body in report, "the report's C24.0 is not byte-identical to the prereg"


# ------------------------------------------------------- implementation equivalence


def test_the_device_trainer_reproduces_the_molecular_trainer_on_cpu():
    """`generality.train_head_on_device(..., 'cpu')` == `heads.train_head`, bitwise.

    This is what licenses calling the C24 head "the same recipe" as the molecular one.
    """
    import torch

    from property_to_go.generality import train_head_on_device
    from property_to_go.heads import MLPHead, train_head

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 16)).astype(np.float32)
    y = rng.integers(0, 4, 600)
    cfg = {"seed": 7, "lr": 1e-3, "weight_decay": 0.01, "batch_size": 64,
           "max_epochs": 4, "patience": 3}

    a = MLPHead(16, 8, 4)
    b = MLPHead(16, 8, 4)
    b.load_state_dict(a.state_dict())
    ra = train_head(a, x[:500], y[:500], x[500:], y[500:], cfg)
    rb = train_head_on_device(b, x[:500], y[:500], x[500:], y[500:], cfg, "cpu")

    assert ra.best_epoch == rb.best_epoch
    assert ra.best_val_nll == rb.best_val_nll
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), k


def test_the_decoding_rule_is_the_molecular_function():
    """C24 imports `guidance.combine_scores`; it does not re-derive the rule."""
    from property_to_go import generality, guidance

    assert generality.combine_scores is guidance.combine_scores
    src = (ROOT / "src" / "property_to_go" / "generality.py").read_text()
    assert "combine_scores(cand_lp, q, lam, eps)" in src


def test_c24_never_loads_the_molecular_generator():
    """The external-validity check must not be able to move a molecular number."""
    for p in sorted((ROOT / "scripts").glob("19_c24_*.py")) + [
        ROOT / "src" / "property_to_go" / "generality.py"
    ]:
        text = p.read_text()
        assert "load_generator" not in text, p
        assert "GP-MoLFormer-Uniq" not in text, p


#: Modules permitted to import `generality`, with the reason.  C31 runs the molecular
#: pipeline on a **second molecular generator**, `entropy/gpt2_zinc_87m`, which is a GPT-2
#: -- so `generality.repeat_cache_gpt2` (the standard-attention KV-cache repeat) and
#: `generality.train_head_on_device` (the device-aware trainer this file already asserts is
#: bit-identical to `heads.train_head` on CPU) are exactly the right things to reuse, and
#: re-deriving them would be the copy-paste fork C30 established the project must not do.
#: The invariant this test exists to protect -- *nothing that can move a GP-MoLFormer
#: number depends on C24* -- is preserved and is asserted directly by
#: `test_c31_never_loads_the_molecular_generator` below.
GENERALITY_IMPORT_ALLOWLIST = {
    "generality.py",
    "second_generator.py",              # C31's adapter for entropy/gpt2_zinc_87m
    "25_c31_second_generator.py",       # C31's experiment script
    "25_summarise_c31.py",              # C31's summariser (imports the script above)
}


def _imported_module_names(path: Path) -> set[str]:
    """Top-level module names this file actually imports, via the AST.

    A string search would also match a *docstring* that names the module, which is how
    `guidance.py` and `probe_layers.py` explain why their one C31 hook exists.  The
    invariant is about dependency, not about the word appearing in prose, so the check is
    made on the import graph.
    """
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
            names.update(a.name.rsplit(".", 1)[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.update(mod.split("."))
            names.update(a.name for a in node.names)
    return names


def test_no_molecular_module_imports_the_generality_module():
    """Separation in the other direction: nothing molecular depends on C24."""
    for p in list((ROOT / "src" / "property_to_go").glob("*.py")) + \
            [q for q in (ROOT / "scripts").glob("*.py") if not q.name.startswith("19_c24")]:
        if p.name in GENERALITY_IMPORT_ALLOWLIST:
            continue
        assert "generality" not in _imported_module_names(p), p


def test_c31_never_loads_the_molecular_generator():
    """The allowlist above may not become a back door.

    C31 is allowed to import `generality`, so the guarantee that C24 cannot move a
    GP-MoLFormer number has to be re-established from the other side: no C31 file may
    load GP-MoLFormer at all.  Only `property_to_go.second_generator.load_zinc_generator`
    is ever called; `model_io.load_generator` never is.

    Checked on the import graph, not on raw text: C31's summariser legitimately *names*
    GP-MoLFormer in a JSON field that records which generator the C23-C30 numbers came
    from, and a string search would forbid saying what is being compared against.
    """
    import ast

    files = [ROOT / "src" / "property_to_go" / "second_generator.py"] + \
        sorted((ROOT / "scripts").glob("25_c31_*.py")) + \
        sorted((ROOT / "scripts").glob("25_summarise_c31.py"))
    assert files
    for p in files:
        imported = _imported_module_names(p)
        assert "model_io" not in imported, p
        assert "load_generator" not in imported, p
        called = {
            node.func.id for node in ast.walk(ast.parse(p.read_text()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr for node in ast.walk(ast.parse(p.read_text()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "load_generator" not in called, p


# ------------------------------------------------------------------- the oracles


def test_the_attribute_oracles_are_exact():
    from property_to_go import generality as G

    assert G.digit_count("a1b22c333") == 6.0
    assert G.digit_count("no digits here") == 0.0
    # "AbC DeF" carries four upper-case characters: A, C, D, F.  The first version of
    # this assertion said 3.0 and failed against a correct oracle; the prereg (C24.0,
    # attribute table) defines `upper_count` as "number of upper-case characters", so
    # the assertion was wrong, not the oracle.  Both a 3-upper and a 4-upper string are
    # checked now so a miscount in either direction is caught.
    assert G.upper_count("AbC DeF") == 4.0
    assert G.upper_count("AbC def") == 2.0
    assert G.upper_count("no upper here") == 0.0
    assert G.mean_word_length("aa bbbb") == 3.0
    assert G.mean_word_length("   ") == 0.0
    assert set(G.ATTRIBUTES) == set(G.ATTRIBUTE_ORDER)


def test_every_count_attribute_is_declared_integer_valued():
    """The docs/HANDOFF.md §4 boundary bug, guarded in the text domain.

    A `digit_count` of exactly `hi` is one unit outside the target `[lo, hi)`, not zero
    units outside, and a genuine hit must outrank it.
    """
    from property_to_go.bestofn import selection_key, target_error
    from property_to_go.generality import ATTRIBUTES, INTEGER_ATTRIBUTES

    derived = {name for name in ATTRIBUTES
               if name.endswith("_count") or name.startswith("n_")}
    assert derived <= INTEGER_ATTRIBUTES

    lo, hi = 7.0, 13.0
    assert target_error(13.0, lo, hi, integer_valued=True) == 1.0
    assert target_error(12.0, lo, hi, integer_valued=True) == 0.0
    assert selection_key(12.0, lo, hi) < selection_key(13.0, lo, hi)


def test_the_target_band_rule_is_deterministic_and_respects_the_gate():
    from property_to_go.generality import resolve_target_band

    rng = np.random.default_rng(3)
    v = rng.poisson(4.0, 20000).astype(float)
    a = resolve_target_band(v, integer_valued=True)
    b = resolve_target_band(v, integer_valued=True)
    assert a == b
    assert a["lo"] == round(a["lo"]) and a["hi"] == round(a["hi"])
    assert abs(a["base_rate"] - 0.10) < 0.10


def test_the_gpt2_cache_repeat_agrees_with_full_recomputation():
    """The cached candidate backend must equal full-prefix recomputation.

    This is the text-domain analogue of `test_candidate_backends_agree`, and it is the
    load-bearing assumption behind C24's two-way token accounting: `actual` counts one
    token per candidate only because the cached path is the same computation.

    Unlike the molecular backends, the two paths are **not** bit-identical: standard
    attention reduces over the key dimension in a different order when the prefix comes
    from a cache, so float32 agreement is to ~1e-3 on states of scale ~100.  The measured
    residual is in `outputs/c24_gate/gate.json` and is quoted in the report rather than
    asserted away.
    """
    import torch

    from property_to_go import generality as G

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        gen = G.load_text_generator("cpu")
    except Exception as exc:  # pragma: no cover - depends on local HF cache
        pytest.skip(f"gpt2 unavailable: {exc}")

    prefix = torch.tensor([[G.TEXT_BOS_ID, 464, 2068, 7586, 21831]], dtype=torch.long)
    cands = torch.tensor([[262, 290, 284, 257]], dtype=torch.long)
    k = cands.shape[1]

    with torch.no_grad():
        res = gen.model.transformer(input_ids=prefix, use_cache=True, return_dict=True)
        cached = gen.model.transformer(
            input_ids=cands.reshape(k, 1),
            past_key_values=G.repeat_cache_gpt2(res.past_key_values, k),
            use_cache=True, output_hidden_states=True, return_dict=True,
        )
        ext = torch.cat([prefix.expand(k, -1), cands.reshape(k, 1)], dim=1)
        full = gen.model.transformer(
            input_ids=ext, use_cache=False, output_hidden_states=True, return_dict=True
        )
    for L in range(gen.n_probe_points):
        a = cached.hidden_states[L][:, 0, :]
        b = full.hidden_states[L][:, -1, :]
        assert torch.allclose(a, b, atol=2e-3), f"probe point {L} disagrees"


# ---------------------------------------------------------------- artefact binding


@pytest.mark.skipif(not _have("c24_gate"), reason="C24 gate not present")
def test_the_validity_gate_is_recorded_and_passes():
    g = _load(OUT / "c24_gate" / "gate.json")
    assert g["within_tolerance"], g["max_abs_difference"]
    assert len(g["max_abs_difference_by_probe_point"]) == 13
    # the residual must be reported, not hidden behind the word "matches"
    assert g["relative_to_state_scale"] < 1e-3


@requires_artifacts
def test_the_base_rate_gate_was_met(summary):
    for a, r in summary["attributes"].items():
        rate = _load(OUT / "c24_dataset" / "target_intervals.json")["intervals"][a][
            "base_rate"
        ]
        assert 0.05 <= rate <= 0.20, (a, rate)
    assert summary["n_attributes"] >= 2


@requires_artifacts
def test_the_target_interval_is_a_union_of_bins():
    cov = _load(OUT / "c24_dataset" / "target_intervals.json")["interval_mask_coverage"]
    for a, c in cov.items():
        assert c["is_exact"], (a, c)


@requires_artifacts
def test_the_identity_is_exact_at_eps_zero(summary):
    """Tripwire on the data, not the prose.

    Claim 1's algebraic core: a power calibration at λ=1 and the raw head at λ=α are the
    same sampler.  If this ever stops being 1.000 the section must be rewritten.
    """
    for a, r in summary["attributes"].items():
        assert r["identity"]["eps0"]["identical_fraction"] == 1.0, a
        assert r["identity"]["eps0"]["hit_rate_difference"] == 0.0, a


@requires_artifacts
def test_platt_leaves_auroc_bit_identical(summary):
    for a, r in summary["attributes"].items():
        assert abs(r["auroc_delta_platt"]) <= 1e-12, (a, r["auroc_delta_platt"])


@requires_artifacts
def test_the_best_of_n_match_is_real_and_the_realised_ratio_is_recorded(summary):
    seen = 0
    for a, arms in summary["best_of_n"].items():
        for name, b in arms.items():
            assert b["n_candidates"] >= 1
            assert 0.8 <= b["realised_token_ratio"] <= 1.2, (a, name, b)
            seen += 1
    assert seen >= 6


@requires_artifacts
@requires_report
def test_the_depth_table_matches_the_report(report):
    d = _load(OUT / "c24_probe_layers" / "probe_layer_metrics.json")
    for a, r in d["attributes"].items():
        for L in range(d["n_probe_points"]):
            v = r["probe_points"][str(L)]["target_auroc"]["mean"]
            assert_in_report(report, f"{v:.4f}", f"{a} probe point {L} AUROC")
        assert_in_report(report, f"{r['trivial']['target_auroc']['mean']:.4f}",
                         f"{a} trivial AUROC")


@requires_artifacts
@requires_report
def test_the_calibration_table_matches_the_report(report):
    c = _load(OUT / "c24_calibration" / "calibration_metrics.json")["attributes"]
    for a, r in c.items():
        assert_in_report(report, f"{r['platt_slope']:.4f}", f"{a} platt slope")
        assert_in_report(report, f"{r['calibrated']['uncalibrated']['ece']:.4f}",
                         f"{a} ECE uncalibrated")
        assert_in_report(report, f"{r['calibrated']['platt']['ece']:.4f}",
                         f"{a} ECE platt")
        assert_in_report(report, f"{r['calibrated']['isotonic']['ece']:.4f}",
                         f"{a} ECE isotonic")


@requires_artifacts
@requires_report
def test_the_per_position_and_end_to_end_columns_both_appear(report, summary):
    """Both signs must be printed; neither may stand in for the other."""
    for a, r in summary["attributes"].items():
        assert_in_report(report, f"{r['per_position_gain_final']:+.5f}",
                         f"{a} per-position gain, final probe point")
        assert_in_report(report, f"{r['per_position_gain_best']:+.5f}",
                         f"{a} per-position gain, best probe point")
        assert_in_report(report, f"{r['endtoend_hit_final']:.4f}",
                         f"{a} end-to-end hit rate, final probe point")
        if r["endtoend_hit_best"] is not None:
            assert_in_report(report, f"{r['endtoend_hit_best']:.4f}",
                             f"{a} end-to-end hit rate, best probe point")


@requires_artifacts
@requires_report
def test_the_per_seed_hit_rates_are_published(report, summary):
    """The reviewer's noise floor: per-seed values, not just means."""
    for a, arms in summary["attributes"].items():
        for name in ("unguided",):
            for v in arms["per_seed_hit_rates"][name]:
                assert_in_report(report, f"{v:.4f}", f"{a}/{name} per-seed hit rate")


@requires_artifacts
@requires_report
def test_the_verdicts_in_the_report_are_the_ones_the_artefacts_compute(report, summary):
    assert_in_report(report, summary["claim_1_calibration"]["VERDICT"], "Claim 1 verdict")
    assert_in_report(report, summary["claim_2_depth"]["2e_VERDICT"].split(" - ")[0],
                     "Claim 2 verdict")
    assert_in_report(report, summary["claim_2_depth"]["2c_verdict"], "2c verdict")
    assert_in_report(report, summary["claim_2_depth"]["2d_verdict"], "2d verdict")


@requires_artifacts
@requires_report
def test_the_report_records_where_the_prereg_failed(report, summary):
    """A pre-registration that is only ever scored as passing is not being scored."""
    c1 = summary["claim_1_calibration"]
    if not c1["1a_all_platt_slopes_below_one"]:
        assert "1a" in report and "FAILS" in report
        offenders = [a for a, s in c1["1a_slopes"].items() if s >= 1.0]
        for a in offenders:
            assert_in_report(report, f"{c1['1a_slopes'][a]:.4f}",
                             f"{a} slope above 1 must be quoted")
    if not c1["1c_ece_halved_by_both"]:
        assert "1c" in report


@requires_artifacts
def test_guidance_moved_the_attribute_at_all(summary):
    """C24.0.10: with no steering effect, Claims 1e and 2d would be vacuous.

    The key this asserts used to be called `guided_beats_unguided_all` while being
    computed against the **truncation control**.  The two references disagree in *sign*
    for `mean_word_length` (+0.0202 against the truncation control, -0.0658 against
    `unguided`), so the misnomer was load-bearing: it read as a claim about `unguided`
    that the artefact did not support.  `scripts/19_c24_summarise.py` now publishes both
    under honest names and this test asserts the same condition it always asserted --
    positive lift at **every** attribute against the truncation control -- which is
    strictly stronger than C24.0.10's own criterion ("not positive at *any* attribute").
    Both references are asserted below so neither can be quoted as the other.
    """
    s = summary["sanity_C24_0_10"]
    assert s["guided_beats_truncation_control_all"]
    assert s["guided_beats_unguided_any"], "C24.0.10's own uninterpretability criterion"
    # and the disagreement itself is a recorded fact, not an accident of naming
    assert not s["guided_beats_unguided_all"]
    assert all(v < 0 for v in s["truncation_effect"].values()), s["truncation_effect"]


@requires_artifacts
def test_the_truncation_control_is_present_and_is_not_a_formality(summary):
    """§7.9's control.

    On this substrate top-8 truncation is itself a large intervention, so the report has
    to quote it: measuring lift against `unguided` alone would charge the truncation
    penalty to guidance.  This test failed to exist in the first C24 draft and that
    omission produced a wrong reading of the digit-count arm; it is recorded in the
    section's "what I got wrong" note.
    """
    for a, r in summary["attributes"].items():
        assert "truncation_control" in r["per_seed_hit_rates"], a
        assert r["truncation_control_hit"] is not None


@requires_artifacts
def test_processed_tokens_are_reported_and_no_wall_clock_claim_is_made(summary, ):
    assert summary["processed_tokens"]["total"] > 0
    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text().lower()
        for banned in ("seconds faster", "x faster in wall", "wall-clock speedup"):
            assert banned not in text


# ------------------------------------------ artefact binding, the numbers C24 prints
#
# Everything below binds a number that `reports/section_c24_generality.md` prints, in the
# exact format it prints it.  The rule for this section is that a number may not appear in
# the prose unless a test here re-derives it from JSON and requires it in the text, so a
# re-run that moves a number fails a test rather than leaving a stale figure standing.


@requires_artifacts
@requires_report
def test_the_truncation_control_numbers_are_published(report, summary):
    """The arm the first C24 draft was missing, and the contrast it decides.

    `mean_word_length` is the case that matters: -0.0658 against `unguided` and +0.0202
    against the truncation control.  Both must be in the text.
    """
    for a, r in summary["attributes"].items():
        assert_in_report(report, f"{r['truncation_control_hit']:.4f}", f"{a} trunc hit")
        assert_in_report(report, f"{r['unguided_hit']:.4f}", f"{a} unguided hit")
        assert_in_report(report, f"{r['truncation_effect']:+.4f}", f"{a} trunc effect")
        assert_in_report(report, f"{r['lift_uncalibrated']:+.4f}", f"{a} lift vs trunc")
        assert_in_report(report, f"{r['lift_uncalibrated_vs_unguided']:+.4f}",
                         f"{a} lift vs unguided")
        for v in r["per_seed_hit_rates"]["truncation_control"]:
            assert_in_report(report, f"{v:.4f}", f"{a} trunc per-seed hit rate")
        share = 100.0 * (r["unguided_hit"] - r["truncation_control_hit"]) / r["unguided_hit"]
        assert 0.0 < share < 100.0, (a, share)
    shares = [100.0 * (r["unguided_hit"] - r["truncation_control_hit"]) / r["unguided_hit"]
              for r in summary["attributes"].values()]
    assert_in_report(report, f"{min(shares):.1f}%", "smallest truncation penalty share")
    assert_in_report(report, f"{max(shares):.1f}%", "largest truncation penalty share")


@requires_artifacts
@requires_report
def test_every_end_to_end_arm_row_is_in_the_report(report, summary):
    """Every cell of the three per-attribute arm tables."""
    e2e = _load(OUT / "c24_endtoend" / "endtoend_metrics.json")["attributes"]
    seen = 0
    for a, r in summary["attributes"].items():
        trunc, ung = r["truncation_control_hit"], r["unguided_hit"]
        for name, arm in e2e[a]["arms"].items():
            m = arm["hit_rate_mean"]
            assert_in_report(report, f"{m:.4f}", f"{a}/{name} hit rate")
            assert_in_report(report, f"{arm['hit_rate_sd']:.4f}", f"{a}/{name} sd")
            for v in r["per_seed_hit_rates"][name]:
                assert_in_report(report, f"{v:.4f}", f"{a}/{name} per-seed hit rate")
            assert_in_report(report, f"{m - trunc:+.4f}", f"{a}/{name} lift vs trunc")
            assert_in_report(report, f"{m - ung:+.4f}", f"{a}/{name} lift vs unguided")
            assert_in_report(report, f"{arm['compute']['tokens_per_molecule_actual']:.1f}",
                             f"{a}/{name} tokens per sequence")
            b = arm.get("best_of_n")
            if b is not None:
                assert_in_report(report, f"{b['hit_rate']:.4f}", f"{a}/{name} best-of-N hit")
                assert_in_report(report, f"{b['advantage_guided_minus_bestofn']:+.4f}",
                                 f"{a}/{name} best-of-N advantage")
                assert_in_report(report, f"{b['realised_token_ratio']:.4f}",
                                 f"{a}/{name} realised token ratio")
            seen += 1
    assert seen == 30, seen


@requires_artifacts
@requires_report
def test_the_depth_summary_row_is_in_the_report(report, summary):
    d = _load(OUT / "c24_probe_layers" / "probe_layer_metrics.json")["attributes"]
    for a, r in d.items():
        dep = r["depth"]
        ci = dep["bonferroni_ci_best_minus_final"]
        assert_in_report(report, f"{dep['auroc_best']:.4f}", f"{a} AUROC at L*")
        assert_in_report(report, f"{dep['auroc_final']:.4f}", f"{a} AUROC at 12")
        assert_in_report(report, f"{dep['gain_over_final']:+.4f}", f"{a} depth gain")
        assert_in_report(report, f"{ci['mean']:+.4f}", f"{a} Bonferroni CI mean")
        assert_in_report(report, f"{ci['lo']:+.4f}", f"{a} Bonferroni CI lo")
        assert_in_report(report, f"{ci['hi']:+.4f}", f"{a} Bonferroni CI hi")
        assert_in_report(report, f"{dep['margin_over_trivial']:+.4f}",
                         f"{a} margin over trivial")
        for k, v in dep["neighbours"].items():
            assert_in_report(report, f"{v:+.4f}", f"{a} neighbour {k}")
        # the `trivial` head beats the probe at every probe point, for every attribute --
        # the caveat §C24.4 has to carry, guarded here so it cannot be dropped silently
        assert dep["margin_over_trivial"] < 0, a
        assert "trivial" in report and "margin over" in report


@requires_artifacts
@requires_report
def test_the_calibration_extras_are_in_the_report(report, summary):
    """The columns beyond ECE and slope: factors, AUROC deltas, the policy gaps."""
    for a, r in summary["attributes"].items():
        assert_in_report(report, f"{r['ece_factor_platt']:.2f}x", f"{a} ECE factor Platt")
        assert_in_report(report, f"{r['ece_factor_isotonic']:.2f}x", f"{a} ECE factor iso")
        assert_in_report(report, f"{r['auroc_delta_platt']:.6f}", f"{a} dAUROC Platt")
        assert_in_report(report, f"{r['auroc_delta_isotonic']:+.6f}", f"{a} dAUROC iso")
        assert_in_report(report, f"{r['off_policy_factor']:.4f}", f"{a} off-policy factor")
        assert_in_report(report, f"{r['on_policy_factor']:.4f}", f"{a} on-policy factor")
    deltas = [abs(r["auroc_delta_isotonic"]) for r in summary["attributes"].values()]
    assert_in_report(report, f"{max(deltas):.6f}", "largest isotonic AUROC move")


@requires_artifacts
@requires_report
def test_the_identity_table_is_in_the_report(report, summary):
    """Claim 1's algebraic core, cell by cell, at both epsilons."""
    cal = _load(OUT / "c24_calibration" / "calibration_metrics.json")["attributes"]
    for a, r in summary["attributes"].items():
        for tag in ("eps0", "epsdep"):
            i = r["identity"][tag]
            assert_in_report(report, f"{i['identical']} / {i['n']}",
                             f"{a} identical count at {tag}")
            assert_in_report(report, f"{i['identical_fraction']:.4f}",
                             f"{a} identical fraction at {tag}")
            assert_in_report(report, f"{i['hit_rate_difference']:+.6f}",
                             f"{a} identity hit-rate difference at {tag}")
        num = cal[a]["identity_numeric"]["eps_0"]
        assert num["argmax_agreement"] == 1.0, a
        assert_in_report(report, f"{num['max_abs_weight_difference']:.6e}",
                         f"{a} identity weight residual")
    # the deployed-epsilon arm is allowed to break the identity; the report must say by
    # how much rather than rounding it away
    frac = min(r["identity"]["epsdep"]["identical_fraction"]
               for r in summary["attributes"].values())
    assert frac < 1.0, "no attribute broke the identity at eps>0; §C24.6 must be rewritten"


@requires_artifacts
@requires_report
def test_the_claim_1_lift_ratios_are_in_the_report(report, summary):
    for a, r in summary["attributes"].items():
        assert_in_report(report, f"{r['ratio_platt']:.4f}", f"{a} Platt lift ratio")
        assert_in_report(report, f"{r['ratio_isotonic']:.4f}", f"{a} isotonic lift ratio")
    # 1e is scored as "below 1 at >= 2 of 3 for both fitters"; the counts are printed
    c1 = summary["claim_1_calibration"]
    assert c1["1e_platt_below_one_count"] == 2 and c1["1e_isotonic_below_one_count"] == 2


@requires_artifacts
@requires_report
def test_the_depth_bootstrap_and_per_position_relatives_are_in_the_report(report, summary):
    for a, r in summary["attributes"].items():
        assert_in_report(report, f"{r['per_position_relative']:+.4f}",
                         f"{a} per-position relative")
        if r["endtoend_depth_ratio"] is not None:
            assert_in_report(report, f"{r['endtoend_depth_ratio']:.4f}",
                             f"{a} end-to-end depth lift ratio")
        b = r["endtoend_depth_seed_interval"]
        if b is not None:
            assert_in_report(report, f"{b['mean_difference']:+.4f}", f"{a} depth mean")
            assert_in_report(report, f"{b['sd']:.4f}", f"{a} depth sd")
            assert_in_report(report, f"{b['lo']:+.4f}", f"{a} depth t lo")
            assert_in_report(report, f"{b['hi']:+.4f}", f"{a} depth t hi")
            for v in b["per_seed_difference"]:
                assert_in_report(report, f"{v:+}", f"{a} per-seed depth difference")
            # all three seeds agree in sign -- the sentence §C24.8 leans on
            assert len({v > 0 for v in b["per_seed_difference"]}) == 1, a


@requires_artifacts
def test_the_three_seed_bootstrap_was_replaced_by_an_honest_interval(summary):
    """At n=3 a percentile bootstrap of a mean is exactly [min, max] and conveys only a
    sign test at p_null = 0.25.  C24 reported one as a CI and two of three rows read as
    significant because of it.  This test fails if a bootstrap is ever reinstated, and
    checks the replacement really is the wider t interval rather than a relabelling."""
    import math
    for a, r in summary["attributes"].items():
        b = r.get("endtoend_depth_seed_interval")
        assert "endtoend_depth_bootstrap" not in r, a
        if b is None:
            continue
        assert b["interval"].startswith("Student t"), a
        assert b["sign_test_p_two_sided"] == 0.25, a
        ps = b["per_seed_difference"]
        # strictly wider than the degenerate bootstrap it replaced
        assert b["lo"] < min(ps) and b["hi"] > max(ps), a
        half = 4.302653 * b["sd"] / math.sqrt(len(ps))
        assert math.isclose(b["hi"] - b["mean_difference"], half, rel_tol=1e-9), a


@requires_artifacts
@requires_report
def test_the_2c_margin_is_quoted_because_the_verdict_turns_on_it(report, summary):
    """2c is decided by a hair and §C24.12 must say by how much."""
    c2 = summary["claim_2_depth"]
    med = c2["2c_median_relative"]
    assert_in_report(report, f"{med:+.4f}", "2c median relative improvement")
    assert_in_report(report, f"{0.25 - med:.4f}", "2c margin below the +0.25 threshold")
    assert c2["2c_verdict"] == "NOT MATERIAL" and 0.0 < 0.25 - med < 0.05, (
        "2c is no longer marginal; §C24.12 item 1 must be rewritten"
    )
    assert_in_report(report, str(c2["2c_per_position_improves_count"]), "2c improve count")
    assert_in_report(report, str(c2["2d_endtoend_improves_count"]), "2d improve count")


@pytest.mark.skipif(not _have("c24_gate"), reason="C24 gate not present")
@requires_report
def test_the_gate_residuals_are_quoted(report):
    g = _load(OUT / "c24_gate" / "gate.json")
    assert_in_report(report, f"{g['max_abs_difference']:.3e}", "gate max abs difference")
    assert_in_report(report, f"{g['relative_to_state_scale']:.3e}", "gate relative residual")
    assert_in_report(report, f"{g['hidden_state_max_abs_value']:.2f}", "hidden state scale")
    assert not g["bit_identical"], "the two paths are bit-identical; §C24.2 must be rewritten"
    by_pp = g["max_abs_difference_by_probe_point"]
    assert by_pp["0"] == 0.0
    assert by_pp["12"] == g["max_abs_difference"]
    fp = g["generator_fingerprint"]
    assert_in_report(report, f"{fp['n_parameters']:,}", "generator parameter count")
    assert_in_report(report, fp["revision"], "generator revision")
    assert_in_report(report, str(fp["hidden_size"]), "generator hidden size")


@requires_artifacts
@requires_report
def test_the_dataset_shape_and_targets_are_quoted(report):
    d = _load(OUT / "c24_dataset" / "dataset_metrics.json")
    ti = _load(OUT / "c24_dataset" / "target_intervals.json")
    assert_in_report(report, f"{d['n_sequences']:,}", "n sequences")
    assert_in_report(report, f"{d['unique_texts']:,}", "unique texts")
    for k, v in d["split_counts"].items():
        assert_in_report(report, f"{v:,}", f"{k} split count")
    for k, v in d["row_split_counts"].items():
        assert_in_report(report, f"{v:,}", f"{k} row split count")
    assert d["generator_fingerprint"] == _load(OUT / "c24_gate" / "gate.json")[
        "generator_fingerprint"
    ], "the dataset and the gate did not use the same frozen generator"
    for a, band in ti["intervals"].items():
        assert_in_report(report, f"{band['base_rate']:.4f}", f"{a} base rate")
        for edge in ("lo", "hi"):
            v = band[edge]
            s = f"{v:g}" if float(v).is_integer() else f"{v:.4f}"
            assert_in_report(report, s, f"{a} interval {edge}")


@requires_artifacts
@requires_report
def test_the_processed_token_totals_are_quoted(report, summary):
    pt = summary["processed_tokens"]
    for k, v in pt.items():
        assert_in_report(report, f"{v:,}", f"processed tokens, {k}")
    assert pt["total"] == sum(v for k, v in pt.items() if k != "total")


@requires_artifacts
@requires_report
def test_the_best_of_n_summary_is_in_the_report(report, summary):
    """The compute-matched comparison, and the claim that nothing ever beats it."""
    assert not summary["any_arm_anywhere_beats_compute_matched_best_of_n"]
    guided = {}
    for a, arms in summary["best_of_n"].items():
        g = {n: b for n, b in arms.items() if n != "truncation_control"}
        assert {b["n_candidates"] for b in g.values()} == {9}, (a, g)
        assert arms["truncation_control"]["n_candidates"] == 1, a
        guided[a] = g
    advantages = [b["advantage"] for g in guided.values() for b in g.values()]
    assert_in_report(report, f"{max(advantages):+.4f}", "smallest best-of-N deficit")
    assert_in_report(report, f"{min(advantages):+.4f}", "largest best-of-N deficit")
    # the "best guided arm" column of §C24.9
    e2e = _load(OUT / "c24_endtoend" / "endtoend_metrics.json")["attributes"]
    for a in summary["attributes"]:
        cand = {n: r["hit_rate_mean"] for n, r in e2e[a]["arms"].items()
                if n not in ("unguided", "truncation_control")}
        best = max(cand, key=cand.get)
        assert_in_report(report, f"{cand[best]:.4f}", f"{a} best guided arm hit rate")
        assert_in_report(
            report,
            f"{e2e[a]['arms'][best]['best_of_n']['advantage_guided_minus_bestofn']:+.4f}",
            f"{a} best guided arm best-of-N advantage",
        )


@requires_report
def test_the_molecular_numbers_quoted_for_comparison_really_are_molecular_numbers(report):
    """C24 may quote the molecular result; it may not invent it.

    Each figure below is asserted to appear both in this section and in the molecular
    document it is attributed to, so a mis-transcribed comparison fails here rather than
    propagating into a generality claim.
    """
    sources = {
        ROOT / "reports" / "pilot_report.md": ["0.405", "0.618"],
        # 0.2988 was added 2026-08-03. The section originally compared probe point 4's
        # 0.3689 against "0.3395", which is probe point *6*; probe point 12 -- the arm the
        # comparison is about -- is 0.2988. Binding it here means a future edit cannot
        # silently drop the corrected figure and leave the wrong one standing.
        ROOT / "reports" / "section_c23_layer_end_to_end.md":
            ["0.8474", "0.7878", "0.8269", "0.3689", "0.3395", "0.2988"],
    }
    for path, values in sources.items():
        if not path.exists():  # pragma: no cover
            pytest.skip(f"{path.name} not present")
        text = path.read_text()
        for v in values:
            assert v in text, f"{v} is attributed to {path.name} but is not in it"
            assert_in_report(report, v, f"molecular comparison figure {v}")


@requires_artifacts
@requires_report
def test_the_report_states_the_external_validity_boundary_prominently(report):
    """The C24-specific rule: the boundary must be visible, near the top, not buried."""
    head = report[:4000]
    assert "external-validity check" in head
    assert "non-molecular generator" in head
    assert "no second generator" in head
    assert "never part of the main molecular result" in head.lower() or \
           "never as part of the main molecular result" in head.lower()


@requires_artifacts
@requires_report
def test_the_report_does_not_edit_the_molecular_report(report):
    """C24 flags conflicts; it does not resolve them in `pilot_report.md`."""
    assert "Nothing in `reports/pilot_report.md` was edited by C24" in report
