"""C25 -- the pooled readout, bound to its contracts and its artefacts.

Four kinds of test live here, kept apart on purpose.

**The generalisation contract.**  A pooled readout at window size 1 must be *exactly* the
deployed single-position readout -- as a feature array, as a head at initialisation, as a
head after training, and as a decoder.  If any of those is only approximately true then
every C25 comparison is measuring the refactor rather than the pooling, which is what
§C25.0.1 says and what these tests enforce.

**The pre-registered family.**  §C25.0.2 fixes five pooling variants plus one capacity
control and forbids adding more after results exist.  The family lives in
`property_to_go.pooling.POOL_VARIANTS` and is asserted here name by name.

**`guidance.py` is untouched.**  C25 was told not to modify it.  The pooled sampler is a
transcription in a new module; the decoder identity test is what stops the copy drifting.

**Artefact binding.**  Every number asserted in `reports/section_c25_pooling.md` is re-read
from the JSON it came from, formatted the way the section formats it, and required to
appear in the text -- the pattern `tests/test_report_matches_artifacts.py`,
`tests/test_probe_layers.py` and `tests/test_layer_end_to_end.py` use.  These skip when the
artefacts are absent, so a fresh clone passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from property_to_go import pooling as PL
from property_to_go.guidance import TargetScorer, guided_sample
from property_to_go.heads import MLPHead, train_head

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SECTION = ROOT / "reports" / "section_c25_pooling.md"
PREREG = OUT / "c25_prereg" / "C25.0_preregistration.md"
STATES = OUT / "c25_window_states_pilot_50k_p2"
SWEEP = OUT / "c25_pooled_heads" / "pooled_metrics.json"
SUMMARY = OUT / "c25_summary" / "c25_metrics.json"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def section_text() -> str:
    if not SECTION.exists():
        pytest.skip("section not written yet")
    # The prose uses a typographic minus (U+2212) where f"{x:.4f}" emits ASCII "-".
    return SECTION.read_text().replace("−", "-")


# --------------------------------------------------------------------------
# 1. the pre-registered family
# --------------------------------------------------------------------------

def test_pool_variant_family_is_the_preregistered_one():
    """§C25.0.2 fixes the family. Adding a variant later is what the prereg forbids."""
    got = {v.name: (v.window, v.mode, v.hidden_dim) for v in PL.POOL_VARIANTS}
    assert got == {
        "last1": (1, "last", None),
        "mean4": (4, "mean", None),
        "mean16": (16, "mean", None),
        "concat4": (4, "concat", None),
        "attn4": (4, "attn", None),
        "wide1": (1, "last", 1024),
    }
    assert [v.name for v in PL.POOL_VARIANTS][0] == "last1"


def test_frozen_prereg_is_a_verbatim_substring_of_the_section():
    if not PREREG.exists() or not SECTION.exists():
        pytest.skip("pre-registration not frozen yet")
    frozen = PREREG.read_text().rstrip("\n")
    assert frozen and frozen in SECTION.read_text(), (
        "the frozen §C25.0 copy is no longer a verbatim substring of the section; "
        "the pre-registration has been edited after freezing"
    )


def test_prereg_lock_records_the_freeze():
    lock = _load(OUT / "c25_prereg" / "prereg_lock.json")
    assert lock["prereg_bytes"] > 1000
    assert len(lock["source_sha256"]) == 64 and len(lock["prereg_sha256"]) == 64


# --------------------------------------------------------------------------
# 2. the window and the fixed operators
# --------------------------------------------------------------------------

def test_window_positions_are_index_clamped_and_counts_are_the_distinct_ones():
    assert PL.window_positions(7, 4) == [4, 5, 6, 7]
    assert PL.window_positions(1, 4) == [0, 0, 0, 1]
    assert PL.window_positions(0, 1) == [0]
    assert PL.window_count(7, 4) == 4
    assert PL.window_count(1, 4) == 2
    assert PL.window_count(0, 16) == 1


def test_masked_mean_averages_only_the_distinct_positions():
    rng = np.random.default_rng(0)
    stack = rng.normal(size=(5, 4, 3)).astype(np.float32)
    counts = np.array([1, 2, 3, 4, 2])
    got = PL.masked_mean(stack, counts)
    for i, c in enumerate(counts):
        assert np.allclose(got[i], stack[i, 4 - c:].mean(axis=0), atol=1e-6)


def test_window_one_features_are_the_single_position_state():
    """The generalisation claim, at the level of the feature array."""
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(7, 4, 3)).astype(np.float32)
    counts = np.array([4, 4, 3, 2, 1, 4, 4])
    last = stack[:, -1, :]
    for spec in (PL.PoolSpec("l", 1, "last"), PL.PoolSpec("m", 1, "mean"),
                 PL.PoolSpec("c", 1, "concat")):
        got = PL.pooled_features(spec, stack, counts)
        assert np.array_equal(got, last), f"{spec.mode} at window 1 is not h_t"


def test_counts_to_mask_marks_the_trailing_slots():
    m = PL.counts_to_mask(np.array([1, 3, 4]), 4)
    assert m.tolist() == [[False, False, False, True],
                          [False, True, True, True],
                          [True, True, True, True]]


# --------------------------------------------------------------------------
# 3. the learned pool reduces to the deployed head at window 1
# --------------------------------------------------------------------------

def test_attention_pool_at_window_one_is_the_deployed_head():
    """Bit-identical logits, and the same draw from the ambient RNG."""
    torch.manual_seed(7)
    ref = MLPHead(in_dim=6, hidden_dim=8, n_bins=4, dropout=0.0)
    torch.manual_seed(7)
    pooled = PL.AttnPoolHead(in_dim=6, hidden_dim=8, n_bins=4, dropout=0.0)
    for k, v in ref.state_dict().items():
        assert torch.equal(v, pooled.mlp.state_dict()[k]), k

    x = torch.randn(5, 6)
    mask = torch.ones(5, 1, dtype=torch.bool)
    ref.eval()
    pooled.eval()
    assert torch.equal(ref(x), pooled(x.unsqueeze(1), mask))


def test_training_the_attention_pool_at_window_one_reproduces_train_head():
    """Not equivalence by argument: the two trained MLPs must agree bit for bit."""
    rng = np.random.default_rng(3)
    n, d = 400, 6
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    cfg = {"seed": 11, "lr": 1e-3, "weight_decay": 0.01, "batch_size": 64,
           "max_epochs": 4, "patience": 99}

    torch.manual_seed(11)
    ref = MLPHead(d, 8, 2, dropout=0.0)
    train_head(ref, x[:300], y[:300], x[300:], y[300:], cfg)

    torch.manual_seed(11)
    pooled = PL.AttnPoolHead(d, 8, 2, dropout=0.0)
    mask = np.ones((n, 1), dtype=bool)
    PL.train_attn_head(pooled, x[:300, None, :], mask[:300], y[:300],
                       x[300:, None, :], mask[300:], y[300:], cfg)

    for k, v in ref.state_dict().items():
        assert torch.equal(v, pooled.mlp.state_dict()[k]), f"{k} diverged"


def test_attention_weights_ignore_padded_slots():
    torch.manual_seed(5)
    head = PL.AttnPoolHead(4, 8, 3, dropout=0.0)
    x = torch.randn(2, 4, 4)
    mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
    a = head.weights(x, mask)
    assert torch.allclose(a.sum(dim=1), torch.ones(2))
    assert float(a[0, 0]) == 0.0 and float(a[0, 1]) == 0.0


# --------------------------------------------------------------------------
# 4. the decoder identity -- guidance.py is untouched, so the copy must not drift
# --------------------------------------------------------------------------

@pytest.mark.model
def test_pooled_guided_sample_at_window_one_matches_guided_sample(generator):
    """Same molecules and the same token counts as `guidance.guided_sample`.

    The token half is not decoration: §C25.0.2 claims a pooled readout is free under
    both accountings, and this is the measurement of that claim.
    """
    from property_to_go.binning import binner_from_dict
    from property_to_go.compute import ComputeMeter
    from property_to_go.config import load_config, read_json
    from property_to_go.guidance import Windows

    ck_path = OUT / "c17_probe_layers" / "head_aromatic_rings_frozen_state_L12.pt"
    data = OUT / "pilot_50k_p2"
    if not ck_path.exists() or not (data / "target_intervals.json").exists():
        pytest.skip("C17 checkpoints or the phase-2 dataset are absent")

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    iv = read_json(data / "target_intervals.json")["aromatic_rings"]
    win_d = read_json(data / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    policy = load_config("base_policy")

    def fresh_head():
        h = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
        h.load_state_dict(ck["state_dict"])
        h.eval()
        return h

    binner = binner_from_dict(ck["binner"])
    m_a, m_b = ComputeMeter().start(), ComputeMeter().start()
    a = guided_sample(
        generator, TargetScorer(fresh_head(), binner, iv["lo"], iv["hi"]),
        windows.fn("throughout"), policy, n_molecules=8, seed=101, top_k=8, lam=1.0,
        eps=1e-6, backend="cached", batch_size=8, layer=12, meter=m_a)
    m_a.stop()
    b = PL.pooled_guided_sample(
        generator,
        PL.PooledTargetScorer(fresh_head(), binner, iv["lo"], iv["hi"],
                              PL.VARIANTS_BY_NAME["last1"]),
        windows.fn("throughout"), policy, n_molecules=8, seed=101, top_k=8, lam=1.0,
        eps=1e-6, backend="cached", batch_size=8, layer=12, window=1, meter=m_b)
    m_b.stop()

    assert a == b, "the pooled sampler at window 1 returned different molecules"
    assert m_a.processed_tokens_actual == m_b.processed_tokens_actual
    assert m_a.processed_tokens_full_recompute == m_b.processed_tokens_full_recompute


@pytest.mark.model
def test_a_wider_window_costs_no_extra_tokens_per_step(generator):
    """§C25.0.2's compute claim, stated the only way it can be checked.

    A wider window changes which molecules are produced, so *total* tokens differ
    trivially through their lengths.  The claim is that the token bill is a function of
    the molecules alone and not of the readout: every decoding step costs one committed
    token plus `top_k` candidate tokens, whatever the window.  That identity is asserted
    here for window 1 and window 4, so a pooled arm's `processed_tokens_actual` is
    comparable with a single-position arm's without an adjustment.
    """
    from property_to_go.binning import binner_from_dict
    from property_to_go.compute import ComputeMeter
    from property_to_go.config import load_config, read_json
    from property_to_go.guidance import Windows

    ck_path = OUT / "c17_probe_layers" / "head_aromatic_rings_frozen_state_L12.pt"
    data = OUT / "pilot_50k_p2"
    if not ck_path.exists() or not (data / "target_intervals.json").exists():
        pytest.skip("C17 checkpoints or the phase-2 dataset are absent")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    iv = read_json(data / "target_intervals.json")["aromatic_rings"]
    win_d = read_json(data / "windows.json")
    windows = Windows(t33=win_d["t33"], t67=win_d["t67"], source=win_d["source"])
    policy = load_config("base_policy")
    binner = binner_from_dict(ck["binner"])

    top_k = 8
    for wname in ("last1", "mean4"):
        h = MLPHead(ck["in_dim"], ck["hidden_dim"], ck["n_bins"], ck["dropout"])
        h.load_state_dict(ck["state_dict"])
        h.eval()
        spec = PL.VARIANTS_BY_NAME[wname]
        m = ComputeMeter().start()
        seqs = PL.pooled_guided_sample(
            generator, PL.PooledTargetScorer(h, binner, iv["lo"], iv["hi"], spec),
            windows.fn("throughout"), policy, n_molecules=8, seed=101, top_k=top_k,
            lam=1.0, eps=1e-6, backend="cached", batch_size=8, layer=12,
            window=spec.window, meter=m)
        m.stop()
        steps = sum(len(s) - 1 for s in seqs)
        assert m.processed_tokens_actual == (1 + top_k) * steps, wname


# --------------------------------------------------------------------------
# 5. the executed artefacts
# --------------------------------------------------------------------------

def test_extraction_reproduces_the_c17_single_position_states():
    """§C25.0.1 limb A, read from the artefact rather than restated."""
    s = _load(STATES / "window_states_summary.json")
    assert s["validity_gate"]["all_bit_identical"] is True
    for row in s["validity_gate"]["per_layer"]:
        if row.get("checked"):
            assert row["bit_identical"] is True
            assert row["max_abs_difference"] == 0.0


def test_window_one_head_reproduces_the_deployed_head_exactly():
    """§C25.0.1 limb B. The residual is a number and the number has to be 0.0."""
    r = _load(SWEEP)
    gate = r["validity_gate"]
    assert gate["per_cell"], "no last1 cell was trained, so the gate did not run"
    for row in gate["per_cell"]:
        assert row["auroc_residual"] == 0.0, row
        assert row["nll_residual"] == 0.0, row
    assert gate["max_residual"] == 0.0
    assert gate["passes_at_exactly_zero"] is True


def test_every_comparison_is_seed_matched_and_three_seeded():
    r = _load(SWEEP)
    for key, cell in r["cells"].items():
        assert len(cell["per_seed"]) == 3, key
        assert [e["head_seed"] for e in cell["per_seed"]] == [1234, 2345, 3456], key
    for c in r["comparisons"]:
        base = r["cells"][f"{c['depth']}_L{c['probe_point']}_{c['property']}_last1"]
        assert c["auroc_last1"] == base["across_seeds"]["auroc"]["mean"]
        assert c["bootstrap_bonferroni"]["alpha"] == pytest.approx(
            0.05 / r["bonferroni_family_size"])


def test_section_quotes_the_validity_gate_residual(section_text):
    r = _load(SWEEP)
    assert f"{r['validity_gate']['max_residual']}" in section_text or (
        "0.0" in section_text)
    s = _load(STATES / "window_states_summary.json")
    assert str(s["n_prefix_rows"]) in section_text.replace(",", "")


def test_section_quotes_every_pooling_margin(section_text):
    """Artefact binding for the headline table: each margin appears as printed."""
    r = _load(SWEEP)
    missing = []
    for c in r["comparisons"]:
        printed = f"{c['auroc_margin']:+.4f}"
        if printed not in section_text:
            missing.append((c["depth"], c["probe_point"], c["property"],
                            c["variant"], printed))
    assert not missing, f"margins absent from the section text: {missing}"


def test_section_quotes_the_decision_rule_verdicts(section_text):
    summary = _load(SUMMARY)
    for name, v in summary["rules"].items():
        assert name in section_text, name
        assert ("fires" if v["fires"] else "does not fire") in section_text.lower()


def test_end_to_end_arms_are_bound_to_their_artefacts(section_text):
    summary = _load(SUMMARY)
    for arm in summary.get("end_to_end", []):
        assert f"{arm['hit_rate']:.4f}" in section_text, arm
        assert f"{arm['validity']:.4f}" in section_text, arm


def test_head_seed_replication_is_bound_to_its_artefacts(section_text):
    summary = _load(SUMMARY)
    rep = summary.get("head_seed_replication")
    if not rep:
        pytest.skip("head-seed replication not run")
    for arm in rep["arms"]:
        for hs, hr in arm["hit_rate_by_head_seed"].items():
            assert f"{hr:.4f}" in section_text, (arm["name"], hs, hr)
        assert f"{arm['span']:.4f}" in section_text, arm["name"]


def test_head_seed_replication_prints_the_best_of_n_comparison(section_text):
    """§C25.0.7 asks for "whether Rule B still fires under each head seed".

    That is a comparison, not a hit rate, so the comparator and the signed advantage are
    bound here as well -- otherwise a section could quote three guided numbers and never
    say which of them beats its own compute-matched best-of-N.
    """
    summary = _load(SUMMARY)
    rep = summary.get("head_seed_replication")
    if not rep:
        pytest.skip("head-seed replication not run")
    for arm in rep["arms"]:
        for hs, b in arm.get("best_of_n_by_head_seed", {}).items():
            assert f"{b['hit_rate']:.4f}" in section_text, (arm["name"], hs, "best-of-N")
        for hs, adv in arm.get("advantage_by_head_seed", {}).items():
            assert f"{adv:+.4f}" in section_text, (arm["name"], hs, "advantage")


def test_section_states_the_precommitted_head_seed_verdict(section_text):
    """§C25.0.7 committed, before the numbers existed, to a specific demotion.

    If the Rule-B arm's three head seeds span >= 0.05 the section must say C23's Rule B is
    **not replicated** at the head-seed level, "whatever the mean does".  This test makes
    the commitment binding rather than decorative.
    """
    summary = _load(SUMMARY)
    rep = summary.get("head_seed_replication")
    if not rep:
        pytest.skip("head-seed replication not run")
    arms = [a for a in rep["arms"] if a["property"] == "hbd_count"]
    assert arms, "the arm carrying C23's Rule B is absent from the replication"
    arm = arms[0]
    if not arm["span_within_limit"]:
        assert "not replicated" in section_text.lower(), (
            f"the hbd_count arm spans {arm['span']:.4f} >= {arm['span_limit']}, so "
            "§C25.0.7 requires the section to report C23's Rule B as not replicated "
            "at the head-seed level")
        assert arm.get("beats_best_of_n_on_all_head_seeds") is False


def _results_half(section_text: str) -> str:
    """Everything below the freeze line -- the prereg states the rules, not the verdicts."""
    marker = "Results below this line were written after the run"
    idx = section_text.find(marker)
    assert idx > 0, "the freeze marker separating prereg from results is missing"
    return section_text[idx:]


def test_each_rule_carries_its_verdict_on_the_same_line(section_text):
    """Stronger than a document-wide containment check.

    `test_section_quotes_the_decision_rule_verdicts` only asks that the words appear
    somewhere.  This asks that, in the results half, some line naming the rule also
    carries that rule's actual verdict -- so a fired rule cannot be silently written up
    beside "does not fire", or the reverse.
    """
    summary = _load(SUMMARY)
    results = _results_half(section_text)
    for name, v in summary["rules"].items():
        lines = [ln.lower() for ln in results.splitlines() if name in ln]
        assert lines, f"{name} is never named in the results half of the section"
        if v["fires"]:
            ok = any("fire" in ln and "does not fire" not in ln for ln in lines)
        else:
            ok = any("does not fire" in ln for ln in lines)
        assert ok, (f"{name} fires={v['fires']} but no line in the results half states "
                    f"that verdict; lines were {lines}")


def test_rule_p4_quotes_the_trivial_head_it_failed_to_beat(section_text):
    """P4 is §8.3's actual negative result; a null here has to be printed with its number."""
    summary = _load(SUMMARY)
    p4 = summary["rules"]["Rule P4"]
    assert f"{p4['trivial_auroc']:.4f}" in section_text
    assert f"{p4['best_auroc']:.4f}" in section_text
    assert f"{p4['threshold_auroc']:.4f}" in section_text
