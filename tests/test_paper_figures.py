"""Artefact-binding tests for the two paper figures.

`scripts/28_paper_figures.py` draws the two exhibits `reports/PAPER_WORKSHOP_DRAFT.md`
specifies.  A figure is a claim, so the numbers it plots are re-derived here from the
summary JSON they came from, in the style of `tests/test_report_matches_artifacts.py`.

The point of these tests is not that the PNGs exist.  It is that:

  * the §3.2 gap curve is `oracle_selected - head_selected` at **matched N** on both
    generators, computed by the same subtraction, and agrees with C27's independently
    recorded `E2_price_of_ground_truth.gap_per_n` on generator 1;
  * the cross-generator agreement the draft quotes (0.0111 / 0.0081 / 0.0369) is what the
    plotted lines actually show;
  * the deployed-arm budget markers are the quantity `pilot_report.md` §25.4 says the two
    generators never matched -- gen 1 at N ~ 8-9, gen 2 at N ~ 3.6 -- because the visual
    claim of figure 1 is that the *curves* coincide while the *markers* do not; and
  * every number the draft and `pilot_report.md` §25.2 print for this curve is the number
    the figure draws, to the decimal place they print it at.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"

ANCHORS = ("aromatic_rings", "hbd_count", "qed")
GRID = (1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32)

#: The draft's §3.2 table and `pilot_report.md` §25.2, transcribed from the prose.
PUBLISHED_GAPS = {
    "aromatic_rings": {
        "g1": {2: 0.0347, 4: 0.1278, 9: 0.3024, 16: 0.3792, 32: 0.3724},
        "g2": {2: 0.0200, 4: 0.0855, 9: 0.2282, 16: 0.3442, 32: 0.3764},
    },
    "hbd_count": {
        "g1": {2: 0.0177, 4: 0.0736, 9: 0.2172, 16: 0.3491, 32: 0.4415},
        "g2": {2: 0.0186, 4: 0.0775, 9: 0.2289, 16: 0.3591, 32: 0.4750},
    },
    "qed": {
        "g1": {2: 0.0331, 4: 0.1257, 9: 0.3534, 16: 0.5344, 32: 0.6836},
        "g2": {2: 0.0344, 4: 0.1329, 9: 0.3630, 16: 0.5327, 32: 0.6597},
    },
}

#: Mean absolute cross-generator difference over the whole grid, as the draft quotes it.
PUBLISHED_MAD = {"aromatic_rings": 0.0369, "hbd_count": 0.0111, "qed": 0.0081}


def _load_module():
    path = ROOT / "scripts" / "28_paper_figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def gaps(mod):
    return mod.gap_curves()


def test_gap_is_the_difference_of_the_two_curves(gaps):
    """The plotted gap is exactly oracle_selected - head_selected, both generators."""
    for tag, name in (("g1", "c27"), ("g2", "c33")):
        metrics = json.loads((OUT / f"{name}_summary" / f"{name}_metrics.json").read_text())
        for prop in ANCHORS:
            curves = metrics["properties"][prop]["curves"]
            for n in GRID:
                expected = (curves["oracle_selected"][str(n)]["hit_rate_mean"]
                            - curves["head_selected"][str(n)]["hit_rate_mean"])
                assert gaps[tag][prop][n] == pytest.approx(expected, abs=0.0)


def test_generator_1_gap_agrees_with_c27s_own_recorded_curve(gaps):
    """C27 recorded this curve itself; recomputing it must not change it."""
    c27 = json.loads((OUT / "c27_summary" / "c27_metrics.json").read_text())
    for prop in ANCHORS:
        recorded = c27["properties"][prop]["E2_price_of_ground_truth"]["gap_per_n"]
        for n in GRID:
            assert gaps["g1"][prop][n] == pytest.approx(recorded[str(n)], abs=1e-12)


def test_gap_is_zero_at_n1_by_construction(gaps):
    """With one candidate there is nothing to select, so the arms cannot differ."""
    for tag in ("g1", "g2"):
        for prop in ANCHORS:
            assert gaps[tag][prop][1] == 0.0


def test_published_gap_values_are_what_the_figure_draws(gaps):
    for prop, per_gen in PUBLISHED_GAPS.items():
        for tag, per_n in per_gen.items():
            for n, published in per_n.items():
                assert round(gaps[tag][prop][n], 4) == published, (prop, tag, n)


def test_cross_generator_agreement_is_what_the_draft_quotes(gaps):
    for prop, published in PUBLISHED_MAD.items():
        diffs = [abs(gaps["g1"][prop][n] - gaps["g2"][prop][n]) for n in GRID]
        assert round(sum(diffs) / len(diffs), 4) == published, prop


def test_aromatic_rings_gap_on_generator_1_is_not_monotone(gaps):
    """The caption must say "grows", not "grows monotonically" -- this is why."""
    g1 = gaps["g1"]["aromatic_rings"]
    assert g1[24] > g1[32], "gen-1 aromatic rings peaks at N=24 and dips at N=32"
    assert round(g1[24], 4) == 0.3839 and round(g1[32], 4) == 0.3724
    for prop in ("hbd_count", "qed"):
        for tag in ("g1", "g2"):
            series = [gaps[tag][prop][n] for n in GRID]
            assert series == sorted(series), (prop, tag)


def test_oracle_share_of_baseline_hit_rate_at_n32():
    """§25.2 states two units and they are different quantities; both are checked."""
    bounds = []
    for tag, name in (("g1", "c27"), ("g2", "c33")):
        metrics = json.loads((OUT / f"{name}_summary" / f"{name}_metrics.json").read_text())
        for prop in ANCHORS:
            curves = metrics["properties"][prop]["curves"]
            oracle = curves["oracle_selected"]["32"]["hit_rate_mean"]
            head = curves["head_selected"]["32"]["hit_rate_mean"]
            bounds.append((oracle - head) / oracle)
    assert round(min(bounds), 2) == 0.37
    assert round(max(bounds), 2) == 0.71


def test_deployed_budget_markers_do_not_coincide(mod):
    """Figure 1's whole point: the curves coincide, the budget markers do not."""
    marks = mod.deployed_budgets_in_n()
    g1 = list(marks["g1"].values())
    g2 = list(marks["g2"].values())
    assert 8.0 < min(g1) and max(g1) < 10.0, g1
    assert 3.5 < min(g2) and max(g2) < 4.0, g2
    assert min(g1) > max(g2) * 2, "the two generators' arms are not within 2x on the N axis"


def test_frontier_panels_carry_every_k_sweep_cell(mod):
    """30 guided cells on generator 2 (C31) and 30 on generator 1 (C28)."""
    assert len(mod.frontier_and_cells("g2")["cells"]) == 30
    assert len(mod.frontier_and_cells("g1")["cells"]) == 30
    for tag in ("g1", "g2"):
        frontiers = mod.frontier_and_cells(tag)["frontiers"]
        for prop in ANCHORS:
            f = frontiers[prop]
            assert len(f["tokens"]) == len(GRID)
            assert f["oracle"][0] == pytest.approx(f["head"][0], abs=0.0)
            assert all(a >= b for a, b in zip(f["oracle"], f["head"]))


def test_figures_are_reproducible_from_committed_artefacts_only(mod, tmp_path):
    """The script must draw from `outputs/*_summary/` and write nothing else."""
    one = mod.fig1(tmp_path / "fig1.png")
    two = mod.fig2(tmp_path / "fig2.png")
    assert one.exists() and one.stat().st_size > 10_000
    assert two.exists() and two.stat().st_size > 10_000
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig1.png", "fig2.png"]


def test_committed_figures_exist():
    for name in ("fig1_oracle_gap_vs_n.png", "fig2_frontiers.png"):
        assert (OUT / "paper_figures" / name).exists(), name
