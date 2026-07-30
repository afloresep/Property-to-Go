"""The quality descriptors must actually detect the degeneracies they claim to.

Every threshold in `DEGENERACY_RULES` is checked against a molecule that should trip
it and a drug-like molecule that should not, because a degeneracy detector that never
fires would silently turn "guidance produces junk" into "no evidence of junk".
"""

from __future__ import annotations

import numpy as np
import pytest

from property_to_go import quality as Q

IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_a_drug_like_molecule_trips_nothing():
    for smi in (IBUPROFEN, CAFFEINE):
        flags = Q.degeneracy_flags(Q.molecule_quality(smi))
        assert not any(flags.values()), f"{smi} should look like a normal molecule: {flags}"


@pytest.mark.parametrize(
    "smiles,rule",
    [
        ("CCCCCCCCCCCCCCCC", "long_alkyl_tail"),          # hexadecane
        ("CCCCCCCCCCCCCCCC", "mostly_carbon"),
        ("C1CCCCCCCCCCC1", "macrocyclic"),                # cyclododecane
        ("CCO.CCO.CCO", "multi_fragment"),
        ("CC(=O)[O-]", "net_charged"),
    ],
)
def test_each_degeneracy_rule_fires_on_its_target(smiles, rule):
    flags = Q.degeneracy_flags(Q.molecule_quality(smiles))
    assert flags[rule], f"{rule} should fire on {smiles}"


def test_the_multi_fragment_carbanion_case_from_the_pilot_is_caught():
    """A real string from the late-guided aromatic-ring run that RDKit calls valid.

    It parses, it has three aromatic rings, and it would be scored as a target hit.
    It is not a molecule.
    """
    smi = "C.C.C.C=CC=c1[n-]c(CC(CC)CCC)c(CC(C)C)c1=CC.[CH2-]C[CH2-]"
    q = Q.molecule_quality(smi)
    assert q is not None, "RDKit does parse this, which is the whole problem"
    flags = Q.degeneracy_flags(q)
    assert flags["multi_fragment"] and flags["net_charged"]


def test_invalid_smiles_returns_none():
    assert Q.molecule_quality("this is not a molecule") is None
    assert Q.molecule_quality("c1ccccc1C(") is None


class TestLongestAcyclicCarbonPath:
    """Ring atoms are excluded, so the induced subgraph is a forest and the longest
    path is a tree diameter. These pin down that reading."""

    def _chain(self, smi: str) -> float:
        return Q.molecule_quality(smi)["longest_chain"]

    def test_straight_chain_counts_atoms_not_bonds(self):
        assert self._chain("CCCC") == 4

    def test_ring_carbons_are_not_counted(self):
        assert self._chain("c1ccccc1") == 0
        assert self._chain("C1CCCCC1") == 0

    def test_a_substituent_measures_only_the_tail(self):
        assert self._chain("CCCCc1ccccc1") == 4

    def test_branching_takes_the_longest_path_through_the_branch_point(self):
        # neopentyl-ish: the longest path crosses the quaternary carbon, so it is
        # methyl -> centre -> chain, not the chain alone
        assert self._chain("CC(C)(C)CCC") == 5

    def test_two_separate_tails_do_not_add_up(self):
        # each tail is 3 carbons; a ring sits between them, so the answer is 3, not 6
        assert self._chain("CCCc1ccc(CCC)cc1") == 3

    def test_heteroatoms_break_the_chain(self):
        assert self._chain("CCCOCCC") == 3

    def test_acyclic_sp2_carbons_count_too(self):
        # A long polyene is as implausible as a long alkane, so counting it is wanted;
        # the descriptor is "acyclic carbon path", not "saturated chain".
        assert self._chain("C=CC=CC") == 5


class TestQualityPanel:
    def test_empty_input_reports_zero_rather_than_crashing(self):
        assert Q.quality_panel([]) == {"n": 0}

    def test_panel_reports_every_declared_descriptor(self):
        qs = [Q.molecule_quality(s) for s in (IBUPROFEN, CAFFEINE, "CCCCCCCCCCCCCCCC")]
        panel = Q.quality_panel(qs)
        assert panel["n"] == 3
        for key in Q.QUALITY_KEYS:
            assert key in panel["descriptors"], f"{key} missing from panel"
        assert panel["degeneracy_rate"]["any"] == pytest.approx(1 / 3)


class TestBootstrapDifference:
    def test_identical_samples_give_a_ci_containing_zero(self):
        x = np.linspace(0.0, 1.0, 200)
        out = Q.bootstrap_difference(x, x.copy(), n_boot=500, seed=1)
        assert out["difference"] == pytest.approx(0.0)
        assert not out["excludes_zero"]

    def test_a_large_shift_is_detected(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 0.1, 300)
        b = rng.normal(0.0, 0.1, 300)
        out = Q.bootstrap_difference(a, b, n_boot=500, seed=1)
        assert out["difference"] > 0.9
        assert out["excludes_zero"]
        assert out["lo"] > 0

    def test_empty_sample_is_nan_not_an_exception(self):
        out = Q.bootstrap_difference(np.array([]), np.array([1.0, 2.0]), n_boot=10)
        assert np.isnan(out["difference"])
