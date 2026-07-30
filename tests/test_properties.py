import math

import pytest

from property_to_go import properties
from property_to_go import properties as P


def test_known_molecules():
    benzene = properties.compute_properties("c1ccccc1")
    assert benzene["aromatic_rings"] == 1
    assert benzene["n_heavy_atoms"] == 6
    assert benzene["mol_weight"] == pytest.approx(78.11, abs=0.05)

    naphthalene = properties.compute_properties("c1ccc2ccccc2c1")
    assert naphthalene["aromatic_rings"] == 2

    cyclohexane = properties.compute_properties("C1CCCCC1")
    assert cyclohexane["aromatic_rings"] == 0, "saturated ring is not aromatic"

    # a fused aromatic + saturated ring system counts only the aromatic ring
    tetralin = properties.compute_properties("c1ccc2c(c1)CCCC2")
    assert tetralin["aromatic_rings"] == 1


def test_clogp_is_monotone_in_chain_length():
    values = [properties.compute_properties("C" * n)["clogp"] for n in (2, 4, 6, 8)]
    assert values == sorted(values), "cLogP must increase with alkane chain length"


def test_invalid_smiles_returns_none():
    for bad in ["", "c1ccccc", "C(((", "XYZ", "[Zz]"]:
        assert properties.compute_properties(bad) is None
        assert properties.canonical_smiles(bad) is None


def test_canonicalisation_is_stable():
    a = properties.canonical_smiles("C1=CC=CC=C1")
    b = properties.canonical_smiles("c1ccccc1")
    assert a == b
    assert properties.canonical_smiles(a) == a, "canonical form must be a fixed point"


def test_properties_invariant_to_smiles_writing():
    a = properties.compute_properties("OCCc1ccccc1")
    b = properties.compute_properties("c1ccccc1CCO")
    assert a["canonical_smiles"] == b["canonical_smiles"]
    assert math.isclose(a["clogp"], b["clogp"])
    assert a["aromatic_rings"] == b["aromatic_rings"]


def test_validity_and_uniqueness():
    smiles = ["c1ccccc1", "C1=CC=CC=C1", "CCO", "not_a_molecule"]
    assert properties.validity(smiles) == pytest.approx(0.75)
    # benzene written two ways collapses to one canonical molecule out of three valid
    assert properties.uniqueness(smiles) == pytest.approx(2 / 3)


class TestStagedCandidateProperties:
    """The four phase-2 properties (docs/LEXICAL_LOCALITY.md).

    These are staged, not wired in. The first test is the important one: adding them
    must not change anything already executed.
    """

    def test_candidates_are_not_in_all_properties(self):
        """Guards every executed result. If this fails, phase-1 numbers may have moved."""
        for prop in P.CANDIDATE_PROPERTIES:
            assert prop not in P.ALL_PROPERTIES, (
                f"{prop} leaked into ALL_PROPERTIES; scripts 02/04 would change behaviour"
            )

    def test_known_values(self):
        # aspirin: one phenol-free carboxylic acid O-H -> 1 donor; TPSA 63.6
        q = P.compute_candidate_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert q["hbd_count"] == 1
        assert q["tpsa"] == pytest.approx(63.6, abs=0.5)
        assert 0.0 <= q["qed"] <= 1.0
        # ethane has no rotatable bond under the strict definition
        assert P.compute_candidate_properties("CC")["rotatable_bonds"] == 0
        # butane has exactly one
        assert P.compute_candidate_properties("CCCC")["rotatable_bonds"] == 1

    def test_ring_bonds_do_not_count_as_rotatable(self):
        """This is why rotatable_bonds is only mid-locality: ring membership is a
        global veto, so a local token count cannot determine it."""
        assert P.compute_candidate_properties("C1CCCCC1")["rotatable_bonds"] == 0

    def test_invalid_smiles_returns_none(self):
        assert P.compute_candidate_properties("not a molecule") is None

    def test_integer_properties_are_declared_integer(self):
        """The boundary-selection bug in docs/HANDOFF.md §4 reappears for any
        integer-valued property missing from INTEGER_PROPERTIES."""
        q = P.compute_candidate_properties("CC(=O)Oc1ccccc1C(=O)O")
        for prop in P.CANDIDATE_DISCRETE_PROPERTIES:
            assert float(q[prop]).is_integer(), f"{prop} is not integer-valued"

    def test_predicted_locality_order_covers_the_phase_two_battery(self):
        """The pre-registered ordering must name every property it will rank."""
        battery = set(P.PRIMARY_PROPERTIES) | set(P.CANDIDATE_PROPERTIES)
        assert set(P.PREDICTED_LOCALITY_ORDER) == battery
        assert len(P.PREDICTED_LOCALITY_ORDER) == len(battery), "no duplicates"
