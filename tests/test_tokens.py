import numpy as np

from property_to_go.tokens import (
    FEATURE_NAMES,
    N_FEATURES,
    classify,
    prefix_features,
    prefix_features_from_ids,
)


def f(tokens):
    return dict(zip(FEATURE_NAMES, prefix_features(tokens)))


def test_atom_classification():
    assert classify("C").is_atom and not classify("C").is_aromatic_atom
    assert classify("c").is_atom and classify("c").is_aromatic_atom
    assert classify("Cl").is_atom and classify("Cl").element == "Cl"
    assert classify("[nH]").is_atom and classify("[nH]").is_aromatic_atom
    assert classify("[nH]").element == "N"
    assert classify("[C@@H]").is_atom and classify("[C@@H]").element == "C"
    assert classify("[2H]").is_atom, "isotope prefix must not hide the atom"
    assert classify("[N+]").element == "N"


def test_non_atom_classification():
    for tok in ["(", ")", "=", "#", "/", "\\", ".", "<bos>", "<eos>", "<pad>"]:
        assert not classify(tok).is_atom, tok
    for tok in ["1", "9", "%10", "%12"]:
        assert classify(tok).is_ring_marker, tok
        assert not classify(tok).is_atom


def test_feature_vector_shape_and_names_align():
    v = prefix_features(["C", "c", "1"])
    assert v.shape == (N_FEATURES,)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert v.dtype == np.float32


def test_benzene_prefix_counts():
    d = f(["c", "1", "c", "c", "c", "c", "c", "1"])
    assert d["prefix_len"] == 8
    assert d["n_atom_tokens"] == 6
    assert d["n_aromatic_atom_tokens"] == 6
    assert d["n_ring_markers"] == 2
    assert d["n_open_rings"] == 0, "the ring digit appears twice, so the ring is closed"
    assert d["n_element_C"] == 6
    assert d["frac_aromatic_atoms"] == 1.0


def test_open_ring_tracking():
    assert f(["c", "1", "c", "c"])["n_open_rings"] == 1
    assert f(["C", "1", "C", "C", "2", "C"])["n_open_rings"] == 2
    assert f(["C", "1", "C", "C", "2", "C", "1"])["n_open_rings"] == 1


def test_branch_depth():
    d = f(["C", "(", "C", "(", "C"])
    assert d["n_branch_open"] == 2 and d["branch_depth"] == 2
    d = f(["C", "(", "C", ")", "C"])
    assert d["branch_depth"] == 0 and d["n_branch_close"] == 1


def test_element_counts_and_other_bucket():
    d = f(["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "[Se]", "[Si]"])
    for e in ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P"):
        assert d[f"n_element_{e}"] == 1, e
    assert d["n_element_other"] == 2, "Se and Si fall in the other bucket"
    assert d["n_bracket_atoms"] == 2


def test_empty_prefix_is_all_zero_and_finite():
    v = prefix_features([])
    assert np.all(np.isfinite(v))
    assert v.sum() == 0.0


def test_special_tokens_are_stripped_by_id_helper():
    id_to_token = {0: "<bos>", 4: "C", 5: "c", 1: "<eos>"}
    a = prefix_features_from_ids([0, 4, 4, 5, 1], id_to_token)
    b = prefix_features(["C", "C", "c"])
    assert np.allclose(a, b)
