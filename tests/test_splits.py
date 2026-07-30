import numpy as np
import pytest

from property_to_go.splits import (
    assign_split,
    check_no_group_leakage,
    split_by_group,
)

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_same_molecule_never_crosses_splits():
    # every canonical molecule appears in four prefix rows, as in the real dataset
    molecules = [f"C{i}" for i in range(2000)]
    groups = [m for m in molecules for _ in range(4)]
    splits = split_by_group(groups, FRACTIONS, seed=11)
    check_no_group_leakage(np.array(groups), splits)  # raises on leakage

    train = {g for g, s in zip(groups, splits) if s == "train"}
    test = {g for g, s in zip(groups, splits) if s == "test"}
    val = {g for g, s in zip(groups, splits) if s == "val"}
    assert train & test == set()
    assert train & val == set()
    assert val & test == set()


def test_leakage_check_actually_catches_leakage():
    groups = np.array(["A", "A", "B"])
    bad = np.array(["train", "test", "train"])
    with pytest.raises(AssertionError):
        check_no_group_leakage(groups, bad)


def test_fractions_are_approximately_respected():
    groups = [f"mol{i}" for i in range(20000)]
    splits = split_by_group(groups, FRACTIONS, seed=11)
    for name, want in FRACTIONS.items():
        got = float((splits == name).mean())
        assert abs(got - want) < 0.015, f"{name}: {got:.3f} vs {want}"


def test_assignment_is_deterministic_and_order_independent():
    groups = [f"mol{i}" for i in range(500)]
    a = split_by_group(groups, FRACTIONS, seed=3)
    b = split_by_group(list(reversed(groups)), FRACTIONS, seed=3)
    assert list(a) == list(reversed(list(b)))
    assert assign_split("mol7", FRACTIONS, 3) == assign_split("mol7", FRACTIONS, 3)


def test_seed_changes_the_split():
    groups = [f"mol{i}" for i in range(2000)]
    a = split_by_group(groups, FRACTIONS, seed=1)
    b = split_by_group(groups, FRACTIONS, seed=2)
    assert (a != b).mean() > 0.1


def test_split_does_not_depend_on_dataset_size():
    small = [f"mol{i}" for i in range(100)]
    large = small + [f"mol{i}" for i in range(100, 5000)]
    a = split_by_group(small, FRACTIONS, seed=11)
    b = split_by_group(large, FRACTIONS, seed=11)[:100]
    assert list(a) == list(b), "adding data must not move existing molecules between splits"
