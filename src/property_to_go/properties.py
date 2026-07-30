"""RDKit property calculation.

Primary properties (the scientific claims rest on these two):
    clogp           -- Crippen cLogP, continuous, spread over many fragments
    aromatic_rings  -- number of aromatic rings, discrete and structurally local

Diagnostic control only (never a primary result):
    mol_weight      -- too confounded with atom count and sequence length

Candidates staged for the next phase (see `docs/LEXICAL_LOCALITY.md`), deliberately NOT
in `ALL_PROPERTIES` yet so that adding them cannot silently change any executed result:
    hbd_count       -- H-bond donors, discrete, locally visible (N-H / O-H per atom)
    rotatable_bonds -- discrete, bond-local, but ring membership is a global veto
    tpsa            -- continuous, additive over polar atoms
    qed             -- continuous, nonlinear function of eight descriptors; the most
                       diffuse of the set, and an actual PMO benchmark task

They are ordered above by predicted lexical locality, high to low. That ordering is a
pre-registered prediction, not an observation.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

PRIMARY_PROPERTIES = ("clogp", "aromatic_rings")
DIAGNOSTIC_PROPERTIES = ("mol_weight",)
ALL_PROPERTIES = PRIMARY_PROPERTIES + DIAGNOSTIC_PROPERTIES

DISCRETE_PROPERTIES = frozenset({"aromatic_rings"})

#: Staged for the next phase. NOT in `ALL_PROPERTIES`, so nothing downstream changes
#: until a phase-2 script opts in. Wiring them in requires, in order:
#:   1. add to `ALL_PROPERTIES`;
#:   2. add integer-valued ones to `bestofn.INTEGER_PROPERTIES` -- see docs/HANDOFF.md §4
#:      for the boundary-selection bug this prevents;
#:   3. add a `target_interval_rule` entry in configs/guidance.yaml and regenerate
#:      `target_intervals.json` BEFORE inspecting any guided result.
CANDIDATE_PROPERTIES = ("hbd_count", "rotatable_bonds", "tpsa", "qed")
CANDIDATE_DISCRETE_PROPERTIES = frozenset({"hbd_count", "rotatable_bonds"})

#: Predicted lexical locality, high to low, from how SMILES writes each property.
#: This is a PRE-REGISTERED PREDICTION recorded before measurement, not a result.
#: See docs/LEXICAL_LOCALITY.md §5. Do not edit to match observed data; if the data
#: disagrees, that falsifies prediction P1 and is the finding.
PREDICTED_LOCALITY_ORDER = (
    "aromatic_rings",    # one lowercase atom + ring digit pair = one ring
    "hbd_count",         # per-atom N-H / O-H, additive, locally visible
    "rotatable_bonds",   # bond-local, but ring membership is a global veto
    "tpsa",              # additive over polar atoms, many small contributions
    "clogp",             # sum of atom-type contributions over the whole molecule
    "qed",               # nonlinear function of eight descriptors; most diffuse
)


def parse(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string, returning None when RDKit rejects it."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(smiles: str) -> str | None:
    mol = parse(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_properties(smiles: str) -> dict[str, Any] | None:
    """Return canonical SMILES plus the three properties, or None if invalid.

    Returns None for anything RDKit will not parse, so callers never have to
    decide what an "invalid" property value means.
    """
    mol = parse(smiles)
    if mol is None:
        return None
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "clogp": float(Crippen.MolLogP(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "mol_weight": float(Descriptors.MolWt(mol)),
        "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
    }


def compute_candidate_properties(smiles: str) -> dict[str, Any] | None:
    """The four staged phase-2 properties, or None if RDKit will not parse the SMILES.

    Kept separate from `compute_properties` so that adding them cannot change any
    already-executed result. A phase-2 caller merges both dicts.

    Definitional choices, recorded because they are the kind of thing that silently
    differs between tools and then cannot be reconciled later:

    * `hbd_count` uses `Lipinski.NumHDonors`, the Lipinski definition (N-H and O-H),
      not `rdMolDescriptors.CalcNumHBD`, which uses the stricter Gasteiger
      definition and gives different counts on amides.
    * `rotatable_bonds` uses `Lipinski.NumRotatableBonds` (RDKit's strict pattern:
      excludes amides and ring bonds).
    * `qed` can raise on pathological structures, so it is caught and the molecule is
      reported as unusable rather than silently scored as 0.0 -- a QED of 0.0 is a
      meaningful value and must not double as an error code.
    """
    mol = parse(smiles)
    if mol is None:
        return None
    try:
        qed_value = float(QED.qed(mol))
    except Exception:
        return None
    return {
        "hbd_count": int(Lipinski.NumHDonors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "qed": qed_value,
    }


def validity(smiles_list: list[str]) -> float:
    if not smiles_list:
        return 0.0
    return sum(parse(s) is not None for s in smiles_list) / len(smiles_list)


def uniqueness(smiles_list: list[str]) -> float:
    """Fraction of distinct canonical SMILES among the valid molecules."""
    canon = [c for c in (canonical_smiles(s) for s in smiles_list) if c is not None]
    if not canon:
        return 0.0
    return len(set(canon)) / len(canon)
