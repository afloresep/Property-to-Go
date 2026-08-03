"""RDKit property calculation.

Primary properties of the executed pilot (its headline claims rest on these two):
    clogp           -- Crippen cLogP, continuous, spread over many fragments
    aromatic_rings  -- number of aromatic rings, discrete and structurally local

Diagnostic control only (never a primary result):
    mol_weight      -- too confounded with atom count and sequence length

Wired in for phase 2 (see `docs/LEXICAL_LOCALITY.md`) on 2026-07-30:
    hbd_count       -- H-bond donors, discrete, locally visible (N-H / O-H per atom)
    rotatable_bonds -- discrete, bond-local, but ring membership is a global veto
    tpsa            -- continuous, additive over polar atoms
    qed             -- continuous, nonlinear function of eight descriptors; the most
                       diffuse of the set, and an actual PMO benchmark task

The phase-2 battery is the six properties of `PREDICTED_LOCALITY_ORDER`. cLogP is
**demoted, not dropped**: one property of six, out of the title, and it does not carry
the headline number (decision A7, docs/TODO.md). `mol_weight` stays a diagnostic and is
not on the locality scatter.

Adding the four cannot move any executed pilot number, and that is checked rather than
asserted:

* `compute_properties` is unchanged, so molecule *validity* -- which is what every
  pilot validity, uniqueness and hit rate is computed over -- is still decided by
  `Chem.MolFromSmiles` alone. `compute_all_properties` never returns None for a
  parseable molecule; a QED failure yields `qed: None` for that one field.
* The phase-2 target intervals come from new `target_interval_rule` entries, so the
  three frozen pilot intervals are re-derived by the identical code path.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

PRIMARY_PROPERTIES = ("clogp", "aromatic_rings")
DIAGNOSTIC_PROPERTIES = ("mol_weight",)

#: Wired in 2026-07-30 (docs/TODO.md C2). Was `CANDIDATE_PROPERTIES`.
PHASE2_PROPERTIES = ("hbd_count", "rotatable_bonds", "tpsa", "qed")

#: Kept as an alias so phase-1 call sites keep working.
CANDIDATE_PROPERTIES = PHASE2_PROPERTIES

ALL_PROPERTIES = PRIMARY_PROPERTIES + DIAGNOSTIC_PROPERTIES + PHASE2_PROPERTIES

#: Integer-valued. Every one of these MUST also be in `bestofn.INTEGER_PROPERTIES`
#: or the boundary-selection bug of docs/HANDOFF.md §4 comes back; a test asserts it.
DISCRETE_PROPERTIES = frozenset({"aromatic_rings", "hbd_count", "rotatable_bonds"})
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

#: The properties on the locality scatter: the six of PREDICTED_LOCALITY_ORDER.
#: `mol_weight` is excluded because it is a diagnostic control, not a target.
LOCALITY_BATTERY = PREDICTED_LOCALITY_ORDER

#: Prediction P2 is stated over two *named* classes, and it names them slightly
#: differently from the locality ordering above.  Transcribed here exactly as written in
#: docs/LEXICAL_LOCALITY.md §4 rather than tidied, because tidying a pre-registration
#: after the fact is the failure mode the pre-registration exists to prevent:
#:
#:   "declines with position for diffuse properties (cLogP, QED, TPSA) and stays
#:    roughly flat for local count properties (rings, HBD, largest ring)"
#:
#: Two consequences, both reported rather than resolved by us:
#:   * TPSA is "diffuse" in P2 but "medium" locality in §5's ordering. That is a
#:     tension inside the pre-registration itself.
#:   * `largest ring size` was never added to the battery, and `rotatable_bonds` --
#:     which is in the battery -- is named in *neither* P2 class. It is therefore
#:     unassigned for P2 and reported on its own, not quietly folded into whichever
#:     group would help.
P2_DIFFUSE_PROPERTIES = ("tpsa", "clogp", "qed")
P2_LOCAL_COUNT_PROPERTIES = ("aromatic_rings", "hbd_count")
P2_UNASSIGNED_PROPERTIES = ("rotatable_bonds",)


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


def compute_all_properties(
    smiles: str, extras: frozenset[str] | set[str] | tuple[str, ...] | None = None
) -> dict[str, Any] | None:
    """Every property in `ALL_PROPERTIES`, or None if RDKit will not parse the SMILES.

    `extras` restricts which phase-2 fields are computed, and exists purely for cost:
    `QED.qed` is ~1 ms, which is nothing per molecule and real over the ~160 k
    candidates a single `full_recompute`-matched best-of-N draws.  Restricting it never
    changes a value, only which keys are present.  Pass None for all four.

    This is the phase-2 entry point.  It differs from
    `compute_properties` | `compute_candidate_properties` in exactly one way, and the
    difference is load-bearing: **a QED failure does not invalidate the molecule.**

    `compute_candidate_properties` returns None when `QED.qed` raises, which is right
    for a function whose whole job is the four staged properties.  Reused here it would
    silently reclassify a parseable molecule as invalid, moving every validity,
    uniqueness and hit rate the pilot reported.  So validity stays exactly what
    `compute_properties` says it is -- `Chem.MolFromSmiles` succeeded -- and a molecule
    whose QED cannot be computed carries `qed: None`.

    Downstream, a `None` property value is never a hit and never enters a mean; callers
    must filter.  `phase2_property_array` does that and reports how many it dropped.
    """
    base = compute_properties(smiles)
    if base is None:
        return None
    want = set(PHASE2_PROPERTIES if extras is None else extras)
    mol = parse(smiles)
    out = dict(base)
    if "hbd_count" in want:
        out["hbd_count"] = int(Lipinski.NumHDonors(mol))
    if "rotatable_bonds" in want:
        out["rotatable_bonds"] = int(Lipinski.NumRotatableBonds(mol))
    if "tpsa" in want:
        out["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
    if "qed" in want:
        try:
            out["qed"] = float(QED.qed(mol))
        except Exception:
            # A QED of 0.0 is a meaningful value and must not double as an error code.
            out["qed"] = None
    return out


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
