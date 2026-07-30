"""Chemical-quality descriptors, for asking whether steered molecules are junk.

Hit rate answers "did the property land in the interval". It says nothing about
whether the molecule that got there is a plausible compound. The classic failure
mode in property optimisation is a controller that satisfies the objective by
degenerate means -- a long alkane tail to raise cLogP, a pile of fused rings to
raise a ring count -- and hit rate scores those as successes.

So every conditon is also described by:

    sa_score        synthetic accessibility, 1 (easy) to 10 (hard); Ertl & Schuffenhauer
    qed             quantitative estimate of drug-likeness, 0 to 1
    longest_chain   longest path of acyclic aliphatic carbons -- the cLogP-hack signature
    carbon_fraction carbons / heavy atoms; degenerate greasy molecules approach 1.0
    max_ring_size   ring sizes above 7 are rare in real small molecules
    n_macrocycles   rings of size >= 8
    n_fragments     disconnected components; > 1 means the string is not one molecule
    abs_formal_charge  RDKit accepts free carbanions, a real molecule generally is not one
    fraction_csp3, n_spiro, n_bridgehead, n_rotatable, lipinski_violations

`n_fragments` and `abs_formal_charge` matter because RDKit's parser is the validity
check used throughout this pilot, and it happily accepts strings like
`C.C.C.C=CC=c1[n-]c(...)` -- four loose methanes beside a charged ring. Those score
as "valid" and can score as target hits, so they have to be counted separately.

None of these are ground truth about synthesisability. They are the descriptors the
molecular-optimisation literature uses to catch reward hacking, and they are cheap.
The comparison that matters is always *within* the set of molecules that hit the
target: base-policy hits versus guided hits. Best-of-N returns base-policy hits by
construction, so base-policy-hit quality is exactly the quality of the baseline's
output.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, RDConfig, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

_SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_DIR not in sys.path:
    sys.path.append(_SA_DIR)
import sascorer  # noqa: E402  (RDKit contrib, only importable after the path append)

#: Descriptors where a *larger* value means a less plausible molecule.
HIGHER_IS_WORSE = ("sa_score", "longest_chain", "carbon_fraction", "max_ring_size",
                   "n_macrocycles", "lipinski_violations", "n_fragments",
                   "abs_formal_charge")
#: Descriptors where a *smaller* value means a less plausible molecule.
HIGHER_IS_BETTER = ("qed",)

QUALITY_KEYS = HIGHER_IS_WORSE + HIGHER_IS_BETTER + (
    "fraction_csp3", "n_spiro", "n_bridgehead", "n_rotatable", "n_rings",
)


def _longest_acyclic_carbon_path(mol: Chem.Mol) -> int:
    """Longest chain of non-aromatic, non-ring carbons, in atoms.

    Excluding ring atoms leaves an induced subgraph with no cycles, so each
    component is a tree and its longest path is the tree diameter -- two breadth
    first searches, no exponential path enumeration.
    """
    keep = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C" and not a.GetIsAromatic() and not a.IsInRing()
    }
    if not keep:
        return 0

    def neighbours(i: int) -> list[int]:
        return [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in keep]

    def farthest(src: int) -> tuple[int, int, set[int]]:
        dist = {src: 1}
        q = deque([src])
        far, best = src, 1
        while q:
            u = q.popleft()
            for v in neighbours(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if dist[v] > best:
                        far, best = v, dist[v]
                    q.append(v)
        return far, best, set(dist)

    seen: set[int] = set()
    diameter = 0
    for start in keep:
        if start in seen:
            continue
        end, _, component = farthest(start)      # end is an endpoint of a longest path
        seen |= component
        _, length, _ = farthest(end)             # its eccentricity is the diameter
        diameter = max(diameter, length)
    return diameter


def molecule_quality(smiles: str) -> dict[str, float] | None:
    """Quality descriptors for one SMILES, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return None
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    n_carbon = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    violations = sum(
        (
            Descriptors.MolWt(mol) > 500.0,
            Crippen.MolLogP(mol) > 5.0,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        )
    )
    return {
        "sa_score": float(sascorer.calculateScore(mol)),
        "qed": float(QED.qed(mol)),
        "longest_chain": float(_longest_acyclic_carbon_path(mol)),
        "carbon_fraction": float(n_carbon / heavy),
        "max_ring_size": float(max(ring_sizes)) if ring_sizes else 0.0,
        "n_macrocycles": float(sum(1 for s in ring_sizes if s >= 8)),
        "n_rings": float(len(ring_sizes)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "n_spiro": float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "n_bridgehead": float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "n_rotatable": float(Lipinski.NumRotatableBonds(mol)),
        "lipinski_violations": float(violations),
        "n_fragments": float(len(Chem.GetMolFrags(mol))),
        "abs_formal_charge": float(abs(Chem.GetFormalCharge(mol))),
    }


#: A molecule trips this if it shows any single unambiguous degeneracy signature.
#: Thresholds are deliberately generous: SA > 6 is the usual "hard to make" line,
#: an 8-carbon acyclic tail is longer than almost anything in a drug, and rings of
#: 8+ atoms are rare outside macrocyclic natural products.
DEGENERACY_RULES = {
    "hard_to_synthesise": lambda q: q["sa_score"] > 6.0,
    "long_alkyl_tail": lambda q: q["longest_chain"] >= 8,
    "mostly_carbon": lambda q: q["carbon_fraction"] > 0.90,
    "macrocyclic": lambda q: q["max_ring_size"] >= 8,
    "multi_fragment": lambda q: q["n_fragments"] > 1,
    "net_charged": lambda q: q["abs_formal_charge"] > 0,
}


def degeneracy_flags(q: dict[str, float]) -> dict[str, bool]:
    return {name: bool(rule(q)) for name, rule in DEGENERACY_RULES.items()}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def quality_panel(qualities: list[dict[str, float]]) -> dict[str, Any]:
    """Aggregate per-molecule descriptors into one condition's quality panel."""
    if not qualities:
        return {"n": 0}
    panel: dict[str, Any] = {"n": len(qualities), "descriptors": {}}
    for key in QUALITY_KEYS:
        panel["descriptors"][key] = _summary(np.array([q[key] for q in qualities]))
    flags = [degeneracy_flags(q) for q in qualities]
    panel["degeneracy_rate"] = {
        name: float(np.mean([f[name] for f in flags])) for name in DEGENERACY_RULES
    }
    panel["degeneracy_rate"]["any"] = float(np.mean([any(f.values()) for f in flags]))
    return panel


def bootstrap_difference(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), two independent samples."""
    if len(a) == 0 or len(b) == 0:
        return {"difference": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(
            b, len(b), replace=True
        ).mean()
    return {
        "difference": float(a.mean() - b.mean()),
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "excludes_zero": bool(np.quantile(diffs, 0.025) > 0 or np.quantile(diffs, 0.975) < 0),
    }
