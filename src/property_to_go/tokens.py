"""SMILES-token bookkeeping and the trivial-prefix feature set.

These features define the baseline the frozen hidden state has to beat.  The
baseline is deliberately built to be as strong as cheap prefix statistics can
reasonably get (element counts, ring bookkeeping, branch depth, ratios), because a
weak baseline would make the frozen-state head look good for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SPECIAL_TOKENS = frozenset({"<bos>", "<eos>", "<pad>", "<mask>", "<unk>"})
BOND_TOKENS = frozenset({"=", "#", "-", "/", "\\", ":", "~"})
BRANCH_OPEN = "("
BRANCH_CLOSE = ")"
DOT = "."

_RING_RE = re.compile(r"^(%\d{2}|\d)$")
_BRACKET_RE = re.compile(r"^\[(?P<iso>\d+)?(?P<sym>[A-Za-z][a-z]?|\*)(?P<rest>.*)\]$")
_PLAIN_RE = re.compile(r"^(Cl|Br|[BCNOSPFIbcnosp]|\*)$")

TRACKED_ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P")

FEATURE_NAMES = (
    "prefix_len",
    "n_atom_tokens",
    "n_aromatic_atom_tokens",
    "n_ring_markers",
    "n_open_rings",
    "n_branch_open",
    "n_branch_close",
    "branch_depth",
    "n_bond_tokens",
    "n_dot",
    "n_bracket_atoms",
    *[f"n_element_{e}" for e in TRACKED_ELEMENTS],
    "n_element_other",
    "frac_atom_tokens",
    "frac_aromatic_atoms",
    "atoms_per_ring_marker",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class TokenKind:
    is_atom: bool
    is_aromatic_atom: bool
    is_ring_marker: bool
    is_bracket_atom: bool
    element: str | None


@lru_cache(maxsize=8192)
def classify(token: str) -> TokenKind:
    """Classify one SMILES vocabulary token."""
    if token in SPECIAL_TOKENS:
        return TokenKind(False, False, False, False, None)
    if _RING_RE.match(token):
        return TokenKind(False, False, True, False, None)
    if token in BOND_TOKENS or token in (BRANCH_OPEN, BRANCH_CLOSE, DOT):
        return TokenKind(False, False, False, False, None)

    m = _BRACKET_RE.match(token)
    if m:
        sym = m.group("sym")
        aromatic = sym[0].islower()
        element = sym.capitalize() if sym != "*" else "other"
        return TokenKind(True, aromatic, False, True, element)

    if _PLAIN_RE.match(token):
        aromatic = token[0].islower()
        element = token.capitalize() if token != "*" else "other"
        return TokenKind(True, aromatic, False, False, element)

    # Anything unrecognised is counted as a non-atom structural token rather than
    # being silently treated as an atom.
    return TokenKind(False, False, False, False, None)


def prefix_features(tokens: list[str]) -> np.ndarray:
    """Trivial prefix statistics for a list of *content* tokens (no specials).

    Returns a float32 vector aligned with FEATURE_NAMES.
    """
    n_atom = n_arom = n_ring = n_open = n_close = n_bond = n_dot = n_bracket = 0
    depth = 0
    elements = {e: 0 for e in TRACKED_ELEMENTS}
    other = 0
    ring_open: set[str] = set()

    for tok in tokens:
        kind = classify(tok)
        if kind.is_atom:
            n_atom += 1
            n_arom += kind.is_aromatic_atom
            n_bracket += kind.is_bracket_atom
            if kind.element in elements:
                elements[kind.element] += 1
            else:
                other += 1
        elif kind.is_ring_marker:
            n_ring += 1
            # A ring digit toggles: first appearance opens the ring bond, second closes it.
            if tok in ring_open:
                ring_open.discard(tok)
            else:
                ring_open.add(tok)
        elif tok == BRANCH_OPEN:
            n_open += 1
            depth += 1
        elif tok == BRANCH_CLOSE:
            n_close += 1
            depth = max(0, depth - 1)
        elif tok == DOT:
            n_dot += 1
        elif tok in BOND_TOKENS:
            n_bond += 1

    n_tok = len(tokens)
    values = [
        n_tok,
        n_atom,
        n_arom,
        n_ring,
        len(ring_open),
        n_open,
        n_close,
        depth,
        n_bond,
        n_dot,
        n_bracket,
        *[elements[e] for e in TRACKED_ELEMENTS],
        other,
        n_atom / n_tok if n_tok else 0.0,
        n_arom / n_atom if n_atom else 0.0,
        n_atom / n_ring if n_ring else 0.0,
    ]
    return np.asarray(values, dtype=np.float32)


def prefix_features_from_ids(ids: list[int], id_to_token: dict[int, str]) -> np.ndarray:
    toks = [id_to_token[i] for i in ids]
    toks = [t for t in toks if t not in SPECIAL_TOKENS]
    return prefix_features(toks)
