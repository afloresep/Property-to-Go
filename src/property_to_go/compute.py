"""Generator-compute accounting.

Compute matching in this project is done on *processed generator tokens*, not on
Python forward-call counts and not on returned molecule counts.  One "processed
token" is one token position pushed through the 12 transformer blocks.

Two counters are kept:

processed_tokens_actual
    what the implementation really did.  For the cached candidate backend a
    candidate evaluation costs 1 token (one step from a shared prefix cache).

processed_tokens_full_recompute
    what the README's reference implementation would have cost, i.e. re-running
    every candidate's whole prefix from scratch.  Reported alongside so the
    compute-matched baseline can be quoted under either accounting rule.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class ComputeMeter:
    processed_tokens_actual: int = 0
    processed_tokens_full_recompute: int = 0
    forward_calls: int = 0
    wall_seconds: float = 0.0
    molecules_returned: int = 0
    _t0: float | None = field(default=None, repr=False)

    def start(self) -> "ComputeMeter":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> "ComputeMeter":
        if self._t0 is not None:
            self.wall_seconds += time.perf_counter() - self._t0
            self._t0 = None
        return self

    def __enter__(self) -> "ComputeMeter":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def add_forward(self, actual_tokens: int, full_recompute_tokens: int | None = None) -> None:
        """Record one forward pass over `actual_tokens` non-padding token positions."""
        self.processed_tokens_actual += int(actual_tokens)
        self.processed_tokens_full_recompute += int(
            actual_tokens if full_recompute_tokens is None else full_recompute_tokens
        )
        self.forward_calls += 1

    def merge(self, other: "ComputeMeter") -> None:
        self.processed_tokens_actual += other.processed_tokens_actual
        self.processed_tokens_full_recompute += other.processed_tokens_full_recompute
        self.forward_calls += other.forward_calls
        self.wall_seconds += other.wall_seconds
        self.molecules_returned += other.molecules_returned

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        if d["molecules_returned"]:
            d["tokens_per_molecule_actual"] = (
                d["processed_tokens_actual"] / d["molecules_returned"]
            )
            d["tokens_per_molecule_full_recompute"] = (
                d["processed_tokens_full_recompute"] / d["molecules_returned"]
            )
        return d


def solve_best_of_n(
    target_tokens_per_molecule: float, base_tokens_per_molecule: float
) -> int:
    """Largest N whose best-of-N cost does not exceed the guided cost per molecule.

    best-of-N draws N base molecules and returns one, so it costs
    N * base_tokens_per_molecule tokens per returned molecule.
    """
    if base_tokens_per_molecule <= 0:
        raise ValueError("base_tokens_per_molecule must be positive")
    return max(1, int(target_tokens_per_molecule // base_tokens_per_molecule))
