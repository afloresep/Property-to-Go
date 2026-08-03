"""C17 -- probing every layer of the frozen generator, not only the last one.

Every number in `reports/pilot_report.md` reads the **final** hidden layer, because
`configs/model.yaml` sets `hidden_layer: -1`.  §8.3 records that as an unaddressed
limitation, and §13.1's aromatic-ring crossover -- the project's most-defended claim --
is stated about a representation while being measured at one depth.

This module is additive.  It changes no existing behaviour and no existing artefact.

Two things live here.

1.  `hidden_states_all_layers`, which extracts **every** probe point in a single forward
    pass.  GP-MoLFormer returns 13 `hidden_states` entries (index 0 = embedding output,
    1..12 = the twelve transformer layers), so a 13-point sweep costs exactly the same
    *processed tokens* as the single-layer extraction the pipeline already runs.  That is
    the whole reason C17 is cheap, and it is why the cost is reported in tokens rather
    than in wall time (§11.7).

    The batching contract is `generation.hidden_states_for_positions`'s, unchanged:
    right padding with an explicit attention mask, batches formed over length-sorted
    sequences.  `tests/test_probe_layers.py` asserts that probe point 12 of this function
    is **bit-identical** to `generation.hidden_states_for_positions(..., layer=-1)`, so
    the two paths cannot silently diverge.

2.  `train_one_probe`, the per-layer head trainer.  It is deliberately a transcription of
    the `frozen_state` branch of `scripts/03_train_heads.py` rather than a tidier
    reimplementation: the point of C17 is a *cross-layer* comparison, and any difference
    in recipe between "the final layer as script 03 trained it" and "an earlier layer as
    C17 trained it" would be indistinguishable from a layer effect.  The identity is
    checked empirically rather than by inspection -- `scripts/16_probe_layer_sweep.py`
    refuses to report anything unless probe point 12 reproduces
    `outputs/pilot_50k_heads_p2/head_metrics.json` to four decimal places.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from . import metrics as M
from .binning import interval_probability
from .compute import ComputeMeter
from .heads import MLPHead, train_head
from .model_io import FrozenGenerator

#: Probe points of GP-MoLFormer: index 0 is the embedding output, 1..12 the layers.
#: Kept as a function of the loaded model rather than a literal, so a different
#: checkpoint cannot silently be swept over the wrong range.
def probe_points(gen: FrozenGenerator) -> tuple[int, ...]:
    return tuple(range(int(gen.model.config.num_hidden_layers) + 1))


@torch.no_grad()
def hidden_states_all_layers(
    gen: FrozenGenerator,
    sequences: Sequence[Sequence[int]],
    positions: Sequence[Sequence[int]],
    layers: Sequence[int],
    out: dict[int, np.ndarray] | None = None,
    row_offsets: Sequence[int] | None = None,
    batch_size: int = 96,
    meter: ComputeMeter | None = None,
) -> dict[int, list[np.ndarray]] | dict[int, np.ndarray]:
    """Frozen hidden states at every requested probe point, one forward pass per batch.

    Same contract as `generation.hidden_states_for_positions` -- position `k` means the
    state after consuming tokens `0..k` inclusive -- extended to return a dict keyed by
    probe point.

    `out` and `row_offsets` exist for the 8 GB case: pass a dict of preallocated
    (n_rows, hidden) arrays (`np.lib.format.open_memmap` works) plus, for each sequence,
    the row at which its positions start, and the states are written straight to disk
    instead of being concatenated in RAM.
    """
    layers = list(layers)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    streaming = out is not None
    if streaming and row_offsets is None:
        raise ValueError("row_offsets is required when out is given")
    collected: dict[int, list[np.ndarray | None]] = {
        L: [None] * len(sequences) for L in layers
    }

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        lens = [len(sequences[i]) for i in chunk]
        maxlen = max(lens)
        ids = torch.full((len(chunk), maxlen), gen.pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, : lens[r]] = torch.tensor(sequences[i], dtype=torch.long)
            mask[r, : lens[r]] = 1

        res = gen.model(
            input_ids=ids.to(gen.device),
            attention_mask=mask.to(gen.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        for L in layers:
            hs = res.hidden_states[L].float().cpu().numpy()
            for r, i in enumerate(chunk):
                block = hs[r, list(positions[i]), :]
                if streaming:
                    o = int(row_offsets[i])
                    out[L][o : o + len(block)] = block
                else:
                    collected[L][i] = block.copy()
        # One forward pass serves every probe point, so the token cost is charged once.
        if meter is not None:
            meter.add_forward(int(sum(lens)))
        del res

    if streaming:
        return out
    return {L: [c for c in collected[L]] for L in layers}  # type: ignore[return-value]


def train_one_probe(
    x: np.ndarray,
    y: np.ndarray,
    y_bin: np.ndarray,
    binner,
    masks: dict[str, np.ndarray],
    quartile: np.ndarray,
    intervals: dict[str, tuple[float, float]],
    head_cfg: dict[str, Any],
    head_seed: int,
    trainer=None,
) -> tuple[dict, np.ndarray, MLPHead]:
    """Train and evaluate one `frozen_state` head. Mirrors scripts/03_train_heads.py.

    The `torch.manual_seed` before `MLPHead(...)` is load-bearing and is script 03's:
    `MLPHead.__init__` draws its Linear initialisation from the ambient RNG, so seeding
    only inside `train_head` would leave initialisation incidental.  Removing it here
    would make every layer's head a different draw and quietly convert seed noise into
    a depth effect.

    `trainer` is an optional drop-in for `heads.train_head` with the identical
    signature `(head, x_train, y_train, x_val, y_val, cfg) -> TrainResult`.  It exists
    for C31, which trains 13 probe points x 3 head seeds x 3 properties on GPU via
    `generality.train_head_on_device` -- a function `tests/test_generality.py` already
    asserts is bit-identical to `heads.train_head` on CPU.  Default `None` means
    `heads.train_head`, so every existing caller is unchanged and no molecular number
    can move.
    """
    train = trainer if trainer is not None else train_head
    torch.manual_seed(int(head_seed))
    head = MLPHead(
        in_dim=x.shape[1],
        hidden_dim=int(head_cfg["hidden_dim"]),
        n_bins=binner.n_bins,
        dropout=float(head_cfg["dropout"]),
    )
    tr = train(
        head,
        x[masks["train"]],
        y_bin[masks["train"]],
        x[masks["val"]],
        y_bin[masks["val"]],
        {**head_cfg, "seed": int(head_seed)},
    )
    probs_test = head.predict_proba(x[masks["test"]])
    entry = {
        "head_seed": int(head_seed),
        "input_dim": int(x.shape[1]),
        "best_epoch": tr.best_epoch,
        "epochs_run": len(tr.history),
        "test": M.evaluate(
            probs_test, y[masks["test"]], y_bin[masks["test"]], binner, intervals
        ),
        "test_by_quartile": M.evaluate_by_group(
            probs_test,
            y[masks["test"]],
            y_bin[masks["test"]],
            quartile[masks["test"]],
            binner,
            intervals,
        ),
    }
    return entry, probs_test, head


def across_seeds(per_seed: list[dict]) -> dict:
    """Spread of the headline metrics over head-training seeds (script 03's `_across_seeds`)."""

    def col(fn):
        v = np.array([fn(e) for e in per_seed], dtype=np.float64)
        return {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "values": v.tolist(),
        }

    return {
        "n_seeds": len(per_seed),
        "nll": col(lambda e: e["test"]["nll"]),
        "auroc": col(lambda e: e["test"]["intervals"]["target"]["auroc"]),
        "expected_value_mae": col(lambda e: e["test"]["expected_value_mae"]),
    }


def paired_bootstrap_diff(fn, a_args, b_args, n_boot=1000, seed=0, alpha=0.05):
    """95% CI for metric(a) - metric(b), resampling the same rows for both heads.

    Byte-for-byte the estimator in `scripts/03_train_heads.py`, with `alpha` exposed so
    C17.0.6's Bonferroni-corrected level (alpha = 0.05/13) can be requested without a
    second implementation.  `tests/test_probe_layers.py` asserts the two agree at
    alpha = 0.05 on a fixed input.
    """
    rng = np.random.default_rng(seed)
    n = len(a_args[0])
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(fn(*[x[idx] for x in a_args]) - fn(*[x[idx] for x in b_args]))
        except Exception:
            continue
    if not diffs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "mean": float(np.mean(diffs)),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
    }


def target_interval_scores(probs: np.ndarray, binner, lo: float, hi: float) -> np.ndarray:
    """`P(y in [lo, hi))` under a head's bin distribution."""
    return interval_probability(probs, binner, lo, hi)


# ---------------------------------------------------------------------------------
# The C17.0 decision rules, in code, so the verdict is a lookup rather than a judgement
# formed after seeing the table.  Thresholds are the pre-registered ones and are
# arguments with pre-registered defaults, not literals buried in a branch.
# ---------------------------------------------------------------------------------

#: One head-seed sd (pilot_report.md §13.2 measures at most 0.0041, rounded down).
SEED_SD = 0.004
#: §13.2's own stated safe-margin floor for a claim about a difference in AUROC.
MATERIAL_MARGIN = 0.010
#: C17.0.6 rule 2: a neighbour has to move too, or the maximum is called noise.
NEIGHBOUR_MARGIN = 0.005


def crossover_verdict(
    auroc_by_layer: dict[int, float],
    trivial_auroc: float,
    seed_sd: float = SEED_SD,
    material: float = MATERIAL_MARGIN,
) -> dict:
    """Score C17.0.4's three-way rule. Pure function of the numbers; no side conditions.

    The caller still has to apply the bootstrap and NLL conditions before promoting
    `ARTEFACT` to a claim -- `verdict` here is the AUROC arm of the rule only, and that
    is stated in the returned dict rather than left implicit.
    """
    best_layer = max(auroc_by_layer, key=lambda L: auroc_by_layer[L])
    best = auroc_by_layer[best_layer]
    if best >= trivial_auroc + material:
        verdict = "ARTEFACT"
    elif best >= trivial_auroc - seed_sd:
        verdict = "TIE_AT_THE_BEST_LAYER"
    else:
        verdict = "REPRESENTATION"
    return {
        "verdict_auroc_arm": verdict,
        "best_layer": int(best_layer),
        "best_auroc": float(best),
        "trivial_auroc": float(trivial_auroc),
        "margin_over_trivial": float(best - trivial_auroc),
        "thresholds": {"seed_sd": seed_sd, "material_margin": material},
        "note": (
            "ARTEFACT is the AUROC arm only. C17.0.4 additionally requires the "
            "Bonferroni-corrected paired-bootstrap CI to exclude zero and the layer's "
            "NLL not to be worse than the trivial head's."
        ),
    }


def isolated_spike(auroc_by_layer: dict[int, float], layer: int, reference_layer: int,
                   material: float = MATERIAL_MARGIN,
                   neighbour: float = NEIGHBOUR_MARGIN) -> dict:
    """C17.0.6 rule 2: is `layer`'s advantage over `reference_layer` a one-point spike?

    Returns the components rather than a bare bool, so the write-up can quote why a
    maximum was or was not accepted.
    """
    ref = auroc_by_layer[reference_layer]
    gain = auroc_by_layer[layer] - ref
    neighbours = [L for L in (layer - 1, layer + 1) if L in auroc_by_layer and L != reference_layer]
    n_gains = {int(L): float(auroc_by_layer[L] - ref) for L in neighbours}
    supported = bool(neighbours) and all(g >= neighbour for g in n_gains.values())
    return {
        "layer": int(layer),
        "reference_layer": int(reference_layer),
        "gain_over_reference": float(gain),
        "clears_material_margin": bool(gain >= material),
        "neighbour_gains": n_gains,
        "neighbours_support_it": supported,
        "genuinely_better": bool(gain >= material and supported),
    }
