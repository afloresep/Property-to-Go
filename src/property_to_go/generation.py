"""Sampling from the frozen generator, and frozen hidden-state extraction.

Everything here treats the generator as read-only.  Two facts about GP-MoLFormer's
linear attention are relied on and are asserted in tests/:

1.  Attention is causal, so a single forward pass over a completed sequence yields,
    at position t, exactly the hidden state of the prefix x_{<=t}.  Four prefix
    states per trajectory therefore cost one forward pass, not four.
2.  Right padding is exact.  Padded keys are zeroed by the attention mask and the
    causal cumulative sum never lets a later position influence an earlier one, so
    a right-padded batch gives bitwise-identical states to unpadded singles.
"""

from __future__ import annotations

import numpy as np
import torch

from .compute import ComputeMeter
from .model_io import FrozenGenerator


def sequence_content(ids: list[int], bos: int, eos: int, pad: int) -> list[int]:
    """Strip bos/eos/pad, returning the SMILES content tokens."""
    return [i for i in ids if i not in (bos, eos, pad)]


def _generation_kwargs(gen: FrozenGenerator, policy: dict) -> dict:
    kw = dict(
        do_sample=bool(policy["do_sample"]),
        temperature=float(policy["temperature"]),
        top_k=policy["top_k"],
        top_p=policy["top_p"],
        max_length=int(policy["max_length"]),
        pad_token_id=gen.pad_id,
        eos_token_id=gen.eos_id,
        bos_token_id=gen.bos_id,
    )
    # transformers only installs a top-k warper when top_k is a non-zero int, so
    # top_k=None means full-vocabulary sampling.  Kept explicit to stop the model's
    # own generation_config default (top_k=50) leaking in.
    return kw


@torch.no_grad()
def sample_unconditional(
    gen: FrozenGenerator,
    policy: dict,
    n: int,
    seed: int,
    meter: ComputeMeter | None = None,
) -> list[list[int]]:
    """Draw n unconditional trajectories under the fixed base policy pi_0.

    Returns full token-id sequences including <bos> and <eos>.
    """
    torch.manual_seed(seed)
    batch_size = int(policy["batch_size"])
    out: list[list[int]] = []
    kw = _generation_kwargs(gen, policy)

    while len(out) < n:
        b = min(batch_size, n - len(out))
        seqs = gen.model.generate(num_return_sequences=b, **kw)
        for row in seqs.tolist():
            trimmed = _trim_after_eos(row, gen.eos_id, gen.pad_id)
            out.append(trimmed)
            if meter is not None:
                meter.add_forward(len(trimmed))
                meter.molecules_returned += 1
    return out[:n]


def _trim_after_eos(row: list[int], eos: int, pad: int) -> list[int]:
    """Keep everything up to and including the first eos; drop trailing padding."""
    trimmed: list[int] = []
    for t in row:
        if t == pad and trimmed:
            break
        trimmed.append(t)
        if t == eos and len(trimmed) > 1:
            break
    return trimmed


@torch.no_grad()
def continue_from_prefixes(
    gen: FrozenGenerator,
    prefixes: list[list[int]],
    n_each: int,
    policy: dict,
    seed: int,
    meter: ComputeMeter | None = None,
    batch_size: int = 256,
) -> list[list[list[int]]]:
    """Sample `n_each` base-policy continuations of every prefix.

    Prefixes are grouped by exact length so no padding is ever needed, which
    sidesteps the left-padding question entirely.

    Returns continuations[i][j] = full token sequence for prefix i, sample j.
    """
    torch.manual_seed(seed)
    kw = _generation_kwargs(gen, policy)
    results: list[list[list[int]]] = [[] for _ in prefixes]

    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(prefixes):
        by_len.setdefault(len(p), []).append(i)

    for length, idxs in sorted(by_len.items()):
        per_batch = max(1, batch_size // n_each)
        for start in range(0, len(idxs), per_batch):
            chunk = idxs[start : start + per_batch]
            inp = torch.tensor(
                [prefixes[i] for i in chunk for _ in range(n_each)],
                dtype=torch.long,
                device=gen.device,
            )
            seqs = gen.model.generate(input_ids=inp, **kw)
            rows = seqs.tolist()
            for c, i in enumerate(chunk):
                for j in range(n_each):
                    row = _trim_after_eos(rows[c * n_each + j], gen.eos_id, gen.pad_id)
                    results[i].append(row)
                    if meter is not None:
                        meter.add_forward(len(row))
                        meter.molecules_returned += 1
    return results


@torch.no_grad()
def hidden_states_for_positions(
    gen: FrozenGenerator,
    sequences: list[list[int]],
    positions: list[list[int]],
    layer: int = -1,
    batch_size: int = 96,
    meter: ComputeMeter | None = None,
) -> list[np.ndarray]:
    """Frozen hidden states of `sequences[i]` at each index in `positions[i]`.

    Position k means the state after consuming tokens 0..k inclusive, i.e. the
    representation of the prefix ending at token k.  One forward pass per sequence
    serves all its requested positions.

    Sequences are sorted by length before batching so padding stays minimal.
    """
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    out: list[np.ndarray | None] = [None] * len(sequences)

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        lens = [len(sequences[i]) for i in chunk]
        maxlen = max(lens)
        ids = torch.full((len(chunk), maxlen), gen.pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, : lens[r]] = torch.tensor(sequences[i], dtype=torch.long)
            mask[r, : lens[r]] = 1
        ids = ids.to(gen.device)
        mask = mask.to(gen.device)

        res = gen.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = res.hidden_states[layer].float().cpu().numpy()
        for r, i in enumerate(chunk):
            out[i] = hs[r, positions[i], :].copy()
        if meter is not None:
            meter.add_forward(int(sum(lens)))

    return [o for o in out]  # type: ignore[return-value]


@torch.no_grad()
def top_k_next_tokens(
    gen: FrozenGenerator,
    sequences: list[list[int]],
    k: int,
    temperature: float = 1.0,
    batch_size: int = 96,
    meter: ComputeMeter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The base policy's top-`k` next tokens after each sequence, and their logprobs.

    Returns `(ids, logprobs)`, both `(len(sequences), k)`.  `logprobs` are
    `log_softmax` over the **full** vocabulary, exactly as `guided_sample` computes
    them, so the renormalised base weights over the top-k are recoverable and the
    headroom measurement scores the same candidate set the decoder would see.

    Right-padded with an explicit attention mask and batched by sorted length, the
    same contract `hidden_states_for_positions` uses and
    `test_right_padding_does_not_change_hidden_states` asserts is exact here.
    """
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    ids_out = np.zeros((len(sequences), k), dtype=np.int64)
    lp_out = np.zeros((len(sequences), k), dtype=np.float64)

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
            return_dict=True,
        )
        logits = res.logits.float() / float(temperature)
        lp = torch.log_softmax(logits, dim=-1)
        for r, i in enumerate(chunk):
            # the last *real* position, not the last padded one
            top = torch.topk(lp[r, lens[r] - 1, :], k)
            ids_out[i] = top.indices.cpu().numpy()
            lp_out[i] = top.values.cpu().numpy()
        if meter is not None:
            meter.add_forward(int(sum(lens)))

    return ids_out, lp_out


@torch.no_grad()
def base_logprobs_and_states(
    gen: FrozenGenerator,
    sequences: list[list[int]],
    layer: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience for the compatibility spike: last-position logprobs and state."""
    ids = torch.tensor(sequences, dtype=torch.long, device=gen.device)
    res = gen.model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
    logprobs = torch.log_softmax(res.logits[:, -1, :].float(), dim=-1).cpu().numpy()
    states = res.hidden_states[layer][:, -1, :].float().cpu().numpy()
    return logprobs, states


def repeat_cache(past_key_values, repeats: int):
    """Repeat a Molformer linear-attention cache along the batch dimension.

    The cache contract used by the released code is, per layer, a pair

        (key_running_sum, key_value_outer_running_sum)

    where only the last slice of `key_running_sum` is ever read and its length
    along dim 2 encodes how many tokens have been consumed.  The repeat therefore
    materialises only the last slice and re-expands it, which keeps this cheap.
    """
    out = []
    for layer_cache in past_key_values:
        k, kv = layer_cache[0], layer_cache[1]
        seq_len = k.shape[2]
        k_last = k[:, :, -1:].repeat_interleave(repeats, dim=0)
        k_rep = k_last.expand(-1, -1, seq_len, -1)
        kv_rep = kv.repeat_interleave(repeats, dim=0)
        out.append((k_rep, kv_rep) + tuple(layer_cache[2:]))
    return tuple(out)
