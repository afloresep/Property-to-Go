"""Contracts the pilot depends on that can only be checked against the real checkpoint.

These are the assumptions that would silently corrupt every downstream number if
they were wrong, so they are asserted against the frozen model rather than assumed.
"""

import numpy as np
import pytest
import torch

from property_to_go import generation
from property_to_go.config import load_config
from property_to_go.guidance import _candidate_states_cached, _candidate_states_full

pytestmark = pytest.mark.model

PREFIXES = [
    "CCOc1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc2[nH]ccc2c1",
    "CN1CCN(CC1)c1ncnc2[nH]ccc12",
]


def _ids(generator, smiles):
    return generator.tokenizer(smiles, return_tensors="pt").input_ids[0, :-1].tolist()


def test_forward_pass_is_deterministic(generator):
    """deterministic_eval must pin the random feature map, or nothing is reproducible."""
    assert generator.model.config.deterministic_eval is True
    ids = torch.tensor([_ids(generator, PREFIXES[0])], device=generator.device)
    with torch.no_grad():
        a = generator.model(ids, output_hidden_states=True, use_cache=False)
        b = generator.model(ids, output_hidden_states=True, use_cache=False)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.hidden_states[-1], b.hidden_states[-1])


def test_right_padding_does_not_change_hidden_states(generator):
    """Batched prefix extraction pads on the right; that must be exact."""
    seqs = [_ids(generator, s) for s in PREFIXES]
    positions = [[len(s) - 1] for s in seqs]
    batched = generation.hidden_states_for_positions(generator, seqs, positions, batch_size=4)
    singles = generation.hidden_states_for_positions(generator, seqs, positions, batch_size=1)
    for b, s in zip(batched, singles):
        assert np.abs(b - s).max() < 1e-5


def test_a_prefix_state_from_a_full_pass_equals_a_pass_over_the_prefix(generator):
    """Causality: one forward over the completed sequence serves all four prefixes."""
    full = _ids(generator, PREFIXES[1])
    ks = [2, 5, 8]
    from_full = generation.hidden_states_for_positions(generator, [full], [ks])[0]
    for row, k in enumerate(ks):
        alone = generation.hidden_states_for_positions(generator, [full[: k + 1]], [[k]])[0][0]
        assert np.abs(from_full[row] - alone).max() < 1e-5


def test_candidate_backends_agree(generator):
    """cached (shared-prefix) and full recomputation must give the same states."""
    for smi in PREFIXES:
        ids = torch.tensor([_ids(generator, smi)], device=generator.device)
        with torch.no_grad():
            out = generator.model(ids, use_cache=True, return_dict=True)
            lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
            cand = torch.topk(lp, 8, dim=-1).indices
            h_full = _candidate_states_full(generator, ids, cand, layer=-1)
            h_cached = _candidate_states_cached(generator, out.past_key_values, cand, layer=-1)
        assert h_full.shape == h_cached.shape == (1, 8, generator.hidden_size)
        assert float((h_full - h_cached).abs().max()) < 1e-3, smi


def test_candidate_state_equals_extending_the_prefix_by_that_token(generator):
    """The candidate state must be the state of prefix+a, not of the prefix."""
    ids_list = _ids(generator, PREFIXES[0])
    ids = torch.tensor([ids_list], device=generator.device)
    with torch.no_grad():
        out = generator.model(ids, use_cache=True, return_dict=True)
        cand = torch.topk(torch.log_softmax(out.logits[:, -1, :].float(), -1), 8, -1).indices
        h_cached = _candidate_states_cached(generator, out.past_key_values, cand, layer=-1)[0]
    for j in range(8):
        extended = ids_list + [int(cand[0, j])]
        direct = generation.hidden_states_for_positions(
            generator, [extended], [[len(extended) - 1]]
        )[0][0]
        assert np.abs(direct - h_cached[j].numpy()).max() < 1e-3


def test_continuations_preserve_the_prefix_and_are_seed_reproducible(generator):
    policy = load_config("base_policy")
    prefix = _ids(generator, PREFIXES[0])
    a = generation.continue_from_prefixes(generator, [prefix], 4, policy, seed=1234)[0]
    b = generation.continue_from_prefixes(generator, [prefix], 4, policy, seed=1234)[0]
    c = generation.continue_from_prefixes(generator, [prefix], 4, policy, seed=999)[0]
    assert a == b, "same seed must reproduce the same continuations"
    assert a != c, "different seeds must explore"
    for seq in a:
        assert seq[: len(prefix)] == prefix


def test_top8_candidates_are_the_true_argmax(generator):
    ids = torch.tensor([_ids(generator, PREFIXES[2])], device=generator.device)
    with torch.no_grad():
        lp = torch.log_softmax(generator.model(ids).logits[:, -1, :].float(), -1)[0]
    top = torch.topk(lp, 8).indices.tolist()
    assert top == list(np.argsort(-lp.numpy())[:8])


def test_sequence_content_strips_specials(generator):
    seqs = generation.sample_unconditional(
        generator, load_config("base_policy") | {"batch_size": 4}, 4, seed=7
    )
    for s in seqs:
        content = generation.sequence_content(s, generator.bos_id, generator.eos_id, generator.pad_id)
        assert generator.bos_id not in content
        assert generator.eos_id not in content
        assert len(content) < len(s)


def test_batch_rows_are_independent(generator):
    """Guided decoding feeds pad tokens to finished rows without an attention mask.

    That is only safe if one row's tokens cannot influence another's, so assert it
    directly: changing row 1 entirely must leave row 0's outputs bit-identical.
    """
    a = _ids(generator, PREFIXES[0])
    b = _ids(generator, PREFIXES[1])[: len(a)]
    pads = [generator.pad_id] * len(a)

    with torch.no_grad():
        with_b = generator.model(
            torch.tensor([a, b], device=generator.device), output_hidden_states=True, use_cache=False
        )
        with_pads = generator.model(
            torch.tensor([a, pads], device=generator.device), output_hidden_states=True, use_cache=False
        )
    assert torch.equal(with_b.logits[0], with_pads.logits[0])
    assert torch.equal(with_b.hidden_states[-1][0], with_pads.hidden_states[-1][0])


def test_cached_stepping_is_row_independent(generator):
    """Same guarantee along the cache path used by guided decoding."""
    a = _ids(generator, PREFIXES[0])
    b = _ids(generator, PREFIXES[3])[: len(a)]
    with torch.no_grad():
        pair = generator.model(
            torch.tensor([a, b], device=generator.device), use_cache=True, return_dict=True
        )
        solo = generator.model(
            torch.tensor([a], device=generator.device), use_cache=True, return_dict=True
        )
        nxt = torch.tensor([[7], [9]], device=generator.device)
        step_pair = generator.model(
            nxt, past_key_values=pair.past_key_values, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        step_solo = generator.model(
            nxt[:1], past_key_values=solo.past_key_values, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
    diff = (step_pair.hidden_states[-1][0] - step_solo.hidden_states[-1][0]).abs().max()
    assert float(diff) < 1e-5


def test_unguided_loop_matches_hf_generate_distribution(generator):
    """The guidance script's `unguided` arm and best-of-N's sampler must be the same policy.

    guided_sample() decodes with its own loop while sample_unconditional() calls
    model.generate(); if their sampling policies differed, every guided-vs-baseline
    comparison would be confounded.  Compared distributionally over 256 molecules
    each, with tolerances set from the sampling error.
    """
    from property_to_go.guidance import Windows, guided_sample
    from property_to_go.properties import compute_properties

    policy = load_config("base_policy") | {"batch_size": 128}
    n = 256

    hf = generation.sample_unconditional(generator, policy, n, seed=31337)
    loop = guided_sample(
        generator, scorer=None, window_fn=Windows(t33=1, t67=2).fn("unguided"),
        policy=policy, n_molecules=n, seed=31337, batch_size=128,
    )

    def summarise(seqs):
        lens, logps = [], []
        for s in seqs:
            content = generation.sequence_content(s, generator.bos_id, generator.eos_id, generator.pad_id)
            lens.append(len(content))
            p = compute_properties(generator.tokenizer.decode(s, skip_special_tokens=True))
            if p:
                logps.append(p["clogp"])
        return np.array(lens), np.array(logps)

    lh, ch = summarise(hf)
    ll, cl = summarise(loop)

    # 4 standard errors of the difference of means
    len_tol = 4 * np.sqrt(lh.var() / len(lh) + ll.var() / len(ll))
    clogp_tol = 4 * np.sqrt(ch.var() / len(ch) + cl.var() / len(cl))
    assert abs(lh.mean() - ll.mean()) < len_tol, (lh.mean(), ll.mean(), len_tol)
    assert abs(ch.mean() - cl.mean()) < clogp_tol, (ch.mean(), cl.mean(), clogp_tol)
    assert len(ch) / len(hf) > 0.95 and len(cl) / len(loop) > 0.95, "both must be ~fully valid"
