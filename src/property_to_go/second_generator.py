"""C31 -- the second frozen molecular generator: `entropy/gpt2_zinc_87m`.

**This module adds a generator.  It adds nothing else.**  Every scientific component of
the pipeline -- the decoding rule, the property calculators, the binner, the interval
semantics, the token meter, the grouped splitter, the probe trainer, the best-of-N
selection key -- is the molecular one, imported.  That is the whole design: if the
crossing fails to replicate here, the failure cannot be an implementation difference in
the method, because there is no second implementation of the method.

Why this generator, and why the rule that forbade one is lifted: see
`outputs/c31_prereg/C31.0_preregistration.md` C31.0.1.  In one line -- the original brief
banned a second generator so that a *negative* result could not be blamed on generator
choice; the owner has explicitly instructed this run, and testing whether a *positive*
generalises is the inverse use.

What differs from GP-MoLFormer, and what that costs:

    architecture      GPT-2, full softmax attention, 12 blocks, 768 wide, 87.3M params.
                      GP-MoLFormer is linear attention, 12 blocks, 46.8M params.  Both
                      expose 13 probe points, which is a convenience and not a design.
    tokenizer         byte-level BPE over SMILES, 2707 tokens, multi-character merges
                      ('Cc', 'ccc', '(=').  GP-MoLFormer's vocabulary is atom-level.
                      This is why `smiles_atom_tokens` exists -- see below.
    KV cache          the standard `(key, value)` per layer.  `generation.repeat_cache`
                      is written for a linear-attention running-sum cache and is not
                      applicable; `generality.repeat_cache_gpt2` is, and is reused
                      rather than re-derived.  `guidance._candidate_states_cached`
                      dispatches on `gen.repeat_cache_fn`, which is the single line of
                      the molecular library C31 touches.
    specials          bos=0 `<s>`, eos=2 `</s>`, pad=1 `<pad>`, unk=3 `<unk>`.  The
                      model's own `config.pad_token_id` is unset, so `pad_id` reads the
                      tokenizer, which is why this class does not simply reuse
                      `FrozenGenerator`.

Right padding is exact here for the same reason it is exact for GP-MoLFormer: attention
is causal, and GPT-2 derives `position_ids` as `arange(seq_len)`, so with padding on the
right every real token sits at its true position and no later position can influence an
earlier one.  `tests/test_second_generator.py` asserts this rather than assuming it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .generality import repeat_cache_gpt2

# --------------------------------------------------------------------------- pins

#: The frozen second generator, pinned by commit exactly as `configs/model.yaml` pins
#: GP-MoLFormer, so "the model" is a checkable object and not a name.  Duplicated from
#: `configs/c31_second_generator.yaml` only as a default; every script loads the config.
ZINC_MODEL_REPO = "entropy/gpt2_zinc_87m"
ZINC_MODEL_REVISION = "f42a5a10e24c0350aeadb50865bd90a714d0b2bf"


@dataclass
class ZincGPT2Generator:
    """A frozen GPT-2 SMILES generator with `FrozenGenerator`'s surface.

    Deliberately duck-typed to `model_io.FrozenGenerator` rather than subclassing it:
    `FrozenGenerator.fingerprint` reads `config.deterministic_eval`, a GP-MoLFormer flag
    that does not exist here, and `pad_id` reads the model config, which this checkpoint
    leaves unset.  Everything the pipeline actually calls -- `model`, `device`, `bos_id`,
    `eos_id`, `pad_id`, `decode`, `id_to_token` -- is present with identical semantics.
    """

    model: Any
    tokenizer: Any
    device: torch.device
    config: dict[str, Any]

    #: `guidance._candidate_states_cached` dispatches on this.  Absent on
    #: `FrozenGenerator`, where the linear-attention `generation.repeat_cache` is used.
    repeat_cache_fn: Any = staticmethod(repeat_cache_gpt2)

    @property
    def bos_id(self) -> int:
        return int(self.model.config.bos_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.model.config.eos_token_id)

    @property
    def pad_id(self) -> int:
        # The checkpoint's config has no pad_token_id; the tokenizer does (<pad> = 1).
        pid = getattr(self.model.config, "pad_token_id", None)
        if pid is None:
            pid = self.tokenizer.pad_token_id
        return int(pid)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.n_embd)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.n_layer)

    @property
    def n_probe_points(self) -> int:
        """`hidden_states` has n_layer + 1 entries; index 0 is the embedding output."""
        return self.n_layers + 1

    @property
    def max_length(self) -> int:
        return int(self.model.config.n_positions)

    def id_to_token(self) -> dict[int, str]:
        return {i: s for s, i in self.tokenizer.get_vocab().items()}

    def decode(self, ids) -> list[str]:
        return self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=bool(
                self.config.get("clean_up_tokenization_spaces", False)
            ),
        )

    def fingerprint(self) -> dict[str, Any]:
        """Weight checksum, so two runs can be proven to use the same frozen model."""
        total = 0.0
        n = 0
        for p in self.model.parameters():
            total += float(p.detach().double().sum())
            n += p.numel()
        return {
            "n_parameters": n,
            "parameter_sum": total,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "n_probe_points": self.n_probe_points,
            "vocab_size": int(self.model.config.vocab_size),
            "repo": self.config.get("model_repo", ZINC_MODEL_REPO),
            "revision": self.config.get("model_revision", ZINC_MODEL_REVISION),
            "all_parameters_frozen": all(
                not p.requires_grad for p in self.model.parameters()
            ),
            "training_mode": bool(self.model.training),
        }


def load_zinc_generator(model_cfg: dict[str, Any]) -> ZincGPT2Generator:
    """Load the pinned revision, frozen, in eval mode.

    `AutoConfig.from_pretrained(..., revision=...)` resolves first so the revision is
    recorded from the object that was actually loaded rather than from the string that
    was asked for.
    """
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, GPT2TokenizerFast

    threads = os.environ.get("PTG_NUM_THREADS") or model_cfg.get(
        "torch_num_threads", torch.get_num_threads()
    )
    torch.set_num_threads(int(threads))

    repo = model_cfg["model_repo"]
    rev = model_cfg["model_revision"]
    cfg = AutoConfig.from_pretrained(repo, revision=rev)

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=rev,
        config=cfg,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "float32")),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = torch.device(model_cfg.get("device", "cpu"))
    model.to(device)

    tokenizer = GPT2TokenizerFast.from_pretrained(
        model_cfg["tokenizer_repo"],
        revision=model_cfg["tokenizer_revision"],
        max_len=int(cfg.n_positions),
    )
    return ZincGPT2Generator(
        model=model, tokenizer=tokenizer, device=device, config=dict(model_cfg)
    )


# ------------------------------------------------------- atom-level SMILES features

#: The standard SMILES atom-level tokenisation regex (Schwaller et al. 2019).
#: Ordered so that two-character elements and bracket atoms win over their prefixes.
_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+]|Br|Cl|B|C|N|O|S|P|F|I|b|c|n|o|s|p|@@|@|/|\\|=|#|-|\+|\(|\)|\.|%\d{2}|\d|~|:|\*|\$)"
)


def smiles_atom_tokens(text: str) -> list[str]:
    """Re-split a (possibly partial) SMILES string at atom level.

    `tokens.prefix_features` -- the trivial prefix-statistics baseline the frozen hidden
    state has to beat -- is written against an **atom-level** vocabulary: it classifies
    'Cl', '[nH]', 'c', '1', '(' and so on.  GP-MoLFormer's tokenizer is atom-level, so
    the molecular pipeline feeds it token strings directly.

    This generator's vocabulary is byte-level BPE with multi-character merges ('Cc',
    'ccc', '(='), and feeding those to `tokens.classify` would classify almost
    everything as an unrecognised structural token.  The trivial baseline would then be
    artificially weak, and "the hidden state beats surface counting" would be true for
    free -- the exact failure `tokens.py`'s own docstring says the baseline exists to
    prevent.

    So the prefix is decoded to a string and re-split here, and `prefix_features` is then
    called **unchanged**.  The baseline reads the same 21 features on both generators.
    """
    return _SMILES_TOKEN_RE.findall(text or "")


def trivial_features_from_prefix_ids(gen: ZincGPT2Generator, prefix_ids) -> np.ndarray:
    """`tokens.prefix_features` on the atom-level re-split of a decoded prefix."""
    from .tokens import prefix_features

    text = gen.tokenizer.decode(
        list(prefix_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=bool(
            gen.config.get("clean_up_tokenization_spaces", False)
        ),
    )
    return prefix_features(smiles_atom_tokens(text))
