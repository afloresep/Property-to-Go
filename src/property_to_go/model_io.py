"""Loading the frozen GP-MoLFormer generator and its tokenizer.

The generator is frozen everywhere in this project: parameters are never updated,
and the only config field we set is `deterministic_eval`, which pins the released
random-feature projections instead of resampling them on every forward pass
(see configs/model.yaml for why).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class FrozenGenerator:
    model: Any
    tokenizer: Any
    device: torch.device
    config: dict[str, Any]

    @property
    def bos_id(self) -> int:
        return int(self.model.config.bos_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.model.config.eos_token_id)

    @property
    def pad_id(self) -> int:
        return int(self.model.config.pad_token_id)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def max_length(self) -> int:
        return int(self.model.config.max_position_embeddings)

    def id_to_token(self) -> dict[int, str]:
        return {i: s for s, i in self.tokenizer.get_vocab().items()}

    def decode(self, ids) -> list[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

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
            "deterministic_eval": bool(self.model.config.deterministic_eval),
        }


def load_generator(model_cfg: dict[str, Any]) -> FrozenGenerator:
    # PTG_NUM_THREADS lets concurrent runs share a machine without editing the
    # config; the effective value is recorded in every provenance.json.
    import os

    threads = os.environ.get("PTG_NUM_THREADS") or model_cfg.get(
        "torch_num_threads", torch.get_num_threads()
    )
    torch.set_num_threads(int(threads))

    cfg = AutoConfig.from_pretrained(
        model_cfg["model_repo"],
        revision=model_cfg["model_revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    cfg.deterministic_eval = bool(model_cfg["deterministic_eval"])

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_repo"],
        revision=model_cfg["model_revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        config=cfg,
        torch_dtype=getattr(torch, model_cfg.get("dtype", "float32")),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = torch.device(model_cfg.get("device", "cpu"))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["tokenizer_repo"],
        revision=model_cfg["tokenizer_revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    return FrozenGenerator(model=model, tokenizer=tokenizer, device=device, config=model_cfg)
