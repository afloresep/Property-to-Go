"""The future-property heads.

Two heads are compared under an identical training recipe, so that any difference
between them is attributable to the input representation and not to tuning:

  frozen_state -- two-layer MLP on the frozen final-layer hidden state h_t
  trivial      -- two-layer MLP on cheap prefix statistics (tokens.FEATURE_NAMES)
  combined     -- both, kept only as a diagnostic on whether the two are additive

All predict a categorical distribution over property bins and are trained with
cross-entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Small two-layer MLP with input standardisation folded in."""

    def __init__(self, in_dim: int, hidden_dim: int, n_bins: int, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("scale", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )
        self.in_dim = in_dim
        self.n_bins = n_bins

    def set_standardiser(self, x: np.ndarray) -> None:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32)
            out.append(torch.softmax(self(xb), dim=-1).numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_bins))


@dataclass
class TrainResult:
    best_epoch: int
    best_val_nll: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_head(
    head: MLPHead,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
) -> TrainResult:
    """Train with AdamW, select the epoch with the best validation NLL."""
    torch.manual_seed(int(cfg["seed"]))
    head.set_standardiser(x_train)

    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    xv = torch.as_tensor(x_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.long)

    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    n = len(xt)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    best_epoch = -1
    history: list[dict[str, float]] = []
    patience = int(cfg["patience"])
    since_best = 0
    generator = torch.Generator().manual_seed(int(cfg["seed"]))

    for epoch in range(int(cfg["max_epochs"])):
        head.train()
        perm = torch.randperm(n, generator=generator)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        train_nll = total / n

        head.eval()
        with torch.no_grad():
            val_nll = float(loss_fn(head(xv), yv))
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})

        if val_nll < best_val - 1e-5:
            best_val = val_nll
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    head.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_nll=best_val, history=history)


class MarginalHead:
    """Predicts the training marginal regardless of input: the floor every head must beat."""

    def __init__(self, y_train: np.ndarray, n_bins: int):
        counts = np.bincount(y_train, minlength=n_bins).astype(np.float64)
        self.probs = (counts + 1.0) / (counts.sum() + n_bins)
        self.n_bins = n_bins

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self.probs, (len(x), 1))
