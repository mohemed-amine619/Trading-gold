"""
core/ai/deep_nn.py — dual-stream deep network (temporal + market-location).

Imports torch at module level, so import it lazily (``nn_model`` does).

Architecture (widths per preset):

* Temporal branch: channel dropout on the input, two stacked bidirectional
  LSTMs (``large`` = 256 -> 512 units), LayerNorm, an 8-head self-attention
  block with a residual connection, then mean + last-step pooling.
* Structural branch: residual MLP blocks (w -> 2w -> w, LayerNorm, SiLU, skip)
  over the market-location vector of the bar being decided.
* Fusion head: concat -> three widening-to-narrowing dense layers (``large``
  512 -> 256 -> 128) with dropout, then
    - 3 logits  [NOISE, LONG, SHORT]  (first-barrier classes, calibrated later
      by temperature scaling exactly as the GRU is), and
    - an expected-excursion multiplier in ATR (softplus, so it is >= 0).

Capacity is a choice, not a virtue: the ``large`` preset has millions of
parameters and a few hundred training windows will overfit it. Weight decay
(AdamW), dropout, early stopping and a plateau scheduler are all here, but the
real safeguard is unchanged: the predictor refuses any model whose measured
out-of-sample edge is missing or below its floor, and a challenger only
replaces the champion by beating it on bars neither saw.

``forward`` returns logits only so every existing helper (``logits_for``,
``ensemble_proba``, the calibrator) works on this model unchanged.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.ai.deep_layout import N_SEQ
from core.ai.nn_features import N_CLASSES


@dataclass(frozen=True)
class Preset:
    lstm1: int          # hidden units of the first LSTM (per direction)
    lstm2: int          # hidden units of the second LSTM (per direction)
    heads: int          # attention heads
    struct: int         # structural MLP base width
    fuse: tuple         # fusion head widths


PRESETS = {
    # Default: close to the capacity the 300-600 windows we actually have can support.
    "small": Preset(lstm1=16, lstm2=32, heads=4, struct=32, fuse=(64, 32, 16)),
    "medium": Preset(lstm1=64, lstm2=128, heads=8, struct=128, fuse=(256, 128, 64)),
    # The specified high-capacity network.
    "large": Preset(lstm1=256, lstm2=512, heads=8, struct=256, fuse=(512, 256, 128)),
}


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.drop(F.silu(self.up(self.norm(x)))))
        return x + self.drop(h)


class DualStreamNet(nn.Module):
    def __init__(self, n_seq: int, n_struct: int, preset: str = "small",
                 dropout: float = 0.2):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        p = PRESETS[preset]
        self.n_seq, self.n_struct = n_seq, n_struct

        # temporal branch
        self.in_drop = nn.Dropout1d(min(dropout, 0.3))      # drops whole feature channels
        self.lstm1 = nn.LSTM(n_seq, p.lstm1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * p.lstm1, p.lstm2, batch_first=True, bidirectional=True)
        t_dim = 2 * p.lstm2
        self.t_norm = nn.LayerNorm(t_dim)
        self.attn = nn.MultiheadAttention(t_dim, p.heads, dropout=dropout, batch_first=True)
        self.a_norm = nn.LayerNorm(t_dim)

        # structural branch
        self.s_in = nn.Linear(n_struct, p.struct)
        self.s_blocks = nn.Sequential(
            _ResidualBlock(p.struct, 2 * p.struct, dropout),
            _ResidualBlock(p.struct, 2 * p.struct, dropout),
        )
        self.s_norm = nn.LayerNorm(p.struct)

        # fusion head
        dims = [2 * t_dim + p.struct, *p.fuse]
        layers: list = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.fuse = nn.Sequential(*layers)
        self.cls = nn.Linear(dims[-1], N_CLASSES)
        self.exc = nn.Linear(dims[-1], 1)

    def forward_full(self, x: torch.Tensor):
        seq = x[:, :, :self.n_seq]
        struct = x[:, -1, self.n_seq:]
        # Dropout1d expects (N, C, L): channels are features.
        seq = self.in_drop(seq.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm1(seq)
        h, _ = self.lstm2(h)
        h = self.t_norm(h)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.a_norm(h + a)
        temporal = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)

        s = self.s_norm(self.s_blocks(self.s_in(struct)))
        z = self.fuse(torch.cat([temporal, s], dim=1))
        return self.cls(z), F.softplus(self.exc(z)).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_full(x)[0]


def build_dual(n_features: int, preset: str, dropout: float) -> DualStreamNet:
    """``n_features`` is the full per-step width (sequential + structural)."""
    n_struct = n_features - N_SEQ
    if n_struct <= 0:
        raise ValueError("dual architecture needs structural feature columns")
    return DualStreamNet(N_SEQ, n_struct, preset, dropout)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fit_dual(x_train, y_train, x_val, y_val, cfg, aux_train=None, aux_val=None):
    """Train with AdamW (L2 decay), a plateau LR schedule and early stopping on
    validation loss; returns the best weights. ``cfg`` is a ``TrainConfig``."""
    from core.ai.nn_model import _batches, balanced_class_weights, fp_penalised_loss

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_dual(x_train.shape[2], cfg.preset, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, min_lr=1e-5)
    weights = balanced_class_weights(y_train)

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)
    use_aux = aux_train is not None and cfg.aux_weight > 0
    at = torch.tensor(np.nan_to_num(aux_train, nan=0.0), dtype=torch.float32) if use_aux else None
    am = torch.tensor(np.isfinite(aux_train), dtype=torch.float32) if use_aux else None
    if use_aux and aux_val is not None and len(xv):
        av = torch.tensor(np.nan_to_num(aux_val, nan=0.0), dtype=torch.float32)
        avm = torch.tensor(np.isfinite(aux_val), dtype=torch.float32)
    else:
        av = avm = None

    def total_loss(logits, exc, y, a, m):
        loss = fp_penalised_loss(logits, y, weights, cfg.fp_penalty)
        if a is not None and m.sum() > 0:
            loss = loss + cfg.aux_weight * (F.smooth_l1_loss(exc, a, reduction="none") * m).sum() / m.sum()
        return loss

    best_loss, best_state, bad, history = math.inf, None, 0, []
    for _ in range(cfg.epochs):
        model.train()
        for idx in _batches(len(xt), cfg.batch, rng):
            opt.zero_grad()
            logits, exc = model.forward_full(xt[idx])
            loss = total_loss(logits, exc, yt[idx],
                              at[idx] if use_aux else None, am[idx] if use_aux else None)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            if len(xv):
                lg, ex = model.forward_full(xv)
                val_loss = float(total_loss(lg, ex, yv, av, avm))
            else:
                val_loss = math.nan
        history.append(val_loss)
        if len(xv) == 0:
            best_state = copy.deepcopy(model.state_dict())
            continue
        sched.step(val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"val_loss": history, "best_val_loss": best_loss, "epochs_run": len(history),
                   "lr_final": opt.param_groups[0]["lr"], "params": n_parameters(model)}


def excursion_for(model: DualStreamNet, x: np.ndarray) -> np.ndarray:
    """Expected excursion (ATR multiple) per window."""
    model.eval()
    with torch.no_grad():
        return model.forward_full(torch.tensor(x, dtype=torch.float32))[1].numpy()
