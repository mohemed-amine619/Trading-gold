"""
core/ai/deep_model.py — two-branch deep network, its training, calibration,
walk-forward validation and the saved artifact. Imports torch at module level,
so import it lazily (``core/ai/deep_predictor.py`` does).

What it predicts
----------------
For each bar, a calibrated probability over the triple-barrier outcome plus an
expected-excursion multiplier, from two views of the market:

* temporal branch  - a window of multi-timeframe returns, EMA slopes, RSI/MACD
  derivatives, ATR regime and session flags (``core/ai/deep_features.py``);
* structural branch - a dense vector extracted from ``core/market/location.py``:
  trend state, BOS/CHoCH recency, distance and score of the nearest S/R zone,
  proximity to liquidity pools (``core/ai/loc_features.py``).

Class order (the project's, so ``trade_metrics`` and ``should_promote`` work
unchanged): ``NOISE`` 0, ``LONG`` 1, ``SHORT`` 2. ``NOISE`` is the "chop / no
trade" class. The brief's order ``[P(long), P(short), P(chop)]`` is
``proba[:, [LONG, SHORT, NOISE]]``; ``to_brief_order`` does that reordering.

``excursion`` is the expected favourable move, in ATR multiples (>= 0, softplus),
of the barrier-hit move CONDITIONAL on a trade-class outcome: it is trained on
LONG/SHORT rows only and is meaningless for a NOISE call. Read it only when you
are acting on a LONG/SHORT signal.

Architecture (every width comes from ``DeepConfig``)
----------------------------------------------------
temporal : BiLSTM(256 = 2x128) -> channel dropout -> BiLSTM(512 = 2x256)
           -> 8-head self-attention + residual + LayerNorm
           -> GELU feed-forward + residual + LayerNorm
           -> pooling = last step || mean over time
structural: Linear -> 2 pre-LayerNorm residual blocks (256 -> 512 -> 256)
           -> LayerNorm -> Linear -> SiLU
fusion    : concat -> 3 x (Linear, LayerNorm, SiLU, Dropout) -> two heads:
           3 class logits and one softplus excursion unit.

Leakage rules (same as ``core/ai/nn_features.py``, enforced here)
----------------------------------------------------------------
* a feature row at bar ``t`` is a causal function of bars ``<= t``, so one
  dataset built over the whole frame is safe to reuse across folds;
* the scalers are fitted per fold on that fold's TRAINING windows only and then
  frozen; test rows never touch them;
* a label looks ``horizon`` bars ahead, so training windows must END more than
  ``horizon`` bars before the first validation window (purge), and the test fold
  starts ``horizon`` bars after the training fold ends (walk-forward gap).

Nothing here sizes a position, moves a stop or opens an order.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from core.ai.deep_features import (
    N_SEQ_FEATURES, SEQ_FEATURE_NAMES, build_dataset, build_sequence_features,
    make_excursion, make_windows_multi,
)
from core.ai.loc_features import LOC_FEATURE_NAMES, N_LOC_FEATURES, location_matrix
from core.ai.nn_features import (
    LONG, N_CLASSES, NOISE, SHORT, UNKNOWN, FeatureScaler, make_labels,
    walk_forward_splits,
)
from core.ai.nn_model import (
    MIN_TRADES_FOR_EDGE, balanced_class_weights, fit_temperature, fp_penalised_loss,
    softmax_np, trade_metrics,
)

logger = logging.getLogger(__name__)

ARTIFACT_KIND = "deep"
BRIEF_ORDER = (LONG, SHORT, NOISE)       # columns of [P(long), P(short), P(chop)]

FF_EXPANSION = 2          # attention feed-forward hidden width = FF_EXPANSION * temporal_dims[1]
STRUCT_BLOCKS = 2         # residual blocks in the structural branch
LR_FLOOR_FRACTION = 0.05  # the cosine schedule ends at this fraction of the peak lr
GRAD_CLIP = 1.0
MIN_FOLD_WINDOWS = 50     # a walk-forward fold with fewer training windows is skipped
MIN_FIT_WINDOWS = 100     # fit_deep_artifact refuses to train on fewer windows

_TUPLE_FIELDS = ("temporal_dims", "struct_dims", "fusion_dims")


class DeepArtifactError(ValueError):
    """The artifact cannot be used (wrong kind, missing keys, bad weights...)."""


class FeatureMismatchError(DeepArtifactError):
    """The artifact was trained on different feature columns than the current code
    produces. Feeding it anyway would be a silent column mismatch."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepConfig:
    # data / labels
    window: int = 48
    horizon: int = 24
    barrier_mult: float = 1.5
    stride: int = 2
    loc_lookback: int = 200
    # architecture
    temporal_dims: tuple = (256, 512)       # BiLSTM total widths (half per direction)
    heads: int = 8                          # attention heads; must divide temporal_dims[1]
    struct_dims: tuple = (256, 512, 256)    # (width, hidden of the residual blocks, output)
    fusion_dims: tuple = (512, 256, 128)
    dropout: float = 0.2
    spatial_dropout: float = 0.1            # whole-channel dropout between the two LSTMs
    # optimisation
    epochs: int = 40
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2              # AdamW decoupled decay (the L2 regulariser)
    warmup_epochs: int = 2
    fp_penalty: float = 2.0
    exc_weight: float = 0.2
    val_fraction: float = 0.2
    patience: int = 6
    seed: int = 7
    threshold: float = 0.72
    n_models: int = 1                       # seeds averaged; more = lower variance

    def __post_init__(self) -> None:
        # torch.save / JSON may hand tuples back as lists.
        for name in _TUPLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))


def config_from_dict(d: Mapping, *, require_all: bool = False) -> DeepConfig:
    """Rebuild a ``DeepConfig`` from a plain dict (an artifact's ``config``).

    Lists are turned back into tuples. An UNKNOWN key is refused rather than
    ignored. With ``require_all`` a MISSING key is refused too instead of being
    filled with the dataclass default: ``deep_members_from_artifact`` uses that,
    because a defaulted ``window`` or ``loc_lookback`` would silently feed the
    model the wrong inputs.
    """
    names = {f.name for f in fields(DeepConfig)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise DeepArtifactError(f"unknown config keys: {unknown}")
    if require_all:
        missing = sorted(names - set(d))
        if missing:
            raise DeepArtifactError(f"config is missing keys: {missing}")
    return DeepConfig(**{k: d[k] for k in d})


def _dims(name: str, value, length: Optional[int]) -> tuple:
    ok = isinstance(value, (tuple, list)) and all(
        isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0 for v in value)
    if not ok or (length is not None and len(value) != length) or (length is None and not value):
        want = f"{length} positive integers" if length else "at least one positive integer"
        raise ValueError(f"{name} must be {want}, got {value!r}")
    return tuple(int(v) for v in value)


def validate_config(cfg: DeepConfig) -> None:
    """Refuse an architecture that cannot be built, with a message saying why."""
    td = _dims("temporal_dims", cfg.temporal_dims, 2)
    if any(d % 2 for d in td):
        raise ValueError(
            f"temporal_dims must be even (each bidirectional LSTM splits its width "
            f"between the two directions), got {td}")
    if not isinstance(cfg.heads, (int, np.integer)) or isinstance(cfg.heads, bool) \
            or cfg.heads < 1 or td[1] % cfg.heads:
        raise ValueError(
            f"heads={cfg.heads!r} must be a positive divisor of temporal_dims[1]={td[1]} "
            f"(the attention width)")
    _dims("struct_dims", cfg.struct_dims, 3)
    _dims("fusion_dims", cfg.fusion_dims, None)
    for name in ("dropout", "spatial_dropout"):
        value = getattr(cfg, name)
        if not (isinstance(value, (int, float)) and 0.0 <= value < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {value!r}")


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

class _TemporalBranch(nn.Module):
    """BiLSTM -> channel dropout -> BiLSTM -> self-attention -> feed-forward -> pool."""

    def __init__(self, n_seq: int, dims: tuple, heads: int, dropout: float,
                 spatial_dropout: float):
        super().__init__()
        d1, d2 = dims
        self.lstm1 = nn.LSTM(n_seq, d1 // 2, batch_first=True, bidirectional=True)
        self.channel_drop = nn.Dropout1d(spatial_dropout)
        self.lstm2 = nn.LSTM(d1, d2 // 2, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(d2, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d2)
        self.ff = nn.Sequential(nn.Linear(d2, FF_EXPANSION * d2), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(FF_EXPANSION * d2, d2))
        self.ff_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d2)
        self.out_dim = 2 * d2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm1(x)                                          # [B, T, d1]
        # Dropout1d zeroes whole channels, so channels go second: [B, d1, T].
        h = self.channel_drop(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm2(h)                                          # [B, T, d2]
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + self.attn_drop(a))
        h = self.norm2(h + self.ff_drop(self.ff(h)))
        return torch.cat([h[:, -1, :], h.mean(dim=1)], dim=1)         # [B, 2 * d2]


class _ResBlock(nn.Module):
    """x + Linear(hidden -> width)(Dropout(SiLU(Linear(width -> hidden)(LayerNorm(x)))))."""

    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(self.drop(self.act(self.up(self.norm(x)))))


class _StructuralBranch(nn.Module):
    """Wide residual MLP over the market-location vector."""

    def __init__(self, n_loc: int, dims: tuple, dropout: float):
        super().__init__()
        width, hidden, out = dims
        self.proj = nn.Linear(n_loc, width)
        self.blocks = nn.ModuleList([_ResBlock(width, hidden, dropout)
                                     for _ in range(STRUCT_BLOCKS)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out)
        self.act = nn.SiLU()
        self.out_dim = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return self.act(self.out(self.norm(h)))


class DeepSignalNet(nn.Module):
    """``forward(x_seq[B, T, F], x_loc[B, L]) -> (class_logits[B, 3], excursion[B])``.

    Class logits are ordered ``[NOISE, LONG, SHORT]``. ``excursion`` is in ATR
    multiples and is >= 0 (softplus).
    """

    def __init__(self, n_seq: int, n_loc: int, cfg: DeepConfig):
        super().__init__()
        validate_config(cfg)
        if n_seq < 1 or n_loc < 1:
            raise ValueError(f"need at least one sequence and one location feature, "
                             f"got n_seq={n_seq}, n_loc={n_loc}")
        self.n_seq, self.n_loc = int(n_seq), int(n_loc)
        self.temporal = _TemporalBranch(self.n_seq, cfg.temporal_dims, cfg.heads, cfg.dropout,
                                        cfg.spatial_dropout)
        self.structural = _StructuralBranch(self.n_loc, cfg.struct_dims, cfg.dropout)
        layers: list = []
        width = self.temporal.out_dim + self.structural.out_dim
        for d in cfg.fusion_dims:
            layers += [nn.Linear(width, d), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(cfg.dropout)]
            width = d
        self.fusion = nn.Sequential(*layers)
        self.cls_head = nn.Linear(width, N_CLASSES)
        self.exc_head = nn.Linear(width, 1)

    def forward(self, x_seq: torch.Tensor, x_loc: torch.Tensor):
        if x_seq.dim() != 3 or x_seq.shape[-1] != self.n_seq:
            raise ValueError(f"x_seq must be [batch, time, {self.n_seq}], got {tuple(x_seq.shape)}")
        if x_loc.dim() != 2 or x_loc.shape[-1] != self.n_loc or x_loc.shape[0] != x_seq.shape[0]:
            raise ValueError(f"x_loc must be [{x_seq.shape[0]}, {self.n_loc}], "
                             f"got {tuple(x_loc.shape)}")
        z = self.fusion(torch.cat([self.temporal(x_seq), self.structural(x_loc)], dim=1))
        return self.cls_head(z), F.softplus(self.exc_head(z)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters (every parameter here is trainable)."""
    return int(sum(p.numel() for p in model.parameters()))


def to_brief_order(proba: np.ndarray) -> np.ndarray:
    """Columns ``[NOISE, LONG, SHORT]`` -> the brief's ``[P(long), P(short), P(chop)]``."""
    return np.asarray(proba)[..., list(BRIEF_ORDER)]


# ---------------------------------------------------------------------------
# loss and learning-rate schedule
# ---------------------------------------------------------------------------

def deep_loss(logits: torch.Tensor, excursion: torch.Tensor, y: torch.Tensor,
              exc_target: torch.Tensor, class_weights: torch.Tensor, fp_penalty: float,
              exc_weight: float) -> torch.Tensor:
    """False-positive-penalised class loss + ``exc_weight`` x SmoothL1 on excursion.

    The excursion term is averaged over TRADE rows only: for a NOISE row there is
    no direction, so no excursion exists and its target must not teach anything.
    """
    cls = fp_penalised_loss(logits, y, class_weights, fp_penalty)
    if exc_weight <= 0:
        return cls
    trade = y != NOISE
    if not bool(trade.any()):
        return cls
    return cls + exc_weight * F.smooth_l1_loss(excursion[trade], exc_target[trade])


def _lr_at(epoch: int, cfg: DeepConfig) -> float:
    """Learning rate for a 0-based ``epoch``: linear warm-up, then cosine decay.

    Epochs ``0 .. warmup-1`` rise as ``lr * (e + 1) / (warmup + 1)``; epoch
    ``warmup`` is the peak ``lr``; the cosine then reaches ``LR_FLOOR_FRACTION *
    lr`` exactly at the last epoch (``cfg.epochs - 1``) and never goes below it.
    Early stopping may end training before the schedule completes.
    """
    base = float(cfg.lr)
    floor = LR_FLOOR_FRACTION * base
    warm = max(0, int(cfg.warmup_epochs))
    epoch = max(0, int(epoch))
    if epoch < warm:
        return base * (epoch + 1) / (warm + 1)
    span = int(cfg.epochs) - 1 - warm
    progress = 0.0 if span <= 0 else min(1.0, (epoch - warm) / span)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _arrays(data: Optional[Mapping], what: str):
    """Validate one split ``{"seq","loc","y","exc"}`` (already SCALED) -> float32/int64 arrays."""
    if data is None:
        return None
    seq = np.asarray(data["seq"], dtype=np.float32)
    loc = np.asarray(data["loc"], dtype=np.float32)
    y = np.asarray(data["y"])
    exc = np.asarray(data["exc"], dtype=np.float32)
    if seq.ndim != 3 or loc.ndim != 2:
        raise ValueError(f"{what}: seq must be [N, T, F] and loc [N, L]")
    if not (len(seq) == len(loc) == len(y) == len(exc)):
        raise ValueError(f"{what}: seq, loc, y and exc must have the same length")
    if len(y) and (y.min() < 0 or y.max() >= N_CLASSES):
        raise ValueError(f"{what}: labels must be in 0..{N_CLASSES - 1} "
                         f"(UNKNOWN rows must be dropped before training)")
    for name, arr in (("seq", seq), ("loc", loc), ("exc", exc)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{what}: {name} contains non-finite values")
    return seq, loc, y.astype(np.int64), exc


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _make_optimizer(model: nn.Module, cfg: DeepConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay on weight matrices only (not biases or LayerNorm gains)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)


def _init_excursion_bias(model: DeepSignalNet, y: np.ndarray, exc: np.ndarray) -> None:
    """Start the excursion head at the TRAINING mean (inverse softplus), so the first
    epochs do not spend their budget moving a bias."""
    trade = y != NOISE
    if not trade.any():
        return
    mean = float(exc[trade].mean())
    if math.isfinite(mean) and mean > 1e-3:
        with torch.no_grad():
            model.exc_head.bias.fill_(math.log(math.expm1(mean)))


def _check_fit_config(cfg: DeepConfig) -> None:
    if cfg.epochs < 1 or cfg.batch < 1 or cfg.patience < 1 or cfg.warmup_epochs < 0:
        raise ValueError("epochs, batch and patience must be >= 1 and warmup_epochs >= 0")
    if not cfg.lr > 0 or cfg.weight_decay < 0 or cfg.exc_weight < 0 or cfg.fp_penalty < 0:
        raise ValueError("lr must be > 0 and weight_decay, exc_weight, fp_penalty >= 0")


def _forward_all(model: DeepSignalNet, seq: torch.Tensor, loc: torch.Tensor, chunk: int):
    """Eval-mode forward over many rows in chunks -> (logits, excursion) tensors.

    The loss is computed ONCE on the concatenated outputs, not per chunk: the
    class-weighted cross-entropy normalises by the batch's weights, so a mean of
    chunk losses would not equal the loss of the whole set.
    """
    model.eval()
    logits, exc = [], []
    with torch.no_grad():
        for i in range(0, len(seq), chunk):
            lg, ex = model(seq[i:i + chunk], loc[i:i + chunk])
            logits.append(lg)
            exc.append(ex)
    return torch.cat(logits), torch.cat(exc)


def fit_deep_model(tr: Mapping, va: Optional[Mapping], cfg: DeepConfig):
    """Train one network; returns ``(model, info)`` with the model in eval mode.

    ``tr`` and ``va`` are mappings with already-SCALED float arrays ``seq[N,T,F]``,
    ``loc[N,L]``, class labels ``y[N]`` in ``{NOISE, LONG, SHORT}`` and the
    excursion target ``exc[N]``. ``va`` may be ``None`` or empty.

    * AdamW (decoupled weight decay), warm-up then cosine learning rate
      (``_lr_at``; the lr used each epoch is recorded in ``info["lr"]``),
      gradient clipping at 1.0.
    * Loss: ``deep_loss`` (false-positive-penalised, class-balanced cross-entropy
      + SmoothL1 excursion on trade rows).
    * Early stopping on the validation loss with ``cfg.patience``; the weights of
      the best epoch are restored. With NO validation rows there is nothing to
      stop on: all ``cfg.epochs`` run and the LAST weights are kept
      (``info["best_val_loss"]`` is then ``None``, never 0).
    * Deterministic for a given ``cfg.seed`` and thread count.
    """
    _check_fit_config(cfg)
    train = _arrays(tr, "training split")
    if train is None or len(train[2]) == 0:
        raise ValueError("no training windows")
    x_seq, x_loc, y, exc = train
    val = _arrays(va, "validation split")
    has_val = val is not None and len(val[2]) > 0
    if has_val and (val[0].shape[1:] != x_seq.shape[1:] or val[1].shape[1:] != x_loc.shape[1:]):
        raise ValueError("training and validation splits have different feature shapes")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = DeepSignalNet(x_seq.shape[2], x_loc.shape[1], cfg)
    _init_excursion_bias(model, y, exc)
    opt = _make_optimizer(model, cfg)
    weights = balanced_class_weights(y)

    t_seq, t_loc = torch.from_numpy(x_seq), torch.from_numpy(x_loc)
    t_y, t_exc = torch.from_numpy(y), torch.from_numpy(exc)
    if has_val:
        v_seq, v_loc = torch.from_numpy(val[0]), torch.from_numpy(val[1])
        v_y, v_exc = torch.from_numpy(val[2]), torch.from_numpy(val[3])

    train_hist, val_hist, lr_hist = [], [], []
    best_loss, best_state, best_epoch, bad = math.inf, None, -1, 0
    for epoch in range(cfg.epochs):
        lr = _lr_at(epoch, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        lr_hist.append(opt.param_groups[0]["lr"])

        model.train()
        total, seen = 0.0, 0
        order = rng.permutation(len(t_y))
        for i in range(0, len(order), cfg.batch):
            idx = torch.from_numpy(order[i:i + cfg.batch])
            opt.zero_grad(set_to_none=True)
            logits, ex = model(t_seq[idx], t_loc[idx])
            loss = deep_loss(logits, ex, t_y[idx], t_exc[idx], weights, cfg.fp_penalty,
                             cfg.exc_weight)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}; the run "
                                   f"diverged and its weights are not usable")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += value * len(idx)
            seen += len(idx)
        train_hist.append(total / max(seen, 1))

        if not has_val:
            val_hist.append(math.nan)
            continue
        lg, ex = _forward_all(model, v_seq, v_loc, chunk=max(cfg.batch, 256))
        val_loss = float(deep_loss(lg, ex, v_y, v_exc, weights, cfg.fp_penalty, cfg.exc_weight))
        if not math.isfinite(val_loss):
            raise RuntimeError(f"non-finite validation loss at epoch {epoch}")
        val_hist.append(val_loss)
        logger.debug("deep epoch %d lr=%.2e train=%.4f val=%.4f", epoch, lr, train_hist[-1],
                     val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, bad = val_loss, _snapshot(model), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    epochs_run = len(train_hist)
    info = {
        "train_loss": train_hist, "val_loss": val_hist, "lr": lr_hist,
        "best_val_loss": best_loss if has_val else None,
        "best_epoch": best_epoch if has_val else epochs_run - 1,
        "epochs_run": epochs_run, "stopped_early": epochs_run < cfg.epochs,
        "n_train": int(len(y)), "n_val": int(len(val[2])) if has_val else 0,
        "n_params": count_parameters(model),
    }
    return model, info


# ---------------------------------------------------------------------------
# inference and calibration
# ---------------------------------------------------------------------------

def deep_logits(model: DeepSignalNet, seq: np.ndarray, loc: np.ndarray, chunk: int = 256):
    """Raw ``(class_logits[N, 3], excursion[N])`` of one model (eval mode, no grad).

    ``seq`` and ``loc`` must already be scaled. Rows are processed in chunks so a
    large evaluation set does not need one huge activation tensor.
    """
    seq = np.asarray(seq, dtype=np.float32)
    loc = np.asarray(loc, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((0, N_CLASSES), dtype=np.float32), np.zeros(0, dtype=np.float32)
    logits, exc = _forward_all(model, torch.from_numpy(seq), torch.from_numpy(loc), chunk)
    return logits.numpy(), exc.numpy()


def deep_predict(members: list, seq: np.ndarray, loc: np.ndarray):
    """Ensemble ``(proba[N, 3], excursion[N])``: the mean of the members' calibrated
    class probabilities (rows sum to 1) and of their excursions. Class order is
    ``[NOISE, LONG, SHORT]``; ``to_brief_order`` reorders it."""
    if not members:
        raise ValueError("no ensemble members")
    probs, excs = [], []
    for m in members:
        logits, exc = deep_logits(m["model"], seq, loc)
        probs.append(softmax_np(logits, m["temperature"]))
        excs.append(np.asarray(exc, dtype=float))
    return np.mean(probs, axis=0), np.mean(excs, axis=0)


def fit_deep_ensemble(tr: Mapping, va: Optional[Mapping], cfg: DeepConfig) -> list:
    """Train ``cfg.n_models`` networks that differ only by seed, each calibrated by
    temperature scaling on the validation rows (``T = 1`` when there are none).
    Averaging lowers the variance of one unlucky initialisation; it does not create
    information the data does not hold."""
    members = []
    has_val = va is not None and len(va["y"]) > 0
    for i in range(max(1, cfg.n_models)):
        model, info = fit_deep_model(tr, va, replace(cfg, seed=cfg.seed + 101 * i))
        temp = 1.0
        if has_val:
            logits, _ = deep_logits(model, va["seq"], va["loc"])
            temp = fit_temperature(logits, np.asarray(va["y"]))
        members.append({"model": model, "temperature": temp, "info": info})
    return members


# ---------------------------------------------------------------------------
# dataset, splits and scaling
# ---------------------------------------------------------------------------

def make_deep_dataset(df: pd.DataFrame, cfg: DeepConfig) -> dict:
    """``build_dataset`` with this config's window / horizon / barrier / stride / lookback."""
    return build_dataset(df, cfg.window, cfg.horizon, cfg.barrier_mult, cfg.stride,
                         cfg.loc_lookback)


def _check_dataset(ds: Mapping, cfg: DeepConfig, n_bars: Optional[int] = None) -> None:
    seq, loc = ds["seq"], ds["loc"]
    if seq.ndim != 3 or seq.shape[1] != cfg.window or seq.shape[2] != N_SEQ_FEATURES \
            or loc.ndim != 2 or loc.shape[1] != N_LOC_FEATURES:
        raise ValueError(
            f"dataset shapes seq{tuple(seq.shape)} loc{tuple(loc.shape)} do not match "
            f"window={cfg.window} and the current feature sets ({N_SEQ_FEATURES} / "
            f"{N_LOC_FEATURES}); build it with make_deep_dataset(df, cfg)")
    if n_bars is not None and len(ds["seq_rows"]) != n_bars:
        raise ValueError(f"dataset covers {len(ds['seq_rows'])} bars but df has {n_bars}")


def split_train_val(ends: np.ndarray, cfg: DeepConfig):
    """Split windows (ascending ``ends``) into ``(fit_pos, val_pos)`` positions.

    Validation is the newest ``val_fraction``. Training keeps only windows that END
    more than ``cfg.horizon`` bars before the first validation window, so no
    training label (which looks ``horizon`` bars ahead) reaches a validation bar.
    ``val_fraction = 0`` means no validation: everything trains.
    """
    ends = np.asarray(ends)
    n = len(ends)
    if n and not (np.diff(ends) > 0).all():
        raise ValueError("window ends must be strictly increasing")
    if cfg.val_fraction <= 0 or n == 0:
        return np.arange(n), np.zeros(0, dtype=int)
    cut = min(max(int(n * (1.0 - cfg.val_fraction)), 0), n - 1)
    fit_pos = np.flatnonzero(ends < ends[cut] - cfg.horizon)
    if len(fit_pos) == 0:
        raise ValueError("no training windows are left after the purge between training "
                         "and validation; use more bars or a smaller horizon")
    return fit_pos, np.arange(cut, n)


@dataclass
class FoldData:
    """Row indices (into the dataset) and the scalers of one walk-forward fold."""
    tr_idx: np.ndarray      # every window ending inside the training fold
    fit_idx: np.ndarray     # tr_idx minus the validation windows and the purge gap
    val_idx: np.ndarray     # the newest val_fraction of tr_idx
    te_idx: np.ndarray      # windows ending inside the test fold
    seq_scaler: FeatureScaler
    loc_scaler: FeatureScaler


def prepare_fold(ds: Mapping, cfg: DeepConfig, tr: slice, te: slice) -> Optional[FoldData]:
    """Select the rows of one fold and fit its scalers on the TRAINING windows only.

    ``None`` when the fold has too few training windows or no test window.
    """
    ends = ds["ends"]
    tr_idx = np.flatnonzero((ends >= tr.start) & (ends < tr.stop))
    te_idx = np.flatnonzero((ends >= te.start) & (ends < te.stop))
    if len(tr_idx) < MIN_FOLD_WINDOWS or len(te_idx) == 0:
        return None
    fit_pos, val_pos = split_train_val(ends[tr_idx], cfg)
    # Statistics from the training windows only; test rows are not even read here.
    seq_scaler = FeatureScaler().fit(ds["seq"][tr_idx].reshape(-1, ds["seq"].shape[2]))
    loc_scaler = FeatureScaler().fit(ds["loc"][tr_idx])
    return FoldData(tr_idx, tr_idx[fit_pos], tr_idx[val_pos], te_idx, seq_scaler, loc_scaler)


def scaled_rows(ds: Mapping, idx: np.ndarray, seq_scaler: FeatureScaler,
                loc_scaler: FeatureScaler) -> dict:
    """The rows ``idx`` of ``ds`` scaled with frozen scalers, as float32 arrays."""
    return {"seq": seq_scaler.transform(ds["seq"][idx]).astype(np.float32),
            "loc": loc_scaler.transform(ds["loc"][idx]).astype(np.float32),
            "y": ds["y"][idx], "exc": ds["exc"][idx]}


def _exc_mae(pred: np.ndarray, target: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mean absolute excursion error over TRADE rows (``None`` if there are none)."""
    trade = y != NOISE
    return float(np.abs(pred[trade] - target[trade]).mean()) if trade.any() else None


# ---------------------------------------------------------------------------
# walk-forward validation
# ---------------------------------------------------------------------------

def deep_walk_forward_evaluate(df: Optional[pd.DataFrame], cfg: DeepConfig, train_bars: int,
                               test_bars: int, step: Optional[int] = None,
                               dataset: Optional[Mapping] = None) -> dict:
    """Rolling-origin out-of-sample test; same result shape as ``walk_forward_evaluate``.

    Per fold: the scalers and the network are fitted on the training windows only
    (``prepare_fold``), validation is the newest ``val_fraction`` of those with a
    ``horizon`` purge before it, and the metrics are taken on the test windows,
    which start ``horizon`` bars after the training fold ends. Folds are pooled
    before the pooled metrics are computed. ``edge`` is ``None`` (unmeasured, never
    zero) below ``MIN_TRADES_FOR_EDGE`` trades.

    The dataset is built ONCE over the whole frame. That is safe because every
    input row is causal: the sequence features are rolling / EWM functions of
    bars ``<= t`` and the location vector of window-end ``e`` is computed from
    ``df.iloc[:e + 1]`` only. Only the TARGETS look ahead, and the purge and the
    walk-forward gap keep them out of validation and test respectively
    (tests/unit/test_deep_model.py proves both). ``df`` is only needed to build
    the dataset; pass ``dataset`` (from ``make_deep_dataset(df, cfg)``) to reuse
    one, in which case ``df`` may be ``None``.
    """
    if dataset is None:
        if df is None:
            raise ValueError("need df or a prebuilt dataset")
        ds = make_deep_dataset(df, cfg)
    else:
        ds = dataset
    n_bars = len(ds["seq_rows"])
    _check_dataset(ds, cfg, None if df is None else len(df))

    folds, pooled_p, pooled_y, pooled_exc, pooled_exc_y = [], [], [], [], []
    for tr, te in walk_forward_splits(n_bars, train_bars, test_bars, cfg.horizon, step):
        fold = prepare_fold(ds, cfg, tr, te)
        if fold is None:
            continue
        members = fit_deep_ensemble(
            scaled_rows(ds, fold.fit_idx, fold.seq_scaler, fold.loc_scaler),
            scaled_rows(ds, fold.val_idx, fold.seq_scaler, fold.loc_scaler), cfg)
        test = scaled_rows(ds, fold.te_idx, fold.seq_scaler, fold.loc_scaler)
        proba, exc_pred = deep_predict(members, test["seq"], test["loc"])
        pooled_p.append(proba)
        pooled_y.append(test["y"])
        pooled_exc.append(exc_pred)
        pooled_exc_y.append(test["exc"])
        fit_trade = ds["y"][fold.fit_idx] != NOISE
        base = float(ds["exc"][fold.fit_idx][fit_trade].mean()) if fit_trade.any() else None
        folds.append({
            "train": (tr.start, tr.stop), "test": (te.start, te.stop),
            "train_windows": int(len(fold.fit_idx)), "val_windows": int(len(fold.val_idx)),
            "test_windows": int(len(fold.te_idx)),
            "epochs_run": [m["info"]["epochs_run"] for m in members],
            "temperatures": [m["temperature"] for m in members],
            "exc_mae": _exc_mae(exc_pred, test["exc"], test["y"]),
            "exc_mae_const": None if base is None else _exc_mae(
                np.full(len(test["y"]), base), test["exc"], test["y"]),
            **trade_metrics(proba, test["y"], cfg.threshold),
        })
        logger.info("deep walk-forward fold %d: %s", len(folds), {
            k: folds[-1][k] for k in ("train", "test", "trades", "precision", "edge")})

    if not folds:
        return {"folds": [], "trades": 0, "coverage": 0.0, "precision": None, "edge": None,
                "windows": 0, "exc_mae": None}
    p, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    return {"folds": folds, "windows": int(len(yy)),
            "exc_mae": _exc_mae(np.concatenate(pooled_exc), np.concatenate(pooled_exc_y), yy),
            **trade_metrics(p, yy, cfg.threshold)}


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def _plain(obj):
    """Recursively reduce numpy scalars/arrays to plain Python so the file loads
    under ``torch.load(weights_only=True)``. Tensors are kept (on CPU)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {_plain(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_plain(v) for v in obj)
    return obj


def _make_artifact(members: list, seq_scaler: FeatureScaler, loc_scaler: FeatureScaler,
                   cfg: DeepConfig, df: pd.DataFrame) -> dict:
    return {
        "kind": ARTIFACT_KIND,
        "members": [{"state_dict": {k: v.detach().cpu().clone()
                                    for k, v in m["model"].state_dict().items()},
                     "temperature": float(m["temperature"]),
                     "epochs_run": int(m["info"]["epochs_run"])} for m in members],
        "seq_scaler": seq_scaler.to_dict(),
        "loc_scaler": loc_scaler.to_dict(),
        "config": asdict(cfg),
        "seq_feature_names": list(SEQ_FEATURE_NAMES),
        "loc_feature_names": list(LOC_FEATURE_NAMES),
        "oos": {},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_through": str(df.index[-1]),
        "n_bars": int(len(df)),
    }


def fit_deep_artifact(df: pd.DataFrame, cfg: DeepConfig,
                      dataset: Optional[Mapping] = None) -> dict:
    """Fit the model that would go live on ``df``, with NO out-of-sample test.

    ``oos`` is left empty on purpose: the caller must measure it on bars the model
    never saw (``evaluate_deep_artifact`` or ``train_deep_final``). ``trained_through``
    is the last bar of ``df``, so a later evaluation can start strictly after it.
    The scalers are fitted on all windows used for training and validation.
    """
    ds = make_deep_dataset(df, cfg) if dataset is None else dataset
    _check_dataset(ds, cfg, len(df))
    n = len(ds["y"])
    if n < MIN_FIT_WINDOWS:
        raise ValueError(f"not enough training windows ({n}, need {MIN_FIT_WINDOWS})")
    seq_scaler = FeatureScaler().fit(ds["seq"].reshape(-1, ds["seq"].shape[2]))
    loc_scaler = FeatureScaler().fit(ds["loc"])
    fit_pos, val_pos = split_train_val(ds["ends"], cfg)
    everything = np.arange(n)
    members = fit_deep_ensemble(
        scaled_rows(ds, everything[fit_pos], seq_scaler, loc_scaler),
        scaled_rows(ds, everything[val_pos], seq_scaler, loc_scaler), cfg)
    return _make_artifact(members, seq_scaler, loc_scaler, cfg, df)


def train_deep_final(df: pd.DataFrame, cfg: DeepConfig, train_bars: int,
                     test_bars: int) -> dict:
    """Measure the out-of-sample edge by walk-forward, then fit the model for use.

    The artifact carries the walk-forward result in ``oos`` (without the per-fold
    list, as ``train_final`` does). The predictor refuses a model whose edge is
    unmeasured or below its floor.
    """
    ds = make_deep_dataset(df, cfg)                 # built once, shared by both steps
    oos = deep_walk_forward_evaluate(df, cfg, train_bars, test_bars, dataset=ds)
    art = fit_deep_artifact(df, cfg, dataset=ds)
    art["oos"] = {k: v for k, v in oos.items() if k != "folds"} | {"n_folds": len(oos["folds"])}
    return art


def _require(art: Mapping, key: str):
    if key not in art:
        raise DeepArtifactError(f"artifact has no {key!r}")
    return art[key]


def _name_diff(label: str, got: list, want: tuple) -> str:
    missing, extra = [n for n in want if n not in got], [n for n in got if n not in want]
    detail = f"missing {missing}, unexpected {extra}" if (missing or extra) else "same names, different order"
    return (f"{label} feature names differ from the current ones ({len(got)} in the artifact, "
            f"{len(want)} now): {detail}")


def check_artifact_features(art: Mapping) -> None:
    """Refuse an artifact trained on different feature columns than the current code
    builds. A silent column mismatch would feed the model the wrong inputs."""
    for label, key, want in (("sequence", "seq_feature_names", SEQ_FEATURE_NAMES),
                             ("location", "loc_feature_names", LOC_FEATURE_NAMES)):
        got = list(_require(art, key))
        if got != list(want):
            raise FeatureMismatchError(_name_diff(label, got, want))


def _scaler_from(d: Mapping, n: int, label: str) -> FeatureScaler:
    s = FeatureScaler.from_dict(d)
    if s.mean.shape != (n,) or s.std.shape != (n,):
        raise DeepArtifactError(f"{label} scaler has shape {s.mean.shape}, expected ({n},)")
    if not (np.isfinite(s.mean).all() and np.isfinite(s.std).all() and (s.std > 0).all()):
        raise DeepArtifactError(f"{label} scaler holds non-finite or non-positive statistics")
    return s


def scalers_from_artifact(art: Mapping):
    """``(seq_scaler, loc_scaler)`` rebuilt from an artifact and shape-checked."""
    return (_scaler_from(_require(art, "seq_scaler"), N_SEQ_FEATURES, "sequence"),
            _scaler_from(_require(art, "loc_scaler"), N_LOC_FEATURES, "location"))


def deep_members_from_artifact(art: Mapping) -> list:
    """Rebuild the ensemble members (networks in eval mode + temperatures).

    Raises ``FeatureMismatchError`` when the artifact's feature names differ from
    the current ``SEQ_FEATURE_NAMES`` / ``LOC_FEATURE_NAMES``, and
    ``DeepArtifactError`` for any other unusable content.
    """
    if _require(art, "kind") != ARTIFACT_KIND:
        raise DeepArtifactError(f"not a deep artifact (kind={art.get('kind')!r})")
    check_artifact_features(art)
    cfg = config_from_dict(_require(art, "config"), require_all=True)
    raw = _require(art, "members")
    if not raw:
        raise DeepArtifactError("artifact has no members")
    members = []
    for i, m in enumerate(raw):
        temp = float(_require(m, "temperature"))
        if not (math.isfinite(temp) and temp > 0):
            raise DeepArtifactError(f"member {i} has an invalid temperature {temp!r}")
        model = DeepSignalNet(len(art["seq_feature_names"]), len(art["loc_feature_names"]), cfg)
        try:
            model.load_state_dict(_require(m, "state_dict"))
        except RuntimeError as exc:
            raise DeepArtifactError(f"member {i} weights do not fit the architecture: "
                                    f"{str(exc)[:200]}") from exc
        model.eval()
        members.append({"model": model, "temperature": temp})
    return members


def evaluate_deep_artifact(art: Mapping, df: pd.DataFrame, start_row: int,
                           threshold: Optional[float] = None,
                           min_trades: int = MIN_TRADES_FOR_EDGE,
                           stride: Optional[int] = None) -> dict:
    """Score a saved model on the windows of ``df`` that END at or after ``start_row``.
    Rows before it are used only as history for the features.

    This is how a champion and a challenger are compared fairly: the same
    function on the same bars, bars neither was trained on. The evaluation grid is
    ``start_row, start_row + stride, ...`` (``stride`` defaults to the artifact's
    own), so two artifacts with the same stride are scored on identical bars
    whatever their window length. Only the rows that are scored get a location
    vector, so a short holdout at the end of a long frame stays cheap.
    """
    members = deep_members_from_artifact(art)
    seq_scaler, loc_scaler = scalers_from_artifact(art)
    cfg = config_from_dict(art["config"], require_all=True)
    step = max(1, int(stride if stride is not None else cfg.stride))

    seq = build_sequence_features(df).to_numpy()
    labels = make_labels(df, cfg.barrier_mult, cfg.horizon)
    exc = make_excursion(df, labels, cfg.horizon)
    finite = np.isfinite(seq).all(axis=1)
    first = max(0, int(start_row))
    cand = [e for e in range(first, len(df), step)
            if e >= cfg.window - 1 and labels[e] != UNKNOWN
            and finite[e - cfg.window + 1:e + 1].all()]
    empty = {"windows": 0, "trades": 0, "coverage": 0.0, "precision": None, "edge": None}
    if not cand:
        return empty
    loc = np.zeros((len(df), N_LOC_FEATURES), dtype=np.float64)
    loc[np.asarray(cand)] = location_matrix(df, cand, cfg.loc_lookback)
    # Same row builder as training (windows of ``seq`` ending at the candidates).
    keep = np.full(len(df), UNKNOWN)
    keep[np.asarray(cand)] = labels[np.asarray(cand)]
    x_seq, x_loc, y, _, _ = make_windows_multi(seq, loc, keep, exc, cfg.window, 1)
    if len(y) == 0:
        return empty
    proba, _ = deep_predict(members, seq_scaler.transform(x_seq).astype(np.float32),
                            loc_scaler.transform(x_loc).astype(np.float32))
    return {"windows": int(len(y)),
            **trade_metrics(proba, y, threshold if threshold is not None else cfg.threshold,
                            min_trades)}


def save_deep_artifact(path: str, artifact: Mapping) -> None:
    """Write the artifact. Numpy values are reduced to plain Python first so the file
    can be read back with ``weights_only=True``."""
    torch.save(_plain(dict(artifact)), path)


def load_deep_artifact(path: str) -> dict:
    """Load with ``weights_only=True``: the file may hold only tensors and plain
    Python types, so a tampered file cannot execute code."""
    return torch.load(path, map_location="cpu", weights_only=True)
