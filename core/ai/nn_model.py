"""
core/ai/nn_model.py — GRU sequence classifier, training and walk-forward
validation. Imports torch at module level, so import it lazily
(``core/ai/nn_predictor.py`` does).

What it predicts: for each bar, a calibrated probability over
[NOISE, LONG, SHORT] from the last ``window`` candles, where the classes are
the triple-barrier outcomes from ``core/ai/nn_features.py``.

Two design points worth knowing:

* False-positive penalty. Class weights only make a MISSED class expensive. To
  make a wrong TRADE expensive, the loss adds ``fp_penalty`` times the
  probability the model put on a trade class that turned out not to happen.
  It trades some recall for precision on purpose.
* Honest edge. ``trade_metrics`` scores the trades taken at the confidence
  threshold in R-multiples: +1 when the called barrier was hit first, -1 when
  the opposite barrier was, 0 when neither was (the trade times out flat).
  ``edge`` is the mean of that, before spread and commission, and is ``None``
  (unmeasured, never zero) below ``min_trades`` trades: a handful of lucky
  trades is not evidence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from core.ai.nn_features import (
    FEATURE_NAMES, LONG, N_CLASSES, NOISE, SHORT, FeatureScaler, build_features,
    make_labels, make_windows, walk_forward_splits,
)


@dataclass
class TrainConfig:
    window: int = 32
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.1
    epochs: int = 30
    batch: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    fp_penalty: float = 2.0
    barrier_mult: float = 1.5
    horizon: int = 24
    val_fraction: float = 0.2
    patience: int = 5
    seed: int = 7
    threshold: float = 0.72
    n_models: int = 1                 # seeds averaged; more = lower variance


class SeqClassifier(nn.Module):
    """GRU over the window; the last hidden state feeds a 3-way head."""

    def __init__(self, n_features: int, hidden: int = 32, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def balanced_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights. A class absent from ``y`` gets weight 1."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=N_CLASSES).astype(float)
    n = counts.sum()
    w = np.where(counts > 0, n / (N_CLASSES * np.maximum(counts, 1.0)), 1.0)
    return torch.tensor(w, dtype=torch.float32)


def fp_penalised_loss(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor,
                      fp_penalty: float) -> torch.Tensor:
    """Weighted cross-entropy + expected false-positive trade probability."""
    ce = F.cross_entropy(logits, y, weight=class_weights)
    if fp_penalty <= 0:
        return ce
    p = F.softmax(logits, dim=1)
    fp = p[:, LONG] * (y != LONG).float() + p[:, SHORT] * (y != SHORT).float()
    return ce + fp_penalty * fp.mean()


def _batches(n: int, batch: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, batch):
        yield order[i:i + batch]


def fit_sequence_model(x_train: np.ndarray, y_train: np.ndarray,
                       x_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig
                       ) -> tuple[SeqClassifier, dict]:
    """Train with early stopping on validation loss; returns the best weights."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = SeqClassifier(x_train.shape[2], cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = balanced_class_weights(y_train)

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    best_loss, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        for idx in _batches(len(xt), cfg.batch, rng):
            opt.zero_grad()
            loss = fp_penalised_loss(model(xt[idx]), yt[idx], weights, cfg.fp_penalty)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(fp_penalised_loss(model(xv), yv, weights, cfg.fp_penalty)) \
                if len(xv) else math.nan
        history.append(val_loss)
        if len(xv) == 0:
            best_state = copy.deepcopy(model.state_dict())
            continue
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"val_loss": history, "best_val_loss": best_loss, "epochs_run": len(history)}


def logits_for(model: SeqClassifier, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: the single scalar T minimising validation NLL.

    T > 1 softens an over-confident model. Fitted on validation data only.
    """
    if len(logits) < 10:
        return 1.0
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / torch.exp(log_t), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(torch.exp(log_t).item(), 0.05, 10.0))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_ensemble(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, cfg: TrainConfig) -> list:
    """Train ``cfg.n_models`` models that differ only by seed, each calibrated
    on the validation rows. Averaging them lowers the variance of one unlucky
    initialisation; it does not create information the data does not hold."""
    members = []
    for i in range(max(1, cfg.n_models)):
        model, info = fit_sequence_model(x_train, y_train, x_val, y_val,
                                         replace(cfg, seed=cfg.seed + 101 * i))
        temp = fit_temperature(logits_for(model, x_val), y_val)
        members.append({"model": model, "temperature": temp, "info": info})
    return members


def ensemble_proba(members: list, x: np.ndarray) -> np.ndarray:
    """Mean of the members' calibrated class probabilities (rows sum to 1)."""
    return np.mean([softmax_np(logits_for(m["model"], x), m["temperature"])
                    for m in members], axis=0)


MIN_TRADES_FOR_EDGE = 30


def trade_metrics(proba: np.ndarray, y: np.ndarray, threshold: float,
                  min_trades: int = MIN_TRADES_FOR_EDGE) -> dict:
    """Score the trades the model would take at ``threshold``.

    A trade is taken when the most likely class is LONG or SHORT and its
    probability is at least ``threshold``. ``precision`` is wins / trades.
    ``edge`` is the mean R-multiple (wins minus losses, over trades) and is
    ``None`` below ``min_trades``: an unmeasured edge is not zero.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    call = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    taken = (call != NOISE) & (conf >= threshold)
    n_trades = int(taken.sum())
    if n_trades == 0:
        return {"trades": 0, "coverage": 0.0, "precision": None, "edge": None}
    wins = int((call[taken] == y[taken]).sum())
    losses = int(((y[taken] != NOISE) & (call[taken] != y[taken])).sum())
    edge = (wins - losses) / n_trades if n_trades >= min_trades else None
    return {"trades": n_trades, "coverage": n_trades / max(len(y), 1),
            "precision": wins / n_trades, "edge": edge}


def _scaled_windows(feats: np.ndarray, labels: np.ndarray, scaler: FeatureScaler, window: int):
    return make_windows(scaler.transform(feats), labels, window)


def walk_forward_evaluate(df: pd.DataFrame, cfg: TrainConfig, train_bars: int,
                          test_bars: int, step: Optional[int] = None) -> dict:
    """Rolling-origin out-of-sample test.

    Per fold: the scaler and the model are fitted on the training rows only, the
    temperature on the last ``val_fraction`` of those, and the metrics are taken
    on the test rows, which start ``horizon`` bars after training ends. Results
    from every fold are pooled before the metrics are computed.
    """
    feats = build_features(df).to_numpy()
    labels = make_labels(df, cfg.barrier_mult, cfg.horizon)
    folds, pooled_p, pooled_y = [], [], []

    for tr, te in walk_forward_splits(len(df), train_bars, test_bars, cfg.horizon, step):
        scaler = FeatureScaler().fit(feats[tr])
        x, y, ends = _scaled_windows(feats, labels, scaler, cfg.window)
        tr_mask = (ends >= tr.start) & (ends < tr.stop)
        te_mask = (ends >= te.start) & (ends < te.stop)
        x_tr, y_tr, e_tr = x[tr_mask], y[tr_mask], ends[tr_mask]
        if len(x_tr) < 50 or te_mask.sum() == 0:
            continue
        # Validation = the newest windows, separated from training by the horizon.
        cut = int(len(x_tr) * (1.0 - cfg.val_fraction))
        keep = e_tr < e_tr[cut] - cfg.horizon
        members = fit_ensemble(x_tr[keep], y_tr[keep], x_tr[cut:], y_tr[cut:], cfg)
        proba = ensemble_proba(members, x[te_mask])
        pooled_p.append(proba)
        pooled_y.append(y[te_mask])
        folds.append({"train": (tr.start, tr.stop), "test": (te.start, te.stop),
                      "temperatures": [m["temperature"] for m in members],
                      **trade_metrics(proba, y[te_mask], cfg.threshold)})

    if not folds:
        return {"folds": [], "trades": 0, "coverage": 0.0, "precision": None, "edge": None,
                "windows": 0}
    p, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    return {"folds": folds, "windows": int(len(yy)), **trade_metrics(p, yy, cfg.threshold)}


def fit_artifact(df: pd.DataFrame, cfg: TrainConfig) -> dict:
    """Fit the model that would go live on ``df``, with no out-of-sample test.

    ``oos`` is left empty on purpose: the caller must measure it on bars the
    model never saw (``evaluate_artifact``). ``trained_through`` records the
    last bar used, so later evaluation can start strictly after it.
    """
    feats = build_features(df).to_numpy()
    labels = make_labels(df, cfg.barrier_mult, cfg.horizon)
    finite = np.isfinite(feats).all(axis=1) & (labels >= 0)
    rows = np.flatnonzero(finite)
    if len(rows) < 2:
        raise ValueError("not enough usable rows to train")
    scaler = FeatureScaler().fit(feats[rows])
    x, y, ends = _scaled_windows(feats, labels, scaler, cfg.window)
    if len(x) < 100:
        raise ValueError(f"not enough training windows ({len(x)})")
    cut = int(len(x) * (1.0 - cfg.val_fraction))
    keep = ends < ends[cut] - cfg.horizon
    members = fit_ensemble(x[keep], y[keep], x[cut:], y[cut:], cfg)
    return {
        "members": [{"state_dict": m["model"].state_dict(), "temperature": m["temperature"]}
                    for m in members],
        "scaler": scaler.to_dict(),
        "config": asdict(cfg),
        "feature_names": list(FEATURE_NAMES),
        "oos": {},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_through": str(df.index[-1]),
        "n_bars": int(len(df)),
    }


def train_final(df: pd.DataFrame, cfg: TrainConfig, train_bars: int, test_bars: int) -> dict:
    """Measure out-of-sample edge by walk-forward, then fit the model for use.

    The returned artifact carries the measured walk-forward result in ``oos``.
    The predictor refuses a model whose edge is unmeasured or below its floor.
    """
    oos = walk_forward_evaluate(df, cfg, train_bars, test_bars)
    art = fit_artifact(df, cfg)
    art["oos"] = {k: v for k, v in oos.items() if k != "folds"} | {"n_folds": len(oos["folds"])}
    return art


def members_from_artifact(art: dict) -> list:
    """Rebuild the ensemble members (models in eval mode + temperatures).

    Reads the legacy single-model layout (``state_dict`` / ``temperature``) too.
    """
    cfg = art["config"]
    raw = art.get("members") or [{"state_dict": art["state_dict"],
                                  "temperature": art["temperature"]}]
    members = []
    for m in raw:
        model = SeqClassifier(len(art["feature_names"]), cfg["hidden"], cfg["layers"],
                              cfg["dropout"])
        model.load_state_dict(m["state_dict"])
        model.eval()
        members.append({"model": model, "temperature": float(m["temperature"])})
    return members


def evaluate_artifact(art: dict, df: pd.DataFrame, start_row: int,
                      threshold: Optional[float] = None,
                      min_trades: int = MIN_TRADES_FOR_EDGE) -> dict:
    """Score a saved model on the windows of ``df`` that END at or after
    ``start_row``. Rows before it are used only as history for the features.

    This is how a champion and a challenger are compared fairly: the same
    function, the same bars, bars that neither model was trained on.
    """
    cfg = art["config"]
    feats = build_features(df).to_numpy()
    labels = make_labels(df, cfg["barrier_mult"], cfg["horizon"])
    scaler = FeatureScaler.from_dict(art["scaler"])
    x, y, ends = make_windows(scaler.transform(feats), labels, cfg["window"])
    mask = ends >= start_row
    n = int(mask.sum())
    if n == 0:
        return {"windows": 0, "trades": 0, "coverage": 0.0, "precision": None, "edge": None}
    proba = ensemble_proba(members_from_artifact(art), x[mask])
    return {"windows": n, **trade_metrics(proba, y[mask],
                                          threshold if threshold is not None else cfg["threshold"],
                                          min_trades)}


def save_artifact(path: str, artifact: dict) -> None:
    torch.save(artifact, path)


def load_artifact(path: str) -> dict:
    """Load with ``weights_only=True``: the file may hold only tensors and
    plain Python types, so a tampered file cannot execute code."""
    return torch.load(path, map_location="cpu", weights_only=True)
