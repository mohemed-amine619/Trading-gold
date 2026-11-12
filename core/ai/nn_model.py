"""
core/ai/nn_model.py — GRU sequence classifier, training, calibration, honest
trade scoring and walk-forward validation. Imports torch at module level, so
import it lazily (``core/ai/nn_predictor.py`` does).

What it predicts: for each bar, a calibrated probability over
[NOISE, LONG, SHORT] from the last ``window`` candles, where the classes are
the triple-barrier outcomes from ``core/ai/nn_features.py``.

Design points worth knowing (each one fixes a measured failure, see
docs/PROJECT_STATUS.md "NN trainer: why no model was ever promoted"):

* Plain cross-entropy by default. The loss used to add inverse-frequency class
  weights and a false-positive penalty. On gold M5 NOISE is ~3-4 % of labels,
  so its weight was ~8-12 and the loss optimum of an uninformed model moved to
  ~[0.58, 0.21, 0.21]. Caution belongs in the decision threshold, not in the
  loss; both knobs are still reachable (``class_weighting``, ``fp_penalty``)
  to reproduce old runs.
* Ensemble-level calibration with a per-class bias. One scalar temperature
  per member cannot remove a class bias; it saturated at its 10.0 clip and
  flattened every probability to ~1/3, so no bar ever reached a trade
  threshold. Now the members' raw probabilities are averaged and the map
  ``softmax(a * log p + b)`` (one shared scale ``a``, one bias ``b`` per class;
  ``calibration="vector"`` gives one scale per class) is fitted by NLL on the
  validation rows only (``fit_calibration_map``). There is no clip, and the map
  is stored in the artifact as ``art["calibration"]``. Artifacts without it
  (old files) keep the old per-member temperatures and behave exactly as
  before.
* Honest trade counting. A label looks ``horizon`` bars ahead, so consecutive
  signals are the SAME position, not new evidence. ``trade_metrics`` takes
  trades one position at a time in time order: after a trade on the window
  ending at bar ``e`` it ignores signals until bar ``e + horizon``. ``signals``
  is the raw bar count, ``trades`` the non-overlapping count. ``edge`` is the
  mean R (+1 when the called barrier was hit first, -1 when the opposite one
  was, 0 when neither: the trade times out flat), before spread and
  commission, and ``edge_lcb`` its one-sided lower confidence bound
  (mean - z*sd/sqrt(n)). Both are ``None`` (unmeasured, never zero) below
  ``min_trades`` non-overlapping trades.
* Naive baselines. ``edge_report`` also scores always-LONG and always-SHORT on
  the same rows with the same counting; ``baseline_edge`` is the better of the
  two, and ``excess_lcb`` is the lower bound of the model's per-trade R minus
  the naive edge of the direction it chose (timing skill beyond a direction
  bias). A model that only rides a trend is no better than the baseline.
* Threshold chosen on validation. ``TrainConfig.threshold`` is the MINIMUM
  allowed decision threshold. ``select_threshold`` scans
  ``[max(threshold, 0.40), threshold_max]`` on the validation rows and keeps
  the threshold with the highest ``edge_lcb`` among those with at least
  ``min_trades`` non-overlapping trades and an edge above the validation
  baseline; with none it falls back to the minimum. The holdout is never used
  to choose it. It is stored as ``art["threshold"]``; the evaluation and the
  predictor use it.
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

from core.ai.deep_layout import (
    ARCH_DUAL, ARCH_GRU, build_model_features, feature_names, make_excursion,
)
from core.ai.nn_features import (  # noqa: F401  (FEATURE_NAMES/build_features re-exported)
    FEATURE_NAMES, LONG, MIN_TRADES_FOR_EDGE, N_CLASSES, NOISE, SHORT, FeatureScaler,
    build_features, make_labels, make_windows, walk_forward_splits, window_end_indices,
)

LCB_Z = 1.645                     # one-sided 95 % lower confidence bound
THRESHOLD_SEARCH_FLOOR = 0.40     # never search below this, whatever the configured minimum
CAL_KIND_VECTOR = "vector"               # stored map: softmax(scale * log p + bias), per class
# TrainConfig.calibration values
CAL_BIAS = "bias"                        # default: shared temperature + per-class bias
CAL_VECTOR = "vector"                    # per-class scale + per-class bias
CAL_KIND_TEMPERATURE = "temperature"     # legacy: one scalar per member, no ensemble map
_MIN_CAL_ROWS = 30


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
    # Loss. Plain cross-entropy by default; the old behaviour was
    # class_weighting=True, fp_penalty=2.0 (see the module docstring).
    fp_penalty: float = 0.0
    class_weighting: bool = False
    barrier_mult: float = 1.5
    horizon: int = 24
    val_fraction: float = 0.2
    patience: int = 5
    seed: int = 7
    # MINIMUM decision threshold. The trainer searches [max(threshold, 0.40),
    # threshold_max] on validation and stores the chosen one in the artifact.
    threshold: float = 0.72
    threshold_max: float = 0.90
    threshold_step: float = 0.01
    threshold_search: bool = True     # False: use ``threshold`` as is (old behaviour)
    # Ensemble calibration: "bias" (temperature + per-class bias, default),
    # "vector" (per-class scale + bias) or "temperature" (legacy per-member scalar).
    calibration: str = CAL_BIAS
    lcb_z: float = LCB_Z
    n_models: int = 1                 # seeds averaged; more = lower variance
    # Architecture. "gru" is the original model and the default. "dual" is the
    # dual-stream net in core/ai/deep_nn.py (temporal + market-location).
    arch: str = "gru"
    preset: str = "small"             # dual only: small | medium | large
    struct_stride: int = 8            # dual only: location read every N bars
    aux_weight: float = 0.2           # dual only: weight of the excursion head loss


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


def class_weights_for(y: np.ndarray, cfg) -> torch.Tensor:
    """The loss weights ``cfg`` asks for: balanced when ``class_weighting`` is on
    (the old default), otherwise all ones (plain cross-entropy)."""
    if getattr(cfg, "class_weighting", False):
        return balanced_class_weights(y)
    return torch.ones(N_CLASSES, dtype=torch.float32)


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
                       x_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig,
                       aux_train: Optional[np.ndarray] = None,
                       aux_val: Optional[np.ndarray] = None,
                       ) -> tuple[nn.Module, dict]:
    """Train with early stopping on validation loss; returns the best weights.

    ``arch="dual"`` hands over to ``core.ai.deep_nn.fit_dual`` (the optional
    ``aux_*`` arrays are its excursion targets); the GRU path ignores them.
    """
    if cfg.arch == "dual":
        from core.ai.deep_nn import fit_dual
        return fit_dual(x_train, y_train, x_val, y_val, cfg, aux_train, aux_val)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = SeqClassifier(x_train.shape[2], cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = class_weights_for(y_train, cfg)

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
    """Legacy temperature scaling: the single scalar T minimising validation NLL,
    clipped to [0.05, 10].

    Kept for ``calibration="temperature"`` and for old artifacts. It cannot
    remove a class BIAS (it rescales every logit equally), which is why it
    saturated at 10 on the class-weighted model; see ``fit_calibration_map``.
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


# ── calibration of the ensemble ──────────────────────────────────────────────

def identity_calibration() -> dict:
    return {"kind": CAL_KIND_VECTOR, "fit": "identity", "scale": [1.0] * N_CLASSES,
            "bias": [0.0] * N_CLASSES, "fitted": False, "rows": 0}


def fit_calibration_map(proba: np.ndarray, y: np.ndarray, per_class_scale: bool = False,
                        l2: float = 1e-4) -> dict:
    """Calibrate averaged class probabilities: ``softmax(a * log(p) + b)``,
    fitted by NLL on VALIDATION rows only.

    ``b`` is one bias per class: it removes a class bias, which one scalar
    temperature cannot do. ``a`` is one shared scale (temperature + per-class
    bias, the default) or, with ``per_class_scale``, one per class (vector
    scaling). Measured on synthetic gold-like M5 data the shared scale
    generalises better: per-class scales lower the validation NLL but raise the
    holdout NLL. ``a = exp(u)`` is positive by construction (the map never
    inverts a class ranking) and there is no clip to get stuck on; a tiny L2 on
    ``u`` and ``b`` only keeps the solution unique when the model is
    uninformative, where the map then returns the validation base rates, not
    1/3 each.

    Too few rows (or a single class) returns the identity map, flagged
    ``fitted: False``.
    """
    fit_kind = "vector" if per_class_scale else "bias"
    p = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(p) < _MIN_CAL_ROWS or len(np.unique(y)) < 2:
        return identity_calibration()
    lp = torch.tensor(np.log(np.clip(p, 1e-12, 1.0)), dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    u = torch.zeros(N_CLASSES if per_class_scale else 1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(N_CLASSES, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([u, b], lr=1.0, max_iter=500, tolerance_grad=1e-10,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lp * torch.exp(u) + b, yt) + l2 * ((u ** 2).sum() + (b ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    a = np.exp(u.detach().numpy()) * np.ones(N_CLASSES)
    bias = b.detach().numpy()
    bias = bias - bias.mean()                         # softmax is shift-invariant
    if not (np.isfinite(a).all() and np.isfinite(bias).all()):
        return identity_calibration()
    return {"kind": CAL_KIND_VECTOR, "fit": fit_kind, "scale": [float(v) for v in a],
            "bias": [float(v) for v in bias], "fitted": True, "rows": int(len(y))}


def apply_calibration(proba: np.ndarray, calibration: Optional[dict]) -> np.ndarray:
    """Apply a stored calibration map. ``None`` (old artifacts) is the identity.
    An unknown kind raises: a file this code cannot interpret must not trade."""
    p = np.asarray(proba, dtype=float)
    if not calibration:
        return p
    kind = calibration.get("kind")
    if kind != CAL_KIND_VECTOR:
        raise ValueError(f"unknown calibration kind {kind!r}")
    a = np.asarray(calibration["scale"], dtype=float)
    b = np.asarray(calibration["bias"], dtype=float)
    if a.shape != (N_CLASSES,) or b.shape != (N_CLASSES,):
        raise ValueError("calibration needs one scale and one bias per class")
    return softmax_np(np.log(np.clip(p, 1e-12, 1.0)) * a + b)


def fit_ensemble(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, cfg: TrainConfig,
                 aux_train: Optional[np.ndarray] = None,
                 aux_val: Optional[np.ndarray] = None) -> list:
    """Train ``cfg.n_models`` models that differ only by seed. Averaging them
    lowers the variance of one unlucky initialisation; it does not create
    information the data does not hold.

    By default each member keeps temperature 1.0: the ensemble AVERAGE is
    calibrated afterwards (``calibrate_ensemble``). With the legacy
    ``calibration="temperature"`` each member gets its own scalar temperature.
    """
    legacy = getattr(cfg, "calibration", CAL_BIAS) == CAL_KIND_TEMPERATURE
    members = []
    for i in range(max(1, cfg.n_models)):
        model, info = fit_sequence_model(x_train, y_train, x_val, y_val,
                                         replace(cfg, seed=cfg.seed + 101 * i),
                                         aux_train, aux_val)
        temp = fit_temperature(logits_for(model, x_val), y_val) if legacy else 1.0
        members.append({"model": model, "temperature": temp, "info": info})
    return members


def calibrate_ensemble(members: list, x_val: np.ndarray, y_val: np.ndarray,
                       cfg: TrainConfig) -> Optional[dict]:
    """The ensemble-level calibration map for ``cfg`` (``None`` for the legacy
    per-member temperatures). Fitted on the validation rows only."""
    kind = getattr(cfg, "calibration", CAL_BIAS)
    if kind == CAL_KIND_TEMPERATURE:
        return None
    if kind not in (CAL_BIAS, CAL_VECTOR):
        raise ValueError(f"unknown calibration {kind!r}")
    if len(x_val) == 0:
        return identity_calibration()
    return fit_calibration_map(ensemble_proba(members, x_val), y_val,
                               per_class_scale=kind == CAL_VECTOR)


def ensemble_proba(members: list, x: np.ndarray,
                   calibration: Optional[dict] = None) -> np.ndarray:
    """Mean of the members' class probabilities (each at its own temperature),
    then the ensemble calibration map when one is given. Rows sum to 1."""
    p = np.mean([softmax_np(logits_for(m["model"], x), m["temperature"])
                 for m in members], axis=0)
    return apply_calibration(p, calibration)


# ── honest trade scoring ─────────────────────────────────────────────────────

def _trades(proba: np.ndarray, y: np.ndarray, threshold, ends: Optional[np.ndarray],
            horizon: int):
    """The positions taken at ``threshold``, one at a time in time order.

    Returns ``(r, direction, n_signals)``: the R-multiple and the class of each
    non-overlapping trade, and the raw number of bars that signalled.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int), 0
    e = np.arange(n) if ends is None else np.asarray(ends, dtype=np.int64)
    call = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    taken = (call != NOISE) & (conf >= np.asarray(threshold, dtype=float))
    n_signals = int(taken.sum())
    order = np.argsort(e, kind="stable")
    idx = order[taken[order]]
    picked = []
    next_free = -np.inf
    h = max(int(horizon), 0)
    for i in idx:
        if e[i] >= next_free:
            picked.append(i)
            next_free = e[i] + h
    picked = np.asarray(picked, dtype=int)
    c, t = call[picked], y[picked]
    r = np.where(c == t, 1.0, np.where(t == NOISE, 0.0, -1.0))
    return r, c, n_signals


def _summary(r: np.ndarray, min_trades: int, z: float) -> tuple:
    """``(mean, lower bound)`` of ``r``; ``None`` below ``min_trades`` (and the
    bound also below 2 values, where a spread does not exist)."""
    n = len(r)
    if n == 0 or n < min_trades:
        return None, None
    mean = float(r.mean())
    if n < 2:
        return mean, None
    sd = float(r.std(ddof=1))
    return mean, float(mean - z * sd / math.sqrt(n))


def trade_metrics(proba: np.ndarray, y: np.ndarray, threshold,
                  min_trades: int = MIN_TRADES_FOR_EDGE, ends: Optional[np.ndarray] = None,
                  horizon: int = 0, z: float = LCB_Z) -> dict:
    """Score the trades the model would take at ``threshold``.

    A signal is a bar whose most likely class is LONG or SHORT with probability
    at least ``threshold`` (a scalar, or one value per row). Trades are taken
    ONE POSITION AT A TIME in time order: after a trade on the window ending at
    bar ``e`` (``ends``), signals are ignored until bar ``e + horizon``, the
    bar the label stops looking at. Pass the window ends and the label horizon;
    without them every signal is its own trade (the old count, which inflates
    the evidence because a label spans ``horizon`` bars).

    Returns ``signals`` (raw bar count), ``trades`` (non-overlapping), ``wins``,
    ``losses``, ``precision`` (wins / trades), ``edge`` (mean R: +1 win, -1
    loss, 0 timeout) and ``edge_lcb`` (one-sided lower bound, mean -
    z*sd/sqrt(n)). ``edge`` and ``edge_lcb`` are ``None`` below ``min_trades``:
    an unmeasured edge is not zero.
    """
    r, _, n_signals = _trades(proba, y, threshold, ends, horizon)
    n = len(r)
    edge, lcb = _summary(r, min_trades, z)
    return {"signals": n_signals, "trades": int(n),
            "coverage": n_signals / max(len(np.asarray(y)), 1),
            "wins": int((r > 0).sum()), "losses": int((r < 0).sum()),
            "precision": float((r > 0).sum() / n) if n else None,
            "edge": edge, "edge_lcb": lcb}


def baseline_metrics(y: np.ndarray, ends: Optional[np.ndarray], horizon: int,
                     min_trades: int = MIN_TRADES_FOR_EDGE, z: float = LCB_Z) -> dict:
    """Edges of the naive always-LONG and always-SHORT rules on the same rows,
    counted exactly like the model's trades. ``baseline_edge`` is the better of
    the two (``None`` when unmeasured)."""
    y = np.asarray(y, dtype=int)
    out = {}
    for name, cls in (("long", LONG), ("short", SHORT)):
        proba = np.zeros((len(y), N_CLASSES))
        proba[:, cls] = 1.0
        r, _, _ = _trades(proba, y, 0.0, ends, horizon)
        out[f"baseline_{name}_edge"] = _summary(r, min_trades, z)[0]
        out["baseline_trades"] = int(len(r))
    lo, sh = out["baseline_long_edge"], out["baseline_short_edge"]
    if lo is None or sh is None:
        out["baseline_edge"], out["baseline_direction"] = None, None
    else:
        out["baseline_edge"] = max(lo, sh)
        out["baseline_direction"] = "long" if lo >= sh else "short"
    return out


def edge_report(proba: np.ndarray, y: np.ndarray, threshold, ends: Optional[np.ndarray],
                horizon: int, min_trades: int = MIN_TRADES_FOR_EDGE,
                z: float = LCB_Z) -> dict:
    """``trade_metrics`` + the naive baselines + the timing skill beyond them.

    ``excess`` is the mean over the model's trades of (R minus the naive edge
    of the direction that trade took) and ``excess_lcb`` its lower bound: a
    model that only rides the period's drift has ``excess`` ~ 0. All ``None``
    when either side is unmeasured.
    """
    m = trade_metrics(proba, y, threshold, min_trades, ends, horizon, z)
    base = baseline_metrics(y, ends, horizon, min_trades, z)
    excess = excess_lcb = None
    if m["edge"] is not None and base["baseline_edge"] is not None:
        r, c, _ = _trades(proba, y, threshold, ends, horizon)
        naive = np.where(c == LONG, base["baseline_long_edge"], base["baseline_short_edge"])
        excess, excess_lcb = _summary(r - naive, min_trades, z)
    return {**m, **base, "excess": excess, "excess_lcb": excess_lcb}


def threshold_grid(cfg) -> np.ndarray:
    lo = max(float(cfg.threshold), THRESHOLD_SEARCH_FLOOR)
    hi = max(lo, float(getattr(cfg, "threshold_max", 0.90)))
    step = max(float(getattr(cfg, "threshold_step", 0.01)), 1e-3)
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def select_threshold(proba: np.ndarray, y: np.ndarray, ends: np.ndarray, cfg,
                     min_trades: int = MIN_TRADES_FOR_EDGE) -> dict:
    """Choose the decision threshold on VALIDATION rows.

    Among the grid ``[max(cfg.threshold, 0.40), cfg.threshold_max]``, keep the
    thresholds with at least ``min_trades`` non-overlapping trades and an edge
    above the naive baseline on the same rows, and take the one with the
    highest ``edge_lcb`` (ties: the lowest threshold). With none, or with
    ``threshold_search`` off, it falls back to the minimum; the holdout then
    judges the model there (typically unmeasured, so refused).
    """
    grid = threshold_grid(cfg)
    fallback = float(grid[0])
    if not getattr(cfg, "threshold_search", True):
        return {"threshold": float(cfg.threshold), "source": "config", "qualifying": 0,
                "validation": None}
    z = float(getattr(cfg, "lcb_z", LCB_Z))
    y = np.asarray(y, dtype=int)
    if len(y) == 0:
        return {"threshold": fallback, "source": "fallback: no validation rows",
                "qualifying": 0, "validation": None}
    base = baseline_metrics(y, ends, cfg.horizon, min_trades, z)
    best, qualifying = None, 0
    for t in grid:
        m = trade_metrics(proba, y, float(t), min_trades, ends, cfg.horizon, z)
        if m["edge_lcb"] is None:
            continue
        if base["baseline_edge"] is not None and m["edge"] <= base["baseline_edge"]:
            continue
        qualifying += 1
        if best is None or m["edge_lcb"] > best[1]["edge_lcb"]:
            best = (float(t), m)
    if best is None:
        return {"threshold": fallback, "qualifying": 0,
                "source": f"fallback: no threshold in [{grid[0]:.2f}, {grid[-1]:.2f}] had "
                          f"{min_trades} validation trades beating the naive baseline",
                "validation": {**trade_metrics(proba, y, fallback, min_trades, ends,
                                               cfg.horizon, z), **base}}
    return {"threshold": best[0], "source": "validation", "qualifying": qualifying,
            "validation": {**best[1], **base}}


# ── walk-forward and the saved model ─────────────────────────────────────────

def _scaler_for_training(feats: np.ndarray, train_ends: np.ndarray, window: int) -> FeatureScaler:
    """A scaler fitted on the feature rows that feed the TRAINING windows only
    (not the validation rows that pick the threshold and the calibration)."""
    lo = max(int(train_ends.min()) - window + 1, 0)
    return FeatureScaler().fit(feats[lo:int(train_ends.max()) + 1])


def _train_val_split(ends: np.ndarray, cfg) -> tuple:
    """``(keep_mask, cut)``: validation is the newest ``val_fraction`` of the
    windows; training keeps windows ending more than ``horizon`` bars before it."""
    cut = int(len(ends) * (1.0 - cfg.val_fraction))
    cut = min(max(cut, 0), len(ends))
    if cut >= len(ends):
        return np.ones(len(ends), dtype=bool), len(ends)
    return ends < ends[cut] - cfg.horizon, cut


def _fit_calibrate_select(x_tr, y_tr, x_val, y_val, e_val, cfg, a_tr=None, a_val=None):
    members = fit_ensemble(x_tr, y_tr, x_val, y_val, cfg, a_tr, a_val)
    calibration = calibrate_ensemble(members, x_val, y_val, cfg)
    p_val = ensemble_proba(members, x_val, calibration) if len(x_val) else np.zeros((0, N_CLASSES))
    selection = select_threshold(p_val, y_val, e_val, cfg)
    return members, calibration, selection


def walk_forward_evaluate(df: pd.DataFrame, cfg: TrainConfig, train_bars: int,
                          test_bars: int, step: Optional[int] = None) -> dict:
    """Rolling-origin out-of-sample test.

    Per fold: the scaler is fitted on the rows feeding the training windows, the
    model on those windows, the calibration and the threshold on the newest
    ``val_fraction`` of the fold (after a ``horizon`` purge), and the metrics
    are taken on the test rows, which start ``horizon`` bars after training
    ends. Folds are pooled (each fold at its own threshold) and the pooled
    trades are counted one position at a time, like everywhere else.
    """
    feats = build_model_features(df, cfg.arch, cfg.struct_stride)
    labels = make_labels(df, cfg.barrier_mult, cfg.horizon)
    exc = make_excursion(df, cfg.horizon) if cfg.arch == ARCH_DUAL else None
    all_ends = window_end_indices(feats, labels, cfg.window)
    z = float(cfg.lcb_z)
    folds, pooled_p, pooled_y, pooled_e, pooled_t = [], [], [], [], []

    for tr, te in walk_forward_splits(len(df), train_bars, test_bars, cfg.horizon, step):
        tr_ends = all_ends[(all_ends >= tr.start) & (all_ends < tr.stop)]
        if len(tr_ends) < 50 or not ((all_ends >= te.start) & (all_ends < te.stop)).any():
            continue
        keep_pre, cut_pre = _train_val_split(tr_ends, cfg)
        if not keep_pre.any():
            continue
        scaler = _scaler_for_training(feats, tr_ends[keep_pre], cfg.window)
        x, y, ends = make_windows(scaler.transform(feats), labels, cfg.window)
        tr_mask = (ends >= tr.start) & (ends < tr.stop)
        te_mask = (ends >= te.start) & (ends < te.stop)
        x_tr, y_tr, e_tr = x[tr_mask], y[tr_mask], ends[tr_mask]
        keep, cut = _train_val_split(e_tr, cfg)
        a_tr = exc[e_tr] if exc is not None else None
        members, calibration, sel = _fit_calibrate_select(
            x_tr[keep], y_tr[keep], x_tr[cut:], y_tr[cut:], e_tr[cut:], cfg,
            a_tr[keep] if a_tr is not None else None,
            a_tr[cut:] if a_tr is not None else None)
        proba = ensemble_proba(members, x[te_mask], calibration)
        thr = sel["threshold"]
        pooled_p.append(proba)
        pooled_y.append(y[te_mask])
        pooled_e.append(ends[te_mask])
        pooled_t.append(np.full(int(te_mask.sum()), thr))
        folds.append({"train": (tr.start, tr.stop), "test": (te.start, te.stop),
                      "temperatures": [m["temperature"] for m in members],
                      "calibration": calibration, "threshold": thr,
                      "threshold_source": sel["source"],
                      **edge_report(proba, y[te_mask], thr, ends[te_mask], cfg.horizon,
                                    MIN_TRADES_FOR_EDGE, z)})

    if not folds:
        return {"folds": [], "signals": 0, "trades": 0, "coverage": 0.0, "precision": None,
                "edge": None, "edge_lcb": None, "baseline_edge": None, "excess_lcb": None,
                "windows": 0, "thresholds": []}
    p, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    ee, tt = np.concatenate(pooled_e), np.concatenate(pooled_t)
    return {"folds": folds, "windows": int(len(yy)),
            "thresholds": [f["threshold"] for f in folds],
            **edge_report(p, yy, tt, ee, cfg.horizon, MIN_TRADES_FOR_EDGE, z)}


def fit_artifact(df: pd.DataFrame, cfg: TrainConfig) -> dict:
    """Fit the model that would go live on ``df``, with no out-of-sample test.

    The scaler and the network see the training windows only; the calibration
    map and the decision threshold are fitted on the newest ``val_fraction`` of
    windows (after a ``horizon`` purge). ``oos`` is left empty on purpose: the
    caller must measure it on bars the model never saw (``evaluate_artifact``).
    ``trained_through`` records the last bar used, so later evaluation can start
    strictly after it.
    """
    feats = build_model_features(df, cfg.arch, cfg.struct_stride)
    labels = make_labels(df, cfg.barrier_mult, cfg.horizon)
    finite = np.isfinite(feats).all(axis=1) & (labels >= 0)
    if int(finite.sum()) < 2:
        raise ValueError("not enough usable rows to train")
    pre_ends = window_end_indices(feats, labels, cfg.window)
    if len(pre_ends) < 100:
        raise ValueError(f"not enough training windows ({len(pre_ends)})")
    keep_pre, _ = _train_val_split(pre_ends, cfg)
    if not keep_pre.any():
        raise ValueError("no training windows left before the validation purge")
    scaler = _scaler_for_training(feats, pre_ends[keep_pre], cfg.window)
    x, y, ends = make_windows(scaler.transform(feats), labels, cfg.window)
    keep, cut = _train_val_split(ends, cfg)
    a = make_excursion(df, cfg.horizon)[ends] if cfg.arch == ARCH_DUAL else None
    members, calibration, sel = _fit_calibrate_select(
        x[keep], y[keep], x[cut:], y[cut:], ends[cut:], cfg,
        a[keep] if a is not None else None, a[cut:] if a is not None else None)
    counts = np.bincount(y[keep], minlength=N_CLASSES).astype(float)
    freq = counts / max(counts.sum(), 1.0)
    art = {
        "members": [{"state_dict": m["model"].state_dict(), "temperature": m["temperature"]}
                    for m in members],
        "scaler": scaler.to_dict(),
        "config": asdict(cfg),
        "feature_names": feature_names(cfg.arch),
        "threshold": float(sel["threshold"]),
        "threshold_selection": _plain(sel),
        "train_label_freq": [float(v) for v in freq],
        # Majority trade direction of the training labels (information only; the
        # gates compare against the better of always-long / always-short).
        "baseline_direction": "long" if counts[LONG] >= counts[SHORT] else "short",
        "oos": {},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_through": str(df.index[-1]),
        "n_bars": int(len(df)),
    }
    if calibration is not None:
        art["calibration"] = calibration
    return art


def _plain(obj):
    """Numpy scalars -> Python, recursively, so ``weights_only`` loading works."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    return obj


def train_final(df: pd.DataFrame, cfg: TrainConfig, train_bars: int, test_bars: int) -> dict:
    """Measure out-of-sample edge by walk-forward, then fit the model for use.

    The returned artifact carries the measured walk-forward result in ``oos``.
    The predictor refuses a model whose edge is unmeasured, not confidently
    positive, or no better than the naive baseline.
    """
    oos = walk_forward_evaluate(df, cfg, train_bars, test_bars)
    art = fit_artifact(df, cfg)
    art["oos"] = _plain({k: v for k, v in oos.items() if k != "folds"}
                        | {"n_folds": len(oos["folds"]), "source": "walk-forward"})
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
        if cfg.get("arch", ARCH_GRU) == ARCH_DUAL:
            from core.ai.deep_nn import build_dual
            model = build_dual(len(art["feature_names"]), cfg.get("preset", "small"),
                               cfg["dropout"])
        else:
            model = SeqClassifier(len(art["feature_names"]), cfg["hidden"], cfg["layers"],
                                  cfg["dropout"])
        model.load_state_dict(m["state_dict"])
        model.eval()
        members.append({"model": model, "temperature": float(m["temperature"])})
    return members


def artifact_threshold(art: dict) -> float:
    """The threshold the artifact's edge is measured at: the stored one, or the
    config's for artifacts saved before thresholds were stored."""
    stored = art.get("threshold")
    return float(stored) if stored is not None else float(art["config"]["threshold"])


def evaluate_artifact(art: dict, df: pd.DataFrame, start_row: int,
                      threshold: Optional[float] = None,
                      min_trades: int = MIN_TRADES_FOR_EDGE) -> dict:
    """Score a saved model on the windows of ``df`` that END at or after
    ``start_row``. Rows before it are used only as history for the features.

    This is how a champion and a challenger are compared fairly: the same
    function, the same bars, bars that neither model was trained on, each at
    its own stored threshold and calibration. Trades are counted one position
    at a time and compared with the naive baselines (``edge_report``).
    """
    cfg = art["config"]
    feats = build_model_features(df, cfg.get("arch", ARCH_GRU), cfg.get("struct_stride", 8))
    labels = make_labels(df, cfg["barrier_mult"], cfg["horizon"])
    scaler = FeatureScaler.from_dict(art["scaler"])
    x, y, ends = make_windows(scaler.transform(feats), labels, cfg["window"])
    mask = ends >= start_row
    n = int(mask.sum())
    thr = float(threshold) if threshold is not None else artifact_threshold(art)
    if n == 0:
        return {"windows": 0, "threshold": thr, "signals": 0, "trades": 0, "coverage": 0.0,
                "precision": None, "edge": None, "edge_lcb": None, "baseline_edge": None,
                "excess": None, "excess_lcb": None}
    proba = ensemble_proba(members_from_artifact(art), x[mask], art.get("calibration"))
    return {"windows": n, "threshold": thr,
            **edge_report(proba, y[mask], thr, ends[mask], int(cfg["horizon"]), min_trades,
                          float(cfg.get("lcb_z", LCB_Z)))}


def save_artifact(path: str, artifact: dict) -> None:
    torch.save(artifact, path)


def load_artifact(path: str) -> dict:
    """Load with ``weights_only=True``: the file may hold only tensors and
    plain Python types, so a tampered file cannot execute code."""
    return torch.load(path, map_location="cpu", weights_only=True)
