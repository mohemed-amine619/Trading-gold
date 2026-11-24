"""
core/risk/progression.py — position-size progressions and position adding:
martingale, anti-martingale, Kelly, grid and DCA.

ALL OF THESE ARE OFF BY DEFAULT and were added at the owner's explicit request
(2026-11-24), overriding the project's earlier "no martingale, no grid" rule.
They are dangerous in specific, well-known ways, and the guards below exist
because of them:

* Martingale raises the risk of the next trade after every loss. A losing
  streak, which WILL happen, compounds geometrically. Capped by ``max_steps``
  and by an absolute ``max_mult`` on the multiplier.
* Grid / DCA add positions to a trade that is going against you. Every add here
  SHARES THE FIRST POSITION'S STOP-LOSS (nothing is ever opened without a stop,
  and no stop is ever widened), so the total loss if that stop is hit is known
  before the add: ``total_risk_if_stopped`` reports it and ``max_total_risk``
  refuses an add that would exceed it.
* Kelly sizing is only as good as the win rate and payoff it is fed. It uses a
  FRACTION of Kelly, needs ``min_trades`` closed trades first, and is capped.

Pure functions, no state, no broker import. The caller (run.py) supplies the
recent closed-trade P&L (oldest first) and the open positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_EPS = 1e-9

MODES = ("off", "martingale", "anti_martingale", "kelly")
GRID_MODES = ("off", "grid", "dca")


def _run(pnls: Sequence[float], losing: bool) -> int:
    """Length of the unbroken run of losses (or wins) at the end of ``pnls``."""
    n = 0
    for p in reversed(pnls):
        if losing and p < -_EPS:
            n += 1
        elif not losing and p > _EPS:
            n += 1
        else:
            break
    return n


def martingale_multiplier(pnls: Sequence[float], factor: float = 2.0, max_steps: int = 3) -> float:
    """``factor ** losses`` after a run of consecutive losses, at most ``max_steps`` steps.
    A win or a breakeven resets it to 1."""
    if factor < 1.0:
        raise ValueError("martingale factor must be >= 1")
    return float(factor) ** min(_run(pnls, True), max(0, int(max_steps)))


def anti_martingale_multiplier(pnls: Sequence[float], factor: float = 1.5,
                               max_steps: int = 2) -> float:
    """``factor ** wins`` after a run of consecutive wins (press winners), at most
    ``max_steps`` steps. A loss resets it to 1."""
    if factor < 1.0:
        raise ValueError("anti-martingale factor must be >= 1")
    return float(factor) ** min(_run(pnls, False), max(0, int(max_steps)))


def kelly_fraction(pnls: Sequence[float]) -> Optional[float]:
    """Full Kelly fraction ``W - (1 - W) / R`` from closed trades, or None when there is
    no win or no loss to measure R from."""
    wins = [p for p in pnls if p > _EPS]
    losses = [-p for p in pnls if p < -_EPS]
    if not wins or not losses:
        return None
    w = len(wins) / (len(wins) + len(losses))
    r = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    return w - (1.0 - w) / r


def kelly_multiplier(pnls: Sequence[float], risk_per_trade: float, fraction: float = 0.25,
                     min_trades: int = 30, floor: float = 0.25, cap: float = 2.0) -> float:
    """Size multiplier so that the risk per trade becomes ``fraction x Kelly``.

    1.0 until ``min_trades`` closed trades exist. A negative Kelly (no edge in the
    record) gives ``floor``, never zero and never a larger size.
    """
    if len(pnls) < max(1, int(min_trades)) or risk_per_trade <= 0:
        return 1.0
    f = kelly_fraction(pnls)
    if f is None:
        return 1.0
    if f <= 0:
        return float(floor)
    return float(min(cap, max(floor, fraction * f / risk_per_trade)))


def progression_multiplier(mode: str, pnls: Sequence[float], *, factor: float = 2.0,
                           max_steps: int = 3, max_mult: float = 4.0,
                           risk_per_trade: float = 0.01, kelly_fraction_: float = 0.25,
                           kelly_min_trades: int = 30) -> Tuple[float, str]:
    """(multiplier, reason) for the configured mode, clipped to ``max_mult``.
    Unknown modes behave as "off" (fail closed: no size change)."""
    mode = (mode or "off").strip().lower()
    if mode == "martingale":
        m = martingale_multiplier(pnls, factor, max_steps)
        why = f"martingale after {_run(pnls, True)} loss(es)"
    elif mode == "anti_martingale":
        m = anti_martingale_multiplier(pnls, factor, max_steps)
        why = f"anti-martingale after {_run(pnls, False)} win(s)"
    elif mode == "kelly":
        m = kelly_multiplier(pnls, risk_per_trade, kelly_fraction_, kelly_min_trades,
                             cap=max_mult)
        why = f"{kelly_fraction_:.2f} Kelly over {len(pnls)} trades"
    else:
        return 1.0, ""
    return float(min(max(m, 0.0), max(1.0, float(max_mult)))), why


# ── grid / DCA ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AddPlan:
    allowed: bool
    reason: str
    level: int = 0                 # 1 = first add (second position)
    size_mult: float = 1.0
    stop: float = 0.0              # the shared stop (the first position's)
    target: float = 0.0            # the shared take-profit


def grid_add(direction: int, first_entry: float, first_sl: float, first_tp: float,
             price: float, atr: float, n_open: int, *, mode: str = "grid",
             spacing_atr: float = 1.0, max_levels: int = 3, dca_scale: float = 1.5) -> AddPlan:
    """May one more position be added to an open trade?

    * ``grid``: equal size at every ``spacing_atr`` ATR against the first entry.
    * ``dca``: same levels, each add ``dca_scale`` times the previous one.

    Refused when: the mode is off, the first position has no stop, the price is
    already at/through that stop, the level for the next add has not been
    reached, or ``max_levels`` adds exist.
    """
    mode = (mode or "off").strip().lower()
    if mode not in ("grid", "dca"):
        return AddPlan(False, "grid/DCA off")
    if direction not in (1, -1) or n_open < 1:
        return AddPlan(False, "no open trade to add to")
    if not first_sl or first_sl <= 0:
        return AddPlan(False, "first position has no stop-loss: never add to an unprotected trade")
    if (price - first_sl) * direction <= 0:
        return AddPlan(False, "price is at or through the shared stop")
    if atr <= 0 or spacing_atr <= 0:
        return AddPlan(False, "no usable ATR")
    level = n_open                                   # adds already made = n_open - 1
    if level > max(0, int(max_levels)):
        return AddPlan(False, f"grid full ({max_levels} add(s))")
    trigger = first_entry - direction * level * spacing_atr * atr
    if (price - trigger) * direction > 0:
        return AddPlan(False, f"next level {trigger:.5f} not reached")
    size = 1.0 if mode == "grid" else float(dca_scale) ** level
    return AddPlan(True, f"{mode} add #{level} at {price:.5f} (level {trigger:.5f})",
                   level, size, first_sl, first_tp)


def total_risk_if_stopped(direction: int, stop: float,
                          legs: Sequence[Tuple[float, float]], value_per_point_per_lot: float) -> float:
    """Money lost if the shared stop is hit, for legs of ``(entry, volume)``."""
    return sum(max(0.0, (entry - stop) * direction) * vol * value_per_point_per_lot
               for entry, vol in legs)
